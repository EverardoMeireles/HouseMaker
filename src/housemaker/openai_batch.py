# ### Imports ###
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# ### Constants ###
LUNA_MODEL = "gpt-5.6-luna"
LUNA_BATCH_MODEL = "gpt-5.6-luna-batch"
OPENAI_API_URL = "https://api.openai.com/v1"
MAX_BATCH_REQUESTS = 50_000
MAX_INPUT_BYTES = 200_000_000
MAX_OUTPUT_BYTES = 256 * 1024 * 1024
MAX_METADATA_BYTES = 2 * 1024 * 1024
NETWORK_TIMEOUT_SECONDS = 15.0
POLL_INTERVAL_SECONDS = 5.0
CANCELLATION_POLL_SECONDS = 0.05
MAX_WAIT_SECONDS = 24 * 60 * 60 + 10 * 60
TERMINAL_STATUSES = frozenset({"completed", "failed", "expired", "cancelled"})
KNOWN_STATUSES = TERMINAL_STATUSES | {
    "validating",
    "in_progress",
    "finalizing",
    "cancelling",
}
_CUSTOM_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_FILE_ID_PATTERN = re.compile(r"file[-_][A-Za-z0-9_-]{1,200}\Z")
_BATCH_ID_PATTERN = re.compile(r"batch_[A-Za-z0-9_-]{1,200}\Z")
_LOGGER = logging.getLogger(__name__)


# ### Public errors and results ###
class OpenAIBatchError(RuntimeError):
    """A Batch API error safe to display without exposing request images or keys."""


class OpenAIBatchCancelled(OpenAIBatchError):
    pass


@dataclass(frozen=True)
class BatchResult:
    body: dict | None
    error: str | None


# ### Public entry points ###
def to_openai_model(model: str) -> str:
    return LUNA_MODEL if model == LUNA_BATCH_MODEL else model


def request_luna_batch_response(
    body: dict,
    *,
    api_key: str,
    cancellation_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[str], None] | None = None,
    opener: Any = None,
) -> dict:
    result = run_responses_batch(
        {"request-1": body},
        api_key=api_key,
        cancellation_check=cancellation_check,
        progress_callback=progress_callback,
        opener=opener,
    )["request-1"]
    if result.error is not None or result.body is None:
        raise OpenAIBatchError(result.error or "The Luna batch returned no response.")
    return result.body


def run_responses_batch(
    requests: Mapping[str, dict],
    *,
    api_key: str,
    cancellation_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[str], None] | None = None,
    opener: Any = None,
) -> dict[str, BatchResult]:
    """Submit one real 24-hour Responses batch and return custom-id keyed results.

    Calls are synchronous for an existing background worker. Cancellation sends
    one remote cancel request but never waits for its (up to ten-minute) finish.
    Nothing retries billable POST requests or falls back to the realtime API.
    """
    _raise_if_cancelled(cancellation_check)
    if not isinstance(api_key, str) or not api_key.strip():
        raise OpenAIBatchError("An OpenAI API key is required for Luna Batch.")
    api_key = api_key.strip()
    input_jsonl = _prepare_requests(requests)
    transport = _Transport(api_key, opener or urlopen, cancellation_check)
    input_file_id: str | None = None
    batch_id: str | None = None
    owned_file_ids: set[str] = set()
    creation_started = False
    terminal = False
    results_mapped = False
    try:
        _report(
            progress_callback, "Submitting Luna Batch (24-hour completion window)..."
        )
        upload = transport.upload(input_jsonl)
        input_file_id = _validated_id(upload.get("id"), "file")
        owned_file_ids.add(input_file_id)
        _raise_if_cancelled(cancellation_check)
        # A timed-out creation POST may still have succeeded remotely. Do not
        # repeat it or delete the input of a potentially running, unknown batch.
        creation_started = True
        batch = transport.json_request(
            "POST",
            "/batches",
            {
                "input_file_id": input_file_id,
                "endpoint": "/v1/responses",
                "completion_window": "24h",
            },
        )
        batch_id = _validated_id(batch.get("id"), "batch")
        deadline = time.monotonic() + MAX_WAIT_SECONDS
        while True:
            _raise_if_cancelled(cancellation_check)
            if batch.get("id") != batch_id:
                raise OpenAIBatchError(
                    "The Batch API returned a different batch identifier."
                )
            status = batch.get("status")
            if status not in KNOWN_STATUSES:
                raise OpenAIBatchError(
                    "The Batch API returned an unknown batch status."
                )
            _report(progress_callback, _progress_message(batch, batch_id, status))
            if status in TERMINAL_STATUSES:
                terminal = True
                break
            if time.monotonic() >= deadline:
                raise OpenAIBatchError(
                    f"Luna Batch {batch_id} exceeded its 24-hour completion window."
                )
            _wait_for_poll(cancellation_check, deadline)
            batch = transport.json_request("GET", f"/batches/{batch_id}")

        files: list[bytes] = []
        for field in ("output_file_id", "error_file_id"):
            if batch.get(field):
                file_id = _validated_id(batch[field], "file")
                owned_file_ids.add(file_id)
                _raise_if_cancelled(cancellation_check)
                files.append(transport.content(file_id))
        _raise_if_cancelled(cancellation_check)
        batch_error = None
        if batch["status"] != "completed":
            reason = _describe_error(batch.get("errors"), api_key)
            batch_error = f"Luna Batch {batch_id} {batch['status']}: {reason}"
        results = _parse_results(files, requests, api_key, batch_error)
        results_mapped = True
        return results
    except OpenAIBatchCancelled as exc:
        cancel_reason = ""
        if batch_id is not None and not terminal:
            cancel_reason = _cancel_owned_batch(transport, batch_id, progress_callback)
        elif batch_id is not None:
            cancel_reason = _retained_results_message(batch_id)
        suffix = f" {cancel_reason}" if cancel_reason else ""
        raise OpenAIBatchCancelled(f"Luna Batch cancelled.{suffix}") from exc
    except OpenAIBatchError as exc:
        suffix = ""
        if batch_id is not None and not terminal:
            suffix = " " + _cancel_owned_batch(transport, batch_id, progress_callback)
        elif batch_id is not None:
            suffix = " " + _retained_results_message(batch_id)
        elif creation_started and batch_id is None:
            suffix = (
                " Batch creation could not be confirmed; check the OpenAI Batch dashboard "
                "before resubmitting. The uploaded input was retained."
            )
        raise OpenAIBatchError(f"{_sanitize(str(exc), api_key)}{suffix}") from exc
    except Exception as exc:  # noqa: BLE001 - cancel a submitted paid job before surfacing failures
        # Even unexpected local failures (for example a destroyed Qt callback)
        # must not silently abandon a running, billable job.
        suffix = ""
        if batch_id is not None and not terminal:
            suffix = " " + _cancel_owned_batch(transport, batch_id, progress_callback)
        elif batch_id is not None:
            suffix = " " + _retained_results_message(batch_id)
        raise OpenAIBatchError(
            f"Luna Batch processing failed: {_sanitize(str(exc), api_key)}{suffix}"
        ) from None
    finally:
        if results_mapped or (input_file_id is not None and not creation_started):
            # Output/error identifiers come only from our own terminal batch.
            # Never remove a running batch's input, or paid results that could
            # still be recovered after a download, parsing or cancellation error.
            for file_id in sorted(owned_file_ids):
                try:
                    transport.json_request(
                        "DELETE", f"/files/{file_id}", cancellable=False
                    )
                except OpenAIBatchError as exc:
                    _report(progress_callback, f"Batch file cleanup failed: {exc}")


# ### Input and output validation ###
def _prepare_requests(requests: Mapping[str, dict]) -> bytes:
    if not isinstance(requests, Mapping) or not requests:
        raise OpenAIBatchError("Luna Batch requires at least one request.")
    if len(requests) > MAX_BATCH_REQUESTS:
        raise OpenAIBatchError("A Luna Batch may contain at most 50,000 requests.")
    lines: list[bytes] = []
    total = 0
    for custom_id, body in requests.items():
        if not isinstance(custom_id, str) or not _CUSTOM_ID_PATTERN.fullmatch(
            custom_id
        ):
            raise OpenAIBatchError("A Luna Batch custom_id is invalid.")
        if not isinstance(body, dict) or body.get("model") not in (
            LUNA_MODEL,
            LUNA_BATCH_MODEL,
        ):
            raise OpenAIBatchError(
                "Only GPT-5.6 Luna Responses requests support Luna Batch."
            )
        if body.get("stream") or body.get("background"):
            raise OpenAIBatchError(
                "Luna Batch does not accept streaming or background responses."
            )
        wire_body = dict(body, model=LUNA_MODEL)
        try:
            line = (
                json.dumps(
                    {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/responses",
                        "body": wire_body,
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
        except (TypeError, ValueError, UnicodeError) as exc:
            raise OpenAIBatchError("A Luna Batch request is not valid JSON.") from exc
        total += len(line)
        if total > MAX_INPUT_BYTES:
            raise OpenAIBatchError(
                "The Luna Batch input exceeds the 200 MB file limit."
            )
        lines.append(line)
    return b"".join(lines)


def _parse_results(
    files: list[bytes],
    requests: Mapping[str, dict],
    api_key: str,
    batch_error: str | None,
) -> dict[str, BatchResult]:
    results: dict[str, BatchResult] = {}
    for contents in files:
        for line in contents.splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except (ValueError, UnicodeError) as exc:
                raise OpenAIBatchError(
                    "Luna Batch returned malformed output JSONL."
                ) from exc
            custom_id = record.get("custom_id") if isinstance(record, dict) else None
            if not isinstance(custom_id, str) or custom_id not in requests:
                raise OpenAIBatchError(
                    "Luna Batch output contains an unknown custom_id."
                )
            if custom_id in results:
                results[custom_id] = BatchResult(
                    None, "Luna Batch returned duplicate custom_id results."
                )
                continue
            response = record.get("response")
            if record.get("error"):
                results[custom_id] = BatchResult(
                    None, _describe_error(record["error"], api_key)
                )
            elif not isinstance(response, dict):
                results[custom_id] = BatchResult(
                    None, "Luna Batch returned no response for this request."
                )
            else:
                body = response.get("body")
                code = response.get("status_code")
                if not isinstance(code, int) or not 200 <= code < 300:
                    reason = _describe_error(
                        body.get("error") if isinstance(body, dict) else None, api_key
                    )
                    results[custom_id] = BatchResult(
                        None, f"Luna Batch request failed ({code}): {reason}"
                    )
                elif not isinstance(body, dict):
                    results[custom_id] = BatchResult(
                        None, "Luna Batch response body is not an object."
                    )
                elif body.get("error") or body.get("status") in {
                    "failed",
                    "incomplete",
                    "cancelled",
                }:
                    reason = _describe_error(
                        body.get("error") or body.get("incomplete_details"), api_key
                    )
                    results[custom_id] = BatchResult(
                        None, f"Luna Batch response was not completed: {reason}"
                    )
                else:
                    results[custom_id] = BatchResult(body, None)
    return {
        custom_id: results.get(
            custom_id,
            BatchResult(
                None,
                batch_error or "Luna Batch omitted the result for this custom_id.",
            ),
        )
        for custom_id in requests
    }


# ### HTTP transport ###
class _Transport:
    def __init__(
        self,
        api_key: str,
        opener: Any,
        cancellation_check: Callable[[], bool] | None,
    ) -> None:
        self.api_key = api_key
        self.opener = opener.open if hasattr(opener, "open") else opener
        self.cancellation_check = cancellation_check

    def json_request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        *,
        cancellable: bool = True,
    ) -> dict:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        contents = self.request(
            method,
            path,
            data,
            "application/json",
            MAX_METADATA_BYTES,
            cancellable=cancellable,
        )
        try:
            result = json.loads(contents)
        except (ValueError, UnicodeError) as exc:
            raise OpenAIBatchError(
                "The OpenAI Batch API returned invalid JSON."
            ) from exc
        if not isinstance(result, dict):
            raise OpenAIBatchError("The OpenAI Batch API returned invalid metadata.")
        return result

    def upload(self, contents: bytes) -> dict:
        boundary = f"housemaker-{uuid.uuid4().hex}"
        data = (
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nbatch\r\n'
                f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="requests.jsonl"\r\n'
                "Content-Type: application/jsonl\r\n\r\n"
            ).encode("ascii")
            + contents
            + f"\r\n--{boundary}--\r\n".encode("ascii")
        )
        result = self.request(
            "POST",
            "/files",
            data,
            f"multipart/form-data; boundary={boundary}",
            MAX_METADATA_BYTES,
        )
        try:
            payload = json.loads(result)
        except (ValueError, UnicodeError) as exc:
            raise OpenAIBatchError(
                "OpenAI returned an invalid batch file upload response."
            ) from exc
        if not isinstance(payload, dict):
            raise OpenAIBatchError("OpenAI returned invalid batch file metadata.")
        return payload

    def content(self, file_id: str) -> bytes:
        return self.request(
            "GET", f"/files/{file_id}/content", None, None, MAX_OUTPUT_BYTES
        )

    def request(
        self,
        method: str,
        path: str,
        data: bytes | None,
        content_type: str | None,
        limit: int,
        *,
        cancellable: bool = True,
    ) -> bytes:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if content_type is not None:
            headers["Content-Type"] = content_type
        request = Request(
            OPENAI_API_URL + path, data=data, headers=headers, method=method
        )
        try:
            with self.opener(request, timeout=NETWORK_TIMEOUT_SECONDS) as response:
                # Do not interrupt upload/create metadata halfway through: first
                # obtain its identifier so cancellation can target our own job.
                check = (
                    self.cancellation_check if cancellable and method == "GET" else None
                )
                return _read_bounded(response, limit, check)
        except HTTPError as exc:
            try:
                raw = _read_bounded(exc, MAX_METADATA_BYTES)
                error = json.loads(raw)
                reason = _describe_error(
                    error.get("error") if isinstance(error, dict) else None,
                    self.api_key,
                )
            except (OpenAIBatchError, ValueError, UnicodeError):
                reason = "No usable server reason was returned."
            finally:
                exc.close()
            raise OpenAIBatchError(
                f"OpenAI Batch API request failed ({exc.code}): {reason}"
            ) from None
        except (URLError, OSError, TimeoutError) as exc:
            reason = _sanitize(str(getattr(exc, "reason", exc)), self.api_key)
            raise OpenAIBatchError(
                f"OpenAI Batch API network request failed: {reason}"
            ) from None
        except OpenAIBatchError:
            raise
        except Exception as exc:  # noqa: BLE001 - sanitize external transport errors and their causes
            raise OpenAIBatchError(
                f"OpenAI Batch API transport failed: {_sanitize(str(exc), self.api_key)}"
            ) from None


# ### Cancellation, progress and safe messages ###
def _retained_results_message(batch_id: str) -> str:
    return (
        f"Files for Luna Batch {batch_id} were retained. Retrieve its paid results "
        "from the OpenAI Batch dashboard before resubmitting."
    )


def _raise_if_cancelled(check: Callable[[], bool] | None) -> None:
    if check is not None and check():
        raise OpenAIBatchCancelled("Luna Batch cancelled.")


def _wait_for_poll(check: Callable[[], bool] | None, deadline: float) -> None:
    end = min(time.monotonic() + POLL_INTERVAL_SECONDS, deadline)
    while time.monotonic() < end:
        _raise_if_cancelled(check)
        time.sleep(min(CANCELLATION_POLL_SECONDS, max(0, end - time.monotonic())))
    _raise_if_cancelled(check)


def _cancel_owned_batch(
    transport: _Transport,
    batch_id: str,
    callback: Callable[[str], None] | None,
) -> str:
    try:
        transport.json_request("POST", f"/batches/{batch_id}/cancel", cancellable=False)
        message = (
            f"Cancellation requested for {batch_id}; remote cancellation can take up to "
            "10 minutes. Its files were retained until it is terminal."
        )
    except OpenAIBatchError as exc:
        message = f"Remote cancellation of {batch_id} could not be confirmed: {exc}"
    _report(callback, message)
    return message


def _validated_id(value: Any, kind: str) -> str:
    pattern = _FILE_ID_PATTERN if kind == "file" else _BATCH_ID_PATTERN
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise OpenAIBatchError(
            f"The OpenAI Batch API returned an invalid {kind} identifier."
        )
    return value


def _read_bounded(
    response: Any,
    limit: int,
    cancellation_check: Callable[[], bool] | None = None,
) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        _raise_if_cancelled(cancellation_check)
        chunk = response.read(min(64 * 1024, limit - total + 1))
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > limit:
            raise OpenAIBatchError(
                "An OpenAI Batch API response exceeded the safe download limit."
            )
        chunks.append(chunk)


def _sanitize(message: str, api_key: str) -> str:
    message = message.replace(api_key, "[redacted]") if api_key else message
    message = re.sub(
        r"data:[^\s\"'<>]+", "[image data redacted]", message, flags=re.IGNORECASE
    )
    message = re.sub(
        r"Bearer\s+\S+|sk-[A-Za-z0-9_-]+", "[redacted]", message, flags=re.IGNORECASE
    )
    return " ".join(message.split())[:600]


def _describe_error(error: Any, api_key: str) -> str:
    if isinstance(error, dict):
        entries = error.get("data")
        if isinstance(entries, list):
            reasons = [_describe_error(entry, api_key) for entry in entries[:3]]
            return _sanitize("; ".join(reasons), api_key)
        text = error.get("message") or error.get("reason") or error.get("code")
        if isinstance(text, str):
            return _sanitize(text, api_key)
    if isinstance(error, str):
        return _sanitize(error, api_key)
    return "No usable server reason was returned."


def _progress_message(batch: dict, batch_id: str, status: str) -> str:
    counts = batch.get("request_counts")
    detail = ""
    if isinstance(counts, dict):
        safe_counts = {
            key: value
            for key, value in counts.items()
            if type(value) is int and value >= 0
        }
        detail = (
            f" ({safe_counts.get('completed', 0)} completed, "
            f"{safe_counts.get('failed', 0)} failed / {safe_counts.get('total', 0)})"
        )
    return f"Luna Batch {batch_id}: {status}{detail}; completion window up to 24 hours."


def _report(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        try:
            callback(message)
        except Exception:  # noqa: BLE001 - a broken UI callback must not orphan a paid job
            # A closed UI must not abandon an already submitted, billable batch.
            _LOGGER.debug("The OpenAI Batch progress callback is unavailable.")
