# ### Imports ###
from __future__ import annotations

import base64
import math
import re
import time
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

from .meshy_generation import (
    MeshyGenerationError,
    SleepFunction,
    UrlOpenFunction,
    _build_http_error,
    _encode_json,
    _request_json,
    _require_api_key,
    _require_task_id,
)

# ### Constants ###
MESHY_IMAGE_TO_IMAGE_ENDPOINT = "https://api.meshy.ai/openapi/v1/image-to-image"
MESHY_REFERENCE_AI_MODEL = "nano-banana"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0
DEFAULT_POLL_INTERVAL_SECONDS = 2.0
DEFAULT_MAX_POLLS = 360
MAX_REFERENCE_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_DOWNLOAD_BYTES = 64 * 1024 * 1024
DOWNLOAD_CHUNK_BYTES = 64 * 1024
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
KNOWN_TASK_STATUSES = frozenset(
    {"PENDING", "IN_PROGRESS", "SUCCEEDED", "FAILED", "CANCELED"}
)
URL_IN_ERROR_PATTERN = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)


# ### Exceptions ###
class MeshyReferenceGenerationError(RuntimeError):
    """Safe, user-displayable Meshy reference-image failure."""


class MeshyReferenceGenerationCancelled(MeshyReferenceGenerationError):
    """The caller stopped waiting for its Meshy reference image."""


# ### Public API ###
def request_meshy_reference_image(
    image_png: bytes,
    *,
    api_key: str,
    prompt: str,
    output_size: tuple[int, int] = (1024, 1024),
    cancellation_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[str, int], None] | None = None,
    opener: UrlOpenFunction | None = None,
    sleep: SleepFunction = time.sleep,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    max_polls: int = DEFAULT_MAX_POLLS,
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> bytes:
    """Edit one reference through Meshy's image API, without retrying paid creates.

    Meshy controls the image resolution; the caller decodes and resizes the result.
    Cancellation stops local polling, not the remote task: DELETE is destructive
    and cannot cancel an image task once Meshy starts processing it.
    """
    normalized_key, image, normalized_prompt = _validate_request(
        api_key, image_png, prompt, output_size
    )
    poll_interval = _finite_number(poll_interval_seconds, "poll interval", 0.0)
    timeout = _finite_number(timeout_seconds, "request timeout", 0.001)
    if isinstance(max_polls, bool) or not isinstance(max_polls, int) or max_polls <= 0:
        raise ValueError("Meshy reference max polls must be a positive integer.")
    secrets = (normalized_key,)
    _raise_if_cancelled(cancellation_check)
    body = {
        "ai_model": MESHY_REFERENCE_AI_MODEL,
        "prompt": normalized_prompt,
        "reference_image_urls": [
            "data:image/png;base64," + base64.b64encode(image).decode("ascii")
        ],
        "generate_multi_view": False,
        "aspect_ratio": "1:1",
        "remove_background": False,
    }
    headers = {
        "Authorization": f"Bearer {normalized_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    created = _safe_request_json(
        Request(
            MESHY_IMAGE_TO_IMAGE_ENDPOINT, data=_encode_json(body), headers=headers
        ),
        timeout,
        opener,
        secrets,
    )
    _raise_if_cancelled(cancellation_check)
    task_id = created.get("result")
    if not isinstance(task_id, str):
        raise MeshyReferenceGenerationError(
            "Meshy did not return an image task identifier."
        )
    try:
        task_id = _require_task_id(task_id)
    except ValueError as error:
        raise MeshyReferenceGenerationError(
            "Meshy returned an invalid image task identifier."
        ) from error

    task_url = f"{MESHY_IMAGE_TO_IMAGE_ENDPOINT}/{quote(task_id, safe='')}"
    for poll_index in range(max_polls):
        _raise_if_cancelled(cancellation_check)
        task = _safe_request_json(
            Request(task_url, headers=headers), timeout, opener, secrets
        )
        _raise_if_cancelled(cancellation_check)
        raw_status = task.get("status")
        status = raw_status.upper() if isinstance(raw_status, str) else ""
        if status not in KNOWN_TASK_STATUSES:
            raise MeshyReferenceGenerationError(
                "Meshy returned an invalid image task status."
            )
        if progress_callback is not None:
            progress_callback(status, _task_progress(task))
        _raise_if_cancelled(cancellation_check)
        if status == "SUCCEEDED":
            image_url = _image_url(task)
            return _download_image(
                image_url, normalized_key, timeout, opener, cancellation_check
            )
        if status in {"FAILED", "CANCELED"}:
            task_error = task.get("task_error")
            message = (
                task_error.get("message") if isinstance(task_error, dict) else None
            )
            detail = (
                message.strip()
                if isinstance(message, str) and message.strip()
                else "provider error"
            )
            raise MeshyReferenceGenerationError(
                _safe_error(
                    f"Meshy reference image {status.lower()}: {detail}", secrets
                )
            )
        if poll_index + 1 < max_polls:
            _interruptible_wait(poll_interval, sleep, cancellation_check)
    raise MeshyReferenceGenerationError(
        "Meshy reference image timed out before completion."
    )


# ### Input and task validation ###
def _validate_request(
    api_key: str,
    image_png: bytes,
    prompt: str,
    output_size: tuple[int, int],
) -> tuple[str, bytes, str]:
    if not isinstance(api_key, str):
        raise TypeError("A Meshy API key is required.")
    normalized_key = _require_api_key(api_key)
    if "\r" in normalized_key or "\n" in normalized_key:
        raise ValueError("The Meshy API key is invalid.")
    if not isinstance(image_png, (bytes, bytearray, memoryview)):
        raise TypeError("The Meshy reference image must contain PNG bytes.")
    image = bytes(image_png)
    if not image.startswith(PNG_SIGNATURE) or len(image) <= len(PNG_SIGNATURE):
        raise ValueError("The Meshy reference image must contain PNG bytes.")
    if len(image) > MAX_REFERENCE_IMAGE_BYTES:
        raise ValueError("The Meshy reference image exceeds the 20 MB limit.")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("A Meshy reference prompt is required.")
    if (
        not isinstance(output_size, (tuple, list))
        or len(output_size) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in output_size
        )
        or output_size[0] != output_size[1]
    ):
        raise ValueError("Meshy reference generation requires a square output size.")
    return normalized_key, image, prompt.strip()


def _finite_number(value: float, label: str, minimum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Meshy reference {label} is invalid.") from error
    if isinstance(value, bool) or not math.isfinite(number) or number < minimum:
        raise ValueError(f"Meshy reference {label} is invalid.")
    return number


def _task_progress(task: dict[str, Any]) -> int:
    value = task.get("progress", 0)
    if isinstance(value, bool):
        return 0
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError, OverflowError):
        return 0


def _image_url(task: dict[str, Any]) -> str:
    images = task.get("image_urls")
    if (
        not isinstance(images, list)
        or len(images) != 1
        or not isinstance(images[0], str)
    ):
        raise MeshyReferenceGenerationError(
            "Meshy succeeded without one reference image."
        )
    url = images[0].strip()
    if not _is_https_url(url):
        raise MeshyReferenceGenerationError(
            "Meshy returned an invalid reference image URL."
        )
    return url


def _is_https_url(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        return (
            parsed.scheme.lower() == "https"
            and bool(parsed.hostname)
            and parsed.username is None
            and parsed.password is None
            and not any(character.isspace() for character in url)
        )
    except ValueError:
        return False


# ### HTTP and error handling ###
def _safe_request_json(
    request: Request,
    timeout: float,
    opener: UrlOpenFunction | None,
    secrets: tuple[str, ...],
) -> dict[str, Any]:
    try:
        return _request_json(request, timeout, opener, secrets)
    except MeshyGenerationError as error:
        raise MeshyReferenceGenerationError(_safe_error(str(error), secrets)) from error


def _download_image(
    image_url: str,
    api_key: str,
    timeout: float,
    opener: UrlOpenFunction | None,
    cancellation_check: Callable[[], bool] | None,
) -> bytes:
    _raise_if_cancelled(cancellation_check)
    url_opener = urlopen if opener is None else opener
    secrets = (api_key, image_url)
    # Signed asset URLs authorize themselves; never forward the API key to a CDN.
    request = Request(image_url, headers={"Accept": "image/*"})
    try:
        with url_opener(request, timeout=timeout) as response:
            _raise_if_cancelled(cancellation_check)
            get_final_url = getattr(response, "geturl", None)
            if callable(get_final_url) and not _is_https_url(get_final_url()):
                raise MeshyReferenceGenerationError(
                    "Meshy image download redirected to an unsafe URL."
                )
            headers = getattr(response, "headers", {})
            mime_type = (
                str(headers.get("Content-Type", "")).split(";", 1)[0].strip().lower()
            )
            if mime_type and not (
                mime_type.startswith("image/")
                or mime_type == "application/octet-stream"
            ):
                raise MeshyReferenceGenerationError(
                    "Meshy returned a non-image reference download."
                )
            chunks: list[bytes] = []
            size = 0
            while True:
                _raise_if_cancelled(cancellation_check)
                chunk = response.read(
                    min(DOWNLOAD_CHUNK_BYTES, MAX_IMAGE_DOWNLOAD_BYTES + 1 - size)
                )
                _raise_if_cancelled(cancellation_check)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_IMAGE_DOWNLOAD_BYTES:
                    raise MeshyReferenceGenerationError(
                        "Meshy returned an oversized reference image."
                    )
                chunks.append(chunk)
    except HTTPError as error:
        safe_error = _build_http_error(error, secrets)
        raise MeshyReferenceGenerationError(
            _safe_error(str(safe_error), secrets)
        ) from error
    except (URLError, TimeoutError, OSError) as error:
        reason = str(getattr(error, "reason", None) or error).strip()
        raise MeshyReferenceGenerationError(
            _safe_error(f"Meshy reference image download failed: {reason}", secrets)
        ) from error
    image = b"".join(chunks)
    if not image:
        raise MeshyReferenceGenerationError("Meshy returned an empty reference image.")
    return image


def _safe_error(message: str, secrets: tuple[str, ...]) -> str:
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[redacted]")
    return URL_IN_ERROR_PATTERN.sub("[redacted URL]", message)


# ### Cancellation ###
def _raise_if_cancelled(cancellation_check: Callable[[], bool] | None) -> None:
    if cancellation_check is not None and cancellation_check():
        raise MeshyReferenceGenerationCancelled(
            "Meshy reference generation canceled locally. Its remote task may still finish and consume credits."
        )


def _interruptible_wait(
    seconds: float,
    sleep: SleepFunction,
    cancellation_check: Callable[[], bool] | None,
) -> None:
    remaining = seconds
    while remaining > 0:
        _raise_if_cancelled(cancellation_check)
        interval = min(remaining, 0.1)
        sleep(interval)
        remaining = max(0.0, remaining - interval)
    _raise_if_cancelled(cancellation_check)
