# ### Imports ###
from __future__ import annotations

import io
import json
import traceback
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from housemaker import openai_batch as batch


# ### Fixtures ###
def _body(text: str = "Test") -> dict:
    return {"model": batch.LUNA_BATCH_MODEL, "input": text}


def _metadata(status: str = "completed", **overrides: object) -> dict:
    return dict({"id": "batch_test", "status": status}, **overrides)


def _line(custom_id: str, *, body: dict | None = None, code: int = 200) -> bytes:
    return (
        json.dumps(
            {
                "custom_id": custom_id,
                "response": {
                    "status_code": code,
                    "body": {"status": "completed", "id": f"response-{custom_id}"}
                    if body is None
                    else body,
                },
                "error": None,
            }
        ).encode()
        + b"\n"
    )


class _Opener:
    def __init__(self, routes: list[tuple[str, str, object]]) -> None:
        self.routes = list(routes)
        self.calls: list[object] = []

    def __call__(self, request: object, *, timeout: float) -> io.BytesIO:
        self.calls.append(request)
        if not 0 < timeout <= 30:
            raise AssertionError("Unbounded network timeout")
        if not self.routes:
            raise AssertionError(
                f"Unexpected request: {request.method} {request.full_url}"
            )
        method, path, payload = self.routes.pop(0)
        if request.method != method or request.full_url != batch.OPENAI_API_URL + path:
            raise AssertionError(
                f"Expected {method} {path}, got {request.method} {request.full_url}"
            )
        if isinstance(payload, Exception):
            raise payload
        return io.BytesIO(
            payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        )


def _completed_opener(
    contents: bytes,
    *,
    cleanup: bool = True,
    **metadata: object,
) -> _Opener:
    routes = [
        ("POST", "/files", {"id": "file-input"}),
        ("POST", "/batches", _metadata(output_file_id="file-output", **metadata)),
        ("GET", "/files/file-output/content", contents),
    ]
    if cleanup:
        routes.extend(
            [
                ("DELETE", "/files/file-input", {"deleted": True}),
                ("DELETE", "/files/file-output", {"deleted": True}),
            ]
        )
    return _Opener(routes)


# ### Input validation tests ###
class OpenAIBatchValidationTests(unittest.TestCase):
    def test_only_batch_alias_is_normalized(self) -> None:
        self.assertEqual(
            batch.to_openai_model(batch.LUNA_BATCH_MODEL), batch.LUNA_MODEL
        )
        self.assertEqual(batch.to_openai_model(batch.LUNA_MODEL), batch.LUNA_MODEL)
        self.assertEqual(batch.to_openai_model("other-model"), "other-model")

    def test_invalid_inputs_never_make_a_network_call(self) -> None:
        cases = [
            ({}, "test-key"),
            ({"x": _body()}, " "),
            ({"not safe\n": _body()}, "test-key"),
            ({"x": {"model": "different"}}, "test-key"),
            ({"x": dict(_body(), stream=True)}, "test-key"),
            ({"x": dict(_body(), background=True)}, "test-key"),
            ({"x": dict(_body(), input=float("nan"))}, "test-key"),
            ({"x": None}, "test-key"),
        ]
        for requests, key in cases:
            with self.subTest(requests=requests, key=key):
                opener = _Opener([])
                with self.assertRaises(batch.OpenAIBatchError):
                    batch.run_responses_batch(requests, api_key=key, opener=opener)
                self.assertEqual(opener.calls, [])

    def test_request_count_and_input_byte_limit_are_checked_before_upload(self) -> None:
        with (
            patch.object(batch, "MAX_BATCH_REQUESTS", 1),
            self.assertRaisesRegex(batch.OpenAIBatchError, "50,000"),
        ):
            batch.run_responses_batch({"a": _body(), "b": _body()}, api_key="key")
        with (
            patch.object(batch, "MAX_INPUT_BYTES", 5),
            self.assertRaisesRegex(batch.OpenAIBatchError, "200 MB"),
        ):
            batch.run_responses_batch({"a": _body()}, api_key="key")


# ### Successful lifecycle and result tests ###
class OpenAIBatchLifecycleTests(unittest.TestCase):
    def test_multipart_upload_and_real_batch_creation_preserve_inputs(self) -> None:
        requests = {"one": _body("Original"), "two": _body("Second")}
        original = json.loads(json.dumps(requests))
        opener = _completed_opener(_line("two") + _line("one"))
        progress: list[str] = []
        results = batch.run_responses_batch(
            requests,
            api_key="test-key",
            opener=opener,
            progress_callback=progress.append,
        )
        self.assertEqual(requests, original)
        self.assertEqual(list(results), ["one", "two"])
        self.assertEqual(results["one"].body["id"], "response-one")
        self.assertEqual(results["two"].body["id"], "response-two")
        self.assertTrue(all(result.error is None for result in results.values()))
        upload, create = opener.calls[:2]
        self.assertEqual(upload.get_header("Authorization"), "Bearer test-key")
        self.assertIn(b'name="purpose"\r\n\r\nbatch', upload.data)
        self.assertIn(b'filename="requests.jsonl"', upload.data)
        self.assertIn(b'"url":"/v1/responses"', upload.data)
        self.assertIn(b'"model":"gpt-5.6-luna"', upload.data)
        self.assertNotIn(batch.LUNA_BATCH_MODEL.encode(), upload.data)
        self.assertEqual(
            json.loads(create.data),
            {
                "input_file_id": "file-input",
                "endpoint": "/v1/responses",
                "completion_window": "24h",
            },
        )
        self.assertTrue(
            any("batch_test" in text and "24 hours" in text for text in progress)
        )
        self.assertEqual(opener.routes, [])

    def test_poll_status_and_request_counts_are_reported(self) -> None:
        opener = _Opener(
            [
                ("POST", "/files", {"id": "file-input"}),
                ("POST", "/batches", _metadata("validating")),
                (
                    "GET",
                    "/batches/batch_test",
                    _metadata(
                        "in_progress",
                        request_counts={"total": 1, "completed": 0, "failed": 0},
                    ),
                ),
                ("GET", "/batches/batch_test", _metadata(output_file_id="file-output")),
                ("GET", "/files/file-output/content", _line("request-1")),
                ("DELETE", "/files/file-input", {}),
                ("DELETE", "/files/file-output", {}),
            ]
        )
        progress: list[str] = []
        with patch.object(batch, "POLL_INTERVAL_SECONDS", 0):
            result = batch.request_luna_batch_response(
                _body(),
                api_key="key",
                opener=opener,
                progress_callback=progress.append,
            )
        self.assertEqual(result["status"], "completed")
        self.assertTrue(
            any(
                "in_progress" in text and "0 completed, 0 failed / 1" in text
                for text in progress
            )
        )
        self.assertEqual(opener.routes, [])

    def test_output_and_error_files_are_mapped_by_id_with_partial_success(self) -> None:
        error = json.dumps(
            {
                "custom_id": "bad",
                "error": {"code": "invalid_prompt", "message": "Bad input"},
            }
        ).encode()
        opener = _Opener(
            [
                ("POST", "/files", {"id": "file-input"}),
                (
                    "POST",
                    "/batches",
                    _metadata(output_file_id="file-output", error_file_id="file-error"),
                ),
                ("GET", "/files/file-output/content", _line("good")),
                ("GET", "/files/file-error/content", error),
                ("DELETE", "/files/file-error", {}),
                ("DELETE", "/files/file-input", {}),
                ("DELETE", "/files/file-output", {}),
            ]
        )
        results = batch.run_responses_batch(
            {"bad": _body(), "good": _body(), "missing": _body()},
            api_key="key",
            opener=opener,
        )
        self.assertEqual(results["bad"].error, "Bad input")
        self.assertIsNotNone(results["good"].body)
        self.assertIn("omitted", results["missing"].error)
        self.assertEqual(opener.routes, [])

    def test_duplicate_ids_fail_that_result_without_corrupting_other_results(
        self,
    ) -> None:
        opener = _completed_opener(_line("bad") + _line("good") + _line("bad"))
        results = batch.run_responses_batch(
            {"bad": _body(), "good": _body()},
            api_key="key",
            opener=opener,
        )
        self.assertIn("duplicate", results["bad"].error)
        self.assertIsNone(results["bad"].body)
        self.assertIsNotNone(results["good"].body)

    def test_unknown_custom_id_and_invalid_json_are_explicit_errors(self) -> None:
        for contents, reason in [
            (_line("foreign"), "unknown custom_id"),
            (b"invalid json", "malformed"),
        ]:
            with self.subTest(contents=contents):
                opener = _completed_opener(contents, cleanup=False)
                with self.assertRaisesRegex(batch.OpenAIBatchError, reason) as caught:
                    batch.run_responses_batch(
                        {"one": _body()}, api_key="key", opener=opener
                    )
                self.assertIn("batch_test", str(caught.exception))
                self.assertIn("OpenAI Batch dashboard", str(caught.exception))
                self.assertEqual(opener.routes, [])

    def test_failed_expired_and_cancelled_batches_show_server_reason(self) -> None:
        for status in ("failed", "expired", "cancelled"):
            with self.subTest(status=status):
                opener = _Opener(
                    [
                        ("POST", "/files", {"id": "file-input"}),
                        (
                            "POST",
                            "/batches",
                            _metadata(
                                status,
                                errors={"data": [{"message": "Token quota exceeded"}]},
                            ),
                        ),
                        ("DELETE", "/files/file-input", {}),
                    ]
                )
                with self.assertRaisesRegex(
                    batch.OpenAIBatchError, f"{status}: Token quota exceeded"
                ):
                    batch.request_luna_batch_response(
                        _body(), api_key="key", opener=opener
                    )

    def test_http_response_failure_and_incomplete_body_are_not_successes(self) -> None:
        for response, code in [
            ({"error": {"message": "Unsupported image"}}, 400),
            (
                {
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                },
                200,
            ),
        ]:
            with self.subTest(response=response):
                opener = _completed_opener(_line("request-1", body=response, code=code))
                with self.assertRaises(batch.OpenAIBatchError):
                    batch.request_luna_batch_response(
                        _body(), api_key="key", opener=opener
                    )


# ### Cancellation, network and safety tests ###
class OpenAIBatchSafetyTests(unittest.TestCase):
    def test_already_cancelled_does_not_submit_anything(self) -> None:
        opener = _Opener([])
        with self.assertRaises(batch.OpenAIBatchCancelled):
            batch.request_luna_batch_response(
                _body(), api_key="key", opener=opener, cancellation_check=lambda: True
            )
        self.assertEqual(opener.calls, [])

    def test_cancel_before_creation_removes_only_our_uploaded_input(self) -> None:
        opener = _Opener(
            [
                ("POST", "/files", {"id": "file-input"}),
                ("DELETE", "/files/file-input", {}),
            ]
        )
        with self.assertRaises(batch.OpenAIBatchCancelled):
            batch.request_luna_batch_response(
                _body(),
                api_key="key",
                opener=opener,
                cancellation_check=lambda: len(opener.calls) >= 1,
            )
        self.assertEqual(opener.routes, [])

    def test_cancel_submitted_batch_does_not_wait_or_delete_its_files(self) -> None:
        opener = _Opener(
            [
                ("POST", "/files", {"id": "file-input"}),
                ("POST", "/batches", _metadata("in_progress")),
                ("POST", "/batches/batch_test/cancel", _metadata("cancelling")),
            ]
        )
        with self.assertRaisesRegex(batch.OpenAIBatchCancelled, "10 minutes"):
            batch.request_luna_batch_response(
                _body(),
                api_key="key",
                opener=opener,
                cancellation_check=lambda: len(opener.calls) >= 2,
            )
        self.assertEqual(len(opener.calls), 3)
        self.assertEqual(opener.routes, [])

    def test_deadline_requests_remote_cancellation(self) -> None:
        opener = _Opener(
            [
                ("POST", "/files", {"id": "file-input"}),
                ("POST", "/batches", _metadata("in_progress")),
                ("POST", "/batches/batch_test/cancel", _metadata("cancelling")),
            ]
        )
        with (
            patch.object(batch, "MAX_WAIT_SECONDS", 0),
            self.assertRaisesRegex(batch.OpenAIBatchError, "exceeded its 24-hour"),
        ):
            batch.request_luna_batch_response(_body(), api_key="key", opener=opener)
        self.assertEqual(opener.routes, [])

    def test_creation_network_failure_is_not_retried_and_retains_input(self) -> None:
        opener = _Opener(
            [
                ("POST", "/files", {"id": "file-input"}),
                ("POST", "/batches", URLError("timeout")),
            ]
        )
        with self.assertRaisesRegex(
            batch.OpenAIBatchError, "check the OpenAI Batch dashboard"
        ):
            batch.request_luna_batch_response(_body(), api_key="key", opener=opener)
        self.assertEqual(len(opener.calls), 2)

    def test_http_error_reason_is_sanitized(self) -> None:
        error = HTTPError(
            "https://api.openai.com/v1/files",
            401,
            "Unauthorized",
            {},
            io.BytesIO(
                json.dumps(
                    {
                        "error": {
                            "message": "Rejected secret-key Bearer sk-private data:image/png;base64,PRIVATE"
                        }
                    }
                ).encode()
            ),
        )
        opener = _Opener([("POST", "/files", error)])
        with self.assertRaises(batch.OpenAIBatchError) as caught:
            batch.request_luna_batch_response(
                _body(), api_key="secret-key", opener=opener
            )
        text = str(caught.exception)
        self.assertIn("401", text)
        self.assertIn("Rejected", text)
        for secret in ("secret-key", "sk-private", "PRIVATE", "data:image"):
            self.assertNotIn(secret, text)

    def test_invalid_server_file_id_never_becomes_a_url_or_delete_target(self) -> None:
        opener = _Opener([("POST", "/files", {"id": "../../foreign"})])
        with self.assertRaisesRegex(batch.OpenAIBatchError, "invalid file identifier"):
            batch.request_luna_batch_response(_body(), api_key="key", opener=opener)
        self.assertEqual(len(opener.calls), 1)

    def test_both_official_file_id_prefixes_are_accepted_without_path_characters(
        self,
    ) -> None:
        for identifier in ("file-abc123", "file_123"):
            with self.subTest(identifier=identifier):
                self.assertEqual(batch._validated_id(identifier, "file"), identifier)
        for identifier in (
            "file_../foreign",
            "file-abc?query",
            "file_abc#fragment",
            "file_abc/extra",
        ):
            with (
                self.subTest(identifier=identifier),
                self.assertRaises(batch.OpenAIBatchError),
            ):
                batch._validated_id(identifier, "file")

    def test_unexpected_local_failure_cancels_our_submitted_batch(self) -> None:
        opener = _Opener(
            [
                ("POST", "/files", {"id": "file-input"}),
                ("POST", "/batches", _metadata("in_progress")),
                ("POST", "/batches/batch_test/cancel", _metadata("cancelling")),
            ]
        )

        def cancellation_check() -> bool:
            if len(opener.calls) >= 2:
                raise RuntimeError("Deleted UI")
            return False

        with self.assertRaisesRegex(batch.OpenAIBatchError, "Deleted UI"):
            batch.request_luna_batch_response(
                _body(),
                api_key="key",
                opener=opener,
                cancellation_check=cancellation_check,
            )
        self.assertEqual(opener.routes, [])

    def test_oversized_paid_output_is_retained_for_manual_recovery(self) -> None:
        opener = _completed_opener(b"x" * 101, cleanup=False)
        with (
            patch.object(batch, "MAX_OUTPUT_BYTES", 100),
            self.assertRaisesRegex(
                batch.OpenAIBatchError, "safe download limit"
            ) as caught,
        ):
            batch.request_luna_batch_response(_body(), api_key="key", opener=opener)
        self.assertIn("batch_test", str(caught.exception))
        self.assertIn("OpenAI Batch dashboard", str(caught.exception))
        self.assertFalse(any(request.method == "DELETE" for request in opener.calls))
        self.assertEqual(opener.routes, [])

    def test_malformed_paid_output_is_retained_for_manual_recovery(self) -> None:
        opener = _completed_opener(b"not valid JSONL", cleanup=False)
        with self.assertRaisesRegex(batch.OpenAIBatchError, "malformed") as caught:
            batch.request_luna_batch_response(_body(), api_key="key", opener=opener)
        self.assertIn("batch_test", str(caught.exception))
        self.assertIn("OpenAI Batch dashboard", str(caught.exception))
        self.assertFalse(any(request.method == "DELETE" for request in opener.calls))
        self.assertEqual(opener.routes, [])

    def test_failing_progress_callback_does_not_abandon_submitted_work(self) -> None:
        opener = _completed_opener(_line("request-1"))
        result = batch.request_luna_batch_response(
            _body(),
            api_key="key",
            opener=opener,
            progress_callback=lambda message: (_ for _ in ()).throw(
                RuntimeError("UI closed")
            ),
        )
        self.assertEqual(result["status"], "completed")

    def test_network_error_tracebacks_do_not_reveal_secret_causes(self) -> None:
        opener = _Opener(
            [
                (
                    "POST",
                    "/files",
                    URLError("private-api-key data:image/png;base64,IMAGESECRET"),
                )
            ]
        )
        with self.assertRaises(batch.OpenAIBatchError) as caught:
            batch.request_luna_batch_response(
                _body(), api_key="private-api-key", opener=opener
            )
        text = "".join(traceback.format_exception(caught.exception))
        self.assertNotIn("private-api-key", text)
        self.assertNotIn("IMAGESECRET", text)

    def test_remote_cancel_failure_is_visible_without_deleting_running_input(
        self,
    ) -> None:
        opener = _Opener(
            [
                ("POST", "/files", {"id": "file-input"}),
                ("POST", "/batches", _metadata("in_progress")),
                ("POST", "/batches/batch_test/cancel", URLError("offline")),
            ]
        )
        with self.assertRaisesRegex(
            batch.OpenAIBatchCancelled, "cancellation.*could not be confirmed"
        ):
            batch.request_luna_batch_response(
                _body(),
                api_key="key",
                opener=opener,
                cancellation_check=lambda: len(opener.calls) >= 2,
            )
        self.assertEqual(opener.routes, [])

    def test_unexpected_status_cancels_only_created_batch(self) -> None:
        opener = _Opener(
            [
                ("POST", "/files", {"id": "file-input"}),
                ("POST", "/batches", _metadata("unrecognized")),
                ("POST", "/batches/batch_test/cancel", _metadata("cancelling")),
            ]
        )
        with self.assertRaisesRegex(batch.OpenAIBatchError, "unknown batch status"):
            batch.request_luna_batch_response(_body(), api_key="key", opener=opener)
        self.assertEqual(opener.routes, [])

    def test_cleanup_failure_is_reported_without_discarding_success(self) -> None:
        opener = _Opener(
            [
                ("POST", "/files", {"id": "file-input"}),
                ("POST", "/batches", _metadata(output_file_id="file-output")),
                ("GET", "/files/file-output/content", _line("request-1")),
                ("DELETE", "/files/file-input", URLError("offline")),
                ("DELETE", "/files/file-output", {}),
            ]
        )
        progress: list[str] = []
        result = batch.request_luna_batch_response(
            _body(),
            api_key="key",
            opener=opener,
            progress_callback=progress.append,
        )
        self.assertEqual(result["status"], "completed")
        self.assertTrue(any("cleanup failed" in message for message in progress))
        self.assertEqual(opener.routes, [])

    def test_cancellation_while_downloading_terminal_output_is_checked(self) -> None:
        opener = _completed_opener(_line("request-1"), cleanup=False)
        with self.assertRaisesRegex(batch.OpenAIBatchCancelled, "batch_test.*retained"):
            batch.request_luna_batch_response(
                _body(),
                api_key="key",
                opener=opener,
                cancellation_check=lambda: len(opener.calls) >= 3,
            )
        # No cancel POST is required for a terminal batch, and paid results are
        # kept remotely because the cancelled download did not map them locally.
        self.assertEqual(opener.routes, [])


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
