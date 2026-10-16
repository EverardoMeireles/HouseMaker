# ### Imports ###
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request

import numpy as np

from housemaker.ceiling_height_estimation import (
    open_ceiling_height_estimator,
)
from housemaker.qwen_ceiling_height_estimation import (
    LOCAL_QWEN_CEILING_HEIGHT_MODEL,
    QWEN_SERVER_CONTEXT_TOKENS,
    QWEN_SERVER_HOST,
    QWEN_SERVER_IMAGE_MAX_TOKENS,
    QWEN_SERVER_IMAGE_MIN_TOKENS,
    QwenCeilingHeightError,
    QwenCeilingHeightModelBundle,
    QwenCeilingHeightServer,
    build_local_qwen_request_body,
    default_qwen_ceiling_height_bundle,
    extract_local_qwen_result,
)


# ### Test doubles ###
class _BytesResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._was_read = False

    def __enter__(self) -> _BytesResponse:  # noqa: PYI034
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, _size: int = -1) -> bytes:
        if self._was_read:
            return b""
        self._was_read = True
        return self._payload


class _Process:
    def __init__(self, return_code: int | None = None) -> None:
        self.return_code = return_code
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.return_code

    def terminate(self) -> None:
        self.terminated = True
        self.return_code = -15

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return 0 if self.return_code is None else self.return_code

    def kill(self) -> None:
        self.killed = True
        self.return_code = -9


# ### Fixture helpers ###
def _write_file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")
    return path


def _create_bundle(directory: Path) -> QwenCeilingHeightModelBundle:
    runtime_directory = directory / "llama-runtime"
    weights_directory = directory / "weights"
    cuda_directory = directory / "runtime"
    cuda_directory.mkdir(parents=True, exist_ok=True)
    return QwenCeilingHeightModelBundle(
        executable=_write_file(runtime_directory / "llama-server.exe"),
        language_model=_write_file(
            weights_directory / "Qwen3VL-8B-Instruct-Q4_K_M.gguf"
        ),
        vision_projector=_write_file(
            weights_directory / "mmproj-Qwen3VL-8B-Instruct-F16.gguf"
        ),
        cuda_runtime_directory=cuda_directory,
    )


def _result_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "estimate_m": {"type": "number"},
        },
        "required": ["estimate_m"],
        "additionalProperties": False,
    }


# ### Default bundle tests ###
class QwenCeilingHeightDefaultBundleTests(unittest.TestCase):
    def test_default_bundle_reuses_downloaded_qwen_directory(self) -> None:
        model_directory = Path("C:/LocalAppData/HouseMaker/models/qwen-image-2.1")
        with patch(
            "housemaker.qwen_ceiling_height_estimation.default_qwen_model_directory",
            return_value=model_directory,
        ):
            bundle = default_qwen_ceiling_height_bundle()

        self.assertEqual(
            bundle.executable,
            model_directory / "llama-runtime/llama-server.exe",
        )
        self.assertEqual(
            bundle.language_model,
            model_directory / "weights/Qwen3VL-8B-Instruct-Q4_K_M.gguf",
        )
        self.assertEqual(
            bundle.vision_projector,
            model_directory / "weights/mmproj-Qwen3VL-8B-Instruct-F16.gguf",
        )
        self.assertEqual(
            bundle.cuda_runtime_directory,
            model_directory / "runtime",
        )


# ### Request tests ###
class QwenCeilingHeightRequestTests(unittest.TestCase):
    def test_additional_plan_details_and_output_budget_do_not_change_ceiling_defaults(self):
        kwargs = {"prompt": "Read dimensions.", "result_schema": _result_schema()}
        regular = build_local_qwen_request_body("data:image/png;base64,YWJj", **kwargs)
        detailed = build_local_qwen_request_body(
            "data:image/png;base64,YWJj", **kwargs,
            additional_images=(("Raw detail at normalized bounds", "data:image/png;base64,ZGV0"),),
            max_output_tokens=6144,
        )

        self.assertEqual(regular["max_tokens"], 500)
        self.assertEqual(detailed["max_tokens"], 6144)
        content = detailed["messages"][0]["content"]
        self.assertEqual(content[-2]["text"], "Raw detail at normalized bounds")
        self.assertEqual(content[-1]["image_url"]["url"], "data:image/png;base64,ZGV0")

    def test_request_uses_image_input_and_strict_json_schema(self) -> None:
        schema = _result_schema()
        body = build_local_qwen_request_body(
            "data:image/jpeg;base64,YWJj",
            prompt="Estimate the finished ceiling height.",
            result_schema=schema,
        )

        self.assertEqual(body["model"], LOCAL_QWEN_CEILING_HEIGHT_MODEL)
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["reasoning_effort"], "none")
        self.assertEqual(
            body["chat_template_kwargs"],
            {"enable_thinking": False},
        )
        content = body["messages"][0]["content"]
        self.assertEqual(
            content[0],
            {
                "type": "text",
                "text": "Estimate the finished ceiling height.",
            },
        )
        self.assertEqual(
            next(item for item in content if item["type"] == "image_url"),
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,YWJj"},
            },
        )
        response_format = body["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["json_schema"]["strict"])
        self.assertEqual(response_format["json_schema"]["schema"], schema)

    def test_request_orders_previous_frame_before_current_frame(self) -> None:
        body = build_local_qwen_request_body(
            "data:image/jpeg;base64,Y3VycmVudA==",
            previous_image_data_uri="data:image/jpeg;base64,cHJldmlvdXM=",
            prompt="Detect movement between levels.",
            result_schema=_result_schema(),
        )

        images = [
            item["image_url"]["url"]
            for item in body["messages"][0]["content"]
            if item["type"] == "image_url"
        ]
        self.assertEqual(
            images,
            [
                "data:image/jpeg;base64,cHJldmlvdXM=",
                "data:image/jpeg;base64,Y3VycmVudA==",
            ],
        )

    def test_request_places_annotated_plan_before_video_frames(self) -> None:
        body = build_local_qwen_request_body(
            "data:image/jpeg;base64,Y3VycmVudA==",
            previous_image_data_uri="data:image/jpeg;base64,cHJldmlvdXM=",
            plan_image_data_uri="data:image/jpeg;base64,cGxhbg==",
            prompt="Use M1 only if its span is visible.",
            result_schema=_result_schema(),
        )

        images = [
            item["image_url"]["url"]
            for item in body["messages"][0]["content"]
            if item["type"] == "image_url"
        ]

        self.assertEqual(
            images,
            [
                "data:image/jpeg;base64,cGxhbg==",
                "data:image/jpeg;base64,cHJldmlvdXM=",
                "data:image/jpeg;base64,Y3VycmVudA==",
            ],
        )

    def test_request_rejects_unsafe_or_empty_input(self) -> None:
        cases = (
            ("https://example.invalid/frame.jpg", "Estimate height."),
            ("data:image/jpeg;base64,YWJj", "   "),
        )
        for image_uri, prompt in cases:
            with (
                self.subTest(image_uri=image_uri, prompt=prompt),
                self.assertRaises(QwenCeilingHeightError),
            ):
                build_local_qwen_request_body(
                    image_uri,
                    prompt=prompt,
                    result_schema=_result_schema(),
                )


# ### Response parsing tests ###
class QwenCeilingHeightResponseTests(unittest.TestCase):
    def test_extracts_structured_result_from_chat_completion(self) -> None:
        result = {"estimate_m": 2.65, "confidence": "medium"}
        payload = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(result),
                    }
                }
            ]
        }

        self.assertEqual(extract_local_qwen_result(payload), result)

    def test_rejects_missing_or_malformed_completion_content(self) -> None:
        cases = (
            None,
            {},
            {"choices": []},
            {"choices": [{"message": {"content": ""}}]},
            {"choices": [{"message": {"content": "not-json"}}]},
        )
        for payload in cases:
            with (
                self.subTest(payload=payload),
                self.assertRaises(QwenCeilingHeightError),
            ):
                extract_local_qwen_result(payload)

    def test_server_estimate_parses_mocked_loopback_response(self) -> None:
        captured: list[tuple[Request, dict[str, object]]] = []
        result = {"estimate_m": 2.7}

        def open_request(request: Request, **kwargs: object) -> _BytesResponse:
            captured.append((request, kwargs))
            return _BytesResponse(
                json.dumps(
                    {
                        "choices": [
                            {"message": {"content": json.dumps(result)}}
                        ]
                    }
                ).encode("utf-8")
            )

        with tempfile.TemporaryDirectory() as raw_directory:
            server = QwenCeilingHeightServer(
                _create_bundle(Path(raw_directory)),
                opener=open_request,
            )
            server._process = object()  # type: ignore[assignment]
            server._port = 41921
            server._api_key = "local-test-key"
            estimate = server.estimate(
                "data:image/jpeg;base64,YWJj",
                prompt="Estimate height.",
                result_schema=_result_schema(),
            )

        self.assertEqual(estimate, result)
        request, _kwargs = captured[0]
        self.assertEqual(
            request.full_url,
            f"http://{QWEN_SERVER_HOST}:41921/v1/chat/completions",
        )
        self.assertEqual(
            request.get_header("Authorization"),
            "Bearer local-test-key",
        )

    def test_dead_server_is_rejected_before_sending_frame(self) -> None:
        request_count = 0

        def unexpected_request(*_args: object, **_kwargs: object) -> object:
            nonlocal request_count
            request_count += 1
            raise AssertionError("A dead local server must not receive a frame.")

        with tempfile.TemporaryDirectory() as raw_directory:
            server = QwenCeilingHeightServer(
                _create_bundle(Path(raw_directory)),
                opener=unexpected_request,
            )
            server._process = _Process(return_code=9)  # type: ignore[assignment]
            server._port = 41921
            server._api_key = "local-test-key"
            try:
                with self.assertRaises(QwenCeilingHeightError):
                    server.estimate(
                        "data:image/jpeg;base64,YWJj",
                        prompt="Estimate height.",
                        result_schema=_result_schema(),
                    )
            finally:
                server.close()

        self.assertEqual(request_count, 0)


# ### Bundle and command tests ###
class QwenCeilingHeightBundleTests(unittest.TestCase):
    def test_valid_bundle_builds_gpu_server_command(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            bundle = _create_bundle(Path(raw_directory))
            server = QwenCeilingHeightServer(bundle)
            server._port = 41783
            server._api_key = "private-loopback-key"
            command = server._build_command()

        expected_values = {
            "--model": bundle.language_model.resolve(),
            "--mmproj": bundle.vision_projector.resolve(),
            "--host": QWEN_SERVER_HOST,
            "--port": "41783",
            "--ctx-size": str(QWEN_SERVER_CONTEXT_TOKENS),
            "--n-gpu-layers": "all",
            "--fit": "on",
            "--flash-attn": "on",
            "--parallel": "1",
            "--image-min-tokens": str(QWEN_SERVER_IMAGE_MIN_TOKENS),
            "--image-max-tokens": str(QWEN_SERVER_IMAGE_MAX_TOKENS),
            "--api-key": "private-loopback-key",
        }
        for option, expected in expected_values.items():
            with self.subTest(option=option):
                self.assertIn(option, command)
                self.assertEqual(
                    command[command.index(option) + 1],
                    str(expected),
                )
        self.assertIn("--jinja", command)
        self.assertIn("--no-webui", command)

    def test_valid_bundle_prepends_runtime_paths_to_environment(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            bundle = _create_bundle(Path(raw_directory))
            server = QwenCeilingHeightServer(bundle)
            environment = server._build_environment()

        paths = environment["PATH"].split(os.pathsep)
        self.assertEqual(paths[0], str(bundle.executable.parent.resolve()))
        self.assertEqual(
            paths[1],
            str(bundle.cuda_runtime_directory.resolve()),
        )

    def test_rejects_missing_and_incompatible_bundle_files(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            bundle = _create_bundle(directory)
            missing = QwenCeilingHeightModelBundle(
                directory / "missing-llama-server.exe",
                bundle.language_model,
                bundle.vision_projector,
                bundle.cuda_runtime_directory,
            )
            wrong_model = _write_file(directory / "Qwen3VL-8B-F16.gguf")
            incompatible = QwenCeilingHeightModelBundle(
                bundle.executable,
                wrong_model,
                bundle.vision_projector,
                bundle.cuda_runtime_directory,
            )

            for invalid in (missing, incompatible):
                with (
                    self.subTest(bundle=invalid),
                    self.assertRaises(QwenCeilingHeightError),
                ):
                    QwenCeilingHeightServer(invalid)


# ### Safe failure tests ###
class QwenCeilingHeightSafeFailureTests(unittest.TestCase):
    def test_server_start_hides_process_diagnostics(self) -> None:
        secret = "private-driver-and-model-path"
        with tempfile.TemporaryDirectory() as raw_directory:
            server = QwenCeilingHeightServer(
                _create_bundle(Path(raw_directory))
            )
            with (
                patch(
                    "housemaker.qwen_ceiling_height_estimation.subprocess.Popen",
                    side_effect=OSError(secret),
                ),
                self.assertRaises(QwenCeilingHeightError) as caught,
            ):
                server.start()

        self.assertNotIn(secret, str(caught.exception))
        self.assertIn("could not be started", str(caught.exception))

    def test_request_failure_hides_network_diagnostics(self) -> None:
        secret = "private-network-diagnostic"

        def fail_request(*_args: object, **_kwargs: object) -> object:
            raise OSError(secret)

        with tempfile.TemporaryDirectory() as raw_directory:
            server = QwenCeilingHeightServer(
                _create_bundle(Path(raw_directory)),
                opener=fail_request,
            )
            with self.assertRaises(QwenCeilingHeightError) as caught:
                server._run_request_cancellably(
                    Request("http://127.0.0.1:1/v1/chat/completions"),
                    None,
                )

        self.assertNotIn(secret, str(caught.exception))
        self.assertIn("did not return a usable", str(caught.exception))

    def test_unexpected_startup_exception_cleans_all_server_resources(
        self,
    ) -> None:
        secret = "private-unexpected-startup-detail"
        process = _Process()
        captured: dict[str, object] = {}
        with tempfile.TemporaryDirectory() as raw_directory:
            server = QwenCeilingHeightServer(
                _create_bundle(Path(raw_directory))
            )

            def fail_while_waiting() -> None:
                temporary = server._temporary_directory
                assert temporary is not None
                captured["temporary_path"] = Path(temporary.name)
                captured["log_file"] = server._log_file
                raise KeyError(secret)

            raised: BaseException | None = None
            with (
                patch(
                    "housemaker.qwen_ceiling_height_estimation.subprocess.Popen",
                    return_value=process,
                ),
                patch.object(
                    server,
                    "_wait_until_ready",
                    side_effect=fail_while_waiting,
                ),
            ):
                try:
                    server.start()
                except BaseException as error:  # noqa: BLE001 - regression capture.
                    raised = error

            try:
                self.assertIsInstance(raised, QwenCeilingHeightError)
                assert raised is not None
                self.assertNotIn(secret, str(raised))
                self.assertTrue(process.terminated)
                self.assertFalse(process.killed)
                self.assertIsNone(server._process)
                self.assertIsNone(server._temporary_directory)
                self.assertIsNone(server._log_file)
                self.assertIsNone(server._port)
                self.assertEqual(server._api_key, "")
                self.assertFalse(server._owns_inference_lock)
                temporary_path = captured["temporary_path"]
                self.assertIsInstance(temporary_path, Path)
                assert isinstance(temporary_path, Path)
                self.assertFalse(temporary_path.exists())
                log_file = captured["log_file"]
                self.assertTrue(log_file.closed)  # type: ignore[attr-defined]
            finally:
                server.close()


# ### Session lifecycle tests ###
class QwenCeilingHeightSessionTests(unittest.TestCase):
    def test_one_local_server_session_is_reused_for_multiple_frames(self) -> None:
        sessions: list[object] = []

        class FakeServer:
            def __init__(self, **_kwargs: object) -> None:
                self.enter_count = 0
                self.exit_count = 0
                self.estimate_count = 0
                sessions.append(self)

            def __enter__(self) -> FakeServer:  # noqa: PYI034
                self.enter_count += 1
                return self

            def __exit__(self, *_args: object) -> None:
                self.exit_count += 1

            def estimate(self, *_args: object, **_kwargs: object) -> object:
                self.estimate_count += 1
                return {
                    "status": "estimated",
                    "estimate_m": 2.65,
                    "minimum_m": 2.4,
                    "maximum_m": 2.9,
                    "confidence": "medium",
                    "basis": "Visible architectural scale cues agree.",
                }

        with (
            patch(
                "housemaker.ceiling_height_estimation.QwenCeilingHeightServer",
                FakeServer,
            ),
            open_ceiling_height_estimator(
                model=LOCAL_QWEN_CEILING_HEIGHT_MODEL,
                api_key="",
            ) as estimator,
        ):
            first = estimator(np.zeros((8, 8, 3), dtype=np.uint8))
            second = estimator(np.ones((8, 8, 3), dtype=np.uint8))

        self.assertEqual(len(sessions), 1)
        session = sessions[0]
        self.assertEqual(session.enter_count, 1)  # type: ignore[attr-defined]
        self.assertEqual(session.exit_count, 1)  # type: ignore[attr-defined]
        self.assertEqual(session.estimate_count, 2)  # type: ignore[attr-defined]
        self.assertEqual(first.estimate_m, 2.6)
        self.assertEqual(second.estimate_m, 2.6)


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
