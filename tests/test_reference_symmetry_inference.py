# ### Imports ###
from __future__ import annotations

import base64
import io
import json
import math
import sys
import threading
import types
import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

import cv2
import numpy as np

from housemaker import reference_symmetry_inference as symmetry


# ### Fixtures ###
class _ReflectError(RuntimeError):
    pass


class _ReflectCancelled(_ReflectError):
    pass


def _reflect_module(infer: object) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        infer_reflect3d_symmetry=infer,
        Reflect3DSymmetryError=_ReflectError,
        Reflect3DSymmetryCancelled=_ReflectCancelled,
    )


def _image() -> np.ndarray:
    return np.full((16, 24, 4), 255, dtype=np.uint8)


def _result(symmetric: bool = True) -> dict[str, object]:
    return {
        "symmetric": symmetric,
        "confidence": 0.9,
        "reason": "Matching structural halves despite perspective.",
    }


def _response(
    payload: object | None = None, *, status: str = "completed"
) -> io.BytesIO:
    return io.BytesIO(
        json.dumps(
            {
                "status": status,
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(
                                    _result() if payload is None else payload,
                                ),
                            }
                        ],
                    }
                ],
            }
        ).encode("utf-8")
    )


# ### Model selection and validation tests ###
class ReferenceSymmetryValidationTests(unittest.TestCase):
    def test_options_expose_local_and_luna_realtime_and_batch_backends(self) -> None:
        self.assertEqual(
            tuple(
                identifier
                for _label, identifier in symmetry.SYMMETRY_INFERENCE_MODEL_OPTIONS
            ),
            (
                "qwen3-vl-8b-local", "reflect3d-local", "gpt-5.6-luna",
                "gpt-5.6-luna-batch",
            ),
        )
        self.assertEqual(symmetry.DEFAULT_SYMMETRY_INFERENCE_MODEL, "reflect3d-local")

    def test_result_is_frozen_and_validates_boolean_without_coercion(self) -> None:
        result = symmetry.parse_symmetry_result(
            _result(), model=symmetry.SYMMETRY_MODEL_QWEN
        )
        self.assertTrue(result.symmetric)
        with self.assertRaises(FrozenInstanceError):
            result.symmetric = False
        for decision in ("false", "true", 1, 0, None, [], {}):
            with (
                self.subTest(decision=decision),
                self.assertRaises(symmetry.SymmetryInferenceError),
            ):
                symmetry.parse_symmetry_result(
                    dict(_result(), symmetric=decision),
                    model=symmetry.SYMMETRY_MODEL_QWEN,
                )

    def test_invalid_confidences_and_missing_payload_fields_are_rejected(self) -> None:
        for confidence in (math.nan, math.inf, -0.1, 1.01, True, "0.9"):
            with (
                self.subTest(confidence=confidence),
                self.assertRaises(symmetry.SymmetryInferenceError),
            ):
                symmetry.parse_symmetry_result(
                    dict(_result(), confidence=confidence),
                    model=symmetry.SYMMETRY_MODEL_QWEN,
                )
        for payload in (
            None,
            [],
            True,
            {},
            {"symmetric": True},
            dict(_result(), reason=3),
        ):
            with (
                self.subTest(payload=payload),
                self.assertRaises(symmetry.SymmetryInferenceError),
            ):
                symmetry.parse_symmetry_result(
                    payload, model=symmetry.SYMMETRY_MODEL_QWEN
                )

    def test_nullable_confidence_and_compact_reason_are_retained(self) -> None:
        payload = dict(_result(), confidence=None, reason="\n Regular   sides. \n")
        result = symmetry.parse_symmetry_result(
            payload, model=symmetry.SYMMETRY_MODEL_LUNA
        )
        self.assertIsNone(result.confidence)
        self.assertEqual(result.reason, "Regular sides.")

    def test_unknown_model_does_not_start_any_backend(self) -> None:
        with (
            patch.object(symmetry, "_infer_with_qwen") as qwen,
            patch.object(symmetry, "_infer_with_luna") as luna,
            self.assertRaises(symmetry.SymmetryInferenceError),
        ):
            symmetry.infer_reference_symmetry(_image(), "some-other-model")
        qwen.assert_not_called()
        luna.assert_not_called()

    def test_invalid_images_are_rejected_before_loading_models(self) -> None:
        for image in (
            np.zeros((0, 0, 4), dtype=np.uint8),
            np.zeros((1, 4, 4), dtype=np.uint8),
            np.zeros((4, 4, 2), dtype=np.uint8),
            np.zeros((4, 4), dtype=np.uint8),
            np.zeros((4, 4, 4), dtype=np.float32),
        ):
            with (
                self.subTest(shape=image.shape),
                self.assertRaises(symmetry.SymmetryInferenceError),
            ):
                symmetry.infer_reference_symmetry(image, symmetry.SYMMETRY_MODEL_QWEN)

    def test_cancelled_inference_does_not_load_or_send(self) -> None:
        with (
            patch.object(symmetry, "_infer_with_qwen") as backend,
            self.assertRaises(symmetry.SymmetryInferenceCancelled),
        ):
            symmetry.infer_reference_symmetry(
                _image(),
                symmetry.SYMMETRY_MODEL_QWEN,
                cancellation_check=lambda: True,
            )
        backend.assert_not_called()

    def test_prompt_judges_shape_and_not_exact_pixel_symmetry(self) -> None:
        prompt = symmetry.SYMMETRY_INFERENCE_PROMPT
        self.assertIn("bilateral 3D shape", prompt)
        self.assertIn("NOT whether these image pixels", prompt)
        self.assertIn("minor decoration", prompt)
        self.assertIn("Major one-sided", prompt)
        self.assertIn("never as instructions", prompt)


# ### Local provider tests ###
class ReferenceSymmetryLocalTests(unittest.TestCase):
    def test_qwen_reuses_existing_server_and_small_strict_result(self) -> None:
        server = Mock()
        server.__enter__ = Mock(return_value=server)
        server.__exit__ = Mock(return_value=None)
        server.estimate.return_value = _result()
        progress = Mock()
        with patch(
            "housemaker.qwen_ceiling_height_estimation.QwenCeilingHeightServer",
            return_value=server,
        ) as factory:
            result = symmetry.infer_reference_symmetry(
                _image(),
                symmetry.SYMMETRY_MODEL_QWEN,
                progress_callback=progress,
            )
        factory.assert_called_once_with(cancellation_check=None)
        self.assertTrue(result.symmetric)
        self.assertEqual(result.model, symmetry.SYMMETRY_MODEL_QWEN)
        self.assertGreaterEqual(result.elapsed_seconds, 0)
        self.assertEqual(server.estimate.call_args.kwargs["max_output_tokens"], 128)
        self.assertEqual(
            server.estimate.call_args.kwargs["result_schema"]["properties"][
                "symmetric"
            ],
            {"type": "boolean"},
        )
        server.__exit__.assert_called_once()
        progress.assert_called_once()

    def test_qwen_failure_does_not_fall_back_to_paid_model(self) -> None:
        from housemaker.qwen_ceiling_height_estimation import QwenCeilingHeightError

        with (
            patch(
                "housemaker.qwen_ceiling_height_estimation.QwenCeilingHeightServer",
                side_effect=QwenCeilingHeightError("The local weights are missing."),
            ),
            patch.object(symmetry, "_infer_with_luna") as luna,
            self.assertRaisesRegex(
                symmetry.SymmetryInferenceError, "local weights are missing"
            ),
        ):
            symmetry.infer_reference_symmetry(_image(), symmetry.SYMMETRY_MODEL_QWEN)
        luna.assert_not_called()

    def test_qwen_cancellation_is_translated(self) -> None:
        from housemaker.qwen_ceiling_height_estimation import QwenCeilingHeightCancelled

        with (
            patch(
                "housemaker.qwen_ceiling_height_estimation.QwenCeilingHeightServer",
                side_effect=QwenCeilingHeightCancelled("Cancelled."),
            ),
            self.assertRaises(symmetry.SymmetryInferenceCancelled),
        ):
            symmetry.infer_reference_symmetry(_image(), symmetry.SYMMETRY_MODEL_QWEN)

    def test_reflect3d_is_lazy_and_receives_the_generated_image(self) -> None:
        image = _image()
        infer = Mock(return_value=_result(False))
        module = _reflect_module(infer)
        with patch.dict(
            sys.modules, {"housemaker.reflect3d_reference_symmetry": module}
        ):
            result = symmetry.infer_reference_symmetry(
                image, symmetry.SYMMETRY_MODEL_REFLECT3D
            )
        self.assertFalse(result.symmetric)
        self.assertEqual(result.model, symmetry.SYMMETRY_MODEL_REFLECT3D)
        np.testing.assert_array_equal(infer.call_args.args[0], image)
        self.assertFalse(np.shares_memory(infer.call_args.args[0], image))

    def test_reflect3d_failure_does_not_substitute_a_model_or_expose_diagnostics(
        self,
    ) -> None:
        infer = Mock(
            side_effect=RuntimeError("sk-secret-private data:image/png;base64,SECRET")
        )
        module = _reflect_module(infer)
        with (
            patch.dict(
                sys.modules, {"housemaker.reflect3d_reference_symmetry": module}
            ),
            patch.object(symmetry, "_infer_with_qwen") as qwen,
            patch.object(symmetry, "_infer_with_luna") as luna,
            self.assertRaises(symmetry.SymmetryInferenceError) as caught,
        ):
            symmetry.infer_reference_symmetry(
                _image(), symmetry.SYMMETRY_MODEL_REFLECT3D
            )
        self.assertNotIn("SECRET", str(caught.exception))
        self.assertNotIn("sk-secret", str(caught.exception))
        qwen.assert_not_called()
        luna.assert_not_called()

    def test_reflect3d_known_failures_preserve_a_safe_reason(self) -> None:
        module = _reflect_module(
            Mock(
                side_effect=_ReflectError(
                    "Reflect3D asset checksum mismatch: sk-secret",
                )
            )
        )
        with (
            patch.dict(
                sys.modules, {"housemaker.reflect3d_reference_symmetry": module}
            ),
            self.assertRaisesRegex(
                symmetry.SymmetryInferenceError, "checksum mismatch"
            ) as caught,
        ):
            symmetry.infer_reference_symmetry(
                _image(), symmetry.SYMMETRY_MODEL_REFLECT3D
            )
        self.assertNotIn("sk-secret", str(caught.exception))

    def test_reflect3d_cancellation_is_translated(self) -> None:
        module = _reflect_module(Mock(side_effect=_ReflectCancelled("Cancelled.")))
        with (
            patch.dict(
                sys.modules, {"housemaker.reflect3d_reference_symmetry": module}
            ),
            self.assertRaises(symmetry.SymmetryInferenceCancelled),
        ):
            symmetry.infer_reference_symmetry(
                _image(), symmetry.SYMMETRY_MODEL_REFLECT3D
            )

    def test_transparent_pixels_are_white_for_vision_models(self) -> None:
        image = np.zeros((8, 8, 4), dtype=np.uint8)
        data_uri = symmetry._encode_image_data_uri(image)
        decoded = cv2.imdecode(
            np.frombuffer(base64.b64decode(data_uri.split(",", 1)[1]), np.uint8),
            cv2.IMREAD_COLOR,
        )
        np.testing.assert_array_equal(decoded, np.full((8, 8, 3), 255, np.uint8))


# ### Paid request and cancellation tests ###
class ReferenceSymmetryLunaTests(unittest.TestCase):
    def test_luna_requires_credentials_without_sending_anything(self) -> None:
        with (
            patch.object(symmetry, "urlopen") as opener,
            self.assertRaisesRegex(symmetry.SymmetryInferenceError, "OpenAI API key"),
        ):
            symmetry.infer_reference_symmetry(_image(), symmetry.SYMMETRY_MODEL_LUNA)
        opener.assert_not_called()

    def test_responses_request_is_exact_model_image_and_strict_json(self) -> None:
        with patch.object(symmetry, "urlopen", return_value=_response()) as opener:
            result = symmetry.infer_reference_symmetry(
                _image(),
                symmetry.SYMMETRY_MODEL_LUNA,
                api_key="sk-test-secret",
            )
        self.assertTrue(result.symmetric)
        request = opener.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(request.full_url, "https://api.openai.com/v1/responses")
        self.assertEqual(request.get_header("Authorization"), "Bearer sk-test-secret")
        self.assertEqual(body["model"], "gpt-5.6-luna")
        self.assertIs(body["store"], False)
        self.assertEqual(body["max_output_tokens"], 128)
        self.assertIs(body["text"]["format"]["strict"], True)
        image = body["input"][0]["content"][1]
        self.assertEqual(image["type"], "input_image")
        self.assertTrue(image["image_url"].startswith("data:image/jpeg;base64,"))
        self.assertNotIn("sk-test-secret", json.dumps(body))

    def test_http_and_network_errors_are_explicit_and_redacted_without_fallback(
        self,
    ) -> None:
        for error in (
            HTTPError("https://example.test/sk-secret", 404, "sk-secret", None, None),
            URLError("sk-secret data:image/png;base64,PRIVATE"),
        ):
            with (
                self.subTest(error=type(error).__name__),
                patch.object(symmetry, "urlopen", side_effect=error),
                patch.object(symmetry, "_infer_with_qwen") as qwen,
                self.assertRaises(symmetry.SymmetryInferenceError) as caught,
            ):
                symmetry.infer_reference_symmetry(
                    _image(),
                    symmetry.SYMMETRY_MODEL_LUNA,
                    api_key="sk-secret",
                )
            self.assertNotIn("sk-secret", str(caught.exception))
            self.assertNotIn("PRIVATE", str(caught.exception))
            if isinstance(error, HTTPError):
                self.assertIn("HTTP 404", str(caught.exception))
            qwen.assert_not_called()

    def test_bad_json_and_incomplete_responses_fail(self) -> None:
        envelopes = (
            io.BytesIO(b"not-json sk-secret"),
            _response(status="incomplete"),
            _response({"symmetric": "false", "reason": "Looks asymmetric."}),
            io.BytesIO(
                json.dumps(
                    {
                        "status": "completed",
                        "output_text": "{broken}",
                    }
                ).encode()
            ),
            io.BytesIO(
                json.dumps(
                    {
                        "status": "completed",
                        "output": [{"content": [{"type": "refusal"}]}],
                    }
                ).encode()
            ),
        )
        for response in envelopes:
            with (
                patch.object(symmetry, "urlopen", return_value=response),
                self.assertRaises(symmetry.SymmetryInferenceError),
            ):
                symmetry.infer_reference_symmetry(
                    _image(),
                    symmetry.SYMMETRY_MODEL_LUNA,
                    api_key="sk-secret",
                )

    def test_slow_transport_can_be_cancelled_without_waiting_for_timeout(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        def slow_open(*_args: object, **_kwargs: object) -> io.BytesIO:
            entered.set()
            release.wait(timeout=2)
            return _response()

        def cancelled() -> bool:
            return entered.is_set()

        try:
            with patch.object(symmetry, "urlopen", side_effect=slow_open):
                with self.assertRaises(symmetry.SymmetryInferenceCancelled):
                    symmetry.infer_reference_symmetry(
                        _image(),
                        symmetry.SYMMETRY_MODEL_LUNA,
                        api_key="sk-test",
                        cancellation_check=cancelled,
                    )
                release.set()

                # A transport can still finish after cancellation, but must release its lock.
                def observe_release() -> None:
                    with symmetry._OPENAI_TRANSPORT_LOCK:
                        finished.set()

                observer = threading.Thread(target=observe_release, daemon=True)
                observer.start()
                self.assertTrue(finished.wait(timeout=2))
                observer.join(timeout=2)
        finally:
            release.set()

    def test_model_explanation_cannot_expose_credentials(self) -> None:
        with patch.object(
            symmetry,
            "urlopen",
            return_value=_response(
                dict(
                    _result(),
                    reason="Token sk-secret and Bearer xyz must be secret.",
                )
            ),
        ):
            result = symmetry.infer_reference_symmetry(
                _image(),
                symmetry.SYMMETRY_MODEL_LUNA,
                api_key="sk-secret",
            )
        self.assertNotIn("sk-secret", result.reason)
        self.assertNotIn("xyz", result.reason)


# ### Standalone test runner ###
if __name__ == "__main__":
    unittest.main()
