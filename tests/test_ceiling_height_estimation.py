# ### Imports ###
from __future__ import annotations

import json
import unittest

import numpy as np

from housemaker import ceiling_height_estimation as estimation
from housemaker.ceiling_height_estimation import (
    CEILING_HEIGHT_INFERENCE_MODEL,
    CeilingHeightEstimationCancelled,
    CeilingHeightEstimationError,
    build_ceiling_height_request_body,
    format_ceiling_height_estimate,
    infer_ceiling_height,
)


# ### Fixtures ###
class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._was_read = False

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, _size: int = -1) -> bytes:
        if self._was_read:
            return b""
        self._was_read = True
        return self._payload


def _response_payload(result: dict[str, object]) -> bytes:
    return json.dumps(
        {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps(result),
                        }
                    ],
                }
            ]
        }
    ).encode("utf-8")


def _estimated_result() -> dict[str, object]:
    return {
        "status": "estimated",
        "estimate_m": 2.6,
        "minimum_m": 2.35,
        "maximum_m": 2.85,
        "confidence": "medium",
        "basis": "Visible door and floor-to-ceiling boundaries agree.",
    }


# ### Request tests ###
class CeilingHeightRequestTests(unittest.TestCase):
    def test_request_uses_vision_input_and_strict_structured_output(self) -> None:
        body = build_ceiling_height_request_body(
            "data:image/jpeg;base64,YWJj"
        )

        self.assertEqual(body["model"], CEILING_HEIGHT_INFERENCE_MODEL)
        self.assertFalse(body["store"])
        content = body["input"][0]["content"]
        self.assertEqual(content[0]["type"], "input_text")
        self.assertIn("finished-floor", content[0]["text"])
        self.assertEqual(
            content[1],
            {
                "type": "input_image",
                "image_url": "data:image/jpeg;base64,YWJj",
                "detail": "high",
            },
        )
        output_format = body["text"]["format"]
        self.assertEqual(output_format["type"], "json_schema")
        self.assertTrue(output_format["strict"])
        self.assertFalse(
            output_format["schema"]["additionalProperties"]
        )

    def test_inference_encodes_frame_and_validates_estimate(self) -> None:
        captured: list[tuple[object, dict[str, object]]] = []

        def fake_open(request, **kwargs):
            captured.append((request, kwargs))
            return _FakeResponse(_response_payload(_estimated_result()))

        frame = np.full((24, 32, 3), (15, 90, 180), dtype=np.uint8)
        result = infer_ceiling_height(
            frame,
            api_key="sk-test-secret",
            opener=fake_open,
        )

        self.assertTrue(result.has_estimate)
        self.assertEqual(result.estimate_m, 2.6)
        self.assertEqual(result.minimum_m, 2.3)
        self.assertEqual(result.maximum_m, 2.9)
        request, kwargs = captured[0]
        request_body = json.loads(request.data.decode("utf-8"))
        image_input = request_body["input"][0]["content"][1]
        self.assertTrue(
            image_input["image_url"].startswith(
                "data:image/jpeg;base64,"
            )
        )
        self.assertEqual(
            request.get_header("Authorization"),
            "Bearer sk-test-secret",
        )
        self.assertEqual(
            kwargs["timeout"],
            estimation.OPENAI_NETWORK_TIMEOUT_SECONDS,
        )

    def test_insufficient_evidence_keeps_metric_values_empty(self) -> None:
        result = infer_ceiling_height(
            np.zeros((8, 8, 3), dtype=np.uint8),
            api_key="sk-test",
            opener=lambda *_args, **_kwargs: _FakeResponse(
                _response_payload(
                    {
                        "status": "insufficient_evidence",
                        "estimate_m": None,
                        "minimum_m": None,
                        "maximum_m": None,
                        "confidence": "low",
                        "basis": "The ceiling boundary is not visible.",
                    }
                )
            ),
        )

        self.assertFalse(result.has_estimate)
        self.assertIn(
            "Could not estimate reliably",
            format_ceiling_height_estimate(result),
        )

    def test_implausible_or_inconsistent_ranges_are_rejected(self) -> None:
        invalid = _estimated_result()
        invalid["minimum_m"] = 3.0
        invalid["estimate_m"] = 2.6

        with self.assertRaisesRegex(
            CeilingHeightEstimationError,
            "implausible",
        ):
            infer_ceiling_height(
                np.zeros((8, 8, 3), dtype=np.uint8),
                api_key="sk-test",
                opener=lambda *_args, **_kwargs: _FakeResponse(
                    _response_payload(invalid)
                ),
            )

    def test_cancellation_is_checked_before_sending_a_request(self) -> None:
        with self.assertRaises(CeilingHeightEstimationCancelled):
            infer_ceiling_height(
                np.zeros((8, 8, 3), dtype=np.uint8),
                api_key="sk-test",
                cancellation_check=lambda: True,
                opener=lambda *_args, **_kwargs: self.fail(
                    "A cancelled request must not be sent."
                ),
            )

    def test_incomplete_and_refused_responses_are_rejected(self) -> None:
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        incomplete = json.dumps(
            {
                "status": "incomplete",
                "output_text": json.dumps(_estimated_result()),
            }
        ).encode("utf-8")
        refusal = json.dumps(
            {
                "status": "completed",
                "output_text": json.dumps(_estimated_result()),
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "refusal",
                                "refusal": "Cannot analyze this image.",
                            }
                        ],
                    }
                ],
            }
        ).encode("utf-8")

        for response, message in (
            (incomplete, "could not finish"),
            (refusal, "declined"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(
                CeilingHeightEstimationError,
                message,
            ):
                infer_ceiling_height(
                    frame,
                    api_key="sk-test",
                    opener=lambda *_args, response=response, **_kwargs: (
                        _FakeResponse(response)
                    ),
                )


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
