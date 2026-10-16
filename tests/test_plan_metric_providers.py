# ### Imports ###
from __future__ import annotations

import io
import json
import unittest
from urllib.error import HTTPError

from housemaker.plan_correction_models import (
    PLAN_CORRECTION_MODEL_GPT_5_6_LUNA,
    PLAN_CORRECTION_MODEL_GPT_5_6_TERRA,
    PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
    PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1,
)
from housemaker.plan_metric_batch import (
    EncodedPlanMeasurementPair,
    build_multi_plan_measurement_result_schema,
)
from housemaker.plan_metric_calibration import (
    PlanMetricCalibrationCancelled,
    PlanMetricCalibrationInferenceError,
)
from housemaker.plan_metric_providers import (
    LOCAL_QWEN_CONTEXT_TOKENS,
    LOCAL_QWEN_MAX_MEASUREMENTS,
    LOCAL_QWEN_MAX_OUTPUT_TOKENS,
    LocalQwenMultiPlanMeasurementAnalyzer,
    LocalQwenPlanMeasurementAnalyzer,
    OpenAIMultiPlanMeasurementAnalyzer,
    OpenAIPlanMeasurementAnalyzer,
    build_local_qwen_plan_measurement_prompt,
    cap_local_qwen_measurement_schema,
    create_plan_measurement_analyzer,
    plan_measurement_model_for_correction_model,
)
from housemaker.plan_metric_views import PlanMeasurementDetailImage

# ### Fixtures ###
ORIGINAL_URI = "data:image/png;base64,b3JpZ2luYWw="
REFERENCE_URI = "data:image/png;base64,cmVmZXJlbmNl"
CORRECTED_URI = "data:image/png;base64,Y29ycmVjdGVk"
DETAIL = PlanMeasurementDetailImage(
    "DETAIL of ORIGINAL PLAN (RAW); full-source normalized bounds "
    "[0.0, 0.0, 0.55, 0.55]; NEVER crop-local coordinates.",
    "data:image/png;base64,ZGV0YWls", "original", (0.0, 0.0, 0.55, 0.55),
)
RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "measurements": {
            "type": "array",
            "maxItems": 64,
            "items": {"type": "object"},
        }
    },
    "required": ["measurements"],
    "additionalProperties": False,
}
MEASUREMENT_RESULT = {
    "measurements": [
        {
            "label": "610",
            "value_meters": 6.1,
            "start_normalized": [0.1, 0.2],
            "end_normalized": [0.8, 0.2],
            "corrected_start_normalized": [0.12, 0.18],
            "corrected_end_normalized": [0.82, 0.18],
            "confidence": "high",
        }
    ]
}


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._was_read = False

    def __enter__(self) -> _FakeResponse:  # noqa: PYI034
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, _size: int = -1) -> bytes:
        if self._was_read:
            return b""
        self._was_read = True
        return self._payload


def _openai_response(result: object) -> bytes:
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
            ],
        }
    ).encode("utf-8")


class _FakeQwenServer:
    def __init__(self, calls: list[tuple[str, object]]) -> None:
        self._calls = calls

    def __enter__(self) -> _FakeQwenServer:  # noqa: PYI034
        self._calls.append(("enter", None))
        return self

    def __exit__(self, *_args: object) -> None:
        self._calls.append(("exit", None))

    def estimate(self, image_data_uri: str, **kwargs: object) -> object:
        self._calls.append(
            (
                "estimate",
                {
                    "image_data_uri": image_data_uri,
                    **kwargs,
                },
            )
        )
        return MEASUREMENT_RESULT


# ### Provider selection tests ###
class PlanMeasurementProviderSelectionTests(unittest.TestCase):
    def test_correction_models_route_to_supported_measurement_models(self) -> None:
        self.assertEqual(
            plan_measurement_model_for_correction_model(
                PLAN_CORRECTION_MODEL_GPT_IMAGE_2
            ),
            PLAN_CORRECTION_MODEL_GPT_5_6_LUNA,
        )
        self.assertEqual(
            plan_measurement_model_for_correction_model(
                PLAN_CORRECTION_MODEL_GPT_5_6_LUNA
            ),
            PLAN_CORRECTION_MODEL_GPT_5_6_LUNA,
        )
        self.assertEqual(
            plan_measurement_model_for_correction_model(
                PLAN_CORRECTION_MODEL_GPT_5_6_TERRA
            ),
            PLAN_CORRECTION_MODEL_GPT_5_6_TERRA,
        )
        self.assertEqual(
            plan_measurement_model_for_correction_model(
                PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1
            ),
            PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1,
        )

    def test_factory_uses_luna_for_gpt_image_correction(self) -> None:
        analyzer = create_plan_measurement_analyzer(
            PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
            api_key="sk-test",
        )

        self.assertIsInstance(analyzer, OpenAIPlanMeasurementAnalyzer)
        self.assertEqual(analyzer._model, PLAN_CORRECTION_MODEL_GPT_5_6_LUNA)

    def test_qwen_factory_is_lazy_until_analysis(self) -> None:
        calls: list[tuple[str, object]] = []

        def server_factory(**kwargs: object) -> _FakeQwenServer:
            calls.append(("factory", kwargs))
            return _FakeQwenServer(calls)

        analyzer = create_plan_measurement_analyzer(
            PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1,
            qwen_server_factory=server_factory,
        )

        self.assertIsInstance(analyzer, LocalQwenPlanMeasurementAnalyzer)
        self.assertEqual(calls, [])

        result = analyzer(
            ORIGINAL_URI,
            CORRECTED_URI,
            prompt="Read plan measurements.",
            result_schema=RESULT_SCHEMA,
            cancellation_check=None,
        )

        self.assertEqual(result, MEASUREMENT_RESULT)
        self.assertEqual([name for name, _value in calls], [
            "factory",
            "enter",
            "estimate",
            "exit",
        ])
        estimate = calls[2][1]
        assert isinstance(estimate, dict)
        self.assertEqual(estimate["image_data_uri"], CORRECTED_URI)
        self.assertEqual(
            estimate["previous_image_data_uri"],
            ORIGINAL_URI,
        )
        self.assertNotIn("plan_image_data_uri", estimate)
        self.assertEqual(
            estimate["result_schema"]["properties"]["measurements"][
                "maxItems"
            ],
            LOCAL_QWEN_MAX_MEASUREMENTS,
        )

    def test_unsupported_correction_model_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            PlanMetricCalibrationInferenceError,
            "cannot transcribe",
        ):
            create_plan_measurement_analyzer("unknown-model")


# ### OpenAI analyzer tests ###
class OpenAIPlanMeasurementAnalyzerTests(unittest.TestCase):
    def test_detail_crop_follows_labeled_overviews(self) -> None:
        captured = []

        def fake_open(request, **_kwargs):
            captured.append(request)
            return _FakeResponse(_openai_response(MEASUREMENT_RESULT))

        analyzer = OpenAIPlanMeasurementAnalyzer("test", api_key="test", opener=fake_open)
        analyzer(
            ORIGINAL_URI, CORRECTED_URI,
            measurement_reference_image_data_uri=REFERENCE_URI,
            detail_images=(DETAIL,), prompt="Read dimensions.",
            result_schema=RESULT_SCHEMA, cancellation_check=None,
        )
        content = json.loads(captured[0].data)["input"][0]["content"]
        self.assertEqual(
            [item["image_url"] for item in content if item["type"] == "input_image"],
            [ORIGINAL_URI, REFERENCE_URI, CORRECTED_URI, DETAIL.image_data_uri],
        )
        self.assertEqual(content[-2]["text"], DETAIL.label)

    def test_strict_response_request_is_parsed_as_json(self) -> None:
        captured: list[tuple[object, dict[str, object]]] = []

        def fake_open(request, **kwargs):
            captured.append((request, kwargs))
            return _FakeResponse(_openai_response(MEASUREMENT_RESULT))

        analyzer = create_plan_measurement_analyzer(
            PLAN_CORRECTION_MODEL_GPT_5_6_TERRA,
            api_key="sk-test-secret",
            opener=fake_open,
        )
        result = analyzer(
            ORIGINAL_URI,
            CORRECTED_URI,
            prompt="Use these exact instructions.",
            result_schema=RESULT_SCHEMA,
            cancellation_check=None,
        )

        self.assertEqual(result, MEASUREMENT_RESULT)
        request, kwargs = captured[0]
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["model"], PLAN_CORRECTION_MODEL_GPT_5_6_TERRA)
        self.assertFalse(body["store"])
        self.assertEqual(body["reasoning"], {"effort": "none"})
        self.assertEqual(
            body["input"][0]["content"][0]["text"],
            "Use these exact instructions.",
        )
        self.assertEqual(
            [
                item["image_url"]
                for item in body["input"][0]["content"]
                if item["type"] == "input_image"
            ],
            [ORIGINAL_URI, CORRECTED_URI],
        )
        output_format = body["text"]["format"]
        self.assertEqual(output_format["type"], "json_schema")
        self.assertTrue(output_format["strict"])
        self.assertEqual(output_format["schema"], RESULT_SCHEMA)
        self.assertEqual(
            request.get_header("Authorization"),
            "Bearer sk-test-secret",
        )
        self.assertEqual(kwargs["timeout"], 120.0)

    def test_optional_reference_keeps_raw_and_final_in_request(self) -> None:
        captured: list[object] = []

        def fake_open(request, **_kwargs):
            captured.append(request)
            return _FakeResponse(_openai_response(MEASUREMENT_RESULT))

        analyzer = create_plan_measurement_analyzer(
            PLAN_CORRECTION_MODEL_GPT_5_6_TERRA,
            api_key="sk-test-secret",
            opener=fake_open,
        )
        analyzer(
            ORIGINAL_URI,
            CORRECTED_URI,
            measurement_reference_image_data_uri=REFERENCE_URI,
            prompt="Use all supplied plan images safely.",
            result_schema=RESULT_SCHEMA,
            cancellation_check=None,
        )

        body = json.loads(captured[0].data.decode("utf-8"))
        content = body["input"][0]["content"]
        self.assertEqual(
            [item["image_url"] for item in content if item["type"] == "input_image"],
            [ORIGINAL_URI, REFERENCE_URI, CORRECTED_URI],
        )
        labels = [item["text"] for item in content if item["type"] == "input_text"]
        self.assertIn("ORIGINAL PLAN (RAW):", labels)
        self.assertIn("RECTIFIED ANNOTATED PLAN:", labels)
        self.assertIn("CORRECTED PLAN (FINAL):", labels)

    def test_http_failure_never_exposes_credentials_or_provider_body(self) -> None:
        api_key = "sk-super-secret-credential"
        provider_secret = "data:image/png;base64,private-image"

        def reject_request(*_args: object, **_kwargs: object) -> object:
            payload = json.dumps(
                {"error": {"message": f"{api_key} {provider_secret}"}}
            ).encode("utf-8")
            raise HTTPError(
                "https://api.openai.com/v1/responses",
                400,
                "rejected",
                {},
                io.BytesIO(payload),
            )

        analyzer = create_plan_measurement_analyzer(
            PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
            api_key=api_key,
            opener=reject_request,
        )
        with self.assertRaises(PlanMetricCalibrationInferenceError) as caught:
            analyzer(
                ORIGINAL_URI,
                CORRECTED_URI,
                prompt="Read dimensions.",
                result_schema=RESULT_SCHEMA,
                cancellation_check=None,
            )

        message = str(caught.exception)
        self.assertIn("rejected (400)", message)
        self.assertNotIn(api_key, message)
        self.assertNotIn(provider_secret, message)

    def test_preflight_cancellation_does_not_start_transport(self) -> None:
        opened = False

        def fake_open(*_args: object, **_kwargs: object) -> object:
            nonlocal opened
            opened = True
            return _FakeResponse(_openai_response(MEASUREMENT_RESULT))

        analyzer = create_plan_measurement_analyzer(
            PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
            api_key="sk-test",
            opener=fake_open,
        )
        with self.assertRaises(PlanMetricCalibrationCancelled):
            analyzer(
                ORIGINAL_URI,
                CORRECTED_URI,
                prompt="Read dimensions.",
                result_schema=RESULT_SCHEMA,
                cancellation_check=lambda: True,
            )

        self.assertFalse(opened)


# ### Local Qwen request tests ###
class LocalQwenPlanMeasurementTests(unittest.TestCase):
    def test_qwen_receives_detail_labels_and_sufficient_context_and_output_budget(self):
        calls = []

        def server_factory(**kwargs):
            calls.append(("factory", kwargs))
            return _FakeQwenServer(calls)

        analyzer = LocalQwenPlanMeasurementAnalyzer(server_factory=server_factory)
        analyzer(
            ORIGINAL_URI, CORRECTED_URI,
            measurement_reference_image_data_uri=REFERENCE_URI,
            detail_images=(DETAIL,), prompt="Read dimensions.",
            result_schema=RESULT_SCHEMA, cancellation_check=None,
        )

        self.assertEqual(calls[0][1]["context_tokens"], LOCAL_QWEN_CONTEXT_TOKENS)
        request = calls[2][1]
        self.assertEqual(request["additional_images"], ((DETAIL.label, DETAIL.image_data_uri),))
        self.assertEqual(request["max_output_tokens"], LOCAL_QWEN_MAX_OUTPUT_TOKENS)
        self.assertIn("two horizontal and two vertical", request["prompt"])
        self.assertIn("Never invent evidence", request["prompt"])

    def test_prompt_prioritizes_a_bounded_number_of_measurements(self) -> None:
        prompt = build_local_qwen_plan_measurement_prompt(
            "Read all explicit dimensions."
        )

        self.assertIn("MEASUREMENT REFERENCE PLAN", prompt)
        self.assertIn("CORRECTED PLAN", prompt)
        self.assertIn(
            f"no more than {LOCAL_QWEN_MAX_MEASUREMENTS}",
            prompt,
        )
        self.assertIn("strongest independent evidence", prompt)

    def test_three_image_prompt_requires_raw_reference_agreement(self) -> None:
        prompt = build_local_qwen_plan_measurement_prompt(
            "Read all explicit dimensions.",
            has_measurement_reference=True,
        )

        self.assertIn("PLAN REFERENCE is the raw ORIGINAL PLAN", prompt)
        self.assertIn("PREVIOUS is the RECTIFIED ANNOTATED PLAN", prompt)
        self.assertIn("Omit a measurement unless", prompt)

    def test_qwen_uses_plan_previous_and_current_for_three_images(self) -> None:
        calls: list[tuple[str, object]] = []

        def server_factory(**kwargs: object) -> _FakeQwenServer:
            calls.append(("factory", kwargs))
            return _FakeQwenServer(calls)

        analyzer = create_plan_measurement_analyzer(
            PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1,
            qwen_server_factory=server_factory,
        )
        analyzer(
            ORIGINAL_URI,
            CORRECTED_URI,
            measurement_reference_image_data_uri=REFERENCE_URI,
            prompt="Read plan measurements.",
            result_schema=RESULT_SCHEMA,
            cancellation_check=None,
        )

        estimate = calls[2][1]
        self.assertEqual(estimate["image_data_uri"], CORRECTED_URI)
        self.assertEqual(estimate["previous_image_data_uri"], REFERENCE_URI)
        self.assertEqual(estimate["plan_image_data_uri"], ORIGINAL_URI)

    def test_schema_cap_does_not_mutate_caller_schema(self) -> None:
        original = json.loads(json.dumps(RESULT_SCHEMA))

        capped = cap_local_qwen_measurement_schema(original)

        self.assertEqual(
            capped["properties"]["measurements"]["maxItems"],
            LOCAL_QWEN_MAX_MEASUREMENTS,
        )
        self.assertEqual(
            original["properties"]["measurements"]["maxItems"],
            64,
        )


# ### Grouped detail requests ###
class GroupedPlanMeasurementDetailTests(unittest.TestCase):
    def test_remote_request_keeps_details_attached_to_their_plan_id(self):
        requests = []

        def opener(request, **_kwargs):
            requests.append(request)
            return _FakeResponse(_openai_response({"plans": {"L2": MEASUREMENT_RESULT}}))

        analyzer = OpenAIMultiPlanMeasurementAnalyzer("test", api_key="test", opener=opener)
        pair = EncodedPlanMeasurementPair(
            "L2", ORIGINAL_URI, CORRECTED_URI, REFERENCE_URI, detail_images=(DETAIL,),
        )
        analyzer(
            (pair,), prompt="Prioritize vertical evidence for L2.",
            result_schema=build_multi_plan_measurement_result_schema(("L2",)),
            cancellation_check=None,
        )
        content = json.loads(requests[0].data)["input"][0]["content"]
        self.assertEqual(content[-2]["text"], f"PLAN ID L2 - {DETAIL.label}")
        self.assertEqual(content[-1]["image_url"], DETAIL.image_data_uri)

    def test_local_batch_honors_targeted_retry_prompt_and_keeps_detail_provenance(self):
        calls = []
        analyzer = LocalQwenMultiPlanMeasurementAnalyzer(
            server_factory=lambda **_kwargs: _FakeQwenServer(calls)
        )
        pair = EncodedPlanMeasurementPair(
            "L2", ORIGINAL_URI, CORRECTED_URI, REFERENCE_URI, detail_images=(DETAIL,),
        )
        result = analyzer(
            (pair,), prompt="RETRY L2: find two vertical spans near the bottom.",
            result_schema=build_multi_plan_measurement_result_schema(("L2",)),
            cancellation_check=None,
        )
        request = next(value for kind, value in calls if kind == "estimate")
        self.assertIn("RETRY L2: find two vertical spans near the bottom.", request["prompt"])
        self.assertIn("Other plan pairs are not visible", request["prompt"])
        self.assertEqual(
            request["additional_images"],
            ((f"PLAN ID L2 - {DETAIL.label}", DETAIL.image_data_uri),),
        )
        self.assertEqual(result, {"plans": {"L2": MEASUREMENT_RESULT}})


if __name__ == "__main__":
    unittest.main()
