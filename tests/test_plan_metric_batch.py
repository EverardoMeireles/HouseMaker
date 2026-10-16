# ### Imports ###
from __future__ import annotations

import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from housemaker.plan_correction_models import (
    PLAN_CORRECTION_MODEL_GPT_5_6_TERRA,
    PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
    PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1,
)
from housemaker.plan_metric_batch import (
    MAX_BATCH_MEASUREMENTS_PER_PLAN,
    MULTI_PLAN_MEASUREMENT_PROMPT,
    EncodedPlanMeasurementPair,
    PlanMeasurementRasterPair,
    PlanMetricBatchCoordinator,
    build_multi_plan_measurement_result_schema,
    parse_multi_plan_measurement_result,
    transcribe_multi_plan_measurements,
)
from housemaker.plan_metric_calibration import (
    PlanMetricCalibrationCancelled,
    PlanMetricCalibrationInferenceError,
)
from housemaker.plan_metric_providers import (
    LOCAL_QWEN_MAX_MEASUREMENTS,
    LocalQwenMultiPlanMeasurementAnalyzer,
    OpenAIMultiPlanMeasurementAnalyzer,
    create_multi_plan_measurement_analyzer,
)

# ### Fixtures ###
ORIGINAL_A_URI = "data:image/png;base64,b3JpZ2luYWwtYQ=="
REFERENCE_A_URI = "data:image/png;base64,cmVmZXJlbmNlLWE="
CORRECTED_A_URI = "data:image/png;base64,Y29ycmVjdGVkLWE="
ORIGINAL_B_URI = "data:image/png;base64,b3JpZ2luYWwtYg=="
REFERENCE_B_URI = "data:image/png;base64,cmVmZXJlbmNlLWI="
CORRECTED_B_URI = "data:image/png;base64,Y29ycmVjdGVkLWI="


def _measurement(label: str, value: float) -> dict[str, object]:
    return {
        "label": label,
        "value_meters": value,
        "start_normalized": [0.1, 0.2],
        "end_normalized": [0.8, 0.2],
        "corrected_start_normalized": [0.12, 0.18],
        "corrected_end_normalized": [0.82, 0.18],
        "confidence": "high",
    }


def _grouped_result() -> dict[str, object]:
    return {
        "plans": {
            "L2": {"measurements": [_measurement("Ground width", 6.1)]},
            "L3": {"measurements": [_measurement("Story width", 6.1)]},
        }
    }


def _calibratable_measurements(
    prefix: str,
    scale_multiplier: float,
) -> list[dict[str, object]]:
    return [
        {
            **_measurement(f"{prefix} room width", 1.0 * scale_multiplier),
            "corrected_start_normalized": [0.10, 0.20],
            "corrected_end_normalized": [0.35, 0.20],
        },
        {
            **_measurement(f"{prefix} room length", 1.2 * scale_multiplier),
            "corrected_start_normalized": [0.20, 0.10],
            "corrected_end_normalized": [0.20, 0.40],
        },
    ]


def _calibratable_grouped_result() -> dict[str, object]:
    return {
        "plans": {
            "L2": {"measurements": _calibratable_measurements("Ground", 1.0)},
            "L3": {"measurements": _calibratable_measurements("Story", 2.0)},
        }
    }


def _encoded_pairs() -> tuple[EncodedPlanMeasurementPair, ...]:
    return (
        EncodedPlanMeasurementPair("L2", ORIGINAL_A_URI, CORRECTED_A_URI),
        EncodedPlanMeasurementPair("L3", ORIGINAL_B_URI, CORRECTED_B_URI),
    )


def _encoded_reference_pairs() -> tuple[EncodedPlanMeasurementPair, ...]:
    return (
        EncodedPlanMeasurementPair(
            "L2",
            ORIGINAL_A_URI,
            CORRECTED_A_URI,
            REFERENCE_A_URI,
        ),
        EncodedPlanMeasurementPair(
            "L3",
            ORIGINAL_B_URI,
            CORRECTED_B_URI,
            REFERENCE_B_URI,
        ),
    )


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._read = False

    def __enter__(self) -> _FakeResponse:  # noqa: PYI034
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, _size: int = -1) -> bytes:
        if self._read:
            return b""
        self._read = True
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
                {"image_data_uri": image_data_uri, **kwargs},
            )
        )
        return {
            "measurements": [
                _measurement(f"Measurement {len(self._calls)}", 6.1)
            ]
        }


# ### Provider-neutral batch tests ###
class PlanMetricBatchTests(unittest.TestCase):
    def test_schema_is_strict_and_keyed_by_stable_plan_id(self) -> None:
        schema = build_multi_plan_measurement_result_schema(("L2", "L3"))

        plans = schema["properties"]["plans"]
        self.assertEqual(plans["required"], ["L2", "L3"])
        self.assertFalse(plans["additionalProperties"])
        self.assertEqual(set(plans["properties"]), {"L2", "L3"})
        self.assertTrue(
            all(
                plan_schema["properties"]["measurements"]["maxItems"]
                == MAX_BATCH_MEASUREMENTS_PER_PLAN
                for plan_schema in plans["properties"].values()
            )
        )
        self.assertFalse(schema["additionalProperties"])

    def test_parser_keeps_measurements_in_their_own_plan_group(self) -> None:
        groups = parse_multi_plan_measurement_result(
            _grouped_result(),
            expected_plan_ids=("L2", "L3"),
        )

        self.assertEqual([group.plan_id for group in groups], ["L2", "L3"])
        self.assertEqual(groups[0].measurements[0].label, "Ground width")
        self.assertEqual(groups[1].measurements[0].label, "Story width")

    def test_parser_rejects_missing_or_unrequested_plan_groups(self) -> None:
        payload = _grouped_result()
        del payload["plans"]["L3"]

        with self.assertRaisesRegex(
            PlanMetricCalibrationInferenceError,
            "wrong plan IDs",
        ):
            parse_multi_plan_measurement_result(
                payload,
                expected_plan_ids=("L2", "L3"),
            )

    def test_transcriber_encodes_all_pairs_in_one_analyzer_call(self) -> None:
        image = np.full((48, 64, 3), 255, dtype=np.uint8)
        calls: list[tuple[object, dict[str, object]]] = []

        def analyzer(plan_pairs: object, **kwargs: object) -> object:
            calls.append((plan_pairs, kwargs))
            return _grouped_result()

        groups = transcribe_multi_plan_measurements(
            (
                PlanMeasurementRasterPair("L2", image, image.copy()),
                PlanMeasurementRasterPair("L3", image, image.copy()),
            ),
            analyzer,
        )

        self.assertEqual(len(calls), 1)
        pairs, kwargs = calls[0]
        self.assertEqual([pair.plan_id for pair in pairs], ["L2", "L3"])
        self.assertTrue(
            all(
                pair.original_image_data_uri.startswith("data:image/png;base64,")
                for pair in pairs
            )
        )
        self.assertEqual(kwargs["prompt"], MULTI_PLAN_MEASUREMENT_PROMPT)
        self.assertEqual([group.plan_id for group in groups], ["L2", "L3"])

    def test_transcriber_keeps_raw_and_rectified_images_separate(self) -> None:
        original = np.full((48, 64, 3), 32, dtype=np.uint8)
        reference = np.full((48, 64, 3), 160, dtype=np.uint8)
        corrected = np.full((48, 64, 3), 255, dtype=np.uint8)
        calls: list[object] = []

        def analyzer(plan_pairs: object, **_kwargs: object) -> object:
            calls.append(plan_pairs)
            return _grouped_result()

        transcribe_multi_plan_measurements(
            (
                PlanMeasurementRasterPair(
                    "L2",
                    original,
                    corrected,
                    reference,
                ),
                PlanMeasurementRasterPair(
                    "L3",
                    original,
                    corrected,
                    reference,
                ),
            ),
            analyzer,
        )

        pairs = calls[0]
        self.assertTrue(
            all(pair.measurement_reference_image_data_uri for pair in pairs)
        )
        self.assertTrue(
            all(
                pair.original_image_data_uri
                != pair.measurement_reference_image_data_uri
                for pair in pairs
            )
        )


# ### Coordinated native-resolution calibration tests ###
class PlanMetricBatchCoordinatorTests(unittest.TestCase):
    def test_waits_for_every_plan_and_keeps_raster_scales_independent(self) -> None:
        analyzer_calls: list[tuple[str, ...]] = []

        def analyzer(plan_pairs: object, **_kwargs: object) -> object:
            analyzer_calls.append(tuple(pair.plan_id for pair in plan_pairs))
            return _calibratable_grouped_result()

        coordinator = PlanMetricBatchCoordinator(("L2", "L3"), analyzer)
        image = np.full((400, 400, 3), 255, dtype=np.uint8)
        first_check_called = threading.Event()

        def first_check() -> bool:
            first_check_called.set()
            return False

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(
                coordinator.normalize,
                "L2",
                image,
                image.copy(),
                cancellation_check=first_check,
            )
            self.assertTrue(first_check_called.wait(1.0))
            self.assertFalse(first.done())
            second = executor.submit(
                coordinator.normalize,
                "L3",
                image,
                image.copy(),
            )
            first_result = first.result(timeout=3.0)
            second_result = second.result(timeout=3.0)

        self.assertEqual(analyzer_calls, [("L2", "L3"), ("L2", "L3")])
        self.assertAlmostEqual(
            first_result.calibration.estimated_meters_per_pixel,
            0.01,
        )
        self.assertAlmostEqual(
            second_result.calibration.estimated_meters_per_pixel,
            0.02,
        )
        self.assertEqual(first_result.image.shape[:2], (400, 400))
        self.assertEqual(second_result.image.shape[:2], (400, 400))
        np.testing.assert_array_equal(first_result.image, image)
        np.testing.assert_array_equal(second_result.image, image)
        self.assertEqual(first_result.calibration.normalization_scale, 1.0)
        self.assertEqual(second_result.calibration.normalization_scale, 1.0)
        self.assertAlmostEqual(first_result.calibration.target_meters_per_pixel, 0.01)
        self.assertAlmostEqual(second_result.calibration.target_meters_per_pixel, 0.02)

    def test_cancel_wakes_a_worker_waiting_at_the_batch_barrier(self) -> None:
        coordinator = PlanMetricBatchCoordinator(
            ("L2", "L3"),
            lambda *_args, **_kwargs: _calibratable_grouped_result(),
        )
        image = np.full((64, 64, 3), 255, dtype=np.uint8)
        check_called = threading.Event()

        def cancellation_check() -> bool:
            check_called.set()
            return False

        with ThreadPoolExecutor(max_workers=1) as executor:
            waiting = executor.submit(
                coordinator.normalize,
                "L2",
                image,
                image.copy(),
                cancellation_check=cancellation_check,
            )
            self.assertTrue(check_called.wait(1.0))
            coordinator.cancel()
            with self.assertRaisesRegex(
                PlanMetricCalibrationCancelled,
                "cancelled",
            ):
                waiting.result(timeout=1.0)

        with self.assertRaises(PlanMetricCalibrationCancelled):
            coordinator.normalize("L3", image, image.copy())

    def test_grouped_provider_failure_wakes_every_worker(self) -> None:
        def analyzer(*_args: object, **_kwargs: object) -> object:
            raise PlanMetricCalibrationInferenceError("Provider failed safely.")

        coordinator = PlanMetricBatchCoordinator(("L2", "L3"), analyzer)
        image = np.full((64, 64, 3), 255, dtype=np.uint8)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = tuple(
                executor.submit(
                    coordinator.normalize,
                    plan_id,
                    image,
                    image.copy(),
                )
                for plan_id in ("L2", "L3")
            )
            for future in futures:
                with self.assertRaisesRegex(
                    PlanMetricCalibrationInferenceError,
                    "Provider failed safely",
                ):
                    future.result(timeout=2.0)

    def test_one_unsolved_plan_does_not_discard_another_local_solution(self) -> None:
        payload = _calibratable_grouped_result()
        payload["plans"]["L3"] = {"measurements": []}
        coordinator = PlanMetricBatchCoordinator(
            ("L2", "L3"),
            lambda *_args, **_kwargs: payload,
        )
        image = np.full((400, 400, 3), 255, dtype=np.uint8)
        with ThreadPoolExecutor(max_workers=2) as executor:
            valid = executor.submit(
                coordinator.normalize,
                "L2",
                image,
                image.copy(),
            )
            invalid = executor.submit(
                coordinator.normalize,
                "L3",
                image,
                image.copy(),
            )
            self.assertAlmostEqual(
                valid.result(timeout=3.0).calibration.estimated_meters_per_pixel,
                0.01,
            )
            with self.assertRaisesRegex(
                PlanMetricCalibrationInferenceError,
                "No explicit usable",
            ):
                invalid.result(timeout=3.0)


# ### OpenAI grouped-provider tests ###
class OpenAIMultiPlanMeasurementTests(unittest.TestCase):
    def test_factory_routes_openai_models_to_grouped_provider(self) -> None:
        analyzer = create_multi_plan_measurement_analyzer(
            PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
            api_key="sk-test",
        )

        self.assertIsInstance(analyzer, OpenAIMultiPlanMeasurementAnalyzer)

    def test_one_request_contains_every_labeled_pair_and_grouped_schema(self) -> None:
        captured: list[object] = []

        def fake_open(request, **_kwargs):
            captured.append(request)
            return _FakeResponse(_openai_response(_grouped_result()))

        analyzer = create_multi_plan_measurement_analyzer(
            PLAN_CORRECTION_MODEL_GPT_5_6_TERRA,
            api_key="sk-test-secret",
            opener=fake_open,
        )
        schema = build_multi_plan_measurement_result_schema(("L2", "L3"))
        result = analyzer(
            _encoded_pairs(),
            prompt=MULTI_PLAN_MEASUREMENT_PROMPT,
            result_schema=schema,
            cancellation_check=None,
        )

        self.assertEqual(result, _grouped_result())
        self.assertEqual(len(captured), 1)
        body = json.loads(captured[0].data.decode("utf-8"))
        content = body["input"][0]["content"]
        self.assertEqual(
            [item["image_url"] for item in content if item["type"] == "input_image"],
            [
                ORIGINAL_A_URI,
                CORRECTED_A_URI,
                ORIGINAL_B_URI,
                CORRECTED_B_URI,
            ],
        )
        labels = [item["text"] for item in content if item["type"] == "input_text"]
        self.assertIn("PLAN ID L2 - MEASUREMENT REFERENCE PLAN:", labels)
        self.assertIn("PLAN ID L3 - CORRECTED PLAN:", labels)
        output_format = body["text"]["format"]
        self.assertTrue(output_format["strict"])
        self.assertEqual(output_format["schema"], schema)

    def test_one_request_contains_raw_reference_and_final_for_every_plan(self) -> None:
        captured: list[object] = []

        def fake_open(request, **_kwargs):
            captured.append(request)
            return _FakeResponse(_openai_response(_grouped_result()))

        analyzer = create_multi_plan_measurement_analyzer(
            PLAN_CORRECTION_MODEL_GPT_5_6_TERRA,
            api_key="sk-test-secret",
            opener=fake_open,
        )
        analyzer(
            _encoded_reference_pairs(),
            prompt=MULTI_PLAN_MEASUREMENT_PROMPT,
            result_schema=build_multi_plan_measurement_result_schema(
                ("L2", "L3")
            ),
            cancellation_check=None,
        )

        body = json.loads(captured[0].data.decode("utf-8"))
        content = body["input"][0]["content"]
        self.assertEqual(
            [item["image_url"] for item in content if item["type"] == "input_image"],
            [
                ORIGINAL_A_URI,
                REFERENCE_A_URI,
                CORRECTED_A_URI,
                ORIGINAL_B_URI,
                REFERENCE_B_URI,
                CORRECTED_B_URI,
            ],
        )
        labels = [item["text"] for item in content if item["type"] == "input_text"]
        self.assertIn("PLAN ID L2 - RECTIFIED ANNOTATED PLAN:", labels)

    def test_preflight_cancellation_does_not_open_transport(self) -> None:
        opened = False

        def fake_open(*_args: object, **_kwargs: object) -> object:
            nonlocal opened
            opened = True
            return _FakeResponse(_openai_response(_grouped_result()))

        analyzer = create_multi_plan_measurement_analyzer(
            PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
            api_key="sk-test",
            opener=fake_open,
        )
        with self.assertRaises(PlanMetricCalibrationCancelled):
            analyzer(
                _encoded_pairs(),
                prompt=MULTI_PLAN_MEASUREMENT_PROMPT,
                result_schema=build_multi_plan_measurement_result_schema(
                    ("L2", "L3")
                ),
                cancellation_check=lambda: True,
            )

        self.assertFalse(opened)


# ### Local Qwen grouped-provider tests ###
class LocalQwenMultiPlanMeasurementTests(unittest.TestCase):
    def test_qwen_fallback_is_sequential_inside_one_server_lifetime(self) -> None:
        calls: list[tuple[str, object]] = []

        def server_factory(**kwargs: object) -> _FakeQwenServer:
            calls.append(("factory", kwargs))
            return _FakeQwenServer(calls)

        analyzer = create_multi_plan_measurement_analyzer(
            PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1,
            qwen_server_factory=server_factory,
        )
        self.assertIsInstance(analyzer, LocalQwenMultiPlanMeasurementAnalyzer)
        result = analyzer(
            _encoded_pairs(),
            prompt=MULTI_PLAN_MEASUREMENT_PROMPT,
            result_schema=build_multi_plan_measurement_result_schema(
                ("L2", "L3")
            ),
            cancellation_check=None,
        )

        self.assertEqual(list(result["plans"]), ["L2", "L3"])
        self.assertEqual(
            [name for name, _value in calls],
            ["factory", "enter", "estimate", "estimate", "exit"],
        )
        first = calls[2][1]
        second = calls[3][1]
        self.assertEqual(first["image_data_uri"], CORRECTED_A_URI)
        self.assertEqual(first["previous_image_data_uri"], ORIGINAL_A_URI)
        self.assertNotIn("plan_image_data_uri", first)
        self.assertEqual(second["image_data_uri"], CORRECTED_B_URI)
        self.assertIn("only PLAN ID L2", first["prompt"])
        self.assertIn("not visible in the local fallback", first["prompt"])
        self.assertEqual(
            first["result_schema"]["properties"]["measurements"]["maxItems"],
            LOCAL_QWEN_MAX_MEASUREMENTS,
        )

    def test_qwen_uses_raw_plan_reference_with_rectified_previous(self) -> None:
        calls: list[tuple[str, object]] = []

        def server_factory(**kwargs: object) -> _FakeQwenServer:
            calls.append(("factory", kwargs))
            return _FakeQwenServer(calls)

        analyzer = create_multi_plan_measurement_analyzer(
            PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1,
            qwen_server_factory=server_factory,
        )
        analyzer(
            _encoded_reference_pairs(),
            prompt=MULTI_PLAN_MEASUREMENT_PROMPT,
            result_schema=build_multi_plan_measurement_result_schema(
                ("L2", "L3")
            ),
            cancellation_check=None,
        )

        first = calls[2][1]
        self.assertEqual(first["image_data_uri"], CORRECTED_A_URI)
        self.assertEqual(first["previous_image_data_uri"], REFERENCE_A_URI)
        self.assertEqual(first["plan_image_data_uri"], ORIGINAL_A_URI)
        self.assertIn("PLAN REFERENCE is the raw ORIGINAL PLAN", first["prompt"])


if __name__ == "__main__":
    unittest.main()
