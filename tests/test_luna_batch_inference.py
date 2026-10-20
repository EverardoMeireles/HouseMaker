# ### Imports ###
from __future__ import annotations

import json
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from housemaker import ceiling_height_estimation as ceiling
from housemaker import plan_metric_providers as metric
from housemaker import reference_symmetry_inference as symmetry
from housemaker import wall_llm_verification as walls
from housemaker.ceiling_height_ranges import CeilingHeightVideoRange
from housemaker.models import VertexData
from housemaker.openai_batch import (
    LUNA_BATCH_MODEL,
    LUNA_MODEL,
    BatchResult,
    OpenAIBatchCancelled,
    OpenAIBatchError,
)
from housemaker.plan_metric_batch import EncodedPlanMeasurementPair

# ### Fixtures ###
IMAGE = np.full((24, 32, 3), 255, np.uint8)
URI = "data:image/png;base64,dGVzdA=="
HEIGHT = {
    "status": "estimated", "estimate_m": 2.6,
    "minimum_m": 2.4, "maximum_m": 2.8, "confidence": "medium",
    "basis": "Room geometry.", "level_movement": "stable",
    "movement_confidence": "low",
}
SYMMETRY = {"symmetric": True, "confidence": 0.9, "reason": "Matching halves."}


def envelope(payload: object) -> dict:
    return {
        "status": "completed",
        "output": [{"type": "message", "content": [
            {"type": "output_text", "text": json.dumps(payload)},
        ]}],
    }


# ### Selection and request tests ###
class BatchInferenceSelectionTests(unittest.TestCase):
    def test_batch_available_alongside_luna_for_all_inference_modes(self) -> None:
        for options in (
            ceiling.CEILING_HEIGHT_MODEL_OPTIONS,
            symmetry.SYMMETRY_INFERENCE_MODEL_OPTIONS,
            walls.WALL_VERIFICATION_MODEL_OPTIONS,
        ):
            self.assertIn(LUNA_BATCH_MODEL, dict(options).values())
            self.assertIn(LUNA_MODEL, dict(options).values())
        self.assertEqual(
            metric.plan_measurement_model_for_correction_model(LUNA_BATCH_MODEL),
            LUNA_BATCH_MODEL,
        )

    def test_ceiling_wire_uses_actual_model_and_estimate_is_half_price(self) -> None:
        self.assertEqual(
            ceiling.build_ceiling_height_request_body(URI, model=LUNA_BATCH_MODEL)["model"],
            LUNA_MODEL,
        )
        arguments = {"frame_count": 350, "interval": 15, "frame_width": 1280, "frame_height": 720}
        regular = ceiling.estimate_ceiling_height_api_cost_usd(model=LUNA_MODEL, **arguments)
        batch = ceiling.estimate_ceiling_height_api_cost_usd(model=LUNA_BATCH_MODEL, **arguments)
        self.assertAlmostEqual(batch * 2, regular)


# ### Single request transport routing tests ###
class BatchInferenceRoutingTests(unittest.TestCase):
    def test_ceiling_uses_batch_without_realtime_transport_lock(self) -> None:
        cancelled = lambda: False
        opener = Mock()
        with (
            patch.object(ceiling, "request_luna_batch_response", return_value=envelope(HEIGHT)) as request,
            patch.object(ceiling, "_request_ceiling_height_estimate") as realtime,
        ):
            result = ceiling.infer_ceiling_height(
                IMAGE, api_key="test", model=LUNA_BATCH_MODEL,
                cancellation_check=cancelled, opener=opener,
            )
        self.assertEqual(result.estimate_m, 2.6)
        self.assertEqual(request.call_args.args[0]["model"], LUNA_MODEL)
        self.assertIs(request.call_args.kwargs["cancellation_check"], cancelled)
        self.assertIs(request.call_args.kwargs["opener"], opener)
        realtime.assert_not_called()

    def test_symmetry_uses_batch_and_preserves_alias_in_result(self) -> None:
        progress = Mock()
        with (
            patch.object(symmetry, "request_luna_batch_response", return_value=envelope(SYMMETRY)) as request,
            patch.object(symmetry, "_request_luna_result") as realtime,
        ):
            result = symmetry.infer_reference_symmetry(
                IMAGE, LUNA_BATCH_MODEL, api_key="test", progress_callback=progress,
            )
        self.assertTrue(result.symmetric)
        self.assertEqual(result.model, LUNA_BATCH_MODEL)
        self.assertEqual(request.call_args.args[0]["model"], LUNA_MODEL)
        self.assertIs(request.call_args.kwargs["progress_callback"], progress)
        realtime.assert_not_called()

    def test_wall_uses_batch_and_existing_result_parser(self) -> None:
        payload = {"new_vertices": [], "additions": [], "summary": "No additions."}
        with (
            patch.object(walls, "_prepare_input", return_value=(URI, URI, (100, 100))),
            patch.object(walls, "request_luna_batch_response", return_value=envelope(payload)) as request,
            patch.object(walls, "_request_openai") as realtime,
        ):
            result = walls.verify_plan_walls(
                Path("unused.png"), VertexData(), api_key="test", model=LUNA_BATCH_MODEL,
            )
        self.assertFalse(result.proposals)
        self.assertEqual(request.call_args.args[0]["model"], LUNA_MODEL)
        realtime.assert_not_called()

    def test_plan_single_and_multi_transcriptions_use_batch(self) -> None:
        for multi in (False, True):
            with (
                self.subTest(multi=multi),
                patch.object(metric, "request_luna_batch_response", return_value=envelope({"measurements": []})) as request,
                patch.object(metric, "_request_openai_measurements") as realtime,
            ):
                analyzer = (
                    metric.create_multi_plan_measurement_analyzer if multi
                    else metric.create_plan_measurement_analyzer
                )(LUNA_BATCH_MODEL, api_key="test")
                args = (
                    ((EncodedPlanMeasurementPair("L1", URI, URI),),) if multi
                    else (URI, URI)
                )
                result = analyzer(*args, prompt="Read dimensions", result_schema={}, cancellation_check=None)
            self.assertEqual(result, {"measurements": []})
            self.assertEqual(request.call_args.args[0]["model"], LUNA_MODEL)
            realtime.assert_not_called()

    def test_typed_cancellation_and_error_reasons_are_not_hidden_or_fallback(self) -> None:
        for module, invoke, cancelled_type, error_type in (
            (ceiling, lambda: ceiling.infer_ceiling_height(IMAGE, api_key="test", model=LUNA_BATCH_MODEL),
             ceiling.CeilingHeightEstimationCancelled, ceiling.CeilingHeightEstimationError),
            (symmetry, lambda: symmetry.infer_reference_symmetry(IMAGE, LUNA_BATCH_MODEL, api_key="test"),
             symmetry.SymmetryInferenceCancelled, symmetry.SymmetryInferenceError),
        ):
            for cause, expected in (
                (OpenAIBatchCancelled("User cancelled batch"), cancelled_type),
                (OpenAIBatchError("Batch model not available"), error_type),
            ):
                with (
                    self.subTest(module=module.__name__, cause=type(cause)),
                    patch.object(module, "request_luna_batch_response", side_effect=cause),
                    self.assertRaisesRegex(expected, str(cause)),
                ):
                    invoke()

    def test_plan_cancellation_preserves_typed_error_and_reason(self) -> None:
        analyzer = metric.create_plan_measurement_analyzer(LUNA_BATCH_MODEL, api_key="test")
        with (
            patch.object(metric, "request_luna_batch_response", side_effect=OpenAIBatchCancelled("Cancelled remotely")),
            self.assertRaisesRegex(metric.PlanMetricCalibrationCancelled, "Cancelled remotely"),
        ):
            analyzer(URI, URI, prompt="Read dimensions", result_schema={}, cancellation_check=None)

    def test_wall_cancellation_preserves_typed_error_and_reason(self) -> None:
        with (
            patch.object(walls, "_prepare_input", return_value=(URI, URI, (100, 100))),
            patch.object(walls, "request_luna_batch_response", side_effect=OpenAIBatchCancelled("Cancelled remotely")),
            self.assertRaisesRegex(walls.WallVerificationCancelled, "Cancelled remotely"),
        ):
            walls.verify_plan_walls(Path("unused.png"), VertexData(), api_key="test", model=LUNA_BATCH_MODEL)


# ### Grouped workflow tests ###
class GroupedBatchInferenceTests(unittest.TestCase):
    def test_malformed_json_is_per_item_and_does_not_lose_other_results(self) -> None:
        invalid = envelope(HEIGHT)
        invalid["output"][0]["content"][0]["text"] = "not JSON"
        for module, invoke, valid_payload, expected_type in (
            (ceiling, lambda: ceiling.infer_ceiling_height_batch({"a": {}, "b": {}}, api_key="test"),
             HEIGHT, ceiling.CeilingHeightEstimationError),
            (symmetry, lambda: symmetry.infer_reference_symmetry_batch({"a": IMAGE, "b": IMAGE}, api_key="test"),
             SYMMETRY, symmetry.SymmetryInferenceError),
        ):
            with self.subTest(module=module.__name__), patch.object(module, "run_responses_batch", return_value={
                "a": BatchResult(invalid, None), "b": BatchResult(envelope(valid_payload), None),
            }):
                results = invoke()
            self.assertIsInstance(results["a"], expected_type)
            self.assertNotIsInstance(results["b"], Exception)

    def test_group_cancellation_is_not_returned_as_an_item_error(self) -> None:
        for module, invoke, expected_type in (
            (ceiling, lambda: ceiling.infer_ceiling_height_batch({"a": {}}, api_key="test"),
             ceiling.CeilingHeightEstimationCancelled),
            (symmetry, lambda: symmetry.infer_reference_symmetry_batch({"a": IMAGE}, api_key="test"),
             symmetry.SymmetryInferenceCancelled),
        ):
            with (
                self.subTest(module=module.__name__),
                patch.object(module, "run_responses_batch", side_effect=OpenAIBatchCancelled("Cancelled remotely")),
                self.assertRaisesRegex(expected_type, "Cancelled remotely"),
            ):
                invoke()

    def test_symmetry_groups_images_and_keeps_partial_errors(self) -> None:
        with patch.object(symmetry, "run_responses_batch", return_value={
            "a": BatchResult(envelope(SYMMETRY), None),
            "b": BatchResult(None, "Unavailable input"),
        }) as request:
            results = symmetry.infer_reference_symmetry_batch(
                {"a": IMAGE, "b": IMAGE}, api_key="test",
            )
        self.assertEqual(request.call_count, 1)
        self.assertEqual(set(request.call_args.args[0]), {"a", "b"})
        self.assertEqual(results["a"].model, LUNA_BATCH_MODEL)
        self.assertIsInstance(results["b"], symmetry.SymmetryInferenceError)

    def test_ceiling_groups_pairs_and_keeps_partial_errors(self) -> None:
        body = ceiling.build_ceiling_height_batch_request(IMAGE, previous_frame_bgr=IMAGE)
        with patch.object(ceiling, "run_responses_batch", return_value={
            "a": BatchResult(envelope(HEIGHT), None),
            "b": BatchResult(None, "Unavailable input"),
        }) as request:
            results = ceiling.infer_ceiling_height_batch({"a": body, "b": body}, api_key="test")
        self.assertEqual(request.call_count, 1)
        self.assertEqual(results["a"].estimate_m, 2.6)
        self.assertIsInstance(results["b"], ceiling.CeilingHeightEstimationError)

    def test_ceiling_worker_submits_one_group_and_resets_previous_at_each_range(self) -> None:
        from housemaker import merged_generation_workspace as merged

        thread = merged._CeilingHeightInferenceThread(
            "unused.avi", ((0, 1), (1, 1), (4, 2), (5, 2)), "test", LUNA_BATCH_MODEL,
            None, ((1, "Ground"), (2, "Upstairs")), (),
            (CeilingHeightVideoRange(1, 0, 1), CeilingHeightVideoRange(2, 4, 5)),
        )
        reader = Mock()
        reader.get_frame.return_value = IMAGE
        captured_previous = []

        def prepare(frame, *, previous_frame_bgr, **_kwargs):
            captured_previous.append(previous_frame_bgr)
            return {"model": LUNA_MODEL}

        estimate = ceiling.parse_ceiling_height_estimate(HEIGHT)
        with (
            patch.object(merged, "VideoFrameSource", return_value=nullcontext(reader)),
            patch.object(ceiling, "build_ceiling_height_batch_request", side_effect=prepare),
            patch.object(ceiling, "infer_ceiling_height_batch", return_value={
                str(index): estimate for index in range(1, 5)
            }) as request,
            patch.object(merged, "open_ceiling_height_estimator") as realtime,
        ):
            thread.run()
        self.assertIsNone(thread.error_message)
        self.assertIsNotNone(thread.result)
        self.assertEqual(request.call_count, 1)
        self.assertEqual(tuple(item is None for item in captured_previous), (True, False, True, False))
        self.assertEqual(thread._api_key, "")
        realtime.assert_not_called()


# ### Test runner ###
if __name__ == "__main__":
    unittest.main()
