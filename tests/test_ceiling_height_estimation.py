# ### Imports ###
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from housemaker import ceiling_height_estimation as estimation
from housemaker.ceiling_height_estimation import (
    CEILING_HEIGHT_INFERENCE_MODEL,
    CEILING_HEIGHT_MODEL_OPTIONS,
    CeilingHeightAnalysisResult,
    CeilingHeightEstimate,
    CeilingHeightEstimationCancelled,
    CeilingHeightEstimationError,
    CeilingHeightLevelEstimate,
    aggregate_ceiling_height_estimates,
    aggregate_level_ceiling_height_estimates,
    aggregate_marked_level_ceiling_height_estimates,
    build_ceiling_height_plan_image_data_uri,
    build_ceiling_height_prompt,
    build_ceiling_height_request_body,
    ceiling_height_model_requires_api_key,
    estimate_ceiling_height_api_cost_usd,
    format_ceiling_height_estimate,
    infer_ceiling_height,
    sample_entire_video_frame_indices,
    sample_video_frame_indices,
)
from housemaker.plan_metric_state import (
    PlanMeasurementEvidence,
    PlanMetricCalibration,
)
from housemaker.qwen_ceiling_height_estimation import (
    LOCAL_QWEN_CEILING_HEIGHT_MODEL,
)


# ### Metric plan-reference tests ###
class CeilingHeightPlanReferenceTests(unittest.TestCase):
    def test_annotated_plan_reference_is_encoded_for_vision_input(self) -> None:
        measurements = (
            PlanMeasurementEvidence(
                label="Room width",
                value_meters=4.0,
                start_normalized=(0.1, 0.25),
                end_normalized=(0.9, 0.25),
                corrected_start_normalized=(0.1, 0.25),
                corrected_end_normalized=(0.9, 0.25),
                confidence="high",
            ),
            PlanMeasurementEvidence(
                label="Room depth",
                value_meters=3.0,
                start_normalized=(0.2, 0.1),
                end_normalized=(0.2, 0.9),
                corrected_start_normalized=(0.2, 0.1),
                corrected_end_normalized=(0.2, 0.9),
                confidence="high",
            ),
        )
        calibration = PlanMetricCalibration(
            estimated_meters_per_pixel=0.02,
            target_meters_per_pixel=0.02,
            normalization_scale=1.0,
            relative_residual=0.01,
            confidence="high",
            source_image_size=(64, 48),
            pre_normalization_size=(64, 48),
            normalized_image_size=(64, 48),
            measurements=measurements,
        )
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "plan.png"
            self.assertTrue(
                estimation.cv2.imwrite(
                    str(image_path),
                    np.full((48, 64, 3), 255, dtype=np.uint8),
                )
            )

            data_uri = build_ceiling_height_plan_image_data_uri(
                image_path,
                calibration,
            )

        self.assertTrue(data_uri.startswith("data:image/jpeg;base64,"))


# ### Fixtures ###
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
        "level_movement": "stable",
        "movement_confidence": "low",
    }


def _estimate(height: float) -> CeilingHeightEstimate:
    return CeilingHeightEstimate(
        status="estimated",
        estimate_m=height,
        minimum_m=height - 0.2,
        maximum_m=height + 0.2,
        confidence="medium",
        basis="Visible architectural scale cues agree.",
    )


def _movement(
    level_movement: str,
    *,
    height: float | None = None,
    confidence: str = "medium",
) -> CeilingHeightEstimate:
    if height is None:
        return CeilingHeightEstimate(
            status="insufficient_evidence",
            estimate_m=None,
            minimum_m=None,
            maximum_m=None,
            confidence="low",
            basis="The camera is moving between levels.",
            level_movement=level_movement,
            movement_confidence=confidence,
        )
    return CeilingHeightEstimate(
        status="estimated",
        estimate_m=height,
        minimum_m=height - 0.2,
        maximum_m=height + 0.2,
        confidence="medium",
        basis="Visible architectural scale cues agree.",
        level_movement=level_movement,
        movement_confidence=confidence,
    )


def _insufficient() -> CeilingHeightEstimate:
    return CeilingHeightEstimate(
        status="insufficient_evidence",
        estimate_m=None,
        minimum_m=None,
        maximum_m=None,
        confidence="low",
        basis="No reliable scale cue is visible.",
    )


# ### Video analysis tests ###
class CeilingHeightVideoAnalysisTests(unittest.TestCase):
    def test_entire_video_sampling_starts_at_zero_and_includes_last_frame(
        self,
    ) -> None:
        sampled = sample_entire_video_frame_indices(350, 15)

        self.assertEqual(len(sampled), 25)
        self.assertEqual(sampled[:3], (0, 15, 30))
        self.assertEqual(sampled[-2:], (345, 349))
        self.assertEqual(sample_entire_video_frame_indices(0, 15), ())

    def test_frame_sampling_is_inclusive_and_always_appends_end(self) -> None:
        self.assertEqual(sample_video_frame_indices(4, 14, 5), (4, 9, 14))
        self.assertEqual(sample_video_frame_indices(4, 15, 5), (4, 9, 14, 15))
        self.assertEqual(sample_video_frame_indices(7, 7, 20), (7,))

    def test_frame_sampling_rejects_invalid_ranges_and_intervals(self) -> None:
        for arguments in ((-1, 10, 2), (10, 5, 2), (0, 10, 0), (0, 10, True)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                sample_video_frame_indices(*arguments)

    def test_aggregation_discards_distant_values_and_tracks_all_counts(self) -> None:
        result = aggregate_ceiling_height_estimates(
            (
                (0, _estimate(2.6)),
                (10, _estimate(2.7)),
                (20, _estimate(2.8)),
                (30, _estimate(5.4)),
                (40, _insufficient()),
            )
        )

        self.assertTrue(result.has_consensus)
        self.assertEqual(result.average_height_m, 2.7)
        self.assertEqual(result.retained_frame_indices, (0, 10, 20))
        self.assertEqual(result.outlier_frame_indices, (30,))
        self.assertEqual(result.insufficient_frame_indices, (40,))
        self.assertEqual(result.retained_count, 3)
        self.assertEqual(result.outlier_count, 1)
        self.assertEqual(result.insufficient_count, 1)
        self.assertEqual(result.analyzed_count, 5)

    def test_aggregation_uses_the_densest_strict_majority_cluster(self) -> None:
        result = aggregate_ceiling_height_estimates(
            tuple(
                (index * 10, _estimate(height))
                for index, height in enumerate((2.5, 2.6, 2.7, 3.5, 3.6))
            )
        )

        self.assertTrue(result.has_consensus)
        self.assertEqual(result.average_height_m, 2.6)
        self.assertEqual(result.retained_frame_indices, (0, 10, 20))
        self.assertEqual(result.outlier_frame_indices, (30, 40))

    def test_aggregation_reports_no_consensus_for_sparse_or_split_estimates(
        self,
    ) -> None:
        cases = (
            ((0, _estimate(2.6)), (10, _insufficient())),
            (
                (0, _estimate(2.5)),
                (10, _estimate(2.6)),
                (20, _estimate(4.0)),
                (30, _estimate(4.1)),
            ),
        )

        for estimates in cases:
            with self.subTest(estimates=estimates):
                result = aggregate_ceiling_height_estimates(estimates)
                self.assertFalse(result.has_consensus)
                self.assertIsNone(result.average_height_m)
                self.assertEqual(
                    result.analyzed_count,
                    len(estimates),
                )

    def test_one_sample_range_preserves_single_frame_fallback(self) -> None:
        result = aggregate_ceiling_height_estimates(((12, _estimate(2.65)),))

        self.assertTrue(result.has_consensus)
        self.assertEqual(result.average_height_m, 2.65)
        self.assertEqual(result.retained_frame_indices, (12,))

    def test_metric_plan_context_is_bounded_and_explicitly_horizontal(self) -> None:
        prompt = build_ceiling_height_prompt(
            "L2 Ground salon span: 6.1 m; doorway width: 0.8 m."
        )

        self.assertIn("CALIBRATED FLOOR-PLAN CONTEXT", prompt)
        self.assertIn("6.1 m", prompt)
        self.assertIn("horizontal X/Y distances only", prompt)
        self.assertIn("never be converted directly", prompt)

        body = build_ceiling_height_request_body(
            "data:image/png;base64,current",
            plan_context="room span: 6.1 m",
        )
        prompt_item = body["input"][0]["content"][0]
        self.assertIn("room span: 6.1 m", prompt_item["text"])

    def test_aggregation_rejects_duplicate_frame_indices(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            aggregate_ceiling_height_estimates(
                ((5, _estimate(2.6)), (5, _estimate(2.7)))
            )

    def test_level_aggregation_assigns_stable_frames_across_an_ascent(
        self,
    ) -> None:
        result = aggregate_level_ceiling_height_estimates(
            (
                (0, _estimate(2.6)),
                (15, _estimate(2.7)),
                (30, _movement("ascending")),
                (45, _movement("ascending")),
                (60, _movement("arrived_up")),
                (75, _estimate(2.8)),
                (90, _estimate(2.9)),
            ),
            starting_level_index=1,
            levels=((0, "Underground 1"), (1, "Ground"), (2, "Story 1")),
        )

        self.assertIsInstance(result, CeilingHeightAnalysisResult)
        self.assertEqual(result.sampled_frame_indices, (0, 15, 30, 45, 60, 75, 90))
        self.assertEqual(result.transition_frame_indices, (30, 45, 60))
        self.assertEqual(result.unassigned_frame_indices, (30, 45, 60))
        self.assertEqual(
            tuple(
                (entry.level_index, entry.level_name)
                for entry in result.level_estimates
            ),
            ((1, "Ground"), (2, "Story 1")),
        )
        self.assertTrue(
            all(
                isinstance(entry, CeilingHeightLevelEstimate)
                for entry in result.level_estimates
            )
        )
        ground, story = result.level_estimates
        self.assertEqual(ground.analysis.average_height_m, 2.65)
        self.assertEqual(ground.analysis.retained_frame_indices, (0, 15))
        self.assertEqual(story.analysis.average_height_m, 2.85)
        self.assertEqual(story.analysis.retained_frame_indices, (75, 90))

    def test_level_aggregation_changes_level_only_on_arrival_events(self) -> None:
        result = aggregate_level_ceiling_height_estimates(
            (
                (0, _estimate(2.7)),
                (15, _movement("descending")),
                (30, _movement("descending")),
                (45, _movement("arrived_down")),
                (60, _estimate(2.4)),
                (75, _movement("uncertain")),
            ),
            starting_level_index=2,
            levels=((0, "Underground 1"), (1, "Ground"), (2, "Story 1")),
        )

        self.assertEqual(result.transition_frame_indices, (15, 30, 45))
        self.assertEqual(result.unassigned_frame_indices, (15, 30, 45, 75))
        self.assertEqual(
            tuple(entry.level_index for entry in result.level_estimates),
            (2, 1),
        )
        by_level = {
            entry.level_index: entry.analysis
            for entry in result.level_estimates
        }
        self.assertEqual(by_level[2].retained_frame_indices, (0,))
        self.assertEqual(by_level[1].retained_frame_indices, (60,))

    def test_level_aggregation_rejects_one_medium_transition_observation(
        self,
    ) -> None:
        result = aggregate_level_ceiling_height_estimates(
            (
                (0, _estimate(2.6)),
                (15, _movement("ascending", confidence="medium")),
                (30, _estimate(2.7)),
            ),
            starting_level_index=1,
            levels=((1, "Ground"), (2, "Story 1")),
        )

        self.assertEqual(result.transition_frame_indices, (15,))
        self.assertEqual(result.unassigned_frame_indices, (15,))
        self.assertEqual(len(result.level_estimates), 1)
        self.assertEqual(result.level_estimates[0].level_index, 1)
        self.assertEqual(
            result.level_estimates[0].analysis.retained_frame_indices,
            (0, 30),
        )

    def test_level_aggregation_commits_one_high_confidence_transition_once(
        self,
    ) -> None:
        result = aggregate_level_ceiling_height_estimates(
            (
                (0, _estimate(2.6)),
                (15, _movement("ascending", confidence="high")),
                (30, _movement("arrived_up", confidence="high")),
                (45, _estimate(2.9)),
            ),
            starting_level_index=1,
            levels=((1, "Ground"), (2, "Story 1"), (3, "Story 2")),
        )

        self.assertEqual(result.transition_frame_indices, (15, 30))
        self.assertEqual(
            tuple(entry.level_index for entry in result.level_estimates),
            (1, 2),
        )

    def test_repeated_arrival_labels_do_not_skip_a_level(self) -> None:
        result = aggregate_level_ceiling_height_estimates(
            (
                (0, _estimate(2.6)),
                (15, _estimate(2.7)),
                (30, _movement("arrived_up", height=2.8)),
                (45, _movement("arrived_up", height=3.1)),
                (60, _estimate(2.9)),
            ),
            starting_level_index=0,
            levels=((0, "Ground"), (1, "Story 1"), (2, "Story 2")),
        )

        self.assertEqual(
            tuple(entry.level_index for entry in result.level_estimates),
            (0, 1),
        )
        self.assertIn(45, result.unassigned_frame_indices)

    def test_level_result_requires_two_stable_height_estimates(self) -> None:
        result = aggregate_level_ceiling_height_estimates(
            ((0, _estimate(2.7)),),
            starting_level_index=1,
            levels=((1, "Ground"),),
        )

        self.assertFalse(result.has_consensus)
        self.assertFalse(result.level_estimates[0].has_consensus)
        self.assertEqual(
            result.level_estimates[0].analysis.retained_frame_indices,
            (0,),
        )

    def test_level_aggregation_does_not_walk_beyond_known_levels(self) -> None:
        result = aggregate_level_ceiling_height_estimates(
            (
                (0, _movement("ascending")),
                (15, _movement("arrived_up")),
                (30, _estimate(2.8)),
            ),
            starting_level_index=2,
            levels=((1, "Ground"), (2, "Story 1")),
        )

        self.assertEqual(result.transition_frame_indices, (0, 15))
        self.assertEqual(result.unassigned_frame_indices, (0, 15))
        self.assertEqual(len(result.level_estimates), 1)
        self.assertEqual(result.level_estimates[0].level_index, 2)
        self.assertEqual(
            result.level_estimates[0].analysis.retained_frame_indices,
            (30,),
        )


# ### Explicit marked-level aggregation tests ###
class MarkedLevelCeilingHeightTests(unittest.TestCase):
    def test_multiple_marks_are_combined_only_for_their_assigned_levels(self) -> None:
        result = aggregate_marked_level_ceiling_height_estimates(
            (
                (2, 30, _estimate(3.0)),
                (1, 0, _estimate(2.4)),
                (1, 10, _estimate(2.5)),
                (2, 40, _estimate(3.1)),
                (1, 60, _estimate(2.6)),
                (1, 70, _estimate(5.5)),
                (2, 50, _insufficient()),
            ),
            levels=((0, "Unmarked cellar"), (1, "Ground"), (2, "Story 1")),
        )

        self.assertEqual(result.sampled_frame_indices, (0, 10, 30, 40, 50, 60, 70))
        self.assertEqual(tuple(item.level_index for item in result.level_estimates), (1, 2))
        ground, story = result.level_estimates
        self.assertEqual(ground.analysis.average_height_m, 2.5)
        self.assertEqual(ground.analysis.outlier_frame_indices, (70,))
        self.assertEqual(story.analysis.average_height_m, 3.05)
        self.assertEqual(story.analysis.insufficient_frame_indices, (50,))
        self.assertEqual(result.transition_frame_indices, ())
        self.assertEqual(result.unassigned_frame_indices, ())

    def test_movement_never_changes_explicit_level_assignment(self) -> None:
        result = aggregate_marked_level_ceiling_height_estimates(
            (
                (1, 0, _movement("arrived_up", height=2.5)),
                (1, 15, _movement("ascending", height=2.6)),
            ),
            levels=((1, "Ground"), (2, "Unmarked story")),
        )

        self.assertEqual(len(result.level_estimates), 1)
        self.assertEqual(result.level_estimates[0].level_index, 1)
        self.assertEqual(result.level_estimates[0].analysis.average_height_m, 2.55)

    def test_one_sample_does_not_establish_a_level_consensus(self) -> None:
        result = aggregate_marked_level_ceiling_height_estimates(
            ((1, 7, _estimate(2.4)),),
            levels=((1, "Ground"),),
        )

        self.assertFalse(result.has_consensus)
        self.assertEqual(result.level_estimates[0].analysis.retained_frame_indices, (7,))

    def test_empty_marks_do_not_create_results_for_unmarked_levels(self) -> None:
        result = aggregate_marked_level_ceiling_height_estimates(
            (), levels=((1, "Ground"),)
        )

        self.assertEqual(result.level_estimates, ())
        self.assertEqual(result.sampled_frame_indices, ())

    def test_unknown_levels_and_duplicate_frames_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "no longer available"):
            aggregate_marked_level_ceiling_height_estimates(
                ((2, 0, _estimate(2.5)),), levels=((1, "Ground"),)
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            aggregate_marked_level_ceiling_height_estimates(
                ((1, 0, _estimate(2.5)), (2, 0, _estimate(3.0))),
                levels=((1, "Ground"), (2, "Story")),
            )


# ### Request tests ###
class CeilingHeightRequestTests(unittest.TestCase):
    def test_model_options_include_supported_vision_models(self) -> None:
        self.assertEqual(
            CEILING_HEIGHT_MODEL_OPTIONS,
            (
                ("GPT-5.6 Luna", "gpt-5.6-luna"),
                ("GPT-5.6 Terra", "gpt-5.6-terra"),
                ("Qwen3-VL 8B (Local)", LOCAL_QWEN_CEILING_HEIGHT_MODEL),
            ),
        )

    def test_only_local_qwen_skips_the_openai_api_key(self) -> None:
        self.assertFalse(
            ceiling_height_model_requires_api_key(
                LOCAL_QWEN_CEILING_HEIGHT_MODEL
            )
        )
        self.assertTrue(ceiling_height_model_requires_api_key("gpt-5.6-luna"))

    def test_cost_estimate_covers_paired_frames_and_paid_models_only(self) -> None:
        arguments = {
            "frame_count": 350,
            "interval": 15,
            "frame_width": 1_920,
            "frame_height": 1_080,
        }
        luna_cost = estimate_ceiling_height_api_cost_usd(
            model="gpt-5.6-luna",
            **arguments,
        )
        terra_cost = estimate_ceiling_height_api_cost_usd(
            model="gpt-5.6-terra",
            **arguments,
        )

        self.assertIsNotNone(luna_cost)
        self.assertIsNotNone(terra_cost)
        assert luna_cost is not None and terra_cost is not None
        self.assertGreater(luna_cost, 0.02)
        self.assertLess(luna_cost, 0.05)
        self.assertAlmostEqual(terra_cost, luna_cost * 10.0)
        plan_guided_cost = estimate_ceiling_height_api_cost_usd(
            model="gpt-5.6-luna",
            plan_reference_size=(1_200, 900),
            **arguments,
        )
        self.assertIsNotNone(plan_guided_cost)
        assert plan_guided_cost is not None
        self.assertGreater(plan_guided_cost, luna_cost)
        self.assertIsNone(
            estimate_ceiling_height_api_cost_usd(
                model=LOCAL_QWEN_CEILING_HEIGHT_MODEL,
                **arguments,
            )
        )

    def test_cost_override_uses_marked_sample_count_instead_of_whole_video(self) -> None:
        arguments = {
            "model": "gpt-5.6-luna",
            "frame_count": 350,
            "interval": 15,
            "frame_width": 1_920,
            "frame_height": 1_080,
        }
        full_cost = estimate_ceiling_height_api_cost_usd(**arguments)
        marked_cost = estimate_ceiling_height_api_cost_usd(**arguments, sample_count=4)
        matching_count_cost = estimate_ceiling_height_api_cost_usd(
            **{**arguments, "frame_count": 46}
        )

        self.assertIsNotNone(full_cost)
        self.assertIsNotNone(marked_cost)
        assert full_cost is not None and marked_cost is not None
        self.assertLess(marked_cost, full_cost)
        self.assertEqual(marked_cost, matching_count_cost)
        self.assertEqual(
            estimate_ceiling_height_api_cost_usd(**arguments, sample_count=0), 0.0
        )
        with self.assertRaises(ValueError):
            estimate_ceiling_height_api_cost_usd(**arguments, sample_count=-1)
        with self.assertRaises(TypeError):
            estimate_ceiling_height_api_cost_usd(**arguments, sample_count=True)

    def test_request_uses_vision_input_and_strict_structured_output(self) -> None:
        body = build_ceiling_height_request_body(
            "data:image/jpeg;base64,YWJj"
        )

        self.assertEqual(body["model"], CEILING_HEIGHT_INFERENCE_MODEL)
        self.assertFalse(body["store"])
        self.assertEqual(body["reasoning"], {"effort": "none"})
        content = body["input"][0]["content"]
        self.assertEqual(content[0]["type"], "input_text")
        self.assertIn("finished-floor", content[0]["text"])
        self.assertIn("any height", content[0]["text"])
        images = [item for item in content if item["type"] == "input_image"]
        self.assertEqual(
            images,
            [
                {
                    "type": "input_image",
                    "image_url": "data:image/jpeg;base64,YWJj",
                    "detail": "high",
                }
            ],
        )
        output_format = body["text"]["format"]
        self.assertEqual(output_format["type"], "json_schema")
        self.assertTrue(output_format["strict"])
        self.assertFalse(
            output_format["schema"]["additionalProperties"]
        )
        movement_schema = output_format["schema"]["properties"][
            "level_movement"
        ]
        self.assertEqual(
            set(movement_schema["enum"]),
            {
                "stable",
                "ascending",
                "descending",
                "arrived_up",
                "arrived_down",
                "uncertain",
            },
        )
        self.assertEqual(
            set(
                output_format["schema"]["properties"][
                    "movement_confidence"
                ]["enum"]
            ),
            {"low", "medium", "high"},
        )

    def test_request_orders_previous_frame_before_current_frame(self) -> None:
        body = build_ceiling_height_request_body(
            "data:image/jpeg;base64,Y3VycmVudA==",
            previous_image_data_uri="data:image/jpeg;base64,cHJldmlvdXM=",
        )

        content = body["input"][0]["content"]
        images = [item for item in content if item["type"] == "input_image"]
        self.assertEqual(
            [item["image_url"] for item in images],
            [
                "data:image/jpeg;base64,cHJldmlvdXM=",
                "data:image/jpeg;base64,Y3VycmVudA==",
            ],
        )

    def test_request_places_annotated_plan_before_video_frames(self) -> None:
        body = build_ceiling_height_request_body(
            "data:image/jpeg;base64,Y3VycmVudA==",
            previous_image_data_uri="data:image/jpeg;base64,cHJldmlvdXM=",
            plan_context="M1 - Salon width: 6.1 m.",
            plan_image_data_uri="data:image/jpeg;base64,cGxhbg==",
        )

        content = body["input"][0]["content"]
        images = [item for item in content if item["type"] == "input_image"]

        self.assertEqual(
            [item["image_url"] for item in images],
            [
                "data:image/jpeg;base64,cGxhbg==",
                "data:image/jpeg;base64,cHJldmlvdXM=",
                "data:image/jpeg;base64,Y3VycmVudA==",
            ],
        )
        self.assertIn("PLAN REFERENCE", content[1]["text"])

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
        image_input = next(
            item
            for item in request_body["input"][0]["content"]
            if item["type"] == "input_image"
        )
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
                        "level_movement": "stable",
                        "movement_confidence": "low",
                    }
                )
            ),
        )

        self.assertFalse(result.has_estimate)
        self.assertIn(
            "Could not estimate reliably",
            format_ceiling_height_estimate(result),
        )

    def test_insufficient_evidence_ignores_local_numeric_placeholders(self) -> None:
        result = infer_ceiling_height(
            np.zeros((8, 8, 3), dtype=np.uint8),
            api_key="sk-test",
            opener=lambda *_args, **_kwargs: _FakeResponse(
                _response_payload(
                    {
                        "status": "insufficient_evidence",
                        "estimate_m": 0.0,
                        "minimum_m": 0.0,
                        "maximum_m": 0.0,
                        "confidence": "low",
                        "basis": "The camera is between stable levels.",
                        "level_movement": "ascending",
                        "movement_confidence": "medium",
                    }
                )
            ),
        )

        self.assertFalse(result.has_estimate)
        self.assertIsNone(result.estimate_m)
        self.assertIsNone(result.minimum_m)
        self.assertIsNone(result.maximum_m)
        self.assertEqual(result.level_movement, "ascending")
        self.assertEqual(result.movement_confidence, "medium")

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
