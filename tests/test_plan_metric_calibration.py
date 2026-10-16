# ### Imports ###
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from housemaker.models import PIXEL_TO_METER
from housemaker.plan_image_correction import (
    STAGE_METRIC_CALIBRATION,
    calculate_output_size,
    correct_plan_image,
)
from housemaker.plan_metric_calibration import (
    PLAN_MEASUREMENT_PROMPT,
    PLAN_MEASUREMENT_REQUEST_NAME,
    PLAN_MEASUREMENT_RESULT_SCHEMA,
    PlanMetricCalibrationInferenceError,
    PlanMetricEvidenceSet,
    PlanMetricScaleSolution,
    build_plan_measurement_request_body,
    calibrate_and_normalize_plan,
    normalize_plan_raster,
    normalize_plan_with_scale_solution,
    parse_plan_measurement_result,
    solve_plan_metric_scale,
    solve_plan_metric_scale_batch,
)
from housemaker.plan_metric_state import (
    PlanMeasurementEvidence,
    PlanMetricCalibration,
)


# ### Fixture helpers ###
def _evidence(
    label: str,
    value_meters: float,
    corrected_start: tuple[float, float],
    corrected_end: tuple[float, float],
    *,
    confidence: str = "high",
) -> PlanMeasurementEvidence:
    return PlanMeasurementEvidence(
        label=label,
        value_meters=value_meters,
        start_normalized=corrected_start,
        end_normalized=corrected_end,
        corrected_start_normalized=corrected_start,
        corrected_end_normalized=corrected_end,
        confidence=confidence,
    )


def _measurement_payload(
    label: str,
    value_meters: float,
    corrected_start: tuple[float, float],
    corrected_end: tuple[float, float],
) -> dict[str, object]:
    return _evidence(
        label,
        value_meters,
        corrected_start,
        corrected_end,
    ).to_dict()


def _png_bytes(image: np.ndarray) -> bytes:
    encoded_ok, encoded = cv2.imencode(".png", image)
    assert encoded_ok
    return encoded.tobytes()


# ### Structured request tests ###
class PlanMeasurementRequestTests(unittest.TestCase):
    def test_request_contains_ordered_images_and_strict_schema(self) -> None:
        original = "data:image/png;base64,b3JpZ2luYWw="
        corrected = "data:image/png;base64,Y29ycmVjdGVk"

        body = build_plan_measurement_request_body(
            original,
            corrected,
            model="gpt-test",
        )

        self.assertEqual(body["model"], "gpt-test")
        self.assertFalse(body["store"])
        content = body["input"][0]["content"]
        image_inputs = [item for item in content if item["type"] == "input_image"]
        self.assertEqual(
            [item["image_url"] for item in image_inputs],
            [original, corrected],
        )
        self.assertTrue(all(item["detail"] == "original" for item in image_inputs))
        output_format = body["text"]["format"]
        self.assertEqual(output_format["name"], PLAN_MEASUREMENT_REQUEST_NAME)
        self.assertTrue(output_format["strict"])
        self.assertEqual(output_format["schema"], PLAN_MEASUREMENT_RESULT_SCHEMA)

    def test_request_places_optional_reference_between_raw_and_final(self) -> None:
        original = "data:image/png;base64,b3JpZ2luYWw="
        reference = "data:image/png;base64,cmVmZXJlbmNl"
        corrected = "data:image/png;base64,Y29ycmVjdGVk"

        body = build_plan_measurement_request_body(
            original,
            corrected,
            measurement_reference_image_data_uri=reference,
        )

        content = body["input"][0]["content"]
        self.assertEqual(
            [
                item["image_url"]
                for item in content
                if item["type"] == "input_image"
            ],
            [original, reference, corrected],
        )
        labels = [item["text"] for item in content if item["type"] == "input_text"]
        self.assertIn("ORIGINAL PLAN (RAW):", labels)
        self.assertIn("RECTIFIED ANNOTATED PLAN:", labels)
        self.assertIn("CORRECTED PLAN (FINAL):", labels)

    def test_parser_requires_both_corrected_endpoints(self) -> None:
        payload = _measurement_payload("610", 6.1, (0.1, 0.2), (0.71, 0.2))
        payload.pop("corrected_end_normalized")

        with self.assertRaisesRegex(
            PlanMetricCalibrationInferenceError,
            "invalid evidence",
        ):
            parse_plan_measurement_result({"measurements": [payload]})

    def test_prompt_warns_against_blind_millimetre_conversion(self) -> None:
        self.assertIn("unit convention", PLAN_MEASUREMENT_PROMPT)
        self.assertIn("610 and 390", PLAN_MEASUREMENT_PROMPT)
        self.assertIn("6100 and 3900", PLAN_MEASUREMENT_PROMPT)
        self.assertIn("authoritative source", PLAN_MEASUREMENT_PROMPT)
        self.assertIn("readings disagree", PLAN_MEASUREMENT_PROMPT)


# ### Robust calibration tests ###
class PlanMetricScaleTests(unittest.TestCase):
    def test_solver_uses_both_pixel_axes_and_rejects_large_outlier(self) -> None:
        measurements = (
            _evidence("horizontal", 4.0, (0.1, 0.1), (0.5, 0.1)),
            _evidence("vertical", 3.0, (0.2, 0.1), (0.2, 0.7)),
            _evidence("second horizontal", 6.0, (0.2, 0.8), (0.8, 0.8)),
            _evidence("second vertical", 2.0, (0.8, 0.2), (0.8, 0.6)),
            _evidence("bad OCR units", 40.0, (0.1, 0.4), (0.5, 0.4)),
        )

        estimate, residual, confidence, accepted = solve_plan_metric_scale(
            measurements,
            corrected_image_size=(1_000, 500),
        )

        self.assertAlmostEqual(estimate, 0.01, places=8)
        self.assertAlmostEqual(residual, 0.0, places=8)
        self.assertEqual(confidence, "high")
        self.assertEqual(
            {item.label for item in accepted},
            {"horizontal", "vertical", "second horizontal", "second vertical"},
        )

    def test_solver_rejects_two_incompatible_measurements(self) -> None:
        measurements = (
            _evidence("first", 4.0, (0.1, 0.1), (0.5, 0.1)),
            _evidence("second", 8.0, (0.1, 0.2), (0.5, 0.2)),
        )

        with self.assertRaisesRegex(
            PlanMetricCalibrationInferenceError,
            "reliable scale",
        ):
            solve_plan_metric_scale(
                measurements,
                corrected_image_size=(1_000, 500),
            )

    def test_solver_rejects_inconsistent_horizontal_and_vertical_scale(self) -> None:
        measurements = (
            _evidence("h1", 4.0, (0.1, 0.1), (0.5, 0.1)),
            _evidence("h2", 6.0, (0.2, 0.2), (0.8, 0.2)),
            _evidence("v1", 3.33, (0.1, 0.1), (0.1, 0.7)),
            _evidence("v2", 2.22, (0.2, 0.3), (0.2, 0.7)),
        )

        with self.assertRaisesRegex(
            PlanMetricCalibrationInferenceError,
            "horizontal and vertical scale",
        ):
            solve_plan_metric_scale(
                measurements,
                corrected_image_size=(1_000, 500),
            )

    def test_single_axis_evidence_cannot_claim_high_confidence(self) -> None:
        measurements = (
            _evidence("room width", 4.0, (0.1, 0.1), (0.5, 0.1)),
            _evidence("hall width", 6.0, (0.2, 0.3), (0.8, 0.3)),
            _evidence("kitchen width", 3.0, (0.3, 0.5), (0.6, 0.5)),
        )

        _estimate, _residual, confidence, _accepted = solve_plan_metric_scale(
            measurements,
            corrected_image_size=(1_000, 500),
        )

        self.assertEqual(confidence, "medium")

    def test_one_measurement_is_not_allowed_to_rescale_the_plan(self) -> None:
        with self.assertRaisesRegex(
            PlanMetricCalibrationInferenceError,
            "two consistent, reliable",
        ):
            solve_plan_metric_scale(
                (_evidence("610", 6.1, (0.1, 0.2), (0.71, 0.2)),),
                corrected_image_size=(1_000, 800),
            )

    def test_duplicate_measurement_span_does_not_count_twice(self) -> None:
        duplicate_spans = (
            _evidence("610", 6.1, (0.1, 0.2), (0.71, 0.2)),
            _evidence("610 repeated", 6.1, (0.101, 0.201), (0.711, 0.201)),
        )

        with self.assertRaisesRegex(
            PlanMetricCalibrationInferenceError,
            "two consistent, reliable",
        ):
            solve_plan_metric_scale(
                duplicate_spans,
                corrected_image_size=(1_000, 800),
            )


# ### Cross-plan reconciliation tests ###
class PlanMetricBatchScaleTests(unittest.TestCase):
    def test_batch_preserves_different_local_pixel_scales(self) -> None:
        first = PlanMetricEvidenceSet(
            plan_key="ground",
            corrected_image_size=(1_000, 800),
            measurements=(
                _evidence(
                    "Salon interior width walls",
                    4.0,
                    (0.1, 0.2),
                    (0.5, 0.2),
                ),
                _evidence(
                    "Salon interior depth walls",
                    3.0,
                    (0.2, 0.1),
                    (0.2, 0.475),
                ),
            ),
        )
        second = PlanMetricEvidenceSet(
            plan_key="story-1",
            corrected_image_size=(2_000, 1_000),
            measurements=(
                _evidence(
                    "Salon interior width walls",
                    4.0,
                    (0.1, 0.2),
                    (0.2, 0.2),
                ),
                _evidence(
                    "Salon interior depth walls",
                    3.0,
                    (0.2, 0.1),
                    (0.2, 0.25),
                ),
            ),
        )

        result = solve_plan_metric_scale_batch((first, second))

        self.assertEqual(result.failures, ())
        solutions = {item.plan_key: item for item in result.solutions}
        self.assertAlmostEqual(
            solutions["ground"].estimated_meters_per_pixel,
            0.01,
            places=8,
        )
        self.assertAlmostEqual(
            solutions["story-1"].estimated_meters_per_pixel,
            0.02,
            places=8,
        )
        self.assertEqual(
            solutions["ground"].corroborated_measurement_count,
            2,
        )
        self.assertEqual(
            solutions["story-1"].corroborated_measurement_count,
            2,
        )

    def test_corroboration_breaks_an_otherwise_ambiguous_local_tie(self) -> None:
        ambiguous = PlanMetricEvidenceSet(
            plan_key="ambiguous",
            corrected_image_size=(1_000, 800),
            measurements=(
                _evidence(
                    "Interior width shared salon",
                    4.0,
                    (0.1, 0.1),
                    (0.5, 0.1),
                ),
                _evidence(
                    "Interior depth shared salon",
                    3.0,
                    (0.1, 0.2),
                    (0.1, 0.575),
                ),
                _evidence(
                    "Unrelated building outer width",
                    8.0,
                    (0.1, 0.7),
                    (0.5, 0.7),
                ),
                _evidence(
                    "Unrelated building outer depth",
                    6.0,
                    (0.8, 0.1),
                    (0.8, 0.475),
                ),
            ),
        )
        supporting = PlanMetricEvidenceSet(
            plan_key="supporting",
            corrected_image_size=(1_000, 800),
            measurements=(
                _evidence(
                    "Shared salon interior width",
                    4.0,
                    (0.1, 0.1),
                    (0.5, 0.1),
                ),
                _evidence(
                    "Shared salon interior depth",
                    3.0,
                    (0.1, 0.2),
                    (0.1, 0.575),
                ),
            ),
        )
        with self.assertRaisesRegex(
            PlanMetricCalibrationInferenceError,
            "reliable scale",
        ):
            solve_plan_metric_scale(
                ambiguous.measurements,
                corrected_image_size=ambiguous.corrected_image_size,
            )

        result = solve_plan_metric_scale_batch((ambiguous, supporting))

        self.assertEqual(result.failures, ())
        solution = next(
            item for item in result.solutions if item.plan_key == "ambiguous"
        )
        self.assertAlmostEqual(
            solution.estimated_meters_per_pixel,
            0.01,
            places=8,
        )
        self.assertEqual(solution.corroborated_measurement_count, 2)

    def test_batch_rejects_corroborated_power_of_ten_unit_mismatch(self) -> None:
        plans = (
            PlanMetricEvidenceSet(
                plan_key="ground",
                corrected_image_size=(1_000, 800),
                measurements=(
                    _evidence(
                        "Shared facade overall width",
                        40.0,
                        (0.1, 0.1),
                        (0.5, 0.1),
                    ),
                    _evidence(
                        "Ground room interior width",
                        4.0,
                        (0.1, 0.3),
                        (0.5, 0.3),
                    ),
                    _evidence(
                        "Ground room interior depth",
                        3.0,
                        (0.2, 0.1),
                        (0.2, 0.475),
                    ),
                ),
            ),
            PlanMetricEvidenceSet(
                plan_key="story-1",
                corrected_image_size=(1_000, 800),
                measurements=(
                    _evidence(
                        "Shared facade overall width",
                        4.0,
                        (0.1, 0.1),
                        (0.5, 0.1),
                    ),
                    _evidence(
                        "Story room interior depth",
                        3.0,
                        (0.2, 0.1),
                        (0.2, 0.475),
                    ),
                ),
            ),
            PlanMetricEvidenceSet(
                plan_key="underground",
                corrected_image_size=(1_000, 800),
                measurements=(
                    _evidence(
                        "Shared facade overall width",
                        4.0,
                        (0.1, 0.1),
                        (0.5, 0.1),
                    ),
                    _evidence(
                        "Basement room interior depth",
                        3.0,
                        (0.2, 0.1),
                        (0.2, 0.475),
                    ),
                ),
            ),
        )

        result = solve_plan_metric_scale_batch(plans)

        self.assertEqual(result.failures, ())
        ground = next(
            item for item in result.solutions if item.plan_key == "ground"
        )
        self.assertAlmostEqual(
            ground.estimated_meters_per_pixel,
            0.01,
            places=8,
        )
        self.assertEqual(ground.rejected_unit_mismatch_count, 1)

    def test_cross_plan_support_does_not_replace_two_local_spans(self) -> None:
        weak = PlanMetricEvidenceSet(
            plan_key="weak",
            corrected_image_size=(1_000, 800),
            measurements=(
                _evidence(
                    "Shared salon interior width",
                    4.0,
                    (0.1, 0.1),
                    (0.5, 0.1),
                ),
            ),
        )
        strong = PlanMetricEvidenceSet(
            plan_key="strong",
            corrected_image_size=(1_000, 800),
            measurements=(
                _evidence(
                    "Shared salon interior width",
                    4.0,
                    (0.1, 0.1),
                    (0.5, 0.1),
                ),
                _evidence(
                    "Strong salon interior depth",
                    3.0,
                    (0.2, 0.1),
                    (0.2, 0.475),
                ),
            ),
        )

        result = solve_plan_metric_scale_batch((weak, strong))

        self.assertEqual(
            tuple(item.plan_key for item in result.solutions),
            ("strong",),
        )
        self.assertEqual(
            tuple(item.plan_key for item in result.failures),
            ("weak",),
        )
        self.assertIn("two consistent", result.failures[0].reason)


# ### Persisted state validation tests ###
class PlanMetricStateValidationTests(unittest.TestCase):
    @staticmethod
    def _measurements() -> tuple[PlanMeasurementEvidence, ...]:
        return (
            _evidence("first", 4.0, (0.1, 0.2), (0.5, 0.2)),
            _evidence("second", 3.0, (0.2, 0.3), (0.2, 0.6)),
        )

    def test_native_density_record_round_trips_without_raster_resize(self) -> None:
        calibration = PlanMetricCalibration(
            estimated_meters_per_pixel=0.0075,
            target_meters_per_pixel=0.0075,
            normalization_scale=1.0,
            relative_residual=0.01,
            confidence="medium",
            source_image_size=(2_000, 1_000),
            pre_normalization_size=(1_600, 800),
            normalized_image_size=(1_600, 800),
            measurements=self._measurements(),
        )

        restored = PlanMetricCalibration.from_dict(calibration.to_dict())

        self.assertEqual(restored, calibration)
        self.assertEqual(restored.normalization_scale, 1.0)
        self.assertEqual(restored.target_meters_per_pixel, 0.0075)

    def test_legacy_resampled_record_retains_current_image_density(self) -> None:
        calibration = PlanMetricCalibration(
            estimated_meters_per_pixel=0.01,
            target_meters_per_pixel=0.02,
            normalization_scale=0.5,
            relative_residual=0.01,
            confidence="medium",
            source_image_size=(2_000, 1_000),
            pre_normalization_size=(2_000, 1_000),
            normalized_image_size=(1_000, 500),
            measurements=self._measurements(),
        )
        payload = calibration.to_dict()
        payload.pop("diagnostics")

        restored = PlanMetricCalibration.from_dict(payload)

        self.assertEqual(restored, calibration)
        self.assertEqual(restored.version, 1)
        self.assertEqual(restored.target_meters_per_pixel, 0.02)
        self.assertEqual(restored.normalized_image_size, (1_000, 500))

    def test_calibration_requires_density_ratio_as_normalization_scale(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "pixel densities"):
            PlanMetricCalibration(
                estimated_meters_per_pixel=0.01,
                target_meters_per_pixel=0.02,
                normalization_scale=0.75,
                relative_residual=0.01,
                confidence="high",
                source_image_size=(2_000, 1_000),
                pre_normalization_size=(2_000, 1_000),
                normalized_image_size=(1_500, 750),
                measurements=self._measurements(),
            )

    def test_calibration_requires_normalized_raster_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "Normalized plan size"):
            PlanMetricCalibration(
                estimated_meters_per_pixel=0.01,
                target_meters_per_pixel=0.02,
                normalization_scale=0.5,
                relative_residual=0.01,
                confidence="high",
                source_image_size=(2_000, 1_000),
                pre_normalization_size=(2_000, 1_000),
                normalized_image_size=(999, 500),
                measurements=self._measurements(),
            )

    def test_calibration_requires_two_independent_spans(self) -> None:
        first = _evidence("first", 4.0, (0.1, 0.2), (0.5, 0.2))
        duplicate = _evidence(
            "duplicate",
            4.0,
            (0.101, 0.201),
            (0.501, 0.201),
        )

        with self.assertRaisesRegex(ValueError, "independent"):
            PlanMetricCalibration(
                estimated_meters_per_pixel=0.01,
                target_meters_per_pixel=0.02,
                normalization_scale=0.5,
                relative_residual=0.01,
                confidence="high",
                source_image_size=(2_000, 1_000),
                pre_normalization_size=(2_000, 1_000),
                normalized_image_size=(1_000, 500),
                measurements=(first, duplicate),
            )

    def test_solver_rejects_consistent_centimetres_mistaken_for_millimetres(
        self,
    ) -> None:
        measurements = (
            _evidence("room width", 0.61, (0.1, 0.2), (0.71, 0.2)),
            _evidence("room depth", 0.39, (0.2, 0.1), (0.2, 0.62)),
            _evidence("hall width", 0.36, (0.2, 0.8), (0.56, 0.8)),
        )

        with self.assertRaisesRegex(
            PlanMetricCalibrationInferenceError,
            "centimetre or millimetre",
        ):
            solve_plan_metric_scale(
                measurements,
                corrected_image_size=(2_000, 1_500),
            )

    def test_solver_accepts_a_compact_but_plausible_plan(self) -> None:
        measurements = (
            _evidence("compact room width", 1.5, (0.1, 0.2), (0.7, 0.2)),
            _evidence("compact room depth", 1.0, (0.2, 0.2), (0.2, 0.7)),
        )

        estimate, _residual, _confidence, accepted = solve_plan_metric_scale(
            measurements,
            corrected_image_size=(1_000, 800),
        )

        self.assertAlmostEqual(estimate, 0.0025, places=8)
        self.assertEqual(len(accepted), 2)

    def test_solver_rejects_consistent_units_that_make_plan_implausibly_large(
        self,
    ) -> None:
        measurements = (
            _evidence("first span", 400.0, (0.1, 0.2), (0.5, 0.2)),
            _evidence("second span", 300.0, (0.2, 0.3), (0.5, 0.3)),
        )

        with self.assertRaisesRegex(
            PlanMetricCalibrationInferenceError,
            "implausibly large",
        ):
            solve_plan_metric_scale(
                measurements,
                corrected_image_size=(1_000, 800),
            )


# ### Raster normalization tests ###
class PlanRasterNormalizationTests(unittest.TestCase):
    def test_native_calibration_retains_every_pixel_and_uses_local_density(self) -> None:
        image = np.arange(60 * 100 * 3, dtype=np.uint8).reshape(60, 100, 3)
        measurements = (
            _evidence("three metres", 3.0, (0.1, 0.2), (0.5, 0.2)),
            _evidence("two metres", 2.0, (0.2, 0.3), (0.2, 0.6)),
        )
        solution = PlanMetricScaleSolution(
            plan_key="native",
            estimated_meters_per_pixel=0.0075,
            relative_residual=0.01,
            confidence="medium",
            measurements=measurements,
        )

        result = normalize_plan_with_scale_solution(image, image, solution)

        np.testing.assert_array_equal(result.image, image)
        self.assertFalse(np.shares_memory(result.image, image))
        self.assertEqual(result.calibration.target_meters_per_pixel, 0.0075)
        self.assertEqual(result.calibration.normalization_scale, 1.0)
        self.assertEqual(result.calibration.normalized_image_size, (100, 60))

    def test_explicit_legacy_density_remains_opt_in(self) -> None:
        image = np.full((80, 100, 3), 255, dtype=np.uint8)
        solution = PlanMetricScaleSolution(
            plan_key="legacy",
            estimated_meters_per_pixel=0.01,
            relative_residual=0.01,
            confidence="medium",
            measurements=(
                _evidence("first", 3.0, (0.1, 0.2), (0.5, 0.2)),
                _evidence("second", 2.0, (0.2, 0.3), (0.2, 0.6)),
            ),
        )

        result = normalize_plan_with_scale_solution(
            image, image, solution, target_meters_per_pixel=0.02
        )

        self.assertEqual(result.image.shape, (40, 50, 3))
        self.assertEqual(result.calibration.target_meters_per_pixel, 0.02)
        self.assertEqual(result.calibration.normalization_scale, 0.5)

    def test_normalization_resizes_the_complete_raster_uniformly(self) -> None:
        image = np.full((80, 100, 3), 255, dtype=np.uint8)
        image[10:20, 10:20] = 0
        normalized, scale = normalize_plan_raster(
            image,
            estimated_meters_per_pixel=0.01,
        )

        self.assertAlmostEqual(scale, 0.5, places=8)
        self.assertEqual(normalized.shape, (40, 50, 3))
        self.assertEqual(normalized.dtype, np.uint8)
        self.assertLess(int(normalized[7, 7, 0]), 255)

    def test_synthetic_span_keeps_its_metric_length_after_normalization(self) -> None:
        image = np.full((600, 1_000), 255, dtype=np.uint8)
        measurements = (
            _evidence("five metres", 5.0, (0.2, 0.4), (0.7, 0.4)),
            _evidence("three metres", 3.0, (0.3, 0.6), (0.6, 0.6)),
        )
        estimate, _residual, _confidence, _accepted = solve_plan_metric_scale(
            measurements,
            corrected_image_size=(1_000, 600),
        )
        normalized, scale = normalize_plan_raster(
            image,
            estimated_meters_per_pixel=estimate,
        )

        self.assertAlmostEqual(estimate, 0.01, places=8)
        self.assertEqual(normalized.shape, (300, 500))
        normalized_span = 500 * (0.7 - 0.2)
        self.assertAlmostEqual(
            normalized_span * PIXEL_TO_METER,
            5.0,
            places=8,
        )
        self.assertAlmostEqual(scale, 0.5, places=8)


# ### Pipeline integration tests ###
class PlanMetricPipelineIntegrationTests(unittest.TestCase):
    def test_calibration_analyzer_receives_images_without_resampling(self) -> None:
        source = np.full((400, 600, 3), 255, dtype=np.uint8)
        corrected = source.copy()
        calls: list[dict[str, object]] = []

        def analyzer(
            original_uri: str,
            corrected_uri: str,
            **kwargs: object,
        ) -> object:
            calls.append(
                {
                    "original_uri": original_uri,
                    "corrected_uri": corrected_uri,
                    **kwargs,
                }
            )
            return {
                "measurements": [
                    _measurement_payload(
                        "three metres",
                        3.0,
                        (0.1, 0.2),
                        (0.6, 0.2),
                    ),
                    _measurement_payload(
                        "two metres",
                        2.0,
                        (0.2, 0.4),
                        (0.2 + (1.0 / 3.0), 0.4),
                    ),
                ]
            }

        result = calibrate_and_normalize_plan(source, corrected, analyzer)

        self.assertEqual(result.image.shape[:2], corrected.shape[:2])
        np.testing.assert_array_equal(result.image, corrected)
        self.assertEqual(result.calibration.normalization_scale, 1.0)
        self.assertAlmostEqual(result.calibration.target_meters_per_pixel, 0.01)
        self.assertEqual(result.calibration.pre_normalization_size, (600, 400))
        self.assertEqual(result.calibration.normalized_image_size, (600, 400))
        self.assertAlmostEqual(
            result.calibration.estimated_meters_per_pixel,
            0.01,
            places=8,
        )
        self.assertEqual(len(calls), 2)
        self.assertTrue(
            str(calls[0]["original_uri"]).startswith("data:image/png;base64,")
        )
        self.assertTrue(
            str(calls[0]["corrected_uri"]).startswith("data:image/png;base64,")
        )
        self.assertEqual(calls[0]["prompt"], PLAN_MEASUREMENT_PROMPT)
        self.assertEqual(calls[0]["result_schema"], PLAN_MEASUREMENT_RESULT_SCHEMA)

    def test_reference_aids_transcription_without_replacing_raw_metadata(self) -> None:
        source = np.full((400, 600, 3), 32, dtype=np.uint8)
        reference = np.full((300, 500, 3), 160, dtype=np.uint8)
        corrected = np.full((400, 600, 3), 255, dtype=np.uint8)
        calls: list[dict[str, object]] = []

        def analyzer(
            original_uri: str,
            corrected_uri: str,
            **kwargs: object,
        ) -> object:
            calls.append(
                {
                    "original_uri": original_uri,
                    "corrected_uri": corrected_uri,
                    **kwargs,
                }
            )
            return {
                "measurements": [
                    _measurement_payload(
                        "three metres",
                        3.0,
                        (0.1, 0.2),
                        (0.6, 0.2),
                    ),
                    _measurement_payload(
                        "two metres",
                        2.0,
                        (0.2, 0.4),
                        (0.2 + (1.0 / 3.0), 0.4),
                    ),
                ]
            }

        result = calibrate_and_normalize_plan(
            source,
            corrected,
            analyzer,
            measurement_reference_image=reference,
        )

        self.assertEqual(len(calls), 2)
        self.assertIn("measurement_reference_image_data_uri", calls[0])
        self.assertNotEqual(
            calls[0]["original_uri"],
            calls[0]["measurement_reference_image_data_uri"],
        )
        self.assertEqual(result.calibration.source_image_size, (600, 400))

    def test_plan_correction_exposes_optional_calibration_and_saved_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            input_path = directory / "photo.png"
            output_path = directory / "corrected.png"
            source = np.full((480, 640, 3), 255, dtype=np.uint8)
            cv2.rectangle(source, (80, 80), (560, 400), (0, 0, 0), 5)
            Image.fromarray(cv2.cvtColor(source, cv2.COLOR_BGR2RGB)).save(input_path)
            editor_size = calculate_output_size(640, 480)
            editor_image = cv2.resize(source, editor_size)

            def editor(*_args: object, **_kwargs: object) -> bytes:
                return _png_bytes(editor_image)

            def analyzer(*_args: object, **_kwargs: object) -> object:
                measured_pixels = editor_size[0] * 0.5
                return {
                    "measurements": [
                        _measurement_payload(
                            "known span",
                            measured_pixels * 0.01,
                            (0.2, 0.3),
                            (0.7, 0.3),
                        ),
                        _measurement_payload(
                            "second known span",
                            measured_pixels * 0.006,
                            (0.1, 0.6),
                            (0.4, 0.6),
                        ),
                    ]
                }

            result = correct_plan_image(
                input_path,
                output_path,
                api_key="test-key",
                image_editor=editor,
                measurement_analyzer=analyzer,
            )

            self.assertIsNotNone(result.metric_calibration)
            assert result.metric_calibration is not None
            self.assertEqual(
                len(result.metric_calibration.normalized_image_sha256),
                64,
            )
            self.assertIn(STAGE_METRIC_CALIBRATION, result.stages)
            self.assertEqual(
                result.output_size,
                editor_size,
            )
            with Image.open(output_path) as saved:
                self.assertEqual(saved.size, result.output_size)

    def test_failed_metric_consensus_keeps_the_corrected_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            input_path = directory / "photo.png"
            output_path = directory / "corrected.png"
            source = np.full((480, 640, 3), 255, dtype=np.uint8)
            cv2.rectangle(source, (80, 80), (560, 400), (0, 0, 0), 5)
            Image.fromarray(cv2.cvtColor(source, cv2.COLOR_BGR2RGB)).save(input_path)
            editor_size = calculate_output_size(640, 480)
            editor_image = cv2.resize(source, editor_size)

            def editor(*_args: object, **_kwargs: object) -> bytes:
                return _png_bytes(editor_image)

            def weak_analyzer(*_args: object, **_kwargs: object) -> object:
                return {
                    "measurements": [
                        _measurement_payload(
                            "single span",
                            3.0,
                            (0.1, 0.2),
                            (0.6, 0.2),
                        )
                    ]
                }

            result = correct_plan_image(
                input_path,
                output_path,
                api_key="test-key",
                image_editor=editor,
                measurement_analyzer=weak_analyzer,
            )

            self.assertTrue(output_path.is_file())
            self.assertIsNone(result.metric_calibration)
            self.assertIn("two consistent, reliable", result.metric_warning)
            self.assertEqual(result.output_size, editor_size)

    def test_consistent_wrong_units_do_not_shrink_the_corrected_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            input_path = directory / "photo.png"
            output_path = directory / "corrected.png"
            source = np.full((480, 640, 3), 255, dtype=np.uint8)
            cv2.rectangle(source, (80, 80), (560, 400), (0, 0, 0), 5)
            Image.fromarray(cv2.cvtColor(source, cv2.COLOR_BGR2RGB)).save(input_path)
            editor_size = calculate_output_size(640, 480)
            editor_image = cv2.resize(source, editor_size)
            wrong_density = 0.001

            def editor(*_args: object, **_kwargs: object) -> bytes:
                return _png_bytes(editor_image)

            def analyzer(*_args: object, **_kwargs: object) -> object:
                width, height = editor_size
                return {
                    "measurements": [
                        _measurement_payload(
                            "room width",
                            0.61,
                            (0.1, 0.2),
                            (0.1 + 0.61 / (width * wrong_density), 0.2),
                        ),
                        _measurement_payload(
                            "room depth",
                            0.39,
                            (0.2, 0.1),
                            (0.2, 0.1 + 0.39 / (height * wrong_density)),
                        ),
                    ]
                }

            result = correct_plan_image(
                input_path,
                output_path,
                api_key="test-key",
                image_editor=editor,
                measurement_analyzer=analyzer,
            )

            self.assertIsNone(result.metric_calibration)
            self.assertIn("centimetre or millimetre", result.metric_warning)
            self.assertEqual(result.output_size, editor_size)
            with Image.open(output_path) as saved:
                self.assertEqual(saved.size, editor_size)


if __name__ == "__main__":
    unittest.main()
