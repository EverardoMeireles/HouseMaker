# ### Imports ###
from __future__ import annotations

import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from PIL import Image

from housemaker.models import LevelData
from housemaker.plan_correction_exchange import ExternalPlanCorrection
from housemaker.plan_correction_import import (
    align_external_plan_levels,
    apply_external_plan_metadata,
    matches_external_plan_image,
    prepare_external_plan_calibration,
)
from housemaker.plan_metric_state import PlanMeasurementEvidence


# ### Test helpers ###
def _correction(
    level_index: int = 1,
    image_size: tuple[int, int] = (100, 200),
    alignment_point: tuple[float, float] = (0.2, 0.3),
) -> ExternalPlanCorrection:
    return ExternalPlanCorrection(
        level_index=level_index,
        image_filename=f"L{level_index}_corrected.png",
        image_size=image_size,
        meters_per_pixel=0.01,
        confidence="high",
        alignment_point_normalized=alignment_point,
        reference_level_index=2,
    )


def _measurement(
    label: str,
    value_meters: float,
    start: tuple[float, float],
    end: tuple[float, float],
) -> PlanMeasurementEvidence:
    return PlanMeasurementEvidence(
        label=label,
        value_meters=value_meters,
        start_normalized=start,
        end_normalized=end,
        corrected_start_normalized=start,
        corrected_end_normalized=end,
        confidence="high",
    )


def _level(correction: ExternalPlanCorrection) -> LevelData:
    return LevelData(
        index=correction.level_index,
        name=f"Level {correction.level_index}",
        image_path=correction.image_filename,
        image_size_pixels=correction.image_size,
        external_plan_correction=correction,
    )


# ### Corrected-image binding tests ###
class ExternalPlanImageBindingTests(unittest.TestCase):
    def test_matching_requires_corrected_basename_and_exact_dimensions(self) -> None:
        correction = _correction()
        self.assertTrue(matches_external_plan_image(
            correction, "somewhere/l1_CORRECTED.PNG", (100.0, 200.0)
        ))
        for path, size in (
            ("L1_original.jpg", (100, 200)),
            ("L1_corrected.png", (200, 100)),
            ("L1_corrected.png", (100, 201)),
            (None, (100, 200)),
            ("L1_corrected.png", None),
        ):
            with self.subTest(path=path, size=size):
                self.assertFalse(matches_external_plan_image(correction, path, size))

    def test_file_dimensions_are_verified_instead_of_trusting_ui_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            correction = _correction()
            path = Path(directory) / correction.image_filename
            Image.new("RGB", (101, 200), "white").save(path)

            with self.assertRaisesRegex(ValueError, "100 x 200"):
                prepare_external_plan_calibration(correction, path)

    def test_calibration_keeps_native_resolution_and_records_image_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            correction = replace(_correction(), measurements=(
                _measurement("80 cm", 0.8, (0.1, 0.2), (0.9, 0.2)),
                _measurement("160 cm", 1.6, (0.2, 0.1), (0.2, 0.9)),
            ))
            path = Path(directory) / correction.image_filename
            Image.new("RGB", correction.image_size, "white").save(path)
            original_bytes = path.read_bytes()

            calibration = prepare_external_plan_calibration(correction, path)
            level = _level(correction)
            level.scale = 1.75
            level.canvas_level_scale = 1.5
            level.height_meters = 2.45
            apply_external_plan_metadata(level, correction, calibration)

            self.assertIsNotNone(calibration)
            assert calibration is not None
            self.assertEqual(calibration.normalized_image_size, (100, 200))
            self.assertEqual(calibration.normalization_scale, 1.0)
            self.assertAlmostEqual(calibration.relative_residual, 0.0)
            self.assertEqual(calibration.target_meters_per_pixel, 0.01)
            self.assertEqual(calibration.normalized_image_sha256,
                             hashlib.sha256(original_bytes).hexdigest())
            self.assertEqual(path.read_bytes(), original_bytes)
            self.assertEqual(level.meters_per_pixel, 0.01)
            self.assertEqual(level.scale, 1.0)
            self.assertEqual(level.canvas_level_scale, 1.5)
            self.assertEqual(level.height_meters, 2.45)

    def test_density_without_independent_measurements_is_retained_without_evidence(self) -> None:
        measurement = _measurement("80 cm", 0.8, (0.1, 0.2), (0.9, 0.2))
        for measurements in ((), (measurement,), (measurement, measurement)):
            with (
                self.subTest(measurements=measurements),
                tempfile.TemporaryDirectory() as directory,
            ):
                correction = replace(_correction(), measurements=measurements)
                path = Path(directory) / correction.image_filename
                Image.new("RGB", correction.image_size, "white").save(path)

                calibration = prepare_external_plan_calibration(correction, path)
                level = _level(correction)
                apply_external_plan_metadata(level, correction, calibration)

                self.assertIsNone(calibration)
                self.assertIsNone(level.plan_metric_calibration)
                self.assertEqual(level.plan_meters_per_pixel, 0.01)
                self.assertEqual(level.meters_per_pixel, 0.01)

    def test_unknown_density_does_not_invent_calibration_or_reset_manual_scale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            correction = replace(_correction(), meters_per_pixel=None)
            path = Path(directory) / correction.image_filename
            Image.new("RGB", correction.image_size, "white").save(path)
            level = _level(correction)
            level.scale = 1.75

            calibration = prepare_external_plan_calibration(correction, path)
            apply_external_plan_metadata(level, correction, calibration)

            self.assertIsNone(calibration)
            self.assertIsNone(level.plan_meters_per_pixel)
            self.assertEqual(level.scale, 1.75)


# ### Offset-only alignment tests ###
class ExternalPlanAlignmentTests(unittest.TestCase):
    def test_different_raster_sizes_align_landmarks_without_changing_scales(self) -> None:
        reference = _level(_correction(2, (1000, 500), (0.25, 0.4)))
        target = _level(_correction(1, (800, 1600), (0.6, 0.25)))
        reference.canvas_level_scale = 1.5
        reference.canvas_offset_x_pixels = 25.0
        reference.canvas_offset_y_pixels = -35.0
        reference.scale = 1.4
        target.canvas_level_scale = 0.75
        target.scale = 1.8
        target.offset_x_meters = 3.0
        target.offset_y_meters = -4.0
        target.height_meters = 2.45

        align_external_plan_levels([target, reference], (1000.0, 800.0))

        # Reference displayed landmark: (-375, -75) + offset (25, -35).
        # Target displayed landmark before offset: (30, -150).
        self.assertAlmostEqual(target.canvas_offset_x_pixels, -380.0)
        self.assertAlmostEqual(target.canvas_offset_y_pixels, 40.0)
        self.assertEqual(reference.canvas_offset_x_pixels, 25.0)
        self.assertEqual(reference.canvas_offset_y_pixels, -35.0)
        self.assertEqual(reference.canvas_level_scale, 1.5)
        self.assertEqual(target.canvas_level_scale, 0.75)
        self.assertEqual(reference.scale, 1.4)
        self.assertEqual(target.scale, 1.8)
        self.assertEqual(target.offset_x_meters, 3.0)
        self.assertEqual(target.offset_y_meters, -4.0)
        self.assertEqual(target.height_meters, 2.45)
        align_external_plan_levels([target, reference], (1000.0, 800.0))
        self.assertAlmostEqual(target.canvas_offset_x_pixels, -380.0)
        self.assertAlmostEqual(target.canvas_offset_y_pixels, 40.0)

    def test_alignment_waits_for_both_manually_loaded_corrected_images(self) -> None:
        for pending_level_index in (1, 2):
            with self.subTest(pending=pending_level_index):
                reference = _level(_correction(2, (1000, 500), (0.25, 0.4)))
                target = _level(_correction(1, (800, 1600), (0.6, 0.25)))
                target.canvas_offset_x_pixels = 11.0
                target.canvas_offset_y_pixels = -12.0
                pending = target if pending_level_index == 1 else reference
                pending.image_path = f"L{pending_level_index}_original.jpg"

                align_external_plan_levels([reference, target], (1000.0, 800.0))

                self.assertEqual(target.canvas_offset_x_pixels, 11.0)
                self.assertEqual(target.canvas_offset_y_pixels, -12.0)
                pending.image_path = f"L{pending_level_index}_corrected.png"
                align_external_plan_levels([reference, target], (1000.0, 800.0))
                self.assertAlmostEqual(target.canvas_offset_x_pixels, -290.0)
                self.assertAlmostEqual(target.canvas_offset_y_pixels, 150.0)

    def test_missing_landmark_preserves_offsets(self) -> None:
        reference = _level(_correction(2))
        target = _level(replace(_correction(1), alignment_point_normalized=None))
        target.canvas_offset_x_pixels = 11.0
        target.canvas_offset_y_pixels = -12.0

        align_external_plan_levels([reference, target], (1000.0, 800.0))

        self.assertEqual(target.canvas_offset_x_pixels, 11.0)
        self.assertEqual(target.canvas_offset_y_pixels, -12.0)


if __name__ == "__main__":
    unittest.main()
