# ### Imports ###
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from housemaker.models import PIXEL_TO_METER, LevelData
from housemaker.plan_metric_state import (
    PlanMeasurementEvidence,
    PlanMetricCalibration,
)
from housemaker.project_io import load_project, save_project


# ### Test helpers ###
def _plan_metric_calibration() -> PlanMetricCalibration:
    measurements = tuple(
        PlanMeasurementEvidence(
            label=label,
            value_meters=value_meters,
            start_normalized=start,
            end_normalized=end,
            corrected_start_normalized=start,
            corrected_end_normalized=end,
            confidence="high",
        )
        for label, value_meters, start, end in (
            ("610", 6.10, (0.10, 0.20), (0.90, 0.20)),
            ("390", 3.90, (0.20, 0.10), (0.20, 0.80)),
        )
    )
    return PlanMetricCalibration(
        estimated_meters_per_pixel=0.01,
        target_meters_per_pixel=0.02,
        normalization_scale=0.5,
        relative_residual=0.015,
        confidence="high",
        source_image_size=(2000, 1000),
        pre_normalization_size=(2000, 1000),
        normalized_image_size=(1000, 500),
        measurements=measurements,
    )


# ### Tests ###
class PlanImagePersistenceTests(unittest.TestCase):
    def test_level_density_uses_current_raster_not_pre_resampling_estimate(
        self,
    ) -> None:
        legacy = _plan_metric_calibration()
        native = replace(
            legacy,
            target_meters_per_pixel=legacy.estimated_meters_per_pixel,
            normalization_scale=1.0,
            normalized_image_size=legacy.pre_normalization_size,
        )
        for calibration, expected_density in (
            (None, PIXEL_TO_METER),
            (legacy, 0.02),
            (native, 0.01),
        ):
            with self.subTest(calibration=calibration):
                level = LevelData(
                    index=2,
                    name="Ground",
                    scale=2.25,
                    plan_metric_calibration=calibration,
                )

                self.assertEqual(level.meters_per_pixel, expected_density)
                self.assertEqual(level.scale, 2.25)

    def test_native_calibration_save_load_preserves_image_and_geometry_coordinates(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "native-plan.png"
            image_path.write_bytes(b"unchanged native plan raster")
            calibration = replace(
                _plan_metric_calibration(),
                estimated_meters_per_pixel=0.0075,
                target_meters_per_pixel=0.0075,
                normalization_scale=1.0,
                normalized_image_size=(2000, 1000),
                normalized_image_sha256=hashlib.sha256(
                    image_path.read_bytes()
                ).hexdigest(),
            )
            level = LevelData(
                index=2,
                name="Ground",
                image_path=str(image_path),
                image_size_pixels=(2000.0, 1000.0),
                plan_metric_calibration=calibration,
                scale=1.25,
                height_meters=2.45,
                canvas_level_scale=1.75,
                canvas_offset_x_pixels=125.0,
                canvas_offset_y_pixels=-75.0,
                offset_x_meters=1.5,
                offset_y_meters=-0.5,
            )
            first = level.vertex_data.add_vertex(120.0, 300.0)
            second = level.vertex_data.add_vertex(520.0, 300.0)
            level.vertex_data.add_edge(first.id, second.id)
            project_path = root / "project.json"

            save_project(project_path, 2, [level])
            restored = load_project(project_path).levels[2]

            self.assertEqual(restored.plan_metric_calibration, calibration)
            self.assertEqual(restored.meters_per_pixel, 0.0075)
            self.assertEqual(restored.image_size_pixels, level.image_size_pixels)
            self.assertEqual(restored.vertex_data, level.vertex_data)
            for field in (
                "scale",
                "height_meters",
                "canvas_level_scale",
                "canvas_offset_x_pixels",
                "canvas_offset_y_pixels",
                "offset_x_meters",
                "offset_y_meters",
            ):
                self.assertEqual(getattr(restored, field), getattr(level, field))
            self.assertEqual(image_path.read_bytes(), b"unchanged native plan raster")

    def test_original_plan_image_path_round_trips_with_corrected_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            project_path = root / "project.json"
            source_path = root / "source-photo.jpg"
            corrected_path = root / "corrected-plan.png"
            level = LevelData(
                index=2,
                name="Ground",
                image_path=str(corrected_path),
                original_image_path=str(source_path),
                image_size_pixels=(1200.0, 800.0),
            )

            save_project(project_path, 2, [level])
            restored = load_project(project_path).levels[2]

            self.assertEqual(restored.image_path, str(corrected_path.resolve()))
            self.assertEqual(
                restored.original_image_path,
                str(source_path.resolve()),
            )

    def test_legacy_plan_without_original_path_keeps_field_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            project_path = root / "project.json"
            image_path = root / "plan.png"
            save_project(
                project_path,
                2,
                [
                    LevelData(
                        index=2,
                        name="Ground",
                        image_path=str(image_path),
                    )
                ],
            )
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            payload["levels"][0].pop("original_image_path")
            project_path.write_text(json.dumps(payload), encoding="utf-8")

            restored = load_project(project_path).levels[2]

            self.assertIsNone(restored.original_image_path)

    def test_metric_plan_calibration_round_trips_with_level(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "project.json"
            calibration = _plan_metric_calibration()
            level = LevelData(
                index=2,
                name="Ground",
                image_size_pixels=tuple(
                    float(value) for value in calibration.normalized_image_size
                ),
                plan_metric_calibration=calibration,
            )

            save_project(project_path, 2, [level])
            restored = load_project(project_path).levels[2]

            self.assertEqual(restored.plan_metric_calibration, calibration)
            self.assertEqual(restored.meters_per_pixel, 0.02)

    def test_missing_or_malformed_metric_calibration_is_uncalibrated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "project.json"
            save_project(
                project_path,
                2,
                [
                    LevelData(
                        index=2,
                        name="Ground",
                        image_size_pixels=(1000.0, 500.0),
                        plan_metric_calibration=_plan_metric_calibration(),
                    )
                ],
            )
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            raw_level = payload["levels"][0]

            raw_level.pop("plan_metric_calibration")
            project_path.write_text(json.dumps(payload), encoding="utf-8")
            legacy = load_project(project_path).levels[2]
            self.assertIsNone(legacy.plan_metric_calibration)

            raw_level["plan_metric_calibration"] = {
                "target_meters_per_pixel": "not-a-number"
            }
            project_path.write_text(json.dumps(payload), encoding="utf-8")
            malformed = load_project(project_path).levels[2]
            self.assertIsNone(malformed.plan_metric_calibration)

    def test_metric_calibration_for_another_raster_size_is_discarded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "project.json"
            calibration = _plan_metric_calibration()
            level = LevelData(
                index=2,
                name="Ground",
                image_size_pixels=(1000.0, 500.0),
                plan_metric_calibration=calibration,
            )
            save_project(project_path, 2, [level])
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            payload["levels"][0]["image_size_pixels"] = [999.0, 500.0]
            project_path.write_text(json.dumps(payload), encoding="utf-8")

            restored = load_project(project_path).levels[2]

            self.assertEqual(restored.image_size_pixels, (999.0, 500.0))
            self.assertIsNone(restored.plan_metric_calibration)

    def test_metric_calibration_for_replaced_plan_image_is_discarded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            project_path = root / "project.json"
            image_path = root / "corrected-plan.png"
            image_path.write_bytes(b"original corrected plan")
            calibration = replace(
                _plan_metric_calibration(),
                normalized_image_sha256=hashlib.sha256(
                    image_path.read_bytes()
                ).hexdigest(),
            )
            level = LevelData(
                index=2,
                name="Ground",
                image_path=str(image_path),
                image_size_pixels=(1000.0, 500.0),
                plan_metric_calibration=calibration,
            )
            save_project(project_path, 2, [level])
            image_path.write_bytes(b"different image with the same dimensions")

            restored = load_project(project_path).levels[2]

            self.assertIsNone(restored.plan_metric_calibration)
            self.assertEqual(restored.meters_per_pixel, 0.02)

    def test_missing_or_recolored_plan_preserves_authored_native_coordinate_units(
        self,
    ) -> None:
        for image_change in ("missing", "recolored"):
            with (
                self.subTest(image_change=image_change),
                tempfile.TemporaryDirectory() as temporary_directory,
            ):
                root = Path(temporary_directory)
                project_path = root / "project.json"
                image_path = root / "native-plan.png"
                image_path.write_bytes(b"original native plan raster")
                calibration = replace(
                    _plan_metric_calibration(),
                    estimated_meters_per_pixel=0.0075,
                    target_meters_per_pixel=0.0075,
                    normalization_scale=1.0,
                    normalized_image_size=(2000, 1000),
                    normalized_image_sha256=hashlib.sha256(
                        image_path.read_bytes()
                    ).hexdigest(),
                )
                level = LevelData(
                    index=2,
                    name="Ground",
                    image_path=str(image_path),
                    image_size_pixels=(2000.0, 1000.0),
                    plan_metric_calibration=calibration,
                    scale=1.25,
                    offset_x_meters=1.5,
                )
                first = level.vertex_data.add_vertex(120.0, 300.0)
                second = level.vertex_data.add_vertex(520.0, 300.0)
                level.vertex_data.add_edge(first.id, second.id)
                previous_world_x = (
                    first.x - 1000.0
                ) * level.meters_per_pixel * level.scale + level.offset_x_meters
                save_project(project_path, 2, [level])
                if image_change == "missing":
                    image_path.unlink()
                else:
                    image_path.write_bytes(b"recolored native plan raster")

                restored = load_project(project_path).levels[2]

                self.assertIsNone(restored.plan_metric_calibration)
                self.assertEqual(restored.plan_meters_per_pixel, 0.0075)
                self.assertEqual(restored.meters_per_pixel, 0.0075)
                self.assertEqual(restored.vertex_data, level.vertex_data)
                restored_first = restored.vertex_data.vertices[0]
                self.assertAlmostEqual(
                    (restored_first.x - 1000.0)
                    * restored.meters_per_pixel
                    * restored.scale
                    + restored.offset_x_meters,
                    previous_world_x,
                )
                # A second save must preserve the fallback even though the
                # measurement evidence was already discarded on first load.
                save_project(project_path, 2, [restored])
                restored_again = load_project(project_path).levels[2]
                self.assertEqual(restored_again.meters_per_pixel, 0.0075)

    def test_old_save_with_stale_evidence_derives_current_target_density(self) -> None:
        for native in (False, True):
            with (
                self.subTest(native=native),
                tempfile.TemporaryDirectory() as temporary_directory,
            ):
                root = Path(temporary_directory)
                project_path = root / "project.json"
                image_path = root / "corrected-plan.png"
                image_path.write_bytes(b"original calibrated raster")
                calibration = _plan_metric_calibration()
                if native:
                    calibration = replace(
                        calibration,
                        target_meters_per_pixel=0.01,
                        normalization_scale=1.0,
                        normalized_image_size=(2000, 1000),
                    )
                calibration = replace(
                    calibration,
                    normalized_image_sha256=hashlib.sha256(
                        image_path.read_bytes()
                    ).hexdigest(),
                )
                level = LevelData(
                    index=2,
                    name="Ground",
                    image_path=str(image_path),
                    image_size_pixels=calibration.normalized_image_size,
                    plan_metric_calibration=calibration,
                )
                save_project(project_path, 2, [level])
                payload = json.loads(project_path.read_text(encoding="utf-8"))
                payload["levels"][0].pop("plan_meters_per_pixel", None)
                project_path.write_text(json.dumps(payload), encoding="utf-8")
                image_path.write_bytes(b"externally edited calibrated raster")

                restored = load_project(project_path).levels[2]

                self.assertIsNone(restored.plan_metric_calibration)
                self.assertEqual(
                    restored.plan_meters_per_pixel,
                    calibration.target_meters_per_pixel,
                )
                self.assertEqual(
                    restored.meters_per_pixel,
                    calibration.target_meters_per_pixel,
                )


if __name__ == "__main__":
    unittest.main()
