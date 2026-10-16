# ### Imports ###
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from housemaker.models import PIXEL_TO_METER, LevelData
from housemaker.plan_correction_exchange import ExternalPlanCorrection
from housemaker.project_io import load_project, save_project


# ### Test helpers ###
def _external_correction() -> ExternalPlanCorrection:
    return ExternalPlanCorrection(
        level_index=1,
        image_filename="L1_corrected.png",
        image_size=(1200, 800),
        meters_per_pixel=0.01,
        confidence="high",
        measurements=(),
        alignment_point_normalized=(0.4, 0.6),
        notes="Keep native image resolution.",
        reference_level_index=2,
    )


# ### Persistence tests ###
class ExternalPlanCorrectionPersistenceTests(unittest.TestCase):
    def test_pending_correction_round_trips_without_applying_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "L1_original.jpg"
            level = LevelData(
                index=1,
                name="Underground 1",
                image_path=str(source),
                original_image_path=str(source),
                image_size_pixels=(1600.0, 1000.0),
                scale=1.5,
                canvas_level_scale=1.25,
                canvas_offset_x_pixels=30.0,
                canvas_offset_y_pixels=-40.0,
                external_plan_correction=_external_correction(),
            )
            project = root / "project.json"

            save_project(project, 1, [level])
            restored = load_project(project).levels[1]

            self.assertEqual(restored.external_plan_correction, _external_correction())
            self.assertEqual(restored.image_path, str(source.resolve()))
            self.assertEqual(restored.image_size_pixels, (1600.0, 1000.0))
            self.assertIsNone(restored.plan_metric_calibration)
            self.assertEqual(restored.meters_per_pixel, PIXEL_TO_METER)
            self.assertEqual(restored.scale, 1.5)
            self.assertEqual(restored.canvas_level_scale, 1.25)
            self.assertEqual(restored.canvas_offset_x_pixels, 30.0)
            self.assertEqual(restored.canvas_offset_y_pixels, -40.0)

    def test_applied_descriptor_retains_reference_for_later_level_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            level = LevelData(
                index=1,
                name="Underground 1",
                image_path=str(root / "L1_corrected.png"),
                image_size_pixels=(1200.0, 800.0),
                plan_meters_per_pixel=0.01,
                external_plan_correction=_external_correction(),
            )
            project = root / "project.json"

            save_project(project, 1, [level])
            payload = json.loads(project.read_text(encoding="utf-8"))
            restored = load_project(project).levels[1]

            descriptor = payload["levels"][0]["external_plan_correction"]
            self.assertEqual(descriptor["reference_level"], "L2")
            self.assertEqual(restored.external_plan_correction, _external_correction())
            self.assertEqual(restored.meters_per_pixel, 0.01)

    def test_invalid_optional_descriptor_does_not_break_project_loading(self) -> None:
        invalid_values = (
            "invalid",
            {},
            {**_external_correction().to_dict(), "image_size": [-1, 800]},
            {**_external_correction().to_dict(), "meters_per_pixel": -1.0},
            {**_external_correction().to_dict(), "level": "L3"},
        )
        for invalid_value in invalid_values:
            with (
                self.subTest(payload=invalid_value),
                tempfile.TemporaryDirectory() as directory,
            ):
                project = Path(directory) / "project.json"
                level = LevelData(index=1, name="Underground 1")
                save_project(project, 1, [level])
                payload = json.loads(project.read_text(encoding="utf-8"))
                payload["levels"][0]["external_plan_correction"] = invalid_value
                project.write_text(json.dumps(payload), encoding="utf-8")

                restored = load_project(project).levels[1]

                self.assertIsNone(restored.external_plan_correction)
                self.assertEqual(restored.meters_per_pixel, PIXEL_TO_METER)

    def test_legacy_project_without_descriptor_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project.json"
            level = LevelData(index=1, name="Underground 1")
            save_project(project, 1, [level])
            payload = json.loads(project.read_text(encoding="utf-8"))
            payload["levels"][0].pop("external_plan_correction")
            project.write_text(json.dumps(payload), encoding="utf-8")

            restored = load_project(project).levels[1]

            self.assertIsNone(restored.external_plan_correction)


if __name__ == "__main__":
    unittest.main()
