# ### Imports ###
from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from PIL import Image

from housemaker.models import LevelData
from housemaker.plan_correction_exchange import ExternalPlanCorrection
from housemaker.plan_correction_import import (
    external_plan_uses_loaded_coordinates,
    prepare_external_plan_calibration,
)
from housemaker.plan_metric_state import PlanMeasurementEvidence


# ### Test helpers ###
def _correction() -> ExternalPlanCorrection:
    return ExternalPlanCorrection(
        level_index=2,
        image_filename="L2_corrected.png",
        image_size=(100, 200),
        meters_per_pixel=0.01,
        confidence="high",
        measurements=tuple(
            PlanMeasurementEvidence(
                label=label,
                value_meters=value,
                start_normalized=start,
                end_normalized=end,
                corrected_start_normalized=start,
                corrected_end_normalized=end,
                confidence="high",
            )
            for label, value, start, end in (
                ("80 cm", 0.8, (0.1, 0.2), (0.9, 0.2)),
                ("160 cm", 1.6, (0.2, 0.1), (0.2, 0.9)),
            )
        ),
    )


def _level(path: Path) -> LevelData:
    return LevelData(
        index=2,
        name="Ground",
        image_path=str(path),
        image_size_pixels=(100.0, 200.0),
    )


# ### Loaded-coordinate binding tests ###
class LoadedPlanCoordinateBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "L2_corrected.png"
        Image.new("RGB", (100, 200), "white").save(self.path)
        self.correction = _correction()
        self.level = _level(self.path)

    def test_unknown_legacy_revision_allows_metadata_for_loaded_image(self) -> None:
        self.assertTrue(external_plan_uses_loaded_coordinates(
            self.level, self.correction, self.path,
        ))
        self.assertIsNone(self.level.plan_metric_calibration)

    def test_resolved_path_equivalence_is_accepted(self) -> None:
        self.assertTrue(external_plan_uses_loaded_coordinates(
            self.level, self.correction,
            self.path.parent / "unused" / ".." / self.path.name,
        ))

    def test_same_basename_in_another_directory_is_replacement(self) -> None:
        another_path = self.path.parent / "another" / self.path.name
        self.assertFalse(external_plan_uses_loaded_coordinates(
            self.level, self.correction, another_path,
        ))

    def test_matching_requires_loaded_dimensions_name_and_level(self) -> None:
        for correction, level in (
            (replace(self.correction, image_size=(100, 201)), self.level),
            (replace(self.correction, image_filename="L2_other.png"), self.level),
            (replace(self.correction, level_index=3), self.level),
            (self.correction, replace(self.level, image_size_pixels=None)),
            (self.correction, replace(self.level, image_path=None)),
        ):
            with self.subTest(correction=correction, level=level):
                self.assertFalse(external_plan_uses_loaded_coordinates(
                    level, correction, self.path,
                ))
        self.assertFalse(external_plan_uses_loaded_coordinates(
            self.level, self.correction, None,
        ))

    def test_equal_known_revisions_allow_metadata(self) -> None:
        revision = (str(self.path), 200, 100, 110)
        self.assertTrue(external_plan_uses_loaded_coordinates(
            self.level, self.correction, self.path,
            loaded_revision=revision, current_revision=revision,
        ))

    def test_changed_revision_is_rejected_even_with_same_dimensions(self) -> None:
        loaded = (str(self.path), 200, 100, 110)
        for current in ((str(self.path), 200, 101, 111), None):
            with (
                self.subTest(current=current),
                self.assertRaisesRegex(ValueError, "changed outside the app"),
            ):
                external_plan_uses_loaded_coordinates(
                    self.level, self.correction, self.path,
                    loaded_revision=loaded, current_revision=current,
                )

    def test_known_fingerprint_accepts_unchanged_image(self) -> None:
        calibration = prepare_external_plan_calibration(self.correction, self.path)
        self.assertIsNotNone(calibration)
        self.level.plan_metric_calibration = calibration
        self.assertTrue(external_plan_uses_loaded_coordinates(
            self.level, self.correction, self.path,
        ))

    def test_hash_rejects_same_name_size_and_revision_with_replaced_contents(self) -> None:
        calibration = prepare_external_plan_calibration(self.correction, self.path)
        self.assertIsNotNone(calibration)
        self.level.plan_metric_calibration = calibration
        Image.new("RGB", (100, 200), "black").save(self.path)
        revision = (str(self.path), 200, 100, 110)
        with self.assertRaisesRegex(ValueError, "scale-only correction"):
            external_plan_uses_loaded_coordinates(
                self.level, self.correction, self.path,
                loaded_revision=revision, current_revision=revision,
            )

    def test_replacement_path_is_false_instead_of_stale_image_error(self) -> None:
        self.assertFalse(external_plan_uses_loaded_coordinates(
            self.level, self.correction, self.path.parent / "new" / self.path.name,
            loaded_revision=("old", 1, 2, 3), current_revision=("new", 4, 5, 6),
        ))


# ### Test runner ###
if __name__ == "__main__":
    unittest.main()
