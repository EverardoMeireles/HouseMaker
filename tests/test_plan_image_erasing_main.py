# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.blueprint_canvas import PlanImageEraseCommit
from housemaker.main import BlueprintWorkspace, _build_local_file_revision
from housemaker.models import VertexData
from housemaker.plan_correction_exchange import ExternalPlanCorrection
from housemaker.plan_correction_import import (
    apply_external_plan_metadata,
    matches_external_plan_image,
)
from housemaker.plan_metric_state import (
    PlanMeasurementEvidence,
    PlanMetricCalibration,
)
from housemaker.project_io import load_project, save_project

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Test helpers ###
def _plan_metric_calibration() -> PlanMetricCalibration:
    return PlanMetricCalibration(
        estimated_meters_per_pixel=0.01,
        target_meters_per_pixel=0.02,
        normalization_scale=0.5,
        relative_residual=0.01,
        confidence="high",
        source_image_size=(192, 128),
        pre_normalization_size=(192, 128),
        normalized_image_size=(96, 64),
        measurements=(
            PlanMeasurementEvidence(
                label="100",
                value_meters=1.0,
                start_normalized=(0.1, 0.5),
                end_normalized=(0.9, 0.5),
                corrected_start_normalized=(0.1, 0.5),
                corrected_end_normalized=(0.9, 0.5),
                confidence="high",
            ),
            PlanMeasurementEvidence(
                label="80",
                value_meters=0.8,
                start_normalized=(0.2, 0.2),
                end_normalized=(0.2, 0.8),
                corrected_start_normalized=(0.2, 0.2),
                corrected_end_normalized=(0.2, 0.8),
                confidence="high",
            ),
        ),
    )


# ### Main-workspace integration tests ###
class PlanImageErasingMainTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.temporary_path = Path(self._temporary_directory.name)
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(
                self.temporary_path / "settings.json"
            )
        )
        self.workspace.resize(1200, 800)
        self.workspace.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        self.workspace.deleteLater()
        _qt_application.processEvents()
        self._temporary_directory.cleanup()

    def _load_source_image(self) -> Path:
        source_path = self.temporary_path / "source-plan.png"
        image = Image.new("RGB", (96, 64), "white")
        for x in range(20, 76):
            image.putpixel((x, 32), (0, 0, 0))
        image.save(source_path, format="PNG")
        self.workspace._set_current_level_image(str(source_path))
        _qt_application.processEvents()
        return source_path.resolve()

    def _emit_successful_commit(
        self,
        source_path: Path,
        *,
        replacement_name: str = "erased-plan.png",
    ) -> tuple[Path, PlanImageEraseCommit]:
        previous_revision = self.workspace.canvas.get_blueprint_image_revision()
        replacement_path = self.temporary_path / replacement_name
        with Image.open(source_path) as source:
            replacement = source.convert("RGB")
        for y in range(28, 37):
            for x in range(42, 55):
                replacement.putpixel((x, y), (255, 255, 255))
        replacement.save(replacement_path, format="PNG")
        replacement_path = replacement_path.resolve()

        self.assertTrue(
            self.workspace.canvas.load_blueprint_image_preserving_view(
                str(replacement_path)
            )
        )
        replacement_revision = (
            self.workspace.canvas.get_blueprint_image_revision()
        )
        self.assertIsNotNone(replacement_revision)
        assert replacement_revision is not None
        commit = PlanImageEraseCommit(
            previous_path=str(source_path),
            replacement_path=str(replacement_path),
            previous_revision=previous_revision,
            replacement_revision=replacement_revision,
            image_size_pixels=(96.0, 64.0),
        )
        self.workspace.canvas.plan_image_erase_committed.emit(commit)
        _qt_application.processEvents()
        return replacement_path, commit

    def test_erase_button_tracks_canvas_mode_and_loaded_image_state(self) -> None:
        button = self.workspace.erase_plan_image_button
        canvas = self.workspace.canvas

        self.assertEqual(button.text(), "Erase")
        self.assertFalse(button.isEnabled())
        self.assertFalse(button.isChecked())

        self._load_source_image()

        self.assertTrue(button.isEnabled())
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        _qt_application.processEvents()
        self.assertTrue(canvas.is_plan_image_erasing())
        self.assertTrue(button.isChecked())

        # The Canvas signal is authoritative even when mode changes do not
        # originate from the button itself.
        canvas.stop_plan_image_erasing()
        _qt_application.processEvents()
        self.assertFalse(canvas.is_plan_image_erasing())
        self.assertFalse(button.isChecked())

        canvas.start_plan_image_erasing(
            self.temporary_path / "direct-canvas-erased-plan.png"
        )
        _qt_application.processEvents()
        self.assertTrue(canvas.is_plan_image_erasing())
        self.assertTrue(button.isChecked())

        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        _qt_application.processEvents()
        self.assertFalse(canvas.is_plan_image_erasing())
        self.assertFalse(button.isChecked())

    def test_successful_commit_updates_level_provenance_revision_and_history(
        self,
    ) -> None:
        source_path = self._load_source_image()
        source_bytes = source_path.read_bytes()
        level = self.workspace.current_level
        calibration = _plan_metric_calibration()
        level.plan_metric_calibration = calibration
        initial_history_size = len(self.workspace._canvas_undo_stack)

        replacement_path, commit = self._emit_successful_commit(source_path)

        self.assertEqual(level.image_path, str(replacement_path))
        self.assertEqual(level.original_image_path, str(source_path))
        self.assertEqual(level.image_size_pixels, (96.0, 64.0))
        self.assertEqual(
            level.plan_metric_calibration,
            replace(
                calibration,
                normalized_image_sha256=hashlib.sha256(
                    replacement_path.read_bytes()
                ).hexdigest(),
            ),
        )
        self.assertEqual(self.workspace.canvas.blueprint_path, str(replacement_path))
        self.assertEqual(
            self.workspace.canvas.get_blueprint_image_revision(),
            commit.replacement_revision,
        )
        self.assertEqual(
            self.workspace._level_blueprint_image_revisions[level.index],
            commit.replacement_revision,
        )
        self.assertEqual(
            _build_local_file_revision(replacement_path),
            commit.replacement_revision,
        )
        self.assertEqual(
            len(self.workspace._canvas_undo_stack),
            initial_history_size + 1,
        )
        self.assertEqual(source_path.read_bytes(), source_bytes)

    def test_ctrl_z_restores_the_previous_plan_path_and_revision(self) -> None:
        source_path = self._load_source_image()
        source_revision = self.workspace.canvas.get_blueprint_image_revision()
        source_bytes = source_path.read_bytes()
        replacement_path, _commit = self._emit_successful_commit(source_path)
        self.workspace.canvas.stop_plan_image_erasing()
        self.workspace.canvas.setFocus()
        _qt_application.processEvents()

        QTest.keyClick(
            self.workspace.canvas,
            Qt.Key.Key_Z,
            Qt.KeyboardModifier.ControlModifier,
        )
        _qt_application.processEvents()

        level = self.workspace.current_level
        self.assertEqual(level.image_path, str(source_path))
        self.assertEqual(level.original_image_path, str(source_path))
        self.assertEqual(self.workspace.canvas.blueprint_path, str(source_path))
        self.assertEqual(
            self.workspace.canvas.get_blueprint_image_revision(),
            source_revision,
        )
        self.assertEqual(
            self.workspace._level_blueprint_image_revisions[level.index],
            source_revision,
        )
        self.assertEqual(self.workspace._canvas_undo_stack, [])
        self.assertTrue(replacement_path.is_file())
        self.assertEqual(source_path.read_bytes(), source_bytes)

    def test_erase_and_undo_keep_native_metric_density_after_save_load(self) -> None:
        source_path = self._load_source_image()
        level = self.workspace.current_level
        calibration = replace(
            _plan_metric_calibration(),
            target_meters_per_pixel=0.01,
            normalization_scale=1.0,
            pre_normalization_size=(96, 64),
            normalized_image_sha256=hashlib.sha256(
                source_path.read_bytes()
            ).hexdigest(),
        )
        level.plan_metric_calibration = calibration
        level.scale = 1.25
        project_path = self.temporary_path / "project.json"

        replacement_path, _commit = self._emit_successful_commit(source_path)
        save_project(project_path, level.index, [level])
        restored_erased = load_project(project_path).levels[level.index]

        self.assertIsNotNone(restored_erased.plan_metric_calibration)
        self.assertEqual(restored_erased.meters_per_pixel, 0.01)
        self.assertEqual(restored_erased.scale, 1.25)
        self.assertEqual(
            restored_erased.plan_metric_calibration.normalized_image_sha256,
            hashlib.sha256(replacement_path.read_bytes()).hexdigest(),
        )
        self.workspace.canvas.stop_plan_image_erasing()
        self.workspace.canvas.setFocus()
        QTest.keyClick(
            self.workspace.canvas,
            Qt.Key.Key_Z,
            Qt.KeyboardModifier.ControlModifier,
        )
        _qt_application.processEvents()
        undone = self.workspace.current_level
        save_project(project_path, undone.index, [undone])
        restored_undone = load_project(project_path).levels[undone.index]

        self.assertEqual(restored_undone.image_path, str(source_path))
        self.assertEqual(restored_undone.plan_metric_calibration, calibration)
        self.assertEqual(restored_undone.meters_per_pixel, 0.01)
        self.assertEqual(restored_undone.scale, 1.25)

    def test_starting_erase_invalidates_a_generated_wall_preview(self) -> None:
        self._load_source_image()
        candidate = VertexData()
        first = candidate.add_vertex(10.0, 10.0)
        second = candidate.add_vertex(80.0, 10.0)
        candidate.add_edge(first.id, second.id)
        self.assertTrue(self.workspace.canvas.set_generated_wall_preview(candidate))
        self.assertIsNotNone(
            self.workspace.canvas.get_generated_wall_preview()
        )

        QTest.mouseClick(
            self.workspace.erase_plan_image_button,
            Qt.MouseButton.LeftButton,
        )
        _qt_application.processEvents()

        self.assertTrue(self.workspace.canvas.is_plan_image_erasing())
        self.assertIsNone(self.workspace.canvas.get_generated_wall_preview())

    def test_imported_plan_erase_rebinds_descriptor_and_ctrl_z_restores_it(self) -> None:
        original_path = self._load_source_image()
        level = self.workspace.current_level
        corrected_path = self.temporary_path / f"L{level.index}_corrected.png"
        with Image.open(original_path) as image:
            image.save(corrected_path, format="PNG")
        self.workspace._set_current_level_image(
            str(corrected_path), original_image_path=str(original_path)
        )
        calibration = replace(
            _plan_metric_calibration(),
            target_meters_per_pixel=0.01,
            normalization_scale=1.0,
            pre_normalization_size=(96, 64),
            normalized_image_sha256=hashlib.sha256(
                corrected_path.read_bytes()
            ).hexdigest(),
        )
        correction = ExternalPlanCorrection(
            level_index=level.index,
            image_filename=corrected_path.name,
            image_size=(96, 64),
            meters_per_pixel=0.01,
            confidence="high",
            measurements=calibration.measurements,
            alignment_point_normalized=(0.4, 0.6),
            reference_level_index=level.index,
        )
        apply_external_plan_metadata(level, correction, calibration)
        project_path = self.temporary_path / "imported-plan.json"

        replacement_path, _commit = self._emit_successful_commit(corrected_path)
        rebound = replace(correction, image_filename=replacement_path.name)
        self.assertEqual(level.external_plan_correction, rebound)
        self.assertTrue(matches_external_plan_image(
            rebound, level.image_path, level.image_size_pixels
        ))
        save_project(project_path, level.index, [level])
        restored_erased = load_project(project_path).levels[level.index]
        self.assertEqual(restored_erased.external_plan_correction, rebound)
        self.assertEqual(restored_erased.meters_per_pixel, 0.01)

        self.workspace.canvas.stop_plan_image_erasing()
        self.workspace.canvas.setFocus()
        QTest.keyClick(
            self.workspace.canvas,
            Qt.Key.Key_Z,
            Qt.KeyboardModifier.ControlModifier,
        )
        _qt_application.processEvents()

        self.assertEqual(level.image_path, str(corrected_path))
        self.assertEqual(level.external_plan_correction, correction)
        self.assertEqual(level.plan_metric_calibration, calibration)
        self.assertTrue(matches_external_plan_image(
            correction, level.image_path, level.image_size_pixels
        ))
        save_project(project_path, level.index, [level])
        restored_undone = load_project(project_path).levels[level.index]
        self.assertEqual(restored_undone.external_plan_correction, correction)
        self.assertEqual(restored_undone.plan_metric_calibration, calibration)


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
