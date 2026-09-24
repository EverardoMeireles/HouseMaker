# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.blueprint_canvas import PlanImageEraseCommit
from housemaker.main import BlueprintWorkspace, _build_local_file_revision
from housemaker.models import VertexData

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


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
        initial_history_size = len(self.workspace._canvas_undo_stack)

        replacement_path, commit = self._emit_successful_commit(source_path)

        self.assertEqual(level.image_path, str(replacement_path))
        self.assertEqual(level.original_image_path, str(source_path))
        self.assertEqual(level.image_size_pixels, (96.0, 64.0))
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


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
