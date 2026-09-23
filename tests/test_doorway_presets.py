# ### Environment setup ###
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ### Imports ###
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget

from housemaker.models import (
    DoorwayPreset,
    create_default_levels,
    create_fallback_doorway_preset,
)
from housemaker.project_io import ProjectData, load_project, save_project

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Tests ###
class DoorwayPresetInvariantTests(unittest.TestCase):
    def setUp(self) -> None:
        self._widgets: list[QWidget] = []

    def tearDown(self) -> None:
        for widget in reversed(self._widgets):
            widget.close()
            widget.deleteLater()
        _qt_application.processEvents()

    def _track_widget(self, widget: QWidget) -> QWidget:
        self._widgets.append(widget)
        return widget

    def test_removing_last_preset_replaces_it_with_one_meter_square(self) -> None:
        from housemaker.main import BlueprintWorkspace

        workspace = self._track_widget(BlueprintWorkspace())
        self.assertIsInstance(workspace, BlueprintWorkspace)
        workspace.doorway_presets = [
            DoorwayPreset(width_meters=1.4, height_meters=2.3)
        ]
        workspace._refresh_doorway_preset_list(selected_index=0)

        self.assertTrue(workspace.remove_doorway_preset_button.isEnabled())
        QTest.mouseClick(
            workspace.remove_doorway_preset_button,
            Qt.MouseButton.LeftButton,
        )
        _qt_application.processEvents()

        self.assertEqual(
            workspace.doorway_presets,
            [create_fallback_doorway_preset()],
        )
        self.assertEqual(workspace.doorway_preset_list.currentRow(), 0)
        self.assertEqual(
            workspace.doorway_preset_list.currentItem().text(),
            "1.00 m × 1.00 m",
        )
        self.assertTrue(workspace.remove_doorway_preset_button.isEnabled())

    def test_empty_project_data_receives_fallback_preset(self) -> None:
        project = ProjectData(
            blueprint_path=None,
            current_level_index=2,
            levels=create_default_levels(),
            doorway_presets=[],
        )

        self.assertEqual(
            project.doorway_presets,
            [create_fallback_doorway_preset()],
        )

    def test_empty_saved_preset_list_round_trips_with_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "empty-presets.json"
            save_project(
                project_path,
                current_level_index=2,
                levels=create_default_levels(),
                doorway_presets=[],
            )

            payload = json.loads(project_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["doorway_presets"]), 1)
            self.assertEqual(payload["doorway_presets"][0]["width_meters"], 1.0)
            self.assertEqual(payload["doorway_presets"][0]["height_meters"], 1.0)
            self.assertEqual(
                load_project(project_path).doorway_presets,
                [create_fallback_doorway_preset()],
            )


if __name__ == "__main__":
    unittest.main()
