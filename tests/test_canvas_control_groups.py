# ### Environment setup ###
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QGroupBox, QLabel, QWidget

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.main import BlueprintWorkspace

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Widget helpers ###
def _nearest_group_box(widget: QWidget) -> QGroupBox | None:
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QGroupBox):
            return parent
        parent = parent.parentWidget()
    return None


# ### Canvas control grouping tests ###
class CanvasControlGroupingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        settings_path = Path(self.temporary_directory.name) / "settings.json"
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(settings_path)
        )
        self.workspace.resize(1600, 900)
        self.workspace.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        self.workspace.deleteLater()
        self.temporary_directory.cleanup()
        _qt_application.processEvents()

    def _group_named(self, title: str) -> QGroupBox:
        matching_groups = [
            group
            for group in self.workspace.side_panel.findChildren(QGroupBox)
            if group.title() == title
        ]
        self.assertEqual(
            len(matching_groups),
            1,
            f"Expected exactly one Canvas group titled {title!r}",
        )
        return matching_groups[0]

    def _assert_group_contains(
        self,
        title: str,
        *widgets: QWidget,
        labels: set[str] | None = None,
    ) -> None:
        group = self._group_named(title)
        for widget in widgets:
            self.assertIs(
                _nearest_group_box(widget),
                group,
                f"{widget!r} is not contained by the {title!r} group",
            )
        if labels:
            group_labels = {label.text() for label in group.findChildren(QLabel)}
            self.assertTrue(labels <= group_labels)

    def test_levels_group_contains_blueprint_and_include_controls(self) -> None:
        self._assert_group_contains(
            "Levels",
            self.workspace.load_image_button,
            self.workspace.blueprint_name_label,
            self.workspace.levels_list,
            self.workspace.include_yes_radio,
            self.workspace.include_no_radio,
            labels={"Include"},
        )

    def test_doorways_group_contains_all_doorway_controls(self) -> None:
        self._assert_group_contains(
            "Doorways",
            self.workspace.selected_doorway_arch_checkbox,
            self.workspace.selected_doorway_arch_amount_spinbox,
            self.workspace.doorway_preset_list,
            self.workspace.save_doorway_template_button,
            self.workspace.remove_doorway_preset_button,
            self.workspace.place_doorway_button,
            labels={"Arch amount"},
        )

    def test_wall_mirrors_group_follows_doorways_with_three_actions(self) -> None:
        self._assert_group_contains(
            "Wall mirrors",
            self.workspace.wall_mirror_up_button,
            self.workspace.wall_mirror_undo_button,
            self.workspace.wall_mirror_down_button,
        )
        self.assertEqual(self.workspace.wall_mirror_up_button.text(), "Up")
        self.assertEqual(self.workspace.wall_mirror_undo_button.text(), "Undo")
        self.assertEqual(self.workspace.wall_mirror_down_button.text(), "Down")

        mirrors_layout = self.workspace.wall_mirrors_group.layout()
        assert mirrors_layout is not None
        self.assertIs(
            mirrors_layout.itemAt(0).widget(),
            self.workspace.wall_mirror_up_button,
        )
        self.assertIs(
            mirrors_layout.itemAt(1).widget(),
            self.workspace.wall_mirror_undo_button,
        )
        self.assertIs(
            mirrors_layout.itemAt(2).widget(),
            self.workspace.wall_mirror_down_button,
        )

        side_layout = self.workspace.doorways_group.parentWidget().layout()
        assert side_layout is not None
        self.assertEqual(
            side_layout.indexOf(self.workspace.wall_mirrors_group),
            side_layout.indexOf(self.workspace.doorways_group) + 1,
        )

    def test_stairs_group_contains_type_status_and_add_controls(self) -> None:
        self._assert_group_contains(
            "Stairs",
            self.workspace.stair_style_combo,
            self.workspace.stair_status_label,
            self.workspace.add_stairs_button,
            labels={"Stair type"},
        )

    def test_open_spaces_group_contains_status_and_add_controls(self) -> None:
        self._assert_group_contains(
            "Open spaces",
            self.workspace.open_space_status_label,
            self.workspace.add_open_space_button,
        )

    def test_level_dimensions_group_contains_both_numeric_fields(self) -> None:
        self._assert_group_contains(
            "Level dimensions",
            self.workspace.height_level_spinbox,
            self.workspace.floor_thickness_spinbox,
            labels={"Height level", "Floor thickness"},
        )

    def test_narrow_workspace_keeps_the_full_canvas_interactive(self) -> None:
        self.workspace.resize(760, 700)
        _qt_application.processEvents()
        canvas = self.workspace.canvas
        visible_bounds = canvas.visibleRegion().boundingRect()

        self.assertEqual(visible_bounds, canvas.rect())
        top_right = QPoint(canvas.width() - 4, 12)
        self.assertIs(
            QApplication.widgetAt(canvas.mapToGlobal(top_right)),
            canvas,
        )

        canvas.blueprint_image = QImage(
            100,
            100,
            QImage.Format.Format_RGB32,
        )
        existing_vertex = canvas.vertex_data.add_vertex(90.0, 10.0)
        QTest.mouseClick(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=canvas._image_to_widget(90.0, 10.0).toPoint(),
        )
        self.assertEqual(canvas.selected_vertex_id, existing_vertex.id)

        QTest.mouseClick(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=canvas._image_to_widget(90.0, 70.0).toPoint(),
        )
        self.assertEqual(len(canvas.vertex_data.vertices), 2)


if __name__ == "__main__":
    unittest.main()
