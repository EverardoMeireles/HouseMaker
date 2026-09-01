# ### Environment setup ###
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ### Imports ###
from PySide6.QtWidgets import QApplication, QLabel

from housemaker.blueprint_canvas import BlueprintCanvas
from housemaker.main import BlueprintWorkspace

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Tests ###
class CanvasFeatureRemovalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.widgets: list[object] = []

    def tearDown(self) -> None:
        for widget in reversed(self.widgets):
            close = getattr(widget, "close", None)
            if callable(close):
                close()
        _qt_application.processEvents()

    def _track_widget(self, widget: object) -> object:
        self.widgets.append(widget)
        return widget

    def test_canvas_general_tab_omits_retired_sections(self) -> None:
        workspace = self._track_widget(BlueprintWorkspace())
        self.assertIsInstance(workspace, BlueprintWorkspace)

        general_tab = workspace.side_tabs.widget(0)
        self.assertIsNotNone(general_tab)
        general_labels = {
            label.text() for label in general_tab.findChildren(QLabel)
        }
        self.assertNotIn("Initial first person camera", general_labels)
        self.assertNotIn("Rooms", general_labels)
        self.assertNotIn("Blueprint scale", general_labels)
        self.assertNotIn("Blueprint X offset", general_labels)
        self.assertNotIn("Blueprint Y offset", general_labels)

        for removed_attribute in (
            "first_person_camera_status_label",
            "first_person_camera_z_spinbox",
            "first_person_camera_light_slider",
            "set_first_person_camera_button",
            "clear_first_person_camera_button",
            "rooms_list",
            "room_name_field",
            "room_height_spinbox",
            "designate_room_button",
            "image_scale_spinbox",
            "image_x_offset_spinbox",
            "image_y_offset_spinbox",
        ):
            with self.subTest(removed_attribute=removed_attribute):
                self.assertFalse(hasattr(workspace, removed_attribute))

        self.assertEqual(
            [
                workspace.side_tabs.tabText(index)
                for index in range(workspace.side_tabs.count())
            ],
            ["Generals"],
        )
        for removed_attribute in (
            "uvs_tab",
            "uv_canvas",
            "uv_rooms_list",
            "images_tab",
            "image_thumbnail_list",
            "texture_creator_tab",
            "texture_creator_canvas",
            "png_export_button",
        ):
            with self.subTest(removed_texture_attribute=removed_attribute):
                self.assertFalse(hasattr(workspace, removed_attribute))

    def test_blueprint_canvas_omits_retired_editing_entry_points(self) -> None:
        canvas = self._track_widget(BlueprintCanvas())
        self.assertIsInstance(canvas, BlueprintCanvas)

        for removed_attribute in (
            "start_room_designation",
            "delete_room_at_index",
            "pending_room_name",
            "start_first_person_camera_placement",
            "clear_first_person_camera",
            "initial_first_person_camera",
            "first_person_camera_changed",
        ):
            with self.subTest(removed_attribute=removed_attribute):
                self.assertFalse(hasattr(canvas, removed_attribute))

        self.assertTrue(hasattr(canvas, "rooms"))
        self.assertTrue(hasattr(canvas, "rooms_changed"))


# ### Test runner ###
if __name__ == "__main__":
    unittest.main()
