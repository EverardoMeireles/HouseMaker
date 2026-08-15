# ### Environment setup ###
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.main import BlueprintWorkspace, _format_stair_label
from housemaker.models import (
    GROUND_LEVEL_INDEX,
    STAIR_STYLE_FLOATING,
    STAIR_STYLE_FLOATING_WITH_RISER,
)


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _make_current_canvas_clickable(workspace: BlueprintWorkspace) -> None:
    """Give the selected level a small in-memory blueprint for click tests."""

    workspace.current_level.image_size_pixels = (100.0, 100.0)
    workspace._sync_canvas_to_current_level()
    workspace.canvas.blueprint_image = QImage(
        100,
        100,
        QImage.Format.Format_RGB32,
    )
    workspace.canvas.blueprint_image.fill(Qt.GlobalColor.white)
    workspace.canvas.set_stair_context(workspace.stairs, workspace.current_level)
    workspace.canvas.update()
    _qt_application.processEvents()


def _image_position(
    workspace: BlueprintWorkspace,
    image_x: float,
    image_y: float,
):
    return workspace.canvas._image_to_widget(image_x, image_y).toPoint()


def _add_wall_segment(
    workspace: BlueprintWorkspace,
    start: tuple[float, float],
    end: tuple[float, float],
) -> None:
    """Add one selectable wall segment to the active level fixture."""

    start_vertex = workspace.current_level.vertex_data.add_vertex(*start)
    end_vertex = workspace.current_level.vertex_data.add_vertex(*end)
    workspace.current_level.vertex_data.add_edge(
        start_vertex.id,
        end_vertex.id,
    )


# ### Main integration tests ###
class StairMainIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(
                Path(self._temporary_directory.name) / "settings.json"
            )
        )
        self.workspace.resize(1400, 850)
        self.workspace.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self._temporary_directory.cleanup()

    def test_add_stairs_uses_two_levels_and_commits_the_selected_style(self) -> None:
        _add_wall_segment(self.workspace, (20.0, 35.0), (45.0, 35.0))
        _make_current_canvas_clickable(self.workspace)
        self.workspace.stair_style_combo.setCurrentIndex(
            self.workspace.stair_style_combo.findData(STAIR_STYLE_FLOATING)
        )

        with patch("housemaker.main.QMessageBox.information"):
            self.workspace.add_stairs_button.click()

        self.assertTrue(self.workspace.canvas.is_stair_placement_active())
        self.assertIsNone(self.workspace.canvas.get_pending_stair_placement())
        self.assertFalse(self.workspace.add_stairs_button.isEnabled())

        QTest.mouseClick(
            self.workspace.canvas,
            Qt.MouseButton.LeftButton,
            pos=_image_position(self.workspace, 20.0, 35.0),
        )
        QTest.mouseClick(
            self.workspace.canvas,
            Qt.MouseButton.LeftButton,
            pos=_image_position(self.workspace, 45.0, 35.0),
        )
        _qt_application.processEvents()

        pending = self.workspace.canvas.get_pending_stair_placement()
        self.assertIsNotNone(pending)
        self.assertEqual(pending.start_level_index, GROUND_LEVEL_INDEX)  # type: ignore[union-attr]
        self.assertEqual(pending.style, STAIR_STYLE_FLOATING)  # type: ignore[union-attr]

        destination_level_index = GROUND_LEVEL_INDEX + 1
        self.workspace._handle_level_selection_changed(destination_level_index)
        _add_wall_segment(self.workspace, (50.0, 58.0), (90.0, 58.0))
        _make_current_canvas_clickable(self.workspace)

        QTest.mouseClick(
            self.workspace.canvas,
            Qt.MouseButton.LeftButton,
            pos=_image_position(self.workspace, 50.0, 58.0),
        )
        QTest.mouseClick(
            self.workspace.canvas,
            Qt.MouseButton.LeftButton,
            pos=_image_position(self.workspace, 90.0, 58.0),
        )
        _qt_application.processEvents()

        self.assertTrue(self.workspace.canvas.is_stair_placement_active())
        self.assertTrue(
            self.workspace.canvas.is_stair_ready_for_confirmation()
        )
        self.assertEqual(self.workspace.stairs, [])
        self.assertEqual(
            self.workspace.add_stairs_button.text(),
            "Confirm stairs",
        )
        self.assertTrue(self.workspace.add_stairs_button.isEnabled())

        self.workspace.add_stairs_button.click()
        _qt_application.processEvents()

        self.assertFalse(self.workspace.canvas.is_stair_placement_active())
        self.assertEqual(len(self.workspace.stairs), 1)
        stair = self.workspace.stairs[0]
        self.assertEqual(stair.style, STAIR_STYLE_FLOATING)
        self.assertEqual(stair.start_level_index, GROUND_LEVEL_INDEX)
        self.assertEqual(stair.end_level_index, destination_level_index)
        self.assertEqual(self.workspace.stairs_list.count(), 1)
        self.assertIn("Added floating stairs", self.workspace.stair_status_label.text())

        viewer_model = self.workspace.viewer.model
        self.assertIsNotNone(viewer_model)
        self.assertIn(
            "stair_1_floating",
            viewer_model.scene.geometry,  # type: ignore[union-attr]
        )

    def test_stair_style_combo_offers_floating_with_riser(self) -> None:
        style_index = self.workspace.stair_style_combo.findData(
            STAIR_STYLE_FLOATING_WITH_RISER
        )

        self.assertGreaterEqual(style_index, 0)
        self.assertEqual(
            self.workspace.stair_style_combo.itemText(style_index),
            "Floating with riser",
        )

        stair = type(
            "Stair",
            (),
            {
                "style": STAIR_STYLE_FLOATING_WITH_RISER,
                "start_level_index": GROUND_LEVEL_INDEX,
                "end_level_index": GROUND_LEVEL_INDEX + 1,
                "intermediate_sections": (),
            },
        )()
        self.assertEqual(
            _format_stair_label(stair),  # type: ignore[arg-type]
            "Floating with riser: L2 to L3 (straight)",
        )

    def test_curve_guides_remain_draft_until_confirmed(self) -> None:
        _make_current_canvas_clickable(self.workspace)
        with patch("housemaker.main.QMessageBox.information"):
            self.workspace.add_stairs_button.click()

        for point in ((20.0, 30.0), (40.0, 30.0)):
            QTest.mouseClick(
                self.workspace.canvas,
                Qt.MouseButton.LeftButton,
                pos=_image_position(self.workspace, *point),
            )

        destination_level_index = GROUND_LEVEL_INDEX + 1
        self.workspace._handle_level_selection_changed(destination_level_index)
        _make_current_canvas_clickable(self.workspace)
        for point in ((70.0, 70.0), (90.0, 70.0)):
            QTest.mouseClick(
                self.workspace.canvas,
                Qt.MouseButton.LeftButton,
                pos=_image_position(self.workspace, *point),
            )

        for point in (
            (40.0, 65.0),
            (60.0, 65.0),
            (50.0, 52.0),
            (70.0, 52.0),
        ):
            QTest.mouseClick(
                self.workspace.canvas,
                Qt.MouseButton.LeftButton,
                pos=_image_position(self.workspace, *point),
            )
        _qt_application.processEvents()

        draft = self.workspace.canvas.get_stair_placement_draft()
        self.assertIsNotNone(draft)
        self.assertEqual(
            len(draft.intermediate_sections),  # type: ignore[union-attr]
            2,
        )
        self.assertEqual(self.workspace.stairs, [])
        self.assertEqual(
            self.workspace.add_stairs_button.text(),
            "Confirm stairs",
        )

        self.workspace.add_stairs_button.click()
        _qt_application.processEvents()

        self.assertEqual(len(self.workspace.stairs), 1)
        self.assertEqual(
            len(self.workspace.stairs[0].intermediate_sections),
            2,
        )
        self.assertEqual(
            [
                section.to_dict()
                for section in self.workspace.stairs[0].intermediate_sections
            ],
            [
                {
                    "level_index": section.level_index,
                    "a_x": section.a_x,
                    "a_y": section.a_y,
                    "b_x": section.b_x,
                    "b_y": section.b_y,
                    "a_vertex_id": section.a_vertex_id,
                    "b_vertex_id": section.b_vertex_id,
                }
                for section in draft.intermediate_sections  # type: ignore[union-attr]
            ],
        )
        self.assertIn(
            "curved, 2 guides",
            self.workspace.stairs_list.item(0).text(),
        )
        self.assertEqual(self.workspace.add_stairs_button.text(), "Add stairs")

    def test_completed_stair_refits_the_3d_preview(self) -> None:
        placement = type(
            "Placement",
            (),
            {
                "start_level_index": GROUND_LEVEL_INDEX,
                "start_a_x": 20.0,
                "start_a_y": 35.0,
                "start_b_x": 45.0,
                "start_b_y": 35.0,
                "end_level_index": GROUND_LEVEL_INDEX + 1,
                "end_a_x": 50.0,
                "end_a_y": 58.0,
                "end_b_x": 90.0,
                "end_b_y": 58.0,
                "style": STAIR_STYLE_FLOATING,
            },
        )()

        with patch.object(
            self.workspace,
            "_schedule_viewer_preview_refresh",
        ) as refresh_mock:
            self.workspace._handle_stair_placement_completed(placement)

        self.assertEqual(len(self.workspace.stairs), 1)
        refresh_mock.assert_called_once_with(preserve_camera=False)

    def test_loading_a_project_cancels_a_pending_stair_placement(self) -> None:
        _add_wall_segment(self.workspace, (20.0, 35.0), (45.0, 35.0))
        _make_current_canvas_clickable(self.workspace)
        self.workspace.canvas.start_stair_placement()
        QTest.mouseClick(
            self.workspace.canvas,
            Qt.MouseButton.LeftButton,
            pos=_image_position(self.workspace, 20.0, 35.0),
        )
        QTest.mouseClick(
            self.workspace.canvas,
            Qt.MouseButton.LeftButton,
            pos=_image_position(self.workspace, 45.0, 35.0),
        )
        self.assertIsNotNone(self.workspace.canvas.get_pending_stair_placement())

        self.workspace._apply_project_state(
            levels=self.workspace.levels,
            current_level_index=GROUND_LEVEL_INDEX,
        )

        self.assertFalse(self.workspace.canvas.is_stair_placement_active())
        self.assertIsNone(self.workspace.canvas.get_pending_stair_placement())

    def test_switching_levels_mid_opening_directs_user_back_to_its_level(
        self,
    ) -> None:
        _make_current_canvas_clickable(self.workspace)
        with patch("housemaker.main.QMessageBox.information"):
            self.workspace.add_stairs_button.click()
        QTest.mouseClick(
            self.workspace.canvas,
            Qt.MouseButton.LeftButton,
            pos=_image_position(self.workspace, 20.0, 35.0),
        )

        self.workspace._handle_level_selection_changed(GROUND_LEVEL_INDEX + 1)

        status_text = self.workspace.stair_status_label.text()
        self.assertIn("Return to", status_text)
        self.assertIn("Ground", status_text)
        self.assertIsNotNone(self.workspace.canvas.get_pending_stair_point())

    def test_zero_run_stair_is_rejected_before_it_can_break_the_3d_preview(
        self,
    ) -> None:
        _add_wall_segment(self.workspace, (35.0, 45.0), (65.0, 45.0))
        _make_current_canvas_clickable(self.workspace)
        with patch("housemaker.main.QMessageBox.information"):
            self.workspace.add_stairs_button.click()
        start_position = _image_position(self.workspace, 35.0, 45.0)
        QTest.mouseClick(
            self.workspace.canvas,
            Qt.MouseButton.LeftButton,
            pos=start_position,
        )
        QTest.mouseClick(
            self.workspace.canvas,
            Qt.MouseButton.LeftButton,
            pos=_image_position(self.workspace, 65.0, 45.0),
        )

        self.workspace._handle_level_selection_changed(GROUND_LEVEL_INDEX + 1)
        _add_wall_segment(self.workspace, (35.0, 45.0), (65.0, 45.0))
        _make_current_canvas_clickable(self.workspace)
        QTest.mouseClick(
            self.workspace.canvas,
            Qt.MouseButton.LeftButton,
            pos=_image_position(self.workspace, 35.0, 45.0),
        )
        QTest.mouseClick(
            self.workspace.canvas,
            Qt.MouseButton.LeftButton,
            pos=_image_position(self.workspace, 65.0, 45.0),
        )
        _qt_application.processEvents()

        self.assertEqual(self.workspace.stairs, [])
        self.assertEqual(
            self.workspace.add_stairs_button.text(),
            "Confirm stairs",
        )
        self.workspace.add_stairs_button.click()
        _qt_application.processEvents()

        self.assertEqual(self.workspace.stairs, [])
        self.assertIn(
            "separated horizontally",
            self.workspace.stair_status_label.text(),
        )
        self.assertTrue(self.workspace.canvas.is_stair_placement_active())
        self.assertTrue(
            self.workspace.canvas.is_stair_ready_for_confirmation()
        )
        self.assertEqual(
            self.workspace.add_stairs_button.text(),
            "Confirm stairs",
        )

    def test_add_stairs_switches_the_canvas_workspace_to_its_2d_view(self) -> None:
        _make_current_canvas_clickable(self.workspace)
        self.workspace.canvas_viewer_tabs.setCurrentWidget(self.workspace.viewer)
        self.assertIs(self.workspace.canvas_viewer_tabs.currentWidget(), self.workspace.viewer)

        with patch("housemaker.main.QMessageBox.information"):
            self.workspace.add_stairs_button.click()

        self.assertIs(
            self.workspace.canvas_viewer_tabs.currentWidget(),
            self.workspace.canvas,
        )
        self.assertTrue(self.workspace.canvas.is_stair_placement_active())


if __name__ == "__main__":
    unittest.main()
