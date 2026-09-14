# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.main import (
    LEVEL_OFFSET_SLIDER_FACTOR,
    LEVEL_SCALE_SLIDER_FACTOR,
    BlueprintWorkspace,
)
from housemaker.models import LevelData, VertexData
from housemaker.viewer import CANVAS_SURFACE_SELECTION_COLOR

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_square_level() -> LevelData:
    vertex_data = VertexData()
    vertices = tuple(
        vertex_data.add_vertex(*point)
        for point in (
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (0.0, 100.0),
        )
    )
    for first, second in zip(vertices, (*vertices[1:], vertices[0])):
        vertex_data.add_edge(first.id, second.id)
    return LevelData(index=2, name="Ground", vertex_data=vertex_data)


def _is_yellow_outline_item(item: object) -> bool:
    color = getattr(item, "color", None)
    if callable(color) or color is None:
        return False
    values = np.asarray(color, dtype=float)
    return values.shape == (4,) and bool(
        np.allclose(values, CANVAS_SURFACE_SELECTION_COLOR)
    )


# ### Delayed transform integration tests ###
class CanvasLevelTransformPreviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(
                Path(self.temporary_directory.name) / "settings.json"
            )
        )
        self.level = _build_square_level()
        self.workspace._apply_project_state([self.level], 0)
        self.workspace.resize(1400, 850)
        self.workspace.show()
        self.workspace.workspace_tabs.setCurrentWidget(
            self.workspace.scene_3d_workspace
        )
        _qt_application.processEvents()
        self.workspace._set_mesh_edit_update_delay_seconds(60.0)
        self.workspace._canvas_undo_stack.clear()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        self.workspace.deleteLater()
        _qt_application.processEvents()
        self.temporary_directory.cleanup()

    def test_level_transform_values_commit_together_after_debounce(
        self,
    ) -> None:
        revision_before = self.workspace._viewer_preview_revision
        self.assertEqual(
            self.workspace._level_transform_mesh_update_timer.interval(),
            60_000,
        )

        with patch.object(
            self.workspace,
            "_schedule_viewer_preview_refresh",
        ) as schedule_refresh:
            self.workspace._handle_level_scale_changed(1.5)
            self.workspace._handle_level_x_offset_changed(1.2)
            self.workspace._handle_level_y_offset_changed(-0.8)

            self.assertAlmostEqual(self.level.scale, 1.0)
            self.assertAlmostEqual(self.level.offset_x_meters, 0.0)
            self.assertAlmostEqual(self.level.offset_y_meters, 0.0)
            self.assertIsNotNone(self.workspace._pending_level_transform)
            self.assertTrue(
                self.workspace._level_transform_mesh_update_timer.isActive()
            )
            self.assertEqual(
                self.workspace._viewer_preview_revision,
                revision_before,
            )
            schedule_refresh.assert_not_called()

            outline_positions = (
                self.workspace.viewer._level_transform_preview_outline_positions
            )
            self.assertIsNotNone(outline_positions)
            outline = np.asarray(outline_positions, dtype=float)
            self.assertEqual(outline.ndim, 2)
            self.assertEqual(outline.shape[1], 3)
            self.assertGreater(len(outline), 0)
            self.assertTrue(np.all(np.isfinite(outline)))
            self.assertTrue(
                any(
                    _is_yellow_outline_item(item)
                    for item in self.workspace.viewer.view.items
                )
            )

            self.workspace._commit_pending_level_transform_update()

        self.assertAlmostEqual(self.level.scale, 1.5)
        self.assertAlmostEqual(self.level.offset_x_meters, 1.2)
        self.assertAlmostEqual(self.level.offset_y_meters, -0.8)
        self.assertIsNone(self.workspace._pending_level_transform)
        self.assertFalse(
            self.workspace._level_transform_mesh_update_timer.isActive()
        )
        self.assertEqual(len(self.workspace._canvas_undo_stack), 1)
        schedule_refresh.assert_called_once()

    def test_drag_starts_delay_only_on_release_and_commits_one_undo_step(
        self,
    ) -> None:
        self.workspace._handle_level_transform_drag_started()
        self.workspace.level_scale_slider.setValue(
            round(1.35 * LEVEL_SCALE_SLIDER_FACTOR)
        )
        self.workspace.level_x_offset_slider.setValue(
            round(0.7 * LEVEL_OFFSET_SLIDER_FACTOR)
        )
        self.workspace.level_x_offset_slider.setValue(
            round(1.4 * LEVEL_OFFSET_SLIDER_FACTOR)
        )
        self.workspace.level_y_offset_slider.setValue(
            round(-0.6 * LEVEL_OFFSET_SLIDER_FACTOR)
        )

        self.assertAlmostEqual(self.level.scale, 1.0)
        self.assertAlmostEqual(self.level.offset_x_meters, 0.0)
        self.assertAlmostEqual(self.level.offset_y_meters, 0.0)
        self.assertIsNotNone(self.workspace._pending_level_transform)
        self.assertFalse(
            self.workspace._level_transform_mesh_update_timer.isActive()
        )
        self.assertIsNotNone(
            self.workspace.viewer._level_transform_preview_outline_positions
        )
        self.assertEqual(self.workspace._canvas_undo_stack, [])

        self.workspace._handle_level_transform_drag_finished()

        self.assertTrue(
            self.workspace._level_transform_mesh_update_timer.isActive()
        )
        self.assertAlmostEqual(self.level.scale, 1.0)
        self.assertAlmostEqual(self.level.offset_x_meters, 0.0)
        self.assertAlmostEqual(self.level.offset_y_meters, 0.0)

        self.workspace._commit_pending_level_transform_update()

        self.assertAlmostEqual(self.level.scale, 1.35)
        self.assertAlmostEqual(self.level.offset_x_meters, 1.4)
        self.assertAlmostEqual(self.level.offset_y_meters, -0.6)
        self.assertEqual(len(self.workspace._canvas_undo_stack), 1)

        self.workspace._handle_canvas_undo_requested()

        self.assertAlmostEqual(self.level.scale, 1.0)
        self.assertAlmostEqual(self.level.offset_x_meters, 0.0)
        self.assertAlmostEqual(self.level.offset_y_meters, 0.0)
        self.assertIsNone(self.workspace._pending_level_transform)
        self.assertFalse(
            self.workspace._level_transform_mesh_update_timer.isActive()
        )
        self.assertEqual(self.workspace._canvas_undo_stack, [])

    def test_timer_applies_a_keyboard_level_transform(self) -> None:
        self.workspace._set_mesh_edit_update_delay_seconds(0.01)

        self.workspace._handle_level_x_offset_changed(0.75)

        self.assertAlmostEqual(self.level.offset_x_meters, 0.0)
        self.assertTrue(
            self.workspace._level_transform_mesh_update_timer.isActive()
        )
        QTest.qWait(80)
        _qt_application.processEvents()

        self.assertAlmostEqual(self.level.offset_x_meters, 0.75)
        self.assertIsNone(self.workspace._pending_level_transform)
        self.assertEqual(len(self.workspace._canvas_undo_stack), 1)
        self.assertEqual(
            self.workspace._canvas_viewer_preview_revision,
            self.workspace._viewer_preview_revision,
        )
        self.assertIsNone(
            self.workspace.viewer._level_transform_preview_outline_positions
        )

    def test_canvas_offset_side_buttons_nudge_exactly_one_pixel(self) -> None:
        self.assertAlmostEqual(self.level.canvas_offset_x_pixels, 0.0)
        self.assertAlmostEqual(self.level.canvas_offset_y_pixels, 0.0)

        self.workspace.canvas_x_offset_increase_button.click()
        self.workspace.canvas_y_offset_decrease_button.click()

        self.assertAlmostEqual(self.level.canvas_offset_x_pixels, 1.0)
        self.assertAlmostEqual(self.level.canvas_offset_y_pixels, -1.0)
        self.assertAlmostEqual(self.workspace.canvas.canvas_offset_x_pixels, 1.0)
        self.assertAlmostEqual(self.workspace.canvas.canvas_offset_y_pixels, -1.0)

        self.workspace.canvas_x_offset_decrease_button.click()
        self.workspace.canvas_y_offset_increase_button.click()

        self.assertAlmostEqual(self.level.canvas_offset_x_pixels, 0.0)
        self.assertAlmostEqual(self.level.canvas_offset_y_pixels, 0.0)
        self.assertAlmostEqual(self.workspace.canvas.canvas_offset_x_pixels, 0.0)
        self.assertAlmostEqual(self.workspace.canvas.canvas_offset_y_pixels, 0.0)

        self.level.canvas_offset_x_pixels = 0.425
        self.level.canvas_offset_y_pixels = -0.425
        self.workspace._sync_level_controls()
        self.workspace.canvas.set_canvas_level_offsets(0.425, -0.425)
        self.assertEqual(
            self.workspace.canvas_x_offset_value_label.text(),
            "0.425 px",
        )

        self.workspace.canvas_x_offset_increase_button.click()
        self.workspace.canvas_y_offset_decrease_button.click()

        self.assertAlmostEqual(self.level.canvas_offset_x_pixels, 1.425)
        self.assertAlmostEqual(self.level.canvas_offset_y_pixels, -1.425)
        self.assertEqual(
            self.workspace.canvas_x_offset_value_label.text(),
            "1.425 px",
        )


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
