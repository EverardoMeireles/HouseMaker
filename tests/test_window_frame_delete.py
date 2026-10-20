"""Deleting a selected frame preserves wall topology and reusable fixtures."""

# ### Environment setup ###
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.canvas_openings import CANVAS_OPENING_WINDOW, CanvasOpeningReference
from housemaker.door_state import (
    DoorLibraryData,
    WindowPlacement,
    create_window_definition_for_frame,
)
from housemaker.main import BlueprintWorkspace
from housemaker.surface_geometry import build_fixed_surfaces
from tests.test_window_solid_fit_canvas import (
    _application,
    _configure_canvas,
    _hover,
    _solid_level,
)


# ### Fixtures ###
class WindowFrameDeletionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(
                Path(self.temporary.name) / "settings.json"
            )
        )
        self.workspace._set_mesh_edit_update_delay_seconds(60.0)
        self.level = _solid_level()
        self.workspace.levels = [self.level]
        self.workspace.current_level_index = 0
        self.workspace._sync_canvas_to_current_level()
        self.workspace._reset_viewer_doorway_snapshots()
        _configure_canvas(self.workspace.canvas, self.level)
        placement = _hover(self.workspace.canvas)
        self.workspace._handle_canvas_window_placement_requested(placement)
        self.workspace._commit_pending_doorway_mesh_update()
        # These synthetic levels have no file-backed image and therefore no
        # viewer model to publish targets. Seed the semantic scene explicitly,
        # like the other Canvas opening integration fixtures do.
        self.workspace._set_canvas_viewer_targets(build_fixed_surfaces([self.level]))
        self.frame = self.level.windows[0]
        self.workspace.canvas.cancel_window_placement()
        self.addition_undo_stack = list(self.workspace._canvas_undo_stack)
        self.workspace._canvas_undo_stack.clear()
        self.before_graph = self.level.vertex_data.to_dict()
        self.reference = CanvasOpeningReference(
            kind=CANVAS_OPENING_WINDOW,
            level_index=self.level.index,
            item_index=0,
            stable_id=self.frame.window_id,
        )

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _application.processEvents()
        self.temporary.cleanup()

    def _select_on_canvas(self) -> None:
        self.workspace.canvas._set_selected_window_id(self.frame.window_id)

    def _install_reusable_window(self):
        asset = create_window_definition_for_frame(
            self.frame.window_id,
            1.0,
            1.0,
            name="Reusable test window",
            window_id="reusable-window",
        )
        placement = WindowPlacement(
            placement_id="installed-window",
            window_id=asset.window_id,
            level_index=self.level.index,
            window_frame_id=self.frame.window_id,
        )
        self.workspace.doors_workspace.set_data(
            DoorLibraryData(windows=[asset], window_placements=[placement])
        )
        self.workspace._sync_canvas_opening_fixture_keys()
        return asset, placement

    # ### Structural frame deletion ###
    def test_canvas_delete_removes_only_frame_and_ctrl_z_restores_it(self):
        self.workspace.show()
        self.workspace.canvas.setFocus()
        _application.processEvents()
        self._select_on_canvas()

        QTest.keyClick(self.workspace.canvas, Qt.Key.Key_Delete)

        self.assertEqual(self.level.windows, [])
        self.assertEqual(self.level.vertex_data.to_dict(), self.before_graph)
        self.assertEqual(len(self.workspace._canvas_undo_stack), 1)
        QTest.keyClick(
            self.workspace.canvas,
            Qt.Key.Key_Z,
            Qt.KeyboardModifier.ControlModifier,
        )
        self.assertEqual(self.level.windows, [self.frame])
        self.assertEqual(self.level.vertex_data.to_dict(), self.before_graph)
        self.assertEqual(self.workspace._canvas_undo_stack, [])

    def test_3d_delete_signal_removes_frame_and_shared_undo_restores_it(self):
        self.workspace.viewer.select_canvas_opening(None)
        self.assertTrue(self.workspace.viewer.select_canvas_opening(self.reference))

        self.workspace.viewer._handle_view_delete_requested()

        self.assertEqual(self.level.windows, [])
        self.assertEqual(self.level.vertex_data.to_dict(), self.before_graph)
        self.assertEqual(len(self.workspace._canvas_undo_stack), 1)
        self.workspace._handle_canvas_undo_requested()
        self.assertEqual(self.level.windows, [self.frame])
        self.assertEqual(self.level.vertex_data.to_dict(), self.before_graph)

    def test_undo_frame_deletion_then_placement_restores_original_unsplit_walls(self):
        self.workspace._canvas_undo_stack[:] = self.addition_undo_stack
        original_graph = _solid_level().vertex_data.to_dict()
        self._select_on_canvas()
        QTest.keyClick(self.workspace.canvas, Qt.Key.Key_Delete)
        self.assertEqual(self.level.windows, [])

        self.workspace._handle_canvas_undo_requested()

        self.assertEqual(self.level.windows, [self.frame])
        self.assertEqual(self.level.vertex_data.to_dict(), self.before_graph)
        self.workspace._handle_canvas_undo_requested()
        self.assertEqual(self.level.windows, [])
        self.assertEqual(self.level.vertex_data.to_dict(), original_graph)
        self.assertEqual(self.workspace._canvas_undo_stack, [])

    # ### Installed fixture removal ###
    def test_first_delete_unplaces_window_but_keeps_frame_and_library_asset(self):
        asset, _placement = self._install_reusable_window()
        self._select_on_canvas()

        QTest.keyClick(self.workspace.canvas, Qt.Key.Key_Delete)

        self.assertEqual(self.level.windows, [self.frame])
        self.assertEqual(self.level.vertex_data.to_dict(), self.before_graph)
        library = self.workspace.doors_workspace.data()
        self.assertEqual(library.window_placements, [])
        self.assertEqual(library.windows, [asset])

    def test_auto_repeat_after_unplacing_window_does_not_delete_frame(self):
        asset, _placement = self._install_reusable_window()
        self._select_on_canvas()
        QTest.keyClick(self.workspace.canvas, Qt.Key.Key_Delete)
        self.assertEqual(self.level.windows, [self.frame])
        event = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_Delete,
            Qt.KeyboardModifier.NoModifier,
            "",
            True,
            2,
        )

        QApplication.sendEvent(self.workspace.canvas, event)

        self.assertEqual(self.level.windows, [self.frame])
        self.assertEqual(self.level.vertex_data.to_dict(), self.before_graph)
        self.assertEqual(self.workspace.doors_workspace.data().windows, [asset])

    def test_3d_auto_repeat_after_unplacing_window_does_not_delete_frame(self):
        asset, _placement = self._install_reusable_window()
        self.workspace.viewer.select_canvas_opening(None)
        self.assertTrue(self.workspace.viewer.select_canvas_opening(self.reference))
        QTest.keyClick(self.workspace.viewer.view, Qt.Key.Key_Delete)
        self.assertEqual(self.level.windows, [self.frame])
        event = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_Delete,
            Qt.KeyboardModifier.NoModifier,
            "",
            True,
            2,
        )

        QApplication.sendEvent(self.workspace.viewer.view, event)

        self.assertEqual(self.level.windows, [self.frame])
        self.assertEqual(self.level.vertex_data.to_dict(), self.before_graph)
        self.assertEqual(self.workspace.doors_workspace.data().windows, [asset])

    def test_second_deliberate_delete_removes_unoccupied_frame_only(self):
        asset, _placement = self._install_reusable_window()
        self._select_on_canvas()
        QTest.keyClick(self.workspace.canvas, Qt.Key.Key_Delete)
        self.assertEqual(self.level.windows, [self.frame])
        self._select_on_canvas()

        QTest.keyClick(self.workspace.canvas, Qt.Key.Key_Delete)

        self.assertEqual(self.level.windows, [])
        self.assertEqual(self.level.vertex_data.to_dict(), self.before_graph)
        self.assertEqual(self.workspace.doors_workspace.data().windows, [asset])


# ### Test runner ###
if __name__ == "__main__":
    unittest.main()
