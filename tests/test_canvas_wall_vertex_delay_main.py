# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QImage, QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.blueprint_canvas import BlueprintCanvas
from housemaker.main import BlueprintWorkspace
from housemaker.models import LevelData, VertexData


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_level(index: int = 2) -> LevelData:
    return LevelData(
        index=index,
        name=f"Level {index}",
        vertex_data=VertexData(),
    )


def _build_level_with_wall(index: int = 2) -> LevelData:
    level = _build_level(index)
    first = level.vertex_data.add_vertex(10.0, 50.0)
    second = level.vertex_data.add_vertex(90.0, 50.0)
    level.vertex_data.add_edge(first.id, second.id)
    return level


def _send_drag_move(canvas: BlueprintCanvas, position: QPoint) -> None:
    event = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(position),
        QPointF(canvas.mapToGlobal(position)),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(canvas, event)


# ### Wall vertex delay integration tests ###
class CanvasWallVertexDelayMainTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(
                Path(self._temporary_directory.name) / "settings.json"
            )
        )
        self.workspace.resize(1400, 850)
        self.workspace.show()
        self.workspace.workspace_tabs.setCurrentWidget(
            self.workspace.canvas_viewer_workspace
        )
        _qt_application.processEvents()
        self._install_levels([_build_level()])

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self._temporary_directory.cleanup()

    def _install_levels(
        self,
        levels: list[LevelData],
        *,
        current_level_index: int = 0,
    ) -> None:
        self.workspace.levels = levels
        self.workspace.current_level_index = current_level_index
        self.workspace._reset_viewer_doorway_snapshots()
        self.workspace._sync_level_controls()
        self.workspace._sync_canvas_to_current_level()
        self.workspace.canvas.blueprint_image = QImage(
            100,
            100,
            QImage.Format.Format_RGB32,
        )
        self.workspace.canvas.blueprint_image.fill(Qt.GlobalColor.white)
        self.workspace.canvas.update()
        _qt_application.processEvents()

    def _image_position(self, x: float, y: float) -> QPoint:
        return self.workspace.canvas._image_to_widget(x, y).toPoint()

    def _assert_pointer_hold_defers_timer(self) -> None:
        self.assertTrue(self.workspace._pending_wall_vertex_mesh_update)
        self.assertFalse(self.workspace._wall_vertex_update_timer.isActive())

    def test_new_vertex_defers_mesh_work_for_fifteen_seconds(self) -> None:
        timer = self.workspace._wall_vertex_update_timer
        self.assertTrue(timer.isSingleShot())
        self.assertEqual(timer.interval(), 15_000)

        with (
            patch.object(
                self.workspace.surface_texture_generation,
                "reconcile_assignments_with_levels",
                return_value=False,
            ) as reconcile_assignments,
            patch.object(
                self.workspace,
                "_schedule_viewer_preview_refresh",
            ) as schedule_refresh,
        ):
            self.workspace.canvas._handle_new_vertex_click((20.0, 30.0))

            self.assertEqual(
                len(self.workspace.current_level.vertex_data.vertices),
                1,
            )
            self.assertTrue(self.workspace._pending_wall_vertex_mesh_update)
            self.assertTrue(timer.isActive())
            reconcile_assignments.assert_not_called()
            schedule_refresh.assert_not_called()

            self.workspace._commit_pending_wall_vertex_update()

        reconcile_assignments.assert_called_once_with(self.workspace.levels)
        schedule_refresh.assert_called_once_with(preserve_camera=True)
        self.assertFalse(self.workspace._pending_wall_vertex_mesh_update)
        self.assertFalse(timer.isActive())

    def test_blank_addition_starts_timer_only_on_left_release(self) -> None:
        canvas = self.workspace.canvas
        position = self._image_position(20.0, 30.0)

        with (
            patch.object(
                self.workspace.surface_texture_generation,
                "reconcile_assignments_with_levels",
                return_value=False,
            ) as reconcile_assignments,
            patch.object(
                self.workspace,
                "_schedule_viewer_preview_refresh",
            ) as schedule_refresh,
        ):
            QTest.mousePress(
                canvas,
                Qt.MouseButton.LeftButton,
                pos=position,
            )

            self.assertEqual(len(canvas.vertex_data.vertices), 1)
            self._assert_pointer_hold_defers_timer()
            reconcile_assignments.assert_not_called()
            schedule_refresh.assert_not_called()

            QTest.mouseRelease(
                canvas,
                Qt.MouseButton.LeftButton,
                pos=position,
            )

        self.assertTrue(self.workspace._pending_wall_vertex_mesh_update)
        self.assertTrue(self.workspace._wall_vertex_update_timer.isActive())
        reconcile_assignments.assert_not_called()
        schedule_refresh.assert_not_called()

    def test_edge_addition_starts_timer_only_on_left_release(self) -> None:
        level = _build_level_with_wall()
        self._install_levels([level])
        canvas = self.workspace.canvas
        position = self._image_position(50.0, 50.0)

        QTest.mousePress(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=position,
        )

        self.assertEqual(len(level.vertex_data.vertices), 3)
        self.assertEqual(len(level.vertex_data.edges), 2)
        self._assert_pointer_hold_defers_timer()

        QTest.mouseRelease(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=position,
        )

        self.assertTrue(self.workspace._pending_wall_vertex_mesh_update)
        self.assertTrue(self.workspace._wall_vertex_update_timer.isActive())

    def test_dragging_new_vertex_pauses_timer_until_left_release(
        self,
    ) -> None:
        canvas = self.workspace.canvas
        canvas._handle_new_vertex_click((20.0, 20.0))
        timer = self.workspace._wall_vertex_update_timer
        self.assertTrue(timer.isActive())
        start = self._image_position(20.0, 20.0)
        target = self._image_position(70.0, 60.0)

        QTest.mousePress(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=start,
        )

        self._assert_pointer_hold_defers_timer()

        _send_drag_move(canvas, target)

        self._assert_pointer_hold_defers_timer()
        moved_vertex = canvas.vertex_data.vertices[0]
        self.assertNotEqual((moved_vertex.x, moved_vertex.y), (20.0, 20.0))

        QTest.mouseRelease(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=target,
        )

        self.assertTrue(self.workspace._pending_wall_vertex_mesh_update)
        self.assertTrue(timer.isActive())

    def test_new_vertex_press_without_movement_pauses_until_release(
        self,
    ) -> None:
        canvas = self.workspace.canvas
        canvas._handle_new_vertex_click((20.0, 20.0))
        timer = self.workspace._wall_vertex_update_timer
        self.assertTrue(timer.isActive())
        position = self._image_position(20.0, 20.0)

        QTest.mousePress(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=position,
        )

        self._assert_pointer_hold_defers_timer()

        QTest.mouseRelease(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=position,
        )

        self.assertTrue(self.workspace._pending_wall_vertex_mesh_update)
        self.assertTrue(timer.isActive())

    def test_repeated_additions_restart_and_coalesce_one_update(self) -> None:
        timer = self.workspace._wall_vertex_update_timer
        with (
            patch.object(
                self.workspace.surface_texture_generation,
                "reconcile_assignments_with_levels",
                return_value=False,
            ) as reconcile_assignments,
            patch.object(
                self.workspace,
                "_schedule_viewer_preview_refresh",
            ) as schedule_refresh,
        ):
            self.workspace.canvas._handle_new_vertex_click((10.0, 10.0))
            QTest.qWait(30)
            remaining_before_second_addition = timer.remainingTime()

            self.workspace.canvas._handle_new_vertex_click((90.0, 10.0))

            self.assertGreater(
                timer.remainingTime(),
                remaining_before_second_addition,
            )
            self.assertEqual(
                len(self.workspace.current_level.vertex_data.vertices),
                2,
            )
            self.assertEqual(
                len(self.workspace.current_level.vertex_data.edges),
                1,
            )
            reconcile_assignments.assert_not_called()
            schedule_refresh.assert_not_called()

            self.workspace._commit_pending_wall_vertex_update()
            self.workspace._commit_pending_wall_vertex_update()

        reconcile_assignments.assert_called_once_with(self.workspace.levels)
        schedule_refresh.assert_called_once_with(preserve_camera=True)

    def test_general_change_during_pending_update_restarts_debounce(
        self,
    ) -> None:
        timer = self.workspace._wall_vertex_update_timer
        self.workspace.canvas._handle_new_vertex_click((20.0, 20.0))
        QTest.qWait(30)
        remaining_before_general_change = timer.remainingTime()

        with (
            patch.object(
                self.workspace.surface_texture_generation,
                "reconcile_assignments_with_levels",
                return_value=False,
            ) as reconcile_assignments,
            patch.object(
                self.workspace,
                "_schedule_viewer_preview_refresh",
            ) as schedule_refresh,
        ):
            self.workspace.canvas.geometry_changed.emit()

        self.assertTrue(self.workspace._pending_wall_vertex_mesh_update)
        self.assertTrue(timer.isActive())
        self.assertGreater(timer.remainingTime(), remaining_before_general_change)
        reconcile_assignments.assert_not_called()
        schedule_refresh.assert_not_called()

    def test_ordinary_general_change_remains_immediate(self) -> None:
        timer = self.workspace._wall_vertex_update_timer
        with (
            patch.object(
                self.workspace.surface_texture_generation,
                "reconcile_assignments_with_levels",
                return_value=False,
            ) as reconcile_assignments,
            patch.object(
                self.workspace,
                "_schedule_viewer_preview_refresh",
            ) as schedule_refresh,
        ):
            self.workspace.canvas.geometry_changed.emit()

        reconcile_assignments.assert_called_once_with(self.workspace.levels)
        schedule_refresh.assert_called_once_with()
        self.assertFalse(self.workspace._pending_wall_vertex_mesh_update)
        self.assertFalse(timer.isActive())

    def test_wall_and_mesh_edit_delay_timers_are_independent(self) -> None:
        wall_timer = self.workspace._wall_vertex_update_timer
        doorway_timer = self.workspace._doorway_mesh_update_timer
        surface_timer = self.workspace._canvas_surface_mesh_update_timer

        self.workspace._set_mesh_edit_update_delay_seconds(2.4)

        self.assertEqual(doorway_timer.interval(), 2_400)
        self.assertEqual(surface_timer.interval(), 2_400)
        self.assertEqual(wall_timer.interval(), 15_000)

        self.workspace._set_wall_vertex_update_delay_seconds(18.5)

        self.assertEqual(wall_timer.interval(), 18_500)
        self.assertEqual(doorway_timer.interval(), 2_400)
        self.assertEqual(surface_timer.interval(), 2_400)

    def test_surface_selection_commits_pending_vertex_update(self) -> None:
        self.workspace.canvas._handle_new_vertex_click((30.0, 40.0))
        with patch.object(
            self.workspace,
            "_reconcile_canvas_surface_edit_and_refresh",
        ) as reconcile_and_refresh:
            self.workspace._handle_canvas_surface_selection_changed(())

        reconcile_and_refresh.assert_called_once_with()
        self.assertFalse(self.workspace._pending_wall_vertex_mesh_update)
        self.assertFalse(self.workspace._wall_vertex_update_timer.isActive())

    def test_project_replacement_and_shutdown_cancel_pending_update(
        self,
    ) -> None:
        self.workspace.canvas._handle_new_vertex_click((10.0, 15.0))
        self.assertTrue(self.workspace._pending_wall_vertex_mesh_update)

        replacement_level = _build_level(index=7)
        with patch.object(
            self.workspace,
            "_reconcile_canvas_surface_edit_and_refresh",
        ) as reconcile_and_refresh:
            self.workspace._apply_project_state(
                levels=[replacement_level],
                current_level_index=0,
            )

        reconcile_and_refresh.assert_not_called()
        self.assertFalse(self.workspace._pending_wall_vertex_mesh_update)
        self.assertFalse(self.workspace._wall_vertex_update_timer.isActive())
        self.assertIs(self.workspace.current_level, replacement_level)

        self.workspace._handle_canvas_wall_vertex_added()
        self.assertTrue(self.workspace._pending_wall_vertex_mesh_update)
        with patch.object(
            self.workspace,
            "_reconcile_canvas_surface_edit_and_refresh",
        ) as reconcile_and_refresh:
            self.workspace.shutdown()

        reconcile_and_refresh.assert_not_called()
        self.assertFalse(self.workspace._pending_wall_vertex_mesh_update)
        self.assertFalse(self.workspace._wall_vertex_update_timer.isActive())


if __name__ == "__main__":
    unittest.main()
