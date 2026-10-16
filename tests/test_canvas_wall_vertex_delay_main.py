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
from housemaker.door_state import (
    DoorLibraryData,
    WindowPlacement,
    create_window_definition_for_frame,
)
from housemaker.main import BlueprintWorkspace
from housemaker.models import (
    DoorwayData,
    LevelData,
    OpenSpaceData,
    VertexData,
    WindowData,
)
from housemaker.project_io import ProjectData
from housemaker.surface_geometry import SURFACE_TYPE_WALL, build_fixed_surfaces
from tests.opening_dimension_assertions import get_opening_physical_size

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


def _build_level_with_disconnected_vertices(index: int = 2) -> LevelData:
    level = _build_level(index)
    level.vertex_data.add_vertex(10.0, 50.0)
    level.vertex_data.add_vertex(90.0, 50.0)
    return level


def _build_level_with_two_walls(index: int = 2) -> LevelData:
    level = _build_level(index)
    first = level.vertex_data.add_vertex(10.0, 25.0)
    second = level.vertex_data.add_vertex(90.0, 25.0)
    third = level.vertex_data.add_vertex(10.0, 75.0)
    fourth = level.vertex_data.add_vertex(90.0, 75.0)
    level.vertex_data.add_edge(first.id, second.id)
    level.vertex_data.add_edge(third.id, fourth.id)
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

    def test_new_vertex_uses_the_shared_mesh_edit_delay(self) -> None:
        timer = self.workspace._wall_vertex_update_timer
        self.assertTrue(timer.isSingleShot())
        self.assertEqual(timer.interval(), 1_000)

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
            self.assertFalse(timer.isActive())
            reconcile_assignments.assert_not_called()
            schedule_refresh.assert_not_called()

            self.workspace.canvas._apply_active_chain()
            self.assertTrue(timer.isActive())
            self.workspace._commit_pending_wall_vertex_update()

        reconcile_assignments.assert_called_once_with(self.workspace.levels)
        schedule_refresh.assert_called_once_with(preserve_camera=True)
        self.assertFalse(self.workspace._pending_wall_vertex_mesh_update)
        self.assertFalse(timer.isActive())

    def test_blank_addition_keeps_timer_frozen_after_left_release(self) -> None:
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

            self.assertEqual(len(canvas.vertex_data.vertices), 0)
            self.assertFalse(self.workspace._pending_wall_vertex_mesh_update)
            self.assertFalse(
                self.workspace._wall_vertex_update_timer.isActive()
            )
            reconcile_assignments.assert_not_called()
            schedule_refresh.assert_not_called()

            QTest.mouseRelease(
                canvas,
                Qt.MouseButton.LeftButton,
                pos=position,
            )

        self.assertEqual(len(canvas.vertex_data.vertices), 1)
        self.assertTrue(self.workspace._pending_wall_vertex_mesh_update)
        self.assertFalse(self.workspace._wall_vertex_update_timer.isActive())
        reconcile_assignments.assert_not_called()
        schedule_refresh.assert_not_called()

    def test_active_vertex_chain_keeps_mesh_delay_frozen_between_points(
        self,
    ) -> None:
        canvas = self.workspace.canvas
        timer = self.workspace._wall_vertex_update_timer
        timer.setInterval(20)
        first_position = self._image_position(20.0, 30.0)
        second_position = self._image_position(80.0, 30.0)

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
            QTest.mouseClick(
                canvas,
                Qt.MouseButton.LeftButton,
                pos=first_position,
            )

            self.assertIsNotNone(canvas.active_vertex_id)
            self.assertTrue(self.workspace._pending_wall_vertex_mesh_update)
            self.assertFalse(timer.isActive())
            QTest.qWait(50)
            reconcile_assignments.assert_not_called()
            schedule_refresh.assert_not_called()

            QTest.mouseClick(
                canvas,
                Qt.MouseButton.LeftButton,
                pos=second_position,
            )

            self.assertIsNotNone(canvas.active_vertex_id)
            self.assertEqual(len(canvas.vertex_data.vertices), 2)
            self.assertEqual(len(canvas.vertex_data.edges), 1)
            self.assertTrue(self.workspace._pending_wall_vertex_mesh_update)
            self.assertFalse(timer.isActive())
            QTest.qWait(50)

        reconcile_assignments.assert_not_called()
        schedule_refresh.assert_not_called()

    def test_finishing_vertex_chain_starts_delay_and_rebuilds_once(self) -> None:
        canvas = self.workspace.canvas
        timer = self.workspace._wall_vertex_update_timer
        timer.setInterval(20)
        position = self._image_position(20.0, 30.0)
        QTest.mouseClick(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=position,
        )
        self.assertIsNotNone(canvas.active_vertex_id)
        self.assertTrue(self.workspace._pending_wall_vertex_mesh_update)

        # Keep this assertion focused on the completion transition: ending a
        # pending chain must arm the shared delay even when it is disarmed.
        timer.stop()
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
            QTest.mouseClick(
                canvas,
                Qt.MouseButton.RightButton,
                pos=position,
            )

            self.assertIsNone(canvas.active_vertex_id)
            self.assertTrue(timer.isActive())
            reconcile_assignments.assert_not_called()
            schedule_refresh.assert_not_called()

            QTest.qWait(80)

        self.assertFalse(self.workspace._pending_wall_vertex_mesh_update)
        self.assertFalse(timer.isActive())
        reconcile_assignments.assert_called_once_with(self.workspace.levels)
        schedule_refresh.assert_called_once_with(preserve_camera=True)

    def test_edge_addition_keeps_timer_frozen_after_left_release(self) -> None:
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
        self.assertFalse(self.workspace._wall_vertex_update_timer.isActive())

    def test_dragging_chain_vertex_keeps_timer_frozen_after_left_release(
        self,
    ) -> None:
        canvas = self.workspace.canvas
        canvas._handle_new_vertex_click((20.0, 20.0))
        timer = self.workspace._wall_vertex_update_timer
        self.assertFalse(timer.isActive())
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
        self.assertFalse(timer.isActive())

    def test_moving_existing_vertex_waits_until_release_and_then_debounces(
        self,
    ) -> None:
        level = _build_level_with_wall()
        self._install_levels([level])
        canvas = self.workspace.canvas
        timer = self.workspace._wall_vertex_update_timer
        start = self._image_position(10.0, 50.0)
        target = self._image_position(30.0, 30.0)

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
                pos=start,
            )
            _send_drag_move(canvas, target)

            moved_vertex = level.vertex_data.vertices[0]
            self.assertNotEqual((moved_vertex.x, moved_vertex.y), (10.0, 50.0))
            self.assertFalse(timer.isActive())
            reconcile_assignments.assert_not_called()
            schedule_refresh.assert_not_called()

            QTest.mouseRelease(
                canvas,
                Qt.MouseButton.LeftButton,
                pos=target,
            )

        self.assertTrue(self.workspace._pending_wall_vertex_mesh_update)
        self.assertTrue(timer.isActive())
        reconcile_assignments.assert_not_called()
        schedule_refresh.assert_not_called()

    def test_delayed_vertex_move_preserves_window_size_and_undo_data(
        self,
    ) -> None:
        level = _build_level_with_wall()
        host_wall = next(
            surface
            for surface in build_fixed_surfaces((level,))
            if surface.surface_type == SURFACE_TYPE_WALL
        )
        original_window = WindowData(
            window_id="vertex-window",
            wall_surface_id=host_wall.surface_id,
            start_ratio=0.25,
            end_ratio=0.75,
            bottom_ratio=0.2,
            top_ratio=0.7,
        )
        level.windows = [original_window]
        self._install_levels([level])
        self.workspace._canvas_undo_stack.clear()
        initial_size = get_opening_physical_size(level, original_window.window_id)
        canvas = self.workspace.canvas
        start = self._image_position(90.0, 50.0)
        target = self._image_position(100.0, 50.0)

        QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=start)
        _send_drag_move(canvas, target)
        QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=target)
        self.assertTrue(self.workspace._pending_wall_vertex_mesh_update)

        self.workspace._commit_pending_wall_vertex_update()

        self.assertNotEqual(level.windows[0], original_window)
        current_size = get_opening_physical_size(level, original_window.window_id)
        self.assertAlmostEqual(current_size[0], initial_size[0])
        self.assertAlmostEqual(current_size[1], initial_size[1])

        self.workspace._handle_canvas_undo_requested()

        self.assertEqual(level.windows, [original_window])
        self.assertEqual(
            get_opening_physical_size(level, original_window.window_id),
            initial_size,
        )

    def test_active_vertex_chain_keeps_first_window_dimension_baseline(
        self,
    ) -> None:
        level = _build_level_with_wall()
        host_wall = next(
            surface
            for surface in build_fixed_surfaces((level,))
            if surface.surface_type == SURFACE_TYPE_WALL
        )
        window = WindowData(
            window_id="chain-window",
            wall_surface_id=host_wall.surface_id,
            start_ratio=0.25,
            end_ratio=0.75,
            bottom_ratio=0.2,
            top_ratio=0.7,
        )
        level.windows = [window]
        self._install_levels([level])
        self.workspace._canvas_undo_stack.clear()
        initial_size = get_opening_physical_size(level, window.window_id)
        canvas = self.workspace.canvas
        first_vertex, second_vertex = level.vertex_data.vertices

        canvas._set_active_vertex_id(first_vertex.id)
        canvas._push_undo_state()
        level.vertex_data.move_vertex(second_vertex.id, 100.0, 50.0)
        self.workspace._handle_canvas_wall_vertex_added()

        canvas._push_undo_state()
        canvas._set_active_vertex_id(None)
        self.workspace._commit_pending_wall_vertex_update()

        current_size = get_opening_physical_size(level, window.window_id)
        self.assertAlmostEqual(current_size[0], initial_size[0])
        self.assertAlmostEqual(current_size[1], initial_size[1])

    def test_connecting_existing_vertices_waits_until_release_and_debounces(
        self,
    ) -> None:
        level = _build_level_with_disconnected_vertices()
        self._install_levels([level])
        canvas = self.workspace.canvas
        timer = self.workspace._wall_vertex_update_timer
        first_position = self._image_position(10.0, 50.0)
        second_position = self._image_position(90.0, 50.0)
        first_vertex, second_vertex = level.vertex_data.vertices

        QTest.mouseClick(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=first_position,
        )
        self.assertEqual(canvas.active_vertex_id, first_vertex.id)
        self.assertFalse(self.workspace._pending_wall_vertex_mesh_update)
        self.assertFalse(timer.isActive())

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
                pos=second_position,
            )

            self.assertFalse(
                level.vertex_data.has_edge(first_vertex.id, second_vertex.id)
            )
            self.assertFalse(timer.isActive())
            reconcile_assignments.assert_not_called()
            schedule_refresh.assert_not_called()

            QTest.mouseRelease(
                canvas,
                Qt.MouseButton.LeftButton,
                pos=second_position,
            )

        self.assertTrue(
            level.vertex_data.has_edge(first_vertex.id, second_vertex.id)
        )
        self.assertTrue(self.workspace._pending_wall_vertex_mesh_update)
        self.assertFalse(timer.isActive())
        reconcile_assignments.assert_not_called()
        schedule_refresh.assert_not_called()

    def test_existing_vertex_selection_click_does_not_queue_mesh_work(
        self,
    ) -> None:
        level = _build_level_with_wall()
        self._install_levels([level])
        canvas = self.workspace.canvas
        position = self._image_position(10.0, 50.0)

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
            QTest.mouseClick(
                canvas,
                Qt.MouseButton.LeftButton,
                pos=position,
            )

        self.assertEqual(canvas.selected_vertex_id, level.vertex_data.vertices[0].id)
        self.assertFalse(self.workspace._pending_wall_vertex_mesh_update)
        self.assertFalse(self.workspace._wall_vertex_update_timer.isActive())
        reconcile_assignments.assert_not_called()
        schedule_refresh.assert_not_called()

    def test_new_vertex_press_without_movement_keeps_chain_frozen(
        self,
    ) -> None:
        canvas = self.workspace.canvas
        canvas._handle_new_vertex_click((20.0, 20.0))
        timer = self.workspace._wall_vertex_update_timer
        self.assertFalse(timer.isActive())
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
        self.assertFalse(timer.isActive())

    def test_repeated_additions_stay_frozen_and_coalesce_one_update(self) -> None:
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
            self.assertFalse(timer.isActive())

            self.workspace.canvas._handle_new_vertex_click((90.0, 10.0))

            self.assertFalse(timer.isActive())
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

            self.workspace.canvas._apply_active_chain()
            self.assertTrue(timer.isActive())
            self.workspace._commit_pending_wall_vertex_update()
            self.workspace._commit_pending_wall_vertex_update()

        reconcile_assignments.assert_called_once_with(self.workspace.levels)
        schedule_refresh.assert_called_once_with(preserve_camera=True)

    def test_general_change_during_pending_update_restarts_debounce(
        self,
    ) -> None:
        timer = self.workspace._wall_vertex_update_timer
        self.workspace.canvas._handle_new_vertex_click((20.0, 20.0))
        self.workspace.canvas._apply_active_chain()
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

    def test_multi_vertex_delete_rebuilds_the_3d_geometry_after_delay(
        self,
    ) -> None:
        level = _build_level_with_two_walls()
        self._install_levels([level])
        self.workspace.workspace_tabs.setCurrentWidget(
            self.workspace.scene_3d_workspace
        )
        _qt_application.processEvents()
        installed_model = self.workspace.viewer.model
        assert installed_model is not None
        installed_face_count = len(installed_model.mesh.faces)
        original_model = self.workspace._build_viewer_preview_model(None)
        assert original_model is not None
        original_face_count = len(original_model.mesh.faces)
        self.assertEqual(installed_face_count, original_face_count)
        original_revision = self.workspace._viewer_preview_revision
        deleted_vertex_ids = tuple(
            vertex.id for vertex in level.vertex_data.vertices[:2]
        )
        self.workspace.canvas.set_selected_vertex_ids(deleted_vertex_ids)

        self.workspace.canvas._delete_selected_vertices()

        self.assertTrue(self.workspace._pending_wall_vertex_mesh_update)
        self.assertTrue(self.workspace._wall_vertex_update_timer.isActive())
        self.assertEqual(
            self.workspace._viewer_preview_revision,
            original_revision,
        )
        self.assertTrue(
            all(
                level.vertex_data.get_vertex(vertex_id) is None
                for vertex_id in deleted_vertex_ids
            )
        )

        self.workspace._commit_pending_wall_vertex_update()

        self.assertFalse(self.workspace._pending_wall_vertex_mesh_update)
        self.assertGreater(
            self.workspace._viewer_preview_revision,
            original_revision,
        )
        rebuilt_model = self.workspace._build_viewer_preview_model(None)
        assert rebuilt_model is not None
        self.assertLess(len(rebuilt_model.mesh.faces), original_face_count)
        _qt_application.processEvents()
        refreshed_model = self.workspace.viewer.model
        assert refreshed_model is not None
        self.assertLess(len(refreshed_model.mesh.faces), installed_face_count)

    def test_vertex_delete_defers_dependent_doorway_and_surface_work(self) -> None:
        level = _build_level_with_wall()
        level.doorways.append(
            DoorwayData(
                center_x=50.0,
                center_y=50.0,
                width_meters=0.9,
                height_meters=2.1,
                rotation_degrees=90.0,
            )
        )
        self._install_levels([level])
        original_revision = self.workspace._viewer_preview_revision
        committed_doorways = self.workspace._viewer_doorways_by_level_index[
            level.index
        ]
        self.assertEqual(len(committed_doorways), 1)
        deleted_vertex_id = level.vertex_data.vertices[0].id
        self.workspace.canvas.set_selected_vertex_ids((deleted_vertex_id,))

        with (
            patch.object(
                self.workspace,
                "_reconcile_door_placements",
                return_value=False,
            ) as reconcile_doors,
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
            self.workspace.canvas._delete_selected_vertices()

            self.assertEqual(level.doorways, [])
            self.assertEqual(
                self.workspace._viewer_doorways_by_level_index[level.index],
                committed_doorways,
            )
            self.assertEqual(
                self.workspace._viewer_preview_revision,
                original_revision,
            )
            self.assertTrue(self.workspace._pending_wall_vertex_mesh_update)
            self.assertTrue(self.workspace._wall_vertex_update_timer.isActive())
            reconcile_doors.assert_not_called()
            reconcile_assignments.assert_not_called()
            schedule_refresh.assert_not_called()

            self.workspace._commit_pending_wall_vertex_update()

        self.assertEqual(
            self.workspace._viewer_doorways_by_level_index[level.index],
            (),
        )
        reconcile_doors.assert_called_once_with()
        reconcile_assignments.assert_called_once_with(self.workspace.levels)
        schedule_refresh.assert_called_once_with(preserve_camera=True)

    def test_open_space_corner_release_waits_for_mesh_edit_delay(self) -> None:
        level = self.workspace.current_level
        opening = OpenSpaceData("opening-a", 20.0, 20.0, 80.0, 80.0)
        level.open_spaces.append(opening)
        self.workspace.canvas.selected_open_space_id = opening.open_space_id
        self.workspace._set_mesh_edit_update_delay_seconds(2.4)
        original_revision = self.workspace._viewer_preview_revision
        start = self._image_position(20.0, 20.0)
        target = self._image_position(10.0, 12.0)

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
                self.workspace.canvas,
                Qt.MouseButton.LeftButton,
                pos=start,
            )
            _send_drag_move(self.workspace.canvas, target)
            QTest.mouseRelease(
                self.workspace.canvas,
                Qt.MouseButton.LeftButton,
                pos=target,
            )

            self.assertEqual(len(level.open_spaces), 1)
            resized_opening = level.open_spaces[0]
            self.assertEqual(resized_opening.open_space_id, "opening-a")
            self.assertAlmostEqual(resized_opening.minimum_x, 10.0, places=1)
            self.assertAlmostEqual(resized_opening.minimum_y, 12.0, places=1)
            self.assertEqual(
                (resized_opening.maximum_x, resized_opening.maximum_y),
                (80.0, 80.0),
            )
            self.assertEqual(
                self.workspace._viewer_preview_revision,
                original_revision,
            )
            self.assertTrue(self.workspace._pending_wall_vertex_mesh_update)
            self.assertTrue(self.workspace._wall_vertex_update_timer.isActive())
            self.assertEqual(
                self.workspace._wall_vertex_update_timer.interval(),
                2_400,
            )
            reconcile_assignments.assert_not_called()
            schedule_refresh.assert_not_called()

            self.workspace._commit_pending_wall_vertex_update()

        reconcile_assignments.assert_called_once_with(self.workspace.levels)
        schedule_refresh.assert_called_once_with(preserve_camera=True)

    def test_deleting_the_last_wall_clears_the_live_3d_geometry_after_delay(
        self,
    ) -> None:
        level = _build_level_with_wall()
        self._install_levels([level])
        self.workspace.workspace_tabs.setCurrentWidget(
            self.workspace.scene_3d_workspace
        )
        _qt_application.processEvents()
        self.assertIsNotNone(self.workspace.viewer.model)
        deleted_vertex_id = level.vertex_data.vertices[0].id
        self.workspace.canvas.set_selected_vertex_ids((deleted_vertex_id,))

        self.workspace.canvas._delete_selected_vertices()
        self.workspace._commit_pending_wall_vertex_update()
        _qt_application.processEvents()

        self.assertIsNone(level.vertex_data.get_vertex(deleted_vertex_id))
        self.assertEqual(len(level.vertex_data.vertices), 1)
        self.assertEqual(level.vertex_data.edges, [])
        self.assertIsNone(self.workspace.viewer.model)

    def test_wall_delete_removes_orphan_window_and_undo_restores_fixture(
        self,
    ) -> None:
        level = _build_level_with_two_walls()
        host_wall = next(
            surface
            for surface in build_fixed_surfaces((level,))
            if surface.surface_type == SURFACE_TYPE_WALL
            and surface.wall_key == "1:2"
        )
        frame = WindowData(
            window_id="deleted-wall-window-frame",
            wall_surface_id=host_wall.surface_id,
            start_ratio=0.25,
            end_ratio=0.75,
            bottom_ratio=0.2,
            top_ratio=0.7,
        )
        level.windows = [frame]
        self._install_levels([level])
        window = create_window_definition_for_frame(
            frame.window_id,
            1.0,
            1.0,
            name="Reusable window",
            window_id="reusable-window",
        )
        placement = WindowPlacement(
            placement_id="placed-window",
            window_id=window.window_id,
            level_index=level.index,
            window_frame_id=frame.window_id,
        )
        self.workspace.doors_workspace.set_data(
            DoorLibraryData(
                windows=[window],
                window_placements=[placement],
            )
        )
        self.workspace._canvas_undo_stack.clear()
        deleted_vertex_ids = tuple(
            vertex.id for vertex in level.vertex_data.vertices[:2]
        )
        self.workspace.canvas.set_selected_vertex_ids(deleted_vertex_ids)

        self.workspace.canvas._delete_selected_vertices()

        self.assertEqual(level.windows, [frame])
        self.assertEqual(
            self.workspace.doors_workspace.data().window_placements,
            [placement],
        )

        self.workspace._commit_pending_wall_vertex_update()

        detached_library = self.workspace.doors_workspace.data()
        self.assertEqual(level.windows, [])
        self.assertEqual(detached_library.window_placements, [])
        self.assertEqual(detached_library.get_window(window.window_id), window)

        self.workspace._handle_canvas_undo_requested()

        restored_library = self.workspace.doors_workspace.data()
        self.assertEqual(level.windows, [frame])
        self.assertEqual(restored_library.window_placements, [placement])
        self.assertEqual(restored_library.get_window(window.window_id), window)

    def test_project_load_removes_only_stale_orphan_window_frames(self) -> None:
        level = _build_level_with_wall()
        host_wall = next(
            surface
            for surface in build_fixed_surfaces((level,))
            if surface.surface_type == SURFACE_TYPE_WALL
        )
        live_frame = WindowData(
            window_id="live-frame",
            wall_surface_id=host_wall.surface_id,
            start_ratio=0.1,
            end_ratio=0.35,
            bottom_ratio=0.2,
            top_ratio=0.7,
        )
        stale_frame = WindowData(
            window_id="stale-frame",
            wall_surface_id=f"level:{level.index}/wall:998:999",
            start_ratio=0.55,
            end_ratio=0.8,
            bottom_ratio=0.2,
            top_ratio=0.7,
        )
        level.windows = [live_frame, stale_frame]
        window = create_window_definition_for_frame(
            stale_frame.window_id,
            1.0,
            1.0,
            name="Stale-frame asset",
            window_id="stale-frame-window",
        )
        placement = WindowPlacement(
            placement_id="stale-frame-placement",
            window_id=window.window_id,
            level_index=level.index,
            window_frame_id=stale_frame.window_id,
        )
        project = ProjectData(
            blueprint_path=None,
            current_level_index=0,
            levels=[level],
            doors=DoorLibraryData(
                windows=[window],
                window_placements=[placement],
            ),
        )

        self.workspace._apply_loaded_project(project)

        self.assertEqual(self.workspace.current_level.windows, [live_frame])
        loaded_library = self.workspace.doors_workspace.data()
        self.assertEqual(loaded_library.window_placements, [])
        self.assertEqual(loaded_library.get_window(window.window_id), window)

    def test_deleting_the_last_wall_clears_a_detached_3d_window(
        self,
    ) -> None:
        level = _build_level_with_wall()
        self._install_levels([level])
        screen = QApplication.primaryScreen()
        assert screen is not None
        with patch(
            "housemaker.main.resolve_fullscreen_3d_viewer_screen",
            return_value=screen,
        ):
            self.workspace._apply_scene_3d_display_screen(
                "screen:vertex-delete-test"
            )
        _qt_application.processEvents()
        self.assertTrue(self.workspace._external_scene_3d_host.is_active)
        self.assertIsNotNone(self.workspace.viewer.model)
        deleted_vertex_id = level.vertex_data.vertices[0].id
        self.workspace.canvas.set_selected_vertex_ids((deleted_vertex_id,))

        self.workspace.canvas._delete_selected_vertices()
        self.workspace._commit_pending_wall_vertex_update()
        _qt_application.processEvents()

        self.assertIsNone(self.workspace.viewer.model)
        self.assertEqual(
            self.workspace._canvas_viewer_preview_revision,
            self.workspace._viewer_preview_revision,
        )

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

    def test_mesh_edit_delay_updates_every_mesh_edit_timer(self) -> None:
        wall_timer = self.workspace._wall_vertex_update_timer
        doorway_timer = self.workspace._doorway_mesh_update_timer
        surface_timer = self.workspace._canvas_surface_mesh_update_timer

        self.workspace._handle_canvas_wall_vertex_added()
        self.assertTrue(wall_timer.isActive())

        self.workspace._set_mesh_edit_update_delay_seconds(2.4)

        self.assertEqual(doorway_timer.interval(), 2_400)
        self.assertEqual(surface_timer.interval(), 2_400)
        self.assertEqual(wall_timer.interval(), 2_400)
        self.assertTrue(wall_timer.isActive())

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
