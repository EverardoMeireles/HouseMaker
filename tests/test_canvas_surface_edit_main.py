# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.canvas_surface_edits import (
    CANVAS_SURFACE_EDIT_AXIS_X,
    CANVAS_SURFACE_EDIT_LEVEL_HEIGHT,
    CANVAS_SURFACE_EDIT_ROOM_HEIGHT,
    CANVAS_SURFACE_EDIT_WALL_VERTEX,
    CanvasSurfaceEdit,
    CanvasSurfaceEditHandleTarget,
)
from housemaker.level_coordinates import level_image_to_world_xy
from housemaker.main import BlueprintWorkspace
from housemaker.models import LevelData, RoomData, VertexData
from housemaker.surface_geometry import (
    SURFACE_TYPE_CEILING,
    SURFACE_TYPE_WALL,
    FixedSurface,
    build_fixed_surfaces,
)


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_plain_connected_level() -> LevelData:
    """Return two walls that share the endpoint edited by the tests."""

    vertex_data = VertexData()
    first = vertex_data.add_vertex(0.0, 0.0)
    shared = vertex_data.add_vertex(100.0, 0.0)
    third = vertex_data.add_vertex(100.0, 100.0)
    vertex_data.add_edge(first.id, shared.id)
    vertex_data.add_edge(shared.id, third.id)
    return LevelData(
        index=2,
        name="Ground",
        height_meters=3.0,
        vertex_data=vertex_data,
    )


def _build_square_level(*, with_room: bool) -> LevelData:
    vertex_data = VertexData()
    boundary_ids = tuple(
        vertex_data.add_vertex(*point).id
        for point in (
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (0.0, 100.0),
        )
    )
    for start_id, end_id in zip(
        boundary_ids,
        (*boundary_ids[1:], boundary_ids[0]),
    ):
        vertex_data.add_edge(start_id, end_id)

    rooms: list[RoomData] = []
    if with_room:
        center = vertex_data.add_vertex(50.0, 50.0)
        rooms.append(
            RoomData(
                name="Room",
                vertex_ids=boundary_ids,
                center_vertex_id=center.id,
                color_rgb=(120, 160, 200),
                height_meters=3.5,
            )
        )
    return LevelData(
        index=2,
        name="Ground",
        height_meters=3.0,
        vertex_data=vertex_data,
        rooms=rooms,
    )


def _surface(
    surfaces: tuple[FixedSurface, ...],
    *,
    surface_type: str,
    room_owned: bool | None = None,
    wall_key: str | None = None,
) -> FixedSurface:
    return next(
        surface
        for surface in surfaces
        if surface.surface_type == surface_type
        and (room_owned is None or (surface.room_index is not None) == room_owned)
        and (wall_key is None or surface.wall_key == wall_key)
    )


def _edit(
    target: CanvasSurfaceEditHandleTarget,
    delta_meters: float,
) -> CanvasSurfaceEdit:
    return CanvasSurfaceEdit(
        reference=target.reference,
        surface_id=target.surface_id,
        delta_meters=delta_meters,
    )


# ### Main integration tests ###
class CanvasSurfaceEditMainTests(unittest.TestCase):
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
        self.workspace.canvas_viewer_tabs.setCurrentIndex(
            self.workspace.canvas_3d_view_tab_index
        )
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self._temporary_directory.cleanup()

    def _install_level(
        self,
        level: LevelData,
    ) -> tuple[FixedSurface, ...]:
        self.workspace.viewer.cancel_canvas_surface_edit()
        self.workspace.levels = [level]
        self.workspace.current_level_index = 0
        self.workspace._reset_viewer_doorway_snapshots()
        self.workspace._sync_level_controls()
        self.workspace._sync_canvas_to_current_level()
        surfaces = tuple(build_fixed_surfaces([level]))
        self.workspace._set_canvas_viewer_targets(surfaces)
        # The fixture installs semantic targets directly without asking Main
        # to build a replacement model. Treat that manually installed view as
        # current so cancellation assertions only observe the edit itself.
        self.workspace._canvas_viewer_preview_revision = (
            self.workspace._viewer_preview_revision
        )
        return surfaces

    def _target(
        self,
        surface_id: str,
        kind: str,
        *,
        vertex_id: int | None = None,
        axis_index: int | None = None,
    ) -> CanvasSurfaceEditHandleTarget:
        return next(
            target
            for target in self.workspace._canvas_surface_edit_targets_by_key.values()
            if target.surface_id == surface_id
            and target.reference.kind == kind
            and (
                vertex_id is None
                or target.reference.vertex_id == vertex_id
            )
            and (
                axis_index is None
                or target.reference.axis_index == axis_index
            )
        )

    def test_wall_preview_is_absolute_live_and_does_not_rebuild(self) -> None:
        level = _build_plain_connected_level()
        surfaces = self._install_level(level)
        wall = _surface(
            surfaces,
            surface_type=SURFACE_TYPE_WALL,
            wall_key="1:2",
        )
        target = self._target(
            wall.surface_id,
            CANVAS_SURFACE_EDIT_WALL_VERTEX,
            vertex_id=2,
            axis_index=CANVAS_SURFACE_EDIT_AXIS_X,
        )
        initial_edges = tuple(level.vertex_data.edges)

        with (
            patch.object(
                self.workspace,
                "_build_viewer_preview_model",
            ) as build_preview,
            patch.object(
                self.workspace,
                "_schedule_viewer_preview_refresh",
            ) as schedule_refresh,
            patch.object(
                self.workspace.surface_texture_generation,
                "reconcile_assignments_with_levels",
            ) as reconcile_assignments,
        ):
            self.workspace._handle_canvas_surface_edit_started(
                _edit(target, 0.0)
            )
            self.workspace._handle_canvas_surface_edit_preview_changed(
                _edit(target, 0.4)
            )
            first_preview_x = level.vertex_data.get_vertex(2).x
            self.workspace._handle_canvas_surface_edit_preview_changed(
                _edit(target, 0.8)
            )

            self.assertFalse(
                self.workspace._canvas_surface_mesh_update_timer.isActive()
            )
            self.assertFalse(
                self.workspace._pending_canvas_surface_mesh_update
            )

        self.assertAlmostEqual(first_preview_x, 120.0)
        self.assertAlmostEqual(level.vertex_data.get_vertex(2).x, 140.0)
        self.assertIs(
            self.workspace.canvas.vertex_data,
            level.vertex_data,
        )
        self.assertEqual(tuple(level.vertex_data.edges), initial_edges)
        connected_wall = next(
            surface
            for surface in build_fixed_surfaces([level])
            if surface.wall_key == "2:3"
        )
        shared_world_x = level_image_to_world_xy(level, 140.0, 0.0)[0]
        self.assertTrue(
            any(
                abs(point[0] - shared_world_x) <= 1e-9
                for point in (
                    connected_wall.wall_start_world,
                    connected_wall.wall_end_world,
                )
                if point is not None
            )
        )
        build_preview.assert_not_called()
        schedule_refresh.assert_not_called()
        reconcile_assignments.assert_not_called()
        self.workspace._handle_canvas_surface_edit_cancelled(
            _edit(target, 0.0)
        )

    def test_wall_cancel_restores_the_exact_drag_baseline(self) -> None:
        level = _build_plain_connected_level()
        surfaces = self._install_level(level)
        wall = _surface(
            surfaces,
            surface_type=SURFACE_TYPE_WALL,
            wall_key="1:2",
        )
        target = self._target(
            wall.surface_id,
            CANVAS_SURFACE_EDIT_WALL_VERTEX,
            vertex_id=2,
            axis_index=CANVAS_SURFACE_EDIT_AXIS_X,
        )
        baseline_vertices = tuple(level.vertex_data.vertices)
        baseline_offsets = (
            level.offset_x_meters,
            level.offset_y_meters,
        )
        revision_before = self.workspace._viewer_preview_revision

        with patch.object(
            self.workspace,
            "_build_viewer_preview_model",
        ) as build_preview:
            self.workspace._handle_canvas_surface_edit_started(
                _edit(target, 0.0)
            )
            self.workspace._handle_canvas_surface_edit_preview_changed(
                _edit(target, 0.75)
            )
            self.workspace._handle_canvas_surface_edit_cancelled(
                _edit(target, 0.0)
            )
            _qt_application.processEvents()

        self.assertEqual(tuple(level.vertex_data.vertices), baseline_vertices)
        self.assertEqual(
            (level.offset_x_meters, level.offset_y_meters),
            baseline_offsets,
        )
        self.assertIsNone(self.workspace._active_canvas_surface_edit_target)
        self.assertEqual(
            self.workspace._viewer_preview_revision,
            revision_before,
        )
        build_preview.assert_not_called()

    def test_wall_finish_waits_for_delay_before_reconcile_and_refresh(
        self,
    ) -> None:
        level = _build_plain_connected_level()
        surfaces = self._install_level(level)
        wall = _surface(
            surfaces,
            surface_type=SURFACE_TYPE_WALL,
            wall_key="1:2",
        )
        target = self._target(
            wall.surface_id,
            CANVAS_SURFACE_EDIT_WALL_VERTEX,
            vertex_id=2,
            axis_index=CANVAS_SURFACE_EDIT_AXIS_X,
        )
        final_edit = _edit(target, 0.5)
        self.workspace._handle_canvas_surface_edit_started(
            _edit(target, 0.0)
        )
        self.workspace._handle_canvas_surface_edit_preview_changed(final_edit)

        with (
            patch(
                "housemaker.canvas_surface_edits.build_fixed_surfaces",
                wraps=build_fixed_surfaces,
            ) as validate_surfaces,
            patch.object(
                self.workspace.surface_texture_generation,
                "reconcile_assignments_with_levels",
                return_value=False,
            ) as reconcile_assignments,
            patch.object(
                self.workspace,
                "_schedule_viewer_preview_refresh",
            ) as schedule_refresh,
            patch.object(
                self.workspace,
                "_build_viewer_preview_model",
            ) as build_preview,
        ):
            self.workspace._handle_canvas_surface_edit_finished(
                final_edit,
                True,
            )

            self.assertTrue(
                self.workspace._canvas_surface_mesh_update_timer.isActive()
            )
            self.assertTrue(
                self.workspace._pending_canvas_surface_mesh_update
            )
            reconcile_assignments.assert_not_called()
            schedule_refresh.assert_not_called()
            validate_surfaces.assert_not_called()

            self.workspace._commit_pending_canvas_surface_mesh_update()

        validate_surfaces.assert_called_once()
        reconcile_assignments.assert_called_once_with([level])
        schedule_refresh.assert_called_once_with(preserve_camera=True)
        build_preview.assert_not_called()
        self.assertFalse(
            self.workspace._canvas_surface_mesh_update_timer.isActive()
        )
        self.assertFalse(self.workspace._pending_canvas_surface_mesh_update)
        self.assertIsNone(self.workspace._active_canvas_surface_edit_target)
        self.assertEqual(self.workspace._canvas_surface_edit_targets_by_key, {})
        self.assertEqual(
            self.workspace.viewer._get_active_canvas_surface_edit_targets(),
            (),
        )
        self.assertAlmostEqual(level.vertex_data.get_vertex(2).x, 125.0)

    def test_delayed_wall_validation_failure_restores_and_rearms(self) -> None:
        level = _build_plain_connected_level()
        surfaces = self._install_level(level)
        wall = _surface(
            surfaces,
            surface_type=SURFACE_TYPE_WALL,
            wall_key="1:2",
        )
        target = self._target(
            wall.surface_id,
            CANVAS_SURFACE_EDIT_WALL_VERTEX,
            vertex_id=2,
            axis_index=CANVAS_SURFACE_EDIT_AXIS_X,
        )
        final_edit = _edit(target, 0.5)
        self.workspace.viewer.select_canvas_surface_target(wall.surface_id)
        self.workspace._handle_canvas_surface_edit_started(_edit(target, 0.0))
        self.workspace._handle_canvas_surface_edit_preview_changed(final_edit)

        with (
            patch(
                "housemaker.canvas_surface_edits._validate_generated_surfaces",
                side_effect=ValueError("invalid delayed wall"),
            ),
            patch.object(
                self.workspace.surface_texture_generation,
                "reconcile_assignments_with_levels",
            ) as reconcile_assignments,
        ):
            self.workspace._handle_canvas_surface_edit_finished(final_edit, True)
            self.assertAlmostEqual(level.vertex_data.get_vertex(2).x, 125.0)
            self.workspace._commit_pending_canvas_surface_mesh_update()

        self.assertAlmostEqual(level.vertex_data.get_vertex(2).x, 100.0)
        self.assertFalse(self.workspace._pending_canvas_surface_mesh_update)
        self.assertIsNone(
            self.workspace._pending_canvas_surface_mesh_baseline
        )
        self.assertTrue(self.workspace._canvas_surface_edit_targets_by_key)
        self.assertTrue(
            self.workspace.viewer._get_active_canvas_surface_edit_targets()
        )
        assert self.workspace.viewer.window_tools_status_label is not None
        self.assertIn(
            "invalid delayed wall",
            self.workspace.viewer.window_tools_status_label.text(),
        )
        reconcile_assignments.assert_not_called()

    def test_pending_wall_edit_blocks_unrelated_preview_rebuilds(self) -> None:
        level = _build_plain_connected_level()
        surfaces = self._install_level(level)
        wall = _surface(
            surfaces,
            surface_type=SURFACE_TYPE_WALL,
            wall_key="1:2",
        )
        target = self._target(
            wall.surface_id,
            CANVAS_SURFACE_EDIT_WALL_VERTEX,
            vertex_id=2,
            axis_index=CANVAS_SURFACE_EDIT_AXIS_X,
        )
        final_edit = _edit(target, 0.5)
        self.workspace._handle_canvas_surface_edit_started(_edit(target, 0.0))
        self.workspace._handle_canvas_surface_edit_preview_changed(final_edit)
        self.workspace._handle_canvas_surface_edit_finished(final_edit, True)
        revision_before = self.workspace._viewer_preview_revision

        with patch.object(
            self.workspace,
            "_build_viewer_preview_model",
        ) as build_preview:
            self.workspace._schedule_viewer_preview_refresh(
                preserve_camera=True
            )
            self.workspace._refresh_viewer_preview(preserve_camera=True)
            self.workspace._run_scheduled_viewer_preview_refresh()
            _qt_application.processEvents()

        self.assertEqual(
            self.workspace._viewer_preview_revision,
            revision_before + 1,
        )
        self.assertTrue(self.workspace._pending_canvas_surface_mesh_update)
        self.assertFalse(self.workspace._is_viewer_refresh_scheduled)
        build_preview.assert_not_called()

    def test_shared_delay_change_restarts_pending_wall_timer(self) -> None:
        level = _build_plain_connected_level()
        surfaces = self._install_level(level)
        wall = _surface(
            surfaces,
            surface_type=SURFACE_TYPE_WALL,
            wall_key="1:2",
        )
        target = self._target(
            wall.surface_id,
            CANVAS_SURFACE_EDIT_WALL_VERTEX,
            vertex_id=2,
            axis_index=CANVAS_SURFACE_EDIT_AXIS_X,
        )
        final_edit = _edit(target, 0.5)
        self.workspace._handle_canvas_surface_edit_started(_edit(target, 0.0))
        self.workspace._handle_canvas_surface_edit_preview_changed(final_edit)
        self.workspace._handle_canvas_surface_edit_finished(final_edit, True)
        self.assertTrue(
            self.workspace._canvas_surface_mesh_update_timer.isActive()
        )
        remaining_before = (
            self.workspace._canvas_surface_mesh_update_timer.remainingTime()
        )

        self.workspace._set_mesh_edit_update_delay_seconds(2.4)

        surface_timer = self.workspace._canvas_surface_mesh_update_timer
        self.assertEqual(surface_timer.interval(), 2400)
        self.assertEqual(
            self.workspace._doorway_mesh_update_timer.interval(),
            2400,
        )
        self.assertTrue(surface_timer.isActive())
        self.assertGreater(surface_timer.remainingTime(), remaining_before)

    def test_room_ceiling_finish_remains_immediate(self) -> None:
        level = _build_square_level(with_room=True)
        surfaces = self._install_level(level)
        ceiling = _surface(
            surfaces,
            surface_type=SURFACE_TYPE_CEILING,
            room_owned=True,
        )
        target = self._target(
            ceiling.surface_id,
            CANVAS_SURFACE_EDIT_ROOM_HEIGHT,
        )
        final_edit = _edit(target, 0.2)
        self.workspace._handle_canvas_surface_edit_started(_edit(target, 0.0))
        self.workspace._handle_canvas_surface_edit_preview_changed(final_edit)

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
            self.workspace._handle_canvas_surface_edit_finished(
                final_edit,
                True,
            )

        reconcile_assignments.assert_called_once_with([level])
        schedule_refresh.assert_called_once_with(preserve_camera=True)
        self.assertFalse(self.workspace._pending_canvas_surface_mesh_update)
        self.assertFalse(
            self.workspace._canvas_surface_mesh_update_timer.isActive()
        )

    def test_room_ceiling_edits_only_its_stable_room_owner(self) -> None:
        level = _build_square_level(with_room=True)
        surfaces = self._install_level(level)
        ceiling = _surface(
            surfaces,
            surface_type=SURFACE_TYPE_CEILING,
            room_owned=True,
        )
        target = self._target(
            ceiling.surface_id,
            CANVAS_SURFACE_EDIT_ROOM_HEIGHT,
        )
        level_height_before = level.height_meters
        room = level.rooms[0]

        self.workspace._handle_canvas_surface_edit_started(
            _edit(target, 0.0)
        )
        self.workspace._handle_canvas_surface_edit_preview_changed(
            _edit(target, 0.6)
        )

        self.assertEqual(
            target.reference.room_center_vertex_id,
            room.center_vertex_id,
        )
        self.assertAlmostEqual(room.height_meters, 4.1)
        self.assertAlmostEqual(level.height_meters, level_height_before)
        self.assertAlmostEqual(
            self.workspace.height_level_spinbox.value(),
            level_height_before,
        )
        self.workspace._handle_canvas_surface_edit_cancelled(
            _edit(target, 0.0)
        )

    def test_inferred_ceiling_edits_level_height(self) -> None:
        level = _build_square_level(with_room=False)
        surfaces = self._install_level(level)
        ceiling = _surface(
            surfaces,
            surface_type=SURFACE_TYPE_CEILING,
            room_owned=False,
        )
        target = self._target(
            ceiling.surface_id,
            CANVAS_SURFACE_EDIT_LEVEL_HEIGHT,
        )

        self.workspace._handle_canvas_surface_edit_started(
            _edit(target, 0.0)
        )
        self.workspace._handle_canvas_surface_edit_preview_changed(
            _edit(target, -0.5)
        )

        self.assertAlmostEqual(level.height_meters, 2.5)
        self.assertAlmostEqual(
            self.workspace.height_level_spinbox.value(),
            2.5,
        )
        self.workspace._handle_canvas_surface_edit_cancelled(
            _edit(target, 0.0)
        )

    def test_singular_wall_restoration_rearms_add_window(self) -> None:
        level = _build_plain_connected_level()
        surfaces = self._install_level(level)
        wall = _surface(
            surfaces,
            surface_type=SURFACE_TYPE_WALL,
            wall_key="1:2",
        )
        viewer = self.workspace.viewer
        self.assertTrue(viewer.select_wall_target(wall.surface_id))
        self.assertEqual(
            self.workspace._desired_canvas_surface_ids,
            (wall.surface_id,),
        )
        assert viewer.add_window_button is not None
        self.assertTrue(viewer.add_window_button.isEnabled())

        self.workspace._is_syncing_canvas_scene_selection = True
        try:
            viewer.select_wall_target(None)
            self.workspace._restore_desired_canvas_scene_selection()
        finally:
            self.workspace._is_syncing_canvas_scene_selection = False

        self.assertEqual(
            viewer.get_selected_canvas_surface_ids(),
            (wall.surface_id,),
        )
        self.assertEqual(
            viewer.get_selected_wall_surface_id(),
            wall.surface_id,
        )
        self.assertTrue(viewer.add_window_button.isEnabled())


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
