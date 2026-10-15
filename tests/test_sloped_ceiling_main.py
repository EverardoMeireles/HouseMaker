# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import trimesh
from PySide6.QtCore import QPointF
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.door_state import (
    DoorLibraryData,
    WindowPlacement,
    create_window_definition_for_frame,
)
from housemaker.glb import (
    GeneratedModel,
    build_placed_generated_model_transform,
)
from housemaker.main import BlueprintWorkspace, _CanvasSlopedCeilingUndoState
from housemaker.models import LevelData, SlopedCeilingData, VertexData
from housemaker.sloped_ceiling import (
    SLOPED_CEILING_HANDLE_CEILING_TRANSITION_RADIUS,
    SLOPED_CEILING_HANDLE_CREATE_DEPTH,
    SLOPED_CEILING_HANDLE_DEPTH,
    SLOPED_CEILING_HANDLE_END,
    SLOPED_CEILING_HANDLE_SIDE_TRANSITION_RADIUS,
    SLOPED_CEILING_HANDLE_START,
    SLOPED_CEILING_HANDLE_WALL_DROP,
    SLOPED_CEILING_HANDLE_WALL_TRANSITION_RADIUS,
    SlopedCeilingEdit,
    SlopedCeilingEditHandle,
    SlopedCeilingEditTarget,
    add_sloped_ceiling_window,
    apply_sloped_ceiling_edit,
    build_sloped_ceiling_window_hosts,
    build_sloped_ceiling_window_placement,
    get_sloped_ceiling_window_world_corners,
)
from housemaker.surface_geometry import build_fixed_surfaces

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_square_level(
    *,
    index: int = 2,
    name: str = "Ground",
) -> LevelData:
    vertex_data = VertexData()
    vertices = tuple(
        vertex_data.add_vertex(*point)
        for point in (
            (0.0, 0.0),
            (200.0, 0.0),
            (200.0, 200.0),
            (0.0, 200.0),
        )
    )
    for first, second in zip(vertices, (*vertices[1:], vertices[0])):
        vertex_data.add_edge(first.id, second.id)
    return LevelData(
        index=index,
        name=name,
        height_meters=3.0,
        vertex_data=vertex_data,
    )


def _build_unequal_reversed_parallel_wall_level() -> LevelData:
    """Build two disconnected parallel walls with opposite local directions."""

    vertex_data = VertexData()
    first_start = vertex_data.add_vertex(0.0, 0.0)
    first_end = vertex_data.add_vertex(200.0, 0.0)
    second_start = vertex_data.add_vertex(300.0, 200.0)
    second_end = vertex_data.add_vertex(0.0, 200.0)
    vertex_data.add_edge(first_start.id, first_end.id)
    vertex_data.add_edge(second_start.id, second_end.id)
    return LevelData(
        index=2,
        name="Ground",
        height_meters=3.0,
        vertex_data=vertex_data,
    )


def _build_unequal_closed_parallel_wall_level() -> LevelData:
    """Build a closed room whose opposite horizontal walls differ in length."""

    vertex_data = VertexData()
    vertices = tuple(
        vertex_data.add_vertex(*point)
        for point in (
            (0.0, 0.0),
            (200.0, 0.0),
            (201.0, 200.0),
            (0.0, 200.0),
        )
    )
    for first, second in zip(vertices, (*vertices[1:], vertices[0])):
        vertex_data.add_edge(first.id, second.id)
    return LevelData(
        index=2,
        name="Ground",
        height_meters=3.0,
        vertex_data=vertex_data,
    )


def _edit(
    target: SlopedCeilingEditTarget,
    handle_kind: str,
    value: float,
) -> SlopedCeilingEdit:
    return SlopedCeilingEdit(
        slope_id=target.slope_id,
        wall_surface_id=target.wall_surface_id,
        level_index=target.level_index,
        handle_kind=handle_kind,
        value=value,
        creation_start_ratio=target.creation_start_ratio,
        creation_end_ratio=target.creation_end_ratio,
    )


# ### Controller integration tests ###
class SlopedCeilingMainTests(unittest.TestCase):
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
            self.workspace.scene_3d_workspace
        )
        self.workspace._set_mesh_edit_update_delay_seconds(60.0)
        self.level = _build_square_level()
        self.workspace.levels = [self.level]
        self.workspace.current_level_index = 0
        self.workspace._reset_viewer_doorway_snapshots()
        self.workspace._sync_level_controls()
        self.workspace._sync_canvas_to_current_level()
        self.surfaces = tuple(build_fixed_surfaces((self.level,)))
        self.workspace._set_canvas_viewer_targets(self.surfaces)
        self.workspace._canvas_viewer_preview_revision = (
            self.workspace._viewer_preview_revision
        )
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self._temporary_directory.cleanup()

    def _creation_target(self) -> SlopedCeilingEditTarget:
        return next(
            target
            for target in (
                self.workspace._sloped_ceiling_edit_targets_by_wall_id.values()
            )
            if target.slope_id is None
        )

    def _create_slope(self, *, commit: bool) -> None:
        target = self._creation_target()
        self.assertTrue(
            self.workspace.viewer.select_canvas_surface_target(target.wall_surface_id)
        )
        start = _edit(target, SLOPED_CEILING_HANDLE_CREATE_DEPTH, 0.0)
        finish = _edit(target, SLOPED_CEILING_HANDLE_CREATE_DEPTH, 0.8)
        self.workspace._handle_sloped_ceiling_edit_started(start)
        self.workspace._handle_sloped_ceiling_edit_preview_changed(finish)
        self.workspace._handle_sloped_ceiling_edit_finished(finish, True)
        if commit:
            self.workspace._commit_pending_sloped_ceiling_mesh_update()

    def _add_centered_slope_window(self) -> None:
        """Add one window with enough clearance for every non-lateral edit."""

        slope = self.level.sloped_ceilings[0]
        host = next(
            candidate
            for candidate in build_sloped_ceiling_window_hosts(
                self.workspace.levels,
                self.surfaces,
            )
            if candidate.slope_id == slope.slope_id
        )
        placement = build_sloped_ceiling_window_placement(
            host,
            host.local_to_world(0.3, 0.3),
            host.local_to_world(0.7, 0.7),
        )
        add_sloped_ceiling_window(
            self.workspace.levels,
            placement,
            window_id="slope-window",
        )

    def _get_slope_window_dimensions(self) -> tuple[float, float]:
        """Return the selected slope window's physical width and height."""

        window = self.level.windows[0]
        host = next(
            candidate
            for candidate in build_sloped_ceiling_window_hosts(
                self.workspace.levels,
                self.surfaces,
            )
            if candidate.slope_id == window.sloped_ceiling_id
        )
        corners = np.asarray(
            get_sloped_ceiling_window_world_corners(host, window),
            dtype=float,
        )
        return (
            float(np.linalg.norm(corners[1] - corners[0])),
            float(np.linalg.norm(corners[3] - corners[0])),
        )

    def test_creation_uses_lightweight_preview_until_delayed_commit(self) -> None:
        target = self._creation_target()
        self.assertTrue(
            self.workspace.viewer.select_canvas_surface_target(target.wall_surface_id)
        )
        start = _edit(target, SLOPED_CEILING_HANDLE_CREATE_DEPTH, 0.0)
        finish = _edit(target, SLOPED_CEILING_HANDLE_CREATE_DEPTH, 0.75)

        with patch.object(
            self.workspace,
            "_schedule_viewer_preview_refresh",
        ) as schedule_refresh:
            self.workspace._handle_sloped_ceiling_edit_started(start)
            self.workspace._handle_sloped_ceiling_edit_preview_changed(finish)

            self.assertEqual(self.level.sloped_ceilings, [])
            self.assertTrue(
                self.workspace.viewer._sloped_ceiling_preview_meshes
            )
            schedule_refresh.assert_not_called()

            self.workspace._handle_sloped_ceiling_edit_finished(finish, True)

            self.assertEqual(len(self.level.sloped_ceilings), 1)
            self.assertAlmostEqual(
                self.level.sloped_ceilings[0].depth_meters,
                0.75,
            )
            self.assertTrue(
                self.workspace._sloped_ceiling_mesh_update_timer.isActive()
            )
            self.assertEqual(
                self.workspace._build_viewer_preview_levels()[0].sloped_ceilings,
                [],
            )
            self.assertEqual(
                self.workspace.viewer.get_selected_sloped_ceiling_wall_surface_id(),
                target.wall_surface_id,
            )
            rebased = self.workspace._sloped_ceiling_edit_targets_by_wall_id[
                target.wall_surface_id
            ]
            self.assertEqual(
                rebased.slope_id,
                self.level.sloped_ceilings[0].slope_id,
            )
            schedule_refresh.assert_not_called()

            self.workspace._commit_pending_sloped_ceiling_mesh_update()

        self.assertFalse(
            self.workspace._sloped_ceiling_mesh_update_timer.isActive()
        )
        self.assertFalse(self.workspace.viewer._sloped_ceiling_preview_meshes)
        self.assertEqual(
            self.workspace._build_viewer_preview_levels()[0].sloped_ceilings,
            self.level.sloped_ceilings,
        )
        self.assertIsInstance(
            self.workspace._canvas_undo_stack[-1],
            _CanvasSlopedCeilingUndoState,
        )
        schedule_refresh.assert_called_once_with(preserve_camera=True)

    def test_tuple_batch_previews_and_commits_two_walls_as_one_action(self) -> None:
        targets = tuple(
            list(self.workspace._sloped_ceiling_edit_targets_by_key.values())[:2]
        )
        self.assertEqual(len(targets), 2)
        starts = tuple(
            _edit(target, SLOPED_CEILING_HANDLE_CREATE_DEPTH, 0.0)
            for target in targets
        )
        finishes = tuple(
            _edit(target, SLOPED_CEILING_HANDLE_CREATE_DEPTH, 0.72)
            for target in targets
        )

        self.workspace._handle_sloped_ceiling_edit_started(starts)
        self.workspace._handle_sloped_ceiling_edit_preview_changed(finishes)

        self.assertEqual(self.level.sloped_ceilings, [])
        self.assertEqual(
            len(self.workspace.viewer._sloped_ceiling_preview_meshes),
            2,
        )

        self.workspace._handle_sloped_ceiling_edit_finished(finishes, True)

        self.assertEqual(len(self.level.sloped_ceilings), 2)
        self.assertTrue(
            all(
                slope.joined_group_id is None
                for slope in self.level.sloped_ceilings
            )
        )
        self.assertIsNotNone(self.workspace._pending_sloped_ceiling_undo_state)
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self.workspace._commit_pending_sloped_ceiling_mesh_update()
        self.assertEqual(len(self.workspace._canvas_undo_stack), 1)
        self.assertIsInstance(
            self.workspace._canvas_undo_stack[-1],
            _CanvasSlopedCeilingUndoState,
        )

        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self.workspace._handle_canvas_undo_requested()
        self.assertEqual(self.level.sloped_ceilings, [])

    def test_unequal_parallel_slopes_stay_joined_after_lateral_release(self) -> None:
        """A target rebuild must retain one control set for unequal wall lengths."""

        self.level = _build_unequal_reversed_parallel_wall_level()
        self.workspace.levels = [self.level]
        self.workspace.current_level_index = 0
        self.workspace._reset_viewer_doorway_snapshots()
        self.workspace._sync_level_controls()
        self.workspace._sync_canvas_to_current_level()
        self.surfaces = tuple(build_fixed_surfaces((self.level,)))
        self.workspace._set_canvas_viewer_targets(self.surfaces)
        self.workspace._canvas_viewer_preview_revision = (
            self.workspace._viewer_preview_revision
        )

        creation_targets = tuple(
            target
            for target in self.workspace._sloped_ceiling_edit_targets_by_key.values()
            if target.slope_id is None
        )
        self.assertEqual(len(creation_targets), 2)
        selected_wall_ids = tuple(
            target.wall_surface_id for target in creation_targets
        )
        self.workspace.viewer.set_selected_canvas_surface_ids(selected_wall_ids)
        active_wall_id = (
            self.workspace.viewer.get_selected_sloped_ceiling_wall_surface_id()
        )
        creation_target = next(
            target
            for target in creation_targets
            if target.wall_surface_id == active_wall_id
        )
        creation_handle = creation_target.handles[0]

        def camera_ray(
            handle: SlopedCeilingEditHandle,
            displacement_meters: float,
        ) -> tuple[np.ndarray, np.ndarray]:
            axis = np.asarray(handle.axis_world, dtype=float)
            axis /= np.linalg.norm(axis)
            camera_direction = np.eye(3, dtype=float)[
                int(np.argmin(np.abs(axis)))
            ]
            hit = (
                np.asarray(handle.origin_world, dtype=float)
                + axis * displacement_meters
            )
            return hit - camera_direction, camera_direction

        with patch.object(
            self.workspace.viewer.view,
            "build_camera_ray",
            return_value=camera_ray(creation_handle, 0.0),
        ):
            self.assertTrue(
                self.workspace.viewer._begin_sloped_ceiling_edit_drag(
                    creation_target,
                    creation_handle,
                    QPointF(),
                )
            )
        with patch.object(
            self.workspace.viewer.view,
            "build_camera_ray",
            return_value=camera_ray(creation_handle, 0.8),
        ):
            self.assertTrue(
                self.workspace.viewer._finish_sloped_ceiling_edit_drag(QPointF())
            )

        self.assertEqual(len(self.level.sloped_ceilings), 2)
        self.assertEqual(
            len(
                {
                    slope.joined_group_id
                    for slope in self.level.sloped_ceilings
                    if slope.joined_group_id is not None
                }
            ),
            1,
        )
        initial_target_lengths = {
            target.wall_surface_id: float(
                np.linalg.norm(
                    np.asarray(target.wall_end_world, dtype=float)
                    - np.asarray(target.wall_start_world, dtype=float)
                )
            )
            for target in self.workspace._sloped_ceiling_edit_targets_by_key.values()
        }
        np.testing.assert_allclose(
            tuple(
                (slope.end_ratio - slope.start_ratio)
                * initial_target_lengths[slope.wall_surface_id]
                for slope in self.level.sloped_ceilings
            ),
            (4.0, 4.0),
            atol=1e-9,
        )
        rebuilt_target = next(
            target
            for target in self.workspace._sloped_ceiling_edit_targets_by_key.values()
            if target.wall_surface_id == active_wall_id
            and target.slope_id is not None
        )
        lateral_handle = next(
            handle
            for handle in rebuilt_target.handles
            if handle.handle_kind == SLOPED_CEILING_HANDLE_START
        )
        next_value = lateral_handle.value + 0.05
        displacement_meters = (
            next_value - lateral_handle.value
        ) / lateral_handle.value_per_meter
        self.assertEqual(
            len(
                self.workspace.viewer._get_compatible_sloped_ceiling_target_handles(
                    rebuilt_target,
                    lateral_handle,
                )
            ),
            2,
        )

        with patch.object(
            self.workspace.viewer.view,
            "build_camera_ray",
            return_value=camera_ray(lateral_handle, 0.0),
        ):
            self.assertTrue(
                self.workspace.viewer._begin_sloped_ceiling_edit_drag(
                    rebuilt_target,
                    lateral_handle,
                    QPointF(),
                )
            )
        with patch.object(
            self.workspace.viewer.view,
            "build_camera_ray",
            return_value=camera_ray(lateral_handle, displacement_meters),
        ):
            self.assertTrue(
                self.workspace.viewer._finish_sloped_ceiling_edit_drag(QPointF())
            )

        self.assertEqual(
            self.workspace.viewer.get_selected_sloped_ceiling_wall_surface_ids(),
            selected_wall_ids,
        )
        displayed_targets = (
            self.workspace.viewer._get_displayed_sloped_ceiling_targets()
        )
        self.assertEqual(
            sum(target.slope_id is not None for target in displayed_targets),
            1,
        )
        self.assertEqual(
            sum(target.slope_id is None for target in displayed_targets),
            1,
        )
        world_intervals = []
        targets_by_wall = {
            target.wall_surface_id: target
            for target in self.workspace._sloped_ceiling_edit_targets_by_key.values()
            if target.slope_id is not None
        }
        for slope in self.level.sloped_ceilings:
            target = targets_by_wall[slope.wall_surface_id]
            wall_start = np.asarray(target.wall_start_world, dtype=float)
            wall_end = np.asarray(target.wall_end_world, dtype=float)
            tangent = wall_end - wall_start
            world_intervals.append(
                sorted(
                    (
                        float((wall_start + tangent * slope.start_ratio)[0]),
                        float((wall_start + tangent * slope.end_ratio)[0]),
                    )
                )
            )
        for interval in world_intervals:
            np.testing.assert_allclose(interval, (0.0, 3.7), atol=1e-9)

    def test_joined_unequal_parallel_slope_with_window_accepts_side_edits(
        self,
    ) -> None:
        """A slope window must not strand joined lateral or rounding drags."""

        self.level = _build_unequal_closed_parallel_wall_level()
        self.workspace.levels = [self.level]
        self.workspace.current_level_index = 0
        self.workspace._reset_viewer_doorway_snapshots()
        self.workspace._sync_level_controls()
        self.workspace._sync_canvas_to_current_level()
        self.surfaces = tuple(build_fixed_surfaces((self.level,)))
        self.workspace._set_canvas_viewer_targets(self.surfaces)

        creation_targets = tuple(
            target
            for target in self.workspace._sloped_ceiling_edit_targets_by_key.values()
            if (
                target.slope_id is None
                and abs(
                    target.wall_end_world[1] - target.wall_start_world[1]
                )
                < 1e-9
            )
        )
        self.assertEqual(len(creation_targets), 2)
        wall_ids = tuple(target.wall_surface_id for target in creation_targets)
        self.workspace.viewer.set_selected_canvas_surface_ids(wall_ids)
        active_wall_id = (
            self.workspace.viewer.get_selected_sloped_ceiling_wall_surface_id()
        )
        target = next(
            item
            for item in creation_targets
            if item.wall_surface_id == active_wall_id
        )
        handle = target.handles[0]

        def drag(
            drag_target: SlopedCeilingEditTarget,
            drag_handle: SlopedCeilingEditHandle,
            displacement_meters: float,
            *,
            assert_refresh_deferred: bool = False,
        ) -> None:
            axis = np.asarray(drag_handle.axis_world, dtype=float)
            axis /= np.linalg.norm(axis)
            camera_direction = np.eye(3, dtype=float)[
                int(np.argmin(np.abs(axis)))
            ]

            def ray(offset: float) -> tuple[np.ndarray, np.ndarray]:
                hit = (
                    np.asarray(drag_handle.origin_world, dtype=float)
                    + axis * offset
                )
                return hit - camera_direction, camera_direction

            with patch.object(
                self.workspace.viewer.view,
                "build_camera_ray",
                return_value=ray(0.0),
            ):
                self.assertTrue(
                    self.workspace.viewer._begin_sloped_ceiling_edit_drag(
                        drag_target,
                        drag_handle,
                        QPointF(),
                    )
                )
            if assert_refresh_deferred:
                self.workspace._mark_viewer_preview_dirty()
                with patch.object(
                    self.workspace,
                    "_refresh_viewer_preview",
                ) as refresh:
                    self.workspace._run_scheduled_viewer_preview_refresh()
                refresh.assert_not_called()
                self.assertIsNotNone(
                    self.workspace.viewer._sloped_ceiling_edit_drag
                )
                self.assertTrue(self.workspace._active_sloped_ceiling_edits)
            with patch.object(
                self.workspace.viewer.view,
                "build_camera_ray",
                return_value=ray(displacement_meters),
            ):
                self.assertTrue(
                    self.workspace.viewer._finish_sloped_ceiling_edit_drag(
                        QPointF()
                    )
                )

        drag(target, handle, 2.0)
        self.workspace._commit_pending_sloped_ceiling_mesh_update()
        self.assertEqual(len(self.level.sloped_ceilings), 2)

        hosts = build_sloped_ceiling_window_hosts(
            self.workspace.levels,
            self.surfaces,
        )
        self.assertEqual(len(hosts), 2)
        window_id_by_slope_id: dict[str, str] = {}
        for host in hosts:
            placement = build_sloped_ceiling_window_placement(
                host,
                host.local_to_world(0.05, 0.3),
                host.local_to_world(0.25, 0.7),
            )
            self.workspace._handle_canvas_window_placement_requested(placement)
            window_id_by_slope_id[host.slope_id] = self.level.windows[-1].window_id
        self.assertEqual(len(self.level.windows), 2)
        active_slope_id = next(
            slope.slope_id
            for slope in self.level.sloped_ceilings
            if slope.wall_surface_id == active_wall_id
        )
        self.surfaces = tuple(self.workspace._canvas_surface_targets_by_id.values())
        self.workspace._set_mesh_edit_update_delay_seconds(0.02)

        for handle_kind, value_delta in (
            (SLOPED_CEILING_HANDLE_SIDE_TRANSITION_RADIUS, 0.2),
            (SLOPED_CEILING_HANDLE_START, 0.35),
        ):
            slope_target = next(
                item
                for item in self.workspace._sloped_ceiling_edit_targets_by_key.values()
                if item.wall_surface_id == active_wall_id
                and item.slope_id is not None
            )
            edit_handle = next(
                item
                for item in slope_target.handles
                if item.handle_kind == handle_kind
            )
            self.assertEqual(
                len(
                    self.workspace.viewer._get_compatible_sloped_ceiling_target_handles(
                        slope_target,
                        edit_handle,
                    )
                ),
                2,
            )
            previous_records = tuple(self.level.sloped_ceilings)
            displacement = value_delta / edit_handle.value_per_meter
            drag(
                slope_target,
                edit_handle,
                displacement,
                assert_refresh_deferred=True,
            )
            self.assertNotEqual(tuple(self.level.sloped_ceilings), previous_records)
            self.assertTrue(
                self.workspace._sloped_ceiling_mesh_update_timer.isActive()
            )
            expected_window_ids = (
                set(window_id_by_slope_id.values())
                if handle_kind
                == SLOPED_CEILING_HANDLE_SIDE_TRANSITION_RADIUS
                else {
                    window_id
                    for slope_id, window_id in window_id_by_slope_id.items()
                    if slope_id != active_slope_id
                }
            )
            self.assertEqual(
                {window.window_id for window in self.level.windows},
                expected_window_ids,
            )
            self.assertNotIn(
                "stopped",
                self.workspace.viewer.surface_tools_status_label.text().lower(),
            )
            self.workspace._mark_viewer_preview_dirty()
            with patch.object(
                self.workspace,
                "_refresh_viewer_preview",
            ) as refresh:
                self.workspace._run_scheduled_viewer_preview_refresh()
            refresh.assert_not_called()
            self.assertIsNotNone(
                self.workspace._pending_sloped_ceiling_undo_state
            )
            self.assertTrue(
                self.workspace._sloped_ceiling_mesh_update_timer.isActive()
            )
            QTest.qWait(100)
            self.assertFalse(
                self.workspace._sloped_ceiling_mesh_update_timer.isActive()
            )
            self.assertFalse(self.workspace.viewer._sloped_ceiling_preview_meshes)

    def test_joined_slope_window_accepts_every_shared_profile_handle(
        self,
    ) -> None:
        """A real slope-window rebuild must not strand any shared gizmo."""

        edit_cases = (
            (SLOPED_CEILING_HANDLE_DEPTH, 0.12),
            (SLOPED_CEILING_HANDLE_WALL_DROP, 0.12),
            (SLOPED_CEILING_HANDLE_START, 0.04),
            (SLOPED_CEILING_HANDLE_END, -0.04),
            (SLOPED_CEILING_HANDLE_SIDE_TRANSITION_RADIUS, 0.03),
        )
        field_by_handle = {
            SLOPED_CEILING_HANDLE_DEPTH: "depth_meters",
            SLOPED_CEILING_HANDLE_WALL_DROP: "wall_drop_meters",
            SLOPED_CEILING_HANDLE_START: "start_ratio",
            SLOPED_CEILING_HANDLE_END: "end_ratio",
            SLOPED_CEILING_HANDLE_SIDE_TRANSITION_RADIUS: (
                "side_transition_radius_meters"
            ),
        }

        def camera_ray(
            handle: SlopedCeilingEditHandle,
            displacement_meters: float,
        ) -> tuple[np.ndarray, np.ndarray]:
            axis = np.asarray(handle.axis_world, dtype=float)
            axis /= np.linalg.norm(axis)
            camera_direction = np.eye(3, dtype=float)[
                int(np.argmin(np.abs(axis)))
            ]
            hit = (
                np.asarray(handle.origin_world, dtype=float)
                + axis * displacement_meters
            )
            return hit - camera_direction, camera_direction

        for handle_kind, value_delta in edit_cases:
            with self.subTest(handle_kind=handle_kind):
                self.workspace._commit_pending_sloped_ceiling_mesh_update()
                self.level = _build_unequal_closed_parallel_wall_level()
                self.workspace.levels = [self.level]
                self.workspace.current_level_index = 0
                self.workspace._canvas_undo_stack.clear()
                self.workspace._reset_viewer_doorway_snapshots()
                self.workspace._sync_level_controls()
                self.workspace._sync_canvas_to_current_level()
                self.surfaces = tuple(build_fixed_surfaces((self.level,)))
                self.workspace._set_canvas_viewer_targets(self.surfaces)

                creation_targets = tuple(
                    target
                    for target in (
                        self.workspace._sloped_ceiling_edit_targets_by_key.values()
                    )
                    if (
                        target.slope_id is None
                        and abs(
                            target.wall_end_world[1]
                            - target.wall_start_world[1]
                        )
                        < 1e-9
                    )
                )
                self.assertEqual(len(creation_targets), 2)
                wall_ids = tuple(
                    target.wall_surface_id for target in creation_targets
                )
                self.workspace.viewer.set_selected_canvas_surface_ids(wall_ids)
                active_wall_id = (
                    self.workspace.viewer.get_selected_sloped_ceiling_wall_surface_id()
                )
                creation_target = next(
                    target
                    for target in creation_targets
                    if target.wall_surface_id == active_wall_id
                )
                creation_handle = creation_target.handles[0]
                with patch.object(
                    self.workspace.viewer.view,
                    "build_camera_ray",
                    return_value=camera_ray(creation_handle, 0.0),
                ):
                    self.assertTrue(
                        self.workspace.viewer._begin_sloped_ceiling_edit_drag(
                            creation_target,
                            creation_handle,
                            QPointF(),
                        )
                    )
                with patch.object(
                    self.workspace.viewer.view,
                    "build_camera_ray",
                    return_value=camera_ray(creation_handle, 2.0),
                ):
                    self.assertTrue(
                        self.workspace.viewer._finish_sloped_ceiling_edit_drag(
                            QPointF()
                        )
                    )
                self.workspace._commit_pending_sloped_ceiling_mesh_update()
                self.assertEqual(len(self.level.sloped_ceilings), 2)
                self.assertEqual(
                    len(
                        {
                            slope.joined_group_id
                            for slope in self.level.sloped_ceilings
                            if slope.joined_group_id is not None
                        }
                    ),
                    1,
                )

                active_slope = next(
                    slope
                    for slope in self.level.sloped_ceilings
                    if slope.wall_surface_id == active_wall_id
                )
                host = next(
                    candidate
                    for candidate in build_sloped_ceiling_window_hosts(
                        self.workspace.levels,
                        self.surfaces,
                    )
                    if candidate.slope_id == active_slope.slope_id
                )
                placement = build_sloped_ceiling_window_placement(
                    host,
                    host.local_to_world(0.35, 0.3),
                    host.local_to_world(0.65, 0.7),
                )
                self.workspace._handle_canvas_window_placement_requested(
                    placement
                )
                self.assertEqual(len(self.level.windows), 1)
                window_id = self.level.windows[0].window_id
                self.workspace._set_mesh_edit_update_delay_seconds(0.02)

                target = next(
                    candidate
                    for candidate in (
                        self.workspace._sloped_ceiling_edit_targets_by_key.values()
                    )
                    if (
                        candidate.wall_surface_id == active_wall_id
                        and candidate.slope_id == active_slope.slope_id
                    )
                )
                handle_by_kind = {
                    candidate.handle_kind: candidate
                    for candidate in target.handles
                }
                self.assertIn(
                    handle_kind,
                    handle_by_kind,
                    (
                        "The window-frame rebuild removed the shared slope "
                        f"handle; remaining handles: {tuple(handle_by_kind)}"
                    ),
                )
                handle = handle_by_kind[handle_kind]
                compatible = (
                    self.workspace.viewer._get_compatible_sloped_ceiling_target_handles(
                        target,
                        handle,
                    )
                )
                self.assertEqual(len(compatible), 2)
                next_value = float(
                    np.clip(
                        handle.value + value_delta,
                        handle.minimum_value,
                        handle.maximum_value,
                    )
                )
                self.assertNotAlmostEqual(next_value, handle.value)
                previous_slopes = tuple(self.level.sloped_ceilings)
                displacement = (
                    next_value - handle.value
                ) / handle.value_per_meter

                with patch.object(
                    self.workspace.viewer.view,
                    "build_camera_ray",
                    return_value=camera_ray(handle, 0.0),
                ):
                    self.assertTrue(
                        self.workspace.viewer._begin_sloped_ceiling_edit_drag(
                            target,
                            handle,
                            QPointF(),
                        )
                    )
                with patch.object(
                    self.workspace.viewer.view,
                    "build_camera_ray",
                    return_value=camera_ray(handle, displacement),
                ):
                    self.assertTrue(
                        self.workspace.viewer._update_sloped_ceiling_edit_drag(
                            QPointF()
                        )
                    )
                    self.assertTrue(
                        self.workspace.viewer._finish_sloped_ceiling_edit_drag(
                            QPointF()
                        )
                    )

                self.assertEqual(
                    tuple(window.window_id for window in self.level.windows),
                    (window_id,),
                )
                self.assertNotEqual(
                    tuple(self.level.sloped_ceilings),
                    previous_slopes,
                )
                edited_slope = next(
                    slope
                    for slope in self.level.sloped_ceilings
                    if slope.slope_id == active_slope.slope_id
                )
                self.assertAlmostEqual(
                    getattr(edited_slope, field_by_handle[handle_kind]),
                    next_value,
                )
                self.assertTrue(
                    self.workspace._sloped_ceiling_mesh_update_timer.isActive()
                )
                self.assertTrue(
                    self.workspace.viewer._sloped_ceiling_preview_meshes
                )
                status = (
                    self.workspace.viewer.surface_tools_status_label.text().lower()
                )
                self.assertNotIn("stopped", status)
                self.assertNotIn("failed", status)

                QTest.qWait(100)

                self.assertFalse(
                    self.workspace._sloped_ceiling_mesh_update_timer.isActive()
                )
                self.assertIsNone(
                    self.workspace._pending_sloped_ceiling_undo_state
                )
                self.assertFalse(
                    self.workspace.viewer._sloped_ceiling_preview_meshes
                )
                self.assertEqual(
                    tuple(window.window_id for window in self.level.windows),
                    (window_id,),
                )
                rebuilt_target = next(
                    candidate
                    for candidate in (
                        self.workspace._sloped_ceiling_edit_targets_by_key.values()
                    )
                    if (
                        candidate.wall_surface_id == active_wall_id
                        and candidate.slope_id == active_slope.slope_id
                    )
                )
                rebuilt_handle = next(
                    candidate
                    for candidate in rebuilt_target.handles
                    if candidate.handle_kind == handle_kind
                )
                self.assertAlmostEqual(rebuilt_handle.value, next_value)

    def test_new_drag_survives_committing_a_different_pending_slope(self) -> None:
        """A target refresh must not cancel the drag that triggered it."""

        self._create_slope(commit=False)
        first_wall_surface_id = self.level.sloped_ceilings[0].wall_surface_id
        target = next(
            candidate
            for candidate in self.workspace._sloped_ceiling_edit_targets_by_key.values()
            if candidate.slope_id is None
            and candidate.wall_surface_id != first_wall_surface_id
        )
        handle = next(
            candidate
            for candidate in target.handles
            if candidate.handle_kind == SLOPED_CEILING_HANDLE_CREATE_DEPTH
        )
        self.workspace.viewer.set_selected_canvas_surface_ids(
            (target.wall_surface_id,)
        )
        axis = np.asarray(handle.axis_world, dtype=float)
        axis /= np.linalg.norm(axis)
        camera_direction = np.eye(3, dtype=float)[int(np.argmin(np.abs(axis)))]

        def camera_ray(offset_meters: float) -> tuple[np.ndarray, np.ndarray]:
            hit = np.asarray(handle.origin_world, dtype=float) + axis * offset_meters
            return hit - camera_direction, camera_direction

        with patch.object(
            self.workspace.viewer.view,
            "build_camera_ray",
            return_value=camera_ray(0.0),
        ):
            self.assertTrue(
                self.workspace.viewer._begin_sloped_ceiling_edit_drag(
                    target,
                    handle,
                    QPointF(),
                )
            )

        self.assertIsNotNone(self.workspace.viewer._sloped_ceiling_edit_drag)
        self.assertTrue(self.workspace._active_sloped_ceiling_edits)
        self.assertIsNone(self.workspace._pending_sloped_ceiling_undo_state)

        with patch.object(
            self.workspace.viewer.view,
            "build_camera_ray",
            return_value=camera_ray(0.65),
        ):
            self.assertTrue(
                self.workspace.viewer._update_sloped_ceiling_edit_drag(QPointF())
            )
            self.assertTrue(
                self.workspace.viewer._finish_sloped_ceiling_edit_drag(QPointF())
            )

        self.assertFalse(self.workspace._active_sloped_ceiling_edits)
        self.assertEqual(len(self.level.sloped_ceilings), 2)

    def test_second_double_slope_drag_survives_pending_first_pair(self) -> None:
        creation_targets = tuple(
            target
            for target in self.workspace._sloped_ceiling_edit_targets_by_key.values()
            if target.slope_id is None
        )
        targets_by_axis: dict[int, list[SlopedCeilingEditTarget]] = {}
        for target in creation_targets:
            direction = np.abs(
                np.asarray(target.wall_end_world, dtype=float)
                - np.asarray(target.wall_start_world, dtype=float)
            )
            targets_by_axis.setdefault(int(np.argmax(direction[:2])), []).append(
                target
            )
        target_pairs = tuple(tuple(targets) for targets in targets_by_axis.values())
        self.assertEqual(tuple(map(len, target_pairs)), (2, 2))
        first_pair, second_pair = target_pairs
        first_starts = tuple(
            _edit(target, SLOPED_CEILING_HANDLE_CREATE_DEPTH, 0.0)
            for target in first_pair
        )
        first_finishes = tuple(
            _edit(target, SLOPED_CEILING_HANDLE_CREATE_DEPTH, 0.6)
            for target in first_pair
        )
        self.workspace._handle_sloped_ceiling_edit_started(first_starts)
        self.workspace._handle_sloped_ceiling_edit_finished(first_finishes, True)
        self.assertEqual(len(self.level.sloped_ceilings), 2)

        second_wall_ids = tuple(target.wall_surface_id for target in second_pair)
        refreshed_second_targets = tuple(
            target
            for target in self.workspace._sloped_ceiling_edit_targets_by_key.values()
            if target.slope_id is None
            and target.wall_surface_id in second_wall_ids
        )
        self.assertEqual(len(refreshed_second_targets), 2)
        target = refreshed_second_targets[-1]
        handle = next(
            candidate
            for candidate in target.handles
            if candidate.handle_kind == SLOPED_CEILING_HANDLE_CREATE_DEPTH
        )
        self.workspace.viewer.set_selected_canvas_surface_ids(second_wall_ids)
        axis = np.asarray(handle.axis_world, dtype=float)
        axis /= np.linalg.norm(axis)
        camera_direction = np.eye(3, dtype=float)[int(np.argmin(np.abs(axis)))]

        def camera_ray(offset_meters: float) -> tuple[np.ndarray, np.ndarray]:
            hit = np.asarray(handle.origin_world, dtype=float) + axis * offset_meters
            return hit - camera_direction, camera_direction

        with patch.object(
            self.workspace.viewer.view,
            "build_camera_ray",
            return_value=camera_ray(0.0),
        ):
            self.assertTrue(
                self.workspace.viewer._begin_sloped_ceiling_edit_drag(
                    target,
                    handle,
                    QPointF(),
                )
            )

        drag = self.workspace.viewer._sloped_ceiling_edit_drag
        self.assertIsNotNone(drag)
        assert drag is not None
        self.assertEqual(len(drag.affected_target_handles), 2)
        with patch.object(
            self.workspace.viewer.view,
            "build_camera_ray",
            return_value=camera_ray(0.55),
        ):
            self.assertTrue(
                self.workspace.viewer._finish_sloped_ceiling_edit_drag(QPointF())
            )

        self.assertFalse(self.workspace._active_sloped_ceiling_edits)
        self.assertEqual(len(self.level.sloped_ceilings), 4)

    def test_second_double_slope_stays_joined_after_mesh_rebuild(self) -> None:
        creation_targets = tuple(
            target
            for target in self.workspace._sloped_ceiling_edit_targets_by_key.values()
            if target.slope_id is None
        )
        horizontal_targets = tuple(
            target
            for target in creation_targets
            if abs(target.wall_end_world[0] - target.wall_start_world[0]) > 1.0
        )
        self.assertEqual(len(horizontal_targets), 2)
        wall_ids = tuple(target.wall_surface_id for target in horizontal_targets)
        self.workspace.viewer.set_selected_canvas_surface_ids(wall_ids)

        def drag_handle(
            target: SlopedCeilingEditTarget,
            handle: SlopedCeilingEditHandle,
            displacement_meters: float,
        ) -> None:
            axis = np.asarray(handle.axis_world, dtype=float)
            axis /= np.linalg.norm(axis)
            camera_direction = np.eye(3, dtype=float)[
                int(np.argmin(np.abs(axis)))
            ]

            def camera_ray(offset_meters: float) -> tuple[np.ndarray, np.ndarray]:
                hit = (
                    np.asarray(handle.origin_world, dtype=float)
                    + axis * offset_meters
                )
                return hit - camera_direction, camera_direction

            with patch.object(
                self.workspace.viewer.view,
                "build_camera_ray",
                return_value=camera_ray(0.0),
            ):
                self.assertTrue(
                    self.workspace.viewer._begin_sloped_ceiling_edit_drag(
                        target,
                        handle,
                        QPointF(),
                    )
                )
            with patch.object(
                self.workspace.viewer.view,
                "build_camera_ray",
                return_value=camera_ray(displacement_meters),
            ):
                self.assertTrue(
                    self.workspace.viewer._finish_sloped_ceiling_edit_drag(
                        QPointF()
                    )
                )

        first_target = next(
            target
            for target in horizontal_targets
            if target.wall_surface_id
            == self.workspace.viewer.get_selected_sloped_ceiling_wall_surface_id()
        )
        first_handle = first_target.handles[0]
        self.assertEqual(
            len(
                self.workspace.viewer._get_compatible_sloped_ceiling_target_handles(
                    first_target,
                    first_handle,
                )
            ),
            2,
        )
        drag_handle(first_target, first_handle, 0.8)
        self.workspace._commit_pending_sloped_ceiling_mesh_update()

        first_slope_target = next(
            target
            for target in self.workspace._sloped_ceiling_edit_targets_by_key.values()
            if target.wall_surface_id == first_target.wall_surface_id
            and target.slope_id is not None
        )
        lateral_handle = next(
            handle
            for handle in first_slope_target.handles
            if handle.handle_kind == SLOPED_CEILING_HANDLE_START
        )
        next_value = 0.25
        lateral_displacement = (
            next_value - lateral_handle.value
        ) / lateral_handle.value_per_meter
        drag_handle(first_slope_target, lateral_handle, lateral_displacement)
        self.workspace._commit_pending_sloped_ceiling_mesh_update()

        second_target = next(
            target
            for target in self.workspace._sloped_ceiling_edit_targets_by_key.values()
            if target.wall_surface_id == first_target.wall_surface_id
            and target.slope_id is None
        )
        second_handle = second_target.handles[0]
        self.assertEqual(
            len(
                self.workspace.viewer._get_compatible_sloped_ceiling_target_handles(
                    second_target,
                    second_handle,
                )
            ),
            2,
        )
        drag_handle(second_target, second_handle, 0.55)
        self.workspace._commit_pending_sloped_ceiling_mesh_update()

        self.workspace.viewer.set_selected_canvas_surface_ids(wall_ids)
        displayed_slopes = tuple(
            target
            for target in (
                self.workspace.viewer._get_displayed_sloped_ceiling_targets()
            )
            if target.slope_id is not None
        )
        self.assertEqual(len(self.level.sloped_ceilings), 4)
        self.assertEqual(len(displayed_slopes), 2)
        for target in displayed_slopes:
            depth_handle = next(
                handle
                for handle in target.handles
                if handle.handle_kind == SLOPED_CEILING_HANDLE_DEPTH
            )
            with self.subTest(slope_id=target.slope_id):
                compatible_handles = (
                    self.workspace.viewer._get_compatible_sloped_ceiling_target_handles(
                        target,
                        depth_handle,
                    )
                )
                self.assertEqual(
                    len(compatible_handles),
                    2,
                )

    def test_tuple_batch_rolls_back_when_one_wall_cannot_be_applied(self) -> None:
        targets = tuple(
            list(self.workspace._sloped_ceiling_edit_targets_by_key.values())[:2]
        )
        starts = tuple(
            _edit(target, SLOPED_CEILING_HANDLE_CREATE_DEPTH, 0.0)
            for target in targets
        )
        finishes = tuple(
            _edit(target, SLOPED_CEILING_HANDLE_CREATE_DEPTH, 0.72)
            for target in targets
        )
        apply_count = 0

        def fail_second_apply(
            levels: list[LevelData],
            edit: SlopedCeilingEdit,
        ) -> SlopedCeilingData:
            nonlocal apply_count
            apply_count += 1
            if apply_count == 2:
                raise ValueError("second wall failed")
            return apply_sloped_ceiling_edit(levels, edit)

        self.workspace._handle_sloped_ceiling_edit_started(starts)
        with patch(
            "housemaker.main.apply_sloped_ceiling_edit",
            side_effect=fail_second_apply,
        ):
            self.workspace._handle_sloped_ceiling_edit_finished(finishes, True)

        self.assertEqual(self.level.sloped_ceilings, [])
        self.assertIsNone(self.workspace._pending_sloped_ceiling_undo_state)
        self.assertFalse(self.workspace._sloped_ceiling_mesh_update_timer.isActive())

    def test_target_map_keeps_multiple_slopes_and_gaps_on_the_same_wall(self) -> None:
        creation_target = self._creation_target()
        self.level.sloped_ceilings = [
            SlopedCeilingData(
                slope_id="a" * 32,
                wall_surface_id=creation_target.wall_surface_id,
                start_ratio=0.25,
                end_ratio=0.75,
            )
        ]

        self.workspace._refresh_sloped_ceiling_edit_targets(
            preserve_selection=True
        )

        same_wall_targets = tuple(
            target
            for (wall_surface_id, _target_id), target in (
                self.workspace._sloped_ceiling_edit_targets_by_key.items()
            )
            if wall_surface_id == creation_target.wall_surface_id
        )
        self.assertEqual(len(same_wall_targets), 3)
        self.assertEqual(
            len(
                {
                    target.target_id
                    for target in same_wall_targets
                }
            ),
            3,
        )
        self.assertEqual(
            len(
                [
                    target
                    for (wall_surface_id, _target_id), target in (
                        self.workspace.viewer._sloped_ceiling_edit_targets.items()
                    )
                    if wall_surface_id == creation_target.wall_surface_id
                ]
            ),
            3,
        )
        self.assertEqual(
            self.workspace._sloped_ceiling_edit_targets_by_wall_id[
                creation_target.wall_surface_id
            ].slope_id,
            "a" * 32,
        )

    def test_tuple_batch_tracks_and_undoes_every_affected_level(self) -> None:
        upper_level = _build_square_level(index=7, name="Upper")
        self.workspace.levels = [self.level, upper_level]
        self.workspace._reset_viewer_doorway_snapshots()
        surfaces = tuple(build_fixed_surfaces(self.workspace.levels))
        self.workspace._set_canvas_viewer_targets(surfaces)
        targets_by_level = {
            target.level_index: target
            for target in self.workspace._sloped_ceiling_edit_targets_by_key.values()
            if target.slope_id is None
        }
        targets = (targets_by_level[self.level.index], targets_by_level[7])
        starts = tuple(
            _edit(target, SLOPED_CEILING_HANDLE_CREATE_DEPTH, 0.0)
            for target in targets
        )
        finishes = tuple(
            _edit(target, SLOPED_CEILING_HANDLE_CREATE_DEPTH, 0.68)
            for target in targets
        )

        self.workspace._handle_sloped_ceiling_edit_started(starts)
        self.workspace._handle_sloped_ceiling_edit_finished(finishes, True)

        pending_state = self.workspace._pending_sloped_ceiling_undo_state
        self.assertIsNotNone(pending_state)
        assert pending_state is not None
        self.assertEqual(
            {
                level_index
                for level_index, _slopes in pending_state.sloped_ceilings_by_level
            },
            {self.level.index, upper_level.index},
        )
        self.assertEqual(len(self.level.sloped_ceilings), 1)
        self.assertEqual(len(upper_level.sloped_ceilings), 1)

        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self.workspace._commit_pending_sloped_ceiling_mesh_update()
            self.workspace._handle_canvas_undo_requested()

        self.assertEqual(self.level.sloped_ceilings, [])
        self.assertEqual(upper_level.sloped_ceilings, [])

    def test_ctrl_z_restores_released_edit_before_delayed_commit(self) -> None:
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self._create_slope(commit=True)
        self.workspace._canvas_undo_stack.clear()
        original = self.level.sloped_ceilings[0]
        target = self.workspace._sloped_ceiling_edit_targets_by_wall_id[
            original.wall_surface_id
        ]
        start = _edit(
            target,
            SLOPED_CEILING_HANDLE_WALL_DROP,
            original.wall_drop_meters,
        )
        finish = _edit(target, SLOPED_CEILING_HANDLE_WALL_DROP, 0.42)

        self.workspace._handle_sloped_ceiling_edit_started(start)
        self.workspace._handle_sloped_ceiling_edit_preview_changed(finish)
        self.assertEqual(self.level.sloped_ceilings[0], original)
        self.workspace._handle_sloped_ceiling_edit_finished(finish, True)
        self.assertAlmostEqual(
            self.level.sloped_ceilings[0].wall_drop_meters,
            0.42,
        )

        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self.workspace._handle_canvas_undo_requested()

        self.assertEqual(self.level.sloped_ceilings, [original])
        self.assertFalse(
            self.workspace._sloped_ceiling_mesh_update_timer.isActive()
        )
        self.assertFalse(self.workspace.viewer._sloped_ceiling_preview_meshes)
        self.assertEqual(self.workspace._canvas_undo_stack, [])

    def test_every_existing_slope_handle_updates_its_persisted_field(self) -> None:
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self._create_slope(commit=True)
        self.workspace._canvas_undo_stack.clear()
        slope = self.level.sloped_ceilings[0]
        edits = (
            (SLOPED_CEILING_HANDLE_START, "start_ratio", 0.2),
            (SLOPED_CEILING_HANDLE_END, "end_ratio", 0.82),
            (SLOPED_CEILING_HANDLE_DEPTH, "depth_meters", 1.1),
            (SLOPED_CEILING_HANDLE_WALL_DROP, "wall_drop_meters", 0.55),
            (
                SLOPED_CEILING_HANDLE_CEILING_TRANSITION_RADIUS,
                "transition_radius_meters",
                0.12,
            ),
            (
                SLOPED_CEILING_HANDLE_WALL_TRANSITION_RADIUS,
                "wall_transition_radius_meters",
                0.08,
            ),
            (
                SLOPED_CEILING_HANDLE_SIDE_TRANSITION_RADIUS,
                "side_transition_radius_meters",
                0.09,
            ),
        )

        for handle_kind, field_name, next_value in edits:
            with self.subTest(handle_kind=handle_kind):
                target = self.workspace._sloped_ceiling_edit_targets_by_wall_id[
                    slope.wall_surface_id
                ]
                handle = next(
                    candidate
                    for candidate in target.handles
                    if candidate.handle_kind == handle_kind
                )
                start = _edit(target, handle_kind, handle.value)
                finish = _edit(target, handle_kind, next_value)
                self.workspace._handle_sloped_ceiling_edit_started(start)
                self.workspace._handle_sloped_ceiling_edit_preview_changed(finish)
                self.workspace._handle_sloped_ceiling_edit_finished(finish, True)
                self.assertAlmostEqual(
                    getattr(self.level.sloped_ceilings[0], field_name),
                    next_value,
                )

        self.assertTrue(
            self.workspace._sloped_ceiling_mesh_update_timer.isActive()
        )
        self.assertEqual(self.workspace._canvas_undo_stack, [])
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self.workspace._commit_pending_sloped_ceiling_mesh_update()
        self.assertEqual(len(self.workspace._canvas_undo_stack), 1)

    def test_non_lateral_slope_edits_preserve_window_physical_dimensions(
        self,
    ) -> None:
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self._create_slope(commit=True)
        self._add_centered_slope_window()
        edits = (
            (SLOPED_CEILING_HANDLE_DEPTH, 1.1),
            (SLOPED_CEILING_HANDLE_WALL_DROP, 0.65),
            (SLOPED_CEILING_HANDLE_CEILING_TRANSITION_RADIUS, 0.08),
            (SLOPED_CEILING_HANDLE_WALL_TRANSITION_RADIUS, 0.06),
        )

        for handle_kind, next_value in edits:
            with self.subTest(handle_kind=handle_kind):
                before_dimensions = self._get_slope_window_dimensions()
                before_window_id = self.level.windows[0].window_id
                slope = self.level.sloped_ceilings[0]
                target = next(
                    candidate
                    for candidate in (
                        self.workspace._sloped_ceiling_edit_targets_by_key.values()
                    )
                    if candidate.slope_id == slope.slope_id
                )
                handle = next(
                    candidate
                    for candidate in target.handles
                    if candidate.handle_kind == handle_kind
                )
                start = _edit(target, handle_kind, handle.value)
                finish = _edit(target, handle_kind, next_value)

                self.workspace._handle_sloped_ceiling_edit_started(start)
                self.workspace._handle_sloped_ceiling_edit_preview_changed(finish)
                self.workspace._handle_sloped_ceiling_edit_finished(finish, True)

                self.assertEqual(self.level.windows[0].window_id, before_window_id)
                np.testing.assert_allclose(
                    self._get_slope_window_dimensions(),
                    before_dimensions,
                    atol=1e-7,
                    rtol=1e-7,
                )
                with patch.object(
                    self.workspace,
                    "_schedule_viewer_preview_refresh",
                ):
                    self.workspace._commit_pending_sloped_ceiling_mesh_update()

    def test_side_rounding_preserves_window_dimensions_when_it_fits(self) -> None:
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self._create_slope(commit=True)
        self._add_centered_slope_window()

        for handle_kind, next_value in (
            (SLOPED_CEILING_HANDLE_START, 0.1),
            (SLOPED_CEILING_HANDLE_END, 0.9),
        ):
            slope = self.level.sloped_ceilings[0]
            target = next(
                candidate
                for candidate in (
                    self.workspace._sloped_ceiling_edit_targets_by_key.values()
                )
                if candidate.slope_id == slope.slope_id
            )
            handle = next(
                candidate
                for candidate in target.handles
                if candidate.handle_kind == handle_kind
            )
            before_dimensions = self._get_slope_window_dimensions()
            self.workspace._handle_sloped_ceiling_edit_started(
                _edit(target, handle_kind, handle.value)
            )
            self.workspace._handle_sloped_ceiling_edit_finished(
                _edit(target, handle_kind, next_value),
                True,
            )
            np.testing.assert_allclose(
                self._get_slope_window_dimensions(),
                before_dimensions,
                atol=1e-7,
                rtol=1e-7,
            )
            with patch.object(
                self.workspace,
                "_schedule_viewer_preview_refresh",
            ):
                self.workspace._commit_pending_sloped_ceiling_mesh_update()

        slope = self.level.sloped_ceilings[0]
        target = next(
            candidate
            for candidate in (
                self.workspace._sloped_ceiling_edit_targets_by_key.values()
            )
            if candidate.slope_id == slope.slope_id
        )
        handle = next(
            candidate
            for candidate in target.handles
            if candidate.handle_kind
            == SLOPED_CEILING_HANDLE_SIDE_TRANSITION_RADIUS
        )
        before_dimensions = self._get_slope_window_dimensions()
        self.workspace._handle_sloped_ceiling_edit_started(
            _edit(target, handle.handle_kind, handle.value)
        )
        self.workspace._handle_sloped_ceiling_edit_finished(
            _edit(target, handle.handle_kind, 0.05),
            True,
        )

        np.testing.assert_allclose(
            self._get_slope_window_dimensions(),
            before_dimensions,
            atol=1e-7,
            rtol=1e-7,
        )

    def test_lateral_slope_edge_crossing_removes_frame_and_is_undoable(
        self,
    ) -> None:
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self._create_slope(commit=True)
        self._add_centered_slope_window()
        original_slope = self.level.sloped_ceilings[0]
        original_window = self.level.windows[0]
        target = next(
            candidate
            for candidate in (
                self.workspace._sloped_ceiling_edit_targets_by_key.values()
            )
            if candidate.slope_id == original_slope.slope_id
        )
        handle = next(
            candidate
            for candidate in target.handles
            if candidate.handle_kind == SLOPED_CEILING_HANDLE_START
        )
        start = _edit(target, handle.handle_kind, handle.value)
        finish = _edit(target, handle.handle_kind, 0.5)

        self.workspace._handle_sloped_ceiling_edit_started(start)
        self.workspace._handle_sloped_ceiling_edit_preview_changed(finish)
        self.assertNotIn(
            "preview stopped",
            self.workspace.viewer.surface_tools_status_label.text().lower(),
        )
        self.workspace._handle_sloped_ceiling_edit_finished(finish, True)

        self.assertAlmostEqual(self.level.sloped_ceilings[0].start_ratio, 0.5)
        self.assertEqual(self.level.windows, [])
        self.assertIsNotNone(self.workspace._pending_sloped_ceiling_undo_state)
        self.assertTrue(
            self.workspace._sloped_ceiling_mesh_update_timer.isActive()
        )

        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self.workspace._handle_canvas_undo_requested()

        self.assertEqual(self.level.sloped_ceilings, [original_slope])
        self.assertEqual(self.level.windows, [original_window])

    def test_crossed_slope_window_detaches_fixture_without_deleting_asset(
        self,
    ) -> None:
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self._create_slope(commit=True)
        self._add_centered_slope_window()
        frame = self.level.windows[0]
        reusable_window = create_window_definition_for_frame(
            frame.window_id,
            0.8,
            0.8,
            name="Slope window",
            window_id="reusable-window",
        )
        fixture = WindowPlacement(
            placement_id="slope-window-fixture",
            window_id=reusable_window.window_id,
            level_index=self.level.index,
            window_frame_id=frame.window_id,
        )
        self.workspace.doors_workspace.set_data(
            DoorLibraryData(
                windows=[reusable_window],
                window_placements=[fixture],
            )
        )
        slope = self.level.sloped_ceilings[0]
        target = next(
            candidate
            for candidate in (
                self.workspace._sloped_ceiling_edit_targets_by_key.values()
            )
            if candidate.slope_id == slope.slope_id
        )
        handle = next(
            candidate
            for candidate in target.handles
            if candidate.handle_kind == SLOPED_CEILING_HANDLE_START
        )

        self.workspace._handle_sloped_ceiling_edit_started(
            _edit(target, handle.handle_kind, handle.value)
        )
        self.workspace._handle_sloped_ceiling_edit_finished(
            _edit(target, handle.handle_kind, 0.5),
            True,
        )

        library = self.workspace.doors_workspace.data()
        self.assertEqual(
            library.get_window(reusable_window.window_id),
            reusable_window,
        )
        self.assertEqual(library.window_placements, [])
        self.assertEqual(self.level.windows, [])

        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self.workspace._commit_pending_sloped_ceiling_mesh_update()
        self.assertIsInstance(
            self.workspace._canvas_undo_stack[-1],
            _CanvasSlopedCeilingUndoState,
        )

        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self.workspace._handle_canvas_undo_requested()

        restored_library = self.workspace.doors_workspace.data()
        self.assertEqual(restored_library.window_placements, [fixture])
        self.assertEqual(self.level.windows, [frame])

    def test_joined_blue_boundary_can_shrink_through_installed_window(
        self,
    ) -> None:
        """Crossing a window removes its placement and still commits the slope."""

        self.level = _build_unequal_closed_parallel_wall_level()
        self.workspace.levels = [self.level]
        self.workspace.current_level_index = 0
        self.workspace._reset_viewer_doorway_snapshots()
        self.workspace._sync_level_controls()
        self.workspace._sync_canvas_to_current_level()
        self.surfaces = tuple(build_fixed_surfaces((self.level,)))
        self.workspace._set_canvas_viewer_targets(self.surfaces)

        creation_targets = tuple(
            target
            for target in self.workspace._sloped_ceiling_edit_targets_by_key.values()
            if (
                target.slope_id is None
                and abs(
                    target.wall_end_world[1] - target.wall_start_world[1]
                )
                < 1e-9
            )
        )
        self.assertEqual(len(creation_targets), 2)
        wall_ids = tuple(
            target.wall_surface_id for target in creation_targets
        )
        self.workspace.viewer.set_selected_canvas_surface_ids(wall_ids)
        active_wall_id = (
            self.workspace.viewer.get_selected_sloped_ceiling_wall_surface_id()
        )
        creation_target = next(
            target
            for target in creation_targets
            if target.wall_surface_id == active_wall_id
        )
        creation_handle = creation_target.handles[0]

        def camera_ray(
            handle: SlopedCeilingEditHandle,
            displacement_meters: float,
        ) -> tuple[np.ndarray, np.ndarray]:
            axis = np.asarray(handle.axis_world, dtype=float)
            axis /= np.linalg.norm(axis)
            camera_direction = np.eye(3, dtype=float)[
                int(np.argmin(np.abs(axis)))
            ]
            hit = (
                np.asarray(handle.origin_world, dtype=float)
                + axis * displacement_meters
            )
            return hit - camera_direction, camera_direction

        with patch.object(
            self.workspace.viewer.view,
            "build_camera_ray",
            return_value=camera_ray(creation_handle, 0.0),
        ):
            self.assertTrue(
                self.workspace.viewer._begin_sloped_ceiling_edit_drag(
                    creation_target,
                    creation_handle,
                    QPointF(),
                )
            )
        with patch.object(
            self.workspace.viewer.view,
            "build_camera_ray",
            return_value=camera_ray(creation_handle, 2.0),
        ):
            self.assertTrue(
                self.workspace.viewer._finish_sloped_ceiling_edit_drag(
                    QPointF()
                )
            )
        with patch.object(
            self.workspace,
            "_schedule_viewer_preview_refresh",
        ):
            self.workspace._commit_pending_sloped_ceiling_mesh_update()

        joined_group_ids = {
            slope.joined_group_id
            for slope in self.level.sloped_ceilings
            if slope.joined_group_id is not None
        }
        self.assertEqual(len(self.level.sloped_ceilings), 2)
        self.assertEqual(len(joined_group_ids), 1)
        original_slopes = tuple(self.level.sloped_ceilings)
        active_slope = next(
            slope
            for slope in self.level.sloped_ceilings
            if slope.wall_surface_id == active_wall_id
        )
        host = next(
            candidate
            for candidate in build_sloped_ceiling_window_hosts(
                self.workspace.levels,
                self.surfaces,
            )
            if candidate.slope_id == active_slope.slope_id
        )
        placement = build_sloped_ceiling_window_placement(
            host,
            host.local_to_world(0.35, 0.3),
            host.local_to_world(0.65, 0.7),
        )
        self.workspace._handle_canvas_window_placement_requested(placement)
        self.assertEqual(len(self.level.windows), 1)
        frame = self.level.windows[0]
        reusable_window = create_window_definition_for_frame(
            frame.window_id,
            0.8,
            0.8,
            name="Joined slope window",
            window_id="joined-slope-window",
        )
        installed_window = WindowPlacement(
            placement_id="joined-slope-window-placement",
            window_id=reusable_window.window_id,
            level_index=self.level.index,
            window_frame_id=frame.window_id,
        )
        self.workspace.doors_workspace.set_data(
            DoorLibraryData(
                windows=[reusable_window],
                window_placements=[installed_window],
            )
        )
        self.workspace._canvas_undo_stack.clear()
        self.workspace._set_mesh_edit_update_delay_seconds(0.02)

        target = next(
            candidate
            for candidate in self.workspace._sloped_ceiling_edit_targets_by_key.values()
            if (
                candidate.wall_surface_id == active_wall_id
                and candidate.slope_id == active_slope.slope_id
            )
        )
        blue_handle = next(
            handle
            for handle in target.handles
            if handle.handle_kind == SLOPED_CEILING_HANDLE_START
        )
        crossing_value = 0.55
        crossing_displacement = (
            crossing_value - blue_handle.value
        ) / blue_handle.value_per_meter
        self.assertEqual(
            len(
                self.workspace.viewer._get_compatible_sloped_ceiling_target_handles(
                    target,
                    blue_handle,
                )
            ),
            2,
        )

        with patch.object(
            self.workspace.viewer.view,
            "build_camera_ray",
            return_value=camera_ray(blue_handle, 0.0),
        ):
            self.assertTrue(
                self.workspace.viewer._begin_sloped_ceiling_edit_drag(
                    target,
                    blue_handle,
                    QPointF(),
                )
            )
        with patch.object(
            self.workspace.viewer.view,
            "build_camera_ray",
            return_value=camera_ray(
                blue_handle,
                crossing_displacement,
            ),
        ):
            self.assertTrue(
                self.workspace.viewer._update_sloped_ceiling_edit_drag(
                    QPointF()
                )
            )
            self.assertTrue(
                self.workspace.viewer._finish_sloped_ceiling_edit_drag(
                    QPointF()
                )
            )

        edited_slope = next(
            slope
            for slope in self.level.sloped_ceilings
            if slope.slope_id == active_slope.slope_id
        )
        self.assertAlmostEqual(edited_slope.start_ratio, crossing_value)
        self.assertEqual(self.level.windows, [])
        library = self.workspace.doors_workspace.data()
        self.assertEqual(library.window_placements, [])
        self.assertEqual(
            library.get_window(reusable_window.window_id),
            reusable_window,
        )
        self.assertIsNotNone(
            self.workspace._pending_sloped_ceiling_undo_state
        )
        self.assertTrue(
            self.workspace._sloped_ceiling_mesh_update_timer.isActive()
        )
        self.assertTrue(self.workspace.viewer._sloped_ceiling_preview_meshes)

        QTest.qWait(100)

        self.assertFalse(
            self.workspace._sloped_ceiling_mesh_update_timer.isActive()
        )
        self.assertIsNone(
            self.workspace._pending_sloped_ceiling_undo_state
        )
        self.assertFalse(self.workspace.viewer._sloped_ceiling_preview_meshes)
        self.assertIsInstance(
            self.workspace._canvas_undo_stack[-1],
            _CanvasSlopedCeilingUndoState,
        )

        with patch.object(
            self.workspace,
            "_schedule_viewer_preview_refresh",
        ):
            self.workspace._handle_canvas_undo_requested()

        self.assertEqual(tuple(self.level.sloped_ceilings), original_slopes)
        self.assertEqual(self.level.windows, [frame])
        restored_library = self.workspace.doors_workspace.data()
        self.assertEqual(
            restored_library.get_window(reusable_window.window_id),
            reusable_window,
        )
        self.assertEqual(
            restored_library.window_placements,
            [installed_window],
        )

    def test_ctrl_z_restores_rebased_slope_window(self) -> None:
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self._create_slope(commit=True)
        self._add_centered_slope_window()
        original_window = self.level.windows[0]
        slope = self.level.sloped_ceilings[0]
        target = next(
            candidate
            for candidate in (
                self.workspace._sloped_ceiling_edit_targets_by_key.values()
            )
            if candidate.slope_id == slope.slope_id
        )
        depth_handle = next(
            candidate
            for candidate in target.handles
            if candidate.handle_kind == SLOPED_CEILING_HANDLE_DEPTH
        )
        self.workspace._handle_sloped_ceiling_edit_started(
            _edit(target, depth_handle.handle_kind, depth_handle.value)
        )
        self.workspace._handle_sloped_ceiling_edit_finished(
            _edit(target, depth_handle.handle_kind, 1.1),
            True,
        )

        self.assertNotEqual(self.level.windows[0], original_window)
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self.workspace._handle_canvas_undo_requested()

        self.assertEqual(self.level.windows, [original_window])
        self.assertEqual(
            self.workspace._viewer_windows_by_level_index[self.level.index],
            (original_window,),
        )

    def test_committed_creation_undo_removes_the_slope(self) -> None:
        self.workspace._canvas_undo_stack.clear()
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self._create_slope(commit=True)
        self.assertEqual(len(self.level.sloped_ceilings), 1)
        self.assertIsInstance(
            self.workspace._canvas_undo_stack[-1],
            _CanvasSlopedCeilingUndoState,
        )

        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self.workspace._handle_canvas_undo_requested()

        self.assertEqual(self.level.sloped_ceilings, [])
        self.assertEqual(
            self.workspace._build_viewer_preview_levels()[0].sloped_ceilings,
            [],
        )
        self.assertEqual(self.workspace._canvas_undo_stack, [])

    def test_deleting_host_wall_removes_slope_and_undo_restores_it(self) -> None:
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self._create_slope(commit=True)
        original = self.level.sloped_ceilings[0]
        host_vertex_id = int(original.wall_surface_id.rsplit(":", 2)[-2])
        self.workspace._canvas_undo_stack.clear()
        self.workspace._sync_canvas_to_current_level()

        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self.workspace.canvas.set_selected_vertex_ids((host_vertex_id,))
            self.workspace.canvas._delete_selected_vertices()
            self.workspace._commit_pending_wall_vertex_update()

        self.assertEqual(self.level.sloped_ceilings, [])
        self.assertEqual(
            self.workspace._build_viewer_preview_levels()[0].sloped_ceilings,
            [],
        )

        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self.workspace._handle_canvas_undo_requested()

        self.assertEqual(self.level.sloped_ceilings, [original])

    def test_selected_slope_routes_window_placement_through_the_3d_view(self) -> None:
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self._create_slope(commit=True)
        surfaces = tuple(build_fixed_surfaces((self.level,)))
        self.workspace._set_canvas_viewer_targets(surfaces)
        self.workspace.viewer.set_ignore_top_down_ceiling(False)
        host = next(iter(self.workspace.viewer._sloped_ceiling_window_hosts.values()))
        self.workspace.viewer._set_sloped_ceiling_window_host_selection(host.slope_id)

        self.workspace._handle_add_window_button_toggled(True)

        self.assertTrue(self.workspace.viewer.is_window_placement_active())
        self.assertFalse(self.workspace.canvas.is_window_placement_active())

        placement = build_sloped_ceiling_window_placement(
            host,
            host.local_to_world(0.2, 0.2),
            host.local_to_world(0.7, 0.65),
        )
        with (
            patch.object(
                self.workspace,
                "_build_model_with_stable_dependencies",
                return_value=(object(), object()),
            ),
            patch.object(self.workspace, "_apply_canvas_window_preview"),
        ):
            self.workspace._handle_canvas_window_placement_requested(placement)

        self.assertEqual(len(self.level.windows), 1)
        self.assertEqual(self.level.windows[0].sloped_ceiling_id, host.slope_id)
        self.assertEqual(
            self.level.windows[0].wall_surface_id,
            host.wall_surface_id,
        )

    def test_window_fixture_transform_uses_the_inclined_host_basis(self) -> None:
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self._create_slope(commit=True)
        surfaces = tuple(build_fixed_surfaces((self.level,)))
        self.workspace._set_canvas_viewer_targets(surfaces)
        self.workspace.viewer.set_ignore_top_down_ceiling(False)
        host = next(iter(self.workspace.viewer._sloped_ceiling_window_hosts.values()))
        placement = build_sloped_ceiling_window_placement(
            host,
            host.local_to_world(0.2, 0.2),
            host.local_to_world(0.7, 0.65),
        )
        window = add_sloped_ceiling_window(self.workspace.levels, placement)
        self.workspace._set_canvas_viewer_targets(
            tuple(build_fixed_surfaces((self.level,)))
        )
        target = next(
            target
            for target in self.workspace._canvas_opening_targets_by_key.values()
            if target.reference.stable_id == window.window_id
        )
        mesh = trimesh.creation.box(extents=(1.0, 0.2, 1.5))
        mesh.apply_translation((0.2, -0.15, 0.75))
        model = GeneratedModel(
            mesh=mesh,
            scene=trimesh.Scene(mesh),
            glb_bytes=b"",
        )

        placed = self.workspace._build_window_placement_model(
            SimpleNamespace(name="Inclined window"),
            self.level,
            target,
            scene_object_id="window-placement:test",
            model=model,
        )

        self.assertIsNotNone(placed)
        assert placed is not None
        transform = build_placed_generated_model_transform(placed)
        midpoint = (target.bounds.start_ratio + target.bounds.end_ratio) * 0.5
        expected_anchor = target.local_to_world(
            midpoint,
            target.bounds.bottom_ratio,
        )
        np.testing.assert_allclose(
            (transform @ np.asarray((0.0, 0.0, 0.0, 1.0)))[:3],
            expected_anchor,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            transform[:3, 0] / np.linalg.norm(transform[:3, 0]),
            target.wall_tangent_world,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            transform[:3, 2] / np.linalg.norm(transform[:3, 2]),
            target.vertical_axis_world,
            atol=1e-8,
        )


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
