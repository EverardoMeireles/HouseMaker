# ### Environment setup ###
from __future__ import annotations

import os
import unittest
from dataclasses import replace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import numpy as np
import trimesh
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from housemaker.canvas_surface_edits import (
    CANVAS_SURFACE_EDIT_WALL_TRANSLATION,
    CanvasSurfaceEditHandleTarget,
    CanvasSurfaceEditReference,
)
from housemaker.sloped_ceiling import (
    SLOPED_CEILING_HANDLE_CEILING_TRANSITION_RADIUS,
    SLOPED_CEILING_HANDLE_CREATE_DEPTH,
    SLOPED_CEILING_HANDLE_DEPTH,
    SLOPED_CEILING_HANDLE_END,
    SLOPED_CEILING_HANDLE_SIDE_TRANSITION_RADIUS,
    SLOPED_CEILING_HANDLE_START,
    SLOPED_CEILING_HANDLE_WALL_DROP,
    SLOPED_CEILING_HANDLE_WALL_TRANSITION_RADIUS,
    SlopedCeilingEditHandle,
    SlopedCeilingEditTarget,
)
from housemaker.surface_geometry import SURFACE_TYPE_WALL, FixedSurface
from housemaker.viewer import (
    SLOPED_CEILING_HANDLE_COLORS,
    GlbViewerWidget,
    _get_sloped_ceiling_creation_peer_distance,
)

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_wall(surface_id: str = "level:0/wall:1:2") -> FixedSurface:
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (4.0, 0.0, 0.0),
            (4.0, 0.0, 3.0),
            (0.0, 0.0, 3.0),
        ),
        dtype=float,
    )
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        process=False,
    )
    return FixedSurface(
        surface_id=surface_id,
        surface_type=SURFACE_TYPE_WALL,
        level_index=0,
        room_index=None,
        mesh=mesh,
        area_square_meters=12.0,
        wall_key="1:2",
        wall_start_world=(0.0, 0.0, 0.0),
        wall_end_world=(4.0, 0.0, 0.0),
        wall_height_meters=3.0,
    )


def _build_child_wall(wall: FixedSurface) -> FixedSurface:
    """Build one editable child face that resolves back to its source wall."""

    return FixedSurface(
        surface_id=f"{wall.surface_id}/edit-face:child",
        surface_type=SURFACE_TYPE_WALL,
        level_index=wall.level_index,
        room_index=wall.room_index,
        mesh=wall.mesh.copy(),
        area_square_meters=wall.area_square_meters,
        source_surface_id=wall.surface_id,
    )


def _build_regular_wall_gizmo(
    wall: FixedSurface,
) -> CanvasSurfaceEditHandleTarget:
    """Build a normal wall gizmo that must coexist with slope controls."""

    return CanvasSurfaceEditHandleTarget(
        reference=CanvasSurfaceEditReference(
            kind=CANVAS_SURFACE_EDIT_WALL_TRANSLATION,
            level_index=wall.level_index,
            axis_index=1,
            wall_vertex_ids=(1, 2),
        ),
        surface_id=wall.surface_id,
        origin_world=(2.0, 0.0, 0.0),
        axis_world=(0.0, 1.0, 0.0),
        minimum_delta_meters=-1.0,
        maximum_delta_meters=1.0,
        chain_vertex_ids=(1, 2),
        chain_vertex_ratios=(0.0, 1.0),
        chain_vertex_image_positions=((0.0, 0.0), (4.0, 0.0)),
    )


def _build_handle(
    handle_kind: str,
    *,
    origin_x: float = 2.0,
    value: float = 0.5,
    minimum: float = 0.0,
    maximum: float = 3.0,
    value_per_meter: float | None = None,
) -> SlopedCeilingEditHandle:
    is_wall_drop = handle_kind == SLOPED_CEILING_HANDLE_WALL_DROP
    return SlopedCeilingEditHandle(
        handle_kind=handle_kind,
        origin_world=(origin_x, 0.0, 3.0),
        axis_world=(0.0, 0.0, 1.0) if is_wall_drop else (1.0, 0.0, 0.0),
        value=value,
        minimum_value=minimum,
        maximum_value=maximum,
        value_per_meter=(
            (-1.0 if is_wall_drop else 1.0)
            if value_per_meter is None
            else value_per_meter
        ),
    )


def _build_target(
    wall: FixedSurface,
    *,
    existing: bool,
    expose_start: bool = True,
    expose_end: bool = True,
    slope_id: str = "slope-1",
    selection_start_x: float | None = None,
    selection_end_x: float | None = None,
    handle_start_x: float = 0.5,
    creation_start_ratio: float | None = None,
    creation_end_ratio: float | None = None,
) -> SlopedCeilingEditTarget:
    if existing:
        handle_kinds = [
            SLOPED_CEILING_HANDLE_DEPTH,
            SLOPED_CEILING_HANDLE_WALL_DROP,
            SLOPED_CEILING_HANDLE_CEILING_TRANSITION_RADIUS,
            SLOPED_CEILING_HANDLE_WALL_TRANSITION_RADIUS,
            SLOPED_CEILING_HANDLE_START,
            SLOPED_CEILING_HANDLE_END,
        ]
        if expose_start or expose_end:
            handle_kinds.append(SLOPED_CEILING_HANDLE_SIDE_TRANSITION_RADIUS)
        handles = tuple(
            _build_handle(kind, origin_x=handle_start_x + index * 0.65)
            for index, kind in enumerate(handle_kinds)
        )
    else:
        handles = (
            _build_handle(
                SLOPED_CEILING_HANDLE_CREATE_DEPTH,
                origin_x=handle_start_x,
            ),
        )
    default_start_x = 0.4 if existing else 0.0
    default_end_x = 3.6 if existing else 4.0
    selection_start = (
        default_start_x if selection_start_x is None else selection_start_x,
        0.0,
        2.6 if existing else 3.0,
    )
    selection_end = (
        default_end_x if selection_end_x is None else selection_end_x,
        0.0,
        2.6 if existing else 3.0,
    )
    return SlopedCeilingEditTarget(
        slope_id=slope_id if existing else None,
        wall_surface_id=wall.surface_id,
        level_index=0,
        handles=handles,
        wall_start_world=(0.0, 0.0, 0.0),
        wall_end_world=(4.0, 0.0, 0.0),
        inward_world=(0.0, 1.0, 0.0),
        ceiling_z_meters=3.0,
        selection_edge_start_world=selection_start,
        selection_edge_end_world=selection_end,
        creation_start_ratio=(
            selection_start[0] / 4.0
            if creation_start_ratio is None
            else creation_start_ratio
        ),
        creation_end_ratio=(
            selection_end[0] / 4.0 if creation_end_ratio is None else creation_end_ratio
        ),
    )


def _configure_lateral_target(
    target: SlopedCeilingEditTarget,
    *,
    wall_start_x: float,
    wall_end_x: float,
    start_ratio: float,
    end_ratio: float,
) -> SlopedCeilingEditTarget:
    """Give an existing target realistic lateral controls and wall orientation."""

    wall_length = abs(wall_end_x - wall_start_x)
    tangent_x = 1.0 if wall_end_x > wall_start_x else -1.0

    def world_x(ratio: float) -> float:
        return wall_start_x + (wall_end_x - wall_start_x) * ratio

    handles: list[SlopedCeilingEditHandle] = []
    for handle in target.handles:
        if handle.handle_kind == SLOPED_CEILING_HANDLE_START:
            handle = replace(
                handle,
                origin_world=(world_x(start_ratio), 0.0, 2.6),
                axis_world=(-tangent_x, 0.0, 0.0),
                value=start_ratio,
                minimum_value=0.0,
                maximum_value=end_ratio - 0.01,
                value_per_meter=-1.0 / wall_length,
            )
        elif handle.handle_kind == SLOPED_CEILING_HANDLE_END:
            handle = replace(
                handle,
                origin_world=(world_x(end_ratio), 0.0, 2.6),
                axis_world=(tangent_x, 0.0, 0.0),
                value=end_ratio,
                minimum_value=start_ratio + 0.01,
                maximum_value=1.0,
                value_per_meter=1.0 / wall_length,
            )
        handles.append(handle)
    return replace(
        target,
        handles=tuple(handles),
        wall_start_world=(wall_start_x, 0.0, 0.0),
        wall_end_world=(wall_end_x, 0.0, 0.0),
        selection_edge_start_world=(world_x(start_ratio), 0.0, 2.6),
        selection_edge_end_world=(world_x(end_ratio), 0.0, 2.6),
    )


def _ray_at_x(x: float) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray((x, -2.0, 3.0), dtype=float),
        np.asarray((0.0, 1.0, 0.0), dtype=float),
    )


def _ray_at_z(z: float) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray((2.0, -2.0, z), dtype=float),
        np.asarray((0.0, 1.0, 0.0), dtype=float),
    )


def _build_parallel_wall_target(
    surface_id: str,
    y: float,
    *,
    inward_y: float,
    existing: bool = False,
) -> tuple[FixedSurface, SlopedCeilingEditTarget]:
    """Place a realistic parallel wall and its controls across the room."""

    source = _build_wall(surface_id)
    mesh = source.mesh.copy()
    mesh.apply_translation((0.0, y, 0.0))
    wall = replace(
        source,
        mesh=mesh,
        wall_start_world=(0.0, y, 0.0),
        wall_end_world=(4.0, y, 0.0),
    )
    target = _build_target(wall, existing=existing)
    target = replace(
        target,
        wall_start_world=wall.wall_start_world,
        wall_end_world=wall.wall_end_world,
        inward_world=(0.0, inward_y, 0.0),
        selection_edge_start_world=(0.0, y, 3.0),
        selection_edge_end_world=(4.0, y, 3.0),
        handles=tuple(
            replace(handle, origin_world=(2.0, y, 3.0), axis_world=(0.0, inward_y, 0.0))
            for handle in target.handles
        ),
    )
    return wall, target


# ### Automatic slope pairing geometry tests ###
class SlopedCeilingCreationPeerTests(unittest.TestCase):
    def test_only_parallel_same_level_overlapping_walls_are_eligible(
        self,
    ) -> None:
        _wall, reference = _build_parallel_wall_target("wall:first", 0.0, inward_y=1.0)
        _wall, peer = _build_parallel_wall_target("wall:second", 2.0, inward_y=-1.0)
        self.assertEqual(
            _get_sloped_ceiling_creation_peer_distance(reference, peer), 2.0
        )
        rejected = {
            "existing slope": replace(peer, slope_id="existing"),
            "other level": replace(peer, level_index=1),
            "different ceiling": replace(peer, ceiling_z_meters=4.0),
            "different base": replace(
                peer,
                wall_start_world=(0.0, 2.0, 1.0),
                wall_end_world=(4.0, 2.0, 1.0),
            ),
            "collinear": replace(
                peer,
                wall_start_world=(0.0, 0.0, 0.0),
                wall_end_world=(4.0, 0.0, 0.0),
            ),
            "perpendicular": replace(peer, wall_end_world=(0.0, 6.0, 0.0)),
            "disjoint span": replace(
                peer,
                selection_edge_start_world=(5.0, 2.0, 3.0),
                selection_edge_end_world=(9.0, 2.0, 3.0),
            ),
        }
        for name, candidate in rejected.items():
            with self.subTest(name=name):
                self.assertIsNone(
                    _get_sloped_ceiling_creation_peer_distance(reference, candidate)
                )

    def test_physical_back_wall_is_eligible_regardless_of_inward_sign_or_winding(
        self,
    ) -> None:
        _wall, reference = _build_parallel_wall_target("wall:first", 0.0, inward_y=1.0)
        for y in (-0.075, 0.075):
            for inward_y in (-1.0, 1.0):
                with self.subTest(y=y, inward_y=inward_y):
                    _wall, peer = _build_parallel_wall_target(
                        "wall:back", y, inward_y=inward_y
                    )
                    self.assertAlmostEqual(
                        _get_sloped_ceiling_creation_peer_distance(reference, peer),
                        0.075,
                    )

    def test_reversed_unequal_walls_remain_eligible(self) -> None:
        _wall, reference = _build_parallel_wall_target("wall:first", 0.0, inward_y=1.0)
        _wall, peer = _build_parallel_wall_target("wall:second", 2.0, inward_y=-1.0)
        peer = replace(
            peer,
            wall_start_world=(5.0, 2.0, 0.0),
            wall_end_world=(0.0, 2.0, 0.0),
            selection_edge_start_world=(5.0, 2.0, 3.0),
            selection_edge_end_world=(0.0, 2.0, 3.0),
        )
        self.assertEqual(
            _get_sloped_ceiling_creation_peer_distance(reference, peer), 2.0
        )


# ### Viewer tests ###
class SlopedCeilingViewerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.widgets: list[GlbViewerWidget] = []

    def tearDown(self) -> None:
        for widget in self.widgets:
            widget.close()
            widget.deleteLater()
        _qt_application.processEvents()

    def _build_viewer(
        self,
        target: SlopedCeilingEditTarget,
    ) -> GlbViewerWidget:
        wall = _build_wall(target.wall_surface_id)
        viewer = GlbViewerWidget(window_editing_enabled=True)
        viewer.set_wall_targets((wall,))
        viewer.set_sloped_ceiling_edit_targets((target,))
        self.widgets.append(viewer)
        return viewer

    def test_created_slope_top_and_bottom_handles_are_both_orange(self) -> None:
        self.assertNotEqual(
            SLOPED_CEILING_HANDLE_COLORS[SLOPED_CEILING_HANDLE_CREATE_DEPTH],
            SLOPED_CEILING_HANDLE_COLORS[SLOPED_CEILING_HANDLE_DEPTH],
        )
        self.assertEqual(
            SLOPED_CEILING_HANDLE_COLORS[SLOPED_CEILING_HANDLE_DEPTH],
            SLOPED_CEILING_HANDLE_COLORS[SLOPED_CEILING_HANDLE_WALL_DROP],
        )

    def test_one_wall_green_drag_selects_nearest_peer_and_keeps_primary_active(
        self,
    ) -> None:
        wall, target = _build_parallel_wall_target("wall:first", 0.0, inward_y=1.0)
        near_wall, near_target = _build_parallel_wall_target(
            "wall:near", -0.075, inward_y=1.0
        )
        far_wall, far_target = _build_parallel_wall_target(
            "wall:far", 4.0, inward_y=-1.0
        )
        viewer = self._build_viewer(target)
        viewer.set_wall_targets((wall, far_wall, near_wall))
        viewer.set_sloped_ceiling_edit_targets((target, far_target, near_target))
        viewer.set_selected_canvas_surface_ids((wall.surface_id,))
        started: list[object] = []
        finished: list[object] = []
        viewer.sloped_ceiling_edit_started.connect(started.append)
        viewer.sloped_ceiling_edit_finished.connect(
            lambda edits, _changed: finished.append(edits)
        )
        self.assertEqual(viewer.get_selected_canvas_surface_ids(), (wall.surface_id,))
        handle = target.handles[0]
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=(np.asarray((2.0, 0.0, 5.0)), np.asarray((0.0, 0.0, -1.0))),
        ):
            self.assertTrue(
                viewer._begin_sloped_ceiling_edit_drag(target, handle, QPointF())
            )
        self.assertEqual(
            viewer.get_selected_canvas_surface_ids(),
            (near_wall.surface_id, wall.surface_id),
        )
        self.assertEqual(
            viewer.get_selected_sloped_ceiling_wall_surface_id(), wall.surface_id
        )
        self.assertEqual(len(started[0]), 2)
        self.assertEqual(viewer._get_displayed_sloped_ceiling_targets(), (target,))
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=(np.asarray((2.0, 0.7, 5.0)), np.asarray((0.0, 0.0, -1.0))),
        ):
            self.assertTrue(viewer._finish_sloped_ceiling_edit_drag(QPointF()))
        self.assertEqual(len(finished[0]), 2)
        np.testing.assert_allclose([edit.value for edit in finished[0]], (1.2, 1.2))

    def test_auto_pair_prefers_largest_overlap_on_nearest_wall_plane(self) -> None:
        wall, target = _build_parallel_wall_target("wall:first", 0.0, inward_y=1.0)
        stub_wall, stub = _build_parallel_wall_target(
            "wall:aaa-stub", -0.075, inward_y=1.0
        )
        stub = replace(
            stub,
            wall_end_world=(0.05, -0.075, 0.0),
            selection_edge_end_world=(0.05, -0.075, 3.0),
        )
        peer_wall, peer = _build_parallel_wall_target(
            "wall:zzz-main", -0.07500000001, inward_y=1.0
        )
        viewer = self._build_viewer(target)
        viewer.set_wall_targets((wall, stub_wall, peer_wall))
        viewer.set_sloped_ceiling_edit_targets((target, stub, peer))
        viewer.set_selected_canvas_surface_ids((wall.surface_id,))
        viewer._select_sloped_ceiling_creation_peer(target, target.handles[0])
        self.assertEqual(
            viewer.get_selected_canvas_surface_ids(),
            (peer_wall.surface_id, wall.surface_id),
        )

    def test_auto_pair_does_not_expand_existing_or_manual_multiwall_selection(
        self,
    ) -> None:
        wall, target = _build_parallel_wall_target("wall:first", 0.0, inward_y=1.0)
        peer_wall, peer = _build_parallel_wall_target("wall:peer", 2.0, inward_y=-1.0)
        far_wall, far = _build_parallel_wall_target("wall:far", 4.0, inward_y=-1.0)
        viewer = self._build_viewer(target)
        viewer.set_wall_targets((wall, peer_wall, far_wall))
        viewer.set_sloped_ceiling_edit_targets((target, peer, far))
        viewer.set_selected_canvas_surface_ids((wall.surface_id, far_wall.surface_id))
        viewer._select_sloped_ceiling_creation_peer(target, target.handles[0])
        self.assertEqual(
            viewer.get_selected_canvas_surface_ids(),
            (wall.surface_id, far_wall.surface_id),
        )
        existing = _build_target(wall, existing=True)
        viewer.set_sloped_ceiling_edit_targets((existing, peer, far))
        viewer.set_selected_canvas_surface_ids((wall.surface_id,))
        viewer._select_sloped_ceiling_creation_peer(existing, existing.handles[0])
        self.assertEqual(viewer.get_selected_canvas_surface_ids(), (wall.surface_id,))

    def test_failed_gizmo_pick_does_not_select_a_peer(self) -> None:
        wall, target = _build_parallel_wall_target("wall:first", 0.0, inward_y=1.0)
        peer_wall, peer = _build_parallel_wall_target("wall:peer", 2.0, inward_y=-1.0)
        viewer = self._build_viewer(target)
        viewer.set_wall_targets((wall, peer_wall))
        viewer.set_sloped_ceiling_edit_targets((target, peer))
        viewer.set_selected_canvas_surface_ids((wall.surface_id,))
        with patch.object(viewer.view, "build_camera_ray", return_value=None):
            self.assertFalse(
                viewer._begin_sloped_ceiling_edit_drag(
                    target, target.handles[0], QPointF()
                )
            )
        self.assertEqual(viewer.get_selected_canvas_surface_ids(), (wall.surface_id,))

    def test_auto_pair_keeps_selected_child_active_and_skips_invisible_walls(self) -> None:
        wall, target = _build_parallel_wall_target("wall:first", 0.0, inward_y=1.0)
        child = _build_child_wall(wall)
        hidden_wall, hidden_target = _build_parallel_wall_target(
            "wall:hidden", 1.0, inward_y=-1.0
        )
        peer_wall, peer = _build_parallel_wall_target("wall:peer", 2.0, inward_y=-1.0)
        viewer = self._build_viewer(target)
        viewer.set_wall_targets((wall, child, hidden_wall, peer_wall))
        viewer.set_sloped_ceiling_edit_targets((target, hidden_target, peer))
        viewer.select_canvas_surface_target(child.surface_id)
        with patch.object(
            viewer,
            "_canvas_surface_is_visible",
            side_effect=lambda surface: surface.surface_id != hidden_wall.surface_id,
        ):
            viewer._select_sloped_ceiling_creation_peer(target, target.handles[0])
        self.assertEqual(
            viewer.get_selected_canvas_surface_ids(),
            (peer_wall.surface_id, child.surface_id),
        )
        self.assertEqual(viewer.get_active_canvas_surface_id(), child.surface_id)
        self.assertEqual(
            viewer.get_selected_sloped_ceiling_wall_surface_id(), wall.surface_id
        )

    def test_wall_body_selection_immediately_exposes_creation_handle(self) -> None:
        wall = _build_wall()
        target = _build_target(wall, existing=False)
        viewer = self._build_viewer(target)

        with (
            patch.object(viewer.view, "build_camera_ray", return_value=_ray_at_x(2.0)),
            patch.object(viewer, "_pick_directional_light_id", return_value=None),
            patch(
                "housemaker.viewer._get_nearest_fixed_surface_ray_hit",
                return_value=(
                    wall,
                    np.asarray((2.0, 0.0, 1.5), dtype=float),
                    2.0,
                ),
            ),
        ):
            viewer._handle_window_wall_pick_requested(QPointF(20.0, 20.0))

        self.assertEqual(
            viewer.get_selected_sloped_ceiling_wall_surface_id(),
            wall.surface_id,
        )
        self.assertEqual(
            viewer.get_selected_sloped_ceiling_wall_surface_ids(),
            (wall.surface_id,),
        )
        self.assertEqual(viewer.get_selected_canvas_surface_ids(), (wall.surface_id,))
        self.assertEqual(
            tuple(handle.handle_kind for handle in target.handles),
            (SLOPED_CEILING_HANDLE_CREATE_DEPTH,),
        )

    def test_child_wall_body_selection_maps_slope_controls_to_source_wall(self) -> None:
        wall = _build_wall()
        child = _build_child_wall(wall)
        target = _build_target(wall, existing=False)
        viewer = self._build_viewer(target)
        viewer.set_wall_targets((wall, child))

        with (
            patch.object(viewer.view, "build_camera_ray", return_value=_ray_at_x(2.0)),
            patch.object(viewer, "_pick_directional_light_id", return_value=None),
            patch(
                "housemaker.viewer._get_nearest_fixed_surface_ray_hit",
                return_value=(
                    child,
                    np.asarray((2.0, 0.0, 1.5), dtype=float),
                    2.0,
                ),
            ),
        ):
            viewer._handle_window_wall_pick_requested(QPointF(20.0, 20.0))

        self.assertEqual(viewer.get_selected_canvas_surface_ids(), (child.surface_id,))
        self.assertEqual(
            viewer.get_selected_sloped_ceiling_wall_surface_id(),
            wall.surface_id,
        )
        self.assertEqual(
            viewer.get_selected_sloped_ceiling_wall_surface_ids(),
            (wall.surface_id,),
        )

    def test_uncreated_slope_and_regular_wall_gizmos_render_together(self) -> None:
        wall = _build_wall()
        target = _build_target(wall, existing=False)
        viewer = self._build_viewer(target)
        regular_gizmo = _build_regular_wall_gizmo(wall)
        viewer.set_canvas_surface_edit_targets((regular_gizmo,))

        with patch.object(viewer.view, "pixelSize", return_value=0.01):
            self.assertTrue(viewer.select_canvas_surface_target(wall.surface_id))
            viewer._refresh_sloped_ceiling_gizmo_items()

        self.assertEqual(
            tuple(handle.handle_kind for handle in target.handles),
            (SLOPED_CEILING_HANDLE_CREATE_DEPTH,),
        )
        self.assertEqual(len(viewer._sloped_ceiling_gizmo_items), 3)
        self.assertEqual(
            viewer._get_active_canvas_surface_edit_targets(),
            (regular_gizmo,),
        )

    def test_one_wall_renders_existing_and_free_span_creation_targets(self) -> None:
        wall = _build_wall()
        existing = _build_target(
            wall,
            existing=True,
            selection_start_x=0.0,
            selection_end_x=2.4,
        )
        creation = _build_target(
            wall,
            existing=False,
            selection_start_x=2.4,
            selection_end_x=4.0,
            handle_start_x=3.2,
        )
        viewer = self._build_viewer(existing)
        viewer.set_sloped_ceiling_edit_targets((existing, creation))

        with patch.object(viewer.view, "pixelSize", return_value=0.01):
            viewer.select_canvas_surface_target(wall.surface_id)
            viewer._refresh_sloped_ceiling_gizmo_items()

        self.assertEqual(len(viewer._sloped_ceiling_edit_targets), 2)
        self.assertEqual(
            viewer.get_selected_sloped_ceiling_wall_surface_ids(),
            (wall.surface_id,),
        )
        self.assertEqual(len(viewer._sloped_ceiling_gizmo_sizes), 8)
        self.assertEqual(len(viewer._sloped_ceiling_gizmo_items), 18)

    def test_parallel_selected_walls_render_one_shared_gizmo_set(self) -> None:
        first_wall = _build_wall("level:0/wall:1:2")
        second_wall = _build_wall("level:0/wall:3:4")
        first_target = _build_target(first_wall, existing=False)
        second_target = replace(
            _build_target(second_wall, existing=False),
            wall_start_world=(4.0, 0.0, 0.0),
            wall_end_world=(0.0, 0.0, 0.0),
        )
        viewer = GlbViewerWidget(window_editing_enabled=True)
        viewer.set_wall_targets((first_wall, second_wall))
        viewer.set_sloped_ceiling_edit_targets((first_target, second_target))
        self.widgets.append(viewer)

        with patch.object(viewer.view, "pixelSize", return_value=0.01):
            viewer.set_selected_canvas_surface_ids(
                (first_wall.surface_id, second_wall.surface_id)
            )
            viewer._refresh_sloped_ceiling_gizmo_items()

        self.assertEqual(
            viewer.get_selected_sloped_ceiling_wall_surface_ids(),
            (first_wall.surface_id, second_wall.surface_id),
        )
        self.assertEqual(
            viewer.get_selected_sloped_ceiling_wall_surface_id(),
            second_wall.surface_id,
        )
        self.assertEqual(len(viewer._sloped_ceiling_gizmo_sizes), 1)
        self.assertEqual(len(viewer._sloped_ceiling_gizmo_items), 3)
        self.assertEqual(
            {
                wall_surface_id
                for wall_surface_id, _target_id, _handle_kind in (
                    viewer._sloped_ceiling_gizmo_sizes
                )
            },
            {second_wall.surface_id},
        )
        active_handle = second_target.handles[0]
        gizmo_size = next(iter(viewer._sloped_ceiling_gizmo_sizes.values()))
        picked = viewer._pick_sloped_ceiling_gizmo_handle(
            *_ray_at_x(active_handle.origin_world[0] + gizmo_size)
        )
        self.assertIsNotNone(picked)
        assert picked is not None
        self.assertIs(picked[0], second_target)

    def test_parallel_incompatible_targets_keep_both_gizmo_sets(self) -> None:
        first_wall = _build_wall("level:0/wall:1:2")
        second_wall = _build_wall("level:0/wall:3:4")
        first_target = _build_target(first_wall, existing=False)
        second_target = _build_target(second_wall, existing=True)
        viewer = GlbViewerWidget(window_editing_enabled=True)
        viewer.set_wall_targets((first_wall, second_wall))
        viewer.set_sloped_ceiling_edit_targets((first_target, second_target))
        self.widgets.append(viewer)

        with patch.object(viewer.view, "pixelSize", return_value=0.01):
            viewer.set_selected_canvas_surface_ids(
                (first_wall.surface_id, second_wall.surface_id)
            )
            viewer._refresh_sloped_ceiling_gizmo_items()

        self.assertEqual(len(viewer._sloped_ceiling_gizmo_sizes), 8)
        self.assertEqual(len(viewer._sloped_ceiling_gizmo_items), 18)

    def test_parallel_different_spans_keep_both_gizmo_sets(self) -> None:
        first_wall = _build_wall("level:0/wall:1:2")
        second_wall = _build_wall("level:0/wall:3:4")
        first_target = _build_target(
            first_wall,
            existing=False,
            selection_start_x=0.0,
            selection_end_x=2.0,
        )
        second_target = _build_target(
            second_wall,
            existing=False,
            selection_start_x=2.0,
            selection_end_x=4.0,
        )
        viewer = GlbViewerWidget(window_editing_enabled=True)
        viewer.set_wall_targets((first_wall, second_wall))
        viewer.set_sloped_ceiling_edit_targets((first_target, second_target))
        self.widgets.append(viewer)

        with patch.object(viewer.view, "pixelSize", return_value=0.01):
            viewer.set_selected_canvas_surface_ids(
                (first_wall.surface_id, second_wall.surface_id)
            )
            viewer._refresh_sloped_ceiling_gizmo_items()

        self.assertEqual(len(viewer._sloped_ceiling_gizmo_sizes), 2)
        self.assertEqual(len(viewer._sloped_ceiling_gizmo_items), 6)

    def test_nonparallel_selected_walls_keep_independent_gizmo_sets(self) -> None:
        first_wall = _build_wall("level:0/wall:1:2")
        second_wall = _build_wall("level:0/wall:3:4")
        first_target = _build_target(first_wall, existing=False)
        second_target = replace(
            _build_target(second_wall, existing=False),
            wall_start_world=(0.0, 0.0, 0.0),
            wall_end_world=(0.0, 4.0, 0.0),
        )
        viewer = GlbViewerWidget(window_editing_enabled=True)
        viewer.set_wall_targets((first_wall, second_wall))
        viewer.set_sloped_ceiling_edit_targets((first_target, second_target))
        self.widgets.append(viewer)

        with patch.object(viewer.view, "pixelSize", return_value=0.01):
            viewer.set_selected_canvas_surface_ids(
                (first_wall.surface_id, second_wall.surface_id)
            )
            viewer._refresh_sloped_ceiling_gizmo_items()

        self.assertEqual(len(viewer._sloped_ceiling_gizmo_sizes), 2)
        self.assertEqual(len(viewer._sloped_ceiling_gizmo_items), 6)

    def test_duplicate_wall_creation_handles_are_picked_independently(self) -> None:
        wall = _build_wall()
        left_target = _build_target(
            wall,
            existing=False,
            selection_start_x=0.0,
            selection_end_x=1.0,
            handle_start_x=0.5,
        )
        right_target = _build_target(
            wall,
            existing=False,
            selection_start_x=2.5,
            selection_end_x=4.0,
            handle_start_x=3.25,
        )
        viewer = self._build_viewer(left_target)
        viewer.set_sloped_ceiling_edit_targets((left_target, right_target))

        with patch.object(viewer.view, "pixelSize", return_value=0.01):
            viewer.select_canvas_surface_target(wall.surface_id)
            viewer._refresh_sloped_ceiling_gizmo_items()
        right_handle = right_target.handles[0]
        right_size = viewer._sloped_ceiling_gizmo_sizes[
            next(
                key
                for key in viewer._sloped_ceiling_gizmo_sizes
                if key[1] == right_target.target_id
            )
        ]
        right_endpoint_x = right_handle.origin_world[0] + right_size

        picked = viewer._pick_sloped_ceiling_gizmo_handle(*_ray_at_x(right_endpoint_x))

        self.assertIsNotNone(picked)
        assert picked is not None
        self.assertIs(picked[0], right_target)
        self.assertIs(picked[1], right_handle)

    def test_drag_emits_one_compatible_edit_per_selected_wall(self) -> None:
        first_wall = _build_wall("level:0/wall:1:2")
        second_wall = _build_wall("level:0/wall:3:4")
        first_target = _build_target(first_wall, existing=False)
        second_target = _build_target(second_wall, existing=False)
        active_handle = second_target.handles[0]
        viewer = GlbViewerWidget(window_editing_enabled=True)
        viewer.set_wall_targets((first_wall, second_wall))
        viewer.set_sloped_ceiling_edit_targets((first_target, second_target))
        viewer.set_selected_canvas_surface_ids(
            (first_wall.surface_id, second_wall.surface_id)
        )
        self.widgets.append(viewer)
        started: list[object] = []
        previews: list[object] = []
        finished: list[tuple[object, bool]] = []
        viewer.sloped_ceiling_edit_started.connect(started.append)
        viewer.sloped_ceiling_edit_preview_changed.connect(previews.append)
        viewer.sloped_ceiling_edit_finished.connect(
            lambda edits, changed: finished.append((edits, changed))
        )

        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at_x(active_handle.origin_world[0]),
        ):
            self.assertTrue(
                viewer._begin_sloped_ceiling_edit_drag(
                    second_target,
                    active_handle,
                    QPointF(),
                )
            )
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at_x(active_handle.origin_world[0] + 0.7),
        ):
            self.assertTrue(viewer._update_sloped_ceiling_edit_drag(QPointF()))
            self.assertTrue(viewer._finish_sloped_ceiling_edit_drag(QPointF()))

        self.assertEqual(len(started), 1)
        self.assertEqual(len(started[0]), 2)
        self.assertEqual(len(previews), 1)
        self.assertEqual(
            {edit.wall_surface_id for edit in previews[0]},
            {first_wall.surface_id, second_wall.surface_id},
        )
        self.assertEqual({edit.value for edit in previews[0]}, {1.2})
        self.assertEqual(len(finished), 1)
        self.assertEqual(len(finished[0][0]), 2)
        self.assertTrue(finished[0][1])

    def test_shared_drag_uses_the_common_handle_value_range(self) -> None:
        first_wall = _build_wall("level:0/wall:1:2")
        second_wall = _build_wall("level:0/wall:3:4")
        first_target = _build_target(first_wall, existing=False)
        first_target = replace(
            first_target,
            handles=(
                replace(
                    first_target.handles[0],
                    maximum_value=0.9,
                ),
            ),
        )
        second_target = _build_target(second_wall, existing=False)
        active_handle = second_target.handles[0]
        viewer = GlbViewerWidget(window_editing_enabled=True)
        viewer.set_wall_targets((first_wall, second_wall))
        viewer.set_sloped_ceiling_edit_targets((first_target, second_target))
        viewer.set_selected_canvas_surface_ids(
            (first_wall.surface_id, second_wall.surface_id)
        )
        self.widgets.append(viewer)
        previews: list[object] = []
        viewer.sloped_ceiling_edit_preview_changed.connect(previews.append)

        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at_x(active_handle.origin_world[0]),
        ):
            self.assertTrue(
                viewer._begin_sloped_ceiling_edit_drag(
                    second_target,
                    active_handle,
                    QPointF(),
                )
            )
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at_x(active_handle.origin_world[0] + 0.7),
        ):
            self.assertTrue(viewer._update_sloped_ceiling_edit_drag(QPointF()))

        self.assertEqual(len(previews), 1)
        self.assertEqual({edit.value for edit in previews[0]}, {0.9})

    def test_reversed_parallel_lateral_handle_moves_the_same_world_side(self) -> None:
        first_wall = _build_wall("level:0/wall:1:2")
        second_wall = _build_wall("level:0/wall:3:4")
        first_target = _configure_lateral_target(
            _build_target(first_wall, existing=True, slope_id="first-slope"),
            wall_start_x=0.0,
            wall_end_x=4.0,
            start_ratio=0.1,
            end_ratio=0.8,
        )
        second_target = _configure_lateral_target(
            _build_target(second_wall, existing=True, slope_id="second-slope"),
            wall_start_x=4.0,
            wall_end_x=0.0,
            start_ratio=0.2,
            end_ratio=0.9,
        )
        active_handle = next(
            handle
            for handle in second_target.handles
            if handle.handle_kind == SLOPED_CEILING_HANDLE_END
        )
        viewer = GlbViewerWidget(window_editing_enabled=True)
        viewer.set_wall_targets((first_wall, second_wall))
        viewer.set_sloped_ceiling_edit_targets((first_target, second_target))
        viewer.set_selected_canvas_surface_ids(
            (first_wall.surface_id, second_wall.surface_id)
        )
        self.widgets.append(viewer)
        previews: list[tuple[object, ...]] = []
        viewer.sloped_ceiling_edit_preview_changed.connect(previews.append)

        affected = viewer._get_compatible_sloped_ceiling_target_handles(
            second_target,
            active_handle,
        )
        self.assertEqual(
            tuple(handle.handle_kind for _target, handle in affected),
            (SLOPED_CEILING_HANDLE_END, SLOPED_CEILING_HANDLE_START),
        )
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at_x(active_handle.origin_world[0]),
        ):
            self.assertTrue(
                viewer._begin_sloped_ceiling_edit_drag(
                    second_target,
                    active_handle,
                    QPointF(),
                )
            )
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at_x(active_handle.origin_world[0] + 0.4),
        ):
            self.assertTrue(viewer._update_sloped_ceiling_edit_drag(QPointF()))

        self.assertEqual(len(previews), 1)
        edits_by_wall = {edit.wall_surface_id: edit for edit in previews[0]}
        self.assertEqual(
            edits_by_wall[first_wall.surface_id].handle_kind,
            SLOPED_CEILING_HANDLE_START,
        )
        self.assertEqual(
            edits_by_wall[second_wall.surface_id].handle_kind,
            SLOPED_CEILING_HANDLE_END,
        )
        self.assertAlmostEqual(edits_by_wall[first_wall.surface_id].value, 0.2)
        self.assertAlmostEqual(edits_by_wall[second_wall.surface_id].value, 0.8)

    def test_unequal_parallel_walls_stay_joined_after_lateral_release(self) -> None:
        first_wall = _build_wall("level:0/wall:1:2")
        second_wall = _build_wall("level:0/wall:3:4")
        first_target = _configure_lateral_target(
            _build_target(first_wall, existing=True, slope_id="first-slope"),
            wall_start_x=0.0,
            wall_end_x=4.0,
            start_ratio=0.1,
            end_ratio=1.0,
        )
        second_target = _configure_lateral_target(
            _build_target(second_wall, existing=True, slope_id="second-slope"),
            wall_start_x=6.0,
            wall_end_x=0.0,
            start_ratio=0.0,
            end_ratio=1.0 - (0.4 / 6.0),
        )
        viewer = GlbViewerWidget(window_editing_enabled=True)
        viewer.set_wall_targets((first_wall, second_wall))
        viewer.set_sloped_ceiling_edit_targets((first_target, second_target))
        self.widgets.append(viewer)

        with patch.object(viewer.view, "pixelSize", return_value=0.01):
            viewer.set_selected_canvas_surface_ids(
                (first_wall.surface_id, second_wall.surface_id)
            )
            viewer._refresh_sloped_ceiling_gizmo_items()

        self.assertEqual(viewer._get_displayed_sloped_ceiling_targets(), (second_target,))
        self.assertEqual(len(viewer._sloped_ceiling_gizmo_sizes), 7)
        for handle_kind, peer_kind in (
            (SLOPED_CEILING_HANDLE_START, SLOPED_CEILING_HANDLE_END),
            (SLOPED_CEILING_HANDLE_END, SLOPED_CEILING_HANDLE_START),
        ):
            with self.subTest(handle_kind=handle_kind):
                handle = next(
                    candidate
                    for candidate in second_target.handles
                    if candidate.handle_kind == handle_kind
                )
                affected = viewer._get_compatible_sloped_ceiling_target_handles(
                    second_target,
                    handle,
                )
                self.assertEqual(len(affected), 2)
                self.assertEqual(affected[1][1].handle_kind, peer_kind)

    def test_unequal_parallel_walls_share_the_freed_creation_span(self) -> None:
        first_wall = _build_wall("level:0/wall:1:2")
        second_wall = _build_wall("level:0/wall:3:4")
        first_slope = _configure_lateral_target(
            _build_target(first_wall, existing=True, slope_id="first-slope"),
            wall_start_x=0.0,
            wall_end_x=4.0,
            start_ratio=0.1,
            end_ratio=1.0,
        )
        first_creation = _configure_lateral_target(
            _build_target(
                first_wall,
                existing=False,
                selection_start_x=0.0,
                selection_end_x=0.4,
            ),
            wall_start_x=0.0,
            wall_end_x=4.0,
            start_ratio=0.0,
            end_ratio=0.1,
        )
        second_slope = _configure_lateral_target(
            _build_target(second_wall, existing=True, slope_id="second-slope"),
            wall_start_x=6.0,
            wall_end_x=0.0,
            start_ratio=0.0,
            end_ratio=1.0 - (0.4 / 6.0),
        )
        second_creation = _configure_lateral_target(
            _build_target(
                second_wall,
                existing=False,
                selection_start_x=0.0,
                selection_end_x=0.4,
            ),
            wall_start_x=6.0,
            wall_end_x=0.0,
            start_ratio=1.0 - (0.4 / 6.0),
            end_ratio=1.0,
        )
        viewer = GlbViewerWidget(window_editing_enabled=True)
        viewer.set_wall_targets((first_wall, second_wall))
        viewer.set_sloped_ceiling_edit_targets(
            (first_slope, first_creation, second_slope, second_creation)
        )
        self.widgets.append(viewer)

        viewer.set_selected_canvas_surface_ids(
            (first_wall.surface_id, second_wall.surface_id)
        )

        self.assertEqual(
            viewer._get_displayed_sloped_ceiling_targets(),
            (second_slope, second_creation),
        )
        affected = viewer._get_compatible_sloped_ceiling_target_handles(
            second_creation,
            second_creation.handles[0],
        )
        self.assertEqual(len(affected), 2)

    def test_disjoint_parallel_creation_spans_are_not_joined(self) -> None:
        first_wall = _build_wall("level:0/wall:1:2")
        second_wall = _build_wall("level:0/wall:3:4")
        first_target = _configure_lateral_target(
            _build_target(first_wall, existing=False),
            wall_start_x=0.0,
            wall_end_x=4.0,
            start_ratio=0.0,
            end_ratio=1.0,
        )
        second_target = _configure_lateral_target(
            _build_target(second_wall, existing=False),
            wall_start_x=10.0,
            wall_end_x=6.0,
            start_ratio=0.0,
            end_ratio=1.0,
        )
        viewer = GlbViewerWidget(window_editing_enabled=True)
        viewer.set_wall_targets((first_wall, second_wall))
        viewer.set_sloped_ceiling_edit_targets((first_target, second_target))
        viewer.set_selected_canvas_surface_ids(
            (first_wall.surface_id, second_wall.surface_id)
        )
        self.widgets.append(viewer)

        affected = viewer._get_compatible_sloped_ceiling_target_handles(
            first_target,
            first_target.handles[0],
        )

        self.assertEqual(affected, ((first_target, first_target.handles[0]),))
        self.assertEqual(
            viewer._get_displayed_sloped_ceiling_targets(),
            (first_target, second_target),
        )

    def test_parallel_creation_spans_on_different_levels_are_not_joined(
        self,
    ) -> None:
        first_wall = _build_wall("level:0/wall:1:2")
        second_wall = replace(
            _build_wall("level:1/wall:3:4"),
            level_index=1,
        )
        first_target = _build_target(first_wall, existing=False)
        second_target = replace(
            _build_target(second_wall, existing=False),
            level_index=1,
        )
        viewer = GlbViewerWidget(window_editing_enabled=True)
        viewer.set_wall_targets((first_wall, second_wall))
        viewer.set_sloped_ceiling_edit_targets((first_target, second_target))
        viewer.set_selected_canvas_surface_ids(
            (first_wall.surface_id, second_wall.surface_id)
        )
        self.widgets.append(viewer)

        affected = viewer._get_compatible_sloped_ceiling_target_handles(
            first_target,
            first_target.handles[0],
        )

        self.assertEqual(affected, ((first_target, first_target.handles[0]),))

    def test_equal_width_rebuilt_gaps_pair_by_shared_direction_order(self) -> None:
        first_wall = _build_wall("level:0/wall:1:2")
        second_wall = _build_wall("level:0/wall:3:4")
        first_left = _configure_lateral_target(
            _build_target(first_wall, existing=False),
            wall_start_x=0.0,
            wall_end_x=4.0,
            start_ratio=0.1,
            end_ratio=0.3,
        )
        first_right = _configure_lateral_target(
            _build_target(first_wall, existing=False),
            wall_start_x=0.0,
            wall_end_x=4.0,
            start_ratio=0.6,
            end_ratio=0.8,
        )
        second_left = _configure_lateral_target(
            _build_target(second_wall, existing=False),
            wall_start_x=4.0,
            wall_end_x=0.0,
            start_ratio=0.7,
            end_ratio=0.9,
        )
        second_right = _configure_lateral_target(
            _build_target(second_wall, existing=False),
            wall_start_x=4.0,
            wall_end_x=0.0,
            start_ratio=0.2,
            end_ratio=0.4,
        )
        targets = (first_left, first_right, second_left, second_right)

        left_group = GlbViewerWidget._get_parallel_sloped_ceiling_target_group(
            second_left,
            targets,
            (first_wall.surface_id, second_wall.surface_id),
        )
        right_group = GlbViewerWidget._get_parallel_sloped_ceiling_target_group(
            second_right,
            targets,
            (first_wall.surface_id, second_wall.surface_id),
        )

        self.assertEqual(left_group, (second_left, first_left))
        self.assertEqual(right_group, (second_right, first_right))

    def test_rebuilt_multiple_slopes_pair_by_shared_direction_order(self) -> None:
        first_wall = _build_wall("level:0/wall:1:2")
        second_wall = _build_wall("level:0/wall:3:4")
        first_left = _configure_lateral_target(
            _build_target(first_wall, existing=True, slope_id="first-left"),
            wall_start_x=0.0,
            wall_end_x=4.0,
            start_ratio=0.1,
            end_ratio=0.3,
        )
        first_right = _configure_lateral_target(
            _build_target(first_wall, existing=True, slope_id="first-right"),
            wall_start_x=0.0,
            wall_end_x=4.0,
            start_ratio=0.6,
            end_ratio=0.8,
        )
        second_left = _configure_lateral_target(
            _build_target(second_wall, existing=True, slope_id="second-left"),
            wall_start_x=4.0,
            wall_end_x=0.0,
            start_ratio=0.699,
            end_ratio=0.899,
        )
        second_right = _configure_lateral_target(
            _build_target(second_wall, existing=True, slope_id="second-right"),
            wall_start_x=4.0,
            wall_end_x=0.0,
            start_ratio=0.199,
            end_ratio=0.399,
        )
        targets = (first_left, first_right, second_left, second_right)

        left_group = GlbViewerWidget._get_parallel_sloped_ceiling_target_group(
            second_left,
            targets,
            (first_wall.surface_id, second_wall.surface_id),
        )
        right_group = GlbViewerWidget._get_parallel_sloped_ceiling_target_group(
            second_right,
            targets,
            (first_wall.surface_id, second_wall.surface_id),
        )

        self.assertEqual(left_group, (second_left, first_left))
        self.assertEqual(right_group, (second_right, first_right))

    def test_parallel_asymmetric_values_keep_one_shared_gizmo_set(
        self,
    ) -> None:
        first_wall = _build_wall("level:0/wall:1:2")
        second_wall = _build_wall("level:0/wall:3:4")
        first_target = _build_target(first_wall, existing=False)
        first_target = replace(
            first_target,
            handles=(
                replace(
                    first_target.handles[0],
                    maximum_value=0.9,
                ),
            ),
        )
        second_target = _build_target(second_wall, existing=False)
        second_target = replace(
            second_target,
            handles=(
                replace(
                    second_target.handles[0],
                    value=1.2,
                ),
            ),
        )
        viewer = GlbViewerWidget(window_editing_enabled=True)
        viewer.set_wall_targets((first_wall, second_wall))
        viewer.set_sloped_ceiling_edit_targets((first_target, second_target))
        self.widgets.append(viewer)

        with patch.object(viewer.view, "pixelSize", return_value=0.01):
            viewer.set_selected_canvas_surface_ids(
                (first_wall.surface_id, second_wall.surface_id)
            )
            viewer._refresh_sloped_ceiling_gizmo_items()

        self.assertEqual(len(viewer._sloped_ceiling_gizmo_sizes), 1)
        self.assertEqual(len(viewer._sloped_ceiling_gizmo_items), 3)

    def test_rebuilt_parallel_slopes_with_asymmetric_limits_stay_shared(
        self,
    ) -> None:
        first_wall = _build_wall("level:0/wall:1:2")
        second_wall = _build_wall("level:0/wall:3:4")
        first_target = _build_target(
            first_wall,
            existing=True,
            slope_id="first-slope",
        )
        first_target = replace(
            first_target,
            handles=tuple(
                replace(handle, maximum_value=0.9)
                if handle.handle_kind == SLOPED_CEILING_HANDLE_DEPTH
                else handle
                for handle in first_target.handles
            ),
        )
        second_target = _build_target(
            second_wall,
            existing=True,
            slope_id="second-slope",
        )
        second_target = replace(
            second_target,
            handles=tuple(
                replace(handle, value=1.2)
                if handle.handle_kind == SLOPED_CEILING_HANDLE_DEPTH
                else handle
                for handle in second_target.handles
            ),
        )
        viewer = GlbViewerWidget(window_editing_enabled=True)
        viewer.set_wall_targets((first_wall, second_wall))
        viewer.set_sloped_ceiling_edit_targets((first_target, second_target))
        self.widgets.append(viewer)

        with patch.object(viewer.view, "pixelSize", return_value=0.01):
            viewer.set_selected_canvas_surface_ids(
                (first_wall.surface_id, second_wall.surface_id)
            )
            viewer._refresh_sloped_ceiling_gizmo_items()

        self.assertEqual(len(viewer._sloped_ceiling_gizmo_sizes), 7)
        self.assertEqual(len(viewer._sloped_ceiling_gizmo_items), 15)

    def test_existing_partial_slope_renders_all_seven_edit_controls(self) -> None:
        wall = _build_wall()
        target = _build_target(wall, existing=True)
        viewer = self._build_viewer(target)

        with patch.object(viewer.view, "pixelSize", return_value=0.01):
            viewer.select_sloped_ceiling_edge(wall.surface_id)
            viewer._refresh_sloped_ceiling_gizmo_items()

        self.assertEqual(
            {handle.handle_kind for handle in target.handles},
            {
                SLOPED_CEILING_HANDLE_DEPTH,
                SLOPED_CEILING_HANDLE_WALL_DROP,
                SLOPED_CEILING_HANDLE_CEILING_TRANSITION_RADIUS,
                SLOPED_CEILING_HANDLE_WALL_TRANSITION_RADIUS,
                SLOPED_CEILING_HANDLE_START,
                SLOPED_CEILING_HANDLE_END,
                SLOPED_CEILING_HANDLE_SIDE_TRANSITION_RADIUS,
            },
        )
        self.assertEqual(len(viewer._sloped_ceiling_gizmo_items), 15)
        np.testing.assert_allclose(
            np.asarray(viewer._sloped_ceiling_gizmo_items[0].pos, dtype=float),
            np.asarray(
                (
                    target.selection_edge_start_world,
                    target.selection_edge_end_world,
                ),
                dtype=float,
            ),
            atol=1e-9,
        )

    def test_existing_full_width_slope_omits_lateral_rounding_handles(self) -> None:
        wall = _build_wall()
        target = _build_target(
            wall,
            existing=True,
            expose_start=False,
            expose_end=False,
        )
        viewer = self._build_viewer(target)

        with patch.object(viewer.view, "pixelSize", return_value=0.01):
            viewer.select_canvas_surface_target(wall.surface_id)
            viewer._refresh_sloped_ceiling_gizmo_items()

        self.assertNotIn(
            SLOPED_CEILING_HANDLE_SIDE_TRANSITION_RADIUS,
            {handle.handle_kind for handle in target.handles},
        )
        self.assertEqual(len(target.handles), 6)
        self.assertEqual(len(viewer._sloped_ceiling_gizmo_items), 13)

    def test_either_or_both_exposed_sides_render_one_shared_control(self) -> None:
        for expose_start, expose_end in ((True, False), (False, True), (True, True)):
            with self.subTest(
                expose_start=expose_start,
                expose_end=expose_end,
            ):
                wall = _build_wall()
                target = _build_target(
                    wall,
                    existing=True,
                    expose_start=expose_start,
                    expose_end=expose_end,
                )
                viewer = self._build_viewer(target)
                with patch.object(viewer.view, "pixelSize", return_value=0.01):
                    viewer.select_canvas_surface_target(wall.surface_id)
                    viewer._refresh_sloped_ceiling_gizmo_items()

                self.assertEqual(
                    sum(
                        handle.handle_kind
                        == SLOPED_CEILING_HANDLE_SIDE_TRANSITION_RADIUS
                        for handle in target.handles
                    ),
                    1,
                )
                self.assertEqual(len(target.handles), 7)
                self.assertEqual(len(viewer._sloped_ceiling_gizmo_items), 15)
                self.assertEqual(len(viewer._sloped_ceiling_gizmo_sizes), 7)

    def test_clearing_regular_surface_selection_hides_slope_controls(self) -> None:
        wall = _build_wall()
        target = _build_target(wall, existing=False)
        viewer = self._build_viewer(target)
        viewer.select_canvas_surface_target(wall.surface_id)
        self.assertEqual(
            viewer.get_selected_sloped_ceiling_wall_surface_id(),
            wall.surface_id,
        )

        viewer.select_canvas_surface_target(None)

        self.assertIsNone(viewer.get_selected_sloped_ceiling_wall_surface_id())
        self.assertEqual(viewer._sloped_ceiling_gizmo_items, [])

    def test_drag_emits_absolute_payloads_and_finishes_changed(self) -> None:
        wall = _build_wall()
        target = _build_target(wall, existing=True)
        handle = target.handles[0]
        viewer = self._build_viewer(target)
        viewer.select_sloped_ceiling_edge(wall.surface_id)
        started: list[object] = []
        previews: list[object] = []
        finished: list[tuple[object, bool]] = []
        viewer.sloped_ceiling_edit_started.connect(started.append)
        viewer.sloped_ceiling_edit_preview_changed.connect(previews.append)
        viewer.sloped_ceiling_edit_finished.connect(
            lambda edit, changed: finished.append((edit, changed))
        )

        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at_x(handle.origin_world[0]),
        ):
            self.assertTrue(
                viewer._begin_sloped_ceiling_edit_drag(target, handle, QPointF())
            )
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at_x(handle.origin_world[0] + 0.7),
        ):
            self.assertTrue(viewer._update_sloped_ceiling_edit_drag(QPointF()))
            self.assertTrue(viewer._finish_sloped_ceiling_edit_drag(QPointF()))

        self.assertEqual(len(started), 1)
        self.assertEqual(len(started[0]), 1)
        self.assertEqual(started[0][0].handle_kind, handle.handle_kind)
        self.assertAlmostEqual(started[0][0].value, handle.value)
        self.assertEqual(len(previews), 1)
        self.assertEqual(len(previews[0]), 1)
        self.assertAlmostEqual(previews[0][0].value, handle.value + 0.7)
        self.assertEqual(len(finished), 1)
        self.assertEqual(len(finished[0][0]), 1)
        self.assertAlmostEqual(finished[0][0][0].value, handle.value + 0.7)
        self.assertTrue(finished[0][1])
        self.assertFalse(viewer.view.is_primary_pointer_drag_reserved)

    def test_upward_orange_drag_reduces_drop_value(self) -> None:
        wall = _build_wall()
        target = _build_target(wall, existing=True)
        handle = next(
            candidate
            for candidate in target.handles
            if candidate.handle_kind == SLOPED_CEILING_HANDLE_WALL_DROP
        )
        viewer = self._build_viewer(target)
        viewer.select_canvas_surface_target(wall.surface_id)
        previews: list[object] = []
        viewer.sloped_ceiling_edit_preview_changed.connect(previews.append)

        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at_z(handle.origin_world[2]),
        ):
            self.assertTrue(
                viewer._begin_sloped_ceiling_edit_drag(target, handle, QPointF())
            )
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at_z(handle.origin_world[2] + 0.2),
        ):
            self.assertTrue(viewer._update_sloped_ceiling_edit_drag(QPointF()))

        self.assertEqual(len(previews), 1)
        self.assertEqual(len(previews[0]), 1)
        self.assertAlmostEqual(previews[0][0].value, handle.value - 0.2)

    def test_cancel_restores_baseline_payload_and_cleanup_removes_items(self) -> None:
        wall = _build_wall()
        target = _build_target(wall, existing=True)
        handle = target.handles[0]
        viewer = self._build_viewer(target)
        viewer.select_sloped_ceiling_edge(wall.surface_id)
        cancelled: list[object] = []
        viewer.sloped_ceiling_edit_cancelled.connect(cancelled.append)

        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at_x(handle.origin_world[0]),
        ):
            viewer._begin_sloped_ceiling_edit_drag(target, handle, QPointF())
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at_x(handle.origin_world[0] + 0.4),
        ):
            viewer._update_sloped_ceiling_edit_drag(QPointF())
        self.assertTrue(viewer.cancel_sloped_ceiling_edit())
        self.assertEqual(len(cancelled), 1)
        self.assertEqual(len(cancelled[0]), 1)
        self.assertAlmostEqual(cancelled[0][0].value, handle.value)

        preview_mesh = trimesh.creation.box(extents=(1.0, 1.0, 0.1))
        viewer.set_sloped_ceiling_preview_meshes((preview_mesh,))
        self.assertEqual(len(viewer._sloped_ceiling_preview_items), 1)
        viewer.clear_sloped_ceiling_preview()
        self.assertEqual(viewer._sloped_ceiling_preview_items, [])
        viewer.select_sloped_ceiling_edge(None)
        self.assertEqual(viewer._sloped_ceiling_gizmo_items, [])


if __name__ == "__main__":
    unittest.main()
