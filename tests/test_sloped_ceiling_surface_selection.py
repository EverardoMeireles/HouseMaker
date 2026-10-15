# ### Environment setup ###
from __future__ import annotations

import os
import unittest
from dataclasses import replace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import numpy as np
import shapely
import trimesh
from OpenGL import GL
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication
from shapely import Polygon

from housemaker.models import LevelData, RoomData, SlopedCeilingData, VertexData
from housemaker.sloped_ceiling import (
    SLOPED_CEILING_HANDLE_DEPTH,
    SlopedCeilingEditHandle,
    SlopedCeilingEditTarget,
    SlopedCeilingWindowHost,
    apply_sloped_ceiling_geometry,
)
from housemaker.surface_geometry import (
    SURFACE_TYPE_CEILING,
    SURFACE_TYPE_WALL,
    FixedSurface,
    build_base_fixed_surfaces,
)
from housemaker.surface_texture_state import (
    get_surface_type_for_id,
    normalize_surface_id,
)
from housemaker.viewer import (
    CANVAS_OPENING_OVERLAY_DEPTH_VALUE,
    GlbViewerWidget,
)

# ### Constants ###
FIRST_SLOPE_ID = "a" * 32
SECOND_SLOPE_ID = "b" * 32
THIRD_SLOPE_ID = "c" * 32
FOURTH_SLOPE_ID = "d" * 32


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Geometry fixture helpers ###
def _build_room_level() -> LevelData:
    vertex_data = VertexData()
    boundary_ids = tuple(
        vertex_data.add_vertex(*point).id
        for point in (
            (0.0, 0.0),
            (400.0, 0.0),
            (400.0, 400.0),
            (0.0, 400.0),
        )
    )
    for start_id, end_id in zip(
        boundary_ids,
        (*boundary_ids[1:], boundary_ids[0]),
        strict=True,
    ):
        vertex_data.add_edge(start_id, end_id)
    center = vertex_data.add_vertex(200.0, 200.0)
    return LevelData(
        index=2,
        name="Ground",
        height_meters=3.0,
        floor_thickness_meters=0.3,
        vertex_data=vertex_data,
        rooms=[
            RoomData(
                name="Room",
                vertex_ids=boundary_ids,
                center_vertex_id=center.id,
                color_rgb=(140, 180, 220),
            )
        ],
    )


def _host_wall(surfaces: list[FixedSurface]) -> FixedSurface:
    return next(
        surface
        for surface in surfaces
        if surface.surface_type == SURFACE_TYPE_WALL
        and np.allclose(surface.wall_start_world[1], 0.0)
        and np.allclose(surface.wall_end_world[1], 0.0)
    )


def _projected_mesh_coverage(mesh: trimesh.Trimesh) -> object:
    polygons = []
    for triangle in np.asarray(mesh.triangles, dtype=float):
        polygon = Polygon(triangle[:, :2])
        if polygon.area > 1e-10:
            polygons.append(polygon)
    return shapely.union_all(polygons)


def _canonical_triangles(mesh: trimesh.Trimesh) -> tuple[object, ...]:
    triangles = []
    for triangle in np.asarray(mesh.triangles, dtype=float):
        points = tuple(
            sorted(
                tuple(round(float(value), 8) for value in point) for point in triangle
            )
        )
        triangles.append(points)
    return tuple(sorted(triangles))


# ### Viewer fixture helpers ###
def _build_wall() -> FixedSurface:
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
        surface_id="level:0/wall:1:2",
        surface_type=SURFACE_TYPE_WALL,
        level_index=0,
        room_index=None,
        mesh=mesh,
        area_square_meters=float(mesh.area),
        wall_key="1:2",
        wall_start_world=(0.0, 0.0, 0.0),
        wall_end_world=(4.0, 0.0, 0.0),
        wall_height_meters=3.0,
    )


def _build_slope_surface(
    slope_id: str,
    wall: FixedSurface,
    *,
    x_offset: float,
) -> FixedSurface:
    vertices = np.asarray(
        (
            (x_offset, 0.0, 2.4),
            (x_offset + 1.0, 0.0, 2.4),
            (x_offset + 1.0, 1.0, 3.0),
            (x_offset, 1.0, 3.0),
        ),
        dtype=float,
    )
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        process=False,
    )
    return FixedSurface(
        surface_id=f"level:0/sloped-ceiling:{slope_id}:ceiling",
        surface_type=SURFACE_TYPE_CEILING,
        level_index=0,
        room_index=None,
        mesh=mesh,
        area_square_meters=float(mesh.area),
        sloped_ceiling_id=slope_id,
        sloped_ceiling_wall_surface_id=wall.surface_id,
        sloped_ceiling_host_surface_id="level:0/ceiling",
    )


def _build_slope_target(
    slope_id: str,
    wall: FixedSurface,
    *,
    x_offset: float,
) -> SlopedCeilingEditTarget:
    handle = SlopedCeilingEditHandle(
        handle_kind=SLOPED_CEILING_HANDLE_DEPTH,
        origin_world=(x_offset + 0.5, 1.0, 3.0),
        axis_world=(0.0, 1.0, 0.0),
        value=1.0,
        minimum_value=0.1,
        maximum_value=3.0,
        value_per_meter=1.0,
    )
    return SlopedCeilingEditTarget(
        slope_id=slope_id,
        wall_surface_id=wall.surface_id,
        level_index=0,
        handles=(handle,),
        wall_start_world=(0.0, 0.0, 0.0),
        wall_end_world=(4.0, 0.0, 0.0),
        inward_world=(0.0, 1.0, 0.0),
        ceiling_z_meters=3.0,
        selection_edge_start_world=(x_offset, 0.0, 2.4),
        selection_edge_end_world=(x_offset + 1.0, 0.0, 2.4),
        creation_start_ratio=x_offset / 4.0,
        creation_end_ratio=(x_offset + 1.0) / 4.0,
    )


def _build_slope_window_host(
    slope_id: str,
    wall: FixedSurface,
) -> SlopedCeilingWindowHost:
    """Build one exact window-placement plane for a semantic slope."""

    return SlopedCeilingWindowHost(
        slope_id=slope_id,
        wall_surface_id=wall.surface_id,
        level_index=0,
        room_index=None,
        plane_start_world=(0.0, 0.0, 2.0),
        wall_tangent_world=(1.0, 0.0, 0.0),
        wall_normal_world=(0.0, -1.0, 0.0),
        vertical_axis_world=(0.0, 0.0, 1.0),
        wall_width_meters=4.0,
        wall_height_meters=1.0,
    )


# ### Semantic slope-surface regressions ###
class SlopedCeilingSurfaceGeometryTests(unittest.TestCase):
    def test_each_slope_is_a_stable_texturable_surface_disjoint_from_ceiling(
        self,
    ) -> None:
        level = _build_room_level()
        flat_surfaces = build_base_fixed_surfaces(
            [level],
            apply_sloped_ceilings=False,
        )
        wall = _host_wall(flat_surfaces)
        host_ceiling = next(
            surface
            for surface in flat_surfaces
            if surface.surface_type == SURFACE_TYPE_CEILING
        )
        level.sloped_ceilings = [
            SlopedCeilingData(
                slope_id=FIRST_SLOPE_ID,
                wall_surface_id=wall.surface_id,
                depth_meters=1.0,
                wall_drop_meters=0.5,
                transition_radius_meters=0.0,
                wall_transition_radius_meters=0.0,
                side_transition_radius_meters=0.0,
                start_ratio=0.05,
                end_ratio=0.35,
            ),
            SlopedCeilingData(
                slope_id=SECOND_SLOPE_ID,
                wall_surface_id=wall.surface_id,
                depth_meters=1.0,
                wall_drop_meters=0.7,
                transition_radius_meters=0.0,
                wall_transition_radius_meters=0.0,
                side_transition_radius_meters=0.0,
                start_ratio=0.65,
                end_ratio=0.95,
            ),
        ]

        first_build = apply_sloped_ceiling_geometry([level], flat_surfaces)
        slope_surfaces = {
            (surface.sloped_ceiling_id, surface.sloped_ceiling_part): surface
            for surface in first_build
            if surface.sloped_ceiling_id is not None
        }

        self.assertEqual(
            {slope_id for slope_id, _part in slope_surfaces},
            {FIRST_SLOPE_ID, SECOND_SLOPE_ID},
        )
        self.assertIn((FIRST_SLOPE_ID, "main"), slope_surfaces)
        self.assertIn((FIRST_SLOPE_ID, "start-return"), slope_surfaces)
        self.assertIn((FIRST_SLOPE_ID, "end-return"), slope_surfaces)
        self.assertIn((SECOND_SLOPE_ID, "main"), slope_surfaces)
        for (slope_id, part), surface in slope_surfaces.items():
            with self.subTest(slope_id=slope_id, part=part):
                self.assertEqual(surface.surface_type, SURFACE_TYPE_CEILING)
                self.assertEqual(
                    surface.sloped_ceiling_wall_surface_id, wall.surface_id
                )
                self.assertEqual(
                    surface.sloped_ceiling_host_surface_id,
                    host_ceiling.surface_id,
                )
                self.assertEqual(
                    normalize_surface_id(surface.surface_id), surface.surface_id
                )
                self.assertEqual(
                    get_surface_type_for_id(surface.surface_id),
                    SURFACE_TYPE_CEILING,
                )

        residual_ceiling_coverage = shapely.union_all(
            [
                _projected_mesh_coverage(surface.mesh)
                for surface in first_build
                if surface.surface_type == SURFACE_TYPE_CEILING
                and surface.sloped_ceiling_id is None
            ]
        )
        slope_coverages = {
            slope_id: shapely.union_all(
                [
                    _projected_mesh_coverage(surface.mesh)
                    for (candidate_id, _part), surface in slope_surfaces.items()
                    if candidate_id == slope_id
                ]
            )
            for slope_id in (FIRST_SLOPE_ID, SECOND_SLOPE_ID)
        }
        for slope_id, coverage in slope_coverages.items():
            with self.subTest(slope_id=slope_id):
                self.assertAlmostEqual(
                    float(coverage.intersection(residual_ceiling_coverage).area),
                    0.0,
                    places=8,
                )
        self.assertAlmostEqual(
            float(
                slope_coverages[FIRST_SLOPE_ID]
                .intersection(slope_coverages[SECOND_SLOPE_ID])
                .area
            ),
            0.0,
            places=8,
        )

        second_build = apply_sloped_ceiling_geometry([level], first_build)
        rebuilt_slope_surfaces = {
            (surface.sloped_ceiling_id, surface.sloped_ceiling_part): surface
            for surface in second_build
            if surface.sloped_ceiling_id is not None
        }
        self.assertEqual(
            {
                key: surface.surface_id
                for key, surface in rebuilt_slope_surfaces.items()
            },
            {
                key: surface.surface_id
                for key, surface in slope_surfaces.items()
            },
        )
        for key, surface in slope_surfaces.items():
            with self.subTest(reapplied_slope_part=key):
                self.assertEqual(
                    _canonical_triangles(rebuilt_slope_surfaces[key].mesh),
                    _canonical_triangles(surface.mesh),
                )


# ### Viewer selection and preview regressions ###
class SlopedCeilingSurfaceViewerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.widgets: list[GlbViewerWidget] = []

    def tearDown(self) -> None:
        for widget in self.widgets:
            widget.close()
            widget.deleteLater()
        _qt_application.processEvents()

    def test_selecting_one_slope_surface_keeps_it_selected_and_shows_only_its_gizmos(
        self,
    ) -> None:
        wall = _build_wall()
        first_surface = _build_slope_surface(FIRST_SLOPE_ID, wall, x_offset=0.25)
        second_surface = _build_slope_surface(SECOND_SLOPE_ID, wall, x_offset=2.75)
        first_target = _build_slope_target(FIRST_SLOPE_ID, wall, x_offset=0.25)
        second_target = _build_slope_target(SECOND_SLOPE_ID, wall, x_offset=2.75)
        viewer = GlbViewerWidget(window_editing_enabled=True)
        viewer.set_wall_targets((wall, first_surface, second_surface))
        viewer.set_sloped_ceiling_edit_targets((first_target, second_target))
        self.widgets.append(viewer)

        with patch.object(viewer.view, "pixelSize", return_value=0.01):
            self.assertTrue(
                viewer.set_selected_canvas_surface_ids((second_surface.surface_id,))
            )
            viewer._refresh_sloped_ceiling_gizmo_items()

        self.assertEqual(
            viewer.get_selected_canvas_surface_ids(),
            (second_surface.surface_id,),
        )
        self.assertEqual(
            viewer.get_active_canvas_surface_id(),
            second_surface.surface_id,
        )
        self.assertEqual(
            viewer.get_selected_sloped_ceiling_wall_surface_id(),
            wall.surface_id,
        )
        self.assertEqual(
            viewer._get_displayed_sloped_ceiling_targets(),
            (second_target,),
        )
        self.assertEqual(len(viewer._sloped_ceiling_gizmo_sizes), 1)
        self.assertEqual(len(viewer._sloped_ceiling_gizmo_items), 3)

    def test_mixed_root_wall_and_slope_selection_uses_per_wall_scope(self) -> None:
        first_wall = _build_wall()
        second_wall = replace(
            _build_wall(),
            surface_id="level:0/wall:3:4",
            wall_key="3:4",
        )
        first_surface = _build_slope_surface(
            FIRST_SLOPE_ID,
            first_wall,
            x_offset=0.25,
        )
        targets = (
            _build_slope_target(FIRST_SLOPE_ID, first_wall, x_offset=0.25),
            _build_slope_target(SECOND_SLOPE_ID, first_wall, x_offset=2.75),
            _build_slope_target(THIRD_SLOPE_ID, second_wall, x_offset=0.25),
            _build_slope_target(FOURTH_SLOPE_ID, second_wall, x_offset=2.75),
        )
        viewer = GlbViewerWidget(window_editing_enabled=True)
        viewer.set_wall_targets((first_wall, second_wall, first_surface))
        viewer.set_sloped_ceiling_edit_targets(targets)
        self.widgets.append(viewer)

        viewer.set_selected_canvas_surface_ids(
            (first_surface.surface_id, second_wall.surface_id)
        )

        self.assertEqual(
            viewer._get_selected_sloped_ceiling_targets(),
            (targets[0], targets[2], targets[3]),
        )

    def test_selecting_one_joined_slope_surface_keeps_one_shared_gizmo_set(
        self,
    ) -> None:
        first_wall = _build_wall()
        second_wall = replace(
            _build_wall(),
            surface_id="level:0/wall:3:4",
            wall_key="3:4",
        )
        first_surface = _build_slope_surface(
            FIRST_SLOPE_ID,
            first_wall,
            x_offset=0.25,
        )
        second_surface = _build_slope_surface(
            SECOND_SLOPE_ID,
            second_wall,
            x_offset=0.25,
        )
        joined_group_id = "e" * 32
        first_target = replace(
            _build_slope_target(FIRST_SLOPE_ID, first_wall, x_offset=0.25),
            joined_group_id=joined_group_id,
        )
        second_target = replace(
            _build_slope_target(SECOND_SLOPE_ID, second_wall, x_offset=0.25),
            joined_group_id=joined_group_id,
        )
        viewer = GlbViewerWidget(window_editing_enabled=True)
        viewer.set_wall_targets(
            (first_wall, second_wall, first_surface, second_surface)
        )
        viewer.set_sloped_ceiling_edit_targets((first_target, second_target))
        self.widgets.append(viewer)

        viewer.set_selected_canvas_surface_ids((first_surface.surface_id,))

        self.assertEqual(
            viewer._get_selected_sloped_ceiling_targets(),
            (first_target, second_target),
        )
        self.assertEqual(
            set(viewer.get_selected_sloped_ceiling_wall_surface_ids()),
            {first_wall.surface_id, second_wall.surface_id},
        )
        self.assertEqual(
            viewer._get_displayed_sloped_ceiling_targets(),
            (first_target,),
        )

    def test_active_slope_surface_is_the_primary_gizmo_target(self) -> None:
        wall = _build_wall()
        first_surface = _build_slope_surface(FIRST_SLOPE_ID, wall, x_offset=0.25)
        second_surface = _build_slope_surface(SECOND_SLOPE_ID, wall, x_offset=2.75)
        first_target = _build_slope_target(FIRST_SLOPE_ID, wall, x_offset=0.25)
        second_target = _build_slope_target(SECOND_SLOPE_ID, wall, x_offset=2.75)
        viewer = GlbViewerWidget(window_editing_enabled=True)
        viewer.set_wall_targets((wall, first_surface, second_surface))
        viewer.set_sloped_ceiling_edit_targets((first_target, second_target))
        self.widgets.append(viewer)

        viewer.set_selected_canvas_surface_ids(
            (first_surface.surface_id, second_surface.surface_id)
        )

        self.assertEqual(viewer.get_active_canvas_surface_id(), second_surface.surface_id)
        self.assertIs(viewer._get_selected_sloped_ceiling_target(), second_target)

    def test_shift_clicking_slope_hosts_preserves_multi_selection_and_active_host(
        self,
    ) -> None:
        wall = _build_wall()
        first_surface = _build_slope_surface(FIRST_SLOPE_ID, wall, x_offset=0.25)
        second_surface = _build_slope_surface(SECOND_SLOPE_ID, wall, x_offset=2.75)
        first_host = _build_slope_window_host(FIRST_SLOPE_ID, wall)
        second_host = _build_slope_window_host(SECOND_SLOPE_ID, wall)
        viewer = GlbViewerWidget(window_editing_enabled=True)
        viewer.set_wall_targets((wall, first_surface, second_surface))
        viewer.set_sloped_ceiling_window_hosts((first_host, second_host))
        self.widgets.append(viewer)

        with (
            patch.object(
                viewer.view,
                "build_camera_ray",
                return_value=(
                    np.asarray((0.0, -1.0, 2.5), dtype=float),
                    np.asarray((0.0, 1.0, 0.0), dtype=float),
                ),
            ),
            patch.object(viewer, "_pick_directional_light_id", return_value=None),
            patch(
                "housemaker.viewer._get_nearest_sloped_ceiling_window_host_ray_hit",
                side_effect=(
                    (first_host, np.zeros(3, dtype=float), 1.0),
                    (second_host, np.zeros(3, dtype=float), 1.0),
                    (second_host, np.zeros(3, dtype=float), 1.0),
                ),
            ),
            patch(
                "housemaker.viewer._get_nearest_fixed_surface_ray_hit",
                return_value=None,
            ),
        ):
            viewer._handle_window_wall_pick_requested(QPointF(), additive=False)
            viewer._handle_window_wall_pick_requested(QPointF(), additive=True)
            self.assertEqual(
                viewer.get_selected_canvas_surface_ids(),
                (first_surface.surface_id, second_surface.surface_id),
            )
            self.assertEqual(
                viewer.get_selected_sloped_ceiling_window_host_id(),
                second_host.target_id,
            )

            viewer._handle_window_wall_pick_requested(QPointF(), additive=True)

        self.assertEqual(
            viewer.get_selected_canvas_surface_ids(),
            (first_surface.surface_id,),
        )
        self.assertEqual(
            viewer.get_selected_sloped_ceiling_window_host_id(),
            first_host.target_id,
        )

    def test_slope_preview_renders_through_scene_meshes(self) -> None:
        viewer = GlbViewerWidget(window_editing_enabled=True)
        self.widgets.append(viewer)
        preview_mesh = trimesh.creation.box(extents=(1.0, 1.0, 0.1))

        viewer.set_sloped_ceiling_preview_meshes((preview_mesh,))

        self.assertEqual(len(viewer._sloped_ceiling_preview_items), 1)
        item = viewer._sloped_ceiling_preview_items[0]
        gl_options = item._GLGraphicsItem__glOpts
        self.assertFalse(gl_options[GL.GL_DEPTH_TEST])
        self.assertTrue(gl_options[GL.GL_BLEND])
        self.assertEqual(
            item.depthValue(),
            CANVAS_OPENING_OVERLAY_DEPTH_VALUE + 1.0,
        )


if __name__ == "__main__":
    unittest.main()
