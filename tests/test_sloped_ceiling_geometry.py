# ### Imports ###
from __future__ import annotations

import math
import unittest
from dataclasses import replace
from io import BytesIO
from itertools import combinations

import numpy as np
import shapely
import trimesh
from PIL import Image
from shapely import Point, Polygon

import housemaker.sloped_ceiling as sloped_ceiling_module
from housemaker.glb import convert_to_glb, convert_to_preview_model
from housemaker.models import (
    DEFAULT_SLOPED_CEILING_WALL_DROP_METERS,
    DoorwayData,
    LevelData,
    RoomData,
    SlopedCeilingData,
    VertexData,
    WindowData,
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
    SlopedCeilingEdit,
    SlopedCeilingEditHandle,
    SlopedCeilingEditTarget,
    SlopedCeilingWindowPlacement,
    add_sloped_ceiling_window,
    apply_sloped_ceiling_edit,
    build_sloped_ceiling_surface_id,
    build_sloped_ceiling_edit_targets,
    build_sloped_ceiling_preview_meshes,
    build_sloped_ceiling_window_hosts,
    build_sloped_ceiling_window_opening_targets,
    build_sloped_ceiling_window_placement,
    get_sloped_ceiling_window_world_corners,
)
from housemaker.surface_geometry import (
    FixedSurface,
    build_base_fixed_surfaces,
    build_fixed_surfaces,
)

# ### Constants ###
SLOPE_ID = "a" * 32
CEILING_CLEARANCE_METERS = 0.001


# ### Fixture helpers ###
def _build_room_level() -> LevelData:
    vertex_data = VertexData()
    boundary_ids = tuple(
        vertex_data.add_vertex(*point).id
        for point in (
            (0.0, 0.0),
            (200.0, 0.0),
            (200.0, 200.0),
            (0.0, 200.0),
        )
    )
    for start_id, end_id in zip(
        boundary_ids,
        (*boundary_ids[1:], boundary_ids[0]),
        strict=True,
    ):
        vertex_data.add_edge(start_id, end_id)
    center = vertex_data.add_vertex(100.0, 100.0)
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


def _build_level_with_internal_wall() -> LevelData:
    """Build one residual ceiling that extends across both sides of a wall."""

    vertex_data = VertexData()
    boundary_ids = tuple(
        vertex_data.add_vertex(*point).id
        for point in (
            (0.0, 0.0),
            (200.0, 0.0),
            (200.0, 200.0),
            (0.0, 200.0),
        )
    )
    for start_id, end_id in zip(
        boundary_ids,
        (*boundary_ids[1:], boundary_ids[0]),
        strict=True,
    ):
        vertex_data.add_edge(start_id, end_id)
    internal_start = vertex_data.add_vertex(100.0, 50.0)
    internal_end = vertex_data.add_vertex(100.0, 150.0)
    vertex_data.add_edge(internal_start.id, internal_end.id)
    return LevelData(
        index=2,
        name="Ground",
        height_meters=3.0,
        floor_thickness_meters=0.3,
        vertex_data=vertex_data,
    )


def _build_room_level_with_spanning_inner_wall() -> LevelData:
    """Build a room with one inner wall under a joined slope side."""

    level = _build_room_level()
    wall_start = level.vertex_data.add_vertex(100.0, 0.0)
    wall_end = level.vertex_data.add_vertex(100.0, 200.0)
    level.vertex_data.add_edge(wall_start.id, wall_end.id)
    return level


def _build_room_level_with_partial_inner_wall() -> LevelData:
    """Build a room where an inner wall owns half of one slope side."""

    level = _build_room_level()
    wall_start = level.vertex_data.add_vertex(20.0, 0.0)
    wall_end = level.vertex_data.add_vertex(20.0, 100.0)
    level.vertex_data.add_edge(wall_start.id, wall_end.id)
    return level


def _build_room_level_with_crossing_inner_wall() -> LevelData:
    """Build a room with a wall that crosses each slope side at one point."""

    level = _build_room_level()
    wall_start = level.vertex_data.add_vertex(0.0, 100.0)
    wall_end = level.vertex_data.add_vertex(200.0, 100.0)
    level.vertex_data.add_edge(wall_start.id, wall_end.id)
    return level


def _surface_by_id(surfaces: list[FixedSurface]) -> dict[str, FixedSurface]:
    return {surface.surface_id: surface for surface in surfaces}


def _combined_ceiling_surface(
    surfaces: list[FixedSurface] | tuple[FixedSurface, ...],
    *,
    host_surface_id: str | None = None,
    room_index: int | None = None,
) -> FixedSurface:
    """Reassemble one ceiling host from its residual and slope parts."""

    if host_surface_id is None:
        host = next(
            (
                surface
                for surface in surfaces
                if surface.surface_type == "ceiling"
                and surface.sloped_ceiling_id is None
                and (room_index is None or surface.room_index == room_index)
            ),
            None,
        )
        if host is None:
            host = next(
                surface
                for surface in surfaces
                if surface.surface_type == "ceiling"
                and surface.sloped_ceiling_id is not None
                and (room_index is None or surface.room_index == room_index)
            )
            host_surface_id = host.sloped_ceiling_host_surface_id
        else:
            host_surface_id = host.surface_id
    else:
        host = next(
            surface
            for surface in surfaces
            if surface.surface_id == host_surface_id
        )
    parts = [
        surface
        for surface in surfaces
        if surface.surface_id == host_surface_id
        or surface.sloped_ceiling_host_surface_id == host_surface_id
    ]
    combined_mesh = trimesh.util.concatenate(
        tuple(surface.mesh for surface in parts)
    )
    return replace(
        host,
        mesh=combined_mesh,
        area_square_meters=float(combined_mesh.area),
    )


def _sloped_ceiling_surface(
    surfaces: list[FixedSurface] | tuple[FixedSurface, ...],
    slope_id: str = SLOPE_ID,
    part: str = "main",
) -> FixedSurface:
    """Return one exact stable semantic part authored for a slope."""

    return next(
        surface
        for surface in surfaces
        if surface.sloped_ceiling_id == slope_id
        and surface.sloped_ceiling_part == part
    )


def _host_wall(flat_surfaces: list[FixedSurface]) -> FixedSurface:
    return next(
        surface
        for surface in flat_surfaces
        if surface.surface_type == "wall"
        and np.allclose(surface.wall_start_world[1], 0.0)
        and np.allclose(surface.wall_end_world[1], 0.0)
    )


def _solid_png() -> bytes:
    output = BytesIO()
    Image.new("RGBA", (8, 8), (170, 190, 210, 255)).save(
        output,
        format="PNG",
    )
    return output.getvalue()


def _add_slope(level: LevelData, wall_surface_id: str) -> None:
    level.sloped_ceilings.append(
        SlopedCeilingData(
            slope_id=SLOPE_ID,
            wall_surface_id=wall_surface_id,
            depth_meters=1.0,
            angle_degrees=30.0,
            transition_radius_meters=0.2,
            start_ratio=0.2,
            end_ratio=0.8,
        )
    )


def _target_handle(
    target: SlopedCeilingEditTarget,
    handle_kind: str,
) -> SlopedCeilingEditHandle:
    """Return one uniquely named handle from a generated edit target."""

    return next(
        handle
        for handle in target.handles
        if handle.handle_kind == handle_kind
    )


def _ceiling_height_at_xy(
    ceiling: FixedSurface,
    point_xy: tuple[float, float],
) -> float:
    """Interpolate the lowest non-vertical ceiling skin above one XY point."""

    point = np.asarray(point_xy, dtype=float)
    heights: list[float] = []
    for triangle in np.asarray(ceiling.mesh.triangles, dtype=float):
        first = triangle[1, :2] - triangle[0, :2]
        second = triangle[2, :2] - triangle[0, :2]
        matrix = np.column_stack((first, second))
        determinant = float(np.linalg.det(matrix))
        if abs(determinant) <= 1e-10:
            continue
        first_weight, second_weight = np.linalg.solve(
            matrix,
            point - triangle[0, :2],
        )
        origin_weight = 1.0 - first_weight - second_weight
        if min(origin_weight, first_weight, second_weight) < -1e-8:
            continue
        heights.append(
            float(
                origin_weight * triangle[0, 2]
                + first_weight * triangle[1, 2]
                + second_weight * triangle[2, 2]
            )
        )
    if not heights:
        raise AssertionError(f"No ceiling triangle covers XY point {point_xy!r}.")
    return min(heights)


def _assert_no_internal_ceiling_boundary_edges(
    test_case: unittest.TestCase,
    ceiling: FixedSurface,
) -> None:
    """Require a consistently wound manifold ceiling away from its perimeter."""

    vertices = np.asarray(ceiling.mesh.vertices, dtype=float)
    faces = np.asarray(ceiling.mesh.faces, dtype=np.int64)
    edge_counts: dict[
        tuple[tuple[float, float, float], tuple[float, float, float]],
        int,
    ] = {}
    for face in faces:
        for first_index, second_index in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            endpoints = tuple(
                sorted(
                    (
                        tuple(np.round(vertices[first_index], 8)),
                        tuple(np.round(vertices[second_index], 8)),
                    )
                )
            )
            edge_counts[endpoints] = edge_counts.get(endpoints, 0) + 1

    test_case.assertTrue(ceiling.mesh.is_winding_consistent)
    bounds = np.asarray(ceiling.mesh.bounds, dtype=float)
    for edge, count in edge_counts.items():
        test_case.assertLessEqual(
            count,
            2,
            f"Non-manifold ceiling edge has {count} incident faces: {edge!r}",
        )
        if count != 1:
            continue
        first, second = np.asarray(edge, dtype=float)
        lies_on_perimeter = any(
            np.isclose(first[axis], bound, atol=1e-7)
            and np.isclose(second[axis], bound, atol=1e-7)
            for axis in (0, 1)
            for bound in (bounds[0, axis], bounds[1, axis])
        )
        test_case.assertTrue(
            lies_on_perimeter,
            f"Unexpected open ceiling edge inside the room: {edge!r}",
        )


def _coplanar_mesh_overlap_area(
    first_mesh: trimesh.Trimesh,
    second_mesh: trimesh.Trimesh,
) -> float:
    """Measure shared area for two meshes lying on one common plane."""

    first_triangles = np.asarray(first_mesh.triangles, dtype=float)
    second_triangles = np.asarray(second_mesh.triangles, dtype=float)
    if not len(first_triangles) or not len(second_triangles):
        return 0.0
    first_normal = np.array(first_mesh.face_normals[0], dtype=float)
    second_normal = np.array(second_mesh.face_normals[0], dtype=float)
    first_normal /= np.linalg.norm(first_normal)
    second_normal /= np.linalg.norm(second_normal)
    alignment = float(np.dot(first_normal, second_normal))
    if not np.isclose(abs(alignment), 1.0, atol=1e-8):
        return 0.0
    if alignment < 0.0:
        second_normal = -second_normal
    first_offset = float(np.dot(first_normal, first_triangles[0, 0]))
    second_offset = float(np.dot(second_normal, second_triangles[0, 0]))
    if not np.isclose(first_offset, second_offset, atol=1e-8):
        return 0.0

    removed_axis = int(np.argmax(np.abs(first_normal)))
    retained_axes = tuple(axis for axis in range(3) if axis != removed_axis)
    first_projection = shapely.union_all(
        [Polygon(triangle[:, retained_axes]) for triangle in first_triangles]
    )
    second_projection = shapely.union_all(
        [Polygon(triangle[:, retained_axes]) for triangle in second_triangles]
    )
    projected_overlap = first_projection.intersection(second_projection).area
    return float(projected_overlap / abs(first_normal[removed_axis]))


def _constant_plane_key(mesh: trimesh.Trimesh) -> tuple[int, float]:
    """Return the constant world-axis plane occupied by one side return."""

    vertices = np.asarray(mesh.vertices, dtype=float)
    spans = np.ptp(vertices, axis=0)
    constant_axes = np.flatnonzero(spans <= 1e-8)
    if len(constant_axes) != 1:
        raise AssertionError(
            "A joined slope side must occupy exactly one axis-aligned plane."
        )
    axis = int(constant_axes[0])
    return axis, round(float(np.mean(vertices[:, axis])), 8)


def _canonical_mesh_triangles(mesh: trimesh.Trimesh) -> tuple[object, ...]:
    """Return a face-order-independent signature for stable rebuilt geometry."""

    return tuple(
        sorted(
            tuple(
                sorted(
                    tuple(round(float(value), 8) for value in point)
                    for point in triangle
                )
            )
            for triangle in np.asarray(mesh.triangles, dtype=float)
        )
    )


# ### Semantic geometry tests ###
class SlopedCeilingSemanticGeometryTests(unittest.TestCase):
    def test_joined_side_returns_are_two_stable_wall_to_wall_surfaces(
        self,
    ) -> None:
        level = _build_room_level()
        flat_surfaces = build_base_fixed_surfaces(
            [level],
            apply_sloped_ceilings=False,
        )
        first_wall = _host_wall(flat_surfaces)
        second_wall = next(
            surface
            for surface in flat_surfaces
            if surface.surface_type == "wall"
            and np.allclose(surface.wall_start_world[1], -4.0)
            and np.allclose(surface.wall_end_world[1], -4.0)
        )
        first_slope_id = "1" * 32
        second_slope_id = "2" * 32
        joined_group_id = "3" * 32
        level.sloped_ceilings = [
            SlopedCeilingData(
                slope_id=slope_id,
                wall_surface_id=wall.surface_id,
                joined_group_id=joined_group_id,
                depth_meters=2.5,
                wall_drop_meters=0.6,
                transition_radius_meters=0.1,
                wall_transition_radius_meters=0.1,
                side_transition_radius_meters=0.1,
                start_ratio=0.1,
                end_ratio=0.9,
            )
            for slope_id, wall in (
                (first_slope_id, first_wall),
                (second_slope_id, second_wall),
            )
        ]

        def build_side_returns() -> dict[tuple[int, float], FixedSurface]:
            return {
                _constant_plane_key(surface.mesh): surface
                for surface in build_fixed_surfaces([level])
                if surface.sloped_ceiling_part
                in {"start-return", "end-return"}
            }

        side_returns = build_side_returns()
        self.assertEqual(set(side_returns), {(0, 0.4), (0, 3.6)})
        self.assertEqual(len(side_returns), 2)
        expected_aliases_by_plane = {
            (0, 0.4): (
                build_sloped_ceiling_surface_id(
                    level.index,
                    second_slope_id,
                    "end-return",
                ),
            ),
            (0, 3.6): (
                build_sloped_ceiling_surface_id(
                    level.index,
                    second_slope_id,
                    "start-return",
                ),
            ),
        }
        for surface in side_returns.values():
            self.assertGreater(float(surface.mesh.area), 0.0)
            constant_axis = _constant_plane_key(surface.mesh)[0]
            retained_axes = tuple(
                axis for axis in range(3) if axis != constant_axis
            )
            projected_side = shapely.union_all(
                [
                    Polygon(triangle[:, retained_axes])
                    for triangle in np.asarray(
                        surface.mesh.triangles,
                        dtype=float,
                    )
                ]
            )
            self.assertIsInstance(projected_side, Polygon)
            bounds = np.asarray(surface.mesh.bounds, dtype=float)
            self.assertAlmostEqual(float(bounds[0, 1]), -4.0, places=8)
            self.assertAlmostEqual(float(bounds[1, 1]), 0.0, places=8)
            triangle_centers = np.mean(
                np.asarray(surface.mesh.triangles, dtype=float),
                axis=1,
            )
            self.assertTrue(np.any(triangle_centers[:, 1] < -2.0))
            self.assertTrue(np.any(triangle_centers[:, 1] > -2.0))
            self.assertEqual(
                surface.surface_alias_ids,
                expected_aliases_by_plane[_constant_plane_key(surface.mesh)],
            )
        for first, second in combinations(side_returns.values(), 2):
            self.assertAlmostEqual(
                _coplanar_mesh_overlap_area(first.mesh, second.mesh),
                0.0,
                places=10,
            )

        first_build_signatures = {
            key: (
                surface.surface_id,
                surface.surface_alias_ids,
                _canonical_mesh_triangles(surface.mesh),
            )
            for key, surface in side_returns.items()
        }
        level.sloped_ceilings.reverse()
        rebuilt_side_returns = build_side_returns()
        self.assertEqual(set(rebuilt_side_returns), set(side_returns))
        self.assertEqual(
            {
                key: (
                    surface.surface_id,
                    surface.surface_alias_ids,
                    _canonical_mesh_triangles(surface.mesh),
                )
                for key, surface in rebuilt_side_returns.items()
            },
            first_build_signatures,
        )

    def test_inner_wall_owns_coincident_joined_slope_side_coverage(
        self,
    ) -> None:
        level = _build_room_level_with_spanning_inner_wall()
        flat_surfaces = build_base_fixed_surfaces(
            [level],
            apply_sloped_ceilings=False,
        )
        first_wall = _host_wall(flat_surfaces)
        second_wall = next(
            surface
            for surface in flat_surfaces
            if surface.surface_type == "wall"
            and np.allclose(surface.wall_start_world[1], -4.0)
            and np.allclose(surface.wall_end_world[1], -4.0)
        )
        joined_group_id = "6" * 32
        level.sloped_ceilings = [
            SlopedCeilingData(
                slope_id=slope_id,
                wall_surface_id=wall.surface_id,
                joined_group_id=joined_group_id,
                depth_meters=2.5,
                wall_drop_meters=0.6,
                transition_radius_meters=0.1,
                wall_transition_radius_meters=0.1,
                side_transition_radius_meters=0.1,
                start_ratio=0.5,
                end_ratio=0.9,
            )
            for slope_id, wall in (
                ("4" * 32, first_wall),
                ("5" * 32, second_wall),
            )
        ]

        surfaces = build_fixed_surfaces([level])
        inner_wall = next(
            surface
            for surface in surfaces
            if surface.surface_type == "wall"
            and np.allclose(surface.wall_start_world[0], 2.0)
            and np.allclose(surface.wall_end_world[0], 2.0)
        )
        side_returns = tuple(
            surface
            for surface in surfaces
            if surface.sloped_ceiling_part
            in {"start-return", "end-return"}
        )

        self.assertFalse(
            any(
                _constant_plane_key(surface.mesh) == (0, 2.0)
                for surface in side_returns
            )
        )
        for surface in side_returns:
            with self.subTest(surface_id=surface.surface_id):
                self.assertAlmostEqual(
                    _coplanar_mesh_overlap_area(inner_wall.mesh, surface.mesh),
                    0.0,
                    places=10,
                )

    def test_perpendicular_wall_crossing_keeps_joined_side_returns(self) -> None:
        def build_side_returns(
            level: LevelData,
        ) -> dict[tuple[int, float], FixedSurface]:
            flat_surfaces = build_base_fixed_surfaces(
                [level],
                apply_sloped_ceilings=False,
            )
            first_wall = _host_wall(flat_surfaces)
            second_wall = next(
                surface
                for surface in flat_surfaces
                if surface.surface_type == "wall"
                and np.allclose(surface.wall_start_world[1], -4.0)
                and np.allclose(surface.wall_end_world[1], -4.0)
            )
            level.sloped_ceilings = [
                SlopedCeilingData(
                    slope_id=slope_id,
                    wall_surface_id=wall.surface_id,
                    joined_group_id="9" * 32,
                    depth_meters=2.5,
                    wall_drop_meters=0.6,
                    transition_radius_meters=0.1,
                    wall_transition_radius_meters=0.1,
                    side_transition_radius_meters=0.1,
                    start_ratio=0.1,
                    end_ratio=0.9,
                )
                for slope_id, wall in (
                    ("7" * 32, first_wall),
                    ("8" * 32, second_wall),
                )
            ]
            return {
                _constant_plane_key(surface.mesh): surface
                for surface in build_fixed_surfaces([level])
                if surface.sloped_ceiling_part
                in {"start-return", "end-return"}
            }

        baseline = build_side_returns(_build_room_level())
        crossed = build_side_returns(_build_room_level_with_crossing_inner_wall())

        self.assertEqual(set(crossed), set(baseline))
        for key, expected in baseline.items():
            with self.subTest(side_plane=key):
                actual = crossed[key]
                np.testing.assert_allclose(actual.mesh.bounds, expected.mesh.bounds)
                self.assertAlmostEqual(
                    actual.mesh.area,
                    expected.mesh.area,
                    places=10,
                )
                self.assertEqual(
                    _canonical_mesh_triangles(actual.mesh),
                    _canonical_mesh_triangles(expected.mesh),
                )

    def test_partial_inner_wall_owns_only_its_joined_side_interval(self) -> None:
        level = _build_room_level_with_partial_inner_wall()
        flat_surfaces = build_base_fixed_surfaces(
            [level],
            apply_sloped_ceilings=False,
        )
        first_wall = _host_wall(flat_surfaces)
        second_wall = next(
            surface
            for surface in flat_surfaces
            if surface.surface_type == "wall"
            and np.allclose(surface.wall_start_world[1], -4.0)
            and np.allclose(surface.wall_end_world[1], -4.0)
        )
        level.sloped_ceilings = [
            SlopedCeilingData(
                slope_id=slope_id,
                wall_surface_id=wall.surface_id,
                joined_group_id="c" * 32,
                depth_meters=2.5,
                wall_drop_meters=0.6,
                transition_radius_meters=0.1,
                wall_transition_radius_meters=0.1,
                side_transition_radius_meters=0.1,
                start_ratio=0.1,
                end_ratio=0.9,
            )
            for slope_id, wall in (
                ("a" * 32, first_wall),
                ("b" * 32, second_wall),
            )
        ]

        surfaces = build_fixed_surfaces([level])
        partial_wall = next(
            surface
            for surface in surfaces
            if surface.surface_type == "wall"
            and np.allclose(surface.wall_start_world[0], 0.4)
            and np.allclose(surface.wall_end_world[0], 0.4)
        )
        exposed_side = next(
            surface
            for surface in surfaces
            if surface.sloped_ceiling_part
            in {"start-return", "end-return"}
            and _constant_plane_key(surface.mesh) == (0, 0.4)
        )

        self.assertAlmostEqual(
            _coplanar_mesh_overlap_area(partial_wall.mesh, exposed_side.mesh),
            0.0,
            places=10,
        )
        bounds = np.asarray(exposed_side.mesh.bounds, dtype=float)
        self.assertAlmostEqual(float(bounds[0, 1]), -4.0, places=8)
        self.assertAlmostEqual(float(bounds[1, 1]), -2.0, places=8)

    def test_interior_wall_slope_keeps_the_opposite_ceiling_flat_and_sealed(
        self,
    ) -> None:
        level = _build_level_with_internal_wall()
        flat_surfaces = build_base_fixed_surfaces(
            [level],
            apply_sloped_ceilings=False,
        )
        host_wall = next(
            surface
            for surface in flat_surfaces
            if surface.surface_type == "wall"
            and np.allclose(surface.wall_start_world[0], 2.0)
            and np.allclose(surface.wall_end_world[0], 2.0)
            and np.isclose(
                abs(surface.wall_end_world[1] - surface.wall_start_world[1]),
                2.0,
            )
        )
        level.sloped_ceilings = [
            SlopedCeilingData(
                slope_id=SLOPE_ID,
                wall_surface_id=host_wall.surface_id,
                depth_meters=0.8,
                wall_drop_meters=0.5,
                side_transition_radius_meters=0.2,
                start_ratio=0.2,
                end_ratio=0.8,
            )
        ]

        frame = sloped_ceiling_module._build_slope_frames(
            [level],
            flat_surfaces,
        )[0]
        sloped_surfaces = build_base_fixed_surfaces([level])
        ceiling = _combined_ceiling_surface(
            sloped_surfaces,
            room_index=None,
        )
        nominal_ceiling_z = (
            level.floor_thickness_meters
            + level.height_meters
            - CEILING_CLEARANCE_METERS
        )
        wall_midpoint = (frame.wall_start[:2] + frame.wall_end[:2]) * 0.5
        for outward_distance in (0.05, 0.5, 1.0):
            point = wall_midpoint - frame.inward_xy * outward_distance
            with self.subTest(outward_distance=outward_distance):
                self.assertAlmostEqual(
                    _ceiling_height_at_xy(ceiling, tuple(point)),
                    nominal_ceiling_z,
                    places=7,
                )

        inward_point = wall_midpoint + frame.inward_xy * 0.1
        self.assertLess(
            _ceiling_height_at_xy(ceiling, tuple(inward_point)),
            nominal_ceiling_z,
        )

        vertical_returns = []
        for triangle in np.asarray(ceiling.mesh.triangles, dtype=float):
            relative_xy = triangle[:, :2] - frame.wall_start[:2]
            depth_coordinates = relative_xy @ frame.inward_xy
            tangent_coordinates = relative_xy @ frame.tangent_xy
            if (
                np.all(np.abs(depth_coordinates) <= 1e-7)
                and np.min(tangent_coordinates) >= frame.start_distance - 1e-7
                and np.max(tangent_coordinates) <= frame.end_distance + 1e-7
                and float(np.ptp(triangle[:, 2])) > 1e-7
            ):
                vertical_returns.append(triangle)
        self.assertTrue(vertical_returns)
        _assert_no_internal_ceiling_boundary_edges(self, ceiling)

    def test_slope_preserves_ids_and_wall_top_meets_ceiling(self) -> None:
        level = _build_room_level()
        flat_surfaces = build_base_fixed_surfaces(
            [level],
            apply_sloped_ceilings=False,
        )
        host_wall = _host_wall(flat_surfaces)
        ceiling_id = next(
            surface.surface_id
            for surface in flat_surfaces
            if surface.surface_type == "ceiling"
        )
        _add_slope(level, host_wall.surface_id)

        sloped_surfaces = build_base_fixed_surfaces([level])
        sloped_by_id = _surface_by_id(sloped_surfaces)

        flat_surface_ids = set(_surface_by_id(flat_surfaces))
        self.assertTrue(flat_surface_ids.issubset(sloped_by_id))
        slope_surfaces = [
            surface
            for surface in sloped_surfaces
            if surface.sloped_ceiling_id == SLOPE_ID
        ]
        self.assertEqual(
            set(sloped_by_id) - flat_surface_ids,
            {surface.surface_id for surface in slope_surfaces},
        )
        self.assertEqual(
            {surface.sloped_ceiling_part for surface in slope_surfaces},
            {"main", "start-return", "end-return"},
        )
        for semantic_part in slope_surfaces:
            self.assertEqual(semantic_part.sloped_ceiling_id, SLOPE_ID)
            self.assertEqual(
                semantic_part.sloped_ceiling_wall_surface_id,
                host_wall.surface_id,
            )
            self.assertEqual(
                semantic_part.sloped_ceiling_host_surface_id,
                ceiling_id,
            )
        ceiling = _combined_ceiling_surface(
            sloped_surfaces,
            host_surface_id=ceiling_id,
        )
        sloped_wall = sloped_by_id[host_wall.surface_id]
        self.assertLess(
            float(ceiling.mesh.bounds[0, 2]),
            float(ceiling.mesh.bounds[1, 2]),
        )

        wall_vertices = np.asarray(sloped_wall.mesh.vertices, dtype=float)
        wall_heights_by_xy: dict[tuple[float, float], list[float]] = {}
        for vertex in wall_vertices:
            key = (round(float(vertex[0]), 9), round(float(vertex[1]), 9))
            wall_heights_by_xy.setdefault(key, []).append(float(vertex[2]))
        wall_top_z_by_xy = {
            point_xy: max(heights)
            for point_xy, heights in wall_heights_by_xy.items()
        }
        ceiling_triangles = np.asarray(ceiling.mesh.triangles, dtype=float)
        for (point_x, point_y), wall_heights in wall_heights_by_xy.items():
            if not np.isclose(point_y, 0.0, atol=1e-9):
                continue
            matching_mask = np.isclose(
                ceiling_triangles[:, :, 0], point_x, atol=1e-9
            ) & np.isclose(ceiling_triangles[:, :, 1], point_y, atol=1e-9)
            matching_heights = ceiling_triangles[:, :, 2][matching_mask]
            if not len(matching_heights):
                continue
            unique_ceiling_heights = np.unique(np.round(matching_heights, 9))
            for ceiling_height in unique_ceiling_heights:
                self.assertTrue(
                    any(
                        np.isclose(
                            wall_height,
                            ceiling_height,
                            atol=1e-8,
                        )
                        for wall_height in wall_heights
                    ),
                    (
                        "The wall does not meet the ceiling at "
                        f"{(point_x, point_y, ceiling_height)!r}."
                    ),
                )
            self.assertAlmostEqual(
                max(wall_heights),
                float(np.max(unique_ceiling_heights)),
                places=8,
            )

        expected_core_drop = 1.0 * np.tan(np.radians(30.0))
        expected_core_top = (
            level.floor_thickness_meters
            + level.height_meters
            - CEILING_CLEARANCE_METERS
            - expected_core_drop
        )
        core_heights = [
            height
            for (point_x, point_y), height in wall_top_z_by_xy.items()
            if np.isclose(point_y, 0.0, atol=1e-9)
            and 0.8 - 1e-9 <= point_x <= 3.2 + 1e-9
        ]
        self.assertTrue(core_heights)
        self.assertAlmostEqual(min(core_heights), expected_core_top, places=8)

    def test_adjacent_wall_is_clipped_under_same_ceiling_skin(self) -> None:
        level = _build_room_level()
        flat_surfaces = build_base_fixed_surfaces(
            [level],
            apply_sloped_ceilings=False,
        )
        host_wall = _host_wall(flat_surfaces)
        level.sloped_ceilings.append(
            SlopedCeilingData(
                slope_id=SLOPE_ID,
                wall_surface_id=host_wall.surface_id,
                depth_meters=1.0,
                angle_degrees=35.0,
            )
        )

        surfaces = build_fixed_surfaces([level])
        walls = [surface for surface in surfaces if surface.surface_type == "wall"]
        adjacent = next(
            wall
            for wall in walls
            if wall.surface_id != host_wall.surface_id
            and any(
                np.allclose(endpoint[:2], host_wall.wall_start_world[:2])
                for endpoint in (wall.wall_start_world, wall.wall_end_world)
            )
        )
        adjacent_vertices = np.asarray(adjacent.mesh.vertices, dtype=float)
        shared_mask = (
            np.linalg.norm(
                adjacent_vertices[:, :2]
                - np.asarray(host_wall.wall_start_world[:2], dtype=float),
                axis=1,
            )
            < 1e-8
        )
        remote_mask = (
            np.linalg.norm(
                adjacent_vertices[:, :2]
                - np.asarray(host_wall.wall_start_world[:2], dtype=float),
                axis=1,
            )
            > 0.9
        )
        self.assertLess(
            float(np.max(adjacent_vertices[shared_mask, 2])),
            float(np.max(adjacent_vertices[remote_mask, 2])),
        )

    def test_sloped_host_wall_preserves_doorway_opening(self) -> None:
        level = _build_room_level()
        level.doorways.append(
            DoorwayData(
                center_x=100.0,
                center_y=0.0,
                width_meters=0.8,
                height_meters=2.0,
                depth_meters=0.2,
                rotation_degrees=90.0,
            )
        )
        flat_surfaces = build_base_fixed_surfaces(
            [level],
            apply_sloped_ceilings=False,
        )
        host_wall = _host_wall(flat_surfaces)
        level.sloped_ceilings.append(
            SlopedCeilingData(
                slope_id=SLOPE_ID,
                wall_surface_id=host_wall.surface_id,
                depth_meters=0.5,
                angle_degrees=20.0,
            )
        )

        sloped_wall = next(
            surface
            for surface in build_fixed_surfaces([level])
            if surface.surface_id == host_wall.surface_id
        )
        vertices = np.asarray(sloped_wall.mesh.vertices, dtype=float)
        polygons = []
        for face in np.asarray(sloped_wall.mesh.faces, dtype=np.int64):
            triangle = vertices[face]
            if not np.allclose(triangle[:, 1], 0.0, atol=1e-9):
                continue
            polygons.append(Polygon(triangle[:, (0, 2)]))
        visible_wall = shapely.union_all(polygons)
        self.assertFalse(visible_wall.covers(Point(2.0, 1.0)))
        self.assertTrue(visible_wall.covers(Point(2.0, 2.5)))

    def test_partial_sharp_slope_has_closed_side_returns(self) -> None:
        level = _build_room_level()
        flat_surfaces = build_base_fixed_surfaces(
            [level],
            apply_sloped_ceilings=False,
        )
        host_wall = _host_wall(flat_surfaces)
        level.sloped_ceilings.append(
            SlopedCeilingData(
                slope_id=SLOPE_ID,
                wall_surface_id=host_wall.surface_id,
                depth_meters=1.0,
                angle_degrees=30.0,
                transition_radius_meters=0.0,
                start_ratio=0.25,
                end_ratio=0.75,
            )
        )

        ceiling = _combined_ceiling_surface(build_base_fixed_surfaces([level]))
        normals = np.asarray(ceiling.mesh.face_normals, dtype=float)
        side_mask = (
            (np.abs(normals[:, 0]) > 0.99)
            & (np.abs(normals[:, 1]) < 1e-8)
            & (np.abs(normals[:, 2]) < 1e-8)
        )
        side_areas = np.asarray(ceiling.mesh.area_faces, dtype=float)[side_mask]
        expected_side_area = 0.5 * np.tan(np.radians(30.0))
        self.assertEqual(len(side_areas), 2)
        np.testing.assert_allclose(
            side_areas,
            np.full(2, expected_side_area),
            atol=1e-8,
        )
        self.assertEqual(
            {round(float(value), 6) for value in normals[side_mask, 0]},
            {-1.0, 1.0},
        )

        nominal_ceiling_z = (
            level.floor_thickness_meters
            + level.height_meters
            - CEILING_CLEARANCE_METERS
        )
        non_vertical_triangles = np.asarray(ceiling.mesh.triangles, dtype=float)[
            ~side_mask
        ]
        for triangle in non_vertical_triangles:
            centroid_x = float(np.mean(triangle[:, 0]))
            if centroid_x < 1.0 - 1e-8 or centroid_x > 3.0 + 1e-8:
                self.assertTrue(
                    np.allclose(
                        triangle[:, 2],
                        nominal_ceiling_z,
                        atol=1e-8,
                    )
                )

    def test_rounded_side_transitions_stay_inside_blue_gizmo_bounds(self) -> None:
        level = _build_room_level()
        flat_surfaces = build_base_fixed_surfaces(
            [level],
            apply_sloped_ceilings=False,
        )
        host_wall = _host_wall(flat_surfaces)
        level.sloped_ceilings.append(
            SlopedCeilingData(
                slope_id=SLOPE_ID,
                wall_surface_id=host_wall.surface_id,
                depth_meters=1.0,
                angle_degrees=30.0,
                transition_radius_meters=0.0,
                side_transition_radius_meters=0.3,
                start_ratio=0.25,
                end_ratio=0.75,
            )
        )

        ceiling = _combined_ceiling_surface(build_base_fixed_surfaces([level]))
        nominal_ceiling_z = (
            level.floor_thickness_meters
            + level.height_meters
            - CEILING_CLEARANCE_METERS
        )
        vertices = np.asarray(ceiling.mesh.vertices, dtype=float)
        outside = (vertices[:, 0] < 1.0 - 1e-8) | (
            vertices[:, 0] > 3.0 + 1e-8
        )
        self.assertTrue(np.any(outside))
        np.testing.assert_allclose(
            vertices[outside, 2],
            nominal_ceiling_z,
            atol=1e-8,
        )

    def test_ceiling_and_wall_rounding_affect_only_their_own_contacts(self) -> None:
        level = _build_room_level()
        flat_surfaces = build_base_fixed_surfaces(
            [level],
            apply_sloped_ceilings=False,
        )
        host_wall = _host_wall(flat_surfaces)

        def build_ceiling(
            *,
            ceiling_radius: float = 0.0,
            wall_radius: float = 0.0,
        ) -> FixedSurface:
            level.sloped_ceilings = [
                SlopedCeilingData(
                    slope_id=SLOPE_ID,
                    wall_surface_id=host_wall.surface_id,
                    depth_meters=1.0,
                    wall_drop_meters=0.6,
                    transition_radius_meters=ceiling_radius,
                    wall_transition_radius_meters=wall_radius,
                )
            ]
            return _combined_ceiling_surface(
                build_base_fixed_surfaces([level])
            )

        sharp = build_ceiling()
        ceiling_rounded = build_ceiling(ceiling_radius=0.25)
        wall_rounded = build_ceiling(wall_radius=0.25)
        target = next(
            target
            for target in build_sloped_ceiling_edit_targets([level], flat_surfaces)
            if target.wall_surface_id == host_wall.surface_id
        )
        inward = np.asarray(target.inward_world[:2], dtype=float)
        wall_center = (
            np.asarray(host_wall.wall_start_world[:2], dtype=float)
            + np.asarray(host_wall.wall_end_world[:2], dtype=float)
        ) * 0.5

        def height(surface: FixedSurface, depth: float) -> float:
            return _ceiling_height_at_xy(
                surface,
                tuple(wall_center + inward * depth),
            )

        def local_curvature(
            surface: FixedSurface,
            first_depth: float,
            middle_depth: float,
            last_depth: float,
        ) -> float:
            return abs(
                height(surface, first_depth)
                - 2.0 * height(surface, middle_depth)
                + height(surface, last_depth)
            )

        for surface in (ceiling_rounded, wall_rounded):
            self.assertAlmostEqual(height(surface, 0.0), height(sharp, 0.0))
            self.assertAlmostEqual(height(surface, 1.0), height(sharp, 1.0))

        self.assertLess(
            local_curvature(ceiling_rounded, 0.0, 0.025, 0.05),
            1e-7,
        )
        self.assertGreater(
            local_curvature(ceiling_rounded, 0.95, 0.975, 1.0),
            1e-4,
        )
        self.assertGreater(
            local_curvature(wall_rounded, 0.0, 0.025, 0.05),
            1e-4,
        )
        self.assertLess(
            local_curvature(wall_rounded, 0.95, 0.975, 1.0),
            1e-7,
        )

    def test_shared_side_rounding_preserves_contacts_core_and_closed_seams(
        self,
    ) -> None:
        level = _build_room_level()
        flat_surfaces = build_base_fixed_surfaces(
            [level],
            apply_sloped_ceilings=False,
        )
        host_wall = _host_wall(flat_surfaces)

        def build_ceiling(side_radius: float) -> FixedSurface:
            level.sloped_ceilings = [
                SlopedCeilingData(
                    slope_id=SLOPE_ID,
                    wall_surface_id=host_wall.surface_id,
                    depth_meters=1.0,
                    wall_drop_meters=0.6,
                    transition_radius_meters=0.15,
                    wall_transition_radius_meters=0.12,
                    side_transition_radius_meters=side_radius,
                    start_ratio=0.25,
                    end_ratio=0.75,
                )
            ]
            return _combined_ceiling_surface(
                build_base_fixed_surfaces([level])
            )

        sharp = build_ceiling(0.0)
        rounded = build_ceiling(0.25)
        nominal_ceiling_z = (
            level.floor_thickness_meters
            + level.height_meters
            - CEILING_CLEARANCE_METERS
        )
        target = next(
            target
            for target in build_sloped_ceiling_edit_targets([level], flat_surfaces)
            if target.wall_surface_id == host_wall.surface_id
        )
        inward = np.asarray(target.inward_world[:2], dtype=float)
        wall_start = np.asarray(host_wall.wall_start_world[:2], dtype=float)
        tangent = np.asarray(host_wall.wall_end_world[:2], dtype=float) - wall_start
        tangent /= np.linalg.norm(tangent)

        def height(surface: FixedSurface, along: float, depth: float) -> float:
            return _ceiling_height_at_xy(
                surface,
                tuple(wall_start + tangent * along + inward * depth),
            )

        for along in (1.02, 1.1, 2.0, 2.9, 2.98):
            self.assertAlmostEqual(
                height(rounded, along, 0.0),
                nominal_ceiling_z - 0.6,
                places=7,
            )
            self.assertAlmostEqual(
                height(rounded, along, 1.0),
                nominal_ceiling_z,
                places=7,
            )
            self.assertAlmostEqual(
                height(rounded, along, 0.0),
                height(sharp, along, 0.0),
                places=7,
            )
            self.assertAlmostEqual(
                height(rounded, along, 1.0),
                height(sharp, along, 1.0),
                places=7,
            )

        for depth in (0.0, 0.1, 0.5, 0.9, 1.0):
            self.assertAlmostEqual(
                height(rounded, 2.0, depth),
                height(sharp, 2.0, depth),
                places=7,
            )

        sharp_start_middle = height(sharp, 1.1, 0.5)
        sharp_end_middle = height(sharp, 2.9, 0.5)
        rounded_start_middle = height(rounded, 1.1, 0.5)
        rounded_end_middle = height(rounded, 2.9, 0.5)
        self.assertNotAlmostEqual(
            rounded_start_middle,
            sharp_start_middle,
            places=5,
        )
        self.assertNotAlmostEqual(
            rounded_end_middle,
            sharp_end_middle,
            places=5,
        )
        self.assertAlmostEqual(rounded_start_middle, rounded_end_middle, places=7)
        self.assertAlmostEqual(
            rounded_start_middle - sharp_start_middle,
            rounded_end_middle - sharp_end_middle,
            places=7,
        )

        vertices = np.asarray(rounded.mesh.vertices, dtype=float)
        outside = (vertices[:, 0] < 1.0 - 1e-8) | (
            vertices[:, 0] > 3.0 + 1e-8
        )
        np.testing.assert_allclose(
            vertices[outside, 2],
            nominal_ceiling_z,
            atol=1e-8,
        )
        _assert_no_internal_ceiling_boundary_edges(self, sharp)
        _assert_no_internal_ceiling_boundary_edges(self, rounded)

    def test_shared_side_rounding_is_an_exact_full_width_geometry_no_op(
        self,
    ) -> None:
        level = _build_room_level()
        flat_surfaces = build_base_fixed_surfaces(
            [level],
            apply_sloped_ceilings=False,
        )
        host_wall = _host_wall(flat_surfaces)

        def build_ceiling(side_radius: float) -> FixedSurface:
            level.sloped_ceilings = [
                SlopedCeilingData(
                    slope_id=SLOPE_ID,
                    wall_surface_id=host_wall.surface_id,
                    depth_meters=1.0,
                    wall_drop_meters=0.6,
                    transition_radius_meters=0.15,
                    wall_transition_radius_meters=0.12,
                    side_transition_radius_meters=side_radius,
                    start_ratio=0.0,
                    end_ratio=1.0,
                )
            ]
            return _combined_ceiling_surface(
                build_base_fixed_surfaces([level])
            )

        sharp = build_ceiling(0.0)
        stored_radius = build_ceiling(0.25)

        np.testing.assert_array_equal(
            np.asarray(stored_radius.mesh.vertices),
            np.asarray(sharp.mesh.vertices),
        )
        np.testing.assert_array_equal(
            np.asarray(stored_radius.mesh.faces),
            np.asarray(sharp.mesh.faces),
        )

    def test_shared_side_rounding_preserves_both_vertical_transition_bands(
        self,
    ) -> None:
        level = _build_room_level()
        flat_surfaces = build_base_fixed_surfaces(
            [level],
            apply_sloped_ceilings=False,
        )
        host_wall = _host_wall(flat_surfaces)
        target = next(
            target
            for target in build_sloped_ceiling_edit_targets([level], flat_surfaces)
            if target.wall_surface_id == host_wall.surface_id
        )
        inward = np.asarray(target.inward_world[:2], dtype=float)
        wall_start = np.asarray(host_wall.wall_start_world[:2], dtype=float)
        tangent = np.asarray(host_wall.wall_end_world[:2], dtype=float) - wall_start
        tangent /= np.linalg.norm(tangent)

        def build_ceiling(
            start_ratio: float,
            end_ratio: float,
            side_radius: float,
        ) -> FixedSurface:
            level.sloped_ceilings = [
                SlopedCeilingData(
                    slope_id=SLOPE_ID,
                    wall_surface_id=host_wall.surface_id,
                    depth_meters=1.0,
                    wall_drop_meters=0.6,
                    transition_radius_meters=0.15,
                    wall_transition_radius_meters=0.12,
                    side_transition_radius_meters=side_radius,
                    start_ratio=start_ratio,
                    end_ratio=end_ratio,
                )
            ]
            return _combined_ceiling_surface(
                build_base_fixed_surfaces([level])
            )

        def height(surface: FixedSurface, along: float, depth: float) -> float:
            return _ceiling_height_at_xy(
                surface,
                tuple(wall_start + tangent * along + inward * depth),
            )

        cases = (
            (0.25, 1.0, (1.1,)),
            (0.0, 0.75, (2.9,)),
            (0.25, 0.75, (1.1, 2.9)),
        )
        for start_ratio, end_ratio, lateral_probes in cases:
            with self.subTest(start_ratio=start_ratio, end_ratio=end_ratio):
                sharp = build_ceiling(start_ratio, end_ratio, 0.0)
                rounded = build_ceiling(start_ratio, end_ratio, 0.25)
                frame = sloped_ceiling_module._build_slope_frames(
                    [level],
                    flat_surfaces,
                )[0]
                profile = frame.profile
                wall_band_end = profile.wall_tangent_depth_meters
                ceiling_band_start = profile.ceiling_tangent_depth_meters
                depth = profile.depth_meters
                transition_depths = (
                    0.0,
                    *(wall_band_end * fraction for fraction in (0.2, 0.4, 0.6, 0.8)),
                    *(
                        ceiling_band_start
                        + (depth - ceiling_band_start) * fraction
                        for fraction in (0.2, 0.4, 0.6, 0.8)
                    ),
                    depth,
                )
                for along in lateral_probes:
                    for depth in transition_depths:
                        with self.subTest(along=along, depth=depth):
                            self.assertAlmostEqual(
                                height(rounded, along, depth),
                                height(sharp, along, depth),
                                places=7,
                            )

    def test_side_rounding_is_a_local_constant_radius_edge_fillet(
        self,
    ) -> None:
        level = _build_room_level()
        flat_surfaces = build_base_fixed_surfaces(
            [level],
            apply_sloped_ceilings=False,
        )
        host_wall = _host_wall(flat_surfaces)

        def build_ceiling(
            side_radius: float,
        ) -> tuple[FixedSurface, object]:
            level.sloped_ceilings = [
                SlopedCeilingData(
                    slope_id=SLOPE_ID,
                    wall_surface_id=host_wall.surface_id,
                    depth_meters=1.0,
                    wall_drop_meters=0.7,
                    transition_radius_meters=0.0,
                    wall_transition_radius_meters=0.0,
                    side_transition_radius_meters=side_radius,
                    start_ratio=0.25,
                    end_ratio=0.75,
                )
            ]
            ceiling = _combined_ceiling_surface(
                build_base_fixed_surfaces([level])
            )
            frame = sloped_ceiling_module._build_slope_frames(
                [level],
                flat_surfaces,
            )[0]
            return ceiling, frame

        side_radius = 0.1
        sharp, _sharp_frame = build_ceiling(0.0)
        rounded, frame = build_ceiling(side_radius)

        def height(
            surface: FixedSurface,
            along: float,
            depth: float,
        ) -> float:
            point = (
                frame.wall_start[:2]
                + frame.tangent_xy * along
                + frame.inward_xy * depth
            )
            return _ceiling_height_at_xy(surface, tuple(point))

        outside_strip_probes = (
            frame.start_distance + side_radius + 1e-4,
            (frame.start_distance + frame.end_distance) * 0.5,
            frame.end_distance - side_radius - 1e-4,
        )
        for along in outside_strip_probes:
            for depth in np.linspace(0.0, frame.profile.depth_meters, 9):
                with self.subTest(along=along, depth=depth):
                    self.assertAlmostEqual(
                        height(rounded, along, float(depth)),
                        height(sharp, along, float(depth)),
                        places=7,
                    )

        sample_angle = math.pi * 0.25
        lateral_offset = side_radius * (1.0 - math.cos(sample_angle))
        expected_lift = side_radius - math.sqrt(
            side_radius * side_radius
            - (side_radius - lateral_offset) ** 2
        )
        measured_lifts: list[float] = []
        for depth in (0.25, 0.5):
            for along in (
                frame.start_distance + lateral_offset,
                frame.end_distance - lateral_offset,
            ):
                measured_lift = height(rounded, along, depth) - height(
                    sharp,
                    along,
                    depth,
                )
                measured_lifts.append(measured_lift)
                with self.subTest(along=along, depth=depth):
                    self.assertAlmostEqual(
                        measured_lift,
                        expected_lift,
                        places=6,
                    )
        np.testing.assert_allclose(
            measured_lifts,
            expected_lift,
            rtol=0.0,
            atol=1e-6,
        )

        nominal_ceiling_z = (
            level.floor_thickness_meters
            + level.height_meters
            - CEILING_CLEARANCE_METERS
        )
        middle_depth = 0.5
        for along in (frame.start_distance, frame.end_distance):
            rounded_boundary_height = height(
                rounded,
                along,
                middle_depth,
            )
            sharp_boundary_height = height(
                sharp,
                along,
                middle_depth,
            )
            with self.subTest(boundary=along):
                self.assertLess(
                    rounded_boundary_height,
                    nominal_ceiling_z - 1e-6,
                )
                self.assertLessEqual(
                    rounded_boundary_height - sharp_boundary_height,
                    side_radius + 1e-6,
                )

        _assert_no_internal_ceiling_boundary_edges(self, rounded)

    def test_side_rounding_radius_changes_only_the_lateral_axis(self) -> None:
        level = _build_room_level()
        flat_surfaces = build_base_fixed_surfaces(
            [level],
            apply_sloped_ceilings=False,
        )
        host_wall = _host_wall(flat_surfaces)

        def build_ceiling(
            side_radius: float,
        ) -> tuple[FixedSurface, object]:
            level.sloped_ceilings = [
                SlopedCeilingData(
                    slope_id=SLOPE_ID,
                    wall_surface_id=host_wall.surface_id,
                    depth_meters=1.0,
                    wall_drop_meters=0.6,
                    transition_radius_meters=0.15,
                    wall_transition_radius_meters=0.12,
                    side_transition_radius_meters=side_radius,
                    start_ratio=0.25,
                    end_ratio=0.75,
                )
            ]
            ceiling = _combined_ceiling_surface(
                build_base_fixed_surfaces([level])
            )
            frame = sloped_ceiling_module._build_slope_frames(
                [level],
                flat_surfaces,
            )[0]
            return ceiling, frame

        small_radius = 0.1
        large_radius = 0.25
        sharp, _sharp_frame = build_ceiling(0.0)
        small, small_frame = build_ceiling(small_radius)
        large, large_frame = build_ceiling(large_radius)

        small_depth_breakpoints = (
            sloped_ceiling_module._slope_depth_breakpoints(small_frame)
        )
        large_depth_breakpoints = (
            sloped_ceiling_module._slope_depth_breakpoints(large_frame)
        )
        np.testing.assert_allclose(
            small_depth_breakpoints,
            large_depth_breakpoints,
            rtol=0.0,
            atol=1e-12,
        )

        small_tangent_breakpoints = (
            sloped_ceiling_module._slope_tangent_breakpoints(small_frame)
        )
        large_tangent_breakpoints = (
            sloped_ceiling_module._slope_tangent_breakpoints(large_frame)
        )
        self.assertNotEqual(
            small_tangent_breakpoints,
            large_tangent_breakpoints,
        )
        self.assertTrue(
            any(
                math.isclose(
                    value,
                    small_frame.start_distance + small_radius,
                    abs_tol=1e-12,
                )
                for value in small_tangent_breakpoints
            )
        )
        self.assertTrue(
            any(
                math.isclose(
                    value,
                    large_frame.start_distance + large_radius,
                    abs_tol=1e-12,
                )
                for value in large_tangent_breakpoints
            )
        )

        def height(
            surface: FixedSurface,
            frame: object,
            along: float,
            depth: float,
        ) -> float:
            point = (
                frame.wall_start[:2]
                + frame.tangent_xy * along
                + frame.inward_xy * depth
            )
            return _ceiling_height_at_xy(surface, tuple(point))

        profile = small_frame.profile
        middle_depth = (
            profile.wall_tangent_depth_meters
            + profile.ceiling_tangent_depth_meters
        ) * 0.5
        lower_probe = profile.wall_tangent_depth_meters + 0.03
        upper_probe = profile.ceiling_tangent_depth_meters - 0.03

        normalized_depth_effects: list[tuple[float, float]] = []

        def rounding_effect(
            rounded: FixedSurface,
            frame: object,
            along: float,
            depth: float,
        ) -> float:
            return height(rounded, frame, along, depth) - height(
                sharp,
                frame,
                along,
                depth,
            )

        for side_radius, rounded, frame in (
            (small_radius, small, small_frame),
            (large_radius, large, large_frame),
        ):
            along = frame.start_distance + side_radius * 0.5
            middle_effect = rounding_effect(
                rounded,
                frame,
                along,
                middle_depth,
            )
            self.assertGreater(abs(middle_effect), 1e-6)
            normalized_depth_effects.append(
                (
                    rounding_effect(
                        rounded,
                        frame,
                        along,
                        lower_probe,
                    )
                    / middle_effect,
                    rounding_effect(
                        rounded,
                        frame,
                        along,
                        upper_probe,
                    )
                    / middle_effect,
                )
            )

            protected_depths = (
                *np.linspace(
                    0.0,
                    profile.wall_tangent_depth_meters,
                    5,
                ),
                *np.linspace(
                    profile.ceiling_tangent_depth_meters,
                    profile.depth_meters,
                    5,
                ),
            )
            for depth in protected_depths:
                with self.subTest(side_radius=side_radius, depth=depth):
                    self.assertAlmostEqual(
                        height(rounded, frame, along, float(depth)),
                        height(sharp, frame, along, float(depth)),
                        places=7,
                    )

        np.testing.assert_allclose(
            normalized_depth_effects[0],
            normalized_depth_effects[1],
            rtol=0.0,
            atol=1e-7,
        )

        fixed_tangent_offset = 0.15
        fixed_along = small_frame.start_distance + fixed_tangent_offset
        small_effect = height(
            small,
            small_frame,
            fixed_along,
            middle_depth,
        ) - height(
            sharp,
            small_frame,
            fixed_along,
            middle_depth,
        )
        large_effect = height(
            large,
            large_frame,
            fixed_along,
            middle_depth,
        ) - height(
            sharp,
            large_frame,
            fixed_along,
            middle_depth,
        )
        self.assertAlmostEqual(small_effect, 0.0, places=7)
        self.assertGreater(abs(large_effect), 1e-6)

    def test_shared_side_rounding_affects_only_the_one_exposed_boundary(
        self,
    ) -> None:
        level = _build_room_level()
        flat_surfaces = build_base_fixed_surfaces(
            [level],
            apply_sloped_ceilings=False,
        )
        host_wall = _host_wall(flat_surfaces)
        target = next(
            target
            for target in build_sloped_ceiling_edit_targets([level], flat_surfaces)
            if target.wall_surface_id == host_wall.surface_id
        )
        inward = np.asarray(target.inward_world[:2], dtype=float)
        wall_start = np.asarray(host_wall.wall_start_world[:2], dtype=float)
        tangent = np.asarray(host_wall.wall_end_world[:2], dtype=float) - wall_start
        tangent /= np.linalg.norm(tangent)

        def build_ceiling(
            start_ratio: float,
            end_ratio: float,
            side_radius: float,
        ) -> FixedSurface:
            level.sloped_ceilings = [
                SlopedCeilingData(
                    slope_id=SLOPE_ID,
                    wall_surface_id=host_wall.surface_id,
                    depth_meters=1.0,
                    wall_drop_meters=0.6,
                    side_transition_radius_meters=side_radius,
                    start_ratio=start_ratio,
                    end_ratio=end_ratio,
                )
            ]
            return _combined_ceiling_surface(
                build_base_fixed_surfaces([level])
            )

        def height(
            surface: FixedSurface,
            along: float,
            depth: float = 0.5,
        ) -> float:
            return _ceiling_height_at_xy(
                surface,
                tuple(wall_start + tangent * along + inward * depth),
            )

        cases = (
            (0.25, 1.0, 1.1, 3.9),
            (0.0, 0.75, 2.9, 0.1),
        )
        for start_ratio, end_ratio, exposed_probe, joined_probe in cases:
            with self.subTest(start_ratio=start_ratio, end_ratio=end_ratio):
                sharp = build_ceiling(start_ratio, end_ratio, 0.0)
                rounded = build_ceiling(start_ratio, end_ratio, 0.25)
                self.assertNotAlmostEqual(
                    height(rounded, exposed_probe),
                    height(sharp, exposed_probe),
                    places=5,
                )
                for depth in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0):
                    with self.subTest(depth=depth):
                        self.assertAlmostEqual(
                            height(rounded, joined_probe, depth),
                            height(sharp, joined_probe, depth),
                            places=7,
                        )
                _assert_no_internal_ceiling_boundary_edges(self, rounded)

    def test_side_radius_rebuild_keeps_nominal_height_from_sloped_input(
        self,
    ) -> None:
        level = _build_room_level()
        flat_surfaces = build_base_fixed_surfaces(
            [level],
            apply_sloped_ceilings=False,
        )
        host_wall = _host_wall(flat_surfaces)
        level.sloped_ceilings = [
            SlopedCeilingData(
                slope_id=SLOPE_ID,
                wall_surface_id=host_wall.surface_id,
                depth_meters=1.0,
                wall_drop_meters=0.6,
                transition_radius_meters=0.15,
                wall_transition_radius_meters=0.12,
                side_transition_radius_meters=0.0,
                start_ratio=0.25,
                end_ratio=0.75,
            )
        ]
        first_commit = sloped_ceiling_module.apply_sloped_ceiling_geometry(
            [level],
            flat_surfaces,
        )
        stale_lowered_surfaces: list[FixedSurface] = []
        for surface in first_commit:
            if surface.surface_type != "ceiling":
                stale_lowered_surfaces.append(surface)
                continue
            lowered_mesh = surface.mesh.copy()
            lowered_mesh.apply_translation((0.0, 0.0, -0.4))
            stale_lowered_surfaces.append(
                replace(
                    surface,
                    mesh=lowered_mesh,
                    area_square_meters=float(lowered_mesh.area),
                )
            )

        apply_sloped_ceiling_edit(
            [level],
            SlopedCeilingEdit(
                slope_id=SLOPE_ID,
                wall_surface_id=host_wall.surface_id,
                level_index=level.index,
                handle_kind=SLOPED_CEILING_HANDLE_SIDE_TRANSITION_RADIUS,
                value=0.25,
            ),
        )
        rebuilt_from_stale = sloped_ceiling_module.apply_sloped_ceiling_geometry(
            [level],
            stale_lowered_surfaces,
        )
        rebuilt_from_flat = sloped_ceiling_module.apply_sloped_ceiling_geometry(
            [level],
            flat_surfaces,
        )
        stale_ceiling = _combined_ceiling_surface(rebuilt_from_stale)
        flat_ceiling = _combined_ceiling_surface(rebuilt_from_flat)
        frame = sloped_ceiling_module._build_slope_frames(
            [level],
            flat_surfaces,
        )[0]
        outside_xy = (
            frame.wall_start[:2]
            + frame.tangent_xy * frame.wall_length * 0.1
            + frame.inward_xy * frame.data.depth_meters * 0.5
        )
        inside_xy = (
            frame.wall_start[:2]
            + frame.tangent_xy * frame.wall_length * 0.5
            + frame.inward_xy * frame.data.depth_meters * 0.5
        )

        self.assertAlmostEqual(
            float(np.max(stale_ceiling.mesh.vertices[:, 2])),
            frame.ceiling_z,
        )
        for point in (outside_xy, inside_xy):
            self.assertAlmostEqual(
                _ceiling_height_at_xy(stale_ceiling, tuple(point)),
                _ceiling_height_at_xy(flat_ceiling, tuple(point)),
            )

    def test_two_disjoint_slopes_on_one_wall_keep_the_gap_flat(self) -> None:
        level = _build_room_level()
        flat_surfaces = build_base_fixed_surfaces(
            [level],
            apply_sloped_ceilings=False,
        )
        host_wall = _host_wall(flat_surfaces)
        level.sloped_ceilings = [
            SlopedCeilingData(
                slope_id="a" * 32,
                wall_surface_id=host_wall.surface_id,
                depth_meters=1.0,
                wall_drop_meters=0.5,
                start_ratio=0.05,
                end_ratio=0.35,
            ),
            SlopedCeilingData(
                slope_id="b" * 32,
                wall_surface_id=host_wall.surface_id,
                depth_meters=1.0,
                wall_drop_meters=0.7,
                start_ratio=0.65,
                end_ratio=0.95,
            ),
        ]

        ceiling = _combined_ceiling_surface(build_base_fixed_surfaces([level]))
        frame = sloped_ceiling_module._build_slope_frames(
            [level],
            flat_surfaces,
        )[0]

        def height_at_ratio(ratio: float) -> float:
            point = (
                frame.wall_start[:2]
                + frame.tangent_xy * frame.wall_length * ratio
                + frame.inward_xy * 0.2
            )
            return _ceiling_height_at_xy(ceiling, tuple(point))

        self.assertLess(height_at_ratio(0.1), frame.ceiling_z)
        self.assertAlmostEqual(height_at_ratio(0.5), frame.ceiling_z)
        self.assertLess(height_at_ratio(0.9), frame.ceiling_z)


# ### Edit and preview tests ###
class SlopedCeilingEditApiTests(unittest.TestCase):
    def test_create_target_apply_and_preview_share_stable_wall(self) -> None:
        level = _build_room_level()
        surfaces = build_base_fixed_surfaces([level])
        host_wall = _host_wall(surfaces)
        targets = build_sloped_ceiling_edit_targets([level], surfaces)
        target = next(
            item for item in targets if item.wall_surface_id == host_wall.surface_id
        )
        self.assertIsNone(target.slope_id)
        self.assertEqual(
            [handle.handle_kind for handle in target.handles],
            [SLOPED_CEILING_HANDLE_CREATE_DEPTH],
        )

        edit = SlopedCeilingEdit(
            slope_id=None,
            wall_surface_id=host_wall.surface_id,
            level_index=level.index,
            handle_kind=SLOPED_CEILING_HANDLE_CREATE_DEPTH,
            value=0.75,
        )
        preview_meshes = build_sloped_ceiling_preview_meshes(
            [level],
            surfaces,
            edit,
        )
        self.assertEqual(len(preview_meshes), 1)
        self.assertAlmostEqual(
            float(np.min(preview_meshes[0].vertices[:, 2])),
            target.ceiling_z_meters
            - DEFAULT_SLOPED_CEILING_WALL_DROP_METERS,
        )
        record = apply_sloped_ceiling_edit([level], edit)
        self.assertEqual(record.wall_surface_id, host_wall.surface_id)
        self.assertAlmostEqual(record.depth_meters, 0.75)
        self.assertAlmostEqual(
            record.wall_drop_meters,
            DEFAULT_SLOPED_CEILING_WALL_DROP_METERS,
        )
        with self.assertRaisesRegex(ValueError, "overlaps"):
            apply_sloped_ceiling_edit([level], edit)

        existing_target = next(
            item
            for item in build_sloped_ceiling_edit_targets([level], surfaces)
            if item.wall_surface_id == host_wall.surface_id
        )
        self.assertEqual(existing_target.slope_id, record.slope_id)
        self.assertLess(
            existing_target.selection_edge_start_world[2],
            existing_target.ceiling_z_meters,
        )

    def test_one_wall_exposes_each_slope_and_every_free_interval(self) -> None:
        level = _build_room_level()
        surfaces = build_base_fixed_surfaces(
            [level],
            apply_sloped_ceilings=False,
        )
        host_wall = _host_wall(surfaces)
        level.sloped_ceilings = [
            SlopedCeilingData(
                slope_id="a" * 32,
                wall_surface_id=host_wall.surface_id,
                start_ratio=0.2,
                end_ratio=0.4,
            ),
            SlopedCeilingData(
                slope_id="b" * 32,
                wall_surface_id=host_wall.surface_id,
                start_ratio=0.65,
                end_ratio=0.8,
            ),
        ]

        first_targets = tuple(
            target
            for target in build_sloped_ceiling_edit_targets([level], surfaces)
            if target.wall_surface_id == host_wall.surface_id
        )
        second_targets = tuple(
            target
            for target in build_sloped_ceiling_edit_targets([level], surfaces)
            if target.wall_surface_id == host_wall.surface_id
        )
        existing = tuple(
            target for target in first_targets if target.slope_id is not None
        )
        creation = tuple(
            target for target in first_targets if target.slope_id is None
        )

        self.assertEqual(len(existing), 2)
        self.assertEqual(
            [
                (target.creation_start_ratio, target.creation_end_ratio)
                for target in creation
            ],
            [(0.0, 0.2), (0.4, 0.65), (0.8, 1.0)],
        )
        self.assertEqual(
            [target.target_id for target in first_targets],
            [target.target_id for target in second_targets],
        )
        self.assertEqual(
            len({target.target_id for target in first_targets}),
            len(first_targets),
        )
        first_end = _target_handle(existing[0], SLOPED_CEILING_HANDLE_END)
        second_start = _target_handle(existing[1], SLOPED_CEILING_HANDLE_START)
        self.assertAlmostEqual(first_end.maximum_value, 0.65)
        self.assertAlmostEqual(second_start.minimum_value, 0.4)

    def test_create_in_free_interval_preserves_existing_same_wall_slope(
        self,
    ) -> None:
        level = _build_room_level()
        surfaces = build_base_fixed_surfaces(
            [level],
            apply_sloped_ceilings=False,
        )
        host_wall = _host_wall(surfaces)
        existing = SlopedCeilingData(
            slope_id=SLOPE_ID,
            wall_surface_id=host_wall.surface_id,
            start_ratio=0.0,
            end_ratio=0.4,
        )
        level.sloped_ceilings = [existing]
        creation_target = next(
            target
            for target in build_sloped_ceiling_edit_targets([level], surfaces)
            if target.wall_surface_id == host_wall.surface_id
            and target.slope_id is None
        )
        edit = SlopedCeilingEdit(
            slope_id=None,
            wall_surface_id=host_wall.surface_id,
            level_index=level.index,
            handle_kind=SLOPED_CEILING_HANDLE_CREATE_DEPTH,
            value=0.9,
            creation_start_ratio=creation_target.creation_start_ratio,
            creation_end_ratio=creation_target.creation_end_ratio,
        )

        preview_meshes = build_sloped_ceiling_preview_meshes(
            [level],
            surfaces,
            edit,
        )
        created = apply_sloped_ceiling_edit([level], edit)

        self.assertEqual(len(preview_meshes), 2)
        self.assertEqual(level.sloped_ceilings[0], existing)
        self.assertNotEqual(created.slope_id, existing.slope_id)
        self.assertEqual(
            (created.start_ratio, created.end_ratio),
            (0.4, 1.0),
        )
        self.assertEqual(len(level.sloped_ceilings), 2)

    def test_creating_or_resizing_a_slope_cannot_overlap_its_neighbor(
        self,
    ) -> None:
        level = _build_room_level()
        surfaces = build_base_fixed_surfaces(
            [level],
            apply_sloped_ceilings=False,
        )
        host_wall = _host_wall(surfaces)
        level.sloped_ceilings = [
            SlopedCeilingData(
                slope_id="a" * 32,
                wall_surface_id=host_wall.surface_id,
                start_ratio=0.0,
                end_ratio=0.4,
            ),
            SlopedCeilingData(
                slope_id="b" * 32,
                wall_surface_id=host_wall.surface_id,
                start_ratio=0.6,
                end_ratio=1.0,
            ),
        ]

        with self.assertRaisesRegex(ValueError, "overlaps"):
            apply_sloped_ceiling_edit(
                [level],
                SlopedCeilingEdit(
                    slope_id=None,
                    wall_surface_id=host_wall.surface_id,
                    level_index=level.index,
                    handle_kind=SLOPED_CEILING_HANDLE_CREATE_DEPTH,
                    value=0.5,
                    creation_start_ratio=0.3,
                    creation_end_ratio=0.7,
                ),
            )
        with self.assertRaisesRegex(ValueError, "overlaps"):
            apply_sloped_ceiling_edit(
                [level],
                SlopedCeilingEdit(
                    slope_id="a" * 32,
                    wall_surface_id=host_wall.surface_id,
                    level_index=level.index,
                    handle_kind=SLOPED_CEILING_HANDLE_END,
                    value=0.7,
                ),
            )
        self.assertEqual(
            [(slope.start_ratio, slope.end_ratio) for slope in level.sloped_ceilings],
            [(0.0, 0.4), (0.6, 1.0)],
        )

    def test_initial_depth_stops_at_opposite_ceiling_edge(self) -> None:
        level = _build_room_level()
        surfaces = build_base_fixed_surfaces(
            [level],
            apply_sloped_ceilings=False,
        )
        host_wall = _host_wall(surfaces)
        target = next(
            item
            for item in build_sloped_ceiling_edit_targets([level], surfaces)
            if item.wall_surface_id == host_wall.surface_id
        )
        create_handle = _target_handle(
            target,
            SLOPED_CEILING_HANDLE_CREATE_DEPTH,
        )
        self.assertAlmostEqual(create_handle.maximum_value, 4.0)

        apply_sloped_ceiling_edit(
            [level],
            SlopedCeilingEdit(
                slope_id=None,
                wall_surface_id=host_wall.surface_id,
                level_index=level.index,
                handle_kind=SLOPED_CEILING_HANDLE_CREATE_DEPTH,
                value=20.0,
            ),
        )
        ceiling = _combined_ceiling_surface(build_base_fixed_surfaces([level]))
        wall_start = np.asarray(host_wall.wall_start_world[:2], dtype=float)
        wall_end = np.asarray(host_wall.wall_end_world[:2], dtype=float)
        wall_midpoint = (wall_start + wall_end) * 0.5
        inward = np.asarray(target.inward_world[:2], dtype=float)

        self.assertAlmostEqual(
            _ceiling_height_at_xy(
                ceiling,
                tuple(wall_midpoint + inward * create_handle.maximum_value),
            ),
            target.ceiling_z_meters,
        )
        self.assertLess(
            _ceiling_height_at_xy(
                ceiling,
                tuple(wall_midpoint + inward * 2.0),
            ),
            target.ceiling_z_meters,
        )

    def test_depth_limit_detects_narrow_void_across_selected_width(self) -> None:
        ceiling_footprint = Polygon(
            (
                (0.0, 0.0),
                (10.0, 0.0),
                (10.0, 5.0),
                (0.0, 5.0),
            ),
            holes=(
                (
                    (6.0, 1.0),
                    (6.1, 1.5),
                    (6.0, 2.0),
                    (5.9, 1.5),
                ),
            ),
        )
        record = SlopedCeilingData(
            slope_id=SLOPE_ID,
            wall_surface_id="level:2/room:5/wall:1:2",
        )

        maximum_depth = sloped_ceiling_module._resolve_maximum_slope_depth(
            np.asarray((0.0, 0.0, 0.0), dtype=float),
            np.asarray((1.0, 0.0), dtype=float),
            np.asarray((0.0, 1.0), dtype=float),
            10.0,
            record,
            ceiling_footprint,
        )

        self.assertAlmostEqual(maximum_depth, 1.0)

    def test_depth_limit_handles_angled_host_boundaries(self) -> None:
        record = SlopedCeilingData(
            slope_id=SLOPE_ID,
            wall_surface_id="level:2/room:5/wall:1:2",
        )
        cases = (
            (
                Polygon(
                    ((0.0, 0.0), (10.0, 0.0), (8.0, 5.0), (2.0, 5.0))
                ),
                5.0,
            ),
            (
                Polygon(
                    ((0.0, 0.0), (10.0, 0.0), (12.0, 5.0), (2.0, 5.0))
                ),
                5.0,
            ),
            (
                Polygon(((0.0, 0.0), (10.0, 0.0), (5.0, 2.0), (0.0, 2.0))),
                2.0,
            ),
            (
                Polygon(
                    (
                        (0.0, 0.0),
                        (10.0, 0.0),
                        (10.0, 5.0),
                        (6.0, 5.0),
                        (4.0, 1.0),
                        (0.0, 1.0),
                    )
                ),
                3.0,
            ),
        )

        for ceiling_footprint, expected_depth in cases:
            with self.subTest(bounds=ceiling_footprint.bounds):
                maximum_depth = (
                    sloped_ceiling_module._resolve_maximum_slope_depth(
                        np.asarray((0.0, 0.0, 0.0), dtype=float),
                        np.asarray((1.0, 0.0), dtype=float),
                        np.asarray((0.0, 1.0), dtype=float),
                        10.0,
                        record,
                        ceiling_footprint,
                    )
                )

                self.assertAlmostEqual(maximum_depth, expected_depth)

    def test_green_depth_handle_moves_only_the_ceiling_contact(self) -> None:
        level = _build_room_level()
        surfaces = build_base_fixed_surfaces(
            [level],
            apply_sloped_ceilings=False,
        )
        host_wall = _host_wall(surfaces)
        level.sloped_ceilings = [
            SlopedCeilingData(
                slope_id=SLOPE_ID,
                wall_surface_id=host_wall.surface_id,
                depth_meters=0.8,
                wall_drop_meters=0.6,
            )
        ]
        before = next(
            target
            for target in build_sloped_ceiling_edit_targets([level], surfaces)
            if target.wall_surface_id == host_wall.surface_id
        )
        before_top = _target_handle(before, SLOPED_CEILING_HANDLE_DEPTH)
        before_bottom = _target_handle(before, SLOPED_CEILING_HANDLE_WALL_DROP)

        apply_sloped_ceiling_edit(
            [level],
            SlopedCeilingEdit(
                slope_id=SLOPE_ID,
                wall_surface_id=host_wall.surface_id,
                level_index=level.index,
                handle_kind=SLOPED_CEILING_HANDLE_DEPTH,
                value=1.2,
            ),
        )
        after = next(
            target
            for target in build_sloped_ceiling_edit_targets([level], surfaces)
            if target.wall_surface_id == host_wall.surface_id
        )
        after_top = _target_handle(after, SLOPED_CEILING_HANDLE_DEPTH)
        after_bottom = _target_handle(after, SLOPED_CEILING_HANDLE_WALL_DROP)
        inward = np.asarray(after.inward_world, dtype=float)
        top_motion = (
            np.asarray(after_top.origin_world, dtype=float)
            - np.asarray(before_top.origin_world, dtype=float)
        )

        self.assertAlmostEqual(float(np.dot(top_motion, inward)), 0.4)
        self.assertAlmostEqual(after_top.origin_world[2], before_top.origin_world[2])
        np.testing.assert_allclose(
            after_bottom.origin_world,
            before_bottom.origin_world,
            atol=1e-9,
        )
        self.assertAlmostEqual(level.sloped_ceilings[0].wall_drop_meters, 0.6)
        self.assertAlmostEqual(
            level.sloped_ceilings[0].angle_degrees,
            math.degrees(math.atan2(0.6, 1.2)),
        )

    def test_orange_wall_drop_handle_moves_only_the_wall_contact_upward(self) -> None:
        level = _build_room_level()
        surfaces = build_base_fixed_surfaces(
            [level],
            apply_sloped_ceilings=False,
        )
        host_wall = _host_wall(surfaces)
        level.sloped_ceilings = [
            SlopedCeilingData(
                slope_id=SLOPE_ID,
                wall_surface_id=host_wall.surface_id,
                depth_meters=1.0,
                wall_drop_meters=0.7,
            )
        ]
        before = next(
            target
            for target in build_sloped_ceiling_edit_targets([level], surfaces)
            if target.wall_surface_id == host_wall.surface_id
        )
        before_top = _target_handle(before, SLOPED_CEILING_HANDLE_DEPTH)
        before_bottom = _target_handle(before, SLOPED_CEILING_HANDLE_WALL_DROP)
        self.assertGreater(before_bottom.axis_world[2], 0.0)
        self.assertLess(before_bottom.value_per_meter, 0.0)

        apply_sloped_ceiling_edit(
            [level],
            SlopedCeilingEdit(
                slope_id=SLOPE_ID,
                wall_surface_id=host_wall.surface_id,
                level_index=level.index,
                handle_kind=SLOPED_CEILING_HANDLE_WALL_DROP,
                value=0.4,
            ),
        )
        after = next(
            target
            for target in build_sloped_ceiling_edit_targets([level], surfaces)
            if target.wall_surface_id == host_wall.surface_id
        )
        after_top = _target_handle(after, SLOPED_CEILING_HANDLE_DEPTH)
        after_bottom = _target_handle(after, SLOPED_CEILING_HANDLE_WALL_DROP)

        np.testing.assert_allclose(
            after_top.origin_world,
            before_top.origin_world,
            atol=1e-9,
        )
        self.assertAlmostEqual(
            after_bottom.origin_world[2] - before_bottom.origin_world[2],
            0.3,
        )

    def test_shared_side_rounding_preview_matches_the_committed_edit(self) -> None:
        level = _build_room_level()
        surfaces = build_base_fixed_surfaces(
            [level],
            apply_sloped_ceilings=False,
        )
        host_wall = _host_wall(surfaces)
        level.sloped_ceilings = [
            SlopedCeilingData(
                slope_id=SLOPE_ID,
                wall_surface_id=host_wall.surface_id,
                side_transition_radius_meters=0.0,
                start_ratio=0.2,
                end_ratio=0.8,
            )
        ]
        edit = SlopedCeilingEdit(
            slope_id=SLOPE_ID,
            wall_surface_id=host_wall.surface_id,
            level_index=level.index,
            handle_kind=SLOPED_CEILING_HANDLE_SIDE_TRANSITION_RADIUS,
            value=0.25,
        )

        preview_meshes = build_sloped_ceiling_preview_meshes(
            [level],
            surfaces,
            edit,
        )
        self.assertAlmostEqual(
            level.sloped_ceilings[0].side_transition_radius_meters,
            0.0,
        )
        apply_sloped_ceiling_edit([level], edit)
        committed_meshes = build_sloped_ceiling_preview_meshes(
            [level],
            surfaces,
        )

        self.assertEqual(len(preview_meshes), 1)
        self.assertEqual(len(committed_meshes), 1)
        np.testing.assert_allclose(
            preview_meshes[0].vertices,
            committed_meshes[0].vertices,
            atol=1e-9,
        )
        np.testing.assert_array_equal(
            preview_meshes[0].faces,
            committed_meshes[0].faces,
        )

    def test_one_shared_lateral_handle_exists_when_any_blue_bound_is_internal(
        self,
    ) -> None:
        level = _build_room_level()
        surfaces = build_base_fixed_surfaces(
            [level],
            apply_sloped_ceilings=False,
        )
        host_wall = _host_wall(surfaces)
        tangent = np.asarray(host_wall.wall_end_world, dtype=float) - np.asarray(
            host_wall.wall_start_world,
            dtype=float,
        )
        tangent /= np.linalg.norm(tangent)
        cases = (
            (0.0, 1.0, 0, None),
            (0.2, 1.0, 1, 1.0),
            (0.0, 0.8, 1, -1.0),
            (0.2, 0.8, 1, None),
        )

        for start_ratio, end_ratio, expected_count, expected_axis_sign in cases:
            with self.subTest(start_ratio=start_ratio, end_ratio=end_ratio):
                level.sloped_ceilings = [
                    SlopedCeilingData(
                        slope_id=SLOPE_ID,
                        wall_surface_id=host_wall.surface_id,
                        start_ratio=start_ratio,
                        end_ratio=end_ratio,
                    )
                ]
                target = next(
                    target
                    for target in build_sloped_ceiling_edit_targets(
                        [level],
                        surfaces,
                    )
                    if target.wall_surface_id == host_wall.surface_id
                )
                kinds = {handle.handle_kind for handle in target.handles}
                side_handles = [
                    handle
                    for handle in target.handles
                    if handle.handle_kind
                    == SLOPED_CEILING_HANDLE_SIDE_TRANSITION_RADIUS
                ]
                self.assertEqual(len(side_handles), expected_count)
                if expected_axis_sign is not None:
                    self.assertAlmostEqual(
                        float(np.dot(side_handles[0].axis_world, tangent)),
                        expected_axis_sign,
                    )
                self.assertIn(
                    SLOPED_CEILING_HANDLE_CEILING_TRANSITION_RADIUS,
                    kinds,
                )
                self.assertIn(
                    SLOPED_CEILING_HANDLE_WALL_TRANSITION_RADIUS,
                    kinds,
                )

    def test_vertical_rounding_handles_are_anchored_at_both_contacts(self) -> None:
        level = _build_room_level()
        surfaces = build_base_fixed_surfaces(
            [level],
            apply_sloped_ceilings=False,
        )
        host_wall = _host_wall(surfaces)
        level.sloped_ceilings = [
            SlopedCeilingData(
                slope_id=SLOPE_ID,
                wall_surface_id=host_wall.surface_id,
                depth_meters=1.0,
                wall_drop_meters=0.6,
                start_ratio=0.2,
                end_ratio=0.8,
            )
        ]
        target = next(
            target
            for target in build_sloped_ceiling_edit_targets([level], surfaces)
            if target.wall_surface_id == host_wall.surface_id
        )
        top = _target_handle(target, SLOPED_CEILING_HANDLE_DEPTH)
        bottom = _target_handle(target, SLOPED_CEILING_HANDLE_WALL_DROP)
        ceiling_rounding = _target_handle(
            target,
            SLOPED_CEILING_HANDLE_CEILING_TRANSITION_RADIUS,
        )
        wall_rounding = _target_handle(
            target,
            SLOPED_CEILING_HANDLE_WALL_TRANSITION_RADIUS,
        )

        np.testing.assert_allclose(
            ceiling_rounding.origin_world,
            top.origin_world,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            wall_rounding.origin_world,
            bottom.origin_world,
            atol=1e-9,
        )
        purple_kinds = {
            SLOPED_CEILING_HANDLE_CEILING_TRANSITION_RADIUS,
            SLOPED_CEILING_HANDLE_WALL_TRANSITION_RADIUS,
            SLOPED_CEILING_HANDLE_SIDE_TRANSITION_RADIUS,
        }
        self.assertEqual(
            len(
                [
                    handle
                    for handle in target.handles
                    if handle.handle_kind in purple_kinds
                ]
            ),
            3,
        )

    def test_zero_depth_restores_a_flat_wall_target(self) -> None:
        level = _build_room_level()
        surfaces = build_base_fixed_surfaces([level])
        host_wall = _host_wall(surfaces)
        _add_slope(level, host_wall.surface_id)
        slope = level.sloped_ceilings[0]

        removed = apply_sloped_ceiling_edit(
            [level],
            SlopedCeilingEdit(
                slope_id=slope.slope_id,
                wall_surface_id=slope.wall_surface_id,
                level_index=level.index,
                handle_kind=SLOPED_CEILING_HANDLE_DEPTH,
                value=0.0,
            ),
        )

        self.assertEqual(removed, slope)
        self.assertEqual(level.sloped_ceilings, [])
        target = next(
            item
            for item in build_sloped_ceiling_edit_targets([level], surfaces)
            if item.wall_surface_id == host_wall.surface_id
        )
        self.assertIsNone(target.slope_id)
        self.assertEqual(
            tuple(handle.handle_kind for handle in target.handles),
            (SLOPED_CEILING_HANDLE_CREATE_DEPTH,),
        )

    def test_editable_child_wall_does_not_disable_other_slope_targets(self) -> None:
        level = _build_room_level()
        surfaces = build_base_fixed_surfaces([level])
        host_wall = _host_wall(surfaces)
        child = FixedSurface(
            surface_id="level:2/edit-face:child/wall",
            surface_type="wall",
            level_index=level.index,
            room_index=host_wall.room_index,
            mesh=host_wall.mesh.copy(),
            area_square_meters=host_wall.area_square_meters,
            source_surface_id=host_wall.surface_id,
        )

        targets = build_sloped_ceiling_edit_targets(
            [level],
            (*surfaces, child),
        )

        self.assertTrue(targets)
        self.assertNotIn(
            child.surface_id,
            {target.wall_surface_id for target in targets},
        )

    def test_manual_wall_winding_does_not_reverse_the_slope_direction(self) -> None:
        level = _build_room_level()
        surfaces = build_base_fixed_surfaces([level])
        host_wall = _host_wall(surfaces)
        original_target = next(
            target
            for target in build_sloped_ceiling_edit_targets([level], surfaces)
            if target.wall_surface_id == host_wall.surface_id
        )
        flipped_mesh = host_wall.mesh.copy()
        flipped_mesh.faces = np.asarray(flipped_mesh.faces, dtype=np.int64)[
            :, ::-1
        ]
        flipped_wall = replace(host_wall, mesh=flipped_mesh)
        flipped_surfaces = tuple(
            flipped_wall if surface.surface_id == host_wall.surface_id else surface
            for surface in surfaces
        )

        flipped_target = next(
            target
            for target in build_sloped_ceiling_edit_targets(
                [level],
                flipped_surfaces,
            )
            if target.wall_surface_id == host_wall.surface_id
        )

        np.testing.assert_allclose(
            flipped_target.inward_world,
            original_target.inward_world,
            atol=1e-9,
        )


# ### Sloped-window tests ###
class SlopedCeilingWindowTests(unittest.TestCase):
    def _build_window_fixture(
        self,
    ) -> tuple[
        LevelData,
        list[FixedSurface],
        FixedSurface,
        SlopedCeilingData,
    ]:
        level = _build_room_level()
        flat_surfaces = build_base_fixed_surfaces(
            [level],
            apply_sloped_ceilings=False,
        )
        host_wall = _host_wall(flat_surfaces)
        slope = SlopedCeilingData(
            slope_id=SLOPE_ID,
            wall_surface_id=host_wall.surface_id,
            depth_meters=1.2,
            wall_drop_meters=0.8,
            transition_radius_meters=0.1,
            wall_transition_radius_meters=0.1,
            side_transition_radius_meters=0.1,
            start_ratio=0.1,
            end_ratio=0.9,
        )
        level.sloped_ceilings = [slope]
        return level, flat_surfaces, host_wall, slope

    def test_host_uses_only_the_straight_central_slope_band(self) -> None:
        level, surfaces, host_wall, _slope = self._build_window_fixture()

        host = build_sloped_ceiling_window_hosts([level], surfaces)[0]

        self.assertEqual(host.target_id, SLOPE_ID)
        self.assertEqual(host.wall_surface_id, host_wall.surface_id)
        self.assertEqual(host.level_index, level.index)
        wall_length = float(
            np.linalg.norm(
                np.asarray(host_wall.wall_end_world, dtype=float)
                - np.asarray(host_wall.wall_start_world, dtype=float)
            )
        )
        self.assertAlmostEqual(
            host.wall_width_meters,
            wall_length * 0.8 - 0.2,
        )
        self.assertGreater(host.wall_height_meters, 0.05)
        axes = np.asarray(
            (
                host.wall_tangent_world,
                host.vertical_axis_world,
                host.wall_normal_world,
            ),
            dtype=float,
        )
        np.testing.assert_allclose(np.linalg.norm(axes, axis=1), 1.0)
        np.testing.assert_allclose(axes @ axes.T, np.eye(3), atol=1e-9)
        point = host.local_to_world(0.35, 0.65)
        np.testing.assert_allclose(host.world_to_local(point), (0.35, 0.65))

    def test_placement_commit_and_canvas_target_share_one_planar_frame(
        self,
    ) -> None:
        level, surfaces, _host_wall, _slope = self._build_window_fixture()
        host = build_sloped_ceiling_window_hosts([level], surfaces)[0]
        first = host.local_to_world(0.2, 0.25)
        second = host.local_to_world(0.8, 0.75)

        placement = build_sloped_ceiling_window_placement(host, first, second)
        window = add_sloped_ceiling_window(
            [level],
            placement,
            window_id="slope-window",
        )
        target = build_sloped_ceiling_window_opening_targets(
            [level],
            surfaces,
        )[0]

        self.assertIsInstance(placement, SlopedCeilingWindowPlacement)
        self.assertEqual(window.sloped_ceiling_id, SLOPE_ID)
        self.assertEqual(target.reference.stable_id, window.window_id)
        self.assertEqual(target.vertical_axis_world, host.vertical_axis_world)
        np.testing.assert_allclose(
            target.get_world_corners(),
            get_sloped_ceiling_window_world_corners(host, placement),
        )

    def test_window_cuts_the_slope_and_builds_four_sealed_reveals(self) -> None:
        level, flat_surfaces, _host_wall, _slope = self._build_window_fixture()
        host = build_sloped_ceiling_window_hosts([level], flat_surfaces)[0]
        placement = build_sloped_ceiling_window_placement(
            host,
            host.local_to_world(0.25, 0.3),
            host.local_to_world(0.75, 0.7),
        )
        add_sloped_ceiling_window(
            [level],
            placement,
            window_id="slope-window",
        )

        sloped_surfaces = build_base_fixed_surfaces([level])
        ceiling = _combined_ceiling_surface(
            sloped_surfaces,
            room_index=host.room_index,
        )
        rebuilt_host = build_sloped_ceiling_window_hosts(
            [level],
            sloped_surfaces,
        )[0]
        self.assertAlmostEqual(
            rebuilt_host.wall_width_meters,
            host.wall_width_meters,
        )
        self.assertAlmostEqual(
            rebuilt_host.wall_height_meters,
            host.wall_height_meters,
        )
        inner = np.asarray(
            get_sloped_ceiling_window_world_corners(host, placement),
            dtype=float,
        )
        outer = inner + np.asarray(host.wall_normal_world) * 0.5
        expected = np.concatenate((inner, outer), axis=0)
        matching_faces = 0
        for triangle in np.asarray(ceiling.mesh.triangles, dtype=float):
            if all(
                np.any(np.all(np.isclose(expected, point, atol=1e-8), axis=1))
                for point in triangle
            ) and not (
                all(
                    np.any(np.all(np.isclose(inner, point, atol=1e-8), axis=1))
                    for point in triangle
                )
                or all(
                    np.any(np.all(np.isclose(outer, point, atol=1e-8), axis=1))
                    for point in triangle
                )
            ):
                matching_faces += 1
        self.assertEqual(matching_faces, 8)

        center = np.mean(inner, axis=0)
        aligned_skin_at_center = False
        host_normal = np.asarray(host.wall_normal_world, dtype=float)
        for triangle, face_normal in zip(
            np.asarray(ceiling.mesh.triangles, dtype=float),
            np.asarray(ceiling.mesh.face_normals, dtype=float),
        ):
            if abs(float(np.dot(face_normal, host_normal))) < 0.999:
                continue
            projected = Polygon(triangle[:, :2])
            if projected.buffer(1e-9).covers(Point(tuple(center[:2]))):
                aligned_skin_at_center = True
                break
        self.assertFalse(aligned_skin_at_center)

        rebuilt_again = sloped_ceiling_module.apply_sloped_ceiling_geometry(
            [level],
            sloped_surfaces,
        )
        second_ceiling = _combined_ceiling_surface(
            rebuilt_again,
            host_surface_id=ceiling.surface_id,
        )
        self.assertAlmostEqual(second_ceiling.mesh.area, ceiling.mesh.area)

    def test_orphaned_slope_binding_does_not_cut_a_ceiling(self) -> None:
        level, flat_surfaces, host_wall, _slope = self._build_window_fixture()
        level.windows = [
            WindowData(
                window_id="orphan",
                wall_surface_id=host_wall.surface_id,
                start_ratio=0.2,
                end_ratio=0.8,
                bottom_ratio=0.2,
                top_ratio=0.8,
                sloped_ceiling_id="b" * 32,
            )
        ]

        self.assertEqual(
            build_sloped_ceiling_window_opening_targets(
                [level],
                flat_surfaces,
            ),
            (),
        )


# ### GLB replacement tests ###
class SlopedCeilingExportTests(unittest.TestCase):
    def test_preview_export_and_textured_surface_use_semantic_slope(self) -> None:
        level = _build_room_level()
        flat_surfaces = build_base_fixed_surfaces(
            [level],
            apply_sloped_ceilings=False,
        )
        host_wall = _host_wall(flat_surfaces)
        _add_slope(level, host_wall.surface_id)
        semantic_ceiling = _sloped_ceiling_surface(
            build_fixed_surfaces([level])
        )

        preview_model = convert_to_preview_model([level])
        exported_model = convert_to_glb([level])
        self.assertAlmostEqual(
            float(preview_model.mesh.bounds[0, 2]),
            float(exported_model.mesh.bounds[0, 2]),
            places=9,
        )
        self.assertTrue(
            any("slope_geometry" in name for name in exported_model.scene.geometry)
        )

        textured_model = convert_to_glb(
            [level],
            surface_materials={semantic_ceiling.surface_id: _solid_png()},
        )
        previews = {
            preview.surface_id: preview
            for preview in textured_model.preview_textured_surfaces
        }
        self.assertIn(semantic_ceiling.surface_id, previews)
        np.testing.assert_allclose(
            previews[semantic_ceiling.surface_id].mesh.triangles,
            semantic_ceiling.mesh.triangles,
            atol=1e-9,
        )

    def test_textured_sharp_partial_slope_keeps_its_side_returns(self) -> None:
        level = _build_room_level()
        flat_surfaces = build_base_fixed_surfaces(
            [level],
            apply_sloped_ceilings=False,
        )
        host_wall = _host_wall(flat_surfaces)
        level.sloped_ceilings.append(
            SlopedCeilingData(
                slope_id=SLOPE_ID,
                wall_surface_id=host_wall.surface_id,
                depth_meters=1.0,
                angle_degrees=30.0,
                transition_radius_meters=0.0,
                start_ratio=0.25,
                end_ratio=0.75,
            )
        )
        slope_surfaces = [
            surface
            for surface in build_fixed_surfaces([level])
            if surface.sloped_ceiling_id == SLOPE_ID
        ]
        side_surfaces = [
            surface
            for surface in slope_surfaces
            if surface.sloped_ceiling_part in {"start-return", "end-return"}
        ]
        self.assertEqual(len(side_surfaces), 2)

        textured_model = convert_to_glb(
            [level],
            surface_materials={
                surface.surface_id: _solid_png()
                for surface in side_surfaces
            },
        )
        previews = {
            item.surface_id: item
            for item in textured_model.preview_textured_surfaces
        }
        self.assertEqual(
            set(previews),
            {surface.surface_id for surface in side_surfaces},
        )
        side_normals = np.concatenate(
            tuple(
                np.asarray(previews[surface.surface_id].mesh.face_normals)
                for surface in side_surfaces
            ),
            axis=0,
        )

        self.assertEqual(len(side_normals), 2)
        self.assertEqual(
            {round(float(value), 6) for value in side_normals[:, 0]},
            {-1.0, 1.0},
        )


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
