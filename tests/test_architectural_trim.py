# ### Imports ###
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from housemaker.architectural_trim import (
    TRIM_PART_FRONT,
    ArchitecturalTrimPlacementRequest,
    add_architectural_trim,
    build_architectural_trim_edit_targets,
    build_architectural_trim_geometry,
    build_architectural_trim_placement_preview_meshes,
    build_architectural_trim_surface_id,
    build_cornice_miter_descriptors,
    is_architectural_trim_surface_id,
    remove_architectural_trim,
    update_architectural_trim,
)
from housemaker.models import (
    TRIM_KIND_CORNICE,
    TRIM_KIND_EDGING_STRIP,
    TRIM_KIND_SKIRTING_BOARD,
    ArchitecturalTrimData,
    DoorwayData,
    LevelData,
    RoomData,
    VertexData,
)
from housemaker.project_io import load_project, save_project
from housemaker.surface_geometry import (
    FixedSurface,
    build_base_fixed_surfaces,
    build_fixed_surfaces,
)


# ### Test helpers ###
def _wall_surface(
    surface_id: str,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    *,
    reverse_faces: bool = False,
) -> FixedSurface:
    start_vertex = np.asarray(start, dtype=float)
    end_vertex = np.asarray(end, dtype=float)
    top_offset = np.asarray((0.0, 0.0, 3.0), dtype=float)
    faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    if reverse_faces:
        faces = faces[:, ::-1]
    mesh = trimesh.Trimesh(
        vertices=np.asarray(
            (
                start_vertex,
                end_vertex,
                end_vertex + top_offset,
                start_vertex + top_offset,
            ),
            dtype=float,
        ),
        faces=faces,
        process=False,
    )
    return FixedSurface(
        surface_id=surface_id,
        surface_type="wall",
        level_index=2,
        room_index=None,
        mesh=mesh,
        area_square_meters=float(mesh.area),
        wall_key=surface_id.rsplit("wall:", 1)[1],
        wall_start_world=start,
        wall_end_world=end,
        wall_height_meters=3.0,
    )


def _corner_level_and_surfaces() -> tuple[LevelData, tuple[FixedSurface, ...]]:
    vertices = VertexData()
    first_id = vertices.add_vertex(0.0, 0.0).id
    corner_id = vertices.add_vertex(100.0, 0.0).id
    last_id = vertices.add_vertex(100.0, 100.0).id
    vertices.add_edge(first_id, corner_id)
    vertices.add_edge(corner_id, last_id)
    level = LevelData(index=2, name="Ground", vertex_data=vertices)
    surfaces = (
        _wall_surface(
            f"level:2/wall:{first_id}:{corner_id}",
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
        ),
        _wall_surface(
            f"level:2/wall:{corner_id}:{last_id}",
            (2.0, 0.0, 0.0),
            (2.0, 2.0, 0.0),
        ),
    )
    return level, surfaces


def _wall_surface_with_base_gap() -> FixedSurface:
    rectangles = (
        ((0.0, 0.0, 0.0), (1.5, 0.0, 0.0)),
        ((2.5, 0.0, 0.0), (4.0, 0.0, 0.0)),
    )
    vertices: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []
    top = np.asarray((0.0, 0.0, 3.0), dtype=float)
    for start, end in rectangles:
        offset = len(vertices)
        start_point = np.asarray(start, dtype=float)
        end_point = np.asarray(end, dtype=float)
        vertices.extend((start_point, end_point, end_point + top, start_point + top))
        faces.extend(
            (
                (offset, offset + 1, offset + 2),
                (offset, offset + 2, offset + 3),
            )
        )
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    return FixedSurface(
        surface_id="level:2/wall:1:2",
        surface_type="wall",
        level_index=2,
        room_index=None,
        mesh=mesh,
        area_square_meters=float(mesh.area),
        wall_key="1:2",
        wall_start_world=(0.0, 0.0, 0.0),
        wall_end_world=(4.0, 0.0, 0.0),
        wall_height_meters=3.0,
    )


def _convex_hull_area(points: np.ndarray) -> float:
    """Return the area of a small 2D point set's convex hull."""

    unique_points = sorted(
        {
            (round(float(point[0]), 9), round(float(point[1]), 9))
            for point in np.asarray(points, dtype=float)
        }
    )
    if len(unique_points) < 3:
        return 0.0

    def cross(
        origin: tuple[float, float],
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    return (
        abs(
            sum(
                first[0] * second[1] - first[1] * second[0]
                for first, second in zip(hull, (*hull[1:], hull[0]), strict=True)
            )
        )
        * 0.5
    )


def _normalized_wall_profile(
    mesh: trimesh.Trimesh,
    *,
    hangs_from_ceiling: bool,
) -> set[tuple[float, float]]:
    """Return outward/depth profile points independent of vertical placement."""

    vertices = np.asarray(mesh.vertices, dtype=float)
    vertical_coordinates = (
        np.max(vertices[:, 2]) - vertices[:, 2]
        if hangs_from_ceiling
        else vertices[:, 2] - np.min(vertices[:, 2])
    )
    return {
        (round(float(-vertex[1]), 7), round(float(vertical), 7))
        for vertex, vertical in zip(vertices, vertical_coordinates, strict=True)
    }


def _cornice_sloped_face_mask(mesh: trimesh.Trimesh) -> np.ndarray:
    """Select profile faces that bridge both wall depth and ceiling height."""

    normals = np.asarray(mesh.face_normals, dtype=float)
    return (
        (np.abs(normals[:, 0]) < 0.25)
        & (np.abs(normals[:, 1]) > 0.08)
        & (np.abs(normals[:, 2]) > 0.08)
    )


def _unique_cornice_slope_normals(mesh: trimesh.Trimesh) -> set[tuple[float, float]]:
    """Return unique wall-normal/vertical directions across the crown face."""

    normals = np.asarray(mesh.face_normals, dtype=float)
    return {
        (round(float(normal[1]), 6), round(float(normal[2]), 6))
        for normal in normals[_cornice_sloped_face_mask(mesh)]
    }


def _cornice_sloped_profile_points(
    mesh: trimesh.Trimesh,
) -> set[tuple[float, float]]:
    """Return outward/downward points belonging to the visible crown face."""

    face_indices = np.flatnonzero(_cornice_sloped_face_mask(mesh))
    if not len(face_indices):
        return set()
    vertex_indices = np.unique(np.asarray(mesh.faces)[face_indices].reshape(-1))
    vertices = np.asarray(mesh.vertices, dtype=float)[vertex_indices]
    ceiling_z = float(np.max(np.asarray(mesh.vertices, dtype=float)[:, 2]))
    return {
        (round(float(-vertex[1]), 7), round(ceiling_z - float(vertex[2]), 7))
        for vertex in vertices
    }


# ### Domain and persistence tests ###
class ArchitecturalTrimDataTests(unittest.TestCase):
    def test_round_trip_preserves_level_owned_trim(self) -> None:
        level, surfaces = _corner_level_and_surfaces()
        level.architectural_trims.append(
            ArchitecturalTrimData(
                trim_id="a" * 32,
                kind=TRIM_KIND_EDGING_STRIP,
                wall_surface_ids=tuple(surface.surface_id for surface in surfaces),
                corner_vertex_id=2,
                width_meters=0.06,
                height_meters=2.7,
                depth_meters=0.03,
            )
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "trim.json"
            save_project(project_path, level.index, [level])
            restored = load_project(project_path)

        self.assertEqual(
            restored.levels[level.index].architectural_trims, level.architectural_trims
        )

    def test_edging_requires_two_walls_and_a_corner_vertex(self) -> None:
        with self.assertRaisesRegex(ValueError, "two walls"):
            ArchitecturalTrimData(
                trim_id="b" * 32,
                kind=TRIM_KIND_EDGING_STRIP,
                wall_surface_ids=("level:2/wall:1:2",),
            )

    def test_crud_retains_identity_and_validates_against_live_walls(self) -> None:
        level, surfaces = _corner_level_and_surfaces()
        request = ArchitecturalTrimPlacementRequest(
            trim_id="c" * 32,
            kind=TRIM_KIND_SKIRTING_BOARD,
            wall_surface_ids=(surfaces[0].surface_id,),
        )
        added = add_architectural_trim(level, request, surfaces)
        updated = update_architectural_trim(
            level,
            added.trim_id,
            height_meters=0.16,
            wall_surfaces=surfaces,
        )
        removed = remove_architectural_trim(level, added.trim_id)

        self.assertEqual(updated.trim_id, added.trim_id)
        self.assertEqual(updated.height_meters, 0.16)
        self.assertEqual(removed, updated)
        self.assertEqual(level.architectural_trims, [])

    def test_edging_occupancy_is_per_corner_instead_of_per_wall(self) -> None:
        vertices = VertexData()
        left_corner = vertices.add_vertex(0.0, 0.0).id
        right_corner = vertices.add_vertex(100.0, 0.0).id
        left_end = vertices.add_vertex(0.0, 100.0).id
        right_end = vertices.add_vertex(100.0, 100.0).id
        level = LevelData(index=2, name="Ground", vertex_data=vertices)
        surfaces = (
            _wall_surface(
                f"level:2/wall:{left_corner}:{right_corner}",
                (0.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
            ),
            _wall_surface(
                f"level:2/wall:{left_corner}:{left_end}",
                (0.0, 0.0, 0.0),
                (0.0, 2.0, 0.0),
            ),
            _wall_surface(
                f"level:2/wall:{right_corner}:{right_end}",
                (2.0, 0.0, 0.0),
                (2.0, 2.0, 0.0),
            ),
        )
        first = add_architectural_trim(
            level,
            ArchitecturalTrimPlacementRequest(
                trim_id="e" * 32,
                kind=TRIM_KIND_EDGING_STRIP,
                wall_surface_ids=(surfaces[0].surface_id, surfaces[1].surface_id),
                corner_vertex_id=left_corner,
            ),
            surfaces,
        )
        second = add_architectural_trim(
            level,
            ArchitecturalTrimPlacementRequest(
                trim_id="f" * 32,
                kind=TRIM_KIND_EDGING_STRIP,
                wall_surface_ids=(surfaces[0].surface_id, surfaces[2].surface_id),
                corner_vertex_id=right_corner,
            ),
            surfaces,
        )

        self.assertEqual((first.corner_vertex_id, second.corner_vertex_id), (1, 2))
        with self.assertRaisesRegex(ValueError, "location"):
            add_architectural_trim(
                level,
                ArchitecturalTrimPlacementRequest(
                    trim_id="a" * 32,
                    kind=TRIM_KIND_EDGING_STRIP,
                    wall_surface_ids=(
                        surfaces[0].surface_id,
                        surfaces[1].surface_id,
                    ),
                    corner_vertex_id=left_corner,
                ),
                surfaces,
            )


# ### Geometry tests ###
class ArchitecturalTrimGeometryTests(unittest.TestCase):
    def test_trim_follows_a_manually_flipped_host_wall(self) -> None:
        level, surfaces = _corner_level_and_surfaces()
        wall_id = surfaces[0].surface_id
        level.architectural_trims.append(
            ArchitecturalTrimData(
                trim_id="9" * 32,
                kind=TRIM_KIND_SKIRTING_BOARD,
                wall_surface_ids=(wall_id,),
            )
        )

        original_front = next(
            surface
            for surface in build_fixed_surfaces([level])
            if surface.surface_id
            == build_architectural_trim_surface_id("9" * 32, TRIM_PART_FRONT)
        )
        level.flipped_surface_ids.add(wall_id)
        flipped_front = next(
            surface
            for surface in build_fixed_surfaces([level])
            if surface.surface_id
            == build_architectural_trim_surface_id("9" * 32, TRIM_PART_FRONT)
        )

        self.assertLess(float(original_front.mesh.centroid[1]), 0.0)
        self.assertGreater(float(flipped_front.mesh.centroid[1]), 0.0)

    def test_edging_strip_wraps_both_host_walls_with_an_l_profile(self) -> None:
        level, surfaces = _corner_level_and_surfaces()
        width = 0.24
        depth = 0.03
        level.architectural_trims.append(
            ArchitecturalTrimData(
                trim_id="e" * 32,
                kind=TRIM_KIND_EDGING_STRIP,
                wall_surface_ids=tuple(surface.surface_id for surface in surfaces),
                corner_vertex_id=2,
                width_meters=width,
                height_meters=2.5,
                depth_meters=depth,
            )
        )

        mesh = build_architectural_trim_geometry([level], surfaces).runs[0].mesh
        vertices = np.asarray(mesh.vertices, dtype=float)
        footprint = np.unique(np.round(vertices[:, :2], decimals=8), axis=0)
        upward_faces = np.asarray(mesh.face_normals, dtype=float)[:, 2] > 0.99
        top_area = float(np.sum(np.asarray(mesh.area_faces)[upward_faces]))
        hull_area = _convex_hull_area(footprint)

        self.assertGreaterEqual(len(footprint), 6)
        self.assertLess(top_area, hull_area - 1e-5)
        first_wall_contacts = footprint[np.abs(footprint[:, 1]) <= 1e-6]
        second_wall_contacts = footprint[np.abs(footprint[:, 0] - 2.0) <= 1e-6]
        self.assertGreaterEqual(len(first_wall_contacts), 2)
        self.assertGreaterEqual(len(second_wall_contacts), 2)
        self.assertGreater(
            float(np.ptp(first_wall_contacts[:, 0])),
            width * 0.75,
        )
        self.assertGreater(
            float(np.ptp(second_wall_contacts[:, 1])),
            width * 0.75,
        )

    def test_edging_front_parts_follow_both_walls_when_wall_order_is_reversed(
        self,
    ) -> None:
        level, surfaces = _corner_level_and_surfaces()
        level.architectural_trims.append(
            ArchitecturalTrimData(
                trim_id="f" * 32,
                kind=TRIM_KIND_EDGING_STRIP,
                wall_surface_ids=tuple(
                    surface.surface_id for surface in reversed(surfaces)
                ),
                corner_vertex_id=2,
                width_meters=0.24,
                height_meters=2.5,
                depth_meters=0.03,
            )
        )

        geometry = build_architectural_trim_geometry([level], surfaces)
        front = next(
            part for part in geometry.parts if part.part_kind == TRIM_PART_FRONT
        )
        expected_normals = tuple(
            np.asarray(surface.mesh.face_normals[0], dtype=float)
            for surface in surfaces
        )

        self.assertTrue(geometry.runs[0].mesh.is_watertight)
        self.assertTrue(geometry.runs[0].mesh.is_winding_consistent)
        for normal in np.asarray(front.mesh.face_normals, dtype=float):
            self.assertGreater(
                max(float(np.dot(normal, expected)) for expected in expected_normals),
                0.99,
            )

    def test_cornice_zero_radius_uses_a_straight_profile_at_the_ceiling(
        self,
    ) -> None:
        height = 0.24
        depth = 0.12
        cornice_level, cornice_surfaces = _corner_level_and_surfaces()
        cornice_level.architectural_trims.append(
            ArchitecturalTrimData(
                trim_id="b" * 32,
                kind=TRIM_KIND_CORNICE,
                wall_surface_ids=(cornice_surfaces[0].surface_id,),
                height_meters=height,
                depth_meters=depth,
                corner_radius_meters=0.0,
            )
        )
        cornice_mesh = (
            build_architectural_trim_geometry(
                [cornice_level],
                cornice_surfaces,
            )
            .runs[0]
            .mesh
        )

        cornice_profile = _normalized_wall_profile(
            cornice_mesh,
            hangs_from_ceiling=True,
        )
        sloped_points = _cornice_sloped_profile_points(cornice_mesh)
        slope_normals = _unique_cornice_slope_normals(cornice_mesh)

        self.assertAlmostEqual(float(cornice_mesh.bounds[1, 2]), 3.0)
        self.assertAlmostEqual(float(cornice_mesh.bounds[0, 2]), 3.0 - height)
        self.assertTrue(
            {(0.0, 0.0), (0.0, height), (depth, 0.0)}.issubset(cornice_profile)
        )
        self.assertNotIn((depth, height), cornice_profile)
        self.assertEqual(len(slope_normals), 1)
        self.assertLessEqual(min(point[0] for point in sloped_points), depth * 0.25)
        self.assertGreaterEqual(max(point[0] for point in sloped_points), depth * 0.95)
        self.assertLessEqual(min(point[1] for point in sloped_points), height * 0.05)
        self.assertGreaterEqual(max(point[1] for point in sloped_points), height * 0.95)

    def test_cornice_positive_radius_uses_a_smooth_rounded_profile(self) -> None:
        height = 0.24
        depth = 0.12
        cornice_level, cornice_surfaces = _corner_level_and_surfaces()
        cornice_level.architectural_trims.append(
            ArchitecturalTrimData(
                trim_id="b" * 32,
                kind=TRIM_KIND_CORNICE,
                wall_surface_ids=(cornice_surfaces[0].surface_id,),
                height_meters=height,
                depth_meters=depth,
                corner_radius_meters=0.04,
            )
        )

        cornice_mesh = (
            build_architectural_trim_geometry(
                [cornice_level],
                cornice_surfaces,
            )
            .runs[0]
            .mesh
        )
        cornice_profile = _normalized_wall_profile(
            cornice_mesh,
            hangs_from_ceiling=True,
        )
        sloped_points = _cornice_sloped_profile_points(cornice_mesh)
        slope_normals = _unique_cornice_slope_normals(cornice_mesh)
        middle_curve_points = {
            point
            for point in cornice_profile
            if depth * 0.2 < point[0] < depth * 0.9
            and height * 0.2 < point[1] < height * 0.8
        }

        self.assertAlmostEqual(float(cornice_mesh.bounds[1, 2]), 3.0)
        self.assertGreater(len(cornice_profile), 4)
        self.assertGreaterEqual(len(slope_normals), 4)
        self.assertTrue(middle_curve_points)
        self.assertLessEqual(min(point[0] for point in sloped_points), depth * 0.25)
        self.assertGreaterEqual(max(point[0] for point in sloped_points), depth * 0.95)
        self.assertLessEqual(min(point[1] for point in sloped_points), height * 0.05)
        self.assertGreaterEqual(max(point[1] for point in sloped_points), height * 0.95)

    def test_cornice_preview_uses_the_same_profile_as_committed_geometry(
        self,
    ) -> None:
        level, surfaces = _corner_level_and_surfaces()
        request = ArchitecturalTrimPlacementRequest(
            trim_id="9" * 32,
            kind=TRIM_KIND_CORNICE,
            wall_surface_ids=(surfaces[0].surface_id,),
            height_meters=0.24,
            depth_meters=0.12,
            corner_radius_meters=0.04,
        )
        level.architectural_trims.append(request.to_data())

        committed = build_architectural_trim_geometry([level], surfaces).runs[0].mesh
        previews = build_architectural_trim_placement_preview_meshes(
            request,
            surfaces,
        )

        self.assertEqual(len(previews), 1)
        preview = previews[0]
        self.assertEqual(
            _normalized_wall_profile(preview, hangs_from_ceiling=True),
            _normalized_wall_profile(committed, hangs_from_ceiling=True),
        )
        self.assertEqual(
            _unique_cornice_slope_normals(preview),
            _unique_cornice_slope_normals(committed),
        )
        np.testing.assert_allclose(preview.bounds, committed.bounds)

    def test_compatible_cornices_report_matching_outer_corner_miters(
        self,
    ) -> None:
        level, surfaces = _corner_level_and_surfaces()
        depth = 0.12
        first_id = "1" * 32
        second_id = "2" * 32
        level.architectural_trims.extend(
            (
                ArchitecturalTrimData(
                    trim_id=first_id,
                    kind=TRIM_KIND_CORNICE,
                    wall_surface_ids=(surfaces[0].surface_id,),
                    depth_meters=depth,
                ),
                ArchitecturalTrimData(
                    trim_id=second_id,
                    kind=TRIM_KIND_CORNICE,
                    wall_surface_ids=(surfaces[1].surface_id,),
                    depth_meters=depth,
                ),
            )
        )

        descriptors = {
            descriptor.trim_id: descriptor
            for descriptor in build_cornice_miter_descriptors([level], surfaces)
        }
        first_joined = tuple(
            endpoint
            for endpoint in (
                descriptors[first_id].minimum_x,
                descriptors[first_id].maximum_x,
            )
            if endpoint.is_joined
        )
        second_joined = tuple(
            endpoint
            for endpoint in (
                descriptors[second_id].minimum_x,
                descriptors[second_id].maximum_x,
            )
            if endpoint.is_joined
        )

        self.assertEqual(len(first_joined), 1)
        self.assertEqual(len(second_joined), 1)
        self.assertEqual(first_joined[0].neighbor_trim_id, second_id)
        self.assertEqual(second_joined[0].neighbor_trim_id, first_id)
        np.testing.assert_allclose(
            first_joined[0].miter_point_world(depth),
            second_joined[0].miter_point_world(depth),
        )
        np.testing.assert_allclose(
            first_joined[0].world_shift_per_depth,
            second_joined[0].world_shift_per_depth,
        )
        self.assertAlmostEqual(abs(first_joined[0].local_x_shift_per_depth), 1.0)
        self.assertAlmostEqual(abs(second_joined[0].local_x_shift_per_depth), 1.0)

    def test_reversed_outer_faces_report_matching_cornice_miters(self) -> None:
        level, surfaces = _corner_level_and_surfaces()
        reversed_surfaces = tuple(
            _wall_surface(
                surface.surface_id,
                surface.wall_start_world,
                surface.wall_end_world,
                reverse_faces=True,
            )
            for surface in surfaces
        )
        first_id = "3" * 32
        second_id = "4" * 32
        level.architectural_trims.extend(
            (
                ArchitecturalTrimData(
                    trim_id=first_id,
                    kind=TRIM_KIND_CORNICE,
                    wall_surface_ids=(reversed_surfaces[0].surface_id,),
                ),
                ArchitecturalTrimData(
                    trim_id=second_id,
                    kind=TRIM_KIND_CORNICE,
                    wall_surface_ids=(reversed_surfaces[1].surface_id,),
                ),
            )
        )

        descriptors = {
            descriptor.trim_id: descriptor
            for descriptor in build_cornice_miter_descriptors(
                [level],
                reversed_surfaces,
            )
        }
        joined = {
            trim_id: next(
                endpoint
                for endpoint in (
                    descriptor.minimum_x,
                    descriptor.maximum_x,
                )
                if endpoint.is_joined
            )
            for trim_id, descriptor in descriptors.items()
        }

        self.assertEqual(joined[first_id].neighbor_trim_id, second_id)
        self.assertEqual(joined[second_id].neighbor_trim_id, first_id)
        np.testing.assert_allclose(
            joined[first_id].miter_point_world(0.05),
            joined[second_id].miter_point_world(0.05),
        )
        self.assertNotEqual(
            joined[first_id].boundary,
            joined[second_id].boundary,
        )

    def test_open_or_different_style_cornices_report_no_miter(self) -> None:
        level, surfaces = _corner_level_and_surfaces()
        level.architectural_trims.extend(
            (
                ArchitecturalTrimData(
                    trim_id="5" * 32,
                    kind=TRIM_KIND_CORNICE,
                    wall_surface_ids=(surfaces[0].surface_id,),
                ),
                ArchitecturalTrimData(
                    trim_id="6" * 32,
                    kind=TRIM_KIND_CORNICE,
                    wall_surface_ids=(surfaces[1].surface_id,),
                    depth_meters=0.04,
                ),
            )
        )

        descriptors = build_cornice_miter_descriptors([level], surfaces)

        self.assertEqual(len(descriptors), 2)
        for descriptor in descriptors:
            for endpoint in (descriptor.minimum_x, descriptor.maximum_x):
                self.assertFalse(endpoint.is_joined)
                self.assertIsNone(endpoint.neighbor_trim_id)
                self.assertEqual(endpoint.local_x_shift_per_depth, 0.0)
                self.assertEqual(
                    endpoint.miter_point_world(0.25),
                    endpoint.wall_point_world,
                )

    def test_compatible_corner_components_form_one_watertight_mitered_run(
        self,
    ) -> None:
        level, surfaces = _corner_level_and_surfaces()
        level.architectural_trims.extend(
            (
                ArchitecturalTrimData(
                    trim_id="1" * 32,
                    kind=TRIM_KIND_SKIRTING_BOARD,
                    wall_surface_ids=(surfaces[0].surface_id,),
                ),
                ArchitecturalTrimData(
                    trim_id="2" * 32,
                    kind=TRIM_KIND_SKIRTING_BOARD,
                    wall_surface_ids=(surfaces[1].surface_id,),
                ),
            )
        )

        geometry = build_architectural_trim_geometry([level], surfaces)

        self.assertEqual(len(geometry.runs), 1)
        self.assertEqual(geometry.runs[0].trim_ids, ("1" * 32, "2" * 32))
        self.assertTrue(geometry.runs[0].mesh.is_watertight)
        self.assertTrue(geometry.runs[0].mesh.is_winding_consistent)
        self.assertGreater(geometry.runs[0].mesh.volume, 0.0)
        self.assertEqual(len(geometry.runs[0].mesh.faces), 20)
        self.assertEqual(
            {part.trim_id for part in geometry.parts},
            {"1" * 32, "2" * 32},
        )
        expected_front_normals = {
            "1" * 32: np.asarray((0.0, -1.0, 0.0)),
            "2" * 32: np.asarray((1.0, 0.0, 0.0)),
        }
        for part in geometry.parts:
            if part.part_kind != TRIM_PART_FRONT:
                continue
            dots = np.dot(part.mesh.face_normals, expected_front_normals[part.trim_id])
            self.assertTrue(np.all(dots > 0.99))

    def test_different_profiles_do_not_join(self) -> None:
        level, surfaces = _corner_level_and_surfaces()
        level.architectural_trims.extend(
            (
                ArchitecturalTrimData(
                    trim_id="3" * 32,
                    kind=TRIM_KIND_CORNICE,
                    wall_surface_ids=(surfaces[0].surface_id,),
                ),
                ArchitecturalTrimData(
                    trim_id="4" * 32,
                    kind=TRIM_KIND_CORNICE,
                    wall_surface_ids=(surfaces[1].surface_id,),
                    depth_meters=0.04,
                ),
            )
        )

        geometry = build_architectural_trim_geometry([level], surfaces)

        self.assertEqual(len(geometry.runs), 2)

    def test_reversed_outer_faces_still_form_one_watertight_mitered_run(
        self,
    ) -> None:
        level, surfaces = _corner_level_and_surfaces()
        reversed_surfaces = tuple(
            _wall_surface(
                surface.surface_id,
                surface.wall_start_world,
                surface.wall_end_world,
                reverse_faces=True,
            )
            for surface in surfaces
        )
        level.architectural_trims.extend(
            (
                ArchitecturalTrimData(
                    trim_id="7" * 32,
                    kind=TRIM_KIND_CORNICE,
                    wall_surface_ids=(reversed_surfaces[0].surface_id,),
                ),
                ArchitecturalTrimData(
                    trim_id="8" * 32,
                    kind=TRIM_KIND_CORNICE,
                    wall_surface_ids=(reversed_surfaces[1].surface_id,),
                ),
            )
        )

        geometry = build_architectural_trim_geometry(
            [level],
            reversed_surfaces,
        )

        self.assertEqual(len(geometry.runs), 1)
        self.assertTrue(geometry.runs[0].mesh.is_watertight)
        self.assertTrue(geometry.runs[0].mesh.is_winding_consistent)
        self.assertGreater(geometry.runs[0].mesh.volume, 0.0)

    def test_opposing_host_face_sides_stay_separate_with_outward_fronts(
        self,
    ) -> None:
        level, surfaces = _corner_level_and_surfaces()
        mismatched_surfaces = (
            surfaces[0],
            _wall_surface(
                surfaces[1].surface_id,
                surfaces[1].wall_start_world,
                surfaces[1].wall_end_world,
                reverse_faces=True,
            ),
        )
        for index, surface in enumerate(mismatched_surfaces, start=1):
            level.architectural_trims.append(
                ArchitecturalTrimData(
                    trim_id=str(index) * 32,
                    kind=TRIM_KIND_SKIRTING_BOARD,
                    wall_surface_ids=(surface.surface_id,),
                )
            )

        geometry = build_architectural_trim_geometry([level], mismatched_surfaces)

        self.assertEqual(len(geometry.runs), 2)
        for part in geometry.parts:
            if part.part_kind != TRIM_PART_FRONT:
                continue
            surface = mismatched_surfaces[int(part.trim_id[0]) - 1]
            host_normal = np.mean(surface.mesh.face_normals, axis=0)
            self.assertTrue(np.all(np.dot(part.mesh.face_normals, host_normal) > 0.99))

    def test_closed_room_run_has_shared_closing_ring_and_no_end_caps(self) -> None:
        vertex_data = VertexData()
        vertex_ids = tuple(
            vertex_data.add_vertex(*point).id
            for point in (
                (0.0, 0.0),
                (100.0, 0.0),
                (100.0, 100.0),
                (0.0, 100.0),
            )
        )
        wall_pairs = tuple(
            zip(vertex_ids, (*vertex_ids[1:], vertex_ids[0]), strict=True)
        )
        for start_id, end_id in wall_pairs:
            vertex_data.add_edge(start_id, end_id)
        level = LevelData(
            index=2,
            name="Ground",
            vertex_data=vertex_data,
        )
        points = (
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 2.0, 0.0),
            (0.0, 2.0, 0.0),
        )
        surfaces = tuple(
            _wall_surface(
                f"level:2/wall:{start_id}:{end_id}",
                points[index],
                points[(index + 1) % len(points)],
            )
            for index, (start_id, end_id) in enumerate(wall_pairs)
        )
        for index, surface in enumerate(surfaces, start=10):
            level.architectural_trims.append(
                ArchitecturalTrimData(
                    trim_id=f"{index:x}" * 32,
                    kind=TRIM_KIND_SKIRTING_BOARD,
                    wall_surface_ids=(surface.surface_id,),
                )
            )

        geometry = build_architectural_trim_geometry([level], surfaces)

        self.assertEqual(len(geometry.runs), 1)
        run_mesh = geometry.runs[0].mesh
        self.assertTrue(run_mesh.is_watertight)
        self.assertTrue(run_mesh.is_winding_consistent)
        self.assertGreater(run_mesh.volume, 0.0)
        self.assertEqual(len(run_mesh.vertices), 16)
        self.assertEqual(len(run_mesh.faces), 32)

    def test_unambiguous_pair_joins_at_higher_incidence_junction(self) -> None:
        level = LevelData(index=2, name="Ground")
        surfaces = (
            _wall_surface(
                "level:2/wall:1:2",
                (-2.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            ),
            _wall_surface(
                "level:2/wall:2:3",
                (0.0, 0.0, 0.0),
                (0.0, 2.0, 0.0),
            ),
            _wall_surface(
                "level:2/wall:2:4",
                (0.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
            ),
        )
        for index, surface in enumerate(surfaces, start=1):
            level.architectural_trims.append(
                ArchitecturalTrimData(
                    trim_id=str(index) * 32,
                    kind=TRIM_KIND_SKIRTING_BOARD,
                    wall_surface_ids=(surface.surface_id,),
                )
            )

        geometry = build_architectural_trim_geometry([level], surfaces)

        self.assertEqual(len(geometry.runs), 2)
        self.assertIn(
            ("1" * 32, "2" * 32),
            {run.trim_ids for run in geometry.runs},
        )
        self.assertTrue(all(run.mesh.is_watertight for run in geometry.runs))

    def test_skirting_uses_real_wall_mesh_to_cap_doorway_gap(self) -> None:
        level = LevelData(index=2, name="Ground")
        surface = _wall_surface_with_base_gap()
        level.architectural_trims.append(
            ArchitecturalTrimData(
                trim_id="b" * 32,
                kind=TRIM_KIND_SKIRTING_BOARD,
                wall_surface_ids=(surface.surface_id,),
            )
        )

        geometry = build_architectural_trim_geometry([level], (surface,))

        self.assertEqual(len(geometry.runs), 1)
        run = geometry.runs[0]
        self.assertEqual(len(run.mesh.faces), 24)
        self.assertTrue(run.mesh.is_watertight)
        self.assertTrue(run.mesh.is_winding_consistent)
        self.assertGreater(run.mesh.volume, 0.0)
        x_coordinates = np.asarray(run.mesh.vertices, dtype=float)[:, 0]
        self.assertFalse(np.any((x_coordinates > 1.5) & (x_coordinates < 2.5)))
        self.assertEqual(len({part.semantic_id for part in run.parts}), 4)

    def test_generated_doorway_reveals_do_not_confuse_skirting_clipping(
        self,
    ) -> None:
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
            strict=True,
        ):
            vertex_data.add_edge(start_id, end_id)
        center = vertex_data.add_vertex(50.0, 50.0)
        level = LevelData(
            index=2,
            name="Ground",
            vertex_data=vertex_data,
            rooms=[
                RoomData(
                    name="Room",
                    vertex_ids=boundary_ids,
                    center_vertex_id=center.id,
                    color_rgb=(120, 140, 160),
                )
            ],
            doorways=[
                DoorwayData(
                    center_x=50.0,
                    center_y=0.0,
                    width_meters=0.8,
                    height_meters=2.0,
                    depth_meters=0.2,
                    rotation_degrees=90.0,
                )
            ],
        )
        surfaces = tuple(build_base_fixed_surfaces([level]))
        doorway_wall = next(
            surface
            for surface in surfaces
            if surface.surface_type == "wall"
            and surface.surface_id.endswith("wall:1:2")
        )
        level.architectural_trims.append(
            ArchitecturalTrimData(
                trim_id="d" * 32,
                kind=TRIM_KIND_SKIRTING_BOARD,
                wall_surface_ids=(doorway_wall.surface_id,),
            )
        )

        run = build_architectural_trim_geometry([level], surfaces).runs[0]

        self.assertEqual(len(run.mesh.faces), 24)
        self.assertTrue(run.mesh.is_watertight)
        x_coordinates = np.asarray(run.mesh.vertices, dtype=float)[:, 0]
        self.assertFalse(np.any((x_coordinates > 0.6) & (x_coordinates < 1.4)))

    def test_cornice_height_handle_moves_down_and_reports_clamped_height(
        self,
    ) -> None:
        level, surfaces = _corner_level_and_surfaces()
        level.architectural_trims.append(
            ArchitecturalTrimData(
                trim_id="c" * 32,
                kind=TRIM_KIND_CORNICE,
                wall_surface_ids=(surfaces[0].surface_id,),
                height_meters=4.0,
            )
        )

        geometry = build_architectural_trim_geometry([level], surfaces)
        target = build_architectural_trim_edit_targets([level], surfaces)[0]
        height_handle = next(
            handle for handle in target.handles if handle.handle_kind == "height"
        )

        np.testing.assert_allclose(height_handle.axis_world, (0.0, 0.0, -1.0))
        self.assertAlmostEqual(height_handle.origin_world[2], 0.0)
        self.assertAlmostEqual(height_handle.value_meters, 3.0)
        np.testing.assert_allclose(geometry.runs[0].mesh.bounds[:, 2], (0.0, 3.0))

    def test_edging_strip_builds_selectable_parts_and_edit_handles(self) -> None:
        level, surfaces = _corner_level_and_surfaces()
        level.architectural_trims.append(
            ArchitecturalTrimData(
                trim_id="5" * 32,
                kind=TRIM_KIND_EDGING_STRIP,
                wall_surface_ids=tuple(surface.surface_id for surface in surfaces),
                corner_vertex_id=2,
                height_meters=2.5,
            )
        )

        geometry = build_architectural_trim_geometry([level], surfaces)
        targets = build_architectural_trim_edit_targets([level], surfaces)

        self.assertEqual(len(geometry.runs), 1)
        self.assertTrue(geometry.runs[0].mesh.is_watertight)
        self.assertEqual(
            {part.part_kind for part in geometry.parts},
            {"front", "top", "bottom", "sides"},
        )
        self.assertEqual(
            {handle.handle_kind for handle in targets[0].handles},
            {"width", "height", "depth"},
        )
        handles = {
            handle.handle_kind: np.asarray(handle.axis_world, dtype=float)
            for handle in targets[0].handles
        }
        np.testing.assert_allclose(
            handles["depth"],
            np.asarray((1.0, -1.0, 0.0)) / np.sqrt(2.0),
        )
        self.assertAlmostEqual(float(np.dot(handles["width"], handles["depth"])), 0.0)

    def test_linear_trim_gizmos_follow_a_diagonal_host_wall(self) -> None:
        level, _surfaces = _corner_level_and_surfaces()
        diagonal = _wall_surface(
            "level:2/wall:1:2",
            (0.0, 0.0, 0.0),
            (2.0, 2.0, 0.0),
        )
        level.architectural_trims.append(
            ArchitecturalTrimData(
                trim_id="d" * 32,
                kind=TRIM_KIND_SKIRTING_BOARD,
                wall_surface_ids=(diagonal.surface_id,),
            )
        )

        target = build_architectural_trim_edit_targets([level], (diagonal,))[0]
        handles = {
            handle.handle_kind: np.asarray(handle.axis_world, dtype=float)
            for handle in target.handles
        }

        np.testing.assert_allclose(
            handles["depth"],
            np.asarray((1.0, -1.0, 0.0)) / np.sqrt(2.0),
        )
        self.assertAlmostEqual(float(np.dot(handles["depth"], (1.0, 1.0, 0.0))), 0.0)

    def test_non_right_angle_linear_chain_is_rejected(self) -> None:
        level, surfaces = _corner_level_and_surfaces()
        diagonal = _wall_surface(
            "level:2/wall:2:3",
            (2.0, 0.0, 0.0),
            (3.0, 1.0, 0.0),
        )
        request = ArchitecturalTrimPlacementRequest(
            kind=TRIM_KIND_SKIRTING_BOARD,
            wall_surface_ids=(surfaces[0].surface_id, diagonal.surface_id),
        )
        with self.assertRaisesRegex(ValueError, "90-degree"):
            add_architectural_trim(level, request, (surfaces[0], diagonal))

    def test_semantic_id_helpers_are_strict_and_round_trip(self) -> None:
        surface_id = build_architectural_trim_surface_id(
            "6" * 32,
            TRIM_PART_FRONT,
        )
        self.assertEqual(surface_id, f"trim:{'6' * 32}/part:front:wall")
        self.assertTrue(is_architectural_trim_surface_id(surface_id))
        self.assertFalse(is_architectural_trim_surface_id("trim:bad/part:front:wall"))


if __name__ == "__main__":
    unittest.main()
