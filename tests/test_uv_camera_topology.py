# ### Imports ###
from __future__ import annotations

from collections import defaultdict
from io import BytesIO
import math
import unittest
from unittest.mock import patch

import numpy as np
import trimesh
from shapely.geometry import Polygon
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.camera_uv_projection import (
    CameraUvProjectionCancelled,
    CameraUvProjectionOptions,
    project_uvs_from_camera_views_from_glb,
)


# ### Topology constants ###
SMOOTH_NORMAL_COSINE = math.cos(math.radians(10.0))


# ### Fixture helpers ###
def _scene_glb(*meshes: tuple[str, trimesh.Trimesh]) -> bytes:
    scene = trimesh.Scene()
    for name, mesh in meshes:
        scene.add_geometry(mesh, geom_name=name, node_name=name)
    return bytes(scene.export(file_type="glb"))


def _expanded_mesh(
    triangle_points: np.ndarray,
    triangle_normals: np.ndarray,
    *,
    material: PBRMaterial | None = None,
    collapsed_source_uv: bool = False,
) -> trimesh.Trimesh:
    triangles = np.asarray(triangle_points, dtype=float)
    normals = np.asarray(triangle_normals, dtype=float)
    vertices = triangles.reshape((-1, 3))
    vertex_normals = normals.reshape((-1, 3))
    faces = np.arange(len(vertices), dtype=np.int64).reshape((-1, 3))
    visual = None
    if material is not None or collapsed_source_uv:
        visual = TextureVisuals(
            uv=np.zeros((len(vertices), 2), dtype=float),
            material=material,
        )
    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=vertex_normals,
        visual=visual,
        process=False,
    )


def _unwelded_grid_plane_glb(grid_size: int = 4) -> bytes:
    triangles: list[np.ndarray] = []
    for row in range(grid_size):
        for column in range(grid_size):
            lower_left = np.asarray((column, row, 0.0), dtype=float)
            lower_right = np.asarray((column + 1.0, row, 0.0), dtype=float)
            upper_right = np.asarray((column + 1.0, row + 1.0, 0.0), dtype=float)
            upper_left = np.asarray((column, row + 1.0, 0.0), dtype=float)
            triangles.extend(
                (
                    np.vstack((lower_left, lower_right, upper_right)),
                    np.vstack((lower_left, upper_right, upper_left)),
                )
            )
    triangle_array = np.asarray(triangles, dtype=float)
    normal_array = np.zeros_like(triangle_array)
    normal_array[:, :, 2] = 1.0
    material = PBRMaterial(
        name="plane_material",
        baseColorFactor=np.asarray((0.25, 0.5, 0.75, 1.0), dtype=float),
    )
    mesh = _expanded_mesh(
        triangle_array,
        normal_array,
        material=material,
        collapsed_source_uv=True,
    )
    return _scene_glb(("unwelded_grid", mesh))


def _unwelded_flat_cube_glb() -> bytes:
    indexed_cube = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    triangle_array = np.asarray(
        indexed_cube.vertices[indexed_cube.faces],
        dtype=float,
    )
    normal_array = np.repeat(
        np.asarray(indexed_cube.face_normals, dtype=float)[:, np.newaxis, :],
        3,
        axis=1,
    )
    mesh = _expanded_mesh(triangle_array, normal_array)
    return _scene_glb(("unwelded_flat_cube", mesh))


def _authored_normal_seam_plane_glb() -> bytes:
    triangle_array = np.asarray(
        (
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
            ((0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        ),
        dtype=float,
    )
    first_normal = np.asarray((0.0, 0.0, 1.0), dtype=float)
    angle = math.radians(10.0)
    second_normal = np.asarray((0.0, math.sin(angle), math.cos(angle)))
    normal_array = np.asarray(
        (
            np.repeat(first_normal[np.newaxis, :], 3, axis=0),
            np.repeat(second_normal[np.newaxis, :], 3, axis=0),
        ),
        dtype=float,
    )
    mesh = _expanded_mesh(triangle_array, normal_array)
    return _scene_glb(("authored_normal_seam", mesh))


def _unwelded_smooth_icosphere_glb() -> bytes:
    indexed_sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    triangle_array = np.asarray(
        indexed_sphere.vertices[indexed_sphere.faces],
        dtype=float,
    )
    normal_array = triangle_array.copy()
    normal_array /= np.linalg.norm(normal_array, axis=2)[:, :, np.newaxis]
    mesh = _expanded_mesh(triangle_array, normal_array)
    return _scene_glb(("unwelded_smooth_icosphere", mesh))


def _corrugated_camera_boundary_plane_glb(
    column_count: int = 12,
    row_count: int = 4,
) -> bytes:
    column_heights = [0.0]
    for column in range(column_count):
        slope = 0.92 if column % 2 == 0 else 1.08
        column_heights.append(column_heights[-1] + slope)
    house_triangles: list[np.ndarray] = []
    for row in range(row_count):
        for column in range(column_count):
            lower_left = np.asarray(
                (column, row, column_heights[column]),
                dtype=float,
            )
            lower_right = np.asarray(
                (column + 1.0, row, column_heights[column + 1]),
                dtype=float,
            )
            upper_right = lower_right + np.asarray((0.0, 1.0, 0.0))
            upper_left = lower_left + np.asarray((0.0, 1.0, 0.0))
            house_triangles.extend(
                (
                    np.vstack((lower_left, lower_right, upper_right)),
                    np.vstack((lower_left, upper_right, upper_left)),
                )
            )
    house_triangle_array = np.asarray(house_triangles, dtype=float)
    first_edges = house_triangle_array[:, 1] - house_triangle_array[:, 0]
    second_edges = house_triangle_array[:, 2] - house_triangle_array[:, 0]
    house_face_normals = np.cross(first_edges, second_edges)
    house_face_normals /= np.linalg.norm(
        house_face_normals,
        axis=1,
    )[:, np.newaxis]
    house_normal_array = np.repeat(
        house_face_normals[:, np.newaxis, :],
        3,
        axis=1,
    )

    gltf_triangle_array = house_triangle_array[:, :, (0, 2, 1)].copy()
    gltf_triangle_array[:, :, 2] *= -1.0
    gltf_normal_array = house_normal_array[:, :, (0, 2, 1)].copy()
    gltf_normal_array[:, :, 2] *= -1.0
    mesh = _expanded_mesh(gltf_triangle_array, gltf_normal_array)
    return _scene_glb(("corrugated_boundary_plane", mesh))


def _folded_overlapping_ribbon_glb() -> bytes:
    ribbon_path = np.asarray(
        (
            (0.0, 2.0),
            (2.0, 2.0),
            (1.0, 1.0),
            (3.0, 1.0),
            (2.1, 0.0),
            (4.1, 0.0),
        ),
        dtype=float,
    )
    house_triangles: list[np.ndarray] = []
    for segment_index in range(len(ribbon_path) - 1):
        first_x, first_z = ribbon_path[segment_index]
        second_x, second_z = ribbon_path[segment_index + 1]
        lower_left = np.asarray((first_x, 0.0, first_z), dtype=float)
        lower_right = np.asarray((second_x, 0.0, second_z), dtype=float)
        upper_right = np.asarray((second_x, 1.0, second_z), dtype=float)
        upper_left = np.asarray((first_x, 1.0, first_z), dtype=float)
        house_triangles.extend(
            (
                np.vstack((lower_left, lower_right, upper_right)),
                np.vstack((lower_left, upper_right, upper_left)),
            )
        )
    house_triangle_array = np.asarray(house_triangles, dtype=float)
    first_edges = house_triangle_array[:, 1] - house_triangle_array[:, 0]
    second_edges = house_triangle_array[:, 2] - house_triangle_array[:, 0]
    house_face_normals = np.cross(first_edges, second_edges)
    house_face_normals /= np.linalg.norm(
        house_face_normals,
        axis=1,
    )[:, np.newaxis]
    house_normal_array = np.repeat(
        house_face_normals[:, np.newaxis, :],
        3,
        axis=1,
    )
    gltf_triangle_array = house_triangle_array[:, :, (0, 2, 1)].copy()
    gltf_triangle_array[:, :, 2] *= -1.0
    gltf_normal_array = house_normal_array[:, :, (0, 2, 1)].copy()
    gltf_normal_array[:, :, 2] *= -1.0
    mesh = _expanded_mesh(gltf_triangle_array, gltf_normal_array)
    return _scene_glb(("folded_overlapping_ribbon", mesh))


def _stacked_overlapping_panels_glb(
    panel_count: int = 6,
) -> bytes:
    meshes: list[tuple[str, trimesh.Trimesh]] = []
    panel_width = max(2.5, panel_count * 0.2 + 0.5)
    for panel_index in range(panel_count):
        minimum_x = panel_index * 0.2
        maximum_x = minimum_x + panel_width
        height = float(panel_count - panel_index)
        house_triangles = np.asarray(
            (
                (
                    (minimum_x, 0.0, height),
                    (maximum_x, 0.0, height),
                    (maximum_x, 2.0, height),
                ),
                (
                    (minimum_x, 0.0, height),
                    (maximum_x, 2.0, height),
                    (minimum_x, 2.0, height),
                ),
            ),
            dtype=float,
        )
        house_normals = np.zeros_like(house_triangles)
        house_normals[:, :, 2] = 1.0
        gltf_triangles = house_triangles[:, :, (0, 2, 1)].copy()
        gltf_triangles[:, :, 2] *= -1.0
        gltf_normals = house_normals[:, :, (0, 2, 1)].copy()
        gltf_normals[:, :, 2] *= -1.0
        meshes.append(
            (
                f"stacked_panel_{panel_index}",
                _expanded_mesh(gltf_triangles, gltf_normals),
            )
        )
    return _scene_glb(*meshes)


def _adjacent_material_planes_glb() -> bytes:
    first_points = np.asarray(
        (
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
            ((0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        ),
        dtype=float,
    )
    second_points = first_points.copy()
    second_points[:, :, 0] += 1.0
    normals = np.zeros_like(first_points)
    normals[:, :, 2] = 1.0
    red = PBRMaterial(
        name="red_material",
        baseColorFactor=np.asarray((1.0, 0.0, 0.0, 1.0), dtype=float),
    )
    green = PBRMaterial(
        name="green_material",
        baseColorFactor=np.asarray((0.0, 1.0, 0.0, 1.0), dtype=float),
    )
    return _scene_glb(
        ("red_plane", _expanded_mesh(first_points, normals, material=red)),
        ("green_plane", _expanded_mesh(second_points, normals, material=green)),
    )


# ### Coordinate-welded topology helpers ###
def _coordinate_key(point: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(value) for value in np.round(point, 7))


def _uv_key(point: np.ndarray) -> tuple[float, float]:
    return tuple(float(value) for value in np.round(point, 9))


def _normal_key(normal: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(value) for value in np.round(normal, 6))


def _position_normal_signature(
    mesh: trimesh.Trimesh,
) -> list[tuple[tuple[float, ...], tuple[float, ...]]]:
    return sorted(
        (
            tuple(float(value) for value in np.round(vertex, 7)),
            tuple(float(value) for value in np.round(normal, 6)),
        )
        for vertex, normal in zip(
            np.asarray(mesh.vertices, dtype=float),
            np.asarray(mesh.vertex_normals, dtype=float),
        )
    )


def _coordinate_edge_records(
    mesh: trimesh.Trimesh,
) -> dict[
    tuple[tuple[float, float, float], tuple[float, float, float]],
    list[
        tuple[
            int,
            dict[tuple[float, float, float], tuple[float, float]],
            dict[tuple[float, float, float], tuple[float, float, float]],
        ]
    ],
]:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    uv = np.asarray(mesh.visual.uv, dtype=float)
    normals = np.asarray(mesh.vertex_normals, dtype=float)
    records = defaultdict(list)
    for face_index, face in enumerate(faces):
        for corner_index in range(3):
            first_index = int(face[corner_index])
            second_index = int(face[(corner_index + 1) % 3])
            first_position = _coordinate_key(vertices[first_index])
            second_position = _coordinate_key(vertices[second_index])
            edge_key = tuple(sorted((first_position, second_position)))
            records[edge_key].append(
                (
                    face_index,
                    {
                        first_position: _uv_key(uv[first_index]),
                        second_position: _uv_key(uv[second_index]),
                    },
                    {
                        first_position: _normal_key(normals[first_index]),
                        second_position: _normal_key(normals[second_index]),
                    },
                )
            )
    return records


def _edge_normals_are_compatible(first: dict, second: dict) -> bool:
    if first.keys() != second.keys():
        return False
    return all(
        float(np.dot(first[position], second[position]))
        >= SMOOTH_NORMAL_COSINE
        for position in first
    )


def _compatible_shared_edge_counts(mesh: trimesh.Trimesh) -> tuple[int, int]:
    shared_count = 0
    seam_count = 0
    for records in _coordinate_edge_records(mesh).values():
        if len(records) != 2:
            continue
        first, second = records
        if not _edge_normals_are_compatible(first[2], second[2]):
            continue
        shared_count += 1
        if first[1] != second[1]:
            seam_count += 1
    return shared_count, seam_count


def _coordinate_uv_island_count(mesh: trimesh.Trimesh) -> int:
    parent = list(range(len(mesh.faces)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for records in _coordinate_edge_records(mesh).values():
        if len(records) != 2:
            continue
        first, second = records
        if first[1] == second[1]:
            union(first[0], second[0])
    return len({find(index) for index in range(len(parent))})


def _index_face_component_count(mesh: trimesh.Trimesh) -> int:
    parent = list(range(len(mesh.faces)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    faces_by_vertex: dict[int, list[int]] = defaultdict(list)
    for face_index, face in enumerate(np.asarray(mesh.faces, dtype=np.int64)):
        for vertex_index in face:
            faces_by_vertex[int(vertex_index)].append(face_index)
    for face_indices in faces_by_vertex.values():
        first_face = face_indices[0]
        for face_index in face_indices[1:]:
            union(first_face, face_index)
    return len({find(index) for index in range(len(parent))})


# ### UV assertion helpers ###
def _assert_uv_triangles_are_valid_and_disjoint(
    test_case: unittest.TestCase,
    scene: trimesh.Scene,
) -> None:
    triangles: list[Polygon] = []
    for geometry in scene.geometry.values():
        if not isinstance(geometry, trimesh.Trimesh):
            continue
        uv = np.asarray(geometry.visual.uv, dtype=float)
        triangles.extend(Polygon(uv[face]) for face in geometry.faces)
    test_case.assertTrue(triangles)
    for triangle in triangles:
        test_case.assertTrue(triangle.is_valid)
        test_case.assertGreater(triangle.area, 0.0)
        minimum_x, minimum_y, maximum_x, maximum_y = triangle.bounds
        test_case.assertGreaterEqual(minimum_x, -1e-8)
        test_case.assertGreaterEqual(minimum_y, -1e-8)
        test_case.assertLessEqual(maximum_x, 1.0 + 1e-8)
        test_case.assertLessEqual(maximum_y, 1.0 + 1e-8)
    for first_index, first in enumerate(triangles):
        for second in triangles[first_index + 1 :]:
            test_case.assertLessEqual(
                float(first.intersection(second).area),
                1e-10,
            )


def _assert_fallback_reservation_is_bottom_left_and_disjoint(
    test_case: unittest.TestCase,
    chart_regions,
) -> None:
    fallback_regions = [region for region in chart_regions if region.is_leftover]
    primary_regions = [region for region in chart_regions if not region.is_leftover]
    test_case.assertTrue(fallback_regions)
    test_case.assertTrue(primary_regions)
    fallback_right = max(region.x + region.width for region in fallback_regions)
    fallback_top = max(region.y + region.height for region in fallback_regions)
    test_case.assertEqual(min(region.x for region in fallback_regions), 0)
    test_case.assertEqual(min(region.y for region in fallback_regions), 0)
    for region in primary_regions:
        test_case.assertTrue(
            region.x >= fallback_right or region.y >= fallback_top,
            msg=(
                f"Primary chart {region.chart_index} overlaps the bottom-left "
                "fallback reservation."
            ),
        )


def _only_mesh(scene: trimesh.Scene) -> trimesh.Trimesh:
    meshes = [
        geometry
        for geometry in scene.geometry.values()
        if isinstance(geometry, trimesh.Trimesh)
    ]
    if len(meshes) != 1:
        raise AssertionError(f"Expected one mesh, found {len(meshes)}.")
    return meshes[0]


# ### Topology-aware projection tests ###
class CameraUvTopologyTests(unittest.TestCase):
    def test_unwelded_tessellated_plane_becomes_one_coherent_uv_island(
        self,
    ) -> None:
        result = project_uvs_from_camera_views_from_glb(
            _unwelded_grid_plane_glb(),
            options=CameraUvProjectionOptions(capture_image_size=128),
        )
        output_mesh = _only_mesh(result.model.scene)

        self.assertEqual(len(output_mesh.faces), 32)
        shared_count, seam_count = _compatible_shared_edge_counts(output_mesh)
        self.assertEqual(shared_count, 40)
        self.assertEqual(seam_count, 0)
        self.assertEqual(_coordinate_uv_island_count(output_mesh), 1)
        self.assertEqual(_index_face_component_count(output_mesh), 1)
        self.assertEqual(len(output_mesh.vertices), 25)
        self.assertEqual(result.output_vertex_count, 25)
        self.assertEqual(len(result.chart_regions), 1)
        _assert_uv_triangles_are_valid_and_disjoint(self, result.model.scene)

    def test_corrugated_camera_boundary_avoids_checkerboard_islands(self) -> None:
        result = project_uvs_from_camera_views_from_glb(
            _corrugated_camera_boundary_plane_glb(),
            options=CameraUvProjectionOptions(capture_image_size=256),
        )
        output_mesh = _only_mesh(result.model.scene)

        island_count = _coordinate_uv_island_count(output_mesh)
        shared_count, seam_count = _compatible_shared_edge_counts(output_mesh)
        self.assertEqual(shared_count, 128)
        self.assertEqual(seam_count, 0)
        self.assertEqual(island_count, 1)
        self.assertEqual(len(result.chart_regions), 1)
        self.assertEqual(
            sum(count > 0 for count in result.camera_face_counts.values()),
            1,
        )
        self.assertEqual(result.leftover_face_count, 0)
        _assert_uv_triangles_are_valid_and_disjoint(self, result.model.scene)

    def test_folded_overlap_groups_conflict_free_primary_layers(self) -> None:
        result = project_uvs_from_camera_views_from_glb(
            _folded_overlapping_ribbon_glb(),
            options=CameraUvProjectionOptions(capture_image_size=256),
        )
        output_mesh = _only_mesh(result.model.scene)

        island_count = _coordinate_uv_island_count(output_mesh)
        self.assertEqual(island_count, 5)
        self.assertEqual(result.invisible_face_count, 0)
        self.assertEqual(result.quality_fallback_face_count, 0)
        self.assertEqual(result.conflict_fallback_face_count, 0)
        self.assertEqual(result.leftover_face_count, 0)
        self.assertEqual(result.camera_face_counts["top"], 2)
        self.assertEqual(result.camera_face_counts["pos_x"], 4)
        self.assertEqual(result.camera_face_counts["bottom"], 4)
        for camera_id in ("top", "bottom", "pos_x"):
            self.assertEqual(
                sum(
                    not region.is_leftover and region.camera_id == camera_id
                    for region in result.chart_regions
                ),
                1,
            )
        _assert_uv_triangles_are_valid_and_disjoint(self, result.model.scene)

    def test_conflict_layers_over_budget_move_to_bottom_left_fallback(self) -> None:
        result = project_uvs_from_camera_views_from_glb(
            _stacked_overlapping_panels_glb(),
            options=CameraUvProjectionOptions(capture_image_size=256),
        )
        primary_regions = [
            region for region in result.chart_regions if not region.is_leftover
        ]
        fallback_regions = [
            region for region in result.chart_regions if region.is_leftover
        ]

        self.assertEqual(result.original_face_count, 12)
        self.assertEqual(result.invisible_face_count, 0)
        self.assertEqual(result.quality_fallback_face_count, 0)
        self.assertEqual(result.conflict_fallback_face_count, 8)
        self.assertEqual(
            result.leftover_face_count,
            result.invisible_face_count
            + result.quality_fallback_face_count
            + result.conflict_fallback_face_count,
        )
        self.assertEqual(
            result.original_face_count,
            sum(result.camera_face_counts.values()) + result.leftover_face_count,
        )
        self.assertEqual(len(primary_regions), 2)
        self.assertEqual(
            {region.camera_id for region in primary_regions},
            {"top", "bottom"},
        )
        self.assertEqual(result.camera_face_counts["top"], 2)
        self.assertEqual(result.camera_face_counts["bottom"], 2)
        self.assertEqual(
            sum(region.face_count for region in fallback_regions),
            result.leftover_face_count,
        )
        output_meshes = [
            geometry
            for geometry in result.model.scene.geometry.values()
            if isinstance(geometry, trimesh.Trimesh)
        ]
        output_island_count = sum(
            _coordinate_uv_island_count(mesh) for mesh in output_meshes
        )
        self.assertGreaterEqual(output_island_count, 6)
        self.assertLessEqual(output_island_count, 8)
        _assert_fallback_reservation_is_bottom_left_and_disjoint(
            self,
            result.chart_regions,
        )
        _assert_uv_triangles_are_valid_and_disjoint(self, result.model.scene)

    def test_cancellation_is_polled_during_invisible_face_layout(self) -> None:
        assigning_complete = False
        post_assign_poll_count = 0

        def report_progress(progress) -> None:
            nonlocal assigning_complete
            if (
                progress.stage == "assigning"
                and progress.completed_face_count == progress.total_face_count
            ):
                assigning_complete = True

        def cancel_during_fallback_layout() -> bool:
            nonlocal post_assign_poll_count
            if not assigning_complete:
                return False
            post_assign_poll_count += 1
            return post_assign_poll_count >= 2

        with (
            patch(
                "housemaker.camera_uv_projection.capture_visible_face_indices",
                return_value=frozenset(),
            ),
            patch(
                "housemaker.camera_uv_projection._pack_charts",
                side_effect=AssertionError(
                    "Packing was reached before fallback-layout cancellation."
                ),
            ),
        ):
            with self.assertRaises(CameraUvProjectionCancelled):
                project_uvs_from_camera_views_from_glb(
                    _unwelded_grid_plane_glb(),
                    options=CameraUvProjectionOptions(capture_image_size=128),
                    cancel_requested=cancel_during_fallback_layout,
                    progress_callback=report_progress,
                )

    def test_unwelded_flat_cube_keeps_hard_edges_and_joins_each_flat_side(
        self,
    ) -> None:
        result = project_uvs_from_camera_views_from_glb(
            _unwelded_flat_cube_glb(),
            options=CameraUvProjectionOptions(capture_image_size=128),
        )
        output_mesh = _only_mesh(result.model.scene)

        shared_count, seam_count = _compatible_shared_edge_counts(output_mesh)
        self.assertEqual(shared_count, 6)
        self.assertEqual(seam_count, 0)
        self.assertEqual(_coordinate_uv_island_count(output_mesh), 6)
        self.assertEqual(_index_face_component_count(output_mesh), 6)
        self.assertEqual(len(output_mesh.vertices), 24)
        self.assertEqual(result.output_vertex_count, 24)
        self.assertEqual(len(result.chart_regions), 6)
        for corner in np.unique(np.asarray(output_mesh.vertices), axis=0):
            matching = np.all(
                np.isclose(output_mesh.vertices, corner, atol=1e-7),
                axis=1,
            )
            corner_normals = {
                _normal_key(normal)
                for normal in np.asarray(output_mesh.vertex_normals)[matching]
            }
            self.assertEqual(len(corner_normals), 3)
        _assert_uv_triangles_are_valid_and_disjoint(self, result.model.scene)

    def test_smooth_icosphere_forms_six_coherent_camera_regions(self) -> None:
        result = project_uvs_from_camera_views_from_glb(
            _unwelded_smooth_icosphere_glb(),
            options=CameraUvProjectionOptions(capture_image_size=256),
        )
        output_mesh = _only_mesh(result.model.scene)

        shared_count, seam_count = _compatible_shared_edge_counts(output_mesh)
        self.assertEqual(shared_count, 480)
        self.assertLessEqual(seam_count, 96)
        self.assertEqual(_coordinate_uv_island_count(output_mesh), 6)
        self.assertEqual(len(result.chart_regions), 6)
        self.assertTrue(all(result.camera_face_counts.values()))
        self.assertEqual(result.leftover_face_count, 0)
        _assert_uv_triangles_are_valid_and_disjoint(self, result.model.scene)

    def test_authored_normal_seam_is_not_welded_away(self) -> None:
        source_glb = _authored_normal_seam_plane_glb()
        source_scene = trimesh.load(
            BytesIO(source_glb),
            file_type="glb",
            force="scene",
            process=False,
        )
        source_mesh = _only_mesh(source_scene)

        result = project_uvs_from_camera_views_from_glb(
            source_glb,
            options=CameraUvProjectionOptions(capture_image_size=128),
        )
        output_mesh = _only_mesh(result.model.scene)

        self.assertEqual(
            _position_normal_signature(output_mesh),
            _position_normal_signature(source_mesh),
        )
        self.assertEqual(len(output_mesh.vertices), 6)
        self.assertEqual(_index_face_component_count(output_mesh), 2)
        self.assertEqual(_coordinate_uv_island_count(output_mesh), 1)
        self.assertEqual(len(result.chart_regions), 1)
        _assert_uv_triangles_are_valid_and_disjoint(self, result.model.scene)

    def test_adjacent_material_primitives_remain_distinct(self) -> None:
        result = project_uvs_from_camera_views_from_glb(
            _adjacent_material_planes_glb(),
            options=CameraUvProjectionOptions(capture_image_size=128),
        )
        output_meshes = [
            geometry
            for geometry in result.model.scene.geometry.values()
            if isinstance(geometry, trimesh.Trimesh)
        ]

        self.assertEqual(len(output_meshes), 2)
        material_names = {
            str(getattr(mesh.visual.material, "name", ""))
            for mesh in output_meshes
        }
        self.assertEqual(material_names, {"red_material", "green_material"})
        base_colors = {
            tuple(
                int(value)
                for value in np.asarray(
                    mesh.visual.material.baseColorFactor,
                    dtype=np.uint8,
                )
            )
            for mesh in output_meshes
        }
        self.assertEqual(base_colors, {(255, 0, 0, 255), (0, 255, 0, 255)})
        _assert_uv_triangles_are_valid_and_disjoint(self, result.model.scene)


if __name__ == "__main__":
    unittest.main()
