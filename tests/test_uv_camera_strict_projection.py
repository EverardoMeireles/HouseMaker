# ### Imports ###
from __future__ import annotations

from collections import Counter
from io import BytesIO
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import trimesh
from shapely import STRtree
from shapely.geometry import Polygon

from housemaker.camera_uv_projection import (
    CameraUvProjectionCancelled,
    CameraUvProjectionOptions,
    _MeshTopology,
    _select_global_conflict_free_camera_assignments,
    _solve_camera_assignment_component_exactly,
    project_uvs_from_camera_views_from_glb,
)
from housemaker.glb import GLTF_Y_UP_TO_Z_UP_TRANSFORM
from housemaker.unused_face_removal import (
    ALL_CAMERA_IDS,
    get_fixed_camera_view,
)


# ### Strict-layout constants ###
BASE_TEXTURE_SIZE = 512
FALLBACK_SIDE_PIXELS = 128
MINIMUM_PRIMARY_CAMERA_QUALITY = 0.6
UV_TOLERANCE = 2e-6
OVERLAP_AREA_TOLERANCE = 1e-10
REAL_ARTIFACT_NAMES = (
    "c0e9802a87044d97869a238d5270507f.postprocessed.glb",
    "198f4e186d614ad3b7d433251a47a678.postprocessed.glb",
)
REAL_ARTIFACT_EXPECTATIONS = {
    REAL_ARTIFACT_NAMES[0]: {
        "minimum_primary": 900,
        "invisible": 36,
        "quality_fallback": 224,
    },
    REAL_ARTIFACT_NAMES[1]: {
        "minimum_primary": 800,
        "invisible": 47,
        "quality_fallback": 234,
    },
}


# ### Fixture helpers ###
def _z_up_to_gltf_points(points: np.ndarray) -> np.ndarray:
    converted = np.asarray(points, dtype=float)[..., (0, 2, 1)].copy()
    converted[..., 2] *= -1.0
    return converted


def _expanded_z_up_mesh(triangles: np.ndarray) -> trimesh.Trimesh:
    z_up_triangles = np.asarray(triangles, dtype=float)
    weighted_normals = np.cross(
        z_up_triangles[:, 1] - z_up_triangles[:, 0],
        z_up_triangles[:, 2] - z_up_triangles[:, 0],
    )
    weighted_normals /= np.linalg.norm(
        weighted_normals,
        axis=1,
    )[:, np.newaxis]
    z_up_normals = np.repeat(weighted_normals[:, np.newaxis, :], 3, axis=1)
    gltf_triangles = _z_up_to_gltf_points(z_up_triangles)
    gltf_normals = _z_up_to_gltf_points(z_up_normals)
    vertices = gltf_triangles.reshape((-1, 3))
    return trimesh.Trimesh(
        vertices=vertices,
        faces=np.arange(len(vertices), dtype=np.int64).reshape((-1, 3)),
        vertex_normals=gltf_normals.reshape((-1, 3)),
        process=False,
    )


def _scene_glb(meshes: list[tuple[str, trimesh.Trimesh]]) -> bytes:
    scene = trimesh.Scene()
    for name, mesh in meshes:
        scene.add_geometry(mesh, geom_name=name, node_name=name)
    return bytes(scene.export(file_type="glb"))


def _small_allocator_counterexample() -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[dict[int, set[int]], ...],
    _MeshTopology,
]:
    candidates = np.zeros((4, len(ALL_CAMERA_IDS)), dtype=bool)
    candidates[:, :3] = np.asarray(
        (
            (True, True, False),
            (False, True, True),
            (False, True, True),
            (True, True, False),
        ),
        dtype=bool,
    )
    quality = np.zeros_like(candidates, dtype=float)
    quality[:, :3] = np.asarray(
        (
            (0.697, 0.718, 0.0),
            (0.0, 0.671, 0.906),
            (0.0, 0.942, 0.868),
            (0.618, 0.628, 0.0),
        ),
        dtype=float,
    )
    conflicts = tuple(
        {index: set() for index in range(4)}
        for _camera_id in ALL_CAMERA_IDS
    )

    def connect(camera_index: int, first: int, second: int) -> None:
        conflicts[camera_index][first].add(second)
        conflicts[camera_index][second].add(first)

    connect(0, 0, 3)
    for first, second in ((0, 1), (0, 2), (0, 3), (2, 3)):
        connect(1, first, second)
    connect(2, 1, 2)
    topology = _MeshTopology(
        face_neighbors=((), (), (), ()),
        welded_vertex_ids_by_instance=(),
    )
    return candidates, quality, conflicts, topology


def _disconnected_top_patches_glb() -> bytes:
    triangles = np.asarray(
        (
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
            ((0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
            ((3.0, 2.0, 0.0), (5.0, 2.0, 0.0), (5.0, 3.0, 0.0)),
            ((3.0, 2.0, 0.0), (5.0, 3.0, 0.0), (3.0, 3.0, 0.0)),
        ),
        dtype=float,
    )
    return _scene_glb([("disconnected_top_patches", _expanded_z_up_mesh(triangles))])


def _stacked_top_panels_glb(panel_count: int = 6) -> bytes:
    triangles: list[np.ndarray] = []
    panel_width = max(2.5, panel_count * 0.2 + 0.5)
    for panel_index in range(panel_count):
        minimum_x = panel_index * 0.2
        maximum_x = minimum_x + panel_width
        height = float(panel_count - panel_index)
        triangles.extend(
            (
                np.asarray(
                    (
                        (minimum_x, 0.0, height),
                        (maximum_x, 0.0, height),
                        (maximum_x, 2.0, height),
                    )
                ),
                np.asarray(
                    (
                        (minimum_x, 0.0, height),
                        (maximum_x, 2.0, height),
                        (minimum_x, 2.0, height),
                    )
                ),
            )
        )
    return _scene_glb(
        [("stacked_top_panels", _expanded_z_up_mesh(np.asarray(triangles)))]
    )


def _nested_cubes_glb() -> bytes:
    meshes: list[tuple[str, trimesh.Trimesh]] = []
    for name, extent in (("outer", 4.0), ("inner", 1.0)):
        z_up_mesh = trimesh.creation.box(extents=(extent, extent, extent))
        z_up_mesh.apply_transform(
            np.asarray(
                (
                    (1.0, 0.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, -1.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
                dtype=float,
            )
        )
        meshes.append((name, z_up_mesh))
    return _scene_glb(meshes)


def _quality_adversary_glb() -> bytes:
    """Return a smooth ribbon spanning several preferred camera axes."""

    slopes = (0.0, 0.35, 0.72, 1.15, 1.8, 2.8, 4.2)
    path = [(0.0, 0.0)]
    for slope in slopes:
        previous_x, previous_z = path[-1]
        path.append((previous_x + 1.0, previous_z + slope))
    triangles: list[np.ndarray] = []
    for first, second in zip(path, path[1:]):
        first_x, first_z = first
        second_x, second_z = second
        triangles.extend(
            (
                np.asarray(
                    (
                        (first_x, 0.0, first_z),
                        (second_x, 0.0, second_z),
                        (second_x, 1.0, second_z),
                    )
                ),
                np.asarray(
                    (
                        (first_x, 0.0, first_z),
                        (second_x, 1.0, second_z),
                        (first_x, 1.0, first_z),
                    )
                ),
            )
        )
    return _scene_glb(
        [("quality_adversary", _expanded_z_up_mesh(np.asarray(triangles)))]
    )


def _best_camera_conflict_with_alternate_glb() -> bytes:
    first = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, -0.8),
            (0.0, 1.0, 0.0),
        ),
        dtype=float,
    )
    second = first + np.asarray((0.0, 0.0, 2.0), dtype=float)
    return _scene_glb(
        [
            (
                "best_camera_conflict_with_alternate",
                _expanded_z_up_mesh(np.asarray((first, second))),
            )
        ]
    )


# ### Output inspection helpers ###
def _iter_world_faces_with_uv(
    scene: trimesh.Scene,
):
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph.get(node_name)
        geometry = scene.geometry[geometry_name]
        if not isinstance(geometry, trimesh.Trimesh):
            continue
        gltf_world_vertices = trimesh.transform_points(
            np.asarray(geometry.vertices, dtype=float),
            np.asarray(transform, dtype=float),
        )
        z_up_world_vertices = trimesh.transform_points(
            gltf_world_vertices,
            GLTF_Y_UP_TO_Z_UP_TRANSFORM,
        )
        uv = np.asarray(geometry.visual.uv, dtype=float)
        for face in np.asarray(geometry.faces, dtype=np.int64):
            yield z_up_world_vertices[face], uv[face]


def _matching_region(uv_triangle: np.ndarray, regions):
    pixel_centroid = np.mean(uv_triangle, axis=0) * BASE_TEXTURE_SIZE
    matches = [
        region
        for region in regions
        if (
            region.x - UV_TOLERANCE
            <= pixel_centroid[0]
            <= region.x + region.width + UV_TOLERANCE
            and region.y - UV_TOLERANCE
            <= pixel_centroid[1]
            <= region.y + region.height + UV_TOLERANCE
        )
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"UV face centroid {pixel_centroid.tolist()} matched "
            f"{len(matches)} chart regions."
        )
    return matches[0]


def _faces_by_chart(result) -> dict[int, list[tuple[np.ndarray, np.ndarray]]]:
    faces_by_chart: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for points, uv in _iter_world_faces_with_uv(result.model.scene):
        region = _matching_region(uv, result.chart_regions)
        faces_by_chart.setdefault(region.chart_index, []).append((points, uv))
    return faces_by_chart


def _camera_projection_quality(points: np.ndarray, camera_id: str) -> float:
    weighted_normal = np.cross(points[1] - points[0], points[2] - points[0])
    best_axis_area = float(np.max(np.abs(weighted_normal)))
    if best_axis_area <= np.finfo(float).eps:
        return 0.0
    depth_axis = np.asarray(get_fixed_camera_view(camera_id).depth_axis, dtype=float)
    return abs(float(weighted_normal @ depth_axis)) / best_axis_area


def _camera_affine_fit(
    faces: list[tuple[np.ndarray, np.ndarray]],
    camera_id: str,
) -> tuple[np.ndarray, float]:
    view = get_fixed_camera_view(camera_id)
    horizontal_axis = np.asarray(view.horizontal_axis, dtype=float)
    vertical_axis = np.asarray(view.vertical_axis, dtype=float)
    projected_points: list[np.ndarray] = []
    uv_points: list[np.ndarray] = []
    for points, uv in faces:
        projected_points.extend(
            np.column_stack((points @ horizontal_axis, points @ vertical_axis))
        )
        uv_points.extend(uv)
    projected = np.asarray(projected_points, dtype=float)
    normalized_uv = np.asarray(uv_points, dtype=float)
    design = np.column_stack((projected, np.ones(len(projected))))
    coefficients, _residuals, _rank, _singular_values = np.linalg.lstsq(
        design,
        normalized_uv,
        rcond=None,
    )
    fitted = design @ coefficients
    maximum_error = float(np.max(np.abs(fitted - normalized_uv)))
    return coefficients, maximum_error


def _positive_uv_overlap_count(scene: trimesh.Scene) -> int:
    polygons = [
        Polygon(uv)
        for _points, uv in _iter_world_faces_with_uv(scene)
    ]
    tree = STRtree(polygons)
    overlap_count = 0
    for first_index, polygon in enumerate(polygons):
        for raw_second_index in tree.query(polygon):
            second_index = int(raw_second_index)
            if second_index <= first_index:
                continue
            if (
                float(polygon.intersection(polygons[second_index]).area)
                > OVERLAP_AREA_TOLERANCE
            ):
                overlap_count += 1
    return overlap_count


# ### Strict invariant assertions ###
def _assert_single_primary_chart_per_camera(
    test_case: unittest.TestCase,
    result,
) -> None:
    counts = Counter(
        region.camera_id
        for region in result.chart_regions
        if not region.is_leftover
    )
    test_case.assertTrue(set(counts).issubset(ALL_CAMERA_IDS))
    test_case.assertTrue(
        all(count <= 1 for count in counts.values()),
        msg=f"Multiple primary charts were emitted for one camera: {counts}",
    )


def _assert_fallback_accounting_and_containment(
    test_case: unittest.TestCase,
    result,
) -> None:
    fallback_regions = [
        region for region in result.chart_regions if region.is_leftover
    ]
    primary_regions = [
        region for region in result.chart_regions if not region.is_leftover
    ]
    test_case.assertEqual(
        result.leftover_face_count,
        result.invisible_face_count
        + result.quality_fallback_face_count
        + result.conflict_fallback_face_count,
    )
    test_case.assertEqual(
        result.leftover_face_count,
        sum(region.face_count for region in fallback_regions),
    )
    test_case.assertEqual(
        result.projected_face_count,
        sum(region.face_count for region in primary_regions),
    )
    for region in fallback_regions:
        test_case.assertGreaterEqual(region.x, 0)
        test_case.assertGreaterEqual(region.y, 0)
        test_case.assertLessEqual(
            region.x + region.width,
            FALLBACK_SIDE_PIXELS,
        )
        test_case.assertLessEqual(
            region.y + region.height,
            FALLBACK_SIDE_PIXELS,
        )
    if fallback_regions:
        for region in primary_regions:
            test_case.assertTrue(
                region.x >= FALLBACK_SIDE_PIXELS
                or region.y >= FALLBACK_SIDE_PIXELS
            )
    faces_by_chart = _faces_by_chart(result)
    for region in fallback_regions:
        for _points, uv in faces_by_chart.get(region.chart_index, ()):
            test_case.assertGreaterEqual(float(np.min(uv)), -UV_TOLERANCE)
            test_case.assertLessEqual(
                float(np.max(uv)),
                FALLBACK_SIDE_PIXELS / BASE_TEXTURE_SIZE + UV_TOLERANCE,
            )


def _assert_primary_projection_geometry(
    test_case: unittest.TestCase,
    result,
) -> None:
    faces_by_chart = _faces_by_chart(result)
    for region in result.chart_regions:
        if region.is_leftover:
            continue
        chart_faces = faces_by_chart[region.chart_index]
        coefficients, maximum_error = _camera_affine_fit(
            chart_faces,
            region.camera_id,
        )
        test_case.assertLessEqual(maximum_error, UV_TOLERANCE)
        horizontal_scale = float(coefficients[0, 0])
        vertical_scale = float(coefficients[1, 1])
        test_case.assertGreater(horizontal_scale, 0.0)
        test_case.assertGreater(vertical_scale, 0.0)
        test_case.assertAlmostEqual(
            horizontal_scale,
            vertical_scale,
            delta=UV_TOLERANCE,
        )
        test_case.assertAlmostEqual(
            float(coefficients[0, 1]),
            0.0,
            delta=UV_TOLERANCE,
        )
        test_case.assertAlmostEqual(
            float(coefficients[1, 0]),
            0.0,
            delta=UV_TOLERANCE,
        )
        for points, _uv in chart_faces:
            quality = _camera_projection_quality(points, region.camera_id)
            test_case.assertGreaterEqual(
                quality + UV_TOLERANCE,
                MINIMUM_PRIMARY_CAMERA_QUALITY,
                msg=(
                    f"Primary camera {region.camera_id} accepted face quality "
                    f"{quality:.6f}."
                ),
            )


def _assert_all_strict_invariants(test_case: unittest.TestCase, result) -> None:
    _assert_single_primary_chart_per_camera(test_case, result)
    _assert_fallback_accounting_and_containment(test_case, result)
    _assert_primary_projection_geometry(test_case, result)
    test_case.assertEqual(_positive_uv_overlap_count(result.model.scene), 0)


# ### Strict camera-projection tests ###
class CameraUvStrictProjectionTests(unittest.TestCase):
    def test_disconnected_patches_share_one_rigid_full_camera_frame(self) -> None:
        def top_visibility(
            _vertices,
            _faces,
            camera_id,
            **_options,
        ) -> frozenset[int]:
            return frozenset(range(4)) if camera_id == "top" else frozenset()

        with patch(
            "housemaker.camera_uv_projection.capture_visible_face_indices",
            side_effect=top_visibility,
        ):
            result = project_uvs_from_camera_views_from_glb(
                _disconnected_top_patches_glb(),
                options=CameraUvProjectionOptions(capture_image_size=128),
            )

        primary_regions = [
            region for region in result.chart_regions if not region.is_leftover
        ]
        self.assertEqual(len(primary_regions), 1)
        self.assertEqual(primary_regions[0].face_count, 4)
        self.assertEqual(result.leftover_face_count, 0)
        _assert_all_strict_invariants(self, result)

    def test_secondary_projection_layers_are_entirely_fallback(self) -> None:
        def top_visibility(
            _vertices,
            _faces,
            camera_id,
            **_options,
        ) -> frozenset[int]:
            return frozenset(range(12)) if camera_id == "top" else frozenset()

        with patch(
            "housemaker.camera_uv_projection.capture_visible_face_indices",
            side_effect=top_visibility,
        ):
            result = project_uvs_from_camera_views_from_glb(
                _stacked_top_panels_glb(),
                options=CameraUvProjectionOptions(capture_image_size=256),
            )

        self.assertEqual(result.invisible_face_count, 0)
        self.assertEqual(result.quality_fallback_face_count, 0)
        self.assertEqual(result.conflict_fallback_face_count, 10)
        self.assertEqual(result.projected_face_count, 2)
        _assert_all_strict_invariants(self, result)

    def test_invisible_faces_are_entirely_fallback(self) -> None:
        result = project_uvs_from_camera_views_from_glb(
            _nested_cubes_glb(),
            options=CameraUvProjectionOptions(capture_image_size=256),
        )

        self.assertEqual(result.invisible_face_count, 12)
        self.assertEqual(result.quality_fallback_face_count, 0)
        self.assertEqual(result.conflict_fallback_face_count, 0)
        _assert_all_strict_invariants(self, result)

    def test_primary_faces_respect_camera_quality_floor(self) -> None:
        result = project_uvs_from_camera_views_from_glb(
            _quality_adversary_glb(),
            options=CameraUvProjectionOptions(capture_image_size=256),
        )

        self.assertGreater(result.projected_face_count, 0)
        _assert_all_strict_invariants(self, result)

    def test_global_assignment_uses_alternate_camera_before_fallback(self) -> None:
        def visible_faces_for_camera(
            _vertices,
            _faces,
            camera_id,
            **_options,
        ) -> frozenset[int]:
            if camera_id in {"top", "pos_x"}:
                return frozenset((0, 1))
            return frozenset()

        with patch(
            "housemaker.camera_uv_projection.capture_visible_face_indices",
            side_effect=visible_faces_for_camera,
        ):
            result = project_uvs_from_camera_views_from_glb(
                _best_camera_conflict_with_alternate_glb(),
                options=CameraUvProjectionOptions(capture_image_size=128),
            )

        self.assertEqual(result.original_face_count, 2)
        self.assertEqual(result.projected_face_count, 2)
        self.assertEqual(result.leftover_face_count, 0)
        self.assertEqual(result.conflict_fallback_face_count, 0)
        self.assertGreaterEqual(result.camera_face_counts["pos_x"], 1)
        self.assertEqual(
            result.camera_face_counts["top"]
            + result.camera_face_counts["pos_x"],
            2,
        )
        _assert_all_strict_invariants(self, result)

    def test_small_conflict_component_is_recolored_to_maximum_coverage(
        self,
    ) -> None:
        candidates, quality, conflicts, topology = (
            _small_allocator_counterexample()
        )

        assignment = _select_global_conflict_free_camera_assignments(
            candidates,
            quality,
            conflicts,
            topology,
            None,
            1,
        )

        self.assertEqual(np.count_nonzero(assignment >= 0), 4)
        for camera_index, camera_conflicts in enumerate(conflicts):
            for face_index, neighbors in camera_conflicts.items():
                for neighbor_index in neighbors:
                    self.assertFalse(
                        assignment[face_index] == camera_index
                        and assignment[neighbor_index] == camera_index
                    )

    def test_exact_component_search_polls_cancellation(self) -> None:
        candidates, quality, conflicts, _topology = (
            _small_allocator_counterexample()
        )
        incumbent = np.asarray((-1, 2, 1, 0), dtype=np.int8)

        with self.assertRaises(CameraUvProjectionCancelled):
            _solve_camera_assignment_component_exactly(
                tuple(range(4)),
                incumbent,
                candidates,
                quality,
                conflicts,
                lambda: True,
                1,
            )

    def test_exact_component_search_budget_preserves_incumbent(self) -> None:
        candidates, quality, conflicts, _topology = (
            _small_allocator_counterexample()
        )
        incumbent = np.asarray((-1, 2, 1, 0), dtype=np.int8)

        with patch(
            "housemaker.camera_uv_projection."
            "EXACT_CAMERA_ASSIGNMENT_MAX_SEARCH_NODES",
            0,
        ):
            assignment = _solve_camera_assignment_component_exactly(
                tuple(range(4)),
                incumbent,
                candidates,
                quality,
                conflicts,
                None,
                1,
            )

        np.testing.assert_array_equal(assignment, incumbent)


# ### Optional retained-artifact regression ###
class CameraUvStrictRetainedArtifactTests(unittest.TestCase):
    def test_retained_real_meshy_artifacts_meet_strict_invariants(self) -> None:
        local_app_data = os.environ.get("LOCALAPPDATA")
        if not local_app_data:
            self.skipTest("LOCALAPPDATA is unavailable.")
        artifact_directory = Path(local_app_data) / "HouseMaker" / "generated"
        artifact_paths = [
            artifact_directory / artifact_name
            for artifact_name in REAL_ARTIFACT_NAMES
        ]
        if not all(path.is_file() for path in artifact_paths):
            self.skipTest("Retained Meshy camera-UV artifacts are unavailable.")

        for artifact_path in artifact_paths:
            with self.subTest(artifact=artifact_path.name):
                result = project_uvs_from_camera_views_from_glb(
                    artifact_path.read_bytes(),
                    options=CameraUvProjectionOptions(capture_image_size=512),
                )
                expected = REAL_ARTIFACT_EXPECTATIONS[artifact_path.name]
                self.assertGreaterEqual(
                    result.projected_face_count,
                    expected["minimum_primary"],
                )
                self.assertEqual(
                    result.invisible_face_count,
                    expected["invisible"],
                )
                self.assertEqual(
                    result.quality_fallback_face_count,
                    expected["quality_fallback"],
                )
                _assert_all_strict_invariants(self, result)


if __name__ == "__main__":
    unittest.main()
