# ### Imports ###
from __future__ import annotations

import gc
from io import BytesIO
import unittest

import numpy as np
import trimesh
from shapely.geometry import Polygon

from housemaker.camera_uv_projection import (
    CAMERA_UV_TEXTURE_SIZES,
    CameraUvProjectionCancelled,
    CameraUvProjectionOptions,
    project_uvs_from_camera_views,
    project_uvs_from_camera_views_from_glb,
)
from housemaker.glb import import_generated_glb
from housemaker.unused_face_removal import ALL_CAMERA_IDS


# ### Module cleanup ###
def tearDownModule() -> None:
    """Release native geometry objects after this focused suite."""

    gc.collect()


# ### Fixture helpers ###
def _scene_glb(
    *meshes: tuple[str, trimesh.Trimesh, np.ndarray | None],
) -> bytes:
    scene = trimesh.Scene()
    for name, mesh, transform in meshes:
        scene.add_geometry(
            mesh,
            geom_name=name,
            node_name=name,
            transform=transform,
        )
    return bytes(scene.export(file_type="glb"))


def _cube_glb() -> bytes:
    return _scene_glb(
        ("cube", trimesh.creation.box(extents=(2.0, 2.0, 2.0)), None)
    )


def _nested_cube_glb() -> bytes:
    return _scene_glb(
        ("outer", trimesh.creation.box(extents=(2.0, 2.0, 2.0)), None),
        ("inner", trimesh.creation.box(extents=(1.0, 1.0, 1.0)), None),
    )


def _overlapping_projection_glb() -> bytes:
    vertices = np.asarray(
        (
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (-1.0, 1.0, 1.0),
            (-0.5, -0.5, 0.0),
            (1.5, -0.5, 0.0),
            (1.5, 1.5, 0.0),
        ),
        dtype=float,
    )
    faces = np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return _scene_glb(("overlapping_depths", mesh, None))


def _instanced_cube_glb() -> bytes:
    mesh = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name="shared", node_name="first")
    scene.graph.update(
        frame_to="second",
        frame_from=scene.graph.base_frame,
        matrix=trimesh.transformations.translation_matrix((4.0, 1.0, -0.5)),
        geometry="shared",
    )
    return bytes(scene.export(file_type="glb"))


def _authored_normal_tetrahedron_glb() -> bytes:
    vertices = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=float,
    )
    faces = np.asarray(
        ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)),
        dtype=np.int64,
    )
    authored_normals = np.asarray(
        ((1.0, 1.0, 1.0), (-1.0, 1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, -1.0)),
        dtype=float,
    )
    authored_normals /= np.linalg.norm(authored_normals, axis=1)[:, np.newaxis]
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=authored_normals,
        process=False,
    )
    return _scene_glb(("authored_normals", mesh, None))


def _billion_unit_span_glb() -> bytes:
    near_vertices = np.asarray(
        (
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (-1.0, 1.0, 1.0),
            (-0.5, -0.5, 0.0),
            (1.5, -0.5, 0.0),
            (1.5, 1.5, 0.0),
        ),
        dtype=float,
    )
    near_mesh = trimesh.Trimesh(
        vertices=near_vertices,
        faces=np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64),
        process=False,
    )
    far_mesh = trimesh.Trimesh(
        vertices=np.asarray(
            ((0.0, -1.0, 0.0), (2.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=float,
        ),
        faces=np.asarray(((0, 1, 2),), dtype=np.int64),
        process=False,
    )
    return _scene_glb(
        ("near", near_mesh, None),
        (
            "far",
            far_mesh,
            trimesh.transformations.translation_matrix(
                (10.0, 0.0, -1_000_000_000.0)
            ),
        ),
    )


# ### UV assertion helpers ###
def _all_uv_triangles(model) -> list[Polygon]:
    triangles: list[Polygon] = []
    for geometry in model.scene.geometry.values():
        if not isinstance(geometry, trimesh.Trimesh):
            continue
        uv = np.asarray(geometry.visual.uv, dtype=float)
        triangles.extend(Polygon(uv[face]) for face in geometry.faces)
    return triangles


def _assert_uv_triangles_are_valid_and_disjoint(
    test_case: unittest.TestCase,
    model,
) -> None:
    triangles = _all_uv_triangles(model)
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


def _face_geometry_signature(mesh: trimesh.Trimesh) -> list[tuple[float, ...]]:
    signatures: list[tuple[float, ...]] = []
    vertices = np.asarray(mesh.vertices, dtype=float)
    for face in np.asarray(mesh.faces, dtype=np.int64):
        sorted_points = sorted(
            tuple(float(value) for value in np.round(vertices[index], 7))
            for index in face
        )
        signatures.append(tuple(value for point in sorted_points for value in point))
    return sorted(signatures)


def _rectangles_overlap(first, second) -> bool:
    return not (
        first.x + first.width <= second.x
        or second.x + second.width <= first.x
        or first.y + first.height <= second.y
        or second.y + second.height <= first.y
    )


# ### Six-view assignment tests ###
class CameraUvAssignmentTests(unittest.TestCase):
    def test_cube_assigns_one_side_to_each_of_the_six_cameras(self) -> None:
        result = project_uvs_from_camera_views_from_glb(_cube_glb())

        self.assertEqual(result.original_face_count, 12)
        self.assertEqual(result.projected_face_count, 12)
        self.assertEqual(result.leftover_face_count, 0)
        self.assertEqual(
            result.camera_face_counts,
            {camera_id: 2 for camera_id in ALL_CAMERA_IDS},
        )
        self.assertEqual(sum(region.face_count for region in result.chart_regions), 12)
        _assert_uv_triangles_are_valid_and_disjoint(self, result.model)

    def test_nested_hidden_faces_are_reserved_at_the_bottom_left(self) -> None:
        result = project_uvs_from_camera_views_from_glb(_nested_cube_glb())

        self.assertEqual(result.original_face_count, 24)
        self.assertEqual(result.projected_face_count, 12)
        self.assertEqual(result.leftover_face_count, 12)
        leftover_regions = [
            region for region in result.chart_regions if region.is_leftover
        ]
        visible_regions = [
            region for region in result.chart_regions if not region.is_leftover
        ]
        self.assertTrue(leftover_regions)
        self.assertTrue(visible_regions)
        self.assertEqual(min(region.x for region in leftover_regions), 0)
        self.assertEqual(min(region.y for region in leftover_regions), 0)
        self.assertEqual(
            sum(region.face_count for region in leftover_regions),
            result.leftover_face_count,
        )
        reserved_right = max(
            region.x + region.width for region in leftover_regions
        )
        reserved_top = max(
            region.y + region.height for region in leftover_regions
        )
        self.assertTrue(
            all(
                region.x >= reserved_right or region.y >= reserved_top
                for region in visible_regions
            )
        )
        _assert_uv_triangles_are_valid_and_disjoint(self, result.model)

    def test_partially_overlapping_depth_projections_become_separate_charts(
        self,
    ) -> None:
        result = project_uvs_from_camera_views_from_glb(
            _overlapping_projection_glb()
        )

        self.assertEqual(result.original_face_count, 2)
        self.assertEqual(result.projected_face_count, 2)
        self.assertEqual(result.invisible_face_count, 0)
        self.assertEqual(result.quality_fallback_face_count, 0)
        self.assertEqual(result.conflict_fallback_face_count, 0)
        self.assertEqual(result.leftover_face_count, 0)
        self.assertEqual(len(result.chart_regions), 2)
        self.assertEqual(
            sum(not region.is_leftover for region in result.chart_regions),
            2,
        )
        self.assertTrue(
            all(region.face_count == 1 for region in result.chart_regions)
        )
        _assert_uv_triangles_are_valid_and_disjoint(self, result.model)


# ### Instance and geometry preservation tests ###
class CameraUvInstanceTests(unittest.TestCase):
    def test_transformed_instances_keep_shape_faces_and_node_transforms(self) -> None:
        source_glb = _instanced_cube_glb()
        source_model = import_generated_glb(source_glb)

        result = project_uvs_from_camera_views_from_glb(source_glb)

        self.assertEqual(result.original_face_count, 24)
        self.assertEqual(len(result.model.mesh.faces), 24)
        self.assertEqual(len(result.model.scene.geometry), 2)
        self.assertEqual(
            _face_geometry_signature(result.model.mesh),
            _face_geometry_signature(source_model.mesh),
        )
        source_scene = trimesh.load(
            BytesIO(source_glb),
            file_type="glb",
            force="scene",
            process=False,
        )
        for node_name in ("first", "second"):
            source_transform, _source_geometry = source_scene.graph.get(node_name)
            output_transform, _output_geometry = result.model.scene.graph.get(
                node_name
            )
            np.testing.assert_allclose(
                output_transform,
                source_transform,
                atol=1e-7,
            )
        _assert_uv_triangles_are_valid_and_disjoint(self, result.model)

    def test_authored_vertex_normals_survive_seam_duplication_and_round_trip(
        self,
    ) -> None:
        source_glb = _authored_normal_tetrahedron_glb()
        source_scene = trimesh.load(
            BytesIO(source_glb),
            file_type="glb",
            force="scene",
            process=False,
        )
        source_mesh = source_scene.geometry["authored_normals"]
        source_normals = np.asarray(source_mesh.vertex_normals, dtype=float)
        expected_by_position = {
            tuple(np.round(vertex, 7)): normal
            for vertex, normal in zip(source_mesh.vertices, source_normals)
        }

        result = project_uvs_from_camera_views_from_glb(source_glb)

        output_mesh = result.model.scene.geometry["authored_normals"]
        output_normals = np.asarray(output_mesh.vertex_normals, dtype=float)
        self.assertGreater(len(output_mesh.vertices), len(source_mesh.vertices))
        for vertex, output_normal in zip(output_mesh.vertices, output_normals):
            expected_normal = expected_by_position[tuple(np.round(vertex, 7))]
            np.testing.assert_allclose(
                output_normal,
                expected_normal,
                atol=1e-6,
            )

    def test_billion_unit_transformed_span_does_not_create_false_conflicts(
        self,
    ) -> None:
        source_glb = _billion_unit_span_glb()

        result = project_uvs_from_camera_views_from_glb(source_glb)

        self.assertEqual(result.original_face_count, 3)
        self.assertEqual(result.projected_face_count + result.leftover_face_count, 3)
        self.assertEqual(len(result.model.mesh.faces), 3)
        far_transform, _far_geometry = result.model.scene.graph.get("far")
        self.assertAlmostEqual(float(far_transform[2, 3]), -1_000_000_000.0)
        _assert_uv_triangles_are_valid_and_disjoint(self, result.model)


# ### Resolution and packing tests ###
class CameraUvResolutionTests(unittest.TestCase):
    def test_one_normalized_layout_scales_to_all_texture_resolutions(self) -> None:
        mesh = trimesh.creation.box(extents=(1.3, 2.7, 0.91))
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(0.37, (1.0, 0.3, 0.2))
        )
        result = project_uvs_from_camera_views_from_glb(
            _scene_glb(("rotated_box", mesh, None))
        )

        self.assertEqual(result.compatible_texture_sizes, CAMERA_UV_TEXTURE_SIZES)
        all_uv = np.vstack(
            [
                np.asarray(geometry.visual.uv, dtype=float)
                for geometry in result.model.scene.geometry.values()
                if isinstance(geometry, trimesh.Trimesh)
            ]
        )
        self.assertTrue(np.all(np.isfinite(all_uv)))
        self.assertTrue(np.all(all_uv >= -1e-8))
        self.assertTrue(np.all(all_uv <= 1.0 + 1e-8))
        base_pixels = all_uv * CAMERA_UV_TEXTURE_SIZES[0]
        for texture_size in CAMERA_UV_TEXTURE_SIZES[1:]:
            scale = texture_size // CAMERA_UV_TEXTURE_SIZES[0]
            np.testing.assert_allclose(
                all_uv * texture_size,
                base_pixels * scale,
                atol=1e-6,
            )
        for region in result.chart_regions:
            self.assertIsInstance(region.x, int)
            self.assertIsInstance(region.y, int)
            self.assertIsInstance(region.width, int)
            self.assertIsInstance(region.height, int)
            self.assertGreaterEqual(region.x, 0)
            self.assertGreaterEqual(region.y, 0)
            self.assertLessEqual(region.x + region.width, 512)
            self.assertLessEqual(region.y + region.height, 512)
        for first_index, first in enumerate(result.chart_regions):
            for second in result.chart_regions[first_index + 1 :]:
                self.assertFalse(_rectangles_overlap(first, second))
        _assert_uv_triangles_are_valid_and_disjoint(self, result.model)


# ### Cancellation and validation tests ###
class CameraUvValidationTests(unittest.TestCase):
    def test_cancellation_inside_a_visibility_capture_uses_public_exception(
        self,
    ) -> None:
        callback_count = 0

        def cancel_during_capture() -> bool:
            nonlocal callback_count
            callback_count += 1
            return callback_count >= 3

        with self.assertRaises(CameraUvProjectionCancelled):
            project_uvs_from_camera_views_from_glb(
                _cube_glb(),
                cancel_requested=cancel_during_capture,
            )

    def test_cancellation_at_export_does_not_report_completion(self) -> None:
        cancel_now = False
        reported_stages: list[str] = []

        def report_progress(progress) -> None:
            nonlocal cancel_now
            reported_stages.append(progress.stage)
            if progress.stage == "exporting":
                cancel_now = True

        with self.assertRaises(CameraUvProjectionCancelled):
            project_uvs_from_camera_views_from_glb(
                _cube_glb(),
                cancel_requested=lambda: cancel_now,
                progress_callback=report_progress,
            )

        self.assertIn("exporting", reported_stages)
        self.assertNotIn("complete", reported_stages)

    def test_malformed_empty_and_over_limit_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            project_uvs_from_camera_views_from_glb(b"")
        with self.assertRaisesRegex(ValueError, "could not be loaded"):
            project_uvs_from_camera_views_from_glb(b"not a GLB")
        with self.assertRaisesRegex(ValueError, "camera UV limit is 11"):
            project_uvs_from_camera_views_from_glb(
                _cube_glb(),
                options=CameraUvProjectionOptions(max_face_count=11),
            )
        with self.assertRaisesRegex(TypeError, "GeneratedModel"):
            project_uvs_from_camera_views(object())  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "padding must be an integer"):
            CameraUvProjectionOptions(
                padding_pixels=2.5,  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(
            ValueError,
            "maximum face count must be an integer",
        ):
            CameraUvProjectionOptions(
                max_face_count=True,  # type: ignore[arg-type]
            )


# ### Test runner ###
if __name__ == "__main__":
    unittest.main()
