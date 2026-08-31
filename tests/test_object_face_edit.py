# ### Imports ###
from __future__ import annotations

import unittest
from io import BytesIO

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import MultiMaterial, PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.object_face_edit import (
    _filter_face_materials,
    delete_object_faces_preserving_uvs,
    load_object_face_geometry,
    load_object_face_geometry_from_scene,
)


# ### Fixture constants ###
_REPEATING_UVS = np.asarray(
    (
        (-0.25, 0.10),
        (0.20, -0.30),
        (1.40, 0.20),
        (0.80, 1.30),
        (1.20, 1.40),
        (-0.40, 0.90),
        (0.40, 0.50),
        (0.70, 0.60),
    ),
    dtype=float,
)


# ### Fixture helpers ###
def _pattern_texture(width: int, height: int, seed: int) -> Image.Image:
    y_coordinates, x_coordinates = np.indices((height, width))
    pixels = np.empty((height, width, 4), dtype=np.uint8)
    pixels[..., 0] = (x_coordinates * 31 + y_coordinates * 17 + seed) % 256
    pixels[..., 1] = (x_coordinates * 7 + y_coordinates * 43 + seed * 2) % 256
    pixels[..., 2] = (x_coordinates * 19 + y_coordinates * 11 + seed * 3) % 256
    pixels[..., 3] = 255
    return Image.fromarray(pixels, mode="RGBA")


def _authored_normals(vertices: np.ndarray) -> np.ndarray:
    normals = np.asarray(vertices, dtype=float) + np.asarray((2.0, 3.0, 4.0))
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)


def _textured_box(
    material_name: str,
    texture_size: tuple[int, int],
    texture_seed: int,
    *,
    metadata: dict[str, object] | None = None,
) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=(1.0, 1.5, 2.0))
    mesh.metadata = dict(metadata or {})
    mesh.visual = TextureVisuals(
        uv=_REPEATING_UVS.copy(),
        material=PBRMaterial(
            name=material_name,
            baseColorTexture=_pattern_texture(*texture_size, texture_seed),
            baseColorFactor=(0.8, 0.7, 0.6, 1.0),
            metallicFactor=0.25,
            roughnessFactor=0.75,
        ),
    )
    mesh.vertex_normals = _authored_normals(mesh.vertices)
    return mesh


def _two_node_glb(
    *,
    metadata: dict[str, object] | None = None,
) -> bytes:
    scene = trimesh.Scene(metadata=metadata)
    scene.add_geometry(
        _textured_box(
            "right-material",
            (11, 6),
            13,
            metadata={
                "asset_role": "right",
                "nested": {"editable": True, "ordinal": 2},
            },
        ),
        geom_name="right-geometry",
        node_name="z-right-node",
        transform=trimesh.transformations.concatenate_matrices(
            trimesh.transformations.translation_matrix((3.0, 1.0, -2.0)),
            trimesh.transformations.rotation_matrix(0.35, (0.0, 0.0, 1.0)),
        ),
    )
    scene.add_geometry(
        _textured_box(
            "left-material",
            (7, 5),
            29,
            metadata={
                "asset_role": "left",
                "nested": {"editable": True, "ordinal": 1},
            },
        ),
        geom_name="left-geometry",
        node_name="a-left-node",
        transform=trimesh.transformations.concatenate_matrices(
            trimesh.transformations.translation_matrix((-3.0, -1.0, 2.0)),
            trimesh.transformations.rotation_matrix(-0.2, (1.0, 0.0, 0.0)),
        ),
    )
    return bytes(scene.export(file_type="glb"))


def _load_scene(glb_bytes: bytes) -> trimesh.Scene:
    loaded = trimesh.load(
        BytesIO(glb_bytes),
        file_type="glb",
        force="scene",
        process=False,
    )
    assert isinstance(loaded, trimesh.Scene)
    return loaded


def _node_mesh(
    scene: trimesh.Scene,
    node_name: str,
) -> tuple[np.ndarray, trimesh.Trimesh]:
    transform, geometry_name = scene.graph.get(node_name)
    mesh = scene.geometry[geometry_name]
    assert isinstance(mesh, trimesh.Trimesh)
    return np.asarray(transform, dtype=float), mesh


def _texture_pixels(mesh: trimesh.Trimesh) -> np.ndarray:
    texture = mesh.visual.material.baseColorTexture
    return np.asarray(texture.convert("RGBA"), dtype=np.uint8)


def _triangle_keys(vertices: np.ndarray, faces: np.ndarray) -> list[tuple[float, ...]]:
    keys: list[tuple[float, ...]] = []
    for triangle in vertices[faces]:
        ordered_vertices = sorted(tuple(np.round(vertex, 6)) for vertex in triangle)
        keys.append(tuple(float(value) for vertex in ordered_vertices for value in vertex))
    return sorted(keys)


def _scaled_box_glb(scale: float) -> bytes:
    scene = trimesh.Scene()
    scene.add_geometry(
        _textured_box("scaled", (9, 4), 47),
        geom_name="scaled-geometry",
        node_name="scaled-node",
        transform=np.diag((scale, scale, scale, 1.0)),
    )
    return bytes(scene.export(file_type="glb"))


def _box_with_degenerate_face_glb() -> bytes:
    mesh = _textured_box("degenerate", (8, 8), 61)
    mesh.faces = np.vstack(
        (
            np.asarray(mesh.faces, dtype=np.int64),
            np.asarray(((0, 0, 1),), dtype=np.int64),
        )
    )
    return bytes(trimesh.Scene(mesh).export(file_type="glb"))


def _untextured_box_glb() -> bytes:
    return bytes(trimesh.Scene(trimesh.creation.box()).export(file_type="glb"))


def _mixed_uv_node_glb() -> bytes:
    scene = trimesh.Scene()
    scene.add_geometry(
        trimesh.creation.box(),
        geom_name="z-untextured-geometry",
        node_name="a-untextured-node",
    )
    scene.add_geometry(
        _textured_box("textured", (8, 8), 73),
        geom_name="a-textured-geometry",
        node_name="z-textured-node",
    )
    return bytes(scene.export(file_type="glb"))


# ### Stable face-index tests ###
class ObjectFaceGeometryTests(unittest.TestCase):
    def test_loader_has_deterministic_node_order_and_z_up_geometry(self) -> None:
        source = _two_node_glb()

        first = load_object_face_geometry(source)
        second = load_object_face_geometry(source)
        from_scene = load_object_face_geometry_from_scene(_load_scene(source))

        self.assertEqual(first.face_count, 24)
        np.testing.assert_array_equal(first.vertices, second.vertices)
        np.testing.assert_array_equal(first.faces, second.faces)
        np.testing.assert_array_equal(first.vertices, from_scene.vertices)
        np.testing.assert_array_equal(first.faces, from_scene.faces)
        np.testing.assert_array_equal(
            first.uv_face_indices,
            from_scene.uv_face_indices,
        )
        first_triangle_center = np.mean(first.vertices[first.faces[0]], axis=0)
        second_node_center = np.mean(first.vertices[first.faces[12]], axis=0)
        self.assertLess(first_triangle_center[0], 0.0)
        self.assertGreater(second_node_center[0], 0.0)

    def test_loader_maps_uv_triangles_to_the_same_global_face_order(self) -> None:
        source = _two_node_glb()
        source_scene = _load_scene(source)

        geometry = load_object_face_geometry(source)

        expected_uv_triangles: list[np.ndarray] = []
        for node_name in sorted(source_scene.graph.nodes_geometry, key=str):
            _transform, mesh = _node_mesh(source_scene, node_name)
            expected_uv_triangles.extend(
                np.asarray(mesh.visual.uv, dtype=float)[
                    np.asarray(mesh.faces, dtype=np.int64)
                ]
            )
        np.testing.assert_array_equal(
            geometry.uv_face_indices,
            np.arange(geometry.face_count, dtype=np.int64),
        )
        np.testing.assert_array_equal(
            geometry.uv_triangles,
            np.asarray(expected_uv_triangles, dtype=float),
        )

    def test_untextured_nodes_do_not_compress_later_uv_face_ids(self) -> None:
        geometry = load_object_face_geometry(_mixed_uv_node_glb())

        self.assertEqual(geometry.face_count, 24)
        np.testing.assert_array_equal(
            geometry.uv_face_indices,
            np.arange(12, 24, dtype=np.int64),
        )
        self.assertEqual(geometry.uv_triangles.shape, (12, 3, 2))


# ### Topology-only deletion tests ###
class ObjectFaceDeletionTests(unittest.TestCase):
    def test_deletion_preserves_exact_retained_face_uvs_and_texture_pixels(
        self,
    ) -> None:
        source = _two_node_glb()
        source_scene = _load_scene(source)
        deleted_indices = {0, 1, 2, 3, 16, 18, 19, 21, 23}

        result = delete_object_faces_preserving_uvs(source, deleted_indices)
        output_scene = _load_scene(result.glb_bytes)

        self.assertTrue(result.preserved_textured_uvs)
        self.assertEqual(result.original_face_count, 24)
        self.assertEqual(result.retained_face_count, 15)
        self.assertEqual(result.deleted_face_count, 9)
        face_offset = 0
        output_uv_values: list[np.ndarray] = []
        for node_name in sorted(source_scene.graph.nodes_geometry, key=str):
            _source_transform, source_mesh = _node_mesh(source_scene, node_name)
            _output_transform, output_mesh = _node_mesh(output_scene, node_name)
            source_faces = np.asarray(source_mesh.faces, dtype=np.int64)
            local_keep = np.asarray(
                [
                    face_offset + local_index not in deleted_indices
                    for local_index in range(len(source_faces))
                ],
                dtype=bool,
            )
            expected_uv_triangles = np.asarray(
                source_mesh.visual.uv,
                dtype=float,
            )[source_faces[local_keep]]
            output_faces = np.asarray(output_mesh.faces, dtype=np.int64)
            actual_uv_triangles = np.asarray(
                output_mesh.visual.uv,
                dtype=float,
            )[output_faces]
            np.testing.assert_array_equal(
                actual_uv_triangles,
                expected_uv_triangles,
            )
            source_pixels = _texture_pixels(source_mesh)
            output_pixels = _texture_pixels(output_mesh)
            self.assertEqual(output_pixels.shape, source_pixels.shape)
            np.testing.assert_array_equal(output_pixels, source_pixels)
            output_uv_values.append(actual_uv_triangles.reshape((-1, 2)))
            face_offset += len(source_faces)

        retained_uvs = np.vstack(output_uv_values)
        self.assertLess(float(np.min(retained_uvs)), 0.0)
        self.assertGreater(float(np.max(retained_uvs)), 1.0)

    def test_deletion_preserves_faces_transforms_normals_and_metadata(self) -> None:
        source = _two_node_glb(
            metadata={
                "housemaker": {
                    "asset_id": "metadata-regression",
                    "face_edit_revision": 3,
                },
                "labels": ["generated", "editable"],
            }
        )
        source_scene = _load_scene(source)
        source_geometry = load_object_face_geometry(source)
        deleted_indices = {0, 13}

        result = delete_object_faces_preserving_uvs(source, deleted_indices)

        output_scene = _load_scene(result.glb_bytes)
        output_geometry = load_object_face_geometry(result.glb_bytes)
        expected_faces = np.delete(
            source_geometry.faces,
            sorted(deleted_indices),
            axis=0,
        )
        self.assertEqual(
            _triangle_keys(output_geometry.vertices, output_geometry.faces),
            _triangle_keys(source_geometry.vertices, expected_faces),
        )
        self.assertEqual(output_scene.metadata, source_scene.metadata)
        self.assertEqual(
            set(output_scene.graph.nodes_geometry),
            {"a-left-node", "z-right-node"},
        )

        face_offset = 0
        for node_name in sorted(source_scene.graph.nodes_geometry, key=str):
            source_transform, source_mesh = _node_mesh(source_scene, node_name)
            output_transform, output_mesh = _node_mesh(output_scene, node_name)
            np.testing.assert_array_equal(output_transform, source_transform)
            self.assertEqual(output_mesh.metadata, source_mesh.metadata)
            self.assertEqual(
                output_mesh.visual.material.name,
                source_mesh.visual.material.name,
            )
            source_faces = np.asarray(source_mesh.faces, dtype=np.int64)
            local_keep = np.asarray(
                [
                    face_offset + local_index not in deleted_indices
                    for local_index in range(len(source_faces))
                ],
                dtype=bool,
            )
            expected_normal_triangles = np.asarray(
                source_mesh.vertex_normals,
                dtype=float,
            )[source_faces[local_keep]]
            output_faces = np.asarray(output_mesh.faces, dtype=np.int64)
            actual_normal_triangles = np.asarray(
                output_mesh.vertex_normals,
                dtype=float,
            )[output_faces]
            np.testing.assert_allclose(
                actual_normal_triangles,
                expected_normal_triangles,
                rtol=0.0,
                atol=1e-7,
            )
            face_offset += len(source_faces)

    def test_small_node_scale_preserves_every_unselected_face(self) -> None:
        source = _scaled_box_glb(1e-7)

        result = delete_object_faces_preserving_uvs(source, {0})

        self.assertEqual(result.original_face_count, 12)
        self.assertEqual(result.retained_face_count, 11)
        self.assertEqual(result.deleted_face_count, 1)
        self.assertTrue(result.preserved_textured_uvs)
        self.assertEqual(
            load_object_face_geometry(result.glb_bytes).face_count,
            11,
        )

    def test_untextured_mesh_reports_that_textured_uvs_were_not_preserved(
        self,
    ) -> None:
        result = delete_object_faces_preserving_uvs(_untextured_box_glb(), {0})

        self.assertFalse(result.preserved_textured_uvs)
        self.assertEqual(result.retained_face_count, 11)


# ### Multi-material filtering tests ###
class MultiMaterialFilteringTests(unittest.TestCase):
    def test_face_material_indices_are_subset_with_retained_faces(self) -> None:
        mesh = trimesh.creation.box()
        original_face_materials = np.arange(len(mesh.faces), dtype=np.int64) % 2
        materials = MultiMaterial(
            materials=(
                PBRMaterial(
                    name="even",
                    baseColorTexture=_pattern_texture(4, 4, 71),
                ),
                PBRMaterial(
                    name="odd",
                    baseColorTexture=_pattern_texture(4, 4, 83),
                ),
            )
        )
        mesh.visual = TextureVisuals(
            uv=_REPEATING_UVS.copy(),
            material=materials,
            face_materials=original_face_materials.copy(),
        )
        keep_faces = np.asarray(
            [index not in {1, 4, 8} for index in range(len(mesh.faces))],
            dtype=bool,
        )

        _filter_face_materials(mesh, keep_faces)
        mesh.update_faces(keep_faces)

        np.testing.assert_array_equal(
            mesh.visual.face_materials,
            original_face_materials[keep_faces],
        )
        self.assertIs(mesh.visual.material, materials)


# ### Validation tests ###
class ObjectFaceDeletionValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = _two_node_glb()

    def test_empty_selection_is_rejected_as_a_no_op(self) -> None:
        with self.assertRaisesRegex(ValueError, "Select at least one"):
            delete_object_faces_preserving_uvs(self.source, [])

    def test_delete_all_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot remove every"):
            delete_object_faces_preserving_uvs(self.source, range(24))

    def test_invalid_indices_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the object"):
            delete_object_faces_preserving_uvs(self.source, {-1})
        with self.assertRaisesRegex(ValueError, "outside the object"):
            delete_object_faces_preserving_uvs(self.source, {24})
        with self.assertRaisesRegex(TypeError, "must be integers"):
            delete_object_faces_preserving_uvs(self.source, {1.5})
        with self.assertRaisesRegex(TypeError, "iterable of integers"):
            delete_object_faces_preserving_uvs(self.source, "1")

    def test_duplicate_indices_are_deleted_once(self) -> None:
        result = delete_object_faces_preserving_uvs(self.source, [0, 0, 0])

        self.assertEqual(result.deleted_face_count, 1)
        self.assertEqual(result.retained_face_count, 23)

    def test_empty_glb_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "GLB is empty"):
            load_object_face_geometry(b"")

    def test_unselected_degenerate_faces_do_not_block_deletion(self) -> None:
        source = _box_with_degenerate_face_glb()
        self.assertEqual(load_object_face_geometry(source).face_count, 13)

        result = delete_object_faces_preserving_uvs(source, {0})

        self.assertEqual(result.deleted_face_count, 1)
        self.assertEqual(result.retained_face_count, 12)
        retained = load_object_face_geometry(result.glb_bytes)
        triangles = retained.vertices[retained.faces]
        doubled_areas = np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            ),
            axis=1,
        )
        self.assertTrue(np.any(doubled_areas == 0.0))


# ### Test runner ###
if __name__ == "__main__":
    unittest.main()
