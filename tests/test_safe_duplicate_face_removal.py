# ### Imports ###
from __future__ import annotations

import unittest
from io import BytesIO

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.color import ColorVisuals
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.object_face_edit import load_object_face_geometry
from housemaker.safe_duplicate_face_removal import (
    DEFAULT_RELATIVE_VERTEX_TOLERANCE,
    SafeDuplicateFaceRemovalCancelled,
    remove_safe_duplicate_faces_from_glb,
)


# ### Fixture helpers ###
def _triangle_mesh(
    vertices: np.ndarray,
    face: tuple[int, int, int] = (0, 1, 2),
    *,
    material: PBRMaterial | None = None,
) -> trimesh.Trimesh:
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray((face,), dtype=np.int64),
        process=False,
    )
    if material is not None:
        mesh.visual = TextureVisuals(
            uv=np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
            material=material,
        )
    return mesh


def _base_triangle(*, x_offset: float = 0.0) -> np.ndarray:
    return np.asarray(
        (
            (x_offset, 0.0, 0.0),
            (x_offset, 1.0, 0.0),
            (x_offset, 0.0, 1.0),
        ),
        dtype=float,
    )


def _scene_glb(
    *entries: tuple[str, trimesh.Trimesh, np.ndarray | None],
) -> bytes:
    scene = trimesh.Scene(metadata={"fixture": "duplicate-removal"})
    for name, mesh, transform in entries:
        scene.add_geometry(
            mesh,
            geom_name=name,
            node_name=name,
            transform=transform,
        )
    return bytes(scene.export(file_type="glb"))


def _load_scene(glb_bytes: bytes) -> trimesh.Scene:
    scene = trimesh.load(
        BytesIO(glb_bytes),
        file_type="glb",
        force="scene",
        process=False,
    )
    assert isinstance(scene, trimesh.Scene)
    return scene


def _material(name: str, color: tuple[int, int, int, int]) -> PBRMaterial:
    return PBRMaterial(
        name=name,
        baseColorTexture=Image.new("RGBA", (2, 2), color),
    )


def _set_uniform_vertex_normal(
    mesh: trimesh.Trimesh,
    normal: tuple[float, float, float] | np.ndarray,
) -> trimesh.Trimesh:
    normalized = np.asarray(normal, dtype=float)
    normalized /= np.linalg.norm(normalized)
    mesh.vertex_normals = np.tile(normalized, (len(mesh.vertices), 1))
    return mesh


# ### Duplicate-removal tests ###
class SafeDuplicateFaceRemovalTests(unittest.TestCase):
    def test_same_facing_duplicate_keeps_first_face_deterministically(self) -> None:
        shared_material = _material("shared", (180, 90, 30, 255))
        source = _scene_glb(
            (
                "z_duplicate",
                _triangle_mesh(_base_triangle(), material=shared_material),
                None,
            ),
            (
                "a_retained",
                _triangle_mesh(
                    _base_triangle(),
                    material=_material("shared", (180, 90, 30, 255)),
                ),
                None,
            ),
        )

        result = remove_safe_duplicate_faces_from_glb(source)

        self.assertTrue(result.changed)
        self.assertEqual(result.original_face_count, 2)
        self.assertEqual(result.retained_face_count, 1)
        self.assertEqual(result.removed_face_count, 1)
        self.assertEqual(result.duplicate_group_count, 1)
        self.assertEqual(tuple(_load_scene(result.glb_bytes).geometry), ("a_retained",))

    def test_corner_order_does_not_prevent_same_facing_removal(self) -> None:
        source = _scene_glb(
            ("first", _triangle_mesh(_base_triangle()), None),
            ("second", _triangle_mesh(_base_triangle(), (1, 2, 0)), None),
        )

        result = remove_safe_duplicate_faces_from_glb(source)

        self.assertEqual(result.removed_face_count, 1)
        self.assertEqual(result.retained_face_count, 1)

    def test_serialization_scale_noise_is_tolerated(self) -> None:
        noisy = _base_triangle().copy()
        noisy[:, 1:] += np.asarray(
            ((2e-8, -1e-8), (-2e-8, 1e-8), (1e-8, 2e-8))
        )
        source = _scene_glb(
            ("first", _triangle_mesh(_base_triangle()), None),
            ("noisy", _triangle_mesh(noisy), None),
        )

        result = remove_safe_duplicate_faces_from_glb(source)

        self.assertAlmostEqual(
            result.vertex_tolerance,
            np.sqrt(2.0) * DEFAULT_RELATIVE_VERTEX_TOLERANCE,
        )
        self.assertEqual(result.removed_face_count, 1)

    def test_close_offset_layer_is_preserved(self) -> None:
        source = _scene_glb(
            ("first", _triangle_mesh(_base_triangle()), None),
            ("offset", _triangle_mesh(_base_triangle(x_offset=1e-5)), None),
        )

        result = remove_safe_duplicate_faces_from_glb(source)

        self.assertFalse(result.changed)
        self.assertEqual(result.glb_bytes, source)
        self.assertEqual(result.retained_face_count, 2)

    def test_opposite_winding_coincident_sheet_is_preserved(self) -> None:
        source = _scene_glb(
            ("front", _triangle_mesh(_base_triangle()), None),
            ("back", _triangle_mesh(_base_triangle(), (0, 2, 1)), None),
        )

        result = remove_safe_duplicate_faces_from_glb(source)

        self.assertEqual(result.removed_face_count, 0)
        self.assertEqual(result.retained_face_count, 2)

    def test_different_material_layers_are_preserved(self) -> None:
        source = _scene_glb(
            (
                "red",
                _triangle_mesh(
                    _base_triangle(),
                    material=_material("red", (255, 0, 0, 255)),
                ),
                None,
            ),
            (
                "blue",
                _triangle_mesh(
                    _base_triangle(),
                    material=_material("blue", (0, 0, 255, 255)),
                ),
                None,
            ),
        )

        result = remove_safe_duplicate_faces_from_glb(source)

        self.assertEqual(result.removed_face_count, 0)
        self.assertEqual(result.retained_face_count, 2)

    def test_different_per_corner_uvs_are_preserved(self) -> None:
        first = _triangle_mesh(
            _base_triangle(),
            material=_material("shared", (90, 130, 170, 255)),
        )
        second = _triangle_mesh(
            _base_triangle(),
            material=_material("shared", (90, 130, 170, 255)),
        )
        second.visual.uv = np.asarray(
            ((1.0, 1.0), (0.0, 1.0), (1.0, 0.0)),
            dtype=float,
        )
        source = _scene_glb(
            ("first", first, None),
            ("different_uv", second, None),
        )

        result = remove_safe_duplicate_faces_from_glb(source)

        self.assertEqual(result.removed_face_count, 0)
        self.assertEqual(result.retained_face_count, 2)

    def test_different_per_vertex_colors_are_preserved(self) -> None:
        first = _triangle_mesh(_base_triangle())
        first.visual = ColorVisuals(
            mesh=first,
            vertex_colors=np.asarray(
                (
                    (255, 0, 0, 255),
                    (0, 255, 0, 255),
                    (0, 0, 255, 255),
                ),
                dtype=np.uint8,
            ),
        )
        second = _triangle_mesh(_base_triangle())
        second.visual = ColorVisuals(
            mesh=second,
            vertex_colors=np.asarray(
                (
                    (0, 255, 0, 255),
                    (255, 0, 0, 255),
                    (0, 0, 255, 255),
                ),
                dtype=np.uint8,
            ),
        )
        source = _scene_glb(
            ("first", first, None),
            ("different_colors", second, None),
        )

        result = remove_safe_duplicate_faces_from_glb(source)

        self.assertEqual(result.removed_face_count, 0)
        self.assertEqual(result.retained_face_count, 2)

    def test_different_authored_vertex_normals_are_preserved(self) -> None:
        first = _set_uniform_vertex_normal(
            _triangle_mesh(
                _base_triangle(),
                material=_material("shared", (70, 110, 150, 255)),
            ),
            (1.0, 0.0, 0.0),
        )
        smooth_variant = _set_uniform_vertex_normal(
            _triangle_mesh(
                _base_triangle(),
                material=_material("shared", (70, 110, 150, 255)),
            ),
            (1.0, 1.0, 0.0),
        )
        source = _scene_glb(
            ("first", first, None),
            ("smooth_variant", smooth_variant, None),
        )

        result = remove_safe_duplicate_faces_from_glb(source)

        self.assertEqual(result.removed_face_count, 0)
        self.assertEqual(result.retained_face_count, 2)

    def test_equal_rendered_normals_survive_nonuniform_transform_basis(self) -> None:
        target_world_normal = np.asarray((1.0, 1.0, 1.0), dtype=float)
        first = _set_uniform_vertex_normal(
            _triangle_mesh(
                _base_triangle(),
                material=_material("shared", (45, 95, 145, 255)),
            ),
            target_world_normal,
        )
        local_vertices = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (0.0, 0.5, 0.0),
                (0.0, 0.0, 2.0),
            ),
            dtype=float,
        )
        nonuniform_transform = np.diag((1.0, 2.0, 0.5, 1.0))
        transformed = _set_uniform_vertex_normal(
            _triangle_mesh(
                local_vertices,
                material=_material("shared", (45, 95, 145, 255)),
            ),
            nonuniform_transform[:3, :3].T @ target_world_normal,
        )
        source = _scene_glb(
            ("first", first, None),
            ("transformed", transformed, nonuniform_transform),
        )

        result = remove_safe_duplicate_faces_from_glb(source)

        self.assertEqual(result.removed_face_count, 1)
        self.assertEqual(result.retained_face_count, 1)

    def test_equal_rendered_normals_survive_mirrored_transform_parity(self) -> None:
        target_world_normal = np.asarray((1.0, 1.0, 0.0), dtype=float)
        first = _set_uniform_vertex_normal(
            _triangle_mesh(
                _base_triangle(),
                material=_material("shared", (55, 105, 155, 255)),
            ),
            target_world_normal,
        )
        mirrored_vertices = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=float,
        )
        mirrored_transform = np.diag((1.0, -1.0, 1.0, 1.0))
        mirrored = _set_uniform_vertex_normal(
            _triangle_mesh(
                mirrored_vertices,
                material=_material("shared", (55, 105, 155, 255)),
            ),
            mirrored_transform[:3, :3].T @ target_world_normal,
        )
        source = _scene_glb(
            ("first", first, None),
            ("mirrored", mirrored, mirrored_transform),
        )

        result = remove_safe_duplicate_faces_from_glb(source)

        self.assertEqual(result.removed_face_count, 1)
        self.assertEqual(result.retained_face_count, 1)

    def test_retained_uv_texture_normals_transform_and_metadata_survive(self) -> None:
        vertices = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 1.0, 1.0),
            )
        )
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=np.asarray(((0, 1, 2), (1, 3, 2), (0, 1, 2))),
            process=False,
        )
        mesh.visual = TextureVisuals(
            uv=np.asarray(((0.1, 0.2), (0.8, 0.2), (0.1, 0.9), (0.8, 0.9))),
            material=_material("preserved", (31, 79, 151, 255)),
        )
        authored_normals = np.tile((1.0, 0.0, 0.0), (len(vertices), 1))
        mesh.vertex_normals = authored_normals
        transform = trimesh.transformations.translation_matrix((3.0, 4.0, 5.0))
        source = _scene_glb(("surface", mesh, transform))
        source_face_geometry = load_object_face_geometry(source)

        result = remove_safe_duplicate_faces_from_glb(source)

        output_scene = _load_scene(result.glb_bytes)
        output_transform, geometry_name = output_scene.graph.get("surface")
        output_mesh = output_scene.geometry[geometry_name]
        self.assertEqual(output_scene.metadata, {"fixture": "duplicate-removal"})
        np.testing.assert_allclose(output_transform, transform, atol=1e-7)
        self.assertEqual(len(output_mesh.faces), 2)
        self.assertEqual(
            output_mesh.visual.material.baseColorTexture.getpixel((0, 0)),
            (31, 79, 151, 255),
        )
        output_face_geometry = load_object_face_geometry(result.glb_bytes)
        self.assertEqual(output_face_geometry.face_count, 2)
        np.testing.assert_array_equal(
            output_face_geometry.uv_triangles,
            source_face_geometry.uv_triangles[:2],
        )
        np.testing.assert_allclose(
            output_mesh.vertex_normals,
            authored_normals,
            atol=1e-7,
        )

    def test_all_pbr_maps_survive_duplicate_removal(self) -> None:
        material = _material("complete-pbr", (31, 79, 151, 255))
        material.normalTexture = Image.new(
            "RGBA",
            (2, 2),
            (128, 128, 255, 255),
        )
        packed_orm = Image.new("RGBA", (2, 2), (83, 61, 197, 255))
        material.metallicRoughnessTexture = packed_orm
        material.occlusionTexture = packed_orm
        material.emissiveTexture = Image.new(
            "RGBA",
            (2, 2),
            (40, 80, 120, 255),
        )
        material.emissiveFactor = (0.25, 0.5, 0.75)
        mesh = trimesh.Trimesh(
            vertices=np.asarray(
                (
                    (0.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                    (0.0, 1.0, 1.0),
                ),
                dtype=float,
            ),
            faces=np.asarray(
                ((0, 1, 2), (1, 3, 2), (0, 1, 2)),
                dtype=np.int64,
            ),
            process=False,
        )
        mesh.visual = TextureVisuals(
            uv=np.asarray(
                ((0.1, 0.2), (0.8, 0.2), (0.1, 0.9), (0.8, 0.9)),
                dtype=float,
            ),
            material=material,
        )

        result = remove_safe_duplicate_faces_from_glb(
            _scene_glb(("pbr-surface", mesh, None))
        )

        self.assertEqual(result.removed_face_count, 1)
        output_material = next(
            iter(_load_scene(result.glb_bytes).geometry.values())
        ).visual.material
        self.assertEqual(
            output_material.baseColorTexture.getpixel((0, 0)),
            (31, 79, 151, 255),
        )
        self.assertEqual(
            output_material.normalTexture.getpixel((0, 0)),
            (128, 128, 255, 255),
        )
        self.assertEqual(
            output_material.metallicRoughnessTexture.getpixel((0, 0)),
            (83, 61, 197, 255),
        )
        self.assertEqual(
            output_material.occlusionTexture.getpixel((0, 0)),
            (83, 61, 197, 255),
        )
        self.assertEqual(
            output_material.emissiveTexture.getpixel((0, 0)),
            (40, 80, 120, 255),
        )
        np.testing.assert_allclose(
            output_material.emissiveFactor,
            (0.25, 0.5, 0.75),
        )


# ### Validation tests ###
class SafeDuplicateFaceRemovalValidationTests(unittest.TestCase):
    def test_invalid_relative_tolerance_is_rejected(self) -> None:
        source = _scene_glb(("triangle", _triangle_mesh(_base_triangle()), None))
        for invalid in (0.0, -1.0, float("nan"), True, "1e-7"):
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    remove_safe_duplicate_faces_from_glb(
                        source,
                        relative_vertex_tolerance=invalid,  # type: ignore[arg-type]
                    )

    def test_cancellation_is_checked_before_processing(self) -> None:
        source = _scene_glb(("triangle", _triangle_mesh(_base_triangle()), None))

        with self.assertRaises(SafeDuplicateFaceRemovalCancelled):
            remove_safe_duplicate_faces_from_glb(
                source,
                cancel_requested=lambda: True,
            )


# ### Test runner ###
if __name__ == "__main__":
    unittest.main()
