# ### Imports ###
from __future__ import annotations

import unittest
from io import BytesIO

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.architectural_trim_uv import (
    CORNICE_UV_MAX_STRIP_ASPECT,
    build_cornice_segmented_uv_glb,
)
from housemaker.glb import (
    Z_UP_TO_GLTF_Y_UP_TRANSFORM,
    import_generated_glb,
)
from housemaker.scan_projection_layout import (
    SCAN_PROJECTION_LAYOUT_METADATA_KEY,
    build_scan_projection_layout_metadata,
    remap_scan_projection_scene_uvs,
)


# ### Fixture helpers ###
def _long_textured_box_glb(*, triangle_soup: bool = False) -> bytes:
    mesh = trimesh.creation.box(extents=(8.0, 0.25, 0.5))
    if triangle_soup:
        triangles = np.asarray(mesh.triangles, dtype=np.float64)
        mesh = trimesh.Trimesh(
            vertices=triangles.reshape((-1, 3)),
            faces=np.arange(len(triangles) * 3, dtype=np.int64).reshape((-1, 3)),
            process=False,
            validate=False,
        )
    mesh.visual = TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2), dtype=np.float64),
        material=PBRMaterial(
            name="ornamental_cornice",
            baseColorTexture=Image.new("RGBA", (4, 4), (230, 220, 200, 255)),
            metallicFactor=0.0,
            roughnessFactor=0.7,
        ),
    )
    gltf_mesh = mesh.copy()
    gltf_mesh.apply_transform(Z_UP_TO_GLTF_Y_UP_TRANSFORM)
    scene = trimesh.Scene()
    transform = trimesh.transformations.translation_matrix((2.0, 1.0, -3.0))
    scene.add_geometry(
        gltf_mesh,
        geom_name="cornice_geometry",
        node_name="cornice_node",
        transform=transform,
    )
    exported = scene.export(file_type="glb")
    if not isinstance(exported, bytes):
        raise TypeError("The cornice UV test fixture could not be serialized.")
    return exported


def _box_with_thin_detached_triangle_glb() -> bytes:
    """Build valid relief geometry whose thin chart xatlas may collapse."""

    box = trimesh.creation.box(extents=(0.4, 0.1, 0.02))
    triangle = trimesh.Trimesh(
        vertices=np.asarray(
            (
                (0.0, 0.0, 0.011),
                (0.2, 0.0, 0.011),
                (0.2, 1e-8, 0.011),
            ),
            dtype=np.float64,
        ),
        faces=np.asarray(((0, 1, 2),), dtype=np.int64),
        process=False,
        validate=False,
    )
    mesh = trimesh.util.concatenate((box, triangle))
    mesh.visual = TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2), dtype=np.float64),
        material=PBRMaterial(
            name="thin_relief_cornice",
            baseColorTexture=Image.new("RGBA", (4, 4), (230, 220, 200, 255)),
        ),
    )
    gltf_mesh = mesh.copy()
    gltf_mesh.apply_transform(Z_UP_TO_GLTF_Y_UP_TRANSFORM)
    exported = trimesh.Scene(gltf_mesh).export(file_type="glb")
    if not isinstance(exported, bytes):
        raise TypeError("The thin cornice UV test fixture could not be serialized.")
    return exported


def _face_components(faces: np.ndarray) -> tuple[np.ndarray, ...]:
    parents = np.arange(len(faces), dtype=np.int64)

    def find(face_index: int) -> int:
        while int(parents[face_index]) != face_index:
            parents[face_index] = parents[int(parents[face_index])]
            face_index = int(parents[face_index])
        return face_index

    first_face_by_vertex: dict[int, int] = {}
    for face_index, face in enumerate(faces):
        for raw_vertex_index in face:
            vertex_index = int(raw_vertex_index)
            previous = first_face_by_vertex.get(vertex_index)
            if previous is None:
                first_face_by_vertex[vertex_index] = face_index
                continue
            first_root = find(face_index)
            second_root = find(previous)
            if first_root != second_root:
                parents[second_root] = first_root
    groups: dict[int, list[int]] = {}
    for face_index in range(len(faces)):
        groups.setdefault(find(face_index), []).append(face_index)
    return tuple(np.asarray(group, dtype=np.int64) for group in groups.values())


def _uv_density_scales(model: object) -> np.ndarray:
    scales: list[float] = []
    for geometry in model.scene.geometry.values():
        vertices = np.asarray(geometry.vertices, dtype=np.float64)
        faces = np.asarray(geometry.faces, dtype=np.int64)
        uvs = np.asarray(geometry.visual.uv, dtype=np.float64)
        triangles = vertices[faces]
        world_areas = 0.5 * np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            ),
            axis=1,
        )
        uv_triangles = uvs[faces]
        first_edges = uv_triangles[:, 1] - uv_triangles[:, 0]
        second_edges = uv_triangles[:, 2] - uv_triangles[:, 0]
        uv_areas = 0.5 * np.abs(
            first_edges[:, 0] * second_edges[:, 1]
            - first_edges[:, 1] * second_edges[:, 0]
        )
        valid = (world_areas > 1e-12) & (uv_areas > 1e-12)
        scales.extend(np.sqrt(uv_areas[valid] / world_areas[valid]))
    return np.asarray(scales, dtype=np.float64)


def _triangle_uv_area(triangle: np.ndarray) -> float:
    """Return one UV triangle's unsigned area."""

    points = np.asarray(triangle, dtype=np.float64)
    first_edge = points[1] - points[0]
    second_edge = points[2] - points[0]
    return 0.5 * abs(
        float(
            first_edge[0] * second_edge[1]
            - first_edge[1] * second_edge[0]
        )
    )


# ### Segmented cornice UV tests ###
class ArchitecturalTrimUvTests(unittest.TestCase):
    def test_long_cornice_is_split_without_changing_geometry_or_material(self) -> None:
        source = import_generated_glb(_long_textured_box_glb())

        result = import_generated_glb(
            build_cornice_segmented_uv_glb(
                source.glb_bytes,
                texture_resolution=512,
            )
        )

        np.testing.assert_allclose(
            result.mesh.bounds,
            source.mesh.bounds,
            rtol=0.0,
            atol=1e-6,
        )
        self.assertAlmostEqual(result.mesh.area, source.mesh.area, places=5)
        self.assertGreater(len(result.scene.geometry), 1)
        maximum_segment_length = (
            CORNICE_UV_MAX_STRIP_ASPECT
            * min(float(source.mesh.extents[1]), float(source.mesh.extents[2]))
        )
        for geometry in result.scene.geometry.values():
            self.assertLessEqual(
                float(np.ptp(geometry.vertices[:, 0])),
                maximum_segment_length + 1e-6,
            )
            material = geometry.visual.material
            self.assertEqual(material.name, "ornamental_cornice")
            self.assertIsNotNone(material.baseColorTexture)

    def test_uvs_use_one_density_and_pack_charts_horizontally(self) -> None:
        result = import_generated_glb(
            build_cornice_segmented_uv_glb(
                _long_textured_box_glb(),
                texture_resolution=512,
            )
        )

        for geometry in result.scene.geometry.values():
            uvs = np.asarray(geometry.visual.uv, dtype=np.float64)
            self.assertEqual(uvs.shape, (len(geometry.vertices), 2))
            self.assertTrue(np.all(np.isfinite(uvs)))
            self.assertGreaterEqual(float(np.min(uvs)), 0.0)
            self.assertLessEqual(float(np.max(uvs)), 1.0)
            faces = np.asarray(geometry.faces, dtype=np.int64)
            for component in _face_components(faces):
                chart_vertices = np.unique(faces[component].reshape(-1))
                chart_extents = np.ptp(uvs[chart_vertices], axis=0)
                self.assertGreaterEqual(
                    float(chart_extents[0]) + 1e-6,
                    float(chart_extents[1]),
                )

        density_scales = _uv_density_scales(result)
        self.assertGreater(len(density_scales), 0)
        self.assertLess(
            float(np.max(density_scales) / np.min(density_scales)),
            1.01,
        )

    def test_triangle_soup_is_welded_and_unwrapped_reliably(self) -> None:
        result = import_generated_glb(
            build_cornice_segmented_uv_glb(
                _long_textured_box_glb(triangle_soup=True),
                texture_resolution=256,
            )
        )

        self.assertGreater(len(result.mesh.faces), 0)
        self.assertGreater(len(result.scene.geometry), 1)
        for geometry in result.scene.geometry.values():
            uvs = np.asarray(geometry.visual.uv, dtype=np.float64)
            self.assertTrue(np.all(np.isfinite(uvs)))
            self.assertTrue(np.all(uvs >= 0.0))
            self.assertTrue(np.all(uvs <= 1.0))

    def test_xatlas_collapsed_thin_face_gets_an_isometric_uv_chart(self) -> None:
        source = import_generated_glb(_box_with_thin_detached_triangle_glb())

        result = import_generated_glb(
            build_cornice_segmented_uv_glb(
                source.glb_bytes,
                texture_resolution=256,
            )
        )

        self.assertGreaterEqual(len(result.mesh.faces), len(source.mesh.faces))
        self.assertAlmostEqual(result.mesh.area, source.mesh.area, places=7)
        np.testing.assert_allclose(
            result.mesh.bounds,
            source.mesh.bounds,
            rtol=0.0,
            atol=1e-6,
        )
        for geometry in result.scene.geometry.values():
            faces = np.asarray(geometry.faces, dtype=np.int64)
            uvs = np.asarray(geometry.visual.uv, dtype=np.float64)
            uv_triangles = uvs[faces]
            uv_areas = np.asarray(
                [_triangle_uv_area(triangle_uvs) for triangle_uvs in uv_triangles],
                dtype=np.float64,
            )
            self.assertTrue(np.all(uv_areas > 0.0))

    def test_same_input_produces_same_geometry_and_uv_layout(self) -> None:
        source = _long_textured_box_glb(triangle_soup=True)

        first = import_generated_glb(
            build_cornice_segmented_uv_glb(source, texture_resolution=256)
        )
        second = import_generated_glb(
            build_cornice_segmented_uv_glb(source, texture_resolution=256)
        )

        first_geometries = sorted(first.scene.geometry.items(), key=lambda item: item[0])
        second_geometries = sorted(
            second.scene.geometry.items(),
            key=lambda item: item[0],
        )
        self.assertEqual(
            [name for name, _geometry in first_geometries],
            [name for name, _geometry in second_geometries],
        )
        for (_name, first_geometry), (_other_name, second_geometry) in zip(
            first_geometries,
            second_geometries,
            strict=True,
        ):
            np.testing.assert_array_equal(first_geometry.faces, second_geometry.faces)
            np.testing.assert_allclose(
                first_geometry.vertices,
                second_geometry.vertices,
                rtol=0.0,
                atol=0.0,
            )
            np.testing.assert_allclose(
                first_geometry.visual.uv,
                second_geometry.visual.uv,
                rtol=0.0,
                atol=0.0,
            )

    def test_replacing_scan_uvs_clears_scan_variant_metadata(self) -> None:
        source_scene = trimesh.load(
            BytesIO(_long_textured_box_glb()),
            file_type="glb",
            force="scene",
            process=False,
        )
        for geometry in source_scene.geometry.values():
            geometry.metadata[SCAN_PROJECTION_LAYOUT_METADATA_KEY] = (
                build_scan_projection_layout_metadata(
                    canonical_texture_resolution=2048,
                    version=2,
                )
            )
        tagged_glb = bytes(source_scene.export(file_type="glb"))

        result = import_generated_glb(
            build_cornice_segmented_uv_glb(
                tagged_glb,
                texture_resolution=2048,
            )
        )

        for geometry in result.scene.geometry.values():
            self.assertNotIn(
                SCAN_PROJECTION_LAYOUT_METADATA_KEY,
                geometry.metadata,
            )
        self.assertFalse(remap_scan_projection_scene_uvs(result.scene, 512))

    def test_rejects_invalid_texture_resolution(self) -> None:
        source = _long_textured_box_glb()

        with self.assertRaisesRegex(TypeError, "must be an integer"):
            build_cornice_segmented_uv_glb(source, texture_resolution=True)
        with self.assertRaisesRegex(ValueError, "must be between"):
            build_cornice_segmented_uv_glb(source, texture_resolution=32)


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
