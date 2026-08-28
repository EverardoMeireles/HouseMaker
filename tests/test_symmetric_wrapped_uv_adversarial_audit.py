# ### Imports ###
from __future__ import annotations

import io
import math
import unittest

import cv2
import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import MultiMaterial, PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.object_symmetry import (
    _normalize_mesh_repeat_uvs,
    _normalize_scene_repeat_uvs,
    _repeat_tile_indices,
    build_symmetric_half_texture_variants,
)


# ### Constants ###
_RESOLUTION = 2048
_OPAQUE_BLACK = np.asarray((0, 0, 0, 255), dtype=np.uint8)


# ### Fixture helpers ###
def _edge_texture() -> np.ndarray:
    texture = np.full(
        (_RESOLUTION, _RESOLUTION, 4),
        (0, 180, 0, 255),
        dtype=np.uint8,
    )
    texture[:, 0] = (255, 0, 0, 255)
    texture[:, -1] = (0, 0, 255, 255)
    return texture


def _phase_texture() -> np.ndarray:
    rows, columns = np.indices((_RESOLUTION, _RESOLUTION))
    texture = np.empty((_RESOLUTION, _RESOLUTION, 4), dtype=np.uint8)
    texture[:, :, 0] = (columns % 5) * 50
    texture[:, :, 1] = (rows % 7) * 35
    texture[:, :, 2] = 80
    texture[:, :, 3] = 255
    return texture


def _alpha_edge_texture() -> np.ndarray:
    texture = np.full(
        (_RESOLUTION, _RESOLUTION, 4),
        (0, 240, 40, 255),
        dtype=np.uint8,
    )
    texture[:, -1] = (255, 0, 255, 0)
    return texture


def _textured_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    *,
    texture: np.ndarray | None = None,
) -> trimesh.Trimesh:
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    source_texture = _edge_texture() if texture is None else texture
    mesh.visual = TextureVisuals(
        uv=np.asarray(uvs, dtype=float),
        material=PBRMaterial(
            name="wrapped-audit-material",
            baseColorTexture=Image.fromarray(source_texture, mode="RGBA"),
        ),
    )
    return mesh


def _scene_glb(mesh: trimesh.Trimesh) -> bytes:
    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name="wrapped-geometry", node_name="wrapped-node")
    return bytes(scene.export(file_type="glb"))


def _load_mesh(payload: bytes) -> trimesh.Trimesh:
    scene = trimesh.load(
        io.BytesIO(payload),
        file_type="glb",
        force="scene",
        process=False,
    )
    if not isinstance(scene, trimesh.Scene):
        raise AssertionError("The wrapped-UV fixture did not load as a scene.")
    mesh = scene.to_geometry()
    if not isinstance(mesh, trimesh.Trimesh):
        raise AssertionError("The wrapped-UV fixture did not contain a mesh.")
    return mesh


def _sample_repeat_bilinear(
    texture: np.ndarray,
    uv: np.ndarray,
) -> np.ndarray:
    height, width = texture.shape[:2]
    wrapped = np.asarray(uv, dtype=float) - np.floor(uv)
    column = float(wrapped[0]) * width - 0.5
    row = (1.0 - float(wrapped[1])) * height - 0.5
    column0_raw = math.floor(column)
    row0_raw = math.floor(row)
    column_fraction = column - column0_raw
    row_fraction = row - row0_raw
    column0 = column0_raw % width
    column1 = (column0_raw + 1) % width
    row0 = row0_raw % height
    row1 = (row0_raw + 1) % height
    top = (
        np.asarray(texture[row0, column0], dtype=float)
        * (1.0 - column_fraction)
        + np.asarray(texture[row0, column1], dtype=float) * column_fraction
    )
    bottom = (
        np.asarray(texture[row1, column0], dtype=float)
        * (1.0 - column_fraction)
        + np.asarray(texture[row1, column1], dtype=float) * column_fraction
    )
    return top * (1.0 - row_fraction) + bottom * row_fraction


def _edge_pressure_fixture(texture: np.ndarray) -> bytes:
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        dtype=float,
    )
    faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    repeated_uvs = np.asarray(
        ((1.0, 0.2), (1.5, 0.2), (1.5, 0.8), (1.0, 0.8)),
        dtype=float,
    )
    return _scene_glb(
        _textured_mesh(
            vertices,
            faces,
            repeated_uvs,
            texture=texture,
        )
    )


def _zero_gutter_pressure_fixture() -> tuple[bytes, np.ndarray]:
    texture = _edge_texture()
    return _edge_pressure_fixture(texture), texture


def _unaligned_two_chart_fixture() -> tuple[bytes, np.ndarray]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    uvs: list[tuple[float, float]] = []
    rectangles = ((100, 100, 300, 300), (500, 500, 700, 700))
    for chart_index, (x0, y0, x1, y1) in enumerate(rectangles):
        base = len(vertices)
        position_x = float(chart_index * 2)
        vertices.extend(
            (
                (position_x, 0.0, 0.0),
                (position_x + 1.0, 0.0, 0.0),
                (position_x + 1.0, 1.0, 0.0),
                (position_x, 1.0, 0.0),
            )
        )
        faces.extend(
            ((base, base + 1, base + 2), (base, base + 2, base + 3))
        )
        tile_offset = float(chart_index)
        uvs.extend(
            (
                (tile_offset + (x0 + 0.5) / _RESOLUTION, 1.0 - (y1 + 0.5) / _RESOLUTION),
                (tile_offset + (x1 + 0.5) / _RESOLUTION, 1.0 - (y1 + 0.5) / _RESOLUTION),
                (tile_offset + (x1 + 0.5) / _RESOLUTION, 1.0 - (y0 + 0.5) / _RESOLUTION),
                (tile_offset + (x0 + 0.5) / _RESOLUTION, 1.0 - (y0 + 0.5) / _RESOLUTION),
            )
        )
    texture = _phase_texture()
    return (
        _scene_glb(
            _textured_mesh(
                np.asarray(vertices, dtype=float),
                np.asarray(faces, dtype=np.int64),
                np.asarray(uvs, dtype=float),
                texture=texture,
            )
        ),
        texture,
    )


# ### Tile ownership and topology tests ###
class WrappedUvTileOwnershipAuditTests(unittest.TestCase):
    def test_exact_negative_tile_has_one_half_open_owner(self) -> None:
        self.assertEqual(list(_repeat_tile_indices(np.asarray((-1.0, 0.0)))), [-1])

        vertices = np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2),), dtype=np.int64)
        source_uvs = np.asarray(
            ((-1.0, -2.0), (0.0, -2.0), (-1.0, -1.0)),
            dtype=float,
        )
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.visual = TextureVisuals(uv=source_uvs, material=PBRMaterial())

        output, clipping_work = _normalize_mesh_repeat_uvs(
            mesh,
            source_uvs,
            maximum_output_faces=8,
            maximum_clipping_work=8,
        )

        self.assertEqual(clipping_work, 0)
        self.assertEqual(len(output.faces), 1)
        self.assertAlmostEqual(float(output.area), 0.5, places=12)
        np.testing.assert_allclose(
            np.asarray(output.visual.uv, dtype=float),
            ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
            atol=0.0,
        )

    def test_subpixel_integer_noise_does_not_create_phantom_tiles(self) -> None:
        noise = 5e-7
        vertices = np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2),), dtype=np.int64)
        source_uvs = np.asarray(
            (
                (-1.0 - noise, -2.0 - noise),
                (0.0 + noise, -2.0 - noise),
                (-1.0 - noise, -1.0 + noise),
            ),
            dtype=float,
        )
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.visual = TextureVisuals(uv=source_uvs, material=PBRMaterial())

        output, clipping_work = _normalize_mesh_repeat_uvs(
            mesh,
            source_uvs,
            maximum_output_faces=8,
            maximum_clipping_work=8,
        )

        self.assertEqual(clipping_work, 0)
        self.assertEqual(len(output.faces), 1)
        self.assertAlmostEqual(float(output.area), 0.5, places=12)


# ### Attribute and no-op tests ###
class WrappedUvAttributeAuditTests(unittest.TestCase):
    def test_child_faces_keep_their_source_material_indices(self) -> None:
        vertices = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (3.0, 0.0, 0.0),
                (4.0, 0.0, 0.0),
                (3.0, 1.0, 0.0),
            ),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64)
        source_uvs = np.asarray(
            (
                (-0.2, 0.2),
                (1.2, 0.2),
                (-0.2, 1.2),
                (2.1, -1.9),
                (2.4, -1.9),
                (2.1, -1.6),
            ),
            dtype=float,
        )
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.visual = TextureVisuals(
            uv=source_uvs,
            material=MultiMaterial(
                (PBRMaterial(name="first"), PBRMaterial(name="second"))
            ),
            face_materials=np.asarray((0, 1), dtype=np.int64),
        )

        output, _work = _normalize_mesh_repeat_uvs(
            mesh,
            source_uvs,
            maximum_output_faces=64,
            maximum_clipping_work=64,
        )
        output_vertices = np.asarray(output.vertices, dtype=float)
        output_materials = np.asarray(output.visual.face_materials, dtype=np.int64)

        for face, material_index in zip(
            np.asarray(output.faces, dtype=np.int64),
            output_materials,
            strict=True,
        ):
            centroid_x = float(np.mean(output_vertices[face, 0]))
            self.assertEqual(int(material_index), 0 if centroid_x < 2.0 else 1)
        self.assertEqual(
            [material.name for material in output.visual.material.materials],
            ["first", "second"],
        )

    def test_in_range_scene_normalization_is_an_identity_operation(self) -> None:
        texture = np.full((4, 4, 4), (20, 40, 60, 255), dtype=np.uint8)
        mesh = _textured_mesh(
            np.asarray(
                ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
            ),
            np.asarray(((0, 1, 2),), dtype=np.int64),
            np.asarray(((0.1, 0.2), (0.4, 0.2), (0.1, 0.6))),
            texture=texture,
        )
        transform = trimesh.transformations.translation_matrix((3.0, -2.0, 5.0))
        scene = trimesh.Scene()
        scene.add_geometry(
            mesh,
            geom_name="identity-geometry",
            node_name="identity-node",
            transform=transform,
        )
        original_geometry = scene.geometry["identity-geometry"]
        original_uvs = np.asarray(original_geometry.visual.uv).copy()

        _normalize_scene_repeat_uvs(scene)

        self.assertIs(scene.geometry["identity-geometry"], original_geometry)
        np.testing.assert_array_equal(
            np.asarray(scene.geometry["identity-geometry"].visual.uv),
            original_uvs,
        )
        output_transform, geometry_name = scene.graph.get("identity-node")
        self.assertEqual(geometry_name, "identity-geometry")
        np.testing.assert_array_equal(output_transform, transform)


# ### Texture reconstruction tests ###
class WrappedUvTextureAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source, texture = _zero_gutter_pressure_fixture()
        cls.source_glb = source
        cls.source_texture = texture
        cls.variants = build_symmetric_half_texture_variants(source)
        alpha_texture = _alpha_edge_texture()
        cls.alpha_source_texture = alpha_texture
        cls.alpha_variants = build_symmetric_half_texture_variants(
            _edge_pressure_fixture(alpha_texture)
        )
        phase_source, phase_texture = _unaligned_two_chart_fixture()
        cls.phase_source_texture = phase_texture
        cls.phase_variants = build_symmetric_half_texture_variants(phase_source)

    def test_exact_repeat_seam_keeps_opposite_edge_filter_samples(self) -> None:
        for resolution in (2048, 1024, 512):
            with self.subTest(resolution=resolution):
                output_mesh = _load_mesh(
                    self.variants.glb_by_resolution[resolution]
                )
                output_texture = self.variants.preview_rgba_by_resolution[
                    resolution
                ]
                reference_texture = cv2.resize(
                    self.source_texture,
                    (resolution, resolution),
                    interpolation=cv2.INTER_AREA,
                )
                seam_samples = []
                for position, output_uv in zip(
                    np.asarray(output_mesh.vertices, dtype=float),
                    np.asarray(output_mesh.visual.uv, dtype=float),
                    strict=True,
                ):
                    if abs(float(position[0])) > 1e-7:
                        continue
                    source_uv = np.asarray((1.0, 0.2 + 0.6 * position[1]))
                    expected = _sample_repeat_bilinear(
                        reference_texture,
                        source_uv,
                    )
                    actual = _sample_repeat_bilinear(output_texture, output_uv)
                    np.testing.assert_allclose(actual, expected, atol=3.0)
                    seam_samples.append(actual)
                self.assertGreaterEqual(len(seam_samples), 2)

    def test_repeated_export_is_byte_deterministic(self) -> None:
        repeated = build_symmetric_half_texture_variants(self.source_glb)

        self.assertEqual(
            repeated.glb_by_resolution,
            self.variants.glb_by_resolution,
        )
        self.assertEqual(
            repeated.texture_png_by_resolution,
            self.variants.texture_png_by_resolution,
        )

    def test_lower_variants_keep_rigid_chart_resampling_phase(self) -> None:
        rectangles = ((100, 100, 300, 300), (500, 500, 700, 700))
        for resolution in (2048, 1024, 512):
            with self.subTest(resolution=resolution):
                output_mesh = _load_mesh(
                    self.phase_variants.glb_by_resolution[resolution]
                )
                output_texture = self.phase_variants.preview_rgba_by_resolution[
                    resolution
                ]
                reference_texture = cv2.resize(
                    self.phase_source_texture,
                    (resolution, resolution),
                    interpolation=cv2.INTER_AREA,
                )
                output_vertices = np.asarray(output_mesh.vertices, dtype=float)
                output_uvs = np.asarray(output_mesh.visual.uv, dtype=float)
                for face in np.asarray(output_mesh.faces, dtype=np.int64):
                    position = np.mean(output_vertices[face], axis=0)
                    output_uv = np.mean(output_uvs[face], axis=0)
                    chart_index = 0 if position[0] < 1.5 else 1
                    x0, y0, x1, y1 = rectangles[chart_index]
                    local_x = float(position[0] - chart_index * 2)
                    local_y = float(position[1])
                    source_uv = np.asarray(
                        (
                            (x0 + (x1 - x0) * local_x + 0.5) / _RESOLUTION,
                            1.0
                            - (y1 - (y1 - y0) * local_y + 0.5)
                            / _RESOLUTION,
                        )
                    )
                    expected = _sample_repeat_bilinear(
                        reference_texture,
                        source_uv,
                    )
                    actual = _sample_repeat_bilinear(
                        output_texture,
                        output_uv,
                    )
                    np.testing.assert_allclose(actual, expected, atol=3.0)

    def test_repeat_seam_alpha_has_no_hidden_color_fringe(self) -> None:
        expected_visible_color = np.asarray((0.0, 240.0, 40.0))
        for resolution in (2048, 1024, 512):
            with self.subTest(resolution=resolution):
                output_mesh = _load_mesh(
                    self.alpha_variants.glb_by_resolution[resolution]
                )
                output_texture = self.alpha_variants.preview_rgba_by_resolution[
                    resolution
                ]
                reference_texture = cv2.resize(
                    self.alpha_source_texture,
                    (resolution, resolution),
                    interpolation=cv2.INTER_AREA,
                )
                checked = 0
                for position, output_uv in zip(
                    np.asarray(output_mesh.vertices, dtype=float),
                    np.asarray(output_mesh.visual.uv, dtype=float),
                    strict=True,
                ):
                    if abs(float(position[0])) > 1e-7:
                        continue
                    source_uv = np.asarray((1.0, 0.2 + 0.6 * position[1]))
                    expected_alpha = _sample_repeat_bilinear(
                        reference_texture,
                        source_uv,
                    )[3]
                    actual = _sample_repeat_bilinear(output_texture, output_uv)
                    np.testing.assert_allclose(
                        actual[:3],
                        expected_visible_color,
                        atol=3.0,
                    )
                    self.assertAlmostEqual(
                        float(actual[3]),
                        float(expected_alpha),
                        delta=3.0,
                    )
                    checked += 1
                self.assertGreaterEqual(checked, 2)

    def test_512_variant_uses_only_the_left_quarter_texture_region(self) -> None:
        output_mesh = _load_mesh(self.variants.glb_by_resolution[512])
        output_uvs = np.asarray(output_mesh.visual.uv, dtype=float)
        output_texture = self.variants.preview_rgba_by_resolution[512]

        self.assertGreaterEqual(float(np.min(output_uvs)), 0.0)
        self.assertLessEqual(float(np.max(output_uvs[:, 0])), 0.5)
        self.assertLessEqual(float(np.max(output_uvs[:, 1])), 1.0)
        expected_right = np.empty((512, 256, 4), dtype=np.uint8)
        expected_right[:] = _OPAQUE_BLACK
        np.testing.assert_array_equal(output_texture[:, 256:], expected_right)
        used = np.any(output_texture[:, :256, :3] != 0, axis=2)
        self.assertTrue(np.any(used))
        _rows, columns = np.nonzero(used)
        self.assertLessEqual(int(np.min(columns)), 4)


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
