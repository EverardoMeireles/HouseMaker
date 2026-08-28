# ### Imports ###
from __future__ import annotations

import io
import unittest

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.object_symmetry import build_symmetric_half_texture_variants


# ### Constants ###
_RESOLUTION = 2048
_OPAQUE_BLACK = np.asarray((0, 0, 0, 255), dtype=np.uint8)


# ### Fixture helpers ###
def _coordinate_texture() -> np.ndarray:
    rows, columns = np.indices((_RESOLUTION, _RESOLUTION))
    texture = np.empty((_RESOLUTION, _RESOLUTION, 4), dtype=np.uint8)
    texture[:, :, 0] = np.rint(columns * (255.0 / 2047.0)).astype(np.uint8)
    texture[:, :, 1] = np.rint(rows * (255.0 / 2047.0)).astype(np.uint8)
    texture[:, :, 2] = 73
    texture[:, :, 3] = 255
    return texture


def _repeat_seam_texture() -> np.ndarray:
    texture = np.zeros((_RESOLUTION, _RESOLUTION, 4), dtype=np.uint8)
    texture[:, :, 3] = 255
    texture[:, 0] = (255, 0, 0, 255)
    texture[:, -1] = (0, 0, 255, 255)
    return texture


def _textured_glb(
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    texture: np.ndarray,
) -> bytes:
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    mesh.visual = TextureVisuals(
        uv=np.asarray(uvs, dtype=float),
        material=PBRMaterial(
            name="degenerate-uv-audit-material",
            baseColorTexture=Image.fromarray(texture, mode="RGBA"),
        ),
    )
    return bytes(trimesh.Scene(mesh).export(file_type="glb"))


def _load_mesh(payload: bytes) -> trimesh.Trimesh:
    loaded = trimesh.load(
        io.BytesIO(payload),
        file_type="glb",
        force="scene",
        process=False,
    )
    if not isinstance(loaded, trimesh.Scene):
        raise AssertionError("The adversarial fixture did not load as a scene.")
    geometry = loaded.to_geometry()
    if not isinstance(geometry, trimesh.Trimesh):
        raise AssertionError("The adversarial fixture contains no triangle mesh.")
    return geometry


def _face_uvs(mesh: trimesh.Trimesh, face_index: int) -> np.ndarray:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    uvs = np.asarray(mesh.visual.uv, dtype=float)
    return uvs[faces[face_index]]


def _uv_determinant(triangle: np.ndarray) -> float:
    return float(
        np.linalg.det(
            np.vstack((triangle[1] - triangle[0], triangle[2] - triangle[0]))
        )
    )


def _sample_repeat_bilinear(texture: np.ndarray, uv: np.ndarray) -> np.ndarray:
    height, width = texture.shape[:2]
    wrapped = np.asarray(uv, dtype=float) - np.floor(uv)
    column = float(wrapped[0]) * width - 0.5
    row = (1.0 - float(wrapped[1])) * height - 0.5
    column0_raw = int(np.floor(column))
    row0_raw = int(np.floor(row))
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


def _assert_right_half_black(
    test_case: unittest.TestCase,
    texture: np.ndarray,
) -> None:
    expected = np.empty((texture.shape[0], texture.shape[1] // 2, 4), dtype=np.uint8)
    expected[:] = _OPAQUE_BLACK
    np.testing.assert_array_equal(texture[:, texture.shape[1] // 2 :], expected)


# ### Adversarial repair tests ###
class SymmetricDegenerateUvAdversarialAuditTests(unittest.TestCase):
    def test_point_repair_at_repeat_seam_preserves_wrapped_color(self) -> None:
        vertices = np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2),), dtype=np.int64)
        source_uv = np.asarray((1.0, 0.5), dtype=float)
        uvs = np.repeat(source_uv[np.newaxis, :], 3, axis=0)
        source_texture = _repeat_seam_texture()

        variants = build_symmetric_half_texture_variants(
            _textured_glb(vertices, faces, uvs, source_texture)
        )
        output_mesh = _load_mesh(variants.glb_by_resolution[2048])
        output_uvs = _face_uvs(output_mesh, 0)
        output_texture = variants.preview_rgba_by_resolution[2048]
        expected = _sample_repeat_bilinear(source_texture, source_uv)

        self.assertGreater(abs(_uv_determinant(output_uvs)), 0.0)
        for barycentric in (
            np.asarray((1.0, 0.0, 0.0)),
            np.asarray((0.0, 1.0, 0.0)),
            np.asarray((0.0, 0.0, 1.0)),
            np.asarray((1.0 / 3.0,) * 3),
        ):
            np.testing.assert_allclose(
                _sample_repeat_bilinear(
                    output_texture,
                    barycentric @ output_uvs,
                ),
                expected,
                atol=1.0,
            )
        _assert_right_half_black(self, output_texture)

    def test_private_point_and_line_scratch_charts_do_not_coalesce(self) -> None:
        vertices = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                (0.5, 1.0, 0.0),
                (3.0, 0.0, 0.0),
                (4.0, 0.0, 0.0),
                (3.0, 1.0, 0.0),
            ),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64)
        line_uvs = np.asarray(
            ((0.1, 0.1), (0.9, 0.9), (0.4, 0.4)),
            dtype=float,
        )
        point_uv = np.asarray((0.8, 0.15), dtype=float)
        point_uvs = np.repeat(point_uv[np.newaxis, :], 3, axis=0)
        uvs = np.vstack((line_uvs, point_uvs))
        source_texture = _coordinate_texture()

        variants = build_symmetric_half_texture_variants(
            _textured_glb(vertices, faces, uvs, source_texture)
        )
        output_mesh = _load_mesh(variants.glb_by_resolution[2048])
        output_texture = variants.preview_rgba_by_resolution[2048]
        output_line = _face_uvs(output_mesh, 0)
        output_point = _face_uvs(output_mesh, 1)

        self.assertGreater(abs(_uv_determinant(output_line)), 0.0)
        self.assertGreater(abs(_uv_determinant(output_point)), 0.0)
        for barycentric in (
            np.asarray((0.8, 0.1, 0.1)),
            np.asarray((0.1, 0.8, 0.1)),
            np.asarray((0.1, 0.1, 0.8)),
            np.asarray((0.2, 0.3, 0.5)),
        ):
            np.testing.assert_allclose(
                _sample_repeat_bilinear(
                    output_texture,
                    barycentric @ output_line,
                ),
                _sample_repeat_bilinear(
                    source_texture,
                    barycentric @ line_uvs,
                ),
                atol=4.0,
            )
        np.testing.assert_allclose(
            _sample_repeat_bilinear(output_texture, np.mean(output_point, axis=0)),
            _sample_repeat_bilinear(source_texture, point_uvs[0]),
            atol=1.0,
        )

    def test_normal_face_does_not_hide_connected_long_subpixel_face(self) -> None:
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
        uvs = np.asarray(
            (
                (0.1, 0.25),
                (0.1, 0.8),
                (0.9, 0.25),
                (0.9, 0.25000012),
            ),
            dtype=float,
        )
        source_texture = _coordinate_texture()
        source_glb = _textured_glb(vertices, faces, uvs, source_texture)
        source_mesh = _load_mesh(source_glb)

        variants = build_symmetric_half_texture_variants(source_glb)
        output_mesh = _load_mesh(variants.glb_by_resolution[2048])
        output_texture = variants.preview_rgba_by_resolution[2048]
        source_normal = _face_uvs(source_mesh, 0)
        source_thin = _face_uvs(source_mesh, 1)
        output_normal = _face_uvs(output_mesh, 0)
        output_thin = _face_uvs(output_mesh, 1)
        normal_scale_squared = abs(
            _uv_determinant(output_normal) / _uv_determinant(source_normal)
        )
        thin_scale_squared = abs(
            _uv_determinant(output_thin) / _uv_determinant(source_thin)
        )

        self.assertGreater(abs(_uv_determinant(output_thin)), 0.0)
        np.testing.assert_allclose(
            thin_scale_squared,
            normal_scale_squared,
            rtol=2e-3,
            atol=1e-6,
        )
        for barycentric in (
            np.asarray((1.0, 0.0, 0.0)),
            np.asarray((0.0, 1.0, 0.0)),
            np.asarray((0.0, 0.0, 1.0)),
            np.asarray((0.2, 0.3, 0.5)),
        ):
            np.testing.assert_allclose(
                _sample_repeat_bilinear(
                    output_texture,
                    barycentric @ output_thin,
                ),
                _sample_repeat_bilinear(
                    source_texture,
                    barycentric @ source_thin,
                ),
                atol=4.0,
            )

    def test_zero_area_3d_face_is_removed_before_point_uv_repair(self) -> None:
        vertices = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (2.0, 0.0, 0.0),
                (3.0, 0.0, 0.0),
                (4.0, 0.0, 0.0),
            ),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64)
        uvs = np.asarray(
            (
                (0.1, 0.1),
                (0.4, 0.1),
                (0.1, 0.4),
                (0.75, 0.75),
                (0.75, 0.75),
                (0.75, 0.75),
            ),
            dtype=float,
        )

        variants = build_symmetric_half_texture_variants(
            _textured_glb(vertices, faces, uvs, _coordinate_texture())
        )
        output_mesh = _load_mesh(variants.glb_by_resolution[2048])

        self.assertEqual(len(output_mesh.faces), 1)
        self.assertEqual(len(output_mesh.vertices), 3)
        self.assertGreater(abs(_uv_determinant(_face_uvs(output_mesh, 0))), 0.0)
        _assert_right_half_black(
            self,
            variants.preview_rgba_by_resolution[2048],
        )

    def test_normal_uv_path_is_byte_deterministic(self) -> None:
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
        uvs = np.asarray(
            ((0.1, 0.1), (0.35, 0.1), (0.35, 0.4), (0.1, 0.4)),
            dtype=float,
        )
        source_glb = _textured_glb(
            vertices,
            faces,
            uvs,
            _coordinate_texture(),
        )

        first = build_symmetric_half_texture_variants(source_glb)
        second = build_symmetric_half_texture_variants(source_glb)

        self.assertEqual(first.glb_by_resolution, second.glb_by_resolution)
        self.assertEqual(
            first.texture_png_by_resolution,
            second.texture_png_by_resolution,
        )
        for resolution in first.preview_rgba_by_resolution:
            np.testing.assert_array_equal(
                first.preview_rgba_by_resolution[resolution],
                second.preview_rgba_by_resolution[resolution],
            )


if __name__ == "__main__":
    unittest.main()
