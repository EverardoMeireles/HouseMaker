# ### Imports ###
from __future__ import annotations

import functools
import io
import math
import unittest

import cv2
import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.camera_uv_integrity import build_camera_uv_fingerprint
from housemaker.object_symmetry import build_symmetric_half_texture_variants
from housemaker.object_texture_variants import TEXTURE_RESOLUTIONS


# ### Constants ###
_RESOLUTION = 2048
_OPAQUE_BLACK = np.asarray((0, 0, 0, 255), dtype=np.uint8)


# ### Texture fixtures ###
@functools.lru_cache(maxsize=1)
def _periodic_texture() -> np.ndarray:
    """Return a non-black atlas that joins continuously under GL_REPEAT."""

    centers = (np.arange(_RESOLUTION, dtype=float) + 0.5) / _RESOLUTION
    horizontal = np.rint(
        128.0 + 90.0 * np.sin(2.0 * np.pi * centers)
    ).astype(np.uint8)
    vertical = np.rint(
        128.0 + 80.0 * np.cos(2.0 * np.pi * centers)
    ).astype(np.uint8)
    texture = np.empty((_RESOLUTION, _RESOLUTION, 4), dtype=np.uint8)
    texture[:, :, 0] = horizontal[np.newaxis, :]
    texture[:, :, 1] = vertical[:, np.newaxis]
    texture[:, :, 2] = 73
    texture[:, :, 3] = 255
    texture.setflags(write=False)
    return texture


@functools.lru_cache(maxsize=4)
def _internal_edge_texture(
    pixel_rectangle: tuple[int, int, int, int],
) -> np.ndarray:
    """Put a sharp retained color inside arbitrary internal atlas edges."""

    x0, y0, x1, y1 = pixel_rectangle
    texture = np.empty((_RESOLUTION, _RESOLUTION, 4), dtype=np.uint8)
    texture[:] = (251, 1, 249, 255)
    texture[y0:y1, x0:x1] = (17, 223, 41, 255)
    texture.setflags(write=False)
    return texture


# ### GLB fixture helpers ###
def _textured_glb(
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    *,
    texture: np.ndarray | None = None,
) -> bytes:
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    mesh.visual = TextureVisuals(
        uv=np.asarray(uvs, dtype=float),
        material=PBRMaterial(
            name="preserved-smart-fallback-material",
            baseColorTexture=Image.fromarray(
                _periodic_texture() if texture is None else texture,
                mode="RGBA",
            ),
        ),
    )
    scene = trimesh.Scene()
    scene.add_geometry(
        mesh,
        geom_name="preserved-smart-fallback-geometry",
        node_name="preserved-smart-fallback-node",
    )
    return bytes(scene.export(file_type="glb"))


def _quad_glb(
    uvs: tuple[tuple[float, float], ...],
    *,
    texture: np.ndarray | None = None,
) -> bytes:
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
    return _textured_glb(
        vertices,
        faces,
        np.asarray(uvs, dtype=float),
        texture=texture,
    )


def _internal_edge_quad_glb(
    pixel_rectangle: tuple[int, int, int, int],
) -> tuple[bytes, np.ndarray]:
    x0, y0, x1, y1 = pixel_rectangle
    pixel_points = (
        (x0 - 0.5, y1 - 0.5),
        (x1 - 0.5, y1 - 0.5),
        (x1 - 0.5, y0 - 0.5),
        (x0 - 0.5, y0 - 0.5),
    )
    uvs = tuple(
        (
            (column + 0.5) / _RESOLUTION,
            1.0 - (row + 0.5) / _RESOLUTION,
        )
        for column, row in pixel_points
    )
    texture = _internal_edge_texture(pixel_rectangle)
    return _quad_glb(uvs, texture=texture), texture


def _repeated_triangle_glb() -> bytes:
    vertices = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=float,
    )
    faces = np.asarray(((0, 1, 2),), dtype=np.int64)
    uvs = np.asarray(
        ((-0.75, -0.35), (1.25, 0.15), (0.15, 1.35)),
        dtype=float,
    )
    return _textured_glb(vertices, faces, uvs)


def _load_scene(payload: bytes) -> trimesh.Scene:
    loaded = trimesh.load(
        io.BytesIO(payload),
        file_type="glb",
        force="scene",
        process=False,
    )
    if not isinstance(loaded, trimesh.Scene):
        raise AssertionError("The preserved-UV audit fixture is not a scene.")
    return loaded


def _load_mesh(payload: bytes) -> trimesh.Trimesh:
    mesh = _load_scene(payload).to_geometry()
    if not isinstance(mesh, trimesh.Trimesh):
        raise AssertionError("The preserved-UV audit fixture has no mesh.")
    return mesh


def _first_texture(payload: bytes) -> np.ndarray:
    geometry = next(iter(_load_scene(payload).geometry.values()))
    texture = geometry.visual.material.baseColorTexture
    return np.asarray(texture.convert("RGBA"), dtype=np.uint8)


# ### UV and appearance helpers ###
def _uvs_by_position(mesh: trimesh.Trimesh) -> dict[tuple[float, ...], np.ndarray]:
    result: dict[tuple[float, ...], np.ndarray] = {}
    for position, uv in zip(mesh.vertices, mesh.visual.uv, strict=True):
        key = tuple(np.round(np.asarray(position, dtype=float), 7))
        normalized_uv = np.asarray(uv, dtype=float)
        previous = result.get(key)
        if previous is not None:
            np.testing.assert_allclose(previous, normalized_uv, atol=1e-7)
        result[key] = normalized_uv
    return result


def _sample_repeat_bilinear(texture: np.ndarray, uv: np.ndarray) -> np.ndarray:
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
        texture[row0, column0].astype(float) * (1.0 - column_fraction)
        + texture[row0, column1].astype(float) * column_fraction
    )
    bottom = (
        texture[row1, column0].astype(float) * (1.0 - column_fraction)
        + texture[row1, column1].astype(float) * column_fraction
    )
    return top * (1.0 - row_fraction) + bottom * row_fraction


def _barycentric_weights(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    offset = np.asarray(point[:2] - triangle[0, :2], dtype=float)
    coefficients = np.linalg.solve(
        np.column_stack(
            (
                triangle[1, :2] - triangle[0, :2],
                triangle[2, :2] - triangle[0, :2],
            )
        ),
        offset,
    )
    return np.asarray(
        (1.0 - coefficients[0] - coefficients[1], *coefficients),
        dtype=float,
    )


def _assert_quad_appearance(
    test: unittest.TestCase,
    source_glb: bytes,
    output_glb: bytes,
) -> None:
    source_mesh = _load_mesh(source_glb)
    output_mesh = _load_mesh(output_glb)
    source_uvs = _uvs_by_position(source_mesh)
    source_texture = _first_texture(source_glb)
    output_texture = _first_texture(output_glb)
    output_vertices = np.asarray(output_mesh.vertices, dtype=float)
    output_uvs = np.asarray(output_mesh.visual.uv, dtype=float)
    for face in np.asarray(output_mesh.faces, dtype=np.int64):
        source_uv = np.mean(
            [
                source_uvs[tuple(np.round(output_vertices[index], 7))]
                for index in face
            ],
            axis=0,
        )
        output_uv = np.mean(output_uvs[face], axis=0)
        np.testing.assert_allclose(
            _sample_repeat_bilinear(output_texture, output_uv),
            _sample_repeat_bilinear(source_texture, source_uv),
            atol=3.0,
            rtol=0.0,
        )


def _assert_repeated_triangle_appearance(
    source_glb: bytes,
    output_glb: bytes,
) -> None:
    source_mesh = _load_mesh(source_glb)
    output_mesh = _load_mesh(output_glb)
    source_face = np.asarray(source_mesh.faces[0], dtype=np.int64)
    source_triangle = np.asarray(source_mesh.vertices[source_face], dtype=float)
    source_uv_triangle = np.asarray(source_mesh.visual.uv, dtype=float)[source_face]
    source_texture = _first_texture(source_glb)
    output_texture = _first_texture(output_glb)
    output_vertices = np.asarray(output_mesh.vertices, dtype=float)
    output_uvs = np.asarray(output_mesh.visual.uv, dtype=float)
    for face in np.asarray(output_mesh.faces, dtype=np.int64):
        position = np.mean(output_vertices[face], axis=0)
        weights = _barycentric_weights(position, source_triangle)
        source_uv = weights @ source_uv_triangle
        output_uv = np.mean(output_uvs[face], axis=0)
        np.testing.assert_allclose(
            _sample_repeat_bilinear(output_texture, output_uv),
            _sample_repeat_bilinear(source_texture, source_uv),
            atol=4.0,
            rtol=0.0,
        )


def _assert_only_left_half_is_used(test: unittest.TestCase, variants: object) -> None:
    for resolution in TEXTURE_RESOLUTIONS:
        with test.subTest(resolution=resolution):
            texture = variants.preview_rgba_by_resolution[resolution]
            test.assertEqual(texture.shape, (resolution, resolution, 4))
            expected_black = np.empty(
                (resolution, resolution // 2, 4),
                dtype=np.uint8,
            )
            expected_black[:] = _OPAQUE_BLACK
            np.testing.assert_array_equal(
                texture[:, resolution // 2 :],
                expected_black,
            )
            test.assertTrue(
                np.any(texture[:, : resolution // 2] != _OPAQUE_BLACK)
            )
            mesh = _load_mesh(variants.glb_by_resolution[resolution])
            uvs = np.asarray(mesh.visual.uv, dtype=float)
            test.assertGreaterEqual(float(np.min(uvs)), -1e-7)
            test.assertLessEqual(float(np.max(uvs[:, 0])), 0.5 + 1e-7)
            test.assertLessEqual(float(np.max(uvs[:, 1])), 1.0 + 1e-7)


def _interpolate_quad_uv(
    uvs_by_position: dict[tuple[float, ...], np.ndarray],
    x: float,
    y: float,
) -> np.ndarray:
    bottom_left = uvs_by_position[(0.0, 0.0, 0.0)]
    bottom_right = uvs_by_position[(1.0, 0.0, 0.0)]
    top_left = uvs_by_position[(0.0, 1.0, 0.0)]
    return (
        bottom_left
        + x * (bottom_right - bottom_left)
        + y * (top_left - bottom_left)
    )


def _assert_internal_edge_appearance_at_every_resolution(
    source_glb: bytes,
    source_texture: np.ndarray,
    variants: object,
    *,
    tolerance: float,
) -> None:
    source_uvs = _uvs_by_position(_load_mesh(source_glb))
    sample_positions = (
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (0.0, 0.37),
        (1.0, 0.63),
        (0.29, 0.0),
        (0.71, 1.0),
    )
    for resolution in TEXTURE_RESOLUTIONS:
        reference = cv2.resize(
            source_texture,
            (resolution, resolution),
            interpolation=cv2.INTER_AREA,
        )
        output_texture = variants.preview_rgba_by_resolution[resolution]
        output_uvs = _uvs_by_position(
            _load_mesh(variants.glb_by_resolution[resolution])
        )
        for x, y in sample_positions:
            expected = _sample_repeat_bilinear(
                reference,
                _interpolate_quad_uv(source_uvs, x, y),
            )
            actual = _sample_repeat_bilinear(
                output_texture,
                _interpolate_quad_uv(output_uvs, x, y),
            )
            np.testing.assert_allclose(
                actual,
                expected,
                atol=tolerance,
                rtol=0.0,
                err_msg=(
                    f"resolution={resolution}, local_position={(x, y)}"
                ),
            )


def _assert_nonblack_neighborhood_is_bounded(
    test: unittest.TestCase,
    variants: object,
    *,
    maximum_margin: int,
) -> None:
    texture = variants.preview_rgba_by_resolution[2048]
    output = _load_mesh(variants.glb_by_resolution[2048])
    uvs = np.asarray(output.visual.uv, dtype=float)
    columns = uvs[:, 0] * _RESOLUTION - 0.5
    rows = (1.0 - uvs[:, 1]) * _RESOLUTION - 0.5
    nonblack_rows, nonblack_columns = np.nonzero(
        np.any(texture != _OPAQUE_BLACK, axis=2)
    )
    test.assertGreater(len(nonblack_rows), 0)
    test.assertGreaterEqual(
        int(np.min(nonblack_columns)),
        math.floor(float(np.min(columns))) - maximum_margin,
    )
    test.assertLessEqual(
        int(np.max(nonblack_columns)),
        math.ceil(float(np.max(columns))) + maximum_margin,
    )
    test.assertGreaterEqual(
        int(np.min(nonblack_rows)),
        math.floor(float(np.min(rows))) - maximum_margin,
    )
    test.assertLessEqual(
        int(np.max(nonblack_rows)),
        math.ceil(float(np.max(rows))) + maximum_margin,
    )


def _assert_canonical_pixels_inverse_sample_source(
    source_glb: bytes,
    source_texture: np.ndarray,
    variants: object,
) -> None:
    source_uvs = _uvs_by_position(_load_mesh(source_glb))
    output_uvs = _uvs_by_position(
        _load_mesh(variants.glb_by_resolution[2048])
    )
    positions = sorted(source_uvs)
    source_points = np.asarray([source_uvs[position] for position in positions])
    output_points = np.asarray([output_uvs[position] for position in positions])
    affine = np.linalg.lstsq(
        np.column_stack((source_points, np.ones(len(source_points)))),
        output_points,
        rcond=None,
    )[0]
    linear = affine[:2]
    translation = affine[2]
    inverse = np.linalg.inv(linear)
    texture = variants.preview_rgba_by_resolution[2048]
    nonblack = np.any(texture != _OPAQUE_BLACK, axis=2)
    sample_pixels: set[tuple[int, int]] = set()
    occupied_rows = np.flatnonzero(np.any(nonblack, axis=1))
    occupied_columns = np.flatnonzero(np.any(nonblack, axis=0))
    for row in occupied_rows[
        np.linspace(0, len(occupied_rows) - 1, 12, dtype=np.int64)
    ]:
        columns = np.flatnonzero(nonblack[row])
        for column in columns[[0, len(columns) // 2, -1]]:
            sample_pixels.add((int(row), int(column)))
    for column in occupied_columns[
        np.linspace(0, len(occupied_columns) - 1, 12, dtype=np.int64)
    ]:
        rows = np.flatnonzero(nonblack[:, column])
        for row in rows[[0, len(rows) // 2, -1]]:
            sample_pixels.add((int(row), int(column)))
    for row, column in sorted(sample_pixels):
        output_uv = np.asarray(
            (
                (column + 0.5) / _RESOLUTION,
                1.0 - (row + 0.5) / _RESOLUTION,
            ),
            dtype=float,
        )
        source_uv = (output_uv - translation) @ inverse
        np.testing.assert_allclose(
            texture[row, column],
            _sample_repeat_bilinear(source_texture, source_uv),
            atol=1.0,
            rtol=0.0,
            err_msg=f"canonical destination pixel={(column, row)}",
        )


# ### Preserved-UV compatibility audit tests ###
class SymmetricPreservedUvSmartFallbackAuditTests(unittest.TestCase):
    def test_rigid_internal_uv_edges_keep_raw_filter_neighborhoods(self) -> None:
        source, source_texture = _internal_edge_quad_glb(
            (302, 404, 703, 805)
        )

        variants = build_symmetric_half_texture_variants(source)

        source_uvs = _uvs_by_position(_load_mesh(source))
        output_uvs = _uvs_by_position(
            _load_mesh(variants.glb_by_resolution[2048])
        )
        source_edge = source_uvs[(1.0, 0.0, 0.0)] - source_uvs[
            (0.0, 0.0, 0.0)
        ]
        output_edge = output_uvs[(1.0, 0.0, 0.0)] - output_uvs[
            (0.0, 0.0, 0.0)
        ]
        self.assertAlmostEqual(
            float(np.linalg.norm(output_edge)),
            float(np.linalg.norm(source_edge)),
            places=7,
        )
        _assert_internal_edge_appearance_at_every_resolution(
            source,
            source_texture,
            variants,
            tolerance=3.0,
        )
        _assert_nonblack_neighborhood_is_bounded(
            self,
            variants,
            maximum_margin=19,
        )
        _assert_only_left_half_is_used(self, variants)

    def test_uniform_internal_uv_edges_restore_core_and_bounded_gutter(self) -> None:
        source, source_texture = _internal_edge_quad_glb(
            (80, 100, 1900, 1900)
        )

        variants = build_symmetric_half_texture_variants(source)

        source_uvs = _uvs_by_position(_load_mesh(source))
        output_uvs = _uvs_by_position(
            _load_mesh(variants.glb_by_resolution[2048])
        )
        source_edge = source_uvs[(1.0, 0.0, 0.0)] - source_uvs[
            (0.0, 0.0, 0.0)
        ]
        output_edge = output_uvs[(1.0, 0.0, 0.0)] - output_uvs[
            (0.0, 0.0, 0.0)
        ]
        self.assertLess(
            float(np.linalg.norm(output_edge)),
            float(np.linalg.norm(source_edge)),
        )
        _assert_canonical_pixels_inverse_sample_source(
            source,
            source_texture,
            variants,
        )
        _assert_nonblack_neighborhood_is_bounded(
            self,
            variants,
            maximum_margin=15,
        )
        _assert_only_left_half_is_used(self, variants)

    def test_boundary_touching_left_chart_repacks_for_repeat_filtering(self) -> None:
        source = _quad_glb(
            ((0.0, 0.0), (0.50, 0.0), (0.50, 0.80), (0.0, 0.80))
        )

        fresh = build_symmetric_half_texture_variants(source)
        compatible = build_symmetric_half_texture_variants(
            source,
            uvs_already_left_packed=True,
        )

        self.assertEqual(compatible.glb_by_resolution, fresh.glb_by_resolution)
        self.assertEqual(
            compatible.texture_png_by_resolution,
            fresh.texture_png_by_resolution,
        )
        source_mesh = _load_mesh(source)
        output_mesh = _load_mesh(compatible.glb_by_resolution[2048])
        source_texture = _first_texture(source)
        output_texture = _first_texture(compatible.glb_by_resolution[2048])
        source_uvs = _uvs_by_position(source_mesh)
        output_uvs = _uvs_by_position(output_mesh)
        for position, source_uv in source_uvs.items():
            np.testing.assert_allclose(
                _sample_repeat_bilinear(output_texture, output_uvs[position]),
                _sample_repeat_bilinear(source_texture, source_uv),
                atol=3.0,
                rtol=0.0,
            )
        self.assertNotEqual(
            build_camera_uv_fingerprint(compatible.glb_by_resolution[2048]),
            build_camera_uv_fingerprint(source),
        )
        _assert_only_left_half_is_used(self, compatible)

    def test_proven_left_packed_uvs_stay_fixed_across_reentry(self) -> None:
        source = _quad_glb(
            ((0.08, 0.20), (0.42, 0.20), (0.42, 0.80), (0.08, 0.80))
        )

        first = build_symmetric_half_texture_variants(
            source,
            uvs_already_left_packed=True,
        )
        second = build_symmetric_half_texture_variants(
            first.glb_by_resolution[2048],
            uvs_already_left_packed=True,
        )

        source_fingerprint = build_camera_uv_fingerprint(source)
        self.assertEqual(
            build_camera_uv_fingerprint(first.glb_by_resolution[2048]),
            source_fingerprint,
        )
        self.assertEqual(
            build_camera_uv_fingerprint(second.glb_by_resolution[2048]),
            source_fingerprint,
        )
        source_uvs = _uvs_by_position(_load_mesh(source))
        first_uvs = _uvs_by_position(_load_mesh(first.glb_by_resolution[2048]))
        second_uvs = _uvs_by_position(_load_mesh(second.glb_by_resolution[2048]))
        self.assertEqual(source_uvs.keys(), first_uvs.keys())
        self.assertEqual(source_uvs.keys(), second_uvs.keys())
        for position in source_uvs:
            np.testing.assert_allclose(first_uvs[position], source_uvs[position], atol=1e-7)
            np.testing.assert_allclose(second_uvs[position], source_uvs[position], atol=1e-7)
        _assert_quad_appearance(self, source, first.glb_by_resolution[2048])
        _assert_only_left_half_is_used(self, first)
        _assert_only_left_half_is_used(self, second)

    def test_legacy_full_atlas_hint_matches_fresh_uniform_repack(self) -> None:
        source = _quad_glb(
            ((0.05, 0.10), (0.95, 0.10), (0.95, 0.90), (0.05, 0.90))
        )

        fresh = build_symmetric_half_texture_variants(source)
        compatible = build_symmetric_half_texture_variants(
            source,
            uvs_already_left_packed=True,
        )
        canonical = compatible.glb_by_resolution[2048]
        reentered = build_symmetric_half_texture_variants(
            canonical,
            uvs_already_left_packed=True,
        )

        self.assertEqual(compatible.glb_by_resolution, fresh.glb_by_resolution)
        self.assertEqual(
            compatible.texture_png_by_resolution,
            fresh.texture_png_by_resolution,
        )
        self.assertNotEqual(
            build_camera_uv_fingerprint(canonical),
            build_camera_uv_fingerprint(source),
        )
        self.assertEqual(
            build_camera_uv_fingerprint(reentered.glb_by_resolution[2048]),
            build_camera_uv_fingerprint(canonical),
        )
        source_by_position = _uvs_by_position(_load_mesh(source))
        output_by_position = _uvs_by_position(_load_mesh(canonical))
        positions = sorted(source_by_position)
        source_uvs = np.asarray([source_by_position[key] for key in positions])
        output_uvs = np.asarray([output_by_position[key] for key in positions])
        source_distances = np.linalg.norm(
            source_uvs[:, np.newaxis] - source_uvs[np.newaxis, :],
            axis=2,
        )
        output_distances = np.linalg.norm(
            output_uvs[:, np.newaxis] - output_uvs[np.newaxis, :],
            axis=2,
        )
        nonzero = source_distances > 1e-8
        scale_ratios = output_distances[nonzero] / source_distances[nonzero]
        self.assertLess(float(scale_ratios[0]), 1.0)
        np.testing.assert_allclose(
            scale_ratios,
            np.full_like(scale_ratios, scale_ratios[0]),
            rtol=2e-5,
            atol=2e-7,
        )
        _assert_quad_appearance(self, source, canonical)
        _assert_only_left_half_is_used(self, compatible)
        _assert_only_left_half_is_used(self, reentered)

    def test_negative_two_axis_repeats_use_the_same_fallback_as_fresh(self) -> None:
        source = _repeated_triangle_glb()

        fresh = build_symmetric_half_texture_variants(source)
        compatible = build_symmetric_half_texture_variants(
            source,
            uvs_already_left_packed=True,
        )

        self.assertEqual(compatible.glb_by_resolution, fresh.glb_by_resolution)
        self.assertEqual(
            compatible.texture_png_by_resolution,
            fresh.texture_png_by_resolution,
        )
        output = _load_mesh(compatible.glb_by_resolution[2048])
        triangles = np.asarray(output.triangles, dtype=float)
        areas = 0.5 * np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            ),
            axis=1,
        )
        self.assertGreater(len(output.faces), 1)
        self.assertTrue(np.all(areas > 1e-10))
        self.assertAlmostEqual(float(np.sum(areas)), 0.5, places=5)
        _assert_repeated_triangle_appearance(
            source,
            compatible.glb_by_resolution[2048],
        )
        _assert_only_left_half_is_used(self, compatible)

    def test_nonfinite_preserved_uvs_remain_an_explicit_validation_error(self) -> None:
        source = _quad_glb(
            ((float("nan"), 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8))
        )

        with self.assertRaises(ValueError) as raised:
            build_symmetric_half_texture_variants(
                source,
                uvs_already_left_packed=True,
            )

        self.assertIn("invalid UV coordinates", str(raised.exception))
        self.assertNotIn("inside the atlas", str(raised.exception))


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
