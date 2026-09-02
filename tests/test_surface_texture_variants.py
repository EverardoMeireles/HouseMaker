# ### Imports ###
from __future__ import annotations

import unittest
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from housemaker.pbr_maps import (
    ATLAS_MAP_BASE_COLOR,
    PBR_MAP_NORMAL,
)
from housemaker.surface_texture_variants import (
    CANONICAL_SURFACE_TEXTURE_RESOLUTION,
    DEFAULT_SURFACE_TEXTURE_RESOLUTION,
    SURFACE_TEXTURE_RESOLUTIONS,
    SurfaceTextureVariants,
    _repair_flat_normal_map,
    build_surface_texture_variants,
)


# ### Fixture helpers ###
def _encode_png(source_rgba: np.ndarray) -> bytes:
    destination = BytesIO()
    Image.fromarray(source_rgba, mode="RGBA").save(destination, format="PNG")
    return destination.getvalue()


def _decode_png(payload: bytes) -> np.ndarray:
    with Image.open(BytesIO(payload)) as image:
        return np.asarray(image.convert("RGBA"), dtype=np.uint8)


def _nonuniform_2048_rgba() -> np.ndarray:
    """Build pixels whose direct and chained area reductions differ."""

    averaging_block = np.asarray(
        (
            (5, 7, 5, 5),
            (7, 5, 0, 6),
            (7, 4, 6, 0),
            (1, 1, 6, 6),
        ),
        dtype=np.uint8,
    )
    channel = np.tile(averaging_block, (512, 512))
    rgba = np.empty((2048, 2048, 4), dtype=np.uint8)
    rgba[:, :, 0] = channel
    rgba[:, :, 1] = 255 - channel
    rgba[:, :, 2] = channel * 31
    rgba[:, :, 3] = channel * 37
    return rgba


def _detailed_luminance_rgba(height: int = 48, width: int = 64) -> np.ndarray:
    x_coordinates = np.linspace(0.0, 4.0 * np.pi, width, dtype=np.float32)
    y_coordinates = np.linspace(0.0, 3.0 * np.pi, height, dtype=np.float32)
    luminance = (
        128.0
        + 70.0 * np.sin(x_coordinates)[None, :]
        + 35.0 * np.cos(y_coordinates)[:, None]
    )
    channel = np.clip(np.rint(luminance), 0, 255).astype(np.uint8)
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgba[:, :, :3] = channel[:, :, None]
    rgba[:, :, 3] = 255
    return rgba


def _decoded_normal_lengths(normal_rgba: np.ndarray) -> np.ndarray:
    vectors = normal_rgba[:, :, :3].astype(np.float32) / 127.5 - 1.0
    return np.linalg.norm(vectors, axis=2)


def _detailed_normal_rgba(height: int = 24, width: int = 36) -> np.ndarray:
    x_coordinates = np.linspace(0.0, 2.0 * np.pi, width, dtype=np.float32)
    y_coordinates = np.linspace(0.0, 2.0 * np.pi, height, dtype=np.float32)
    normal_x = 0.45 * np.sin(x_coordinates)[None, :]
    normal_y = 0.35 * np.cos(y_coordinates)[:, None]
    normal_x = np.broadcast_to(normal_x, (height, width))
    normal_y = np.broadcast_to(normal_y, (height, width))
    normal_z = np.sqrt(np.maximum(1.0 - normal_x**2 - normal_y**2, 0.0))
    vectors = np.stack((normal_x, normal_y, normal_z), axis=2)
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgba[:, :, :3] = np.clip(
        np.rint((vectors * 0.5 + 0.5) * 255.0),
        0,
        255,
    ).astype(np.uint8)
    rgba[:, :, 3] = 255
    return rgba


def _expected_resized_normal(
    normal_rgba: np.ndarray,
    resolution: int,
) -> np.ndarray:
    vectors = normal_rgba[:, :, :3].astype(np.float32) / 127.5 - 1.0
    vectors = cv2.resize(
        vectors,
        (resolution, resolution),
        interpolation=cv2.INTER_LANCZOS4,
    )
    lengths = np.linalg.norm(vectors, axis=2, keepdims=True)
    normalized = vectors / np.maximum(lengths, 1e-6)
    expected = np.empty((resolution, resolution, 4), dtype=np.uint8)
    expected[:, :, :3] = np.clip(
        np.rint((normalized * 0.5 + 0.5) * 255.0),
        0,
        255,
    ).astype(np.uint8)
    expected[:, :, 3] = 255
    return expected


def _four_pixel_period_normal_rgba(
    height: int = 64,
    width: int = 64,
) -> np.ndarray:
    normal_x = np.asarray((-0.45, -0.15, 0.15, 0.45), dtype=np.float32)
    normal_x = np.tile(normal_x, width // 4 + 1)[:width]
    normal_x = np.broadcast_to(normal_x[None, :], (height, width))
    normal_y = np.zeros((height, width), dtype=np.float32)
    normal_z = np.sqrt(np.maximum(1.0 - normal_x**2, 0.0))
    vectors = np.stack((normal_x, normal_y, normal_z), axis=2)
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgba[:, :, :3] = np.clip(
        np.rint((vectors * 0.5 + 0.5) * 255.0),
        0,
        255,
    ).astype(np.uint8)
    rgba[:, :, 3] = 255
    return rgba


# ### Variant algorithm tests ###
class SurfaceTextureVariantAlgorithmTests(unittest.TestCase):
    def test_native_2048_pixels_are_preserved_and_downsizes_are_direct(self) -> None:
        source_rgba = _nonuniform_2048_rgba()

        variants = build_surface_texture_variants(_encode_png(source_rgba))

        self.assertEqual(
            set(variants.texture_png_by_resolution),
            set(SURFACE_TEXTURE_RESOLUTIONS),
        )
        canonical = _decode_png(
            variants.texture_png_by_resolution[
                CANONICAL_SURFACE_TEXTURE_RESOLUTION
            ]
        )
        texture_1024 = _decode_png(variants.texture_png_by_resolution[1024])
        texture_512 = _decode_png(variants.texture_png_by_resolution[512])
        expected_1024 = cv2.resize(
            source_rgba,
            (1024, 1024),
            interpolation=cv2.INTER_AREA,
        )
        expected_512 = cv2.resize(
            source_rgba,
            (512, 512),
            interpolation=cv2.INTER_AREA,
        )
        chained_512 = cv2.resize(
            expected_1024,
            (512, 512),
            interpolation=cv2.INTER_AREA,
        )

        np.testing.assert_array_equal(canonical, source_rgba)
        np.testing.assert_array_equal(texture_1024, expected_1024)
        np.testing.assert_array_equal(texture_512, expected_512)
        self.assertTrue(np.any(texture_512 != chained_512))

    def test_non_square_source_is_normalized_once_before_direct_downsizes(
        self,
    ) -> None:
        source_rgba = np.arange(8 * 12 * 4, dtype=np.uint8).reshape((8, 12, 4))

        variants = build_surface_texture_variants(_encode_png(source_rgba))

        expected_canonical = cv2.resize(
            source_rgba,
            (2048, 2048),
            interpolation=cv2.INTER_LANCZOS4,
        )
        canonical = _decode_png(variants.texture_png_by_resolution[2048])
        texture_512 = _decode_png(variants.texture_png_by_resolution[512])
        np.testing.assert_array_equal(canonical, expected_canonical)
        np.testing.assert_array_equal(
            texture_512,
            cv2.resize(
                expected_canonical,
                (512, 512),
                interpolation=cv2.INTER_AREA,
            ),
        )

    def test_invalid_png_inputs_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            build_surface_texture_variants("not bytes")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            build_surface_texture_variants(b"not a png")

    def test_pbr_maps_receive_every_exact_resolution(self) -> None:
        base = np.full((8, 12, 4), (20, 40, 60, 255), dtype=np.uint8)
        normal = np.full((8, 12, 4), (128, 128, 255, 255), dtype=np.uint8)

        variants = build_surface_texture_variants(
            _encode_png(base),
            {PBR_MAP_NORMAL: _encode_png(normal)},
        )

        self.assertEqual(
            variants.available_map_types,
            (ATLAS_MAP_BASE_COLOR, PBR_MAP_NORMAL),
        )
        assert variants.map_png_by_resolution is not None
        self.assertEqual(
            variants.map_png_by_resolution[ATLAS_MAP_BASE_COLOR],
            variants.texture_png_by_resolution,
        )
        for resolution in SURFACE_TEXTURE_RESOLUTIONS:
            decoded = _decode_png(
                variants.map_png_by_resolution[PBR_MAP_NORMAL][resolution]
            )
            self.assertEqual(decoded.shape[:2], (resolution, resolution))
            np.testing.assert_array_equal(decoded[0, 0], normal[0, 0])

    def test_pbr_maps_are_normalized_independently_from_base_color(self) -> None:
        base = np.full((8, 12, 4), (20, 40, 60, 255), dtype=np.uint8)
        normal = np.full((9, 7, 4), (128, 128, 255, 255), dtype=np.uint8)

        variants = build_surface_texture_variants(
            _encode_png(base),
            {PBR_MAP_NORMAL: _encode_png(normal)},
        )

        assert variants.map_png_by_resolution is not None
        for resolution in SURFACE_TEXTURE_RESOLUTIONS:
            self.assertEqual(
                _decode_png(
                    variants.map_png_by_resolution[PBR_MAP_NORMAL][resolution]
                ).shape[:2],
                (resolution, resolution),
            )

    def test_flat_provider_normal_is_repaired_from_detailed_base_color(self) -> None:
        base = _detailed_luminance_rgba()
        flat_normal = np.full(
            (37, 53, 4),
            (128, 128, 255, 255),
            dtype=np.uint8,
        )

        variants = build_surface_texture_variants(
            _encode_png(base),
            {PBR_MAP_NORMAL: _encode_png(flat_normal)},
        )

        assert variants.map_png_by_resolution is not None
        repaired = _decode_png(
            variants.map_png_by_resolution[PBR_MAP_NORMAL][2048]
        )
        tangent_xy = repaired[:, :, :2].astype(np.float32) / 127.5 - 1.0
        slope_magnitude = np.linalg.norm(tangent_xy, axis=2)
        self.assertGreater(float(np.percentile(slope_magnitude, 95.0)), 0.08)
        self.assertGreater(float(np.std(repaired[:, :, 0])), 2.0)

    def test_uniform_base_color_keeps_flat_provider_normal(self) -> None:
        base = np.full((31, 47, 4), (90, 90, 90, 255), dtype=np.uint8)
        flat_normal = np.full(
            (23, 39, 4),
            (128, 128, 255, 255),
            dtype=np.uint8,
        )

        variants = build_surface_texture_variants(
            _encode_png(base),
            {PBR_MAP_NORMAL: _encode_png(flat_normal)},
        )

        assert variants.map_png_by_resolution is not None
        for resolution in SURFACE_TEXTURE_RESOLUTIONS:
            decoded = _decode_png(
                variants.map_png_by_resolution[PBR_MAP_NORMAL][resolution]
            )
            np.testing.assert_array_equal(decoded[0, 0], flat_normal[0, 0])
            self.assertEqual(np.unique(decoded.reshape((-1, 4)), axis=0).shape[0], 1)

    def test_detailed_provider_normal_is_not_replaced_from_base_color(self) -> None:
        base = _detailed_luminance_rgba()
        provider_normal = _detailed_normal_rgba()

        variants = build_surface_texture_variants(
            _encode_png(base),
            {PBR_MAP_NORMAL: _encode_png(provider_normal)},
        )

        assert variants.map_png_by_resolution is not None
        canonical = _decode_png(
            variants.map_png_by_resolution[PBR_MAP_NORMAL][2048]
        )
        np.testing.assert_array_equal(
            canonical,
            _expected_resized_normal(provider_normal, 2048),
        )

    def test_four_pixel_period_normal_detail_is_not_classified_as_flat(self) -> None:
        base = _detailed_luminance_rgba(height=64, width=64)
        provider_normal = _four_pixel_period_normal_rgba()

        repaired = _repair_flat_normal_map(base, provider_normal)

        np.testing.assert_array_equal(repaired, provider_normal)

    def test_normal_vectors_remain_unit_length_at_every_resolution(self) -> None:
        base = _detailed_luminance_rgba()
        flat_normal = np.full(
            (29, 41, 4),
            (128, 128, 255, 255),
            dtype=np.uint8,
        )

        variants = build_surface_texture_variants(
            _encode_png(base),
            {PBR_MAP_NORMAL: _encode_png(flat_normal)},
        )

        assert variants.map_png_by_resolution is not None
        for resolution in SURFACE_TEXTURE_RESOLUTIONS:
            decoded = _decode_png(
                variants.map_png_by_resolution[PBR_MAP_NORMAL][resolution]
            )
            lengths = _decoded_normal_lengths(decoded)
            self.assertLess(float(np.max(np.abs(lengths - 1.0))), 0.01)
            self.assertTrue(np.all(decoded[:, :, 3] == 255))

    def test_variant_container_requires_every_exact_resolution(self) -> None:
        self.assertEqual(DEFAULT_SURFACE_TEXTURE_RESOLUTION, 1024)
        with self.assertRaises(ValueError):
            SurfaceTextureVariants({512: b"png"})


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
