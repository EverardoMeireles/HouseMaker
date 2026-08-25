# ### Imports ###
from __future__ import annotations

import unittest
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from housemaker.surface_texture_variants import (
    CANONICAL_SURFACE_TEXTURE_RESOLUTION,
    DEFAULT_SURFACE_TEXTURE_RESOLUTION,
    SURFACE_TEXTURE_RESOLUTIONS,
    SurfaceTextureVariants,
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

    def test_variant_container_requires_every_exact_resolution(self) -> None:
        self.assertEqual(DEFAULT_SURFACE_TEXTURE_RESOLUTION, 1024)
        with self.assertRaises(ValueError):
            SurfaceTextureVariants({512: b"png"})


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
