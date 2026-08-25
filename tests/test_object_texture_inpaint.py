# ### Imports ###
from __future__ import annotations

import unittest

import numpy as np

from housemaker.object_texture_inpaint import (
    TEXTURE_UV_MODE_ERASE,
    TEXTURE_UV_MODE_PAINT,
    TextureUvPoint,
    TextureUvStroke,
    pick_texture_uv_from_ray,
    rasterize_texture_uv_strokes,
)


# ### Fixture helpers ###
class _VisualWithoutUv:
    uv = None


class _MeshWithoutUv:
    vertices = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=float,
    )
    faces = np.asarray(((0, 1, 2),), dtype=np.int64)
    visual = _VisualWithoutUv()


# ### UV model tests ###
class TextureUvModelTests(unittest.TestCase):
    def test_stroke_round_trips_without_losing_normalized_coordinates(self) -> None:
        stroke = TextureUvStroke(
            mode=TEXTURE_UV_MODE_PAINT,
            radius_normalized=0.025,
            points=(
                TextureUvPoint(0.1, 0.2),
                TextureUvPoint(0.8, 0.9),
            ),
        )

        self.assertEqual(
            TextureUvStroke.from_dict(stroke.to_dict()),
            stroke,
        )

    def test_invalid_points_modes_and_radii_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "U must"):
            TextureUvPoint(-0.1, 0.5)
        with self.assertRaisesRegex(ValueError, "finite"):
            TextureUvPoint(0.5, float("nan"))
        with self.assertRaisesRegex(ValueError, "Unknown"):
            TextureUvStroke("unknown", 0.1, (TextureUvPoint(0.5, 0.5),))
        with self.assertRaisesRegex(ValueError, "radius"):
            TextureUvStroke(
                TEXTURE_UV_MODE_ERASE,
                0.0,
                (TextureUvPoint(0.5, 0.5),),
            )

    def test_rasterizer_returns_an_owned_empty_mask_for_invalid_size(self) -> None:
        result = rasterize_texture_uv_strokes((0, 100), ())

        self.assertEqual(result.shape, (0, 0))
        self.assertEqual(result.dtype, np.uint8)

    def test_ray_picker_handles_missing_uvs_misses_and_invalid_directions(self) -> None:
        mesh = _MeshWithoutUv()

        self.assertIsNone(
            pick_texture_uv_from_ray(
                mesh,
                (0.25, 0.25, 1.0),
                (0.0, 0.0, -1.0),
            )
        )
        with self.assertRaisesRegex(ValueError, "zero length"):
            pick_texture_uv_from_ray(
                mesh,
                (0.25, 0.25, 1.0),
                (0.0, 0.0, 0.0),
            )


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
