# ### Imports ###
from __future__ import annotations

import unittest
from io import BytesIO
from unittest.mock import patch

import cv2
import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.texture import TextureVisuals

import housemaker.object_texture_inpaint as inpaint_module
from housemaker.object_texture_inpaint import (
    OBJECT_TEXTURE_INPAINT_RESOLUTION,
    TEXTURE_UV_MODE_PAINT,
    TextureUvStroke,
    build_texture_uv_stamp_stroke_from_screen_brush,
    composite_object_texture_inpaint,
    pick_texture_uv_from_ray,
    rasterize_texture_uv_strokes,
    sample_texture_uv_hits_from_screen_brush,
    validate_object_texture_inpaint_outside_mask,
)


# ### Fixture helpers ###
def _two_island_mesh() -> trimesh.Trimesh:
    vertices = np.asarray(
        (
            (-2.0, -1.0, 0.0),
            (-0.4, -1.0, 0.0),
            (-0.4, 1.0, 0.0),
            (-2.0, 1.0, 0.0),
            (0.4, -1.0, 0.0),
            (2.0, -1.0, 0.0),
            (2.0, 1.0, 0.0),
            (0.4, 1.0, 0.0),
        ),
        dtype=float,
    )
    faces = np.asarray(
        ((0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7)),
        dtype=np.int64,
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual = TextureVisuals(
        uv=np.asarray(
            (
                (0.05, 0.2),
                (0.20, 0.2),
                (0.20, 0.8),
                (0.05, 0.8),
                (0.80, 0.2),
                (0.95, 0.2),
                (0.95, 0.8),
                (0.80, 0.8),
            ),
            dtype=float,
        )
    )
    return mesh


def _orthographic_ray_builder(
    screen_point: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray((screen_point[0], screen_point[1], 1.0), dtype=float),
        np.asarray((0.0, 0.0, -1.0), dtype=float),
    )


def _encode_rgba_png(pixels: np.ndarray) -> bytes:
    did_encode, encoded = cv2.imencode(
        ".png",
        cv2.cvtColor(
            np.ascontiguousarray(pixels, dtype=np.uint8),
            cv2.COLOR_RGBA2BGRA,
        ),
    )
    if not did_encode:
        raise RuntimeError("Test PNG encoding failed.")
    return bytes(encoded)


def _encode_mask_png(mask: np.ndarray) -> bytes:
    did_encode, encoded = cv2.imencode(
        ".png",
        np.ascontiguousarray(mask, dtype=np.uint8),
    )
    if not did_encode:
        raise RuntimeError("Test mask encoding failed.")
    return bytes(encoded)


def _decode_rgba_png(payload: bytes) -> np.ndarray:
    with Image.open(BytesIO(payload)) as image:
        return np.asarray(image.convert("RGBA"), dtype=np.uint8)


# ### Screen-space brush tests ###
class TextureScreenBrushTests(unittest.TestCase):
    def test_surrounding_samples_hit_disconnected_islands_when_center_misses(
        self,
    ) -> None:
        mesh = _two_island_mesh()
        center_ray = _orthographic_ray_builder((0.0, 0.0))
        self.assertIsNone(
            pick_texture_uv_from_ray(mesh, center_ray[0], center_ray[1])
        )

        stroke = build_texture_uv_stamp_stroke_from_screen_brush(
            mesh,
            cursor_position=(0.0, 0.0),
            radius_pixels=1.5,
            ray_builder=_orthographic_ray_builder,
            mode=TEXTURE_UV_MODE_PAINT,
            stamp_radius_normalized=0.015,
            sample_spacing_pixels=0.5,
            maximum_sample_count=25,
        )

        self.assertIsNotNone(stroke)
        assert stroke is not None
        self.assertFalse(stroke.connect_points)
        self.assertTrue(any(point.u < 0.3 for point in stroke.points))
        self.assertTrue(any(point.u > 0.7 for point in stroke.points))
        mask = rasterize_texture_uv_strokes((101, 101), (stroke,))
        self.assertGreater(np.count_nonzero(mask[:, :30]), 0)
        self.assertGreater(np.count_nonzero(mask[:, 70:]), 0)
        self.assertEqual(np.count_nonzero(mask[:, 45:56]), 0)

    def test_sampling_is_deterministic_and_bounded_before_ray_building(
        self,
    ) -> None:
        first_positions: list[tuple[float, float]] = []
        second_positions: list[tuple[float, float]] = []

        def first_builder(position: tuple[float, float]):
            first_positions.append(position)
            return None

        def second_builder(position: tuple[float, float]):
            second_positions.append(position)
            return None

        first_hits = sample_texture_uv_hits_from_screen_brush(
            _two_island_mesh(),
            (12.0, 20.0),
            10_000.0,
            first_builder,
            sample_spacing_pixels=0.01,
            maximum_sample_count=25,
        )
        second_hits = sample_texture_uv_hits_from_screen_brush(
            _two_island_mesh(),
            (12.0, 20.0),
            10_000.0,
            second_builder,
            sample_spacing_pixels=0.01,
            maximum_sample_count=25,
        )

        self.assertEqual(first_hits, ())
        self.assertEqual(second_hits, ())
        self.assertEqual(first_positions, second_positions)
        self.assertEqual(first_positions[0], (12.0, 20.0))
        self.assertLessEqual(len(first_positions), 25)
        self.assertGreater(len(first_positions), 1)

    def test_triangle_arrays_are_prepared_once_for_the_bounded_ray_group(
        self,
    ) -> None:
        original_prepare = inpaint_module._prepare_uv_ray_mesh
        with patch.object(
            inpaint_module,
            "_prepare_uv_ray_mesh",
            wraps=original_prepare,
        ) as prepare_mesh:
            sample_texture_uv_hits_from_screen_brush(
                _two_island_mesh(),
                (0.0, 0.0),
                24.0,
                _orthographic_ray_builder,
                maximum_sample_count=25,
            )

        prepare_mesh.assert_called_once()

    def test_disconnected_stamp_serialization_is_backward_compatible(self) -> None:
        stroke = build_texture_uv_stamp_stroke_from_screen_brush(
            _two_island_mesh(),
            (0.0, 0.0),
            1.5,
            _orthographic_ray_builder,
            mode=TEXTURE_UV_MODE_PAINT,
            stamp_radius_normalized=0.01,
            sample_spacing_pixels=0.5,
            maximum_sample_count=49,
        )
        self.assertIsNotNone(stroke)
        assert stroke is not None
        self.assertEqual(TextureUvStroke.from_dict(stroke.to_dict()), stroke)
        legacy_payload = stroke.to_dict()
        legacy_payload.pop("connect_points")
        self.assertTrue(
            TextureUvStroke.from_dict(legacy_payload).connect_points
        )


# ### Feathered composite tests ###
class ObjectTextureInpaintFeatherTests(unittest.TestCase):
    def test_feather_is_inward_and_every_outside_pixel_stays_exact(self) -> None:
        resolution = OBJECT_TEXTURE_INPAINT_RESOLUTION
        existing = np.full(
            (resolution, resolution, 4),
            (20, 40, 60, 255),
            dtype=np.uint8,
        )
        generated = np.full_like(existing, (220, 180, 140, 255))
        mask = np.zeros((resolution, resolution), dtype=np.uint8)
        mask[100:140, 200:240] = 255
        existing_png = _encode_rgba_png(existing)
        mask_png = _encode_mask_png(mask)

        result_png = composite_object_texture_inpaint(
            existing_png,
            mask_png,
            _encode_rgba_png(generated),
        )
        result = _decode_rgba_png(result_png)

        np.testing.assert_array_equal(result[mask == 0], existing[mask == 0])
        np.testing.assert_array_equal(result[120, 220], generated[120, 220])
        boundary = result[100, 220, :3]
        self.assertTrue(np.all(boundary > existing[100, 220, :3]))
        self.assertTrue(np.all(boundary < generated[100, 220, :3]))
        validate_object_texture_inpaint_outside_mask(
            existing_png,
            mask_png,
            result_png,
        )


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
