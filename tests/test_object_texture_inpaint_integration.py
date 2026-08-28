# ### Imports ###
from __future__ import annotations

import threading
import unittest
from io import BytesIO
from unittest.mock import Mock

import cv2
import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.uv_integrity import build_uv_fingerprint
from housemaker.object_texture_inpaint import (
    OBJECT_TEXTURE_INPAINT_RESOLUTION,
    TEXTURE_UV_MODE_ERASE,
    TEXTURE_UV_MODE_PAINT,
    DefaultObjectTextureInpaintProvider,
    ObjectTextureInpaintCancelled,
    ObjectTextureInpaintRequest,
    TextureUvPoint,
    TextureUvStroke,
    composite_object_texture_inpaint,
    pick_texture_uv_from_ray,
    rasterize_texture_uv_strokes,
    validate_object_texture_inpaint_outside_mask,
)
from housemaker.object_texture_variants import (
    TEXTURE_RESOLUTIONS,
    build_object_texture_variants_from_texture,
)
from housemaker.surface_texture_providers import SurfaceTextureResult


# ### Fixture helpers ###
def _encode_rgba_png(pixels: np.ndarray) -> bytes:
    rgba = np.ascontiguousarray(pixels, dtype=np.uint8)
    did_encode, encoded = cv2.imencode(
        ".png",
        cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA),
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


def _inpaint_images() -> tuple[bytes, bytes, np.ndarray]:
    resolution = OBJECT_TEXTURE_INPAINT_RESOLUTION
    base = np.full((resolution, resolution, 4), (11, 22, 33, 44), dtype=np.uint8)
    base[0, 0] = (1, 2, 3, 4)
    base[-1, -1] = (251, 252, 253, 254)
    mask = np.zeros((resolution, resolution), dtype=np.uint8)
    mask[100:112, 200:216] = 73
    return _encode_rgba_png(base), _encode_mask_png(mask), base


def _request() -> ObjectTextureInpaintRequest:
    existing, mask, _base = _inpaint_images()
    reference = _encode_rgba_png(
        np.full((4, 4, 4), (90, 80, 70, 255), dtype=np.uint8)
    )
    return ObjectTextureInpaintRequest(
        object_id="chair",
        provider="meshy",
        api_key="msy-secret",
        reference_pngs=(reference,),
        prompt="repair the scratched wood",
        existing_texture_png=existing,
        edit_mask_png=mask,
    )


def _textured_glb() -> bytes:
    mesh = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    mesh.visual = TextureVisuals(
        uv=np.linspace(0.05, 0.95, len(mesh.vertices) * 2).reshape((-1, 2)),
        material=PBRMaterial(
            baseColorTexture=Image.new(
                "RGBA",
                (
                    OBJECT_TEXTURE_INPAINT_RESOLUTION,
                    OBJECT_TEXTURE_INPAINT_RESOLUTION,
                ),
                (25, 50, 75, 255),
            )
        ),
    )
    scene = trimesh.Scene()
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = (4.0, 5.0, 6.0)
    scene.add_geometry(
        mesh,
        geom_name="box",
        node_name="box-node",
        transform=transform,
    )
    return bytes(scene.export(file_type="glb"))


# ### UV paint tests ###
class TextureUvPaintTests(unittest.TestCase):
    def test_rasterization_uses_bottom_origin_uvs_and_replays_erasure(self) -> None:
        top_point = TextureUvPoint(0.5, 1.0)
        paint = TextureUvStroke(
            TEXTURE_UV_MODE_PAINT,
            0.05,
            (top_point,),
        )

        painted = rasterize_texture_uv_strokes((21, 21), (paint,))

        self.assertEqual(painted.shape, (21, 21))
        self.assertEqual(painted.dtype, np.uint8)
        self.assertEqual(painted[0, 10], 255)
        self.assertEqual(painted[20, 10], 0)

        erased = rasterize_texture_uv_strokes(
            (21, 21),
            (
                paint,
                TextureUvStroke(
                    TEXTURE_UV_MODE_ERASE,
                    0.05,
                    (top_point,),
                ),
            ),
        )

        self.assertEqual(np.count_nonzero(erased), 0)

    def test_ray_pick_returns_nearest_interpolated_uv(self) -> None:
        vertices = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, -1.0),
                (1.0, 0.0, -1.0),
                (0.0, 1.0, -1.0),
            ),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.visual = TextureVisuals(
            uv=np.asarray(
                (
                    (0.0, 0.0),
                    (1.0, 0.0),
                    (0.0, 1.0),
                    (0.5, 0.5),
                    (1.0, 0.5),
                    (0.5, 1.0),
                ),
                dtype=float,
            )
        )

        hit = pick_texture_uv_from_ray(
            mesh,
            ray_origin=(0.25, 0.25, 1.0),
            ray_direction=(0.0, 0.0, -1.0),
        )

        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.face_index, 0)
        self.assertAlmostEqual(hit.distance, 1.0)
        self.assertAlmostEqual(hit.point.u, 0.25)
        self.assertAlmostEqual(hit.point.v, 0.25)
        self.assertFalse(hit.is_back_facing)
        self.assertAlmostEqual(sum(hit.barycentric_weights), 1.0)


# ### Request and provider tests ###
class ObjectTextureInpaintProviderTests(unittest.TestCase):
    def test_request_owns_inputs_and_normalizes_its_binary_mask(self) -> None:
        existing, mask, _base = _inpaint_images()
        reference = bytearray(
            _encode_rgba_png(
                np.full((2, 2, 4), (5, 6, 7, 255), dtype=np.uint8)
            )
        )
        existing_buffer = bytearray(existing)
        mask_buffer = bytearray(mask)

        request = ObjectTextureInpaintRequest(
            object_id="  chair  ",
            provider="  meshy  ",
            api_key="  msy-secret  ",
            reference_pngs=(reference,),
            prompt="  repair the wood  ",
            existing_texture_png=existing_buffer,
            edit_mask_png=mask_buffer,
        )
        reference[:] = b"changed"
        existing_buffer[:] = b"changed"
        mask_buffer[:] = b"changed"

        self.assertEqual(request.object_id, "chair")
        self.assertEqual(request.provider, "meshy")
        self.assertEqual(request.api_key, "msy-secret")
        self.assertEqual(request.prompt, "repair the wood")
        self.assertTrue(request.reference_pngs[0].startswith(b"\x89PNG"))
        normalized_mask = cv2.imdecode(
            np.frombuffer(request.edit_mask_png, dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE,
        )
        self.assertEqual(set(np.unique(normalized_mask)), {0, 255})
        self.assertNotIn("msy-secret", repr(request))

    def test_provider_forwards_partial_edit_and_enforces_exact_outside_pixels(
        self,
    ) -> None:
        request = _request()
        generated = _encode_rgba_png(
            np.full((8, 8, 4), (201, 202, 203, 204), dtype=np.uint8)
        )
        requester = Mock(
            return_value=SurfaceTextureResult(
                provider="meshy",
                texture_png=generated,
                task_id="inpaint-task",
            )
        )
        cancel_event = threading.Event()
        progress = Mock()

        result = DefaultObjectTextureInpaintProvider(requester).inpaint(
            request,
            progress,
            cancel_event,
        )

        requester.assert_called_once_with(
            "meshy",
            "msy-secret",
            request.reference_pngs,
            "repair the scratched wood",
            existing_texture_png=request.existing_texture_png,
            edit_mask_png=request.edit_mask_png,
            progress_callback=progress,
            cancel_event=cancel_event,
        )
        self.assertEqual(result.object_id, "chair")
        self.assertEqual(result.provider, "meshy")
        self.assertEqual(result.task_id, "inpaint-task")
        base = _decode_rgba_png(request.existing_texture_png)
        mask = cv2.imdecode(
            np.frombuffer(request.edit_mask_png, dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE,
        )
        output = _decode_rgba_png(result.texture_png)
        np.testing.assert_array_equal(output[mask == 0], base[mask == 0])
        self.assertTrue(
            np.any(output[mask > 0] != base[mask > 0])
        )
        boundary_pixel = output[100, 200].astype(np.int16)
        self.assertTrue(
            np.all(boundary_pixel > base[100, 200].astype(np.int16))
        )
        self.assertTrue(
            np.all(
                boundary_pixel
                < np.asarray((201, 202, 203, 204), dtype=np.int16)
            )
        )

    def test_provider_honors_cancellation_before_external_request(self) -> None:
        requester = Mock()
        cancel_event = threading.Event()
        cancel_event.set()

        with self.assertRaises(ObjectTextureInpaintCancelled):
            DefaultObjectTextureInpaintProvider(requester).inpaint(
                _request(),
                cancel_event=cancel_event,
            )

        requester.assert_not_called()


# ### Mask enforcement tests ###
class ObjectTextureInpaintMaskTests(unittest.TestCase):
    def test_composite_and_validator_reject_any_rogue_outside_pixel(self) -> None:
        existing, mask, base = _inpaint_images()
        generated = _encode_rgba_png(
            np.full_like(base, (150, 140, 130, 120), dtype=np.uint8)
        )

        composited = composite_object_texture_inpaint(
            existing,
            mask,
            generated,
        )

        validate_object_texture_inpaint_outside_mask(
            existing,
            mask,
            composited,
        )
        rogue = _decode_rgba_png(composited).copy()
        rogue[0, 0] = (9, 9, 9, 9)
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_object_texture_inpaint_outside_mask(
                existing,
                mask,
                _encode_rgba_png(rogue),
            )


# ### Variant integrity tests ###
class ObjectTextureInpaintVariantTests(unittest.TestCase):
    def test_replacement_variants_preserve_geometry_transforms_and_uvs(
        self,
    ) -> None:
        source_glb = _textured_glb()
        replacement = np.full(
            (
                OBJECT_TEXTURE_INPAINT_RESOLUTION,
                OBJECT_TEXTURE_INPAINT_RESOLUTION,
                4,
            ),
            (210, 120, 30, 240),
            dtype=np.uint8,
        )
        source_scene = trimesh.load(
            BytesIO(source_glb),
            file_type="glb",
            force="scene",
            process=False,
        )
        source_fingerprint = build_uv_fingerprint(source_glb)

        variants = build_object_texture_variants_from_texture(
            source_glb,
            _encode_rgba_png(replacement),
        )

        np.testing.assert_array_equal(
            variants.preview_rgba_by_resolution[2048],
            replacement,
        )
        for resolution in TEXTURE_RESOLUTIONS:
            variant_glb = variants.glb_by_resolution[resolution]
            self.assertEqual(
                build_uv_fingerprint(variant_glb),
                source_fingerprint,
            )
            variant_scene = trimesh.load(
                BytesIO(variant_glb),
                file_type="glb",
                force="scene",
                process=False,
            )
            self.assertEqual(set(variant_scene.geometry), set(source_scene.geometry))
            for geometry_name in source_scene.geometry:
                source_geometry = source_scene.geometry[geometry_name]
                variant_geometry = variant_scene.geometry[geometry_name]
                np.testing.assert_array_equal(
                    variant_geometry.faces,
                    source_geometry.faces,
                )
                np.testing.assert_allclose(
                    variant_geometry.vertices,
                    source_geometry.vertices,
                )
                np.testing.assert_allclose(
                    variant_geometry.visual.uv,
                    source_geometry.visual.uv,
                )
            np.testing.assert_allclose(
                variant_scene.graph.get("box-node")[0],
                source_scene.graph.get("box-node")[0],
            )


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
