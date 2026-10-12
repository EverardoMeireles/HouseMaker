# ### Imports ###
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.glb import (
    GeneratedModel,
    PlacedGeneratedModel,
    compose_generated_model_instance_sources,
    compose_placed_generated_models,
    compose_placed_generated_models_preview,
    export_glb_file,
    import_generated_glb,
    import_generated_glb_preview,
)


# ### Fixture helpers ###
def _build_large_textured_glb() -> bytes:
    base_color = np.empty((1024, 1024, 4), dtype=np.uint8)
    base_color[:, :] = (180, 70, 35, 210)
    normal = np.empty_like(base_color)
    normal[:, :] = (128, 128, 255, 255)
    packed = np.empty_like(base_color)
    packed[:, :] = (255, 91, 37, 255)
    mesh = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    mesh.visual = TextureVisuals(
        uv=np.asarray(
            (
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 1.0),
                (0.0, 1.0),
                (0.2, 0.2),
                (0.8, 0.2),
                (0.8, 0.8),
                (0.2, 0.8),
            ),
            dtype=float,
        ),
        material=PBRMaterial(
            name="Authored oak",
            baseColorFactor=(0.7, 0.8, 0.9, 0.65),
            baseColorTexture=Image.fromarray(base_color, mode="RGBA"),
            normalTexture=Image.fromarray(normal, mode="RGBA"),
            metallicRoughnessTexture=Image.fromarray(packed, mode="RGBA"),
            metallicFactor=0.4,
            roughnessFactor=0.6,
            alphaMode="BLEND",
            alphaCutoff=0.31,
            doubleSided=True,
        ),
    )
    return bytes(trimesh.Scene(mesh).export(file_type="glb"))


def _first_mesh(model: GeneratedModel) -> trimesh.Trimesh:
    mesh = next(iter(model.scene.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("The fixture did not load as a triangle mesh.")
    return mesh


def _solid_proxy(
    color: tuple[int, int, int, int],
    *,
    height: int = 128,
    width: int = 256,
) -> np.ndarray:
    proxy = np.empty((height, width, 4), dtype=np.uint8)
    proxy[:, :] = color
    return proxy


# ### Preview import tests ###
class GeneratedGlbPreviewImportTests(unittest.TestCase):
    def test_preview_bypasses_embedded_images_and_preserves_authoritative_data(
        self,
    ) -> None:
        payload = _build_large_textured_glb()
        authoritative = import_generated_glb(payload)
        authoritative_mesh = _first_mesh(authoritative)
        authoritative_material = authoritative_mesh.visual.material
        opened_image_sizes: list[tuple[int, int]] = []
        original_open = Image.open

        def tracked_open(*args, **kwargs):
            image = original_open(*args, **kwargs)
            opened_image_sizes.append(tuple(image.size))
            return image

        base_proxy = _solid_proxy((12, 34, 56, 255))
        normal_proxy = _solid_proxy((128, 128, 255, 255))
        roughness_proxy = _solid_proxy((77, 77, 77, 255))
        metallic_proxy = _solid_proxy((99, 99, 99, 255))
        with patch("PIL.Image.open", side_effect=tracked_open):
            preview = import_generated_glb_preview(
                payload,
                base_color_rgba=base_proxy,
                normal_rgba=normal_proxy,
                roughness_rgba=roughness_proxy,
                metallic_rgba=metallic_proxy,
            )

        preview_mesh = _first_mesh(preview)
        preview_material = preview_mesh.visual.material
        self.assertTrue(preview.preview_only)
        self.assertEqual(preview.glb_bytes, payload)
        self.assertNotIn((1024, 1024), opened_image_sizes)
        self.assertTrue(opened_image_sizes)
        self.assertTrue(all(max(size) == 1 for size in opened_image_sizes))
        np.testing.assert_allclose(preview.mesh.vertices, authoritative.mesh.vertices)
        np.testing.assert_array_equal(preview.mesh.faces, authoritative.mesh.faces)
        np.testing.assert_allclose(preview_mesh.visual.uv, authoritative_mesh.visual.uv)
        self.assertEqual(preview_material.name, authoritative_material.name)
        np.testing.assert_array_equal(
            preview_material.baseColorFactor,
            authoritative_material.baseColorFactor,
        )
        self.assertEqual(preview_material.alphaMode, authoritative_material.alphaMode)
        self.assertEqual(
            preview_material.alphaCutoff,
            authoritative_material.alphaCutoff,
        )
        self.assertEqual(preview_material.doubleSided, authoritative_material.doubleSided)
        self.assertEqual(preview_material.baseColorTexture.size, (256, 128))
        np.testing.assert_array_equal(
            np.asarray(preview_material.baseColorTexture),
            base_proxy,
        )
        np.testing.assert_array_equal(
            np.asarray(preview_material.normalTexture),
            normal_proxy,
        )
        np.testing.assert_array_equal(
            np.asarray(preview_material.roughnessTexture),
            roughness_proxy,
        )
        np.testing.assert_array_equal(
            np.asarray(preview_material.metallicTexture),
            metallic_proxy,
        )

    def test_preview_only_model_cannot_reach_serialized_export(self) -> None:
        payload = _build_large_textured_glb()
        preview = import_generated_glb_preview(
            payload,
            base_color_rgba=_solid_proxy((20, 40, 60, 255)),
        )
        base = GeneratedModel(
            mesh=trimesh.Trimesh(process=False),
            scene=trimesh.Scene(),
            glb_bytes=b"",
        )
        placement = PlacedGeneratedModel(
            object_id="proxy-chair",
            model=preview,
            world_position=(0.0, 0.0, 0.0),
        )

        composed = compose_placed_generated_models_preview(base, (placement,))

        self.assertTrue(composed.preview_only)
        with self.assertRaisesRegex(ValueError, "Preview-only"):
            compose_placed_generated_models(base, (placement,))
        with self.assertRaisesRegex(ValueError, "Preview-only"):
            compose_generated_model_instance_sources(base, (placement,))
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "preview.glb"
            with self.assertRaisesRegex(ValueError, "Preview-only"):
                export_glb_file(preview, output_path)
            self.assertFalse(output_path.exists())

    def test_preview_proxy_requires_bounded_rgba_pixels(self) -> None:
        payload = _build_large_textured_glb()

        with self.assertRaisesRegex(ValueError, "cannot exceed 256"):
            import_generated_glb_preview(
                payload,
                base_color_rgba=np.zeros((257, 1, 4), dtype=np.uint8),
            )
        with self.assertRaisesRegex(ValueError, "RGBA uint8"):
            import_generated_glb_preview(
                payload,
                base_color_rgba=np.zeros((16, 16, 3), dtype=np.uint8),
            )


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
