# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.generation_workspace import (
    SELECTED_TEXTURE_RESOLUTION_PIPELINE_KEY,
    SYMMETRIC_DIVISION_PIPELINE_KEY,
    TEXTURE_VARIANTS_PIPELINE_KEY,
    GenerationRequest,
    GenerationWorkspace,
)
from housemaker.glb import import_generated_glb
from housemaker.meshy_generation import MeshyGenerationResult
from housemaker.object_symmetry import (
    AUTOMATIC_SYMMETRIC_DIVISION_METADATA_VERSION,
    SYMMETRIC_DIVISION_ORIENTATION_VERTICAL,
    SYMMETRIC_TEXTURE_CONTENT_HALF_LEFT,
    SYMMETRIC_TEXTURE_PACKING_MODE_PAIR,
)
from housemaker.object_texture_variants import build_object_texture_variants
from housemaker.settings_widget import GenerationServiceSettings


# ### Test application ###
_qt_application = QApplication.instance() or QApplication([])


# ### Fixture helpers ###
def _periodic_texture() -> np.ndarray:
    coordinates = np.arange(2048, dtype=np.uint16)
    texture = np.empty((2048, 2048, 4), dtype=np.uint8)
    texture[:, :, 0] = (coordinates[np.newaxis, :] % 251).astype(np.uint8)
    texture[:, :, 1] = (coordinates[:, np.newaxis] % 241).astype(np.uint8)
    texture[:, :, 2] = 117
    texture[:, :, 3] = 255
    return texture


def _repeated_uv_asymmetric_glb() -> bytes:
    """Build one left triangle and two right triangles for deterministic choice."""

    vertices = np.asarray(
        (
            (-2.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (-1.5, 0.0, 1.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
        ),
        dtype=float,
    )
    faces = np.asarray(
        ((0, 1, 2), (3, 4, 5), (3, 5, 6)),
        dtype=np.int64,
    )
    uv = np.asarray(
        (
            (-0.8, -0.7),
            (1.2, -0.3),
            (0.3, 1.4),
            (0.1, 0.1),
            (0.9, 0.1),
            (0.9, 0.9),
            (0.1, 0.9),
        ),
        dtype=float,
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual = TextureVisuals(
        uv=uv,
        material=PBRMaterial(
            name="repeated-material",
            baseColorTexture=Image.fromarray(
                _periodic_texture(),
                mode="RGBA",
            ),
        ),
    )
    return bytes(trimesh.Scene(mesh).export(file_type="glb"))


def _load_mesh(payload: bytes) -> trimesh.Trimesh:
    scene = trimesh.load(
        io.BytesIO(payload),
        file_type="glb",
        force="scene",
        process=False,
    )
    if not isinstance(scene, trimesh.Scene):
        raise AssertionError("The integration fixture did not load as a scene.")
    mesh = scene.to_geometry()
    if not isinstance(mesh, trimesh.Trimesh):
        raise AssertionError("The integration fixture did not contain a mesh.")
    return mesh


# ### End-to-end Generation integration ###
class GenerationAutomaticSymmetryIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.asset_directory = (
            Path(self.temporary_directory.name) / "generated"
        )
        self.workspace = GenerationWorkspace(
            asset_directory=self.asset_directory
        )

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        self.temporary_directory.cleanup()

    def test_real_generate_commits_square_pair_variants(
        self,
    ) -> None:
        source_glb = _repeated_uv_asymmetric_glb()
        source_variants = build_object_texture_variants(source_glb)
        assert source_variants is not None
        source_model = import_generated_glb(
            source_variants.glb_by_resolution[1024]
        )
        source_model.object_texture_variants = source_variants
        self.workspace._active_generation_request = GenerationRequest(
            frame_index=4,
            selected_object_bgra=np.full((4, 4, 4), 255, dtype=np.uint8),
            settings=GenerationServiceSettings(meshy_api_key="key"),
            symmetric_division_enabled=True,
            symmetric_division_orientation=(
                SYMMETRIC_DIVISION_ORIENTATION_VERTICAL
            ),
        )
        generated_spy = QSignalSpy(self.workspace.generation_completed)

        self.workspace._handle_generation_succeeded(
            MeshyGenerationResult("real-auto", source_glb, "Repeated UV"),
            source_model,
        )

        self.assertEqual(generated_spy.count(), 1)
        record = self.workspace.get_data().generated_objects[0]
        self.assertEqual(
            record.pipeline[SELECTED_TEXTURE_RESOLUTION_PIPELINE_KEY],
            1024,
        )
        variants = record.pipeline[TEXTURE_VARIANTS_PIPELINE_KEY]
        self.assertEqual(set(variants), {"512", "1024"})
        metadata = record.pipeline[SYMMETRIC_DIVISION_PIPELINE_KEY]
        self.assertEqual(
            metadata["version"],
            AUTOMATIC_SYMMETRIC_DIVISION_METADATA_VERSION,
        )
        self.assertEqual(metadata["orientation"], "vertical")
        self.assertEqual(metadata["kept_side"], "left")
        self.assertEqual(
            metadata["triangle_count_by_side"],
            {"left": 1, "right": 2},
        )
        self.assertEqual(
            metadata["packing_mode"],
            SYMMETRIC_TEXTURE_PACKING_MODE_PAIR,
        )
        self.assertEqual(
            metadata["texture_content_half"],
            SYMMETRIC_TEXTURE_CONTENT_HALF_LEFT,
        )

        output_path = self.asset_directory / variants["1024"][
            "glb_asset_path"
        ]
        output_mesh = _load_mesh(output_path.read_bytes())
        self.assertGreater(len(output_mesh.faces), 1)
        self.assertLessEqual(float(np.max(output_mesh.vertices[:, 0])), 1e-8)
        output_uv = np.asarray(output_mesh.visual.uv, dtype=float)
        self.assertTrue(np.isfinite(output_uv).all())
        self.assertGreaterEqual(float(np.min(output_uv[:, 0])), 0.0)
        self.assertLessEqual(float(np.max(output_uv[:, 0])), 0.5)
        self.assertGreaterEqual(float(np.min(output_uv[:, 1])), 0.0)
        self.assertLessEqual(float(np.max(output_uv[:, 1])), 1.0)

        texture_path = self.asset_directory / variants["1024"][
            "texture_asset_path"
        ]
        with Image.open(texture_path) as image:
            texture = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        self.assertEqual(texture.shape, (1024, 1024, 4))
        opaque_black = np.asarray((0, 0, 0, 255), dtype=np.uint8)
        self.assertTrue(np.all(texture[:, 512:] == opaque_black))
        self.assertTrue(np.any(texture[:, :512, :3] != 0))
        texture_512_path = self.asset_directory / variants["512"][
            "texture_asset_path"
        ]
        with Image.open(texture_512_path) as image:
            texture_512 = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        self.assertEqual(texture_512.shape, (512, 512, 4))
        self.assertTrue(np.all(texture_512[:, 256:] == opaque_black))
        self.assertIsNone(
            self.workspace.get_texture_variant(record.object_id, 2048)
        )


if __name__ == "__main__":
    unittest.main()
