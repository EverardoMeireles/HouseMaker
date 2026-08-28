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
    SYMMETRIC_DIVISION_PIPELINE_KEY,
    TEXTURE_VARIANTS_PIPELINE_KEY,
    GenerationRequest,
    GenerationWorkspace,
)
from housemaker.glb import import_generated_glb
from housemaker.meshy_generation import MeshyGenerationResult
from housemaker.object_symmetry import (
    SYMMETRIC_DIVISION_ORIENTATION_VERTICAL,
    SYMMETRIC_TEXTURE_PACKING_MODE_PAIR,
)
from housemaker.object_texture_variants import build_object_texture_variants
from housemaker.settings_widget import GenerationServiceSettings


# ### Test application ###
_qt_application = QApplication.instance() or QApplication([])


# ### Fixture helpers ###
def _long_taper_provider_glb() -> bytes:
    """Return a Meshy-like atlas whose retained UV triangle has a thin apex."""

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
    uvs = np.asarray(
        (
            (0.848756015, 0.171862006),
            (0.850714028, 0.000185012817),
            (0.850714028, 0.173541009),
            (0.1, 0.1),
            (0.9, 0.1),
            (0.9, 0.9),
            (0.1, 0.9),
        ),
        dtype=float,
    )
    texture = np.full((2048, 2048, 4), (37, 83, 149, 255), dtype=np.uint8)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual = TextureVisuals(
        uv=uvs,
        material=PBRMaterial(
            name="meshy-provider-material",
            baseColorTexture=Image.fromarray(texture, mode="RGBA"),
        ),
    )
    return bytes(trimesh.Scene(mesh).export(file_type="glb"))


def _load_only_mesh(payload: bytes) -> trimesh.Trimesh:
    scene = trimesh.load(
        io.BytesIO(payload),
        file_type="glb",
        force="scene",
        process=False,
    )
    if not isinstance(scene, trimesh.Scene):
        raise AssertionError("The tapered-UV fixture did not load as a scene.")
    mesh = scene.to_geometry()
    if not isinstance(mesh, trimesh.Trimesh):
        raise AssertionError("The tapered-UV fixture did not contain a mesh.")
    return mesh


# ### Generation regression ###
class GenerationTaperedUvSymmetryRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.asset_directory = Path(self.temporary_directory.name) / "generated"
        self.workspace = GenerationWorkspace(asset_directory=self.asset_directory)

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        self.temporary_directory.cleanup()

    def test_full_generation_keeps_long_taper_apex_inside_pair_atlas(self) -> None:
        source_glb = _long_taper_provider_glb()
        ordinary_variants = build_object_texture_variants(source_glb)
        self.assertIsNotNone(ordinary_variants)
        assert ordinary_variants is not None
        self.assertEqual(set(ordinary_variants.glb_by_resolution), {512, 1024, 2048})

        canonical_mesh = _load_only_mesh(
            ordinary_variants.glb_by_resolution[2048]
        )
        canonical_uvs = np.asarray(canonical_mesh.visual.uv, dtype=float)
        self.assertLess(float(np.min(canonical_uvs[:, 1])), 0.001)

        generated_model = import_generated_glb(
            ordinary_variants.glb_by_resolution[1024]
        )
        generated_model.object_texture_variants = ordinary_variants
        self.workspace._active_generation_request = GenerationRequest(
            frame_index=7,
            selected_object_bgra=np.full((4, 4, 4), 255, dtype=np.uint8),
            settings=GenerationServiceSettings(meshy_api_key="key"),
            symmetric_division_enabled=True,
            symmetric_division_orientation=(
                SYMMETRIC_DIVISION_ORIENTATION_VERTICAL
            ),
        )
        generated_spy = QSignalSpy(self.workspace.generation_completed)

        self.workspace._handle_generation_succeeded(
            MeshyGenerationResult("tapered", source_glb, "Tapered UV"),
            generated_model,
        )

        self.assertEqual(generated_spy.count(), 1)
        records = self.workspace.get_data().generated_objects
        self.assertEqual(len(records), 1)
        record = records[0]
        metadata = record.pipeline[SYMMETRIC_DIVISION_PIPELINE_KEY]
        self.assertEqual(metadata["kept_side"], "left")
        self.assertEqual(metadata["packing_mode"], SYMMETRIC_TEXTURE_PACKING_MODE_PAIR)
        variants = record.pipeline[TEXTURE_VARIANTS_PIPELINE_KEY]
        self.assertEqual(set(variants), {"512", "1024"})

        output_glb_path = self.asset_directory / variants["1024"][
            "glb_asset_path"
        ]
        output_mesh = _load_only_mesh(output_glb_path.read_bytes())
        output_uvs = np.asarray(output_mesh.visual.uv, dtype=float)
        self.assertTrue(np.isfinite(output_uvs).all())
        self.assertGreaterEqual(float(np.min(output_uvs[:, 0])), 0.0)
        self.assertLessEqual(float(np.max(output_uvs[:, 0])), 0.5)
        self.assertGreaterEqual(float(np.min(output_uvs[:, 1])), 0.0)
        self.assertLessEqual(float(np.max(output_uvs[:, 1])), 1.0)
        self.assertNotIn(
            "Symmetric division requires UVs inside the atlas",
            self.workspace.status_label.text(),
        )


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
