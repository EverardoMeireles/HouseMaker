# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh
from PySide6.QtWidgets import QApplication

from housemaker.generation_state import (
    GeneratedObjectRecord,
    GenerationData,
)
from housemaker.generation_workspace import (
    TEXTURE_INPAINT_STROKES_PIPELINE_KEY,
    TextureRegenerationOutcome,
    TextureRegenerationRequest,
    UncheckedCameraFacePurgeOutcome,
    UncheckedCameraFacePurgeRequest,
    GenerationWorkspace,
    _build_regenerated_texture_pipeline,
    _build_unchecked_camera_face_purge_pipeline,
)
from housemaker.glb import import_generated_glb
from housemaker.meshy_generation import MeshyGenerationResult
from housemaker.object_texture_inpaint import (
    TEXTURE_UV_MODE_PAINT,
    TextureUvPoint,
    TextureUvStroke,
)
from housemaker.settings_widget import GenerationServiceSettings
from housemaker.unused_face_removal import UncheckedCameraFacePurgeResult


# ### Test application ###
_qt_application = QApplication.instance() or QApplication([])


# ### Fixture helpers ###
def _stroke() -> TextureUvStroke:
    return TextureUvStroke(
        mode=TEXTURE_UV_MODE_PAINT,
        radius_normalized=0.04,
        points=(
            TextureUvPoint(0.25, 0.75),
            TextureUvPoint(0.4, 0.6),
        ),
    )


def _box_glb() -> bytes:
    return bytes(trimesh.Scene(trimesh.creation.box()).export(file_type="glb"))


def _variant_metadata() -> dict[str, dict[str, str]]:
    return {
        str(resolution): {
            "glb_asset_path": f"replacement.texture-{resolution}.glb",
            "texture_asset_path": f"replacement.texture-{resolution}.png",
        }
        for resolution in (512, 1024, 2048)
    }


def _record(
    *,
    object_id: str = "chair",
    asset_path: str = "chair.glb",
) -> GeneratedObjectRecord:
    return GeneratedObjectRecord(
        object_id=object_id,
        frame_index=3,
        object_name="Chair",
        pipeline={
            TEXTURE_INPAINT_STROKES_PIPELINE_KEY: [_stroke().to_dict()],
            "retained_provenance": "keep",
        },
        provider="meshy",
        provider_task_id="task-chair",
        asset_path=asset_path,
    )


# ### Geometry and texture invalidation tests ###
class InpaintStrokePipelineInvalidationTests(unittest.TestCase):
    def test_manual_face_purge_discards_saved_uv_mask_strokes(self) -> None:
        record = _record()
        model = import_generated_glb(_box_glb())
        request = UncheckedCameraFacePurgeRequest(
            object_id=record.object_id,
            model_glb=model.glb_bytes,
            unchecked_camera_ids=("bottom",),
        )
        result = UncheckedCameraFacePurgeResult(
            model=model,
            unchecked_camera_ids=("bottom",),
            original_face_count=12,
            retained_face_count=10,
            removed_face_count=2,
        )

        pipeline = _build_unchecked_camera_face_purge_pipeline(
            record,
            UncheckedCameraFacePurgeOutcome(request=request, result=result),
            _variant_metadata(),
        )

        self.assertNotIn(TEXTURE_INPAINT_STROKES_PIPELINE_KEY, pipeline)
        self.assertEqual(pipeline["retained_provenance"], "keep")

    def test_full_retexture_discards_strokes_when_uvs_may_change(self) -> None:
        record = _record()
        request = TextureRegenerationRequest(
            object_id=record.object_id,
            reference_frame_index=4,
            reference_image_bgra=np.full(
                (4, 6, 4),
                (10, 20, 30, 255),
                dtype=np.uint8,
            ),
            model_glb=_box_glb(),
            settings=GenerationServiceSettings(meshy_api_key="msy-test-key"),
            enable_original_uv=False,
        )
        result = MeshyGenerationResult(
            task_id="new-texture-task",
            glb_bytes=_box_glb(),
            name="Chair",
        )

        pipeline = _build_regenerated_texture_pipeline(
            record,
            TextureRegenerationOutcome(request=request, result=result),
            _variant_metadata(),
        )

        self.assertNotIn(TEXTURE_INPAINT_STROKES_PIPELINE_KEY, pipeline)
        self.assertEqual(pipeline["retained_provenance"], "keep")


# ### Object deletion invalidation tests ###
class InpaintStrokeDeletionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.asset_directory = Path(self.temporary_directory.name) / "generated"
        self.asset_directory.mkdir()
        self.asset_directory.joinpath("chair.glb").write_bytes(_box_glb())
        self.workspace = GenerationWorkspace(
            asset_directory=self.asset_directory,
        )
        self.workspace.set_data(
            GenerationData(generated_objects=[_record()])
        )

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self.temporary_directory.cleanup()

    def test_deleting_object_removes_its_persisted_inpaint_mask_state(self) -> None:
        self.assertEqual(
            self.workspace.get_texture_inpaint_strokes("chair"),
            (_stroke(),),
        )

        self.assertTrue(self.workspace.delete_generated_object("chair"))

        self.assertEqual(self.workspace.get_texture_inpaint_strokes("chair"), ())
        self.assertEqual(self.workspace.get_data().generated_objects, [])


# ### Test runner ###
if __name__ == "__main__":
    unittest.main()
