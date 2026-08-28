# ### Environment setup ###
from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh
from PySide6.QtWidgets import QApplication

from housemaker.generation_state import GeneratedObjectRecord
from housemaker.generation_workspace import (
    ALL_CAMERA_IDS,
    MAX_OBJECT_OPERATION_UNDO_COUNT,
    OBJECT_OPERATION_GENERATE_TEXTURE,
    OBJECT_OPERATION_UNDO_STACK_PIPELINE_KEY,
    GenerationRequest,
    GenerationWorkspace,
    MeshyImagePlanner,
    MeshyModelExecutor,
    StagedMeshyGenerationResult,
    TextureRegenerationOutcome,
    TextureRegenerationRequest,
    _build_regenerated_texture_pipeline,
    _get_generated_object_asset_paths,
    _get_object_operation_undo_stack,
    _push_object_operation_undo_snapshot,
    _restore_object_operation_snapshot,
)
from housemaker.meshy_generation import MeshyGenerationResult
from housemaker.settings_widget import GenerationServiceSettings


# ### Test application ###
_qt_application = QApplication.instance() or QApplication([])


# ### Fixture helpers ###
def _box_glb() -> bytes:
    return bytes(trimesh.Scene(trimesh.creation.box()).export(file_type="glb"))


def _record(index: int) -> GeneratedObjectRecord:
    return GeneratedObjectRecord(
        object_id="object",
        frame_index=3,
        object_name="Chair",
        pipeline={"postprocessed_asset_path": f"geometry-{index}.glb"},
        provider_task_id=f"task-{index}",
        asset_path=f"model-{index}.glb",
    )


# ### Geometry-only planner tests ###
class GeometryOnlyPlannerTests(unittest.TestCase):
    def test_geometry_only_runs_unchecked_camera_purge_before_returning(
        self,
    ) -> None:
        geometry_glb = _box_glb()
        enabled_camera_ids = ALL_CAMERA_IDS[:-1]
        request = GenerationRequest(
            frame_index=2,
            selected_object_bgra=np.full((8, 8, 4), 255, dtype=np.uint8),
            settings=GenerationServiceSettings(meshy_api_key="test-key"),
            enabled_camera_ids=enabled_camera_ids,
            geometry_only=True,
        )

        with (
            patch(
                "housemaker.generation_workspace.request_image_to_3d_model",
                return_value=MeshyGenerationResult(
                    "geometry-task",
                    geometry_glb,
                    "Table",
                ),
            ),
            patch(
                "housemaker.generation_workspace."
                "purge_faces_visible_from_unchecked_cameras_from_glb",
                return_value=SimpleNamespace(
                    glb_bytes=b"purged geometry",
                    original_face_count=12,
                    retained_face_count=10,
                    removed_face_count=2,
                ),
            ) as purge,
            patch(
                "housemaker.generation_workspace.request_retextured_model"
            ) as generate_texture,
        ):
            result = MeshyImagePlanner().plan(request)

        self.assertIsInstance(result, StagedMeshyGenerationResult)
        assert isinstance(result, StagedMeshyGenerationResult)
        self.assertTrue(result.geometry_only)
        self.assertTrue(result.camera_face_purge_applied)
        self.assertEqual(result.glb_bytes, b"purged geometry")
        self.assertEqual(result.unchecked_camera_ids, (ALL_CAMERA_IDS[-1],))
        self.assertEqual(
            purge.call_args.kwargs["unchecked_camera_ids"],
            (ALL_CAMERA_IDS[-1],),
        )
        generate_texture.assert_not_called()

    def test_geometry_only_skips_retexture_and_returns_processed_revision(
        self,
    ) -> None:
        geometry_glb = _box_glb()
        request = GenerationRequest(
            frame_index=2,
            selected_object_bgra=np.full((8, 8, 4), 255, dtype=np.uint8),
            settings=GenerationServiceSettings(meshy_api_key="test-key"),
            enabled_camera_ids=ALL_CAMERA_IDS,
            geometry_only=True,
        )

        with (
            patch(
                "housemaker.generation_workspace.request_image_to_3d_model",
                return_value=MeshyGenerationResult(
                    "geometry-task",
                    geometry_glb,
                    "Table",
                ),
            ) as generate_geometry,
            patch(
                "housemaker.generation_workspace.request_retextured_model"
            ) as generate_texture,
        ):
            result = MeshyImagePlanner().plan(request)

        self.assertIsInstance(result, StagedMeshyGenerationResult)
        assert isinstance(result, StagedMeshyGenerationResult)
        self.assertTrue(result.geometry_only)
        self.assertEqual(result.task_id, "geometry-task")
        self.assertEqual(result.glb_bytes, geometry_glb)
        self.assertEqual(result.source_glb_bytes, geometry_glb)
        self.assertEqual(result.postprocessed_glb_bytes, geometry_glb)
        self.assertFalse(generate_geometry.call_args.kwargs["should_texture"])
        generate_texture.assert_not_called()

        model = MeshyModelExecutor().execute(result)
        self.assertIsNone(model.object_texture_variants)
        self.assertEqual(model.glb_bytes, geometry_glb)

    def test_geometry_only_success_creates_selected_untextured_object(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = GenerationWorkspace(
                asset_directory=Path(temporary_directory),
            )
            try:
                geometry_glb = _box_glb()
                result = StagedMeshyGenerationResult(
                    task_id="geometry-task",
                    glb_bytes=geometry_glb,
                    name="Table",
                    geometry_task_id="geometry-task",
                    source_glb_bytes=geometry_glb,
                    postprocessed_glb_bytes=geometry_glb,
                    enabled_camera_ids=ALL_CAMERA_IDS,
                    geometry_only=True,
                )

                workspace._handle_generation_succeeded(
                    result,
                    MeshyModelExecutor().execute(result),
                )

                record = workspace.get_data().generated_objects[0]
                self.assertEqual(workspace._selected_object_id, record.object_id)
                self.assertTrue(record.pipeline["geometry_only"])
                self.assertNotIn("texture_variants", record.pipeline)
                self.assertTrue(
                    Path(temporary_directory, str(record.asset_path)).is_file()
                )
                self.assertEqual(
                    workspace.result_view.model.glb_bytes,
                    geometry_glb,
                )
                self.assertIn("Generated geometry", workspace.status_label.text())
            finally:
                workspace.shutdown()
                workspace.close()
                _qt_application.processEvents()


# ### Persistent undo tests ###
class PersistentObjectUndoTests(unittest.TestCase):
    def test_stack_is_bounded_flat_and_restores_latest_record(self) -> None:
        record = _record(0)
        total_operations = MAX_OBJECT_OPERATION_UNDO_COUNT + 3
        for index in range(1, total_operations + 1):
            next_record = _record(index)
            pipeline = _push_object_operation_undo_snapshot(
                record,
                next_record.pipeline,
                operation=OBJECT_OPERATION_GENERATE_TEXTURE,
            )
            record = replace(
                next_record,
                pipeline=pipeline,
            )

        stack = _get_object_operation_undo_stack(record)
        self.assertEqual(len(stack), MAX_OBJECT_OPERATION_UNDO_COUNT)
        self.assertTrue(
            all(
                OBJECT_OPERATION_UNDO_STACK_PIPELINE_KEY
                not in snapshot["pipeline"]
                for snapshot in stack
            )
        )
        retained_paths = set(_get_generated_object_asset_paths(record))
        self.assertNotIn("model-0.glb", retained_paths)
        self.assertIn("model-3.glb", retained_paths)
        self.assertIn(f"model-{total_operations}.glb", retained_paths)

        restored = _restore_object_operation_snapshot(
            record,
            stack[-1],
            stack[:-1],
        )
        self.assertEqual(
            restored.provider_task_id,
            f"task-{total_operations - 1}",
        )
        self.assertEqual(
            restored.asset_path,
            f"model-{total_operations - 1}.glb",
        )
        self.assertEqual(
            len(_get_object_operation_undo_stack(restored)),
            MAX_OBJECT_OPERATION_UNDO_COUNT - 1,
        )

    def test_texture_success_normalizes_geometry_mode_and_undo_restores_it(
        self,
    ) -> None:
        geometry_glb = _box_glb()
        original = GeneratedObjectRecord(
            object_id="object",
            frame_index=3,
            object_name="Chair",
            pipeline={
                "mode": "processed_geometry_only",
                "geometry_only": True,
                "postprocessed_asset_path": "geometry.glb",
            },
            provider_task_id="geometry-task",
            asset_path="geometry.glb",
        )
        request = TextureRegenerationRequest(
            object_id=original.object_id,
            reference_frame_index=3,
            reference_image_bgra=np.full((8, 8, 4), 255, dtype=np.uint8),
            model_glb=geometry_glb,
            settings=GenerationServiceSettings(meshy_api_key="test-key"),
        )
        outcome = TextureRegenerationOutcome(
            request=request,
            result=MeshyGenerationResult(
                "texture-task",
                geometry_glb,
                original.object_name,
            ),
        )
        variant_metadata = {
            str(resolution): {
                "glb_asset_path": f"textured-{resolution}.glb",
                "texture_asset_path": f"textured-{resolution}.png",
            }
            for resolution in (512, 1024, 2048)
        }

        next_pipeline = _build_regenerated_texture_pipeline(
            original,
            outcome,
            variant_metadata,
        )
        textured = replace(
            original,
            pipeline=_push_object_operation_undo_snapshot(
                original,
                next_pipeline,
                operation=OBJECT_OPERATION_GENERATE_TEXTURE,
            ),
            provider_task_id="texture-task",
            asset_path="textured-1024.glb",
        )

        self.assertFalse(textured.pipeline["geometry_only"])
        self.assertEqual(textured.pipeline["mode"], "processed")
        undo_stack = _get_object_operation_undo_stack(textured)
        restored = _restore_object_operation_snapshot(
            textured,
            undo_stack[-1],
            undo_stack[:-1],
        )
        self.assertEqual(restored, original)


if __name__ == "__main__":
    unittest.main()
