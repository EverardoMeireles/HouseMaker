"""Exact blob provenance and asynchronous Meshy result selection, without AI calls."""

# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ### Imports ###
import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import trimesh
from PySide6.QtWidgets import QApplication

from housemaker.generation_state import (
    MASK_MODE_PAINT,
    GeneratedObjectRecord,
    GenerationData,
    MaskPoint,
    MaskStroke,
)
from housemaker.generation_workspace import (
    INPAINT_SOURCE_PIPELINE_KEY,
    GenerationRequest,
    GenerationWorkspace,
    StagedMeshyGenerationResult,
    _apply_inpaint_source_pipeline_metadata,
    _build_generation_asset_revision,
    _SavedObjectGeneration,
)
from housemaker.glb import GeneratedModel
from housemaker.meshy_generation import MeshyGenerationResult
from housemaker.settings_widget import GenerationServiceSettings
from housemaker.video_source import VideoMetadata

# ### Model-free fixtures ###
_application = QApplication.instance() or QApplication([])
_application.setQuitOnLastWindowClosed(False)


def _image(value: int = 150) -> np.ndarray:
    image = np.full((20, 24, 4), value, dtype=np.uint8)
    image[:, :, 3] = 255
    return image


def _request() -> GenerationRequest:
    return GenerationRequest(
        frame_index=2,
        selected_object_bgra=_image(),
        settings=GenerationServiceSettings(),
    )


def _source_metadata(request: GenerationRequest) -> dict[str, object]:
    return {
        "frame_index": request.inpaint_key[0],
        "mask_sha256": request.inpaint_key[1],
        "video_path": (
            None if request.inpaint_video_path is None
            else os.path.normcase(os.path.abspath(request.inpaint_video_path))
        ),
    }


def _model() -> GeneratedModel:
    mesh = trimesh.creation.box(extents=(1.0, 0.5, 0.75))
    scene = trimesh.Scene(mesh)
    return GeneratedModel(mesh=mesh, scene=scene, glb_bytes=scene.export(file_type="glb"))


class _VideoSource:
    def __init__(self, path: Path) -> None:
        self.metadata = VideoMetadata(str(path), 6, 30.0, 200, 100)

    def get_frame(self, index: int) -> np.ndarray:
        return np.full((100, 200, 3), 30 + index * 20, dtype=np.uint8)

    def close(self) -> None:
        pass


# ### Serializable source metadata tests ###
class InpaintSourceMetadataTests(unittest.TestCase):
    def test_ordinary_requests_have_no_blob_identity(self) -> None:
        request = _request()
        self.assertIsNone(request.inpaint_key)
        self.assertIsNone(request.inpaint_video_path)

    def test_none_or_standalone_request_does_not_modify_pipeline(self) -> None:
        for request in (None, _request()):
            with self.subTest(request=request):
                pipeline = {"existing": {"value": [1, 2]}}
                previous = copy.deepcopy(pipeline)
                _apply_inpaint_source_pipeline_metadata(pipeline, request)
                self.assertEqual(pipeline, previous)

    def test_exact_key_and_normalized_video_path_are_persisted(self) -> None:
        request = _request()
        request.inpaint_key = (2, "ab" * 32)
        request.inpaint_video_path = os.path.join("relative", "source.mp4")
        pipeline = {"existing": True}

        _apply_inpaint_source_pipeline_metadata(pipeline, request)

        self.assertTrue(pipeline["existing"])
        self.assertEqual(pipeline[INPAINT_SOURCE_PIPELINE_KEY], _source_metadata(request))
        self.assertEqual(request.inpaint_key, (2, "ab" * 32))
        self.assertEqual(request.inpaint_video_path, os.path.join("relative", "source.mp4"))

    def test_blob_identity_without_video_still_has_explicit_null_path(self) -> None:
        request = _request()
        request.inpaint_key = (2, "cd" * 32)
        pipeline = {}

        _apply_inpaint_source_pipeline_metadata(pipeline, request)

        self.assertEqual(pipeline[INPAINT_SOURCE_PIPELINE_KEY], _source_metadata(request))
        self.assertIsNone(pipeline[INPAINT_SOURCE_PIPELINE_KEY]["video_path"])

    def test_source_metadata_round_trips_without_project_schema_changes(self) -> None:
        request = _request()
        request.inpaint_key = (2, "ef" * 32)
        request.inpaint_video_path = os.path.join("reference", "source.mp4")
        pipeline = {}
        _apply_inpaint_source_pipeline_metadata(pipeline, request)
        record = GeneratedObjectRecord(
            object_id="test-object", frame_index=2, object_name="object_1",
            pipeline=pipeline, provider_task_id="test-task", asset_path="object.glb",
        )

        restored = GeneratedObjectRecord.from_dict(json.loads(json.dumps(record.to_dict())))

        self.assertEqual(restored.pipeline[INPAINT_SOURCE_PIPELINE_KEY], _source_metadata(request))
        restored.pipeline[INPAINT_SOURCE_PIPELINE_KEY]["mask_sha256"] = "changed"
        self.assertEqual(record.pipeline[INPAINT_SOURCE_PIPELINE_KEY]["mask_sha256"], "ef" * 32)


# ### Frozen per-blob request tests ###
class InpaintGenerationProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.workspace = GenerationWorkspace(asset_directory=self.root)
        self.source = _VideoSource(self.root / "source.mp4")
        self.workspace._video_source = self.source
        self.workspace._data = GenerationData(video_metadata=self.source.metadata)
        self.workspace.set_runtime_settings(GenerationServiceSettings(meshy_api_key="test"))
        self.workspace.show_frame(0)

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _application.processEvents()
        self.temporary.cleanup()

    def _paint(self, *positions: float) -> None:
        self.workspace.video_view.set_strokes([
            MaskStroke(MASK_MODE_PAINT, 0.035, (MaskPoint(x, 0.5),))
            for x in positions
        ])
        self.workspace._handle_video_strokes_changed(self.workspace.video_view.get_strokes())

    def _three_objects(self) -> None:
        self._paint(0.2, 0.8)
        self.workspace._data.set_frame_strokes(3, [
            MaskStroke(MASK_MODE_PAINT, 0.035, (MaskPoint(0.5, 0.5),)),
        ])

    def _complete(
        self, request: GenerationRequest, *, saved: bool = False,
    ) -> GeneratedObjectRecord:
        model = _model()
        task_id = f"task-{len(self.workspace._data.generated_objects) + 1}"
        if request.geometry_only:
            result = StagedMeshyGenerationResult(
                task_id=task_id, glb_bytes=model.glb_bytes,
                geometry_task_id=task_id, source_glb_bytes=model.glb_bytes,
                postprocessed_glb_bytes=model.glb_bytes, geometry_only=True,
            )
        else:
            result = MeshyGenerationResult(task_id=task_id, glb_bytes=model.glb_bytes)
        self.workspace._active_generation_request = request
        if saved:
            asset_path = f"{task_id}.glb"
            (self.root / asset_path).write_bytes(model.glb_bytes)
            result = _SavedObjectGeneration(
                result=result, object_id=task_id, pipeline={}, asset_path=asset_path,
                preview_model=model,
                preview_asset_revision=_build_generation_asset_revision(self.root, asset_path),
                symmetry=None, persisted_asset_paths=(asset_path,),
            )
        self.workspace._handle_generation_succeeded(result, model)
        self.assertIsNone(self.workspace._active_generation_request)
        record = self.workspace._data.generated_objects[-1]
        self.assertEqual(record.provider_task_id, task_id, self.workspace.status_label.text())
        return record

    def test_all_requests_capture_exact_keys_even_when_seekbar_has_no_blob(self) -> None:
        self._three_objects()
        expected = self.workspace._current_inpaint_keys()
        self.workspace.show_frame(2)

        requests = self.workspace._build_generation_requests()

        self.assertEqual(tuple(request.inpaint_key for request in requests), expected)
        self.assertEqual([request.frame_index for request in requests], [0, 0, 3])
        self.assertNotEqual(requests[0].inpaint_key, requests[1].inpaint_key)
        expected_path = os.path.normcase(os.path.abspath(self.source.metadata.path))
        self.assertTrue(all(request.inpaint_video_path == expected_path for request in requests))
        self.assertEqual(self.workspace._data.current_frame_index, 2)

    def test_snapshot_identity_survives_later_seek_and_mask_changes(self) -> None:
        self._three_objects()
        requests = self.workspace._build_generation_requests()
        expected = [(request.inpaint_key, request.inpaint_video_path) for request in requests]

        self.workspace.show_frame(3)
        self._paint(0.65)
        self.workspace.show_frame(0)
        self._paint(0.4)

        self.assertEqual(
            [(request.inpaint_key, request.inpaint_video_path) for request in requests], expected,
        )
        self.assertNotEqual(self.workspace._current_inpaint_keys(), tuple(item[0] for item in expected))

    def test_geometry_only_requests_capture_same_blob_keys(self) -> None:
        self._three_objects()
        expected = self.workspace._current_inpaint_keys()

        requests = self.workspace._build_generation_requests(geometry_only=True)

        self.assertEqual(tuple(request.inpaint_key for request in requests), expected)
        self.assertTrue(all(request.geometry_only for request in requests))

    def test_generated_reference_does_not_replace_source_mask_identity(self) -> None:
        self._paint(0.3)
        key = self.workspace.current_inpaint_reference_key()
        self.workspace.set_temporary_object_reference(_image(240))

        request = self.workspace._build_generation_requests()[0]

        self.assertEqual(request.inpaint_key, key)
        np.testing.assert_array_equal(request.selected_object_bgra, _image(240))

    def test_standalone_loaded_reference_does_not_gain_a_blob_identity(self) -> None:
        self.workspace.set_temporary_object_reference(_image())

        request = self.workspace._build_generation_requests()[0]

        self.assertIsNone(request.inpaint_key)

    # ### Real local GLB commit coverage ###
    def test_ordinary_completion_persists_and_selects_matching_blob(self) -> None:
        self._three_objects()
        request = self.workspace._build_generation_requests()[0]

        record = self._complete(request)

        self.assertEqual(record.pipeline[INPAINT_SOURCE_PIPELINE_KEY], _source_metadata(request))
        self.assertEqual(self.workspace._selected_object_id, record.object_id)
        self.assertIsNotNone(self.workspace.result_view.model)

    def test_saved_completion_persists_and_selects_matching_blob(self) -> None:
        self._three_objects()
        request = self.workspace._build_generation_requests()[0]

        record = self._complete(request, saved=True)

        self.assertEqual(record.pipeline[INPAINT_SOURCE_PIPELINE_KEY], _source_metadata(request))
        self.assertEqual(self.workspace._selected_object_id, record.object_id)
        self.assertIsNotNone(self.workspace.result_view.model)

    def test_geometry_only_completion_paths_preserve_blob_provenance(self) -> None:
        self._three_objects()
        request = self.workspace._build_generation_requests(geometry_only=True)[0]
        for saved in (False, True):
            with self.subTest(saved=saved):
                record = self._complete(request, saved=saved)
                self.assertEqual(record.pipeline[INPAINT_SOURCE_PIPELINE_KEY], _source_metadata(request))
                self.assertEqual(self.workspace._selected_object_id, record.object_id)

    def test_out_of_order_completions_do_not_select_other_blob_models(self) -> None:
        self._three_objects()
        requests = self.workspace._build_generation_requests()
        matching_record = self._complete(requests[0])
        matching_model = self.workspace.result_view.model

        for request, saved in ((requests[1], False), (requests[2], True)):
            with self.subTest(frame=request.frame_index, saved=saved):
                other_record = self._complete(request, saved=saved)
                self.assertNotEqual(other_record.object_id, matching_record.object_id)
                self.assertEqual(self.workspace._selected_object_id, matching_record.object_id)
                self.assertIs(self.workspace.result_view.model, matching_model)

    def test_other_blob_completion_does_not_populate_an_empty_preview(self) -> None:
        self._three_objects()
        request = self.workspace._build_generation_requests()[1]
        for saved in (False, True):
            with self.subTest(saved=saved):
                self._complete(request, saved=saved)
                self.assertIsNone(self.workspace._selected_object_id)
                self.assertIsNone(self.workspace.result_view.model)

    def test_edited_mask_completion_does_not_select_the_obsolete_blob(self) -> None:
        self._paint(0.3)
        request = self.workspace._build_generation_requests()[0]
        self._paint(0.4)

        record = self._complete(request)

        self.assertNotEqual(request.inpaint_key, self.workspace.current_inpaint_reference_key())
        self.assertEqual(record.pipeline[INPAINT_SOURCE_PIPELINE_KEY], _source_metadata(request))
        self.assertIsNone(self.workspace._selected_object_id)
        self.assertIsNone(self.workspace.result_view.model)

    def test_same_mask_in_different_video_does_not_select_old_video_result(self) -> None:
        self._paint(0.3)
        request = self.workspace._build_generation_requests()[0]
        self.workspace._data.video_metadata = replace(
            self.source.metadata, path=str(self.root / "different-source.mp4"),
        )

        record = self._complete(request, saved=True)

        self.assertEqual(request.inpaint_key, self.workspace.current_inpaint_reference_key())
        self.assertEqual(record.pipeline[INPAINT_SOURCE_PIPELINE_KEY], _source_metadata(request))
        self.assertIsNone(self.workspace._selected_object_id)
        self.assertIsNone(self.workspace.result_view.model)

    def test_ordinary_unlinked_completion_keeps_existing_autoselection(self) -> None:
        self.workspace.set_temporary_object_reference(_image())
        request = self.workspace._build_generation_requests()[0]

        record = self._complete(request)

        self.assertNotIn(INPAINT_SOURCE_PIPELINE_KEY, record.pipeline)
        self.assertEqual(self.workspace._selected_object_id, record.object_id)


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
