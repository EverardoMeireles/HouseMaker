"""Match manually queued inpaints to their generated 3D previews without inference."""

# ### Environment and imports ###
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import trimesh
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

from housemaker.generation_jobs import GenerationJobManager
from housemaker.generation_state import (
    MASK_MODE_PAINT,
    GeneratedObjectRecord,
    GenerationData,
    MaskPoint,
    MaskStroke,
)
from housemaker.generation_workspace import (
    INPAINT_SOURCE_PIPELINE_KEY,
    GenerationWorkspace,
)
from housemaker.glb import import_generated_glb
from housemaker.merged_generation_workspace import MergedGenerationWorkspace
from housemaker.meshy_generation import MeshyGenerationResult
from housemaker.settings_widget import GenerationServiceSettings
from housemaker.surface_texture_workspace import SurfaceTextureGenerationWorkspace

# ### Local fixtures ###
APP = QApplication.instance() or QApplication([])
APP.setQuitOnLastWindowClosed(False)


def _reference(value: int) -> np.ndarray:
    image = np.full((20, 24, 4), value, dtype=np.uint8)
    image[:, :, 3] = 255
    return image


def _write_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48),
    )
    if not writer.isOpened():
        raise unittest.SkipTest("The MJPG video writer is unavailable")
    try:
        for value in (40, 80, 120):
            writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    finally:
        writer.release()


# ### Reference and generated-object navigation tests ###
class InpaintGeneratedModelNavigationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manager = GenerationJobManager()
        self.objects = GenerationWorkspace(
            asset_directory=self.root / "objects", job_manager=self.manager,
        )
        self.objects.set_runtime_settings(
            GenerationServiceSettings(meshy_api_key="unused-test-key"),
        )
        self.surfaces = SurfaceTextureGenerationWorkspace(
            asset_directory=self.root / "surfaces", job_manager=self.manager,
            shared_controls=self.objects.get_shared_controls(),
        )
        self.workspace = MergedGenerationWorkspace(
            self.surfaces, self.objects, self.manager,
        )
        self.video = self.root / "references.avi"
        _write_video(self.video)
        self.objects.load_video(str(self.video))
        self.surfaces.load_video(str(self.video))
        self._paint(0, 0.25, 0.75)
        self._paint(2, 0.5)
        self.workspace.seekbar.setValue(0)
        self.workspace.video_view.select_blob(0)
        self.workspace.sync_shared_controls()
        self.keys = tuple(
            key for key, _image in self.objects.snapshot_inpaint_reference_inputs()
        )
        self.assertEqual(len(self.keys), 3)
        for key, value in zip(self.keys, (101, 102, 103)):
            self.assertTrue(self.objects.set_inpaint_reference(key, _reference(value)))
        self.objects._handle_inpaint_selection_changed()
        self.assets: dict[str, bytes] = {}

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.objects.shutdown()
        self.surfaces.shutdown()
        self.workspace.close()
        self.workspace.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        APP.processEvents()
        self.temporary.cleanup()

    def _paint(self, frame: int, *positions: float) -> None:
        self.workspace.seekbar.setValue(frame)
        self.workspace.video_view.set_strokes([
            MaskStroke(MASK_MODE_PAINT, 0.07, (MaskPoint(x, 0.5),))
            for x in positions
        ])
        self.workspace.video_view.strokes_changed.emit(
            self.workspace.video_view.get_strokes(),
        )

    def _record(
        self, object_id: str, key: tuple[int, str] | None,
        *, video_path: str | None = None,
    ) -> GeneratedObjectRecord:
        mesh = trimesh.creation.box(extents=(1 + len(self.assets) * 0.1, 0.5, 0.75))
        glb = bytes(trimesh.Scene(mesh).export(file_type="glb"))
        self.objects._asset_directory.mkdir(parents=True, exist_ok=True)
        (self.objects._asset_directory / f"{object_id}.glb").write_bytes(glb)
        self.assets[object_id] = glb
        pipeline = {} if key is None else {
            INPAINT_SOURCE_PIPELINE_KEY: {
                "frame_index": key[0], "mask_sha256": key[1],
                "video_path": os.path.normcase(os.path.abspath(
                    str(self.video) if video_path is None else video_path,
                )),
            },
        }
        record = GeneratedObjectRecord(
            object_id=object_id, frame_index=0 if key is None else key[0],
            object_name=object_id, pipeline=pipeline, provider_task_id=object_id,
            asset_path=f"{object_id}.glb",
        )
        self.objects._data.generated_objects.append(record)
        return record

    def _seed_generated_models(self) -> None:
        for key, object_id in zip(self.keys, ("first", "second", "third")):
            self._record(object_id, key)

    def _assert_selected(self, object_id: str | None, reference_value: int) -> None:
        self.assertEqual(self.objects._selected_object_id, object_id)
        if object_id is None:
            self.assertIsNone(self.objects.result_view.model)
        else:
            self.assertIsNotNone(self.objects.result_view.model)
            self.assertEqual(
                self.objects.result_view.model.glb_bytes, self.assets[object_id],
            )
        np.testing.assert_array_equal(
            self.objects.get_temporary_object_reference(), _reference(reference_value),
        )
        np.testing.assert_array_equal(
            self.surfaces.get_temporary_reference(), _reference(reference_value),
        )
        self.assertTrue(self.workspace.video_view.has_reference_preview())

    def test_previous_and_next_show_matching_models_and_references_across_frames(self) -> None:
        self._seed_generated_models()
        for button, object_id, value, frame in (
            (self.workspace.next_inpaint_button, "second", 102, 0),
            (self.workspace.next_inpaint_button, "third", 103, 2),
            (self.workspace.next_inpaint_button, "first", 101, 0),
            (self.workspace.previous_inpaint_button, "third", 103, 2),
            (self.workspace.previous_inpaint_button, "second", 102, 0),
        ):
            self.assertTrue(button.isEnabled())
            button.click()
            self._assert_selected(object_id, value)
            self.assertEqual(self.objects._displayed_frame_index, frame)
            self.assertEqual(self.surfaces._displayed_frame_index, frame)

    def test_unfinished_blob_clears_a_different_models_preview(self) -> None:
        self._record("first", self.keys[0])
        self.objects.select_generated_object("first")
        self.workspace.next_inpaint_button.click()
        self._assert_selected(None, 102)
        self.workspace.previous_inpaint_button.click()
        self._assert_selected("first", 101)

    def test_latest_matching_generation_wins_without_using_same_frame_as_identity(self) -> None:
        self._record("second-old", self.keys[1])
        self._record("second-new", self.keys[1])
        self._record("unrelated-last", None)
        self._record("other-blob-last", self.keys[0])
        self.workspace.next_inpaint_button.click()
        self._assert_selected("second-new", 102)

    def test_one_blob_navigation_restores_model_after_an_unrelated_atlas_selection(self) -> None:
        self._paint(0, 0.25)
        self.objects._data.set_frame_strokes(2, [])
        key = self.objects.current_inpaint_reference_key()
        self.assertIsNotNone(key)
        self.objects.set_inpaint_reference(key, _reference(101))
        self._record("matching", key)
        self._record("atlas-selection", None)
        self.assertTrue(self.objects.select_generated_object("atlas-selection"))
        self.workspace.sync_shared_controls()
        self.assertTrue(self.workspace.next_inpaint_button.isEnabled())
        self.workspace.next_inpaint_button.click()
        self._assert_selected("matching", 101)
        self.objects.select_generated_object("atlas-selection")
        self.workspace.previous_inpaint_button.click()
        self._assert_selected("matching", 101)

    def test_navigating_to_the_same_model_does_not_reload_its_glb(self) -> None:
        self._paint(0, 0.25)
        self.objects._data.set_frame_strokes(2, [])
        key = self.objects.current_inpaint_reference_key()
        self.objects.set_inpaint_reference(key, _reference(101))
        self._record("matching", key)
        self.objects.select_generated_object("matching")
        original_model = self.objects.result_view.model
        with patch.object(
            self.objects, "_load_generated_object_model",
            wraps=self.objects._load_generated_object_model,
        ) as load:
            self.workspace.next_inpaint_button.click()
            self.workspace.previous_inpaint_button.click()
        load.assert_not_called()
        self.assertIs(self.objects.result_view.model, original_model)

    def test_changed_mask_does_not_select_a_model_from_its_previous_shape(self) -> None:
        self._record("original-second", self.keys[1])
        self._paint(0, 0.25, 0.6)
        self.workspace.video_view.select_blob(0)
        self.workspace.next_inpaint_button.click()
        self.assertNotEqual(self.objects.current_inpaint_reference_key(), self.keys[1])
        self.assertIsNone(self.objects._selected_object_id)
        self.assertIsNone(self.objects.result_view.model)

    def test_same_mask_in_a_different_video_does_not_match(self) -> None:
        self._record("another-video-second", self.keys[1], video_path=str(self.root / "other.avi"))
        self.objects.select_generated_object("another-video-second")
        self.workspace.next_inpaint_button.click()
        self._assert_selected(None, 102)

    def test_existing_record_without_blob_metadata_does_not_match_by_frame(self) -> None:
        self._record("legacy-frame-only", None)
        self.objects.select_generated_object("legacy-frame-only")
        self.workspace.next_inpaint_button.click()
        self._assert_selected(None, 102)

    def test_json_roundtrip_retains_source_association_for_model_navigation(self) -> None:
        self._seed_generated_models()
        saved = json.loads(json.dumps(self.objects.get_data().to_dict()))
        restored = GenerationData.from_dict(saved)
        self.assertEqual(
            restored.generated_objects[1].pipeline[INPAINT_SOURCE_PIPELINE_KEY],
            saved["generated_objects"][1]["pipeline"][INPAINT_SOURCE_PIPELINE_KEY],
        )
        self.objects.set_data(restored)
        for key, value in zip(self.keys, (101, 102, 103)):
            self.objects.set_inpaint_reference(key, _reference(value))
        self.workspace.sync_shared_controls()
        self.workspace.next_inpaint_button.click()
        self._assert_selected("second", 102)

    def test_full_and_geometry_requests_capture_exact_blob_and_source_video(self) -> None:
        for geometry_only in (False, True):
            with self.subTest(geometry_only=geometry_only):
                requests = self.objects._build_generation_requests(geometry_only=geometry_only)
                self.assertEqual([request.inpaint_key for request in requests], list(self.keys))
                self.assertEqual(
                    [request.inpaint_video_path for request in requests],
                    [os.path.normcase(os.path.abspath(self.video))] * 3,
                )
                self.assertTrue(all(request.geometry_only is geometry_only for request in requests))

    def test_completed_meshy_generation_records_original_blob_even_after_navigation(self) -> None:
        request = self.objects._build_generation_requests()[1]
        self.workspace.next_inpaint_button.click()
        self.workspace.next_inpaint_button.click()
        mesh = trimesh.creation.box(extents=(1, 0.5, 0.75))
        glb = bytes(trimesh.Scene(mesh).export(file_type="glb"))
        self.objects._active_generation_request = request
        self.objects._handle_generation_succeeded(
            MeshyGenerationResult(task_id="finished-request", glb_bytes=glb, name="Generated"),
            import_generated_glb(glb),
        )
        record = self.objects.get_data().generated_objects[-1]
        self.assertEqual(record.pipeline[INPAINT_SOURCE_PIPELINE_KEY], {
            "frame_index": self.keys[1][0],
            "mask_sha256": self.keys[1][1],
            "video_path": os.path.normcase(os.path.abspath(self.video)),
        })
        self.assets[record.object_id] = glb
        self.workspace.previous_inpaint_button.click()
        self._assert_selected(record.object_id, 102)


# ### Test runner ###
if __name__ == "__main__":
    unittest.main()
