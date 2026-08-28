# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import trimesh
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication

from housemaker.generation_state import (
    GeneratedObjectPlacement,
    GeneratedObjectRecord,
    GenerationData,
)
from housemaker.generation_workspace import (
    OBJECT_OPERATION_GENERATE_MODEL,
    GenerationRequest,
    GenerationWorker,
    GenerationWorkspace,
    TextureRegenerationRequest,
    UncheckedCameraFacePurgeRequest,
)
from housemaker.glb import GeneratedModel, import_generated_glb
from housemaker.meshy_generation import MeshyGenerationResult
from housemaker.object_texture_variants import (
    TEXTURE_RESOLUTIONS,
    ObjectTextureVariants,
)
from housemaker.settings_widget import GenerationServiceSettings
from housemaker.unused_face_removal import UncheckedCameraFacePurgeResult


# ### Test application ###
_qt_application = QApplication.instance() or QApplication([])


# ### Fixture helpers ###
def _box_glb(scale: float = 1.0) -> bytes:
    mesh = trimesh.creation.box(
        extents=(float(scale), float(scale) * 0.5, float(scale) * 0.75)
    )
    return bytes(trimesh.Scene(mesh).export(file_type="glb"))


def _plain_model(scale: float = 1.0) -> GeneratedModel:
    return import_generated_glb(_box_glb(scale))


def _png_bytes(color: tuple[int, int, int, int]) -> bytes:
    pixels = np.full((8, 8, 4), color, dtype=np.uint8)
    encoded, payload = cv2.imencode(".png", pixels)
    if not encoded:
        raise RuntimeError("Test PNG encoding failed.")
    return bytes(payload)


def _texture_variants(generation_index: int) -> ObjectTextureVariants:
    return ObjectTextureVariants(
        glb_by_resolution={
            resolution: _box_glb(
                1.0 + generation_index + resolution / 10_000.0
            )
            for resolution in TEXTURE_RESOLUTIONS
        },
        texture_png_by_resolution={
            resolution: _png_bytes(
                (30 + generation_index * 20, resolution // 16 % 255, 90, 255)
            )
            for resolution in TEXTURE_RESOLUTIONS
        },
        preview_rgba_by_resolution={
            resolution: np.full(
                (8, 8, 4),
                (30 + generation_index * 20, 70, 90, 255),
                dtype=np.uint8,
            )
            for resolution in TEXTURE_RESOLUTIONS
        },
    )


def _model_with_variants(variants: ObjectTextureVariants) -> GeneratedModel:
    model = import_generated_glb(variants.glb_by_resolution[1024])
    model.object_texture_variants = variants
    return model


def _variant_paths(record: GeneratedObjectRecord) -> set[str]:
    raw_variants = record.pipeline["texture_variants"]
    assert isinstance(raw_variants, dict)
    return {
        str(path)
        for variant in raw_variants.values()
        if isinstance(variant, dict)
        for path in variant.values()
    }


class _ImmediatePlanner:
    def __init__(self, result: MeshyGenerationResult) -> None:
        self.result = result

    def plan(self, _request: GenerationRequest) -> MeshyGenerationResult:
        return self.result


class _BlockingPlanner(_ImmediatePlanner):
    def __init__(self, result: MeshyGenerationResult) -> None:
        super().__init__(result)
        self.started = threading.Event()
        self.release = threading.Event()

    def plan(self, request: GenerationRequest) -> MeshyGenerationResult:
        self.started.set()
        if not self.release.wait(timeout=5.0):
            raise RuntimeError("Blocking generation fixture timed out.")
        return super().plan(request)


class _ImmediateTextureRegenerator:
    def __init__(self, result: MeshyGenerationResult) -> None:
        self.result = result

    def regenerate(
        self,
        _request: TextureRegenerationRequest,
    ) -> MeshyGenerationResult:
        return self.result


class _BlockingTextureRegenerator(_ImmediateTextureRegenerator):
    def __init__(self, result: MeshyGenerationResult) -> None:
        super().__init__(result)
        self.started = threading.Event()
        self.release = threading.Event()

    def regenerate(
        self,
        request: TextureRegenerationRequest,
    ) -> MeshyGenerationResult:
        self.started.set()
        if not self.release.wait(timeout=5.0):
            raise RuntimeError("Blocking texture fixture timed out.")
        return super().regenerate(request)


class _ImmediateExecutor:
    def __init__(self, model: GeneratedModel) -> None:
        self.model = model
        self.calls = 0

    def execute(self, _result: MeshyGenerationResult) -> GeneratedModel:
        self.calls += 1
        return self.model


# ### Cancellation tests ###
class GenerationCancellationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.asset_directory = Path(self.temporary_directory.name) / "assets"
        self.workspace = GenerationWorkspace(
            asset_directory=self.asset_directory
        )
        self.workspace.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self.temporary_directory.cleanup()

    def _generation_request(self) -> GenerationRequest:
        return GenerationRequest(
            frame_index=0,
            selected_object_bgra=np.zeros((4, 4, 4), dtype=np.uint8),
            settings=GenerationServiceSettings(meshy_api_key="key"),
        )

    def _texture_request(
        self,
        record: GeneratedObjectRecord,
    ) -> TextureRegenerationRequest:
        return TextureRegenerationRequest(
            object_id=record.object_id,
            reference_frame_index=0,
            reference_image_bgra=np.zeros((4, 4, 4), dtype=np.uint8),
            model_glb=self.asset_directory.joinpath(record.asset_path).read_bytes(),
            settings=GenerationServiceSettings(meshy_api_key="key"),
        )

    def _seed_textured_object(
        self,
    ) -> tuple[GeneratedObjectRecord, ObjectTextureVariants]:
        variants = _texture_variants(0)
        variant_metadata: dict[str, dict[str, str]] = {}
        for resolution in TEXTURE_RESOLUTIONS:
            glb_path = f"chair.texture-{resolution}.glb"
            png_path = f"chair.texture-{resolution}.png"
            self.asset_directory.mkdir(parents=True, exist_ok=True)
            self.asset_directory.joinpath(glb_path).write_bytes(
                variants.glb_by_resolution[resolution]
            )
            self.asset_directory.joinpath(png_path).write_bytes(
                variants.texture_png_by_resolution[resolution]
            )
            variant_metadata[str(resolution)] = {
                "glb_asset_path": glb_path,
                "texture_asset_path": png_path,
            }
        record = GeneratedObjectRecord(
            object_id="chair",
            frame_index=0,
            object_name="Chair",
            pipeline={
                "texture_variants": variant_metadata,
                "selected_texture_resolution": 1024,
            },
            provider_task_id="original-task",
            asset_path=variant_metadata["1024"]["glb_asset_path"],
        )
        self.workspace.set_data(GenerationData(generated_objects=[record]))
        return record, variants

    def _wait_for_event(self, event: threading.Event) -> None:
        deadline = time.monotonic() + 3.0
        while not event.is_set() and time.monotonic() < deadline:
            _qt_application.processEvents()
            QTest.qWait(5)
        self.assertTrue(event.is_set())

    def _wait_until_idle(self) -> None:
        deadline = time.monotonic() + 5.0
        while self.workspace.is_generating and time.monotonic() < deadline:
            _qt_application.processEvents()
            QTest.qWait(5)
        _qt_application.processEvents()
        self.assertFalse(self.workspace.is_generating)

    def test_cancel_active_generation_is_idempotent_and_discards_late_result(
        self,
    ) -> None:
        result = MeshyGenerationResult("late-task", _box_glb(), "Late chair")
        planner = _BlockingPlanner(result)
        executor = _ImmediateExecutor(_plain_model())
        self.workspace.set_meshy_planner(planner)
        self.workspace.set_meshy_executor(executor)
        completed = QSignalSpy(self.workspace.generation_completed)
        cancelled = QSignalSpy(self.workspace.operation_cancelled)
        finished = QSignalSpy(self.workspace.operation_finished)
        active_during_finished: list[bool] = []
        self.workspace.operation_finished.connect(
            lambda token: active_during_finished.append(
                self.workspace._active_object_operation is not None
                and self.workspace._active_object_operation.operation_id
                == token
            )
        )

        self.workspace._start_generation(self._generation_request())
        self._wait_for_event(planner.started)
        operation_id = self.workspace._active_object_operation.operation_id

        self.assertTrue(self.workspace.cancel_operation_button.isEnabled())
        self.assertTrue(self.workspace.cancel_current_operation())
        self.assertTrue(self.workspace.cancel_current_operation())
        self.assertFalse(self.workspace.cancel_operation_button.isEnabled())
        planner.release.set()
        self._wait_until_idle()

        self.assertEqual(completed.count(), 0)
        self.assertEqual(executor.calls, 0)
        self.assertEqual(cancelled.count(), 1)
        self.assertEqual(cancelled.at(0)[0], OBJECT_OPERATION_GENERATE_MODEL)
        self.assertEqual(finished.count(), 1)
        self.assertEqual(finished.at(0)[0], operation_id)
        self.assertEqual(active_during_finished, [True])
        self.assertEqual(self.workspace.get_data().generated_objects, [])
        self.assertFalse(self.asset_directory.exists())
        self.assertIn("cancelled", self.workspace.status_label.text().lower())

    def test_post_commit_generation_cancel_deletes_exact_generated_model(
        self,
    ) -> None:
        result = MeshyGenerationResult("task", _box_glb(), "Chair")
        self.workspace.set_meshy_planner(_ImmediatePlanner(result))
        self.workspace.set_meshy_executor(_ImmediateExecutor(_plain_model()))
        deleted = QSignalSpy(self.workspace.generated_object_deleted)
        cancel_results: list[bool] = []
        self.workspace.generation_completed.connect(
            lambda _record, _model: cancel_results.append(
                self.workspace.cancel_current_operation()
            )
        )

        self.workspace._start_generation(self._generation_request())
        self._wait_until_idle()

        self.assertEqual(cancel_results, [True])
        self.assertEqual(deleted.count(), 1)
        self.assertEqual(self.workspace.get_data().generated_objects, [])
        self.assertEqual(
            list(self.asset_directory.glob("*")),
            [],
        )
        self.assertIn("model was deleted", self.workspace.status_label.text())

    def test_cancel_active_texture_keeps_exact_old_record_and_files(self) -> None:
        original, _variants = self._seed_textured_object()
        original_paths = _variant_paths(original)
        original_bytes = {
            path: self.asset_directory.joinpath(path).read_bytes()
            for path in original_paths
        }
        next_variants = _texture_variants(1)
        result = MeshyGenerationResult(
            "late-texture",
            next_variants.glb_by_resolution[1024],
            "Chair",
        )
        regenerator = _BlockingTextureRegenerator(result)
        self.workspace.set_texture_regenerator(regenerator)
        self.workspace.set_meshy_executor(
            _ImmediateExecutor(_model_with_variants(next_variants))
        )
        changed = QSignalSpy(self.workspace.generated_object_changed)

        self.workspace._start_texture_regeneration(
            self._texture_request(original)
        )
        self._wait_for_event(regenerator.started)
        self.assertFalse(self.workspace.place_object_button.isEnabled())
        self.assertFalse(self.workspace.request_active_object_placement())
        self.assertTrue(self.workspace.cancel_current_operation())
        regenerator.release.set()
        self._wait_until_idle()

        self.assertEqual(self.workspace.get_data().generated_objects, [original])
        self.assertEqual(changed.count(), 0)
        self.assertEqual(
            {
                path: self.asset_directory.joinpath(path).read_bytes()
                for path in original_paths
            },
            original_bytes,
        )
        self.assertEqual(
            {path.name for path in self.asset_directory.iterdir()},
            original_paths,
        )

    def test_post_commit_texture_cancel_restores_old_texture_and_assets(
        self,
    ) -> None:
        original, original_variants = self._seed_textured_object()
        original_paths = _variant_paths(original)
        next_variants = _texture_variants(2)
        result = MeshyGenerationResult(
            "new-texture",
            next_variants.glb_by_resolution[1024],
            "Chair",
        )
        self.workspace.set_texture_regenerator(
            _ImmediateTextureRegenerator(result)
        )
        self.workspace.set_meshy_executor(
            _ImmediateExecutor(_model_with_variants(next_variants))
        )
        changed = QSignalSpy(self.workspace.generated_object_changed)
        cancel_results: list[bool] = []
        self.workspace.texture_regeneration_completed.connect(
            lambda _record, _model: cancel_results.append(
                self.workspace.cancel_current_operation()
            )
        )

        self.workspace._start_texture_regeneration(
            self._texture_request(original)
        )
        self._wait_until_idle()

        restored = self.workspace.get_data().generated_objects[0]
        self.assertEqual(cancel_results, [True])
        self.assertEqual(restored, original)
        self.assertEqual(changed.count(), 2)
        self.assertEqual(
            self.workspace.result_view.model.glb_bytes,
            original_variants.glb_by_resolution[1024],
        )
        self.assertEqual(
            {path.name for path in self.asset_directory.iterdir()},
            original_paths,
        )
        self.assertIn(
            "previous texture was restored",
            self.workspace.status_label.text(),
        )

    def test_place_token_is_exact_and_attaches_to_successful_record(self) -> None:
        result = MeshyGenerationResult("task", _box_glb(), "Chair")
        planner = _BlockingPlanner(result)
        self.workspace.set_meshy_planner(planner)
        self.workspace.set_meshy_executor(_ImmediateExecutor(_plain_model()))
        requested = QSignalSpy(self.workspace.placement_requested)

        self.workspace._start_generation(self._generation_request())
        self._wait_for_event(planner.started)
        self.assertTrue(self.workspace.place_object_button.isEnabled())
        self.assertTrue(self.workspace.request_active_object_placement())
        operation_id = str(requested.at(0)[0])
        with self.assertRaises(AttributeError):
            setattr(
                self.workspace._active_object_operation,
                "operation_id",
                "mutated",
            )
        placement = GeneratedObjectPlacement(2, 123.5, 48.25)

        self.assertFalse(
            self.workspace.set_active_object_placement("stale", placement)
        )
        self.assertTrue(
            self.workspace.set_active_object_placement(operation_id, placement)
        )
        planner.release.set()
        self._wait_until_idle()

        record = self.workspace.get_data().generated_objects[0]
        self.assertEqual(record.placement, placement)
        self.assertIsNotNone(
            self.workspace.get_generated_object_model(record.object_id)
        )
        self.assertIsNone(self.workspace.get_generated_object_model("missing"))
        self.assertFalse(
            self.workspace.set_active_object_placement(operation_id, placement)
        )

    def test_completed_object_can_be_repositioned_with_a_fresh_bound_token(
        self,
    ) -> None:
        original, _variants = self._seed_textured_object()
        requested = QSignalSpy(self.workspace.placement_requested)
        finished = QSignalSpy(self.workspace.placement_request_finished)
        placement_changed = QSignalSpy(
            self.workspace.generated_object_placement_changed
        )
        object_changed = QSignalSpy(self.workspace.generated_object_changed)

        self.assertTrue(self.workspace.place_object_button.isEnabled())
        self.assertTrue(self.workspace.request_object_placement())
        first_request_id = str(requested.at(0)[0])
        self.assertTrue(self.workspace.request_object_placement())
        second_request_id = str(requested.at(1)[0])

        self.assertNotEqual(first_request_id, second_request_id)
        self.assertEqual(finished.count(), 1)
        self.assertEqual(str(finished.at(0)[0]), first_request_id)
        placement = GeneratedObjectPlacement(7, 88.5, 42.25)
        self.assertFalse(
            self.workspace.set_active_object_placement(
                first_request_id,
                placement,
            )
        )
        self.assertTrue(
            self.workspace.set_active_object_placement(
                second_request_id,
                placement,
            )
        )

        replacement = self.workspace.get_data().generated_objects[0]
        self.assertEqual(replacement.object_id, original.object_id)
        self.assertEqual(replacement.placement, placement)
        self.assertEqual(placement_changed.count(), 1)
        self.assertEqual(placement_changed.at(0)[0], replacement)
        self.assertEqual(object_changed.count(), 0)
        self.assertFalse(
            self.workspace.set_active_object_placement(
                second_request_id,
                GeneratedObjectPlacement(8, 1.0, 2.0),
            )
        )

    def test_closed_completed_object_picker_invalidates_only_its_token(
        self,
    ) -> None:
        self._seed_textured_object()
        requested = QSignalSpy(self.workspace.placement_requested)
        self.assertTrue(self.workspace.request_object_placement())
        request_id = str(requested.at(0)[0])

        self.assertFalse(
            self.workspace.cancel_object_placement_request("wrong-token")
        )
        self.assertTrue(
            self.workspace.cancel_object_placement_request(request_id)
        )
        self.assertFalse(
            self.workspace.set_active_object_placement(
                request_id,
                GeneratedObjectPlacement(2, 10.0, 20.0),
            )
        )
        self.assertIsNone(
            self.workspace.get_data().generated_objects[0].placement
        )

    def test_completed_placement_token_stays_bound_after_selection_changes(
        self,
    ) -> None:
        chair, _variants = self._seed_textured_object()
        table_asset_path = "table.glb"
        self.asset_directory.joinpath(table_asset_path).write_bytes(_box_glb())
        table = GeneratedObjectRecord(
            object_id="table",
            frame_index=0,
            object_name="Table",
            pipeline={},
            provider_task_id="table-task",
            asset_path=table_asset_path,
        )
        self.workspace.set_data(
            GenerationData(generated_objects=[chair, table])
        )
        self.workspace.generated_objects_list.setCurrentRow(1)
        requested = QSignalSpy(self.workspace.placement_requested)

        self.assertTrue(self.workspace.request_object_placement())
        request_id = str(requested.at(0)[0])
        self.workspace.generated_objects_list.setCurrentRow(0)
        placement = GeneratedObjectPlacement(4, 31.0, 57.0)
        self.assertTrue(
            self.workspace.set_active_object_placement(
                request_id,
                placement,
            )
        )

        records = {
            record.object_id: record
            for record in self.workspace.get_data().generated_objects
        }
        self.assertIsNone(records[chair.object_id].placement)
        self.assertEqual(records[table.object_id].placement, placement)

    def test_post_commit_face_purge_cancel_restores_exact_old_model(self) -> None:
        self.asset_directory.mkdir(parents=True, exist_ok=True)
        old_glb = _box_glb(1.0)
        self.asset_directory.joinpath("chair.glb").write_bytes(old_glb)
        original = GeneratedObjectRecord(
            object_id="chair",
            frame_index=0,
            object_name="Chair",
            pipeline={},
            provider_task_id="original-task",
            asset_path="chair.glb",
        )
        self.workspace.set_data(GenerationData(generated_objects=[original]))
        request = UncheckedCameraFacePurgeRequest(
            object_id=original.object_id,
            model_glb=old_glb,
            unchecked_camera_ids=("pos_x",),
        )
        purged_model = _plain_model(0.8)
        purge_result = UncheckedCameraFacePurgeResult(
            model=purged_model,
            unchecked_camera_ids=("pos_x",),
            original_face_count=12,
            retained_face_count=10,
            removed_face_count=2,
        )
        cancel_results: list[bool] = []
        self.workspace.face_purge_completed.connect(
            lambda _record, _model: cancel_results.append(
                self.workspace.cancel_current_operation()
            )
        )

        with patch(
            "housemaker.generation_workspace."
            "purge_faces_visible_from_unchecked_cameras_from_glb",
            return_value=purge_result,
        ):
            self.workspace._start_unchecked_camera_face_purge(request)
            self._wait_until_idle()

        self.assertEqual(cancel_results, [True])
        self.assertEqual(self.workspace.get_data().generated_objects, [original])
        self.assertEqual(
            self.workspace.result_view.model.glb_bytes,
            old_glb,
        )
        self.assertEqual(
            {path.name for path in self.asset_directory.iterdir()},
            {"chair.glb"},
        )

    def test_old_worker_signal_cannot_commit_into_new_operation(self) -> None:
        old_result = MeshyGenerationResult("old", _box_glb(), "Old")
        old_worker = GenerationWorker(
            _ImmediatePlanner(old_result),
            _ImmediateExecutor(_plain_model()),
            self._generation_request(),
        )
        old_worker.succeeded.connect(self.workspace._handle_generation_succeeded)
        current_result = MeshyGenerationResult("current", _box_glb(), "Current")
        current_planner = _BlockingPlanner(current_result)
        self.workspace.set_meshy_planner(current_planner)
        self.workspace.set_meshy_executor(_ImmediateExecutor(_plain_model()))

        self.workspace._start_generation(self._generation_request())
        self._wait_for_event(current_planner.started)
        active_request = self.workspace._active_generation_request
        old_worker.succeeded.emit(old_result, _plain_model())
        _qt_application.processEvents()

        self.assertEqual(self.workspace.get_data().generated_objects, [])
        self.assertIs(self.workspace._active_generation_request, active_request)
        current_planner.release.set()
        self._wait_until_idle()
        records = self.workspace.get_data().generated_objects
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].provider_task_id, "current")


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
