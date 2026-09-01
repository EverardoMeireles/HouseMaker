# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import trimesh
from PySide6.QtCore import QThread
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication

from housemaker.generation_state import (
    GeneratedObjectPlacement,
    GeneratedObjectRecord,
    GenerationData,
)
from housemaker.generation_jobs import (
    JOB_STATUS_CANCELLED,
    JOB_STATUS_COMPLETED,
    GenerationJobManager,
)
from housemaker.generation_workspace import (
    OBJECT_OPERATION_GENERATE_MODEL,
    GenerationRequest,
    GenerationWorker,
    GenerationWorkspace,
    TextureRegenerationRequest,
    _persist_generated_named_asset,
)
from housemaker.glb import GeneratedModel, import_generated_glb
from housemaker.meshy_generation import MeshyGenerationResult
from housemaker.object_texture_variants import (
    TEXTURE_RESOLUTIONS,
    ObjectTextureVariants,
)
from housemaker.settings_widget import GenerationServiceSettings


# ### Test application ###
_qt_application = QApplication.instance() or QApplication([])


# ### Timing constants ###
WORKER_START_TIMEOUT_SECONDS = 10.0


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
    paths: set[str] = set()
    for variant in raw_variants.values():
        if not isinstance(variant, dict):
            continue
        paths.update(
            str(variant[path_key])
            for path_key in ("glb_asset_path", "texture_asset_path")
        )
        paths.update(
            str(path)
            for path in variant.get("map_texture_asset_paths", {}).values()
        )
    return paths


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


class _PerFrameBlockingPlanner:
    """Let concurrent model requests be released independently in tests."""

    def __init__(self) -> None:
        self.started = {index: threading.Event() for index in (0, 1)}
        self.release = {index: threading.Event() for index in (0, 1)}

    def plan(self, request: GenerationRequest) -> MeshyGenerationResult:
        frame_index = request.frame_index
        self.started[frame_index].set()
        if not self.release[frame_index].wait(timeout=5.0):
            raise RuntimeError("Concurrent model fixture timed out.")
        return MeshyGenerationResult(
            f"model-task-{frame_index}",
            _box_glb(1.0 + frame_index),
            f"Provider object {frame_index}",
        )


class _PerObjectBlockingTextureRegenerator:
    """Let different object texture requests overlap deterministically."""

    def __init__(
        self,
        results: dict[str, MeshyGenerationResult],
    ) -> None:
        self.results = results
        self.started = {
            object_id: threading.Event() for object_id in results
        }
        self.release = {
            object_id: threading.Event() for object_id in results
        }

    def regenerate(
        self,
        request: TextureRegenerationRequest,
    ) -> MeshyGenerationResult:
        object_id = request.object_id
        self.started[object_id].set()
        if not self.release[object_id].wait(timeout=5.0):
            raise RuntimeError("Concurrent texture fixture timed out.")
        return self.results[object_id]


class _ResultModelExecutor:
    """Resolve an independent prepared model for each provider task."""

    def __init__(self, models: dict[str, GeneratedModel]) -> None:
        self.models = models

    def execute(self, result: MeshyGenerationResult) -> GeneratedModel:
        return self.models[result.task_id]


# ### Cancellation tests ###
class GenerationCancellationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.asset_directory = Path(self.temporary_directory.name) / "assets"
        self.job_manager = GenerationJobManager()
        self.workspace = GenerationWorkspace(
            asset_directory=self.asset_directory,
            job_manager=self.job_manager,
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
        deadline = time.monotonic() + WORKER_START_TIMEOUT_SECONDS
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

    def _wait_for_record_count(self, expected_count: int) -> None:
        deadline = time.monotonic() + 5.0
        while (
            len(self.workspace.get_data().generated_objects) < expected_count
            and time.monotonic() < deadline
        ):
            _qt_application.processEvents()
            QTest.qWait(5)
        self.assertEqual(
            len(self.workspace.get_data().generated_objects),
            expected_count,
        )

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

    def test_direct_placement_update_replaces_only_the_requested_object(
        self,
    ) -> None:
        chair, _variants = self._seed_textured_object()
        chair_placement = GeneratedObjectPlacement(2, 10.0, 20.0)
        table_placement = GeneratedObjectPlacement(3, 30.0, 40.0)
        chair = replace(chair, placement=chair_placement)
        table = replace(
            chair,
            object_id="table",
            object_name="Table",
            provider_task_id="table-task",
            placement=table_placement,
        )
        self.workspace.set_data(
            GenerationData(generated_objects=[chair, table])
        )
        data_changed = QSignalSpy(self.workspace.data_changed)
        placement_changed = QSignalSpy(
            self.workspace.generated_object_placement_changed
        )
        replacement_placement = GeneratedObjectPlacement(
            2,
            55.0,
            65.0,
            height_offset_meters=1.4,
            rotation_degrees=(10.0, 25.0, -40.0),
        )

        self.assertTrue(
            self.workspace.update_generated_object_placement(
                "chair",
                replacement_placement,
            )
        )

        records = {
            record.object_id: record
            for record in self.workspace.get_data().generated_objects
        }
        self.assertEqual(records["chair"].placement, replacement_placement)
        self.assertEqual(records["table"].placement, table_placement)
        self.assertEqual(data_changed.count(), 1)
        self.assertEqual(placement_changed.count(), 1)
        self.assertEqual(placement_changed.at(0)[0], records["chair"])

        quiet_placement = replace(
            replacement_placement,
            image_x=75.0,
        )
        self.assertTrue(
            self.workspace.update_generated_object_placement(
                "chair",
                quiet_placement,
                emit_change_signals=False,
            )
        )
        self.assertEqual(
            self.workspace.get_generated_object_placement("chair"),
            quiet_placement,
        )
        self.assertEqual(data_changed.count(), 1)
        self.assertEqual(placement_changed.count(), 1)

        self.assertFalse(
            self.workspace.update_generated_object_placement(
                "missing",
                replacement_placement,
            )
        )
        self.assertEqual(data_changed.count(), 1)
        self.assertEqual(placement_changed.count(), 1)

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

    def test_two_model_jobs_overlap_and_commit_in_reverse_order(self) -> None:
        planner = _PerFrameBlockingPlanner()
        self.workspace.set_meshy_planner(planner)
        self.workspace.set_meshy_executor(
            _ResultModelExecutor(
                {
                    "model-task-0": _plain_model(1.0),
                    "model-task-1": _plain_model(2.0),
                }
            )
        )
        first_request = self._generation_request()
        second_request = GenerationRequest(
            frame_index=1,
            selected_object_bgra=np.zeros((4, 4, 4), dtype=np.uint8),
            settings=GenerationServiceSettings(meshy_api_key="key"),
        )

        self.workspace._start_generation(
            first_request,
            requested_name="  Custom chair  ",
        )
        self.workspace._start_generation(second_request, requested_name="")
        self._wait_for_event(planner.started[0])
        self._wait_for_event(planner.started[1])

        self.assertEqual(len(self.workspace._object_job_runtimes), 2)
        planner.release[1].set()
        self._wait_for_record_count(1)
        self.assertEqual(
            self.workspace.get_data().generated_objects[0].object_name,
            "Provider object 1",
        )
        self.assertTrue(self.workspace.is_generating)

        planner.release[0].set()
        self._wait_until_idle()

        records = self.workspace.get_data().generated_objects
        self.assertEqual(
            [record.provider_task_id for record in records],
            ["model-task-1", "model-task-0"],
        )
        self.assertEqual(
            [record.object_name for record in records],
            ["Provider object 1", "Custom chair"],
        )
        self.assertTrue(
            all(
                job.status == JOB_STATUS_COMPLETED
                for job in self.job_manager.jobs()
            )
        )
        self.assertEqual(
            {job.name for job in self.job_manager.jobs()},
            {"Custom chair", "Provider object 1"},
        )

    def test_cancelling_one_model_job_leaves_its_sibling_running(self) -> None:
        planner = _PerFrameBlockingPlanner()
        self.workspace.set_meshy_planner(planner)
        self.workspace.set_meshy_executor(
            _ResultModelExecutor(
                {
                    "model-task-0": _plain_model(1.0),
                    "model-task-1": _plain_model(2.0),
                }
            )
        )
        first_request = self._generation_request()
        second_request = GenerationRequest(
            frame_index=1,
            selected_object_bgra=np.zeros((4, 4, 4), dtype=np.uint8),
            settings=GenerationServiceSettings(meshy_api_key="key"),
        )

        self.workspace._start_generation(first_request)
        self.workspace._start_generation(second_request)
        self._wait_for_event(planner.started[0])
        self._wait_for_event(planner.started[1])
        runtimes = tuple(self.workspace._object_job_runtimes.values())
        cancelled_runtime = runtimes[0]
        assert cancelled_runtime.managed_job_id is not None

        self.assertTrue(
            self.job_manager.cancel_job(cancelled_runtime.managed_job_id)
        )
        planner.release[0].set()
        planner.release[1].set()
        self._wait_until_idle()

        records = self.workspace.get_data().generated_objects
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].provider_task_id, "model-task-1")
        statuses = {job.status for job in self.job_manager.jobs()}
        self.assertEqual(
            statuses,
            {JOB_STATUS_CANCELLED, JOB_STATUS_COMPLETED},
        )

    def test_texture_jobs_overlap_by_object_and_reject_same_target(self) -> None:
        chair, _variants = self._seed_textured_object()
        table = replace(
            chair,
            object_id="table",
            object_name="Table",
            provider_task_id="table-original-task",
        )
        self.workspace.set_data(
            GenerationData(generated_objects=[chair, table])
        )
        chair_request = self._texture_request(chair)
        table_request = self._texture_request(table)
        chair_variants = _texture_variants(3)
        table_variants = _texture_variants(4)
        results = {
            "chair": MeshyGenerationResult(
                "chair-new-texture",
                chair_variants.glb_by_resolution[1024],
                "Chair",
            ),
            "table": MeshyGenerationResult(
                "table-new-texture",
                table_variants.glb_by_resolution[1024],
                "Table",
            ),
        }
        regenerator = _PerObjectBlockingTextureRegenerator(results)
        self.workspace.set_texture_regenerator(regenerator)
        self.workspace.set_meshy_executor(
            _ResultModelExecutor(
                {
                    "chair-new-texture": _model_with_variants(
                        chair_variants
                    ),
                    "table-new-texture": _model_with_variants(
                        table_variants
                    ),
                }
            )
        )

        self.workspace._start_texture_regeneration(
            chair_request,
            requested_name="Chair polish",
        )
        self.workspace._start_texture_regeneration(table_request)
        self._wait_for_event(regenerator.started["chair"])
        self._wait_for_event(regenerator.started["table"])
        self.assertEqual(len(self.workspace._object_job_runtimes), 2)

        self.workspace._start_texture_regeneration(chair_request)
        self.assertEqual(len(self.workspace._object_job_runtimes), 2)
        regenerator.release["table"].set()
        deadline = time.monotonic() + 5.0
        while (
            self.workspace.get_data().generated_objects[1].provider_task_id
            != "table-new-texture"
            and time.monotonic() < deadline
        ):
            _qt_application.processEvents()
            QTest.qWait(5)
        regenerator.release["chair"].set()
        self._wait_until_idle()

        records = {
            record.object_id: record
            for record in self.workspace.get_data().generated_objects
        }
        self.assertEqual(
            records["chair"].provider_task_id,
            "chair-new-texture",
        )
        self.assertEqual(
            records["table"].provider_task_id,
            "table-new-texture",
        )
        self.assertEqual(len(self.job_manager.jobs()), 2)
        self.assertTrue(
            all(
                job.status == JOB_STATUS_COMPLETED
                for job in self.job_manager.jobs()
            )
        )
        self.assertTrue(
            self.workspace.status_label.text().startswith("Chair polish: ")
        )

    def test_worker_preparation_is_off_gui_and_rejected_assets_are_removed(
        self,
    ) -> None:
        gui_thread = _qt_application.thread()
        persistence_threads: list[QThread] = []

        def persist_with_thread_capture(
            asset_directory: Path,
            file_name: str,
            payload: bytes,
        ) -> str:
            persistence_threads.append(QThread.currentThread())
            return _persist_generated_named_asset(
                asset_directory,
                file_name,
                payload,
            )

        result = MeshyGenerationResult("prepared", _box_glb(), "Prepared")
        self.workspace.set_meshy_planner(_ImmediatePlanner(result))
        self.workspace.set_meshy_executor(_ImmediateExecutor(_plain_model()))
        with (
            patch(
                "housemaker.generation_workspace."
                "_persist_generated_named_asset",
                side_effect=persist_with_thread_capture,
            ),
            patch.object(
                self.workspace,
                "_commit_saved_object_generation",
            ),
        ):
            self.workspace._start_generation(self._generation_request())
            self._wait_until_idle()

        self.assertTrue(persistence_threads)
        self.assertTrue(
            all(thread != gui_thread for thread in persistence_threads)
        )
        self.assertEqual(tuple(self.asset_directory.glob("*")), ())

        symmetry_threads: list[QThread] = []
        variants = _texture_variants(7)
        self.workspace.set_meshy_executor(
            _ImmediateExecutor(_model_with_variants(variants))
        )

        def reject_symmetric_preparation(*_args: object) -> object:
            symmetry_threads.append(QThread.currentThread())
            raise RuntimeError("injected symmetric preparation rejection")

        symmetric_request = GenerationRequest(
            frame_index=0,
            selected_object_bgra=np.zeros((4, 4, 4), dtype=np.uint8),
            settings=GenerationServiceSettings(meshy_api_key="key"),
            symmetric_division_enabled=True,
        )
        with patch(
            "housemaker.generation_workspace."
            "build_automatic_symmetric_object_variants",
            side_effect=reject_symmetric_preparation,
        ):
            self.workspace._start_generation(symmetric_request)
            self._wait_until_idle()

        self.assertEqual(len(symmetry_threads), 1)
        self.assertNotEqual(symmetry_threads[0], gui_thread)
        self.assertEqual(tuple(self.asset_directory.glob("*")), ())


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
