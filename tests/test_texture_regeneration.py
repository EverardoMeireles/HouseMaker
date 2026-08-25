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
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication, QMessageBox
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.camera_uv_integrity import (
    CAMERA_UV_FINGERPRINT_VERSION,
    CameraUvFingerprint,
    build_camera_uv_fingerprint,
)
from housemaker.generation_state import (
    MASK_MODE_PAINT,
    GeneratedObjectRecord,
    GenerationData,
    MaskPoint,
    MaskStroke,
)
from housemaker.generation_workspace import (
    TEXTURE_INPAINT_STROKES_PIPELINE_KEY,
    TextureRegenerationRequest,
    TextureRegenerationWorker,
    GenerationWorkspace,
    MeshyTextureRegenerator,
    _GenerationCancelled,
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


# ### Fixture helpers ###
def _box_glb(scale: float = 1.0) -> bytes:
    mesh = trimesh.creation.box(
        extents=(float(scale), float(scale) * 0.5, float(scale) * 0.75)
    )
    return bytes(trimesh.Scene(mesh).export(file_type="glb"))


def _record(object_id: str, asset_path: str) -> GeneratedObjectRecord:
    return GeneratedObjectRecord(
        object_id=object_id,
        frame_index=0,
        object_name=f"Object {object_id}",
        pipeline={},
        provider_task_id=f"task-{object_id}",
        asset_path=asset_path,
    )


def _write_test_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (80, 60),
    )
    if not writer.isOpened():
        raise unittest.SkipTest("MJPG writer is unavailable")
    try:
        for frame_index in range(2):
            writer.write(
                np.full(
                    (60, 80, 3),
                    (frame_index * 40, 40, 160),
                    dtype=np.uint8,
                )
            )
    finally:
        writer.release()


def _camera_uv_glb(*, mutate_uv: bool = False) -> bytes:
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        dtype=float,
    )
    faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    uv = np.asarray(
        ((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)),
        dtype=float,
    )
    if mutate_uv:
        uv[-1] = (0.3, 0.25)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual = TextureVisuals(uv=uv, material=PBRMaterial())
    return bytes(trimesh.Scene(mesh).export(file_type="glb"))


def _png_bytes(color: tuple[int, int, int, int]) -> bytes:
    pixels = np.full((8, 8, 4), color, dtype=np.uint8)
    encoded, payload = cv2.imencode(".png", pixels)
    if not encoded:
        raise RuntimeError("Test PNG encoding failed.")
    return bytes(payload)


def _texture_variants(
    generation_index: int,
) -> ObjectTextureVariants:
    glb_by_resolution = {
        resolution: _box_glb(
            1.0 + generation_index + resolution / 10_000.0
        )
        for resolution in TEXTURE_RESOLUTIONS
    }
    texture_png_by_resolution = {
        resolution: _png_bytes(
            (
                20 + generation_index * 30,
                resolution // 16 % 255,
                140,
                255,
            )
        )
        for resolution in TEXTURE_RESOLUTIONS
    }
    preview_rgba_by_resolution = {
        resolution: np.full(
            (8, 8, 4),
            (20 + generation_index * 30, 80, 140, 255),
            dtype=np.uint8,
        )
        for resolution in TEXTURE_RESOLUTIONS
    }
    return ObjectTextureVariants(
        glb_by_resolution=glb_by_resolution,
        texture_png_by_resolution=texture_png_by_resolution,
        preview_rgba_by_resolution=preview_rgba_by_resolution,
    )


def _model_with_variants(variants: ObjectTextureVariants) -> GeneratedModel:
    model = import_generated_glb(variants.glb_by_resolution[1024])
    model.object_texture_variants = variants
    return model


def _record_variant_paths(record: GeneratedObjectRecord) -> set[str]:
    raw_variants = record.pipeline["texture_variants"]
    return {
        str(path)
        for variant in raw_variants.values()
        for path in variant.values()
    }


def _asset_bytes(
    asset_directory: Path,
    raw_paths: set[str],
) -> dict[str, bytes]:
    return {
        raw_path: asset_directory.joinpath(raw_path).read_bytes()
        for raw_path in raw_paths
    }


# ### Provider fixtures ###
class _SequenceTextureRegenerator:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[TextureRegenerationRequest] = []

    def regenerate(
        self,
        request: TextureRegenerationRequest,
    ) -> MeshyGenerationResult:
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        assert isinstance(outcome, MeshyGenerationResult)
        return outcome


class _SequenceExecutor:
    def __init__(self, models: list[GeneratedModel]) -> None:
        self.models = list(models)
        self.results: list[MeshyGenerationResult] = []

    def execute(self, result: MeshyGenerationResult) -> GeneratedModel:
        self.results.append(result)
        return self.models.pop(0)


class _BlockingTextureRegenerator:
    def __init__(self, result: MeshyGenerationResult) -> None:
        self.result = result
        self.requests: list[TextureRegenerationRequest] = []
        self.started = threading.Event()
        self.release = threading.Event()

    def regenerate(
        self,
        request: TextureRegenerationRequest,
    ) -> MeshyGenerationResult:
        self.requests.append(request)
        self.started.set()
        if not self.release.wait(timeout=5.0):
            raise RuntimeError("Blocking texture test timed out.")
        return self.result


# ### Request and provider tests ###
class TextureRegenerationRequestTests(unittest.TestCase):
    def test_request_owns_reference_and_model_and_validates_uv_pairing(
        self,
    ) -> None:
        reference = np.full((6, 8, 4), (10, 20, 30, 255), dtype=np.uint8)
        model_buffer = bytearray(b"owned model")
        settings = GenerationServiceSettings(meshy_api_key="msy-key")

        request = TextureRegenerationRequest(
            object_id="  chair  ",
            reference_frame_index=7,
            reference_image_bgra=reference,
            model_glb=model_buffer,
            settings=settings,
        )
        reference[:, :] = 0
        model_buffer[:] = b"changedxxxx"

        self.assertEqual(request.object_id, "chair")
        self.assertEqual(request.reference_frame_index, 7)
        self.assertTrue(
            np.all(
                request.reference_image_bgra
                == np.asarray((10, 20, 30, 255), dtype=np.uint8)
            )
        )
        self.assertTrue(request.reference_image_bgra.flags.c_contiguous)
        self.assertEqual(request.model_glb, b"owned model")
        self.assertIs(request.settings, settings)

        fingerprint = CameraUvFingerprint(
            CAMERA_UV_FINGERPRINT_VERSION,
            "a" * 64,
            2,
        )
        with self.assertRaisesRegex(ValueError, "requires a UV fingerprint"):
            TextureRegenerationRequest(
                object_id="chair",
                reference_frame_index=0,
                reference_image_bgra=np.zeros((2, 2, 4), dtype=np.uint8),
                model_glb=b"model",
                settings=settings,
                enable_original_uv=True,
            )
        with self.assertRaisesRegex(ValueError, "only valid"):
            TextureRegenerationRequest(
                object_id="chair",
                reference_frame_index=0,
                reference_image_bgra=np.zeros((2, 2, 4), dtype=np.uint8),
                model_glb=b"model",
                settings=settings,
                submitted_uv_fingerprint=fingerprint,
            )

    def test_request_rejects_empty_identity_image_and_model(self) -> None:
        settings = GenerationServiceSettings(meshy_api_key="msy-key")
        valid_image = np.zeros((2, 2, 4), dtype=np.uint8)
        invalid_cases = (
            {"object_id": "", "reference_image_bgra": valid_image, "model_glb": b"m"},
            {"object_id": "chair", "reference_image_bgra": np.zeros((2, 2, 3)), "model_glb": b"m"},
            {"object_id": "chair", "reference_image_bgra": valid_image, "model_glb": b""},
        )
        for invalid in invalid_cases:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    TextureRegenerationRequest(
                        reference_frame_index=0,
                        settings=settings,
                        **invalid,
                    )


class MeshyTextureRegeneratorTests(unittest.TestCase):
    def test_provider_sends_owned_model_reference_and_uv_flag(self) -> None:
        reference = np.zeros((5, 7, 4), dtype=np.uint8)
        reference[:, :] = (11, 22, 33, 177)
        model_glb = _camera_uv_glb()
        fingerprint = build_camera_uv_fingerprint(model_glb)
        request = TextureRegenerationRequest(
            object_id="chair",
            reference_frame_index=3,
            reference_image_bgra=reference,
            model_glb=model_glb,
            settings=GenerationServiceSettings(meshy_api_key="msy-secret"),
            enable_original_uv=True,
            submitted_uv_fingerprint=fingerprint,
        )
        expected = MeshyGenerationResult("texture-task", model_glb, "Chair")
        progress_messages: list[str] = []

        def fake_retexture(**kwargs: object) -> MeshyGenerationResult:
            callback = kwargs["progress_callback"]
            callback("PENDING", 0)  # type: ignore[operator]
            callback("IN_PROGRESS", 47)  # type: ignore[operator]
            callback("SUCCEEDED", 100)  # type: ignore[operator]
            return expected

        with patch(
            "housemaker.generation_workspace.request_retextured_model",
            side_effect=fake_retexture,
        ) as retexture:
            actual = MeshyTextureRegenerator().regenerate(
                request,
                progress_messages.append,
                threading.Event(),
            )

        self.assertIs(actual, expected)
        self.assertEqual(retexture.call_args.kwargs["api_key"], "msy-secret")
        self.assertEqual(retexture.call_args.kwargs["model_glb"], model_glb)
        self.assertTrue(retexture.call_args.kwargs["enable_original_uv"])
        references = retexture.call_args.kwargs["reference_images_png"]
        self.assertEqual(len(references), 1)
        decoded = cv2.imdecode(
            np.frombuffer(references[0], dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
        np.testing.assert_array_equal(decoded, reference)
        self.assertIn("queued", " ".join(progress_messages).lower())
        self.assertIn("47%", " ".join(progress_messages))
        self.assertIn("complete", " ".join(progress_messages).lower())

    def test_provider_does_not_preserve_uvs_when_request_disables_it(
        self,
    ) -> None:
        request = TextureRegenerationRequest(
            object_id="chair",
            reference_frame_index=0,
            reference_image_bgra=np.zeros((2, 2, 4), dtype=np.uint8),
            model_glb=b"ordinary model",
            settings=GenerationServiceSettings(meshy_api_key="msy-secret"),
        )
        expected = MeshyGenerationResult(
            "texture-task",
            b"regenerated model",
            "Chair",
        )

        with patch(
            "housemaker.generation_workspace.request_retextured_model",
            return_value=expected,
        ) as retexture:
            actual = MeshyTextureRegenerator().regenerate(request)

        self.assertIs(actual, expected)
        self.assertFalse(retexture.call_args.kwargs["enable_original_uv"])
        self.assertEqual(
            retexture.call_args.kwargs["model_glb"],
            b"ordinary model",
        )

    def test_provider_honors_pre_cancel_before_paid_request(self) -> None:
        request = TextureRegenerationRequest(
            object_id="chair",
            reference_frame_index=0,
            reference_image_bgra=np.zeros((2, 2, 4), dtype=np.uint8),
            model_glb=b"model",
            settings=GenerationServiceSettings(meshy_api_key="msy-secret"),
        )
        cancel_event = threading.Event()
        cancel_event.set()

        with patch(
            "housemaker.generation_workspace.request_retextured_model"
        ) as retexture, self.assertRaises(_GenerationCancelled):
            MeshyTextureRegenerator().regenerate(
                request,
                cancel_event=cancel_event,
            )

        retexture.assert_not_called()

    def test_worker_rejects_returned_camera_uv_changes_before_executor(self) -> None:
        submitted_glb = _camera_uv_glb()
        request = TextureRegenerationRequest(
            object_id="chair",
            reference_frame_index=0,
            reference_image_bgra=np.zeros((2, 2, 4), dtype=np.uint8),
            model_glb=submitted_glb,
            settings=GenerationServiceSettings(meshy_api_key="msy-secret"),
            enable_original_uv=True,
            submitted_uv_fingerprint=build_camera_uv_fingerprint(submitted_glb),
        )
        result = MeshyGenerationResult(
            "changed-task",
            _camera_uv_glb(mutate_uv=True),
            "Changed",
        )
        regenerator = _SequenceTextureRegenerator([result])
        executor = _SequenceExecutor(
            [_model_with_variants(_texture_variants(9))]
        )
        worker = TextureRegenerationWorker(regenerator, executor, request)
        succeeded = QSignalSpy(worker.succeeded)
        failed = QSignalSpy(worker.failed)
        finished = QSignalSpy(worker.finished)

        worker.run()

        self.assertEqual(succeeded.count(), 0)
        self.assertEqual(failed.count(), 1)
        self.assertIn("UV", failed.at(0)[0])
        self.assertEqual(finished.count(), 1)
        self.assertEqual(executor.results, [])


# ### Regenerate-texture UI tests ###
class TextureRegenerationUiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.asset_directory = (
            Path(self.temporary_directory.name) / "generated"
        )
        self.asset_directory.mkdir(parents=True)
        self.video_path = Path(self.temporary_directory.name) / "source.avi"
        _write_test_video(self.video_path)
        self.workspace = GenerationWorkspace(
            asset_directory=self.asset_directory,
        )
        self.workspace.resize(1400, 850)
        self.workspace.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self.temporary_directory.cleanup()

    def _write_record(self, object_id: str) -> GeneratedObjectRecord:
        asset_name = f"{object_id}.glb"
        self.asset_directory.joinpath(asset_name).write_bytes(_box_glb())
        return _record(object_id, asset_name)

    def _enable_meshy(self) -> None:
        self.workspace.set_runtime_settings(
            GenerationServiceSettings(meshy_api_key="msy-test-key")
        )

    def _load_video(self) -> None:
        self.workspace.load_video(str(self.video_path))

    def _paint_current_frame_mask(self) -> None:
        stroke = MaskStroke(
            mode=MASK_MODE_PAINT,
            radius_normalized=0.12,
            points=(MaskPoint(0.5, 0.5),),
        )
        self.workspace.video_view.set_strokes([stroke])
        self.workspace.video_view.strokes_changed.emit([stroke])
        _qt_application.processEvents()

    def test_button_is_beside_generate_and_tracks_selected_id(self) -> None:
        first = self._write_record("first")
        second = self._write_record("second")
        self._enable_meshy()
        self.workspace.set_data(
            GenerationData(generated_objects=[first, second])
        )
        self._load_video()
        self._paint_current_frame_mask()

        self.assertEqual(
            self.workspace.regenerate_texture_button.objectName(),
            "generate_texture_button",
        )
        self.assertEqual(
            self.workspace.regenerate_texture_button.text(),
            "Generate texture",
        )
        self.assertLess(
            self.workspace.generate_button.geometry().left(),
            self.workspace.regenerate_texture_button.geometry().left(),
        )
        self.assertIn(
            "credits",
            self.workspace.regenerate_texture_button.toolTip().lower(),
        )
        self.assertTrue(self.workspace.regenerate_texture_button.isEnabled())
        self.assertEqual(self.workspace._selected_object_id, "second")

        self.workspace.generated_objects_list.setCurrentRow(0)
        self.assertEqual(self.workspace._selected_object_id, "first")
        self.assertTrue(self.workspace.regenerate_texture_button.isEnabled())

    def test_eligibility_tracks_key_selection_video_mask_busy_and_delete(
        self,
    ) -> None:
        valid = self._write_record("valid")
        self.workspace.set_data(GenerationData(generated_objects=[valid]))

        self.assertFalse(self.workspace.regenerate_texture_button.isEnabled())

        self._enable_meshy()
        self.assertFalse(self.workspace.regenerate_texture_button.isEnabled())

        self._load_video()
        self.assertFalse(self.workspace.regenerate_texture_button.isEnabled())

        self._paint_current_frame_mask()
        self.assertTrue(self.workspace.regenerate_texture_button.isEnabled())

        self.workspace.generated_objects_list.setCurrentRow(-1)
        self.assertFalse(self.workspace.regenerate_texture_button.isEnabled())

        self.workspace.generated_objects_list.setCurrentRow(0)
        self.assertTrue(self.workspace.regenerate_texture_button.isEnabled())
        self.workspace.video_view.clear_mask()
        self.assertFalse(self.workspace.regenerate_texture_button.isEnabled())
        self._paint_current_frame_mask()
        self.assertTrue(self.workspace.regenerate_texture_button.isEnabled())
        with patch.object(
            self.workspace,
            "_generation_thread",
            object(),
        ):
            self.workspace._sync_controls()
            self.assertFalse(
                self.workspace.regenerate_texture_button.isEnabled()
            )
        self.workspace._sync_controls()

        self.assertTrue(self.workspace.delete_generated_object("valid"))
        self.assertFalse(self.workspace.regenerate_texture_button.isEnabled())

    def test_asset_location_is_not_a_ui_eligibility_gate(self) -> None:
        self._enable_meshy()
        missing = _record("missing", "missing.glb")
        self.workspace.set_data(GenerationData(generated_objects=[missing]))
        self._load_video()
        self._paint_current_frame_mask()

        self.assertTrue(self.workspace.regenerate_texture_button.isEnabled())

        outside_path = Path(self.temporary_directory.name) / "outside.glb"
        outside_path.write_bytes(_box_glb())
        unsafe = _record("unsafe", "../outside.glb")
        self.workspace.set_data(GenerationData(generated_objects=[unsafe]))
        self._load_video()
        self._paint_current_frame_mask()

        self.assertTrue(self.workspace.regenerate_texture_button.isEnabled())

    def test_external_mode_keeps_local_button_and_selected_id_current(self) -> None:
        first = self._write_record("first")
        second = self._write_record("second")
        self._enable_meshy()
        self.workspace.set_data(
            GenerationData(generated_objects=[first, second])
        )
        self._load_video()
        self._paint_current_frame_mask()

        self.assertEqual(self.workspace.ambient_light_slider.value(), 100)
        self.assertEqual(
            self.workspace.result_view.get_ambient_light_intensity(),
            1.0,
        )

        self.workspace.set_external_3d_viewer_active(True)
        _qt_application.processEvents()

        self.assertTrue(
            self.workspace.object_3d_panel.is_external_presentation_active
        )
        self.assertIs(
            self.workspace.right_view_stack.currentWidget(),
            self.workspace.texture_view_page,
        )
        self.assertTrue(
            self.workspace.regenerate_texture_button.isVisibleTo(
                self.workspace
            )
        )
        self.assertTrue(self.workspace.regenerate_texture_button.isEnabled())
        self.assertEqual(self.workspace.ambient_light_slider.value(), 100)
        self.assertEqual(
            self.workspace.result_view.get_ambient_light_intensity(),
            1.0,
        )

        self.workspace.generated_objects_list.setCurrentRow(0)
        self.assertEqual(self.workspace._selected_object_id, "first")
        self.assertTrue(self.workspace.regenerate_texture_button.isEnabled())

        self.workspace.set_external_3d_viewer_active(False)
        _qt_application.processEvents()

        self.assertEqual(self.workspace.ambient_light_slider.value(), 100)
        self.assertEqual(
            self.workspace.result_view.get_ambient_light_intensity(),
            1.0,
        )
        self.assertIs(
            self.workspace.right_view_stack.currentWidget(),
            self.workspace.object_3d_page,
        )


# ### Regeneration pipeline tests ###
class TextureRegenerationPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.asset_directory = (
            Path(self.temporary_directory.name) / "generated"
        )
        self.asset_directory.mkdir(parents=True)
        self.video_path = Path(self.temporary_directory.name) / "source.avi"
        _write_test_video(self.video_path)
        self.workspace = GenerationWorkspace(
            asset_directory=self.asset_directory,
        )
        self.workspace.resize(1400, 850)
        self.workspace.show()
        self.workspace.set_runtime_settings(
            GenerationServiceSettings(meshy_api_key="msy-test-key")
        )
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self.temporary_directory.cleanup()

    def _seed_object(
        self,
        generation_index: int,
        *,
        name: str,
        task_id: str,
    ) -> tuple[GeneratedObjectRecord, ObjectTextureVariants]:
        variants = _texture_variants(generation_index)
        model = _model_with_variants(variants)
        result = MeshyGenerationResult(
            task_id,
            variants.glb_by_resolution[1024],
            name,
        )
        self.workspace._handle_generation_succeeded(result, model)
        return self.workspace.get_data().generated_objects[-1], variants

    def _replace_record(
        self,
        record: GeneratedObjectRecord,
        **pipeline_updates: object,
    ) -> GeneratedObjectRecord:
        pipeline = dict(record.pipeline)
        pipeline.update(pipeline_updates)
        replacement = replace(record, pipeline=pipeline)
        record_index = self.workspace._data.generated_objects.index(record)
        self.workspace._data.generated_objects[record_index] = replacement
        selected_id = self.workspace._selected_object_id
        self.workspace._refresh_generated_objects_list(selected_id)
        return replacement

    def _load_reference(self, frame_index: int = 0) -> np.ndarray:
        self.workspace.load_video(str(self.video_path))
        if frame_index:
            self.workspace.show_frame(frame_index)
        stroke = MaskStroke(
            mode=MASK_MODE_PAINT,
            radius_normalized=0.12,
            points=(MaskPoint(0.5, 0.5),),
        )
        self.workspace.video_view.set_strokes([stroke])
        self.workspace.video_view.strokes_changed.emit([stroke])
        _qt_application.processEvents()
        return self.workspace.video_view.build_selected_object_crop()

    def _wait_until_idle(self, timeout_seconds: float = 8.0) -> None:
        deadline = time.monotonic() + timeout_seconds
        while self.workspace.is_generating and time.monotonic() < deadline:
            _qt_application.processEvents()
            QTest.qWait(10)
        _qt_application.processEvents()
        self.assertFalse(
            self.workspace.is_generating,
            "Texture-regeneration worker did not finish.",
        )

    def _wait_for_event(
        self,
        event: threading.Event,
        timeout_seconds: float = 5.0,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        while not event.is_set() and time.monotonic() < deadline:
            _qt_application.processEvents()
            QTest.qWait(10)
        self.assertTrue(event.is_set(), "Blocking provider did not start.")

    def test_global_resolution_selection_updates_object_generation_once(
        self,
    ) -> None:
        record, variants = self._seed_object(
            0,
            name="Chair",
            task_id="geometry-task",
        )
        data_changed = QSignalSpy(self.workspace.data_changed)
        content_changed = QSignalSpy(self.workspace.generated_object_changed)

        self.assertTrue(
            self.workspace.select_object_texture_resolution(
                record.object_id,
                2048,
            )
        )

        updated = self.workspace.get_data().generated_objects[0]
        self.assertEqual(updated.pipeline["selected_texture_resolution"], 2048)
        self.assertEqual(
            updated.asset_path,
            updated.pipeline["texture_variants"]["2048"]["glb_asset_path"],
        )
        self.assertEqual(
            self.workspace.result_view.model.glb_bytes,
            variants.glb_by_resolution[2048],
        )
        self.assertEqual(
            self.workspace.texture_view.selected_atlas_id,
            f"{record.object_id}:resolution:2048",
        )
        self.assertEqual(data_changed.count(), 1)
        self.assertEqual(content_changed.count(), 0)

        self.assertTrue(
            self.workspace.select_object_texture_resolution(
                record.object_id,
                2048,
            )
        )
        self.assertEqual(data_changed.count(), 1)
        self.assertFalse(
            self.workspace.select_object_texture_resolution(
                record.object_id,
                4096,
            )
        )
        self.assertEqual(data_changed.count(), 1)

    def test_request_uses_current_setting_to_preserve_camera_uv_provenance(
        self,
    ) -> None:
        record, variants = self._seed_object(
            0,
            name="Chair",
            task_id="geometry-task",
        )
        self.workspace.texture_view.select_atlas(
            f"{record.object_id}:resolution:2048"
        )
        expected_reference = self._load_reference(frame_index=1)

        ordinary_request = self.workspace._build_texture_regeneration_request()

        self.assertIsNotNone(ordinary_request)
        assert ordinary_request is not None
        self.assertEqual(
            ordinary_request.model_glb,
            variants.glb_by_resolution[1024],
        )
        self.assertEqual(ordinary_request.reference_frame_index, 1)
        np.testing.assert_array_equal(
            ordinary_request.reference_image_bgra,
            expected_reference,
        )
        self.assertFalse(ordinary_request.enable_original_uv)
        self.assertIsNone(ordinary_request.submitted_uv_fingerprint)

        postprocessed_glb = _camera_uv_glb()
        postprocessed_name = f"{record.object_id}.postprocessed.glb"
        self.asset_directory.joinpath(postprocessed_name).write_bytes(
            postprocessed_glb
        )
        fingerprint = build_camera_uv_fingerprint(postprocessed_glb)
        current = self.workspace.get_data().generated_objects[0]
        projected = self._replace_record(
            current,
            postprocessed_asset_path=postprocessed_name,
            camera_uv_projection_applied=True,
            retexture_enable_original_uv=True,
            camera_uv_fingerprint_version=(
                CAMERA_UV_FINGERPRINT_VERSION
            ),
            camera_uv_submitted_fingerprint=fingerprint.sha256,
            camera_uv_integrity_face_count=fingerprint.face_count,
            retained_provenance="keep-me",
        )

        projected_request = self.workspace._build_texture_regeneration_request()

        self.assertIsNotNone(projected_request)
        assert projected_request is not None
        self.assertEqual(projected_request.object_id, projected.object_id)
        self.assertEqual(projected_request.model_glb, postprocessed_glb)
        self.assertFalse(projected_request.enable_original_uv)
        self.assertIsNone(projected_request.submitted_uv_fingerprint)

        self.workspace.set_runtime_settings(
            GenerationServiceSettings(
                meshy_api_key="msy-test-key",
                project_uvs_from_camera_views=True,
            )
        )
        projected_request = self.workspace._build_texture_regeneration_request()

        self.assertIsNotNone(projected_request)
        assert projected_request is not None
        self.assertTrue(projected_request.enable_original_uv)
        self.assertEqual(
            projected_request.submitted_uv_fingerprint,
            fingerprint,
        )

        corrupt = self._replace_record(
            projected,
            camera_uv_submitted_fingerprint="0" * 64,
        )
        regenerator = _SequenceTextureRegenerator(
            [MeshyGenerationResult("should-not-run", postprocessed_glb, "Chair")]
        )
        self.workspace.set_texture_regenerator(regenerator)

        self.assertFalse(self.workspace.regenerate_selected_object_texture())
        self.assertEqual(regenerator.requests, [])
        self.assertFalse(self.workspace.is_generating)
        self.assertIn("no longer matches", self.workspace.status_label.text())
        self.assertEqual(
            self.workspace.get_data().generated_objects[0],
            corrupt,
        )

    def test_success_replaces_variants_repeats_and_refreshes_external_view(
        self,
    ) -> None:
        original, original_variants = self._seed_object(
            0,
            name="Chair",
            task_id="original-task",
        )
        original = self._replace_record(
            original,
            texture_inpaint_strokes=[
                {
                    "mode": "paint",
                    "radius_normalized": 0.04,
                    "points": [{"u": 0.25, "v": 0.75}],
                }
            ],
            retained_provenance={"geometry": "unchanged"},
        )
        self.workspace.texture_view.select_atlas(
            f"{original.object_id}:resolution:2048"
        )
        original = self.workspace.get_data().generated_objects[0]
        original_paths = _record_variant_paths(original)
        original_bytes = _asset_bytes(self.asset_directory, original_paths)
        expected_reference = self._load_reference(frame_index=1)
        first_variants = _texture_variants(1)
        second_variants = _texture_variants(2)
        first_result = MeshyGenerationResult(
            "regenerated-task-1",
            first_variants.glb_by_resolution[1024],
            "Chair",
        )
        second_result = MeshyGenerationResult(
            "regenerated-task-2",
            second_variants.glb_by_resolution[1024],
            "Chair",
        )
        regenerator = _SequenceTextureRegenerator(
            [first_result, second_result]
        )
        executor = _SequenceExecutor(
            [
                _model_with_variants(first_variants),
                _model_with_variants(second_variants),
            ]
        )
        self.workspace.set_texture_regenerator(regenerator)
        self.workspace.set_meshy_executor(executor)
        self.workspace.wireframe_checkbox.setChecked(True)
        self.workspace.set_external_3d_viewer_active(True)
        changed = QSignalSpy(self.workspace.data_changed)
        completed = QSignalSpy(
            self.workspace.texture_regeneration_completed
        )
        generated = QSignalSpy(self.workspace.generation_completed)

        self.assertTrue(self.workspace.regenerate_selected_object_texture())
        self._wait_until_idle()

        self.assertEqual(len(regenerator.requests), 1)
        first_request = regenerator.requests[0]
        self.assertEqual(first_request.object_id, original.object_id)
        self.assertEqual(
            first_request.model_glb,
            original_variants.glb_by_resolution[1024],
        )
        np.testing.assert_array_equal(
            first_request.reference_image_bgra,
            expected_reference,
        )
        first_record = self.workspace.get_data().generated_objects[0]
        first_paths = _record_variant_paths(first_record)
        self.assertEqual(first_record.object_id, original.object_id)
        self.assertEqual(first_record.object_name, original.object_name)
        self.assertEqual(first_record.frame_index, original.frame_index)
        self.assertEqual(first_record.provider_task_id, "regenerated-task-1")
        self.assertEqual(
            first_record.pipeline["retained_provenance"],
            {"geometry": "unchanged"},
        )
        self.assertNotIn(
            TEXTURE_INPAINT_STROKES_PIPELINE_KEY,
            first_record.pipeline,
        )
        self.assertEqual(
            first_record.pipeline["selected_texture_resolution"],
            2048,
        )
        self.assertEqual(first_record.pipeline["texture_regeneration_count"], 1)
        self.assertEqual(
            first_record.pipeline["latest_texture_task_id"],
            "regenerated-task-1",
        )
        self.assertTrue(first_paths.isdisjoint(original_paths))
        self.assertTrue(
            all(path.startswith("regenerated-") for path in first_paths)
        )
        self.assertTrue(
            all(self.asset_directory.joinpath(path).is_file() for path in original_paths)
        )
        self.assertNotEqual(
            _asset_bytes(self.asset_directory, first_paths),
            original_bytes,
        )
        self.assertEqual(
            self.workspace.result_view.model.glb_bytes,
            first_variants.glb_by_resolution[2048],
        )
        self.assertEqual(len(self.workspace.texture_view.entries), 3)
        self.assertEqual(
            self.workspace.texture_view.selected_atlas_id,
            f"{original.object_id}:resolution:2048",
        )
        self.assertTrue(self.workspace.texture_view.uv_overlay_enabled)
        self.assertTrue(self.workspace.result_view.get_wireframe_enabled())
        self.assertIs(
            self.workspace.right_view_stack.currentWidget(),
            self.workspace.texture_view_page,
        )

        self.assertTrue(self.workspace.regenerate_selected_object_texture())
        self._wait_until_idle()

        self.assertEqual(len(regenerator.requests), 2)
        self.assertEqual(
            regenerator.requests[1].model_glb,
            first_variants.glb_by_resolution[1024],
        )
        second_record = self.workspace.get_data().generated_objects[0]
        second_paths = _record_variant_paths(second_record)
        self.assertEqual(second_record.provider_task_id, "regenerated-task-2")
        self.assertEqual(second_record.pipeline["texture_regeneration_count"], 2)
        self.assertEqual(
            [
                entry["task_id"]
                for entry in second_record.pipeline["texture_regeneration_history"]
            ],
            ["regenerated-task-1", "regenerated-task-2"],
        )
        self.assertTrue(second_paths.isdisjoint(first_paths))
        self.assertTrue(
            all(self.asset_directory.joinpath(path).is_file() for path in first_paths)
        )
        self.assertTrue(
            all(self.asset_directory.joinpath(path).is_file() for path in second_paths)
        )
        self.assertEqual(changed.count(), 2)
        self.assertEqual(completed.count(), 2)
        self.assertEqual(generated.count(), 0)

        saved = self.workspace.get_data()
        self.workspace.set_data(saved)
        restored = self.workspace.get_data().generated_objects[0]
        self.assertEqual(restored, second_record)
        active = self.workspace.get_active_texture_variant(restored.object_id)
        self.assertIsNotNone(active)
        assert active is not None
        self.assertEqual(active.resolution, 2048)
        self.assertEqual(active.glb_asset_path.read_bytes(), second_variants.glb_by_resolution[2048])

    def test_undo_restores_texture_record_resolution_and_provenance(self) -> None:
        original, original_variants = self._seed_object(
            0,
            name="Chair",
            task_id="geometry-task",
        )
        self.workspace.texture_view.select_atlas(
            f"{original.object_id}:resolution:2048"
        )
        original = self.workspace.get_data().generated_objects[0]
        original_paths = _record_variant_paths(original)
        self._load_reference()

        generated_variants = _texture_variants(9)
        self.workspace.set_texture_regenerator(
            _SequenceTextureRegenerator(
                [
                    MeshyGenerationResult(
                        "texture-task",
                        generated_variants.glb_by_resolution[1024],
                        "Chair",
                    )
                ]
            )
        )
        self.workspace.set_meshy_executor(
            _SequenceExecutor([_model_with_variants(generated_variants)])
        )
        changed = QSignalSpy(self.workspace.generated_object_changed)

        self.assertTrue(self.workspace.generate_selected_object_texture())
        self._wait_until_idle()

        generated = self.workspace.get_data().generated_objects[0]
        generated_paths = _record_variant_paths(generated)
        self.assertEqual(generated.provider_task_id, "texture-task")
        self.assertTrue(self.workspace.undo_object_change_button.isEnabled())
        self.assertTrue(
            all(self.asset_directory.joinpath(path).is_file() for path in original_paths)
        )

        self.assertTrue(self.workspace.undo_selected_object_change())

        restored = self.workspace.get_data().generated_objects[0]
        self.assertEqual(restored, original)
        self.assertEqual(
            self.workspace.result_view.model.glb_bytes,
            original_variants.glb_by_resolution[2048],
        )
        self.assertTrue(
            all(self.asset_directory.joinpath(path).is_file() for path in original_paths)
        )
        self.assertTrue(
            all(not self.asset_directory.joinpath(path).exists() for path in generated_paths)
        )
        self.assertFalse(self.workspace.undo_object_change_button.isEnabled())
        self.assertEqual(changed.count(), 2)

    def test_selection_change_during_task_updates_target_without_hijacking_view(
        self,
    ) -> None:
        first, _first_variants = self._seed_object(
            0,
            name="First",
            task_id="first-task",
        )
        second, second_variants = self._seed_object(
            5,
            name="Second",
            task_id="second-task",
        )
        self.workspace.generated_objects_list.setCurrentRow(0)
        self.assertEqual(self.workspace._selected_object_id, first.object_id)
        first = self.workspace.get_data().generated_objects[0]
        second = self.workspace.get_data().generated_objects[1]
        first_old_paths = _record_variant_paths(first)
        second_old_paths = _record_variant_paths(second)
        second_old_bytes = _asset_bytes(self.asset_directory, second_old_paths)
        self._load_reference()

        new_variants = _texture_variants(6)
        result = MeshyGenerationResult(
            "first-regenerated",
            new_variants.glb_by_resolution[1024],
            "First",
        )
        regenerator = _BlockingTextureRegenerator(result)
        executor = _SequenceExecutor([_model_with_variants(new_variants)])
        self.workspace.set_texture_regenerator(regenerator)
        self.workspace.set_meshy_executor(executor)
        self.workspace.set_external_3d_viewer_active(True)
        completed = QSignalSpy(
            self.workspace.texture_regeneration_completed
        )

        self.assertTrue(self.workspace.regenerate_selected_object_texture())
        self._wait_for_event(regenerator.started)
        self.workspace.generated_objects_list.setCurrentRow(1)
        self.assertEqual(self.workspace._selected_object_id, second.object_id)
        regenerator.release.set()
        self._wait_until_idle()

        records = self.workspace.get_data().generated_objects
        self.assertEqual([record.object_id for record in records], [first.object_id, second.object_id])
        self.assertEqual(records[0].provider_task_id, "first-regenerated")
        self.assertEqual(records[1], second)
        self.assertEqual(self.workspace._selected_object_id, second.object_id)
        self.assertEqual(
            self.workspace.result_view.model.glb_bytes,
            second_variants.glb_by_resolution[1024],
        )
        self.assertEqual(
            _asset_bytes(self.asset_directory, second_old_paths),
            second_old_bytes,
        )
        self.assertTrue(
            all(self.asset_directory.joinpath(path).is_file() for path in first_old_paths)
        )
        self.assertEqual(completed.count(), 1)
        self.assertEqual(completed.at(0)[0].object_id, first.object_id)

    def test_provider_failure_keeps_record_files_cache_and_display(self) -> None:
        record, _variants = self._seed_object(
            0,
            name="Chair",
            task_id="original-task",
        )
        self._load_reference()
        record = self.workspace.get_data().generated_objects[0]
        old_paths = _record_variant_paths(record)
        old_bytes = _asset_bytes(self.asset_directory, old_paths)
        old_model = self.workspace.result_view.model
        regenerator = _SequenceTextureRegenerator(
            [RuntimeError("provider rejected reference")]
        )
        executor = _SequenceExecutor([])
        self.workspace.set_texture_regenerator(regenerator)
        self.workspace.set_meshy_executor(executor)
        changed = QSignalSpy(self.workspace.data_changed)
        completed = QSignalSpy(
            self.workspace.texture_regeneration_completed
        )

        with patch.object(QMessageBox, "warning") as warning:
            self.assertTrue(self.workspace.regenerate_selected_object_texture())
            self._wait_until_idle()

        self.assertEqual(self.workspace.get_data().generated_objects[0], record)
        self.assertEqual(_asset_bytes(self.asset_directory, old_paths), old_bytes)
        self.assertEqual(
            {path.name for path in self.asset_directory.iterdir()},
            old_paths,
        )
        self.assertIs(self.workspace.result_view.model, old_model)
        self.assertEqual(executor.results, [])
        self.assertEqual(changed.count(), 0)
        self.assertEqual(completed.count(), 0)
        warning.assert_called_once()
        self.assertIn("provider rejected", self.workspace.status_label.text())

    def test_partial_write_and_post_persist_import_failures_roll_back(self) -> None:
        failure_modes = ("write", "import")
        for failure_mode in failure_modes:
            with self.subTest(failure_mode=failure_mode):
                self.workspace.set_data(None)
                for path in tuple(self.asset_directory.iterdir()):
                    path.unlink()
                record, _variants = self._seed_object(
                    0,
                    name="Chair",
                    task_id="original-task",
                )
                self._load_reference()
                record = self.workspace.get_data().generated_objects[0]
                old_paths = _record_variant_paths(record)
                old_bytes = _asset_bytes(self.asset_directory, old_paths)
                new_variants = _texture_variants(4)
                result = MeshyGenerationResult(
                    "regenerated-task",
                    new_variants.glb_by_resolution[1024],
                    "Chair",
                )
                regenerator = _SequenceTextureRegenerator([result])
                executor = _SequenceExecutor(
                    [_model_with_variants(new_variants)]
                )
                self.workspace.set_texture_regenerator(regenerator)
                self.workspace.set_meshy_executor(executor)
                original_persist = self.workspace._persist_meshy_named_asset
                write_count = 0
                import_count = 0

                def persist_or_fail(file_name: str, payload: bytes) -> str:
                    nonlocal write_count
                    write_count += 1
                    if failure_mode == "write" and write_count == 4:
                        raise OSError("injected fourth write failure")
                    return original_persist(file_name, payload)

                def import_or_fail(payload: bytes) -> GeneratedModel:
                    nonlocal import_count
                    import_count += 1
                    if failure_mode == "import" and import_count == 2:
                        raise RuntimeError("injected reimport failure")
                    return import_generated_glb(payload)

                import_patch = patch(
                    "housemaker.generation_workspace.import_generated_glb",
                    side_effect=import_or_fail,
                )
                with (
                    patch.object(
                        self.workspace,
                        "_persist_meshy_named_asset",
                        side_effect=persist_or_fail,
                    ),
                    import_patch,
                    patch.object(QMessageBox, "warning") as warning,
                ):
                    self.assertTrue(
                        self.workspace.regenerate_selected_object_texture()
                    )
                    self._wait_until_idle()

                self.assertEqual(
                    self.workspace.get_data().generated_objects[0],
                    record,
                )
                self.assertEqual(
                    _asset_bytes(self.asset_directory, old_paths),
                    old_bytes,
                )
                self.assertEqual(
                    {path.name for path in self.asset_directory.iterdir()},
                    old_paths,
                )
                warning.assert_called_once()

    def test_shutdown_cancellation_keeps_existing_transaction(self) -> None:
        record, _variants = self._seed_object(
            0,
            name="Chair",
            task_id="original-task",
        )
        self._load_reference()
        record = self.workspace.get_data().generated_objects[0]
        old_paths = _record_variant_paths(record)
        old_bytes = _asset_bytes(self.asset_directory, old_paths)
        new_variants = _texture_variants(8)
        result = MeshyGenerationResult(
            "late-task",
            new_variants.glb_by_resolution[1024],
            "Chair",
        )
        regenerator = _BlockingTextureRegenerator(result)
        executor = _SequenceExecutor([_model_with_variants(new_variants)])
        self.workspace.set_texture_regenerator(regenerator)
        self.workspace.set_meshy_executor(executor)
        changed = QSignalSpy(self.workspace.data_changed)
        completed = QSignalSpy(
            self.workspace.texture_regeneration_completed
        )

        self.assertTrue(self.workspace.regenerate_selected_object_texture())
        self._wait_for_event(regenerator.started)
        self.workspace.shutdown()
        regenerator.release.set()
        QTest.qWait(30)
        _qt_application.processEvents()

        self.assertFalse(self.workspace.is_generating)
        self.assertEqual(self.workspace.get_data().generated_objects[0], record)
        self.assertEqual(_asset_bytes(self.asset_directory, old_paths), old_bytes)
        self.assertEqual(
            {path.name for path in self.asset_directory.iterdir()},
            old_paths,
        )
        self.assertEqual(executor.results, [])
        self.assertEqual(changed.count(), 0)
        self.assertEqual(completed.count(), 0)


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
