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
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import trimesh
from PIL import Image
from PySide6.QtCore import QPointF
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication, QMessageBox
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.camera_uv_integrity import build_camera_uv_fingerprint
from housemaker.generation_state import (
    MASK_MODE_PAINT,
    GeneratedObjectRecord,
    MaskPoint,
    MaskStroke,
)
from housemaker.generation_workspace import (
    DEFAULT_TEXTURE_INPAINT_PROMPT,
    SELECTED_TEXTURE_RESOLUTION_PIPELINE_KEY,
    TEXTURE_INPAINT_STROKES_PIPELINE_KEY,
    TEXTURE_VARIANTS_PIPELINE_KEY,
    GenerationWorkspace,
)
from housemaker.glb import import_generated_glb
from housemaker.meshy_generation import MeshyGenerationResult
from housemaker.object_texture_inpaint import (
    OBJECT_TEXTURE_INPAINT_RESOLUTION,
    TEXTURE_UV_MODE_ERASE,
    TEXTURE_UV_MODE_PAINT,
    ObjectTextureInpaintRequest,
    ObjectTextureInpaintResult,
    TextureUvPoint,
    TextureUvStroke,
    composite_object_texture_inpaint,
)
from housemaker.object_texture_variants import (
    ObjectTextureVariants,
    build_object_texture_variants,
)
from housemaker.settings_widget import (
    SURFACE_TEXTURE_PROVIDER_GPT_5_6_TERRA,
    GenerationServiceSettings,
)


# ### Test application ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Image and model fixtures ###
def _encode_rgba_png(pixels: np.ndarray) -> bytes:
    rgba = np.ascontiguousarray(pixels, dtype=np.uint8)
    did_encode, encoded = cv2.imencode(
        ".png",
        cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA),
    )
    if not did_encode:
        raise RuntimeError("Test PNG encoding failed.")
    return bytes(encoded)


def _decode_rgba_png(payload: bytes) -> np.ndarray:
    with Image.open(BytesIO(payload)) as image:
        return np.asarray(image.convert("RGBA"), dtype=np.uint8)


@lru_cache(maxsize=8)
def _textured_glb(color_seed: int = 0) -> bytes:
    mesh = trimesh.creation.box(extents=(1.0 + color_seed * 0.01, 2.0, 3.0))
    mesh.visual = TextureVisuals(
        uv=np.linspace(0.05, 0.95, len(mesh.vertices) * 2).reshape((-1, 2)),
        material=PBRMaterial(
            baseColorTexture=Image.new(
                "RGBA",
                (
                    OBJECT_TEXTURE_INPAINT_RESOLUTION,
                    OBJECT_TEXTURE_INPAINT_RESOLUTION,
                ),
                (25 + color_seed, 50, 75, 255),
            )
        ),
    )
    return bytes(trimesh.Scene(mesh).export(file_type="glb"))


@lru_cache(maxsize=8)
def _variants(color_seed: int = 0) -> ObjectTextureVariants:
    variants = build_object_texture_variants(_textured_glb(color_seed))
    if variants is None:
        raise RuntimeError("Test GLB has no texture variants.")
    return variants


@lru_cache(maxsize=16)
def _replacement_texture_png(color_seed: int = 0) -> bytes:
    resolution = OBJECT_TEXTURE_INPAINT_RESOLUTION
    pixels = np.full(
        (resolution, resolution, 4),
        (180, 70 + color_seed, 40, 255),
        dtype=np.uint8,
    )
    pixels[0, 0] = (color_seed, 2, 3, 255)
    return _encode_rgba_png(pixels)


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
                    (20 + frame_index * 40, 50, 160),
                    dtype=np.uint8,
                )
            )
    finally:
        writer.release()


def _variant_paths(record: GeneratedObjectRecord) -> set[str]:
    variants = record.pipeline[TEXTURE_VARIANTS_PIPELINE_KEY]
    return {
        str(path)
        for variant in variants.values()
        for path in variant.values()
    }


def _asset_bytes(asset_directory: Path, paths: set[str]) -> dict[str, bytes]:
    return {
        path: asset_directory.joinpath(path).read_bytes()
        for path in paths
    }


# ### Provider fixtures ###
class _SequenceInpaintProvider:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[ObjectTextureInpaintRequest] = []

    def inpaint(
        self,
        request: ObjectTextureInpaintRequest,
        progress_callback=None,
        cancel_event=None,
    ) -> ObjectTextureInpaintResult:
        self.requests.append(request)
        if progress_callback is not None:
            progress_callback("in_progress", 45)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if callable(outcome):
            outcome = outcome(request)
        assert isinstance(outcome, ObjectTextureInpaintResult)
        return outcome


class _BlockingInpaintProvider:
    def __init__(self, texture_png: bytes) -> None:
        self.texture_png = texture_png
        self.requests: list[ObjectTextureInpaintRequest] = []
        self.cancel_events: list[threading.Event | None] = []
        self.started = threading.Event()
        self.release = threading.Event()

    def inpaint(
        self,
        request: ObjectTextureInpaintRequest,
        progress_callback=None,
        cancel_event=None,
    ) -> ObjectTextureInpaintResult:
        self.requests.append(request)
        self.cancel_events.append(cancel_event)
        self.started.set()
        if not self.release.wait(timeout=8.0):
            raise RuntimeError("Blocking object-inpaint test timed out.")
        return ObjectTextureInpaintResult(
            object_id=request.object_id,
            provider=request.provider,
            texture_png=self.texture_png,
            task_id="blocked-inpaint-task",
        )


# ### Workspace fixture ###
class ObjectTextureInpaintWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        temporary_root = Path(self.temporary_directory.name)
        self.asset_directory = temporary_root / "generated"
        self.video_path = temporary_root / "source.avi"
        _write_test_video(self.video_path)
        self.provider = _SequenceInpaintProvider([])
        self.workspace = GenerationWorkspace(
            asset_directory=self.asset_directory,
            object_texture_inpaint_provider=self.provider,
        )
        self.workspace.resize(1500, 900)
        self.workspace.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self.temporary_directory.cleanup()

    def _seed_object(
        self,
        generation_index: int = 0,
        *,
        name: str = "Chair",
    ) -> GeneratedObjectRecord:
        variants = _variants(generation_index)
        model = import_generated_glb(variants.glb_by_resolution[1024])
        model.object_texture_variants = variants
        result = MeshyGenerationResult(
            f"generation-task-{generation_index}",
            variants.glb_by_resolution[1024],
            name,
        )
        self.workspace._handle_generation_succeeded(result, model)
        self.workspace._sync_controls()
        return self.workspace.get_data().generated_objects[-1]

    def _replace_record_pipeline(
        self,
        record: GeneratedObjectRecord,
        **updates: object,
    ) -> GeneratedObjectRecord:
        pipeline = dict(record.pipeline)
        pipeline.update(updates)
        replacement = replace(record, pipeline=pipeline)
        index = self.workspace._data.generated_objects.index(record)
        self.workspace._data.generated_objects[index] = replacement
        self.workspace._refresh_generated_objects_list(
            self.workspace._selected_object_id
        )
        return replacement

    def _load_video_reference(self, frame_index: int = 0) -> np.ndarray:
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

    def _add_texture_stroke(
        self,
        record: GeneratedObjectRecord,
        *,
        mode: str = TEXTURE_UV_MODE_PAINT,
    ) -> GeneratedObjectRecord:
        stroke = TextureUvStroke(
            mode=mode,
            radius_normalized=0.01,
            points=(TextureUvPoint(0.4, 0.6),),
        )
        return self.workspace._replace_texture_inpaint_strokes(
            record,
            (stroke,),
        )

    def _configure_valid_job(
        self,
        *,
        provider: str = "meshy",
        frame_index: int = 0,
    ) -> tuple[GeneratedObjectRecord, np.ndarray]:
        settings = GenerationServiceSettings(
            meshy_api_key="msy-test-key",
            openai_api_key="sk-test-key",
            surface_texture_provider=provider,
        )
        self.workspace.set_runtime_settings(settings)
        record = self._seed_object()
        expected_reference = self._load_video_reference(frame_index)
        record = self.workspace.get_data().generated_objects[0]
        record = self._add_texture_stroke(record)
        return record, expected_reference

    def _wait_until_idle(self, timeout_seconds: float = 12.0) -> None:
        deadline = time.monotonic() + timeout_seconds
        while self.workspace.is_generating and time.monotonic() < deadline:
            _qt_application.processEvents()
            QTest.qWait(10)
        _qt_application.processEvents()
        self.assertFalse(
            self.workspace.is_generating,
            "Object texture-inpaint worker did not finish.",
        )

    def _wait_for_event(
        self,
        event: threading.Event,
        timeout_seconds: float = 6.0,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        while not event.is_set() and time.monotonic() < deadline:
            _qt_application.processEvents()
            QTest.qWait(10)
        self.assertTrue(event.is_set(), "Blocking inpaint provider did not start.")

    def _successful_result(
        self,
        request: ObjectTextureInpaintRequest,
        *,
        color_seed: int = 0,
        task_id: str = "inpaint-task",
    ) -> ObjectTextureInpaintResult:
        texture_png = composite_object_texture_inpaint(
            request.existing_texture_png,
            request.edit_mask_png,
            _replacement_texture_png(color_seed),
        )
        return ObjectTextureInpaintResult(
            object_id=request.object_id,
            provider=request.provider,
            texture_png=texture_png,
            task_id=task_id,
        )

    def test_ui_order_and_eligibility_require_every_inpaint_input(self) -> None:
        self.assertEqual(
            self.workspace.paint_texture_mask_button.text(),
            "Paint texture mask",
        )
        self.assertEqual(
            self.workspace.inpaint_texture_button.text(),
            "Inpaint texture",
        )
        self.assertEqual(
            self.workspace.generate_geometry_button.text(),
            "Generate geometry",
        )
        self.assertEqual(
            self.workspace.generate_texture_button.text(),
            "Generate texture",
        )
        self.assertEqual(self.workspace.undo_object_change_button.text(), "Undo")
        controls_layout = self.workspace.status_label.parentWidget().layout()
        main_buttons = controls_layout.itemAt(1).layout()
        action_indexes = [
            main_buttons.indexOf(button)
            for button in (
                self.workspace.generate_button,
                self.workspace.generate_geometry_button,
                self.workspace.generate_texture_button,
                self.workspace.undo_object_change_button,
                self.workspace.purge_faces_button,
            )
        ]
        self.assertEqual(
            action_indexes,
            sorted(action_indexes),
        )
        generate_index = action_indexes[0]
        action_gap = main_buttons.itemAt(generate_index + 1).spacerItem()
        self.assertIsNotNone(action_gap)
        assert action_gap is not None
        self.assertEqual(action_gap.sizeHint().width(), 30)
        self.assertEqual(action_indexes[1], generate_index + 2)
        inpaint_controls = controls_layout.itemAt(2).layout()
        self.assertEqual(
            [
                inpaint_controls.itemAt(index).widget()
                for index in (0, 2, 3, 4, 5)
            ],
            [
                self.workspace.paint_texture_mask_button,
                self.workspace.texture_inpaint_instructions_edit,
                self.workspace.undo_texture_stroke_button,
                self.workspace.clear_texture_mask_button,
                self.workspace.inpaint_texture_button,
            ],
        )
        self.assertLess(
            self.workspace.generate_geometry_button.geometry().left(),
            self.workspace.generate_texture_button.geometry().left(),
        )
        self.assertLess(
            self.workspace.generate_texture_button.geometry().left(),
            self.workspace.undo_object_change_button.geometry().left(),
        )
        self.assertLess(
            self.workspace.undo_object_change_button.geometry().left(),
            self.workspace.purge_faces_button.geometry().left(),
        )
        self.assertGreater(
            self.workspace.inpaint_texture_button.geometry().top(),
            self.workspace.generate_button.geometry().top(),
        )
        self.assertFalse(self.workspace.inpaint_texture_button.isEnabled())

        record = self._seed_object()
        self.assertTrue(self.workspace.paint_texture_mask_button.isEnabled())
        self.assertFalse(self.workspace.inpaint_texture_button.isEnabled())
        self.workspace.set_runtime_settings(
            GenerationServiceSettings(meshy_api_key="msy-test-key")
        )
        self._load_video_reference()
        self.assertFalse(self.workspace.inpaint_texture_button.isEnabled())
        record = self.workspace.get_data().generated_objects[0]
        record = self._add_texture_stroke(record)
        self.assertTrue(self.workspace.inpaint_texture_button.isEnabled())

        self.workspace.video_view.clear_mask()
        self.assertFalse(self.workspace.inpaint_texture_button.isEnabled())
        self.assertFalse(self.workspace.inpaint_selected_object_texture())
        self.assertIn("paint", self.workspace.status_label.text().lower())

    def test_3d_pointer_paints_erases_undoes_clears_and_persists_uv_strokes(
        self,
    ) -> None:
        record = self._seed_object()
        self.workspace.paint_texture_mask_button.setChecked(True)
        self.assertTrue(self.workspace.result_view.view.texture_inpaint_enabled)
        first_stamp = TextureUvStroke(
            mode="paint",
            radius_normalized=0.01,
            points=(TextureUvPoint(0.40, 0.60),),
            connect_points=False,
        )
        second_stamp = replace(
            first_stamp,
            points=(TextureUvPoint(0.41, 0.60),),
        )
        with patch(
            "housemaker.generation_workspace."
            "build_texture_uv_stamp_stroke_from_screen_brush",
            side_effect=(first_stamp, second_stamp, first_stamp),
        ) as build_stamp:
            self.workspace.result_view.view.texture_inpaint_pointer_pressed.emit(
                QPointF(10.0, 10.0)
            )
            self.workspace.result_view.view.texture_inpaint_pointer_moved.emit(
                QPointF(11.0, 10.0)
            )
            self.workspace.result_view.view.texture_inpaint_pointer_released.emit(
                QPointF(11.0, 10.0)
            )
            self.workspace.erase_mask_button.setChecked(True)
            self.workspace.result_view.view.texture_inpaint_pointer_pressed.emit(
                QPointF(10.0, 10.0)
            )
            self.workspace.result_view.view.texture_inpaint_pointer_released.emit(
                QPointF(10.0, 10.0)
            )

        self.assertEqual(build_stamp.call_count, 3)
        self.assertEqual(build_stamp.call_args_list[0].args[1], (10.0, 10.0))
        self.assertEqual(
            build_stamp.call_args_list[0].kwargs["maximum_sample_count"],
            25,
        )
        strokes = self.workspace.get_texture_inpaint_strokes(record.object_id)
        self.assertEqual([stroke.mode for stroke in strokes], ["paint", "erase"])
        self.assertEqual(len(strokes[0].points), 2)
        self.assertTrue(all(not stroke.connect_points for stroke in strokes))
        self.assertFalse(self.workspace.texture_view.edit_mask_image.isNull())

        self.workspace.undo_texture_stroke_button.click()
        self.assertEqual(
            [
                stroke.mode
                for stroke in self.workspace.get_texture_inpaint_strokes(
                    record.object_id
                )
            ],
            ["paint"],
        )
        saved = self.workspace.get_data()
        self.workspace.set_data(saved)
        self.assertEqual(
            [
                stroke.mode
                for stroke in self.workspace.get_texture_inpaint_strokes(
                    record.object_id
                )
            ],
            ["paint"],
        )
        self.assertIn(
            TEXTURE_INPAINT_STROKES_PIPELINE_KEY,
            self.workspace.get_data().generated_objects[0].pipeline,
        )

        self.workspace.clear_texture_mask_button.click()
        self.assertEqual(
            self.workspace.get_texture_inpaint_strokes(record.object_id),
            (),
        )
        self.assertNotIn(
            TEXTURE_INPAINT_STROKES_PIPELINE_KEY,
            self.workspace.get_data().generated_objects[0].pipeline,
        )
        self.assertTrue(self.workspace.texture_view.edit_mask_image.isNull())

    def test_job_uses_canonical_2048_texture_current_reference_and_provider(
        self,
    ) -> None:
        record, expected_reference = self._configure_valid_job(
            provider=SURFACE_TEXTURE_PROVIDER_GPT_5_6_TERRA,
            frame_index=1,
        )
        self.workspace.texture_view.select_atlas(
            f"{record.object_id}:resolution:512"
        )
        self.workspace.texture_inpaint_instructions_edit.setText(
            "repair only the pale scratch"
        )

        job = self.workspace._build_object_texture_inpaint_job()

        self.assertIsNotNone(job)
        assert job is not None
        canonical = self.workspace.get_texture_variant(record.object_id, 2048)
        self.assertIsNotNone(canonical)
        assert canonical is not None
        self.assertEqual(job.model_glb, canonical.glb_asset_path.read_bytes())
        self.assertEqual(
            job.request.existing_texture_png,
            canonical.texture_asset_path.read_bytes(),
        )
        self.assertEqual(job.reference_frame_index, 1)
        self.assertEqual(job.request.provider, SURFACE_TEXTURE_PROVIDER_GPT_5_6_TERRA)
        self.assertEqual(job.request.api_key, "sk-test-key")
        self.assertIn(DEFAULT_TEXTURE_INPAINT_PROMPT, job.request.prompt)
        self.assertIn("repair only the pale scratch", job.request.prompt)
        decoded_reference = cv2.imdecode(
            np.frombuffer(job.request.reference_pngs[0], dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
        np.testing.assert_array_equal(decoded_reference, expected_reference)
        self.assertEqual(len(job.request.reference_pngs), 2)
        decoded_scene_reference = cv2.imdecode(
            np.frombuffer(job.request.reference_pngs[1], dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
        np.testing.assert_array_equal(
            decoded_scene_reference,
            self.workspace.video_view.get_frame_bgr(),
        )
        decoded_mask = cv2.imdecode(
            np.frombuffer(job.request.edit_mask_png, dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE,
        )
        self.assertEqual(decoded_mask.shape, (2048, 2048))
        self.assertGreater(np.count_nonzero(decoded_mask), 0)

    def test_success_is_transactional_and_preserves_identity_resolution_and_uvs(
        self,
    ) -> None:
        record, _reference = self._configure_valid_job(frame_index=1)
        self.workspace.texture_view.select_atlas(
            f"{record.object_id}:resolution:512"
        )
        record = self.workspace.get_data().generated_objects[0]
        record = self._replace_record_pipeline(
            record,
            retained_provenance={"geometry": "keep"},
        )
        old_paths = _variant_paths(record)
        old_bytes = _asset_bytes(self.asset_directory, old_paths)
        canonical_before = self.workspace.get_texture_variant(
            record.object_id,
            2048,
        )
        self.assertIsNotNone(canonical_before)
        assert canonical_before is not None
        fingerprint_before = build_camera_uv_fingerprint(
            canonical_before.glb_asset_path.read_bytes()
        )
        self.provider.outcomes.append(
            lambda request: self._successful_result(
                request,
                color_seed=9,
                task_id="inpaint-success",
            )
        )
        changed = QSignalSpy(self.workspace.data_changed)
        completed = QSignalSpy(self.workspace.texture_inpaint_completed)
        self.workspace.set_external_3d_viewer_active(True)
        self.assertTrue(
            self.workspace.object_3d_panel.is_external_presentation_active
        )
        self.assertTrue(self.workspace.texture_view.edit_mask_enabled)
        self.assertIsNotNone(self.workspace.result_view._texture_edit_mask)

        self.assertTrue(self.workspace.inpaint_selected_object_texture())
        self._wait_until_idle()

        updated = self.workspace.get_data().generated_objects[0]
        new_paths = _variant_paths(updated)
        self.assertEqual(updated.object_id, record.object_id)
        self.assertEqual(updated.object_name, record.object_name)
        self.assertEqual(updated.frame_index, record.frame_index)
        self.assertEqual(updated.provider, record.provider)
        self.assertEqual(updated.provider_task_id, record.provider_task_id)
        self.assertEqual(
            updated.pipeline["retained_provenance"],
            {"geometry": "keep"},
        )
        self.assertEqual(
            updated.pipeline[SELECTED_TEXTURE_RESOLUTION_PIPELINE_KEY],
            512,
        )
        self.assertEqual(updated.pipeline["texture_inpaint_count"], 1)
        self.assertEqual(
            updated.pipeline["latest_texture_inpaint_task_id"],
            "inpaint-success",
        )
        self.assertEqual(
            updated.pipeline["last_texture_inpaint_reference_frame_index"],
            1,
        )
        self.assertEqual(
            updated.pipeline["last_texture_inpaint_reference_image_count"],
            2,
        )
        self.assertEqual(
            len(
                updated.pipeline[
                    "last_texture_inpaint_reference_image_sha256s"
                ]
            ),
            2,
        )
        self.assertNotIn(TEXTURE_INPAINT_STROKES_PIPELINE_KEY, updated.pipeline)
        self.assertEqual(len(updated.pipeline["texture_inpaint_history"]), 1)
        self.assertTrue(new_paths.isdisjoint(old_paths))
        self.assertTrue(
            all(path.startswith("inpainted-") for path in new_paths)
        )
        self.assertTrue(
            all(not self.asset_directory.joinpath(path).exists() for path in old_paths)
        )
        self.assertNotEqual(
            _asset_bytes(self.asset_directory, new_paths),
            old_bytes,
        )
        canonical_after = self.workspace.get_texture_variant(
            record.object_id,
            2048,
        )
        self.assertIsNotNone(canonical_after)
        assert canonical_after is not None
        self.assertEqual(
            build_camera_uv_fingerprint(
                canonical_after.glb_asset_path.read_bytes()
            ),
            fingerprint_before,
        )
        active_after = self.workspace.get_active_texture_variant(record.object_id)
        self.assertIsNotNone(active_after)
        assert active_after is not None
        self.assertEqual(active_after.resolution, 512)
        self.assertEqual(
            self.workspace.result_view.model.glb_bytes,
            active_after.glb_asset_path.read_bytes(),
        )
        self.assertIs(
            self.workspace.right_view_stack.currentWidget(),
            self.workspace.texture_view_page,
        )
        self.assertTrue(self.workspace.texture_view.edit_mask_image.isNull())
        self.assertIsNone(self.workspace.result_view._texture_edit_mask)
        self.assertEqual(changed.count(), 1)
        self.assertEqual(completed.count(), 1)
        self.assertEqual(completed.at(0)[0].object_id, record.object_id)
        self.workspace.set_external_3d_viewer_active(False)
        self.assertFalse(
            self.workspace.object_3d_panel.is_external_presentation_active
        )
        self.assertIs(
            self.workspace.right_view_stack.currentWidget(),
            self.workspace.object_3d_page,
        )
        self.assertEqual(
            self.workspace.result_view.model.glb_bytes,
            active_after.glb_asset_path.read_bytes(),
        )

    def test_provider_failure_and_partial_write_restore_record_files_and_display(
        self,
    ) -> None:
        for failure_mode in ("provider", "write", "import"):
            with self.subTest(failure_mode=failure_mode):
                self.workspace.set_data(None)
                for path in tuple(self.asset_directory.glob("*")):
                    path.unlink()
                self.provider.outcomes.clear()
                record, _reference = self._configure_valid_job()
                record = self.workspace.get_data().generated_objects[0]
                old_paths = _variant_paths(record)
                old_bytes = _asset_bytes(self.asset_directory, old_paths)
                old_model_bytes = self.workspace.result_view.model.glb_bytes
                if failure_mode == "provider":
                    self.provider.outcomes.append(
                        RuntimeError("provider rejected inpaint")
                    )
                else:
                    self.provider.outcomes.append(
                        lambda request: self._successful_result(request)
                    )

                original_persist = self.workspace._persist_meshy_named_asset
                write_count = 0
                import_count = 0

                def persist_or_fail(file_name: str, payload: bytes) -> str:
                    nonlocal write_count
                    write_count += 1
                    if failure_mode == "write" and write_count == 4:
                        raise OSError("injected inpaint write failure")
                    return original_persist(file_name, payload)

                def import_or_fail(payload: bytes):
                    nonlocal import_count
                    import_count += 1
                    if failure_mode == "import" and import_count == 2:
                        raise RuntimeError("injected inpaint preview import failure")
                    return import_generated_glb(payload)

                with (
                    patch.object(
                        self.workspace,
                        "_persist_meshy_named_asset",
                        side_effect=persist_or_fail,
                    ),
                    patch(
                        "housemaker.generation_workspace.import_generated_glb",
                        side_effect=import_or_fail,
                    ),
                    patch.object(QMessageBox, "warning") as warning,
                ):
                    self.assertTrue(
                        self.workspace.inpaint_selected_object_texture()
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
                self.assertEqual(
                    self.workspace.result_view.model.glb_bytes,
                    old_model_bytes,
                )
                warning.assert_called_once()

    def test_mask_preview_switches_per_object_and_composes_before_wireframe(
        self,
    ) -> None:
        first = self._seed_object(0, name="First")
        first_stroke = TextureUvStroke(
            TEXTURE_UV_MODE_PAINT,
            0.02,
            (TextureUvPoint(0.2, 0.3),),
        )
        self.workspace._replace_texture_inpaint_strokes(
            first,
            (first_stroke,),
        )
        second = self._seed_object(3, name="Second")
        second_stroke = TextureUvStroke(
            TEXTURE_UV_MODE_PAINT,
            0.02,
            (TextureUvPoint(0.8, 0.7),),
        )
        self.workspace._replace_texture_inpaint_strokes(
            second,
            (second_stroke,),
        )
        self.workspace.wireframe_checkbox.setChecked(True)

        second_mask = self.workspace.texture_view.edit_mask_image
        self.assertFalse(second_mask.isNull())
        self.assertTrue(self.workspace.texture_view.edit_mask_enabled)
        self.assertTrue(self.workspace.texture_view.uv_overlay_enabled)
        self.assertTrue(self.workspace.texture_view.uv_overlay_triangles)
        self.workspace.generated_objects_list.setCurrentRow(0)
        _qt_application.processEvents()
        first_mask = self.workspace.texture_view.edit_mask_image
        self.assertFalse(first_mask.isNull())
        self.assertNotEqual(first_mask, second_mask)
        self.assertEqual(
            [
                stroke.mode
                for stroke in self.workspace.get_texture_inpaint_strokes(
                    first.object_id
                )
            ],
            [TEXTURE_UV_MODE_PAINT],
        )
        self.assertEqual(
            self.workspace.get_texture_inpaint_strokes(second.object_id),
            (second_stroke,),
        )

        composition_order = Mock()
        with (
            patch(
                "housemaker.texture_atlas_view._compose_edit_mask_overlay"
            ) as compose_mask,
            patch(
                "housemaker.texture_atlas_view._compose_uv_overlay"
            ) as compose_uv,
        ):
            composition_order.attach_mock(compose_mask, "mask")
            composition_order.attach_mock(compose_uv, "uv")
            self.workspace.texture_view.preview_label._sync_scaled_pixmap(
                defer_overlay=False
            )

        self.assertEqual(
            [call[0] for call in composition_order.mock_calls],
            ["mask", "uv"],
        )

    def test_selection_change_during_inpaint_updates_only_original_target(self) -> None:
        self.workspace.set_runtime_settings(
            GenerationServiceSettings(meshy_api_key="msy-test-key")
        )
        self._load_video_reference()
        first = self._seed_object(0, name="First")
        first = self._add_texture_stroke(first)
        second = self._seed_object(2, name="Second")
        second_paths = _variant_paths(second)
        second_bytes = _asset_bytes(self.asset_directory, second_paths)
        self.workspace.generated_objects_list.setCurrentRow(0)
        blocking = _BlockingInpaintProvider(_replacement_texture_png(3))
        self.workspace.set_object_texture_inpaint_provider(blocking)

        self.assertTrue(self.workspace.inpaint_selected_object_texture())
        self._wait_for_event(blocking.started)
        self.workspace.generated_objects_list.setCurrentRow(1)
        self.assertEqual(self.workspace._selected_object_id, second.object_id)
        blocking.release.set()
        self._wait_until_idle()

        records = self.workspace.get_data().generated_objects
        self.assertEqual(records[0].object_id, first.object_id)
        self.assertEqual(records[0].pipeline["texture_inpaint_count"], 1)
        self.assertEqual(records[1], second)
        self.assertEqual(
            _asset_bytes(self.asset_directory, second_paths),
            second_bytes,
        )
        self.assertEqual(self.workspace._selected_object_id, second.object_id)
        active_second = self.workspace.get_active_texture_variant(
            second.object_id
        )
        self.assertIsNotNone(active_second)
        assert active_second is not None
        self.assertEqual(
            self.workspace.result_view.model.glb_bytes,
            active_second.glb_asset_path.read_bytes(),
        )

    def test_shutdown_cancellation_keeps_existing_transaction(self) -> None:
        record, _reference = self._configure_valid_job()
        record = self.workspace.get_data().generated_objects[0]
        old_paths = _variant_paths(record)
        old_bytes = _asset_bytes(self.asset_directory, old_paths)
        blocking = _BlockingInpaintProvider(_replacement_texture_png(5))
        self.workspace.set_object_texture_inpaint_provider(blocking)
        changed = QSignalSpy(self.workspace.data_changed)
        completed = QSignalSpy(self.workspace.texture_inpaint_completed)

        self.assertTrue(self.workspace.inpaint_selected_object_texture())
        self._wait_for_event(blocking.started)
        self.workspace.shutdown()
        self.assertTrue(blocking.cancel_events)
        self.assertIsNotNone(blocking.cancel_events[0])
        self.assertTrue(blocking.cancel_events[0].is_set())
        blocking.release.set()
        QTest.qWait(50)
        _qt_application.processEvents()

        self.assertFalse(self.workspace.is_generating)
        self.assertEqual(self.workspace.get_data().generated_objects[0], record)
        self.assertEqual(_asset_bytes(self.asset_directory, old_paths), old_bytes)
        self.assertEqual(
            {path.name for path in self.asset_directory.iterdir()},
            old_paths,
        )
        self.assertEqual(changed.count(), 0)
        self.assertEqual(completed.count(), 0)


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
