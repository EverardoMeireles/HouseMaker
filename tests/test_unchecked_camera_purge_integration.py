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
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import trimesh
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from housemaker.camera_uv_integrity import (
    CAMERA_UV_FINGERPRINT_VERSION,
    CameraUvFingerprint,
)
from housemaker.generation_state import GeneratedObjectRecord
from housemaker.generation_workspace import (
    GenerationRequest,
    GenerationWorkspace,
    MeshyImagePlanner,
    StagedMeshyGenerationResult,
    TEXTURE_INPAINT_STROKES_PIPELINE_KEY,
    UncheckedCameraFacePurgeRequest,
    UncheckedCameraFacePurgeWorker,
)
from housemaker.glb import GeneratedModel, import_generated_glb
from housemaker.meshy_generation import MeshyGenerationResult
from housemaker.object_texture_variants import (
    TEXTURE_RESOLUTIONS,
    ObjectTextureVariants,
)
from housemaker.settings_widget import GenerationServiceSettings
from housemaker.unused_face_removal import (
    ALL_CAMERA_IDS,
    UncheckedCameraFacePurgeOptions,
    UncheckedCameraFacePurgeResult,
    UnusedFaceRemovalCancelled,
)


# ### Test application ###
_qt_application = QApplication.instance() or QApplication([])


# ### Fixture helpers ###
def _generation_request(
    *,
    enabled_camera_ids: tuple[str, ...],
    unused_face_removal: bool = False,
    project_camera_uvs: bool = False,
) -> GenerationRequest:
    return GenerationRequest(
        frame_index=2,
        selected_object_bgra=np.full(
            (6, 8, 4),
            (20, 80, 160, 255),
            dtype=np.uint8,
        ),
        settings=GenerationServiceSettings(
            meshy_api_key="meshy-test-key",
            unused_face_removal=unused_face_removal,
            project_uvs_from_camera_views=project_camera_uvs,
        ),
        enabled_camera_ids=enabled_camera_ids,
    )


def _purge_result(glb_bytes: bytes = b"purged geometry") -> object:
    return SimpleNamespace(
        glb_bytes=glb_bytes,
        original_face_count=120,
        retained_face_count=96,
        removed_face_count=24,
    )


def _box_glb(scale: float = 1.0) -> bytes:
    mesh = trimesh.creation.box(
        extents=(scale, scale * 0.6, scale * 0.8)
    )
    return bytes(trimesh.Scene(mesh).export(file_type="glb"))


def _png_bytes(color: tuple[int, int, int, int]) -> bytes:
    pixels = np.full((8, 8, 4), color, dtype=np.uint8)
    encoded, payload = cv2.imencode(".png", pixels)
    if not encoded:
        raise RuntimeError("Test PNG encoding failed.")
    return bytes(payload)


def _texture_variants(
    generation_index: int,
    *,
    glb_bytes: bytes | None = None,
) -> ObjectTextureVariants:
    return ObjectTextureVariants(
        glb_by_resolution={
            resolution: (
                glb_bytes
                if glb_bytes is not None
                else _box_glb(
                    1.0 + generation_index + resolution / 10_000.0
                )
            )
            for resolution in TEXTURE_RESOLUTIONS
        },
        texture_png_by_resolution={
            resolution: _png_bytes(
                (
                    25 + generation_index * 20,
                    resolution // 16 % 255,
                    135,
                    255,
                )
            )
            for resolution in TEXTURE_RESOLUTIONS
        },
        preview_rgba_by_resolution={
            resolution: np.full(
                (8, 8, 4),
                (25 + generation_index * 20, 70, 135, 255),
                dtype=np.uint8,
            )
            for resolution in TEXTURE_RESOLUTIONS
        },
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


def _asset_bytes(asset_directory: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in asset_directory.iterdir()
        if path.is_file()
    }


# ### Automatic generation integration tests ###
class UncheckedCameraGenerationPipelineTests(unittest.TestCase):
    def test_any_unchecked_camera_forces_geometry_then_purge_and_texture(self) -> None:
        enabled_camera_ids = tuple(
            camera_id
            for camera_id in ALL_CAMERA_IDS
            if camera_id not in {"neg_x", "bottom"}
        )
        request = _generation_request(
            enabled_camera_ids=enabled_camera_ids,
        )
        image_calls: list[dict[str, object]] = []

        def request_image(**kwargs: object) -> MeshyGenerationResult:
            image_calls.append(kwargs)
            return MeshyGenerationResult(
                task_id="geometry-task",
                glb_bytes=b"source geometry",
                name="Geometry",
            )

        with (
            patch(
                "housemaker.generation_workspace.request_image_to_3d_model",
                side_effect=request_image,
            ),
            patch(
                "housemaker.generation_workspace."
                "purge_faces_visible_from_unchecked_cameras_from_glb",
                return_value=_purge_result(),
            ) as purge,
            patch(
                "housemaker.generation_workspace.request_retextured_model",
                return_value=MeshyGenerationResult(
                    task_id="texture-task",
                    glb_bytes=b"textured geometry",
                    name="Textured",
                ),
            ) as retexture,
        ):
            result = MeshyImagePlanner().plan(request)

        self.assertEqual(len(image_calls), 1)
        self.assertFalse(image_calls[0]["should_texture"])
        self.assertEqual(purge.call_args.args, (b"source geometry",))
        self.assertEqual(
            purge.call_args.kwargs["unchecked_camera_ids"],
            ("neg_x", "bottom"),
        )
        self.assertIsInstance(
            purge.call_args.kwargs["options"],
            UncheckedCameraFacePurgeOptions,
        )
        self.assertEqual(
            retexture.call_args.kwargs["model_glb"],
            b"purged geometry",
        )
        self.assertFalse(retexture.call_args.kwargs["enable_original_uv"])
        self.assertIsInstance(result, StagedMeshyGenerationResult)
        self.assertTrue(result.camera_face_purge_applied)
        self.assertEqual(result.unchecked_camera_ids, ("neg_x", "bottom"))
        self.assertEqual(result.purge_original_face_count, 120)
        self.assertEqual(result.purge_retained_face_count, 96)
        self.assertEqual(result.purge_removed_face_count, 24)
        self.assertEqual(result.source_glb_bytes, b"source geometry")
        self.assertEqual(result.postprocessed_glb_bytes, b"purged geometry")

    def test_purge_is_first_before_removal_projection_and_retexture(self) -> None:
        request = _generation_request(
            enabled_camera_ids=("pos_x", "neg_x", "pos_y", "top"),
            unused_face_removal=True,
            project_camera_uvs=True,
        )
        call_order: list[str] = []
        fingerprint = CameraUvFingerprint(
            version=CAMERA_UV_FINGERPRINT_VERSION,
            sha256="a" * 64,
            face_count=72,
        )

        def request_geometry(**_kwargs: object) -> MeshyGenerationResult:
            call_order.append("geometry")
            return MeshyGenerationResult("geometry-task", b"source geometry")

        def purge_geometry(glb_bytes: bytes, **_kwargs: object) -> object:
            self.assertEqual(glb_bytes, b"source geometry")
            call_order.append("purge")
            return _purge_result()

        def remove_unused(glb_bytes: bytes, **_kwargs: object) -> object:
            self.assertEqual(glb_bytes, b"purged geometry")
            call_order.append("unused-removal")
            return SimpleNamespace(
                glb_bytes=b"visible geometry",
                original_face_count=96,
                retained_face_count=80,
                removed_face_count=16,
                protected_face_count=80,
            )

        def project_uvs(glb_bytes: bytes, **_kwargs: object) -> object:
            self.assertEqual(glb_bytes, b"visible geometry")
            call_order.append("camera-uv")
            return SimpleNamespace(
                glb_bytes=b"projected geometry",
                camera_face_counts={camera_id: 12 for camera_id in ALL_CAMERA_IDS},
                leftover_face_count=0,
                invisible_face_count=0,
                quality_fallback_face_count=0,
                conflict_fallback_face_count=0,
            )

        def retexture(**kwargs: object) -> MeshyGenerationResult:
            self.assertEqual(kwargs["model_glb"], b"projected geometry")
            call_order.append("retexture")
            return MeshyGenerationResult("texture-task", b"textured geometry")

        with (
            patch(
                "housemaker.generation_workspace.request_image_to_3d_model",
                side_effect=request_geometry,
            ),
            patch(
                "housemaker.generation_workspace."
                "purge_faces_visible_from_unchecked_cameras_from_glb",
                side_effect=purge_geometry,
            ) as purge,
            patch(
                "housemaker.generation_workspace.remove_unused_faces_from_glb",
                side_effect=remove_unused,
            ) as unused_removal,
            patch(
                "housemaker.generation_workspace."
                "project_uvs_from_camera_views_from_glb",
                side_effect=project_uvs,
            ) as projection,
            patch(
                "housemaker.generation_workspace.build_camera_uv_fingerprint",
                return_value=fingerprint,
            ),
            patch(
                "housemaker.generation_workspace."
                "validate_camera_uv_retexture_integrity"
            ),
            patch(
                "housemaker.generation_workspace.request_retextured_model",
                side_effect=retexture,
            ),
        ):
            result = MeshyImagePlanner().plan(request)

        self.assertEqual(
            call_order,
            ["geometry", "purge", "unused-removal", "camera-uv", "retexture"],
        )
        self.assertEqual(
            purge.call_args.kwargs["unchecked_camera_ids"],
            ("neg_y", "bottom"),
        )
        self.assertEqual(
            unused_removal.call_args.args,
            (b"purged geometry",),
        )
        self.assertEqual(projection.call_args.args, (b"visible geometry",))
        self.assertTrue(result.camera_face_purge_applied)
        self.assertTrue(result.unused_face_removal_applied)
        self.assertTrue(result.camera_uv_projection_applied)
        self.assertEqual(result.postprocessed_glb_bytes, b"projected geometry")

    def test_all_checked_without_other_processing_keeps_direct_generation(self) -> None:
        request = _generation_request(enabled_camera_ids=ALL_CAMERA_IDS)

        with (
            patch(
                "housemaker.generation_workspace.request_image_to_3d_model",
                return_value=MeshyGenerationResult(
                    task_id="direct-task",
                    glb_bytes=b"direct textured glb",
                    name="Direct",
                ),
            ) as image_request,
            patch(
                "housemaker.generation_workspace."
                "purge_faces_visible_from_unchecked_cameras_from_glb"
            ) as purge,
            patch(
                "housemaker.generation_workspace.request_retextured_model"
            ) as retexture,
        ):
            result = MeshyImagePlanner().plan(request)

        self.assertNotIn("should_texture", image_request.call_args.kwargs)
        purge.assert_not_called()
        retexture.assert_not_called()
        self.assertNotIsInstance(result, StagedMeshyGenerationResult)
        self.assertEqual(result.task_id, "direct-task")


# ### Manual request and worker tests ###
class UncheckedCameraFacePurgeWorkerTests(unittest.TestCase):
    def test_request_owns_model_bytes_and_canonicalizes_camera_ids(self) -> None:
        source = bytearray(b"owned source")

        request = UncheckedCameraFacePurgeRequest(
            object_id="  chair  ",
            model_glb=source,
            unchecked_camera_ids=("bottom", "neg_x", "bottom"),
        )
        source[:] = b"changedxxxxx"

        self.assertEqual(request.object_id, "chair")
        self.assertEqual(request.model_glb, b"owned source")
        self.assertEqual(request.unchecked_camera_ids, ("neg_x", "bottom"))
        with self.assertRaisesRegex(ValueError, "Uncheck at least one"):
            UncheckedCameraFacePurgeRequest(
                object_id="chair",
                model_glb=b"model",
                unchecked_camera_ids=(),
            )

    def test_worker_routes_the_exact_snapshot_and_returns_counts(self) -> None:
        request = UncheckedCameraFacePurgeRequest(
            object_id="chair",
            model_glb=b"canonical 2048 glb",
            unchecked_camera_ids=("neg_y", "bottom"),
        )
        model = import_generated_glb(_box_glb())
        result = UncheckedCameraFacePurgeResult(
            model=model,
            unchecked_camera_ids=request.unchecked_camera_ids,
            original_face_count=12,
            retained_face_count=8,
            removed_face_count=4,
        )
        worker = UncheckedCameraFacePurgeWorker(request)
        succeeded = QSignalSpy(worker.succeeded)
        failed = QSignalSpy(worker.failed)
        finished = QSignalSpy(worker.finished)

        with patch(
            "housemaker.generation_workspace."
            "purge_faces_visible_from_unchecked_cameras_from_glb",
            return_value=result,
        ) as purge:
            worker.run()

        self.assertEqual(succeeded.count(), 1)
        self.assertEqual(failed.count(), 0)
        self.assertEqual(finished.count(), 1)
        outcome = succeeded.at(0)[0]
        self.assertIs(outcome.request, request)
        self.assertIs(outcome.result, result)
        self.assertEqual(purge.call_args.args, (b"canonical 2048 glb",))
        self.assertEqual(
            purge.call_args.kwargs["unchecked_camera_ids"],
            ("neg_y", "bottom"),
        )
        self.assertTrue(callable(purge.call_args.kwargs["cancel_requested"]))

    def test_pre_cancel_is_silent_and_never_reports_success(self) -> None:
        request = UncheckedCameraFacePurgeRequest(
            object_id="chair",
            model_glb=b"canonical 2048 glb",
            unchecked_camera_ids=("bottom",),
        )
        worker = UncheckedCameraFacePurgeWorker(request)
        worker.cancel()
        succeeded = QSignalSpy(worker.succeeded)
        failed = QSignalSpy(worker.failed)
        finished = QSignalSpy(worker.finished)

        def cancelled_core(_glb_bytes: bytes, **kwargs: object) -> object:
            cancel_requested = kwargs["cancel_requested"]
            self.assertTrue(cancel_requested())  # type: ignore[operator]
            raise UnusedFaceRemovalCancelled("cancelled")

        with patch(
            "housemaker.generation_workspace."
            "purge_faces_visible_from_unchecked_cameras_from_glb",
            side_effect=cancelled_core,
        ):
            worker.run()

        self.assertEqual(succeeded.count(), 0)
        self.assertEqual(failed.count(), 0)
        self.assertEqual(finished.count(), 1)


# ### Manual purge workspace tests ###
class UncheckedCameraFacePurgeWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.asset_directory = Path(self.temporary_directory.name) / "generated"
        self.asset_directory.mkdir(parents=True)
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

    def _seed_object(
        self,
        *,
        generation_index: int = 0,
    ) -> tuple[GeneratedObjectRecord, ObjectTextureVariants]:
        variants = _texture_variants(generation_index)
        model = _model_with_variants(variants)
        result = MeshyGenerationResult(
            task_id="original-task",
            glb_bytes=variants.glb_by_resolution[1024],
            name="Chair",
        )
        self.workspace._handle_generation_succeeded(result, model)
        return self.workspace.get_data().generated_objects[0], variants

    def _uncheck(self, *camera_ids: str) -> None:
        checkboxes = self.workspace.object_3d_panel.unused_face_camera_checkboxes
        for camera_id in camera_ids:
            checkboxes[camera_id].setChecked(False)
        _qt_application.processEvents()

    def _wait_until_idle(self, timeout_seconds: float = 8.0) -> None:
        deadline = time.monotonic() + timeout_seconds
        while self.workspace.is_generating and time.monotonic() < deadline:
            _qt_application.processEvents()
            QTest.qWait(10)
        _qt_application.processEvents()
        self.assertFalse(
            self.workspace.is_generating,
            "Face-purge worker did not finish.",
        )

    def test_button_order_and_eligibility_track_selection_cameras_and_busy(
        self,
    ) -> None:
        self.assertEqual(self.workspace.purge_faces_button.text(), "Purge faces")
        self.assertEqual(
            self.workspace.purge_faces_button.objectName(),
            "purge_faces_button",
        )
        self.assertLess(
            self.workspace.generate_geometry_button.geometry().left(),
            self.workspace.generate_texture_button.geometry().left(),
        )
        self.assertLess(
            self.workspace.generate_texture_button.geometry().left(),
            self.workspace.purge_faces_button.geometry().left(),
        )
        self.assertLess(
            self.workspace.purge_faces_button.geometry().left(),
            self.workspace.undo_object_change_button.geometry().left(),
        )
        self.assertLess(
            self.workspace.generate_button.geometry().left() + 30,
            self.workspace.generate_geometry_button.geometry().left(),
        )
        self.assertTrue(
            self.workspace.object_3d_panel.unused_face_camera_controls.isEnabled()
        )
        self.assertFalse(self.workspace.purge_faces_button.isEnabled())

        self._seed_object()
        self.assertFalse(self.workspace.purge_faces_button.isEnabled())
        self._uncheck("neg_x")
        self.assertTrue(self.workspace.purge_faces_button.isEnabled())

        self.workspace.generated_objects_list.setCurrentRow(-1)
        self.assertFalse(self.workspace.purge_faces_button.isEnabled())
        self.workspace.generated_objects_list.setCurrentRow(0)
        self.assertTrue(self.workspace.purge_faces_button.isEnabled())

        self.workspace._generation_thread = object()  # type: ignore[assignment]
        try:
            self.workspace._sync_controls()
            self.assertFalse(self.workspace.purge_faces_button.isEnabled())
        finally:
            self.workspace._generation_thread = None
            self.workspace._sync_controls()
        self.assertTrue(self.workspace.purge_faces_button.isEnabled())

    def test_request_snapshots_exact_unchecked_ids_and_canonical_2048_glb(
        self,
    ) -> None:
        record, variants = self._seed_object()
        self.workspace.texture_view.select_atlas(
            f"{record.object_id}:resolution:512"
        )
        self._uncheck("neg_x", "bottom")

        request = self.workspace._build_unchecked_camera_face_purge_request()

        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.object_id, record.object_id)
        self.assertEqual(request.unchecked_camera_ids, ("neg_x", "bottom"))
        self.assertEqual(request.model_glb, variants.glb_by_resolution[2048])
        self.workspace.object_3d_panel.unused_face_camera_checkboxes[
            "pos_y"
        ].setChecked(False)
        self.assertEqual(request.unchecked_camera_ids, ("neg_x", "bottom"))

    def test_success_atomically_replaces_geometry_variants_and_emits_signal(
        self,
    ) -> None:
        record, original_variants = self._seed_object()
        self.workspace.texture_view.select_atlas(
            f"{record.object_id}:resolution:512"
        )
        current_record = self.workspace.get_data().generated_objects[0]
        marked_record = replace(
            current_record,
            pipeline={
                **current_record.pipeline,
                TEXTURE_INPAINT_STROKES_PIPELINE_KEY: [
                    {
                        "mode": "paint",
                        "radius_normalized": 0.04,
                        "points": [{"u": 0.25, "v": 0.75}],
                    }
                ],
                "retained_provenance": "keep",
            },
        )
        self.workspace._data.generated_objects[0] = marked_record
        self.workspace._refresh_generated_objects_list(marked_record.object_id)
        self._uncheck("neg_x", "bottom")
        old_paths = _record_variant_paths(marked_record)

        simplified_mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
        simplified_glb = bytes(
            trimesh.Scene(simplified_mesh).export(file_type="glb")
        )
        purged_model = import_generated_glb(simplified_glb)
        purge_result = UncheckedCameraFacePurgeResult(
            model=purged_model,
            unchecked_camera_ids=("neg_x", "bottom"),
            original_face_count=120,
            retained_face_count=80,
            removed_face_count=40,
        )
        replacement_variants = _texture_variants(
            1,
            glb_bytes=simplified_glb,
        )
        changed = QSignalSpy(self.workspace.data_changed)
        completed = QSignalSpy(self.workspace.face_purge_completed)

        with (
            patch(
                "housemaker.generation_workspace."
                "purge_faces_visible_from_unchecked_cameras_from_glb",
                return_value=purge_result,
            ) as purge,
            patch(
                "housemaker.generation_workspace.build_object_texture_variants",
                return_value=replacement_variants,
            ),
        ):
            self.assertTrue(self.workspace.purge_selected_object_faces())
            self._wait_until_idle()

        self.assertEqual(
            purge.call_args.args,
            (original_variants.glb_by_resolution[2048],),
        )
        self.assertEqual(
            purge.call_args.kwargs["unchecked_camera_ids"],
            ("neg_x", "bottom"),
        )
        replacement = self.workspace.get_data().generated_objects[0]
        new_paths = _record_variant_paths(replacement)
        self.assertEqual(replacement.object_id, marked_record.object_id)
        self.assertEqual(replacement.object_name, marked_record.object_name)
        self.assertEqual(replacement.frame_index, marked_record.frame_index)
        self.assertEqual(
            replacement.provider_task_id,
            marked_record.provider_task_id,
        )
        self.assertEqual(replacement.pipeline["retained_provenance"], "keep")
        self.assertNotIn(
            TEXTURE_INPAINT_STROKES_PIPELINE_KEY,
            replacement.pipeline,
        )
        self.assertEqual(replacement.pipeline["selected_texture_resolution"], 512)
        self.assertEqual(replacement.pipeline["manual_face_purge_count"], 1)
        self.assertEqual(
            replacement.pipeline["unchecked_camera_ids"],
            ["neg_x", "bottom"],
        )
        self.assertEqual(
            replacement.pipeline["latest_face_purge_removed_face_count"],
            40,
        )
        self.assertTrue(new_paths.isdisjoint(old_paths))
        self.assertTrue(
            all(path.startswith("purged-") for path in new_paths)
        )
        self.assertTrue(
            all(self.asset_directory.joinpath(path).is_file() for path in old_paths)
        )
        self.assertTrue(
            all(self.asset_directory.joinpath(path).is_file() for path in new_paths)
        )
        self.assertEqual(changed.count(), 1)
        self.assertEqual(completed.count(), 1)
        self.assertEqual(completed.at(0)[0].object_id, marked_record.object_id)
        self.assertEqual(
            completed.at(0)[1].glb_bytes,
            replacement_variants.glb_by_resolution[512],
        )
        self.assertEqual(
            self.workspace.result_view.model.glb_bytes,
            replacement_variants.glb_by_resolution[512],
        )
        self.assertIn("80 triangles", self.workspace.model_statistics_label.text())

    def test_untextured_geometry_purge_can_be_undone(self) -> None:
        original_glb = _box_glb(1.0)
        original_result = MeshyGenerationResult(
            "geometry-task",
            original_glb,
            "Untextured table",
        )
        self.workspace._handle_generation_succeeded(
            original_result,
            import_generated_glb(original_glb),
        )
        original = self.workspace.get_data().generated_objects[0]
        original_path = self.asset_directory / str(original.asset_path)
        self._uncheck("bottom")

        purged_glb = _box_glb(2.0)
        purge_result = UncheckedCameraFacePurgeResult(
            model=import_generated_glb(purged_glb),
            unchecked_camera_ids=("bottom",),
            original_face_count=12,
            retained_face_count=10,
            removed_face_count=2,
        )
        changed = QSignalSpy(self.workspace.generated_object_changed)
        with patch(
            "housemaker.generation_workspace."
            "purge_faces_visible_from_unchecked_cameras_from_glb",
            return_value=purge_result,
        ):
            self.assertTrue(self.workspace.purge_selected_object_faces())
            self._wait_until_idle()

        purged = self.workspace.get_data().generated_objects[0]
        self.assertNotIn("texture_variants", purged.pipeline)
        self.assertTrue(str(purged.asset_path).startswith("purged-"))
        purged_path = self.asset_directory / str(purged.asset_path)
        self.assertTrue(original_path.is_file())
        self.assertTrue(purged_path.is_file())
        self.assertTrue(self.workspace.undo_object_change_button.isEnabled())

        self.assertTrue(self.workspace.undo_selected_object_change())

        restored = self.workspace.get_data().generated_objects[0]
        self.assertEqual(restored, original)
        self.assertEqual(
            self.workspace.result_view.model.glb_bytes,
            original_glb,
        )
        self.assertTrue(original_path.is_file())
        self.assertFalse(purged_path.exists())
        self.assertFalse(self.workspace.undo_object_change_button.isEnabled())
        self.assertEqual(changed.count(), 2)

    def test_core_failure_keeps_record_files_cache_and_display(self) -> None:
        record, _variants = self._seed_object()
        self._uncheck("bottom")
        record = self.workspace.get_data().generated_objects[0]
        before_assets = _asset_bytes(self.asset_directory)
        old_model = self.workspace.result_view.model
        changed = QSignalSpy(self.workspace.data_changed)
        completed = QSignalSpy(self.workspace.face_purge_completed)

        with (
            patch(
                "housemaker.generation_workspace."
                "purge_faces_visible_from_unchecked_cameras_from_glb",
                side_effect=RuntimeError("injected purge failure"),
            ),
            patch.object(QMessageBox, "warning") as warning,
        ):
            self.assertTrue(self.workspace.purge_selected_object_faces())
            self._wait_until_idle()

        self.assertEqual(self.workspace.get_data().generated_objects[0], record)
        self.assertEqual(_asset_bytes(self.asset_directory), before_assets)
        self.assertIs(self.workspace.result_view.model, old_model)
        self.assertEqual(changed.count(), 0)
        self.assertEqual(completed.count(), 0)
        warning.assert_called_once()
        self.assertIn("injected purge failure", self.workspace.status_label.text())

    def test_partial_persistence_failure_rolls_back_new_files(self) -> None:
        record, _variants = self._seed_object()
        self._uncheck("bottom")
        record = self.workspace.get_data().generated_objects[0]
        before_assets = _asset_bytes(self.asset_directory)
        old_model = self.workspace.result_view.model
        purged_glb = _box_glb(2.0)
        purge_result = UncheckedCameraFacePurgeResult(
            model=import_generated_glb(purged_glb),
            unchecked_camera_ids=("bottom",),
            original_face_count=12,
            retained_face_count=10,
            removed_face_count=2,
        )
        replacement_variants = _texture_variants(3, glb_bytes=purged_glb)
        original_persist = self.workspace._persist_meshy_named_asset
        write_count = 0

        def persist_or_fail(file_name: str, payload: bytes) -> str:
            nonlocal write_count
            write_count += 1
            if write_count == 4:
                raise OSError("injected fourth write failure")
            return original_persist(file_name, payload)

        with (
            patch(
                "housemaker.generation_workspace."
                "purge_faces_visible_from_unchecked_cameras_from_glb",
                return_value=purge_result,
            ),
            patch(
                "housemaker.generation_workspace.build_object_texture_variants",
                return_value=replacement_variants,
            ),
            patch.object(
                self.workspace,
                "_persist_meshy_named_asset",
                side_effect=persist_or_fail,
            ),
            patch.object(QMessageBox, "warning") as warning,
        ):
            self.assertTrue(self.workspace.purge_selected_object_faces())
            self._wait_until_idle()

        self.assertEqual(self.workspace.get_data().generated_objects[0], record)
        self.assertEqual(_asset_bytes(self.asset_directory), before_assets)
        self.assertIs(self.workspace.result_view.model, old_model)
        warning.assert_called_once()

    def test_shutdown_cancellation_keeps_existing_transaction(self) -> None:
        record, _variants = self._seed_object()
        self._uncheck("bottom")
        record = self.workspace.get_data().generated_objects[0]
        before_assets = _asset_bytes(self.asset_directory)
        old_model = self.workspace.result_view.model
        started = threading.Event()
        release = threading.Event()
        completed = QSignalSpy(self.workspace.face_purge_completed)

        def blocking_purge(_glb_bytes: bytes, **kwargs: object) -> object:
            started.set()
            if not release.wait(timeout=5.0):
                raise RuntimeError("Blocking face purge timed out.")
            cancel_requested = kwargs["cancel_requested"]
            if cancel_requested():  # type: ignore[operator]
                raise UnusedFaceRemovalCancelled("cancelled")
            raise AssertionError("Face purge was not cancelled.")

        with patch(
            "housemaker.generation_workspace."
            "purge_faces_visible_from_unchecked_cameras_from_glb",
            side_effect=blocking_purge,
        ):
            self.assertTrue(self.workspace.purge_selected_object_faces())
            deadline = time.monotonic() + 3.0
            while not started.is_set() and time.monotonic() < deadline:
                _qt_application.processEvents()
                QTest.qWait(10)
            self.assertTrue(started.is_set())
            active_thread = self.workspace._generation_thread
            self.workspace.shutdown()
            release.set()
            if active_thread is not None:
                active_thread.wait(2000)
            _qt_application.processEvents()

        self.assertEqual(self.workspace.get_data().generated_objects[0], record)
        self.assertEqual(_asset_bytes(self.asset_directory), before_assets)
        self.assertIs(self.workspace.result_view.model, old_model)
        self.assertEqual(completed.count(), 0)


# ### Test runner ###
if __name__ == "__main__":
    unittest.main()
