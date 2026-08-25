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
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import trimesh
from PIL import Image
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from housemaker.camera_uv_integrity import (
    CAMERA_UV_FINGERPRINT_VERSION,
    CAMERA_UV_PROJECTION_VERSION,
    CameraUvIntegrityError,
)
from housemaker.generation_state import (
    MASK_MODE_ERASE,
    MASK_MODE_PAINT,
    GeneratedObjectRecord,
    GenerationData,
    MaskPoint,
    MaskStroke,
)
from housemaker.generation_views import VideoInpaintView, rasterize_mask_strokes
from housemaker.generation_workspace import (
    GENERATION_BACKEND_MESHY,
    GenerationRequest,
    GenerationWorker,
    GenerationWorkspace,
    MeshyImagePlanner,
    MeshyModelExecutor,
    StagedMeshyGenerationResult,
    _GenerationCancelled,
    _format_staged_generation_status,
    _format_model_statistics,
    _collect_model_uv_triangles,
    _staged_generation_mode,
)
from housemaker.glb import GeneratedModel, import_generated_glb
from housemaker.meshy_generation import MeshyGenerationResult
from housemaker.settings_widget import GenerationServiceSettings
from housemaker.unused_face_removal import (
    ALL_CAMERA_IDS,
    UnusedFaceRemovalCancelled,
    UnusedFaceRemovalResult,
)
from housemaker.video_source import VideoMetadata
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])


# ### Fixture helpers ###
def _test_model() -> GeneratedModel:
    mesh = trimesh.creation.box(extents=(1.0, 0.5, 0.75))
    scene = trimesh.Scene(mesh)
    return GeneratedModel(
        mesh=mesh,
        scene=scene,
        glb_bytes=scene.export(file_type="glb"),
    )


def _test_textured_model(color: tuple[int, int, int, int]) -> GeneratedModel:
    mesh = trimesh.creation.box(extents=(1.0, 0.5, 0.75))
    texture = Image.new("RGBA", (8, 6), color)
    mesh.visual = TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2), dtype=float),
        material=PBRMaterial(baseColorTexture=texture),
    )
    scene = trimesh.Scene(mesh)
    return GeneratedModel(
        mesh=mesh,
        scene=scene,
        glb_bytes=scene.export(file_type="glb"),
    )


def _test_meshy_result(name: str = "Test chair") -> MeshyGenerationResult:
    return MeshyGenerationResult(
        task_id="task-test-chair",
        glb_bytes=_test_model().glb_bytes,
        name=name,
    )


def _test_staged_meshy_result(
    final_model: GeneratedModel | None = None,
) -> StagedMeshyGenerationResult:
    final_model = final_model or _test_model()
    return StagedMeshyGenerationResult(
        task_id="texture-task-123",
        glb_bytes=final_model.glb_bytes,
        name="Post-processed chair",
        geometry_task_id="geometry-task-123",
        source_glb_bytes=b"source geometry glb",
        postprocessed_glb_bytes=b"post-processed geometry glb",
        original_face_count=120,
        retained_face_count=80,
        removed_face_count=40,
        protected_face_count=80,
        enabled_camera_ids=("pos_x", "top"),
    )


def _test_camera_uv_glb(
    *,
    redundant_vertices: bool = False,
    mutate_uv: bool = False,
) -> bytes:
    """Build two equivalent ordered UV triangles with optional reindexing."""

    shared_vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        dtype=float,
    )
    shared_uv = np.asarray(
        (
            (0.1, 0.1),
            (0.9, 0.1),
            (0.9, 0.9),
            (0.1, 0.9),
        ),
        dtype=float,
    )
    if redundant_vertices:
        vertices = shared_vertices[[2, 0, 1, 3, 2, 0]]
        uv = shared_uv[[2, 0, 1, 3, 2, 0]]
        faces = np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64)
    else:
        vertices = shared_vertices
        uv = shared_uv.copy()
        faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    if mutate_uv:
        uv[-1] = (0.25, 0.25)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual = TextureVisuals(uv=uv, material=PBRMaterial())
    return bytes(trimesh.Scene(mesh).export(file_type="glb"))


def _test_stroke(
    mode: str = MASK_MODE_PAINT,
    x: float = 0.5,
    y: float = 0.5,
) -> MaskStroke:
    return MaskStroke(
        mode=mode,
        radius_normalized=0.1,
        points=(MaskPoint(x, y),),
    )


def _write_test_video(path: Path, frame_count: int = 3) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (80, 60),
    )
    if not writer.isOpened():
        raise unittest.SkipTest("MJPG writer is unavailable")
    try:
        for frame_index in range(frame_count):
            writer.write(
                np.full(
                    (60, 80, 3),
                    (frame_index * 40, 40, 160),
                    dtype=np.uint8,
                )
            )
    finally:
        writer.release()


class _FakeMeshyPlanner:
    def __init__(self, result: MeshyGenerationResult | None = None) -> None:
        self.result = result or _test_meshy_result()
        self.requests: list[GenerationRequest] = []

    def plan(self, request: GenerationRequest) -> MeshyGenerationResult:
        self.requests.append(request)
        return self.result


class _FakeMeshyExecutor:
    def __init__(self, model: GeneratedModel | None = None) -> None:
        self.model = model or _test_model()
        self.results: list[MeshyGenerationResult] = []

    def execute(self, result: MeshyGenerationResult) -> GeneratedModel:
        self.results.append(result)
        return self.model


class _BlockingMeshyPlanner:
    """Meshy fixture that simulates an in-flight provider request."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def plan(self, _request: GenerationRequest) -> MeshyGenerationResult:
        self.started.set()
        if not self.release.wait(timeout=5.0):
            raise RuntimeError("Blocking Meshy planner test timed out.")
        return _test_meshy_result("Late chair")


# ### State tests ###
class GenerationStateTests(unittest.TestCase):
    def test_generation_data_round_trip_preserves_meshy_records(self) -> None:
        metadata = VideoMetadata(
            path="house.avi",
            frame_count=10,
            fps=24.0,
            width=640,
            height=480,
        )
        original = GenerationData(
            video_metadata=metadata,
            current_frame_index=4,
            frame_strokes={4: [_test_stroke()]},
            generated_objects=[
                GeneratedObjectRecord(
                    object_id="chair-1",
                    frame_index=4,
                    object_name="Chair",
                    pipeline={},
                    provider=GENERATION_BACKEND_MESHY,
                    provider_task_id="task-chair-1",
                    asset_path="chair-1.glb",
                )
            ],
        )

        loaded = GenerationData.from_dict(original.to_dict())

        self.assertEqual(loaded, original)
        self.assertIsNot(loaded.frame_strokes, original.frame_strokes)
        self.assertNotIn("api", str(original.to_dict()).lower())

    def test_legacy_simplification_pipeline_round_trip_is_lossless(self) -> None:
        legacy_pipeline = {
            "schema_version": 3,
            "mode": "unused_face_removal_and_rectangular_face_cleanup",
            "source_asset_path": "chair.source.glb",
            "postprocessed_asset_path": "chair.postprocessed.glb",
            "camera_validated_simplification_applied": True,
            "simplification_input_face_count": 120,
            "simplified_face_count": 74,
            "simplification_removed_face_count": 46,
            "simplification_pixel_tolerance_percent": 0.25,
            "simplification_detected_rectangle_count": 5,
            "simplification_accepted_rectangle_count": 4,
            "simplification_rejected_rectangle_count": 1,
        }
        original_record = GeneratedObjectRecord(
            object_id="legacy-simplified-chair",
            frame_index=3,
            object_name="Legacy simplified chair",
            pipeline=legacy_pipeline,
            provider=GENERATION_BACKEND_MESHY,
            provider_task_id="legacy-texture-task",
            asset_path="chair.glb",
        )
        original = GenerationData(generated_objects=[original_record])

        restored = GenerationData.from_dict(original.to_dict())

        self.assertEqual(restored.generated_objects, [original_record])
        self.assertEqual(
            restored.generated_objects[0].pipeline,
            legacy_pipeline,
        )
        self.assertIsNot(
            restored.generated_objects[0].pipeline,
            original_record.pipeline,
        )

    def test_legacy_procedural_records_are_ignored_on_load(self) -> None:
        loaded = GenerationData.from_dict(
            {
                "generated_objects": [
                    {
                        "object_id": "legacy-chair",
                        "frame_index": 0,
                        "object_name": "Legacy chair",
                        "pipeline": {"schema_version": 1},
                        "provider": "procedural",
                    }
                ]
            }
        )

        self.assertEqual(loaded.generated_objects, [])

    def test_invalid_normalized_mask_data_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MaskPoint(1.1, 0.5)
        with self.assertRaises(ValueError):
            MaskStroke("unknown", 0.1, (MaskPoint(0.5, 0.5),))
        with self.assertRaises(ValueError):
            MaskStroke(
                MASK_MODE_PAINT,
                0.1,
                (MaskPoint(0.4, 0.5), MaskPoint(0.6, 0.5)),
                is_fill=True,
            )

    def test_enclosed_fill_action_round_trips_with_frame_strokes(self) -> None:
        fill = MaskStroke(
            MASK_MODE_PAINT,
            0.000001,
            (MaskPoint(0.5, 0.5),),
            is_fill=True,
        )
        original = GenerationData(frame_strokes={2: [fill]})

        restored = GenerationData.from_dict(original.to_dict())

        self.assertEqual(restored.frame_strokes, {2: [fill]})


# ### Mask-view tests ###
class GenerationMaskViewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.view = VideoInpaintView()
        self.view.resize(400, 300)
        self.view.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.view.close()
        _qt_application.processEvents()

    def test_paint_erase_undo_clear_and_object_crop(self) -> None:
        frame = np.full((100, 200, 3), 120, dtype=np.uint8)
        self.view.set_frame(frame)
        changed_spy = QSignalSpy(self.view.strokes_changed)

        QTest.mousePress(
            self.view,
            Qt.MouseButton.LeftButton,
            pos=QPoint(190, 150),
        )
        QTest.mouseMove(self.view, QPoint(215, 150), delay=1)
        QTest.mouseRelease(
            self.view,
            Qt.MouseButton.LeftButton,
            pos=QPoint(215, 150),
        )
        _qt_application.processEvents()

        self.assertEqual(len(self.view.get_strokes()), 1)
        self.assertTrue(self.view.has_selection())
        object_crop = self.view.build_selected_object_crop()
        self.assertEqual(object_crop.shape[2], 4)
        transparent_pixels = object_crop[:, :, 3] == 0
        self.assertTrue(np.any(transparent_pixels))
        self.assertTrue(np.all(object_crop[transparent_pixels, :3] == 0))
        self.assertEqual(changed_spy.count(), 1)

        self.view.undo_last_stroke()
        self.assertFalse(self.view.has_selection())

        self.view.set_strokes([_test_stroke()])
        self.view.clear_mask()
        self.assertFalse(self.view.has_selection())

    def test_normalized_strokes_rasterize_consistently_at_new_resolution(self) -> None:
        paint = MaskStroke(
            mode=MASK_MODE_PAINT,
            radius_normalized=0.12,
            points=(MaskPoint(0.2, 0.5), MaskPoint(0.8, 0.5)),
        )
        erase = _test_stroke(MASK_MODE_ERASE)

        small_mask = rasterize_mask_strokes((100, 50), [paint, erase])
        large_mask = rasterize_mask_strokes((200, 100), [paint, erase])

        self.assertEqual(small_mask.shape, (50, 100))
        self.assertEqual(large_mask.shape, (100, 200))
        self.assertGreater(np.count_nonzero(small_mask), 0)
        self.assertGreater(np.count_nonzero(large_mask), 0)
        self.assertEqual(small_mask[25, 50], 0)
        self.assertEqual(large_mask[50, 100], 0)

    def test_right_click_fills_closed_outline_and_undoes_as_one_action(
        self,
    ) -> None:
        frame = np.full((100, 200, 3), 120, dtype=np.uint8)
        outline = MaskStroke(
            MASK_MODE_PAINT,
            0.03,
            (
                MaskPoint(0.25, 0.25),
                MaskPoint(0.75, 0.25),
                MaskPoint(0.75, 0.75),
                MaskPoint(0.25, 0.75),
                MaskPoint(0.25, 0.25),
            ),
        )
        self.view.set_frame(frame, [outline])
        changed_spy = QSignalSpy(self.view.strokes_changed)

        QTest.mouseClick(
            self.view,
            Qt.MouseButton.RightButton,
            pos=QPoint(200, 150),
        )
        _qt_application.processEvents()

        strokes = self.view.get_strokes()
        self.assertEqual(len(strokes), 2)
        self.assertTrue(strokes[-1].is_fill)
        self.assertEqual(changed_spy.count(), 1)
        mask = self.view.get_mask()
        self.assertEqual(mask[50, 100], 255)
        self.assertEqual(mask[5, 5], 0)

        self.view.undo_last_stroke()

        self.assertEqual(self.view.get_mask()[50, 100], 0)

    def test_right_click_does_not_fill_an_open_or_unbounded_region(self) -> None:
        frame = np.full((100, 200, 3), 120, dtype=np.uint8)
        open_outline = MaskStroke(
            MASK_MODE_PAINT,
            0.03,
            (
                MaskPoint(0.25, 0.25),
                MaskPoint(0.75, 0.25),
                MaskPoint(0.75, 0.75),
            ),
        )
        self.view.set_frame(frame, [open_outline])
        changed_spy = QSignalSpy(self.view.strokes_changed)

        QTest.mouseClick(
            self.view,
            Qt.MouseButton.RightButton,
            pos=QPoint(200, 150),
        )
        _qt_application.processEvents()

        self.assertEqual(len(self.view.get_strokes()), 1)
        self.assertEqual(changed_spy.count(), 0)
        self.assertEqual(self.view.get_mask()[50, 100], 0)


# ### Meshy-adapter tests ###
class MeshyGenerationAdapterTests(unittest.TestCase):
    def test_meshy_planner_sends_the_selected_png_and_meshy_settings(self) -> None:
        selected = np.zeros((7, 11, 4), dtype=np.uint8)
        selected[1:6, 2:9] = (20, 80, 190, 255)
        request = GenerationRequest(
            frame_index=3,
            selected_object_bgra=selected,
            settings=GenerationServiceSettings(
                meshy_api_key="meshy-test-key",
                meshy_target_polycount=7_200,
            ),
        )
        captured: dict[str, object] = {}

        def fake_meshy_request(**kwargs: object) -> MeshyGenerationResult:
            captured.update(kwargs)
            return _test_meshy_result()

        with (
            patch(
                "housemaker.generation_workspace.request_image_to_3d_model",
                side_effect=fake_meshy_request,
            ),
            patch(
                "housemaker.generation_workspace.remove_unused_faces_from_glb"
            ) as removal_mock,
            patch(
                "housemaker.generation_workspace.request_retextured_model"
            ) as retexture_mock,
        ):
            result = MeshyImagePlanner().plan(request)

        self.assertEqual(result.task_id, "task-test-chair")
        self.assertEqual(captured["api_key"], "meshy-test-key")
        self.assertEqual(captured["target_polycount"], 7_200)
        self.assertNotIn("should_texture", captured)
        removal_mock.assert_not_called()
        retexture_mock.assert_not_called()
        encoded_png = captured["image_png"]
        self.assertIsInstance(encoded_png, bytes)
        decoded = cv2.imdecode(
            np.frombuffer(encoded_png, dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
        self.assertEqual(decoded.shape, selected.shape)
        np.testing.assert_array_equal(decoded, selected)

    def test_staged_planner_generates_geometry_removes_faces_then_textures(
        self,
    ) -> None:
        selected = np.zeros((9, 13, 4), dtype=np.uint8)
        selected[2:8, 3:11] = (30, 90, 170, 255)
        selected_camera_ids = ("pos_x", "neg_y", "top")
        request = GenerationRequest(
            frame_index=4,
            selected_object_bgra=selected,
            settings=GenerationServiceSettings(
                meshy_api_key="meshy-staged-key",
                meshy_target_polycount=5_400,
                unused_face_removal=True,
            ),
            enabled_camera_ids=selected_camera_ids,
        )
        geometry_model = _test_model()
        processed_model = _test_model()
        final_model = _test_model()
        geometry_result = MeshyGenerationResult(
            task_id="geometry-task",
            glb_bytes=geometry_model.glb_bytes,
            name="Geometry",
        )
        textured_result = MeshyGenerationResult(
            task_id="texture-task",
            glb_bytes=final_model.glb_bytes,
            name="Textured chair",
        )
        removal_result = UnusedFaceRemovalResult(
            model=processed_model,
            enabled_camera_ids=selected_camera_ids,
            original_face_count=12,
            retained_face_count=9,
            removed_face_count=3,
            protected_face_count=9,
        )
        image_calls: list[dict[str, object]] = []
        purge_calls: list[tuple[bytes, dict[str, object]]] = []
        removal_calls: list[tuple[bytes, dict[str, object]]] = []
        texture_calls: list[dict[str, object]] = []
        progress_messages: list[str] = []

        def fake_image_request(**kwargs: object) -> MeshyGenerationResult:
            image_calls.append(kwargs)
            return geometry_result

        def fake_removal(
            glb_bytes: bytes,
            **kwargs: object,
        ) -> UnusedFaceRemovalResult:
            removal_calls.append((glb_bytes, kwargs))
            return removal_result

        def fake_purge(glb_bytes: bytes, **kwargs: object) -> object:
            purge_calls.append((glb_bytes, kwargs))
            return SimpleNamespace(
                glb_bytes=geometry_model.glb_bytes,
                original_face_count=12,
                retained_face_count=12,
                removed_face_count=0,
            )

        def fake_retexture_request(**kwargs: object) -> MeshyGenerationResult:
            texture_calls.append(kwargs)
            return textured_result

        with (
            patch(
                "housemaker.generation_workspace.request_image_to_3d_model",
                side_effect=fake_image_request,
            ),
            patch(
                "housemaker.generation_workspace."
                "purge_faces_visible_from_unchecked_cameras_from_glb",
                side_effect=fake_purge,
            ),
            patch(
                "housemaker.generation_workspace.remove_unused_faces_from_glb",
                side_effect=fake_removal,
            ),
            patch(
                "housemaker.generation_workspace.request_retextured_model",
                side_effect=fake_retexture_request,
            ),
        ):
            result = MeshyImagePlanner().plan(
                request,
                progress_callback=progress_messages.append,
                cancel_event=threading.Event(),
            )

        self.assertEqual(len(image_calls), 1)
        self.assertFalse(image_calls[0]["should_texture"])
        self.assertEqual(image_calls[0]["target_polycount"], 5_400)
        self.assertEqual(purge_calls[0][0], geometry_model.glb_bytes)
        self.assertEqual(
            purge_calls[0][1]["unchecked_camera_ids"],
            tuple(
                camera_id
                for camera_id in ALL_CAMERA_IDS
                if camera_id not in selected_camera_ids
            ),
        )
        self.assertEqual(removal_calls[0][0], geometry_model.glb_bytes)
        removal_options = removal_calls[0][1]["options"]
        self.assertEqual(
            removal_options.enabled_camera_ids,  # type: ignore[union-attr]
            selected_camera_ids,
        )
        self.assertEqual(
            texture_calls[0]["model_glb"],
            processed_model.glb_bytes,
        )
        self.assertFalse(texture_calls[0]["enable_original_uv"])
        reference_images = texture_calls[0]["reference_images_png"]
        self.assertEqual(len(reference_images), 1)  # type: ignore[arg-type]
        decoded_reference = cv2.imdecode(
            np.frombuffer(reference_images[0], dtype=np.uint8),  # type: ignore[index]
            cv2.IMREAD_UNCHANGED,
        )
        np.testing.assert_array_equal(decoded_reference, selected)
        self.assertIsInstance(result, StagedMeshyGenerationResult)
        self.assertEqual(result.task_id, "texture-task")
        self.assertEqual(result.geometry_task_id, "geometry-task")
        self.assertEqual(result.source_glb_bytes, geometry_model.glb_bytes)
        self.assertEqual(
            result.postprocessed_glb_bytes,
            processed_model.glb_bytes,
        )
        self.assertEqual(result.removed_face_count, 3)
        self.assertEqual(result.enabled_camera_ids, selected_camera_ids)
        self.assertIn("Submitting geometry-only Meshy task...", progress_messages)

    def test_staged_planner_stops_before_retexture_on_removal_cancellation(
        self,
    ) -> None:
        request = GenerationRequest(
            frame_index=0,
            selected_object_bgra=np.zeros((4, 4, 4), dtype=np.uint8),
            settings=GenerationServiceSettings(
                meshy_api_key="meshy-key",
                unused_face_removal=True,
            ),
            enabled_camera_ids=("pos_x",),
        )

        with (
            patch(
                "housemaker.generation_workspace.request_image_to_3d_model",
                return_value=_test_meshy_result(),
            ),
            patch(
                "housemaker.generation_workspace.remove_unused_faces_from_glb",
                side_effect=UnusedFaceRemovalCancelled("cancelled"),
            ),
            patch(
                "housemaker.generation_workspace.request_retextured_model"
            ) as retexture_mock,
        ):
            with self.assertRaises(UnusedFaceRemovalCancelled):
                MeshyImagePlanner().plan(request)

        retexture_mock.assert_not_called()

    def test_camera_uv_projection_alone_forces_staged_generation(self) -> None:
        request = GenerationRequest(
            frame_index=0,
            selected_object_bgra=np.zeros((6, 8, 4), dtype=np.uint8),
            settings=GenerationServiceSettings(
                meshy_api_key="meshy-camera-uv-key",
                project_uvs_from_camera_views=True,
            ),
            enabled_camera_ids=(),
        )
        projected_glb = _test_camera_uv_glb()
        returned_glb = _test_camera_uv_glb(redundant_vertices=True)
        geometry_result = MeshyGenerationResult(
            task_id="geometry-task",
            glb_bytes=b"geometry-only glb",
            name="Geometry",
        )
        textured_result = MeshyGenerationResult(
            task_id="texture-task",
            glb_bytes=returned_glb,
            name="Textured object",
        )
        projected = SimpleNamespace(
            glb_bytes=projected_glb,
            camera_face_counts={
                camera_id: index + 1
                for index, camera_id in enumerate(ALL_CAMERA_IDS)
            },
            leftover_face_count=3,
            invisible_face_count=1,
            quality_fallback_face_count=1,
            conflict_fallback_face_count=1,
        )
        progress_messages: list[str] = []

        with (
            patch(
                "housemaker.generation_workspace.request_image_to_3d_model",
                return_value=geometry_result,
            ) as image_request,
            patch(
                "housemaker.generation_workspace."
                "purge_faces_visible_from_unchecked_cameras_from_glb",
                return_value=SimpleNamespace(
                    glb_bytes=b"geometry-only glb",
                    original_face_count=12,
                    retained_face_count=12,
                    removed_face_count=0,
                ),
            ) as purge,
            patch(
                "housemaker.generation_workspace.remove_unused_faces_from_glb"
            ) as removal,
            patch(
                "housemaker.generation_workspace."
                "project_uvs_from_camera_views_from_glb",
                return_value=projected,
            ) as projection,
            patch(
                "housemaker.generation_workspace.request_retextured_model",
                return_value=textured_result,
            ) as retexture,
        ):
            result = MeshyImagePlanner().plan(
                request,
                progress_callback=progress_messages.append,
                cancel_event=threading.Event(),
            )

        self.assertFalse(image_request.call_args.kwargs["should_texture"])
        self.assertEqual(
            purge.call_args.kwargs["unchecked_camera_ids"],
            ALL_CAMERA_IDS,
        )
        removal.assert_not_called()
        self.assertEqual(projection.call_args.args, (b"geometry-only glb",))
        self.assertTrue(
            callable(projection.call_args.kwargs["cancel_requested"])
        )
        self.assertEqual(
            retexture.call_args.kwargs["model_glb"],
            projected_glb,
        )
        self.assertTrue(retexture.call_args.kwargs["enable_original_uv"])
        self.assertIsInstance(result, StagedMeshyGenerationResult)
        self.assertTrue(result.camera_uv_projection_applied)
        self.assertEqual(result.postprocessed_glb_bytes, projected_glb)
        self.assertEqual(
            result.camera_uv_projection_version,
            "camera-view-uv-v3-strict",
        )
        self.assertEqual(
            result.camera_uv_fingerprint_version,
            CAMERA_UV_FINGERPRINT_VERSION,
        )
        self.assertEqual(
            result.camera_uv_submitted_fingerprint,
            result.camera_uv_final_fingerprint,
        )
        self.assertEqual(result.camera_uv_integrity_face_count, 2)
        self.assertEqual(
            tuple(camera_id for camera_id, _count in result.camera_uv_face_counts),
            ALL_CAMERA_IDS,
        )
        self.assertEqual(result.camera_uv_leftover_face_count, 3)
        self.assertEqual(result.camera_uv_invisible_face_count, 1)
        self.assertEqual(result.camera_uv_quality_fallback_face_count, 1)
        self.assertEqual(result.camera_uv_conflict_fallback_face_count, 1)
        self.assertIn(
            "Projecting UVs from six fixed camera views...",
            progress_messages,
        )

    def test_combined_postprocessing_orders_removal_before_uv_projection(
        self,
    ) -> None:
        request = GenerationRequest(
            frame_index=0,
            selected_object_bgra=np.zeros((4, 4, 4), dtype=np.uint8),
            settings=GenerationServiceSettings(
                meshy_api_key="meshy-key",
                unused_face_removal=True,
                project_uvs_from_camera_views=True,
            ),
            enabled_camera_ids=("pos_x",),
        )
        call_order: list[str] = []
        projected_glb = _test_camera_uv_glb()
        returned_glb = _test_camera_uv_glb(redundant_vertices=True)

        def generate_geometry(**_kwargs: object) -> MeshyGenerationResult:
            call_order.append("geometry")
            return MeshyGenerationResult("geometry-task", b"geometry")

        def purge_faces(glb_bytes: bytes, **_kwargs: object) -> object:
            self.assertEqual(glb_bytes, b"geometry")
            call_order.append("purge")
            return SimpleNamespace(
                glb_bytes=b"purged geometry",
                original_face_count=14,
                retained_face_count=12,
                removed_face_count=2,
            )

        def remove_faces(
            glb_bytes: bytes,
            **_kwargs: object,
        ) -> UnusedFaceRemovalResult:
            self.assertEqual(glb_bytes, b"purged geometry")
            call_order.append("removal")
            result = SimpleNamespace(
                glb_bytes=b"visible geometry",
                original_face_count=12,
                retained_face_count=10,
                removed_face_count=2,
                protected_face_count=10,
            )
            return result  # type: ignore[return-value]

        def project_uvs(glb_bytes: bytes, **_kwargs: object) -> object:
            self.assertEqual(glb_bytes, b"visible geometry")
            call_order.append("projection")
            return SimpleNamespace(
                glb_bytes=projected_glb,
                camera_face_counts={camera_id: 1 for camera_id in ALL_CAMERA_IDS},
                leftover_face_count=0,
                invisible_face_count=0,
                quality_fallback_face_count=0,
                conflict_fallback_face_count=0,
            )

        def retexture_model(**kwargs: object) -> MeshyGenerationResult:
            self.assertEqual(kwargs["model_glb"], projected_glb)
            self.assertTrue(kwargs["enable_original_uv"])
            call_order.append("retexture")
            return MeshyGenerationResult("texture-task", returned_glb)

        with (
            patch(
                "housemaker.generation_workspace.request_image_to_3d_model",
                side_effect=generate_geometry,
            ),
            patch(
                "housemaker.generation_workspace."
                "purge_faces_visible_from_unchecked_cameras_from_glb",
                side_effect=purge_faces,
            ),
            patch(
                "housemaker.generation_workspace.remove_unused_faces_from_glb",
                side_effect=remove_faces,
            ),
            patch(
                "housemaker.generation_workspace."
                "project_uvs_from_camera_views_from_glb",
                side_effect=project_uvs,
            ),
            patch(
                "housemaker.generation_workspace.request_retextured_model",
                side_effect=retexture_model,
            ),
        ):
            result = MeshyImagePlanner().plan(request)

        self.assertEqual(
            call_order,
            ["geometry", "purge", "removal", "projection", "retexture"],
        )
        self.assertIsInstance(result, StagedMeshyGenerationResult)
        self.assertEqual(
            _staged_generation_mode(result),  # type: ignore[arg-type]
            "unchecked_camera_face_purge_and_unused_face_removal_and_"
            "camera_uv_projection",
        )

    def test_camera_uv_mutation_rejects_result_before_variant_executor(
        self,
    ) -> None:
        projected_glb = _test_camera_uv_glb()
        mutated_glb = _test_camera_uv_glb(
            redundant_vertices=True,
            mutate_uv=True,
        )
        request = GenerationRequest(
            frame_index=0,
            selected_object_bgra=np.zeros((4, 4, 4), dtype=np.uint8),
            settings=GenerationServiceSettings(
                meshy_api_key="meshy-key",
                project_uvs_from_camera_views=True,
            ),
        )
        executor = _FakeMeshyExecutor()
        worker = GenerationWorker(MeshyImagePlanner(), executor, request)
        failed_spy = QSignalSpy(worker.failed)

        with (
            patch(
                "housemaker.generation_workspace.request_image_to_3d_model",
                return_value=MeshyGenerationResult(
                    "geometry-task",
                    b"geometry",
                ),
            ),
            patch(
                "housemaker.generation_workspace."
                "project_uvs_from_camera_views_from_glb",
                return_value=SimpleNamespace(
                    glb_bytes=projected_glb,
                    camera_face_counts={
                        camera_id: 0 for camera_id in ALL_CAMERA_IDS
                    },
                    leftover_face_count=2,
                    invisible_face_count=2,
                    quality_fallback_face_count=0,
                    conflict_fallback_face_count=0,
                ),
            ),
            patch(
                "housemaker.generation_workspace.request_retextured_model",
                return_value=MeshyGenerationResult(
                    "texture-task",
                    mutated_glb,
                ),
            ),
        ):
            worker.run()

        self.assertEqual(failed_spy.count(), 1)
        self.assertIn(
            "changed the camera-projected UV layout",
            failed_spy.at(0)[0],
        )
        self.assertEqual(executor.results, [])

    def test_camera_uv_cancellation_before_retexture_skips_paid_task(self) -> None:
        cancel_event = threading.Event()
        request = GenerationRequest(
            frame_index=0,
            selected_object_bgra=np.zeros((4, 4, 4), dtype=np.uint8),
            settings=GenerationServiceSettings(
                meshy_api_key="meshy-key",
                project_uvs_from_camera_views=True,
            ),
        )

        def finish_projection_then_cancel(
            _glb_bytes: bytes,
            **_kwargs: object,
        ) -> object:
            cancel_event.set()
            return SimpleNamespace(
                glb_bytes=b"uv-authored geometry",
                camera_face_counts={camera_id: 1 for camera_id in ALL_CAMERA_IDS},
                leftover_face_count=0,
                invisible_face_count=0,
                quality_fallback_face_count=0,
                conflict_fallback_face_count=0,
            )

        with (
            patch(
                "housemaker.generation_workspace.request_image_to_3d_model",
                return_value=MeshyGenerationResult(
                    "geometry-task",
                    b"geometry",
                ),
            ),
            patch(
                "housemaker.generation_workspace."
                "project_uvs_from_camera_views_from_glb",
                side_effect=finish_projection_then_cancel,
            ),
            patch(
                "housemaker.generation_workspace.request_retextured_model"
            ) as retexture,
        ):
            with self.assertRaises(_GenerationCancelled):
                MeshyImagePlanner().plan(request, cancel_event=cancel_event)

        retexture.assert_not_called()

    def test_camera_uv_core_cancellation_uses_silent_worker_control_flow(
        self,
    ) -> None:
        cancel_event = threading.Event()
        request = GenerationRequest(
            frame_index=0,
            selected_object_bgra=np.zeros((4, 4, 4), dtype=np.uint8),
            settings=GenerationServiceSettings(
                meshy_api_key="meshy-key",
                project_uvs_from_camera_views=True,
            ),
        )

        def cancel_projection(
            _glb_bytes: bytes,
            **_kwargs: object,
        ) -> object:
            cancel_event.set()
            raise RuntimeError("projection core cancelled")

        with (
            patch(
                "housemaker.generation_workspace.request_image_to_3d_model",
                return_value=MeshyGenerationResult(
                    "geometry-task",
                    b"geometry",
                ),
            ),
            patch(
                "housemaker.generation_workspace."
                "project_uvs_from_camera_views_from_glb",
                side_effect=cancel_projection,
            ),
            patch(
                "housemaker.generation_workspace.request_retextured_model"
            ) as retexture,
        ):
            with self.assertRaises(_GenerationCancelled):
                MeshyImagePlanner().plan(request, cancel_event=cancel_event)

        retexture.assert_not_called()

    def test_worker_invokes_meshy_planner_and_executor(self) -> None:
        planner = _FakeMeshyPlanner()
        executor = _FakeMeshyExecutor()
        request = GenerationRequest(
            frame_index=2,
            selected_object_bgra=np.zeros((10, 10, 4), dtype=np.uint8),
            settings=GenerationServiceSettings(meshy_api_key="secret"),
        )
        worker = GenerationWorker(planner, executor, request)
        succeeded_spy = QSignalSpy(worker.succeeded)
        failed_spy = QSignalSpy(worker.failed)

        worker.run()

        self.assertEqual(succeeded_spy.count(), 1)
        self.assertEqual(failed_spy.count(), 0)
        self.assertEqual(planner.requests[0].frame_index, 2)
        self.assertEqual(len(executor.results), 1)

    def test_worker_redacts_the_meshy_api_key_from_errors(self) -> None:
        key = "meshy-never-show-this"

        def failing_planner(_request: GenerationRequest) -> MeshyGenerationResult:
            raise RuntimeError(f"provider echoed {key}")

        request = GenerationRequest(
            frame_index=0,
            selected_object_bgra=np.zeros((4, 4, 4), dtype=np.uint8),
            settings=GenerationServiceSettings(meshy_api_key=key),
        )
        worker = GenerationWorker(failing_planner, _FakeMeshyExecutor(), request)
        failed_spy = QSignalSpy(worker.failed)

        worker.run()

        self.assertEqual(failed_spy.count(), 1)
        self.assertNotIn(key, failed_spy.at(0)[0])
        self.assertIn("[redacted]", failed_spy.at(0)[0])

    def test_worker_prepares_a_custom_model_processor_before_planning(self) -> None:
        events: list[str] = []

        def planner(_request: GenerationRequest) -> MeshyGenerationResult:
            events.append("paid-meshy-request")
            return _test_meshy_result()

        class PreparedExecutor(_FakeMeshyExecutor):
            def prepare(self) -> None:
                events.append("model-processor-prepare")

        request = GenerationRequest(
            frame_index=0,
            selected_object_bgra=np.zeros((4, 4, 4), dtype=np.uint8),
            settings=GenerationServiceSettings(meshy_api_key="secret"),
        )

        GenerationWorker(planner, PreparedExecutor(), request).run()

        self.assertEqual(
            events,
            ["model-processor-prepare", "paid-meshy-request"],
        )

    def test_failed_custom_model_processor_prepare_skips_paid_request(self) -> None:
        planner_was_called = False

        def planner(_request: GenerationRequest) -> MeshyGenerationResult:
            nonlocal planner_was_called
            planner_was_called = True
            return _test_meshy_result()

        class FailingPreparedExecutor(_FakeMeshyExecutor):
            def prepare(self) -> None:
                raise RuntimeError("model processor unavailable")

        request = GenerationRequest(
            frame_index=0,
            selected_object_bgra=np.zeros((4, 4, 4), dtype=np.uint8),
            settings=GenerationServiceSettings(meshy_api_key="secret"),
        )
        worker = GenerationWorker(planner, FailingPreparedExecutor(), request)
        failed_spy = QSignalSpy(worker.failed)

        worker.run()

        self.assertFalse(planner_was_called)
        self.assertEqual(failed_spy.count(), 1)
        self.assertIn("model processor unavailable", failed_spy.at(0)[0])


# ### Workspace tests ###
class GenerationWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.meshy_planner = _FakeMeshyPlanner()
        self.meshy_executor = _FakeMeshyExecutor()
        self.workspace = GenerationWorkspace(
            meshy_planner=self.meshy_planner,
            meshy_executor=self.meshy_executor,
        )
        self.workspace.resize(900, 600)
        self.workspace.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()

    def test_layout_controls_and_video_seek_keep_masks_per_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_path = Path(temporary_directory) / "source.avi"
            _write_test_video(video_path)
            self.workspace.load_video(str(video_path))

            self.assertEqual(self.workspace.seekbar.maximum(), 2)
            self.assertEqual(self.workspace.frame_label.text(), "Frame 1 / 3")
            self.assertTrue(self.workspace.load_video_button.isEnabled())
            self.assertFalse(self.workspace.generate_button.isEnabled())

            self.workspace.video_view.set_strokes([_test_stroke()])
            self.workspace._handle_video_strokes_changed(
                self.workspace.video_view.get_strokes()
            )
            self.workspace.show_frame(1)
            self.assertEqual(self.workspace.video_view.get_strokes(), [])
            self.workspace.show_frame(0)
            self.assertEqual(self.workspace.video_view.get_strokes(), [_test_stroke()])

            saved = self.workspace.get_data()
            self.assertEqual(saved.current_frame_index, 0)
            self.assertEqual(saved.strokes_for_frame(0), [_test_stroke()])

            self.workspace.set_data(saved)
            self.assertEqual(
                self.workspace.video_view.get_strokes(),
                [_test_stroke()],
            )
            self.workspace.shutdown()

    def test_meshy_is_the_only_generation_path_and_requires_its_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_path = Path(temporary_directory) / "source.avi"
            _write_test_video(video_path, frame_count=1)
            self.workspace.load_video(str(video_path))
            self.workspace.video_view.set_strokes([_test_stroke()])
            self.workspace._handle_video_strokes_changed(
                self.workspace.video_view.get_strokes()
            )

            self.assertFalse(hasattr(self.workspace, "generation_backend_combo"))
            self.assertTrue(self.workspace.meshy_target_polycount_control.isVisible())
            self.assertFalse(self.workspace.generate_button.isEnabled())

            self.workspace.set_runtime_settings(
                GenerationServiceSettings(meshy_api_key="meshy-key")
            )
            self.assertTrue(self.workspace.generate_button.isEnabled())
            self.workspace.shutdown()

    def test_meshy_target_triangles_are_always_visible_and_preserve_user_value(
        self,
    ) -> None:
        control = self.workspace.meshy_target_polycount_control
        spinbox = self.workspace.meshy_target_polycount_spinbox

        self.assertTrue(control.isVisible())
        self.workspace.set_runtime_settings(
            GenerationServiceSettings(
                meshy_api_key="meshy-key",
                meshy_target_polycount=7_200,
            )
        )
        self.assertEqual(spinbox.minimum(), 100)
        self.assertEqual(spinbox.maximum(), 15_000)
        self.assertEqual(spinbox.value(), 7_200)

        spinbox.setValue(5_600)
        self.assertEqual(
            self.workspace.get_runtime_settings().meshy_target_polycount,
            5_600,
        )
        self.workspace.set_runtime_settings(
            GenerationServiceSettings(meshy_api_key="replacement-key")
        )
        self.assertEqual(spinbox.value(), 5_600)

    def test_unused_face_cameras_default_to_all_and_are_snapshotted_per_request(
        self,
    ) -> None:
        checkboxes = self.workspace.object_3d_panel.unused_face_camera_checkboxes
        viewer = self.workspace.object_3d_panel.viewer

        self.assertEqual(tuple(checkboxes), ALL_CAMERA_IDS)
        self.assertTrue(all(checkbox.isChecked() for checkbox in checkboxes.values()))
        self.assertEqual(
            self.workspace.object_3d_panel.get_enabled_postprocess_camera_ids(),
            ALL_CAMERA_IDS,
        )
        self.assertTrue(
            self.workspace.object_3d_panel.unused_face_camera_controls.isEnabled()
        )
        self.assertTrue(viewer.get_unused_face_camera_indicators_visible())

        with tempfile.TemporaryDirectory() as temporary_directory:
            video_path = Path(temporary_directory) / "source.avi"
            _write_test_video(video_path, frame_count=1)
            self.workspace.set_runtime_settings(
                GenerationServiceSettings(
                    meshy_api_key="meshy-key",
                    unused_face_removal=True,
                )
            )
            self.workspace.load_video(str(video_path))
            self.workspace.video_view.set_strokes([_test_stroke()])
            self.workspace._handle_video_strokes_changed(
                self.workspace.video_view.get_strokes()
            )
            viewer.set_model(_test_model())

            self.assertTrue(viewer.get_unused_face_camera_indicators_visible())
            self.assertEqual(
                viewer.get_enabled_unused_face_camera_ids(),
                ALL_CAMERA_IDS,
            )
            self.assertTrue(
                all(
                    item.visible()
                    for camera_items in (
                        viewer.unused_face_camera_indicator_items.values()
                    )
                    for item in camera_items
                )
            )

            checkboxes["neg_x"].setChecked(False)
            checkboxes["bottom"].setChecked(False)

            expected_camera_ids = tuple(
                camera_id
                for camera_id in ALL_CAMERA_IDS
                if camera_id not in {"neg_x", "bottom"}
            )
            self.assertTrue(
                self.workspace.object_3d_panel.unused_face_camera_controls.isEnabled()
            )
            request = self.workspace._build_generation_request()
            self.assertIsNotNone(request)
            self.assertEqual(
                request.enabled_camera_ids,  # type: ignore[union-attr]
                expected_camera_ids,
            )
            self.assertEqual(
                viewer.get_enabled_unused_face_camera_ids(),
                expected_camera_ids,
            )
            self.assertTrue(
                all(
                    item.visible()
                    == (camera_id in expected_camera_ids)
                    for camera_id, camera_items in (
                        viewer.unused_face_camera_indicator_items.items()
                    )
                    for item in camera_items
                )
            )

            checkboxes["pos_x"].setChecked(False)
            self.assertEqual(
                request.enabled_camera_ids,  # type: ignore[union-attr]
                expected_camera_ids,
            )
            self.assertNotEqual(
                viewer.get_enabled_unused_face_camera_ids(),
                request.enabled_camera_ids,  # type: ignore[union-attr]
            )
            for checkbox in checkboxes.values():
                checkbox.setChecked(False)
            generate_is_enabled = self.workspace.generate_button.isEnabled()
            self.workspace.shutdown()
            self.assertFalse(generate_is_enabled)

    def test_camera_uv_only_allows_generation_with_removal_cameras_off(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_path = Path(temporary_directory) / "source.avi"
            _write_test_video(video_path, frame_count=1)
            self.workspace.set_runtime_settings(
                GenerationServiceSettings(
                    meshy_api_key="meshy-key",
                    project_uvs_from_camera_views=True,
                )
            )
            self.workspace.load_video(str(video_path))
            self.workspace.video_view.set_strokes([_test_stroke()])
            self.workspace._handle_video_strokes_changed(
                self.workspace.video_view.get_strokes()
            )
            camera_controls = (
                self.workspace.object_3d_panel.unused_face_camera_controls
            )
            checkboxes = (
                self.workspace.object_3d_panel.unused_face_camera_checkboxes
            )
            for checkbox in checkboxes.values():
                checkbox.setChecked(False)

            request = self.workspace._build_generation_request()

            self.assertTrue(camera_controls.isEnabled())
            self.assertTrue(
                self.workspace.result_view.get_unused_face_camera_indicators_visible()
            )
            self.assertTrue(self.workspace.generate_button.isEnabled())
            self.assertIsNotNone(request)
            self.assertEqual(request.enabled_camera_ids, ())  # type: ignore[union-attr]
            self.workspace.shutdown()

    def test_staged_success_persists_all_revisions_and_task_provenance(
        self,
    ) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()

        final_model = _test_model()
        staged_result = _test_staged_meshy_result(final_model)
        meshy_planner = _FakeMeshyPlanner(staged_result)
        meshy_executor = _FakeMeshyExecutor(final_model)

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            asset_directory = temporary_path / "generation_assets"
            video_path = temporary_path / "source.avi"
            _write_test_video(video_path, frame_count=1)
            self.workspace = GenerationWorkspace(
                meshy_planner=meshy_planner,
                meshy_executor=meshy_executor,
                asset_directory=asset_directory,
            )
            self.workspace.show()
            self.workspace.set_runtime_settings(
                GenerationServiceSettings(
                    meshy_api_key="meshy-key",
                    unused_face_removal=True,
                )
            )
            self.workspace.load_video(str(video_path))
            self.workspace.video_view.set_strokes([_test_stroke()])
            self.workspace._handle_video_strokes_changed(
                self.workspace.video_view.get_strokes()
            )
            completed_spy = QSignalSpy(self.workspace.generation_completed)

            self.workspace.generate()

            deadline = time.monotonic() + 3.0
            while completed_spy.count() == 0 and time.monotonic() < deadline:
                _qt_application.processEvents()
                QTest.qWait(5)
            self.assertEqual(completed_spy.count(), 1)
            while self.workspace.is_generating and time.monotonic() < deadline:
                _qt_application.processEvents()
                QTest.qWait(5)
            self.assertFalse(self.workspace.is_generating)
            self.assertEqual(len(meshy_planner.requests), 1)
            self.assertTrue(
                meshy_planner.requests[0].settings.unused_face_removal
            )
            self.assertEqual(
                meshy_planner.requests[0].enabled_camera_ids,
                ALL_CAMERA_IDS,
            )
            self.assertEqual(meshy_executor.results, [staged_result])
            self.assertIs(self.workspace.result_view.model, final_model)

            saved_data = self.workspace.get_data()
            self.assertEqual(len(saved_data.generated_objects), 1)
            record = saved_data.generated_objects[0]
            self.assertEqual(record.provider_task_id, "texture-task-123")
            self.assertEqual(record.pipeline["mode"], "unused_face_removal")
            self.assertEqual(
                record.pipeline["geometry_task_id"],
                "geometry-task-123",
            )
            self.assertEqual(
                record.pipeline["enabled_camera_ids"],
                ["pos_x", "top"],
            )
            self.assertEqual(record.pipeline["removed_face_count"], 40)
            self.assertEqual(record.pipeline["retained_face_count"], 80)
            self.assertEqual(
                (asset_directory / str(record.asset_path)).read_bytes(),
                final_model.glb_bytes,
            )
            self.assertEqual(
                (
                    asset_directory
                    / str(record.pipeline["source_asset_path"])
                ).read_bytes(),
                staged_result.source_glb_bytes,
            )
            self.assertEqual(
                (
                    asset_directory
                    / str(record.pipeline["postprocessed_asset_path"])
                ).read_bytes(),
                staged_result.postprocessed_glb_bytes,
            )
            restored_data = GenerationData.from_dict(saved_data.to_dict())
            self.assertEqual(restored_data.generated_objects[0], record)
            self.assertIn("Removed 40 of 120 faces", self.workspace.status_label.text())
            self.workspace.shutdown()

    def test_camera_uv_success_persists_projection_provenance(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        final_model = _test_model()
        source_glb = b"source geometry glb"
        uv_authored_glb = b"uv-authored geometry glb"
        camera_face_counts = tuple(
            (camera_id, index + 1)
            for index, camera_id in enumerate(ALL_CAMERA_IDS)
        )
        result = StagedMeshyGenerationResult(
            task_id="texture-task-uv",
            glb_bytes=final_model.glb_bytes,
            name="Camera UV chair",
            geometry_task_id="geometry-task-uv",
            source_glb_bytes=source_glb,
            postprocessed_glb_bytes=uv_authored_glb,
            camera_uv_projection_applied=True,
            camera_uv_face_counts=camera_face_counts,
            camera_uv_leftover_face_count=4,
            camera_uv_invisible_face_count=2,
            camera_uv_quality_fallback_face_count=1,
            camera_uv_conflict_fallback_face_count=1,
            camera_uv_projection_version=CAMERA_UV_PROJECTION_VERSION,
            camera_uv_fingerprint_version=CAMERA_UV_FINGERPRINT_VERSION,
            camera_uv_submitted_fingerprint="a" * 64,
            camera_uv_final_fingerprint="a" * 64,
            camera_uv_integrity_face_count=24,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            asset_directory = Path(temporary_directory) / "generation_assets"
            self.workspace = GenerationWorkspace(
                asset_directory=asset_directory,
            )
            self.workspace._handle_generation_succeeded(result, final_model)

            record = self.workspace.get_data().generated_objects[0]
            pipeline = record.pipeline
            self.assertEqual(pipeline["mode"], "camera_uv_projection")
            self.assertTrue(pipeline["camera_uv_projection_applied"])
            self.assertEqual(
                pipeline["camera_uv_projection_camera_ids"],
                list(ALL_CAMERA_IDS),
            )
            self.assertEqual(
                pipeline["camera_uv_projected_face_count"],
                sum(range(1, len(ALL_CAMERA_IDS) + 1)),
            )
            self.assertEqual(
                pipeline["camera_uv_face_counts"],
                dict(camera_face_counts),
            )
            self.assertEqual(pipeline["camera_uv_leftover_face_count"], 4)
            self.assertEqual(pipeline["camera_uv_invisible_face_count"], 2)
            self.assertEqual(
                pipeline["camera_uv_quality_fallback_face_count"],
                1,
            )
            self.assertEqual(
                pipeline["camera_uv_conflict_fallback_face_count"],
                1,
            )
            self.assertTrue(pipeline["retexture_enable_original_uv"])
            self.assertEqual(
                pipeline["camera_uv_projection_version"],
                "camera-view-uv-v3-strict",
            )
            self.assertEqual(
                pipeline["camera_uv_fingerprint_version"],
                CAMERA_UV_FINGERPRINT_VERSION,
            )
            self.assertEqual(
                pipeline["camera_uv_submitted_fingerprint"],
                "a" * 64,
            )
            self.assertEqual(
                pipeline["camera_uv_final_fingerprint"],
                "a" * 64,
            )
            self.assertEqual(pipeline["camera_uv_integrity_face_count"], 24)
            self.assertEqual(
                (
                    asset_directory
                    / str(pipeline["source_asset_path"])
                ).read_bytes(),
                source_glb,
            )
            self.assertEqual(
                (
                    asset_directory
                    / str(pipeline["postprocessed_asset_path"])
                ).read_bytes(),
                uv_authored_glb,
            )
            self.assertIn(
                "Projected 21 faces from six fixed camera views; fallback UV "
                "islands contain 2 invisible faces, 1 quality-rejected "
                "depth-visible face, and 1 projection-conflict face.",
                self.workspace.status_label.text(),
            )

    def test_legacy_camera_uv_status_uses_total_fallback_count(self) -> None:
        legacy_result = StagedMeshyGenerationResult(
            task_id="legacy-texture-task",
            glb_bytes=b"legacy textured glb",
            camera_uv_projection_applied=True,
            camera_uv_face_counts=(("pos_x", 7),),
            camera_uv_leftover_face_count=3,
        )

        status = _format_staged_generation_status(
            "Legacy object",
            legacy_result,
        )

        self.assertIn("3 faces use fallback UV islands", status)
        self.assertNotIn("0 invisible faces", status)

    def test_meshy_success_displays_saves_persists_and_rebuilds_glb(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()

        model = _test_model()
        meshy_result = MeshyGenerationResult(
            task_id="task-complex-chair",
            glb_bytes=model.glb_bytes,
        )
        meshy_planner = _FakeMeshyPlanner(meshy_result)
        meshy_executor = _FakeMeshyExecutor(model)

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            asset_directory = temporary_path / "generation_assets"
            video_path = temporary_path / "source.avi"
            _write_test_video(video_path, frame_count=1)
            self.workspace = GenerationWorkspace(
                meshy_planner=meshy_planner,
                meshy_executor=meshy_executor,
                asset_directory=asset_directory,
            )
            restored_workspace: GenerationWorkspace | None = None
            try:
                self.workspace.set_runtime_settings(
                    GenerationServiceSettings(meshy_api_key="meshy-test-key")
                )
                self.workspace.load_video(str(video_path))
                self.workspace.video_view.set_strokes([_test_stroke()])
                self.workspace._handle_video_strokes_changed(
                    self.workspace.video_view.get_strokes()
                )
                completed_spy = QSignalSpy(self.workspace.generation_completed)

                self.workspace.generate()

                deadline = time.monotonic() + 3.0
                while completed_spy.count() == 0 and time.monotonic() < deadline:
                    _qt_application.processEvents()
                    QTest.qWait(5)
                self.assertEqual(completed_spy.count(), 1)
                while self.workspace.is_generating and time.monotonic() < deadline:
                    _qt_application.processEvents()
                    QTest.qWait(5)
                self.assertFalse(self.workspace.is_generating)
                self.assertEqual(len(meshy_planner.requests), 1)
                routed_request = meshy_planner.requests[0]
                self.assertGreater(
                    np.count_nonzero(routed_request.selected_object_bgra[:, :, 3]),
                    0,
                )
                self.assertEqual(meshy_executor.results, [meshy_result])
                self.assertIs(self.workspace.result_view.model, model)

                saved_data = self.workspace.get_data()
                self.assertEqual(len(saved_data.generated_objects), 1)
                record = saved_data.generated_objects[0]
                self.assertEqual(record.provider, GENERATION_BACKEND_MESHY)
                self.assertEqual(record.provider_task_id, "task-complex-chair")
                self.assertEqual(record.pipeline, {})
                self.assertIsNotNone(record.asset_path)
                self.assertEqual(
                    (asset_directory / str(record.asset_path)).read_bytes(),
                    model.glb_bytes,
                )

                restored_workspace = GenerationWorkspace(
                    asset_directory=asset_directory
                )
                restored_workspace.set_data(saved_data)
                rebuilt = restored_workspace.result_view.model
                self.assertIsNotNone(rebuilt)
                self.assertEqual(rebuilt.glb_bytes, model.glb_bytes)
            finally:
                if restored_workspace is not None:
                    restored_workspace.shutdown()
                    restored_workspace.close()
                self.workspace.shutdown()
                self.workspace.close()
                _qt_application.processEvents()

    def test_display_options_control_texture_and_wireframe_visibility(self) -> None:
        self.assertTrue(self.workspace.textures_checkbox.isChecked())
        self.assertFalse(self.workspace.wireframe_checkbox.isChecked())
        self.assertTrue(self.workspace.result_view.get_textures_enabled())
        self.assertFalse(self.workspace.result_view.get_wireframe_enabled())
        self.assertFalse(self.workspace.texture_view.uv_overlay_enabled)
        self.assertIn("UV", self.workspace.wireframe_checkbox.toolTip())

        self.workspace.textures_checkbox.setChecked(False)
        self.workspace.wireframe_checkbox.setChecked(True)

        self.assertFalse(self.workspace.result_view.get_textures_enabled())
        self.assertTrue(self.workspace.result_view.get_wireframe_enabled())
        self.assertTrue(self.workspace.texture_view.uv_overlay_enabled)

    def test_model_uv_triangles_are_collected_per_face_for_texture_preview(
        self,
    ) -> None:
        model = import_generated_glb(_test_camera_uv_glb())

        triangles = _collect_model_uv_triangles(model)

        np.testing.assert_allclose(
            np.asarray(triangles),
            np.asarray(
                (
                    ((0.1, 0.1), (0.9, 0.1), (0.9, 0.9)),
                    ((0.1, 0.1), (0.9, 0.9), (0.1, 0.9)),
                )
            ),
            rtol=0.0,
            atol=1e-7,
        )

    def test_failed_object_load_clears_stale_uv_geometry_but_keeps_toggle(
        self,
    ) -> None:
        self.workspace.wireframe_checkbox.setChecked(True)
        self.workspace.texture_view.set_uv_overlay_triangles(
            (((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),)
        )
        missing_record = GeneratedObjectRecord(
            object_id="missing-object",
            frame_index=0,
            object_name="Missing",
            pipeline={},
            provider=GENERATION_BACKEND_MESHY,
            provider_task_id="missing-task",
            asset_path="missing.glb",
        )
        self.workspace._data.generated_objects = [missing_record]

        self.workspace._display_generated_object(missing_record)

        self.assertEqual(self.workspace.texture_view.uv_overlay_triangles, ())
        self.assertTrue(self.workspace.texture_view.uv_overlay_enabled)
        self.assertIn(
            "could not be rebuilt",
            self.workspace.status_label.text(),
        )

    def test_generated_model_statistics_are_displayed_and_reset(self) -> None:
        model = _test_model()
        self.workspace._handle_generation_succeeded(_test_meshy_result(), model)

        statistics = self.workspace.model_statistics_label.text()
        self.assertIn("8 vertices", statistics)
        self.assertIn("12 triangles", statistics)
        self.assertIn("material color", statistics)
        self.assertEqual(_format_model_statistics(None), "No generated object")

        self.workspace.set_data(GenerationData())
        self.assertEqual(
            self.workspace.model_statistics_label.text(),
            "No generated object",
        )

    def test_object_list_selects_saved_models_and_current_texture_preview(
        self,
    ) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()

        with tempfile.TemporaryDirectory() as temporary_directory:
            asset_directory = Path(temporary_directory) / "generation_assets"
            asset_directory.mkdir()
            first_model = _test_textured_model((220, 20, 30, 255))
            second_model = _test_textured_model((25, 200, 45, 255))
            (asset_directory / "first.glb").write_bytes(first_model.glb_bytes)
            (asset_directory / "second.glb").write_bytes(second_model.glb_bytes)
            data = GenerationData(
                generated_objects=[
                    GeneratedObjectRecord(
                        object_id="first-object",
                        frame_index=2,
                        object_name="First chair",
                        pipeline={},
                        provider=GENERATION_BACKEND_MESHY,
                        provider_task_id="first-task",
                        asset_path="first.glb",
                    ),
                    GeneratedObjectRecord(
                        object_id="second-object",
                        frame_index=5,
                        object_name="Second chair",
                        pipeline={},
                        provider=GENERATION_BACKEND_MESHY,
                        provider_task_id="second-task",
                        asset_path="second.glb",
                    ),
                ]
            )
            self.workspace = GenerationWorkspace(
                asset_directory=asset_directory
            )
            self.workspace.set_data(data)

            self.assertEqual(self.workspace.generated_objects_list.count(), 2)
            self.assertEqual(
                self.workspace.generated_objects_list.currentRow(),
                1,
            )
            self.assertIsNotNone(self.workspace.result_view.model)
            self.assertEqual(self.workspace.texture_view.entries, ())
            self.assertIsNone(self.workspace.texture_view.selected_entry)
            self.assertEqual(
                self.workspace.texture_view.preview_label.text(),
                "No texture resolutions available",
            )

            self.workspace.generated_objects_list.setCurrentRow(0)
            _qt_application.processEvents()

            self.assertEqual(self.workspace.texture_view.entries, ())
            self.assertIsNone(self.workspace.texture_view.selected_entry)

            self.workspace.set_external_3d_viewer_active(True)
            self.assertIs(
                self.workspace.right_view_stack.currentWidget(),
                self.workspace.texture_view_page,
            )
            self.workspace.set_external_3d_viewer_active(False)
            self.assertIs(
                self.workspace.right_view_stack.currentWidget(),
                self.workspace.object_3d_page,
            )

    def test_ambient_slider_updates_generated_object_view_lighting(self) -> None:
        self.workspace.ambient_light_slider.setValue(72)
        self.assertAlmostEqual(
            self.workspace.result_view.get_ambient_light_intensity(),
            0.72,
        )

    def test_shutdown_discards_an_in_flight_meshy_result(self) -> None:
        planner = _BlockingMeshyPlanner()
        self.workspace.set_meshy_planner(planner)
        request = GenerationRequest(
            frame_index=0,
            selected_object_bgra=np.zeros((4, 4, 4), dtype=np.uint8),
            settings=GenerationServiceSettings(meshy_api_key="key"),
        )
        completed_spy = QSignalSpy(self.workspace.generation_completed)

        self.workspace._start_generation(request)
        self.assertTrue(planner.started.wait(timeout=1.0))
        active_thread = self.workspace._generation_thread
        self.assertIsNotNone(active_thread)

        self.workspace.shutdown()
        planner.release.set()
        self.assertTrue(active_thread.wait(2000))
        _qt_application.processEvents()

        self.assertEqual(completed_spy.count(), 0)
        self.assertEqual(self.workspace.get_data().generated_objects, [])


# ### Generated-object deletion tests ###
class GeneratedObjectDeletionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.asset_directory = (
            Path(self.temporary_directory.name) / "generation_assets"
        )
        self.asset_directory.mkdir()
        self.workspace = GenerationWorkspace(
            asset_directory=self.asset_directory
        )
        self.workspace.resize(900, 600)
        self.workspace.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self.temporary_directory.cleanup()

    def _set_generated_objects(
        self,
        object_specs: list[tuple[str, str, tuple[int, int, int, int]]],
    ) -> list[GeneratedObjectRecord]:
        records: list[GeneratedObjectRecord] = []
        for frame_index, (object_id, name, color) in enumerate(object_specs):
            model = _test_textured_model(color)
            asset_path = f"{object_id}.glb"
            (self.asset_directory / asset_path).write_bytes(model.glb_bytes)
            records.append(
                GeneratedObjectRecord(
                    object_id=object_id,
                    frame_index=frame_index,
                    object_name=name,
                    pipeline={},
                    provider=GENERATION_BACKEND_MESHY,
                    provider_task_id=f"task-{object_id}",
                    asset_path=asset_path,
                )
            )
        self.workspace.set_data(GenerationData(generated_objects=records))
        _qt_application.processEvents()
        return records

    def test_delete_button_is_visible_and_requires_a_selected_object(
        self,
    ) -> None:
        button = self.workspace.delete_generated_object_button

        self.assertIs(
            button,
            self.workspace.object_3d_panel.delete_object_button,
        )
        self.assertEqual(button.text(), "Delete object")
        self.assertTrue(button.isVisible())
        self.assertFalse(button.isEnabled())

        self._set_generated_objects(
            [("chair", "Chair", (180, 30, 20, 255))]
        )
        self.assertTrue(button.isEnabled())

        self.workspace.generated_objects_list.setCurrentRow(-1)
        _qt_application.processEvents()
        self.assertFalse(button.isEnabled())
        self.assertFalse(self.workspace.delete_selected_generated_object())

    def test_delete_button_confirms_and_cancel_preserves_the_object(self) -> None:
        self._set_generated_objects(
            [("chair", "Chair", (180, 30, 20, 255))]
        )
        button = self.workspace.delete_generated_object_button

        with patch(
            "housemaker.generation_workspace.QMessageBox.question",
            return_value=QMessageBox.StandardButton.No,
        ) as question_mock:
            QTest.mouseClick(button, Qt.MouseButton.LeftButton)
            _qt_application.processEvents()

        question_mock.assert_called_once()
        self.assertEqual(
            [
                record.object_id
                for record in self.workspace.get_data().generated_objects
            ],
            ["chair"],
        )
        self.assertTrue(button.isEnabled())

        with patch(
            "housemaker.generation_workspace.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Yes,
        ) as question_mock:
            QTest.mouseClick(button, Qt.MouseButton.LeftButton)
            _qt_application.processEvents()

        question_mock.assert_called_once()
        self.assertEqual(self.workspace.get_data().generated_objects, [])
        self.assertEqual(self.workspace.generated_objects_list.count(), 0)
        self.assertFalse(button.isEnabled())
        self.assertIsNone(self.workspace.result_view.model)

    def test_delete_selects_successor_then_previous_for_the_last_row(
        self,
    ) -> None:
        self._set_generated_objects(
            [
                ("first", "First", (180, 30, 20, 255)),
                ("second", "Second", (20, 180, 30, 255)),
                ("third", "Third", (20, 30, 180, 255)),
            ]
        )
        self.workspace.generated_objects_list.setCurrentRow(1)
        _qt_application.processEvents()

        self.assertTrue(self.workspace.delete_selected_generated_object())
        self.assertEqual(self.workspace._selected_object_id, "third")
        self.assertEqual(self.workspace.generated_objects_list.currentRow(), 1)
        self.assertEqual(
            [
                record.object_id
                for record in self.workspace.get_data().generated_objects
            ],
            ["first", "third"],
        )

        self.assertTrue(self.workspace.delete_generated_object("third"))
        self.assertEqual(self.workspace._selected_object_id, "first")
        self.assertEqual(self.workspace.generated_objects_list.currentRow(), 0)
        self.assertEqual(
            [
                record.object_id
                for record in self.workspace.get_data().generated_objects
            ],
            ["first"],
        )

    def test_delete_removes_record_model_cache_and_texture_preview(
        self,
    ) -> None:
        self._set_generated_objects(
            [
                ("first", "First", (180, 30, 20, 255)),
                ("second", "Second", (20, 180, 30, 255)),
            ]
        )
        self.workspace.generated_objects_list.setCurrentRow(0)
        _qt_application.processEvents()
        changed_spy = QSignalSpy(self.workspace.data_changed)

        self.assertIn("first", self.workspace._generated_model_cache)
        self.assertTrue(self.workspace.delete_generated_object("first"))

        self.assertEqual(changed_spy.count(), 1)
        self.assertNotIn("first", self.workspace._generated_model_cache)
        self.assertEqual(self.workspace.texture_view.entries, ())
        self.assertEqual(self.workspace._selected_object_id, "second")
        self.assertIsNotNone(self.workspace.result_view.model)

        self.assertFalse(self.workspace.delete_generated_object("unknown"))
        self.assertEqual(changed_spy.count(), 1)
        self.assertEqual(
            [
                record.object_id
                for record in self.workspace.get_data().generated_objects
            ],
            ["second"],
        )

    def test_delete_unlinks_final_and_revision_glbs_except_shared_assets(
        self,
    ) -> None:
        model = _test_textured_model((180, 30, 20, 255))
        final_path = self.asset_directory / "deleted.glb"
        source_path = self.asset_directory / "deleted.geometry.glb"
        shared_path = self.asset_directory / "shared.glb"
        for asset_path in (final_path, source_path, shared_path):
            asset_path.write_bytes(model.glb_bytes)
        deleted_record = GeneratedObjectRecord(
            object_id="deleted",
            frame_index=0,
            object_name="Deleted",
            pipeline={
                "source_asset_path": source_path.name,
                "postprocessed_asset_path": shared_path.name,
            },
            provider=GENERATION_BACKEND_MESHY,
            provider_task_id="task-deleted",
            asset_path=final_path.name,
        )
        keeper_record = GeneratedObjectRecord(
            object_id="keeper",
            frame_index=1,
            object_name="Keeper",
            pipeline={},
            provider=GENERATION_BACKEND_MESHY,
            provider_task_id="task-keeper",
            asset_path=shared_path.name,
        )
        self.workspace.set_data(
            GenerationData(generated_objects=[deleted_record, keeper_record])
        )

        self.assertTrue(self.workspace.delete_generated_object("deleted"))

        self.assertFalse(final_path.exists())
        self.assertFalse(source_path.exists())
        self.assertTrue(shared_path.exists())

    def test_delete_refuses_outside_and_non_glb_paths_and_allows_missing(
        self,
    ) -> None:
        outside_path = self.asset_directory.parent / "outside.glb"
        notes_path = self.asset_directory / "notes.txt"
        outside_path.write_bytes(b"outside")
        notes_path.write_bytes(b"notes")
        record = GeneratedObjectRecord(
            object_id="unsafe",
            frame_index=0,
            object_name="Unsafe paths",
            pipeline={
                "source_asset_path": "../outside.glb",
                "postprocessed_asset_path": notes_path.name,
            },
            provider=GENERATION_BACKEND_MESHY,
            provider_task_id="task-unsafe",
            asset_path="already-missing.glb",
        )
        self.workspace.set_data(GenerationData(generated_objects=[record]))

        self.assertTrue(self.workspace.delete_generated_object("unsafe"))

        self.assertTrue(outside_path.exists())
        self.assertTrue(notes_path.exists())
        self.assertEqual(self.workspace.get_data().generated_objects, [])

    def test_asset_unlink_failure_does_not_restore_the_deleted_record(
        self,
    ) -> None:
        model = _test_textured_model((180, 30, 20, 255))
        locked_path = self.asset_directory / "locked.glb"
        locked_path.write_bytes(model.glb_bytes)
        record = GeneratedObjectRecord(
            object_id="locked",
            frame_index=0,
            object_name="Locked",
            pipeline={},
            provider=GENERATION_BACKEND_MESHY,
            provider_task_id="task-locked",
            asset_path=locked_path.name,
        )
        self.workspace.set_data(GenerationData(generated_objects=[record]))

        with patch.object(Path, "unlink", side_effect=OSError("in use")):
            self.assertTrue(self.workspace.delete_generated_object("locked"))

        self.assertTrue(locked_path.exists())
        self.assertEqual(self.workspace.get_data().generated_objects, [])
        self.assertIn("could not be removed", self.workspace.status_label.text())

    def test_delete_is_disabled_and_refused_while_generating(self) -> None:
        self._set_generated_objects(
            [("chair", "Chair", (180, 30, 20, 255))]
        )
        planner = _BlockingMeshyPlanner()
        self.workspace.set_meshy_planner(planner)
        request = GenerationRequest(
            frame_index=0,
            selected_object_bgra=np.zeros((4, 4, 4), dtype=np.uint8),
            settings=GenerationServiceSettings(meshy_api_key="key"),
        )

        self.workspace._start_generation(request)
        self.assertTrue(planner.started.wait(timeout=1.0))
        self.assertFalse(
            self.workspace.delete_generated_object_button.isEnabled()
        )
        self.assertFalse(self.workspace.delete_generated_object("chair"))
        self.assertEqual(
            [
                record.object_id
                for record in self.workspace.get_data().generated_objects
            ],
            ["chair"],
        )

        active_thread = self.workspace._generation_thread
        self.assertIsNotNone(active_thread)
        self.workspace.shutdown()
        planner.release.set()
        self.assertTrue(active_thread.wait(2000))
        _qt_application.processEvents()


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
