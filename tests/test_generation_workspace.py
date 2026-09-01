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
from unittest.mock import Mock, patch

import cv2
import numpy as np
import trimesh
from PIL import Image
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from housemaker.generation_state import (
    MASK_MODE_ERASE,
    MASK_MODE_PAINT,
    GeneratedObjectPlacement,
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
    LOCALLY_AUTHORED_UVS_PIPELINE_KEY,
    SCAN_PROJECTION_PIPELINE_KEY,
    ScanProjectedMeshyGenerationResult,
    StagedMeshyGenerationResult,
    TEXTURE_VARIANTS_PIPELINE_KEY,
    TEXTURE_VARIANT_GLB_PATH_KEY,
    VISIBILITY_UV_UNWRAP_PIPELINE_KEY,
    _ObjectGenerationProgressMapper,
    _GenerationCancelled,
    _build_geometry_fingerprint,
    _build_staged_generation_pipeline_metadata,
    _build_texture_resolution_entries,
    _format_model_statistics,
    _collect_model_uv_triangles,
    _resolve_staged_postprocessed_asset_path,
    _staged_generation_mode,
    update_projection_camera_percentage,
)
from housemaker.glb import GeneratedModel, import_generated_glb
from housemaker.meshy_generation import MeshyGenerationResult
from housemaker.object_uv_raycast import (
    UV_TARGET_DOMAIN_FULL,
    VISIBILITY_UV_UNWRAP_VERSION,
    VisibilityUvUnwrapStats,
)
from housemaker.object_uv_scan_projection import (
    DEFAULT_PROJECTION_CAMERA_PERCENTAGES,
    SCAN_PROJECTION_TARGET_FULL,
    SCAN_PROJECTION_VERSION,
    ScanProjectionCancelled,
    ScanProjectionResult,
    ScanProjectionStats,
)
from housemaker.object_texture_variants import (
    PBR_MAP_METALLIC,
    PBR_MAP_NORMAL,
    PBR_MAP_ROUGHNESS,
    PBR_MAP_TYPES,
)
from housemaker.settings_widget import GenerationServiceSettings
from housemaker.texture_atlas_view import TextureAtlasEntry
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
def _test_model(
    extents: tuple[float, float, float] = (1.0, 0.5, 0.75),
) -> GeneratedModel:
    mesh = trimesh.creation.box(extents=extents)
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
    )


def _test_visibility_uv_stats() -> VisibilityUvUnwrapStats:
    return VisibilityUvUnwrapStats(
        face_count=12,
        instance_face_count=12,
        chart_count=6,
        exterior_face_count=10,
        hidden_face_count=2,
        camera_count=14,
        ray_sample_count=1_024,
        texture_resolution=2_048,
        gutter_pixels=8,
        effective_gutter_pixels=8.0,
        atlas_width=2_048,
        atlas_height=2_048,
        atlas_utilization=0.86,
        requested_exterior_uv_share=0.95,
        achieved_exterior_uv_share=0.94,
        uv_triangle_occupancy=0.82,
        exterior_face_indices=tuple(range(10)),
        visibility_hits=(1,) * 10 + (0, 0),
    )


def _test_scan_projection_stats() -> ScanProjectionStats:
    return ScanProjectionStats(
        version=SCAN_PROJECTION_VERSION,
        camera_percentages=DEFAULT_PROJECTION_CAMERA_PERCENTAGES,
        view_face_counts=(2, 2, 2, 2, 2, 2),
        view_pixel_counts=(170, 170, 170, 170, 160, 160),
        face_count=12,
        output_face_count=12,
        source_vertex_count=8,
        output_vertex_count=36,
        texture_resolution=2_048,
        target_domain=SCAN_PROJECTION_TARGET_FULL,
        target_width=2_048,
        target_height=2_048,
        island_padding_pixels=0,
        outer_safety_inset_pixels=0,
        usable_pixel_count=1_000,
        covered_pixel_count=990,
        triangle_occupancy=0.99,
    )


def _test_uv_glb() -> bytes:
    """Build a textured quad for UV-preview coverage."""

    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        dtype=float,
    )
    uv = np.asarray(
        (
            (0.1, 0.1),
            (0.9, 0.1),
            (0.9, 0.9),
            (0.1, 0.9),
        ),
        dtype=float,
    )
    faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
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

    def test_paint_erase_clear_and_object_crop(self) -> None:
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

    def test_right_click_fills_closed_outline_as_one_action(
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
    def test_geometry_fingerprint_ignores_face_order_but_detects_surface_change(
        self,
    ) -> None:
        source = _test_model()
        reordered_mesh = source.mesh.copy()
        reordered_mesh.faces = reordered_mesh.faces[::-1, ::-1]
        reordered_glb = bytes(
            trimesh.Scene(reordered_mesh).export(file_type="glb")
        )
        changed = _test_model((1.01, 0.5, 0.75))

        self.assertEqual(
            _build_geometry_fingerprint(source.glb_bytes),
            _build_geometry_fingerprint(reordered_glb),
        )
        self.assertNotEqual(
            _build_geometry_fingerprint(source.glb_bytes),
            _build_geometry_fingerprint(changed.glb_bytes),
        )

    def test_staged_progress_reserves_separate_geometry_and_texture_phases(
        self,
    ) -> None:
        request = GenerationRequest(
            frame_index=0,
            selected_object_bgra=np.zeros((4, 4, 4), dtype=np.uint8),
            settings=GenerationServiceSettings(
                meshy_api_key="key",
                unused_face_removal=True,
            ),
        )
        mapper = _ObjectGenerationProgressMapper(request)

        geometry_complete = mapper.map_provider_message(
            "Meshy is generating: 100%"
        )
        texture_submitted = mapper.map_provider_message(
            "Submitting Meshy texture task..."
        )
        texture_started = mapper.map_provider_message(
            "Meshy is texturing: 0%"
        )
        texture_complete = mapper.map_provider_message(
            "Meshy is texturing: 100%"
        )

        self.assertIn("48%", geometry_complete)
        self.assertIn("56%", texture_submitted)
        self.assertIn("56%", texture_started)
        self.assertIn("80%", texture_complete)

    def test_weighted_projection_uses_one_normal_meshy_progress_range(
        self,
    ) -> None:
        request = GenerationRequest(
            frame_index=0,
            selected_object_bgra=np.zeros((4, 4, 4), dtype=np.uint8),
            settings=GenerationServiceSettings(
                meshy_api_key="key",
                use_uv_raycast_for_object_generation=True,
            ),
        )
        mapper = _ObjectGenerationProgressMapper(request)

        generation_complete = mapper.map_provider_message(
            "Meshy is generating: 100%"
        )

        self.assertIn("80%", generation_complete)

    def test_symmetric_weighted_projection_uses_normal_meshy_progress(
        self,
    ) -> None:
        request = GenerationRequest(
            frame_index=0,
            selected_object_bgra=np.zeros((4, 4, 4), dtype=np.uint8),
            settings=GenerationServiceSettings(
                meshy_api_key="key",
                use_uv_raycast_for_object_generation=True,
            ),
            symmetric_division_enabled=True,
        )
        mapper = _ObjectGenerationProgressMapper(request)

        generation_complete = mapper.map_provider_message(
            "Meshy is generating: 100%"
        )

        self.assertIn("80%", generation_complete)

    def test_uv_raycast_stage_persists_versioned_provenance(self) -> None:
        result = replace(
            _test_staged_meshy_result(),
            visibility_uv_stats=_test_visibility_uv_stats(),
        )
        variant_metadata = {
            "2048": {
                TEXTURE_VARIANT_GLB_PATH_KEY: "chair.texture-2048.glb",
            }
        }

        pipeline = _build_staged_generation_pipeline_metadata(
            result,
            "chair.geometry.glb",
        )
        postprocessed_path = _resolve_staged_postprocessed_asset_path(
            result,
            "chair.texture-1024.glb",
            variant_metadata,
        )

        self.assertTrue(pipeline[LOCALLY_AUTHORED_UVS_PIPELINE_KEY])
        provenance = pipeline[VISIBILITY_UV_UNWRAP_PIPELINE_KEY]
        self.assertIsInstance(provenance, dict)
        assert isinstance(provenance, dict)
        self.assertEqual(
            provenance["version"],
            VISIBILITY_UV_UNWRAP_VERSION,
        )
        self.assertEqual(provenance["exterior_face_count"], 10)
        self.assertEqual(provenance["hidden_face_count"], 2)
        self.assertEqual(provenance["requested_exterior_uv_share"], 0.95)
        self.assertEqual(provenance["target_domain"], UV_TARGET_DOMAIN_FULL)
        self.assertEqual(
            provenance["packing_strategy"],
            "rotate_and_align_charts",
        )
        self.assertEqual(postprocessed_path, "chair.texture-2048.glb")
        self.assertEqual(
            _resolve_staged_postprocessed_asset_path(
                replace(result, geometry_only=True),
                "chair.glb",
                None,
            ),
            "chair.glb",
        )

    def test_weighted_scan_stage_persists_zero_padding_provenance(self) -> None:
        result = replace(
            _test_staged_meshy_result(),
            scan_projection_stats=_test_scan_projection_stats(),
        )
        variant_metadata = {
            "2048": {
                TEXTURE_VARIANT_GLB_PATH_KEY: "chair.texture-2048.glb",
            }
        }

        pipeline = _build_staged_generation_pipeline_metadata(
            result,
            "chair.geometry.glb",
        )

        self.assertTrue(pipeline[LOCALLY_AUTHORED_UVS_PIPELINE_KEY])
        provenance = pipeline[SCAN_PROJECTION_PIPELINE_KEY]
        self.assertIsInstance(provenance, dict)
        assert isinstance(provenance, dict)
        self.assertEqual(provenance["version"], SCAN_PROJECTION_VERSION)
        self.assertEqual(
            tuple(provenance["camera_percentages"].values()),
            DEFAULT_PROJECTION_CAMERA_PERCENTAGES,
        )
        self.assertEqual(provenance["island_padding_pixels"], 0)
        self.assertEqual(provenance["triangle_occupancy"], 0.99)
        self.assertEqual(
            _staged_generation_mode(result),
            "unused_face_removal_and_weighted_camera_scan_projection",
        )
        self.assertEqual(
            _resolve_staged_postprocessed_asset_path(
                result,
                "chair.texture-1024.glb",
                variant_metadata,
            ),
            "chair.texture-2048.glb",
        )

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

    def test_weighted_projection_scans_one_normal_textured_meshy_result(
        self,
    ) -> None:
        request = GenerationRequest(
            frame_index=3,
            selected_object_bgra=np.zeros((7, 11, 4), dtype=np.uint8),
            settings=GenerationServiceSettings(
                meshy_api_key="meshy-test-key",
                use_uv_raycast_for_object_generation=True,
            ),
        )
        provider_result = MeshyGenerationResult(
            task_id="texture-task",
            glb_bytes=b"provider-textured-glb",
            name="Textured chair",
        )
        scan_result = ScanProjectionResult(
            glb_bytes=b"scan-projected-glb",
            stats=_test_scan_projection_stats(),
        )
        events: list[str] = []

        def fake_image_to_3d(**kwargs: object) -> MeshyGenerationResult:
            events.append("image_to_3d")
            self.assertNotIn("should_texture", kwargs)
            return provider_result

        def fake_scan_projection(
            glb_bytes: bytes,
            percentages: tuple[int, ...],
            **kwargs: object,
        ) -> ScanProjectionResult:
            events.append("scan_projection")
            self.assertEqual(glb_bytes, provider_result.glb_bytes)
            self.assertEqual(
                percentages,
                DEFAULT_PROJECTION_CAMERA_PERCENTAGES,
            )
            self.assertEqual(
                kwargs["target_domain"],
                SCAN_PROJECTION_TARGET_FULL,
            )
            return scan_result

        with (
            patch(
                "housemaker.generation_workspace.request_image_to_3d_model",
                side_effect=fake_image_to_3d,
            ),
            patch(
                "housemaker.generation_workspace.scan_project_textured_glb",
                side_effect=fake_scan_projection,
            ),
        ):
            result = MeshyImagePlanner().plan(request)

        self.assertEqual(events, ["image_to_3d", "scan_projection"])
        self.assertNotIsInstance(result, StagedMeshyGenerationResult)
        self.assertEqual(result.task_id, provider_result.task_id)
        self.assertEqual(result.name, provider_result.name)
        self.assertEqual(result.glb_bytes, scan_result.glb_bytes)
        self.assertIs(result.scan_projection_stats, scan_result.stats)

    def test_symmetric_weighted_projection_is_deferred_until_half_is_known(
        self,
    ) -> None:
        request = GenerationRequest(
            frame_index=3,
            selected_object_bgra=np.zeros((7, 11, 4), dtype=np.uint8),
            settings=GenerationServiceSettings(
                meshy_api_key="meshy-test-key",
                use_uv_raycast_for_object_generation=True,
            ),
            symmetric_division_enabled=True,
            symmetric_division_orientation="vertical",
        )
        provider_result = MeshyGenerationResult(
            task_id="image-to-3d-task",
            glb_bytes=b"provider-textured-glb",
            name="Textured chair",
        )

        with (
            patch(
                "housemaker.generation_workspace.request_image_to_3d_model",
                return_value=provider_result,
            ) as image_to_3d,
            patch(
                "housemaker.generation_workspace."
                "scan_project_textured_glb",
            ) as scan_projection,
            patch(
                "housemaker.generation_workspace.request_retextured_model",
            ) as retexture,
        ):
            result = MeshyImagePlanner().plan(request)

        self.assertIs(result, provider_result)
        self.assertNotIsInstance(result, StagedMeshyGenerationResult)
        image_to_3d.assert_called_once()
        self.assertNotIn("should_texture", image_to_3d.call_args.kwargs)
        scan_projection.assert_not_called()
        retexture.assert_not_called()

    def test_textured_staged_result_builds_normal_object_variants(self) -> None:
        staged = replace(
            _test_staged_meshy_result(),
            glb_bytes=_test_model().glb_bytes,
        )
        expected_variants = object()

        with patch(
            "housemaker.generation_workspace.build_object_texture_variants",
            return_value=expected_variants,
        ) as build_regular_variants:
            model = MeshyModelExecutor().execute(staged)

        build_regular_variants.assert_called_once_with(staged.glb_bytes)
        self.assertIs(model.object_texture_variants, expected_variants)

    def test_weighted_projection_does_not_modify_geometry_only_output(
        self,
    ) -> None:
        request = GenerationRequest(
            frame_index=0,
            selected_object_bgra=np.zeros((4, 4, 4), dtype=np.uint8),
            settings=GenerationServiceSettings(
                meshy_api_key="meshy-key",
                use_uv_raycast_for_object_generation=True,
            ),
            geometry_only=True,
        )
        geometry_result = MeshyGenerationResult(
            task_id="geometry-task",
            glb_bytes=b"geometry-glb",
            name="Geometry",
        )
        with (
            patch(
                "housemaker.generation_workspace.request_image_to_3d_model",
                return_value=geometry_result,
            ) as image_to_3d,
            patch(
                "housemaker.generation_workspace.scan_project_textured_glb",
            ) as scan_projection,
            patch(
                "housemaker.generation_workspace.request_retextured_model"
            ) as retexture_mock,
        ):
            result = MeshyImagePlanner().plan(request)

        self.assertFalse(image_to_3d.call_args.kwargs["should_texture"])
        scan_projection.assert_not_called()
        retexture_mock.assert_not_called()
        self.assertIsInstance(result, StagedMeshyGenerationResult)
        assert isinstance(result, StagedMeshyGenerationResult)
        self.assertTrue(result.geometry_only)
        self.assertEqual(result.glb_bytes, geometry_result.glb_bytes)
        self.assertEqual(
            result.postprocessed_glb_bytes,
            geometry_result.glb_bytes,
        )
        self.assertIsNone(result.scan_projection_stats)

    def test_weighted_projection_cancellation_discards_provider_result(
        self,
    ) -> None:
        request = GenerationRequest(
            frame_index=0,
            selected_object_bgra=np.zeros((4, 4, 4), dtype=np.uint8),
            settings=GenerationServiceSettings(
                meshy_api_key="meshy-key",
                use_uv_raycast_for_object_generation=True,
            ),
        )

        with (
            patch(
                "housemaker.generation_workspace.request_image_to_3d_model",
                return_value=_test_meshy_result(),
            ),
            patch(
                "housemaker.generation_workspace.scan_project_textured_glb",
                side_effect=ScanProjectionCancelled("cancelled"),
            ) as scan_projection,
        ):
            with self.assertRaises(_GenerationCancelled):
                MeshyImagePlanner().plan(request)

        scan_projection.assert_called_once()

    def test_symmetric_face_removal_defers_weighted_scan_projection(
        self,
    ) -> None:
        request = GenerationRequest(
            frame_index=0,
            selected_object_bgra=np.zeros((4, 4, 4), dtype=np.uint8),
            settings=GenerationServiceSettings(
                meshy_api_key="meshy-key",
                unused_face_removal=True,
                use_uv_raycast_for_object_generation=True,
            ),
            symmetric_division_enabled=True,
        )
        geometry_model = _test_model()
        retained_model = _test_model()
        textured_model = _test_model()
        geometry_result = MeshyGenerationResult(
            task_id="geometry-task",
            glb_bytes=geometry_model.glb_bytes,
            name="Geometry",
        )
        textured_result = MeshyGenerationResult(
            task_id="texture-task",
            glb_bytes=textured_model.glb_bytes,
            name="Textured chair",
        )
        removal_result = UnusedFaceRemovalResult(
            model=retained_model,
            enabled_camera_ids=ALL_CAMERA_IDS,
            original_face_count=12,
            retained_face_count=10,
            removed_face_count=2,
            protected_face_count=10,
        )
        final_removal_result = UnusedFaceRemovalResult(
            model=textured_model,
            enabled_camera_ids=ALL_CAMERA_IDS,
            original_face_count=12,
            retained_face_count=12,
            removed_face_count=0,
            protected_face_count=12,
        )

        with (
            patch(
                "housemaker.generation_workspace.request_image_to_3d_model",
                return_value=geometry_result,
            ) as image_to_3d,
            patch(
                "housemaker.generation_workspace.remove_unused_faces_from_glb",
                side_effect=(removal_result, final_removal_result),
            ) as removal_mock,
            patch(
                "housemaker.generation_workspace.scan_project_textured_glb",
            ) as scan_projection,
            patch(
                "housemaker.generation_workspace.request_retextured_model",
                return_value=textured_result,
            ) as retexture,
        ):
            result = MeshyImagePlanner().plan(request)

        self.assertFalse(image_to_3d.call_args.kwargs["should_texture"])
        self.assertEqual(removal_mock.call_count, 2)
        self.assertEqual(
            removal_mock.call_args_list[1].args[0],
            textured_model.glb_bytes,
        )
        self.assertEqual(
            removal_mock.call_args_list[0].kwargs[
                "options"
            ].minimum_visible_fraction,
            0.05,
        )
        scan_projection.assert_not_called()
        retexture.assert_called_once()
        self.assertEqual(
            retexture.call_args.kwargs["model_glb"],
            retained_model.glb_bytes,
        )
        self.assertFalse(retexture.call_args.kwargs["enable_original_uv"])
        self.assertIsInstance(result, StagedMeshyGenerationResult)
        assert isinstance(result, StagedMeshyGenerationResult)
        self.assertEqual(result.removed_face_count, 2)
        self.assertEqual(result.glb_bytes, textured_model.glb_bytes)
        self.assertEqual(result.postprocessed_glb_bytes, retained_model.glb_bytes)
        self.assertTrue(result.final_face_removal_applied)
        self.assertEqual(result.final_removed_face_count, 0)
        self.assertFalse(result.retexture_topology_changed)
        self.assertIsNone(result.scan_projection_stats)

    def test_staged_planner_generates_geometry_removes_faces_then_textures(
        self,
    ) -> None:
        selected = np.zeros((9, 13, 4), dtype=np.uint8)
        selected[2:8, 3:11] = (30, 90, 170, 255)
        request = GenerationRequest(
            frame_index=4,
            selected_object_bgra=selected,
            settings=GenerationServiceSettings(
                meshy_api_key="meshy-staged-key",
                meshy_target_polycount=5_400,
                unused_face_removal=True,
                minimum_face_visibility_percentage=7,
            ),
        )
        geometry_model = _test_model()
        processed_model = _test_model()
        final_model = _test_model((1.05, 0.5, 0.75))
        final_cleaned_mesh = final_model.mesh.copy()
        final_cleaned_mesh.update_faces(
            np.arange(len(final_cleaned_mesh.faces) - 1)
        )
        final_cleaned_mesh.remove_unreferenced_vertices()
        final_cleaned_scene = trimesh.Scene(final_cleaned_mesh)
        final_cleaned_model = GeneratedModel(
            mesh=final_cleaned_mesh,
            scene=final_cleaned_scene,
            glb_bytes=bytes(final_cleaned_scene.export(file_type="glb")),
        )
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
            enabled_camera_ids=ALL_CAMERA_IDS,
            original_face_count=12,
            retained_face_count=9,
            removed_face_count=3,
            protected_face_count=9,
        )
        final_removal_result = UnusedFaceRemovalResult(
            model=final_cleaned_model,
            enabled_camera_ids=ALL_CAMERA_IDS,
            original_face_count=12,
            retained_face_count=11,
            removed_face_count=1,
            protected_face_count=11,
            visibility_removed_face_count=1,
        )
        image_calls: list[dict[str, object]] = []
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
            return (
                removal_result
                if len(removal_calls) == 1
                else final_removal_result
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
        self.assertEqual(removal_calls[0][0], geometry_model.glb_bytes)
        self.assertEqual(removal_calls[1][0], final_model.glb_bytes)
        removal_options = removal_calls[0][1]["options"]
        self.assertEqual(
            removal_options.enabled_camera_ids,  # type: ignore[union-attr]
            ALL_CAMERA_IDS,
        )
        self.assertEqual(
            removal_options.minimum_visible_fraction,  # type: ignore[union-attr]
            0.07,
        )
        self.assertIs(removal_calls[1][1]["options"], removal_options)
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
        self.assertEqual(result.minimum_face_visibility_percentage, 7)
        self.assertTrue(result.final_face_removal_applied)
        self.assertEqual(result.final_removed_face_count, 1)
        self.assertEqual(result.final_visibility_removed_face_count, 1)
        self.assertTrue(result.retexture_topology_changed)
        self.assertEqual(result.glb_bytes, final_removal_result.glb_bytes)
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


# ### Projection camera percentage tests ###
class ProjectionCameraPercentageTests(unittest.TestCase):
    def test_decrease_preserves_free_capacity_for_later_increases(self) -> None:
        decreased = update_projection_camera_percentage(
            DEFAULT_PROJECTION_CAMERA_PERCENTAGES,
            ALL_CAMERA_IDS[0],
            16,
        )

        self.assertEqual(decreased, (16, 17, 17, 17, 16, 16))
        self.assertEqual(sum(decreased), 99)
        self.assertEqual(
            update_projection_camera_percentage(
                decreased,
                ALL_CAMERA_IDS[1],
                18,
            ),
            (16, 18, 17, 17, 16, 16),
        )

    def test_overflow_dilutes_larger_cameras_proportionally(self) -> None:
        updated = update_projection_camera_percentage(
            (10, 40, 25, 10, 10, 5),
            ALL_CAMERA_IDS[0],
            20,
        )

        self.assertEqual(updated, (20, 35, 22, 9, 9, 5))
        self.assertEqual(sum(updated), 100)

    def test_rounding_is_deterministic_and_every_camera_keeps_one_percent(
        self,
    ) -> None:
        self.assertEqual(
            update_projection_camera_percentage(
                DEFAULT_PROJECTION_CAMERA_PERCENTAGES,
                ALL_CAMERA_IDS[0],
                18,
            ),
            (18, 16, 17, 17, 16, 16),
        )
        self.assertEqual(
            update_projection_camera_percentage(
                (94, 2, 1, 1, 1, 1),
                ALL_CAMERA_IDS[0],
                100,
            ),
            (95, 1, 1, 1, 1, 1),
        )

    def test_rejects_invalid_live_states_and_unknown_cameras(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 1"):
            update_projection_camera_percentage(
                (20, 20, 20, 20, 20, 0),
                ALL_CAMERA_IDS[0],
                19,
            )
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            update_projection_camera_percentage(
                (20, 20, 20, 20, 20, 1),
                ALL_CAMERA_IDS[0],
                19,
            )
        with self.assertRaisesRegex(ValueError, "Unknown projection camera"):
            update_projection_camera_percentage(
                DEFAULT_PROJECTION_CAMERA_PERCENTAGES,
                "diagonal",
                10,
            )


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

    def test_face_purge_and_legacy_camera_checkboxes_are_absent(self) -> None:
        panel = self.workspace.object_3d_panel

        self.assertFalse(hasattr(panel, "unused_face_camera_controls"))
        self.assertFalse(hasattr(panel, "unused_face_camera_checkboxes"))
        self.assertFalse(hasattr(panel, "get_enabled_postprocess_camera_ids"))
        self.assertFalse(hasattr(self.workspace, "purge_faces_button"))
        self.assertFalse(
            hasattr(self.workspace, "purge_selected_object_faces")
        )

    def test_weighted_camera_allocations_require_exact_total_and_snapshot(
        self,
    ) -> None:
        panel = self.workspace.object_3d_panel
        controls = panel.projection_camera_percentage_spinboxes

        self.assertTrue(
            panel.viewer.get_projection_camera_indicators_visible()
        )
        self.assertEqual(tuple(controls), ALL_CAMERA_IDS)
        self.assertEqual(
            panel.get_projection_camera_percentages(),
            DEFAULT_PROJECTION_CAMERA_PERCENTAGES,
        )
        self.assertTrue(panel.projection_camera_percentages_are_valid())
        self.assertEqual(
            panel.projection_camera_total_label.text(),
            "Total: 100%",
        )
        self.assertTrue(
            all(
                control.minimum() == 1 and control.maximum() == 95
                for control in controls.values()
            )
        )

        self.workspace.set_runtime_settings(
            GenerationServiceSettings(
                meshy_api_key="meshy-key",
                use_uv_raycast_for_object_generation=True,
            )
        )
        controls[ALL_CAMERA_IDS[0]].setValue(
            DEFAULT_PROJECTION_CAMERA_PERCENTAGES[0] - 1
        )
        _qt_application.processEvents()
        self.assertFalse(panel.projection_camera_percentages_are_valid())
        self.assertEqual(
            panel.get_projection_camera_percentages(),
            (16, 17, 17, 17, 16, 16),
        )
        self.assertIn(
            "must equal 100%",
            panel.projection_camera_total_label.text(),
        )

        request_patches = (
            patch.object(
                self.workspace.video_view,
                "get_frame_bgr",
                return_value=np.zeros((8, 8, 3), dtype=np.uint8),
            ),
            patch.object(
                self.workspace.video_view,
                "has_selection",
                return_value=True,
            ),
            patch.object(
                self.workspace.video_view,
                "build_selected_object_crop",
                return_value=np.full((4, 4, 4), 255, dtype=np.uint8),
            ),
        )
        with request_patches[0], request_patches[1], request_patches[2]:
            self.assertIsNone(self.workspace._build_generation_request())
            geometry_request = self.workspace._build_generation_request(
                geometry_only=True
            )
        assert geometry_request is not None
        self.assertEqual(
            geometry_request.projection_camera_percentages,
            DEFAULT_PROJECTION_CAMERA_PERCENTAGES,
        )

        controls[ALL_CAMERA_IDS[0]].setValue(
            DEFAULT_PROJECTION_CAMERA_PERCENTAGES[0]
        )
        _qt_application.processEvents()
        expected_snapshot = panel.get_projection_camera_percentages()
        self.assertEqual(sum(expected_snapshot), 100)
        with (
            patch.object(
                self.workspace.video_view,
                "get_frame_bgr",
                return_value=np.zeros((8, 8, 3), dtype=np.uint8),
            ),
            patch.object(
                self.workspace.video_view,
                "has_selection",
                return_value=True,
            ),
            patch.object(
                self.workspace.video_view,
                "build_selected_object_crop",
                return_value=np.full((4, 4, 4), 255, dtype=np.uint8),
            ),
        ):
            request = self.workspace._build_generation_request()
        assert request is not None
        controls[ALL_CAMERA_IDS[0]].setValue(
            DEFAULT_PROJECTION_CAMERA_PERCENTAGES[0] + 1
        )
        self.assertEqual(
            panel.get_projection_camera_percentages(),
            (18, 16, 17, 17, 16, 16),
        )
        self.assertEqual(
            request.projection_camera_percentages,
            expected_snapshot,
        )

    def test_camera_fields_and_viewport_steps_share_percentage_pipeline(
        self,
    ) -> None:
        panel = self.workspace.object_3d_panel
        changed_spy = QSignalSpy(panel.projection_camera_percentages_changed)

        with patch.object(
            panel.viewer,
            "set_projection_camera_percentages",
            wraps=panel.viewer.set_projection_camera_percentages,
        ) as viewer_sync:
            panel.projection_camera_percentage_spinboxes[
                ALL_CAMERA_IDS[0]
            ].setValue(18)
            field_percentages = panel.get_projection_camera_percentages()

            self.assertEqual(field_percentages, (18, 16, 17, 17, 16, 16))
            viewer_sync.assert_called_once_with(field_percentages)

            panel.viewer.projection_camera_percentage_step_requested.emit(
                ALL_CAMERA_IDS[0],
                -1,
            )
            stepped_percentages = panel.get_projection_camera_percentages()

        self.assertEqual(stepped_percentages, (17, 16, 17, 17, 16, 16))
        self.assertEqual(
            tuple(
                panel.projection_camera_percentage_spinboxes[
                    camera_id
                ].value()
                for camera_id in ALL_CAMERA_IDS
            ),
            stepped_percentages,
        )
        self.assertEqual(changed_spy.count(), 2)
        self.assertEqual(viewer_sync.call_args_list[-1].args, (stepped_percentages,))

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
            self.assertFalse(
                hasattr(meshy_planner.requests[0], "enabled_camera_ids")
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
            self.assertNotIn("enabled_camera_ids", record.pipeline)
            self.assertNotIn("unchecked_camera_ids", record.pipeline)
            self.assertNotIn("camera_face_purge_applied", record.pipeline)
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

    def test_pbr_map_checkboxes_use_three_rows_in_stable_map_order(self) -> None:
        layout = self.workspace.pbr_map_control.layout()

        self.assertIsNotNone(layout)
        self.assertEqual(layout.rowCount(), 3)
        self.assertEqual(layout.columnCount(), 1)
        self.assertEqual(tuple(self.workspace.pbr_map_checkboxes), PBR_MAP_TYPES)
        expected_labels = ("Normal", "Roughness", "Metallic")
        for row, (map_type, expected_label) in enumerate(
            zip(PBR_MAP_TYPES, expected_labels, strict=True)
        ):
            checkbox = self.workspace.pbr_map_checkboxes[map_type]
            item = layout.itemAtPosition(row, 0)
            self.assertIsNotNone(item)
            assert item is not None
            self.assertIs(item.widget(), checkbox)
            self.assertEqual(checkbox.text(), expected_label)
            self.assertFalse(checkbox.isChecked())

    def test_pbr_toggle_updates_viewer_without_reloading_its_model(self) -> None:
        model = _test_textured_model((70, 100, 130, 255))
        viewer = self.workspace.result_view
        viewer.set_model(model)

        with (
            patch.object(
                viewer,
                "set_pbr_maps_enabled",
                wraps=viewer.set_pbr_maps_enabled,
            ) as apply_pbr_maps,
            patch.object(viewer, "set_model") as reload_model,
        ):
            self.workspace.pbr_map_checkboxes[PBR_MAP_NORMAL].setChecked(True)
            self.workspace.pbr_map_checkboxes[PBR_MAP_METALLIC].setChecked(True)

        self.assertEqual(apply_pbr_maps.call_count, 2)
        self.assertEqual(
            apply_pbr_maps.call_args.args,
            ((PBR_MAP_NORMAL, PBR_MAP_METALLIC),),
        )
        reload_model.assert_not_called()
        self.assertIs(viewer.model, model)
        self.assertEqual(
            viewer.get_pbr_maps_enabled(),
            {
                PBR_MAP_NORMAL: True,
                PBR_MAP_ROUGHNESS: False,
                PBR_MAP_METALLIC: True,
            },
        )

    def test_enabled_pbr_map_is_snapshotted_and_requests_meshy_pbr(self) -> None:
        self.workspace.set_runtime_settings(
            GenerationServiceSettings(meshy_api_key="meshy-key")
        )
        roughness_checkbox = self.workspace.pbr_map_checkboxes[
            PBR_MAP_ROUGHNESS
        ]
        roughness_checkbox.setChecked(True)
        with (
            patch.object(
                self.workspace.video_view,
                "get_frame_bgr",
                return_value=np.zeros((8, 8, 3), dtype=np.uint8),
            ),
            patch.object(
                self.workspace.video_view,
                "has_selection",
                return_value=True,
            ),
            patch.object(
                self.workspace.video_view,
                "build_selected_object_crop",
                return_value=np.full((4, 4, 4), 255, dtype=np.uint8),
            ),
        ):
            request = self.workspace._build_generation_request()

        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.enabled_pbr_maps, (PBR_MAP_ROUGHNESS,))
        roughness_checkbox.setChecked(False)
        self.workspace.pbr_map_checkboxes[PBR_MAP_NORMAL].setChecked(True)
        self.assertEqual(request.enabled_pbr_maps, (PBR_MAP_ROUGHNESS,))

        provider_result = _test_meshy_result("PBR chair")
        with patch(
            "housemaker.generation_workspace.request_image_to_3d_model",
            return_value=provider_result,
        ) as request_model:
            result = MeshyImagePlanner().plan(request)

        self.assertIs(result, provider_result)
        self.assertIs(request_model.call_args.kwargs["enable_pbr"], True)

    def test_glass_conversion_snapshots_faces_maps_and_button_prerequisites(
        self,
    ) -> None:
        button = self.workspace.convert_faces_to_glass_button
        self.assertFalse(button.isEnabled())

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            asset_directory = temporary_path / "generation_assets"
            asset_directory.mkdir()
            self.workspace._asset_directory = asset_directory
            self.workspace._video_source = Mock()
            with (
                patch.object(
                    self.workspace.video_view,
                    "get_frame_bgr",
                    return_value=np.zeros((8, 8, 3), dtype=np.uint8),
                ),
                patch.object(
                    self.workspace.video_view,
                    "has_selection",
                    return_value=True,
                ),
                patch.object(
                    self.workspace.video_view,
                    "build_selected_object_crop",
                    return_value=np.full((4, 4, 4), 255, dtype=np.uint8),
                ),
            ):
                self.workspace.set_runtime_settings(
                    GenerationServiceSettings(meshy_api_key="meshy-key")
                )
                model = _test_textured_model((80, 110, 140, 255))
                self.workspace._handle_generation_succeeded(
                    MeshyGenerationResult(
                        "task-test-chair",
                        model.glb_bytes,
                        "Glass cabinet",
                    ),
                    model,
                )

                self.assertFalse(button.isEnabled())
                self.workspace.result_view.set_selected_face_indices((4, 1))
                _qt_application.processEvents()
                self.assertTrue(button.isEnabled())

                self.workspace.set_runtime_settings(GenerationServiceSettings())
                self.assertFalse(button.isEnabled())
                self.workspace.set_runtime_settings(
                    GenerationServiceSettings(meshy_api_key="meshy-key")
                )
                self.assertTrue(button.isEnabled())

                with patch.object(
                    self.workspace,
                    "_start_texture_regeneration",
                    return_value=True,
                ) as start_regeneration:
                    converted = self.workspace.convert_selected_faces_to_glass()

        self.assertTrue(converted)
        start_regeneration.assert_called_once()
        request = start_regeneration.call_args.args[0]
        self.assertEqual(request.glass_face_indices, (1, 4))
        self.assertEqual(request.enabled_pbr_maps, PBR_MAP_TYPES)
        self.assertTrue(request.enable_original_uv)
        self.assertEqual(
            start_regeneration.call_args.kwargs,
            {"requested_name": ""},
        )
        self.assertTrue(
            all(
                checkbox.isChecked()
                for checkbox in self.workspace.pbr_map_checkboxes.values()
            )
        )
        self.assertEqual(
            self.workspace.result_view.get_pbr_maps_enabled(),
            {map_type: True for map_type in PBR_MAP_TYPES},
        )

    def test_glass_conversion_requires_existing_complete_uvs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            asset_directory = Path(temporary_directory) / "generation_assets"
            asset_directory.mkdir()
            self.workspace._asset_directory = asset_directory
            self.workspace._video_source = Mock()
            with (
                patch.object(
                    self.workspace.video_view,
                    "get_frame_bgr",
                    return_value=np.zeros((8, 8, 3), dtype=np.uint8),
                ),
                patch.object(
                    self.workspace.video_view,
                    "has_selection",
                    return_value=True,
                ),
                patch.object(
                    self.workspace.video_view,
                    "build_selected_object_crop",
                    return_value=np.full((4, 4, 4), 255, dtype=np.uint8),
                ),
            ):
                self.workspace.set_runtime_settings(
                    GenerationServiceSettings(meshy_api_key="meshy-key")
                )
                model = _test_model()
                self.workspace._handle_generation_succeeded(
                    _test_meshy_result("Untextured cabinet"),
                    model,
                )
                self.workspace.result_view.set_selected_face_indices((0,))
                _qt_application.processEvents()

                self.assertFalse(
                    self.workspace.convert_faces_to_glass_button.isEnabled()
                )
                self.assertFalse(
                    self.workspace.convert_selected_faces_to_glass()
                )
                self.assertIn(
                    "requires a textured object with UVs",
                    self.workspace.status_label.text(),
                )

    def test_model_uv_triangles_are_collected_per_face_for_texture_preview(
        self,
    ) -> None:
        model = import_generated_glb(_test_uv_glb())

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

    def test_scan_projected_success_persists_uv_authority_metadata(self) -> None:
        model = _test_model()
        stats = _test_scan_projection_stats()
        result = ScanProjectedMeshyGenerationResult(
            task_id="scan-task",
            glb_bytes=model.glb_bytes,
            name="Scanned chair",
            scan_projection_stats=stats,
        )

        self.workspace._handle_generation_succeeded(result, model)

        record = self.workspace.get_data().generated_objects[0]
        self.assertTrue(record.pipeline[LOCALLY_AUTHORED_UVS_PIPELINE_KEY])
        self.assertEqual(
            record.pipeline[SCAN_PROJECTION_PIPELINE_KEY],
            stats.to_pipeline_dict(),
        )

    def test_visibility_uv_geometry_only_reuses_saved_asset_as_revision(
        self,
    ) -> None:
        uv_glb = _test_uv_glb()
        result = StagedMeshyGenerationResult(
            task_id="geometry-task",
            glb_bytes=uv_glb,
            name="UV chair",
            geometry_task_id="geometry-task",
            source_glb_bytes=_test_model().glb_bytes,
            postprocessed_glb_bytes=uv_glb,
            visibility_uv_stats=_test_visibility_uv_stats(),
            geometry_only=True,
        )

        self.workspace._handle_generation_succeeded(
            result,
            import_generated_glb(uv_glb),
        )

        record = self.workspace.get_data().generated_objects[0]
        self.assertEqual(
            record.pipeline["postprocessed_asset_path"],
            record.asset_path,
        )
        self.assertTrue(record.pipeline[LOCALLY_AUTHORED_UVS_PIPELINE_KEY])
        self.assertEqual(
            record.pipeline[VISIBILITY_UV_UNWRAP_PIPELINE_KEY]["version"],
            VISIBILITY_UV_UNWRAP_VERSION,
        )
        saved_glbs = tuple(
            self.workspace._asset_directory.glob(f"{record.object_id}*.glb")
        )
        self.assertEqual(len(saved_glbs), 2)
        self.assertFalse(
            any(path.name.endswith(".postprocessed.glb") for path in saved_glbs)
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

    def test_unchanged_object_preview_refresh_preserves_resources(
        self,
    ) -> None:
        model = _test_model()
        self.workspace._handle_generation_succeeded(
            _test_meshy_result(),
            model,
        )
        record = self.workspace._data.generated_objects[-1]
        original_item = self.workspace.generated_objects_list.item(0)

        with (
            patch.object(self.workspace.result_view, "clear_model") as clear,
            patch.object(self.workspace.result_view, "set_model") as set_model,
            patch(
                "housemaker.generation_workspace._collect_model_uv_triangles"
            ) as collect_uvs,
            patch(
                "housemaker.generation_workspace."
                "_build_texture_resolution_entries"
            ) as build_texture_entries,
        ):
            self.workspace.refresh_file_backed_previews()

        self.assertIs(
            self.workspace.generated_objects_list.item(0),
            original_item,
        )
        clear.assert_not_called()
        set_model.assert_not_called()
        collect_uvs.assert_not_called()
        build_texture_entries.assert_not_called()

    def test_in_place_preview_record_change_invalidates_display_snapshot(
        self,
    ) -> None:
        model = _test_model()
        self.workspace._handle_generation_succeeded(
            _test_meshy_result(),
            model,
        )
        record = self.workspace._data.generated_objects[-1]
        record.pipeline["preview_revision"] = 2

        with (
            patch.object(self.workspace.result_view, "clear_model") as clear,
            patch.object(self.workspace.result_view, "set_model") as set_model,
            patch.object(
                self.workspace,
                "_refresh_object_texture_atlases",
            ) as refresh_textures,
        ):
            self.workspace._refresh_generated_objects_list(record.object_id)

        clear.assert_called_once_with()
        set_model.assert_called_once_with(model)
        refresh_textures.assert_called_once_with(record.object_id)

    def test_same_path_glb_replacement_reloads_preview_and_signature(
        self,
    ) -> None:
        model = _test_model()
        self.workspace._handle_generation_succeeded(
            _test_meshy_result(),
            model,
        )
        record = self.workspace._data.generated_objects[-1]
        placed_record = replace(
            record,
            placement=GeneratedObjectPlacement(
                level_index=0,
                image_x=20.0,
                image_y=30.0,
            ),
        )
        self.workspace._data.generated_objects[-1] = placed_record
        original_signature = (
            self.workspace.get_placed_preview_dependency_signature()
        )
        asset_path = self.workspace._resolve_meshy_asset_path(
            placed_record.asset_path
        )
        replacement_scene = trimesh.Scene(
            trimesh.creation.icosphere(subdivisions=1)
        )
        asset_path.write_bytes(replacement_scene.export(file_type="glb"))
        current_stat = asset_path.stat()
        os.utime(
            asset_path,
            ns=(current_stat.st_atime_ns, current_stat.st_mtime_ns + 1_000_000),
        )

        with patch(
            "housemaker.generation_workspace.import_generated_glb",
            wraps=import_generated_glb,
        ) as import_model:
            self.workspace.refresh_file_backed_previews()

        import_model.assert_called_once()
        self.assertIsNot(self.workspace.result_view.model, model)
        self.assertNotEqual(
            self.workspace.get_placed_preview_dependency_signature(),
            original_signature,
        )

    def test_model_cache_retries_when_glb_changes_during_read(self) -> None:
        first_model = _test_model()
        second_model = _test_model()
        self.workspace._handle_generation_succeeded(
            _test_meshy_result(),
            first_model,
        )
        record = self.workspace._data.generated_objects[-1]
        self.workspace._generated_model_cache.clear()
        self.workspace._generated_model_cache_revisions.clear()
        revision_before = ("asset.glb", 10, 100, 200)
        revision_after = ("asset.glb", 10, 101, 201)

        with (
            patch(
                "housemaker.generation_workspace."
                "_build_generation_asset_revision",
                side_effect=(
                    revision_before,
                    revision_after,
                    revision_after,
                    revision_after,
                ),
            ),
            patch(
                "housemaker.generation_workspace.import_generated_glb",
                side_effect=(first_model, second_model),
            ) as import_model,
        ):
            loaded_model = self.workspace._load_generated_object_model(record)

        self.assertIs(loaded_model, second_model)
        self.assertEqual(import_model.call_count, 2)
        self.assertIs(
            self.workspace._generated_model_cache[record.object_id],
            second_model,
        )
        self.assertEqual(
            self.workspace._generated_model_cache_revisions[record.object_id],
            revision_after,
        )

    def test_texture_resolution_entries_are_cached_by_content_revision(
        self,
    ) -> None:
        model = _test_model()
        self.workspace._handle_generation_succeeded(
            _test_meshy_result(),
            model,
        )
        record = self.workspace._data.generated_objects[-1]
        self.workspace._texture_resolution_entry_cache.clear()

        with patch(
            "housemaker.generation_workspace._build_texture_resolution_entries",
            wraps=_build_texture_resolution_entries,
        ) as build_entries:
            self.workspace._refresh_object_texture_atlases(record.object_id)
            self.workspace._refresh_object_texture_atlases(record.object_id)
            record.pipeline[TEXTURE_VARIANTS_PIPELINE_KEY] = {}
            self.workspace._refresh_object_texture_atlases(record.object_id)

        self.assertEqual(build_entries.call_count, 2)

    def test_known_missing_texture_variant_is_cached_until_reappearance(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            asset_directory = Path(temporary_directory)
            self.workspace._asset_directory = asset_directory
            model = _test_model()
            valid_glb = "cache-512.glb"
            valid_png = "cache-512.png"
            missing_glb = "cache-1024.glb"
            missing_png = "cache-1024.png"
            (asset_directory / valid_glb).write_bytes(model.glb_bytes)
            Image.new("RGBA", (32, 32), (30, 70, 110, 255)).save(
                asset_directory / valid_png
            )
            record = GeneratedObjectRecord(
                object_id="partial-cache",
                frame_index=0,
                object_name="Partial cache",
                pipeline={
                    TEXTURE_VARIANTS_PIPELINE_KEY: {
                        "512": {
                            "glb_asset_path": valid_glb,
                            "texture_asset_path": valid_png,
                        },
                        "1024": {
                            "glb_asset_path": missing_glb,
                            "texture_asset_path": missing_png,
                        },
                    },
                    "selected_texture_resolution": 512,
                },
                provider_task_id="partial-cache-task",
                asset_path=valid_glb,
            )
            self.workspace._data.generated_objects = [record]

            with patch(
                "housemaker.generation_workspace."
                "_build_texture_resolution_entries",
                wraps=_build_texture_resolution_entries,
            ) as build_entries:
                self.workspace._refresh_object_texture_atlases(record.object_id)
                self.workspace._refresh_object_texture_atlases(record.object_id)
                self.assertEqual(build_entries.call_count, 1)
                self.assertEqual(len(self.workspace.texture_view.entries), 1)

                (asset_directory / missing_glb).write_bytes(model.glb_bytes)
                Image.new("RGBA", (32, 32), (110, 70, 30, 255)).save(
                    asset_directory / missing_png
                )
                self.workspace._refresh_object_texture_atlases(record.object_id)

            self.assertEqual(build_entries.call_count, 2)
            self.assertEqual(len(self.workspace.texture_view.entries), 2)

    def test_activation_does_not_repair_a_temporarily_missing_selection(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            asset_directory = Path(temporary_directory)
            self.workspace._asset_directory = asset_directory
            model = _test_model()
            variants: dict[str, dict[str, str]] = {}
            for resolution, color in (
                (512, (30, 70, 110, 255)),
                (1024, (110, 70, 30, 255)),
            ):
                glb_name = f"selection-{resolution}.glb"
                png_name = f"selection-{resolution}.png"
                (asset_directory / glb_name).write_bytes(model.glb_bytes)
                Image.new("RGBA", (32, 32), color).save(
                    asset_directory / png_name
                )
                variants[str(resolution)] = {
                    "glb_asset_path": glb_name,
                    "texture_asset_path": png_name,
                }
            record = GeneratedObjectRecord(
                object_id="selection-cache",
                frame_index=0,
                object_name="Selection cache",
                pipeline={
                    TEXTURE_VARIANTS_PIPELINE_KEY: variants,
                    "selected_texture_resolution": 1024,
                },
                provider_task_id="selection-cache-task",
                asset_path="selection-1024.glb",
            )
            self.workspace.set_data(
                GenerationData(generated_objects=[record])
            )
            selected_glb = asset_directory / "selection-1024.glb"
            selected_png = asset_directory / "selection-1024.png"
            glb_payload = selected_glb.read_bytes()
            png_payload = selected_png.read_bytes()
            selected_glb.unlink()
            selected_png.unlink()

            self.workspace.refresh_file_backed_previews()
            missing_record = self.workspace._data.generated_objects[0]
            self.assertEqual(
                missing_record.pipeline["selected_texture_resolution"],
                1024,
            )
            self.assertEqual(missing_record.asset_path, "selection-1024.glb")

            selected_glb.write_bytes(glb_payload)
            selected_png.write_bytes(png_payload)
            self.workspace.refresh_file_backed_previews()

            restored_record = self.workspace._data.generated_objects[0]
            self.assertEqual(
                restored_record.pipeline["selected_texture_resolution"],
                1024,
            )
            self.assertEqual(restored_record.asset_path, "selection-1024.glb")
            self.assertIsNotNone(self.workspace.result_view.model)

    def test_failed_object_thumbnail_decode_retries_same_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            asset_directory = Path(temporary_directory)
            self.workspace._asset_directory = asset_directory
            model = _test_model()
            glb_path = "retry-thumbnail.glb"
            png_path = "retry-thumbnail.png"
            (asset_directory / glb_path).write_bytes(model.glb_bytes)
            Image.new("RGBA", (32, 32), (30, 70, 110, 255)).save(
                asset_directory / png_path
            )
            record = GeneratedObjectRecord(
                object_id="retry-thumbnail",
                frame_index=0,
                object_name="Retry thumbnail",
                pipeline={
                    TEXTURE_VARIANTS_PIPELINE_KEY: {
                        "512": {
                            "glb_asset_path": glb_path,
                            "texture_asset_path": png_path,
                        },
                    },
                    "selected_texture_resolution": 512,
                },
                provider_task_id="retry-thumbnail-task",
                asset_path=glb_path,
            )
            self.workspace._data.generated_objects = [record]
            failed_once = False

            def build_entry(*args, **kwargs):
                nonlocal failed_once
                if not failed_once:
                    failed_once = True
                    raise ValueError("temporary thumbnail decode failure")
                return TextureAtlasEntry(*args, **kwargs)

            with patch(
                "housemaker.generation_workspace.TextureAtlasEntry",
                side_effect=build_entry,
            ) as entry_builder:
                self.workspace._refresh_object_texture_atlases(record.object_id)
                self.assertNotIn(
                    record.object_id,
                    self.workspace._texture_resolution_entry_cache,
                )
                self.workspace._refresh_object_texture_atlases(record.object_id)

            self.assertEqual(entry_builder.call_count, 2)
            self.assertEqual(len(self.workspace.texture_view.entries), 1)
            self.assertIn(
                record.object_id,
                self.workspace._texture_resolution_entry_cache,
            )

    def test_unvalidated_model_is_not_bound_to_a_newer_file_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            asset_directory = Path(temporary_directory)
            self.workspace._asset_directory = asset_directory
            cached_model = _test_model()
            replacement_model = _test_textured_model((20, 80, 140, 255))
            record = GeneratedObjectRecord(
                object_id="cache-race",
                frame_index=0,
                object_name="Cache race",
                pipeline={},
                provider_task_id="cache-race-task",
                asset_path="cache-race.glb",
            )
            asset_path = asset_directory / record.asset_path
            asset_path.write_bytes(replacement_model.glb_bytes)

            self.workspace._cache_generated_model(record, cached_model)

            self.assertNotIn(
                record.object_id,
                self.workspace._generated_model_cache,
            )
            asset_path.write_bytes(cached_model.glb_bytes)
            self.workspace._cache_generated_model(record, cached_model)
            self.assertIs(
                self.workspace._generated_model_cache[record.object_id],
                cached_model,
            )

    def test_generated_object_view_uses_fixed_maximum_ambient_light(self) -> None:
        self.assertFalse(hasattr(self.workspace, "ambient_light_slider"))
        self.assertAlmostEqual(
            self.workspace.result_view.get_ambient_light_intensity(),
            1.0,
        )

    def test_paint_and_erase_controls_are_stacked_vertically(self) -> None:
        layout = self.workspace.mask_mode_control.layout()

        self.assertIsNotNone(layout)
        self.assertEqual(layout.getContentsMargins(), (0, 0, 0, 0))
        self.assertEqual(layout.spacing(), 0)
        self.assertEqual(layout.count(), 2)
        self.assertIs(layout.itemAt(0).widget(), self.workspace.paint_mask_button)
        self.assertIs(layout.itemAt(1).widget(), self.workspace.erase_mask_button)

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

    def test_unrelated_delete_remains_available_while_generating(self) -> None:
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
        self.assertTrue(
            self.workspace.delete_generated_object_button.isEnabled()
        )
        self.assertTrue(self.workspace.delete_generated_object("chair"))
        self.assertEqual(
            [
                record.object_id
                for record in self.workspace.get_data().generated_objects
            ],
            [],
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
