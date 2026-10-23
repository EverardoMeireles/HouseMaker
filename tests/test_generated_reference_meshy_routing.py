# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import trimesh
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.generation_jobs import GenerationJobManager
from housemaker.generation_state import (
    MASK_MODE_PAINT,
    GeneratedObjectRecord,
    MaskPoint,
    MaskStroke,
)
from housemaker.generation_views import _build_masked_frame_crop
from housemaker.generation_workspace import (
    DOOR_COMPONENT_PIPELINE_KEY,
    GenerationWorkspace,
    MeshyImagePlanner,
    MeshyTextureRegenerator,
    _materialize_texture_regeneration_preflight,
)
from housemaker.merged_generation_workspace import MergedGenerationWorkspace
from housemaker.meshy_generation import MeshyGenerationResult
from housemaker.surface_geometry import FixedSurface
from housemaker.surface_texture_workspace import SurfaceTextureGenerationWorkspace

# ### Shared Qt state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _generated_reference() -> np.ndarray:
    """Make generated pixels unmistakably different from the source video."""

    reference = np.zeros((64, 64, 4), dtype=np.uint8)
    reference[:, :, :3] = (220, 30, 160)
    reference[:, :, 1] += np.arange(64, dtype=np.uint8)[None, :]
    reference[:, :, 3] = 255
    return reference


def _write_source_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    if not writer.isOpened():
        raise unittest.SkipTest("MJPG video writing is unavailable.")
    try:
        for value in (40, 90):
            writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    finally:
        writer.release()


# ### Generated-reference routing regressions ###
class GeneratedReferenceMeshyRoutingTests(unittest.TestCase):
    """Capture provider-bound requests without starting any paid API jobs."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.objects_directory = self.root / "objects"
        self.objects_directory.mkdir()
        self.job_manager = GenerationJobManager()
        self.objects = GenerationWorkspace(
            asset_directory=self.objects_directory,
            job_manager=self.job_manager,
        )
        self.surfaces = SurfaceTextureGenerationWorkspace(
            asset_directory=self.root / "surfaces",
            application_settings=ApplicationSettingsStore(self.root / "settings.json"),
            job_manager=self.job_manager,
            shared_controls=self.objects.get_shared_controls(),
        )
        self.workspace = MergedGenerationWorkspace(
            self.surfaces, self.objects, self.job_manager
        )
        self.objects.set_runtime_settings(
            replace(self.objects._settings, meshy_api_key="mock-api-key")
        )
        self.surfaces.set_runtime_settings(
            replace(
                self.surfaces.get_runtime_settings(),
                meshy_api_key="mock-api-key",
            )
        )
        video_path = self.root / "source.avi"
        _write_source_video(video_path)
        self.objects.load_video(str(video_path))
        self.surfaces.load_video(str(video_path))
        self.workspace.video_view.set_strokes(
            [
                MaskStroke(MASK_MODE_PAINT, 0.08, (MaskPoint(0.2, 0.5),)),
                MaskStroke(MASK_MODE_PAINT, 0.08, (MaskPoint(0.8, 0.5),)),
            ]
        )
        self.reference = _generated_reference()
        self.workspace._activate_temporary_object_reference(
            self.reference, source_label="Generated reference"
        )

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.surfaces.shutdown()
        self.objects.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self.temporary_directory.cleanup()

    def _add_existing_object(
        self, *, door_slot: str | None = None
    ) -> GeneratedObjectRecord:
        asset_path = self.objects_directory / "fixture.glb"
        asset_path.write_bytes(trimesh.creation.box().export(file_type="glb"))
        record = GeneratedObjectRecord(
            object_id="existing-object",
            frame_index=0,
            object_name="Existing object",
            pipeline=(
                {}
                if door_slot is None
                else {
                    DOOR_COMPONENT_PIPELINE_KEY: {
                        "door_id": "door-1",
                        "slot_id": door_slot,
                    }
                }
            ),
            provider_task_id="mock-geometry-task",
            asset_path=asset_path.name,
        )
        self.objects._data.generated_objects.append(record)
        self.objects._selected_object_id = record.object_id
        return record

    def _assert_per_blob_generation_references(self, *, geometry_only: bool) -> None:
        reference_key = self.objects.current_inpaint_reference_key()
        frame = self.objects.video_view.get_frame_bgr()
        expected = {
            self.objects._inpaint_reference_key(0, mask): _build_masked_frame_crop(
                frame, mask, 0.08
            )
            for mask in self.objects._frame_inpaint_masks(0)
        }
        self.assertIn(reference_key, expected)
        expected[reference_key] = self.reference
        with patch.object(
            self.objects, "_start_generation", return_value="mock-job"
        ) as start:
            if geometry_only:
                self.objects.generate_geometry()
            else:
                self.objects.generate()

        self.assertEqual(start.call_count, 2)
        self.assertEqual(
            {call.args[0].inpaint_key for call in start.call_args_list}, set(expected)
        )
        for call in start.call_args_list:
            request = call.args[0]
            self.assertEqual(request.geometry_only, geometry_only)
            np.testing.assert_array_equal(
                request.selected_object_bgra, expected[request.inpaint_key]
            )

    def test_full_generate_keeps_corrected_and_raw_references_private_to_each_blob(self) -> None:
        self._assert_per_blob_generation_references(geometry_only=False)

    def test_geometry_generate_keeps_corrected_and_raw_references_private_to_each_blob(self) -> None:
        self._assert_per_blob_generation_references(geometry_only=True)

    def test_texture_generate_uses_reference_not_the_video_mask(self) -> None:
        self._add_existing_object()
        with patch.object(
            self.objects, "_start_texture_regeneration", return_value=True
        ) as start:
            self.assertTrue(self.objects.generate_selected_object_texture())

        start.assert_called_once()
        np.testing.assert_array_equal(
            start.call_args.args[0].reference_image_bgra, self.reference
        )

    def test_door_body_generate_keeps_the_generated_reference(self) -> None:
        self.assertTrue(
            self.objects.set_door_slot_editing_target(
                door_id="door-1",
                slot_id="body",
                object_id="door-body",
                object_name="Door body",
                slot_display_name="Body",
                generate_displacement=True,
                door_body_fit_dimensions=(0.9, 0.05, 2.1),
            )
        )
        with patch.object(
            self.objects, "_start_generation", return_value="mock-job"
        ) as start:
            self.objects.generate()

        start.assert_called_once()
        request = start.call_args.args[0]
        self.assertFalse(request.geometry_only)
        self.assertEqual(start.call_args.kwargs["door_slot_target"].slot_id, "body")
        np.testing.assert_array_equal(request.selected_object_bgra, self.reference)

    def test_handle_geometry_generate_keeps_the_generated_reference(self) -> None:
        self.assertTrue(
            self.objects.set_door_slot_editing_target(
                door_id="door-1",
                slot_id="door_knob",
                object_id="door-handle",
                object_name="Handle",
                slot_display_name="Handle",
            )
        )
        with patch.object(
            self.objects, "_start_generation", return_value="mock-job"
        ) as start:
            self.objects.generate_geometry()

        start.assert_called_once()
        request = start.call_args.args[0]
        self.assertTrue(request.geometry_only)
        self.assertEqual(
            start.call_args.kwargs["door_slot_target"].slot_id, "door_knob"
        )
        np.testing.assert_array_equal(request.selected_object_bgra, self.reference)

    def test_existing_handle_texture_uses_the_generated_reference(self) -> None:
        record = self._add_existing_object(door_slot="door_knob")
        self.assertTrue(
            self.objects.set_door_slot_editing_target(
                door_id="door-1",
                slot_id="door_knob",
                object_id=record.object_id,
                object_name="Handle",
                slot_display_name="Handle",
                defer_file_backed_preview=True,
            )
        )
        with patch.object(
            self.objects, "_start_texture_regeneration", return_value=True
        ) as start:
            self.assertTrue(self.objects.generate_selected_object_texture())

        start.assert_called_once()
        request = start.call_args.args[0]
        self.assertTrue(request.enable_original_uv)
        np.testing.assert_array_equal(request.reference_image_bgra, self.reference)

    def test_surface_generation_uses_the_same_generated_reference(self) -> None:
        mesh = trimesh.creation.box(extents=(1.0, 0.1, 1.0))
        surface = FixedSurface(
            surface_id=f"trim:{'a' * 32}/part:front:wall",
            surface_type="wall",
            level_index=0,
            room_index=None,
            mesh=mesh,
            area_square_meters=float(mesh.area),
        )
        self.surfaces.set_scene_surface_selection((surface,))

        request = self.surfaces._build_request(defer_reference_preparation=True)

        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(len(request.reference_pngs), 1)
        decoded = cv2.imdecode(
            np.frombuffer(request.reference_pngs[0], dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
        np.testing.assert_array_equal(decoded, self.reference)

    def test_leaving_the_frame_removes_the_reference_from_both_pipelines(self) -> None:
        self.workspace.seekbar.setValue(1)
        _qt_application.processEvents()

        self.assertFalse(self.objects.has_temporary_object_reference())
        self.assertFalse(self.surfaces.has_temporary_reference())
        self.assertFalse(self.workspace.video_view.has_reference_preview())

    def test_meshy_image_adapter_receives_the_generated_pixels(self) -> None:
        self.objects.set_runtime_settings(
            replace(
                self.objects._settings,
                unused_face_removal=False,
                use_uv_raycast_for_object_generation=False,
            )
        )
        provider_result = MeshyGenerationResult(
            task_id="mock-model-task",
            glb_bytes=trimesh.creation.box().export(file_type="glb"),
            name="Mock model",
        )
        duplicate_stats = SimpleNamespace(removed_face_count=0, duplicate_group_count=0)
        for geometry_only in (False, True):
            with self.subTest(geometry_only=geometry_only):
                request = self.objects._build_generation_request(
                    geometry_only=geometry_only
                )
                self.assertIsNotNone(request)
                assert request is not None
                with (
                    patch(
                        "housemaker.generation_workspace.request_image_to_3d_model",
                        return_value=provider_result,
                    ) as send,
                    patch(
                        "housemaker.generation_workspace._remove_safe_duplicates_from_meshy_result",
                        return_value=(provider_result, duplicate_stats),
                    ),
                    patch(
                        "housemaker.generation_workspace._scan_project_provider_result",
                        return_value=provider_result,
                    ),
                ):
                    MeshyImagePlanner().plan(request)

                send.assert_called_once()
                decoded = cv2.imdecode(
                    np.frombuffer(send.call_args.kwargs["image_png"], dtype=np.uint8),
                    cv2.IMREAD_UNCHANGED,
                )
                np.testing.assert_array_equal(decoded, self.reference)

    def test_meshy_retexture_adapter_receives_the_generated_pixels(self) -> None:
        self._add_existing_object()
        preflight = self.objects._build_texture_regeneration_request()
        self.assertIsNotNone(preflight)
        assert preflight is not None
        materialized = _materialize_texture_regeneration_preflight(
            preflight, self.objects_directory
        )
        provider_result = MeshyGenerationResult(
            task_id="mock-texture-task",
            glb_bytes=materialized.request.model_glb,
            name="Mock textured model",
        )

        with patch(
            "housemaker.generation_workspace.request_retextured_model",
            return_value=provider_result,
        ) as send:
            MeshyTextureRegenerator().regenerate(materialized.request)

        send.assert_called_once()
        reference_images = send.call_args.kwargs["reference_images_png"]
        self.assertEqual(len(reference_images), 1)
        decoded = cv2.imdecode(
            np.frombuffer(reference_images[0], dtype=np.uint8), cv2.IMREAD_UNCHANGED
        )
        np.testing.assert_array_equal(decoded, self.reference)


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
