# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import io
import shutil
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.camera_models import CameraPose, InitialFirstPersonCamera
from housemaker.generation_state import MASK_MODE_PAINT, MaskPoint, MaskStroke
from housemaker.models import LevelData, RoomData, VertexData
from housemaker.settings_widget import (
    SURFACE_TEXTURE_PROVIDER_GPT_4O_MINI,
    SURFACE_TEXTURE_PROVIDER_GPT_5_6_LUNA,
    SURFACE_TEXTURE_PROVIDER_GPT_5_6_TERRA,
    SURFACE_TEXTURE_PROVIDER_MESHY,
    SURFACE_TEXTURE_PROVIDER_SETTING_KEY,
    GenerationServiceSettings,
)
from housemaker.surface_texture_providers import SurfaceTextureResult
from housemaker.surface_texture_state import (
    SurfaceTextureAssignment,
    SurfaceTextureData,
)
from housemaker.surface_texture_workspace import (
    DefaultSurfaceTextureProvider,
    SurfaceTextureGenerationWorkspace,
    SurfaceTextureRequest,
    _build_masked_crop,
    _build_surface_texture_outputs,
    _decode_png_rgba,
    _encode_png,
    _encode_rgba_png,
)


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _test_level() -> LevelData:
    vertex_data = VertexData()
    for point_x, point_y in (
        (0.0, 0.0),
        (100.0, 0.0),
        (100.0, 100.0),
        (0.0, 100.0),
        (50.0, 50.0),
    ):
        vertex_data.add_vertex(point_x, point_y)
    for start_vertex_id, end_vertex_id in ((1, 2), (2, 3), (3, 4), (4, 1)):
        vertex_data.add_edge(start_vertex_id, end_vertex_id)
    return LevelData(
        index=2,
        name="Ground",
        vertex_data=vertex_data,
        rooms=[
            RoomData(
                name="Room",
                vertex_ids=(1, 2, 3, 4),
                center_vertex_id=5,
                color_rgb=(120, 150, 180),
            )
        ],
        image_size_pixels=(100.0, 100.0),
        floor_contour_vertex_ids=(1, 2, 3, 4),
    )


def _test_stroke(x: float = 0.5) -> MaskStroke:
    return MaskStroke(
        mode=MASK_MODE_PAINT,
        radius_normalized=0.12,
        points=(MaskPoint(x=x, y=0.5),),
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
                    (20 + frame_index * 30, 70, 150),
                    dtype=np.uint8,
                )
            )
    finally:
        writer.release()


def _texture_png() -> bytes:
    did_encode, encoded = cv2.imencode(
        ".png",
        np.full((8, 12, 3), (30, 120, 210), dtype=np.uint8),
    )
    if not did_encode:
        raise AssertionError("Unable to build PNG fixture")
    return bytes(encoded)


def _colored_texture_png(color: tuple[int, int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGBA", (10, 7), color).save(output, format="PNG")
    return output.getvalue()


def _set_current_mask(
    workspace: SurfaceTextureGenerationWorkspace,
    stroke: MaskStroke,
) -> None:
    workspace.video_view.set_strokes([stroke])
    workspace._handle_video_strokes_changed([stroke])


# ### Provider fixtures ###
class _FakeProvider:
    def __init__(self) -> None:
        self.requests: list[SurfaceTextureRequest] = []
        self.result = SurfaceTextureResult(
            provider=SURFACE_TEXTURE_PROVIDER_GPT_5_6_LUNA,
            texture_png=_texture_png(),
            task_id="texture-task-1",
        )

    def generate(self, request: SurfaceTextureRequest) -> SurfaceTextureResult:
        self.requests.append(request)
        return self.result


class _BlockingProvider:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def generate(self, _request: SurfaceTextureRequest) -> SurfaceTextureResult:
        self.started.set()
        if not self.release.wait(timeout=5.0):
            raise RuntimeError("Blocking surface provider timed out")
        return SurfaceTextureResult(
            provider="meshy",
            texture_png=_texture_png(),
            task_id="late-task",
        )


# ### Provider integration tests ###
class SurfaceTextureDefaultProviderIntegrationTests(unittest.TestCase):
    def test_default_adapter_forwards_partial_texture_edit_inputs(self) -> None:
        base = _texture_png()
        edit_mask = _texture_png()
        request = SurfaceTextureRequest(
            provider=SURFACE_TEXTURE_PROVIDER_MESHY,
            api_key="meshy-test-key",
            reference_pngs=(_texture_png(),),
            reference_frame_indices=(4,),
            surface_type="wall",
            surface_ids=("wall-one",),
            combined_area_m2=2.5,
            prompt="Repair the marked plaster",
            existing_texture_png=base,
            edit_mask_png=edit_mask,
        )
        expected = SurfaceTextureResult(
            provider=SURFACE_TEXTURE_PROVIDER_MESHY,
            texture_png=_texture_png(),
        )

        with patch(
            "housemaker.surface_texture_workspace.request_surface_texture",
            return_value=expected,
        ) as generate:
            result = DefaultSurfaceTextureProvider().generate(request)

        self.assertIs(result, expected)
        generate.assert_called_once_with(
            provider=SURFACE_TEXTURE_PROVIDER_MESHY,
            api_key="meshy-test-key",
            reference_pngs=request.reference_pngs,
            prompt=request.prompt,
            existing_texture_png=base,
            edit_mask_png=edit_mask,
            progress_callback=None,
            cancel_event=None,
        )


# ### Workspace tests ###
class SurfaceTextureGenerationWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_path = Path(tempfile.mkdtemp())
        self.provider = _FakeProvider()
        self.initial_camera = InitialFirstPersonCamera(
            level_index=2,
            pose=CameraPose(
                x=0.5,
                y=0.75,
                z=1.7,
                yaw_degrees=35.0,
                pitch_degrees=-4.0,
            ),
        )
        self.workspace = SurfaceTextureGenerationWorkspace(
            provider=self.provider,
            asset_directory=self._temporary_path / "surface_assets",
            application_settings=ApplicationSettingsStore(
                self._temporary_path / "settings.json"
            ),
        )
        self.workspace.set_levels([_test_level()], self.initial_camera)
        self.workspace.resize(1000, 700)
        self.workspace.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        shutil.rmtree(self._temporary_path, ignore_errors=True)

    def test_provider_selector_has_exact_options_and_persists_selection(self) -> None:
        combo = self.workspace.surface_texture_provider_combo
        options = [
            (combo.itemText(index), combo.itemData(index))
            for index in range(combo.count())
        ]
        self.assertEqual(
            options,
            [
                ("Meshy", SURFACE_TEXTURE_PROVIDER_MESHY),
                ("GPT-4o-mini", SURFACE_TEXTURE_PROVIDER_GPT_4O_MINI),
                ("GPT-5.6 Luna", SURFACE_TEXTURE_PROVIDER_GPT_5_6_LUNA),
                ("GPT-5.6 Terra", SURFACE_TEXTURE_PROVIDER_GPT_5_6_TERRA),
            ],
        )

        terra_index = combo.findData(SURFACE_TEXTURE_PROVIDER_GPT_5_6_TERRA)
        combo.setCurrentIndex(terra_index)

        settings_store = ApplicationSettingsStore(
            self._temporary_path / "settings.json"
        )
        self.assertEqual(
            settings_store.get(SURFACE_TEXTURE_PROVIDER_SETTING_KEY),
            SURFACE_TEXTURE_PROVIDER_GPT_5_6_TERRA,
        )
        self.assertEqual(
            self.workspace.get_runtime_settings().surface_texture_provider,
            SURFACE_TEXTURE_PROVIDER_GPT_5_6_TERRA,
        )

        restored = SurfaceTextureGenerationWorkspace(
            asset_directory=self._temporary_path / "restored_assets",
            application_settings=settings_store,
        )
        try:
            self.assertEqual(
                restored.surface_texture_provider_combo.currentData(),
                SURFACE_TEXTURE_PROVIDER_GPT_5_6_TERRA,
            )
        finally:
            restored.shutdown()
            restored.close()

    def test_equal_views_initial_camera_and_homogeneous_shift_selection(self) -> None:
        view_sizes = self.workspace.views_splitter.sizes()
        self.assertEqual(len(view_sizes), 2)
        self.assertLessEqual(abs(view_sizes[0] - view_sizes[1]), 2)
        self.assertEqual(
            self.workspace.surface_view.get_camera_pose(),
            self.initial_camera.pose,
        )

        viewer = self.workspace.surface_view
        first_wall = "level:2/room:5/wall:1:2"
        second_wall = "level:2/room:5/wall:2:3"
        floor = "level:2/room:5/floor"
        self.assertTrue(viewer.select_surface(first_wall))
        self.assertTrue(viewer.select_surface(second_wall, shift_pressed=True))
        self.assertEqual(viewer.get_selected_surface_ids(), (first_wall, second_wall))

        self.assertFalse(viewer.select_surface(floor, shift_pressed=True))
        self.assertEqual(viewer.get_selected_surface_ids(), (first_wall, second_wall))
        self.assertTrue(viewer.select_surface(first_wall, shift_pressed=True))
        self.assertEqual(viewer.get_selected_surface_ids(), (second_wall,))
        self.assertTrue(viewer.select_surface(floor))
        self.assertEqual(viewer.get_selected_surface_ids(), (floor,))
        self.assertEqual(viewer.get_selected_surface_type(), "floor")

    def test_video_view_is_left_of_fixed_surfaces_view(self) -> None:
        splitter = self.workspace.views_splitter

        self.assertIs(splitter.widget(0), self.workspace.video_view.parentWidget())
        self.assertIs(splitter.widget(1), self.workspace.right_view_stack)
        self.assertIs(
            self.workspace.surface_view.parentWidget(),
            self.workspace.surface_3d_page,
        )
        self.assertLessEqual(abs(splitter.sizes()[0] - splitter.sizes()[1]), 2)

    def test_texture_view_tracks_latest_selected_surface_assignment(self) -> None:
        asset_directory = self._temporary_path / "surface_assets"
        asset_directory.mkdir(exist_ok=True)
        first_png = _colored_texture_png((210, 20, 35, 255))
        second_png = _colored_texture_png((25, 190, 50, 255))
        (asset_directory / "first.png").write_bytes(first_png)
        (asset_directory / "second.png").write_bytes(second_png)
        surface_id = "level:2/room:5/floor"
        self.workspace.set_data(
            SurfaceTextureData(
                selected_surface_type="floor",
                selected_surface_ids=(surface_id,),
                assignments=[
                    SurfaceTextureAssignment(
                        assignment_id="first-atlas",
                        surface_type="floor",
                        surface_ids=(surface_id,),
                        provider="meshy",
                        asset_path="first.png",
                    ),
                    SurfaceTextureAssignment(
                        assignment_id="second-atlas",
                        surface_type="floor",
                        surface_ids=(surface_id,),
                        provider="meshy",
                        asset_path="second.png",
                    ),
                ],
            )
        )

        self.assertEqual(len(self.workspace.texture_view.entries), 2)
        self.assertEqual(
            self.workspace.texture_view.selected_atlas_id,
            "second-atlas",
        )
        selected_image = self.workspace.texture_view.selected_entry.get_image()
        self.assertEqual(selected_image.pixelColor(0, 0).green(), 190)

        self.workspace.set_external_3d_viewer_active(True)
        self.assertIs(
            self.workspace.right_view_stack.currentWidget(),
            self.workspace.texture_view_page,
        )
        self.workspace.set_external_3d_viewer_active(False)
        self.assertIs(
            self.workspace.right_view_stack.currentWidget(),
            self.workspace.surface_3d_page,
        )

    def test_texture_view_falls_back_to_latest_valid_surface_atlas(self) -> None:
        asset_directory = self._temporary_path / "surface_assets"
        asset_directory.mkdir(exist_ok=True)
        (asset_directory / "valid.png").write_bytes(
            _colored_texture_png((85, 120, 210, 255))
        )
        surface_id = "level:2/room:5/floor"
        self.workspace.set_data(
            SurfaceTextureData(
                selected_surface_type="floor",
                selected_surface_ids=(surface_id,),
                assignments=[
                    SurfaceTextureAssignment(
                        assignment_id="valid-atlas",
                        surface_type="floor",
                        surface_ids=(surface_id,),
                        provider="meshy",
                        asset_path="valid.png",
                    ),
                    SurfaceTextureAssignment(
                        assignment_id="missing-atlas",
                        surface_type="floor",
                        surface_ids=(surface_id,),
                        provider="meshy",
                        asset_path="missing.png",
                    ),
                ],
            )
        )

        self.assertEqual(
            [entry.atlas_id for entry in self.workspace.texture_view.entries],
            ["valid-atlas"],
        )
        self.assertEqual(
            self.workspace.texture_view.selected_atlas_id,
            "valid-atlas",
        )
        cached_entry = self.workspace.texture_view.entries[0]

        self.workspace._handle_surface_selection_changed((surface_id,))

        self.assertIs(self.workspace.texture_view.entries[0], cached_entry)

    def test_multiframe_masks_build_request_with_selected_surface_area(self) -> None:
        video_path = self._temporary_path / "walkthrough.avi"
        _write_test_video(video_path)
        self.workspace.load_video(str(video_path))
        _set_current_mask(self.workspace, _test_stroke(0.25))
        self.workspace.show_frame(2)
        _set_current_mask(self.workspace, _test_stroke(0.75))
        self.workspace.show_frame(0)

        self.assertEqual(
            self.workspace.video_view.get_strokes(),
            [_test_stroke(0.25)],
        )
        self.workspace.surface_view.select_surface("level:2/room:5/floor")
        self.workspace.material_notes_edit.setText("pale oak")
        self.workspace.set_runtime_settings(
            GenerationServiceSettings(
                openai_api_key="openai-test-key",
                surface_texture_provider=SURFACE_TEXTURE_PROVIDER_GPT_5_6_LUNA,
            )
        )

        request = self.workspace._build_request()

        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.provider, SURFACE_TEXTURE_PROVIDER_GPT_5_6_LUNA)
        self.assertEqual(request.api_key, "openai-test-key")
        self.assertEqual(request.reference_frame_indices, (0, 2))
        self.assertEqual(len(request.reference_pngs), 2)
        self.assertTrue(all(image.startswith(b"\x89PNG") for image in request.reference_pngs))
        self.assertEqual(request.surface_type, "floor")
        self.assertEqual(request.surface_ids, ("level:2/room:5/floor",))
        self.assertAlmostEqual(request.combined_area_m2, 4.0)
        self.assertIn("pale oak", request.prompt)

    def test_3d_inpaint_request_targets_only_painted_surfaces(self) -> None:
        video_path = self._temporary_path / "partial-walkthrough.avi"
        _write_test_video(video_path, frame_count=1)
        self.workspace.load_video(str(video_path))
        _set_current_mask(self.workspace, _test_stroke())
        first_wall = "level:2/room:5/wall:1:2"
        second_wall = "level:2/room:5/wall:2:3"
        texture = np.full((32, 32, 4), (15, 35, 75, 255), dtype=np.uint8)
        self.workspace.surface_view.set_surface_texture(
            (first_wall, second_wall),
            texture,
        )
        self.workspace.surface_view.set_selected_surface_ids(
            (first_wall, second_wall)
        )
        self.workspace.surface_view.add_texture_mask_stroke(
            first_wall,
            _test_stroke(0.25),
        )
        self.workspace.set_runtime_settings(
            GenerationServiceSettings(
                openai_api_key="openai-test-key",
                surface_texture_provider=SURFACE_TEXTURE_PROVIDER_GPT_5_6_LUNA,
            )
        )

        request = self.workspace._build_request()

        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.surface_ids, (first_wall,))
        self.assertIsNotNone(request.existing_texture_png)
        self.assertIsNotNone(request.edit_mask_png)
        self.assertEqual(
            [surface_id for surface_id, _mask in request.surface_edit_mask_pngs],
            [first_wall],
        )
        self.assertAlmostEqual(request.combined_area_m2, 6.0)

    def test_async_result_is_persisted_applied_and_restored(self) -> None:
        video_path = self._temporary_path / "walkthrough.avi"
        _write_test_video(video_path, frame_count=1)
        self.workspace.load_video(str(video_path))
        _set_current_mask(self.workspace, _test_stroke())
        surface_id = "level:2/room:5/floor"
        self.workspace.surface_view.select_surface(surface_id)
        self.workspace.set_runtime_settings(
            GenerationServiceSettings(
                openai_api_key="openai-test-key",
                surface_texture_provider=SURFACE_TEXTURE_PROVIDER_GPT_5_6_LUNA,
            )
        )
        completed = QSignalSpy(self.workspace.generation_completed)

        self.workspace.generate()
        deadline = time.monotonic() + 3.0
        while completed.count() == 0 and time.monotonic() < deadline:
            _qt_application.processEvents()
            QTest.qWait(5)
        while self.workspace.is_generating and time.monotonic() < deadline:
            _qt_application.processEvents()
            QTest.qWait(5)

        self.assertEqual(completed.count(), 1)
        self.assertFalse(self.workspace.is_generating)
        self.assertEqual(len(self.provider.requests), 1)
        saved_data = self.workspace.get_data()
        self.assertEqual(len(saved_data.assignments), 1)
        assignment = saved_data.assignments[0]
        self.assertEqual(assignment.surface_ids, (surface_id,))
        self.assertEqual(assignment.provider_task_id, "texture-task-1")
        self.assertAlmostEqual(assignment.combined_area_m2, 4.0)
        self.assertEqual(assignment.reference_frame_indices, (0,))
        self.assertEqual((assignment.texture_width, assignment.texture_height), (12, 8))
        self.assertEqual(
            (self._temporary_path / "surface_assets" / assignment.asset_path).read_bytes(),
            _texture_png(),
        )
        self.assertIn(surface_id, self.workspace.surface_view._surface_textures)

        restored = SurfaceTextureGenerationWorkspace(
            asset_directory=self._temporary_path / "surface_assets"
        )
        try:
            restored.set_levels([_test_level()], self.initial_camera)
            restored.set_data(saved_data)
            self.assertIn(surface_id, restored.surface_view._surface_textures)
        finally:
            restored.shutdown()
            restored.close()

    def test_legacy_room_index_assignment_is_not_retargeted_on_restore(self) -> None:
        asset_path = self._temporary_path / "surface_assets" / "legacy.png"
        asset_path.parent.mkdir(parents=True)
        asset_path.write_bytes(_texture_png())
        legacy_assignment = SurfaceTextureAssignment(
            assignment_id="legacy-floor",
            surface_type="floor",
            surface_ids=("level:2/room:0/floor",),
            provider="meshy",
            asset_path=asset_path.name,
        )

        self.workspace.set_data(
            SurfaceTextureData(assignments=[legacy_assignment])
        )

        self.assertEqual(self.workspace.surface_view._surface_textures, {})
        self.assertNotIn(
            "level:2/room:5/floor",
            self.workspace.surface_view._surface_textures,
        )

    def test_shutdown_discards_inflight_result_and_busy_state_blocks_replacement(self) -> None:
        blocker = _BlockingProvider()
        self.workspace.set_provider(blocker)
        request = SurfaceTextureRequest(
            provider="meshy",
            api_key="meshy-test-key",
            reference_pngs=(_texture_png(),),
            reference_frame_indices=(0,),
            surface_type="floor",
            surface_ids=("level:2/room:5/floor",),
            combined_area_m2=4.0,
            prompt="Test floor material",
        )
        completed = QSignalSpy(self.workspace.generation_completed)

        self.workspace._start_generation(request)
        self.assertTrue(blocker.started.wait(timeout=1.0))
        active_thread = self.workspace._generation_thread
        self.assertIsNotNone(active_thread)
        with self.assertRaises(RuntimeError):
            self.workspace.set_data(SurfaceTextureData())
        with self.assertRaises(RuntimeError):
            self.workspace.load_video(str(self._temporary_path / "unused.avi"))

        self.workspace.shutdown()
        blocker.release.set()
        assert active_thread is not None
        self.assertTrue(active_thread.wait(2_000))
        _qt_application.processEvents()
        self.assertEqual(completed.count(), 0)
        self.assertEqual(self.workspace.get_data().assignments, [])


# ### Reference crop tests ###
class SurfaceTextureReferenceCropTests(unittest.TestCase):
    def test_pixels_outside_the_painted_mask_are_fully_transparent_black(self) -> None:
        frame = np.full((20, 20, 3), (25, 75, 125), dtype=np.uint8)

        crop = _build_masked_crop(frame, [_test_stroke()])

        self.assertGreater(crop.size, 0)
        transparent_pixels = crop[:, :, 3] == 0
        self.assertTrue(np.any(transparent_pixels))
        self.assertTrue(np.all(crop[transparent_pixels, :3] == 0))

    def test_partial_output_composites_each_surface_with_its_own_mask(self) -> None:
        base = np.full((8, 8, 4), (10, 20, 30, 255), dtype=np.uint8)
        generated = np.full((8, 8, 4), (200, 150, 100, 255), dtype=np.uint8)
        left_mask = np.zeros((8, 8), dtype=np.uint8)
        left_mask[:, :2] = 255
        right_mask = np.zeros((8, 8), dtype=np.uint8)
        right_mask[:, -2:] = 255
        request = SurfaceTextureRequest(
            provider="meshy",
            api_key="test",
            reference_pngs=(_texture_png(),),
            reference_frame_indices=(0,),
            surface_type="wall",
            surface_ids=("wall-one", "wall-two"),
            combined_area_m2=12.0,
            prompt="Test",
            existing_texture_png=_encode_rgba_png(base),
            edit_mask_png=_encode_png(np.maximum(left_mask, right_mask)),
            surface_edit_mask_pngs=(
                ("wall-one", _encode_png(left_mask)),
                ("wall-two", _encode_png(right_mask)),
            ),
        )

        outputs = _build_surface_texture_outputs(
            request,
            _encode_rgba_png(generated),
        )

        self.assertEqual([surface_ids for surface_ids, _png in outputs], [
            ("wall-one",),
            ("wall-two",),
        ])
        first = _decode_png_rgba(outputs[0][1], "First output")
        second = _decode_png_rgba(outputs[1][1], "Second output")
        np.testing.assert_array_equal(first[:, :2], generated[:, :2])
        np.testing.assert_array_equal(first[:, 2:], base[:, 2:])
        np.testing.assert_array_equal(second[:, -2:], generated[:, -2:])
        np.testing.assert_array_equal(second[:, :-2], base[:, :-2])


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
