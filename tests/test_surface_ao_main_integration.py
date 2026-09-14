# ### Environment setup ###
from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import numpy as np
import trimesh
from PIL import Image
from PySide6.QtGui import QGuiApplication, QVector3D
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.atlas_export import (
    SurfaceAmbientOcclusionAtlasContext,
    SurfaceAmbientOcclusionBakeResult,
)
from housemaker.generation_jobs import (
    JOB_STATUS_CANCELLED,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
)
from housemaker.glb import GeneratedModel
from housemaker.main import LAST_PROJECT_PATH_SETTING_KEY, BlueprintWorkspace
from housemaker.models import GROUND_LEVEL_INDEX, create_default_levels
from housemaker.pbr_maps import (
    ATLAS_MAP_BASE_COLOR,
    PBR_MAP_ROUGHNESS,
)
from housemaker.project_io import ProjectData
from housemaker.texture_atlas_state import (
    TextureAtlasData,
    TextureAtlasPlacement,
)
from housemaker.texture_atlas_workspace import (
    ATLAS_MAP_AMBIENT_OCCLUSION,
)

# ### Test application ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _generated_box_model() -> GeneratedModel:
    mesh = trimesh.creation.box()
    return GeneratedModel(
        mesh=mesh,
        scene=trimesh.Scene(mesh),
        glb_bytes=b"",
    )


def _generated_textured_box_model(pixel_value: int) -> GeneratedModel:
    """Build a model whose installed material is easy to distinguish."""

    mesh = trimesh.creation.box()
    pixels = np.full((4, 4, 3), int(pixel_value), dtype=np.uint8)
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2), dtype=np.float64),
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(pixels, mode="RGB"),
        ),
    )
    return GeneratedModel(
        mesh=mesh,
        scene=trimesh.Scene(mesh),
        glb_bytes=b"",
    )


def _viewer_camera_signature(viewer) -> tuple[tuple[str, object], ...]:
    """Return a value-comparable snapshot of the viewer's orbit camera."""

    signature: list[tuple[str, object]] = []
    for key, value in sorted(viewer._capture_camera_state().items()):
        if isinstance(value, QVector3D):
            normalized: object = (value.x(), value.y(), value.z())
        else:
            normalized = value
        signature.append((key, normalized))
    return tuple(signature)


def _primary_screen():
    screen = QGuiApplication.primaryScreen()
    if screen is None:
        raise RuntimeError("A QScreen is required for external viewer tests.")
    return screen


def _wait_until(predicate, timeout_seconds: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for the Qt worker.")
        _qt_application.processEvents()
        time.sleep(0.005)
    _qt_application.processEvents()


def _project_data(level_name: str) -> ProjectData:
    levels = create_default_levels()
    levels[GROUND_LEVEL_INDEX].name = level_name
    return ProjectData(
        blueprint_path=None,
        current_level_index=GROUND_LEVEL_INDEX,
        levels=levels,
    )


# ### Main integration tests ###
class SurfaceAmbientOcclusionMainIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        temporary_path = Path(self._temporary_directory.name)
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(
                temporary_path / "settings.json"
            )
        )
        self.workspace.resize(1200, 800)
        self.workspace.show()
        _qt_application.processEvents()
        self.model = _generated_box_model()
        self.atlas_data = TextureAtlasData()
        self.atlas_a = self.atlas_data.create_atlas(
            "Atlas A",
            2048,
            atlas_id="atlas-a",
        )
        self.atlas_b = self.atlas_data.create_atlas(
            "Atlas B",
            2048,
            atlas_id="atlas-b",
        )
        self.atlas_a.placements.append(
            TextureAtlasPlacement(
                object_id="source-a",
                texture_path="source-a.png",
                texture_resolution=512,
                x=0,
                y=0,
                size=512,
            )
        )
        self.atlas_b.placements.append(
            TextureAtlasPlacement(
                object_id="source-b",
                texture_path="source-b.png",
                texture_resolution=512,
                x=0,
                y=0,
                size=512,
            )
        )
        self.workspace.texture_atlas_workspace.set_data(self.atlas_data)
        self.atlas_context = tuple(
            self._ao_atlas_context(atlas.atlas_id)
            for atlas in (self.atlas_a, self.atlas_b)
        )

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        self.workspace.deleteLater()
        _qt_application.processEvents()
        self._temporary_directory.cleanup()

    def _ao_atlas_context(
        self,
        atlas_id: str,
    ) -> SurfaceAmbientOcclusionAtlasContext:
        atlas = self.atlas_data.atlas_by_id(atlas_id)
        assert atlas is not None
        return SurfaceAmbientOcclusionAtlasContext(
            atlas=atlas,
            active_map_types=frozenset((ATLAS_MAP_BASE_COLOR, PBR_MAP_ROUGHNESS)),
        )

    def _patch_surface_ao_scene_inputs(self, stack: ExitStack) -> None:
        """Replace scene loading while retaining the real GUI lifecycle."""

        stack.enter_context(
            patch.object(
                self.workspace,
                "_capture_surface_ao_scene_snapshot",
                return_value=SimpleNamespace(
                    required_source_ids=("source-a", "source-b"),
                    surface_source_ids=(),
                ),
            )
        )
        stack.enter_context(
            patch.object(
                self.workspace.texture_atlas_workspace,
                "prepare_surface_ao_atlas_context",
                return_value=self.atlas_context,
            )
        )
        stack.enter_context(
            patch(
                "housemaker.main._build_surface_ao_pre_atlas_scene",
                return_value=SimpleNamespace(
                    model=self.model,
                    placed_models=(),
                    surface_source_ids={},
                ),
            )
        )

    def _request_patches(self, stack: ExitStack):
        self._patch_surface_ao_scene_inputs(stack)
        return stack.enter_context(
            patch.object(
                self.workspace.texture_atlas_workspace,
                "commit_surface_ambient_occlusion_bake",
            )
        )

    def _commit_cached_surface_ao(
        self,
        atlas_id: str = "atlas-a",
        geometry_signature: str = "geometry-a",
    ) -> Path:
        """Publish a valid cached AO image through the production API."""

        atlas = self.workspace.texture_atlas_workspace.get_data().atlas_by_id(atlas_id)
        assert atlas is not None
        return (
            self.workspace.texture_atlas_workspace
            .commit_surface_ambient_occlusion_bake(
                atlas_id,
                np.full(
                    (atlas.resolution, atlas.resolution),
                    210,
                    dtype=np.uint8,
                ),
                geometry_signature,
            )
        )

    def _select_preview_map(self, map_type: str) -> None:
        preview = self.workspace.texture_atlas_workspace.map_previews[map_type]
        self.workspace.texture_atlas_workspace.preview_tabs.setCurrentWidget(preview)
        _qt_application.processEvents()

    def _install_canvas_preview(self, model: GeneratedModel) -> None:
        """Install and register one ordinary Canvas scene preview."""

        self.workspace.viewer.set_model(model)
        self.assertTrue(
            self.workspace._remember_current_canvas_preview_model(
                model,
                validated_dependency_signature=(
                    self.workspace._build_viewer_preview_dependency_signature()
                ),
            )
        )

    def _assert_project_load_cancels_active_bake(
        self,
        load_action,
        *,
        expected_level_name: str,
    ) -> object:
        started = threading.Event()
        stopped = threading.Event()

        def bake(*_args, **kwargs):
            started.set()
            try:
                while not kwargs["cancellation_check"]():
                    time.sleep(0.005)
            finally:
                stopped.set()
            raise RuntimeError("cancelled")

        with ExitStack() as stack:
            commit = self._request_patches(stack)
            stack.enter_context(
                patch(
                    "housemaker.main.bake_surface_ambient_occlusion_for_atlas",
                    side_effect=bake,
                )
            )
            stack.enter_context(patch("housemaker.main.QMessageBox.warning"))

            self.workspace.texture_atlas_workspace.ambient_occlusion_bake_requested.emit(
                "atlas-a",
                1.0,
            )
            self.assertTrue(started.wait(1.0))
            runtime = self.workspace._surface_ao_bake_runtimes["atlas-a"]
            original_set_data = self.workspace.texture_atlas_workspace.set_data
            safe_atlas_replacements: list[bool] = []

            def replace_atlas_data(data):
                safe_atlas_replacements.append(
                    stopped.is_set()
                    and not runtime.thread.isRunning()
                    and not self.workspace._surface_ao_bake_runtimes
                )
                return original_set_data(data)

            with patch.object(
                self.workspace.texture_atlas_workspace,
                "set_data",
                side_effect=replace_atlas_data,
            ):
                result = load_action()

            self.assertEqual(safe_atlas_replacements, [True])
            self.assertFalse(runtime.thread.isRunning())
            self.assertFalse(self.workspace._surface_ao_bake_runtimes)
            self.assertEqual(
                self.workspace.job_manager.get_job(runtime.job_id).status,
                JOB_STATUS_CANCELLED,
            )
            _qt_application.processEvents()
            commit.assert_not_called()

        self.assertEqual(self.workspace.current_level.name, expected_level_name)
        return result

    # ### AO-only Atlas preview integration ###
    def test_ambient_occlusion_tab_prepares_cached_preview_in_background(
        self,
    ) -> None:
        self._commit_cached_surface_ao()
        normal_model = _generated_textured_box_model(190)
        preview_model = _generated_textured_box_model(45)
        self._install_canvas_preview(normal_model)
        preparation_started = threading.Event()
        preparation_threads: list[object] = []

        def prepare_preview(*_args, **_kwargs):
            preparation_threads.append(threading.current_thread())
            preparation_started.set()
            return preview_model

        with ExitStack() as stack:
            self._patch_surface_ao_scene_inputs(stack)
            prepare = stack.enter_context(
                patch(
                    "housemaker.main."
                    "prepare_surface_ambient_occlusion_preview_for_atlas",
                    side_effect=prepare_preview,
                )
            )
            stack.enter_context(
                patch.object(
                    self.workspace,
                    "_surface_ambient_occlusion_snapshot_is_current",
                    return_value=True,
                )
            )

            self._select_preview_map(ATLAS_MAP_AMBIENT_OCCLUSION)
            self.assertTrue(preparation_started.wait(1.0))
            _wait_until(
                lambda: (
                    not self.workspace._surface_ao_preview_runtimes
                    and self.workspace.viewer.model is preview_model
                )
            )

        prepare.assert_called_once()
        self.assertEqual(len(preparation_threads), 1)
        self.assertIsNot(preparation_threads[0], threading.current_thread())
        self.assertIs(self.workspace.viewer.model, preview_model)
        ao_texture = self.workspace.viewer.model.mesh.visual.material.baseColorTexture
        self.assertEqual(int(np.asarray(ao_texture)[0, 0, 0]), 45)
        self.assertTrue(self.workspace.viewer.get_textures_enabled())
        self.assertFalse(self.workspace.viewer.get_wireframe_enabled())
        self.assertFalse(self.workspace.viewer.get_wireframe_only())
        self.assertFalse(any(self.workspace.viewer.get_pbr_maps_enabled().values()))
        self.assertIn(
            "Ambient-occlusion-only preview",
            self.workspace.texture_atlas_workspace.status_label.text(),
        )

    def test_leaving_ambient_occlusion_tab_restores_canvas_scene_and_view_state(
        self,
    ) -> None:
        self._commit_cached_surface_ao()
        ambient_occlusion_model = _generated_textured_box_model(35)
        normal_model = _generated_textured_box_model(210)
        self._install_canvas_preview(normal_model)
        self.workspace.viewer._restore_camera_state(
            {
                "center": QVector3D(3.0, -2.0, 7.5),
                "distance": 18.25,
                "elevation": 23.0,
                "azimuth": -41.0,
                "fov": 52.0,
            }
        )
        expected_camera = _viewer_camera_signature(self.workspace.viewer)
        self.workspace.viewer.set_ambient_light_intensity(0.37)
        self.workspace.viewer.set_textures_enabled(False)
        self.workspace.viewer.set_wireframe_enabled(True)
        self.workspace.viewer.set_wireframe_only(True)
        self.workspace.viewer.set_pbr_maps_enabled((PBR_MAP_ROUGHNESS,))
        expected_pbr_state = self.workspace.viewer.get_pbr_maps_enabled()

        with ExitStack() as stack:
            self._patch_surface_ao_scene_inputs(stack)
            stack.enter_context(
                patch(
                    "housemaker.main."
                    "prepare_surface_ambient_occlusion_preview_for_atlas",
                    return_value=ambient_occlusion_model,
                )
            )
            stack.enter_context(
                patch.object(
                    self.workspace,
                    "_surface_ambient_occlusion_snapshot_is_current",
                    return_value=True,
                )
            )
            stack.enter_context(
                patch.object(
                    self.workspace.texture_atlas_workspace,
                    "request_selected_object_preview",
                    return_value=True,
                )
            )

            self._select_preview_map(ATLAS_MAP_AMBIENT_OCCLUSION)
            _wait_until(lambda: self.workspace.viewer.model is ambient_occlusion_model)
            self.assertEqual(
                _viewer_camera_signature(self.workspace.viewer),
                expected_camera,
            )
            self._select_preview_map(ATLAS_MAP_BASE_COLOR)

        self.assertIs(self.workspace.viewer.model, normal_model)
        self.assertEqual(
            _viewer_camera_signature(self.workspace.viewer),
            expected_camera,
        )
        self.assertAlmostEqual(
            self.workspace.viewer.get_ambient_light_intensity(),
            0.37,
        )
        self.assertFalse(self.workspace.viewer.get_textures_enabled())
        self.assertTrue(self.workspace.viewer.get_wireframe_enabled())
        self.assertTrue(self.workspace.viewer.get_wireframe_only())
        self.assertEqual(
            self.workspace.viewer.get_pbr_maps_enabled(),
            expected_pbr_state,
        )
        self.assertIsNone(self.workspace._atlas_preview_display_state)

    def test_ambient_occlusion_tab_without_bake_hides_textures_without_worker(
        self,
    ) -> None:
        self._install_canvas_preview(self.model)

        with patch(
            "housemaker.main._SurfaceAmbientOcclusionPreviewThread.start"
        ) as start:
            self._select_preview_map(ATLAS_MAP_AMBIENT_OCCLUSION)

        start.assert_not_called()
        self.assertFalse(self.workspace._surface_ao_preview_runtimes)
        self.assertIs(self.workspace.viewer.model, self.model)
        self.assertFalse(self.workspace.viewer.get_textures_enabled())
        self.assertIn(
            "No ambient-occlusion bake exists",
            self.workspace.texture_atlas_workspace.status_label.text(),
        )

    def test_preview_finishing_after_tab_leave_cannot_replace_normal_view(
        self,
    ) -> None:
        self._commit_cached_surface_ao()
        stale_model = _generated_box_model()
        normal_model = _generated_box_model()
        self._install_canvas_preview(normal_model)
        preparation_started = threading.Event()
        release_preparation = threading.Event()

        def prepare_preview(*_args, **_kwargs):
            preparation_started.set()
            release_preparation.wait(2.0)
            return stale_model

        with ExitStack() as stack:
            self._patch_surface_ao_scene_inputs(stack)
            stack.enter_context(
                patch(
                    "housemaker.main."
                    "prepare_surface_ambient_occlusion_preview_for_atlas",
                    side_effect=prepare_preview,
                )
            )
            stack.enter_context(
                patch.object(
                    self.workspace.texture_atlas_workspace,
                    "request_selected_object_preview",
                    return_value=True,
                )
            )

            self._select_preview_map(ATLAS_MAP_AMBIENT_OCCLUSION)
            self.assertTrue(preparation_started.wait(1.0))
            self._select_preview_map(ATLAS_MAP_BASE_COLOR)
            release_preparation.set()
            _wait_until(lambda: not self.workspace._surface_ao_preview_runtimes)

        self.assertIs(self.workspace.viewer.model, normal_model)
        self.assertIsNot(self.workspace.viewer.model, stale_model)

    def test_normal_canvas_refresh_cannot_replace_active_ao_only_scene(
        self,
    ) -> None:
        """A queued ordinary refresh must respect the active AO override."""

        self._commit_cached_surface_ao()
        normal_model = _generated_textured_box_model(215)
        ambient_occlusion_model = _generated_textured_box_model(40)
        self._install_canvas_preview(normal_model)
        self.workspace.workspace_tabs.setCurrentWidget(
            self.workspace.scene_3d_workspace
        )

        with ExitStack() as stack:
            self._patch_surface_ao_scene_inputs(stack)
            stack.enter_context(
                patch(
                    "housemaker.main."
                    "prepare_surface_ambient_occlusion_preview_for_atlas",
                    return_value=ambient_occlusion_model,
                )
            )
            stack.enter_context(
                patch.object(
                    self.workspace,
                    "_surface_ambient_occlusion_snapshot_is_current",
                    return_value=True,
                )
            )
            stack.enter_context(
                patch.object(
                    self.workspace.texture_atlas_workspace,
                    "request_selected_object_preview",
                    return_value=True,
                )
            )

            self._select_preview_map(ATLAS_MAP_AMBIENT_OCCLUSION)
            _wait_until(
                lambda: self.workspace.viewer.model is ambient_occlusion_model
            )
            self.assertIs(self.workspace._viewer_preview_model, normal_model)

            self.workspace._canvas_viewer_preview_revision = -1
            with patch.object(
                self.workspace.viewer,
                "set_model",
                wraps=self.workspace.viewer.set_model,
            ) as set_model:
                self.workspace._refresh_viewer_preview(
                    preserve_camera=True
                )

            set_model.assert_not_called()
            self.assertIs(
                self.workspace.viewer.model,
                ambient_occlusion_model,
            )
            self._select_preview_map(ATLAS_MAP_BASE_COLOR)

        self.assertIs(self.workspace.viewer.model, normal_model)
        self.assertFalse(self.workspace._surface_ao_preview_runtimes)

    def test_successful_bake_displays_returned_preview_when_ao_tab_is_active(
        self,
    ) -> None:
        preview_model = _generated_box_model()
        normal_model = _generated_box_model()
        self._install_canvas_preview(normal_model)

        def bake(*_args, **_kwargs):
            return SurfaceAmbientOcclusionBakeResult(
                atlas_id="atlas-a",
                ambient_occlusion=np.full(
                    (2048, 2048),
                    205,
                    dtype=np.uint8,
                ),
                geometry_signature="fresh-preview-geometry",
                preview_model=preview_model,
            )

        self._select_preview_map(ATLAS_MAP_AMBIENT_OCCLUSION)
        with ExitStack() as stack:
            self._patch_surface_ao_scene_inputs(stack)
            stack.enter_context(
                patch.object(
                    self.workspace,
                    "_surface_ambient_occlusion_snapshot_is_current",
                    return_value=True,
                )
            )
            stack.enter_context(
                patch(
                    "housemaker.main.bake_surface_ambient_occlusion_for_atlas",
                    side_effect=bake,
                )
            )
            cached_preparation = stack.enter_context(
                patch(
                    "housemaker.main."
                    "prepare_surface_ambient_occlusion_preview_for_atlas"
                )
            )
            stack.enter_context(patch("housemaker.main.QMessageBox.warning"))

            self.workspace.texture_atlas_workspace.ambient_occlusion_bake_requested.emit(
                "atlas-a",
                1.0,
            )
            _wait_until(lambda: not self.workspace._surface_ao_bake_runtimes)

        cached_preparation.assert_not_called()
        self.assertIs(self.workspace.viewer.model, preview_model)
        self.assertIs(
            self.workspace._surface_ao_preview_cache["atlas-a"].model,
            preview_model,
        )
        self.assertEqual(
            self.workspace.job_manager.jobs()[-1].status,
            JOB_STATUS_COMPLETED,
        )

    def test_detached_canvas_viewer_receives_and_restores_ao_only_scene(
        self,
    ) -> None:
        """The external Canvas window must host the same switched viewer."""

        self._commit_cached_surface_ao()
        normal_model = _generated_textured_box_model(205)
        ambient_occlusion_model = _generated_textured_box_model(30)
        self._install_canvas_preview(normal_model)
        self.workspace.workspace_tabs.setCurrentWidget(
            self.workspace.scene_3d_workspace
        )
        with patch(
            "housemaker.main.resolve_fullscreen_3d_viewer_screen",
            return_value=_primary_screen(),
        ):
            self.workspace._apply_scene_3d_display_screen(
                "screen:ao-preview-test"
            )
        self.assertTrue(self.workspace._external_scene_3d_host.is_active)
        self.assertIs(
            self.workspace._external_scene_3d_host.viewer,
            self.workspace.scene_3d_workspace,
        )

        try:
            with ExitStack() as stack:
                self._patch_surface_ao_scene_inputs(stack)
                stack.enter_context(
                    patch(
                        "housemaker.main."
                        "prepare_surface_ambient_occlusion_preview_for_atlas",
                        return_value=ambient_occlusion_model,
                    )
                )
                stack.enter_context(
                    patch.object(
                        self.workspace,
                        "_surface_ambient_occlusion_snapshot_is_current",
                        return_value=True,
                    )
                )
                stack.enter_context(
                    patch.object(
                        self.workspace.texture_atlas_workspace,
                        "request_selected_object_preview",
                        return_value=True,
                    )
                )

                self._select_preview_map(ATLAS_MAP_AMBIENT_OCCLUSION)
                _wait_until(
                    lambda: self.workspace.viewer.model
                    is ambient_occlusion_model
                )
                self.assertIs(
                    self.workspace._external_scene_3d_host.viewer,
                    self.workspace.scene_3d_workspace,
                )
                self._select_preview_map(ATLAS_MAP_BASE_COLOR)

            self.assertIs(
                self.workspace.viewer.model,
                normal_model,
            )
        finally:
            self.workspace._apply_scene_3d_display_screen(None)

    # ### Atlas AO bake integration ###
    def test_successful_bake_runs_in_background_and_commits_once(self) -> None:
        observed_worker_threads: list[object] = []
        observed_atlas_context_ids: list[tuple[str, ...]] = []

        def bake(*_args, **kwargs):
            observed_worker_threads.append(threading.current_thread())
            observed_atlas_context_ids.append(
                tuple(item.atlas.atlas_id for item in kwargs["atlas_context"])
            )
            kwargs["progress_callback"]("Tracing ambient occlusion (70%)")
            return SurfaceAmbientOcclusionBakeResult(
                atlas_id="atlas-a",
                ambient_occlusion=np.full((2048, 2048), 210, dtype=np.uint8),
                geometry_signature="geometry-a",
            )

        with ExitStack() as stack:
            commit = self._request_patches(stack)
            stack.enter_context(
                patch(
                    "housemaker.main.bake_surface_ambient_occlusion_for_atlas",
                    side_effect=bake,
                )
            )
            stack.enter_context(patch("housemaker.main.QMessageBox.warning"))

            self.workspace.texture_atlas_workspace.ambient_occlusion_bake_requested.emit(
                "atlas-a",
                0.65,
            )
            _wait_until(lambda: not self.workspace._surface_ao_bake_runtimes)

        self.assertEqual(len(observed_worker_threads), 1)
        self.assertIsNot(observed_worker_threads[0], threading.current_thread())
        self.assertEqual(
            observed_atlas_context_ids,
            [("atlas-a", "atlas-b")],
        )
        self.assertEqual(commit.call_count, 1)
        atlas_id, pixels, signature = commit.call_args.args
        self.assertEqual(atlas_id, "atlas-a")
        self.assertEqual(signature, "geometry-a")
        self.assertEqual(pixels.shape, (2048, 2048))
        self.assertEqual(
            self.workspace.job_manager.jobs()[-1].status,
            JOB_STATUS_COMPLETED,
        )

    def test_bake_all_starts_one_independent_job_per_used_atlas(self) -> None:
        baked_atlas_ids: list[str] = []
        baked_atlas_lock = threading.Lock()

        def bake(_model, atlas, **_kwargs):
            atlas_id = atlas.atlas.atlas_id
            with baked_atlas_lock:
                baked_atlas_ids.append(atlas_id)
            return SurfaceAmbientOcclusionBakeResult(
                atlas_id=atlas_id,
                ambient_occlusion=np.full((2048, 2048), 220, dtype=np.uint8),
                geometry_signature=f"geometry-{atlas_id}",
            )

        with ExitStack() as stack:
            commit = self._request_patches(stack)
            scene_capture = self.workspace._capture_surface_ao_scene_snapshot
            context_capture = (
                self.workspace.texture_atlas_workspace
                .prepare_surface_ao_atlas_context
            )
            stack.enter_context(
                patch.object(
                    self.workspace,
                    "_build_required_scene_atlas_source_ids",
                    return_value=("source-a", "source-b"),
                )
            )
            stack.enter_context(
                patch(
                    "housemaker.main.bake_surface_ambient_occlusion_for_atlas",
                    side_effect=bake,
                )
            )
            stack.enter_context(patch("housemaker.main.QMessageBox.warning"))

            first_job_count = len(self.workspace.job_manager.jobs())
            self.workspace.texture_atlas_workspace.ambient_occlusion_bake_all_requested.emit()
            _wait_until(lambda: not self.workspace._surface_ao_bake_runtimes)

        self.assertEqual(sorted(baked_atlas_ids), ["atlas-a", "atlas-b"])
        self.assertEqual(commit.call_count, 2)
        self.assertEqual(scene_capture.call_count, 1)
        self.assertEqual(context_capture.call_count, 1)
        jobs = self.workspace.job_manager.jobs()[first_job_count:]
        self.assertEqual(len(jobs), 2)
        self.assertTrue(all(job.status == JOB_STATUS_COMPLETED for job in jobs))
        self.assertTrue(all(job.kind == "Ambient occlusion" for job in jobs))

    def test_scene_preparation_is_background_and_skips_atlas_materialization(
        self,
    ) -> None:
        preparation_started = threading.Event()
        release_preparation = threading.Event()
        preparation_threads: list[object] = []

        def build_scene(*_args, **_kwargs):
            preparation_threads.append(threading.current_thread())
            preparation_started.set()
            release_preparation.wait(2.0)
            return SimpleNamespace(
                model=self.model,
                placed_models=(),
                surface_source_ids={},
            )

        def bake(_model, atlas, **_kwargs):
            return SurfaceAmbientOcclusionBakeResult(
                atlas_id=atlas.atlas.atlas_id,
                ambient_occlusion=np.full((2048, 2048), 255, dtype=np.uint8),
                geometry_signature="background-geometry",
            )

        with ExitStack() as stack:
            commit = self._request_patches(stack)
            legacy_scene_builder = stack.enter_context(
                patch.object(
                    self.workspace,
                    "_build_pre_atlas_export_scene",
                    side_effect=AssertionError("legacy scene build was used"),
                )
            )
            atlas_materializer = stack.enter_context(
                patch.object(
                    self.workspace.texture_atlas_workspace,
                    "prepare_export_atlases",
                    side_effect=AssertionError("Atlas PNGs were materialized"),
                )
            )
            stack.enter_context(
                patch(
                    "housemaker.main._build_surface_ao_pre_atlas_scene",
                    side_effect=build_scene,
                )
            )
            stack.enter_context(
                patch(
                    "housemaker.main.bake_surface_ambient_occlusion_for_atlas",
                    side_effect=bake,
                )
            )
            stack.enter_context(patch("housemaker.main.QMessageBox.warning"))

            self.workspace.texture_atlas_workspace.ambient_occlusion_bake_requested.emit(
                "atlas-a",
                1.0,
            )
            self.assertTrue(preparation_started.wait(1.0))
            self.assertIn("atlas-a", self.workspace._surface_ao_bake_runtimes)
            release_preparation.set()
            _wait_until(lambda: not self.workspace._surface_ao_bake_runtimes)

        self.assertEqual(len(preparation_threads), 1)
        self.assertIsNot(preparation_threads[0], threading.current_thread())
        legacy_scene_builder.assert_not_called()
        atlas_materializer.assert_not_called()
        commit.assert_called_once()

    def test_same_atlas_is_deduplicated_while_other_atlas_runs(self) -> None:
        release = threading.Event()
        started_atlases: list[str] = []
        started_lock = threading.Lock()

        def bake(_model, atlas, **kwargs):
            with started_lock:
                started_atlases.append(atlas.atlas.atlas_id)
            while not release.wait(0.005):
                if kwargs["cancellation_check"]():
                    raise RuntimeError("cancelled")
            return SurfaceAmbientOcclusionBakeResult(
                atlas_id=atlas.atlas.atlas_id,
                ambient_occlusion=np.full((2048, 2048), 255, dtype=np.uint8),
                geometry_signature=f"geometry-{atlas.atlas.atlas_id}",
            )

        with ExitStack() as stack:
            commit = self._request_patches(stack)
            stack.enter_context(
                patch(
                    "housemaker.main.bake_surface_ambient_occlusion_for_atlas",
                    side_effect=bake,
                )
            )
            stack.enter_context(patch("housemaker.main.QMessageBox.warning"))

            signal = (
                self.workspace.texture_atlas_workspace
                .ambient_occlusion_bake_requested
            )
            signal.emit("atlas-a", 1.0)
            signal.emit("atlas-a", 1.0)
            signal.emit("atlas-b", 1.0)
            _wait_until(lambda: len(started_atlases) == 2)
            self.assertEqual(len(self.workspace.job_manager.jobs()), 2)
            self.assertEqual(
                set(self.workspace._surface_ao_bake_runtimes),
                {"atlas-a", "atlas-b"},
            )
            release.set()
            _wait_until(lambda: not self.workspace._surface_ao_bake_runtimes)

        self.assertEqual(sorted(started_atlases), ["atlas-a", "atlas-b"])
        self.assertEqual(commit.call_count, 2)

    def test_changed_scene_discards_completed_bake(self) -> None:
        release = threading.Event()
        started = threading.Event()

        def bake(*_args, **_kwargs):
            started.set()
            release.wait(2.0)
            return SurfaceAmbientOcclusionBakeResult(
                atlas_id="atlas-a",
                ambient_occlusion=np.full((2048, 2048), 200, dtype=np.uint8),
                geometry_signature="stale-geometry",
            )

        with ExitStack() as stack:
            commit = self._request_patches(stack)
            stack.enter_context(
                patch(
                    "housemaker.main.bake_surface_ambient_occlusion_for_atlas",
                    side_effect=bake,
                )
            )
            warning = stack.enter_context(patch("housemaker.main.QMessageBox.warning"))

            self.workspace.texture_atlas_workspace.ambient_occlusion_bake_requested.emit(
                "atlas-a",
                1.0,
            )
            self.assertTrue(started.wait(1.0))
            self.workspace._viewer_preview_revision += 1
            release.set()
            _wait_until(lambda: not self.workspace._surface_ao_bake_runtimes)

        commit.assert_not_called()
        self.assertEqual(
            self.workspace.job_manager.jobs()[-1].status,
            JOB_STATUS_FAILED,
        )
        self.assertIn("stale result", warning.call_args.args[2])

    def test_jobs_cancel_interrupts_only_the_target_atlas(self) -> None:
        release_b = threading.Event()
        started = {
            "atlas-a": threading.Event(),
            "atlas-b": threading.Event(),
        }

        def bake(_model, atlas, **kwargs):
            atlas_id = atlas.atlas.atlas_id
            started[atlas_id].set()
            while atlas_id == "atlas-a" or not release_b.is_set():
                if kwargs["cancellation_check"]():
                    raise RuntimeError("cancelled")
                time.sleep(0.005)
            return SurfaceAmbientOcclusionBakeResult(
                atlas_id=atlas_id,
                ambient_occlusion=np.full((2048, 2048), 255, dtype=np.uint8),
                geometry_signature=f"geometry-{atlas_id}",
            )

        with ExitStack() as stack:
            commit = self._request_patches(stack)
            stack.enter_context(
                patch(
                    "housemaker.main.bake_surface_ambient_occlusion_for_atlas",
                    side_effect=bake,
                )
            )
            stack.enter_context(patch("housemaker.main.QMessageBox.warning"))

            signal = (
                self.workspace.texture_atlas_workspace
                .ambient_occlusion_bake_requested
            )
            signal.emit("atlas-a", 1.0)
            signal.emit("atlas-b", 1.0)
            self.assertTrue(started["atlas-a"].wait(1.0))
            self.assertTrue(started["atlas-b"].wait(1.0))
            cancelled_job_id = self.workspace._surface_ao_bake_runtimes[
                "atlas-a"
            ].job_id
            retained_job_id = self.workspace._surface_ao_bake_runtimes["atlas-b"].job_id

            self.assertTrue(self.workspace.job_manager.cancel_job(cancelled_job_id))
            _wait_until(
                lambda: "atlas-a" not in self.workspace._surface_ao_bake_runtimes
            )
            self.assertIn("atlas-b", self.workspace._surface_ao_bake_runtimes)
            self.assertEqual(
                self.workspace.job_manager.get_job(cancelled_job_id).status,
                JOB_STATUS_CANCELLED,
            )
            release_b.set()
            _wait_until(lambda: not self.workspace._surface_ao_bake_runtimes)

        self.assertEqual(commit.call_count, 1)
        self.assertEqual(commit.call_args.args[0], "atlas-b")
        self.assertEqual(
            self.workspace.job_manager.get_job(retained_job_id).status,
            JOB_STATUS_COMPLETED,
        )

    def test_shutdown_interrupts_and_joins_active_bakes(self) -> None:
        started = threading.Event()

        def bake(*_args, **kwargs):
            started.set()
            while not kwargs["cancellation_check"]():
                time.sleep(0.005)
            raise RuntimeError("cancelled")

        with ExitStack() as stack:
            self._request_patches(stack)
            stack.enter_context(
                patch(
                    "housemaker.main.bake_surface_ambient_occlusion_for_atlas",
                    side_effect=bake,
                )
            )
            stack.enter_context(patch("housemaker.main.QMessageBox.warning"))

            self.workspace.texture_atlas_workspace.ambient_occlusion_bake_requested.emit(
                "atlas-a",
                1.0,
            )
            self.assertTrue(started.wait(1.0))
            runtime = self.workspace._surface_ao_bake_runtimes["atlas-a"]
            self.workspace.shutdown()

        self.assertFalse(runtime.thread.isRunning())
        self.assertFalse(self.workspace._surface_ao_bake_runtimes)
        self.assertEqual(
            self.workspace.job_manager.get_job(runtime.job_id).status,
            JOB_STATUS_CANCELLED,
        )

    def test_manual_project_load_cancels_and_joins_active_bake(self) -> None:
        level_name = "Manually loaded while AO was active"
        load_path = Path(self._temporary_directory.name) / "manual.json"

        with (
            patch(
                "housemaker.main.QFileDialog.getOpenFileName",
                return_value=(str(load_path), "JSON Files (*.json)"),
            ),
            patch(
                "housemaker.main.load_project",
                return_value=_project_data(level_name),
            ),
        ):
            self._assert_project_load_cancels_active_bake(
                self.workspace._handle_load_clicked,
                expected_level_name=level_name,
            )

    def test_startup_restore_cancels_and_joins_active_bake(self) -> None:
        level_name = "Restored while AO was active"
        load_path = Path(self._temporary_directory.name) / "restored.json"
        application_settings = self.workspace._application_settings
        assert application_settings is not None
        application_settings.set(
            LAST_PROJECT_PATH_SETTING_KEY,
            str(load_path),
        )

        with patch(
            "housemaker.main.load_project",
            return_value=_project_data(level_name),
        ):
            restored = self._assert_project_load_cancels_active_bake(
                self.workspace.restore_last_project,
                expected_level_name=level_name,
            )

        self.assertTrue(restored)


if __name__ == "__main__":
    unittest.main()
