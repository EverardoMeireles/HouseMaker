# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import trimesh
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.camera_models import CameraPose
from housemaker.generation_state import MaskPoint, MaskStroke
from housemaker.glb import GeneratedModel
from housemaker.main import BlueprintWorkspace
from housemaker.models import GROUND_LEVEL_INDEX, create_default_levels
from housemaker.project_io import ProjectData, load_project, save_project
from housemaker.settings_widget import (
    SURFACE_TEXTURE_PROVIDER_GPT_5_6_LUNA,
    GenerationServiceSettings,
)
from housemaker.surface_texture_providers import SurfaceTextureResult
from housemaker.surface_texture_state import (
    SurfaceTextureAssignment,
    SurfaceTextureData,
)
from housemaker.surface_texture_workspace import SurfaceTextureRequest
from housemaker.video_source import VideoMetadata


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _stroke() -> MaskStroke:
    return MaskStroke(
        mode="paint",
        radius_normalized=0.08,
        points=(MaskPoint(0.3, 0.6),),
    )


def _surface_state(*, complete: bool = False) -> SurfaceTextureData:
    basic_state = SurfaceTextureData(
        frame_strokes={0: [_stroke()]},
        camera_pose=CameraPose(
            x=1.25,
            y=-0.5,
            z=1.7,
            yaw_degrees=42.0,
        ),
    )
    if not complete:
        return basic_state
    return SurfaceTextureData(
        video_metadata=VideoMetadata(
            path="walkthrough.mp4",
            frame_count=12,
            fps=24.0,
            width=1280,
            height=720,
        ),
        current_frame_index=4,
        frame_strokes={0: [_stroke()], 4: [_stroke()]},
        camera_pose=basic_state.camera_pose,
        selected_surface_type="floor",
        selected_surface_ids=("level:2/room:5/floor",),
        assignments=[
            SurfaceTextureAssignment(
                assignment_id="floor-oak-1",
                surface_type="floor",
                surface_ids=("level:2/room:5/floor",),
                provider=SURFACE_TEXTURE_PROVIDER_GPT_5_6_LUNA,
                asset_path="floor-oak-1.png",
                combined_area_m2=18.5,
                area_description="One floor surface, 18.50 m2",
                reference_frame_indices=(0, 4),
                texture_width=1024,
                texture_height=1024,
            )
        ],
    )


def _texture_png() -> bytes:
    did_encode, encoded = cv2.imencode(
        ".png",
        np.full((4, 4, 3), 160, dtype=np.uint8),
    )
    if not did_encode:
        raise AssertionError("Unable to build PNG fixture")
    return bytes(encoded)


class _BlockingSurfaceProvider:
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


def _surface_request() -> SurfaceTextureRequest:
    return SurfaceTextureRequest(
        provider="meshy",
        api_key="meshy-test-key",
        reference_pngs=(_texture_png(),),
        reference_frame_indices=(0,),
        surface_type="floor",
        surface_ids=("level:2/room:5/floor",),
        combined_area_m2=4.0,
        prompt="Test floor material",
    )


# ### Project persistence tests ###
class SurfaceTextureProjectPersistenceTests(unittest.TestCase):
    def test_surface_state_round_trips_and_old_projects_default_to_empty(self) -> None:
        surface_state = _surface_state(complete=True)
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "surface-project.json"
            save_project(
                project_path,
                current_level_index=GROUND_LEVEL_INDEX,
                levels=create_default_levels(),
                surface_texture_generation=surface_state,
            )
            raw_payload = json.loads(project_path.read_text(encoding="utf-8"))
            loaded = load_project(project_path)

            self.assertEqual(
                raw_payload["surface_texture_generation"],
                surface_state.to_dict(),
            )
            self.assertEqual(loaded.surface_texture_generation, surface_state)

            raw_payload.pop("surface_texture_generation")
            project_path.write_text(json.dumps(raw_payload), encoding="utf-8")
            legacy_loaded = load_project(project_path)
            self.assertEqual(
                legacy_loaded.surface_texture_generation,
                SurfaceTextureData(),
            )

    def test_project_json_excludes_both_provider_secrets(self) -> None:
        meshy_secret = "msy-SurfaceProjectMustNotContainThis"
        openai_secret = "sk-SurfaceProjectMustNotContainThis"
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "no-secrets.json"
            with patch.dict(
                os.environ,
                {
                    "MESHY_API_KEY": meshy_secret,
                    "OPENAI_API_KEY": openai_secret,
                },
            ):
                save_project(
                    project_path,
                    current_level_index=GROUND_LEVEL_INDEX,
                    levels=create_default_levels(),
                    surface_texture_generation=_surface_state(complete=True),
                )
            serialized = project_path.read_text(encoding="utf-8")

        self.assertNotIn(meshy_secret, serialized)
        self.assertNotIn(openai_secret, serialized)
        self.assertNotIn("meshy_api_key", serialized.lower())
        self.assertNotIn("openai_api_key", serialized.lower())


# ### Main workspace integration tests ###
class SurfaceTextureMainIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(
                Path(self._temporary_directory.name) / "settings.json"
            )
        )
        self.workspace.resize(1400, 850)
        self.workspace.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.surface_texture_generation.shutdown()
        self.workspace.generation.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self._temporary_directory.cleanup()

    def test_exact_tab_order_full_width_and_openai_settings_propagation(self) -> None:
        tab_names = [
            self.workspace.workspace_tabs.tabText(index)
            for index in range(self.workspace.workspace_tabs.count())
        ]
        self.assertEqual(
            tab_names,
            [
                "Canvas",
                "Atlas",
                "Surface texture generation",
                "Object generation",
                "Settings",
            ],
        )
        surface_index = tab_names.index("Surface texture generation")
        self.workspace.workspace_tabs.setCurrentIndex(surface_index)
        _qt_application.processEvents()
        self.assertFalse(self.workspace.side_panel.isVisible())

        settings_widget = self.workspace.settings_widget
        provider_combo = (
            self.workspace.surface_texture_generation.surface_texture_provider_combo
        )
        provider_index = provider_combo.findData(
            SURFACE_TEXTURE_PROVIDER_GPT_5_6_LUNA
        )
        provider_combo.setCurrentIndex(provider_index)
        settings_widget.openai_api_key_edit.setText("openai-main-test")
        _qt_application.processEvents()
        expected = GenerationServiceSettings(
            openai_api_key="openai-main-test",
            surface_texture_provider=SURFACE_TEXTURE_PROVIDER_GPT_5_6_LUNA,
        )
        self.assertEqual(
            self.workspace.surface_texture_generation.get_runtime_settings(),
            expected,
        )
        self.assertEqual(self.workspace.generation.get_runtime_settings(), expected)

    def test_save_and_load_route_surface_state_independently(self) -> None:
        surface_state = _surface_state()
        self.workspace.surface_texture_generation.set_data(surface_state)
        save_path = str(Path(self._temporary_directory.name) / "project.json")
        with (
            patch(
                "housemaker.main.QFileDialog.getSaveFileName",
                return_value=(save_path, "JSON Files (*.json)"),
            ),
            patch("housemaker.main.save_project") as save_mock,
            patch("housemaker.main.QMessageBox.information"),
        ):
            self.workspace._handle_save_clicked()

        save_mock.assert_called_once()
        save_arguments = save_mock.call_args.kwargs
        self.assertEqual(
            save_arguments["surface_texture_generation"],
            surface_state,
        )
        self.assertNotIn("dynamic_generation", save_arguments)

        incoming = ProjectData(
            blueprint_path=None,
            current_level_index=GROUND_LEVEL_INDEX,
            levels=create_default_levels(),
            surface_texture_generation=surface_state,
        )
        with (
            patch(
                "housemaker.main.QFileDialog.getOpenFileName",
                return_value=(save_path, "JSON Files (*.json)"),
            ),
            patch("housemaker.main.load_project", return_value=incoming),
        ):
            self.workspace._handle_load_clicked()
            _qt_application.processEvents()
        self.assertEqual(
            self.workspace.surface_texture_generation.get_data(),
            surface_state,
        )

    def test_surface_generation_blocks_project_load_until_worker_finishes(self) -> None:
        blocker = _BlockingSurfaceProvider()
        surface_workspace = self.workspace.surface_texture_generation
        surface_workspace.set_provider(blocker)
        surface_workspace._start_generation(_surface_request())
        self.assertTrue(blocker.started.wait(timeout=1.0))
        active_thread = surface_workspace._generation_thread
        try:
            with (
                patch("housemaker.main.QFileDialog.getOpenFileName") as dialog_mock,
                patch("housemaker.main.QMessageBox.critical") as critical_mock,
                patch("housemaker.main.load_project") as load_mock,
            ):
                self.workspace._handle_load_clicked()

            dialog_mock.assert_not_called()
            load_mock.assert_not_called()
            critical_mock.assert_called_once()
            self.assertEqual(critical_mock.call_args.args[1], "Project load failed")
        finally:
            surface_workspace.shutdown()
            blocker.release.set()
            if active_thread is not None:
                self.assertTrue(active_thread.wait(2_000))
            _qt_application.processEvents()

    def test_surface_tab_uses_the_exact_canvas_preview_model(self) -> None:
        surface_workspace = self.workspace.surface_texture_generation
        preview_mesh = trimesh.creation.box(extents=(2.0, 3.0, 2.5))
        expected_model = GeneratedModel(
            mesh=preview_mesh,
            scene=trimesh.Scene(preview_mesh.copy()),
            glb_bytes=b"shared-preview-model",
        )
        with patch.object(
            self.workspace,
            "_build_generated_model",
            return_value=expected_model,
        ):
            self.workspace.workspace_tabs.setCurrentWidget(surface_workspace)
            _qt_application.processEvents()
            _qt_application.processEvents()

        canvas_model = self.workspace.viewer.model
        self.assertIs(canvas_model, expected_model)
        self.assertIs(
            surface_workspace.surface_view.get_scene_model(),
            canvas_model,
        )
        self.assertIsNotNone(
            surface_workspace.surface_view._canvas_scene_render_items.mesh_item
        )


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
