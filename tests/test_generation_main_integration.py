# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import copy
import threading
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.camera_models import CameraPose, InitialFirstPersonCamera
from housemaker.generation_state import GenerationData, MaskPoint, MaskStroke
from housemaker.generation_workspace import GenerationRequest
from housemaker.main import BlueprintWorkspace
from housemaker.models import GROUND_LEVEL_INDEX, create_default_levels
from housemaker.project_io import ProjectData
from housemaker.settings_widget import GenerationServiceSettings, SettingsWidget


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _generation_data() -> GenerationData:
    return GenerationData(
        frame_strokes={
            0: [
                MaskStroke(
                    mode="paint",
                    radius_normalized=0.08,
                    points=(MaskPoint(0.35, 0.6),),
                )
            ]
        }
    )


def _project_data(generation: GenerationData) -> ProjectData:
    levels = create_default_levels()
    return ProjectData(
        blueprint_path=None,
        current_level_index=GROUND_LEVEL_INDEX,
        levels=levels,
        generation=generation,
    )


class _BlockingMeshyPlanner:
    """Meshy fixture that keeps project loading in an active-generation state."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def plan(self, _request: GenerationRequest):
        self.started.set()
        if not self.release.wait(timeout=5.0):
            raise RuntimeError("Blocking Meshy planner test timed out.")
        from housemaker.meshy_generation import MeshyGenerationResult

        return MeshyGenerationResult("task-late", b"cancelled")


def _generation_request() -> GenerationRequest:
    return GenerationRequest(
        frame_index=0,
        selected_object_bgra=np.zeros((4, 4, 4), dtype=np.uint8),
        settings=GenerationServiceSettings(meshy_api_key="key"),
    )


# ### Main-workspace integration tests ###
class GenerationMainIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        application_settings = ApplicationSettingsStore(
            Path(self._temporary_directory.name) / "settings.json"
        )
        self.workspace = BlueprintWorkspace(
            application_settings=application_settings
        )

        self.workspace.resize(1400, 850)
        self.workspace.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.generation.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self._temporary_directory.cleanup()

    def test_workspace_tabs_and_full_width_side_panel_behavior(self) -> None:
        tab_names = [
            self.workspace.workspace_tabs.tabText(tab_index)
            for tab_index in range(self.workspace.workspace_tabs.count())
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

        expected_side_panel_visibility = {
            "Canvas": True,
            "Atlas": False,
            "Surface texture generation": False,
            "Object generation": False,
            "Settings": False,
        }
        for tab_name, should_be_visible in expected_side_panel_visibility.items():
            with self.subTest(tab=tab_name):
                tab_index = tab_names.index(tab_name)
                self.workspace.workspace_tabs.setCurrentIndex(tab_index)
                _qt_application.processEvents()
                self.assertEqual(
                    self.workspace.side_panel.isVisible(),
                    should_be_visible,
                )

    def test_canvas_tab_uses_dedicated_2d_and_3d_subtabs(self) -> None:
        canvas_tab_index = self.workspace.workspace_tabs.indexOf(
            self.workspace.canvas_viewer_workspace
        )

        self.assertGreaterEqual(canvas_tab_index, 0)
        self.assertEqual(
            self.workspace.workspace_tabs.tabText(canvas_tab_index),
            "Canvas",
        )
        self.assertEqual(
            [
                self.workspace.canvas_viewer_tabs.tabText(tab_index)
                for tab_index in range(
                    self.workspace.canvas_viewer_tabs.count()
                )
            ],
            ["2D view", "3D view"],
        )
        self.assertIs(
            self.workspace.canvas_viewer_tabs.widget(0),
            self.workspace.canvas,
        )
        self.assertIs(
            self.workspace.canvas_viewer_tabs.widget(1),
            self.workspace.viewer,
        )

    def test_canvas_navigation_hotkey_toggles_first_person_from_canvas_camera(
        self,
    ) -> None:
        camera = InitialFirstPersonCamera(
            level_index=GROUND_LEVEL_INDEX,
            pose=CameraPose(
                x=2.5,
                y=-1.25,
                z=1.7,
                yaw_degrees=42.0,
                pitch_degrees=-6.0,
                fov_degrees=72.0,
            ),
        )
        self.workspace._handle_canvas_first_person_camera_changed(camera)
        self.workspace.canvas_viewer_tabs.setCurrentIndex(
            self.workspace.canvas_3d_view_tab_index
        )
        _qt_application.processEvents()

        self.workspace.viewer.view.setFocus()
        QTest.keyClick(self.workspace.viewer.view, Qt.Key.Key_N)
        _qt_application.processEvents()

        self.assertEqual(
            self.workspace.viewer.get_navigation_mode(),
            "first_person",
        )
        self.assertEqual(
            self.workspace.viewer.get_first_person_camera_pose(),
            camera.pose,
        )
        self.assertEqual(
            self.workspace.canvas_viewer_tabs.tabText(
                self.workspace.canvas_3d_view_tab_index
            ),
            "3D view (first person)",
        )

        QTest.keyClick(self.workspace.viewer.view, Qt.Key.Key_N)
        _qt_application.processEvents()

        self.assertEqual(
            self.workspace.viewer.get_navigation_mode(),
            "orbit",
        )
        self.assertEqual(
            self.workspace.canvas_viewer_tabs.tabText(
                self.workspace.canvas_3d_view_tab_index
            ),
            "3D view",
        )

    def test_canvas_navigation_hotkey_follows_settings_and_canvas_scope(
        self,
    ) -> None:
        self.workspace.settings_widget.canvas_3d_navigation_toggle_hotkey_edit.setKeySequence(
            QKeySequence(
                "Ctrl+Alt+F",
                QKeySequence.SequenceFormat.PortableText,
            )
        )
        _qt_application.processEvents()

        self.assertEqual(
            self.workspace.canvas_3d_navigation_shortcut.key().toString(
                QKeySequence.SequenceFormat.PortableText
            ),
            "Ctrl+Alt+F",
        )

        self.workspace.workspace_tabs.setCurrentWidget(
            self.workspace.settings_widget
        )
        self.workspace.canvas_3d_navigation_shortcut.activated.emit()

        self.assertEqual(
            self.workspace.viewer.get_navigation_mode(),
            "orbit",
        )

    def test_external_canvas_3d_viewer_hides_local_subtabs(self) -> None:
        canvas_viewer_tabs = self.workspace.canvas_viewer_tabs
        canvas_viewer_tabs.setCurrentIndex(
            self.workspace.canvas_3d_view_tab_index
        )

        self.workspace.set_canvas_3d_viewer_external_display_active(True)

        self.assertEqual(
            canvas_viewer_tabs.currentIndex(),
            self.workspace.canvas_2d_view_tab_index,
        )
        self.assertFalse(
            canvas_viewer_tabs.isTabEnabled(
                self.workspace.canvas_3d_view_tab_index
            )
        )
        self.assertTrue(canvas_viewer_tabs.tabBar().isHidden())

        self.workspace.set_canvas_3d_viewer_external_display_active(False)

        self.assertTrue(
            canvas_viewer_tabs.isTabEnabled(
                self.workspace.canvas_3d_view_tab_index
            )
        )
        self.assertFalse(canvas_viewer_tabs.tabBar().isHidden())

    def test_canvas_tab_refreshes_standard_viewer_preview(self) -> None:
        self.workspace.workspace_tabs.setCurrentWidget(
            self.workspace.canvas_viewer_workspace
        )

        with patch.object(
            self.workspace,
            "_refresh_viewer_preview",
        ) as refresh_mock:
            self.workspace._schedule_viewer_preview_refresh(
                preserve_camera=False
            )
            _qt_application.processEvents()

        refresh_mock.assert_called_once_with(preserve_camera=False)

    def test_settings_changes_propagate_to_generation_workspace(self) -> None:
        settings_widget = self.workspace.settings_widget
        settings_widget.meshy_api_key_edit.setText("meshy-session-test")
        _qt_application.processEvents()

        self.assertEqual(
            self.workspace.generation.get_runtime_settings(),
            GenerationServiceSettings(
                meshy_api_key="meshy-session-test",
            ),
        )

    def test_save_passes_generation_data_without_legacy_sync_state(self) -> None:
        generation = _generation_data()
        self.workspace.generation.set_data(generation)
        save_path = str(Path(self._temporary_directory.name) / "project.json")

        with (
            patch(
                "housemaker.main.QFileDialog.getSaveFileName",
                return_value=(save_path, "JSON Files (*.json)"),
            ),
            patch("housemaker.main.save_project") as save_project_mock,
            patch("housemaker.main.QMessageBox.information"),
        ):
            self.workspace._handle_save_clicked()

        save_project_mock.assert_called_once()
        save_arguments = save_project_mock.call_args.kwargs
        self.assertEqual(save_arguments["generation"], generation)
        self.assertNotIn("dynamic_generation", save_arguments)

    def test_load_applies_project_generation_data(self) -> None:
        generation = _generation_data()
        loaded_project = _project_data(generation)
        load_path = str(Path(self._temporary_directory.name) / "project.json")

        with (
            patch(
                "housemaker.main.QFileDialog.getOpenFileName",
                return_value=(load_path, "JSON Files (*.json)"),
            ),
            patch(
                "housemaker.main.load_project",
                return_value=loaded_project,
            ) as load_project_mock,
        ):
            self.workspace._handle_load_clicked()
            _qt_application.processEvents()

        load_project_mock.assert_called_once_with(load_path)
        self.assertEqual(self.workspace.generation.get_data(), generation)

    def test_load_during_generation_is_rejected_without_partial_state(self) -> None:
        current_generation = _generation_data()
        self.workspace.generation.set_data(current_generation)
        self.workspace.levels[0].name = "Current project sentinel"
        original_levels = copy.deepcopy(self.workspace.levels)
        original_level_index = self.workspace.current_level_index
        original_library_paths = list(self.workspace.image_library_paths)
        original_doorway_presets = list(self.workspace.doorway_presets)
        original_camera = self.workspace.initial_first_person_camera

        incoming_generation = GenerationData()
        incoming_project = _project_data(incoming_generation)
        incoming_project.levels[0].name = "Incoming project sentinel"
        incoming_project.current_level_index = 5
        load_path = str(Path(self._temporary_directory.name) / "incoming.json")

        planner = _BlockingMeshyPlanner()
        self.workspace.generation.set_meshy_planner(planner)
        self.workspace.generation._start_generation(_generation_request())
        self.assertTrue(planner.started.wait(timeout=1.0))

        try:
            with (
                patch(
                    "housemaker.main.QFileDialog.getOpenFileName",
                    return_value=(load_path, "JSON Files (*.json)"),
                ),
                patch(
                    "housemaker.main.load_project",
                    return_value=incoming_project,
                ),
                patch("housemaker.main.QMessageBox.critical") as critical_mock,
            ):
                self.workspace._handle_load_clicked()
                _qt_application.processEvents()

            critical_mock.assert_called_once()
            self.assertEqual(critical_mock.call_args.args[1], "Project load failed")
            self.assertEqual(self.workspace.levels, original_levels)
            self.assertEqual(
                self.workspace.current_level_index,
                original_level_index,
            )
            self.assertEqual(
                self.workspace.image_library_paths,
                original_library_paths,
            )
            self.assertEqual(
                self.workspace.doorway_presets,
                original_doorway_presets,
            )
            self.assertEqual(
                self.workspace.initial_first_person_camera,
                original_camera,
            )
            self.assertEqual(
                self.workspace.generation.get_data(),
                current_generation,
            )
        finally:
            planner.release.set()

    def test_legacy_dynamic_sync_controls_and_attributes_are_absent(self) -> None:
        self.assertFalse(hasattr(self.workspace, "dynamic_generation"))

        forbidden_generation_attributes = (
            "plan_view",
            "first_person_view",
            "score_timeline",
            "manual_alignment_button",
            "start_sync_button",
            "stop_sync_button",
            "pass_list",
            "stop_after_passes_spinbox",
            "_sync_thread",
            "_sync_worker",
        )
        for attribute_name in forbidden_generation_attributes:
            with self.subTest(attribute=attribute_name):
                self.assertFalse(
                    hasattr(self.workspace.generation, attribute_name)
                )

        button_texts = {
            button.text()
            for button in self.workspace.generation.findChildren(QPushButton)
        }
        self.assertTrue(
            {
                "Manual alignment",
                "Start sync",
                "Stop sync",
                "Stop pass",
                "Start pass",
            }.isdisjoint(button_texts)
        )


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
