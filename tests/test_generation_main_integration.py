# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import copy
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton, QWidget

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.camera_models import CameraPose
from housemaker.first_person_navigation import (
    FIRST_PERSON_NAVIGATION_MODE_NOCLIP,
)
from housemaker.generation_state import GenerationData, MaskPoint, MaskStroke
from housemaker.generation_workspace import GenerationRequest
from housemaker.main import BlueprintWorkspace
from housemaker.models import GROUND_LEVEL_INDEX, create_default_levels
from housemaker.project_io import ProjectData
from housemaker.settings_widget import GenerationServiceSettings

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
                "3D scene",
                "Atlas",
                "Generation",
                "Settings",
            ],
        )

        expected_side_panel_visibility = {
            "Canvas": True,
            "3D scene": True,
            "Atlas": False,
            "Generation": False,
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

    def test_canvas_side_panel_stays_inside_the_window_after_tab_round_trip(
        self,
    ) -> None:
        requested_width = 1400
        self.workspace.resize(requested_width, self.workspace.height())
        _qt_application.processEvents()

        for full_width_workspace in (
            self.workspace.texture_atlas_workspace,
            self.workspace.merged_generation_workspace,
            self.workspace.settings_widget,
        ):
            with self.subTest(
                tab=self.workspace.workspace_tabs.tabText(
                    self.workspace.workspace_tabs.indexOf(full_width_workspace)
                )
            ):
                self.workspace.workspace_tabs.setCurrentWidget(
                    full_width_workspace
                )
                _qt_application.processEvents()
                self.workspace.resize(
                    requested_width,
                    self.workspace.height(),
                )
                _qt_application.processEvents()
                self.workspace.workspace_tabs.setCurrentWidget(
                    self.workspace.canvas_viewer_workspace
                )
                _qt_application.processEvents()

                splitter_rect = self.workspace.workspace_splitter.rect()
                side_panel_geometry = self.workspace.side_panel.geometry()
                self.assertEqual(self.workspace.width(), requested_width)
                self.assertTrue(self.workspace.side_panel.isVisible())
                self.assertGreater(side_panel_geometry.width(), 0)
                self.assertGreaterEqual(
                    side_panel_geometry.left(),
                    splitter_rect.left(),
                )
                self.assertLessEqual(
                    side_panel_geometry.right(),
                    splitter_rect.right(),
                )

    def test_generation_uses_only_the_compact_generated_object_view(self) -> None:
        merged = self.workspace.merged_generation_workspace

        self.assertIs(
            merged.views_splitter.widget(1),
            merged.object_workspace.object_3d_page,
        )
        self.assertIsNone(merged.surface_workspace.surface_view)
        self.assertFalse(
            merged.isAncestorOf(self.workspace.viewer)
        )
        self.assertTrue(
            merged.isAncestorOf(self.workspace.generation.object_3d_panel)
        )

    def test_generation_controls_follow_the_merged_workflow_layout(self) -> None:
        merged = self.workspace.merged_generation_workspace
        generation = self.workspace.generation
        object_panel = generation.object_3d_panel
        primary_column = merged.findChild(
            QWidget,
            "merged_generation_object_primary_column",
        )
        self.assertIsNotNone(primary_column)
        assert primary_column is not None
        primary_layout = primary_column.layout()
        assert primary_layout is not None

        editing_section = merged.object_editing_section
        creation_section = merged.object_creation_section
        self.assertLess(
            primary_layout.indexOf(editing_section),
            primary_layout.indexOf(creation_section),
        )
        self.assertLess(
            primary_layout.indexOf(creation_section),
            primary_layout.indexOf(generation.model_statistics_label),
        )
        for editing_control in (
            generation.textures_checkbox,
            generation.wireframe_checkbox,
            generation.delete_selected_faces_button,
            generation.convert_faces_to_glass_button,
        ):
            self.assertTrue(editing_section.isAncestorOf(editing_control))
        for creation_control in (
            generation.symmetric_division_checkbox,
            generation.meshy_target_polycount_control,
            generation.generate_geometry_button,
            generation.generate_texture_button,
            generation.generate_button,
            generation.place_button,
        ):
            self.assertTrue(creation_section.isAncestorOf(creation_control))
        self.assertTrue(
            merged.object_controls.isAncestorOf(
                object_panel.projection_camera_controls
            )
        )

        self.assertTrue(
            merged.shared_material_section.isAncestorOf(
                merged.pbr_map_control
            )
        )
        self.assertTrue(
            merged.shared_material_section.isAncestorOf(
                merged.ai_prompt_edit
            )
        )
        self.assertTrue(
            merged.shared_material_section.isAncestorOf(merged.cancel_button)
        )

        shared_fields = merged.shared_controls.findChild(
            QWidget,
            "merged_generation_shared_fields",
        )
        self.assertIsNotNone(shared_fields)
        shared_layout = shared_fields.layout()
        assert shared_layout is not None
        self.assertEqual(shared_layout.count(), 1)
        shared_column = shared_layout.itemAt(0).widget()
        self.assertIsNotNone(shared_column)
        for shared_control in (
            merged.load_video_button,
            merged.infer_ceiling_height_button,
            merged.ceiling_height_result_label,
            merged.pbr_map_control,
            merged.mask_mode_control,
            merged.clear_mask_button,
            merged.ai_prompt_edit,
            merged.cancel_button,
        ):
            with self.subTest(
                shared_control=(
                    shared_control.objectName()
                    or type(shared_control).__name__
                )
            ):
                self.assertTrue(shared_column.isAncestorOf(shared_control))

        forbidden_attributes = (
            "delete_generated_object_button",
            "frame_label",
            "glass_double_sided_checkbox",
            "job_name_edit",
        )
        for attribute_name in forbidden_attributes:
            with self.subTest(attribute=attribute_name):
                self.assertFalse(hasattr(generation, attribute_name))

        button_texts = {
            button.text()
            for button in merged.findChildren(QPushButton)
        }
        self.assertTrue(
            {"Delete object", "Double-sided", "Generate mask"}.isdisjoint(
                button_texts
            )
        )

    def test_canvas_and_3d_scene_are_dedicated_top_level_tabs(self) -> None:
        canvas_tab_index = self.workspace.workspace_tabs.indexOf(
            self.workspace.canvas_viewer_workspace
        )
        scene_tab_index = self.workspace.workspace_tabs.indexOf(
            self.workspace.scene_3d_workspace
        )

        self.assertGreaterEqual(canvas_tab_index, 0)
        self.assertGreaterEqual(scene_tab_index, 0)
        self.assertEqual(
            self.workspace.workspace_tabs.tabText(canvas_tab_index),
            "Canvas",
        )
        self.assertEqual(
            self.workspace.workspace_tabs.tabText(scene_tab_index),
            "3D scene",
        )
        self.assertTrue(
            self.workspace.canvas_viewer_workspace.isAncestorOf(
                self.workspace.canvas
            )
        )
        self.assertTrue(
            self.workspace.scene_3d_workspace.isAncestorOf(
                self.workspace.viewer
            )
        )

    def test_canvas_navigation_hotkey_toggles_first_person_from_current_pose(
        self,
    ) -> None:
        camera_pose = CameraPose(
            x=2.5,
            y=-1.25,
            z=1.7,
            yaw_degrees=42.0,
            pitch_degrees=-6.0,
            fov_degrees=72.0,
        )
        self.workspace.viewer.set_first_person_camera_pose(camera_pose)
        self.workspace.workspace_tabs.setCurrentWidget(
            self.workspace.scene_3d_workspace
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
            camera_pose,
        )
        QTest.keyClick(self.workspace.viewer.view, Qt.Key.Key_N)
        _qt_application.processEvents()

        self.assertEqual(
            self.workspace.viewer.get_navigation_mode(),
            "orbit",
        )

    def test_a_shortcut_toggles_generation_frame_only_in_first_person(
        self,
    ) -> None:
        shortcut = self.workspace.first_person_generation_frame_shortcut
        viewer = self.workspace.viewer
        frame_bgr = np.zeros((5, 7, 3), dtype=np.uint8)
        frame_bgr[:, :] = (31, 97, 223)
        self.workspace.merged_generation_workspace.video_view.set_frame(
            frame_bgr
        )

        self.assertEqual(
            shortcut.key().toString(QKeySequence.SequenceFormat.PortableText),
            "A",
        )
        self.workspace.workspace_tabs.setCurrentWidget(
            self.workspace.scene_3d_workspace
        )
        viewer.view.setFocus()
        QTest.keyClick(viewer.view, Qt.Key.Key_A)
        self.assertFalse(viewer.is_first_person_frame_overlay_visible)

        viewer.enter_first_person_mode()
        QTest.keyClick(viewer.view, Qt.Key.Key_A)

        self.assertTrue(viewer.is_first_person_frame_overlay_visible)

        QTest.keyClick(viewer.view, Qt.Key.Key_A)

        self.assertFalse(viewer.is_first_person_frame_overlay_visible)

    def test_visible_generation_frame_overlay_live_syncs_and_clears(
        self,
    ) -> None:
        shortcut = self.workspace.first_person_generation_frame_shortcut
        viewer = self.workspace.viewer
        first_frame = np.full((4, 6, 3), 45, dtype=np.uint8)
        second_frame = np.full((4, 6, 3), 185, dtype=np.uint8)
        self.workspace.merged_generation_workspace.video_view.set_frame(
            first_frame
        )
        viewer.enter_first_person_mode()

        with patch.object(
            viewer,
            "set_first_person_frame_overlay",
            wraps=viewer.set_first_person_frame_overlay,
        ) as set_overlay:
            shortcut.activated.emit()
            self.assertTrue(viewer.is_first_person_frame_overlay_visible)
            np.testing.assert_array_equal(
                set_overlay.call_args.args[0],
                first_frame,
            )

            self.workspace.merged_generation_workspace.video_view.set_frame(
                second_frame
            )

            self.assertEqual(set_overlay.call_count, 2)
            np.testing.assert_array_equal(
                set_overlay.call_args.args[0],
                second_frame,
            )

            self.workspace.merged_generation_workspace.video_view.clear_frame()

        self.assertFalse(viewer.is_first_person_frame_overlay_visible)

    def test_clear_mask_hotkey_follows_settings_dropdown(self) -> None:
        shortcut = self.workspace.merged_generation_workspace.clear_mask_shortcut
        combo = self.workspace.settings_widget.clear_mask_hotkey_combo
        self.assertEqual(
            shortcut.key().toString(QKeySequence.SequenceFormat.PortableText),
            "Ctrl+Shift+M",
        )

        combo.setCurrentIndex(combo.findData("Alt+C"))
        _qt_application.processEvents()
        self.assertEqual(
            shortcut.key().toString(QKeySequence.SequenceFormat.PortableText),
            "Alt+C",
        )
        self.assertTrue(shortcut.isEnabled())

        combo.setCurrentIndex(combo.findData(""))
        _qt_application.processEvents()
        self.assertFalse(shortcut.isEnabled())

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

    def test_first_person_mode_updates_live_without_resetting_canvas(self) -> None:
        camera_pose = CameraPose(
            x=1.0,
            y=2.0,
            z=3.0,
            yaw_degrees=25.0,
            pitch_degrees=15.0,
        )
        viewer = self.workspace.viewer
        viewer.set_first_person_camera_pose(camera_pose)
        viewer.enter_first_person_mode()
        pointer_was_captured = viewer.is_first_person_pointer_captured
        combo = self.workspace.settings_widget.first_person_navigation_combo

        combo.setCurrentIndex(
            combo.findData(FIRST_PERSON_NAVIGATION_MODE_NOCLIP)
        )
        _qt_application.processEvents()

        self.assertEqual(
            viewer.get_first_person_movement_mode(),
            FIRST_PERSON_NAVIGATION_MODE_NOCLIP,
        )
        self.assertEqual(viewer.get_navigation_mode(), "first_person")
        self.assertEqual(viewer.get_first_person_camera_pose(), camera_pose)
        self.assertEqual(
            viewer.is_first_person_pointer_captured,
            pointer_was_captured,
        )
        self.assertIsNone(self.workspace.surface_texture_generation.surface_view)

    def test_ignore_top_down_ceiling_setting_updates_shared_scene(self) -> None:
        checkbox = (
            self.workspace.settings_widget.ignore_top_down_ceiling_checkbox
        )
        self.assertTrue(checkbox.isChecked())

        with patch.object(
            self.workspace.viewer,
            "set_ignore_top_down_ceiling",
        ) as setter:
            checkbox.setChecked(False)
            _qt_application.processEvents()

        setter.assert_called_once_with(False)
        self.assertFalse(
            self.workspace._generation_settings.ignore_top_down_ceiling
        )

    def test_include_control_is_export_only_for_scene_preview(self) -> None:
        level = self.workspace.current_level
        self.assertTrue(level.include_in_export)

        with patch.object(
            self.workspace,
            "_schedule_viewer_preview_refresh",
        ) as schedule_refresh:
            self.workspace.include_no_radio.setChecked(True)
            _qt_application.processEvents()

        self.assertFalse(level.include_in_export)
        schedule_refresh.assert_not_called()
        preview_level = next(
            candidate
            for candidate in self.workspace._build_viewer_preview_levels()
            if candidate.index == level.index
        )
        self.assertTrue(preview_level.include_in_export)

    def test_scene_level_selector_receives_only_levels_with_vertices(self) -> None:
        lowest_level = min(
            self.workspace.levels,
            key=lambda candidate: candidate.index,
        )
        highest_level = max(
            self.workspace.levels,
            key=lambda candidate: candidate.index,
        )
        lowest_level.vertex_data.add_vertex(10.0, 20.0)
        highest_level.vertex_data.add_vertex(30.0, 40.0)

        with patch.object(
            self.workspace.viewer,
            "set_canvas_scene_levels",
        ) as setter:
            self.workspace._sync_viewer_scene_levels()

        setter.assert_called_once_with(
            (
                (highest_level.index, highest_level.display_name),
                (lowest_level.index, lowest_level.display_name),
            ),
            placed_object_levels={},
            reset_visibility=False,
        )

    def test_canvas_snap_filter_is_controlled_from_settings(self) -> None:
        self.assertFalse(
            hasattr(self.workspace, "snap_middle_equal_angle_radio")
        )
        self.assertTrue(self.workspace.canvas.snap_middle_equal_angle_only)

        snap_checkbox = (
            self.workspace.settings_widget.snap_middle_equal_angle_only_checkbox
        )
        snap_checkbox.setChecked(False)
        _qt_application.processEvents()

        self.assertFalse(self.workspace.canvas.snap_middle_equal_angle_only)

    def test_canvas_has_no_nested_3d_viewer_tabs(self) -> None:
        self.assertFalse(hasattr(self.workspace, "canvas_viewer_tabs"))
        self.assertIs(
            self.workspace.viewer.parentWidget(),
            self.workspace.scene_3d_workspace,
        )

    def test_canvas_tab_refreshes_standard_viewer_preview(self) -> None:
        self.workspace.workspace_tabs.setCurrentWidget(
            self.workspace.scene_3d_workspace
        )
        _qt_application.processEvents()

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
