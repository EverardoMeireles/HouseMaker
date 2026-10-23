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
from PySide6.QtGui import QImage, QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton, QWidget

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.camera_models import CameraPose
from housemaker.ceiling_height_estimation import (
    CEILING_HEIGHT_MODEL_OPTIONS,
    CeilingHeightAnalysisResult,
    CeilingHeightLevelEstimate,
    CeilingHeightVideoEstimate,
)
from housemaker.door_state import (
    DOOR_SLOT_BODY,
    DOOR_SLOT_KNOB,
    DoorDefinition,
    DoorLibraryData,
    DoorSlotData,
    WindowDefinition,
)
from housemaker.first_person_navigation import (
    FIRST_PERSON_NAVIGATION_MODE_NOCLIP,
)
from housemaker.generation_state import GenerationData, MaskPoint, MaskStroke
from housemaker.generation_workspace import GenerationRequest
from housemaker.geometry_generation_models import GEOMETRY_MODEL_7_1
from housemaker.main import BlueprintWorkspace
from housemaker.models import GROUND_LEVEL_INDEX, create_default_levels
from housemaker.object_texture_variants import (
    PBR_MAP_NORMAL,
    PBR_MAP_ROUGHNESS,
)
from housemaker.project_io import ProjectData
from housemaker.reference_generation_models import (
    DEFAULT_REFERENCE_GENERATION_MODEL,
    REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION,
)
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


def _ceiling_height_result(
    *level_estimates: tuple[int, str, float],
    unassigned_frame_indices: tuple[int, ...] = (),
) -> CeilingHeightAnalysisResult:
    """Build one concise, reliable multi-level result fixture."""

    estimates = tuple(
        CeilingHeightLevelEstimate(
            level_index=level_index,
            level_name=level_name,
            analysis=CeilingHeightVideoEstimate(
                average_height_m=height_meters,
                retained_frame_indices=(0, 15, 30),
                outlier_frame_indices=(45,),
                insufficient_frame_indices=(),
            ),
        )
        for level_index, level_name, height_meters in level_estimates
    )
    return CeilingHeightAnalysisResult(
        level_estimates=estimates,
        sampled_frame_indices=(0, 15, 30, 45),
        transition_frame_indices=unassigned_frame_indices,
        unassigned_frame_indices=unassigned_frame_indices,
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

    def test_reference_model_settings_update_generation_controls(self) -> None:
        merged = self.workspace.merged_generation_workspace
        settings = self.workspace.settings_widget
        combo = settings.reference_generation_model_combo
        self.assertTrue(settings.isAncestorOf(combo))
        self.assertFalse(merged.isAncestorOf(combo))

        for model, stable_diffusion in (
            (REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION, True),
            (DEFAULT_REFERENCE_GENERATION_MODEL, False),
        ):
            with self.subTest(model=model):
                combo.setCurrentIndex(combo.findData(model))
                _qt_application.processEvents()
                self.assertEqual(merged._reference_generation_model, model)
                self.assertEqual(settings.get_settings().reference_generation_model, model)
                self.assertEqual(
                    merged.reference_generation_strength_control.isHidden(),
                    not stable_diffusion,
                )
                self.assertEqual(
                    merged.reference_generation_negative_prompt_control.isHidden(),
                    not stable_diffusion,
                )

    def test_ceiling_height_model_settings_update_analysis_dialog(self) -> None:
        merged = self.workspace.merged_generation_workspace
        settings = self.workspace.settings_widget
        combo = settings.ceiling_height_estimation_model_combo
        dialog = merged._ensure_ceiling_height_analysis_dialog()

        self.assertTrue(settings.isAncestorOf(combo))
        self.assertFalse(dialog.isAncestorOf(combo))
        self.assertFalse(hasattr(dialog, "model_combo"))
        self.assertEqual(
            dialog.selected_model,
            settings.get_settings().ceiling_height_estimation_model,
        )

        for _label, model in CEILING_HEIGHT_MODEL_OPTIONS:
            with self.subTest(model=model):
                combo.setCurrentIndex(combo.findData(model))
                _qt_application.processEvents()
                self.assertEqual(dialog.selected_model, model)
                self.assertEqual(
                    settings.get_settings().ceiling_height_estimation_model,
                    model,
                )

    def test_workspace_tabs_and_canvas_panel_visibility(self) -> None:
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
                "Doors and Windows",
                "Tour",
                "Settings",
            ],
        )

        self.assertEqual(self.workspace.workspace_splitter.count(), 3)
        self.assertIs(
            self.workspace.workspace_splitter.widget(0),
            self.workspace.left_panel,
        )
        self.assertIs(
            self.workspace.workspace_splitter.widget(1),
            self.workspace.workspace_tabs,
        )
        self.assertIs(
            self.workspace.workspace_splitter.widget(2),
            self.workspace.side_panel,
        )

        expected_panel_visibility = {
            "Canvas": True,
            "3D scene": True,
            "Atlas": False,
            "Generation": False,
            "Doors and Windows": False,
            "Tour": False,
            "Settings": False,
        }
        for tab_name, should_be_visible in expected_panel_visibility.items():
            with self.subTest(tab=tab_name):
                tab_index = tab_names.index(tab_name)
                self.workspace.workspace_tabs.setCurrentIndex(tab_index)
                _qt_application.processEvents()
                self.assertEqual(
                    self.workspace.side_panel.isVisible(),
                    should_be_visible,
                )
                self.assertEqual(
                    self.workspace.left_panel.isVisible(),
                    should_be_visible,
                )

    def test_canvas_panels_stay_inside_window_after_tab_round_trip(
        self,
    ) -> None:
        requested_width = 1400
        self.workspace.resize(requested_width, self.workspace.height())
        _qt_application.processEvents()

        for full_width_workspace in (
            self.workspace.scene_3d_workspace,
            self.workspace.texture_atlas_workspace,
            self.workspace.merged_generation_workspace,
            self.workspace.doors_workspace,
            self.workspace.tour_workspace,
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
                self.assertEqual(self.workspace.width(), requested_width)
                for panel in (
                    self.workspace.left_panel,
                    self.workspace.side_panel,
                ):
                    with self.subTest(panel=panel):
                        panel_geometry = panel.geometry()
                        self.assertTrue(panel.isVisible())
                        self.assertGreater(panel_geometry.width(), 0)
                        self.assertGreaterEqual(
                            panel_geometry.left(), splitter_rect.left()
                        )
                        self.assertLessEqual(
                            panel_geometry.right(), splitter_rect.right()
                        )
                tabs_geometry = self.workspace.workspace_tabs.geometry()
                self.assertGreater(self.workspace.canvas.width(), 0)
                self.assertLessEqual(
                    self.workspace.canvas.width(), tabs_geometry.width()
                )
                self.assertLess(
                    self.workspace.left_panel.geometry().right(),
                    tabs_geometry.left(),
                )
                self.assertLess(
                    tabs_geometry.right(),
                    self.workspace.side_panel.geometry().left(),
                )

    # ### Hidden tall tabs must not force a larger top-level window ###
    def test_hidden_workspace_pages_do_not_force_window_height_on_short_screens(self) -> None:
        for requested_height in (900, 700):
            for index in range(self.workspace.workspace_tabs.count()):
                with self.subTest(
                    height=requested_height,
                    tab=self.workspace.workspace_tabs.tabText(index),
                ):
                    self.workspace.workspace_tabs.setCurrentIndex(index)
                    self.workspace.resize(1_400, requested_height)
                    _qt_application.processEvents()
                    _qt_application.processEvents()
                    self.assertLessEqual(self.workspace.height(), requested_height)
                    self.assertLessEqual(self.workspace.minimumSizeHint().height(), requested_height)
                    tabs = self.workspace.workspace_tabs
                    self.assertGreater(tabs.height(), 0)
                    self.assertGreaterEqual(tabs.geometry().top(), 0)
                    self.assertLessEqual(tabs.geometry().bottom(), self.workspace.rect().bottom())
                    for column in (self.workspace.left_panel, self.workspace.side_panel):
                        if column.isVisible():
                            self.assertGreaterEqual(column.geometry().top(), 0)
                            self.assertLessEqual(column.geometry().bottom(), self.workspace.rect().bottom())

    def test_short_window_keeps_atlas_and_tour_tools_readable_and_scrollable(self) -> None:
        atlas = self.workspace.texture_atlas_workspace
        tour = self.workspace.tour_workspace
        tour.start_new_tour()
        self.assertTrue(tour.accept_world_point((4.0, 5.0, 1.0)))
        self.assertTrue(tour.accept_world_point((0.0, 0.0, 1.5)))
        self.assertTrue(tour.accept_world_point((5.0, 0.0, 1.5)))
        self.assertTrue(tour.finish_curve())
        tour.add_floating_tooltip_button.click()
        cases = (
            (
                atlas, atlas.texture_controls_scroll, atlas.color_balance_apply_button,
                atlas.object_preview_container,
                (atlas.color_balance_reset_button, atlas.color_balance_apply_button),
            ),
            (
                tour, tour.steps_controls_scroll, tour.tooltip_style_edit,
                tour.preview_container,
                (tour.add_text_button, tour.add_floating_tooltip_button, tour.add_wait_button,
                 tour.add_idle_camera_button, tour.add_speed_override_button),
            ),
        )
        for page, scroll, last_control, preview, buttons in cases:
            with self.subTest(page=type(page).__name__):
                self.workspace.workspace_tabs.setCurrentWidget(page)
                self.workspace.resize(1_400, 700)
                _qt_application.processEvents()
                _qt_application.processEvents()
                self.assertLessEqual(self.workspace.height(), 700)
                self.assertFalse(scroll.isAncestorOf(preview))
                self.assertGreater(scroll.verticalScrollBar().maximum(), 0)
                for button in buttons:
                    self.assertGreaterEqual(button.height(), button.minimumSizeHint().height())
                preview_geometry = preview.geometry()
                scroll.ensureWidgetVisible(last_control)
                _qt_application.processEvents()
                target_center = last_control.mapTo(scroll.viewport(), last_control.rect().center())
                self.assertTrue(
                    scroll.viewport().rect().contains(target_center),
                    (target_center, scroll.viewport().rect(), last_control.isVisibleTo(page),
                     scroll.verticalScrollBar().value(), scroll.verticalScrollBar().maximum()),
                )
                self.assertEqual(preview.geometry(), preview_geometry)

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

    def test_confirmed_ceiling_analysis_updates_levels_as_one_undo_action(
        self,
    ) -> None:
        ground_level = self.workspace._get_level_by_index(GROUND_LEVEL_INDEX)
        story_level = self.workspace._get_level_by_index(GROUND_LEVEL_INDEX + 1)
        assert ground_level is not None
        assert story_level is not None
        original_ground_height = ground_level.height_meters
        original_story_height = story_level.height_meters
        undo_count = len(self.workspace._canvas_undo_stack)
        result = _ceiling_height_result(
            (GROUND_LEVEL_INDEX, ground_level.display_name, 2.74),
            (GROUND_LEVEL_INDEX + 1, story_level.display_name, 2.91),
            unassigned_frame_indices=(60, 75),
        )

        with patch(
            "housemaker.main.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Yes,
        ) as question:
            self.workspace.merged_generation_workspace.ceiling_height_estimate_changed.emit(
                result
            )

        self.assertEqual(question.call_count, 1)
        question_text = question.call_args.args[2]
        self.assertIn(ground_level.display_name, question_text)
        self.assertIn(story_level.display_name, question_text)
        self.assertIn("2 transition or uncertain frames", question_text)
        self.assertAlmostEqual(ground_level.height_meters, 2.74)
        self.assertAlmostEqual(story_level.height_meters, 2.91)
        self.assertAlmostEqual(self.workspace.height_level_spinbox.value(), 2.74)
        self.assertEqual(len(self.workspace._canvas_undo_stack), undo_count + 1)

        self.workspace._handle_canvas_undo_requested()

        self.assertAlmostEqual(ground_level.height_meters, original_ground_height)
        self.assertAlmostEqual(story_level.height_meters, original_story_height)
        self.assertEqual(len(self.workspace._canvas_undo_stack), undo_count)

    def test_confirmed_ceiling_analysis_commits_pending_scale_before_height(
        self,
    ) -> None:
        ground_level = self.workspace._get_level_by_index(GROUND_LEVEL_INDEX)
        story_level = self.workspace._get_level_by_index(GROUND_LEVEL_INDEX + 1)
        assert ground_level is not None
        assert story_level is not None
        original_scale = ground_level.scale
        original_ground_height = ground_level.height_meters
        original_story_height = story_level.height_meters
        undo_count = len(self.workspace._canvas_undo_stack)
        next_scale = original_scale + 0.25
        self.workspace._handle_level_scale_changed(next_scale)
        self.assertIsNotNone(self.workspace._pending_level_transform)
        self.assertAlmostEqual(ground_level.scale, original_scale)
        result = _ceiling_height_result(
            (GROUND_LEVEL_INDEX, ground_level.display_name, 2.74),
            (GROUND_LEVEL_INDEX + 1, story_level.display_name, 2.91),
        )

        with patch(
            "housemaker.main.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Yes,
        ):
            self.workspace.merged_generation_workspace.ceiling_height_estimate_changed.emit(
                result
            )

        self.assertIsNone(self.workspace._pending_level_transform)
        self.assertAlmostEqual(ground_level.scale, next_scale)
        self.assertAlmostEqual(ground_level.height_meters, 2.74)
        self.assertAlmostEqual(story_level.height_meters, 2.91)
        self.assertEqual(len(self.workspace._canvas_undo_stack), undo_count + 2)

        self.workspace._handle_canvas_undo_requested()

        self.assertAlmostEqual(ground_level.scale, next_scale)
        self.assertAlmostEqual(ground_level.height_meters, original_ground_height)
        self.assertAlmostEqual(story_level.height_meters, original_story_height)
        self.assertEqual(len(self.workspace._canvas_undo_stack), undo_count + 1)

        self.workspace._handle_canvas_undo_requested()

        self.assertAlmostEqual(ground_level.scale, original_scale)
        self.assertEqual(len(self.workspace._canvas_undo_stack), undo_count)

    def test_rejected_ceiling_analysis_keeps_current_level_height(self) -> None:
        original_height = self.workspace.current_level.height_meters
        result = _ceiling_height_result(
            (
                self.workspace.current_level.index,
                self.workspace.current_level.display_name,
                2.74,
            )
        )

        with patch(
            "housemaker.main.QMessageBox.question",
            return_value=QMessageBox.StandardButton.No,
        ):
            self.workspace.merged_generation_workspace.ceiling_height_estimate_changed.emit(
                result
            )

        self.assertAlmostEqual(
            self.workspace.current_level.height_meters,
            original_height,
        )

    def test_ceiling_height_level_to_mark_tracks_image_backed_canvas_selection(
        self,
    ) -> None:
        merged = self.workspace.merged_generation_workspace
        plan_path = Path(self._temporary_directory.name) / "ceiling-plan.png"
        plan_image = QImage(80, 80, QImage.Format.Format_RGB32)
        plan_image.fill(Qt.GlobalColor.white)
        self.assertTrue(plan_image.save(str(plan_path)))
        image_level_indices = (GROUND_LEVEL_INDEX, GROUND_LEVEL_INDEX + 1)
        for level in self.workspace.levels:
            if level.index in image_level_indices:
                level.image_path = str(plan_path)
        merged.set_ceiling_height_levels(
            self.workspace.levels, self.workspace.current_level.index
        )
        dialog = merged._ensure_ceiling_height_analysis_dialog()
        target_position = next(
            position
            for position, level in enumerate(self.workspace.levels)
            if level.index == GROUND_LEVEL_INDEX + 1
        )

        self.assertEqual(dialog.selected_level_index, GROUND_LEVEL_INDEX)
        self.assertEqual(
            {
                dialog.level_to_mark_combo.itemData(index)
                for index in range(dialog.level_to_mark_combo.count())
            },
            set(image_level_indices),
        )
        self.assertFalse(hasattr(dialog, "current_level_combo"))

        self.workspace._handle_level_selection_changed(target_position)

        self.assertEqual(dialog.selected_level_index, GROUND_LEVEL_INDEX + 1)
        self.assertEqual(
            dialog.level_to_mark_combo.currentText(),
            self.workspace.current_level.display_name,
        )

    def test_shared_pbr_maps_control_only_requested_preview_lights(self) -> None:
        generation_viewer = self.workspace.generation.result_view
        doors_viewer = self.workspace.doors_workspace.preview_viewer
        atlas_viewer = self.workspace.atlas_object_preview_viewer
        normal = self.workspace.generation.pbr_map_checkboxes[PBR_MAP_NORMAL]
        roughness = self.workspace.generation.pbr_map_checkboxes[
            PBR_MAP_ROUGHNESS
        ]

        for checkbox in self.workspace.generation.pbr_map_checkboxes.values():
            checkbox.setChecked(False)

        normal.setChecked(True)

        self.assertTrue(generation_viewer.is_orbiting_point_light_enabled())
        self.assertTrue(doors_viewer.is_orbiting_point_light_enabled())
        self.assertTrue(atlas_viewer.is_orbiting_point_light_enabled())
        self.assertFalse(self.workspace.viewer.is_orbiting_point_light_enabled())

        roughness.setChecked(True)
        normal.setChecked(False)

        self.assertTrue(generation_viewer.is_orbiting_point_light_enabled())
        self.assertTrue(doors_viewer.is_orbiting_point_light_enabled())
        self.assertTrue(atlas_viewer.is_orbiting_point_light_enabled())

        roughness.setChecked(False)

        self.assertFalse(generation_viewer.is_orbiting_point_light_enabled())
        self.assertFalse(doors_viewer.is_orbiting_point_light_enabled())
        self.assertFalse(atlas_viewer.is_orbiting_point_light_enabled())

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
        self.assertTrue(generation.cancel_operation_button.isHidden())
        self.assertFalse(
            merged.shared_material_section.isAncestorOf(
                generation.cancel_operation_button
            )
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
            merged.pbr_map_control,
            merged.mask_mode_control,
            merged.clear_mask_button,
            merged.ai_prompt_edit,
        ):
            with self.subTest(
                shared_control=(
                    shared_control.objectName()
                    or type(shared_control).__name__
                )
            ):
                self.assertTrue(shared_column.isAncestorOf(shared_control))

        for room_analysis_control in (
            merged.infer_ceiling_height_button,
            merged.ceiling_height_result_label,
        ):
            with self.subTest(room_control=room_analysis_control.objectName()):
                self.assertFalse(shared_fields.isAncestorOf(room_analysis_control))
                self.assertFalse(merged.isAncestorOf(room_analysis_control))
                self.assertTrue(
                    self.workspace.side_panel.isAncestorOf(room_analysis_control)
                )

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

    def test_main_window_keeps_generation_controls_shrinkable_after_room_analysis_move(self) -> None:
        merged = self.workspace.merged_generation_workspace
        self.workspace.workspace_tabs.setCurrentWidget(merged)
        _qt_application.processEvents()
        self.assertFalse(merged.isAncestorOf(merged.ceiling_height_section))
        self.assertEqual(merged.controls_scroll.minimumHeight(), 96)
        self.assertLess(
            merged.controls_scroll.minimumHeight(), merged.controls_row.sizeHint().height(),
        )
        self.assertIs(merged.rows_splitter.widget(0), merged.preview_row)
        self.assertIs(merged.rows_splitter.widget(1), merged.controls_scroll)
        merged.rows_splitter.setSizes([merged.rows_splitter.height() - 120, 120])
        _qt_application.processEvents()
        self.assertLessEqual(merged.rows_splitter.sizes()[1], 140)
        self.assertGreater(merged.controls_scroll.verticalScrollBar().maximum(), 0)

    def test_generation_row_splitter_keeps_user_size_after_tab_round_trip(self) -> None:
        merged = self.workspace.merged_generation_workspace
        self.workspace.workspace_tabs.setCurrentWidget(merged)
        _qt_application.processEvents()
        merged.rows_splitter.setSizes([merged.rows_splitter.height() - 140, 140])
        _qt_application.processEvents()
        before = merged.rows_splitter.sizes()[1]
        self.workspace.workspace_tabs.setCurrentWidget(self.workspace.canvas_viewer_workspace)
        _qt_application.processEvents()
        self.workspace.workspace_tabs.setCurrentWidget(merged)
        _qt_application.processEvents()
        self.assertLessEqual(abs(merged.rows_splitter.sizes()[1] - before), 5)
        self.assertGreater(merged.controls_scroll.verticalScrollBar().maximum(), 0)

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
        speed_slider = self.workspace.settings_widget.first_person_speed_slider

        combo.setCurrentIndex(
            combo.findData(FIRST_PERSON_NAVIGATION_MODE_NOCLIP)
        )
        speed_slider.setValue(38)
        _qt_application.processEvents()

        self.assertEqual(
            viewer.get_first_person_movement_mode(),
            FIRST_PERSON_NAVIGATION_MODE_NOCLIP,
        )
        self.assertEqual(viewer.get_first_person_speed(), 3.8)
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

    def test_surface_sidedness_settings_rebuild_the_shared_scene(self) -> None:
        settings_widget = self.workspace.settings_widget

        with patch.object(
            self.workspace,
            "_schedule_viewer_preview_refresh",
        ) as schedule_refresh:
            settings_widget.make_floors_double_sided_checkbox.setChecked(True)
            _qt_application.processEvents()
            settings_widget.make_ceilings_double_sided_checkbox.setChecked(True)
            _qt_application.processEvents()

        self.assertTrue(
            self.workspace._generation_settings.make_floors_double_sided
        )
        self.assertTrue(
            self.workspace._generation_settings.make_ceilings_double_sided
        )
        self.assertEqual(schedule_refresh.call_count, 2)
        schedule_refresh.assert_called_with(preserve_camera=True)

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

    def test_point_light_intensity_updates_the_doors_preview_live(self) -> None:
        settings_widget = self.workspace.settings_widget

        settings_widget.doors_and_windows_point_light_intensity_spinbox.setValue(
            3.2
        )
        _qt_application.processEvents()
        doors_preview = self.workspace.doors_workspace.preview_viewer

        self.assertEqual(
            doors_preview.get_point_light_intensity(),
            3.2,
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

    def test_restored_opening_profile_stays_hidden_until_slot_generation(
        self,
    ) -> None:
        merged = self.workspace.merged_generation_workspace
        doors = self.workspace.doors_workspace

        for asset_kind in ("door", "window"):
            with self.subTest(asset_kind=asset_kind):
                asset_id = f"{asset_kind}-1"
                slots = (
                    DoorSlotData(
                        slot_id=DOOR_SLOT_BODY,
                        source_object_id=f"{asset_id}-body",
                        joined=True,
                    ),
                    DoorSlotData(
                        slot_id=DOOR_SLOT_KNOB,
                        source_object_id=f"{asset_id}-handle",
                    ),
                )
                if asset_kind == "door":
                    library = DoorLibraryData(
                        doors=[
                            DoorDefinition(
                                door_id=asset_id,
                                name="Restored door",
                                source_doorway_id="doorway-1",
                                width_meters=0.9,
                                height_meters=2.1,
                                slots=slots,
                            )
                        ]
                    )
                else:
                    library = DoorLibraryData(
                        windows=[
                            WindowDefinition(
                                window_id=asset_id,
                                name="Restored window",
                                source_window_frame_id="frame-1",
                                width_meters=1.2,
                                height_meters=1.0,
                                slots=slots,
                            )
                        ]
                    )

                self.workspace._apply_project_state(
                    levels=create_default_levels(),
                    current_level_index=GROUND_LEVEL_INDEX,
                    doors=library,
                )
                _qt_application.processEvents()

                self.assertEqual(doors.selected_asset_kind(), asset_kind)
                target = self.workspace.generation._door_slot_editing_target
                self.assertIsNotNone(target)
                assert target is not None
                self.assertFalse(target.generation_ready)
                self.assertTrue(merged.opening_profile_options_section.isHidden())
                self.assertFalse(merged.division_options_section.isHidden())

                doors.asset_slot_edit_requested.emit(
                    asset_kind,
                    asset_id,
                    DOOR_SLOT_BODY,
                )
                _qt_application.processEvents()

                target = self.workspace.generation._door_slot_editing_target
                self.assertIsNotNone(target)
                assert target is not None
                self.assertTrue(target.generation_ready)
                self.assertFalse(merged.opening_profile_options_section.isHidden())
                self.assertIs(self.workspace.workspace_tabs.currentWidget(), merged)

                self.assertTrue(
                    doors.select_asset_part(
                        asset_kind,
                        asset_id,
                        slot_id=DOOR_SLOT_KNOB,
                    )
                )
                self.assertTrue(merged.opening_profile_options_section.isHidden())
                self.assertFalse(merged.division_options_section.isHidden())

    # ### Restored selections must not capture ordinary pasted references ###
    def test_restored_opening_does_not_capture_pasted_object_geometry(self) -> None:
        merged = self.workspace.merged_generation_workspace
        generation = self.workspace.generation
        image = QImage(48, 32, QImage.Format.Format_RGBA8888)
        image.fill(0xFF804020)
        for asset_kind in ("door", "window"):
            with self.subTest(asset_kind=asset_kind):
                asset_id = f"passive-{asset_kind}"
                common = {
                    "name": f"Restored {asset_kind}",
                    "width_meters": 0.9,
                    "height_meters": 2.1,
                    "slots": (
                        DoorSlotData(DOOR_SLOT_BODY, f"{asset_id}-body", joined=True),
                        DoorSlotData(DOOR_SLOT_KNOB, f"{asset_id}-handle"),
                    ),
                }
                if asset_kind == "door":
                    library = DoorLibraryData(doors=[DoorDefinition(
                        door_id=asset_id, source_doorway_id="source", **common,
                    )])
                else:
                    library = DoorLibraryData(windows=[WindowDefinition(
                        window_id=asset_id, source_window_frame_id="source", **common,
                    )])
                self.workspace._apply_project_state(
                    levels=create_default_levels(),
                    current_level_index=GROUND_LEVEL_INDEX,
                    doors=library,
                )
                self.workspace.workspace_tabs.setCurrentWidget(merged)
                generation.set_runtime_settings(
                    GenerationServiceSettings(meshy_api_key="mock-api-key")
                )
                target = generation._door_slot_editing_target
                self.assertIsNotNone(target)
                self.assertFalse(target.generation_ready)
                with patch(
                    "housemaker.merged_generation_workspace._read_clipboard_image",
                    return_value=image,
                ):
                    merged._handle_paste_inpaint_clicked()
                reference = generation.get_temporary_object_reference()
                self.assertIsNotNone(reference)
                self.assertTrue(generation.generate_geometry_button.isEnabled())

                with (
                    patch.object(generation, "_start_generation", return_value="ordinary") as start,
                    patch.object(generation, "_start_door_slot_model_generation") as slot_start,
                ):
                    generation.generate_geometry_button.click()

                slot_start.assert_not_called()
                start.assert_called_once()
                request = start.call_args.args[0]
                self.assertTrue(request.geometry_only)
                self.assertIsNone(request.door_body_fit_dimensions)
                self.assertIsNone(start.call_args.kwargs.get("door_slot_target"))
                np.testing.assert_array_equal(request.selected_object_bgra, reference)
                self.assertFalse(generation._door_slot_editing_target.generation_ready)
                self.assertTrue(merged.opening_profile_options_section.isHidden())
                self.assertTrue(merged.door_slot_generation_indicator.isHidden())

    # ### Independent geometry provider controls ###
    def test_meshy71_remains_available_without_multi_view_generation(self) -> None:
        merged = self.workspace.merged_generation_workspace
        self.workspace.workspace_tabs.setCurrentWidget(merged)
        provider = merged.geometry_provider_combo
        self.assertGreaterEqual(provider.findData(GEOMETRY_MODEL_7_1), 0)

        provider.setCurrentIndex(provider.findData(GEOMETRY_MODEL_7_1))
        _qt_application.processEvents()

        self.assertTrue(provider.isEnabled())
        self.assertEqual(
            self.workspace.generation.geometry_generation_options(),
            (GEOMETRY_MODEL_7_1, "standard"),
        )
        self.assertFalse(merged.geometry_resolution_control.isHidden())
        self.assertEqual(
            [
                merged.geometry_resolution_combo.itemData(index)
                for index in range(merged.geometry_resolution_combo.count())
            ],
            ["standard", "2k"],
        )
        self.assertFalse(hasattr(merged, "multi_view_generation_checkbox"))
        self.assertFalse(hasattr(merged, "multi_view_slots_control"))

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
