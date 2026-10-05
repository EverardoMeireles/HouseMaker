# ### Environment setup ###
from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QStyle, QStyleOptionSlider

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.level_coordinates import build_level_base_z_lookup
from housemaker.main import TOUR_CURVE_EYE_HEIGHT_METERS, BlueprintWorkspace
from housemaker.models import GROUND_LEVEL_INDEX, create_default_levels
from housemaker.tour_state import (
    TourCurveData,
    TourData,
    TourFloatingTooltipData,
    TourStepData,
    TourText3DData,
)
from housemaker.tour_workspace import (
    TOUR_POINT_ACTION_CURVE,
    TOUR_POINT_ACTION_TRIGGER,
)
from housemaker.viewer import (
    TOUR_EDIT_TARGET_ACTION,
    TOUR_EDIT_TARGET_ACTION_ROTATION,
    TOUR_EDIT_TARGET_CAMERA_TARGET,
    TOUR_EDIT_TARGET_CURVE_POINT,
    TOUR_EDIT_TARGET_TRIGGER,
    TOUR_POINT_KIND_CURVE,
    TOUR_POINT_KIND_TRIGGER,
    evaluate_open_catmull_rom_curve,
)

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _tour(*, tour_id: str = "main-tour") -> TourData:
    return TourData(
        tour_id=tour_id,
        name="Main integration tour",
        trigger_point=(1.0, 2.0, 0.0),
        curve=TourCurveData(
            points=((0.0, 0.0, 1.6), (10.0, 0.0, 1.6)),
            tension=0.4,
        ),
        steps=(
            TourStepData(
                step_id=f"{tour_id}-start",
                progress=0.0,
                camera_target=(0.0, 1.0, 1.6),
            ),
            TourStepData(
                step_id=f"{tour_id}-end",
                progress=1.0,
                camera_target=(10.0, 1.0, 1.6),
            ),
        ),
        duration_seconds=12.0,
    )


def _process_events() -> None:
    _qt_application.processEvents()


# ### Main integration tests ###
class TourMainIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(
                Path(self._temporary_directory.name) / "settings.json"
            )
        )
        self.workspace.resize(1400, 850)
        self.workspace.show()
        _process_events()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        self.workspace.deleteLater()
        _process_events()
        self._temporary_directory.cleanup()

    def test_tour_tab_is_present_full_width_and_hosts_its_preview(self) -> None:
        tab_names = [
            self.workspace.workspace_tabs.tabText(index)
            for index in range(self.workspace.workspace_tabs.count())
        ]

        self.assertEqual(
            tab_names,
            ["Canvas", "3D scene", "Atlas", "Generation", "Tour", "Settings"],
        )
        self.assertEqual(
            self.workspace.tour_workspace_tab_index,
            tab_names.index("Tour"),
        )
        self.assertIsNone(self.workspace.tour_preview_viewer)
        self.workspace.workspace_tabs.setCurrentWidget(self.workspace.tour_workspace)
        _process_events()

        self.assertFalse(self.workspace.side_panel.isVisible())
        self.assertIsNotNone(self.workspace.tour_preview_viewer)
        self.assertIs(
            self.workspace.tour_workspace._preview_widget,
            self.workspace.tour_preview_viewer,
        )
        self.assertIsNotNone(
            self.workspace.tour_preview_viewer.tour_tooltip_overlay
        )
        self.assertIsNone(self.workspace.viewer.tour_tooltip_overlay)
        self.assertTrue(
            self.workspace.tour_workspace.isAncestorOf(
                self.workspace.tour_preview_viewer
            )
        )

    def test_trigger_to_curve_signal_workflow_arms_and_commits_a_tour(self) -> None:
        viewer = self.workspace.viewer
        tour_workspace = self.workspace.tour_workspace
        level_base_z = build_level_base_z_lookup(self.workspace.levels)[
            self.workspace.current_level.index
        ]

        with (
            patch.object(self.workspace, "_ensure_viewer_preview_current"),
            patch.object(self.workspace, "_refresh_viewer_preview"),
            patch.object(
                viewer,
                "begin_tour_point_placement",
                return_value=True,
            ) as begin_placement,
            patch.object(viewer, "focus_navigation"),
        ):
            tour_workspace.start_new_tour()
            _process_events()

            self.assertEqual(
                tour_workspace.pending_point_action(),
                TOUR_POINT_ACTION_TRIGGER,
            )
            self.assertIs(
                self.workspace.workspace_tabs.currentWidget(),
                self.workspace.scene_3d_workspace,
            )
            begin_placement.assert_called_with(
                TOUR_POINT_KIND_TRIGGER,
                repeat=False,
                vertical_offset_meters=0.0,
                fallback_plane_z=level_base_z,
            )

            viewer.tour_point_placed.emit(
                TOUR_POINT_KIND_TRIGGER,
                {
                    "center": (2.0, 3.0, 0.0),
                    "size": (4.0, 2.5),
                },
            )
            _process_events()

            self.assertEqual(
                tour_workspace.pending_point_action(),
                TOUR_POINT_ACTION_CURVE,
            )
            begin_placement.assert_called_with(
                TOUR_POINT_KIND_CURVE,
                repeat=True,
                vertical_offset_meters=TOUR_CURVE_EYE_HEIGHT_METERS,
                fallback_plane_z=level_base_z,
            )

            viewer.tour_point_placed.emit(
                TOUR_POINT_KIND_CURVE,
                (0.0, 0.0, 1.6),
            )
            _process_events()
            viewer.tour_point_placed.emit(
                TOUR_POINT_KIND_CURVE,
                (4.0, 0.0, 1.6),
            )
            _process_events()
            viewer.tour_curve_finished.emit(((0.0, 0.0, 1.6), (4.0, 0.0, 1.6)))
            _process_events()

        self.assertIsNone(tour_workspace.pending_point_action())
        self.assertEqual(len(tour_workspace.tours()), 1)
        created = tour_workspace.tours()[0]
        self.assertEqual(created.trigger_point, (2.0, 3.0, 0.0))
        self.assertEqual(created.trigger_area_size, (4.0, 2.5))
        self.assertEqual(
            created.curve.points,
            ((0.0, 0.0, 1.6), (4.0, 0.0, 1.6)),
        )
        self.assertIsNone(self.workspace._tour_draft_overlay)
        self.assertEqual(set(viewer._tour_overlays), {created.tour_id})
        self.assertIs(
            self.workspace.workspace_tabs.currentWidget(),
            tour_workspace,
        )

    def test_project_apply_and_save_preserve_tours(self) -> None:
        tour = _tour()
        self.workspace._apply_project_state(
            levels=create_default_levels(),
            current_level_index=GROUND_LEVEL_INDEX,
            tours=(tour,),
        )
        self.assertEqual(self.workspace.tour_workspace.tours(), (tour,))

        save_path = Path(self._temporary_directory.name) / "tour-project.json"
        with (
            patch(
                "housemaker.main.QFileDialog.getSaveFileName",
                return_value=(str(save_path), "JSON Files (*.json)"),
            ),
            patch("housemaker.main.save_project") as save_project,
            patch("housemaker.main.QMessageBox.information"),
        ):
            self.workspace._handle_save_clicked()

        save_project.assert_called_once()
        self.assertEqual(save_project.call_args.kwargs["tours"], (tour,))

    def test_preview_uses_interpolated_target_in_embedded_tour_viewer(self) -> None:
        tour = _tour()
        tour = replace(
            tour,
            steps=(
                replace(
                    tour.steps[0],
                    camera_target=(1.0, 2.0, 3.0),
                ),
                replace(
                    tour.steps[1],
                    camera_target=(5.0, 6.0, 7.0),
                ),
            ),
        )
        self.workspace.tour_workspace.set_tours((tour,))
        self.workspace.workspace_tabs.setCurrentWidget(self.workspace.tour_workspace)
        _process_events()

        with (
            patch.object(self.workspace, "_refresh_viewer_preview"),
            patch.object(
                self.workspace.tour_preview_viewer,
                "set_tour_preview",
            ) as set_tour_preview,
            patch.object(self.workspace.viewer, "set_tour_preview") as shared_preview,
        ):
            self.workspace._handle_tour_preview_requested(tour, 0.5)

        set_tour_preview.assert_called_once()
        preview_progress, camera_target = set_tour_preview.call_args.args
        self.assertEqual(preview_progress, 0.5)
        np.testing.assert_allclose(
            camera_target,
            (3.0, 4.0, 5.0),
            atol=1e-9,
        )
        shared_preview.assert_not_called()

    def test_step_markers_stay_synchronized_in_scene_and_tour_preview(
        self,
    ) -> None:
        tour = _tour()
        tour_workspace = self.workspace.tour_workspace
        tour_workspace.set_tours((tour,))
        self.workspace.workspace_tabs.setCurrentWidget(tour_workspace)
        _process_events()
        preview_viewer = self.workspace.tour_preview_viewer
        assert preview_viewer is not None

        for viewer in (self.workspace.viewer, preview_viewer):
            self.assertEqual(set(viewer._tour_step_overlay_items), {tour.tour_id})
            marker_item = viewer._tour_step_overlay_items[tour.tour_id]
            np.testing.assert_allclose(
                marker_item.pos,
                ((0.0, 0.0, 1.6), (10.0, 0.0, 1.6)),
            )
            self.assertEqual(marker_item.color, (1.0, 1.0, 0.0, 1.0))

        self.assertTrue(
            tour_workspace.select_step(tour.tour_id, tour.steps[0].step_id)
        )
        tour_workspace.step_progress_spin.setValue(25.0)
        _process_events()

        self.assertAlmostEqual(
            tour_workspace.current_tour().steps[0].progress,  # type: ignore[union-attr]
            0.25,
        )
        expected_first_marker = evaluate_open_catmull_rom_curve(
            tour.curve.points,
            0.25,
            tension=tour.curve.tension,
        )
        for viewer in (self.workspace.viewer, preview_viewer):
            np.testing.assert_allclose(
                viewer._tour_step_overlay_items[tour.tour_id].pos,
                (expected_first_marker, (10.0, 0.0, 1.6)),
            )

    def test_idle_preview_pose_only_moves_the_embedded_tour_camera(self) -> None:
        tour = _tour()
        tour_workspace = self.workspace.tour_workspace
        tour_workspace.set_tours((tour,))
        self.workspace.workspace_tabs.setCurrentWidget(tour_workspace)
        _process_events()
        preview_viewer = self.workspace.tour_preview_viewer
        assert preview_viewer is not None
        camera_position = (0.08, 0.0, 1.6)
        camera_target = (0.0, 1.0, 1.6)

        with (
            patch.object(preview_viewer, "set_tour_preview") as set_preview,
            patch.object(
                self.workspace.viewer,
                "set_tour_timeline_progress",
            ) as set_scene_progress,
            patch.object(self.workspace.viewer, "set_tour_preview") as set_scene_pose,
        ):
            tour_workspace.preview_pose_requested.emit(
                tour,
                0.0,
                camera_position,
                camera_target,
            )

        set_preview.assert_called_once_with(
            0.0,
            camera_target,
            camera_position=camera_position,
        )
        set_scene_progress.assert_called_once_with(
            0.0,
            camera_position=camera_position,
            camera_target=camera_target,
        )
        set_scene_pose.assert_not_called()

    def test_timeline_boundary_switches_target_and_repopulates_action_list(
        self,
    ) -> None:
        first_action = TourText3DData(
            component_id="first-step-action",
            text="First step",
            position=(1.0, 1.0, 1.6),
        )
        final_actions = (
            TourText3DData(
                component_id="final-step-action-one",
                text="Final one",
                position=(9.0, 1.0, 1.6),
            ),
            TourText3DData(
                component_id="final-step-action-two",
                text="Final two",
                position=(10.0, 1.0, 1.6),
            ),
        )
        base_tour = _tour()
        tour = replace(
            base_tour,
            steps=(
                replace(base_tour.steps[0], actions=(first_action,)),
                replace(base_tour.steps[1], actions=final_actions),
            ),
        )
        tour_workspace = self.workspace.tour_workspace
        tour_workspace.set_tours((tour,))
        self.workspace.workspace_tabs.setCurrentWidget(tour_workspace)
        _process_events()
        preview_viewer = self.workspace.tour_preview_viewer
        assert preview_viewer is not None

        expected_first_target = {
            (tour.tour_id, tour.steps[0].step_id),
            (tour.tour_id, tour.steps[1].step_id),
        }
        self.assertEqual(
            set(self.workspace.viewer._tour_camera_target_overlay_items),
            expected_first_target,
        )
        self.assertEqual(
            set(preview_viewer._tour_camera_target_overlay_items),
            expected_first_target,
        )

        tour_workspace.set_preview_progress(1.0)
        _process_events()

        self.assertEqual(tour_workspace.selected_step(), tour.steps[1])
        self.assertIsNone(tour_workspace.selected_action())
        self.assertEqual(tour_workspace.action_list.count(), 2)
        self.assertEqual(
            {
                tour_workspace.action_list.item(index).data(
                    Qt.ItemDataRole.UserRole
                )
                for index in range(tour_workspace.action_list.count())
            },
            {action.component_id for action in final_actions},
        )
        expected_final_target = {(tour.tour_id, tour.steps[1].step_id)}
        self.assertEqual(
            set(self.workspace.viewer._tour_camera_target_overlay_items),
            expected_final_target,
        )
        self.assertEqual(
            set(preview_viewer._tour_camera_target_overlay_items),
            expected_final_target,
        )

    def test_next_camera_target_pick_advances_both_viewers(self) -> None:
        base_tour = _tour()
        middle_step = TourStepData(
            step_id="middle-step",
            progress=0.5,
            camera_target=(5.0, 2.0, 1.6),
        )
        tour = replace(
            base_tour,
            steps=(base_tour.steps[0], middle_step, base_tour.steps[1]),
        )
        tour_workspace = self.workspace.tour_workspace
        tour_workspace.set_tours((tour,))
        self.workspace.workspace_tabs.setCurrentWidget(tour_workspace)
        _process_events()
        preview_viewer = self.workspace.tour_preview_viewer
        assert preview_viewer is not None

        self.workspace.viewer._apply_tour_overlay_pick(
            SimpleNamespace(
                tour_id=tour.tour_id,
                kind=TOUR_EDIT_TARGET_CAMERA_TARGET,
                reference=middle_step.step_id,
                step_id=middle_step.step_id,
            )
        )
        _process_events()

        self.assertEqual(tour_workspace.selected_step(), middle_step)
        expected_targets = {
            (tour.tour_id, middle_step.step_id),
            (tour.tour_id, tour.steps[-1].step_id),
        }
        for viewer in (self.workspace.viewer, preview_viewer):
            self.assertEqual(
                set(viewer._tour_camera_target_overlay_items),
                expected_targets,
            )
            self.assertEqual(len(viewer._tour_gizmo_items), 6)
            self.assertEqual(
                viewer._tour_camera_target_overlay_items[
                    (tour.tour_id, middle_step.step_id)
                ]._housemaker_tour_target_timing,
                "current",
            )
            self.assertEqual(
                viewer._tour_camera_target_overlay_items[
                    (tour.tour_id, tour.steps[-1].step_id)
                ]._housemaker_tour_target_timing,
                "next",
            )

    def test_timeline_track_press_and_drag_switch_step_actions_and_target(self) -> None:
        """Exercise scrubbing that begins away from the current slider handle."""

        actions = tuple(
            TourText3DData(
                component_id=f"drag-action-{index}",
                text=f"Drag action {index}",
                position=(float(index * 4), 1.0, 1.6),
            )
            for index in range(3)
        )
        base_tour = _tour()
        tour = replace(
            base_tour,
            steps=tuple(
                TourStepData(
                    step_id=f"drag-step-{index}",
                    progress=progress,
                    camera_target=(float(index * 4), 3.0, 1.6),
                    actions=(actions[index],),
                )
                for index, progress in enumerate((0.0, 0.5, 1.0))
            ),
        )
        tour_workspace = self.workspace.tour_workspace
        tour_workspace.set_tours((tour,))
        self.workspace.workspace_tabs.setCurrentWidget(tour_workspace)
        _process_events()
        preview_viewer = self.workspace.tour_preview_viewer
        assert preview_viewer is not None

        self.assertTrue(
            tour_workspace.select_action(
                tour.tour_id,
                tour.steps[-1].step_id,
                actions[-1].component_id,
            )
        )
        self.assertTrue(
            tour_workspace.set_tour_camera_target(
                tour.tour_id,
                tour.steps[-1].step_id,
                (9.0, 4.0, 1.6),
            )
        )
        _process_events()

        slider = tour_workspace.timeline_slider
        option = QStyleOptionSlider()
        slider.initStyleOption(option)
        handle = slider.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            option,
            QStyle.SubControl.SC_SliderHandle,
            slider,
        )
        groove = slider.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            option,
            QStyle.SubControl.SC_SliderGroove,
            slider,
        )
        left = groove.left() + handle.width() // 2
        right = groove.right() - handle.width() // 2 + 1
        span = max(1, right - left)
        y = handle.center().y()

        middle_position = QPoint(slider._marker_x(0.5) + 4, y)
        QTest.mousePress(
            slider,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            middle_position,
        )
        _process_events()

        self.assertEqual(slider.value(), round(0.5 * slider.maximum()))
        self.assertEqual(tour_workspace.selected_step().step_id, "drag-step-1")  # type: ignore[union-attr]
        self.assertEqual(tour_workspace.action_list.count(), 1)
        self.assertEqual(
            tour_workspace.action_list.item(0).data(Qt.ItemDataRole.UserRole),
            actions[1].component_id,
        )
        expected_middle_target = {
            (tour.tour_id, "drag-step-1"),
            (tour.tour_id, "drag-step-2"),
        }
        self.assertEqual(
            set(self.workspace.viewer._tour_camera_target_overlay_items),
            expected_middle_target,
        )
        self.assertEqual(
            set(preview_viewer._tour_camera_target_overlay_items),
            expected_middle_target,
        )

        first_position = QPoint(round(left + span * 0.25), y)
        QTest.mouseMove(slider, first_position, 20)
        QTest.mouseRelease(
            slider,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            first_position,
        )
        _process_events()

        self.assertEqual(tour_workspace.selected_step().step_id, "drag-step-0")  # type: ignore[union-attr]
        self.assertEqual(tour_workspace.action_list.count(), 1)
        self.assertEqual(
            tour_workspace.action_list.item(0).data(Qt.ItemDataRole.UserRole),
            actions[0].component_id,
        )
        expected_first_target = {
            (tour.tour_id, "drag-step-0"),
            (tour.tour_id, "drag-step-1"),
        }
        self.assertEqual(
            set(self.workspace.viewer._tour_camera_target_overlay_items),
            expected_first_target,
        )
        self.assertEqual(
            set(preview_viewer._tour_camera_target_overlay_items),
            expected_first_target,
        )

    def test_camera_target_uses_preview_forward_point_without_moving_camera(
        self,
    ) -> None:
        tour = _tour()
        tour_workspace = self.workspace.tour_workspace
        tour_workspace.set_tours((tour,))
        self.workspace.workspace_tabs.setCurrentWidget(tour_workspace)
        _process_events()
        viewer = self.workspace.tour_preview_viewer
        assert viewer is not None

        self.assertTrue(
            viewer.set_tour_preview(
                0.0,
                (2.0, 4.0, 1.6),
                camera_position=(2.0, 3.0, 1.6),
            )
        )
        camera_position_before = viewer.view.cameraPosition()
        position_before = (
            camera_position_before.x(),
            camera_position_before.y(),
            camera_position_before.z(),
        )
        direction_before = viewer.get_tour_camera_look_direction()

        tour_workspace.set_camera_target_button.click()
        _process_events()

        np.testing.assert_allclose(
            tour_workspace.selected_step().camera_target,  # type: ignore[union-attr]
            (2.0, 4.0, 1.6),
            atol=1e-6,
        )
        camera_position_after = viewer.view.cameraPosition()
        np.testing.assert_allclose(
            (
                camera_position_after.x(),
                camera_position_after.y(),
                camera_position_after.z(),
            ),
            position_before,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            viewer.get_tour_camera_look_direction(),
            direction_before,
            atol=1e-6,
        )
        selected_step_id = tour_workspace.selected_step().step_id  # type: ignore[union-attr]
        for active_viewer in (self.workspace.viewer, viewer):
            np.testing.assert_allclose(
                active_viewer._tour_camera_target_overlay_items[
                    (tour.tour_id, selected_step_id)
                ].pos[0],
                (2.0, 4.0, 1.6),
                atol=1e-6,
            )
        self.assertIsNone(tour_workspace.pending_point_action())

        with patch.object(
            viewer,
            "get_tour_screen_center_world_point",
            return_value=(2.0, 3.0, 4.0),
        ):
            tour_workspace.add_text_button.click()
        action = tour_workspace.selected_action()
        self.assertIsNotNone(action)
        self.assertEqual(action.position, (2.0, 3.0, 4.0))  # type: ignore[union-attr]
        self.assertEqual(action.text, "Lorem ipsum")  # type: ignore[union-attr]
        self.assertEqual(
            [item.text for item in self.workspace.viewer._tour_text_overlay_items],
            ["Lorem ipsum"],
        )
        self.assertEqual(
            [item.text for item in viewer._tour_text_overlay_items],
            ["Lorem ipsum"],
        )
        self.assertEqual(len(self.workspace.viewer._tour_gizmo_items), 9)
        self.assertEqual(len(viewer._tour_gizmo_items), 9)

        moved_position = (6.0, 7.0, 8.0)
        self.workspace.viewer.tour_edit_finished.emit(
            tour.tour_id,
            TOUR_EDIT_TARGET_ACTION,
            action.component_id,  # type: ignore[union-attr]
            moved_position,
            True,
        )

        moved_action = tour_workspace.selected_action()
        self.assertEqual(moved_action.position, moved_position)  # type: ignore[union-attr]
        np.testing.assert_allclose(
            self.workspace.viewer._get_tour_edit_target_position(
                self.workspace.viewer._tour_edit_target
            ),
            moved_position,
        )
        np.testing.assert_allclose(
            viewer._get_tour_edit_target_position(viewer._tour_edit_target),
            moved_position,
        )

    def test_floating_tooltip_uses_preview_screen_center_as_its_anchor(self) -> None:
        tour = _tour()
        tour_workspace = self.workspace.tour_workspace
        tour_workspace.set_tours((tour,))
        self.workspace.workspace_tabs.setCurrentWidget(tour_workspace)
        _process_events()
        viewer = self.workspace.tour_preview_viewer
        assert viewer is not None
        anchor_point = (3.0, 4.0, 5.0)

        with patch.object(
            viewer,
            "get_tour_screen_center_world_point",
            return_value=anchor_point,
        ) as get_screen_center:
            tour_workspace.add_floating_tooltip_button.click()

        get_screen_center.assert_called_once_with()
        action = tour_workspace.selected_action()
        self.assertIsInstance(action, TourFloatingTooltipData)
        self.assertEqual(action.anchor_point, anchor_point)  # type: ignore[union-attr]

    def test_floating_tooltip_gizmo_commits_anchor_but_not_rotation(self) -> None:
        tour = _tour()
        tooltip = TourFloatingTooltipData(
            component_id="movable-tooltip",
            anchor_point=(1.0, 2.0, 3.0),
        )
        tour = replace(
            tour,
            steps=(replace(tour.steps[0], actions=(tooltip,)), tour.steps[1]),
        )
        tour_workspace = self.workspace.tour_workspace
        tour_workspace.set_tours((tour,))
        moved_anchor = (6.0, 7.0, 8.0)

        self.workspace.viewer.tour_edit_finished.emit(
            tour.tour_id,
            TOUR_EDIT_TARGET_ACTION,
            tooltip.component_id,
            moved_anchor,
            True,
        )

        action = tour_workspace.selected_action()
        self.assertIsInstance(action, TourFloatingTooltipData)
        self.assertEqual(action.anchor_point, moved_anchor)  # type: ignore[union-attr]

        self.workspace.viewer.tour_edit_finished.emit(
            tour.tour_id,
            TOUR_EDIT_TARGET_ACTION_ROTATION,
            tooltip.component_id,
            (10.0, 20.0, 30.0),
            True,
        )

        unchanged_action = tour_workspace.selected_action()
        self.assertIsInstance(unchanged_action, TourFloatingTooltipData)
        self.assertEqual(  # type: ignore[union-attr]
            unchanged_action.anchor_point,
            moved_anchor,
        )

    def test_action_visual_properties_update_both_views_immediately(self) -> None:
        tour = _tour()
        action = TourText3DData(
            component_id="live-action-properties",
            text="Before",
            position=(1.0, 2.0, 3.0),
        )
        tour = replace(
            tour,
            steps=(replace(tour.steps[0], actions=(action,)), tour.steps[1]),
        )
        tour_workspace = self.workspace.tour_workspace
        tour_workspace.set_tours((tour,))
        self.workspace.workspace_tabs.setCurrentWidget(tour_workspace)
        _process_events()
        preview_viewer = self.workspace.tour_preview_viewer
        assert preview_viewer is not None
        initial_widths = tuple(
            float(
                np.ptp(
                    viewer._tour_text_overlay_items[0]
                    .opts["meshdata"]
                    .vertexes()[:, 0]
                )
            )
            for viewer in (self.workspace.viewer, preview_viewer)
        )

        tour_workspace.action_text_edit.setText("Updated live")
        tour_workspace.action_color_edit.setText("#123456")
        tour_workspace.action_size_spin.setValue(24.0)
        tour_workspace.action_position_spins[0].setValue(7.0)

        expected_color = (0x12 / 255.0, 0x34 / 255.0, 0x56 / 255.0)
        for viewer, initial_width in zip(
            (self.workspace.viewer, preview_viewer),
            initial_widths,
            strict=True,
        ):
            self.assertEqual(len(viewer._tour_text_overlay_items), 1)
            text_item = viewer._tour_text_overlay_items[0]
            self.assertEqual(text_item.text, "Updated live")
            np.testing.assert_allclose(text_item.opts["color"][:3], expected_color)
            self.assertGreater(
                float(np.ptp(text_item.opts["meshdata"].vertexes()[:, 0])),
                initial_width,
            )
            np.testing.assert_allclose(
                viewer._get_tour_edit_target_position(viewer._tour_edit_target),
                (7.0, 2.0, 3.0),
            )

    def test_cross_tour_trigger_pick_keeps_its_gizmo_and_commits_move(self) -> None:
        first = _tour(tour_id="first-tour")
        second_base = _tour(tour_id="second-tour")
        action = TourText3DData(
            component_id="second-action",
            text="Second",
            position=(2.0, 2.0, 2.0),
        )
        second = replace(
            second_base,
            trigger_point=(7.0, 8.0, 0.0),
            steps=(
                replace(second_base.steps[0], actions=(action,)),
                second_base.steps[1],
            ),
        )
        tour_workspace = self.workspace.tour_workspace
        tour_workspace.set_tours((first, second))
        self.workspace._sync_tour_overlays()

        self.workspace.viewer._apply_tour_overlay_pick(
            SimpleNamespace(
                tour_id=second.tour_id,
                kind=TOUR_EDIT_TARGET_TRIGGER,
                reference=None,
                step_id=None,
            )
        )

        self.assertIs(tour_workspace.current_tour(), second)
        self.assertIsNone(tour_workspace.selected_action())
        self.assertEqual(
            self.workspace.viewer._tour_edit_target.kind,
            TOUR_EDIT_TARGET_TRIGGER,
        )
        self.assertEqual(len(self.workspace.viewer._tour_gizmo_items), 6)

        moved_point = (9.0, 10.0, 1.0)
        self.workspace.viewer.tour_edit_finished.emit(
            second.tour_id,
            TOUR_EDIT_TARGET_TRIGGER,
            None,
            moved_point,
            True,
        )

        self.assertEqual(tour_workspace.current_tour().trigger_point, moved_point)  # type: ignore[union-attr]
        self.assertEqual(
            self.workspace.viewer._tour_edit_target.kind,
            TOUR_EDIT_TARGET_TRIGGER,
        )

        self.workspace.viewer._apply_tour_overlay_pick(
            SimpleNamespace(
                tour_id=second.tour_id,
                kind=TOUR_EDIT_TARGET_ACTION,
                reference=action.component_id,
                step_id=second.steps[0].step_id,
            )
        )
        self.assertEqual(
            tour_workspace.selected_action().component_id,  # type: ignore[union-attr]
            action.component_id,
        )

    def test_curve_point_selection_does_not_move_the_preview_camera(self) -> None:
        tour = _tour()
        tour_workspace = self.workspace.tour_workspace
        tour_workspace.set_tours((tour,))
        self.workspace.workspace_tabs.setCurrentWidget(tour_workspace)
        _process_events()
        preview_viewer = self.workspace.tour_preview_viewer
        assert preview_viewer is not None

        with patch.object(preview_viewer, "set_tour_preview") as set_preview:
            self.workspace.viewer._apply_tour_overlay_pick(
                SimpleNamespace(
                    tour_id=tour.tour_id,
                    kind=TOUR_EDIT_TARGET_CURVE_POINT,
                    reference=0,
                    step_id=None,
                )
            )

        set_preview.assert_not_called()
        self.assertEqual(
            self.workspace.viewer._tour_edit_target.kind,
            TOUR_EDIT_TARGET_CURVE_POINT,
        )
        self.assertEqual(
            preview_viewer._tour_edit_target.kind,
            TOUR_EDIT_TARGET_CURVE_POINT,
        )

    def test_camera_target_pick_selects_step_and_commits_gizmo_move(self) -> None:
        tour = _tour()
        tour_workspace = self.workspace.tour_workspace
        tour_workspace.set_tours((tour,))
        self.workspace.workspace_tabs.setCurrentWidget(tour_workspace)
        _process_events()
        preview_viewer = self.workspace.tour_preview_viewer
        assert preview_viewer is not None
        target_step = tour.steps[1]

        with patch.object(preview_viewer, "set_tour_preview") as set_preview:
            self.workspace.viewer._apply_tour_overlay_pick(
                SimpleNamespace(
                    tour_id=tour.tour_id,
                    kind=TOUR_EDIT_TARGET_CAMERA_TARGET,
                    reference=target_step.step_id,
                    step_id=target_step.step_id,
                )
            )

        set_preview.assert_not_called()
        self.assertEqual(
            tour_workspace.selected_step().step_id,  # type: ignore[union-attr]
            target_step.step_id,
        )
        self.assertIsNone(tour_workspace.selected_action())

        moved_target = (12.0, 5.0, 2.5)
        self.workspace.viewer.tour_edit_finished.emit(
            tour.tour_id,
            TOUR_EDIT_TARGET_CAMERA_TARGET,
            target_step.step_id,
            moved_target,
            True,
        )

        self.assertEqual(
            tour_workspace.selected_step().camera_target,  # type: ignore[union-attr]
            moved_target,
        )
        np.testing.assert_allclose(
            self.workspace.viewer._get_tour_edit_target_position(
                self.workspace.viewer._tour_edit_target
            ),
            moved_target,
        )
        np.testing.assert_allclose(
            preview_viewer._get_tour_edit_target_position(
                preview_viewer._tour_edit_target
            ),
            moved_target,
        )

    def test_camera_target_drag_updates_preview_camera_live_from_both_views(
        self,
    ) -> None:
        tour = _tour()
        tour_workspace = self.workspace.tour_workspace
        tour_workspace.set_tours((tour,))
        self.workspace.workspace_tabs.setCurrentWidget(tour_workspace)
        _process_events()
        scene_viewer = self.workspace.viewer
        preview_viewer = self.workspace.tour_preview_viewer
        assert preview_viewer is not None
        step = tour.steps[0]
        self.assertTrue(
            scene_viewer.select_tour_edit_target(
                tour.tour_id,
                TOUR_EDIT_TARGET_CAMERA_TARGET,
                step.step_id,
                step.step_id,
                emit=True,
            )
        )

        for source_viewer, temporary_target in (
            (scene_viewer, (3.0, 4.0, 5.0)),
            (preview_viewer, (6.0, 7.0, 8.0)),
        ):
            with self.subTest(source=source_viewer.objectName()):
                with patch.object(preview_viewer, "set_tour_preview") as set_preview:
                    source_viewer.tour_edit_preview_changed.emit(
                        tour.tour_id,
                        TOUR_EDIT_TARGET_CAMERA_TARGET,
                        step.step_id,
                        temporary_target,
                    )

                set_preview.assert_called_once_with(
                    tour_workspace.preview_progress(),
                    temporary_target,
                )
                self.assertEqual(
                    tour_workspace.selected_step().camera_target,  # type: ignore[union-attr]
                    step.camera_target,
                )

        live_target = (4.0, 6.0, 3.0)
        self.assertTrue(
            scene_viewer.set_tour_edit_preview_position(
                tour.tour_id,
                TOUR_EDIT_TARGET_CAMERA_TARGET,
                step.step_id,
                live_target,
            )
        )
        scene_viewer.tour_edit_preview_changed.emit(
            tour.tour_id,
            TOUR_EDIT_TARGET_CAMERA_TARGET,
            step.step_id,
            live_target,
        )
        self.assertEqual(preview_viewer._tour_preview_target, live_target)
        camera_position = preview_viewer.view.cameraPosition()
        np.testing.assert_allclose(
            (camera_position.x(), camera_position.y(), camera_position.z()),
            tour.curve.points[0],
            atol=1e-6,
        )
        expected_direction = np.asarray(live_target) - np.asarray(
            tour.curve.points[0]
        )
        expected_direction /= np.linalg.norm(expected_direction)
        direction_item = next(
            item
            for item in scene_viewer._tour_preview_overlay_items
            if item._housemaker_tour_role == "timeline_direction"
        )
        np.testing.assert_allclose(
            direction_item._housemaker_tour_direction,
            expected_direction,
            atol=1e-6,
        )
        self.assertNotIn(
            "timeline_direction",
            {
                item._housemaker_tour_role
                for item in preview_viewer._tour_preview_overlay_items
            },
        )

        committed_target = (9.0, 10.0, 11.0)
        preview_viewer.tour_edit_finished.emit(
            tour.tour_id,
            TOUR_EDIT_TARGET_CAMERA_TARGET,
            step.step_id,
            committed_target,
            True,
        )

        self.assertEqual(
            tour_workspace.selected_step().camera_target,  # type: ignore[union-attr]
            committed_target,
        )

    def test_tour_preview_text_pick_selects_the_action_in_both_views(self) -> None:
        tour = _tour()
        action = TourText3DData(
            component_id="preview-pick-action",
            text="Pick me",
            position=(1.0, 2.0, 3.0),
        )
        tour = replace(
            tour,
            steps=(replace(tour.steps[0], actions=(action,)), tour.steps[1]),
        )
        tour_workspace = self.workspace.tour_workspace
        tour_workspace.set_tours((tour,))
        self.workspace.workspace_tabs.setCurrentWidget(tour_workspace)
        _process_events()
        preview_viewer = self.workspace.tour_preview_viewer
        assert preview_viewer is not None

        preview_viewer._apply_tour_overlay_pick(
            SimpleNamespace(
                tour_id=tour.tour_id,
                kind=TOUR_EDIT_TARGET_ACTION,
                reference=action.component_id,
                step_id=tour.steps[0].step_id,
            )
        )

        self.assertEqual(
            tour_workspace.selected_action().component_id,  # type: ignore[union-attr]
            action.component_id,
        )
        for viewer in (self.workspace.viewer, preview_viewer):
            self.assertEqual(viewer._tour_edit_target.kind, TOUR_EDIT_TARGET_ACTION)
            self.assertEqual(viewer._tour_edit_target.reference, action.component_id)

    def test_text_rotation_drag_previews_peer_and_commits_on_release(self) -> None:
        tour = _tour()
        action = TourText3DData(
            component_id="rotating-action",
            text="Rotate me",
            position=(1.0, 2.0, 3.0),
        )
        tour = replace(
            tour,
            steps=(replace(tour.steps[0], actions=(action,)), tour.steps[1]),
        )
        tour_workspace = self.workspace.tour_workspace
        tour_workspace.set_tours((tour,))
        self.workspace.workspace_tabs.setCurrentWidget(tour_workspace)
        _process_events()
        scene_viewer = self.workspace.viewer
        preview_viewer = self.workspace.tour_preview_viewer
        assert preview_viewer is not None
        self.assertTrue(
            preview_viewer.select_tour_edit_target(
                tour.tour_id,
                TOUR_EDIT_TARGET_ACTION,
                action.component_id,
                tour.steps[0].step_id,
                emit=True,
            )
        )
        rotation = (15.0, 30.0, 45.0)

        with patch.object(scene_viewer, "set_tour_edit_preview_rotation") as preview:
            preview_viewer.tour_edit_preview_changed.emit(
                tour.tour_id,
                TOUR_EDIT_TARGET_ACTION_ROTATION,
                action.component_id,
                rotation,
            )

        preview.assert_called_once_with(
            tour.tour_id,
            action.component_id,
            rotation,
        )
        self.assertEqual(
            tour_workspace.selected_action().rotation_degrees,  # type: ignore[union-attr]
            (0.0, 0.0, 0.0),
        )

        preview_viewer.tour_edit_finished.emit(
            tour.tour_id,
            TOUR_EDIT_TARGET_ACTION_ROTATION,
            action.component_id,
            rotation,
            True,
        )

        self.assertEqual(
            tour_workspace.selected_action().rotation_degrees,  # type: ignore[union-attr]
            rotation,
        )

    def test_shared_scene_gizmo_previews_every_tour_point_in_peer_view(self) -> None:
        tour = _tour()
        action = TourText3DData(
            component_id="live-preview-action",
            text="Live preview",
            position=(1.0, 2.0, 3.0),
        )
        tour = replace(
            tour,
            steps=(replace(tour.steps[0], actions=(action,)), tour.steps[1]),
        )
        tour_workspace = self.workspace.tour_workspace
        tour_workspace.set_tours((tour,))
        self.workspace.workspace_tabs.setCurrentWidget(tour_workspace)
        _process_events()
        source_viewer = self.workspace.viewer
        peer_viewer = self.workspace.tour_preview_viewer
        assert peer_viewer is not None

        cases = (
            (
                TOUR_EDIT_TARGET_TRIGGER,
                None,
                None,
                (8.0, 9.0, 1.0),
            ),
            (
                TOUR_EDIT_TARGET_CURVE_POINT,
                0,
                None,
                (4.0, 5.0, 6.0),
            ),
            (
                TOUR_EDIT_TARGET_CAMERA_TARGET,
                tour.steps[0].step_id,
                tour.steps[0].step_id,
                (3.0, 4.0, 5.0),
            ),
            (
                TOUR_EDIT_TARGET_ACTION,
                action.component_id,
                tour.steps[0].step_id,
                (6.0, 7.0, 8.0),
            ),
        )
        for kind, reference, step_id, preview_position in cases:
            with self.subTest(kind=kind):
                self.assertTrue(
                    source_viewer.select_tour_edit_target(
                        tour.tour_id,
                        kind,
                        reference,
                        step_id,
                        emit=True,
                    )
                )
                override_key = (tour.tour_id, kind, reference)
                source_viewer._tour_position_overrides[override_key] = preview_position
                source_viewer.tour_edit_preview_changed.emit(
                    tour.tour_id,
                    kind,
                    reference,
                    preview_position,
                )

                self.assertEqual(
                    peer_viewer._tour_position_overrides[override_key],
                    preview_position,
                )
                np.testing.assert_allclose(
                    peer_viewer._get_tour_edit_target_position(
                        peer_viewer._tour_edit_target
                    ),
                    preview_position,
                )
                authoritative_tour = tour_workspace.current_tour()
                assert authoritative_tour is not None
                if kind == TOUR_EDIT_TARGET_TRIGGER:
                    self.assertEqual(
                        authoritative_tour.trigger_point, tour.trigger_point
                    )
                elif kind == TOUR_EDIT_TARGET_CURVE_POINT:
                    self.assertEqual(
                        authoritative_tour.curve.points[0],
                        tour.curve.points[0],
                    )
                elif kind == TOUR_EDIT_TARGET_CAMERA_TARGET:
                    self.assertEqual(
                        tour_workspace.selected_step().camera_target,  # type: ignore[union-attr]
                        tour.steps[0].camera_target,
                    )
                else:
                    self.assertEqual(
                        tour_workspace.selected_action().position,  # type: ignore[union-attr]
                        action.position,
                    )

                source_viewer.tour_edit_finished.emit(
                    tour.tour_id,
                    kind,
                    reference,
                    preview_position,
                    False,
                )
                self.assertFalse(source_viewer._tour_position_overrides)
                self.assertFalse(peer_viewer._tour_position_overrides)

    def test_tour_preview_gizmo_updates_scene_and_commits_on_release(self) -> None:
        tour = _tour()
        action = TourText3DData(
            component_id="preview-view-action",
            text="Preview view",
            position=(1.0, 2.0, 3.0),
        )
        tour = replace(
            tour,
            steps=(replace(tour.steps[0], actions=(action,)), tour.steps[1]),
        )
        tour_workspace = self.workspace.tour_workspace
        tour_workspace.set_tours((tour,))
        self.workspace.workspace_tabs.setCurrentWidget(tour_workspace)
        _process_events()
        source_viewer = self.workspace.tour_preview_viewer
        scene_viewer = self.workspace.viewer
        assert source_viewer is not None
        self.assertTrue(
            source_viewer.select_tour_edit_target(
                tour.tour_id,
                TOUR_EDIT_TARGET_ACTION,
                action.component_id,
                tour.steps[0].step_id,
                emit=True,
            )
        )

        moved_position = (9.0, 8.0, 7.0)
        override_key = (
            tour.tour_id,
            TOUR_EDIT_TARGET_ACTION,
            action.component_id,
        )
        source_viewer._tour_position_overrides[override_key] = moved_position
        source_viewer.tour_edit_preview_changed.emit(
            tour.tour_id,
            TOUR_EDIT_TARGET_ACTION,
            action.component_id,
            moved_position,
        )

        self.assertEqual(
            scene_viewer._tour_position_overrides[override_key],
            moved_position,
        )
        self.assertEqual(
            tour_workspace.selected_action().position,  # type: ignore[union-attr]
            action.position,
        )

        source_viewer.tour_edit_finished.emit(
            tour.tour_id,
            TOUR_EDIT_TARGET_ACTION,
            action.component_id,
            moved_position,
            True,
        )

        self.assertEqual(
            tour_workspace.selected_action().position,  # type: ignore[union-attr]
            moved_position,
        )
        self.assertFalse(source_viewer._tour_position_overrides)
        self.assertFalse(scene_viewer._tour_position_overrides)

    def test_rejected_curve_point_move_restores_the_authoritative_point(self) -> None:
        tour = _tour()
        tour_workspace = self.workspace.tour_workspace
        tour_workspace.set_tours((tour,))
        self.workspace._sync_tour_overlays()
        viewer = self.workspace.viewer
        invalid_point = tour.curve.points[1]
        viewer._tour_position_overrides[
            (tour.tour_id, TOUR_EDIT_TARGET_CURVE_POINT, 0)
        ] = invalid_point

        viewer.tour_edit_finished.emit(
            tour.tour_id,
            TOUR_EDIT_TARGET_CURVE_POINT,
            0,
            invalid_point,
            True,
        )

        self.assertEqual(
            tour_workspace.current_tour().curve.points,  # type: ignore[union-attr]
            tour.curve.points,
        )
        self.assertFalse(viewer._tour_position_overrides)
        np.testing.assert_allclose(
            viewer._get_tour_curve_display_points(viewer._tour_overlays[tour.tour_id])[
                0
            ],
            tour.curve.points[0],
        )

    def test_deselecting_a_step_hides_its_scene_actions(self) -> None:
        tour = _tour()
        action = TourText3DData(
            component_id="visible-action",
            text="Visible",
            position=(1.0, 2.0, 3.0),
            fade_duration_ms=0.0,
        )
        tour = replace(
            tour,
            steps=(replace(tour.steps[0], actions=(action,)), tour.steps[1]),
        )
        tour_workspace = self.workspace.tour_workspace
        tour_workspace.set_tours((tour,))
        self.workspace._sync_tour_overlays()
        self.assertEqual(
            [item.text for item in self.workspace.viewer._tour_text_overlay_items],
            ["Visible"],
        )

        selected_step_item = tour_workspace.step_list.currentItem()
        assert selected_step_item is not None
        selected_step_item.setSelected(False)
        _process_events()

        self.assertIsNone(tour_workspace.selected_step())
        self.assertFalse(self.workspace.viewer._tour_text_overlay_items)

    def test_deleting_selected_tour_synchronously_updates_both_overlays(self) -> None:
        first = _tour(tour_id="first-tour")
        second = replace(
            _tour(tour_id="second-tour"),
            trigger_point=(7.0, 8.0, 0.0),
        )
        tour_workspace = self.workspace.tour_workspace
        tour_workspace.set_tours((first, second))
        self.workspace.workspace_tabs.setCurrentWidget(tour_workspace)
        _process_events()
        self.assertIsNotNone(self.workspace.tour_preview_viewer)

        tour_workspace.delete_tour_button.click()

        self.assertIs(tour_workspace.current_tour(), second)
        self.assertEqual(
            self.workspace.viewer._tour_trigger_point,
            second.trigger_point,
        )
        self.assertEqual(
            self.workspace.tour_preview_viewer._tour_trigger_point,
            second.trigger_point,
        )

        tour_workspace.delete_tour_button.click()

        self.assertIsNone(tour_workspace.current_tour())
        self.assertIsNone(self.workspace.viewer._tour_trigger_point)
        self.assertIsNone(self.workspace.tour_preview_viewer._tour_trigger_point)

    def test_export_passes_authored_tours_to_runtime_manifest(self) -> None:
        tour = _tour()
        self.workspace.tour_workspace.set_tours((tour,))
        export_path = Path(self._temporary_directory.name) / "tour-scene.glb"
        manifest_path = export_path.with_suffix(".json")
        generated_model = object()

        with (
            patch.object(
                self.workspace,
                "_sync_atlas_object_texture_sources",
            ),
            patch.object(
                self.workspace,
                "_show_unpacked_scene_texture_export_error",
                return_value=False,
            ),
            patch(
                "housemaker.main.QFileDialog.getSaveFileName",
                return_value=(str(export_path), "GLB Files (*.glb)"),
            ),
            patch.object(
                self.workspace,
                "_build_model_with_stable_dependencies",
                return_value=(generated_model, ("revision",)),
            ),
            patch.object(
                self.workspace,
                "_build_export_placed_models",
                return_value=((), ()),
            ),
            patch("housemaker.main.export_glb_file", return_value=export_path),
            patch(
                "housemaker.main.write_runtime_scene_manifest",
                return_value=manifest_path,
            ) as write_manifest,
            patch.object(self.workspace, "_ensure_viewer_preview_current"),
            patch("housemaker.main.QMessageBox.information"),
        ):
            self.workspace._handle_glb_export_clicked()

        write_manifest.assert_called_once_with(
            export_path,
            source_placements={},
            instance_placements=(),
            tours=(tour,),
        )


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
