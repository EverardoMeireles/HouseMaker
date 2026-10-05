# ### Environment setup ###
from __future__ import annotations

import math
import os
from dataclasses import replace
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import unittest

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QGroupBox, QPushButton, QWidget

from housemaker.tour_state import (
    DEFAULT_TEXT_COLOR,
    DEFAULT_TEXT_FADE_DELAY_MS,
    DEFAULT_TEXT_FADE_DURATION_MS,
    DEFAULT_TEXT_ROTATION_DEGREES,
    DEFAULT_TEXT_SIZE_POINTS,
    DEFAULT_TOOLTIP_HTML_BODY,
    DEFAULT_TOOLTIP_POSITION,
    DEFAULT_TOOLTIP_STYLE,
    TourFloatingTooltipData,
    TourIdleCameraAnimationData,
    TourSpeedOverrideData,
    TourText3DData,
    TourWaitForKeyPressData,
)
from housemaker.tour_workspace import (
    TOUR_POINT_ACTION_CURVE,
    TOUR_POINT_ACTION_TRIGGER,
    TourWorkspace,
    _playback_progress_after_elapsed,
    _tour_speed_segments,
)

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Tour workspace tests ###
class TourWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = TourWorkspace()

    def tearDown(self) -> None:
        self.workspace.close()
        self.workspace.deleteLater()
        _qt_application.processEvents()

    # ### Fixture helpers ###
    def _create_tour(self) -> None:
        self.workspace.start_new_tour()
        self.assertTrue(self.workspace.accept_world_point((4.0, 5.0, 1.0)))
        self.assertTrue(self.workspace.accept_world_point((0.0, 0.0, 1.5)))
        self.assertTrue(self.workspace.accept_world_point((5.0, 0.0, 1.5)))
        self.assertTrue(self.workspace.finish_curve())

    # ### Creation workflow ###
    def test_trigger_and_open_curve_workflow_creates_timeline(self) -> None:
        placement_actions: list[str] = []
        changed_payloads: list[object] = []
        self.workspace.point_placement_requested.connect(
            lambda action, _context: placement_actions.append(action)
        )
        self.workspace.data_changed.connect(changed_payloads.append)

        self.workspace.start_new_tour()
        self.assertEqual(placement_actions, [TOUR_POINT_ACTION_TRIGGER])
        self.assertTrue(
            self.workspace.accept_world_point(
                {
                    "center": (1.0, 2.0, 0.5),
                    "size": (4.0, 3.0),
                }
            )
        )
        self.assertEqual(placement_actions[-1], TOUR_POINT_ACTION_CURVE)
        self.assertTrue(self.workspace.accept_world_point((0.0, 0.0, 1.5)))
        self.assertFalse(self.workspace.accept_world_point((0.0, 0.0, 1.5)))
        self.assertTrue(self.workspace.accept_world_point((3.0, 0.0, 1.5)))

        self.workspace.duration_spin.setValue(16.5)
        self.workspace.progress_speed_spin.setValue(1.75)
        self.assertTrue(self.workspace.finish_curve())

        self.assertEqual(len(self.workspace.tours()), 1)
        tour = self.workspace.tours()[0]
        self.assertEqual(tour.trigger_point, (1.0, 2.0, 0.5))
        self.assertEqual(tour.trigger_area_size, (4.0, 3.0))
        self.assertEqual(tour.curve.points, ((0.0, 0.0, 1.5), (3.0, 0.0, 1.5)))
        self.assertFalse(tour.curve.closed)
        self.assertEqual([step.progress for step in tour.steps], [0.0, 1.0])
        self.assertEqual(tour.steps[0].camera_target, (1.0, 0.0, 1.5))
        self.assertEqual(tour.steps[-1].camera_target, (4.0, 0.0, 1.5))
        self.assertEqual(tour.duration_seconds, 16.5)
        self.assertEqual(tour.progress_speed, 1.75)
        self.assertTrue(changed_payloads)

    # ### Step and action UI ###
    def test_step_panel_uses_requested_action_sections_and_compact_list(self) -> None:
        group_titles = {
            group.title() for group in self.workspace.findChildren(QGroupBox)
        }
        button_titles = {
            button.text() for button in self.workspace.findChildren(QPushButton)
        }

        self.assertTrue(
            {
                "Tour properties",
                "Action list",
                "Actions",
                "Action properties",
            }.issubset(group_titles)
        )
        self.assertIn("Set camera target", button_titles)
        self.assertIn("Add 3D text", button_titles)
        self.assertIn("Add floating tooltip", button_titles)
        self.assertIn("Wait for key press", button_titles)
        self.assertIn("Add idle camera animation", button_titles)
        self.assertIn("Speed override", button_titles)
        self.assertNotIn("Place 3D text", button_titles)
        self.assertLessEqual(self.workspace.step_list.maximumHeight(), 120)
        self.assertFalse(self.workspace.action_properties_group.isEnabled())

    def test_progress_speed_and_speed_action_edit_their_own_scope(self) -> None:
        self._create_tour()

        self.workspace.progress_speed_spin.setValue(1.5)
        self.workspace.add_speed_override_button.click()
        self.workspace.speed_override_spin.setValue(2.25)
        self.workspace.speed_easing_combo.setCurrentIndex(
            self.workspace.speed_easing_combo.findData("ease_in_out_sine")
        )

        tour = self.workspace.current_tour()
        step = self.workspace.selected_step()
        assert tour is not None and step is not None
        self.assertEqual(tour.progress_speed, 1.5)
        self.assertEqual(step.speed_override, 2.25)
        speed_action = self.workspace.selected_action()
        self.assertIsInstance(speed_action, TourSpeedOverrideData)
        assert isinstance(speed_action, TourSpeedOverrideData)
        self.assertEqual(speed_action.easing, "ease_in_out_sine")
        self.assertFalse(self.workspace.add_speed_override_button.isEnabled())

        self.workspace.remove_action_button.click()
        step = self.workspace.selected_step()
        assert step is not None
        self.assertIsNone(step.speed_override)
        self.assertTrue(self.workspace.add_speed_override_button.isEnabled())

    def test_camera_target_requests_the_selected_preview_step_immediately(self) -> None:
        self._create_tour()
        requests: list[tuple[str, str]] = []
        self.workspace.camera_target_requested.connect(
            lambda tour_id, step_id: requests.append((tour_id, step_id))
        )
        selected = self.workspace.selected_step()
        tour = self.workspace.current_tour()
        assert selected is not None and tour is not None

        self.workspace.set_camera_target_button.click()

        self.assertEqual(requests, [(tour.tour_id, selected.step_id)])
        self.assertIsNone(self.workspace.pending_point_action())

    def test_add_text_requests_preview_center_then_applies_defaults(self) -> None:
        self._create_tour()
        requests: list[object] = []
        self.workspace.text_action_position_requested.connect(requests.append)
        selected = self.workspace.selected_step()
        tour = self.workspace.current_tour()
        assert selected is not None and tour is not None

        self.workspace.add_text_button.click()

        self.assertEqual(
            requests,
            [{"tour_id": tour.tour_id, "step_id": selected.step_id}],
        )
        self.assertEqual(self.workspace.selected_step().actions, ())  # type: ignore[union-attr]
        self.assertIsNone(self.workspace.pending_point_action())

        self.assertTrue(self.workspace.add_text_action_at_position((2.0, 3.0, 1.0)))
        action = self.workspace.selected_action()
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.text, "Lorem ipsum")
        self.assertEqual(action.position, (2.0, 3.0, 1.0))
        self.assertEqual(action.size_points, DEFAULT_TEXT_SIZE_POINTS)
        self.assertEqual(action.color, DEFAULT_TEXT_COLOR)
        self.assertEqual(action.fade_delay_ms, DEFAULT_TEXT_FADE_DELAY_MS)
        self.assertEqual(action.fade_duration_ms, DEFAULT_TEXT_FADE_DURATION_MS)
        self.assertEqual(action.rotation_degrees, DEFAULT_TEXT_ROTATION_DEGREES)
        self.assertEqual(self.workspace.action_list.count(), 1)
        self.assertIn("3D text", self.workspace.action_list.item(0).text())
        self.assertTrue(self.workspace.action_properties_group.isEnabled())

    def test_add_floating_tooltip_creates_selects_and_requests_anchor(self) -> None:
        self._create_tour()
        requests: list[tuple[str, str, str]] = []
        self.workspace.floating_tooltip_position_requested.connect(
            lambda tour_id, step_id, action_id: requests.append(
                (tour_id, step_id, action_id)
            )
        )
        tour = self.workspace.current_tour()
        step = self.workspace.selected_step()
        assert tour is not None and step is not None

        self.workspace.add_floating_tooltip_button.click()

        action = self.workspace.selected_action()
        self.assertIsInstance(action, TourFloatingTooltipData)
        assert isinstance(action, TourFloatingTooltipData)
        self.assertEqual(action.anchor_point, step.camera_target)
        self.assertEqual(action.tooltip_position, DEFAULT_TOOLTIP_POSITION)
        self.assertEqual(action.html_body, DEFAULT_TOOLTIP_HTML_BODY)
        self.assertEqual(action.style, DEFAULT_TOOLTIP_STYLE)
        self.assertEqual(
            requests,
            [(tour.tour_id, step.step_id, action.component_id)],
        )
        self.assertIs(
            self.workspace.action_properties_stack.currentWidget(),
            self.workspace.floating_tooltip_properties_page,
        )
        self.assertEqual(self.workspace.action_list.item(0).text(), "Floating tooltip")
        self.assertIsNone(self.workspace.pending_point_action())

    def test_floating_tooltip_properties_and_anchor_update_live(self) -> None:
        self._create_tour()
        self.workspace.add_floating_tooltip_button.click()
        action = self.workspace.selected_action()
        tour = self.workspace.current_tour()
        step = self.workspace.selected_step()
        assert isinstance(action, TourFloatingTooltipData)
        assert tour is not None and step is not None
        changed_payloads: list[object] = []
        preview_payloads: list[tuple[object, float]] = []
        self.workspace.data_changed.connect(changed_payloads.append)
        self.workspace.preview_requested.connect(
            lambda current_tour, progress: preview_payloads.append(
                (current_tour, progress)
            )
        )

        self.workspace.tooltip_position_combo.setCurrentIndex(
            self.workspace.tooltip_position_combo.findData("left")
        )
        self.workspace.tooltip_html_edit.setPlainText(
            '<section><img src="photo.png"><b>Details</b></section>'
        )
        self.workspace.tooltip_style_edit.setPlainText(
            "background: #111; color: white;"
        )

        updated = self.workspace.selected_action()
        assert isinstance(updated, TourFloatingTooltipData)
        self.assertEqual(updated.tooltip_position, "left")
        self.assertEqual(
            updated.html_body,
            '<section><img src="photo.png"><b>Details</b></section>',
        )
        self.assertEqual(updated.style, "background: #111; color: white;")
        self.assertTrue(changed_payloads)
        self.assertTrue(preview_payloads)

        self.assertTrue(
            self.workspace.set_floating_tooltip_action_anchor_point(
                tour.tour_id,
                step.step_id,
                action.component_id,
                (7.0, 8.0, 9.0),
            )
        )
        moved = self.workspace.selected_action()
        assert isinstance(moved, TourFloatingTooltipData)
        self.assertEqual(moved.anchor_point, (7.0, 8.0, 9.0))

    def test_wait_idle_and_speed_are_buttons_with_typed_property_pages(self) -> None:
        self._create_tour()

        self.assertEqual(self.workspace.wait_input_combo.itemText(0), "any")
        self.assertFalse(self.workspace.add_idle_camera_button.isEnabled())
        self.workspace.add_wait_button.click()

        wait_action = self.workspace.selected_action()
        self.assertIsInstance(wait_action, TourWaitForKeyPressData)
        assert isinstance(wait_action, TourWaitForKeyPressData)
        self.assertEqual(wait_action.easing, "smoothstep")
        self.assertEqual(self.workspace.wait_easing_combo.currentData(), "smoothstep")
        self.assertIs(
            self.workspace.action_properties_stack.currentWidget(),
            self.workspace.wait_action_properties_page,
        )
        self.assertTrue(self.workspace.add_idle_camera_button.isEnabled())
        self.workspace.add_idle_camera_button.click()

        idle_action = self.workspace.selected_action()
        self.assertIsInstance(idle_action, TourIdleCameraAnimationData)
        assert isinstance(idle_action, TourIdleCameraAnimationData)
        self.assertEqual(
            idle_action.pivot_point,
            self.workspace.selected_step().camera_target,  # type: ignore[union-attr]
        )
        self.assertIs(
            self.workspace.action_properties_stack.currentWidget(),
            self.workspace.idle_action_properties_page,
        )
        self.assertFalse(self.workspace.add_idle_camera_button.isEnabled())
        self.workspace.add_speed_override_button.click()

        speed_action = self.workspace.selected_action()
        self.assertIsInstance(speed_action, TourSpeedOverrideData)
        self.assertIs(
            self.workspace.action_properties_stack.currentWidget(),
            self.workspace.speed_action_properties_page,
        )
        self.assertEqual(self.workspace.action_list.count(), 3)
        labels = {
            self.workspace.action_list.item(index).text()
            for index in range(self.workspace.action_list.count())
        }
        self.assertTrue(any(label.startswith("Wait for key press") for label in labels))
        self.assertIn("Idle camera animation", labels)
        self.assertTrue(any(label.startswith("Speed override") for label in labels))

    def test_wait_easing_owns_resume_curve_when_speed_action_coexists(self) -> None:
        self._create_tour()
        self.workspace.add_speed_override_button.click()
        self.workspace.speed_override_spin.setValue(2.0)
        self.workspace.speed_easing_combo.setCurrentIndex(
            self.workspace.speed_easing_combo.findData("ease_out_quad")
        )
        speed_action = self.workspace.selected_action()
        assert isinstance(speed_action, TourSpeedOverrideData)

        self.workspace.add_wait_button.click()
        self.workspace.wait_easing_combo.setCurrentIndex(
            self.workspace.wait_easing_combo.findData("smootherstep")
        )
        wait_action = self.workspace.selected_action()
        assert isinstance(wait_action, TourWaitForKeyPressData)
        self.assertEqual(wait_action.easing, "smootherstep")
        self.assertTrue(self.workspace.wait_easing_combo.isEnabled())

        tour = self.workspace.current_tour()
        step = self.workspace.selected_step()
        assert tour is not None and step is not None
        self.assertTrue(
            self.workspace.select_action(
                tour.tour_id,
                step.step_id,
                speed_action.component_id,
            )
        )
        self.assertFalse(self.workspace.speed_easing_combo.isEnabled())
        self.assertEqual(
            self.workspace.speed_easing_combo.currentData(),
            "ease_out_quad",
        )
        segment = _tour_speed_segments(self.workspace.current_tour())[0]  # type: ignore[arg-type]
        self.assertEqual(segment.end_speed, 2.0)
        self.assertEqual(segment.easing, "none")
        self.assertEqual(segment.resume_easing, "smootherstep")

        self.assertTrue(
            self.workspace.select_action(
                tour.tour_id,
                step.step_id,
                wait_action.component_id,
            )
        )
        self.workspace.remove_action_button.click()
        remaining_step = self.workspace.selected_step()
        assert remaining_step is not None
        remaining_speed = next(
            action
            for action in remaining_step.actions
            if isinstance(action, TourSpeedOverrideData)
        )
        self.assertTrue(
            self.workspace.select_action(
                tour.tour_id,
                remaining_step.step_id,
                remaining_speed.component_id,
            )
        )
        self.assertTrue(self.workspace.speed_easing_combo.isEnabled())
        self.assertEqual(remaining_speed.easing, "ease_out_quad")

    def test_removing_wait_also_removes_its_dependent_idle_action(self) -> None:
        self._create_tour()
        self.workspace.add_wait_button.click()
        wait_action = self.workspace.selected_action()
        assert isinstance(wait_action, TourWaitForKeyPressData)
        self.workspace.add_idle_camera_button.click()
        step = self.workspace.selected_step()
        assert step is not None
        self.assertEqual(len(step.actions), 2)

        self.assertTrue(
            self.workspace.select_action(
                self.workspace.current_tour().tour_id,  # type: ignore[union-attr]
                step.step_id,
                wait_action.component_id,
            )
        )
        self.workspace.remove_action_button.click()

        step = self.workspace.selected_step()
        assert step is not None
        self.assertEqual(step.actions, ())
        self.assertFalse(self.workspace.add_idle_camera_button.isEnabled())

    def test_detect_input_captures_the_next_application_key_press(self) -> None:
        self._create_tour()
        self.workspace.add_wait_button.click()
        self.workspace.show()
        _qt_application.processEvents()

        self.workspace.detect_input_button.click()
        self.assertEqual(
            self.workspace.detect_input_button.text(),
            "Press a key or click...",
        )
        QTest.keyClick(self.workspace, Qt.Key.Key_K)
        _qt_application.processEvents()

        wait_action = self.workspace.selected_action()
        self.assertIsInstance(wait_action, TourWaitForKeyPressData)
        assert isinstance(wait_action, TourWaitForKeyPressData)
        self.assertEqual(wait_action.input_token, "key:k")
        self.assertEqual(self.workspace.wait_input_combo.currentData(), "key:k")
        self.assertEqual(self.workspace.detect_input_button.text(), "Detect input")

    def test_text_action_properties_edit_the_selected_action(self) -> None:
        self._create_tour()
        self.assertTrue(self.workspace.add_text_action_at_position((2.0, 3.0, 1.0)))

        self.workspace.action_text_edit.setText("Welcome")
        self.workspace.action_size_spin.setValue(24.0)
        self.workspace.action_color_edit.setText("#12ABef")
        for spin, value in zip(
            self.workspace.action_position_spins,
            (4.0, 5.0, 6.0),
            strict=True,
        ):
            spin.setValue(value)
        self.workspace.action_fade_delay_spin.setValue(350.0)
        self.workspace.action_fade_duration_spin.setValue(750.0)
        for spin, value in zip(
            self.workspace.action_rotation_spins,
            (15.0, -25.0, 35.0),
            strict=True,
        ):
            spin.setValue(value)

        action = self.workspace.selected_action()
        assert action is not None
        self.assertEqual(action.text, "Welcome")
        self.assertEqual(action.size_points, 24.0)
        self.assertEqual(action.color, "#12abef")
        self.assertEqual(action.position, (4.0, 5.0, 6.0))
        self.assertEqual(action.fade_delay_ms, 350.0)
        self.assertEqual(action.fade_duration_ms, 750.0)
        self.assertEqual(action.rotation_degrees, (15.0, -25.0, 35.0))

    def test_every_visual_action_property_publishes_live_updates(self) -> None:
        self._create_tour()
        self.assertTrue(self.workspace.add_text_action_at_position((2.0, 3.0, 1.0)))
        changed_payloads: list[object] = []
        preview_payloads: list[tuple[object, float]] = []
        self.workspace.data_changed.connect(changed_payloads.append)
        self.workspace.preview_requested.connect(
            lambda tour, progress: preview_payloads.append((tour, progress))
        )

        edits = (
            lambda: self.workspace.action_text_edit.setText("Live text"),
            lambda: self.workspace.action_size_spin.setValue(25.0),
            lambda: self.workspace.action_color_edit.setText("#123456"),
            lambda: self.workspace.action_position_spins[0].setValue(7.0),
            lambda: self.workspace.action_rotation_spins[1].setValue(45.0),
            lambda: self.workspace.action_fade_delay_spin.setValue(300.0),
            lambda: self.workspace.action_fade_duration_spin.setValue(800.0),
        )
        for edit in edits:
            previous_change_count = len(changed_payloads)
            previous_preview_count = len(preview_payloads)
            edit()
            self.assertGreater(len(changed_payloads), previous_change_count)
            self.assertGreater(len(preview_payloads), previous_preview_count)

        action = self.workspace.selected_action()
        assert action is not None
        self.assertEqual(action.text, "Live text")
        self.assertEqual(action.size_points, 25.0)
        self.assertEqual(action.color, "#123456")
        self.assertEqual(action.position, (7.0, 3.0, 1.0))
        self.assertEqual(action.rotation_degrees, (0.0, 45.0, 0.0))
        self.assertEqual(action.fade_delay_ms, 300.0)
        self.assertEqual(action.fade_duration_ms, 800.0)

    def test_incomplete_color_input_does_not_replace_the_live_color(self) -> None:
        self._create_tour()
        self.assertTrue(self.workspace.add_text_action_at_position((2.0, 3.0, 1.0)))
        self.workspace.action_color_edit.setText("#123456")

        changed_payloads: list[object] = []
        self.workspace.data_changed.connect(changed_payloads.append)
        self.workspace.action_color_edit.setText("#")

        action = self.workspace.selected_action()
        assert action is not None
        self.assertEqual(action.color, "#123456")
        self.assertEqual(self.workspace.action_color_edit.text(), "#")
        self.assertFalse(changed_payloads)

    def test_action_list_only_shows_the_current_steps_actions(self) -> None:
        self._create_tour()
        self.assertTrue(self.workspace.add_text_action_at_position((2.0, 3.0, 1.0)))
        self.assertEqual(self.workspace.action_list.count(), 1)

        self.workspace.step_list.setCurrentRow(1)

        self.assertEqual(self.workspace.action_list.count(), 0)
        self.assertIsNone(self.workspace.selected_action())
        self.assertFalse(self.workspace.action_properties_group.isEnabled())
        self.workspace.step_list.setCurrentRow(0)
        self.assertEqual(self.workspace.action_list.count(), 1)

    def test_timeline_crossing_a_step_refreshes_its_action_list(self) -> None:
        self._create_tour()
        tour = self.workspace.current_tour()
        assert tour is not None
        first_action = TourText3DData(
            component_id="first-action",
            text="First step",
            position=(1.0, 1.0, 1.0),
        )
        second_action = TourText3DData(
            component_id="second-action",
            text="Second step",
            position=(2.0, 2.0, 2.0),
        )
        first_step = replace(tour.steps[0], progress=0.2, actions=(first_action,))
        second_step = replace(tour.steps[1], progress=0.6, actions=(second_action,))
        updated_tour = replace(tour, steps=(first_step, second_step))
        self.workspace.set_tours((updated_tour,))
        self.workspace._preview_timer.start(60_000)

        self.workspace.set_preview_progress(0.6)

        self.assertEqual(self.workspace.selected_step(), second_step)
        self.assertEqual(self.workspace.action_list.count(), 1)
        self.assertEqual(
            self.workspace.action_list.item(0).data(Qt.ItemDataRole.UserRole),
            second_action.component_id,
        )
        self.assertAlmostEqual(self.workspace.preview_progress(), 0.6)
        self.assertTrue(self.workspace._preview_timer.isActive())

    def test_timeline_step_boundaries_are_clamped_and_stable(self) -> None:
        self._create_tour()
        tour = self.workspace.current_tour()
        assert tour is not None
        first_step = replace(tour.steps[0], progress=0.2)
        second_step = replace(tour.steps[1], progress=0.6)
        updated_tour = replace(tour, steps=(first_step, second_step))
        self.workspace.set_tours((updated_tour,))

        self.workspace.set_preview_progress(0.0)
        self.assertEqual(self.workspace.selected_step(), first_step)
        self.workspace.set_preview_progress(0.59)
        self.assertEqual(self.workspace.selected_step(), first_step)
        self.workspace.set_preview_progress(0.6)
        self.assertEqual(self.workspace.selected_step(), second_step)
        self.workspace.set_preview_progress(1.0)
        self.assertEqual(self.workspace.selected_step(), second_step)

    def test_dragging_to_painted_end_marker_selects_the_final_step(self) -> None:
        self._create_tour()
        tour = self.workspace.current_tour()
        assert tour is not None
        final_action = TourText3DData(
            component_id="final-action",
            text="Final step",
            position=(5.0, 1.0, 1.0),
        )
        final_step = replace(tour.steps[-1], actions=(final_action,))
        updated_tour = replace(tour, steps=(*tour.steps[:-1], final_step))
        self.workspace.set_tours((updated_tour,))
        self.workspace.resize(1_000, 500)
        self.workspace.show()
        _qt_application.processEvents()
        slider = self.workspace.timeline_slider
        slider.setValue(slider.minimum())
        start = QPoint(slider._marker_x(0.0), slider.height() // 2)
        end = QPoint(slider._marker_x(1.0), slider.height() // 2)

        QTest.mousePress(
            slider,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            start,
        )
        QTest.mouseMove(slider, end)
        QTest.mouseRelease(
            slider,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            end,
        )
        _qt_application.processEvents()

        self.assertEqual(slider.value(), slider.maximum())
        self.assertEqual(self.workspace.selected_step(), final_step)
        self.assertEqual(self.workspace.action_list.count(), 1)
        self.assertEqual(
            self.workspace.action_list.item(0).data(Qt.ItemDataRole.UserRole),
            final_action.component_id,
        )

    def test_equal_progress_keeps_the_already_selected_step(self) -> None:
        self._create_tour()
        tour = self.workspace.current_tour()
        assert tour is not None
        first_step = replace(tour.steps[0], progress=0.5)
        second_step = replace(tour.steps[1], progress=0.5)
        updated_tour = replace(tour, steps=(first_step, second_step))
        self.workspace.set_tours((updated_tour,))
        self.assertTrue(
            self.workspace.select_step(updated_tour.tour_id, second_step.step_id)
        )

        self.workspace.set_preview_progress(0.51)
        self.workspace.set_preview_progress(0.5)

        self.assertEqual(self.workspace.selected_step(), second_step)
        self.assertAlmostEqual(self.workspace.preview_progress(), 0.5)

    def test_deselecting_a_step_clears_its_actions_and_properties(self) -> None:
        self._create_tour()
        self.assertTrue(self.workspace.add_text_action_at_position((2.0, 3.0, 1.0)))
        selected_step_item = self.workspace.step_list.currentItem()
        assert selected_step_item is not None

        selected_step_item.setSelected(False)
        _qt_application.processEvents()

        self.assertIsNone(self.workspace.selected_step())
        self.assertIsNone(self.workspace.selected_action())
        self.assertEqual(self.workspace.action_list.count(), 0)
        self.assertFalse(self.workspace.action_properties_group.isEnabled())

    # ### Viewer integration APIs ###
    def test_public_gizmo_mutators_preserve_ids_and_validate_inputs(self) -> None:
        self._create_tour()
        self.assertTrue(self.workspace.add_text_action_at_position((1.0, 1.0, 1.0)))
        original = self.workspace.current_tour()
        action = self.workspace.selected_action()
        step = self.workspace.selected_step()
        assert original is not None and action is not None and step is not None
        identity = (
            original.tour_id,
            tuple(item.step_id for item in original.steps),
            action.component_id,
        )

        self.assertTrue(
            self.workspace.set_tour_trigger_point(original.tour_id, (8.0, 9.0, 1.0))
        )
        self.assertTrue(
            self.workspace.set_tour_curve_point(
                original.tour_id,
                0,
                (-1.0, 0.0, 1.5),
            )
        )
        self.assertTrue(
            self.workspace.set_tour_camera_target(
                original.tour_id,
                step.step_id,
                (3.0, 4.0, 5.0),
            )
        )
        self.assertTrue(
            self.workspace.set_text_action_position(
                original.tour_id,
                step.step_id,
                action.component_id,
                (7.0, 6.0, 2.0),
            )
        )
        self.assertTrue(
            self.workspace.set_text_action_rotation(
                original.tour_id,
                step.step_id,
                action.component_id,
                (20.0, 30.0, 40.0),
            )
        )

        updated = self.workspace.current_tour()
        selected_action = self.workspace.selected_action()
        assert updated is not None and selected_action is not None
        self.assertEqual(updated.trigger_point, (8.0, 9.0, 1.0))
        self.assertEqual(updated.curve.points[0], (-1.0, 0.0, 1.5))
        updated_step = next(
            item for item in updated.steps if item.step_id == step.step_id
        )
        self.assertEqual(updated_step.camera_target, (3.0, 4.0, 5.0))
        self.assertEqual(selected_action.position, (7.0, 6.0, 2.0))
        self.assertEqual(selected_action.rotation_degrees, (20.0, 30.0, 40.0))
        self.assertEqual(
            (
                updated.tour_id,
                tuple(item.step_id for item in updated.steps),
                selected_action.component_id,
            ),
            identity,
        )
        self.assertFalse(
            self.workspace.set_tour_curve_point(
                original.tour_id,
                0,
                updated.curve.points[1],
            )
        )
        self.assertFalse(
            self.workspace.set_tour_curve_point(
                original.tour_id,
                99,
                (0.0, 0.0, 0.0),
            )
        )
        self.assertFalse(
            self.workspace.set_tour_trigger_point(
                original.tour_id,
                (math.nan, 0.0, 0.0),
            )
        )
        self.assertFalse(
            self.workspace.set_text_action_position(
                original.tour_id,
                step.step_id,
                "missing-action",
                (0.0, 0.0, 0.0),
            )
        )
        self.assertFalse(
            self.workspace.set_text_action_rotation(
                original.tour_id,
                step.step_id,
                action.component_id,
                (0.0, math.nan, 0.0),
            )
        )
        self.assertFalse(
            self.workspace.set_tour_camera_target(
                original.tour_id,
                "missing-step",
                (0.0, 0.0, 0.0),
            )
        )

    def test_tour_and_action_selection_can_be_synchronized_or_cleared(self) -> None:
        self._create_tour()
        self.assertTrue(self.workspace.add_text_action_at_position((1.0, 2.0, 3.0)))
        first = self.workspace.current_tour()
        first_action = self.workspace.selected_action()
        assert first is not None and first_action is not None
        second = replace(first, tour_id="second-tour", name="Second tour")
        self.workspace.set_tours((first, second))
        selections: list[object] = []
        self.workspace.action_selection_changed.connect(selections.append)

        self.assertTrue(
            self.workspace.select_action(
                second.tour_id,
                second.steps[0].step_id,
                second.steps[0].actions[0].component_id,
            )
        )
        self.assertIs(self.workspace.current_tour(), second)
        self.assertEqual(
            self.workspace.selected_action().component_id,  # type: ignore[union-attr]
            first_action.component_id,
        )
        self.assertIs(selections[-1], second.steps[0].actions[0])

        self.assertTrue(self.workspace.select_tour(second.tour_id, select_action=False))
        self.assertIsNone(self.workspace.selected_action())
        self.assertFalse(self.workspace.action_properties_group.isEnabled())
        self.assertIsNone(selections[-1])

        self.assertTrue(
            self.workspace.select_action(
                second.tour_id,
                second.steps[0].step_id,
                second.steps[0].actions[0].component_id,
            )
        )
        self.workspace.clear_action_selection()
        self.assertIsNone(self.workspace.selected_action())
        self.assertEqual(self.workspace.action_list.count(), 1)
        self.assertFalse(self.workspace.action_properties_group.isEnabled())
        self.assertIsNone(selections[-1])
        self.workspace.duration_spin.setValue(
            self.workspace.duration_spin.value() + 1.0
        )
        self.workspace.curve_tension_spin.setValue(0.75)
        self.assertIsNone(self.workspace.selected_action())
        self.assertTrue(
            self.workspace.set_tour_trigger_point(
                second.tour_id,
                (9.0, 8.0, 1.0),
            )
        )
        self.assertIsNone(self.workspace.selected_action())

        self.assertTrue(self.workspace.select_tour(None))
        self.assertIsNone(self.workspace.current_tour())
        self.assertIsNone(self.workspace.selected_step())
        self.assertIsNone(self.workspace.selected_action())
        self.assertEqual(self.workspace.step_list.count(), 0)
        self.assertEqual(self.workspace.action_list.count(), 0)
        self.assertIsNone(selections[-1])
        self.assertFalse(self.workspace.select_tour("missing-tour"))

    def test_camera_target_sphere_can_select_its_step_without_an_action(self) -> None:
        self._create_tour()
        self.assertTrue(self.workspace.add_text_action_at_position((1.0, 2.0, 3.0)))
        tour = self.workspace.current_tour()
        assert tour is not None
        target_step = tour.steps[-1]

        self.assertTrue(self.workspace.select_step(tour.tour_id, target_step.step_id))

        self.assertEqual(self.workspace.selected_step(), target_step)
        self.assertIsNone(self.workspace.selected_action())
        self.assertAlmostEqual(self.workspace.preview_progress(), target_step.progress)
        self.assertFalse(self.workspace.select_step(tour.tour_id, "missing-step"))

    def test_changing_tours_aligns_playhead_with_the_selected_step(self) -> None:
        self._create_tour()
        first = self.workspace.current_tour()
        assert first is not None
        second = replace(
            first,
            tour_id="second-tour",
            name="Second tour",
            steps=(
                replace(first.steps[0], step_id="second-start", progress=0.25),
                replace(first.steps[1], step_id="second-end"),
            ),
        )
        self.workspace.set_tours((first, second))
        self.workspace.set_preview_progress(0.8)

        self.assertTrue(self.workspace.select_tour(second.tour_id))

        self.assertEqual(self.workspace.selected_step().step_id, "second-start")  # type: ignore[union-attr]
        self.assertAlmostEqual(self.workspace.preview_progress(), 0.25)

    # ### Preview lifecycle ###
    def test_playback_stops_when_its_tour_context_is_replaced_or_hidden(self) -> None:
        self._create_tour()
        tours = self.workspace.tours()

        self.workspace.play_preview_button.click()
        self.assertTrue(self.workspace._preview_timer.isActive())
        self.workspace.set_tours(tours)
        self.assertFalse(self.workspace._preview_timer.isActive())

        self.workspace.play_preview_button.click()
        self.assertTrue(self.workspace._preview_timer.isActive())
        self.workspace.start_new_tour()
        self.assertFalse(self.workspace._preview_timer.isActive())
        self.workspace.cancel_point_placement(remove_draft=True)

        self.workspace.show()
        _qt_application.processEvents()
        self.workspace.play_preview_button.click()
        self.assertTrue(self.workspace._preview_timer.isActive())
        self.workspace.hide()
        _qt_application.processEvents()
        self.assertFalse(self.workspace._preview_timer.isActive())

    def test_deleting_a_tour_stops_playback(self) -> None:
        self._create_tour()
        self.workspace.play_preview_button.click()
        self.assertTrue(self.workspace._preview_timer.isActive())

        self.workspace.delete_tour_button.click()

        self.assertFalse(self.workspace._preview_timer.isActive())
        self.assertEqual(self.workspace.tours(), ())

    def test_playback_progress_uses_elapsed_time_and_tour_duration(self) -> None:
        self._create_tour()
        elapsed_timer = Mock()
        elapsed_timer.isValid.return_value = True
        elapsed_timer.elapsed.return_value = 5_000
        self.workspace._preview_elapsed_timer = elapsed_timer

        self.workspace.play_preview_button.click()
        self.workspace._advance_preview()

        self.assertAlmostEqual(self.workspace.preview_progress(), 0.5, places=4)

    def test_playback_integrates_step_speed_across_its_boundary(self) -> None:
        self._create_tour()
        tour = self.workspace.current_tour()
        assert tour is not None
        middle_step = replace(
            tour.steps[-1],
            step_id="middle-step",
            progress=0.5,
            actions=(
                TourSpeedOverrideData(
                    component_id="middle-speed",
                    speed=0.5,
                    easing="none",
                ),
            ),
        )
        updated_tour = replace(
            tour,
            steps=(tour.steps[0], middle_step, tour.steps[-1]),
        )
        self.workspace.set_tours((updated_tour,))
        elapsed_timer = Mock()
        elapsed_timer.isValid.return_value = True
        elapsed_timer.elapsed.return_value = 7_500
        self.workspace._preview_elapsed_timer = elapsed_timer

        self.workspace.play_preview_button.click()
        self.workspace._advance_preview()

        self.assertAlmostEqual(self.workspace.preview_progress(), 0.625, places=4)
        self.assertEqual(self.workspace.selected_step(), middle_step)

    def test_playback_speed_resets_when_next_step_has_no_override(self) -> None:
        self._create_tour()
        tour = self.workspace.current_tour()
        assert tour is not None
        first = replace(
            tour.steps[0],
            actions=(
                TourSpeedOverrideData(
                    component_id="first-speed",
                    speed=2.0,
                    easing="none",
                ),
            ),
        )
        middle = replace(
            tour.steps[-1],
            step_id="middle-step",
            progress=0.5,
            actions=(),
        )
        updated_tour = replace(tour, steps=(first, middle, tour.steps[-1]))

        progress = _playback_progress_after_elapsed(updated_tour, 0.0, 3.0)

        self.assertAlmostEqual(progress, 0.55)

    def test_speed_segments_use_last_equal_position_step(self) -> None:
        self._create_tour()
        tour = self.workspace.current_tour()
        assert tour is not None
        first = replace(
            tour.steps[0],
            actions=(
                TourSpeedOverrideData(
                    component_id="first-speed",
                    speed=2.0,
                    easing="none",
                ),
            ),
        )
        replacement = replace(
            first,
            step_id="same-position-step",
            actions=(
                TourSpeedOverrideData(
                    component_id="replacement-speed",
                    speed=0.5,
                    easing="none",
                ),
            ),
        )
        updated_tour = replace(
            tour,
            steps=(first, replacement, tour.steps[-1]),
        )

        segments = _tour_speed_segments(updated_tour)

        self.assertEqual(segments[0].start_progress, 0.0)
        self.assertEqual(segments[0].end_progress, 1.0)
        self.assertEqual(segments[0].end_speed, 0.5)
        self.assertEqual(segments[0].easing, "none")

    def test_wait_at_zero_pauses_until_the_matching_input_without_time_jump(
        self,
    ) -> None:
        self._create_tour()
        self.workspace.add_wait_button.click()
        self.workspace.wait_input_combo.setCurrentIndex(
            self.workspace.wait_input_combo.findData("key:space")
        )
        elapsed_timer = Mock()
        elapsed_timer.isValid.return_value = True
        elapsed_timer.elapsed.return_value = 5_000
        self.workspace._preview_elapsed_timer = elapsed_timer

        self.workspace.play_preview_button.click()
        self.workspace._advance_preview()

        self.assertAlmostEqual(self.workspace.preview_progress(), 0.0)
        self.assertTrue(self.workspace._preview_wait_action_id)
        self.assertTrue(self.workspace.play_preview_button.text().startswith("Waiting"))
        QTest.keyClick(self.workspace, Qt.Key.Key_A)
        self.assertTrue(self.workspace._preview_wait_action_id)
        QTest.keyClick(self.workspace, Qt.Key.Key_Space)
        self.assertFalse(self.workspace._preview_wait_action_id)
        elapsed_timer.restart.assert_called_once()

        elapsed_timer.elapsed.return_value = 1_000
        self.workspace._advance_preview()
        self.assertAlmostEqual(self.workspace.preview_progress(), 0.028, places=4)

    def test_idle_camera_emits_explicit_60_hz_pose_while_waiting(self) -> None:
        self._create_tour()
        self.workspace.add_wait_button.click()
        self.workspace.add_idle_camera_button.click()
        idle_action = self.workspace.selected_action()
        assert isinstance(idle_action, TourIdleCameraAnimationData)
        playback_timer = Mock()
        playback_timer.isValid.return_value = True
        playback_timer.elapsed.return_value = 2_000
        wait_timer = Mock()
        wait_timer.isValid.return_value = True
        wait_timer.elapsed.return_value = 1_500
        self.workspace._preview_elapsed_timer = playback_timer
        self.workspace._preview_wait_elapsed_timer = wait_timer
        poses: list[tuple[object, float, object, object]] = []
        self.workspace.preview_pose_requested.connect(
            lambda tour, progress, position, target: poses.append(
                (tour, progress, position, target)
            )
        )

        self.workspace.play_preview_button.click()
        self.workspace._advance_preview()

        self.assertTrue(self.workspace._preview_timer.isActive())
        self.assertEqual(len(poses), 1)
        tour, progress, position, target = poses[0]
        self.assertIs(tour, self.workspace.current_tour())
        self.assertEqual(progress, 0.0)
        self.assertEqual(target, idle_action.pivot_point)
        self.assertNotEqual(
            position,
            self.workspace.current_tour().curve.points[0],  # type: ignore[union-attr]
        )

    def test_speed_easing_transitions_instead_of_jumping(self) -> None:
        self._create_tour()
        tour = self.workspace.current_tour()
        assert tour is not None
        smooth_step = replace(
            tour.steps[0],
            actions=(
                TourSpeedOverrideData(
                    component_id="smooth-speed",
                    speed=2.0,
                    easing="smoothstep",
                ),
            ),
        )
        immediate_step = replace(
            tour.steps[0],
            actions=(
                TourSpeedOverrideData(
                    component_id="immediate-speed",
                    speed=2.0,
                    easing="none",
                ),
            ),
        )
        smooth_tour = replace(tour, steps=(smooth_step, tour.steps[-1]))
        immediate_tour = replace(tour, steps=(immediate_step, tour.steps[-1]))

        smooth_progress = _playback_progress_after_elapsed(smooth_tour, 0.0, 2.5)
        immediate_progress = _playback_progress_after_elapsed(
            immediate_tour,
            0.0,
            2.5,
        )

        self.assertGreater(smooth_progress, 0.25)
        self.assertLess(smooth_progress, immediate_progress)
        self.assertAlmostEqual(immediate_progress, 0.5, places=4)

    def test_wait_easing_maps_resume_time_without_composing_speed_easing(
        self,
    ) -> None:
        self._create_tour()
        tour = self.workspace.current_tour()
        assert tour is not None
        first_step = replace(
            tour.steps[0],
            actions=(
                TourSpeedOverrideData(
                    component_id="conflicting-speed-easing",
                    speed=2.0,
                    easing="ease_out_quad",
                ),
                TourWaitForKeyPressData(
                    component_id="authoritative-wait-easing",
                    easing="smoothstep",
                ),
            ),
        )
        eased_tour = replace(tour, steps=(first_step, tour.steps[-1]))

        quarter_time_progress = _playback_progress_after_elapsed(
            eased_tour,
            0.0,
            1.25,
        )
        halfway_time_progress = _playback_progress_after_elapsed(
            eased_tour,
            0.0,
            2.5,
        )

        self.assertAlmostEqual(quarter_time_progress, 0.15625, places=4)
        self.assertAlmostEqual(halfway_time_progress, 0.5, places=4)
        segment = _tour_speed_segments(eased_tour)[0]
        self.assertEqual(segment.resume_easing, "smoothstep")
        self.assertEqual(segment.easing, "none")

    def test_playback_uses_precise_drift_resistant_60_hz_deadlines(self) -> None:
        self._create_tour()
        elapsed_timer = Mock()
        elapsed_timer.isValid.return_value = True
        elapsed_timer.elapsed.return_value = 0
        self.workspace._preview_elapsed_timer = elapsed_timer

        self.workspace.play_preview_button.click()

        self.assertTrue(self.workspace._preview_timer.isSingleShot())
        self.assertEqual(
            self.workspace._preview_timer.timerType(),
            Qt.TimerType.PreciseTimer,
        )
        self.assertIn(self.workspace._preview_timer.interval(), {16, 17})
        elapsed_timer.elapsed.return_value = 17
        self.workspace._advance_preview()
        self.assertIn(self.workspace._preview_timer.interval(), {16, 17})

    def test_preview_host_can_attach_and_release_a_viewer(self) -> None:
        preview = QWidget()
        self.workspace.set_preview_widget(preview)

        self.assertIs(self.workspace.clear_preview_widget(), preview)
        self.assertIsNone(preview.parent())


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
