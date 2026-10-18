# ### Environment setup ###
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QFormLayout,
    QGroupBox,
    QLabel,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QWidget,
)

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.main import BlueprintWorkspace

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Widget helpers ###
def _nearest_group_box(widget: QWidget) -> QGroupBox | None:
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QGroupBox):
            return parent
        parent = parent.parentWidget()
    return None


# ### Canvas control grouping tests ###
class CanvasControlGroupingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        settings_path = Path(self.temporary_directory.name) / "settings.json"
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(settings_path)
        )
        self.workspace.resize(1600, 900)
        self.workspace.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        self.workspace.deleteLater()
        self.temporary_directory.cleanup()
        _qt_application.processEvents()

    def _group_named(self, title: str) -> QGroupBox:
        matching_groups = [
            group
            for panel in (self.workspace.left_panel, self.workspace.side_panel)
            for group in panel.findChildren(QGroupBox)
            if group.title() == title
        ]
        self.assertEqual(
            len(matching_groups),
            1,
            f"Expected exactly one Canvas group titled {title!r}",
        )
        return matching_groups[0]

    def test_canvas_controls_are_split_between_left_and_right_panels(self) -> None:
        workspace = self.workspace
        self.assertFalse(hasattr(workspace, "side_tabs"))
        self.assertEqual(workspace.left_panel.findChildren(QTabWidget), [])
        self.assertEqual(workspace.side_panel.findChildren(QTabWidget), [])
        self.assertEqual(workspace.workspace_splitter.count(), 3)
        self.assertIs(workspace.workspace_splitter.widget(0), workspace.left_panel)
        self.assertIs(workspace.workspace_splitter.widget(1), workspace.workspace_tabs)
        self.assertIs(workspace.workspace_splitter.widget(2), workspace.side_panel)

        for group in (
            workspace.level_dimensions_group,
            workspace.merged_generation_workspace.ceiling_height_section,
            workspace.level_transform_group,
            workspace.canvas_transform_group,
            self._group_named("Outer wall marking"),
            workspace.levels_group,
        ):
            with self.subTest(right_group=group.title()):
                self.assertTrue(workspace.side_panel.isAncestorOf(group))
                self.assertTrue(workspace.canvas_right_scroll_area.isAncestorOf(group))
                self.assertFalse(workspace.left_panel.isAncestorOf(group))

        for control in (
            workspace.open_spaces_group,
            workspace.stairs_group,
            workspace.doorways_group,
            workspace.windows_group,
            workspace.doors_group,
            workspace.add_directional_light_button,
        ):
            with self.subTest(left_control=control.objectName()):
                self.assertTrue(workspace.left_panel.isAncestorOf(control))
                self.assertTrue(workspace.canvas_left_scroll_area.isAncestorOf(control))
                self.assertFalse(workspace.side_panel.isAncestorOf(control))

    def test_room_analysis_follows_dimensions_in_the_right_panel(self) -> None:
        workspace = self.workspace
        merged = workspace.merged_generation_workspace
        group = self._group_named("Room analysis")
        self.assertIs(group, merged.ceiling_height_section)
        self.assertTrue(group.isAncestorOf(merged.infer_ceiling_height_button))
        self.assertTrue(group.isAncestorOf(merged.ceiling_height_result_label))
        self.assertFalse(merged.isAncestorOf(group))
        right_layout = workspace.canvas_right_controls.layout()
        assert right_layout is not None
        self.assertEqual(
            right_layout.indexOf(group),
            right_layout.indexOf(workspace.level_dimensions_group) + 1,
        )
        self.assertEqual(
            right_layout.indexOf(workspace.transform_controls_scroll_area),
            right_layout.indexOf(group) + 1,
        )
        self.assertTrue(merged.infer_ceiling_height_button.isEnabled())

    def test_project_buttons_share_a_right_panel_row_below_levels(self) -> None:
        workspace = self.workspace
        controls_layout = workspace.canvas_right_controls.layout()
        assert controls_layout is not None
        levels_index = controls_layout.indexOf(workspace.levels_group)
        self.assertGreaterEqual(levels_index, 0)
        buttons_item = controls_layout.itemAt(levels_index + 1)
        assert buttons_item is not None
        buttons_layout = buttons_item.layout()
        assert buttons_layout is not None
        project_buttons = (
            workspace.save_button,
            workspace.load_button,
            workspace.export_button,
        )
        self.assertEqual(buttons_layout.count(), len(project_buttons))
        for index, button in enumerate(project_buttons):
            with self.subTest(button=button.text()):
                self.assertIs(buttons_layout.itemAt(index).widget(), button)
                self.assertTrue(workspace.side_panel.isAncestorOf(button))
                self.assertTrue(workspace.canvas_right_scroll_area.isAncestorOf(button))
                self.assertFalse(workspace.left_panel.isAncestorOf(button))

    def test_both_canvas_panels_hide_when_opening_another_workspace(self) -> None:
        workspace = self.workspace
        self.assertTrue(workspace.left_panel.isVisible())
        self.assertTrue(workspace.side_panel.isVisible())

        workspace.workspace_tabs.setCurrentIndex(workspace.settings_workspace_tab_index)
        _qt_application.processEvents()
        self.assertTrue(workspace.left_panel.isHidden())
        self.assertTrue(workspace.side_panel.isHidden())

        workspace.workspace_tabs.setCurrentIndex(workspace.canvas_workspace_tab_index)
        _qt_application.processEvents()
        self.assertTrue(workspace.left_panel.isVisible())
        self.assertTrue(workspace.side_panel.isVisible())

    def _assert_group_contains(
        self,
        title: str,
        *widgets: QWidget,
        labels: set[str] | None = None,
    ) -> None:
        group = self._group_named(title)
        for widget in widgets:
            self.assertIs(
                _nearest_group_box(widget),
                group,
                f"{widget!r} is not contained by the {title!r} group",
            )
        if labels:
            group_labels = {label.text() for label in group.findChildren(QLabel)}
            self.assertTrue(labels <= group_labels)

    def test_levels_group_contains_blueprint_and_include_controls(self) -> None:
        self._assert_group_contains(
            "Levels",
            self.workspace.load_image_button,
            self.workspace.erase_plan_image_button,
            self.workspace.generate_walls_button,
            self.workspace.image_correction_button,
            self.workspace.blueprint_name_label,
            self.workspace.levels_list,
            self.workspace.include_yes_radio,
            self.workspace.include_no_radio,
            labels={"Include"},
        )

    def test_automatic_wall_controls_share_a_dedicated_group(self) -> None:
        self._assert_group_contains(
            "Automatic wall placement",
            *self.workspace._get_plan_wall_generation_sliders(),
            self.workspace.plan_wall_generation_status_label,
        )
        levels_layout = self.workspace.levels_group.layout()
        self.assertIsNotNone(levels_layout)
        assert levels_layout is not None
        buttons_layout = levels_layout.itemAt(0).layout()
        self.assertIsNotNone(buttons_layout)
        assert buttons_layout is not None
        self.assertEqual(
            buttons_layout.indexOf(self.workspace.erase_plan_image_button),
            buttons_layout.indexOf(self.workspace.load_image_button) + 1,
        )
        self.assertEqual(
            buttons_layout.indexOf(self.workspace.generate_walls_button),
            buttons_layout.indexOf(self.workspace.erase_plan_image_button) + 1,
        )

    def test_doorways_group_contains_all_doorway_controls(self) -> None:
        self._assert_group_contains(
            "Doorways",
            self.workspace.selected_doorway_arch_checkbox,
            self.workspace.selected_doorway_arch_amount_spinbox,
            self.workspace.doorway_preset_list,
            self.workspace.save_doorway_template_button,
            self.workspace.remove_doorway_preset_button,
            self.workspace.place_doorway_button,
            labels={"Arch amount"},
        )

    def test_automatic_wall_controls_can_be_hidden_without_resetting_the_preview(
        self,
    ) -> None:
        toggle = self.workspace.toggle_plan_wall_controls_button
        group = self.workspace.plan_wall_controls_group
        sliders = self.workspace._get_plan_wall_generation_sliders()
        values_before = tuple(slider.value() for slider in sliders)
        self.assertTrue(toggle.isCheckable())
        self.assertFalse(toggle.isChecked())
        self.assertTrue(toggle.text().startswith("Show"))
        self.assertTrue(group.isHidden())
        self.assertFalse(group.isEnabled())
        self.assertFalse(group.isAncestorOf(toggle))
        self.assertTrue(toggle.isEnabled())

        toggle.click()
        _qt_application.processEvents()
        self.assertTrue(toggle.isChecked())
        self.assertTrue(toggle.text().startswith("Hide"))
        self.assertFalse(group.isHidden())
        self.assertFalse(group.isEnabled())
        self.assertEqual(tuple(slider.value() for slider in sliders), values_before)

        toggle.click()
        _qt_application.processEvents()
        self.assertFalse(toggle.isChecked())
        self.assertTrue(toggle.text().startswith("Show"))
        self.assertTrue(group.isHidden())
        self.assertFalse(group.isEnabled())
        self.assertEqual(tuple(slider.value() for slider in sliders), values_before)

        group.setEnabled(True)
        self.assertTrue(group.isHidden())
        toggle.click()
        _qt_application.processEvents()
        self.assertTrue(toggle.isChecked())
        self.assertTrue(toggle.text().startswith("Hide"))
        self.assertFalse(group.isHidden())
        self.assertTrue(group.isEnabled())
        self.assertEqual(tuple(slider.value() for slider in sliders), values_before)

    def test_level_list_has_room_for_eight_rows(self) -> None:
        levels = self.workspace.levels_list
        levels.blockSignals(True)
        try:
            levels.clear()
            for index in range(9):
                levels.addItem(QListWidgetItem(f"L{index + 1} Test level"))
            levels.setCurrentRow(0)
            levels.scrollToTop()
            levels.doItemsLayout()
            _qt_application.processEvents()

            viewport = levels.viewport().rect()
            for index in range(8):
                self.assertTrue(
                    viewport.contains(levels.visualItemRect(levels.item(index))),
                    f"Level row {index + 1} does not fit in the level list",
                )
            self.assertFalse(viewport.contains(levels.visualItemRect(levels.item(8))))
            self.assertGreater(levels.verticalScrollBar().maximum(), 0)
        finally:
            levels.blockSignals(False)

    def test_doorway_window_and_door_lists_are_half_the_previous_height(
        self,
    ) -> None:
        for list_widget, expected_height in (
            (self.workspace.doorway_preset_list, 56),
            (self.workspace.canvas_window_list, 48),
            (self.workspace.canvas_door_list, 48),
        ):
            with self.subTest(list_widget=list_widget.objectName()):
                self.assertEqual(list_widget.minimumHeight(), expected_height)
                self.assertEqual(list_widget.maximumHeight(), expected_height)
                self.assertEqual(list_widget.height(), expected_height)

    def test_level_and_canvas_transforms_share_aligned_side_by_side_rows(
        self,
    ) -> None:
        self.workspace.workspace_splitter.setSizes([380, 500, 700])
        _qt_application.processEvents()
        _qt_application.processEvents()
        row = self.workspace.transform_controls_row
        level = self.workspace.level_transform_group
        canvas = self.workspace.canvas_transform_group
        local_scroll = self.workspace.transform_controls_scroll_area
        self.assertEqual(local_scroll.horizontalScrollBar().maximum(), 0)
        self.assertEqual(local_scroll.verticalScrollBar().maximum(), 0)
        self.assertTrue(row.isAncestorOf(level))
        self.assertTrue(row.isAncestorOf(canvas))
        self.assertEqual(
            level.mapTo(row, QPoint(0, 0)).y(),
            canvas.mapTo(row, QPoint(0, 0)).y(),
        )
        self.assertLess(
            level.mapTo(row, QPoint(level.width() - 1, 0)).x(),
            canvas.mapTo(row, QPoint(0, 0)).x(),
        )
        self.assertLessEqual(abs(level.width() - canvas.width()), 2)

        for level_slider, canvas_slider in (
            (
                self.workspace.level_scale_slider,
                self.workspace.canvas_level_scale_slider,
            ),
            (
                self.workspace.level_x_offset_slider,
                self.workspace.canvas_x_offset_slider,
            ),
            (
                self.workspace.level_y_offset_slider,
                self.workspace.canvas_y_offset_slider,
            ),
        ):
            with self.subTest(level_slider=level_slider.objectName()):
                level_center = level_slider.mapTo(row, level_slider.rect().center())
                canvas_center = canvas_slider.mapTo(row, canvas_slider.rect().center())
                self.assertLessEqual(abs(level_center.y() - canvas_center.y()), 2)
                self.assertGreaterEqual(level_slider.width(), 24)
                self.assertGreaterEqual(canvas_slider.width(), 24)

        scroll_area = self.workspace.canvas_right_scroll_area
        self.assertIsInstance(scroll_area, QScrollArea)
        assert isinstance(scroll_area, QScrollArea)
        viewport = scroll_area.viewport()
        scroll_area.ensureWidgetVisible(row, 0, 0)
        _qt_application.processEvents()
        for group in (level, canvas):
            for widget in group.findChildren(QWidget):
                if widget.isHidden():
                    continue
                top_left = widget.mapTo(viewport, QPoint(0, 0))
                bottom_right = widget.mapTo(
                    viewport,
                    QPoint(widget.width() - 1, widget.height() - 1),
                )
                self.assertTrue(viewport.rect().contains(top_left))
                self.assertTrue(viewport.rect().contains(bottom_right))

    def test_narrow_transform_rows_scroll_locally_without_clipping_the_canvas(
        self,
    ) -> None:
        self.workspace.resize(760, 900)
        _qt_application.processEvents()
        _qt_application.processEvents()
        scroll_area = self.workspace.canvas_right_scroll_area
        self.assertIsInstance(scroll_area, QScrollArea)
        assert isinstance(scroll_area, QScrollArea)
        outer_viewport = scroll_area.viewport()
        local_scroll = self.workspace.transform_controls_scroll_area
        row = self.workspace.transform_controls_row
        level_group = self.workspace.level_transform_group
        canvas_group = self.workspace.canvas_transform_group

        self.assertEqual(local_scroll.verticalScrollBar().maximum(), 0)
        self.assertGreater(local_scroll.horizontalScrollBar().maximum(), 0)
        self.assertGreater(row.width(), local_scroll.viewport().width())
        self.assertEqual(
            level_group.mapTo(row, QPoint(0, 0)).y(),
            canvas_group.mapTo(row, QPoint(0, 0)).y(),
        )
        self.assertLess(
            level_group.mapTo(row, QPoint(level_group.width() - 1, 0)).x(),
            canvas_group.mapTo(row, QPoint(0, 0)).x(),
        )
        self.assertLessEqual(abs(level_group.width() - canvas_group.width()), 2)
        self.assertLessEqual(
            local_scroll.mapTo(outer_viewport, QPoint(0, 0)).x()
            + local_scroll.width(),
            outer_viewport.width(),
        )

        scroll_area.ensureWidgetVisible(local_scroll, 0, 0)
        local_scroll.horizontalScrollBar().setValue(
            local_scroll.horizontalScrollBar().maximum()
        )
        _qt_application.processEvents()
        local_viewport = local_scroll.viewport()
        for control in (
            self.workspace.canvas_level_scale_increase_button,
            self.workspace.canvas_x_offset_increase_button,
            self.workspace.canvas_y_offset_increase_button,
        ):
            with self.subTest(control=control.objectName()):
                top_left = control.mapTo(local_viewport, QPoint(0, 0))
                bottom_right = control.mapTo(
                    local_viewport,
                    QPoint(control.width() - 1, control.height() - 1),
                )
                self.assertTrue(local_viewport.rect().contains(top_left))
                self.assertTrue(local_viewport.rect().contains(bottom_right))
                self.assertEqual(control.visibleRegion().boundingRect(), control.rect())

        canvas = self.workspace.canvas
        self.assertEqual(canvas.visibleRegion().boundingRect(), canvas.rect())
        top_right = QPoint(canvas.width() - 4, 12)
        self.assertIs(QApplication.widgetAt(canvas.mapToGlobal(top_right)), canvas)

    def test_canvas_and_scene_expose_shared_window_frame_controls(self) -> None:
        self._assert_group_contains(
            "Window frames",
            self.workspace.add_window_button,
        )
        self.assertEqual(
            self.workspace.add_window_button.text(),
            "Add window frame",
        )
        self.assertTrue(self.workspace.add_window_button.isCheckable())
        scene_button = self.workspace.viewer.add_window_frame_button
        self.assertIsNotNone(scene_button)
        assert scene_button is not None
        self.assertEqual(scene_button.text(), "Add window frame")
        self.assertTrue(scene_button.isCheckable())
        self.assertIsNotNone(self.workspace.viewer.canvas_tools_panel)
        assert self.workspace.viewer.canvas_tools_panel is not None
        self.assertTrue(
            self.workspace.viewer.canvas_tools_panel.isAncestorOf(scene_button)
        )

        side_layout = self.workspace.windows_group.parentWidget().layout()
        self.assertIsNotNone(side_layout)
        assert side_layout is not None
        self.assertEqual(
            side_layout.indexOf(self.workspace.windows_group),
            side_layout.indexOf(self.workspace.doorways_group) + 1,
        )
        self.assertEqual(
            side_layout.indexOf(self.workspace.doors_group),
            side_layout.indexOf(self.workspace.windows_group) + 1,
        )

    def test_outer_wall_marking_is_on_the_right_with_edit_action(
        self,
    ) -> None:
        marking_group = self._group_named("Outer wall marking")
        buttons = marking_group.findChildren(QPushButton)
        buttons_by_text = {button.text(): button for button in buttons}
        self.assertEqual(set(buttons_by_text), {"Mark", "Auto mark", "Edit"})
        self.assertFalse(
            hasattr(self.workspace, "outer_wall_mark_undo_button")
        )

        marking_layout = marking_group.layout()
        assert marking_layout is not None
        self.assertIs(marking_layout.itemAt(0).widget(), buttons_by_text["Mark"])
        self.assertIs(
            marking_layout.itemAt(1).widget(),
            buttons_by_text["Auto mark"],
        )
        self.assertIs(marking_layout.itemAt(2).widget(), buttons_by_text["Edit"])

        side_layout = marking_group.parentWidget().layout()
        assert side_layout is not None
        self.assertEqual(
            side_layout.indexOf(marking_group),
            side_layout.indexOf(self.workspace.transform_controls_scroll_area) + 1,
        )
        self.assertEqual(
            side_layout.indexOf(self.workspace.levels_group),
            side_layout.indexOf(marking_group) + 1,
        )

    def test_stairs_group_contains_editor_status_and_add_controls(self) -> None:
        self._assert_group_contains(
            "Stairs",
            self.workspace.stair_type_combo,
            self.workspace.stair_step_rise_target_spinbox,
            self.workspace.stair_tread_thickness_spinbox,
            self.workspace.stair_nosing_overhang_spinbox,
            self.workspace.stair_nosing_left_checkbox,
            self.workspace.stair_nosing_right_checkbox,
            self.workspace.stair_nosing_front_checkbox,
            self.workspace.stair_tread_edge_combo,
            self.workspace.stair_tread_edge_radius_spinbox,
            self.workspace.stair_starting_step_combo,
            self.workspace.stair_starting_step_edge_radius_spinbox,
            self.workspace.stair_starting_step_edge_points_spinbox,
            self.workspace.stair_stringer_placement_combo,
            self.workspace.stair_calculated_step_count_label,
            self.workspace.stair_actual_rise_label,
            self.workspace.stair_status_label,
            self.workspace.add_stairs_button,
            labels={
                "Type",
                "Step rise target",
                "Tread thickness",
                "Nosing overhang",
                "Nosing placement",
                "Tread edge",
                "Edge radius",
                "Starting step",
                "Points",
                "Stringer placement",
                "Calculated step count",
                "Actual rise",
            },
        )

    def test_nosing_editor_fits_the_default_canvas_side_panel_viewport(self) -> None:
        scroll_area = self.workspace.canvas_left_scroll_area
        self.assertIsInstance(scroll_area, QScrollArea)
        assert isinstance(scroll_area, QScrollArea)
        viewport = scroll_area.viewport()
        self.assertGreater(viewport.width(), 0)
        self.assertLessEqual(viewport.width(), self.workspace.left_panel.width())

        stair_parameters_layout = self.workspace.stairs_group.findChild(QFormLayout)
        self.assertIsNotNone(stair_parameters_layout)
        assert stair_parameters_layout is not None
        _row_index, role = stair_parameters_layout.getWidgetPosition(
            self.workspace.stair_nosing_row_widget
        )
        self.assertEqual(role, QFormLayout.ItemRole.SpanningRole)
        self.assertGreaterEqual(
            self.workspace.stair_nosing_row_widget.width(),
            min(
                stair_parameters_layout.geometry().width(),
                viewport.width() - 24,
            )
            - 4,
        )

        overhang_label_center = self.workspace.stair_nosing_overhang_label.mapTo(
            self.workspace.stair_nosing_row_widget,
            self.workspace.stair_nosing_overhang_label.rect().center(),
        )
        overhang_field_center = self.workspace.stair_nosing_overhang_spinbox.mapTo(
            self.workspace.stair_nosing_row_widget,
            self.workspace.stair_nosing_overhang_spinbox.rect().center(),
        )
        placement_label_center = self.workspace.stair_nosing_placement_label.mapTo(
            self.workspace.stair_nosing_row_widget,
            self.workspace.stair_nosing_placement_label.rect().center(),
        )
        left_center = self.workspace.stair_nosing_left_checkbox.mapTo(
            self.workspace.stair_nosing_row_widget,
            self.workspace.stair_nosing_left_checkbox.rect().center(),
        )
        self.assertGreater(overhang_field_center.x(), overhang_label_center.x())
        self.assertLessEqual(
            abs(overhang_field_center.y() - overhang_label_center.y()),
            2,
        )
        if abs(left_center.y() - placement_label_center.y()) <= 2:
            self.assertGreater(left_center.x(), placement_label_center.x())
        else:
            self.assertGreater(left_center.y(), placement_label_center.y())
        self.assertGreater(left_center.y(), overhang_field_center.y())
        self.assertFalse(self.workspace.stair_nosing_overhang_label.wordWrap())
        self.assertFalse(self.workspace.stair_nosing_placement_label.wordWrap())
        self.assertGreaterEqual(
            self.workspace.stair_nosing_overhang_label.width(),
            self.workspace.stair_nosing_overhang_label.sizeHint().width(),
        )
        self.assertGreaterEqual(
            self.workspace.stair_nosing_placement_label.width(),
            self.workspace.stair_nosing_placement_label.sizeHint().width(),
        )
        for checkbox in (
            self.workspace.stair_nosing_left_checkbox,
            self.workspace.stair_nosing_right_checkbox,
            self.workspace.stair_nosing_front_checkbox,
        ):
            self.assertGreaterEqual(
                checkbox.width(),
                checkbox.minimumSizeHint().width(),
            )

        scroll_area.ensureWidgetVisible(self.workspace.stair_nosing_row_widget, 0, 0)
        _qt_application.processEvents()
        for widget in (
            self.workspace.stair_nosing_overhang_label,
            self.workspace.stair_nosing_overhang_spinbox,
            self.workspace.stair_nosing_placement_label,
            self.workspace.stair_nosing_left_checkbox,
            self.workspace.stair_nosing_right_checkbox,
            self.workspace.stair_nosing_front_checkbox,
        ):
            top_left = widget.mapTo(viewport, QPoint(0, 0))
            bottom_right = widget.mapTo(
                viewport,
                QPoint(widget.width() - 1, widget.height() - 1),
            )
            self.assertTrue(
                viewport.rect().contains(top_left),
                f"{widget.objectName() or widget.__class__.__name__} starts outside "
                "the Canvas side-panel viewport",
            )
            self.assertTrue(
                viewport.rect().contains(bottom_right),
                f"{widget.objectName() or widget.__class__.__name__} is clipped by "
                "the Canvas side-panel viewport",
            )
            self.assertEqual(widget.visibleRegion().boundingRect(), widget.rect())

        _row_index, role = stair_parameters_layout.getWidgetPosition(
            self.workspace.stair_starting_step_row_widget
        )
        self.assertEqual(role, QFormLayout.ItemRole.SpanningRole)
        self.assertGreaterEqual(
            self.workspace.stair_starting_step_row_widget.width(),
            min(
                stair_parameters_layout.geometry().width(),
                viewport.width() - 24,
            )
            - 4,
        )
        starting_step_center = self.workspace.stair_starting_step_combo.mapTo(
            self.workspace.stair_starting_step_row_widget,
            self.workspace.stair_starting_step_combo.rect().center(),
        )
        starting_radius_center = (
            self.workspace.stair_starting_step_edge_radius_spinbox.mapTo(
                self.workspace.stair_starting_step_row_widget,
                self.workspace.stair_starting_step_edge_radius_spinbox.rect().center(),
            )
        )
        self.assertGreater(starting_radius_center.y(), starting_step_center.y())
        starting_points_center = (
            self.workspace.stair_starting_step_edge_points_spinbox.mapTo(
                self.workspace.stair_starting_step_row_widget,
                self.workspace.stair_starting_step_edge_points_spinbox.rect().center(),
            )
        )
        self.assertGreater(starting_points_center.x(), starting_radius_center.x())
        self.assertLessEqual(
            abs(starting_points_center.y() - starting_radius_center.y()),
            2,
        )

        def assert_starting_step_fields_share_the_row() -> None:
            scroll_area.ensureWidgetVisible(
                self.workspace.stair_starting_step_row_widget,
                0,
                0,
            )
            _qt_application.processEvents()
            fields = (
                self.workspace.stair_starting_step_combo,
                self.workspace.stair_starting_step_edge_radius_spinbox,
                self.workspace.stair_starting_step_edge_points_spinbox,
            )
            detail_groups = (
                self.workspace.stair_starting_step_edge_radius_field_widget,
                self.workspace.stair_starting_step_edge_points_field_widget,
            )
            group_widths = tuple(group.width() for group in detail_groups)
            self.assertLessEqual(max(group_widths) - min(group_widths), 2)
            self.assertGreater(min(group_widths), 0)
            row = self.workspace.stair_starting_step_row_widget
            self.assertGreaterEqual(
                self.workspace.stair_starting_step_field_widget.width(),
                row.width() - 2,
            )
            self.assertGreaterEqual(
                self.workspace.stair_starting_step_detail_row_widget.width(),
                row.width() - 2,
            )
            for field in fields:
                self.assertGreaterEqual(field.width(), field.minimumSizeHint().width())

            for label in (
                self.workspace.stair_starting_step_label,
                self.workspace.stair_starting_step_edge_radius_label,
                self.workspace.stair_starting_step_edge_points_label,
            ):
                self.assertFalse(label.wordWrap())
                self.assertGreaterEqual(label.width(), label.sizeHint().width())
            for widget in (
                self.workspace.stair_starting_step_label,
                fields[0],
                self.workspace.stair_starting_step_edge_radius_label,
                fields[1],
                self.workspace.stair_starting_step_edge_points_label,
                fields[2],
            ):
                top_left_in_row = widget.mapTo(row, QPoint(0, 0))
                bottom_right_in_row = widget.mapTo(
                    row,
                    QPoint(widget.width() - 1, widget.height() - 1),
                )
                self.assertTrue(row.rect().contains(top_left_in_row))
                self.assertTrue(row.rect().contains(bottom_right_in_row))

                top_left_in_viewport = widget.mapTo(viewport, QPoint(0, 0))
                bottom_right_in_viewport = widget.mapTo(
                    viewport,
                    QPoint(widget.width() - 1, widget.height() - 1),
                )
                self.assertTrue(viewport.rect().contains(top_left_in_viewport))
                self.assertTrue(viewport.rect().contains(bottom_right_in_viewport))
                self.assertEqual(widget.visibleRegion().boundingRect(), widget.rect())

        assert_starting_step_fields_share_the_row()

        _row_index, role = stair_parameters_layout.getWidgetPosition(
            self.workspace.stair_tread_edge_row_widget
        )
        self.assertEqual(role, QFormLayout.ItemRole.SpanningRole)
        self.assertGreaterEqual(
            self.workspace.stair_tread_edge_row_widget.width(),
            min(
                stair_parameters_layout.geometry().width(),
                viewport.width() - 24,
            )
            - 4,
        )
        edge_combo_center = self.workspace.stair_tread_edge_combo.mapTo(
            self.workspace.stair_tread_edge_row_widget,
            self.workspace.stair_tread_edge_combo.rect().center(),
        )
        radius_center = self.workspace.stair_tread_edge_radius_spinbox.mapTo(
            self.workspace.stair_tread_edge_row_widget,
            self.workspace.stair_tread_edge_radius_spinbox.rect().center(),
        )
        self.assertGreater(radius_center.x(), edge_combo_center.x())
        self.assertLessEqual(abs(radius_center.y() - edge_combo_center.y()), 2)
        tread_field_widths = (
            self.workspace.stair_tread_edge_field_widget.width(),
            self.workspace.stair_tread_edge_radius_field_widget.width(),
        )
        self.assertLessEqual(
            max(tread_field_widths) - min(tread_field_widths),
            2,
        )
        self.assertGreaterEqual(
            self.workspace.stair_tread_edge_combo.width(),
            self.workspace.stair_tread_edge_combo.minimumSizeHint().width(),
        )
        self.assertGreaterEqual(
            self.workspace.stair_tread_edge_radius_spinbox.width(),
            self.workspace.stair_tread_edge_radius_spinbox.minimumSizeHint().width(),
        )
        for label in (
            self.workspace.stair_tread_edge_label,
            self.workspace.stair_tread_edge_radius_label,
        ):
            self.assertFalse(label.wordWrap())
            self.assertGreaterEqual(label.width(), label.sizeHint().width())

        scroll_area.ensureWidgetVisible(
            self.workspace.stair_tread_edge_row_widget,
            0,
            0,
        )
        _qt_application.processEvents()
        for widget in (
            self.workspace.stair_tread_edge_label,
            self.workspace.stair_tread_edge_combo,
            self.workspace.stair_tread_edge_radius_label,
            self.workspace.stair_tread_edge_radius_spinbox,
        ):
            top_left = widget.mapTo(viewport, QPoint(0, 0))
            bottom_right = widget.mapTo(
                viewport,
                QPoint(widget.width() - 1, widget.height() - 1),
            )
            self.assertTrue(viewport.rect().contains(top_left))
            self.assertTrue(viewport.rect().contains(bottom_right))
            self.assertEqual(widget.visibleRegion().boundingRect(), widget.rect())

        default_viewport_width = viewport.width()
        self.workspace.workspace_splitter.setSizes([700, 500, 400])
        _qt_application.processEvents()
        self.assertGreater(viewport.width(), default_viewport_width)
        assert_starting_step_fields_share_the_row()

    def test_stairs_editor_reflows_without_clipped_text_on_narrower_panels(
        self,
    ) -> None:
        scroll_area = self.workspace.canvas_left_scroll_area
        self.assertIsInstance(scroll_area, QScrollArea)
        assert isinstance(scroll_area, QScrollArea)
        viewport = scroll_area.viewport()
        form = self.workspace.stairs_group.findChild(QFormLayout)
        self.assertIsNotNone(form)
        assert form is not None

        for panel_width, expected_wrap, local_scroll_expected in (
            (560, QFormLayout.RowWrapPolicy.DontWrapRows, False),
            (420, QFormLayout.RowWrapPolicy.WrapLongRows, False),
            (220, QFormLayout.RowWrapPolicy.WrapLongRows, True),
        ):
            with self.subTest(panel_width=panel_width):
                self.workspace.workspace_splitter.setSizes(
                    [panel_width, 1120 - panel_width, 460]
                )
                _qt_application.processEvents()
                _qt_application.processEvents()
                self.assertEqual(form.rowWrapPolicy(), expected_wrap)
                local_scroll = self.workspace.stairs_scroll_area
                self.assertLessEqual(
                    local_scroll.mapTo(viewport, QPoint(0, 0)).x()
                    + local_scroll.width(),
                    viewport.width(),
                )
                self.assertEqual(
                    local_scroll.horizontalScrollBar().maximum() > 0,
                    local_scroll_expected,
                )
                if not local_scroll_expected:
                    self.assertLessEqual(
                        self.workspace.stairs_group.mapTo(viewport, QPoint(0, 0)).x()
                        + self.workspace.stairs_group.width(),
                        viewport.width(),
                    )
                else:
                    self.assertGreater(
                        self.workspace.stairs_group.width(),
                        local_scroll.viewport().width(),
                    )
                expected_form_width = self.workspace.stairs_group.width() - 16
                for row in (
                    self.workspace.stair_nosing_row_widget,
                    self.workspace.stair_tread_edge_row_widget,
                    self.workspace.stair_starting_step_row_widget,
                ):
                    self.assertGreaterEqual(row.width(), expected_form_width - 2)
                placement_label_top = self.workspace.stair_nosing_placement_label.mapTo(
                    self.workspace.stair_nosing_row_widget, QPoint(0, 0)
                )
                checkbox_top = self.workspace.stair_nosing_left_checkbox.mapTo(
                    self.workspace.stair_nosing_row_widget, QPoint(0, 0)
                )
                if panel_width == 560:
                    self.assertEqual(checkbox_top.y(), placement_label_top.y())
                else:
                    self.assertGreater(checkbox_top.y(), placement_label_top.y())
                checkbox_widths = tuple(
                    checkbox.width()
                    for checkbox in (
                        self.workspace.stair_nosing_left_checkbox,
                        self.workspace.stair_nosing_right_checkbox,
                        self.workspace.stair_nosing_front_checkbox,
                    )
                )
                self.assertLessEqual(max(checkbox_widths) - min(checkbox_widths), 2)

                labels = (
                    self.workspace.stair_nosing_overhang_label,
                    self.workspace.stair_nosing_placement_label,
                    self.workspace.stair_tread_edge_label,
                    self.workspace.stair_tread_edge_radius_label,
                    self.workspace.stair_starting_step_label,
                    self.workspace.stair_starting_step_edge_radius_label,
                    self.workspace.stair_starting_step_edge_points_label,
                    *(form.labelForField(control) for control in (
                        self.workspace.stair_type_combo,
                        self.workspace.stair_step_rise_target_spinbox,
                        self.workspace.stair_tread_thickness_spinbox,
                        self.workspace.stair_stringer_placement_combo,
                        self.workspace.stair_calculated_step_count_label,
                        self.workspace.stair_actual_rise_label,
                    )),
                )
                for label in labels:
                    self.assertIsInstance(label, QLabel)
                    assert isinstance(label, QLabel)
                    self.assertFalse(label.wordWrap())
                    self.assertGreaterEqual(label.width(), label.sizeHint().width())
                    scroll_area.ensureWidgetVisible(label, 0, 0)
                    _qt_application.processEvents()
                    if not local_scroll_expected:
                        self.assertTrue(
                            viewport.rect().contains(
                                label.mapTo(viewport, QPoint(label.width() - 1, 0))
                            ),
                            f"{label.text()!r} is cut off at panel width {panel_width}",
                        )

                for control in (
                    self.workspace.stair_nosing_overhang_spinbox,
                    self.workspace.stair_nosing_left_checkbox,
                    self.workspace.stair_nosing_right_checkbox,
                    self.workspace.stair_nosing_front_checkbox,
                    self.workspace.stair_tread_edge_combo,
                    self.workspace.stair_tread_edge_radius_spinbox,
                    self.workspace.stair_starting_step_combo,
                    self.workspace.stair_starting_step_edge_radius_spinbox,
                    self.workspace.stair_starting_step_edge_points_spinbox,
                    self.workspace.stair_type_combo,
                    self.workspace.stair_step_rise_target_spinbox,
                    self.workspace.stair_tread_thickness_spinbox,
                    self.workspace.stair_stringer_placement_combo,
                ):
                    self.assertGreaterEqual(
                        control.width(), control.minimumSizeHint().width()
                    )

                if local_scroll_expected:
                    local_scroll.horizontalScrollBar().setValue(
                        local_scroll.horizontalScrollBar().maximum()
                    )
                    _qt_application.processEvents()
                    self.assertGreater(
                        self.workspace.stair_nosing_front_checkbox.mapTo(
                            local_scroll.viewport(), QPoint(0, 0)
                        ).x(),
                        0,
                    )
                    scroll_area.ensureWidgetVisible(
                        self.workspace.add_stairs_button, 0, 0
                    )
                    _qt_application.processEvents()
                    self.assertEqual(
                        self.workspace.add_stairs_button.visibleRegion()
                        .boundingRect()
                        .height(),
                        self.workspace.add_stairs_button.height(),
                    )

        for panel_width, local_scroll_expected in ((560, False), (220, True)):
            with self.subTest(transition_width=panel_width):
                self.workspace.workspace_splitter.setSizes(
                    [panel_width, 1120 - panel_width, 460]
                )
                _qt_application.processEvents()
                _qt_application.processEvents()
                local_scroll = self.workspace.stairs_scroll_area
                self.assertEqual(
                    local_scroll.horizontalScrollBar().maximum() > 0,
                    local_scroll_expected,
                )
                self.assertEqual(local_scroll.verticalScrollBar().maximum(), 0)
                scroll_area.ensureWidgetVisible(self.workspace.add_stairs_button, 0, 0)
                _qt_application.processEvents()
                self.assertEqual(
                    self.workspace.add_stairs_button.visibleRegion()
                    .boundingRect()
                    .height(),
                    self.workspace.add_stairs_button.height(),
                )

    def test_open_spaces_group_contains_status_and_add_controls(self) -> None:
        self._assert_group_contains(
            "Open spaces",
            self.workspace.open_space_status_label,
            self.workspace.add_open_space_button,
        )

    def test_level_dimensions_group_contains_both_numeric_fields(self) -> None:
        self._assert_group_contains(
            "Level dimensions",
            self.workspace.height_level_spinbox,
            self.workspace.floor_thickness_spinbox,
            labels={"Height level", "Floor thickness"},
        )

    def test_narrow_workspace_keeps_the_full_canvas_interactive(self) -> None:
        self.workspace.resize(760, 700)
        _qt_application.processEvents()
        canvas = self.workspace.canvas
        visible_bounds = canvas.visibleRegion().boundingRect()

        self.assertEqual(visible_bounds, canvas.rect())
        top_right = QPoint(canvas.width() - 4, 12)
        self.assertIs(
            QApplication.widgetAt(canvas.mapToGlobal(top_right)),
            canvas,
        )

        canvas.blueprint_image = QImage(
            100,
            100,
            QImage.Format.Format_RGB32,
        )
        existing_vertex = canvas.vertex_data.add_vertex(90.0, 10.0)
        QTest.mouseClick(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=canvas._image_to_widget(90.0, 10.0).toPoint(),
        )
        self.assertEqual(canvas.selected_vertex_id, existing_vertex.id)

        QTest.mouseClick(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=canvas._image_to_widget(90.0, 70.0).toPoint(),
        )
        self.assertEqual(len(canvas.vertex_data.vertices), 2)


if __name__ == "__main__":
    unittest.main()
