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
    QScrollArea,
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
            for group in self.workspace.side_panel.findChildren(QGroupBox)
            if group.title() == title
        ]
        self.assertEqual(
            len(matching_groups),
            1,
            f"Expected exactly one Canvas group titled {title!r}",
        )
        return matching_groups[0]

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
            self.workspace.blueprint_name_label,
            self.workspace.levels_list,
            self.workspace.include_yes_radio,
            self.workspace.include_no_radio,
            labels={"Include"},
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

    def test_wall_mirrors_group_follows_doorways_with_three_actions(self) -> None:
        self._assert_group_contains(
            "Wall mirrors",
            self.workspace.wall_mirror_up_button,
            self.workspace.wall_mirror_undo_button,
            self.workspace.wall_mirror_down_button,
        )
        self.assertEqual(self.workspace.wall_mirror_up_button.text(), "Up")
        self.assertEqual(self.workspace.wall_mirror_undo_button.text(), "Undo")
        self.assertEqual(self.workspace.wall_mirror_down_button.text(), "Down")

        mirrors_layout = self.workspace.wall_mirrors_group.layout()
        assert mirrors_layout is not None
        self.assertIs(
            mirrors_layout.itemAt(0).widget(),
            self.workspace.wall_mirror_up_button,
        )
        self.assertIs(
            mirrors_layout.itemAt(1).widget(),
            self.workspace.wall_mirror_undo_button,
        )
        self.assertIs(
            mirrors_layout.itemAt(2).widget(),
            self.workspace.wall_mirror_down_button,
        )

        side_layout = self.workspace.doorways_group.parentWidget().layout()
        assert side_layout is not None
        self.assertEqual(
            side_layout.indexOf(self.workspace.wall_mirrors_group),
            side_layout.indexOf(self.workspace.doorways_group) + 1,
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
        scroll_area = self.workspace.side_tabs.currentWidget()
        self.assertIsInstance(scroll_area, QScrollArea)
        assert isinstance(scroll_area, QScrollArea)
        viewport = scroll_area.viewport()
        self.assertGreaterEqual(viewport.width(), 450)
        self.assertLessEqual(viewport.width(), 500)

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
        self.assertGreater(left_center.x(), placement_label_center.x())
        self.assertLessEqual(abs(left_center.y() - placement_label_center.y()), 2)
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
        self.workspace.workspace_splitter.setSizes([900, 700])
        _qt_application.processEvents()
        self.assertGreater(viewport.width(), default_viewport_width)
        assert_starting_step_fields_share_the_row()

    def test_stairs_editor_reflows_without_clipped_text_on_narrower_panels(
        self,
    ) -> None:
        scroll_area = self.workspace.side_tabs.currentWidget()
        self.assertIsInstance(scroll_area, QScrollArea)
        assert isinstance(scroll_area, QScrollArea)
        viewport = scroll_area.viewport()
        form = self.workspace.stairs_group.findChild(QFormLayout)
        self.assertIsNotNone(form)
        assert form is not None

        for workspace_width, expected_wrap, local_scroll_expected in (
            (1500, QFormLayout.RowWrapPolicy.DontWrapRows, False),
            (1300, QFormLayout.RowWrapPolicy.WrapLongRows, False),
            (760, QFormLayout.RowWrapPolicy.WrapLongRows, True),
        ):
            with self.subTest(workspace_width=workspace_width):
                self.workspace.resize(workspace_width, 900)
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
                            f"{label.text()!r} is cut off at width {workspace_width}",
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

        for workspace_width, local_scroll_expected in ((1600, False), (760, True)):
            with self.subTest(transition_width=workspace_width):
                self.workspace.resize(workspace_width, 900)
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
