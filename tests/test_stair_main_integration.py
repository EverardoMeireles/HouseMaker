# ### Environment setup ###
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import trimesh
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QImage, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.glb import (
    STAIR_PART_SUPPORT,
    STAIR_PART_TREADS,
    GeneratedModel,
    build_canvas_stair_part_targets,
)
from housemaker.main import BlueprintWorkspace
from housemaker.models import (
    DEFAULT_STAIR_NOSING_PLACEMENTS,
    DEFAULT_STAIR_STARTING_STEP,
    DEFAULT_STAIR_STARTING_STEP_EDGE_POINTS,
    DEFAULT_STAIR_STARTING_STEP_EDGE_RADIUS_METERS,
    DEFAULT_STAIR_TARGET_RISE_METERS,
    DEFAULT_STAIR_TREAD_EDGE_RADIUS_METERS,
    DEFAULT_STAIR_TREAD_OVERHANG_METERS,
    DEFAULT_STAIR_TREAD_THICKNESS_METERS,
    GROUND_LEVEL_INDEX,
    MAX_STAIR_STARTING_STEP_EDGE_POINTS,
    MIN_STAIR_STARTING_STEP_EDGE_POINTS,
    STAIR_NOSING_FRONT,
    STAIR_NOSING_LEFT,
    STAIR_NOSING_RIGHT,
    STAIR_STARTING_STEP_BULLNOSE,
    STAIR_STARTING_STEP_CURTAIL,
    STAIR_STARTING_STEP_NONE,
    STAIR_STRINGER_BOTH,
    STAIR_STRINGER_LEFT,
    STAIR_STRINGER_NONE,
    STAIR_STRINGER_RIGHT,
    STAIR_STYLE_FLOATING,
    STAIR_TREAD_EDGE_ROUNDED,
    STAIR_TREAD_EDGE_STRAIGHT,
    STAIR_TYPE_FLOATING,
    STAIR_TYPE_SUPPORTED,
    StairData,
)
from housemaker.surface_texture_state import (
    SurfaceTextureAssignment,
    SurfaceTextureData,
)

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _make_current_canvas_clickable(workspace: BlueprintWorkspace) -> None:
    """Give the selected level a small in-memory blueprint for click tests."""

    workspace.current_level.image_size_pixels = (100.0, 100.0)
    workspace._sync_canvas_to_current_level()
    workspace.canvas.blueprint_image = QImage(
        100,
        100,
        QImage.Format.Format_RGB32,
    )
    workspace.canvas.blueprint_image.fill(Qt.GlobalColor.white)
    workspace.canvas.set_stair_context(workspace.stairs, workspace.current_level)
    workspace.canvas.update()
    _qt_application.processEvents()


def _image_position(
    workspace: BlueprintWorkspace,
    image_x: float,
    image_y: float,
):
    return workspace.canvas._image_to_widget(image_x, image_y).toPoint()


def _add_wall_segment(
    workspace: BlueprintWorkspace,
    start: tuple[float, float],
    end: tuple[float, float],
) -> None:
    """Add one selectable wall segment to the active level fixture."""

    start_vertex = workspace.current_level.vertex_data.add_vertex(*start)
    end_vertex = workspace.current_level.vertex_data.add_vertex(*end)
    workspace.current_level.vertex_data.add_edge(
        start_vertex.id,
        end_vertex.id,
    )


def _make_editable_stair(**changes: object) -> StairData:
    """Return one modern stair with semantic parts for editor tests."""

    values: dict[str, object] = {
        "start_level_index": GROUND_LEVEL_INDEX,
        "start_a_x": 20.0,
        "start_a_y": 30.0,
        "start_b_x": 50.0,
        "start_b_y": 30.0,
        "end_level_index": GROUND_LEVEL_INDEX + 1,
        "end_a_x": 20.0,
        "end_a_y": 70.0,
        "end_b_x": 50.0,
        "end_b_y": 70.0,
        "stair_type": STAIR_TYPE_SUPPORTED,
        "stringer_placement": STAIR_STRINGER_BOTH,
    }
    values.update(changes)
    return StairData(**values)


def _send_wheel_event(widget: QWidget, delta: int) -> QWheelEvent:
    """Send one vertical wheel step to a visible widget."""

    position = QPointF(widget.rect().center())
    global_position = QPointF(widget.mapToGlobal(position.toPoint()))
    event = QWheelEvent(
        position,
        global_position,
        QPoint(),
        QPoint(0, int(delta)),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )
    QApplication.sendEvent(widget, event)
    return event


# ### Main integration tests ###
class StairMainIntegrationTests(unittest.TestCase):
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
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self._temporary_directory.cleanup()

    def test_add_stairs_uses_two_levels_and_commits_the_selected_type(self) -> None:
        _add_wall_segment(self.workspace, (20.0, 35.0), (45.0, 35.0))
        _make_current_canvas_clickable(self.workspace)
        self.workspace.stair_type_combo.setCurrentIndex(
            self.workspace.stair_type_combo.findData(STAIR_TYPE_FLOATING)
        )
        self.workspace.stair_starting_step_combo.setCurrentIndex(
            self.workspace.stair_starting_step_combo.findData(
                STAIR_STARTING_STEP_CURTAIL
            )
        )

        with patch("housemaker.main.QMessageBox.information"):
            self.workspace.add_stairs_button.click()

        self.assertTrue(self.workspace.canvas.is_stair_placement_active())
        self.assertIsNone(self.workspace.canvas.get_pending_stair_placement())
        self.assertFalse(self.workspace.add_stairs_button.isEnabled())
        self.assertFalse(self.workspace.stair_starting_step_combo.isEnabled())

        QTest.mouseClick(
            self.workspace.canvas,
            Qt.MouseButton.LeftButton,
            pos=_image_position(self.workspace, 20.0, 35.0),
        )
        QTest.mouseClick(
            self.workspace.canvas,
            Qt.MouseButton.LeftButton,
            pos=_image_position(self.workspace, 45.0, 35.0),
        )
        _qt_application.processEvents()

        pending = self.workspace.canvas.get_pending_stair_placement()
        self.assertIsNotNone(pending)
        self.assertEqual(pending.start_level_index, GROUND_LEVEL_INDEX)  # type: ignore[union-attr]
        self.assertEqual(pending.style, STAIR_STYLE_FLOATING)  # type: ignore[union-attr]

        destination_level_index = GROUND_LEVEL_INDEX + 1
        self.workspace._handle_level_selection_changed(destination_level_index)
        _add_wall_segment(self.workspace, (50.0, 58.0), (90.0, 58.0))
        _make_current_canvas_clickable(self.workspace)

        QTest.mouseClick(
            self.workspace.canvas,
            Qt.MouseButton.LeftButton,
            pos=_image_position(self.workspace, 50.0, 58.0),
        )
        QTest.mouseClick(
            self.workspace.canvas,
            Qt.MouseButton.LeftButton,
            pos=_image_position(self.workspace, 90.0, 58.0),
        )
        _qt_application.processEvents()

        self.assertTrue(self.workspace.canvas.is_stair_placement_active())
        self.assertTrue(
            self.workspace.canvas.is_stair_ready_for_confirmation()
        )
        self.assertEqual(self.workspace.stairs, [])
        self.assertEqual(
            self.workspace.add_stairs_button.text(),
            "Confirm stairs",
        )
        self.assertTrue(self.workspace.add_stairs_button.isEnabled())

        self.workspace.add_stairs_button.click()
        _qt_application.processEvents()

        self.assertFalse(self.workspace.canvas.is_stair_placement_active())
        self.assertTrue(self.workspace.stair_starting_step_combo.isEnabled())
        self.assertEqual(len(self.workspace.stairs), 1)
        stair = self.workspace.stairs[0]
        self.assertEqual(stair.stair_type, STAIR_TYPE_FLOATING)
        self.assertEqual(stair.style, STAIR_STYLE_FLOATING)
        self.assertEqual(stair.starting_step, STAIR_STARTING_STEP_CURTAIL)
        self.assertEqual(stair.start_level_index, GROUND_LEVEL_INDEX)
        self.assertEqual(stair.end_level_index, destination_level_index)
        self.assertFalse(hasattr(self.workspace, "stairs_list"))
        self.assertIn("Added floating stairs", self.workspace.stair_status_label.text())

        self.workspace.workspace_tabs.setCurrentWidget(
            self.workspace.scene_3d_workspace
        )
        _qt_application.processEvents()
        viewer_model = self.workspace.viewer.model
        self.assertIsNotNone(viewer_model)
        self.assertIn(
            "stair_1_floating",
            viewer_model.scene.geometry,  # type: ignore[union-attr]
        )

    def test_stair_type_combo_offers_only_supported_and_floating(self) -> None:
        combo = self.workspace.stair_type_combo

        self.assertEqual(
            tuple(combo.itemData(index) for index in range(combo.count())),
            (STAIR_TYPE_SUPPORTED, STAIR_TYPE_FLOATING),
        )
        self.assertEqual(
            tuple(combo.itemText(index) for index in range(combo.count())),
            ("Supported", "Floating"),
        )

    def test_stair_type_combo_ignores_wheel_and_accepts_keyboard(self) -> None:
        combo = self.workspace.stair_type_combo
        combo.setCurrentIndex(0)

        wheel_event = _send_wheel_event(combo, -120)

        self.assertTrue(wheel_event.isAccepted())
        self.assertEqual(combo.currentIndex(), 0)

        combo.setFocus(Qt.FocusReason.OtherFocusReason)
        QTest.keyClick(combo, Qt.Key.Key_Down)

        self.assertEqual(combo.currentData(), STAIR_TYPE_FLOATING)

    def test_starting_step_combo_ignores_wheel_input(self) -> None:
        combo = self.workspace.stair_starting_step_combo
        combo.setCurrentIndex(combo.findData(STAIR_STARTING_STEP_NONE))

        wheel_event = _send_wheel_event(combo, -120)

        self.assertTrue(wheel_event.isAccepted())
        self.assertEqual(combo.currentData(), STAIR_STARTING_STEP_NONE)

    def test_stair_type_combo_accepts_popup_clicks(self) -> None:
        combo = self.workspace.stair_type_combo
        combo.setCurrentIndex(0)
        target_index = combo.model().index(1, 0)
        combo.showPopup()
        _qt_application.processEvents()

        QTest.mouseClick(
            combo.view().viewport(),
            Qt.MouseButton.LeftButton,
            pos=combo.view().visualRect(target_index).center(),
        )
        _qt_application.processEvents()

        self.assertEqual(
            combo.currentData(),
            STAIR_TYPE_FLOATING,
        )

    def test_stair_editor_uses_centimeters_and_constrained_choices(self) -> None:
        self.assertEqual(
            self.workspace.stair_step_rise_target_spinbox.value(),
            DEFAULT_STAIR_TARGET_RISE_METERS * 100.0,
        )
        self.assertEqual(
            self.workspace.stair_tread_thickness_spinbox.value(),
            DEFAULT_STAIR_TREAD_THICKNESS_METERS * 100.0,
        )
        self.assertEqual(
            self.workspace.stair_nosing_overhang_spinbox.value(),
            DEFAULT_STAIR_TREAD_OVERHANG_METERS * 100.0,
        )
        self.assertEqual(
            self.workspace.stair_tread_edge_radius_spinbox.value(),
            DEFAULT_STAIR_TREAD_EDGE_RADIUS_METERS * 100.0,
        )
        self.assertFalse(
            self.workspace.stair_tread_edge_radius_spinbox.isEnabled()
        )
        self.assertFalse(self.workspace.stair_tread_edge_radius_label.isEnabled())
        self.assertFalse(self.workspace.stair_nosing_left_checkbox.isChecked())
        self.assertFalse(self.workspace.stair_nosing_right_checkbox.isChecked())
        self.assertTrue(self.workspace.stair_nosing_front_checkbox.isChecked())
        self.assertEqual(
            self.workspace._read_stair_editor_parameters().nosing_placements,
            DEFAULT_STAIR_NOSING_PLACEMENTS,
        )
        self.assertEqual(
            self.workspace.stair_starting_step_combo.currentData(),
            DEFAULT_STAIR_STARTING_STEP,
        )
        self.assertEqual(
            self.workspace.stair_starting_step_edge_radius_spinbox.value(),
            DEFAULT_STAIR_STARTING_STEP_EDGE_RADIUS_METERS * 100.0,
        )
        self.assertEqual(
            self.workspace.stair_starting_step_edge_points_spinbox.value(),
            DEFAULT_STAIR_STARTING_STEP_EDGE_POINTS,
        )
        self.assertEqual(
            self.workspace.stair_starting_step_edge_points_spinbox.minimum(),
            MIN_STAIR_STARTING_STEP_EDGE_POINTS,
        )
        self.assertEqual(
            self.workspace.stair_starting_step_edge_points_spinbox.maximum(),
            MAX_STAIR_STARTING_STEP_EDGE_POINTS,
        )
        self.assertFalse(
            self.workspace.stair_starting_step_edge_radius_spinbox.isEnabled()
        )
        self.assertFalse(
            self.workspace.stair_starting_step_edge_radius_label.isEnabled()
        )
        self.assertFalse(
            self.workspace.stair_starting_step_edge_points_spinbox.isEnabled()
        )
        self.assertFalse(
            self.workspace.stair_starting_step_edge_points_label.isEnabled()
        )
        self.assertIn(
            "clamped",
            self.workspace.stair_starting_step_edge_radius_spinbox.toolTip(),
        )
        self.assertIn(
            "clamped",
            self.workspace.stair_tread_edge_radius_spinbox.toolTip(),
        )
        self.assertIn(
            "matching point on each side",
            self.workspace.stair_starting_step_edge_points_spinbox.toolTip(),
        )
        self.assertEqual(
            self.workspace._stair_preview_update_timer.interval(),
            35,
        )
        self.assertIn(
            "actual rise is adjusted",
            self.workspace.stair_step_rise_target_spinbox.toolTip(),
        )
        for spinbox in (
            self.workspace.stair_step_rise_target_spinbox,
            self.workspace.stair_tread_thickness_spinbox,
            self.workspace.stair_nosing_overhang_spinbox,
            self.workspace.stair_tread_edge_radius_spinbox,
            self.workspace.stair_starting_step_edge_radius_spinbox,
        ):
            self.assertEqual(spinbox.suffix(), " cm")
        self.assertEqual(
            tuple(
                self.workspace.stair_tread_edge_combo.itemData(index)
                for index in range(self.workspace.stair_tread_edge_combo.count())
            ),
            (STAIR_TREAD_EDGE_STRAIGHT, STAIR_TREAD_EDGE_ROUNDED),
        )
        self.assertEqual(
            tuple(
                self.workspace.stair_stringer_placement_combo.itemData(index)
                for index in range(
                    self.workspace.stair_stringer_placement_combo.count()
                )
            ),
            (
                STAIR_STRINGER_NONE,
                STAIR_STRINGER_LEFT,
                STAIR_STRINGER_RIGHT,
                STAIR_STRINGER_BOTH,
            ),
        )
        self.assertEqual(
            tuple(
                self.workspace.stair_starting_step_combo.itemData(index)
                for index in range(
                    self.workspace.stair_starting_step_combo.count()
                )
            ),
            (
                STAIR_STARTING_STEP_NONE,
                STAIR_STARTING_STEP_BULLNOSE,
                STAIR_STARTING_STEP_CURTAIL,
            ),
        )
        self.assertEqual(
            tuple(
                self.workspace.stair_starting_step_combo.itemText(index)
                for index in range(
                    self.workspace.stair_starting_step_combo.count()
                )
            ),
            ("None", "Bullnose", "Curtail"),
        )

        rounded_index = self.workspace.stair_tread_edge_combo.findData(
            STAIR_TREAD_EDGE_ROUNDED
        )
        self.workspace.stair_tread_edge_combo.setCurrentIndex(rounded_index)
        self.assertTrue(
            self.workspace.stair_tread_edge_radius_spinbox.isEnabled()
        )
        self.assertTrue(self.workspace.stair_tread_edge_radius_label.isEnabled())
        self.workspace.stair_tread_edge_radius_spinbox.setValue(2.5)
        self.assertAlmostEqual(
            self.workspace._read_stair_editor_parameters().tread_edge_radius_meters,
            0.025,
        )

        straight_index = self.workspace.stair_tread_edge_combo.findData(
            STAIR_TREAD_EDGE_STRAIGHT
        )
        self.workspace.stair_tread_edge_combo.setCurrentIndex(straight_index)
        self.assertFalse(
            self.workspace.stair_tread_edge_radius_spinbox.isEnabled()
        )
        self.assertFalse(self.workspace.stair_tread_edge_radius_label.isEnabled())

        bullnose_index = self.workspace.stair_starting_step_combo.findData(
            STAIR_STARTING_STEP_BULLNOSE
        )
        self.workspace.stair_starting_step_combo.setCurrentIndex(bullnose_index)
        self.assertTrue(
            self.workspace.stair_starting_step_edge_radius_spinbox.isEnabled()
        )
        self.assertTrue(
            self.workspace.stair_starting_step_edge_radius_label.isEnabled()
        )
        self.assertTrue(
            self.workspace.stair_starting_step_edge_points_spinbox.isEnabled()
        )
        self.assertTrue(
            self.workspace.stair_starting_step_edge_points_label.isEnabled()
        )
        self.workspace.stair_starting_step_edge_radius_spinbox.setValue(18.0)
        self.workspace.stair_starting_step_edge_points_spinbox.setValue(4)
        self.assertAlmostEqual(
            self.workspace._read_stair_editor_parameters().starting_step_edge_radius_meters,
            0.18,
        )
        self.assertEqual(
            self.workspace._read_stair_editor_parameters().starting_step_edge_points,
            4,
        )

        none_index = self.workspace.stair_starting_step_combo.findData(
            STAIR_STARTING_STEP_NONE
        )
        self.workspace.stair_starting_step_combo.setCurrentIndex(none_index)
        self.assertFalse(
            self.workspace.stair_starting_step_edge_radius_spinbox.isEnabled()
        )
        self.assertFalse(
            self.workspace.stair_starting_step_edge_radius_label.isEnabled()
        )
        self.assertFalse(
            self.workspace.stair_starting_step_edge_points_spinbox.isEnabled()
        )
        self.assertFalse(
            self.workspace.stair_starting_step_edge_points_label.isEnabled()
        )

    def test_selecting_any_stair_part_loads_owner_and_survives_refresh(self) -> None:
        stair = _make_editable_stair(
            stair_type=STAIR_TYPE_FLOATING,
            target_rise_meters=0.2,
            tread_thickness_meters=0.12,
            tread_overhang_meters=0.04,
            nosing_placements=(STAIR_NOSING_LEFT, STAIR_NOSING_RIGHT),
            tread_edge_profile=STAIR_TREAD_EDGE_ROUNDED,
            tread_edge_radius_meters=0.025,
            starting_step=STAIR_STARTING_STEP_BULLNOSE,
            starting_step_edge_radius_meters=0.18,
            starting_step_edge_points=5,
            stringer_placement=STAIR_STRINGER_NONE,
        )
        self.workspace.stairs = [stair]
        self.workspace.canvas.set_stair_context(
            self.workspace.stairs,
            self.workspace.current_level,
        )
        self.workspace._set_canvas_viewer_targets(())
        semantic_id = next(iter(self.workspace._canvas_stair_part_targets_by_id))

        self.assertTrue(
            self.workspace.viewer.select_canvas_stair_part_target(semantic_id)
        )

        self.assertEqual(self.workspace._editing_stair_index, 0)
        self.assertEqual(
            self.workspace.stair_type_combo.currentData(),
            STAIR_TYPE_FLOATING,
        )
        self.assertEqual(
            self.workspace.stair_tread_edge_combo.currentData(),
            STAIR_TREAD_EDGE_ROUNDED,
        )
        self.assertEqual(
            self.workspace.stair_starting_step_combo.currentData(),
            STAIR_STARTING_STEP_BULLNOSE,
        )
        self.assertEqual(
            self.workspace.stair_stringer_placement_combo.currentData(),
            STAIR_STRINGER_NONE,
        )
        self.assertTrue(
            self.workspace.stair_tread_edge_radius_spinbox.isEnabled()
        )
        self.assertAlmostEqual(
            self.workspace.stair_tread_edge_radius_spinbox.value(),
            2.5,
        )
        self.assertAlmostEqual(
            self.workspace.stair_starting_step_edge_radius_spinbox.value(),
            18.0,
        )
        self.assertEqual(
            self.workspace.stair_starting_step_edge_points_spinbox.value(),
            5,
        )
        self.assertTrue(self.workspace.stair_nosing_left_checkbox.isChecked())
        self.assertTrue(self.workspace.stair_nosing_right_checkbox.isChecked())
        self.assertFalse(self.workspace.stair_nosing_front_checkbox.isChecked())
        self.assertEqual(
            self.workspace.surface_texture_generation.get_selected_surface_ids(),
            (semantic_id,),
        )
        self.assertEqual(
            self.workspace._atlas_surface_assignment_target_ids,
            (semantic_id,),
        )
        self.assertNotEqual(
            self.workspace.stair_calculated_step_count_label.text(),
            "—",
        )
        self.assertTrue(
            self.workspace.stair_actual_rise_label.text().endswith(" cm")
        )

        self.workspace._set_canvas_viewer_targets(())

        self.assertEqual(
            self.workspace.viewer.get_selected_canvas_stair_part_ids(),
            (semantic_id,),
        )
        self.assertEqual(
            self.workspace._atlas_surface_assignment_target_ids,
            (semantic_id,),
        )
        self.assertEqual(self.workspace._editing_stair_index, 0)

        self.workspace._set_atlas_canvas_surface_highlights((semantic_id,))

        self.assertEqual(
            self.workspace.viewer.get_highlighted_canvas_stair_part_ids(),
            (semantic_id,),
        )
        self.assertEqual(
            self.workspace.viewer.get_highlighted_canvas_surface_ids(),
            (),
        )

    def test_stair_changes_preview_then_apply_as_one_undo_action(self) -> None:
        original = _make_editable_stair(tread_thickness_meters=0.08)
        self.workspace.stairs = [original]
        self.workspace._set_canvas_viewer_targets(())
        semantic_id = next(iter(self.workspace._canvas_stair_part_targets_by_id))
        self.workspace.viewer.select_canvas_stair_part_target(semantic_id)
        self.assertEqual(
            self.workspace.add_stairs_button.text(),
            "Apply changes to stair",
        )
        self.assertFalse(self.workspace.add_stairs_button.isEnabled())

        with patch("housemaker.main.build_stair_meshes") as redundant_build:
            self.workspace.stair_tread_thickness_spinbox.setValue(12.0)
            self.workspace.stair_nosing_left_checkbox.setChecked(True)
            self.workspace.stair_nosing_front_checkbox.setChecked(False)
            self.workspace.stair_tread_edge_combo.setCurrentIndex(
                self.workspace.stair_tread_edge_combo.findData(
                    STAIR_TREAD_EDGE_ROUNDED
                )
            )
            self.workspace.stair_tread_edge_radius_spinbox.setValue(2.0)
            self.workspace.stair_starting_step_combo.setCurrentIndex(
                self.workspace.stair_starting_step_combo.findData(
                    STAIR_STARTING_STEP_CURTAIL
                )
            )
            self.workspace.stair_starting_step_edge_points_spinbox.setValue(6)
            self.workspace.stair_stringer_placement_combo.setCurrentIndex(
                self.workspace.stair_stringer_placement_combo.findData(
                    STAIR_STRINGER_NONE
                )
            )

            self.assertEqual(self.workspace.stairs, [original])
            self.assertTrue(self.workspace._stair_preview_update_timer.isActive())
            self.assertEqual(
                self.workspace.add_stairs_button.text(),
                "Apply changes to stair",
            )

            QTest.qWait(80)
            _qt_application.processEvents()
            redundant_build.assert_not_called()

        self.assertIsNotNone(self.workspace._staged_stair)
        self.assertEqual(
            self.workspace.viewer._canvas_stair_preview_stair_index,
            0,
        )
        self.workspace.add_stairs_button.click()

        self.assertEqual(
            self.workspace.stairs[0].tread_thickness_meters,
            0.12,
        )
        self.assertEqual(
            self.workspace.stairs[0].nosing_placements,
            (STAIR_NOSING_LEFT,),
        )
        self.assertEqual(
            self.workspace.stairs[0].tread_edge_profile,
            STAIR_TREAD_EDGE_ROUNDED,
        )
        self.assertAlmostEqual(
            self.workspace.stairs[0].tread_edge_radius_meters,
            0.02,
        )
        self.assertEqual(
            self.workspace.stairs[0].starting_step,
            STAIR_STARTING_STEP_CURTAIL,
        )
        self.assertEqual(self.workspace.stairs[0].starting_step_edge_points, 6)
        self.assertEqual(
            self.workspace.stairs[0].stringer_placement,
            STAIR_STRINGER_NONE,
        )
        self.assertFalse(self.workspace.stairs[0].legacy_part_layout)
        self.assertIsNone(self.workspace._staged_stair)
        self.assertEqual(
            self.workspace.add_stairs_button.text(),
            "Apply changes to stair",
        )
        self.assertFalse(self.workspace.add_stairs_button.isEnabled())

        self.workspace._handle_canvas_undo_requested()

        self.assertEqual(self.workspace.stairs, [original])
        self.assertEqual(
            self.workspace.stair_tread_thickness_spinbox.value(),
            8.0,
        )
        self.assertEqual(
            self.workspace._read_stair_editor_parameters().nosing_placements,
            (STAIR_NOSING_FRONT,),
        )
        self.assertEqual(
            self.workspace.stair_starting_step_combo.currentData(),
            STAIR_STARTING_STEP_NONE,
        )

    def test_setting_restores_selected_stair_mesh_during_active_preview(
        self,
    ) -> None:
        self.workspace.stairs = [_make_editable_stair()]
        self.workspace._set_canvas_viewer_targets(())
        parts = tuple(self.workspace._canvas_stair_part_targets_by_id.values())
        mesh = trimesh.util.concatenate(part.mesh for part in parts)
        self.workspace.viewer.set_model(
            GeneratedModel(
                mesh=mesh,
                scene=trimesh.Scene(mesh),
                glb_bytes=b"",
                preview_stair_parts=list(parts),
            )
        )
        semantic_id = next(iter(self.workspace._canvas_stair_part_targets_by_id))
        self.workspace.viewer.select_canvas_stair_part_target(semantic_id)
        original = self.workspace.viewer._get_display_mesh()
        assert original is not None
        original_face_count = len(original.faces)
        self.assertTrue(
            self.workspace.settings_widget.get_settings().hide_stair_mesh_when_previewing
        )

        self.workspace.stair_tread_thickness_spinbox.setValue(12.0)
        QTest.qWait(80)
        _qt_application.processEvents()
        hidden = self.workspace.viewer._get_display_mesh()
        assert hidden is not None
        self.assertLess(len(hidden.faces), original_face_count)

        checkbox = (
            self.workspace.settings_widget.hide_stair_mesh_when_previewing_checkbox
        )
        checkbox.setChecked(False)
        shown = self.workspace.viewer._get_display_mesh()
        assert shown is not None
        self.assertEqual(len(shown.faces), original_face_count)
        checkbox.setChecked(True)
        hidden_again = self.workspace.viewer._get_display_mesh()
        assert hidden_again is not None
        self.assertLess(len(hidden_again.faces), original_face_count)

        self.workspace.viewer.clear_canvas_stair_preview()
        restored = self.workspace.viewer._get_display_mesh()
        assert restored is not None
        self.assertEqual(len(restored.faces), original_face_count)

    def test_reverting_stair_fields_before_preview_skips_geometry_build(self) -> None:
        original = _make_editable_stair(tread_thickness_meters=0.08)
        self.workspace.stairs = [original]
        self.workspace._set_canvas_viewer_targets(())
        semantic_id = next(iter(self.workspace._canvas_stair_part_targets_by_id))
        self.workspace.viewer.select_canvas_stair_part_target(semantic_id)

        self.workspace.stair_tread_thickness_spinbox.setValue(12.0)
        self.workspace.stair_tread_thickness_spinbox.setValue(8.0)
        with patch("housemaker.main.build_canvas_stair_part_targets") as build:
            QTest.qWait(80)
            _qt_application.processEvents()

        build.assert_not_called()
        self.assertIsNone(self.workspace._staged_stair)
        self.assertIsNone(
            self.workspace.viewer._canvas_stair_preview_stair_index
        )

    def test_reverting_starting_step_keeps_legacy_stair_layout(self) -> None:
        original = _make_editable_stair(
            stringer_placement=STAIR_STRINGER_NONE,
            legacy_part_layout=True,
        )
        self.assertTrue(original.uses_legacy_part_layout)
        self.workspace.stairs = [original]
        self.workspace._set_canvas_viewer_targets(())
        semantic_id = next(iter(self.workspace._canvas_stair_part_targets_by_id))
        self.workspace.viewer.select_canvas_stair_part_target(semantic_id)

        self.workspace.stair_starting_step_combo.setCurrentIndex(
            self.workspace.stair_starting_step_combo.findData(
                STAIR_STARTING_STEP_BULLNOSE
            )
        )
        self.workspace.stair_starting_step_combo.setCurrentIndex(
            self.workspace.stair_starting_step_combo.findData(
                STAIR_STARTING_STEP_NONE
            )
        )
        with patch("housemaker.main.build_canvas_stair_part_targets") as build:
            QTest.qWait(80)
            _qt_application.processEvents()

        build.assert_not_called()
        self.assertIsNone(self.workspace._staged_stair)
        self.assertTrue(self.workspace.stairs[0].uses_legacy_part_layout)

    def test_level_height_change_refreshes_selected_stair_measurements(self) -> None:
        stair = _make_editable_stair(target_rise_meters=0.25)
        self.workspace.stairs = [stair]
        self.workspace._set_canvas_viewer_targets(())
        semantic_id = next(iter(self.workspace._canvas_stair_part_targets_by_id))
        self.workspace.viewer.select_canvas_stair_part_target(semantic_id)
        step_count_before = self.workspace.stair_calculated_step_count_label.text()
        actual_rise_before = self.workspace.stair_actual_rise_label.text()

        self.workspace._handle_height_level_changed(
            self.workspace.current_level.height_meters + 1.0
        )

        self.assertNotEqual(
            self.workspace.stair_calculated_step_count_label.text(),
            step_count_before,
        )
        self.assertNotEqual(
            self.workspace.stair_actual_rise_label.text(),
            actual_rise_before,
        )

    def test_level_height_change_rebuilds_a_staged_stair_preview(self) -> None:
        stair = _make_editable_stair(tread_overhang_meters=0.03)
        self.workspace.stairs = [stair]
        self.workspace._set_canvas_viewer_targets(())
        semantic_id = next(iter(self.workspace._canvas_stair_part_targets_by_id))
        self.workspace.viewer.select_canvas_stair_part_target(semantic_id)
        self.workspace.stair_nosing_overhang_spinbox.setValue(5.0)
        QTest.qWait(160)
        _qt_application.processEvents()
        preview_before = tuple(
            part.mesh.bounds.copy()
            for part in self.workspace.viewer._canvas_stair_preview_parts
        )

        self.workspace._handle_height_level_changed(
            self.workspace.current_level.height_meters + 1.0
        )

        preview_after = tuple(
            part.mesh.bounds.copy()
            for part in self.workspace.viewer._canvas_stair_preview_parts
        )
        self.assertTrue(preview_before)
        self.assertEqual(len(preview_after), len(preview_before))
        self.assertTrue(
            any(
                not (before == after).all()
                for before, after in zip(preview_before, preview_after)
            )
        )

    def test_escape_discards_a_stair_change_before_preview_debounce(self) -> None:
        original = _make_editable_stair(tread_overhang_meters=0.03)
        self.workspace.stairs = [original]
        self.workspace._set_canvas_viewer_targets(())
        semantic_id = next(iter(self.workspace._canvas_stair_part_targets_by_id))
        self.workspace.viewer.select_canvas_stair_part_target(semantic_id)

        self.workspace.stair_nosing_overhang_spinbox.setValue(8.0)
        self.assertTrue(self.workspace._stair_preview_update_timer.isActive())

        self.workspace.viewer.view.escape_requested.emit()

        self.assertFalse(self.workspace._stair_preview_update_timer.isActive())
        self.assertIsNone(self.workspace._pending_stair_parameters)
        self.assertEqual(self.workspace.stairs, [original])
        self.assertEqual(
            self.workspace.stair_nosing_overhang_spinbox.value(),
            3.0,
        )

    def test_project_load_preserves_stair_texture_targets_and_selection(self) -> None:
        stair = _make_editable_stair(
            tread_thickness_meters=0.13,
            tread_edge_profile=STAIR_TREAD_EDGE_ROUNDED,
            starting_step=STAIR_STARTING_STEP_BULLNOSE,
            stringer_placement=STAIR_STRINGER_LEFT,
        )
        semantic_id = next(
            part.semantic_id
            for part in build_canvas_stair_part_targets(
                self.workspace.levels,
                (stair,),
            )
            if part.part_kind == STAIR_PART_SUPPORT
        )
        assignment = SurfaceTextureAssignment(
            assignment_id="stair-support",
            surface_type="wall",
            surface_ids=(semantic_id,),
            provider="meshy",
            asset_path="missing.png",
        )

        self.workspace._apply_project_state(
            levels=self.workspace.levels,
            current_level_index=GROUND_LEVEL_INDEX,
            stairs=[stair],
            surface_texture_generation=SurfaceTextureData(
                assignments=[assignment],
                selected_surface_type="wall",
                selected_surface_ids=(semantic_id,),
            ),
        )

        restored = self.workspace.surface_texture_generation.get_assignment(
            assignment.assignment_id
        )
        assert restored is not None
        self.assertEqual(restored.surface_ids, (semantic_id,))
        self.assertEqual(
            self.workspace._desired_canvas_stair_part_ids,
            (semantic_id,),
        )
        self.assertEqual(self.workspace._desired_canvas_surface_ids, ())
        self.assertEqual(self.workspace._editing_stair_index, 0)
        self.assertAlmostEqual(
            self.workspace.stair_tread_thickness_spinbox.value(),
            13.0,
        )
        self.assertEqual(
            self.workspace.stair_tread_edge_combo.currentData(),
            STAIR_TREAD_EDGE_ROUNDED,
        )
        self.assertEqual(
            self.workspace.stair_starting_step_combo.currentData(),
            STAIR_STARTING_STEP_BULLNOSE,
        )
        self.assertEqual(
            self.workspace.stair_stringer_placement_combo.currentData(),
            STAIR_STRINGER_LEFT,
        )

    def test_project_load_cancels_an_old_pending_stair_preview(self) -> None:
        old_stair = _make_editable_stair(tread_thickness_meters=0.08)
        self.workspace.stairs = [old_stair]
        self.workspace._set_canvas_viewer_targets(())
        old_target_id = next(
            semantic_id
            for semantic_id, target in (
                self.workspace._canvas_stair_part_targets_by_id.items()
            )
            if target.part_kind == STAIR_PART_TREADS
        )
        self.workspace.viewer.select_canvas_stair_part_target(old_target_id)
        self.workspace.stair_tread_thickness_spinbox.setValue(12.0)
        self.assertTrue(self.workspace._stair_preview_update_timer.isActive())

        loaded_stair = _make_editable_stair(tread_thickness_meters=0.09)
        loaded_target_id = next(
            part.semantic_id
            for part in build_canvas_stair_part_targets(
                self.workspace.levels,
                (loaded_stair,),
            )
            if part.part_kind == STAIR_PART_TREADS
        )
        self.workspace._apply_project_state(
            levels=self.workspace.levels,
            current_level_index=GROUND_LEVEL_INDEX,
            stairs=[loaded_stair],
            surface_texture_generation=SurfaceTextureData(
                selected_surface_type="floor",
                selected_surface_ids=(loaded_target_id,),
            ),
        )
        QTest.qWait(160)
        _qt_application.processEvents()

        self.assertFalse(self.workspace._stair_preview_update_timer.isActive())
        self.assertIsNone(self.workspace._pending_stair_parameters)
        self.assertIsNone(self.workspace._staged_stair)
        self.assertIsNone(
            self.workspace.viewer._canvas_stair_preview_stair_index
        )
        self.assertEqual(self.workspace.stairs, [loaded_stair])
        self.assertAlmostEqual(
            self.workspace.stair_tread_thickness_spinbox.value(),
            9.0,
        )

    def test_stair_edit_undo_restores_removed_part_texture_target(self) -> None:
        stair = _make_editable_stair()
        self.workspace.stairs = [stair]
        self.workspace._set_canvas_viewer_targets(())
        support_id = next(
            semantic_id
            for semantic_id, part in (
                self.workspace._canvas_stair_part_targets_by_id.items()
            )
            if part.part_kind == STAIR_PART_SUPPORT
        )
        assignment = SurfaceTextureAssignment(
            assignment_id="stair-support",
            surface_type="wall",
            surface_ids=(support_id,),
            provider="meshy",
            asset_path="missing.png",
        )
        self.workspace.surface_texture_generation.set_data(
            SurfaceTextureData(assignments=[assignment])
        )
        self.workspace.viewer.select_canvas_stair_part_target(support_id)

        self.workspace.stair_type_combo.setCurrentIndex(
            self.workspace.stair_type_combo.findData(STAIR_TYPE_FLOATING)
        )
        QTest.qWait(160)
        _qt_application.processEvents()
        self.workspace.add_stairs_button.click()

        edited_assignment = (
            self.workspace.surface_texture_generation.get_assignment(
                assignment.assignment_id
            )
        )
        assert edited_assignment is not None
        self.assertEqual(edited_assignment.surface_ids, ())
        self.assertTrue(self.workspace._desired_canvas_stair_part_ids)
        selected_part = self.workspace._canvas_stair_part_targets_by_id[
            self.workspace._desired_canvas_stair_part_ids[-1]
        ]
        self.assertEqual(selected_part.part_kind, "treads")

        self.workspace._handle_canvas_undo_requested()

        restored_assignment = (
            self.workspace.surface_texture_generation.get_assignment(
                assignment.assignment_id
            )
        )
        assert restored_assignment is not None
        self.assertEqual(restored_assignment.surface_ids, (support_id,))
        self.assertEqual(self.workspace.stairs, [stair])

    def test_invalid_stair_does_not_drop_other_stair_texture_targets(self) -> None:
        broken_stair = _make_editable_stair(stair_id="1" * 32)
        valid_stair = _make_editable_stair(
            stair_id="2" * 32,
            start_a_x=60.0,
            start_b_x=90.0,
            end_a_x=60.0,
            end_b_x=90.0,
        )
        self.workspace.stairs = [broken_stair, valid_stair]
        self.workspace._sync_canvas_stair_semantic_targets(self.workspace.levels)
        valid_target_id = next(
            semantic_id
            for semantic_id, target in (
                self.workspace._canvas_stair_part_targets_by_id.items()
            )
            if target.stair_id == valid_stair.stair_id
        )
        assignment = SurfaceTextureAssignment(
            assignment_id="valid-stair-texture",
            surface_type="floor",
            surface_ids=(valid_target_id,),
            provider="meshy",
            asset_path="missing.png",
        )
        self.workspace.surface_texture_generation.set_data(
            SurfaceTextureData(assignments=[assignment])
        )

        def _build_one_stair(levels, stairs, **kwargs):
            if stairs[0].stair_id == broken_stair.stair_id:
                raise ValueError("Broken bound stair route.")
            return build_canvas_stair_part_targets(levels, stairs, **kwargs)

        with patch(
            "housemaker.main.build_canvas_stair_part_targets",
            side_effect=_build_one_stair,
        ):
            self.workspace._reconcile_surface_assignments_with_scene()

        restored_assignment = (
            self.workspace.surface_texture_generation.get_assignment(
                assignment.assignment_id
            )
        )
        assert restored_assignment is not None
        self.assertEqual(restored_assignment.surface_ids, (valid_target_id,))
        self.assertTrue(
            all(
                target.stair_id == valid_stair.stair_id
                for target in self.workspace._canvas_stair_part_targets_by_id.values()
            )
        )

    def test_selected_canvas_stair_can_be_deleted_without_a_list(self) -> None:
        stair = StairData(
            start_level_index=GROUND_LEVEL_INDEX,
            start_a_x=20.0,
            start_a_y=30.0,
            start_b_x=50.0,
            start_b_y=30.0,
            end_level_index=GROUND_LEVEL_INDEX + 1,
            end_a_x=20.0,
            end_a_y=70.0,
            end_b_x=50.0,
            end_b_y=70.0,
        )
        self.workspace.stairs = [stair]
        _make_current_canvas_clickable(self.workspace)

        QTest.mouseClick(
            self.workspace.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.AltModifier,
            pos=_image_position(self.workspace, 20.0, 30.0),
        )
        self.assertEqual(self.workspace.canvas.selected_stair_index, 0)
        self.assertFalse(hasattr(self.workspace, "stairs_list"))
        self.assertFalse(
            hasattr(self.workspace, "delete_selected_stair_button")
        )

        QTest.keyClick(self.workspace.canvas, Qt.Key.Key_Delete)
        _qt_application.processEvents()

        self.assertEqual(self.workspace.stairs, [])
        self.assertEqual(self.workspace.canvas.stairs, [])
        self.assertEqual(
            self.workspace.stair_status_label.text(),
            "Stair deleted.",
        )

    def test_curve_guides_remain_draft_until_confirmed(self) -> None:
        _make_current_canvas_clickable(self.workspace)
        with patch("housemaker.main.QMessageBox.information"):
            self.workspace.add_stairs_button.click()

        for point in ((20.0, 30.0), (40.0, 30.0)):
            QTest.mouseClick(
                self.workspace.canvas,
                Qt.MouseButton.LeftButton,
                pos=_image_position(self.workspace, *point),
            )

        destination_level_index = GROUND_LEVEL_INDEX + 1
        self.workspace._handle_level_selection_changed(destination_level_index)
        _make_current_canvas_clickable(self.workspace)
        for point in ((70.0, 70.0), (90.0, 70.0)):
            QTest.mouseClick(
                self.workspace.canvas,
                Qt.MouseButton.LeftButton,
                pos=_image_position(self.workspace, *point),
            )

        for point in (
            (40.0, 65.0),
            (60.0, 65.0),
            (50.0, 52.0),
            (70.0, 52.0),
        ):
            QTest.mouseClick(
                self.workspace.canvas,
                Qt.MouseButton.LeftButton,
                pos=_image_position(self.workspace, *point),
            )
        _qt_application.processEvents()

        draft = self.workspace.canvas.get_stair_placement_draft()
        self.assertIsNotNone(draft)
        self.assertEqual(
            len(draft.intermediate_sections),  # type: ignore[union-attr]
            2,
        )
        self.assertEqual(self.workspace.stairs, [])
        self.assertEqual(
            self.workspace.add_stairs_button.text(),
            "Confirm stairs",
        )

        self.workspace.add_stairs_button.click()
        _qt_application.processEvents()

        self.assertEqual(len(self.workspace.stairs), 1)
        self.assertEqual(
            len(self.workspace.stairs[0].intermediate_sections),
            2,
        )
        self.assertEqual(
            [
                section.to_dict()
                for section in self.workspace.stairs[0].intermediate_sections
            ],
            [
                {
                    "level_index": section.level_index,
                    "a_x": section.a_x,
                    "a_y": section.a_y,
                    "b_x": section.b_x,
                    "b_y": section.b_y,
                    "a_vertex_id": section.a_vertex_id,
                    "b_vertex_id": section.b_vertex_id,
                }
                for section in draft.intermediate_sections  # type: ignore[union-attr]
            ],
        )
        self.assertEqual(self.workspace.add_stairs_button.text(), "Add stairs")

    def test_completed_stair_refits_the_3d_preview(self) -> None:
        placement = type(
            "Placement",
            (),
            {
                "start_level_index": GROUND_LEVEL_INDEX,
                "start_a_x": 20.0,
                "start_a_y": 35.0,
                "start_b_x": 45.0,
                "start_b_y": 35.0,
                "end_level_index": GROUND_LEVEL_INDEX + 1,
                "end_a_x": 50.0,
                "end_a_y": 58.0,
                "end_b_x": 90.0,
                "end_b_y": 58.0,
                "style": STAIR_STYLE_FLOATING,
            },
        )()

        with patch.object(
            self.workspace,
            "_schedule_viewer_preview_refresh",
        ) as refresh_mock:
            self.workspace._handle_stair_placement_completed(placement)

        self.assertEqual(len(self.workspace.stairs), 1)
        refresh_mock.assert_called_once_with(preserve_camera=False)

    def test_loading_a_project_cancels_a_pending_stair_placement(self) -> None:
        _add_wall_segment(self.workspace, (20.0, 35.0), (45.0, 35.0))
        _make_current_canvas_clickable(self.workspace)
        self.workspace.canvas.start_stair_placement()
        QTest.mouseClick(
            self.workspace.canvas,
            Qt.MouseButton.LeftButton,
            pos=_image_position(self.workspace, 20.0, 35.0),
        )
        QTest.mouseClick(
            self.workspace.canvas,
            Qt.MouseButton.LeftButton,
            pos=_image_position(self.workspace, 45.0, 35.0),
        )
        self.assertIsNotNone(self.workspace.canvas.get_pending_stair_placement())

        self.workspace._apply_project_state(
            levels=self.workspace.levels,
            current_level_index=GROUND_LEVEL_INDEX,
        )

        self.assertFalse(self.workspace.canvas.is_stair_placement_active())
        self.assertIsNone(self.workspace.canvas.get_pending_stair_placement())

    def test_switching_levels_mid_opening_directs_user_back_to_its_level(
        self,
    ) -> None:
        _make_current_canvas_clickable(self.workspace)
        with patch("housemaker.main.QMessageBox.information"):
            self.workspace.add_stairs_button.click()
        QTest.mouseClick(
            self.workspace.canvas,
            Qt.MouseButton.LeftButton,
            pos=_image_position(self.workspace, 20.0, 35.0),
        )

        self.workspace._handle_level_selection_changed(GROUND_LEVEL_INDEX + 1)

        status_text = self.workspace.stair_status_label.text()
        self.assertIn("Return to", status_text)
        self.assertIn("Ground", status_text)
        self.assertIsNotNone(self.workspace.canvas.get_pending_stair_point())

    def test_zero_run_stair_is_rejected_before_it_can_break_the_3d_preview(
        self,
    ) -> None:
        _add_wall_segment(self.workspace, (35.0, 45.0), (65.0, 45.0))
        _make_current_canvas_clickable(self.workspace)
        with patch("housemaker.main.QMessageBox.information"):
            self.workspace.add_stairs_button.click()
        start_position = _image_position(self.workspace, 35.0, 45.0)
        QTest.mouseClick(
            self.workspace.canvas,
            Qt.MouseButton.LeftButton,
            pos=start_position,
        )
        QTest.mouseClick(
            self.workspace.canvas,
            Qt.MouseButton.LeftButton,
            pos=_image_position(self.workspace, 65.0, 45.0),
        )

        self.workspace._handle_level_selection_changed(GROUND_LEVEL_INDEX + 1)
        _add_wall_segment(self.workspace, (35.0, 45.0), (65.0, 45.0))
        _make_current_canvas_clickable(self.workspace)
        QTest.mouseClick(
            self.workspace.canvas,
            Qt.MouseButton.LeftButton,
            pos=_image_position(self.workspace, 35.0, 45.0),
        )
        QTest.mouseClick(
            self.workspace.canvas,
            Qt.MouseButton.LeftButton,
            pos=_image_position(self.workspace, 65.0, 45.0),
        )
        _qt_application.processEvents()

        self.assertEqual(self.workspace.stairs, [])
        self.assertEqual(
            self.workspace.add_stairs_button.text(),
            "Confirm stairs",
        )
        self.workspace.add_stairs_button.click()
        _qt_application.processEvents()

        self.assertEqual(self.workspace.stairs, [])
        self.assertIn(
            "separated horizontally",
            self.workspace.stair_status_label.text(),
        )
        self.assertTrue(self.workspace.canvas.is_stair_placement_active())
        self.assertTrue(
            self.workspace.canvas.is_stair_ready_for_confirmation()
        )
        self.assertEqual(
            self.workspace.add_stairs_button.text(),
            "Confirm stairs",
        )

    def test_add_stairs_switches_from_3d_scene_to_canvas_workspace(self) -> None:
        _make_current_canvas_clickable(self.workspace)
        self.workspace.workspace_tabs.setCurrentWidget(
            self.workspace.scene_3d_workspace
        )
        self.assertIs(
            self.workspace.workspace_tabs.currentWidget(),
            self.workspace.scene_3d_workspace,
        )

        with patch("housemaker.main.QMessageBox.information"):
            self.workspace.add_stairs_button.click()

        self.assertIs(
            self.workspace.workspace_tabs.currentWidget(),
            self.workspace.canvas_viewer_workspace,
        )
        self.assertTrue(self.workspace.canvas.is_stair_placement_active())


if __name__ == "__main__":
    unittest.main()
