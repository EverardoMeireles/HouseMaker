# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import unittest
from dataclasses import replace
from unittest.mock import patch

import pyqtgraph.opengl as gl
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.door_geometry import (
    assemble_door_model,
    build_default_door_slot_models,
    build_door_body_model,
)
from housemaker.door_state import (
    DOOR_SLOT_BACK_BODY,
    DOOR_SLOT_BACK_KNOB,
    DOOR_SLOT_BODY,
    DOOR_SLOT_KNOB,
    DoorDefinition,
    DoorLibraryData,
    DoorPlacement,
    DoorSlotData,
)
from housemaker.doors_workspace import DoorsWorkspace

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Test fixtures ###
def _door(
    door_id: str = "door-1",
    *,
    name: str = "New door 1",
    shape: str = "rectangular",
    arch_amount: float = 0.0,
) -> DoorDefinition:
    return DoorDefinition(
        door_id=door_id,
        name=name,
        source_doorway_id=f"doorway-{door_id}",
        width_meters=0.9,
        height_meters=2.1,
        thickness_meters=0.04,
        shape=shape,
        arch_amount=arch_amount,
        slots=(
            DoorSlotData(
                slot_id=DOOR_SLOT_BODY,
                source_object_id=f"{door_id}-body",
                joined=True,
            ),
            DoorSlotData(
                slot_id=DOOR_SLOT_KNOB,
                source_object_id=f"{door_id}-knob",
            ),
        ),
    )


def _complete_door_model(door: DoorDefinition):
    """Build generated preview meshes for every editable door slot."""

    slot_models = build_default_door_slot_models(door)
    for slot in door.slots:
        if slot.slot_id not in {DOOR_SLOT_KNOB, DOOR_SLOT_BACK_KNOB}:
            continue
        slot_models[slot.source_object_id] = build_door_body_model(
            0.08,
            0.04,
            0.12,
            shape="rectangular",
            arch_amount=0.0,
            geometry_name=f"generated_{slot.slot_id}_fixture",
        )
    return assemble_door_model(
        door,
        slot_models,
        include_unjoined=True,
    )


# ### Doors workspace tests ###
class DoorsWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = DoorsWorkspace()

    def tearDown(self) -> None:
        self.workspace.close()
        self.workspace.deleteLater()
        _qt_application.processEvents()

    # ### Tree presentation ###
    def test_tree_shows_each_door_and_the_two_ordered_slots(self) -> None:
        door = _door(shape="arch", arch_amount=0.35)

        self.workspace.set_data(DoorLibraryData(doors=[door]))

        self.assertEqual(self.workspace.door_tree.columnCount(), 1)
        self.assertEqual(
            self.workspace.door_tree.headerItem().text(0),
            "Door / slot",
        )
        self.assertEqual(self.workspace.door_tree.topLevelItemCount(), 1)
        door_item = self.workspace.door_tree.topLevelItem(0)
        self.assertEqual(door_item.text(0), "New door 1")
        self.assertEqual(door_item.childCount(), 2)
        self.assertEqual(
            [door_item.child(index).text(0) for index in range(2)],
            ["Body", "Door knob/handle"],
        )
        self.assertEqual(self.workspace.profile_shape_label.text(), "Arched (35%)")
        self.assertIn("2.100 m H", self.workspace.profile_dimensions_label.text())

    def test_double_sided_door_groups_front_and_back_slots(self) -> None:
        door = _door().with_double_sided(True)
        requests: list[tuple[str, str]] = []
        self.workspace.slot_edit_requested.connect(
            lambda door_id, slot_id: requests.append((door_id, slot_id))
        )

        self.workspace.set_data(DoorLibraryData(doors=[door]))

        door_item = self.workspace.door_tree.topLevelItem(0)
        self.assertEqual(door_item.childCount(), 2)
        front_item = door_item.child(0)
        back_item = door_item.child(1)
        self.assertEqual(front_item.text(0), "Front")
        self.assertEqual(back_item.text(0), "Back")
        self.assertTrue(front_item.isExpanded())
        self.assertTrue(back_item.isExpanded())
        self.assertEqual(
            [front_item.child(index).text(0) for index in range(2)],
            ["Body", "Door knob/handle"],
        )
        self.assertEqual(
            [back_item.child(index).text(0) for index in range(2)],
            ["Body", "Door knob/handle"],
        )
        self.assertTrue(self.workspace.make_double_sided_checkbox.isChecked())
        self.assertTrue(
            self.workspace.select_door_slot(
                door.door_id,
                DOOR_SLOT_BACK_KNOB,
            )
        )
        self.assertEqual(
            self.workspace.selected_slot_id(),
            DOOR_SLOT_BACK_KNOB,
        )
        self.workspace.door_tree.itemDoubleClicked.emit(back_item.child(0), 0)
        self.assertEqual(requests, [(door.door_id, DOOR_SLOT_BACK_BODY)])

    def test_make_double_sided_checkbox_requests_a_model_transition(self) -> None:
        door = _door()
        requests: list[tuple[str, bool]] = []
        self.workspace.make_double_sided_changed.connect(
            lambda door_id, checked: requests.append((door_id, checked))
        )
        self.workspace.set_data(DoorLibraryData(doors=[door]))

        self.workspace.make_double_sided_checkbox.click()

        self.assertEqual(requests, [(door.door_id, True)])

        updated = door.with_double_sided(True)
        self.workspace.upsert_door(updated)
        self.workspace.make_double_sided_checkbox.click()

        self.assertEqual(
            requests,
            [(door.door_id, True), (door.door_id, False)],
        )

    def test_generate_displacement_checkbox_tracks_door_and_requests_change(
        self,
    ) -> None:
        door = _door()
        requests: list[tuple[str, bool]] = []
        self.workspace.generate_displacement_changed.connect(
            lambda door_id, checked: requests.append((door_id, checked))
        )
        self.workspace.set_data(DoorLibraryData(doors=[door]))

        self.assertFalse(
            self.workspace.generate_displacement_checkbox.isChecked()
        )
        self.workspace.generate_displacement_checkbox.click()

        self.assertEqual(requests, [(door.door_id, True)])

        displaced = replace(door, generate_displacement=True)
        self.workspace.upsert_door(displaced)

        self.assertTrue(
            self.workspace.generate_displacement_checkbox.isChecked()
        )
        self.workspace.generate_displacement_checkbox.click()
        self.assertEqual(
            requests,
            [(door.door_id, True), (door.door_id, False)],
        )

    def test_door_option_checkboxes_are_adjacent(self) -> None:
        make_double_sided = self.workspace.make_double_sided_checkbox
        generate_displacement = self.workspace.generate_displacement_checkbox
        option_layout = make_double_sided.parentWidget().layout()

        self.assertIs(
            make_double_sided.parentWidget(),
            generate_displacement.parentWidget(),
        )
        assert option_layout is not None
        self.assertEqual(option_layout.indexOf(make_double_sided), 0)
        self.assertEqual(option_layout.indexOf(generate_displacement), 1)

    # ### Selection synchronization ###
    def test_slot_selection_emits_ids_and_survives_door_update(self) -> None:
        door = _door()
        self.workspace.set_data(DoorLibraryData(doors=[door]))
        selections: list[tuple[object, object]] = []
        self.workspace.selection_changed.connect(
            lambda door_id, slot_id: selections.append((door_id, slot_id))
        )

        self.assertTrue(
            self.workspace.select_door_slot(door.door_id, DOOR_SLOT_KNOB)
        )
        self.assertEqual(selections[-1], (door.door_id, DOOR_SLOT_KNOB))

        self.workspace.upsert_door(replace(door, name="Front door"))

        self.assertEqual(self.workspace.selected_door_id(), door.door_id)
        self.assertEqual(self.workspace.selected_slot_id(), DOOR_SLOT_KNOB)
        self.assertEqual(self.workspace.profile_name_label.text(), "Front door")

    def test_select_body_can_be_requested_when_inserting_a_door(self) -> None:
        door = _door()

        self.workspace.upsert_door(door, select_body=True)

        self.assertEqual(self.workspace.selected_door_id(), door.door_id)
        self.assertEqual(self.workspace.selected_slot_id(), DOOR_SLOT_BODY)

    def test_side_duplication_controls_are_not_part_of_the_doors_tab(self) -> None:
        self.assertFalse(hasattr(self.workspace, "side_duplication_checkbox"))
        self.assertFalse(hasattr(self.workspace, "side_duplication_front_radio"))
        self.assertFalse(hasattr(self.workspace, "side_duplication_back_radio"))

    def test_double_click_requests_editing_only_for_a_component_slot(self) -> None:
        door = _door()
        self.workspace.set_data(DoorLibraryData(doors=[door]))
        requests: list[tuple[str, str]] = []
        self.workspace.slot_edit_requested.connect(
            lambda door_id, slot_id: requests.append((door_id, slot_id))
        )
        door_item = self.workspace.door_tree.topLevelItem(0)
        knob_item = door_item.child(1)

        self.workspace.door_tree.itemDoubleClicked.emit(door_item, 0)
        self.assertEqual(requests, [])

        self.workspace.door_tree.itemDoubleClicked.emit(knob_item, 0)

        self.assertEqual(requests, [(door.door_id, DOOR_SLOT_KNOB)])

    # ### Slot confirmation ###
    def test_confirm_is_only_enabled_for_pending_non_body_slot(self) -> None:
        door = _door()
        self.workspace.set_data(DoorLibraryData(doors=[door]))
        requests: list[tuple[str, str]] = []
        self.workspace.slot_confirmation_requested.connect(
            lambda door_id, slot_id: requests.append((door_id, slot_id))
        )

        self.workspace.select_door_slot(door.door_id, DOOR_SLOT_BODY)
        self.assertTrue(
            self.workspace.set_slot_position_pending(
                door.door_id,
                DOOR_SLOT_BODY,
            )
        )
        self.assertFalse(self.workspace.confirm_slot_position_button.isEnabled())

        self.workspace.select_door_slot(door.door_id, DOOR_SLOT_KNOB)
        self.assertFalse(self.workspace.confirm_slot_position_button.isEnabled())
        self.assertTrue(
            self.workspace.set_slot_position_pending(
                door.door_id,
                DOOR_SLOT_KNOB,
            )
        )
        self.assertTrue(self.workspace.confirm_slot_position_button.isEnabled())

        self.workspace.confirm_slot_position_button.click()

        self.assertEqual(requests, [(door.door_id, DOOR_SLOT_KNOB)])
        selected_item = self.workspace.door_tree.currentItem()
        assert selected_item is not None
        self.assertIn(
            "transform pending",
            self.workspace.selected_slot_label.text(),
        )
        self.assertEqual(
            self.workspace.confirm_slot_position_button.text(),
            "Confirm slot transform",
        )

    def test_replace_slot_updates_model_without_losing_selection(self) -> None:
        door = _door()
        self.workspace.set_data(DoorLibraryData(doors=[door]))
        self.workspace.select_door_slot(door.door_id, DOOR_SLOT_KNOB)
        knob = door.get_slot(DOOR_SLOT_KNOB)
        assert knob is not None

        self.assertTrue(
            self.workspace.replace_slot(
                door.door_id,
                knob.with_transform(
                    position_meters=(0.25, 0.02, 0.95),
                    joined=True,
                ),
            )
        )

        self.assertEqual(self.workspace.selected_slot_id(), DOOR_SLOT_KNOB)
        selected_slot = self.workspace.selected_slot()
        assert selected_slot is not None
        self.assertEqual(selected_slot.position_meters, (0.25, 0.02, 0.95))
        self.assertTrue(selected_slot.joined)

    # ### Model ownership ###
    def test_data_is_detached_and_removing_a_door_prunes_placements(self) -> None:
        door = _door()
        placement = DoorPlacement(
            placement_id="placement-1",
            door_id=door.door_id,
            level_index=0,
            doorway_id="doorway-1",
        )
        incoming = DoorLibraryData(doors=[door], placements=[placement])
        self.workspace.set_data(incoming)
        incoming.doors.clear()

        self.assertEqual(len(self.workspace.doors()), 1)
        detached = self.workspace.data()
        detached.doors.clear()
        self.assertEqual(len(self.workspace.doors()), 1)

        self.assertTrue(self.workspace.remove_door(door.door_id))
        self.assertEqual(self.workspace.data().doors, [])
        self.assertEqual(self.workspace.data().placements, [])

    # ### Complete door preview ###
    def test_preview_viewer_accepts_and_clears_a_complete_door_model(self) -> None:
        door = _door()
        slot_models = build_default_door_slot_models(door)
        complete_model = assemble_door_model(
            door,
            {
                door.slots[0].source_object_id: slot_models[DOOR_SLOT_BODY]
            },
            include_unjoined=True,
        )

        self.workspace.set_preview_model(complete_model)

        self.assertEqual(
            self.workspace.preview_viewer.objectName(),
            "door_complete_preview_viewer",
        )
        self.assertIs(self.workspace.preview_viewer.model, complete_model)

        self.workspace.set_preview_model(None)

        self.assertIsNone(self.workspace.preview_viewer.model)

    def test_preview_owns_the_selected_generated_hardware_gizmo(self) -> None:
        door = _door()
        knob = door.get_slot(DOOR_SLOT_KNOB)
        assert knob is not None
        slot_models = build_default_door_slot_models(door)
        slot_models[knob.source_object_id] = build_door_body_model(
            0.08,
            0.12,
            0.04,
            shape="rectangular",
            arch_amount=0.0,
            geometry_name="generated_knob_fixture",
        )
        complete_model = assemble_door_model(
            door,
            slot_models,
            include_unjoined=True,
        )
        transforms: list[tuple[object, ...]] = []
        axis_scale_changes: list[tuple[object, ...]] = []
        self.workspace.slot_transform_changed.connect(
            lambda *values: transforms.append(values)
        )
        self.workspace.slot_axis_scales_changed.connect(
            lambda *values: axis_scale_changes.append(values)
        )

        self.workspace.set_data(DoorLibraryData(doors=[door]))
        self.workspace.select_door_slot(door.door_id, DOOR_SLOT_KNOB)
        self.workspace.set_preview_model(complete_model)

        self.assertEqual(
            self.workspace.preview_viewer.get_selected_placed_object_ids(),
            (knob.source_object_id,),
        )
        self.assertEqual(
            self.workspace.preview_viewer.get_placed_object_gizmo_mode(),
            "transform",
        )
        self.assertTrue(
            self.workspace.preview_viewer._toggle_placed_object_gizmo_mode()
        )
        self.assertEqual(
            self.workspace.preview_viewer.get_placed_object_gizmo_mode(),
            "scale",
        )
        self.assertFalse(
            self.workspace.preview_viewer.view._object_scale_wheel_steps_enabled
        )
        self.assertEqual(
            self.workspace.preview_viewer._placed_object_instance_gizmo_items,
            [],
        )
        self.workspace.preview_viewer.placed_object_transform_changed.emit(
            knob.source_object_id,
            (0.2, 0.0, 0.9),
            (0.0, 0.0, 15.0),
        )
        self.workspace.preview_viewer.placed_object_axis_scales_changed.emit(
            knob.source_object_id,
            (1.2, 0.8, 1.1),
        )
        self.assertEqual(
            transforms,
            [
                (
                    door.door_id,
                    DOOR_SLOT_KNOB,
                    (0.2, 0.0, 0.9),
                    (0.0, 0.0, 15.0),
                )
            ],
        )
        self.assertEqual(
            axis_scale_changes,
            [
                (
                    door.door_id,
                    DOOR_SLOT_KNOB,
                    (1.2, 0.8, 1.1),
                )
            ],
        )

    def test_preview_mesh_selection_selects_its_slot_before_gizmo_edits(
        self,
    ) -> None:
        door = _door()
        knob = door.get_slot(DOOR_SLOT_KNOB)
        assert knob is not None
        complete_model = _complete_door_model(door)

        self.workspace.set_data(DoorLibraryData(doors=[door]))
        self.workspace.select_door_slot(door.door_id, DOOR_SLOT_BODY)
        self.workspace.set_preview_model(complete_model)

        self.workspace.preview_viewer.placed_object_selection_changed.emit(
            knob.source_object_id
        )

        self.assertEqual(self.workspace.selected_slot_id(), DOOR_SLOT_KNOB)
        self.assertEqual(
            self.workspace.preview_viewer.get_selected_placed_object_ids(),
            (knob.source_object_id,),
        )
        self.assertEqual(
            self.workspace.preview_viewer.get_placed_object_gizmo_mode(),
            "transform",
        )
        self.assertTrue(
            self.workspace.preview_viewer._toggle_placed_object_gizmo_mode()
        )
        self.assertEqual(
            self.workspace.preview_viewer.get_placed_object_gizmo_mode(),
            "scale",
        )

    def test_clicking_hardware_twice_shows_scale_cube_handles(self) -> None:
        """Exercise the viewport click path used by the Doors tab."""

        door = _door()
        knob = door.get_slot(DOOR_SLOT_KNOB)
        assert knob is not None
        separated_knob = knob.with_transform(position_meters=(2.0, 0.0, 0.0))
        door = door.replace_slot(separated_knob)
        complete_model = _complete_door_model(door)
        viewer = self.workspace.preview_viewer

        self.workspace.set_data(DoorLibraryData(doors=[door]))
        self.workspace.set_preview_model(complete_model)
        self.workspace.resize(1000, 700)
        self.workspace.show()
        _qt_application.processEvents()

        hardware_ray = (
            (2.0, -5.0, 0.02),
            (0.0, 1.0, 0.0),
        )
        click_position = QPoint(160, 120)
        with (
            patch.object(
                viewer.view,
                "build_camera_ray",
                return_value=hardware_ray,
            ),
            patch.object(viewer.view, "pixelSize", return_value=0.01),
        ):
            QTest.mouseClick(
                viewer.view,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
                click_position,
            )
            _qt_application.processEvents()

            self.assertEqual(self.workspace.selected_slot_id(), DOOR_SLOT_KNOB)
            self.assertEqual(viewer.get_placed_object_gizmo_mode(), "transform")

            QTest.mouseClick(
                viewer.view,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
                click_position,
            )
            _qt_application.processEvents()

        self.assertEqual(viewer.get_placed_object_gizmo_mode(), "scale")
        self.assertEqual(len(viewer._transform_gizmo_items), 6)
        self.assertEqual(
            sum(
                isinstance(item, gl.GLMeshItem)
                for item in viewer._transform_gizmo_items
            ),
            3,
        )

    def test_double_clicking_a_preview_part_requests_generation_editing(
        self,
    ) -> None:
        door = _door()
        knob = door.get_slot(DOOR_SLOT_KNOB)
        assert knob is not None
        door = door.replace_slot(
            knob.with_transform(position_meters=(2.0, 0.0, 0.0))
        )
        knob = door.get_slot(DOOR_SLOT_KNOB)
        assert knob is not None
        viewer = self.workspace.preview_viewer
        requests: list[tuple[str, str]] = []
        self.workspace.slot_edit_requested.connect(
            lambda door_id, slot_id: requests.append((door_id, slot_id))
        )
        self.workspace.set_data(DoorLibraryData(doors=[door]))
        self.workspace.set_preview_model(_complete_door_model(door))
        self.workspace.resize(1000, 700)
        self.workspace.show()
        _qt_application.processEvents()

        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=((2.0, -5.0, 0.02), (0.0, 1.0, 0.0)),
        ):
            QTest.mouseDClick(
                viewer.view,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
                QPoint(160, 120),
            )
            _qt_application.processEvents()

        self.assertEqual(requests, [(door.door_id, DOOR_SLOT_KNOB)])
        self.assertEqual(self.workspace.selected_slot_id(), DOOR_SLOT_KNOB)

    def test_hardware_scale_gizmo_mode_survives_preview_reassembly(self) -> None:
        door = _door()
        knob = door.get_slot(DOOR_SLOT_KNOB)
        assert knob is not None
        complete_model = _complete_door_model(door)

        self.workspace.set_data(DoorLibraryData(doors=[door]))
        self.workspace.select_door_slot(door.door_id, DOOR_SLOT_KNOB)
        self.workspace.set_preview_model(complete_model)
        self.assertTrue(
            self.workspace.preview_viewer._toggle_placed_object_gizmo_mode()
        )

        self.workspace.set_preview_model(_complete_door_model(door))

        self.assertEqual(
            self.workspace.preview_viewer.get_selected_placed_object_ids(),
            (knob.source_object_id,),
        )
        self.assertEqual(
            self.workspace.preview_viewer.get_placed_object_gizmo_mode(),
            "scale",
        )

    def test_generated_body_uses_scale_only_gizmos(self) -> None:
        door = _door()
        body = door.get_slot(DOOR_SLOT_BODY)
        assert body is not None
        complete_model = assemble_door_model(
            door,
            {
                body.source_object_id: build_default_door_slot_models(door)[
                    DOOR_SLOT_BODY
                ]
            },
            include_unjoined=True,
        )
        axis_scale_changes: list[tuple[object, ...]] = []
        self.workspace.slot_axis_scales_changed.connect(
            lambda *values: axis_scale_changes.append(values)
        )

        self.workspace.set_data(DoorLibraryData(doors=[door]))
        self.workspace.select_door_slot(door.door_id, DOOR_SLOT_BODY)
        self.workspace.set_preview_model(complete_model)

        self.assertEqual(
            self.workspace.preview_viewer.get_selected_placed_object_ids(),
            (body.source_object_id,),
        )
        self.assertEqual(
            self.workspace.preview_viewer.get_placed_object_gizmo_mode(),
            "scale",
        )
        self.assertFalse(
            self.workspace.preview_viewer._toggle_placed_object_gizmo_mode()
        )
        self.workspace.preview_viewer.placed_object_axis_scales_changed.emit(
            body.source_object_id,
            (0.9, 1.1, 1.2),
        )

        self.assertEqual(
            axis_scale_changes,
            [(door.door_id, DOOR_SLOT_BODY, (0.9, 1.1, 1.2))],
        )

    def test_complete_door_preview_disables_mirror_fading(self) -> None:
        self.assertFalse(
            self.workspace.preview_viewer._symmetric_preview_fade_enabled
        )

    def test_wireframe_checkbox_controls_complete_door_preview(self) -> None:
        self.assertEqual(self.workspace.wireframe_checkbox.text(), "Wireframe")
        self.assertFalse(self.workspace.wireframe_checkbox.isChecked())
        self.assertFalse(
            self.workspace.preview_viewer.get_wireframe_enabled()
        )

        self.workspace.wireframe_checkbox.click()

        self.assertTrue(self.workspace.wireframe_checkbox.isChecked())
        self.assertTrue(
            self.workspace.preview_viewer.get_wireframe_enabled()
        )

        self.workspace.wireframe_checkbox.click()

        self.assertFalse(self.workspace.wireframe_checkbox.isChecked())
        self.assertFalse(
            self.workspace.preview_viewer.get_wireframe_enabled()
        )

    def test_ctrl_z_from_a_child_control_requests_doors_undo(self) -> None:
        requests: list[bool] = []
        self.workspace.undo_requested.connect(lambda: requests.append(True))
        self.workspace.show()
        self.workspace.door_tree.setFocus()
        _qt_application.processEvents()

        QTest.keyClick(
            self.workspace.door_tree,
            Qt.Key.Key_Z,
            Qt.KeyboardModifier.ControlModifier,
        )
        _qt_application.processEvents()

        self.assertEqual(requests, [True])

    def test_preview_undo_request_uses_the_same_doors_signal(self) -> None:
        requests: list[bool] = []
        self.workspace.undo_requested.connect(lambda: requests.append(True))

        self.workspace.preview_viewer.undo_requested.emit()

        self.assertEqual(requests, [True])


if __name__ == "__main__":
    unittest.main()
