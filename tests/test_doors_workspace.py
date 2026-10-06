# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import unittest
from dataclasses import replace

from PySide6.QtWidgets import QApplication

from housemaker.door_geometry import (
    assemble_door_model,
    build_default_door_slot_models,
)
from housemaker.door_state import (
    DOOR_SIDE_DUPLICATION_BACK,
    DOOR_SLOT_BODY,
    DOOR_SLOT_HINGES,
    DOOR_SLOT_KNOB,
    DoorDefinition,
    DoorLibraryData,
    DoorPlacement,
    DoorSideDuplication,
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
                slot_id=DOOR_SLOT_HINGES,
                source_object_id=f"{door_id}-hinges",
            ),
            DoorSlotData(
                slot_id=DOOR_SLOT_KNOB,
                source_object_id=f"{door_id}-knob",
            ),
        ),
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
    def test_tree_shows_each_door_and_the_three_ordered_slots(self) -> None:
        door = _door(shape="arch", arch_amount=0.35)

        self.workspace.set_data(DoorLibraryData(doors=[door]))

        self.assertEqual(self.workspace.door_tree.topLevelItemCount(), 1)
        door_item = self.workspace.door_tree.topLevelItem(0)
        self.assertEqual(door_item.text(0), "New door 1")
        self.assertEqual(door_item.childCount(), 3)
        self.assertEqual(
            [door_item.child(index).text(0) for index in range(3)],
            ["Body", "Hinges", "Door knob"],
        )
        self.assertIn("0.900 m W", door_item.child(0).text(1))
        self.assertIn("Arched (35%)", door_item.child(0).text(1))
        self.assertEqual(self.workspace.profile_shape_label.text(), "Arched (35%)")
        self.assertIn("2.100 m H", self.workspace.profile_dimensions_label.text())

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

    def test_side_duplication_controls_update_the_selected_door(self) -> None:
        door = _door()
        self.workspace.set_data(DoorLibraryData(doors=[door]))
        changed: list[DoorDefinition] = []
        self.workspace.door_definition_changed.connect(changed.append)

        self.workspace.side_duplication_checkbox.click()
        self.workspace.side_duplication_back_radio.click()

        updated = self.workspace.selected_door()
        assert updated is not None
        self.assertEqual(
            updated.side_duplication,
            DoorSideDuplication(kept_side=DOOR_SIDE_DUPLICATION_BACK),
        )
        self.assertEqual(changed[-1], updated)
        self.assertTrue(self.workspace.side_duplication_back_radio.isEnabled())

        self.workspace.side_duplication_checkbox.click()

        updated = self.workspace.selected_door()
        assert updated is not None
        self.assertIsNone(updated.side_duplication)
        self.assertFalse(self.workspace.side_duplication_front_radio.isEnabled())
        self.assertFalse(self.workspace.side_duplication_back_radio.isEnabled())

    def test_persisted_side_duplication_resyncs_and_locks_controls(self) -> None:
        door = _door()
        self.workspace.set_data(DoorLibraryData(doors=[door]))
        changed: list[DoorDefinition] = []
        self.workspace.door_definition_changed.connect(changed.append)

        synchronized = self.workspace.set_persisted_side_duplication(
            door.door_id,
            DoorSideDuplication(kept_side=DOOR_SIDE_DUPLICATION_BACK),
        )

        self.assertIsNotNone(synchronized)
        selected = self.workspace.selected_door()
        assert selected is not None
        self.assertEqual(
            selected.side_duplication,
            DoorSideDuplication(kept_side=DOOR_SIDE_DUPLICATION_BACK),
        )
        self.assertTrue(self.workspace.side_duplication_checkbox.isChecked())
        self.assertTrue(self.workspace.side_duplication_back_radio.isChecked())
        self.assertFalse(self.workspace.side_duplication_checkbox.isEnabled())
        self.assertFalse(self.workspace.side_duplication_front_radio.isEnabled())
        self.assertFalse(self.workspace.side_duplication_back_radio.isEnabled())
        self.assertIn(
            "already applied",
            self.workspace.side_duplication_checkbox.toolTip(),
        )

        self.workspace.side_duplication_checkbox.click()
        self.workspace.side_duplication_front_radio.click()

        self.assertEqual(self.workspace.selected_door(), selected)
        self.assertEqual(changed, [])

        self.workspace.set_persisted_side_duplication(door.door_id, None)

        self.assertTrue(self.workspace.side_duplication_checkbox.isEnabled())
        self.assertTrue(self.workspace.side_duplication_front_radio.isEnabled())
        self.assertTrue(self.workspace.side_duplication_back_radio.isEnabled())

    def test_double_click_requests_editing_only_for_a_component_slot(self) -> None:
        door = _door()
        self.workspace.set_data(DoorLibraryData(doors=[door]))
        requests: list[tuple[str, str]] = []
        self.workspace.slot_edit_requested.connect(
            lambda door_id, slot_id: requests.append((door_id, slot_id))
        )
        door_item = self.workspace.door_tree.topLevelItem(0)
        knob_item = door_item.child(2)

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

        self.workspace.select_door_slot(door.door_id, DOOR_SLOT_HINGES)
        self.assertFalse(self.workspace.confirm_slot_position_button.isEnabled())
        self.assertTrue(
            self.workspace.set_slot_position_pending(
                door.door_id,
                DOOR_SLOT_HINGES,
            )
        )
        self.assertTrue(self.workspace.confirm_slot_position_button.isEnabled())

        self.workspace.confirm_slot_position_button.click()

        self.assertEqual(requests, [(door.door_id, DOOR_SLOT_HINGES)])
        selected_item = self.workspace.door_tree.currentItem()
        assert selected_item is not None
        self.assertEqual(selected_item.text(1), "Position pending")

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
                slot.source_object_id: slot_models[slot.slot_id]
                for slot in door.slots
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


if __name__ == "__main__":
    unittest.main()
