# ### Imports ###
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from housemaker.door_state import (
    DOOR_LIBRARY_SCHEMA_VERSION,
    DOOR_SIDE_DUPLICATION_BACK,
    DOOR_SIDE_DUPLICATION_FRONT,
    DOOR_SLOT_BACK_BODY,
    DOOR_SLOT_BACK_KNOB,
    DOOR_SLOT_BODY,
    DOOR_SLOT_KNOB,
    DoorLibraryData,
    DoorPlacement,
    DoorSideDuplication,
    DoorSlotData,
    create_door_definition_for_doorway,
    door_fits_doorway,
    door_slot_component,
    door_slot_side,
    is_door_body_slot,
    is_door_knob_slot,
    next_door_name,
    set_door_double_sided,
)
from housemaker.models import DOORWAY_SHAPE_ARCH, DoorwayData, create_default_levels
from housemaker.project_io import load_project, save_project


# ### Door-library fixtures ###
def _doorway(*, doorway_id: str = "doorway-a") -> DoorwayData:
    return DoorwayData(
        doorway_id=doorway_id,
        center_x=100.0,
        center_y=200.0,
        width_meters=0.9,
        height_meters=2.1,
        depth_meters=0.2,
        shape=DOORWAY_SHAPE_ARCH,
        arch_amount=0.6,
    )


# ### Door state tests ###
class DoorStateTests(unittest.TestCase):
    def test_library_round_trip_preserves_slots_and_stable_placement_binding(
        self,
    ) -> None:
        door = create_door_definition_for_doorway(
            _doorway(),
            door_id="door-a",
            name="New door 1",
        )
        knob = door.get_slot(DOOR_SLOT_KNOB)
        assert knob is not None
        door = replace(
            door.replace_slot(
                knob.with_transform(
                    position_meters=(0.31, 0.0, 0.88),
                    rotation_degrees=(0.0, 0.0, 15.0),
                    axis_scales=(1.2, 0.8, 1.4),
                    joined=True,
                )
            ),
            side_duplication=DoorSideDuplication(
                kept_side=DOOR_SIDE_DUPLICATION_BACK
            ),
        )
        library = DoorLibraryData(
            doors=[door],
            placements=[
                DoorPlacement(
                    placement_id="placement-a",
                    door_id=door.door_id,
                    level_index=2,
                    doorway_id="doorway-destination",
                    mirrored_horizontally=True,
                )
            ],
        )

        restored = DoorLibraryData.from_dict(library.to_dict())

        self.assertEqual(restored, library)
        restored_knob = restored.doors[0].get_slot(DOOR_SLOT_KNOB)
        assert restored_knob is not None
        self.assertTrue(restored_knob.joined)
        self.assertEqual(restored_knob.position_meters, (0.31, 0.0, 0.88))
        self.assertEqual(restored_knob.axis_scales, (1.2, 0.8, 1.4))
        self.assertEqual(restored.placements[0].doorway_id, "doorway-destination")
        self.assertEqual(
            restored.doors[0].side_duplication,
            DoorSideDuplication(kept_side=DOOR_SIDE_DUPLICATION_BACK),
        )
        self.assertEqual(
            library.to_dict()["schema_version"],
            DOOR_LIBRARY_SCHEMA_VERSION,
        )

    def test_displacement_generation_round_trip_and_legacy_default(self) -> None:
        door = create_door_definition_for_doorway(
            _doorway(),
            door_id="door-a",
            name="New door 1",
        )
        displaced = replace(door, generate_displacement=True)

        restored = type(door).from_dict(displaced.to_dict())
        legacy_payload = door.to_dict()
        legacy_payload.pop("generate_displacement")
        restored_legacy = type(door).from_dict(legacy_payload)

        self.assertTrue(restored.generate_displacement)
        self.assertFalse(restored_legacy.generate_displacement)

    def test_displacement_generation_state_must_be_boolean(self) -> None:
        door = create_door_definition_for_doorway(
            _doorway(),
            door_id="door-a",
            name="New door 1",
        )

        with self.assertRaisesRegex(TypeError, "displacement-generation"):
            replace(door, generate_displacement=1)

    def test_legacy_door_defaults_to_no_side_duplication(self) -> None:
        door = create_door_definition_for_doorway(
            _doorway(),
            door_id="legacy-door",
            name="New door 1",
        )
        payload = door.to_dict()
        payload.pop("side_duplication")

        restored = type(door).from_dict(payload)

        self.assertIsNone(restored.side_duplication)
        self.assertFalse(restored.make_double_sided)

    def test_legacy_hinge_slot_is_removed_during_migration(self) -> None:
        door = create_door_definition_for_doorway(
            _doorway(),
            door_id="legacy-door",
            name="New door 1",
        )
        payload = door.to_dict()
        payload.pop("make_double_sided")
        payload["slots"].insert(
            1,
            {
                "slot_id": "hinges",
                "source_object_id": "door:legacy-door:slot:hinges",
                "position_meters": [-0.4, 0.0, 0.4],
                "rotation_degrees": [0.0, 0.0, 0.0],
                "joined": True,
            },
        )

        restored = type(door).from_dict(payload)

        self.assertEqual(
            tuple(slot.slot_id for slot in restored.slots),
            (DOOR_SLOT_BODY, DOOR_SLOT_KNOB),
        )
        self.assertFalse(restored.make_double_sided)
        with self.assertRaises(ValueError):
            DoorSlotData(
                slot_id="hinges",
                source_object_id="new-hinge",
            )

    def test_legacy_slot_defaults_to_unit_axis_scales(self) -> None:
        slot = DoorSlotData(
            slot_id=DOOR_SLOT_KNOB,
            source_object_id="legacy-knob",
        )
        payload = slot.to_dict()
        payload.pop("axis_scales")

        restored = DoorSlotData.from_dict(payload)

        self.assertEqual(restored.axis_scales, (1.0, 1.0, 1.0))

    def test_slot_axis_scales_must_be_finite_and_positive(self) -> None:
        for invalid_scales in (
            (0.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
            (float("inf"), 1.0, 1.0),
        ):
            with self.subTest(axis_scales=invalid_scales), self.assertRaises(
                ValueError
            ):
                DoorSlotData(
                    slot_id=DOOR_SLOT_KNOB,
                    source_object_id="invalid-knob",
                    axis_scales=invalid_scales,
                )

    def test_factory_creates_required_slots_and_incremental_names(self) -> None:
        first = create_door_definition_for_doorway(
            _doorway(),
            door_id="door-a",
            name="New door 1",
        )
        third = replace(first, door_id="door-c", name="New door 3")

        self.assertEqual(
            {slot.slot_id for slot in first.slots},
            {DOOR_SLOT_BODY, DOOR_SLOT_KNOB},
        )
        body = first.get_slot(DOOR_SLOT_BODY)
        assert body is not None
        self.assertTrue(body.joined)
        self.assertEqual(next_door_name((first, third)), "New door 4")
        self.assertTrue(door_fits_doorway(first, _doorway()))
        self.assertTrue(
            door_fits_doorway(
                first,
                replace(
                    _doorway(),
                    width_meters=1.2,
                    height_meters=2.8,
                ),
            )
        )
        self.assertFalse(
            door_fits_doorway(
                first,
                replace(_doorway(), shape="arch", arch_amount=0.5),
            )
        )

    def test_double_sided_toggle_preserves_front_slot_identities(self) -> None:
        original = create_door_definition_for_doorway(
            _doorway(),
            door_id="door-a",
            name="New door 1",
        )

        doubled = set_door_double_sided(original, True)

        self.assertTrue(doubled.make_double_sided)
        self.assertEqual(
            tuple(slot.slot_id for slot in doubled.slots),
            (
                DOOR_SLOT_BODY,
                DOOR_SLOT_KNOB,
                DOOR_SLOT_BACK_BODY,
                DOOR_SLOT_BACK_KNOB,
            ),
        )
        for slot_id in (DOOR_SLOT_BODY, DOOR_SLOT_KNOB):
            self.assertEqual(
                doubled.get_slot(slot_id),
                original.get_slot(slot_id),
            )
        back_body = doubled.get_slot(DOOR_SLOT_BACK_BODY)
        back_knob = doubled.get_slot(DOOR_SLOT_BACK_KNOB)
        assert back_body is not None
        assert back_knob is not None
        self.assertEqual(
            back_body.source_object_id,
            "door:door-a:slot:back_body",
        )
        self.assertEqual(
            back_knob.source_object_id,
            "door:door-a:slot:back_door_knob",
        )
        self.assertTrue(back_body.joined)
        self.assertFalse(back_knob.joined)
        self.assertEqual(doubled.with_double_sided(False), original)

    def test_double_sided_door_round_trip_and_slot_helpers(self) -> None:
        door = create_door_definition_for_doorway(
            _doorway(),
            door_id="door-a",
            name="New door 1",
        ).with_double_sided(True)

        restored = type(door).from_dict(door.to_dict())

        self.assertEqual(restored, door)
        self.assertTrue(restored.make_double_sided)
        self.assertTrue(is_door_body_slot(DOOR_SLOT_BODY))
        self.assertTrue(is_door_body_slot(DOOR_SLOT_BACK_BODY))
        self.assertTrue(is_door_knob_slot(DOOR_SLOT_KNOB))
        self.assertTrue(is_door_knob_slot(DOOR_SLOT_BACK_KNOB))
        self.assertEqual(
            door_slot_side(DOOR_SLOT_BODY),
            DOOR_SIDE_DUPLICATION_FRONT,
        )
        self.assertEqual(
            door_slot_side(DOOR_SLOT_BACK_KNOB),
            DOOR_SIDE_DUPLICATION_BACK,
        )
        self.assertEqual(
            door_slot_component(DOOR_SLOT_BACK_BODY),
            DOOR_SLOT_BODY,
        )
        self.assertEqual(
            door_slot_component(DOOR_SLOT_BACK_KNOB),
            DOOR_SLOT_KNOB,
        )

    def test_double_sided_and_side_duplication_are_mutually_exclusive(
        self,
    ) -> None:
        door = create_door_definition_for_doorway(
            _doorway(),
            door_id="door-a",
            name="New door 1",
        )
        duplicated = replace(
            door,
            side_duplication=DoorSideDuplication(),
        )

        with self.assertRaisesRegex(ValueError, "must be regenerated"):
            set_door_double_sided(duplicated, True)
        with self.assertRaisesRegex(ValueError, "cannot use side door"):
            replace(
                door.with_double_sided(True),
                side_duplication=DoorSideDuplication(),
            )

    def test_malformed_library_entries_are_skipped_without_dangling_placements(
        self,
    ) -> None:
        door = create_door_definition_for_doorway(
            _doorway(),
            door_id="door-a",
            name="New door 1",
        )
        payload = {
            "doors": [door.to_dict(), {"broken": True}],
            "placements": [
                DoorPlacement(
                    placement_id="valid-placement",
                    door_id="door-a",
                    level_index=2,
                    doorway_id="doorway-a",
                ).to_dict(),
                {
                    "placement_id": "dangling-placement",
                    "door_id": "missing-door",
                    "level_index": 2,
                    "doorway_id": "doorway-a",
                },
            ],
        }

        restored = DoorLibraryData.from_dict(payload)

        self.assertEqual([item.door_id for item in restored.doors], ["door-a"])
        self.assertEqual(
            [item.placement_id for item in restored.placements],
            ["valid-placement"],
        )


# ### Doorway identity persistence tests ###
class DoorwayIdentityTests(unittest.TestCase):
    def test_project_round_trip_preserves_doorway_identity(self) -> None:
        levels = create_default_levels()
        levels[2].doorways = [_doorway(doorway_id="stable-doorway")]
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "doorway-id.json"
            save_project(project_path, current_level_index=2, levels=levels)

            payload = json.loads(project_path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["levels"][2]["doorways"][0]["doorway_id"],
                "stable-doorway",
            )
            restored = load_project(project_path)
            self.assertEqual(
                restored.levels[2].doorways[0].doorway_id,
                "stable-doorway",
            )

    def test_legacy_doorway_identity_is_repeatable(self) -> None:
        levels = create_default_levels()
        levels[2].doorways = [_doorway()]
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "legacy-doorway-id.json"
            save_project(project_path, current_level_index=2, levels=levels)
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            payload["levels"][2]["doorways"][0].pop("doorway_id")
            project_path.write_text(json.dumps(payload), encoding="utf-8")

            first_id = load_project(project_path).levels[2].doorways[0].doorway_id
            second_id = load_project(project_path).levels[2].doorways[0].doorway_id

            self.assertEqual(first_id, second_id)
            self.assertTrue(first_id)


if __name__ == "__main__":
    unittest.main()
