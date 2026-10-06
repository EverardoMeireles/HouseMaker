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
    DOOR_SLOT_BODY,
    DOOR_SLOT_KNOB,
    DoorLibraryData,
    DoorPlacement,
    DoorSideDuplication,
    create_door_definition_for_doorway,
    door_fits_doorway,
    next_door_name,
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
        self.assertEqual(restored.placements[0].doorway_id, "doorway-destination")
        self.assertEqual(
            restored.doors[0].side_duplication,
            DoorSideDuplication(kept_side=DOOR_SIDE_DUPLICATION_BACK),
        )
        self.assertEqual(
            library.to_dict()["schema_version"],
            DOOR_LIBRARY_SCHEMA_VERSION,
        )

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

    def test_factory_creates_required_slots_and_incremental_names(self) -> None:
        first = create_door_definition_for_doorway(
            _doorway(),
            door_id="door-a",
            name="New door 1",
        )
        third = replace(first, door_id="door-c", name="New door 3")

        self.assertEqual(
            {slot.slot_id for slot in first.slots},
            {"body", "hinges", "door_knob"},
        )
        body = first.get_slot(DOOR_SLOT_BODY)
        assert body is not None
        self.assertTrue(body.joined)
        self.assertEqual(next_door_name((first, third)), "New door 4")
        self.assertTrue(door_fits_doorway(first, _doorway()))
        self.assertFalse(
            door_fits_doorway(
                first,
                replace(_doorway(), width_meters=1.2),
            )
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
