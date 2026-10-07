# ### Imports ###
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from housemaker.door_state import (
    DOOR_SLOT_KNOB,
    DoorLibraryData,
    DoorPlacement,
    create_door_definition_for_doorway,
)
from housemaker.models import DoorwayData, create_default_levels
from housemaker.project_io import load_project, save_project


# ### Door project persistence tests ###
class DoorProjectPersistenceTests(unittest.TestCase):
    def test_project_round_trip_preserves_door_library_and_placements(self) -> None:
        levels = create_default_levels()
        doorway = DoorwayData(
            doorway_id="doorway-ground-a",
            center_x=120.0,
            center_y=90.0,
            width_meters=0.9,
            height_meters=2.1,
            depth_meters=0.25,
        )
        levels[2].doorways = [doorway]
        door = create_door_definition_for_doorway(
            doorway,
            door_id="door-a",
            name="New door 1",
        )
        knob = door.get_slot(DOOR_SLOT_KNOB)
        assert knob is not None
        door = door.replace_slot(
            knob.with_transform(
                position_meters=(0.32, 0.01, 0.94),
                rotation_degrees=(0.0, 15.0, 0.0),
                axis_scales=(1.2, 0.8, 1.1),
                joined=True,
            )
        )
        door = replace(door, generate_displacement=True)
        doors = DoorLibraryData(
            doors=[door],
            placements=[
                DoorPlacement(
                    placement_id="door-placement-a",
                    door_id=door.door_id,
                    level_index=levels[2].index,
                    doorway_id=doorway.doorway_id,
                    mirrored_horizontally=True,
                )
            ],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "doors-project.json"
            save_project(
                project_path,
                current_level_index=levels[2].index,
                levels=levels,
                doors=doors,
            )

            payload = json.loads(project_path.read_text(encoding="utf-8"))
            restored = load_project(project_path)

        self.assertEqual(payload["doors"], doors.to_dict())
        self.assertEqual(restored.doors, doors)
        self.assertTrue(restored.doors.doors[0].generate_displacement)

    def test_project_without_doors_loads_an_empty_library(self) -> None:
        levels = create_default_levels()
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "legacy-project.json"
            save_project(
                project_path,
                current_level_index=levels[2].index,
                levels=levels,
            )
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            payload.pop("doors")
            project_path.write_text(json.dumps(payload), encoding="utf-8")

            restored = load_project(project_path)

        self.assertEqual(restored.doors, DoorLibraryData())

    def test_malformed_door_library_does_not_block_project_loading(self) -> None:
        levels = create_default_levels()
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "damaged-doors.json"
            save_project(
                project_path,
                current_level_index=levels[2].index,
                levels=levels,
            )
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            payload["doors"] = "not-a-door-library"
            project_path.write_text(json.dumps(payload), encoding="utf-8")

            restored = load_project(project_path)

        self.assertEqual(restored.doors, DoorLibraryData())


if __name__ == "__main__":
    unittest.main()
