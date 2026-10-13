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
    WindowPlacement,
    create_door_definition_for_doorway,
    create_glass_area_data,
    create_window_definition_for_frame,
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
        door = replace(
            door,
            generate_displacement=True,
            preserve_profile=True,
        )
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
        self.assertTrue(restored.doors.doors[0].preserve_profile)

    def test_legacy_half_placement_loads_as_a_regular_door(self) -> None:
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
        library = DoorLibraryData(
            doors=[door],
            placements=[
                DoorPlacement(
                    placement_id="placement-a",
                    door_id=door.door_id,
                    level_index=levels[2].index,
                    doorway_id=doorway.doorway_id,
                )
            ],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "legacy-half-door.json"
            save_project(
                project_path,
                current_level_index=levels[2].index,
                levels=levels,
                doors=library,
            )
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            payload["doors"]["placements"][0]["fit_mode"] = "right"
            project_path.write_text(json.dumps(payload), encoding="utf-8")

            restored = load_project(project_path)

        self.assertEqual(restored.doors, library)
        self.assertNotIn(
            "fit_mode",
            restored.doors.placements[0].to_dict(),
        )

    def test_project_round_trip_preserves_window_assets_and_glass_areas(
        self,
    ) -> None:
        levels = create_default_levels()
        window = create_window_definition_for_frame(
            "window-frame-ground-a",
            1.4,
            1.1,
            window_id="window-a",
            name="New window 1",
        )
        glass_area = create_glass_area_data(
            window.window_id,
            (),
            glass_area_id="glass-a",
            minimum_x_ratio=0.1,
            maximum_x_ratio=0.9,
            minimum_z_ratio=0.1,
            maximum_z_ratio=0.9,
        )
        window = replace(window, glass_areas=(glass_area,))
        window = window.with_double(True).with_double_sided(True)
        library = DoorLibraryData(
            windows=[window],
            window_placements=[
                WindowPlacement(
                    placement_id="window-placement-a",
                    window_id=window.window_id,
                    level_index=levels[2].index,
                    window_frame_id="window-frame-ground-b",
                )
            ],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "windows-project.json"
            save_project(
                project_path,
                current_level_index=levels[2].index,
                levels=levels,
                doors=library,
            )

            restored = load_project(project_path)

        self.assertEqual(restored.doors, library)
        restored_window = restored.doors.windows[0]
        self.assertTrue(restored_window.make_double)
        self.assertTrue(restored_window.make_double_sided)

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
