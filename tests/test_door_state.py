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
    DOOR_SLOT_BACK_LEFT_BODY,
    DOOR_SLOT_BACK_LEFT_KNOB,
    DOOR_SLOT_BODY,
    DOOR_SLOT_KNOB,
    DOOR_SLOT_LEFT_BODY,
    DOOR_SLOT_LEFT_KNOB,
    DoorLibraryData,
    DoorPlacement,
    DoorSideDuplication,
    DoorSlotData,
    GlassAreaData,
    WindowPlacement,
    create_door_definition_for_doorway,
    create_glass_area_data,
    create_window_definition_for_frame,
    door_fits_doorway,
    door_slot_component,
    door_slot_counterpart,
    door_slot_display_name,
    door_slot_leaf,
    door_slot_side,
    is_door_body_slot,
    is_door_knob_slot,
    next_door_name,
    next_glass_area_name,
    next_window_name,
    set_door_double,
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
    def test_legacy_handle_slot_id_uses_short_display_label(self) -> None:
        slot = DoorSlotData(
            slot_id=DOOR_SLOT_KNOB,
            source_object_id="legacy-handle",
        )

        self.assertEqual(slot.slot_id, "door_knob")
        self.assertEqual(slot.display_name, "Handle")

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
        displaced = replace(door, generate_displacement=False)

        restored = type(door).from_dict(displaced.to_dict())
        legacy_payload = door.to_dict()
        legacy_payload.pop("generate_displacement")
        restored_legacy = type(door).from_dict(legacy_payload)

        self.assertTrue(door.generate_displacement)
        self.assertFalse(restored.generate_displacement)
        self.assertFalse(restored_legacy.generate_displacement)

    def test_profile_preservation_round_trip_and_legacy_default(self) -> None:
        door = create_door_definition_for_doorway(
            _doorway(),
            door_id="door-a",
            name="New door 1",
        )
        preserving = replace(door, preserve_profile=True)

        restored = type(door).from_dict(preserving.to_dict())
        legacy_payload = door.to_dict()
        legacy_payload.pop("preserve_profile")
        restored_legacy = type(door).from_dict(legacy_payload)

        self.assertTrue(restored.preserve_profile)
        self.assertFalse(restored_legacy.preserve_profile)

    def test_displacement_generation_state_must_be_boolean(self) -> None:
        door = create_door_definition_for_doorway(
            _doorway(),
            door_id="door-a",
            name="New door 1",
        )

        with self.assertRaisesRegex(TypeError, "displacement-generation"):
            replace(door, generate_displacement=1)

        with self.assertRaisesRegex(TypeError, "profile-preservation"):
            replace(door, preserve_profile=1)

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

    def test_double_toggle_builds_independent_mirrored_leaf_slots(self) -> None:
        original = create_door_definition_for_doorway(
            _doorway(),
            door_id="door-a",
            name="New door 1",
        )

        doubled = set_door_double(original, True)

        self.assertTrue(doubled.make_double)
        self.assertEqual(
            tuple(slot.slot_id for slot in doubled.slots),
            (
                DOOR_SLOT_BODY,
                DOOR_SLOT_LEFT_BODY,
                DOOR_SLOT_KNOB,
                DOOR_SLOT_LEFT_KNOB,
            ),
        )
        self.assertEqual(
            len({slot.source_object_id for slot in doubled.slots}),
            len(doubled.slots),
        )
        right_body = doubled.get_slot(DOOR_SLOT_BODY)
        left_body = doubled.get_slot(DOOR_SLOT_LEFT_BODY)
        right_handle = doubled.get_slot(DOOR_SLOT_KNOB)
        left_handle = doubled.get_slot(DOOR_SLOT_LEFT_KNOB)
        assert right_body is not None and left_body is not None
        assert right_handle is not None and left_handle is not None
        self.assertEqual(right_body.position_meters[0], 0.225)
        self.assertEqual(left_body.position_meters[0], -0.225)
        self.assertEqual(right_body.axis_scales[0], 0.5)
        self.assertEqual(left_body.axis_scales[0], 0.5)
        self.assertTrue(right_body.mirrored_horizontally)
        self.assertFalse(left_body.mirrored_horizontally)
        self.assertEqual(
            left_handle.position_meters[0],
            -right_handle.position_meters[0],
        )
        self.assertAlmostEqual(right_handle.position_meters[0], 0.04)
        self.assertTrue(right_handle.mirrored_horizontally)
        self.assertFalse(left_handle.mirrored_horizontally)
        self.assertEqual(type(doubled).from_dict(doubled.to_dict()), doubled)
        self.assertEqual(doubled.with_double(False), original)

    def test_double_toggle_exactly_restores_custom_source_handedness(self) -> None:
        original = create_door_definition_for_doorway(
            _doorway(),
            door_id="custom-handed-door",
            name="New door 1",
        )
        body = original.get_slot(DOOR_SLOT_BODY)
        handle = original.get_slot(DOOR_SLOT_KNOB)
        assert body is not None and handle is not None
        original = original.replace_slot(
            body.with_transform(mirrored_horizontally=True)
        ).replace_slot(
            handle.with_transform(
                position_meters=(0.21, -0.03, 0.82),
                mirrored_horizontally=True,
            )
        )

        restored = original.with_double(True).with_double(False)

        self.assertEqual(restored, original)

    def test_double_and_double_sided_slots_combine_without_shared_ids(
        self,
    ) -> None:
        original = create_door_definition_for_doorway(
            _doorway(),
            door_id="door-a",
            name="New door 1",
        )
        doubled = original.with_double(True)

        complete = doubled.with_double_sided(True)

        expected_slot_ids = {
            DOOR_SLOT_BODY,
            DOOR_SLOT_KNOB,
            DOOR_SLOT_LEFT_BODY,
            DOOR_SLOT_LEFT_KNOB,
            DOOR_SLOT_BACK_BODY,
            DOOR_SLOT_BACK_KNOB,
            DOOR_SLOT_BACK_LEFT_BODY,
            DOOR_SLOT_BACK_LEFT_KNOB,
        }
        self.assertEqual(
            {slot.slot_id for slot in complete.slots},
            expected_slot_ids,
        )
        self.assertEqual(
            len({slot.source_object_id for slot in complete.slots}),
            len(expected_slot_ids),
        )
        self.assertEqual(
            door_slot_counterpart(DOOR_SLOT_BACK_BODY),
            DOOR_SLOT_BACK_LEFT_BODY,
        )
        self.assertEqual(door_slot_leaf(DOOR_SLOT_LEFT_KNOB), "left")
        self.assertEqual(
            door_slot_display_name(DOOR_SLOT_BODY, make_double=True),
            "Body right",
        )
        self.assertEqual(
            door_slot_display_name(DOOR_SLOT_LEFT_KNOB, make_double=True),
            "Handle left",
        )
        self.assertEqual(type(complete).from_dict(complete.to_dict()), complete)
        self.assertEqual(complete.with_double_sided(False), doubled)
        self.assertEqual(
            complete.with_double(False),
            original.with_double_sided(True),
        )

    def test_legacy_door_and_slot_payloads_default_to_single_unmirrored(
        self,
    ) -> None:
        original = create_door_definition_for_doorway(
            _doorway(),
            door_id="legacy-door",
            name="New door 1",
        )
        payload = original.to_dict()
        payload.pop("make_double")
        for raw_slot in payload["slots"]:
            raw_slot.pop("mirrored_horizontally")

        restored = type(original).from_dict(payload)

        self.assertEqual(restored, original)
        self.assertFalse(restored.make_double)
        self.assertTrue(
            all(not slot.mirrored_horizontally for slot in restored.slots)
        )

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

    def test_glass_areas_validate_bounds_and_use_monotonic_names(self) -> None:
        first = create_glass_area_data(
            "door-a",
            (),
            glass_area_id="glass-a",
            minimum_x_ratio=0.1,
            maximum_x_ratio=0.4,
            minimum_z_ratio=0.2,
            maximum_z_ratio=0.8,
        )
        third = replace(
            first,
            glass_area_id="glass-c",
            source_object_id="opening:door-a:glass:glass-c",
            name="Glass area 3",
        )
        second = create_glass_area_data(
            "door-a",
            (first, third),
            glass_area_id="glass-d",
            minimum_x_ratio=0.5,
            maximum_x_ratio=0.9,
            minimum_z_ratio=0.1,
            maximum_z_ratio=0.7,
        )

        self.assertEqual(first.name, "Glass area 1")
        self.assertEqual(second.name, "Glass area 4")
        self.assertEqual(next_glass_area_name((first, third)), "Glass area 4")
        self.assertEqual(
            GlassAreaData.from_dict(first.to_dict()),
            first,
        )
        for invalid_bounds in (
            (-0.1, 0.4, 0.2, 0.8),
            (0.1, 1.1, 0.2, 0.8),
            (0.4, 0.4, 0.2, 0.8),
            (0.1, 0.4, 0.8, 0.2),
        ):
            with self.subTest(bounds=invalid_bounds), self.assertRaises(
                ValueError
            ):
                GlassAreaData(
                    glass_area_id="invalid",
                    source_object_id="invalid-source",
                    name="Glass area 1",
                    minimum_x_ratio=invalid_bounds[0],
                    maximum_x_ratio=invalid_bounds[1],
                    minimum_z_ratio=invalid_bounds[2],
                    maximum_z_ratio=invalid_bounds[3],
                )

    def test_door_glass_areas_round_trip_and_legacy_payload_defaults_empty(
        self,
    ) -> None:
        door = create_door_definition_for_doorway(
            _doorway(),
            door_id="door-a",
            name="New door 1",
        )
        glass_area = create_glass_area_data(
            door.door_id,
            (),
            glass_area_id="glass-a",
            minimum_x_ratio=0.2,
            maximum_x_ratio=0.8,
            minimum_z_ratio=0.25,
            maximum_z_ratio=0.75,
        )
        with_glass = replace(door, glass_areas=(glass_area,))

        restored = type(door).from_dict(with_glass.to_dict())
        legacy_payload = door.to_dict()
        legacy_payload.pop("glass_areas")
        restored_legacy = type(door).from_dict(legacy_payload)
        legacy_area_payload = glass_area.to_dict()
        legacy_area_payload.pop("body_slot_id")

        self.assertEqual(restored, with_glass)
        self.assertEqual(
            restored.get_glass_area(glass_area.glass_area_id),
            glass_area,
        )
        self.assertEqual(restored_legacy.glass_areas, ())
        self.assertEqual(
            GlassAreaData.from_dict(legacy_area_payload),
            glass_area,
        )
        with self.assertRaisesRegex(ValueError, "source object IDs"):
            replace(
                door,
                glass_areas=(
                    glass_area,
                    replace(
                        glass_area,
                        glass_area_id="glass-b",
                        name="Glass area 2",
                    ),
                ),
            )
        with self.assertRaisesRegex(ValueError, "cannot overlap"):
            replace(
                door,
                glass_areas=(
                    glass_area,
                    replace(
                        glass_area,
                        glass_area_id="glass-overlap",
                        source_object_id="glass-overlap-source",
                        name="Glass area 2",
                        minimum_x_ratio=0.7,
                        maximum_x_ratio=0.9,
                    ),
                ),
            )

    def test_glass_areas_can_share_ratios_on_independent_door_leaves(
        self,
    ) -> None:
        door = create_door_definition_for_doorway(
            _doorway(),
            door_id="double-door",
            name="New door 1",
        ).with_double(True)
        right_area = create_glass_area_data(
            door.door_id,
            (),
            glass_area_id="glass-right",
            source_object_id="glass-source-right",
            minimum_x_ratio=0.2,
            maximum_x_ratio=0.8,
            minimum_z_ratio=0.2,
            maximum_z_ratio=0.8,
            body_slot_id=DOOR_SLOT_BODY,
        )
        left_area = create_glass_area_data(
            door.door_id,
            (right_area,),
            glass_area_id="glass-left",
            source_object_id="glass-source-left",
            minimum_x_ratio=0.2,
            maximum_x_ratio=0.8,
            minimum_z_ratio=0.2,
            maximum_z_ratio=0.8,
            body_slot_id=DOOR_SLOT_LEFT_BODY,
        )

        with_glass = replace(door, glass_areas=(right_area, left_area))

        self.assertEqual(
            tuple(area.body_slot_id for area in with_glass.glass_areas),
            (DOOR_SLOT_BODY, DOOR_SLOT_LEFT_BODY),
        )
        self.assertEqual(
            type(door).from_dict(with_glass.to_dict()),
            with_glass,
        )
        self.assertEqual(
            with_glass.with_double(False).glass_areas,
            (right_area,),
        )
        with self.assertRaisesRegex(ValueError, "active body slot"):
            replace(
                with_glass.with_double(False),
                glass_areas=(left_area,),
            )

    def test_door_placements_allow_only_one_door_per_doorway(self) -> None:
        door = create_door_definition_for_doorway(
            _doorway(),
            door_id="door-a",
            name="New door 1",
        )
        placement = DoorPlacement(
            placement_id="first",
            door_id=door.door_id,
            level_index=2,
            doorway_id="destination",
        )
        library = DoorLibraryData(
            doors=[door],
            placements=[placement],
        )

        self.assertEqual(
            DoorLibraryData.from_dict(library.to_dict()),
            library,
        )
        self.assertNotIn(
            "fit_mode",
            placement.to_dict(),
        )

        with self.assertRaisesRegex(ValueError, "bindings must be unique"):
            DoorLibraryData(
                doors=[door],
                placements=[
                    placement,
                    replace(placement, placement_id="duplicate"),
                ],
            )

    def test_legacy_half_placement_mode_is_ignored_on_load(self) -> None:
        payload = {
            "placement_id": "legacy-left",
            "door_id": "door-a",
            "level_index": 2,
            "doorway_id": "destination",
            "mirrored_horizontally": True,
            "fit_mode": "left",
        }

        placement = DoorPlacement.from_dict(payload)

        self.assertEqual(placement.placement_id, "legacy-left")
        self.assertTrue(placement.mirrored_horizontally)
        self.assertNotIn("fit_mode", placement.to_dict())

    def test_window_definition_and_placement_round_trip(self) -> None:
        first = create_window_definition_for_frame(
            "frame-a",
            1.4,
            1.1,
            window_id="window-a",
            name="New window 1",
        )
        third = replace(first, window_id="window-c", name="New window 3")
        glass_area = create_glass_area_data(
            first.window_id,
            (),
            glass_area_id="glass-window-a",
            minimum_x_ratio=0.1,
            maximum_x_ratio=0.9,
            minimum_z_ratio=0.1,
            maximum_z_ratio=0.9,
        )
        first = replace(first, glass_areas=(glass_area,))
        placement = WindowPlacement(
            placement_id="window-placement-a",
            window_id=first.window_id,
            level_index=2,
            window_frame_id="frame-destination",
            mirrored_horizontally=True,
        )
        library = DoorLibraryData(
            windows=[first],
            window_placements=[placement],
        )

        restored = DoorLibraryData.from_dict(library.to_dict())

        self.assertEqual(restored, library)
        self.assertTrue(first.generate_displacement)
        self.assertFalse(first.preserve_profile)
        self.assertEqual(
            tuple(slot.display_name for slot in first.slots),
            ("Body", "Handle"),
        )
        self.assertTrue(first.get_slot(DOOR_SLOT_BODY).joined)  # type: ignore[union-attr]
        self.assertEqual(first.get_glass_area("glass-window-a"), glass_area)
        self.assertEqual(next_window_name((first, third)), "New window 4")
        self.assertEqual(
            library.to_dict()["schema_version"],
            DOOR_LIBRARY_SCHEMA_VERSION,
        )

        legacy_payload = first.to_dict()
        legacy_payload.pop("generate_displacement")
        legacy_payload.pop("preserve_profile")
        restored_legacy = type(first).from_dict(legacy_payload)
        self.assertFalse(restored_legacy.generate_displacement)
        self.assertFalse(restored_legacy.preserve_profile)

    def test_window_double_structure_round_trips_with_window_source_ids(
        self,
    ) -> None:
        original = create_window_definition_for_frame(
            "frame-a",
            1.2,
            1.0,
            window_id="window-a",
            name="New window 1",
        )

        complete = original.with_double(True).with_double_sided(True)

        expected_slot_ids = {
            DOOR_SLOT_BODY,
            DOOR_SLOT_KNOB,
            DOOR_SLOT_LEFT_BODY,
            DOOR_SLOT_LEFT_KNOB,
            DOOR_SLOT_BACK_BODY,
            DOOR_SLOT_BACK_KNOB,
            DOOR_SLOT_BACK_LEFT_BODY,
            DOOR_SLOT_BACK_LEFT_KNOB,
        }
        self.assertTrue(complete.make_double)
        self.assertTrue(complete.make_double_sided)
        self.assertEqual(
            {slot.slot_id for slot in complete.slots},
            expected_slot_ids,
        )
        self.assertTrue(
            all(
                slot.source_object_id.startswith("window:window-a:slot:")
                for slot in complete.slots
            )
        )
        self.assertEqual(
            type(complete).from_dict(complete.to_dict()),
            complete,
        )
        self.assertEqual(
            complete.with_double_sided(False).with_double(False),
            original,
        )
        self.assertEqual(
            complete.with_double(False).with_double_sided(False),
            original,
        )

    def test_window_rejects_slots_for_inactive_structure(self) -> None:
        window = create_window_definition_for_frame(
            "frame-a",
            1.2,
            1.0,
            window_id="window-a",
            name="New window 1",
        ).with_double(True)

        with self.assertRaisesRegex(ValueError, "inactive component slots"):
            replace(window, make_double=False)

    def test_library_load_skips_conflicting_and_dangling_window_entries(
        self,
    ) -> None:
        window = create_window_definition_for_frame(
            "frame-a",
            1.0,
            1.0,
            window_id="window-a",
            name="New window 1",
        )
        valid = WindowPlacement(
            placement_id="valid",
            window_id=window.window_id,
            level_index=2,
            window_frame_id="frame-target",
        )
        payload = {
            "windows": [window.to_dict(), {"broken": True}],
            "window_placements": [
                valid.to_dict(),
                replace(valid, placement_id="duplicate").to_dict(),
                {
                    "placement_id": "dangling",
                    "window_id": "missing",
                    "level_index": 2,
                    "window_frame_id": "other-frame",
                },
            ],
        }

        restored = DoorLibraryData.from_dict(payload)

        self.assertEqual(restored.windows, [window])
        self.assertEqual(restored.window_placements, [valid])


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
