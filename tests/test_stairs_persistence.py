# ### Imports ###
from __future__ import annotations

import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from housemaker.level_coordinates import level_image_to_world_xy
from housemaker.models import (
    DEFAULT_STAIR_NOSING_PLACEMENTS,
    DEFAULT_STAIR_STARTING_STEP,
    DEFAULT_STAIR_STARTING_STEP_EDGE_POINTS,
    DEFAULT_STAIR_STARTING_STEP_EDGE_RADIUS_METERS,
    DEFAULT_STAIR_TARGET_RISE_METERS,
    DEFAULT_STAIR_TREAD_EDGE_PROFILE,
    DEFAULT_STAIR_TREAD_EDGE_RADIUS_METERS,
    DEFAULT_STAIR_TREAD_OVERHANG_METERS,
    DEFAULT_STAIR_TREAD_THICKNESS_METERS,
    STAIR_NOSING_FRONT,
    STAIR_NOSING_LEFT,
    STAIR_NOSING_RIGHT,
    STAIR_STARTING_STEP_BULLNOSE,
    STAIR_STARTING_STEP_CURTAIL,
    STAIR_STARTING_STEP_NONE,
    STAIR_STARTING_STEPS,
    STAIR_STRINGER_BOTH,
    STAIR_STRINGER_LEFT,
    STAIR_STRINGER_NONE,
    STAIR_STRINGER_PLACEMENTS,
    STAIR_STYLE_FLOATING,
    STAIR_STYLE_FLOATING_WITH_RISER,
    STAIR_STYLE_SUPPORTED,
    STAIR_TREAD_EDGE_ROUNDED,
    STAIR_TYPE_FLOATING,
    STAIR_TYPE_SUPPORTED,
    StairData,
    StairSectionData,
    calculate_stair_step_layout,
    create_default_levels,
)
from housemaker.project_io import load_project, save_project


# ### Fixture helpers ###
def _supported_stair() -> StairData:
    return StairData(
        start_level_index=2,
        start_a_x=100.5,
        start_a_y=190.25,
        start_b_x=140.5,
        start_b_y=210.25,
        end_level_index=3,
        end_a_x=290.75,
        end_a_y=400.5,
        end_b_x=330.75,
        end_b_y=420.5,
        style=STAIR_STYLE_SUPPORTED,
        start_a_vertex_id=4,
        end_b_vertex_id=8,
    )


def _floating_stair() -> StairData:
    return StairData(
        start_level_index=3,
        start_a_x=25.0,
        start_a_y=50.0,
        start_b_x=65.0,
        start_b_y=70.0,
        end_level_index=2,
        end_a_x=60.0,
        end_a_y=90.0,
        end_b_x=100.0,
        end_b_y=110.0,
        style=STAIR_STYLE_FLOATING,
    )


def _floating_with_riser_stair() -> StairData:
    return StairData(
        start_level_index=2,
        start_a_x=25.0,
        start_a_y=50.0,
        start_b_x=65.0,
        start_b_y=70.0,
        end_level_index=3,
        end_a_x=160.0,
        end_a_y=190.0,
        end_b_x=200.0,
        end_b_y=210.0,
        style=STAIR_STYLE_FLOATING_WITH_RISER,
    )


def _curved_stair() -> StairData:
    return StairData(
        start_level_index=2,
        start_a_x=0.0,
        start_a_y=0.0,
        start_b_x=0.0,
        start_b_y=40.0,
        end_level_index=3,
        end_a_x=200.0,
        end_a_y=160.0,
        end_b_x=160.0,
        end_b_y=160.0,
        intermediate_sections=(
            StairSectionData(
                level_index=2,
                a_x=100.0,
                a_y=0.0,
                b_x=100.0,
                b_y=40.0,
                a_vertex_id=5,
            ),
            StairSectionData(
                level_index=3,
                a_x=200.0,
                a_y=60.0,
                b_x=160.0,
                b_y=60.0,
                b_vertex_id=7,
            ),
        ),
    )


# ### Stair model tests ###
class StairDataTests(unittest.TestCase):
    def test_parameterized_stair_round_trip_and_replace_preserve_identity(
        self,
    ) -> None:
        stair = replace(
            _supported_stair(),
            stair_type=STAIR_TYPE_FLOATING,
            target_rise_meters=0.19,
            tread_thickness_meters=0.065,
            tread_overhang_meters=0.04,
            nosing_placements=(
                STAIR_NOSING_LEFT,
                STAIR_NOSING_RIGHT,
                STAIR_NOSING_FRONT,
            ),
            tread_edge_profile=STAIR_TREAD_EDGE_ROUNDED,
            tread_edge_radius_meters=0.025,
            starting_step=STAIR_STARTING_STEP_CURTAIL,
            starting_step_edge_radius_meters=0.18,
            starting_step_edge_points=4,
            stringer_placement=STAIR_STRINGER_LEFT,
        )

        restored = StairData.from_dict(stair.to_dict())

        self.assertEqual(restored, stair)
        self.assertEqual(stair.stair_type, STAIR_TYPE_FLOATING)
        self.assertEqual(stair.style, STAIR_STYLE_FLOATING)
        self.assertEqual(restored.stair_id, stair.stair_id)
        self.assertEqual(len(stair.stair_id), 32)
        self.assertEqual(stair.tread_edge_radius_meters, 0.025)
        self.assertEqual(stair.starting_step, STAIR_STARTING_STEP_CURTAIL)
        self.assertEqual(stair.starting_step_edge_radius_meters, 0.18)
        self.assertEqual(stair.starting_step_edge_points, 4)
        self.assertEqual(
            stair.to_dict()["starting_step"],
            STAIR_STARTING_STEP_CURTAIL,
        )
        self.assertEqual(
            stair.to_dict()["nosing_placements"],
            [STAIR_NOSING_LEFT, STAIR_NOSING_RIGHT, STAIR_NOSING_FRONT],
        )

    def test_nosing_placements_accept_empty_and_canonicalize_iterables(self) -> None:
        without_nosing = replace(_supported_stair(), nosing_placements=[])
        reordered = replace(
            _supported_stair(),
            nosing_placements=(
                STAIR_NOSING_FRONT,
                STAIR_NOSING_LEFT,
                STAIR_NOSING_FRONT,
            ),
        )

        self.assertEqual(without_nosing.nosing_placements, ())
        self.assertEqual(
            reordered.nosing_placements,
            (STAIR_NOSING_LEFT, STAIR_NOSING_FRONT),
        )

    def test_starting_step_canonicalizes_and_enables_modern_layout(self) -> None:
        stair = replace(
            _supported_stair(),
            starting_step="  BULLNOSE  ",
        )

        self.assertEqual(stair.starting_step, STAIR_STARTING_STEP_BULLNOSE)
        self.assertFalse(stair.uses_legacy_part_layout)
        self.assertEqual(
            STAIR_STARTING_STEPS,
            (
                STAIR_STARTING_STEP_NONE,
                STAIR_STARTING_STEP_BULLNOSE,
                STAIR_STARTING_STEP_CURTAIL,
            ),
        )

    def test_legacy_stair_defaults_keep_the_old_single_part_layout(self) -> None:
        payload = _supported_stair().to_dict()
        for field_name in (
            "stair_id",
            "stair_type",
            "target_rise_meters",
            "tread_thickness_meters",
            "tread_overhang_meters",
            "nosing_placements",
            "tread_edge_profile",
            "tread_edge_radius_meters",
            "starting_step",
            "starting_step_edge_radius_meters",
            "starting_step_edge_points",
            "stringer_placement",
            "legacy_part_layout",
        ):
            payload.pop(field_name)

        stair = StairData.from_dict(payload)

        self.assertEqual(stair.stair_type, STAIR_TYPE_SUPPORTED)
        self.assertEqual(stair.stringer_placement, STAIR_STRINGER_NONE)
        self.assertEqual(
            stair.target_rise_meters,
            DEFAULT_STAIR_TARGET_RISE_METERS,
        )
        self.assertEqual(
            stair.tread_thickness_meters,
            DEFAULT_STAIR_TREAD_THICKNESS_METERS,
        )
        self.assertEqual(
            stair.tread_overhang_meters,
            DEFAULT_STAIR_TREAD_OVERHANG_METERS,
        )
        self.assertEqual(
            stair.nosing_placements,
            DEFAULT_STAIR_NOSING_PLACEMENTS,
        )
        self.assertEqual(
            stair.tread_edge_profile,
            DEFAULT_STAIR_TREAD_EDGE_PROFILE,
        )
        self.assertEqual(
            stair.tread_edge_radius_meters,
            DEFAULT_STAIR_TREAD_EDGE_RADIUS_METERS,
        )
        self.assertEqual(stair.starting_step, DEFAULT_STAIR_STARTING_STEP)
        self.assertEqual(
            stair.starting_step_edge_radius_meters,
            DEFAULT_STAIR_STARTING_STEP_EDGE_RADIUS_METERS,
        )
        self.assertEqual(
            stair.starting_step_edge_points,
            DEFAULT_STAIR_STARTING_STEP_EDGE_POINTS,
        )
        self.assertTrue(stair.uses_legacy_part_layout)

    def test_explicit_no_stringer_uses_modern_layout_and_round_trips(self) -> None:
        modern_stair = StairData(
            start_level_index=2,
            start_x=0.0,
            start_y=0.0,
            end_level_index=3,
            end_x=100.0,
            end_y=0.0,
            stringer_placement=STAIR_STRINGER_NONE,
        )

        restored = StairData.from_dict(modern_stair.to_dict())

        self.assertIn(STAIR_STRINGER_NONE, STAIR_STRINGER_PLACEMENTS)
        self.assertFalse(modern_stair.uses_legacy_part_layout)
        self.assertFalse(restored.uses_legacy_part_layout)
        self.assertFalse(restored.to_dict()["legacy_part_layout"])

    def test_unmarked_no_stringer_payload_retains_legacy_layout(self) -> None:
        payload = replace(
            _supported_stair(),
            stringer_placement=STAIR_STRINGER_NONE,
            legacy_part_layout=False,
        ).to_dict()
        payload.pop("legacy_part_layout")

        restored = StairData.from_dict(payload)

        self.assertTrue(restored.uses_legacy_part_layout)

    def test_step_layout_uses_nearest_count_and_an_even_actual_rise(self) -> None:
        step_count, actual_rise = calculate_stair_step_layout(3.0, 0.19)

        self.assertEqual(step_count, 16)
        self.assertAlmostEqual(actual_rise, 0.1875)
        self.assertAlmostEqual(step_count * actual_rise, 3.0)

    def test_parameterized_stair_rejects_invalid_values(self) -> None:
        valid_stair = _supported_stair()
        invalid_changes = (
            {"stair_type": "spiral"},
            {"target_rise_meters": 0.0},
            {"tread_thickness_meters": float("nan")},
            {"tread_overhang_meters": -0.01},
            {"nosing_placements": STAIR_NOSING_FRONT},
            {"nosing_placements": (STAIR_NOSING_FRONT, "back")},
            {"nosing_placements": (STAIR_NOSING_FRONT, 1)},
            {"nosing_placements": None},
            {"tread_edge_profile": "jagged"},
            {"tread_edge_radius_meters": -0.01},
            {"tread_edge_radius_meters": float("inf")},
            {"starting_step": "winder"},
            {"starting_step": None},
            {"starting_step_edge_radius_meters": -0.01},
            {"starting_step_edge_radius_meters": float("inf")},
            {"starting_step_edge_points": 0},
            {"starting_step_edge_points": 17},
            {"starting_step_edge_points": 1.5},
            {"starting_step_edge_points": True},
            {"stringer_placement": "center"},
            {"legacy_part_layout": "no"},
            {"legacy_part_layout": None},
            {"stair_id": "not-a-uuid"},
        )

        for changes in invalid_changes:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    replace(valid_stair, **changes)

    def test_new_stringer_configuration_removes_legacy_riser_variant(self) -> None:
        legacy = _floating_with_riser_stair()

        configured = replace(
            legacy,
            stringer_placement=STAIR_STRINGER_BOTH,
        )
        configured_without_stringer = replace(
            legacy,
            stringer_placement=STAIR_STRINGER_NONE,
            legacy_part_layout=False,
        )

        self.assertEqual(configured.stair_type, STAIR_TYPE_FLOATING)
        self.assertEqual(configured.style, STAIR_STYLE_FLOATING)
        self.assertFalse(configured.has_legacy_riser_panels)
        self.assertEqual(
            configured_without_stringer.style,
            STAIR_STYLE_FLOATING,
        )
        self.assertFalse(configured_without_stringer.has_legacy_riser_panels)

    def test_stair_round_trip_preserves_local_endpoints_and_style(self) -> None:
        stair = _supported_stair()
        legacy_payload = stair.to_dict()
        legacy_payload.pop("style")

        self.assertEqual(StairData.from_dict(stair.to_dict()), stair)
        self.assertEqual(StairData.from_dict(legacy_payload), stair)
        self.assertEqual(
            stair.start_points,
            ((100.5, 190.25), (140.5, 210.25)),
        )
        self.assertEqual(stair.start_x, 120.5)
        self.assertEqual(stair.start_y, 200.25)
        self.assertFalse(stair.is_floating)
        self.assertTrue(_floating_stair().is_floating)
        riser_stair = _floating_with_riser_stair()
        self.assertTrue(riser_stair.is_floating)
        self.assertEqual(
            StairData.from_dict(riser_stair.to_dict()),
            riser_stair,
        )

    def test_curved_stair_round_trip_preserves_ordered_sections(self) -> None:
        stair = _curved_stair()

        restored_stair = StairData.from_dict(stair.to_dict())

        self.assertEqual(restored_stair, stair)
        self.assertEqual(
            restored_stair.intermediate_sections,
            stair.intermediate_sections,
        )
        self.assertEqual(
            restored_stair.sections,
            (
                restored_stair.start_section,
                *restored_stair.intermediate_sections,
                restored_stair.end_section,
            ),
        )
        self.assertEqual(
            restored_stair.intermediate_sections[0].points,
            ((100.0, 0.0), (100.0, 40.0)),
        )
        self.assertEqual(restored_stair.intermediate_sections[0].a_vertex_id, 5)

    def test_four_point_json_without_sections_remains_compatible(self) -> None:
        legacy_four_point_payload = _supported_stair().to_dict()
        legacy_four_point_payload.pop("intermediate_sections")

        restored_stair = StairData.from_dict(legacy_four_point_payload)

        self.assertEqual(restored_stair.intermediate_sections, ())

    def test_legacy_single_point_payload_migrates_to_width_segments(self) -> None:
        legacy_payload = {
            "start_level_index": 2,
            "start_x": 100.0,
            "start_y": 50.0,
            "end_level_index": 3,
            "end_x": 200.0,
            "end_y": 50.0,
            "style": STAIR_STYLE_FLOATING,
        }

        stair = StairData.from_dict(legacy_payload)

        self.assertEqual((stair.start_x, stair.start_y), (100.0, 50.0))
        self.assertEqual((stair.end_x, stair.end_y), (200.0, 50.0))
        self.assertAlmostEqual(
            math.dist(stair.start_points[0], stair.start_points[1]),
            50.0,
        )
        self.assertAlmostEqual(
            math.dist(stair.end_points[0], stair.end_points[1]),
            50.0,
        )
        self.assertNotIn("start_x", stair.to_dict())
        self.assertIn("start_a_x", stair.to_dict())

    def test_stair_rejects_invalid_endpoints_and_styles(self) -> None:
        valid_stair = _supported_stair()
        invalid_payloads = (
            valid_stair.to_dict() | {"start_level_index": 2, "end_level_index": 2},
            valid_stair.to_dict() | {"start_a_x": float("nan")},
            valid_stair.to_dict() | {"end_b_y": "not a coordinate"},
            valid_stair.to_dict() | {"style": "spiral"},
            valid_stair.to_dict() | {"start_level_index": True},
            valid_stair.to_dict()
            | {"start_b_x": valid_stair.start_a_x, "start_b_y": valid_stair.start_a_y},
            valid_stair.to_dict() | {"start_a_vertex_id": 0},
            valid_stair.to_dict() | {"end_b_vertex_id": True},
            valid_stair.to_dict() | {"start_x": 10.0},
            valid_stair.to_dict() | {"intermediate_sections": None},
            valid_stair.to_dict() | {"intermediate_sections": [None]},
            valid_stair.to_dict()
            | {
                "intermediate_sections": [
                    {
                        "level_index": 2,
                        "a_x": 10.0,
                        "a_y": 10.0,
                        "b_x": 10.0,
                        "b_y": 10.0,
                    }
                ]
            },
        )

        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    StairData.from_dict(payload)


# ### Project persistence tests ###
class StairProjectPersistenceTests(unittest.TestCase):
    def test_duplicate_persisted_stair_ids_are_repaired_deterministically(self) -> None:
        first = _supported_stair()
        second = replace(
            _floating_stair(),
            stair_id=first.stair_id,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "duplicate-stair-ids.json"
            save_project(
                project_path,
                current_level_index=2,
                levels=create_default_levels(),
                stairs=[first, second],
            )

            first_load = load_project(project_path)
            second_load = load_project(project_path)

        first_ids = tuple(stair.stair_id for stair in first_load.stairs)
        second_ids = tuple(stair.stair_id for stair in second_load.stairs)
        self.assertEqual(first_ids, second_ids)
        self.assertEqual(len(first_ids), 2)
        self.assertEqual(len(set(first_ids)), 2)
        self.assertEqual(first_ids[0], first.stair_id)

    def test_legacy_project_load_assigns_a_repeatable_stair_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "legacy-stairs.json"
            save_project(
                project_path,
                current_level_index=2,
                levels=create_default_levels(),
                stairs=[_supported_stair()],
            )
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            for field_name in (
                "stair_id",
                "stair_type",
                "target_rise_meters",
                "tread_thickness_meters",
                "tread_overhang_meters",
                "nosing_placements",
                "tread_edge_profile",
                "tread_edge_radius_meters",
                "starting_step",
                "starting_step_edge_radius_meters",
                "starting_step_edge_points",
                "stringer_placement",
                "legacy_part_layout",
            ):
                payload["stairs"][0].pop(field_name)
            project_path.write_text(json.dumps(payload), encoding="utf-8")

            first_load = load_project(project_path)
            second_load = load_project(project_path)

        self.assertEqual(
            first_load.stairs[0].stair_id,
            second_load.stairs[0].stair_id,
        )
        self.assertEqual(len(first_load.stairs[0].stair_id), 32)
        self.assertEqual(
            first_load.stairs[0].nosing_placements,
            DEFAULT_STAIR_NOSING_PLACEMENTS,
        )
        self.assertEqual(
            first_load.stairs[0].starting_step,
            DEFAULT_STAIR_STARTING_STEP,
        )
        self.assertEqual(
            first_load.stairs[0].starting_step_edge_radius_meters,
            DEFAULT_STAIR_STARTING_STEP_EDGE_RADIUS_METERS,
        )
        self.assertEqual(
            first_load.stairs[0].starting_step_edge_points,
            DEFAULT_STAIR_STARTING_STEP_EDGE_POINTS,
        )

    def test_project_round_trip_preserves_supported_and_floating_stairs(
        self,
    ) -> None:
        stairs = [
            _supported_stair(),
            _floating_stair(),
            _floating_with_riser_stair(),
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "stairs-project.json"
            save_project(
                project_path,
                current_level_index=2,
                levels=create_default_levels(),
                stairs=stairs,
            )
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            loaded_project = load_project(project_path)

        self.assertEqual(
            payload["stairs"],
            [stair.to_dict() for stair in stairs],
        )
        self.assertEqual(loaded_project.stairs, stairs)

    def test_project_round_trip_preserves_curved_stair_sections(self) -> None:
        stair = _curved_stair()

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "curved-stairs.json"
            save_project(
                project_path,
                current_level_index=2,
                levels=create_default_levels(),
                stairs=[stair],
            )
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            loaded_project = load_project(project_path)

        self.assertEqual(
            payload["stairs"][0]["intermediate_sections"],
            [section.to_dict() for section in stair.intermediate_sections],
        )
        self.assertEqual(loaded_project.stairs, [stair])

    def test_stair_with_invalid_intermediate_level_is_skipped(self) -> None:
        stair = _curved_stair()

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "missing-route-level.json"
            save_project(
                project_path,
                current_level_index=2,
                levels=create_default_levels(),
                stairs=[stair],
            )
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            payload["stairs"][0]["intermediate_sections"][0]["level_index"] = 99
            project_path.write_text(json.dumps(payload), encoding="utf-8")

            loaded_project = load_project(project_path)

        self.assertEqual(loaded_project.stairs, [])

    def test_legacy_project_without_stairs_loads_an_empty_list(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "legacy-project.json"
            save_project(
                project_path,
                current_level_index=2,
                levels=create_default_levels(),
            )
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            payload.pop("stairs")
            project_path.write_text(json.dumps(payload), encoding="utf-8")

            loaded_project = load_project(project_path)

        self.assertEqual(loaded_project.stairs, [])

    def test_malformed_stair_entries_are_skipped_without_losing_valid_entries(
        self,
    ) -> None:
        valid_stair = _floating_stair()
        malformed_stairs: list[object] = [
            valid_stair.to_dict(),
            None,
            {"start_level_index": 2},
            valid_stair.to_dict() | {"start_level_index": 2},
            valid_stair.to_dict() | {"style": "unsupported"},
            valid_stair.to_dict() | {"start_x": "invalid"},
            valid_stair.to_dict() | {"end_level_index": 99},
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "malformed-stairs.json"
            save_project(
                project_path,
                current_level_index=2,
                levels=create_default_levels(),
            )
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            payload["stairs"] = malformed_stairs
            project_path.write_text(json.dumps(payload), encoding="utf-8")

            loaded_project = load_project(project_path)

        self.assertEqual(loaded_project.stairs, [valid_stair])


# ### Coordinate binding tests ###
class StairCoordinateBindingTests(unittest.TestCase):
    def test_endpoint_world_position_uses_each_level_current_transform(self) -> None:
        stair = StairData(
            start_level_index=2,
            start_a_x=80.0,
            start_a_y=40.0,
            start_b_x=120.0,
            start_b_y=60.0,
            end_level_index=3,
            end_a_x=180.0,
            end_a_y=140.0,
            end_b_x=220.0,
            end_b_y=160.0,
        )
        levels = create_default_levels()
        start_level = levels[stair.start_level_index]
        end_level = levels[stair.end_level_index]

        initial_start_world = level_image_to_world_xy(
            start_level,
            stair.start_x,
            stair.start_y,
        )
        initial_end_world = level_image_to_world_xy(
            end_level,
            stair.end_x,
            stair.end_y,
        )

        start_level.scale = 1.5
        start_level.offset_x_meters = 2.0
        end_level.scale = 0.5
        end_level.offset_y_meters = -1.0

        scaled_start_world = level_image_to_world_xy(
            start_level,
            stair.start_x,
            stair.start_y,
        )
        scaled_end_world = level_image_to_world_xy(
            end_level,
            stair.end_x,
            stair.end_y,
        )

        self.assertEqual((stair.start_x, stair.start_y), (100.0, 50.0))
        self.assertEqual((stair.end_x, stair.end_y), (200.0, 150.0))
        self.assertNotEqual(initial_start_world, scaled_start_world)
        self.assertNotEqual(initial_end_world, scaled_end_world)
        self.assertEqual(scaled_start_world, (5.0, -1.5))
        self.assertEqual(scaled_end_world, (2.0, -2.5))


# ### Test runner ###
if __name__ == "__main__":
    unittest.main()
