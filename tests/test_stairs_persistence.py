# ### Imports ###
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from housemaker.level_coordinates import level_image_to_world_xy
from housemaker.models import (
    STAIR_STYLE_FLOATING,
    STAIR_STYLE_FLOATING_WITH_RISER,
    STAIR_STYLE_SUPPORTED,
    StairData,
    StairSectionData,
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
            [
                section.to_dict()
                for section in stair.intermediate_sections
            ],
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
