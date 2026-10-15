# ### Imports ###
from __future__ import annotations

import json
import math
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from housemaker.models import (
    DEFAULT_SLOPED_CEILING_ANGLE_DEGREES,
    DEFAULT_SLOPED_CEILING_DEPTH_METERS,
    DEFAULT_SLOPED_CEILING_TRANSITION_RADIUS_METERS,
    LevelData,
    SlopedCeilingData,
    VertexData,
)
from housemaker.project_io import load_project, save_project


# ### Fixture helpers ###
def _build_level() -> LevelData:
    vertices = VertexData()
    first = vertices.add_vertex(0.0, 0.0)
    second = vertices.add_vertex(200.0, 0.0)
    vertices.add_edge(first.id, second.id)
    return LevelData(
        index=2,
        name="Ground",
        vertex_data=vertices,
    )


def _build_slope(**changes: object) -> SlopedCeilingData:
    values: dict[str, object] = {
        "slope_id": "a" * 32,
        "wall_surface_id": "level:2/wall:1:2",
        "depth_meters": 0.8,
        "angle_degrees": 37.5,
        "wall_drop_meters": 0.62,
        "transition_radius_meters": 0.12,
        "wall_transition_radius_meters": 0.08,
        "side_transition_radius_meters": 0.06,
        "start_ratio": 0.15,
        "end_ratio": 0.85,
    }
    values.update(changes)
    return SlopedCeilingData(**values)  # type: ignore[arg-type]


# ### Domain model tests ###
class SlopedCeilingDataTests(unittest.TestCase):
    def test_model_normalizes_and_round_trips_every_parameter(self) -> None:
        slope = _build_slope(
            slope_id=" " + ("A" * 32) + " ",
            wall_surface_id=" level:2/room:5/wall:1:2 ",
            depth_meters="0.8",
            angle_degrees="37.5",
            wall_drop_meters="0.62",
            wall_transition_radius_meters="0.08",
            side_transition_radius_meters="0.06",
        )

        self.assertEqual(slope.slope_id, "a" * 32)
        self.assertEqual(
            slope.wall_surface_id,
            "level:2/room:5/wall:1:2",
        )
        expected_angle = math.degrees(math.atan2(0.62, 0.8))
        self.assertAlmostEqual(slope.angle_degrees, expected_angle)
        self.assertEqual(SlopedCeilingData.from_dict(slope.to_dict()), slope)
        self.assertEqual(
            slope.to_dict(),
            {
                "slope_id": "a" * 32,
                "wall_surface_id": "level:2/room:5/wall:1:2",
                "depth_meters": 0.8,
                "angle_degrees": expected_angle,
                "wall_drop_meters": 0.62,
                "transition_radius_meters": 0.12,
                "wall_transition_radius_meters": 0.08,
                "side_transition_radius_meters": 0.06,
                "start_ratio": 0.15,
                "end_ratio": 0.85,
            },
        )
        with self.assertRaises(FrozenInstanceError):
            slope.angle_degrees = 50.0  # type: ignore[misc]

    def test_missing_optional_parameters_use_stable_defaults(self) -> None:
        slope = SlopedCeilingData.from_dict(
            {
                "slope_id": "b" * 32,
                "wall_surface_id": "level:2/wall:1:2",
            }
        )

        self.assertEqual(
            slope.depth_meters,
            DEFAULT_SLOPED_CEILING_DEPTH_METERS,
        )
        self.assertEqual(
            slope.angle_degrees,
            DEFAULT_SLOPED_CEILING_ANGLE_DEGREES,
        )
        self.assertAlmostEqual(
            slope.wall_drop_meters,
            slope.depth_meters
            * math.tan(math.radians(DEFAULT_SLOPED_CEILING_ANGLE_DEGREES)),
        )
        self.assertEqual(
            slope.transition_radius_meters,
            DEFAULT_SLOPED_CEILING_TRANSITION_RADIUS_METERS,
        )
        self.assertEqual(slope.wall_transition_radius_meters, 0.0)
        self.assertEqual(slope.side_transition_radius_meters, 0.0)
        self.assertEqual((slope.start_ratio, slope.end_ratio), (0.0, 1.0))
        self.assertIsNone(slope.joined_group_id)

    def test_joined_parallel_slope_identity_round_trips(self) -> None:
        slope = _build_slope(joined_group_id="d" * 32)

        payload = slope.to_dict()

        self.assertEqual(payload["joined_group_id"], "d" * 32)
        self.assertEqual(SlopedCeilingData.from_dict(payload), slope)

    def test_legacy_angle_and_shared_radius_migrate_without_losing_shape(self) -> None:
        slope = SlopedCeilingData.from_dict(
            {
                "slope_id": "b" * 32,
                "wall_surface_id": "level:2/wall:1:2",
                "depth_meters": 0.9,
                "angle_degrees": 30.0,
                "transition_radius_meters": 0.17,
                "start_ratio": 0.2,
                "end_ratio": 0.8,
            }
        )

        self.assertAlmostEqual(
            slope.wall_drop_meters,
            0.9 * math.tan(math.radians(30.0)),
        )
        self.assertAlmostEqual(slope.angle_degrees, 30.0)
        self.assertEqual(slope.transition_radius_meters, 0.17)
        self.assertEqual(slope.wall_transition_radius_meters, 0.0)
        self.assertEqual(slope.side_transition_radius_meters, 0.17)
        self.assertEqual(SlopedCeilingData.from_dict(slope.to_dict()), slope)

    def test_intermediate_independent_side_radii_migrate_to_the_larger_value(
        self,
    ) -> None:
        slope = SlopedCeilingData.from_dict(
            {
                "slope_id": "c" * 32,
                "wall_surface_id": "level:2/wall:1:2",
                "depth_meters": 0.9,
                "wall_drop_meters": 0.5,
                "transition_radius_meters": 0.08,
                "start_transition_radius_meters": 0.11,
                "end_transition_radius_meters": 0.19,
                "start_ratio": 0.2,
                "end_ratio": 0.8,
            }
        )

        self.assertEqual(slope.side_transition_radius_meters, 0.19)
        canonical = slope.to_dict()
        self.assertNotIn("start_transition_radius_meters", canonical)
        self.assertNotIn("end_transition_radius_meters", canonical)
        self.assertEqual(SlopedCeilingData.from_dict(canonical), slope)

    def test_invalid_identifiers_dimensions_and_ranges_are_rejected(self) -> None:
        invalid_changes = (
            {"slope_id": "not-a-uuid"},
            {"wall_surface_id": "level:2/ceiling"},
            {"depth_meters": 0.0},
            {"depth_meters": float("inf")},
            {"wall_drop_meters": None, "angle_degrees": 0.0},
            {"wall_drop_meters": None, "angle_degrees": 90.0},
            {"wall_drop_meters": -0.01},
            {"wall_drop_meters": float("nan")},
            {"transition_radius_meters": -0.01},
            {"transition_radius_meters": float("nan")},
            {"wall_transition_radius_meters": -0.01},
            {"wall_transition_radius_meters": float("nan")},
            {"side_transition_radius_meters": -0.01},
            {"side_transition_radius_meters": float("nan")},
            {"start_ratio": -0.01},
            {"end_ratio": 1.01},
            {"start_ratio": 0.5, "end_ratio": 0.5},
            {"start_ratio": 0.75, "end_ratio": 0.25},
        )

        for changes in invalid_changes:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                _build_slope(**changes)


# ### Project round-trip tests ###
class SlopedCeilingProjectTests(unittest.TestCase):
    def test_project_round_trip_preserves_level_owned_slopes(self) -> None:
        level = _build_level()
        level.sloped_ceilings = [
            _build_slope(
                start_ratio=0.0,
                end_ratio=0.4,
                joined_group_id="f" * 32,
            ),
            _build_slope(
                slope_id="b" * 32,
                wall_surface_id="level:2/wall:1:2",
                start_ratio=0.6,
                end_ratio=1.0,
                wall_drop_meters=None,
                angle_degrees=62.0,
                joined_group_id="f" * 32,
            ),
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "slopes.json"
            save_project(project_path, level.index, [level])
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            restored = load_project(project_path).levels[level.index]

        self.assertEqual(
            payload["levels"][0]["sloped_ceilings"],
            [slope.to_dict() for slope in level.sloped_ceilings],
        )
        self.assertEqual(restored.sloped_ceilings, level.sloped_ceilings)

    def test_legacy_project_without_slopes_loads_an_empty_collection(self) -> None:
        level = _build_level()

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "legacy.json"
            save_project(project_path, level.index, [level])
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            payload["levels"][0].pop("sloped_ceilings")
            project_path.write_text(json.dumps(payload), encoding="utf-8")

            restored = load_project(project_path).levels[level.index]

        self.assertEqual(restored.sloped_ceilings, [])

    def test_loading_isolates_malformed_duplicate_and_cross_level_slopes(
        self,
    ) -> None:
        level = _build_level()
        valid = _build_slope()
        duplicate = valid.to_dict() | {"angle_degrees": 55.0}
        duplicate_wall = valid.to_dict() | {
            "slope_id": "e" * 32,
            "angle_degrees": 42.0,
        }
        malformed = valid.to_dict() | {
            "slope_id": "c" * 32,
            "depth_meters": -1.0,
        }
        cross_level = valid.to_dict() | {
            "slope_id": "d" * 32,
            "wall_surface_id": "level:3/wall:1:2",
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "invalid-slopes.json"
            save_project(project_path, level.index, [level])
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            payload["levels"][0]["sloped_ceilings"] = [
                valid.to_dict(),
                duplicate,
                duplicate_wall,
                malformed,
                cross_level,
                None,
            ]
            project_path.write_text(json.dumps(payload), encoding="utf-8")

            restored = load_project(project_path).levels[level.index]

        self.assertEqual(restored.sloped_ceilings, [valid])

    def test_loading_keeps_disjoint_same_wall_slopes_and_drops_overlap(
        self,
    ) -> None:
        level = _build_level()
        first = _build_slope(start_ratio=0.0, end_ratio=0.3)
        second = _build_slope(
            slope_id="b" * 32,
            start_ratio=0.55,
            end_ratio=1.0,
        )
        overlapping = _build_slope(
            slope_id="c" * 32,
            start_ratio=0.2,
            end_ratio=0.7,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "same-wall-slopes.json"
            save_project(project_path, level.index, [level])
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            payload["levels"][0]["sloped_ceilings"] = [
                first.to_dict(),
                overlapping.to_dict(),
                second.to_dict(),
            ]
            project_path.write_text(json.dumps(payload), encoding="utf-8")

            restored = load_project(project_path).levels[level.index]

        self.assertEqual(restored.sloped_ceilings, [first, second])

    def test_non_collection_slope_payload_is_ignored(self) -> None:
        level = _build_level()

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "bad-list.json"
            save_project(project_path, level.index, [level])
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            payload["levels"][0]["sloped_ceilings"] = {"broken": True}
            project_path.write_text(json.dumps(payload), encoding="utf-8")

            restored = load_project(project_path).levels[level.index]

        self.assertEqual(restored.sloped_ceilings, [])


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
