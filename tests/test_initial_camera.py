# ### Imports ###
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from housemaker.doorway_geometry import build_doorway_cross_section_outline
from housemaker.level_coordinates import (
    build_doorway_world_outline_positions,
    build_level_base_z_lookup,
    get_level_world_pivot,
    level_image_to_world_xy,
    level_world_to_image_xy,
)
from housemaker.models import (
    DOORWAY_SHAPE_ARCH,
    DoorwayData,
    LevelData,
    create_default_levels,
)
from housemaker.project_io import load_project, save_project
# ### Outline test helpers ###
def _get_first_depth_face_profile(
    positions: np.ndarray,
) -> np.ndarray:
    """Recover one closed profile from paired extruded outline positions."""

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise AssertionError("Doorway test positions must be shaped (N, 3).")
    if positions.shape[0] % 6 != 0:
        raise AssertionError("An extruded outline must contain three edge sets.")

    profile_edge_count = positions.shape[0] // 6
    face_edges = positions[: profile_edge_count * 2].reshape(
        profile_edge_count,
        2,
        3,
    )
    np.testing.assert_allclose(face_edges[:-1, 1], face_edges[1:, 0])
    return np.vstack((face_edges[0, 0], face_edges[:, 1]))


# ### Coordinate tests ###
class LevelCoordinateTests(unittest.TestCase):
    def test_image_world_round_trip_honors_center_scale_pivot_and_offsets(
        self,
    ) -> None:
        level = LevelData(
            index=2,
            name="Ground",
            scale=1.75,
            offset_x_meters=2.25,
            offset_y_meters=-3.5,
            image_size_pixels=(1000.0, 800.0),
        )
        level.vertex_data.add_vertex(100.0, 200.0)
        level.vertex_data.add_vertex(500.0, 600.0)

        self.assertEqual(get_level_world_pivot(level), (-4.0, 0.0))
        world_point = level_image_to_world_xy(level, 350.0, 275.0)
        self.assertAlmostEqual(world_point[0], 0.0)
        self.assertAlmostEqual(world_point[1], 0.875)

        image_point = level_world_to_image_xy(level, *world_point)
        self.assertAlmostEqual(image_point[0], 350.0)
        self.assertAlmostEqual(image_point[1], 275.0)

    def test_image_world_round_trip_without_image_or_vertices(self) -> None:
        level = LevelData(
            index=2,
            name="Ground",
            scale=2.0,
            offset_x_meters=1.0,
            offset_y_meters=-2.0,
        )

        world_point = level_image_to_world_xy(level, 25.0, -10.0)
        self.assertEqual(world_point, (2.0, -1.6))
        image_point = level_world_to_image_xy(level, *world_point)
        self.assertAlmostEqual(image_point[0], 25.0)
        self.assertAlmostEqual(image_point[1], -10.0)

    def test_level_base_z_lookup_accumulates_above_and_below_ground(self) -> None:
        levels = create_default_levels()
        levels[0].height_meters = 2.0
        levels[1].height_meters = 2.5
        levels[2].height_meters = 3.0
        levels[3].height_meters = 4.0

        base_z = build_level_base_z_lookup(levels)

        self.assertEqual(base_z[0], -4.5)
        self.assertEqual(base_z[1], -2.5)
        self.assertEqual(base_z[2], 0.0)
        self.assertEqual(base_z[3], 3.0)
        self.assertEqual(base_z[4], 7.0)

    def test_doorway_outline_matches_scaled_level_xy_and_unscaled_height(
        self,
    ) -> None:
        ground_level = LevelData(index=2, name="Ground", height_meters=3.0)
        upper_level = LevelData(
            index=3,
            name="Upper",
            scale=2.0,
            offset_x_meters=3.0,
            offset_y_meters=-4.0,
            image_size_pixels=(100.0, 100.0),
        )
        doorway = DoorwayData(
            center_x=50.0,
            center_y=50.0,
            width_meters=0.8,
            height_meters=2.2,
            depth_meters=0.2,
            rotation_degrees=0.0,
        )

        positions = np.asarray(
            build_doorway_world_outline_positions(
                (ground_level, upper_level),
                upper_level,
                doorway,
            ),
            dtype=float,
        )

        self.assertEqual(positions.shape, (24, 3))
        unique_positions = np.unique(positions, axis=0)
        self.assertEqual(unique_positions.shape, (8, 3))
        np.testing.assert_allclose(
            np.min(unique_positions, axis=0),
            (2.8, -4.8, 3.0),
        )
        np.testing.assert_allclose(
            np.max(unique_positions, axis=0),
            (3.2, -3.2, 5.2),
        )
        edge_lengths = np.linalg.norm(
            positions[0::2] - positions[1::2],
            axis=1,
        )
        np.testing.assert_allclose(
            np.sort(edge_lengths),
            np.sort(np.asarray((0.4,) * 4 + (1.6,) * 4 + (2.2,) * 4)),
        )

    def test_arch_doorway_outline_extrudes_the_shared_cross_section(
        self,
    ) -> None:
        level = LevelData(
            index=2,
            name="Ground",
            image_size_pixels=(100.0, 100.0),
        )
        doorway = DoorwayData(
            center_x=50.0,
            center_y=50.0,
            width_meters=0.8,
            height_meters=2.2,
            depth_meters=0.2,
            rotation_degrees=0.0,
            shape=DOORWAY_SHAPE_ARCH,
        )
        cross_section = build_doorway_cross_section_outline(
            doorway.width_meters,
            doorway.height_meters,
            doorway.shape,
            arch_amount=doorway.arch_amount,
        )
        negative_depth_profile = tuple(
            (-0.1, -width_offset, height_offset)
            for width_offset, height_offset in cross_section
        )
        positive_depth_profile = tuple(
            (0.1, -width_offset, height_offset)
            for width_offset, height_offset in cross_section
        )
        expected_positions: list[tuple[float, float, float]] = []
        for profile in (negative_depth_profile, positive_depth_profile):
            for first_position, second_position in zip(profile, profile[1:]):
                expected_positions.extend((first_position, second_position))
        for first_position, second_position in zip(
            negative_depth_profile[:-1],
            positive_depth_profile[:-1],
        ):
            expected_positions.extend((first_position, second_position))

        positions = np.asarray(
            build_doorway_world_outline_positions((level,), level, doorway),
            dtype=float,
        )

        self.assertEqual(
            positions.shape,
            (6 * (len(cross_section) - 1), 3),
        )
        self.assertGreater(positions.shape[0], 24)
        np.testing.assert_allclose(positions, expected_positions, atol=1e-12)
        self.assertAlmostEqual(float(np.min(positions[:, 2])), 0.0)
        self.assertAlmostEqual(
            float(np.max(positions[:, 2])),
            doorway.height_meters,
        )

    def test_arch_amount_changes_smooth_curve_but_preserves_bounds(
        self,
    ) -> None:
        level = LevelData(
            index=2,
            name="Ground",
            image_size_pixels=(100.0, 100.0),
        )

        def build_positions(arch_amount: float) -> np.ndarray:
            doorway = DoorwayData(
                center_x=50.0,
                center_y=50.0,
                width_meters=0.8,
                height_meters=2.2,
                depth_meters=0.2,
                rotation_degrees=0.0,
                shape=DOORWAY_SHAPE_ARCH,
                arch_amount=arch_amount,
            )
            return np.asarray(
                build_doorway_world_outline_positions((level,), level, doorway),
                dtype=float,
            )

        rectangular_positions = np.asarray(
            build_doorway_world_outline_positions(
                (level,),
                level,
                DoorwayData(
                    center_x=50.0,
                    center_y=50.0,
                    width_meters=0.8,
                    height_meters=2.2,
                    depth_meters=0.2,
                    rotation_degrees=0.0,
                ),
            ),
            dtype=float,
        )
        flat_positions = build_positions(0.0)
        shallow_positions = build_positions(0.25)
        full_positions = build_positions(1.0)

        np.testing.assert_allclose(flat_positions, rectangular_positions)
        self.assertEqual(flat_positions.shape, (24, 3))
        for positions in (shallow_positions, full_positions):
            self.assertGreater(positions.shape[0], 24)
            self.assertEqual(positions.shape[0] % 2, 0)
            profile = _get_first_depth_face_profile(positions)
            self.assertAlmostEqual(float(np.ptp(profile[:, 1])), 0.8)
            self.assertAlmostEqual(float(np.max(profile[:, 2])), 2.2)

        shallow_profile = _get_first_depth_face_profile(shallow_positions)
        full_profile = _get_first_depth_face_profile(full_positions)

        def get_spring_height(profile: np.ndarray) -> float:
            half_width = float(np.max(np.abs(profile[:, 1])))
            spring_points = profile[
                np.isclose(np.abs(profile[:, 1]), half_width, atol=1e-9)
            ]
            return float(np.max(spring_points[:, 2]))

        shallow_spring_height = get_spring_height(shallow_profile)
        full_spring_height = get_spring_height(full_profile)
        self.assertGreater(shallow_spring_height, full_spring_height)
        self.assertLess(
            2.2 - shallow_spring_height,
            2.2 - full_spring_height,
        )
        self.assertFalse(np.allclose(shallow_positions, full_positions))

        for profile, spring_height in (
            (shallow_profile, shallow_spring_height),
            (full_profile, full_spring_height),
        ):
            profile_edges = np.stack((profile[:-1], profile[1:]), axis=1)
            curved_edges = profile_edges[
                (np.min(profile_edges[:, :, 2], axis=1) >= spring_height - 1e-9)
                & (np.max(profile_edges[:, :, 2], axis=1) > spring_height + 1e-9)
            ]
            self.assertGreater(curved_edges.shape[0], 2)
            curved_deltas = np.diff(curved_edges, axis=1)[:, 0, :]
            self.assertTrue(np.all(np.abs(curved_deltas[:, 1]) > 1e-9))
            self.assertTrue(np.all(np.abs(curved_deltas[:, 2]) > 1e-9))

    def test_doorway_outline_rejects_missing_levels_and_invalid_dimensions(
        self,
    ) -> None:
        level = LevelData(index=2, name="Ground")
        doorway = DoorwayData(
            center_x=20.0,
            center_y=30.0,
            width_meters=0.0,
            height_meters=2.1,
        )

        with self.assertRaisesRegex(ValueError, "present"):
            build_doorway_world_outline_positions((), level, doorway)
        with self.assertRaisesRegex(ValueError, "width"):
            build_doorway_world_outline_positions((level,), level, doorway)


# ### Removed initial-camera persistence tests ###
class RemovedInitialCameraPersistenceTests(unittest.TestCase):
    def test_projects_omit_and_ignore_the_retired_initial_camera_field(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "project.json"
            save_project(
                project_path,
                current_level_index=2,
                levels=create_default_levels(),
            )
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            self.assertNotIn("initial_first_person_camera", payload)

            payload["initial_first_person_camera"] = {
                "level_index": 2,
                "pose": {
                    "x": 1.0,
                    "y": 2.0,
                    "z": 1.65,
                    "yaw_degrees": 0.0,
                    "pitch_degrees": 0.0,
                    "roll_degrees": 0.0,
                    "fov_degrees": 70.0,
                },
            }
            project_path.write_text(json.dumps(payload), encoding="utf-8")
            project = load_project(project_path)

        self.assertFalse(hasattr(project, "initial_first_person_camera"))


# ### Test runner ###
if __name__ == "__main__":
    unittest.main()
