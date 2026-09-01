# ### Imports ###
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from housemaker.glb import (
    _build_level_doorway_reveal_mesh,
    _build_wall_openings,
)
from housemaker.level_coordinates import build_doorway_world_outline_positions
from housemaker.models import (
    DEFAULT_DOORWAY_BOTTOM_HEIGHT_METERS,
    DOORWAY_SHAPE_ARCH,
    DoorwayData,
    LevelData,
    VertexData,
    create_default_levels,
)
from housemaker.project_io import load_project, save_project
from housemaker.surface_geometry import build_fixed_surfaces


# ### Fixture helpers ###
def _add_wall(
    vertex_data: VertexData,
    start: tuple[float, float],
    end: tuple[float, float],
) -> None:
    start_vertex = vertex_data.add_vertex(*start)
    end_vertex = vertex_data.add_vertex(*end)
    vertex_data.add_edge(start_vertex.id, end_vertex.id)


def _build_parallel_wall_level(bottom_height_meters: float) -> LevelData:
    vertex_data = VertexData()
    _add_wall(vertex_data, (10.0, 45.0), (90.0, 45.0))
    _add_wall(vertex_data, (10.0, 55.0), (90.0, 55.0))
    return LevelData(
        index=2,
        name="Ground",
        vertex_data=vertex_data,
        doorways=[
            DoorwayData(
                center_x=50.0,
                center_y=50.0,
                width_meters=0.9,
                height_meters=1.8,
                depth_meters=0.2,
                rotation_degrees=90.0,
                bottom_height_meters=bottom_height_meters,
            )
        ],
    )


def _mesh_has_horizontal_face_at_height(mesh: object, height: float) -> bool:
    triangles = np.asarray(getattr(mesh, "triangles"), dtype=float)
    return bool(
        np.any(
            np.all(
                np.isclose(triangles[:, :, 2], height, atol=1e-8),
                axis=1,
            )
        )
    )


# ### Tests ###
class DoorwayBottomHeightTests(unittest.TestCase):
    def test_bottom_height_is_appended_for_positional_compatibility(self) -> None:
        doorway = DoorwayData(
            10.0,
            20.0,
            0.9,
            2.1,
            0.2,
            90.0,
            DOORWAY_SHAPE_ARCH,
            0.4,
        )

        self.assertEqual(
            doorway.bottom_height_meters,
            DEFAULT_DOORWAY_BOTTOM_HEIGHT_METERS,
        )

    def test_bottom_height_rejects_non_finite_or_negative_values(self) -> None:
        for value in (-0.01, float("inf"), float("nan"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                DoorwayData(
                    center_x=0.0,
                    center_y=0.0,
                    width_meters=0.9,
                    height_meters=2.1,
                    bottom_height_meters=value,
                )

    def test_project_round_trip_and_legacy_default(self) -> None:
        levels = create_default_levels()
        levels[2].doorways = [
            DoorwayData(
                center_x=40.0,
                center_y=30.0,
                width_meters=0.8,
                height_meters=1.7,
                bottom_height_meters=0.45,
            )
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "raised-doorway.json"
            save_project(project_path, current_level_index=2, levels=levels)
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            raw_doorway = payload["levels"][2]["doorways"][0]
            self.assertEqual(raw_doorway["bottom_height_meters"], 0.45)
            self.assertEqual(
                load_project(project_path).levels[2].doorways[0],
                levels[2].doorways[0],
            )

            raw_doorway.pop("bottom_height_meters")
            project_path.write_text(json.dumps(payload), encoding="utf-8")
            legacy_doorway = load_project(project_path).levels[2].doorways[0]
            self.assertEqual(
                legacy_doorway.bottom_height_meters,
                DEFAULT_DOORWAY_BOTTOM_HEIGHT_METERS,
            )

    def test_world_outline_starts_at_the_persisted_bottom_height(self) -> None:
        lower_level = LevelData(index=2, name="Ground", height_meters=3.0)
        upper_level = LevelData(index=3, name="Upper")
        doorway = DoorwayData(
            center_x=0.0,
            center_y=0.0,
            width_meters=0.8,
            height_meters=1.6,
            bottom_height_meters=0.55,
        )

        positions = np.asarray(
            build_doorway_world_outline_positions(
                (lower_level, upper_level),
                upper_level,
                doorway,
            ),
            dtype=float,
        )

        self.assertAlmostEqual(float(np.min(positions[:, 2])), 3.55)
        self.assertAlmostEqual(float(np.max(positions[:, 2])), 5.15)

    def test_raised_doorway_is_cut_at_height_and_gets_a_sealed_sill(self) -> None:
        raised_level = _build_parallel_wall_level(0.4)
        floor_level = _build_parallel_wall_level(0.0)
        opening = _build_wall_openings(raised_level.doorways)[0]

        self.assertAlmostEqual(opening.bottom_height_meters, 0.4)
        self.assertAlmostEqual(opening.top_height_meters, 2.2)

        raised_reveal = _build_level_doorway_reveal_mesh(
            raised_level,
            base_z_meters=0.0,
            blueprint_size_pixels=None,
            room_vertex_sets=(),
        )
        floor_reveal = _build_level_doorway_reveal_mesh(
            floor_level,
            base_z_meters=0.0,
            blueprint_size_pixels=None,
            room_vertex_sets=(),
        )
        self.assertIsNotNone(raised_reveal)
        self.assertIsNotNone(floor_reveal)
        assert raised_reveal is not None
        assert floor_reveal is not None
        self.assertTrue(_mesh_has_horizontal_face_at_height(raised_reveal, 0.4))
        self.assertFalse(_mesh_has_horizontal_face_at_height(floor_reveal, 0.0))

        raised_surfaces = build_fixed_surfaces([raised_level])
        self.assertTrue(
            any(
                _mesh_has_horizontal_face_at_height(surface.mesh, 0.4)
                for surface in raised_surfaces
            )
        )


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
