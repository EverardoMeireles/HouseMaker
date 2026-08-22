# ### Imports ###
from __future__ import annotations

import unittest

import numpy as np

from housemaker.models import DoorwayData, LevelData, RoomData, VertexData
from housemaker.surface_geometry import (
    SURFACE_TYPE_CEILING,
    SURFACE_TYPE_FLOOR,
    SURFACE_TYPE_WALL,
    build_fixed_surfaces,
    get_combined_surface_area,
)


# ### Fixture helpers ###
def _build_square_level(*, doorway: bool = False, with_room: bool = True) -> LevelData:
    vertex_data = VertexData()
    boundary_ids = tuple(
        vertex_data.add_vertex(*point).id
        for point in ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0))
    )
    for start_id, end_id in zip(
        boundary_ids,
        (*boundary_ids[1:], boundary_ids[0]),
    ):
        vertex_data.add_edge(start_id, end_id)
    rooms: list[RoomData] = []
    if with_room:
        center = vertex_data.add_vertex(50.0, 50.0)
        rooms.append(
            RoomData(
                name="Living room",
                vertex_ids=boundary_ids,
                center_vertex_id=center.id,
                color_rgb=(140, 180, 220),
            )
        )
    doorways = (
        [
            DoorwayData(
                center_x=50.0,
                center_y=0.0,
                width_meters=0.8,
                height_meters=2.0,
                depth_meters=0.2,
                rotation_degrees=90.0,
            )
        ]
        if doorway
        else []
    )
    return LevelData(
        index=2,
        name="Ground",
        vertex_data=vertex_data,
        rooms=rooms,
        doorways=doorways,
        floor_contour_vertex_ids=boundary_ids,
    )


def _build_concave_u_level() -> LevelData:
    vertex_data = VertexData()
    boundary_ids = tuple(
        vertex_data.add_vertex(*point).id
        for point in (
            (0.0, 0.0),
            (300.0, 0.0),
            (300.0, 300.0),
            (200.0, 300.0),
            (200.0, 100.0),
            (100.0, 100.0),
            (100.0, 300.0),
            (0.0, 300.0),
        )
    )
    for start_id, end_id in zip(
        boundary_ids,
        (*boundary_ids[1:], boundary_ids[0]),
    ):
        vertex_data.add_edge(start_id, end_id)
    center = vertex_data.add_vertex(50.0, 150.0)
    room = RoomData(
        name="U room",
        vertex_ids=boundary_ids,
        center_vertex_id=center.id,
        color_rgb=(140, 180, 220),
    )
    return LevelData(
        index=2,
        name="Ground",
        vertex_data=vertex_data,
        rooms=[room],
        floor_contour_vertex_ids=boundary_ids,
    )


def _surface_by_id(level: LevelData) -> dict[str, object]:
    return {
        surface.surface_id: surface
        for surface in build_fixed_surfaces([level])
    }


# ### Tests ###
class FixedSurfaceGeometryTests(unittest.TestCase):
    def test_room_surfaces_have_stable_ids_types_and_physical_areas(self) -> None:
        surfaces = build_fixed_surfaces([_build_square_level()])

        self.assertEqual(len(surfaces), 6)
        self.assertEqual(
            {surface.surface_type for surface in surfaces},
            {SURFACE_TYPE_WALL, SURFACE_TYPE_FLOOR, SURFACE_TYPE_CEILING},
        )
        self.assertIn("level:2/room:5/floor", {item.surface_id for item in surfaces})
        self.assertIn(
            "level:2/room:5/ceiling",
            {item.surface_id for item in surfaces},
        )
        wall_ids = {
            item.surface_id
            for item in surfaces
            if item.surface_type == SURFACE_TYPE_WALL
        }
        self.assertEqual(
            wall_ids,
            {
                "level:2/room:5/wall:1:2",
                "level:2/room:5/wall:1:4",
                "level:2/room:5/wall:2:3",
                "level:2/room:5/wall:3:4",
            },
        )
        floor = next(
            item for item in surfaces if item.surface_type == SURFACE_TYPE_FLOOR
        )
        ceiling = next(
            item for item in surfaces if item.surface_type == SURFACE_TYPE_CEILING
        )
        self.assertAlmostEqual(floor.area_square_meters, 4.0)
        self.assertAlmostEqual(ceiling.area_square_meters, 4.0)
        self.assertTrue(np.all(floor.mesh.face_normals[:, 2] > 0.0))
        self.assertTrue(np.all(ceiling.mesh.face_normals[:, 2] < 0.0))
        for wall in (
            item for item in surfaces if item.surface_type == SURFACE_TYPE_WALL
        ):
            self.assertAlmostEqual(wall.area_square_meters, 6.0)

    def test_doorway_cut_and_connected_reveals_belong_to_owning_wall(self) -> None:
        surfaces = _surface_by_id(_build_square_level(doorway=True))
        doorway_wall = surfaces["level:2/room:5/wall:1:2"]

        self.assertAlmostEqual(doorway_wall.area_square_meters, 5.36)  # type: ignore[attr-defined]
        self.assertEqual(len(doorway_wall.mesh.faces), 12)  # type: ignore[attr-defined]
        np.testing.assert_allclose(
            doorway_wall.mesh.bounds[:, 1],  # type: ignore[attr-defined]
            np.asarray((-0.1, 0.1)),
            atol=1e-6,
        )
        self.assertAlmostEqual(
            surfaces["level:2/room:5/wall:3:4"].area_square_meters,  # type: ignore[attr-defined]
            6.0,
        )

    def test_combined_area_counts_each_requested_surface_once(self) -> None:
        surfaces = build_fixed_surfaces([_build_square_level()])
        selected_ids = (
            "level:2/room:5/wall:1:2",
            "level:2/room:5/wall:1:4",
            "level:2/room:5/wall:1:2",
            "unknown",
        )

        self.assertAlmostEqual(
            get_combined_surface_area(surfaces, selected_ids),
            12.0,
        )

    def test_level_contour_fallback_exposes_floor_ceiling_and_plain_walls(self) -> None:
        surfaces = build_fixed_surfaces(
            [_build_square_level(with_room=False)]
        )

        self.assertEqual(len(surfaces), 6)
        self.assertIn("level:2/floor", {item.surface_id for item in surfaces})
        self.assertIn("level:2/ceiling", {item.surface_id for item in surfaces})
        self.assertEqual(
            sum(item.surface_type == SURFACE_TYPE_WALL for item in surfaces),
            4,
        )

    def test_level_contour_residual_outside_rooms_is_selectable_without_overlap(
        self,
    ) -> None:
        level = _build_square_level()
        outer_ids = tuple(
            level.vertex_data.add_vertex(*point).id
            for point in (
                (-50.0, -50.0),
                (150.0, -50.0),
                (150.0, 150.0),
                (-50.0, 150.0),
            )
        )
        level.floor_contour_vertex_ids = outer_ids

        surfaces = build_fixed_surfaces([level])
        floor_surfaces = [
            surface
            for surface in surfaces
            if surface.surface_type == SURFACE_TYPE_FLOOR
        ]

        self.assertEqual(len(floor_surfaces), 2)
        self.assertIn("level:2/floor", {item.surface_id for item in floor_surfaces})
        self.assertAlmostEqual(
            sum(surface.area_square_meters for surface in floor_surfaces),
            16.0,
        )

    def test_level_scale_and_offsets_are_reflected_in_surface_geometry(self) -> None:
        level = _build_square_level()
        level.scale = 2.0
        level.offset_x_meters = 1.25
        level.offset_y_meters = -0.75

        surfaces = build_fixed_surfaces([level])
        floor = next(
            item for item in surfaces if item.surface_type == SURFACE_TYPE_FLOOR
        )

        self.assertAlmostEqual(floor.area_square_meters, 16.0)
        np.testing.assert_allclose(
            floor.mesh.bounds,
            np.asarray(((0.25, -3.75, 0.0), (4.25, 0.25, 0.0))),
        )

    def test_concave_room_wall_normals_point_locally_into_the_room(self) -> None:
        surfaces = _surface_by_id(_build_concave_u_level())

        right_inner_wall = surfaces["level:2/room:9/wall:4:5"]
        left_inner_wall = surfaces["level:2/room:9/wall:6:7"]

        self.assertTrue(
            np.all(right_inner_wall.mesh.face_normals[:, 0] > 0.9)  # type: ignore[attr-defined]
        )
        self.assertTrue(
            np.all(left_inner_wall.mesh.face_normals[:, 0] < -0.9)  # type: ignore[attr-defined]
        )


if __name__ == "__main__":
    unittest.main()
