# ### Imports ###
from __future__ import annotations

import unittest

import numpy as np
import shapely
from shapely import Polygon
from shapely.geometry.base import BaseGeometry

from housemaker.models import (
    DoorwayData,
    LevelData,
    OpenSpaceData,
    RoomData,
    VertexData,
)
from housemaker.surface_geometry import (
    SURFACE_GEOMETRY_EPSILON,
    SURFACE_TYPE_CEILING,
    SURFACE_TYPE_FLOOR,
    SURFACE_TYPE_WALL,
    FixedSurface,
    _build_horizontal_surface,
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
    )


def _build_concave_u_level(
    *,
    with_room: bool = True,
    reversed_edge_indices: frozenset[int] = frozenset(),
) -> LevelData:
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
    for edge_index, (start_id, end_id) in enumerate(
        zip(boundary_ids, (*boundary_ids[1:], boundary_ids[0]))
    ):
        if edge_index in reversed_edge_indices:
            start_id, end_id = end_id, start_id
        vertex_data.add_edge(start_id, end_id)
    rooms: list[RoomData] = []
    if with_room:
        center = vertex_data.add_vertex(50.0, 150.0)
        rooms.append(
            RoomData(
                name="U room",
                vertex_ids=boundary_ids,
                center_vertex_id=center.id,
                color_rgb=(140, 180, 220),
            )
        )
    return LevelData(
        index=2,
        name="Ground",
        vertex_data=vertex_data,
        rooms=rooms,
    )


def _build_partially_room_owned_square_level() -> LevelData:
    """Build one legacy room that owns only half of an outer wall loop."""

    vertex_data = VertexData()
    boundary_ids = tuple(
        vertex_data.add_vertex(*point).id
        for point in (
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (0.0, 100.0),
        )
    )
    for start_id, end_id in (
        (boundary_ids[0], boundary_ids[1]),
        (boundary_ids[1], boundary_ids[2]),
        (boundary_ids[3], boundary_ids[2]),
        (boundary_ids[0], boundary_ids[3]),
    ):
        vertex_data.add_edge(start_id, end_id)
    center = vertex_data.add_vertex(65.0, 35.0)
    legacy_room = RoomData(
        name="Legacy partial room",
        vertex_ids=boundary_ids[:3],
        center_vertex_id=center.id,
        color_rgb=(140, 180, 220),
    )
    return LevelData(
        index=2,
        name="Ground",
        vertex_data=vertex_data,
        rooms=[legacy_room],
    )


def _build_nested_plain_wall_level() -> LevelData:
    """Build mixed-direction inner and outer wall contours without rooms."""

    vertex_data = VertexData()
    boundary_loops = (
        ((0.0, 0.0), (500.0, 0.0), (500.0, 500.0), (0.0, 500.0)),
        ((20.0, 20.0), (480.0, 20.0), (480.0, 480.0), (20.0, 480.0)),
    )
    for loop_index, points in enumerate(boundary_loops):
        vertex_ids = tuple(vertex_data.add_vertex(*point).id for point in points)
        for edge_index, (start_id, end_id) in enumerate(
            zip(vertex_ids, (*vertex_ids[1:], vertex_ids[0]))
        ):
            if loop_index == 1 and edge_index % 2 == 1:
                start_id, end_id = end_id, start_id
            vertex_data.add_edge(start_id, end_id)
    return LevelData(index=2, name="Ground", vertex_data=vertex_data)


def _build_gapped_nested_plain_wall_level() -> LevelData:
    """Build nested contours whose collinear top edges both have one gap."""

    return _build_level_from_segments(
        (
            ((0.0, 0.0), (240.0, 0.0)),
            ((500.0, 0.0), (260.0, 0.0)),
            ((500.0, 0.0), (500.0, 500.0)),
            ((0.0, 500.0), (500.0, 500.0)),
            ((0.0, 500.0), (0.0, 0.0)),
            ((20.0, 20.0), (240.0, 20.0)),
            ((480.0, 20.0), (260.0, 20.0)),
            ((480.0, 480.0), (480.0, 20.0)),
            ((20.0, 480.0), (480.0, 480.0)),
            ((20.0, 20.0), (20.0, 480.0)),
        )
    )


def _build_level_from_segments(
    segments: tuple[
        tuple[tuple[float, float], tuple[float, float]],
        ...,
    ],
) -> LevelData:
    vertex_data = VertexData()
    vertex_by_point = {}
    for start_point, end_point in segments:
        for point in (start_point, end_point):
            if point not in vertex_by_point:
                vertex_by_point[point] = vertex_data.add_vertex(*point)
        vertex_data.add_edge(
            vertex_by_point[start_point].id,
            vertex_by_point[end_point].id,
        )
    return LevelData(index=2, name="Ground", vertex_data=vertex_data)


def _surface_by_id(level: LevelData) -> dict[str, FixedSurface]:
    return {
        surface.surface_id: surface
        for surface in build_fixed_surfaces([level])
    }


# ### Mesh inspection helpers ###
def _get_horizontal_surface_plan_geometry(surface: FixedSurface) -> BaseGeometry:
    """Return one fixed horizontal surface's dissolved plan geometry."""

    mesh = surface.mesh
    return shapely.union_all(
        [Polygon(triangle[:, :2]) for triangle in mesh.triangles]
    )


# ### Tests ###
class FixedSurfaceGeometryTests(unittest.TestCase):
    def test_horizontal_triangulation_discards_microscopic_slivers(self) -> None:
        polygon = Polygon(
            (
                (0.0, 0.0),
                (10.0, 0.0),
                (10.0, 1e-9),
                (10.0, 10.0),
                (0.0, 10.0),
            )
        )

        surface = _build_horizontal_surface(
            polygon,
            "level:2/floor",
            SURFACE_TYPE_FLOOR,
            2,
            None,
            0.3,
            True,
        )

        self.assertIsNotNone(surface)
        assert surface is not None
        self.assertEqual(len(surface.mesh.faces), 2)
        self.assertTrue(
            np.all(
                np.asarray(surface.mesh.area_faces, dtype=float)
                > SURFACE_GEOMETRY_EPSILON
            )
        )
        self.assertAlmostEqual(float(surface.mesh.area), polygon.area)

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
        lintel_normals = doorway_wall.mesh.face_normals[  # type: ignore[attr-defined]
            np.abs(doorway_wall.mesh.face_normals[:, 2]) > 0.9  # type: ignore[attr-defined]
        ]
        self.assertEqual(len(lintel_normals), 2)
        self.assertTrue(np.all(lintel_normals[:, 2] < -0.9))
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

    def test_plain_closed_wall_loop_exposes_walls_floor_and_ceiling(
        self,
    ) -> None:
        surfaces = build_fixed_surfaces(
            [_build_square_level(with_room=False)]
        )

        self.assertEqual(len(surfaces), 6)
        self.assertEqual(
            sum(item.surface_type == SURFACE_TYPE_WALL for item in surfaces),
            4,
        )
        self.assertIn("level:2/floor", {item.surface_id for item in surfaces})
        self.assertIn("level:2/ceiling", {item.surface_id for item in surfaces})
        floor = next(
            item for item in surfaces if item.surface_type == SURFACE_TYPE_FLOOR
        )
        ceiling = next(
            item
            for item in surfaces
            if item.surface_type == SURFACE_TYPE_CEILING
        )
        self.assertAlmostEqual(floor.area_square_meters, 4.0)
        self.assertAlmostEqual(ceiling.area_square_meters, 4.0)
        self.assertTrue(np.all(floor.mesh.face_normals[:, 2] > 0.0))
        self.assertTrue(np.all(ceiling.mesh.face_normals[:, 2] < 0.0))
        self.assertEqual(
            {
                surface.surface_id
                for surface in surfaces
                if surface.surface_type == SURFACE_TYPE_WALL
            },
            {
                "level:2/wall:1:2",
                "level:2/wall:1:4",
                "level:2/wall:2:3",
                "level:2/wall:3:4",
            },
        )

    def test_floor_and_ceiling_keep_all_closed_components_and_ignore_open_walls(
        self,
    ) -> None:
        level = _build_level_from_segments(
            (
                ((0.0, 0.0), (200.0, 0.0)),
                ((200.0, 0.0), (200.0, 200.0)),
                ((200.0, 200.0), (0.0, 200.0)),
                ((0.0, 200.0), (0.0, 0.0)),
                ((200.0, 200.0), (200.0, 201.0)),
                ((400.0, 0.0), (450.0, 0.0)),
                ((450.0, 0.0), (450.0, 50.0)),
                ((450.0, 50.0), (400.0, 50.0)),
                ((400.0, 50.0), (400.0, 0.0)),
                ((600.0, 0.0), (700.0, 0.0)),
                ((700.0, 0.0), (700.0, 100.0)),
            )
        )

        surfaces = _surface_by_id(level)

        floor = surfaces["level:2/floor"]
        ceiling = surfaces["level:2/ceiling"]
        floor_geometry = _get_horizontal_surface_plan_geometry(floor)
        ceiling_geometry = _get_horizontal_surface_plan_geometry(ceiling)
        for surface, geometry in (
            (floor, floor_geometry),
            (ceiling, ceiling_geometry),
        ):
            self.assertAlmostEqual(surface.area_square_meters, 17.0)
            self.assertAlmostEqual(float(geometry.area), 17.0)
            self.assertEqual(
                sum(
                    isinstance(component, Polygon)
                    for component in shapely.get_parts(geometry)
                ),
                2,
            )
            self.assertFalse(geometry.covers(shapely.Point(6.0, -0.5)))
        self.assertAlmostEqual(
            float(floor_geometry.symmetric_difference(ceiling_geometry).area),
            0.0,
        )

    def test_orientation_fallback_does_not_create_horizontal_surfaces(
        self,
    ) -> None:
        level = _build_level_from_segments(
            (
                ((10.0, 0.0), (90.0, 0.0)),
                ((100.0, 10.0), (100.0, 90.0)),
                ((90.0, 100.0), (10.0, 100.0)),
                ((0.0, 90.0), (0.0, 10.0)),
            )
        )

        surfaces = _surface_by_id(level)

        self.assertNotIn("level:2/floor", surfaces)
        self.assertNotIn("level:2/ceiling", surfaces)
        self.assertEqual(
            sum(
                surface.surface_type == SURFACE_TYPE_WALL
                for surface in surfaces.values()
            ),
            4,
        )

    def test_gapped_outer_walls_share_one_floor_and_ceiling_footprint(
        self,
    ) -> None:
        level = _build_level_from_segments(
            (
                ((0.0, 0.0), (210.0, 0.0)),
                ((250.0, 0.0), (500.0, 0.0)),
                ((500.0, 0.0), (500.0, 170.0)),
                ((500.0, 200.0), (500.0, 400.0)),
                ((500.0, 400.0), (320.0, 400.0)),
                ((270.0, 400.0), (0.0, 400.0)),
                ((0.0, 400.0), (0.0, 260.0)),
                ((0.0, 220.0), (0.0, 0.0)),
                ((300.0, 80.0), (380.0, 80.0)),
                ((380.0, 80.0), (380.0, 140.0)),
                ((380.0, 140.0), (300.0, 140.0)),
                ((300.0, 140.0), (300.0, 80.0)),
            )
        )

        surfaces = build_fixed_surfaces([level])
        horizontal_surfaces = {
            surface.surface_type: surface
            for surface in surfaces
            if surface.surface_type in (
                SURFACE_TYPE_FLOOR,
                SURFACE_TYPE_CEILING,
            )
        }

        self.assertEqual(
            set(horizontal_surfaces),
            {SURFACE_TYPE_FLOOR, SURFACE_TYPE_CEILING},
        )
        for surface in horizontal_surfaces.values():
            self.assertAlmostEqual(surface.area_square_meters, 80.0)
            np.testing.assert_allclose(
                surface.mesh.bounds[:, :2],
                np.asarray(((0.0, -8.0), (10.0, 0.0))),
                atol=1e-9,
            )

    def test_mixed_room_and_reversed_plain_edges_share_inferred_interior(
        self,
    ) -> None:
        surfaces = _surface_by_id(_build_partially_room_owned_square_level())

        self.assertIn("level:2/room:5/wall:1:2", surfaces)
        self.assertIn("level:2/room:5/wall:2:3", surfaces)
        expected_plain_normals = {
            "level:2/wall:3:4": (0.0, 1.0, 0.0),
            "level:2/wall:1:4": (1.0, 0.0, 0.0),
        }
        for surface_id, expected_normal in expected_plain_normals.items():
            np.testing.assert_allclose(
                surfaces[surface_id].mesh.face_normals,  # type: ignore[attr-defined]
                np.tile(expected_normal, (2, 1)),
                atol=1e-7,
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
            np.asarray(
                (
                    (0.25, -3.75, level.floor_thickness_meters),
                    (4.25, 0.25, level.floor_thickness_meters),
                )
            ),
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

    def test_plain_concave_wall_normals_ignore_mixed_edge_directions(self) -> None:
        surfaces = _surface_by_id(
            _build_concave_u_level(
                with_room=False,
                reversed_edge_indices=frozenset((0, 3, 6)),
            )
        )
        expected_normals = {
            "level:2/wall:1:2": (0.0, -1.0, 0.0),
            "level:2/wall:2:3": (-1.0, 0.0, 0.0),
            "level:2/wall:3:4": (0.0, 1.0, 0.0),
            "level:2/wall:4:5": (1.0, 0.0, 0.0),
            "level:2/wall:5:6": (0.0, 1.0, 0.0),
            "level:2/wall:6:7": (-1.0, 0.0, 0.0),
            "level:2/wall:7:8": (0.0, 1.0, 0.0),
            "level:2/wall:1:8": (1.0, 0.0, 0.0),
        }

        for surface_id, expected_normal in expected_normals.items():
            np.testing.assert_allclose(
                surfaces[surface_id].mesh.face_normals,  # type: ignore[attr-defined]
                np.tile(expected_normal, (2, 1)),
                atol=1e-7,
            )

    def test_nested_plain_wall_normals_face_the_open_interior(self) -> None:
        surfaces = _surface_by_id(_build_nested_plain_wall_level())
        expected_inner_normals = {
            "level:2/wall:5:6": (0.0, -1.0, 0.0),
            "level:2/wall:6:7": (-1.0, 0.0, 0.0),
            "level:2/wall:7:8": (0.0, 1.0, 0.0),
            "level:2/wall:5:8": (1.0, 0.0, 0.0),
        }

        for surface_id, expected_normal in expected_inner_normals.items():
            np.testing.assert_allclose(
                surfaces[surface_id].mesh.face_normals,  # type: ignore[attr-defined]
                np.tile(expected_normal, (2, 1)),
                atol=1e-7,
            )

    def test_nested_plain_outer_walls_face_away_from_the_wall_cavity(self) -> None:
        surfaces = _surface_by_id(_build_nested_plain_wall_level())
        expected_outer_normals = {
            "level:2/wall:1:2": (0.0, 1.0, 0.0),
            "level:2/wall:2:3": (1.0, 0.0, 0.0),
            "level:2/wall:3:4": (0.0, -1.0, 0.0),
            "level:2/wall:1:4": (-1.0, 0.0, 0.0),
        }

        for surface_id, expected_normal in expected_outer_normals.items():
            np.testing.assert_allclose(
                surfaces[surface_id].mesh.face_normals,  # type: ignore[attr-defined]
                np.tile(expected_normal, (2, 1)),
                atol=1e-7,
            )

    def test_gapped_nested_plain_walls_preserve_the_wall_cavity(self) -> None:
        surfaces = _surface_by_id(_build_gapped_nested_plain_wall_level())
        expected_normals = {
            "level:2/wall:1:2": (0.0, 1.0, 0.0),
            "level:2/wall:3:4": (0.0, 1.0, 0.0),
            "level:2/wall:3:5": (1.0, 0.0, 0.0),
            "level:2/wall:5:6": (0.0, -1.0, 0.0),
            "level:2/wall:1:6": (-1.0, 0.0, 0.0),
            "level:2/wall:7:8": (0.0, -1.0, 0.0),
            "level:2/wall:9:10": (0.0, -1.0, 0.0),
            "level:2/wall:9:11": (-1.0, 0.0, 0.0),
            "level:2/wall:11:12": (0.0, 1.0, 0.0),
            "level:2/wall:7:12": (1.0, 0.0, 0.0),
        }

        self.assertEqual(
            set(surfaces), {*expected_normals, "level:2/floor", "level:2/ceiling"}
        )
        for surface_id, expected_normal in expected_normals.items():
            np.testing.assert_allclose(
                surfaces[surface_id].mesh.face_normals,  # type: ignore[attr-defined]
                np.tile(expected_normal, (2, 1)),
                atol=1e-7,
            )

    def test_matching_open_space_marks_a_nested_annulus_as_exposed(self) -> None:
        level = _build_nested_plain_wall_level()
        level.open_spaces.append(OpenSpaceData("courtyard", 20.0, 20.0, 480.0, 480.0))
        surfaces = _surface_by_id(level)
        expected_normals = {
            "level:2/wall:1:2": (0.0, -1.0, 0.0),
            "level:2/wall:2:3": (-1.0, 0.0, 0.0),
            "level:2/wall:3:4": (0.0, 1.0, 0.0),
            "level:2/wall:1:4": (1.0, 0.0, 0.0),
            "level:2/wall:5:6": (0.0, 1.0, 0.0),
            "level:2/wall:6:7": (1.0, 0.0, 0.0),
            "level:2/wall:7:8": (0.0, -1.0, 0.0),
            "level:2/wall:5:8": (-1.0, 0.0, 0.0),
        }

        for surface_id, expected_normal in expected_normals.items():
            np.testing.assert_allclose(
                surfaces[surface_id].mesh.face_normals,  # type: ignore[attr-defined]
                np.tile(expected_normal, (2, 1)),
                atol=1e-7,
            )

    def test_small_open_space_does_not_flip_an_ordinary_room(self) -> None:
        level = _build_square_level(with_room=False)
        level.open_spaces.append(OpenSpaceData("small opening", 40.0, 40.0, 60.0, 60.0))

        wall = _surface_by_id(level)["level:2/wall:1:2"]

        np.testing.assert_allclose(
            wall.mesh.face_normals,  # type: ignore[attr-defined]
            np.tile((0.0, -1.0, 0.0), (2, 1)),
            atol=1e-7,
        )

    def test_gapped_plain_wall_normals_ignore_mixed_edge_directions(self) -> None:
        level = _build_level_from_segments(
            (
                ((0.0, 0.0), (210.0, 0.0)),
                ((500.0, 0.0), (250.0, 0.0)),
                ((500.0, 0.0), (500.0, 170.0)),
                ((500.0, 400.0), (500.0, 200.0)),
                ((500.0, 400.0), (320.0, 400.0)),
                ((0.0, 400.0), (270.0, 400.0)),
                ((0.0, 400.0), (0.0, 260.0)),
                ((0.0, 0.0), (0.0, 220.0)),
            )
        )
        wall_surfaces = [
            surface
            for surface in build_fixed_surfaces([level])
            if surface.surface_type == SURFACE_TYPE_WALL
        ]
        level_center = np.asarray((5.0, -4.0), dtype=float)

        self.assertEqual(len(wall_surfaces), 8)
        for surface in wall_surfaces:
            wall_midpoint = np.mean(
                np.asarray(surface.mesh.vertices, dtype=float)[:, :2],
                axis=0,
            )
            direction_to_center = level_center - wall_midpoint
            direction_to_center /= np.linalg.norm(direction_to_center)
            facing_alignment = np.dot(
                np.asarray(surface.mesh.face_normals, dtype=float)[:, :2],
                direction_to_center,
            )
            self.assertTrue(np.all(facing_alignment > 0.5))


if __name__ == "__main__":
    unittest.main()
