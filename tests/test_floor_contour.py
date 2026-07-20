# ### Environment setup ###
from __future__ import annotations

import json
import os
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np
import trimesh

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ### Imports ###
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QScrollArea

from housemaker.blueprint_canvas import BlueprintCanvas
from housemaker.glb import convert_to_glb
from housemaker.models import (
    DEFAULT_FLOOR_THICKNESS_METERS,
    DEFAULT_LEVEL_OFFSET_METERS,
    DEFAULT_LEVEL_SCALE,
    LevelData,
    RoomData,
    VertexData,
    create_default_levels,
)
from housemaker.project_io import load_project, save_project

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)
_qt_widgets: list[object] = []
HOUSE_CONTOUR_POINTS = [
    (0.0, 0.0),
    (500.0, 0.0),
    (500.0, 400.0),
    (0.0, 400.0),
]


# ### Fixture helpers ###
def _add_wall_segments(
    vertex_data: VertexData,
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
) -> None:
    vertex_by_point = {
        (vertex.x, vertex.y): vertex for vertex in vertex_data.vertices
    }
    for start_point, end_point in segments:
        if start_point not in vertex_by_point:
            vertex_by_point[start_point] = vertex_data.add_vertex(*start_point)
        if end_point not in vertex_by_point:
            vertex_by_point[end_point] = vertex_data.add_vertex(*end_point)

        vertex_data.add_edge(
            vertex_by_point[start_point].id,
            vertex_by_point[end_point].id,
        )


def _add_closed_wall_loop(
    vertex_data: VertexData,
    points: list[tuple[float, float]],
) -> tuple[int, ...]:
    vertices = [vertex_data.add_vertex(*point) for point in points]
    for start_vertex, end_vertex in zip(
        vertices,
        vertices[1:] + vertices[:1],
    ):
        vertex_data.add_edge(start_vertex.id, end_vertex.id)

    return tuple(vertex.id for vertex in vertices)


def _get_vertex_ids_for_points(
    vertex_data: VertexData,
    points: list[tuple[float, float]],
) -> tuple[int, ...]:
    vertex_id_by_point = {
        (vertex.x, vertex.y): vertex.id
        for vertex in vertex_data.vertices
    }
    return tuple(vertex_id_by_point[point] for point in points)


def _build_level_from_segments(
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
    floor_contour_points: list[tuple[float, float]] | None = None,
    floor_thickness_meters: float = DEFAULT_FLOOR_THICKNESS_METERS,
) -> LevelData:
    vertex_data = VertexData()
    _add_wall_segments(vertex_data, segments)
    floor_contour_vertex_ids = (
        _get_vertex_ids_for_points(vertex_data, floor_contour_points)
        if floor_contour_points is not None
        else ()
    )
    return LevelData(
        index=2,
        name="Ground",
        floor_thickness_meters=floor_thickness_meters,
        vertex_data=vertex_data,
        floor_contour_vertex_ids=floor_contour_vertex_ids,
    )


def _build_square_level(
    index: int = 2,
    name: str = "Ground",
    floor_thickness_meters: float = DEFAULT_FLOOR_THICKNESS_METERS,
    with_room: bool = False,
) -> LevelData:
    vertex_data = VertexData()
    outer_vertex_ids = _add_closed_wall_loop(
        vertex_data,
        [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
    )
    rooms: list[RoomData] = []
    if with_room:
        room_center = vertex_data.add_vertex(50.0, 50.0)
        rooms.append(
            RoomData(
                name="Room",
                vertex_ids=outer_vertex_ids,
                center_vertex_id=room_center.id,
                color_rgb=(140, 180, 220),
            )
        )

    return LevelData(
        index=index,
        name=name,
        floor_thickness_meters=floor_thickness_meters,
        vertex_data=vertex_data,
        rooms=rooms,
        floor_contour_vertex_ids=outer_vertex_ids,
    )


def _assert_slab_sides_follow_only_outer_bounds(
    test_case: unittest.TestCase,
    floor_mesh: trimesh.Trimesh,
) -> None:
    vertical_face_mask = np.abs(floor_mesh.face_normals[:, 1]) < 0.1
    side_centers = floor_mesh.triangles_center[vertical_face_mask]
    minimum_x, minimum_z = floor_mesh.bounds[0, [0, 2]]
    maximum_x, maximum_z = floor_mesh.bounds[1, [0, 2]]
    for side_center in side_centers:
        touches_outer_boundary = any(
            np.isclose(side_center[coordinate_index], boundary_value)
            for coordinate_index, boundary_value in (
                (0, minimum_x),
                (0, maximum_x),
                (2, minimum_z),
                (2, maximum_z),
            )
        )
        test_case.assertTrue(touches_outer_boundary)


def _get_scene_world_mesh(
    scene: trimesh.Scene,
    object_name: str,
) -> trimesh.Trimesh:
    transform, geometry_name = scene.graph.get(object_name)
    mesh = scene.geometry[geometry_name].copy()
    mesh.apply_transform(transform)
    return mesh


# ### Tests ###
class FloorContourTests(unittest.TestCase):
    def test_corridor_doorway_produces_one_full_house_slab(self) -> None:
        level = _build_level_from_segments(
            [
                ((0.0, 0.0), (200.0, 0.0)),
                ((300.0, 0.0), (500.0, 0.0)),
                ((500.0, 0.0), (500.0, 400.0)),
                ((500.0, 400.0), (0.0, 400.0)),
                ((0.0, 400.0), (0.0, 0.0)),
                ((200.0, 0.0), (200.0, 400.0)),
                ((300.0, 0.0), (300.0, 400.0)),
                ((200.0, 200.0), (300.0, 200.0)),
            ],
            floor_contour_points=HOUSE_CONTOUR_POINTS,
        )
        model = convert_to_glb([level])
        floor_mesh = model.scene.geometry["l2_ground_floor"]

        self.assertEqual(
            sum(name.endswith("_floor") for name in model.scene.geometry),
            1,
        )
        self.assertTrue(floor_mesh.is_volume)
        self.assertEqual(floor_mesh.euler_number, 2)
        self.assertAlmostEqual(abs(float(floor_mesh.volume)), 24.0)
        self.assertAlmostEqual(float(floor_mesh.area), 170.8)
        np.testing.assert_allclose(
            floor_mesh.bounds,
            np.asarray([[0.0, -0.3, 0.0], [10.0, 0.0, 8.0]]),
        )
        _assert_slab_sides_follow_only_outer_bounds(self, floor_mesh)

    def test_corridor_ending_at_back_wall_is_included_in_slab(self) -> None:
        level = _build_level_from_segments(
            [
                ((0.0, 0.0), (200.0, 0.0)),
                ((300.0, 0.0), (500.0, 0.0)),
                ((500.0, 0.0), (500.0, 400.0)),
                ((500.0, 400.0), (0.0, 400.0)),
                ((0.0, 400.0), (0.0, 0.0)),
                ((200.0, 0.0), (200.0, 200.0)),
                ((300.0, 0.0), (300.0, 200.0)),
                ((200.0, 200.0), (300.0, 200.0)),
            ],
            floor_contour_points=HOUSE_CONTOUR_POINTS,
        )
        floor_mesh = convert_to_glb([level]).scene.geometry[
            "l2_ground_floor"
        ]

        self.assertEqual(floor_mesh.euler_number, 2)
        self.assertAlmostEqual(abs(float(floor_mesh.volume)), 24.0)
        self.assertAlmostEqual(float(floor_mesh.area), 170.8)
        _assert_slab_sides_follow_only_outer_bounds(self, floor_mesh)

    def test_wide_and_multiple_exterior_openings_keep_full_contour(self) -> None:
        cases = {
            "partition_inside_wide_opening": [
                ((0.0, 0.0), (175.0, 0.0)),
                ((325.0, 0.0), (500.0, 0.0)),
                ((500.0, 0.0), (500.0, 400.0)),
                ((500.0, 400.0), (0.0, 400.0)),
                ((0.0, 400.0), (0.0, 0.0)),
                ((0.0, 200.0), (500.0, 200.0)),
                ((250.0, 0.0), (250.0, 200.0)),
            ],
            "two_exterior_openings": [
                ((0.0, 0.0), (175.0, 0.0)),
                ((325.0, 0.0), (500.0, 0.0)),
                ((500.0, 0.0), (500.0, 400.0)),
                ((500.0, 400.0), (275.0, 400.0)),
                ((225.0, 400.0), (0.0, 400.0)),
                ((0.0, 400.0), (0.0, 0.0)),
                ((175.0, 0.0), (175.0, 200.0)),
            ],
        }
        for case_name, segments in cases.items():
            with self.subTest(case_name=case_name):
                floor_mesh = convert_to_glb(
                    [
                        _build_level_from_segments(
                            segments,
                            floor_contour_points=HOUSE_CONTOUR_POINTS,
                        )
                    ]
                ).scene.geometry["l2_ground_floor"]
                self.assertEqual(floor_mesh.euler_number, 2)
                self.assertAlmostEqual(abs(float(floor_mesh.volume)), 24.0)
                _assert_slab_sides_follow_only_outer_bounds(self, floor_mesh)

    def test_opposite_doors_between_enclosed_sides_make_one_slab(self) -> None:
        level = _build_level_from_segments(
            [
                ((0.0, 0.0), (200.0, 0.0)),
                ((300.0, 0.0), (500.0, 0.0)),
                ((500.0, 0.0), (500.0, 400.0)),
                ((500.0, 400.0), (300.0, 400.0)),
                ((200.0, 400.0), (0.0, 400.0)),
                ((0.0, 400.0), (0.0, 0.0)),
                ((200.0, 0.0), (200.0, 400.0)),
                ((300.0, 0.0), (300.0, 400.0)),
            ],
            floor_contour_points=HOUSE_CONTOUR_POINTS,
        )
        floor_mesh = convert_to_glb([level]).scene.geometry[
            "l2_ground_floor"
        ]

        self.assertEqual(floor_mesh.euler_number, 2)
        self.assertAlmostEqual(abs(float(floor_mesh.volume)), 24.0)
        self.assertAlmostEqual(float(floor_mesh.area), 170.8)
        _assert_slab_sides_follow_only_outer_bounds(self, floor_mesh)

    def test_entrance_between_enclosed_and_open_sides_has_floor(self) -> None:
        level = _build_level_from_segments(
            [
                ((0.0, 0.0), (200.0, 0.0)),
                ((300.0, 0.0), (500.0, 0.0)),
                ((500.0, 0.0), (500.0, 400.0)),
                ((500.0, 400.0), (300.0, 400.0)),
                ((200.0, 400.0), (0.0, 400.0)),
                ((0.0, 400.0), (0.0, 0.0)),
                ((200.0, 0.0), (200.0, 400.0)),
            ],
            floor_contour_points=HOUSE_CONTOUR_POINTS,
        )
        floor_mesh = convert_to_glb([level]).scene.geometry[
            "l2_ground_floor"
        ]

        self.assertEqual(floor_mesh.euler_number, 2)
        self.assertAlmostEqual(abs(float(floor_mesh.volume)), 24.0)
        self.assertAlmostEqual(float(floor_mesh.area), 170.8)
        _assert_slab_sides_follow_only_outer_bounds(self, floor_mesh)

    def test_two_open_entrances_with_internal_doorway_share_floor(self) -> None:
        level = _build_level_from_segments(
            [
                ((0.0, 0.0), (100.0, 0.0)),
                ((400.0, 0.0), (500.0, 0.0)),
                ((500.0, 0.0), (500.0, 400.0)),
                ((500.0, 400.0), (0.0, 400.0)),
                ((0.0, 400.0), (0.0, 0.0)),
                ((250.0, 0.0), (250.0, 150.0)),
                ((250.0, 250.0), (250.0, 400.0)),
            ],
            floor_contour_points=HOUSE_CONTOUR_POINTS,
        )
        floor_mesh = convert_to_glb([level]).scene.geometry[
            "l2_ground_floor"
        ]

        self.assertEqual(floor_mesh.euler_number, 2)
        self.assertAlmostEqual(abs(float(floor_mesh.volume)), 24.0)
        self.assertAlmostEqual(float(floor_mesh.area), 170.8)
        _assert_slab_sides_follow_only_outer_bounds(self, floor_mesh)

    def test_two_separate_entrances_with_internal_doorway_share_floor(self) -> None:
        level = _build_level_from_segments(
            [
                ((0.0, 0.0), (100.0, 0.0)),
                ((200.0, 0.0), (300.0, 0.0)),
                ((400.0, 0.0), (500.0, 0.0)),
                ((500.0, 0.0), (500.0, 400.0)),
                ((500.0, 400.0), (0.0, 400.0)),
                ((0.0, 400.0), (0.0, 0.0)),
                ((250.0, 0.0), (250.0, 150.0)),
                ((250.0, 250.0), (250.0, 400.0)),
            ],
            floor_contour_points=HOUSE_CONTOUR_POINTS,
        )
        floor_mesh = convert_to_glb([level]).scene.geometry[
            "l2_ground_floor"
        ]

        self.assertEqual(floor_mesh.euler_number, 2)
        self.assertAlmostEqual(abs(float(floor_mesh.volume)), 24.0)
        self.assertAlmostEqual(float(floor_mesh.area), 170.8)
        _assert_slab_sides_follow_only_outer_bounds(self, floor_mesh)

    def test_two_open_entrances_with_jamb_walls_share_floor(self) -> None:
        level = _build_level_from_segments(
            [
                ((0.0, 0.0), (100.0, 0.0)),
                ((400.0, 0.0), (500.0, 0.0)),
                ((500.0, 0.0), (500.0, 400.0)),
                ((500.0, 400.0), (0.0, 400.0)),
                ((0.0, 400.0), (0.0, 0.0)),
                ((100.0, 0.0), (100.0, 200.0)),
                ((400.0, 0.0), (400.0, 200.0)),
                ((250.0, 0.0), (250.0, 150.0)),
                ((250.0, 250.0), (250.0, 400.0)),
            ],
            floor_contour_points=HOUSE_CONTOUR_POINTS,
        )
        floor_mesh = convert_to_glb([level]).scene.geometry[
            "l2_ground_floor"
        ]

        self.assertEqual(floor_mesh.euler_number, 2)
        self.assertAlmostEqual(abs(float(floor_mesh.volume)), 24.0)
        self.assertAlmostEqual(float(floor_mesh.area), 170.8)
        _assert_slab_sides_follow_only_outer_bounds(self, floor_mesh)

    def test_explicit_concave_contour_excludes_the_outside_recess(self) -> None:
        vertex_data = VertexData()
        contour_vertex_ids = _add_closed_wall_loop(
            vertex_data,
            [
                (0.0, 0.0),
                (500.0, 0.0),
                (500.0, 100.0),
                (100.0, 100.0),
                (100.0, 300.0),
                (500.0, 300.0),
                (500.0, 400.0),
                (0.0, 400.0),
            ],
        )
        level = LevelData(
            index=2,
            name="Ground",
            vertex_data=vertex_data,
            floor_contour_vertex_ids=contour_vertex_ids,
        )
        floor_mesh = convert_to_glb([level]).scene.geometry["l2_ground_floor"]

        self.assertEqual(floor_mesh.euler_number, 2)
        self.assertAlmostEqual(abs(float(floor_mesh.volume)), 14.4)
        self.assertAlmostEqual(float(floor_mesh.area), 111.6)

    def test_open_walls_use_the_explicit_floor_contour(self) -> None:
        level = _build_level_from_segments(
            [
                ((0.0, 0.0), (100.0, 0.0)),
                ((100.0, 0.0), (100.0, 100.0)),
                ((100.0, 100.0), (0.0, 100.0)),
            ],
            floor_contour_points=[
                (0.0, 0.0),
                (100.0, 0.0),
                (100.0, 100.0),
                (0.0, 100.0),
            ],
        )
        floor_mesh = convert_to_glb([level]).scene.geometry[
            "l2_ground_floor"
        ]

        self.assertEqual(floor_mesh.euler_number, 2)
        self.assertAlmostEqual(abs(float(floor_mesh.volume)), 1.2)
        np.testing.assert_allclose(
            floor_mesh.bounds,
            np.asarray([[0.0, -0.3, 0.0], [2.0, 0.0, 2.0]]),
        )

    def test_level_without_an_explicit_contour_has_no_floor(self) -> None:
        level = _build_level_from_segments(
            [
                ((0.0, 0.0), (100.0, 0.0)),
                ((100.0, 0.0), (100.0, 100.0)),
                ((100.0, 100.0), (0.0, 100.0)),
            ]
        )

        model = convert_to_glb([level])

        self.assertNotIn("l2_ground_floor", model.scene.geometry)

    def test_internal_cells_do_not_change_the_explicit_contour(self) -> None:
        vertex_data = VertexData()
        _add_wall_segments(
            vertex_data,
            [
                ((0.0, 0.0), (500.0, 0.0)),
                ((500.0, 0.0), (500.0, 400.0)),
                ((0.0, 400.0), (0.0, 0.0)),
            ],
        )
        _add_closed_wall_loop(
            vertex_data,
            [
                (100.0, 100.0),
                (200.0, 100.0),
                (200.0, 200.0),
                (100.0, 200.0),
            ],
        )
        level = LevelData(
            index=2,
            name="Ground",
            vertex_data=vertex_data,
            floor_contour_vertex_ids=_get_vertex_ids_for_points(
                vertex_data,
                HOUSE_CONTOUR_POINTS,
            ),
        )
        floor_mesh = convert_to_glb([level]).scene.geometry[
            "l2_ground_floor"
        ]

        self.assertEqual(floor_mesh.euler_number, 2)
        self.assertAlmostEqual(abs(float(floor_mesh.volume)), 24.0)
        np.testing.assert_allclose(
            floor_mesh.bounds,
            np.asarray([[0.0, -0.3, 0.0], [10.0, 0.0, 8.0]]),
        )

    def test_one_explicit_contour_exports_one_floor_object(self) -> None:
        vertex_data = VertexData()
        _add_closed_wall_loop(
            vertex_data,
            [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
        )
        _add_closed_wall_loop(
            vertex_data,
            [(300.0, 0.0), (400.0, 0.0), (400.0, 100.0), (300.0, 100.0)],
        )
        level = LevelData(
            index=2,
            name="Ground",
            vertex_data=vertex_data,
            floor_contour_vertex_ids=_get_vertex_ids_for_points(
                vertex_data,
                [(0.0, 0.0), (400.0, 0.0), (400.0, 100.0), (0.0, 100.0)],
            ),
        )
        model = convert_to_glb([level])
        floor_mesh = model.scene.geometry["l2_ground_floor"]

        self.assertEqual(
            sum(name.endswith("_floor") for name in model.scene.geometry),
            1,
        )
        self.assertEqual(floor_mesh.euler_number, 2)
        self.assertAlmostEqual(abs(float(floor_mesh.volume)), 4.8)
        np.testing.assert_allclose(
            floor_mesh.bounds,
            np.asarray([[0.0, -0.3, 0.0], [8.0, 0.0, 2.0]]),
        )

    def test_room_designation_does_not_change_floor_contour(self) -> None:
        without_room = _build_square_level()
        with_room = _build_square_level(with_room=True)
        plain_floor = convert_to_glb([without_room]).scene.geometry[
            "l2_ground_floor"
        ]
        room_floor = convert_to_glb([with_room]).scene.geometry[
            "l2_ground_floor"
        ]

        np.testing.assert_allclose(room_floor.bounds, plain_floor.bounds)
        self.assertAlmostEqual(
            abs(float(room_floor.volume)),
            abs(float(plain_floor.volume)),
        )

    def test_floor_contour_follows_moved_wall_vertices(self) -> None:
        level = _build_square_level()
        moved_vertex_id = level.floor_contour_vertex_ids[1]

        level.vertex_data.move_vertex(moved_vertex_id, 200.0, 0.0)
        floor_mesh = convert_to_glb([level]).scene.geometry[
            "l2_ground_floor"
        ]

        self.assertAlmostEqual(abs(float(floor_mesh.volume)), 1.8)
        np.testing.assert_allclose(
            floor_mesh.bounds,
            np.asarray([[0.0, -0.3, 0.0], [4.0, 0.0, 2.0]]),
        )

    def test_floor_contour_accepts_either_winding_direction(self) -> None:
        clockwise_level = _build_square_level()
        counterclockwise_level = _build_square_level()
        counterclockwise_level.floor_contour_vertex_ids = tuple(
            reversed(counterclockwise_level.floor_contour_vertex_ids)
        )

        clockwise_floor = convert_to_glb([clockwise_level]).scene.geometry[
            "l2_ground_floor"
        ]
        counterclockwise_floor = convert_to_glb(
            [counterclockwise_level]
        ).scene.geometry["l2_ground_floor"]

        self.assertAlmostEqual(
            abs(float(clockwise_floor.volume)),
            abs(float(counterclockwise_floor.volume)),
        )

    def test_thickness_changes_only_floor_geometry(self) -> None:
        level = _build_square_level(with_room=True)
        non_floor_vertices: dict[float, dict[str, np.ndarray]] = {}

        for thickness_meters in (0.2, 0.8):
            level.floor_thickness_meters = thickness_meters
            model = convert_to_glb([level])
            floor_mesh = model.scene.geometry["l2_ground_floor"]
            np.testing.assert_allclose(
                floor_mesh.bounds[:, 1],
                np.asarray([-thickness_meters, 0.0]),
            )
            non_floor_vertices[thickness_meters] = {
                name: np.asarray(mesh.vertices, dtype=float)
                for name, mesh in model.scene.geometry.items()
                if name != "l2_ground_floor"
            }

        self.assertEqual(
            set(non_floor_vertices[0.2]),
            set(non_floor_vertices[0.8]),
        )
        for object_name in non_floor_vertices[0.2]:
            np.testing.assert_allclose(
                non_floor_vertices[0.2][object_name],
                non_floor_vertices[0.8][object_name],
            )

    def test_each_level_floor_extends_down_from_its_surface(self) -> None:
        levels = [
            _build_square_level(1, "Underground 1", 0.1),
            _build_square_level(2, "Ground", 0.25),
            _build_square_level(3, "Story 1", 0.4),
        ]
        model = convert_to_glb(levels)
        expected_vertical_bounds = {
            "l1_underground_1_floor": (-3.1, -3.0),
            "l1_underground_1": (-3.0, 0.0),
            "l2_ground_floor": (-0.25, 0.0),
            "l2_ground": (0.0, 3.0),
            "l3_story_1_floor": (2.6, 3.0),
            "l3_story_1": (3.0, 6.0),
        }
        for object_name, expected_bounds in expected_vertical_bounds.items():
            np.testing.assert_allclose(
                model.scene.geometry[object_name].bounds[:, 1],
                np.asarray(expected_bounds),
            )

    def test_level_scale_applies_to_every_level_object_and_glb_nodes(
        self,
    ) -> None:
        level = _build_square_level()
        room_vertex_ids = _add_closed_wall_loop(
            level.vertex_data,
            [(25.0, 25.0), (75.0, 25.0), (75.0, 75.0), (25.0, 75.0)],
        )
        room_center = level.vertex_data.add_vertex(50.0, 50.0)
        level.rooms.append(
            RoomData(
                name="Room",
                vertex_ids=room_vertex_ids,
                center_vertex_id=room_center.id,
                color_rgb=(140, 180, 220),
            )
        )
        level.scale = 2.0

        model = convert_to_glb([level])
        np.testing.assert_allclose(
            model.mesh.bounds,
            np.asarray([[-1.0, -3.0, -0.3], [3.0, 1.0, 3.0]]),
        )
        preview_points = np.asarray(
            [
                point
                for preview_wall in model.preview_textured_walls
                for point in (preview_wall.start_point, preview_wall.end_point)
            ]
        )
        np.testing.assert_allclose(
            preview_points[:, :2].min(axis=0),
            np.asarray([0.0, -2.0]),
        )
        np.testing.assert_allclose(
            preview_points[:, :2].max(axis=0),
            np.asarray([2.0, 0.0]),
        )
        self.assertTrue(
            all(
                preview_wall.height_meters == 3.0
                for preview_wall in model.preview_textured_walls
            )
        )

        expected_world_bounds = {
            "l2_ground_floor": np.asarray(
                [[-1.0, -0.3, -1.0], [3.0, 0.0, 3.0]]
            ),
            "l2_ground": np.asarray(
                [[-1.0, 0.0, -1.0], [3.0, 3.0, 3.0]]
            ),
            "l2_ground_room_1": np.asarray(
                [[0.0, 0.0, 0.0], [2.0, 3.0, 2.0]]
            ),
        }
        for object_name, expected_bounds in expected_world_bounds.items():
            transform, _ = model.scene.graph.get(object_name)
            np.testing.assert_allclose(
                np.diag(transform)[:3],
                np.asarray([2.0, 1.0, 2.0]),
            )
            np.testing.assert_allclose(
                _get_scene_world_mesh(model.scene, object_name).bounds,
                expected_bounds,
            )

        loaded_scene = trimesh.load(BytesIO(model.glb_bytes), file_type="glb")
        self.assertIsInstance(loaded_scene, trimesh.Scene)
        for object_name, expected_bounds in expected_world_bounds.items():
            transform, _ = loaded_scene.graph.get(object_name)
            np.testing.assert_allclose(
                np.diag(transform)[:3],
                np.asarray([2.0, 1.0, 2.0]),
            )
            np.testing.assert_allclose(
                _get_scene_world_mesh(loaded_scene, object_name).bounds,
                expected_bounds,
                atol=1e-6,
            )

    def test_level_offsets_apply_to_viewer_preview_and_every_exported_object(
        self,
    ) -> None:
        level = _build_square_level()
        room_vertex_ids = _add_closed_wall_loop(
            level.vertex_data,
            [(25.0, 25.0), (75.0, 25.0), (75.0, 75.0), (25.0, 75.0)],
        )
        room_center = level.vertex_data.add_vertex(50.0, 50.0)
        level.rooms.append(
            RoomData(
                name="Room",
                vertex_ids=room_vertex_ids,
                center_vertex_id=room_center.id,
                color_rgb=(140, 180, 220),
            )
        )
        level.scale = 2.0
        level.offset_x_meters = 1.25
        level.offset_y_meters = -0.75

        model = convert_to_glb([level])
        np.testing.assert_allclose(
            model.mesh.bounds,
            np.asarray([[0.25, -3.75, -0.3], [4.25, 0.25, 3.0]]),
        )

        preview_points = np.asarray(
            [
                point
                for preview_wall in model.preview_textured_walls
                for point in (preview_wall.start_point, preview_wall.end_point)
            ]
        )
        np.testing.assert_allclose(
            preview_points[:, :2].min(axis=0),
            np.asarray([1.25, -2.75]),
        )
        np.testing.assert_allclose(
            preview_points[:, :2].max(axis=0),
            np.asarray([3.25, -0.75]),
        )

        expected_world_bounds = {
            "l2_ground_floor": np.asarray(
                [[0.25, -0.3, -0.25], [4.25, 0.0, 3.75]]
            ),
            "l2_ground": np.asarray(
                [[0.25, 0.0, -0.25], [4.25, 3.0, 3.75]]
            ),
            "l2_ground_room_1": np.asarray(
                [[1.25, 0.0, 0.75], [3.25, 3.0, 2.75]]
            ),
        }
        expected_node_translation = np.asarray([0.25, 0.0, -0.25])
        for object_name, expected_bounds in expected_world_bounds.items():
            transform, _ = model.scene.graph.get(object_name)
            np.testing.assert_allclose(
                np.diag(transform)[:3],
                np.asarray([2.0, 1.0, 2.0]),
            )
            np.testing.assert_allclose(
                transform[:3, 3],
                expected_node_translation,
            )
            np.testing.assert_allclose(
                _get_scene_world_mesh(model.scene, object_name).bounds,
                expected_bounds,
            )

        loaded_scene = trimesh.load(BytesIO(model.glb_bytes), file_type="glb")
        self.assertIsInstance(loaded_scene, trimesh.Scene)
        for object_name, expected_bounds in expected_world_bounds.items():
            transform, _ = loaded_scene.graph.get(object_name)
            np.testing.assert_allclose(
                np.diag(transform)[:3],
                np.asarray([2.0, 1.0, 2.0]),
            )
            np.testing.assert_allclose(
                transform[:3, 3],
                expected_node_translation,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                _get_scene_world_mesh(loaded_scene, object_name).bounds,
                expected_bounds,
                atol=1e-6,
            )

    def test_invalid_level_scale_is_rejected(self) -> None:
        level = _build_square_level()
        for invalid_scale in (0.0, -0.1, float("nan"), float("inf"), True):
            with self.subTest(invalid_scale=invalid_scale):
                level.scale = invalid_scale
                with self.assertRaisesRegex(ValueError, "scale"):
                    convert_to_glb([level])

    def test_invalid_level_offsets_are_rejected(self) -> None:
        level = _build_square_level()
        for attribute_name in ("offset_x_meters", "offset_y_meters"):
            for invalid_offset in (
                float("nan"),
                float("inf"),
                float("-inf"),
                True,
                "not-a-number",
            ):
                with self.subTest(
                    attribute_name=attribute_name,
                    invalid_offset=invalid_offset,
                ):
                    level.offset_x_meters = DEFAULT_LEVEL_OFFSET_METERS
                    level.offset_y_meters = DEFAULT_LEVEL_OFFSET_METERS
                    setattr(level, attribute_name, invalid_offset)
                    with self.assertRaisesRegex(ValueError, "offset"):
                        convert_to_glb([level])

    def test_invalid_floor_thickness_is_rejected(self) -> None:
        level = _build_square_level()
        for invalid_thickness in (0.0, -0.1, float("nan")):
            with self.subTest(invalid_thickness=invalid_thickness):
                level.floor_thickness_meters = invalid_thickness
                with self.assertRaisesRegex(ValueError, "floor thickness"):
                    convert_to_glb([level])

    def test_self_intersecting_floor_contour_is_rejected(self) -> None:
        level = _build_square_level()
        first_id, second_id, third_id, fourth_id = (
            level.floor_contour_vertex_ids
        )
        level.floor_contour_vertex_ids = (
            first_id,
            third_id,
            fourth_id,
            second_id,
        )

        with self.assertRaisesRegex(ValueError, "floor contour is invalid"):
            convert_to_glb([level])

    def test_floor_contour_with_a_missing_vertex_is_rejected(self) -> None:
        level = _build_square_level()
        level.floor_contour_vertex_ids = (
            *level.floor_contour_vertex_ids[:-1],
            999,
        )

        with self.assertRaisesRegex(ValueError, "missing vertex 999"):
            convert_to_glb([level])

    def test_floor_name_survives_glb_round_trip(self) -> None:
        model = convert_to_glb([_build_square_level()])
        loaded_scene = trimesh.load(BytesIO(model.glb_bytes), file_type="glb")

        self.assertIsInstance(loaded_scene, trimesh.Scene)
        self.assertIn("l2_ground_floor", loaded_scene.geometry)
        self.assertIn("l2_ground_floor", loaded_scene.graph.nodes_geometry)

    def test_level_and_floor_settings_persist_and_legacy_projects_use_defaults(
        self,
    ) -> None:
        levels = create_default_levels()
        levels[2].scale = 1.75
        levels[2].offset_x_meters = 1.25
        levels[2].offset_y_meters = -0.75
        levels[2].floor_thickness_meters = 0.37
        levels[2].floor_contour_vertex_ids = _add_closed_wall_loop(
            levels[2].vertex_data,
            [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "floor-project.json"
            save_project(project_path, 2, levels)
            loaded_level = load_project(project_path).levels[2]
            self.assertAlmostEqual(loaded_level.scale, 1.75)
            self.assertAlmostEqual(loaded_level.offset_x_meters, 1.25)
            self.assertAlmostEqual(loaded_level.offset_y_meters, -0.75)
            self.assertAlmostEqual(loaded_level.floor_thickness_meters, 0.37)
            self.assertEqual(
                loaded_level.floor_contour_vertex_ids,
                levels[2].floor_contour_vertex_ids,
            )

            payload = json.loads(project_path.read_text(encoding="utf-8"))
            payload["levels"][2].pop("scale")
            payload["levels"][2].pop("offset_x_meters")
            payload["levels"][2].pop("offset_y_meters")
            payload["levels"][2].pop("floor_thickness_meters")
            payload["levels"][2].pop("floor_contour_vertex_ids")
            project_path.write_text(json.dumps(payload), encoding="utf-8")
            legacy_level = load_project(project_path).levels[2]
            self.assertAlmostEqual(legacy_level.scale, DEFAULT_LEVEL_SCALE)
            self.assertAlmostEqual(
                legacy_level.offset_x_meters,
                DEFAULT_LEVEL_OFFSET_METERS,
            )
            self.assertAlmostEqual(
                legacy_level.offset_y_meters,
                DEFAULT_LEVEL_OFFSET_METERS,
            )
            self.assertAlmostEqual(
                legacy_level.floor_thickness_meters,
                DEFAULT_FLOOR_THICKNESS_METERS,
            )
            self.assertEqual(legacy_level.floor_contour_vertex_ids, ())

    def test_floor_thickness_control_updates_selected_level(self) -> None:
        from housemaker.main import BlueprintWorkspace

        workspace = BlueprintWorkspace()
        _qt_widgets.append(workspace)
        self.assertAlmostEqual(DEFAULT_FLOOR_THICKNESS_METERS, 0.3)
        self.assertAlmostEqual(workspace.floor_thickness_spinbox.value(), 0.3)
        workspace.floor_thickness_spinbox.setValue(0.65)
        _qt_application.processEvents()

        self.assertAlmostEqual(
            workspace.current_level.floor_thickness_meters,
            0.65,
        )
        self.assertEqual(
            workspace.levels[1].floor_thickness_meters,
            DEFAULT_FLOOR_THICKNESS_METERS,
        )

    def test_level_scale_control_updates_only_the_selected_level(self) -> None:
        from housemaker.main import BlueprintWorkspace

        workspace = BlueprintWorkspace()
        _qt_widgets.append(workspace)
        self.assertAlmostEqual(workspace.level_scale_spinbox.value(), 1.0)
        workspace.level_scale_spinbox.setValue(1.75)
        _qt_application.processEvents()

        self.assertAlmostEqual(workspace.current_level.scale, 1.75)
        self.assertAlmostEqual(
            workspace.levels[1].scale,
            DEFAULT_LEVEL_SCALE,
        )

    def test_level_offset_controls_update_only_the_selected_level(self) -> None:
        from housemaker.main import BlueprintWorkspace

        workspace = BlueprintWorkspace()
        _qt_widgets.append(workspace)
        self.assertAlmostEqual(
            workspace.level_x_offset_spinbox.value(),
            DEFAULT_LEVEL_OFFSET_METERS,
        )
        self.assertAlmostEqual(
            workspace.level_y_offset_spinbox.value(),
            DEFAULT_LEVEL_OFFSET_METERS,
        )

        workspace.level_x_offset_spinbox.setValue(1.25)
        workspace.level_y_offset_spinbox.setValue(-0.75)
        _qt_application.processEvents()

        self.assertAlmostEqual(workspace.current_level.offset_x_meters, 1.25)
        self.assertAlmostEqual(workspace.current_level.offset_y_meters, -0.75)
        self.assertAlmostEqual(
            workspace.levels[1].offset_x_meters,
            DEFAULT_LEVEL_OFFSET_METERS,
        )
        self.assertAlmostEqual(
            workspace.levels[1].offset_y_meters,
            DEFAULT_LEVEL_OFFSET_METERS,
        )

        workspace.levels_list.setCurrentRow(1)
        _qt_application.processEvents()
        self.assertAlmostEqual(
            workspace.level_x_offset_spinbox.value(),
            DEFAULT_LEVEL_OFFSET_METERS,
        )
        self.assertAlmostEqual(
            workspace.level_y_offset_spinbox.value(),
            DEFAULT_LEVEL_OFFSET_METERS,
        )
        workspace.level_x_offset_spinbox.setValue(-2.0)
        workspace.level_y_offset_spinbox.setValue(3.5)
        _qt_application.processEvents()

        self.assertAlmostEqual(workspace.levels[1].offset_x_meters, -2.0)
        self.assertAlmostEqual(workspace.levels[1].offset_y_meters, 3.5)
        self.assertAlmostEqual(workspace.levels[2].offset_x_meters, 1.25)
        self.assertAlmostEqual(workspace.levels[2].offset_y_meters, -0.75)

    def test_level_controls_place_floor_thickness_after_height(self) -> None:
        from housemaker.main import BlueprintWorkspace

        workspace = BlueprintWorkspace()
        _qt_widgets.append(workspace)
        workspace.resize(1600, 900)
        workspace.show()
        _qt_application.processEvents()

        controls = (
            workspace.height_level_spinbox,
            workspace.floor_thickness_spinbox,
            workspace.level_scale_spinbox,
            workspace.level_x_offset_spinbox,
            workspace.level_y_offset_spinbox,
        )
        top_positions = [
            control.mapTo(workspace, QPoint()).y()
            for control in controls
        ]
        self.assertEqual(top_positions, sorted(top_positions))

        workspace.close()

    def test_generals_tab_scrolls_without_crowding_image_controls(self) -> None:
        from housemaker.main import BlueprintWorkspace

        workspace = BlueprintWorkspace()
        _qt_widgets.append(workspace)
        workspace.resize(1600, 900)
        workspace.show()
        _qt_application.processEvents()

        generals_tab = workspace.side_tabs.widget(0)
        self.assertIsInstance(generals_tab, QScrollArea)
        self.assertGreater(generals_tab.verticalScrollBar().maximum(), 0)

        row_widgets = [
            workspace.image_scale_spinbox,
            workspace.image_x_offset_spinbox,
            workspace.image_y_offset_spinbox,
            workspace.include_yes_radio,
        ]
        row_bounds = []
        for widget in row_widgets:
            top = widget.mapTo(workspace, QPoint()).y()
            row_bounds.append((top, top + widget.height()))

        for previous_bounds, next_bounds in zip(row_bounds, row_bounds[1:]):
            self.assertGreaterEqual(next_bounds[0] - previous_bounds[1], 8)

        workspace.close()

    def test_canvas_builds_an_ordered_contour_without_adding_walls(self) -> None:
        vertex_data = VertexData()
        contour_vertex_ids = _add_closed_wall_loop(
            vertex_data,
            [(10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0)],
        )
        canvas = BlueprintCanvas()
        _qt_widgets.append(canvas)
        canvas.resize(500, 500)
        canvas.set_level_data(
            vertex_data=vertex_data,
            rooms=[],
            image_path=None,
            floor_contour_vertex_ids=(),
        )
        canvas.blueprint_image = QImage(
            100,
            100,
            QImage.Format.Format_RGB32,
        )
        canvas.blueprint_image.fill(Qt.GlobalColor.white)
        canvas.show()
        _qt_application.processEvents()

        emitted_contours: list[tuple[int, ...]] = []
        canvas.floor_contour_changed.connect(emitted_contours.append)
        original_edges = tuple(vertex_data.edges)
        canvas.start_floor_contour_designation()
        clicked_vertices = [
            *(vertex_data.get_vertex(vertex_id) for vertex_id in contour_vertex_ids),
            vertex_data.get_vertex(contour_vertex_ids[0]),
        ]
        for vertex in clicked_vertices:
            self.assertIsNotNone(vertex)
            QTest.mouseClick(
                canvas,
                Qt.MouseButton.LeftButton,
                pos=canvas._image_to_widget(vertex.x, vertex.y).toPoint(),
            )
            _qt_application.processEvents()

        self.assertEqual(canvas.floor_contour_vertex_ids, contour_vertex_ids)
        self.assertEqual(emitted_contours, [contour_vertex_ids])
        self.assertEqual(tuple(vertex_data.edges), original_edges)
        canvas.close()


if __name__ == "__main__":
    unittest.main()
