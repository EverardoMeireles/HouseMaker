# ### Imports ###
from __future__ import annotations

from io import BytesIO
import unittest

import numpy as np
import shapely
import trimesh
from shapely import Polygon

from housemaker.glb import convert_to_glb
from housemaker.models import LevelData, VertexData


# ### Fixture helpers ###
def _add_closed_wall_loop(
    vertex_data: VertexData,
    points: tuple[tuple[float, float], ...],
) -> tuple[int, ...]:
    vertex_ids = tuple(
        vertex_data.add_vertex(*point).id
        for point in points
    )
    for edge_index, (start_id, end_id) in enumerate(
        zip(vertex_ids, (*vertex_ids[1:], vertex_ids[0]))
    ):
        if edge_index % 2:
            start_id, end_id = end_id, start_id
        vertex_data.add_edge(start_id, end_id)
    return vertex_ids


def _build_closed_level(
    points: tuple[tuple[float, float], ...],
    *,
    floor_thickness_meters: float = 0.3,
) -> LevelData:
    vertex_data = VertexData()
    _add_closed_wall_loop(vertex_data, points)
    return LevelData(
        index=2,
        name="Ground",
        floor_thickness_meters=floor_thickness_meters,
        vertex_data=vertex_data,
    )


def _build_square_level(*, floor_thickness_meters: float = 0.3) -> LevelData:
    return _build_closed_level(
        (
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (0.0, 100.0),
        ),
        floor_thickness_meters=floor_thickness_meters,
    )


def _build_level_from_segments(
    segments: tuple[
        tuple[tuple[float, float], tuple[float, float]],
        ...,
    ],
) -> LevelData:
    vertex_data = VertexData()
    vertex_by_point = {}
    for edge_index, (start_point, end_point) in enumerate(segments):
        for point in (start_point, end_point):
            if point not in vertex_by_point:
                vertex_by_point[point] = vertex_data.add_vertex(*point)
        start = vertex_by_point[start_point]
        end = vertex_by_point[end_point]
        if edge_index % 2:
            start, end = end, start
        vertex_data.add_edge(start.id, end.id)
    return LevelData(
        index=2,
        name="Ground",
        vertex_data=vertex_data,
    )


# ### Mesh inspection helpers ###
def _get_horizontal_cap_metrics(
    mesh: trimesh.Trimesh,
    normal_y_sign: float,
) -> tuple[float, float, int]:
    face_mask = mesh.face_normals[:, 1] * normal_y_sign > 0.9
    polygons = [
        Polygon(triangle[:, (0, 2)])
        for triangle in mesh.triangles[face_mask]
    ]
    cap_geometry = shapely.union_all(polygons)
    component_count = sum(
        isinstance(component, Polygon)
        for component in shapely.get_parts(cap_geometry)
    )
    return (
        float(np.sum(mesh.area_faces[face_mask])),
        float(cap_geometry.area),
        component_count,
    )


def _get_interior_vertical_face_centers(
    mesh: trimesh.Trimesh,
) -> np.ndarray:
    side_face_mask = np.abs(mesh.face_normals[:, 1]) < 0.1
    side_centers = mesh.triangles_center[side_face_mask]
    minimum_x, minimum_z = mesh.bounds[0, (0, 2)]
    maximum_x, maximum_z = mesh.bounds[1, (0, 2)]
    return np.asarray(
        [
            center
            for center in side_centers
            if not any(
                np.isclose(center[coordinate_index], boundary_value)
                for coordinate_index, boundary_value in (
                    (0, minimum_x),
                    (0, maximum_x),
                    (2, minimum_z),
                    (2, maximum_z),
                )
            )
        ],
        dtype=float,
    )


# ### Automatic floor geometry tests ###
class AutomaticFloorGeometryTests(unittest.TestCase):
    def test_combined_outermost_wall_area_preserves_concave_footprint(
        self,
    ) -> None:
        level = _build_closed_level(
            (
                (0.0, 0.0),
                (200.0, 0.0),
                (200.0, 100.0),
                (100.0, 100.0),
                (100.0, 200.0),
                (0.0, 200.0),
                (0.0, 100.0),
            )
        )
        level.vertex_data.add_edge(4, 7)

        model = convert_to_glb([level])

        self.assertIn("l2_ground_floor", model.scene.geometry)
        floor_mesh = model.scene.geometry["l2_ground_floor"]
        self.assertTrue(floor_mesh.is_volume)
        self.assertAlmostEqual(abs(float(floor_mesh.volume)), 3.6)
        np.testing.assert_allclose(
            floor_mesh.bounds,
            np.asarray(((0.0, 0.0, 0.0), (4.0, 0.3, 4.0))),
            atol=1e-9,
        )

    def test_floor_thickness_raises_geometry_resting_on_the_slab(self) -> None:
        thin_model = convert_to_glb(
            [_build_square_level(floor_thickness_meters=0.12)]
        )
        thick_model = convert_to_glb(
            [_build_square_level(floor_thickness_meters=0.65)]
        )

        self.assertEqual(
            set(thin_model.scene.geometry),
            set(thick_model.scene.geometry),
        )
        self.assertIn("l2_ground", thin_model.scene.geometry)
        self.assertIn("l2_ground_floor", thin_model.scene.geometry)
        thickness_delta = 0.65 - 0.12
        for geometry_name in thin_model.scene.geometry:
            if geometry_name == "l2_ground_floor":
                continue
            thin_mesh = thin_model.scene.geometry[geometry_name]
            thick_mesh = thick_model.scene.geometry[geometry_name]
            np.testing.assert_array_equal(thick_mesh.faces, thin_mesh.faces)
            np.testing.assert_allclose(
                thick_mesh.vertices[:, (0, 2)],
                thin_mesh.vertices[:, (0, 2)],
                atol=1e-12,
            )
            np.testing.assert_allclose(
                thick_mesh.vertices[:, 1] - thin_mesh.vertices[:, 1],
                thickness_delta,
                atol=1e-12,
            )

        thin_floor = thin_model.scene.geometry["l2_ground_floor"]
        thick_floor = thick_model.scene.geometry["l2_ground_floor"]
        np.testing.assert_allclose(
            thick_floor.bounds[:, (0, 2)],
            thin_floor.bounds[:, (0, 2)],
            atol=1e-12,
        )
        np.testing.assert_allclose(thin_floor.bounds[:, 1], (0.0, 0.12))
        np.testing.assert_allclose(thick_floor.bounds[:, 1], (0.0, 0.65))

    def test_partition_walls_do_not_create_vertical_slab_seams(self) -> None:
        level = _build_closed_level(
            (
                (0.0, 0.0),
                (100.0, 0.0),
                (200.0, 0.0),
                (200.0, 100.0),
                (100.0, 100.0),
                (0.0, 100.0),
            )
        )
        level.vertex_data.add_edge(2, 5)

        floor_mesh = convert_to_glb([level]).scene.geometry[
            "l2_ground_floor"
        ]

        self.assertGreater(
            sum(np.abs(floor_mesh.face_normals[:, 1]) < 0.1),
            0,
        )
        self.assertEqual(len(_get_interior_vertical_face_centers(floor_mesh)), 0)

    def test_open_interior_walls_leave_one_full_floor_cap(self) -> None:
        level = _build_closed_level(
            (
                (0.0, 0.0),
                (100.0, 0.0),
                (200.0, 0.0),
                (200.0, 100.0),
                (100.0, 100.0),
                (0.0, 100.0),
            ),
            floor_thickness_meters=0.65,
        )
        center = level.vertex_data.add_vertex(100.0, 50.0)
        level.vertex_data.add_edge(2, center.id)
        disconnected_start = level.vertex_data.add_vertex(40.0, 70.0)
        disconnected_end = level.vertex_data.add_vertex(80.0, 70.0)
        level.vertex_data.add_edge(
            disconnected_end.id,
            disconnected_start.id,
        )

        model = convert_to_glb([level])
        floor_mesh = model.scene.geometry["l2_ground_floor"]

        for normal_y_sign in (-1.0, 1.0):
            triangle_area, union_area, component_count = (
                _get_horizontal_cap_metrics(floor_mesh, normal_y_sign)
            )
            self.assertAlmostEqual(triangle_area, 8.0)
            self.assertAlmostEqual(union_area, 8.0)
            self.assertEqual(component_count, 1)
        self.assertEqual(len(_get_interior_vertical_face_centers(floor_mesh)), 0)
        for geometry_name, mesh in model.scene.geometry.items():
            if geometry_name == "l2_ground_floor":
                continue
            self.assertGreaterEqual(float(mesh.bounds[0, 1]), 0.65)

    def test_nested_perimeter_is_filled_without_inner_slab_sides(self) -> None:
        level = _build_closed_level(
            (
                (0.0, 0.0),
                (200.0, 0.0),
                (200.0, 200.0),
                (0.0, 200.0),
            )
        )
        _add_closed_wall_loop(
            level.vertex_data,
            (
                (50.0, 50.0),
                (150.0, 50.0),
                (150.0, 150.0),
                (50.0, 150.0),
            ),
        )

        floor_mesh = convert_to_glb([level]).scene.geometry[
            "l2_ground_floor"
        ]

        for normal_y_sign in (-1.0, 1.0):
            triangle_area, union_area, component_count = (
                _get_horizontal_cap_metrics(floor_mesh, normal_y_sign)
            )
            self.assertAlmostEqual(triangle_area, 16.0)
            self.assertAlmostEqual(union_area, 16.0)
            self.assertEqual(component_count, 1)
        self.assertEqual(len(_get_interior_vertical_face_centers(floor_mesh)), 0)

    def test_duplicated_reversed_perimeter_still_builds_one_slab(self) -> None:
        perimeter = (
            (0.0, 0.0),
            (200.0, 0.0),
            (200.0, 100.0),
            (0.0, 100.0),
        )
        level = _build_closed_level(perimeter)
        duplicate_ids = tuple(
            level.vertex_data.add_vertex(*point).id
            for point in perimeter
        )
        for start_id, end_id in zip(
            duplicate_ids,
            (*duplicate_ids[1:], duplicate_ids[0]),
        ):
            level.vertex_data.add_edge(end_id, start_id)

        model = convert_to_glb([level])
        floor_mesh = model.scene.geometry["l2_ground_floor"]

        self.assertEqual(
            sum(name == "l2_ground_floor" for name in model.scene.geometry),
            1,
        )
        for normal_y_sign in (-1.0, 1.0):
            triangle_area, union_area, component_count = (
                _get_horizontal_cap_metrics(floor_mesh, normal_y_sign)
            )
            self.assertAlmostEqual(triangle_area, 8.0)
            self.assertAlmostEqual(union_area, 8.0)
            self.assertEqual(component_count, 1)
        self.assertEqual(len(_get_interior_vertical_face_centers(floor_mesh)), 0)

    def test_gapped_outer_walls_override_small_polygonizable_cells(self) -> None:
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
                ((180.0, 0.0), (180.0, 160.0)),
                ((180.0, 210.0), (180.0, 400.0)),
                ((180.0, 200.0), (400.0, 200.0)),
                ((400.0, 200.0), (400.0, 320.0)),
                ((300.0, 80.0), (380.0, 80.0)),
                ((380.0, 80.0), (380.0, 140.0)),
                ((380.0, 140.0), (300.0, 140.0)),
                ((300.0, 140.0), (300.0, 80.0)),
            )
        )

        model = convert_to_glb([level])
        floor_mesh = model.scene.geometry["l2_ground_floor"]

        np.testing.assert_allclose(
            floor_mesh.bounds,
            np.asarray(((0.0, 0.0, 0.0), (10.0, 0.3, 8.0))),
            atol=1e-9,
        )
        for normal_y_sign in (-1.0, 1.0):
            triangle_area, union_area, component_count = (
                _get_horizontal_cap_metrics(floor_mesh, normal_y_sign)
            )
            self.assertAlmostEqual(triangle_area, 80.0)
            self.assertAlmostEqual(union_area, 80.0)
            self.assertEqual(component_count, 1)
        self.assertEqual(len(_get_interior_vertical_face_centers(floor_mesh)), 0)
        for geometry_name, mesh in model.scene.geometry.items():
            if geometry_name == "l2_ground_floor":
                continue
            self.assertGreaterEqual(float(mesh.bounds[0, 1]), 0.3)

    def test_gapped_concave_perimeter_does_not_fill_its_recess(self) -> None:
        level = _build_level_from_segments(
            (
                ((0.0, 0.0), (220.0, 0.0)),
                ((260.0, 0.0), (500.0, 0.0)),
                ((500.0, 0.0), (500.0, 80.0)),
                ((500.0, 120.0), (500.0, 200.0)),
                ((500.0, 200.0), (420.0, 200.0)),
                ((380.0, 200.0), (300.0, 200.0)),
                ((300.0, 200.0), (300.0, 280.0)),
                ((300.0, 320.0), (300.0, 400.0)),
                ((300.0, 400.0), (190.0, 400.0)),
                ((150.0, 400.0), (0.0, 400.0)),
                ((0.0, 400.0), (0.0, 220.0)),
                ((0.0, 180.0), (0.0, 0.0)),
                ((150.0, 0.0), (150.0, 160.0)),
                ((150.0, 210.0), (150.0, 400.0)),
                ((150.0, 180.0), (280.0, 180.0)),
                ((350.0, 60.0), (430.0, 60.0)),
                ((430.0, 60.0), (430.0, 130.0)),
                ((430.0, 130.0), (350.0, 130.0)),
                ((350.0, 130.0), (350.0, 60.0)),
            )
        )

        floor_mesh = convert_to_glb([level]).scene.geometry[
            "l2_ground_floor"
        ]

        np.testing.assert_allclose(
            floor_mesh.bounds,
            np.asarray(((0.0, 0.0, 0.0), (10.0, 0.3, 8.0))),
            atol=1e-9,
        )
        for normal_y_sign in (-1.0, 1.0):
            triangle_area, union_area, component_count = (
                _get_horizontal_cap_metrics(floor_mesh, normal_y_sign)
            )
            self.assertAlmostEqual(triangle_area, 64.0)
            self.assertAlmostEqual(union_area, 64.0)
            self.assertEqual(component_count, 1)

        side_centers = floor_mesh.triangles_center[
            np.abs(floor_mesh.face_normals[:, 1]) < 0.1
        ]
        for center in side_centers:
            on_expected_boundary = bool(
                np.isclose(center[2], 0.0)
                or np.isclose(center[0], 0.0)
                or (
                    np.isclose(center[0], 10.0)
                    and center[2] <= 4.0
                )
                or (
                    np.isclose(center[2], 4.0)
                    and center[0] >= 6.0
                )
                or (
                    np.isclose(center[0], 6.0)
                    and center[2] >= 4.0
                )
                or (
                    np.isclose(center[2], 8.0)
                    and center[0] <= 6.0
                )
            )
            self.assertTrue(on_expected_boundary)

    def test_floor_thickness_extrudes_up_from_level_base(self) -> None:
        for thickness_meters in (0.12, 0.65):
            with self.subTest(thickness_meters=thickness_meters):
                model = convert_to_glb(
                    [
                        _build_square_level(
                            floor_thickness_meters=thickness_meters
                        )
                    ]
                )
                floor_mesh = model.scene.geometry["l2_ground_floor"]

                np.testing.assert_allclose(
                    floor_mesh.bounds[:, 1],
                    np.asarray((0.0, thickness_meters)),
                    atol=1e-9,
                )
                self.assertAlmostEqual(
                    abs(float(floor_mesh.volume)),
                    4.0 * thickness_meters,
                )

    def test_floor_footprint_updates_with_the_current_wall_area(self) -> None:
        level = _build_square_level()
        level.vertex_data.move_vertex(2, 200.0, 0.0)
        level.vertex_data.move_vertex(3, 200.0, 100.0)

        model = convert_to_glb([level])
        floor_mesh = model.scene.geometry["l2_ground_floor"]

        np.testing.assert_allclose(
            floor_mesh.bounds,
            np.asarray(((0.0, 0.0, 0.0), (4.0, 0.3, 2.0))),
            atol=1e-9,
        )
        self.assertAlmostEqual(abs(float(floor_mesh.volume)), 2.4)

    def test_each_level_floor_and_walls_form_an_upward_stack(self) -> None:
        ground = _build_square_level(floor_thickness_meters=0.2)
        ground.height_meters = 3.0
        upper = _build_square_level(floor_thickness_meters=0.5)
        upper.index = 3
        upper.name = "First floor"

        model = convert_to_glb([ground, upper])

        np.testing.assert_allclose(
            model.scene.geometry["l2_ground_floor"].bounds[:, 1],
            np.asarray((0.0, 0.2)),
            atol=1e-9,
        )
        np.testing.assert_allclose(
            model.scene.geometry["l2_ground"].bounds[:, 1],
            np.asarray((0.2, 3.2)),
            atol=1e-9,
        )
        np.testing.assert_allclose(
            model.scene.geometry["l3_first_floor_floor"].bounds[:, 1],
            np.asarray((3.2, 3.7)),
            atol=1e-9,
        )
        np.testing.assert_allclose(
            model.scene.geometry["l3_first_floor"].bounds[:, 1],
            np.asarray((3.7, 6.7)),
            atol=1e-9,
        )

    def test_lower_floor_thickness_delta_shifts_every_mesh_above_it(self) -> None:
        thin_ground = _build_square_level(floor_thickness_meters=0.2)
        thin_ground.height_meters = 3.0
        thin_upper = _build_square_level(floor_thickness_meters=0.5)
        thin_upper.index = 3
        thin_upper.name = "First floor"

        thick_ground = _build_square_level(floor_thickness_meters=0.65)
        thick_ground.height_meters = 3.0
        thick_upper = _build_square_level(floor_thickness_meters=0.5)
        thick_upper.index = 3
        thick_upper.name = "First floor"

        thin_model = convert_to_glb([thin_ground, thin_upper])
        thick_model = convert_to_glb([thick_ground, thick_upper])
        thickness_delta = 0.65 - 0.2

        for geometry_name in (
            "l2_ground",
            "l3_first_floor_floor",
            "l3_first_floor",
        ):
            thin_mesh = thin_model.scene.geometry[geometry_name]
            thick_mesh = thick_model.scene.geometry[geometry_name]
            np.testing.assert_array_equal(thick_mesh.faces, thin_mesh.faces)
            np.testing.assert_allclose(
                thick_mesh.vertices[:, (0, 2)],
                thin_mesh.vertices[:, (0, 2)],
                atol=1e-12,
            )
            np.testing.assert_allclose(
                thick_mesh.vertices[:, 1] - thin_mesh.vertices[:, 1],
                thickness_delta,
                atol=1e-12,
            )

    def test_floor_is_separate_and_no_wall_extends_below_its_top(self) -> None:
        level = _build_square_level(floor_thickness_meters=0.65)

        model = convert_to_glb([level])

        self.assertIn("l2_ground_floor", model.scene.geometry)
        self.assertIn("l2_ground", model.scene.geometry)
        floor_mesh = model.scene.geometry["l2_ground_floor"]
        wall_mesh = model.scene.geometry["l2_ground"]
        self.assertAlmostEqual(float(floor_mesh.bounds[0, 1]), 0.0)
        self.assertAlmostEqual(float(floor_mesh.bounds[1, 1]), 0.65)
        self.assertGreaterEqual(
            float(wall_mesh.bounds[0, 1]),
            float(floor_mesh.bounds[1, 1]),
        )
        self.assertAlmostEqual(
            float(np.ptp(wall_mesh.bounds[:, 1])),
            level.height_meters,
        )

    def test_basement_floor_grows_up_and_pushes_the_ground_stack(self) -> None:
        basement = _build_square_level(floor_thickness_meters=0.4)
        basement.index = 1
        basement.name = "Basement"
        basement.height_meters = 2.0
        ground = _build_square_level(floor_thickness_meters=0.2)
        ground.height_meters = 3.0

        model = convert_to_glb([basement, ground])

        np.testing.assert_allclose(
            model.scene.geometry["l1_basement_floor"].bounds[:, 1],
            np.asarray((-2.3, -1.9)),
            atol=1e-9,
        )
        np.testing.assert_allclose(
            model.scene.geometry["l1_basement"].bounds[:, 1],
            np.asarray((-1.9, 0.1)),
            atol=1e-9,
        )
        np.testing.assert_allclose(
            model.scene.geometry["l2_ground_floor"].bounds[:, 1],
            np.asarray((0.1, 0.3)),
            atol=1e-9,
        )
        np.testing.assert_allclose(
            model.scene.geometry["l2_ground"].bounds[:, 1],
            np.asarray((0.3, 3.3)),
            atol=1e-9,
        )

    def test_basement_thickness_delta_moves_every_higher_mesh(self) -> None:
        thin_basement = _build_square_level(floor_thickness_meters=0.2)
        thin_basement.index = 1
        thin_basement.name = "Basement"
        thin_basement.height_meters = 2.0
        thin_ground = _build_square_level(floor_thickness_meters=0.2)

        thick_basement = _build_square_level(floor_thickness_meters=0.65)
        thick_basement.index = 1
        thick_basement.name = "Basement"
        thick_basement.height_meters = 2.0
        thick_ground = _build_square_level(floor_thickness_meters=0.2)

        thin_model = convert_to_glb([thin_basement, thin_ground])
        thick_model = convert_to_glb([thick_basement, thick_ground])
        thickness_delta = 0.65 - 0.2

        thin_floor = thin_model.scene.geometry["l1_basement_floor"]
        thick_floor = thick_model.scene.geometry["l1_basement_floor"]
        self.assertAlmostEqual(
            float(thin_floor.bounds[0, 1]),
            float(thick_floor.bounds[0, 1]),
        )
        for geometry_name in (
            "l1_basement",
            "l2_ground_floor",
            "l2_ground",
        ):
            thin_mesh = thin_model.scene.geometry[geometry_name]
            thick_mesh = thick_model.scene.geometry[geometry_name]
            np.testing.assert_allclose(
                thick_mesh.vertices[:, 1] - thin_mesh.vertices[:, 1],
                thickness_delta,
                atol=1e-12,
            )

    def test_open_wall_chain_does_not_create_a_floor(self) -> None:
        vertex_data = VertexData()
        vertex_ids = tuple(
            vertex_data.add_vertex(*point).id
            for point in ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0))
        )
        for start_id, end_id in zip(vertex_ids, vertex_ids[1:]):
            vertex_data.add_edge(start_id, end_id)
        level = LevelData(
            index=2,
            name="Ground",
            vertex_data=vertex_data,
        )

        model = convert_to_glb([level])

        self.assertNotIn("l2_ground_floor", model.scene.geometry)

    def test_unresolved_walls_do_not_extrude_only_small_closed_cells(
        self,
    ) -> None:
        level = _build_level_from_segments(
            (
                ((0.0, 0.0), (100.0, 0.0)),
                ((100.0, 0.0), (100.0, 100.0)),
                ((100.0, 100.0), (0.0, 100.0)),
                ((0.0, 100.0), (0.0, 0.0)),
                ((500.0, 0.0), (1000.0, 0.0)),
            )
        )

        model = convert_to_glb([level])

        self.assertNotIn("l2_ground_floor", model.scene.geometry)

    def test_invalid_floor_thickness_is_rejected(self) -> None:
        for invalid_thickness in (0.0, -0.1, float("nan"), True):
            with self.subTest(invalid_thickness=invalid_thickness):
                level = _build_square_level()
                level.floor_thickness_meters = invalid_thickness

                with self.assertRaisesRegex(ValueError, "floor thickness"):
                    convert_to_glb([level])

    def test_floor_node_survives_glb_round_trip(self) -> None:
        model = convert_to_glb([_build_square_level()])

        loaded_scene = trimesh.load(BytesIO(model.glb_bytes), file_type="glb")

        self.assertIsInstance(loaded_scene, trimesh.Scene)
        self.assertIn("l2_ground_floor", loaded_scene.geometry)
        self.assertIn("l2_ground_floor", loaded_scene.graph.nodes_geometry)


# ### Test runner ###
if __name__ == "__main__":
    unittest.main()
