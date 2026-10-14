# ### Imports ###
from __future__ import annotations

import unittest

import numpy as np
import shapely
import trimesh
from shapely import Point, Polygon

from housemaker.floor_geometry import FLOOR_FASCIA_GUARD_HALF_WIDTH_METERS
from housemaker.glb import (
    build_stair_slab_openings_by_level,
    convert_to_preview_model,
)
from housemaker.models import LevelData, OpenSpaceData, StairData, VertexData
from housemaker.surface_geometry import FixedSurface, build_fixed_surfaces


# ### Fixture helpers ###
def _build_square_level(
    index: int,
    *,
    scale: float = 1.0,
    offset_x_meters: float = 0.0,
    offset_y_meters: float = 0.0,
) -> LevelData:
    vertex_data = VertexData()
    vertex_ids = tuple(
        vertex_data.add_vertex(*point).id
        for point in (
            (0.0, 0.0),
            (200.0, 0.0),
            (200.0, 200.0),
            (0.0, 200.0),
        )
    )
    for start_id, end_id in zip(
        vertex_ids,
        (*vertex_ids[1:], vertex_ids[0]),
        strict=True,
    ):
        vertex_data.add_edge(start_id, end_id)
    return LevelData(
        index=index,
        name="Lower" if index == 2 else "Upper",
        scale=scale,
        offset_x_meters=offset_x_meters,
        offset_y_meters=offset_y_meters,
        vertex_data=vertex_data,
    )


def _build_interrupted_upper_hallway() -> LevelData:
    vertex_data = VertexData()
    vertex_ids = tuple(
        vertex_data.add_vertex(*point).id
        for point in (
            (50.0, 0.0),
            (150.0, 0.0),
            (150.0, 200.0),
            (100.0, 200.0),
        )
    )
    for vertex_index in range(len(vertex_ids) - 1):
        vertex_data.add_edge(
            vertex_ids[vertex_index],
            vertex_ids[vertex_index + 1],
        )
    return LevelData(index=3, name="Upper", vertex_data=vertex_data)


def _build_stair(*, descending: bool = False) -> StairData:
    arguments = {
        "start_level_index": 2,
        "start_a_x": 50.0,
        "start_a_y": 50.0,
        "start_b_x": 50.0,
        "start_b_y": 100.0,
        "end_level_index": 3,
        "end_a_x": 150.0,
        "end_a_y": 50.0,
        "end_b_x": 150.0,
        "end_b_y": 100.0,
    }
    if descending:
        arguments = {
            "start_level_index": arguments["end_level_index"],
            "start_a_x": arguments["end_a_x"],
            "start_a_y": arguments["end_a_y"],
            "start_b_x": arguments["end_b_x"],
            "start_b_y": arguments["end_b_y"],
            "end_level_index": arguments["start_level_index"],
            "end_a_x": arguments["start_a_x"],
            "end_a_y": arguments["start_a_y"],
            "end_b_x": arguments["start_b_x"],
            "end_b_y": arguments["start_b_y"],
        }
    return StairData(**arguments)


# ### Geometry inspection helpers ###
def _surface_geometry(surface: FixedSurface) -> object:
    mesh = surface.mesh
    return shapely.union_all(
        [
            Polygon(triangle[:, :2])
            for triangle in np.asarray(mesh.triangles, dtype=float)
        ]
    )


def _world_scene_mesh(
    scene: trimesh.Scene,
    geometry_name: str,
) -> trimesh.Trimesh:
    transform, resolved_geometry_name = scene.graph.get(geometry_name)
    assert resolved_geometry_name == geometry_name
    mesh = scene.geometry[geometry_name].copy()
    mesh.apply_transform(transform)
    return mesh


def _upward_cap_area(mesh: trimesh.Trimesh) -> float:
    upward_faces = mesh.face_normals[:, 1] > 0.9
    return float(np.sum(mesh.area_faces[upward_faces]))


# ### Stair slab opening tests ###
class StairSlabOpeningTests(unittest.TestCase):
    def test_stair_cuts_upper_floor_and_immediate_lower_ceiling(self) -> None:
        lower = _build_square_level(2)
        upper = _build_square_level(3)
        stair = _build_stair()

        surfaces = {
            surface.surface_id: surface
            for surface in build_fixed_surfaces(
                (lower, upper),
                stairs=(stair,),
            )
        }

        self.assertAlmostEqual(surfaces["level:2/floor"].area_square_meters, 16.0)
        self.assertAlmostEqual(
            surfaces["level:2/ceiling"].area_square_meters,
            14.0,
        )
        self.assertAlmostEqual(surfaces["level:3/floor"].area_square_meters, 14.0)
        self.assertAlmostEqual(
            surfaces["level:3/ceiling"].area_square_meters,
            16.0,
        )
        stair_center = Point(2.0, -1.5)
        self.assertFalse(
            _surface_geometry(surfaces["level:3/floor"]).covers(stair_center)
        )
        self.assertFalse(
            _surface_geometry(surfaces["level:2/ceiling"]).covers(stair_center)
        )

    def test_stair_perimeter_closes_interrupted_upper_hallway_surface(
        self,
    ) -> None:
        lower = _build_square_level(2)
        upper = _build_interrupted_upper_hallway()
        stair = StairData(
            start_level_index=2,
            start_a_x=0.0,
            start_a_y=0.0,
            start_b_x=0.0,
            start_b_y=200.0,
            end_level_index=3,
            end_a_x=100.0,
            end_a_y=0.0,
            end_b_x=100.0,
            end_b_y=200.0,
        )

        surfaces = {
            surface.surface_id: surface
            for surface in build_fixed_surfaces(
                (lower, upper),
                stairs=(stair,),
            )
        }

        self.assertIn("level:3/floor", surfaces)
        upper_floor = surfaces["level:3/floor"]
        upper_floor_geometry = _surface_geometry(upper_floor)
        fascia_guard = FLOOR_FASCIA_GUARD_HALF_WIDTH_METERS
        self.assertAlmostEqual(
            upper_floor.area_square_meters,
            4.0 + 8.0 * fascia_guard - 2.0 * fascia_guard**2,
        )
        self.assertTrue(upper_floor_geometry.covers(Point(2.5, -2.0)))
        self.assertFalse(upper_floor_geometry.covers(Point(1.0, -2.0)))

    def test_wall_orientation_ignores_snapped_stair_opening_perimeter(
        self,
    ) -> None:
        lower = _build_square_level(2)
        upper = _build_interrupted_upper_hallway()
        stair = StairData(
            start_level_index=2,
            start_a_x=200.0,
            start_a_y=50.0,
            start_b_x=150.0,
            start_b_y=50.0,
            end_level_index=3,
            end_a_x=150.0,
            end_a_y=200.0,
            end_b_x=100.0,
            end_b_y=200.0,
        )

        baseline_surfaces = {
            surface.surface_id: surface
            for surface in build_fixed_surfaces((lower, upper))
        }
        stair_surfaces = {
            surface.surface_id: surface
            for surface in build_fixed_surfaces(
                (lower, upper),
                stairs=(stair,),
            )
        }
        baseline_wall = baseline_surfaces["level:3/wall:2:3"]
        stair_wall = stair_surfaces["level:3/wall:2:3"]

        self.assertAlmostEqual(
            stair_wall.area_square_meters,
            baseline_wall.area_square_meters,
        )
        np.testing.assert_allclose(
            stair_wall.mesh.vertices,
            baseline_wall.mesh.vertices,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            stair_wall.mesh.face_normals,
            baseline_wall.mesh.face_normals,
            atol=1e-9,
        )

    def test_descending_stair_owns_the_same_upper_slab_opening(self) -> None:
        levels = (_build_square_level(2), _build_square_level(3))

        ascending = build_stair_slab_openings_by_level(
            levels,
            (_build_stair(),),
        )[3]
        descending = build_stair_slab_openings_by_level(
            levels,
            (_build_stair(descending=True),),
        )[3]

        self.assertTrue(ascending.equals_exact(descending, 1e-9))
        self.assertAlmostEqual(float(ascending.area), 2.0)

    def test_overlapping_explicit_and_stair_openings_are_unioned_once(self) -> None:
        lower = _build_square_level(2)
        upper = _build_square_level(3)
        upper.open_spaces.append(
            OpenSpaceData("same-opening", 50.0, 50.0, 150.0, 100.0)
        )

        surfaces = {
            surface.surface_id: surface
            for surface in build_fixed_surfaces(
                (lower, upper),
                stairs=(_build_stair(),),
            )
        }

        self.assertAlmostEqual(surfaces["level:3/floor"].area_square_meters, 14.0)
        self.assertAlmostEqual(
            surfaces["level:2/ceiling"].area_square_meters,
            14.0,
        )

    def test_explicit_opening_overrides_mostly_overlapping_stair_route(
        self,
    ) -> None:
        lower = _build_square_level(2)
        upper = _build_square_level(3)
        upper.open_spaces.append(
            OpenSpaceData(
                "manual-stairwell",
                minimum_x=55.0,
                minimum_y=50.0,
                maximum_x=150.0,
                maximum_y=100.0,
            )
        )

        surfaces = {
            surface.surface_id: surface
            for surface in build_fixed_surfaces(
                (lower, upper),
                stairs=(_build_stair(),),
            )
        }

        self.assertAlmostEqual(surfaces["level:3/floor"].area_square_meters, 14.1)
        self.assertAlmostEqual(
            surfaces["level:2/ceiling"].area_square_meters,
            14.1,
        )
        self.assertTrue(
            _surface_geometry(surfaces["level:3/floor"]).covers(
                Point(1.05, -1.5)
            )
        )
        model = convert_to_preview_model(
            (lower, upper),
            stairs=(_build_stair(),),
        )
        self.assertAlmostEqual(
            _upward_cap_area(
                _world_scene_mesh(model.scene, "l3_upper_floor")
            ),
            14.1,
        )

    def test_small_explicit_overlap_keeps_complete_stair_route(self) -> None:
        lower = _build_square_level(2)
        upper = _build_square_level(3)
        upper.open_spaces.append(
            OpenSpaceData(
                "adjacent-opening",
                minimum_x=50.0,
                minimum_y=50.0,
                maximum_x=75.0,
                maximum_y=100.0,
            )
        )

        surfaces = {
            surface.surface_id: surface
            for surface in build_fixed_surfaces(
                (lower, upper),
                stairs=(_build_stair(),),
            )
        }

        self.assertAlmostEqual(surfaces["level:3/floor"].area_square_meters, 14.0)
        self.assertAlmostEqual(
            surfaces["level:2/ceiling"].area_square_meters,
            14.0,
        )
        self.assertFalse(
            _surface_geometry(surfaces["level:3/floor"]).covers(
                Point(2.0, -1.5)
            )
        )
        model = convert_to_preview_model(
            (lower, upper),
            stairs=(_build_stair(),),
        )
        self.assertAlmostEqual(
            _upward_cap_area(
                _world_scene_mesh(model.scene, "l3_upper_floor")
            ),
            14.0,
        )

    def test_stair_with_excluded_endpoint_does_not_cut_included_level(self) -> None:
        lower = _build_square_level(2)
        upper = _build_square_level(3)
        upper.include_in_export = False

        openings = build_stair_slab_openings_by_level(
            (lower, upper),
            (_build_stair(),),
        )
        surfaces = {
            surface.surface_id: surface
            for surface in build_fixed_surfaces(
                (lower, upper),
                stairs=(_build_stair(),),
            )
        }

        self.assertEqual(openings, {})
        self.assertAlmostEqual(
            surfaces["level:2/ceiling"].area_square_meters,
            16.0,
        )

    def test_transformed_level_floor_mesh_uses_world_stair_opening(self) -> None:
        lower = _build_square_level(
            2,
            scale=1.5,
            offset_x_meters=4.0,
            offset_y_meters=-3.0,
        )
        upper = _build_square_level(
            3,
            scale=1.5,
            offset_x_meters=4.0,
            offset_y_meters=-3.0,
        )

        model = convert_to_preview_model(
            (lower, upper),
            stairs=(_build_stair(),),
        )
        upper_floor = _world_scene_mesh(model.scene, "l3_upper_floor")

        self.assertAlmostEqual(_upward_cap_area(upper_floor), 31.5)
        surfaces = {
            surface.surface_id: surface
            for surface in build_fixed_surfaces(
                (lower, upper),
                stairs=(_build_stair(),),
            )
        }
        self.assertAlmostEqual(
            surfaces["level:3/floor"].area_square_meters,
            31.5,
        )


# ### Test runner ###
if __name__ == "__main__":
    unittest.main()
