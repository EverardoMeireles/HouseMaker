# ### Imports ###
from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import numpy as np
import shapely
from shapely import Point, Polygon

from housemaker.architectural_surface_edits import insert_surface_vertex
from housemaker.level_coordinates import level_world_to_image_xy
from housemaker.models import LevelData, OpenSpaceData, VertexData
from housemaker.surface_geometry import (
    build_base_fixed_surfaces,
    build_fixed_surfaces,
)


# ### Fixture helpers ###
def _build_square_level(index: int, size_pixels: float = 100.0) -> LevelData:
    vertex_data = VertexData()
    vertex_ids = tuple(
        vertex_data.add_vertex(*point).id
        for point in (
            (0.0, 0.0),
            (size_pixels, 0.0),
            (size_pixels, size_pixels),
            (0.0, size_pixels),
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
        name=f"Level {index}",
        vertex_data=vertex_data,
    )


def _surface_geometry(surface) -> object:
    return shapely.union_all(
        [
            Polygon(triangle[:, :2])
            for triangle in np.asarray(surface.mesh.triangles, dtype=float)
        ]
    )


def _open_space_around_world_point(
    level: LevelData,
    world_x: float,
    world_y: float,
    half_side_meters: float,
) -> OpenSpaceData:
    image_corners = tuple(
        level_world_to_image_xy(level, world_x + delta_x, world_y + delta_y)
        for delta_x, delta_y in (
            (-half_side_meters, -half_side_meters),
            (-half_side_meters, half_side_meters),
            (half_side_meters, -half_side_meters),
            (half_side_meters, half_side_meters),
        )
    )
    image_x_values = tuple(point[0] for point in image_corners)
    image_y_values = tuple(point[1] for point in image_corners)
    return OpenSpaceData(
        "opening-a",
        min(image_x_values),
        min(image_y_values),
        max(image_x_values),
        max(image_y_values),
    )


# ### Surface clipping regressions ###
class OpenSpaceSurfaceClippingTests(unittest.TestCase):
    def test_excluded_upper_level_does_not_cut_included_lower_ceiling(
        self,
    ) -> None:
        lower = _build_square_level(1)
        upper = _build_square_level(2)
        source_ceiling = next(
            surface
            for surface in build_fixed_surfaces((lower, upper))
            if surface.surface_id == "level:1/ceiling"
        )
        insertion_point = tuple(
            float(component)
            for component in np.asarray(
                source_ceiling.mesh.triangles[0],
                dtype=float,
            ).mean(axis=0)
        )
        insert_surface_vertex(
            (lower, upper),
            source_ceiling.surface_id,
            insertion_point,
        )
        expected_ceiling_ids = {
            surface.surface_id
            for surface in build_fixed_surfaces((lower, upper))
            if surface.level_index == 1
            and surface.surface_type == "ceiling"
        }
        upper.include_in_export = False
        upper.open_spaces.append(
            OpenSpaceData("opening-a", 25.0, 25.0, 75.0, 75.0)
        )

        surfaces = [
            surface
            for surface in build_fixed_surfaces((lower, upper))
        ]
        lower_ceilings = [
            surface
            for surface in surfaces
            if surface.level_index == 1
            and surface.surface_type == "ceiling"
        ]

        self.assertFalse(any(surface.level_index == 2 for surface in surfaces))
        self.assertEqual(
            {surface.surface_id for surface in lower_ceilings},
            expected_ceiling_ids,
        )
        self.assertAlmostEqual(
            sum(surface.area_square_meters for surface in lower_ceilings),
            4.0,
        )
        self.assertTrue(
            shapely.union_all(
                [_surface_geometry(surface) for surface in lower_ceilings]
            ).covers(
                Point(1.0, -1.0),
            )
        )

    def test_hole_clips_only_affected_authored_face_and_keeps_stable_ids(
        self,
    ) -> None:
        level = _build_square_level(2)
        source_floor = next(
            surface
            for surface in build_fixed_surfaces((level,))
            if surface.surface_id == "level:2/floor"
        )
        insertion_point = tuple(
            float(component)
            for component in np.asarray(
                source_floor.mesh.triangles[0],
                dtype=float,
            ).mean(axis=0)
        )
        insert_surface_vertex((level,), source_floor.surface_id, insertion_point)
        before_by_id = {
            surface.surface_id: surface
            for surface in build_fixed_surfaces((level,))
            if surface.surface_type == "floor"
        }
        target = max(
            before_by_id.values(),
            key=lambda surface: _surface_geometry(surface)
            .representative_point()
            .distance(_surface_geometry(surface).boundary),
        )
        target_geometry = _surface_geometry(target)
        target_point = target_geometry.representative_point()
        target_clearance = target_point.distance(target_geometry.boundary)
        half_side = target_clearance / (3.0 * math.sqrt(2.0))
        level.open_spaces.append(
            _open_space_around_world_point(
                level,
                target_point.x,
                target_point.y,
                half_side,
            )
        )

        after_by_id = {
            surface.surface_id: surface
            for surface in build_fixed_surfaces((level,))
            if surface.surface_type == "floor"
        }

        self.assertEqual(set(after_by_id), set(before_by_id))
        hole_area = (2.0 * half_side) ** 2
        self.assertAlmostEqual(
            after_by_id[target.surface_id].area_square_meters,
            target.area_square_meters - hole_area,
        )
        self.assertFalse(
            _surface_geometry(after_by_id[target.surface_id]).covers(
                target_point
            )
        )
        for surface_id, before_surface in before_by_id.items():
            if surface_id == target.surface_id:
                continue
            self.assertTrue(
                _surface_geometry(after_by_id[surface_id]).equals_exact(
                    _surface_geometry(before_surface),
                    1e-10,
                )
            )

    def test_open_space_builds_base_surfaces_only_once(self) -> None:
        level = _build_square_level(2)
        level.open_spaces.append(
            OpenSpaceData("opening-a", 25.0, 25.0, 75.0, 75.0)
        )

        with patch(
            "housemaker.surface_geometry.build_base_fixed_surfaces",
            wraps=build_base_fixed_surfaces,
        ) as base_builder:
            build_fixed_surfaces((level,))

        self.assertEqual(base_builder.call_count, 1)


if __name__ == "__main__":
    unittest.main()
