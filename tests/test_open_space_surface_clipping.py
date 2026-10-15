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
from housemaker.models import (
    LevelData,
    OpenSpaceData,
    SlopedCeilingData,
    StairData,
    VertexData,
)
from housemaker.surface_geometry import (
    _build_slab_openings_by_level,
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


def _host_ceiling_surfaces(
    surfaces,
    host_surface_id: str,
) -> tuple:
    """Return one ceiling host and all semantic slope children it owns."""

    return tuple(
        surface
        for surface in surfaces
        if surface.surface_id == host_surface_id
        or surface.sloped_ceiling_host_surface_id == host_surface_id
    )


def _combined_host_geometry(
    surfaces,
    host_surface_id: str,
) -> object:
    """Combine the horizontal projections of one semantic ceiling host."""

    return shapely.union_all(
        [
            _surface_geometry(surface)
            for surface in _host_ceiling_surfaces(
                surfaces,
                host_surface_id,
            )
        ]
    )


def _combined_host_z_values(
    surfaces,
    host_surface_id: str,
) -> np.ndarray:
    """Collect Z values across a ceiling host and every slope child."""

    return np.concatenate(
        [
            np.asarray(surface.mesh.vertices, dtype=float)[:, 2]
            for surface in _host_ceiling_surfaces(
                surfaces,
                host_surface_id,
            )
        ]
    )


def _slope_children_by_id(
    surfaces,
    host_surface_id: str,
) -> dict[str, object]:
    """Index one host's semantic slope children by their stable IDs."""

    return {
        surface.surface_id: surface
        for surface in _host_ceiling_surfaces(surfaces, host_surface_id)
        if surface.sloped_ceiling_id is not None
    }


def _assert_meshes_equal(
    test_case: unittest.TestCase,
    actual,
    expected,
) -> None:
    """Assert exact retained slope geometry, independent of mesh object identity."""

    np.testing.assert_array_equal(
        np.asarray(actual.vertices, dtype=float),
        np.asarray(expected.vertices, dtype=float),
    )
    np.testing.assert_array_equal(
        np.asarray(actual.faces, dtype=np.int64),
        np.asarray(expected.faces, dtype=np.int64),
    )
    test_case.assertAlmostEqual(float(actual.area), float(expected.area))


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


def _add_test_sloped_ceiling(
    lower: LevelData,
    upper: LevelData,
) -> None:
    flat_surfaces = build_base_fixed_surfaces(
        (lower, upper),
        clip_open_spaces=False,
        apply_sloped_ceilings=False,
    )
    host_wall = next(
        surface
        for surface in flat_surfaces
        if surface.level_index == lower.index
        and surface.surface_type == "wall"
        and np.allclose(surface.wall_start_world[1], 0.0)
        and np.allclose(surface.wall_end_world[1], 0.0)
    )
    lower.sloped_ceilings.append(
        SlopedCeilingData(
            slope_id="b" * 32,
            wall_surface_id=host_wall.surface_id,
            depth_meters=0.4,
            wall_drop_meters=0.4,
            start_ratio=0.1,
            end_ratio=0.9,
            side_transition_radius_meters=0.1,
        )
    )


# ### Surface clipping regressions ###
class OpenSpaceSurfaceClippingTests(unittest.TestCase):
    def test_upper_open_space_does_not_flatten_lower_sloped_ceiling(
        self,
    ) -> None:
        lower = _build_square_level(1)
        upper = _build_square_level(2)
        _add_test_sloped_ceiling(lower, upper)
        host_surface_id = "level:1/ceiling"
        before_surfaces = build_fixed_surfaces((lower, upper))
        before_z = _combined_host_z_values(
            before_surfaces,
            host_surface_id,
        )
        before_children = _slope_children_by_id(
            before_surfaces,
            host_surface_id,
        )
        upper.open_spaces.append(OpenSpaceData("opening-a", 25.0, 25.0, 75.0, 75.0))

        after_surfaces = build_fixed_surfaces((lower, upper))
        after_z = _combined_host_z_values(
            after_surfaces,
            host_surface_id,
        )
        after_children = _slope_children_by_id(
            after_surfaces,
            host_surface_id,
        )

        self.assertFalse(
            _combined_host_geometry(
                after_surfaces,
                host_surface_id,
            ).covers(Point(1.0, -1.0))
        )
        self.assertAlmostEqual(float(np.min(after_z)), float(np.min(before_z)))
        self.assertAlmostEqual(float(np.max(after_z)), float(np.max(before_z)))
        self.assertGreater(float(np.ptp(after_z)), 0.39)
        self.assertEqual(set(after_children), set(before_children))
        for surface_id, before_child in before_children.items():
            _assert_meshes_equal(
                self,
                after_children[surface_id].mesh,
                before_child.mesh,
            )

    def test_automatic_stair_opening_preserves_crossed_sloped_ceiling(
        self,
    ) -> None:
        lower = _build_square_level(1)
        upper = _build_square_level(2)
        _add_test_sloped_ceiling(lower, upper)
        host_surface_id = "level:1/ceiling"
        before_surfaces = build_fixed_surfaces((lower, upper))
        before_z = _combined_host_z_values(
            before_surfaces,
            host_surface_id,
        )
        before_children = _slope_children_by_id(
            before_surfaces,
            host_surface_id,
        )
        stair = StairData(
            start_level_index=1,
            start_a_x=40.0,
            start_a_y=0.0,
            start_b_x=60.0,
            start_b_y=0.0,
            end_level_index=2,
            end_a_x=40.0,
            end_a_y=60.0,
            end_b_x=60.0,
            end_b_y=60.0,
        )

        after_surfaces = build_fixed_surfaces(
            (lower, upper),
            stairs=(stair,),
        )
        after_z = _combined_host_z_values(
            after_surfaces,
            host_surface_id,
        )
        after_children = _slope_children_by_id(
            after_surfaces,
            host_surface_id,
        )

        self.assertFalse(
            _combined_host_geometry(
                after_surfaces,
                host_surface_id,
            ).covers(Point(1.0, -0.2))
        )
        self.assertAlmostEqual(float(np.min(after_z)), float(np.min(before_z)))
        self.assertAlmostEqual(float(np.max(after_z)), float(np.max(before_z)))
        self.assertGreater(float(np.ptp(after_z)), 0.39)
        self.assertEqual(set(after_children), set(before_children))
        opening_geometry = _build_slab_openings_by_level(
            (lower, upper),
            (stair,),
        )[upper.index]
        for surface_id, before_child in before_children.items():
            after_child = after_children[surface_id]
            if before_child.sloped_ceiling_part != "main":
                _assert_meshes_equal(
                    self,
                    after_child.mesh,
                    before_child.mesh,
                )
                continue
            expected_geometry = _surface_geometry(before_child).difference(
                opening_geometry
            )
            self.assertLessEqual(
                _surface_geometry(after_child)
                .symmetric_difference(expected_geometry)
                .area,
                1e-10,
            )

    def test_supplied_base_surfaces_preserve_opening_and_ceiling_slope(
        self,
    ) -> None:
        lower = _build_square_level(1)
        upper = _build_square_level(2)
        _add_test_sloped_ceiling(lower, upper)
        host_surface_id = "level:1/ceiling"
        before_surfaces = build_fixed_surfaces((lower, upper))
        before_z = _combined_host_z_values(
            before_surfaces,
            host_surface_id,
        )
        before_children = _slope_children_by_id(
            before_surfaces,
            host_surface_id,
        )
        upper.open_spaces.append(
            _open_space_around_world_point(
                upper,
                world_x=1.0,
                world_y=-0.2,
                half_side_meters=0.1,
            )
        )
        project_load_base_surfaces = build_base_fixed_surfaces(
            (lower, upper),
            clip_open_spaces=False,
        )

        after_surfaces = build_fixed_surfaces(
            (lower, upper),
            base_surfaces=project_load_base_surfaces,
        )
        after_z = _combined_host_z_values(
            after_surfaces,
            host_surface_id,
        )
        after_children = _slope_children_by_id(
            after_surfaces,
            host_surface_id,
        )

        self.assertFalse(
            _combined_host_geometry(
                after_surfaces,
                host_surface_id,
            ).covers(Point(1.0, -0.2))
        )
        self.assertAlmostEqual(float(np.min(after_z)), float(np.min(before_z)))
        self.assertAlmostEqual(float(np.max(after_z)), float(np.max(before_z)))
        self.assertGreater(float(np.ptp(after_z)), 0.39)
        self.assertEqual(set(after_children), set(before_children))
        opening_geometry = _build_slab_openings_by_level(
            (lower, upper),
            (),
        )[upper.index]
        for surface_id, before_child in before_children.items():
            after_child = after_children[surface_id]
            if before_child.sloped_ceiling_part != "main":
                _assert_meshes_equal(
                    self,
                    after_child.mesh,
                    before_child.mesh,
                )
                continue
            expected_geometry = _surface_geometry(before_child).difference(
                opening_geometry
            )
            self.assertLessEqual(
                _surface_geometry(after_child)
                .symmetric_difference(expected_geometry)
                .area,
                1e-10,
            )

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
            if surface.level_index == 1 and surface.surface_type == "ceiling"
        }
        upper.include_in_export = False
        upper.open_spaces.append(OpenSpaceData("opening-a", 25.0, 25.0, 75.0, 75.0))

        surfaces = [surface for surface in build_fixed_surfaces((lower, upper))]
        lower_ceilings = [
            surface
            for surface in surfaces
            if surface.level_index == 1 and surface.surface_type == "ceiling"
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
            key=lambda surface: (
                _surface_geometry(surface)
                .representative_point()
                .distance(_surface_geometry(surface).boundary)
            ),
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
            _surface_geometry(after_by_id[target.surface_id]).covers(target_point)
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
        level.open_spaces.append(OpenSpaceData("opening-a", 25.0, 25.0, 75.0, 75.0))

        with patch(
            "housemaker.surface_geometry.build_base_fixed_surfaces",
            wraps=build_base_fixed_surfaces,
        ) as base_builder:
            build_fixed_surfaces((level,))

        self.assertEqual(base_builder.call_count, 1)


if __name__ == "__main__":
    unittest.main()
