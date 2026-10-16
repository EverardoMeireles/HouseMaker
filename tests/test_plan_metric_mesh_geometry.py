# ### Imports ###
from __future__ import annotations

import unittest
from io import BytesIO

import numpy as np
import trimesh

from housemaker.glb import (
    WallSource,
    _build_level_source_transform,
    _build_level_wall_openings,
    _build_opening_polygon_on_wall,
    _build_window_depth_footprint,
    _point_to_world_xy,
    _resolve_stair_section_points,
    convert_to_glb,
)
from housemaker.level_coordinates import level_image_to_world_xy
from housemaker.models import (
    DOORWAY_SHAPE_ARCH,
    DoorwayData,
    LevelData,
    RoomData,
    StairSectionData,
    VertexData,
)
from housemaker.plan_metric_state import PlanMeasurementEvidence, PlanMetricCalibration
from housemaker.surface_geometry import build_fixed_surfaces
from housemaker.uv_layout import build_uv_wall_layout, calculate_minimum_uv_map_size


# ### Fixtures ###
def _calibrated_level(meters_per_pixel: float) -> LevelData:
    image_size = (round(10.0 / meters_per_pixel), round(8.0 / meters_per_pixel))
    evidence = tuple(
        PlanMeasurementEvidence(
            label=label,
            value_meters=value,
            start_normalized=start,
            end_normalized=end,
            corrected_start_normalized=start,
            corrected_end_normalized=end,
            confidence="high",
        )
        for label, value, start, end in (
            ("width", 8.0, (0.1, 0.125), (0.9, 0.125)),
            ("depth", 6.0, (0.1, 0.125), (0.1, 0.875)),
        )
    )
    vertices = VertexData()
    boundary = tuple(
        vertices.add_vertex(x / meters_per_pixel, y / meters_per_pixel).id
        for x, y in ((1.0, 1.0), (9.0, 1.0), (9.0, 7.0), (1.0, 7.0))
    )
    for first, second in zip(boundary, (*boundary[1:], boundary[0])):
        vertices.add_edge(first, second)
    return LevelData(
        index=2,
        name="Ground",
        vertex_data=vertices,
        image_size_pixels=image_size,
        height_meters=2.45,
        floor_thickness_meters=0.3,
        plan_metric_calibration=PlanMetricCalibration(
            estimated_meters_per_pixel=meters_per_pixel,
            target_meters_per_pixel=meters_per_pixel,
            normalization_scale=1.0,
            relative_residual=0.0,
            confidence="medium",
            source_image_size=image_size,
            pre_normalization_size=image_size,
            normalized_image_size=image_size,
            measurements=evidence,
        ),
    )


def _room_for_level(level: LevelData) -> RoomData:
    ids = tuple(vertex.id for vertex in level.vertex_data.vertices)
    center = level.vertex_data.add_vertex(
        5.0 / level.meters_per_pixel, 4.0 / level.meters_per_pixel
    )
    return RoomData(
        name="Room",
        vertex_ids=ids,
        center_vertex_id=center.id,
        color_rgb=(120, 120, 120),
        height_meters=level.height_meters,
        uv_map_width=2048,
        uv_map_height=2048,
    )


# ### Geometry and export regression tests ###
class PlanMetricMeshGeometryTests(unittest.TestCase):
    def test_native_density_changes_xy_not_height_or_slab_thickness(self) -> None:
        for density in (0.01, 0.02, 0.04):
            with self.subTest(density=density):
                level = _calibrated_level(density)
                model = convert_to_glb([level])
                np.testing.assert_allclose(model.mesh.extents, (8.0, 6.0, 2.75))
                scene = trimesh.load(BytesIO(model.glb_bytes), file_type="glb")
                np.testing.assert_allclose(scene.extents, (8.0, 2.75, 6.0), atol=1e-6)
                floors = [
                    s
                    for s in build_fixed_surfaces([level])
                    if s.surface_type == "floor"
                ]
                self.assertTrue(floors)
                self.assertAlmostEqual(sum(s.area_square_meters for s in floors), 48.0)

    def test_source_transform_matches_coordinate_api_with_manual_edits(self) -> None:
        level = _calibrated_level(0.01)
        level.scale = 1.4
        level.offset_x_meters = 2.5
        level.offset_y_meters = -0.75
        transform = _build_level_source_transform(level, level.image_size_pixels)
        for vertex in level.vertex_data.vertices:
            source = _point_to_world_xy((vertex.x, vertex.y), level.image_size_pixels)
            transformed = transform @ np.asarray((*source, 2.45, 1.0))
            np.testing.assert_allclose(
                transformed[:2], level_image_to_world_xy(level, vertex.x, vertex.y)
            )
            self.assertAlmostEqual(transformed[2], 2.45)

    def test_arch_cut_keeps_metric_width_and_height_at_each_density(self) -> None:
        for density in (0.01, 0.04):
            with self.subTest(density=density):
                level = _calibrated_level(density)
                level.doorways.append(
                    DoorwayData(
                        center_x=5.0 / density,
                        center_y=1.0 / density,
                        width_meters=1.0,
                        height_meters=2.1,
                        depth_meters=0.2,
                        rotation_degrees=90.0,
                        shape=DOORWAY_SHAPE_ARCH,
                    )
                )
                opening = _build_level_wall_openings(level)[0]
                self.assertAlmostEqual(opening.half_width_pixels * density * 2.0, 1.0)
                polygon = _build_opening_polygon_on_wall(
                    (1.0 / density, 1.0 / density),
                    (9.0 / density, 1.0 / density),
                    2.45,
                    opening,
                )
                self.assertIsNotNone(polygon)
                left, bottom, right, top = polygon.bounds
                self.assertAlmostEqual((right - left) * 8.0, 1.0)
                self.assertAlmostEqual(top - bottom, 2.1)

    def test_window_penetration_is_fifty_centimetres_at_each_density(self) -> None:
        for density in (0.01, 0.04):
            with self.subTest(density=density):
                level = _calibrated_level(density)
                level.scale = 1.5
                target = WallSource(
                    key="wall",
                    start_point=(0.0, 0.0),
                    end_point=(100.0, 0.0),
                    height_meters=2.45,
                    exterior_direction=(0.0, 1.0),
                )
                center, _direction, half_depth = _build_window_depth_footprint(
                    level, target, (0.0, 0.0)
                )
                self.assertAlmostEqual(
                    (center[1] + half_depth) * density * level.scale, 0.5
                )

    def test_stair_points_on_different_native_densities_use_same_world_position(
        self,
    ) -> None:
        positions = []
        for density in (0.01, 0.04):
            level = _calibrated_level(density)
            section = StairSectionData(
                level_index=level.index,
                a_x=4.0 / density,
                a_y=3.0 / density,
                b_x=5.0 / density,
                b_y=3.0 / density,
            )
            positions.append(_resolve_stair_section_points(section, level))
        np.testing.assert_allclose(positions[0], positions[1])
        self.assertAlmostEqual(np.linalg.norm(positions[0][1] - positions[0][0]), 1.0)


# ### UV aspect and density regression tests ###
class PlanMetricUvGeometryTests(unittest.TestCase):
    def test_equivalent_rooms_have_equivalent_uv_layouts_without_resizing_plans(
        self,
    ) -> None:
        layouts = []
        minimum_sizes = []
        for density in (0.01, 0.02, 0.04):
            level = _calibrated_level(density)
            room = _room_for_level(level)
            layouts.append(
                build_uv_wall_layout(
                    room,
                    level.vertex_data,
                    level.height_meters,
                    meters_per_pixel=level.meters_per_pixel,
                )
            )
            minimum_sizes.append(
                calculate_minimum_uv_map_size(
                    room,
                    level.vertex_data,
                    level.height_meters,
                    meters_per_pixel=level.meters_per_pixel,
                )
            )
        self.assertTrue(layouts[0].placements)
        for layout in layouts[1:]:
            self.assertEqual(
                [(p.natural_size, p.uv_rect) for p in layouts[0].placements],
                [(p.natural_size, p.uv_rect) for p in layout.placements],
            )
        self.assertEqual(minimum_sizes, [minimum_sizes[0]] * 3)
        first_wall = layouts[0].placements[0].wall
        last_wall = layouts[-1].placements[0].wall
        self.assertNotEqual(first_wall.start_point, last_wall.start_point)
        self.assertEqual(first_wall.length, last_wall.length)

    def test_invalid_uv_density_is_rejected(self) -> None:
        level = _calibrated_level(0.02)
        room = _room_for_level(level)
        for density in (0.0, -1.0, float("nan"), True):
            with self.subTest(density=density), self.assertRaises(ValueError):
                build_uv_wall_layout(
                    room,
                    level.vertex_data,
                    level.height_meters,
                    meters_per_pixel=density,
                )
