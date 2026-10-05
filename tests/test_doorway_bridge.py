"""Regression tests for Shift-assisted doorway bridge targeting."""

# ### Imports ###
from __future__ import annotations

import unittest
from itertools import pairwise

from housemaker.doorway_bridge import (
    find_doorway_bridge_target,
    find_doorway_bridges_covered_by_rectangle,
)
from housemaker.models import VertexData


# ### Test cases ###
class DoorwayBridgeTests(unittest.TestCase):
    def test_aligned_isolated_vertices_are_bridge_targets(self) -> None:
        vertex_data = VertexData()
        first = vertex_data.add_vertex(40.0, 50.0)
        second = vertex_data.add_vertex(60.0, 50.0)

        target = find_doorway_bridge_target(
            vertex_data,
            set(),
            (50.0, 50.0),
            preset_width_meters=0.9,
            center_tolerance=5.0,
        )

        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(
            {target.first_vertex_id, target.second_vertex_id},
            {first.id, second.id},
        )

    def test_visibly_offset_isolated_vertices_are_not_bridge_targets(self) -> None:
        vertex_data = VertexData()
        vertex_data.add_vertex(40.0, 50.0)
        vertex_data.add_vertex(60.0, 51.0)

        target = find_doorway_bridge_target(
            vertex_data,
            set(),
            (50.0, 50.5),
            preset_width_meters=0.9,
            center_tolerance=5.0,
        )

        self.assertIsNone(target)

    def test_subpixel_aligned_isolated_vertices_are_bridge_targets(self) -> None:
        vertex_data = VertexData()
        vertex_data.add_vertex(40.0, 50.0)
        vertex_data.add_vertex(60.0, 50.25)

        target = find_doorway_bridge_target(
            vertex_data,
            set(),
            (50.0, 50.125),
            preset_width_meters=0.9,
            center_tolerance=5.0,
        )

        self.assertIsNotNone(target)

    def test_directly_connected_endpoints_are_not_bridge_targets(self) -> None:
        vertex_data = VertexData()
        first = vertex_data.add_vertex(40.0, 50.0)
        second = vertex_data.add_vertex(60.0, 50.0)
        vertex_data.add_edge(first.id, second.id)

        target = find_doorway_bridge_target(
            vertex_data,
            set(),
            (50.0, 50.0),
            preset_width_meters=0.9,
            center_tolerance=5.0,
        )

        self.assertIsNone(target)

    def test_gap_endpoints_in_one_wall_component_can_be_bridged(self) -> None:
        vertex_data = VertexData()
        points = [
            vertex_data.add_vertex(*point)
            for point in (
                (40.0, 50.0),
                (20.0, 50.0),
                (20.0, 20.0),
                (80.0, 20.0),
                (80.0, 50.0),
                (60.0, 50.0),
            )
        ]
        for first, second in pairwise(points):
            vertex_data.add_edge(first.id, second.id)

        target = find_doorway_bridge_target(
            vertex_data,
            set(),
            (50.0, 50.0),
            preset_width_meters=0.9,
            center_tolerance=5.0,
        )

        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(
            {target.first_vertex_id, target.second_vertex_id},
            {points[0].id, points[-1].id},
        )

    def test_misaligned_wall_endpoints_are_not_bridge_targets(self) -> None:
        vertex_data = VertexData()
        first_outer = vertex_data.add_vertex(40.0, 20.0)
        first_gap = vertex_data.add_vertex(40.0, 50.0)
        second_gap = vertex_data.add_vertex(60.0, 50.0)
        second_outer = vertex_data.add_vertex(60.0, 20.0)
        vertex_data.add_edge(first_outer.id, first_gap.id)
        vertex_data.add_edge(second_gap.id, second_outer.id)

        target = find_doorway_bridge_target(
            vertex_data,
            set(),
            (50.0, 50.0),
            preset_width_meters=0.9,
            center_tolerance=5.0,
        )

        self.assertIsNone(target)

    def test_visibly_kinked_wall_endpoints_are_not_bridge_targets(self) -> None:
        vertex_data = VertexData()
        first_outer = vertex_data.add_vertex(20.0, 48.0)
        first_gap = vertex_data.add_vertex(40.0, 50.0)
        second_gap = vertex_data.add_vertex(60.0, 50.0)
        second_outer = vertex_data.add_vertex(80.0, 48.0)
        vertex_data.add_edge(first_outer.id, first_gap.id)
        vertex_data.add_edge(second_gap.id, second_outer.id)

        target = find_doorway_bridge_target(
            vertex_data,
            set(),
            (50.0, 50.0),
            preset_width_meters=0.9,
            center_tolerance=5.0,
        )

        self.assertIsNone(target)

    def test_existing_subdivided_wall_span_is_not_bridge_target(self) -> None:
        vertex_data = VertexData()
        points = [
            vertex_data.add_vertex(*point)
            for point in (
                (20.0, 50.0),
                (40.0, 50.0),
                (50.0, 50.0),
                (60.0, 50.0),
                (80.0, 50.0),
            )
        ]
        for first, second in pairwise(points):
            vertex_data.add_edge(first.id, second.id)

        target = find_doorway_bridge_target(
            vertex_data,
            set(),
            (50.0, 50.0),
            preset_width_meters=0.9,
            center_tolerance=5.0,
        )

        self.assertIsNone(target)

    def test_rectangle_finds_each_parallel_gap_covered_by_doorway(self) -> None:
        vertex_data = VertexData()
        expected_pairs: set[frozenset[int]] = set()
        for y_position in (50.0, 60.0):
            left_outer = vertex_data.add_vertex(10.0, y_position)
            left_gap = vertex_data.add_vertex(40.0, y_position)
            right_gap = vertex_data.add_vertex(60.0, y_position)
            right_outer = vertex_data.add_vertex(90.0, y_position)
            vertex_data.add_edge(left_outer.id, left_gap.id)
            vertex_data.add_edge(right_gap.id, right_outer.id)
            expected_pairs.add(frozenset((left_gap.id, right_gap.id)))

        targets = find_doorway_bridges_covered_by_rectangle(
            vertex_data,
            set(),
            center=(50.0, 55.0),
            width_direction=(1.0, 0.0),
            width_pixels=20.0,
            depth_pixels=12.0,
        )

        self.assertEqual(
            {
                frozenset((target.first_vertex_id, target.second_vertex_id))
                for target in targets
            },
            expected_pairs,
        )

    def test_rectangle_ignores_parallel_gap_outside_doorway_depth(self) -> None:
        vertex_data = VertexData()
        left_outer = vertex_data.add_vertex(10.0, 70.0)
        left_gap = vertex_data.add_vertex(40.0, 70.0)
        right_gap = vertex_data.add_vertex(60.0, 70.0)
        right_outer = vertex_data.add_vertex(90.0, 70.0)
        vertex_data.add_edge(left_outer.id, left_gap.id)
        vertex_data.add_edge(right_gap.id, right_outer.id)

        targets = find_doorway_bridges_covered_by_rectangle(
            vertex_data,
            set(),
            center=(50.0, 50.0),
            width_direction=(1.0, 0.0),
            width_pixels=20.0,
            depth_pixels=20.0,
        )

        self.assertEqual(targets, ())


if __name__ == "__main__":
    unittest.main()
