# ### Imports ###
from __future__ import annotations

import math
import unittest

from housemaker.canvas_wall_angles import (
    DEFAULT_NON_RIGHT_ANGLE_TOLERANCE_DEGREES,
    WallAngleHighlights,
    find_non_right_angle_wall_parts,
)
from housemaker.models import Edge, Vertex


# ### Fixture helpers ###
def _corner(angle_degrees: float) -> tuple[list[Vertex], list[Edge]]:
    angle = math.radians(angle_degrees)
    return (
        [
            Vertex(1, 0.0, 0.0),
            Vertex(2, 1.0, 0.0),
            Vertex(3, math.cos(angle), math.sin(angle)),
        ],
        [Edge(1, 2), Edge(1, 3)],
    )


# ### Geometry tests ###
class CanvasWallAngleTests(unittest.TestCase):
    def test_empty_result_is_immutable_and_defaults_empty(self) -> None:
        result = WallAngleHighlights()
        self.assertEqual(result, find_non_right_angle_wall_parts([], []))
        self.assertIsInstance(result.vertex_ids, frozenset)
        self.assertIsInstance(result.edge_keys, frozenset)
        self.assertIsInstance(result.dead_end_vertex_ids, frozenset)

    def test_perpendicular_and_straight_corners_are_not_highlighted(self) -> None:
        for angle in (90.0, -90.0, 180.0, -180.0):
            with self.subTest(angle=angle):
                self.assertEqual(
                    find_non_right_angle_wall_parts(*_corner(angle)),
                    WallAngleHighlights(dead_end_vertex_ids=frozenset({2, 3})),
                )

    def test_non_right_corner_marks_only_corner_and_both_incident_edges(self) -> None:
        for angle in (0.0, 15.0, 89.0, 91.0, 135.0, 179.0):
            with self.subTest(angle=angle):
                self.assertEqual(
                    find_non_right_angle_wall_parts(*_corner(angle)),
                    WallAngleHighlights(
                        frozenset({1}),
                        frozenset({(1, 2), (1, 3)}),
                        dead_end_vertex_ids=frozenset({2, 3}),
                    ),
                )

    def test_unconnected_and_leaf_vertices_do_not_define_corners(self) -> None:
        vertices, edges = _corner(35.0)
        edges.pop()
        self.assertEqual(
            find_non_right_angle_wall_parts(vertices, edges),
            WallAngleHighlights(dead_end_vertex_ids=frozenset({1, 2})),
        )

    def test_rotated_rectangle_is_valid_regardless_of_scale(self) -> None:
        rotation = math.radians(31.0)
        for scale in (1e-12, 1.0, 1e100):
            vertices = [
                Vertex(
                    index,
                    scale * (x * math.cos(rotation) - y * math.sin(rotation)),
                    scale * (x * math.sin(rotation) + y * math.cos(rotation)),
                )
                for index, (x, y) in enumerate(
                    ((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)), 1
                )
            ]
            with self.subTest(scale=scale):
                self.assertEqual(
                    find_non_right_angle_wall_parts(
                        vertices, [Edge(1, 2), Edge(2, 3), Edge(3, 4), Edge(4, 1)]
                    ),
                    WallAngleHighlights(),
                )

    def test_straight_subdivisions_t_junctions_and_crosses_are_valid(self) -> None:
        vertices = [
            Vertex(1, 0.0, 0.0),
            Vertex(2, -1.0, 0.0),
            Vertex(3, 1.0, 0.0),
            Vertex(4, 0.0, -1.0),
            Vertex(5, 0.0, 1.0),
        ]
        for degree in (2, 3, 4):
            with self.subTest(degree=degree):
                self.assertEqual(
                    find_non_right_angle_wall_parts(
                        vertices,
                        [Edge(1, neighbor_id) for neighbor_id in range(2, degree + 2)],
                    ),
                    WallAngleHighlights(
                        dead_end_vertex_ids=frozenset(range(2, degree + 2))
                    ),
                )

    def test_only_bad_pair_edges_are_flagged_in_multiway_junction(self) -> None:
        vertices, edges = _corner(45.0)
        vertices.append(Vertex(4, -1.0, 0.0))
        result = find_non_right_angle_wall_parts(vertices, [*edges, Edge(1, 4)])
        self.assertEqual(result.vertex_ids, frozenset({1}))
        self.assertEqual(result.edge_keys, frozenset({(1, 2), (1, 3), (1, 4)}))
        self.assertEqual(result.dead_end_vertex_ids, frozenset({2, 3, 4}))

    def test_neighbor_at_end_of_highlighted_edge_is_not_highlighted_automatically(self) -> None:
        vertices = [
            Vertex(1, 0.0, 0.0),
            Vertex(2, 3.0, 0.0),
            Vertex(3, 1.0, 1.0),
            Vertex(4, 3.0, 4.0),
        ]
        result = find_non_right_angle_wall_parts(
            vertices, [Edge(2, 1), Edge(3, 1), Edge(4, 2)]
        )
        self.assertEqual(result.vertex_ids, frozenset({1}))
        self.assertEqual(result.edge_keys, frozenset({(1, 2), (1, 3)}))
        self.assertEqual(result.dead_end_vertex_ids, frozenset({3, 4}))

    def test_duplicate_edges_and_reverse_edges_do_not_create_false_corners(self) -> None:
        vertices, edges = _corner(90.0)
        self.assertEqual(
            find_non_right_angle_wall_parts(vertices, [*edges, *edges, Edge(2, 1)]),
            WallAngleHighlights(dead_end_vertex_ids=frozenset({2, 3})),
        )

    def test_excluded_room_center_spokes_do_not_create_false_corners(self) -> None:
        vertices, edges = _corner(90.0)
        vertices.append(Vertex(4, 0.3, 0.3))
        self.assertEqual(
            find_non_right_angle_wall_parts(
                vertices,
                [*edges, Edge(1, 4), Edge(4, 3)],
                excluded_vertex_ids=iter([4]),
            ),
            WallAngleHighlights(dead_end_vertex_ids=frozenset({2, 3})),
        )

    def test_missing_self_loop_zero_length_and_nonfinite_edges_are_ignored(self) -> None:
        vertices, edges = _corner(90.0)
        vertices.extend(
            [Vertex(4, 0.0, 0.0), Vertex(5, math.inf, 1.0), Vertex(6, math.nan, 1.0)]
        )
        self.assertEqual(
            find_non_right_angle_wall_parts(
                vertices,
                [*edges, Edge(1, 1), Edge(1, 4), Edge(1, 5), Edge(6, 1), Edge(1, 99)],
            ),
            WallAngleHighlights(dead_end_vertex_ids=frozenset({2, 3})),
        )

    def test_default_tolerance_handles_boundary_and_genuine_small_errors(self) -> None:
        self.assertEqual(DEFAULT_NON_RIGHT_ANGLE_TOLERANCE_DEGREES, 0.1)
        for angle in (89.9, 90.0, 90.1, 179.9):
            with self.subTest(angle=angle):
                self.assertEqual(
                    find_non_right_angle_wall_parts(*_corner(angle)),
                    WallAngleHighlights(dead_end_vertex_ids=frozenset({2, 3})),
                )
        for angle in (89.899, 90.101, 179.899):
            with self.subTest(angle=angle):
                self.assertEqual(
                    find_non_right_angle_wall_parts(*_corner(angle)).vertex_ids,
                    frozenset({1}),
                )

    def test_custom_zero_tolerance_still_allows_numerical_roundoff(self) -> None:
        self.assertEqual(
            find_non_right_angle_wall_parts(*_corner(90.0), tolerance_degrees=0.0),
            WallAngleHighlights(dead_end_vertex_ids=frozenset({2, 3})),
        )
        self.assertEqual(
            find_non_right_angle_wall_parts(
                *_corner(90.001), tolerance_degrees=0.0
            ).vertex_ids,
            frozenset({1}),
        )

    def test_invalid_tolerances_are_rejected(self) -> None:
        for tolerance in (
            -1.0, 45.0, 90.0, math.nan, math.inf, -math.inf, None, "not-a-number"
        ):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                find_non_right_angle_wall_parts([], [], tolerance_degrees=tolerance)


# ### Dead-end detection tests ###
class CanvasWallDeadEndTests(unittest.TestCase):
    def test_isolated_vertices_are_not_dead_ends(self) -> None:
        result = find_non_right_angle_wall_parts(
            [Vertex(1, 0.0, 0.0), Vertex(2, 1.0, 1.0)], []
        )
        self.assertEqual(result.dead_end_vertex_ids, frozenset())

    def test_only_vertices_with_exactly_one_neighbor_are_dead_ends(self) -> None:
        result = find_non_right_angle_wall_parts(
            [
                Vertex(1, 0.0, 0.0),
                Vertex(2, 1.0, 0.0),
                Vertex(3, 2.0, 0.0),
                Vertex(4, 3.0, 0.0),
                Vertex(5, 4.0, 0.0),
            ],
            [Edge(1, 2), Edge(2, 3), Edge(3, 4)],
        )
        self.assertEqual(result.dead_end_vertex_ids, frozenset({1, 4}))
        self.assertEqual(result.vertex_ids, frozenset())
        self.assertEqual(result.edge_keys, frozenset())

    def test_duplicate_and_reverse_edges_count_as_one_neighbor(self) -> None:
        result = find_non_right_angle_wall_parts(
            [Vertex(1, 0.0, 0.0), Vertex(2, 1.0, 0.0)],
            [Edge(1, 2), Edge(2, 1), Edge(1, 2)],
        )
        self.assertEqual(result.dead_end_vertex_ids, frozenset({1, 2}))

    def test_invalid_and_excluded_edges_do_not_count_as_neighbors(self) -> None:
        result = find_non_right_angle_wall_parts(
            [
                Vertex(1, 0.0, 0.0),
                Vertex(2, 1.0, 0.0),
                Vertex(3, 2.0, 0.0),
                Vertex(4, 1.0, 0.0),
                Vertex(5, math.nan, 1.0),
                Vertex(6, 1.0, 1.0),
            ],
            [
                Edge(1, 2),
                Edge(2, 3),
                Edge(2, 4),
                Edge(2, 5),
                Edge(2, 99),
                Edge(2, 2),
                Edge(3, 6),
            ],
            excluded_vertex_ids=[3],
        )
        self.assertEqual(result.dead_end_vertex_ids, frozenset({1, 2}))


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
