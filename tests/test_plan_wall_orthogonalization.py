# ### Imports ###
from __future__ import annotations

import math
import unittest

import cv2
import numpy as np

from housemaker.models import VertexData
from housemaker.plan_wall_orthogonalization import orthogonalize_wall_graph


# ### Fixture helpers ###
def _graph(
    points: tuple[tuple[float, float], ...],
    edges: tuple[tuple[int, int], ...],
) -> VertexData:
    graph = VertexData()
    for point in points:
        graph.add_vertex(*point)
    for start_index, end_index in edges:
        graph.add_edge(graph.vertices[start_index].id, graph.vertices[end_index].id)
    return graph


def _positions(graph: VertexData) -> dict[int, tuple[float, float]]:
    return {vertex.id: (vertex.x, vertex.y) for vertex in graph.vertices}


def _all_edges_are_axis_aligned(graph: VertexData) -> bool:
    points = _positions(graph)
    return all(
        points[edge.start_vertex_id][0] == points[edge.end_vertex_id][0]
        or points[edge.start_vertex_id][1] == points[edge.end_vertex_id][1]
        for edge in graph.edges
    )


# ### Orthogonal constraint regressions ###
class PlanWallOrthogonalizationTests(unittest.TestCase):
    def test_crooked_closed_rectangle_becomes_exactly_orthogonal(self) -> None:
        graph = _graph(
            ((20.0, 20.0), (120.0, 21.5), (121.2, 100.0), (19.0, 98.5)),
            ((0, 1), (1, 2), (2, 3), (3, 0)),
        )
        original_ids = tuple(vertex.id for vertex in graph.vertices)
        original_edges = tuple(graph.edges)
        original_allocator = graph._next_vertex_id
        original_revision = graph.revision

        changed = orthogonalize_wall_graph(graph)

        self.assertTrue(changed)
        self.assertTrue(_all_edges_are_axis_aligned(graph))
        self.assertEqual(tuple(vertex.id for vertex in graph.vertices), original_ids)
        self.assertEqual(tuple(graph.edges), original_edges)
        self.assertEqual(graph._next_vertex_id, original_allocator)
        self.assertEqual(graph.revision, original_revision + 1)

    def test_entire_collinear_chain_uses_one_common_coordinate(self) -> None:
        graph = _graph(
            ((30.0, 10.0), (31.0, 60.0), (29.0, 110.0), (30.5, 150.0)),
            ((0, 1), (1, 2), (2, 3)),
        )

        self.assertTrue(orthogonalize_wall_graph(graph))

        self.assertEqual(len({vertex.x for vertex in graph.vertices}), 1)
        self.assertEqual(
            [vertex.y for vertex in graph.vertices], [10.0, 60.0, 110.0, 150.0]
        )

    def test_one_fixed_anchor_aligns_the_whole_connected_run(self) -> None:
        graph = _graph(
            ((30.0, 10.0), (31.0, 60.0), (29.0, 110.0)),
            ((0, 1), (1, 2)),
        )
        anchor = graph.vertices[0]

        self.assertTrue(orthogonalize_wall_graph(graph, fixed_vertex_ids=(anchor.id,)))

        self.assertEqual(graph.vertices[0], anchor)
        self.assertEqual([vertex.x for vertex in graph.vertices], [30.0, 30.0, 30.0])

    def test_tiny_member_adjustments_still_share_the_exact_run_coordinate(self) -> None:
        graph = _graph(
            ((30.0, 10.0), (30.0 + 1e-7, 60.0), (31.0, 110.0)),
            ((0, 1), (1, 2)),
        )
        anchor = graph.vertices[0]

        self.assertTrue(orthogonalize_wall_graph(graph, fixed_vertex_ids=(anchor.id,)))

        self.assertEqual(graph.vertices[0], anchor)
        self.assertEqual([vertex.x for vertex in graph.vertices], [30.0, 30.0, 30.0])

    def test_conflicting_fixed_anchors_leave_the_run_unchanged(self) -> None:
        graph = _graph(
            ((30.0, 10.0), (31.0, 60.0), (32.0, 110.0)),
            ((0, 1), (1, 2)),
        )
        before = graph.clone()

        changed = orthogonalize_wall_graph(
            graph,
            fixed_vertex_ids=(graph.vertices[0].id, graph.vertices[-1].id),
        )

        self.assertFalse(changed)
        self.assertEqual(graph.vertices, before.vertices)
        self.assertEqual(graph.edges, before.edges)
        self.assertEqual(graph.revision, before.revision)

    def test_intentional_diagonal_and_both_its_endpoints_are_preserved(self) -> None:
        graph = _graph(
            ((50.0, 10.0), (51.0, 50.0), (90.0, 80.0)),
            ((0, 1), (1, 2)),
        )
        diagonal_endpoints = tuple(graph.vertices[1:])

        self.assertTrue(orthogonalize_wall_graph(graph))

        self.assertEqual(tuple(graph.vertices[1:]), diagonal_endpoints)
        self.assertEqual(graph.vertices[0].x, 51.0)
        self.assertEqual(graph.vertices[0].y, 10.0)

    def test_short_real_staircase_jogs_are_not_merged(self) -> None:
        graph = _graph(
            ((40.0, 10.0), (40.0, 48.0), (43.0, 48.0), (43.0, 120.0)),
            ((0, 1), (1, 2), (2, 3)),
        )
        before = graph.clone()

        self.assertFalse(orthogonalize_wall_graph(graph))

        self.assertEqual(graph.vertices, before.vertices)
        self.assertEqual(graph.edges, before.edges)
        self.assertEqual(graph.revision, before.revision)

    def test_total_displacement_is_bounded_in_both_axes_together(self) -> None:
        graph = _graph(
            ((20.0, 20.0), (120.0, 26.0), (126.0, 126.0), (26.0, 120.0)),
            ((0, 1), (1, 2), (2, 3), (3, 0)),
        )
        before = _positions(graph)

        orthogonalize_wall_graph(graph, maximum_adjustment_pixels=3.0)

        self.assertTrue(
            all(
                math.dist(before[vertex.id], (vertex.x, vertex.y)) <= 3.0 + 1e-9
                for vertex in graph.vertices
            )
        )

    def test_long_drifting_chain_cannot_accumulate_unbounded_movement(self) -> None:
        graph = _graph(
            tuple((30.0 + index, 10.0 + index * 20.0) for index in range(10)),
            tuple((index, index + 1) for index in range(9)),
        )
        before = graph.clone()

        self.assertFalse(orthogonalize_wall_graph(graph, maximum_adjustment_pixels=2.0))

        self.assertEqual(graph.vertices, before.vertices)

    def test_narrow_wall_faces_remain_separate(self) -> None:
        graph = _graph(
            ((20.0, 20.0), (120.0, 21.0), (120.0, 25.0), (20.0, 24.0)),
            ((0, 1), (1, 2), (2, 3), (3, 0)),
        )

        self.assertTrue(orthogonalize_wall_graph(graph))

        self.assertTrue(_all_edges_are_axis_aligned(graph))
        self.assertGreater(graph.vertices[3].y - graph.vertices[0].y, 0.0)
        self.assertEqual(len(set(_positions(graph).values())), 4)


# ### Topology and image-support regressions ###
class PlanWallOrthogonalizationSafetyTests(unittest.TestCase):
    def test_new_crossing_with_an_unrelated_wall_is_not_introduced(self) -> None:
        graph = _graph(
            ((48.0, 10.0), (50.0, 100.0), (49.0, 50.0), (51.0, 50.0)),
            ((0, 1), (2, 3)),
        )
        before = graph.clone()

        self.assertFalse(
            orthogonalize_wall_graph(
                graph,
                fixed_vertex_ids=(
                    graph.vertices[1].id,
                    graph.vertices[2].id,
                    graph.vertices[3].id,
                ),
            )
        )

        self.assertEqual(graph.vertices, before.vertices)
        self.assertEqual(graph.edges, before.edges)

    def test_new_unconnected_endpoint_contact_is_not_introduced(self) -> None:
        graph = _graph(
            ((48.0, 10.0), (50.0, 50.0), (50.0, 10.0), (80.0, 10.0)),
            ((0, 1), (2, 3)),
        )
        before = graph.clone()

        self.assertFalse(
            orthogonalize_wall_graph(
                graph,
                fixed_vertex_ids=(
                    graph.vertices[1].id,
                    graph.vertices[2].id,
                    graph.vertices[3].id,
                ),
            )
        )

        self.assertEqual(graph.vertices, before.vertices)

    def test_two_runs_cannot_collapse_a_short_connecting_edge(self) -> None:
        graph = _graph(
            ((48.0, 10.0), (50.0, 100.0), (49.0, 10.0), (50.0, 120.0)),
            ((0, 1), (2, 3), (0, 2)),
        )
        original_edges = tuple(graph.edges)

        orthogonalize_wall_graph(
            graph,
            fixed_vertex_ids=(graph.vertices[1].id, graph.vertices[3].id),
        )

        points = _positions(graph)
        self.assertEqual(tuple(graph.edges), original_edges)
        self.assertTrue(
            all(
                math.dist(points[edge.start_vertex_id], points[edge.end_vertex_id])
                > 1e-6
                for edge in graph.edges
            )
        )

    def test_separated_parallel_runs_cannot_lose_most_of_their_spacing(self) -> None:
        graph = _graph(
            ((10.0, 10.0), (11.0, 110.0), (28.0, 10.0), (29.0, 110.0)),
            ((0, 1), (2, 3)),
        )
        ink = np.zeros((130, 60), dtype=bool)
        ink[:, 16] = True
        ink[:, 23] = True
        original_bounding_gap = 28.0 - 11.0

        orthogonalize_wall_graph(graph, ink=ink, maximum_adjustment_pixels=6.0)

        first_maximum_x = max(vertex.x for vertex in graph.vertices[:2])
        second_minimum_x = min(vertex.x for vertex in graph.vertices[2:])
        self.assertGreaterEqual(
            second_minimum_x - first_maximum_x,
            original_bounding_gap * 0.5,
        )

    def test_horizontal_run_keeps_both_ends_on_the_black_wall_stroke(self) -> None:
        graph = _graph(((20.0, 48.0), (120.0, 52.0)), ((0, 1),))
        ink = np.zeros((160, 160), dtype=bool)
        ink[50, 10:131] = True

        self.assertTrue(orthogonalize_wall_graph(graph, ink=ink, image_size=(160, 160)))

        self.assertEqual([vertex.y for vertex in graph.vertices], [50.0, 50.0])
        self.assertEqual([vertex.x for vertex in graph.vertices], [20.0, 120.0])
        self.assertTrue(
            all(ink[round(vertex.y), round(vertex.x)] for vertex in graph.vertices)
        )

    def test_vertical_run_keeps_both_ends_on_the_black_wall_stroke(self) -> None:
        graph = _graph(((48.0, 20.0), (52.0, 120.0)), ((0, 1),))
        ink = np.zeros((160, 160), dtype=bool)
        ink[10:131, 50] = True

        self.assertTrue(orthogonalize_wall_graph(graph, ink=ink, image_size=(160, 160)))

        self.assertEqual([vertex.x for vertex in graph.vertices], [50.0, 50.0])
        self.assertEqual([vertex.y for vertex in graph.vertices], [20.0, 120.0])
        self.assertTrue(
            all(ink[round(vertex.y), round(vertex.x)] for vertex in graph.vertices)
        )

    def test_real_slanted_black_stroke_is_not_replaced_by_an_unsupported_axis(
        self,
    ) -> None:
        graph = _graph(((20.0, 20.0), (120.0, 27.0)), ((0, 1),))
        pixels = np.zeros((160, 160), dtype=np.uint8)
        cv2.line(pixels, (20, 20), (120, 27), 1, 1)
        before = graph.clone()

        self.assertFalse(
            orthogonalize_wall_graph(
                graph,
                ink=pixels.astype(bool),
                image_size=(160, 160),
            )
        )

        self.assertEqual(graph.vertices, before.vertices)

    def test_full_span_support_is_required_not_just_two_black_endpoints(self) -> None:
        graph = _graph(((20.0, 20.0), (120.0, 27.0)), ((0, 1),))
        pixels = np.zeros((160, 160), dtype=np.uint8)
        cv2.line(pixels, (20, 20), (120, 27), 1, 1)
        pixels[20:28, 20] = 1
        pixels[20:28, 120] = 1
        before = graph.clone()

        self.assertFalse(
            orthogonalize_wall_graph(
                graph,
                ink=pixels.astype(bool),
                image_size=(160, 160),
            )
        )

        self.assertEqual(graph.vertices, before.vertices)

    def test_existing_out_of_image_vertices_are_not_relocated_or_clipped(self) -> None:
        graph = _graph(((-10.0, 20.0), (50.0, 20.0)), ((0, 1),))
        before = graph.clone()

        self.assertFalse(
            orthogonalize_wall_graph(
                graph,
                fixed_vertex_ids=(vertex.id for vertex in graph.vertices),
                image_size=(160, 160),
            )
        )

        self.assertEqual(graph.vertices, before.vertices)

    def test_empty_graph_and_zero_adjustment_are_safe_no_ops(self) -> None:
        self.assertFalse(orthogonalize_wall_graph(VertexData()))
        graph = _graph(((20.0, 20.0), (120.0, 22.0)), ((0, 1),))
        before = graph.clone()

        self.assertFalse(orthogonalize_wall_graph(graph, maximum_adjustment_pixels=0.0))

        self.assertEqual(graph.vertices, before.vertices)
        self.assertEqual(graph.revision, before.revision)

    def test_repeated_fits_are_deterministic_and_second_fit_is_a_no_op(self) -> None:
        graph = _graph(
            ((20.0, 20.0), (120.0, 21.5), (121.2, 100.0), (19.0, 98.5)),
            ((0, 1), (1, 2), (2, 3), (3, 0)),
        )
        other = graph.clone()

        self.assertTrue(orthogonalize_wall_graph(graph))
        self.assertTrue(orthogonalize_wall_graph(other))

        self.assertEqual(graph.vertices, other.vertices)
        self.assertEqual(graph.edges, other.edges)
        first_fit = graph.clone()
        self.assertFalse(orthogonalize_wall_graph(graph))
        self.assertEqual(graph.vertices, first_fit.vertices)
        self.assertEqual(graph.revision, first_fit.revision)


# ### Standalone test entry point ###
if __name__ == "__main__":
    unittest.main()
