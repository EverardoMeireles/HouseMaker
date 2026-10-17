# ### Imports ###
from __future__ import annotations

import unittest

import numpy as np

from housemaker.automatic_wall_detection import (
    PlanWallDetectionOptions,
    _append_segments_to_graph,
    _canonical_endpoints,
    _node_segments,
    _Segment,
)
from housemaker.models import VertexData


# ### Fixture helpers ###
def _segment(start: tuple[float, float], end: tuple[float, float]) -> _Segment:
    first, second = _canonical_endpoints(
        np.asarray(start, dtype=np.float64),
        np.asarray(end, dtype=np.float64),
    )
    return _Segment(first, second, 1.0)


def _edges(graph: VertexData) -> set[tuple[tuple[float, float], ...]]:
    positions = {vertex.id: (vertex.x, vertex.y) for vertex in graph.vertices}
    return {
        tuple(sorted((positions[edge.start_vertex_id], positions[edge.end_vertex_id])))
        for edge in graph.edges
    }


def _build(*segments: _Segment, **overrides: float) -> VertexData:
    graph = VertexData()
    _append_segments_to_graph(
        graph,
        segments,
        160,
        160,
        PlanWallDetectionOptions(**overrides),
    )
    return graph


# ### Generated corner consolidation regressions ###
class PlanWallCornerOrthogonalityTests(unittest.TestCase):
    def test_short_horizontal_step_does_not_bend_its_vertical_walls(self) -> None:
        segments = (
            _segment((40.0, 10.0), (40.0, 48.0)),
            _segment((40.0, 48.0), (43.0, 48.0)),
            _segment((43.0, 48.0), (43.0, 120.0)),
        )

        graph = _build(*segments)

        self.assertEqual(
            _edges(graph),
            {tuple(sorted((tuple(segment.start), tuple(segment.end)))) for segment in segments},
        )

    def test_short_vertical_step_does_not_bend_its_horizontal_walls(self) -> None:
        segments = (
            _segment((10.0, 40.0), (48.0, 40.0)),
            _segment((48.0, 40.0), (48.0, 43.0)),
            _segment((48.0, 43.0), (120.0, 43.0)),
        )

        graph = _build(*segments)

        self.assertEqual(
            _edges(graph),
            {tuple(sorted((tuple(segment.start), tuple(segment.end)))) for segment in segments},
        )

    def test_adjacent_corner_and_t_junction_stay_on_their_support_lines(self) -> None:
        segments = (
            _segment((20.0, 46.0), (81.0, 46.0)),
            _segment((81.0, 46.0), (84.0, 46.0)),
            _segment((84.0, 31.0), (84.0, 43.0)),
            _segment((84.0, 43.0), (84.0, 46.0)),
            _segment((84.0, 43.0), (95.0, 43.0)),
            _segment((84.0, 46.0), (84.0, 63.0)),
        )

        graph = _build(*segments)

        self.assertTrue(all(
            first[0] == second[0] or first[1] == second[1]
            for first, second in _edges(graph)
        ))
        self.assertIn(((84.0, 43.0), (95.0, 43.0)), _edges(graph))

    def test_collinear_gaps_still_consolidate(self) -> None:
        graph = _build(
            _segment((10.0, 20.0), (48.0, 20.0)),
            _segment((51.0, 20.0), (110.0, 20.0)),
        )

        self.assertEqual(
            _edges(graph),
            {((10.0, 20.0), (49.5, 20.0)), ((49.5, 20.0), (110.0, 20.0))},
        )

    def test_oblique_pixel_fragment_does_not_pull_a_right_angle_corner_off_axis(self) -> None:
        graph = _build(
            _segment((40.0, 10.0), (40.0, 48.0)),
            _segment((40.0, 48.0), (110.0, 48.0)),
            _segment((40.0, 45.0), (70.0, 42.0)),
        )

        self.assertIn(((40.0, 10.0), (40.0, 48.0)), _edges(graph))
        self.assertIn(((40.0, 48.0), (110.0, 48.0)), _edges(graph))

    def test_near_missed_perpendicular_lines_still_join_at_their_intersection(self) -> None:
        options = PlanWallDetectionOptions()
        graph = _build(*_node_segments((
            _segment((40.0, 10.0), (40.0, 48.0)),
            _segment((43.0, 50.0), (110.0, 50.0)),
        ), options))

        self.assertEqual(
            _edges(graph),
            {((40.0, 10.0), (40.0, 50.0)), ((40.0, 50.0), (110.0, 50.0))},
        )

    def test_oblique_wall_is_not_forced_onto_an_axis(self) -> None:
        graph = _build(_segment((20.0, 20.0), (70.0, 50.0)))

        self.assertEqual(_edges(graph), {((20.0, 20.0), (70.0, 50.0))})

    def test_overlapping_duplicate_stroke_detections_still_consolidate(self) -> None:
        graph = _build(
            _segment((10.0, 40.0), (100.0, 40.0)),
            _segment((13.0, 42.0), (103.0, 42.0)),
        )

        self.assertEqual(_edges(graph), {((11.5, 41.0), (101.5, 41.0))})

    def test_authored_anchor_positions_and_identifiers_remain_unchanged(self) -> None:
        graph = VertexData()
        first = graph.add_vertex(30.0, 30.0)
        second = graph.add_vertex(30.0, 90.0)
        graph.add_edge(first.id, second.id)
        original_vertices = graph.to_dict()["vertices"]
        original_edge = graph.edges[0]

        _append_segments_to_graph(
            graph,
            (_segment((32.0, 90.0), (110.0, 90.0)),),
            160,
            160,
            PlanWallDetectionOptions(),
        )

        self.assertEqual(graph.to_dict()["vertices"][:2], original_vertices)
        self.assertEqual(graph.edges[0], original_edge)
        self.assertIn(((30.0, 90.0), (110.0, 90.0)), _edges(graph))


# ### Standalone test entry point ###
if __name__ == "__main__":
    unittest.main()
