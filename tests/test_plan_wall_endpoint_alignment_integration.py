# ### Imports ###
from __future__ import annotations

import unittest
from dataclasses import replace

import cv2
import numpy as np

from housemaker.automatic_wall_detection import (
    PlanWallAnalysis,
    PlanWallDetectionOptions,
    WallLineEvidence,
    reconstruct_plan_walls,
)
from housemaker.models import VertexData

# ### Fixture helpers ###
_Point = tuple[float, float]
_Edge = tuple[_Point, _Point]
_LEFT_WALLS: tuple[_Edge, ...] = (
    ((40.0, 40.0), (40.0, 160.0)),
    ((52.0, 40.0), (52.0, 160.0)),
)
_RIGHT_WALLS: tuple[_Edge, ...] = (
    ((240.0, 48.0), (240.0, 160.0)),
    ((252.0, 48.0), (252.0, 160.0)),
)


def _options(**overrides: float) -> PlanWallDetectionOptions:
    values = {
        "minimum_wall_separation_pixels": 8.0,
        "maximum_wall_separation_pixels": 20.0,
        "parallel_angle_tolerance_degrees": 5.0,
        "minimum_parallel_overlap_ratio": 0.7,
        "maximum_gap_bridge_pixels": 0.0,
        "endpoint_snap_distance_pixels": 0.0,
        "maximum_vertex_distance_pixels": 0.0,
        "minimum_wall_length_pixels": 10.0,
        "confidence_threshold": 0.35,
        "endpoint_alignment_tolerance_pixels": 16.0,
        "endpoint_alignment_search_distance_pixels": 256.0,
    }
    values.update(overrides)
    return PlanWallDetectionOptions(**values)


def _graph(*edges: _Edge) -> VertexData:
    graph = VertexData()
    vertex_ids: dict[_Point, int] = {}
    for start, end in edges:
        for point in (start, end):
            if point not in vertex_ids:
                vertex_ids[point] = graph.add_vertex(*point).id
        graph.add_edge(vertex_ids[start], vertex_ids[end])
    return graph


def _staggered_analysis(
    *,
    include_extension_ink: bool = True,
    dashed_gap: bool = False,
) -> PlanWallAnalysis:
    """Model detected short ends over a plan whose true vertical ink extends up."""

    ink = np.zeros((220, 300), dtype=np.uint8)
    for x in (40, 52, 240, 252):
        start_y = 40 if x < 200 or include_extension_ink else 48
        cv2.line(ink, (x, start_y), (x, 160), 255, 1)
    if dashed_gap:
        for x in range(65, 230, 18):
            cv2.line(ink, (x, 40), (min(x + 5, 230), 40), 255, 1)
    readonly_ink = ink.astype(np.bool_)
    readonly_ink.setflags(write=False)
    return PlanWallAnalysis(
        image_width=300,
        image_height=220,
        line_evidence=tuple(
            WallLineEvidence(start, end, 0.95)
            for start, end in (*_LEFT_WALLS, *_RIGHT_WALLS)
        ),
        structural_ink=readonly_ink,
        normalized_ink=readonly_ink,
    )


def _edge_coordinates(graph: VertexData) -> set[_Edge]:
    points = {
        vertex.id: (round(vertex.x, 4), round(vertex.y, 4))
        for vertex in graph.vertices
    }
    return {
        tuple(sorted((points[edge.start_vertex_id], points[edge.end_vertex_id])))
        for edge in graph.edges
    }


def _top_y(graph: VertexData, x: float) -> float:
    return min(vertex.y for vertex in graph.vertices if abs(vertex.x - x) < 1e-4)


# ### Public reconstruction integration regressions ###
class PlanWallEndpointAlignmentIntegrationTests(unittest.TestCase):
    def test_extends_staggered_opposite_tip_without_tilting_wall(self) -> None:
        result = reconstruct_plan_walls(_staggered_analysis(), _options())

        for x in (240.0, 252.0):
            self.assertAlmostEqual(_top_y(result.vertex_data, x), 40.0)
        for start, end in _edge_coordinates(result.vertex_data):
            self.assertTrue(start[0] == end[0] or start[1] == end[1])

    def test_adjustment_tolerance_is_not_the_distance_between_opposite_tips(self) -> None:
        analysis = _staggered_analysis()

        allowed = reconstruct_plan_walls(
            analysis, _options(endpoint_alignment_tolerance_pixels=8.0)
        )
        too_small = reconstruct_plan_walls(
            analysis, _options(endpoint_alignment_tolerance_pixels=7.0)
        )

        self.assertAlmostEqual(_top_y(allowed.vertex_data, 240.0), 40.0)
        self.assertAlmostEqual(_top_y(too_small.vertex_data, 240.0), 48.0)

    def test_opposite_tip_search_limit_is_independent_of_adjustment_tolerance(self) -> None:
        analysis = _staggered_analysis()

        near_only = reconstruct_plan_walls(
            analysis, _options(endpoint_alignment_search_distance_pixels=150.0)
        )
        far_enough = reconstruct_plan_walls(
            analysis, _options(endpoint_alignment_search_distance_pixels=256.0)
        )

        self.assertAlmostEqual(_top_y(near_only.vertex_data, 240.0), 48.0)
        self.assertAlmostEqual(_top_y(far_enough.vertex_data, 240.0), 40.0)

    def test_either_zero_control_disables_endpoint_alignment(self) -> None:
        for disabled in (
            {"endpoint_alignment_tolerance_pixels": 0.0},
            {"endpoint_alignment_search_distance_pixels": 0.0},
        ):
            with self.subTest(disabled=disabled):
                result = reconstruct_plan_walls(
                    _staggered_analysis(), _options(**disabled)
                )

                self.assertAlmostEqual(_top_y(result.vertex_data, 240.0), 48.0)

    def test_dashed_gap_does_not_become_an_invented_connecting_wall(self) -> None:
        result = reconstruct_plan_walls(
            _staggered_analysis(dashed_gap=True), _options()
        )

        self.assertAlmostEqual(_top_y(result.vertex_data, 240.0), 40.0)
        for start, end in _edge_coordinates(result.vertex_data):
            self.assertFalse(start[0] < 100.0 and end[0] > 200.0)

    def test_short_alignment_extension_does_not_require_continuous_plan_ink(self) -> None:
        result = reconstruct_plan_walls(
            _staggered_analysis(include_extension_ink=False), _options()
        )

        self.assertAlmostEqual(_top_y(result.vertex_data, 240.0), 40.0)
        for start, end in _edge_coordinates(result.vertex_data):
            self.assertFalse(start[0] < 100.0 and end[0] > 200.0)

    def test_existing_vertices_and_edges_remain_immutable_alignment_anchors(self) -> None:
        existing = _graph(*_LEFT_WALLS)
        original = existing.clone()

        result = reconstruct_plan_walls(
            _staggered_analysis(), _options(), existing
        )

        self.assertEqual(result.vertex_data.vertices[:4], original.vertices)
        self.assertEqual(result.vertex_data.edges[:2], original.edges)
        self.assertEqual(existing.vertices, original.vertices)
        self.assertEqual(existing.edges, original.edges)
        self.assertAlmostEqual(_top_y(result.vertex_data, 240.0), 40.0)

    def test_extension_cannot_cross_an_existing_wall(self) -> None:
        existing = _graph(((235.0, 44.0), (245.0, 44.0)))
        analysis = _staggered_analysis()
        ink = analysis.structural_ink.astype(np.uint8)
        cv2.line(ink, (235, 44), (245, 44), 1, 1)
        readonly_ink = ink.astype(np.bool_)
        readonly_ink.setflags(write=False)
        analysis = replace(
            analysis, structural_ink=readonly_ink, normalized_ink=readonly_ink
        )

        result = reconstruct_plan_walls(analysis, _options(), existing)

        self.assertEqual(result.vertex_data.vertices[:2], existing.vertices)
        self.assertEqual(result.vertex_data.edges[:1], existing.edges)
        self.assertAlmostEqual(_top_y(result.vertex_data, 240.0), 48.0)

    def test_repeated_slider_reconstruction_and_reversed_evidence_are_deterministic(
        self,
    ) -> None:
        analysis = _staggered_analysis()
        reversed_analysis = replace(
            analysis,
            line_evidence=tuple(
                WallLineEvidence(line.end, line.start, line.confidence)
                for line in reversed(analysis.line_evidence)
            ),
        )

        original = reconstruct_plan_walls(analysis, _options()).vertex_data
        repeated = reconstruct_plan_walls(analysis, _options()).vertex_data
        reversed_result = reconstruct_plan_walls(
            reversed_analysis, _options()
        ).vertex_data

        self.assertEqual(_edge_coordinates(original), _edge_coordinates(repeated))
        self.assertEqual(
            _edge_coordinates(original), _edge_coordinates(reversed_result)
        )
        self.assertEqual(_top_y(original, 240.0), 40.0)

    def test_legitimate_oblique_wall_pair_is_not_forced_onto_global_axes(self) -> None:
        analysis = PlanWallAnalysis(
            image_width=300,
            image_height=220,
            line_evidence=(
                WallLineEvidence((40.0, 70.0), (140.0, 170.0), 0.95),
                WallLineEvidence((52.0, 58.0), (152.0, 158.0), 0.95),
            ),
        )

        aligned = reconstruct_plan_walls(analysis, _options()).vertex_data
        disabled = reconstruct_plan_walls(
            analysis, _options(endpoint_alignment_tolerance_pixels=0.0)
        ).vertex_data

        self.assertEqual(_edge_coordinates(aligned), _edge_coordinates(disabled))
        self.assertIn(
            ((40.0, 70.0), (140.0, 170.0)), _edge_coordinates(aligned)
        )


# ### Standalone test entry point ###
if __name__ == "__main__":
    unittest.main()
