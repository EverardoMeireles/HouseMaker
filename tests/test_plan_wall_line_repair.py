# ### Imports ###
from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

import cv2
import numpy as np

from housemaker.automatic_wall_detection import (
    PlanWallAnalysis,
    PlanWallDetectionOptions,
    WallLineEvidence,
    _repair_best_supported_path,
    _Segment,
    analyze_plan_wall_evidence,
    reconstruct_plan_walls,
)
from housemaker.models import VertexData

# ### Fixture helpers ###
_IMAGE_SIZE = 240
_Point = tuple[float, float]
_EdgeCoordinates = tuple[_Point, _Point]


def _options(**overrides: float) -> PlanWallDetectionOptions:
    values = {
        "minimum_wall_separation_pixels": 8.0,
        "maximum_wall_separation_pixels": 20.0,
        "parallel_angle_tolerance_degrees": 5.0,
        "minimum_parallel_overlap_ratio": 0.7,
        "maximum_gap_bridge_pixels": 14.0,
        "endpoint_snap_distance_pixels": 4.0,
        "maximum_vertex_distance_pixels": 2.0,
        "minimum_wall_length_pixels": 10.0,
        "confidence_threshold": 0.35,
    }
    values.update(overrides)
    return PlanWallDetectionOptions(**values)


def _graph(*edges: _EdgeCoordinates) -> VertexData:
    graph = VertexData()
    vertex_ids: dict[_Point, int] = {}
    for start, end in edges:
        for point in (start, end):
            if point not in vertex_ids:
                vertex_ids[point] = graph.add_vertex(*point).id
        graph.add_edge(vertex_ids[start], vertex_ids[end])
    return graph


def _analysis(*lines: _EdgeCoordinates) -> PlanWallAnalysis:
    ink = np.zeros((_IMAGE_SIZE, _IMAGE_SIZE), dtype=np.uint8)
    for start, end in lines:
        cv2.line(
            ink,
            tuple(round(value) for value in start),
            tuple(round(value) for value in end),
            255,
            1,
        )
    structural_ink = ink.astype(np.bool_)
    structural_ink.setflags(write=False)
    return PlanWallAnalysis(
        image_width=_IMAGE_SIZE,
        image_height=_IMAGE_SIZE,
        line_evidence=tuple(
            WallLineEvidence(start, end, 0.95) for start, end in lines
        ),
        structural_ink=structural_ink,
    )


def _edge_coordinates(graph: VertexData) -> set[_EdgeCoordinates]:
    vertices = {
        vertex.id: (round(vertex.x, 4), round(vertex.y, 4))
        for vertex in graph.vertices
    }
    return {
        tuple(sorted((vertices[edge.start_vertex_id], vertices[edge.end_vertex_id])))
        for edge in graph.edges
    }


def _reachable(graph: VertexData, start: _Point, end: _Point) -> bool:
    neighbors: dict[_Point, set[_Point]] = {}
    for first, second in _edge_coordinates(graph):
        neighbors.setdefault(first, set()).add(second)
        neighbors.setdefault(second, set()).add(first)
    pending = [start]
    visited = set()
    while pending:
        current = pending.pop()
        if current == end:
            return True
        if current not in visited:
            visited.add(current)
            pending.extend(neighbors.get(current, set()) - visited)
    return False


# ### Supported connection regressions ###
class PlanWallLineRepairTests(unittest.TestCase):
    def test_recovers_line_between_newly_detected_wall_pairs(self) -> None:
        analysis = _analysis(
            ((30.0, 20.0), (100.0, 20.0)),
            ((30.0, 32.0), (100.0, 32.0)),
            ((30.0, 180.0), (100.0, 180.0)),
            ((30.0, 192.0), (100.0, 192.0)),
            ((30.0, 32.0), (30.0, 180.0)),
        )

        result = reconstruct_plan_walls(analysis, _options())

        self.assertIn(
            ((30.0, 32.0), (30.0, 180.0)),
            _edge_coordinates(result.vertex_data),
        )
        self.assertTrue(
            _reachable(result.vertex_data, (100.0, 20.0), (100.0, 192.0))
        )

    def test_recovers_long_unpaired_line_between_supported_endpoints(self) -> None:
        existing = _graph(
            ((20.0, 20.0), (50.0, 20.0)),
            ((150.0, 20.0), (190.0, 20.0)),
        )
        analysis = _analysis(((50.0, 20.0), (150.0, 20.0)))

        result = reconstruct_plan_walls(analysis, _options(), existing)

        self.assertIn(
            ((50.0, 20.0), (150.0, 20.0)),
            _edge_coordinates(result.vertex_data),
        )
        self.assertTrue(
            _reachable(result.vertex_data, (20.0, 20.0), (190.0, 20.0))
        )

    def test_recovers_unpaired_perpendicular_connector(self) -> None:
        existing = _graph(
            ((20.0, 40.0), (60.0, 40.0)),
            ((60.0, 70.0), (110.0, 70.0)),
        )
        analysis = _analysis(((60.0, 40.0), (60.0, 70.0)))

        result = reconstruct_plan_walls(analysis, _options(), existing)

        self.assertIn(
            ((60.0, 40.0), (60.0, 70.0)),
            _edge_coordinates(result.vertex_data),
        )

    def test_follows_multiple_supported_segments_around_right_angle_turns(self) -> None:
        existing = _graph(
            ((20.0, 20.0), (50.0, 20.0)),
            ((130.0, 140.0), (170.0, 140.0)),
        )
        path = (
            ((50.0, 20.0), (100.0, 20.0)),
            ((100.0, 20.0), (100.0, 140.0)),
            ((100.0, 140.0), (130.0, 140.0)),
        )

        result = reconstruct_plan_walls(_analysis(*path), _options(), existing)

        self.assertTrue(set(path).issubset(_edge_coordinates(result.vertex_data)))
        self.assertTrue(
            _reachable(result.vertex_data, (20.0, 20.0), (170.0, 140.0))
        )

    def test_splits_generated_target_wall_at_supported_t_junction(self) -> None:
        existing = _graph(((90.0, 10.0), (90.0, 30.0)))
        analysis = _analysis(
            ((20.0, 140.0), (180.0, 140.0)),
            ((20.0, 152.0), (180.0, 152.0)),
            ((90.0, 30.0), (90.0, 140.0)),
        )

        result = reconstruct_plan_walls(analysis, _options(), existing)

        edges = _edge_coordinates(result.vertex_data)
        self.assertIn(((90.0, 30.0), (90.0, 140.0)), edges)
        self.assertIn(((20.0, 140.0), (90.0, 140.0)), edges)
        self.assertIn(((90.0, 140.0), (180.0, 140.0)), edges)
        self.assertNotIn(((20.0, 140.0), (180.0, 140.0)), edges)


# ### End-to-end OpenCV detection regressions ###
class PlanWallLineRepairImageTests(unittest.TestCase):
    def test_detected_unpaired_line_connects_two_new_wall_pairs(self) -> None:
        image = np.full((260, 260), 255, dtype=np.uint8)
        for start, end in (
            ((30, 20), (150, 20)),
            ((30, 32), (150, 32)),
            ((30, 210), (150, 210)),
            ((30, 222), (150, 222)),
            ((30, 32), (30, 210)),
        ):
            cv2.line(image, start, end, 0, 3)
        analysis = analyze_plan_wall_evidence(image)

        initial = reconstruct_plan_walls(
            replace(analysis, structural_ink=None, normalized_ink=None), _options()
        )
        repaired = reconstruct_plan_walls(analysis, _options())

        for graph, should_connect in (
            (initial.vertex_data, False),
            (repaired.vertex_data, True),
        ):
            upper = [
                (round(vertex.x, 4), round(vertex.y, 4))
                for vertex in graph.vertices
                if vertex.x > 140.0 and vertex.y < 40.0
            ]
            lower = [
                (round(vertex.x, 4), round(vertex.y, 4))
                for vertex in graph.vertices
                if vertex.x > 140.0 and vertex.y > 200.0
            ]
            self.assertTrue(upper)
            self.assertTrue(lower)
            self.assertEqual(
                any(_reachable(graph, start, end) for start in upper for end in lower),
                should_connect,
            )

    def test_detected_line_reuses_exact_existing_wall_endpoints(self) -> None:
        image = np.full((260, 260), 255, dtype=np.uint8)
        cv2.line(image, (50, 100), (180, 100), 0, 3)
        existing = _graph(
            ((20.0, 100.0), (50.0, 100.0)),
            ((180.0, 100.0), (220.0, 100.0)),
        )

        result = reconstruct_plan_walls(
            analyze_plan_wall_evidence(image), _options(), existing
        )

        self.assertEqual(result.added_edge_count, 1)
        self.assertEqual(result.vertex_data.vertices, existing.vertices)
        self.assertEqual(result.vertex_data.edges[:2], existing.edges)
        self.assertIn(
            ((50.0, 100.0), (180.0, 100.0)),
            _edge_coordinates(result.vertex_data),
        )
        self.assertTrue(
            _reachable(result.vertex_data, (20.0, 100.0), (220.0, 100.0))
        )


# ### Unsupported connection safeguards ###
class PlanWallLineRepairSafetyTests(unittest.TestCase):
    def test_large_endpoint_snap_does_not_extend_across_white_image_gaps(self) -> None:
        existing = _graph(
            ((20.0, 30.0), (50.0, 30.0)),
            ((130.0, 30.0), (170.0, 30.0)),
        )
        analysis = _analysis(((60.0, 30.0), (120.0, 30.0)))

        result = reconstruct_plan_walls(
            analysis,
            _options(
                endpoint_snap_distance_pixels=30.0,
                maximum_vertex_distance_pixels=30.0,
            ),
            existing,
        )

        self.assertFalse(
            _reachable(result.vertex_data, (20.0, 30.0), (170.0, 30.0))
        )
        self.assertEqual(result.added_edge_count, 0)

    def test_does_not_bridge_blank_doorway_with_large_gap_slider(self) -> None:
        existing = _graph(
            ((20.0, 30.0), (50.0, 30.0)),
            ((150.0, 30.0), (190.0, 30.0)),
        )
        analysis = _analysis(
            ((50.0, 30.0), (85.0, 30.0)),
            ((115.0, 30.0), (150.0, 30.0)),
        )

        result = reconstruct_plan_walls(
            analysis,
            _options(maximum_gap_bridge_pixels=100.0),
            existing,
        )

        self.assertFalse(
            _reachable(result.vertex_data, (20.0, 30.0), (190.0, 30.0))
        )
        for start, end in _edge_coordinates(result.vertex_data):
            self.assertFalse(start[0] <= 85.0 and end[0] >= 115.0)

    def test_rejects_isolated_annotation_without_structural_anchors(self) -> None:
        existing = _graph(((20.0, 20.0), (60.0, 20.0)))
        analysis = _analysis(((30.0, 100.0), (180.0, 100.0)))

        result = reconstruct_plan_walls(analysis, _options(), existing)

        self.assertEqual(result.vertex_data.vertices, existing.vertices)
        self.assertEqual(result.vertex_data.edges, existing.edges)
        self.assertEqual(result.added_edge_count, 0)

    def test_rejects_single_anchored_line_that_has_no_relevant_target(self) -> None:
        existing = _graph(((20.0, 20.0), (60.0, 20.0)))
        analysis = _analysis(((60.0, 20.0), (60.0, 100.0)))

        result = reconstruct_plan_walls(analysis, _options(), existing)

        self.assertEqual(result.vertex_data.edges, existing.edges)
        self.assertEqual(result.added_edge_count, 0)

    def test_rejects_diagonal_shortcut_between_axis_aligned_wall_endpoints(self) -> None:
        existing = _graph(
            ((20.0, 20.0), (50.0, 20.0)),
            ((130.0, 100.0), (170.0, 100.0)),
        )
        analysis = _analysis(((50.0, 20.0), (130.0, 100.0)))

        result = reconstruct_plan_walls(analysis, _options(), existing)

        self.assertEqual(result.vertex_data.edges, existing.edges)
        self.assertEqual(result.added_edge_count, 0)

    def test_does_not_restore_redundant_interior_wall_layer(self) -> None:
        analysis = _analysis(
            ((10.0, 20.0), (140.0, 20.0)),
            ((10.0, 30.0), (140.0, 30.0)),
            ((10.0, 40.0), (140.0, 40.0)),
        )
        existing = _graph(
            ((10.0, 10.0), (10.0, 30.0)),
            ((140.0, 10.0), (140.0, 30.0)),
        )

        result = reconstruct_plan_walls(
            analysis,
            _options(maximum_wall_separation_pixels=30.0),
            existing,
        )

        self.assertNotIn(
            ((10.0, 30.0), (140.0, 30.0)),
            _edge_coordinates(result.vertex_data),
        )
        self.assertFalse(
            any(start[1] == end[1] == 30.0 for start, end in _edge_coordinates(result.vertex_data))
        )

    def test_clips_interior_section_of_a_longer_guide_crossing_wall_strip(self) -> None:
        analysis = _analysis(
            ((20.0, 20.0), (80.0, 20.0)),
            ((20.0, 40.0), (80.0, 40.0)),
            ((10.0, 30.0), (90.0, 30.0)),
        )
        existing = _graph(
            ((0.0, 30.0), (10.0, 30.0)),
            ((90.0, 30.0), (100.0, 30.0)),
        )

        result = reconstruct_plan_walls(analysis, _options(), existing)

        for start, end in _edge_coordinates(result.vertex_data):
            interior_horizontal = (
                start[1] == end[1] == 30.0
                and end[0] > 20.0
                and start[0] < 80.0
            )
            self.assertFalse(interior_horizontal, (start, end))


# ### Competing-route priority regressions ###
class PlanWallLineRepairPriorityTests(unittest.TestCase):
    def test_prefers_straight_continuation_to_shorter_right_angle_branch(self) -> None:
        points = {
            1: np.asarray((50.0, 100.0)),
            2: np.asarray((150.0, 100.0)),
            3: np.asarray((50.0, 140.0)),
        }
        source_wall = _Segment(np.asarray((20.0, 100.0)), points[1], 1.0)
        straight = _Segment(points[1], points[2], 1.0)
        turn = _Segment(points[1], points[3], 1.0)
        straight_target = _Segment(points[2], np.asarray((190.0, 100.0)), 1.0)
        turned_target = _Segment(points[3], np.asarray((90.0, 140.0)), 1.0)

        path = _repair_best_supported_path(
            source_id=1,
            points=points,
            adjacency={1: [(2, straight), (3, turn)]},
            incidents={1: [source_wall], 2: [straight_target], 3: [turned_target]},
            anchors={1, 2, 3},
            obstructions={1, 2, 3},
            accepted={},
            options=_options(),
        )

        self.assertEqual(path, (1, 2))

    def test_does_not_backtrack_into_a_shorter_target_behind_source(self) -> None:
        points = {
            1: np.asarray((50.0, 100.0)),
            2: np.asarray((150.0, 100.0)),
            3: np.asarray((10.0, 100.0)),
        }
        source_wall = _Segment(np.asarray((20.0, 100.0)), points[1], 1.0)
        straight = _Segment(points[1], points[2], 1.0)
        backward = _Segment(points[1], points[3], 1.0)
        straight_target = _Segment(points[2], np.asarray((190.0, 100.0)), 1.0)
        backward_target = _Segment(points[3], np.asarray((10.0, 140.0)), 1.0)

        path = _repair_best_supported_path(
            source_id=1,
            points=points,
            adjacency={1: [(2, straight), (3, backward)]},
            incidents={1: [source_wall], 2: [straight_target], 3: [backward_target]},
            anchors={1, 2, 3},
            obstructions={1, 2, 3},
            accepted={},
            options=_options(),
        )

        self.assertEqual(path, (1, 2))


# ### Stable graph and reusable analysis regressions ###
class PlanWallLineRepairStabilityTests(unittest.TestCase):
    def test_preserves_existing_records_and_never_adds_duplicate_connections(self) -> None:
        existing = _graph(
            ((20.0, 20.0), (50.0, 20.0)),
            ((150.0, 20.0), (190.0, 20.0)),
        )
        baseline = existing.clone()
        analysis = _analysis(((50.0, 20.0), (150.0, 20.0)))

        result = reconstruct_plan_walls(analysis, _options(), existing)
        repeated = reconstruct_plan_walls(analysis, _options(), result.vertex_data)

        self.assertEqual(existing, baseline)
        self.assertEqual(result.vertex_data.vertices[:4], baseline.vertices)
        self.assertEqual(result.vertex_data.edges[:2], baseline.edges)
        self.assertEqual(result.added_edge_count, 1)
        self.assertEqual(repeated.added_edge_count, 0)
        self.assertEqual(repeated.vertex_data.edges, result.vertex_data.edges)
        self.assertEqual(
            len(result.vertex_data.edges), len(_edge_coordinates(result.vertex_data))
        )

    def test_repair_result_is_independent_of_evidence_order(self) -> None:
        existing = _graph(
            ((20.0, 20.0), (50.0, 20.0)),
            ((130.0, 140.0), (170.0, 140.0)),
        )
        path = (
            ((50.0, 20.0), (100.0, 20.0)),
            ((100.0, 20.0), (100.0, 140.0)),
            ((100.0, 140.0), (130.0, 140.0)),
        )

        first = reconstruct_plan_walls(_analysis(*path), _options(), existing)
        reordered = reconstruct_plan_walls(
            _analysis(*reversed(path)), _options(), existing
        )

        self.assertEqual(first.vertex_data, reordered.vertex_data)
        self.assertGreater(first.added_edge_count, 0)

    def test_slider_reconstruction_reuses_cached_read_only_ink_without_opencv(self) -> None:
        image = np.full((240, 240), 255, dtype=np.uint8)
        cv2.line(image, (30, 60), (210, 60), 0, 3)
        cv2.line(image, (30, 76), (210, 76), 0, 3)
        original_image = image.copy()
        analysis = analyze_plan_wall_evidence(image)
        original_ink = analysis.structural_ink.copy()

        with (
            patch(
                "housemaker.automatic_wall_detection.cv2.createLineSegmentDetector",
                side_effect=AssertionError("The image detector ran during a slider update"),
            ),
            patch(
                "housemaker.automatic_wall_detection.cv2.GaussianBlur",
                side_effect=AssertionError("The image was normalized again"),
            ),
            patch(
                "housemaker.automatic_wall_detection.cv2.morphologyEx",
                side_effect=AssertionError("The image was filtered again"),
            ),
        ):
            first = reconstruct_plan_walls(analysis, _options())
            adjusted = reconstruct_plan_walls(
                analysis, _options(endpoint_snap_distance_pixels=6.0)
            )

        self.assertGreater(first.added_edge_count, 0)
        self.assertGreater(adjusted.added_edge_count, 0)
        self.assertEqual(analysis.structural_ink.shape, image.shape)
        self.assertEqual(analysis.structural_ink.dtype, np.dtype(np.bool_))
        self.assertFalse(analysis.structural_ink.flags.writeable)
        np.testing.assert_array_equal(image, original_image)
        np.testing.assert_array_equal(analysis.structural_ink, original_ink)


# ### Direct test execution ###
if __name__ == "__main__":
    unittest.main()
