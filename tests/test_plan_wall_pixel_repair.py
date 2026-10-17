# ### Imports ###
from __future__ import annotations

import unittest
from dataclasses import replace
from itertools import pairwise
from unittest.mock import Mock, patch

import cv2
import numpy as np

from housemaker.automatic_wall_detection import (
    PlanWallAnalysis,
    PlanWallDetectionOptions,
    WallLineEvidence,
    analyze_plan_wall_evidence,
    reconstruct_plan_walls,
)
from housemaker.models import VertexData

# ### Fixture helpers ###
_Point = tuple[float, float]
_Edge = tuple[_Point, _Point]
_IMAGE_SIZE = 420


def _options(**overrides: float) -> PlanWallDetectionOptions:
    values = {
        "minimum_wall_separation_pixels": 8.0,
        "maximum_wall_separation_pixels": 20.0,
        "minimum_parallel_overlap_ratio": 0.7,
        "maximum_gap_bridge_pixels": 14.0,
        "endpoint_snap_distance_pixels": 4.0,
        "maximum_vertex_distance_pixels": 2.0,
        "minimum_wall_length_pixels": 10.0,
        "confidence_threshold": 0.35,
    }
    values.update(overrides)
    return PlanWallDetectionOptions(**values)


def _graph(*edges: _Edge) -> VertexData:
    graph = VertexData()
    vertices: dict[_Point, int] = {}
    for start, end in edges:
        for point in (start, end):
            if point not in vertices:
                vertices[point] = graph.add_vertex(*point).id
        graph.add_edge(vertices[start], vertices[end])
    return graph


def _image(*paths: tuple[_Point, ...], thickness: int = 3) -> np.ndarray:
    image = np.full((_IMAGE_SIZE, _IMAGE_SIZE), 255, dtype=np.uint8)
    for points in paths:
        for start, end in pairwise(points):
            cv2.line(
                image,
                tuple(round(value) for value in start),
                tuple(round(value) for value in end),
                0,
                thickness,
            )
    return image


def _analyze_without_detected_lines(image: np.ndarray) -> PlanWallAnalysis:
    detector = Mock()
    detector.detect.return_value = (None, None, None, None)
    with patch(
        "housemaker.automatic_wall_detection.cv2.createLineSegmentDetector",
        return_value=detector,
    ):
        analysis = analyze_plan_wall_evidence(image)
    if analysis.line_evidence:
        raise AssertionError("This fixture must exercise repair without detected lines")
    return analysis


def _edge_coordinates(graph: VertexData) -> set[_Edge]:
    points = {
        vertex.id: (round(vertex.x, 4), round(vertex.y, 4))
        for vertex in graph.vertices
    }
    return {
        tuple(sorted((points[edge.start_vertex_id], points[edge.end_vertex_id])))
        for edge in graph.edges
    }


def _reachable(graph: VertexData, start: _Point, end: _Point) -> bool:
    neighbors: dict[_Point, set[_Point]] = {}
    for first, second in _edge_coordinates(graph):
        neighbors.setdefault(first, set()).add(second)
        neighbors.setdefault(second, set()).add(first)
    pending = [start]
    visited: set[_Point] = set()
    while pending:
        current = pending.pop()
        if current == end:
            return True
        if current not in visited:
            visited.add(current)
            pending.extend(neighbors.get(current, set()) - visited)
    return False


def _point_on_graph(graph: VertexData, point: _Point, tolerance: float = 2.0) -> bool:
    query = np.asarray(point, dtype=float)
    for start, end in _edge_coordinates(graph):
        origin = np.asarray(start, dtype=float)
        direction = np.asarray(end, dtype=float) - origin
        fraction = np.clip(
            float(np.dot(query - origin, direction) / np.dot(direction, direction)),
            0.0,
            1.0,
        )
        if float(np.linalg.norm(query - origin - fraction * direction)) <= tolerance:
            return True
    return False


# ### Pixel tracing and local corner recovery regressions ###
class PlanWallPixelRecoveryTests(unittest.TestCase):
    def test_recovers_straight_ink_when_line_detector_finds_nothing(self) -> None:
        existing = _graph(
            ((20.0, 80.0), (60.0, 80.0)),
            ((300.0, 80.0), (350.0, 80.0)),
        )
        analysis = _analyze_without_detected_lines(
            _image(((20.0, 80.0), (350.0, 80.0)))
        )

        result = reconstruct_plan_walls(analysis, _options(), existing)

        self.assertTrue(
            _reachable(result.vertex_data, (20.0, 80.0), (350.0, 80.0))
        )
        self.assertIn(
            ((60.0, 80.0), (300.0, 80.0)), _edge_coordinates(result.vertex_data)
        )

    def test_recovers_long_outer_line_and_shorter_inner_line_through_corner(self) -> None:
        # The unequal horizontal lengths deliberately fail normal overlap pairing.
        path = (
            (190.0, 160.0), (190.0, 55.0), (20.0, 55.0),
            (20.0, 40.0), (340.0, 40.0), (340.0, 160.0),
        )
        existing = _graph(
            ((190.0, 200.0), path[0]),
            (path[-1], (340.0, 200.0)),
        )
        analysis = _analyze_without_detected_lines(
            _image(((190.0, 200.0), *path, (340.0, 200.0)))
        )

        result = reconstruct_plan_walls(analysis, _options(), existing)

        self.assertTrue(_reachable(result.vertex_data, path[0], path[-1]))
        for point in path:
            self.assertTrue(_point_on_graph(result.vertex_data, point), point)

    def test_recovers_stepped_outline_across_six_right_angle_turns(self) -> None:
        path = (
            (40.0, 120.0), (100.0, 120.0), (100.0, 40.0), (220.0, 40.0),
            (220.0, 240.0), (140.0, 240.0), (140.0, 320.0), (200.0, 320.0),
        )
        existing = _graph(
            ((20.0, 120.0), path[0]),
            (path[-1], (250.0, 320.0)),
        )
        analysis = _analyze_without_detected_lines(
            _image(((20.0, 120.0), *path, (250.0, 320.0)))
        )

        result = reconstruct_plan_walls(analysis, _options(), existing)

        self.assertTrue(_reachable(result.vertex_data, path[0], path[-1]))
        for point in path:
            self.assertTrue(_point_on_graph(result.vertex_data, point), point)

    def test_restores_thin_connector_removed_by_structural_cleanup(self) -> None:
        path = ((60.0, 80.0), (160.0, 80.0), (160.0, 260.0), (300.0, 260.0))
        existing = _graph(
            ((40.0, 80.0), path[0]),
            (path[-1], (340.0, 260.0)),
        )
        image = _image(
            ((40.0, 80.0), (160.0, 80.0)),
            ((160.0, 260.0), (340.0, 260.0)),
            thickness=5,
        )
        cv2.line(image, (160, 80), (160, 260), 0, 1)
        analysis = _analyze_without_detected_lines(image)
        self.assertFalse(analysis.structural_ink[180, 160])
        self.assertTrue(analysis.normalized_ink[180, 160])

        result = reconstruct_plan_walls(analysis, _options(), existing)

        self.assertTrue(_reachable(result.vertex_data, path[0], path[-1]))
        self.assertTrue(_point_on_graph(result.vertex_data, (160.0, 180.0)))

    def test_pixel_connector_splits_newly_detected_target_at_a_t_junction(self) -> None:
        existing = _graph(((160.0, 40.0), (160.0, 80.0)))
        image = _image(
            ((160.0, 40.0), (160.0, 240.0)),
            ((40.0, 240.0), (350.0, 240.0)),
            ((40.0, 254.0), (350.0, 254.0)),
        )
        # Only the target wall pair was detected; its connecting stroke must be
        # recovered from the image and must become a real shared graph vertex.
        analysis = replace(
            _analyze_without_detected_lines(image),
            line_evidence=(
                WallLineEvidence((40.0, 240.0), (350.0, 240.0), 0.95),
                WallLineEvidence((40.0, 254.0), (350.0, 254.0), 0.95),
            ),
        )

        result = reconstruct_plan_walls(analysis, _options(), existing)

        edges = _edge_coordinates(result.vertex_data)
        self.assertTrue(
            _reachable(result.vertex_data, (160.0, 40.0), (350.0, 240.0))
        )
        self.assertIn(((160.0, 80.0), (160.0, 240.0)), edges)
        self.assertIn(((40.0, 240.0), (160.0, 240.0)), edges)
        self.assertIn(((160.0, 240.0), (350.0, 240.0)), edges)
        self.assertNotIn(((40.0, 240.0), (350.0, 240.0)), edges)

    def test_keeps_supported_turn_at_branch_and_discards_unanchored_spur(self) -> None:
        existing = _graph(
            ((20.0, 120.0), (60.0, 120.0)),
            ((160.0, 260.0), (160.0, 320.0)),
        )
        analysis = _analyze_without_detected_lines(
            _image(
                ((20.0, 120.0), (300.0, 120.0)),
                ((160.0, 120.0), (160.0, 320.0)),
            )
        )

        result = reconstruct_plan_walls(analysis, _options(), existing)

        self.assertTrue(
            _reachable(result.vertex_data, (60.0, 120.0), (160.0, 260.0))
        )
        self.assertTrue(_point_on_graph(result.vertex_data, (160.0, 120.0)))
        self.assertFalse(_point_on_graph(result.vertex_data, (250.0, 120.0)))


# ### Unsupported image content safeguards ###
class PlanWallPixelSafetyTests(unittest.TestCase):
    def test_keeps_blank_doorway_gap_even_with_large_gap_and_snap_sliders(self) -> None:
        existing = _graph(
            ((20.0, 80.0), (60.0, 80.0)),
            ((300.0, 80.0), (350.0, 80.0)),
        )
        analysis = _analyze_without_detected_lines(
            _image(
                ((20.0, 80.0), (155.0, 80.0)),
                ((185.0, 80.0), (350.0, 80.0)),
            )
        )

        result = reconstruct_plan_walls(
            analysis,
            _options(
                maximum_gap_bridge_pixels=100.0,
                endpoint_snap_distance_pixels=40.0,
                maximum_vertex_distance_pixels=40.0,
            ),
            existing,
        )

        self.assertFalse(
            _reachable(result.vertex_data, (20.0, 80.0), (350.0, 80.0))
        )
        self.assertFalse(_point_on_graph(result.vertex_data, (170.0, 80.0)))

    def test_ignores_isolated_text_and_noise_without_a_supported_wall_route(self) -> None:
        existing = _graph(((20.0, 40.0), (70.0, 40.0)))
        image = _image(((20.0, 40.0), (70.0, 40.0)))
        cv2.putText(image, "ROOM 204", (90, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 1)
        cv2.rectangle(image, (240, 240), (290, 285), 0, 1)
        random = np.random.default_rng(82)
        coordinates = random.integers(100, 390, size=(80, 2))
        image[coordinates[:, 1], coordinates[:, 0]] = 0
        analysis = _analyze_without_detected_lines(image)

        result = reconstruct_plan_walls(analysis, _options(), existing)

        self.assertEqual(result.vertex_data, existing)
        self.assertEqual(result.added_edge_count, 0)

    def test_does_not_make_an_unanchored_ink_outline_into_a_wall(self) -> None:
        image = _image(
            ((80.0, 100.0), (160.0, 100.0), (160.0, 240.0), (300.0, 240.0))
        )
        analysis = _analyze_without_detected_lines(image)

        result = reconstruct_plan_walls(analysis, _options())

        self.assertEqual(result.added_edge_count, 0)
        self.assertEqual(result.vertex_data.edges, [])


# ### Cached analysis and stable graph regressions ###
class PlanWallPixelStabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = ((60.0, 80.0), (160.0, 80.0), (160.0, 260.0), (300.0, 260.0))
        self.existing = _graph(
            ((40.0, 80.0), self.path[0]),
            (self.path[-1], (340.0, 260.0)),
        )
        self.image = _image(((40.0, 80.0), *self.path, (340.0, 260.0)))
        self.analysis = _analyze_without_detected_lines(self.image)

    def test_repair_preserves_original_ids_positions_and_has_no_duplicates(self) -> None:
        baseline = self.existing.clone()

        result = reconstruct_plan_walls(self.analysis, _options(), self.existing)
        repeated = reconstruct_plan_walls(self.analysis, _options(), result.vertex_data)

        self.assertGreater(result.added_edge_count, 0)
        self.assertEqual(self.existing, baseline)
        self.assertEqual(
            result.vertex_data.vertices[:len(baseline.vertices)], baseline.vertices
        )
        self.assertEqual(result.vertex_data.edges[:len(baseline.edges)], baseline.edges)
        self.assertEqual(repeated.vertex_data, result.vertex_data)
        self.assertEqual(repeated.added_edge_count, 0)
        self.assertEqual(
            len(result.vertex_data.edges), len(_edge_coordinates(result.vertex_data))
        )

    def test_pixel_evidence_order_does_not_change_the_repaired_graph(self) -> None:
        self.assertGreater(len(self.analysis.pixel_line_evidence), 1)
        reordered = replace(
            self.analysis,
            pixel_line_evidence=tuple(reversed(self.analysis.pixel_line_evidence)),
        )

        first = reconstruct_plan_walls(self.analysis, _options(), self.existing)
        second = reconstruct_plan_walls(reordered, _options(), self.existing)

        self.assertGreater(first.added_edge_count, 0)
        self.assertEqual(first.vertex_data, second.vertex_data)

    def test_slider_updates_reuse_cached_pixels_and_traced_paths(self) -> None:
        original_pixels = self.analysis.normalized_ink.copy()
        original_image = self.image.copy()
        with (
            patch(
                "housemaker.automatic_wall_detection.trace_plan_ink_paths",
                side_effect=AssertionError("Pixel paths were retraced during a slider update"),
            ),
            patch(
                "housemaker.automatic_wall_detection.cv2.createLineSegmentDetector",
                side_effect=AssertionError("Line detection ran during a slider update"),
            ),
            patch(
                "housemaker.automatic_wall_detection.cv2.GaussianBlur",
                side_effect=AssertionError("Image normalization ran during a slider update"),
            ),
        ):
            first = reconstruct_plan_walls(self.analysis, _options(), self.existing)
            adjusted = reconstruct_plan_walls(
                self.analysis,
                _options(endpoint_snap_distance_pixels=6.0),
                self.existing,
            )

        self.assertGreater(first.added_edge_count, 0)
        self.assertGreater(adjusted.added_edge_count, 0)
        self.assertEqual(self.analysis.normalized_ink.dtype, np.dtype(np.bool_))
        self.assertEqual(self.analysis.normalized_ink.shape, self.image.shape)
        self.assertFalse(self.analysis.normalized_ink.flags.writeable)
        np.testing.assert_array_equal(self.analysis.normalized_ink, original_pixels)
        np.testing.assert_array_equal(self.image, original_image)

    def test_does_not_duplicate_detected_wall_faces_at_the_ink_centerline(self) -> None:
        image = _image(
            ((60.0, 80.0), (300.0, 80.0)),
            ((60.0, 94.0), (300.0, 94.0)),
            ((60.0, 80.0), (60.0, 94.0)),
            ((300.0, 80.0), (300.0, 94.0)),
        )
        analysis = replace(
            _analyze_without_detected_lines(image),
            line_evidence=(
                WallLineEvidence((60.0, 79.0), (300.0, 79.0), 0.95),
                WallLineEvidence((60.0, 95.0), (300.0, 95.0), 0.95),
            ),
        )
        without_pixels = replace(
            analysis, normalized_ink=None, pixel_line_evidence=(),
        )

        baseline = reconstruct_plan_walls(without_pixels, _options())
        repaired = reconstruct_plan_walls(analysis, _options())

        self.assertGreater(baseline.added_edge_count, 0)
        self.assertEqual(repaired.vertex_data, baseline.vertex_data)


# ### Direct test execution ###
if __name__ == "__main__":
    unittest.main()
