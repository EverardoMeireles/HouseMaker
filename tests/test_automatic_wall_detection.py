# ### Imports ###
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from housemaker.automatic_wall_detection import (
    PlanWallAnalysis,
    PlanWallDetectionCancelled,
    PlanWallDetectionError,
    PlanWallDetectionOptions,
    WallLineEvidence,
    _append_segments_to_graph,
    _close_wall_face_pair_ends,
    _Segment,
    _WallFacePair,
    analyze_plan_wall_evidence,
    analyze_plan_wall_image,
    reconstruct_plan_walls,
)
from housemaker.models import Edge, Vertex, VertexData


# ### Fixture helpers ###
def _options(**overrides: float) -> PlanWallDetectionOptions:
    values: dict[str, float] = {
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


def _evidence(
    start: tuple[float, float],
    end: tuple[float, float],
    confidence: float = 0.95,
) -> WallLineEvidence:
    return WallLineEvidence(start=start, end=end, confidence=confidence)


def _analysis(*lines: WallLineEvidence) -> PlanWallAnalysis:
    return PlanWallAnalysis(
        image_width=160,
        image_height=120,
        line_evidence=tuple(lines),
    )


def _segment(
    start: tuple[float, float],
    end: tuple[float, float],
) -> _Segment:
    """Build one exact support segment for graph-construction regressions."""

    return _Segment(
        start=np.asarray(start, dtype=np.float64),
        end=np.asarray(end, dtype=np.float64),
        confidence=1.0,
    )


def _wall_face_pair(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> _WallFacePair:
    """Build one exact pair of detected wall faces."""

    return _WallFacePair(
        first=_segment(first_start, first_end),
        second=_segment(second_start, second_end),
    )


def _segment_coordinate_key(
    segment: _Segment,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return direction-independent rounded coordinates for one segment."""

    endpoints = sorted(
        (
            (round(float(segment.start[0]), 4), round(float(segment.start[1]), 4)),
            (round(float(segment.end[0]), 4), round(float(segment.end[1]), 4)),
        )
    )
    return endpoints[0], endpoints[1]


def _wall_face_pair_coordinate_keys(
    wall_face_pairs: list[_WallFacePair],
) -> list[
    tuple[
        tuple[tuple[float, float], tuple[float, float]],
        tuple[tuple[float, float], tuple[float, float]],
    ]
]:
    """Return pair and endpoint order-independent wall geometry."""

    return sorted(
        tuple(
            sorted(
                (
                    _segment_coordinate_key(pair.first),
                    _segment_coordinate_key(pair.second),
                )
            )
        )
        for pair in wall_face_pairs
    )


def _edge_coordinates(
    vertex_data: VertexData,
) -> set[tuple[tuple[float, float], tuple[float, float]]]:
    vertices = {
        vertex.id: (round(vertex.x, 4), round(vertex.y, 4))
        for vertex in vertex_data.vertices
    }
    coordinates = set()
    for edge in vertex_data.edges:
        endpoints = sorted(
            (vertices[edge.start_vertex_id], vertices[edge.end_vertex_id])
        )
        coordinates.add((endpoints[0], endpoints[1]))
    return coordinates


def _add_parallel_existing_walls(
    vertex_data: VertexData,
    *,
    start_x: float = 10.0,
    end_x: float = 110.0,
    first_y: float = 20.0,
    second_y: float = 32.0,
) -> None:
    """Add one pair of immutable horizontal wall faces to a graph."""

    first_start = vertex_data.add_vertex(start_x, first_y)
    first_end = vertex_data.add_vertex(end_x, first_y)
    second_start = vertex_data.add_vertex(start_x, second_y)
    second_end = vertex_data.add_vertex(end_x, second_y)
    vertex_data.add_edge(first_start.id, first_end.id)
    vertex_data.add_edge(second_start.id, second_end.id)


def _add_split_parallel_existing_walls(
    vertex_data: VertexData,
    *,
    start_x: float = 10.0,
    split_x: float = 60.0,
    end_x: float = 110.0,
    first_y: float = 20.0,
    second_y: float = 32.0,
) -> None:
    """Add a wall pair whose faces are noded into multiple edge records."""

    for y_value in (first_y, second_y):
        start = vertex_data.add_vertex(start_x, y_value)
        split = vertex_data.add_vertex(split_x, y_value)
        end = vertex_data.add_vertex(end_x, y_value)
        vertex_data.add_edge(start.id, split.id)
        vertex_data.add_edge(split.id, end.id)


# ### Image analysis tests ###
class PlanWallImageAnalysisTests(unittest.TestCase):
    def test_extracts_reusable_line_evidence_without_modifying_image(self) -> None:
        image = np.full((100, 140), 255, dtype=np.uint8)
        cv2.line(image, (10, 30), (130, 30), 0, 2)
        cv2.line(image, (10, 44), (130, 44), 0, 2)
        original = image.copy()

        analysis = analyze_plan_wall_evidence(image)

        np.testing.assert_array_equal(image, original)
        self.assertEqual(analysis.image_width, 140)
        self.assertEqual(analysis.image_height, 100)
        self.assertGreaterEqual(len(analysis.line_evidence), 2)
        self.assertTrue(
            all(0.0 <= line.confidence <= 1.0 for line in analysis.line_evidence)
        )

    def test_loads_unicode_image_path(self) -> None:
        image = np.full((60, 80), 255, dtype=np.uint8)
        cv2.line(image, (8, 20), (72, 20), 0, 2)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "étage corrigé.png"
            encoded, payload = cv2.imencode(".png", image)
            self.assertTrue(encoded)
            payload.tofile(path)

            analysis = analyze_plan_wall_image(path)

        self.assertEqual((analysis.image_width, analysis.image_height), (80, 60))

    def test_honors_cancellation_before_loading(self) -> None:
        with self.assertRaises(PlanWallDetectionCancelled):
            analyze_plan_wall_image("does-not-matter.png", lambda: True)

    def test_rejects_missing_or_invalid_inputs_with_safe_errors(self) -> None:
        with self.assertRaises(PlanWallDetectionError):
            analyze_plan_wall_image("missing-plan.png")
        with self.assertRaises(PlanWallDetectionError):
            analyze_plan_wall_evidence(np.zeros((5, 5, 3), dtype=np.uint8))

    def test_marks_thin_annotations_below_default_wall_confidence(self) -> None:
        image = np.full((1000, 1000), 255, dtype=np.uint8)
        for y_value in (100, 130):
            cv2.line(image, (100, y_value), (900, y_value), 0, 5)
        for y_value in (300, 330):
            cv2.line(image, (150, y_value), (850, y_value), 0, 1)

        analysis = analyze_plan_wall_evidence(image)
        result = reconstruct_plan_walls(analysis)

        structural_evidence = [
            line
            for line in analysis.line_evidence
            if 90.0 <= (line.start[1] + line.end[1]) / 2.0 <= 140.0
            and line.length > 500.0
        ]
        annotation_evidence = [
            line
            for line in analysis.line_evidence
            if 290.0 <= (line.start[1] + line.end[1]) / 2.0 <= 340.0
            and line.length > 500.0
        ]
        self.assertTrue(structural_evidence)
        self.assertTrue(annotation_evidence)
        self.assertGreater(
            min(line.confidence for line in structural_evidence),
            max(line.confidence for line in annotation_evidence),
        )
        self.assertEqual(result.added_edge_count, 4)
        incident_counts = {
            vertex.id: sum(
                vertex.id in (edge.start_vertex_id, edge.end_vertex_id)
                for edge in result.vertex_data.edges
            )
            for vertex in result.vertex_data.vertices
        }
        self.assertEqual(set(incident_counts.values()), {2})
        self.assertTrue(all(vertex.y < 200.0 for vertex in result.vertex_data.vertices))


# ### Reconstruction tests ###
class PlanWallReconstructionTests(unittest.TestCase):
    def test_adds_both_faces_of_a_parallel_wall_pair(self) -> None:
        analysis = _analysis(
            _evidence((10.0, 20.0), (110.0, 20.0)),
            _evidence((10.0, 32.0), (110.0, 32.0)),
        )

        result = reconstruct_plan_walls(analysis, _options())

        self.assertEqual(result.added_edge_count, 4)
        self.assertEqual(
            _edge_coordinates(result.vertex_data),
            {
                ((10.0, 20.0), (10.0, 32.0)),
                ((10.0, 20.0), (110.0, 20.0)),
                ((10.0, 32.0), (110.0, 32.0)),
                ((110.0, 20.0), (110.0, 32.0)),
            },
        )

    def test_bridges_collinear_doorway_sized_gaps_before_pairing(self) -> None:
        analysis = _analysis(
            _evidence((10.0, 20.0), (48.0, 20.0)),
            _evidence((58.0, 20.0), (110.0, 20.0)),
            _evidence((10.0, 32.0), (48.0, 32.0)),
            _evidence((58.0, 32.0), (110.0, 32.0)),
        )

        bridged = reconstruct_plan_walls(analysis, _options())
        unbridged = reconstruct_plan_walls(
            analysis,
            _options(maximum_gap_bridge_pixels=4.0),
        )

        self.assertEqual(
            _edge_coordinates(bridged.vertex_data),
            {
                ((10.0, 20.0), (10.0, 32.0)),
                ((10.0, 20.0), (110.0, 20.0)),
                ((10.0, 32.0), (110.0, 32.0)),
                ((110.0, 20.0), (110.0, 32.0)),
            },
        )
        self.assertEqual(
            _edge_coordinates(unbridged.vertex_data),
            {
                ((10.0, 20.0), (10.0, 32.0)),
                ((10.0, 20.0), (48.0, 20.0)),
                ((10.0, 32.0), (48.0, 32.0)),
                ((48.0, 20.0), (48.0, 32.0)),
                ((58.0, 20.0), (58.0, 32.0)),
                ((58.0, 20.0), (110.0, 20.0)),
                ((58.0, 32.0), (110.0, 32.0)),
                ((110.0, 20.0), (110.0, 32.0)),
            },
        )

    def test_nodes_generated_crossings_into_shared_vertices(self) -> None:
        analysis = _analysis(
            _evidence((10.0, 20.0), (120.0, 20.0)),
            _evidence((10.0, 32.0), (120.0, 32.0)),
            _evidence((50.0, 5.0), (50.0, 90.0)),
            _evidence((62.0, 5.0), (62.0, 90.0)),
        )

        result = reconstruct_plan_walls(analysis, _options())

        intersection_vertices = {
            (vertex.x, vertex.y)
            for vertex in result.vertex_data.vertices
            if vertex.x in (50.0, 62.0) and vertex.y in (20.0, 32.0)
        }
        self.assertEqual(
            intersection_vertices,
            {(50.0, 20.0), (50.0, 32.0), (62.0, 20.0), (62.0, 32.0)},
        )
        self.assertEqual(result.added_edge_count, 16)
        self.assertTrue(
            {
                ((10.0, 20.0), (10.0, 32.0)),
                ((120.0, 20.0), (120.0, 32.0)),
                ((50.0, 5.0), (62.0, 5.0)),
                ((50.0, 90.0), (62.0, 90.0)),
            }.issubset(_edge_coordinates(result.vertex_data))
        )

    def test_allows_oblique_parallel_wall_faces(self) -> None:
        analysis = _analysis(
            _evidence((10.0, 20.0), (90.0, 60.0)),
            _evidence((5.0, 30.0), (85.0, 70.0)),
        )

        result = reconstruct_plan_walls(
            analysis,
            _options(
                minimum_wall_separation_pixels=7.0,
                maximum_wall_separation_pixels=14.0,
            ),
        )

        self.assertEqual(
            _edge_coordinates(result.vertex_data),
            {
                ((5.0, 30.0), (10.0, 20.0)),
                ((5.0, 30.0), (85.0, 70.0)),
                ((10.0, 20.0), (90.0, 60.0)),
                ((85.0, 70.0), (90.0, 60.0)),
            },
        )

    def test_confidence_slider_filters_weak_pairs(self) -> None:
        analysis = _analysis(
            _evidence((10.0, 20.0), (110.0, 20.0), 0.4),
            _evidence((10.0, 32.0), (110.0, 32.0), 0.4),
        )

        result = reconstruct_plan_walls(
            analysis,
            _options(confidence_threshold=0.8),
        )

        self.assertEqual(result.added_edge_count, 0)

    def test_maximum_vertex_distance_consolidates_close_generated_endpoints(
        self,
    ) -> None:
        analysis = _analysis(
            _evidence((10.0, 20.0), (48.0, 20.0)),
            _evidence((51.0, 20.0), (110.0, 20.0)),
            _evidence((10.0, 32.0), (48.0, 32.0)),
            _evidence((51.0, 32.0), (110.0, 32.0)),
        )

        separate = reconstruct_plan_walls(
            analysis,
            _options(
                maximum_gap_bridge_pixels=0.0,
                endpoint_snap_distance_pixels=0.0,
                maximum_vertex_distance_pixels=0.0,
            ),
        )
        consolidated = reconstruct_plan_walls(
            analysis,
            _options(
                maximum_gap_bridge_pixels=0.0,
                endpoint_snap_distance_pixels=0.0,
                maximum_vertex_distance_pixels=4.0,
            ),
        )

        self.assertEqual(len(separate.vertex_data.vertices), 8)
        self.assertEqual(len(consolidated.vertex_data.vertices), 6)
        for y_value in (20.0, 32.0):
            nearby_vertices = [
                vertex
                for vertex in consolidated.vertex_data.vertices
                if vertex.y == y_value and 47.0 <= vertex.x <= 52.0
            ]
            self.assertEqual(len(nearby_vertices), 1)

    def test_maximum_vertex_distance_uses_full_value_for_collinear_gaps(
        self,
    ) -> None:
        analysis = _analysis(
            _evidence((10.0, 20.0), (48.0, 20.0)),
            _evidence((58.0, 20.0), (110.0, 20.0)),
            _evidence((10.0, 32.0), (48.0, 32.0)),
            _evidence((58.0, 32.0), (110.0, 32.0)),
        )

        result = reconstruct_plan_walls(
            analysis,
            _options(
                maximum_gap_bridge_pixels=0.0,
                endpoint_snap_distance_pixels=0.0,
                maximum_vertex_distance_pixels=12.0,
            ),
        )

        self.assertEqual(
            _edge_coordinates(result.vertex_data),
            {
                ((10.0, 20.0), (10.0, 32.0)),
                ((10.0, 20.0), (53.0, 20.0)),
                ((10.0, 32.0), (53.0, 32.0)),
                ((53.0, 20.0), (110.0, 20.0)),
                ((53.0, 32.0), (110.0, 32.0)),
                ((110.0, 20.0), (110.0, 32.0)),
            },
        )

    def test_maximum_vertex_distance_does_not_collapse_wall_thickness(self) -> None:
        analysis = _analysis(
            _evidence((10.0, 20.0), (110.0, 20.0)),
            _evidence((10.0, 28.0), (110.0, 28.0)),
        )

        result = reconstruct_plan_walls(
            analysis,
            _options(
                minimum_wall_separation_pixels=8.0,
                maximum_vertex_distance_pixels=50.0,
            ),
        )

        self.assertEqual(result.added_edge_count, 4)
        self.assertEqual(
            _edge_coordinates(result.vertex_data),
            {
                ((10.0, 20.0), (10.0, 28.0)),
                ((10.0, 20.0), (110.0, 20.0)),
                ((10.0, 28.0), (110.0, 28.0)),
                ((110.0, 20.0), (110.0, 28.0)),
            },
        )

    def test_maximum_vertex_distance_protects_staggered_wall_faces(self) -> None:
        analysis = _analysis(
            _evidence((10.0, 20.0), (100.0, 20.0)),
            _evidence((20.0, 28.0), (110.0, 28.0)),
        )

        result = reconstruct_plan_walls(
            analysis,
            _options(maximum_vertex_distance_pixels=15.0),
        )

        self.assertEqual(
            _edge_coordinates(result.vertex_data),
            {
                ((10.0, 20.0), (100.0, 20.0)),
                ((20.0, 28.0), (110.0, 28.0)),
            },
        )

    def test_collapsed_short_segments_do_not_leave_isolated_vertices(self) -> None:
        analysis = _analysis(
            _evidence((10.0, 20.0), (18.0, 20.0)),
            _evidence((10.0, 40.0), (18.0, 40.0)),
        )

        result = reconstruct_plan_walls(
            analysis,
            _options(
                minimum_wall_separation_pixels=20.0,
                maximum_wall_separation_pixels=30.0,
                maximum_gap_bridge_pixels=0.0,
                endpoint_snap_distance_pixels=0.0,
                maximum_vertex_distance_pixels=50.0,
                minimum_wall_length_pixels=5.0,
            ),
        )

        self.assertEqual(result.added_edge_count, 0)
        self.assertEqual(result.vertex_data.vertices, [])
        self.assertEqual(result.vertex_data.edges, [])

    def test_uses_outer_faces_of_a_redundant_parallel_outline_stack(self) -> None:
        analysis = _analysis(
            _evidence((10.0, 20.0), (140.0, 20.0)),
            _evidence((10.0, 30.0), (140.0, 30.0)),
            _evidence((10.0, 40.0), (140.0, 40.0)),
        )

        result = reconstruct_plan_walls(
            analysis,
            _options(maximum_wall_separation_pixels=30.0),
        )

        self.assertEqual(
            _edge_coordinates(result.vertex_data),
            {
                ((10.0, 20.0), (10.0, 40.0)),
                ((10.0, 20.0), (140.0, 20.0)),
                ((10.0, 40.0), (140.0, 40.0)),
                ((140.0, 20.0), (140.0, 40.0)),
            },
        )

    def test_rejects_short_annotation_paired_with_one_long_wall_face(self) -> None:
        analysis = _analysis(
            _evidence((10.0, 20.0), (140.0, 20.0)),
            _evidence((50.0, 32.0), (90.0, 32.0)),
        )

        result = reconstruct_plan_walls(analysis, _options())

        self.assertEqual(result.added_edge_count, 0)


# ### Existing-graph and determinism tests ###
class ExistingWallAnchorTests(unittest.TestCase):
    def test_maximum_vertex_distance_reuses_a_nearby_existing_wall_vertex(
        self,
    ) -> None:
        existing = VertexData(
            vertices=[
                Vertex(id=1, x=10.0, y=20.0),
                Vertex(id=2, x=45.0, y=20.0),
            ],
            edges=[Edge(start_vertex_id=1, end_vertex_id=2)],
            _next_vertex_id=3,
        )
        original = existing.to_dict()
        analysis = _analysis(
            _evidence((48.0, 20.0), (110.0, 20.0)),
            _evidence((48.0, 32.0), (110.0, 32.0)),
        )

        result = reconstruct_plan_walls(
            analysis,
            _options(
                endpoint_snap_distance_pixels=0.0,
                maximum_vertex_distance_pixels=4.0,
            ),
            existing,
        )

        self.assertEqual(existing.to_dict(), original)
        self.assertFalse(
            any(
                vertex.x == 48.0 and vertex.y == 20.0
                for vertex in result.vertex_data.vertices
            )
        )
        coordinates_by_id = {
            vertex.id: (vertex.x, vertex.y) for vertex in result.vertex_data.vertices
        }
        self.assertTrue(
            any(
                {
                    coordinates_by_id[edge.start_vertex_id],
                    coordinates_by_id[edge.end_vertex_id],
                }
                == {(45.0, 20.0), (110.0, 20.0)}
                for edge in result.vertex_data.edges
            )
        )

    def test_maximum_vertex_distance_does_not_move_or_merge_existing_vertices(
        self,
    ) -> None:
        existing = VertexData(
            vertices=[
                Vertex(id=1, x=20.0, y=20.0),
                Vertex(id=2, x=21.0, y=20.0),
            ],
            edges=[Edge(start_vertex_id=1, end_vertex_id=2)],
            _next_vertex_id=3,
        )
        original = existing.to_dict()

        result = reconstruct_plan_walls(
            _analysis(),
            _options(maximum_vertex_distance_pixels=10.0),
            existing,
        )

        self.assertEqual(result.vertex_data.to_dict(), original)
        self.assertEqual(existing.to_dict(), original)

    def test_preserves_existing_graph_and_advances_allocator(self) -> None:
        existing = VertexData(
            vertices=[Vertex(id=7, x=10.0, y=20.0), Vertex(id=9, x=40.0, y=20.0)],
            edges=[Edge(start_vertex_id=7, end_vertex_id=9)],
            _next_vertex_id=2,
        )
        original = existing.to_dict()
        analysis = _analysis(
            _evidence((40.0, 20.0), (110.0, 20.0)),
            _evidence((40.0, 32.0), (110.0, 32.0)),
        )

        result = reconstruct_plan_walls(analysis, _options(), existing)

        self.assertEqual(existing.to_dict(), original)
        self.assertEqual(result.vertex_data.vertices[:2], existing.vertices)
        self.assertEqual(result.vertex_data.edges[0], existing.edges[0])
        self.assertTrue(
            all(vertex.id > 9 for vertex in result.vertex_data.vertices[2:])
        )

    def test_does_not_snap_to_unconnected_room_center_vertex(self) -> None:
        existing = VertexData(
            vertices=[Vertex(id=1, x=10.0, y=20.0)],
            _next_vertex_id=2,
        )
        analysis = _analysis(
            _evidence((11.0, 20.0), (100.0, 20.0)),
            _evidence((11.0, 32.0), (100.0, 32.0)),
        )

        result = reconstruct_plan_walls(analysis, _options(), existing)

        generated_at_start = [
            vertex
            for vertex in result.vertex_data.vertices
            if (vertex.x, vertex.y) == (11.0, 20.0)
        ]
        self.assertEqual(len(generated_at_start), 1)
        self.assertNotEqual(generated_at_start[0].id, 1)

    def test_is_deterministic_when_evidence_order_changes(self) -> None:
        lines = (
            _evidence((10.0, 20.0), (110.0, 20.0)),
            _evidence((10.0, 32.0), (110.0, 32.0)),
            _evidence((50.0, 5.0), (50.0, 90.0)),
            _evidence((62.0, 5.0), (62.0, 90.0)),
        )

        forward = reconstruct_plan_walls(_analysis(*lines), _options())
        reversed_result = reconstruct_plan_walls(
            _analysis(*reversed(lines)),
            _options(),
        )

        self.assertEqual(
            forward.vertex_data.to_dict(),
            reversed_result.vertex_data.to_dict(),
        )

    def test_rejects_malformed_existing_graph(self) -> None:
        malformed = VertexData(
            vertices=[Vertex(id=1, x=10.0, y=20.0)],
            edges=[Edge(start_vertex_id=1, end_vertex_id=2)],
        )

        with self.assertRaisesRegex(PlanWallDetectionError, "existing vertices"):
            reconstruct_plan_walls(_analysis(), _options(), malformed)

    def test_exact_existing_wall_overlap_adds_nothing(self) -> None:
        existing = VertexData()
        _add_parallel_existing_walls(existing)
        original = existing.to_dict()
        analysis = _analysis(
            _evidence((10.0, 20.0), (110.0, 20.0)),
            _evidence((10.0, 32.0), (110.0, 32.0)),
        )

        result = reconstruct_plan_walls(analysis, _options(), existing)

        self.assertEqual(result.added_edge_count, 0)
        self.assertEqual(result.vertex_data.to_dict(), original)
        self.assertEqual(existing.to_dict(), original)

    def test_near_parallel_existing_wall_overlap_adds_nothing(self) -> None:
        existing = VertexData()
        _add_parallel_existing_walls(existing)
        original = existing.to_dict()
        analysis = _analysis(
            _evidence((12.0, 21.5), (108.0, 21.5)),
            _evidence((12.0, 33.5), (108.0, 33.5)),
        )

        result = reconstruct_plan_walls(analysis, _options(), existing)

        self.assertEqual(result.added_edge_count, 0)
        self.assertEqual(result.vertex_data.to_dict(), original)
        self.assertEqual(existing.to_dict(), original)

    def test_full_vertex_distance_aligns_a_shifted_existing_wall_pair(
        self,
    ) -> None:
        existing = VertexData()
        _add_parallel_existing_walls(existing)
        original = existing.to_dict()
        analysis = _analysis(
            _evidence((30.0, 25.0), (90.0, 25.0)),
            _evidence((30.0, 37.0), (90.0, 37.0)),
        )

        result = reconstruct_plan_walls(
            analysis,
            _options(maximum_vertex_distance_pixels=6.0),
            existing,
        )

        self.assertEqual(result.added_edge_count, 0)
        self.assertEqual(result.vertex_data.to_dict(), original)
        self.assertEqual(existing.to_dict(), original)

    def test_default_distance_removes_a_close_duplicate_wall_pair(self) -> None:
        existing = VertexData()
        _add_parallel_existing_walls(existing)
        original = existing.to_dict()
        analysis = _analysis(
            _evidence((30.0, 24.0), (90.0, 24.0)),
            _evidence((30.0, 36.0), (90.0, 36.0)),
        )

        result = reconstruct_plan_walls(
            analysis,
            PlanWallDetectionOptions(),
            existing,
        )

        self.assertEqual(result.added_edge_count, 0)
        self.assertEqual(result.vertex_data.to_dict(), original)

    def test_shifted_detection_matches_a_noded_existing_wall_pair(self) -> None:
        existing = VertexData()
        _add_split_parallel_existing_walls(existing)
        original = existing.to_dict()
        analysis = _analysis(
            _evidence((20.0, 25.0), (100.0, 25.0)),
            _evidence((20.0, 37.0), (100.0, 37.0)),
        )

        result = reconstruct_plan_walls(
            analysis,
            _options(maximum_vertex_distance_pixels=6.0),
            existing,
        )

        self.assertEqual(result.added_edge_count, 0)
        self.assertEqual(result.vertex_data.to_dict(), original)

    def test_shifted_oblique_detection_matches_existing_wall_pair(self) -> None:
        existing = VertexData()
        first_start = existing.add_vertex(10.0, 10.0)
        first_end = existing.add_vertex(105.0, 105.0)
        second_start = existing.add_vertex(10.0, 22.0)
        second_end = existing.add_vertex(105.0, 117.0)
        existing.add_edge(first_start.id, first_end.id)
        existing.add_edge(second_start.id, second_end.id)
        original = existing.to_dict()
        analysis = _analysis(
            _evidence((30.0, 36.0), (80.0, 86.0)),
            _evidence((30.0, 48.0), (80.0, 98.0)),
        )

        result = reconstruct_plan_walls(
            analysis,
            _options(maximum_vertex_distance_pixels=5.0),
            existing,
        )

        self.assertEqual(result.added_edge_count, 0)
        self.assertEqual(result.vertex_data.to_dict(), original)

    def test_wide_pair_alignment_rejects_unrelated_existing_lines(self) -> None:
        existing = VertexData()
        _add_parallel_existing_walls(existing, second_y=45.0)
        original = existing.to_dict()
        analysis = _analysis(
            _evidence((30.0, 28.0), (90.0, 28.0)),
            _evidence((30.0, 36.0), (90.0, 36.0)),
        )

        result = reconstruct_plan_walls(
            analysis,
            _options(maximum_vertex_distance_pixels=10.0),
            existing,
        )

        self.assertEqual(result.added_edge_count, 4)
        self.assertEqual(existing.to_dict(), original)
        self.assertIn(
            ((30.0, 28.0), (90.0, 28.0)),
            _edge_coordinates(result.vertex_data),
        )
        self.assertIn(
            ((30.0, 36.0), (90.0, 36.0)),
            _edge_coordinates(result.vertex_data),
        )
        self.assertTrue(
            {
                ((30.0, 28.0), (30.0, 36.0)),
                ((90.0, 28.0), (90.0, 36.0)),
            }.issubset(_edge_coordinates(result.vertex_data))
        )

    def test_close_detection_reuses_one_existing_face_only(self) -> None:
        existing = VertexData()
        start = existing.add_vertex(10.0, 20.0)
        end = existing.add_vertex(110.0, 20.0)
        existing.add_edge(start.id, end.id)
        analysis = _analysis(
            _evidence((30.0, 24.0), (90.0, 24.0)),
            _evidence((30.0, 36.0), (90.0, 36.0)),
        )

        result = reconstruct_plan_walls(
            analysis,
            _options(maximum_vertex_distance_pixels=5.0),
            existing,
        )

        self.assertEqual(result.added_edge_count, 1)
        self.assertEqual(
            _edge_coordinates(result.vertex_data),
            {
                ((10.0, 20.0), (110.0, 20.0)),
                ((30.0, 36.0), (90.0, 36.0)),
            },
        )

    def test_one_existing_edge_cannot_consume_both_detected_faces(self) -> None:
        existing = VertexData()
        start = existing.add_vertex(10.0, 24.0)
        end = existing.add_vertex(110.0, 24.0)
        existing.add_edge(start.id, end.id)
        analysis = _analysis(
            _evidence((30.0, 20.0), (90.0, 20.0)),
            _evidence((30.0, 28.0), (90.0, 28.0)),
        )

        result = reconstruct_plan_walls(
            analysis,
            _options(maximum_vertex_distance_pixels=5.0),
            existing,
        )

        self.assertEqual(result.added_edge_count, 1)
        self.assertIn(
            ((30.0, 28.0), (90.0, 28.0)),
            _edge_coordinates(result.vertex_data),
        )

    def test_wide_pair_alignment_preserves_only_uncovered_extensions(self) -> None:
        existing = VertexData()
        _add_parallel_existing_walls(existing, end_x=60.0)
        analysis = _analysis(
            _evidence((30.0, 25.0), (110.0, 25.0)),
            _evidence((30.0, 37.0), (110.0, 37.0)),
        )

        result = reconstruct_plan_walls(
            analysis,
            _options(maximum_vertex_distance_pixels=6.0),
            existing,
        )

        self.assertEqual(result.added_edge_count, 3)
        self.assertIn(
            ((60.0, 20.0), (110.0, 20.0)),
            _edge_coordinates(result.vertex_data),
        )
        self.assertIn(
            ((60.0, 32.0), (110.0, 32.0)),
            _edge_coordinates(result.vertex_data),
        )
        self.assertIn(
            ((110.0, 20.0), (110.0, 32.0)),
            _edge_coordinates(result.vertex_data),
        )

    def test_out_of_bounds_mirrored_walls_remain_valid_anchors(self) -> None:
        existing = VertexData()
        _add_parallel_existing_walls(
            existing,
            start_x=-10.0,
            end_x=110.0,
        )
        original = existing.to_dict()
        analysis = _analysis(
            _evidence((0.0, 21.5), (100.0, 21.5)),
            _evidence((0.0, 33.5), (100.0, 33.5)),
        )

        result = reconstruct_plan_walls(analysis, _options(), existing)

        self.assertEqual(result.added_edge_count, 0)
        self.assertEqual(result.vertex_data.to_dict(), original)
        self.assertEqual(existing.to_dict(), original)

    def test_partial_existing_overlap_adds_only_uncovered_extensions(self) -> None:
        existing = VertexData()
        _add_parallel_existing_walls(existing, end_x=60.0)
        original = existing.to_dict()
        analysis = _analysis(
            _evidence((10.0, 20.0), (110.0, 20.0)),
            _evidence((10.0, 32.0), (110.0, 32.0)),
        )

        result = reconstruct_plan_walls(analysis, _options(), existing)

        self.assertEqual(result.added_edge_count, 3)
        self.assertEqual(existing.to_dict(), original)
        self.assertEqual(
            _edge_coordinates(result.vertex_data),
            {
                ((10.0, 20.0), (60.0, 20.0)),
                ((10.0, 32.0), (60.0, 32.0)),
                ((60.0, 20.0), (110.0, 20.0)),
                ((60.0, 32.0), (110.0, 32.0)),
                ((110.0, 20.0), (110.0, 32.0)),
            },
        )
        self.assertNotIn(
            ((10.0, 20.0), (110.0, 20.0)),
            _edge_coordinates(result.vertex_data),
        )

    def test_near_parallel_partial_overlap_adds_aligned_extensions(self) -> None:
        existing = VertexData()
        _add_parallel_existing_walls(existing, end_x=60.0)
        original = existing.to_dict()
        analysis = _analysis(
            _evidence((10.0, 21.5), (110.0, 21.5)),
            _evidence((10.0, 33.5), (110.0, 33.5)),
        )

        result = reconstruct_plan_walls(analysis, _options(), existing)

        self.assertEqual(existing.to_dict(), original)
        self.assertEqual(result.added_edge_count, 3)
        self.assertEqual(
            _edge_coordinates(result.vertex_data),
            {
                ((10.0, 20.0), (60.0, 20.0)),
                ((10.0, 32.0), (60.0, 32.0)),
                ((60.0, 20.0), (110.0, 20.0)),
                ((60.0, 32.0), (110.0, 32.0)),
                ((110.0, 20.0), (110.0, 32.0)),
            },
        )

    def test_near_parallel_continuation_snaps_to_existing_endpoints(self) -> None:
        existing = VertexData()
        _add_parallel_existing_walls(existing, end_x=60.0)
        analysis = _analysis(
            _evidence((63.0, 21.5), (110.0, 21.5)),
            _evidence((63.0, 33.5), (110.0, 33.5)),
        )

        result = reconstruct_plan_walls(analysis, _options(), existing)

        self.assertEqual(result.added_edge_count, 3)
        self.assertIn(
            ((60.0, 20.0), (110.0, 20.0)),
            _edge_coordinates(result.vertex_data),
        )
        self.assertIn(
            ((60.0, 32.0), (110.0, 32.0)),
            _edge_coordinates(result.vertex_data),
        )
        self.assertIn(
            ((110.0, 20.0), (110.0, 32.0)),
            _edge_coordinates(result.vertex_data),
        )

    def test_large_endpoint_snap_cannot_collapse_a_short_continuation(
        self,
    ) -> None:
        existing = VertexData()
        _add_parallel_existing_walls(existing, end_x=60.0)
        analysis = _analysis(
            _evidence((61.0, 21.5), (90.0, 21.5)),
            _evidence((61.0, 33.5), (90.0, 33.5)),
        )

        result = reconstruct_plan_walls(
            analysis,
            _options(endpoint_snap_distance_pixels=100.0),
            existing,
        )

        self.assertEqual(result.added_edge_count, 3)
        self.assertIn(
            ((60.0, 20.0), (90.0, 20.0)),
            _edge_coordinates(result.vertex_data),
        )
        self.assertIn(
            ((60.0, 32.0), (90.0, 32.0)),
            _edge_coordinates(result.vertex_data),
        )
        self.assertIn(
            ((90.0, 20.0), (90.0, 32.0)),
            _edge_coordinates(result.vertex_data),
        )

    def test_alignment_cannot_collapse_a_real_parallel_wall_face(self) -> None:
        existing = VertexData()
        existing_start = existing.add_vertex(10.0, 20.0)
        existing_end = existing.add_vertex(110.0, 20.0)
        existing.add_edge(existing_start.id, existing_end.id)
        analysis = _analysis(
            _evidence((10.0, 28.0), (110.0, 28.0)),
            _evidence((10.0, 36.0), (110.0, 36.0)),
        )

        result = reconstruct_plan_walls(
            analysis,
            _options(maximum_vertex_distance_pixels=20.0),
            existing,
        )

        self.assertEqual(result.added_edge_count, 4)
        self.assertEqual(
            _edge_coordinates(result.vertex_data),
            {
                ((10.0, 20.0), (110.0, 20.0)),
                ((10.0, 28.0), (10.0, 36.0)),
                ((10.0, 28.0), (110.0, 28.0)),
                ((10.0, 36.0), (110.0, 36.0)),
                ((110.0, 28.0), (110.0, 36.0)),
            },
        )

    def test_existing_wall_alignment_is_deterministic(self) -> None:
        existing = VertexData()
        _add_parallel_existing_walls(existing, end_x=60.0)
        lines = (
            _evidence((10.0, 21.5), (110.0, 21.5)),
            _evidence((10.0, 33.5), (110.0, 33.5)),
        )

        forward = reconstruct_plan_walls(_analysis(*lines), _options(), existing)
        reverse = reconstruct_plan_walls(
            _analysis(*reversed(lines)),
            _options(),
            existing,
        )

        self.assertEqual(
            forward.vertex_data.to_dict(),
            reverse.vertex_data.to_dict(),
        )

    def test_generated_walls_are_noded_at_immutable_existing_crossings(self) -> None:
        existing = VertexData()
        existing_start = existing.add_vertex(10.0, 50.0)
        existing_end = existing.add_vertex(110.0, 50.0)
        existing.add_edge(existing_start.id, existing_end.id)
        original_edge = existing.edges[0]
        analysis = _analysis(
            _evidence((40.0, 10.0), (40.0, 90.0)),
            _evidence((52.0, 10.0), (52.0, 90.0)),
        )

        result = reconstruct_plan_walls(analysis, _options(), existing)

        self.assertEqual(result.vertex_data.edges[0], original_edge)
        self.assertEqual(result.added_edge_count, 6)
        crossing_coordinates = {
            (vertex.x, vertex.y)
            for vertex in result.vertex_data.vertices
            if vertex.y == 50.0
        }
        self.assertEqual(
            crossing_coordinates,
            {(10.0, 50.0), (40.0, 50.0), (52.0, 50.0), (110.0, 50.0)},
        )
        self.assertTrue(
            {
                ((40.0, 10.0), (52.0, 10.0)),
                ((40.0, 90.0), (52.0, 90.0)),
            }.issubset(_edge_coordinates(result.vertex_data))
        )


# ### Wall-face end connector tests ###
class WallFaceEndConnectorTests(unittest.TestCase):
    def test_adds_perpendicular_connector_between_aligned_pair_ends(self) -> None:
        pair = _wall_face_pair(
            (10.0, 20.0),
            (100.0, 20.0),
            (0.0, 32.0),
            (100.0, 32.0),
        )

        adjusted_pairs, connectors = _close_wall_face_pair_ends(
            [pair],
            VertexData(),
            _options(),
        )

        self.assertEqual(len(adjusted_pairs), 1)
        self.assertEqual(
            {_segment_coordinate_key(connector) for connector in connectors},
            {((100.0, 20.0), (100.0, 32.0))},
        )

    def test_aligns_slightly_staggered_ends_before_connecting_them(self) -> None:
        pair = _wall_face_pair(
            (10.0, 20.0),
            (100.0, 20.0),
            (0.0, 32.0),
            (100.5, 32.0),
        )

        adjusted_pairs, connectors = _close_wall_face_pair_ends(
            [pair],
            VertexData(),
            _options(endpoint_snap_distance_pixels=4.0),
        )

        self.assertEqual(len(connectors), 1)
        connector = connectors[0]
        adjusted_pair = adjusted_pairs[0]
        self.assertAlmostEqual(float(connector.start[0]), float(connector.end[0]))
        self.assertAlmostEqual(
            float(np.dot(connector.vector, adjusted_pair.first.vector)),
            0.0,
        )
        adjusted_endpoints = {
            tuple(float(value) for value in endpoint)
            for segment in (adjusted_pair.first, adjusted_pair.second)
            for endpoint in (segment.start, segment.end)
        }
        self.assertIn(
            tuple(float(value) for value in connector.start),
            adjusted_endpoints,
        )
        self.assertIn(
            tuple(float(value) for value in connector.end),
            adjusted_endpoints,
        )

    def test_aligns_noisy_dangling_ends_before_testing_perpendicularity(
        self,
    ) -> None:
        pair = _wall_face_pair(
            (10.0, 20.0),
            (100.0, 20.0),
            (0.0, 32.0),
            (106.0, 32.0),
        )

        _adjusted_pairs, connectors = _close_wall_face_pair_ends(
            [pair],
            VertexData(),
            _options(
                endpoint_snap_distance_pixels=8.0,
                parallel_angle_tolerance_degrees=5.0,
            ),
        )

        self.assertEqual(len(connectors), 1)
        connector = connectors[0]
        self.assertAlmostEqual(float(connector.start[0]), float(connector.end[0]))

    def test_preliminary_topology_rejects_already_connected_pair_ends(
        self,
    ) -> None:
        pair = _wall_face_pair(
            (10.0, 20.0),
            (100.0, 20.0),
            (0.0, 32.0),
            (100.0, 32.0),
        )
        topology = VertexData()
        for y_value in (20.0, 32.0):
            left = topology.add_vertex(10.0, y_value)
            junction = topology.add_vertex(100.0, y_value)
            right = topology.add_vertex(110.0, y_value)
            branch = topology.add_vertex(100.0, y_value + 5.0)
            topology.add_edge(left.id, junction.id)
            topology.add_edge(junction.id, right.id)
            topology.add_edge(junction.id, branch.id)

        _adjusted_pairs, connectors = _close_wall_face_pair_ends(
            [pair],
            VertexData(),
            _options(),
            endpoint_topology=topology,
        )

        self.assertEqual(connectors, [])

    def test_does_not_connect_stagger_beyond_endpoint_snap_distance(self) -> None:
        pair = _wall_face_pair(
            (10.0, 20.0),
            (100.0, 20.0),
            (0.0, 32.0),
            (105.0, 32.0),
        )

        adjusted_pairs, connectors = _close_wall_face_pair_ends(
            [pair],
            VertexData(),
            _options(
                endpoint_snap_distance_pixels=4.0,
                parallel_angle_tolerance_degrees=30.0,
            ),
        )

        self.assertEqual(connectors, [])
        self.assertEqual(
            _wall_face_pair_coordinate_keys(adjusted_pairs),
            _wall_face_pair_coordinate_keys([pair]),
        )

    def test_does_not_connect_pair_outside_wall_separation_range(self) -> None:
        pair = _wall_face_pair(
            (10.0, 20.0),
            (100.0, 20.0),
            (0.0, 42.0),
            (100.0, 42.0),
        )

        _adjusted_pairs, connectors = _close_wall_face_pair_ends(
            [pair],
            VertexData(),
            _options(maximum_wall_separation_pixels=20.0),
        )

        self.assertEqual(connectors, [])

    def test_rejects_connector_that_crosses_another_selected_face(self) -> None:
        target_pair = _wall_face_pair(
            (10.0, 20.0),
            (100.0, 20.0),
            (0.0, 40.0),
            (100.0, 40.0),
        )
        crossing_pair = _wall_face_pair(
            (90.0, 30.0),
            (110.0, 30.0),
            (90.0, 42.0),
            (110.0, 42.0),
        )

        _adjusted_pairs, connectors = _close_wall_face_pair_ends(
            [target_pair, crossing_pair],
            VertexData(),
            _options(),
        )

        self.assertNotIn(
            ((100.0, 20.0), (100.0, 40.0)),
            {_segment_coordinate_key(connector) for connector in connectors},
        )

    def test_keeps_existing_graph_immutable(self) -> None:
        graph = VertexData()
        start = graph.add_vertex(20.0, 60.0)
        end = graph.add_vertex(80.0, 60.0)
        graph.add_edge(start.id, end.id)
        original = graph.to_dict()
        pair = _wall_face_pair(
            (10.0, 20.0),
            (100.0, 20.0),
            (0.0, 32.0),
            (100.0, 32.0),
        )

        _close_wall_face_pair_ends(
            [pair],
            graph,
            _options(),
        )

        self.assertEqual(graph.to_dict(), original)

    def test_geometry_is_deterministic_for_reversed_pair_and_endpoints(self) -> None:
        forward_pair = _wall_face_pair(
            (10.0, 20.0),
            (100.0, 20.0),
            (0.0, 32.0),
            (100.5, 32.0),
        )
        reversed_pair = _wall_face_pair(
            (100.5, 32.0),
            (0.0, 32.0),
            (100.0, 20.0),
            (10.0, 20.0),
        )
        options = _options(endpoint_snap_distance_pixels=4.0)

        forward_pairs, forward_connectors = _close_wall_face_pair_ends(
            [forward_pair],
            VertexData(),
            options,
        )
        reversed_pairs, reversed_connectors = _close_wall_face_pair_ends(
            [reversed_pair],
            VertexData(),
            options,
        )

        self.assertEqual(
            _wall_face_pair_coordinate_keys(forward_pairs),
            _wall_face_pair_coordinate_keys(reversed_pairs),
        )
        self.assertEqual(
            {_segment_coordinate_key(segment) for segment in forward_connectors},
            {_segment_coordinate_key(segment) for segment in reversed_connectors},
        )


# ### Generated vertex alignment tests ###
class GeneratedVertexAlignmentTests(unittest.TestCase):
    def test_nearby_redundant_endpoint_does_not_bend_exact_axis_lines(
        self,
    ) -> None:
        graph = VertexData()
        segments = (
            _segment((0.0, 10.0), (50.0, 10.0)),
            _segment((50.0, 10.0), (100.0, 10.0)),
            _segment((50.0, 0.0), (50.0, 10.0)),
            _segment((50.0, 10.0), (50.0, 100.0)),
            _segment((48.0, 10.0), (50.0, 10.0)),
        )

        _append_segments_to_graph(
            graph,
            segments,
            image_width=160,
            image_height=120,
            options=_options(maximum_vertex_distance_pixels=5.0),
        )

        self.assertEqual(
            _edge_coordinates(graph),
            {
                ((0.0, 10.0), (50.0, 10.0)),
                ((50.0, 0.0), (50.0, 10.0)),
                ((50.0, 10.0), (50.0, 100.0)),
                ((50.0, 10.0), (100.0, 10.0)),
            },
        )

    def test_generated_alignment_keeps_existing_vertices_immutable(self) -> None:
        graph = VertexData()
        anchor = graph.add_vertex(48.0, 10.0)
        anchor_end = graph.add_vertex(48.0, 100.0)
        graph.add_edge(anchor.id, anchor_end.id)
        original_vertices = tuple(
            (vertex.id, vertex.x, vertex.y) for vertex in graph.vertices
        )
        original_edge = graph.edges[0]
        segments = (
            _segment((0.0, 10.0), (50.0, 10.0)),
            _segment((50.0, 10.0), (100.0, 10.0)),
            _segment((48.0, 10.0), (50.0, 10.0)),
        )

        _append_segments_to_graph(
            graph,
            segments,
            image_width=160,
            image_height=120,
            options=_options(maximum_vertex_distance_pixels=5.0),
        )

        self.assertEqual(
            tuple(
                (vertex.id, vertex.x, vertex.y)
                for vertex in graph.vertices[: len(original_vertices)]
            ),
            original_vertices,
        )
        self.assertEqual(graph.edges[0], original_edge)

    def test_near_parallel_support_lines_cannot_move_a_vertex_far_away(
        self,
    ) -> None:
        graph = VertexData()
        nearby_points = ((50.0, 10.0), (48.0, 12.0))
        segments = (
            _segment((0.0, 10.0), nearby_points[0]),
            _segment(nearby_points[0], (100.0, 10.0)),
            _segment(nearby_points[1], (300.0, 17.0)),
        )

        _append_segments_to_graph(
            graph,
            segments,
            image_width=400,
            image_height=120,
            options=_options(maximum_vertex_distance_pixels=5.0),
        )

        incident_counts = {
            vertex.id: sum(
                vertex.id in (edge.start_vertex_id, edge.end_vertex_id)
                for edge in graph.edges
            )
            for vertex in graph.vertices
        }
        joined_vertex = next(
            vertex
            for vertex in graph.vertices
            if incident_counts[vertex.id] == 3
        )
        for point in nearby_points:
            self.assertLessEqual(
                float(np.linalg.norm(np.asarray(point) - (joined_vertex.x, joined_vertex.y))),
                5.0,
            )

    def test_generated_alignment_is_deterministic_for_reversed_segments(
        self,
    ) -> None:
        segments = (
            _segment((0.0, 10.0), (50.0, 10.0)),
            _segment((50.0, 10.0), (100.0, 10.0)),
            _segment((50.0, 0.0), (50.0, 10.0)),
            _segment((50.0, 10.0), (50.0, 100.0)),
            _segment((48.0, 10.0), (50.0, 10.0)),
        )
        forward = VertexData()
        reverse = VertexData()
        options = _options(maximum_vertex_distance_pixels=5.0)

        _append_segments_to_graph(forward, segments, 160, 120, options)
        _append_segments_to_graph(reverse, tuple(reversed(segments)), 160, 120, options)

        self.assertEqual(forward.to_dict(), reverse.to_dict())


# ### Option validation tests ###
class PlanWallDetectionOptionTests(unittest.TestCase):
    def test_rejects_inverted_separation_range(self) -> None:
        with self.assertRaises(PlanWallDetectionError):
            _options(
                minimum_wall_separation_pixels=30.0,
                maximum_wall_separation_pixels=20.0,
            )

    def test_rejects_non_finite_values(self) -> None:
        with self.assertRaises(PlanWallDetectionError):
            _options(endpoint_snap_distance_pixels=float("nan"))

    def test_rejects_negative_maximum_vertex_distance(self) -> None:
        with self.assertRaisesRegex(
            PlanWallDetectionError,
            "vertex distance",
        ):
            _options(maximum_vertex_distance_pixels=-1.0)


if __name__ == "__main__":
    unittest.main()
