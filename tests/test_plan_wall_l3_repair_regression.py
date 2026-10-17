# ### Imports ###
from __future__ import annotations

import unittest

import cv2
import numpy as np

from housemaker.automatic_wall_detection import (
    PlanWallAnalysis,
    PlanWallDetectionOptions,
    WallLineEvidence,
    _repair_rebased_graph_geometry,
    reconstruct_plan_walls,
)
from housemaker.models import VertexData


# ### Numeric L3 crop fixture ###
def _l3_crop_analysis() -> PlanWallAnalysis:
    """Retain L3's shortened tips and traced guides, translated by (-880, -100)."""

    ink = np.zeros((240, 760), dtype=np.uint8)
    strokes = (
        ((60, 31), (741, 31)),
        ((60, 42), (731, 42)),
        ((60, 31), (60, 187)),
        ((68, 42), (68, 217)),
        ((28, 187), (68, 187)),
        ((28, 217), (68, 217)),
        ((28, 187), (28, 217)),
    )
    for start, end in strokes:
        cv2.line(ink, start, end, 255, 1)
    readonly_ink = ink.astype(np.bool_)
    readonly_ink.setflags(write=False)

    def evidence(start: tuple[float, float], end: tuple[float, float]) -> WallLineEvidence:
        return WallLineEvidence(start, end, 0.95)

    return PlanWallAnalysis(
        image_width=760,
        image_height=240,
        line_evidence=(
            evidence((60.62494, 31.41075), (740.625, 31.41075)),
            evidence((73.12408, 41.83015), (730.6248, 41.83015)),
            evidence((28.1243, 187.28), (65.6297, 187.28)),
            evidence((28.1243, 217.359), (66.8793, 217.359)),
        ),
        structural_ink=readonly_ink,
        normalized_ink=readonly_ink,
        pixel_line_evidence=(
            evidence((60.20415, 31.0), (60.20415, 186.0)),
            evidence((68.02621, 31.0), (68.02621, 186.0)),
            evidence((60.20415, 41.30769), (73.0, 41.30769)),
            evidence((60.20415, 42.0), (731.0, 42.0)),
            evidence((60.20415, 186.0), (68.02621, 186.0)),
            evidence((68.02621, 186.0), (68.02621, 217.0)),
            evidence((28.0, 187.0), (68.0, 187.0)),
            evidence((28.0, 217.0), (68.0, 217.0)),
            evidence((28.0, 187.0), (28.0, 217.0)),
        ),
    )


def _options() -> PlanWallDetectionOptions:
    return PlanWallDetectionOptions(
        parallel_angle_tolerance_degrees=8.0,
        minimum_parallel_overlap_ratio=0.5,
    )


def _edge_coordinates(
    graph: VertexData,
) -> set[tuple[tuple[float, float], tuple[float, float]]]:
    points = {vertex.id: (vertex.x, vertex.y) for vertex in graph.vertices}
    return {
        tuple(sorted((points[edge.start_vertex_id], points[edge.end_vertex_id])))
        for edge in graph.edges
    }


# ### Public pipeline regressions ###
class PlanWallL3RepairRegressionTests(unittest.TestCase):
    def test_provisional_rounded_tip_does_not_bend_the_top_corner(self) -> None:
        result = reconstruct_plan_walls(_l3_crop_analysis(), _options())
        edges = _edge_coordinates(result.vertex_data)

        self.assertIn(((60.0, 31.0), (60.0, 42.0)), edges)
        for start, end in edges:
            self.assertTrue(start[0] == end[0] or start[1] == end[1], (start, end))

    def test_straight_wall_reaches_window_head_after_joint_anchor_resolution(self) -> None:
        result = reconstruct_plan_walls(_l3_crop_analysis(), _options())
        edges = _edge_coordinates(result.vertex_data)

        # Previously the short upper cap consumed both dangling sources, and
        # neither of the continuous pixel-supported vertical faces was added.
        self.assertIn(((60.0, 42.0), (60.0, 187.0)), edges)
        self.assertIn(((28.0, 187.0), (60.0, 187.0)), edges)

    def test_true_authored_anchors_remain_immutable_in_joint_network(self) -> None:
        existing = VertexData()
        first = existing.add_vertex(61.0, 31.0)
        second = existing.add_vertex(300.0, 31.0)
        existing.add_edge(first.id, second.id)
        original = existing.to_dict()

        result = reconstruct_plan_walls(_l3_crop_analysis(), _options(), existing)

        self.assertEqual(existing.to_dict(), original)
        self.assertEqual(result.vertex_data.vertices[:2], existing.vertices)
        self.assertEqual(result.vertex_data.edges[:1], existing.edges)


# ### Temporary network identity regression ###
class PlanWallRepairAnchorIdentityTests(unittest.TestCase):
    def test_coincident_authored_vertices_keep_their_separate_identifiers(self) -> None:
        graph = VertexData()
        first = graph.add_vertex(10.0, 20.0)
        duplicate = graph.add_vertex(10.0, 20.0)
        right = graph.add_vertex(30.0, 20.0)
        lower = graph.add_vertex(10.0, 40.0)
        graph.add_edge(first.id, right.id)
        graph.add_edge(duplicate.id, lower.id)

        segments, incidents = _repair_rebased_graph_geometry(graph, graph.clone())

        self.assertEqual(len(segments), 2)
        self.assertEqual(set(incidents), {first.id, duplicate.id, right.id, lower.id})
        self.assertEqual(len(incidents[first.id]), 1)
        self.assertEqual(len(incidents[duplicate.id]), 1)


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
