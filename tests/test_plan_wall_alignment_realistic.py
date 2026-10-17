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

# ### Realistic reconstruction fixtures ###
_Point = tuple[float, float]
_Edge = tuple[_Point, _Point]
_LEFT_FACES: tuple[_Edge, ...] = (
    ((40.0, 40.0), (40.0, 160.0)),
    ((52.0, 40.0), (52.0, 160.0)),
)
_RIGHT_FACES: tuple[_Edge, ...] = (
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
        "endpoint_alignment_tolerance_pixels": 8.0,
        "endpoint_alignment_search_distance_pixels": 256.0,
    }
    values.update(overrides)
    return PlanWallDetectionOptions(**values)


def _analysis(*faces: _Edge, dashed_gap: bool = False) -> PlanWallAnalysis:
    """Keep the photograph's actual short wall ends, not idealized full ink."""

    raster = np.zeros((220, 300), dtype=np.uint8)
    for start, end in faces:
        cv2.line(
            raster,
            tuple(round(value) for value in start),
            tuple(round(value) for value in end),
            1,
            1,
        )
    if dashed_gap:
        for x in range(65, 230, 20):
            cv2.line(raster, (x, 40), (min(x + 8, 230), 40), 1, 1)
    ink = raster.astype(np.bool_)
    ink.setflags(write=False)
    return PlanWallAnalysis(
        image_width=raster.shape[1],
        image_height=raster.shape[0],
        line_evidence=tuple(WallLineEvidence(start, end, 0.95) for start, end in faces),
        structural_ink=ink,
        normalized_ink=ink,
    )


def _graph(*edges: _Edge) -> VertexData:
    graph = VertexData()
    vertex_ids: dict[_Point, int] = {}
    for start, end in edges:
        for point in (start, end):
            if point not in vertex_ids:
                vertex_ids[point] = graph.add_vertex(*point).id
        graph.add_edge(vertex_ids[start], vertex_ids[end])
    return graph


def _edges(graph: VertexData) -> set[_Edge]:
    positions = {
        vertex.id: (round(vertex.x, 6), round(vertex.y, 6)) for vertex in graph.vertices
    }
    return {
        tuple(sorted((positions[edge.start_vertex_id], positions[edge.end_vertex_id])))
        for edge in graph.edges
    }


def _top_y(graph: VertexData, x: float) -> float:
    return min(vertex.y for vertex in graph.vertices if abs(vertex.x - x) < 1e-6)


# ### Missing and noisy endpoint ink regressions ###
class PlanWallRealisticEndpointAlignmentTests(unittest.TestCase):
    def test_short_extension_aligns_even_when_the_photograph_has_no_ink_there(
        self,
    ) -> None:
        analysis = _analysis(*_LEFT_FACES, *_RIGHT_FACES)
        self.assertFalse(analysis.normalized_ink[40:48, 240].any())
        self.assertFalse(analysis.normalized_ink[40:48, 252].any())

        result = reconstruct_plan_walls(analysis, _options()).vertex_data

        self.assertEqual({vertex.x for vertex in result.vertices}, {40, 52, 240, 252})
        for x in (40.0, 52.0, 240.0, 252.0):
            self.assertEqual(_top_y(result, x), 40.0)
        self.assertTrue(
            all(
                start[0] == end[0] or start[1] == end[1]
                for start, end in _edges(result)
            )
        )

    def test_a_few_noise_pixels_do_not_veto_the_same_short_adjustment(self) -> None:
        analysis = _analysis(*_LEFT_FACES, *_RIGHT_FACES)
        sparse_ink = analysis.normalized_ink.copy()
        sparse_ink[42, 240] = True
        sparse_ink[45, 252] = True
        sparse_ink.setflags(write=False)
        analysis = replace(
            analysis, normalized_ink=sparse_ink, structural_ink=sparse_ink
        )

        result = reconstruct_plan_walls(analysis, _options()).vertex_data

        self.assertEqual(_top_y(result, 240.0), 40.0)
        self.assertEqual(_top_y(result, 252.0), 40.0)

    def test_small_adjustment_and_long_search_have_independent_limits(self) -> None:
        analysis = _analysis(*_LEFT_FACES, *_RIGHT_FACES)
        for controls, expected_y in (
            ({"endpoint_alignment_tolerance_pixels": 7.0}, 48.0),
            ({"endpoint_alignment_search_distance_pixels": 150.0}, 48.0),
            ({"endpoint_alignment_tolerance_pixels": 0.0}, 48.0),
            ({"endpoint_alignment_search_distance_pixels": 0.0}, 48.0),
            ({}, 40.0),
        ):
            with self.subTest(controls=controls):
                result = reconstruct_plan_walls(
                    analysis, _options(**controls)
                ).vertex_data

                self.assertEqual(_top_y(result, 240.0), expected_y)
                self.assertEqual(_top_y(result, 252.0), expected_y)

    def test_white_or_dashed_room_gap_is_not_filled_by_alignment(self) -> None:
        for dashed_gap in (False, True):
            with self.subTest(dashed_gap=dashed_gap):
                result = reconstruct_plan_walls(
                    _analysis(*_LEFT_FACES, *_RIGHT_FACES, dashed_gap=dashed_gap),
                    _options(),
                ).vertex_data

                self.assertEqual(_top_y(result, 240.0), 40.0)
                self.assertFalse(
                    any(
                        start[0] < 100 and end[0] > 200 for start, end in _edges(result)
                    )
                )


# ### Immutable anchors and safe adjustment limits ###
class PlanWallRealisticEndpointSafetyTests(unittest.TestCase):
    def test_new_vertices_align_to_an_unchanged_authored_opposite_wall(self) -> None:
        existing = _graph(*_LEFT_FACES)
        before = existing.clone()

        result = reconstruct_plan_walls(
            _analysis(*_LEFT_FACES, *_RIGHT_FACES), _options(), existing
        ).vertex_data

        self.assertEqual(result.vertices[: len(before.vertices)], before.vertices)
        self.assertEqual(result.edges[: len(before.edges)], before.edges)
        self.assertEqual(existing.vertices, before.vertices)
        self.assertEqual(existing.edges, before.edges)
        self.assertEqual(_top_y(result, 240.0), 40.0)

    def test_confirmed_short_wall_does_not_move_when_the_same_scan_is_repeated(
        self,
    ) -> None:
        existing = _graph(*_LEFT_FACES, *_RIGHT_FACES)
        before = existing.clone()

        result = reconstruct_plan_walls(
            _analysis(*_LEFT_FACES, *_RIGHT_FACES), _options(), existing
        ).vertex_data

        self.assertEqual(result.vertices, before.vertices)
        self.assertEqual(result.edges, before.edges)
        self.assertEqual(_top_y(result, 240.0), 48.0)
        self.assertEqual(existing.vertices, before.vertices)

    def test_extension_does_not_cross_an_existing_wall(self) -> None:
        barrier: _Edge = ((235.0, 44.0), (260.0, 44.0))
        existing = _graph(barrier)

        result = reconstruct_plan_walls(
            _analysis(*_LEFT_FACES, *_RIGHT_FACES), _options(), existing
        ).vertex_data

        self.assertEqual(_top_y(result, 240.0), 48.0)
        self.assertEqual(_top_y(result, 252.0), 48.0)
        self.assertEqual(result.vertices[:2], existing.vertices)
        self.assertEqual(result.edges[0], existing.edges[0])

    def test_out_of_image_authored_anchor_cannot_pull_new_vertices_outside_image(
        self,
    ) -> None:
        outside_anchor = _graph(
            ((40.0, -4.0), (40.0, 160.0)),
            ((52.0, -4.0), (52.0, 160.0)),
        )
        generated_faces: tuple[_Edge, ...] = (
            ((240.0, 4.0), (240.0, 160.0)),
            ((252.0, 4.0), (252.0, 160.0)),
        )

        result = reconstruct_plan_walls(
            _analysis(*generated_faces), _options(), outside_anchor
        ).vertex_data

        self.assertEqual(result.vertices[:4], outside_anchor.vertices)
        self.assertEqual(_top_y(result, 240.0), 4.0)
        self.assertEqual(_top_y(result, 252.0), 4.0)
        self.assertTrue(all(vertex.y >= 0 for vertex in result.vertices[4:]))

    def test_true_oblique_walls_are_not_squared_off_by_alignment(self) -> None:
        oblique_faces: tuple[_Edge, ...] = (
            ((40.0, 70.0), (140.0, 170.0)),
            ((52.0, 58.0), (152.0, 158.0)),
        )
        analysis = _analysis(*oblique_faces)

        enabled = reconstruct_plan_walls(analysis, _options()).vertex_data
        disabled = reconstruct_plan_walls(
            analysis, _options(endpoint_alignment_tolerance_pixels=0.0)
        ).vertex_data

        self.assertEqual(_edges(enabled), _edges(disabled))
        self.assertTrue(
            {tuple(sorted(edge)) for edge in oblique_faces} <= _edges(enabled)
        )

    def test_slider_reconstruction_does_not_accumulate_endpoint_drift(self) -> None:
        analysis = _analysis(*_LEFT_FACES, *_RIGHT_FACES)

        first = reconstruct_plan_walls(analysis, _options()).vertex_data
        reconstruct_plan_walls(
            analysis, _options(endpoint_alignment_tolerance_pixels=0.0)
        )
        repeated = reconstruct_plan_walls(analysis, _options()).vertex_data

        self.assertEqual(_edges(first), _edges(repeated))
        self.assertEqual(_top_y(repeated, 240.0), 40.0)


# ### Standalone test entry point ###
if __name__ == "__main__":
    unittest.main()
