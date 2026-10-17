# ### Imports ###
from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from housemaker.automatic_wall_detection import (
    PlanWallAnalysis,
    PlanWallDetectionOptions,
    _align_opposite_wall_endpoints,
    _canonical_endpoints,
    _Segment,
    _WallFacePair,
)
from housemaker.models import VertexData

# ### Numeric L3 fixture helpers ###
_Point = tuple[float, float]
_Edge = tuple[_Point, _Point]
_INNER_START = (953.12408, 141.83015)
_INNER_END = (1610.6248, 141.83015)


def _segment(start: _Point, end: _Point) -> _Segment:
    first, second = _canonical_endpoints(np.asarray(start), np.asarray(end))
    return _Segment(first, second, 1.0)


def _pair() -> _WallFacePair:
    return _WallFacePair(
        _segment((940.62494, 131.41075), (1620.625, 131.41075)),
        _segment(_INNER_START, _INNER_END),
    )


def _graph(*edges: _Edge) -> VertexData:
    graph = VertexData()
    ids: dict[_Point, int] = {}
    for start, end in edges:
        for point in (start, end):
            if point not in ids:
                ids[point] = graph.add_vertex(*point).id
        graph.add_edge(ids[start], ids[end])
    return graph


def _topology(
    *obstructions: _Edge,
    target_points_right: bool = True,
) -> VertexData:
    """Use actual L3 raw/rounded coordinates with a controlled target cap."""

    target_end_x = 1000.0 if target_points_right else 908.0
    return _graph(
        ((941.0, 131.0), (1150.0, 131.0)),
        ((953.0, 142.0), (1522.0, 142.0)),
        ((946.0, 287.0), (target_end_x, 287.0)),
        *obstructions,
    )


def _align(
    topology: VertexData,
    *,
    existing: VertexData | None = None,
    **option_overrides: float,
) -> _WallFacePair:
    options = replace(PlanWallDetectionOptions(), **option_overrides)
    return _align_opposite_wall_endpoints(
        PlanWallAnalysis(2100, 2112, ()),
        [_pair()],
        topology,
        existing if existing is not None else VertexData(),
        options,
    )[0]


# ### Fractional source incident-edge regression ###
class PlanWallL3AlignmentRegressionTests(unittest.TestCase):
    def test_raw_tip_can_extend_past_its_rounded_incident_edge(self) -> None:
        topology = _topology()
        before = topology.clone()

        aligned = _align(topology)

        np.testing.assert_array_equal(aligned.second.start, (946.0, 141.83015))
        np.testing.assert_array_equal(aligned.second.end, _INNER_END)
        self.assertEqual(topology.vertices, before.vertices)
        self.assertEqual(topology.edges, before.edges)

    def test_unrelated_collinear_edge_is_not_exempted_with_the_source_edge(
        self,
    ) -> None:
        topology = _topology(((949.0, 142.0), (951.0, 142.0)))

        aligned = _align(topology)

        np.testing.assert_array_equal(aligned.second.start, _INNER_START)

    def test_unrelated_crossing_edge_still_blocks_the_extension(self) -> None:
        topology = _topology(((950.0, 135.0), (950.0, 150.0)))

        aligned = _align(topology)

        np.testing.assert_array_equal(aligned.second.start, _INNER_START)

    # ### Existing eligibility and target-direction guards ###
    def test_same_direction_window_cap_does_not_pull_tip_past_its_guide(self) -> None:
        aligned = _align(_topology(target_points_right=False))

        np.testing.assert_array_equal(aligned.second.start, _INNER_START)

    def test_existing_source_vertices_and_edges_remain_immutable(self) -> None:
        topology = _topology()
        existing = topology.clone()
        before = existing.clone()

        aligned = _align(topology, existing=existing)

        np.testing.assert_array_equal(aligned.second.start, _INNER_START)
        self.assertEqual(existing.vertices, before.vertices)
        self.assertEqual(existing.edges, before.edges)

    def test_non_dangling_source_is_not_extended(self) -> None:
        topology = _topology(((953.0, 142.0), (953.0, 170.0)))

        aligned = _align(topology)

        np.testing.assert_array_equal(aligned.second.start, _INNER_START)

    def test_source_edge_exemption_does_not_relax_tolerance_or_search_limits(
        self,
    ) -> None:
        for overrides in (
            {"endpoint_alignment_tolerance_pixels": 7.0},
            {"endpoint_alignment_search_distance_pixels": 140.0},
        ):
            with self.subTest(overrides=overrides):
                aligned = _align(_topology(), **overrides)

                np.testing.assert_array_equal(aligned.second.start, _INNER_START)


# ### Standalone test entry point ###
if __name__ == "__main__":
    unittest.main()
