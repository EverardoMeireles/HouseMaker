# ### Imports ###
from __future__ import annotations

import math
import unittest

import numpy as np

from housemaker.automatic_wall_detection import (
    PlanWallAnalysis,
    PlanWallDetectionOptions,
    _align_opposite_wall_endpoints,
    _alignment_extension_crosses_geometry,
    _canonical_endpoints,
    _Segment,
    _WallFacePair,
)
from housemaker.models import VertexData


# ### Fixture helpers ###
def _segment(start: tuple[float, float], end: tuple[float, float]) -> _Segment:
    first, second = _canonical_endpoints(np.asarray(start), np.asarray(end))
    return _Segment(first, second, 1.0)


def _pair(left: float, top: float, *, right: float | None = None) -> _WallFacePair:
    return _WallFacePair(
        _segment((left, top), (left, 160.0)),
        _segment(
            (left + 12.0 if right is None else right, top),
            (left + 12.0 if right is None else right, 160.0),
        ),
    )


def _graph(pairs: list[_WallFacePair]) -> VertexData:
    graph = VertexData()
    for pair in pairs:
        for face in (pair.first, pair.second):
            start = graph.add_vertex(*face.start)
            end = graph.add_vertex(*face.end)
            graph.add_edge(start.id, end.id)
    return graph


def _align(pairs: list[_WallFacePair]) -> list[_WallFacePair]:
    return _align_opposite_wall_endpoints(
        PlanWallAnalysis(600, 600, ()),
        pairs,
        _graph(pairs),
        VertexData(),
        PlanWallDetectionOptions(
            endpoint_alignment_tolerance_pixels=16.0,
            endpoint_alignment_search_distance_pixels=400.0,
        ),
    )


def _coordinates(pairs: list[_WallFacePair]) -> set[tuple[float, ...]]:
    return {
        tuple(np.round(np.concatenate((face.start, face.end)), 4))
        for pair in pairs
        for face in (pair.first, pair.second)
    }


# ### Endpoint alignment safety regressions ###
class PlanWallEndpointAlignmentTests(unittest.TestCase):
    def test_rotated_parallel_walls_align_without_tilting(self) -> None:
        angle = math.radians(30.0)
        rotation = np.asarray(
            (
                (math.cos(angle), -math.sin(angle)),
                (math.sin(angle), math.cos(angle)),
            )
        )
        translation = np.asarray((150.0, 50.0))
        original = [_pair(40.0, 40.0), _pair(240.0, 48.0)]
        rotated = [
            _WallFacePair(
                *(
                    _segment(
                        tuple(rotation @ face.start + translation),
                        tuple(rotation @ face.end + translation),
                    )
                    for face in (pair.first, pair.second)
                )
            )
            for pair in original
        ]

        adjusted = _align(rotated)

        expected = [_pair(40.0, 40.0), _pair(240.0, 40.0)]
        expected_rotated = [
            _WallFacePair(
                *(
                    _segment(
                        tuple(rotation @ face.start + translation),
                        tuple(rotation @ face.end + translation),
                    )
                    for face in (pair.first, pair.second)
                )
            )
            for pair in expected
        ]
        self.assertEqual(_coordinates(adjusted), _coordinates(expected_rotated))

    def test_equally_relevant_conflicting_targets_are_not_guessed(self) -> None:
        source = _pair(200.0, 48.0)
        targets = [_pair(28.0, 40.0), _pair(360.0, 36.0)]

        adjusted = _align([source, *targets])

        self.assertTrue(np.array_equal(adjusted[0].first.start, source.first.start))

    def test_shuffled_pair_and_face_order_preserve_results(self) -> None:
        pairs = [_pair(200.0, 48.0), _pair(28.0, 40.0), _pair(360.0, 36.0)]
        shuffled = [_WallFacePair(pair.second, pair.first) for pair in reversed(pairs)]

        self.assertEqual(_coordinates(_align(pairs)), _coordinates(_align(shuffled)))

    def test_single_wall_pair_does_not_align_to_its_own_opposite_face(self) -> None:
        pair = _WallFacePair(
            _segment((40.0, 40.0), (40.0, 160.0)),
            _segment((52.0, 48.0), (52.0, 160.0)),
        )

        self.assertEqual(_coordinates(_align([pair])), _coordinates([pair]))

    def test_collinear_obstruction_cannot_be_passed_by_extension(self) -> None:
        extension = _segment((240.0, 40.0), (240.0, 48.0))
        obstruction = _segment((240.0, 40.0), (240.0, 44.0))

        self.assertTrue(_alignment_extension_crosses_geometry(extension, [obstruction]))

    def test_touching_incident_wall_is_not_an_obstruction(self) -> None:
        extension = _segment((240.0, 40.0), (240.0, 48.0))
        incident_wall = _segment((240.0, 48.0), (240.0, 160.0))

        self.assertFalse(
            _alignment_extension_crosses_geometry(extension, [incident_wall])
        )


# ### Standalone test entry point ###
if __name__ == "__main__":
    unittest.main()
