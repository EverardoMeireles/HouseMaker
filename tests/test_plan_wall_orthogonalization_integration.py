# ### Imports ###
from __future__ import annotations

import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np

from housemaker import automatic_wall_detection as detection
from housemaker.models import VertexData
from housemaker.plan_wall_orthogonalization import orthogonalize_wall_graph


# ### Fixture helpers ###
def _ink_at_y(y: int) -> np.ndarray:
    ink = np.zeros((160, 160), dtype=bool)
    ink[y, 10:131] = True
    ink.setflags(write=False)
    return ink


def _reconstruct_after_late_endpoint_shift(
    analysis: detection.PlanWallAnalysis,
    existing: VertexData | None = None,
) -> tuple[detection.PlanWallDetectionResult, dict[str, object], tuple[float, ...]]:
    """Inject a small endpoint shift at graph assembly, after line straightening.

    Empty evidence isolates the final-stage wiring; graph assembly and the
    orthogonal correction themselves still execute their real implementations.
    """

    real_append = detection._append_segments_to_graph
    append_count = 0
    correction_kwargs: dict[str, object] = {}
    y_before_correction: tuple[float, ...] = ()

    def append_crooked_segment(graph: VertexData, *args: object) -> None:
        nonlocal append_count
        real_append(graph, *args)
        append_count += 1
        start = graph.vertices[0] if existing is not None else graph.add_vertex(20, 48)
        end = graph.add_vertex(120, 52)
        graph.add_edge(start.id, end.id)

    def correct(graph: VertexData, **kwargs: object) -> bool:
        nonlocal y_before_correction
        # The preliminary topology and final preview graph must both have
        # finished assembling before the final correction executes once.
        if append_count != 2:
            raise AssertionError(
                "Orthogonal correction ran before final graph assembly."
            )
        correction_kwargs.update(kwargs)
        y_before_correction = tuple(vertex.y for vertex in graph.vertices)
        return orthogonalize_wall_graph(graph, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                detection,
                "_append_segments_to_graph",
                side_effect=append_crooked_segment,
            )
        )
        stack.enter_context(
            patch.object(
                detection, "_repair_image_supported_wall_lines", return_value=[]
            )
        )
        stack.enter_context(
            patch.object(
                detection, "_repair_pixel_supported_wall_lines", return_value=[]
            )
        )
        correction = stack.enter_context(
            patch.object(detection, "orthogonalize_wall_graph", side_effect=correct)
        )
        result = detection.reconstruct_plan_walls(
            analysis, existing_vertex_data=existing
        )
    correction.assert_called_once()
    return result, correction_kwargs, y_before_correction


# ### Final preview integration regressions ###
class PlanWallOrthogonalizationIntegrationTests(unittest.TestCase):
    def test_final_graph_is_corrected_before_green_preview_is_returned(self) -> None:
        result, kwargs, before = _reconstruct_after_late_endpoint_shift(
            detection.PlanWallAnalysis(160, 160, ())
        )

        self.assertEqual(before, (48.0, 52.0))
        self.assertEqual(len({vertex.y for vertex in result.vertex_data.vertices}), 1)
        self.assertEqual(result.added_edge_count, 1)
        self.assertEqual(kwargs["fixed_vertex_ids"], frozenset())
        self.assertEqual(kwargs["image_size"], (160, 160))
        self.assertIsNone(kwargs["ink"])

    def test_existing_vertices_are_passed_as_fixed_and_remain_unchanged(self) -> None:
        existing = VertexData()
        start = existing.add_vertex(20, 48)
        end = existing.add_vertex(20, 120)
        existing.add_edge(start.id, end.id)
        before = existing.clone()

        result, kwargs, _ = _reconstruct_after_late_endpoint_shift(
            detection.PlanWallAnalysis(160, 160, ()), existing
        )

        self.assertEqual(kwargs["fixed_vertex_ids"], frozenset((start.id, end.id)))
        self.assertEqual(result.vertex_data.vertices[:2], before.vertices)
        self.assertEqual(result.vertex_data.edges[:1], before.edges)
        self.assertEqual(result.vertex_data.vertices[-1].y, start.y)
        self.assertEqual(result.added_edge_count, 1)
        self.assertEqual(existing.vertices, before.vertices)
        self.assertEqual(existing.edges, before.edges)
        self.assertEqual(existing.revision, before.revision)

    def test_normalized_black_lines_take_precedence_over_structural_ink(self) -> None:
        normalized = _ink_at_y(49)
        structural = _ink_at_y(51)
        result, kwargs, _ = _reconstruct_after_late_endpoint_shift(
            detection.PlanWallAnalysis(
                160, 160, (), structural_ink=structural, normalized_ink=normalized
            )
        )

        self.assertIs(kwargs["ink"], normalized)
        self.assertEqual(
            [vertex.y for vertex in result.vertex_data.vertices], [49.0, 49.0]
        )

    def test_structural_ink_is_used_when_normalized_ink_is_unavailable(self) -> None:
        structural = _ink_at_y(50)
        result, kwargs, _ = _reconstruct_after_late_endpoint_shift(
            detection.PlanWallAnalysis(160, 160, (), structural_ink=structural)
        )

        self.assertIs(kwargs["ink"], structural)
        self.assertEqual(
            [vertex.y for vertex in result.vertex_data.vertices], [50.0, 50.0]
        )


# ### Standalone test entry point ###
if __name__ == "__main__":
    unittest.main()
