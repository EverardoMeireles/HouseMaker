# ### Imports ###
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations

from housemaker.models import Edge, Vertex

# ### Constants ###
DEFAULT_NON_RIGHT_ANGLE_TOLERANCE_DEGREES = 0.1
_ANGLE_COMPARISON_EPSILON_DEGREES = 1e-12


# ### Result models ###
@dataclass(frozen=True)
class WallAngleHighlights:
    """Non-right wall corners, their edges, and disconnected wall ends."""

    vertex_ids: frozenset[int] = frozenset()
    edge_keys: frozenset[tuple[int, int]] = frozenset()
    dead_end_vertex_ids: frozenset[int] = frozenset()


# ### Public helpers ###
def find_non_right_angle_wall_parts(
    vertices: Sequence[Vertex],
    edges: Sequence[Edge],
    *,
    excluded_vertex_ids: Iterable[int] = (),
    tolerance_degrees: float = DEFAULT_NON_RIGHT_ANGLE_TOLERANCE_DEGREES,
) -> WallAngleHighlights:
    """Find connected corners that are neither perpendicular nor straight.

    Angles are relative to neighboring edges, not the canvas axes, so a
    rotated rectangular room remains valid. Endpoints do not define a corner,
    but are returned separately when they have exactly one valid neighbor.
    """

    tolerance = _validate_angle_tolerance(tolerance_degrees)
    excluded_ids = frozenset(excluded_vertex_ids)
    vertices_by_id = {
        vertex.id: vertex
        for vertex in vertices
        if vertex.id not in excluded_ids
        and math.isfinite(vertex.x)
        and math.isfinite(vertex.y)
    }
    neighbors: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    seen_edge_keys: set[tuple[int, int]] = set()
    for edge in edges:
        start_id = edge.start_vertex_id
        end_id = edge.end_vertex_id
        edge_key = (min(start_id, end_id), max(start_id, end_id))
        if start_id == end_id or edge_key in seen_edge_keys:
            continue
        seen_edge_keys.add(edge_key)
        start = vertices_by_id.get(start_id)
        end = vertices_by_id.get(end_id)
        if start is None or end is None:
            continue
        delta_x = end.x - start.x
        delta_y = end.y - start.y
        length = math.hypot(delta_x, delta_y)
        if length == 0.0 or not math.isfinite(length):
            continue
        direction_x = delta_x / length
        direction_y = delta_y / length
        neighbors[start_id].append((end_id, direction_x, direction_y))
        neighbors[end_id].append((start_id, -direction_x, -direction_y))

    highlighted_ids: set[int] = set()
    highlighted_edges: set[tuple[int, int]] = set()
    for vertex_id, incident_neighbors in neighbors.items():
        for first, second in combinations(incident_neighbors, 2):
            first_id, first_x, first_y = first
            second_id, second_x, second_y = second
            angle = math.degrees(
                math.atan2(
                    abs(first_x * second_y - first_y * second_x),
                    first_x * second_x + first_y * second_y,
                )
            )
            if min(abs(angle - 90.0), abs(angle - 180.0)) <= (
                tolerance + _ANGLE_COMPARISON_EPSILON_DEGREES
            ):
                continue
            highlighted_ids.add(vertex_id)
            highlighted_edges.add(
                (min(vertex_id, first_id), max(vertex_id, first_id))
            )
            highlighted_edges.add(
                (min(vertex_id, second_id), max(vertex_id, second_id))
            )
    return WallAngleHighlights(
        vertex_ids=frozenset(highlighted_ids),
        edge_keys=frozenset(highlighted_edges),
        dead_end_vertex_ids=frozenset(
            vertex_id
            for vertex_id, incident_neighbors in neighbors.items()
            if len(incident_neighbors) == 1
        ),
    )


# ### Validation helpers ###
def _validate_angle_tolerance(value: float) -> float:
    try:
        tolerance = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Wall-angle tolerance must be a finite value from 0 to below 45 degrees."
        ) from exc
    if not math.isfinite(tolerance) or not 0.0 <= tolerance < 45.0:
        raise ValueError(
            "Wall-angle tolerance must be a finite value from 0 to below 45 degrees."
        )
    return tolerance
