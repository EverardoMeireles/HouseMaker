"""Derive wall spans owned by four-corner doorway connections."""

# ### Imports ###
from __future__ import annotations

import math
from collections.abc import Iterable

from housemaker.models import DoorwayData, Edge, LevelData, VertexData

# ### Constants ###
DOORWAY_CONNECTION_EPSILON_PIXELS = 0.5


# ### Structural connection helpers ###
def get_doorway_structural_edges(
    vertex_data: VertexData,
    doorways: Iterable[DoorwayData],
    *,
    ignored_vertex_ids: Iterable[int] = (),
) -> tuple[Edge, ...]:
    """Return authored edges plus valid doorway-owned wall-face spans.

    The two across-opening spans are not editable/saved wall edges. Geometry
    needs them temporarily to build the lintel/arch above the opening, and
    floors need them to keep the structural perimeter closed.
    """

    linked_doorways = tuple(
        doorway
        for doorway in doorways
        if getattr(doorway, "corner_vertex_ids", None) is not None
    )
    if not linked_doorways:
        return tuple(vertex_data.edges)
    edges = list(vertex_data.edges)
    edge_keys = {
        tuple(sorted((edge.start_vertex_id, edge.end_vertex_id))) for edge in edges
    }
    authored_edge_keys = set(edge_keys)
    vertices = {vertex.id: vertex for vertex in vertex_data.vertices}
    ignored_ids = set(ignored_vertex_ids)
    for doorway in linked_doorways:
        corner_ids = getattr(doorway, "corner_vertex_ids", None)
        if (
            not isinstance(corner_ids, tuple | list)
            or len(corner_ids) != 4
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in corner_ids
            )
            or len(set(corner_ids)) != 4
            or any(
                value not in vertices or value in ignored_ids for value in corner_ids
            )
        ):
            continue
        # Each jamb joins the two parallel wall faces at one side of the gap.
        if any(
            tuple(sorted((corner_ids[first], corner_ids[second])))
            not in authored_edge_keys
            for first, second in ((0, 2), (1, 3))
        ):
            continue
        try:
            angle = math.radians(float(doorway.rotation_degrees))
            width_axis = (-math.sin(angle), math.cos(angle))
            depth_axis = (math.cos(angle), math.sin(angle))
            local_points = tuple(
                (
                    vertices[vertex_id].x * width_axis[0]
                    + vertices[vertex_id].y * width_axis[1],
                    vertices[vertex_id].x * depth_axis[0]
                    + vertices[vertex_id].y * depth_axis[1],
                )
                for vertex_id in corner_ids
            )
        except (TypeError, ValueError, OverflowError):
            continue
        if not _has_rectangular_corners(local_points):
            continue
        for first, second in ((0, 1), (2, 3)):
            edge_key = tuple(sorted((corner_ids[first], corner_ids[second])))
            if edge_key not in edge_keys:
                edges.append(Edge(corner_ids[first], corner_ids[second]))
                edge_keys.add(edge_key)
    return tuple(edges)


def get_level_structural_edges(level: LevelData) -> tuple[Edge, ...]:
    """Include doorway connections without mutating the level's wall graph."""

    return get_doorway_structural_edges(
        level.vertex_data,
        level.doorways,
        ignored_vertex_ids=(room.center_vertex_id for room in level.rooms),
    )


def get_level_structural_vertex_data(level: LevelData) -> VertexData:
    """Provide a read-only geometry graph for room boundary/UV builders."""

    edges = get_level_structural_edges(level)
    if len(edges) == len(level.vertex_data.edges):
        return level.vertex_data
    return VertexData(
        vertices=list(level.vertex_data.vertices),
        edges=list(edges),
        _next_vertex_id=level.vertex_data._next_vertex_id,
    )


# ### Corner validation helpers ###
def _has_rectangular_corners(
    points: tuple[tuple[float, float], ...],
) -> bool:
    if not all(math.isfinite(value) for point in points for value in point):
        return False
    tolerance = DOORWAY_CONNECTION_EPSILON_PIXELS
    return (
        abs(points[0][0] - points[2][0]) <= tolerance
        and abs(points[1][0] - points[3][0]) <= tolerance
        and abs(points[0][1] - points[1][1]) <= tolerance
        and abs(points[2][1] - points[3][1]) <= tolerance
        and abs(points[0][0] - points[1][0]) > tolerance
        and abs(points[0][1] - points[2][1]) > tolerance
    )
