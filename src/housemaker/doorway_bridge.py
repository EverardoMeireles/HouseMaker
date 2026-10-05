"""Find safe vertex pairs for Shift-assisted doorway placement."""

# ### Imports ###
from __future__ import annotations

import math
from dataclasses import dataclass

from housemaker.models import (
    MAX_DOORWAY_WIDTH_METERS,
    MIN_DOORWAY_WIDTH_METERS,
    PIXEL_TO_METER,
    Vertex,
    VertexData,
)

# ### Constants ###
STRAIGHT_ALIGNMENT_TOLERANCE_PIXELS = 0.5


# ### Result models ###
@dataclass(frozen=True)
class DoorwayBridgeTarget:
    """Two unconnected wall points that one doorway can bridge."""

    first_vertex_id: int
    second_vertex_id: int
    center: tuple[float, float]
    width_meters: float
    rotation_degrees: float
    cursor_distance: float


# ### Public helpers ###
def find_doorway_bridge_target(
    vertex_data: VertexData,
    room_center_vertex_ids: set[int],
    raw_center: tuple[float, float],
    preset_width_meters: float,
    center_tolerance: float,
) -> DoorwayBridgeTarget | None:
    """Find aligned endpoints without an existing edge near the cursor."""

    targets = _find_doorway_bridge_targets(
        vertex_data,
        room_center_vertex_ids,
        raw_center,
        preset_width_meters,
        center_tolerance,
    )
    return targets[0] if targets else None


def find_doorway_bridges_covered_by_rectangle(
    vertex_data: VertexData,
    room_center_vertex_ids: set[int],
    center: tuple[float, float],
    width_direction: tuple[float, float],
    width_pixels: float,
    depth_pixels: float,
    position_tolerance_pixels: float = STRAIGHT_ALIGNMENT_TOLERANCE_PIXELS,
) -> tuple[DoorwayBridgeTarget, ...]:
    """Find unconnected straight vertex pairs covered by one doorway."""

    direction_length = math.hypot(*width_direction)
    if direction_length <= 1e-6 or width_pixels <= 0.0 or depth_pixels <= 0.0:
        return ()
    width_axis = (
        width_direction[0] / direction_length,
        width_direction[1] / direction_length,
    )
    depth_axis = (-width_axis[1], width_axis[0])
    half_width = width_pixels * 0.5
    half_depth = depth_pixels * 0.5
    tolerance = max(float(position_tolerance_pixels), 0.0)
    candidates = _find_doorway_bridge_targets(
        vertex_data,
        room_center_vertex_ids,
        center,
        width_pixels * PIXEL_TO_METER,
        math.hypot(half_width, half_depth) + tolerance,
    )
    vertices_by_id = {vertex.id: vertex for vertex in vertex_data.vertices}
    covered_targets: list[DoorwayBridgeTarget] = []
    for candidate in candidates:
        first = vertices_by_id.get(candidate.first_vertex_id)
        second = vertices_by_id.get(candidate.second_vertex_id)
        if first is None or second is None:
            continue
        local_positions = tuple(
            (
                (vertex.x - center[0]) * width_axis[0]
                + (vertex.y - center[1]) * width_axis[1],
                (vertex.x - center[0]) * depth_axis[0]
                + (vertex.y - center[1]) * depth_axis[1],
            )
            for vertex in (first, second)
        )
        if any(
            abs(position[0]) > half_width + tolerance
            for position in local_positions
        ):
            continue
        if any(
            abs(position[1]) > half_depth + tolerance
            for position in local_positions
        ):
            continue
        if abs(local_positions[0][1] - local_positions[1][1]) > tolerance:
            continue
        covered_targets.append(candidate)
    return tuple(covered_targets)


# ### Candidate helpers ###
def _find_doorway_bridge_targets(
    vertex_data: VertexData,
    room_center_vertex_ids: set[int],
    raw_center: tuple[float, float],
    preset_width_meters: float,
    center_tolerance: float,
) -> tuple[DoorwayBridgeTarget, ...]:
    vertices_by_id = {vertex.id: vertex for vertex in vertex_data.vertices}
    neighbor_ids_by_vertex_id, existing_edge_keys = _build_wall_lookups(
        vertex_data
    )
    candidate_vertices = tuple(
        vertex
        for vertex in vertex_data.vertices
        if vertex.id not in room_center_vertex_ids
    )
    candidates: list[DoorwayBridgeTarget] = []
    for first, second in _iter_nearby_vertex_pairs(
        candidate_vertices,
        raw_center,
        center_tolerance,
    ):
        if tuple(sorted((first.id, second.id))) in existing_edge_keys:
            continue

        delta_x = second.x - first.x
        delta_y = second.y - first.y
        span_pixels = math.hypot(delta_x, delta_y)
        width_meters = span_pixels * PIXEL_TO_METER
        if not MIN_DOORWAY_WIDTH_METERS <= width_meters <= MAX_DOORWAY_WIDTH_METERS:
            continue
        center = (
            (first.x + second.x) * 0.5,
            (first.y + second.y) * 0.5,
        )
        cursor_distance = math.dist(raw_center, center)
        first_neighbors = _get_incident_neighbors(
            first, neighbor_ids_by_vertex_id, vertices_by_id
        )
        second_neighbors = _get_incident_neighbors(
            second, neighbor_ids_by_vertex_id, vertices_by_id
        )
        if first_neighbors and not _has_aligned_outward_wall_branch(
            first, second, first_neighbors
        ):
            continue
        if second_neighbors and not _has_aligned_outward_wall_branch(
            second, first, second_neighbors
        ):
            continue
        if (
            not first_neighbors
            and not second_neighbors
            and not _uses_standard_free_point_alignment(first, second)
        ):
            continue
        candidates.append(
            DoorwayBridgeTarget(
                first_vertex_id=first.id,
                second_vertex_id=second.id,
                center=center,
                width_meters=width_meters,
                rotation_degrees=(
                    math.degrees(math.atan2(delta_y, delta_x)) + 90.0
                )
                % 180.0,
                cursor_distance=cursor_distance,
            )
        )

    ordered_candidates = sorted(
        candidates,
        key=lambda candidate: (
            candidate.cursor_distance,
            abs(candidate.width_meters - float(preset_width_meters)),
            candidate.width_meters,
            candidate.first_vertex_id,
            candidate.second_vertex_id,
        ),
    )
    valid_candidates: list[DoorwayBridgeTarget] = []
    for candidate in ordered_candidates:
        first = vertices_by_id.get(candidate.first_vertex_id)
        second = vertices_by_id.get(candidate.second_vertex_id)
        if (
            first is not None
            and second is not None
            and not _bridge_crosses_wall(vertex_data, first, second)
        ):
            valid_candidates.append(candidate)
    return tuple(valid_candidates)


# ### Lookup helpers ###
def _build_wall_lookups(
    vertex_data: VertexData,
) -> tuple[dict[int, list[int]], set[tuple[int, int]]]:
    neighbor_ids_by_vertex_id: dict[int, list[int]] = {
        vertex.id: [] for vertex in vertex_data.vertices
    }
    existing_edge_keys: set[tuple[int, int]] = set()
    for edge in vertex_data.edges:
        existing_edge_keys.add(
            tuple(sorted((edge.start_vertex_id, edge.end_vertex_id)))
        )
        neighbor_ids_by_vertex_id.setdefault(edge.start_vertex_id, []).append(
            edge.end_vertex_id
        )
        neighbor_ids_by_vertex_id.setdefault(edge.end_vertex_id, []).append(
            edge.start_vertex_id
        )
    return neighbor_ids_by_vertex_id, existing_edge_keys


def _get_incident_neighbors(
    vertex: Vertex,
    neighbor_ids_by_vertex_id: dict[int, list[int]],
    vertices_by_id: dict[int, Vertex],
) -> tuple[Vertex, ...]:
    return tuple(
        neighbor
        for neighbor_id in neighbor_ids_by_vertex_id.get(vertex.id, ())
        if (neighbor := vertices_by_id.get(neighbor_id)) is not None
    )


# ### Spatial helpers ###
def _iter_nearby_vertex_pairs(
    vertices: tuple[Vertex, ...],
    raw_center: tuple[float, float],
    center_tolerance: float,
) -> tuple[tuple[Vertex, Vertex], ...]:
    """Find pairs with a cursor-near midpoint without an all-pairs scan."""

    query_radius = max(center_tolerance * 2.0, 1e-6)
    vertices_by_cell: dict[tuple[int, int], list[Vertex]] = {}
    for vertex in vertices:
        cell = (
            math.floor(vertex.x / query_radius),
            math.floor(vertex.y / query_radius),
        )
        vertices_by_cell.setdefault(cell, []).append(vertex)

    pairs: list[tuple[Vertex, Vertex]] = []
    for first in vertices:
        reflected_x = raw_center[0] * 2.0 - first.x
        reflected_y = raw_center[1] * 2.0 - first.y
        minimum_cell_x = math.floor(
            (reflected_x - query_radius) / query_radius
        )
        maximum_cell_x = math.floor(
            (reflected_x + query_radius) / query_radius
        )
        minimum_cell_y = math.floor(
            (reflected_y - query_radius) / query_radius
        )
        maximum_cell_y = math.floor(
            (reflected_y + query_radius) / query_radius
        )
        for cell_x in range(minimum_cell_x, maximum_cell_x + 1):
            for cell_y in range(minimum_cell_y, maximum_cell_y + 1):
                for second in vertices_by_cell.get((cell_x, cell_y), ()):
                    if second.id <= first.id:
                        continue
                    center = (
                        (first.x + second.x) * 0.5,
                        (first.y + second.y) * 0.5,
                    )
                    if math.dist(center, raw_center) <= center_tolerance:
                        pairs.append((first, second))
    return tuple(pairs)


# ### Validation helpers ###
def _has_aligned_outward_wall_branch(
    vertex: Vertex,
    opposite_vertex: Vertex,
    incident_neighbors: tuple[Vertex, ...],
) -> bool:
    """Require wall support behind the point and none across the proposed gap."""

    bridge_delta_x = opposite_vertex.x - vertex.x
    bridge_delta_y = opposite_vertex.y - vertex.y
    bridge_length = math.hypot(bridge_delta_x, bridge_delta_y)
    if bridge_length <= 1e-6:
        return False
    has_outward_branch = False
    for neighbor in incident_neighbors:
        wall_delta_x = neighbor.x - vertex.x
        wall_delta_y = neighbor.y - vertex.y
        wall_length = math.hypot(wall_delta_x, wall_delta_y)
        if wall_length <= 1e-6:
            continue
        lateral_offset = abs(
            bridge_delta_x * wall_delta_y
            - bridge_delta_y * wall_delta_x
        ) / bridge_length
        if lateral_offset > STRAIGHT_ALIGNMENT_TOLERANCE_PIXELS:
            continue
        direction_dot = (
            wall_delta_x * bridge_delta_x + wall_delta_y * bridge_delta_y
        )
        if direction_dot > 1e-6:
            return False
        if direction_dot < -1e-6:
            has_outward_branch = True
    return has_outward_branch


def _uses_standard_free_point_alignment(
    first: Vertex,
    second: Vertex,
) -> bool:
    """Accept two free points only when they follow a 45-degree snap axis."""

    delta_x = abs(second.x - first.x)
    delta_y = abs(second.y - first.y)
    axis_offsets = (
        delta_x,
        delta_y,
        abs(delta_x - delta_y) / math.sqrt(2.0),
    )
    return min(axis_offsets) <= STRAIGHT_ALIGNMENT_TOLERANCE_PIXELS


def _bridge_crosses_wall(
    vertex_data: VertexData,
    first: Vertex,
    second: Vertex,
) -> bool:
    bridge_vertex_ids = {first.id, second.id}
    vertices_by_id = {vertex.id: vertex for vertex in vertex_data.vertices}
    for edge in vertex_data.edges:
        if bridge_vertex_ids.intersection(
            (edge.start_vertex_id, edge.end_vertex_id)
        ):
            continue
        edge_start = vertices_by_id.get(edge.start_vertex_id)
        edge_end = vertices_by_id.get(edge.end_vertex_id)
        if edge_start is None or edge_end is None:
            continue
        if _segments_intersect(first, second, edge_start, edge_end):
            return True
    return False


def _segments_intersect(
    first_start: Vertex,
    first_end: Vertex,
    second_start: Vertex,
    second_end: Vertex,
) -> bool:
    first_delta_x = first_end.x - first_start.x
    first_delta_y = first_end.y - first_start.y
    second_delta_x = second_end.x - second_start.x
    second_delta_y = second_end.y - second_start.y
    cross_product = (
        first_delta_x * second_delta_y - first_delta_y * second_delta_x
    )
    if abs(cross_product) <= 1e-6:
        offset_cross_product = (
            (second_start.x - first_start.x) * first_delta_y
            - (second_start.y - first_start.y) * first_delta_x
        )
        if abs(offset_cross_product) > 1e-6:
            return False
        if abs(first_delta_x) >= abs(first_delta_y):
            first_interval = sorted((first_start.x, first_end.x))
            second_interval = sorted((second_start.x, second_end.x))
        else:
            first_interval = sorted((first_start.y, first_end.y))
            second_interval = sorted((second_start.y, second_end.y))
        return (
            min(first_interval[1], second_interval[1])
            - max(first_interval[0], second_interval[0])
            > 1e-6
        )

    start_delta_x = second_start.x - first_start.x
    start_delta_y = second_start.y - first_start.y
    first_ratio = (
        start_delta_x * second_delta_y - start_delta_y * second_delta_x
    ) / cross_product
    second_ratio = (
        start_delta_x * first_delta_y - start_delta_y * first_delta_x
    ) / cross_product
    return (
        -1e-6 <= first_ratio <= 1.0 + 1e-6
        and -1e-6 <= second_ratio <= 1.0 + 1e-6
    )
