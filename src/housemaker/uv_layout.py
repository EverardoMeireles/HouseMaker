# ### Imports ###
from __future__ import annotations

import heapq
import math
from collections.abc import Iterator
from dataclasses import dataclass

from housemaker.models import (
    DEFAULT_UV_MAP_HEIGHT,
    DEFAULT_UV_MAP_WIDTH,
    DEFAULT_WALL_UV_ROTATION_DEGREES,
    DEFAULT_WALL_UV_SCALE,
    PIXEL_TO_METER,
    RoomData,
    Vertex,
    VertexData,
)

# ### Constants ###
STRAIGHT_WALL_TOLERANCE = 1e-5
UV_MAP_PADDING = 12.0
UV_MAP_SIZE_POWERS = (64, 128, 256, 512, 1024, 2048, 4096, 8192)
UV_OPTIMIZATION_BEAM_WIDTH = 96
UV_OPTIMIZATION_ROTATION_CANDIDATES = tuple(range(360))
UV_OPTIMIZATION_SCALE_STEP = 0.01


# ### Data models ###
@dataclass(frozen=True)
class RoomWall:
    key: str
    start_vertex_id: int
    end_vertex_id: int
    start_point: tuple[float, float]
    end_point: tuple[float, float]
    length: float
    projection_direction: str


@dataclass(frozen=True)
class UvWallPlacement:
    wall: RoomWall
    uv_rect: tuple[float, float, float, float]
    natural_size: tuple[float, float]
    rotation_degrees: int


@dataclass(frozen=True)
class UvLayout:
    placements: list[UvWallPlacement]
    hidden_wall_count: int


@dataclass(frozen=True)
class UvOptimizationState:
    rotations: tuple[int, ...]
    scales: tuple[float, ...]
    cursor_x: float
    cursor_y: float
    row_height: float
    occupied_area: float
    bounding_area: float
    placed_wall_count: int
    hidden_wall_count: int
    scale_change_amount: float
    rotation_change_amount: int


@dataclass(frozen=True)
class UvOptimizationResult:
    wall_uv_rotations: dict[str, int]
    wall_uv_scales: dict[str, float]


@dataclass(frozen=True)
class UvOptimizationCandidate:
    rotation_degrees: int
    wall_scale: float
    wall_width: float
    wall_height: float
    bounding_width: float
    bounding_height: float
    scale_change_amount: float
    rotation_change_amount: int


# ### Public helpers ###
def build_uv_wall_layout(
    room: RoomData,
    vertex_data: VertexData,
    wall_height_meters: float,
) -> UvLayout:
    walls = build_room_walls(room, vertex_data)
    return _build_uv_wall_layout_for_size(
        walls=walls,
        map_width=float(room.uv_map_width),
        map_height=float(room.uv_map_height),
        wall_height_meters=wall_height_meters,
        wall_uv_scales=room.wall_uv_scales,
        wall_uv_rotations=room.wall_uv_rotations,
    )


def calculate_unoccupied_uv_pixels(
    room: RoomData,
    vertex_data: VertexData,
    wall_height_meters: float,
) -> int:
    layout = build_uv_wall_layout(room, vertex_data, wall_height_meters)
    map_area = max(1.0, float(room.uv_map_width)) * max(
        1.0,
        float(room.uv_map_height),
    )
    occupied_area = sum(
        _get_placement_area(placement)
        for placement in layout.placements
    )
    return max(0, int(round(map_area - occupied_area)))


def optimize_room_wall_uv_rotations(
    room: RoomData,
    vertex_data: VertexData,
    wall_height_meters: float,
) -> dict[str, int]:
    result = optimize_room_wall_uvs(
        room=room,
        vertex_data=vertex_data,
        wall_height_meters=wall_height_meters,
        max_scale_variation=0.0,
    )
    return result.wall_uv_rotations


def optimize_room_wall_uvs(
    room: RoomData,
    vertex_data: VertexData,
    wall_height_meters: float,
    max_scale_variation: float,
) -> UvOptimizationResult:
    walls = build_room_walls(room, vertex_data)
    if not walls:
        return UvOptimizationResult(wall_uv_rotations={}, wall_uv_scales={})

    current_rotations = {
        wall.key: _normalize_wall_uv_rotation(
            room.wall_uv_rotations.get(
                wall.key,
                DEFAULT_WALL_UV_ROTATION_DEGREES,
            )
        )
        for wall in walls
    }
    current_scales = {
        wall.key: _normalize_wall_uv_scale(
            room.wall_uv_scales.get(wall.key, DEFAULT_WALL_UV_SCALE)
        )
        for wall in walls
    }
    current_result = UvOptimizationResult(
        wall_uv_rotations=current_rotations,
        wall_uv_scales=current_scales,
    )
    rotation_only_result = _find_optimized_uv_settings(
        walls=walls,
        map_width=float(room.uv_map_width),
        map_height=float(room.uv_map_height),
        wall_height_meters=wall_height_meters,
        wall_uv_scales=current_scales,
        wall_uv_rotations=current_rotations,
        max_scale_variation=0.0,
    )
    optimized_result = _find_optimized_uv_settings(
        walls=walls,
        map_width=float(room.uv_map_width),
        map_height=float(room.uv_map_height),
        wall_height_meters=wall_height_meters,
        wall_uv_scales=current_scales,
        wall_uv_rotations=current_rotations,
        max_scale_variation=max_scale_variation,
    )
    return max(
        (current_result, rotation_only_result, optimized_result),
        key=lambda result: _score_uv_optimization_result(
            result=result,
            walls=walls,
            map_width=float(room.uv_map_width),
            map_height=float(room.uv_map_height),
            wall_height_meters=wall_height_meters,
        ),
    )


def calculate_minimum_uv_map_size(
    room: RoomData,
    vertex_data: VertexData,
    wall_height_meters: float,
) -> tuple[int, int]:
    walls = build_room_walls(room, vertex_data)
    if not walls:
        return DEFAULT_UV_MAP_WIDTH, DEFAULT_UV_MAP_HEIGHT

    best_size: tuple[int, int] | None = None
    best_score: tuple[int, int, int] | None = None
    for map_width in UV_MAP_SIZE_POWERS:
        for map_height in UV_MAP_SIZE_POWERS:
            layout = _build_uv_wall_layout_for_size(
                walls=walls,
                map_width=float(map_width),
                map_height=float(map_height),
                wall_height_meters=wall_height_meters,
                wall_uv_scales={},
                wall_uv_rotations={},
            )
            if layout.hidden_wall_count > 0:
                continue

            score = (
                map_width * map_height,
                abs(map_width - map_height),
                map_width,
            )
            if best_score is None or score < best_score:
                best_size = (map_width, map_height)
                best_score = score

    if best_size is None:
        return UV_MAP_SIZE_POWERS[-1], UV_MAP_SIZE_POWERS[-1]

    return best_size


def initialize_room_uv_map_size(
    room: RoomData,
    vertex_data: VertexData,
    wall_height_meters: float,
) -> None:
    uv_map_width, uv_map_height = calculate_minimum_uv_map_size(
        room=room,
        vertex_data=vertex_data,
        wall_height_meters=wall_height_meters,
    )
    room.uv_map_width = uv_map_width
    room.uv_map_height = uv_map_height


def build_room_walls(room: RoomData, vertex_data: VertexData | None) -> list[RoomWall]:
    if vertex_data is None:
        return []

    room_vertices = [
        vertex
        for vertex_id in room.vertex_ids
        if (vertex := vertex_data.get_vertex(vertex_id)) is not None
    ]
    if len(room_vertices) < 3:
        return []

    center_point = _get_room_center_point(room, vertex_data, room_vertices)
    ordered_vertices = _order_vertices_around_center(room_vertices, center_point)
    wall_vertices = _remove_straight_through_vertices(ordered_vertices)
    if len(wall_vertices) < 3:
        return []

    return [
        _build_room_wall(
            start_vertex=wall_vertices[index],
            end_vertex=wall_vertices[(index + 1) % len(wall_vertices)],
            center_point=center_point,
        )
        for index in range(len(wall_vertices))
    ]


def get_room_wall_keys(
    room: RoomData,
    vertex_data: VertexData | None,
) -> set[str]:
    return {wall.key for wall in build_room_walls(room, vertex_data)}


# ### Layout helpers ###
def _build_uv_wall_layout_for_size(
    walls: list[RoomWall],
    map_width: float,
    map_height: float,
    wall_height_meters: float,
    wall_uv_scales: dict[str, float],
    wall_uv_rotations: dict[str, int],
) -> UvLayout:
    map_width = max(1.0, map_width)
    map_height = max(1.0, map_height)
    max_right = map_width - UV_MAP_PADDING
    max_bottom = map_height - UV_MAP_PADDING
    wall_height_pixels = max(1.0, wall_height_meters / PIXEL_TO_METER)
    placements: list[UvWallPlacement] = []
    hidden_wall_count = 0
    cursor_x = UV_MAP_PADDING
    cursor_y = UV_MAP_PADDING
    row_height = 0.0

    for wall in walls:
        wall_scale = _normalize_wall_uv_scale(
            wall_uv_scales.get(wall.key, DEFAULT_WALL_UV_SCALE)
        )
        wall_rotation = _normalize_wall_uv_rotation(
            wall_uv_rotations.get(wall.key, DEFAULT_WALL_UV_ROTATION_DEGREES)
        )
        wall_width, wall_height = _calculate_scaled_wall_natural_size(
            wall=wall,
            wall_height_pixels=wall_height_pixels,
            wall_scale=wall_scale,
        )
        bounding_width, bounding_height = _calculate_rotated_bounds(
            width=wall_width,
            height=wall_height,
            rotation_degrees=wall_rotation,
        )

        if bounding_width > map_width - UV_MAP_PADDING * 2.0:
            hidden_wall_count += 1
            continue
        if bounding_height > map_height - UV_MAP_PADDING * 2.0:
            hidden_wall_count += 1
            continue

        if cursor_x + bounding_width > max_right and cursor_x > UV_MAP_PADDING:
            cursor_x = UV_MAP_PADDING
            cursor_y += row_height + UV_MAP_PADDING
            row_height = 0.0

        if cursor_y + bounding_height > max_bottom:
            hidden_wall_count += 1
            continue

        placements.append(
            UvWallPlacement(
                wall=wall,
                uv_rect=(cursor_x, cursor_y, bounding_width, bounding_height),
                natural_size=(wall_width, wall_height),
                rotation_degrees=wall_rotation,
            )
        )
        cursor_x += bounding_width + UV_MAP_PADDING
        row_height = max(row_height, bounding_height)

    return UvLayout(placements=placements, hidden_wall_count=hidden_wall_count)


def _find_optimized_uv_settings(
    walls: list[RoomWall],
    map_width: float,
    map_height: float,
    wall_height_meters: float,
    wall_uv_scales: dict[str, float],
    wall_uv_rotations: dict[str, int],
    max_scale_variation: float,
) -> UvOptimizationResult:
    map_width = max(1.0, map_width)
    map_height = max(1.0, map_height)
    max_right = map_width - UV_MAP_PADDING
    max_bottom = map_height - UV_MAP_PADDING
    wall_height_pixels = max(1.0, wall_height_meters / PIXEL_TO_METER)
    states = [
        UvOptimizationState(
            rotations=(),
            scales=(),
            cursor_x=UV_MAP_PADDING,
            cursor_y=UV_MAP_PADDING,
            row_height=0.0,
            occupied_area=0.0,
            bounding_area=0.0,
            placed_wall_count=0,
            hidden_wall_count=0,
            scale_change_amount=0.0,
            rotation_change_amount=0,
        )
    ]

    for wall in walls:
        wall_scale = wall_uv_scales.get(wall.key, DEFAULT_WALL_UV_SCALE)
        wall_rotation = wall_uv_rotations.get(
            wall.key,
            DEFAULT_WALL_UV_ROTATION_DEGREES,
        )
        states = heapq.nlargest(
            UV_OPTIMIZATION_BEAM_WIDTH,
            _iter_extended_uv_optimization_states(
                states=states,
                wall=wall,
                wall_height_pixels=wall_height_pixels,
                current_scale=wall_scale,
                current_rotation=wall_rotation,
                max_scale_variation=max_scale_variation,
                map_width=map_width,
                map_height=map_height,
                max_right=max_right,
                max_bottom=max_bottom,
            ),
            key=_score_uv_optimization_state,
        )

    best_state = max(states, key=_score_uv_optimization_state)
    return UvOptimizationResult(
        wall_uv_rotations={
            wall.key: best_state.rotations[wall_index]
            for wall_index, wall in enumerate(walls)
        },
        wall_uv_scales={
            wall.key: best_state.scales[wall_index]
            for wall_index, wall in enumerate(walls)
        },
    )


def _iter_extended_uv_optimization_states(
    states: list[UvOptimizationState],
    wall: RoomWall,
    wall_height_pixels: float,
    current_scale: float,
    current_rotation: int,
    max_scale_variation: float,
    map_width: float,
    map_height: float,
    max_right: float,
    max_bottom: float,
) -> Iterator[UvOptimizationState]:
    uv_candidates = _build_uv_optimization_candidates(
        wall=wall,
        wall_height_pixels=wall_height_pixels,
        current_scale=current_scale,
        current_rotation=current_rotation,
        max_scale_variation=max_scale_variation,
    )
    for state in states:
        for uv_candidate in uv_candidates:
            yield _extend_uv_optimization_state(
                state=state,
                uv_candidate=uv_candidate,
                map_width=map_width,
                map_height=map_height,
                max_right=max_right,
                max_bottom=max_bottom,
            )


def _extend_uv_optimization_state(
    state: UvOptimizationState,
    uv_candidate: UvOptimizationCandidate,
    map_width: float,
    map_height: float,
    max_right: float,
    max_bottom: float,
) -> UvOptimizationState:
    rotations = state.rotations + (uv_candidate.rotation_degrees,)
    scales = state.scales + (uv_candidate.wall_scale,)
    scale_change_amount = (
        state.scale_change_amount + uv_candidate.scale_change_amount
    )
    rotation_change_amount = (
        state.rotation_change_amount + uv_candidate.rotation_change_amount
    )
    if uv_candidate.bounding_width > map_width - UV_MAP_PADDING * 2.0:
        return _extend_hidden_uv_optimization_state(
            state=state,
            rotations=rotations,
            scales=scales,
            scale_change_amount=scale_change_amount,
            rotation_change_amount=rotation_change_amount,
        )
    if uv_candidate.bounding_height > map_height - UV_MAP_PADDING * 2.0:
        return _extend_hidden_uv_optimization_state(
            state=state,
            rotations=rotations,
            scales=scales,
            scale_change_amount=scale_change_amount,
            rotation_change_amount=rotation_change_amount,
        )

    cursor_x = state.cursor_x
    cursor_y = state.cursor_y
    row_height = state.row_height
    if (
        cursor_x + uv_candidate.bounding_width > max_right
        and cursor_x > UV_MAP_PADDING
    ):
        cursor_x = UV_MAP_PADDING
        cursor_y += row_height + UV_MAP_PADDING
        row_height = 0.0

    if cursor_y + uv_candidate.bounding_height > max_bottom:
        return _extend_hidden_uv_optimization_state(
            state=state,
            rotations=rotations,
            scales=scales,
            scale_change_amount=scale_change_amount,
            rotation_change_amount=rotation_change_amount,
        )

    return UvOptimizationState(
        rotations=rotations,
        scales=scales,
        cursor_x=cursor_x + uv_candidate.bounding_width + UV_MAP_PADDING,
        cursor_y=cursor_y,
        row_height=max(row_height, uv_candidate.bounding_height),
        occupied_area=(
            state.occupied_area
            + uv_candidate.wall_width * uv_candidate.wall_height
        ),
        bounding_area=(
            state.bounding_area
            + uv_candidate.bounding_width * uv_candidate.bounding_height
        ),
        placed_wall_count=state.placed_wall_count + 1,
        hidden_wall_count=state.hidden_wall_count,
        scale_change_amount=scale_change_amount,
        rotation_change_amount=rotation_change_amount,
    )


def _extend_hidden_uv_optimization_state(
    state: UvOptimizationState,
    rotations: tuple[int, ...],
    scales: tuple[float, ...],
    scale_change_amount: float,
    rotation_change_amount: int,
) -> UvOptimizationState:
    return UvOptimizationState(
        rotations=rotations,
        scales=scales,
        cursor_x=state.cursor_x,
        cursor_y=state.cursor_y,
        row_height=state.row_height,
        occupied_area=state.occupied_area,
        bounding_area=state.bounding_area,
        placed_wall_count=state.placed_wall_count,
        hidden_wall_count=state.hidden_wall_count + 1,
        scale_change_amount=scale_change_amount,
        rotation_change_amount=rotation_change_amount,
    )


def _build_uv_optimization_candidates(
    wall: RoomWall,
    wall_height_pixels: float,
    current_scale: float,
    current_rotation: int,
    max_scale_variation: float,
) -> tuple[UvOptimizationCandidate, ...]:
    return tuple(
        _build_uv_optimization_candidate(
            wall=wall,
            wall_height_pixels=wall_height_pixels,
            current_scale=current_scale,
            current_rotation=current_rotation,
            wall_scale=wall_scale,
            rotation_degrees=rotation_degrees,
        )
        for wall_scale in _build_wall_scale_candidates(
            wall_scale=current_scale,
            max_scale_variation=max_scale_variation,
        )
        for rotation_degrees in _build_wall_rotation_candidates(current_rotation)
    )


def _build_uv_optimization_candidate(
    wall: RoomWall,
    wall_height_pixels: float,
    current_scale: float,
    current_rotation: int,
    wall_scale: float,
    rotation_degrees: int,
) -> UvOptimizationCandidate:
    wall_width, wall_height = _calculate_scaled_wall_natural_size(
        wall=wall,
        wall_height_pixels=wall_height_pixels,
        wall_scale=wall_scale,
    )
    bounding_width, bounding_height = _calculate_rotated_bounds(
        width=wall_width,
        height=wall_height,
        rotation_degrees=rotation_degrees,
    )
    return UvOptimizationCandidate(
        rotation_degrees=rotation_degrees,
        wall_scale=wall_scale,
        wall_width=wall_width,
        wall_height=wall_height,
        bounding_width=bounding_width,
        bounding_height=bounding_height,
        scale_change_amount=abs(wall_scale - current_scale),
        rotation_change_amount=_get_rotation_delta(
            rotation_degrees,
            current_rotation,
        ),
    )


def _calculate_scaled_wall_natural_size(
    wall: RoomWall,
    wall_height_pixels: float,
    wall_scale: float,
) -> tuple[float, float]:
    return (
        max(1.0, wall.length * wall_scale),
        max(1.0, wall_height_pixels * wall_scale),
    )


def _build_wall_scale_candidates(
    wall_scale: float,
    max_scale_variation: float,
) -> tuple[float, ...]:
    scale_variation = max(0.0, float(max_scale_variation))
    variation_steps = int(round(scale_variation / UV_OPTIMIZATION_SCALE_STEP))
    scale_candidates = {
        max(
            0.01,
            round(
                wall_scale + step * UV_OPTIMIZATION_SCALE_STEP,
                3,
            ),
        )
        for step in range(-variation_steps, variation_steps + 1)
    }
    return tuple(sorted(scale_candidates))


def _build_wall_rotation_candidates(current_rotation: int) -> tuple[int, ...]:
    normalized_rotation = _normalize_wall_uv_rotation(current_rotation)
    return (normalized_rotation,) + tuple(
        rotation
        for rotation in UV_OPTIMIZATION_ROTATION_CANDIDATES
        if rotation != normalized_rotation
    )


def _score_uv_optimization_state(
    state: UvOptimizationState,
) -> tuple[float, int, int, float, float, float, float, int]:
    return (
        state.occupied_area,
        -state.hidden_wall_count,
        state.placed_wall_count,
        -state.bounding_area,
        -(state.cursor_y + state.row_height),
        -state.cursor_x,
        -state.scale_change_amount,
        -state.rotation_change_amount,
    )


def _score_uv_layout(
    layout: UvLayout,
) -> tuple[float, int, int, float, float]:
    occupied_area = sum(
        _get_placement_area(placement)
        for placement in layout.placements
    )
    bounding_area = sum(
        placement.uv_rect[2] * placement.uv_rect[3]
        for placement in layout.placements
    )
    layout_bottom = max(
        (
            placement.uv_rect[1] + placement.uv_rect[3]
            for placement in layout.placements
        ),
        default=0.0,
    )
    return (
        occupied_area,
        -layout.hidden_wall_count,
        len(layout.placements),
        -bounding_area,
        -layout_bottom,
    )


def _score_uv_optimization_result(
    result: UvOptimizationResult,
    walls: list[RoomWall],
    map_width: float,
    map_height: float,
    wall_height_meters: float,
) -> tuple[float, int, int, float, float]:
    layout = _build_uv_wall_layout_for_size(
        walls=walls,
        map_width=map_width,
        map_height=map_height,
        wall_height_meters=wall_height_meters,
        wall_uv_scales=result.wall_uv_scales,
        wall_uv_rotations=result.wall_uv_rotations,
    )
    return _score_uv_layout(layout)


def get_rotated_uv_corners(
    placement: UvWallPlacement,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]:
    uv_x, uv_y, uv_width, uv_height = placement.uv_rect
    wall_width, wall_height = placement.natural_size
    center_x = uv_x + uv_width / 2.0
    center_y = uv_y + uv_height / 2.0
    return tuple(
        _rotate_uv_point(
            center_x=center_x,
            center_y=center_y,
            offset_x=offset_x,
            offset_y=offset_y,
            rotation_degrees=placement.rotation_degrees,
        )
        for offset_x, offset_y in (
            (-wall_width / 2.0, -wall_height / 2.0),
            (wall_width / 2.0, -wall_height / 2.0),
            (wall_width / 2.0, wall_height / 2.0),
            (-wall_width / 2.0, wall_height / 2.0),
        )
    )


def _get_placement_area(placement: UvWallPlacement) -> float:
    natural_width, natural_height = placement.natural_size
    return natural_width * natural_height


def _calculate_rotated_bounds(
    width: float,
    height: float,
    rotation_degrees: int,
) -> tuple[float, float]:
    rotation_radians = math.radians(rotation_degrees % 360)
    cosine = abs(math.cos(rotation_radians))
    sine = abs(math.sin(rotation_radians))
    return (
        width * cosine + height * sine,
        width * sine + height * cosine,
    )


def _rotate_uv_point(
    center_x: float,
    center_y: float,
    offset_x: float,
    offset_y: float,
    rotation_degrees: int,
) -> tuple[float, float]:
    rotation_radians = math.radians(rotation_degrees % 360)
    cosine = math.cos(rotation_radians)
    sine = math.sin(rotation_radians)
    return (
        center_x + offset_x * cosine - offset_y * sine,
        center_y + offset_x * sine + offset_y * cosine,
    )


# ### Geometry helpers ###
def _get_room_center_point(
    room: RoomData,
    vertex_data: VertexData,
    room_vertices: list[Vertex],
) -> tuple[float, float]:
    center_vertex = vertex_data.get_vertex(room.center_vertex_id)
    if center_vertex is not None:
        return center_vertex.x, center_vertex.y

    return (
        sum(vertex.x for vertex in room_vertices) / len(room_vertices),
        sum(vertex.y for vertex in room_vertices) / len(room_vertices),
    )


def _order_vertices_around_center(
    vertices: list[Vertex],
    center_point: tuple[float, float],
) -> list[Vertex]:
    center_x, center_y = center_point
    return sorted(
        vertices,
        key=lambda vertex: math.atan2(vertex.y - center_y, vertex.x - center_x),
    )


def _remove_straight_through_vertices(vertices: list[Vertex]) -> list[Vertex]:
    simplified_vertices = list(vertices)
    changed = True
    while changed and len(simplified_vertices) > 3:
        changed = False
        for current_index, current_vertex in enumerate(list(simplified_vertices)):
            previous_vertex = simplified_vertices[current_index - 1]
            next_vertex = simplified_vertices[
                (current_index + 1) % len(simplified_vertices)
            ]
            if not _is_straight_through(previous_vertex, current_vertex, next_vertex):
                continue

            simplified_vertices.remove(current_vertex)
            changed = True
            break

    return simplified_vertices


def _is_straight_through(
    previous_vertex: Vertex,
    current_vertex: Vertex,
    next_vertex: Vertex,
) -> bool:
    previous_vector = (
        previous_vertex.x - current_vertex.x,
        previous_vertex.y - current_vertex.y,
    )
    next_vector = (
        next_vertex.x - current_vertex.x,
        next_vertex.y - current_vertex.y,
    )
    previous_length = math.hypot(previous_vector[0], previous_vector[1])
    next_length = math.hypot(next_vector[0], next_vector[1])
    if previous_length <= STRAIGHT_WALL_TOLERANCE:
        return False
    if next_length <= STRAIGHT_WALL_TOLERANCE:
        return False

    cross_product = (
        previous_vector[0] * next_vector[1]
        - previous_vector[1] * next_vector[0]
    )
    dot_product = (
        previous_vector[0] * next_vector[0]
        + previous_vector[1] * next_vector[1]
    )
    normalized_cross = abs(cross_product) / (previous_length * next_length)
    return normalized_cross <= STRAIGHT_WALL_TOLERANCE and dot_product < 0.0


def _build_room_wall(
    start_vertex: Vertex,
    end_vertex: Vertex,
    center_point: tuple[float, float],
) -> RoomWall:
    wall_midpoint = (
        (start_vertex.x + end_vertex.x) / 2.0,
        (start_vertex.y + end_vertex.y) / 2.0,
    )
    wall_length = math.hypot(
        end_vertex.x - start_vertex.x,
        end_vertex.y - start_vertex.y,
    )
    return RoomWall(
        key=_build_wall_key(start_vertex.id, end_vertex.id),
        start_vertex_id=start_vertex.id,
        end_vertex_id=end_vertex.id,
        start_point=(start_vertex.x, start_vertex.y),
        end_point=(end_vertex.x, end_vertex.y),
        length=wall_length,
        projection_direction=_get_projection_direction(center_point, wall_midpoint),
    )


def _build_wall_key(start_vertex_id: int, end_vertex_id: int) -> str:
    return f"{min(start_vertex_id, end_vertex_id)}:{max(start_vertex_id, end_vertex_id)}"


def _get_projection_direction(
    center_point: tuple[float, float],
    wall_midpoint: tuple[float, float],
) -> str:
    delta_x = wall_midpoint[0] - center_point[0]
    delta_y = wall_midpoint[1] - center_point[1]
    if abs(delta_x) >= abs(delta_y):
        return "East" if delta_x >= 0.0 else "West"

    return "South" if delta_y >= 0.0 else "North"


# ### Numeric helpers ###
def _normalize_wall_uv_rotation(rotation_degrees: int) -> int:
    try:
        return int(round(float(rotation_degrees))) % 360
    except (TypeError, ValueError):
        return 0


def _normalize_wall_uv_scale(wall_scale: object) -> float:
    try:
        return max(0.01, float(wall_scale))
    except (TypeError, ValueError):
        return DEFAULT_WALL_UV_SCALE


def _get_rotation_delta(rotation_degrees: int, current_rotation: int) -> int:
    rotation_delta = abs(
        _normalize_wall_uv_rotation(rotation_degrees)
        - _normalize_wall_uv_rotation(current_rotation)
    )
    return min(rotation_delta, 360 - rotation_delta)
