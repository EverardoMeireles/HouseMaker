# ### Imports ###
from __future__ import annotations

import math
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

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
UV_UNIFORM_SCALE_SEARCH_PASSES = 18
UV_COMPLEX_SCALE_STEP = 0.05
UV_COMPLEX_FAILURE_LIMIT = 100
UV_COMPLEX_MAX_LOOPS_PER_PASS = 5000
UV_COMPLEX_ROTATION_CANDIDATE_COUNT = 36
UV_SUBDIVISION_EPSILON = 1e-6


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
    segment_index: int = 0
    segment_count: int = 1
    source_start_ratio: float = 0.0
    source_end_ratio: float = 1.0


@dataclass(frozen=True)
class UvLayout:
    placements: list[UvWallPlacement]
    hidden_wall_count: int


@dataclass(frozen=True)
class UvFreeRectangle:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class UvOptimizationResult:
    wall_uv_rotations: dict[str, int]
    wall_uv_scales: dict[str, float]
    wall_uv_positions: dict[str, tuple[float, float]]
    wall_subdivisions: dict[str, int] = field(default_factory=dict)
    wall_subdivision_positions: dict[
        str,
        tuple[tuple[float, float], ...],
    ] = field(default_factory=dict)
    wall_subdivision_source_ranges: dict[
        str,
        tuple[tuple[float, float], ...],
    ] = field(default_factory=dict)


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
        wall_uv_positions=room.wall_uv_positions,
        wall_subdivisions=room.wall_subdivisions,
        wall_subdivision_positions=room.wall_subdivision_positions,
        wall_subdivision_source_ranges=room.wall_subdivision_source_ranges,
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


def optimize_room_wall_uvs(
    room: RoomData,
    vertex_data: VertexData,
    wall_height_meters: float,
    use_complex_optimization: bool = False,
    use_subdivision_optimization: bool = False,
    complex_optimization_passes: int = 1,
) -> UvOptimizationResult:
    walls = build_room_walls(room, vertex_data)
    if not walls:
        return UvOptimizationResult(
            wall_uv_rotations={},
            wall_uv_scales={},
            wall_uv_positions={},
        )

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

    if use_subdivision_optimization:
        optimized_result = _find_subdivision_uv_optimization_result(
            walls=walls,
            map_width=float(room.uv_map_width),
            map_height=float(room.uv_map_height),
            wall_height_meters=wall_height_meters,
            base_wall_uv_scales=current_scales,
        )
        if optimized_result is not None:
            return optimized_result

        return UvOptimizationResult(
            wall_uv_rotations={},
            wall_uv_scales={},
            wall_uv_positions={},
        )

    if use_complex_optimization:
        current_result = _build_current_uv_optimization_result(
            room=room,
            walls=walls,
            map_width=float(room.uv_map_width),
            map_height=float(room.uv_map_height),
            wall_height_meters=wall_height_meters,
            wall_uv_scales=current_scales,
            wall_uv_rotations=current_rotations,
        )
        if current_result is None:
            return UvOptimizationResult(
                wall_uv_rotations={},
                wall_uv_scales={},
                wall_uv_positions={},
            )

        return _find_complex_uv_optimization_result(
            walls=walls,
            map_width=float(room.uv_map_width),
            map_height=float(room.uv_map_height),
            wall_height_meters=wall_height_meters,
            initial_result=current_result,
            pass_count=complex_optimization_passes,
        )

    phase_one_result = _find_max_uniform_scale_uv_result(
        walls=walls,
        map_width=float(room.uv_map_width),
        map_height=float(room.uv_map_height),
        wall_height_meters=wall_height_meters,
        wall_uv_rotations=current_rotations,
    )
    if phase_one_result is None:
        return UvOptimizationResult(
            wall_uv_rotations={},
            wall_uv_scales={},
            wall_uv_positions={},
        )

    phase_two_result = _find_simple_rotation_uniform_scale_uv_result(
        walls=walls,
        map_width=float(room.uv_map_width),
        map_height=float(room.uv_map_height),
        wall_height_meters=wall_height_meters,
        fallback_result=phase_one_result,
    )

    return phase_two_result


def rebuild_room_subdivision_uvs(
    room: RoomData,
    vertex_data: VertexData,
    wall_height_meters: float,
) -> UvOptimizationResult | None:
    if not room.wall_subdivisions:
        return None

    walls = build_room_walls(room, vertex_data)
    if not walls:
        return None

    return _build_subdivision_row_uv_result(
        walls=walls,
        map_width=float(room.uv_map_width),
        map_height=float(room.uv_map_height),
        wall_height_meters=wall_height_meters,
        base_wall_uv_scales={
            wall.key: _normalize_wall_uv_scale(
                room.wall_uv_scales.get(wall.key, DEFAULT_WALL_UV_SCALE)
            )
            for wall in walls
        },
        wall_uv_rotations={
            wall.key: _normalize_wall_uv_rotation(
                room.wall_uv_rotations.get(
                    wall.key,
                    DEFAULT_WALL_UV_ROTATION_DEGREES,
                )
            )
            for wall in walls
        },
        scale_multiplier=1.0,
    )


# ### Optimization helpers ###
def _find_subdivision_uv_optimization_result(
    walls: list[RoomWall],
    map_width: float,
    map_height: float,
    wall_height_meters: float,
    base_wall_uv_scales: dict[str, float],
) -> UvOptimizationResult | None:
    wall_uv_rotations = {
        wall.key: DEFAULT_WALL_UV_ROTATION_DEGREES
        for wall in walls
    }
    low_multiplier = 0.01
    high_multiplier = 1.0
    best_result = _build_subdivision_row_uv_result(
        walls=walls,
        map_width=map_width,
        map_height=map_height,
        wall_height_meters=wall_height_meters,
        base_wall_uv_scales=base_wall_uv_scales,
        wall_uv_rotations=wall_uv_rotations,
        scale_multiplier=low_multiplier,
    )
    if best_result is None:
        return None

    candidate_result = _build_subdivision_row_uv_result(
        walls=walls,
        map_width=map_width,
        map_height=map_height,
        wall_height_meters=wall_height_meters,
        base_wall_uv_scales=base_wall_uv_scales,
        wall_uv_rotations=wall_uv_rotations,
        scale_multiplier=high_multiplier,
    )
    if candidate_result is not None:
        best_result = candidate_result
        low_multiplier = high_multiplier
        high_multiplier *= 2.0
        while high_multiplier <= 100.0:
            candidate_result = _build_subdivision_row_uv_result(
                walls=walls,
                map_width=map_width,
                map_height=map_height,
                wall_height_meters=wall_height_meters,
                base_wall_uv_scales=base_wall_uv_scales,
                wall_uv_rotations=wall_uv_rotations,
                scale_multiplier=high_multiplier,
            )
            if candidate_result is None:
                break

            best_result = candidate_result
            low_multiplier = high_multiplier
            high_multiplier *= 2.0

    high_multiplier = min(high_multiplier, 100.0)
    for _ in range(UV_UNIFORM_SCALE_SEARCH_PASSES):
        next_multiplier = (low_multiplier + high_multiplier) / 2.0
        candidate_result = _build_subdivision_row_uv_result(
            walls=walls,
            map_width=map_width,
            map_height=map_height,
            wall_height_meters=wall_height_meters,
            base_wall_uv_scales=base_wall_uv_scales,
            wall_uv_rotations=wall_uv_rotations,
            scale_multiplier=next_multiplier,
        )
        if candidate_result is None:
            high_multiplier = next_multiplier
            continue

        best_result = candidate_result
        low_multiplier = next_multiplier

    return best_result


def _build_subdivision_row_uv_result(
    walls: list[RoomWall],
    map_width: float,
    map_height: float,
    wall_height_meters: float,
    base_wall_uv_scales: dict[str, float],
    wall_uv_rotations: dict[str, int],
    scale_multiplier: float,
) -> UvOptimizationResult | None:
    map_width = max(1.0, map_width)
    map_height = max(1.0, map_height)
    wall_height_pixels = max(1.0, wall_height_meters / PIXEL_TO_METER)
    row_x = 0.0
    row_y = 0.0
    row_height = 0.0
    optimized_scales: dict[str, float] = {}
    optimized_rotations: dict[str, int] = {}
    wall_subdivisions: dict[str, int] = {}
    wall_subdivision_positions: dict[str, tuple[tuple[float, float], ...]] = {}
    wall_subdivision_source_ranges: dict[str, tuple[tuple[float, float], ...]] = {}

    for wall in walls:
        wall_scale = _normalize_wall_uv_scale(
            base_wall_uv_scales.get(wall.key, DEFAULT_WALL_UV_SCALE)
            * max(0.01, scale_multiplier)
        )
        rotation_degrees = _normalize_wall_uv_rotation(
            wall_uv_rotations.get(wall.key, DEFAULT_WALL_UV_ROTATION_DEGREES)
        )
        wall_width, wall_height = _calculate_scaled_wall_natural_size(
            wall=wall,
            wall_height_pixels=wall_height_pixels,
            wall_scale=wall_scale,
        )
        remaining_wall_width = wall_width
        consumed_wall_width = 0.0
        segment_positions: list[tuple[float, float]] = []
        segment_source_ranges: list[tuple[float, float]] = []

        while remaining_wall_width > UV_SUBDIVISION_EPSILON:
            available_row_width = map_width - row_x
            if row_x > 0.0 and available_row_width < 1.0:
                row_x = 0.0
                row_y += row_height
                row_height = 0.0
                available_row_width = map_width

            segment_width = _calculate_row_overflow_segment_width(
                remaining_wall_width=remaining_wall_width,
                available_row_width=available_row_width,
                wall_height=wall_height,
                rotation_degrees=rotation_degrees,
            )
            if segment_width is None:
                if row_x <= 0.0:
                    return None

                row_x = 0.0
                row_y += row_height
                row_height = 0.0
                continue

            bounding_width, bounding_height = _calculate_rotated_bounds(
                width=segment_width,
                height=wall_height,
                rotation_degrees=rotation_degrees,
            )
            if bounding_width > map_width or bounding_height > map_height:
                return None

            if row_y + bounding_height > map_height:
                return None

            source_start_ratio = consumed_wall_width / wall_width
            consumed_wall_width = min(wall_width, consumed_wall_width + segment_width)
            source_end_ratio = consumed_wall_width / wall_width
            segment_positions.append((row_x, row_y))
            segment_source_ranges.append((source_start_ratio, source_end_ratio))
            row_x += bounding_width
            row_height = max(row_height, bounding_height)
            remaining_wall_width = wall_width - consumed_wall_width

            if remaining_wall_width > UV_SUBDIVISION_EPSILON:
                row_x = 0.0
                row_y += row_height
                row_height = 0.0

        optimized_scales[wall.key] = wall_scale
        optimized_rotations[wall.key] = rotation_degrees
        wall_subdivisions[wall.key] = len(segment_positions)
        wall_subdivision_positions[wall.key] = tuple(segment_positions)
        wall_subdivision_source_ranges[wall.key] = tuple(segment_source_ranges)

    return UvOptimizationResult(
        wall_uv_rotations=optimized_rotations,
        wall_uv_scales=optimized_scales,
        wall_uv_positions={},
        wall_subdivisions=wall_subdivisions,
        wall_subdivision_positions=wall_subdivision_positions,
        wall_subdivision_source_ranges=wall_subdivision_source_ranges,
    )


def _calculate_row_overflow_segment_width(
    remaining_wall_width: float,
    available_row_width: float,
    wall_height: float,
    rotation_degrees: int,
) -> float | None:
    full_bounding_width, _ = _calculate_rotated_bounds(
        width=remaining_wall_width,
        height=wall_height,
        rotation_degrees=rotation_degrees,
    )
    if full_bounding_width <= available_row_width + UV_SUBDIVISION_EPSILON:
        return remaining_wall_width

    rotation_radians = math.radians(rotation_degrees % 360)
    cosine = abs(math.cos(rotation_radians))
    sine = abs(math.sin(rotation_radians))
    if cosine <= UV_SUBDIVISION_EPSILON:
        return None

    segment_width = (available_row_width - wall_height * sine) / cosine
    if segment_width < 1.0:
        return None

    return min(remaining_wall_width, segment_width)


def _build_current_uv_optimization_result(
    room: RoomData,
    walls: list[RoomWall],
    map_width: float,
    map_height: float,
    wall_height_meters: float,
    wall_uv_scales: dict[str, float],
    wall_uv_rotations: dict[str, int],
) -> UvOptimizationResult | None:
    layout = _build_uv_wall_layout_for_size(
        walls=walls,
        map_width=map_width,
        map_height=map_height,
        wall_height_meters=wall_height_meters,
        wall_uv_scales=wall_uv_scales,
        wall_uv_rotations=wall_uv_rotations,
        wall_uv_positions=room.wall_uv_positions,
    )
    if layout.hidden_wall_count == 0 and len(layout.placements) == len(walls):
        current_result = UvOptimizationResult(
            wall_uv_rotations=dict(wall_uv_rotations),
            wall_uv_scales=dict(wall_uv_scales),
            wall_uv_positions={
                placement.wall.key: (placement.uv_rect[0], placement.uv_rect[1])
                for placement in layout.placements
            },
        )
        if _is_uv_optimization_result_valid(
            result=current_result,
            walls=walls,
            map_width=map_width,
            map_height=map_height,
            wall_height_meters=wall_height_meters,
        ):
            return current_result

    return _build_packed_uv_result_from_settings(
        walls=walls,
        map_width=map_width,
        map_height=map_height,
        wall_height_meters=wall_height_meters,
        wall_uv_scales=wall_uv_scales,
        wall_uv_rotations=wall_uv_rotations,
        map_padding=0.0,
        uv_spacing=0.0,
    )


def _find_max_uniform_scale_uv_result(
    walls: list[RoomWall],
    map_width: float,
    map_height: float,
    wall_height_meters: float,
    wall_uv_rotations: dict[str, int],
) -> UvOptimizationResult | None:
    scale_ceiling = _calculate_uniform_uv_scale_ceiling(
        walls=walls,
        map_width=map_width,
        map_height=map_height,
        wall_height_meters=wall_height_meters,
        wall_uv_rotations=wall_uv_rotations,
    )
    low_scale = 0.01
    best_result = _build_uniform_scale_uv_result(
        walls=walls,
        map_width=map_width,
        map_height=map_height,
        wall_height_meters=wall_height_meters,
        wall_uv_rotations=wall_uv_rotations,
        wall_scale=low_scale,
    )
    if best_result is None:
        return None

    high_scale = max(low_scale, scale_ceiling)
    for _ in range(UV_UNIFORM_SCALE_SEARCH_PASSES):
        next_scale = (low_scale + high_scale) / 2.0
        candidate_result = _build_uniform_scale_uv_result(
            walls=walls,
            map_width=map_width,
            map_height=map_height,
            wall_height_meters=wall_height_meters,
            wall_uv_rotations=wall_uv_rotations,
            wall_scale=next_scale,
        )
        if candidate_result is None:
            high_scale = next_scale
            continue

        best_result = candidate_result
        low_scale = next_scale

    return best_result


def _find_simple_rotation_uniform_scale_uv_result(
    walls: list[RoomWall],
    map_width: float,
    map_height: float,
    wall_height_meters: float,
    fallback_result: UvOptimizationResult,
) -> UvOptimizationResult:
    best_result = fallback_result
    best_score = _score_uv_optimization_result(
        result=best_result,
        walls=walls,
        map_width=map_width,
        map_height=map_height,
        wall_height_meters=wall_height_meters,
    )
    for wall_uv_rotations in _iter_simple_uv_rotation_combinations(walls):
        candidate_result = _find_max_uniform_scale_uv_result(
            walls=walls,
            map_width=map_width,
            map_height=map_height,
            wall_height_meters=wall_height_meters,
            wall_uv_rotations=wall_uv_rotations,
        )
        if candidate_result is None:
            continue

        candidate_score = _score_uv_optimization_result(
            result=candidate_result,
            walls=walls,
            map_width=map_width,
            map_height=map_height,
            wall_height_meters=wall_height_meters,
        )
        if candidate_score > best_score:
            best_result = candidate_result
            best_score = candidate_score

    return best_result


def _find_complex_uv_optimization_result(
    walls: list[RoomWall],
    map_width: float,
    map_height: float,
    wall_height_meters: float,
    initial_result: UvOptimizationResult,
    pass_count: int,
) -> UvOptimizationResult:
    random_generator = random.Random()
    best_result = initial_result
    best_score = _score_complex_uv_optimization_result(
        result=best_result,
        walls=walls,
        map_width=map_width,
        map_height=map_height,
        wall_height_meters=wall_height_meters,
    )
    for pass_index in range(1, max(1, int(pass_count)) + 1):
        pass_result = _try_complex_uv_repack_step(
            walls=walls,
            map_width=map_width,
            map_height=map_height,
            wall_height_meters=wall_height_meters,
            result=initial_result,
            random_generator=random_generator,
        )
        if pass_result is None:
            pass_result = initial_result
        failure_count = 0
        loop_count = 0
        while (
            failure_count < UV_COMPLEX_FAILURE_LIMIT
            and loop_count < UV_COMPLEX_MAX_LOOPS_PER_PASS
        ):
            loop_count += 1
            candidate_result = _try_complex_uv_growth_step(
                walls=walls,
                map_width=map_width,
                map_height=map_height,
                wall_height_meters=wall_height_meters,
                result=pass_result,
                random_generator=random_generator,
            )
            if candidate_result is None:
                failure_count += 1
                continue

            pass_result = candidate_result
            failure_count = 0

        pass_score = _score_complex_uv_optimization_result(
            result=pass_result,
            walls=walls,
            map_width=map_width,
            map_height=map_height,
            wall_height_meters=wall_height_meters,
        )
        if pass_score > best_score:
            best_result = pass_result
            best_score = pass_score

    return best_result


def _try_complex_uv_repack_step(
    walls: list[RoomWall],
    map_width: float,
    map_height: float,
    wall_height_meters: float,
    result: UvOptimizationResult,
    random_generator: random.Random,
) -> UvOptimizationResult | None:
    selected_wall = random_generator.choice(walls)
    remaining_walls = [
        wall
        for wall in walls
        if wall.key != selected_wall.key
    ]
    random_generator.shuffle(remaining_walls)
    return _build_complex_repacked_uv_result(
        walls=(selected_wall, *remaining_walls),
        map_width=map_width,
        map_height=map_height,
        wall_height_meters=wall_height_meters,
        wall_uv_scales=result.wall_uv_scales,
        wall_uv_rotations=result.wall_uv_rotations,
        random_generator=random_generator,
    )


def _score_complex_uv_optimization_result(
    result: UvOptimizationResult,
    walls: list[RoomWall],
    map_width: float,
    map_height: float,
    wall_height_meters: float,
) -> tuple[float, int, int, int, float, float]:
    layout_score = _score_uv_optimization_result(
        result=result,
        walls=walls,
        map_width=map_width,
        map_height=map_height,
        wall_height_meters=wall_height_meters,
    )
    occupied_area, hidden_score, placement_count, bounding_score, used_score = (
        layout_score
    )
    return (
        occupied_area,
        hidden_score,
        placement_count,
        _count_non_cardinal_uv_rotations(result.wall_uv_rotations.values()),
        bounding_score,
        used_score,
    )


def _count_non_cardinal_uv_rotations(rotations: Iterable[int]) -> int:
    return sum(
        1
        for rotation_degrees in rotations
        if _normalize_wall_uv_rotation(rotation_degrees) % 90 != 0
    )


def _calculate_uniform_uv_scale_ceiling(
    walls: list[RoomWall],
    map_width: float,
    map_height: float,
    wall_height_meters: float,
    wall_uv_rotations: dict[str, int],
) -> float:
    wall_height_pixels = max(1.0, wall_height_meters / PIXEL_TO_METER)
    map_area = max(1.0, map_width) * max(1.0, map_height)
    scale_ceiling = float("inf")
    total_area_at_scale_one = 0.0
    for wall in walls:
        wall_width, wall_height = _calculate_scaled_wall_natural_size(
            wall=wall,
            wall_height_pixels=wall_height_pixels,
            wall_scale=1.0,
        )
        total_area_at_scale_one += wall_width * wall_height
        bounding_width, bounding_height = _calculate_rotated_bounds(
            width=wall_width,
            height=wall_height,
            rotation_degrees=wall_uv_rotations.get(
                wall.key,
                DEFAULT_WALL_UV_ROTATION_DEGREES,
            ),
        )
        scale_ceiling = min(
            scale_ceiling,
            map_width / max(1.0, bounding_width),
            map_height / max(1.0, bounding_height),
        )

    if total_area_at_scale_one > 0.0:
        scale_ceiling = min(
            scale_ceiling,
            math.sqrt(map_area / total_area_at_scale_one),
        )
    if not math.isfinite(scale_ceiling):
        return DEFAULT_WALL_UV_SCALE

    return max(0.01, scale_ceiling)


def _build_uniform_scale_uv_result(
    walls: list[RoomWall],
    map_width: float,
    map_height: float,
    wall_height_meters: float,
    wall_uv_rotations: dict[str, int],
    wall_scale: float,
) -> UvOptimizationResult | None:
    uniform_scale = _normalize_wall_uv_scale(wall_scale)
    return _build_packed_uv_result_from_settings(
        walls=walls,
        map_width=map_width,
        map_height=map_height,
        wall_height_meters=wall_height_meters,
        wall_uv_scales={wall.key: uniform_scale for wall in walls},
        wall_uv_rotations=wall_uv_rotations,
        map_padding=0.0,
        uv_spacing=0.0,
    )


def _iter_simple_uv_rotation_combinations(
    walls: list[RoomWall],
) -> Iterator[dict[str, int]]:
    wall_count = len(walls)
    for rotation_mask in range(1 << wall_count):
        rotations = tuple(
            90 if rotation_mask & (1 << wall_index) else 0
            for wall_index in range(wall_count)
        )
        yield {
            wall.key: rotations[wall_index]
            for wall_index, wall in enumerate(walls)
        }


def _try_complex_uv_growth_step(
    walls: list[RoomWall],
    map_width: float,
    map_height: float,
    wall_height_meters: float,
    result: UvOptimizationResult,
    random_generator: random.Random,
) -> UvOptimizationResult | None:
    candidate_scales = {
        wall.key: round(
            _normalize_wall_uv_scale(
                result.wall_uv_scales.get(wall.key, DEFAULT_WALL_UV_SCALE)
            )
            + UV_COMPLEX_SCALE_STEP,
            3,
        )
        for wall in walls
    }
    if _calculate_total_uv_wall_area(
        walls=walls,
        wall_height_meters=wall_height_meters,
        wall_uv_scales=candidate_scales,
    ) > max(1.0, map_width) * max(1.0, map_height):
        return None

    selected_wall = random_generator.choice(walls)
    remaining_walls = [
        wall
        for wall in walls
        if wall.key != selected_wall.key
    ]
    random_generator.shuffle(remaining_walls)
    return _build_complex_repacked_uv_result(
        walls=(selected_wall, *remaining_walls),
        map_width=map_width,
        map_height=map_height,
        wall_height_meters=wall_height_meters,
        wall_uv_scales=candidate_scales,
        wall_uv_rotations=result.wall_uv_rotations,
        random_generator=random_generator,
    )


def _build_complex_repacked_uv_result(
    walls: tuple[RoomWall, ...],
    map_width: float,
    map_height: float,
    wall_height_meters: float,
    wall_uv_scales: dict[str, float],
    wall_uv_rotations: dict[str, int],
    random_generator: random.Random,
) -> UvOptimizationResult | None:
    wall_height_pixels = max(1.0, wall_height_meters / PIXEL_TO_METER)
    placed_rectangles: list[UvFreeRectangle] = []
    wall_uv_positions: dict[str, tuple[float, float]] = {}
    optimized_rotations: dict[str, int] = {}
    for wall in walls:
        placement = _find_complex_wall_placement(
            wall=wall,
            placed_rectangles=placed_rectangles,
            map_width=map_width,
            map_height=map_height,
            wall_height_pixels=wall_height_pixels,
            wall_scale=wall_uv_scales.get(wall.key, DEFAULT_WALL_UV_SCALE),
            current_rotation=wall_uv_rotations.get(
                wall.key,
                DEFAULT_WALL_UV_ROTATION_DEGREES,
            ),
            random_generator=random_generator,
        )
        if placement is None:
            return None

        placement_rectangle, rotation_degrees = placement
        placed_rectangles.append(placement_rectangle)
        wall_uv_positions[wall.key] = (
            placement_rectangle.x,
            placement_rectangle.y,
        )
        optimized_rotations[wall.key] = rotation_degrees

    return UvOptimizationResult(
        wall_uv_rotations=optimized_rotations,
        wall_uv_scales={
            wall.key: _normalize_wall_uv_scale(
                wall_uv_scales.get(wall.key, DEFAULT_WALL_UV_SCALE)
            )
            for wall in walls
        },
        wall_uv_positions=wall_uv_positions,
    )


def _find_complex_wall_placement(
    wall: RoomWall,
    placed_rectangles: list[UvFreeRectangle],
    map_width: float,
    map_height: float,
    wall_height_pixels: float,
    wall_scale: float,
    current_rotation: int,
    random_generator: random.Random,
) -> tuple[UvFreeRectangle, int] | None:
    natural_width, natural_height = _calculate_scaled_wall_natural_size(
        wall=wall,
        wall_height_pixels=wall_height_pixels,
        wall_scale=wall_scale,
    )
    step_x = max(1.0, natural_width / 10.0)
    step_y = max(1.0, natural_height / 10.0)
    max_origin_x = map_width - min(natural_width, natural_height)
    max_origin_y = map_height - min(natural_width, natural_height)
    rotation_candidates = _build_complex_wall_rotation_candidates(
        current_rotation=current_rotation,
        random_generator=random_generator,
    )
    for origin_y in _iter_uv_origin_values(max_origin_y, step_y):
        for origin_x in _iter_uv_origin_values(max_origin_x, step_x):
            for rotation_degrees in rotation_candidates:
                bounding_width, bounding_height = _calculate_rotated_bounds(
                    width=natural_width,
                    height=natural_height,
                    rotation_degrees=rotation_degrees,
                )
                if origin_x + bounding_width > map_width:
                    continue
                if origin_y + bounding_height > map_height:
                    continue

                rectangle = UvFreeRectangle(
                    x=origin_x,
                    y=origin_y,
                    width=bounding_width,
                    height=bounding_height,
                )
                if _does_uv_rectangle_intersect_any(
                    rectangle,
                    placed_rectangles,
                ):
                    continue

                return rectangle, rotation_degrees

    return None


def _build_complex_wall_rotation_candidates(
    current_rotation: int,
    random_generator: random.Random,
) -> tuple[int, ...]:
    random_start = random_generator.randrange(360)
    rotations = [
        (random_start + rotation_offset) % 360
        for rotation_offset in range(UV_COMPLEX_ROTATION_CANDIDATE_COUNT)
    ]
    normalized_rotation = _normalize_wall_uv_rotation(current_rotation)
    rotations.extend(
        (normalized_rotation + rotation_offset) % 360
        for rotation_offset in (0, 90, 180, 270)
    )

    return tuple(dict.fromkeys(rotations))


def _calculate_total_uv_wall_area(
    walls: list[RoomWall],
    wall_height_meters: float,
    wall_uv_scales: dict[str, float],
) -> float:
    wall_height_pixels = max(1.0, wall_height_meters / PIXEL_TO_METER)
    total_area = 0.0
    for wall in walls:
        wall_width, wall_height = _calculate_scaled_wall_natural_size(
            wall=wall,
            wall_height_pixels=wall_height_pixels,
            wall_scale=wall_uv_scales.get(wall.key, DEFAULT_WALL_UV_SCALE),
        )
        total_area += wall_width * wall_height

    return total_area


def _build_uv_rectangle_for_wall(
    wall: RoomWall,
    wall_height_pixels: float,
    wall_scale: float,
    rotation_degrees: int,
    wall_position: tuple[float, float] | None,
) -> UvFreeRectangle | None:
    if wall_position is None:
        return None

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
    return UvFreeRectangle(
        x=wall_position[0],
        y=wall_position[1],
        width=bounding_width,
        height=bounding_height,
    )


def _is_uv_optimization_result_valid(
    result: UvOptimizationResult,
    walls: list[RoomWall],
    map_width: float,
    map_height: float,
    wall_height_meters: float,
) -> bool:
    wall_height_pixels = max(1.0, wall_height_meters / PIXEL_TO_METER)
    rectangles: list[UvFreeRectangle] = []
    for wall in walls:
        rectangle = _build_uv_rectangle_for_wall(
            wall=wall,
            wall_height_pixels=wall_height_pixels,
            wall_scale=result.wall_uv_scales.get(
                wall.key,
                DEFAULT_WALL_UV_SCALE,
            ),
            rotation_degrees=result.wall_uv_rotations.get(
                wall.key,
                DEFAULT_WALL_UV_ROTATION_DEGREES,
            ),
            wall_position=result.wall_uv_positions.get(wall.key),
        )
        if rectangle is None:
            return False
        if not _is_uv_rectangle_inside_map(rectangle, map_width, map_height):
            return False
        if _does_uv_rectangle_intersect_any(rectangle, rectangles):
            return False

        rectangles.append(rectangle)

    return True


def _iter_uv_origin_values(max_origin: float, step: float) -> Iterator[float]:
    if max_origin < 0.0:
        return

    origin = 0.0
    while origin < max_origin:
        yield origin
        origin += max(1.0, step)

    yield max_origin


def _is_uv_rectangle_inside_map(
    rectangle: UvFreeRectangle,
    map_width: float,
    map_height: float,
) -> bool:
    return (
        rectangle.x >= 0.0
        and rectangle.y >= 0.0
        and _get_uv_rectangle_right(rectangle) <= map_width
        and _get_uv_rectangle_bottom(rectangle) <= map_height
    )


def _does_uv_rectangle_intersect_any(
    rectangle: UvFreeRectangle,
    other_rectangles: (
        Iterator[UvFreeRectangle]
        | list[UvFreeRectangle]
        | tuple[UvFreeRectangle, ...]
    ),
) -> bool:
    return any(
        _do_uv_rectangles_intersect(rectangle, other_rectangle)
        for other_rectangle in other_rectangles
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
                wall_uv_positions={},
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
    edge_based_walls = _build_room_walls_from_edges(
        room_vertices=room_vertices,
        vertex_data=vertex_data,
        center_point=center_point,
    )
    if edge_based_walls:
        return edge_based_walls

    ordered_vertices = _order_vertices_around_center(room_vertices, center_point)
    wall_vertices = _remove_straight_through_vertices(ordered_vertices)
    if len(wall_vertices) < 3:
        return []

    walls = [
        _build_room_wall(
            start_vertex=wall_vertices[index],
            end_vertex=wall_vertices[(index + 1) % len(wall_vertices)],
            center_point=center_point,
        )
        for index in range(len(wall_vertices))
    ]
    return _sort_room_walls_around_center(walls, center_point)


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
    wall_uv_positions: dict[str, tuple[float, float]] | None = None,
    wall_subdivisions: dict[str, int] | None = None,
    wall_subdivision_positions: (
        dict[str, tuple[tuple[float, float], ...]] | None
    ) = None,
    wall_subdivision_source_ranges: (
        dict[str, tuple[tuple[float, float], ...]] | None
    ) = None,
) -> UvLayout:
    map_width = max(1.0, map_width)
    map_height = max(1.0, map_height)
    wall_height_pixels = max(1.0, wall_height_meters / PIXEL_TO_METER)
    placements: list[UvWallPlacement] = []
    hidden_wall_count = 0
    free_rectangles = _build_initial_uv_free_rectangles(
        map_width=map_width,
        map_height=map_height,
    )

    for wall in walls:
        wall_scale = _normalize_wall_uv_scale(
            wall_uv_scales.get(wall.key, DEFAULT_WALL_UV_SCALE)
        )
        wall_rotation = _normalize_wall_uv_rotation(
            wall_uv_rotations.get(wall.key, DEFAULT_WALL_UV_ROTATION_DEGREES)
        )
        wall_has_subdivision_layout = (
            wall_subdivisions is not None and wall.key in wall_subdivisions
        )
        wall_position = None
        if wall_uv_positions is not None and not wall_has_subdivision_layout:
            wall_position = wall_uv_positions.get(wall.key)

        wall_width, wall_height = _calculate_scaled_wall_natural_size(
            wall=wall,
            wall_height_pixels=wall_height_pixels,
            wall_scale=wall_scale,
        )
        if wall_has_subdivision_layout:
            segment_count = _normalize_wall_subdivision_count(
                wall_subdivisions.get(wall.key, 1) if wall_subdivisions else 1
            )
            segment_positions = None
            if wall_subdivision_positions is not None:
                segment_positions = wall_subdivision_positions.get(wall.key)
            segment_source_ranges = None
            if wall_subdivision_source_ranges is not None:
                segment_source_ranges = wall_subdivision_source_ranges.get(wall.key)

            placement_option = _build_wall_segment_placement_option(
                wall=wall,
                wall_width=wall_width,
                wall_height=wall_height,
                rotation_degrees=wall_rotation,
                map_width=map_width,
                map_height=map_height,
                free_rectangles=free_rectangles,
                segment_count=segment_count,
                segment_positions=segment_positions,
                segment_source_ranges=segment_source_ranges,
            )
            if placement_option is None:
                hidden_wall_count += 1
                continue

            wall_placements, free_rectangles = placement_option
            placements.extend(wall_placements)
            continue

        bounding_width, bounding_height = _calculate_rotated_bounds(
            width=wall_width,
            height=wall_height,
            rotation_degrees=wall_rotation,
        )
        if wall_position is None:
            placement_option = _find_best_uv_free_placement(
                free_rectangles=free_rectangles,
                width=bounding_width,
                height=bounding_height,
            )
        else:
            placement_option = _build_manual_uv_placement_option(
                free_rectangles=free_rectangles,
                map_width=map_width,
                map_height=map_height,
                wall_position=wall_position,
                width=bounding_width,
                height=bounding_height,
            )

        if placement_option is None:
            hidden_wall_count += 1
            continue

        placement_rect, free_rectangles = placement_option
        placements.append(
            UvWallPlacement(
                wall=wall,
                uv_rect=(
                    placement_rect.x,
                    placement_rect.y,
                    placement_rect.width,
                    placement_rect.height,
                ),
                natural_size=(wall_width, wall_height),
                rotation_degrees=wall_rotation,
            )
        )

    return UvLayout(placements=placements, hidden_wall_count=hidden_wall_count)


def _build_wall_segment_placement_option(
    wall: RoomWall,
    wall_width: float,
    wall_height: float,
    rotation_degrees: int,
    map_width: float,
    map_height: float,
    free_rectangles: tuple[UvFreeRectangle, ...],
    segment_count: int,
    segment_positions: tuple[tuple[float, float], ...] | None,
    segment_source_ranges: tuple[tuple[float, float], ...] | None,
) -> tuple[list[UvWallPlacement], tuple[UvFreeRectangle, ...]] | None:
    segment_count = _normalize_wall_subdivision_count(segment_count)
    if segment_positions is not None and len(segment_positions) != segment_count:
        segment_positions = None
    if (
        segment_source_ranges is not None
        and len(segment_source_ranges) != segment_count
    ):
        segment_source_ranges = None
    if segment_source_ranges is None:
        segment_source_ranges = tuple(
            (segment_index / segment_count, (segment_index + 1) / segment_count)
            for segment_index in range(segment_count)
        )

    next_free_rectangles = free_rectangles
    placements: list[UvWallPlacement] = []
    for segment_index in range(segment_count):
        source_start_ratio, source_end_ratio = _normalize_uv_source_range(
            segment_source_ranges[segment_index]
        )
        segment_width = max(
            1.0,
            wall_width * (source_end_ratio - source_start_ratio),
        )
        bounding_width, bounding_height = _calculate_rotated_bounds(
            width=segment_width,
            height=wall_height,
            rotation_degrees=rotation_degrees,
        )
        segment_position = (
            segment_positions[segment_index]
            if segment_positions is not None
            else None
        )
        if segment_position is None:
            placement_option = _find_best_uv_free_placement(
                free_rectangles=next_free_rectangles,
                width=bounding_width,
                height=bounding_height,
            )
        else:
            placement_option = _build_manual_uv_placement_option(
                free_rectangles=next_free_rectangles,
                map_width=map_width,
                map_height=map_height,
                wall_position=segment_position,
                width=bounding_width,
                height=bounding_height,
            )

        if placement_option is None:
            return None

        placement_rect, next_free_rectangles = placement_option
        placements.append(
            UvWallPlacement(
                wall=wall,
                uv_rect=(
                    placement_rect.x,
                    placement_rect.y,
                    placement_rect.width,
                    placement_rect.height,
                ),
                natural_size=(segment_width, wall_height),
                rotation_degrees=rotation_degrees,
                segment_index=segment_index,
                segment_count=segment_count,
                source_start_ratio=source_start_ratio,
                source_end_ratio=source_end_ratio,
            )
        )

    return placements, next_free_rectangles


def _build_packed_uv_result_from_settings(
    walls: list[RoomWall],
    map_width: float,
    map_height: float,
    wall_height_meters: float,
    wall_uv_scales: dict[str, float],
    wall_uv_rotations: dict[str, int],
    map_padding: float,
    uv_spacing: float,
) -> UvOptimizationResult | None:
    wall_height_pixels = max(1.0, wall_height_meters / PIXEL_TO_METER)
    free_rectangles = _build_initial_uv_free_rectangles(
        map_width=map_width,
        map_height=map_height,
        map_padding=map_padding,
    )
    wall_uv_positions: dict[str, tuple[float, float]] = {}
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
        placement_option = _find_best_uv_free_placement(
            free_rectangles=free_rectangles,
            width=bounding_width,
            height=bounding_height,
            uv_spacing=uv_spacing,
        )
        if placement_option is None:
            return None

        placement_rect, free_rectangles = placement_option
        wall_uv_positions[wall.key] = (placement_rect.x, placement_rect.y)

    return UvOptimizationResult(
        wall_uv_rotations={
            wall.key: _normalize_wall_uv_rotation(
                wall_uv_rotations.get(
                    wall.key,
                    DEFAULT_WALL_UV_ROTATION_DEGREES,
                )
            )
            for wall in walls
        },
        wall_uv_scales={
            wall.key: _normalize_wall_uv_scale(
                wall_uv_scales.get(wall.key, DEFAULT_WALL_UV_SCALE)
            )
            for wall in walls
        },
        wall_uv_positions=wall_uv_positions,
    )


def _build_initial_uv_free_rectangles(
    map_width: float,
    map_height: float,
    map_padding: float = UV_MAP_PADDING,
) -> tuple[UvFreeRectangle, ...]:
    map_padding = max(0.0, map_padding)
    free_width = map_width - map_padding * 2.0
    free_height = map_height - map_padding * 2.0
    if free_width <= 0.0 or free_height <= 0.0:
        return ()

    return (
        UvFreeRectangle(
            x=map_padding,
            y=map_padding,
            width=free_width,
            height=free_height,
        ),
    )


def _find_best_uv_free_placement(
    free_rectangles: tuple[UvFreeRectangle, ...],
    width: float,
    height: float,
    uv_spacing: float = UV_MAP_PADDING,
) -> tuple[UvFreeRectangle, tuple[UvFreeRectangle, ...]] | None:
    placement_options = _iter_uv_free_placement_options(
        free_rectangles=free_rectangles,
        width=width,
        height=height,
        uv_spacing=uv_spacing,
    )
    return max(
        placement_options,
        key=_score_uv_free_placement_option,
        default=None,
    )


def _build_manual_uv_placement_option(
    free_rectangles: tuple[UvFreeRectangle, ...],
    map_width: float,
    map_height: float,
    wall_position: tuple[float, float],
    width: float,
    height: float,
) -> tuple[UvFreeRectangle, tuple[UvFreeRectangle, ...]] | None:
    if width > map_width or height > map_height:
        return None

    placement_rectangle = UvFreeRectangle(
        x=_clamp_float(wall_position[0], 0.0, map_width - width),
        y=_clamp_float(wall_position[1], 0.0, map_height - height),
        width=width,
        height=height,
    )
    return (
        placement_rectangle,
        _split_uv_free_rectangles(
            free_rectangles=free_rectangles,
            used_rectangle=_build_padded_uv_used_rectangle(placement_rectangle),
        ),
    )


def _iter_uv_free_placement_options(
    free_rectangles: tuple[UvFreeRectangle, ...],
    width: float,
    height: float,
    uv_spacing: float = UV_MAP_PADDING,
) -> Iterator[tuple[UvFreeRectangle, tuple[UvFreeRectangle, ...]]]:
    for free_rectangle in free_rectangles:
        if width > free_rectangle.width or height > free_rectangle.height:
            continue

        placement_rectangle = UvFreeRectangle(
            x=free_rectangle.x,
            y=free_rectangle.y,
            width=width,
            height=height,
        )
        yield (
            placement_rectangle,
            _split_uv_free_rectangles(
                free_rectangles=free_rectangles,
                used_rectangle=_build_padded_uv_used_rectangle(
                    placement_rectangle,
                    uv_spacing=uv_spacing,
                ),
            ),
        )


def _score_uv_free_placement_option(
    placement_option: tuple[UvFreeRectangle, tuple[UvFreeRectangle, ...]],
) -> tuple[float, float, int, float, float]:
    placement_rectangle, free_rectangles = placement_option
    return (
        _get_largest_uv_free_rectangle_area(free_rectangles),
        sum(_get_uv_rectangle_area(rectangle) for rectangle in free_rectangles),
        -len(free_rectangles),
        -placement_rectangle.y,
        -placement_rectangle.x,
    )


def _split_uv_free_rectangles(
    free_rectangles: tuple[UvFreeRectangle, ...],
    used_rectangle: UvFreeRectangle,
) -> tuple[UvFreeRectangle, ...]:
    split_rectangles: list[UvFreeRectangle] = []
    for free_rectangle in free_rectangles:
        if not _do_uv_rectangles_intersect(free_rectangle, used_rectangle):
            split_rectangles.append(free_rectangle)
            continue

        split_rectangles.extend(
            _split_uv_free_rectangle(
                free_rectangle=free_rectangle,
                used_rectangle=used_rectangle,
            )
        )

    return _prune_uv_free_rectangles(split_rectangles)


def _split_uv_free_rectangle(
    free_rectangle: UvFreeRectangle,
    used_rectangle: UvFreeRectangle,
) -> list[UvFreeRectangle]:
    free_right = _get_uv_rectangle_right(free_rectangle)
    free_bottom = _get_uv_rectangle_bottom(free_rectangle)
    used_right = _get_uv_rectangle_right(used_rectangle)
    used_bottom = _get_uv_rectangle_bottom(used_rectangle)
    overlap_top = max(free_rectangle.y, used_rectangle.y)
    overlap_bottom = min(free_bottom, used_bottom)
    split_rectangles: list[UvFreeRectangle] = []

    split_rectangles.append(
        UvFreeRectangle(
            x=free_rectangle.x,
            y=free_rectangle.y,
            width=free_rectangle.width,
            height=used_rectangle.y - free_rectangle.y,
        )
    )
    split_rectangles.append(
        UvFreeRectangle(
            x=free_rectangle.x,
            y=used_bottom,
            width=free_rectangle.width,
            height=free_bottom - used_bottom,
        )
    )
    split_rectangles.append(
        UvFreeRectangle(
            x=free_rectangle.x,
            y=overlap_top,
            width=used_rectangle.x - free_rectangle.x,
            height=overlap_bottom - overlap_top,
        )
    )
    split_rectangles.append(
        UvFreeRectangle(
            x=used_right,
            y=overlap_top,
            width=free_right - used_right,
            height=overlap_bottom - overlap_top,
        )
    )
    return [
        rectangle
        for rectangle in split_rectangles
        if rectangle.width > 0.0 and rectangle.height > 0.0
    ]


def _prune_uv_free_rectangles(
    free_rectangles: list[UvFreeRectangle],
) -> tuple[UvFreeRectangle, ...]:
    pruned_rectangles: list[UvFreeRectangle] = []
    for rectangle in sorted(
        free_rectangles,
        key=lambda item: (_get_uv_rectangle_area(item), item.y, item.x),
        reverse=True,
    ):
        if any(
            _is_uv_rectangle_contained(rectangle, pruned_rectangle)
            for pruned_rectangle in pruned_rectangles
        ):
            continue

        pruned_rectangles = [
            pruned_rectangle
            for pruned_rectangle in pruned_rectangles
            if not _is_uv_rectangle_contained(pruned_rectangle, rectangle)
        ]
        pruned_rectangles.append(rectangle)

    return tuple(
        sorted(
            pruned_rectangles,
            key=lambda rectangle: (rectangle.y, rectangle.x),
        )
    )


def _build_padded_uv_used_rectangle(
    rectangle: UvFreeRectangle,
    uv_spacing: float = UV_MAP_PADDING,
) -> UvFreeRectangle:
    uv_spacing = max(0.0, uv_spacing)
    return UvFreeRectangle(
        x=rectangle.x,
        y=rectangle.y,
        width=rectangle.width + uv_spacing,
        height=rectangle.height + uv_spacing,
    )


def _do_uv_rectangles_intersect(
    first_rectangle: UvFreeRectangle,
    second_rectangle: UvFreeRectangle,
) -> bool:
    return (
        first_rectangle.x < _get_uv_rectangle_right(second_rectangle)
        and _get_uv_rectangle_right(first_rectangle) > second_rectangle.x
        and first_rectangle.y < _get_uv_rectangle_bottom(second_rectangle)
        and _get_uv_rectangle_bottom(first_rectangle) > second_rectangle.y
    )


def _is_uv_rectangle_contained(
    inner_rectangle: UvFreeRectangle,
    outer_rectangle: UvFreeRectangle,
) -> bool:
    return (
        inner_rectangle.x >= outer_rectangle.x
        and inner_rectangle.y >= outer_rectangle.y
        and _get_uv_rectangle_right(inner_rectangle)
        <= _get_uv_rectangle_right(outer_rectangle)
        and _get_uv_rectangle_bottom(inner_rectangle)
        <= _get_uv_rectangle_bottom(outer_rectangle)
    )


def _get_largest_uv_free_rectangle_area(
    free_rectangles: tuple[UvFreeRectangle, ...],
) -> float:
    return max(
        (_get_uv_rectangle_area(rectangle) for rectangle in free_rectangles),
        default=0.0,
    )


def _get_uv_rectangle_area(rectangle: UvFreeRectangle) -> float:
    return rectangle.width * rectangle.height


def _get_uv_rectangle_right(rectangle: UvFreeRectangle) -> float:
    return rectangle.x + rectangle.width


def _get_uv_rectangle_bottom(rectangle: UvFreeRectangle) -> float:
    return rectangle.y + rectangle.height


def _calculate_scaled_wall_natural_size(
    wall: RoomWall,
    wall_height_pixels: float,
    wall_scale: float,
) -> tuple[float, float]:
    return (
        max(1.0, wall.length * wall_scale),
        max(1.0, wall_height_pixels * wall_scale),
    )


def _normalize_wall_subdivision_count(segment_count: int | float | object) -> int:
    try:
        normalized_count = int(segment_count)
    except (TypeError, ValueError):
        return 1

    return max(1, normalized_count)


def _normalize_uv_source_range(
    source_range: tuple[float, float] | list[float],
) -> tuple[float, float]:
    source_start = min(max(0.0, float(source_range[0])), 1.0)
    source_end = min(max(source_start, float(source_range[1])), 1.0)
    if source_end <= source_start:
        source_end = min(1.0, source_start + UV_SUBDIVISION_EPSILON)

    return source_start, source_end


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
    layout_right = max(
        (
            placement.uv_rect[0] + placement.uv_rect[2]
            for placement in layout.placements
        ),
        default=UV_MAP_PADDING,
    )
    layout_bottom = max(
        (
            placement.uv_rect[1] + placement.uv_rect[3]
            for placement in layout.placements
        ),
        default=UV_MAP_PADDING,
    )
    used_area = max(0.0, layout_right - UV_MAP_PADDING) * max(
        0.0,
        layout_bottom - UV_MAP_PADDING,
    )
    return (
        occupied_area,
        -layout.hidden_wall_count,
        len(layout.placements),
        -bounding_area,
        -used_area,
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
        wall_uv_positions=result.wall_uv_positions,
        wall_subdivisions=result.wall_subdivisions,
        wall_subdivision_positions=result.wall_subdivision_positions,
        wall_subdivision_source_ranges=result.wall_subdivision_source_ranges,
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


def _build_room_walls_from_edges(
    room_vertices: list[Vertex],
    vertex_data: VertexData,
    center_point: tuple[float, float],
) -> list[RoomWall]:
    room_vertices_by_id = {vertex.id: vertex for vertex in room_vertices}
    room_vertex_ids = set(room_vertices_by_id)
    adjacency = {
        vertex_id: set()
        for vertex_id in room_vertex_ids
    }
    unvisited_edges: set[tuple[int, int]] = set()
    for edge in vertex_data.edges:
        if edge.start_vertex_id not in room_vertex_ids:
            continue
        if edge.end_vertex_id not in room_vertex_ids:
            continue

        edge_key = _build_edge_key(edge.start_vertex_id, edge.end_vertex_id)
        if edge_key in unvisited_edges:
            continue

        unvisited_edges.add(edge_key)
        adjacency[edge.start_vertex_id].add(edge.end_vertex_id)
        adjacency[edge.end_vertex_id].add(edge.start_vertex_id)

    if not unvisited_edges:
        return []

    walls: list[RoomWall] = []
    while unvisited_edges:
        start_vertex_id, end_vertex_id = next(iter(unvisited_edges))
        unvisited_edges.remove((start_vertex_id, end_vertex_id))
        wall_chain = [start_vertex_id, end_vertex_id]
        _extend_room_wall_chain(
            wall_chain=wall_chain,
            adjacency=adjacency,
            unvisited_edges=unvisited_edges,
            vertices_by_id=room_vertices_by_id,
            forward=True,
        )
        _extend_room_wall_chain(
            wall_chain=wall_chain,
            adjacency=adjacency,
            unvisited_edges=unvisited_edges,
            vertices_by_id=room_vertices_by_id,
            forward=False,
        )

        start_vertex = room_vertices_by_id[wall_chain[0]]
        end_vertex = room_vertices_by_id[wall_chain[-1]]
        if _get_vertex_distance(start_vertex, end_vertex) <= STRAIGHT_WALL_TOLERANCE:
            continue

        walls.append(
            _build_room_wall(
                start_vertex=start_vertex,
                end_vertex=end_vertex,
                center_point=center_point,
            )
        )

    return _sort_room_walls_around_center(walls, center_point)


def _extend_room_wall_chain(
    wall_chain: list[int],
    adjacency: dict[int, set[int]],
    unvisited_edges: set[tuple[int, int]],
    vertices_by_id: dict[int, Vertex],
    forward: bool,
) -> None:
    while len(wall_chain) >= 2:
        endpoint_index = -1 if forward else 0
        previous_index = -2 if forward else 1
        endpoint_id = wall_chain[endpoint_index]
        previous_id = wall_chain[previous_index]
        next_id = _find_straight_wall_chain_candidate(
            previous_id=previous_id,
            endpoint_id=endpoint_id,
            adjacency=adjacency,
            unvisited_edges=unvisited_edges,
            vertices_by_id=vertices_by_id,
        )
        if next_id is None:
            return

        unvisited_edges.remove(_build_edge_key(endpoint_id, next_id))
        if forward:
            wall_chain.append(next_id)
        else:
            wall_chain.insert(0, next_id)


def _find_straight_wall_chain_candidate(
    previous_id: int,
    endpoint_id: int,
    adjacency: dict[int, set[int]],
    unvisited_edges: set[tuple[int, int]],
    vertices_by_id: dict[int, Vertex],
) -> int | None:
    candidates = [
        next_id
        for next_id in adjacency[endpoint_id]
        if _build_edge_key(endpoint_id, next_id) in unvisited_edges
        and _is_straight_through(
            vertices_by_id[previous_id],
            vertices_by_id[endpoint_id],
            vertices_by_id[next_id],
        )
    ]
    if len(candidates) != 1:
        return None

    return candidates[0]


def _sort_room_walls_around_center(
    walls: list[RoomWall],
    center_point: tuple[float, float],
) -> list[RoomWall]:
    center_x, center_y = center_point
    return sorted(
        walls,
        key=lambda wall: math.atan2(
            (wall.start_point[1] + wall.end_point[1]) / 2.0 - center_y,
            (wall.start_point[0] + wall.end_point[0]) / 2.0 - center_x,
        ),
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
    start_vertex, end_vertex = _orient_wall_vertices_toward_room_center(
        start_vertex=start_vertex,
        end_vertex=end_vertex,
        center_point=center_point,
    )
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


def _orient_wall_vertices_toward_room_center(
    start_vertex: Vertex,
    end_vertex: Vertex,
    center_point: tuple[float, float],
) -> tuple[Vertex, Vertex]:
    wall_midpoint = (
        (start_vertex.x + end_vertex.x) / 2.0,
        (start_vertex.y + end_vertex.y) / 2.0,
    )
    wall_delta = (
        end_vertex.x - start_vertex.x,
        end_vertex.y - start_vertex.y,
    )
    center_delta = (
        center_point[0] - wall_midpoint[0],
        center_point[1] - wall_midpoint[1],
    )
    room_side_cross = (
        wall_delta[0] * center_delta[1]
        - wall_delta[1] * center_delta[0]
    )
    if room_side_cross < -STRAIGHT_WALL_TOLERANCE:
        return end_vertex, start_vertex

    return start_vertex, end_vertex


def _build_wall_key(start_vertex_id: int, end_vertex_id: int) -> str:
    return f"{min(start_vertex_id, end_vertex_id)}:{max(start_vertex_id, end_vertex_id)}"


def _build_edge_key(start_vertex_id: int, end_vertex_id: int) -> tuple[int, int]:
    return tuple(sorted((start_vertex_id, end_vertex_id)))


def _get_vertex_distance(first_vertex: Vertex, second_vertex: Vertex) -> float:
    return math.hypot(first_vertex.x - second_vertex.x, first_vertex.y - second_vertex.y)


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


def _clamp_float(value: float, minimum: float, maximum: float) -> float:
    if maximum < minimum:
        return minimum

    return min(max(float(value), minimum), maximum)
