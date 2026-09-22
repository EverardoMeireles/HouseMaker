# ### Imports ###
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from housemaker.models import (
    DEFAULT_DOORWAY_ARCH_AMOUNT,
    DOORWAY_SHAPE_ARCH,
    DOORWAY_SHAPE_RECTANGULAR,
    PIXEL_TO_METER,
    DoorwayData,
    Edge,
    VertexData,
    normalize_doorway_arch_amount,
    normalize_doorway_shape,
)

# ### Constants ###
DOORWAY_ARCH_SEGMENT_COUNT = 64
DOORWAY_PROFILE_EPSILON = 1e-9
DOORWAY_WALL_ALIGNMENT_MINIMUM = 0.95


# ### Wall ownership helpers ###
def doorway_indices_on_removed_wall_edges(
    vertex_data: VertexData,
    doorways: Sequence[DoorwayData],
    removed_edges: Iterable[Edge],
) -> set[int]:
    """Find openings supported by removed wall edges in image coordinates.

    Doorways predate stable wall IDs, so ownership is determined by overlap
    with their oriented width/depth footprint. Perpendicular crossing walls
    and parallel walls outside that footprint are not owners.
    """

    edge_points: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for edge in removed_edges:
        start = vertex_data.get_vertex(edge.start_vertex_id)
        end = vertex_data.get_vertex(edge.end_vertex_id)
        if start is not None and end is not None:
            edge_points.append(((start.x, start.y), (end.x, end.y)))

    return {
        index
        for index, doorway in enumerate(doorways)
        if any(
            _wall_edge_overlaps_doorway(doorway, start, end)
            for start, end in edge_points
        )
    }


def _wall_edge_overlaps_doorway(
    doorway: DoorwayData,
    start: tuple[float, float],
    end: tuple[float, float],
) -> bool:
    angle = math.radians(doorway.rotation_degrees)
    depth_axis = (math.cos(angle), math.sin(angle))
    width_axis = (-depth_axis[1], depth_axis[0])
    delta = (end[0] - start[0], end[1] - start[1])
    edge_length = math.hypot(*delta)
    if edge_length <= DOORWAY_PROFILE_EPSILON:
        return False
    alignment = abs(
        (delta[0] * width_axis[0] + delta[1] * width_axis[1])
        / edge_length
    )
    if alignment < DOORWAY_WALL_ALIGNMENT_MINIMUM:
        return False

    local_points = tuple(
        (
            offset_x * width_axis[0] + offset_y * width_axis[1],
            offset_x * depth_axis[0] + offset_y * depth_axis[1],
        )
        for offset_x, offset_y in (
            (start[0] - doorway.center_x, start[1] - doorway.center_y),
            (end[0] - doorway.center_x, end[1] - doorway.center_y),
        )
    )
    half_width = doorway.width_meters / PIXEL_TO_METER / 2.0
    half_depth = doorway.depth_meters / PIXEL_TO_METER / 2.0
    width_overlap = min(max(point[0] for point in local_points), half_width) - max(
        min(point[0] for point in local_points), -half_width
    )
    depth_overlap = min(max(point[1] for point in local_points), half_depth) - max(
        min(point[1] for point in local_points), -half_depth
    )
    return width_overlap > DOORWAY_PROFILE_EPSILON and depth_overlap >= -1e-6


# ### Public profile helpers ###
def build_doorway_cross_section_outline(
    width_meters: float,
    height_meters: float,
    shape: str,
    arch_amount: float = DEFAULT_DOORWAY_ARCH_AMOUNT,
) -> tuple[tuple[float, float], ...]:
    """Return one closed, smooth polygonal doorway profile.

    Points contain doorway-local width and height in meters. An arch amount of
    one uses the largest semicircular rise that fits the doorway. Smaller
    positive values flatten that semi-ellipse while preserving the requested
    total doorway height. Zero deliberately has the exact rectangular profile.
    """

    width = _normalize_positive_measurement(width_meters, "width")
    height = _normalize_positive_measurement(height_meters, "height")
    normalized_shape = normalize_doorway_shape(shape)
    normalized_arch_amount = normalize_doorway_arch_amount(arch_amount)
    half_width = width / 2.0
    if (
        normalized_shape == DOORWAY_SHAPE_RECTANGULAR
        or normalized_arch_amount <= DOORWAY_PROFILE_EPSILON
    ):
        return _build_rectangular_outline(half_width, height)

    if normalized_shape != DOORWAY_SHAPE_ARCH:
        raise ValueError(f"Unsupported doorway shape: {normalized_shape!r}.")

    arch_rise = min(half_width, height) * normalized_arch_amount
    if arch_rise <= DOORWAY_PROFILE_EPSILON:
        return _build_rectangular_outline(half_width, height)
    spring_height = height - arch_rise
    points: list[tuple[float, float]] = [
        (-half_width, 0.0),
        (half_width, 0.0),
        (half_width, spring_height),
    ]
    for segment_index in range(1, DOORWAY_ARCH_SEGMENT_COUNT + 1):
        angle = math.pi * segment_index / DOORWAY_ARCH_SEGMENT_COUNT
        points.append(
            (
                half_width * math.cos(angle),
                spring_height + arch_rise * math.sin(angle),
            )
        )
    return _close_profile_without_duplicate_vertices(points)


# ### Profile construction helpers ###
def _build_rectangular_outline(
    half_width_meters: float,
    height_meters: float,
) -> tuple[tuple[float, float], ...]:
    return (
        (-half_width_meters, 0.0),
        (half_width_meters, 0.0),
        (half_width_meters, height_meters),
        (-half_width_meters, height_meters),
        (-half_width_meters, 0.0),
    )


def _close_profile_without_duplicate_vertices(
    points: list[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    """Remove zero-length edges and retain exactly one closing point."""

    clean_points: list[tuple[float, float]] = []
    for point in points:
        if (
            clean_points
            and math.isclose(
                point[0],
                clean_points[-1][0],
                abs_tol=DOORWAY_PROFILE_EPSILON,
            )
            and math.isclose(
                point[1],
                clean_points[-1][1],
                abs_tol=DOORWAY_PROFILE_EPSILON,
            )
        ):
            continue
        clean_points.append(point)
    if len(clean_points) > 1 and (
        math.isclose(
            clean_points[-1][0],
            clean_points[0][0],
            abs_tol=DOORWAY_PROFILE_EPSILON,
        )
        and math.isclose(
            clean_points[-1][1],
            clean_points[0][1],
            abs_tol=DOORWAY_PROFILE_EPSILON,
        )
    ):
        clean_points.pop()
    clean_points.append(clean_points[0])
    return tuple(clean_points)


# ### Validation helpers ###
def _normalize_positive_measurement(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Doorway {field_name} must be a finite positive number.")
    try:
        measurement = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"Doorway {field_name} must be a finite positive number."
        ) from error
    if not math.isfinite(measurement) or measurement <= 0.0:
        raise ValueError(f"Doorway {field_name} must be a finite positive number.")
    return measurement
