# ### Imports ###
from __future__ import annotations

import math

from housemaker.models import (
    DEFAULT_DOORWAY_ARCH_AMOUNT,
    DOORWAY_SHAPE_ARCH,
    DOORWAY_SHAPE_RECTANGULAR,
    normalize_doorway_arch_amount,
    normalize_doorway_shape,
)


# ### Constants ###
DOORWAY_ARCH_SEGMENT_COUNT = 64
DOORWAY_PROFILE_EPSILON = 1e-9


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
