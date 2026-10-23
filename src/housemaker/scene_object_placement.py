"""Compact, footprint-aware placement for a batch of generated objects."""

# ### Imports ###
from __future__ import annotations

import math
from collections.abc import Sequence

# ### Constants ###
DEFAULT_OBJECT_PLACEMENT_GAP_METERS = 0.25


# ### Public placement helpers ###
def build_object_placement_grid_offsets(
    footprint_bounds: Sequence[tuple[float, float, float, float]],
    *,
    gap_meters: float = DEFAULT_OBJECT_PLACEMENT_GAP_METERS,
) -> tuple[tuple[float, float, float], ...]:
    """Center physical footprints in a row-major grid around the batch anchor.

    Bounds are (minimum X/Y, maximum X/Y), relative to each object's placement
    anchor after rotation, mirroring and physical scaling. Per-column widths
    and per-row depths keep differently sized objects separated. The returned
    offsets remain physical metres; source-level conversion belongs to callers.
    """

    gap = _finite_number(gap_meters, "Object placement gap")
    if gap <= 0.0:
        raise ValueError("Object placement gap must be greater than zero.")
    if isinstance(footprint_bounds, (str, bytes, bytearray)):
        raise TypeError("Object footprints must be a collection of bounds.")
    try:
        footprints = tuple(_normalize_bounds(bounds) for bounds in footprint_bounds)
    except TypeError as error:
        raise TypeError("Object footprints must be a collection of bounds.") from error
    if not footprints:
        raise ValueError("Object placement requires at least one footprint.")

    count = len(footprints)
    columns = math.isqrt(count)
    if columns * columns < count:
        columns += 1
    rows = (count + columns - 1) // columns
    widths = [0.0] * columns
    depths = [0.0] * rows
    for index, bounds in enumerate(footprints):
        column = index % columns
        row = index // columns
        widths[column] = max(widths[column], bounds[2] - bounds[0])
        depths[row] = max(depths[row], bounds[3] - bounds[1])
    column_centers = _centered_cell_centers(widths, gap)
    row_centers = _centered_cell_centers(depths, gap)

    offsets: list[tuple[float, float, float]] = []
    for index, bounds in enumerate(footprints):
        own_center_x = bounds[0] + (bounds[2] - bounds[0]) / 2.0
        own_center_y = bounds[1] + (bounds[3] - bounds[1]) / 2.0
        offset_x = column_centers[index % columns] - own_center_x
        offset_y = row_centers[index // columns] - own_center_y
        if not math.isfinite(offset_x) or not math.isfinite(offset_y):
            raise ValueError("Object placement offsets must be finite.")
        offsets.append((offset_x, offset_y, 0.0))
    return tuple(offsets)


# ### Validation and grid calculations ###
def _finite_number(value: object, description: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{description} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{description} must be a finite number.") from error
    if not math.isfinite(number):
        raise ValueError(f"{description} must be a finite number.")
    return number


def _normalize_bounds(bounds: object) -> tuple[float, float, float, float]:
    if isinstance(bounds, (str, bytes, bytearray)):
        raise TypeError("Object footprint bounds must contain four numbers.")
    values = tuple(bounds)
    if len(values) != 4:
        raise ValueError("Object footprint bounds must contain four numbers.")
    minimum_x, minimum_y, maximum_x, maximum_y = (
        _finite_number(value, "Object footprint bound") for value in values
    )
    width = maximum_x - minimum_x
    depth = maximum_y - minimum_y
    if not math.isfinite(width) or not math.isfinite(depth):
        raise ValueError("Object footprint extents must be finite.")
    if width < 0.0 or depth < 0.0:
        raise ValueError("Object footprint maximum must not precede its minimum.")
    return minimum_x, minimum_y, maximum_x, maximum_y


def _centered_cell_centers(extents: Sequence[float], gap: float) -> tuple[float, ...]:
    try:
        total = math.fsum((*extents, gap * (len(extents) - 1)))
    except OverflowError as error:
        raise ValueError("Object placement grid extents must be finite.") from error
    if not math.isfinite(total):
        raise ValueError("Object placement grid extents must be finite.")
    cursor = -total / 2.0
    centers: list[float] = []
    for extent in extents:
        centers.append(cursor + extent / 2.0)
        cursor += extent + gap
    return tuple(centers)
