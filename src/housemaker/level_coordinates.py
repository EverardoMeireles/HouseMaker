# ### Imports ###
from __future__ import annotations

import math
from collections.abc import Sequence

from housemaker.models import GROUND_LEVEL_INDEX, PIXEL_TO_METER, LevelData


# ### Public coordinate helpers ###
def build_level_base_z_lookup(
    levels: Sequence[LevelData],
) -> dict[int, float]:
    """Return each level floor's absolute Z coordinate in meters."""

    level_lookup = {level.index: level for level in levels}
    base_z_by_index: dict[int, float] = {}
    for level in levels:
        if level.index >= GROUND_LEVEL_INDEX:
            base_z_by_index[level.index] = sum(
                level_lookup[index].height_meters
                for index in range(GROUND_LEVEL_INDEX, level.index)
                if index in level_lookup
            )
        else:
            base_z_by_index[level.index] = -sum(
                level_lookup[index].height_meters
                for index in range(level.index, GROUND_LEVEL_INDEX)
                if index in level_lookup
            )
    return base_z_by_index


def get_level_world_pivot(level: LevelData) -> tuple[float, float]:
    """Return the unscaled world-space center of a level's vertex bounds."""

    if not level.vertex_data.vertices:
        return 0.0, 0.0

    raw_points = [
        _image_to_unscaled_world_xy(level, vertex.x, vertex.y)
        for vertex in level.vertex_data.vertices
    ]
    minimum_x = min(point[0] for point in raw_points)
    maximum_x = max(point[0] for point in raw_points)
    minimum_y = min(point[1] for point in raw_points)
    maximum_y = max(point[1] for point in raw_points)
    return (
        (minimum_x + maximum_x) / 2.0,
        (minimum_y + maximum_y) / 2.0,
    )


def level_image_to_world_xy(
    level: LevelData,
    image_x: float,
    image_y: float,
) -> tuple[float, float]:
    """Transform blueprint-image pixels into the level's world XY coordinates."""

    scale = _get_valid_level_scale(level)
    raw_x, raw_y = _image_to_unscaled_world_xy(level, image_x, image_y)
    pivot_x, pivot_y = get_level_world_pivot(level)
    return (
        pivot_x
        + (raw_x - pivot_x) * scale
        + _get_valid_level_offset(level, "x"),
        pivot_y
        + (raw_y - pivot_y) * scale
        + _get_valid_level_offset(level, "y"),
    )


def level_world_to_image_xy(
    level: LevelData,
    world_x: float,
    world_y: float,
) -> tuple[float, float]:
    """Invert :func:`level_image_to_world_xy` for one world-space point."""

    scale = _get_valid_level_scale(level)
    pivot_x, pivot_y = get_level_world_pivot(level)
    offset_x = _get_valid_level_offset(level, "x")
    offset_y = _get_valid_level_offset(level, "y")
    raw_x = pivot_x + (
        float(world_x)
        - offset_x
        - pivot_x
    ) / scale
    raw_y = pivot_y + (
        float(world_y)
        - offset_y
        - pivot_y
    ) / scale
    return _unscaled_world_to_image_xy(level, raw_x, raw_y)


# ### Internal coordinate helpers ###
def _image_to_unscaled_world_xy(
    level: LevelData,
    image_x: float,
    image_y: float,
) -> tuple[float, float]:
    if level.image_size_pixels is None:
        centered_x = float(image_x)
        centered_y = float(image_y)
    else:
        image_width, image_height = level.image_size_pixels
        centered_x = float(image_x) - float(image_width) / 2.0
        centered_y = float(image_y) - float(image_height) / 2.0
    return centered_x * PIXEL_TO_METER, -centered_y * PIXEL_TO_METER


def _unscaled_world_to_image_xy(
    level: LevelData,
    raw_x: float,
    raw_y: float,
) -> tuple[float, float]:
    image_x = float(raw_x) / PIXEL_TO_METER
    image_y = -float(raw_y) / PIXEL_TO_METER
    if level.image_size_pixels is not None:
        image_width, image_height = level.image_size_pixels
        image_x += float(image_width) / 2.0
        image_y += float(image_height) / 2.0
    return image_x, image_y


# ### Validation helpers ###
def _get_valid_level_scale(level: LevelData) -> float:
    if isinstance(level.scale, bool):
        raise ValueError(
            f"Level {level.index} scale must be a finite number greater than zero."
        )
    try:
        scale = float(level.scale)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"Level {level.index} scale must be a finite number greater than zero."
        ) from error
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            f"Level {level.index} scale must be a finite number greater than zero."
        )
    return scale


def _get_valid_level_offset(level: LevelData, axis: str) -> float:
    raw_offset = getattr(level, f"offset_{axis}_meters")
    if isinstance(raw_offset, bool):
        raise ValueError(
            f"Level {level.index} {axis} offset must be a finite number."
        )
    try:
        offset = float(raw_offset)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"Level {level.index} {axis} offset must be a finite number."
        ) from error
    if not math.isfinite(offset):
        raise ValueError(
            f"Level {level.index} {axis} offset must be a finite number."
        )
    return offset
