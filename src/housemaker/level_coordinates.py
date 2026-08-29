# ### Imports ###
from __future__ import annotations

import math
from collections.abc import Sequence

from housemaker.models import (
    GROUND_LEVEL_INDEX,
    PIXEL_TO_METER,
    DoorwayData,
    LevelData,
)


# ### Constants ###
_DOORWAY_PRISM_EDGE_VERTEX_INDICES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


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


def build_doorway_world_outline_positions(
    levels: Sequence[LevelData],
    level: LevelData,
    doorway: DoorwayData,
) -> tuple[tuple[float, float, float], ...]:
    """Return paired world-space vertices for one doorway wireframe prism."""

    if not isinstance(level, LevelData):
        raise TypeError("A doorway outline requires a LevelData value.")
    if not isinstance(doorway, DoorwayData):
        raise TypeError("A doorway outline requires a DoorwayData value.")

    level_sequence = tuple(levels)
    if not all(isinstance(candidate, LevelData) for candidate in level_sequence):
        raise TypeError("Doorway outline levels must contain LevelData values.")
    base_z_by_level_index = build_level_base_z_lookup(level_sequence)
    if level.index not in base_z_by_level_index:
        raise ValueError("The doorway level must be present in the level sequence.")

    center_x = _get_valid_doorway_outline_value(
        doorway.center_x,
        "center X",
    )
    center_y = _get_valid_doorway_outline_value(
        doorway.center_y,
        "center Y",
    )
    width_meters = _get_valid_doorway_outline_value(
        doorway.width_meters,
        "width",
        must_be_positive=True,
    )
    height_meters = _get_valid_doorway_outline_value(
        doorway.height_meters,
        "height",
        must_be_positive=True,
    )
    depth_meters = _get_valid_doorway_outline_value(
        doorway.depth_meters,
        "depth",
        must_be_positive=True,
    )
    rotation_degrees = _get_valid_doorway_outline_value(
        doorway.rotation_degrees,
        "rotation",
    )

    rotation_radians = math.radians(rotation_degrees)
    depth_direction = (
        math.cos(rotation_radians),
        math.sin(rotation_radians),
    )
    width_direction = (-depth_direction[1], depth_direction[0])
    half_width_pixels = width_meters / PIXEL_TO_METER / 2.0
    half_depth_pixels = depth_meters / PIXEL_TO_METER / 2.0
    image_corners = tuple(
        (
            center_x
            + width_direction[0] * width_sign * half_width_pixels
            + depth_direction[0] * depth_sign * half_depth_pixels,
            center_y
            + width_direction[1] * width_sign * half_width_pixels
            + depth_direction[1] * depth_sign * half_depth_pixels,
        )
        for width_sign, depth_sign in (
            (-1.0, -1.0),
            (1.0, -1.0),
            (1.0, 1.0),
            (-1.0, 1.0),
        )
    )
    world_footprint = tuple(
        level_image_to_world_xy(level, image_x, image_y)
        for image_x, image_y in image_corners
    )
    base_z_meters = _get_valid_doorway_outline_value(
        base_z_by_level_index[level.index],
        "base Z",
    )
    bottom_corners = tuple(
        (world_x, world_y, base_z_meters)
        for world_x, world_y in world_footprint
    )
    top_corners = tuple(
        (world_x, world_y, base_z_meters + height_meters)
        for world_x, world_y in world_footprint
    )
    corners = (*bottom_corners, *top_corners)
    return tuple(
        corners[vertex_index]
        for edge in _DOORWAY_PRISM_EDGE_VERTEX_INDICES
        for vertex_index in edge
    )


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


def _get_valid_doorway_outline_value(
    value: object,
    field_name: str,
    *,
    must_be_positive: bool = False,
) -> float:
    """Normalize one finite doorway outline coordinate or dimension."""

    if isinstance(value, bool):
        raise ValueError(f"Doorway outline {field_name} must be a number.")
    try:
        normalized_value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"Doorway outline {field_name} must be a number."
        ) from error
    if not math.isfinite(normalized_value):
        raise ValueError(f"Doorway outline {field_name} must be finite.")
    if must_be_positive and normalized_value <= 0.0:
        raise ValueError(
            f"Doorway outline {field_name} must be greater than zero."
        )
    return normalized_value
