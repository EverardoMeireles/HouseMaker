# ### Imports ###
from __future__ import annotations

import math
from collections.abc import Sequence

from housemaker.doorway_geometry import build_doorway_cross_section_outline
from housemaker.models import (
    GROUND_LEVEL_INDEX,
    PIXEL_TO_METER,
    DoorwayData,
    LevelData,
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
    """Return paired world-space vertices for one extruded opening outline."""

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
    bottom_height_meters = _get_valid_doorway_outline_value(
        doorway.bottom_height_meters,
        "bottom height",
    )
    if bottom_height_meters < 0.0:
        raise ValueError("Doorway outline bottom height cannot be negative.")

    rotation_radians = math.radians(rotation_degrees)
    depth_direction = (
        math.cos(rotation_radians),
        math.sin(rotation_radians),
    )
    width_direction = (-depth_direction[1], depth_direction[0])
    half_depth_pixels = depth_meters / PIXEL_TO_METER / 2.0
    base_z_meters = _get_valid_doorway_outline_value(
        base_z_by_level_index[level.index],
        "base Z",
    )
    cross_section = build_doorway_cross_section_outline(
        width_meters,
        height_meters,
        doorway.shape,
        arch_amount=doorway.arch_amount,
    )
    depth_faces: list[tuple[tuple[float, float, float], ...]] = []
    for depth_sign in (-1.0, 1.0):
        profile_positions: list[tuple[float, float, float]] = []
        for width_offset_meters, height_offset_meters in cross_section:
            width_offset_pixels = width_offset_meters / PIXEL_TO_METER
            image_x = (
                center_x
                + width_direction[0] * width_offset_pixels
                + depth_direction[0] * depth_sign * half_depth_pixels
            )
            image_y = (
                center_y
                + width_direction[1] * width_offset_pixels
                + depth_direction[1] * depth_sign * half_depth_pixels
            )
            world_x, world_y = level_image_to_world_xy(
                level,
                image_x,
                image_y,
            )
            profile_positions.append(
                (
                    world_x,
                    world_y,
                    base_z_meters
                    + bottom_height_meters
                    + height_offset_meters,
                )
            )
        depth_faces.append(tuple(profile_positions))

    positions: list[tuple[float, float, float]] = []
    for profile_positions in depth_faces:
        for first_position, second_position in zip(
            profile_positions,
            profile_positions[1:],
        ):
            positions.extend((first_position, second_position))

    first_depth_face, second_depth_face = depth_faces
    for first_position, second_position in zip(
        first_depth_face[:-1],
        second_depth_face[:-1],
    ):
        positions.extend((first_position, second_position))
    return tuple(positions)


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
