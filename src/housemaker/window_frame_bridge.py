"""Reuse doorway alignment rules for Shift-snapped window frame bridges."""

# ### Imports ###
from __future__ import annotations

from collections.abc import Sequence

from housemaker.doorway_bridge import find_doorway_bridge_target
from housemaker.models import Edge, LevelData
from housemaker.surface_geometry import (
    MIN_WINDOW_SIZE_METERS,
    WallWindowPlacement,
    build_wall_surface_id,
)


# ### Window bridge previews ###
def build_window_frame_bridge_placement(
    level: LevelData,
    pointer: tuple[float, float],
    center_tolerance: float,
    *,
    preferred_width_meters: float,
    height_meters: float,
    wall_margin_meters: float,
) -> WallWindowPlacement | None:
    """Preview an exact gap-width opening without changing wall topology."""

    target = find_doorway_bridge_target(
        level.vertex_data,
        {room.center_vertex_id for room in level.rooms},
        pointer,
        preferred_width_meters,
        center_tolerance,
        meters_per_pixel=level.meters_per_pixel * float(level.scale),
    )
    available_height = float(level.height_meters) - 2.0 * wall_margin_meters
    if target is None or available_height < MIN_WINDOW_SIZE_METERS:
        return None
    half_height_ratio = (
        min(height_meters, available_height)
        / float(level.height_meters)
        * 0.5
    )
    vertex_ids = (target.first_vertex_id, target.second_vertex_id)
    return WallWindowPlacement(
        wall_surface_id=build_wall_surface_id(
            level.index, f"{min(vertex_ids)}:{max(vertex_ids)}"
        ),
        start_ratio=0.0,
        end_ratio=1.0,
        bottom_ratio=0.5 - half_height_ratio,
        top_ratio=0.5 + half_height_ratio,
        bridge_vertex_ids=vertex_ids,
    )


# ### Transaction helpers ###
def add_window_frame_bridge(
    levels: Sequence[LevelData],
    placement: WallWindowPlacement,
) -> tuple[LevelData, Edge] | None:
    """Revalidate the preview and connect its endpoints for the window commit."""

    vertex_ids = placement.bridge_vertex_ids
    if vertex_ids is None:
        return None
    level = next(
        (
            level
            for level in levels
            if placement.wall_surface_id == build_wall_surface_id(
                level.index, f"{min(vertex_ids)}:{max(vertex_ids)}"
            )
        ),
        None,
    )
    if level is None:
        raise ValueError("The window bridge has no owning level.")
    first = level.vertex_data.get_vertex(vertex_ids[0])
    second = level.vertex_data.get_vertex(vertex_ids[1])
    if first is None or second is None:
        raise ValueError("The window bridge's endpoints no longer exist.")
    target = find_doorway_bridge_target(
        level.vertex_data,
        {room.center_vertex_id for room in level.rooms},
        ((first.x + second.x) * 0.5, (first.y + second.y) * 0.5),
        preset_width_meters=1.0,
        center_tolerance=1e-6,
        meters_per_pixel=level.meters_per_pixel * float(level.scale),
    )
    if target is None or set(vertex_ids) != {
        target.first_vertex_id,
        target.second_vertex_id,
    }:
        raise ValueError(
            "The window bridge's endpoints must remain aligned and unconnected."
        )
    edge = level.vertex_data.add_edge(*vertex_ids)
    if edge is None:
        raise ValueError("The window bridge's wall already exists.")
    return level, edge
