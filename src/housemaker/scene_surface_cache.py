# ### Imports ###
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass

from housemaker.models import LevelData, RoomData, StairData

# ### Geometry dependencies ###
LEVEL_GEOMETRY_FIELDS = (
    "index",
    "include_in_export",
    "height_meters",
    "floor_thickness_meters",
    "scale",
    "offset_x_meters",
    "offset_y_meters",
    "canvas_level_scale",
    "canvas_offset_x_pixels",
    "canvas_offset_y_pixels",
    "image_size_pixels",
    "doorways",
    "windows",
    "open_spaces",
    "editable_surfaces",
    "sloped_ceilings",
    "architectural_trims",
    "landings",
    "flipped_surface_ids",
)
ROOM_GEOMETRY_FIELDS = (
    "vertex_ids",
    "center_vertex_id",
    "height_meters",
    "uv_map_width",
    "uv_map_height",
    "wall_uv_scales",
    "wall_uv_rotations",
    "wall_uv_positions",
    "wall_subdivisions",
    "wall_subdivision_positions",
    "wall_subdivision_source_ranges",
)


# ### Public signature builder ###
def build_scene_surface_geometry_signature(
    levels: Sequence[LevelData],
    stairs: Sequence[StairData],
    *,
    generated_trim_keys: Sequence[tuple[int, str]] = (),
) -> tuple[object, ...]:
    """Snapshot live geometry inputs without retaining mutable model aliases.

    This key is for exported semantic surface IDs, not rendered materials or
    the current selection. It deliberately reads authored geometry rather than
    a viewer revision: pending edits can change the live graph before the
    delayed scene rebuild. Excluded levels remain dependencies because story
    elevations and neighboring slab openings can affect included levels.

    Plan images, OCR evidence and texture assets are not read. Only the plan's
    effective metric density influences geometry. Global multipliers are an
    external scene transform and do not change authored surface IDs.
    """

    return (
        tuple(_build_level_geometry_signature(level) for level in levels),
        _freeze_geometry_value(stairs),
        tuple(sorted(set(generated_trim_keys))),
    )


# ### Immutable model snapshots ###
def _build_level_geometry_signature(level: LevelData) -> tuple[object, ...]:
    graph = level.vertex_data
    return (
        _snapshot_fields(level, LEVEL_GEOMETRY_FIELDS),
        float(level.meters_per_pixel),
        _freeze_geometry_value(graph.vertices),
        _freeze_geometry_value(graph.edges),
        tuple(_build_room_geometry_signature(room) for room in level.rooms),
    )


def _build_room_geometry_signature(room: RoomData) -> tuple[object, ...]:
    # Wall texture files and display names/colors do not alter source geometry.
    return _snapshot_fields(room, ROOM_GEOMETRY_FIELDS)


def _snapshot_fields(model: object, names: Sequence[str]) -> tuple[object, ...]:
    return tuple(
        (name, _freeze_geometry_value(getattr(model, name))) for name in names
    )


def _freeze_geometry_value(value: object) -> object:
    """Copy small model values into hashable containers, without serialization."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__name__,
            _snapshot_fields(value, tuple(item.name for item in fields(value))),
        )
    if isinstance(value, Mapping):
        return frozenset(
            (_freeze_geometry_value(key), _freeze_geometry_value(item))
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_geometry_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_geometry_value(item) for item in value)
    raise TypeError(f"Unsupported scene geometry value: {type(value).__name__}.")
