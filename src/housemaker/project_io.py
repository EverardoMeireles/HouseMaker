# ### Imports ###
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from housemaker.models import (
    DEFAULT_INCLUDE_IN_EXPORT,
    DEFAULT_IMAGE_OFFSET,
    DEFAULT_IMAGE_SCALE,
    DEFAULT_UV_MAP_HEIGHT,
    DEFAULT_UV_MAP_WIDTH,
    GROUND_LEVEL_INDEX,
    LevelData,
    RoomData,
    VertexData,
    WallTextureData,
    create_default_levels,
)

# ### Constants ###
PROJECT_FILE_VERSION = 1


# ### Data models ###
@dataclass
class ProjectData:
    blueprint_path: str | None
    current_level_index: int
    levels: list[LevelData]
    image_library_paths: list[str] = field(default_factory=list)


# ### Public helpers ###
def save_project(
    path: str | Path,
    current_level_index: int,
    levels: list[LevelData],
    image_library_paths: list[str] | None = None,
) -> Path:
    export_path = Path(path)
    payload = {
        "project_version": PROJECT_FILE_VERSION,
        "blueprint_path": None,
        "current_level_index": int(current_level_index),
        "image_library_paths": _serialize_image_library_paths(
            image_library_paths or []
        ),
        "levels": [
            {
                "index": level.index,
                "name": level.name,
                "height_meters": level.height_meters,
                "image_path": _normalize_optional_path(level.image_path),
                "image_size_pixels": _serialize_image_size(level.image_size_pixels),
                "image_scale": float(level.image_scale),
                "image_offset_x": float(level.image_offset_x),
                "image_offset_y": float(level.image_offset_y),
                "include_in_export": bool(level.include_in_export),
                "vertex_data": level.vertex_data.to_dict(),
                "rooms": [_serialize_room(room) for room in level.rooms],
            }
            for level in levels
        ],
    }

    try:
        export_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
    except OSError as error:
        raise ValueError(f"Unable to save project file: {export_path}") from error

    return export_path


def load_project(path: str | Path) -> ProjectData:
    project_path = Path(path)
    try:
        payload = json.loads(project_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"Unable to read project file: {project_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Project file is not valid JSON: {project_path}") from error

    project_version = int(payload.get("project_version", 0))
    if project_version != PROJECT_FILE_VERSION:
        raise ValueError(f"Unsupported project version: {project_version}")

    legacy_blueprint_path = _normalize_optional_path(payload.get("blueprint_path"))

    levels = create_default_levels()
    level_lookup = {level.index: level for level in levels}

    for raw_level in payload.get("levels", []):
        level_index = int(raw_level.get("index", -1))
        level = level_lookup.get(level_index)
        if level is None:
            continue

        level.name = str(raw_level.get("name", level.name))
        level.height_meters = float(raw_level.get("height_meters", level.height_meters))
        level.image_path = _normalize_optional_path(
            raw_level.get("image_path", legacy_blueprint_path)
        )
        level.image_size_pixels = _deserialize_image_size(
            raw_level.get("image_size_pixels")
        )
        level.image_scale = float(raw_level.get("image_scale", DEFAULT_IMAGE_SCALE))
        level.image_offset_x = float(
            raw_level.get("image_offset_x", DEFAULT_IMAGE_OFFSET)
        )
        level.image_offset_y = float(
            raw_level.get("image_offset_y", DEFAULT_IMAGE_OFFSET)
        )
        level.include_in_export = bool(
            raw_level.get("include_in_export", DEFAULT_INCLUDE_IN_EXPORT)
        )
        level.vertex_data = VertexData.from_dict(raw_level.get("vertex_data", {}))
        level.rooms = _deserialize_rooms(raw_level.get("rooms", []))

    image_library_paths = _deserialize_image_library_paths(
        payload.get("image_library_paths", [])
    )
    _clear_image_library_paths_from_levels(levels, image_library_paths)

    current_level_index = int(payload.get("current_level_index", GROUND_LEVEL_INDEX))
    if current_level_index not in level_lookup:
        current_level_index = GROUND_LEVEL_INDEX

    return ProjectData(
        blueprint_path=legacy_blueprint_path,
        current_level_index=current_level_index,
        levels=levels,
        image_library_paths=image_library_paths,
    )


# ### Serialization helpers ###
def _normalize_optional_path(path_value: object) -> str | None:
    path_text = str(path_value or "").strip()
    if not path_text:
        return None

    return str(Path(path_text).resolve())


def _serialize_image_size(
    image_size_pixels: tuple[float, float] | None,
) -> list[float] | None:
    if image_size_pixels is None:
        return None

    return [float(image_size_pixels[0]), float(image_size_pixels[1])]


def _deserialize_image_size(raw_image_size: object) -> tuple[float, float] | None:
    if not isinstance(raw_image_size, list | tuple) or len(raw_image_size) != 2:
        return None

    return (float(raw_image_size[0]), float(raw_image_size[1]))


def _serialize_image_library_paths(image_paths: list[str]) -> list[str]:
    normalized_paths: list[str] = []
    for image_path in image_paths:
        normalized_path = _normalize_optional_path(image_path)
        if normalized_path is None or normalized_path in normalized_paths:
            continue

        normalized_paths.append(normalized_path)

    return normalized_paths


def _deserialize_image_library_paths(raw_image_paths: object) -> list[str]:
    if not isinstance(raw_image_paths, list | tuple):
        return []

    return _serialize_image_library_paths(
        [str(image_path) for image_path in raw_image_paths]
    )


def _clear_image_library_paths_from_levels(
    levels: list[LevelData],
    image_library_paths: list[str],
) -> None:
    image_library_path_lookup = set(image_library_paths)
    for level in levels:
        if level.image_path not in image_library_path_lookup:
            continue

        level.image_path = None
        level.image_size_pixels = None


def _serialize_room(room: RoomData) -> dict[str, object]:
    return {
        "name": room.name,
        "vertex_ids": [int(vertex_id) for vertex_id in room.vertex_ids],
        "center_vertex_id": int(room.center_vertex_id),
        "color_rgb": [int(color_value) for color_value in room.color_rgb],
        "uv_map_width": int(room.uv_map_width),
        "uv_map_height": int(room.uv_map_height),
        "wall_uv_scales": {
            str(wall_key): float(wall_scale)
            for wall_key, wall_scale in room.wall_uv_scales.items()
        },
        "wall_uv_rotations": {
            str(wall_key): int(wall_rotation)
            for wall_key, wall_rotation in room.wall_uv_rotations.items()
        },
        "wall_uv_positions": {
            str(wall_key): [float(wall_position[0]), float(wall_position[1])]
            for wall_key, wall_position in room.wall_uv_positions.items()
        },
        "wall_textures": _serialize_wall_textures(room.wall_textures),
    }


def _deserialize_rooms(raw_rooms: object) -> list[RoomData]:
    if not isinstance(raw_rooms, list):
        return []

    rooms: list[RoomData] = []
    for raw_room in raw_rooms:
        if not isinstance(raw_room, dict):
            continue

        raw_color = raw_room.get("color_rgb", [140, 180, 220])
        if not isinstance(raw_color, list | tuple) or len(raw_color) != 3:
            raw_color = [140, 180, 220]

        rooms.append(
            RoomData(
                name=str(raw_room.get("name", "Room")),
                vertex_ids=tuple(
                    int(vertex_id)
                    for vertex_id in raw_room.get("vertex_ids", [])
                ),
                center_vertex_id=int(raw_room.get("center_vertex_id", 0)),
                color_rgb=(
                    int(raw_color[0]),
                    int(raw_color[1]),
                    int(raw_color[2]),
                ),
                uv_map_width=int(
                    raw_room.get("uv_map_width", DEFAULT_UV_MAP_WIDTH)
                ),
                uv_map_height=int(
                    raw_room.get("uv_map_height", DEFAULT_UV_MAP_HEIGHT)
                ),
                wall_uv_scales=_deserialize_wall_uv_scales(
                    raw_room.get("wall_uv_scales", {})
                ),
                wall_uv_rotations=_deserialize_wall_uv_rotations(
                    raw_room.get("wall_uv_rotations", {})
                ),
                wall_uv_positions=_deserialize_wall_uv_positions(
                    raw_room.get("wall_uv_positions", {})
                ),
                wall_textures=_deserialize_wall_textures(
                    raw_room.get("wall_textures", {})
                ),
            )
        )

    return rooms


def _deserialize_wall_uv_scales(raw_wall_uv_scales: object) -> dict[str, float]:
    if not isinstance(raw_wall_uv_scales, dict):
        return {}

    return {
        str(wall_key): float(wall_scale)
        for wall_key, wall_scale in raw_wall_uv_scales.items()
    }


def _deserialize_wall_uv_rotations(raw_wall_uv_rotations: object) -> dict[str, int]:
    if not isinstance(raw_wall_uv_rotations, dict):
        return {}

    return {
        str(wall_key): _normalize_wall_uv_rotation(wall_rotation)
        for wall_key, wall_rotation in raw_wall_uv_rotations.items()
    }


def _deserialize_wall_uv_positions(
    raw_wall_uv_positions: object,
) -> dict[str, tuple[float, float]]:
    if not isinstance(raw_wall_uv_positions, dict):
        return {}

    return {
        str(wall_key): (float(wall_position[0]), float(wall_position[1]))
        for wall_key, wall_position in raw_wall_uv_positions.items()
        if isinstance(wall_position, list | tuple) and len(wall_position) == 2
    }


def _normalize_wall_uv_rotation(raw_wall_rotation: object) -> int:
    try:
        rotation_degrees = int(round(float(raw_wall_rotation)))
    except (TypeError, ValueError):
        return 0

    return rotation_degrees % 360


def _serialize_wall_textures(
    wall_textures: dict[str, WallTextureData],
) -> dict[str, dict[str, object]]:
    return {
        str(wall_key): {
            "image_path": _normalize_optional_path(texture_data.image_path),
            "source_x": float(texture_data.source_x),
            "source_y": float(texture_data.source_y),
            "source_width": float(texture_data.source_width),
            "source_height": float(texture_data.source_height),
        }
        for wall_key, texture_data in wall_textures.items()
    }


def _deserialize_wall_textures(raw_wall_textures: object) -> dict[str, WallTextureData]:
    if not isinstance(raw_wall_textures, dict):
        return {}

    wall_textures: dict[str, WallTextureData] = {}
    for wall_key, raw_texture in raw_wall_textures.items():
        if not isinstance(raw_texture, dict):
            continue

        image_path = _normalize_optional_path(raw_texture.get("image_path"))
        if image_path is None:
            continue

        wall_textures[str(wall_key)] = WallTextureData(
            image_path=image_path,
            source_x=float(raw_texture.get("source_x", 0.0)),
            source_y=float(raw_texture.get("source_y", 0.0)),
            source_width=max(1.0, float(raw_texture.get("source_width", 1.0))),
            source_height=max(1.0, float(raw_texture.get("source_height", 1.0))),
        )

    return wall_textures
