# ### Imports ###
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from housemaker.models import (
    DEFAULT_INCLUDE_IN_EXPORT,
    DEFAULT_IMAGE_OFFSET,
    DEFAULT_IMAGE_SCALE,
    GROUND_LEVEL_INDEX,
    LevelData,
    RoomData,
    VertexData,
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


# ### Public helpers ###
def save_project(
    path: str | Path,
    current_level_index: int,
    levels: list[LevelData],
) -> Path:
    export_path = Path(path)
    payload = {
        "project_version": PROJECT_FILE_VERSION,
        "blueprint_path": None,
        "current_level_index": int(current_level_index),
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

    current_level_index = int(payload.get("current_level_index", GROUND_LEVEL_INDEX))
    if current_level_index not in level_lookup:
        current_level_index = GROUND_LEVEL_INDEX

    return ProjectData(
        blueprint_path=legacy_blueprint_path,
        current_level_index=current_level_index,
        levels=levels,
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


def _serialize_room(room: RoomData) -> dict[str, object]:
    return {
        "name": room.name,
        "vertex_ids": [int(vertex_id) for vertex_id in room.vertex_ids],
        "center_vertex_id": int(room.center_vertex_id),
        "color_rgb": [int(color_value) for color_value in room.color_rgb],
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
            )
        )

    return rooms
