# ### Imports ###
from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from housemaker.camera_models import CameraPose, InitialFirstPersonCamera
from housemaker.generation_state import GenerationData
from housemaker.surface_texture_state import SurfaceTextureData
from housemaker.texture_atlas_state import TextureAtlasData
from housemaker.models import (
    DEFAULT_DOORWAY_DEPTH_METERS,
    DEFAULT_DOORWAY_HEIGHT_METERS,
    DEFAULT_DOORWAY_WIDTH_METERS,
    DEFAULT_FLOOR_THICKNESS_METERS,
    DEFAULT_INCLUDE_IN_EXPORT,
    DEFAULT_IMAGE_OFFSET,
    DEFAULT_IMAGE_SCALE,
    DEFAULT_LEVEL_OFFSET_METERS,
    DEFAULT_LEVEL_SCALE,
    DEFAULT_ROOM_HEIGHT_METERS,
    DEFAULT_UV_MAP_HEIGHT,
    DEFAULT_UV_MAP_WIDTH,
    GROUND_LEVEL_INDEX,
    MAX_FLOOR_THICKNESS_METERS,
    MAX_DOORWAY_DEPTH_METERS,
    MAX_DOORWAY_HEIGHT_METERS,
    MAX_DOORWAY_WIDTH_METERS,
    MAX_LEVEL_OFFSET_METERS,
    MAX_LEVEL_SCALE,
    MIN_FLOOR_THICKNESS_METERS,
    MIN_DOORWAY_DEPTH_METERS,
    MIN_DOORWAY_HEIGHT_METERS,
    MIN_DOORWAY_WIDTH_METERS,
    MIN_LEVEL_OFFSET_METERS,
    MIN_LEVEL_SCALE,
    DoorwayData,
    DoorwayPreset,
    LevelData,
    RoomData,
    StairData,
    VertexData,
    WallTextureData,
    WindowData,
    create_default_doorway_presets,
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
    doorway_presets: list[DoorwayPreset] = field(
        default_factory=create_default_doorway_presets
    )
    generation: GenerationData = field(default_factory=GenerationData)
    surface_texture_generation: SurfaceTextureData = field(
        default_factory=SurfaceTextureData
    )
    initial_first_person_camera: InitialFirstPersonCamera | None = None
    stairs: list[StairData] = field(default_factory=list)
    texture_atlases: TextureAtlasData = field(default_factory=TextureAtlasData)


# ### Public helpers ###
def save_project(
    path: str | Path,
    current_level_index: int,
    levels: list[LevelData],
    image_library_paths: list[str] | None = None,
    doorway_presets: list[DoorwayPreset] | None = None,
    generation: GenerationData | None = None,
    surface_texture_generation: SurfaceTextureData | None = None,
    initial_first_person_camera: InitialFirstPersonCamera | None = None,
    stairs: list[StairData] | None = None,
    texture_atlases: TextureAtlasData | None = None,
) -> Path:
    export_path = Path(path)
    payload = {
        "project_version": PROJECT_FILE_VERSION,
        "blueprint_path": None,
        "current_level_index": int(current_level_index),
        "image_library_paths": _serialize_image_library_paths(
            image_library_paths or []
        ),
        "doorway_presets": _serialize_doorway_presets(
            doorway_presets
            if doorway_presets is not None
            else create_default_doorway_presets()
        ),
        "generation": (
            generation.to_dict()
            if generation is not None
            else GenerationData().to_dict()
        ),
        "surface_texture_generation": (
            surface_texture_generation.to_dict()
            if surface_texture_generation is not None
            else SurfaceTextureData().to_dict()
        ),
        "texture_atlases": (
            texture_atlases.to_dict()
            if texture_atlases is not None
            else TextureAtlasData().to_dict()
        ),
        "initial_first_person_camera": (
            None
            if initial_first_person_camera is None
            else initial_first_person_camera.to_dict()
        ),
        "stairs": _serialize_stairs(stairs or []),
        "levels": [
            {
                "index": level.index,
                "name": level.name,
                "height_meters": level.height_meters,
                "scale": float(level.scale),
                "offset_x_meters": float(level.offset_x_meters),
                "offset_y_meters": float(level.offset_y_meters),
                "floor_thickness_meters": level.floor_thickness_meters,
                "floor_contour_vertex_ids": list(
                    level.floor_contour_vertex_ids
                ),
                "image_path": _normalize_optional_path(level.image_path),
                "image_size_pixels": _serialize_image_size(level.image_size_pixels),
                "image_scale": float(level.image_scale),
                "image_offset_x": float(level.image_offset_x),
                "image_offset_y": float(level.image_offset_y),
                "include_in_export": bool(level.include_in_export),
                "vertex_data": level.vertex_data.to_dict(),
                "rooms": [_serialize_room(room) for room in level.rooms],
                "doorways": [
                    _serialize_doorway(doorway)
                    for doorway in level.doorways
                ],
                "windows": [window.to_dict() for window in level.windows],
            }
            for level in levels
        ],
    }

    _write_project_json_atomically(export_path, payload)

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
        level.scale = _deserialize_level_scale(
            raw_level.get("scale", DEFAULT_LEVEL_SCALE)
        )
        level.offset_x_meters = _deserialize_level_offset_meters(
            raw_level.get("offset_x_meters", DEFAULT_LEVEL_OFFSET_METERS)
        )
        level.offset_y_meters = _deserialize_level_offset_meters(
            raw_level.get("offset_y_meters", DEFAULT_LEVEL_OFFSET_METERS)
        )
        level.floor_thickness_meters = _deserialize_floor_thickness_meters(
            raw_level.get(
                "floor_thickness_meters",
                DEFAULT_FLOOR_THICKNESS_METERS,
            )
        )
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
        level.floor_contour_vertex_ids = _deserialize_floor_contour_vertex_ids(
            raw_level.get("floor_contour_vertex_ids"),
            level.vertex_data,
        )
        level.rooms = _deserialize_rooms(
            raw_level.get("rooms", []),
            default_height_meters=level.height_meters,
        )
        level.doorways = _deserialize_doorways(raw_level.get("doorways", []))
        level.windows = _deserialize_windows(
            raw_level.get("windows", []),
            level_index=level.index,
        )

    image_library_paths = _deserialize_image_library_paths(
        payload.get("image_library_paths", [])
    )
    doorway_presets = _deserialize_doorway_presets(
        payload.get("doorway_presets")
    )
    generation = _deserialize_generation(
        payload.get("generation"),
        payload.get("dynamic_generation"),
    )
    surface_texture_generation = _deserialize_surface_texture_generation(
        payload.get("surface_texture_generation")
    )
    texture_atlases = _deserialize_texture_atlases(
        payload.get("texture_atlases")
    )
    if "initial_first_person_camera" in payload:
        initial_first_person_camera = _deserialize_initial_first_person_camera(
            payload.get("initial_first_person_camera"),
            valid_level_indices=set(level_lookup),
        )
    else:
        initial_first_person_camera = _build_legacy_initial_camera(
            payload.get("dynamic_generation")
        )
    stairs = _deserialize_stairs(
        payload.get("stairs"),
        valid_level_indices=set(level_lookup),
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
        doorway_presets=doorway_presets,
        generation=generation,
        surface_texture_generation=surface_texture_generation,
        texture_atlases=texture_atlases,
        initial_first_person_camera=initial_first_person_camera,
        stairs=stairs,
    )


# ### Project file helpers ###
def _write_project_json_atomically(
    export_path: Path,
    payload: dict[str, object],
) -> None:
    """Replace a project only after its complete JSON is durable on disk."""

    temporary_path: Path | None = None
    try:
        serialized_payload = json.dumps(payload, indent=2)
        file_descriptor, raw_temporary_path = tempfile.mkstemp(
            prefix=f".{export_path.name}.",
            suffix=".tmp",
            dir=str(export_path.parent),
        )
        temporary_path = Path(raw_temporary_path)
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized_payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, export_path)
    except (OSError, TypeError, ValueError) as error:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise ValueError(f"Unable to save project file: {export_path}") from error


# ### Serialization helpers ###
def _deserialize_generation(
    raw_generation: object,
    raw_legacy_dynamic_generation: object,
) -> GenerationData:
    if isinstance(raw_generation, dict):
        try:
            return GenerationData.from_dict(raw_generation)
        except (KeyError, TypeError, ValueError, OverflowError):
            return GenerationData()

    if not isinstance(raw_legacy_dynamic_generation, dict):
        return GenerationData()
    raw_video_metadata = raw_legacy_dynamic_generation.get("video_metadata")
    if not isinstance(raw_video_metadata, dict):
        return GenerationData()
    try:
        return GenerationData.from_dict(
            {"video_metadata": raw_video_metadata}
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return GenerationData()


def _deserialize_surface_texture_generation(
    raw_surface_texture_generation: object,
) -> SurfaceTextureData:
    """Load optional surface-texture state without breaking older projects."""

    if not isinstance(raw_surface_texture_generation, dict):
        return SurfaceTextureData()
    try:
        return SurfaceTextureData.from_dict(raw_surface_texture_generation)
    except (KeyError, TypeError, ValueError, OverflowError):
        return SurfaceTextureData()


def _deserialize_texture_atlases(
    raw_texture_atlases: object,
) -> TextureAtlasData:
    """Load optional atlas state while keeping old or damaged projects usable."""

    if not isinstance(raw_texture_atlases, dict):
        return TextureAtlasData()
    try:
        return TextureAtlasData.from_dict(raw_texture_atlases)
    except (KeyError, TypeError, ValueError, OverflowError):
        return TextureAtlasData()


def _deserialize_initial_first_person_camera(
    raw_camera: object,
    valid_level_indices: set[int],
) -> InitialFirstPersonCamera | None:
    if not isinstance(raw_camera, dict):
        return None

    try:
        camera = InitialFirstPersonCamera.from_dict(raw_camera)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if camera.level_index not in valid_level_indices:
        return None
    return camera


def _build_legacy_initial_camera(
    raw_dynamic_generation: object,
) -> InitialFirstPersonCamera | None:
    if not isinstance(raw_dynamic_generation, dict):
        return None
    raw_alignments = raw_dynamic_generation.get("alignments")
    if not isinstance(raw_alignments, list):
        return None
    for raw_alignment in raw_alignments:
        if not isinstance(raw_alignment, dict):
            continue
        try:
            is_frame_zero = int(raw_alignment.get("frame_index", -1)) == 0
        except (TypeError, ValueError, OverflowError):
            continue
        is_manual = (
            raw_alignment.get("source") == "manual"
            or raw_alignment.get("manual") is True
        )
        raw_pose = raw_alignment.get("pose")
        if not is_frame_zero or not is_manual or not isinstance(raw_pose, dict):
            continue
        try:
            return InitialFirstPersonCamera(
                level_index=GROUND_LEVEL_INDEX,
                pose=CameraPose.from_dict(raw_pose),
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
    return None


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


def _deserialize_floor_contour_vertex_ids(
    raw_vertex_ids: object,
    vertex_data: VertexData,
) -> tuple[int, ...]:
    if not isinstance(raw_vertex_ids, list | tuple):
        return ()
    if len(raw_vertex_ids) < 3:
        return ()
    if any(type(vertex_id) is not int for vertex_id in raw_vertex_ids):
        return ()

    vertex_ids = tuple(raw_vertex_ids)
    if len(set(vertex_ids)) != len(vertex_ids):
        return ()

    existing_vertex_ids = {vertex.id for vertex in vertex_data.vertices}
    if any(vertex_id not in existing_vertex_ids for vertex_id in vertex_ids):
        return ()

    return vertex_ids


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


# ### Stair serialization helpers ###
def _serialize_stairs(
    stairs: list[StairData],
) -> list[dict[str, object]]:
    return [stair.to_dict() for stair in stairs]


def _deserialize_stairs(
    raw_stairs: object,
    valid_level_indices: set[int],
) -> list[StairData]:
    """Load complete stairs whose complete route references current levels."""

    if not isinstance(raw_stairs, list | tuple):
        return []

    stairs: list[StairData] = []
    for raw_stair in raw_stairs:
        try:
            stair = StairData.from_dict(raw_stair)
        except (TypeError, ValueError):
            continue

        if (
            stair.start_level_index not in valid_level_indices
            or stair.end_level_index not in valid_level_indices
            or any(
                section.level_index not in valid_level_indices
                for section in stair.intermediate_sections
            )
        ):
            continue
        stairs.append(stair)

    return stairs


# ### Doorway serialization helpers ###
def _serialize_doorway_presets(
    doorway_presets: list[DoorwayPreset],
) -> list[dict[str, float]]:
    return [
        {
            "width_meters": float(preset.width_meters),
            "height_meters": float(preset.height_meters),
        }
        for preset in doorway_presets
    ]


def _deserialize_doorway_presets(raw_presets: object) -> list[DoorwayPreset]:
    if not isinstance(raw_presets, list | tuple):
        return create_default_doorway_presets()

    doorway_presets: list[DoorwayPreset] = []
    for raw_preset in raw_presets:
        if not isinstance(raw_preset, dict):
            continue

        doorway_presets.append(
            DoorwayPreset(
                width_meters=_deserialize_doorway_width_meters(
                    raw_preset.get(
                        "width_meters",
                        DEFAULT_DOORWAY_WIDTH_METERS,
                    )
                ),
                height_meters=_deserialize_doorway_height_meters(
                    raw_preset.get(
                        "height_meters",
                        DEFAULT_DOORWAY_HEIGHT_METERS,
                    )
                ),
            )
        )

    return doorway_presets


def _serialize_doorway(doorway: DoorwayData) -> dict[str, float]:
    return {
        "center_x": float(doorway.center_x),
        "center_y": float(doorway.center_y),
        "width_meters": float(doorway.width_meters),
        "height_meters": float(doorway.height_meters),
        "depth_meters": float(doorway.depth_meters),
        "rotation_degrees": float(doorway.rotation_degrees),
    }


def _deserialize_doorways(raw_doorways: object) -> list[DoorwayData]:
    if not isinstance(raw_doorways, list | tuple):
        return []

    doorways: list[DoorwayData] = []
    for raw_doorway in raw_doorways:
        if not isinstance(raw_doorway, dict):
            continue

        center_x = _deserialize_doorway_coordinate(raw_doorway.get("center_x"))
        center_y = _deserialize_doorway_coordinate(raw_doorway.get("center_y"))
        if center_x is None or center_y is None:
            continue

        doorways.append(
            DoorwayData(
                center_x=center_x,
                center_y=center_y,
                width_meters=_deserialize_doorway_width_meters(
                    raw_doorway.get(
                        "width_meters",
                        DEFAULT_DOORWAY_WIDTH_METERS,
                    )
                ),
                height_meters=_deserialize_doorway_height_meters(
                    raw_doorway.get(
                        "height_meters",
                        DEFAULT_DOORWAY_HEIGHT_METERS,
                    )
                ),
                depth_meters=_deserialize_doorway_depth_meters(
                    raw_doorway.get(
                        "depth_meters",
                        DEFAULT_DOORWAY_DEPTH_METERS,
                    )
                ),
                rotation_degrees=_deserialize_doorway_rotation_degrees(
                    raw_doorway.get("rotation_degrees", 0.0)
                ),
            )
        )

    return doorways


def _deserialize_doorway_coordinate(raw_coordinate: object) -> float | None:
    if isinstance(raw_coordinate, bool):
        return None

    try:
        coordinate = float(raw_coordinate)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(coordinate):
        return None

    return coordinate


def _deserialize_doorway_width_meters(raw_width_meters: object) -> float:
    return _deserialize_doorway_measurement_meters(
        raw_width_meters,
        default_value=DEFAULT_DOORWAY_WIDTH_METERS,
        minimum=MIN_DOORWAY_WIDTH_METERS,
        maximum=MAX_DOORWAY_WIDTH_METERS,
    )


def _deserialize_doorway_height_meters(raw_height_meters: object) -> float:
    return _deserialize_doorway_measurement_meters(
        raw_height_meters,
        default_value=DEFAULT_DOORWAY_HEIGHT_METERS,
        minimum=MIN_DOORWAY_HEIGHT_METERS,
        maximum=MAX_DOORWAY_HEIGHT_METERS,
    )


def _deserialize_doorway_depth_meters(raw_depth_meters: object) -> float:
    return _deserialize_doorway_measurement_meters(
        raw_depth_meters,
        default_value=DEFAULT_DOORWAY_DEPTH_METERS,
        minimum=MIN_DOORWAY_DEPTH_METERS,
        maximum=MAX_DOORWAY_DEPTH_METERS,
    )


def _deserialize_doorway_measurement_meters(
    raw_measurement_meters: object,
    default_value: float,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(raw_measurement_meters, bool):
        return default_value

    try:
        measurement_meters = float(raw_measurement_meters)
    except (TypeError, ValueError):
        return default_value

    if not math.isfinite(measurement_meters):
        return default_value

    return min(max(measurement_meters, minimum), maximum)


def _deserialize_doorway_rotation_degrees(raw_rotation_degrees: object) -> float:
    if isinstance(raw_rotation_degrees, bool):
        return 0.0

    try:
        rotation_degrees = float(raw_rotation_degrees)
    except (TypeError, ValueError):
        return 0.0

    if not math.isfinite(rotation_degrees):
        return 0.0

    return rotation_degrees % 360.0


# ### Window serialization helpers ###
def _deserialize_windows(
    raw_windows: object,
    level_index: int,
) -> list[WindowData]:
    if not isinstance(raw_windows, list | tuple):
        return []

    windows: list[WindowData] = []
    window_ids: set[str] = set()
    for raw_window in raw_windows:
        try:
            window = WindowData.from_dict(raw_window)
        except (TypeError, ValueError):
            continue
        if not window.wall_surface_id.startswith(f"level:{level_index}/"):
            continue
        if window.window_id in window_ids:
            continue
        window_ids.add(window.window_id)
        windows.append(window)
    return windows


# ### Room serialization helpers ###
def _serialize_room(room: RoomData) -> dict[str, object]:
    return {
        "name": room.name,
        "vertex_ids": [int(vertex_id) for vertex_id in room.vertex_ids],
        "center_vertex_id": int(room.center_vertex_id),
        "color_rgb": [int(color_value) for color_value in room.color_rgb],
        "height_meters": float(room.height_meters),
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
        "wall_subdivisions": {
            str(wall_key): int(segment_count)
            for wall_key, segment_count in room.wall_subdivisions.items()
        },
        "wall_subdivision_positions": _serialize_wall_subdivision_positions(
            room.wall_subdivision_positions
        ),
        "wall_subdivision_source_ranges": _serialize_wall_subdivision_source_ranges(
            room.wall_subdivision_source_ranges
        ),
        "wall_textures": _serialize_wall_textures(room.wall_textures),
    }


def _deserialize_rooms(
    raw_rooms: object,
    default_height_meters: float = DEFAULT_ROOM_HEIGHT_METERS,
) -> list[RoomData]:
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
                height_meters=_deserialize_room_height_meters(
                    raw_room.get("height_meters", default_height_meters),
                    default_height_meters,
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
                wall_subdivisions=_deserialize_wall_subdivisions(
                    raw_room.get("wall_subdivisions", {})
                ),
                wall_subdivision_positions=_deserialize_wall_subdivision_positions(
                    raw_room.get("wall_subdivision_positions", {})
                ),
                wall_subdivision_source_ranges=(
                    _deserialize_wall_subdivision_source_ranges(
                        raw_room.get("wall_subdivision_source_ranges", {})
                    )
                ),
                wall_textures=_deserialize_wall_textures(
                    raw_room.get("wall_textures", {})
                ),
            )
        )

    return rooms


def _deserialize_room_height_meters(
    raw_height_meters: object,
    default_height_meters: float,
) -> float:
    try:
        height_meters = float(raw_height_meters)
    except (TypeError, ValueError):
        return max(0.1, float(default_height_meters))

    if height_meters <= 0.0:
        return max(0.1, float(default_height_meters))

    return height_meters


def _deserialize_floor_thickness_meters(raw_thickness_meters: object) -> float:
    try:
        thickness_meters = float(raw_thickness_meters)
    except (TypeError, ValueError):
        return DEFAULT_FLOOR_THICKNESS_METERS

    if not math.isfinite(thickness_meters):
        return DEFAULT_FLOOR_THICKNESS_METERS

    return min(
        max(thickness_meters, MIN_FLOOR_THICKNESS_METERS),
        MAX_FLOOR_THICKNESS_METERS,
    )


def _deserialize_level_scale(raw_scale: object) -> float:
    try:
        scale = float(raw_scale)
    except (TypeError, ValueError):
        return DEFAULT_LEVEL_SCALE

    if not math.isfinite(scale):
        return DEFAULT_LEVEL_SCALE

    return min(max(scale, MIN_LEVEL_SCALE), MAX_LEVEL_SCALE)


def _deserialize_level_offset_meters(raw_offset_meters: object) -> float:
    if isinstance(raw_offset_meters, bool):
        return DEFAULT_LEVEL_OFFSET_METERS

    try:
        offset_meters = float(raw_offset_meters)
    except (TypeError, ValueError):
        return DEFAULT_LEVEL_OFFSET_METERS

    if not math.isfinite(offset_meters):
        return DEFAULT_LEVEL_OFFSET_METERS

    return min(
        max(offset_meters, MIN_LEVEL_OFFSET_METERS),
        MAX_LEVEL_OFFSET_METERS,
    )


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


def _deserialize_wall_subdivisions(raw_wall_subdivisions: object) -> dict[str, int]:
    if not isinstance(raw_wall_subdivisions, dict):
        return {}

    wall_subdivisions: dict[str, int] = {}
    for wall_key, raw_segment_count in raw_wall_subdivisions.items():
        try:
            segment_count = int(raw_segment_count)
        except (TypeError, ValueError):
            continue

        wall_subdivisions[str(wall_key)] = max(1, segment_count)

    return wall_subdivisions


def _serialize_wall_subdivision_positions(
    wall_subdivision_positions: dict[str, tuple[tuple[float, float], ...]],
) -> dict[str, list[list[float]]]:
    return {
        str(wall_key): [
            [float(position[0]), float(position[1])]
            for position in segment_positions
        ]
        for wall_key, segment_positions in wall_subdivision_positions.items()
    }


def _deserialize_wall_subdivision_positions(
    raw_wall_subdivision_positions: object,
) -> dict[str, tuple[tuple[float, float], ...]]:
    if not isinstance(raw_wall_subdivision_positions, dict):
        return {}

    wall_subdivision_positions: dict[str, tuple[tuple[float, float], ...]] = {}
    for wall_key, raw_positions in raw_wall_subdivision_positions.items():
        if not isinstance(raw_positions, list | tuple):
            continue

        segment_positions: list[tuple[float, float]] = []
        for raw_position in raw_positions:
            if not isinstance(raw_position, list | tuple) or len(raw_position) != 2:
                continue

            segment_positions.append(
                (float(raw_position[0]), float(raw_position[1]))
            )

        if segment_positions:
            wall_subdivision_positions[str(wall_key)] = tuple(segment_positions)

    return wall_subdivision_positions


def _serialize_wall_subdivision_source_ranges(
    wall_subdivision_source_ranges: dict[str, tuple[tuple[float, float], ...]],
) -> dict[str, list[list[float]]]:
    return {
        str(wall_key): [
            [float(source_range[0]), float(source_range[1])]
            for source_range in source_ranges
        ]
        for wall_key, source_ranges in wall_subdivision_source_ranges.items()
    }


def _deserialize_wall_subdivision_source_ranges(
    raw_wall_subdivision_source_ranges: object,
) -> dict[str, tuple[tuple[float, float], ...]]:
    if not isinstance(raw_wall_subdivision_source_ranges, dict):
        return {}

    wall_subdivision_source_ranges: dict[str, tuple[tuple[float, float], ...]] = {}
    for wall_key, raw_ranges in raw_wall_subdivision_source_ranges.items():
        if not isinstance(raw_ranges, list | tuple):
            continue

        source_ranges: list[tuple[float, float]] = []
        for raw_range in raw_ranges:
            if not isinstance(raw_range, list | tuple) or len(raw_range) != 2:
                continue

            source_start = min(max(0.0, float(raw_range[0])), 1.0)
            source_end = min(max(source_start, float(raw_range[1])), 1.0)
            if source_end <= source_start:
                continue

            source_ranges.append((source_start, source_end))

        if source_ranges:
            wall_subdivision_source_ranges[str(wall_key)] = tuple(source_ranges)

    return wall_subdivision_source_ranges


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
