# ### Imports ###
from __future__ import annotations

import os
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
from PySide6.QtCore import QByteArray, QBuffer, QIODevice, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter, QPen
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.models import (
    DEFAULT_LEVEL_HEIGHT_METERS,
    GROUND_LEVEL_INDEX,
    Edge,
    LevelData,
    PIXEL_TO_METER,
    RoomData,
    Vertex,
    VertexData,
)
from housemaker.uv_layout import (
    UvLayout,
    UvWallPlacement,
    build_room_walls,
    build_uv_wall_layout,
)

# ### Constants ###
DEFAULT_WALL_HEIGHT_METERS = DEFAULT_LEVEL_HEIGHT_METERS
Z_UP_TO_GLTF_Y_UP_TRANSFORM = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)
ROOM_TEXTURE_BACKGROUND_COLOR = QColor("#f7f8fb")
ROOM_TEXTURE_WALL_FILL_COLOR = QColor("#dce0e8")
ROOM_TEXTURE_WALL_BORDER_COLOR = QColor("#d92d20")
ROOM_TEXTURE_TEXT_COLOR = QColor("#20242a")
ROOM_TEXTURE_INDICATOR_BACKGROUND_COLOR = QColor(10, 12, 16, 180)
ROOM_TEXTURE_INDICATOR_TEXT_COLOR = QColor("#f5f7fa")
ROOM_TEXTURE_WALL_BORDER_WIDTH = 2.0
ROOM_TEXTURE_MIN_FONT_SIZE = 8
ROOM_TEXTURE_MAX_FONT_SIZE = 32
FALLBACK_QT_PLATFORM = "offscreen"

# ### Module state ###
_fallback_qt_application: QGuiApplication | None = None

# ### Data models ###
@dataclass
class GeneratedModel:
    mesh: trimesh.Trimesh
    scene: trimesh.Scene
    glb_bytes: bytes
    preview_textured_walls: list["PreviewTexturedWall"] = field(default_factory=list)


@dataclass
class NamedMesh:
    name: str
    mesh: trimesh.Trimesh


@dataclass(frozen=True)
class PreviewTexturedWall:
    start_point: tuple[float, float, float]
    end_point: tuple[float, float, float]
    height_meters: float
    texture_rgba: np.ndarray


@dataclass(frozen=True)
class PngTexture:
    png_bytes: bytes
    format: str = "PNG"

    def save(self, file_object, format: str | None = None) -> None:
        file_object.write(self.png_bytes)

    def copy(self) -> "PngTexture":
        return PngTexture(png_bytes=bytes(self.png_bytes), format=self.format)

    def __array__(self, dtype=None):
        texture_array = np.frombuffer(self.png_bytes, dtype=np.uint8)
        if dtype is None:
            return texture_array

        return texture_array.astype(dtype)


# ### Public helpers ###
def convert_to_glb(
    level_source: VertexData | Sequence[LevelData],
    wall_height_meters: float = DEFAULT_WALL_HEIGHT_METERS,
    blueprint_size_pixels: tuple[float, float] | None = None,
) -> GeneratedModel:
    if isinstance(level_source, VertexData):
        preview_textured_walls: list[PreviewTexturedWall] = []
        wall_meshes = _build_level_meshes(
            vertex_data=level_source,
            wall_height_meters=wall_height_meters,
            base_z_meters=0.0,
            blueprint_size_pixels=blueprint_size_pixels,
        )
        named_meshes = _build_named_meshes_for_single_level(wall_meshes)
    else:
        named_meshes = _build_multi_level_meshes(
            level_source,
            blueprint_size_pixels=blueprint_size_pixels,
        )
        preview_textured_walls = _build_preview_textured_walls(
            level_source,
            blueprint_size_pixels=blueprint_size_pixels,
        )
        wall_meshes = [named_mesh.mesh for named_mesh in named_meshes]

    if not wall_meshes:
        raise ValueError("The current blueprint data does not contain usable edges.")

    combined_mesh = _combine_mesh_geometry(wall_meshes)
    scene = _build_export_scene(named_meshes)
    glb_bytes = scene.export(file_type="glb")
    return GeneratedModel(
        mesh=combined_mesh,
        scene=scene,
        glb_bytes=glb_bytes,
        preview_textured_walls=preview_textured_walls,
    )


def export_glb_file(model: GeneratedModel, path: str | Path) -> Path:
    export_path = Path(path)
    export_path.write_bytes(model.glb_bytes)
    return export_path


# ### Internal helpers ###
def _build_export_scene(named_meshes: list[NamedMesh]) -> trimesh.Scene:
    scene = trimesh.Scene()
    for named_mesh in named_meshes:
        scene.add_geometry(
            _to_gltf_y_up_mesh(named_mesh.mesh),
            geom_name=named_mesh.name,
            node_name=named_mesh.name,
        )
    return scene


def _to_gltf_y_up_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    export_mesh = mesh.copy()
    export_mesh.apply_transform(Z_UP_TO_GLTF_Y_UP_TRANSFORM)
    return export_mesh


def _build_named_meshes_for_single_level(
    wall_meshes: list[trimesh.Trimesh],
) -> list[NamedMesh]:
    if not wall_meshes:
        return []

    return [
        NamedMesh(
            name="level_2_ground",
            mesh=_combine_mesh_geometry(wall_meshes),
        )
    ]


def _build_multi_level_meshes(
    levels: Sequence[LevelData],
    blueprint_size_pixels: tuple[float, float] | None,
) -> list[NamedMesh]:
    if not levels:
        raise ValueError("No levels are available for GLB conversion.")

    sorted_levels = sorted(levels, key=lambda level: level.index)
    level_lookup = {level.index: level for level in sorted_levels}
    named_meshes: list[NamedMesh] = []

    for level in sorted_levels:
        if not level.include_in_export:
            continue

        if level.height_meters <= 0.0:
            raise ValueError(f"Level {level.index} height must be greater than zero.")

        level_named_meshes = _build_named_meshes_for_level(
            level=level,
            level_lookup=level_lookup,
            blueprint_size_pixels=blueprint_size_pixels,
        )
        if not level_named_meshes:
            continue

        named_meshes.extend(level_named_meshes)

    return named_meshes


def _build_named_meshes_for_level(
    level: LevelData,
    level_lookup: dict[int, LevelData],
    blueprint_size_pixels: tuple[float, float] | None,
) -> list[NamedMesh]:
    base_z_meters = _get_level_base_z(level_lookup, level.index)
    level_blueprint_size = level.image_size_pixels or blueprint_size_pixels
    room_vertex_sets = _get_room_vertex_sets(level.rooms)
    named_meshes = _build_room_named_meshes(
        level=level,
        base_z_meters=base_z_meters,
        blueprint_size_pixels=level_blueprint_size,
    )
    regular_wall_meshes = _build_level_meshes(
        vertex_data=level.vertex_data,
        wall_height_meters=level.height_meters,
        base_z_meters=base_z_meters,
        blueprint_size_pixels=level_blueprint_size,
        ignored_vertex_ids=_get_room_center_vertex_ids(level.rooms),
        ignored_room_vertex_sets=room_vertex_sets,
    )

    if regular_wall_meshes:
        named_meshes.insert(
            0,
            NamedMesh(
                name=_get_level_object_name(level),
                mesh=_combine_mesh_geometry(regular_wall_meshes),
            ),
        )

    return named_meshes


def _build_room_named_meshes(
    level: LevelData,
    base_z_meters: float,
    blueprint_size_pixels: tuple[float, float] | None,
) -> list[NamedMesh]:
    named_meshes: list[NamedMesh] = []
    for room_index, room in enumerate(level.rooms):
        room_mesh = _build_room_mesh(
            level=level,
            room=room,
            room_index=room_index,
            base_z_meters=base_z_meters,
            blueprint_size_pixels=blueprint_size_pixels,
        )
        if room_mesh is None:
            continue

        named_meshes.append(
            NamedMesh(
                name=_get_room_object_name(level, room, room_index),
                mesh=room_mesh,
            )
        )

    return named_meshes


def _build_level_meshes(
    vertex_data: VertexData,
    wall_height_meters: float,
    base_z_meters: float,
    blueprint_size_pixels: tuple[float, float] | None,
    ignored_vertex_ids: set[int] | None = None,
    ignored_room_vertex_sets: list[set[int]] | None = None,
) -> list[trimesh.Trimesh]:
    if wall_height_meters <= 0.0:
        raise ValueError("Height level must be greater than zero.")

    ignored_ids = ignored_vertex_ids or set()
    room_vertex_sets = ignored_room_vertex_sets or []
    vertex_lookup = {vertex.id: vertex for vertex in vertex_data.vertices}
    return [
        wall_mesh
        for edge in vertex_data.edges
        if (
            edge.start_vertex_id not in ignored_ids
            and edge.end_vertex_id not in ignored_ids
            and not _is_edge_inside_any_room(edge, room_vertex_sets)
        )
        if (
            wall_mesh := _build_wall_mesh(
                edge=edge,
                vertex_lookup=vertex_lookup,
                wall_height_meters=wall_height_meters,
                base_z_meters=base_z_meters,
                blueprint_size_pixels=blueprint_size_pixels,
            )
        )
        is not None
    ]


def _build_room_mesh(
    level: LevelData,
    room: RoomData,
    room_index: int,
    base_z_meters: float,
    blueprint_size_pixels: tuple[float, float] | None,
) -> trimesh.Trimesh | None:
    room_walls = build_room_walls(room, level.vertex_data)
    if not room_walls:
        return None

    layout = build_uv_wall_layout(
        room=room,
        vertex_data=level.vertex_data,
        wall_height_meters=level.height_meters,
    )
    placements_by_key = {
        placement.wall.key: placement
        for placement in layout.placements
    }
    material = _build_room_material(level, room, room_index, layout)
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    uv_coordinates: list[tuple[float, float]] = []

    for wall in room_walls:
        placement = placements_by_key.get(wall.key)
        wall_vertices = _build_wall_vertices_from_points(
            start_point=wall.start_point,
            end_point=wall.end_point,
            wall_height_meters=level.height_meters,
            base_z_meters=base_z_meters,
            blueprint_size_pixels=blueprint_size_pixels,
        )
        if wall_vertices is None:
            continue

        vertex_offset = len(vertices)
        vertices.extend(wall_vertices)
        faces.extend(_build_wall_faces(vertex_offset))
        if placement is None:
            uv_coordinates.extend(_build_hidden_wall_uv_coordinates())
        else:
            uv_coordinates.extend(_build_wall_uv_coordinates(room, placement))

    if not vertices or not faces:
        return None

    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
        visual=TextureVisuals(
            uv=np.asarray(uv_coordinates, dtype=float),
            material=material,
        ),
        process=False,
    )


# ### Material helpers ###
def _build_room_material(
    level: LevelData,
    room: RoomData,
    room_index: int,
    layout: UvLayout,
) -> PBRMaterial:
    return PBRMaterial(
        name=_get_room_material_name(level, room, room_index),
        baseColorFactor=[255, 255, 255, 255],
        baseColorTexture=_build_room_texture(room, layout),
        metallicFactor=0.0,
        roughnessFactor=0.65,
        doubleSided=True,
    )


def _build_room_texture(room: RoomData, layout: UvLayout) -> PngTexture:
    image = _build_room_texture_image(room, layout)
    return PngTexture(png_bytes=_qimage_to_png_bytes(image))


def _build_room_texture_image(room: RoomData, layout: UvLayout) -> QImage:
    _ensure_qt_application()
    texture_width = max(1, int(room.uv_map_width))
    texture_height = max(1, int(room.uv_map_height))
    image = QImage(
        texture_width,
        texture_height,
        QImage.Format.Format_RGBA8888,
    )
    image.fill(ROOM_TEXTURE_BACKGROUND_COLOR)

    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    for placement in layout.placements:
        _paint_room_texture_wall(painter, placement)

    if layout.hidden_wall_count > 0:
        _paint_room_texture_hidden_indicator(
            painter=painter,
            texture_width=texture_width,
            texture_height=texture_height,
            hidden_wall_count=layout.hidden_wall_count,
        )

    painter.end()
    return image


def _ensure_qt_application() -> None:
    global _fallback_qt_application

    if QGuiApplication.instance() is not None:
        return

    os.environ.setdefault("QT_QPA_PLATFORM", FALLBACK_QT_PLATFORM)
    _fallback_qt_application = QGuiApplication([])


# ### Texture helpers ###
def _paint_room_texture_wall(
    painter: QPainter,
    placement: UvWallPlacement,
) -> None:
    uv_x, uv_y, uv_width, uv_height = placement.uv_rect
    texture_rect = QRectF(uv_x, uv_y, uv_width, uv_height).adjusted(
        0.5,
        0.5,
        -0.5,
        -0.5,
    )
    painter.setPen(QPen(ROOM_TEXTURE_WALL_BORDER_COLOR, ROOM_TEXTURE_WALL_BORDER_WIDTH))
    painter.setBrush(ROOM_TEXTURE_WALL_FILL_COLOR)
    painter.drawRect(texture_rect)

    painter.setPen(QPen(ROOM_TEXTURE_TEXT_COLOR))
    painter.setFont(QFont("Segoe UI", _get_room_texture_label_font_size(texture_rect)))
    painter.drawText(
        texture_rect,
        int(Qt.AlignmentFlag.AlignCenter),
        f"{placement.wall.projection_direction}\n{placement.rotation_degrees} deg",
    )


def _paint_room_texture_hidden_indicator(
    painter: QPainter,
    texture_width: int,
    texture_height: int,
    hidden_wall_count: int,
) -> None:
    indicator_width = min(190.0, max(1.0, texture_width - 20.0))
    indicator_rect = QRectF(
        10.0,
        max(0.0, texture_height - 34.0),
        indicator_width,
        24.0,
    )
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(ROOM_TEXTURE_INDICATOR_BACKGROUND_COLOR)
    painter.drawRoundedRect(indicator_rect, 6.0, 6.0)
    painter.setPen(QPen(ROOM_TEXTURE_INDICATOR_TEXT_COLOR))
    painter.setFont(QFont("Segoe UI", 9))
    painter.drawText(
        indicator_rect,
        int(Qt.AlignmentFlag.AlignCenter),
        f"{hidden_wall_count} walls are not shown",
    )


def _get_room_texture_label_font_size(texture_rect: QRectF) -> int:
    raw_size = int(min(texture_rect.width(), texture_rect.height()) / 4.0)
    return min(
        ROOM_TEXTURE_MAX_FONT_SIZE,
        max(ROOM_TEXTURE_MIN_FONT_SIZE, raw_size),
    )


def _qimage_to_png_bytes(image: QImage) -> bytes:
    byte_array = QByteArray()
    buffer = QBuffer(byte_array)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    buffer.close()
    return bytes(byte_array)


def _qimage_to_gl_rgba_array(image: QImage) -> np.ndarray:
    converted_image = image.convertToFormat(QImage.Format.Format_RGBA8888)
    image_width = converted_image.width()
    image_height = converted_image.height()
    bytes_per_line = converted_image.bytesPerLine()
    image_buffer = converted_image.bits()
    image_array = np.frombuffer(
        image_buffer,
        dtype=np.uint8,
        count=bytes_per_line * image_height,
    )
    image_array = image_array.reshape((image_height, bytes_per_line))
    image_array = image_array[:, : image_width * 4].reshape(
        (image_height, image_width, 4)
    )
    return np.flip(np.swapaxes(image_array, 0, 1), axis=1).copy()


def _crop_placement_texture(
    texture_image: QImage,
    placement: UvWallPlacement,
) -> np.ndarray:
    uv_x, uv_y, uv_width, uv_height = placement.uv_rect
    crop_x = max(0, int(math.floor(uv_x)))
    crop_y = max(0, int(math.floor(uv_y)))
    crop_right = min(texture_image.width(), int(math.ceil(uv_x + uv_width)))
    crop_bottom = min(texture_image.height(), int(math.ceil(uv_y + uv_height)))
    crop_width = max(1, crop_right - crop_x)
    crop_height = max(1, crop_bottom - crop_y)
    cropped_image = texture_image.copy(crop_x, crop_y, crop_width, crop_height)
    texture_rgba = _qimage_to_gl_rgba_array(cropped_image)
    return _rotate_preview_texture(texture_rgba, placement.rotation_degrees)


def _rotate_preview_texture(
    texture_rgba: np.ndarray,
    rotation_degrees: int,
) -> np.ndarray:
    normalized_rotation = int(rotation_degrees) % 360
    if normalized_rotation == 90:
        return np.rot90(texture_rgba, k=-1, axes=(0, 1)).copy()
    if normalized_rotation == 180:
        return np.rot90(texture_rgba, k=2, axes=(0, 1)).copy()
    if normalized_rotation == 270:
        return np.rot90(texture_rgba, k=1, axes=(0, 1)).copy()

    return texture_rgba


# ### Wall geometry helpers ###
def _build_wall_vertices_from_points(
    start_point: tuple[float, float],
    end_point: tuple[float, float],
    wall_height_meters: float,
    base_z_meters: float,
    blueprint_size_pixels: tuple[float, float] | None,
) -> list[list[float]] | None:
    start_xy = _point_to_world_xy(start_point, blueprint_size_pixels)
    end_xy = _point_to_world_xy(end_point, blueprint_size_pixels)
    if float(np.linalg.norm(end_xy - start_xy)) < 1e-6:
        return None

    bottom_z = base_z_meters
    top_z = base_z_meters + wall_height_meters
    return [
        [float(start_xy[0]), float(start_xy[1]), bottom_z],
        [float(end_xy[0]), float(end_xy[1]), bottom_z],
        [float(end_xy[0]), float(end_xy[1]), top_z],
        [float(start_xy[0]), float(start_xy[1]), top_z],
    ]


def _build_wall_faces(vertex_offset: int) -> list[list[int]]:
    return [
        [vertex_offset + 0, vertex_offset + 1, vertex_offset + 2],
        [vertex_offset + 0, vertex_offset + 2, vertex_offset + 3],
        [vertex_offset + 2, vertex_offset + 1, vertex_offset + 0],
        [vertex_offset + 3, vertex_offset + 2, vertex_offset + 0],
    ]


def _build_wall_uv_coordinates(
    room: RoomData,
    placement: UvWallPlacement,
) -> list[tuple[float, float]]:
    uv_x, uv_y, uv_width, uv_height = placement.uv_rect
    texture_width = max(1.0, float(room.uv_map_width))
    texture_height = max(1.0, float(room.uv_map_height))
    left = uv_x / texture_width
    right = (uv_x + uv_width) / texture_width
    top = 1.0 - (uv_y / texture_height)
    bottom = 1.0 - ((uv_y + uv_height) / texture_height)
    top_left = (left, top)
    top_right = (right, top)
    bottom_right = (right, bottom)
    bottom_left = (left, bottom)

    if placement.rotation_degrees == 90:
        return [bottom_right, top_right, top_left, bottom_left]
    if placement.rotation_degrees == 180:
        return [top_right, top_left, bottom_left, bottom_right]
    if placement.rotation_degrees == 270:
        return [top_left, bottom_left, bottom_right, top_right]

    return [bottom_left, bottom_right, top_right, top_left]


def _build_hidden_wall_uv_coordinates() -> list[tuple[float, float]]:
    return [(0.0, 0.0)] * 4


# ### Preview helpers ###
def _build_preview_textured_walls(
    levels: Sequence[LevelData],
    blueprint_size_pixels: tuple[float, float] | None,
) -> list[PreviewTexturedWall]:
    sorted_levels = sorted(levels, key=lambda level: level.index)
    level_lookup = {level.index: level for level in sorted_levels}
    preview_walls: list[PreviewTexturedWall] = []

    for level in sorted_levels:
        if not level.include_in_export:
            continue

        preview_walls.extend(
            _build_level_preview_textured_walls(
                level=level,
                base_z_meters=_get_level_base_z(level_lookup, level.index),
                blueprint_size_pixels=level.image_size_pixels or blueprint_size_pixels,
            )
        )

    return preview_walls


def _build_level_preview_textured_walls(
    level: LevelData,
    base_z_meters: float,
    blueprint_size_pixels: tuple[float, float] | None,
) -> list[PreviewTexturedWall]:
    preview_walls: list[PreviewTexturedWall] = []
    for room in level.rooms:
        layout = build_uv_wall_layout(
            room=room,
            vertex_data=level.vertex_data,
            wall_height_meters=level.height_meters,
        )
        if not layout.placements:
            continue

        texture_image = _build_room_texture_image(room, layout)
        for placement in layout.placements:
            start_xy = _point_to_world_xy(
                placement.wall.start_point,
                blueprint_size_pixels,
            )
            end_xy = _point_to_world_xy(
                placement.wall.end_point,
                blueprint_size_pixels,
            )
            preview_walls.append(
                PreviewTexturedWall(
                    start_point=(
                        float(start_xy[0]),
                        float(start_xy[1]),
                        base_z_meters,
                    ),
                    end_point=(
                        float(end_xy[0]),
                        float(end_xy[1]),
                        base_z_meters,
                    ),
                    height_meters=float(level.height_meters),
                    texture_rgba=_crop_placement_texture(texture_image, placement),
                )
            )

    return preview_walls


# ### Room helpers ###
def _get_room_vertex_sets(rooms: list[RoomData]) -> list[set[int]]:
    return [set(room.vertex_ids) for room in rooms]


def _is_edge_inside_any_room(
    edge: Edge,
    room_vertex_sets: list[set[int]],
) -> bool:
    return any(
        edge.start_vertex_id in room_vertex_set
        and edge.end_vertex_id in room_vertex_set
        for room_vertex_set in room_vertex_sets
    )


def _get_room_center_vertex_ids(rooms: list[RoomData]) -> set[int]:
    return {room.center_vertex_id for room in rooms}


# ### Level helpers ###
def _get_level_base_z(
    level_lookup: dict[int, LevelData],
    level_index: int,
) -> float:
    if level_index >= GROUND_LEVEL_INDEX:
        return sum(
            level_lookup[index].height_meters
            for index in range(GROUND_LEVEL_INDEX, level_index)
            if index in level_lookup
        )

    return -sum(
        level_lookup[index].height_meters
        for index in range(level_index, GROUND_LEVEL_INDEX)
        if index in level_lookup
    )


def _get_level_object_name(level: LevelData) -> str:
    return level.display_name.lower().replace(" ", "_")


def _get_room_object_name(
    level: LevelData,
    room: RoomData,
    room_index: int,
) -> str:
    return f"{_get_level_object_name(level)}_{_slugify_name(room.name)}_{room_index + 1}"


def _get_room_material_name(
    level: LevelData,
    room: RoomData,
    room_index: int,
) -> str:
    return f"{level.display_name} {room.name or 'Room'} {room_index + 1}"


# ### Plain wall helpers ###
def _build_wall_mesh(
    edge: Edge,
    vertex_lookup: dict[int, Vertex],
    wall_height_meters: float,
    base_z_meters: float,
    blueprint_size_pixels: tuple[float, float] | None,
) -> trimesh.Trimesh | None:
    start_vertex = vertex_lookup.get(edge.start_vertex_id)
    end_vertex = vertex_lookup.get(edge.end_vertex_id)
    if start_vertex is None or end_vertex is None:
        return None

    start_xy = _vertex_to_world_xy(start_vertex, blueprint_size_pixels)
    end_xy = _vertex_to_world_xy(end_vertex, blueprint_size_pixels)

    direction = end_xy - start_xy
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return None

    wall_mesh = trimesh.Trimesh(
        vertices=[
            [0.0, 0.0, -wall_height_meters / 2.0],
            [length, 0.0, -wall_height_meters / 2.0],
            [length, 0.0, wall_height_meters / 2.0],
            [0.0, 0.0, wall_height_meters / 2.0],
        ],
        faces=[
            [0, 1, 2],
            [0, 2, 3],
            [2, 1, 0],
            [3, 2, 0],
        ],
        process=False,
    )
    angle_radians = float(np.arctan2(direction[1], direction[0]))
    transform = trimesh.transformations.rotation_matrix(angle_radians, [0.0, 0.0, 1.0])
    transform[:3, 3] = [
        float(start_xy[0]),
        float(start_xy[1]),
        base_z_meters + wall_height_meters / 2.0,
    ]

    wall_mesh.apply_transform(transform)
    return wall_mesh


# ### Coordinate helpers ###
def _vertex_to_world_xy(
    vertex: Vertex,
    blueprint_size_pixels: tuple[float, float] | None,
) -> np.ndarray:
    return _point_to_world_xy((vertex.x, vertex.y), blueprint_size_pixels)


def _point_to_world_xy(
    point: tuple[float, float],
    blueprint_size_pixels: tuple[float, float] | None,
) -> np.ndarray:
    point_x, point_y = point
    if blueprint_size_pixels is None:
        centered_x = point_x
        centered_y = point_y
    else:
        blueprint_width, blueprint_height = blueprint_size_pixels
        centered_x = point_x - blueprint_width / 2.0
        centered_y = point_y - blueprint_height / 2.0

    return np.array(
        [
            centered_x * PIXEL_TO_METER,
            -centered_y * PIXEL_TO_METER,
        ],
        dtype=float,
    )


# ### Mesh helpers ###
def _combine_mesh_geometry(meshes: Sequence[trimesh.Trimesh]) -> trimesh.Trimesh:
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    vertex_offset = 0
    for mesh in meshes:
        mesh_vertices = np.asarray(mesh.vertices, dtype=float)
        mesh_faces = np.asarray(mesh.faces, dtype=np.int64)
        if mesh_vertices.size == 0 or mesh_faces.size == 0:
            continue

        vertices.append(mesh_vertices)
        faces.append(mesh_faces + vertex_offset)
        vertex_offset += len(mesh_vertices)

    if not vertices or not faces:
        return trimesh.Trimesh(process=False)

    return trimesh.Trimesh(
        vertices=np.vstack(vertices),
        faces=np.vstack(faces),
        process=False,
    )


# ### Text helpers ###
def _slugify_name(name: str) -> str:
    normalized_name = "".join(
        character.lower() if character.isalnum() else "_"
        for character in name.strip()
    ).strip("_")
    return normalized_name or "room"
