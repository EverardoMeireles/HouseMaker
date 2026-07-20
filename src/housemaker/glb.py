# ### Imports ###
from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
from PySide6.QtCore import QByteArray, QBuffer, QIODevice, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter, QPen
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.floor_geometry import build_level_floor_mesh
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
from housemaker.texture_mapping import paint_wall_texture_crop
from housemaker.uv_layout import (
    RoomWall,
    UvLayout,
    UvWallPlacement,
    build_room_walls,
    build_uv_wall_layout,
    get_rotated_uv_corners,
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
GLTF_Y_UP_TO_Z_UP_TRANSFORM = np.linalg.inv(Z_UP_TO_GLTF_Y_UP_TRANSFORM)
ROOM_TEXTURE_BACKGROUND_COLOR = QColor("#f7f8fb")
ROOM_TEXTURE_WALL_FILL_COLOR = QColor("#dce0e8")
ROOM_TEXTURE_TEXT_COLOR = QColor("#20242a")
ROOM_TEXTURE_INDICATOR_BACKGROUND_COLOR = QColor(10, 12, 16, 180)
ROOM_TEXTURE_INDICATOR_TEXT_COLOR = QColor("#f5f7fa")
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
    source_transform: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=float)
    )


@dataclass(frozen=True)
class PreviewTexturedWall:
    level_index: int
    room_index: int
    wall_key: str
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

    if not named_meshes:
        raise ValueError("The current blueprint data does not contain usable edges.")

    combined_mesh = _combine_mesh_geometry(
        [
            _build_transformed_named_mesh_copy(named_mesh)
            for named_mesh in named_meshes
        ]
    )
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


def export_room_texture_pngs(
    levels: Sequence[LevelData],
    directory: str | Path,
) -> list[Path]:
    export_directory = Path(directory)
    export_directory.mkdir(parents=True, exist_ok=True)
    exported_paths: list[Path] = []

    for level in levels:
        for room_index, room in enumerate(level.rooms):
            if not build_room_walls(room, level.vertex_data):
                continue

            layout = build_uv_wall_layout(
                room=room,
                vertex_data=level.vertex_data,
                wall_height_meters=room.height_meters,
            )
            texture_path = export_directory / _get_room_texture_file_name(
                level=level,
                room=room,
                room_index=room_index,
            )
            texture_image = _build_room_texture_image(room, layout)
            if not texture_image.save(str(texture_path), "PNG"):
                raise OSError(f"Unable to save PNG texture: {texture_path}")

            exported_paths.append(texture_path)

    return exported_paths


# ### Internal helpers ###
def _build_export_scene(named_meshes: list[NamedMesh]) -> trimesh.Scene:
    scene = trimesh.Scene()
    for named_mesh in named_meshes:
        scene.add_geometry(
            _to_gltf_y_up_mesh(named_mesh.mesh),
            geom_name=named_mesh.name,
            node_name=named_mesh.name,
            transform=_source_to_gltf_y_up_transform(
                named_mesh.source_transform
            ),
        )
    return scene


def _to_gltf_y_up_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    export_mesh = mesh.copy()
    export_mesh.apply_transform(Z_UP_TO_GLTF_Y_UP_TRANSFORM)
    return export_mesh


def _build_transformed_named_mesh_copy(named_mesh: NamedMesh) -> trimesh.Trimesh:
    transformed_mesh = named_mesh.mesh.copy()
    transformed_mesh.apply_transform(
        _get_valid_source_transform(named_mesh.source_transform)
    )
    return transformed_mesh


def _source_to_gltf_y_up_transform(source_transform: np.ndarray) -> np.ndarray:
    valid_source_transform = _get_valid_source_transform(source_transform)
    return (
        Z_UP_TO_GLTF_Y_UP_TRANSFORM
        @ valid_source_transform
        @ GLTF_Y_UP_TO_Z_UP_TRANSFORM
    )


def _get_valid_source_transform(source_transform: np.ndarray) -> np.ndarray:
    try:
        transform = np.asarray(source_transform, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Mesh source transform must be a 4 by 4 matrix.") from error

    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("Mesh source transform must be a finite 4 by 4 matrix.")

    return transform


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

        if not math.isfinite(level.height_meters) or level.height_meters <= 0.0:
            raise ValueError(f"Level {level.index} height must be greater than zero.")
        if (
            not math.isfinite(level.floor_thickness_meters)
            or level.floor_thickness_meters <= 0.0
        ):
            raise ValueError(
                f"Level {level.index} floor thickness must be greater than zero."
            )
        _get_valid_level_scale(level)

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
    level_source_transform = _build_level_scale_source_transform(
        level,
        level_blueprint_size,
    )
    room_vertex_sets = _get_room_vertex_sets(level.rooms)
    named_meshes: list[NamedMesh] = []
    floor_mesh = build_level_floor_mesh(
        level=level,
        floor_surface_z_meters=base_z_meters,
        blueprint_size_pixels=level_blueprint_size,
        point_to_world_xy=_point_to_world_xy,
    )
    if floor_mesh is not None:
        named_meshes.append(
            NamedMesh(
                name=_get_level_floor_object_name(level),
                mesh=floor_mesh,
            )
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
        named_meshes.append(
            NamedMesh(
                name=_get_level_object_name(level),
                mesh=_combine_mesh_geometry(regular_wall_meshes),
            )
        )

    named_meshes.extend(
        _build_room_named_meshes(
            level=level,
            base_z_meters=base_z_meters,
            blueprint_size_pixels=level_blueprint_size,
        )
    )

    for named_mesh in named_meshes:
        named_mesh.source_transform = level_source_transform.copy()

    return named_meshes


def _build_room_named_meshes(
    level: LevelData,
    base_z_meters: float,
    blueprint_size_pixels: tuple[float, float] | None,
) -> list[NamedMesh]:
    named_meshes: list[NamedMesh] = []
    for room_index, room in enumerate(level.rooms):
        if room.height_meters <= 0.0:
            raise ValueError(
                f"Room {room.name or room_index + 1} height must be greater "
                "than zero."
            )

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
        wall_height_meters=room.height_meters,
    )
    placements_by_key = _group_wall_placements_by_key(layout.placements)
    material = _build_room_material(level, room, room_index, layout)
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    uv_coordinates: list[tuple[float, float]] = []

    for wall in room_walls:
        wall_placements = placements_by_key.get(wall.key, [])
        if not wall_placements:
            wall_vertices = _build_wall_vertices_from_points(
                start_point=wall.start_point,
                end_point=wall.end_point,
                wall_height_meters=room.height_meters,
                base_z_meters=base_z_meters,
                blueprint_size_pixels=blueprint_size_pixels,
            )
            if wall_vertices is None:
                continue

            vertex_offset = len(vertices)
            vertices.extend(wall_vertices)
            faces.extend(_build_wall_faces(vertex_offset))
            uv_coordinates.extend(_build_hidden_wall_uv_coordinates())
            continue

        for placement in wall_placements:
            segment_start_point, segment_end_point = _get_wall_segment_points(
                wall=wall,
                placement=placement,
            )
            wall_vertices = _build_wall_vertices_from_points(
                start_point=segment_start_point,
                end_point=segment_end_point,
                wall_height_meters=room.height_meters,
                base_z_meters=base_z_meters,
                blueprint_size_pixels=blueprint_size_pixels,
            )
            if wall_vertices is None:
                continue

            vertex_offset = len(vertices)
            vertices.extend(wall_vertices)
            faces.extend(_build_wall_faces(vertex_offset))
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


def _group_wall_placements_by_key(
    placements: Sequence[UvWallPlacement],
) -> dict[str, list[UvWallPlacement]]:
    placements_by_key: dict[str, list[UvWallPlacement]] = {}
    for placement in placements:
        placements_by_key.setdefault(placement.wall.key, []).append(placement)

    return placements_by_key


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
        _paint_room_texture_wall(painter, room, placement)

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
    room: RoomData,
    placement: UvWallPlacement,
) -> None:
    uv_x, uv_y, uv_width, uv_height = placement.uv_rect
    wall_width, wall_height = placement.natural_size
    texture_rect = QRectF(
        -wall_width / 2.0,
        -wall_height / 2.0,
        wall_width,
        wall_height,
    ).adjusted(0.5, 0.5, -0.5, -0.5)

    painter.save()
    painter.translate(uv_x + uv_width / 2.0, uv_y + uv_height / 2.0)
    painter.rotate(placement.rotation_degrees)
    texture_data = room.wall_textures.get(placement.wall.key)
    did_paint_texture = (
        texture_data is not None
        and paint_wall_texture_crop(
            painter,
            texture_data,
            texture_rect,
            placement.source_start_ratio,
            placement.source_end_ratio,
        )
    )

    if not did_paint_texture:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(ROOM_TEXTURE_WALL_FILL_COLOR)
        painter.drawRect(texture_rect)
        painter.setPen(QPen(ROOM_TEXTURE_TEXT_COLOR))
        painter.setFont(
            QFont("Segoe UI", _get_room_texture_label_font_size(texture_rect))
        )
        painter.drawText(
            texture_rect,
            int(Qt.AlignmentFlag.AlignCenter),
            f"{placement.wall.projection_direction}\n{placement.rotation_degrees} deg",
        )
    painter.restore()


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


def _build_wall_preview_texture(
    room: RoomData,
    placement: UvWallPlacement,
) -> np.ndarray:
    _ensure_qt_application()
    wall_width, wall_height = placement.natural_size
    texture_width = max(1, int(math.ceil(wall_width)))
    texture_height = max(1, int(math.ceil(wall_height)))
    image = QImage(
        texture_width,
        texture_height,
        QImage.Format.Format_RGBA8888,
    )
    image.fill(ROOM_TEXTURE_WALL_FILL_COLOR)

    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    texture_rect = QRectF(
        0.5,
        0.5,
        max(1.0, texture_width - 1.0),
        max(1.0, texture_height - 1.0),
    )
    texture_data = room.wall_textures.get(placement.wall.key)
    did_paint_texture = (
        texture_data is not None
        and paint_wall_texture_crop(
            painter,
            texture_data,
            texture_rect,
            placement.source_start_ratio,
            placement.source_end_ratio,
        )
    )
    if not did_paint_texture:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(ROOM_TEXTURE_WALL_FILL_COLOR)
        painter.drawRect(texture_rect)
        painter.setPen(QPen(ROOM_TEXTURE_TEXT_COLOR))
        painter.setFont(
            QFont("Segoe UI", _get_room_texture_label_font_size(texture_rect))
        )
        painter.drawText(
            texture_rect,
            int(Qt.AlignmentFlag.AlignCenter),
            f"{placement.wall.projection_direction}\n{placement.rotation_degrees} deg",
        )
    painter.end()
    return _qimage_to_gl_rgba_array(image)


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


def _get_wall_segment_points(
    wall: RoomWall,
    placement: UvWallPlacement,
) -> tuple[tuple[float, float], tuple[float, float]]:
    return (
        _interpolate_2d_point(
            wall.start_point,
            wall.end_point,
            placement.source_start_ratio,
        ),
        _interpolate_2d_point(
            wall.start_point,
            wall.end_point,
            placement.source_end_ratio,
        ),
    )


def _interpolate_2d_point(
    start_point: tuple[float, float],
    end_point: tuple[float, float],
    ratio: float,
) -> tuple[float, float]:
    safe_ratio = min(max(0.0, float(ratio)), 1.0)
    return (
        start_point[0] + (end_point[0] - start_point[0]) * safe_ratio,
        start_point[1] + (end_point[1] - start_point[1]) * safe_ratio,
    )


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
    texture_width = max(1.0, float(room.uv_map_width))
    texture_height = max(1.0, float(room.uv_map_height))
    top_left, top_right, bottom_right, bottom_left = get_rotated_uv_corners(placement)
    return [
        _normalize_uv_point(bottom_left, texture_width, texture_height),
        _normalize_uv_point(bottom_right, texture_width, texture_height),
        _normalize_uv_point(top_right, texture_width, texture_height),
        _normalize_uv_point(top_left, texture_width, texture_height),
    ]


def _normalize_uv_point(
    uv_point: tuple[float, float],
    texture_width: float,
    texture_height: float,
) -> tuple[float, float]:
    return (
        uv_point[0] / texture_width,
        1.0 - (uv_point[1] / texture_height),
    )


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

        level_blueprint_size = level.image_size_pixels or blueprint_size_pixels
        preview_walls.extend(
            _build_level_preview_textured_walls(
                level=level,
                base_z_meters=_get_level_base_z(level_lookup, level.index),
                blueprint_size_pixels=level_blueprint_size,
                source_transform=_build_level_scale_source_transform(
                    level,
                    level_blueprint_size,
                ),
            )
        )

    return preview_walls


def _build_level_preview_textured_walls(
    level: LevelData,
    base_z_meters: float,
    blueprint_size_pixels: tuple[float, float] | None,
    source_transform: np.ndarray,
) -> list[PreviewTexturedWall]:
    preview_walls: list[PreviewTexturedWall] = []
    for room_index, room in enumerate(level.rooms):
        layout = build_uv_wall_layout(
            room=room,
            vertex_data=level.vertex_data,
            wall_height_meters=room.height_meters,
        )
        if not layout.placements:
            continue

        for placement in layout.placements:
            segment_start_point, segment_end_point = _get_wall_segment_points(
                wall=placement.wall,
                placement=placement,
            )
            start_xy = _point_to_world_xy(
                segment_start_point,
                blueprint_size_pixels,
            )
            end_xy = _point_to_world_xy(
                segment_end_point,
                blueprint_size_pixels,
            )
            preview_walls.append(
                PreviewTexturedWall(
                    level_index=level.index,
                    room_index=room_index,
                    wall_key=placement.wall.key,
                    start_point=_transform_source_point(
                        (
                            float(start_xy[0]),
                            float(start_xy[1]),
                            base_z_meters,
                        ),
                        source_transform,
                    ),
                    end_point=_transform_source_point(
                        (
                            float(end_xy[0]),
                            float(end_xy[1]),
                            base_z_meters,
                        ),
                        source_transform,
                    ),
                    height_meters=float(room.height_meters),
                    texture_rgba=_build_wall_preview_texture(room, placement),
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


# ### Level transform helpers ###
def _get_valid_level_scale(level: LevelData) -> float:
    raw_scale = level.scale
    if isinstance(raw_scale, bool):
        raise ValueError(
            f"Level {level.index} scale must be a finite number greater than zero."
        )

    try:
        scale = float(raw_scale)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"Level {level.index} scale must be a finite number greater than zero."
        ) from error

    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            f"Level {level.index} scale must be a finite number greater than zero."
        )

    return scale


def _build_level_scale_source_transform(
    level: LevelData,
    blueprint_size_pixels: tuple[float, float] | None,
) -> np.ndarray:
    scale = _get_valid_level_scale(level)
    transform = np.eye(4, dtype=float)
    if scale == 1.0 or not level.vertex_data.vertices:
        return transform

    level_points = np.asarray(
        [
            _vertex_to_world_xy(vertex, blueprint_size_pixels)
            for vertex in level.vertex_data.vertices
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(level_points)):
        raise ValueError(f"Level {level.index} vertices must have finite positions.")

    minimum_point = np.min(level_points, axis=0)
    maximum_point = np.max(level_points, axis=0)
    pivot_point = (minimum_point + maximum_point) / 2.0
    transform[0, 0] = scale
    transform[1, 1] = scale
    transform[0, 3] = float(pivot_point[0] * (1.0 - scale))
    transform[1, 3] = float(pivot_point[1] * (1.0 - scale))
    return transform


def _transform_source_point(
    point: tuple[float, float, float],
    source_transform: np.ndarray,
) -> tuple[float, float, float]:
    source_point = np.array([*point, 1.0], dtype=float)
    transformed_point = _get_valid_source_transform(source_transform) @ source_point
    return (
        float(transformed_point[0]),
        float(transformed_point[1]),
        float(transformed_point[2]),
    )


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


def _get_level_floor_object_name(level: LevelData) -> str:
    return f"{_get_level_object_name(level)}_floor"


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


def _get_room_texture_file_name(
    level: LevelData,
    room: RoomData,
    room_index: int,
) -> str:
    level_name = _slugify_name(level.display_name)
    room_name = _slugify_name(room.name or "Room")
    return f"{level_name}_{room_name}_{room_index + 1}.png"


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
