# ### Imports ###
from __future__ import annotations

import math
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from housemaker.glb import PIXEL_TO_METER
from housemaker.models import (
    DEFAULT_UV_MAP_HEIGHT,
    DEFAULT_UV_MAP_WIDTH,
    DEFAULT_WALL_UV_ROTATION_DEGREES,
    DEFAULT_WALL_UV_SCALE,
    RoomData,
    Vertex,
    VertexData,
)

# ### Constants ###
UV_CANVAS_BACKGROUND_COLOR = QColor("#1c1f24")
UV_MAP_BACKGROUND_COLOR = QColor("#f7f8fb")
UV_MAP_OUTLINE_COLOR = QColor("#6b7280")
UV_WALL_FILL_COLOR = QColor(220, 224, 232, 210)
UV_WALL_BORDER_COLOR = QColor("#d92d20")
UV_SELECTED_WALL_COLOR = QColor("#f6c85f")
UV_TEXT_COLOR = QColor("#f5f7fa")
UV_DARK_TEXT_COLOR = QColor("#20242a")
UV_INDICATOR_BACKGROUND_COLOR = QColor(10, 12, 16, 180)
UV_MAP_PADDING = 12.0
UV_WIDGET_MARGIN = 14.0
STRAIGHT_WALL_TOLERANCE = 1e-5
UV_MAP_SIZE_POWERS = (64, 128, 256, 512, 1024, 2048, 4096, 8192)


# ### Data models ###
@dataclass(frozen=True)
class RoomWall:
    key: str
    start_vertex_id: int
    end_vertex_id: int
    start_point: tuple[float, float]
    end_point: tuple[float, float]
    length: float
    projection_direction: str


@dataclass(frozen=True)
class UvWallPlacement:
    wall: RoomWall
    uv_rect: tuple[float, float, float, float]
    rotation_degrees: int


@dataclass(frozen=True)
class UvLayout:
    placements: list[UvWallPlacement]
    hidden_wall_count: int


# ### Widgets ###
class UvCanvas(QWidget):
    selected_wall_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.room: RoomData | None = None
        self.vertex_data: VertexData | None = None
        self.wall_height_meters = 3.0
        self.selected_wall_key: str | None = None
        self.setMinimumHeight(260)

    def set_room_context(
        self,
        room: RoomData | None,
        vertex_data: VertexData | None,
        wall_height_meters: float,
    ) -> None:
        self.room = room
        self.vertex_data = vertex_data
        self.wall_height_meters = wall_height_meters
        if room is None:
            self.selected_wall_key = None
        elif self.selected_wall_key not in _get_room_wall_keys(room, vertex_data):
            self.selected_wall_key = None
        self.update()

    def set_selected_wall_key(self, wall_key: str | None) -> None:
        self.selected_wall_key = wall_key
        self.update()

    def get_selected_wall_key(self) -> str | None:
        return self.selected_wall_key

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), UV_CANVAS_BACKGROUND_COLOR)

        if self.room is None or self.vertex_data is None:
            self._paint_centered_message(painter, "Select a room")
            return

        map_rect = self._get_map_rect()
        painter.setPen(QPen(UV_MAP_OUTLINE_COLOR, 1.5))
        painter.setBrush(UV_MAP_BACKGROUND_COLOR)
        painter.drawRect(map_rect)

        layout = build_uv_wall_layout(
            room=self.room,
            vertex_data=self.vertex_data,
            wall_height_meters=self.wall_height_meters,
        )
        if not layout.placements:
            self._paint_centered_message(painter, "No walls to show")
            return

        self._paint_wall_placements(painter, map_rect, layout.placements)
        if layout.hidden_wall_count > 0:
            self._paint_hidden_wall_indicator(
                painter,
                map_rect,
                layout.hidden_wall_count,
            )

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if self.room is None or self.vertex_data is None:
            super().mousePressEvent(event)
            return

        map_rect = self._get_map_rect()
        layout = build_uv_wall_layout(
            room=self.room,
            vertex_data=self.vertex_data,
            wall_height_meters=self.wall_height_meters,
        )
        for placement in reversed(layout.placements):
            widget_rect = self._uv_rect_to_widget_rect(placement.uv_rect, map_rect)
            if widget_rect.contains(event.position()):
                self.selected_wall_key = placement.wall.key
                self.selected_wall_changed.emit(placement.wall.key)
                self.update()
                event.accept()
                return

        super().mousePressEvent(event)

    def _paint_wall_placements(
        self,
        painter: QPainter,
        map_rect: QRectF,
        placements: list[UvWallPlacement],
    ) -> None:
        painter.setFont(QFont("Segoe UI", 8))
        for placement in placements:
            widget_rect = self._uv_rect_to_widget_rect(placement.uv_rect, map_rect)
            painter.setPen(QPen(UV_WALL_BORDER_COLOR, 2.0))
            painter.setBrush(UV_WALL_FILL_COLOR)
            painter.drawRect(widget_rect)

            if placement.wall.key == self.selected_wall_key:
                painter.setPen(QPen(UV_SELECTED_WALL_COLOR, 3.0))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(widget_rect.adjusted(2.0, 2.0, -2.0, -2.0))

            painter.setPen(QPen(UV_DARK_TEXT_COLOR))
            label_text = (
                f"{placement.wall.projection_direction}\n"
                f"{placement.rotation_degrees} deg"
            )
            painter.drawText(
                widget_rect,
                int(Qt.AlignmentFlag.AlignCenter),
                label_text,
            )

    def _paint_hidden_wall_indicator(
        self,
        painter: QPainter,
        map_rect: QRectF,
        hidden_wall_count: int,
    ) -> None:
        indicator_text = f"{hidden_wall_count} walls are not shown"
        indicator_rect = QRectF(
            map_rect.left() + 10.0,
            map_rect.bottom() - 34.0,
            190.0,
            24.0,
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(UV_INDICATOR_BACKGROUND_COLOR)
        painter.drawRoundedRect(indicator_rect, 6.0, 6.0)
        painter.setPen(QPen(UV_TEXT_COLOR))
        painter.setFont(QFont("Segoe UI", 9))
        painter.drawText(
            indicator_rect,
            int(Qt.AlignmentFlag.AlignCenter),
            indicator_text,
        )

    def _paint_centered_message(self, painter: QPainter, message: str) -> None:
        painter.setPen(QPen(UV_TEXT_COLOR))
        painter.setFont(QFont("Segoe UI", 11))
        painter.drawText(
            self.rect(),
            int(Qt.AlignmentFlag.AlignCenter),
            message,
        )

    def _get_map_rect(self) -> QRectF:
        if self.room is None:
            return QRectF()

        available_rect = QRectF(
            UV_WIDGET_MARGIN,
            UV_WIDGET_MARGIN,
            max(1.0, self.width() - UV_WIDGET_MARGIN * 2.0),
            max(1.0, self.height() - UV_WIDGET_MARGIN * 2.0),
        )
        map_width = max(1.0, float(self.room.uv_map_width))
        map_height = max(1.0, float(self.room.uv_map_height))
        scale = min(
            available_rect.width() / map_width,
            available_rect.height() / map_height,
        )
        display_width = map_width * scale
        display_height = map_height * scale
        center = available_rect.center()
        return QRectF(
            center.x() - display_width / 2.0,
            center.y() - display_height / 2.0,
            display_width,
            display_height,
        )

    def _uv_rect_to_widget_rect(
        self,
        uv_rect: tuple[float, float, float, float],
        map_rect: QRectF,
    ) -> QRectF:
        if self.room is None:
            return QRectF()

        uv_x, uv_y, uv_width, uv_height = uv_rect
        scale_x = map_rect.width() / max(1.0, float(self.room.uv_map_width))
        scale_y = map_rect.height() / max(1.0, float(self.room.uv_map_height))
        return QRectF(
            map_rect.left() + uv_x * scale_x,
            map_rect.top() + uv_y * scale_y,
            uv_width * scale_x,
            uv_height * scale_y,
        )


# ### Layout helpers ###
def build_uv_wall_layout(
    room: RoomData,
    vertex_data: VertexData,
    wall_height_meters: float,
) -> UvLayout:
    walls = build_room_walls(room, vertex_data)
    return _build_uv_wall_layout_for_size(
        walls=walls,
        map_width=float(room.uv_map_width),
        map_height=float(room.uv_map_height),
        wall_height_meters=wall_height_meters,
        wall_uv_scales=room.wall_uv_scales,
        wall_uv_rotations=room.wall_uv_rotations,
    )


def calculate_unoccupied_uv_pixels(
    room: RoomData,
    vertex_data: VertexData,
    wall_height_meters: float,
) -> int:
    layout = build_uv_wall_layout(room, vertex_data, wall_height_meters)
    map_area = max(1.0, float(room.uv_map_width)) * max(
        1.0,
        float(room.uv_map_height),
    )
    occupied_area = sum(
        placement.uv_rect[2] * placement.uv_rect[3]
        for placement in layout.placements
    )
    return max(0, int(round(map_area - occupied_area)))


def calculate_minimum_uv_map_size(
    room: RoomData,
    vertex_data: VertexData,
    wall_height_meters: float,
) -> tuple[int, int]:
    walls = build_room_walls(room, vertex_data)
    if not walls:
        return DEFAULT_UV_MAP_WIDTH, DEFAULT_UV_MAP_HEIGHT

    best_size: tuple[int, int] | None = None
    best_score: tuple[int, int, int] | None = None
    for map_width in UV_MAP_SIZE_POWERS:
        for map_height in UV_MAP_SIZE_POWERS:
            layout = _build_uv_wall_layout_for_size(
                walls=walls,
                map_width=float(map_width),
                map_height=float(map_height),
                wall_height_meters=wall_height_meters,
                wall_uv_scales={},
                wall_uv_rotations={},
            )
            if layout.hidden_wall_count > 0:
                continue

            score = (
                map_width * map_height,
                abs(map_width - map_height),
                map_width,
            )
            if best_score is None or score < best_score:
                best_size = (map_width, map_height)
                best_score = score

    if best_size is None:
        return UV_MAP_SIZE_POWERS[-1], UV_MAP_SIZE_POWERS[-1]

    return best_size


def initialize_room_uv_map_size(
    room: RoomData,
    vertex_data: VertexData,
    wall_height_meters: float,
) -> None:
    uv_map_width, uv_map_height = calculate_minimum_uv_map_size(
        room=room,
        vertex_data=vertex_data,
        wall_height_meters=wall_height_meters,
    )
    room.uv_map_width = uv_map_width
    room.uv_map_height = uv_map_height


def _build_uv_wall_layout_for_size(
    walls: list[RoomWall],
    map_width: float,
    map_height: float,
    wall_height_meters: float,
    wall_uv_scales: dict[str, float],
    wall_uv_rotations: dict[str, int],
) -> UvLayout:
    map_width = max(1.0, map_width)
    map_height = max(1.0, map_height)
    max_right = map_width - UV_MAP_PADDING
    max_bottom = map_height - UV_MAP_PADDING
    wall_height_pixels = max(1.0, wall_height_meters / PIXEL_TO_METER)
    placements: list[UvWallPlacement] = []
    hidden_wall_count = 0
    cursor_x = UV_MAP_PADDING
    cursor_y = UV_MAP_PADDING
    row_height = 0.0

    for wall in walls:
        wall_scale = float(wall_uv_scales.get(wall.key, DEFAULT_WALL_UV_SCALE))
        wall_rotation = _normalize_wall_uv_rotation(
            wall_uv_rotations.get(wall.key, DEFAULT_WALL_UV_ROTATION_DEGREES)
        )
        wall_width = max(1.0, wall.length * wall_scale)
        wall_height = max(1.0, wall_height_pixels * wall_scale)
        if wall_rotation in (90, 270):
            wall_width, wall_height = wall_height, wall_width

        if wall_width > map_width - UV_MAP_PADDING * 2.0:
            hidden_wall_count += 1
            continue
        if wall_height > map_height - UV_MAP_PADDING * 2.0:
            hidden_wall_count += 1
            continue

        if cursor_x + wall_width > max_right and cursor_x > UV_MAP_PADDING:
            cursor_x = UV_MAP_PADDING
            cursor_y += row_height + UV_MAP_PADDING
            row_height = 0.0

        if cursor_y + wall_height > max_bottom:
            hidden_wall_count += 1
            continue

        placements.append(
            UvWallPlacement(
                wall=wall,
                uv_rect=(cursor_x, cursor_y, wall_width, wall_height),
                rotation_degrees=wall_rotation,
            )
        )
        cursor_x += wall_width + UV_MAP_PADDING
        row_height = max(row_height, wall_height)

    return UvLayout(placements=placements, hidden_wall_count=hidden_wall_count)


def build_room_walls(room: RoomData, vertex_data: VertexData | None) -> list[RoomWall]:
    if vertex_data is None:
        return []

    room_vertices = [
        vertex
        for vertex_id in room.vertex_ids
        if (vertex := vertex_data.get_vertex(vertex_id)) is not None
    ]
    if len(room_vertices) < 3:
        return []

    center_point = _get_room_center_point(room, vertex_data, room_vertices)
    ordered_vertices = _order_vertices_around_center(room_vertices, center_point)
    wall_vertices = _remove_straight_through_vertices(ordered_vertices)
    if len(wall_vertices) < 3:
        return []

    return [
        _build_room_wall(
            start_vertex=wall_vertices[index],
            end_vertex=wall_vertices[(index + 1) % len(wall_vertices)],
            center_point=center_point,
        )
        for index in range(len(wall_vertices))
    ]


def _get_room_wall_keys(
    room: RoomData,
    vertex_data: VertexData | None,
) -> set[str]:
    return {wall.key for wall in build_room_walls(room, vertex_data)}


def _get_room_center_point(
    room: RoomData,
    vertex_data: VertexData,
    room_vertices: list[Vertex],
) -> tuple[float, float]:
    center_vertex = vertex_data.get_vertex(room.center_vertex_id)
    if center_vertex is not None:
        return center_vertex.x, center_vertex.y

    return (
        sum(vertex.x for vertex in room_vertices) / len(room_vertices),
        sum(vertex.y for vertex in room_vertices) / len(room_vertices),
    )


def _order_vertices_around_center(
    vertices: list[Vertex],
    center_point: tuple[float, float],
) -> list[Vertex]:
    center_x, center_y = center_point
    return sorted(
        vertices,
        key=lambda vertex: math.atan2(vertex.y - center_y, vertex.x - center_x),
    )


def _remove_straight_through_vertices(vertices: list[Vertex]) -> list[Vertex]:
    simplified_vertices = list(vertices)
    changed = True
    while changed and len(simplified_vertices) > 3:
        changed = False
        for index, current_vertex in enumerate(list(simplified_vertices)):
            previous_vertex = simplified_vertices[index - 1]
            next_vertex = simplified_vertices[(index + 1) % len(simplified_vertices)]
            if not _is_straight_through(previous_vertex, current_vertex, next_vertex):
                continue

            simplified_vertices.remove(current_vertex)
            changed = True
            break

    return simplified_vertices


def _is_straight_through(
    previous_vertex: Vertex,
    current_vertex: Vertex,
    next_vertex: Vertex,
) -> bool:
    previous_vector = (
        previous_vertex.x - current_vertex.x,
        previous_vertex.y - current_vertex.y,
    )
    next_vector = (
        next_vertex.x - current_vertex.x,
        next_vertex.y - current_vertex.y,
    )
    previous_length = math.hypot(previous_vector[0], previous_vector[1])
    next_length = math.hypot(next_vector[0], next_vector[1])
    if previous_length <= STRAIGHT_WALL_TOLERANCE:
        return False
    if next_length <= STRAIGHT_WALL_TOLERANCE:
        return False

    cross_product = (
        previous_vector[0] * next_vector[1]
        - previous_vector[1] * next_vector[0]
    )
    dot_product = (
        previous_vector[0] * next_vector[0]
        + previous_vector[1] * next_vector[1]
    )
    normalized_cross = abs(cross_product) / (previous_length * next_length)
    return normalized_cross <= STRAIGHT_WALL_TOLERANCE and dot_product < 0.0


def _build_room_wall(
    start_vertex: Vertex,
    end_vertex: Vertex,
    center_point: tuple[float, float],
) -> RoomWall:
    wall_midpoint = (
        (start_vertex.x + end_vertex.x) / 2.0,
        (start_vertex.y + end_vertex.y) / 2.0,
    )
    wall_length = math.hypot(
        end_vertex.x - start_vertex.x,
        end_vertex.y - start_vertex.y,
    )
    return RoomWall(
        key=_build_wall_key(start_vertex.id, end_vertex.id),
        start_vertex_id=start_vertex.id,
        end_vertex_id=end_vertex.id,
        start_point=(start_vertex.x, start_vertex.y),
        end_point=(end_vertex.x, end_vertex.y),
        length=wall_length,
        projection_direction=_get_projection_direction(center_point, wall_midpoint),
    )


def _build_wall_key(start_vertex_id: int, end_vertex_id: int) -> str:
    return f"{min(start_vertex_id, end_vertex_id)}:{max(start_vertex_id, end_vertex_id)}"


def _get_projection_direction(
    center_point: tuple[float, float],
    wall_midpoint: tuple[float, float],
) -> str:
    delta_x = wall_midpoint[0] - center_point[0]
    delta_y = wall_midpoint[1] - center_point[1]
    if abs(delta_x) >= abs(delta_y):
        return "East" if delta_x >= 0.0 else "West"

    return "South" if delta_y >= 0.0 else "North"


def _normalize_wall_uv_rotation(rotation_degrees: int) -> int:
    return (round(int(rotation_degrees) / 90) * 90) % 360
