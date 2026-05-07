# ### Imports ###
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from housemaker.models import RoomData, VertexData
from housemaker.uv_layout import (
    UvWallPlacement,
    build_uv_wall_layout,
    get_room_wall_keys,
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
UV_WIDGET_MARGIN = 14.0


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
        elif self.selected_wall_key not in get_room_wall_keys(room, vertex_data):
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
