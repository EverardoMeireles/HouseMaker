# ### Imports ###
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from housemaker.models import (
    DEFAULT_ROOM_HEIGHT_METERS,
    DEFAULT_WALL_UV_SCALE,
    RoomData,
    VertexData,
)
from housemaker.texture_mapping import paint_wall_texture_crop
from housemaker.uv_layout import (
    UvLayout,
    UvWallPlacement,
    build_uv_wall_layout,
    get_rotated_uv_corners,
    rebuild_room_subdivision_uvs,
)

# ### Constants ###
UV_CANVAS_BACKGROUND_COLOR = QColor("#1c1f24")
UV_MAP_BACKGROUND_COLOR = QColor("#f7f8fb")
UV_MAP_OUTLINE_COLOR = QColor("#6b7280")
UV_WALL_FILL_COLOR = QColor(220, 224, 232, 210)
UV_WALL_BORDER_COLOR = QColor("#d92d20")
UV_FACE_BORDER_COLOR = QColor("#f6c85f")
UV_SELECTED_WALL_COLOR = QColor("#f6c85f")
UV_TEXT_COLOR = QColor("#f5f7fa")
UV_DARK_TEXT_COLOR = QColor("#20242a")
UV_INDICATOR_BACKGROUND_COLOR = QColor(10, 12, 16, 180)
UV_WIDGET_MARGIN = 14.0


# ### Widgets ###
class UvCanvas(QWidget):
    selected_wall_changed = Signal(str)
    uv_values_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.room: RoomData | None = None
        self.vertex_data: VertexData | None = None
        self.wall_height_meters = DEFAULT_ROOM_HEIGHT_METERS
        self.selected_wall_key: str | None = None
        self.drag_mode: str | None = None
        self.drag_wall_key: str | None = None
        self.drag_segment_index = 0
        self.drag_start_uv_point = QPointF()
        self.drag_start_wall_position: tuple[float, float] = (0.0, 0.0)
        self.rotation_center_uv: tuple[float, float] = (0.0, 0.0)
        self.rotation_start_angle = 0.0
        self.rotation_start_degrees = 0
        self._layout_cache_key: tuple[object, ...] | None = None
        self._layout_cache: UvLayout | None = None
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
        elif self.selected_wall_key is not None:
            wall_keys = {
                placement.wall.key
                for placement in self._get_wall_layout().placements
            }
            if self.selected_wall_key not in wall_keys:
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

        layout = self._get_wall_layout()
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
        placement = self._find_placement_at_position(event.position(), map_rect)
        if placement is None:
            super().mousePressEvent(event)
            return

        self._select_wall(placement.wall.key)
        if event.button() == Qt.MouseButton.LeftButton:
            self._start_move_drag(event.position(), placement, map_rect)
            event.accept()
            return

        if event.button() == Qt.MouseButton.RightButton:
            self._start_rotation_drag(event.position(), placement, map_rect)
            event.accept()
            return

        event.accept()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self.drag_mode == "move":
            self._continue_move_drag(event.position())
            event.accept()
            return

        if self.drag_mode == "rotate":
            self._continue_rotation_drag(event.position())
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if self.drag_mode is None:
            super().mouseReleaseEvent(event)
            return

        self.drag_mode = None
        self.drag_wall_key = None
        self.drag_segment_index = 0
        event.accept()

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        if self.room is None or self.vertex_data is None:
            super().wheelEvent(event)
            return

        map_rect = self._get_map_rect()
        placement = self._find_placement_at_position(event.position(), map_rect)
        if placement is not None:
            self._select_wall(placement.wall.key)
        else:
            placement = self._get_selected_wall_placement()

        if placement is None:
            super().wheelEvent(event)
            return

        wheel_delta = event.angleDelta().y()
        if wheel_delta == 0:
            wheel_delta = event.pixelDelta().y()
        if wheel_delta == 0:
            super().wheelEvent(event)
            return

        scale_factor = 1.05 ** (wheel_delta / 120.0)
        current_scale = self.room.wall_uv_scales.get(
            placement.wall.key,
            DEFAULT_WALL_UV_SCALE,
        )
        new_scale = max(0.01, round(current_scale * scale_factor, 3))
        placement_center = self._get_placement_center(placement)
        self.room.wall_uv_scales[placement.wall.key] = new_scale
        self._refresh_subdivision_layout_if_needed(placement.wall.key)
        self._set_wall_position_from_center(placement.wall.key, placement_center)
        self.uv_values_changed.emit()
        self.update()
        event.accept()

    def _find_placement_at_position(
        self,
        widget_position: QPointF,
        map_rect: QRectF,
    ) -> UvWallPlacement | None:
        layout = self._get_wall_layout()
        for placement in reversed(layout.placements):
            widget_polygon = self._uv_placement_to_widget_polygon(placement, map_rect)
            if widget_polygon.containsPoint(
                widget_position,
                Qt.FillRule.OddEvenFill,
            ):
                return placement

        return None

    def _select_wall(self, wall_key: str) -> None:
        self.selected_wall_key = wall_key
        self.selected_wall_changed.emit(wall_key)
        self.update()

    def _start_move_drag(
        self,
        widget_position: QPointF,
        placement: UvWallPlacement,
        map_rect: QRectF,
    ) -> None:
        self.drag_mode = "move"
        self.drag_wall_key = placement.wall.key
        self.drag_segment_index = placement.segment_index
        self.drag_start_uv_point = self._widget_point_to_uv_point(
            widget_position,
            map_rect,
        )
        self.drag_start_wall_position = (placement.uv_rect[0], placement.uv_rect[1])

    def _continue_move_drag(self, widget_position: QPointF) -> None:
        if self.room is None or self.drag_wall_key is None:
            return

        map_rect = self._get_map_rect()
        current_uv_point = self._widget_point_to_uv_point(widget_position, map_rect)
        placement = self._get_wall_placement(
            self.drag_wall_key,
            self.drag_segment_index,
        )
        if placement is None:
            return

        moved_position = (
            self.drag_start_wall_position[0]
            + current_uv_point.x()
            - self.drag_start_uv_point.x(),
            self.drag_start_wall_position[1]
            + current_uv_point.y()
            - self.drag_start_uv_point.y(),
        )
        clamped_position = self._clamp_wall_position(moved_position, placement)
        if self.drag_wall_key in self.room.wall_subdivisions:
            segment_positions = list(
                self.room.wall_subdivision_positions.get(self.drag_wall_key, ())
            )
            if len(segment_positions) == placement.segment_count:
                segment_positions[placement.segment_index] = clamped_position
                self.room.wall_subdivision_positions[self.drag_wall_key] = tuple(
                    segment_positions
                )
        else:
            self.room.wall_uv_positions[self.drag_wall_key] = clamped_position
        self.uv_values_changed.emit()
        self.update()

    def _start_rotation_drag(
        self,
        widget_position: QPointF,
        placement: UvWallPlacement,
        map_rect: QRectF,
    ) -> None:
        self.drag_mode = "rotate"
        self.drag_wall_key = placement.wall.key
        self.drag_segment_index = placement.segment_index
        self.rotation_center_uv = self._get_placement_center(placement)
        self.rotation_start_angle = self._get_uv_angle_from_center(
            widget_position=widget_position,
            map_rect=map_rect,
            center_uv=self.rotation_center_uv,
        )
        self.rotation_start_degrees = placement.rotation_degrees

    def _continue_rotation_drag(self, widget_position: QPointF) -> None:
        if self.room is None or self.drag_wall_key is None:
            return

        current_angle = self._get_uv_angle_from_center(
            widget_position=widget_position,
            map_rect=self._get_map_rect(),
            center_uv=self.rotation_center_uv,
        )
        rotation_delta = _get_shortest_angle_delta(
            current_angle,
            self.rotation_start_angle,
        )
        self.room.wall_uv_rotations[self.drag_wall_key] = int(
            round(self.rotation_start_degrees + rotation_delta)
        ) % 360
        self._refresh_subdivision_layout_if_needed(self.drag_wall_key)
        self._set_wall_position_from_center(
            self.drag_wall_key,
            self.rotation_center_uv,
        )
        self.uv_values_changed.emit()
        self.update()

    def _get_uv_angle_from_center(
        self,
        widget_position: QPointF,
        map_rect: QRectF,
        center_uv: tuple[float, float],
    ) -> float:
        uv_point = self._widget_point_to_uv_point(widget_position, map_rect)
        return math.degrees(
            math.atan2(uv_point.y() - center_uv[1], uv_point.x() - center_uv[0])
        )

    def _get_selected_wall_placement(self) -> UvWallPlacement | None:
        if self.selected_wall_key is None:
            return None

        return self._get_wall_placement(self.selected_wall_key)

    def _get_wall_placement(
        self,
        wall_key: str,
        segment_index: int | None = None,
    ) -> UvWallPlacement | None:
        if self.room is None or self.vertex_data is None:
            return None

        layout = self._get_wall_layout()
        for placement in layout.placements:
            if placement.wall.key != wall_key:
                continue
            if segment_index is None or placement.segment_index == segment_index:
                return placement

        return None

    # ### Layout cache ###
    def _get_wall_layout(self) -> UvLayout:
        """Reuse layout geometry until one of its mutable inputs changes."""

        if self.room is None or self.vertex_data is None:
            return UvLayout(placements=[], hidden_wall_count=0)

        cache_key = _build_uv_layout_cache_key(
            self.room,
            self.vertex_data,
            self.wall_height_meters,
        )
        if (
            cache_key == self._layout_cache_key
            and self._layout_cache is not None
        ):
            return self._layout_cache

        layout = build_uv_wall_layout(
            room=self.room,
            vertex_data=self.vertex_data,
            wall_height_meters=self.wall_height_meters,
        )
        self._layout_cache_key = cache_key
        self._layout_cache = layout
        return layout

    # ### Placement geometry ###
    def _get_placement_center(
        self,
        placement: UvWallPlacement,
    ) -> tuple[float, float]:
        uv_x, uv_y, uv_width, uv_height = placement.uv_rect
        return uv_x + uv_width / 2.0, uv_y + uv_height / 2.0

    def _set_wall_position_from_center(
        self,
        wall_key: str,
        center_uv: tuple[float, float],
    ) -> None:
        if self.room is None:
            return

        placement = self._get_wall_placement(wall_key)
        if placement is None:
            return

        position = (
            center_uv[0] - placement.uv_rect[2] / 2.0,
            center_uv[1] - placement.uv_rect[3] / 2.0,
        )
        self.room.wall_uv_positions[wall_key] = self._clamp_wall_position(
            position,
            placement,
        )

    def _refresh_subdivision_layout_if_needed(self, wall_key: str) -> None:
        if self.room is None or self.vertex_data is None:
            return
        if wall_key not in self.room.wall_subdivisions:
            return

        optimized_result = rebuild_room_subdivision_uvs(
            room=self.room,
            vertex_data=self.vertex_data,
            wall_height_meters=self.wall_height_meters,
        )
        if optimized_result is None:
            return

        self.room.wall_uv_rotations = dict(optimized_result.wall_uv_rotations)
        self.room.wall_uv_scales = dict(optimized_result.wall_uv_scales)
        self.room.wall_uv_positions = dict(optimized_result.wall_uv_positions)
        self.room.wall_subdivisions = dict(optimized_result.wall_subdivisions)
        self.room.wall_subdivision_positions = dict(
            optimized_result.wall_subdivision_positions
        )
        self.room.wall_subdivision_source_ranges = dict(
            optimized_result.wall_subdivision_source_ranges
        )

    def _clamp_wall_position(
        self,
        position: tuple[float, float],
        placement: UvWallPlacement,
    ) -> tuple[float, float]:
        if self.room is None:
            return position

        max_x = max(0.0, float(self.room.uv_map_width) - placement.uv_rect[2])
        max_y = max(0.0, float(self.room.uv_map_height) - placement.uv_rect[3])
        return (
            min(max(position[0], 0.0), max_x),
            min(max(position[1], 0.0), max_y),
        )

    def _paint_wall_placements(
        self,
        painter: QPainter,
        map_rect: QRectF,
        placements: list[UvWallPlacement],
    ) -> None:
        painter.setFont(QFont("Segoe UI", 8))
        for placement in placements:
            widget_rect = self._uv_rect_to_widget_rect(placement.uv_rect, map_rect)
            wall_width, wall_height = placement.natural_size
            scale_x = map_rect.width() / max(1.0, float(self.room.uv_map_width))
            scale_y = map_rect.height() / max(1.0, float(self.room.uv_map_height))
            wall_widget_rect = QRectF(
                -wall_width * scale_x / 2.0,
                -wall_height * scale_y / 2.0,
                wall_width * scale_x,
                wall_height * scale_y,
            )

            painter.save()
            painter.translate(widget_rect.center())
            painter.rotate(placement.rotation_degrees)
            texture_data = self.room.wall_textures.get(placement.wall.key)
            did_paint_texture = (
                texture_data is not None
                and paint_wall_texture_crop(
                    painter,
                    texture_data,
                    wall_widget_rect,
                    placement.source_start_ratio,
                    placement.source_end_ratio,
                )
            )
            painter.setPen(QPen(UV_WALL_BORDER_COLOR, 2.0))
            painter.setBrush(
                Qt.BrushStyle.NoBrush if did_paint_texture else UV_WALL_FILL_COLOR
            )
            painter.drawRect(wall_widget_rect)
            painter.setPen(QPen(UV_FACE_BORDER_COLOR, 1.3))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(wall_widget_rect.adjusted(2.0, 2.0, -2.0, -2.0))

            if placement.wall.key == self.selected_wall_key:
                painter.setPen(QPen(UV_SELECTED_WALL_COLOR, 3.0))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(wall_widget_rect.adjusted(4.0, 4.0, -4.0, -4.0))

            if not did_paint_texture:
                painter.setPen(QPen(UV_DARK_TEXT_COLOR))
                label_text = (
                    f"{placement.wall.projection_direction}\n"
                    f"{placement.rotation_degrees} deg"
                )
                painter.drawText(
                    wall_widget_rect,
                    int(Qt.AlignmentFlag.AlignCenter),
                    label_text,
                )
            painter.restore()

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

    def _uv_placement_to_widget_polygon(
        self,
        placement: UvWallPlacement,
        map_rect: QRectF,
    ) -> QPolygonF:
        return QPolygonF(
            [
                self._uv_point_to_widget_point(uv_point, map_rect)
                for uv_point in get_rotated_uv_corners(placement)
            ]
        )

    def _uv_point_to_widget_point(
        self,
        uv_point: tuple[float, float],
        map_rect: QRectF,
    ) -> QPointF:
        if self.room is None:
            return QPointF()

        scale_x = map_rect.width() / max(1.0, float(self.room.uv_map_width))
        scale_y = map_rect.height() / max(1.0, float(self.room.uv_map_height))
        return QPointF(
            map_rect.left() + uv_point[0] * scale_x,
            map_rect.top() + uv_point[1] * scale_y,
        )

    def _widget_point_to_uv_point(
        self,
        widget_point: QPointF,
        map_rect: QRectF,
    ) -> QPointF:
        if self.room is None:
            return QPointF()

        scale_x = map_rect.width() / max(1.0, float(self.room.uv_map_width))
        scale_y = map_rect.height() / max(1.0, float(self.room.uv_map_height))
        return QPointF(
            (widget_point.x() - map_rect.left()) / scale_x,
            (widget_point.y() - map_rect.top()) / scale_y,
        )


# ### Layout cache helpers ###
def _build_uv_layout_cache_key(
    room: RoomData,
    vertex_data: VertexData,
    wall_height_meters: float,
) -> tuple[object, ...]:
    """Snapshot every geometry and UV value consumed by the layout builder."""

    return (
        tuple(
            (
                vertex.id,
                _freeze_uv_layout_value(vertex.x),
                _freeze_uv_layout_value(vertex.y),
            )
            for vertex in vertex_data.vertices
        ),
        tuple(
            (edge.start_vertex_id, edge.end_vertex_id)
            for edge in vertex_data.edges
        ),
        tuple(room.vertex_ids),
        int(room.center_vertex_id),
        _freeze_uv_layout_value(wall_height_meters),
        int(room.uv_map_width),
        int(room.uv_map_height),
        _freeze_uv_layout_value(room.wall_uv_scales),
        _freeze_uv_layout_value(room.wall_uv_rotations),
        _freeze_uv_layout_value(room.wall_uv_positions),
        _freeze_uv_layout_value(room.wall_subdivisions),
        _freeze_uv_layout_value(room.wall_subdivision_positions),
        _freeze_uv_layout_value(room.wall_subdivision_source_ranges),
    )


def _freeze_uv_layout_value(value: object) -> object:
    """Make nested mutable UV values deterministic and safely comparable."""

    if isinstance(value, dict):
        frozen_items = (
            (
                _freeze_uv_layout_value(key),
                _freeze_uv_layout_value(item_value),
            )
            for key, item_value in value.items()
        )
        return tuple(sorted(frozen_items, key=repr))
    if isinstance(value, list | tuple):
        return tuple(_freeze_uv_layout_value(item) for item in value)
    if isinstance(value, float):
        return ("float", value.hex())
    if value is None or isinstance(value, str | int | bool):
        return value
    return (type(value).__qualname__, repr(value))


# ### Numeric helpers ###
def _get_shortest_angle_delta(current_angle: float, start_angle: float) -> float:
    return (current_angle - start_angle + 180.0) % 360.0 - 180.0
