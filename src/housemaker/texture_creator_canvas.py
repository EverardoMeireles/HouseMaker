# ### Imports ###
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from housemaker.models import RoomData, WallTextureData

# ### Constants ###
TEXTURE_CANVAS_BACKGROUND_COLOR = QColor("#1c1f24")
TEXTURE_IMAGE_BACKGROUND_COLOR = QColor("#101318")
TEXTURE_OUTLINE_COLOR = QColor("#d92d20")
TEXTURE_FACE_OUTLINE_COLOR = QColor("#f6c85f")
TEXTURE_OUTLINE_HANDLE_COLOR = QColor(246, 200, 95, 45)
TEXTURE_TEXT_COLOR = QColor("#f5f7fa")
TEXTURE_WIDGET_MARGIN = 14.0
TEXTURE_MIN_SOURCE_SIZE = 4.0
TEXTURE_WHEEL_SCALE_STEP = 1.08

# ### Widgets ###
class TextureCreatorCanvas(QWidget):
    texture_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.room: RoomData | None = None
        self.wall_key: str | None = None
        self.wall_aspect_ratio = 1.0
        self.wall_segment_count = 1
        self.image_path: str | None = None
        self.source_image = QImage()
        self._context_cache_key: tuple[object, ...] | None = None
        self._decoded_image_cache_key: tuple[object, ...] | None = None
        self._decoded_source_image = QImage()
        self.drag_start_source_point = QPointF()
        self.drag_start_texture_data: WallTextureData | None = None
        self.setMinimumHeight(320)

    def set_context(
        self,
        room: RoomData | None,
        wall_key: str | None,
        wall_size: tuple[float, float] | None,
        image_path: str | None,
        segment_count: int = 1,
    ) -> None:
        normalized_image_path = (
            str(Path(image_path).resolve()) if image_path else None
        )
        image_cache_key = _build_image_cache_key(normalized_image_path)
        self.room = room
        self.wall_key = wall_key
        self.wall_aspect_ratio = _get_wall_aspect_ratio(wall_size)
        self.wall_segment_count = max(1, int(segment_count))
        self.image_path = normalized_image_path
        self.source_image = self._get_cached_source_image(image_cache_key)
        loaded_image_cache_key = (
            image_cache_key if not self.source_image.isNull() else None
        )

        context_cache_key = self._build_context_cache_key(
            loaded_image_cache_key
        )
        if context_cache_key == self._context_cache_key:
            self.update()
            return

        changed = self._ensure_texture_data()
        self._context_cache_key = self._build_context_cache_key(
            loaded_image_cache_key
        )
        self.update()
        if changed:
            self.texture_changed.emit()

    # ### Context and image caches ###
    def _get_cached_source_image(
        self,
        image_cache_key: tuple[object, ...] | None,
    ) -> QImage:
        """Reuse the decoded source while its path metadata stays unchanged."""

        if image_cache_key is None or self.image_path is None:
            return QImage()
        if image_cache_key != self._decoded_image_cache_key:
            if not Path(self.image_path).is_file():
                self._decoded_source_image = QImage()
                self._decoded_image_cache_key = image_cache_key
                return self._decoded_source_image
            decoded_image = _load_source_image(self.image_path)
            if decoded_image.isNull():
                return decoded_image
            self._decoded_source_image = decoded_image
            self._decoded_image_cache_key = image_cache_key
        return self._decoded_source_image

    def _build_context_cache_key(
        self,
        image_cache_key: tuple[object, ...] | None,
    ) -> tuple[object, ...]:
        """Capture context values that can initialize texture placement."""

        return (
            id(self.room),
            self.wall_key,
            self.wall_aspect_ratio.hex(),
            self.wall_segment_count,
            image_cache_key,
            _build_wall_texture_cache_key(self._get_texture_data()),
        )

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), TEXTURE_CANVAS_BACKGROUND_COLOR)

        if self.room is None or self.wall_key is None:
            self._paint_centered_message(painter, "Select a wall in Viewer")
            return

        if self.image_path is None:
            self._paint_blank_wall_outline(painter)
            self._paint_centered_message(painter, "Select an image")
            return

        if self.source_image.isNull():
            self._paint_centered_message(painter, "Image missing")
            return

        image_rect = self._get_image_display_rect()
        painter.fillRect(image_rect, TEXTURE_IMAGE_BACKGROUND_COLOR)
        painter.drawImage(image_rect, self.source_image)
        self._paint_texture_outline(painter, image_rect)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() != Qt.MouseButton.LeftButton or not self._has_editable_image():
            super().mousePressEvent(event)
            return

        source_point = self._widget_point_to_source_point(event.position())
        if source_point is None:
            super().mousePressEvent(event)
            return

        texture_data = self._get_texture_data()
        if texture_data is None:
            super().mousePressEvent(event)
            return

        texture_rect = self._get_texture_widget_rect()
        if texture_rect is not None and not texture_rect.contains(event.position()):
            self._move_texture_center_to_source_point(source_point)
            texture_data = self._get_texture_data()

        self.drag_start_source_point = source_point
        self.drag_start_texture_data = texture_data
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self.drag_start_texture_data is None:
            super().mouseMoveEvent(event)
            return

        source_point = self._widget_point_to_source_point(event.position())
        if source_point is None:
            return

        delta_x = source_point.x() - self.drag_start_source_point.x()
        delta_y = source_point.y() - self.drag_start_source_point.y()
        self._set_texture_data(
            source_x=self.drag_start_texture_data.source_x + delta_x,
            source_y=self.drag_start_texture_data.source_y + delta_y,
            source_width=self.drag_start_texture_data.source_width,
            source_height=self.drag_start_texture_data.source_height,
        )
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if self.drag_start_texture_data is None:
            super().mouseReleaseEvent(event)
            return

        self.drag_start_texture_data = None
        event.accept()

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        if not self._has_editable_image():
            super().wheelEvent(event)
            return

        texture_data = self._get_texture_data()
        if texture_data is None:
            super().wheelEvent(event)
            return

        wheel_delta = event.angleDelta().y() or event.pixelDelta().y()
        if wheel_delta == 0:
            super().wheelEvent(event)
            return

        scale_factor = TEXTURE_WHEEL_SCALE_STEP ** (wheel_delta / 120.0)
        center_x = texture_data.source_x + texture_data.source_width / 2.0
        center_y = texture_data.source_y + texture_data.source_height / 2.0
        new_width = texture_data.source_width * scale_factor
        new_height = new_width / self.wall_aspect_ratio
        self._set_texture_data(
            source_x=center_x - new_width / 2.0,
            source_y=center_y - new_height / 2.0,
            source_width=new_width,
            source_height=new_height,
        )
        event.accept()

    def _has_editable_image(self) -> bool:
        return (
            self.room is not None
            and self.wall_key is not None
            and self.image_path is not None
            and not self.source_image.isNull()
        )

    def _ensure_texture_data(self) -> bool:
        if not self._has_editable_image():
            return False

        texture_data = self._get_texture_data()
        if texture_data is not None and texture_data.image_path == self.image_path:
            return False

        source_rect = _build_default_source_rect(
            source_image=self.source_image,
            aspect_ratio=self.wall_aspect_ratio,
        )
        self._set_texture_data(
            source_x=source_rect.x(),
            source_y=source_rect.y(),
            source_width=source_rect.width(),
            source_height=source_rect.height(),
            emit_changed=False,
        )
        return True

    def _get_texture_data(self) -> WallTextureData | None:
        if self.room is None or self.wall_key is None:
            return None

        return self.room.wall_textures.get(self.wall_key)

    def _set_texture_data(
        self,
        source_x: float,
        source_y: float,
        source_width: float,
        source_height: float,
        emit_changed: bool = True,
    ) -> None:
        if not self._has_editable_image() or self.room is None or self.wall_key is None:
            return

        clamped_rect = _clamp_source_rect(
            source_x=source_x,
            source_y=source_y,
            source_width=source_width,
            source_height=source_height,
            source_image=self.source_image,
            aspect_ratio=self.wall_aspect_ratio,
        )
        self.room.wall_textures[self.wall_key] = WallTextureData(
            image_path=self.image_path or "",
            source_x=clamped_rect.x(),
            source_y=clamped_rect.y(),
            source_width=clamped_rect.width(),
            source_height=clamped_rect.height(),
        )
        self.update()
        if emit_changed:
            self.texture_changed.emit()

    def _move_texture_center_to_source_point(self, source_point: QPointF) -> None:
        texture_data = self._get_texture_data()
        if texture_data is None:
            return

        self._set_texture_data(
            source_x=source_point.x() - texture_data.source_width / 2.0,
            source_y=source_point.y() - texture_data.source_height / 2.0,
            source_width=texture_data.source_width,
            source_height=texture_data.source_height,
        )

    def _paint_texture_outline(self, painter: QPainter, image_rect: QRectF) -> None:
        texture_rect = self._get_texture_widget_rect()
        if texture_rect is None:
            return

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(TEXTURE_OUTLINE_HANDLE_COLOR)
        painter.drawRect(texture_rect)
        painter.setPen(QPen(TEXTURE_OUTLINE_COLOR, 2.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(texture_rect)
        self._paint_face_outlines(painter, texture_rect)

    def _paint_blank_wall_outline(self, painter: QPainter) -> None:
        outline_rect = self._get_blank_wall_outline_rect()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(TEXTURE_OUTLINE_HANDLE_COLOR)
        painter.drawRect(outline_rect)
        painter.setPen(QPen(TEXTURE_OUTLINE_COLOR, 2.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(outline_rect)
        self._paint_face_outlines(painter, outline_rect)

    def _paint_face_outlines(self, painter: QPainter, outline_rect: QRectF) -> None:
        painter.setPen(QPen(TEXTURE_FACE_OUTLINE_COLOR, 1.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(outline_rect.adjusted(2.0, 2.0, -2.0, -2.0))
        if self.wall_segment_count <= 1:
            return

        for segment_index in range(1, self.wall_segment_count):
            segment_ratio = self._get_segment_division_ratio(segment_index)
            segment_x = (
                outline_rect.left()
                + outline_rect.width() * segment_ratio
            )
            painter.drawLine(
                QPointF(segment_x, outline_rect.top()),
                QPointF(segment_x, outline_rect.bottom()),
            )

    def _get_segment_division_ratio(self, segment_index: int) -> float:
        if self.room is None or self.wall_key is None:
            return segment_index / max(1, self.wall_segment_count)

        source_ranges = self.room.wall_subdivision_source_ranges.get(self.wall_key)
        if source_ranges is None or len(source_ranges) != self.wall_segment_count:
            return segment_index / max(1, self.wall_segment_count)

        return min(max(0.0, float(source_ranges[segment_index - 1][1])), 1.0)

    def _paint_centered_message(self, painter: QPainter, message: str) -> None:
        painter.setPen(QPen(TEXTURE_TEXT_COLOR))
        painter.setFont(QFont("Segoe UI", 11))
        painter.drawText(
            self.rect(),
            int(Qt.AlignmentFlag.AlignCenter),
            message,
        )

    def _get_image_display_rect(self) -> QRectF:
        available_rect = QRectF(
            TEXTURE_WIDGET_MARGIN,
            TEXTURE_WIDGET_MARGIN,
            max(1.0, self.width() - TEXTURE_WIDGET_MARGIN * 2.0),
            max(1.0, self.height() - TEXTURE_WIDGET_MARGIN * 2.0),
        )
        image_width = max(1.0, float(self.source_image.width()))
        image_height = max(1.0, float(self.source_image.height()))
        scale = min(
            available_rect.width() / image_width,
            available_rect.height() / image_height,
        )
        display_width = image_width * scale
        display_height = image_height * scale
        return QRectF(
            available_rect.center().x() - display_width / 2.0,
            available_rect.center().y() - display_height / 2.0,
            display_width,
            display_height,
        )

    def _get_blank_wall_outline_rect(self) -> QRectF:
        available_rect = QRectF(
            TEXTURE_WIDGET_MARGIN,
            TEXTURE_WIDGET_MARGIN,
            max(1.0, self.width() - TEXTURE_WIDGET_MARGIN * 2.0),
            max(1.0, self.height() - TEXTURE_WIDGET_MARGIN * 2.0),
        )
        display_width = min(
            available_rect.width() * 0.72,
            available_rect.height() * 0.72 * self.wall_aspect_ratio,
        )
        display_height = display_width / self.wall_aspect_ratio
        return QRectF(
            available_rect.center().x() - display_width / 2.0,
            available_rect.center().y() - display_height / 2.0,
            display_width,
            display_height,
        )

    def _get_texture_widget_rect(self) -> QRectF | None:
        texture_data = self._get_texture_data()
        if texture_data is None or self.source_image.isNull():
            return None

        image_rect = self._get_image_display_rect()
        scale_x = image_rect.width() / max(1.0, float(self.source_image.width()))
        scale_y = image_rect.height() / max(1.0, float(self.source_image.height()))
        return QRectF(
            image_rect.left() + texture_data.source_x * scale_x,
            image_rect.top() + texture_data.source_y * scale_y,
            texture_data.source_width * scale_x,
            texture_data.source_height * scale_y,
        )

    def _widget_point_to_source_point(self, widget_point: QPointF) -> QPointF | None:
        if self.source_image.isNull():
            return None

        image_rect = self._get_image_display_rect()
        if not image_rect.contains(widget_point):
            return None

        scale_x = max(1.0, float(self.source_image.width())) / image_rect.width()
        scale_y = max(1.0, float(self.source_image.height())) / image_rect.height()
        return QPointF(
            (widget_point.x() - image_rect.left()) * scale_x,
            (widget_point.y() - image_rect.top()) * scale_y,
        )


# ### Image cache helpers ###
def _build_image_cache_key(
    image_path: str | None,
) -> tuple[object, ...] | None:
    """Identify an image revision without decoding its pixels."""

    if image_path is None:
        return None
    try:
        image_stat = Path(image_path).stat()
    except OSError:
        return (image_path, None, None, None)
    return (
        image_path,
        int(image_stat.st_mtime_ns),
        int(image_stat.st_ctime_ns),
        int(image_stat.st_size),
    )


def _load_source_image(image_path: str) -> QImage:
    """Decode one source image on a cache miss."""

    return QImage(image_path)


def _build_wall_texture_cache_key(
    texture_data: WallTextureData | None,
) -> tuple[object, ...] | None:
    """Identify external changes to the active wall's crop data."""

    if texture_data is None:
        return None
    return (
        texture_data.image_path,
        float(texture_data.source_x).hex(),
        float(texture_data.source_y).hex(),
        float(texture_data.source_width).hex(),
        float(texture_data.source_height).hex(),
    )


# ### Geometry helpers ###
def _get_wall_aspect_ratio(wall_size: tuple[float, float] | None) -> float:
    if wall_size is None:
        return 1.0

    wall_width, wall_height = wall_size
    return max(0.01, float(wall_width) / max(1.0, float(wall_height)))


def _build_default_source_rect(
    source_image: QImage,
    aspect_ratio: float,
) -> QRectF:
    image_width = max(1.0, float(source_image.width()))
    image_height = max(1.0, float(source_image.height()))
    image_aspect_ratio = image_width / image_height
    if image_aspect_ratio >= aspect_ratio:
        source_height = image_height
        source_width = source_height * aspect_ratio
    else:
        source_width = image_width
        source_height = source_width / aspect_ratio

    return QRectF(
        (image_width - source_width) / 2.0,
        (image_height - source_height) / 2.0,
        source_width,
        source_height,
    )


def _clamp_source_rect(
    source_x: float,
    source_y: float,
    source_width: float,
    source_height: float,
    source_image: QImage,
    aspect_ratio: float,
) -> QRectF:
    image_width = max(1.0, float(source_image.width()))
    image_height = max(1.0, float(source_image.height()))
    max_width = image_width
    max_height = min(image_height, max_width / aspect_ratio)
    if max_height < image_height:
        max_width = max_height * aspect_ratio

    clamped_width = min(
        max(TEXTURE_MIN_SOURCE_SIZE, float(source_width)),
        max_width,
    )
    clamped_height = clamped_width / aspect_ratio
    if clamped_height > max_height:
        clamped_height = max_height
        clamped_width = clamped_height * aspect_ratio

    clamped_x = min(
        max(0.0, float(source_x)),
        max(0.0, image_width - clamped_width),
    )
    clamped_y = min(
        max(0.0, float(source_y)),
        max(0.0, image_height - clamped_height),
    )
    return QRectF(clamped_x, clamped_y, clamped_width, clamped_height)
