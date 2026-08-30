# ### Imports ###
from __future__ import annotations

import math

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QWidget

from housemaker.video_source import normalize_video_frame
from housemaker.generation_state import (
    MASK_MODE_ERASE,
    MASK_MODE_PAINT,
    MaskPoint,
    MaskStroke,
)


# ### Constants ###
VIEW_BACKGROUND_COLOR = QColor("#15181d")
VIEW_EMPTY_TEXT_COLOR = QColor("#e5e7eb")
MASK_OVERLAY_RGB = (48, 190, 255)
MASK_OVERLAY_ALPHA = 112
MASK_OUTLINE_COLOR = QColor(95, 215, 255, 230)
BRUSH_CURSOR_PAINT_COLOR = QColor(95, 215, 255, 235)
BRUSH_CURSOR_ERASE_COLOR = QColor(255, 174, 92, 235)
DEFAULT_BRUSH_RADIUS_PIXELS = 24
MIN_NORMALIZED_BRUSH_RADIUS = 1e-6


# ### Video mask widget ###
class VideoInpaintView(QWidget):
    """Aspect-fit video view with brush and enclosed-region mask editing."""

    strokes_changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame_bgr: np.ndarray | None = None
        self._frame_image = QImage()
        self._mask = np.empty((0, 0), dtype=np.uint8)
        self._mask_overlay_image = QImage()
        self._strokes: list[MaskStroke] = []
        self._active_points: list[MaskPoint] = []
        self._brush_mode = MASK_MODE_PAINT
        self._brush_radius_pixels = DEFAULT_BRUSH_RADIUS_PIXELS
        self._interaction_enabled = True
        self._hover_position: QPointF | None = None
        self._empty_text = "No video loaded"
        self.setMinimumSize(160, 120)
        self.setMouseTracking(True)
        self.setToolTip(
            "Left-drag to paint. Right-click inside a closed painted outline "
            "to fill the enclosed area."
        )

    def set_frame(
        self,
        frame_bgr: np.ndarray | None,
        strokes: list[MaskStroke] | None = None,
    ) -> None:
        self._cancel_active_stroke()
        if frame_bgr is None:
            self._frame_bgr = None
            self._frame_image = QImage()
            self._mask = np.empty((0, 0), dtype=np.uint8)
            self._mask_overlay_image = QImage()
            self._strokes = []
            self.update()
            return

        normalized_frame = normalize_video_frame(np.asarray(frame_bgr))
        self._frame_bgr = np.ascontiguousarray(normalized_frame).copy()
        self._frame_image = _bgr_array_to_qimage(self._frame_bgr)
        self._strokes = list(strokes or [])
        self._rebuild_mask()
        self.update()

    def clear_frame(self, empty_text: str | None = None) -> None:
        if empty_text is not None:
            self._empty_text = str(empty_text)
        self.set_frame(None)

    def set_strokes(self, strokes: list[MaskStroke]) -> None:
        self._cancel_active_stroke()
        self._strokes = list(strokes)
        self._rebuild_mask()
        self.update()

    def get_strokes(self) -> list[MaskStroke]:
        return list(self._strokes)

    def get_mask(self) -> np.ndarray:
        return self._mask.copy()

    def get_frame_bgr(self) -> np.ndarray | None:
        if self._frame_bgr is None:
            return None
        return self._frame_bgr.copy()

    def has_selection(self) -> bool:
        return bool(self._mask.size and np.any(self._mask > 0))

    def set_brush_mode(self, mode: str) -> None:
        if mode not in (MASK_MODE_PAINT, MASK_MODE_ERASE):
            raise ValueError(f"Unknown mask brush mode: {mode!r}.")
        self._brush_mode = mode
        self.update()

    def get_brush_mode(self) -> str:
        return self._brush_mode

    def set_brush_radius_pixels(self, radius_pixels: int) -> None:
        self._brush_radius_pixels = max(1, int(radius_pixels))
        self.update()

    def get_brush_radius_pixels(self) -> int:
        return int(self._brush_radius_pixels)

    def set_interaction_enabled(self, enabled: bool) -> None:
        self._interaction_enabled = bool(enabled)
        if not enabled:
            self._cancel_active_stroke()
        self.update()

    def clear_mask(self) -> None:
        self._cancel_active_stroke()
        if not self._strokes and not self.has_selection():
            return
        self._strokes = []
        self._rebuild_mask()
        self.strokes_changed.emit([])
        self.update()

    def build_selected_object_crop(
        self,
        padding_ratio: float = 0.08,
    ) -> np.ndarray:
        """Return a BGRA crop whose alpha channel is the painted selection."""

        if self._frame_bgr is None or not self.has_selection():
            return np.empty((0, 0, 4), dtype=np.uint8)
        mask_rows, mask_columns = np.nonzero(self._mask > 0)
        frame_height, frame_width = self._mask.shape
        selection_width = int(mask_columns.max() - mask_columns.min() + 1)
        selection_height = int(mask_rows.max() - mask_rows.min() + 1)
        padding = int(
            round(max(selection_width, selection_height) * max(0.0, padding_ratio))
        )
        left = max(0, int(mask_columns.min()) - padding)
        right = min(frame_width, int(mask_columns.max()) + padding + 1)
        top = max(0, int(mask_rows.min()) - padding)
        bottom = min(frame_height, int(mask_rows.max()) + padding + 1)

        crop_bgr = self._frame_bgr[top:bottom, left:right]
        crop_alpha = self._mask[top:bottom, left:right]
        crop_bgra = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
        crop_bgra[crop_alpha == 0, :3] = 0
        crop_bgra[:, :, 3] = crop_alpha
        return np.ascontiguousarray(crop_bgra)

    def build_context_overlay(self) -> np.ndarray:
        """Return the full frame with a translucent colored selection overlay."""

        if self._frame_bgr is None:
            return np.empty((0, 0, 3), dtype=np.uint8)
        result = self._frame_bgr.copy()
        selected = self._mask > 0
        if not np.any(selected):
            return result
        overlay_bgr = np.asarray(
            (MASK_OVERLAY_RGB[2], MASK_OVERLAY_RGB[1], MASK_OVERLAY_RGB[0]),
            dtype=float,
        )
        result[selected] = np.clip(
            result[selected].astype(float) * 0.55 + overlay_bgr * 0.45,
            0.0,
            255.0,
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            np.where(selected, 255, 0).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(result, contours, -1, (255, 215, 95), 2)
        return np.ascontiguousarray(result)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.fillRect(self.rect(), VIEW_BACKGROUND_COLOR)

        if self._frame_image.isNull():
            painter.setPen(VIEW_EMPTY_TEXT_COLOR)
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                self._empty_text,
            )
            return

        target_rect = self._get_image_target_rect()
        painter.drawImage(target_rect, self._frame_image)
        if not self._mask_overlay_image.isNull():
            painter.drawImage(target_rect, self._mask_overlay_image)
        self._paint_mask_outline(painter, target_rect)
        self._paint_brush_cursor(painter, target_rect)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.RightButton and self._can_edit_mask():
            point = self._widget_to_mask_point(event.position())
            if point is not None:
                self._cancel_active_stroke()
                if self._append_enclosed_fill(point):
                    self.strokes_changed.emit(self.get_strokes())
                self.update()
                event.accept()
                return
        if (
            event.button() != Qt.MouseButton.LeftButton
            or not self._can_edit_mask()
        ):
            super().mousePressEvent(event)
            return
        point = self._widget_to_mask_point(event.position())
        if point is None:
            super().mousePressEvent(event)
            return
        self._active_points = [point]
        self._rebuild_mask()
        self.update()
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        self._hover_position = QPointF(event.position())
        if self._active_points and event.buttons() & Qt.MouseButton.LeftButton:
            point = self._widget_to_mask_point(event.position())
            if point is not None and self._point_is_distinct(point):
                self._active_points.append(point)
                self._rebuild_mask()
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() != Qt.MouseButton.LeftButton or not self._active_points:
            super().mouseReleaseEvent(event)
            return
        point = self._widget_to_mask_point(event.position())
        if point is not None and self._point_is_distinct(point):
            self._active_points.append(point)
        self._strokes.append(self._build_active_stroke())
        self._active_points = []
        self._rebuild_mask()
        self.strokes_changed.emit(self.get_strokes())
        self.update()
        event.accept()

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        self._hover_position = None
        self.update()
        super().leaveEvent(event)

    def _can_edit_mask(self) -> bool:
        return self._interaction_enabled and self._frame_bgr is not None

    def _get_image_target_rect(self) -> QRectF:
        if self._frame_image.isNull():
            return QRectF()
        return _get_aspect_fit_rect(
            self._frame_image.width(),
            self._frame_image.height(),
            QRectF(self.rect()),
        )

    def _widget_to_mask_point(self, position: QPointF) -> MaskPoint | None:
        target_rect = self._get_image_target_rect()
        if target_rect.isEmpty() or not target_rect.contains(position):
            return None
        normalized_x = (position.x() - target_rect.left()) / target_rect.width()
        normalized_y = (position.y() - target_rect.top()) / target_rect.height()
        return MaskPoint(
            x=min(max(float(normalized_x), 0.0), 1.0),
            y=min(max(float(normalized_y), 0.0), 1.0),
        )

    def _point_is_distinct(self, point: MaskPoint) -> bool:
        previous = self._active_points[-1]
        target_rect = self._get_image_target_rect()
        delta_x = (point.x - previous.x) * target_rect.width()
        delta_y = (point.y - previous.y) * target_rect.height()
        return math.hypot(delta_x, delta_y) >= 0.75

    def _build_active_stroke(self) -> MaskStroke:
        target_rect = self._get_image_target_rect()
        shortest_side = max(1.0, min(target_rect.width(), target_rect.height()))
        normalized_radius = max(
            MIN_NORMALIZED_BRUSH_RADIUS,
            min(float(self._brush_radius_pixels) / shortest_side, 1.0),
        )
        return MaskStroke(
            mode=self._brush_mode,
            radius_normalized=normalized_radius,
            points=tuple(self._active_points),
        )

    def _append_enclosed_fill(self, point: MaskPoint) -> bool:
        """Record a right-click fill only when the seed is fully enclosed.

        The current painted mask is the boundary.  Keeping the action as one
        normalized seed makes it replayable after a frame switch or project
        reload, unlike storing a one-off pixel mask.
        """

        candidate_mask = self._mask.copy()
        fill = MaskStroke(
            mode=MASK_MODE_PAINT,
            radius_normalized=MIN_NORMALIZED_BRUSH_RADIUS,
            points=(point,),
            is_fill=True,
        )
        if not _rasterize_enclosed_fill(candidate_mask, fill):
            return False
        self._strokes.append(fill)
        self._rebuild_mask()
        return True

    def _rebuild_mask(self) -> None:
        if self._frame_bgr is None:
            self._mask = np.empty((0, 0), dtype=np.uint8)
            self._mask_overlay_image = QImage()
            return
        frame_height, frame_width = self._frame_bgr.shape[:2]
        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        strokes = list(self._strokes)
        if self._active_points:
            strokes.append(self._build_active_stroke())
        for stroke in strokes:
            _rasterize_stroke(mask, stroke)
        self._mask = mask
        self._mask_overlay_image = _build_mask_overlay_qimage(mask)

    def _paint_mask_outline(self, painter: QPainter, target_rect: QRectF) -> None:
        if not self.has_selection():
            return
        contours, _ = cv2.findContours(
            np.where(self._mask > 0, 255, 0).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        scale_x = target_rect.width() / max(self._mask.shape[1], 1)
        scale_y = target_rect.height() / max(self._mask.shape[0], 1)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(MASK_OUTLINE_COLOR, 1.5))
        for contour in contours:
            if len(contour) < 2:
                continue
            polygon = [
                QPointF(
                    target_rect.left() + float(point[0][0]) * scale_x,
                    target_rect.top() + float(point[0][1]) * scale_y,
                )
                for point in contour
            ]
            painter.drawPolygon(polygon)

    def _paint_brush_cursor(self, painter: QPainter, target_rect: QRectF) -> None:
        if (
            not self._can_edit_mask()
            or self._hover_position is None
            or not target_rect.contains(self._hover_position)
        ):
            return
        cursor_color = (
            BRUSH_CURSOR_ERASE_COLOR
            if self._brush_mode == MASK_MODE_ERASE
            else BRUSH_CURSOR_PAINT_COLOR
        )
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(cursor_color, 1.5))
        radius = float(self._brush_radius_pixels)
        painter.drawEllipse(self._hover_position, radius, radius)

    def _cancel_active_stroke(self) -> None:
        if not self._active_points:
            return
        self._active_points = []
        self._rebuild_mask()


# ### Mask rasterization helpers ###
def rasterize_mask_strokes(
    frame_size: tuple[int, int],
    strokes: list[MaskStroke],
) -> np.ndarray:
    """Rasterize normalized strokes into a uint8 mask of ``(width, height)``."""

    frame_width = max(0, int(frame_size[0]))
    frame_height = max(0, int(frame_size[1]))
    if frame_width <= 0 or frame_height <= 0:
        return np.empty((0, 0), dtype=np.uint8)
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    for stroke in strokes:
        _rasterize_stroke(mask, stroke)
    return mask


def _rasterize_stroke(mask: np.ndarray, stroke: MaskStroke) -> None:
    if stroke.is_fill:
        _rasterize_enclosed_fill(mask, stroke)
        return
    frame_height, frame_width = mask.shape
    shortest_side = max(1, min(frame_width, frame_height))
    radius = max(1, int(round(stroke.radius_normalized * shortest_side)))
    value = 0 if stroke.mode == MASK_MODE_ERASE else 255
    pixel_points = [
        (
            min(max(int(round(point.x * (frame_width - 1))), 0), frame_width - 1),
            min(max(int(round(point.y * (frame_height - 1))), 0), frame_height - 1),
        )
        for point in stroke.points
    ]
    cv2.circle(mask, pixel_points[0], radius, value, thickness=-1, lineType=cv2.LINE_8)
    for start_point, end_point in zip(pixel_points, pixel_points[1:]):
        cv2.line(
            mask,
            start_point,
            end_point,
            value,
            thickness=radius * 2,
            lineType=cv2.LINE_8,
        )
        cv2.circle(
            mask,
            end_point,
            radius,
            value,
            thickness=-1,
            lineType=cv2.LINE_8,
        )


def _rasterize_enclosed_fill(mask: np.ndarray, fill: MaskStroke) -> bool:
    """Apply a fill seed when its 4-connected region does not reach an edge."""

    if not fill.is_fill or fill.mode != MASK_MODE_PAINT or mask.size == 0:
        return False
    frame_height, frame_width = mask.shape
    seed = fill.points[0]
    seed_x = min(
        max(int(round(seed.x * (frame_width - 1))), 0),
        frame_width - 1,
    )
    seed_y = min(
        max(int(round(seed.y * (frame_height - 1))), 0),
        frame_height - 1,
    )
    if mask[seed_y, seed_x] != 0:
        return False

    boundaries = np.where(mask > 0, 255, 0).astype(np.uint8)
    flood_mask = np.zeros((frame_height + 2, frame_width + 2), dtype=np.uint8)
    cv2.floodFill(
        boundaries,
        flood_mask,
        (seed_x, seed_y),
        255,
        flags=4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8),
    )
    filled_region = flood_mask[1:-1, 1:-1] == 255
    if not np.any(filled_region):
        return False
    if (
        np.any(filled_region[0, :])
        or np.any(filled_region[-1, :])
        or np.any(filled_region[:, 0])
        or np.any(filled_region[:, -1])
    ):
        return False
    mask[filled_region] = 255
    return True


# ### Image helpers ###
def _bgr_array_to_qimage(frame_bgr: np.ndarray) -> QImage:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = np.ascontiguousarray(frame_rgb)
    image = QImage(
        frame_rgb.data,
        frame_rgb.shape[1],
        frame_rgb.shape[0],
        int(frame_rgb.strides[0]),
        QImage.Format.Format_RGB888,
    )
    return image.copy()


def _build_mask_overlay_qimage(mask: np.ndarray) -> QImage:
    if mask.size == 0 or not np.any(mask > 0):
        return QImage()
    overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    overlay[:, :, 0] = MASK_OVERLAY_RGB[0]
    overlay[:, :, 1] = MASK_OVERLAY_RGB[1]
    overlay[:, :, 2] = MASK_OVERLAY_RGB[2]
    overlay[:, :, 3] = np.where(mask > 0, MASK_OVERLAY_ALPHA, 0).astype(np.uint8)
    overlay = np.ascontiguousarray(overlay)
    image = QImage(
        overlay.data,
        overlay.shape[1],
        overlay.shape[0],
        int(overlay.strides[0]),
        QImage.Format.Format_RGBA8888,
    )
    return image.copy()


def _get_aspect_fit_rect(
    content_width: int,
    content_height: int,
    available_rect: QRectF,
) -> QRectF:
    if content_width <= 0 or content_height <= 0 or available_rect.isEmpty():
        return QRectF()
    scale = min(
        available_rect.width() / float(content_width),
        available_rect.height() / float(content_height),
    )
    width = float(content_width) * scale
    height = float(content_height) * scale
    return QRectF(
        available_rect.center().x() - width / 2.0,
        available_rect.center().y() - height / 2.0,
        width,
        height,
    )
