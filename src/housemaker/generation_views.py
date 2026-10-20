# ### Imports ###
from __future__ import annotations

import math

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QWidget

from housemaker.generation_state import (
    MASK_MODE_ERASE,
    MASK_MODE_PAINT,
    MAX_MASK_STROKES_PER_FRAME,
    MaskPoint,
    MaskStroke,
)
from housemaker.video_source import normalize_video_frame

# ### Constants ###
VIEW_BACKGROUND_COLOR = QColor("#15181d")
VIEW_EMPTY_TEXT_COLOR = QColor("#e5e7eb")
MASK_OVERLAY_RGB = (48, 190, 255)
SELECTED_MASK_OVERLAY_RGB = (255, 60, 60)
MASK_OVERLAY_ALPHA = 112
MASK_OUTLINE_COLOR = QColor(95, 215, 255, 230)
BRUSH_CURSOR_PAINT_COLOR = QColor(95, 215, 255, 235)
BRUSH_CURSOR_ERASE_COLOR = QColor(255, 174, 92, 235)
DEFAULT_BRUSH_RADIUS_PIXELS = 24
MIN_NORMALIZED_BRUSH_RADIUS = 1e-6
PREVIEW_CHECKER_LIGHT_COLOR = QColor("#66707c")
PREVIEW_CHECKER_DARK_COLOR = QColor("#404852")
PREVIEW_CHECKER_SIZE_PIXELS = 12


# ### Video mask widget ###
class VideoInpaintView(QWidget):
    """Aspect-fit video view with brush and enclosed-region mask editing."""

    strokes_changed = Signal(object)
    frame_changed = Signal()
    selected_blob_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame_bgr: np.ndarray | None = None
        self._frame_image = QImage()
        self._mask = np.empty((0, 0), dtype=np.uint8)
        self._mask_overlay_image = QImage()
        self._strokes: list[MaskStroke] = []
        self._blob_labels = np.empty((0, 0), dtype=np.int32)
        self._blob_indices: tuple[int, ...] = ()
        self._selected_blob_index: int | None = None
        self._press_on_blob = False
        self._reference_preview_bgra: np.ndarray | None = None
        self._reference_preview_image = QImage()
        self._active_points: list[MaskPoint] = []
        self._brush_mode = MASK_MODE_PAINT
        self._brush_radius_pixels = DEFAULT_BRUSH_RADIUS_PIXELS
        self._interaction_enabled = True
        self._hover_position: QPointF | None = None
        self._empty_text = "No video loaded"
        self.setMinimumSize(160, 120)
        self.setMouseTracking(True)
        self.setToolTip(
            "Left-drag to paint; click a painted object to select it. "
            "Right-click a painted object to remove it, or inside a closed "
            "painted outline to fill the enclosed area."
        )

    def set_frame(
        self,
        frame_bgr: np.ndarray | None,
        strokes: list[MaskStroke] | None = None,
    ) -> None:
        self._cancel_active_stroke()
        self.clear_reference_preview()
        if frame_bgr is None:
            self._frame_bgr = None
            self._frame_image = QImage()
            self._mask = np.empty((0, 0), dtype=np.uint8)
            self._mask_overlay_image = QImage()
            self._strokes = []
            self._refresh_blob_components()
            self.update()
            self.frame_changed.emit()
            return

        normalized_frame = normalize_video_frame(np.asarray(frame_bgr))
        self._frame_bgr = np.ascontiguousarray(normalized_frame).copy()
        self._frame_image = _bgr_array_to_qimage(self._frame_bgr)
        self._strokes = list(strokes or [])
        self._rebuild_mask()
        self.update()
        self.frame_changed.emit()

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

    def get_mask_shape(self) -> tuple[int, int]:
        """Report the source-mask height and width without copying its pixels."""

        return self._mask.shape

    def get_blob_masks(self) -> tuple[np.ndarray, ...]:
        """Return independent full-frame masks in top-to-bottom, left order."""

        return tuple(
            np.where(self._blob_labels == label, 255, 0).astype(np.uint8)
            for label in self._blob_indices
        )

    def get_blob_count(self) -> int:
        """Count cached connected objects without constructing pixel masks."""

        return len(self._blob_indices)

    def get_selected_blob_index(self) -> int | None:
        return self._selected_blob_index

    def select_blob(self, index: int | None) -> None:
        if index is not None and not 0 <= index < len(self._blob_indices):
            raise IndexError("The selected inpaint object does not exist.")
        if index == self._selected_blob_index:
            return
        self._selected_blob_index = index
        self._update_mask_overlay()
        self.update()
        self.selected_blob_changed.emit()

    def get_selected_blob_mask(self) -> np.ndarray:
        mask = np.zeros_like(self._mask)
        if (
            self._selected_blob_index is not None
            and self._blob_labels.shape == self._mask.shape
        ):
            label = self._blob_indices[self._selected_blob_index]
            mask[self._blob_labels == label] = 255
        return mask

    def get_selected_blob_crop(self, padding_ratio: float = 0.08) -> np.ndarray:
        if self._frame_bgr is None:
            return np.empty((0, 0, 4), dtype=np.uint8)
        return _build_masked_frame_crop(
            self._frame_bgr, self.get_selected_blob_mask(), padding_ratio,
        )

    def get_frame_bgr(self) -> np.ndarray | None:
        if self._frame_bgr is None:
            return None
        return self._frame_bgr.copy()

    def has_frame(self) -> bool:
        """Report frame availability without copying the image buffer."""

        return self._frame_bgr is not None

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

    def set_reference_preview_bgra(self, preview_bgra: np.ndarray) -> None:
        """Show an isolated edited-object preview without mutating source data."""

        preview = np.asarray(preview_bgra)
        if (
            preview.ndim != 3
            or preview.shape[0] <= 0
            or preview.shape[1] <= 0
            or preview.shape[2] != 4
        ):
            raise ValueError(
                "Reference preview must be a non-empty BGRA image."
            )
        self._cancel_active_stroke()
        if preview.dtype != np.uint8:
            preview = np.clip(preview, 0, 255).astype(np.uint8)
        self._reference_preview_bgra = np.ascontiguousarray(preview).copy()
        self._reference_preview_image = _bgra_array_to_qimage(
            self._reference_preview_bgra
        )
        self.update()

    def clear_reference_preview(self) -> None:
        """Restore source-frame display while retaining the object selection."""

        if self._reference_preview_bgra is None:
            return
        self._reference_preview_bgra = None
        self._reference_preview_image = QImage()
        self.update()

    def has_reference_preview(self) -> bool:
        return self._reference_preview_bgra is not None

    def get_reference_preview_bgra(self) -> np.ndarray | None:
        if self._reference_preview_bgra is None:
            return None
        return self._reference_preview_bgra.copy()

    def clear_mask(self) -> None:
        self._cancel_active_stroke()
        selected_mask = self.get_selected_blob_mask()
        rows, columns = np.nonzero(selected_mask)
        if rows.size == 0:
            return
        height, width = selected_mask.shape
        removed = self._append_blob_removal(MaskPoint(
            x=int(columns[0]) / max(width - 1, 1),
            y=int(rows[0]) / max(height - 1, 1),
        ))
        if not removed:
            return
        self.strokes_changed.emit(self.get_strokes())
        self.update()

    def fill_enclosed_regions(self) -> int:
        """Turn every closed painted outline into a replayable filled mask."""

        self._cancel_active_stroke()
        if self._mask.size == 0 or not self.has_selection():
            return 0
        frame_height, frame_width = self._mask.shape
        remaining_stroke_capacity = max(
            0,
            MAX_MASK_STROKES_PER_FRAME - len(self._strokes),
        )
        if remaining_stroke_capacity == 0:
            return 0
        unselected = np.where(self._mask == 0, 1, 0).astype(np.uint8)
        component_count, labels, stats, _centroids = (
            cv2.connectedComponentsWithStats(unselected, connectivity=4)
        )
        fills: list[MaskStroke] = []
        for component_index in range(1, component_count):
            left = int(stats[component_index, cv2.CC_STAT_LEFT])
            top = int(stats[component_index, cv2.CC_STAT_TOP])
            width = int(stats[component_index, cv2.CC_STAT_WIDTH])
            height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
            if (
                left <= 0
                or top <= 0
                or left + width >= frame_width
                or top + height >= frame_height
            ):
                continue
            candidate_rows, candidate_columns = np.nonzero(
                labels == component_index
            )
            if candidate_rows.size == 0:
                continue
            seed_x = int(candidate_columns[0])
            seed_y = int(candidate_rows[0])
            fills.append(
                MaskStroke(
                    mode=MASK_MODE_PAINT,
                    radius_normalized=MIN_NORMALIZED_BRUSH_RADIUS,
                    points=(
                        MaskPoint(
                            x=seed_x / max(frame_width - 1, 1),
                            y=seed_y / max(frame_height - 1, 1),
                        ),
                    ),
                    is_fill=True,
                )
            )
            if len(fills) >= remaining_stroke_capacity:
                break
        if not fills:
            return 0
        self._strokes.extend(fills)
        self._rebuild_mask()
        self.strokes_changed.emit(self.get_strokes())
        self.update()
        return len(fills)

    def build_selected_object_crop(
        self,
        padding_ratio: float = 0.08,
    ) -> np.ndarray:
        """Return the selected object's BGRA crop, not its neighboring blobs."""

        return self.get_selected_blob_crop(padding_ratio)

    def build_selected_object_crops(
        self,
        padding_ratio: float = 0.08,
    ) -> tuple[np.ndarray, ...]:
        """Return one independently masked BGRA crop per disconnected blob."""

        if self._frame_bgr is None or not self.has_selection():
            return ()
        crops: list[np.ndarray] = []
        for component_mask in self.get_blob_masks():
            crop = _build_masked_frame_crop(
                self._frame_bgr,
                component_mask,
                padding_ratio,
            )
            if crop.size:
                crops.append(crop)
        return tuple(crops)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.fillRect(self.rect(), VIEW_BACKGROUND_COLOR)

        display_image = self._get_display_image()
        if display_image.isNull():
            painter.setPen(VIEW_EMPTY_TEXT_COLOR)
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                self._empty_text,
            )
            return

        target_rect = self._get_image_target_rect()
        if self.has_reference_preview():
            self._paint_preview_checkerboard(painter, target_rect)
            painter.drawImage(target_rect, display_image)
            return

        painter.drawImage(target_rect, display_image)
        if not self._mask_overlay_image.isNull():
            painter.drawImage(target_rect, self._mask_overlay_image)
        self._paint_mask_outline(painter, target_rect)
        self._paint_brush_cursor(painter, target_rect)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.RightButton and self._can_edit_mask():
            point = self._widget_to_mask_point(event.position())
            if point is not None:
                self._cancel_active_stroke()
                removed = self._append_blob_removal(point)
                if removed or self._append_enclosed_fill(point):
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
        clicked_blob = self._blob_index_at(point)
        self._press_on_blob = (
            clicked_blob is not None and self._brush_mode == MASK_MODE_PAINT
        )
        if self._press_on_blob:
            self.select_blob(clicked_blob)
        else:
            self._rebuild_mask()
        self.update()
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        self._hover_position = QPointF(event.position())
        if self._active_points and event.buttons() & Qt.MouseButton.LeftButton:
            point = self._widget_to_mask_point(event.position())
            if point is not None and self._point_is_distinct(point):
                self._press_on_blob = False
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
            self._press_on_blob = False
            self._active_points.append(point)
        if self._press_on_blob:
            self._active_points = []
            self._press_on_blob = False
            self.update()
            event.accept()
            return
        preferred_point = self._active_points[-1]
        self._strokes.append(self._build_active_stroke())
        self._active_points = []
        self._rebuild_mask(preferred_point=preferred_point)
        self.strokes_changed.emit(self.get_strokes())
        self.update()
        event.accept()

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        self._hover_position = None
        self.update()
        super().leaveEvent(event)

    def _can_edit_mask(self) -> bool:
        return (
            self._interaction_enabled
            and self._frame_bgr is not None
            and not self.has_reference_preview()
        )

    def _get_image_target_rect(self) -> QRectF:
        display_image = self._get_display_image()
        if display_image.isNull():
            return QRectF()
        return _get_aspect_fit_rect(
            display_image.width(),
            display_image.height(),
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

        if len(self._strokes) >= MAX_MASK_STROKES_PER_FRAME:
            return False
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
        self._rebuild_mask(preferred_point=point)
        return True

    def _append_blob_removal(self, point: MaskPoint) -> bool:
        if (
            len(self._strokes) >= MAX_MASK_STROKES_PER_FRAME
            or self._blob_index_at(point) is None
        ):
            return False
        self._strokes.append(MaskStroke(
            mode=MASK_MODE_ERASE,
            radius_normalized=MIN_NORMALIZED_BRUSH_RADIUS,
            points=(point,),
            is_fill=True,
        ))
        self._rebuild_mask()
        return True

    def _blob_index_at(self, point: MaskPoint) -> int | None:
        if self._blob_labels.size == 0:
            return None
        height, width = self._blob_labels.shape
        x = min(max(round(point.x * (width - 1)), 0), width - 1)
        y = min(max(round(point.y * (height - 1)), 0), height - 1)
        label = int(self._blob_labels[y, x])
        if label == 0:
            return None
        return self._blob_indices.index(label)

    def _rebuild_mask(self, preferred_point: MaskPoint | None = None) -> None:
        if self._frame_bgr is None:
            self._mask = np.empty((0, 0), dtype=np.uint8)
            self._mask_overlay_image = QImage()
            self._refresh_blob_components()
            return
        frame_height, frame_width = self._frame_bgr.shape[:2]
        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        strokes = list(self._strokes)
        if self._active_points:
            strokes.append(self._build_active_stroke())
        for stroke in strokes:
            _rasterize_stroke(mask, stroke)
        self._mask = mask
        if not self._active_points:
            self._refresh_blob_components(preferred_point)
        self._update_mask_overlay()

    def _refresh_blob_components(
        self, preferred_point: MaskPoint | None = None,
    ) -> None:
        """Cache connectivity once per completed edit, retaining object identity."""

        previous_index = self._selected_blob_index
        previous_mask = None
        if previous_index is not None and self._blob_labels.size:
            previous_label = self._blob_indices[previous_index]
            previous_mask = self._blob_labels == previous_label
        if self._mask.size:
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                self._mask, connectivity=8,
            )
            self._blob_labels = labels
            self._blob_indices = tuple(sorted(
                range(1, count),
                key=lambda label: (
                    int(stats[label, cv2.CC_STAT_TOP]),
                    int(stats[label, cv2.CC_STAT_LEFT]),
                ),
            ))
        else:
            self._blob_labels = np.empty((0, 0), dtype=np.int32)
            self._blob_indices = ()
        preserved_index = None
        if (
            previous_mask is not None
            and previous_mask.shape == self._blob_labels.shape
            and self._blob_indices
        ):
            overlap = np.bincount(
                self._blob_labels[previous_mask],
                minlength=max(self._blob_indices) + 1,
            )
            overlap[0] = 0
            label = int(np.argmax(overlap))
            if overlap[label]:
                preserved_index = self._blob_indices.index(label)
        selected_index = (
            self._blob_index_at(preferred_point)
            if preferred_point is not None else None
        )
        if selected_index is None:
            selected_index = preserved_index
        if selected_index is None and self._blob_indices:
            selected_index = min(previous_index or 0, len(self._blob_indices) - 1)
        self._selected_blob_index = selected_index
        changed_object = selected_index != preserved_index
        if previous_index != selected_index or changed_object:
            self.selected_blob_changed.emit()

    def _update_mask_overlay(self) -> None:
        selected_mask = self.get_selected_blob_mask()
        if self._active_points and not self._press_on_blob:
            _rasterize_stroke(selected_mask, self._build_active_stroke())
        self._mask_overlay_image = _build_mask_overlay_qimage(
            self._mask, selected_mask,
        )

    def _paint_mask_outline(
        self,
        painter: QPainter,
        target_rect: QRectF,
    ) -> None:
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
        self._press_on_blob = False
        self._rebuild_mask()

    def _get_display_image(self) -> QImage:
        if not self._reference_preview_image.isNull():
            return self._reference_preview_image
        return self._frame_image

    @staticmethod
    def _paint_preview_checkerboard(
        painter: QPainter,
        target_rect: QRectF,
    ) -> None:
        painter.save()
        painter.setClipRect(target_rect)
        cell_size = PREVIEW_CHECKER_SIZE_PIXELS
        left = math.floor(target_rect.left())
        top = math.floor(target_rect.top())
        right = math.ceil(target_rect.right())
        bottom = math.ceil(target_rect.bottom())
        for row, y in enumerate(range(top, bottom + cell_size, cell_size)):
            for column, x in enumerate(
                range(left, right + cell_size, cell_size)
            ):
                color = (
                    PREVIEW_CHECKER_LIGHT_COLOR
                    if (row + column) % 2 == 0
                    else PREVIEW_CHECKER_DARK_COLOR
                )
                painter.fillRect(x, y, cell_size, cell_size, color)
        painter.restore()


# ### Mask crop helpers ###
def _build_masked_frame_crop(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    padding_ratio: float,
) -> np.ndarray:
    """Crop one frame around the nonzero mask while preserving transparency."""

    bounds = _get_mask_crop_bounds(mask, padding_ratio)
    if bounds is None:
        return np.empty((0, 0, 4), dtype=np.uint8)
    left, top, right, bottom = bounds
    crop_bgr = frame_bgr[top:bottom, left:right]
    crop_alpha = mask[top:bottom, left:right]
    crop_bgra = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
    crop_bgra[crop_alpha == 0, :3] = 0
    crop_bgra[:, :, 3] = crop_alpha
    return np.ascontiguousarray(crop_bgra)


def _get_mask_crop_bounds(
    mask: np.ndarray,
    padding_ratio: float,
) -> tuple[int, int, int, int] | None:
    """Return left/top/right/bottom bounds shared by aligned mask crops."""

    mask_rows, mask_columns = np.nonzero(mask > 0)
    if mask_rows.size == 0:
        return None
    frame_height, frame_width = mask.shape
    selection_width = int(mask_columns.max() - mask_columns.min() + 1)
    selection_height = int(mask_rows.max() - mask_rows.min() + 1)
    padding = round(
        max(selection_width, selection_height) * max(0.0, padding_ratio)
    )
    left = max(0, int(mask_columns.min()) - padding)
    right = min(frame_width, int(mask_columns.max()) + padding + 1)
    top = max(0, int(mask_rows.min()) - padding)
    bottom = min(frame_height, int(mask_rows.max()) + padding + 1)
    return left, top, right, bottom


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
        if stroke.mode == MASK_MODE_ERASE:
            _rasterize_blob_removal(mask, stroke)
        else:
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


def _rasterize_blob_removal(mask: np.ndarray, stroke: MaskStroke) -> None:
    """Replay one connected-object removal using its normalized painted seed."""

    if mask.size == 0:
        return
    height, width = mask.shape
    point = stroke.points[0]
    x = min(max(round(point.x * (width - 1)), 0), width - 1)
    y = min(max(round(point.y * (height - 1)), 0), height - 1)
    if mask[y, x]:
        cv2.floodFill(mask, None, (x, y), 0, flags=8)


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


def _bgra_array_to_qimage(frame_bgra: np.ndarray) -> QImage:
    frame_rgba = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2RGBA)
    frame_rgba = np.ascontiguousarray(frame_rgba)
    image = QImage(
        frame_rgba.data,
        frame_rgba.shape[1],
        frame_rgba.shape[0],
        int(frame_rgba.strides[0]),
        QImage.Format.Format_RGBA8888,
    )
    return image.copy()


def _build_mask_overlay_qimage(
    mask: np.ndarray, selected_mask: np.ndarray | None = None,
) -> QImage:
    if mask.size == 0 or not np.any(mask > 0):
        return QImage()
    overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    overlay[:, :, 0] = MASK_OVERLAY_RGB[0]
    overlay[:, :, 1] = MASK_OVERLAY_RGB[1]
    overlay[:, :, 2] = MASK_OVERLAY_RGB[2]
    if selected_mask is not None:
        overlay[selected_mask > 0, :3] = SELECTED_MASK_OVERLAY_RGB
    overlay[:, :, 3] = np.where(mask > 0, MASK_OVERLAY_ALPHA, 0).astype(
        np.uint8
    )
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
