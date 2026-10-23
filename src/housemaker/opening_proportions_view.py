# ### Imports ###
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QTransform,
)
from PySide6.QtWidgets import QWidget

# ### Constants ###
DEFAULT_OUTLINE = ((0.3, 0.2), (0.7, 0.2), (0.7, 0.8), (0.3, 0.8))
OUTLINE_COLOR = QColor("#57d8ff")
APEX_COLOR = QColor("#ff80bf")
HANDLE_RADIUS = 7.0
HANDLE_HIT_RADIUS = 14.0


# ### Opening outline view ###
class OpeningProportionsView(QWidget):
    """Source-image coordinates stay unchanged while the user zooms and pans."""

    outline_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image = QImage()
        self._outline = DEFAULT_OUTLINE
        self._arch_apex = (0.5, 0.2)
        self._is_doorway = False
        self._zoom = 1.0
        self._pan = QPointF()
        self._active_handle: int | None = None
        self._pan_origin: QPointF | None = None
        self.setMinimumSize(240, 160)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setToolTip(
            "Drag the corners onto the opening. Scroll to zoom; middle-drag to pan."
        )

    # ### Source and annotation state ###
    @property
    def outline(self) -> tuple[tuple[float, float], ...]:
        return self._outline

    @property
    def arch_apex(self) -> tuple[float, float] | None:
        return self._arch_apex if self._is_doorway else None

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._image.width(), self._image.height()

    def has_frame(self) -> bool:
        return not self._image.isNull()

    def set_frame(self, frame: np.ndarray | None) -> None:
        previous_size = self.frame_size
        if frame is None:
            self._image = QImage()
        else:
            array = np.ascontiguousarray(frame, dtype=np.uint8)
            if array.ndim != 3 or array.shape[2] != 3:
                raise ValueError("The opening reference must be a BGR image.")
            height, width = array.shape[:2]
            self._image = QImage(
                array.data,
                width,
                height,
                array.strides[0],
                QImage.Format.Format_BGR888,
            ).copy()
        if previous_size != self.frame_size:
            self.reset_zoom()
        self.update()

    def set_doorway(self, enabled: bool) -> None:
        self._is_doorway = bool(enabled)
        self.update()

    def set_outline(
        self,
        outline: tuple[tuple[float, float], ...],
        arch_apex: tuple[float, float] | None = None,
    ) -> None:
        if len(outline) != 4:
            raise ValueError("An opening outline needs four corners.")
        self._outline = tuple(_normalized_point(point) for point in outline)
        self._arch_apex = (
            _normalized_point(arch_apex)
            if arch_apex is not None
            else self._spring_midpoint()
        )
        self.update()

    def reset_zoom(self) -> None:
        self._zoom = 1.0
        self._pan = QPointF()
        self.update()

    # ### Image-coordinate mapping ###
    def image_rect(self) -> QRectF:
        if self._image.isNull() or self.width() <= 0 or self.height() <= 0:
            return QRectF()
        fit = min(
            self.width() / self._image.width(),
            self.height() / self._image.height(),
        )
        width = self._image.width() * fit * self._zoom
        height = self._image.height() * fit * self._zoom
        return QRectF(
            (self.width() - width) * 0.5 + self._pan.x(),
            (self.height() - height) * 0.5 + self._pan.y(),
            width,
            height,
        )

    def image_to_widget(self, point: tuple[float, float]) -> QPointF:
        rect = self.image_rect()
        return QPointF(
            rect.left() + point[0] * rect.width(),
            rect.top() + point[1] * rect.height(),
        )

    def widget_to_image(self, position: QPointF) -> tuple[float, float]:
        rect = self.image_rect()
        if rect.isEmpty():
            return 0.5, 0.5
        return _normalized_point(
            (
                (position.x() - rect.left()) / rect.width(),
                (position.y() - rect.top()) / rect.height(),
            )
        )

    def _spring_midpoint(self) -> tuple[float, float]:
        left, right = self._outline[:2]
        return (left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5

    def _arch_curve_points(self) -> tuple[tuple[float, float], ...]:
        """Project an elliptical arch from the opening's rectified plane."""
        transform = QTransform()
        square = [QPointF(0, 0), QPointF(1, 0), QPointF(1, 1), QPointF(0, 1)]
        if not QTransform.quadToQuad(
            square, [QPointF(*point) for point in self._outline], transform
        ):
            return self._outline[0], self._arch_apex, self._outline[1]
        inverse, invertible = transform.inverted()
        if not invertible:
            return self._outline[0], self._arch_apex, self._outline[1]
        apex = inverse.map(QPointF(*self._arch_apex))
        points: list[tuple[float, float]] = []
        for angle in np.linspace(0.0, math.pi, 33):
            # A centered apex gives the same half-ellipse as the doorway mesh.
            # A laterally marked apex shears that ellipse without moving its ends.
            sine = math.sin(angle)
            point = transform.map(
                QPointF(
                    0.5 - 0.5 * math.cos(angle) + (apex.x() - 0.5) * sine,
                    apex.y() * sine,
                )
            )
            if not math.isfinite(point.x()) or not math.isfinite(point.y()):
                return self._outline[0], self._arch_apex, self._outline[1]
            points.append((point.x(), point.y()))
        return tuple(points)

    # ### Painting ###
    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#15181d"))
        if self._image.isNull():
            painter.setPen(QColor("#e5e7eb"))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "Load a video or image"
            )
            return
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(self.image_rect(), self._image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        corners = [self.image_to_widget(point) for point in self._outline]
        path = QPainterPath(corners[0])
        if self._is_doorway:
            for point in self._arch_curve_points()[1:]:
                path.lineTo(self.image_to_widget(point))
            guide_color = QColor(APEX_COLOR)
            guide_color.setAlpha(170)
            painter.setPen(QPen(guide_color, 1.0, Qt.PenStyle.DashLine))
            painter.drawLine(corners[0], corners[1])
        else:
            path.lineTo(corners[1])
        path.lineTo(corners[2])
        path.lineTo(corners[3])
        path.closeSubpath()
        painter.setPen(QPen(QColor(0, 0, 0, 210), 4.0))
        painter.drawPath(path)
        painter.setPen(QPen(OUTLINE_COLOR, 2.0))
        painter.drawPath(path)
        for point in corners:
            self._paint_handle(painter, point, OUTLINE_COLOR)
        if self._is_doorway:
            self._paint_handle(
                painter, self.image_to_widget(self._arch_apex), APEX_COLOR
            )

    @staticmethod
    def _paint_handle(painter: QPainter, point: QPointF, color: QColor) -> None:
        painter.setPen(QPen(QColor("#101318"), 2.0))
        painter.setBrush(color)
        painter.drawEllipse(point, HANDLE_RADIUS, HANDLE_RADIUS)

    # ### Mouse input ###
    def _handle_at(self, position: QPointF) -> int | None:
        handles = list(self._outline)
        if self._is_doorway:
            handles.append(self._arch_apex)
        nearest: tuple[float, int] | None = None
        for index, point in enumerate(handles):
            delta = self.image_to_widget(point) - position
            distance = math.hypot(delta.x(), delta.y())
            if distance <= HANDLE_HIT_RADIUS and (
                nearest is None or distance < nearest[0]
            ):
                nearest = distance, index
        return nearest[1] if nearest is not None else None

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if not self.has_frame():
            return
        self.setFocus()
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_origin = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        elif event.button() == Qt.MouseButton.LeftButton:
            self._active_handle = self._handle_at(event.position())
            if self._active_handle is not None:
                event.accept()
            else:
                super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._pan_origin is not None:
            self._pan += event.position() - self._pan_origin
            self._pan_origin = event.position()
            self.update()
            event.accept()
            return
        if self._active_handle is not None:
            self._move_handle(
                self._active_handle, self.widget_to_image(event.position())
            )
            event.accept()
            return
        self.setCursor(
            Qt.CursorShape.SizeAllCursor
            if self.has_frame() and self._handle_at(event.position()) is not None
            else Qt.CursorShape.ArrowCursor
        )

    def _move_handle(self, index: int, point: tuple[float, float]) -> None:
        previous = self._arch_apex if index == 4 else self._outline[index]
        if previous == point:
            return
        if index == 4:
            self._arch_apex = point
        else:
            old_midpoint = self._spring_midpoint()
            outline = list(self._outline)
            outline[index] = point
            self._outline = tuple(outline)
            if index < 2:
                midpoint = self._spring_midpoint()
                self._arch_apex = _normalized_point(
                    (
                        self._arch_apex[0] + midpoint[0] - old_midpoint[0],
                        self._arch_apex[1] + midpoint[1] - old_midpoint[1],
                    )
                )
        self.update()
        self.outline_changed.emit()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._active_handle = None
        self._pan_origin = None
        self.unsetCursor()
        event.accept()

    def wheelEvent(self, event) -> None:
        if not self.has_frame() or event.angleDelta().y() == 0:
            event.ignore()
            return
        anchor = self.widget_to_image(event.position())
        previous_zoom = self._zoom
        self._zoom = min(
            24.0, max(1.0, previous_zoom * 1.2 ** (event.angleDelta().y() / 120.0))
        )
        if self._zoom == previous_zoom:
            return
        self._pan += event.position() - self.image_to_widget(anchor)
        self.update()
        event.accept()


# ### Coordinate helpers ###
def _normalized_point(point) -> tuple[float, float]:
    x, y = float(point[0]), float(point[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("Opening outline points must be finite.")
    return min(1.0, max(0.0, x)), min(1.0, max(0.0, y))
