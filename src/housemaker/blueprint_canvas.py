# ### Imports ###
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import QWidget

from housemaker.models import VERTEX_HIT_RADIUS_SCREEN, Vertex, VertexData, snap_point

# ### Constants ###
CANVAS_BACKGROUND_COLOR = QColor("#1c1f24")
CANVAS_PANEL_COLOR = QColor("#252a31")
EDGE_COLOR = QColor("#63c0ff")
PREVIEW_EDGE_COLOR = QColor("#f6c85f")
GUIDE_COLOR = QColor("#39d98a")
VERTEX_FILL_COLOR = QColor("#ffffff")
ACTIVE_VERTEX_FILL_COLOR = QColor("#ff7f50")
SELECTED_VERTEX_FILL_COLOR = QColor("#90cdf4")
VERTEX_OUTLINE_COLOR = QColor("#20242a")
TEXT_COLOR = QColor("#f5f7fa")
IMAGE_MARGIN = 16.0
VERTEX_RADIUS_SCREEN = 6.0
AXIS_SNAP_TOLERANCE_SCREEN = 10.0
DRAG_THRESHOLD_SCREEN = 4.0

# ### Snapshot models ###
@dataclass
class CanvasSnapshot:
    vertex_data: VertexData
    active_vertex_id: int | None
    selected_vertex_id: int | None
    preview_point: tuple[float, float] | None


@dataclass(frozen=True)
class SnapGuide:
    source_vertex_id: int
    axis: str


@dataclass(frozen=True)
class SnapPreview:
    point: tuple[float, float]
    guides: list[SnapGuide]


@dataclass(frozen=True)
class AxisSnapCandidate:
    source_vertex_id: int
    axis: str
    value: float
    distance: float


# ### Widgets ###
class BlueprintCanvas(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.vertex_data = VertexData()
        self.blueprint_image: QImage | None = None
        self.blueprint_path: str | None = None
        self.active_vertex_id: int | None = None
        self.selected_vertex_id: int | None = None
        self.preview_point: tuple[float, float] | None = None
        self.preview_guides: list[SnapGuide] = []
        self.undo_stack: list[CanvasSnapshot] = []
        self.pressed_vertex_id: int | None = None
        self.drag_vertex_id: int | None = None
        self.drag_press_position: QPointF | None = None

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(640, 480)

        self.undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        self.undo_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.undo_shortcut.activated.connect(self.undo_last_step)

    def load_blueprint(self, file_path: str) -> None:
        image = _load_qimage_from_path(file_path)

        self.blueprint_image = image
        self.blueprint_path = file_path
        self.vertex_data.reset()
        self.active_vertex_id = None
        self.selected_vertex_id = None
        self.preview_point = None
        self.preview_guides = []
        self.undo_stack.clear()
        self._reset_pointer_state()
        self.update()

    def undo_last_step(self) -> None:
        if not self.undo_stack:
            return

        snapshot = self.undo_stack.pop()
        self.vertex_data.copy_from(snapshot.vertex_data)
        self.active_vertex_id = snapshot.active_vertex_id
        self.selected_vertex_id = snapshot.selected_vertex_id
        self.preview_point = snapshot.preview_point
        self.preview_guides = []
        self._reset_pointer_state()
        self.update()

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_Delete:
            self._delete_selected_vertex()
            event.accept()
            return

        super().keyPressEvent(event)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        self.setFocus(Qt.FocusReason.MouseFocusReason)

        if self.blueprint_image is None:
            super().mousePressEvent(event)
            return

        if event.button() == Qt.MouseButton.RightButton:
            self._apply_active_chain()
            event.accept()
            return

        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        hit_vertex = self._find_vertex_at(event.position())
        if hit_vertex is not None:
            self.selected_vertex_id = hit_vertex.id
            self.pressed_vertex_id = hit_vertex.id
            self.drag_press_position = QPointF(event.position())
            self.drag_vertex_id = None
            self.update()
            event.accept()
            return

        image_point = self._widget_to_image(event.position())
        if image_point is None:
            event.accept()
            return

        snap_preview = self._build_connection_preview(image_point, event.modifiers())
        self._handle_new_vertex_click(snap_preview.point)

        self.update()
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self.blueprint_image is None:
            super().mouseMoveEvent(event)
            return

        if self.pressed_vertex_id is not None and event.buttons() & Qt.MouseButton.LeftButton:
            if self._should_start_drag(event.position()):
                self._start_drag()

            if self.drag_vertex_id is not None:
                image_point = self._widget_to_image_clamped(event.position())
                self.vertex_data.move_vertex(
                    self.drag_vertex_id,
                    image_point.x(),
                    image_point.y(),
                )
                self.preview_guides = []
                self.update()
                event.accept()
                return

            event.accept()
            return

        if self.active_vertex_id is None:
            super().mouseMoveEvent(event)
            return

        image_point = self._widget_to_image(event.position())
        if image_point is None:
            self.preview_point = None
            self.preview_guides = []
        else:
            snap_preview = self._build_connection_preview(image_point, event.modifiers())
            self.preview_point = snap_preview.point
            self.preview_guides = snap_preview.guides

        self.update()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() != Qt.MouseButton.LeftButton or self.pressed_vertex_id is None:
            super().mouseReleaseEvent(event)
            return

        if self.drag_vertex_id is not None:
            self._reset_pointer_state()
            self.update()
            event.accept()
            return

        clicked_vertex = self.vertex_data.get_vertex(self.pressed_vertex_id)
        self._reset_pointer_state()
        if clicked_vertex is None:
            event.accept()
            return

        self._handle_existing_vertex_click(clicked_vertex)
        self.update()
        event.accept()

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        if self.drag_vertex_id is None and self.active_vertex_id is not None:
            self.preview_point = None
            self.preview_guides = []
            self.update()
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), CANVAS_BACKGROUND_COLOR)

        if self.blueprint_image is None:
            self._paint_empty_state(painter)
            return

        display_rect = self._image_display_rect()
        painter.fillRect(display_rect, CANVAS_PANEL_COLOR)
        painter.drawImage(display_rect, self.blueprint_image)

        self._paint_edges(painter)
        self._paint_preview_guides(painter)
        self._paint_preview_edge(painter)
        self._paint_vertices(painter)
        self._paint_overlay_text(painter)

    def _apply_active_chain(self) -> None:
        if self.active_vertex_id is None:
            return

        self._push_undo_state()
        self.active_vertex_id = None
        self.preview_point = None
        self.preview_guides = []
        self._reset_pointer_state()
        self.update()

    def _handle_existing_vertex_click(self, vertex: Vertex) -> None:
        self.selected_vertex_id = vertex.id

        if self.active_vertex_id is None:
            self.active_vertex_id = vertex.id
            self.preview_point = (vertex.x, vertex.y)
            self.preview_guides = []
            return

        if vertex.id == self.active_vertex_id:
            return

        self._push_undo_state()
        self.vertex_data.add_edge(self.active_vertex_id, vertex.id)
        self.active_vertex_id = vertex.id
        self.preview_point = (vertex.x, vertex.y)
        self.preview_guides = []

    def _handle_new_vertex_click(self, point: tuple[float, float]) -> None:
        self._push_undo_state()

        new_vertex = self.vertex_data.add_vertex(*point)
        if self.active_vertex_id is not None:
            self.vertex_data.add_edge(self.active_vertex_id, new_vertex.id)

        self.active_vertex_id = new_vertex.id
        self.selected_vertex_id = new_vertex.id
        self.preview_point = point
        self.preview_guides = []

    def _push_undo_state(self) -> None:
        snapshot = CanvasSnapshot(
            vertex_data=self.vertex_data.clone(),
            active_vertex_id=self.active_vertex_id,
            selected_vertex_id=self.selected_vertex_id,
            preview_point=self.preview_point,
        )
        self.undo_stack.append(snapshot)

    def _delete_selected_vertex(self) -> None:
        if self.selected_vertex_id is None:
            return

        if self.vertex_data.get_vertex(self.selected_vertex_id) is None:
            self.selected_vertex_id = None
            self.update()
            return

        self._push_undo_state()
        deleted_vertex_id = self.selected_vertex_id
        self.vertex_data.delete_vertex(deleted_vertex_id)

        if self.active_vertex_id == deleted_vertex_id:
            self.active_vertex_id = None
            self.preview_point = None

        self.selected_vertex_id = None
        self.preview_guides = []
        self._reset_pointer_state()
        self.update()

    def _build_connection_preview(
        self,
        image_point: QPointF,
        modifiers: Qt.KeyboardModifier,
    ) -> SnapPreview:
        raw_x, raw_y = self._clamp_image_point(image_point.x(), image_point.y())
        modifier_bits = getattr(modifiers, "value", modifiers)
        control_modifier = Qt.KeyboardModifier.ControlModifier.value

        if self.active_vertex_id is None:
            return SnapPreview(point=(raw_x, raw_y), guides=[])

        base_vertex = self.vertex_data.get_vertex(self.active_vertex_id)
        if base_vertex is None:
            return SnapPreview(point=(raw_x, raw_y), guides=[])

        axis_candidates = self._find_axis_snap_candidates(
            (raw_x, raw_y),
            excluded_vertex_ids={base_vertex.id},
        )

        if modifier_bits & control_modifier:
            return self._build_axis_only_preview((raw_x, raw_y), axis_candidates)

        snapped_x, snapped_y = snap_point(base_vertex, raw_x, raw_y)
        angle_snapped_point = self._clamp_image_point(snapped_x, snapped_y)
        axis_and_angle_preview = self._build_axis_and_angle_preview(
            base_vertex,
            (raw_x, raw_y),
            axis_candidates,
        )
        if axis_and_angle_preview is not None:
            return axis_and_angle_preview

        return SnapPreview(point=angle_snapped_point, guides=[])

    def _clamp_image_point(self, x: float, y: float) -> tuple[float, float]:
        if self.blueprint_image is None:
            return x, y

        clamped_x = min(max(x, 0.0), float(self.blueprint_image.width() - 1))
        clamped_y = min(max(y, 0.0), float(self.blueprint_image.height() - 1))
        return clamped_x, clamped_y

    def _find_vertex_at(self, widget_point: QPointF) -> Vertex | None:
        closest_vertex: Vertex | None = None
        closest_distance = VERTEX_HIT_RADIUS_SCREEN

        for vertex in self.vertex_data.vertices:
            vertex_point = self._image_to_widget(vertex.x, vertex.y)
            distance = math.hypot(
                widget_point.x() - vertex_point.x(),
                widget_point.y() - vertex_point.y(),
            )
            if distance <= closest_distance:
                closest_vertex = vertex
                closest_distance = distance

        return closest_vertex

    def _should_start_drag(self, widget_point: QPointF) -> bool:
        if self.drag_vertex_id is not None or self.drag_press_position is None:
            return False

        distance = math.hypot(
            widget_point.x() - self.drag_press_position.x(),
            widget_point.y() - self.drag_press_position.y(),
        )
        return distance >= DRAG_THRESHOLD_SCREEN

    def _start_drag(self) -> None:
        if self.pressed_vertex_id is None:
            return

        self._push_undo_state()
        self.drag_vertex_id = self.pressed_vertex_id

    def _reset_pointer_state(self) -> None:
        self.pressed_vertex_id = None
        self.drag_vertex_id = None
        self.drag_press_position = None

    def _find_axis_snap_candidates(
        self,
        point: tuple[float, float],
        excluded_vertex_ids: set[int] | None = None,
    ) -> list[AxisSnapCandidate]:
        excluded_ids = excluded_vertex_ids or set()
        tolerance = self._screen_distance_to_image(AXIS_SNAP_TOLERANCE_SCREEN)
        point_x, point_y = point
        best_x_match: tuple[float, Vertex] | None = None
        best_y_match: tuple[float, Vertex] | None = None

        for vertex in self.vertex_data.vertices:
            if vertex.id in excluded_ids:
                continue

            x_distance = abs(point_x - vertex.x)
            if x_distance <= tolerance and (
                best_x_match is None or x_distance < best_x_match[0]
            ):
                best_x_match = (x_distance, vertex)

            y_distance = abs(point_y - vertex.y)
            if y_distance <= tolerance and (
                best_y_match is None or y_distance < best_y_match[0]
            ):
                best_y_match = (y_distance, vertex)

        candidates: list[AxisSnapCandidate] = []
        if best_x_match is not None:
            candidates.append(
                AxisSnapCandidate(
                    source_vertex_id=best_x_match[1].id,
                    axis="x",
                    value=best_x_match[1].x,
                    distance=best_x_match[0],
                )
            )

        if best_y_match is not None:
            candidates.append(
                AxisSnapCandidate(
                    source_vertex_id=best_y_match[1].id,
                    axis="y",
                    value=best_y_match[1].y,
                    distance=best_y_match[0],
                )
            )

        return candidates

    def _build_axis_and_angle_preview(
        self,
        base_vertex: Vertex,
        raw_point: tuple[float, float],
        axis_candidates: list[AxisSnapCandidate],
    ) -> SnapPreview | None:
        preview_candidates: list[tuple[float, SnapPreview]] = []

        for axis_candidate in axis_candidates:
            snapped_point = self._solve_axis_locked_angle_point(
                base_vertex=base_vertex,
                raw_point=raw_point,
                axis_candidate=axis_candidate,
            )
            if snapped_point is None:
                continue

            preview_candidates.append(
                (
                    self._point_distance(snapped_point, raw_point),
                    SnapPreview(
                        point=snapped_point,
                        guides=[
                            SnapGuide(
                                source_vertex_id=axis_candidate.source_vertex_id,
                                axis=axis_candidate.axis,
                            )
                        ],
                    ),
                )
            )

        if not preview_candidates:
            return None

        preview_candidates.sort(key=lambda candidate: candidate[0])
        return preview_candidates[0][1]

    def _build_axis_only_preview(
        self,
        raw_point: tuple[float, float],
        axis_candidates: list[AxisSnapCandidate],
    ) -> SnapPreview:
        snapped_x, snapped_y = raw_point
        guides: list[SnapGuide] = []

        for axis_candidate in axis_candidates:
            if axis_candidate.axis == "x":
                snapped_x = axis_candidate.value
            else:
                snapped_y = axis_candidate.value

            guides.append(
                SnapGuide(
                    source_vertex_id=axis_candidate.source_vertex_id,
                    axis=axis_candidate.axis,
                )
            )

        snapped_point = self._clamp_image_point(snapped_x, snapped_y)
        return SnapPreview(point=snapped_point, guides=guides)

    def _solve_axis_locked_angle_point(
        self,
        base_vertex: Vertex,
        raw_point: tuple[float, float],
        axis_candidate: AxisSnapCandidate,
    ) -> tuple[float, float] | None:
        best_point: tuple[float, float] | None = None
        best_distance = float("inf")

        for angle_degrees in range(0, 360, 10):
            angle_radians = math.radians(float(angle_degrees))
            direction_x = math.cos(angle_radians)
            direction_y = math.sin(angle_radians)

            if axis_candidate.axis == "x":
                if abs(direction_x) < 1e-6:
                    continue
                ray_distance = (axis_candidate.value - base_vertex.x) / direction_x
                if ray_distance <= 1e-6:
                    continue
                candidate_point = (
                    axis_candidate.value,
                    base_vertex.y + ray_distance * direction_y,
                )
            else:
                if abs(direction_y) < 1e-6:
                    continue
                ray_distance = (axis_candidate.value - base_vertex.y) / direction_y
                if ray_distance <= 1e-6:
                    continue
                candidate_point = (
                    base_vertex.x + ray_distance * direction_x,
                    axis_candidate.value,
                )

            if not self._is_point_within_image(candidate_point):
                continue

            point_distance = self._point_distance(candidate_point, raw_point)
            if point_distance < best_distance:
                best_point = candidate_point
                best_distance = point_distance

        return best_point

    def _is_point_within_image(self, point: tuple[float, float]) -> bool:
        if self.blueprint_image is None:
            return True

        point_x, point_y = point
        return (
            0.0 <= point_x <= float(self.blueprint_image.width() - 1)
            and 0.0 <= point_y <= float(self.blueprint_image.height() - 1)
        )

    def _point_distance(
        self,
        point_a: tuple[float, float],
        point_b: tuple[float, float],
    ) -> float:
        return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])

    def _screen_distance_to_image(self, screen_distance: float) -> float:
        if self.blueprint_image is None:
            return screen_distance

        display_rect = self._image_display_rect()
        scale = display_rect.width() / float(self.blueprint_image.width())
        if scale <= 0.0:
            return screen_distance

        return screen_distance / scale

    def _image_display_rect(self) -> QRectF:
        if self.blueprint_image is None:
            return QRectF()

        available_rect = QRectF(
            IMAGE_MARGIN,
            IMAGE_MARGIN,
            max(1.0, self.width() - IMAGE_MARGIN * 2.0),
            max(1.0, self.height() - IMAGE_MARGIN * 2.0),
        )

        image_width = float(self.blueprint_image.width())
        image_height = float(self.blueprint_image.height())
        scale = min(
            available_rect.width() / image_width,
            available_rect.height() / image_height,
        )

        display_width = image_width * scale
        display_height = image_height * scale
        display_x = available_rect.center().x() - display_width / 2.0
        display_y = available_rect.center().y() - display_height / 2.0
        return QRectF(display_x, display_y, display_width, display_height)

    def _widget_to_image(self, widget_point: QPointF) -> QPointF | None:
        if self.blueprint_image is None:
            return None

        display_rect = self._image_display_rect()
        if not display_rect.contains(widget_point):
            return None

        normalized_x = (widget_point.x() - display_rect.left()) / display_rect.width()
        normalized_y = (widget_point.y() - display_rect.top()) / display_rect.height()
        image_x = normalized_x * float(self.blueprint_image.width())
        image_y = normalized_y * float(self.blueprint_image.height())
        clamped_x, clamped_y = self._clamp_image_point(image_x, image_y)
        return QPointF(clamped_x, clamped_y)

    def _widget_to_image_clamped(self, widget_point: QPointF) -> QPointF:
        if self.blueprint_image is None:
            return QPointF()

        display_rect = self._image_display_rect()
        clamped_widget_x = min(max(widget_point.x(), display_rect.left()), display_rect.right())
        clamped_widget_y = min(max(widget_point.y(), display_rect.top()), display_rect.bottom())
        return self._widget_to_image(QPointF(clamped_widget_x, clamped_widget_y)) or QPointF()

    def _image_to_widget(self, image_x: float, image_y: float) -> QPointF:
        if self.blueprint_image is None:
            return QPointF()

        display_rect = self._image_display_rect()
        widget_x = display_rect.left() + (
            image_x / float(self.blueprint_image.width())
        ) * display_rect.width()
        widget_y = display_rect.top() + (
            image_y / float(self.blueprint_image.height())
        ) * display_rect.height()
        return QPointF(widget_x, widget_y)

    def _paint_empty_state(self, painter: QPainter) -> None:
        painter.setPen(QPen(TEXT_COLOR))
        painter.setFont(QFont("Segoe UI", 15))
        painter.drawText(
            self.rect(),
            int(Qt.AlignmentFlag.AlignCenter),
            "Select a blueprint image to start tracing.",
        )

    def _paint_edges(self, painter: QPainter) -> None:
        edge_pen = QPen(EDGE_COLOR, 2.0)
        painter.setPen(edge_pen)

        for edge in self.vertex_data.edges:
            start_vertex = self.vertex_data.get_vertex(edge.start_vertex_id)
            end_vertex = self.vertex_data.get_vertex(edge.end_vertex_id)
            if start_vertex is None or end_vertex is None:
                continue

            painter.drawLine(
                self._image_to_widget(start_vertex.x, start_vertex.y),
                self._image_to_widget(end_vertex.x, end_vertex.y),
            )

    def _paint_preview_guides(self, painter: QPainter) -> None:
        if self.preview_point is None or not self.preview_guides:
            return

        guide_pen = QPen(GUIDE_COLOR, 1.5, Qt.PenStyle.DashLine)
        guide_pen.setDashPattern([3.0, 5.0])
        painter.setPen(guide_pen)

        for guide in self.preview_guides:
            source_vertex = self.vertex_data.get_vertex(guide.source_vertex_id)
            if source_vertex is None:
                continue

            if guide.axis == "x":
                painter.drawLine(
                    self._image_to_widget(source_vertex.x, source_vertex.y),
                    self._image_to_widget(source_vertex.x, self.preview_point[1]),
                )
                continue

            painter.drawLine(
                self._image_to_widget(source_vertex.x, source_vertex.y),
                self._image_to_widget(self.preview_point[0], source_vertex.y),
            )

    def _paint_preview_edge(self, painter: QPainter) -> None:
        if self.active_vertex_id is None or self.preview_point is None:
            return

        active_vertex = self.vertex_data.get_vertex(self.active_vertex_id)
        if active_vertex is None:
            return

        preview_pen = QPen(PREVIEW_EDGE_COLOR, 2.0, Qt.PenStyle.DashLine)
        painter.setPen(preview_pen)
        painter.drawLine(
            self._image_to_widget(active_vertex.x, active_vertex.y),
            self._image_to_widget(self.preview_point[0], self.preview_point[1]),
        )

    def _paint_vertices(self, painter: QPainter) -> None:
        label_font = QFont("Segoe UI", 9)
        painter.setFont(label_font)

        for vertex in self.vertex_data.vertices:
            center = self._image_to_widget(vertex.x, vertex.y)
            is_active = vertex.id == self.active_vertex_id
            is_selected = vertex.id == self.selected_vertex_id

            painter.setPen(QPen(VERTEX_OUTLINE_COLOR, 1.5))
            if is_active:
                painter.setBrush(ACTIVE_VERTEX_FILL_COLOR)
            elif is_selected:
                painter.setBrush(SELECTED_VERTEX_FILL_COLOR)
            else:
                painter.setBrush(VERTEX_FILL_COLOR)
            painter.drawEllipse(center, VERTEX_RADIUS_SCREEN, VERTEX_RADIUS_SCREEN)

            painter.setPen(QPen(TEXT_COLOR))
            painter.drawText(
                center + QPointF(10.0, -10.0),
                str(vertex.id),
            )

    def _paint_overlay_text(self, painter: QPainter) -> None:
        if self.blueprint_path is None:
            return

        overlay_rect = QRectF(24.0, 24.0, 460.0, 86.0)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(10, 12, 16, 180))
        painter.drawRoundedRect(overlay_rect, 10.0, 10.0)

        painter.setPen(QPen(TEXT_COLOR))
        painter.setFont(QFont("Segoe UI", 9))
        overlay_lines = [
            f"Blueprint: {Path(self.blueprint_path).name}",
            "Left click: add/connect, select, or drag vertices | Right click: apply chain",
            "Delete: remove selected vertex | Ctrl+Z: undo | Hold Ctrl while moving: free placement",
        ]

        line_y = overlay_rect.top() + 24.0
        for line in overlay_lines:
            painter.drawText(QPointF(overlay_rect.left() + 14.0, line_y), line)
            line_y += 22.0


# ### File helpers ###
def _load_qimage_from_path(file_path: str) -> QImage:
    image_bgr = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Unable to open blueprint image: {file_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_height, image_width, channel_count = image_rgb.shape
    bytes_per_line = channel_count * image_width
    return QImage(
        image_rgb.data,
        image_width,
        image_height,
        bytes_per_line,
        QImage.Format.Format_RGB888,
    ).copy()
