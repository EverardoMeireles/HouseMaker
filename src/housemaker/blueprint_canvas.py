# ### Imports ###
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import cv2
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPolygonF,
    QShortcut,
)
from PySide6.QtWidgets import QWidget

from housemaker.models import (
    DEFAULT_IMAGE_OFFSET,
    DEFAULT_IMAGE_SCALE,
    Edge,
    RoomData,
    VERTEX_HIT_RADIUS_SCREEN,
    Vertex,
    VertexData,
    snap_point,
)
from housemaker.uv_layout import initialize_room_uv_map_size

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
ROOM_LABEL_BACKGROUND_COLOR = QColor(10, 12, 16, 170)
IMAGE_MARGIN = 16.0
VERTEX_RADIUS_SCREEN = 6.0
EDGE_HIT_TOLERANCE_SCREEN = 8.0
AXIS_SNAP_TOLERANCE_SCREEN = 10.0
CENTER_SNAP_TOLERANCE_SCREEN = 10.0
CENTER_SNAP_EQUAL_ANGLE_TOLERANCE_DEGREES = 1.0
ROOM_FILL_ALPHA = 72
DRAG_THRESHOLD_SCREEN = 4.0
MIN_ZOOM_SCALE = 1.0
MAX_ZOOM_SCALE = 16.0
ZOOM_STEP_FACTOR = 1.15

# ### Snapshot models ###
@dataclass
class CanvasSnapshot:
    vertex_data: VertexData
    rooms: list[RoomData]
    active_vertex_id: int | None
    selected_vertex_id: int | None
    selected_vertex_ids: set[int]
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
class CenterSnapCandidate:
    source_vertex_ids: tuple[int, int, int, int]
    point: tuple[float, float]
    distance: float


@dataclass(frozen=True)
class AxisSnapCandidate:
    source_vertex_id: int
    axis: str
    value: float
    distance: float


@dataclass(frozen=True)
class VertexPairCenter:
    source_vertex_ids: tuple[int, int]
    point: tuple[float, float]


@dataclass(frozen=True)
class EdgeHit:
    edge: Edge
    point: tuple[float, float]
    distance: float


# ### Widgets ###
class BlueprintCanvas(QWidget):
    rooms_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.vertex_data = VertexData()
        self.rooms: list[RoomData] = []
        self.blueprint_image: QImage | None = None
        self.blueprint_path: str | None = None
        self.image_scale = DEFAULT_IMAGE_SCALE
        self.image_offset_x = DEFAULT_IMAGE_OFFSET
        self.image_offset_y = DEFAULT_IMAGE_OFFSET
        self.active_vertex_id: int | None = None
        self.selected_vertex_id: int | None = None
        self.selected_vertex_ids: set[int] = set()
        self.preview_point: tuple[float, float] | None = None
        self.preview_guides: list[SnapGuide] = []
        self.undo_stack: list[CanvasSnapshot] = []
        self.pressed_vertex_id: int | None = None
        self.drag_vertex_id: int | None = None
        self.drag_press_position: QPointF | None = None
        self.zoom_scale = MIN_ZOOM_SCALE
        self.view_offset = QPointF(0.0, 0.0)
        self.is_panning = False
        self.pan_press_position: QPointF | None = None
        self.pan_start_offset = QPointF(0.0, 0.0)
        self.snap_middle_equal_angle_only = True
        self.pending_room_name: str | None = None
        self.pending_room_vertex_ids: tuple[int, ...] = ()
        self.pending_room_wall_height_meters = 3.0

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(640, 480)

        self.undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        self.undo_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.undo_shortcut.activated.connect(self.undo_last_step)

    def load_blueprint(
        self,
        file_path: str,
        vertex_data: VertexData | None = None,
        rooms: list[RoomData] | None = None,
        image_scale: float = DEFAULT_IMAGE_SCALE,
        image_offset_x: float = DEFAULT_IMAGE_OFFSET,
        image_offset_y: float = DEFAULT_IMAGE_OFFSET,
    ) -> None:
        image = _load_qimage_from_path(file_path)
        self._set_level_contents(
            vertex_data=vertex_data or VertexData(),
            rooms=rooms,
            blueprint_image=image,
            blueprint_path=file_path,
            image_scale=image_scale,
            image_offset_x=image_offset_x,
            image_offset_y=image_offset_y,
        )

    def set_level_vertex_data(self, vertex_data: VertexData) -> None:
        self.set_level_data(
            vertex_data=vertex_data,
            rooms=self.rooms,
            image_path=self.blueprint_path,
            image_scale=self.image_scale,
            image_offset_x=self.image_offset_x,
            image_offset_y=self.image_offset_y,
        )

    def set_level_data(
        self,
        vertex_data: VertexData,
        rooms: list[RoomData] | None,
        image_path: str | None,
        image_scale: float = DEFAULT_IMAGE_SCALE,
        image_offset_x: float = DEFAULT_IMAGE_OFFSET,
        image_offset_y: float = DEFAULT_IMAGE_OFFSET,
    ) -> None:
        blueprint_image: QImage | None = None
        if image_path and Path(image_path).exists():
            try:
                blueprint_image = _load_qimage_from_path(image_path)
            except ValueError:
                blueprint_image = None

        self._set_level_contents(
            vertex_data=vertex_data,
            rooms=rooms,
            blueprint_image=blueprint_image,
            blueprint_path=image_path,
            image_scale=image_scale,
            image_offset_x=image_offset_x,
            image_offset_y=image_offset_y,
        )

    def get_image_size_pixels(self) -> tuple[float, float] | None:
        if self.blueprint_image is None:
            return None

        return (
            float(self.blueprint_image.width()),
            float(self.blueprint_image.height()),
        )

    def set_image_transform(
        self,
        image_scale: float,
        image_offset_x: float,
        image_offset_y: float,
    ) -> None:
        self.image_scale = max(0.01, float(image_scale))
        self.image_offset_x = float(image_offset_x)
        self.image_offset_y = float(image_offset_y)
        self.update()

    def set_snap_middle_equal_angle_only(self, enabled: bool) -> None:
        self.snap_middle_equal_angle_only = bool(enabled)
        self.preview_point = None
        self.preview_guides = []
        self.update()

    def get_selected_vertex_ids(self) -> list[int]:
        existing_vertex_ids = {vertex.id for vertex in self.vertex_data.vertices}
        return sorted(
            vertex_id
            for vertex_id in self.selected_vertex_ids
            if vertex_id in existing_vertex_ids
        )

    def start_room_designation(
        self,
        room_name: str,
        vertex_ids: list[int],
        wall_height_meters: float,
    ) -> None:
        self.pending_room_name = room_name.strip()
        self.pending_room_vertex_ids = tuple(sorted(set(vertex_ids)))
        self.pending_room_wall_height_meters = wall_height_meters
        self.preview_point = None
        self.preview_guides = []
        self.update()

    def delete_room_at_index(self, room_index: int) -> bool:
        if room_index < 0 or room_index >= len(self.rooms):
            return False

        self._push_undo_state()
        del self.rooms[room_index]
        self.rooms_changed.emit()
        self.update()
        return True

    def _set_level_contents(
        self,
        vertex_data: VertexData,
        rooms: list[RoomData] | None,
        blueprint_image: QImage | None,
        blueprint_path: str | None,
        image_scale: float,
        image_offset_x: float,
        image_offset_y: float,
    ) -> None:
        self.blueprint_image = blueprint_image
        self.blueprint_path = blueprint_path
        self.image_scale = max(0.01, float(image_scale))
        self.image_offset_x = float(image_offset_x)
        self.image_offset_y = float(image_offset_y)
        self.vertex_data = vertex_data
        self.rooms = rooms if rooms is not None else []
        self.active_vertex_id = None
        self.selected_vertex_id = None
        self.selected_vertex_ids.clear()
        self.preview_point = None
        self.preview_guides = []
        self.undo_stack.clear()
        self._reset_room_designation()
        self._reset_pointer_state()
        self._reset_view()
        self.update()

    def undo_last_step(self) -> None:
        if not self.undo_stack:
            return

        snapshot = self.undo_stack.pop()
        self.vertex_data.copy_from(snapshot.vertex_data)
        self.rooms.clear()
        self.rooms.extend(copy.deepcopy(snapshot.rooms))
        self.active_vertex_id = snapshot.active_vertex_id
        self.selected_vertex_id = snapshot.selected_vertex_id
        self.selected_vertex_ids = set(snapshot.selected_vertex_ids)
        self.preview_point = snapshot.preview_point
        self.preview_guides = []
        self._reset_room_designation()
        self._reset_pointer_state()
        self.update()
        self.rooms_changed.emit()

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_Delete:
            self._delete_selected_vertex()
            event.accept()
            return

        super().keyPressEvent(event)

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        if self.blueprint_image is None:
            super().wheelEvent(event)
            return

        wheel_delta = event.angleDelta().y()
        if wheel_delta == 0:
            wheel_delta = event.pixelDelta().y()
        if wheel_delta == 0:
            event.accept()
            return

        zoom_factor = ZOOM_STEP_FACTOR ** (wheel_delta / 120.0)
        self._zoom_around_widget_point(event.position(), zoom_factor)
        event.accept()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        self.setFocus(Qt.FocusReason.MouseFocusReason)

        if self.blueprint_image is None:
            super().mousePressEvent(event)
            return

        if event.button() == Qt.MouseButton.RightButton:
            self._apply_active_chain()
            event.accept()
            return

        if event.button() == Qt.MouseButton.MiddleButton:
            self._start_panning(event.position())
            event.accept()
            return

        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        hit_vertex = self._find_vertex_at(event.position())
        if self.pending_room_name is not None:
            if hit_vertex is not None:
                self._finish_room_designation(hit_vertex.id)
                self.update()
            event.accept()
            return

        if hit_vertex is not None:
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self._toggle_vertex_selection(hit_vertex.id)
                self.update()
                event.accept()
                return

            self.selected_vertex_id = hit_vertex.id
            self.selected_vertex_ids = {hit_vertex.id}
            self.pressed_vertex_id = hit_vertex.id
            self.drag_press_position = QPointF(event.position())
            self.drag_vertex_id = None
            self.update()
            event.accept()
            return

        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            event.accept()
            return

        image_point = self._widget_to_image(event.position())
        if image_point is None:
            event.accept()
            return

        hit_edge = self._find_edge_at(event.position())
        if hit_edge is not None:
            self._handle_new_vertex_on_edge_click(hit_edge.point, hit_edge.edge)
            self.update()
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

        if self.is_panning and event.buttons() & Qt.MouseButton.MiddleButton:
            self._update_pan(event.position())
            event.accept()
            return

        if self.pressed_vertex_id is not None and event.buttons() & Qt.MouseButton.LeftButton:
            if self._should_start_drag(event.position()):
                self._start_drag()

            if self.drag_vertex_id is not None:
                image_point = self._widget_to_image_clamped(event.position())
                snap_preview = self._build_drag_preview(
                    image_point,
                    dragged_vertex_id=self.drag_vertex_id,
                )
                self.vertex_data.move_vertex(
                    self.drag_vertex_id,
                    snap_preview.point[0],
                    snap_preview.point[1],
                )
                self.preview_point = snap_preview.point
                self.preview_guides = snap_preview.guides
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
        if event.button() == Qt.MouseButton.MiddleButton:
            self._stop_panning()
            event.accept()
            return

        if event.button() != Qt.MouseButton.LeftButton or self.pressed_vertex_id is None:
            super().mouseReleaseEvent(event)
            return

        if self.drag_vertex_id is not None:
            self._join_dragged_vertex_to_edge(self.drag_vertex_id)
            self.preview_point = None
            self.preview_guides = []
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

        self._paint_rooms(painter)
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
        self.selected_vertex_ids = {vertex.id}

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
        self.selected_vertex_ids = {new_vertex.id}
        self.preview_point = point
        self.preview_guides = []

    def _join_dragged_vertex_to_edge(self, vertex_id: int) -> None:
        vertex = self.vertex_data.get_vertex(vertex_id)
        if vertex is None:
            return

        hit_edge = self._find_edge_at(
            self._image_to_widget(vertex.x, vertex.y),
            excluded_vertex_ids={vertex_id},
        )
        if hit_edge is None:
            return

        self.vertex_data.move_vertex(vertex_id, hit_edge.point[0], hit_edge.point[1])
        self.vertex_data.split_edge(
            hit_edge.edge.start_vertex_id,
            hit_edge.edge.end_vertex_id,
            vertex_id,
        )

    def _handle_new_vertex_on_edge_click(
        self,
        point: tuple[float, float],
        edge: Edge,
    ) -> None:
        self._push_undo_state()

        new_vertex = self.vertex_data.add_vertex(*point)
        self.vertex_data.split_edge(
            edge.start_vertex_id,
            edge.end_vertex_id,
            new_vertex.id,
        )
        if self.active_vertex_id is not None:
            self.vertex_data.add_edge(self.active_vertex_id, new_vertex.id)

        self.active_vertex_id = new_vertex.id
        self.selected_vertex_id = new_vertex.id
        self.selected_vertex_ids = {new_vertex.id}
        self.preview_point = point
        self.preview_guides = []

    def _push_undo_state(self) -> None:
        snapshot = CanvasSnapshot(
            vertex_data=self.vertex_data.clone(),
            rooms=copy.deepcopy(self.rooms),
            active_vertex_id=self.active_vertex_id,
            selected_vertex_id=self.selected_vertex_id,
            selected_vertex_ids=set(self.selected_vertex_ids),
            preview_point=self.preview_point,
        )
        self.undo_stack.append(snapshot)

    def _toggle_vertex_selection(self, vertex_id: int) -> None:
        if vertex_id in self.selected_vertex_ids:
            self.selected_vertex_ids.remove(vertex_id)
            if self.selected_vertex_id == vertex_id:
                self.selected_vertex_id = next(iter(self.selected_vertex_ids), None)
            return

        self.selected_vertex_ids.add(vertex_id)
        self.selected_vertex_id = vertex_id

    def _delete_selected_vertex(self) -> None:
        if self.selected_vertex_id is None:
            return

        if self.vertex_data.get_vertex(self.selected_vertex_id) is None:
            self.selected_vertex_ids.discard(self.selected_vertex_id)
            self.selected_vertex_id = None
            self.update()
            return

        self._push_undo_state()
        deleted_vertex_id = self.selected_vertex_id
        self.vertex_data.delete_vertex(deleted_vertex_id)
        self._remove_vertex_from_rooms(deleted_vertex_id)

        if self.active_vertex_id == deleted_vertex_id:
            self.active_vertex_id = None
            self.preview_point = None

        self.selected_vertex_id = None
        self.selected_vertex_ids.discard(deleted_vertex_id)
        self.preview_guides = []
        self._reset_pointer_state()
        self.update()

    def _finish_room_designation(self, center_vertex_id: int) -> None:
        if self.pending_room_name is None or len(self.pending_room_vertex_ids) < 3:
            self._reset_room_designation()
            return

        existing_vertex_ids = {vertex.id for vertex in self.vertex_data.vertices}
        room_vertex_ids = tuple(
            vertex_id
            for vertex_id in self.pending_room_vertex_ids
            if vertex_id in existing_vertex_ids and vertex_id != center_vertex_id
        )
        if len(room_vertex_ids) < 3:
            self._reset_room_designation()
            return

        self._push_undo_state()
        room = RoomData(
            name=self.pending_room_name,
            vertex_ids=room_vertex_ids,
            center_vertex_id=center_vertex_id,
            color_rgb=_build_random_room_color(),
        )
        initialize_room_uv_map_size(
            room=room,
            vertex_data=self.vertex_data,
            wall_height_meters=self.pending_room_wall_height_meters,
        )
        self.rooms.append(room)
        self.selected_vertex_id = center_vertex_id
        self.selected_vertex_ids.clear()
        self._reset_room_designation()
        self.rooms_changed.emit()

    def _reset_room_designation(self) -> None:
        self.pending_room_name = None
        self.pending_room_vertex_ids = ()
        self.pending_room_wall_height_meters = 3.0

    def _remove_vertex_from_rooms(self, deleted_vertex_id: int) -> None:
        original_rooms = list(self.rooms)
        updated_rooms: list[RoomData] = []
        for room in self.rooms:
            if room.center_vertex_id == deleted_vertex_id:
                continue

            remaining_vertex_ids = tuple(
                vertex_id
                for vertex_id in room.vertex_ids
                if vertex_id != deleted_vertex_id
            )
            if len(remaining_vertex_ids) < 3:
                continue

            updated_rooms.append(
                RoomData(
                    name=room.name,
                    vertex_ids=remaining_vertex_ids,
                    center_vertex_id=room.center_vertex_id,
                    color_rgb=room.color_rgb,
                    uv_map_width=room.uv_map_width,
                    uv_map_height=room.uv_map_height,
                    wall_uv_scales=copy.deepcopy(room.wall_uv_scales),
                    wall_uv_rotations=copy.deepcopy(room.wall_uv_rotations),
                    wall_uv_positions=copy.deepcopy(room.wall_uv_positions),
                    wall_subdivisions=copy.deepcopy(room.wall_subdivisions),
                    wall_subdivision_positions=copy.deepcopy(
                        room.wall_subdivision_positions
                    ),
                    wall_subdivision_source_ranges=copy.deepcopy(
                        room.wall_subdivision_source_ranges
                    ),
                    wall_textures=copy.deepcopy(room.wall_textures),
                )
            )

        self.rooms.clear()
        self.rooms.extend(updated_rooms)
        if self.rooms != original_rooms:
            self.rooms_changed.emit()

    def _build_connection_preview(
        self,
        image_point: QPointF,
        modifiers: Qt.KeyboardModifier,
    ) -> SnapPreview:
        raw_x, raw_y = self._clamp_image_point(image_point.x(), image_point.y())
        raw_point = (raw_x, raw_y)
        modifier_bits = getattr(modifiers, "value", modifiers)
        control_modifier = Qt.KeyboardModifier.ControlModifier.value
        center_candidate = self._find_center_snap_candidate(raw_point)
        if center_candidate is not None:
            return self._build_center_snap_preview(center_candidate)

        if self.active_vertex_id is None:
            return SnapPreview(point=raw_point, guides=[])

        base_vertex = self.vertex_data.get_vertex(self.active_vertex_id)
        if base_vertex is None:
            return SnapPreview(point=raw_point, guides=[])

        axis_candidates = self._find_axis_snap_candidates(
            raw_point,
            excluded_vertex_ids={base_vertex.id},
        )

        if modifier_bits & control_modifier:
            return self._build_axis_only_preview(raw_point, axis_candidates)

        snapped_x, snapped_y = snap_point(base_vertex, raw_x, raw_y)
        angle_snapped_point = self._clamp_image_point(snapped_x, snapped_y)
        axis_and_angle_preview = self._build_axis_and_angle_preview(
            base_vertex,
            raw_point,
            axis_candidates,
        )
        if axis_and_angle_preview is not None:
            return axis_and_angle_preview

        return SnapPreview(point=angle_snapped_point, guides=[])

    def _build_drag_preview(
        self,
        image_point: QPointF,
        dragged_vertex_id: int,
    ) -> SnapPreview:
        raw_point = self._clamp_image_point(image_point.x(), image_point.y())
        center_candidate = self._find_center_snap_candidate(
            raw_point,
            excluded_vertex_ids={dragged_vertex_id},
        )
        if center_candidate is not None:
            return self._build_center_snap_preview(center_candidate)

        axis_candidates = self._find_axis_snap_candidates(
            raw_point,
            excluded_vertex_ids={dragged_vertex_id},
        )
        return self._build_axis_only_preview(raw_point, axis_candidates)

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

    def _find_edge_at(
        self,
        widget_point: QPointF,
        excluded_vertex_ids: set[int] | None = None,
    ) -> EdgeHit | None:
        excluded_ids = excluded_vertex_ids or set()
        closest_hit: EdgeHit | None = None
        closest_distance = EDGE_HIT_TOLERANCE_SCREEN

        for edge in self.vertex_data.edges:
            if edge.start_vertex_id in excluded_ids:
                continue
            if edge.end_vertex_id in excluded_ids:
                continue

            start_vertex = self.vertex_data.get_vertex(edge.start_vertex_id)
            end_vertex = self.vertex_data.get_vertex(edge.end_vertex_id)
            if start_vertex is None or end_vertex is None:
                continue

            projection = _project_point_onto_widget_segment(
                point=widget_point,
                segment_start=self._image_to_widget(start_vertex.x, start_vertex.y),
                segment_end=self._image_to_widget(end_vertex.x, end_vertex.y),
            )
            if projection is None:
                continue

            projected_widget_point, segment_ratio = projection
            distance = math.hypot(
                widget_point.x() - projected_widget_point.x(),
                widget_point.y() - projected_widget_point.y(),
            )
            if distance > closest_distance:
                continue

            closest_hit = EdgeHit(
                edge=edge,
                point=(
                    start_vertex.x
                    + (end_vertex.x - start_vertex.x) * segment_ratio,
                    start_vertex.y
                    + (end_vertex.y - start_vertex.y) * segment_ratio,
                ),
                distance=distance,
            )
            closest_distance = distance

        return closest_hit

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
        self._stop_panning()

    def _reset_view(self) -> None:
        self.zoom_scale = MIN_ZOOM_SCALE
        self.view_offset = QPointF(0.0, 0.0)
        self._stop_panning()

    def _zoom_around_widget_point(
        self,
        widget_point: QPointF,
        zoom_factor: float,
    ) -> None:
        if self.blueprint_image is None:
            return

        old_rect = self._image_display_rect()
        base_rect = self._base_image_display_rect()
        if old_rect.width() <= 0.0 or old_rect.height() <= 0.0:
            return

        old_zoom_scale = self.zoom_scale
        new_zoom_scale = min(
            max(old_zoom_scale * zoom_factor, MIN_ZOOM_SCALE),
            MAX_ZOOM_SCALE,
        )
        if math.isclose(new_zoom_scale, old_zoom_scale):
            return

        if math.isclose(new_zoom_scale, MIN_ZOOM_SCALE):
            self._reset_view()
            self.update()
            return

        normalized_x = (widget_point.x() - old_rect.left()) / old_rect.width()
        normalized_y = (widget_point.y() - old_rect.top()) / old_rect.height()
        new_width = base_rect.width() * new_zoom_scale
        new_height = base_rect.height() * new_zoom_scale
        new_center = QPointF(
            widget_point.x() - (normalized_x - 0.5) * new_width,
            widget_point.y() - (normalized_y - 0.5) * new_height,
        )

        self.zoom_scale = new_zoom_scale
        self.view_offset = new_center - base_rect.center()
        self.update()

    def _start_panning(self, widget_point: QPointF) -> None:
        if self.zoom_scale <= MIN_ZOOM_SCALE:
            return

        self.is_panning = True
        self.pan_press_position = QPointF(widget_point)
        self.pan_start_offset = QPointF(self.view_offset)

    def _update_pan(self, widget_point: QPointF) -> None:
        if not self.is_panning or self.pan_press_position is None:
            return

        self.view_offset = self.pan_start_offset + (widget_point - self.pan_press_position)
        self.update()

    def _stop_panning(self) -> None:
        self.is_panning = False
        self.pan_press_position = None

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

    def _find_center_snap_candidate(
        self,
        point: tuple[float, float],
        excluded_vertex_ids: set[int] | None = None,
    ) -> CenterSnapCandidate | None:
        excluded_ids = excluded_vertex_ids or set()
        snap_vertices = [
            vertex
            for vertex in self.vertex_data.vertices
            if vertex.id not in excluded_ids
        ]
        if len(snap_vertices) < 4:
            return None

        tolerance = self._screen_distance_to_image(CENTER_SNAP_TOLERANCE_SCREEN)
        near_pair_centers: list[VertexPairCenter] = []
        for first_vertex, second_vertex in combinations(snap_vertices, 2):
            pair_center = (
                (first_vertex.x + second_vertex.x) / 2.0,
                (first_vertex.y + second_vertex.y) / 2.0,
            )
            if self._point_distance(pair_center, point) > tolerance:
                continue

            near_pair_centers.append(
                VertexPairCenter(
                    source_vertex_ids=(first_vertex.id, second_vertex.id),
                    point=pair_center,
                )
            )

        best_candidate: CenterSnapCandidate | None = None
        best_candidate_key: tuple[float, float] | None = None
        for first_pair, second_pair in combinations(near_pair_centers, 2):
            if _vertex_pairs_overlap(first_pair, second_pair):
                continue

            midpoint_spread = self._point_distance(first_pair.point, second_pair.point)
            if midpoint_spread > tolerance:
                continue

            center_point = (
                (first_pair.point[0] + second_pair.point[0]) / 2.0,
                (first_pair.point[1] + second_pair.point[1]) / 2.0,
            )
            center_distance = self._point_distance(center_point, point)
            if center_distance > tolerance:
                continue

            candidate_key = (center_distance, midpoint_spread)
            if best_candidate_key is not None and candidate_key >= best_candidate_key:
                continue

            combined_source_ids = sorted(
                first_pair.source_vertex_ids + second_pair.source_vertex_ids
            )
            source_vertex_ids = (
                combined_source_ids[0],
                combined_source_ids[1],
                combined_source_ids[2],
                combined_source_ids[3],
            )
            if self.snap_middle_equal_angle_only and not self._has_equal_corner_angles(
                source_vertex_ids
            ):
                continue

            best_candidate = CenterSnapCandidate(
                source_vertex_ids=source_vertex_ids,
                point=center_point,
                distance=center_distance,
            )
            best_candidate_key = candidate_key

        return best_candidate

    def _has_equal_corner_angles(
        self,
        source_vertex_ids: tuple[int, int, int, int],
    ) -> bool:
        vertices = [
            vertex
            for vertex_id in source_vertex_ids
            if (vertex := self.vertex_data.get_vertex(vertex_id)) is not None
        ]
        if len(vertices) != 4:
            return False

        center_x = sum(vertex.x for vertex in vertices) / 4.0
        center_y = sum(vertex.y for vertex in vertices) / 4.0
        ordered_vertices = sorted(
            vertices,
            key=lambda vertex: math.atan2(vertex.y - center_y, vertex.x - center_x),
        )
        corner_angles = [
            _calculate_corner_angle(
                previous_vertex=ordered_vertices[index - 1],
                current_vertex=ordered_vertices[index],
                next_vertex=ordered_vertices[(index + 1) % len(ordered_vertices)],
            )
            for index in range(len(ordered_vertices))
        ]
        return all(
            abs(corner_angle - 90.0) <= CENTER_SNAP_EQUAL_ANGLE_TOLERANCE_DEGREES
            for corner_angle in corner_angles
        )

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

    def _build_center_snap_preview(
        self,
        center_candidate: CenterSnapCandidate,
    ) -> SnapPreview:
        snapped_point = self._clamp_image_point(
            center_candidate.point[0],
            center_candidate.point[1],
        )
        return SnapPreview(
            point=snapped_point,
            guides=[
                SnapGuide(source_vertex_id=vertex_id, axis="center")
                for vertex_id in center_candidate.source_vertex_ids
            ],
        )

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
        base_rect = self._base_image_display_rect()
        if self.blueprint_image is None:
            return base_rect

        center = base_rect.center() + self.view_offset
        display_width = base_rect.width() * self.zoom_scale
        display_height = base_rect.height() * self.zoom_scale
        return QRectF(
            center.x() - display_width / 2.0,
            center.y() - display_height / 2.0,
            display_width,
            display_height,
        )

    def _base_image_display_rect(self) -> QRectF:
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

        display_width = image_width * scale * self.image_scale
        display_height = image_height * scale * self.image_scale
        display_center = available_rect.center() + QPointF(
            self.image_offset_x,
            self.image_offset_y,
        )
        display_x = display_center.x() - display_width / 2.0
        display_y = display_center.y() - display_height / 2.0
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
        empty_message = "Use Load image to select a blueprint for this level."
        if self.blueprint_path is not None:
            empty_message = (
                f"Image not found:\n{self.blueprint_path}\n\n"
                "Use Load image to select a replacement for this level."
            )
        painter.drawText(
            self.rect(),
            int(Qt.AlignmentFlag.AlignCenter),
            empty_message,
        )

    def _paint_rooms(self, painter: QPainter) -> None:
        label_font = QFont("Segoe UI", 10)
        label_font.setBold(True)

        for room in self.rooms:
            room_vertices = self._get_room_vertices(room)
            if len(room_vertices) < 3:
                continue

            ordered_vertices = _order_vertices_around_center(room_vertices)
            polygon = QPolygonF(
                [
                    self._image_to_widget(vertex.x, vertex.y)
                    for vertex in ordered_vertices
                ]
            )
            fill_color = QColor(
                room.color_rgb[0],
                room.color_rgb[1],
                room.color_rgb[2],
                ROOM_FILL_ALPHA,
            )

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(fill_color)
            painter.drawPolygon(polygon)

            label_point = self._get_room_label_point(room, ordered_vertices)
            label_rect = QRectF(
                label_point.x() - 90.0,
                label_point.y() - 14.0,
                180.0,
                28.0,
            )
            painter.setBrush(ROOM_LABEL_BACKGROUND_COLOR)
            painter.drawRoundedRect(label_rect, 6.0, 6.0)
            painter.setPen(QPen(TEXT_COLOR))
            painter.setFont(label_font)
            painter.drawText(
                label_rect,
                int(Qt.AlignmentFlag.AlignCenter),
                room.name,
            )

    def _get_room_vertices(self, room: RoomData) -> list[Vertex]:
        vertices: list[Vertex] = []
        for vertex_id in room.vertex_ids:
            vertex = self.vertex_data.get_vertex(vertex_id)
            if vertex is not None:
                vertices.append(vertex)

        return vertices

    def _get_room_label_point(
        self,
        room: RoomData,
        ordered_vertices: list[Vertex],
    ) -> QPointF:
        center_vertex = self.vertex_data.get_vertex(room.center_vertex_id)
        if center_vertex is not None:
            return self._image_to_widget(center_vertex.x, center_vertex.y)

        center_x = sum(vertex.x for vertex in ordered_vertices) / len(ordered_vertices)
        center_y = sum(vertex.y for vertex in ordered_vertices) / len(ordered_vertices)
        return self._image_to_widget(center_x, center_y)

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
        preview_widget_point = self._image_to_widget(
            self.preview_point[0],
            self.preview_point[1],
        )

        for guide in self.preview_guides:
            source_vertex = self.vertex_data.get_vertex(guide.source_vertex_id)
            if source_vertex is None:
                continue

            if guide.axis == "center":
                painter.drawLine(
                    self._image_to_widget(source_vertex.x, source_vertex.y),
                    preview_widget_point,
                )
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
        if self.drag_vertex_id is not None:
            return

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
            is_selected = (
                vertex.id == self.selected_vertex_id
                or vertex.id in self.selected_vertex_ids
            )

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

        overlay_lines = [
            f"Blueprint: {Path(self.blueprint_path).name}",
            "Left click: add/connect/select/drag | Shift click: multi-select vertices",
            "Mouse wheel: zoom | Delete: remove selected vertex | Ctrl+Z: undo | Hold Ctrl: free placement",
        ]
        if self.pending_room_name is not None:
            overlay_lines.append("Click a vertex to set the current room center.")

        overlay_rect = QRectF(24.0, 24.0, 520.0, 20.0 + len(overlay_lines) * 22.0)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(10, 12, 16, 180))
        painter.drawRoundedRect(overlay_rect, 10.0, 10.0)

        painter.setPen(QPen(TEXT_COLOR))
        painter.setFont(QFont("Segoe UI", 9))
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


# ### Geometry helpers ###
def _project_point_onto_widget_segment(
    point: QPointF,
    segment_start: QPointF,
    segment_end: QPointF,
) -> tuple[QPointF, float] | None:
    segment_delta_x = segment_end.x() - segment_start.x()
    segment_delta_y = segment_end.y() - segment_start.y()
    segment_length_squared = (
        segment_delta_x * segment_delta_x
        + segment_delta_y * segment_delta_y
    )
    if segment_length_squared <= 1e-6:
        return None

    point_delta_x = point.x() - segment_start.x()
    point_delta_y = point.y() - segment_start.y()
    segment_ratio = (
        point_delta_x * segment_delta_x
        + point_delta_y * segment_delta_y
    ) / segment_length_squared
    if segment_ratio < 0.0 or segment_ratio > 1.0:
        return None

    projected_point = QPointF(
        segment_start.x() + segment_delta_x * segment_ratio,
        segment_start.y() + segment_delta_y * segment_ratio,
    )
    return projected_point, segment_ratio


def _vertex_pairs_overlap(
    first_pair: VertexPairCenter,
    second_pair: VertexPairCenter,
) -> bool:
    first_source_ids = first_pair.source_vertex_ids
    second_source_ids = second_pair.source_vertex_ids
    return (
        first_source_ids[0] in second_source_ids
        or first_source_ids[1] in second_source_ids
    )


def _calculate_corner_angle(
    previous_vertex: Vertex,
    current_vertex: Vertex,
    next_vertex: Vertex,
) -> float:
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
    if previous_length <= 1e-6 or next_length <= 1e-6:
        return 0.0

    dot_product = (
        previous_vector[0] * next_vector[0]
        + previous_vector[1] * next_vector[1]
    )
    cosine = dot_product / (previous_length * next_length)
    clamped_cosine = min(max(cosine, -1.0), 1.0)
    return math.degrees(math.acos(clamped_cosine))


def _order_vertices_around_center(vertices: list[Vertex]) -> list[Vertex]:
    center_x = sum(vertex.x for vertex in vertices) / len(vertices)
    center_y = sum(vertex.y for vertex in vertices) / len(vertices)
    return sorted(
        vertices,
        key=lambda vertex: math.atan2(vertex.y - center_y, vertex.x - center_x),
    )


def _build_random_room_color() -> tuple[int, int, int]:
    return (
        random.randint(70, 230),
        random.randint(70, 230),
        random.randint(70, 230),
    )
