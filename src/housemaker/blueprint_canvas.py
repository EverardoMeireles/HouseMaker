# ### Imports ###
from __future__ import annotations

import copy
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
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

from housemaker.camera_models import CameraPose
from housemaker.level_coordinates import level_world_to_image_xy
from housemaker.models import (
    DEFAULT_DOORWAY_DEPTH_METERS,
    DOORWAY_SHAPE_ARCH,
    DoorwayData,
    DoorwayPreset,
    Edge,
    PIXEL_TO_METER,
    DEFAULT_STAIR_STYLE,
    STAIR_STYLE_FLOATING,
    STAIR_STYLE_FLOATING_WITH_RISER,
    STAIR_STYLE_SUPPORTED,
    RoomData,
    LevelData,
    VERTEX_HIT_RADIUS_SCREEN,
    Vertex,
    VertexData,
    WindowData,
    normalize_doorway_arch_amount,
    normalize_doorway_shape,
    snap_point,
)
from housemaker.surface_geometry import build_wall_surface_id
from housemaker.uv_layout import build_room_walls


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
FLOOR_CONTOUR_FILL_COLOR = QColor(65, 180, 130, 52)
FLOOR_CONTOUR_EDGE_COLOR = QColor("#41d69a")
PENDING_FLOOR_CONTOUR_COLOR = QColor("#ffd166")
DOORWAY_FILL_COLOR = QColor(97, 196, 255, 115)
DOORWAY_EDGE_COLOR = QColor("#32b8ff")
SELECTED_DOORWAY_EDGE_COLOR = QColor("#f6c85f")
PENDING_DOORWAY_FILL_COLOR = QColor(255, 209, 102, 115)
PENDING_DOORWAY_EDGE_COLOR = QColor("#ffd166")
WINDOW_FILL_COLOR = QColor(74, 214, 255, 105)
WINDOW_EDGE_COLOR = QColor("#46d6ff")
STAIR_SUPPORTED_COLOR = QColor("#65d6ff")
STAIR_FLOATING_COLOR = QColor("#c99cff")
STAIR_FLOATING_WITH_RISER_COLOR = QColor("#ff9f6e")
STAIR_SELECTED_COLOR = QColor("#f6c85f")
PENDING_STAIR_COLOR = QColor("#ffd166")
STAIR_ENDPOINT_RADIUS_SCREEN = 10.0
STAIR_ENDPOINT_HIT_RADIUS_SCREEN = 16.0
STAIR_DIRECTION_LENGTH_SCREEN = 26.0
STAIR_ROUTE_CENTERLINE_WIDTH_SCREEN = 1.5
STAIR_STYLES = frozenset(
    (
        STAIR_STYLE_SUPPORTED,
        STAIR_STYLE_FLOATING,
        STAIR_STYLE_FLOATING_WITH_RISER,
    )
)
IMAGE_MARGIN = 16.0
VERTEX_RADIUS_SCREEN = 6.0
EDGE_HIT_TOLERANCE_SCREEN = 8.0
AXIS_SNAP_TOLERANCE_SCREEN = 10.0
CENTER_SNAP_TOLERANCE_SCREEN = 10.0
CENTER_SNAP_EQUAL_ANGLE_TOLERANCE_DEGREES = 1.0
DRAG_THRESHOLD_SCREEN = 4.0
WINDOW_STRIP_HALF_WIDTH_SCREEN = 5.0
MIN_ZOOM_SCALE = 1.0
MAX_ZOOM_SCALE = 16.0
ZOOM_STEP_FACTOR = 1.15
CAMERA_INDICATOR_COLOR = QColor("#ffcf4a")
CAMERA_INDICATOR_OUTLINE_COLOR = QColor("#1a1d22")
CAMERA_INDICATOR_RADIUS_SCREEN = 7.0
CAMERA_INDICATOR_DIRECTION_LENGTH_SCREEN = 34.0
CAMERA_INDICATOR_ARROW_HEAD_LENGTH_SCREEN = 10.0
CAMERA_INDICATOR_ARROW_HEAD_HALF_WIDTH_SCREEN = 6.0

# ### Snapshot models ###
@dataclass
class CanvasSnapshot:
    vertex_data: VertexData
    rooms: list[RoomData]
    doorways: list[DoorwayData]
    floor_contour_vertex_ids: tuple[int, ...]
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


@dataclass(frozen=True)
class DoorwayHit:
    doorway_index: int


@dataclass(frozen=True)
class WindowWallFrame:
    start_point: tuple[float, float]
    end_point: tuple[float, float]


@dataclass(frozen=True)
class WallProjection:
    edge: Edge
    point: tuple[float, float]
    distance: float


@dataclass(frozen=True)
class PendingStairPlacement:
    """The completed start segment retained while another level is chosen."""

    style: str
    start_level_index: int
    start_a_x: float
    start_a_y: float
    start_b_x: float
    start_b_y: float
    start_a_vertex_id: int | None = None
    start_b_vertex_id: int | None = None

    @property
    def start_x(self) -> float:
        """Return the legacy midpoint coordinate for compatibility."""

        return (self.start_a_x + self.start_b_x) * 0.5

    @property
    def start_y(self) -> float:
        """Return the legacy midpoint coordinate for compatibility."""

        return (self.start_a_y + self.start_b_y) * 0.5


@dataclass(frozen=True)
class StairSectionPlacement:
    """One ordered two-point stair cross-section owned by a Canvas level."""

    level_index: int
    a_x: float
    a_y: float
    b_x: float
    b_y: float
    a_vertex_id: int | None = None
    b_vertex_id: int | None = None


@dataclass(frozen=True)
class StairPlacement:
    """Canvas-neutral stair data emitted after its route is confirmed."""

    style: str
    start_level_index: int
    start_a_x: float
    start_a_y: float
    start_b_x: float
    start_b_y: float
    end_level_index: int
    end_a_x: float
    end_a_y: float
    end_b_x: float
    end_b_y: float
    intermediate_sections: tuple[StairSectionPlacement, ...] = ()
    start_a_vertex_id: int | None = None
    start_b_vertex_id: int | None = None
    end_a_vertex_id: int | None = None
    end_b_vertex_id: int | None = None

    @property
    def start_x(self) -> float:
        """Return the legacy start-segment midpoint coordinate."""

        return (self.start_a_x + self.start_b_x) * 0.5

    @property
    def start_y(self) -> float:
        """Return the legacy start-segment midpoint coordinate."""

        return (self.start_a_y + self.start_b_y) * 0.5

    @property
    def end_x(self) -> float:
        """Return the legacy end-segment midpoint coordinate."""

        return (self.end_a_x + self.end_b_x) * 0.5

    @property
    def end_y(self) -> float:
        """Return the legacy end-segment midpoint coordinate."""

        return (self.end_a_y + self.end_b_y) * 0.5


@dataclass(frozen=True)
class StairPointTarget:
    """A free or snapped point used to define one end of a stair segment."""

    point: tuple[float, float]
    guides: list[SnapGuide]
    vertex_id: int | None = None


@dataclass(frozen=True)
class PendingStairPoint:
    """The first point of the segment currently being drawn."""

    level_index: int
    x: float
    y: float
    vertex_id: int | None = None


@dataclass(frozen=True)
class StairHit:
    stair_index: int
    endpoint_name: str


# ### Widgets ###
class BlueprintCanvas(QWidget):
    rooms_changed = Signal()
    doorways_changed = Signal()
    doorway_dimension_preview_changed = Signal()
    doorway_move_drag_started = Signal()
    doorway_move_drag_finished = Signal(bool)
    selected_doorway_changed = Signal(int)
    floor_contour_changed = Signal(object)
    stair_start_placed = Signal(object)
    stair_placement_ready = Signal(object)
    stair_placement_completed = Signal(object)
    stair_placement_cancelled = Signal()
    stair_placement_invalid_endpoint = Signal(str)
    stair_delete_requested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.vertex_data = VertexData()
        self.rooms: list[RoomData] = []
        self.doorways: list[DoorwayData] = []
        self.windows: list[WindowData] = []
        self.floor_contour_vertex_ids: tuple[int, ...] = ()
        self.blueprint_image: QImage | None = None
        self.blueprint_path: str | None = None
        self._blueprint_image_revision: tuple[object, ...] | None = None
        self.active_vertex_id: int | None = None
        self.selected_vertex_id: int | None = None
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
        self.pending_floor_contour_vertex_ids: list[int] | None = None
        self.pending_floor_contour_preview_point: tuple[float, float] | None = None
        self.pending_doorway_preset: DoorwayPreset | None = None
        self.pending_doorway: DoorwayData | None = None
        self.selected_doorway_index: int | None = None
        self.pressed_doorway_index: int | None = None
        self.drag_doorway_index: int | None = None
        self.doorway_drag_press_position: QPointF | None = None
        self.doorway_drag_press_image_point: tuple[float, float] | None = None
        self.doorway_drag_initial_doorway: DoorwayData | None = None
        self.doorway_drag_wall_edge: Edge | None = None
        self.doorway_drag_changed = False
        self.stairs: list[object] = []
        self.selected_stair_index: int | None = None
        self.pending_stair_style: str | None = None
        self.pending_stair_placement: PendingStairPlacement | None = None
        self.pending_stair_draft: StairPlacement | None = None
        self.pending_stair_point: PendingStairPoint | None = None
        self.pending_stair_preview_point: tuple[float, float] | None = None
        self.pending_stair_preview_guides: list[SnapGuide] = []
        self.level_context: LevelData | None = None
        self._camera_indicator_pose: CameraPose | None = None

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
        floor_contour_vertex_ids: tuple[int, ...] = (),
        doorways: list[DoorwayData] | None = None,
        windows: list[WindowData] | None = None,
    ) -> None:
        revision_before = _build_blueprint_image_revision(file_path)
        image = _load_qimage_from_path(file_path)
        revision_after = _build_blueprint_image_revision(file_path)
        self._set_level_contents(
            vertex_data=vertex_data or VertexData(),
            rooms=rooms,
            blueprint_image=image,
            blueprint_path=file_path,
            floor_contour_vertex_ids=floor_contour_vertex_ids,
            doorways=doorways,
            windows=windows,
            blueprint_revision=(
                revision_after
                if revision_before == revision_after
                else None
            ),
        )

    def set_level_vertex_data(
        self,
        vertex_data: VertexData,
        floor_contour_vertex_ids: tuple[int, ...] | None = None,
        doorways: list[DoorwayData] | None = None,
    ) -> None:
        self.set_level_data(
            vertex_data=vertex_data,
            rooms=self.rooms,
            image_path=self.blueprint_path,
            floor_contour_vertex_ids=(
                self.floor_contour_vertex_ids
                if floor_contour_vertex_ids is None
                else floor_contour_vertex_ids
            ),
            doorways=self.doorways if doorways is None else doorways,
            windows=self.windows,
        )

    def set_level_data(
        self,
        vertex_data: VertexData,
        rooms: list[RoomData] | None,
        image_path: str | None,
        floor_contour_vertex_ids: tuple[int, ...] = (),
        doorways: list[DoorwayData] | None = None,
        windows: list[WindowData] | None = None,
    ) -> None:
        blueprint_image: QImage | None = None
        blueprint_revision = (
            None
            if image_path is None
            else _build_blueprint_image_revision(image_path)
        )
        if (
            image_path
            and blueprint_revision is not None
            and _blueprint_revision_has_file(blueprint_revision)
        ):
            try:
                blueprint_image = _load_qimage_from_path(image_path)
            except ValueError:
                blueprint_image = None
                blueprint_revision = None
            else:
                revision_after = _build_blueprint_image_revision(image_path)
                if blueprint_revision != revision_after:
                    blueprint_revision = None

        self._set_level_contents(
            vertex_data=vertex_data,
            rooms=rooms,
            blueprint_image=blueprint_image,
            blueprint_path=image_path,
            floor_contour_vertex_ids=floor_contour_vertex_ids,
            doorways=doorways,
            windows=windows,
            blueprint_revision=blueprint_revision,
        )

    def get_image_size_pixels(self) -> tuple[float, float] | None:
        if self.blueprint_image is None:
            return None

        return (
            float(self.blueprint_image.width()),
            float(self.blueprint_image.height()),
        )

    def get_blueprint_image_revision(self) -> tuple[object, ...] | None:
        """Return the file revision validated for the displayed pixels."""

        return self._blueprint_image_revision

    # ### First-person camera indicator ###
    def set_camera_indicator_pose(self, pose: CameraPose | None) -> None:
        """Display one world-space camera pose over the current blueprint."""

        if pose is not None and not isinstance(pose, CameraPose):
            raise TypeError("The Canvas camera indicator requires a CameraPose.")
        if pose == self._camera_indicator_pose:
            return
        self._camera_indicator_pose = pose
        self.update()

    def get_camera_indicator_pose(self) -> CameraPose | None:
        """Return the world-space pose currently represented on the Canvas."""

        return self._camera_indicator_pose

    def refresh_blueprint_image_if_stale(self) -> bool:
        """Reload changed pixels without resetting Canvas editing state."""

        if self.blueprint_path is None:
            return False
        next_revision = _build_blueprint_image_revision(self.blueprint_path)
        if next_revision == self._blueprint_image_revision:
            return False
        if not _blueprint_revision_has_file(next_revision):
            self.blueprint_image = None
            self._blueprint_image_revision = next_revision
            self.update()
            return True
        try:
            blueprint_image = _load_qimage_from_path(self.blueprint_path)
        except (OSError, ValueError):
            return False
        revision_after = _build_blueprint_image_revision(self.blueprint_path)
        if next_revision != revision_after:
            return False
        self.blueprint_image = blueprint_image
        self._blueprint_image_revision = revision_after
        self.update()
        return True

    # ### Stair context and placement ###
    def set_stair_context(
        self,
        stairs: Iterable[object],
        level: LevelData | None = None,
    ) -> None:
        """Display project-owned stairs without taking ownership of their model."""

        self.stairs = list(stairs)
        if level is not None:
            self.level_context = level
            self.windows = level.windows
        if (
            self.selected_stair_index is not None
            and self.selected_stair_index >= len(self.stairs)
        ):
            self.selected_stair_index = None
        if self._is_stair_placement_active():
            self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()

    def start_stair_placement(
        self,
        style: str = DEFAULT_STAIR_STYLE,
    ) -> None:
        """Start endpoint placement followed by optional curve refinement."""

        if self._is_stair_placement_active():
            # The Add/Confirm button and level changes can both revisit this
            # entry point while the user is adding curve controls.  Preserve
            # the one owned draft instead of silently starting a second stair
            # placement and losing its ordered route.
            self.setCursor(Qt.CursorShape.CrossCursor)
            self.update()
            return
        self._reset_floor_contour_designation()
        self._reset_doorway_placement()
        self._set_selected_doorway_index(None)
        self._reset_doorway_pointer_state()
        self.active_vertex_id = None
        self.selected_vertex_id = None
        self.selected_stair_index = None
        self.preview_point = None
        self.preview_guides = []
        self._reset_pointer_state()
        self.pending_stair_style = _normalize_stair_style(style)
        self.pending_stair_placement = None
        self.pending_stair_draft = None
        self.pending_stair_point = None
        self.pending_stair_preview_point = None
        self.pending_stair_preview_guides = []
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()

    def get_pending_stair_placement(self) -> PendingStairPlacement | None:
        return self.pending_stair_placement

    def get_stair_placement_draft(self) -> StairPlacement | None:
        """Return the complete, unconfirmed stair route being refined."""

        return self.pending_stair_draft

    def get_pending_stair_point(self) -> PendingStairPoint | None:
        """Return point A of the stair segment that is currently being drawn."""

        return self.pending_stair_point

    def is_stair_placement_active(self) -> bool:
        return self._is_stair_placement_active()

    def is_stair_ready_for_confirmation(self) -> bool:
        """Return whether the draft has complete endpoints and control pairs."""

        return (
            self.pending_stair_draft is not None
            and self.pending_stair_point is None
        )

    def confirm_stair_placement(self) -> bool:
        """Commit the refined draft and finish Canvas placement mode."""

        draft = self.pending_stair_draft
        if draft is None:
            return False
        if self.pending_stair_point is not None:
            self.stair_placement_invalid_endpoint.emit(
                "Place the second point of the current stair section before "
                "confirming."
            )
            return False

        self._reset_stair_placement()
        self.unsetCursor()
        self.stair_placement_completed.emit(draft)
        self.update()
        return True

    def remove_last_stair_intermediate_section(self) -> bool:
        """Remove an unfinished point or the newest refinement cross-section."""

        draft = self.pending_stair_draft
        if draft is None:
            return False
        if self.pending_stair_point is not None:
            self.pending_stair_point = None
            self.pending_stair_preview_point = None
            self.pending_stair_preview_guides = []
            self.stair_placement_ready.emit(draft)
            self.update()
            return True
        if not draft.intermediate_sections:
            return False

        updated_draft = replace(
            draft,
            intermediate_sections=draft.intermediate_sections[:-1],
        )
        self.pending_stair_draft = updated_draft
        self.stair_placement_ready.emit(updated_draft)
        self.update()
        return True

    def set_pending_stair_placement(
        self,
        placement: PendingStairPlacement | object | None,
    ) -> bool:
        """Restore a pending first endpoint after an owner-level view changes."""

        pending_placement = _coerce_pending_stair_placement(placement)
        if placement is not None and pending_placement is None:
            return False

        self.pending_stair_placement = pending_placement
        self.pending_stair_draft = None
        self.pending_stair_style = (
            None if pending_placement is None else pending_placement.style
        )
        self.pending_stair_point = None
        self.pending_stair_preview_point = None
        self.pending_stair_preview_guides = []
        if self._is_stair_placement_active():
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.unsetCursor()
        self.update()
        return True

    def cancel_stair_placement(self) -> bool:
        """Cancel either phase of placement and leave existing project stairs intact."""

        if not self._is_stair_placement_active():
            return False

        self._reset_stair_placement()
        self.unsetCursor()
        self.stair_placement_cancelled.emit()
        self.update()
        return True

    def set_snap_middle_equal_angle_only(self, enabled: bool) -> None:
        self.snap_middle_equal_angle_only = bool(enabled)
        self.preview_point = None
        self.preview_guides = []
        self.update()

    def start_floor_contour_designation(self) -> None:
        self._cancel_stair_placement_for_other_mode()
        self._reset_doorway_placement()
        self._set_selected_doorway_index(None)
        self._reset_doorway_pointer_state()
        self.unsetCursor()
        self.active_vertex_id = None
        self.preview_point = None
        self.preview_guides = []
        self.selected_vertex_id = None
        self.pending_floor_contour_vertex_ids = []
        self.pending_floor_contour_preview_point = None
        self._reset_pointer_state()
        self.update()

    def start_doorway_placement(self, preset: DoorwayPreset) -> None:
        """Begin placing one doorway using the selected hole dimensions."""
        self._cancel_stair_placement_for_other_mode()
        self._reset_floor_contour_designation()
        self.active_vertex_id = None
        self.selected_vertex_id = None
        self._set_selected_doorway_index(None)
        self.preview_point = None
        self.preview_guides = []
        self._reset_pointer_state()
        self._reset_doorway_pointer_state()
        self.pending_doorway_preset = preset
        self.pending_doorway = None
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()

    def clear_floor_contour(self) -> None:
        self._reset_floor_contour_designation()
        if not self.floor_contour_vertex_ids:
            self.update()
            return

        self._push_undo_state()
        self.floor_contour_vertex_ids = ()
        self.floor_contour_changed.emit(())
        self.update()

    def _set_level_contents(
        self,
        vertex_data: VertexData,
        rooms: list[RoomData] | None,
        blueprint_image: QImage | None,
        blueprint_path: str | None,
        floor_contour_vertex_ids: tuple[int, ...],
        doorways: list[DoorwayData] | None,
        windows: list[WindowData] | None,
        blueprint_revision: tuple[object, ...] | None,
    ) -> None:
        self.blueprint_image = blueprint_image
        self.blueprint_path = blueprint_path
        self._blueprint_image_revision = blueprint_revision
        self.vertex_data = vertex_data
        self.rooms = rooms if rooms is not None else []
        self.doorways = doorways if doorways is not None else []
        self.windows = windows if windows is not None else []
        self.floor_contour_vertex_ids = self._normalize_floor_contour_vertex_ids(
            floor_contour_vertex_ids
        )
        self.active_vertex_id = None
        self.selected_vertex_id = None
        self.preview_point = None
        self.preview_guides = []
        self.undo_stack.clear()
        self._reset_floor_contour_designation()
        self._reset_doorway_placement()
        self.pending_stair_preview_point = None
        self.pending_stair_preview_guides = []
        self.level_context = None
        self._set_selected_doorway_index(None)
        self.selected_stair_index = None
        self._reset_pointer_state()
        self._reset_doorway_pointer_state()
        self._reset_view()
        if self._is_stair_placement_active():
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.unsetCursor()
        self.update()

    def undo_last_step(self) -> None:
        if not self.undo_stack:
            return

        snapshot = self.undo_stack.pop()
        previous_vertex_data = self.vertex_data.clone()
        previous_rooms = copy.deepcopy(self.rooms)
        previous_floor_contour_vertex_ids = self.floor_contour_vertex_ids
        previous_doorways = copy.deepcopy(self.doorways)
        self.vertex_data.copy_from(snapshot.vertex_data)
        self.rooms.clear()
        self.rooms.extend(copy.deepcopy(snapshot.rooms))
        self.doorways.clear()
        self.doorways.extend(copy.deepcopy(snapshot.doorways))
        self.floor_contour_vertex_ids = snapshot.floor_contour_vertex_ids
        self.active_vertex_id = snapshot.active_vertex_id
        self.selected_vertex_id = snapshot.selected_vertex_id
        self.preview_point = snapshot.preview_point
        self.preview_guides = []
        self._reset_floor_contour_designation()
        self._reset_doorway_placement()
        self._set_selected_doorway_index(None)
        self._reset_pointer_state()
        self._reset_doorway_pointer_state()
        self.update()
        if (
            self.vertex_data != previous_vertex_data
            or self.rooms != previous_rooms
        ):
            self.rooms_changed.emit()
        if self.doorways != previous_doorways:
            self.doorways_changed.emit()
        if self.floor_contour_vertex_ids != previous_floor_contour_vertex_ids:
            self.floor_contour_changed.emit(self.floor_contour_vertex_ids)

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if (
            event.key() == Qt.Key.Key_Backspace
            and self.remove_last_stair_intermediate_section()
        ):
            event.accept()
            return

        if (
            event.key() == Qt.Key.Key_Escape
            and self._is_stair_placement_active()
        ):
            self.cancel_stair_placement()
            event.accept()
            return

        if (
            event.key() == Qt.Key.Key_Delete
            and self._request_selected_stair_deletion()
        ):
            event.accept()
            return

        if (
            event.key() == Qt.Key.Key_Escape
            and self.pending_doorway_preset is not None
        ):
            self._reset_doorway_placement()
            self.unsetCursor()
            self.update()
            event.accept()
            return

        if (
            event.key() == Qt.Key.Key_Escape
            and self.pending_floor_contour_vertex_ids is not None
        ):
            self._reset_floor_contour_designation()
            self.update()
            event.accept()
            return

        if (
            event.key() == Qt.Key.Key_Delete
            and self._delete_selected_doorway()
        ):
            event.accept()
            return

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
            if self._is_stair_placement_active():
                self.cancel_stair_placement()
                event.accept()
                return

            if self.pending_doorway_preset is not None:
                self._reset_doorway_placement()
                self.unsetCursor()
                self.update()
                event.accept()
                return

            if self.pending_floor_contour_vertex_ids is not None:
                self._reset_floor_contour_designation()
                self.update()
                event.accept()
                return

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

        if self._is_stair_placement_active():
            image_point = self._widget_to_image(event.position())
            if image_point is not None:
                self._place_stair_endpoint(
                    image_point,
                    event.modifiers(),
                    event.position(),
                )
            event.accept()
            return

        if self.pending_doorway_preset is not None:
            image_point = self._widget_to_image(event.position())
            if image_point is not None:
                self._update_pending_doorway(image_point)
                self._commit_pending_doorway()
            event.accept()
            return

        if event.modifiers() & Qt.KeyboardModifier.AltModifier:
            stair_hit = self._find_stair_hit(event.position())
            if stair_hit is not None:
                self.selected_stair_index = stair_hit.stair_index
                self._set_selected_doorway_index(None)
                self.selected_vertex_id = None
                self.update()
                event.accept()
                return

        self.selected_stair_index = None
        if self._find_window_at(event.position()) is not None:
            self._set_selected_doorway_index(None)
            self.selected_vertex_id = None
            self.update()
            event.accept()
            return

        doorway_hit = self._find_doorway_hit(event.position())
        if doorway_hit is not None:
            self._set_selected_doorway_index(doorway_hit.doorway_index)
            self.selected_vertex_id = None
            self._reset_doorway_pointer_state()
            doorway = self.doorways[doorway_hit.doorway_index]
            image_point = self._widget_to_image(event.position())
            self.pressed_doorway_index = doorway_hit.doorway_index
            self.doorway_drag_press_position = QPointF(event.position())
            self.doorway_drag_press_image_point = (
                None
                if image_point is None
                else (image_point.x(), image_point.y())
            )
            self.doorway_drag_initial_doorway = copy.deepcopy(doorway)
            nearest_wall = self._find_nearest_wall_projection(
                (doorway.center_x, doorway.center_y)
            )
            self.doorway_drag_wall_edge = (
                None if nearest_wall is None else nearest_wall.edge
            )
            self.doorway_move_drag_started.emit()
            self.update()
            event.accept()
            return

        self._set_selected_doorway_index(None)
        hit_vertex = self._find_vertex_at(event.position())
        if self.pending_floor_contour_vertex_ids is not None:
            if hit_vertex is not None:
                self._handle_floor_contour_vertex_click(hit_vertex.id)
            event.accept()
            return

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

        if self._is_stair_placement_active():
            image_point = self._widget_to_image(event.position())
            if image_point is None:
                self.pending_stair_preview_point = None
                self.pending_stair_preview_guides = []
            else:
                preview = self._build_stair_endpoint_preview(
                    image_point,
                    event.modifiers(),
                    event.position(),
                )
                self.pending_stair_preview_point = (
                    None if preview is None else preview.point
                )
                self.pending_stair_preview_guides = (
                    [] if preview is None else preview.guides
                )
            self.update()
            event.accept()
            return

        if self.pending_doorway_preset is not None:
            image_point = self._widget_to_image(event.position())
            if image_point is None:
                self.pending_doorway = None
            else:
                self._update_pending_doorway(image_point)
            self.update()
            event.accept()
            return

        if (
            self.pressed_doorway_index is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            if self._should_start_doorway_drag(event.position()):
                self._start_doorway_drag()

            if self.drag_doorway_index is not None:
                image_point = self._widget_to_image_clamped(event.position())
                self._move_dragged_doorway(image_point)
                self.update()
                event.accept()
                return

            event.accept()
            return

        if self.pending_floor_contour_vertex_ids is not None:
            image_point = self._widget_to_image(event.position())
            self.pending_floor_contour_preview_point = (
                None
                if image_point is None
                else (image_point.x(), image_point.y())
            )
            self.update()
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

        self._update_edit_hover_cursor(event.position())
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

        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.pressed_doorway_index is not None
        ):
            self._reset_doorway_pointer_state()
            self._update_edit_hover_cursor(event.position())
            self.update()
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
        if self._is_stair_placement_active():
            self.pending_stair_preview_point = None
            self.pending_stair_preview_guides = []
            self.update()
        if self.pending_doorway_preset is not None:
            self.pending_doorway = None
            self.update()
        if self.pending_floor_contour_vertex_ids is not None:
            self.pending_floor_contour_preview_point = None
            self.update()
        if self.drag_vertex_id is None and self.active_vertex_id is not None:
            self.preview_point = None
            self.preview_guides = []
            self.update()
        if (
            self.pending_doorway_preset is None
            and not self._is_stair_placement_active()
        ):
            self.unsetCursor()
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

        self._paint_floor_contour(painter)
        self._paint_edges(painter)
        self._paint_windows(painter)
        self._paint_doorways(painter)
        self._paint_pending_doorway(painter)
        self._paint_pending_floor_contour(painter)
        self._paint_preview_guides(painter)
        self._paint_preview_edge(painter)
        self._paint_vertices(painter)
        self._paint_stairs(painter)
        self._paint_pending_stair_placement(painter)
        self._paint_camera_indicator(painter)
        self._paint_overlay_text(painter)

    # ### First-person camera indicator painting ###
    def _paint_camera_indicator(self, painter: QPainter) -> None:
        """Paint a fixed-size position marker and top-down look-direction arrow."""

        geometry = self._get_camera_indicator_geometry()
        if geometry is None:
            return
        center, direction_tip = geometry
        direction = direction_tip - center
        direction_length = math.hypot(direction.x(), direction.y())
        if direction_length <= 1e-6:
            return

        unit_direction = direction / direction_length
        perpendicular = QPointF(-unit_direction.y(), unit_direction.x())
        shaft_start = center + unit_direction * CAMERA_INDICATOR_RADIUS_SCREEN
        arrow_base = (
            direction_tip
            - unit_direction * CAMERA_INDICATOR_ARROW_HEAD_LENGTH_SCREEN
        )
        arrow_head = QPolygonF(
            (
                direction_tip,
                arrow_base
                + perpendicular * CAMERA_INDICATOR_ARROW_HEAD_HALF_WIDTH_SCREEN,
                arrow_base
                - perpendicular * CAMERA_INDICATOR_ARROW_HEAD_HALF_WIDTH_SCREEN,
            )
        )

        painter.save()
        outline_pen = QPen(CAMERA_INDICATOR_OUTLINE_COLOR, 4.5)
        outline_pen.setCosmetic(True)
        outline_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(outline_pen)
        painter.setBrush(CAMERA_INDICATOR_OUTLINE_COLOR)
        painter.drawLine(shaft_start, arrow_base)
        painter.drawPolygon(arrow_head)
        painter.drawEllipse(
            center,
            CAMERA_INDICATOR_RADIUS_SCREEN + 1.5,
            CAMERA_INDICATOR_RADIUS_SCREEN + 1.5,
        )

        indicator_pen = QPen(CAMERA_INDICATOR_COLOR, 2.5)
        indicator_pen.setCosmetic(True)
        indicator_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(indicator_pen)
        painter.setBrush(CAMERA_INDICATOR_COLOR)
        painter.drawLine(shaft_start, arrow_base)
        painter.drawPolygon(arrow_head)
        painter.drawEllipse(
            center,
            CAMERA_INDICATOR_RADIUS_SCREEN,
            CAMERA_INDICATOR_RADIUS_SCREEN,
        )
        painter.restore()

    def _get_camera_indicator_geometry(
        self,
    ) -> tuple[QPointF, QPointF] | None:
        """Return the marker center and fixed-length arrow tip in widget space."""

        pose = self._camera_indicator_pose
        level = self.level_context
        if pose is None or level is None or self.blueprint_image is None:
            return None

        yaw_radians = math.radians(pose.yaw_degrees)
        try:
            center_image = level_world_to_image_xy(level, pose.x, pose.y)
            direction_image = level_world_to_image_xy(
                level,
                pose.x + math.cos(yaw_radians),
                pose.y + math.sin(yaw_radians),
            )
        except (TypeError, ValueError, OverflowError):
            return None
        center = self._image_to_widget(*center_image)
        direction_point = self._image_to_widget(*direction_image)
        direction = direction_point - center
        direction_length = math.hypot(direction.x(), direction.y())
        if direction_length <= 1e-6:
            return None
        direction *= CAMERA_INDICATOR_DIRECTION_LENGTH_SCREEN / direction_length
        return center, center + direction

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
        self.preview_point = point
        self.preview_guides = []

    # ### Stair helpers ###
    def _is_stair_placement_active(self) -> bool:
        return self.pending_stair_style is not None

    def _cancel_stair_placement_for_other_mode(self) -> None:
        if not self._is_stair_placement_active():
            return

        self._reset_stair_placement()
        self.stair_placement_cancelled.emit()

    def _reset_stair_placement(self) -> None:
        self.pending_stair_style = None
        self.pending_stair_placement = None
        self.pending_stair_draft = None
        self.pending_stair_point = None
        self.pending_stair_preview_point = None
        self.pending_stair_preview_guides = []

    def _build_stair_endpoint_preview(
        self,
        image_point: QPointF,
        modifiers: Qt.KeyboardModifier,
        widget_point: QPointF | None = None,
    ) -> SnapPreview | None:
        """Preview a free or snapped stair point without changing wall data."""

        level = self.level_context
        pending_point = self.pending_stair_point
        pending_placement = self.pending_stair_placement
        pending_draft = self.pending_stair_draft
        if level is None:
            return None
        if (
            pending_point is not None
            and pending_point.level_index != level.index
        ):
            return None
        if (
            pending_placement is not None
            and pending_draft is None
            and pending_point is None
            and pending_placement.start_level_index == level.index
        ):
            return None

        target_widget_point = widget_point
        if target_widget_point is None:
            target_widget_point = self._image_to_widget(
                image_point.x(),
                image_point.y(),
            )
        target = self._build_stair_point_target(
            image_point,
            modifiers,
            target_widget_point,
        )
        return SnapPreview(point=target.point, guides=target.guides)

    def _place_stair_endpoint(
        self,
        image_point: QPointF,
        modifiers: Qt.KeyboardModifier,
        widget_point: QPointF | None = None,
    ) -> None:
        level = self.level_context
        style = self.pending_stair_style
        if level is None or style is None:
            return

        pending_placement = self.pending_stair_placement
        pending_draft = self.pending_stair_draft
        pending_point = self.pending_stair_point
        if pending_point is not None and pending_point.level_index != level.index:
            self.stair_placement_invalid_endpoint.emit(
                f"Return to level {pending_point.level_index} and place the "
                "second point of this stair segment."
            )
            return
        if (
            pending_placement is not None
            and pending_draft is None
            and pending_point is None
            and level.index == pending_placement.start_level_index
        ):
            self.stair_placement_invalid_endpoint.emit(
                "Choose a different level for the stair end segment."
            )
            return

        target_widget_point = widget_point
        if target_widget_point is None:
            target_widget_point = self._image_to_widget(
                image_point.x(),
                image_point.y(),
            )
        target = self._build_stair_point_target(
            image_point,
            modifiers,
            target_widget_point,
        )
        point_x, point_y = target.point
        if pending_point is not None and _points_are_coincident(
            (pending_point.x, pending_point.y),
            target.point,
        ):
            self.stair_placement_invalid_endpoint.emit(
                "The two points of a stair segment must be different."
            )
            return

        if target.vertex_id is None:
            self.selected_vertex_id = None
        else:
            self.selected_vertex_id = target.vertex_id

        if pending_point is None:
            self.pending_stair_point = PendingStairPoint(
                level_index=level.index,
                x=point_x,
                y=point_y,
                vertex_id=target.vertex_id,
            )
            self.pending_stair_preview_point = None
            self.pending_stair_preview_guides = []
            self.update()
            return

        if pending_placement is None:
            pending_placement = PendingStairPlacement(
                style=style,
                start_level_index=level.index,
                start_a_x=pending_point.x,
                start_a_y=pending_point.y,
                start_b_x=point_x,
                start_b_y=point_y,
                start_a_vertex_id=pending_point.vertex_id,
                start_b_vertex_id=target.vertex_id,
            )
            self.pending_stair_placement = pending_placement
            self.pending_stair_point = None
            self.pending_stair_preview_point = None
            self.pending_stair_preview_guides = []
            self.stair_start_placed.emit(pending_placement)
            self.update()
            return

        if pending_draft is None:
            pending_draft = StairPlacement(
                style=style,
                start_level_index=pending_placement.start_level_index,
                start_a_x=pending_placement.start_a_x,
                start_a_y=pending_placement.start_a_y,
                start_b_x=pending_placement.start_b_x,
                start_b_y=pending_placement.start_b_y,
                end_level_index=level.index,
                end_a_x=pending_point.x,
                end_a_y=pending_point.y,
                end_b_x=point_x,
                end_b_y=point_y,
                start_a_vertex_id=pending_placement.start_a_vertex_id,
                start_b_vertex_id=pending_placement.start_b_vertex_id,
                end_a_vertex_id=pending_point.vertex_id,
                end_b_vertex_id=target.vertex_id,
            )
        else:
            intermediate_section = StairSectionPlacement(
                level_index=level.index,
                a_x=pending_point.x,
                a_y=pending_point.y,
                b_x=point_x,
                b_y=point_y,
                a_vertex_id=pending_point.vertex_id,
                b_vertex_id=target.vertex_id,
            )
            pending_draft = replace(
                pending_draft,
                intermediate_sections=(
                    *pending_draft.intermediate_sections,
                    intermediate_section,
                ),
            )

        self.pending_stair_draft = pending_draft
        self.pending_stair_point = None
        self.pending_stair_preview_point = None
        self.pending_stair_preview_guides = []
        self.stair_placement_ready.emit(pending_draft)
        self.update()

    def _build_stair_point_target(
        self,
        image_point: QPointF,
        modifiers: Qt.KeyboardModifier,
        widget_point: QPointF,
    ) -> StairPointTarget:
        """Resolve vertex, wall-edge, or free placement snapping for a point."""

        hit_vertex = self._find_vertex_at(widget_point)
        if hit_vertex is not None:
            return StairPointTarget(
                point=(hit_vertex.x, hit_vertex.y),
                guides=[],
                vertex_id=hit_vertex.id,
            )

        hit_edge = self._find_edge_at(widget_point)
        if hit_edge is not None:
            return StairPointTarget(point=hit_edge.point, guides=[])

        base_point = self.pending_stair_point
        base_vertex = None
        level = self.level_context
        if (
            base_point is not None
            and level is not None
            and base_point.level_index == level.index
        ):
            base_vertex = Vertex(
                id=-1,
                x=base_point.x,
                y=base_point.y,
            )
        preview = self._build_connection_preview_from_base(
            image_point=image_point,
            modifiers=modifiers,
            base_vertex=base_vertex,
        )
        return StairPointTarget(
            point=preview.point,
            guides=preview.guides,
        )

    def _find_stair_hit(self, widget_point: QPointF) -> StairHit | None:
        level = self.level_context
        if level is None:
            return None

        for stair_index in range(len(self.stairs) - 1, -1, -1):
            placement = _coerce_stair_placement(self.stairs[stair_index])
            if placement is None:
                continue

            for section_name, section in _get_stair_sections(placement):
                if section.level_index != level.index:
                    continue
                for point_name, point_x, point_y, vertex_id in (
                    ("a", section.a_x, section.a_y, section.a_vertex_id),
                    ("b", section.b_x, section.b_y, section.b_vertex_id),
                ):
                    resolved_x, resolved_y = self._resolve_stair_canvas_point(
                        point_x,
                        point_y,
                        vertex_id,
                    )
                    endpoint_point = self._image_to_widget(
                        resolved_x,
                        resolved_y,
                    )
                    if _qpoint_distance(widget_point, endpoint_point) <= (
                        STAIR_ENDPOINT_HIT_RADIUS_SCREEN
                    ):
                        return StairHit(
                            stair_index=stair_index,
                            endpoint_name=f"{section_name}_{point_name}",
                        )

        return None

    def _resolve_stair_canvas_point(
        self,
        saved_x: float,
        saved_y: float,
        vertex_id: int | None,
    ) -> tuple[float, float]:
        """Use a live bound vertex, falling back to the stair's saved point."""

        if vertex_id is not None:
            bound_vertex = self.vertex_data.get_vertex(vertex_id)
            if bound_vertex is not None:
                return bound_vertex.x, bound_vertex.y
        return saved_x, saved_y

    def _request_selected_stair_deletion(self) -> bool:
        stair_index = self.selected_stair_index
        if stair_index is None or not (0 <= stair_index < len(self.stairs)):
            return False

        self.selected_stair_index = None
        self.stair_delete_requested.emit(stair_index)
        self.update()
        return True

    # ### Window helpers ###
    def _find_window_at(self, widget_point: QPointF) -> int | None:
        """Return the topmost visible window strip under the pointer."""

        point = (widget_point.x(), widget_point.y())
        wall_frames = self._build_window_wall_frames()
        for window_index in range(len(self.windows) - 1, -1, -1):
            window = self.windows[window_index]
            wall_frame = wall_frames.get(window.wall_surface_id)
            segment = self._get_window_widget_segment(window, wall_frame)
            if segment is None:
                continue
            start, end = segment
            start_tuple = (start.x(), start.y())
            end_tuple = (end.x(), end.y())
            projected_point = _project_point_onto_segment(
                point,
                start_tuple,
                end_tuple,
            )
            if (
                projected_point is not None
                and self._point_distance(point, projected_point)
                <= WINDOW_STRIP_HALF_WIDTH_SCREEN + 4.0
            ):
                return window_index
        return None

    def _update_edit_hover_cursor(self, widget_point: QPointF) -> None:
        """Show doorway movement feedback without making windows editable."""

        if self._find_window_at(widget_point) is not None:
            self.unsetCursor()
            return
        self._update_doorway_hover_cursor(widget_point)

    def _get_window_widget_segment(
        self,
        window: WindowData,
        wall_frame: WindowWallFrame | None = None,
    ) -> tuple[QPointF, QPointF] | None:
        frame = wall_frame or self._get_window_wall_frame(window)
        if frame is None:
            return None
        start_x, start_y = frame.start_point
        delta_x = frame.end_point[0] - start_x
        delta_y = frame.end_point[1] - start_y
        return (
            self._image_to_widget(
                start_x + delta_x * window.start_ratio,
                start_y + delta_y * window.start_ratio,
            ),
            self._image_to_widget(
                start_x + delta_x * window.end_ratio,
                start_y + delta_y * window.end_ratio,
            ),
        )

    def _get_window_wall_frame(
        self,
        window: WindowData,
    ) -> WindowWallFrame | None:
        return self._build_window_wall_frames().get(window.wall_surface_id)

    def _build_window_wall_frames(self) -> dict[str, WindowWallFrame]:
        """Resolve semantic wall IDs without reversing their ratio direction."""

        level = self.level_context
        if level is None:
            return {}
        frames: dict[str, WindowWallFrame] = {}
        room_vertex_sets = [set(room.vertex_ids) for room in level.rooms]
        ignored_vertex_ids = {room.center_vertex_id for room in level.rooms}
        for room in level.rooms:
            for wall in build_room_walls(room, level.vertex_data):
                surface_id = build_wall_surface_id(
                    level.index,
                    wall.key,
                    room.center_vertex_id,
                )
                frames[surface_id] = WindowWallFrame(
                    start_point=wall.start_point,
                    end_point=wall.end_point,
                )

        for edge in level.vertex_data.edges:
            if (
                edge.start_vertex_id in ignored_vertex_ids
                or edge.end_vertex_id in ignored_vertex_ids
                or any(
                    edge.start_vertex_id in vertex_ids
                    and edge.end_vertex_id in vertex_ids
                    for vertex_ids in room_vertex_sets
                )
            ):
                continue
            start_vertex = level.vertex_data.get_vertex(edge.start_vertex_id)
            end_vertex = level.vertex_data.get_vertex(edge.end_vertex_id)
            if start_vertex is None or end_vertex is None:
                continue
            wall_key = (
                f"{min(edge.start_vertex_id, edge.end_vertex_id)}:"
                f"{max(edge.start_vertex_id, edge.end_vertex_id)}"
            )
            surface_id = build_wall_surface_id(level.index, wall_key)
            frames[surface_id] = WindowWallFrame(
                start_point=(start_vertex.x, start_vertex.y),
                end_point=(end_vertex.x, end_vertex.y),
            )
        return frames

    # ### Doorway helpers ###
    def set_selected_doorway_shape(self, shape: str) -> bool:
        """Change the selected opening shape as one undoable preview edit."""

        doorway_index = self.selected_doorway_index
        if doorway_index is None or not (0 <= doorway_index < len(self.doorways)):
            return False

        normalized_shape = normalize_doorway_shape(shape)
        doorway = self.doorways[doorway_index]
        if doorway.shape == normalized_shape:
            return False

        self._push_undo_state()
        self.doorways[doorway_index] = self._copy_doorway_with(
            doorway,
            shape=normalized_shape,
        )
        self.doorway_dimension_preview_changed.emit()
        self.update()
        return True

    def set_selected_doorway_arch_amount(self, arch_amount: float) -> bool:
        """Change the selected arch amount as one undoable preview edit."""

        doorway_index = self.selected_doorway_index
        if doorway_index is None or not (0 <= doorway_index < len(self.doorways)):
            return False

        normalized_arch_amount = normalize_doorway_arch_amount(arch_amount)
        doorway = self.doorways[doorway_index]
        if doorway.shape != DOORWAY_SHAPE_ARCH:
            return False
        if doorway.arch_amount == normalized_arch_amount:
            return False

        self._push_undo_state()
        self.doorways[doorway_index] = self._copy_doorway_with(
            doorway,
            arch_amount=normalized_arch_amount,
        )
        self.doorway_dimension_preview_changed.emit()
        self.update()
        return True

    def _set_selected_doorway_index(self, doorway_index: int | None) -> None:
        """Own doorway selection changes and notify dependent controls once."""

        normalized_index = doorway_index
        if normalized_index is not None and not (
            0 <= normalized_index < len(self.doorways)
        ):
            normalized_index = None
        if normalized_index == self.selected_doorway_index:
            return

        self.selected_doorway_index = normalized_index
        self.selected_doorway_changed.emit(
            -1 if normalized_index is None else normalized_index
        )

    @staticmethod
    def _get_doorway_label_text(doorway: DoorwayData) -> str:
        """Describe one opening without changing its plan-view footprint."""

        opening_name = "Doorway"
        if doorway.shape == DOORWAY_SHAPE_ARCH:
            arch_amount_percent = round(doorway.arch_amount * 100.0, 1)
            opening_name = f"Arch {arch_amount_percent:g}%"
        return (
            f"{opening_name}\n"
            f"{doorway.width_meters:.2f} × {doorway.height_meters:.2f} m"
        )

    def _update_pending_doorway(self, image_point: QPointF) -> None:
        preset = self.pending_doorway_preset
        if preset is None:
            self.pending_doorway = None
            return

        unsnapped_center = self._clamp_image_point(image_point.x(), image_point.y())
        doorway = DoorwayData(
            center_x=unsnapped_center[0],
            center_y=unsnapped_center[1],
            width_meters=float(preset.width_meters),
            height_meters=float(preset.height_meters),
            depth_meters=DEFAULT_DOORWAY_DEPTH_METERS,
            rotation_degrees=0.0,
            shape=preset.shape,
            arch_amount=preset.arch_amount,
        )
        self.pending_doorway = self._snap_doorway_to_walls(
            doorway,
            unsnapped_center,
        )

    def _commit_pending_doorway(self) -> None:
        doorway = self.pending_doorway
        if doorway is None:
            return

        self._push_undo_state()
        self.doorways.append(copy.deepcopy(doorway))
        self._set_selected_doorway_index(len(self.doorways) - 1)
        self._reset_doorway_placement()
        self.unsetCursor()
        self.doorways_changed.emit()
        self.update()

    def _reset_doorway_placement(self) -> None:
        self.pending_doorway_preset = None
        self.pending_doorway = None

    def _reset_doorway_pointer_state(self) -> None:
        drag_was_active = self.pressed_doorway_index is not None
        drag_changed = self.doorway_drag_changed
        if (
            drag_changed
            and self.pressed_doorway_index is not None
            and 0 <= self.pressed_doorway_index < len(self.doorways)
            and self.doorway_drag_initial_doorway is not None
            and self.doorways[self.pressed_doorway_index]
            == self.doorway_drag_initial_doorway
        ):
            drag_changed = False
            if self.undo_stack:
                self.undo_stack.pop()
        self.pressed_doorway_index = None
        self.drag_doorway_index = None
        self.doorway_drag_press_position = None
        self.doorway_drag_press_image_point = None
        self.doorway_drag_initial_doorway = None
        self.doorway_drag_wall_edge = None
        self.doorway_drag_changed = False
        if drag_was_active:
            self.doorway_move_drag_finished.emit(drag_changed)

    def _find_doorway_at(self, widget_point: QPointF) -> int | None:
        doorway_hit = self._find_doorway_hit(widget_point)
        return None if doorway_hit is None else doorway_hit.doorway_index

    def _find_doorway_hit(self, widget_point: QPointF) -> DoorwayHit | None:
        image_point = self._widget_to_image(widget_point)
        if image_point is None:
            return None

        image_hit_tolerance = self._screen_distance_to_image(4.0)
        point = (image_point.x(), image_point.y())
        for doorway_index in range(len(self.doorways) - 1, -1, -1):
            doorway = self.doorways[doorway_index]
            doorway_hit = self._get_doorway_hit_for_point(
                point,
                doorway,
                hit_tolerance_pixels=image_hit_tolerance,
            )
            if doorway_hit is not None:
                return DoorwayHit(
                    doorway_index=doorway_index,
                )

        return None

    def _update_doorway_hover_cursor(self, widget_point: QPointF) -> None:
        doorway_hit = self._find_doorway_hit(widget_point)
        if doorway_hit is None:
            self.unsetCursor()
            return
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def _should_start_doorway_drag(self, widget_point: QPointF) -> bool:
        if (
            self.drag_doorway_index is not None
            or self.doorway_drag_press_position is None
        ):
            return False

        distance = math.hypot(
            widget_point.x() - self.doorway_drag_press_position.x(),
            widget_point.y() - self.doorway_drag_press_position.y(),
        )
        return distance >= DRAG_THRESHOLD_SCREEN

    def _start_doorway_drag(self) -> None:
        doorway_index = self.pressed_doorway_index
        if doorway_index is None or not (0 <= doorway_index < len(self.doorways)):
            return
        if self.doorway_drag_wall_edge is None:
            return

        self.drag_doorway_index = doorway_index
        self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def _move_dragged_doorway(self, image_point: QPointF) -> None:
        """Move one doorway along its original wall while preserving depth."""

        doorway_index = self.drag_doorway_index
        initial_doorway = self.doorway_drag_initial_doorway
        press_image_point = self.doorway_drag_press_image_point
        wall_edge = self.doorway_drag_wall_edge
        if (
            doorway_index is None
            or not (0 <= doorway_index < len(self.doorways))
            or initial_doorway is None
            or press_image_point is None
            or wall_edge is None
        ):
            return

        start_vertex = self.vertex_data.get_vertex(wall_edge.start_vertex_id)
        end_vertex = self.vertex_data.get_vertex(wall_edge.end_vertex_id)
        if start_vertex is None or end_vertex is None:
            return
        wall_delta_x = end_vertex.x - start_vertex.x
        wall_delta_y = end_vertex.y - start_vertex.y
        wall_length_pixels = math.hypot(wall_delta_x, wall_delta_y)
        if wall_length_pixels <= 1e-6:
            return

        tangent_x = wall_delta_x / wall_length_pixels
        tangent_y = wall_delta_y / wall_length_pixels
        initial_wall_position = (
            (initial_doorway.center_x - start_vertex.x) * tangent_x
            + (initial_doorway.center_y - start_vertex.y) * tangent_y
        )
        cursor_delta = (
            (image_point.x() - press_image_point[0]) * tangent_x
            + (image_point.y() - press_image_point[1]) * tangent_y
        )
        half_width_pixels = (
            initial_doorway.width_meters / PIXEL_TO_METER * 0.5
        )
        if half_width_pixels * 2.0 >= wall_length_pixels:
            wall_position = wall_length_pixels * 0.5
        else:
            wall_position = min(
                max(
                    initial_wall_position + cursor_delta,
                    half_width_pixels,
                ),
                wall_length_pixels - half_width_pixels,
            )
        position_delta = wall_position - initial_wall_position
        moved_doorway = self._copy_doorway_with(
            initial_doorway,
            center_x=initial_doorway.center_x + tangent_x * position_delta,
            center_y=initial_doorway.center_y + tangent_y * position_delta,
        )
        current_doorway = self.doorways[doorway_index]
        if moved_doorway == current_doorway:
            return
        if not self.doorway_drag_changed:
            self._push_undo_state()
        self.doorways[doorway_index] = moved_doorway
        self.doorway_drag_changed = True
        self.doorway_dimension_preview_changed.emit()

    def _snap_doorway_to_walls(
        self,
        doorway: DoorwayData,
        raw_center: tuple[float, float],
    ) -> DoorwayData:
        nearest_wall = self._find_nearest_wall_projection(raw_center)
        if nearest_wall is None:
            return self._copy_doorway_with(
                doorway,
                center_x=raw_center[0],
                center_y=raw_center[1],
            )

        aligned_doorway = self._copy_doorway_with(
            doorway,
            rotation_degrees=self._get_wall_normal_rotation_degrees(nearest_wall.edge),
        )
        intersections = self._get_doorway_depth_intersections(
            aligned_doorway,
            raw_center,
        )
        if len(intersections) >= 2:
            first_intersection, second_intersection = self._get_relevant_doorway_pair(
                intersections,
                raw_center,
                aligned_doorway,
            )
            snapped_center = self._clamp_image_point(
                (first_intersection[0] + second_intersection[0]) / 2.0,
                (first_intersection[1] + second_intersection[1]) / 2.0,
            )
        else:
            snapped_center = self._clamp_image_point(*nearest_wall.point)

        return self._copy_doorway_with(
            aligned_doorway,
            center_x=snapped_center[0],
            center_y=snapped_center[1],
        )

    def _get_doorway_depth_intersections(
        self,
        doorway: DoorwayData,
        center: tuple[float, float],
    ) -> list[tuple[float, float]]:
        depth_direction_x, depth_direction_y = self._get_doorway_depth_direction(
            doorway
        )
        half_depth_pixels = doorway.depth_meters / PIXEL_TO_METER / 2.0
        depth_start = (
            center[0] - depth_direction_x * half_depth_pixels,
            center[1] - depth_direction_y * half_depth_pixels,
        )
        depth_end = (
            center[0] + depth_direction_x * half_depth_pixels,
            center[1] + depth_direction_y * half_depth_pixels,
        )
        intersections: list[tuple[float, float]] = []
        for edge in self.vertex_data.edges:
            start_vertex = self.vertex_data.get_vertex(edge.start_vertex_id)
            end_vertex = self.vertex_data.get_vertex(edge.end_vertex_id)
            if start_vertex is None or end_vertex is None:
                continue

            intersection = _find_segment_intersection(
                depth_start,
                depth_end,
                (start_vertex.x, start_vertex.y),
                (end_vertex.x, end_vertex.y),
            )
            if intersection is None or any(
                self._point_distance(intersection, existing_intersection) <= 1e-5
                for existing_intersection in intersections
            ):
                continue

            intersections.append(intersection)

        return intersections

    def _get_relevant_doorway_pair(
        self,
        intersections: list[tuple[float, float]],
        center: tuple[float, float],
        doorway: DoorwayData,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        depth_direction_x, depth_direction_y = self._get_doorway_depth_direction(
            doorway
        )
        ordered_intersections = sorted(
            intersections,
            key=lambda point: (
                (point[0] - center[0]) * depth_direction_x
                + (point[1] - center[1]) * depth_direction_y
            ),
        )
        center_tolerance = 1e-5
        before_center = [
            point
            for point in ordered_intersections
            if (
                (point[0] - center[0]) * depth_direction_x
                + (point[1] - center[1]) * depth_direction_y
            ) < -center_tolerance
        ]
        after_center = [
            point
            for point in ordered_intersections
            if (
                (point[0] - center[0]) * depth_direction_x
                + (point[1] - center[1]) * depth_direction_y
            ) > center_tolerance
        ]
        if before_center and after_center:
            return before_center[-1], after_center[0]

        centered_intersections = [
            point
            for point in ordered_intersections
            if abs(
                (point[0] - center[0]) * depth_direction_x
                + (point[1] - center[1]) * depth_direction_y
            ) <= center_tolerance
        ]
        if centered_intersections:
            other_intersections = [
                point
                for point in ordered_intersections
                if point not in centered_intersections
            ]
            if other_intersections:
                return centered_intersections[0], min(
                    other_intersections,
                    key=lambda point: self._point_distance(point, center),
                )

        closest_intersections = sorted(
            ordered_intersections,
            key=lambda point: self._point_distance(point, center),
        )
        return closest_intersections[0], closest_intersections[1]

    def _find_nearest_wall_projection(
        self,
        point: tuple[float, float],
    ) -> WallProjection | None:
        nearest_projection: WallProjection | None = None
        for edge in self.vertex_data.edges:
            start_vertex = self.vertex_data.get_vertex(edge.start_vertex_id)
            end_vertex = self.vertex_data.get_vertex(edge.end_vertex_id)
            if start_vertex is None or end_vertex is None:
                continue

            projected_point = _project_point_onto_segment(
                point,
                (start_vertex.x, start_vertex.y),
                (end_vertex.x, end_vertex.y),
            )
            if projected_point is None:
                continue

            distance = self._point_distance(point, projected_point)
            if (
                nearest_projection is None
                or distance < nearest_projection.distance
            ):
                nearest_projection = WallProjection(
                    edge=edge,
                    point=projected_point,
                    distance=distance,
                )

        return nearest_projection

    def _get_wall_normal_rotation_degrees(self, edge: Edge) -> float:
        start_vertex = self.vertex_data.get_vertex(edge.start_vertex_id)
        end_vertex = self.vertex_data.get_vertex(edge.end_vertex_id)
        if start_vertex is None or end_vertex is None:
            return 0.0

        wall_delta_x = end_vertex.x - start_vertex.x
        wall_delta_y = end_vertex.y - start_vertex.y
        if math.hypot(wall_delta_x, wall_delta_y) <= 1e-6:
            return 0.0

        return (math.degrees(math.atan2(wall_delta_y, wall_delta_x)) + 90.0) % 180.0

    def _get_doorway_hit_for_point(
        self,
        point: tuple[float, float],
        doorway: DoorwayData,
        hit_tolerance_pixels: float,
    ) -> DoorwayHit | None:
        depth_direction_x, depth_direction_y = self._get_doorway_depth_direction(
            doorway
        )
        width_direction_x = -depth_direction_y
        width_direction_y = depth_direction_x
        point_delta_x = point[0] - doorway.center_x
        point_delta_y = point[1] - doorway.center_y
        depth_position = (
            point_delta_x * depth_direction_x + point_delta_y * depth_direction_y
        )
        width_position = (
            point_delta_x * width_direction_x + point_delta_y * width_direction_y
        )
        half_depth_pixels = doorway.depth_meters / PIXEL_TO_METER / 2.0
        half_width_pixels = doorway.width_meters / PIXEL_TO_METER / 2.0
        if (
            abs(depth_position) > half_depth_pixels + hit_tolerance_pixels
            or abs(width_position) > half_width_pixels + hit_tolerance_pixels
        ):
            return None
        return DoorwayHit(doorway_index=-1)

    def _get_doorway_depth_direction(
        self,
        doorway: DoorwayData,
    ) -> tuple[float, float]:
        rotation_radians = math.radians(doorway.rotation_degrees)
        return math.cos(rotation_radians), math.sin(rotation_radians)

    def _get_doorway_width_direction(
        self,
        doorway: DoorwayData,
    ) -> tuple[float, float]:
        depth_direction_x, depth_direction_y = self._get_doorway_depth_direction(
            doorway
        )
        return -depth_direction_y, depth_direction_x

    def _copy_doorway_with(
        self,
        doorway: DoorwayData,
        *,
        center_x: float | None = None,
        center_y: float | None = None,
        width_meters: float | None = None,
        height_meters: float | None = None,
        depth_meters: float | None = None,
        rotation_degrees: float | None = None,
        shape: str | None = None,
        arch_amount: float | None = None,
        bottom_height_meters: float | None = None,
    ) -> DoorwayData:
        return DoorwayData(
            center_x=doorway.center_x if center_x is None else center_x,
            center_y=doorway.center_y if center_y is None else center_y,
            width_meters=(
                doorway.width_meters if width_meters is None else width_meters
            ),
            height_meters=(
                doorway.height_meters if height_meters is None else height_meters
            ),
            depth_meters=(
                doorway.depth_meters if depth_meters is None else depth_meters
            ),
            rotation_degrees=(
                doorway.rotation_degrees
                if rotation_degrees is None
                else rotation_degrees
            ),
            shape=doorway.shape if shape is None else shape,
            arch_amount=(
                doorway.arch_amount if arch_amount is None else arch_amount
            ),
            bottom_height_meters=(
                doorway.bottom_height_meters
                if bottom_height_meters is None
                else bottom_height_meters
            ),
        )

    def _push_undo_state(self) -> None:
        snapshot = CanvasSnapshot(
            vertex_data=self.vertex_data.clone(),
            rooms=copy.deepcopy(self.rooms),
            doorways=copy.deepcopy(self.doorways),
            floor_contour_vertex_ids=self.floor_contour_vertex_ids,
            active_vertex_id=self.active_vertex_id,
            selected_vertex_id=self.selected_vertex_id,
            preview_point=self.preview_point,
        )
        self.undo_stack.append(snapshot)

    def _delete_selected_doorway(self) -> bool:
        doorway_index = self.selected_doorway_index
        if doorway_index is None or not (0 <= doorway_index < len(self.doorways)):
            return False

        self._push_undo_state()
        del self.doorways[doorway_index]
        self._set_selected_doorway_index(None)
        self._reset_doorway_pointer_state()
        self.doorways_changed.emit()
        self.update()
        return True

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
        self._remove_vertex_from_rooms(deleted_vertex_id)
        if deleted_vertex_id in self.floor_contour_vertex_ids:
            self.floor_contour_vertex_ids = ()
            self._reset_floor_contour_designation()
            self.floor_contour_changed.emit(())

        if self.active_vertex_id == deleted_vertex_id:
            self.active_vertex_id = None
            self.preview_point = None

        self.selected_vertex_id = None
        self.preview_guides = []
        self._reset_pointer_state()
        self.update()

    def _handle_floor_contour_vertex_click(self, vertex_id: int) -> None:
        pending_vertex_ids = self.pending_floor_contour_vertex_ids
        if pending_vertex_ids is None:
            return

        if pending_vertex_ids and vertex_id == pending_vertex_ids[0]:
            if len(pending_vertex_ids) >= 3:
                self._push_undo_state()
                self.floor_contour_vertex_ids = tuple(pending_vertex_ids)
                self._reset_floor_contour_designation()
                self.floor_contour_changed.emit(self.floor_contour_vertex_ids)
                self.update()
            return

        if vertex_id in pending_vertex_ids:
            return

        pending_vertex_ids.append(vertex_id)
        vertex = self.vertex_data.get_vertex(vertex_id)
        self.pending_floor_contour_preview_point = (
            None if vertex is None else (vertex.x, vertex.y)
        )
        self.update()

    def _reset_floor_contour_designation(self) -> None:
        self.pending_floor_contour_vertex_ids = None
        self.pending_floor_contour_preview_point = None

    def _normalize_floor_contour_vertex_ids(
        self,
        vertex_ids: tuple[int, ...],
    ) -> tuple[int, ...]:
        normalized_vertex_ids = tuple(vertex_ids)
        if (
            len(normalized_vertex_ids) >= 2
            and normalized_vertex_ids[-1] == normalized_vertex_ids[0]
        ):
            normalized_vertex_ids = normalized_vertex_ids[:-1]

        if (
            len(normalized_vertex_ids) < 3
            or len(set(normalized_vertex_ids)) != len(normalized_vertex_ids)
        ):
            return ()

        existing_vertex_ids = {
            vertex.id for vertex in self.vertex_data.vertices
        }
        if any(
            vertex_id not in existing_vertex_ids
            for vertex_id in normalized_vertex_ids
        ):
            return ()

        return normalized_vertex_ids

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
                    height_meters=room.height_meters,
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
        base_vertex = (
            None
            if self.active_vertex_id is None
            else self.vertex_data.get_vertex(self.active_vertex_id)
        )
        return self._build_connection_preview_from_base(
            image_point=image_point,
            modifiers=modifiers,
            base_vertex=base_vertex,
        )

    def _build_connection_preview_from_base(
        self,
        image_point: QPointF,
        modifiers: Qt.KeyboardModifier,
        base_vertex: Vertex | None,
    ) -> SnapPreview:
        raw_x, raw_y = self._clamp_image_point(image_point.x(), image_point.y())
        raw_point = (raw_x, raw_y)
        modifier_bits = getattr(modifiers, "value", modifiers)
        control_modifier = Qt.KeyboardModifier.ControlModifier.value
        center_candidate = self._find_center_snap_candidate(raw_point)
        if center_candidate is not None:
            return self._build_center_snap_preview(center_candidate)

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

        display_width = image_width * scale
        display_height = image_height * scale
        display_center = available_rect.center()
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

    def _paint_floor_contour(self, painter: QPainter) -> None:
        contour_vertices = self._get_vertices_for_ids(
            self.floor_contour_vertex_ids
        )
        if len(contour_vertices) < 3:
            return

        contour_polygon = QPolygonF(
            [
                self._image_to_widget(vertex.x, vertex.y)
                for vertex in contour_vertices
            ]
        )
        contour_pen = QPen(
            FLOOR_CONTOUR_EDGE_COLOR,
            2.0,
            Qt.PenStyle.DashLine,
        )
        contour_pen.setDashPattern([6.0, 4.0])
        painter.setPen(contour_pen)
        painter.setBrush(FLOOR_CONTOUR_FILL_COLOR)
        painter.drawPolygon(contour_polygon)

    def _paint_pending_floor_contour(self, painter: QPainter) -> None:
        pending_vertex_ids = self.pending_floor_contour_vertex_ids
        if pending_vertex_ids is None or not pending_vertex_ids:
            return

        pending_vertices = self._get_vertices_for_ids(
            tuple(pending_vertex_ids)
        )
        if not pending_vertices:
            return

        pending_points = [
            self._image_to_widget(vertex.x, vertex.y)
            for vertex in pending_vertices
        ]
        pending_pen = QPen(
            PENDING_FLOOR_CONTOUR_COLOR,
            2.5,
            Qt.PenStyle.DashDotLine,
        )
        painter.setPen(pending_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if len(pending_points) >= 2:
            painter.drawPolyline(QPolygonF(pending_points))

        if self.pending_floor_contour_preview_point is not None:
            preview_point = self._image_to_widget(
                self.pending_floor_contour_preview_point[0],
                self.pending_floor_contour_preview_point[1],
            )
            painter.drawLine(pending_points[-1], preview_point)

        for pending_point in pending_points:
            painter.drawEllipse(pending_point, 4.0, 4.0)
        painter.drawEllipse(pending_points[0], 7.0, 7.0)

    def _get_vertices_for_ids(
        self,
        vertex_ids: tuple[int, ...],
    ) -> list[Vertex]:
        vertices: list[Vertex] = []
        for vertex_id in vertex_ids:
            vertex = self.vertex_data.get_vertex(vertex_id)
            if vertex is not None:
                vertices.append(vertex)

        return vertices

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

    # ### Window painting ###
    def _paint_windows(self, painter: QPainter) -> None:
        """Draw inert plan-view strips for the level's wall windows."""

        wall_frames = self._build_window_wall_frames()
        for window in self.windows:
            segment = self._get_window_widget_segment(
                window,
                wall_frames.get(window.wall_surface_id),
            )
            if segment is None:
                continue
            start, end = segment
            delta_x = end.x() - start.x()
            delta_y = end.y() - start.y()
            segment_length = math.hypot(delta_x, delta_y)
            if segment_length <= 1e-6:
                continue
            perpendicular = QPointF(
                -delta_y / segment_length * WINDOW_STRIP_HALF_WIDTH_SCREEN,
                delta_x / segment_length * WINDOW_STRIP_HALF_WIDTH_SCREEN,
            )
            strip = QPolygonF(
                (
                    start + perpendicular,
                    end + perpendicular,
                    end - perpendicular,
                    start - perpendicular,
                )
            )
            painter.setPen(QPen(WINDOW_EDGE_COLOR, 2.0))
            painter.setBrush(WINDOW_FILL_COLOR)
            painter.drawPolygon(strip)

    def _paint_doorways(self, painter: QPainter) -> None:
        for doorway_index, doorway in enumerate(self.doorways):
            is_selected = doorway_index == self.selected_doorway_index
            doorway_pen = QPen(
                (
                    SELECTED_DOORWAY_EDGE_COLOR
                    if is_selected
                    else DOORWAY_EDGE_COLOR
                ),
                2.5 if is_selected else 2.0,
            )
            painter.setPen(doorway_pen)
            painter.setBrush(DOORWAY_FILL_COLOR)
            painter.drawPolygon(self._get_doorway_widget_polygon(doorway))

            doorway_center = self._image_to_widget(doorway.center_x, doorway.center_y)
            painter.setPen(QPen(TEXT_COLOR))
            painter.setFont(QFont("Segoe UI", 8))
            label_rect = QRectF(
                doorway_center.x() - 56.0,
                doorway_center.y() - 18.0,
                112.0,
                36.0,
            )
            painter.drawText(
                label_rect,
                int(Qt.AlignmentFlag.AlignCenter),
                self._get_doorway_label_text(doorway),
            )

    # ### Stair painting ###
    def _paint_stairs(self, painter: QPainter) -> None:
        level = self.level_context
        if level is None:
            return

        for stair_index, raw_stair in enumerate(self.stairs):
            placement = _coerce_stair_placement(raw_stair)
            if placement is None:
                continue

            self._paint_stair_route_continuity(
                painter=painter,
                placement=placement,
                style=placement.style,
                pending=False,
            )
            for section_name, section in _get_stair_sections(placement):
                if section.level_index != level.index:
                    continue
                point_a_x, point_a_y = self._resolve_stair_canvas_point(
                    section.a_x,
                    section.a_y,
                    section.a_vertex_id,
                )
                point_b_x, point_b_y = self._resolve_stair_canvas_point(
                    section.b_x,
                    section.b_y,
                    section.b_vertex_id,
                )
                if section_name == "start":
                    label = (
                        f"Stairs to L{placement.end_level_index} "
                        f"({_format_stair_style_label(placement.style)})"
                    )
                    destination_level_index = placement.end_level_index
                elif section_name == "end":
                    label = (
                        f"Stairs to L{placement.start_level_index} "
                        f"({_format_stair_style_label(placement.style)})"
                    )
                    destination_level_index = placement.start_level_index
                else:
                    section_number = int(section_name.rsplit("_", 1)[-1]) + 1
                    label = (
                        f"Stair curve {section_number} "
                        f"({_format_stair_style_label(placement.style)})"
                    )
                    destination_level_index = None
                self._paint_stair_segment(
                    painter=painter,
                    point_a=self._image_to_widget(point_a_x, point_a_y),
                    point_b=self._image_to_widget(point_b_x, point_b_y),
                    style=placement.style,
                    label=label,
                    selected=stair_index == self.selected_stair_index,
                    destination_level_index=destination_level_index,
                )

    def _paint_pending_stair_placement(self, painter: QPainter) -> None:
        if not self._is_stair_placement_active():
            return

        level = self.level_context
        style = self.pending_stair_style
        if level is None or style is None:
            return

        pending_placement = self.pending_stair_placement
        pending_draft = self.pending_stair_draft
        if pending_draft is not None:
            self._paint_stair_route_continuity(
                painter=painter,
                placement=pending_draft,
                style=style,
                pending=True,
            )
            for section_name, section in _get_stair_sections(pending_draft):
                if section.level_index != level.index:
                    continue
                point_a_x, point_a_y = self._resolve_stair_canvas_point(
                    section.a_x,
                    section.a_y,
                    section.a_vertex_id,
                )
                point_b_x, point_b_y = self._resolve_stair_canvas_point(
                    section.b_x,
                    section.b_y,
                    section.b_vertex_id,
                )
                label = (
                    "Stair start - add curve sections or confirm"
                    if section_name == "start"
                    else "Stair end - add curve sections or confirm"
                    if section_name == "end"
                    else "Stair curve control"
                )
                self._paint_stair_segment(
                    painter=painter,
                    point_a=self._image_to_widget(point_a_x, point_a_y),
                    point_b=self._image_to_widget(point_b_x, point_b_y),
                    style=style,
                    label=label,
                    selected=True,
                    pending=True,
                )
        elif (
            pending_placement is not None
            and pending_placement.start_level_index == level.index
        ):
            self._paint_stair_segment(
                painter=painter,
                point_a=self._image_to_widget(
                    pending_placement.start_a_x,
                    pending_placement.start_a_y,
                ),
                point_b=self._image_to_widget(
                    pending_placement.start_b_x,
                    pending_placement.start_b_y,
                ),
                style=style,
                label="Stair start segment - choose another level",
                selected=True,
                pending=True,
            )

        pending_point = self.pending_stair_point
        pending_point_widget = None
        if pending_point is not None and pending_point.level_index == level.index:
            pending_point_widget = self._image_to_widget(
                pending_point.x,
                pending_point.y,
            )
            self._paint_stair_endpoint(
                painter=painter,
                point=pending_point_widget,
                other_point=None,
                style=style,
                label=(
                    "Start A"
                    if pending_placement is None
                    else "Curve A"
                    if pending_draft is not None
                    else "End A"
                ),
                selected=True,
                pending=True,
            )

        preview_point = self.pending_stair_preview_point
        if preview_point is None:
            return

        preview_widget_point = self._image_to_widget(
            preview_point[0],
            preview_point[1]
        )
        if pending_point_widget is not None:
            preview_pen = QPen(PENDING_STAIR_COLOR, 2.0, Qt.PenStyle.DashLine)
            preview_pen.setDashPattern([5.0, 4.0])
            painter.setPen(preview_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawLine(pending_point_widget, preview_widget_point)

        self._paint_stair_endpoint(
            painter=painter,
            point=preview_widget_point,
            other_point=None,
            style=style,
            label=(
                "Curve B preview"
                if pending_draft is not None and pending_point is not None
                else "Curve A preview"
                if pending_draft is not None
                else "End B preview"
                if pending_placement is not None and pending_point is not None
                else "End A preview"
                if pending_placement is not None
                else "Start B preview"
                if pending_point is not None
                else "Start A preview"
            ),
            selected=False,
            pending=True,
        )
        self._paint_stair_preview_guides(painter)

    # ### Stair route continuity painting ###
    def _paint_stair_route_continuity(
        self,
        painter: QPainter,
        placement: StairPlacement,
        style: str,
        pending: bool,
    ) -> None:
        """Show that ordered guide cross-sections belong to one stair route.

        Two sections can share a level and therefore have a useful plan-view
        connector.  A cross-level route cannot be drawn in either level's
        local blueprint space, so it receives an explicit directional cue
        instead.  The actual route order always remains start, guides, end.
        """

        level = self.level_context
        if level is None:
            return

        named_sections = _get_stair_sections(placement)
        for (
            previous_name,
            previous_section,
        ), (
            next_name,
            next_section,
        ) in zip(named_sections, named_sections[1:]):
            # A straight stair already paints its start/end transition arrow
            # with the segment itself.  The continuity marker is useful only
            # once an intermediate route control exists.
            if previous_name == "start" and next_name == "end":
                continue

            previous_is_visible = previous_section.level_index == level.index
            next_is_visible = next_section.level_index == level.index
            if previous_is_visible and next_is_visible:
                self._paint_stair_route_centerline(
                    painter=painter,
                    start_point=self._get_stair_section_midpoint_widget(
                        previous_section
                    ),
                    end_point=self._get_stair_section_midpoint_widget(
                        next_section
                    ),
                    style=style,
                    pending=pending,
                )
                continue

            if previous_is_visible:
                self._paint_stair_route_level_transition(
                    painter=painter,
                    point=self._get_stair_section_midpoint_widget(
                        previous_section
                    ),
                    source_level_index=previous_section.level_index,
                    destination_level_index=next_section.level_index,
                    label=(
                        "Curve route to"
                        if next_name != "end"
                        else "Stair route to"
                    ),
                    style=style,
                    pending=pending,
                )
            elif next_is_visible:
                self._paint_stair_route_level_transition(
                    painter=painter,
                    point=self._get_stair_section_midpoint_widget(next_section),
                    source_level_index=next_section.level_index,
                    destination_level_index=previous_section.level_index,
                    label=(
                        "Curve route from"
                        if previous_name != "start"
                        else "Stair route from"
                    ),
                    style=style,
                    pending=pending,
                )

    def _get_stair_section_midpoint_widget(
        self,
        section: StairSectionPlacement,
    ) -> QPointF:
        """Resolve one route section's current Canvas midpoint."""

        point_a_x, point_a_y = self._resolve_stair_canvas_point(
            section.a_x,
            section.a_y,
            section.a_vertex_id,
        )
        point_b_x, point_b_y = self._resolve_stair_canvas_point(
            section.b_x,
            section.b_y,
            section.b_vertex_id,
        )
        return (
            self._image_to_widget(point_a_x, point_a_y)
            + self._image_to_widget(point_b_x, point_b_y)
        ) * 0.5

    def _paint_stair_route_centerline(
        self,
        painter: QPainter,
        start_point: QPointF,
        end_point: QPointF,
        style: str,
        pending: bool,
    ) -> None:
        """Paint the local-plan portion between two consecutive sections."""

        color = PENDING_STAIR_COLOR if pending else _get_stair_style_color(style)
        painter.save()
        route_pen = QPen(
            color,
            STAIR_ROUTE_CENTERLINE_WIDTH_SCREEN,
            Qt.PenStyle.DashLine,
        )
        route_pen.setCosmetic(True)
        route_pen.setDashPattern([4.0, 4.0])
        painter.setPen(route_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawLine(start_point, end_point)
        painter.restore()

    def _paint_stair_route_level_transition(
        self,
        painter: QPainter,
        point: QPointF,
        source_level_index: int,
        destination_level_index: int,
        label: str,
        style: str,
        pending: bool,
    ) -> None:
        """Mark a route segment which continues on another Canvas level."""

        color = PENDING_STAIR_COLOR if pending else _get_stair_style_color(style)
        direction_point = _get_level_transition_indicator_point(
            point,
            source_level_index,
            destination_level_index,
        )
        self._paint_stair_direction_arrow(
            painter,
            point,
            direction_point,
            color,
        )
        painter.save()
        painter.setPen(QPen(TEXT_COLOR))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(
            direction_point + QPointF(6.0, -6.0),
            f"{label} L{destination_level_index}",
        )
        painter.restore()

    def _paint_stair_segment(
        self,
        painter: QPainter,
        point_a: QPointF,
        point_b: QPointF,
        style: str,
        label: str,
        selected: bool,
        destination_level_index: int | None = None,
        pending: bool = False,
    ) -> None:
        """Paint the two-point footprint that anchors stairs on one level."""

        base_color = _get_stair_style_color(style)
        color = PENDING_STAIR_COLOR if pending else base_color
        if selected and not pending:
            color = STAIR_SELECTED_COLOR

        painter.save()
        segment_pen = QPen(
            color,
            4.0 if selected else 3.0,
            Qt.PenStyle.DashLine if pending else Qt.PenStyle.SolidLine,
        )
        segment_pen.setCosmetic(True)
        painter.setPen(segment_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawLine(point_a, point_b)
        painter.restore()

        self._paint_stair_endpoint(
            painter=painter,
            point=point_a,
            other_point=None,
            style=style,
            label="A",
            selected=selected,
            pending=pending,
        )
        self._paint_stair_endpoint(
            painter=painter,
            point=point_b,
            other_point=None,
            style=style,
            label="B",
            selected=selected,
            pending=pending,
        )

        midpoint = (point_a + point_b) * 0.5
        if destination_level_index is not None and self.level_context is not None:
            direction_point = _get_level_transition_indicator_point(
                midpoint,
                self.level_context.index,
                destination_level_index,
            )
            self._paint_stair_direction_arrow(
                painter,
                midpoint,
                direction_point,
                color,
            )

        painter.save()
        painter.setPen(QPen(TEXT_COLOR))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(midpoint + QPointF(13.0, -13.0), label)
        painter.restore()

    def _paint_stair_direction_arrow(
        self,
        painter: QPainter,
        point: QPointF,
        other_point: QPointF,
        color: QColor,
    ) -> None:
        """Paint the up/down level cue from the midpoint of a stair segment."""

        direction = other_point - point
        direction_length = math.hypot(direction.x(), direction.y())
        if direction_length <= 1e-6:
            return

        direction = direction * (STAIR_DIRECTION_LENGTH_SCREEN / direction_length)
        arrow_tip = point + direction
        arrow_side = QPointF(-direction.y(), direction.x()) * 0.25
        painter.save()
        painter.setPen(QPen(color, 2.0))
        painter.setBrush(color)
        painter.drawLine(point, arrow_tip)
        painter.drawPolygon(
            QPolygonF(
                [
                    arrow_tip,
                    arrow_tip - direction * 0.42 + arrow_side * 5.0,
                    arrow_tip - direction * 0.42 - arrow_side * 5.0,
                ]
            )
        )
        painter.restore()

    def _paint_stair_endpoint(
        self,
        painter: QPainter,
        point: QPointF,
        other_point: QPointF | None,
        style: str,
        label: str,
        selected: bool,
        pending: bool = False,
    ) -> None:
        base_color = _get_stair_style_color(style)
        color = PENDING_STAIR_COLOR if pending else base_color
        if selected and not pending:
            color = STAIR_SELECTED_COLOR

        painter.save()
        endpoint_pen = QPen(
            color,
            3.0 if selected else 2.0,
            Qt.PenStyle.DashLine if pending else Qt.PenStyle.SolidLine,
        )
        endpoint_pen.setCosmetic(True)
        painter.setPen(endpoint_pen)
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 88))
        if _is_floating_stair_style(style):
            diamond = QPolygonF(
                [
                    point + QPointF(0.0, -STAIR_ENDPOINT_RADIUS_SCREEN),
                    point + QPointF(STAIR_ENDPOINT_RADIUS_SCREEN, 0.0),
                    point + QPointF(0.0, STAIR_ENDPOINT_RADIUS_SCREEN),
                    point + QPointF(-STAIR_ENDPOINT_RADIUS_SCREEN, 0.0),
                ]
            )
            painter.drawPolygon(diamond)
            if style == STAIR_STYLE_FLOATING_WITH_RISER:
                painter.drawLine(
                    point + QPointF(-6.0, 2.5),
                    point + QPointF(6.0, 2.5),
                )
        else:
            painter.drawEllipse(
                point,
                STAIR_ENDPOINT_RADIUS_SCREEN,
                STAIR_ENDPOINT_RADIUS_SCREEN,
            )
            painter.drawLine(
                point + QPointF(-7.0, 11.0),
                point + QPointF(7.0, 11.0),
            )

        if other_point is not None:
            direction = other_point - point
            direction_length = math.hypot(direction.x(), direction.y())
            if direction_length > 1e-6:
                direction = direction * (STAIR_DIRECTION_LENGTH_SCREEN / direction_length)
                arrow_tip = point + direction
                painter.drawLine(point, arrow_tip)
                arrow_side = QPointF(-direction.y(), direction.x()) * 0.25
                painter.setBrush(color)
                painter.drawPolygon(
                    QPolygonF(
                        [
                            arrow_tip,
                            arrow_tip - direction * 0.42 + arrow_side * 5.0,
                            arrow_tip - direction * 0.42 - arrow_side * 5.0,
                        ]
                    )
                )

        painter.setPen(QPen(TEXT_COLOR))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(point + QPointF(13.0, -13.0), label)
        painter.restore()

    def _paint_pending_doorway(self, painter: QPainter) -> None:
        doorway = self.pending_doorway
        if doorway is None:
            return

        doorway_pen = QPen(
            PENDING_DOORWAY_EDGE_COLOR,
            2.5,
            Qt.PenStyle.DashLine,
        )
        doorway_pen.setDashPattern([6.0, 4.0])
        painter.setPen(doorway_pen)
        painter.setBrush(PENDING_DOORWAY_FILL_COLOR)
        painter.drawPolygon(self._get_doorway_widget_polygon(doorway))

    def _get_doorway_widget_polygon(self, doorway: DoorwayData) -> QPolygonF:
        return QPolygonF(
            [
                self._image_to_widget(point[0], point[1])
                for point in self._get_doorway_corners(doorway)
            ]
        )

    def _get_doorway_corners(
        self,
        doorway: DoorwayData,
    ) -> list[tuple[float, float]]:
        depth_direction_x, depth_direction_y = self._get_doorway_depth_direction(
            doorway
        )
        width_direction_x = -depth_direction_y
        width_direction_y = depth_direction_x
        half_depth_pixels = doorway.depth_meters / PIXEL_TO_METER / 2.0
        half_width_pixels = doorway.width_meters / PIXEL_TO_METER / 2.0
        return [
            (
                doorway.center_x
                + depth_sign * depth_direction_x * half_depth_pixels
                + width_sign * width_direction_x * half_width_pixels,
                doorway.center_y
                + depth_sign * depth_direction_y * half_depth_pixels
                + width_sign * width_direction_y * half_width_pixels,
            )
            for depth_sign, width_sign in ((-1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (1.0, -1.0))
        ]

    def _paint_preview_guides(self, painter: QPainter) -> None:
        self._paint_guides(
            painter,
            self.preview_point,
            self.preview_guides,
        )

    def _paint_stair_preview_guides(self, painter: QPainter) -> None:
        self._paint_guides(
            painter,
            self.pending_stair_preview_point,
            self.pending_stair_preview_guides,
        )

    def _paint_guides(
        self,
        painter: QPainter,
        preview_point: tuple[float, float] | None,
        guides: list[SnapGuide],
    ) -> None:
        if preview_point is None or not guides:
            return

        guide_pen = QPen(GUIDE_COLOR, 1.5, Qt.PenStyle.DashLine)
        guide_pen.setDashPattern([3.0, 5.0])
        painter.setPen(guide_pen)
        preview_widget_point = self._image_to_widget(
            preview_point[0],
            preview_point[1],
        )

        for guide in guides:
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
                    self._image_to_widget(source_vertex.x, preview_point[1]),
                )
                continue

            painter.drawLine(
                self._image_to_widget(source_vertex.x, source_vertex.y),
                self._image_to_widget(preview_point[0], source_vertex.y),
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

        overlay_lines = [
            f"Blueprint: {Path(self.blueprint_path).name}",
            "Left click: add/connect/select/drag",
            "Mouse wheel: zoom | Delete: remove selected vertex | Ctrl+Z: undo | Hold Ctrl: free placement",
        ]
        if self._is_stair_placement_active():
            pending_point = self.pending_stair_point
            pending_stair = self.pending_stair_placement
            pending_draft = self.pending_stair_draft
            if pending_draft is not None and pending_point is None:
                overlay_lines.append(
                    "Stairs: add optional two-point curve sections, or click "
                    "Confirm stairs | Backspace: remove last curve section."
                )
            elif pending_draft is not None:
                overlay_lines.append(
                    f"Stairs: place curve point B on level "
                    f"{pending_point.level_index} | Backspace: discard point A."
                )
            elif pending_stair is None and pending_point is None:
                overlay_lines.append(
                    "Stairs: click start point A anywhere; existing vertices "
                    "and wall edges snap | Right click or Escape: cancel."
                )
            elif pending_stair is None:
                overlay_lines.append(
                    f"Stairs: place start point B on level "
                    f"{pending_point.level_index}."
                )
            elif pending_point is None:
                overlay_lines.append(
                    "Stairs: select a different level, then place end point A."
                )
            else:
                overlay_lines.append(
                    f"Stairs: place end point B on level "
                    f"{pending_point.level_index}."
                )
        if self.pending_doorway_preset is not None:
            overlay_lines.append(
                "Doorway: click to place | Mouse wheel: zoom | Right click or Escape: cancel."
            )
        elif self.doorways:
            overlay_lines.append(
                "Doorway: select and drag it to move it along its wall."
            )
        if self.stairs:
            overlay_lines.append(
                "Stairs: Alt+click any A/B point to select; Delete removes the selected stair."
            )
        if self.pending_floor_contour_vertex_ids is not None:
            overlay_lines.append(
                "Floor contour: click every perimeter corner in order, "
                "including inward corners; click the first again to finish."
            )
            overlay_lines.append("Right click or Escape: cancel floor contour.")

        overlay_rect = QRectF(24.0, 24.0, 680.0, 20.0 + len(overlay_lines) * 22.0)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(10, 12, 16, 180))
        painter.drawRoundedRect(overlay_rect, 10.0, 10.0)

        painter.setPen(QPen(TEXT_COLOR))
        painter.setFont(QFont("Segoe UI", 9))
        line_y = overlay_rect.top() + 24.0
        for line in overlay_lines:
            painter.drawText(QPointF(overlay_rect.left() + 14.0, line_y), line)
            line_y += 22.0


# ### Numeric helpers ###
def _normalize_stair_style(style: object) -> str:
    normalized_style = str(style).strip().lower()
    if normalized_style in STAIR_STYLES:
        return normalized_style
    return DEFAULT_STAIR_STYLE


def _format_stair_style_label(style: str) -> str:
    """Return the Canvas-facing name for a known stair construction style."""

    if style == STAIR_STYLE_FLOATING_WITH_RISER:
        return "Floating with riser"
    if style == STAIR_STYLE_FLOATING:
        return "Floating"
    return "Supported"


def _is_floating_stair_style(style: str) -> bool:
    """Return whether a style uses the floating-stair Canvas marker."""

    return style in (
        STAIR_STYLE_FLOATING,
        STAIR_STYLE_FLOATING_WITH_RISER,
    )


def _get_stair_style_color(style: str) -> QColor:
    if style == STAIR_STYLE_FLOATING_WITH_RISER:
        return STAIR_FLOATING_WITH_RISER_COLOR
    if style == STAIR_STYLE_FLOATING:
        return STAIR_FLOATING_COLOR
    return STAIR_SUPPORTED_COLOR


def _get_stair_sections(
    placement: StairPlacement,
) -> tuple[tuple[str, StairSectionPlacement], ...]:
    """Return the full ordered route as consistently shaped sections."""

    start_section = StairSectionPlacement(
        level_index=placement.start_level_index,
        a_x=placement.start_a_x,
        a_y=placement.start_a_y,
        b_x=placement.start_b_x,
        b_y=placement.start_b_y,
        a_vertex_id=placement.start_a_vertex_id,
        b_vertex_id=placement.start_b_vertex_id,
    )
    end_section = StairSectionPlacement(
        level_index=placement.end_level_index,
        a_x=placement.end_a_x,
        a_y=placement.end_a_y,
        b_x=placement.end_b_x,
        b_y=placement.end_b_y,
        a_vertex_id=placement.end_a_vertex_id,
        b_vertex_id=placement.end_b_vertex_id,
    )
    named_intermediate_sections = tuple(
        (f"intermediate_{section_index}", section)
        for section_index, section in enumerate(
            placement.intermediate_sections
        )
    )
    return (
        ("start", start_section),
        *named_intermediate_sections,
        ("end", end_section),
    )


def _get_level_transition_indicator_point(
    endpoint_point: QPointF,
    source_level_index: int,
    destination_level_index: int,
) -> QPointF:
    direction_y = (
        -STAIR_DIRECTION_LENGTH_SCREEN
        if destination_level_index > source_level_index
        else STAIR_DIRECTION_LENGTH_SCREEN
    )
    return endpoint_point + QPointF(0.0, direction_y)


def _coerce_pending_stair_placement(
    raw_placement: object | None,
) -> PendingStairPlacement | None:
    if raw_placement is None:
        return None
    if isinstance(raw_placement, PendingStairPlacement):
        return raw_placement

    start_level_index = _coerce_stair_level_index(
        _get_stair_value(raw_placement, "start_level_index")
    )
    start_a_x = _coerce_stair_coordinate(
        _get_stair_value(raw_placement, "start_a_x")
    )
    start_a_y = _coerce_stair_coordinate(
        _get_stair_value(raw_placement, "start_a_y")
    )
    start_b_x = _coerce_stair_coordinate(
        _get_stair_value(raw_placement, "start_b_x")
    )
    start_b_y = _coerce_stair_coordinate(
        _get_stair_value(raw_placement, "start_b_y")
    )
    if None in (start_a_x, start_a_y, start_b_x, start_b_y):
        legacy_start_x = _coerce_stair_coordinate(
            _get_stair_value(raw_placement, "start_x")
        )
        legacy_start_y = _coerce_stair_coordinate(
            _get_stair_value(raw_placement, "start_y")
        )
        start_a_x = start_b_x = legacy_start_x
        start_a_y = start_b_y = legacy_start_y
    if (
        start_level_index is None
        or start_a_x is None
        or start_a_y is None
        or start_b_x is None
        or start_b_y is None
    ):
        return None

    return PendingStairPlacement(
        style=_normalize_stair_style(_get_stair_value(raw_placement, "style")),
        start_level_index=start_level_index,
        start_a_x=start_a_x,
        start_a_y=start_a_y,
        start_b_x=start_b_x,
        start_b_y=start_b_y,
        start_a_vertex_id=_coerce_optional_stair_vertex_id(
            _get_stair_value(raw_placement, "start_a_vertex_id")
        ),
        start_b_vertex_id=_coerce_optional_stair_vertex_id(
            _get_stair_value(raw_placement, "start_b_vertex_id")
        ),
    )


def _coerce_stair_placement(raw_stair: object) -> StairPlacement | None:
    if isinstance(raw_stair, StairPlacement):
        return raw_stair

    start_level_index = _coerce_stair_level_index(
        _get_stair_value(raw_stair, "start_level_index")
    )
    end_level_index = _coerce_stair_level_index(
        _get_stair_value(raw_stair, "end_level_index")
    )
    start_a_x = _coerce_stair_coordinate(
        _get_stair_value(raw_stair, "start_a_x")
    )
    start_a_y = _coerce_stair_coordinate(
        _get_stair_value(raw_stair, "start_a_y")
    )
    start_b_x = _coerce_stair_coordinate(
        _get_stair_value(raw_stair, "start_b_x")
    )
    start_b_y = _coerce_stair_coordinate(
        _get_stair_value(raw_stair, "start_b_y")
    )
    end_a_x = _coerce_stair_coordinate(_get_stair_value(raw_stair, "end_a_x"))
    end_a_y = _coerce_stair_coordinate(_get_stair_value(raw_stair, "end_a_y"))
    end_b_x = _coerce_stair_coordinate(_get_stair_value(raw_stair, "end_b_x"))
    end_b_y = _coerce_stair_coordinate(_get_stair_value(raw_stair, "end_b_y"))
    if None in (start_a_x, start_a_y, start_b_x, start_b_y):
        legacy_start_x = _coerce_stair_coordinate(
            _get_stair_value(raw_stair, "start_x")
        )
        legacy_start_y = _coerce_stair_coordinate(
            _get_stair_value(raw_stair, "start_y")
        )
        start_a_x = start_b_x = legacy_start_x
        start_a_y = start_b_y = legacy_start_y
    if None in (end_a_x, end_a_y, end_b_x, end_b_y):
        legacy_end_x = _coerce_stair_coordinate(
            _get_stair_value(raw_stair, "end_x")
        )
        legacy_end_y = _coerce_stair_coordinate(
            _get_stair_value(raw_stair, "end_y")
        )
        end_a_x = end_b_x = legacy_end_x
        end_a_y = end_b_y = legacy_end_y
    if (
        start_level_index is None
        or end_level_index is None
        or start_a_x is None
        or start_a_y is None
        or start_b_x is None
        or start_b_y is None
        or end_a_x is None
        or end_a_y is None
        or end_b_x is None
        or end_b_y is None
    ):
        return None

    return StairPlacement(
        style=_normalize_stair_style(_get_stair_value(raw_stair, "style")),
        start_level_index=start_level_index,
        start_a_x=start_a_x,
        start_a_y=start_a_y,
        start_b_x=start_b_x,
        start_b_y=start_b_y,
        end_level_index=end_level_index,
        end_a_x=end_a_x,
        end_a_y=end_a_y,
        end_b_x=end_b_x,
        end_b_y=end_b_y,
        intermediate_sections=_coerce_stair_intermediate_sections(
            _get_stair_value(raw_stair, "intermediate_sections")
        ),
        start_a_vertex_id=_coerce_optional_stair_vertex_id(
            _get_stair_value(raw_stair, "start_a_vertex_id")
        ),
        start_b_vertex_id=_coerce_optional_stair_vertex_id(
            _get_stair_value(raw_stair, "start_b_vertex_id")
        ),
        end_a_vertex_id=_coerce_optional_stair_vertex_id(
            _get_stair_value(raw_stair, "end_a_vertex_id")
        ),
        end_b_vertex_id=_coerce_optional_stair_vertex_id(
            _get_stair_value(raw_stair, "end_b_vertex_id")
        ),
    )


def _coerce_stair_intermediate_sections(
    raw_sections: object | None,
) -> tuple[StairSectionPlacement, ...]:
    """Convert model or mapping sections into Canvas-neutral data."""

    if raw_sections is None:
        return ()
    if not isinstance(raw_sections, Iterable) or isinstance(
        raw_sections,
        (str, bytes, Mapping),
    ):
        return ()

    sections: list[StairSectionPlacement] = []
    for raw_section in raw_sections:
        section = _coerce_stair_section(raw_section)
        if section is not None:
            sections.append(section)
    return tuple(sections)


def _coerce_stair_section(
    raw_section: object,
) -> StairSectionPlacement | None:
    level_index = _coerce_stair_level_index(
        _get_stair_value(raw_section, "level_index")
    )
    a_x = _coerce_stair_coordinate(_get_stair_value(raw_section, "a_x"))
    a_y = _coerce_stair_coordinate(_get_stair_value(raw_section, "a_y"))
    b_x = _coerce_stair_coordinate(_get_stair_value(raw_section, "b_x"))
    b_y = _coerce_stair_coordinate(_get_stair_value(raw_section, "b_y"))
    if (
        level_index is None
        or a_x is None
        or a_y is None
        or b_x is None
        or b_y is None
    ):
        return None
    return StairSectionPlacement(
        level_index=level_index,
        a_x=a_x,
        a_y=a_y,
        b_x=b_x,
        b_y=b_y,
        a_vertex_id=_coerce_optional_stair_vertex_id(
            _get_stair_value(raw_section, "a_vertex_id")
        ),
        b_vertex_id=_coerce_optional_stair_vertex_id(
            _get_stair_value(raw_section, "b_vertex_id")
        ),
    )


def _get_stair_value(raw_stair: object, field_name: str) -> object | None:
    if isinstance(raw_stair, Mapping):
        return raw_stair.get(field_name)
    return getattr(raw_stair, field_name, None)


def _coerce_stair_level_index(value: object | None) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        numeric_value = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric_value


def _coerce_optional_stair_vertex_id(value: object | None) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        vertex_id = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return vertex_id if vertex_id >= 0 else None


def _coerce_stair_coordinate(value: object | None) -> float | None:
    try:
        numeric_value = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(numeric_value):
        return None
    return numeric_value


def _qpoint_distance(first_point: QPointF, second_point: QPointF) -> float:
    delta = first_point - second_point
    return math.hypot(delta.x(), delta.y())


def _points_are_coincident(
    first_point: tuple[float, float],
    second_point: tuple[float, float],
) -> bool:
    return math.hypot(
        first_point[0] - second_point[0],
        first_point[1] - second_point[1],
    ) <= 1e-6


# ### File helpers ###
def _build_blueprint_image_revision(
    file_path: str,
) -> tuple[object, ...]:
    """Identify one blueprint image revision without decoding pixels."""

    normalized_path = str(Path(file_path).resolve())
    try:
        image_stat = Path(normalized_path).stat()
    except OSError:
        return normalized_path, None, None, None
    return (
        normalized_path,
        int(image_stat.st_size),
        int(image_stat.st_mtime_ns),
        int(image_stat.st_ctime_ns),
    )


def _blueprint_revision_has_file(revision: tuple[object, ...]) -> bool:
    """Distinguish a cacheable missing path from an existing image revision."""

    return len(revision) == 4 and all(value is not None for value in revision[1:])


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
def _project_point_onto_segment(
    point: tuple[float, float],
    segment_start: tuple[float, float],
    segment_end: tuple[float, float],
) -> tuple[float, float] | None:
    segment_delta_x = segment_end[0] - segment_start[0]
    segment_delta_y = segment_end[1] - segment_start[1]
    segment_length_squared = (
        segment_delta_x * segment_delta_x + segment_delta_y * segment_delta_y
    )
    if segment_length_squared <= 1e-6:
        return None

    point_delta_x = point[0] - segment_start[0]
    point_delta_y = point[1] - segment_start[1]
    segment_ratio = min(
        max(
            (
                point_delta_x * segment_delta_x
                + point_delta_y * segment_delta_y
            )
            / segment_length_squared,
            0.0,
        ),
        1.0,
    )
    return (
        segment_start[0] + segment_delta_x * segment_ratio,
        segment_start[1] + segment_delta_y * segment_ratio,
    )


def _find_segment_intersection(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> tuple[float, float] | None:
    first_delta_x = first_end[0] - first_start[0]
    first_delta_y = first_end[1] - first_start[1]
    second_delta_x = second_end[0] - second_start[0]
    second_delta_y = second_end[1] - second_start[1]
    cross_product = (
        first_delta_x * second_delta_y - first_delta_y * second_delta_x
    )
    if abs(cross_product) <= 1e-6:
        return None

    start_delta_x = second_start[0] - first_start[0]
    start_delta_y = second_start[1] - first_start[1]
    first_ratio = (
        start_delta_x * second_delta_y - start_delta_y * second_delta_x
    ) / cross_product
    second_ratio = (
        start_delta_x * first_delta_y - start_delta_y * first_delta_x
    ) / cross_product
    if not (
        -1e-6 <= first_ratio <= 1.0 + 1e-6
        and -1e-6 <= second_ratio <= 1.0 + 1e-6
    ):
        return None

    return (
        first_start[0] + first_delta_x * first_ratio,
        first_start[1] + first_delta_y * first_ratio,
    )


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
