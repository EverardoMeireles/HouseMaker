# ### Imports ###
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from uuid import uuid4

from PySide6.QtCore import QElapsedTimer, QEvent, QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSplitter,
    QStackedLayout,
    QStyle,
    QStyleOptionSlider,
    QVBoxLayout,
    QWidget,
)

from housemaker.tour_state import (
    DEFAULT_IDLE_CAMERA_CYCLE_DURATION_SECONDS,
    DEFAULT_IDLE_CAMERA_RADIUS_METERS,
    DEFAULT_SPEED_EASING,
    DEFAULT_SPEED_OVERRIDE,
    DEFAULT_TEXT_COLOR,
    DEFAULT_TEXT_FADE_DELAY_MS,
    DEFAULT_TEXT_FADE_DURATION_MS,
    DEFAULT_TEXT_ROTATION_DEGREES,
    DEFAULT_TEXT_SIZE_POINTS,
    DEFAULT_TOOLTIP_HTML_BODY,
    DEFAULT_TOOLTIP_POSITION,
    DEFAULT_TOOLTIP_STYLE,
    DEFAULT_TOUR_DURATION_SECONDS,
    DEFAULT_TOUR_PROGRESS_SPEED,
    DEFAULT_TOUR_TRIGGER_AREA_SIZE_METERS,
    DEFAULT_WAIT_INPUT_TOKEN,
    TOUR_SPEED_EASING_OPTIONS,
    TourActionData,
    TourCurveData,
    TourData,
    TourFloatingTooltipData,
    TourIdleCameraAnimationData,
    TourSpeedOverrideData,
    TourStepData,
    TourText3DData,
    TourWaitForKeyPressData,
    effective_step_easing,
    evaluate_catmull_rom_point,
)

# ### Constants ###
TOUR_POINT_ACTION_TRIGGER = "trigger"
TOUR_POINT_ACTION_CURVE = "curve"

DEFAULT_TOUR_TENSION = 0.5
MINIMUM_CURVE_POINT_COUNT = 2
PREVIEW_FRAMES_PER_SECOND = 60.0
PREVIEW_FRAME_INTERVAL_MILLISECONDS = 1_000.0 / PREVIEW_FRAMES_PER_SECOND
TIMELINE_RESOLUTION = 1_000_000
CAMERA_LOOK_AHEAD_PROGRESS = 0.01
STEP_LIST_MAXIMUM_HEIGHT = 120
TIMELINE_MARKER_SNAP_TOLERANCE_PIXELS = 5.0
SPEED_INTEGRATION_SUBDIVISIONS = 64
SPEED_INVERSION_ITERATIONS = 36

INPUT_TOKEN_OPTIONS: tuple[tuple[str, str], ...] = (
    ("any", "any"),
    ("Left mouse button", "mouse:left"),
    ("Middle mouse button", "mouse:middle"),
    ("Right mouse button", "mouse:right"),
    ("Space", "key:space"),
    ("Enter", "key:enter"),
    ("Escape", "key:escape"),
    ("Tab", "key:tab"),
    ("Up arrow", "key:up"),
    ("Down arrow", "key:down"),
    ("Left arrow", "key:left"),
    ("Right arrow", "key:right"),
    *((letter.upper(), f"key:{letter}") for letter in "abcdefghijklmnopqrstuvwxyz"),
    *((digit, f"key:{digit}") for digit in "0123456789"),
)

EASING_LABELS = {
    "none": "None",
    "linear": "Linear",
    "smoothstep": "Smoothstep",
    "smootherstep": "Smootherstep",
    "ease_in_quad": "Ease in quadratic",
    "ease_out_quad": "Ease out quadratic",
    "ease_in_out_sine": "Ease in-out sine",
    "ease_in_out_quad": "Ease in-out quadratic",
    "ease_in_out_cubic": "Ease in-out cubic",
}

ITEM_ID_ROLE = Qt.ItemDataRole.UserRole
ITEM_DRAFT_ROLE = Qt.ItemDataRole.UserRole + 1


# ### Playback value models ###
@dataclass(frozen=True, slots=True)
class _TourSpeedSegment:
    """One progress interval with an optionally eased speed transition."""

    start_progress: float
    end_progress: float
    start_speed: float
    end_speed: float
    easing: str
    resume_easing: str | None = None


# ### Draft state ###
@dataclass
class _TourDraft:
    """Hold the deliberately incomplete state of a tour being drawn."""

    tour_id: str
    name: str
    trigger_point: tuple[float, float, float] | None = None
    trigger_area_size: tuple[float, float] = DEFAULT_TOUR_TRIGGER_AREA_SIZE_METERS
    curve_points: tuple[tuple[float, float, float], ...] = ()
    tension: float = DEFAULT_TOUR_TENSION
    duration_seconds: float = DEFAULT_TOUR_DURATION_SECONDS
    progress_speed: float = DEFAULT_TOUR_PROGRESS_SPEED


@dataclass(frozen=True)
class _PendingPointPlacement:
    """Describe how the next accepted point must be applied."""

    action: str
    tour_id: str

    def context(self) -> dict[str, str]:
        return {"tour_id": self.tour_id}


# ### Timeline widget ###
class TourTimelineSlider(QSlider):
    """A scrubber that draws persistent markers for every tour step."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setObjectName("tour_timeline_slider")
        self.setRange(0, TIMELINE_RESOLUTION)
        self.setSingleStep(1)
        self.setPageStep(25)
        self._step_progresses: tuple[float, ...] = ()
        self._absolute_scrub_active = False

    def set_step_progresses(self, progresses: tuple[float, ...]) -> None:
        self._step_progresses = tuple(
            max(0.0, min(1.0, float(progress))) for progress in progresses
        )
        self.update()

    def _marker_x(self, progress: float) -> int:
        """Map progress to the center of the slider handle's real travel span."""

        return self._marker_x_with_geometry(progress, self._style_geometry())

    def _style_geometry(self) -> tuple[QStyleOptionSlider, int, int, int]:
        """Return the styled handle travel data shared by painting and input."""

        option = QStyleOptionSlider()
        self.initStyleOption(option)
        groove = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            option,
            QStyle.SubControl.SC_SliderGroove,
            self,
        )
        handle = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            option,
            QStyle.SubControl.SC_SliderHandle,
            self,
        )
        slider_minimum = groove.x()
        slider_maximum = groove.right() - handle.width() + 1
        travel = max(0, slider_maximum - slider_minimum)
        return option, slider_minimum, travel, handle.width()

    def _marker_x_with_geometry(
        self,
        progress: float,
        geometry: tuple[QStyleOptionSlider, int, int, int],
    ) -> int:
        """Map progress using one already-resolved style geometry snapshot."""

        option, slider_minimum, travel, handle_width = geometry
        value = round(
            self.minimum()
            + max(0.0, min(1.0, float(progress)))
            * (self.maximum() - self.minimum())
        )
        handle_offset = QStyle.sliderPositionFromValue(
            self.minimum(),
            self.maximum(),
            value,
            travel,
            option.upsideDown,
        )
        return slider_minimum + handle_offset + (handle_width // 2)

    def paintEvent(self, event: object) -> None:  # type: ignore[override]
        super().paintEvent(event)  # type: ignore[arg-type]
        if not self._step_progresses:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(QColor("#f2c94c"), 2))
        marker_top = max(1, self.height() // 2 - 10)
        marker_bottom = min(self.height() - 1, self.height() // 2 + 10)
        geometry = self._style_geometry()
        for progress in self._step_progresses:
            x = self._marker_x_with_geometry(progress, geometry)
            painter.drawLine(x, marker_top, x, marker_bottom)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        """Seek immediately and begin scrubbing from anywhere on the track."""

        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        self._absolute_scrub_active = True
        self.setSliderDown(True)
        self._seek_to_pointer(event.position().x())
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        """Continue seeking even when the initial press missed the handle."""

        if not self._absolute_scrub_active:
            super().mouseMoveEvent(event)
            return
        self._seek_to_pointer(event.position().x())
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        """Finish absolute scrubbing through the normal slider-down lifecycle."""

        if (
            event.button() != Qt.MouseButton.LeftButton
            or not self._absolute_scrub_active
        ):
            super().mouseReleaseEvent(event)
            return
        self._seek_to_pointer(event.position().x())
        self._absolute_scrub_active = False
        self.setSliderDown(False)
        event.accept()

    def _seek_to_pointer(self, pointer_x: float) -> None:
        """Map a pointer position through Qt's styled handle travel range."""

        geometry = self._style_geometry()
        snapped_progress = min(
            self._step_progresses,
            key=lambda progress: abs(
                self._marker_x_with_geometry(progress, geometry) - pointer_x
            ),
            default=None,
        )
        if (
            snapped_progress is not None
            and abs(
                self._marker_x_with_geometry(snapped_progress, geometry) - pointer_x
            )
            <= TIMELINE_MARKER_SNAP_TOLERANCE_PIXELS
        ):
            value = round(
                self.minimum()
                + snapped_progress * (self.maximum() - self.minimum())
            )
            self.setSliderPosition(value)
            return

        option, slider_minimum, resolved_travel, handle_width = geometry
        travel = max(1, resolved_travel)
        position = round(pointer_x - slider_minimum - handle_width / 2.0)
        position = max(0, min(travel, position))
        self.setSliderPosition(
            QStyle.sliderValueFromPosition(
                self.minimum(),
                self.maximum(),
                position,
                travel,
                option.upsideDown,
            )
        )


# ### Tour workspace ###
class TourWorkspace(QWidget):
    """Create, edit, and preview camera tours independently of scene plumbing."""

    data_changed = Signal(object)
    point_placement_requested = Signal(str, object)
    point_placement_cancel_requested = Signal()
    camera_target_requested = Signal(str, str)
    draft_changed = Signal(object)
    tour_selection_changed = Signal(object)
    step_selection_changed = Signal(object)
    action_selection_changed = Signal(object)
    preview_requested = Signal(object, float)
    preview_pose_requested = Signal(object, float, object, object)
    text_action_position_requested = Signal(object)
    floating_tooltip_position_requested = Signal(str, str, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("tour_workspace")
        self._tours: list[TourData] = []
        self._draft: _TourDraft | None = None
        self._pending_point: _PendingPointPlacement | None = None
        self._refreshing = False
        self._preview_widget: QWidget | None = None
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._preview_timer.setInterval(round(PREVIEW_FRAME_INTERVAL_MILLISECONDS))
        self._preview_timer.timeout.connect(self._advance_preview)
        self._preview_elapsed_timer = QElapsedTimer()
        self._preview_start_progress = 0.0
        self._preview_next_frame_index = 1
        self._preview_speed_segments: tuple[_TourSpeedSegment, ...] = ()
        self._preview_consumed_wait_action_ids: set[str] = set()
        self._preview_wait_action_id = ""
        self._preview_wait_elapsed_timer = QElapsedTimer()
        self._detect_input_action_id = ""
        self._build_ui()
        self._connect_signals()
        application = QApplication.instance()
        if application is not None:
            application.installEventFilter(self)
        self._refresh_all()
        self._set_status(self._default_status_text())

    # ### Public model API ###
    def set_tours(self, tours: tuple[TourData, ...] | list[TourData]) -> None:
        """Replace the editable collection with validated immutable tours."""

        prepared = list(tours)
        if any(not isinstance(tour, TourData) for tour in prepared):
            raise TypeError("TourWorkspace.set_tours expects TourData values.")
        tour_ids = [tour.tour_id for tour in prepared]
        if len(tour_ids) != len(set(tour_ids)):
            raise ValueError("TourWorkspace.set_tours requires unique tour IDs.")
        self._stop_preview_playback()
        self.cancel_point_placement(remove_draft=True)
        self._tours = prepared
        self._refresh_all()
        self._set_status(self._default_status_text())

    def tours(self) -> tuple[TourData, ...]:
        """Return the complete, persistence-ready tour collection."""

        return tuple(self._tours)

    def current_tour(self) -> TourData | None:
        tour_id = self._selected_tour_id()
        return self._find_tour(tour_id)

    def selected_step(self) -> TourStepData | None:
        tour = self.current_tour()
        step_id = self._selected_step_id()
        if tour is None or not step_id:
            return None
        return next((step for step in tour.steps if step.step_id == step_id), None)

    def selected_action(self) -> TourActionData | None:
        """Return the selected action for the current step."""

        return self._selected_action()

    def pending_point_action(self) -> str | None:
        """Return the scene-pick action currently awaiting a world point."""

        return self._pending_point.action if self._pending_point is not None else None

    def start_new_tour(self) -> None:
        """Begin the trigger-first point placement workflow."""

        self._stop_preview_playback()
        if self._pending_point is not None or self._draft is not None:
            self.cancel_point_placement(remove_draft=self._draft is not None)
        draft = _TourDraft(
            tour_id=uuid4().hex,
            name=self._next_tour_name(),
        )
        self._draft = draft
        self._pending_point = _PendingPointPlacement(
            TOUR_POINT_ACTION_TRIGGER,
            draft.tour_id,
        )
        self._refresh_all(select_tour_id=draft.tour_id)
        self._set_status(
            "Click and drag in the 3D scene to define the tour trigger area."
        )
        self._emit_draft_changed()
        self._emit_point_placement_request()

    def accept_world_point(self, point: object) -> bool:
        """Apply a point or trigger-area payload supplied by the shared scene."""

        pending = self._pending_point
        if pending is None:
            return False

        if pending.action == TOUR_POINT_ACTION_TRIGGER:
            prepared_area = _coerce_trigger_area(point)
            if prepared_area is None:
                return False
            return self._accept_trigger_point(pending, *prepared_area)
        prepared_point = _coerce_point(point)
        if prepared_point is None:
            return False
        if pending.action == TOUR_POINT_ACTION_CURVE:
            return self._accept_curve_point(pending, prepared_point)
        return False

    def cancel_point_placement(self, *, remove_draft: bool = False) -> None:
        """Cancel the current scene pick without discarding complete tours."""

        had_pending = self._pending_point is not None
        self._pending_point = None
        if remove_draft:
            self._draft = None
        if had_pending:
            self.point_placement_cancel_requested.emit()
        self._refresh_all()
        self._set_status(self._default_status_text())
        self._emit_draft_changed()

    def select_tour(
        self,
        tour_id: str | None,
        *,
        select_action: bool = True,
    ) -> bool:
        """Select a finalized tour, optionally without an action edit target."""

        if tour_id is not None and self._find_tour(str(tour_id)) is None:
            return False
        requested_id = "" if tour_id is None else str(tour_id)
        if requested_id == self._selected_tour_id() and select_action:
            return True

        self._stop_preview_playback()
        selected_step_id = self._selected_step_id()
        with QSignalBlocker(self.tour_list):
            if tour_id is None:
                self.tour_list.clearSelection()
                self.tour_list.setCurrentItem(None)
            else:
                _select_list_item(self.tour_list, requested_id, select_first=False)
        self._refresh_tour_controls()
        self._refresh_steps(
            select_id=selected_step_id,
            select_first_action=select_action,
        )
        if not select_action:
            with QSignalBlocker(self.action_list):
                self.action_list.clearSelection()
                self.action_list.setCurrentItem(None)
            self._refresh_action_controls()
        tour = self.current_tour()
        step = self.selected_step()
        progress = self.preview_progress()
        if step is not None:
            progress = self._set_preview_progress_without_request(step.progress)
        self.tour_selection_changed.emit(tour)
        self.step_selection_changed.emit(step)
        self.action_selection_changed.emit(self._selected_action())
        if tour is not None:
            self.preview_requested.emit(tour, progress)
        return True

    def select_action(
        self,
        tour_id: str,
        step_id: str,
        action_id: str,
    ) -> bool:
        """Select an action and synchronize its owning tour and step."""

        tour = self._find_tour(str(tour_id))
        step = _find_step(tour, str(step_id))
        action = _find_action(step, str(action_id))
        if tour is None or step is None or action is None:
            return False
        self._stop_preview_playback()
        self._refresh_all(
            select_tour_id=tour.tour_id,
            select_step_id=step.step_id,
            select_action_id=action.component_id,
        )
        self._set_preview_progress_without_request(step.progress)
        self.tour_selection_changed.emit(tour)
        self.step_selection_changed.emit(step)
        self.action_selection_changed.emit(action)
        self.preview_requested.emit(tour, step.progress)
        return True

    def select_step(self, tour_id: str, step_id: str) -> bool:
        """Select one tour step without selecting one of its actions."""

        tour = self._find_tour(str(tour_id))
        step = _find_step(tour, str(step_id))
        if tour is None or step is None:
            return False
        self._stop_preview_playback()
        self._refresh_all(
            select_tour_id=tour.tour_id,
            select_step_id=step.step_id,
            select_first_action=False,
        )
        with QSignalBlocker(self.action_list):
            self.action_list.clearSelection()
            self.action_list.setCurrentItem(None)
        self._refresh_action_controls()
        self._set_preview_progress_without_request(step.progress)
        self.tour_selection_changed.emit(tour)
        self.step_selection_changed.emit(step)
        self.action_selection_changed.emit(None)
        self.preview_requested.emit(tour, step.progress)
        return True

    def clear_action_selection(self) -> None:
        """Clear the action editing context while retaining its step's list."""

        with QSignalBlocker(self.action_list):
            self.action_list.clearSelection()
            self.action_list.setCurrentItem(None)
        self._refresh_action_controls()
        self.action_selection_changed.emit(None)

    def add_text_action_at_position(self, position: object) -> bool:
        """Add a default 3D-text action at a preview-derived world position."""

        prepared_position = _coerce_point(position)
        tour = self.current_tour()
        step = self.selected_step()
        if prepared_position is None or tour is None or step is None:
            return False
        action = TourText3DData(
            component_id=uuid4().hex,
            text="Lorem ipsum",
            position=prepared_position,
            size_points=DEFAULT_TEXT_SIZE_POINTS,
            color=DEFAULT_TEXT_COLOR,
            fade_delay_ms=DEFAULT_TEXT_FADE_DELAY_MS,
            fade_duration_ms=DEFAULT_TEXT_FADE_DURATION_MS,
            rotation_degrees=DEFAULT_TEXT_ROTATION_DEGREES,
        )
        updated_step = replace(step, actions=(*step.actions, action))
        self._replace_tour(
            _tour_with_replaced_step(tour, updated_step),
            select_step_id=step.step_id,
            select_action_id=action.component_id,
        )
        self._set_status("3D text added at the preview center.")
        return True

    def set_tour_trigger_point(self, tour_id: str, point: object) -> bool:
        """Move a finalized tour's trigger point without changing its ID."""

        tour = self._find_tour(str(tour_id))
        prepared_point = _coerce_point(point)
        if tour is None or prepared_point is None:
            return False
        if tour.trigger_point == prepared_point:
            return True
        self._replace_tour_preserving_selection(
            replace(tour, trigger_point=prepared_point)
        )
        return True

    def set_tour_curve_point(
        self,
        tour_id: str,
        point_index: int,
        point: object,
    ) -> bool:
        """Move one curve control point when the resulting curve stays valid."""

        tour = self._find_tour(str(tour_id))
        prepared_point = _coerce_point(point)
        if (
            tour is None
            or prepared_point is None
            or isinstance(point_index, bool)
            or not isinstance(point_index, int)
            or not 0 <= point_index < len(tour.curve.points)
        ):
            return False
        if tour.curve.points[point_index] == prepared_point:
            return True
        points = list(tour.curve.points)
        points[point_index] = prepared_point
        try:
            updated_curve = replace(tour.curve, points=tuple(points))
        except (TypeError, ValueError):
            return False
        self._replace_tour_preserving_selection(replace(tour, curve=updated_curve))
        return True

    def set_tour_camera_target(
        self,
        tour_id: str,
        step_id: str,
        target: object,
    ) -> bool:
        """Move one step's camera target while retaining all model identifiers."""

        tour = self._find_tour(str(tour_id))
        step = _find_step(tour, str(step_id))
        prepared_target = _coerce_point(target)
        if tour is None or step is None or prepared_target is None:
            return False
        if step.camera_target == prepared_target:
            return True
        updated_step = replace(step, camera_target=prepared_target)
        self._replace_tour_preserving_selection(
            _tour_with_replaced_step(tour, updated_step)
        )
        self._set_status("Camera target updated.")
        return True

    def set_text_action_position(
        self,
        tour_id: str,
        step_id: str,
        action_id: str,
        position: object,
    ) -> bool:
        """Move one text action while retaining all model identifiers."""

        tour = self._find_tour(str(tour_id))
        step = _find_step(tour, str(step_id))
        action = _find_action(step, str(action_id))
        prepared_position = _coerce_point(position)
        if (
            tour is None
            or step is None
            or not isinstance(action, TourText3DData)
            or prepared_position is None
        ):
            return False
        if action.position == prepared_position:
            return True
        updated_action = replace(action, position=prepared_position)
        updated_step = _step_with_replaced_action(step, updated_action)
        self._replace_tour_preserving_selection(
            _tour_with_replaced_step(tour, updated_step)
        )
        return True

    def set_text_action_rotation(
        self,
        tour_id: str,
        step_id: str,
        action_id: str,
        rotation_degrees: object,
    ) -> bool:
        """Rotate one text action using world-space static XYZ Euler degrees."""

        tour = self._find_tour(str(tour_id))
        step = _find_step(tour, str(step_id))
        action = _find_action(step, str(action_id))
        prepared_rotation = _coerce_point(rotation_degrees)
        if (
            tour is None
            or step is None
            or not isinstance(action, TourText3DData)
            or prepared_rotation is None
        ):
            return False
        if action.rotation_degrees == prepared_rotation:
            return True
        updated_action = replace(action, rotation_degrees=prepared_rotation)
        updated_step = _step_with_replaced_action(step, updated_action)
        self._replace_tour_preserving_selection(
            _tour_with_replaced_step(tour, updated_step)
        )
        return True

    def set_floating_tooltip_action_anchor_point(
        self,
        tour_id: str,
        step_id: str,
        action_id: str,
        anchor_point: object,
    ) -> bool:
        """Move one floating tooltip anchor while retaining its identifiers."""

        tour = self._find_tour(str(tour_id))
        step = _find_step(tour, str(step_id))
        action = _find_action(step, str(action_id))
        prepared_point = _coerce_point(anchor_point)
        if (
            tour is None
            or step is None
            or not isinstance(action, TourFloatingTooltipData)
            or prepared_point is None
        ):
            return False
        if action.anchor_point == prepared_point:
            return True
        updated_action = replace(action, anchor_point=prepared_point)
        updated_step = _step_with_replaced_action(step, updated_action)
        self._replace_tour_preserving_selection(
            _tour_with_replaced_step(tour, updated_step)
        )
        return True

    def set_preview_progress(self, progress: float) -> None:
        """Move the preview playhead using normalized curve progress."""

        prepared = max(0.0, min(1.0, float(progress)))
        self.timeline_slider.setValue(round(prepared * TIMELINE_RESOLUTION))

    def _set_preview_progress_without_request(self, progress: float) -> float:
        """Move the playhead while one caller coordinates the preview signal."""

        prepared = max(0.0, min(1.0, float(progress)))
        with QSignalBlocker(self.timeline_slider):
            self.timeline_slider.setValue(round(prepared * TIMELINE_RESOLUTION))
        self.timeline_position_label.setText(f"{prepared * 100.0:.1f} %")
        return prepared

    def preview_progress(self) -> float:
        """Return the timeline playhead as normalized curve progress."""

        return self.timeline_slider.value() / TIMELINE_RESOLUTION

    def shutdown(self) -> None:
        """Stop playback before the workspace and its preview are destroyed."""

        self._stop_preview_playback()
        application = QApplication.instance()
        if application is not None:
            application.removeEventFilter(self)

    def eventFilter(self, watched: object, event: QEvent) -> bool:  # type: ignore[override]
        """Capture configured authoring input and resume paused tour previews."""

        input_token = _input_token_from_event(event)
        if input_token is None:
            return super().eventFilter(watched, event)
        if self._detect_input_action_id:
            self._complete_input_detection(input_token)
            return True
        if self._preview_wait_action_id:
            wait_action = _find_action_by_id(
                self.current_tour(),
                self._preview_wait_action_id,
            )
            if isinstance(wait_action, TourWaitForKeyPressData) and _input_matches(
                wait_action.input_token,
                input_token,
            ):
                self._resume_waiting_preview(wait_action)
                return True
        return super().eventFilter(watched, event)

    def hideEvent(self, event: object) -> None:  # type: ignore[override]
        """Stop hidden previews from continuing to render in the background."""

        self._stop_preview_playback()
        super().hideEvent(event)  # type: ignore[arg-type]

    def finish_curve(self) -> bool:
        """Commit the active draft once it contains a usable open curve."""

        draft = self._draft
        if (
            draft is None
            or draft.trigger_point is None
            or len(draft.curve_points) < MINIMUM_CURVE_POINT_COUNT
        ):
            return False
        curve = TourCurveData(
            points=draft.curve_points,
            tension=draft.tension,
            closed=False,
        )
        first_step = TourStepData(
            step_id=uuid4().hex,
            progress=0.0,
            camera_target=_default_camera_target(curve, 0.0),
        )
        last_step = TourStepData(
            step_id=uuid4().hex,
            progress=1.0,
            camera_target=_default_camera_target(curve, 1.0),
        )
        tour = TourData(
            tour_id=draft.tour_id,
            name=draft.name,
            trigger_point=draft.trigger_point,
            curve=curve,
            trigger_area_size=draft.trigger_area_size,
            steps=(first_step, last_step),
            duration_seconds=draft.duration_seconds,
            progress_speed=draft.progress_speed,
        )
        self._pending_point = None
        self._draft = None
        self._tours.append(tour)
        self.point_placement_cancel_requested.emit()
        self._refresh_all(
            select_tour_id=tour.tour_id,
            select_step_id=first_step.step_id,
        )
        self._emit_draft_changed()
        self._emit_data_changed()
        self._set_status("Tour curve created. Edit its timeline steps on the right.")
        return True

    def set_status_message(self, text: str) -> None:
        """Display a scene-integration status without changing tour state."""

        self._set_status(str(text))

    # ### Public preview host API ###
    def set_preview_widget(self, widget: QWidget | None) -> None:
        """Show an integration-provided 3D viewer or the explanatory fallback."""

        if widget is self._preview_widget:
            return
        if self._preview_widget is not None:
            self.preview_stack.removeWidget(self._preview_widget)
            self._preview_widget.setParent(None)
        self._preview_widget = widget
        if widget is None:
            self.preview_stack.setCurrentWidget(self.preview_fallback)
            return
        self.preview_stack.addWidget(widget)
        self.preview_stack.setCurrentWidget(widget)

    def clear_preview_widget(self) -> QWidget | None:
        """Detach and return the integration-provided preview widget."""

        widget = self._preview_widget
        self.set_preview_widget(None)
        return widget

    # ### UI construction ###
    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        self.content_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.content_splitter.setObjectName("tour_content_splitter")
        self.content_splitter.addWidget(self._build_tour_panel())
        self.content_splitter.addWidget(self._build_preview_panel())
        self.content_splitter.addWidget(self._build_step_panel())
        self.content_splitter.setStretchFactor(0, 0)
        self.content_splitter.setStretchFactor(1, 3)
        self.content_splitter.setStretchFactor(2, 1)
        self.content_splitter.setSizes([260, 720, 340])
        root_layout.addWidget(self.content_splitter, 1)
        root_layout.addWidget(self._build_timeline_panel())

    def _build_tour_panel(self) -> QWidget:
        panel = QGroupBox("Tours")
        panel.setObjectName("tour_management_group")
        layout = QVBoxLayout(panel)

        self.tour_list = QListWidget()
        self.tour_list.setObjectName("tour_list")
        self.tour_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        layout.addWidget(self.tour_list, 1)

        button_row = QHBoxLayout()
        self.add_tour_button = QPushButton("Add tour")
        self.add_tour_button.setObjectName("add_tour_button")
        self.delete_tour_button = QPushButton("Delete tour")
        self.delete_tour_button.setObjectName("delete_tour_button")
        button_row.addWidget(self.add_tour_button)
        button_row.addWidget(self.delete_tour_button)
        layout.addLayout(button_row)

        properties_group = QGroupBox("Tour properties")
        properties_group.setObjectName("tour_properties_group")
        properties_form = QFormLayout(properties_group)
        self.tour_name_edit = QLineEdit()
        self.tour_name_edit.setObjectName("tour_name_edit")
        properties_form.addRow("Name", self.tour_name_edit)
        self.curve_tension_spin = QDoubleSpinBox()
        self.curve_tension_spin.setObjectName("tour_curve_tension_spin")
        self.curve_tension_spin.setRange(0.0, 1.0)
        self.curve_tension_spin.setDecimals(2)
        self.curve_tension_spin.setSingleStep(0.05)
        properties_form.addRow("Curve tension", self.curve_tension_spin)
        self.progress_speed_spin = QDoubleSpinBox()
        self.progress_speed_spin.setObjectName("tour_progress_speed_spin")
        self.progress_speed_spin.setRange(0.01, 1_000.0)
        self.progress_speed_spin.setDecimals(2)
        self.progress_speed_spin.setSingleStep(0.1)
        self.progress_speed_spin.setSuffix(" x")
        properties_form.addRow("Progress speed", self.progress_speed_spin)
        layout.addWidget(properties_group)

        self.finish_curve_button = QPushButton("Finish curve")
        self.finish_curve_button.setObjectName("finish_tour_curve_button")
        self.cancel_placement_button = QPushButton("Cancel")
        self.cancel_placement_button.setObjectName("cancel_tour_placement_button")
        creation_row = QHBoxLayout()
        creation_row.addWidget(self.finish_curve_button)
        creation_row.addWidget(self.cancel_placement_button)
        layout.addLayout(creation_row)

        self.status_label = QLabel()
        self.status_label.setObjectName("tour_status_label")
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(48)
        layout.addWidget(self.status_label)
        return panel

    def _build_preview_panel(self) -> QWidget:
        panel = QGroupBox("3D tour preview")
        panel.setObjectName("tour_preview_group")
        layout = QVBoxLayout(panel)
        self.preview_container = QFrame()
        self.preview_container.setObjectName("tour_preview_container")
        self.preview_container.setFrameShape(QFrame.Shape.StyledPanel)
        self.preview_stack = QStackedLayout(self.preview_container)
        self.preview_stack.setContentsMargins(0, 0, 0, 0)

        self.preview_fallback = QLabel(
            "The 3D tour preview appears here once the scene viewer is ready.\n\n"
            "Use the timeline below to scrub the camera along the open curve."
        )
        self.preview_fallback.setObjectName("tour_preview_fallback")
        self.preview_fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_fallback.setWordWrap(True)
        self.preview_fallback.setStyleSheet(
            "QLabel { background: #17191c; color: #bfc5cc; padding: 24px; }"
        )
        self.preview_stack.addWidget(self.preview_fallback)
        layout.addWidget(self.preview_container, 1)
        return panel

    def _build_step_panel(self) -> QWidget:
        panel = QGroupBox("Steps and actions")
        panel.setObjectName("tour_steps_group")
        layout = QVBoxLayout(panel)

        self.step_list = QListWidget()
        self.step_list.setObjectName("tour_step_list")
        self.step_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.step_list.setMaximumHeight(STEP_LIST_MAXIMUM_HEIGHT)
        layout.addWidget(self.step_list)

        step_buttons = QHBoxLayout()
        self.add_step_button = QPushButton("Add step")
        self.add_step_button.setObjectName("add_tour_step_button")
        self.remove_step_button = QPushButton("Remove step")
        self.remove_step_button.setObjectName("remove_tour_step_button")
        step_buttons.addWidget(self.add_step_button)
        step_buttons.addWidget(self.remove_step_button)
        layout.addLayout(step_buttons)

        step_form = QFormLayout()
        self.step_progress_spin = QDoubleSpinBox()
        self.step_progress_spin.setObjectName("tour_step_progress_spin")
        self.step_progress_spin.setRange(0.0, 100.0)
        self.step_progress_spin.setDecimals(1)
        self.step_progress_spin.setSuffix(" %")
        self.step_progress_spin.setSingleStep(1.0)
        step_form.addRow("Curve position", self.step_progress_spin)
        self.camera_target_label = QLabel("Not set")
        self.camera_target_label.setObjectName("tour_camera_target_label")
        step_form.addRow("Camera target", self.camera_target_label)
        layout.addLayout(step_form)

        self.set_camera_target_button = QPushButton("Set camera target")
        self.set_camera_target_button.setObjectName("set_tour_camera_target_button")
        layout.addWidget(self.set_camera_target_button)

        action_list_group = QGroupBox("Action list")
        action_list_group.setObjectName("tour_action_list_group")
        action_list_layout = QVBoxLayout(action_list_group)
        self.action_list = QListWidget()
        self.action_list.setObjectName("tour_action_list")
        self.action_list.setMaximumHeight(100)
        action_list_layout.addWidget(self.action_list)
        self.remove_action_button = QPushButton("Remove selected action")
        self.remove_action_button.setObjectName("remove_tour_action_button")
        action_list_layout.addWidget(self.remove_action_button)
        layout.addWidget(action_list_group)

        actions_group = QGroupBox("Actions")
        actions_group.setObjectName("tour_actions_group")
        actions_layout = QVBoxLayout(actions_group)
        self.add_text_button = QPushButton("Add 3D text")
        self.add_text_button.setObjectName("add_tour_text_button")
        self.add_floating_tooltip_button = QPushButton("Add floating tooltip")
        self.add_floating_tooltip_button.setObjectName(
            "add_tour_floating_tooltip_button"
        )
        self.add_wait_button = QPushButton("Wait for key press")
        self.add_wait_button.setObjectName("add_tour_wait_button")
        self.add_idle_camera_button = QPushButton("Add idle camera animation")
        self.add_idle_camera_button.setObjectName("add_tour_idle_camera_button")
        self.add_speed_override_button = QPushButton("Speed override")
        self.add_speed_override_button.setObjectName("add_tour_speed_override_button")
        actions_layout.addWidget(self.add_text_button)
        actions_layout.addWidget(self.add_floating_tooltip_button)
        actions_layout.addWidget(self.add_wait_button)
        actions_layout.addWidget(self.add_idle_camera_button)
        actions_layout.addWidget(self.add_speed_override_button)
        layout.addWidget(actions_group)

        self.action_properties_group = QGroupBox("Action properties")
        self.action_properties_group.setObjectName("tour_action_properties_group")
        action_properties_layout = QVBoxLayout(self.action_properties_group)
        self.action_properties_stack = QStackedLayout()
        action_properties_layout.addLayout(self.action_properties_stack)

        self.no_action_properties_page = QLabel("Select an action to edit it.")
        self.no_action_properties_page.setWordWrap(True)
        self.action_properties_stack.addWidget(self.no_action_properties_page)

        self.text_action_properties_page = QWidget()
        action_form = QFormLayout(self.text_action_properties_page)
        self.action_text_edit = QLineEdit()
        self.action_text_edit.setObjectName("tour_action_text_edit")
        action_form.addRow("Text", self.action_text_edit)
        self.action_size_spin = QDoubleSpinBox()
        self.action_size_spin.setObjectName("tour_action_size_points_spin")
        self.action_size_spin.setRange(0.1, 1_000.0)
        self.action_size_spin.setDecimals(1)
        self.action_size_spin.setSuffix(" pt")
        self.action_size_spin.setSingleStep(1.0)
        action_form.addRow("Size", self.action_size_spin)
        self.action_color_edit = QLineEdit()
        self.action_color_edit.setObjectName("tour_action_color_edit")
        self.action_color_button = QPushButton("Choose...")
        self.action_color_button.setObjectName("tour_action_color_button")
        color_row = QHBoxLayout()
        color_row.addWidget(self.action_color_edit, 1)
        color_row.addWidget(self.action_color_button)
        action_form.addRow("Color", color_row)
        position_row = QHBoxLayout()
        self.action_position_spins: list[QDoubleSpinBox] = []
        for axis_name in ("X", "Y", "Z"):
            position_row.addWidget(QLabel(axis_name))
            spin = QDoubleSpinBox()
            spin.setObjectName(f"tour_action_position_{axis_name.lower()}_spin")
            spin.setRange(-1_000_000.0, 1_000_000.0)
            spin.setDecimals(3)
            spin.setSingleStep(0.1)
            position_row.addWidget(spin, 1)
            self.action_position_spins.append(spin)
        action_form.addRow("Position", position_row)
        rotation_row = QHBoxLayout()
        self.action_rotation_spins: list[QDoubleSpinBox] = []
        for axis_name in ("X", "Y", "Z"):
            rotation_row.addWidget(QLabel(axis_name))
            spin = QDoubleSpinBox()
            spin.setObjectName(f"tour_action_rotation_{axis_name.lower()}_spin")
            spin.setRange(-360_000.0, 360_000.0)
            spin.setDecimals(2)
            spin.setSingleStep(1.0)
            spin.setSuffix("°")
            rotation_row.addWidget(spin, 1)
            self.action_rotation_spins.append(spin)
        action_form.addRow("Rotation", rotation_row)
        self.action_fade_delay_spin = QDoubleSpinBox()
        self.action_fade_delay_spin.setObjectName("tour_action_fade_delay_spin")
        self.action_fade_delay_spin.setRange(0.0, 86_400_000.0)
        self.action_fade_delay_spin.setDecimals(0)
        self.action_fade_delay_spin.setSingleStep(100.0)
        self.action_fade_delay_spin.setSuffix(" ms")
        action_form.addRow("Fade delay", self.action_fade_delay_spin)
        self.action_fade_duration_spin = QDoubleSpinBox()
        self.action_fade_duration_spin.setObjectName("tour_action_fade_duration_spin")
        self.action_fade_duration_spin.setRange(0.0, 86_400_000.0)
        self.action_fade_duration_spin.setDecimals(0)
        self.action_fade_duration_spin.setSingleStep(100.0)
        self.action_fade_duration_spin.setSuffix(" ms")
        action_form.addRow("Fade duration", self.action_fade_duration_spin)
        self.action_properties_stack.addWidget(self.text_action_properties_page)

        self.floating_tooltip_properties_page = QWidget()
        tooltip_form = QFormLayout(self.floating_tooltip_properties_page)
        self.tooltip_position_combo = QComboBox()
        self.tooltip_position_combo.setObjectName("tour_tooltip_position_combo")
        self.tooltip_position_combo.addItem("Left of screen", "left")
        self.tooltip_position_combo.addItem("Right of screen", "right")
        self.tooltip_position_combo.addItem("Opposite side", "opposite")
        tooltip_form.addRow("Tooltip position", self.tooltip_position_combo)
        self.tooltip_html_edit = QPlainTextEdit()
        self.tooltip_html_edit.setObjectName("tour_tooltip_html_edit")
        self.tooltip_html_edit.setPlaceholderText("HTML body")
        self.tooltip_html_edit.setMinimumHeight(100)
        tooltip_form.addRow("HTML body", self.tooltip_html_edit)
        self.tooltip_style_edit = QPlainTextEdit()
        self.tooltip_style_edit.setObjectName("tour_tooltip_style_edit")
        self.tooltip_style_edit.setPlaceholderText("CSS declarations")
        self.tooltip_style_edit.setMinimumHeight(80)
        tooltip_form.addRow("Style", self.tooltip_style_edit)
        self.action_properties_stack.addWidget(
            self.floating_tooltip_properties_page
        )

        self.wait_action_properties_page = QWidget()
        wait_form = QFormLayout(self.wait_action_properties_page)
        self.wait_input_combo = QComboBox()
        self.wait_input_combo.setObjectName("tour_wait_input_combo")
        for label, token in INPUT_TOKEN_OPTIONS:
            self.wait_input_combo.addItem(label, token)
        wait_form.addRow("Input", self.wait_input_combo)
        self.detect_input_button = QPushButton("Detect input")
        self.detect_input_button.setObjectName("tour_detect_wait_input_button")
        wait_form.addRow(self.detect_input_button)
        self.wait_easing_combo = QComboBox()
        self.wait_easing_combo.setObjectName("tour_wait_easing_combo")
        for easing in TOUR_SPEED_EASING_OPTIONS:
            self.wait_easing_combo.addItem(EASING_LABELS.get(easing, easing), easing)
        wait_form.addRow("Ease-in-out function", self.wait_easing_combo)
        self.action_properties_stack.addWidget(self.wait_action_properties_page)

        self.idle_action_properties_page = QWidget()
        idle_form = QFormLayout(self.idle_action_properties_page)
        idle_pivot_row = QHBoxLayout()
        self.idle_pivot_spins: list[QDoubleSpinBox] = []
        for axis_name in ("X", "Y", "Z"):
            idle_pivot_row.addWidget(QLabel(axis_name))
            spin = QDoubleSpinBox()
            spin.setObjectName(f"tour_idle_pivot_{axis_name.lower()}_spin")
            spin.setRange(-1_000_000.0, 1_000_000.0)
            spin.setDecimals(3)
            spin.setSingleStep(0.1)
            idle_pivot_row.addWidget(spin, 1)
            self.idle_pivot_spins.append(spin)
        idle_form.addRow("Pivot point", idle_pivot_row)
        self.idle_radius_spin = QDoubleSpinBox()
        self.idle_radius_spin.setObjectName("tour_idle_radius_spin")
        self.idle_radius_spin.setRange(0.001, 100.0)
        self.idle_radius_spin.setDecimals(3)
        self.idle_radius_spin.setSingleStep(0.01)
        self.idle_radius_spin.setSuffix(" m")
        idle_form.addRow("Movement radius", self.idle_radius_spin)
        self.idle_cycle_duration_spin = QDoubleSpinBox()
        self.idle_cycle_duration_spin.setObjectName("tour_idle_cycle_duration_spin")
        self.idle_cycle_duration_spin.setRange(0.1, 3_600.0)
        self.idle_cycle_duration_spin.setDecimals(1)
        self.idle_cycle_duration_spin.setSingleStep(0.5)
        self.idle_cycle_duration_spin.setSuffix(" s")
        idle_form.addRow("Cycle duration", self.idle_cycle_duration_spin)
        self.action_properties_stack.addWidget(self.idle_action_properties_page)

        self.speed_action_properties_page = QWidget()
        speed_form = QFormLayout(self.speed_action_properties_page)
        self.speed_override_spin = QDoubleSpinBox()
        self.speed_override_spin.setObjectName("tour_speed_override_spin")
        self.speed_override_spin.setRange(0.01, 1_000.0)
        self.speed_override_spin.setDecimals(2)
        self.speed_override_spin.setSingleStep(0.1)
        self.speed_override_spin.setSuffix(" x")
        speed_form.addRow("Speed", self.speed_override_spin)
        self.speed_easing_combo = QComboBox()
        self.speed_easing_combo.setObjectName("tour_speed_easing_combo")
        for easing in TOUR_SPEED_EASING_OPTIONS:
            self.speed_easing_combo.addItem(EASING_LABELS.get(easing, easing), easing)
        speed_form.addRow("Ease-in-out function", self.speed_easing_combo)
        self.action_properties_stack.addWidget(self.speed_action_properties_page)

        layout.addWidget(self.action_properties_group)
        layout.addStretch(1)
        return panel

    def _build_timeline_panel(self) -> QWidget:
        panel = QGroupBox("Timeline")
        panel.setObjectName("tour_timeline_group")
        layout = QHBoxLayout(panel)
        self.play_preview_button = QPushButton("Play")
        self.play_preview_button.setObjectName("play_tour_preview_button")
        self.play_preview_button.setMinimumWidth(72)
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setObjectName("tour_duration_spin")
        self.duration_spin.setRange(0.1, 3_600.0)
        self.duration_spin.setDecimals(1)
        self.duration_spin.setSingleStep(0.5)
        self.duration_spin.setSuffix(" s")
        self.timeline_slider = TourTimelineSlider()
        self.timeline_position_label = QLabel("0.0 %")
        self.timeline_position_label.setObjectName("tour_timeline_position_label")
        self.timeline_position_label.setMinimumWidth(62)
        self.timeline_position_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.play_preview_button)
        layout.addWidget(QLabel("Duration"))
        layout.addWidget(self.duration_spin)
        layout.addWidget(self.timeline_slider, 1)
        layout.addWidget(self.timeline_position_label)
        return panel

    # ### Signal wiring ###
    def _connect_signals(self) -> None:
        self.add_tour_button.clicked.connect(self.start_new_tour)
        self.delete_tour_button.clicked.connect(self._delete_selected_tour)
        self.finish_curve_button.clicked.connect(self.finish_curve)
        self.cancel_placement_button.clicked.connect(
            lambda: self.cancel_point_placement(remove_draft=True)
        )
        self.tour_list.itemSelectionChanged.connect(self._tour_item_changed)
        self.tour_name_edit.editingFinished.connect(self._apply_tour_name)
        self.curve_tension_spin.valueChanged.connect(self._apply_curve_tension)
        self.progress_speed_spin.valueChanged.connect(self._apply_progress_speed)
        self.duration_spin.valueChanged.connect(self._apply_duration)

        self.step_list.itemSelectionChanged.connect(self._step_item_changed)
        self.add_step_button.clicked.connect(self._add_step)
        self.remove_step_button.clicked.connect(self._remove_step)
        self.step_progress_spin.valueChanged.connect(self._apply_step_progress)
        self.set_camera_target_button.clicked.connect(self._request_camera_target)

        self.action_list.itemSelectionChanged.connect(self._action_item_changed)
        self.add_text_button.clicked.connect(self._request_add_text_action)
        self.add_floating_tooltip_button.clicked.connect(
            self._add_floating_tooltip_action
        )
        self.add_wait_button.clicked.connect(self._add_wait_action)
        self.add_idle_camera_button.clicked.connect(self._add_idle_camera_action)
        self.add_speed_override_button.clicked.connect(self._add_speed_override_action)
        self.remove_action_button.clicked.connect(self._remove_action)
        self.action_text_edit.textChanged.connect(self._apply_action_text_live)
        self.action_text_edit.editingFinished.connect(self._apply_action_fields)
        self.action_size_spin.valueChanged.connect(self._apply_action_fields)
        self.action_color_edit.textChanged.connect(self._apply_action_color_live)
        self.action_color_edit.editingFinished.connect(self._apply_action_fields)
        self.action_color_button.clicked.connect(self._choose_action_color)
        for spin in self.action_position_spins:
            spin.valueChanged.connect(self._apply_action_fields)
        for spin in self.action_rotation_spins:
            spin.valueChanged.connect(self._apply_action_fields)
        self.action_fade_delay_spin.valueChanged.connect(self._apply_action_fields)
        self.action_fade_duration_spin.valueChanged.connect(self._apply_action_fields)
        self.tooltip_position_combo.currentIndexChanged.connect(
            self._apply_floating_tooltip_fields
        )
        self.tooltip_html_edit.textChanged.connect(
            self._apply_floating_tooltip_fields
        )
        self.tooltip_style_edit.textChanged.connect(
            self._apply_floating_tooltip_fields
        )
        self.wait_input_combo.currentIndexChanged.connect(self._apply_wait_input)
        self.detect_input_button.clicked.connect(self._begin_input_detection)
        self.wait_easing_combo.currentIndexChanged.connect(self._apply_wait_easing)
        for spin in self.idle_pivot_spins:
            spin.valueChanged.connect(self._apply_idle_action_fields)
        self.idle_radius_spin.valueChanged.connect(self._apply_idle_action_fields)
        self.idle_cycle_duration_spin.valueChanged.connect(
            self._apply_idle_action_fields
        )
        self.speed_override_spin.valueChanged.connect(self._apply_speed_action_fields)
        self.speed_easing_combo.currentIndexChanged.connect(
            self._apply_speed_action_fields
        )

        self.timeline_slider.valueChanged.connect(self._timeline_value_changed)
        self.timeline_slider.sliderPressed.connect(self._stop_preview_playback)
        self.play_preview_button.clicked.connect(self._toggle_preview_playback)

    # ### Tour creation workflow ###
    def _accept_trigger_point(
        self,
        pending: _PendingPointPlacement,
        point: tuple[float, float, float],
        area_size: tuple[float, float],
    ) -> bool:
        draft = self._draft
        if draft is None or draft.tour_id != pending.tour_id:
            return False
        draft.trigger_point = point
        draft.trigger_area_size = area_size
        self._pending_point = _PendingPointPlacement(
            TOUR_POINT_ACTION_CURVE,
            draft.tour_id,
        )
        self._refresh_all(select_tour_id=draft.tour_id)
        self._set_status("Place curve points in the 3D scene, then click Finish curve.")
        self._emit_draft_changed()
        self._emit_point_placement_request()
        return True

    def _accept_curve_point(
        self,
        pending: _PendingPointPlacement,
        point: tuple[float, float, float],
    ) -> bool:
        draft = self._draft
        if draft is None or draft.tour_id != pending.tour_id:
            return False
        if draft.curve_points and draft.curve_points[-1] == point:
            self._set_status("Choose a different position for the next curve point.")
            return False
        draft.curve_points = (*draft.curve_points, point)
        self._refresh_all(select_tour_id=draft.tour_id)
        count = len(draft.curve_points)
        self._set_status(
            f"Curve point {count} placed. Add another point or finish the curve."
        )
        self._emit_draft_changed()
        return True

    # ### Tour editing ###
    def _delete_selected_tour(self) -> None:
        self._stop_preview_playback()
        selected_id = self._selected_tour_id()
        if self._draft is not None and selected_id == self._draft.tour_id:
            self.cancel_point_placement(remove_draft=True)
            return
        if (
            self._pending_point is not None
            and self._pending_point.tour_id == selected_id
        ):
            self.cancel_point_placement()
        before = len(self._tours)
        self._tours = [tour for tour in self._tours if tour.tour_id != selected_id]
        if len(self._tours) == before:
            return
        self._refresh_all()
        self._emit_data_changed()

    def _apply_tour_name(self) -> None:
        if self._refreshing:
            return
        tour_id = self._selected_tour_id()
        name = self.tour_name_edit.text().strip()
        if not name:
            self._refresh_tour_controls()
            return
        if self._draft is not None and self._draft.tour_id == tour_id:
            self._draft.name = name
            self._refresh_tour_list(select_id=tour_id)
            self._emit_draft_changed()
            return
        tour = self._find_tour(tour_id)
        if tour is None or tour.name == name:
            return
        self._replace_tour(replace(tour, name=name))

    def _apply_curve_tension(self, value: float) -> None:
        if self._refreshing:
            return
        tour_id = self._selected_tour_id()
        if self._draft is not None and self._draft.tour_id == tour_id:
            self._draft.tension = float(value)
            self._emit_draft_changed()
            return
        tour = self._find_tour(tour_id)
        if tour is None or math.isclose(tour.curve.tension, float(value)):
            return
        self._replace_tour(
            replace(tour, curve=replace(tour.curve, tension=float(value)))
        )

    def _apply_duration(self, value: float) -> None:
        if self._refreshing:
            return
        tour_id = self._selected_tour_id()
        if self._draft is not None and self._draft.tour_id == tour_id:
            self._draft.duration_seconds = float(value)
            self._emit_draft_changed()
            return
        tour = self._find_tour(tour_id)
        if tour is None or math.isclose(tour.duration_seconds, float(value)):
            return
        self._replace_tour(replace(tour, duration_seconds=float(value)))

    def _apply_progress_speed(self, value: float) -> None:
        """Store the tour-wide normalized progress multiplier."""

        if self._refreshing:
            return
        tour_id = self._selected_tour_id()
        if self._draft is not None and self._draft.tour_id == tour_id:
            self._draft.progress_speed = float(value)
            self._emit_draft_changed()
            return
        tour = self._find_tour(tour_id)
        if tour is None or math.isclose(tour.progress_speed, float(value)):
            return
        self._replace_tour(replace(tour, progress_speed=float(value)))

    # ### Step editing ###
    def _add_step(self) -> None:
        tour = self.current_tour()
        if tour is None:
            return
        progress = self.timeline_slider.value() / TIMELINE_RESOLUTION
        step = TourStepData(
            step_id=uuid4().hex,
            progress=progress,
            camera_target=_default_camera_target(
                tour.curve,
                progress,
            ),
        )
        steps = tuple(sorted((*tour.steps, step), key=lambda value: value.progress))
        self._replace_tour(
            replace(tour, steps=steps),
            select_step_id=step.step_id,
        )

    def _remove_step(self) -> None:
        tour = self.current_tour()
        step = self.selected_step()
        if tour is None or step is None or len(tour.steps) <= 1:
            return
        steps = tuple(value for value in tour.steps if value.step_id != step.step_id)
        self._replace_tour(
            replace(tour, steps=steps),
            select_step_id=steps[0].step_id if steps else "",
        )

    def _apply_step_progress(self, value: float) -> None:
        if self._refreshing:
            return
        tour = self.current_tour()
        step = self.selected_step()
        if tour is None or step is None:
            return
        progress = max(0.0, min(1.0, float(value) / 100.0))
        if math.isclose(progress, step.progress):
            return
        updated = replace(step, progress=progress)
        steps = tuple(
            sorted(
                (
                    updated if item.step_id == step.step_id else item
                    for item in tour.steps
                ),
                key=lambda item: item.progress,
            )
        )
        self._replace_tour(
            replace(tour, steps=steps),
            select_step_id=step.step_id,
        )
        self.set_preview_progress(progress)

    def _request_camera_target(self) -> None:
        """Request an immediate target from the embedded preview camera."""

        tour = self.current_tour()
        step = self.selected_step()
        if tour is None or step is None:
            return
        self.camera_target_requested.emit(tour.tour_id, step.step_id)

    # ### Action editing ###
    def _add_wait_action(self) -> None:
        """Add the single input gate allowed on the selected step."""

        tour = self.current_tour()
        step = self.selected_step()
        if (
            tour is None
            or step is None
            or _find_action_of_type(step, TourWaitForKeyPressData) is not None
        ):
            return
        action = TourWaitForKeyPressData(
            component_id=uuid4().hex,
            input_token=DEFAULT_WAIT_INPUT_TOKEN,
        )
        self._append_action(tour, step, action)

    def _add_idle_camera_action(self) -> None:
        """Add a subtle idle orbit when the selected step already waits."""

        tour = self.current_tour()
        step = self.selected_step()
        if (
            tour is None
            or step is None
            or _find_action_of_type(step, TourWaitForKeyPressData) is None
            or _find_action_of_type(step, TourIdleCameraAnimationData) is not None
        ):
            return
        action = TourIdleCameraAnimationData(
            component_id=uuid4().hex,
            pivot_point=step.camera_target,
            radius_meters=DEFAULT_IDLE_CAMERA_RADIUS_METERS,
            cycle_duration_seconds=DEFAULT_IDLE_CAMERA_CYCLE_DURATION_SECONDS,
        )
        self._append_action(tour, step, action)

    def _add_speed_override_action(self) -> None:
        """Add the single eased progress-speed override for the selected step."""

        tour = self.current_tour()
        step = self.selected_step()
        if (
            tour is None
            or step is None
            or _find_action_of_type(step, TourSpeedOverrideData) is not None
        ):
            return
        action = TourSpeedOverrideData(
            component_id=uuid4().hex,
            speed=DEFAULT_SPEED_OVERRIDE,
            easing=DEFAULT_SPEED_EASING,
        )
        self._append_action(tour, step, action)

    def _add_floating_tooltip_action(self) -> None:
        """Create a tooltip and request its preview-derived anchor point."""

        tour = self.current_tour()
        step = self.selected_step()
        if tour is None or step is None:
            return
        action = TourFloatingTooltipData(
            component_id=uuid4().hex,
            anchor_point=step.camera_target,
            tooltip_position=DEFAULT_TOOLTIP_POSITION,
            html_body=DEFAULT_TOOLTIP_HTML_BODY,
            style=DEFAULT_TOOLTIP_STYLE,
        )
        self._append_action(tour, step, action)
        self._set_status("Finding the center of the 3D tour preview.")
        self.floating_tooltip_position_requested.emit(
            tour.tour_id,
            step.step_id,
            action.component_id,
        )

    def _append_action(
        self,
        tour: TourData,
        step: TourStepData,
        action: TourActionData,
    ) -> None:
        """Append and select one already-validated action."""

        updated_step = replace(step, actions=(*step.actions, action))
        self._replace_tour(
            _tour_with_replaced_step(tour, updated_step),
            select_step_id=step.step_id,
            select_action_id=action.component_id,
        )

    def _request_add_text_action(self) -> None:
        tour = self.current_tour()
        step = self.selected_step()
        if tour is None or step is None:
            return
        self._set_status("Finding the center of the 3D tour preview.")
        self.text_action_position_requested.emit(
            {"tour_id": tour.tour_id, "step_id": step.step_id}
        )

    def _remove_action(self) -> None:
        tour = self.current_tour()
        step = self.selected_step()
        action = self._selected_action()
        if tour is None or step is None or action is None:
            return
        removed_ids = {action.component_id}
        if isinstance(action, TourWaitForKeyPressData):
            removed_ids.update(
                item.component_id
                for item in step.actions
                if isinstance(item, TourIdleCameraAnimationData)
            )
        updated_step = replace(
            step,
            actions=tuple(
                item for item in step.actions if item.component_id not in removed_ids
            ),
        )
        if self._detect_input_action_id in removed_ids:
            self._detect_input_action_id = ""
        self._replace_tour(
            _tour_with_replaced_step(tour, updated_step),
            select_step_id=step.step_id,
        )

    def _apply_wait_input(self, _index: int = -1) -> None:
        action = self._selected_action()
        if self._refreshing or not isinstance(action, TourWaitForKeyPressData):
            return
        input_token = str(self.wait_input_combo.currentData() or "any")
        if input_token == action.input_token:
            return
        self._replace_selected_action_live(replace(action, input_token=input_token))

    def _apply_wait_easing(self, _index: int = -1) -> None:
        action = self._selected_action()
        if self._refreshing or not isinstance(action, TourWaitForKeyPressData):
            return
        easing = str(self.wait_easing_combo.currentData() or DEFAULT_SPEED_EASING)
        if easing == action.easing:
            return
        self._replace_selected_action_live(replace(action, easing=easing))

    def _begin_input_detection(self) -> None:
        action = self._selected_action()
        if not isinstance(action, TourWaitForKeyPressData):
            return
        self._detect_input_action_id = action.component_id
        self.detect_input_button.setText("Press a key or click...")
        self._set_status("Press a key or mouse button to use for this wait action.")

    def _complete_input_detection(self, input_token: str) -> None:
        action = self._selected_action()
        detected_id = self._detect_input_action_id
        self._detect_input_action_id = ""
        self.detect_input_button.setText("Detect input")
        if (
            not isinstance(action, TourWaitForKeyPressData)
            or action.component_id != detected_id
        ):
            return
        self._replace_selected_action_live(replace(action, input_token=input_token))
        _select_combo_data(self.wait_input_combo, input_token)
        self._set_status(f"Wait input set to {_input_token_label(input_token)}.")

    def _apply_idle_action_fields(self, _value: object = None) -> None:
        action = self._selected_action()
        if self._refreshing or not isinstance(action, TourIdleCameraAnimationData):
            return
        updated = replace(
            action,
            pivot_point=tuple(float(spin.value()) for spin in self.idle_pivot_spins),
            radius_meters=float(self.idle_radius_spin.value()),
            cycle_duration_seconds=float(self.idle_cycle_duration_spin.value()),
        )
        if updated != action:
            self._replace_selected_action_live(updated)

    def _apply_speed_action_fields(self, _value: object = None) -> None:
        action = self._selected_action()
        if self._refreshing or not isinstance(action, TourSpeedOverrideData):
            return
        updated = replace(
            action,
            speed=float(self.speed_override_spin.value()),
            easing=str(self.speed_easing_combo.currentData() or DEFAULT_SPEED_EASING),
        )
        if updated != action:
            self._replace_selected_action_live(updated)

    def _apply_floating_tooltip_fields(self, _value: object = None) -> None:
        """Publish valid floating-tooltip property changes immediately."""

        action = self._selected_action()
        if self._refreshing or not isinstance(action, TourFloatingTooltipData):
            return
        html_body = self.tooltip_html_edit.toPlainText()
        if not html_body.strip():
            return
        updated = replace(
            action,
            tooltip_position=str(
                self.tooltip_position_combo.currentData()
                or DEFAULT_TOOLTIP_POSITION
            ),
            html_body=html_body,
            style=self.tooltip_style_edit.toPlainText(),
        )
        if updated != action:
            self._replace_selected_action_live(updated)

    def _apply_action_text_live(self, value: str) -> None:
        """Publish text edits without disturbing the line-edit cursor."""

        action = self._selected_action()
        if self._refreshing or not isinstance(action, TourText3DData):
            return
        text = str(value).strip() or "Lorem ipsum"
        if action.text == text:
            return
        self._replace_selected_action_live(replace(action, text=text))

    def _apply_action_color_live(self, value: str) -> None:
        """Publish each valid color while allowing an incomplete value to be typed."""

        action = self._selected_action()
        color = QColor(str(value).strip())
        if (
            self._refreshing
            or not isinstance(action, TourText3DData)
            or not color.isValid()
        ):
            return
        normalized = color.name(QColor.NameFormat.HexRgb)
        if action.color == normalized:
            return
        self._replace_selected_action_live(replace(action, color=normalized))

    def _apply_action_fields(self, _value: object = None) -> None:
        if self._refreshing:
            return
        tour = self.current_tour()
        step = self.selected_step()
        action = self._selected_action()
        if tour is None or step is None or not isinstance(action, TourText3DData):
            return
        text = self.action_text_edit.text().strip() or "Lorem ipsum"
        color = _normalized_color(self.action_color_edit.text())
        updated = replace(
            action,
            text=text,
            size_points=float(self.action_size_spin.value()),
            color=color,
            position=tuple(float(spin.value()) for spin in self.action_position_spins),
            rotation_degrees=tuple(
                float(spin.value()) for spin in self.action_rotation_spins
            ),
            fade_delay_ms=float(self.action_fade_delay_spin.value()),
            fade_duration_ms=float(self.action_fade_duration_spin.value()),
        )
        updated_step = _step_with_replaced_action(step, updated)
        self._replace_tour(
            _tour_with_replaced_step(tour, updated_step),
            select_step_id=step.step_id,
            select_action_id=action.component_id,
        )

    def _choose_action_color(self) -> None:
        initial = QColor(_normalized_color(self.action_color_edit.text()))
        color = QColorDialog.getColor(initial, self, "3D text color")
        if not color.isValid():
            return
        self.action_color_edit.setText(color.name(QColor.NameFormat.HexRgb))
        self._apply_action_fields()

    # ### Preview playback ###
    def _toggle_preview_playback(self) -> None:
        if self._preview_timer.isActive():
            self._stop_preview_playback()
            return
        tour = self.current_tour()
        if tour is None:
            return
        if self.timeline_slider.value() >= TIMELINE_RESOLUTION:
            self.timeline_slider.setValue(0)
        self._preview_start_progress = self.preview_progress()
        self._preview_speed_segments = _tour_speed_segments(tour)
        self._preview_consumed_wait_action_ids.clear()
        self._preview_wait_action_id = ""
        self._preview_wait_elapsed_timer.invalidate()
        self._preview_next_frame_index = 1
        self._preview_elapsed_timer.start()
        self._schedule_next_preview_frame()
        self.play_preview_button.setText("Pause")

    def _advance_preview(self) -> None:
        tour = self.current_tour()
        if tour is None or not self._preview_elapsed_timer.isValid():
            self._stop_preview_playback()
            return
        if self._preview_wait_action_id:
            self._emit_waiting_preview_pose(tour)
            self._schedule_next_preview_frame()
            return
        progress = _playback_progress_after_elapsed(
            tour,
            self._preview_start_progress,
            self._preview_elapsed_timer.elapsed() / 1_000.0,
            speed_segments=self._preview_speed_segments,
        )
        pending_wait = _first_unconsumed_wait_action(
            tour,
            self._preview_start_progress,
            progress,
            self._preview_consumed_wait_action_ids,
        )
        if pending_wait is not None:
            wait_step, wait_action = pending_wait
            self.timeline_slider.setValue(
                round(wait_step.progress * TIMELINE_RESOLUTION)
            )
            self._preview_wait_action_id = wait_action.component_id
            self._preview_wait_elapsed_timer.start()
            self.play_preview_button.setText(
                f"Waiting for {_input_token_label(wait_action.input_token)}..."
            )
            self._emit_waiting_preview_pose(tour)
            self._schedule_next_preview_frame()
            return
        self.timeline_slider.setValue(round(progress * TIMELINE_RESOLUTION))
        if progress >= 1.0:
            self._stop_preview_playback()
            return
        self._schedule_next_preview_frame()

    def _schedule_next_preview_frame(self) -> None:
        """Schedule the next 60 Hz frame against elapsed wall-clock time.

        Alternating integer timer delays approximate 16.667 milliseconds,
        while the absolute elapsed-time deadline prevents callback latency from
        accumulating into playback drift.
        """

        if not self._preview_elapsed_timer.isValid():
            return
        elapsed_milliseconds = float(self._preview_elapsed_timer.elapsed())
        elapsed_frame = math.floor(
            elapsed_milliseconds / PREVIEW_FRAME_INTERVAL_MILLISECONDS
        )
        self._preview_next_frame_index = max(
            self._preview_next_frame_index,
            elapsed_frame + 1,
        )
        deadline_milliseconds = (
            self._preview_next_frame_index * PREVIEW_FRAME_INTERVAL_MILLISECONDS
        )
        delay_milliseconds = max(
            1,
            round(deadline_milliseconds - elapsed_milliseconds),
        )
        self._preview_next_frame_index += 1
        self._preview_timer.start(delay_milliseconds)

    def _stop_preview_playback(self) -> None:
        self._preview_timer.stop()
        self._preview_elapsed_timer.invalidate()
        self._preview_wait_elapsed_timer.invalidate()
        self._preview_speed_segments = ()
        self._preview_consumed_wait_action_ids.clear()
        self._preview_wait_action_id = ""
        self.play_preview_button.setText("Play")

    def _resume_waiting_preview(self, wait_action: TourWaitForKeyPressData) -> None:
        """Resume at a held marker without counting time spent waiting."""

        tour = self.current_tour()
        if tour is None or wait_action.component_id != self._preview_wait_action_id:
            return
        self._preview_consumed_wait_action_ids.add(wait_action.component_id)
        self._preview_wait_action_id = ""
        self._preview_wait_elapsed_timer.invalidate()
        self._preview_start_progress = self.preview_progress()
        self._preview_next_frame_index = 1
        self._preview_elapsed_timer.restart()
        self.play_preview_button.setText("Pause")
        self.preview_requested.emit(tour, self._preview_start_progress)
        self._schedule_next_preview_frame()

    def _emit_waiting_preview_pose(self, tour: TourData) -> None:
        """Animate an idle camera action while its wait action holds playback."""

        wait_step = _step_containing_action(tour, self._preview_wait_action_id)
        if wait_step is None:
            return
        idle_action = _find_action_of_type(
            wait_step,
            TourIdleCameraAnimationData,
        )
        if not isinstance(idle_action, TourIdleCameraAnimationData):
            return
        elapsed_seconds = (
            self._preview_wait_elapsed_timer.elapsed() / 1_000.0
            if self._preview_wait_elapsed_timer.isValid()
            else 0.0
        )
        camera_position, camera_target = _idle_camera_pose(
            tour,
            wait_step.progress,
            idle_action,
            elapsed_seconds,
        )
        self.preview_pose_requested.emit(
            tour,
            wait_step.progress,
            camera_position,
            camera_target,
        )

    def _timeline_value_changed(self, value: int) -> None:
        progress = max(0.0, min(1.0, value / TIMELINE_RESOLUTION))
        self.timeline_position_label.setText(f"{progress * 100.0:.1f} %")
        tour = self.current_tour()
        if tour is not None:
            self._synchronize_step_selection_to_timeline(tour, progress)
            self.preview_requested.emit(tour, progress)

    def _synchronize_step_selection_to_timeline(
        self,
        tour: TourData,
        progress: float,
    ) -> None:
        """Select the authored step whose interval contains the playhead.

        This deliberately avoids the ordinary step-list signal handler: that
        handler seeks the playhead to the selected step and stops playback.
        Timeline-driven selection must instead leave the playhead and timer
        untouched while refreshing the action editing context.
        """

        selected_step = self.selected_step()
        active_step = _timeline_step_at_progress(
            tour,
            progress,
            selected_step_id=(selected_step.step_id if selected_step else ""),
        )
        if active_step is None or active_step is selected_step:
            return

        selected_action_id = self._selected_action_id()
        preserved_action = _find_action(active_step, selected_action_id)
        with QSignalBlocker(self.step_list), QSignalBlocker(self.action_list):
            _select_list_item(
                self.step_list,
                active_step.step_id,
                select_first=False,
            )
            self.action_list.clearSelection()
            self.action_list.setCurrentItem(None)
        self._refresh_step_controls(
            select_action_id=(
                preserved_action.component_id if preserved_action is not None else ""
            ),
            select_first_action=False,
        )
        selected_action = self._selected_action()
        self.step_selection_changed.emit(active_step)
        self.action_selection_changed.emit(selected_action)

    # ### Selection and refresh ###
    def _tour_item_changed(self, *_args: object) -> None:
        if self._refreshing:
            return
        self._stop_preview_playback()
        tour = self.current_tour()
        self._refresh_tour_controls()
        self._refresh_steps()
        step = self.selected_step()
        progress = self.preview_progress()
        if step is not None:
            progress = self._set_preview_progress_without_request(step.progress)
        self.tour_selection_changed.emit(tour)
        self.step_selection_changed.emit(step)
        self.action_selection_changed.emit(self._selected_action())
        if tour is not None:
            self.preview_requested.emit(tour, progress)

    def _step_item_changed(self, *_args: object) -> None:
        if self._refreshing:
            return
        self._stop_preview_playback()
        step = self.selected_step()
        if step is not None:
            self._set_preview_progress_without_request(step.progress)
        self._refresh_step_controls()
        self.step_selection_changed.emit(step)
        self.action_selection_changed.emit(self._selected_action())
        tour = self.current_tour()
        if tour is not None and step is not None:
            self.preview_requested.emit(tour, step.progress)

    def _action_item_changed(self, *_args: object) -> None:
        if not self._refreshing:
            self._refresh_action_controls()
            self.action_selection_changed.emit(self._selected_action())

    def _refresh_all(
        self,
        *,
        select_tour_id: str = "",
        select_step_id: str = "",
        select_action_id: str = "",
        select_first_tour: bool = True,
        select_first_action: bool = True,
    ) -> None:
        selected_tour_id = select_tour_id or self._selected_tour_id()
        selected_step_id = select_step_id or self._selected_step_id()
        selected_action_id = select_action_id or self._selected_action_id()
        self._refreshing = True
        try:
            self._refresh_tour_list(
                select_id=selected_tour_id,
                select_first=select_first_tour,
            )
            self._refresh_tour_controls()
            self._refresh_steps(
                select_id=selected_step_id,
                select_action_id=selected_action_id,
                select_first_action=select_first_action,
            )
        finally:
            self._refreshing = False

    def _refresh_tour_list(
        self,
        *,
        select_id: str = "",
        select_first: bool = True,
    ) -> None:
        with QSignalBlocker(self.tour_list):
            self.tour_list.clear()
            if self._draft is not None:
                item = QListWidgetItem(f"[Creating] {self._draft.name}")
                item.setData(ITEM_ID_ROLE, self._draft.tour_id)
                item.setData(ITEM_DRAFT_ROLE, True)
                self.tour_list.addItem(item)
            for tour in self._tours:
                item = QListWidgetItem(tour.name)
                item.setData(ITEM_ID_ROLE, tour.tour_id)
                item.setData(ITEM_DRAFT_ROLE, False)
                self.tour_list.addItem(item)
            target_id = select_id
            if select_first and not target_id and self.tour_list.count():
                target_id = str(self.tour_list.item(0).data(ITEM_ID_ROLE))
            _select_list_item(
                self.tour_list,
                target_id,
                select_first=select_first,
            )

    def _refresh_tour_controls(self) -> None:
        selected_id = self._selected_tour_id()
        draft = (
            self._draft if self._draft and self._draft.tour_id == selected_id else None
        )
        tour = self._find_tour(selected_id)
        selected = draft or tour
        with QSignalBlocker(self.tour_name_edit):
            self.tour_name_edit.setText(selected.name if selected is not None else "")
        with QSignalBlocker(self.curve_tension_spin):
            tension = (
                selected.tension
                if isinstance(selected, _TourDraft)
                else selected.curve.tension
                if selected is not None
                else DEFAULT_TOUR_TENSION
            )
            self.curve_tension_spin.setValue(tension)
        with QSignalBlocker(self.duration_spin):
            duration_seconds = (
                selected.duration_seconds
                if selected is not None
                else DEFAULT_TOUR_DURATION_SECONDS
            )
            self.duration_spin.setValue(duration_seconds)
        with QSignalBlocker(self.progress_speed_spin):
            progress_speed = (
                selected.progress_speed
                if selected is not None
                else DEFAULT_TOUR_PROGRESS_SPEED
            )
            self.progress_speed_spin.setValue(progress_speed)
        self.tour_name_edit.setEnabled(selected is not None)
        self.curve_tension_spin.setEnabled(selected is not None)
        self.progress_speed_spin.setEnabled(selected is not None)
        self.duration_spin.setEnabled(selected is not None)
        self.delete_tour_button.setEnabled(selected is not None)
        self.add_tour_button.setEnabled(self._draft is None)
        can_finish = (
            draft is not None
            and draft.trigger_point is not None
            and len(draft.curve_points) >= MINIMUM_CURVE_POINT_COUNT
        )
        self.finish_curve_button.setEnabled(can_finish)
        self.cancel_placement_button.setEnabled(
            self._pending_point is not None or self._draft is not None
        )

    def _refresh_steps(
        self,
        *,
        select_id: str = "",
        select_action_id: str = "",
        select_first_action: bool = True,
    ) -> None:
        tour = self.current_tour()
        with QSignalBlocker(self.step_list):
            self.step_list.clear()
            if tour is not None:
                for index, step in enumerate(tour.steps, start=1):
                    item = QListWidgetItem(
                        f"Step {index} - {step.progress * 100.0:.1f}%"
                    )
                    item.setData(ITEM_ID_ROLE, step.step_id)
                    self.step_list.addItem(item)
                if not select_id and tour.steps:
                    select_id = tour.steps[0].step_id
                _select_list_item(self.step_list, select_id)
        self.timeline_slider.set_step_progresses(
            tuple(step.progress for step in tour.steps) if tour else ()
        )
        self._refresh_step_controls(
            select_action_id=select_action_id,
            select_first_action=select_first_action,
        )

    def _refresh_step_controls(
        self,
        *,
        select_action_id: str = "",
        select_first_action: bool = True,
    ) -> None:
        tour = self.current_tour()
        step = self.selected_step()
        enabled = step is not None
        with QSignalBlocker(self.step_progress_spin):
            self.step_progress_spin.setValue(step.progress * 100.0 if step else 0.0)
        self.camera_target_label.setText(
            _format_point(step.camera_target) if step is not None else "Not set"
        )
        self.add_step_button.setEnabled(tour is not None)
        self.remove_step_button.setEnabled(
            tour is not None and step is not None and len(tour.steps) > 1
        )
        self.step_progress_spin.setEnabled(enabled)
        self.set_camera_target_button.setEnabled(enabled)
        self.add_text_button.setEnabled(enabled)
        self.add_floating_tooltip_button.setEnabled(enabled)
        has_wait = _find_action_of_type(step, TourWaitForKeyPressData) is not None
        has_idle = _find_action_of_type(step, TourIdleCameraAnimationData) is not None
        has_speed = _find_action_of_type(step, TourSpeedOverrideData) is not None
        self.add_wait_button.setEnabled(enabled and not has_wait)
        self.add_idle_camera_button.setEnabled(enabled and has_wait and not has_idle)
        self.add_speed_override_button.setEnabled(enabled and not has_speed)
        self._refresh_actions(
            select_id=select_action_id,
            select_first=select_first_action,
        )

    def _refresh_actions(
        self,
        *,
        select_id: str = "",
        select_first: bool = True,
    ) -> None:
        step = self.selected_step()
        with QSignalBlocker(self.action_list):
            previous_id = select_id or self._selected_action_id()
            self.action_list.clear()
            if step is not None:
                for action in step.actions:
                    item = QListWidgetItem(_action_label(action))
                    item.setData(ITEM_ID_ROLE, action.component_id)
                    self.action_list.addItem(item)
                if select_first and not previous_id and step.actions:
                    previous_id = step.actions[0].component_id
                _select_list_item(
                    self.action_list,
                    previous_id,
                    select_first=select_first,
                )
        self._refresh_action_controls()

    def _refresh_action_controls(self) -> None:
        action = self._selected_action()
        enabled = action is not None
        if self._detect_input_action_id and (
            action is None or action.component_id != self._detect_input_action_id
        ):
            self._detect_input_action_id = ""
            self.detect_input_button.setText("Detect input")
        page = self.no_action_properties_page
        if isinstance(action, TourText3DData):
            page = self.text_action_properties_page
        elif isinstance(action, TourFloatingTooltipData):
            page = self.floating_tooltip_properties_page
        elif isinstance(action, TourWaitForKeyPressData):
            page = self.wait_action_properties_page
        elif isinstance(action, TourIdleCameraAnimationData):
            page = self.idle_action_properties_page
        elif isinstance(action, TourSpeedOverrideData):
            page = self.speed_action_properties_page
        self.action_properties_stack.setCurrentWidget(page)

        with QSignalBlocker(self.action_text_edit):
            self.action_text_edit.setText(
                action.text if isinstance(action, TourText3DData) else ""
            )
        with QSignalBlocker(self.action_size_spin):
            self.action_size_spin.setValue(
                action.size_points
                if isinstance(action, TourText3DData)
                else DEFAULT_TEXT_SIZE_POINTS
            )
        with QSignalBlocker(self.action_color_edit):
            self.action_color_edit.setText(
                action.color
                if isinstance(action, TourText3DData)
                else DEFAULT_TEXT_COLOR
            )
        for axis, spin in enumerate(self.action_position_spins):
            with QSignalBlocker(spin):
                spin.setValue(
                    action.position[axis]
                    if isinstance(action, TourText3DData)
                    else 0.0
                )
        for axis, spin in enumerate(self.action_rotation_spins):
            with QSignalBlocker(spin):
                spin.setValue(
                    action.rotation_degrees[axis]
                    if isinstance(action, TourText3DData)
                    else DEFAULT_TEXT_ROTATION_DEGREES[axis]
                )
        with QSignalBlocker(self.action_fade_delay_spin):
            self.action_fade_delay_spin.setValue(
                action.fade_delay_ms
                if isinstance(action, TourText3DData)
                else DEFAULT_TEXT_FADE_DELAY_MS
            )
        with QSignalBlocker(self.action_fade_duration_spin):
            self.action_fade_duration_spin.setValue(
                action.fade_duration_ms
                if isinstance(action, TourText3DData)
                else DEFAULT_TEXT_FADE_DURATION_MS
            )
        with QSignalBlocker(self.tooltip_position_combo):
            _select_combo_data(
                self.tooltip_position_combo,
                action.tooltip_position
                if isinstance(action, TourFloatingTooltipData)
                else DEFAULT_TOOLTIP_POSITION,
            )
        with QSignalBlocker(self.tooltip_html_edit):
            self.tooltip_html_edit.setPlainText(
                action.html_body
                if isinstance(action, TourFloatingTooltipData)
                else DEFAULT_TOOLTIP_HTML_BODY
            )
        with QSignalBlocker(self.tooltip_style_edit):
            self.tooltip_style_edit.setPlainText(
                action.style
                if isinstance(action, TourFloatingTooltipData)
                else DEFAULT_TOOLTIP_STYLE
            )
        _select_combo_data(
            self.wait_input_combo,
            action.input_token
            if isinstance(action, TourWaitForKeyPressData)
            else DEFAULT_WAIT_INPUT_TOKEN,
        )
        with QSignalBlocker(self.wait_easing_combo):
            _select_combo_data(
                self.wait_easing_combo,
                action.easing
                if isinstance(action, TourWaitForKeyPressData)
                else DEFAULT_SPEED_EASING,
            )
        for axis, spin in enumerate(self.idle_pivot_spins):
            with QSignalBlocker(spin):
                spin.setValue(
                    action.pivot_point[axis]
                    if isinstance(action, TourIdleCameraAnimationData)
                    else 0.0
                )
        with QSignalBlocker(self.idle_radius_spin):
            self.idle_radius_spin.setValue(
                action.radius_meters
                if isinstance(action, TourIdleCameraAnimationData)
                else DEFAULT_IDLE_CAMERA_RADIUS_METERS
            )
        with QSignalBlocker(self.idle_cycle_duration_spin):
            self.idle_cycle_duration_spin.setValue(
                action.cycle_duration_seconds
                if isinstance(action, TourIdleCameraAnimationData)
                else DEFAULT_IDLE_CAMERA_CYCLE_DURATION_SECONDS
            )
        with QSignalBlocker(self.speed_override_spin):
            self.speed_override_spin.setValue(
                action.speed
                if isinstance(action, TourSpeedOverrideData)
                else DEFAULT_SPEED_OVERRIDE
            )
        with QSignalBlocker(self.speed_easing_combo):
            _select_combo_data(
                self.speed_easing_combo,
                action.easing
                if isinstance(action, TourSpeedOverrideData)
                else DEFAULT_SPEED_EASING,
            )
        wait_owns_easing = (
            _find_action_of_type(self.selected_step(), TourWaitForKeyPressData)
            is not None
        )
        self.wait_easing_combo.setEnabled(isinstance(action, TourWaitForKeyPressData))
        self.speed_easing_combo.setEnabled(
            isinstance(action, TourSpeedOverrideData) and not wait_owns_easing
        )
        self.speed_easing_combo.setToolTip(
            "The wait action controls the post-resume easing on this step."
            if wait_owns_easing
            else ""
        )
        self.remove_action_button.setEnabled(enabled)
        self.action_properties_group.setEnabled(enabled)

    # ### Model helpers ###
    def _replace_tour(
        self,
        updated: TourData,
        *,
        select_step_id: str = "",
        select_action_id: str = "",
    ) -> None:
        self._stop_preview_playback()
        select_first_action = bool(select_action_id or self._selected_action_id())
        for index, tour in enumerate(self._tours):
            if tour.tour_id == updated.tour_id:
                self._tours[index] = updated
                break
        else:
            return
        self._refresh_all(
            select_tour_id=updated.tour_id,
            select_step_id=select_step_id,
            select_action_id=select_action_id,
            select_first_action=select_first_action,
        )
        self._emit_data_changed()
        self.preview_requested.emit(
            updated,
            self.timeline_slider.value() / TIMELINE_RESOLUTION,
        )

    def _replace_tour_preserving_selection(self, updated: TourData) -> None:
        selected_tour_id = self._selected_tour_id()
        selected_step_id = self._selected_step_id()
        selected_action_id = self._selected_action_id()
        self._stop_preview_playback()
        for index, tour in enumerate(self._tours):
            if tour.tour_id == updated.tour_id:
                self._tours[index] = updated
                break
        else:
            return
        self._refresh_all(
            select_tour_id=selected_tour_id,
            select_step_id=selected_step_id,
            select_action_id=selected_action_id,
            select_first_tour=bool(selected_tour_id),
            select_first_action=bool(selected_action_id),
        )
        self._emit_data_changed()
        current = self.current_tour()
        if current is not None:
            self.preview_requested.emit(current, self.preview_progress())

    def _replace_selected_action_live(self, updated_action: TourActionData) -> None:
        """Replace one selected action without rebuilding actively edited controls."""

        tour = self.current_tour()
        step = self.selected_step()
        if tour is None or step is None:
            return
        updated_step = _step_with_replaced_action(step, updated_action)
        updated_tour = _tour_with_replaced_step(tour, updated_step)
        self._stop_preview_playback()
        for index, candidate in enumerate(self._tours):
            if candidate.tour_id == updated_tour.tour_id:
                self._tours[index] = updated_tour
                break
        else:
            return
        selected_item = self.action_list.currentItem()
        if selected_item is not None:
            selected_item.setText(_action_label(updated_action))
        self._emit_data_changed()
        self.preview_requested.emit(updated_tour, self.preview_progress())

    def _find_tour(self, tour_id: str) -> TourData | None:
        return next((tour for tour in self._tours if tour.tour_id == tour_id), None)

    def _selected_tour_id(self) -> str:
        selected_items = (
            self.tour_list.selectedItems() if hasattr(self, "tour_list") else []
        )
        item = selected_items[0] if len(selected_items) == 1 else None
        return str(item.data(ITEM_ID_ROLE)) if item is not None else ""

    def _selected_step_id(self) -> str:
        selected_items = (
            self.step_list.selectedItems() if hasattr(self, "step_list") else []
        )
        item = selected_items[0] if len(selected_items) == 1 else None
        return str(item.data(ITEM_ID_ROLE)) if item is not None else ""

    def _selected_action_id(self) -> str:
        selected_items = (
            self.action_list.selectedItems() if hasattr(self, "action_list") else []
        )
        item = selected_items[0] if len(selected_items) == 1 else None
        return str(item.data(ITEM_ID_ROLE)) if item is not None else ""

    def _selected_action(self) -> TourActionData | None:
        return _find_action(self.selected_step(), self._selected_action_id())

    def _next_tour_name(self) -> str:
        used = {tour.name.casefold() for tour in self._tours}
        index = 1
        while f"tour {index}" in used:
            index += 1
        return f"Tour {index}"

    def _emit_data_changed(self) -> None:
        self.data_changed.emit(self.tours())

    def _emit_draft_changed(self) -> None:
        draft = self._draft
        payload = None
        if draft is not None:
            payload = {
                "tour_id": draft.tour_id,
                "name": draft.name,
                "trigger_point": draft.trigger_point,
                "trigger_area_size": draft.trigger_area_size,
                "curve_points": draft.curve_points,
                "tension": draft.tension,
                "duration_seconds": draft.duration_seconds,
                "progress_speed": draft.progress_speed,
            }
        self.draft_changed.emit(payload)

    def _emit_point_placement_request(self) -> None:
        pending = self._pending_point
        if pending is not None:
            self.point_placement_requested.emit(
                pending.action,
                pending.context(),
            )

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _default_status_text(self) -> str:
        if not self._tours:
            return "Click Add tour, then place a trigger and an open camera curve."
        return "Select a tour to edit its steps and actions."


# ### Pure helpers ###
def _tour_speed_segments(
    tour: TourData,
) -> tuple[_TourSpeedSegment, ...]:
    """Resolve each step interval's speed target and transition easing.

    A speed action affects the interval starting at its step. Its easing moves
    from the previous interval's terminal speed to the new target. A step with
    no speed action restores the tour speed immediately.
    """

    ordered_steps = sorted(
        enumerate(tour.steps),
        key=lambda item: (item[1].progress, item[0]),
    )
    segments: list[_TourSpeedSegment] = []
    cursor = 0.0
    active_start_speed = tour.progress_speed
    active_end_speed = tour.progress_speed
    active_easing = "none"
    active_resume_easing: str | None = None
    index = 0
    while index < len(ordered_steps):
        boundary = ordered_steps[index][1].progress
        if boundary > cursor:
            segments.append(
                _TourSpeedSegment(
                    cursor,
                    boundary,
                    active_start_speed,
                    active_end_speed,
                    active_easing,
                    active_resume_easing,
                )
            )
        last_at_boundary = ordered_steps[index][1]
        index += 1
        while (
            index < len(ordered_steps)
            and ordered_steps[index][1].progress == boundary
        ):
            last_at_boundary = ordered_steps[index][1]
            index += 1
        speed_action = _find_action_of_type(
            last_at_boundary,
            TourSpeedOverrideData,
        )
        wait_action = _find_action_of_type(
            last_at_boundary,
            TourWaitForKeyPressData,
        )
        effective_easing = effective_step_easing(last_at_boundary)
        active_start_speed = active_end_speed
        if isinstance(speed_action, TourSpeedOverrideData):
            active_end_speed = speed_action.speed
        else:
            active_end_speed = tour.progress_speed
        if isinstance(wait_action, TourWaitForKeyPressData):
            active_easing = "none"
            active_resume_easing = effective_easing
        elif isinstance(speed_action, TourSpeedOverrideData):
            active_easing = effective_easing or "none"
            active_resume_easing = None
        else:
            active_easing = "none"
            active_resume_easing = None
        cursor = boundary
    if cursor < 1.0:
        segments.append(
            _TourSpeedSegment(
                cursor,
                1.0,
                active_start_speed,
                active_end_speed,
                active_easing,
                active_resume_easing,
            )
        )
    if not segments:
        segments.append(
            _TourSpeedSegment(
                0.0,
                1.0,
                tour.progress_speed,
                tour.progress_speed,
                "none",
            )
        )
    return tuple(segments)


def _playback_progress_after_elapsed(
    tour: TourData,
    start_progress: float,
    elapsed_seconds: float,
    *,
    speed_segments: tuple[_TourSpeedSegment, ...] | None = None,
) -> float:
    """Integrate speed multipliers from one playhead position without drift."""

    start = max(0.0, min(1.0, float(start_progress)))
    elapsed = float(elapsed_seconds)
    if start >= 1.0 or math.isnan(elapsed) or elapsed <= 0.0:
        return start
    if math.isinf(elapsed):
        return 1.0
    segments = (
        _tour_speed_segments(tour)
        if speed_segments is None
        else speed_segments
    )
    remaining_seconds = elapsed
    for segment in segments:
        if segment.end_progress <= start:
            continue
        active_start = max(start, segment.start_progress)
        progress_span = segment.end_progress - active_start
        if progress_span <= 0.0:
            continue
        segment_seconds = _speed_segment_duration_seconds(
            segment,
            active_start,
            segment.end_progress,
            tour.duration_seconds,
        )
        if remaining_seconds <= segment_seconds:
            return _speed_segment_progress_after_elapsed(
                segment,
                active_start,
                remaining_seconds,
                tour.duration_seconds,
            )
        remaining_seconds -= segment_seconds
    return 1.0


def _speed_segment_progress_after_elapsed(
    segment: _TourSpeedSegment,
    start_progress: float,
    elapsed_seconds: float,
    tour_duration_seconds: float,
) -> float:
    """Invert one segment's integrated duration with stable bisection."""

    if elapsed_seconds <= 0.0:
        return start_progress
    low = start_progress
    high = segment.end_progress
    for _iteration in range(SPEED_INVERSION_ITERATIONS):
        midpoint = (low + high) * 0.5
        duration = _speed_segment_duration_seconds(
            segment,
            start_progress,
            midpoint,
            tour_duration_seconds,
        )
        if duration < elapsed_seconds:
            low = midpoint
        else:
            high = midpoint
    return min(1.0, (low + high) * 0.5)


def _speed_segment_duration_seconds(
    segment: _TourSpeedSegment,
    start_progress: float,
    end_progress: float,
    tour_duration_seconds: float,
) -> float:
    """Integrate reciprocal speed over part of a normalized progress segment."""

    start = max(segment.start_progress, min(segment.end_progress, start_progress))
    end = max(start, min(segment.end_progress, end_progress))
    if end <= start:
        return 0.0
    if segment.resume_easing is not None:
        segment_span = segment.end_progress - segment.start_progress
        start_fraction = (start - segment.start_progress) / segment_span
        end_fraction = (end - segment.start_progress) / segment_span
        start_time_fraction = _inverse_eased_fraction(
            segment.resume_easing,
            start_fraction,
        )
        end_time_fraction = _inverse_eased_fraction(
            segment.resume_easing,
            end_fraction,
        )
        nominal_seconds = (
            segment_span * tour_duration_seconds / segment.end_speed
        )
        return (end_time_fraction - start_time_fraction) * nominal_seconds
    if (
        segment.easing == "none"
        or math.isclose(segment.start_speed, segment.end_speed)
    ):
        return (end - start) * tour_duration_seconds / segment.end_speed

    span = segment.end_progress - segment.start_progress
    start_fraction = (start - segment.start_progress) / span
    end_fraction = (end - segment.start_progress) / span
    subdivisions = SPEED_INTEGRATION_SUBDIVISIONS
    fraction_step = (end_fraction - start_fraction) / subdivisions
    reciprocal_sum = 0.0
    for index in range(subdivisions + 1):
        fraction = start_fraction + index * fraction_step
        weight = 1.0 if index in (0, subdivisions) else 4.0 if index % 2 else 2.0
        reciprocal_sum += weight / _speed_at_fraction(segment, fraction)
    reciprocal_integral = reciprocal_sum * fraction_step / 3.0
    return reciprocal_integral * span * tour_duration_seconds


def _speed_at_fraction(segment: _TourSpeedSegment, fraction: float) -> float:
    eased = _eased_fraction(segment.easing, fraction)
    return segment.start_speed + (segment.end_speed - segment.start_speed) * eased


def _eased_fraction(easing: str, fraction: float) -> float:
    value = max(0.0, min(1.0, float(fraction)))
    if easing == "none":
        return 1.0
    if easing == "linear":
        return value
    if easing == "smoothstep":
        return value * value * (3.0 - 2.0 * value)
    if easing == "smootherstep":
        return value**3 * (value * (value * 6.0 - 15.0) + 10.0)
    if easing == "ease_in_quad":
        return value * value
    if easing == "ease_out_quad":
        return 1.0 - (1.0 - value) ** 2
    if easing == "ease_in_out_sine":
        return -(math.cos(math.pi * value) - 1.0) * 0.5
    if easing == "ease_in_out_quad":
        return (
            2.0 * value * value
            if value < 0.5
            else 1.0 - (-2.0 * value + 2.0) ** 2 * 0.5
        )
    if easing == "ease_in_out_cubic":
        return (
            4.0 * value**3
            if value < 0.5
            else 1.0 - (-2.0 * value + 2.0) ** 3 * 0.5
        )
    return value


def _inverse_eased_fraction(easing: str, fraction: float) -> float:
    """Invert one monotonic easing curve for time-to-progress playback."""

    target = max(0.0, min(1.0, float(fraction)))
    if easing in {"none", "linear"} or target in {0.0, 1.0}:
        return target
    low = 0.0
    high = 1.0
    for _iteration in range(SPEED_INVERSION_ITERATIONS):
        midpoint = (low + high) * 0.5
        if _eased_fraction(easing, midpoint) < target:
            low = midpoint
        else:
            high = midpoint
    return (low + high) * 0.5


def _coerce_point(value: object) -> tuple[float, float, float] | None:
    if isinstance(value, (str, bytes, bytearray)):
        return None
    try:
        raw_values = tuple(value)  # type: ignore[arg-type]
    except TypeError:
        return None
    if len(raw_values) != 3 or any(
        isinstance(component, bool) for component in raw_values
    ):
        return None
    try:
        values = tuple(float(component) for component in raw_values)
    except (TypeError, ValueError, OverflowError):
        return None
    if any(not math.isfinite(component) for component in values):
        return None
    return values


def _coerce_trigger_area(
    value: object,
) -> tuple[tuple[float, float, float], tuple[float, float]] | None:
    """Normalize a drag-area payload while accepting legacy point callers."""

    if not isinstance(value, dict):
        point = _coerce_point(value)
        if point is None:
            return None
        return point, DEFAULT_TOUR_TRIGGER_AREA_SIZE_METERS
    point = _coerce_point(value.get("center"))
    size = value.get("size")
    if point is None or isinstance(size, (str, bytes, bytearray)):
        return None
    try:
        dimensions = tuple(size)  # type: ignore[arg-type]
    except TypeError:
        return None
    if len(dimensions) != 2 or any(
        isinstance(dimension, bool) for dimension in dimensions
    ):
        return None
    try:
        normalized_size = tuple(float(dimension) for dimension in dimensions)
    except (TypeError, ValueError, OverflowError):
        return None
    if any(
        not math.isfinite(dimension) or dimension <= 0.0
        for dimension in normalized_size
    ):
        return None
    return point, normalized_size


def _default_camera_target(
    curve: TourCurveData,
    progress: float,
) -> tuple[float, float, float]:
    """Place a new step's target one meter ahead along its camera path."""

    normalized_progress = max(0.0, min(1.0, float(progress)))
    camera_position = evaluate_catmull_rom_point(curve, normalized_progress)
    if normalized_progress < 1.0:
        sample_progress = min(
            1.0,
            normalized_progress + CAMERA_LOOK_AHEAD_PROGRESS,
        )
        sampled_position = evaluate_catmull_rom_point(curve, sample_progress)
        direction = tuple(
            sampled_position[axis] - camera_position[axis] for axis in range(3)
        )
    else:
        sampled_position = evaluate_catmull_rom_point(
            curve,
            max(0.0, 1.0 - CAMERA_LOOK_AHEAD_PROGRESS),
        )
        direction = tuple(
            camera_position[axis] - sampled_position[axis] for axis in range(3)
        )
    direction_length = math.sqrt(sum(value * value for value in direction))
    if direction_length <= 1e-9:
        direction = tuple(
            curve.points[-1][axis] - curve.points[0][axis] for axis in range(3)
        )
        direction_length = math.sqrt(sum(value * value for value in direction))
    if direction_length <= 1e-9:
        direction = (0.0, 1.0, 0.0)
        direction_length = 1.0
    return tuple(
        camera_position[axis] + (direction[axis] / direction_length)
        for axis in range(3)
    )


def _format_point(point: tuple[float, float, float]) -> str:
    return f"({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})"


def _normalized_color(value: str) -> str:
    color = QColor(str(value).strip())
    if not color.isValid():
        return DEFAULT_TEXT_COLOR
    return color.name(QColor.NameFormat.HexRgb)


def _action_label(action: TourActionData) -> str:
    if isinstance(action, TourText3DData):
        return f"3D text — {action.text}"
    if isinstance(action, TourFloatingTooltipData):
        return "Floating tooltip"
    if isinstance(action, TourWaitForKeyPressData):
        return f"Wait for key press — {_input_token_label(action.input_token)}"
    if isinstance(action, TourIdleCameraAnimationData):
        return "Idle camera animation"
    if isinstance(action, TourSpeedOverrideData):
        return f"Speed override — {action.speed:.2f}x"
    return "Action"


def _select_combo_data(combo: QComboBox, value: str) -> None:
    """Select a combo item's data, preserving a loaded custom input token."""

    with QSignalBlocker(combo):
        index = combo.findData(value)
        if index < 0:
            combo.addItem(_input_token_label(value), value)
            index = combo.count() - 1
        combo.setCurrentIndex(index)


def _input_token_from_event(event: QEvent) -> str | None:
    """Convert an application-wide key or mouse press to a stable token."""

    if event.type() == QEvent.Type.MouseButtonPress and isinstance(
        event,
        QMouseEvent,
    ):
        mouse_tokens = {
            Qt.MouseButton.LeftButton: "mouse:left",
            Qt.MouseButton.MiddleButton: "mouse:middle",
            Qt.MouseButton.RightButton: "mouse:right",
        }
        return mouse_tokens.get(event.button())
    if event.type() != QEvent.Type.KeyPress or not isinstance(event, QKeyEvent):
        return None
    key_tokens = {
        Qt.Key.Key_Space: "key:space",
        Qt.Key.Key_Return: "key:enter",
        Qt.Key.Key_Enter: "key:enter",
        Qt.Key.Key_Escape: "key:escape",
        Qt.Key.Key_Tab: "key:tab",
        Qt.Key.Key_Up: "key:up",
        Qt.Key.Key_Down: "key:down",
        Qt.Key.Key_Left: "key:left",
        Qt.Key.Key_Right: "key:right",
    }
    known_token = key_tokens.get(Qt.Key(event.key()))
    if known_token is not None:
        return known_token
    text = event.text().casefold()
    if len(text) == 1 and text.isprintable() and not text.isspace():
        return f"key:{text}"
    return f"key_code:{event.key()}"


def _input_matches(configured_token: str, input_token: str) -> bool:
    return configured_token.casefold() in {"any", input_token.casefold()}


def _input_token_label(input_token: str) -> str:
    normalized = str(input_token).casefold()
    for label, token in INPUT_TOKEN_OPTIONS:
        if token == normalized:
            return label
    if normalized.startswith("key:"):
        return normalized.removeprefix("key:").replace("_", " ").title()
    if normalized.startswith("mouse:"):
        return f"{normalized.removeprefix('mouse:').title()} mouse button"
    return normalized


def _select_list_item(
    widget: QListWidget,
    item_id: str,
    *,
    select_first: bool = True,
) -> None:
    for index in range(widget.count()):
        item = widget.item(index)
        if str(item.data(ITEM_ID_ROLE)) == item_id:
            widget.setCurrentItem(item)
            return
    widget.setCurrentRow(0 if select_first and widget.count() else -1)


def _find_step(tour: TourData | None, step_id: str) -> TourStepData | None:
    if tour is None:
        return None
    return next((step for step in tour.steps if step.step_id == step_id), None)


def _timeline_step_at_progress(
    tour: TourData,
    progress: float,
    *,
    selected_step_id: str = "",
) -> TourStepData | None:
    """Return the latest step reached by the normalized playhead.

    Before the first authored marker, the first step remains active. When
    several steps share the exact marker, an already selected one wins so a
    stationary playhead does not make the editor jump between equal markers.
    """

    if not tour.steps:
        return None
    prepared_progress = max(0.0, min(1.0, float(progress)))
    selected_step = _find_step(tour, selected_step_id)
    if selected_step is not None and math.isclose(
        selected_step.progress,
        prepared_progress,
        abs_tol=1e-12,
    ):
        return selected_step

    first_step: tuple[float, int, TourStepData] | None = None
    reached_step: tuple[float, int, TourStepData] | None = None
    for index, step in enumerate(tour.steps):
        candidate = (step.progress, index, step)
        if first_step is None or candidate[:2] < first_step[:2]:
            first_step = candidate
        if step.progress > prepared_progress + 1e-12:
            continue
        if reached_step is None or candidate[:2] > reached_step[:2]:
            reached_step = candidate
    resolved = reached_step if reached_step is not None else first_step
    return None if resolved is None else resolved[2]


def _find_action(
    step: TourStepData | None,
    action_id: str,
) -> TourActionData | None:
    if step is None:
        return None
    return next(
        (action for action in step.actions if action.component_id == action_id),
        None,
    )


def _find_action_of_type(
    step: TourStepData | None,
    action_type: type,
) -> TourActionData | None:
    if step is None:
        return None
    return next(
        (action for action in step.actions if isinstance(action, action_type)),
        None,
    )


def _find_action_by_id(
    tour: TourData | None,
    action_id: str,
) -> TourActionData | None:
    if tour is None:
        return None
    return next(
        (
            action
            for step in tour.steps
            for action in step.actions
            if action.component_id == action_id
        ),
        None,
    )


def _step_containing_action(
    tour: TourData,
    action_id: str,
) -> TourStepData | None:
    return next(
        (
            step
            for step in tour.steps
            if any(action.component_id == action_id for action in step.actions)
        ),
        None,
    )


def _first_unconsumed_wait_action(
    tour: TourData,
    start_progress: float,
    end_progress: float,
    consumed_action_ids: set[str],
) -> tuple[TourStepData, TourWaitForKeyPressData] | None:
    """Find the first input gate reached, including one at playback start."""

    ordered_steps = sorted(
        enumerate(tour.steps),
        key=lambda item: (item[1].progress, item[0]),
    )
    for _index, step in ordered_steps:
        if step.progress < start_progress - 1e-12:
            continue
        if step.progress > end_progress + 1e-12:
            break
        wait_action = _find_action_of_type(step, TourWaitForKeyPressData)
        if (
            isinstance(wait_action, TourWaitForKeyPressData)
            and wait_action.component_id not in consumed_action_ids
        ):
            return step, wait_action
    return None


def _idle_camera_pose(
    tour: TourData,
    progress: float,
    action: TourIdleCameraAnimationData,
    elapsed_seconds: float,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
]:
    """Create a subtle looping camera offset around an authored pivot."""

    base_position = evaluate_catmull_rom_point(tour.curve, progress)
    pivot = action.pivot_point
    forward = tuple(pivot[axis] - base_position[axis] for axis in range(3))
    forward_length = math.sqrt(sum(component * component for component in forward))
    if forward_length <= 1e-9:
        forward = (0.0, 1.0, 0.0)
        forward_length = 1.0
    forward = tuple(component / forward_length for component in forward)
    reference_up = (0.0, 0.0, 1.0)
    right = (
        forward[1] * reference_up[2] - forward[2] * reference_up[1],
        forward[2] * reference_up[0] - forward[0] * reference_up[2],
        forward[0] * reference_up[1] - forward[1] * reference_up[0],
    )
    right_length = math.sqrt(sum(component * component for component in right))
    if right_length <= 1e-9:
        right = (1.0, 0.0, 0.0)
    else:
        right = tuple(component / right_length for component in right)
    camera_up = (
        right[1] * forward[2] - right[2] * forward[1],
        right[2] * forward[0] - right[0] * forward[2],
        right[0] * forward[1] - right[1] * forward[0],
    )
    phase = (
        math.tau
        * max(0.0, elapsed_seconds)
        / action.cycle_duration_seconds
    )
    horizontal_offset = action.radius_meters * math.sin(phase)
    vertical_offset = action.radius_meters * 0.35 * math.sin(phase * 2.0)
    position = tuple(
        base_position[axis]
        + right[axis] * horizontal_offset
        + camera_up[axis] * vertical_offset
        for axis in range(3)
    )
    return position, pivot


def _tour_with_replaced_step(tour: TourData, updated: TourStepData) -> TourData:
    return replace(
        tour,
        steps=tuple(
            updated if step.step_id == updated.step_id else step for step in tour.steps
        ),
    )


def _step_with_replaced_action(
    step: TourStepData,
    updated: TourActionData,
) -> TourStepData:
    return replace(
        step,
        actions=tuple(
            updated if action.component_id == updated.component_id else action
            for action in step.actions
        ),
    )
