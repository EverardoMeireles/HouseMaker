# ### Imports ###
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QCloseEvent,
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QResizeEvent,
)
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.ceiling_height_estimation import (
    CEILING_HEIGHT_INFERENCE_MODEL,
    CEILING_HEIGHT_MODEL_OPTIONS,
    estimate_ceiling_height_api_cost_usd,
    format_ceiling_height_api_cost,
)
from housemaker.ceiling_height_ranges import (
    CeilingHeightVideoRange,
    sample_marked_video_frames,
)
from housemaker.video_source import VIDEO_FILE_FILTER, VideoFrameSource

# ### Constants ###
CEILING_HEIGHT_INTERVAL_SETTING_KEY = (
    "generation/ceiling_height_frame_analysis_interval"
)
DEFAULT_FRAME_ANALYSIS_INTERVAL = 15
READY_STATUS_TEXT = (
    "Choose a level, mark the beginning and end of each video section to analyse. "
    "Drag a colored section's ends to resize it; select a section to remove it."
)


# ### Stable level colors ###
def _level_range_color(level_index: int) -> QColor:
    """Golden-angle hues remain stable when levels are reordered or added."""
    hue = (0.55 + int(level_index) * 0.618033988749895) % 1.0
    return QColor.fromHsvF(hue, 0.57, 0.93)


# ### Colored, editable range timeline ###
class CeilingHeightRangeTimeline(QWidget):
    """Per-level range bands; the seek slider above remains independent."""

    ranges_edited = Signal(object)
    selection_changed = Signal(int)
    frame_requested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ceiling_height_range_timeline")
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.setMinimumWidth(300)
        self.setMouseTracking(True)
        self._frame_count = 0
        self._current_frame = 0
        self._levels: tuple[tuple[str, int], ...] = ()
        self._ranges: tuple[CeilingHeightVideoRange, ...] = ()
        self._selected_index = -1
        self._drag_endpoint: tuple[int, str] | None = None
        self._editable = True
        self._pending_mark: tuple[int, int] | None = None
        self.set_levels(())

    @property
    def selected_index(self) -> int:
        return self._selected_index

    def set_levels(self, levels: Sequence[tuple[str, int]]) -> None:
        self._levels = tuple(levels)
        self.setFixedHeight(max(44, 26 * len(self._levels) + 8))
        self.update()

    def set_frame_count(self, frame_count: int) -> None:
        self._frame_count = max(0, int(frame_count))
        self.update()

    def set_current_frame(self, frame_index: int) -> None:
        self._current_frame = max(0, int(frame_index))
        self.update()

    def set_pending_mark(self, pending: tuple[int, int] | None) -> None:
        self._pending_mark = pending
        self.update()

    def set_ranges(self, ranges: Sequence[CeilingHeightVideoRange]) -> None:
        self._ranges = tuple(ranges)
        if self._selected_index >= len(self._ranges):
            self.select_range(-1)
        self.update()

    def set_editable(self, editable: bool) -> None:
        self._editable = bool(editable)
        if not self._editable:
            self._drag_endpoint = None

    def select_range(self, index: int) -> None:
        normalized = index if 0 <= index < len(self._ranges) else -1
        if normalized != self._selected_index:
            self._selected_index = normalized
            self.selection_changed.emit(normalized)
            self.update()

    def _track_rect(self) -> QRectF:
        label_width = min(150, max(60, self.width() // 4))
        return QRectF(
            label_width,
            0,
            max(1, self.width() - label_width - 8),
            self.height(),
        )

    def _frame_x(self, frame: int) -> float:
        track = self._track_rect()
        return track.left() + track.width() * frame / max(1, self._frame_count - 1)

    def _x_frame(self, x: float) -> int:
        track = self._track_rect()
        fraction = min(1.0, max(0.0, (x - track.left()) / track.width()))
        return round(fraction * max(0, self._frame_count - 1))

    def _level_row(self, level: int) -> int:
        return next(
            (
                index
                for index, (_label, value) in enumerate(self._levels)
                if value == level
            ),
            0,
        )

    def _range_rect(self, marked: CeilingHeightVideoRange) -> QRectF:
        start = self._frame_x(marked.start_frame)
        end = self._frame_x(marked.end_frame)
        return QRectF(
            start,
            6 + self._level_row(marked.level_index) * 26,
            max(3.0, end - start),
            18,
        )

    def paintEvent(self, _event: object) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        track = self._track_rect()
        for row, (label, level_index) in enumerate(self._levels):
            painter.setPen(_level_range_color(level_index))
            painter.drawText(
                QRectF(0, 4 + row * 26, track.left() - 8, 22),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                painter.fontMetrics().elidedText(
                    label,
                    Qt.TextElideMode.ElideRight,
                    int(track.left() - 8),
                ),
            )
            painter.fillRect(
                QRectF(track.left(), 6 + row * 26, track.width(), 18),
                self.palette().alternateBase(),
            )
        for index, marked in enumerate(self._ranges):
            color = _level_range_color(marked.level_index)
            color.setAlpha(190)
            rect = self._range_rect(marked)
            painter.setBrush(color)
            painter.setPen(
                QPen(
                    QColor("white") if index == self._selected_index else color,
                    2,
                )
            )
            painter.drawRoundedRect(rect, 3, 3)
            painter.setPen(QPen(QColor("white"), 2))
            for x in (rect.left() + 2, rect.right() - 2):
                painter.drawLine(
                    QPointF(x, rect.top() + 3),
                    QPointF(x, rect.bottom() - 3),
                )
        if self._pending_mark is not None:
            level, frame = self._pending_mark
            row = self._level_row(level)
            painter.setPen(QPen(QColor("white"), 2, Qt.PenStyle.DashLine))
            x = self._frame_x(frame)
            painter.drawLine(
                QPointF(x, 4 + row * 26),
                QPointF(x, 26 + row * 26),
            )
        if self._frame_count > 0:
            painter.setPen(QPen(QColor("#ff5252"), 1))
            x = self._frame_x(self._current_frame)
            painter.drawLine(QPointF(x, 0), QPointF(x, self.height()))
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() != Qt.MouseButton.LeftButton or self._frame_count <= 0:
            return
        point = event.position()
        for index in reversed(range(len(self._ranges))):
            rect = self._range_rect(self._ranges[index])
            if rect.adjusted(-6, -3, 6, 3).contains(point):
                self.select_range(index)
                if self._editable:
                    if abs(point.x() - rect.left()) <= 8:
                        self._drag_endpoint = (index, "start")
                    elif abs(point.x() - rect.right()) <= 8:
                        self._drag_endpoint = (index, "end")
                return
        self.select_range(-1)
        if point.x() >= self._track_rect().left():
            self.frame_requested.emit(self._x_frame(point.x()))

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._drag_endpoint is None or not self._editable:
            return
        index, endpoint = self._drag_endpoint
        if index >= len(self._ranges):
            self._drag_endpoint = None
            return
        marked = self._ranges[index]
        frame = self._x_frame(event.position().x())
        start = (
            min(frame, marked.end_frame) if endpoint == "start" else marked.start_frame
        )
        end = max(frame, marked.start_frame) if endpoint == "end" else marked.end_frame
        ranges = list(self._ranges)
        ranges[index] = CeilingHeightVideoRange(marked.level_index, start, end)
        self.ranges_edited.emit(tuple(ranges))

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._drag_endpoint is not None:
            self.mouseMoveEvent(event)
        self._drag_endpoint = None


# ### Dialog ###
class CeilingHeightAnalysisDialog(QDialog):
    """Independent video player and marked multi-level height analysis."""

    analysis_requested = Signal(str, int, object)
    cancellation_requested = Signal()
    source_changed = Signal(str)
    ranges_changed = Signal(object)

    def __init__(
        self,
        *,
        application_settings: ApplicationSettingsStore | None = None,
        model_options: Sequence[tuple[str, str]] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("ceiling_height_analysis_dialog")
        self.setWindowTitle("Infer ceiling height")
        self.setModal(False)
        self._application_settings = application_settings
        self._model_options = tuple(
            CEILING_HEIGHT_MODEL_OPTIONS if model_options is None else model_options
        )
        self._selected_model = CEILING_HEIGHT_INFERENCE_MODEL
        self._is_running = False
        self._is_disposed = False
        self._frame_count = 0
        self._frame_width = 0
        self._frame_height = 0
        self._current_frame = 0
        self._video_source: VideoFrameSource | None = None
        self._video_path = ""
        self._preview_pixmap = QPixmap()
        self._marked_ranges: tuple[CeilingHeightVideoRange, ...] = ()
        self._pending_mark: tuple[int, int] | None = None
        self._plan_reference_size: tuple[int, int] | None = None
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._advance_playback)
        self._build_ui()
        self._restore_preferences()
        self._connect_signals()
        self._refresh_controls()
        self.resize(760, 700)

    # ### Public state ###
    @property
    def selected_model(self) -> str:
        return self._selected_model

    @property
    def video_path(self) -> str:
        return self._video_path

    @property
    def marked_ranges(self) -> tuple[CeilingHeightVideoRange, ...]:
        return self._marked_ranges

    @property
    def frame_analysis_interval(self) -> int:
        return int(self.frame_interval_spinbox.value())

    @property
    def selected_level_index(self) -> int | None:
        value = self.level_to_mark_combo.currentData()
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    def set_model(self, model: str) -> None:
        available = {value for _label, value in self._model_options}
        self._selected_model = (
            model if model in available else CEILING_HEIGHT_INFERENCE_MODEL
        )
        self._refresh_controls()

    def set_level_options(
        self,
        options: Sequence[tuple[str, int]],
        *,
        current_level_index: int | None = None,
    ) -> None:
        previous = self.selected_level_index
        self.level_to_mark_combo.blockSignals(True)
        self.level_to_mark_combo.clear()
        for label, level_index in options:
            if (
                isinstance(level_index, int)
                and not isinstance(level_index, bool)
                and str(label).strip()
            ):
                self.level_to_mark_combo.addItem(str(label).strip(), level_index)
        preferred = current_level_index if current_level_index is not None else previous
        index = self.level_to_mark_combo.findData(preferred)
        if index >= 0:
            self.level_to_mark_combo.setCurrentIndex(index)
        self.level_to_mark_combo.blockSignals(False)
        levels = self._level_options()
        self.range_timeline.set_levels(levels)
        valid_levels = {index for _, index in levels}
        retained = tuple(
            marked
            for marked in self._marked_ranges
            if marked.level_index in valid_levels
        )
        if self._pending_mark is not None and self._pending_mark[0] not in valid_levels:
            self._pending_mark = None
        self._replace_ranges(retained)
        self._refresh_controls()

    def set_video_path(self, path: str | Path, *, current_frame: int = 0) -> None:
        """Load an independent reader, retaining marks only for the same source."""
        normalized = str(Path(path).resolve()) if str(path).strip() else ""
        if (
            normalized
            and normalized == self._video_path
            and self._video_source is not None
        ):
            self.set_current_frame(current_frame)
            return
        source = VideoFrameSource(normalized) if normalized else None
        try:
            if source is not None and source.metadata.frame_count <= 0:
                raise ValueError("The video does not contain readable frames.")
            bounded = (
                max(0, min(int(current_frame), source.metadata.frame_count - 1))
                if source is not None
                else 0
            )
            first_frame = source.get_frame(bounded) if source is not None else None
        except Exception:
            if source is not None:
                source.close()
            raise
        self._stop_playback()
        if self._video_source is not None:
            self._video_source.close()
        self._video_source = source
        self._video_path = normalized
        self._pending_mark = None
        self._replace_ranges(())
        if source is None:
            self.set_video_metadata(0)
            self._preview_pixmap = QPixmap()
            self.video_preview_label.clear()
            self.video_preview_label.setText("Load a video to mark sections")
        else:
            metadata = source.metadata
            self.set_video_metadata(
                metadata.frame_count,
                fps=metadata.fps,
                width=metadata.width,
                height=metadata.height,
            )
            self._display_frame(first_frame)
            self.set_current_frame(bounded)
        self.source_changed.emit(self._video_path)
        self._set_status(READY_STATUS_TEXT)
        self._refresh_controls()

    def set_video_metadata(
        self,
        frame_count: int,
        *,
        fps: float | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        self._frame_count = max(0, int(frame_count))
        self._frame_width = max(0, int(width or 0))
        self._frame_height = max(0, int(height or 0))
        fps_text = f" at {float(fps):.2f} fps" if fps is not None and fps > 0 else ""
        self.video_metadata_label.setText(
            f"Video: {self._frame_count:,} frames{fps_text}"
            if self._frame_count
            else "No video loaded"
        )
        self.seek_slider.setRange(0, max(0, self._frame_count - 1))
        self.range_timeline.set_frame_count(self._frame_count)
        self._refresh_controls()

    def set_current_frame(self, frame_index: int | None) -> None:
        if frame_index is None or self._frame_count <= 0:
            self.current_frame_label.setText("Current frame: -")
            return
        self._current_frame = max(0, min(int(frame_index), self._frame_count - 1))
        if self._video_source is not None:
            try:
                self._display_frame(self._video_source.get_frame(self._current_frame))
            except (ValueError, IndexError) as error:
                self._stop_playback()
                self._set_status(str(error), True)
        self.seek_slider.blockSignals(True)
        self.seek_slider.setValue(self._current_frame)
        self.seek_slider.blockSignals(False)
        self.range_timeline.set_current_frame(self._current_frame)
        self.current_frame_label.setText(
            f"Current frame: {self._current_frame + 1:,} / {self._frame_count:,}"
        )

    def set_marked_ranges(self, ranges: Sequence[CeilingHeightVideoRange]) -> None:
        """Clamp UI edits to the video and reject ambiguous cross-level marks."""
        levels = {index for _, index in self._level_options()}
        if self._frame_count <= 0 and ranges:
            raise ValueError("Load a video before marking sections.")
        normalized: list[CeilingHeightVideoRange] = []
        for marked in ranges:
            if marked.level_index not in levels:
                raise ValueError("The marked level no longer has a loaded plan image.")
            normalized.append(
                CeilingHeightVideoRange(
                    marked.level_index,
                    min(marked.start_frame, self._frame_count - 1),
                    min(marked.end_frame, self._frame_count - 1),
                )
            )
        sample_marked_video_frames(
            normalized,
            self.frame_analysis_interval,
            self._frame_count,
        )
        self._replace_ranges(tuple(normalized))

    def set_plan_reference_size(self, size: tuple[int, int] | None) -> None:
        self._plan_reference_size = (
            None if size is None else (max(1, int(size[0])), max(1, int(size[1])))
        )
        self._refresh_controls()

    def set_running(self, running: bool) -> None:
        self._is_running = bool(running)
        if self._is_running:
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setValue(0)
            self.progress_bar.show()
        else:
            self.progress_bar.hide()
        self._refresh_controls()

    def set_progress(
        self,
        completed_frames: int,
        total_frames: int,
        *,
        current_frame: int | None = None,
    ) -> None:
        total = max(0, int(total_frames))
        completed = max(0, min(int(completed_frames), total))
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(completed)
        self.progress_bar.show()
        text = f"Analysed {completed:,} of {total:,} frame intervals"
        if current_frame is not None:
            text += f" (video frame {int(current_frame) + 1:,})"
        self._set_status(text)

    def set_result(self, text: str, *, is_error: bool = False) -> None:
        self.progress_bar.hide()
        self._set_status(str(text).strip() or READY_STATUS_TEXT, is_error)

    def dispose(self) -> None:
        if self._is_disposed:
            return
        self._is_disposed = True
        self._stop_playback()
        if self._video_source is not None:
            self._video_source.close()
            self._video_source = None
        self.close()
        self.deleteLater()

    # ### QWidget events ###
    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        self._stop_playback()
        if self._is_disposed:
            event.accept()
        else:
            self.hide()
            event.ignore()

    def resizeEvent(self, event: QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._refresh_preview_pixmap()

    # ### UI construction ###
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)
        self.video_preview_label = QLabel("Load a video to mark sections")
        self.video_preview_label.setObjectName("ceiling_height_video_preview")
        self.video_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_preview_label.setMinimumSize(320, 180)
        self.video_preview_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Expanding,
        )
        self.video_preview_label.setStyleSheet("background: #141414; color: white;")
        root.addWidget(self.video_preview_label, 1)
        player_row = QHBoxLayout()
        self.load_video_button = QPushButton("Load video")
        self.load_video_button.setObjectName("ceiling_height_load_video_button")
        self.play_button = QPushButton("Play")
        self.play_button.setObjectName("ceiling_height_play_button")
        self.current_frame_label = QLabel("Current frame: -")
        self.current_frame_label.setObjectName("ceiling_height_current_frame_label")
        for widget in (
            self.load_video_button,
            self.play_button,
            self.current_frame_label,
        ):
            player_row.addWidget(widget)
        player_row.addStretch(1)
        root.addLayout(player_row)
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setObjectName("ceiling_height_video_seek_slider")
        root.addWidget(self.seek_slider)
        self.range_timeline = CeilingHeightRangeTimeline()
        root.addWidget(self.range_timeline)
        self.video_metadata_label = QLabel("No video loaded")
        self.video_metadata_label.setObjectName("ceiling_height_video_metadata_label")
        root.addWidget(self.video_metadata_label)
        form = QFormLayout()
        form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow,
        )
        self.level_to_mark_combo = QComboBox()
        self.level_to_mark_combo.setObjectName("ceiling_height_level_to_mark_combo")
        form.addRow("Level to mark", self.level_to_mark_combo)
        self.frame_interval_spinbox = QSpinBox()
        self.frame_interval_spinbox.setObjectName(
            "ceiling_height_frame_analysis_interval_spinbox",
        )
        self.frame_interval_spinbox.setRange(1, 1_000_000)
        self.frame_interval_spinbox.setSuffix(" frames")
        self.frame_interval_spinbox.setValue(DEFAULT_FRAME_ANALYSIS_INTERVAL)
        form.addRow("Frame analysis interval", self.frame_interval_spinbox)
        root.addLayout(form)
        mark_row = QHBoxLayout()
        self.mark_begin_button = QPushButton("Mark begin")
        self.mark_begin_button.setObjectName("ceiling_height_mark_begin_button")
        self.mark_end_button = QPushButton("Mark end")
        self.mark_end_button.setObjectName("ceiling_height_mark_end_button")
        self.remove_range_button = QPushButton("Remove selected section")
        self.remove_range_button.setObjectName("ceiling_height_remove_range_button")
        for widget in (
            self.mark_begin_button,
            self.mark_end_button,
            self.remove_range_button,
        ):
            mark_row.addWidget(widget)
        root.addLayout(mark_row)
        self.pending_mark_label = QLabel()
        self.pending_mark_label.setObjectName("ceiling_height_pending_mark_label")
        self.pending_mark_label.setWordWrap(True)
        root.addWidget(self.pending_mark_label)
        self.analysis_button = QPushButton("Analyse video's ceiling height")
        self.analysis_button.setObjectName("analyse_video_ceiling_height_button")
        root.addWidget(self.analysis_button)
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("ceiling_height_analysis_progress_bar")
        self.progress_bar.hide()
        root.addWidget(self.progress_bar)
        self.status_label = QLabel(READY_STATUS_TEXT)
        self.status_label.setObjectName("ceiling_height_analysis_status_label")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

    # ### Signal wiring and range editing ###
    def _connect_signals(self) -> None:
        self.analysis_button.clicked.connect(self._handle_analysis_clicked)
        self.frame_interval_spinbox.valueChanged.connect(
            self._handle_frame_interval_changed
        )
        self.level_to_mark_combo.currentIndexChanged.connect(self._refresh_controls)
        self.load_video_button.clicked.connect(self._handle_load_video_clicked)
        self.play_button.clicked.connect(self._toggle_playback)
        self.seek_slider.valueChanged.connect(self.set_current_frame)
        self.mark_begin_button.clicked.connect(self._mark_begin)
        self.mark_end_button.clicked.connect(self._mark_end)
        self.remove_range_button.clicked.connect(self._remove_selected_range)
        self.range_timeline.ranges_edited.connect(self._handle_ranges_edited)
        self.range_timeline.selection_changed.connect(self._refresh_controls)
        self.range_timeline.frame_requested.connect(self.set_current_frame)

    def _level_options(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (self.level_to_mark_combo.itemText(i), self.level_to_mark_combo.itemData(i))
            for i in range(self.level_to_mark_combo.count())
        )

    def _replace_ranges(self, ranges: tuple[CeilingHeightVideoRange, ...]) -> None:
        changed = ranges != self._marked_ranges
        self._marked_ranges = ranges
        self.range_timeline.set_ranges(ranges)
        self._refresh_controls()
        if changed:
            self.ranges_changed.emit(ranges)

    def _mark_begin(self) -> None:
        if (
            self._is_running
            or self._frame_count <= 0
            or self.selected_level_index is None
        ):
            return
        self._pending_mark = (self.selected_level_index, self._current_frame)
        self._refresh_controls()

    def _mark_end(self) -> None:
        if self._is_running or self._pending_mark is None:
            return
        level, first = self._pending_mark
        marked = CeilingHeightVideoRange(
            level,
            min(first, self._current_frame),
            max(first, self._current_frame),
        )
        try:
            self.set_marked_ranges((*self._marked_ranges, marked))
        except ValueError as error:
            self._set_status(str(error), True)
            return
        self._pending_mark = None
        self.range_timeline.select_range(len(self._marked_ranges) - 1)
        self._set_status(READY_STATUS_TEXT)
        self._refresh_controls()

    def _handle_ranges_edited(self, ranges: object) -> None:
        if self._is_running:
            return
        try:
            self.set_marked_ranges(ranges)  # type: ignore[arg-type]
        except ValueError as error:
            self._set_status(str(error), True)

    def _remove_selected_range(self) -> None:
        index = self.range_timeline.selected_index
        if not self._is_running and index >= 0:
            self._replace_ranges(
                tuple(
                    marked for i, marked in enumerate(self._marked_ranges) if i != index
                )
            )
            self.range_timeline.select_range(-1)

    def _handle_analysis_clicked(self) -> None:
        if self._is_running:
            self.cancellation_requested.emit()
        elif self._marked_ranges:
            self._stop_playback()
            self.analysis_requested.emit(
                self.selected_model,
                self.frame_analysis_interval,
                self._marked_ranges,
            )

    # ### Independent player ###
    def _handle_load_video_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load ceiling analysis video",
            self._video_path,
            VIDEO_FILE_FILTER,
        )
        if path:
            try:
                self.set_video_path(path)
            except (ValueError, OSError) as error:
                self._set_status(str(error), True)

    def _toggle_playback(self) -> None:
        if self._play_timer.isActive():
            self._stop_playback()
        elif self._video_source is not None:
            if self._current_frame >= self._frame_count - 1:
                self.set_current_frame(0)
            self._play_timer.start(
                max(1, round(1000 / self._video_source.metadata.fps))
            )
            self.play_button.setText("Pause")

    def _stop_playback(self) -> None:
        self._play_timer.stop()
        self.play_button.setText("Play")

    def _advance_playback(self) -> None:
        if self._current_frame >= self._frame_count - 1:
            self._stop_playback()
        else:
            self.set_current_frame(self._current_frame + 1)

    def _display_frame(self, frame: np.ndarray | None) -> None:
        if frame is None:
            return
        height, width = frame.shape[:2]
        image = QImage(
            frame.data,
            width,
            height,
            int(frame.strides[0]),
            QImage.Format.Format_BGR888,
        ).copy()
        self._preview_pixmap = QPixmap.fromImage(image)
        self._refresh_preview_pixmap()

    def _refresh_preview_pixmap(self) -> None:
        if not self._preview_pixmap.isNull():
            self.video_preview_label.setPixmap(
                self._preview_pixmap.scaled(
                    self.video_preview_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    # ### Control state and estimates ###
    def _refresh_controls(self) -> None:
        has_video = self._frame_count > 0
        editable = has_video and not self._is_running
        self.load_video_button.setEnabled(not self._is_running)
        self.play_button.setEnabled(self._video_source is not None)
        self.seek_slider.setEnabled(has_video)
        self.frame_interval_spinbox.setEnabled(editable)
        self.level_to_mark_combo.setEnabled(editable)
        self.mark_begin_button.setEnabled(
            editable and self.selected_level_index is not None
        )
        self.mark_end_button.setEnabled(editable and self._pending_mark is not None)
        self.remove_range_button.setEnabled(
            editable and self.range_timeline.selected_index >= 0
        )
        self.range_timeline.set_editable(editable)
        self.range_timeline.set_pending_mark(self._pending_mark)
        if self._pending_mark is None:
            self.pending_mark_label.setText("No unfinished section")
        else:
            level, frame = self._pending_mark
            label = {value: label for label, value in self._level_options()}.get(
                level, str(level)
            )
            self.pending_mark_label.setText(
                f"{label}: begin marked at frame {frame + 1:,}; "
                "seek to its end and click Mark end"
            )
        self.analysis_button.setEnabled(
            self._is_running or (editable and bool(self._marked_ranges)),
        )
        self.analysis_button.setText(
            "Cancel ceiling height analysis"
            if self._is_running
            else self._analysis_button_text(),
        )

    def _analysis_button_text(self) -> str:
        base = "Analyse video's ceiling height"
        model = self.selected_model
        if "local" in model.casefold():
            return f"{base} ({model}; Local)"
        try:
            samples = sample_marked_video_frames(
                self._marked_ranges,
                self.frame_analysis_interval,
                self._frame_count,
            )
            cost = estimate_ceiling_height_api_cost_usd(
                model=model,
                frame_count=self._frame_count,
                interval=self.frame_analysis_interval,
                frame_width=self._frame_width,
                frame_height=self._frame_height,
                plan_reference_size=self._plan_reference_size,
                sample_count=len(samples),
            )
            formatted = format_ceiling_height_api_cost(cost).replace(
                "estimated",
                "estimated cost",
                1,
            )
        except ValueError:
            formatted = ""
        return f"{base} ({model}; {formatted})" if formatted else f"{base} ({model})"

    def _set_status(self, text: str, is_error: bool = False) -> None:
        self.status_label.setText(text)
        self.status_label.setStyleSheet("color: #c62828;" if is_error else "")

    # ### Interval preference persistence ###
    def _restore_preferences(self) -> None:
        if self._application_settings is None:
            return
        raw = self._application_settings.get(
            CEILING_HEIGHT_INTERVAL_SETTING_KEY,
            DEFAULT_FRAME_ANALYSIS_INTERVAL,
        )
        if isinstance(raw, bool):
            return
        try:
            interval = int(raw)
        except (TypeError, ValueError, OverflowError):
            return
        if interval >= 1:
            self.frame_interval_spinbox.setValue(interval)

    def _handle_frame_interval_changed(self, interval: int) -> None:
        if self._application_settings is not None:
            self._application_settings.set(
                CEILING_HEIGHT_INTERVAL_SETTING_KEY,
                max(1, int(interval)),
            )
        self._refresh_controls()


# ### Public exports ###
__all__ = [
    "CEILING_HEIGHT_INTERVAL_SETTING_KEY",
    "CeilingHeightAnalysisDialog",
]
