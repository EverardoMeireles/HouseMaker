# ### Imports ###
from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QLabel,
    QProgressBar,
    QPushButton,
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

# ### Constants ###
CEILING_HEIGHT_MODEL_SETTING_KEY = (
    "generation/ceiling_height_estimation_model"
)
CEILING_HEIGHT_INTERVAL_SETTING_KEY = (
    "generation/ceiling_height_frame_analysis_interval"
)
DEFAULT_FRAME_ANALYSIS_INTERVAL = 15
READY_STATUS_TEXT = "Ready to analyse the complete video."


# ### Dialog ###
class CeilingHeightAnalysisDialog(QDialog):
    """Persistent modeless controls for a multi-frame height analysis."""

    analysis_requested = Signal(str, int, int)
    cancellation_requested = Signal()

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
            CEILING_HEIGHT_MODEL_OPTIONS
            if model_options is None
            else model_options
        )
        self._is_running = False
        self._is_disposed = False
        self._frame_count = 0
        self._frame_width = 0
        self._frame_height = 0
        self._plan_reference_size: tuple[int, int] | None = None

        self._build_ui()
        self._restore_preferences()
        self._connect_signals()
        self._refresh_controls()
        self.resize(520, 300)

    # ### Public state ###
    @property
    def selected_model(self) -> str:
        model = self.model_combo.currentData()
        if isinstance(model, str) and model:
            return model
        return CEILING_HEIGHT_INFERENCE_MODEL

    @property
    def frame_analysis_interval(self) -> int:
        return int(self.frame_interval_spinbox.value())

    @property
    def selected_level_index(self) -> int | None:
        level_index = self.current_level_combo.currentData()
        if isinstance(level_index, bool) or not isinstance(level_index, int):
            return None
        return level_index

    def set_level_options(
        self,
        options: Sequence[tuple[str, int]],
        *,
        current_level_index: int | None = None,
    ) -> None:
        """Replace project levels while retaining a valid selection."""

        previous_level_index = self.selected_level_index
        self.current_level_combo.blockSignals(True)
        self.current_level_combo.clear()
        for label, level_index in options:
            if isinstance(level_index, bool) or not isinstance(level_index, int):
                continue
            normalized_label = str(label).strip()
            if normalized_label:
                self.current_level_combo.addItem(normalized_label, level_index)
        preferred_level_index = (
            current_level_index
            if current_level_index is not None
            else previous_level_index
        )
        preferred_index = self.current_level_combo.findData(
            preferred_level_index
        )
        if preferred_index >= 0:
            self.current_level_combo.setCurrentIndex(preferred_index)
        self.current_level_combo.blockSignals(False)
        self._refresh_controls()

    def set_video_metadata(
        self,
        frame_count: int,
        *,
        fps: float | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        """Update the complete-video range and paid-call estimate."""

        self._frame_count = max(0, int(frame_count))
        self._frame_width = max(0, int(width or 0))
        self._frame_height = max(0, int(height or 0))

        if self._frame_count <= 0:
            self.video_metadata_label.setText("No video loaded")
        else:
            fps_text = ""
            if fps is not None and fps > 0.0:
                fps_text = f" at {float(fps):.2f} fps"
            self.video_metadata_label.setText(
                f"Video: {self._frame_count:,} frames{fps_text}"
            )
        self._refresh_controls()

    def set_current_frame(self, frame_index: int | None) -> None:
        if frame_index is None or self._frame_count <= 0:
            self.current_frame_label.setText("Current frame: -")
            return
        bounded_frame = max(
            0,
            min(int(frame_index), self._frame_count - 1),
        )
        self.current_frame_label.setText(
            f"Current frame: {bounded_frame + 1:,}"
        )

    def set_plan_reference_size(
        self,
        size: tuple[int, int] | None,
    ) -> None:
        """Include calibrated plan-image tokens in paid-cost estimates."""

        if size is None:
            self._plan_reference_size = None
        else:
            width, height = size
            self._plan_reference_size = (
                max(1, int(width)),
                max(1, int(height)),
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
        if total <= 0:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(completed)
        self.progress_bar.show()

        progress_text = f"Analysed {completed:,} of {total:,} frames"
        if current_frame is not None:
            progress_text += f" (video frame {int(current_frame) + 1:,})"
        self._set_status(progress_text)

    def set_result(self, text: str, *, is_error: bool = False) -> None:
        self.progress_bar.hide()
        self._set_status(str(text).strip() or READY_STATUS_TEXT, is_error)

    def dispose(self) -> None:
        """Permanently close this otherwise persistent dialog."""

        if self._is_disposed:
            return
        self._is_disposed = True
        self.close()
        self.deleteLater()

    # ### QWidget events ###
    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        if self._is_disposed:
            event.accept()
            return
        self.hide()
        event.ignore()

    # ### UI construction ###
    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        self.video_metadata_label = QLabel("No video loaded")
        self.video_metadata_label.setObjectName(
            "ceiling_height_video_metadata_label"
        )
        root_layout.addWidget(self.video_metadata_label)

        self.current_frame_label = QLabel("Current frame: -")
        self.current_frame_label.setObjectName(
            "ceiling_height_current_frame_label"
        )
        root_layout.addWidget(self.current_frame_label)

        form_layout = QFormLayout()
        form_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )

        self.model_combo = QComboBox()
        self.model_combo.setObjectName("ceiling_height_estimation_model_combo")
        for label, model in self._model_options:
            normalized_model = str(model).strip()
            if normalized_model:
                self.model_combo.addItem(str(label), normalized_model)
        form_layout.addRow("Ceiling height estimation model", self.model_combo)

        self.frame_interval_spinbox = QSpinBox()
        self.frame_interval_spinbox.setObjectName(
            "ceiling_height_frame_analysis_interval_spinbox"
        )
        self.frame_interval_spinbox.setRange(1, 1_000_000)
        self.frame_interval_spinbox.setSuffix(" frames")
        self.frame_interval_spinbox.setValue(DEFAULT_FRAME_ANALYSIS_INTERVAL)
        form_layout.addRow(
            "Frame analysis interval",
            self.frame_interval_spinbox,
        )

        self.current_level_combo = QComboBox()
        self.current_level_combo.setObjectName(
            "ceiling_height_current_level_combo"
        )
        form_layout.addRow("Current level", self.current_level_combo)
        root_layout.addLayout(form_layout)

        self.analysis_button = QPushButton("Analyse video's ceiling height")
        self.analysis_button.setObjectName(
            "analyse_video_ceiling_height_button"
        )
        root_layout.addWidget(self.analysis_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("ceiling_height_analysis_progress_bar")
        self.progress_bar.setTextVisible(True)
        self.progress_bar.hide()
        root_layout.addWidget(self.progress_bar)

        self.status_label = QLabel(READY_STATUS_TEXT)
        self.status_label.setObjectName("ceiling_height_analysis_status_label")
        self.status_label.setWordWrap(True)
        root_layout.addWidget(self.status_label)
        root_layout.addStretch(1)

    # ### Signal wiring ###
    def _connect_signals(self) -> None:
        self.analysis_button.clicked.connect(self._handle_analysis_clicked)
        self.model_combo.currentIndexChanged.connect(
            self._handle_selected_model_changed
        )
        self.frame_interval_spinbox.valueChanged.connect(
            self._handle_frame_interval_changed
        )
        self.current_level_combo.currentIndexChanged.connect(
            self._refresh_controls
        )

    def _handle_analysis_clicked(self) -> None:
        if self._is_running:
            self.cancellation_requested.emit()
            return
        level_index = self.selected_level_index
        if level_index is not None:
            self.analysis_requested.emit(
                self.selected_model,
                self.frame_analysis_interval,
                level_index,
            )

    # ### Control state ###
    def _refresh_controls(self) -> None:
        has_video = self._frame_count > 0
        has_level = self.selected_level_index is not None
        editable = has_video and not self._is_running

        self.model_combo.setEnabled(editable)
        self.frame_interval_spinbox.setEnabled(editable)
        self.current_level_combo.setEnabled(editable)
        self.analysis_button.setEnabled(
            self._is_running or (editable and has_level)
        )
        self.analysis_button.setText(
            "Cancel ceiling height analysis"
            if self._is_running
            else self._analysis_button_text()
        )

    def _analysis_button_text(self) -> str:
        base_text = "Analyse video's ceiling height"
        if (
            self._frame_count <= 0
            or self._frame_width <= 0
            or self._frame_height <= 0
        ):
            return base_text
        try:
            cost = estimate_ceiling_height_api_cost_usd(
                model=self.selected_model,
                frame_count=self._frame_count,
                interval=self.frame_analysis_interval,
                frame_width=self._frame_width,
                frame_height=self._frame_height,
                plan_reference_size=self._plan_reference_size,
            )
        except ValueError:
            return base_text
        formatted_cost = format_ceiling_height_api_cost(cost)
        return (
            f"{base_text} ({formatted_cost})"
            if formatted_cost
            else base_text
        )

    def _set_status(self, text: str, is_error: bool = False) -> None:
        self.status_label.setText(text)
        self.status_label.setStyleSheet(
            "color: #c62828;" if is_error else ""
        )

    # ### Preference persistence ###
    def _restore_preferences(self) -> None:
        saved_model = self._read_setting(
            CEILING_HEIGHT_MODEL_SETTING_KEY,
            CEILING_HEIGHT_INFERENCE_MODEL,
        )
        if isinstance(saved_model, str):
            model_index = self.model_combo.findData(saved_model)
            if model_index >= 0:
                self.model_combo.setCurrentIndex(model_index)
            else:
                default_index = self.model_combo.findData(
                    CEILING_HEIGHT_INFERENCE_MODEL
                )
                if default_index >= 0:
                    self.model_combo.setCurrentIndex(default_index)

        saved_interval = self._read_saved_interval()
        if saved_interval is not None:
            self.frame_interval_spinbox.setValue(saved_interval)

    def _handle_selected_model_changed(self) -> None:
        self._write_setting(
            CEILING_HEIGHT_MODEL_SETTING_KEY,
            self.selected_model,
        )
        self._refresh_controls()

    def _handle_frame_interval_changed(self, interval: int) -> None:
        self._write_setting(
            CEILING_HEIGHT_INTERVAL_SETTING_KEY,
            max(1, int(interval)),
        )
        self._refresh_controls()

    def _read_saved_interval(self) -> int | None:
        raw_interval = self._read_setting(
            CEILING_HEIGHT_INTERVAL_SETTING_KEY,
            None,
        )
        if isinstance(raw_interval, bool):
            return None
        try:
            interval = int(raw_interval)
        except (TypeError, ValueError, OverflowError):
            return None
        return interval if interval >= 1 else None

    def _read_setting(self, key: str, default: object) -> object:
        if self._application_settings is None:
            return default
        return self._application_settings.get(key, default)

    def _write_setting(self, key: str, value: object) -> None:
        if self._application_settings is not None:
            self._application_settings.set(key, value)

# ### Public exports ###
__all__ = [
    "CEILING_HEIGHT_INTERVAL_SETTING_KEY",
    "CEILING_HEIGHT_MODEL_SETTING_KEY",
    "CeilingHeightAnalysisDialog",
]
