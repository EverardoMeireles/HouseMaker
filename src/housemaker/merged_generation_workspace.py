# ### Imports ###
from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
from PySide6.QtCore import (
    QBuffer,
    QByteArray,
    QIODevice,
    QMimeData,
    Qt,
    QThread,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import QImage, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.ceiling_height_dialog import CeilingHeightAnalysisDialog
from housemaker.ceiling_height_estimation import (
    CEILING_HEIGHT_JOB_KIND,
    CeilingHeightAnalysisResult,
    CeilingHeightEstimate,
    CeilingHeightEstimationCancelled,
    CeilingHeightEstimationError,
    CeilingHeightEstimator,
    aggregate_level_ceiling_height_estimates,
    ceiling_height_model_requires_api_key,
    estimate_ceiling_height_api_cost_usd,
    format_ceiling_height_api_cost,
    open_ceiling_height_estimator,
    sample_entire_video_frame_indices,
)
from housemaker.door_state import is_door_body_slot
from housemaker.generation_jobs import (
    JOB_STATUS_RUNNING,
    GenerationJobManager,
)
from housemaker.generation_state import (
    MAX_MASK_STROKES_PER_FRAME,
    GenerationData,
    MaskStroke,
)
from housemaker.generation_workspace import (
    GenerationWorkspace,
)
from housemaker.models import LevelData
from housemaker.settings_widget import DEFAULT_CLEAR_MASK_HOTKEY
from housemaker.surface_texture_state import SurfaceTextureData
from housemaker.surface_texture_workspace import SurfaceTextureGenerationWorkspace
from housemaker.video_source import (
    VIDEO_FILE_FILTER,
    VideoFrameSource,
    VideoMetadata,
    probe_video,
)

# ### Constants ###
WORKFLOW_SECTION_SPACING = 10
GENERATION_CONTROLS_PADDING = 10
CEILING_HEIGHT_SHUTDOWN_WAIT_MILLISECONDS = 100
CEILING_HEIGHT_NOT_ESTIMATED_TEXT = "Ceiling height: Not estimated"
MAX_CONSECUTIVE_CEILING_HEIGHT_FRAME_FAILURES = 3
DOOR_SLOT_GENERATION_BLINK_INTERVAL_MILLISECONDS = 500
DOOR_SLOT_GENERATION_ACTIVE_STYLE = "color: #ff3030; font-weight: 700;"
DOOR_SLOT_GENERATION_HIDDEN_STYLE = (
    "color: rgba(255, 48, 48, 0); font-weight: 700;"
)
REFERENCE_IMAGE_FILE_FILTER = (
    "Image files (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff);;"
    "All files (*.*)"
)
GENERATION_ACTION_GENERATE = "generate"
GENERATION_ACTION_GEOMETRY = "generate_geometry"
GENERATION_ACTION_OBJECT_TEXTURE = "generate_object_texture"
GENERATION_ACTION_SURFACE_TEXTURE = "generate_surface_texture"


# ### Ceiling-height inference jobs ###
class _CeilingHeightInferenceThread(QThread):
    """Analyze sampled video frames with an independent background reader."""

    progress_changed = Signal(int, int, int)

    def __init__(
        self,
        video_path: str,
        frame_indices: tuple[int, ...],
        api_key: str,
        model: str,
        estimator: CeilingHeightEstimator | None,
        level_options: tuple[tuple[int, str], ...],
        starting_level_index: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._video_path = str(video_path)
        self._frame_indices = tuple(int(index) for index in frame_indices)
        self._api_key = str(api_key)
        self._model = str(model)
        self._estimator = estimator
        self._level_options = tuple(level_options)
        self._starting_level_index = int(starting_level_index)
        self.result: CeilingHeightAnalysisResult | None = None
        self.error_message: str | None = None
        self.was_cancelled = False

    def run(self) -> None:  # type: ignore[override]
        try:
            try:
                frame_estimates: list[tuple[int, CeilingHeightEstimate]] = []
                previous_frame: np.ndarray | None = None
                consecutive_failures = 0
                with open_ceiling_height_estimator(
                    model=self._model,
                    api_key=self._api_key,
                    estimator_override=self._estimator,
                    startup_cancellation_check=self.isInterruptionRequested,
                ) as estimator, VideoFrameSource(
                    self._video_path
                ) as video_source:
                    total = len(self._frame_indices)
                    for completed, frame_index in enumerate(
                        self._frame_indices,
                        start=1,
                    ):
                        if self.isInterruptionRequested():
                            raise CeilingHeightEstimationCancelled("Cancelled")
                        current_frame = video_source.get_frame(frame_index)
                        try:
                            estimate = estimator(
                                current_frame,
                                previous_frame_bgr=previous_frame,
                                cancellation_check=(
                                    self.isInterruptionRequested
                                ),
                            )
                        except CeilingHeightEstimationCancelled:
                            raise
                        except CeilingHeightEstimationError:
                            consecutive_failures += 1
                            if (
                                consecutive_failures
                                >= MAX_CONSECUTIVE_CEILING_HEIGHT_FRAME_FAILURES
                            ):
                                raise
                            estimate = CeilingHeightEstimate(
                                status="insufficient_evidence",
                                estimate_m=None,
                                minimum_m=None,
                                maximum_m=None,
                                confidence="low",
                                basis="This sampled frame could not be analysed.",
                                level_movement="uncertain",
                                movement_confidence="low",
                            )
                        else:
                            consecutive_failures = 0
                        frame_estimates.append(
                            (
                                frame_index,
                                estimate,
                            )
                        )
                        previous_frame = current_frame
                        self.progress_changed.emit(
                            completed,
                            total,
                            frame_index,
                        )
                result = aggregate_level_ceiling_height_estimates(
                    frame_estimates,
                    starting_level_index=self._starting_level_index,
                    levels=self._level_options,
                )
            except CeilingHeightEstimationCancelled:
                self.was_cancelled = True
                return
            except CeilingHeightEstimationError as error:
                if self.isInterruptionRequested():
                    self.was_cancelled = True
                else:
                    self.error_message = str(error)
                return
            except Exception:  # noqa: BLE001 - redact unexpected worker failures.
                if self.isInterruptionRequested():
                    self.was_cancelled = True
                else:
                    self.error_message = "Ceiling height inference failed."
                return
            if self.isInterruptionRequested():
                self.was_cancelled = True
                return
            self.result = result
        finally:
            self._api_key = ""
            self._video_path = ""


@dataclass
class _CeilingHeightInferenceRuntime:
    """GUI-owned lifecycle for one source-bound video analysis."""

    video_path: str
    frame_indices: tuple[int, ...]
    level_signature: tuple[tuple[int, str], ...]
    starting_level_index: int
    job_id: str
    thread: _CeilingHeightInferenceThread
    cancel_requested: bool = False


# ### Ceiling-height presentation helpers ###
def _format_cost_confirmation_suffix(cost_usd: float | None) -> str:
    formatted = format_ceiling_height_api_cost(cost_usd)
    return f" ({formatted})" if formatted else ""


def _format_unassigned_frame_suffix(
    result: CeilingHeightAnalysisResult,
) -> str:
    count = len(result.unassigned_frame_indices)
    if count == 0:
        return ""
    return (
        f"; {count} transition or uncertain frame"
        f"{'s were' if count != 1 else ' was'} not assigned to a level"
    )


def _format_level_height_result(
    result: CeilingHeightAnalysisResult,
) -> str:
    lines = ["Detected ceiling heights:"]
    for level_estimate in result.level_estimates:
        analysis = level_estimate.analysis
        if analysis.average_height_m is None:
            lines.append(
                f"{level_estimate.level_name}: no reliable height estimate"
            )
            continue
        line = (
            f"{level_estimate.level_name}: {analysis.average_height_m:.2f} m "
            f"from {analysis.retained_count} consistent estimate"
            f"{'s' if analysis.retained_count != 1 else ''}"
        )
        if analysis.outlier_count:
            line += f"; {analysis.outlier_count} outlier(s) discarded"
        lines.append(line)
    unassigned_suffix = _format_unassigned_frame_suffix(result)
    if unassigned_suffix:
        lines.append(unassigned_suffix.removeprefix("; ").capitalize())
    return "\n".join(lines)


# ### Temporary generation references ###
@dataclass
class _TemporaryObjectReferenceSession:
    """Keep one external reference coherent across same-frame refreshes."""

    active_bgra: np.ndarray
    source_label: str
    generation_data_identity: int
    video_source: object | None
    frame_index: int | None


# ### Merged generation workspace ###
class MergedGenerationWorkspace(QWidget):
    """Present Surface and Object generation around shared reference media."""

    current_video_frame_changed = Signal()
    ceiling_height_estimate_changed = Signal(object)

    def __init__(
        self,
        surface_workspace: SurfaceTextureGenerationWorkspace,
        object_workspace: GenerationWorkspace,
        job_manager: GenerationJobManager,
        parent: QWidget | None = None,
        *,
        application_settings: ApplicationSettingsStore | None = None,
        ceiling_height_estimator: CeilingHeightEstimator | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("merged_generation_workspace")
        self.surface_workspace = surface_workspace
        self.object_workspace = object_workspace
        self.job_manager = job_manager
        self._application_settings = application_settings
        self._ceiling_height_estimator = ceiling_height_estimator
        self._ceiling_height_runtime: _CeilingHeightInferenceRuntime | None = None
        self._ceiling_height_dialog: CeilingHeightAnalysisDialog | None = None
        self._ceiling_height_video_path = ""
        self._ceiling_height_level_options: tuple[tuple[int, str], ...] = ()
        self._ceiling_height_current_level_index: int | None = None
        self._temporary_object_reference_session: (
            _TemporaryObjectReferenceSession | None
        ) = None
        self._door_slot_editing_active = False
        self._door_slot_generation_name = ""
        self._door_slot_generation_blink_visible = True
        self._generation_action_job_ids: dict[str, tuple[str, ...]] = {}
        self._generation_action_buttons: dict[str, QPushButton] = {}
        self._generation_action_default_text: dict[str, str] = {}
        self._generation_action_default_tooltips: dict[str, str] = {}
        self._is_shutdown = False

        self.surface_workspace.setParent(self)
        self.object_workspace.setParent(self)
        self.surface_workspace.hide()
        self.object_workspace.hide()
        self.object_workspace.set_shared_control_state_managed_externally(True)
        self._adopt_shared_controls()
        self._build_ui()
        self._door_slot_generation_blink_timer = QTimer(self)
        self._door_slot_generation_blink_timer.setInterval(
            DOOR_SLOT_GENERATION_BLINK_INTERVAL_MILLISECONDS
        )
        self._door_slot_generation_blink_timer.timeout.connect(
            self._toggle_door_slot_generation_blink
        )
        self._connect_shared_controls()
        self.clear_mask_shortcut = QShortcut(self)
        self.clear_mask_shortcut.setContext(
            Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        self.clear_mask_shortcut.activated.connect(self._clear_mask_from_shortcut)
        self.set_clear_mask_hotkey(DEFAULT_CLEAR_MASK_HOTKEY)
        self.sync_shared_controls()

    def shutdown(self) -> None:
        """Cancel and join Generation-tab analysis workers exactly once."""

        if self._is_shutdown:
            return
        self._is_shutdown = True
        self._door_slot_generation_blink_timer.stop()
        try:
            QApplication.clipboard().dataChanged.disconnect(
                self.sync_shared_controls
            )
        except (RuntimeError, TypeError):
            pass
        runtime = self._ceiling_height_runtime
        if runtime is not None:
            runtime.cancel_requested = True
            runtime.thread.requestInterruption()
            self.job_manager.set_cancel_callback(runtime.job_id, None)
            self.job_manager.mark_cancelled(
                runtime.job_id,
                stage="Cancelled during shutdown",
            )
            while runtime.thread.isRunning():
                runtime.thread.wait(CEILING_HEIGHT_SHUTDOWN_WAIT_MILLISECONDS)
            runtime.thread.deleteLater()
        self._ceiling_height_runtime = None
        if self._ceiling_height_dialog is not None:
            self._ceiling_height_dialog.dispose()
            self._ceiling_height_dialog = None

    def set_clear_mask_hotkey(self, hotkey: str) -> None:
        """Apply the selected Generation-only Clear mask keymapping."""

        self.clear_mask_shortcut.setKey(
            QKeySequence(hotkey, QKeySequence.SequenceFormat.PortableText)
        )
        self.clear_mask_shortcut.setEnabled(bool(hotkey))

    @Slot()
    def _clear_mask_from_shortcut(self) -> None:
        """Use the same enabled-state guard as the shared Clear mask button."""

        if self.clear_mask_button.isEnabled():
            self.clear_mask_button.click()

    def refresh_file_backed_previews(self) -> None:
        """Refresh the generated-object preview at the tab cache boundary."""

        self.object_workspace.refresh_file_backed_previews()

    def focus_navigation(self) -> None:
        """Forward navigation focus to the embedded generated-object view."""

        self.object_workspace.object_3d_panel.focus_navigation()

    def get_current_video_frame_bgr(self) -> np.ndarray | None:
        """Return a defensive copy of the frame displayed by Generation."""

        return self.video_view.get_frame_bgr()

    @staticmethod
    def merge_project_video_state(
        generation: GenerationData | None,
        surface: SurfaceTextureData | None,
    ) -> tuple[GenerationData, SurfaceTextureData]:
        """Migrate two legacy media states into one deterministic state."""

        object_data = GenerationData() if generation is None else generation.clone()
        surface_data = SurfaceTextureData() if surface is None else surface.clone()
        source = _select_richest_video_state(object_data, surface_data)
        if source is None:
            return object_data, surface_data

        metadata, frame_index, primary_frame_strokes = source
        frame_strokes = primary_frame_strokes
        if object_data.video_metadata == surface_data.video_metadata:
            frame_strokes = _merge_legacy_frame_strokes(
                primary_frame_strokes,
                object_data.frame_strokes,
                surface_data.frame_strokes,
            )
        object_data.video_metadata = copy.deepcopy(metadata)
        object_data.current_frame_index = int(frame_index)
        object_data.frame_strokes = copy.deepcopy(frame_strokes)
        surface_data.video_metadata = copy.deepcopy(metadata)
        surface_data.current_frame_index = int(frame_index)
        surface_data.frame_strokes = copy.deepcopy(frame_strokes)
        return object_data, surface_data

    def _adopt_shared_controls(self) -> None:
        """Install the Object-owned controls and migrate legacy duplicates."""

        surface = self.surface_workspace
        objects = self.object_workspace

        self.video_view = objects.video_view
        self.seekbar = objects.seekbar
        self.load_video_button = objects.load_video_button
        self.paint_mask_button = objects.paint_mask_button
        self.erase_mask_button = objects.erase_mask_button
        self.mask_mode_control = objects.mask_mode_control
        self.brush_size_spinbox = objects.brush_size_spinbox
        self.clear_mask_button = objects.clear_mask_button
        self.pbr_map_control = objects.pbr_map_control
        self.pbr_map_checkboxes = objects.pbr_map_checkboxes
        self.ai_prompt_edit = objects.ai_prompt_edit
        objects.cancel_operation_button.hide()

        if surface.video_view is self.video_view:
            return

        retired_surface_widgets = (
            surface.video_view,
            surface.seekbar,
            surface.load_video_button,
            surface.mask_mode_control,
            surface.brush_size_spinbox,
            surface.clear_mask_button,
            surface.ai_prompt_edit,
            surface.pbr_map_control,
        )
        retired_button_group = surface._mask_mode_group

        surface.video_view = self.video_view
        surface.seekbar = self.seekbar
        surface.load_video_button = self.load_video_button
        surface.paint_mask_button = self.paint_mask_button
        surface.erase_mask_button = self.erase_mask_button
        surface.mask_mode_control = self.mask_mode_control
        surface.brush_size_spinbox = self.brush_size_spinbox
        surface.clear_mask_button = self.clear_mask_button
        surface.pbr_map_control = self.pbr_map_control
        surface.pbr_map_checkboxes = self.pbr_map_checkboxes
        surface.ai_prompt_edit = self.ai_prompt_edit
        surface._mask_mode_group = objects._mask_mode_button_group
        surface._shared_controls = objects.get_shared_controls()

        for retired_widget in _unique_widgets(retired_surface_widgets):
            retired_widget.hide()
            retired_widget.setParent(None)
            retired_widget.deleteLater()
        retired_button_group.deleteLater()

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        self.views_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.views_splitter.setObjectName("merged_generation_views_splitter")
        self.views_splitter.setChildrenCollapsible(False)
        self.views_splitter.addWidget(
            _build_labeled_view(
                "Video material references and object mask",
                self.video_view,
            )
        )
        self.views_splitter.addWidget(self.object_workspace.object_3d_page)
        self.views_splitter.setStretchFactor(0, 1)
        self.views_splitter.setStretchFactor(1, 1)
        self.views_splitter.setSizes([1_000, 1_000])
        root_layout.addWidget(self.views_splitter, 1)
        root_layout.addWidget(self.seekbar)

        self.shared_controls = self._build_shared_controls()
        self.object_controls = self._build_object_controls()
        self.controls_row = QWidget()
        self.controls_row.setObjectName("merged_generation_controls_row")
        controls_row_layout = QHBoxLayout(self.controls_row)
        controls_row_layout.setContentsMargins(
            GENERATION_CONTROLS_PADDING,
            GENERATION_CONTROLS_PADDING,
            GENERATION_CONTROLS_PADDING,
            GENERATION_CONTROLS_PADDING,
        )
        controls_row_layout.setSpacing(WORKFLOW_SECTION_SPACING)
        controls_row_layout.addWidget(self.shared_controls, 1)
        controls_row_layout.addWidget(self.object_controls, 2)
        self.controls_scroll = QScrollArea()
        self.controls_scroll.setObjectName("merged_generation_controls_scroll")
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.controls_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.controls_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.controls_scroll.setWidget(self.controls_row)
        self.controls_scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Maximum,
        )
        self.controls_scroll.setMinimumHeight(
            self.controls_row.sizeHint().height()
            + self.controls_scroll.horizontalScrollBar().sizeHint().height()
        )
        root_layout.addWidget(self.controls_scroll)

    def _build_shared_controls(self) -> QWidget:
        section, layout = _build_controls_section("shared")

        shared_column, shared_layout = _build_controls_column(
            "merged_generation_shared_primary_column"
        )
        reference_media_actions = QWidget()
        reference_media_actions.setObjectName(
            "merged_generation_reference_media_actions"
        )
        reference_media_layout = QHBoxLayout(reference_media_actions)
        reference_media_layout.setContentsMargins(0, 0, 0, 0)
        reference_media_layout.setSpacing(6)
        self.load_video_button.setObjectName("load_video_button")
        self.copy_inpaint_button = QPushButton("Copy inpaint")
        self.copy_inpaint_button.setObjectName("copy_inpaint_button")
        self.copy_inpaint_button.setToolTip(
            "Copy the masked object or active temporary reference with "
            "transparency."
        )
        self.paste_inpaint_button = QPushButton("Paste")
        self.paste_inpaint_button.setObjectName("paste_inpaint_button")
        self.paste_inpaint_button.setToolTip(
            "Use a clipboard image as the next Object generation reference."
        )
        self.load_reference_image_button = QPushButton("Load image")
        self.load_reference_image_button.setObjectName(
            "load_reference_image_button"
        )
        self.load_reference_image_button.setToolTip(
            "Use an image as the next Object generation reference without "
            "replacing the loaded video."
        )
        reference_media_layout.addWidget(self.load_video_button)
        reference_media_layout.addWidget(self.copy_inpaint_button)
        reference_media_layout.addWidget(self.paste_inpaint_button)
        reference_media_layout.addWidget(self.load_reference_image_button)
        shared_layout.addWidget(reference_media_actions)

        self.ceiling_height_section, ceiling_height_layout = (
            _build_boxed_section(
                "Room analysis",
                "merged_generation_ceiling_height_section",
            )
        )
        self.infer_ceiling_height_button = QPushButton(
            "Infer ceiling height"
        )
        self.infer_ceiling_height_button.setObjectName(
            "infer_ceiling_height_button"
        )
        self.infer_ceiling_height_button.setToolTip(
            "Open multi-frame video analysis controls for a robust "
            "floor-to-ceiling height estimate."
        )
        ceiling_height_layout.addWidget(self.infer_ceiling_height_button)
        self.ceiling_height_result_label = QLabel(
            CEILING_HEIGHT_NOT_ESTIMATED_TEXT
        )
        self.ceiling_height_result_label.setObjectName(
            "ceiling_height_result_label"
        )
        self.ceiling_height_result_label.setWordWrap(True)
        self.ceiling_height_result_label.setTextFormat(
            Qt.TextFormat.PlainText
        )
        self.ceiling_height_result_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        ceiling_height_layout.addWidget(self.ceiling_height_result_label)
        shared_layout.addWidget(self.ceiling_height_section)

        self.clear_mask_button.setText("Clear mask")
        shared_layout.addWidget(self.mask_mode_control)

        self.shared_material_section, material_layout = _build_boxed_section(
            "Material generation",
            "merged_generation_shared_material_section",
        )
        material_layout.addWidget(self.pbr_map_control)

        self.ai_prompt_edit.setObjectName("generation_ai_prompt_edit")
        self.ai_prompt_edit.setPlaceholderText("AI prompt (optional)")
        self.ai_prompt_edit.setMaxLength(800)
        self.ai_prompt_edit.setMinimumWidth(190)
        self.ai_prompt_edit.setToolTip(
            "Guide Surface material generation and direct new Object "
            "texturing. Staged unused-face removal and later Object "
            "Retexture jobs keep the painted image as their sole style source."
        )
        material_layout.addWidget(
            _build_labeled_inline_control("AI prompt", self.ai_prompt_edit)
        )

        shared_layout.addWidget(self.shared_material_section)
        shared_layout.addStretch(1)

        layout.addWidget(shared_column, 0, 0)
        layout.setColumnStretch(0, 1)
        return section

    def _build_object_controls(self) -> QWidget:
        objects = self.object_workspace
        section, layout = _build_controls_section("object")

        primary_column, primary_layout = _build_controls_column(
            "merged_generation_object_primary_column"
        )

        preview_options = QWidget()
        preview_options.setObjectName("merged_generation_object_preview_options")
        preview_options_layout = QHBoxLayout(preview_options)
        preview_options_layout.setContentsMargins(0, 0, 0, 0)
        preview_options_layout.setSpacing(6)
        preview_options_layout.addWidget(objects.textures_checkbox)
        preview_options_layout.addWidget(objects.wireframe_checkbox)
        preview_options_layout.addStretch(1)

        details_layout = objects.object_3d_panel.details_panel.layout()
        relocated_detail_widgets = (
            objects.delete_selected_faces_button,
            objects.convert_faces_to_glass_button,
            objects.model_statistics_label,
            objects.object_3d_panel.projection_camera_controls,
        )
        if details_layout is not None:
            for widget in relocated_detail_widgets:
                details_layout.removeWidget(widget)

        self.object_editing_section, editing_layout = _build_boxed_section(
            "Object editing",
            "merged_generation_object_editing_section",
        )
        editing_layout.addWidget(preview_options)

        face_actions = QWidget()
        face_actions.setObjectName("merged_generation_object_face_actions")
        face_actions_layout = QHBoxLayout(face_actions)
        face_actions_layout.setContentsMargins(0, 0, 0, 0)
        face_actions_layout.setSpacing(6)
        face_actions_layout.addWidget(objects.delete_selected_faces_button, 1)
        face_actions_layout.addWidget(objects.convert_faces_to_glass_button, 1)
        editing_layout.addWidget(face_actions)
        primary_layout.addWidget(self.object_editing_section)

        self.object_creation_section, creation_layout = _build_boxed_section(
            "",
            "merged_generation_object_creation_section",
        )
        generation_header = QWidget(self.object_creation_section)
        generation_header_layout = QHBoxLayout(generation_header)
        generation_header_layout.setContentsMargins(0, 0, 0, 0)
        generation_header_layout.setSpacing(5)
        self.object_generation_heading_label = QLabel("Generation")
        self.object_generation_heading_label.setObjectName(
            "object_generation_heading_label"
        )
        self.door_slot_generation_indicator = QLabel()
        self.door_slot_generation_indicator.setObjectName(
            "door_slot_generation_indicator"
        )
        self.door_slot_generation_indicator.setStyleSheet(
            DOOR_SLOT_GENERATION_ACTIVE_STYLE
        )
        self.door_slot_generation_indicator.hide()
        generation_header_layout.addWidget(self.object_generation_heading_label)
        generation_header_layout.addWidget(self.door_slot_generation_indicator)
        generation_header_layout.addStretch(1)
        creation_layout.addWidget(generation_header)

        generation_settings = QWidget()
        generation_settings.setObjectName(
            "merged_generation_object_generation_settings"
        )
        generation_settings_layout = QHBoxLayout(generation_settings)
        generation_settings_layout.setContentsMargins(0, 0, 0, 0)
        generation_settings_layout.setSpacing(6)

        self.division_options_section, division_options_layout = (
            _build_boxed_section(
                "Division",
                "merged_generation_division_options_section",
            )
        )
        division_options_row = QWidget()
        division_options_row_layout = QHBoxLayout(division_options_row)
        division_options_row_layout.setContentsMargins(0, 0, 0, 0)
        division_options_row_layout.setSpacing(6)
        division_options_row_layout.addWidget(
            objects.symmetric_division_checkbox
        )
        division_options_row_layout.addWidget(
            objects.side_door_duplication_checkbox
        )
        division_options_layout.addWidget(division_options_row)
        generation_settings_layout.addWidget(self.division_options_section)

        self.opening_profile_options_section, profile_options_layout = (
            _build_boxed_section(
                "Door / window profile",
                "merged_generation_opening_profile_options_section",
            )
        )
        profile_options_row = QWidget()
        profile_options_row_layout = QHBoxLayout(profile_options_row)
        profile_options_row_layout.setContentsMargins(0, 0, 0, 0)
        profile_options_row_layout.setSpacing(6)
        profile_options_row_layout.addWidget(
            objects.generate_displacement_checkbox
        )
        profile_options_row_layout.addWidget(
            objects.preserve_door_profile_checkbox
        )
        profile_options_layout.addWidget(profile_options_row)
        generation_settings_layout.addWidget(
            self.opening_profile_options_section
        )
        generation_settings_layout.addStretch(1)
        generation_settings_layout.addWidget(objects.meshy_target_polycount_control)
        creation_layout.addWidget(generation_settings)

        self.texture_provider_control = _build_labeled_inline_control(
            "Texture provider",
            self.surface_workspace.surface_texture_provider_combo,
        )
        self.texture_provider_control.setObjectName(
            "merged_generation_texture_provider_control"
        )
        creation_layout.addWidget(self.texture_provider_control)

        primary_actions = QWidget()
        primary_actions.setObjectName("merged_generation_primary_object_actions")
        primary_actions_layout = QHBoxLayout(primary_actions)
        primary_actions_layout.setContentsMargins(0, 0, 0, 0)
        primary_actions_layout.setSpacing(0)
        primary_actions_layout.addWidget(objects.generate_geometry_button)
        self.generate_texture_stack = QStackedWidget()
        self.generate_texture_stack.setObjectName(
            "merged_generation_texture_action_stack"
        )
        objects.generate_texture_button.setText("Generate texture")
        self.surface_workspace.generate_button.setText("Generate texture")
        self.generate_texture_stack.addWidget(objects.generate_texture_button)
        self.generate_texture_stack.addWidget(
            self.surface_workspace.generate_button
        )
        primary_actions_layout.addWidget(self.generate_texture_stack)
        creation_layout.addWidget(primary_actions)

        generation_actions = QWidget()
        generation_actions.setObjectName("merged_generation_object_generation_actions")
        generation_actions_layout = QHBoxLayout(generation_actions)
        generation_actions_layout.setContentsMargins(0, 0, 0, 0)
        generation_actions_layout.setSpacing(6)
        generation_actions_layout.addWidget(objects.generate_button, 1)
        self.place_object_button = objects.place_button
        generation_actions_layout.addWidget(self.place_object_button, 1)
        creation_layout.addWidget(generation_actions)
        primary_layout.addWidget(self.object_creation_section)
        primary_layout.addWidget(objects.model_statistics_label)
        primary_layout.addStretch(1)

        projection_controls = objects.object_3d_panel.projection_camera_controls
        projection_controls.setMinimumWidth(0)
        projection_controls.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )

        layout.addWidget(primary_column, 0, 0)
        layout.addWidget(projection_controls, 0, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(1, 1)
        return section

    def _connect_shared_controls(self) -> None:
        surface = self.surface_workspace
        objects = self.object_workspace

        try:
            self.load_video_button.clicked.disconnect()
        except (RuntimeError, TypeError):
            pass
        self.load_video_button.clicked.connect(self._handle_load_video_clicked)
        self.copy_inpaint_button.clicked.connect(
            self._handle_copy_inpaint_clicked
        )
        self.paste_inpaint_button.clicked.connect(
            self._handle_paste_inpaint_clicked
        )
        self.load_reference_image_button.clicked.connect(
            self._handle_load_reference_image_clicked
        )
        QApplication.clipboard().dataChanged.connect(
            self.sync_shared_controls
        )
        self.infer_ceiling_height_button.clicked.connect(
            self._show_ceiling_height_analysis_dialog
        )

        self.video_view.strokes_changed.connect(surface._handle_video_strokes_changed)
        self.video_view.strokes_changed.connect(self.sync_shared_controls)
        self.video_view.frame_changed.connect(self._handle_video_frame_changed)
        objects.door_slot_editing_changed.connect(
            self._handle_door_slot_editing_changed
        )
        objects.architectural_trim_editing_changed.connect(
            self._handle_architectural_trim_editing_changed
        )
        surface.data_changed.connect(
            self._handle_surface_generation_context_changed
        )
        objects.door_slot_generation_status_changed.connect(
            self._handle_door_slot_generation_status_changed
        )
        objects.controls_synced.connect(self._sync_generation_action_buttons)
        surface.controls_synced.connect(self._sync_generation_action_buttons)
        try:
            self.seekbar.valueChanged.disconnect()
        except (RuntimeError, TypeError):
            pass
        self.seekbar.valueChanged.connect(self._handle_seekbar_changed)
        for checkbox in self.pbr_map_checkboxes.values():
            checkbox.toggled.connect(surface._handle_pbr_map_toggled)

        self._connect_generation_action_buttons()
        self.job_manager.job_added.connect(self._handle_job_changed)
        self.job_manager.job_updated.connect(self._handle_job_changed)
        self.job_manager.jobs_cleared.connect(self._sync_generation_action_buttons)

        objects._sync_controls()
        surface._sync_controls()
        self._sync_generation_texture_context()

    def _connect_generation_action_buttons(self) -> None:
        """Route each generation action through its own cancellable job group."""

        objects = self.object_workspace
        actions = {
            GENERATION_ACTION_GENERATE: objects.generate_button,
            GENERATION_ACTION_GEOMETRY: objects.generate_geometry_button,
            GENERATION_ACTION_OBJECT_TEXTURE: objects.generate_texture_button,
            GENERATION_ACTION_SURFACE_TEXTURE: self.surface_workspace.generate_button,
        }
        self._generation_action_buttons = actions
        self._generation_action_default_text = {
            action: button.text() for action, button in actions.items()
        }
        self._generation_action_default_tooltips = {
            action: button.toolTip() for action, button in actions.items()
        }
        for action, button in actions.items():
            try:
                button.clicked.disconnect()
            except (RuntimeError, TypeError):
                pass
            button.clicked.connect(
                partial(self._handle_generation_action_clicked, action)
            )

    @Slot(str, bool)
    def _handle_generation_action_clicked(
        self,
        action: str,
        _checked: bool = False,
    ) -> None:
        """Start one action or cancel only the jobs launched by its last click."""

        if self._generation_action_has_pending_jobs(action):
            self._cancel_generation_action(action)
            return

        previous_job_ids = {job.job_id for job in self.job_manager.jobs()}
        self._start_generation_action(action)
        created_job_ids = tuple(
            job.job_id
            for job in self.job_manager.jobs()
            if job.job_id not in previous_job_ids and not job.is_finished
        )
        if created_job_ids:
            self._generation_action_job_ids[action] = created_job_ids
        self._sync_generation_action_buttons()

    def _start_generation_action(self, action: str) -> None:
        """Invoke the current backend method for one visible action button."""

        if action == GENERATION_ACTION_GENERATE:
            self.object_workspace.generate()
            return
        if action == GENERATION_ACTION_GEOMETRY:
            self.object_workspace.generate_geometry()
            return
        if action == GENERATION_ACTION_OBJECT_TEXTURE:
            self.object_workspace.generate_selected_object_texture()
            return
        if action == GENERATION_ACTION_SURFACE_TEXTURE:
            self.surface_workspace.generate()
            return
        raise ValueError(f"Unknown generation action: {action}")

    def _generation_action_has_pending_jobs(self, action: str) -> bool:
        """Return whether the action still owns a running or cancelling job."""

        return any(
            job is not None and not job.is_finished
            for job_id in self._generation_action_job_ids.get(action, ())
            for job in (self.job_manager.get_job(job_id),)
        )

    def _cancel_generation_action(self, action: str) -> bool:
        """Cancel every still-running job produced by one action click."""

        accepted = False
        for job_id in self._generation_action_job_ids.get(action, ()):
            job = self.job_manager.get_job(job_id)
            if job is None or job.status != JOB_STATUS_RUNNING:
                continue
            accepted = self.job_manager.cancel_job(job_id) or accepted
        self._sync_generation_action_buttons()
        return accepted

    @Slot()
    def _sync_generation_action_buttons(self) -> None:
        """Keep each action's label and availability aligned with its own jobs."""

        for action, button in self._generation_action_buttons.items():
            job_ids = self._generation_action_job_ids.get(action, ())
            pending_jobs = tuple(
                job
                for job_id in job_ids
                if (
                    (job := self.job_manager.get_job(job_id)) is not None
                    and not job.is_finished
                )
            )
            if not pending_jobs:
                self._generation_action_job_ids.pop(action, None)
                button.setText(self._generation_action_default_text[action])
                button.setToolTip(
                    self._generation_action_default_tooltips[action]
                )
                continue

            button.setText("Cancel")
            button.setEnabled(
                any(job.status == JOB_STATUS_RUNNING for job in pending_jobs)
            )
            job_count = len(pending_jobs)
            button.setToolTip(
                "Cancel this generation job."
                if job_count == 1
                else f"Cancel all {job_count} jobs from this generation batch."
            )

    @Slot(bool)
    def _handle_door_slot_editing_changed(self, active: bool) -> None:
        """Keep shared analysis controls out of the door-slot workflow."""

        self._door_slot_editing_active = bool(active)
        self._sync_opening_generation_sections()
        self.sync_shared_controls()

    def _sync_opening_generation_sections(self) -> None:
        """Show opening-only options solely for the slot that can use them."""

        target = self.object_workspace._door_slot_editing_target
        body_slot_is_active = bool(
            target is not None and is_door_body_slot(target.slot_id)
        )
        non_body_slot_is_active = bool(
            target is not None and not body_slot_is_active
        )
        self.division_options_section.setVisible(
            not non_body_slot_is_active
        )
        self.opening_profile_options_section.setVisible(body_slot_is_active)

    @Slot(bool)
    def _handle_architectural_trim_editing_changed(self, active: bool) -> None:
        """Refresh the texture route after trim selection changes."""

        del active
        self._sync_generation_texture_context()

    @Slot(object)
    def _handle_surface_generation_context_changed(self, _data: object) -> None:
        """Follow shared-scene surface selection without duplicating its state."""

        self._sync_generation_texture_context()

    def _sync_generation_texture_context(self) -> None:
        """Show the texture action belonging to the active semantic target."""

        trim_is_selected = bool(
            self.object_workspace._architectural_trim_editing_target is not None
        )
        surface_is_selected = bool(
            self.surface_workspace.get_selected_surface_ids()
        )
        use_surface_texture_pipeline = bool(
            surface_is_selected and not trim_is_selected
        )
        self.object_workspace.set_external_surface_generation_target_selected(
            use_surface_texture_pipeline
        )
        target_button = (
            self.surface_workspace.generate_button
            if use_surface_texture_pipeline
            else self.object_workspace.generate_texture_button
        )
        self.generate_texture_stack.setCurrentWidget(target_button)
        self._sync_generation_action_buttons()

    @Slot(str)
    def _handle_door_slot_generation_status_changed(
        self,
        slot_name: str,
    ) -> None:
        """Blink the exact in-flight opening slot beside Generation."""

        self._door_slot_generation_name = str(slot_name).strip()
        if not self._door_slot_generation_name:
            self._door_slot_generation_blink_timer.stop()
            self.door_slot_generation_indicator.hide()
            return
        asset_kind = self.object_workspace.active_opening_component_kind()
        asset_label = "window" if asset_kind == "window" else "door"
        self.door_slot_generation_indicator.setText(
            f"- Generating the {asset_label}'s {self._door_slot_generation_name}"
        )
        self._door_slot_generation_blink_visible = True
        self.door_slot_generation_indicator.setStyleSheet(
            DOOR_SLOT_GENERATION_ACTIVE_STYLE
        )
        self.door_slot_generation_indicator.show()
        self._door_slot_generation_blink_timer.start()

    @Slot()
    def _toggle_door_slot_generation_blink(self) -> None:
        """Alternate red and transparent text without shifting the layout."""

        if not self._door_slot_generation_name:
            return
        self._door_slot_generation_blink_visible = (
            not self._door_slot_generation_blink_visible
        )
        self.door_slot_generation_indicator.setStyleSheet(
            DOOR_SLOT_GENERATION_ACTIVE_STYLE
            if self._door_slot_generation_blink_visible
            else DOOR_SLOT_GENERATION_HIDDEN_STYLE
        )

    @Slot()
    def _handle_video_frame_changed(self) -> None:
        """Keep whole-video analysis running while tracking the displayed frame."""

        self._restore_temporary_reference_after_same_frame_refresh()
        self._sync_object_reference_preview()
        self._sync_ceiling_height_analysis_source()
        self._sync_ceiling_height_dialog_frame()
        self.current_video_frame_changed.emit()
        self.sync_shared_controls()

    @Slot()
    def _show_ceiling_height_analysis_dialog(self) -> None:
        """Show persistent controls while leaving Generation seeking usable."""

        if self._is_shutdown:
            return
        self._sync_ceiling_height_analysis_source()
        video_source = self._current_video_source()
        if video_source is None:
            self.ceiling_height_result_label.setText(
                "Ceiling height: Load a video first."
            )
            return

        dialog = self._ensure_ceiling_height_analysis_dialog()
        metadata = video_source.metadata
        dialog.set_video_metadata(
            metadata.frame_count,
            fps=metadata.fps,
            width=metadata.width,
            height=metadata.height,
        )
        self._sync_ceiling_height_dialog_levels()
        self._sync_ceiling_height_dialog_frame()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _ensure_ceiling_height_analysis_dialog(
        self,
    ) -> CeilingHeightAnalysisDialog:
        dialog = self._ceiling_height_dialog
        if dialog is not None:
            return dialog
        dialog = CeilingHeightAnalysisDialog(
            application_settings=self._application_settings,
            parent=self,
        )
        dialog.analysis_requested.connect(
            self._start_ceiling_height_inference
        )
        dialog.cancellation_requested.connect(
            self._cancel_active_ceiling_height_inference
        )
        self._ceiling_height_dialog = dialog
        self._sync_ceiling_height_dialog_levels()
        return dialog

    @Slot(str, int, int)
    def _start_ceiling_height_inference(
        self,
        model: str,
        frame_interval: int,
        starting_level_index: int,
    ) -> None:
        """Analyze the complete video as one level-aware background job."""

        if self._is_shutdown or self._ceiling_height_runtime is not None:
            return
        dialog = self._ensure_ceiling_height_analysis_dialog()
        video_source = self._current_video_source()
        if video_source is None:
            dialog.set_result("Load a video before starting analysis.", is_error=True)
            return
        level_options = self._ceiling_height_level_options
        if starting_level_index not in dict(level_options):
            dialog.set_result(
                "Choose a valid starting level before analysis.",
                is_error=True,
            )
            return
        metadata = video_source.metadata
        try:
            frame_indices = sample_entire_video_frame_indices(
                metadata.frame_count,
                frame_interval,
            )
        except ValueError as error:
            dialog.set_result(str(error), is_error=True)
            return
        if not frame_indices:
            dialog.set_result("The loaded video contains no frames.", is_error=True)
            return
        estimated_cost = estimate_ceiling_height_api_cost_usd(
            model=model,
            frame_count=metadata.frame_count,
            interval=frame_interval,
            frame_width=metadata.width,
            frame_height=metadata.height,
        )
        if len(frame_indices) > 120:
            request_description = (
                "paid vision requests"
                if ceiling_height_model_requires_api_key(model)
                else "local vision analyses"
            )
            answer = QMessageBox.question(
                dialog,
                "Analyse many video frames?",
                (
                    f"The complete video will run {len(frame_indices):,} "
                    f"{request_description}"
                    f"{_format_cost_confirmation_suffix(estimated_cost)}. "
                    "Continue?"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        uses_local_model = not ceiling_height_model_requires_api_key(model)
        api_key = ""
        if not uses_local_model:
            api_key = (
                self.object_workspace.get_runtime_settings()
                .openai_api_key.strip()
            )
        if not uses_local_model and not api_key:
            dialog.set_result(
                "Add an OpenAI API key in Settings before starting analysis.",
                is_error=True,
            )
            return

        video_path = str(video_source.metadata.path)
        thread = _CeilingHeightInferenceThread(
            video_path,
            frame_indices,
            api_key,
            model,
            self._ceiling_height_estimator,
            level_options,
            starting_level_index,
            parent=self,
        )
        initial_stage = (
            "Loading local Qwen model"
            if uses_local_model
            else f"Analysing 0 of {len(frame_indices)} video frames"
        )
        job = self.job_manager.create_job(
            kind=CEILING_HEIGHT_JOB_KIND,
            requested_name="",
            default_name="Ceiling height",
            stage=initial_stage,
        )
        runtime = _CeilingHeightInferenceRuntime(
            video_path=video_path,
            frame_indices=frame_indices,
            level_signature=level_options,
            starting_level_index=starting_level_index,
            job_id=job.job_id,
            thread=thread,
        )
        self._ceiling_height_runtime = runtime
        self.job_manager.set_cancel_callback(
            job.job_id,
            partial(self._cancel_ceiling_height_inference, job.job_id),
        )
        thread.finished.connect(
            partial(
                self._handle_ceiling_height_inference_finished,
                job.job_id,
                thread,
            )
        )
        thread.progress_changed.connect(
            partial(
                self._handle_ceiling_height_inference_progress,
                job.job_id,
                thread,
            )
        )
        self.ceiling_height_result_label.setText(
            "Ceiling height: Loading local Qwen model..."
            if uses_local_model
            else f"Ceiling height: Analysing {len(frame_indices)} frames..."
        )
        dialog.set_running(True)
        dialog.set_progress(0, len(frame_indices))
        self.sync_shared_controls()
        thread.start()

    @Slot()
    def _cancel_active_ceiling_height_inference(self) -> None:
        runtime = self._ceiling_height_runtime
        if runtime is not None:
            self.job_manager.cancel_job(runtime.job_id)

    def _cancel_ceiling_height_inference(self, job_id: str) -> bool:
        runtime = self._ceiling_height_runtime
        if runtime is None or runtime.job_id != str(job_id):
            return False
        runtime.cancel_requested = True
        runtime.thread.requestInterruption()
        self.sync_shared_controls()
        return True

    def _handle_ceiling_height_inference_progress(
        self,
        job_id: str,
        thread: _CeilingHeightInferenceThread,
        completed_frames: int,
        total_frames: int,
        current_frame: int,
    ) -> None:
        runtime = self._ceiling_height_runtime
        if (
            runtime is None
            or runtime.job_id != job_id
            or runtime.thread is not thread
        ):
            return
        progress = round(
            100.0 * int(completed_frames) / max(int(total_frames), 1)
        )
        self.job_manager.update_job(
            job_id,
            stage=(
                f"Analysing frame {int(current_frame) + 1:,} "
                f"({int(completed_frames):,}/{int(total_frames):,})"
            ),
            progress=progress,
        )
        if self._ceiling_height_dialog is not None:
            self._ceiling_height_dialog.set_progress(
                completed_frames,
                total_frames,
                current_frame=current_frame,
            )

    def _handle_ceiling_height_inference_finished(
        self,
        job_id: str,
        thread: _CeilingHeightInferenceThread,
    ) -> None:
        """Publish per-level consensus results for the source-bound video."""

        runtime = self._ceiling_height_runtime
        try:
            if (
                runtime is None
                or runtime.job_id != job_id
                or runtime.thread is not thread
            ):
                return
            self.job_manager.set_cancel_callback(job_id, None)
            current_source = self._current_video_source()
            levels_are_stale = (
                self._ceiling_height_level_options
                != runtime.level_signature
            )
            is_stale = bool(
                current_source is None
                or str(current_source.metadata.path) != runtime.video_path
                or levels_are_stale
            )
            if (
                self._is_shutdown
                or runtime.cancel_requested
                or thread.was_cancelled
                or is_stale
            ):
                stage = (
                    "Ignored after the video or project levels changed"
                    if is_stale
                    else "Cancelled"
                )
                self.job_manager.mark_cancelled(job_id, stage=stage)
                if not self._is_shutdown:
                    if is_stale and self._ceiling_height_dialog is not None:
                        self._ceiling_height_dialog.set_result(
                            "Analysis stopped because the video or project "
                            "levels changed."
                        )
                    elif not is_stale:
                        self.ceiling_height_result_label.setText(
                            "Ceiling height: Analysis cancelled."
                        )
                        if self._ceiling_height_dialog is not None:
                            self._ceiling_height_dialog.set_result(
                                "Ceiling height analysis cancelled."
                            )
                return
            if thread.error_message is not None or thread.result is None:
                message = thread.error_message or (
                    "Ceiling height analysis returned no result."
                )
                self.job_manager.fail_job(job_id, stage=f"Failed: {message}")
                self.ceiling_height_result_label.setText(
                    f"Ceiling height analysis failed: {message}"
                )
                if self._ceiling_height_dialog is not None:
                    self._ceiling_height_dialog.set_result(
                        message,
                        is_error=True,
                    )
                return

            analysis = thread.result
            if analysis.has_consensus:
                result_text = _format_level_height_result(analysis)
                self.ceiling_height_result_label.setText(result_text)
                if self._ceiling_height_dialog is not None:
                    self._ceiling_height_dialog.set_result(result_text)
                completion_stage = (
                    "Detected ceiling heights for "
                    f"{sum(item.has_consensus for item in analysis.level_estimates)} "
                    "level(s)"
                )
                self.job_manager.complete_job(job_id, stage=completion_stage)
                QTimer.singleShot(
                    0,
                    partial(
                        self.ceiling_height_estimate_changed.emit,
                        analysis,
                    ),
                )
            else:
                result_text = (
                    "No reliable ceiling-height result was found for any "
                    f"detected level. Analysed {analysis.analyzed_count} "
                    "sampled frames"
                    f"{_format_unassigned_frame_suffix(analysis)}."
                )
                self.ceiling_height_result_label.setText(result_text)
                if self._ceiling_height_dialog is not None:
                    self._ceiling_height_dialog.set_result(
                        result_text,
                        is_error=True,
                    )
                self.job_manager.complete_job(
                    job_id,
                    stage="No reliable ceiling-height consensus",
                )
        finally:
            if runtime is not None and runtime.thread is thread:
                self._ceiling_height_runtime = None
            if self._ceiling_height_dialog is not None:
                self._ceiling_height_dialog.set_running(False)
            self.sync_shared_controls()
            thread.deleteLater()

    # ### Ceiling-height analysis state ###
    def set_ceiling_height_levels(
        self,
        levels: Iterable[LevelData],
        current_level_index: int,
    ) -> None:
        """Publish the project levels available to whole-video analysis."""

        options = tuple(
            sorted(
                (
                    (int(level.index), str(level.display_name))
                    for level in levels
                ),
                key=lambda item: item[0],
            )
        )
        self._ceiling_height_level_options = options
        self._ceiling_height_current_level_index = int(current_level_index)
        self._sync_ceiling_height_dialog_levels()

    def _current_video_source(self) -> VideoFrameSource | None:
        return (
            self.object_workspace._video_source
            or self.surface_workspace._video_source
        )

    def _current_video_frame_index(self) -> int | None:
        video_source = self._current_video_source()
        if video_source is None:
            return None
        displayed_index = self.object_workspace._displayed_frame_index
        if displayed_index is None:
            displayed_index = self.surface_workspace._displayed_frame_index
        if displayed_index is None:
            return None
        return min(
            max(int(displayed_index), 0),
            video_source.metadata.frame_count - 1,
        )

    def _sync_ceiling_height_analysis_source(self) -> None:
        video_source = self._current_video_source()
        next_path = (
            "" if video_source is None else str(video_source.metadata.path)
        )
        if next_path != self._ceiling_height_video_path:
            runtime = self._ceiling_height_runtime
            if runtime is not None and runtime.video_path != next_path:
                runtime.cancel_requested = True
                if not self.job_manager.cancel_job(runtime.job_id):
                    runtime.thread.requestInterruption()
            self._ceiling_height_video_path = next_path
            self.ceiling_height_result_label.setText(
                CEILING_HEIGHT_NOT_ESTIMATED_TEXT
            )
        dialog = self._ceiling_height_dialog
        if dialog is None:
            return
        if video_source is None:
            dialog.set_video_metadata(0)
            return
        metadata = video_source.metadata
        dialog.set_video_metadata(
            metadata.frame_count,
            fps=metadata.fps,
            width=metadata.width,
            height=metadata.height,
        )

    def _sync_ceiling_height_dialog_frame(self) -> None:
        if self._ceiling_height_dialog is not None:
            self._ceiling_height_dialog.set_current_frame(
                self._current_video_frame_index()
            )

    def _sync_ceiling_height_dialog_levels(self) -> None:
        if self._ceiling_height_dialog is None:
            return
        self._ceiling_height_dialog.set_level_options(
            tuple(
                (level_name, level_index)
                for level_index, level_name in reversed(
                    self._ceiling_height_level_options
                )
            ),
            current_level_index=self._ceiling_height_current_level_index,
        )

    # ### Temporary generation references ###
    def _remember_temporary_object_reference(
        self,
        reference_bgra: np.ndarray,
        *,
        source_label: str,
    ) -> _TemporaryObjectReferenceSession:
        """Start one external-reference session bound to the current frame."""

        reference = np.ascontiguousarray(reference_bgra).copy()
        objects = self.object_workspace
        session = _TemporaryObjectReferenceSession(
            active_bgra=reference,
            source_label=str(source_label).strip() or "Temporary image",
            generation_data_identity=id(objects._data),
            video_source=objects._video_source,
            frame_index=(
                None
                if objects._video_source is None
                else objects._displayed_frame_index
            ),
        )
        self._temporary_object_reference_session = session
        return session

    def _clear_temporary_object_reference_session(self) -> None:
        """Forget retry input after the user leaves its owning frame or source."""

        self._temporary_object_reference_session = None

    def _temporary_reference_session_matches_context(
        self,
        session: _TemporaryObjectReferenceSession,
    ) -> bool:
        """Return whether a cached external image still belongs to this frame."""

        objects = self.object_workspace
        return bool(
            session.generation_data_identity == id(objects._data)
            and session.video_source is objects._video_source
            and (
                session.video_source is None
                or session.frame_index == objects._displayed_frame_index
            )
        )

    def _adopt_untracked_temporary_reference(
        self,
    ) -> _TemporaryObjectReferenceSession | None:
        """Mirror an externally assigned temporary reference into shared state."""

        reference = self.object_workspace.get_temporary_object_reference()
        if reference is None:
            return None
        session = self._temporary_object_reference_session
        if (
            session is None
            or not self._temporary_reference_session_matches_context(session)
            or not np.array_equal(reference, session.active_bgra)
        ):
            session = self._remember_temporary_object_reference(
                reference,
                source_label="Temporary image",
            )
        return session

    def _restore_temporary_reference_after_same_frame_refresh(self) -> None:
        """Repair shared backend state after a non-navigating frame refresh."""

        session = self._temporary_object_reference_session
        if session is None:
            return
        if not self._temporary_reference_session_matches_context(session):
            self._clear_temporary_object_reference_session()
            return
        if not self.object_workspace.has_temporary_object_reference():
            self.object_workspace.set_temporary_object_reference(
                session.active_bgra
            )
        surface_reference = self.surface_workspace.get_temporary_reference()
        if (
            surface_reference is None
            or not np.array_equal(surface_reference, session.active_bgra)
        ):
            self.surface_workspace.set_temporary_reference(
                session.active_bgra
            )

    def _sync_object_reference_preview(self) -> None:
        """Render the authoritative override, or restore the ordinary frame."""

        temporary_reference = (
            self.object_workspace.get_temporary_object_reference()
        )
        if temporary_reference is not None:
            self._adopt_untracked_temporary_reference()
            self.video_view.set_reference_preview_bgra(temporary_reference)
            return
        self.video_view.clear_reference_preview()

    @Slot()
    def _handle_load_video_clicked(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load generation reference video",
            str(Path.cwd()),
            VIDEO_FILE_FILTER,
        )
        if not file_path:
            return
        try:
            probe_video(file_path)
            self._clear_temporary_object_reference_session()
            self.object_workspace.load_video(file_path)
            self.surface_workspace.load_video(file_path)
            self.sync_shared_controls()
        except (OSError, RuntimeError, ValueError) as error:
            QMessageBox.critical(self, "Video load failed", str(error))

    @Slot()
    def _handle_copy_inpaint_clicked(self) -> None:
        """Copy the exact current Object reference with its alpha mask."""

        reference = (
            self.object_workspace.get_current_object_generation_reference()
        )
        if reference is None:
            QMessageBox.information(
                self,
                "Nothing to copy",
                "Paint an object mask or load a reference image first.",
            )
            return

        image = _bgra_array_to_qimage(reference)
        mime_data = QMimeData()
        mime_data.setImageData(image)
        encoded_png = _encode_qimage_png(image)
        if encoded_png:
            mime_data.setData("image/png", encoded_png)
        QApplication.clipboard().setMimeData(mime_data)
        self.object_workspace.status_label.setText(
            "Object reference copied to the clipboard."
        )
        self.sync_shared_controls()

    @Slot()
    def _handle_paste_inpaint_clicked(self) -> None:
        """Use the clipboard image as a shared generation reference."""

        image = _read_clipboard_image()
        if image is None:
            QMessageBox.warning(
                self,
                "Paste",
                "The clipboard does not contain a usable image.",
            )
            return
        self._activate_temporary_object_reference(
            _qimage_to_bgra_array(image),
            source_label="Clipboard image",
        )

    @Slot()
    def _handle_load_reference_image_clicked(self) -> None:
        """Load a shared temporary reference without replacing the video."""

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load generation reference image",
            str(Path.cwd()),
            REFERENCE_IMAGE_FILE_FILTER,
        )
        if not file_path:
            return
        image = QImage(file_path)
        if image.isNull():
            QMessageBox.critical(
                self,
                "Image load failed",
                "The selected file could not be decoded as an image.",
            )
            return
        self._activate_temporary_object_reference(
            _qimage_to_bgra_array(image),
            source_label="Loaded image",
        )

    def _activate_temporary_object_reference(
        self,
        reference_bgra: np.ndarray,
        *,
        source_label: str,
    ) -> None:
        """Display and bind one temporary reference to both generators."""

        try:
            self.object_workspace.set_temporary_object_reference(
                reference_bgra
            )
            self.surface_workspace.set_temporary_reference(reference_bgra)
        except ValueError as error:
            self.object_workspace.clear_temporary_object_reference()
            self.surface_workspace.clear_temporary_reference()
            QMessageBox.critical(self, "Reference image failed", str(error))
            return
        active_reference = (
            self.object_workspace.get_temporary_object_reference()
        )
        if active_reference is None:
            return
        self._remember_temporary_object_reference(
            active_reference,
            source_label=source_label,
        )
        self._sync_object_reference_preview()
        message = (
            f"{source_label} is active for Object and Surface generation. Move the "
            "video seekbar to return to the video."
            if self.object_workspace._video_source is not None
            else f"{source_label} is active for Object and Surface generation."
        )
        self.object_workspace.status_label.setText(message)
        self.sync_shared_controls()

    @Slot(int)
    def _handle_seekbar_changed(self, frame_index: int) -> None:
        """Seek once without letting one backend overwrite the other's old frame."""

        surface = self.surface_workspace
        objects = self.object_workspace
        if objects._is_syncing_seekbar or surface._is_syncing_seekbar:
            return
        video_source = objects._video_source or surface._video_source
        if video_source is None:
            return

        current_strokes = self.video_view.get_strokes()
        for workspace in (objects, surface):
            displayed_index = workspace._displayed_frame_index
            if displayed_index is None or workspace._video_source is None:
                continue
            workspace._data.set_frame_strokes(displayed_index, current_strokes)

        safe_index = min(
            max(int(frame_index), 0),
            max(video_source.metadata.frame_count - 1, 0),
        )
        try:
            frame_bgr = video_source.get_frame(safe_index)
        except (IndexError, ValueError) as error:
            objects.status_label.setText(str(error))
            surface.status_label.setText(str(error))
            return
        target_strokes = objects._data.strokes_for_frame(safe_index)
        if not target_strokes:
            target_strokes = surface._data.strokes_for_frame(safe_index)
        current_object_frame_index = objects._displayed_frame_index
        object_frame_is_changing = (
            current_object_frame_index is None
            or int(current_object_frame_index) != safe_index
        )
        if object_frame_is_changing:
            objects.clear_door_slot_editing_target()
            self._clear_temporary_object_reference_session()
            objects.clear_temporary_object_reference_for_video_navigation()
            surface.clear_temporary_reference()
        for workspace in (objects, surface):
            workspace._data.current_frame_index = safe_index
            workspace._displayed_frame_index = safe_index
        self.video_view.set_frame(frame_bgr, target_strokes)
        surface._sync_painted_frames_label()
        objects._sync_controls()
        surface._sync_controls()
        self.sync_shared_controls()

    @Slot(object)
    def _handle_job_changed(self, _job: object) -> None:
        self._sync_generation_action_buttons()

    @Slot()
    def sync_shared_controls(self) -> None:
        """Own availability for widgets consumed by both pipelines."""

        objects = self.object_workspace
        surface = self.surface_workspace
        has_video = bool(
            objects._video_source is not None or surface._video_source is not None
        )
        has_untracked_object_job = bool(
            objects._generation_thread is not None and not objects._object_job_runtimes
        )
        shared_reference_controls_are_available = not has_untracked_object_job
        video_editor_is_available = (
            has_video and shared_reference_controls_are_available
        )
        temporary_reference_is_available = (
            objects.has_temporary_object_reference()
        )
        inpainting_is_available = (
            video_editor_is_available
            and not temporary_reference_is_available
        )
        reference_media_changes_are_available = not has_untracked_object_job
        self.load_video_button.setEnabled(reference_media_changes_are_available)
        self.load_reference_image_button.setEnabled(
            reference_media_changes_are_available
        )
        self.paste_inpaint_button.setEnabled(
            reference_media_changes_are_available
            and _clipboard_has_usable_image()
        )
        self.copy_inpaint_button.setEnabled(
            objects.has_temporary_object_reference()
            or self.video_view.has_selection()
        )
        self.seekbar.setEnabled(video_editor_is_available)
        self.paint_mask_button.setEnabled(inpainting_is_available)
        self.erase_mask_button.setEnabled(inpainting_is_available)
        self.brush_size_spinbox.setEnabled(inpainting_is_available)
        self.clear_mask_button.setEnabled(
            inpainting_is_available and self.video_view.has_selection()
        )
        self.pbr_map_control.setEnabled(not has_untracked_object_job)
        self.ai_prompt_edit.setEnabled(not has_untracked_object_job)
        self.video_view.set_interaction_enabled(inpainting_is_available)
        self.infer_ceiling_height_button.setEnabled(
            not self._is_shutdown
            and not self._door_slot_editing_active
            and has_video
        )
        self.infer_ceiling_height_button.setText("Infer ceiling height")
        self._sync_ceiling_height_analysis_source()
        self._sync_ceiling_height_dialog_frame()
        self._sync_opening_generation_sections()
        self._sync_generation_texture_context()


# ### Reference image helpers ###
def _bgra_array_to_qimage(image_bgra: np.ndarray) -> QImage:
    """Build an owned Qt image while preserving BGRA alpha exactly."""

    image = np.asarray(image_bgra)
    if image.ndim != 3 or image.shape[2] != 4 or image.size == 0:
        return QImage()
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    rgba = np.ascontiguousarray(image[:, :, (2, 1, 0, 3)])
    height, width = rgba.shape[:2]
    return QImage(
        rgba.data,
        width,
        height,
        int(rgba.strides[0]),
        QImage.Format.Format_RGBA8888,
    ).copy()


def _qimage_to_bgra_array(image: QImage) -> np.ndarray:
    """Normalize any decoded Qt image to an owned contiguous BGRA array."""

    if image.isNull() or image.width() <= 0 or image.height() <= 0:
        raise ValueError("The reference image is empty.")
    converted = image.convertToFormat(QImage.Format.Format_RGBA8888)
    height = converted.height()
    width = converted.width()
    bytes_per_line = converted.bytesPerLine()
    rgba_rows = np.frombuffer(
        converted.constBits(),
        dtype=np.uint8,
        count=height * bytes_per_line,
    ).reshape(height, bytes_per_line)
    rgba = rgba_rows[:, : width * 4].reshape(height, width, 4)
    return np.ascontiguousarray(rgba[:, :, (2, 1, 0, 3)])


def _encode_qimage_png(image: QImage) -> QByteArray:
    """Encode a clipboard image explicitly so external apps retain alpha."""

    encoded = QByteArray()
    buffer = QBuffer(encoded)
    if not buffer.open(QIODevice.OpenModeFlag.WriteOnly):
        return QByteArray()
    try:
        if not image.save(buffer, "PNG"):
            return QByteArray()
    finally:
        buffer.close()
    return encoded


def _read_clipboard_image() -> QImage | None:
    """Prefer lossless PNG clipboard data, then use Qt's image fallback."""

    clipboard = QApplication.clipboard()
    mime_data = clipboard.mimeData()
    if mime_data is not None and mime_data.hasFormat("image/png"):
        image = QImage.fromData(mime_data.data("image/png"), "PNG")
        if not image.isNull():
            return image
    image = clipboard.image()
    return None if image.isNull() else image


def _clipboard_has_usable_image() -> bool:
    """Return whether Paste can resolve an image right now."""

    clipboard = QApplication.clipboard()
    mime_data = clipboard.mimeData()
    return bool(
        (
            mime_data is not None
            and (
                mime_data.hasImage()
                or mime_data.hasFormat("image/png")
            )
        )
        or not clipboard.image().isNull()
    )


# ### Widget helpers ###
def _build_labeled_view(title: str, view: QWidget) -> QWidget:
    wrapper = QWidget()
    layout = QVBoxLayout(wrapper)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    title_label = QLabel(title)
    title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    title_label.setWordWrap(True)
    title_label.setStyleSheet("font-weight: 600;")
    layout.addWidget(title_label)
    layout.addWidget(view, 1)
    return wrapper


def _build_controls_section(kind: str) -> tuple[QWidget, QGridLayout]:
    section = QWidget()
    section.setObjectName(f"merged_generation_{kind}_controls")
    outer_layout = QVBoxLayout(section)
    outer_layout.setContentsMargins(0, 0, 0, 0)
    outer_layout.setSpacing(5)

    fields = QWidget()
    fields.setObjectName(f"merged_generation_{kind}_fields")
    fields_layout = QGridLayout(fields)
    fields_layout.setContentsMargins(0, 0, 0, 0)
    fields_layout.setHorizontalSpacing(8)
    fields_layout.setVerticalSpacing(5)
    outer_layout.addWidget(fields, 1)
    return section, fields_layout


def _build_controls_column(object_name: str) -> tuple[QWidget, QVBoxLayout]:
    column = QWidget()
    column.setObjectName(object_name)
    layout = QVBoxLayout(column)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(5)
    return column, layout


def _build_boxed_section(
    title: str,
    object_name: str,
) -> tuple[QGroupBox, QVBoxLayout]:
    section = QGroupBox(title)
    section.setObjectName(object_name)
    layout = QVBoxLayout(section)
    layout.setContentsMargins(7, 7, 7, 7)
    layout.setSpacing(5)
    return section, layout


def _build_labeled_inline_control(label: str, control: QWidget) -> QWidget:
    wrapper = QWidget()
    layout = QHBoxLayout(wrapper)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(5)
    layout.addWidget(QLabel(label))
    layout.addWidget(control, 1)
    return wrapper


def _unique_widgets(widgets: Iterable[QWidget]) -> tuple[QWidget, ...]:
    unique: list[QWidget] = []
    seen_ids: set[int] = set()
    for widget in widgets:
        widget_id = id(widget)
        if widget_id in seen_ids:
            continue
        seen_ids.add(widget_id)
        unique.append(widget)
    return tuple(unique)


# ### Legacy project migration helpers ###
def _select_richest_video_state(
    object_data: GenerationData,
    surface_data: SurfaceTextureData,
) -> tuple[VideoMetadata, int, dict[int, list[MaskStroke]]] | None:
    candidates = []
    for preference, data in enumerate((surface_data, object_data)):
        metadata = data.video_metadata
        if metadata is None:
            continue
        stroke_count = sum(len(strokes) for strokes in data.frame_strokes.values())
        candidates.append(
            (
                len(data.frame_strokes),
                stroke_count,
                preference,
                metadata,
                data.current_frame_index,
                data.frame_strokes,
            )
        )
    if not candidates:
        return None
    _, _, _, metadata, frame_index, frame_strokes = max(
        candidates,
        key=lambda value: value[:3],
    )
    return metadata, int(frame_index), frame_strokes


def _merge_legacy_frame_strokes(
    primary: dict[int, list[MaskStroke]],
    *sources: dict[int, list[MaskStroke]],
) -> dict[int, list[MaskStroke]]:
    """Preserve disjoint legacy masks when both tabs used the same video."""

    merged: dict[int, list[MaskStroke]] = {}
    frame_indices = sorted(
        {frame_index for source in (primary, *sources) for frame_index in source}
    )
    for frame_index in frame_indices:
        strokes = [copy.deepcopy(stroke) for stroke in primary.get(frame_index, ())]
        retained_counts = Counter(strokes)
        for source in sources:
            source_counts: Counter[MaskStroke] = Counter()
            for stroke in source.get(frame_index, ()):
                source_counts[stroke] += 1
                if source_counts[stroke] <= retained_counts[stroke]:
                    continue
                retained_counts[stroke] += 1
                strokes.append(copy.deepcopy(stroke))
                if len(strokes) >= MAX_MASK_STROKES_PER_FRAME:
                    break
            if len(strokes) >= MAX_MASK_STROKES_PER_FRAME:
                break
        if strokes:
            merged[frame_index] = strokes
    return merged
