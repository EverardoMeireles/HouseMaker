# ### Imports ###
from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
from PySide6.QtCore import (
    QBuffer,
    QByteArray,
    QIODevice,
    QMimeData,
    QRect,
    Qt,
    QThread,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QColor,
    QImage,
    QKeySequence,
    QPainter,
    QPaintEvent,
    QPen,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from housemaker.ceiling_height_estimation import (
    CEILING_HEIGHT_JOB_KIND,
    CeilingHeightEstimate,
    CeilingHeightEstimationCancelled,
    CeilingHeightEstimationError,
    CeilingHeightEstimator,
    format_ceiling_height_estimate,
    infer_ceiling_height,
)
from housemaker.generation_jobs import (
    JOB_STATUS_RUNNING,
    GenerationJob,
    GenerationJobManager,
)
from housemaker.generation_state import (
    MAX_MASK_STROKES_PER_FRAME,
    GenerationData,
    MaskStroke,
)
from housemaker.generation_workspace import (
    GENERATION_JOB_KIND_FACE_EDIT,
    GENERATION_JOB_KIND_MODEL,
    GENERATION_JOB_KIND_TEXTURE,
    GenerationWorkspace,
)
from housemaker.models import TRIM_KIND_CORNICE
from housemaker.object_reference_editing import (
    DEFAULT_OBJECT_REFERENCE_EDIT_MODEL,
    OBJECT_REFERENCE_EDIT_MODEL_OPTIONS,
    OBJECT_REFERENCE_EDIT_OUTPUT_SHAPE,
    SUPPORTED_OBJECT_REFERENCE_EDIT_MODELS,
    ObjectReferenceEditingCancelled,
    ObjectReferenceEditingError,
    edit_object_reference,
    object_reference_edit_model_requires_api_key,
)
from housemaker.settings_widget import DEFAULT_CLEAR_MASK_HOTKEY
from housemaker.surface_texture_state import SurfaceTextureData
from housemaker.surface_texture_workspace import (
    SURFACE_TEXTURE_JOB_KIND,
    SurfaceTextureGenerationWorkspace,
)
from housemaker.video_source import VIDEO_FILE_FILTER, VideoMetadata, probe_video

# ### Constants ###
SURFACE_OUTLINE_COLOR = QColor("#388cff")
OBJECT_OUTLINE_COLOR = QColor("#34b878")
WORKFLOW_OUTLINE_WIDTH = 2
WORKFLOW_OUTLINE_INSET = 3
OBJECT_WORKFLOW_OUTLINE_PADDING = 5
WORKFLOW_SECTION_SPACING = 10
CEILING_HEIGHT_SHUTDOWN_WAIT_MILLISECONDS = 100
CEILING_HEIGHT_NOT_ESTIMATED_TEXT = "Ceiling height: Not estimated"
OBJECT_REFERENCE_EDIT_JOB_KIND = "object_reference_edit"
DOOR_SLOT_GENERATION_BLINK_INTERVAL_MILLISECONDS = 500
DOOR_SLOT_GENERATION_ACTIVE_STYLE = "color: #ff3030; font-weight: 700;"
DOOR_SLOT_GENERATION_HIDDEN_STYLE = (
    "color: rgba(255, 48, 48, 0); font-weight: 700;"
)
OBJECT_REFERENCE_EDIT_STATUS_EMPTY = (
    "Select one object, describe the change, then click Edit reference."
)
OBJECT_REFERENCE_EDIT_SOURCE_MASK = "mask"
OBJECT_REFERENCE_EDIT_SOURCE_TEMPORARY = "temporary"
ARCHITECTURAL_TRIM_REFERENCE_EDIT_INSTRUCTION = (
    "Isolate one complete repeating cornice or moulding module. Straighten it "
    "into an orthographic horizontal view. Remove the wall, ceiling, room, "
    "scenery, and perspective distortion. Preserve the module's relief, "
    "profile, proportions, material, and surface detail."
)
OBJECT_REFERENCE_EDIT_PROMPT_PRESETS = (
    "show me a front of it",
    "slightly rotated to the left",
    "slightly rotated to the right",
    "slightly isometric",
    "top-down view",
)
REFERENCE_IMAGE_FILE_FILTER = (
    "Image files (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff);;"
    "All files (*.*)"
)
CANCELLABLE_GENERATION_JOB_KINDS = frozenset(
    {
        GENERATION_JOB_KIND_MODEL,
        GENERATION_JOB_KIND_TEXTURE,
        GENERATION_JOB_KIND_FACE_EDIT,
        SURFACE_TEXTURE_JOB_KIND,
        CEILING_HEIGHT_JOB_KIND,
        OBJECT_REFERENCE_EDIT_JOB_KIND,
    }
)


# ### Ceiling-height inference jobs ###
class _CeilingHeightInferenceThread(QThread):
    """Analyze one immutable Generation frame outside the GUI thread."""

    def __init__(
        self,
        frame_bgr: np.ndarray,
        api_key: str,
        estimator: CeilingHeightEstimator,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._frame_bgr = np.ascontiguousarray(frame_bgr).copy()
        self._api_key = str(api_key)
        self._estimator = estimator
        self.result: CeilingHeightEstimate | None = None
        self.error_message: str | None = None
        self.was_cancelled = False

    def run(self) -> None:  # type: ignore[override]
        try:
            try:
                result = self._estimator(
                    self._frame_bgr,
                    api_key=self._api_key,
                    cancellation_check=self.isInterruptionRequested,
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
            self._frame_bgr = np.empty((0, 0, 3), dtype=np.uint8)


@dataclass
class _CeilingHeightInferenceRuntime:
    """GUI-owned lifecycle and stale-frame guard for one inference."""

    frame_revision: int
    job_id: str
    thread: _CeilingHeightInferenceThread
    cancel_requested: bool = False


# ### Object-reference editing jobs ###
ObjectReferenceEditor = Callable[..., np.ndarray]


class _ObjectReferenceEditThread(QThread):
    """Generate one immutable square reference away from the GUI thread."""

    def __init__(
        self,
        source_bgra: np.ndarray,
        prompt: str,
        model: str,
        api_key: str,
        editor: ObjectReferenceEditor,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._source_bgra = np.ascontiguousarray(source_bgra).copy()
        self._prompt = str(prompt)
        self._model = str(model)
        self._api_key = str(api_key)
        self._editor = editor
        self.result: np.ndarray | None = None
        self.error_message: str | None = None
        self.was_cancelled = False

    def run(self) -> None:  # type: ignore[override]
        try:
            try:
                result = self._editor(
                    self._source_bgra,
                    self._prompt,
                    api_key=self._api_key,
                    model=self._model,
                    cancellation_check=self.isInterruptionRequested,
                )
            except ObjectReferenceEditingCancelled:
                self.was_cancelled = True
                return
            except ObjectReferenceEditingError as error:
                if self.isInterruptionRequested():
                    self.was_cancelled = True
                else:
                    self.error_message = str(error)
                return
            except Exception:  # noqa: BLE001 - redact worker/provider internals.
                if self.isInterruptionRequested():
                    self.was_cancelled = True
                else:
                    self.error_message = "Object reference editing failed."
                return
            if self.isInterruptionRequested():
                self.was_cancelled = True
                return
            prepared_result = np.asarray(result)
            if (
                prepared_result.dtype != np.uint8
                or prepared_result.shape != OBJECT_REFERENCE_EDIT_OUTPUT_SHAPE
                or prepared_result.ndim != 3
                or prepared_result.shape[2] != 4
                or not np.any(prepared_result[:, :, 3] > 0)
            ):
                self.error_message = (
                    "Object reference editing returned an invalid image."
                )
                return
            self.result = np.ascontiguousarray(prepared_result).copy()
        finally:
            self._api_key = ""
            self._model = ""
            self._prompt = ""
            self._source_bgra = np.empty((0, 0, 4), dtype=np.uint8)


@dataclass
class _ObjectReferenceEditRuntime:
    """GUI-owned lifecycle and stale-input guard for one reference edit."""

    signature: tuple[object, ...]
    source_kind: str
    model: str
    prompt: str
    job_id: str
    thread: _ObjectReferenceEditThread
    cancel_requested: bool = False


@dataclass
class _TemporaryObjectReferenceSession:
    """Keep one external reference coherent across same-frame refreshes."""

    source_bgra: np.ndarray
    active_bgra: np.ndarray
    source_label: str
    revision: int
    generation_data_identity: int
    video_source: object | None
    frame_index: int | None


# ### Overlapping workflow outlines ###
class _WorkflowControlsRow(QWidget):
    """Draw Surface and Object workflow bounds around two visible columns."""

    def __init__(
        self,
        shared_controls: QWidget,
        object_controls: QWidget,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("merged_generation_controls_row")
        self._shared_controls = shared_controls
        self._object_controls = object_controls

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(WORKFLOW_SECTION_SPACING)
        layout.addWidget(shared_controls, 1)
        layout.addWidget(object_controls, 2)

    def paintEvent(self, event: QPaintEvent) -> None:  # type: ignore[override]
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        blue_bounds = self._shared_controls.geometry().adjusted(
            -WORKFLOW_OUTLINE_INSET,
            -WORKFLOW_OUTLINE_INSET,
            WORKFLOW_OUTLINE_INSET,
            WORKFLOW_OUTLINE_INSET,
        )
        painter.setPen(QPen(SURFACE_OUTLINE_COLOR, WORKFLOW_OUTLINE_WIDTH))
        painter.drawRoundedRect(blue_bounds, 5, 5)

        green_bounds = _combined_widget_bounds(
            self._shared_controls,
            self._object_controls,
        ).adjusted(
            -WORKFLOW_OUTLINE_INSET + OBJECT_WORKFLOW_OUTLINE_PADDING,
            -WORKFLOW_OUTLINE_INSET + OBJECT_WORKFLOW_OUTLINE_PADDING,
            WORKFLOW_OUTLINE_INSET - OBJECT_WORKFLOW_OUTLINE_PADDING,
            WORKFLOW_OUTLINE_INSET - OBJECT_WORKFLOW_OUTLINE_PADDING,
        )
        painter.setPen(QPen(OBJECT_OUTLINE_COLOR, WORKFLOW_OUTLINE_WIDTH))
        painter.drawRoundedRect(green_bounds, 5, 5)


# ### Merged generation workspace ###
class MergedGenerationWorkspace(QWidget):
    """Present Surface and Object generation around one reference editor."""

    current_video_frame_changed = Signal()
    ceiling_height_estimate_changed = Signal(object)

    def __init__(
        self,
        surface_workspace: SurfaceTextureGenerationWorkspace,
        object_workspace: GenerationWorkspace,
        job_manager: GenerationJobManager,
        parent: QWidget | None = None,
        *,
        ceiling_height_estimator: CeilingHeightEstimator | None = None,
        object_reference_editor: ObjectReferenceEditor | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("merged_generation_workspace")
        self.surface_workspace = surface_workspace
        self.object_workspace = object_workspace
        self.job_manager = job_manager
        self._ceiling_height_estimator = (
            infer_ceiling_height
            if ceiling_height_estimator is None
            else ceiling_height_estimator
        )
        self._ceiling_height_frame_revision = 0
        self._ceiling_height_runtime: _CeilingHeightInferenceRuntime | None = None
        self._object_reference_editor = (
            edit_object_reference
            if object_reference_editor is None
            else object_reference_editor
        )
        self._object_reference_edit_runtime: (
            _ObjectReferenceEditRuntime | None
        ) = None
        self._temporary_object_reference_session: (
            _TemporaryObjectReferenceSession | None
        ) = None
        self._temporary_object_reference_revision = 0
        self._door_slot_editing_active = False
        self._door_slot_generation_name = ""
        self._door_slot_generation_blink_visible = True
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
        runtimes = tuple(
            runtime
            for runtime in (
                self._ceiling_height_runtime,
                self._object_reference_edit_runtime,
            )
            if runtime is not None
        )
        for runtime in runtimes:
            runtime.cancel_requested = True
            runtime.thread.requestInterruption()
            self.job_manager.set_cancel_callback(runtime.job_id, None)
            self.job_manager.mark_cancelled(
                runtime.job_id,
                stage="Cancelled during shutdown",
            )
        for runtime in runtimes:
            while runtime.thread.isRunning():
                runtime.thread.wait(CEILING_HEIGHT_SHUTDOWN_WAIT_MILLISECONDS)
            runtime.thread.deleteLater()
        self._ceiling_height_runtime = None
        self._object_reference_edit_runtime = None

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
        """Migrate two legacy reference editors into one deterministic state."""

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

    @Slot()
    def cancel_latest_generation_job(self) -> bool:
        """Cancel only the newest active job represented by the shared controls."""

        job = self._latest_cancellable_job()
        if job is None:
            self._sync_cancel_button()
            return False
        accepted = self.job_manager.cancel_job(job.job_id)
        self._sync_cancel_button()
        return accepted

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
        self.cancel_button = objects.cancel_operation_button

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
        self.controls_row = _WorkflowControlsRow(
            self.shared_controls,
            self.object_controls,
        )
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
        section, layout = _build_controls_section("Shared", "shared")

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
            "Estimate floor-to-ceiling height from the current video frame "
            "using visible perspective and scale cues."
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

        self.cancel_button.setObjectName("cancel_generation_job_button")
        self.cancel_button.setText("Cancel")
        self.cancel_button.setToolTip(
            "Cancel the newest active generation or frame-analysis job."
        )
        material_layout.addWidget(self.cancel_button)
        shared_layout.addWidget(self.shared_material_section)
        shared_layout.addStretch(1)

        layout.addWidget(shared_column, 0, 0)
        layout.setColumnStretch(0, 1)
        return section

    def _build_object_controls(self) -> QWidget:
        objects = self.object_workspace
        section, layout = _build_controls_section(
            "Object generation",
            "object",
        )

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

        self.reference_editing_section, reference_edit_layout = (
            _build_boxed_section(
                "Reference editing",
                "merged_generation_object_reference_editing_section",
            )
        )
        self.reference_edit_model_combo = QComboBox()
        self.reference_edit_model_combo.setObjectName(
            "object_reference_edit_model"
        )
        for label, model_id in OBJECT_REFERENCE_EDIT_MODEL_OPTIONS:
            self.reference_edit_model_combo.addItem(label, model_id)
        default_model_index = self.reference_edit_model_combo.findData(
            DEFAULT_OBJECT_REFERENCE_EDIT_MODEL
        )
        self.reference_edit_model_combo.setCurrentIndex(default_model_index)
        self.reference_edit_model_combo.setToolTip(
            "Choose the AI model used to edit the selected object. Qwen runs "
            "locally; the other choices use the configured OpenAI API key."
        )
        reference_edit_layout.addWidget(
            _build_labeled_inline_control(
                "Model",
                self.reference_edit_model_combo,
            )
        )
        self.reference_edit_prompt_preset_combo = QComboBox()
        self.reference_edit_prompt_preset_combo.setObjectName(
            "object_reference_edit_prompt_preset"
        )
        self.reference_edit_prompt_preset_combo.addItem(
            "Append view instruction...",
            None,
        )
        for preset in OBJECT_REFERENCE_EDIT_PROMPT_PRESETS:
            self.reference_edit_prompt_preset_combo.addItem(preset, preset)
        self.reference_edit_prompt_preset_combo.setCurrentIndex(0)
        self.reference_edit_prompt_preset_combo.setToolTip(
            "Append a reusable camera-view instruction to the edit prompt."
        )
        reference_edit_layout.addWidget(
            _build_labeled_inline_control(
                "View preset",
                self.reference_edit_prompt_preset_combo,
            )
        )
        self.reference_edit_prompt = QLineEdit()
        self.reference_edit_prompt.setObjectName(
            "object_reference_edit_prompt"
        )
        self.reference_edit_prompt.setPlaceholderText(
            "Example: Remove the cover and reconstruct a bare wooden tabletop"
        )
        self.reference_edit_prompt.setMaxLength(2_000)
        self.reference_edit_prompt.setClearButtonEnabled(True)
        reference_edit_layout.addWidget(
            _build_labeled_inline_control(
                "Edit instruction",
                self.reference_edit_prompt,
            )
        )

        self.edit_reference_button = QPushButton("Edit reference")
        self.edit_reference_button.setObjectName(
            "edit_object_reference_button"
        )
        self.edit_reference_button.setToolTip(
            "Use the current object mask and edit instruction to generate a "
            "new 1024x1024 AI object reference and use it automatically. "
            "Click again to retry, or move the frame slider to undo."
        )
        reference_edit_layout.addWidget(self.edit_reference_button)
        self.reference_edit_status_label = QLabel(
            OBJECT_REFERENCE_EDIT_STATUS_EMPTY
        )
        self.reference_edit_status_label.setObjectName(
            "object_reference_edit_status_label"
        )
        self.reference_edit_status_label.setWordWrap(True)
        reference_edit_layout.addWidget(self.reference_edit_status_label)
        primary_layout.addWidget(self.reference_editing_section)

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
        generation_settings_layout.addWidget(objects.symmetric_division_checkbox)
        generation_settings_layout.addWidget(
            objects.side_door_duplication_checkbox
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
            self._start_ceiling_height_inference
        )
        self.edit_reference_button.clicked.connect(
            self._start_object_reference_edit
        )
        self.reference_edit_prompt_preset_combo.activated.connect(
            self._append_reference_edit_prompt_preset
        )
        self.reference_edit_prompt.textChanged.connect(
            self._handle_reference_edit_inputs_changed
        )
        self.reference_edit_model_combo.currentTextChanged.connect(
            self._handle_reference_edit_inputs_changed
        )

        self.video_view.strokes_changed.connect(surface._handle_video_strokes_changed)
        self.video_view.strokes_changed.connect(self.sync_shared_controls)
        self.video_view.strokes_changed.connect(
            self._handle_reference_edit_inputs_changed
        )
        self.video_view.frame_changed.connect(self._handle_video_frame_changed)
        objects.reference_edit_state_changed.connect(
            self._handle_object_reference_state_changed
        )
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
        try:
            self.seekbar.valueChanged.disconnect()
        except (RuntimeError, TypeError):
            pass
        self.seekbar.valueChanged.connect(self._handle_seekbar_changed)
        for checkbox in self.pbr_map_checkboxes.values():
            checkbox.toggled.connect(surface._handle_pbr_map_toggled)

        try:
            self.cancel_button.clicked.disconnect()
        except (RuntimeError, TypeError):
            pass
        self.cancel_button.clicked.connect(self.cancel_latest_generation_job)
        self.job_manager.job_added.connect(self._handle_job_changed)
        self.job_manager.job_updated.connect(self._handle_job_changed)
        self.job_manager.jobs_cleared.connect(self._sync_cancel_button)

        objects._sync_controls()
        surface._sync_controls()
        self._sync_generation_texture_context()

    @Slot(bool)
    def _handle_door_slot_editing_changed(self, active: bool) -> None:
        """Keep shared analysis controls out of the door-slot workflow."""

        self._door_slot_editing_active = bool(active)
        self.sync_shared_controls()

    @Slot(bool)
    def _handle_architectural_trim_editing_changed(self, active: bool) -> None:
        """Offer a repeat-module edit instruction without replacing user text."""

        if bool(active) and not self.reference_edit_prompt.text().strip():
            self.reference_edit_prompt.setText(
                ARCHITECTURAL_TRIM_REFERENCE_EDIT_INSTRUCTION
            )
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
        trim_target = self.object_workspace._architectural_trim_editing_target
        cornice_is_selected = bool(
            trim_target is not None and trim_target.trim_kind == TRIM_KIND_CORNICE
        )
        surface_is_selected = bool(
            self.surface_workspace.get_selected_surface_ids()
        )
        use_surface_texture_pipeline = bool(
            surface_is_selected and (not trim_is_selected or not cornice_is_selected)
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

    @Slot()
    def _handle_object_reference_state_changed(self) -> None:
        """Keep the visible preview identical to the active generation input."""

        self._sync_object_reference_preview()
        self.sync_shared_controls()

    @Slot(str)
    def _handle_door_slot_generation_status_changed(
        self,
        slot_name: str,
    ) -> None:
        """Blink the exact in-flight door slot beside the Generation heading."""

        self._door_slot_generation_name = str(slot_name).strip()
        if not self._door_slot_generation_name:
            self._door_slot_generation_blink_timer.stop()
            self.door_slot_generation_indicator.hide()
            return
        self.door_slot_generation_indicator.setText(
            f"- Generating the door's {self._door_slot_generation_name}"
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
        """Invalidate estimates and preserve external current-frame updates."""

        self._restore_temporary_reference_after_same_frame_refresh()
        self._sync_object_reference_preview()
        self._ceiling_height_frame_revision += 1
        reference_runtime = self._object_reference_edit_runtime
        if (
            reference_runtime is not None
            and self._get_current_reference_edit_signature(
                reference_runtime.source_kind
            )
            != reference_runtime.signature
        ):
            reference_runtime.cancel_requested = True
            if not self.job_manager.cancel_job(reference_runtime.job_id):
                reference_runtime.thread.requestInterruption()
        runtime = self._ceiling_height_runtime
        if runtime is not None:
            runtime.cancel_requested = True
            if not self.job_manager.cancel_job(runtime.job_id):
                runtime.thread.requestInterruption()
        self.ceiling_height_result_label.setText(
            CEILING_HEIGHT_NOT_ESTIMATED_TEXT
        )
        self._sync_object_reference_status()
        self.current_video_frame_changed.emit()
        self.sync_shared_controls()

    @Slot()
    def _start_ceiling_height_inference(self) -> None:
        """Snapshot and analyze the exact frame currently shown to the user."""

        if self._is_shutdown or self._ceiling_height_runtime is not None:
            return
        frame_bgr = self.get_current_video_frame_bgr()
        if frame_bgr is None:
            self.ceiling_height_result_label.setText(
                "Ceiling height: Load a video frame first."
            )
            self.sync_shared_controls()
            return
        api_key = (
            self.object_workspace.get_runtime_settings().openai_api_key.strip()
        )
        if not api_key:
            self.ceiling_height_result_label.setText(
                "Ceiling height: Add an OpenAI API key in Settings first."
            )
            return

        frame_revision = self._ceiling_height_frame_revision
        thread = _CeilingHeightInferenceThread(
            frame_bgr,
            api_key,
            self._ceiling_height_estimator,
            parent=self,
        )
        job = self.job_manager.create_job(
            kind=CEILING_HEIGHT_JOB_KIND,
            requested_name="",
            default_name="Ceiling height",
            stage="Analyzing current video frame",
        )
        runtime = _CeilingHeightInferenceRuntime(
            frame_revision=frame_revision,
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
        self.ceiling_height_result_label.setText(
            "Ceiling height: Estimating..."
        )
        self.sync_shared_controls()
        thread.start()

    def _cancel_ceiling_height_inference(self, job_id: str) -> bool:
        runtime = self._ceiling_height_runtime
        if runtime is None or runtime.job_id != str(job_id):
            return False
        runtime.cancel_requested = True
        runtime.thread.requestInterruption()
        self.sync_shared_controls()
        return True

    def _handle_ceiling_height_inference_finished(
        self,
        job_id: str,
        thread: _CeilingHeightInferenceThread,
    ) -> None:
        """Publish a result only if the displayed frame is still identical."""

        runtime = self._ceiling_height_runtime
        try:
            if (
                runtime is None
                or runtime.job_id != job_id
                or runtime.thread is not thread
            ):
                return
            self.job_manager.set_cancel_callback(job_id, None)
            is_stale = (
                runtime.frame_revision != self._ceiling_height_frame_revision
            )
            if (
                self._is_shutdown
                or runtime.cancel_requested
                or thread.was_cancelled
                or is_stale
            ):
                stage = (
                    "Ignored after the video frame changed"
                    if is_stale
                    else "Cancelled"
                )
                self.job_manager.mark_cancelled(job_id, stage=stage)
                if not self._is_shutdown and not is_stale:
                    self.ceiling_height_result_label.setText(
                        "Ceiling height: Inference cancelled."
                    )
                return
            if thread.error_message is not None or thread.result is None:
                message = thread.error_message or (
                    "Ceiling height inference returned no result."
                )
                self.job_manager.fail_job(job_id, stage=f"Failed: {message}")
                self.ceiling_height_result_label.setText(
                    f"Ceiling height inference failed: {message}"
                )
                return

            estimate = thread.result
            self.ceiling_height_result_label.setText(
                format_ceiling_height_estimate(estimate)
            )
            if estimate.has_estimate:
                assert estimate.estimate_m is not None
                completion_stage = (
                    f"Estimated ceiling height: {estimate.estimate_m:.2f} m"
                )
            else:
                completion_stage = "Insufficient visual evidence"
            self.job_manager.complete_job(job_id, stage=completion_stage)
            self.ceiling_height_estimate_changed.emit(estimate)
        finally:
            if runtime is not None and runtime.thread is thread:
                self._ceiling_height_runtime = None
            self.sync_shared_controls()
            thread.deleteLater()

    # ### Object-reference editing ###
    def _remember_temporary_object_reference(
        self,
        reference_bgra: np.ndarray,
        *,
        source_label: str,
    ) -> _TemporaryObjectReferenceSession:
        """Start one external-reference session bound to the current frame."""

        reference = np.ascontiguousarray(reference_bgra).copy()
        self._temporary_object_reference_revision += 1
        objects = self.object_workspace
        session = _TemporaryObjectReferenceSession(
            source_bgra=reference.copy(),
            active_bgra=reference,
            source_label=str(source_label).strip() or "Temporary image",
            revision=self._temporary_object_reference_revision,
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
        accepted_reference = (
            self.object_workspace.get_current_accepted_object_reference()
        )
        if accepted_reference is not None:
            self.video_view.set_reference_preview_bgra(accepted_reference)
            return
        self.video_view.clear_reference_preview()

    def _sync_object_reference_status(self) -> None:
        """Describe the same reference source shown by the shared preview."""

        if self.object_workspace.has_temporary_object_reference():
            session = self._adopt_untracked_temporary_reference()
            source_label = (
                "Temporary image" if session is None else session.source_label
            )
            self.reference_edit_status_label.setText(
                f"{source_label} is active for Object and Surface generation."
            )
            return
        if self.object_workspace.has_current_accepted_object_reference():
            self.reference_edit_status_label.setText(
                "Accepted edit active for this frame."
            )
            return
        self.reference_edit_status_label.setText(
            OBJECT_REFERENCE_EDIT_STATUS_EMPTY
        )

    def _build_temporary_reference_edit_inputs(
        self,
    ) -> tuple[np.ndarray, tuple[object, ...]] | None:
        """Return the original external image so Edit remains a true retry."""

        session = self._adopt_untracked_temporary_reference()
        if session is None or not self._temporary_reference_session_matches_context(
            session
        ):
            return None
        return session.source_bgra.copy(), (
            OBJECT_REFERENCE_EDIT_SOURCE_TEMPORARY,
            session.revision,
        )

    def _get_current_reference_edit_signature(
        self,
        source_kind: str,
    ) -> tuple[object, ...] | None:
        """Resolve the current signature through the runtime's source path."""

        if source_kind == OBJECT_REFERENCE_EDIT_SOURCE_TEMPORARY:
            inputs = self._build_temporary_reference_edit_inputs()
            return None if inputs is None else inputs[1]
        return self.object_workspace.get_current_object_reference_edit_signature()

    @Slot(int)
    def _append_reference_edit_prompt_preset(self, index: int) -> None:
        """Append one reusable view phrase without turning it into state."""

        preset = self.reference_edit_prompt_preset_combo.itemData(int(index))
        self.reference_edit_prompt_preset_combo.setCurrentIndex(0)
        if not isinstance(preset, str) or not preset.strip():
            return
        current_prompt = self.reference_edit_prompt.text().strip()
        separator = " " if current_prompt.endswith(",") else ", "
        next_prompt = (
            preset.strip()
            if not current_prompt
            else f"{current_prompt}{separator}{preset.strip()}"
        )
        if len(next_prompt) > self.reference_edit_prompt.maxLength():
            self.reference_edit_status_label.setText(
                "The view preset would exceed the edit instruction limit."
            )
            return
        self.reference_edit_prompt.setText(next_prompt)

    @Slot(object)
    def _handle_reference_edit_inputs_changed(self, _value: object) -> None:
        """Cancel stale work without discarding an accepted frame edit."""

        if self.object_workspace.has_current_accepted_object_reference():
            self.reference_edit_status_label.setText(
                "Edited reference remains active. Click Edit reference to replace "
                "it, or move the frame slider to undo."
            )
        elif self.object_workspace.has_temporary_object_reference():
            self.reference_edit_status_label.setText(
                "Temporary reference remains active. Click Edit reference to "
                "replace its edited result."
            )
        else:
            self.reference_edit_status_label.setText(
                OBJECT_REFERENCE_EDIT_STATUS_EMPTY
            )
        runtime = self._object_reference_edit_runtime
        if runtime is not None:
            runtime.cancel_requested = True
            if not self.job_manager.cancel_job(runtime.job_id):
                runtime.thread.requestInterruption()
        self.sync_shared_controls()

    @Slot()
    def _start_object_reference_edit(self) -> None:
        """Generate one square reference from the selected masked object."""

        if self._is_shutdown or self._object_reference_edit_runtime is not None:
            return
        prompt = self.reference_edit_prompt.text().strip()
        if not prompt:
            self.reference_edit_status_label.setText(
                "Enter how the selected object should be changed."
            )
            return
        model = str(self.reference_edit_model_combo.currentData() or "")
        if model not in SUPPORTED_OBJECT_REFERENCE_EDIT_MODELS:
            self.reference_edit_status_label.setText(
                "Choose a supported object reference editing model."
            )
            return
        api_key = (
            self.object_workspace.get_runtime_settings().openai_api_key.strip()
        )
        if object_reference_edit_model_requires_api_key(model) and not api_key:
            self.reference_edit_status_label.setText(
                "Add an OpenAI API key in Settings before editing a reference."
            )
            return
        temporary_inputs = self._build_temporary_reference_edit_inputs()
        if temporary_inputs is None:
            source_kind = OBJECT_REFERENCE_EDIT_SOURCE_MASK
            try:
                source_bgra, signature = (
                    self.object_workspace.build_object_reference_edit_inputs()
                )
            except ValueError as error:
                self.reference_edit_status_label.setText(str(error))
                return
        else:
            source_kind = OBJECT_REFERENCE_EDIT_SOURCE_TEMPORARY
            source_bgra, signature = temporary_inputs

        thread = _ObjectReferenceEditThread(
            source_bgra,
            prompt,
            model,
            api_key,
            self._object_reference_editor,
            parent=self,
        )
        job = self.job_manager.create_job(
            kind=OBJECT_REFERENCE_EDIT_JOB_KIND,
            requested_name="",
            default_name="Object reference edit",
            stage="Preparing square object reference (10%)",
        )
        runtime = _ObjectReferenceEditRuntime(
            signature=signature,
            source_kind=source_kind,
            model=model,
            prompt=prompt,
            job_id=job.job_id,
            thread=thread,
        )
        self._object_reference_edit_runtime = runtime
        self.job_manager.set_cancel_callback(
            job.job_id,
            partial(self._cancel_object_reference_edit, job.job_id),
        )
        thread.finished.connect(
            partial(
                self._handle_object_reference_edit_finished,
                job.job_id,
                thread,
            )
        )
        self.reference_edit_status_label.setText(
            "Generating a 1024x1024 object reference..."
        )
        self.job_manager.update_job(
            job.job_id,
            stage="Generating edited object reference (20%)",
        )
        self.sync_shared_controls()
        thread.start()

    def _cancel_object_reference_edit(self, job_id: str) -> bool:
        runtime = self._object_reference_edit_runtime
        if runtime is None or runtime.job_id != str(job_id):
            return False
        runtime.cancel_requested = True
        runtime.thread.requestInterruption()
        self.sync_shared_controls()
        return True

    def _handle_object_reference_edit_finished(
        self,
        job_id: str,
        thread: _ObjectReferenceEditThread,
    ) -> None:
        """Atomically accept a successful result from still-current inputs."""

        runtime = self._object_reference_edit_runtime
        try:
            if (
                runtime is None
                or runtime.job_id != job_id
                or runtime.thread is not thread
            ):
                return
            self.job_manager.set_cancel_callback(job_id, None)
            current_signature = self._get_current_reference_edit_signature(
                runtime.source_kind
            )
            is_stale = (
                current_signature != runtime.signature
                or self.reference_edit_model_combo.currentData() != runtime.model
                or self.reference_edit_prompt.text().strip() != runtime.prompt
            )
            if (
                self._is_shutdown
                or runtime.cancel_requested
                or thread.was_cancelled
                or is_stale
            ):
                stage = (
                    "Ignored after the reference inputs changed"
                    if is_stale
                    else "Cancelled"
                )
                self.job_manager.mark_cancelled(job_id, stage=stage)
                if not self._is_shutdown:
                    self.reference_edit_status_label.setText(stage + ".")
                return
            if thread.error_message is not None or thread.result is None:
                message = thread.error_message or (
                    "Object reference editing returned no image."
                )
                self.job_manager.fail_job(job_id, stage=f"Failed: {message}")
                self.reference_edit_status_label.setText(
                    f"Reference edit failed: {message}"
                )
                return

            if runtime.source_kind == OBJECT_REFERENCE_EDIT_SOURCE_TEMPORARY:
                session = self._temporary_object_reference_session
                if session is None:
                    accepted = False
                else:
                    session.active_bgra = np.ascontiguousarray(
                        thread.result
                    ).copy()
                    self.object_workspace.set_temporary_object_reference(
                        session.active_bgra
                    )
                    self.surface_workspace.set_temporary_reference(
                        session.active_bgra
                    )
                    accepted = True
            else:
                accepted = self.object_workspace.accept_object_reference_edit(
                    runtime.signature,
                    thread.result,
                )
            if not accepted:
                message = (
                    "The selected object changed before the edited reference "
                    "could be applied."
                )
                self.job_manager.fail_job(job_id, stage=f"Failed: {message}")
                self.reference_edit_status_label.setText(message)
                return
            self.video_view.set_reference_preview_bgra(thread.result)
            self.reference_edit_status_label.setText(
                "Edited reference active for the next Meshy request. Click "
                "Edit reference again to retry, or move the frame slider to "
                "undo."
            )
            self.job_manager.complete_job(
                job_id,
                stage="Reference edit accepted",
            )
        finally:
            if runtime is not None and runtime.thread is thread:
                self._object_reference_edit_runtime = None
            self.sync_shared_controls()
            thread.deleteLater()

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
        if (
            self.object_workspace._architectural_trim_editing_target is not None
            and np.all(active_reference[:, :, 3] == 255)
        ):
            message += (
                " This is an unmasked scene image; click Edit reference first "
                "to isolate and straighten one repeating trim module."
            )
        self.object_workspace.status_label.setText(message)
        self.reference_edit_status_label.setText(message)
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
            self._clear_temporary_object_reference_session()
            objects.discard_object_reference_before_frame_change(safe_index)
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
        self._sync_cancel_button()

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
        reference_edit_is_running = self._object_reference_edit_runtime is not None
        shared_reference_controls_are_available = (
            not has_untracked_object_job and not reference_edit_is_running
        )
        video_editor_is_available = (
            has_video and shared_reference_controls_are_available
        )
        accepted_edit_is_available = (
            objects.has_current_accepted_object_reference()
        )
        temporary_reference_is_available = (
            objects.has_temporary_object_reference()
        )
        inpainting_is_available = (
            video_editor_is_available
            and not accepted_edit_is_available
            and not temporary_reference_is_available
        )
        reference_media_changes_are_available = (
            not has_untracked_object_job and not reference_edit_is_running
        )
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
            or accepted_edit_is_available
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
        has_object_mask = self.video_view.has_selection()
        has_edit_prompt = bool(self.reference_edit_prompt.text().strip())
        has_openai_key = bool(
            objects.get_runtime_settings().openai_api_key.strip()
        )
        selected_reference_model = str(
            self.reference_edit_model_combo.currentData() or ""
        )
        has_reference_edit_credentials = (
            selected_reference_model in SUPPORTED_OBJECT_REFERENCE_EDIT_MODELS
            and (
                not object_reference_edit_model_requires_api_key(
                    selected_reference_model
                )
                or has_openai_key
            )
        )
        self.edit_reference_button.setEnabled(
            shared_reference_controls_are_available
            and (temporary_reference_is_available or has_object_mask)
            and has_edit_prompt
            and has_reference_edit_credentials
        )
        reference_edit_inputs_are_available = (
            shared_reference_controls_are_available
            and (has_video or temporary_reference_is_available)
        )
        self.reference_edit_prompt.setEnabled(
            reference_edit_inputs_are_available
        )
        self.reference_edit_prompt_preset_combo.setEnabled(
            reference_edit_inputs_are_available
        )
        self.reference_edit_model_combo.setEnabled(
            reference_edit_inputs_are_available
        )
        ceiling_inference_is_running = self._ceiling_height_runtime is not None
        self.infer_ceiling_height_button.setEnabled(
            not self._is_shutdown
            and not self._door_slot_editing_active
            and self.video_view.has_frame()
            and not ceiling_inference_is_running
        )
        self.infer_ceiling_height_button.setText(
            "Inferring..."
            if ceiling_inference_is_running
            else "Infer ceiling height"
        )
        self._sync_generation_texture_context()
        self._sync_cancel_button()

    @Slot()
    def _sync_cancel_button(self) -> None:
        job = self._latest_cancellable_job()
        self.cancel_button.setEnabled(job is not None)
        if job is None:
            self.cancel_button.setToolTip(
                "There is no active generation or frame-analysis job."
            )
            return
        self.cancel_button.setToolTip(f'Cancel the newest active job: "{job.name}".')

    def _latest_cancellable_job(self) -> GenerationJob | None:
        return next(
            (
                job
                for job in reversed(self.job_manager.jobs())
                if job.kind in CANCELLABLE_GENERATION_JOB_KINDS
                and job.status == JOB_STATUS_RUNNING
            ),
            None,
        )


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


def _build_controls_section(
    title: str,
    kind: str,
) -> tuple[QWidget, QGridLayout]:
    section = QWidget()
    section.setObjectName(f"merged_generation_{kind}_controls")
    outer_layout = QVBoxLayout(section)
    outer_layout.setContentsMargins(8, 7, 8, 7)
    outer_layout.setSpacing(5)
    title_label = QLabel(title)
    title_label.setObjectName(f"merged_generation_{kind}_title")
    title_label.setStyleSheet("font-weight: 600;")
    outer_layout.addWidget(title_label)

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


def _combined_widget_bounds(first: QWidget, second: QWidget) -> QRect:
    return first.geometry().united(second.geometry())


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
