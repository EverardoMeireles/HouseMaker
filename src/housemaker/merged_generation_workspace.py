# ### Imports ###
from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from PySide6.QtCore import QRect, Qt, Slot
from PySide6.QtGui import QColor, QKeySequence, QPainter, QPaintEvent, QPen, QShortcut
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
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
CANCELLABLE_GENERATION_JOB_KINDS = frozenset(
    {
        GENERATION_JOB_KIND_MODEL,
        GENERATION_JOB_KIND_TEXTURE,
        GENERATION_JOB_KIND_FACE_EDIT,
        SURFACE_TEXTURE_JOB_KIND,
    }
)


# ### Overlapping workflow outlines ###
class _WorkflowControlsRow(QWidget):
    """Draw overlapping Surface and Object bounds around one shared section."""

    def __init__(
        self,
        surface_controls: QWidget,
        shared_controls: QWidget,
        object_controls: QWidget,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("merged_generation_controls_row")
        self._surface_controls = surface_controls
        self._shared_controls = shared_controls
        self._object_controls = object_controls

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(WORKFLOW_SECTION_SPACING)
        layout.addWidget(surface_controls, 1)
        layout.addWidget(shared_controls, 1)
        layout.addWidget(object_controls, 2)

    def paintEvent(self, event: QPaintEvent) -> None:  # type: ignore[override]
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        blue_bounds = _combined_widget_bounds(
            self._surface_controls,
            self._shared_controls,
        ).adjusted(
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

    def __init__(
        self,
        surface_workspace: SurfaceTextureGenerationWorkspace,
        object_workspace: GenerationWorkspace,
        job_manager: GenerationJobManager,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("merged_generation_workspace")
        self.surface_workspace = surface_workspace
        self.object_workspace = object_workspace
        self.job_manager = job_manager

        self.surface_workspace.setParent(self)
        self.object_workspace.setParent(self)
        self.surface_workspace.hide()
        self.object_workspace.hide()
        self.object_workspace.set_shared_control_state_managed_externally(True)
        self._adopt_shared_controls()
        self._build_ui()
        self._connect_shared_controls()
        self.clear_mask_shortcut = QShortcut(self)
        self.clear_mask_shortcut.setContext(
            Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        self.clear_mask_shortcut.activated.connect(self._clear_mask_from_shortcut)
        self.set_clear_mask_hotkey(DEFAULT_CLEAR_MASK_HOTKEY)
        self.sync_shared_controls()

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

        self.surface_controls = self._build_surface_controls()
        self.shared_controls = self._build_shared_controls()
        self.object_controls = self._build_object_controls()
        self.controls_row = _WorkflowControlsRow(
            self.surface_controls,
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

    def _build_surface_controls(self) -> QWidget:
        surface = self.surface_workspace
        section, layout = _build_controls_section(
            "Surface texture",
            "surface",
        )
        layout.addWidget(surface.painted_frames_label, 0, 0)
        layout.addWidget(surface.selection_label, 1, 0)
        layout.addWidget(
            _build_labeled_inline_control(
                "Provider",
                surface.surface_texture_provider_combo,
            ),
            2,
            0,
        )
        surface.generate_button.setText("Generate surface texture")
        layout.addWidget(surface.generate_button, 3, 0)
        layout.setRowStretch(4, 1)
        return section

    def _build_shared_controls(self) -> QWidget:
        section, layout = _build_controls_section("Shared", "shared")

        shared_column, shared_layout = _build_controls_column(
            "merged_generation_shared_primary_column"
        )
        shared_layout.addWidget(self.load_video_button)
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
            "Cancel the newest active Surface or Object generation job."
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

        self.object_creation_section, creation_layout = _build_boxed_section(
            "Generation",
            "merged_generation_object_creation_section",
        )

        generation_settings = QWidget()
        generation_settings.setObjectName(
            "merged_generation_object_generation_settings"
        )
        generation_settings_layout = QHBoxLayout(generation_settings)
        generation_settings_layout.setContentsMargins(0, 0, 0, 0)
        generation_settings_layout.setSpacing(6)
        generation_settings_layout.addWidget(objects.symmetric_division_checkbox)
        generation_settings_layout.addStretch(1)
        generation_settings_layout.addWidget(objects.meshy_target_polycount_control)
        creation_layout.addWidget(generation_settings)

        primary_actions = QWidget()
        primary_actions.setObjectName("merged_generation_primary_object_actions")
        primary_actions_layout = QHBoxLayout(primary_actions)
        primary_actions_layout.setContentsMargins(0, 0, 0, 0)
        primary_actions_layout.setSpacing(0)
        primary_actions_layout.addWidget(objects.generate_geometry_button)
        primary_actions_layout.addWidget(objects.generate_texture_button)
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

        self.video_view.strokes_changed.connect(surface._handle_video_strokes_changed)
        self.video_view.strokes_changed.connect(self.sync_shared_controls)
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
            self.object_workspace.load_video(file_path)
            self.surface_workspace.load_video(file_path)
            self.sync_shared_controls()
        except (OSError, RuntimeError, ValueError) as error:
            QMessageBox.critical(self, "Video load failed", str(error))

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
        editor_is_available = has_video and not has_untracked_object_job
        self.load_video_button.setEnabled(not has_untracked_object_job)
        self.seekbar.setEnabled(editor_is_available)
        self.paint_mask_button.setEnabled(editor_is_available)
        self.erase_mask_button.setEnabled(editor_is_available)
        self.brush_size_spinbox.setEnabled(editor_is_available)
        self.clear_mask_button.setEnabled(
            editor_is_available and self.video_view.has_selection()
        )
        self.pbr_map_control.setEnabled(not has_untracked_object_job)
        self.ai_prompt_edit.setEnabled(not has_untracked_object_job)
        self.video_view.set_interaction_enabled(editor_is_available)
        self._sync_cancel_button()

    @Slot()
    def _sync_cancel_button(self) -> None:
        job = self._latest_cancellable_job()
        self.cancel_button.setEnabled(job is not None)
        if job is None:
            self.cancel_button.setToolTip(
                "There is no active Surface or Object generation job."
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
