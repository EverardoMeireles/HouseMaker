# ### Imports ###
from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import uuid
from collections.abc import Callable
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, QStandardPaths, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from shiboken6 import isValid as is_valid_qt_object

from housemaker.video_source import (
    VIDEO_FILE_FILTER,
    VideoFrameSource,
    probe_video,
)
from housemaker.generation_state import (
    MASK_MODE_ERASE,
    MASK_MODE_PAINT,
    GeneratedObjectRecord,
    GenerationData,
)
from housemaker.generation_views import VideoInpaintView
from housemaker.glb import GeneratedModel, import_generated_glb
from housemaker.meshy_generation import (
    MeshyGenerationResult,
    request_image_to_3d_model,
)
from housemaker.settings_widget import (
    DEFAULT_MESHY_TARGET_POLYCOUNT,
    MESHY_SMART_TOPOLOGY_MAX_TARGET_POLYCOUNT,
    MESHY_SMART_TOPOLOGY_MIN_TARGET_POLYCOUNT,
    GenerationServiceSettings,
)
from housemaker.texture_atlas_view import TextureAtlasEntry, TextureAtlasView
from housemaker.viewer import GlbViewerWidget


# ### Constants ###
MIN_BRUSH_RADIUS_PIXELS = 2
MAX_BRUSH_RADIUS_PIXELS = 160
DEFAULT_BRUSH_RADIUS_PIXELS = 24
AMBIENT_LIGHT_PERCENT_SCALE = 100
VIEW_STRETCH = 1
CONTROL_STRETCH = 0
INTERRUPT_POLL_SECONDS = 0.01
SHUTDOWN_WAIT_MILLISECONDS = 250
GENERATION_BACKEND_MESHY = "meshy"
OBJECT_ID_ITEM_ROLE = Qt.ItemDataRole.UserRole
OBJECT_LIST_MAXIMUM_HEIGHT = 124


# ### Generation interfaces ###
class MeshyPlanner(Protocol):
    def plan(self, request: "GenerationRequest") -> MeshyGenerationResult:
        """Return a completed Meshy Image-to-3D task result."""


class MeshyExecutor(Protocol):
    def execute(self, result: MeshyGenerationResult) -> GeneratedModel:
        """Import one downloaded Meshy GLB for the result viewer."""


class GenerationRequest:
    """Owned selected-object input passed to Meshy Image-to-3D."""

    def __init__(
        self,
        *,
        frame_index: int,
        selected_object_bgra: np.ndarray,
        settings: GenerationServiceSettings,
    ) -> None:
        self.frame_index = int(frame_index)
        self.selected_object_bgra = np.ascontiguousarray(
            selected_object_bgra
        ).copy()
        self.settings = settings


# ### Default adapters ###
class MeshyImagePlanner:
    """Submit only the isolated object crop to Meshy Image-to-3D."""

    def plan(
        self,
        request: GenerationRequest,
        progress_callback: Callable[[str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> MeshyGenerationResult:
        if not request.settings.meshy_api_key:
            raise ValueError("Set a Meshy AI API key in Settings before generating.")

        def report_progress(status: str, progress: int) -> None:
            if progress_callback is None:
                return
            if status == "PENDING":
                progress_callback("Meshy task queued...")
            elif status == "IN_PROGRESS":
                progress_callback(f"Meshy is generating: {progress}%")
            elif status == "SUCCEEDED":
                progress_callback("Meshy generation complete. Downloading GLB...")

        return request_image_to_3d_model(
            api_key=request.settings.meshy_api_key,
            image_png=_encode_png(request.selected_object_bgra),
            target_polycount=request.settings.meshy_target_polycount,
            progress_callback=report_progress,
            cancel_event=cancel_event,
        )


class MeshyModelExecutor:
    """Validate and adapt one downloaded Meshy GLB for the HouseMaker viewer."""

    def execute(self, result: object) -> GeneratedModel:
        if not isinstance(result, MeshyGenerationResult):
            raise TypeError("Meshy returned an invalid generation result.")
        return import_generated_glb(result.glb_bytes)


# ### Object viewer panel ###
class ObjectGenerationViewerPanel(QWidget):
    """Keep the generated-object selector with its detachable 3D viewer."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.viewer = GlbViewerWidget(wireframe_enabled=False)
        layout.addWidget(self.viewer, 1)

        self.object_list = QListWidget()
        self.object_list.setObjectName("generated_objects_list")
        self.object_list.setMaximumHeight(OBJECT_LIST_MAXIMUM_HEIGHT)
        self.object_list.setAlternatingRowColors(True)
        self.object_list.setToolTip(
            "Select which generated Meshy object is shown in the 3D view."
        )
        layout.addWidget(self.object_list)

        self.statistics_label = QLabel("No generated object")
        self.statistics_label.setObjectName("model_statistics_label")
        self.statistics_label.setWordWrap(True)
        self.statistics_label.setStyleSheet(
            "color: #aeb7c5; padding: 2px 4px;"
        )
        layout.addWidget(self.statistics_label)

    def focus_navigation(self) -> None:
        """Forward external-window focus to the actual OpenGL viewer."""

        self.viewer.focus_navigation()


# ### Background worker ###
class GenerationWorker(QObject):
    succeeded = Signal(object, object)
    failed = Signal(str)
    finished = Signal()
    progress = Signal(str)

    def __init__(
        self,
        planner: MeshyPlanner | Callable[[GenerationRequest], MeshyGenerationResult],
        executor: MeshyExecutor | Callable[[MeshyGenerationResult], GeneratedModel],
        request: GenerationRequest,
    ) -> None:
        super().__init__()
        self._planner = planner
        self._executor = executor
        self._request = request
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    @Slot()
    def run(self) -> None:
        try:
            result = _run_interruptible_stage(
                lambda: _invoke_planner(
                    self._planner,
                    self._request,
                    self.progress.emit,
                    self._cancel_event,
                )
            )
            generated_model = _run_interruptible_stage(
                lambda: _invoke_executor(self._executor, result)
            )
            if not isinstance(generated_model, GeneratedModel):
                raise TypeError("The Meshy executor returned an invalid model.")
        except _GenerationCancelled:
            return
        except Exception as error:
            self.failed.emit(_safe_error_message(error, self._request.settings))
            return
        else:
            self.succeeded.emit(result, generated_model)
        finally:
            self.finished.emit()


# ### Generation workspace ###
class GenerationWorkspace(QWidget):
    """Manual video selection and Meshy Image-to-3D workspace."""

    data_changed = Signal(object)
    generation_completed = Signal(object, object)

    def __init__(
        self,
        meshy_planner: MeshyPlanner
        | Callable[[GenerationRequest], MeshyGenerationResult]
        | None = None,
        meshy_executor: MeshyExecutor
        | Callable[[MeshyGenerationResult], GeneratedModel]
        | None = None,
        asset_directory: str | Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._meshy_planner = meshy_planner or MeshyImagePlanner()
        self._meshy_executor = meshy_executor or MeshyModelExecutor()
        self._asset_directory = (
            _default_generation_asset_directory()
            if asset_directory is None
            else Path(asset_directory).expanduser()
        )
        self._settings = GenerationServiceSettings(
            meshy_api_key="",
            meshy_target_polycount=DEFAULT_MESHY_TARGET_POLYCOUNT,
        )
        self._mesh_target_polycount_is_locally_selected = False
        self._data = GenerationData()
        self._video_source: VideoFrameSource | None = None
        self._displayed_frame_index: int | None = None
        self._is_syncing_seekbar = False
        self._generation_thread: QThread | None = None
        self._generation_worker: GenerationWorker | None = None
        self._generated_model: GeneratedModel | None = None
        self._generated_model_cache: dict[str, GeneratedModel] = {}
        self._texture_atlas_entries_by_object_id: dict[
            str,
            tuple[TextureAtlasEntry, ...],
        ] = {}
        self._selected_object_id: str | None = None

        self._build_ui()
        self._sync_video_controls()
        self._sync_controls()

    def get_data(self) -> GenerationData:
        self._store_current_frame_strokes()
        return self._data.clone()

    def set_data(self, data: GenerationData | None) -> None:
        if self._generation_thread is not None:
            raise RuntimeError("Cannot replace Generation data while generating.")
        self._close_video_source()
        self._displayed_frame_index = None
        self._data = GenerationData() if data is None else data.clone()
        metadata = self._data.video_metadata
        if metadata is not None and Path(metadata.path).exists():
            try:
                self._video_source = VideoFrameSource(metadata.path)
                self._data.video_metadata = self._video_source.metadata
            except ValueError as error:
                self.status_label.setText(str(error))
        elif metadata is not None:
            self.status_label.setText(f"Video missing: {metadata.path}")

        self._data.current_frame_index = self._clamp_frame_index(
            self._data.current_frame_index
        )
        self.video_view.clear_frame("Load the source video to continue")
        self._sync_video_controls()
        if self._video_source is None:
            pass
        else:
            self.show_frame(self._data.current_frame_index)
        self._generated_model_cache.clear()
        self._texture_atlas_entries_by_object_id.clear()
        self._rebuild_generated_objects()
        self._sync_controls()

    def set_runtime_settings(self, settings: GenerationServiceSettings) -> None:
        if not isinstance(settings, GenerationServiceSettings):
            raise TypeError("Generation settings have an invalid type.")
        target_polycount = settings.meshy_target_polycount
        if self._mesh_target_polycount_is_locally_selected:
            target_polycount = self.meshy_target_polycount_spinbox.value()
        self._settings = replace(
            settings,
            meshy_target_polycount=target_polycount,
        )
        self._sync_meshy_target_polycount_value()
        self._sync_controls()

    def get_runtime_settings(self) -> GenerationServiceSettings:
        return self._settings

    def set_external_3d_viewer_active(self, is_active: bool) -> None:
        """Show the local atlas inspector while the 3D panel is external."""

        self.right_view_stack.setCurrentWidget(
            self.texture_view_page if is_active else self.object_3d_page
        )

    @property
    def is_generating(self) -> bool:
        return self._generation_thread is not None

    def set_meshy_planner(
        self,
        planner: MeshyPlanner | Callable[[GenerationRequest], MeshyGenerationResult],
    ) -> None:
        self._meshy_planner = planner

    def set_meshy_executor(
        self,
        executor: MeshyExecutor | Callable[[MeshyGenerationResult], GeneratedModel],
    ) -> None:
        self._meshy_executor = executor

    def load_video(self, video_path: str) -> None:
        if self._generation_thread is not None:
            raise RuntimeError("Cannot replace the video while generating.")
        metadata = probe_video(video_path)
        next_source = VideoFrameSource(metadata.path)
        self._close_video_source()
        self._displayed_frame_index = None
        self.video_view.clear_frame()
        self._video_source = next_source
        self._data.video_metadata = next_source.metadata
        self._data.current_frame_index = 0
        self._data.frame_strokes = {}
        self._sync_video_controls()
        self.show_frame(0)
        self.status_label.setText(
            "Video loaded. Paint over one object, or right-click inside a "
            "closed outline to fill it, then click Generate."
        )
        self._emit_data_changed()
        self._sync_controls()

    def show_frame(self, frame_index: int) -> None:
        if self._video_source is None:
            return
        self._store_displayed_frame_strokes()
        safe_index = self._clamp_frame_index(frame_index)
        try:
            frame_bgr = self._video_source.get_frame(safe_index)
        except (IndexError, ValueError) as error:
            self.status_label.setText(str(error))
            return

        self._data.current_frame_index = safe_index
        self._displayed_frame_index = safe_index
        self.video_view.set_frame(
            frame_bgr,
            self._data.strokes_for_frame(safe_index),
        )
        self._sync_seekbar_value(safe_index)
        self._sync_frame_label()
        self._sync_controls()

    def generate(self) -> None:
        if self._generation_thread is not None:
            return
        request = self._build_generation_request()
        if request is None:
            return
        self._start_generation(request)

    def _start_generation(self, request: GenerationRequest) -> None:
        """Start one owned request; split out for deterministic UI tests."""

        self._generation_thread = QThread(self)
        self._generation_worker = GenerationWorker(
            self._meshy_planner,
            self._meshy_executor,
            request,
        )
        self._generation_worker.moveToThread(self._generation_thread)
        self._generation_thread.started.connect(self._generation_worker.run)
        self._generation_worker.succeeded.connect(
            self._handle_generation_succeeded
        )
        self._generation_worker.failed.connect(self._handle_generation_failed)
        self._generation_worker.progress.connect(self.status_label.setText)
        self._generation_worker.finished.connect(
            self._generation_worker.deleteLater
        )
        self._generation_worker.finished.connect(self._generation_thread.quit)
        self._generation_thread.finished.connect(
            self._handle_generation_thread_finished
        )
        self._generation_thread.finished.connect(
            self._generation_thread.deleteLater
        )
        self.status_label.setText(
            f"Submitting frame {request.frame_index + 1} to Meshy..."
        )
        self._generation_thread.start()
        self._sync_controls()

    def shutdown(self) -> None:
        worker = self._generation_worker
        thread = self._generation_thread
        if thread is not None:
            if not is_valid_qt_object(thread):
                self._generation_thread = None
                self._generation_worker = None
                self._close_video_source()
                return
            if worker is not None and is_valid_qt_object(worker):
                worker.cancel()
                for signal, slot in (
                    (worker.succeeded, self._handle_generation_succeeded),
                    (worker.failed, self._handle_generation_failed),
                ):
                    try:
                        signal.disconnect(slot)
                    except (RuntimeError, TypeError):
                        pass
            try:
                thread.finished.disconnect(
                    self._handle_generation_thread_finished
                )
            except (RuntimeError, TypeError):
                pass
            thread.requestInterruption()
            thread.quit()
            thread.wait(SHUTDOWN_WAIT_MILLISECONDS)
            if self._generation_thread is thread:
                self._generation_thread = None
                self._generation_worker = None
        self._close_video_source()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.shutdown()
        super().closeEvent(event)

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        views_widget = QWidget()
        views_layout = QHBoxLayout(views_widget)
        views_layout.setContentsMargins(0, 0, 0, 0)
        views_layout.setSpacing(8)

        self.video_view = VideoInpaintView()
        self.video_view.strokes_changed.connect(
            self._handle_video_strokes_changed
        )
        views_layout.addWidget(
            _build_labeled_view("Source video and object mask", self.video_view),
            VIEW_STRETCH,
        )

        self.object_3d_panel = ObjectGenerationViewerPanel()
        self.result_view = self.object_3d_panel.viewer
        self.generated_objects_list = self.object_3d_panel.object_list
        self.model_statistics_label = self.object_3d_panel.statistics_label
        self.generated_objects_list.currentItemChanged.connect(
            self._handle_generated_object_selection_changed
        )

        self.object_3d_page = _build_labeled_view(
            "Generated 3D objects",
            self.object_3d_panel,
        )
        self.texture_view = TextureAtlasView()
        self.texture_view.setObjectName("object_texture_atlas_view")
        self.texture_view_page = _build_labeled_view(
            "Texture view",
            self.texture_view,
        )
        self.right_view_stack = QStackedWidget()
        self.right_view_stack.setObjectName("object_generation_right_view_stack")
        self.right_view_stack.addWidget(self.object_3d_page)
        self.right_view_stack.addWidget(self.texture_view_page)
        self.right_view_stack.setCurrentWidget(self.object_3d_page)
        views_layout.addWidget(self.right_view_stack, VIEW_STRETCH)
        root_layout.addWidget(views_widget, 1)

        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(4, 4, 4, 4)
        controls_layout.setSpacing(6)

        self.seekbar = QSlider(Qt.Orientation.Horizontal)
        self.seekbar.setRange(0, 0)
        self.seekbar.valueChanged.connect(self._handle_seekbar_changed)
        controls_layout.addWidget(self.seekbar)

        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(8)

        self.load_video_button = QPushButton("Load video")
        self.load_video_button.clicked.connect(self._handle_load_video_clicked)
        buttons_layout.addWidget(self.load_video_button)

        self.frame_label = QLabel("Frame 0 / 0")
        self.frame_label.setMinimumWidth(120)
        buttons_layout.addWidget(self.frame_label)

        self.meshy_target_polycount_control = QWidget()
        self.meshy_target_polycount_control.setObjectName(
            "meshy_target_polycount_control"
        )
        meshy_target_layout = QHBoxLayout(self.meshy_target_polycount_control)
        meshy_target_layout.setContentsMargins(0, 0, 0, 0)
        meshy_target_layout.setSpacing(4)
        meshy_target_layout.addWidget(QLabel("Target tris"))
        self.meshy_target_polycount_spinbox = QSpinBox()
        self.meshy_target_polycount_spinbox.setObjectName(
            "meshy_target_polycount_spinbox"
        )
        self.meshy_target_polycount_spinbox.setRange(
            MESHY_SMART_TOPOLOGY_MIN_TARGET_POLYCOUNT,
            MESHY_SMART_TOPOLOGY_MAX_TARGET_POLYCOUNT,
        )
        self.meshy_target_polycount_spinbox.setSingleStep(100)
        self.meshy_target_polycount_spinbox.setValue(
            self._settings.meshy_target_polycount
        )
        self.meshy_target_polycount_spinbox.setSuffix(" tris")
        self.meshy_target_polycount_spinbox.setToolTip(
            "Target triangle count for Meshy Smart Topology (100-15,000). "
            "Lower values make lighter, less detailed geometry."
        )
        self.meshy_target_polycount_spinbox.valueChanged.connect(
            self._handle_meshy_target_polycount_changed
        )
        meshy_target_layout.addWidget(self.meshy_target_polycount_spinbox)
        buttons_layout.addWidget(self.meshy_target_polycount_control)

        buttons_layout.addWidget(QLabel("Ambient"))
        self.ambient_light_slider = QSlider(Qt.Orientation.Horizontal)
        self.ambient_light_slider.setObjectName("ambient_light_slider")
        self.ambient_light_slider.setRange(0, AMBIENT_LIGHT_PERCENT_SCALE)
        self.ambient_light_slider.setValue(
            round(
                self.result_view.get_ambient_light_intensity()
                * AMBIENT_LIGHT_PERCENT_SCALE
            )
        )
        self.ambient_light_slider.setFixedWidth(100)
        self.ambient_light_slider.setToolTip(
            "Ambient light keeps the generated object visible on unlit sides."
        )
        self.ambient_light_slider.valueChanged.connect(
            self._handle_ambient_light_changed
        )
        buttons_layout.addWidget(self.ambient_light_slider)

        self.textures_checkbox = QCheckBox("Textures")
        self.textures_checkbox.setObjectName("textures_checkbox")
        self.textures_checkbox.setChecked(True)
        self.textures_checkbox.setToolTip(
            "Show the textures embedded in the generated Meshy model."
        )
        self.textures_checkbox.toggled.connect(
            self.result_view.set_textures_enabled
        )
        buttons_layout.addWidget(self.textures_checkbox)

        self.wireframe_checkbox = QCheckBox("Wireframe")
        self.wireframe_checkbox.setObjectName("wireframe_checkbox")
        self.wireframe_checkbox.setChecked(False)
        self.wireframe_checkbox.setToolTip(
            "Overlay the generated model's triangle edges."
        )
        self.wireframe_checkbox.toggled.connect(
            self.result_view.set_wireframe_enabled
        )
        buttons_layout.addWidget(self.wireframe_checkbox)

        self.paint_mask_button = QRadioButton("Paint")
        self.erase_mask_button = QRadioButton("Erase")
        self.paint_mask_button.setChecked(True)
        self._mask_mode_button_group = QButtonGroup(self)
        self._mask_mode_button_group.addButton(self.paint_mask_button)
        self._mask_mode_button_group.addButton(self.erase_mask_button)
        self.paint_mask_button.toggled.connect(self._handle_mask_mode_changed)
        buttons_layout.addWidget(self.paint_mask_button)
        buttons_layout.addWidget(self.erase_mask_button)

        buttons_layout.addWidget(QLabel("Brush"))
        self.brush_size_spinbox = QSpinBox()
        self.brush_size_spinbox.setRange(
            MIN_BRUSH_RADIUS_PIXELS,
            MAX_BRUSH_RADIUS_PIXELS,
        )
        self.brush_size_spinbox.setValue(DEFAULT_BRUSH_RADIUS_PIXELS)
        self.brush_size_spinbox.setSuffix(" px")
        self.brush_size_spinbox.valueChanged.connect(
            self.video_view.set_brush_radius_pixels
        )
        buttons_layout.addWidget(self.brush_size_spinbox)

        self.undo_mask_button = QPushButton("Undo stroke")
        self.undo_mask_button.clicked.connect(self.video_view.undo_last_stroke)
        buttons_layout.addWidget(self.undo_mask_button)

        self.clear_mask_button = QPushButton("Clear mask")
        self.clear_mask_button.clicked.connect(self.video_view.clear_mask)
        buttons_layout.addWidget(self.clear_mask_button)

        self.generate_button = QPushButton("Generate")
        self.generate_button.setMinimumHeight(38)
        self.generate_button.setToolTip(
            "Requires a painted object mask and a Meshy API key. "
            "Meshy Image-to-3D tasks consume account credits."
        )
        self.generate_button.clicked.connect(self.generate)
        buttons_layout.addWidget(self.generate_button)
        buttons_layout.addStretch(1)
        controls_layout.addLayout(buttons_layout)

        self.status_label = QLabel(
            "Load a video, seek to a useful frame, and paint over an object. "
            "Right-click fills an area enclosed by the painted outline."
        )
        self.status_label.setWordWrap(True)
        controls_layout.addWidget(self.status_label)
        root_layout.addWidget(controls_widget, CONTROL_STRETCH)

    def _handle_load_video_clicked(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load home video",
            str(Path.cwd()),
            VIDEO_FILE_FILTER,
        )
        if not file_path:
            return
        try:
            self.load_video(file_path)
        except (RuntimeError, ValueError) as error:
            QMessageBox.critical(self, "Video load failed", str(error))

    def _handle_seekbar_changed(self, frame_index: int) -> None:
        if self._is_syncing_seekbar or self._generation_thread is not None:
            return
        self.show_frame(frame_index)

    def _handle_mask_mode_changed(self, paint_checked: bool) -> None:
        self.video_view.set_brush_mode(
            MASK_MODE_PAINT if paint_checked else MASK_MODE_ERASE
        )

    def _handle_meshy_target_polycount_changed(self, value: int) -> None:
        self._mesh_target_polycount_is_locally_selected = True
        self._settings = replace(
            self._settings,
            meshy_target_polycount=int(value),
        )

    def _handle_ambient_light_changed(self, value: int) -> None:
        self.result_view.set_ambient_light_intensity(
            int(value) / AMBIENT_LIGHT_PERCENT_SCALE
        )

    def _handle_video_strokes_changed(self, raw_strokes: object) -> None:
        if not isinstance(raw_strokes, list):
            return
        if self._displayed_frame_index is None:
            return
        self._data.set_frame_strokes(
            self._displayed_frame_index,
            self.video_view.get_strokes(),
        )
        self._emit_data_changed()
        self._sync_controls()

    @Slot(object, object)
    def _handle_generation_succeeded(
        self,
        result: object,
        generated_model: GeneratedModel,
    ) -> None:
        if not isinstance(result, MeshyGenerationResult):
            self._handle_generation_failed("Meshy returned an invalid result.")
            return
        object_id = uuid.uuid4().hex
        object_name = result.name
        try:
            asset_path = self._persist_meshy_asset(
                object_id,
                result.glb_bytes,
            )
        except OSError as error:
            self._handle_generation_failed(
                f"The Meshy GLB could not be saved locally: {error}"
            )
            return
        record = GeneratedObjectRecord(
            object_id=object_id,
            frame_index=self._data.current_frame_index,
            object_name=object_name,
            pipeline={},
            provider=GENERATION_BACKEND_MESHY,
            provider_task_id=result.task_id,
            asset_path=asset_path,
        )
        self._data.generated_objects.append(record)
        self._generated_model_cache[object_id] = generated_model
        self._texture_atlas_entries_by_object_id[object_id] = tuple(
            _build_model_texture_atlas_entries(record, generated_model)
        )
        self._selected_object_id = object_id
        self._generated_model = generated_model
        self._refresh_generated_objects_list(object_id)
        self.status_label.setText(f"Generated: {object_name}")
        self._emit_data_changed()
        self.generation_completed.emit(record, generated_model)

    @Slot(str)
    def _handle_generation_failed(self, error_message: str) -> None:
        self.status_label.setText(f"Generation failed: {error_message}")
        QMessageBox.warning(self, "Generation failed", error_message)

    @Slot()
    def _handle_generation_thread_finished(self) -> None:
        if self.sender() is not self._generation_thread:
            return
        self._generation_worker = None
        self._generation_thread = None
        self._sync_controls()

    def _build_generation_request(self) -> GenerationRequest | None:
        if self.video_view.get_frame_bgr() is None:
            self.status_label.setText("Load a video before generating.")
            return None
        if not self.video_view.has_selection():
            self.status_label.setText("Paint over the object to generate first.")
            return None
        selected_crop = self.video_view.build_selected_object_crop()
        if selected_crop.size == 0:
            self.status_label.setText("The selected object mask is empty.")
            return None
        self._store_current_frame_strokes()
        return GenerationRequest(
            frame_index=self._data.current_frame_index,
            selected_object_bgra=selected_crop,
            settings=self._settings,
        )

    def _store_current_frame_strokes(self) -> None:
        self._store_displayed_frame_strokes()

    def _store_displayed_frame_strokes(self) -> None:
        if self._displayed_frame_index is None:
            return
        self._store_frame_strokes(self._displayed_frame_index)

    def _store_frame_strokes(self, frame_index: int) -> None:
        if self._video_source is None:
            return
        self._data.set_frame_strokes(
            frame_index,
            self.video_view.get_strokes(),
        )

    def _sync_video_controls(self) -> None:
        metadata = self._data.video_metadata
        frame_count = 0 if metadata is None else metadata.frame_count
        self.seekbar.setRange(0, max(frame_count - 1, 0))
        self._sync_frame_label()

    def _sync_frame_label(self) -> None:
        metadata = self._data.video_metadata
        if metadata is None:
            self.frame_label.setText("Frame 0 / 0")
            return
        self.frame_label.setText(
            f"Frame {self._data.current_frame_index + 1} / {metadata.frame_count}"
        )

    def _sync_seekbar_value(self, frame_index: int) -> None:
        self._is_syncing_seekbar = True
        self.seekbar.setValue(int(frame_index))
        self._is_syncing_seekbar = False

    def _sync_controls(self) -> None:
        has_video = self._video_source is not None
        is_generating = self._generation_thread is not None
        has_mask = self.video_view.has_selection()
        self.load_video_button.setEnabled(not is_generating)
        self.seekbar.setEnabled(has_video and not is_generating)
        self.paint_mask_button.setEnabled(has_video and not is_generating)
        self.erase_mask_button.setEnabled(has_video and not is_generating)
        self.brush_size_spinbox.setEnabled(has_video and not is_generating)
        self.undo_mask_button.setEnabled(
            has_video
            and bool(self.video_view.get_strokes())
            and not is_generating
        )
        self.clear_mask_button.setEnabled(
            has_video and has_mask and not is_generating
        )
        required_key_is_available = bool(self._settings.meshy_api_key)
        self.meshy_target_polycount_control.setVisible(True)
        self.meshy_target_polycount_spinbox.setEnabled(
            not is_generating
        )
        self.ambient_light_slider.setEnabled(not is_generating)
        self.textures_checkbox.setEnabled(not is_generating)
        self.wireframe_checkbox.setEnabled(not is_generating)
        self.generate_button.setEnabled(
            has_video
            and has_mask
            and required_key_is_available
            and not is_generating
        )
        self.video_view.set_interaction_enabled(has_video and not is_generating)

    def _sync_meshy_target_polycount_value(self) -> None:
        spinbox = self.meshy_target_polycount_spinbox
        was_blocked = spinbox.blockSignals(True)
        try:
            spinbox.setValue(self._settings.meshy_target_polycount)
        finally:
            spinbox.blockSignals(was_blocked)

    def _clamp_frame_index(self, frame_index: int) -> int:
        metadata = self._data.video_metadata
        if metadata is None or metadata.frame_count <= 0:
            return 0
        return min(max(int(frame_index), 0), metadata.frame_count - 1)

    @Slot(QListWidgetItem, QListWidgetItem)
    def _handle_generated_object_selection_changed(
        self,
        current_item: QListWidgetItem | None,
        _previous_item: QListWidgetItem | None,
    ) -> None:
        if current_item is None:
            self._selected_object_id = None
            self._clear_generated_object_display()
            return
        object_id = str(current_item.data(OBJECT_ID_ITEM_ROLE) or "")
        record = self._find_generated_object_record(object_id)
        if record is None:
            self._selected_object_id = None
            self._clear_generated_object_display()
            return
        self._selected_object_id = object_id
        self._display_generated_object(record)

    def _rebuild_generated_objects(self) -> None:
        preferred_id = self._selected_object_id
        if self._find_generated_object_record(preferred_id) is None:
            preferred_id = (
                None
                if not self._data.generated_objects
                else self._data.generated_objects[-1].object_id
            )
        self._refresh_generated_objects_list(preferred_id)

    def _refresh_generated_objects_list(
        self,
        selected_object_id: str | None,
    ) -> None:
        object_list = self.generated_objects_list
        was_blocked = object_list.blockSignals(True)
        try:
            object_list.clear()
            selected_row = -1
            for object_index, record in enumerate(
                self._data.generated_objects,
                start=1,
            ):
                item = QListWidgetItem(
                    f"#{object_index} {record.object_name} · "
                    f"frame {record.frame_index + 1}"
                )
                item.setData(OBJECT_ID_ITEM_ROLE, record.object_id)
                if record.provider_task_id:
                    item.setToolTip(
                        f"Meshy task: {record.provider_task_id}"
                    )
                object_list.addItem(item)
                if record.object_id == selected_object_id:
                    selected_row = object_list.count() - 1
            if selected_row >= 0:
                object_list.setCurrentRow(selected_row)
        finally:
            object_list.blockSignals(was_blocked)

        if selected_row < 0:
            self._selected_object_id = None
            self._clear_generated_object_display()
            return
        self._selected_object_id = selected_object_id
        record = self._find_generated_object_record(selected_object_id)
        if record is None:
            self._clear_generated_object_display()
            return
        self._display_generated_object(record)

    def _find_generated_object_record(
        self,
        object_id: str | None,
    ) -> GeneratedObjectRecord | None:
        if not object_id:
            return None
        return next(
            (
                record
                for record in self._data.generated_objects
                if record.object_id == object_id
            ),
            None,
        )

    def _display_generated_object(
        self,
        record: GeneratedObjectRecord,
    ) -> None:
        self._generated_model = None
        self.result_view.clear_model()
        self._sync_model_statistics(None)
        try:
            generated_model = self._load_generated_object_model(record)
        except Exception as error:
            self.status_label.setText(
                f"Saved generated object could not be rebuilt: {error}"
            )
            self._refresh_object_texture_atlases(record.object_id)
            return
        self._generated_model = generated_model
        self.result_view.set_model(generated_model)
        self._sync_model_statistics(generated_model)
        self._refresh_object_texture_atlases(
            record.object_id,
            generated_model,
        )

    def _clear_generated_object_display(self) -> None:
        self._generated_model = None
        self.result_view.clear_model()
        self._sync_model_statistics(None)
        self.texture_view.clear()

    def _refresh_object_texture_atlases(
        self,
        selected_object_id: str,
        selected_model: GeneratedModel | None = None,
    ) -> None:
        selected_record = self._find_generated_object_record(selected_object_id)
        if (
            selected_record is not None
            and selected_model is not None
            and selected_object_id
            not in self._texture_atlas_entries_by_object_id
        ):
            self._texture_atlas_entries_by_object_id[selected_object_id] = tuple(
                _build_model_texture_atlas_entries(
                    selected_record,
                    selected_model,
                )
            )

        entries: list[TextureAtlasEntry] = []
        live_object_ids = {
            record.object_id for record in self._data.generated_objects
        }
        for cached_object_id in tuple(
            self._texture_atlas_entries_by_object_id
        ):
            if cached_object_id not in live_object_ids:
                self._texture_atlas_entries_by_object_id.pop(
                    cached_object_id,
                    None,
                )
        for record in self._data.generated_objects:
            object_entries = self._texture_atlas_entries_by_object_id.get(
                record.object_id
            )
            if object_entries is None:
                try:
                    model = self._load_generated_object_model(record)
                    object_entries = tuple(
                        _build_model_texture_atlas_entries(record, model)
                    )
                except Exception:
                    object_entries = ()
                self._texture_atlas_entries_by_object_id[
                    record.object_id
                ] = object_entries
            entries.extend(object_entries)

        selected_atlas_id = next(
            (
                entry.atlas_id
                for entry in entries
                if entry.owner_id == selected_object_id
            ),
            None,
        )
        if tuple(entries) != self.texture_view.entries:
            self.texture_view.set_atlases(
                entries,
                selected_atlas_id=selected_atlas_id,
            )
        elif selected_atlas_id is not None:
            self.texture_view.select_atlas(selected_atlas_id)
        else:
            self.texture_view.select_atlas(None)

    def _load_generated_object_model(
        self,
        record: GeneratedObjectRecord,
    ) -> GeneratedModel:
        generated_model = self._generated_model_cache.get(record.object_id)
        if generated_model is not None:
            return generated_model
        asset_path = self._resolve_meshy_asset_path(record.asset_path)
        generated_model = import_generated_glb(asset_path.read_bytes())
        self._generated_model_cache[record.object_id] = generated_model
        return generated_model

    def _sync_model_statistics(self, model: GeneratedModel | None) -> None:
        self.model_statistics_label.setText(_format_model_statistics(model))

    def _persist_meshy_asset(self, object_id: str, glb_bytes: bytes) -> str:
        self._asset_directory.mkdir(parents=True, exist_ok=True)
        file_name = f"{object_id}.glb"
        destination = self._asset_directory / file_name
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{file_name}.",
            suffix=".tmp",
            dir=str(self._asset_directory),
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "wb") as handle:
                handle.write(bytes(glb_bytes))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, destination)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        return file_name

    def _resolve_meshy_asset_path(self, raw_path: str | None) -> Path:
        if not raw_path:
            raise ValueError("The saved Meshy object has no local GLB asset.")
        candidate = (self._asset_directory / raw_path).resolve()
        asset_root = self._asset_directory.resolve()
        try:
            candidate.relative_to(asset_root)
        except ValueError as error:
            raise ValueError("The saved Meshy asset path is unsafe.") from error
        return candidate

    def _emit_data_changed(self) -> None:
        self.data_changed.emit(self._data.clone())

    def _close_video_source(self) -> None:
        if self._video_source is not None:
            self._video_source.close()
        self._video_source = None


# ### Adapter helpers ###
class _GenerationCancelled(Exception):
    """Internal control flow for an abandoned background operation."""


def _run_interruptible_stage(operation: Callable[[], object]) -> object:
    """Run blocking provider work on a daemon and poll Qt cancellation.

    The short-lived QThread remains responsive to shutdown even while a network
    library or injected planner is blocked. The detached daemon owns only the
    immutable request data and never touches Qt widgets.
    """

    completed = threading.Event()
    outcome: list[tuple[bool, object]] = []

    def run_operation() -> None:
        try:
            outcome.append((True, operation()))
        except Exception as error:
            outcome.append((False, error))
        finally:
            completed.set()

    operation_thread = threading.Thread(
        target=run_operation,
        name="HouseMakerGenerationStage",
        daemon=True,
    )
    operation_thread.start()

    qt_thread = QThread.currentThread()
    while not completed.wait(INTERRUPT_POLL_SECONDS):
        if qt_thread.isInterruptionRequested():
            raise _GenerationCancelled
    if qt_thread.isInterruptionRequested():
        raise _GenerationCancelled

    succeeded, value = outcome[0]
    if succeeded:
        return value
    if isinstance(value, Exception):
        raise value
    raise RuntimeError("The generation stage failed without an exception.")


def _invoke_planner(
    planner: MeshyPlanner | Callable[[GenerationRequest], MeshyGenerationResult],
    request: GenerationRequest,
    progress_callback: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> MeshyGenerationResult:
    progress_plan_method = getattr(planner, "plan_with_progress", None)
    if callable(progress_plan_method):
        return progress_plan_method(request, progress_callback, cancel_event)
    plan_method = getattr(planner, "plan", None)
    if callable(plan_method):
        if isinstance(planner, MeshyImagePlanner):
            return planner.plan(request, progress_callback, cancel_event)
        return plan_method(request)
    if callable(planner):
        return planner(request)
    raise TypeError("The Meshy planner is not callable.")


def _invoke_executor(
    executor: MeshyExecutor | Callable[[MeshyGenerationResult], GeneratedModel],
    result: MeshyGenerationResult,
) -> GeneratedModel:
    execute_method = getattr(executor, "execute", None)
    if callable(execute_method):
        return execute_method(result)
    if callable(executor):
        return executor(result)
    raise TypeError("The Meshy executor is not callable.")


def _safe_error_message(
    error: Exception,
    settings: GenerationServiceSettings,
) -> str:
    message = str(error).strip() or error.__class__.__name__
    if settings.meshy_api_key:
        message = message.replace(settings.meshy_api_key, "[redacted]")
    return message


# ### Image helpers ###
def _encode_png(image: np.ndarray) -> bytes:
    did_encode, encoded_image = cv2.imencode(".png", np.asarray(image))
    if not did_encode:
        raise ValueError("Unable to encode the selected object image.")
    return bytes(encoded_image)


def _default_generation_asset_directory() -> Path:
    root = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppDataLocation
    )
    base_directory = Path(root) if root else Path.home() / ".housemaker"
    return base_directory / "generated"


# ### Texture-atlas helpers ###
def _build_model_texture_atlas_entries(
    record: GeneratedObjectRecord,
    model: GeneratedModel,
) -> list[TextureAtlasEntry]:
    """Extract distinct embedded base-color images from every scene material."""

    geometries = list(getattr(model.scene, "geometry", {}).items())
    if not geometries:
        geometries = [("mesh", model.mesh)]

    entries: list[TextureAtlasEntry] = []
    seen_texture_digests: set[str] = set()
    for geometry_name, geometry in sorted(
        geometries,
        key=lambda item: str(item[0]),
    ):
        visual = getattr(geometry, "visual", None)
        material = getattr(visual, "material", None)
        for material_index, texture_source in enumerate(
            _iter_material_texture_sources(material),
            start=1,
        ):
            texture_rgba = _decode_texture_rgba(texture_source)
            if texture_rgba is None:
                continue
            digest_input = (
                np.asarray(texture_rgba.shape, dtype=np.int64).tobytes()
                + texture_rgba.tobytes()
            )
            digest = hashlib.sha256(digest_input).hexdigest()
            if digest in seen_texture_digests:
                continue
            seen_texture_digests.add(digest)
            geometry_label = str(geometry_name).strip() or "material"
            try:
                entry = TextureAtlasEntry(
                    atlas_id=f"{record.object_id}:{digest[:20]}",
                    display_name=(
                        f"{record.object_name} · {geometry_label} "
                        f"· atlas {material_index}"
                    ),
                    image=texture_rgba,
                    owner_id=record.object_id,
                )
            except (TypeError, ValueError):
                continue
            entries.append(entry)
    return entries


def _iter_material_texture_sources(material: object) -> list[object]:
    """Return base-color sources from ordinary and multi-material visuals."""

    if material is None:
        return []
    nested_materials = getattr(material, "materials", None)
    if isinstance(nested_materials, list | tuple):
        sources: list[object] = []
        for nested_material in nested_materials:
            sources.extend(_iter_material_texture_sources(nested_material))
        return sources

    for attribute_name in ("baseColorTexture", "image"):
        texture = getattr(material, attribute_name, None)
        if texture is not None:
            return [texture]
    return []


def _decode_texture_rgba(texture: object) -> np.ndarray | None:
    """Decode one trusted in-memory GLB material image into owned RGBA pixels."""

    try:
        if hasattr(texture, "convert"):
            rgba = np.asarray(texture.convert("RGBA"), dtype=np.uint8)
        elif isinstance(texture, bytes | bytearray | memoryview):
            with Image.open(BytesIO(bytes(texture))) as image:
                rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        else:
            raw_texture = np.asarray(texture)
            if raw_texture.ndim == 1:
                with Image.open(BytesIO(raw_texture.tobytes())) as image:
                    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
            else:
                if np.issubdtype(raw_texture.dtype, np.floating):
                    finite_texture = np.nan_to_num(raw_texture, nan=0.0)
                    if finite_texture.size and float(np.max(finite_texture)) <= 1.0:
                        finite_texture = finite_texture * 255.0
                    raw_texture = finite_texture
                rgba = np.asarray(np.clip(raw_texture, 0, 255), dtype=np.uint8)
    except Exception:
        return None

    if rgba.ndim == 2:
        rgba = np.repeat(rgba[:, :, np.newaxis], 3, axis=2)
    if rgba.ndim != 3 or rgba.shape[2] not in {3, 4}:
        return None
    if rgba.shape[2] == 3:
        alpha = np.full((*rgba.shape[:2], 1), 255, dtype=np.uint8)
        rgba = np.concatenate((rgba, alpha), axis=2)
    if rgba.shape[0] <= 0 or rgba.shape[1] <= 0:
        return None
    return np.ascontiguousarray(rgba)


# ### Model-statistics helpers ###
def _format_model_statistics(model: GeneratedModel | None) -> str:
    """Describe the preview mesh without relying on provider-specific metadata."""

    if model is None:
        return "No generated object"

    mesh = model.mesh
    vertex_count = len(mesh.vertices)
    triangle_count = len(mesh.faces)
    bounds = np.asarray(mesh.bounds, dtype=float)
    dimensions = bounds[1] - bounds[0] if bounds.shape == (2, 3) else None
    dimensions_text = "unknown"
    if dimensions is not None and np.all(np.isfinite(dimensions)):
        dimensions_text = " × ".join(
            f"{dimension:.2f}" for dimension in dimensions
        ) + " m"

    visual_kind = str(getattr(getattr(mesh, "visual", None), "kind", "none"))
    has_texture = visual_kind == "texture"
    has_vertex_colors = visual_kind in {"vertex", "face"}
    material_count, texture_count = _get_model_material_statistics(model)
    appearance = (
        "embedded texture"
        if has_texture
        else "baked texture colors"
        if has_vertex_colors
        else "material color"
    )
    return (
        f"{vertex_count:,} vertices · {triangle_count:,} triangles (polycount) · "
        f"{dimensions_text} · {material_count} material"
        f"{'s' if material_count != 1 else ''} · {texture_count} texture"
        f"{'s' if texture_count != 1 else ''} · {appearance}"
    )


def _get_model_material_statistics(model: GeneratedModel) -> tuple[int, int]:
    """Count distinct preview materials and base-color textures when available."""

    geometries = list(getattr(model.scene, "geometry", {}).values())
    if not geometries:
        geometries = [model.mesh]

    materials: set[int] = set()
    texture_count = 0
    for geometry in geometries:
        visual = getattr(geometry, "visual", None)
        material = getattr(visual, "material", None)
        if material is not None:
            materials.add(id(material))
        if getattr(visual, "kind", None) == "texture":
            texture_count += 1
    return len(materials), texture_count


# ### Widget helpers ###
def _build_labeled_view(title: str, content: QWidget) -> QWidget:
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    title_label = QLabel(title)
    title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    title_label.setStyleSheet("font-weight: 600;")
    layout.addWidget(title_label)
    layout.addWidget(content, 1)
    return container
