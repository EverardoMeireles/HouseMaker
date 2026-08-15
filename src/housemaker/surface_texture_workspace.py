# ### Imports ###
from __future__ import annotations

import math
import os
import tempfile
import threading
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from shiboken6 import isValid as is_valid_qt_object

from housemaker.camera_models import InitialFirstPersonCamera
from housemaker.app_settings import ApplicationSettingsStore
from housemaker.generation_state import MASK_MODE_ERASE, MASK_MODE_PAINT, MaskStroke
from housemaker.generation_views import VideoInpaintView, rasterize_mask_strokes
from housemaker.models import LevelData
from housemaker.settings_widget import (
    SURFACE_TEXTURE_PROVIDER_OPTIONS,
    SURFACE_TEXTURE_PROVIDER_SETTING_KEY,
    GenerationServiceSettings,
    read_surface_texture_provider,
)
from housemaker.surface_texture_providers import (
    SurfaceTextureResult,
    request_surface_texture,
)
from housemaker.surface_texture_state import (
    SurfaceTextureAssignment,
    SurfaceTextureData,
)
from housemaker.surface_texture_viewer import SurfaceTextureViewer
from housemaker.video_source import VIDEO_FILE_FILTER, VideoFrameSource, probe_video


# ### Constants ###
DEFAULT_BRUSH_RADIUS_PIXELS = 24
MIN_BRUSH_RADIUS_PIXELS = 1
MAX_BRUSH_RADIUS_PIXELS = 256
MAX_PROVIDER_REFERENCE_IMAGES = 5
MAX_REFERENCE_FRAMES = 100
REFERENCE_PADDING_RATIO = 0.08
INTERRUPT_POLL_SECONDS = 0.05
SHUTDOWN_WAIT_MILLISECONDS = 250


# ### Request models ###
@dataclass(frozen=True)
class SurfaceTextureRequest:
    provider: str
    api_key: str
    reference_pngs: tuple[bytes, ...]
    reference_frame_indices: tuple[int, ...]
    surface_type: str
    surface_ids: tuple[str, ...]
    combined_area_m2: float
    prompt: str
    existing_texture_png: bytes | None = None
    edit_mask_png: bytes | None = None
    surface_edit_mask_pngs: tuple[tuple[str, bytes], ...] = ()


class SurfaceTextureProvider(Protocol):
    def generate(
        self,
        request: SurfaceTextureRequest,
        progress_callback: Callable[[str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> SurfaceTextureResult:
        ...


# ### Provider adapter ###
class DefaultSurfaceTextureProvider:
    def generate(
        self,
        request: SurfaceTextureRequest,
        progress_callback: Callable[[str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> SurfaceTextureResult:
        return request_surface_texture(
            provider=request.provider,
            api_key=request.api_key,
            reference_pngs=request.reference_pngs,
            prompt=request.prompt,
            existing_texture_png=request.existing_texture_png,
            edit_mask_png=request.edit_mask_png,
            progress_callback=(
                None
                if progress_callback is None
                else lambda status, progress: progress_callback(
                    f"{str(status).replace('_', ' ').title()} ({int(progress)}%)"
                )
            ),
            cancel_event=cancel_event,
        )


# ### Background worker ###
class SurfaceTextureWorker(QObject):
    succeeded = Signal(object, object)
    failed = Signal(str)
    finished = Signal()
    progress = Signal(str)

    def __init__(
        self,
        provider: SurfaceTextureProvider | Callable[[SurfaceTextureRequest], object],
        request: SurfaceTextureRequest,
    ) -> None:
        super().__init__()
        self._provider = provider
        self._request = request
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    @Slot()
    def run(self) -> None:
        try:
            result = _run_interruptible_stage(
                lambda: _invoke_provider(
                    self._provider,
                    self._request,
                    self.progress.emit,
                    self._cancel_event,
                )
            )
            if not isinstance(result, SurfaceTextureResult):
                raise TypeError("The texture provider returned an invalid result.")
            if not self._cancel_event.is_set():
                self.succeeded.emit(self._request, result)
        except _SurfaceTextureCancelled:
            pass
        except Exception as error:
            if not self._cancel_event.is_set():
                self.failed.emit(_redact_error(error, self._request.api_key))
        finally:
            self.finished.emit()


# ### Surface texture workspace ###
class SurfaceTextureGenerationWorkspace(QWidget):
    """Select fixed surfaces and texture them from masked video references."""

    data_changed = Signal(object)
    generation_completed = Signal(object)

    def __init__(
        self,
        provider: SurfaceTextureProvider
        | Callable[[SurfaceTextureRequest], object]
        | None = None,
        asset_directory: str | Path | None = None,
        application_settings: ApplicationSettingsStore | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._provider = provider or DefaultSurfaceTextureProvider()
        self._asset_directory = (
            Path(asset_directory).expanduser()
            if asset_directory is not None
            else Path.cwd() / "surface_textures"
        )
        self._application_settings = (
            application_settings
            if application_settings is not None
            else ApplicationSettingsStore()
        )
        self._settings = GenerationServiceSettings(
            surface_texture_provider=read_surface_texture_provider(
                self._application_settings
            )
        )
        self._is_syncing_provider = False
        self._data = SurfaceTextureData()
        self._levels: list[LevelData] = []
        self._initial_camera: InitialFirstPersonCamera | None = None
        self._video_source: VideoFrameSource | None = None
        self._displayed_frame_index: int | None = None
        self._is_syncing_seekbar = False
        self._generation_thread: QThread | None = None
        self._generation_worker: SurfaceTextureWorker | None = None

        self._build_ui()
        self._sync_video_controls()
        self._sync_controls()

    @property
    def is_generating(self) -> bool:
        return self._generation_thread is not None

    def get_data(self) -> SurfaceTextureData:
        self._store_displayed_frame_strokes()
        self._store_viewer_state()
        return self._data.clone()

    def get_surface_material_sources(self) -> dict[str, Path]:
        """Resolve live assignment assets with later assignments taking priority."""

        material_sources: dict[str, Path] = {}
        for assignment in self._data.assignments:
            try:
                texture_path = self._resolve_asset_path(assignment.asset_path)
            except ValueError:
                continue
            if not texture_path.is_file():
                continue
            for surface_id in assignment.surface_ids:
                material_sources[surface_id] = texture_path
        return material_sources

    def set_data(self, data: SurfaceTextureData | None) -> None:
        if self.is_generating:
            raise RuntimeError(
                "Cannot replace surface texture data while generating."
            )
        self._close_video_source()
        self._displayed_frame_index = None
        self._data = SurfaceTextureData() if data is None else data.clone()
        metadata = self._data.video_metadata
        if metadata is not None and Path(metadata.path).exists():
            try:
                self._video_source = VideoFrameSource(metadata.path)
                self._data.video_metadata = self._video_source.metadata
            except ValueError as error:
                self.status_label.setText(str(error))
        elif metadata is not None:
            self.status_label.setText(f"Video missing: {metadata.path}")

        self.video_view.clear_frame("Load a source video to paint references")
        if self._data.camera_pose is None:
            self.surface_view.set_levels(
                self._levels,
                self._initial_camera,
            )
        self._restore_viewer_state()
        self._restore_assignment_textures()
        self._sync_video_controls()
        if self._video_source is not None:
            self.show_frame(self._data.current_frame_index)
        self._sync_selection_status()
        self._sync_controls()

    def set_levels(
        self,
        levels: Sequence[LevelData],
        initial_camera: InitialFirstPersonCamera | None = None,
    ) -> None:
        if self._levels:
            self._store_viewer_state()
        self._levels = list(levels)
        self._initial_camera = initial_camera
        self.surface_view.set_levels(self._levels, initial_camera)
        self._restore_viewer_state()
        self._restore_assignment_textures()
        self._sync_selection_status()
        self._sync_controls()

    def set_runtime_settings(self, settings: GenerationServiceSettings) -> None:
        if not isinstance(settings, GenerationServiceSettings):
            raise TypeError("Generation settings have an invalid type.")
        self._settings = settings
        self._sync_provider_combo(settings.surface_texture_provider)
        self._sync_controls()

    def get_runtime_settings(self) -> GenerationServiceSettings:
        return self._settings

    def set_provider(
        self,
        provider: SurfaceTextureProvider
        | Callable[[SurfaceTextureRequest], object],
    ) -> None:
        self._provider = provider

    def load_video(self, video_path: str) -> None:
        if self.is_generating:
            raise RuntimeError("Cannot replace the video while generating.")
        metadata = probe_video(video_path)
        next_source = VideoFrameSource(metadata.path)
        self._close_video_source()
        self._displayed_frame_index = None
        self._video_source = next_source
        self._data.video_metadata = next_source.metadata
        self._data.current_frame_index = 0
        self._data.frame_strokes = {}
        self.video_view.clear_frame()
        self._sync_video_controls()
        self.show_frame(0)
        self.status_label.setText(
            "Video loaded. Paint material references on as many frames as useful; "
            "right-click fills an area enclosed by the painted outline."
        )
        self._emit_data_changed()

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
        if self.is_generating:
            return
        request = self._build_request()
        if request is None:
            return
        self._start_generation(request)

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
                thread.finished.disconnect(self._handle_thread_finished)
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

        self.views_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.views_splitter.setChildrenCollapsible(False)

        self.video_view = VideoInpaintView()
        self.video_view.strokes_changed.connect(
            self._handle_video_strokes_changed
        )
        self.views_splitter.addWidget(
            _build_labeled_view("Video material references", self.video_view)
        )

        self.surface_view = SurfaceTextureViewer()
        self.surface_view.selection_changed.connect(
            self._handle_surface_selection_changed
        )
        self.surface_view.camera_pose_changed.connect(
            self._handle_camera_pose_changed
        )
        self.surface_view.texture_mask_strokes_changed.connect(
            self._handle_texture_mask_strokes_changed
        )
        self.views_splitter.addWidget(
            _build_labeled_view("Fixed surfaces", self.surface_view)
        )
        self.views_splitter.setStretchFactor(0, 1)
        self.views_splitter.setStretchFactor(1, 1)
        self.views_splitter.setSizes([1_000, 1_000])
        root_layout.addWidget(self.views_splitter, 1)

        self.seekbar = QSlider(Qt.Orientation.Horizontal)
        self.seekbar.setRange(0, 0)
        self.seekbar.valueChanged.connect(self._handle_seekbar_changed)
        root_layout.addWidget(self.seekbar)

        first_row = QHBoxLayout()
        self.load_video_button = QPushButton("Load video")
        self.load_video_button.clicked.connect(self._handle_load_video_clicked)
        first_row.addWidget(self.load_video_button)
        self.frame_label = QLabel("Frame 0 / 0")
        first_row.addWidget(self.frame_label)
        self.painted_frames_label = QLabel("0 painted frames")
        first_row.addWidget(self.painted_frames_label)
        self.selection_label = QLabel("No surface selected")
        self.selection_label.setObjectName("surface_selection_label")
        first_row.addWidget(self.selection_label, 1)
        first_row.addWidget(QLabel("Surface texture provider"))
        self.surface_texture_provider_combo = QComboBox()
        self.surface_texture_provider_combo.setObjectName(
            "surface_texture_provider_combo"
        )
        for label, provider_id in SURFACE_TEXTURE_PROVIDER_OPTIONS:
            self.surface_texture_provider_combo.addItem(label, provider_id)
        self._sync_provider_combo(self._settings.surface_texture_provider)
        self.surface_texture_provider_combo.currentIndexChanged.connect(
            self._handle_surface_texture_provider_changed
        )
        first_row.addWidget(self.surface_texture_provider_combo)
        root_layout.addLayout(first_row)

        second_row = QHBoxLayout()
        self.paint_mask_button = QRadioButton("Paint")
        self.erase_mask_button = QRadioButton("Erase")
        self.paint_mask_button.setChecked(True)
        self._mask_mode_group = QButtonGroup(self)
        self._mask_mode_group.addButton(self.paint_mask_button)
        self._mask_mode_group.addButton(self.erase_mask_button)
        self.paint_mask_button.toggled.connect(self._handle_mask_mode_changed)
        second_row.addWidget(self.paint_mask_button)
        second_row.addWidget(self.erase_mask_button)
        self.inpaint_3d_button = QPushButton("Inpaint 3D texture")
        self.inpaint_3d_button.setCheckable(True)
        self.inpaint_3d_button.toggled.connect(
            self._handle_3d_inpaint_toggled
        )
        second_row.addWidget(self.inpaint_3d_button)
        second_row.addWidget(QLabel("Brush"))
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
        self.brush_size_spinbox.valueChanged.connect(
            self.surface_view.set_inpaint_brush_radius_pixels
        )
        second_row.addWidget(self.brush_size_spinbox)
        self.undo_mask_button = QPushButton("Undo stroke")
        self.undo_mask_button.clicked.connect(self.video_view.undo_last_stroke)
        second_row.addWidget(self.undo_mask_button)
        self.clear_mask_button = QPushButton("Clear frame mask")
        self.clear_mask_button.clicked.connect(self.video_view.clear_mask)
        second_row.addWidget(self.clear_mask_button)
        self.undo_3d_mask_button = QPushButton("Undo 3D stroke")
        self.undo_3d_mask_button.clicked.connect(
            self.surface_view.undo_last_texture_mask_stroke
        )
        second_row.addWidget(self.undo_3d_mask_button)
        self.clear_3d_mask_button = QPushButton("Clear 3D mask")
        self.clear_3d_mask_button.clicked.connect(
            self.surface_view.clear_texture_mask
        )
        second_row.addWidget(self.clear_3d_mask_button)
        self.material_notes_edit = QLineEdit()
        self.material_notes_edit.setPlaceholderText(
            "Optional material notes, e.g. pale oak or matte plaster"
        )
        second_row.addWidget(self.material_notes_edit, 1)
        self.generate_button = QPushButton("Generate texture")
        self.generate_button.setMinimumHeight(38)
        self.generate_button.clicked.connect(self.generate)
        second_row.addWidget(self.generate_button)
        root_layout.addLayout(second_row)

        self.status_label = QLabel(
            "Left-click the 3D view to enter first-person mode. Move with Z/Q/S/D, "
            "look with the mouse, select at the crosshair, and right-click to exit."
        )
        self.status_label.setWordWrap(True)
        root_layout.addWidget(self.status_label)

    def _start_generation(self, request: SurfaceTextureRequest) -> None:
        if self.inpaint_3d_button.isChecked():
            self.inpaint_3d_button.setChecked(False)
        self._generation_thread = QThread(self)
        self._generation_worker = SurfaceTextureWorker(self._provider, request)
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
        self._generation_thread.finished.connect(self._handle_thread_finished)
        self._generation_thread.finished.connect(
            self._generation_thread.deleteLater
        )
        self.status_label.setText(
            f"Sending {len(request.reference_frame_indices)} painted frame(s) "
            f"to {_get_provider_display_name(request.provider)}..."
        )
        self._generation_thread.start()
        self._sync_controls()

    def _build_request(self) -> SurfaceTextureRequest | None:
        self._store_displayed_frame_strokes()
        selected_ids = tuple(self.surface_view.get_selected_surface_ids())
        surface_type = self.surface_view.get_selected_surface_type()
        if not selected_ids or not surface_type:
            self.status_label.setText("Select at least one fixed surface first.")
            return None
        frame_indices = tuple(sorted(self._data.frame_strokes))
        if not frame_indices:
            self.status_label.setText(
                "Paint at least one material reference in the video."
            )
            return None
        api_key = self._settings.surface_texture_api_key
        if not api_key:
            self.status_label.setText(
                f"Configure the {_get_provider_display_name(self._settings.surface_texture_provider)} "
                "API key in Settings."
            )
            return None
        try:
            reference_pngs, used_frame_indices = self._build_reference_pngs(
                frame_indices
            )
        except (OSError, ValueError) as error:
            self.status_label.setText(str(error))
            return None
        if not reference_pngs:
            self.status_label.setText("The painted reference masks are empty.")
            return None
        existing_texture_png: bytes | None = None
        edit_mask_png: bytes | None = None
        masked_surface_ids = self.surface_view.get_masked_selected_surface_ids()
        if masked_surface_ids:
            try:
                edit_data = self.surface_view.get_selected_texture_edit_data()
            except ValueError as error:
                self.status_label.setText(str(error))
                return None
            if edit_data is None:
                self.status_label.setText("Paint an edit region on the 3D texture.")
                return None
            base_rgba, editable_mask = edit_data
            existing_texture_png = _encode_rgba_png(base_rgba)
            edit_mask_png = _encode_png(editable_mask)
            surface_edit_mask_pngs = tuple(
                (surface_id, _encode_png(mask))
                for surface_id, mask in (
                    self.surface_view.get_selected_texture_edit_masks().items()
                )
            )
            selected_ids = masked_surface_ids
            area_m2 = self.surface_view.get_combined_masked_selected_area()
        else:
            area_m2 = self.surface_view.get_combined_selected_area()
            surface_edit_mask_pngs = ()
        prompt = _build_material_prompt(
            surface_type,
            area_m2,
            self.material_notes_edit.text(),
        )
        return SurfaceTextureRequest(
            provider=self._settings.surface_texture_provider,
            api_key=api_key,
            reference_pngs=reference_pngs,
            reference_frame_indices=used_frame_indices,
            surface_type=surface_type,
            surface_ids=selected_ids,
            combined_area_m2=area_m2,
            prompt=prompt,
            existing_texture_png=existing_texture_png,
            edit_mask_png=edit_mask_png,
            surface_edit_mask_pngs=surface_edit_mask_pngs,
        )

    def _build_reference_pngs(
        self,
        frame_indices: Sequence[int],
    ) -> tuple[tuple[bytes, ...], tuple[int, ...]]:
        if self._video_source is None:
            raise ValueError("Load the source video before generating.")
        limited_indices = _sample_frame_indices(
            tuple(frame_indices),
            MAX_REFERENCE_FRAMES,
        )
        crops: list[np.ndarray] = []
        used_indices: list[int] = []
        for frame_index in limited_indices:
            frame = self._video_source.get_frame(frame_index)
            strokes = self._data.strokes_for_frame(frame_index)
            crop = _build_masked_crop(frame, strokes)
            if crop.size == 0:
                continue
            crops.append(crop)
            used_indices.append(frame_index)
        packed_images = _pack_reference_crops(
            crops,
            MAX_PROVIDER_REFERENCE_IMAGES,
        )
        return tuple(_encode_png(image) for image in packed_images), tuple(used_indices)

    @Slot(object)
    def _handle_surface_selection_changed(self, raw_ids: object) -> None:
        if not isinstance(raw_ids, tuple | list):
            return
        self._data.selected_surface_ids = tuple(str(value) for value in raw_ids)
        self._data.selected_surface_type = self.surface_view.get_selected_surface_type()
        if (
            self.inpaint_3d_button.isChecked()
            and not self.surface_view.can_inpaint_selection()
        ):
            self.inpaint_3d_button.setChecked(False)
        self._sync_selection_status()
        self._emit_data_changed()
        self._sync_controls()

    @Slot(object)
    def _handle_camera_pose_changed(self, pose: object) -> None:
        try:
            self._data.camera_pose = pose
        except (TypeError, ValueError):
            return
        self._emit_data_changed()

    def _handle_video_strokes_changed(self, raw_strokes: object) -> None:
        if not isinstance(raw_strokes, list) or self._displayed_frame_index is None:
            return
        self._data.set_frame_strokes(
            self._displayed_frame_index,
            self.video_view.get_strokes(),
        )
        self._sync_painted_frames_label()
        self._emit_data_changed()
        self._sync_controls()

    @Slot(object)
    def _handle_texture_mask_strokes_changed(self, raw_strokes: object) -> None:
        if not isinstance(raw_strokes, dict):
            return
        self._data.texture_mask_strokes = {
            str(surface_id): list(strokes)
            for surface_id, strokes in raw_strokes.items()
        }
        self._emit_data_changed()
        self._sync_controls()

    @Slot(object, object)
    def _handle_generation_succeeded(
        self,
        request: SurfaceTextureRequest,
        result: SurfaceTextureResult,
    ) -> None:
        try:
            outputs = _build_surface_texture_outputs(request, result.texture_png)
        except ValueError as error:
            self._handle_generation_failed(str(error))
            return
        saved_outputs: list[tuple[tuple[str, ...], str, bytes]] = []
        try:
            for surface_ids, texture_png in outputs:
                assignment_id = uuid.uuid4().hex
                asset_path = self._persist_texture(assignment_id, texture_png)
                saved_outputs.append((surface_ids, asset_path, texture_png))
        except OSError as error:
            for _surface_ids, saved_path, _texture_png in saved_outputs:
                self._resolve_asset_path(saved_path).unlink(missing_ok=True)
            self._handle_generation_failed(
                f"The generated texture could not be saved: {error}"
            )
            return
        assignments: list[SurfaceTextureAssignment] = []
        for surface_ids, asset_path, texture_png in saved_outputs:
            decoded_texture = _decode_png_rgba(texture_png, "Generated texture")
            area_m2 = self.surface_view.get_combined_surface_area(surface_ids)
            assignment = SurfaceTextureAssignment(
                assignment_id=Path(asset_path).stem,
                surface_type=request.surface_type,
                surface_ids=surface_ids,
                provider=result.provider,
                provider_task_id=result.task_id,
                asset_path=asset_path,
                combined_area_m2=area_m2,
                area_description=(
                    f"{len(surface_ids)} {request.surface_type} "
                    f"surface(s), {area_m2:.2f} m²"
                ),
                reference_frame_indices=request.reference_frame_indices,
                texture_width=int(decoded_texture.shape[1]),
                texture_height=int(decoded_texture.shape[0]),
            )
            self._data.assignments.append(assignment)
            assignments.append(assignment)
            self.surface_view.set_surface_texture(surface_ids, texture_png)
        self.surface_view.clear_texture_mask(request.surface_ids)
        self.status_label.setText(
            f"Applied generated texture to {len(request.surface_ids)} "
            f"{request.surface_type} surface(s)."
        )
        self._emit_data_changed()
        for assignment in assignments:
            self.generation_completed.emit(assignment)

    @Slot(str)
    def _handle_generation_failed(self, message: str) -> None:
        self.status_label.setText(message)

    @Slot()
    def _handle_thread_finished(self) -> None:
        self._generation_thread = None
        self._generation_worker = None
        self._sync_controls()

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
        if self._is_syncing_seekbar or self.is_generating:
            return
        self.show_frame(frame_index)

    def _handle_mask_mode_changed(self, paint_checked: bool) -> None:
        mode = MASK_MODE_PAINT if paint_checked else MASK_MODE_ERASE
        self.video_view.set_brush_mode(mode)
        self.surface_view.set_inpaint_brush_mode(mode)

    def _handle_3d_inpaint_toggled(self, checked: bool) -> None:
        try:
            self.surface_view.set_inpaint_enabled(checked)
        except ValueError as error:
            self.inpaint_3d_button.blockSignals(True)
            self.inpaint_3d_button.setChecked(False)
            self.inpaint_3d_button.blockSignals(False)
            self.status_label.setText(str(error))
            return
        self.status_label.setText(
            "Paint the orange edit region directly on selected textured surfaces."
            if checked
            else "3D inpainting stopped."
        )

    def _handle_surface_texture_provider_changed(self, _index: int) -> None:
        if self._is_syncing_provider:
            return
        provider = str(self.surface_texture_provider_combo.currentData())
        self._settings = replace(
            self._settings,
            surface_texture_provider=provider,
        )
        saved = self._application_settings.set(
            SURFACE_TEXTURE_PROVIDER_SETTING_KEY,
            provider,
        )
        if not saved:
            self.status_label.setText(
                "The surface texture provider changed for this session, but "
                "could not be saved to settings.json."
            )
        self._sync_controls()

    def _sync_provider_combo(self, provider: str) -> None:
        if not hasattr(self, "surface_texture_provider_combo"):
            return
        provider_index = self.surface_texture_provider_combo.findData(provider)
        if provider_index < 0:
            provider_index = 0
        self._is_syncing_provider = True
        self.surface_texture_provider_combo.setCurrentIndex(provider_index)
        self._is_syncing_provider = False

    def _store_displayed_frame_strokes(self) -> None:
        if self._displayed_frame_index is None:
            return
        self._data.set_frame_strokes(
            self._displayed_frame_index,
            self.video_view.get_strokes(),
        )

    def _store_viewer_state(self) -> None:
        if not hasattr(self, "surface_view"):
            return
        self._data.camera_pose = self.surface_view.get_camera_pose()
        self._data.selected_surface_ids = tuple(
            self.surface_view.get_selected_surface_ids()
        )
        self._data.selected_surface_type = (
            self.surface_view.get_selected_surface_type()
        )
        self._data.texture_mask_strokes = (
            self.surface_view.get_texture_mask_strokes()
        )

    def _restore_viewer_state(self) -> None:
        if self._data.camera_pose is not None:
            self.surface_view.set_camera_pose(self._data.camera_pose)
        self.surface_view.set_selected_surface_ids(
            self._data.selected_surface_ids
        )
        self.surface_view.set_texture_mask_strokes(
            self._data.texture_mask_strokes
        )

    def _restore_assignment_textures(self) -> None:
        self.surface_view.clear_surface_textures()
        for assignment in self._data.assignments:
            try:
                texture_path = self._resolve_asset_path(assignment.asset_path)
                self.surface_view.set_surface_texture(
                    assignment.surface_ids,
                    texture_path,
                )
            except (OSError, ValueError):
                continue

    def _persist_texture(self, assignment_id: str, texture_png: bytes) -> str:
        self._asset_directory.mkdir(parents=True, exist_ok=True)
        file_name = f"{assignment_id}.png"
        destination = self._asset_directory / file_name
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{file_name}.",
            suffix=".tmp",
            dir=str(self._asset_directory),
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(bytes(texture_png))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, destination)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        return file_name

    def _resolve_asset_path(self, raw_path: str) -> Path:
        candidate = (self._asset_directory / raw_path).resolve()
        root = self._asset_directory.resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError("The saved surface texture path is unsafe.") from error
        return candidate

    def _clamp_frame_index(self, frame_index: int) -> int:
        if self._video_source is None:
            return 0
        return min(
            max(int(frame_index), 0),
            self._video_source.metadata.frame_count - 1,
        )

    def _sync_video_controls(self) -> None:
        if self._video_source is None:
            self.seekbar.setRange(0, 0)
            self.frame_label.setText("Frame 0 / 0")
        else:
            self.seekbar.setRange(
                0,
                self._video_source.metadata.frame_count - 1,
            )
            self._sync_seekbar_value(self._data.current_frame_index)
            self._sync_frame_label()
        self._sync_painted_frames_label()

    def _sync_seekbar_value(self, frame_index: int) -> None:
        self._is_syncing_seekbar = True
        self.seekbar.setValue(int(frame_index))
        self._is_syncing_seekbar = False

    def _sync_frame_label(self) -> None:
        total = (
            0
            if self._video_source is None
            else self._video_source.metadata.frame_count
        )
        current = 0 if total == 0 else self._data.current_frame_index + 1
        self.frame_label.setText(f"Frame {current} / {total}")

    def _sync_painted_frames_label(self) -> None:
        count = len(self._data.frame_strokes)
        suffix = "frame" if count == 1 else "frames"
        self.painted_frames_label.setText(f"{count} painted {suffix}")

    def _sync_selection_status(self) -> None:
        surface_ids = self.surface_view.get_selected_surface_ids()
        surface_type = self.surface_view.get_selected_surface_type()
        if not surface_ids or surface_type is None:
            self.selection_label.setText("No surface selected")
            return
        area = self.surface_view.get_combined_selected_area()
        self.selection_label.setText(
            f"{len(surface_ids)} {surface_type} surface(s), {area:.2f} m²"
        )

    def _sync_controls(self) -> None:
        busy = self.is_generating
        has_video = self._video_source is not None
        has_mask = bool(self._data.frame_strokes) or self.video_view.has_selection()
        has_surface = bool(self.surface_view.get_selected_surface_ids())
        has_key = bool(self._settings.surface_texture_api_key)
        self.load_video_button.setEnabled(not busy)
        self.seekbar.setEnabled(has_video and not busy)
        self.paint_mask_button.setEnabled(has_video and not busy)
        self.erase_mask_button.setEnabled(has_video and not busy)
        self.brush_size_spinbox.setEnabled(has_video and not busy)
        self.undo_mask_button.setEnabled(has_video and not busy)
        self.clear_mask_button.setEnabled(
            has_video and self.video_view.has_selection() and not busy
        )
        can_inpaint_3d = self.surface_view.can_inpaint_selection()
        self.inpaint_3d_button.setEnabled(can_inpaint_3d and not busy)
        self.undo_3d_mask_button.setEnabled(
            self.surface_view.has_selected_texture_mask() and not busy
        )
        self.clear_3d_mask_button.setEnabled(
            self.surface_view.has_selected_texture_mask() and not busy
        )
        self.material_notes_edit.setEnabled(not busy)
        self.surface_texture_provider_combo.setEnabled(not busy)
        self.generate_button.setEnabled(
            has_video and has_mask and has_surface and has_key and not busy
        )
        self.video_view.set_interaction_enabled(has_video and not busy)

    def _emit_data_changed(self) -> None:
        self.data_changed.emit(self._data.clone())

    def _close_video_source(self) -> None:
        if self._video_source is not None:
            self._video_source.close()
        self._video_source = None


# ### Worker helpers ###
class _SurfaceTextureCancelled(Exception):
    pass


def _run_interruptible_stage(operation: Callable[[], object]) -> object:
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
        name="HouseMakerSurfaceTextureStage",
        daemon=True,
    )
    operation_thread.start()
    qt_thread = QThread.currentThread()
    while not completed.wait(INTERRUPT_POLL_SECONDS):
        if qt_thread.isInterruptionRequested():
            raise _SurfaceTextureCancelled
    if qt_thread.isInterruptionRequested():
        raise _SurfaceTextureCancelled
    succeeded, value = outcome[0]
    if succeeded:
        return value
    if isinstance(value, Exception):
        raise value
    raise RuntimeError("Surface texture generation failed without an exception.")


def _invoke_provider(
    provider: SurfaceTextureProvider | Callable[[SurfaceTextureRequest], object],
    request: SurfaceTextureRequest,
    progress_callback: Callable[[str], None],
    cancel_event: threading.Event,
) -> object:
    generate_method = getattr(provider, "generate", None)
    if callable(generate_method):
        if isinstance(provider, DefaultSurfaceTextureProvider):
            return generate_method(request, progress_callback, cancel_event)
        return generate_method(request)
    if callable(provider):
        return provider(request)
    raise TypeError("The surface texture provider is not callable.")


# ### Reference image helpers ###
def _build_masked_crop(
    frame_bgr: np.ndarray,
    strokes: list[MaskStroke],
) -> np.ndarray:
    frame = np.asarray(frame_bgr)
    if frame.ndim != 3 or frame.shape[2] != 3:
        return np.empty((0, 0, 4), dtype=np.uint8)
    mask = rasterize_mask_strokes((frame.shape[1], frame.shape[0]), strokes)
    if mask.size == 0 or not np.any(mask > 0):
        return np.empty((0, 0, 4), dtype=np.uint8)
    rows, columns = np.nonzero(mask > 0)
    width = int(columns.max() - columns.min() + 1)
    height = int(rows.max() - rows.min() + 1)
    padding = int(round(max(width, height) * REFERENCE_PADDING_RATIO))
    left = max(0, int(columns.min()) - padding)
    right = min(frame.shape[1], int(columns.max()) + padding + 1)
    top = max(0, int(rows.min()) - padding)
    bottom = min(frame.shape[0], int(rows.max()) + padding + 1)
    crop = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2BGRA)
    crop_mask = mask[top:bottom, left:right]
    crop[crop_mask == 0, :3] = 0
    crop[:, :, 3] = crop_mask
    return np.ascontiguousarray(crop)


def _pack_reference_crops(
    crops: Sequence[np.ndarray],
    maximum_images: int,
) -> list[np.ndarray]:
    normalized = [np.asarray(crop) for crop in crops if np.asarray(crop).size]
    if len(normalized) <= maximum_images:
        return normalized
    groups: list[list[np.ndarray]] = [[] for _ in range(maximum_images)]
    for index, crop in enumerate(normalized):
        groups[index % maximum_images].append(crop)
    return [_build_contact_sheet(group) for group in groups if group]


def _build_contact_sheet(crops: Sequence[np.ndarray]) -> np.ndarray:
    tile_size = 384
    columns = min(3, max(1, math.ceil(math.sqrt(len(crops)))))
    rows = math.ceil(len(crops) / columns)
    sheet = np.zeros((rows * tile_size, columns * tile_size, 4), dtype=np.uint8)
    for index, crop in enumerate(crops):
        height, width = crop.shape[:2]
        scale = min(tile_size / max(width, 1), tile_size / max(height, 1))
        resized = cv2.resize(
            crop,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        row = index // columns
        column = index % columns
        top = row * tile_size + (tile_size - resized.shape[0]) // 2
        left = column * tile_size + (tile_size - resized.shape[1]) // 2
        sheet[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return sheet


def _sample_frame_indices(
    frame_indices: tuple[int, ...],
    maximum_count: int,
) -> tuple[int, ...]:
    if len(frame_indices) <= maximum_count:
        return frame_indices
    positions = np.linspace(0, len(frame_indices) - 1, maximum_count)
    return tuple(frame_indices[int(round(position))] for position in positions)


def _encode_png(image: np.ndarray) -> bytes:
    did_encode, encoded = cv2.imencode(".png", np.asarray(image))
    if not did_encode:
        raise ValueError("Unable to encode a painted video reference.")
    return bytes(encoded)


def _encode_rgba_png(image_rgba: np.ndarray) -> bytes:
    rgba = np.asarray(image_rgba, dtype=np.uint8)
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("A base texture must contain RGBA pixels.")
    return _encode_png(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))


def _decode_png_rgba(png_bytes: bytes, label: str) -> np.ndarray:
    decoded = cv2.imdecode(
        np.frombuffer(bytes(png_bytes), dtype=np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    if decoded is None:
        raise ValueError(f"{label} is not a valid PNG image.")
    if decoded.ndim == 2:
        decoded = cv2.cvtColor(decoded, cv2.COLOR_GRAY2RGBA)
    elif decoded.shape[2] == 3:
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGBA)
    elif decoded.shape[2] == 4:
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGBA)
    else:
        raise ValueError(f"{label} has unsupported color channels.")
    return np.ascontiguousarray(decoded, dtype=np.uint8)


def _decode_png_mask(png_bytes: bytes, expected_shape: tuple[int, int]) -> np.ndarray:
    decoded = cv2.imdecode(
        np.frombuffer(bytes(png_bytes), dtype=np.uint8),
        cv2.IMREAD_GRAYSCALE,
    )
    if decoded is None or decoded.shape != expected_shape:
        raise ValueError("A 3D edit mask has invalid dimensions.")
    return np.where(decoded > 0, 255, 0).astype(np.uint8)


def _build_surface_texture_outputs(
    request: SurfaceTextureRequest,
    generated_texture_png: bytes,
) -> list[tuple[tuple[str, ...], bytes]]:
    if not request.surface_edit_mask_pngs:
        _decode_png_rgba(generated_texture_png, "Generated texture")
        return [(request.surface_ids, bytes(generated_texture_png))]
    if request.existing_texture_png is None:
        raise ValueError("A partial texture edit is missing its base texture.")
    base = _decode_png_rgba(request.existing_texture_png, "Base texture")
    generated = _decode_png_rgba(generated_texture_png, "Generated texture")
    if generated.shape[:2] != base.shape[:2]:
        generated = cv2.resize(
            generated,
            (base.shape[1], base.shape[0]),
            interpolation=cv2.INTER_LANCZOS4,
        )
    groups: dict[bytes, tuple[list[str], np.ndarray]] = {}
    for surface_id, mask_png in request.surface_edit_mask_pngs:
        mask = _decode_png_mask(mask_png, base.shape[:2])
        key = mask.tobytes()
        group = groups.get(key)
        if group is None:
            groups[key] = ([surface_id], mask)
        else:
            group[0].append(surface_id)
    outputs: list[tuple[tuple[str, ...], bytes]] = []
    for surface_ids, mask in groups.values():
        composite = base.copy()
        composite[mask > 0] = generated[mask > 0]
        outputs.append((tuple(surface_ids), _encode_rgba_png(composite)))
    return outputs


# ### Text helpers ###
def _build_material_prompt(
    surface_type: str,
    area_m2: float,
    notes: str,
) -> str:
    normalized_notes = str(notes).strip()
    prompt = (
        f"Create one square, seamless, tileable base-color material texture for "
        f"a {surface_type}. The selected surfaces cover {area_m2:.2f} square "
        "meters. Match the material, colors, pattern, and physical scale visible "
        "inside every supplied masked reference. Return a flat orthographic "
        "material swatch with even lighting: no perspective, room geometry, "
        "object silhouette, borders, text, shadows, or highlights. Make opposite "
        "edges tile without a seam."
    )
    if normalized_notes:
        prompt += f" Additional direction: {normalized_notes}."
    return prompt


def _get_provider_display_name(provider: str) -> str:
    names = {
        "meshy": "Meshy",
        "gpt-4o-mini": "GPT-4o-mini",
        "gpt-5.6-luna": "GPT-5.6 Luna",
        "gpt-5.6-terra": "GPT-5.6 Terra",
    }
    return names.get(str(provider), str(provider))


def _redact_error(error: Exception, api_key: str) -> str:
    message = str(error).strip() or error.__class__.__name__
    if api_key:
        message = message.replace(api_key, "[redacted]")
    return message


# ### Widget helpers ###
def _build_labeled_view(title: str, content: QWidget) -> QWidget:
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    label = QLabel(title)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setStyleSheet("font-weight: 600;")
    layout.addWidget(label)
    layout.addWidget(content, 1)
    return container
