# ### Imports ###
from __future__ import annotations

import math
import os
import tempfile
import threading
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QObject, QSize, QThread, Qt, Signal, Slot
from PySide6.QtGui import QIcon, QKeyEvent, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from shiboken6 import isValid as is_valid_qt_object

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.camera_models import InitialFirstPersonCamera
from housemaker.generation_state import MASK_MODE_ERASE, MASK_MODE_PAINT, MaskStroke
from housemaker.generation_views import VideoInpaintView, rasterize_mask_strokes
from housemaker.glb import GeneratedModel
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
    MAX_LOCALIZED_INPAINT_UNDO_HISTORY,
    SURFACE_TYPE_WALL,
    SurfaceTextureAssignment,
    SurfaceTextureData,
    SurfaceTextureInpaintUndoSnapshot,
    SurfaceTextureVariant,
)
from housemaker.surface_texture_variants import (
    DEFAULT_SURFACE_TEXTURE_RESOLUTION,
    SURFACE_TEXTURE_RESOLUTIONS,
    SurfaceTextureVariants,
    build_surface_texture_variants,
)
from housemaker.surface_texture_viewer import (
    SurfaceTextureViewer,
    rasterize_texture_mask_strokes,
)
from housemaker.texture_atlas_view import TextureAtlasEntry, TextureAtlasView
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
OTHER_TEXTURE_THUMBNAIL_SIZE = QSize(96, 96)
OTHER_TEXTURE_GRID_SIZE = QSize(152, 128)


# ### Level synchronization helpers ###
def _build_level_sync_signature(
    levels: Sequence[LevelData],
    initial_camera: InitialFirstPersonCamera | None,
) -> tuple[object, ...]:
    """Snapshot mutable level content used by the Surface scene."""

    return (
        _freeze_level_sync_value(tuple(levels)),
        _freeze_level_sync_value(initial_camera),
    )


def _freeze_level_sync_value(value: object) -> object:
    """Convert nested model values into a stable, immutable signature."""

    if is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__qualname__,
            tuple(
                (
                    model_field.name,
                    _freeze_level_sync_value(
                        getattr(value, model_field.name)
                    ),
                )
                for model_field in fields(value)
            ),
        )
    if isinstance(value, dict):
        frozen_items = (
            (
                _freeze_level_sync_value(key),
                _freeze_level_sync_value(item_value),
            )
            for key, item_value in value.items()
        )
        return (
            "dict",
            tuple(sorted(frozen_items, key=repr)),
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_level_sync_value(item) for item in value)
    if isinstance(value, set | frozenset):
        return (
            "set",
            tuple(
                sorted(
                    (_freeze_level_sync_value(item) for item in value),
                    key=repr,
                )
            ),
        )
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, Path):
        return ("path", str(value))
    if value is None or isinstance(value, str | bytes | int | bool):
        return value
    return (type(value).__qualname__, repr(value))


# ### Asset revision helpers ###
def _build_surface_asset_revision(
    asset_directory: Path,
    raw_asset_path: object,
) -> tuple[object, ...]:
    """Return a stable local path and cheap revision for one texture file."""

    normalized_path = str(raw_asset_path or "")
    try:
        asset_root = asset_directory.resolve()
        asset_path = (asset_directory / normalized_path).resolve()
        asset_path.relative_to(asset_root)
        asset_stat = asset_path.stat()
    except (OSError, RuntimeError, ValueError):
        return normalized_path, None, None, None
    return (
        str(asset_path),
        int(asset_stat.st_size),
        int(asset_stat.st_mtime_ns),
        int(asset_stat.st_ctime_ns),
    )


def _get_cached_surface_asset_revision(
    revision_cache: dict[str, tuple[object, ...]],
    asset_directory: Path,
    raw_asset_path: object,
) -> tuple[object, ...]:
    """Stat each logical texture path once within one comparison pass."""

    cache_key = str(raw_asset_path or "")
    revision = revision_cache.get(cache_key)
    if revision is None:
        revision = _build_surface_asset_revision(
            asset_directory,
            raw_asset_path,
        )
        revision_cache[cache_key] = revision
    return revision


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


@dataclass(frozen=True)
class _SavedSurfaceTextureOutput:
    """One completely persisted resolution family awaiting state commit."""

    surface_ids: tuple[str, ...]
    assignment_id: str
    variants: tuple[SurfaceTextureVariant, ...]
    texture_png_by_resolution: tuple[tuple[int, bytes], ...]

    def png_for_resolution(self, resolution: int) -> bytes:
        for candidate_resolution, texture_png in self.texture_png_by_resolution:
            if candidate_resolution == int(resolution):
                return texture_png
        raise ValueError("The saved surface texture resolution is unavailable.")


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
    assignments_removed = Signal(object)
    localized_inpaint_undone = Signal(object)
    surface_content_changed = Signal()

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
        self._level_sync_signature: tuple[object, ...] | None = None
        self._video_source: VideoFrameSource | None = None
        self._displayed_frame_index: int | None = None
        self._is_syncing_seekbar = False
        self._generation_thread: QThread | None = None
        self._generation_worker: SurfaceTextureWorker | None = None
        self._texture_atlas_entry_cache: dict[
            str,
            tuple[tuple[object, ...], TextureAtlasEntry],
        ] = {}
        self._texture_variant_entry_targets: dict[str, tuple[str, int]] = {}
        self._other_texture_entry_targets: dict[
            str,
            tuple[str, int | None],
        ] = {}
        self._restored_assignment_texture_signature: (
            tuple[tuple[object, ...], ...] | None
        ) = None
        self._texture_catalog_dependency_signature: (
            tuple[tuple[object, ...], ...] | None
        ) = None
        self._other_texture_list_signature: (
            tuple[tuple[object, ...], ...] | None
        ) = None
        self._texture_resolution_change_handler: (
            Callable[[str, int], bool] | None
        ) = None
        self._is_refreshing_texture_atlases = False

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

    def get_preview_dependency_signature(
        self,
    ) -> tuple[tuple[object, ...], ...]:
        """Snapshot assigned texture files without cloning or decoding images."""

        return self._build_preview_dependency_signature({})

    def _build_preview_dependency_signature(
        self,
        revision_cache: dict[str, tuple[object, ...]],
    ) -> tuple[tuple[object, ...], ...]:
        """Snapshot active PNGs using one caller-owned stat cache."""

        return tuple(
            (
                assignment.assignment_id,
                assignment.selected_texture_resolution,
                assignment.asset_path,
                _get_cached_surface_asset_revision(
                    revision_cache,
                    self._asset_directory,
                    assignment.asset_path,
                ),
            )
            for assignment in self._data.assignments
        )

    def refresh_file_backed_previews(self) -> None:
        """Reload only Surface assets whose on-disk revisions changed."""

        revision_cache: dict[str, tuple[object, ...]] = {}
        material_signature = self.get_preview_dependency_signature()
        for assignment, signature_item in zip(
            self._data.assignments,
            material_signature,
            strict=False,
        ):
            if len(signature_item) >= 4:
                revision_cache[str(assignment.asset_path)] = signature_item[3]
        if material_signature != self._restored_assignment_texture_signature:
            self._restore_assignment_textures()

        catalog_signature = self._build_texture_catalog_dependency_signature(
            revision_cache
        )
        if catalog_signature != self._texture_catalog_dependency_signature:
            self._refresh_texture_atlases()

    def _build_texture_catalog_dependency_signature(
        self,
        revision_cache: dict[str, tuple[object, ...]] | None = None,
    ) -> tuple[tuple[object, ...], ...]:
        """Snapshot active and alternate PNGs shown by Surface controls."""

        cached_revisions = {} if revision_cache is None else revision_cache
        return tuple(
            (
                assignment.assignment_id,
                assignment.surface_type,
                assignment.selected_texture_resolution,
                assignment.asset_path,
                _get_cached_surface_asset_revision(
                    cached_revisions,
                    self._asset_directory,
                    assignment.asset_path,
                ),
                tuple(
                    (
                        variant.resolution,
                        variant.asset_path,
                        _get_cached_surface_asset_revision(
                            cached_revisions,
                            self._asset_directory,
                            variant.asset_path,
                        ),
                    )
                    for variant in assignment.texture_variants
                ),
            )
            for assignment in self._data.assignments
        )

    def get_assignment_asset_path(
        self,
        assignment_id: str,
        resolution: int | None = None,
    ) -> Path | None:
        """Resolve one live assignment variant without exposing the asset root."""

        normalized_id = str(assignment_id).strip()
        assignment = next(
            (
                candidate
                for candidate in self._data.assignments
                if candidate.assignment_id == normalized_id
            ),
            None,
        )
        if assignment is None:
            return None
        asset_path = assignment.asset_path
        if resolution is not None:
            variant = assignment.texture_variant_for_resolution(resolution)
            if variant is None:
                return None
            asset_path = variant.asset_path
        try:
            texture_path = self._resolve_asset_path(asset_path)
        except ValueError:
            return None
        return texture_path if texture_path.is_file() else None

    def get_assignment(
        self,
        assignment_id: str,
    ) -> SurfaceTextureAssignment | None:
        """Return one immutable assignment without cloning all Surface data."""

        return self._assignment_by_id(assignment_id)

    def get_wall_assignments(self) -> tuple[SurfaceTextureAssignment, ...]:
        """Return immutable wall assignments without cloning Surface data."""

        return tuple(
            assignment
            for assignment in self._data.assignments
            if assignment.surface_type == SURFACE_TYPE_WALL
        )

    def can_select_assignment_texture_resolution(
        self,
        assignment_id: str,
        resolution: int,
    ) -> bool:
        """Return whether an exact generated surface variant can be selected."""

        if self.is_generating:
            return False
        texture_path = self.get_assignment_asset_path(
            assignment_id,
            resolution,
        )
        if texture_path is None:
            return False
        try:
            texture = _decode_png_rgba(
                texture_path.read_bytes(),
                "Surface texture variant",
            )
        except (OSError, ValueError):
            return False
        return texture.shape[:2] == (int(resolution), int(resolution))

    def set_texture_resolution_change_handler(
        self,
        handler: Callable[[str, int], bool] | None,
    ) -> None:
        """Route UI resolution changes through an application-wide transaction."""

        if handler is not None and not callable(handler):
            raise TypeError("The texture resolution change handler must be callable.")
        self._texture_resolution_change_handler = handler

    def select_assignment_texture_resolution(
        self,
        assignment_id: str,
        resolution: int,
        target_surface_ids: Sequence[str] = (),
    ) -> bool:
        """Select a family variant and optionally apply it to selected surfaces."""

        if self.is_generating:
            return False
        normalized_id = str(assignment_id).strip()
        source_assignment = next(
            (
                assignment
                for assignment in self._data.assignments
                if assignment.assignment_id == normalized_id
            ),
            None,
        )
        if source_assignment is None:
            return False
        try:
            target_resolution = int(resolution)
        except (TypeError, ValueError, OverflowError):
            return False
        variant = source_assignment.texture_variant_for_resolution(
            target_resolution
        )
        if variant is None:
            return False
        texture_path = self.get_assignment_asset_path(
            source_assignment.assignment_id,
            target_resolution,
        )
        if texture_path is None:
            return False
        try:
            texture_png = texture_path.read_bytes()
            texture_rgba = _decode_png_rgba(
                texture_png,
                "Surface texture variant",
            )
        except (OSError, ValueError):
            return False
        if texture_rgba.shape[:2] != (
            target_resolution,
            target_resolution,
        ):
            return False
        selected_assignment = replace(
            source_assignment,
            asset_path=variant.asset_path,
            selected_texture_resolution=target_resolution,
            texture_width=target_resolution,
            texture_height=target_resolution,
        )
        return self._apply_assignment_texture_to_surfaces(
            source_assignment,
            selected_assignment,
            texture_png,
            target_surface_ids,
            texture_size=(target_resolution, target_resolution),
        )

    def apply_assignment_texture(
        self,
        assignment_id: str,
        target_surface_ids: Sequence[str] = (),
    ) -> bool:
        """Apply one family's active asset without changing its resolution."""

        if self.is_generating:
            return False
        source_assignment = self._assignment_by_id(assignment_id)
        if source_assignment is None:
            return False
        selected_resolution = source_assignment.selected_texture_resolution
        if selected_resolution is not None:
            return self.select_assignment_texture_resolution(
                source_assignment.assignment_id,
                selected_resolution,
                target_surface_ids,
            )
        if source_assignment.texture_variants:
            return False
        texture_path = self.get_assignment_asset_path(
            source_assignment.assignment_id
        )
        if texture_path is None:
            return False
        try:
            texture_png = texture_path.read_bytes()
            texture_rgba = _decode_png_rgba(
                texture_png,
                "Surface texture",
            )
        except (OSError, ValueError):
            return False
        height, width = texture_rgba.shape[:2]
        return self._apply_assignment_texture_to_surfaces(
            source_assignment,
            source_assignment,
            texture_png,
            target_surface_ids,
            texture_size=(width, height),
        )

    def delete_assignment_texture(self, assignment_id: str) -> bool:
        """Delete one complete texture family from every assigned surface."""

        if self.is_generating:
            return False
        assignment = self._assignment_by_id(assignment_id)
        if assignment is None:
            return False

        previous_assignments = list(self._data.assignments)
        previous_strokes = {
            surface_id: list(strokes)
            for surface_id, strokes in self._data.texture_mask_strokes.items()
        }
        previous_undo_stack = list(
            self._data.localized_inpaint_undo_stack
        )
        affected_surface_ids = set(assignment.surface_ids)
        next_assignments = [
            candidate
            for candidate in previous_assignments
            if candidate.assignment_id != assignment.assignment_id
        ]
        retained_surface_ids = {
            surface_id
            for candidate in next_assignments
            for surface_id in candidate.surface_ids
        }
        untextured_surface_ids = affected_surface_ids.difference(
            retained_surface_ids
        )
        next_strokes = {
            surface_id: strokes
            for surface_id, strokes in previous_strokes.items()
            if surface_id not in untextured_surface_ids
        }

        self._data.assignments = next_assignments
        self._data.texture_mask_strokes = next_strokes
        discarded_undo_assignments = (
            self._clear_localized_inpaint_undo_history()
        )
        try:
            self._restore_assignment_textures()
            self.surface_view.set_texture_mask_strokes(next_strokes)
        except (OSError, TypeError, ValueError):
            self._data.assignments = previous_assignments
            self._data.texture_mask_strokes = previous_strokes
            self._data.localized_inpaint_undo_stack = previous_undo_stack
            try:
                self._restore_assignment_textures()
                self.surface_view.set_texture_mask_strokes(previous_strokes)
            except (OSError, TypeError, ValueError):
                pass
            self.status_label.setText(
                "The selected surface texture could not be deleted."
            )
            self._sync_controls()
            return False

        if (
            self.inpaint_3d_button.isChecked()
            and not self.surface_view.can_inpaint_selection()
        ):
            self.inpaint_3d_button.setChecked(False)

        removed_assignments = [
            assignment,
            *discarded_undo_assignments,
        ]
        self._texture_atlas_entry_cache.clear()
        self._refresh_texture_atlases()
        cleanup_failure_count = self._delete_orphaned_assignment_assets(
            removed_assignments
        )
        removed_assignment_ids = self._unretained_assignment_ids(
            removed_assignments
        )
        if removed_assignment_ids:
            self.assignments_removed.emit(removed_assignment_ids)
        self._emit_data_changed()
        self.surface_content_changed.emit()

        status = (
            f"Deleted the selected {assignment.surface_type} texture from "
            f"{len(assignment.surface_ids)} surface(s)."
        )
        if cleanup_failure_count:
            status += (
                f" {cleanup_failure_count} unused texture file(s) could not "
                "be deleted."
            )
        self.status_label.setText(status)
        self._sync_controls()
        return True

    def _apply_assignment_texture_to_surfaces(
        self,
        source_assignment: SurfaceTextureAssignment,
        applied_assignment: SurfaceTextureAssignment,
        texture_png: bytes,
        target_surface_ids: Sequence[str],
        *,
        texture_size: tuple[int, int],
    ) -> bool:
        """Commit one validated active family asset to homogeneous surfaces."""

        normalized_targets = tuple(
            dict.fromkeys(str(surface_id) for surface_id in target_surface_ids)
        )
        if normalized_targets:
            target_surfaces = tuple(
                self.surface_view.get_surface(surface_id)
                for surface_id in normalized_targets
            )
            if any(surface is None for surface in target_surfaces):
                return False
            if any(
                surface.surface_type != source_assignment.surface_type
                for surface in target_surfaces
                if surface is not None
            ):
                return False
        else:
            normalized_targets = source_assignment.surface_ids
        if (
            applied_assignment == source_assignment
            and set(normalized_targets).issubset(source_assignment.surface_ids)
        ):
            return True

        previous_assignments = list(self._data.assignments)
        previous_strokes = {
            surface_id: list(strokes)
            for surface_id, strokes in self._data.texture_mask_strokes.items()
        }
        source_surface_ids = tuple(
            dict.fromkeys(
                (*source_assignment.surface_ids, *normalized_targets)
            )
        )
        retained_assignments: list[SurfaceTextureAssignment] = []
        removed_assignments: list[SurfaceTextureAssignment] = []
        target_set = set(normalized_targets)
        for assignment in self._data.assignments:
            if assignment.assignment_id == source_assignment.assignment_id:
                continue
            remaining_ids = tuple(
                surface_id
                for surface_id in assignment.surface_ids
                if surface_id not in target_set
            )
            if remaining_ids == assignment.surface_ids:
                retained_assignments.append(assignment)
            elif not remaining_ids:
                removed_assignments.append(assignment)
            else:
                retained_assignments.append(
                    self._assignment_with_surfaces(
                        assignment,
                        remaining_ids,
                    )
                )
        selected_assignment = self._assignment_with_surfaces(
            applied_assignment,
            source_surface_ids,
        )
        next_assignments = [*retained_assignments, selected_assignment]
        changed_family_ids = target_set.difference(
            source_assignment.surface_ids
        )
        next_strokes = {
            surface_id: strokes
            for surface_id, strokes in previous_strokes.items()
            if surface_id not in changed_family_ids
        }

        try:
            self.surface_view.set_surface_texture(
                selected_assignment.surface_ids,
                texture_png,
            )
            self._data.assignments = next_assignments
            self._data.texture_mask_strokes = next_strokes
            self.surface_view.set_texture_mask_strokes(next_strokes)
            # The caller supplied bytes read before this commit.  Do not bind
            # them to a file revision sampled afterward: the backing PNG may
            # have been replaced between the read and this installation.
            self._restored_assignment_texture_signature = None
        except (OSError, TypeError, ValueError):
            self._data.assignments = previous_assignments
            self._data.texture_mask_strokes = previous_strokes
            self._restore_assignment_textures()
            self.surface_view.set_texture_mask_strokes(previous_strokes)
            return False

        discarded_assignments = self._clear_localized_inpaint_undo_history()
        self._texture_atlas_entry_cache.clear()
        self._refresh_texture_atlases()
        cleanup_failure_count = self._delete_orphaned_assignment_assets(
            [*removed_assignments, *discarded_assignments]
        )
        removed_assignment_ids = self._unretained_assignment_ids(
            [*removed_assignments, *discarded_assignments]
        )
        if removed_assignment_ids:
            self.assignments_removed.emit(removed_assignment_ids)
        texture_width, texture_height = texture_size
        self.status_label.setText(
            f"Applied the {texture_width} x {texture_height} texture "
            f"to {len(selected_assignment.surface_ids)} "
            f"{selected_assignment.surface_type} surface(s)."
        )
        if cleanup_failure_count:
            self.status_label.setText(
                self.status_label.text()
                + f" {cleanup_failure_count} unused texture file(s) could not "
                "be deleted."
            )
        self._emit_data_changed()
        self.surface_content_changed.emit()
        self._sync_controls()
        return True

    def set_data(self, data: SurfaceTextureData | None) -> None:
        if self.is_generating:
            raise RuntimeError(
                "Cannot replace surface texture data while generating."
            )
        self._close_video_source()
        self._displayed_frame_index = None
        self._texture_atlas_entry_cache.clear()
        self._texture_variant_entry_targets.clear()
        self._other_texture_entry_targets.clear()
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
        saved_camera_pose = self._data.camera_pose
        self.surface_view.set_levels(
            self._levels,
            self._initial_camera,
        )
        self._data.camera_pose = saved_camera_pose
        self._restore_viewer_state()
        self._restore_assignment_textures()
        self._refresh_texture_atlases()
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
        self._set_level_context(
            levels,
            initial_camera,
            preview_model=None,
            replace_preview_model=False,
        )

    def set_preview_context(
        self,
        levels: Sequence[LevelData],
        initial_camera: InitialFirstPersonCamera | None,
        model: GeneratedModel | None,
    ) -> None:
        """Synchronize semantic levels and the shared model with one GL rebuild."""

        self._set_level_context(
            levels,
            initial_camera,
            preview_model=model,
            replace_preview_model=True,
        )

    def _set_level_context(
        self,
        levels: Sequence[LevelData],
        initial_camera: InitialFirstPersonCamera | None,
        *,
        preview_model: GeneratedModel | None,
        replace_preview_model: bool,
    ) -> None:
        normalized_levels = list(levels)
        next_signature = _build_level_sync_signature(
            normalized_levels,
            initial_camera,
        )
        if next_signature == self._level_sync_signature:
            # Retain the latest model objects without rebuilding equivalent
            # semantic geometry, textures, atlas controls, or OpenGL items.
            self._levels = normalized_levels
            self._initial_camera = initial_camera
            self.refresh_file_backed_previews()
            if replace_preview_model:
                self.set_preview_model(preview_model)
            return

        if self._levels:
            self._store_viewer_state()
        self._levels = normalized_levels
        self._initial_camera = initial_camera
        if replace_preview_model:
            self.surface_view.set_scene_model(
                preview_model,
                repopulate=False,
            )
            self.surface_view.clear_surface_textures()
        self.surface_view.set_levels(
            self._levels,
            initial_camera,
        )
        self._restore_viewer_state()
        self._restore_assignment_textures()
        self._refresh_texture_atlases()
        self._sync_selection_status()
        self._sync_controls()
        self._level_sync_signature = next_signature

    def set_preview_model(self, model: GeneratedModel | None) -> None:
        """Use the Canvas model while retaining semantic selection and inpainting."""

        self.surface_view.set_scene_model(model)

    def set_runtime_settings(self, settings: GenerationServiceSettings) -> None:
        if not isinstance(settings, GenerationServiceSettings):
            raise TypeError("Generation settings have an invalid type.")
        self._settings = settings
        self._sync_provider_combo(settings.surface_texture_provider)
        self._sync_controls()

    def get_runtime_settings(self) -> GenerationServiceSettings:
        return self._settings

    def set_external_3d_viewer_active(self, is_active: bool) -> None:
        """Show the local atlas inspector while fixed surfaces are external."""

        target_page = (
            self.texture_view_page if bool(is_active) else self.surface_3d_page
        )
        if self.right_view_stack.currentWidget() is target_page:
            return
        self.right_view_stack.setCurrentWidget(target_page)

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

    def undo_localized_texture_inpaint(self) -> bool:
        """Restore the complete state before the latest successful 3D inpaint."""

        if self.is_generating:
            return False
        history = self._data.localized_inpaint_undo_stack
        if not history:
            self.status_label.setText("There is no localized texture inpaint to undo.")
            self._sync_controls()
            return False
        snapshot = history[-1]
        try:
            required_assignment_ids = self._validate_undo_snapshot_assets(snapshot)
        except (OSError, ValueError) as error:
            self.status_label.setText(
                f"The localized texture inpaint cannot be undone: {error}"
            )
            return False

        current_assignments = list(self._data.assignments)
        current_strokes = {
            surface_id: list(strokes)
            for surface_id, strokes in self._data.texture_mask_strokes.items()
        }
        restored_strokes = {
            surface_id: list(strokes)
            for surface_id, strokes in current_strokes.items()
        }
        for surface_id in snapshot.affected_surface_ids:
            restored_strokes.pop(surface_id, None)
            previous = snapshot.previous_texture_mask_strokes.get(surface_id)
            if previous:
                restored_strokes[surface_id] = list(previous)

        try:
            self._data.assignments = list(snapshot.previous_assignments)
            self._data.texture_mask_strokes = restored_strokes
            self._restore_assignment_textures(
                required_assignment_ids=required_assignment_ids
            )
            self.surface_view.set_texture_mask_strokes(restored_strokes)
        except (OSError, TypeError, ValueError) as error:
            self._data.assignments = current_assignments
            self._data.texture_mask_strokes = current_strokes
            self._restore_assignment_textures()
            self.surface_view.set_texture_mask_strokes(current_strokes)
            self.status_label.setText(
                f"The localized texture inpaint could not be undone: {error}"
            )
            return False

        history.pop()
        self._texture_atlas_entry_cache.clear()
        self._refresh_texture_atlases()
        cleanup_failure_count = self._delete_orphaned_assignment_assets(
            current_assignments
        )
        removed_assignment_ids = self._unretained_assignment_ids(
            current_assignments
        )
        self.status_label.setText("Restored the texture before localized inpainting.")
        if cleanup_failure_count:
            self.status_label.setText(
                self.status_label.text()
                + f" {cleanup_failure_count} replaced texture file(s) could not "
                "be deleted."
            )
        self._emit_data_changed()
        if removed_assignment_ids:
            self.assignments_removed.emit(removed_assignment_ids)
        self.localized_inpaint_undone.emit(snapshot)
        self.surface_content_changed.emit()
        self._sync_controls()
        return True

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
        self.surface_3d_page = _build_labeled_view(
            "Fixed surfaces",
            self.surface_view,
        )
        self.texture_view = TextureAtlasView(
            empty_preview_text="No texture resolutions available",
            unselected_preview_text="Select a texture resolution",
        )
        self.texture_view.setObjectName("surface_texture_atlas_view")
        self.texture_view.atlas_selected.connect(
            self._handle_texture_variant_selected
        )
        self.texture_view.atlas_activated.connect(
            self._handle_texture_variant_activated
        )
        self.texture_view.atlas_list.installEventFilter(self)
        self.texture_view.atlas_list.itemPressed.connect(
            self._handle_current_texture_item_pressed
        )
        self.other_texture_list = QListWidget()
        self.other_texture_list.setObjectName(
            "other_surface_texture_list"
        )
        self.other_texture_list.setViewMode(QListView.ViewMode.IconMode)
        self.other_texture_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.other_texture_list.setMovement(QListView.Movement.Static)
        self.other_texture_list.setWrapping(True)
        self.other_texture_list.setWordWrap(True)
        self.other_texture_list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.other_texture_list.setIconSize(OTHER_TEXTURE_THUMBNAIL_SIZE)
        self.other_texture_list.setGridSize(OTHER_TEXTURE_GRID_SIZE)
        self.other_texture_list.setToolTip(
            "Double-click a compatible texture to apply it to the selected "
            "surfaces."
        )
        self.other_texture_list.itemDoubleClicked.connect(
            self._handle_other_texture_activated
        )
        self.other_texture_list.currentItemChanged.connect(
            self._handle_other_texture_selection_changed
        )
        self.other_texture_list.installEventFilter(self)
        self.texture_views_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.texture_views_splitter.setObjectName(
            "surface_texture_selection_splitter"
        )
        self.texture_views_splitter.setChildrenCollapsible(False)
        self.texture_views_splitter.addWidget(
            _build_labeled_view(
                "Current texture resolutions",
                self.texture_view,
            )
        )
        self.texture_views_splitter.addWidget(
            _build_labeled_view(
                "Other surface textures",
                self.other_texture_list,
            )
        )
        self.texture_views_splitter.setStretchFactor(0, 1)
        self.texture_views_splitter.setStretchFactor(1, 1)
        self.texture_views_splitter.setSizes([1_000, 1_000])
        self.texture_view_page = _build_labeled_view(
            "Texture view",
            self.texture_views_splitter,
        )
        self.right_view_stack = QStackedWidget()
        self.right_view_stack.setObjectName("surface_texture_right_view_stack")
        self.right_view_stack.addWidget(self.surface_3d_page)
        self.right_view_stack.addWidget(self.texture_view_page)
        self.right_view_stack.setCurrentWidget(self.surface_3d_page)
        self.views_splitter.addWidget(self.right_view_stack)
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
        self.delete_texture_button = QPushButton("Delete texture")
        self.delete_texture_button.setObjectName(
            "delete_surface_texture_button"
        )
        self.delete_texture_button.setToolTip(
            "Delete the selected texture family and all of its resolution "
            "files. Select an item in Other surface textures to delete that "
            "family instead of the current surface texture."
        )
        self.delete_texture_button.clicked.connect(
            self._handle_delete_texture_clicked
        )
        first_row.addWidget(self.delete_texture_button)
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
        self.undo_inpaint_button = QPushButton("Undo inpaint")
        self.undo_inpaint_button.setObjectName(
            "undo_surface_texture_inpaint_button"
        )
        self.undo_inpaint_button.setMinimumHeight(38)
        self.undo_inpaint_button.setToolTip(
            "Restore the texture and painted region before the latest successful "
            "localized 3D inpaint."
        )
        self.undo_inpaint_button.clicked.connect(
            self.undo_localized_texture_inpaint
        )
        second_row.addWidget(self.undo_inpaint_button)
        root_layout.addLayout(second_row)

        self.status_label = QLabel(
            "Left-click the 3D view to enter first-person mode. Move with Z/Q/S/D, "
            "look with the mouse, select at the crosshair, and right-click to exit."
        )
        self.status_label.setWordWrap(True)
        root_layout.addWidget(self.status_label)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Route Delete from either texture selector to the same safe action."""

        texture_view = getattr(self, "texture_view", None)
        current_list = getattr(texture_view, "atlas_list", None)
        other_list = getattr(self, "other_texture_list", None)
        if (
            (watched is current_list or watched is other_list)
            and isinstance(event, QKeyEvent)
            and event.type() == QEvent.Type.KeyPress
            and event.key() == Qt.Key.Key_Delete
        ):
            self._handle_delete_texture_clicked()
            return True
        return super().eventFilter(watched, event)

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
                base_rgba, edit_masks = (
                    self._build_canonical_selected_texture_edit_data(
                        masked_surface_ids
                    )
                )
            except (OSError, ValueError) as error:
                self.status_label.setText(str(error))
                return None
            if not edit_masks:
                self.status_label.setText("Paint an edit region on the 3D texture.")
                return None
            editable_mask = np.maximum.reduce(tuple(edit_masks.values()))
            existing_texture_png = _encode_rgba_png(base_rgba)
            edit_mask_png = _encode_png(editable_mask)
            surface_edit_mask_pngs = tuple(
                (surface_id, _encode_png(mask))
                for surface_id, mask in edit_masks.items()
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

    def _build_canonical_selected_texture_edit_data(
        self,
        masked_surface_ids: Sequence[str],
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Load one family's canonical texture and rasterize normalized masks."""

        base_textures: list[np.ndarray] = []
        strokes_by_surface_id = self.surface_view.get_texture_mask_strokes()
        edit_masks: dict[str, np.ndarray] = {}
        assignments = tuple(
            next(
                (
                    candidate
                    for candidate in reversed(self._data.assignments)
                    if surface_id in candidate.surface_ids
                ),
                None,
            )
            for surface_id in masked_surface_ids
        )
        if any(assignment is None for assignment in assignments):
            edit_data = self.surface_view.get_selected_texture_edit_data()
            if edit_data is None:
                raise ValueError("Paint an edit region on the 3D texture.")
            return (
                edit_data[0],
                self.surface_view.get_selected_texture_edit_masks(),
            )
        for assignment in assignments:
            assert assignment is not None
            target_resolution = (
                2048 if assignment.texture_variants else None
            )
            texture_path = self.get_assignment_asset_path(
                assignment.assignment_id,
                target_resolution,
            )
            if texture_path is None:
                raise ValueError(
                    "The canonical 2048 surface texture is unavailable."
                )
            texture = _decode_png_rgba(
                texture_path.read_bytes(),
                "Base surface texture",
            )
            base_textures.append(texture)
        base_texture = base_textures[0]
        if any(
            texture.shape != base_texture.shape
            or not np.array_equal(texture, base_texture)
            for texture in base_textures[1:]
        ):
            raise ValueError(
                "Selected surfaces must share the same texture for partial "
                "inpainting."
            )
        for surface_id in masked_surface_ids:
            strokes = strokes_by_surface_id.get(surface_id, ())
            if not strokes:
                continue
            edit_masks[surface_id] = rasterize_texture_mask_strokes(
                (base_texture.shape[1], base_texture.shape[0]),
                strokes,
            )
        return base_texture.copy(), edit_masks

    @Slot(object)
    def _handle_surface_selection_changed(self, raw_ids: object) -> None:
        if not isinstance(raw_ids, tuple | list):
            return
        if hasattr(self, "other_texture_list"):
            signals_were_blocked = self.other_texture_list.blockSignals(True)
            try:
                self.other_texture_list.setCurrentRow(-1)
                self.other_texture_list.clearSelection()
            finally:
                self.other_texture_list.blockSignals(signals_were_blocked)
        self._data.selected_surface_ids = tuple(str(value) for value in raw_ids)
        self._data.selected_surface_type = self.surface_view.get_selected_surface_type()
        if (
            self.inpaint_3d_button.isChecked()
            and not self.surface_view.can_inpaint_selection()
        ):
            self.inpaint_3d_button.setChecked(False)
        self._sync_selection_status()
        self._refresh_texture_atlases()
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
        is_localized_inpaint = _is_localized_surface_texture_inpaint(request)
        previous_assignments = tuple(self._data.assignments)
        previous_texture_mask_strokes = {
            surface_id: tuple(self._data.texture_mask_strokes.get(surface_id, ()))
            for surface_id in request.surface_ids
            if self._data.texture_mask_strokes.get(surface_id)
        }
        try:
            outputs = _build_surface_texture_outputs(request, result.texture_png)
        except ValueError as error:
            self._handle_generation_failed(str(error))
            return
        saved_outputs: list[_SavedSurfaceTextureOutput] = []
        try:
            for surface_ids, texture_png in outputs:
                assignment_id = uuid.uuid4().hex
                texture_variants = build_surface_texture_variants(texture_png)
                saved_outputs.append(
                    self._persist_texture_variants(
                        surface_ids,
                        assignment_id,
                        texture_variants,
                    )
                )
        except (OSError, ValueError) as error:
            self._discard_saved_outputs(saved_outputs)
            self._handle_generation_failed(
                f"The generated texture could not be saved: {error}"
            )
            return
        assignments: list[SurfaceTextureAssignment] = []
        try:
            for saved_output in saved_outputs:
                selected_resolution = (
                    self._replacement_texture_resolution(
                        saved_output.surface_ids,
                        previous_assignments,
                    )
                    if is_localized_inpaint
                    else DEFAULT_SURFACE_TEXTURE_RESOLUTION
                )
                active_variant = next(
                    variant
                    for variant in saved_output.variants
                    if variant.resolution == selected_resolution
                )
                area_m2 = self.surface_view.get_combined_surface_area(
                    saved_output.surface_ids
                )
                assignments.append(
                    SurfaceTextureAssignment(
                        assignment_id=saved_output.assignment_id,
                        surface_type=request.surface_type,
                        surface_ids=saved_output.surface_ids,
                        provider=result.provider,
                        provider_task_id=result.task_id,
                        asset_path=active_variant.asset_path,
                        combined_area_m2=area_m2,
                        area_description=(
                            f"{len(saved_output.surface_ids)} "
                            f"{request.surface_type} "
                            f"surface(s), {area_m2:.2f} m²"
                        ),
                        reference_frame_indices=request.reference_frame_indices,
                        texture_width=selected_resolution,
                        texture_height=selected_resolution,
                        texture_variants=saved_output.variants,
                        selected_texture_resolution=selected_resolution,
                    )
                )
        except (OSError, ValueError) as error:
            self._discard_saved_outputs(saved_outputs)
            self._handle_generation_failed(
                f"The generated texture could not be applied: {error}"
            )
            return

        try:
            retained_assignments, removed_assignments = (
                self._replace_assignments_for_surfaces(assignments)
            )
            for saved_output, assignment in zip(
                saved_outputs,
                assignments,
                strict=True,
            ):
                self.surface_view.set_surface_texture(
                    saved_output.surface_ids,
                    saved_output.png_for_resolution(
                        assignment.selected_texture_resolution
                        or DEFAULT_SURFACE_TEXTURE_RESOLUTION
                    ),
                )
        except (OSError, TypeError, ValueError) as error:
            self._restore_assignment_textures()
            self._discard_saved_outputs(saved_outputs)
            self._handle_generation_failed(
                f"The generated texture could not be applied: {error}"
            )
            return

        self._data.assignments = [*retained_assignments, *assignments]
        # These pixels came from the generated in-memory payload.  Leave the
        # file-backed cache unvalidated so activation confirms the exact bytes
        # that were persisted, including a concurrent same-path replacement.
        self._restored_assignment_texture_signature = None
        discarded_snapshots: list[SurfaceTextureInpaintUndoSnapshot] = []
        if is_localized_inpaint:
            self._data.localized_inpaint_undo_stack.append(
                SurfaceTextureInpaintUndoSnapshot(
                    previous_assignments=previous_assignments,
                    replacement_assignment_ids=tuple(
                        assignment.assignment_id for assignment in assignments
                    ),
                    affected_surface_ids=request.surface_ids,
                    previous_texture_mask_strokes=(
                        previous_texture_mask_strokes
                    ),
                )
            )
            excess_count = max(
                0,
                len(self._data.localized_inpaint_undo_stack)
                - MAX_LOCALIZED_INPAINT_UNDO_HISTORY,
            )
            if excess_count:
                discarded_snapshots = (
                    self._data.localized_inpaint_undo_stack[:excess_count]
                )
                del self._data.localized_inpaint_undo_stack[:excess_count]
        else:
            discarded_snapshots = list(
                self._data.localized_inpaint_undo_stack
            )
            self._data.localized_inpaint_undo_stack.clear()
        self._texture_atlas_entry_cache.clear()
        self._refresh_texture_atlases()
        self.surface_view.clear_texture_mask(request.surface_ids)
        discarded_assignments = [
            assignment
            for snapshot in discarded_snapshots
            for assignment in snapshot.previous_assignments
        ]
        cleanup_failure_count = self._delete_orphaned_assignment_assets(
            [*removed_assignments, *discarded_assignments]
        )
        status = (
            f"Applied generated texture to {len(request.surface_ids)} "
            f"{request.surface_type} surface(s)."
        )
        if cleanup_failure_count:
            status += (
                f" {cleanup_failure_count} replaced texture file(s) could not "
                "be deleted."
            )
        self.status_label.setText(status)
        self._emit_data_changed()
        removable_assignments = [
            *([] if is_localized_inpaint else removed_assignments),
            *discarded_assignments,
        ]
        removed_assignment_ids = self._unretained_assignment_ids(
            removable_assignments
        )
        if removed_assignment_ids:
            self.assignments_removed.emit(removed_assignment_ids)
        for assignment in assignments:
            self.generation_completed.emit(assignment)
        self.surface_content_changed.emit()
        self._sync_controls()

    def _replace_assignments_for_surfaces(
        self,
        replacements: Sequence[SurfaceTextureAssignment],
    ) -> tuple[
        list[SurfaceTextureAssignment],
        list[SurfaceTextureAssignment],
    ]:
        replaced_surface_ids = {
            surface_id
            for assignment in replacements
            for surface_id in assignment.surface_ids
        }
        retained: list[SurfaceTextureAssignment] = []
        removed: list[SurfaceTextureAssignment] = []
        for assignment in self._data.assignments:
            remaining_surface_ids = tuple(
                surface_id
                for surface_id in assignment.surface_ids
                if surface_id not in replaced_surface_ids
            )
            if remaining_surface_ids == assignment.surface_ids:
                retained.append(assignment)
                continue
            if not remaining_surface_ids:
                removed.append(assignment)
                continue
            retained.append(
                self._assignment_with_surfaces(
                    assignment,
                    remaining_surface_ids,
                )
            )
        return retained, removed

    def _assignment_with_surfaces(
        self,
        assignment: SurfaceTextureAssignment,
        surface_ids: Sequence[str],
    ) -> SurfaceTextureAssignment:
        """Return assignment metadata recomputed for one surface group."""

        normalized_ids = tuple(dict.fromkeys(str(value) for value in surface_ids))
        area_m2 = self.surface_view.get_combined_surface_area(normalized_ids)
        return replace(
            assignment,
            surface_ids=normalized_ids,
            combined_area_m2=area_m2,
            area_description=(
                f"{len(normalized_ids)} {assignment.surface_type} "
                f"surface(s), {area_m2:.2f} m²"
            ),
        )

    @staticmethod
    def _replacement_texture_resolution(
        surface_ids: Sequence[str],
        previous_assignments: Sequence[SurfaceTextureAssignment],
    ) -> int:
        """Preserve one unambiguous active resolution through inpainting."""

        resolutions: set[int] = set()
        for surface_id in surface_ids:
            assignment = next(
                (
                    candidate
                    for candidate in reversed(previous_assignments)
                    if surface_id in candidate.surface_ids
                ),
                None,
            )
            if (
                assignment is None
                or assignment.selected_texture_resolution is None
            ):
                return DEFAULT_SURFACE_TEXTURE_RESOLUTION
            resolutions.add(assignment.selected_texture_resolution)
        if len(resolutions) != 1:
            return DEFAULT_SURFACE_TEXTURE_RESOLUTION
        resolution = next(iter(resolutions))
        return (
            resolution
            if resolution in SURFACE_TEXTURE_RESOLUTIONS
            else DEFAULT_SURFACE_TEXTURE_RESOLUTION
        )

    def _discard_saved_outputs(
        self,
        saved_outputs: Sequence[_SavedSurfaceTextureOutput],
    ) -> None:
        for saved_output in saved_outputs:
            for variant in saved_output.variants:
                try:
                    self._resolve_asset_path(variant.asset_path).unlink(
                        missing_ok=True
                    )
                except (OSError, ValueError):
                    continue

    def _delete_orphaned_assignment_assets(
        self,
        removed_assignments: Sequence[SurfaceTextureAssignment],
    ) -> int:
        active_asset_paths: set[Path] = set()
        retained_assignments = [
            *self._data.assignments,
            *(
                assignment
                for snapshot in self._data.localized_inpaint_undo_stack
                for assignment in snapshot.previous_assignments
            ),
        ]
        for assignment in retained_assignments:
            for raw_path in self._assignment_asset_relative_paths(assignment):
                try:
                    active_asset_paths.add(self._resolve_asset_path(raw_path))
                except (OSError, ValueError):
                    continue

        orphaned_asset_paths: set[Path] = set()
        for assignment in removed_assignments:
            for raw_path in self._assignment_asset_relative_paths(assignment):
                try:
                    resolved_path = self._resolve_asset_path(raw_path)
                except (OSError, ValueError):
                    continue
                if resolved_path not in active_asset_paths:
                    orphaned_asset_paths.add(resolved_path)
        failure_count = 0
        for asset_path in orphaned_asset_paths:
            try:
                asset_path.unlink(missing_ok=True)
            except OSError:
                failure_count += 1
        return failure_count

    @staticmethod
    def _assignment_asset_relative_paths(
        assignment: SurfaceTextureAssignment,
    ) -> tuple[str, ...]:
        """Return every file retained by one assignment without duplicates."""

        return tuple(
            dict.fromkeys(
                (
                    assignment.asset_path,
                    *(
                        variant.asset_path
                        for variant in assignment.texture_variants
                    ),
                )
            )
        )

    def _unretained_assignment_ids(
        self,
        assignments: Sequence[SurfaceTextureAssignment],
    ) -> tuple[str, ...]:
        retained_ids = {
            assignment.assignment_id for assignment in self._data.assignments
        }
        retained_ids.update(
            assignment.assignment_id
            for snapshot in self._data.localized_inpaint_undo_stack
            for assignment in snapshot.previous_assignments
        )
        return tuple(
            dict.fromkeys(
                assignment.assignment_id
                for assignment in assignments
                if assignment.assignment_id not in retained_ids
            )
        )

    def _clear_localized_inpaint_undo_history(
        self,
    ) -> list[SurfaceTextureAssignment]:
        discarded = [
            assignment
            for snapshot in self._data.localized_inpaint_undo_stack
            for assignment in snapshot.previous_assignments
        ]
        self._data.localized_inpaint_undo_stack.clear()
        return discarded

    def _validate_undo_snapshot_assets(
        self,
        snapshot: SurfaceTextureInpaintUndoSnapshot,
    ) -> set[str]:
        validated_paths: set[Path] = set()
        required_assignment_ids: set[str] = set()
        for surface_id in snapshot.affected_surface_ids:
            assignment = next(
                (
                    candidate
                    for candidate in reversed(snapshot.previous_assignments)
                    if surface_id in candidate.surface_ids
                ),
                None,
            )
            if assignment is None:
                raise ValueError(
                    f"the prior texture for {surface_id!r} is unavailable."
                )
            required_assignment_ids.add(assignment.assignment_id)
            for raw_path in self._assignment_asset_relative_paths(assignment):
                texture_path = self._resolve_asset_path(raw_path)
                if texture_path in validated_paths:
                    continue
                if not texture_path.is_file():
                    raise ValueError(
                        f"the prior texture asset {raw_path!r} is missing."
                    )
                _decode_png_rgba(
                    texture_path.read_bytes(),
                    "Prior surface texture",
                )
                validated_paths.add(texture_path)
        return required_assignment_ids

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
            self.surface_view.set_camera_pose(
                self._data.camera_pose,
                emit_signal=False,
            )
        self.surface_view.set_selected_surface_ids(
            self._data.selected_surface_ids,
            emit_signal=False,
        )
        self._data.selected_surface_ids = tuple(
            self.surface_view.get_selected_surface_ids()
        )
        self._data.selected_surface_type = (
            self.surface_view.get_selected_surface_type()
        )
        self.surface_view.set_texture_mask_strokes(
            self._data.texture_mask_strokes
        )

    def _restore_assignment_textures(
        self,
        *,
        required_assignment_ids: set[str] | None = None,
    ) -> None:
        required_ids = required_assignment_ids or set()
        signature_before = self.get_preview_dependency_signature()
        restore_succeeded = True
        self.surface_view.clear_surface_textures()
        for assignment in self._data.assignments:
            try:
                texture_path = self._resolve_asset_path(assignment.asset_path)
            except (OSError, TypeError, ValueError):
                if assignment.assignment_id in required_ids:
                    raise
                continue
            if not texture_path.is_file():
                if assignment.assignment_id in required_ids:
                    raise OSError("The required surface texture is missing.")
                continue
            try:
                self.surface_view.set_surface_texture(
                    assignment.surface_ids,
                    texture_path,
                )
            except (OSError, TypeError, ValueError):
                if assignment.assignment_id in required_ids:
                    raise
                restore_succeeded = False
                continue
        signature_after = self.get_preview_dependency_signature()
        self._restored_assignment_texture_signature = (
            signature_after
            if restore_succeeded and signature_before == signature_after
            else None
        )

    def _refresh_texture_atlases(self) -> None:
        signature_before = self._build_texture_catalog_dependency_signature()
        assignment_id = self._latest_selected_surface_assignment_id()
        assignment = next(
            (
                candidate
                for candidate in self._data.assignments
                if candidate.assignment_id == assignment_id
            ),
            None,
        )
        entries: list[TextureAtlasEntry] = []
        other_entries: list[TextureAtlasEntry] = []
        next_cache: dict[
            str,
            tuple[tuple[object, ...], TextureAtlasEntry],
        ] = {}
        next_targets: dict[str, tuple[str, int]] = {}
        next_other_targets: dict[str, tuple[str, int | None]] = {}
        next_other_tooltips: dict[str, str] = {}
        catalog_succeeded = True
        for variant in (() if assignment is None else assignment.texture_variants):
            entry_id = (
                f"{assignment.assignment_id}:resolution:{variant.resolution}"
            )
            try:
                texture_path = self._resolve_asset_path(variant.asset_path)
            except (OSError, TypeError, ValueError):
                continue
            if not texture_path.is_file():
                continue
            asset_revision = _build_surface_asset_revision(
                self._asset_directory,
                variant.asset_path,
            )
            try:
                cached_entry = self._texture_atlas_entry_cache.get(entry_id)
                if cached_entry is not None and cached_entry[0] == asset_revision:
                    entry = cached_entry[1]
                else:
                    entry = TextureAtlasEntry(
                        atlas_id=entry_id,
                        display_name=f"{variant.resolution} x {variant.resolution}",
                        image=texture_path,
                        owner_id=assignment.assignment_id,
                    )
            except (OSError, TypeError, ValueError):
                catalog_succeeded = False
                continue
            entries.append(entry)
            next_cache[entry_id] = (asset_revision, entry)
            next_targets[entry_id] = (
                assignment.assignment_id,
                variant.resolution,
            )

        selected_surface_type = self.surface_view.get_selected_surface_type()
        for candidate in self._data.assignments:
            resolution = candidate.selected_texture_resolution
            if (
                selected_surface_type is None
                or candidate.assignment_id == assignment_id
                or candidate.surface_type != selected_surface_type
            ):
                continue
            if candidate.texture_variants:
                active_variant = (
                    None
                    if resolution is None
                    else candidate.texture_variant_for_resolution(resolution)
                )
                if active_variant is None:
                    continue
                display_name = (
                    f"{candidate.surface_type.title()} texture - "
                    f"{resolution} x {resolution}"
                )
            else:
                if resolution is not None:
                    continue
                display_name = (
                    f"{candidate.surface_type.title()} texture - fixed image"
                )
            entry_id = f"{candidate.assignment_id}:other-texture"
            try:
                texture_path = self._resolve_asset_path(candidate.asset_path)
            except (OSError, TypeError, ValueError):
                continue
            if not texture_path.is_file():
                continue
            asset_revision = _build_surface_asset_revision(
                self._asset_directory,
                candidate.asset_path,
            )
            try:
                cached_entry = self._texture_atlas_entry_cache.get(entry_id)
                if (
                    cached_entry is not None
                    and cached_entry[0] == asset_revision
                    and cached_entry[1].display_name == display_name
                ):
                    entry = cached_entry[1]
                else:
                    entry = TextureAtlasEntry(
                        atlas_id=entry_id,
                        display_name=display_name,
                        image=texture_path,
                        owner_id=candidate.assignment_id,
                    )
                image = entry.get_image()
                image_width = image.size().width()
                image_height = image.size().height()
                if (
                    resolution is not None
                    and (
                        image_width != resolution
                        or image_height != resolution
                    )
                ):
                    continue
            except (OSError, TypeError, ValueError):
                catalog_succeeded = False
                continue
            other_entries.append(entry)
            next_cache[entry_id] = (asset_revision, entry)
            next_other_targets[entry_id] = (
                candidate.assignment_id,
                resolution,
            )
            surface_count = len(candidate.surface_ids)
            surface_suffix = "surface" if surface_count == 1 else "surfaces"
            resolution_description = (
                f"Resolution: {resolution} x {resolution}"
                if resolution is not None
                else (
                    f"Fixed image: {image_width} x {image_height} "
                    "(no resolution variants)"
                )
            )
            next_other_tooltips[entry_id] = (
                f"{display_name}\n"
                f"{resolution_description}\n"
                f"Provider: {_get_provider_display_name(candidate.provider)}\n"
                f"Currently assigned to: {surface_count} {surface_suffix}\n"
                f"Texture ID: {candidate.assignment_id}\n"
                f"Double-click to apply it to the selected "
                f"{candidate.surface_type} surfaces."
            )
        self._texture_atlas_entry_cache = next_cache
        self._texture_variant_entry_targets = next_targets
        self._other_texture_entry_targets = next_other_targets

        selected_atlas_id: str | None = None
        if assignment is not None and entries:
            preferred_resolution = (
                assignment.selected_texture_resolution
                or DEFAULT_SURFACE_TEXTURE_RESOLUTION
            )
            selected_atlas_id = min(
                entries,
                key=lambda entry: (
                    abs(
                        next_targets[entry.atlas_id][1]
                        - preferred_resolution
                    ),
                    next_targets[entry.atlas_id][1],
                ),
            ).atlas_id
        self._is_refreshing_texture_atlases = True
        try:
            if tuple(entries) != self.texture_view.entries:
                self.texture_view.set_atlases(
                    entries,
                    selected_atlas_id=selected_atlas_id,
                )
            elif selected_atlas_id is not None:
                self.texture_view.select_atlas(selected_atlas_id)
            elif not entries:
                self.texture_view.select_atlas(None)
            self._rebuild_other_texture_list(
                other_entries,
                next_other_tooltips,
            )
        finally:
            self._is_refreshing_texture_atlases = False
        signature_after = self._build_texture_catalog_dependency_signature()
        self._texture_catalog_dependency_signature = (
            signature_after
            if catalog_succeeded and signature_before == signature_after
            else None
        )

    def _rebuild_other_texture_list(
        self,
        entries: Sequence[TextureAtlasEntry],
        tooltips_by_entry_id: dict[str, str],
    ) -> None:
        """Replace the compact reusable-texture thumbnail library."""

        content_signature = tuple(
            (
                id(entry),
                entry.atlas_id,
                entry.display_name,
                tooltips_by_entry_id.get(entry.atlas_id, entry.display_name),
            )
            for entry in entries
        )
        if content_signature == self._other_texture_list_signature:
            return

        current_item = self.other_texture_list.currentItem()
        current_entry_id = (
            None
            if current_item is None
            else str(current_item.data(Qt.ItemDataRole.UserRole) or "")
        )
        signals_were_blocked = self.other_texture_list.blockSignals(True)
        try:
            self.other_texture_list.clear()
            replacement_item: QListWidgetItem | None = None
            for entry in entries:
                item = QListWidgetItem(
                    QIcon(
                        QPixmap.fromImage(entry.get_image()).scaled(
                            OTHER_TEXTURE_THUMBNAIL_SIZE,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                    ),
                    entry.display_name,
                )
                item.setData(Qt.ItemDataRole.UserRole, entry.atlas_id)
                item.setToolTip(
                    tooltips_by_entry_id.get(
                        entry.atlas_id,
                        entry.display_name,
                    )
                )
                self.other_texture_list.addItem(item)
                if entry.atlas_id == current_entry_id:
                    replacement_item = item
            if replacement_item is not None:
                self.other_texture_list.setCurrentItem(replacement_item)
        finally:
            self.other_texture_list.blockSignals(signals_were_blocked)
        self._other_texture_list_signature = content_signature

    @Slot(object)
    def _handle_texture_variant_selected(self, raw_entry: object) -> None:
        """Make a single-clicked variant the assignment's global resolution."""

        if self._is_refreshing_texture_atlases:
            return
        if not isinstance(raw_entry, TextureAtlasEntry):
            return
        target = self._texture_variant_entry_targets.get(raw_entry.atlas_id)
        if target is None:
            return
        assignment_id, resolution = target
        assignment = self._assignment_by_id(assignment_id)
        if (
            assignment is not None
            and assignment.selected_texture_resolution == resolution
        ):
            return
        if self._request_global_texture_resolution_change(
            assignment_id,
            resolution,
        ):
            return
        self.status_label.setText(
            "The selected surface texture resolution could not be applied "
            "globally; the previous resolution was kept."
        )
        self._refresh_texture_atlases()

    @Slot(object)
    def _handle_current_texture_item_pressed(self, _raw_item: object) -> None:
        """Make the current-family pane the unambiguous deletion target."""

        signals_were_blocked = self.other_texture_list.blockSignals(True)
        try:
            self.other_texture_list.setCurrentRow(-1)
            self.other_texture_list.clearSelection()
        finally:
            self.other_texture_list.blockSignals(signals_were_blocked)
        self._sync_controls()

    @Slot(object, object)
    def _handle_other_texture_selection_changed(
        self,
        _current: object,
        _previous: object,
    ) -> None:
        """Keep deletion availability synchronized with library selection."""

        self._sync_controls()

    @Slot(object)
    def _handle_texture_variant_activated(self, raw_entry: object) -> None:
        """Apply a double-clicked texture family variant to the selection."""

        if not isinstance(raw_entry, TextureAtlasEntry):
            return
        target = self._texture_variant_entry_targets.get(raw_entry.atlas_id)
        selected_surface_ids = self.surface_view.get_selected_surface_ids()
        if target is None or not selected_surface_ids:
            return
        assignment_id, resolution = target
        assignment = self._assignment_by_id(assignment_id)
        if (
            assignment is None
            or (
                assignment.selected_texture_resolution != resolution
                and not self._request_global_texture_resolution_change(
                    assignment_id,
                    resolution,
                )
            )
        ):
            self.status_label.setText(
                "The selected surface texture resolution could not be applied "
                "globally; the previous resolution was kept."
            )
            self._refresh_texture_atlases()
            return
        if not self.select_assignment_texture_resolution(
            assignment_id,
            resolution,
            selected_surface_ids,
        ):
            self.status_label.setText(
                "The selected surface texture resolution could not be applied."
            )

    @Slot(object)
    def _handle_other_texture_activated(self, raw_item: object) -> None:
        """Apply another compatible texture family to selected surfaces."""

        if self.is_generating or not isinstance(raw_item, QListWidgetItem):
            return
        entry_id = str(raw_item.data(Qt.ItemDataRole.UserRole) or "")
        target = self._other_texture_entry_targets.get(entry_id)
        selected_surface_ids = self.surface_view.get_selected_surface_ids()
        selected_surface_type = self.surface_view.get_selected_surface_type()
        if (
            target is None
            or not selected_surface_ids
            or selected_surface_type is None
        ):
            return
        assignment_id, _ = target
        assignment = self._assignment_by_id(assignment_id)
        if assignment is None or assignment.surface_type != selected_surface_type:
            return
        if self.apply_assignment_texture(
            assignment_id,
            selected_surface_ids,
        ):
            return
        self.status_label.setText(
            "The selected surface texture could not be applied."
        )
        self._refresh_texture_atlases()

    def _selected_texture_assignment_id_for_deletion(self) -> str | None:
        """Resolve an explicit library choice, then the current surface family."""

        other_item = self.other_texture_list.currentItem()
        if other_item is not None and other_item.isSelected():
            entry_id = str(
                other_item.data(Qt.ItemDataRole.UserRole) or ""
            )
            target = self._other_texture_entry_targets.get(entry_id)
            if (
                target is not None
                and self._assignment_by_id(target[0]) is not None
            ):
                return target[0]
        assignment_id = self._latest_selected_surface_assignment_id()
        return (
            assignment_id
            if assignment_id is not None
            and self._assignment_by_id(assignment_id) is not None
            else None
        )

    def _handle_delete_texture_clicked(self) -> None:
        """Confirm and delete the texture family selected in either pane."""

        if self.is_generating:
            return
        assignment_id = self._selected_texture_assignment_id_for_deletion()
        assignment = (
            None
            if assignment_id is None
            else self._assignment_by_id(assignment_id)
        )
        if assignment is None:
            self.status_label.setText("Select a surface texture to delete.")
            self._sync_controls()
            return
        answer = QMessageBox.question(
            self,
            "Delete surface texture",
            (
                f"Delete this {assignment.surface_type} texture from all "
                f"{len(assignment.surface_ids)} assigned surface(s)?\n\n"
                f"Texture ID: {assignment.assignment_id}\n\n"
                "Every generated resolution in this texture family will be "
                "deleted."
            ),
            (
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.Cancel
            ),
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.delete_assignment_texture(assignment.assignment_id)

    def _request_global_texture_resolution_change(
        self,
        assignment_id: str,
        resolution: int,
    ) -> bool:
        """Commit through the host transaction, or locally when standalone."""

        handler = self._texture_resolution_change_handler
        if handler is None:
            return self.select_assignment_texture_resolution(
                assignment_id,
                resolution,
            )
        try:
            return bool(handler(assignment_id, resolution))
        except Exception:
            return False

    def _assignment_by_id(
        self,
        assignment_id: str,
    ) -> SurfaceTextureAssignment | None:
        """Return one live assignment without cloning the workspace state."""

        normalized_id = str(assignment_id).strip()
        return next(
            (
                assignment
                for assignment in self._data.assignments
                if assignment.assignment_id == normalized_id
            ),
            None,
        )

    def _latest_selected_surface_assignment_id(
        self,
        valid_assignment_ids: set[str] | None = None,
    ) -> str | None:
        selected_ids = self.surface_view.get_selected_surface_ids()
        if not selected_ids:
            return None
        selected_surface_ids = set(selected_ids)
        for assignment in reversed(self._data.assignments):
            if (
                selected_surface_ids.intersection(assignment.surface_ids)
                and (
                    valid_assignment_ids is None
                    or assignment.assignment_id in valid_assignment_ids
                )
            ):
                return assignment.assignment_id
        return None

    def _persist_texture_variants(
        self,
        surface_ids: tuple[str, ...],
        assignment_id: str,
        texture_variants: SurfaceTextureVariants,
    ) -> _SavedSurfaceTextureOutput:
        """Persist a complete family or remove every partial file on failure."""

        persisted: list[SurfaceTextureVariant] = []
        png_items = tuple(
            (resolution, texture_variants.texture_png_by_resolution[resolution])
            for resolution in SURFACE_TEXTURE_RESOLUTIONS
        )
        try:
            for resolution, texture_png in png_items:
                file_name = f"{assignment_id}.texture-{resolution}.png"
                asset_path = self._persist_texture_file(file_name, texture_png)
                persisted.append(
                    SurfaceTextureVariant(
                        resolution=resolution,
                        asset_path=asset_path,
                    )
                )
        except Exception:
            for variant in persisted:
                try:
                    self._resolve_asset_path(variant.asset_path).unlink(
                        missing_ok=True
                    )
                except (OSError, ValueError):
                    continue
            raise
        return _SavedSurfaceTextureOutput(
            surface_ids=surface_ids,
            assignment_id=assignment_id,
            variants=tuple(persisted),
            texture_png_by_resolution=png_items,
        )

    def _persist_texture_file(
        self,
        file_name: str,
        texture_png: bytes,
    ) -> str:
        self._asset_directory.mkdir(parents=True, exist_ok=True)
        destination = self._asset_directory / file_name
        if destination.exists():
            raise FileExistsError(
                f"A surface texture asset already exists for {file_name}."
            )
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
        self.texture_view.setEnabled(not busy)
        self.other_texture_list.setEnabled(not busy)
        self.delete_texture_button.setEnabled(
            self._selected_texture_assignment_id_for_deletion() is not None
            and not busy
        )
        self.generate_button.setEnabled(
            has_video and has_mask and has_surface and has_key and not busy
        )
        self.undo_inpaint_button.setEnabled(
            bool(self._data.localized_inpaint_undo_stack) and not busy
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


def _is_localized_surface_texture_inpaint(
    request: SurfaceTextureRequest,
) -> bool:
    return bool(
        request.existing_texture_png is not None
        and request.edit_mask_png is not None
        and request.surface_edit_mask_pngs
    )


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
