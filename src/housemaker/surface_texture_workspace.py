# ### Imports ###
from __future__ import annotations

import math
import os
import re
import tempfile
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Protocol

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QObject, QSize, QThread, Qt, Signal, Slot
from PySide6.QtGui import QIcon, QKeyEvent, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
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
from housemaker.generation_state import MASK_MODE_ERASE, MASK_MODE_PAINT, MaskStroke
from housemaker.generation_jobs import GenerationJobManager
from housemaker.generation_views import VideoInpaintView, rasterize_mask_strokes
from housemaker.glb import GeneratedModel
from housemaker.models import LevelData
from housemaker.pbr_maps import (
    ATLAS_MAP_BASE_COLOR,
    ATLAS_MAP_LABELS,
    PBR_MAP_TYPES,
    normalize_pbr_map_types,
)
from housemaker.settings_widget import (
    SURFACE_TEXTURE_PROVIDER_OPTIONS,
    SURFACE_TEXTURE_PROVIDER_SETTING_KEY,
    GenerationServiceSettings,
    read_surface_texture_provider,
)
from housemaker.surface_texture_providers import (
    MESHY_PROVIDER,
    SurfaceTextureResult,
    align_surface_pbr_map_png,
    request_surface_texture,
)
from housemaker.surface_texture_state import (
    SURFACE_PBR_ALIGNMENT_VERSION,
    SURFACE_TYPE_WALL,
    SurfaceTextureAssignment,
    SurfaceTextureData,
    SurfaceTextureVariant,
)
from housemaker.surface_texture_variants import (
    DEFAULT_SURFACE_TEXTURE_RESOLUTION,
    SURFACE_TEXTURE_RESOLUTIONS,
    SurfaceTextureVariants,
    build_surface_texture_variants,
)
from housemaker.surface_texture_viewer import SurfaceTextureViewer
from housemaker.texture_atlas_view import TextureAtlasEntry, TextureAtlasView
from housemaker.video_source import VIDEO_FILE_FILTER, VideoFrameSource, probe_video


# ### Constants ###
DEFAULT_BRUSH_RADIUS_PIXELS = 24
MIN_BRUSH_RADIUS_PIXELS = 1
MAX_BRUSH_RADIUS_PIXELS = 256
MAX_PROVIDER_REFERENCE_IMAGES = 5
MAX_PROVIDER_REFERENCE_EDGE_PIXELS = 2048
MAX_REFERENCE_FRAMES = 100
REFERENCE_PADDING_RATIO = 0.08
INTERRUPT_POLL_SECONDS = 0.05
SHUTDOWN_WAIT_MILLISECONDS = 250
OTHER_TEXTURE_THUMBNAIL_SIZE = QSize(96, 96)
OTHER_TEXTURE_GRID_SIZE = QSize(152, 128)
_PROGRESS_PERCENT_PATTERN = re.compile(r"(?<!\d)(100|[1-9]?\d)\s*%")


# ### Level synchronization helpers ###
def _build_level_sync_signature(
    levels: Sequence[LevelData],
) -> tuple[object, ...]:
    """Snapshot mutable level content used by the Surface scene."""

    return (_freeze_level_sync_value(tuple(levels)),)


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
    display_name: str = ""
    enabled_pbr_maps: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "enabled_pbr_maps",
            normalize_pbr_map_types(
                self.enabled_pbr_maps,
                label="Enabled surface texture PBR maps",
            ),
        )


@dataclass(frozen=True)
class _PreparedSurfaceTextureOutput:
    """One CPU-prepared texture family ready for main-thread persistence."""

    surface_ids: tuple[str, ...]
    texture_variants: SurfaceTextureVariants


@dataclass(frozen=True)
class _SurfaceTextureReferenceInput:
    """Immutable video inputs prepared by one worker."""

    video_path: str
    frame_strokes: tuple[tuple[int, tuple[MaskStroke, ...]], ...]


@dataclass(frozen=True)
class _SavedSurfaceTextureOutput:
    """One completely persisted resolution family awaiting state commit."""

    surface_ids: tuple[str, ...]
    assignment_id: str
    variants: tuple[SurfaceTextureVariant, ...]
    texture_png_by_resolution: tuple[tuple[int, bytes], ...]
    map_png_by_resolution: dict[str, dict[int, bytes]] = field(
        default_factory=dict
    )

    def png_for_resolution(self, resolution: int) -> bytes:
        for candidate_resolution, texture_png in self.texture_png_by_resolution:
            if candidate_resolution == int(resolution):
                return texture_png
        raise ValueError("The saved surface texture resolution is unavailable.")

    def map_pngs_for_resolution(self, resolution: int) -> dict[str, bytes]:
        """Return owned companion-map PNGs for one saved resolution."""

        normalized_resolution = int(resolution)
        return {
            map_type: bytes(resolution_map[normalized_resolution])
            for map_type, resolution_map in self.map_png_by_resolution.items()
            if map_type in PBR_MAP_TYPES
            and normalized_resolution in resolution_map
        }


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
            enabled_pbr_maps=request.enabled_pbr_maps,
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
    succeeded = Signal(str, object, object, object)
    failed = Signal(str, str)
    cancelled = Signal(str)
    finished = Signal(str)
    progress = Signal(str, str)

    def __init__(
        self,
        job_id: str,
        provider: SurfaceTextureProvider | Callable[[SurfaceTextureRequest], object],
        request: SurfaceTextureRequest,
        asset_directory: Path,
        reference_input: _SurfaceTextureReferenceInput | None = None,
    ) -> None:
        super().__init__()
        self._job_id = str(job_id)
        self._provider = provider
        self._request = request
        self._asset_directory = Path(asset_directory)
        self._reference_input = reference_input
        self._cancel_event = threading.Event()
        self._output_lock = threading.Lock()
        self._unclaimed_saved_outputs: tuple[
            _SavedSurfaceTextureOutput,
            ...,
        ] = ()

    def cancel(self) -> None:
        self._cancel_event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def claim_saved_outputs(self) -> None:
        with self._output_lock:
            self._unclaimed_saved_outputs = ()

    def discard_unclaimed_outputs(self) -> None:
        with self._output_lock:
            saved_outputs = self._unclaimed_saved_outputs
            self._unclaimed_saved_outputs = ()
        _discard_surface_texture_outputs(self._asset_directory, saved_outputs)

    @Slot()
    def run(self) -> None:
        try:
            request = self._request
            if self._reference_input is not None:
                self.progress.emit(
                    self._job_id,
                    "Preparing reference frames (5%)",
                )
                reference_pngs, frame_indices = (
                    _build_reference_pngs_from_input(
                        self._reference_input,
                        self._cancel_event,
                    )
                )
                if not reference_pngs:
                    raise ValueError("The painted reference masks are empty.")
                request = replace(
                    request,
                    reference_pngs=reference_pngs,
                    reference_frame_indices=frame_indices,
                )
                self._request = request
            _raise_surface_worker_cancelled(self._cancel_event)
            result = _run_interruptible_stage(
                lambda: _invoke_provider(
                    self._provider,
                    request,
                    lambda message: self.progress.emit(
                        self._job_id,
                        _format_surface_provider_progress(message),
                    ),
                    self._cancel_event,
                )
            )
            if not isinstance(result, SurfaceTextureResult):
                raise TypeError("The texture provider returned an invalid result.")
            _raise_surface_worker_cancelled(self._cancel_event)
            self.progress.emit(
                self._job_id,
                "Preparing texture variants (90%)",
            )
            prepared_outputs = _prepare_surface_texture_outputs(
                request,
                result,
                self._cancel_event,
            )
            _raise_surface_worker_cancelled(self._cancel_event)
            self.progress.emit(
                self._job_id,
                "Saving texture variants (95%)",
            )
            saved_outputs = _persist_prepared_surface_texture_outputs(
                self._asset_directory,
                prepared_outputs,
                self._cancel_event,
            )
            with self._output_lock:
                self._unclaimed_saved_outputs = saved_outputs
            _raise_surface_worker_cancelled(self._cancel_event)
            self.succeeded.emit(
                self._job_id,
                request,
                result,
                saved_outputs,
            )
        except _SurfaceTextureCancelled:
            self.discard_unclaimed_outputs()
            self.cancelled.emit(self._job_id)
        except Exception as error:
            self.discard_unclaimed_outputs()
            if self._cancel_event.is_set():
                self.cancelled.emit(self._job_id)
            else:
                self.failed.emit(
                    self._job_id,
                    _redact_error(error, self._request.api_key),
                )
        finally:
            self.finished.emit(self._job_id)


# ### Surface texture workspace ###
class SurfaceTextureGenerationWorkspace(QWidget):
    """Select fixed surfaces and texture them from masked video references."""

    data_changed = Signal(object)
    generation_completed = Signal(object)
    assignments_removed = Signal(object)
    surface_content_changed = Signal()

    def __init__(
        self,
        provider: SurfaceTextureProvider
        | Callable[[SurfaceTextureRequest], object]
        | None = None,
        asset_directory: str | Path | None = None,
        application_settings: ApplicationSettingsStore | None = None,
        job_manager: GenerationJobManager | None = None,
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
        self._job_manager = (
            job_manager
            if job_manager is not None
            else GenerationJobManager(self)
        )
        self._settings = GenerationServiceSettings(
            surface_texture_provider=read_surface_texture_provider(
                self._application_settings
            )
        )
        self._is_syncing_provider = False
        self._data = SurfaceTextureData()
        self._levels: list[LevelData] = []
        self._level_sync_signature: tuple[object, ...] | None = None
        self._video_source: VideoFrameSource | None = None
        self._displayed_frame_index: int | None = None
        self._is_syncing_seekbar = False
        self._generation_threads: dict[str, QThread] = {}
        self._generation_workers: dict[str, SurfaceTextureWorker] = {}
        self._generation_requests: dict[str, SurfaceTextureRequest] = {}
        self._generation_surface_targets: dict[str, frozenset[str]] = {}
        self._cancelled_generation_job_ids: set[str] = set()
        self._is_shutting_down = False
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
        return bool(self._generation_threads)

    @property
    def _generation_thread(self) -> QThread | None:
        """Compatibility view of the first active Surface worker thread."""

        return next(iter(self._generation_threads.values()), None)

    @property
    def _generation_worker(self) -> SurfaceTextureWorker | None:
        """Compatibility view of the first active Surface worker."""

        return next(iter(self._generation_workers.values()), None)

    def get_data(self) -> SurfaceTextureData:
        self._store_displayed_frame_strokes()
        self._store_viewer_state()
        return self._data.clone()

    def get_surface_material_sources(
        self,
    ) -> dict[str, Path | dict[str, Path]]:
        """Resolve live assignment assets with later assignments taking priority."""

        material_sources: dict[str, Path | dict[str, Path]] = {}
        for assignment in self._data.assignments:
            map_paths = self.get_assignment_map_asset_paths(
                assignment.assignment_id,
                assignment.selected_texture_resolution,
            )
            base_path = map_paths.get(ATLAS_MAP_BASE_COLOR)
            if base_path is None:
                continue
            source: Path | dict[str, Path] = (
                base_path if len(map_paths) == 1 else map_paths
            )
            for surface_id in assignment.surface_ids:
                material_sources[surface_id] = source
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

        signature: list[tuple[object, ...]] = []
        for assignment in self._data.assignments:
            raw_map_paths = self._active_assignment_map_asset_paths(assignment)
            signature.append(
                (
                    assignment.assignment_id,
                    assignment.selected_texture_resolution,
                    assignment.asset_path,
                    tuple(
                        (
                            map_type,
                            asset_path,
                            _get_cached_surface_asset_revision(
                                revision_cache,
                                self._asset_directory,
                                asset_path,
                            ),
                        )
                        for map_type, asset_path in raw_map_paths.items()
                    ),
                )
            )
        return tuple(signature)

    def refresh_file_backed_previews(self) -> None:
        """Reload only Surface assets whose on-disk revisions changed."""

        revision_cache: dict[str, tuple[object, ...]] = {}
        material_signature = self.get_preview_dependency_signature()
        for signature_item in material_signature:
            if len(signature_item) < 4:
                continue
            raw_map_revisions = signature_item[3]
            if not isinstance(raw_map_revisions, tuple | list):
                continue
            for raw_map_revision in raw_map_revisions:
                if (
                    not isinstance(raw_map_revision, tuple | list)
                    or len(raw_map_revision) < 3
                    or not isinstance(raw_map_revision[1], str)
                    or not isinstance(raw_map_revision[2], tuple)
                ):
                    continue
                revision_cache[raw_map_revision[1]] = raw_map_revision[2]
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
                assignment.display_name,
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
                        tuple(
                            (
                                map_type,
                                map_asset_path,
                                _get_cached_surface_asset_revision(
                                    cached_revisions,
                                    self._asset_directory,
                                    map_asset_path,
                                ),
                            )
                            for map_type, map_asset_path in (
                                variant.map_asset_paths.items()
                            )
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

    def get_assignment_map_asset_paths(
        self,
        assignment_id: str,
        resolution: int | None = None,
    ) -> dict[str, Path]:
        """Resolve every live map belonging to one exact assignment variant."""

        assignment = self._assignment_by_id(assignment_id)
        if assignment is None:
            return {}
        raw_paths: Mapping[str, str]
        if assignment.texture_variants:
            target_resolution = (
                assignment.selected_texture_resolution
                if resolution is None
                else int(resolution)
            )
            if target_resolution is None:
                return {}
            variant = assignment.texture_variant_for_resolution(
                target_resolution
            )
            if variant is None:
                return {}
            raw_paths = variant.map_asset_paths
        else:
            raw_paths = {ATLAS_MAP_BASE_COLOR: assignment.asset_path}
        resolved: dict[str, Path] = {}
        for map_type, raw_path in raw_paths.items():
            try:
                map_path = self._resolve_asset_path(raw_path)
            except ValueError:
                continue
            if map_path.is_file():
                resolved[map_type] = map_path
        if ATLAS_MAP_BASE_COLOR not in resolved:
            return {}
        return resolved

    @staticmethod
    def _active_assignment_map_asset_paths(
        assignment: SurfaceTextureAssignment,
    ) -> dict[str, str]:
        """Return logical map paths for the assignment's active resolution."""

        if assignment.texture_variants:
            selected_resolution = assignment.selected_texture_resolution
            if selected_resolution is not None:
                variant = assignment.texture_variant_for_resolution(
                    selected_resolution
                )
                if variant is not None:
                    return dict(variant.map_asset_paths)
        return {ATLAS_MAP_BASE_COLOR: assignment.asset_path}

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

    def get_assignments(self) -> tuple[SurfaceTextureAssignment, ...]:
        """Return every immutable generated-surface assignment."""

        return tuple(self._data.assignments)

    def can_select_assignment_texture_resolution(
        self,
        assignment_id: str,
        resolution: int,
    ) -> bool:
        """Return whether an exact generated surface variant can be selected."""

        assignment = self._assignment_by_id(assignment_id)
        if assignment is None or self._assignment_is_reserved(assignment):
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
        if self._assignment_is_reserved(source_assignment):
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
        map_paths = self.get_assignment_map_asset_paths(
            source_assignment.assignment_id,
            target_resolution,
        )
        texture_path = map_paths.get(ATLAS_MAP_BASE_COLOR)
        if texture_path is None:
            return False
        try:
            texture_png = texture_path.read_bytes()
            map_pngs = {
                map_type: map_path.read_bytes()
                for map_type, map_path in map_paths.items()
                if map_type in PBR_MAP_TYPES
            }
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
            map_texture_pngs=map_pngs,
        )

    def apply_assignment_texture(
        self,
        assignment_id: str,
        target_surface_ids: Sequence[str] = (),
    ) -> bool:
        """Apply one family's active asset without changing its resolution."""

        source_assignment = self._assignment_by_id(assignment_id)
        if source_assignment is None:
            return False
        if self._assignment_is_reserved(source_assignment):
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
            map_texture_pngs={},
        )

    def delete_assignment_texture(self, assignment_id: str) -> bool:
        """Delete one complete texture family from every assigned surface."""

        if self.is_generating and not self._generation_threads:
            return False
        assignment = self._assignment_by_id(assignment_id)
        if assignment is None:
            return False
        if self._assignment_is_reserved(assignment):
            return False

        previous_assignments = list(self._data.assignments)
        next_assignments = [
            candidate
            for candidate in previous_assignments
            if candidate.assignment_id != assignment.assignment_id
        ]

        self._data.assignments = next_assignments
        try:
            self._restore_assignment_textures()
        except (OSError, TypeError, ValueError):
            self._data.assignments = previous_assignments
            try:
                self._restore_assignment_textures()
            except (OSError, RuntimeError, TypeError, ValueError):
                pass
            self.status_label.setText(
                "The selected surface texture could not be deleted."
            )
            self._sync_controls()
            return False

        removed_assignments = [assignment]
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
        map_texture_pngs: Mapping[str, bytes] | None = None,
    ) -> bool:
        """Commit one validated active family asset to homogeneous surfaces."""

        normalized_targets = tuple(
            dict.fromkeys(str(surface_id) for surface_id in target_surface_ids)
        )
        if self._assignment_is_reserved(
            source_assignment
        ) or self._surface_targets_are_reserved(normalized_targets):
            return False
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
        try:
            self._set_surface_view_texture(
                selected_assignment.surface_ids,
                texture_png,
                map_texture_pngs,
            )
            self._data.assignments = next_assignments
            # The caller supplied bytes read before this commit.  Do not bind
            # them to a file revision sampled afterward: the backing PNG may
            # have been replaced between the read and this installation.
            self._restored_assignment_texture_signature = None
        except (OSError, TypeError, ValueError):
            self._data.assignments = previous_assignments
            self._restore_assignment_textures()
            return False

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
        migration_failure_count = self._migrate_legacy_meshy_pbr_alignment()
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
        self.surface_view.set_levels(self._levels)
        self._data.camera_pose = saved_camera_pose
        self._restore_viewer_state()
        self._restore_assignment_textures()
        self._refresh_texture_atlases()
        self._sync_video_controls()
        if self._video_source is not None:
            self.show_frame(self._data.current_frame_index)
        self._sync_selection_status()
        self._sync_controls()
        if migration_failure_count:
            self.status_label.setText(
                f"{migration_failure_count} legacy surface PBR texture set(s) "
                "could not be aligned automatically."
            )

    def set_levels(
        self,
        levels: Sequence[LevelData],
    ) -> None:
        self._set_level_context(
            levels,
            preview_model=None,
            replace_preview_model=False,
        )

    def set_preview_context(
        self,
        levels: Sequence[LevelData],
        model: GeneratedModel | None,
    ) -> None:
        """Synchronize semantic levels and the shared model with one GL rebuild."""

        self._set_level_context(
            levels,
            preview_model=model,
            replace_preview_model=True,
        )

    def _set_level_context(
        self,
        levels: Sequence[LevelData],
        *,
        preview_model: GeneratedModel | None,
        replace_preview_model: bool,
    ) -> None:
        normalized_levels = list(levels)
        next_signature = _build_level_sync_signature(normalized_levels)
        if next_signature == self._level_sync_signature:
            # Retain the latest model objects without rebuilding equivalent
            # semantic geometry, textures, atlas controls, or OpenGL items.
            self._levels = normalized_levels
            self.refresh_file_backed_previews()
            if replace_preview_model:
                self.set_preview_model(preview_model)
            return

        if self._levels:
            self._store_viewer_state()
        self._levels = normalized_levels
        if replace_preview_model:
            self.surface_view.set_scene_model(
                preview_model,
                repopulate=False,
            )
            self.surface_view.clear_surface_textures()
        self.surface_view.set_levels(self._levels)
        self._restore_viewer_state()
        self._restore_assignment_textures()
        self._refresh_texture_atlases()
        self._sync_selection_status()
        self._sync_controls()
        self._level_sync_signature = next_signature

    def _migrate_legacy_meshy_pbr_alignment(self) -> int:
        """Point pre-v1 Meshy families at versioned aligned derivatives."""

        migrated_assignments: list[SurfaceTextureAssignment] = []
        failure_count = 0
        for assignment in self._data.assignments:
            if assignment.pbr_alignment_version >= SURFACE_PBR_ALIGNMENT_VERSION:
                migrated_assignments.append(assignment)
                continue
            if assignment.provider.casefold() != MESHY_PROVIDER:
                migrated_assignments.append(assignment)
                continue

            map_files: dict[tuple[str, str], tuple[Path, Path]] = {}
            aligned_paths: dict[tuple[str, str], str] = {}
            migration_is_valid = True
            for variant in assignment.texture_variants:
                for map_type in PBR_MAP_TYPES:
                    raw_path = variant.map_asset_paths.get(map_type)
                    if raw_path is None:
                        continue
                    try:
                        source_path = self._resolve_asset_path(raw_path)
                        aligned_path = _build_aligned_pbr_asset_path(
                            raw_path,
                            map_type,
                        )
                        destination_path = self._resolve_asset_path(aligned_path)
                    except ValueError:
                        migration_is_valid = False
                        break
                    source_key = (raw_path, map_type)
                    if source_key in map_files:
                        continue
                    if destination_path in {
                        candidate[1] for candidate in map_files.values()
                    }:
                        migration_is_valid = False
                        break
                    map_files[source_key] = (source_path, destination_path)
                    aligned_paths[source_key] = aligned_path
                if not migration_is_valid:
                    break

            if not map_files:
                migrated_assignments.append(assignment)
                continue
            if not migration_is_valid:
                migrated_assignments.append(assignment)
                failure_count += 1
                continue

            aligned_variants = tuple(
                replace(
                    variant,
                    map_asset_paths={
                        map_type: aligned_paths.get(
                            (raw_path, map_type),
                            raw_path,
                        )
                        for map_type, raw_path in variant.map_asset_paths.items()
                    },
                )
                for variant in assignment.texture_variants
            )
            try:
                migrated_assignment = replace(
                    assignment,
                    texture_variants=aligned_variants,
                    pbr_alignment_version=SURFACE_PBR_ALIGNMENT_VERSION,
                )
            except ValueError:
                migrated_assignments.append(assignment)
                failure_count += 1
                continue

            target_snapshots: dict[Path, bytes | None] = {}
            aligned_payloads: dict[Path, bytes] = {}
            try:
                for (raw_path, map_type), (
                    source_path,
                    destination_path,
                ) in map_files.items():
                    payload = source_path.read_bytes()
                    target_snapshots[destination_path] = (
                        destination_path.read_bytes()
                        if destination_path.is_file()
                        else None
                    )
                    aligned_payloads[destination_path] = align_surface_pbr_map_png(
                        payload,
                        map_type=map_type,
                        label=f"Legacy surface {map_type} texture {raw_path}",
                    )
                for path, payload in aligned_payloads.items():
                    _replace_surface_texture_file(path, payload)
            except (OSError, RuntimeError, TypeError, ValueError):
                for path, payload in target_snapshots.items():
                    try:
                        if payload is None:
                            path.unlink(missing_ok=True)
                        else:
                            _replace_surface_texture_file(path, payload)
                    except OSError:
                        pass
                migrated_assignments.append(assignment)
                failure_count += 1
                continue
            migrated_assignments.append(migrated_assignment)
        self._data.assignments = migrated_assignments
        return failure_count

    def set_preview_model(self, model: GeneratedModel | None) -> None:
        """Use the Canvas model while retaining semantic surface selection."""

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
        request = self._build_request(
            display_name=self.job_name_edit.text(),
            defer_reference_preparation=True,
        )
        if request is None:
            return
        reference_input = self._build_reference_input(request)
        if reference_input is None:
            return
        if self._start_generation(request, reference_input=reference_input):
            self.job_name_edit.clear()

    def shutdown(self) -> None:
        if self._is_shutting_down:
            self._close_video_source()
            return
        self._is_shutting_down = True
        active_jobs = tuple(self._generation_threads.items())
        for job_id, thread in active_jobs:
            worker = self._generation_workers.get(job_id)
            self._cancelled_generation_job_ids.add(job_id)
            if worker is not None and is_valid_qt_object(worker):
                worker.cancel()
                worker.discard_unclaimed_outputs()
                for signal, slot in (
                    (worker.succeeded, self._handle_generation_job_succeeded),
                    (worker.failed, self._handle_generation_job_failed),
                    (worker.progress, self._handle_generation_job_progress),
                    (worker.cancelled, self._handle_generation_job_cancelled),
                ):
                    try:
                        signal.disconnect(slot)
                    except (RuntimeError, TypeError):
                        pass
            self._job_manager.mark_cancelled(job_id)
            if is_valid_qt_object(thread):
                thread.requestInterruption()
                thread.quit()
        for job_id, thread in active_jobs:
            while is_valid_qt_object(thread) and thread.isRunning():
                # Provider calls are isolated in interruptible daemon threads.
                # Work left directly on this QThread is finite image/file work,
                # so retain its QObject and wait until it is safe to destroy.
                thread.wait(SHUTDOWN_WAIT_MILLISECONDS)
            worker = self._generation_workers.get(job_id)
            if worker is not None and is_valid_qt_object(worker):
                # The worker may have persisted its files after the first
                # discard but before its thread observed cancellation.
                worker.discard_unclaimed_outputs()
        self._generation_threads.clear()
        self._generation_workers.clear()
        self._generation_requests.clear()
        self._generation_surface_targets.clear()
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
        self.mask_mode_control = QWidget()
        self.mask_mode_control.setObjectName(
            "surface_texture_mask_mode_control"
        )
        mask_mode_layout = QVBoxLayout(self.mask_mode_control)
        mask_mode_layout.setContentsMargins(0, 0, 0, 0)
        mask_mode_layout.setSpacing(0)
        mask_mode_layout.addWidget(self.paint_mask_button)
        mask_mode_layout.addWidget(self.erase_mask_button)
        second_row.addWidget(self.mask_mode_control)
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
        second_row.addWidget(self.brush_size_spinbox)
        self.clear_mask_button = QPushButton("Clear frame mask")
        self.clear_mask_button.clicked.connect(self.video_view.clear_mask)
        second_row.addWidget(self.clear_mask_button)
        self.material_notes_edit = QLineEdit()
        self.material_notes_edit.setPlaceholderText(
            "Optional material notes, e.g. pale oak or matte plaster"
        )
        second_row.addWidget(self.material_notes_edit, 1)
        second_row.addWidget(QLabel("Job name"))
        self.job_name_edit = QLineEdit()
        self.job_name_edit.setObjectName("surface_texture_job_name_edit")
        self.job_name_edit.setPlaceholderText("Optional")
        self.job_name_edit.setMaxLength(256)
        second_row.addWidget(self.job_name_edit)
        self.pbr_map_control = QWidget()
        self.pbr_map_control.setObjectName(
            "surface_texture_pbr_map_control"
        )
        pbr_map_layout = QGridLayout(self.pbr_map_control)
        pbr_map_layout.setContentsMargins(0, 0, 0, 0)
        pbr_map_layout.setHorizontalSpacing(6)
        pbr_map_layout.setVerticalSpacing(0)
        self.pbr_map_checkboxes: dict[str, QCheckBox] = {}
        for index, map_type in enumerate(PBR_MAP_TYPES):
            checkbox = QCheckBox(ATLAS_MAP_LABELS[map_type])
            checkbox.setObjectName(
                f"surface_pbr_{map_type}_checkbox"
            )
            checkbox.setToolTip(
                "Request Meshy's aligned Surface PBR family and dynamically "
                f"apply the {ATLAS_MAP_LABELS[map_type].lower()} map in the "
                "3D preview."
            )
            checkbox.toggled.connect(self._handle_pbr_map_toggled)
            self.pbr_map_checkboxes[map_type] = checkbox
            pbr_map_layout.addWidget(checkbox, index % 3, index // 3)
        second_row.addWidget(self.pbr_map_control)
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

    def _start_generation(
        self,
        request: SurfaceTextureRequest,
        *,
        reference_input: _SurfaceTextureReferenceInput | None = None,
    ) -> bool:
        """Start one independently tracked job unless its surfaces are busy."""

        if self._is_shutting_down:
            return False
        if self._surface_targets_are_reserved(request.surface_ids):
            self.status_label.setText(
                "A surface texture job is already using part of this selection."
            )
            self._sync_controls()
            return False
        job = self._job_manager.create_job(
            kind="Surface texture",
            requested_name=request.display_name,
            default_name=_default_surface_texture_name(request.surface_type),
            stage="Sending texture request",
        )
        job_id = job.job_id
        thread = QThread(self)
        worker = SurfaceTextureWorker(
            job_id,
            self._provider,
            request,
            self._asset_directory,
            reference_input,
        )
        self._generation_threads[job_id] = thread
        self._generation_workers[job_id] = worker
        self._generation_requests[job_id] = request
        self._generation_surface_targets[job_id] = frozenset(
            request.surface_ids
        )
        self._job_manager.set_cancel_callback(
            job_id,
            lambda active_job_id=job_id: self._cancel_generation_job(
                active_job_id
            ),
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.succeeded.connect(self._handle_generation_job_succeeded)
        worker.failed.connect(self._handle_generation_job_failed)
        worker.cancelled.connect(self._handle_generation_job_cancelled)
        worker.progress.connect(self._handle_generation_job_progress)
        worker.finished.connect(self._handle_generation_worker_finished)
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        self.status_label.setText(
            f"{job.name}: sending {len(request.reference_frame_indices)} "
            "painted frame(s) "
            f"to {_get_provider_display_name(request.provider)}..."
        )
        thread.start()
        self._sync_controls()
        return True

    def _build_request(
        self,
        display_name: str = "",
        *,
        defer_reference_preparation: bool = False,
    ) -> SurfaceTextureRequest | None:
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
        if defer_reference_preparation:
            reference_pngs = ()
            used_frame_indices = frame_indices
        else:
            try:
                reference_pngs, used_frame_indices = self._build_reference_pngs(
                    frame_indices
                )
            except (OSError, ValueError) as error:
                self.status_label.setText(str(error))
                return None
            if not reference_pngs:
                self.status_label.setText(
                    "The painted reference masks are empty."
                )
                return None
        area_m2 = self.surface_view.get_combined_selected_area()
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
            display_name=str(display_name).strip()[:256],
            enabled_pbr_maps=(
                self._get_enabled_pbr_maps()
                if self._settings.surface_texture_provider == MESHY_PROVIDER
                else ()
            ),
        )

    def _build_reference_input(
        self,
        request: SurfaceTextureRequest,
    ) -> _SurfaceTextureReferenceInput | None:
        """Snapshot cheap inputs while deferring image preparation."""

        video_source = self._video_source
        if video_source is None:
            self.status_label.setText("Load the source video before generating.")
            return None
        return _SurfaceTextureReferenceInput(
            video_path=video_source.metadata.path,
            frame_strokes=tuple(
                (
                    frame_index,
                    tuple(self._data.strokes_for_frame(frame_index)),
                )
                for frame_index in request.reference_frame_indices
            ),
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
        return tuple(
            _encode_provider_reference_png(image) for image in packed_images
        ), tuple(used_indices)

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

    @Slot(object, object)
    def _handle_generation_succeeded(
        self,
        request: SurfaceTextureRequest,
        result: SurfaceTextureResult,
        prepared_outputs: Sequence[_PreparedSurfaceTextureOutput] | None = None,
        saved_outputs: Sequence[_SavedSurfaceTextureOutput] | None = None,
    ) -> bool:
        if prepared_outputs is None and saved_outputs is None:
            try:
                raw_outputs = _build_surface_texture_outputs(
                    request,
                    result.texture_png,
                )
                prepared_outputs = tuple(
                    _PreparedSurfaceTextureOutput(
                        surface_ids=surface_ids,
                        texture_variants=build_surface_texture_variants(
                            texture_png,
                            result.pbr_texture_pngs,
                        ),
                    )
                    for surface_ids, texture_png in raw_outputs
                )
            except ValueError as error:
                self._handle_generation_failed(str(error))
                return False
        saved_output_items = list(saved_outputs or ())
        if saved_outputs is None:
            assert prepared_outputs is not None
            try:
                for prepared_output in prepared_outputs:
                    assignment_id = uuid.uuid4().hex
                    saved_output_items.append(
                        self._persist_texture_variants(
                            prepared_output.surface_ids,
                            assignment_id,
                            prepared_output.texture_variants,
                        )
                    )
            except (OSError, ValueError) as error:
                self._discard_saved_outputs(saved_output_items)
                self._handle_generation_failed(
                    f"The generated texture could not be saved: {error}"
                )
                return False
        assignments: list[SurfaceTextureAssignment] = []
        try:
            for saved_output in saved_output_items:
                selected_resolution = DEFAULT_SURFACE_TEXTURE_RESOLUTION
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
                        provider_pbr_task_id=result.pbr_task_id,
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
                        display_name=request.display_name,
                        enabled_pbr_maps=request.enabled_pbr_maps,
                        available_pbr_maps=tuple(
                            map_type
                            for map_type in PBR_MAP_TYPES
                            if map_type in active_variant.map_asset_paths
                        ),
                        pbr_alignment_version=SURFACE_PBR_ALIGNMENT_VERSION,
                    )
                )
        except (OSError, ValueError) as error:
            self._discard_saved_outputs(saved_output_items)
            self._handle_generation_failed(
                f"The generated texture could not be applied: {error}"
            )
            return False

        try:
            retained_assignments, removed_assignments = (
                self._replace_assignments_for_surfaces(assignments)
            )
            for saved_output, assignment in zip(
                saved_output_items,
                assignments,
                strict=True,
            ):
                self._set_surface_view_texture(
                    saved_output.surface_ids,
                    saved_output.png_for_resolution(
                        assignment.selected_texture_resolution
                        or DEFAULT_SURFACE_TEXTURE_RESOLUTION
                    ),
                    saved_output.map_pngs_for_resolution(
                        assignment.selected_texture_resolution
                        or DEFAULT_SURFACE_TEXTURE_RESOLUTION
                    ),
                )
        except (OSError, TypeError, ValueError) as error:
            self._restore_assignment_textures()
            self._discard_saved_outputs(saved_output_items)
            self._handle_generation_failed(
                f"The generated texture could not be applied: {error}"
            )
            return False

        self._data.assignments = [*retained_assignments, *assignments]
        # These pixels came from the generated in-memory payload.  Leave the
        # file-backed cache unvalidated so activation confirms the exact bytes
        # that were persisted, including a concurrent same-path replacement.
        self._restored_assignment_texture_signature = None
        self._texture_atlas_entry_cache.clear()
        self._refresh_texture_atlases()
        cleanup_failure_count = self._delete_orphaned_assignment_assets(
            removed_assignments
        )
        status = (
            f"Applied {request.display_name!r} to {len(request.surface_ids)} "
            f"{request.surface_type} surface(s)."
            if request.display_name
            else (
                f"Applied generated texture to {len(request.surface_ids)} "
                f"{request.surface_type} surface(s)."
            )
        )
        if cleanup_failure_count:
            status += (
                f" {cleanup_failure_count} replaced texture file(s) could not "
                "be deleted."
        )
        self.status_label.setText(status)
        self._emit_data_changed()
        removed_assignment_ids = self._unretained_assignment_ids(
            removed_assignments
        )
        if removed_assignment_ids:
            self.assignments_removed.emit(removed_assignment_ids)
        for assignment in assignments:
            self.generation_completed.emit(assignment)
        self.surface_content_changed.emit()
        self._sync_controls()
        return True

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

    def _discard_saved_outputs(
        self,
        saved_outputs: Sequence[_SavedSurfaceTextureOutput],
    ) -> None:
        for saved_output in saved_outputs:
            for variant in saved_output.variants:
                for raw_asset_path in variant.map_asset_paths.values():
                    try:
                        self._resolve_asset_path(raw_asset_path).unlink(
                            missing_ok=True
                        )
                    except (OSError, ValueError):
                        continue

    def _delete_orphaned_assignment_assets(
        self,
        removed_assignments: Sequence[SurfaceTextureAssignment],
    ) -> int:
        active_asset_paths: set[Path] = set()
        for assignment in self._data.assignments:
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
                        map_asset_path
                        for variant in assignment.texture_variants
                        for map_asset_path in (
                            variant.map_asset_paths.values()
                        )
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
        return tuple(
            dict.fromkeys(
                assignment.assignment_id
                for assignment in assignments
                if assignment.assignment_id not in retained_ids
            )
        )

    @Slot(str)
    def _handle_generation_failed(self, message: str) -> None:
        self.status_label.setText(message)

    @Slot(str, object, object, object)
    def _handle_generation_job_succeeded(
        self,
        job_id: str,
        raw_request: object,
        raw_result: object,
        raw_prepared_outputs: object,
    ) -> None:
        """Commit a worker result on the GUI thread for its exact job."""

        worker = self._generation_workers.get(job_id)
        if (
            self._is_shutting_down
            or worker is None
            or job_id in self._cancelled_generation_job_ids
            or not isinstance(raw_request, SurfaceTextureRequest)
            or not isinstance(raw_result, SurfaceTextureResult)
            or not isinstance(raw_prepared_outputs, tuple | list)
            or not all(
                isinstance(output, _SavedSurfaceTextureOutput)
                for output in raw_prepared_outputs
            )
        ):
            if worker is not None:
                worker.discard_unclaimed_outputs()
            return
        missing_targets = tuple(
            surface_id
            for surface_id in raw_request.surface_ids
            if self.surface_view.get_surface(surface_id) is None
        )
        if missing_targets:
            message = (
                "The target surfaces changed before the generated texture "
                "could be applied."
            )
            self._handle_generation_failed(message)
            self._job_manager.fail_job(job_id, message)
            worker.discard_unclaimed_outputs()
            return
        self._job_manager.update_job(
            job_id,
            stage="Applying texture (99%)",
            progress=99,
        )
        worker.claim_saved_outputs()
        committed = self._handle_generation_succeeded(
            raw_request,
            raw_result,
            saved_outputs=tuple(raw_prepared_outputs),
        )
        if committed:
            self._job_manager.complete_job(job_id)
        else:
            self._job_manager.fail_job(job_id, self.status_label.text())

    @Slot(str, str)
    def _handle_generation_job_failed(
        self,
        job_id: str,
        message: str,
    ) -> None:
        if self._is_shutting_down or job_id not in self._generation_workers:
            return
        job = self._job_manager.get_job(job_id)
        prefix = "" if job is None else f"{job.name}: "
        self._handle_generation_failed(prefix + message)
        self._job_manager.fail_job(job_id, message)

    @Slot(str, str)
    def _handle_generation_job_progress(
        self,
        job_id: str,
        stage: str,
    ) -> None:
        managed_job = self._job_manager.get_job(job_id)
        if (
            self._is_shutting_down
            or job_id not in self._generation_workers
            or job_id in self._cancelled_generation_job_ids
            or managed_job is None
            or managed_job.is_finished
        ):
            return
        job = self._job_manager.update_job(job_id, stage=stage)
        if job is not None and not job.is_finished:
            self.status_label.setText(f"{job.name}: {job.stage}")

    @Slot(str)
    def _handle_generation_job_cancelled(self, job_id: str) -> None:
        if job_id not in self._generation_workers:
            return
        self._cancelled_generation_job_ids.add(job_id)
        self._job_manager.mark_cancelled(job_id)
        job = self._job_manager.get_job(job_id)
        self.status_label.setText(
            "Surface texture generation cancelled."
            if job is None
            else f"Cancelled: {job.name}."
        )

    @Slot(str)
    def _handle_generation_worker_finished(self, job_id: str) -> None:
        """Release only the runtime belonging to the finished worker."""

        worker = self._generation_workers.get(job_id)
        if worker is not None:
            worker.discard_unclaimed_outputs()
        managed_job = self._job_manager.get_job(job_id)
        if (
            job_id in self._cancelled_generation_job_ids
            and managed_job is not None
            and not managed_job.is_finished
        ):
            self._job_manager.mark_cancelled(job_id)
        elif managed_job is not None and not managed_job.is_finished:
            self._job_manager.fail_job(
                job_id,
                "The job ended before its result could be committed.",
            )
        self._generation_threads.pop(job_id, None)
        self._generation_workers.pop(job_id, None)
        self._generation_requests.pop(job_id, None)
        self._generation_surface_targets.pop(job_id, None)
        self._cancelled_generation_job_ids.discard(job_id)
        self._sync_controls()

    def _cancel_generation_job(self, job_id: str) -> bool:
        worker = self._generation_workers.get(str(job_id))
        thread = self._generation_threads.get(str(job_id))
        if worker is None or thread is None or worker.is_cancelled:
            return False
        self._cancelled_generation_job_ids.add(str(job_id))
        worker.cancel()
        if is_valid_qt_object(thread):
            thread.requestInterruption()
        return True

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
        if self._is_syncing_seekbar:
            return
        self.show_frame(frame_index)

    def _handle_mask_mode_changed(self, paint_checked: bool) -> None:
        mode = MASK_MODE_PAINT if paint_checked else MASK_MODE_ERASE
        self.video_view.set_brush_mode(mode)

    def _get_enabled_pbr_maps(self) -> tuple[str, ...]:
        """Snapshot the current Surface PBR contribution selection."""

        return tuple(
            map_type
            for map_type in PBR_MAP_TYPES
            if self.pbr_map_checkboxes[map_type].isChecked()
        )

    @Slot(bool)
    def _handle_pbr_map_toggled(self, _enabled: bool) -> None:
        """Apply Surface PBR changes without rebuilding the preview model."""

        self.surface_view.set_pbr_maps_enabled(
            self._get_enabled_pbr_maps()
        )
        self._sync_controls()

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

    def _restore_assignment_textures(self) -> None:
        signature_before = self.get_preview_dependency_signature()
        restore_succeeded = True
        self.surface_view.clear_surface_textures()
        for assignment in self._data.assignments:
            map_paths = self.get_assignment_map_asset_paths(
                assignment.assignment_id,
                assignment.selected_texture_resolution,
            )
            texture_path = map_paths.get(ATLAS_MAP_BASE_COLOR)
            if texture_path is None:
                continue
            try:
                self._set_surface_view_texture(
                    assignment.surface_ids,
                    texture_path,
                    {
                        map_type: map_path
                        for map_type, map_path in map_paths.items()
                        if map_type in PBR_MAP_TYPES
                    },
                )
            except (OSError, TypeError, ValueError):
                restore_succeeded = False
                continue
        signature_after = self.get_preview_dependency_signature()
        self._restored_assignment_texture_signature = (
            signature_after
            if restore_succeeded and signature_before == signature_after
            else None
        )

    def _set_surface_view_texture(
        self,
        surface_ids: Sequence[str],
        texture: object,
        map_textures: Mapping[str, object] | None = None,
    ) -> None:
        """Preserve the legacy two-argument viewer call for base-only assets."""

        if map_textures:
            self.surface_view.set_surface_texture(
                surface_ids,
                texture,  # type: ignore[arg-type]
                map_textures,
            )
            return
        self.surface_view.set_surface_texture(
            surface_ids,
            texture,  # type: ignore[arg-type]
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
            display_name = (
                f"{assignment.display_name} - "
                f"{variant.resolution} x {variant.resolution}"
                if assignment.display_name
                else f"{variant.resolution} x {variant.resolution}"
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
                    f"{_surface_texture_display_name(candidate)} - "
                    f"{resolution} x {resolution}"
                )
            else:
                if resolution is not None:
                    continue
                display_name = (
                    f"{_surface_texture_display_name(candidate)} - fixed image"
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

        if (
            self.is_generating and not self._generation_threads
        ) or not isinstance(raw_item, QListWidgetItem):
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
        if self._surface_targets_are_reserved(
            selected_surface_ids
        ) or self._assignment_is_reserved(assignment):
            self.status_label.setText(
                "A running job is using one of these surface textures."
            )
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
        if self._assignment_is_reserved(assignment):
            self.status_label.setText(
                "Wait for the job using this texture before deleting it."
            )
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

        return _persist_surface_texture_variants(
            self._asset_directory,
            surface_ids,
            assignment_id,
            texture_variants,
        )

    def _persist_texture_file(
        self,
        file_name: str,
        texture_png: bytes,
    ) -> str:
        return _persist_surface_texture_file(
            self._asset_directory,
            file_name,
            texture_png,
        )

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

    def _surface_targets_are_reserved(
        self,
        surface_ids: Sequence[str],
    ) -> bool:
        requested_ids = {str(surface_id) for surface_id in surface_ids}
        return bool(
            requested_ids
            and any(
                requested_ids.intersection(active_ids)
                for active_ids in self._generation_surface_targets.values()
            )
        )

    def _assignment_is_reserved(
        self,
        assignment: SurfaceTextureAssignment,
    ) -> bool:
        return self._surface_targets_are_reserved(assignment.surface_ids)

    def _sync_controls(self) -> None:
        has_video = self._video_source is not None
        has_mask = bool(self._data.frame_strokes) or self.video_view.has_selection()
        selected_surface_ids = self.surface_view.get_selected_surface_ids()
        has_surface = bool(selected_surface_ids)
        selection_is_reserved = self._surface_targets_are_reserved(
            selected_surface_ids
        )
        has_key = bool(self._settings.surface_texture_api_key)
        self.load_video_button.setEnabled(True)
        self.seekbar.setEnabled(has_video)
        self.paint_mask_button.setEnabled(has_video)
        self.erase_mask_button.setEnabled(has_video)
        self.brush_size_spinbox.setEnabled(has_video)
        self.clear_mask_button.setEnabled(
            has_video and self.video_view.has_selection()
        )
        self.material_notes_edit.setEnabled(True)
        self.job_name_edit.setEnabled(True)
        self.surface_texture_provider_combo.setEnabled(True)
        self.pbr_map_control.setEnabled(
            self._settings.surface_texture_provider == MESHY_PROVIDER
        )
        self.texture_view.setEnabled(True)
        self.other_texture_list.setEnabled(True)
        deletion_id = self._selected_texture_assignment_id_for_deletion()
        deletion_assignment = (
            None
            if deletion_id is None
            else self._assignment_by_id(deletion_id)
        )
        self.delete_texture_button.setEnabled(
            deletion_assignment is not None
            and not self._assignment_is_reserved(deletion_assignment)
        )
        self.generate_button.setEnabled(
            has_video
            and has_mask
            and has_surface
            and has_key
            and not selection_is_reserved
        )
        self.video_view.set_interaction_enabled(has_video)

    def _emit_data_changed(self) -> None:
        self.data_changed.emit(self._data.clone())

    def _close_video_source(self) -> None:
        if self._video_source is not None:
            self._video_source.close()
        self._video_source = None


# ### Worker helpers ###
class _SurfaceTextureCancelled(Exception):
    pass


def _raise_surface_worker_cancelled(
    cancel_event: threading.Event,
) -> None:
    if cancel_event.is_set():
        raise _SurfaceTextureCancelled


def _build_reference_pngs_from_input(
    reference_input: _SurfaceTextureReferenceInput,
    cancel_event: threading.Event,
) -> tuple[tuple[bytes, ...], tuple[int, ...]]:
    """Decode, crop, pack, and encode one immutable video snapshot."""

    strokes_by_frame = dict(reference_input.frame_strokes)
    limited_indices = _sample_frame_indices(
        tuple(strokes_by_frame),
        MAX_REFERENCE_FRAMES,
    )
    video_source = VideoFrameSource(reference_input.video_path)
    crops: list[np.ndarray] = []
    used_indices: list[int] = []
    try:
        for frame_index in limited_indices:
            _raise_surface_worker_cancelled(cancel_event)
            frame = video_source.get_frame(frame_index)
            crop = _build_masked_crop(
                frame,
                list(strokes_by_frame.get(frame_index, ())),
            )
            if crop.size == 0:
                continue
            crops.append(crop)
            used_indices.append(frame_index)
    finally:
        video_source.close()
    _raise_surface_worker_cancelled(cancel_event)
    packed_images = _pack_reference_crops(
        crops,
        MAX_PROVIDER_REFERENCE_IMAGES,
    )
    encoded_images: list[bytes] = []
    for image in packed_images:
        _raise_surface_worker_cancelled(cancel_event)
        encoded_images.append(_encode_provider_reference_png(image))
    return tuple(encoded_images), tuple(used_indices)


def _prepare_surface_texture_outputs(
    request: SurfaceTextureRequest,
    result: SurfaceTextureResult,
    cancel_event: threading.Event,
) -> tuple[_PreparedSurfaceTextureOutput, ...]:
    """Build every expensive PNG variant without touching Qt objects."""

    raw_outputs = _build_surface_texture_outputs(
        request,
        result.texture_png,
    )
    prepared_outputs: list[_PreparedSurfaceTextureOutput] = []
    for surface_ids, texture_png in raw_outputs:
        _raise_surface_worker_cancelled(cancel_event)
        prepared_outputs.append(
            _PreparedSurfaceTextureOutput(
                surface_ids=surface_ids,
                texture_variants=build_surface_texture_variants(
                    texture_png,
                    result.pbr_texture_pngs,
                ),
            )
        )
    if not prepared_outputs:
        raise ValueError("The provider returned no usable surface texture output.")
    return tuple(prepared_outputs)


def _persist_prepared_surface_texture_outputs(
    asset_directory: Path,
    prepared_outputs: Sequence[_PreparedSurfaceTextureOutput],
    cancel_event: threading.Event,
) -> tuple[_SavedSurfaceTextureOutput, ...]:
    """Persist complete variant families on the worker thread."""

    saved_outputs: list[_SavedSurfaceTextureOutput] = []
    try:
        for prepared_output in prepared_outputs:
            _raise_surface_worker_cancelled(cancel_event)
            saved_outputs.append(
                _persist_surface_texture_variants(
                    asset_directory,
                    prepared_output.surface_ids,
                    uuid.uuid4().hex,
                    prepared_output.texture_variants,
                    cancel_event,
                )
            )
    except Exception:
        _discard_surface_texture_outputs(asset_directory, saved_outputs)
        raise
    return tuple(saved_outputs)


def _persist_surface_texture_variants(
    asset_directory: Path,
    surface_ids: tuple[str, ...],
    assignment_id: str,
    texture_variants: SurfaceTextureVariants,
    cancel_event: threading.Event | None = None,
) -> _SavedSurfaceTextureOutput:
    """Persist one complete family or remove every partial file on failure."""

    persisted: list[SurfaceTextureVariant] = []
    png_items = tuple(
        (resolution, texture_variants.texture_png_by_resolution[resolution])
        for resolution in SURFACE_TEXTURE_RESOLUTIONS
    )
    map_png_by_resolution = {
        map_type: dict(resolution_map)
        for map_type, resolution_map in (
            texture_variants.map_png_by_resolution or {}
        ).items()
    }
    created_asset_paths: list[str] = []
    try:
        for resolution, texture_png in png_items:
            if cancel_event is not None:
                _raise_surface_worker_cancelled(cancel_event)
            file_name = f"{assignment_id}.texture-{resolution}.png"
            asset_path = _persist_surface_texture_file(
                asset_directory,
                file_name,
                texture_png,
            )
            created_asset_paths.append(asset_path)
            map_asset_paths = {ATLAS_MAP_BASE_COLOR: asset_path}
            for map_type in PBR_MAP_TYPES:
                resolution_map = map_png_by_resolution.get(map_type)
                if resolution_map is None:
                    continue
                if cancel_event is not None:
                    _raise_surface_worker_cancelled(cancel_event)
                map_asset_path = _persist_surface_texture_file(
                    asset_directory,
                    (
                        f"{assignment_id}.texture-{resolution}."
                        f"{map_type}.png"
                    ),
                    resolution_map[resolution],
                )
                created_asset_paths.append(map_asset_path)
                map_asset_paths[map_type] = map_asset_path
            persisted.append(
                SurfaceTextureVariant(
                    resolution=resolution,
                    asset_path=asset_path,
                    map_asset_paths=map_asset_paths,
                )
            )
    except Exception:
        _discard_surface_texture_asset_paths(
            asset_directory,
            created_asset_paths,
        )
        raise
    return _SavedSurfaceTextureOutput(
        surface_ids=surface_ids,
        assignment_id=assignment_id,
        variants=tuple(persisted),
        texture_png_by_resolution=png_items,
        map_png_by_resolution=map_png_by_resolution,
    )


def _build_aligned_pbr_asset_path(
    raw_asset_path: str,
    map_type: str,
) -> str:
    """Build a deterministic derivative path without changing source pixels."""

    source_path = PurePosixPath(str(raw_asset_path).replace("\\", "/"))
    suffix = source_path.suffix or ".png"
    aligned_name = (
        f"{source_path.stem}.housemaker-aligned-v"
        f"{SURFACE_PBR_ALIGNMENT_VERSION}-{map_type}{suffix}"
    )
    return (source_path.parent / aligned_name).as_posix()


def _persist_surface_texture_file(
    asset_directory: Path,
    file_name: str,
    texture_png: bytes,
) -> str:
    asset_directory.mkdir(parents=True, exist_ok=True)
    destination = asset_directory / file_name
    if destination.exists():
        raise FileExistsError(
            f"A surface texture asset already exists for {file_name}."
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{file_name}.",
        suffix=".tmp",
        dir=str(asset_directory),
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


def _replace_surface_texture_file(destination: Path, texture_png: bytes) -> None:
    """Atomically replace one existing derived surface texture PNG."""

    target = Path(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(bytes(texture_png))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _discard_surface_texture_outputs(
    asset_directory: Path,
    saved_outputs: Sequence[_SavedSurfaceTextureOutput],
) -> None:
    try:
        asset_root = asset_directory.resolve()
    except OSError:
        return
    for saved_output in saved_outputs:
        for variant in saved_output.variants:
            for raw_asset_path in variant.map_asset_paths.values():
                try:
                    asset_path = (
                        asset_directory / raw_asset_path
                    ).resolve()
                    asset_path.relative_to(asset_root)
                    asset_path.unlink(missing_ok=True)
                except (OSError, ValueError):
                    continue


def _discard_surface_texture_asset_paths(
    asset_directory: Path,
    raw_asset_paths: Sequence[str],
) -> None:
    """Best-effort cleanup for one partially persisted texture family."""

    try:
        asset_root = asset_directory.resolve()
    except OSError:
        return
    for raw_asset_path in raw_asset_paths:
        try:
            asset_path = (asset_directory / raw_asset_path).resolve()
            asset_path.relative_to(asset_root)
            asset_path.unlink(missing_ok=True)
        except (OSError, ValueError):
            continue


def _format_surface_provider_progress(message: str) -> str:
    """Map provider progress into the provider portion of the whole job."""

    normalized = str(message).strip() or "Generating texture"
    matches = tuple(_PROGRESS_PERCENT_PATTERN.finditer(normalized))
    if not matches:
        return normalized
    match = matches[-1]
    provider_percent = int(match.group(1))
    job_percent = 10 + round(provider_percent * 0.75)
    return (
        normalized[: match.start(1)]
        + str(job_percent)
        + normalized[match.end(1) :]
    )


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
    columns = max(1, math.ceil(math.sqrt(len(crops))))
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


def _prepare_provider_reference_image(image: np.ndarray) -> np.ndarray:
    """Bound one reference while retaining its channels and aspect ratio."""

    source = np.asarray(image)
    if source.ndim not in {2, 3} or source.shape[0] <= 0 or source.shape[1] <= 0:
        raise ValueError("A painted video reference has invalid dimensions.")
    height, width = source.shape[:2]
    longest_edge = max(height, width)
    if longest_edge <= MAX_PROVIDER_REFERENCE_EDGE_PIXELS:
        return np.ascontiguousarray(source)

    scale = MAX_PROVIDER_REFERENCE_EDGE_PIXELS / longest_edge
    target_width = max(1, round(width * scale))
    target_height = max(1, round(height * scale))
    resized = cv2.resize(
        source,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )
    return np.ascontiguousarray(resized)


def _encode_provider_reference_png(image: np.ndarray) -> bytes:
    """Normalize and encode one image sent to the texture provider."""

    return _encode_png(_prepare_provider_reference_image(image))


def _encode_png(image: np.ndarray) -> bytes:
    did_encode, encoded = cv2.imencode(".png", np.asarray(image))
    if not did_encode:
        raise ValueError("Unable to encode a painted video reference.")
    return bytes(encoded)


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


def _build_surface_texture_outputs(
    request: SurfaceTextureRequest,
    generated_texture_png: bytes,
) -> list[tuple[tuple[str, ...], bytes]]:
    _decode_png_rgba(generated_texture_png, "Generated texture")
    return [(request.surface_ids, bytes(generated_texture_png))]


# ### Text helpers ###
def _default_surface_texture_name(surface_type: str) -> str:
    return f"{str(surface_type).strip().title() or 'Surface'} texture"


def _surface_texture_display_name(
    assignment: SurfaceTextureAssignment,
) -> str:
    return (
        assignment.display_name
        or _default_surface_texture_name(assignment.surface_type)
    )


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
