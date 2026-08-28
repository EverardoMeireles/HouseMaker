# ### Imports ###
from __future__ import annotations

import copy
import hashlib
import math
import os
import re
import tempfile
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from PySide6.QtCore import (
    QObject,
    QStandardPaths,
    QThread,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtWidgets import (
    QBoxLayout,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
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

from housemaker.uv_integrity import (
    UV_FINGERPRINT_VERSION,
    UvFingerprint,
    UvIntegrityError,
    build_uv_fingerprint,
)
from housemaker.generation_state import (
    MASK_MODE_ERASE,
    MASK_MODE_PAINT,
    GeneratedObjectRecord,
    GeneratedObjectPlacement,
    GenerationData,
)
from housemaker.generation_jobs import GenerationJobManager
from housemaker.generation_views import VideoInpaintView
from housemaker.glb import GeneratedModel, import_generated_glb
from housemaker.meshy_generation import (
    MeshyGenerationResult,
    request_image_to_3d_model,
    request_retextured_model,
)
from housemaker.object_texture_variants import (
    DEFAULT_TEXTURE_RESOLUTION,
    TEXTURE_RESOLUTION_1024,
    TEXTURE_RESOLUTION_2048,
    TEXTURE_RESOLUTIONS,
    ObjectTextureVariants,
    build_object_texture_variants,
    replace_object_base_color_texture_from_glb,
)
from housemaker.object_symmetry import (
    AUTOMATIC_SYMMETRIC_DIVISION_METADATA_VERSION,
    LEGACY_SYMMETRIC_PAIR_METADATA_VERSION,
    SYMMETRIC_DIVISION_METADATA_VERSION,
    SYMMETRIC_DIVISION_ORIENTATIONS,
    SYMMETRIC_DIVISION_ORIENTATION_HORIZONTAL,
    SYMMETRIC_DIVISION_ORIENTATION_VERTICAL,
    SYMMETRIC_DIVISION_SIDE_ORDER_BY_ORIENTATION,
    SYMMETRIC_DIVISION_SIDES_BY_ORIENTATION,
    SYMMETRIC_QUARTER_METADATA_VERSION,
    SYMMETRIC_SQUARE_PAIR_CONTENT_RESOLUTIONS,
    SymmetricDivisionMetadata,
    SymmetricDivisionResult,
    SymmetricPairTextureVariants,
    SymmetricQuarterTextureVariants,
    SymmetricSquarePairTextureVariants,
    build_automatic_symmetric_object_variants,
    build_symmetric_half_texture_variants,
    build_symmetric_pair_texture_variants,
    build_symmetric_quarter_texture_variants,
    build_symmetric_retexture_proxy_glb,
    build_symmetric_square_pair_texture_variants,
)
from housemaker.settings_widget import (
    DEFAULT_MESHY_TARGET_POLYCOUNT,
    MESHY_SMART_TOPOLOGY_MAX_TARGET_POLYCOUNT,
    MESHY_SMART_TOPOLOGY_MIN_TARGET_POLYCOUNT,
    GenerationServiceSettings,
)
from housemaker.texture_atlas_view import (
    TextureAtlasEntry,
    TextureAtlasView,
    UvTriangle,
)
from housemaker.unused_face_removal import (
    ALL_CAMERA_IDS,
    CAMERA_OPTIONS,
    UncheckedCameraFacePurgeOptions,
    UncheckedCameraFacePurgeResult,
    UnusedFaceRemovalOptions,
    UnusedFaceRemovalProgress,
    purge_faces_visible_from_unchecked_cameras_from_glb,
    remove_unused_faces_from_glb,
)
from housemaker.video_source import (
    VIDEO_FILE_FILTER,
    VideoFrameSource,
    probe_video,
)
from housemaker.viewer import GlbViewerWidget


# ### Constants ###
MIN_BRUSH_RADIUS_PIXELS = 2
MAX_BRUSH_RADIUS_PIXELS = 160
DEFAULT_BRUSH_RADIUS_PIXELS = 24
AMBIENT_LIGHT_PERCENT_SCALE = 100
OBJECT_GENERATION_DEFAULT_AMBIENT_LIGHT_INTENSITY = 1.0
VIEW_STRETCH = 1
CONTROL_STRETCH = 0
INTERRUPT_POLL_SECONDS = 0.01
SHUTDOWN_WAIT_MILLISECONDS = 250
GENERATION_BACKEND_MESHY = "meshy"
OBJECT_ID_ITEM_ROLE = Qt.ItemDataRole.UserRole
OBJECT_LIST_MAXIMUM_HEIGHT = 124
OBJECT_DETAILS_EXTERNAL_MINIMUM_WIDTH = 320
OBJECT_DETAILS_EXTERNAL_MAXIMUM_WIDTH = 440
QT_WIDGET_MAXIMUM_SIZE = 16_777_215
MESHY_REVISION_GEOMETRY = "geometry"
MESHY_REVISION_POSTPROCESSED = "postprocessed"
MESHY_REVISION_NAMES = frozenset(
    {MESHY_REVISION_GEOMETRY, MESHY_REVISION_POSTPROCESSED}
)
MESHY_REVISION_ASSET_PIPELINE_KEYS = (
    "source_asset_path",
    "postprocessed_asset_path",
)
TEXTURE_VARIANTS_PIPELINE_KEY = "texture_variants"
SELECTED_TEXTURE_RESOLUTION_PIPELINE_KEY = "selected_texture_resolution"
TEXTURE_VARIANT_GLB_PATH_KEY = "glb_asset_path"
TEXTURE_VARIANT_PNG_PATH_KEY = "texture_asset_path"
# Legacy-only key retained so later geometry operations can discard obsolete
# masks from projects saved before Object Generation inpainting was removed.
TEXTURE_INPAINT_STROKES_PIPELINE_KEY = "texture_inpaint_strokes"
OBJECT_OPERATION_UNDO_STACK_PIPELINE_KEY = "object_operation_undo_stack"
MAX_OBJECT_OPERATION_UNDO_COUNT = 10
OBJECT_OPERATION_GENERATE_MODEL = "generate_model"
OBJECT_OPERATION_GENERATE_TEXTURE = "generate_texture"
OBJECT_OPERATION_PURGE_FACES = "purge_faces"
GENERATION_JOB_KIND_MODEL = "Object generation"
GENERATION_JOB_KIND_TEXTURE = "Object texture generation"
GENERATION_JOB_KIND_FACE_PURGE = "Object face purge"
SYMMETRIC_DIVISION_PIPELINE_KEY = "symmetric_division"
SYMMETRIC_DIVISION_TEXTURE_CONTENT_HALF = "left"
SYMMETRIC_TEXTURE_RESOLUTIONS = SYMMETRIC_SQUARE_PAIR_CONTENT_RESOLUTIONS
SUPPORTED_SYMMETRIC_METADATA_VERSIONS = frozenset(
    {
        SYMMETRIC_DIVISION_METADATA_VERSION,
        SYMMETRIC_QUARTER_METADATA_VERSION,
        LEGACY_SYMMETRIC_PAIR_METADATA_VERSION,
        AUTOMATIC_SYMMETRIC_DIVISION_METADATA_VERSION,
    }
)
COMPACT_SYMMETRIC_METADATA_VERSIONS = (
    SUPPORTED_SYMMETRIC_METADATA_VERSIONS
    - {SYMMETRIC_DIVISION_METADATA_VERSION}
)
LEFT_HALF_SYMMETRIC_METADATA_VERSIONS = (
    SUPPORTED_SYMMETRIC_METADATA_VERSIONS
    - {SYMMETRIC_QUARTER_METADATA_VERSION}
)
_PROGRESS_PERCENT_PATTERN = re.compile(r"(?<!\d)(100|[1-9]?\d)\s*%")
_OBJECT_PROVIDER_PROGRESS_START = 5
_OBJECT_PROVIDER_PROGRESS_END = 80
_STAGED_GEOMETRY_PROGRESS_END = 48
_STAGED_TEXTURE_PROGRESS_START = 56
_TEXTURE_PROVIDER_PROGRESS_START = 5
_TEXTURE_PROVIDER_PROGRESS_END = 80


# ### Active operation state ###
@dataclass
class _ActiveObjectOperation:
    """Track one cancellable transaction until its worker fully exits."""

    kind: str
    target_object_id: str | None = None
    cancel_requested: bool = False
    committed_object_id: str | None = None
    pending_placement: GeneratedObjectPlacement | None = None
    _operation_id: str = field(
        default_factory=lambda: uuid.uuid4().hex,
        init=False,
        repr=False,
    )

    @property
    def operation_id(self) -> str:
        """Return the immutable identity used by modeless placement UI."""

        return self._operation_id


@dataclass(frozen=True)
class _ExistingObjectPlacementRequest:
    """Bind one modeless placement picker to one completed object."""

    request_id: str
    object_id: str


class _ObjectJobSignalRelay(QObject):
    """Tag worker events on a GUI-affine QObject before workspace dispatch."""

    pair_succeeded = Signal(str, object, object)
    single_succeeded = Signal(str, object)
    failed = Signal(str, str)
    progress = Signal(str, str)
    thread_finished = Signal(str)

    def __init__(self, operation_id: str, parent: QObject) -> None:
        super().__init__(parent)
        self._operation_id = str(operation_id)

    @Slot(object, object)
    def forward_pair_succeeded(self, first: object, second: object) -> None:
        self.pair_succeeded.emit(self._operation_id, first, second)

    @Slot(object)
    def forward_single_succeeded(self, outcome: object) -> None:
        self.single_succeeded.emit(self._operation_id, outcome)

    @Slot(str)
    def forward_failed(self, message: str) -> None:
        self.failed.emit(self._operation_id, str(message))

    @Slot(str)
    def forward_progress(self, message: str) -> None:
        self.progress.emit(self._operation_id, str(message))

    @Slot()
    def forward_thread_finished(self) -> None:
        self.thread_finished.emit(self._operation_id)


@dataclass
class _ObjectJobRuntime:
    """Own one worker transaction independently of every other job."""

    operation: _ActiveObjectOperation
    thread: QThread
    worker: (
        GenerationWorker
        | TextureRegenerationWorker
        | UncheckedCameraFacePurgeWorker
    )
    relay: _ObjectJobSignalRelay
    generation_request: GenerationRequest | None = None
    requested_name: str = ""
    managed_job_id: str | None = None

    @property
    def operation_id(self) -> str:
        return self.operation.operation_id


# ### Symmetric-division metadata ###
@dataclass(frozen=True)
class ObjectSymmetricDivisionMetadata:
    """Validated immutable provenance for one divided generated object."""

    version: int
    orientation: str
    kept_side: str
    plane_coordinate: float
    texture_content_half: str | None = None
    packing_mode: str | None = None
    texture_content_quadrant: str | None = None
    selection_mode: str | None = None
    triangle_count_by_side: tuple[tuple[str, int], ...] = ()
    tie_broken_randomly: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.version, bool)
            or self.version not in SUPPORTED_SYMMETRIC_METADATA_VERSIONS
        ):
            raise ValueError("Unsupported symmetric-division metadata version.")
        if self.orientation not in SYMMETRIC_DIVISION_ORIENTATIONS:
            raise ValueError("Unknown symmetric-division orientation.")
        if self.kept_side not in SYMMETRIC_DIVISION_SIDES_BY_ORIENTATION[
            self.orientation
        ]:
            raise ValueError("The kept side does not match the orientation.")
        if not math.isfinite(self.plane_coordinate):
            raise ValueError("The symmetric-division plane must be finite.")
        if self.version == SYMMETRIC_DIVISION_METADATA_VERSION:
            if (
                self.texture_content_half
                != SYMMETRIC_DIVISION_TEXTURE_CONTENT_HALF
            ):
                raise ValueError("Unknown symmetric-division texture half.")
            if (
                self.packing_mode is not None
                or self.texture_content_quadrant is not None
                or self.selection_mode is not None
                or self.triangle_count_by_side
                or self.tie_broken_randomly
            ):
                raise ValueError(
                    "Legacy symmetric-division metadata has automatic fields."
                )
            return
        SymmetricDivisionMetadata(
            version=self.version,
            orientation=self.orientation,
            kept_side=self.kept_side,
            plane_coordinate=self.plane_coordinate,
            packing_mode=self.packing_mode,
            texture_content_quadrant=self.texture_content_quadrant,
            texture_content_half=self.texture_content_half,
            selection_mode=self.selection_mode,
            triangle_count_by_side=self.triangle_count_by_side,
            tie_broken_randomly=self.tie_broken_randomly,
        )

    def to_pipeline_dict(self) -> dict[str, object]:
        """Return JSON-safe provenance stored with the generated object."""

        pipeline: dict[str, object] = {
            "version": self.version,
            "orientation": self.orientation,
            "kept_side": self.kept_side,
            "plane_coordinate": self.plane_coordinate,
        }
        if self.version == SYMMETRIC_DIVISION_METADATA_VERSION:
            pipeline["texture_content_half"] = self.texture_content_half
        else:
            pipeline["packing_mode"] = self.packing_mode
            if self.version == SYMMETRIC_QUARTER_METADATA_VERSION:
                pipeline["texture_content_quadrant"] = (
                    self.texture_content_quadrant
                )
            else:
                pipeline["texture_content_half"] = (
                    self.texture_content_half
                )
            pipeline.update(
                {
                    "selection_mode": self.selection_mode,
                    "triangle_count_by_side": dict(
                        self.triangle_count_by_side
                    ),
                    "tie_broken_randomly": self.tie_broken_randomly,
                }
            )
        return pipeline


# ### Object-packing transaction types ###
PersistableObjectTextureVariants = (
    ObjectTextureVariants
    | SymmetricQuarterTextureVariants
    | SymmetricPairTextureVariants
    | SymmetricSquarePairTextureVariants
)
ObjectPackingCommit = Callable[[], bool]
ObjectPackingChangeHandler = Callable[
    [
        GeneratedObjectRecord,
        GeneratedObjectRecord,
        GeneratedModel,
        ObjectPackingCommit,
    ],
    bool,
]


# ### Generation interfaces ###
class MeshyPlanner(Protocol):
    def plan(self, request: "GenerationRequest") -> MeshyGenerationResult:
        """Return a completed Meshy Image-to-3D task result."""


class MeshyExecutor(Protocol):
    def execute(self, result: MeshyGenerationResult) -> GeneratedModel:
        """Import one downloaded Meshy GLB for the result viewer."""


class TextureRegenerator(Protocol):
    def regenerate(
        self,
        request: "TextureRegenerationRequest",
    ) -> MeshyGenerationResult:
        """Return a newly textured version of one existing generated model."""


class GenerationRequest:
    """Owned selected-object input passed to Meshy Image-to-3D."""

    def __init__(
        self,
        *,
        frame_index: int,
        selected_object_bgra: np.ndarray,
        settings: GenerationServiceSettings,
        enabled_camera_ids: Sequence[str] = ALL_CAMERA_IDS,
        geometry_only: bool = False,
        symmetric_division_enabled: bool = False,
        symmetric_division_orientation: str = (
            SYMMETRIC_DIVISION_ORIENTATION_VERTICAL
        ),
    ) -> None:
        self.frame_index = int(frame_index)
        self.selected_object_bgra = np.ascontiguousarray(
            selected_object_bgra
        ).copy()
        self.settings = settings
        self.geometry_only = bool(geometry_only)
        self.symmetric_division_enabled = bool(symmetric_division_enabled)
        normalized_orientation = str(
            symmetric_division_orientation
        ).strip().lower()
        if normalized_orientation not in SYMMETRIC_DIVISION_ORIENTATIONS:
            raise ValueError("Unknown generation-time symmetric orientation.")
        if self.geometry_only and self.symmetric_division_enabled:
            raise ValueError(
                "Geometry-only generation cannot apply symmetric division."
            )
        self.symmetric_division_orientation = normalized_orientation
        requested_camera_ids = tuple(str(value) for value in enabled_camera_ids)
        self.enabled_camera_ids = tuple(
            camera_id
            for camera_id in ALL_CAMERA_IDS
            if camera_id in requested_camera_ids
        )


@dataclass(frozen=True)
class TextureRegenerationRequest:
    """Owned model and reference image for one Meshy Retexture task."""

    object_id: str
    reference_frame_index: int
    reference_image_bgra: np.ndarray
    model_glb: bytes
    settings: GenerationServiceSettings
    enable_original_uv: bool = False
    submitted_uv_fingerprint: UvFingerprint | None = None
    preserve_symmetric_uvs: bool = False

    def __post_init__(self) -> None:
        normalized_object_id = str(self.object_id).strip()
        if not normalized_object_id:
            raise ValueError("Texture regeneration requires an object ID.")
        reference_image = np.asarray(self.reference_image_bgra)
        if (
            reference_image.ndim != 3
            or reference_image.shape[2] != 4
            or reference_image.size == 0
        ):
            raise ValueError(
                "Texture regeneration requires a non-empty BGRA reference image."
            )
        model_glb = bytes(self.model_glb)
        if not model_glb:
            raise ValueError("Texture regeneration requires a model GLB.")
        if self.enable_original_uv and self.submitted_uv_fingerprint is None:
            raise ValueError(
                "Original-UV texture regeneration requires a UV fingerprint."
            )
        if not self.enable_original_uv and self.submitted_uv_fingerprint is not None:
            raise ValueError(
                "A UV fingerprint is only valid when original UVs are preserved."
            )
        if self.preserve_symmetric_uvs and not self.enable_original_uv:
            raise ValueError(
                "Symmetric texture regeneration must preserve original UVs."
            )
        if self.enable_original_uv and not self.preserve_symmetric_uvs:
            raise ValueError(
                "Original-UV texture regeneration is only supported for "
                "symmetric objects."
            )
        object.__setattr__(self, "object_id", normalized_object_id)
        object.__setattr__(
            self,
            "reference_frame_index",
            int(self.reference_frame_index),
        )
        object.__setattr__(
            self,
            "reference_image_bgra",
            np.ascontiguousarray(reference_image).copy(),
        )
        object.__setattr__(self, "model_glb", model_glb)
        object.__setattr__(
            self,
            "enable_original_uv",
            bool(self.enable_original_uv),
        )
        object.__setattr__(
            self,
            "preserve_symmetric_uvs",
            bool(self.preserve_symmetric_uvs),
        )


@dataclass(frozen=True)
class _TextureRegenerationPreflight:
    """Lightweight GUI snapshot materialized into a request by the worker."""

    object_id: str
    reference_frame_index: int
    reference_image_bgra: np.ndarray
    source_asset_path: str
    source_asset_revision: tuple[object, ...]
    settings: GenerationServiceSettings
    enable_original_uv: bool = False
    preserve_symmetric_uvs: bool = False

    def __post_init__(self) -> None:
        normalized_object_id = str(self.object_id).strip()
        if not normalized_object_id:
            raise ValueError("Texture regeneration requires an object ID.")
        reference_image = np.asarray(self.reference_image_bgra)
        if (
            reference_image.ndim != 3
            or reference_image.shape[2] != 4
            or reference_image.size == 0
        ):
            raise ValueError(
                "Texture regeneration requires a non-empty BGRA reference image."
            )
        source_asset_path = str(self.source_asset_path).strip()
        source_path = Path(source_asset_path)
        if (
            not source_asset_path
            or source_path.is_absolute()
            or ".." in source_path.parts
            or source_path.suffix.lower() != ".glb"
        ):
            raise ValueError("The texture source asset path is unsafe.")
        if len(self.source_asset_revision) != 4:
            raise ValueError("The texture source asset revision is invalid.")
        if self.preserve_symmetric_uvs and not self.enable_original_uv:
            raise ValueError(
                "Symmetric texture regeneration must preserve original UVs."
            )
        if self.enable_original_uv and not self.preserve_symmetric_uvs:
            raise ValueError(
                "Original-UV texture regeneration is only supported for "
                "symmetric objects."
            )
        object.__setattr__(self, "object_id", normalized_object_id)
        object.__setattr__(
            self,
            "reference_frame_index",
            int(self.reference_frame_index),
        )
        object.__setattr__(
            self,
            "reference_image_bgra",
            np.ascontiguousarray(reference_image).copy(),
        )
        object.__setattr__(self, "source_asset_path", source_asset_path)
        object.__setattr__(
            self,
            "source_asset_revision",
            tuple(self.source_asset_revision),
        )
        object.__setattr__(
            self,
            "enable_original_uv",
            bool(self.enable_original_uv),
        )
        object.__setattr__(
            self,
            "preserve_symmetric_uvs",
            bool(self.preserve_symmetric_uvs),
        )


@dataclass(frozen=True)
class _MaterializedTextureRegeneration:
    """Worker-owned paid request plus its stable source revision."""

    request: TextureRegenerationRequest
    source_asset_path: str | None = None
    source_asset_revision: tuple[object, ...] | None = None


@dataclass(frozen=True)
class TextureRegenerationOutcome:
    """Provider result plus the immutable request and verified final UVs."""

    request: TextureRegenerationRequest
    result: MeshyGenerationResult
    final_uv_fingerprint: UvFingerprint | None = None


@dataclass(frozen=True)
class _SavedObjectGeneration:
    """Fully prepared model assets awaiting a short GUI-thread commit."""

    result: MeshyGenerationResult
    object_id: str
    pipeline: dict[str, object]
    asset_path: str
    preview_model: GeneratedModel
    preview_asset_revision: tuple[object, ...]
    symmetry: ObjectSymmetricDivisionMetadata | None
    persisted_asset_paths: tuple[str, ...]


@dataclass(frozen=True)
class _SavedObjectTextureRegeneration:
    """Fully prepared texture assets awaiting target and Atlas validation."""

    outcome: TextureRegenerationOutcome
    variant_metadata: dict[str, dict[str, str]]
    selected_resolution: int
    base_asset_path: str
    base_provider_task_id: str
    base_pipeline: dict[str, object]
    source_asset_path: str | None
    source_asset_revision: tuple[object, ...] | None
    next_pipeline: dict[str, object]
    preview_model: GeneratedModel
    preview_asset_revision: tuple[object, ...]
    persisted_asset_paths: tuple[str, ...]


@dataclass(frozen=True)
class UncheckedCameraFacePurgeRequest:
    """Owned selected-object input for one local face purge."""

    object_id: str
    model_glb: bytes
    unchecked_camera_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object_id = str(self.object_id).strip()
        if not object_id:
            raise ValueError("Face purge requires an object ID.")
        model_glb = bytes(self.model_glb)
        if not model_glb:
            raise ValueError("Face purge requires a model GLB.")
        requested_ids = tuple(str(value) for value in self.unchecked_camera_ids)
        unchecked_camera_ids = tuple(
            camera_id
            for camera_id in ALL_CAMERA_IDS
            if camera_id in requested_ids
        )
        if not unchecked_camera_ids:
            raise ValueError("Uncheck at least one camera before purging faces.")
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "model_glb", model_glb)
        object.__setattr__(
            self,
            "unchecked_camera_ids",
            unchecked_camera_ids,
        )


@dataclass(frozen=True)
class UncheckedCameraFacePurgeOutcome:
    """Immutable request paired with its filtered geometry result."""

    request: UncheckedCameraFacePurgeRequest
    result: UncheckedCameraFacePurgeResult


@dataclass(frozen=True)
class StagedMeshyGenerationResult(MeshyGenerationResult):
    """Final textured result plus auditable geometry-processing revisions."""

    geometry_task_id: str = ""
    source_glb_bytes: bytes = b""
    postprocessed_glb_bytes: bytes = b""
    original_face_count: int = 0
    retained_face_count: int = 0
    removed_face_count: int = 0
    protected_face_count: int = 0
    enabled_camera_ids: tuple[str, ...] = ()
    unchecked_camera_ids: tuple[str, ...] = ()
    camera_face_purge_applied: bool = False
    purge_original_face_count: int = 0
    purge_retained_face_count: int = 0
    purge_removed_face_count: int = 0
    unused_face_removal_applied: bool = False
    geometry_only: bool = False


@dataclass(frozen=True)
class ActiveObjectTextureVariant:
    """Safe resolved texture choice exposed to the project atlas workspace."""

    object_id: str
    object_name: str
    resolution: int
    glb_asset_relative_path: str
    texture_asset_relative_path: str
    glb_asset_path: Path
    texture_asset_path: Path


@dataclass(frozen=True)
class ObjectTextureImageVariant:
    """One safe exact-resolution PNG, independent of its selectable GLB."""

    object_id: str
    object_name: str
    resolution: int
    texture_asset_relative_path: str
    texture_asset_path: Path


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

        _raise_if_generation_cancelled(cancel_event)
        image_png = _encode_png(request.selected_object_bgra)

        def report_generation_progress(status: str, progress: int) -> None:
            if progress_callback is None:
                return
            if status == "PENDING":
                progress_callback("Meshy task queued...")
            elif status == "IN_PROGRESS":
                progress_callback(f"Meshy is generating: {progress}%")
            elif status == "SUCCEEDED":
                progress_callback("Meshy generation complete. Downloading GLB...")

        use_unused_face_removal = request.settings.unused_face_removal
        unchecked_camera_ids = tuple(
            camera_id
            for camera_id in ALL_CAMERA_IDS
            if camera_id not in request.enabled_camera_ids
        )
        use_camera_face_purge = bool(unchecked_camera_ids)
        if (
            not request.geometry_only
            and not use_camera_face_purge
            and not use_unused_face_removal
        ):
            _raise_if_generation_cancelled(cancel_event)
            return request_image_to_3d_model(
                api_key=request.settings.meshy_api_key,
                image_png=image_png,
                target_polycount=request.settings.meshy_target_polycount,
                progress_callback=report_generation_progress,
                cancel_event=cancel_event,
            )

        if use_unused_face_removal and not request.enabled_camera_ids:
            raise ValueError(
                "Select at least one post-processing camera before generating."
            )
        if progress_callback is not None:
            progress_callback("Submitting geometry-only Meshy task...")
        _raise_if_generation_cancelled(cancel_event)
        geometry_result = request_image_to_3d_model(
            api_key=request.settings.meshy_api_key,
            image_png=image_png,
            target_polycount=request.settings.meshy_target_polycount,
            progress_callback=report_generation_progress,
            cancel_event=cancel_event,
            should_texture=False,
        )

        processed_glb_bytes = geometry_result.glb_bytes
        purge_original_face_count = 0
        purge_retained_face_count = 0
        purge_removed_face_count = 0
        if use_camera_face_purge:
            def report_face_purge(
                update: UnusedFaceRemovalProgress,
            ) -> None:
                if progress_callback is None:
                    return
                if update.stage == "capturing":
                    camera_suffix = (
                        ""
                        if update.camera_id is None
                        else f" ({update.camera_id})"
                    )
                    progress_callback(
                        "Capturing unchecked-camera views"
                        + camera_suffix
                        + "..."
                    )
                elif update.stage == "exporting":
                    progress_callback("Saving camera-purged geometry...")

            purged = purge_faces_visible_from_unchecked_cameras_from_glb(
                processed_glb_bytes,
                unchecked_camera_ids=unchecked_camera_ids,
                options=UncheckedCameraFacePurgeOptions(),
                cancel_requested=(
                    None if cancel_event is None else cancel_event.is_set
                ),
                progress_callback=report_face_purge,
            )
            processed_glb_bytes = purged.glb_bytes
            purge_original_face_count = purged.original_face_count
            purge_retained_face_count = purged.retained_face_count
            purge_removed_face_count = purged.removed_face_count

        original_face_count = 0
        retained_face_count = 0
        removed_face_count = 0
        protected_face_count = 0
        if use_unused_face_removal:
            def report_face_removal(
                update: UnusedFaceRemovalProgress,
            ) -> None:
                if progress_callback is None:
                    return
                if update.stage == "capturing":
                    camera_suffix = (
                        ""
                        if update.camera_id is None
                        else f" ({update.camera_id})"
                    )
                    progress_callback(
                        "Capturing unused-face views" + camera_suffix + "..."
                    )
                elif update.stage == "checking":
                    progress_callback(
                        "Checking faces: "
                        f"{update.completed_face_count}/"
                        f"{update.total_face_count}"
                    )
                elif update.stage == "exporting":
                    progress_callback("Saving visible geometry...")

            removed = remove_unused_faces_from_glb(
                processed_glb_bytes,
                options=UnusedFaceRemovalOptions(
                    enabled_camera_ids=request.enabled_camera_ids,
                ),
                cancel_requested=(
                    None if cancel_event is None else cancel_event.is_set
                ),
                progress_callback=report_face_removal,
            )
            processed_glb_bytes = removed.glb_bytes
            original_face_count = removed.original_face_count
            retained_face_count = removed.retained_face_count
            removed_face_count = removed.removed_face_count
            protected_face_count = removed.protected_face_count

        if request.geometry_only:
            return StagedMeshyGenerationResult(
                task_id=geometry_result.task_id,
                glb_bytes=processed_glb_bytes,
                name=geometry_result.name,
                geometry_task_id=geometry_result.task_id,
                source_glb_bytes=geometry_result.glb_bytes,
                postprocessed_glb_bytes=processed_glb_bytes,
                original_face_count=original_face_count,
                retained_face_count=retained_face_count,
                removed_face_count=removed_face_count,
                protected_face_count=protected_face_count,
                enabled_camera_ids=request.enabled_camera_ids,
                unchecked_camera_ids=unchecked_camera_ids,
                camera_face_purge_applied=use_camera_face_purge,
                purge_original_face_count=purge_original_face_count,
                purge_retained_face_count=purge_retained_face_count,
                purge_removed_face_count=purge_removed_face_count,
                unused_face_removal_applied=use_unused_face_removal,
                geometry_only=True,
            )

        def report_texture_progress(status: str, progress: int) -> None:
            if progress_callback is None:
                return
            if status == "PENDING":
                progress_callback("Meshy texture task queued...")
            elif status == "IN_PROGRESS":
                progress_callback(f"Meshy is texturing: {progress}%")
            elif status == "SUCCEEDED":
                progress_callback("Meshy texturing complete. Downloading GLB...")

        _raise_if_generation_cancelled(cancel_event)
        if progress_callback is not None:
            progress_callback("Submitting Meshy texture task...")
        textured_result = request_retextured_model(
            api_key=request.settings.meshy_api_key,
            model_glb=processed_glb_bytes,
            reference_images_png=(image_png,),
            enable_original_uv=False,
            progress_callback=report_texture_progress,
            cancel_event=cancel_event,
        )
        return StagedMeshyGenerationResult(
            task_id=textured_result.task_id,
            glb_bytes=textured_result.glb_bytes,
            name=textured_result.name,
            geometry_task_id=geometry_result.task_id,
            source_glb_bytes=geometry_result.glb_bytes,
            postprocessed_glb_bytes=processed_glb_bytes,
            original_face_count=original_face_count,
            retained_face_count=retained_face_count,
            removed_face_count=removed_face_count,
            protected_face_count=protected_face_count,
            enabled_camera_ids=request.enabled_camera_ids,
            unchecked_camera_ids=unchecked_camera_ids,
            camera_face_purge_applied=use_camera_face_purge,
            purge_original_face_count=purge_original_face_count,
            purge_retained_face_count=purge_retained_face_count,
            purge_removed_face_count=purge_removed_face_count,
            unused_face_removal_applied=use_unused_face_removal,
        )


class MeshyTextureRegenerator:
    """Submit an existing processed model to Meshy Retexture."""

    def regenerate(
        self,
        request: TextureRegenerationRequest,
        progress_callback: Callable[[str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> MeshyGenerationResult:
        if not request.settings.meshy_api_key:
            raise ValueError("Set a Meshy AI API key before generating texture.")

        reference_png = _encode_png(request.reference_image_bgra)

        def report_texture_progress(status: str, progress: int) -> None:
            if progress_callback is None:
                return
            if status == "PENDING":
                progress_callback("Meshy texture task queued...")
            elif status == "IN_PROGRESS":
                progress_callback(f"Meshy is generating texture: {progress}%")
            elif status == "SUCCEEDED":
                progress_callback(
                    "Meshy texture generation complete. Downloading GLB..."
                )

        _raise_if_generation_cancelled(cancel_event)
        if progress_callback is not None:
            progress_callback("Submitting Meshy texture generation task...")
        _raise_if_generation_cancelled(cancel_event)
        return request_retextured_model(
            api_key=request.settings.meshy_api_key,
            model_glb=request.model_glb,
            reference_images_png=(reference_png,),
            enable_original_uv=request.enable_original_uv,
            progress_callback=report_texture_progress,
            cancel_event=cancel_event,
        )


class MeshyModelExecutor:
    """Validate and adapt one downloaded Meshy GLB for the HouseMaker viewer."""

    def execute(self, result: object) -> GeneratedModel:
        if not isinstance(result, MeshyGenerationResult):
            raise TypeError("Meshy returned an invalid generation result.")
        model = import_generated_glb(result.glb_bytes)
        if not (
            isinstance(result, StagedMeshyGenerationResult)
            and result.geometry_only
        ):
            model.object_texture_variants = build_object_texture_variants(
                result.glb_bytes
            )
        return model


# ### Object viewer panel ###
class ObjectGenerationViewerPanel(QWidget):
    """Keep the generated-object selector with its detachable 3D viewer."""

    camera_selection_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._is_external_presentation_active = False
        self._layout = QBoxLayout(QBoxLayout.Direction.TopToBottom, self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)

        self.viewer = GlbViewerWidget(wireframe_enabled=False)
        self._layout.addWidget(self.viewer, 1)

        self.details_panel = QWidget()
        self.details_panel.setObjectName("object_generation_details_panel")
        details_layout = QVBoxLayout(self.details_panel)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(4)

        self.unused_face_camera_controls = QWidget()
        self.unused_face_camera_controls.setObjectName(
            "unused_face_camera_controls"
        )
        camera_layout = QGridLayout(self.unused_face_camera_controls)
        camera_layout.setContentsMargins(4, 2, 4, 2)
        camera_layout.setSpacing(8)
        self.postprocess_camera_label = QLabel("Post-processing cameras")
        camera_layout.addWidget(self.postprocess_camera_label, 0, 0, 1, 3)
        self.unused_face_camera_checkboxes: dict[str, QCheckBox] = {}
        for index, (camera_id, label) in enumerate(CAMERA_OPTIONS):
            checkbox = QCheckBox(label)
            checkbox.setObjectName(
                f"unused_face_camera_{camera_id}_checkbox"
            )
            checkbox.setChecked(True)
            checkbox.setToolTip(
                "Checked cameras protect visible faces during unused-face "
                "removal. Faces visible from unchecked cameras are deleted "
                "before the next generation or by Purge faces. The camera "
                "marker is illustrative only."
            )
            checkbox.toggled.connect(
                lambda _checked: self.camera_selection_changed.emit()
            )
            self.unused_face_camera_checkboxes[camera_id] = checkbox
            camera_layout.addWidget(checkbox, 1 + index // 3, index % 3)
        camera_layout.setColumnStretch(3, 1)
        details_layout.addWidget(self.unused_face_camera_controls)

        self.object_list = QListWidget()
        self.object_list.setObjectName("generated_objects_list")
        self.object_list.setMaximumHeight(OBJECT_LIST_MAXIMUM_HEIGHT)
        self.object_list.setAlternatingRowColors(True)
        self.object_list.setToolTip(
            "Select which generated Meshy object is shown in the 3D view."
        )
        details_layout.addWidget(self.object_list, 1)

        self.delete_object_button = QPushButton("Delete object")
        self.delete_object_button.setObjectName(
            "delete_generated_object_button"
        )
        self.delete_object_button.setToolTip(
            "Permanently delete the selected generated object, its embedded "
            "textures, and its unreferenced local GLB revisions."
        )
        details_layout.addWidget(self.delete_object_button)

        self.statistics_label = QLabel("No generated object")
        self.statistics_label.setObjectName("model_statistics_label")
        self.statistics_label.setWordWrap(True)
        self.statistics_label.setStyleSheet(
            "color: #aeb7c5; padding: 2px 4px;"
        )
        details_layout.addWidget(self.statistics_label)
        self._layout.addWidget(self.details_panel)

    def focus_navigation(self) -> None:
        """Forward external-window focus to the actual OpenGL viewer."""

        self.viewer.focus_navigation()

    @property
    def is_external_presentation_active(self) -> bool:
        """Whether controls are arranged beside the detached 3D viewport."""

        return self._is_external_presentation_active

    def set_external_presentation_active(self, is_active: bool) -> None:
        """Keep all object controls visible beside an external 3D viewport."""

        is_active = bool(is_active)
        if is_active == self._is_external_presentation_active:
            return
        self._is_external_presentation_active = is_active
        self._layout.setDirection(
            QBoxLayout.Direction.LeftToRight
            if is_active
            else QBoxLayout.Direction.TopToBottom
        )
        self.details_panel.setMinimumWidth(
            OBJECT_DETAILS_EXTERNAL_MINIMUM_WIDTH if is_active else 0
        )
        self.details_panel.setMaximumWidth(
            OBJECT_DETAILS_EXTERNAL_MAXIMUM_WIDTH
            if is_active
            else QT_WIDGET_MAXIMUM_SIZE
        )
        self.object_list.setMaximumHeight(
            QT_WIDGET_MAXIMUM_SIZE
            if is_active
            else OBJECT_LIST_MAXIMUM_HEIGHT
        )
        self._layout.invalidate()

    def get_enabled_postprocess_camera_ids(self) -> tuple[str, ...]:
        """Return checked cameras in the canonical processing order."""

        return tuple(
            camera_id
            for camera_id in ALL_CAMERA_IDS
            if self.unused_face_camera_checkboxes[camera_id].isChecked()
        )

    def set_postprocess_camera_controls_enabled(self, enabled: bool) -> None:
        self.unused_face_camera_controls.setEnabled(bool(enabled))


# ### Background progress mapping ###
class _BoundedProgressMapper:
    """Map provider-local percentages into one monotonic job phase."""

    def __init__(self) -> None:
        self._last_percent = 0

    def _map_message(
        self,
        message: str,
        *,
        phase_start: int,
        phase_end: int,
        phase_floor_without_percent: bool = False,
    ) -> str:
        text = str(message)
        match = _PROGRESS_PERCENT_PATTERN.search(text)
        if match is None:
            if phase_floor_without_percent:
                self._last_percent = max(self._last_percent, phase_start)
                return f"{text} ({self._last_percent}%)"
            return text
        provider_percent = max(0, min(100, int(match.group(1))))
        mapped_percent = round(
            phase_start
            + ((phase_end - phase_start) * provider_percent / 100)
        )
        self._last_percent = max(self._last_percent, mapped_percent)
        return (
            text[: match.start()]
            + f"{self._last_percent}%"
            + text[match.end() :]
        )


class _ObjectGenerationProgressMapper(_BoundedProgressMapper):
    """Keep staged geometry and texturing inside distinct progress ranges."""

    def __init__(self, request: GenerationRequest) -> None:
        super().__init__()
        unchecked_camera_ids = set(ALL_CAMERA_IDS) - set(
            request.enabled_camera_ids
        )
        self._uses_staged_pipeline = bool(
            request.geometry_only
            or request.settings.unused_face_removal
            or unchecked_camera_ids
        )
        self._geometry_only = request.geometry_only
        self._texture_phase = False

    def map_provider_message(self, message: str) -> str:
        text = str(message)
        if self._uses_staged_pipeline and "textur" in text.lower():
            self._texture_phase = True
        if self._texture_phase:
            return self._map_message(
                text,
                phase_start=_STAGED_TEXTURE_PROGRESS_START,
                phase_end=_OBJECT_PROVIDER_PROGRESS_END,
                phase_floor_without_percent=True,
            )
        geometry_end = (
            _STAGED_GEOMETRY_PROGRESS_END
            if self._uses_staged_pipeline and not self._geometry_only
            else _OBJECT_PROVIDER_PROGRESS_END
        )
        return self._map_message(
            text,
            phase_start=_OBJECT_PROVIDER_PROGRESS_START,
            phase_end=geometry_end,
        )


class _ObjectTextureProgressMapper(_BoundedProgressMapper):
    """Reserve the final fifth of progress for local texture preparation."""

    def map_provider_message(self, message: str) -> str:
        return self._map_message(
            str(message),
            phase_start=_TEXTURE_PROVIDER_PROGRESS_START,
            phase_end=_TEXTURE_PROVIDER_PROGRESS_END,
        )


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
        *,
        asset_directory: Path | None = None,
        object_id: str | None = None,
    ) -> None:
        super().__init__()
        self._planner = planner
        self._executor = executor
        self._request = request
        self._asset_directory = (
            None if asset_directory is None else Path(asset_directory)
        )
        self._object_id = None if object_id is None else str(object_id)
        self._cancel_event = threading.Event()
        self._output_lock = threading.Lock()
        self._unclaimed_asset_paths: tuple[str, ...] = ()

    def cancel(self) -> None:
        self._cancel_event.set()

    def claim_saved_output(self) -> None:
        with self._output_lock:
            self._unclaimed_asset_paths = ()

    def discard_unclaimed_output(self) -> None:
        with self._output_lock:
            asset_paths = self._unclaimed_asset_paths
            self._unclaimed_asset_paths = ()
        if self._asset_directory is not None:
            _discard_generated_asset_paths(
                self._asset_directory,
                asset_paths,
            )

    @Slot()
    def run(self) -> None:
        try:
            _raise_if_generation_cancelled(self._cancel_event)
            prepare_executor = getattr(self._executor, "prepare", None)
            if callable(prepare_executor):
                self.progress.emit("Preparing model processor...")
                _run_interruptible_stage(prepare_executor)
                _raise_if_generation_cancelled(self._cancel_event)
            progress_mapper = _ObjectGenerationProgressMapper(self._request)
            result = _run_interruptible_stage(
                lambda: _invoke_planner(
                    self._planner,
                    self._request,
                    lambda message: self.progress.emit(
                        progress_mapper.map_provider_message(message)
                    ),
                    self._cancel_event,
                )
            )
            _raise_if_generation_cancelled(self._cancel_event)
            self.progress.emit(
                "Preparing generated geometry locally (84%)"
                if self._request.geometry_only
                else (
                    "Preparing local 512, 1024 and 2048 texture variants "
                    "(84%)"
                )
            )
            generated_model = _run_interruptible_stage(
                lambda: _invoke_executor(self._executor, result)
            )
            if not isinstance(generated_model, GeneratedModel):
                raise TypeError("The Meshy executor returned an invalid model.")
            _raise_if_generation_cancelled(self._cancel_event)
            success_payload: object = result
            if self._asset_directory is not None and self._object_id is not None:
                self.progress.emit("Saving local object assets (94%)")
                saved_output = _prepare_and_persist_object_generation(
                    self._asset_directory,
                    self._object_id,
                    self._request,
                    result,
                    generated_model,
                    self._cancel_event,
                )
                with self._output_lock:
                    self._unclaimed_asset_paths = (
                        saved_output.persisted_asset_paths
                    )
                generated_model = saved_output.preview_model
                success_payload = saved_output
            _raise_if_generation_cancelled(self._cancel_event)
            self.progress.emit("Waiting to apply generated object (98%)")
        except _GenerationCancelled:
            self.discard_unclaimed_output()
            return
        except Exception as error:
            self.discard_unclaimed_output()
            if self._cancel_event.is_set():
                return
            self.failed.emit(_safe_error_message(error, self._request.settings))
            return
        else:
            self.succeeded.emit(success_payload, generated_model)
        finally:
            self.finished.emit()


class TextureRegenerationWorker(QObject):
    """Run one Retexture task and local resolution build off the UI thread."""

    succeeded = Signal(object, object)
    failed = Signal(str)
    finished = Signal()
    progress = Signal(str)

    def __init__(
        self,
        regenerator: TextureRegenerator
        | Callable[[TextureRegenerationRequest], MeshyGenerationResult],
        executor: MeshyExecutor
        | Callable[[MeshyGenerationResult], GeneratedModel],
        request: TextureRegenerationRequest | _TextureRegenerationPreflight,
        *,
        asset_directory: Path | None = None,
        symmetry: ObjectSymmetricDivisionMetadata | None = None,
        selected_resolution: int = DEFAULT_TEXTURE_RESOLUTION,
        record_snapshot: GeneratedObjectRecord | None = None,
    ) -> None:
        super().__init__()
        self._regenerator = regenerator
        self._executor = executor
        self._request = request
        self._asset_directory = (
            None if asset_directory is None else Path(asset_directory)
        )
        self._symmetry = symmetry
        self._selected_resolution = int(selected_resolution)
        self._record_snapshot = record_snapshot
        self._cancel_event = threading.Event()
        self._output_lock = threading.Lock()
        self._unclaimed_asset_paths: tuple[str, ...] = ()

    def cancel(self) -> None:
        self._cancel_event.set()

    def claim_saved_output(self) -> None:
        with self._output_lock:
            self._unclaimed_asset_paths = ()

    def discard_unclaimed_output(self) -> None:
        with self._output_lock:
            asset_paths = self._unclaimed_asset_paths
            self._unclaimed_asset_paths = ()
        if self._asset_directory is not None:
            _discard_generated_asset_paths(
                self._asset_directory,
                asset_paths,
            )

    @Slot()
    def run(self) -> None:
        try:
            _raise_if_generation_cancelled(self._cancel_event)
            self.progress.emit("Validating texture source model (2%)")
            materialized = _materialize_texture_regeneration_preflight(
                self._request,
                self._asset_directory,
                self._cancel_event,
            )
            request = materialized.request
            self._request = request
            _raise_if_generation_cancelled(self._cancel_event)
            prepare_executor = getattr(self._executor, "prepare", None)
            if callable(prepare_executor):
                self.progress.emit("Preparing model processor...")
                _run_interruptible_stage(prepare_executor)
                _raise_if_generation_cancelled(self._cancel_event)
            provider_request = request
            if request.preserve_symmetric_uvs and self._symmetry is not None:
                self.progress.emit(
                    "Building full symmetric texture reference model (6%)"
                )
                provider_model_glb = build_symmetric_retexture_proxy_glb(
                    request.model_glb,
                    self._symmetry.orientation,
                    self._symmetry.plane_coordinate,
                )
                provider_request = replace(
                    request,
                    model_glb=provider_model_glb,
                    submitted_uv_fingerprint=build_uv_fingerprint(
                        provider_model_glb
                    ),
                )
                _raise_if_generation_cancelled(self._cancel_event)
            progress_mapper = _ObjectTextureProgressMapper()
            result = _run_interruptible_stage(
                lambda: _invoke_texture_regenerator(
                    self._regenerator,
                    provider_request,
                    lambda message: self.progress.emit(
                        progress_mapper.map_provider_message(message)
                    ),
                    self._cancel_event,
                )
            )
            if not isinstance(result, MeshyGenerationResult):
                raise TypeError("Meshy returned an invalid texture result.")
            _raise_if_generation_cancelled(self._cancel_event)
            if request.preserve_symmetric_uvs:
                # The provider proxy keeps the authoritative half in the left
                # UV region. Copy only its atlas because Meshy may still
                # retriangulate the temporary full model.
                self.progress.emit(
                    "Applying texture to preserved symmetric geometry (82%)"
                )
                result = MeshyGenerationResult(
                    task_id=result.task_id,
                    glb_bytes=replace_object_base_color_texture_from_glb(
                        request.model_glb,
                        result.glb_bytes,
                    ),
                    name=result.name,
                )
                _raise_if_generation_cancelled(self._cancel_event)
            final_uv_fingerprint = self._validate_final_uvs(
                result,
                request,
            )
            self.progress.emit(
                "Preparing local 512, 1024 and 2048 texture variants (84%)"
            )
            generated_model = _run_interruptible_stage(
                lambda: _invoke_executor(self._executor, result)
            )
            if not isinstance(generated_model, GeneratedModel):
                raise TypeError("The Meshy executor returned an invalid model.")
            _raise_if_generation_cancelled(self._cancel_event)
            self.progress.emit("Local texture preparation complete. Saving...")
            outcome = TextureRegenerationOutcome(
                request=request,
                result=result,
                final_uv_fingerprint=final_uv_fingerprint,
            )
            success_payload: object = outcome
            if self._asset_directory is not None:
                if self._record_snapshot is None:
                    raise ValueError(
                        "Texture persistence requires an object snapshot."
                    )
                self.progress.emit("Saving local texture assets (94%)")
                saved_output = _prepare_and_persist_texture_regeneration(
                    self._asset_directory,
                    outcome,
                    generated_model,
                    self._symmetry,
                    self._selected_resolution,
                    self._record_snapshot,
                    materialized.source_asset_path,
                    materialized.source_asset_revision,
                    self._cancel_event,
                )
                with self._output_lock:
                    self._unclaimed_asset_paths = (
                        saved_output.persisted_asset_paths
                    )
                generated_model = saved_output.preview_model
                success_payload = saved_output
            _raise_if_generation_cancelled(self._cancel_event)
            self.progress.emit("Waiting to apply generated texture (98%)")
        except _GenerationCancelled:
            self.discard_unclaimed_output()
            return
        except Exception as error:
            self.discard_unclaimed_output()
            if self._cancel_event.is_set():
                return
            self.failed.emit(_safe_error_message(error, self._request.settings))
            return
        else:
            self.succeeded.emit(success_payload, generated_model)
        finally:
            self.finished.emit()

    def _validate_final_uvs(
        self,
        result: MeshyGenerationResult,
        request: TextureRegenerationRequest,
    ) -> UvFingerprint | None:
        submitted = request.submitted_uv_fingerprint
        if submitted is None:
            return None
        try:
            final_fingerprint = build_uv_fingerprint(result.glb_bytes)
        except UvIntegrityError as error:
            raise UvIntegrityError(
                "Meshy Retexture returned a GLB whose preserved packed UV "
                "layout could not be verified. The existing texture was "
                f"kept. Detail: {error}"
            ) from error
        _validate_symmetric_uv_retexture_integrity(
            submitted,
            final_fingerprint,
        )
        return final_fingerprint


class UncheckedCameraFacePurgeWorker(QObject):
    """Remove selected-model faces exposed to currently unchecked cameras."""

    succeeded = Signal(object)
    failed = Signal(str)
    finished = Signal()
    progress = Signal(str)

    def __init__(self, request: UncheckedCameraFacePurgeRequest) -> None:
        super().__init__()
        self._request = request
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    @Slot()
    def run(self) -> None:
        def report_progress(update: UnusedFaceRemovalProgress) -> None:
            if update.stage == "capturing":
                camera_suffix = (
                    "" if update.camera_id is None else f" ({update.camera_id})"
                )
                self.progress.emit(
                    "Capturing unchecked-camera views" + camera_suffix + "..."
                )
            elif update.stage == "exporting":
                self.progress.emit("Saving camera-purged geometry...")

        try:
            result = purge_faces_visible_from_unchecked_cameras_from_glb(
                self._request.model_glb,
                unchecked_camera_ids=self._request.unchecked_camera_ids,
                options=UncheckedCameraFacePurgeOptions(),
                cancel_requested=self._cancel_event.is_set,
                progress_callback=report_progress,
            )
            _raise_if_generation_cancelled(self._cancel_event)
            outcome = UncheckedCameraFacePurgeOutcome(
                request=self._request,
                result=result,
            )
        except _GenerationCancelled:
            return
        except Exception as error:
            if self._cancel_event.is_set():
                return
            self.failed.emit(str(error) or type(error).__name__)
            return
        else:
            self.succeeded.emit(outcome)
        finally:
            self.finished.emit()


# ### Generation workspace ###
class GenerationWorkspace(QWidget):
    """Manual video selection and Meshy Image-to-3D workspace."""

    data_changed = Signal(object)
    generated_object_deleted = Signal(str)
    generation_completed = Signal(object, object)
    texture_regeneration_completed = Signal(object, object)
    face_purge_completed = Signal(object, object)
    generated_object_changed = Signal(object, object)
    generated_object_placement_changed = Signal(object)
    operation_cancelled = Signal(str, object)
    placement_requested = Signal(str)
    placement_request_finished = Signal(str)
    operation_finished = Signal(str)

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
        *,
        meshy_texture_regenerator: TextureRegenerator
        | Callable[[TextureRegenerationRequest], MeshyGenerationResult]
        | None = None,
        job_manager: GenerationJobManager | None = None,
    ) -> None:
        super().__init__(parent)
        self._meshy_planner = meshy_planner or MeshyImagePlanner()
        self._meshy_texture_regenerator = (
            meshy_texture_regenerator or MeshyTextureRegenerator()
        )
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
        self._job_manager = job_manager
        self._object_job_runtimes: dict[str, _ObjectJobRuntime] = {}
        self._generation_thread: QThread | None = None
        self._generation_worker: (
            GenerationWorker
            | TextureRegenerationWorker
            | UncheckedCameraFacePurgeWorker
            | None
        ) = None
        self._generated_model: GeneratedModel | None = None
        self._generated_model_cache: dict[str, GeneratedModel] = {}
        self._generated_model_cache_revisions: dict[
            str,
            tuple[object, ...],
        ] = {}
        self._displayed_object_snapshot: tuple[object, ...] | None = None
        self._texture_resolution_entry_cache: dict[
            str,
            tuple[object, tuple[TextureAtlasEntry, ...]],
        ] = {}
        self._is_syncing_texture_resolution_view = False
        self._texture_resolution_change_handler: (
            Callable[[str, int], bool] | None
        ) = None
        self._object_packing_change_handler: (
            ObjectPackingChangeHandler | None
        ) = None
        self._active_generation_request: GenerationRequest | None = None
        self._active_object_operation: _ActiveObjectOperation | None = None
        self._existing_object_placement_request: (
            _ExistingObjectPlacementRequest | None
        ) = None
        self._is_rebuilding_generation_data = False
        self._is_emitting_texture_repair = False
        self._selected_object_id: str | None = None
        self._build_ui()
        self._sync_video_controls()
        self._sync_controls()

    def get_data(self) -> GenerationData:
        self._store_current_frame_strokes()
        return self._data.clone()

    def get_generated_object_ids(self) -> tuple[str, ...]:
        """Return lightweight immutable IDs without cloning Generation data."""

        return tuple(record.object_id for record in self._data.generated_objects)

    def refresh_file_backed_previews(self) -> None:
        """Reload the selected Object preview only after an asset revision."""

        record = self._find_generated_object_record(self._selected_object_id)
        if record is None:
            self._clear_generated_object_display()
            return
        display_snapshot = _build_generated_object_display_snapshot(
            record,
            self._asset_directory,
        )
        texture_signature = _build_texture_resolution_entry_signature(
            record,
            self._asset_directory,
        )
        cached_entries = self._texture_resolution_entry_cache.get(
            record.object_id
        )
        if (
            display_snapshot == self._displayed_object_snapshot
            and self._generated_model is not None
            and self.result_view.model is self._generated_model
            and cached_entries is not None
            and cached_entries[0] == texture_signature
        ):
            return
        self._display_generated_object(record, repair_missing_variant=False)

    def get_generated_object_placement(
        self,
        object_id: str,
    ) -> GeneratedObjectPlacement | None:
        """Return one immutable placement without cloning all Generation data."""

        record = self._find_generated_object_record(str(object_id).strip())
        return None if record is None else record.placement

    def set_data(self, data: GenerationData | None) -> None:
        if self.is_generating:
            raise RuntimeError("Cannot replace Generation data while generating.")
        self._finish_existing_object_placement_request()
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
        self._generated_model_cache_revisions.clear()
        self._displayed_object_snapshot = None
        self._texture_resolution_entry_cache.clear()
        self._is_rebuilding_generation_data = True
        try:
            self._rebuild_generated_objects()
        finally:
            self._is_rebuilding_generation_data = False
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

    def get_placed_preview_dependency_signature(
        self,
    ) -> tuple[tuple[object, ...], ...]:
        """Snapshot placed model files and metadata without cloning or decoding."""

        signature: list[tuple[object, ...]] = []
        for record in self._data.generated_objects:
            if record.placement is None:
                continue
            signature.append(
                (
                    record.object_id,
                    record.placement,
                    _build_generation_asset_revision(
                        self._asset_directory,
                        record.asset_path,
                    ),
                    _get_selected_texture_resolution(record),
                    _get_object_symmetric_division_metadata(record),
                )
            )
        return tuple(signature)

    def is_generated_object_asset_available(self, object_id: str) -> bool:
        """Report whether one selected GLB currently exists inside the asset root."""

        record = self._find_generated_object_record(str(object_id).strip())
        if record is None:
            return False
        try:
            return self._resolve_meshy_asset_path(record.asset_path).is_file()
        except (OSError, RuntimeError, ValueError):
            return False

    def get_texture_variant_dependency_signature(
        self,
        object_id: str,
    ) -> tuple[tuple[object, ...], ...]:
        """Expose configured GLB/PNG paths and revisions even while files are missing."""

        record = self._find_generated_object_record(str(object_id).strip())
        if record is None:
            return ()
        return tuple(
            (
                resolution,
                None if variant is None else variant[TEXTURE_VARIANT_GLB_PATH_KEY],
                (
                    None
                    if variant is None
                    else _build_generation_asset_revision(
                        self._asset_directory,
                        variant[TEXTURE_VARIANT_GLB_PATH_KEY],
                    )
                ),
                None if variant is None else variant[TEXTURE_VARIANT_PNG_PATH_KEY],
                (
                    None
                    if variant is None
                    else _build_generation_asset_revision(
                        self._asset_directory,
                        variant[TEXTURE_VARIANT_PNG_PATH_KEY],
                    )
                ),
            )
            for resolution in sorted(TEXTURE_RESOLUTIONS)
            for variant in (
                _get_texture_variant_metadata(record, resolution),
            )
        )

    def set_texture_resolution_change_handler(
        self,
        handler: Callable[[str, int], bool] | None,
    ) -> None:
        """Route UI resolution clicks through an application transaction."""

        if handler is not None and not callable(handler):
            raise TypeError("The texture resolution change handler must be callable.")
        self._texture_resolution_change_handler = handler

    def set_object_packing_change_handler(
        self,
        handler: ObjectPackingChangeHandler | None,
    ) -> None:
        """Route object packing transitions through a host transaction."""

        if handler is not None and not callable(handler):
            raise TypeError("The object packing change handler must be callable.")
        self._object_packing_change_handler = handler

    def get_object_symmetric_division(
        self,
        object_id: str,
    ) -> ObjectSymmetricDivisionMetadata | None:
        """Resolve validated immutable division metadata for Atlas pairing."""

        record = self._find_generated_object_record(object_id)
        return _get_object_symmetric_division_metadata(record)

    def resolve_symmetric_division_for_record(
        self,
        record: GeneratedObjectRecord,
    ) -> ObjectSymmetricDivisionMetadata | None:
        """Resolve candidate metadata before a record is globally committed."""

        if not isinstance(record, GeneratedObjectRecord):
            return None
        return _get_object_symmetric_division_metadata(record)

    def get_generated_object_model(
        self,
        object_id: str,
    ) -> GeneratedModel | None:
        """Load one generated model without exposing private asset paths."""

        record = self._find_generated_object_record(object_id)
        if record is None:
            return None
        try:
            return self._load_generated_object_model(record)
        except Exception:
            return None

    def get_active_texture_variant(
        self,
        object_id: str,
    ) -> ActiveObjectTextureVariant | None:
        """Resolve one object's selected GLB and PNG inside the asset root."""

        record = self._find_generated_object_record(object_id)
        if record is None:
            return None
        resolution = _get_selected_texture_resolution(record)
        return self.get_texture_variant(object_id, resolution)

    def get_texture_variant(
        self,
        object_id: str,
        resolution: int,
    ) -> ActiveObjectTextureVariant | None:
        """Resolve one exact persisted variant independent of active choice."""

        image_variant = self.get_texture_image_variant(object_id, resolution)
        if image_variant is None:
            return None
        record = self._find_generated_object_record(object_id)
        assert record is not None
        resolution = image_variant.resolution
        variant = _get_texture_variant_metadata(record, resolution)
        if variant is None:
            return None
        glb_relative_path = variant[TEXTURE_VARIANT_GLB_PATH_KEY]
        try:
            glb_path = self._resolve_meshy_asset_path(glb_relative_path)
        except (OSError, RuntimeError, ValueError):
            return None
        if not glb_path.is_file():
            return None
        return ActiveObjectTextureVariant(
            object_id=image_variant.object_id,
            object_name=image_variant.object_name,
            resolution=resolution,
            glb_asset_relative_path=glb_relative_path,
            texture_asset_relative_path=(
                image_variant.texture_asset_relative_path
            ),
            glb_asset_path=glb_path,
            texture_asset_path=image_variant.texture_asset_path,
        )

    def get_texture_image_variant(
        self,
        object_id: str,
        resolution: int,
    ) -> ObjectTextureImageVariant | None:
        """Resolve one exact PNG even when its material GLB is unavailable."""

        record = self._find_generated_object_record(object_id)
        if record is None:
            return None
        return self.resolve_texture_image_variant_for_record(
            record,
            resolution,
        )

    def resolve_texture_image_variant_for_record(
        self,
        record: GeneratedObjectRecord,
        resolution: int,
    ) -> ObjectTextureImageVariant | None:
        """Resolve a committed or prepared record's exact PNG safely."""

        if not isinstance(record, GeneratedObjectRecord):
            return None
        try:
            normalized_resolution = int(resolution)
        except (TypeError, ValueError):
            return None
        if normalized_resolution not in TEXTURE_RESOLUTIONS:
            return None
        variant = _get_texture_variant_metadata(record, normalized_resolution)
        if variant is None:
            return None
        texture_relative_path = variant[TEXTURE_VARIANT_PNG_PATH_KEY]
        try:
            texture_path = self._resolve_generated_asset_path(
                texture_relative_path,
                allowed_suffixes=frozenset({".png"}),
            )
        except (OSError, RuntimeError, ValueError):
            return None
        if not texture_path.is_file():
            return None
        return ObjectTextureImageVariant(
            object_id=record.object_id,
            object_name=record.object_name,
            resolution=normalized_resolution,
            texture_asset_relative_path=texture_relative_path,
            texture_asset_path=texture_path,
        )

    def select_object_texture_resolution(
        self,
        object_id: str,
        resolution: int,
    ) -> bool:
        """Assign one exact persisted texture variant to an object globally."""

        if self._object_has_active_mutation_job(object_id):
            return False
        record = self._find_generated_object_record(object_id)
        if record is None:
            return False
        try:
            requested_resolution = int(resolution)
        except (TypeError, ValueError):
            return False
        if requested_resolution not in _selectable_texture_resolutions(record):
            return False
        variant = self.get_texture_variant(
            record.object_id,
            requested_resolution,
        )
        if variant is None:
            return False
        normalized_resolution = variant.resolution
        if (
            _get_selected_texture_resolution(record) == normalized_resolution
            and record.asset_path == variant.glb_asset_relative_path
        ):
            return True
        try:
            preview_model = import_generated_glb(
                variant.glb_asset_path.read_bytes()
            )
        except Exception:
            return False
        pipeline = dict(record.pipeline)
        pipeline[SELECTED_TEXTURE_RESOLUTION_PIPELINE_KEY] = (
            normalized_resolution
        )
        replacement = replace(
            record,
            pipeline=pipeline,
            asset_path=variant.glb_asset_relative_path,
        )
        record_index = self._data.generated_objects.index(record)
        self._data.generated_objects[record_index] = replacement
        self._generated_model_cache.pop(record.object_id, None)
        self._generated_model_cache_revisions.pop(record.object_id, None)
        if self._selected_object_id == record.object_id:
            self._cache_generated_model(replacement, preview_model)
            self._display_generated_object(replacement)
            self.status_label.setText(
                f"Selected {normalized_resolution} x {normalized_resolution} "
                f"texture for {record.object_name}."
            )
        self._emit_data_changed()
        return True

    def set_external_3d_viewer_active(self, is_active: bool) -> None:
        """Show the local atlas inspector while the 3D panel is external."""

        is_active = bool(is_active)
        target_page = (
            self.texture_view_page if is_active else self.object_3d_page
        )
        if (
            self.object_3d_panel.is_external_presentation_active == is_active
            and self.right_view_stack.currentWidget() is target_page
        ):
            return
        self.object_3d_panel.set_external_presentation_active(is_active)
        self.right_view_stack.setCurrentWidget(target_page)

    @property
    def is_generating(self) -> bool:
        return bool(self._object_job_runtimes) or (
            self._generation_thread is not None
            and not self._object_job_runtimes
        )

    def has_active_object_job(self, object_id: str) -> bool:
        """Return whether one object's persisted state is currently reserved."""

        return self._object_has_active_mutation_job(object_id)

    def request_object_placement(self) -> bool:
        """Request placement for an in-flight or selected completed object."""

        placeable_jobs = tuple(
            runtime
            for runtime in self._object_job_runtimes.values()
            if self._can_place_active_operation(runtime.operation)
        )
        if len(placeable_jobs) == 1:
            operation = placeable_jobs[0].operation
            self.placement_requested.emit(operation.operation_id)
            return True
        record = self._find_generated_object_record(self._selected_object_id)
        if (
            record is None
            or self._object_has_active_mutation_job(record.object_id)
        ):
            return False

        self._finish_existing_object_placement_request()
        request = _ExistingObjectPlacementRequest(
            request_id=uuid.uuid4().hex,
            object_id=record.object_id,
        )
        self._existing_object_placement_request = request
        self.placement_requested.emit(request.request_id)
        return True

    def request_active_object_placement(self) -> bool:
        """Compatibility alias for the generalized placement request."""

        return self.request_object_placement()

    def set_active_object_placement(
        self,
        operation_id: str,
        placement: GeneratedObjectPlacement,
    ) -> bool:
        """Accept placement only for the exact current request token."""

        if not isinstance(placement, GeneratedObjectPlacement):
            return False
        exact_request_id = str(operation_id)
        runtime = self._object_job_runtimes.get(exact_request_id)
        operation = None if runtime is None else runtime.operation
        if operation is not None and self._can_place_active_operation(operation):
            operation.pending_placement = placement
            self.status_label.setText(
                "Object placement selected. Generation is still in progress."
            )
            return True

        request = self._existing_object_placement_request
        if (
            request is None
            or request.request_id != exact_request_id
        ):
            return False
        record_index = next(
            (
                index
                for index, record in enumerate(self._data.generated_objects)
                if record.object_id == request.object_id
            ),
            None,
        )
        if record_index is None:
            self._existing_object_placement_request = None
            self._sync_controls()
            return False

        record = self._data.generated_objects[record_index]
        if record.placement is not None:
            placement = replace(
                placement,
                height_offset_meters=(
                    record.placement.height_offset_meters
                ),
                rotation_degrees=record.placement.rotation_degrees,
            )
        replacement = replace(record, placement=placement)
        self._existing_object_placement_request = None
        self._data.generated_objects[record_index] = replacement
        self.status_label.setText(f"Placed: {record.object_name}")
        self._emit_data_changed()
        self.generated_object_placement_changed.emit(replacement)
        self._sync_controls()
        return True

    def update_generated_object_placement(
        self,
        object_id: str,
        placement: GeneratedObjectPlacement,
        *,
        emit_change_signals: bool = True,
    ) -> bool:
        """Commit one completed object's placement by stable ID.

        Canvas gizmos may suppress signals after updating their retained preview
        directly. Ordinary placement callers keep the default notifications.
        """

        if not isinstance(placement, GeneratedObjectPlacement):
            return False
        normalized_object_id = str(object_id).strip()
        record_index = next(
            (
                index
                for index, record in enumerate(self._data.generated_objects)
                if record.object_id == normalized_object_id
                and record.placement is not None
            ),
            None,
        )
        if record_index is None:
            return False
        record = self._data.generated_objects[record_index]
        if record.placement == placement:
            return True

        replacement = replace(record, placement=placement)
        self._data.generated_objects[record_index] = replacement
        if emit_change_signals:
            self._emit_data_changed()
            self.generated_object_placement_changed.emit(replacement)
        return True

    def cancel_object_placement_request(self, request_id: str) -> bool:
        """Forget one exact completed-object placement request."""

        request = self._existing_object_placement_request
        if request is None or request.request_id != str(request_id):
            return False
        self._existing_object_placement_request = None
        self._sync_controls()
        return True

    def cancel_current_operation(self) -> bool:
        """Request an idempotent rollback of the active object operation."""

        runtime = self._legacy_active_job_runtime()
        if runtime is None:
            return False
        return self.cancel_operation(runtime.operation_id)

    def cancel_operation(self, operation_id: str) -> bool:
        """Cancel one exact generation job without affecting its siblings."""

        runtime = self._object_job_runtimes.get(str(operation_id))
        if runtime is None:
            return False
        manager = self._job_manager
        if manager is not None and runtime.managed_job_id is not None:
            managed_job = manager.get_job(runtime.managed_job_id)
            if managed_job is not None and managed_job.is_finished:
                return False
        operation = runtime.operation
        if operation.cancel_requested:
            return True
        operation.cancel_requested = True
        operation.pending_placement = None
        worker = runtime.worker
        if worker is not None and is_valid_qt_object(worker):
            worker.cancel()
        thread = runtime.thread
        if is_valid_qt_object(thread):
            thread.requestInterruption()
        self.status_label.setText("Cancelling the current operation...")
        self._sync_controls()
        return True

    def delete_selected_generated_object(self) -> bool:
        """Delete the selected object without showing an interactive prompt."""

        if self._selected_object_id is None:
            return False
        return self.delete_generated_object(self._selected_object_id)

    def purge_selected_object_faces(self) -> bool:
        """Delete faces visible from every currently unchecked camera."""

        request = self._build_unchecked_camera_face_purge_request()
        if request is None:
            return False
        if self._object_has_active_mutation_job(request.object_id):
            self.status_label.setText(
                "Wait for the selected object's active job to finish."
            )
            return False
        self._start_unchecked_camera_face_purge(request)
        return True

    def regenerate_selected_object_texture(self) -> bool:
        """Compatibility alias for :meth:`generate_selected_object_texture`."""

        return self.generate_selected_object_texture()

    def generate_selected_object_texture(self) -> bool:
        """Generate the selected object's texture from the current mask."""

        request = self._build_texture_regeneration_request()
        if request is None:
            return False
        requested_name = self.job_name_edit.text().strip()
        if not self._start_texture_regeneration(
            request,
            requested_name=requested_name,
        ):
            return False
        self.job_name_edit.clear()
        return True

    def undo_selected_object_change(self) -> bool:
        """Undo the selected object's latest local or texture operation."""

        if self._selected_object_id is None:
            return False
        return self._undo_object_change(self._selected_object_id)

    def _undo_object_change(
        self,
        object_id: str,
        *,
        expected_operation: str | None = None,
        allow_operation_id: str | None = None,
    ) -> bool:
        """Undo one exact object's latest operation and its Atlas transition."""

        if self._object_has_active_mutation_job(
            object_id,
            excluding_operation_id=allow_operation_id,
        ):
            return False
        record = self._find_generated_object_record(object_id)
        if record is None:
            return False
        undo_stack = _get_object_operation_undo_stack(record)
        if not undo_stack:
            return False
        snapshot = undo_stack[-1]
        if (
            expected_operation is not None
            and snapshot.get("operation") != expected_operation
        ):
            return False
        try:
            replacement = _restore_object_operation_snapshot(
                record,
                snapshot,
                undo_stack[:-1],
            )
            preview_model = import_generated_glb(
                self._resolve_meshy_asset_path(
                    replacement.asset_path
                ).read_bytes()
            )
        except Exception as error:
            self.status_label.setText(f"Undo could not restore the object: {error}")
            return False

        if not self._request_object_packing_change(
            record,
            replacement,
            preview_model,
        ):
            self.status_label.setText(
                "Undo could not restore the object because the Atlas packing "
                "change was rejected."
            )
            self._sync_controls()
            return False
        cleanup_failed = self._delete_unreferenced_object_assets(record)
        operation = str(snapshot.get("operation", "object change"))
        operation_label = {
            OBJECT_OPERATION_GENERATE_TEXTURE: "texture generation",
            OBJECT_OPERATION_PURGE_FACES: "face purge",
        }.get(operation, "object change")
        status_suffix = (
            " Some superseded files could not be removed."
            if cleanup_failed
            else ""
        )
        self.status_label.setText(
            f"Undid {operation_label}: {record.object_name}." + status_suffix
        )
        self._emit_data_changed()
        self.generated_object_changed.emit(replacement, preview_model)
        self._sync_controls()
        return True

    def delete_generated_object(
        self,
        object_id: str,
        *,
        allow_operation_id: str | None = None,
    ) -> bool:
        """Remove one generated object and its unreferenced GLB/PNG assets.

        This programmatic seam deliberately does not show a confirmation
        dialog. UI callers confirm first. Missing records, active generation,
        unsafe asset paths, and filesystem cleanup failures never partially
        remove another object.
        """

        if self._object_has_active_mutation_job(
            object_id,
            excluding_operation_id=allow_operation_id,
        ):
            self.status_label.setText(
                "Wait for this object's active job to finish before deleting it."
            )
            return False
        if not isinstance(object_id, str) or not object_id:
            return False
        record_index = next(
            (
                index
                for index, record in enumerate(self._data.generated_objects)
                if record.object_id == object_id
            ),
            None,
        )
        if record_index is None:
            return False

        pending_placement = self._existing_object_placement_request
        if (
            pending_placement is not None
            and pending_placement.object_id == object_id
        ):
            self._finish_existing_object_placement_request()
        deleted_record = self._data.generated_objects.pop(record_index)
        self._generated_model_cache.pop(object_id, None)
        self._generated_model_cache_revisions.pop(object_id, None)
        self._texture_resolution_entry_cache.pop(object_id, None)
        asset_cleanup_failed = self._delete_unreferenced_object_assets(
            deleted_record
        )

        preferred_object_id = self._selected_object_id
        if preferred_object_id == object_id:
            preferred_object_id = (
                None
                if not self._data.generated_objects
                else self._data.generated_objects[
                    min(record_index, len(self._data.generated_objects) - 1)
                ].object_id
            )
        self._refresh_generated_objects_list(preferred_object_id)
        if asset_cleanup_failed:
            self.status_label.setText(
                f"Deleted: {deleted_record.object_name}. Some local GLB "
                "files could not be removed."
            )
        else:
            self.status_label.setText(f"Deleted: {deleted_record.object_name}")
        self.generated_object_deleted.emit(deleted_record.object_id)
        self._emit_data_changed()
        self._sync_controls()
        return True

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

    def set_texture_regenerator(
        self,
        regenerator: TextureRegenerator
        | Callable[[TextureRegenerationRequest], MeshyGenerationResult],
    ) -> None:
        """Replace the Retexture adapter, primarily for tests/integrations."""

        self._meshy_texture_regenerator = regenerator

    def load_video(self, video_path: str) -> None:
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
        request = self._build_generation_request()
        if request is None:
            return
        self._start_generation(
            request,
            requested_name=self._take_requested_job_name(),
        )

    def generate_geometry(self) -> None:
        """Generate and locally process geometry without submitting Retexture."""

        if self.symmetric_division_checkbox.isChecked():
            self.status_label.setText(
                "Symmetric division requires the full Generate workflow. "
                "Clear the option to generate geometry only."
            )
            return
        request = self._build_generation_request(geometry_only=True)
        if request is None:
            return
        self._start_generation(
            request,
            requested_name=self._take_requested_job_name(),
        )

    def _start_generation(
        self,
        request: GenerationRequest,
        *,
        requested_name: str | None = None,
    ) -> None:
        """Start one independently owned model-generation request."""

        object_id = uuid.uuid4().hex
        operation = _ActiveObjectOperation(
            kind=OBJECT_OPERATION_GENERATE_MODEL,
        )
        thread = QThread(self)
        worker = GenerationWorker(
            self._meshy_planner,
            self._meshy_executor,
            request,
            asset_directory=self._asset_directory,
            object_id=object_id,
        )
        relay = _ObjectJobSignalRelay(operation.operation_id, self)
        managed_job_id = self._create_managed_job(
            operation,
            kind=GENERATION_JOB_KIND_MODEL,
            requested_name=requested_name,
            default_name=f"Object from frame {request.frame_index + 1}",
            stage=f"Submitting frame {request.frame_index + 1} to Meshy...",
        )
        runtime = _ObjectJobRuntime(
            operation=operation,
            thread=thread,
            worker=worker,
            relay=relay,
            generation_request=request,
            requested_name=str(requested_name or "").strip(),
            managed_job_id=managed_job_id,
        )
        self._register_object_job_runtime(runtime)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.succeeded.connect(relay.forward_pair_succeeded)
        worker.failed.connect(relay.forward_failed)
        worker.progress.connect(relay.forward_progress)
        relay.pair_succeeded.connect(
            self._handle_job_generation_succeeded
        )
        relay.failed.connect(self._handle_job_generation_failed)
        relay.progress.connect(self._handle_job_generation_progress)
        relay.thread_finished.connect(
            self._handle_object_job_thread_finished
        )
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(thread.quit)
        thread.finished.connect(relay.forward_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self.status_label.setText(
            f"Submitting frame {request.frame_index + 1} to Meshy..."
        )
        thread.start()
        self._sync_controls()

    def _start_texture_regeneration(
        self,
        request: TextureRegenerationRequest | _TextureRegenerationPreflight,
        *,
        requested_name: str | None = None,
    ) -> bool:
        """Start a texture-only task for one immutable selected-object snapshot."""

        if self._object_has_active_mutation_job(request.object_id):
            self.status_label.setText(
                "Wait for the selected object's active job to finish."
            )
            return False
        record = self._find_generated_object_record(request.object_id)
        if record is None:
            self.status_label.setText(
                "The selected object no longer exists."
            )
            return False
        operation = _ActiveObjectOperation(
            kind=OBJECT_OPERATION_GENERATE_TEXTURE,
            target_object_id=request.object_id,
        )
        thread = QThread(self)
        worker = TextureRegenerationWorker(
            self._meshy_texture_regenerator,
            self._meshy_executor,
            request,
            asset_directory=self._asset_directory,
            symmetry=_get_object_symmetric_division_metadata(record),
            selected_resolution=_get_selected_texture_resolution(record),
            record_snapshot=replace(
                record,
                pipeline=copy.deepcopy(record.pipeline),
            ),
        )
        relay = _ObjectJobSignalRelay(operation.operation_id, self)
        default_name = (
            f"Texture: {record.object_name}"
        )
        managed_job_id = self._create_managed_job(
            operation,
            kind=GENERATION_JOB_KIND_TEXTURE,
            requested_name=requested_name,
            default_name=default_name,
            stage="Preparing texture generation...",
        )
        runtime = _ObjectJobRuntime(
            operation=operation,
            thread=thread,
            worker=worker,
            relay=relay,
            requested_name=str(requested_name or "").strip(),
            managed_job_id=managed_job_id,
        )
        self._register_object_job_runtime(runtime)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.succeeded.connect(relay.forward_pair_succeeded)
        worker.failed.connect(relay.forward_failed)
        worker.progress.connect(relay.forward_progress)
        relay.pair_succeeded.connect(
            self._handle_job_texture_regeneration_succeeded
        )
        relay.failed.connect(
            self._handle_job_texture_regeneration_failed
        )
        relay.progress.connect(self._handle_job_generation_progress)
        relay.thread_finished.connect(
            self._handle_object_job_thread_finished
        )
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(thread.quit)
        thread.finished.connect(relay.forward_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self.status_label.setText("Preparing texture generation...")
        thread.start()
        self._sync_controls()
        return True

    def _start_unchecked_camera_face_purge(
        self,
        request: UncheckedCameraFacePurgeRequest,
    ) -> None:
        """Start one local face-purge request off the UI thread."""

        if self._object_has_active_mutation_job(request.object_id):
            self.status_label.setText(
                "Wait for the selected object's active job to finish."
            )
            return
        operation = _ActiveObjectOperation(
            kind=OBJECT_OPERATION_PURGE_FACES,
            target_object_id=request.object_id,
        )
        thread = QThread(self)
        worker = UncheckedCameraFacePurgeWorker(request)
        relay = _ObjectJobSignalRelay(operation.operation_id, self)
        record = self._find_generated_object_record(request.object_id)
        managed_job_id = self._create_managed_job(
            operation,
            kind=GENERATION_JOB_KIND_FACE_PURGE,
            requested_name=None,
            default_name=(
                "Object face purge"
                if record is None
                else f"Purge faces: {record.object_name}"
            ),
            stage="Preparing unchecked-camera face purge...",
        )
        runtime = _ObjectJobRuntime(
            operation=operation,
            thread=thread,
            worker=worker,
            relay=relay,
            managed_job_id=managed_job_id,
        )
        self._register_object_job_runtime(runtime)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.succeeded.connect(relay.forward_single_succeeded)
        worker.failed.connect(relay.forward_failed)
        worker.progress.connect(relay.forward_progress)
        relay.single_succeeded.connect(
            self._handle_job_face_purge_succeeded
        )
        relay.failed.connect(self._handle_job_face_purge_failed)
        relay.progress.connect(self._handle_job_generation_progress)
        relay.thread_finished.connect(
            self._handle_object_job_thread_finished
        )
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(thread.quit)
        thread.finished.connect(relay.forward_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self.status_label.setText("Preparing unchecked-camera face purge...")
        thread.start()
        self._sync_controls()

    # ### Multi-job runtime helpers ###
    def _take_requested_job_name(self) -> str:
        """Snapshot and clear the optional name for the next accepted job."""

        requested_name = self.job_name_edit.text().strip()
        self.job_name_edit.clear()
        return requested_name

    def _create_managed_job(
        self,
        operation: _ActiveObjectOperation,
        *,
        kind: str,
        requested_name: str | None,
        default_name: str,
        stage: str,
    ) -> str | None:
        manager = self._job_manager
        if manager is None:
            return None
        job = manager.create_job(
            kind=kind,
            requested_name=str(requested_name or ""),
            default_name=default_name,
            stage=stage,
        )
        return job.job_id

    def _register_object_job_runtime(
        self,
        runtime: _ObjectJobRuntime,
    ) -> None:
        operation_id = runtime.operation_id
        self._object_job_runtimes[operation_id] = runtime
        self._set_legacy_active_job_runtime(runtime)
        manager = self._job_manager
        if manager is not None and runtime.managed_job_id is not None:
            manager.set_cancel_callback(
                runtime.managed_job_id,
                lambda operation_id=operation_id: self.cancel_operation(
                    operation_id
                ),
            )

    def _set_legacy_active_job_runtime(
        self,
        runtime: _ObjectJobRuntime | None,
    ) -> None:
        """Keep historical single-job attributes as aliases for integrations."""

        if runtime is None:
            self._generation_thread = None
            self._generation_worker = None
            self._active_generation_request = None
            self._active_object_operation = None
            return
        self._generation_thread = runtime.thread
        self._generation_worker = runtime.worker
        self._active_generation_request = runtime.generation_request
        self._active_object_operation = runtime.operation

    def _legacy_active_job_runtime(self) -> _ObjectJobRuntime | None:
        operation = self._active_object_operation
        if operation is not None:
            runtime = self._object_job_runtimes.get(operation.operation_id)
            if runtime is not None:
                return runtime
        if not self._object_job_runtimes:
            return None
        return next(reversed(self._object_job_runtimes.values()))

    def _object_has_active_mutation_job(
        self,
        object_id: str,
        *,
        excluding_operation_id: str | None = None,
    ) -> bool:
        normalized_id = str(object_id).strip()
        return any(
            operation_id != excluding_operation_id
            and runtime.operation.target_object_id == normalized_id
            for operation_id, runtime in self._object_job_runtimes.items()
        )

    def _format_object_job_status(
        self,
        runtime: _ObjectJobRuntime | None,
        message: str,
    ) -> str:
        """Prefix Object-tab texture feedback with its optional job name."""

        if (
            runtime is not None
            and runtime.operation.kind == OBJECT_OPERATION_GENERATE_TEXTURE
            and runtime.requested_name
        ):
            return f"{runtime.requested_name}: {message}"
        return str(message)

    def _update_managed_job_progress(
        self,
        runtime: _ObjectJobRuntime,
        message: str,
    ) -> None:
        manager = self._job_manager
        if manager is None or runtime.managed_job_id is None:
            return
        manager.update_job(runtime.managed_job_id, stage=str(message))

    def _complete_managed_job(
        self,
        runtime: _ObjectJobRuntime,
        stage: str,
    ) -> None:
        manager = self._job_manager
        if manager is None or runtime.managed_job_id is None:
            return
        manager.complete_job(runtime.managed_job_id, stage=stage)

    def _fail_managed_job(
        self,
        runtime: _ObjectJobRuntime,
        stage: str,
    ) -> None:
        manager = self._job_manager
        if manager is None or runtime.managed_job_id is None:
            return
        manager.fail_job(runtime.managed_job_id, stage=stage)

    def _mark_managed_job_cancelled(
        self,
        runtime: _ObjectJobRuntime,
        stage: str,
    ) -> None:
        manager = self._job_manager
        if manager is None or runtime.managed_job_id is None:
            return
        manager.mark_cancelled(runtime.managed_job_id, stage=stage)

    def shutdown(self) -> None:
        self._existing_object_placement_request = None
        runtimes = tuple(self._object_job_runtimes.values())
        for runtime in runtimes:
            runtime.operation.cancel_requested = True
            worker = runtime.worker
            thread = runtime.thread
            if isinstance(worker, GenerationWorker | TextureRegenerationWorker):
                worker.discard_unclaimed_output()
            if is_valid_qt_object(worker):
                worker.cancel()
            if is_valid_qt_object(thread):
                thread.requestInterruption()
                thread.quit()
        for runtime in runtimes:
            thread = runtime.thread
            while is_valid_qt_object(thread) and thread.isRunning():
                thread.wait(SHUTDOWN_WAIT_MILLISECONDS)
            worker = runtime.worker
            if isinstance(worker, GenerationWorker | TextureRegenerationWorker):
                worker.discard_unclaimed_output()
            self._mark_managed_job_cancelled(
                runtime,
                "Cancelled during shutdown",
            )
        self._object_job_runtimes.clear()
        self._set_legacy_active_job_runtime(None)
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
        self.result_view.set_ambient_light_intensity(
            OBJECT_GENERATION_DEFAULT_AMBIENT_LIGHT_INTENSITY
        )
        self.generated_objects_list = self.object_3d_panel.object_list
        self.delete_generated_object_button = (
            self.object_3d_panel.delete_object_button
        )
        self.model_statistics_label = self.object_3d_panel.statistics_label
        self.generated_objects_list.currentItemChanged.connect(
            self._handle_generated_object_selection_changed
        )
        self.object_3d_panel.camera_selection_changed.connect(
            self._sync_controls
        )
        self.delete_generated_object_button.clicked.connect(
            self._handle_delete_generated_object_clicked
        )

        self.object_3d_page = _build_labeled_view(
            "Generated 3D objects",
            self.object_3d_panel,
        )
        self.texture_view = TextureAtlasView(
            empty_preview_text="No texture resolutions available",
            unselected_preview_text="Select a texture resolution",
        )
        self.texture_view.setObjectName("object_texture_resolution_view")
        self.texture_view.atlas_selected.connect(
            self._handle_texture_resolution_selected
        )
        self.texture_view_page = _build_labeled_view(
            "Texture resolution",
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
            "Overlay the generated model's triangle edges and show its UV "
            "edges and vertices over the texture-resolution preview."
        )
        self.wireframe_checkbox.toggled.connect(
            self._handle_wireframe_toggled
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

        self.symmetric_division_checkbox = QCheckBox("Symmetric division")
        self.symmetric_division_checkbox.setObjectName(
            "symmetric_division_checkbox"
        )
        self.symmetric_division_checkbox.setToolTip(
            "Automatically keep the generated model half with fewer "
            "triangles and pack its texture into one atlas half. This only "
            "applies to a brand-new Generate request."
        )
        self.symmetric_division_checkbox.toggled.connect(self._sync_controls)
        buttons_layout.addWidget(self.symmetric_division_checkbox)

        self.symmetric_division_orientation_combo = QComboBox()
        self.symmetric_division_orientation_combo.setObjectName(
            "symmetric_division_orientation_combo"
        )
        self.symmetric_division_orientation_combo.addItem(
            "Vertical",
            SYMMETRIC_DIVISION_ORIENTATION_VERTICAL,
        )
        self.symmetric_division_orientation_combo.addItem(
            "Horizontal",
            SYMMETRIC_DIVISION_ORIENTATION_HORIZONTAL,
        )
        self.symmetric_division_orientation_combo.setToolTip(
            "Choose the axis of the automatic midpoint division."
        )
        buttons_layout.addWidget(self.symmetric_division_orientation_combo)

        self.job_name_edit = QLineEdit()
        self.job_name_edit.setObjectName("object_generation_job_name_edit")
        self.job_name_edit.setPlaceholderText("Job name (optional)")
        self.job_name_edit.setClearButtonEnabled(True)
        self.job_name_edit.setMaximumWidth(190)
        self.job_name_edit.setToolTip(
            "Optionally name the next object or object-texture generation job."
        )
        buttons_layout.addWidget(self.job_name_edit)

        self.generate_button = QPushButton("Generate")
        self.generate_button.setMinimumHeight(38)
        self.generate_button.setToolTip(
            "Requires a painted object mask and a Meshy API key. "
            "Meshy Image-to-3D tasks consume account credits."
        )
        self.generate_button.clicked.connect(self.generate)
        buttons_layout.addWidget(self.generate_button)

        buttons_layout.addSpacing(30)

        self.generate_geometry_button = QPushButton("Generate geometry")
        self.generate_geometry_button.setObjectName("generate_geometry_button")
        self.generate_geometry_button.setMinimumHeight(38)
        self.generate_geometry_button.setToolTip(
            "Generate and locally post-process the painted video object's "
            "geometry without submitting a Meshy texture task."
        )
        self.generate_geometry_button.clicked.connect(self.generate_geometry)
        buttons_layout.addWidget(self.generate_geometry_button)

        self.generate_texture_button = QPushButton("Generate texture")
        self.generate_texture_button.setObjectName("generate_texture_button")
        self.generate_texture_button.setMinimumHeight(38)
        self.generate_texture_button.setToolTip(
            "Submit a Meshy texture task for the selected generated object. "
            "Meshy tasks consume account credits."
        )
        self.generate_texture_button.clicked.connect(
            self.generate_selected_object_texture
        )
        buttons_layout.addWidget(self.generate_texture_button)
        self.regenerate_texture_button = self.generate_texture_button

        self.purge_faces_button = QPushButton("Purge faces")
        self.purge_faces_button.setObjectName("purge_faces_button")
        self.purge_faces_button.setMinimumHeight(38)
        self.purge_faces_button.setToolTip(
            "Delete faces of the selected generated object that are visible "
            "from any unchecked post-processing camera."
        )
        self.purge_faces_button.clicked.connect(
            self.purge_selected_object_faces
        )
        buttons_layout.addWidget(self.purge_faces_button)
        buttons_layout.addStretch(1)

        self.place_object_button = QPushButton("Place")
        self.place_object_button.setObjectName("place_generated_object_button")
        self.place_object_button.setMinimumHeight(38)
        self.place_object_button.setToolTip(
            "Choose where the selected object appears on the Canvas, or set "
            "the destination while a new model is still generating."
        )
        self.place_object_button.clicked.connect(
            self.request_object_placement
        )
        buttons_layout.addWidget(self.place_object_button)

        self.undo_object_change_button = QPushButton("Undo")
        self.undo_object_change_button.setObjectName(
            "undo_object_change_button"
        )
        self.undo_object_change_button.setMinimumHeight(38)
        self.undo_object_change_button.setToolTip(
            "Restore the selected object to its state before the latest "
            "Generate texture or Purge faces operation."
        )
        self.undo_object_change_button.clicked.connect(
            self.undo_selected_object_change
        )
        buttons_layout.addWidget(self.undo_object_change_button)

        self.cancel_operation_button = QPushButton("Cancel")
        self.cancel_operation_button.setObjectName(
            "cancel_object_operation_button"
        )
        self.cancel_operation_button.setMinimumHeight(38)
        self.cancel_operation_button.setToolTip(
            "Cancel the current generation or face operation and restore "
            "the object state from before it started."
        )
        self.cancel_operation_button.clicked.connect(
            self.cancel_current_operation
        )
        buttons_layout.addWidget(self.cancel_operation_button)
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

    @Slot()
    def _handle_delete_generated_object_clicked(self) -> None:
        record = self._find_generated_object_record(self._selected_object_id)
        if (
            record is None
            or self._object_has_active_mutation_job(record.object_id)
        ):
            return
        response = QMessageBox.question(
            self,
            "Delete generated object",
            f'Permanently delete "{record.object_name}", its embedded '
            "textures, and its unreferenced local GLB revisions?",
            (
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
            ),
            QMessageBox.StandardButton.No,
        )
        if response != QMessageBox.StandardButton.Yes:
            return
        self.delete_generated_object(record.object_id)

    def _handle_seekbar_changed(self, frame_index: int) -> None:
        if self._is_syncing_seekbar:
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

    @Slot(bool)
    def _handle_wireframe_toggled(self, enabled: bool) -> None:
        self.result_view.set_wireframe_enabled(enabled)
        self.texture_view.set_uv_overlay_enabled(enabled)

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

    # ### Active operation helpers ###
    def _finish_existing_object_placement_request(self) -> None:
        """Invalidate and close the picker for one completed object."""

        request = self._existing_object_placement_request
        if request is None:
            return
        self._existing_object_placement_request = None
        self.placement_request_finished.emit(request.request_id)

    def _can_place_active_operation(
        self,
        operation: _ActiveObjectOperation | None,
    ) -> bool:
        runtime = (
            None
            if operation is None
            else self._object_job_runtimes.get(operation.operation_id)
        )
        return bool(
            operation is not None
            and operation.kind == OBJECT_OPERATION_GENERATE_MODEL
            and not operation.cancel_requested
            and operation.committed_object_id is None
            and runtime is not None
            and is_valid_qt_object(runtime.thread)
            and isinstance(runtime.worker, GenerationWorker)
        )

    @Slot(str)
    def _handle_generation_progress(self, message: str) -> None:
        """Compatibility handler for an explicitly connected legacy worker."""

        operation = self._active_object_operation
        sender = self.sender()
        if sender is not None and sender is not self._generation_worker:
            return
        if operation is not None and operation.cancel_requested:
            return
        self.status_label.setText(str(message))

    @Slot(str, str)
    def _handle_job_generation_progress(
        self,
        operation_id: str,
        message: str,
    ) -> None:
        runtime = self._object_job_runtimes.get(str(operation_id))
        if runtime is None or runtime.operation.cancel_requested:
            return
        manager = self._job_manager
        if manager is not None and runtime.managed_job_id is not None:
            managed_job = manager.get_job(runtime.managed_job_id)
            if managed_job is not None and managed_job.is_finished:
                return
        if runtime.operation.committed_object_id is not None:
            return
        self._update_managed_job_progress(runtime, str(message))
        if runtime is self._legacy_active_job_runtime():
            self.status_label.setText(
                self._format_object_job_status(runtime, str(message))
            )

    def _should_ignore_operation_result(
        self,
        expected_kind: str,
        operation_id: str | None = None,
    ) -> bool:
        if operation_id is not None:
            runtime = self._object_job_runtimes.get(str(operation_id))
            return bool(
                runtime is None
                or runtime.operation.kind != expected_kind
                or runtime.operation.cancel_requested
            )
        sender = self.sender()
        if sender is not None and sender is not self._generation_worker:
            return True
        operation = self._active_object_operation
        return operation is not None and (
            operation.kind != expected_kind
            or operation.cancel_requested
        )

    def _record_operation_commit(
        self,
        expected_kind: str,
        object_id: str,
        operation_id: str | None = None,
    ) -> None:
        operation = self._active_object_operation
        if operation_id is not None:
            runtime = self._object_job_runtimes.get(str(operation_id))
            operation = None if runtime is None else runtime.operation
        if (
            operation is None
            or operation.kind != expected_kind
            or operation.cancel_requested
        ):
            return
        if (
            operation.target_object_id is not None
            and operation.target_object_id != object_id
        ):
            return
        operation.committed_object_id = object_id
        self._sync_controls()

    @Slot(str, object, object)
    def _handle_job_generation_succeeded(
        self,
        operation_id: str,
        result: object,
        generated_model: object,
    ) -> None:
        if not isinstance(generated_model, GeneratedModel):
            self._handle_job_generation_failed(
                operation_id,
                "The generated preview model is invalid.",
            )
            return
        runtime = self._object_job_runtimes.get(str(operation_id))
        if runtime is None:
            return
        self._set_legacy_active_job_runtime(runtime)
        self._handle_generation_succeeded(
            result,
            generated_model,
            operation_id=operation_id,
        )
        if runtime.operation.committed_object_id is not None:
            worker = runtime.worker
            if isinstance(worker, GenerationWorker):
                worker.claim_saved_output()

    @Slot(str, object, object)
    def _handle_job_texture_regeneration_succeeded(
        self,
        operation_id: str,
        outcome: object,
        generated_model: object,
    ) -> None:
        if not isinstance(generated_model, GeneratedModel):
            self._handle_job_texture_regeneration_failed(
                operation_id,
                "The generated preview model is invalid.",
            )
            return
        runtime = self._object_job_runtimes.get(str(operation_id))
        if runtime is None:
            return
        self._set_legacy_active_job_runtime(runtime)
        self._handle_texture_regeneration_succeeded(
            outcome,
            generated_model,
            operation_id=operation_id,
        )
        if runtime.operation.committed_object_id is not None:
            worker = runtime.worker
            if isinstance(worker, TextureRegenerationWorker):
                worker.claim_saved_output()

    @Slot(str, object)
    def _handle_job_face_purge_succeeded(
        self,
        operation_id: str,
        outcome: object,
    ) -> None:
        runtime = self._object_job_runtimes.get(str(operation_id))
        if runtime is None:
            return
        self._set_legacy_active_job_runtime(runtime)
        self._handle_unchecked_camera_face_purge_succeeded(
            outcome,
            operation_id=operation_id,
        )

    @Slot(object, object)
    def _handle_generation_succeeded(
        self,
        result: object,
        generated_model: GeneratedModel,
        *,
        operation_id: str | None = None,
    ) -> None:
        if self._should_ignore_operation_result(
            OBJECT_OPERATION_GENERATE_MODEL,
            operation_id,
        ):
            return
        runtime = (
            None
            if operation_id is None
            else self._object_job_runtimes.get(str(operation_id))
        )
        generation_request = (
            self._active_generation_request
            if runtime is None
            else runtime.generation_request
        )
        if operation_id is None:
            self._active_generation_request = None
        if isinstance(result, _SavedObjectGeneration):
            self._commit_saved_object_generation(
                result,
                generation_request,
                runtime,
                operation_id,
            )
            return
        if not isinstance(result, MeshyGenerationResult):
            self._handle_generation_failed(
                "Meshy returned an invalid result.",
                operation_id=operation_id,
            )
            return
        object_id = uuid.uuid4().hex
        object_name = (
            runtime.requested_name
            if runtime is not None and runtime.requested_name
            else result.name
        )
        pipeline: dict[str, object] = {}
        persisted_asset_paths: list[str] = []
        symmetry: ObjectSymmetricDivisionMetadata | None = None
        try:
            geometry_only = (
                isinstance(result, StagedMeshyGenerationResult)
                and result.geometry_only
            )
            texture_variants = (
                None
                if geometry_only
                else generated_model.object_texture_variants
            )
            symmetric_division_was_requested = bool(
                generation_request is not None
                and generation_request.symmetric_division_enabled
            )
            if symmetric_division_was_requested:
                if geometry_only or not isinstance(
                    texture_variants,
                    ObjectTextureVariants,
                ):
                    raise ValueError(
                        "Symmetric division requires a newly generated "
                        "textured model."
                    )
                division_result = build_automatic_symmetric_object_variants(
                    texture_variants.glb_by_resolution[
                        TEXTURE_RESOLUTION_2048
                    ],
                    generation_request.symmetric_division_orientation,
                )
                symmetry = _validate_automatic_symmetric_division_result(
                    division_result,
                    generation_request.symmetric_division_orientation,
                )
                texture_variants = division_result.variants
            if texture_variants is None:
                asset_path = self._persist_meshy_asset(
                    object_id,
                    result.glb_bytes,
                )
                persisted_asset_paths.append(asset_path)
                if geometry_only:
                    generated_model = import_generated_glb(result.glb_bytes)
            else:
                variant_metadata = self._persist_object_texture_variants(
                    object_id,
                    texture_variants,
                )
                pipeline.update(
                    {
                        TEXTURE_VARIANTS_PIPELINE_KEY: variant_metadata,
                        SELECTED_TEXTURE_RESOLUTION_PIPELINE_KEY: (
                            DEFAULT_TEXTURE_RESOLUTION
                        ),
                    }
                )
                persisted_asset_paths.extend(
                    path
                    for variant in variant_metadata.values()
                    for path in variant.values()
                )
                asset_path = variant_metadata[
                    str(DEFAULT_TEXTURE_RESOLUTION)
                ][TEXTURE_VARIANT_GLB_PATH_KEY]
                generated_model = import_generated_glb(
                    texture_variants.glb_by_resolution[
                        DEFAULT_TEXTURE_RESOLUTION
                    ]
                )
            if isinstance(result, StagedMeshyGenerationResult):
                source_asset_path = self._persist_meshy_revision_asset(
                    object_id,
                    MESHY_REVISION_GEOMETRY,
                    result.source_glb_bytes,
                )
                persisted_asset_paths.append(source_asset_path)
                staged_pipeline: dict[str, object] = {
                    "mode": _staged_generation_mode(result),
                    "geometry_task_id": result.geometry_task_id,
                    "source_asset_path": source_asset_path,
                    "enabled_camera_ids": list(result.enabled_camera_ids),
                    "unchecked_camera_ids": list(
                        result.unchecked_camera_ids
                    ),
                    "camera_face_purge_applied": (
                        result.camera_face_purge_applied
                    ),
                    "unused_face_removal_applied": (
                        _staged_result_used_face_removal(result)
                    ),
                    "geometry_only": result.geometry_only,
                }
                if symmetry is None:
                    postprocessed_asset_path = (
                        self._persist_meshy_revision_asset(
                            object_id,
                            MESHY_REVISION_POSTPROCESSED,
                            result.postprocessed_glb_bytes,
                        )
                    )
                    persisted_asset_paths.append(postprocessed_asset_path)
                    staged_pipeline["postprocessed_asset_path"] = (
                        postprocessed_asset_path
                    )
                pipeline.update(staged_pipeline)
                if _staged_result_used_face_purge(result):
                    pipeline.update(
                        {
                            "purge_original_face_count": (
                                result.purge_original_face_count
                            ),
                            "purge_retained_face_count": (
                                result.purge_retained_face_count
                            ),
                            "purge_removed_face_count": (
                                result.purge_removed_face_count
                            ),
                        }
                    )
                if _staged_result_used_face_removal(result):
                    pipeline.update(
                        {
                            "original_face_count": result.original_face_count,
                            "retained_face_count": result.retained_face_count,
                            "removed_face_count": result.removed_face_count,
                            "protected_face_count": result.protected_face_count,
                        }
                    )
            if symmetry is not None:
                assert isinstance(
                    texture_variants,
                    SymmetricSquarePairTextureVariants,
                )
                pipeline = _build_automatic_symmetric_generation_pipeline(
                    pipeline,
                    symmetry,
                    variant_metadata,
                )
        except Exception as error:
            self._remove_newly_persisted_assets(persisted_asset_paths)
            self._handle_generation_failed(
                f"The Meshy texture variants could not be prepared locally: {error}",
                operation_id=operation_id,
            )
            return
        active_operation = (
            self._active_object_operation
            if runtime is None
            else runtime.operation
        )
        placement = (
            active_operation.pending_placement
            if active_operation is not None
            and active_operation.kind == OBJECT_OPERATION_GENERATE_MODEL
            else None
        )
        record = GeneratedObjectRecord(
            object_id=object_id,
            frame_index=(
                self._data.current_frame_index
                if generation_request is None
                else generation_request.frame_index
            ),
            object_name=object_name,
            pipeline=pipeline,
            provider=GENERATION_BACKEND_MESHY,
            provider_task_id=result.task_id,
            asset_path=asset_path,
            placement=placement,
        )
        self._data.generated_objects.append(record)
        self._cache_generated_model(record, generated_model)
        self._selected_object_id = object_id
        self._generated_model = generated_model
        self._refresh_generated_objects_list(object_id)
        self._record_operation_commit(
            OBJECT_OPERATION_GENERATE_MODEL,
            object_id,
            operation_id,
        )
        if isinstance(result, StagedMeshyGenerationResult):
            if result.geometry_only:
                self.status_label.setText(
                    _format_staged_generation_status(
                        object_name,
                        result,
                        generated_label="Generated geometry",
                    )
                )
            else:
                self.status_label.setText(
                    _format_staged_generation_status(object_name, result)
                )
        else:
            self.status_label.setText(f"Generated: {object_name}")
        if symmetry is not None:
            self.status_label.setText(
                f"{self.status_label.text()} Automatic symmetric division "
                f"kept the {symmetry.kept_side} half."
            )
        self._emit_data_changed()
        self.generation_completed.emit(record, generated_model)
        if runtime is not None and not runtime.operation.cancel_requested:
            self._complete_managed_job(
                runtime,
                f"Generated: {object_name}",
            )

    def _commit_saved_object_generation(
        self,
        saved: _SavedObjectGeneration,
        generation_request: GenerationRequest | None,
        runtime: _ObjectJobRuntime | None,
        operation_id: str | None,
    ) -> None:
        """Commit one worker-prepared model without doing local file work."""

        result = saved.result
        if not isinstance(result, MeshyGenerationResult):
            self._handle_generation_failed(
                "Meshy returned an invalid result.",
                operation_id=operation_id,
            )
            return
        object_name = (
            runtime.requested_name
            if runtime is not None and runtime.requested_name
            else result.name
        )
        active_operation = (
            self._active_object_operation
            if runtime is None
            else runtime.operation
        )
        placement = (
            active_operation.pending_placement
            if active_operation is not None
            and active_operation.kind == OBJECT_OPERATION_GENERATE_MODEL
            else None
        )
        record = GeneratedObjectRecord(
            object_id=saved.object_id,
            frame_index=(
                self._data.current_frame_index
                if generation_request is None
                else generation_request.frame_index
            ),
            object_name=object_name,
            pipeline=copy.deepcopy(saved.pipeline),
            provider=GENERATION_BACKEND_MESHY,
            provider_task_id=result.task_id,
            asset_path=saved.asset_path,
            placement=placement,
        )
        if (
            _build_generation_asset_revision(
                self._asset_directory,
                saved.asset_path,
            )
            != saved.preview_asset_revision
        ):
            self._handle_generation_failed(
                "The prepared object asset changed before it could be applied.",
                operation_id=operation_id,
            )
            return
        self._data.generated_objects.append(record)
        self._cache_generated_model(
            record,
            saved.preview_model,
            asset_revision=saved.preview_asset_revision,
        )
        self._selected_object_id = saved.object_id
        self._generated_model = saved.preview_model
        self._refresh_generated_objects_list(saved.object_id)
        self._record_operation_commit(
            OBJECT_OPERATION_GENERATE_MODEL,
            saved.object_id,
            operation_id,
        )
        if isinstance(result, StagedMeshyGenerationResult):
            status = _format_staged_generation_status(
                object_name,
                result,
                generated_label=(
                    "Generated geometry"
                    if result.geometry_only
                    else "Generated"
                ),
            )
        else:
            status = f"Generated: {object_name}"
        if saved.symmetry is not None:
            status += (
                " Automatic symmetric division kept the "
                f"{saved.symmetry.kept_side} half."
            )
        self.status_label.setText(status)
        self._emit_data_changed()
        self.generation_completed.emit(record, saved.preview_model)
        if runtime is not None and not runtime.operation.cancel_requested:
            manager = self._job_manager
            if (
                not runtime.requested_name
                and manager is not None
                and runtime.managed_job_id is not None
            ):
                manager.rename_job(runtime.managed_job_id, result.name)
            self._complete_managed_job(
                runtime,
                f"Generated: {object_name}",
            )

    @Slot(object, object)
    def _handle_texture_regeneration_succeeded(
        self,
        raw_outcome: object,
        generated_model: GeneratedModel,
        *,
        operation_id: str | None = None,
    ) -> None:
        if self._should_ignore_operation_result(
            OBJECT_OPERATION_GENERATE_TEXTURE,
            operation_id,
        ):
            return
        if isinstance(raw_outcome, _SavedObjectTextureRegeneration):
            self._commit_saved_texture_regeneration(
                raw_outcome,
                operation_id,
            )
            return
        if not isinstance(raw_outcome, TextureRegenerationOutcome):
            self._handle_texture_regeneration_failed(
                "Meshy returned an invalid texture-regeneration result.",
                operation_id=operation_id,
            )
            return
        outcome = raw_outcome
        request = outcome.request
        result = outcome.result
        record = self._find_generated_object_record(request.object_id)
        if record is None:
            self._handle_texture_regeneration_failed(
                "The target generated object no longer exists.",
                operation_id=operation_id,
            )
            return

        persisted_asset_paths: list[str] = []
        try:
            texture_variants = generated_model.object_texture_variants
            if texture_variants is None:
                raise ValueError(
                    "The regenerated model has no selectable texture variants."
                )
            symmetry = _get_object_symmetric_division_metadata(record)
            if symmetry is not None:
                _validate_symmetric_texture_regeneration_uvs(outcome)
                canonical_provider_glb = texture_variants.glb_by_resolution[
                    TEXTURE_RESOLUTION_2048
                ]
                texture_variants = _rebuild_symmetric_texture_variants(
                    canonical_provider_glb,
                    symmetry,
                )
            variant_metadata = self._persist_object_texture_variants(
                record.object_id,
                texture_variants,
                asset_stem=f"regenerated-{uuid.uuid4().hex}",
            )
            persisted_asset_paths.extend(
                path
                for variant in variant_metadata.values()
                for path in variant.values()
            )
            selected_resolution = _get_selected_texture_resolution(record)
            if selected_resolution not in _selectable_texture_resolutions(
                record
            ):
                selected_resolution = DEFAULT_TEXTURE_RESOLUTION
            selected_variant = variant_metadata[str(selected_resolution)]
            preview_model = import_generated_glb(
                texture_variants.glb_by_resolution[selected_resolution]
            )
            next_pipeline = _build_regenerated_texture_pipeline(
                record,
                outcome,
                variant_metadata,
            )
            replacement = replace(
                record,
                pipeline=_push_object_operation_undo_snapshot(
                    record,
                    next_pipeline,
                    operation=OBJECT_OPERATION_GENERATE_TEXTURE,
                ),
                provider_task_id=result.task_id,
                asset_path=selected_variant[TEXTURE_VARIANT_GLB_PATH_KEY],
            )
        except Exception as error:
            self._remove_newly_persisted_assets(persisted_asset_paths)
            self._handle_texture_regeneration_failed(
                "The new texture variants could not be saved locally: "
                f"{error}",
                operation_id=operation_id,
            )
            return

        if not self._request_object_packing_change(
            record,
            replacement,
            preview_model,
        ):
            self._remove_newly_persisted_assets(persisted_asset_paths)
            runtime = (
                None
                if operation_id is None
                else self._object_job_runtimes.get(str(operation_id))
            )
            self.status_label.setText(
                self._format_object_job_status(
                    runtime,
                    "The generated texture was kept out because the Atlas "
                    "packing change could not be committed.",
                )
            )
            if runtime is not None:
                self._fail_managed_job(runtime, self.status_label.text())
            return
        self._record_operation_commit(
            OBJECT_OPERATION_GENERATE_TEXTURE,
            record.object_id,
            operation_id,
        )
        cleanup_failed = self._delete_unreferenced_object_assets(record)
        status_suffix = (
            " Some superseded texture files could not be removed."
            if cleanup_failed
            else ""
        )
        runtime = (
            None
            if operation_id is None
            else self._object_job_runtimes.get(str(operation_id))
        )
        self.status_label.setText(
            self._format_object_job_status(
                runtime,
                f"Generated texture: {record.object_name}." + status_suffix,
            )
        )
        self._emit_data_changed()
        self.texture_regeneration_completed.emit(replacement, preview_model)
        self.generated_object_changed.emit(replacement, preview_model)
        if runtime is not None and not runtime.operation.cancel_requested:
            self._complete_managed_job(
                runtime,
                self.status_label.text(),
            )

    def _commit_saved_texture_regeneration(
        self,
        saved: _SavedObjectTextureRegeneration,
        operation_id: str | None,
    ) -> None:
        """Validate and commit one worker-prepared texture transaction."""

        outcome = saved.outcome
        if not isinstance(outcome, TextureRegenerationOutcome):
            self._handle_texture_regeneration_failed(
                "Meshy returned an invalid texture-regeneration result.",
                operation_id=operation_id,
            )
            return
        record = self._find_generated_object_record(
            outcome.request.object_id
        )
        if record is None:
            self._handle_texture_regeneration_failed(
                "The target generated object no longer exists.",
                operation_id=operation_id,
            )
            return
        if (
            (saved.source_asset_path is None)
            != (saved.source_asset_revision is None)
        ):
            self._handle_texture_regeneration_failed(
                "The prepared texture source revision is invalid.",
                operation_id=operation_id,
            )
            return
        if saved.source_asset_path is not None:
            current_source_asset_path = (
                _texture_regeneration_source_asset_path(record)
            )
            if (
                current_source_asset_path != saved.source_asset_path
                or _build_generation_asset_revision(
                    self._asset_directory,
                    current_source_asset_path,
                )
                != saved.source_asset_revision
            ):
                self._handle_texture_regeneration_failed(
                    "The target object's source model changed before its "
                    "generated texture could be applied.",
                    operation_id=operation_id,
                )
                return
        if (
            record.asset_path != saved.base_asset_path
            or record.provider_task_id != saved.base_provider_task_id
            or record.pipeline != saved.base_pipeline
        ):
            self._handle_texture_regeneration_failed(
                "The target object changed before its generated texture "
                "could be applied.",
                operation_id=operation_id,
            )
            return
        try:
            selected_variant = saved.variant_metadata[
                str(saved.selected_resolution)
            ]
            replacement = replace(
                record,
                pipeline=_push_object_operation_undo_snapshot(
                    record,
                    copy.deepcopy(saved.next_pipeline),
                    operation=OBJECT_OPERATION_GENERATE_TEXTURE,
                ),
                provider_task_id=outcome.result.task_id,
                asset_path=selected_variant[TEXTURE_VARIANT_GLB_PATH_KEY],
            )
        except Exception as error:
            self._handle_texture_regeneration_failed(
                f"The prepared texture transaction is invalid: {error}",
                operation_id=operation_id,
            )
            return

        runtime = (
            None
            if operation_id is None
            else self._object_job_runtimes.get(str(operation_id))
        )
        if not self._request_object_packing_change(
            record,
            replacement,
            saved.preview_model,
            preview_asset_revision=saved.preview_asset_revision,
        ):
            status = self._format_object_job_status(
                runtime,
                "The generated texture was kept out because the Atlas "
                "packing change could not be committed.",
            )
            self.status_label.setText(status)
            if runtime is not None:
                self._fail_managed_job(runtime, status)
            return
        self._record_operation_commit(
            OBJECT_OPERATION_GENERATE_TEXTURE,
            record.object_id,
            operation_id,
        )
        cleanup_failed = self._delete_unreferenced_object_assets(record)
        status_suffix = (
            " Some superseded texture files could not be removed."
            if cleanup_failed
            else ""
        )
        status = self._format_object_job_status(
            runtime,
            f"Generated texture: {record.object_name}." + status_suffix,
        )
        self.status_label.setText(status)
        self._emit_data_changed()
        self.texture_regeneration_completed.emit(
            replacement,
            saved.preview_model,
        )
        self.generated_object_changed.emit(
            replacement,
            saved.preview_model,
        )
        if runtime is not None and not runtime.operation.cancel_requested:
            self._complete_managed_job(runtime, status)

    @Slot(object)
    def _handle_unchecked_camera_face_purge_succeeded(
        self,
        raw_outcome: object,
        *,
        operation_id: str | None = None,
    ) -> None:
        if self._should_ignore_operation_result(
            OBJECT_OPERATION_PURGE_FACES,
            operation_id,
        ):
            return
        if not isinstance(raw_outcome, UncheckedCameraFacePurgeOutcome):
            self._handle_unchecked_camera_face_purge_failed(
                "The face-purge worker returned an invalid result.",
                operation_id=operation_id,
            )
            return
        outcome = raw_outcome
        record = self._find_generated_object_record(outcome.request.object_id)
        if record is None:
            self._handle_unchecked_camera_face_purge_failed(
                "The target generated object no longer exists.",
                operation_id=operation_id,
            )
            return

        persisted_asset_paths: list[str] = []
        try:
            symmetry = _get_object_symmetric_division_metadata(record)
            if (
                symmetry is None
                or symmetry.version == SYMMETRIC_DIVISION_METADATA_VERSION
            ):
                # V1 projects historically retain three physical resolutions.
                texture_variants = build_object_texture_variants(
                    outcome.result.glb_bytes
                )
            else:
                texture_variants = _rebuild_symmetric_texture_variants(
                    outcome.result.glb_bytes,
                    symmetry,
                )
            if texture_variants is None:
                if isinstance(
                    record.pipeline.get(TEXTURE_VARIANTS_PIPELINE_KEY),
                    dict,
                ):
                    raise ValueError(
                        "The purged model unexpectedly lost its texture."
                    )
                asset_path = self._persist_meshy_named_asset(
                    f"purged-{uuid.uuid4().hex}.glb",
                    outcome.result.glb_bytes,
                )
                postprocessed_asset_path = asset_path
                persisted_asset_paths.append(asset_path)
                variant_metadata = None
                preview_model = import_generated_glb(outcome.result.glb_bytes)
            else:
                variant_metadata = self._persist_object_texture_variants(
                    record.object_id,
                    texture_variants,
                    asset_stem=f"purged-{uuid.uuid4().hex}",
                )
                persisted_asset_paths.extend(
                    path
                    for variant in variant_metadata.values()
                    for path in variant.values()
                )
                selected_resolution = _get_selected_texture_resolution(record)
                if selected_resolution not in _selectable_texture_resolutions(
                    record
                ):
                    selected_resolution = DEFAULT_TEXTURE_RESOLUTION
                selected_variant = variant_metadata[str(selected_resolution)]
                asset_path = selected_variant[TEXTURE_VARIANT_GLB_PATH_KEY]
                postprocessed_asset_path = variant_metadata[
                    str(_canonical_texture_resolution(record))
                ][TEXTURE_VARIANT_GLB_PATH_KEY]
                preview_model = import_generated_glb(
                    texture_variants.glb_by_resolution[selected_resolution]
                )
            next_pipeline = _build_unchecked_camera_face_purge_pipeline(
                record,
                outcome,
                variant_metadata,
                postprocessed_asset_path=postprocessed_asset_path,
            )
            replacement = replace(
                record,
                pipeline=_push_object_operation_undo_snapshot(
                    record,
                    next_pipeline,
                    operation=OBJECT_OPERATION_PURGE_FACES,
                ),
                asset_path=asset_path,
            )
        except Exception as error:
            self._remove_newly_persisted_assets(persisted_asset_paths)
            self._handle_unchecked_camera_face_purge_failed(
                f"The purged object could not be saved locally: {error}",
                operation_id=operation_id,
            )
            return

        if not self._request_object_packing_change(
            record,
            replacement,
            preview_model,
        ):
            self._remove_newly_persisted_assets(persisted_asset_paths)
            self.status_label.setText(
                "The purged object was kept out because the Atlas packing "
                "change could not be committed."
            )
            runtime = (
                None
                if operation_id is None
                else self._object_job_runtimes.get(str(operation_id))
            )
            if runtime is not None:
                self._fail_managed_job(runtime, self.status_label.text())
            return
        self._record_operation_commit(
            OBJECT_OPERATION_PURGE_FACES,
            record.object_id,
            operation_id,
        )
        cleanup_failed = self._delete_unreferenced_object_assets(record)
        status_suffix = (
            " Some superseded files could not be removed."
            if cleanup_failed
            else ""
        )
        self.status_label.setText(
            f"Purged {outcome.result.removed_face_count} faces from "
            f"{record.object_name}." + status_suffix
        )
        self._emit_data_changed()
        self.face_purge_completed.emit(replacement, preview_model)
        self.generated_object_changed.emit(replacement, preview_model)
        runtime = (
            None
            if operation_id is None
            else self._object_job_runtimes.get(str(operation_id))
        )
        if runtime is not None and not runtime.operation.cancel_requested:
            self._complete_managed_job(
                runtime,
                f"Purged faces: {record.object_name}",
            )

    @Slot(str)
    def _handle_generation_failed(
        self,
        error_message: str,
        *,
        operation_id: str | None = None,
    ) -> None:
        if self._should_ignore_operation_result(
            OBJECT_OPERATION_GENERATE_MODEL,
            operation_id,
        ):
            return
        runtime = (
            None
            if operation_id is None
            else self._object_job_runtimes.get(str(operation_id))
        )
        if operation_id is None:
            self._active_generation_request = None
        operation = (
            self._active_object_operation
            if runtime is None
            else runtime.operation
        )
        if operation is not None:
            operation.pending_placement = None
        self.status_label.setText(f"Generation failed: {error_message}")
        if runtime is not None:
            self._fail_managed_job(runtime, f"Failed: {error_message}")
        if self._job_manager is None:
            QMessageBox.warning(self, "Generation failed", error_message)

    @Slot(str, str)
    def _handle_job_generation_failed(
        self,
        operation_id: str,
        error_message: str,
    ) -> None:
        runtime = self._object_job_runtimes.get(str(operation_id))
        if runtime is None:
            return
        self._set_legacy_active_job_runtime(runtime)
        self._handle_generation_failed(
            str(error_message),
            operation_id=operation_id,
        )

    @Slot(str)
    def _handle_texture_regeneration_failed(
        self,
        error_message: str,
        *,
        operation_id: str | None = None,
    ) -> None:
        if self._should_ignore_operation_result(
            OBJECT_OPERATION_GENERATE_TEXTURE,
            operation_id,
        ):
            return
        runtime = (
            None
            if operation_id is None
            else self._object_job_runtimes.get(str(operation_id))
        )
        self.status_label.setText(
            self._format_object_job_status(
                runtime,
                f"Texture generation failed: {error_message}",
            )
        )
        if runtime is not None:
            self._fail_managed_job(runtime, self.status_label.text())
        if self._job_manager is None:
            QMessageBox.warning(
                self,
                "Texture generation failed",
                error_message,
            )

    @Slot(str, str)
    def _handle_job_texture_regeneration_failed(
        self,
        operation_id: str,
        error_message: str,
    ) -> None:
        runtime = self._object_job_runtimes.get(str(operation_id))
        if runtime is None:
            return
        self._set_legacy_active_job_runtime(runtime)
        self._handle_texture_regeneration_failed(
            str(error_message),
            operation_id=operation_id,
        )

    @Slot(str)
    def _handle_unchecked_camera_face_purge_failed(
        self,
        error_message: str,
        *,
        operation_id: str | None = None,
    ) -> None:
        if self._should_ignore_operation_result(
            OBJECT_OPERATION_PURGE_FACES,
            operation_id,
        ):
            return
        self.status_label.setText(f"Face purge failed: {error_message}")
        runtime = (
            None
            if operation_id is None
            else self._object_job_runtimes.get(str(operation_id))
        )
        if runtime is not None:
            self._fail_managed_job(runtime, f"Failed: {error_message}")
        if self._job_manager is None:
            QMessageBox.warning(self, "Face purge failed", error_message)

    @Slot(str, str)
    def _handle_job_face_purge_failed(
        self,
        operation_id: str,
        error_message: str,
    ) -> None:
        runtime = self._object_job_runtimes.get(str(operation_id))
        if runtime is None:
            return
        self._set_legacy_active_job_runtime(runtime)
        self._handle_unchecked_camera_face_purge_failed(
            str(error_message),
            operation_id=operation_id,
        )

    def _rollback_cancelled_operation(
        self,
        operation: _ActiveObjectOperation,
        *,
        operation_id: str | None = None,
    ) -> bool:
        object_id = operation.committed_object_id
        if object_id is None:
            return True
        if operation.kind == OBJECT_OPERATION_GENERATE_MODEL:
            if self._find_generated_object_record(object_id) is None:
                return True
            return self.delete_generated_object(
                object_id,
                allow_operation_id=operation_id,
            )
        if operation.kind not in {
            OBJECT_OPERATION_GENERATE_TEXTURE,
            OBJECT_OPERATION_PURGE_FACES,
        }:
            return False
        if operation.target_object_id != object_id:
            return False
        return self._undo_object_change(
            object_id,
            expected_operation=operation.kind,
            allow_operation_id=operation_id,
        )

    def _set_cancelled_operation_status(
        self,
        operation: _ActiveObjectOperation,
        rollback_succeeded: bool,
    ) -> None:
        had_commit = operation.committed_object_id is not None
        if not rollback_succeeded:
            self.status_label.setText(
                "The operation was cancelled, but its completed local "
                "changes could not be rolled back."
            )
            return
        if operation.kind == OBJECT_OPERATION_GENERATE_MODEL:
            suffix = (
                " The generated model was deleted."
                if had_commit
                else ""
            )
            self.status_label.setText("Model generation cancelled." + suffix)
            return
        if operation.kind == OBJECT_OPERATION_GENERATE_TEXTURE:
            outcome = (
                "The generated texture was deleted and the previous texture "
                "was restored."
                if had_commit
                else "The existing texture was kept."
            )
            self.status_label.setText("Texture generation cancelled. " + outcome)
            return
        outcome = (
            "The generated face revision was deleted and the previous model "
            "was restored."
            if had_commit
            else "The existing model was kept."
        )
        self.status_label.setText("Face purge cancelled. " + outcome)

    @Slot()
    def _handle_generation_thread_finished(self) -> None:
        """Compatibility finish handler for a legacy single worker."""

        runtime = self._legacy_active_job_runtime()
        if runtime is None or self.sender() is not runtime.thread:
            return
        self._handle_object_job_thread_finished(runtime.operation_id)

    @Slot(str)
    def _handle_object_job_thread_finished(self, operation_id: str) -> None:
        runtime = self._object_job_runtimes.get(str(operation_id))
        if runtime is None:
            return
        worker = runtime.worker
        if isinstance(worker, GenerationWorker | TextureRegenerationWorker):
            worker.discard_unclaimed_output()
        operation = runtime.operation
        if operation is not None and operation.cancel_requested:
            rollback_succeeded = self._rollback_cancelled_operation(
                operation,
                operation_id=runtime.operation_id,
            )
            self._set_cancelled_operation_status(
                operation,
                rollback_succeeded,
            )
            self._mark_managed_job_cancelled(
                runtime,
                self.status_label.text(),
            )
            self.operation_cancelled.emit(
                operation.kind,
                operation.target_object_id,
            )
        if operation is not None:
            operation.pending_placement = None
            self.operation_finished.emit(operation.operation_id)
        manager = self._job_manager
        if (
            not operation.cancel_requested
            and manager is not None
            and runtime.managed_job_id is not None
        ):
            managed_job = manager.get_job(runtime.managed_job_id)
            if managed_job is not None and not managed_job.is_finished:
                self._fail_managed_job(
                    runtime,
                    "The job ended before its result could be committed.",
                )
        self._object_job_runtimes.pop(runtime.operation_id, None)
        runtime.relay.deleteLater()
        self._set_legacy_active_job_runtime(
            self._legacy_active_job_runtime()
        )
        self._sync_controls()

    def _build_generation_request(
        self,
        *,
        geometry_only: bool = False,
    ) -> GenerationRequest | None:
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
        symmetric_division_enabled = (
            not geometry_only
            and self.symmetric_division_checkbox.isChecked()
        )
        symmetric_division_orientation = str(
            self.symmetric_division_orientation_combo.currentData()
            or SYMMETRIC_DIVISION_ORIENTATION_VERTICAL
        )
        return GenerationRequest(
            frame_index=self._data.current_frame_index,
            selected_object_bgra=selected_crop,
            settings=self._settings,
            enabled_camera_ids=(
                self.object_3d_panel.get_enabled_postprocess_camera_ids()
            ),
            geometry_only=geometry_only,
            symmetric_division_enabled=symmetric_division_enabled,
            symmetric_division_orientation=(
                symmetric_division_orientation
            ),
        )

    def _build_unchecked_camera_face_purge_request(
        self,
    ) -> UncheckedCameraFacePurgeRequest | None:
        record = self._find_generated_object_record(self._selected_object_id)
        if record is None:
            self.status_label.setText(
                "Select a generated object before purging faces."
            )
            return None
        enabled_camera_ids = set(
            self.object_3d_panel.get_enabled_postprocess_camera_ids()
        )
        unchecked_camera_ids = tuple(
            camera_id
            for camera_id in ALL_CAMERA_IDS
            if camera_id not in enabled_camera_ids
        )
        if not unchecked_camera_ids:
            self.status_label.setText(
                "Uncheck at least one camera before purging faces."
            )
            return None
        try:
            variant = self.get_texture_variant(
                record.object_id,
                _canonical_texture_resolution(record),
            )
            source_path = (
                self._resolve_texture_regeneration_source_path(record)
                if variant is None
                else variant.glb_asset_path
            )
            model_glb = source_path.read_bytes()
            import_generated_glb(model_glb)
            return UncheckedCameraFacePurgeRequest(
                object_id=record.object_id,
                model_glb=model_glb,
                unchecked_camera_ids=unchecked_camera_ids,
            )
        except Exception as error:
            self.status_label.setText(f"Face purge could not start: {error}")
            return None

    def _build_texture_regeneration_request(
        self,
    ) -> _TextureRegenerationPreflight | None:
        record = self._find_generated_object_record(self._selected_object_id)
        if not self._can_regenerate_object_texture(record):
            self.status_label.setText(
                "Select a generated object and paint a current video reference "
                "before generating its texture."
            )
            return None
        assert record is not None
        selected_crop = self.video_view.build_selected_object_crop()
        if selected_crop.size == 0:
            self.status_label.setText("The selected texture reference is empty.")
            return None
        try:
            source_asset_path = _texture_regeneration_source_asset_path(
                record
            )
            source_asset_revision = _build_generation_asset_revision(
                self._asset_directory,
                source_asset_path,
            )
            if source_asset_revision[1] is None:
                raise ValueError(
                    "The model revision used for Retexture is missing or unsafe."
                )
            preserve_symmetric_uvs = (
                _get_object_symmetric_division_metadata(record) is not None
            )
            request = _TextureRegenerationPreflight(
                object_id=record.object_id,
                reference_frame_index=self._data.current_frame_index,
                reference_image_bgra=selected_crop,
                source_asset_path=source_asset_path,
                source_asset_revision=source_asset_revision,
                settings=self._settings,
                enable_original_uv=preserve_symmetric_uvs,
                preserve_symmetric_uvs=preserve_symmetric_uvs,
            )
        except Exception as error:
            self.status_label.setText(
                f"Texture generation could not start: {error}"
            )
            return None
        self._store_current_frame_strokes()
        return request

    def _resolve_texture_regeneration_source_path(
        self,
        record: GeneratedObjectRecord,
    ) -> Path:
        """Resolve processed geometry, or a canonical existing textured GLB."""

        source_path = self._resolve_meshy_asset_path(
            _texture_regeneration_source_asset_path(record)
        )
        if not source_path.is_file():
            raise ValueError("The model revision used for Retexture is missing.")
        return source_path

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
        has_untracked_legacy_job = (
            self._generation_thread is not None
            and not self._object_job_runtimes
        )
        has_mask = self.video_view.has_selection()
        selected_record = self._find_generated_object_record(
            self._selected_object_id
        )
        selected_object_is_busy = bool(
            selected_record is not None
            and self._object_has_active_mutation_job(
                selected_record.object_id
            )
        )
        self.load_video_button.setEnabled(not has_untracked_legacy_job)
        self.seekbar.setEnabled(has_video and not has_untracked_legacy_job)
        mask_tool_is_available = has_video and not has_untracked_legacy_job
        self.paint_mask_button.setEnabled(mask_tool_is_available)
        self.erase_mask_button.setEnabled(mask_tool_is_available)
        self.brush_size_spinbox.setEnabled(mask_tool_is_available)
        self.undo_mask_button.setEnabled(
            has_video
            and bool(self.video_view.get_strokes())
            and not has_untracked_legacy_job
        )
        self.clear_mask_button.setEnabled(
            has_video and has_mask and not has_untracked_legacy_job
        )
        required_key_is_available = bool(self._settings.meshy_api_key)
        enabled_camera_ids = (
            self.object_3d_panel.get_enabled_postprocess_camera_ids()
        )
        postprocessing_is_enabled = self._settings.unused_face_removal
        camera_selection_is_valid = (
            not postprocessing_is_enabled or bool(enabled_camera_ids)
        )
        self.object_3d_panel.set_postprocess_camera_controls_enabled(
            not has_untracked_legacy_job
        )
        self.result_view.set_enabled_unused_face_camera_ids(
            enabled_camera_ids
        )
        self.result_view.set_unused_face_camera_indicators_visible(
            True
        )
        self.meshy_target_polycount_control.setVisible(True)
        self.meshy_target_polycount_spinbox.setEnabled(
            not has_untracked_legacy_job
        )
        self.ambient_light_slider.setEnabled(not has_untracked_legacy_job)
        self.textures_checkbox.setEnabled(not has_untracked_legacy_job)
        self.wireframe_checkbox.setEnabled(not has_untracked_legacy_job)
        self.generated_objects_list.setEnabled(not has_untracked_legacy_job)
        self.texture_view.setEnabled(not has_untracked_legacy_job)
        self.delete_generated_object_button.setEnabled(
            selected_record is not None
            and not selected_object_is_busy
        )
        self.regenerate_texture_button.setEnabled(
            self._can_regenerate_object_texture(selected_record)
        )
        self.undo_object_change_button.setEnabled(
            not selected_object_is_busy
            and bool(
                _get_object_operation_undo_stack(
                    selected_record
                )
            )
        )
        self.place_object_button.setEnabled(
            any(
                self._can_place_active_operation(runtime.operation)
                for runtime in self._object_job_runtimes.values()
            )
            or (
                selected_record is not None
                and not selected_object_is_busy
            )
        )
        self.cancel_operation_button.setEnabled(
            self._active_object_operation is not None
            and self._legacy_active_job_runtime() is not None
            and not self._active_object_operation.cancel_requested
        )
        self.purge_faces_button.setEnabled(
            len(enabled_camera_ids) < len(ALL_CAMERA_IDS)
            and selected_record is not None
            and not selected_object_is_busy
            and not has_untracked_legacy_job
        )
        self.symmetric_division_checkbox.setEnabled(
            not has_untracked_legacy_job
        )
        self.symmetric_division_orientation_combo.setEnabled(
            not has_untracked_legacy_job
            and self.symmetric_division_checkbox.isChecked()
        )
        self.job_name_edit.setEnabled(not has_untracked_legacy_job)
        self.generate_button.setEnabled(
            has_video
            and has_mask
            and required_key_is_available
            and camera_selection_is_valid
            and not has_untracked_legacy_job
        )
        self.generate_geometry_button.setEnabled(
            has_video
            and has_mask
            and required_key_is_available
            and camera_selection_is_valid
            and not self.symmetric_division_checkbox.isChecked()
            and not has_untracked_legacy_job
        )
        self.video_view.set_interaction_enabled(
            has_video and not has_untracked_legacy_job
        )

    def _can_regenerate_object_texture(
        self,
        record: GeneratedObjectRecord | None,
    ) -> bool:
        if (
            record is None
            or self._object_has_active_mutation_job(record.object_id)
            or (
                self._generation_thread is not None
                and not self._object_job_runtimes
            )
            or not self._settings.meshy_api_key
            or self._video_source is None
            or self.video_view.get_frame_bgr() is None
            or not self.video_view.has_selection()
            or record.provider != GENERATION_BACKEND_MESHY
        ):
            return False
        return True

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
            self._sync_controls()
            return
        object_id = str(current_item.data(OBJECT_ID_ITEM_ROLE) or "")
        record = self._find_generated_object_record(object_id)
        if record is None:
            self._selected_object_id = None
            self._clear_generated_object_display()
            self._sync_controls()
            return
        self._selected_object_id = object_id
        self._display_generated_object(record)
        self._sync_controls()

    def _rebuild_generated_objects(self) -> None:
        preferred_id = self._selected_object_id
        if self._find_generated_object_record(preferred_id) is None:
            preferred_id = (
                None
                if not self._data.generated_objects
                else self._data.generated_objects[-1].object_id
            )
        self._refresh_generated_objects_list(
            preferred_id,
            repair_missing_variant=True,
        )

    def _refresh_generated_objects_list(
        self,
        selected_object_id: str | None,
        *,
        repair_missing_variant: bool = False,
    ) -> None:
        object_list = self.generated_objects_list
        expected_rows = _build_generated_object_list_rows(
            self._data.generated_objects
        )
        current_rows = tuple(
            (
                str(
                    object_list.item(row_index).data(
                        OBJECT_ID_ITEM_ROLE
                    )
                    or ""
                ),
                object_list.item(row_index).text(),
                object_list.item(row_index).toolTip(),
            )
            for row_index in range(object_list.count())
        )
        if current_rows != expected_rows:
            was_blocked = object_list.blockSignals(True)
            try:
                object_list.clear()
                for object_id, label, tooltip in expected_rows:
                    item = QListWidgetItem(label)
                    item.setData(OBJECT_ID_ITEM_ROLE, object_id)
                    if tooltip:
                        item.setToolTip(tooltip)
                    object_list.addItem(item)
            finally:
                object_list.blockSignals(was_blocked)

        selected_row = next(
            (
                row_index
                for row_index in range(object_list.count())
                if str(
                    object_list.item(row_index).data(
                        OBJECT_ID_ITEM_ROLE
                    )
                    or ""
                )
                == selected_object_id
            ),
            -1,
        )
        if object_list.currentRow() != selected_row:
            was_blocked = object_list.blockSignals(True)
            try:
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
        self._display_generated_object(
            record,
            repair_missing_variant=repair_missing_variant,
        )

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
        *,
        repair_missing_variant: bool = True,
    ) -> None:
        if repair_missing_variant:
            record = self._repair_missing_active_texture_variant(record)
        next_snapshot = _build_generated_object_display_snapshot(
            record,
            self._asset_directory,
        )
        if (
            next_snapshot == self._displayed_object_snapshot
            and self._generated_model is not None
            and self.result_view.model is self._generated_model
        ):
            self._refresh_object_texture_atlases(record.object_id)
            return
        self._displayed_object_snapshot = None
        self._generated_model = None
        self.result_view.clear_model()
        self._sync_model_statistics(None)
        self.texture_view.set_uv_overlay_triangles(())
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
        symmetry = _get_object_symmetric_division_metadata(record)
        self.result_view.set_symmetric_division_preview(
            None if symmetry is None else symmetry.orientation,
            None if symmetry is None else symmetry.plane_coordinate,
        )
        self._sync_model_statistics(generated_model)
        self.texture_view.set_uv_overlay_triangles(
            _collect_model_uv_triangles(generated_model)
        )
        self._refresh_object_texture_atlases(
            record.object_id,
        )
        self._displayed_object_snapshot = next_snapshot

    def _repair_missing_active_texture_variant(
        self,
        record: GeneratedObjectRecord,
    ) -> GeneratedObjectRecord:
        """Select the first complete variant if the saved active one vanished."""

        raw_variants = record.pipeline.get(TEXTURE_VARIANTS_PIPELINE_KEY)
        if not isinstance(raw_variants, dict):
            return record
        selectable_resolutions = _selectable_texture_resolutions(record)
        selected_resolution = _get_selected_texture_resolution(record)
        if (
            selected_resolution in selectable_resolutions
            and self.get_active_texture_variant(record.object_id) is not None
        ):
            return record
        missing_resolution = selected_resolution
        fallback_resolutions = sorted(
            selectable_resolutions,
            key=lambda resolution: (
                resolution != DEFAULT_TEXTURE_RESOLUTION,
                abs(resolution - missing_resolution),
                resolution,
            ),
        )
        for resolution in fallback_resolutions:
            if self.get_texture_variant(record.object_id, resolution) is None:
                continue
            variant = _get_texture_variant_metadata(record, resolution)
            if variant is None:
                continue
            pipeline = dict(record.pipeline)
            pipeline[SELECTED_TEXTURE_RESOLUTION_PIPELINE_KEY] = resolution
            replacement = replace(
                record,
                pipeline=pipeline,
                asset_path=variant[TEXTURE_VARIANT_GLB_PATH_KEY],
            )
            record_index = self._data.generated_objects.index(record)
            self._data.generated_objects[record_index] = replacement
            self._generated_model_cache.pop(record.object_id, None)
            self._generated_model_cache_revisions.pop(record.object_id, None)
            if (
                not self._is_rebuilding_generation_data
                and not self._is_emitting_texture_repair
            ):
                self._is_emitting_texture_repair = True
                try:
                    self._emit_data_changed()
                finally:
                    self._is_emitting_texture_repair = False
            return replacement
        return record

    def _clear_generated_object_display(self) -> None:
        if (
            self._displayed_object_snapshot is None
            and self._generated_model is None
            and self.result_view.model is None
            and not self.texture_view.entries
        ):
            return
        self._displayed_object_snapshot = None
        self._generated_model = None
        self.result_view.clear_model()
        self._sync_model_statistics(None)
        self.texture_view.clear()

    def _refresh_object_texture_atlases(
        self,
        selected_object_id: str,
    ) -> None:
        record = self._find_generated_object_record(selected_object_id)
        if record is None:
            self.texture_view.clear()
            return
        entry_signature = _build_texture_resolution_entry_signature(
            record,
            self._asset_directory,
        )
        cached_entries = self._texture_resolution_entry_cache.get(
            record.object_id
        )
        expected_entry_count = _count_available_texture_resolution_entries(
            record,
            self._asset_directory,
        )
        if (
            cached_entries is not None
            and cached_entries[0] == entry_signature
            and len(cached_entries[1]) == expected_entry_count
        ):
            entries = list(cached_entries[1])
        else:
            entries = _build_texture_resolution_entries(
                record,
                self._asset_directory,
            )
            if len(entries) == expected_entry_count:
                self._texture_resolution_entry_cache[record.object_id] = (
                    entry_signature,
                    tuple(entries),
                )
            else:
                self._texture_resolution_entry_cache.pop(
                    record.object_id,
                    None,
                )
        selected_resolution = _get_selected_texture_resolution(record)
        selected_entry_id = f"{record.object_id}:resolution:{selected_resolution}"
        if not any(entry.atlas_id == selected_entry_id for entry in entries):
            selected_entry_id = entries[0].atlas_id if entries else None

        self._is_syncing_texture_resolution_view = True
        try:
            if tuple(entries) != self.texture_view.entries:
                self.texture_view.set_atlases(
                    entries,
                    selected_atlas_id=selected_entry_id,
                )
            else:
                self.texture_view.select_atlas(selected_entry_id)
        finally:
            self._is_syncing_texture_resolution_view = False

    @Slot(object)
    def _handle_texture_resolution_selected(self, raw_entry: object) -> None:
        if self._is_syncing_texture_resolution_view:
            return
        if not isinstance(raw_entry, TextureAtlasEntry):
            return
        record = self._find_generated_object_record(raw_entry.owner_id)
        if record is None:
            return
        resolution = _parse_texture_resolution_entry_id(
            record.object_id,
            raw_entry.atlas_id,
        )
        if _get_selected_texture_resolution(record) == resolution:
            return
        if self._request_global_texture_resolution_change(
            record.object_id,
            resolution,
        ):
            return
        self.status_label.setText(
            "The selected object texture resolution could not be applied "
            "globally; the previous resolution was kept."
        )
        self._refresh_object_texture_atlases(record.object_id)

    def _request_global_texture_resolution_change(
        self,
        object_id: str,
        resolution: int,
    ) -> bool:
        """Commit through the host transaction, or locally when standalone."""

        handler = self._texture_resolution_change_handler
        if handler is None:
            return self.select_object_texture_resolution(
                object_id,
                resolution,
            )
        try:
            return bool(handler(object_id, resolution))
        except Exception:
            return False

    def _request_object_packing_change(
        self,
        record: GeneratedObjectRecord,
        replacement: GeneratedObjectRecord,
        preview_model: GeneratedModel,
        *,
        preview_asset_revision: tuple[object, ...] | None = None,
    ) -> bool:
        """Commit one prepared packing transition with host rollback."""

        try:
            record_index = self._data.generated_objects.index(record)
        except ValueError:
            return False
        selected_object_id = self._selected_object_id
        cached_model_was_present = record.object_id in self._generated_model_cache
        cached_model = self._generated_model_cache.get(record.object_id)
        cached_model_revision = self._generated_model_cache_revisions.get(
            record.object_id
        )
        is_committed = False

        def restore_record() -> bool:
            nonlocal is_committed
            if not is_committed:
                return True
            self._data.generated_objects[record_index] = record
            self._generated_model_cache.pop(record.object_id, None)
            self._generated_model_cache_revisions.pop(record.object_id, None)
            if cached_model_was_present and cached_model is not None:
                self._generated_model_cache[record.object_id] = cached_model
                if cached_model_revision is not None:
                    self._generated_model_cache_revisions[record.object_id] = (
                        cached_model_revision
                    )
            is_committed = False
            try:
                self._refresh_generated_objects_list(selected_object_id)
            except Exception:
                return False
            return True

        def commit_record() -> bool:
            nonlocal is_committed
            if is_committed:
                return True
            current_record = self._find_generated_object_record(
                record.object_id
            )
            if current_record is not record:
                return False
            if (
                preview_asset_revision is not None
                and _build_generation_asset_revision(
                    self._asset_directory,
                    replacement.asset_path,
                )
                != preview_asset_revision
            ):
                return False
            self._data.generated_objects[record_index] = replacement
            self._generated_model_cache.pop(record.object_id, None)
            self._generated_model_cache_revisions.pop(record.object_id, None)
            if selected_object_id == record.object_id:
                self._cache_generated_model(
                    replacement,
                    preview_model,
                    asset_revision=preview_asset_revision,
                )
            is_committed = True
            try:
                self._refresh_generated_objects_list(selected_object_id)
            except Exception:
                restore_record()
                return False
            return True

        handler = self._object_packing_change_handler
        packing_change_requires_host = (
            _get_object_symmetric_division_metadata(record) is not None
            or _get_object_symmetric_division_metadata(replacement) is not None
        )
        if handler is None or not packing_change_requires_host:
            return commit_record()
        try:
            accepted = bool(
                handler(
                    record,
                    replacement,
                    preview_model,
                    commit_record,
                )
            )
            if accepted and not is_committed:
                accepted = commit_record()
        except Exception:
            accepted = False
        if accepted and is_committed:
            return True
        restore_record()
        return False

    def _load_generated_object_model(
        self,
        record: GeneratedObjectRecord,
    ) -> GeneratedModel:
        for _attempt_index in range(2):
            asset_revision_before = _build_generation_asset_revision(
                self._asset_directory,
                record.asset_path,
            )
            generated_model = self._generated_model_cache.get(record.object_id)
            if (
                generated_model is not None
                and self._generated_model_cache_revisions.get(record.object_id)
                == asset_revision_before
            ):
                return generated_model
            asset_path = self._resolve_meshy_asset_path(record.asset_path)
            generated_model = import_generated_glb(asset_path.read_bytes())
            asset_revision_after = _build_generation_asset_revision(
                self._asset_directory,
                record.asset_path,
            )
            if asset_revision_before != asset_revision_after:
                continue
            self._cache_generated_model(
                record,
                generated_model,
                asset_revision=asset_revision_after,
            )
            return generated_model
        raise OSError("The generated GLB changed repeatedly while loading.")

    def _cache_generated_model(
        self,
        record: GeneratedObjectRecord,
        model: GeneratedModel,
        *,
        asset_revision: tuple[object, ...] | None = None,
    ) -> None:
        """Cache one model only for the exact active on-disk GLB revision."""

        if asset_revision is None:
            for _attempt_index in range(2):
                revision_before = _build_generation_asset_revision(
                    self._asset_directory,
                    record.asset_path,
                )
                try:
                    asset_path = self._resolve_meshy_asset_path(
                        record.asset_path
                    )
                    persisted_payload = asset_path.read_bytes()
                except (OSError, RuntimeError, ValueError):
                    break
                revision_after = _build_generation_asset_revision(
                    self._asset_directory,
                    record.asset_path,
                )
                if revision_before != revision_after:
                    continue
                if persisted_payload != bytes(model.glb_bytes):
                    break
                asset_revision = revision_after
                break
        if asset_revision is None:
            self._generated_model_cache.pop(record.object_id, None)
            self._generated_model_cache_revisions.pop(record.object_id, None)
            return
        self._generated_model_cache[record.object_id] = model
        self._generated_model_cache_revisions[record.object_id] = asset_revision

    def _sync_model_statistics(self, model: GeneratedModel | None) -> None:
        self.model_statistics_label.setText(_format_model_statistics(model))

    def _persist_meshy_asset(self, object_id: str, glb_bytes: bytes) -> str:
        return self._persist_meshy_named_asset(
            f"{object_id}.glb",
            glb_bytes,
        )

    def _persist_meshy_revision_asset(
        self,
        object_id: str,
        revision: str,
        glb_bytes: bytes,
    ) -> str:
        if revision not in MESHY_REVISION_NAMES:
            raise ValueError("Unknown Meshy revision name.")
        return self._persist_meshy_named_asset(
            f"{object_id}.{revision}.glb",
            glb_bytes,
        )

    def _persist_object_texture_variants(
        self,
        object_id: str,
        variants: PersistableObjectTextureVariants,
        *,
        asset_stem: str | None = None,
    ) -> dict[str, dict[str, str]]:
        """Atomically persist each selectable GLB and its atlas-ready PNG."""

        return _persist_object_texture_variants_to_directory(
            self._asset_directory,
            object_id,
            variants,
            asset_stem=asset_stem,
            persist_asset=self._persist_meshy_named_asset,
        )

    def _remove_newly_persisted_assets(self, raw_paths: Sequence[str]) -> None:
        """Best-effort rollback for files written by one failed generation."""

        _discard_generated_asset_paths(self._asset_directory, raw_paths)

    def _persist_meshy_named_asset(
        self,
        file_name: str,
        glb_bytes: bytes,
    ) -> str:
        return _persist_generated_named_asset(
            self._asset_directory,
            file_name,
            glb_bytes,
        )

    def _resolve_meshy_asset_path(self, raw_path: str | None) -> Path:
        return self._resolve_generated_asset_path(
            raw_path,
            allowed_suffixes=frozenset({".glb"}),
        )

    def _resolve_generated_asset_path(
        self,
        raw_path: str | None,
        *,
        allowed_suffixes: frozenset[str],
    ) -> Path:
        if not raw_path:
            raise ValueError("The saved generated object has no local asset.")
        candidate = (self._asset_directory / raw_path).resolve()
        asset_root = self._asset_directory.resolve()
        try:
            candidate.relative_to(asset_root)
        except ValueError as error:
            raise ValueError("The saved Meshy asset path is unsafe.") from error
        if candidate.suffix.lower() not in allowed_suffixes:
            raise ValueError("The saved Meshy asset has an invalid file type.")
        return candidate

    def _delete_unreferenced_object_assets(
        self,
        deleted_record: GeneratedObjectRecord,
    ) -> bool:
        """Unlink safe GLB paths not referenced by any remaining record.

        Returns ``True`` when at least one requested cleanup could not be
        performed. The generated-object record remains deleted either way.
        """

        raw_asset_paths = _get_generated_object_asset_paths(deleted_record)
        if not raw_asset_paths:
            return False
        try:
            asset_root = self._asset_directory.resolve()
        except OSError:
            return True

        remaining_asset_identities: set[str] = set()
        for remaining_record in self._data.generated_objects:
            for raw_path in _get_generated_object_asset_paths(
                remaining_record
            ):
                safe_asset = _resolve_deletable_asset(
                    asset_root,
                    raw_path,
                )
                if safe_asset is not None:
                    _candidate, identity = safe_asset
                    remaining_asset_identities.add(identity)

        cleanup_failed = False
        deletion_candidates: dict[str, tuple[Path, str]] = {}
        for raw_path in raw_asset_paths:
            safe_asset = _resolve_deletable_asset(asset_root, raw_path)
            if safe_asset is None:
                cleanup_failed = True
                continue
            candidate, identity = safe_asset
            candidate_key = os.path.normcase(str(candidate.absolute()))
            deletion_candidates[candidate_key] = (candidate, identity)

        for candidate, identity in deletion_candidates.values():
            if identity in remaining_asset_identities:
                continue
            try:
                candidate.unlink(missing_ok=True)
            except OSError:
                cleanup_failed = True
        return cleanup_failed

    def _emit_data_changed(self) -> None:
        self.data_changed.emit(self._data.clone())

    def _close_video_source(self) -> None:
        if self._video_source is not None:
            self._video_source.close()
        self._video_source = None


# ### Staged-generation helpers ###
def _staged_result_used_face_purge(
    result: StagedMeshyGenerationResult,
) -> bool:
    return bool(
        result.camera_face_purge_applied
        or result.purge_original_face_count
        or result.purge_retained_face_count
        or result.purge_removed_face_count
    )


def _staged_result_used_face_removal(
    result: StagedMeshyGenerationResult,
) -> bool:
    return bool(
        result.unused_face_removal_applied
        or result.original_face_count
        or result.retained_face_count
        or result.removed_face_count
        or result.protected_face_count
    )


def _staged_generation_mode(result: StagedMeshyGenerationResult) -> str:
    stages: list[str] = []
    if _staged_result_used_face_purge(result):
        stages.append("unchecked_camera_face_purge")
    if _staged_result_used_face_removal(result):
        stages.append("unused_face_removal")
    stage_mode = "_and_".join(stages)
    if result.geometry_only:
        return (
            f"{stage_mode}_geometry_only"
            if stage_mode
            else "geometry_only"
        )
    return stage_mode if stage_mode else "geometry_then_retexture"


def _format_staged_generation_status(
    object_name: str,
    result: StagedMeshyGenerationResult,
    *,
    generated_label: str = "Generated",
) -> str:
    messages = [f"{generated_label}: {object_name}."]
    if _staged_result_used_face_purge(result):
        messages.append(
            f"Purged {result.purge_removed_face_count} of "
            f"{result.purge_original_face_count} faces visible from "
            "unchecked cameras."
        )
    if _staged_result_used_face_removal(result):
        messages.append(
            f"Removed {result.removed_face_count} of "
            f"{result.original_face_count} faces."
        )
    return " ".join(messages)


# ### Symmetric-division helpers ###
def _validate_symmetric_texture_regeneration_uvs(
    outcome: TextureRegenerationOutcome,
) -> None:
    """Require proof that the rebuilt result retained its packed UVs."""

    request = outcome.request
    submitted_fingerprint = request.submitted_uv_fingerprint
    final_fingerprint = outcome.final_uv_fingerprint
    if (
        not request.enable_original_uv
        or not request.preserve_symmetric_uvs
        or submitted_fingerprint is None
    ):
        raise UvIntegrityError(
            "Symmetric texture generation must preserve the existing packed "
            "UV layout. The existing texture was kept."
        )
    if final_fingerprint is None:
        raise UvIntegrityError(
            "The symmetric texture result has no verified UV fingerprint. "
            "The existing texture was kept."
        )
    _validate_symmetric_uv_retexture_integrity(
        submitted_fingerprint,
        final_fingerprint,
    )


def _validate_symmetric_uv_retexture_integrity(
    submitted: UvFingerprint,
    returned: UvFingerprint,
) -> None:
    """Reject local reconstruction changes before replacing the object."""

    if submitted.version != returned.version:
        raise UvIntegrityError(
            "The preserved UV integrity versions do not match. The existing "
            "texture was kept; retry texture generation."
        )
    if submitted.face_count != returned.face_count:
        raise UvIntegrityError(
            "HouseMaker could not preserve the symmetric object's face count "
            "while applying its new texture. The existing texture was kept."
        )
    if submitted.sha256 != returned.sha256:
        raise UvIntegrityError(
            "HouseMaker could not preserve the symmetric object's packed UV "
            "layout while applying its new texture. The existing texture was "
            "kept."
        )


def _parse_object_symmetric_division_metadata(
    raw_metadata: object,
) -> ObjectSymmetricDivisionMetadata | None:
    """Reject partial or malformed persisted symmetry provenance."""

    if not isinstance(raw_metadata, Mapping):
        return None
    raw_version = raw_metadata.get("version")
    raw_plane_coordinate = raw_metadata.get("plane_coordinate")
    if (
        isinstance(raw_version, bool)
        or not isinstance(raw_version, int)
        or isinstance(raw_plane_coordinate, bool)
        or not isinstance(raw_plane_coordinate, (int, float))
    ):
        return None
    orientation = str(raw_metadata.get("orientation", ""))
    automatic_fields: dict[str, object] = {}
    if raw_version in COMPACT_SYMMETRIC_METADATA_VERSIONS:
        raw_counts = raw_metadata.get("triangle_count_by_side")
        side_order = SYMMETRIC_DIVISION_SIDE_ORDER_BY_ORIENTATION.get(
            orientation
        )
        if (
            not isinstance(raw_counts, Mapping)
            or side_order is None
            or set(raw_counts) != set(side_order)
        ):
            return None
        counts: list[tuple[str, int]] = []
        for side in side_order:
            raw_count = raw_counts.get(side)
            if (
                isinstance(raw_count, bool)
                or not isinstance(raw_count, int)
                or raw_count < 0
            ):
                return None
            counts.append((side, raw_count))
        raw_tie = raw_metadata.get("tie_broken_randomly")
        if not isinstance(raw_tie, bool):
            return None
        automatic_fields = {
            "packing_mode": str(raw_metadata.get("packing_mode", "")),
            "selection_mode": str(
                raw_metadata.get("selection_mode", "")
            ),
            "triangle_count_by_side": tuple(counts),
            "tie_broken_randomly": raw_tie,
        }
        if raw_version == SYMMETRIC_QUARTER_METADATA_VERSION:
            automatic_fields["texture_content_quadrant"] = str(
                raw_metadata.get("texture_content_quadrant", "")
            )
    try:
        return ObjectSymmetricDivisionMetadata(
            version=raw_version,
            orientation=orientation,
            kept_side=str(raw_metadata.get("kept_side", "")),
            plane_coordinate=float(raw_plane_coordinate),
            texture_content_half=(
                str(raw_metadata.get("texture_content_half", ""))
                if raw_version in LEFT_HALF_SYMMETRIC_METADATA_VERSIONS
                else None
            ),
            **automatic_fields,
        )
    except (TypeError, ValueError):
        return None


def _get_object_symmetric_division_metadata(
    record: GeneratedObjectRecord | None,
) -> ObjectSymmetricDivisionMetadata | None:
    if record is None:
        return None
    return _parse_object_symmetric_division_metadata(
        record.pipeline.get(SYMMETRIC_DIVISION_PIPELINE_KEY)
    )


def _rebuild_symmetric_texture_variants(
    canonical_glb: bytes,
    metadata: ObjectSymmetricDivisionMetadata,
) -> PersistableObjectTextureVariants:
    """Rebuild variants using the exact layout stored by one record."""

    if metadata.version == AUTOMATIC_SYMMETRIC_DIVISION_METADATA_VERSION:
        return build_symmetric_square_pair_texture_variants(
            canonical_glb,
            uvs_already_left_packed=True,
        )
    if metadata.version == LEGACY_SYMMETRIC_PAIR_METADATA_VERSION:
        return build_symmetric_pair_texture_variants(
            canonical_glb,
            uvs_already_left_packed=True,
        )
    if metadata.version == SYMMETRIC_QUARTER_METADATA_VERSION:
        return build_symmetric_quarter_texture_variants(
            canonical_glb,
            uvs_already_top_left_quarter=True,
        )
    if metadata.version == SYMMETRIC_DIVISION_METADATA_VERSION:
        return build_symmetric_half_texture_variants(
            canonical_glb,
            uvs_already_left_packed=True,
        )
    raise ValueError("Unsupported symmetric texture layout version.")


def _uses_compact_symmetric_resolutions(
    metadata: ObjectSymmetricDivisionMetadata | None,
) -> bool:
    """Return whether a saved layout exposes logical 512/1024 choices."""

    return (
        metadata is not None
        and metadata.version in COMPACT_SYMMETRIC_METADATA_VERSIONS
    )


def _selectable_texture_resolutions(
    record: GeneratedObjectRecord,
) -> tuple[int, ...]:
    """Expose only logical content resolutions for automatic symmetry."""

    symmetry = _get_object_symmetric_division_metadata(record)
    if _uses_compact_symmetric_resolutions(symmetry):
        return tuple(SYMMETRIC_TEXTURE_RESOLUTIONS)
    return tuple(TEXTURE_RESOLUTIONS)


def _canonical_texture_resolution(record: GeneratedObjectRecord) -> int:
    """Return the local operation source key for one persisted record."""

    symmetry = _get_object_symmetric_division_metadata(record)
    if _uses_compact_symmetric_resolutions(symmetry):
        return TEXTURE_RESOLUTION_1024
    return TEXTURE_RESOLUTION_2048


def _validate_automatic_symmetric_division_result(
    raw_result: object,
    requested_orientation: str,
) -> ObjectSymmetricDivisionMetadata:
    """Validate one automatic core result before persisting any asset."""

    if not isinstance(raw_result, SymmetricDivisionResult):
        raise ValueError("The symmetric-division transformer returned no result.")
    if not isinstance(
        raw_result.variants,
        SymmetricSquarePairTextureVariants,
    ):
        raise ValueError("The symmetric-division texture variants are invalid.")
    core_metadata = raw_result.metadata
    if (
        not isinstance(core_metadata, SymmetricDivisionMetadata)
        or core_metadata.version
        != AUTOMATIC_SYMMETRIC_DIVISION_METADATA_VERSION
    ):
        raise ValueError("The symmetric-division provenance is invalid.")
    metadata = _parse_object_symmetric_division_metadata(
        core_metadata.to_pipeline_dict()
    )
    if metadata is None:
        raise ValueError("The symmetric-division provenance is malformed.")
    if (
        metadata.orientation != requested_orientation
        or raw_result.orientation != metadata.orientation
        or raw_result.kept_side != metadata.kept_side
        or raw_result.plane_coordinate != metadata.plane_coordinate
    ):
        raise ValueError(
            "The symmetric-division result does not match the request."
        )
    return metadata


def _build_automatic_symmetric_generation_pipeline(
    raw_pipeline: Mapping[str, object],
    metadata: ObjectSymmetricDivisionMetadata,
    variant_metadata: dict[str, dict[str, str]],
) -> dict[str, object]:
    """Make pair variants the final persisted new-generation revision."""

    canonical_variant = variant_metadata[str(TEXTURE_RESOLUTION_1024)]
    pipeline = dict(raw_pipeline)
    pipeline.pop(TEXTURE_INPAINT_STROKES_PIPELINE_KEY, None)
    pipeline.update(
        {
            TEXTURE_VARIANTS_PIPELINE_KEY: variant_metadata,
            SELECTED_TEXTURE_RESOLUTION_PIPELINE_KEY: (
                DEFAULT_TEXTURE_RESOLUTION
            ),
            "postprocessed_asset_path": canonical_variant[
                TEXTURE_VARIANT_GLB_PATH_KEY
            ],
            SYMMETRIC_DIVISION_PIPELINE_KEY: metadata.to_pipeline_dict(),
        }
    )
    return pipeline


# ### Object-operation undo helpers ###
def _get_object_operation_undo_stack(
    record: GeneratedObjectRecord | None,
) -> tuple[dict[str, object], ...]:
    """Return validated persistent snapshots without accepting nested stacks."""

    if record is None:
        return ()
    raw_stack = record.pipeline.get(OBJECT_OPERATION_UNDO_STACK_PIPELINE_KEY)
    if not isinstance(raw_stack, list):
        return ()
    snapshots: list[dict[str, object]] = []
    for raw_snapshot in raw_stack[-MAX_OBJECT_OPERATION_UNDO_COUNT:]:
        if not isinstance(raw_snapshot, dict):
            continue
        operation = raw_snapshot.get("operation")
        asset_path = raw_snapshot.get("asset_path")
        provider_task_id = raw_snapshot.get("provider_task_id")
        pipeline = raw_snapshot.get("pipeline")
        if operation not in {
            OBJECT_OPERATION_GENERATE_TEXTURE,
            OBJECT_OPERATION_PURGE_FACES,
        }:
            continue
        if not isinstance(asset_path, str) or not asset_path.strip():
            continue
        if not isinstance(provider_task_id, str) or not provider_task_id.strip():
            continue
        if not isinstance(pipeline, dict):
            continue
        snapshot_pipeline = copy.deepcopy(pipeline)
        snapshot_pipeline.pop(OBJECT_OPERATION_UNDO_STACK_PIPELINE_KEY, None)
        snapshots.append(
            {
                "operation": operation,
                "asset_path": asset_path,
                "provider_task_id": provider_task_id,
                "pipeline": snapshot_pipeline,
            }
        )
    return tuple(snapshots)


def _push_object_operation_undo_snapshot(
    record: GeneratedObjectRecord,
    next_pipeline: dict[str, object],
    *,
    operation: str,
) -> dict[str, object]:
    """Attach one bounded pre-operation snapshot to a replacement pipeline."""

    if operation not in {
        OBJECT_OPERATION_GENERATE_TEXTURE,
        OBJECT_OPERATION_PURGE_FACES,
    }:
        raise ValueError("Unknown undoable object operation.")
    snapshot_pipeline = copy.deepcopy(record.pipeline)
    snapshot_pipeline.pop(OBJECT_OPERATION_UNDO_STACK_PIPELINE_KEY, None)
    snapshot: dict[str, object] = {
        "operation": operation,
        "asset_path": record.asset_path,
        "provider_task_id": record.provider_task_id,
        "pipeline": snapshot_pipeline,
    }
    undo_stack = [
        *(_get_object_operation_undo_stack(record)),
        snapshot,
    ][-MAX_OBJECT_OPERATION_UNDO_COUNT:]
    pipeline = copy.deepcopy(next_pipeline)
    pipeline[OBJECT_OPERATION_UNDO_STACK_PIPELINE_KEY] = undo_stack
    return pipeline


def _restore_object_operation_snapshot(
    record: GeneratedObjectRecord,
    snapshot: dict[str, object],
    remaining_stack: Sequence[dict[str, object]],
) -> GeneratedObjectRecord:
    """Build an immutable restored record before any workspace state changes."""

    asset_path = snapshot.get("asset_path")
    provider_task_id = snapshot.get("provider_task_id")
    raw_pipeline = snapshot.get("pipeline")
    if not isinstance(asset_path, str) or not asset_path.strip():
        raise ValueError("The saved undo model path is invalid.")
    if not isinstance(provider_task_id, str) or not provider_task_id.strip():
        raise ValueError("The saved undo provider task is invalid.")
    if not isinstance(raw_pipeline, dict):
        raise ValueError("The saved undo provenance is invalid.")
    pipeline = copy.deepcopy(raw_pipeline)
    pipeline.pop(OBJECT_OPERATION_UNDO_STACK_PIPELINE_KEY, None)
    if remaining_stack:
        pipeline[OBJECT_OPERATION_UNDO_STACK_PIPELINE_KEY] = [
            copy.deepcopy(entry) for entry in remaining_stack
        ][-MAX_OBJECT_OPERATION_UNDO_COUNT:]
    return replace(
        record,
        pipeline=pipeline,
        provider_task_id=provider_task_id,
        asset_path=asset_path,
    )


# ### Face-purge helpers ###
def _build_unchecked_camera_face_purge_pipeline(
    record: GeneratedObjectRecord,
    outcome: UncheckedCameraFacePurgeOutcome,
    variant_metadata: dict[str, dict[str, str]] | None,
    *,
    postprocessed_asset_path: str | None = None,
) -> dict[str, object]:
    request = outcome.request
    result = outcome.result
    pipeline: dict[str, object] = dict(record.pipeline)
    pipeline.pop(TEXTURE_INPAINT_STROKES_PIPELINE_KEY, None)
    selected_resolution = _get_selected_texture_resolution(record)
    if selected_resolution not in _selectable_texture_resolutions(record):
        selected_resolution = DEFAULT_TEXTURE_RESOLUTION
    raw_count = pipeline.get("manual_face_purge_count", 0)
    try:
        purge_count = max(int(raw_count), 0) + 1
    except (TypeError, ValueError):
        purge_count = 1
    history_entry: dict[str, object] = {
        "unchecked_camera_ids": list(request.unchecked_camera_ids),
        "original_face_count": result.original_face_count,
        "retained_face_count": result.retained_face_count,
        "removed_face_count": result.removed_face_count,
    }
    raw_history = pipeline.get("manual_face_purge_history")
    history = (
        [dict(entry) for entry in raw_history if isinstance(entry, dict)]
        if isinstance(raw_history, list)
        else []
    )
    history.append(history_entry)
    if postprocessed_asset_path is None:
        if variant_metadata is None:
            raise ValueError("A purged post-processed model path is required.")
        postprocessed_asset_path = variant_metadata[
            str(_canonical_texture_resolution(record))
        ][TEXTURE_VARIANT_GLB_PATH_KEY]
    pipeline.update(
        {
            "postprocessed_asset_path": postprocessed_asset_path,
            "camera_face_purge_applied": True,
            "unchecked_camera_ids": list(request.unchecked_camera_ids),
            "manual_face_purge_count": purge_count,
            "latest_face_purge_original_face_count": (
                result.original_face_count
            ),
            "latest_face_purge_retained_face_count": (
                result.retained_face_count
            ),
            "latest_face_purge_removed_face_count": (
                result.removed_face_count
            ),
            "manual_face_purge_history": history[-25:],
        }
    )
    if variant_metadata is None:
        pipeline.pop(TEXTURE_VARIANTS_PIPELINE_KEY, None)
        pipeline.pop(SELECTED_TEXTURE_RESOLUTION_PIPELINE_KEY, None)
    else:
        pipeline.update(
            {
                TEXTURE_VARIANTS_PIPELINE_KEY: variant_metadata,
                SELECTED_TEXTURE_RESOLUTION_PIPELINE_KEY: selected_resolution,
            }
        )
    return pipeline


# ### Texture-regeneration helpers ###
def _texture_regeneration_source_asset_path(
    record: GeneratedObjectRecord,
) -> str:
    """Return the persisted geometry revision submitted to Retexture."""

    raw_postprocessed_path = record.pipeline.get("postprocessed_asset_path")
    if isinstance(raw_postprocessed_path, str) and raw_postprocessed_path:
        return raw_postprocessed_path
    canonical_variant = _get_texture_variant_metadata(
        record,
        DEFAULT_TEXTURE_RESOLUTION,
    )
    if canonical_variant is None:
        return record.asset_path
    return canonical_variant[TEXTURE_VARIANT_GLB_PATH_KEY]


def _build_regenerated_texture_pipeline(
    record: GeneratedObjectRecord,
    outcome: TextureRegenerationOutcome,
    variant_metadata: dict[str, dict[str, str]],
) -> dict[str, object]:
    request = outcome.request
    result = outcome.result
    pipeline: dict[str, object] = dict(record.pipeline)
    pipeline.pop(TEXTURE_INPAINT_STROKES_PIPELINE_KEY, None)
    selected_resolution = _get_selected_texture_resolution(record)
    if selected_resolution not in _selectable_texture_resolutions(record):
        selected_resolution = DEFAULT_TEXTURE_RESOLUTION
    raw_count = pipeline.get("texture_regeneration_count", 0)
    try:
        regeneration_count = max(int(raw_count), 0) + 1
    except (TypeError, ValueError):
        regeneration_count = 1

    raw_mode = pipeline.get("mode")
    if raw_mode == "geometry_only":
        pipeline["mode"] = "geometry_then_retexture"
    elif isinstance(raw_mode, str) and raw_mode.endswith("_geometry_only"):
        pipeline["mode"] = raw_mode.removesuffix("_geometry_only")

    reference_png = _encode_png(request.reference_image_bgra)
    history_entry: dict[str, object] = {
        "task_id": result.task_id,
        "reference_frame_index": request.reference_frame_index,
        "reference_image_sha256": hashlib.sha256(reference_png).hexdigest(),
        "enable_original_uv": request.enable_original_uv,
        "preserve_symmetric_uvs": request.preserve_symmetric_uvs,
    }
    raw_history = pipeline.get("texture_regeneration_history")
    history = (
        [dict(entry) for entry in raw_history if isinstance(entry, dict)]
        if isinstance(raw_history, list)
        else []
    )
    history.append(history_entry)

    pipeline.update(
        {
            TEXTURE_VARIANTS_PIPELINE_KEY: variant_metadata,
            SELECTED_TEXTURE_RESOLUTION_PIPELINE_KEY: selected_resolution,
            "geometry_only": False,
            "texture_regeneration_count": regeneration_count,
            "latest_texture_task_id": result.task_id,
            "last_texture_regeneration_reference_frame_index": (
                request.reference_frame_index
            ),
            "last_texture_regeneration_reference_image_sha256": (
                history_entry["reference_image_sha256"]
            ),
            "last_texture_regeneration_enable_original_uv": (
                request.enable_original_uv
            ),
            "last_texture_regeneration_preserve_symmetric_uvs": (
                request.preserve_symmetric_uvs
            ),
            "texture_regeneration_history": history[-25:],
        }
    )
    if request.submitted_uv_fingerprint is not None:
        final_fingerprint = outcome.final_uv_fingerprint
        if final_fingerprint is None:
            raise UvIntegrityError(
                "The regenerated texture UV fingerprint is unavailable."
            )
        pipeline.update(
            {
                "retexture_enable_original_uv": True,
                "texture_regeneration_uv_fingerprint_version": (
                    UV_FINGERPRINT_VERSION
                ),
                "texture_regeneration_submitted_uv_fingerprint": (
                    request.submitted_uv_fingerprint.sha256
                ),
                "texture_regeneration_final_uv_fingerprint": (
                    final_fingerprint.sha256
                ),
                "texture_regeneration_uv_face_count": (
                    request.submitted_uv_fingerprint.face_count
                ),
            }
        )
    else:
        pipeline["retexture_enable_original_uv"] = False
    return pipeline


# ### UV preview helpers ###
def _collect_model_uv_triangles(
    model: GeneratedModel,
) -> tuple[UvTriangle, ...]:
    """Return finite per-face UV triangles from every model geometry."""

    triangles: list[UvTriangle] = []
    for _geometry_name, geometry in sorted(
        model.scene.geometry.items(),
        key=lambda item: str(item[0]),
    ):
        raw_faces = getattr(geometry, "faces", None)
        raw_uv = getattr(getattr(geometry, "visual", None), "uv", None)
        if raw_faces is None or raw_uv is None:
            continue
        try:
            faces = np.asarray(raw_faces, dtype=np.int64)
            uv = np.asarray(raw_uv, dtype=float)
        except (TypeError, ValueError):
            continue
        if (
            faces.ndim != 2
            or faces.shape[1:] != (3,)
            or uv.ndim != 2
            or uv.shape[1:] != (2,)
            or not np.all(np.isfinite(uv))
        ):
            continue
        if len(faces) == 0:
            continue
        if np.any(faces < 0) or np.any(faces >= len(uv)):
            continue
        for triangle in uv[faces]:
            triangles.append(
                (
                    (float(triangle[0, 0]), float(triangle[0, 1])),
                    (float(triangle[1, 0]), float(triangle[1, 1])),
                    (float(triangle[2, 0]), float(triangle[2, 1])),
                )
            )
    return tuple(triangles)


# ### Generated-object asset helpers ###
def _get_generated_object_asset_paths(
    record: GeneratedObjectRecord,
) -> tuple[str, ...]:
    """Return current and retained-undo GLB/PNG asset path fields."""

    raw_paths: list[str] = []
    if isinstance(record.asset_path, str) and record.asset_path.strip():
        raw_paths.append(record.asset_path)

    def collect_pipeline_paths(pipeline: dict[str, object]) -> None:
        for pipeline_key in MESHY_REVISION_ASSET_PIPELINE_KEYS:
            raw_path = pipeline.get(pipeline_key)
            if isinstance(raw_path, str) and raw_path.strip():
                raw_paths.append(raw_path)
        raw_variants = pipeline.get(TEXTURE_VARIANTS_PIPELINE_KEY)
        if not isinstance(raw_variants, dict):
            return
        for raw_variant in raw_variants.values():
            if not isinstance(raw_variant, dict):
                continue
            for path_key in (
                TEXTURE_VARIANT_GLB_PATH_KEY,
                TEXTURE_VARIANT_PNG_PATH_KEY,
            ):
                raw_path = raw_variant.get(path_key)
                if isinstance(raw_path, str) and raw_path.strip():
                    raw_paths.append(raw_path)

    collect_pipeline_paths(record.pipeline)
    for snapshot in _get_object_operation_undo_stack(record):
        snapshot_asset_path = snapshot.get("asset_path")
        if (
            isinstance(snapshot_asset_path, str)
            and snapshot_asset_path.strip()
        ):
            raw_paths.append(snapshot_asset_path)
        snapshot_pipeline = snapshot.get("pipeline")
        if isinstance(snapshot_pipeline, dict):
            collect_pipeline_paths(snapshot_pipeline)
    return tuple(dict.fromkeys(raw_paths))


def _resolve_deletable_asset(
    asset_root: Path,
    raw_path: str,
) -> tuple[Path, str] | None:
    """Return a contained GLB unlink path and canonical reference identity."""

    try:
        candidate = asset_root / raw_path
        resolved_candidate = candidate.resolve()
        resolved_candidate.relative_to(asset_root)
    except (OSError, RuntimeError, ValueError):
        return None
    if candidate.suffix.lower() not in {".glb", ".png"}:
        return None
    identity = os.path.normcase(str(resolved_candidate))
    return candidate, identity


# ### Background asset preparation ###
def _persist_generated_named_asset(
    asset_directory: Path,
    file_name: str,
    payload: bytes,
) -> str:
    """Atomically fsync and replace one generated GLB or PNG asset."""

    normalized_file_name = str(file_name)
    if (
        not normalized_file_name
        or Path(normalized_file_name).name != normalized_file_name
        or normalized_file_name in {".", ".."}
        or Path(normalized_file_name).suffix.lower() not in {".glb", ".png"}
    ):
        raise ValueError("Generated asset file name is unsafe.")
    asset_root = Path(asset_directory)
    asset_root.mkdir(parents=True, exist_ok=True)
    destination = asset_root / normalized_file_name
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{normalized_file_name}.",
        suffix=".tmp",
        dir=str(asset_root),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return normalized_file_name


def _persist_object_texture_variants_to_directory(
    asset_directory: Path,
    object_id: str,
    variants: PersistableObjectTextureVariants,
    *,
    asset_stem: str | None = None,
    cancel_event: threading.Event | None = None,
    persist_asset: Callable[[str, bytes], str] | None = None,
) -> dict[str, dict[str, str]]:
    """Persist all selectable variants with rollback on failure/cancel."""

    normalized_asset_stem = (
        str(object_id) if asset_stem is None else str(asset_stem)
    )
    if (
        not normalized_asset_stem
        or Path(normalized_asset_stem).name != normalized_asset_stem
        or normalized_asset_stem in {".", ".."}
    ):
        raise ValueError("Texture variant asset stem is unsafe.")
    resolutions = (
        TEXTURE_RESOLUTIONS
        if isinstance(variants, ObjectTextureVariants)
        else variants.selectable_resolutions
    )
    metadata: dict[str, dict[str, str]] = {}
    created_paths: list[str] = []
    persist = persist_asset
    if persist is None:
        persist = lambda file_name, payload: _persist_generated_named_asset(
            asset_directory,
            file_name,
            payload,
        )
    try:
        for resolution in resolutions:
            _raise_if_generation_cancelled(cancel_event)
            glb_path = persist(
                f"{normalized_asset_stem}.texture-{resolution}.glb",
                variants.glb_by_resolution[resolution],
            )
            created_paths.append(glb_path)
            _raise_if_generation_cancelled(cancel_event)
            png_path = persist(
                f"{normalized_asset_stem}.texture-{resolution}.png",
                variants.texture_png_by_resolution[resolution],
            )
            created_paths.append(png_path)
            metadata[str(resolution)] = {
                TEXTURE_VARIANT_GLB_PATH_KEY: glb_path,
                TEXTURE_VARIANT_PNG_PATH_KEY: png_path,
            }
        return metadata
    except Exception:
        _discard_generated_asset_paths(asset_directory, created_paths)
        raise


def _discard_generated_asset_paths(
    asset_directory: Path,
    raw_paths: Sequence[str],
) -> None:
    """Best-effort deletion for one worker's uncommitted contained assets."""

    try:
        asset_root = Path(asset_directory).resolve()
    except OSError:
        return
    for raw_path in tuple(raw_paths):
        safe_asset = _resolve_deletable_asset(asset_root, str(raw_path))
        if safe_asset is None:
            continue
        candidate, _identity = safe_asset
        try:
            candidate.unlink(missing_ok=True)
        except OSError:
            continue


def _materialize_texture_regeneration_preflight(
    raw_request: TextureRegenerationRequest | _TextureRegenerationPreflight,
    asset_directory: Path | None,
    cancel_event: threading.Event | None = None,
) -> _MaterializedTextureRegeneration:
    """Read and validate deferred Retexture inputs on the worker thread."""

    if isinstance(raw_request, TextureRegenerationRequest):
        return _MaterializedTextureRegeneration(request=raw_request)
    if not isinstance(raw_request, _TextureRegenerationPreflight):
        raise TypeError("Texture generation received an invalid preflight.")
    if asset_directory is None:
        raise ValueError("Texture preflight requires an asset directory.")
    _raise_if_generation_cancelled(cancel_event)
    asset_root = Path(asset_directory).resolve()
    source_path = (asset_root / raw_request.source_asset_path).resolve()
    try:
        source_path.relative_to(asset_root)
    except ValueError as error:
        raise ValueError("The texture source asset path is unsafe.") from error
    if source_path.suffix.lower() != ".glb":
        raise ValueError("The texture source asset must be a GLB.")
    revision_before = _build_generation_asset_revision(
        asset_root,
        raw_request.source_asset_path,
    )
    if revision_before != raw_request.source_asset_revision:
        raise ValueError(
            "The target object's source model changed before texture "
            "generation could start."
        )
    try:
        model_glb = source_path.read_bytes()
    except OSError as error:
        raise ValueError(
            "The model revision used for Retexture could not be read."
        ) from error
    _raise_if_generation_cancelled(cancel_event)
    revision_after_read = _build_generation_asset_revision(
        asset_root,
        raw_request.source_asset_path,
    )
    if revision_after_read != revision_before:
        raise ValueError(
            "The target object's source model changed while texture "
            "generation was starting."
        )
    import_generated_glb(model_glb)
    submitted_fingerprint = (
        build_uv_fingerprint(model_glb)
        if raw_request.enable_original_uv
        else None
    )
    _raise_if_generation_cancelled(cancel_event)
    revision_after_validation = _build_generation_asset_revision(
        asset_root,
        raw_request.source_asset_path,
    )
    if revision_after_validation != revision_before:
        raise ValueError(
            "The target object's source model changed during texture "
            "preflight."
        )
    request = TextureRegenerationRequest(
        object_id=raw_request.object_id,
        reference_frame_index=raw_request.reference_frame_index,
        reference_image_bgra=raw_request.reference_image_bgra,
        model_glb=model_glb,
        settings=raw_request.settings,
        enable_original_uv=raw_request.enable_original_uv,
        submitted_uv_fingerprint=submitted_fingerprint,
        preserve_symmetric_uvs=raw_request.preserve_symmetric_uvs,
    )
    return _MaterializedTextureRegeneration(
        request=request,
        source_asset_path=raw_request.source_asset_path,
        source_asset_revision=revision_before,
    )


def _prepare_and_persist_object_generation(
    asset_directory: Path,
    object_id: str,
    request: GenerationRequest,
    result: MeshyGenerationResult,
    generated_model: GeneratedModel,
    cancel_event: threading.Event | None = None,
) -> _SavedObjectGeneration:
    """Prepare every model revision without touching GUI-owned state."""

    pipeline: dict[str, object] = {}
    persisted_asset_paths: list[str] = []
    symmetry: ObjectSymmetricDivisionMetadata | None = None
    variant_metadata: dict[str, dict[str, str]] | None = None
    try:
        _raise_if_generation_cancelled(cancel_event)
        geometry_only = (
            isinstance(result, StagedMeshyGenerationResult)
            and result.geometry_only
        )
        texture_variants = (
            None
            if geometry_only
            else generated_model.object_texture_variants
        )
        if request.symmetric_division_enabled:
            if geometry_only or not isinstance(
                texture_variants,
                ObjectTextureVariants,
            ):
                raise ValueError(
                    "Symmetric division requires a newly generated textured "
                    "model."
                )
            division_result = build_automatic_symmetric_object_variants(
                texture_variants.glb_by_resolution[TEXTURE_RESOLUTION_2048],
                request.symmetric_division_orientation,
            )
            symmetry = _validate_automatic_symmetric_division_result(
                division_result,
                request.symmetric_division_orientation,
            )
            texture_variants = division_result.variants
            _raise_if_generation_cancelled(cancel_event)

        if texture_variants is None:
            asset_path = _persist_generated_named_asset(
                asset_directory,
                f"{object_id}.glb",
                result.glb_bytes,
            )
            persisted_asset_paths.append(asset_path)
            preview_model = (
                import_generated_glb(result.glb_bytes)
                if geometry_only
                else generated_model
            )
        else:
            variant_metadata = (
                _persist_object_texture_variants_to_directory(
                    asset_directory,
                    object_id,
                    texture_variants,
                    cancel_event=cancel_event,
                )
            )
            persisted_asset_paths.extend(
                path
                for variant in variant_metadata.values()
                for path in variant.values()
            )
            pipeline.update(
                {
                    TEXTURE_VARIANTS_PIPELINE_KEY: variant_metadata,
                    SELECTED_TEXTURE_RESOLUTION_PIPELINE_KEY: (
                        DEFAULT_TEXTURE_RESOLUTION
                    ),
                }
            )
            asset_path = variant_metadata[str(DEFAULT_TEXTURE_RESOLUTION)][
                TEXTURE_VARIANT_GLB_PATH_KEY
            ]
            preview_model = import_generated_glb(
                texture_variants.glb_by_resolution[
                    DEFAULT_TEXTURE_RESOLUTION
                ]
            )
        _raise_if_generation_cancelled(cancel_event)

        if isinstance(result, StagedMeshyGenerationResult):
            source_asset_path = _persist_generated_named_asset(
                asset_directory,
                f"{object_id}.{MESHY_REVISION_GEOMETRY}.glb",
                result.source_glb_bytes,
            )
            persisted_asset_paths.append(source_asset_path)
            staged_pipeline: dict[str, object] = {
                "mode": _staged_generation_mode(result),
                "geometry_task_id": result.geometry_task_id,
                "source_asset_path": source_asset_path,
                "enabled_camera_ids": list(result.enabled_camera_ids),
                "unchecked_camera_ids": list(result.unchecked_camera_ids),
                "camera_face_purge_applied": result.camera_face_purge_applied,
                "unused_face_removal_applied": (
                    _staged_result_used_face_removal(result)
                ),
                "geometry_only": result.geometry_only,
            }
            if symmetry is None:
                postprocessed_asset_path = _persist_generated_named_asset(
                    asset_directory,
                    f"{object_id}.{MESHY_REVISION_POSTPROCESSED}.glb",
                    result.postprocessed_glb_bytes,
                )
                persisted_asset_paths.append(postprocessed_asset_path)
                staged_pipeline["postprocessed_asset_path"] = (
                    postprocessed_asset_path
                )
            pipeline.update(staged_pipeline)
            if _staged_result_used_face_purge(result):
                pipeline.update(
                    {
                        "purge_original_face_count": (
                            result.purge_original_face_count
                        ),
                        "purge_retained_face_count": (
                            result.purge_retained_face_count
                        ),
                        "purge_removed_face_count": (
                            result.purge_removed_face_count
                        ),
                    }
                )
            if _staged_result_used_face_removal(result):
                pipeline.update(
                    {
                        "original_face_count": result.original_face_count,
                        "retained_face_count": result.retained_face_count,
                        "removed_face_count": result.removed_face_count,
                        "protected_face_count": result.protected_face_count,
                    }
                )

        if symmetry is not None:
            if variant_metadata is None:
                raise ValueError("Symmetric texture variants were not saved.")
            pipeline = _build_automatic_symmetric_generation_pipeline(
                pipeline,
                symmetry,
                variant_metadata,
            )
        _raise_if_generation_cancelled(cancel_event)
        preview_asset_revision = _build_generation_asset_revision(
            asset_directory,
            asset_path,
        )
        if preview_asset_revision[1] is None:
            raise OSError("The saved object preview asset is unavailable.")
        return _SavedObjectGeneration(
            result=result,
            object_id=str(object_id),
            pipeline=pipeline,
            asset_path=asset_path,
            preview_model=preview_model,
            preview_asset_revision=preview_asset_revision,
            symmetry=symmetry,
            persisted_asset_paths=tuple(persisted_asset_paths),
        )
    except Exception:
        _discard_generated_asset_paths(
            asset_directory,
            persisted_asset_paths,
        )
        raise


def _prepare_and_persist_texture_regeneration(
    asset_directory: Path,
    outcome: TextureRegenerationOutcome,
    generated_model: GeneratedModel,
    symmetry: ObjectSymmetricDivisionMetadata | None,
    selected_resolution: int,
    record_snapshot: GeneratedObjectRecord,
    source_asset_path: str | None = None,
    source_asset_revision: tuple[object, ...] | None = None,
    cancel_event: threading.Event | None = None,
) -> _SavedObjectTextureRegeneration:
    """Build, save, and import one retexture result on its worker thread."""

    persisted_asset_paths: list[str] = []
    try:
        texture_variants = generated_model.object_texture_variants
        if texture_variants is None:
            raise ValueError(
                "The regenerated model has no selectable texture variants."
            )
        if symmetry is not None:
            _validate_symmetric_texture_regeneration_uvs(outcome)
            canonical_provider_glb = texture_variants.glb_by_resolution[
                TEXTURE_RESOLUTION_2048
            ]
            texture_variants = _rebuild_symmetric_texture_variants(
                canonical_provider_glb,
                symmetry,
            )
        _raise_if_generation_cancelled(cancel_event)
        variant_metadata = _persist_object_texture_variants_to_directory(
            asset_directory,
            outcome.request.object_id,
            texture_variants,
            asset_stem=f"regenerated-{uuid.uuid4().hex}",
            cancel_event=cancel_event,
        )
        persisted_asset_paths.extend(
            path
            for variant in variant_metadata.values()
            for path in variant.values()
        )
        selectable_resolutions = (
            TEXTURE_RESOLUTIONS
            if isinstance(texture_variants, ObjectTextureVariants)
            else texture_variants.selectable_resolutions
        )
        normalized_resolution = int(selected_resolution)
        if normalized_resolution not in selectable_resolutions:
            normalized_resolution = DEFAULT_TEXTURE_RESOLUTION
        preview_model = import_generated_glb(
            texture_variants.glb_by_resolution[normalized_resolution]
        )
        next_pipeline = _build_regenerated_texture_pipeline(
            record_snapshot,
            outcome,
            variant_metadata,
        )
        preview_asset_path = variant_metadata[str(normalized_resolution)][
            TEXTURE_VARIANT_GLB_PATH_KEY
        ]
        preview_asset_revision = _build_generation_asset_revision(
            asset_directory,
            preview_asset_path,
        )
        if preview_asset_revision[1] is None:
            raise OSError("The saved texture preview asset is unavailable.")
        _raise_if_generation_cancelled(cancel_event)
        return _SavedObjectTextureRegeneration(
            outcome=outcome,
            variant_metadata=variant_metadata,
            selected_resolution=normalized_resolution,
            base_asset_path=record_snapshot.asset_path,
            base_provider_task_id=record_snapshot.provider_task_id,
            base_pipeline=copy.deepcopy(record_snapshot.pipeline),
            source_asset_path=source_asset_path,
            source_asset_revision=source_asset_revision,
            next_pipeline=next_pipeline,
            preview_model=preview_model,
            preview_asset_revision=preview_asset_revision,
            persisted_asset_paths=tuple(persisted_asset_paths),
        )
    except Exception:
        _discard_generated_asset_paths(
            asset_directory,
            persisted_asset_paths,
        )
        raise


# ### Adapter helpers ###
class _GenerationCancelled(Exception):
    """Internal control flow for an abandoned background operation."""


def _raise_if_generation_cancelled(
    cancel_event: threading.Event | None,
) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise _GenerationCancelled


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


def _invoke_texture_regenerator(
    regenerator: TextureRegenerator
    | Callable[[TextureRegenerationRequest], MeshyGenerationResult],
    request: TextureRegenerationRequest,
    progress_callback: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> MeshyGenerationResult:
    progress_method = getattr(regenerator, "regenerate_with_progress", None)
    if callable(progress_method):
        return progress_method(request, progress_callback, cancel_event)
    regenerate_method = getattr(regenerator, "regenerate", None)
    if callable(regenerate_method):
        if isinstance(regenerator, MeshyTextureRegenerator):
            return regenerator.regenerate(
                request,
                progress_callback,
                cancel_event,
            )
        return regenerate_method(request)
    if callable(regenerator):
        return regenerator(request)
    raise TypeError("The Meshy texture regenerator is not callable.")


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


# ### Generated-object presentation helpers ###
def _build_generated_object_display_snapshot(
    record: GeneratedObjectRecord,
    asset_directory: Path,
) -> tuple[object, ...]:
    """Snapshot only record content that changes the Object preview."""

    return (
        record.object_id,
        record.asset_path,
        _build_generation_asset_revision(
            asset_directory,
            record.asset_path,
        ),
        copy.deepcopy(record.pipeline),
    )


def _build_generation_asset_revision(
    asset_directory: Path,
    raw_asset_path: object,
) -> tuple[object, ...]:
    """Return a stable local path and cheap revision for one generated GLB."""

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


def _build_generated_object_list_rows(
    records: Sequence[GeneratedObjectRecord],
) -> tuple[tuple[str, str, str], ...]:
    """Return the exact lightweight rows expected by the object selector."""

    return tuple(
        (
            record.object_id,
            f"#{object_index} {record.object_name} · "
            f"frame {record.frame_index + 1}",
            (
                ""
                if not record.provider_task_id
                else f"Meshy task: {record.provider_task_id}"
            ),
        )
        for object_index, record in enumerate(records, start=1)
    )


# ### Texture-atlas helpers ###
def _build_texture_resolution_entry_signature(
    record: GeneratedObjectRecord,
    asset_directory: Path,
) -> tuple[object, ...]:
    """Snapshot variant metadata and cheap file revisions without decoding PNGs."""

    raw_variants = record.pipeline.get(TEXTURE_VARIANTS_PIPELINE_KEY)
    asset_root = asset_directory.resolve()
    file_revisions: list[tuple[object, ...]] = []
    for resolution in _selectable_texture_resolutions(record):
        variant = _get_texture_variant_metadata(record, resolution)
        if variant is None:
            file_revisions.append((resolution, None))
            continue
        for path_key in (
            TEXTURE_VARIANT_GLB_PATH_KEY,
            TEXTURE_VARIANT_PNG_PATH_KEY,
        ):
            raw_path = variant[path_key]
            try:
                asset_path = (asset_directory / raw_path).resolve()
                asset_path.relative_to(asset_root)
                file_stat = asset_path.stat()
                revision = (
                    file_stat.st_size,
                    file_stat.st_mtime_ns,
                    file_stat.st_ctime_ns,
                )
            except (OSError, RuntimeError, ValueError):
                revision = None
            file_revisions.append(
                (resolution, path_key, raw_path, revision)
            )
    return (
        copy.deepcopy(raw_variants),
        tuple(file_revisions),
    )


def _get_selected_texture_resolution(record: GeneratedObjectRecord) -> int:
    raw_resolution = record.pipeline.get(
        SELECTED_TEXTURE_RESOLUTION_PIPELINE_KEY,
        DEFAULT_TEXTURE_RESOLUTION,
    )
    try:
        resolution = int(raw_resolution)
    except (TypeError, ValueError):
        return DEFAULT_TEXTURE_RESOLUTION
    if resolution not in TEXTURE_RESOLUTIONS:
        return DEFAULT_TEXTURE_RESOLUTION
    return resolution


def _get_texture_variant_metadata(
    record: GeneratedObjectRecord,
    resolution: int,
) -> dict[str, str] | None:
    if int(resolution) not in TEXTURE_RESOLUTIONS:
        return None
    raw_variants = record.pipeline.get(TEXTURE_VARIANTS_PIPELINE_KEY)
    if not isinstance(raw_variants, dict):
        return None
    raw_variant = raw_variants.get(str(int(resolution)))
    if not isinstance(raw_variant, dict):
        return None
    glb_path = raw_variant.get(TEXTURE_VARIANT_GLB_PATH_KEY)
    png_path = raw_variant.get(TEXTURE_VARIANT_PNG_PATH_KEY)
    if not isinstance(glb_path, str) or not glb_path.strip():
        return None
    if not isinstance(png_path, str) or not png_path.strip():
        return None
    return {
        TEXTURE_VARIANT_GLB_PATH_KEY: glb_path,
        TEXTURE_VARIANT_PNG_PATH_KEY: png_path,
    }


def _build_texture_resolution_entries(
    record: GeneratedObjectRecord,
    asset_directory: Path,
) -> list[TextureAtlasEntry]:
    entries: list[TextureAtlasEntry] = []
    asset_root = asset_directory.resolve()
    for resolution in _selectable_texture_resolutions(record):
        variant = _get_texture_variant_metadata(record, resolution)
        if variant is None:
            continue
        try:
            image_path = (
                asset_directory / variant[TEXTURE_VARIANT_PNG_PATH_KEY]
            ).resolve()
            glb_path = (
                asset_directory / variant[TEXTURE_VARIANT_GLB_PATH_KEY]
            ).resolve()
            image_path.relative_to(asset_root)
            glb_path.relative_to(asset_root)
        except (OSError, RuntimeError, ValueError):
            continue
        if (
            image_path.suffix.lower() != ".png"
            or glb_path.suffix.lower() != ".glb"
            or not image_path.is_file()
            or not glb_path.is_file()
        ):
            continue
        try:
            entries.append(
                TextureAtlasEntry(
                    atlas_id=(
                        f"{record.object_id}:resolution:{resolution}"
                    ),
                    display_name=f"{resolution} x {resolution}",
                    image=image_path,
                    owner_id=record.object_id,
                )
            )
        except (OSError, TypeError, ValueError):
            continue
    return entries


def _count_available_texture_resolution_entries(
    record: GeneratedObjectRecord,
    asset_directory: Path,
) -> int:
    """Count existing safe variant pairs without decoding their PNG pixels."""

    entry_count = 0
    asset_root = asset_directory.resolve()
    for resolution in _selectable_texture_resolutions(record):
        variant = _get_texture_variant_metadata(record, resolution)
        if variant is None:
            continue
        try:
            image_path = (
                asset_directory / variant[TEXTURE_VARIANT_PNG_PATH_KEY]
            ).resolve()
            glb_path = (
                asset_directory / variant[TEXTURE_VARIANT_GLB_PATH_KEY]
            ).resolve()
            image_path.relative_to(asset_root)
            glb_path.relative_to(asset_root)
        except (OSError, RuntimeError, ValueError):
            continue
        if (
            image_path.suffix.lower() == ".png"
            and glb_path.suffix.lower() == ".glb"
            and image_path.is_file()
            and glb_path.is_file()
        ):
            entry_count += 1
    return entry_count


def _parse_texture_resolution_entry_id(
    object_id: str,
    entry_id: str,
) -> int:
    prefix = f"{object_id}:resolution:"
    if not str(entry_id).startswith(prefix):
        return -1
    try:
        resolution = int(str(entry_id)[len(prefix):])
    except ValueError:
        return -1
    return resolution if resolution in TEXTURE_RESOLUTIONS else -1


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
