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
from housemaker.glass_material import (
    DEFAULT_HOUSEMAKER_GLASS_DOUBLE_SIDED,
    HOUSEMAKER_GLASS_MATERIAL_NAME,
    HOUSEMAKER_GLASS_MATERIAL_PROFILE,
    build_housemaker_glass_material,
    get_housemaker_glass_runtime_key,
    is_housemaker_glass_material,
)
from housemaker.glb import GeneratedModel, import_generated_glb
from housemaker.meshy_generation import (
    MeshyGenerationResult,
    request_image_to_3d_model,
    request_retextured_model,
)
from housemaker.object_texture_variants import (
    ATLAS_MAP_BASE_COLOR,
    ATLAS_MAP_TYPES,
    DEFAULT_TEXTURE_RESOLUTION,
    PBR_MAP_METALLIC,
    PBR_MAP_NORMAL,
    PBR_MAP_ROUGHNESS,
    PBR_MAP_TYPES,
    TEXTURE_RESOLUTION_1024,
    TEXTURE_RESOLUTION_2048,
    TEXTURE_RESOLUTIONS,
    ObjectTextureVariants,
    build_object_texture_variants,
    replace_object_base_color_texture_from_glb,
)
from housemaker.object_face_edit import (
    ObjectFaceDeletionResult,
    ObjectFaceGeometry,
    _delete_object_faces_preserving_uvs_with_geometry,
    load_object_face_geometry,
    load_object_face_geometry_from_scene,
)
from housemaker.object_uv_scan_projection import (
    DEFAULT_PROJECTION_CAMERA_PERCENTAGES,
    SCAN_PROJECTION_TARGET_FULL,
    SCAN_PROJECTION_TARGET_LEFT_HALF,
    SCAN_PROJECTION_TARGET_TOP_LEFT_QUARTER,
    ScanProjectionCancelled,
    ScanProjectionResult,
    ScanProjectionStats,
    normalize_projection_camera_percentages,
    scan_project_textured_glb,
)
from housemaker.object_uv_raycast import (
    VISIBILITY_UV_UNWRAP_VERSION,
    VisibilityUvUnwrapStats,
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
    SYMMETRIC_SELECTION_MODE_FEWEST_TRIANGLES_RANDOM_TIE,
    SYMMETRIC_SQUARE_PAIR_CONTENT_RESOLUTIONS,
    SYMMETRIC_TEXTURE_CONTENT_HALF_LEFT,
    SYMMETRIC_TEXTURE_PACKING_MODE_PAIR,
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
    UvFaceSelectionRequest,
)
from housemaker.unused_face_removal import (
    ALL_CAMERA_IDS,
    CAMERA_OPTIONS,
    UnusedFaceRemovalOptions,
    UnusedFaceRemovalProgress,
    remove_unused_faces_from_glb,
)
from housemaker.safe_duplicate_face_removal import (
    SafeDuplicateFaceRemovalCancelled,
    SafeDuplicateFaceRemovalResult,
    remove_safe_duplicate_faces_from_glb,
)
from housemaker.video_source import (
    VIDEO_FILE_FILTER,
    VideoFrameSource,
    probe_video,
)
from housemaker.viewer import (
    FACE_SELECTION_TOGGLE,
    GlbViewerWidget,
)


# ### Constants ###
MIN_BRUSH_RADIUS_PIXELS = 2
MAX_BRUSH_RADIUS_PIXELS = 160
DEFAULT_BRUSH_RADIUS_PIXELS = 24
OBJECT_GENERATION_AMBIENT_LIGHT_INTENSITY = 1.0
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
OBJECT_FACE_GEOMETRY_CACHE_MAX_ENTRIES = 16
GEOMETRY_FINGERPRINT_DECIMALS = 6
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
TEXTURE_VARIANT_MAP_PNG_PATHS_KEY = "map_texture_asset_paths"
PBR_MAPS_AVAILABLE_PIPELINE_KEY = "pbr_maps_available"
PBR_MAPS_ENABLED_PIPELINE_KEY = "pbr_maps_enabled"
GLASS_CONVERSION_PIPELINE_KEY = "glass_conversion"
GLASS_MATERIAL_SOURCE_PREFAB = "housemaker_prefab"
SAFE_DUPLICATE_REMOVAL_PIPELINE_KEY = "safe_duplicate_face_removal"
# Legacy-only key retained so later geometry operations can discard obsolete
# masks from projects saved before Object Generation inpainting was removed.
TEXTURE_INPAINT_STROKES_PIPELINE_KEY = "texture_inpaint_strokes"
OBJECT_OPERATION_UNDO_STACK_PIPELINE_KEY = "object_operation_undo_stack"
MAX_OBJECT_OPERATION_UNDO_COUNT = 10
OBJECT_OPERATION_GENERATE_MODEL = "generate_model"
OBJECT_OPERATION_GENERATE_TEXTURE = "generate_texture"
OBJECT_OPERATION_DELETE_FACES = "delete_faces"
FACE_EDIT_REVISION_PIPELINE_KEY = "face_edit_revision"
FACE_EDIT_TEXTURE_STALE_PIPELINE_KEY = "face_edit_texture_stale"
FACE_EDIT_ATLAS_PLACEHOLDERS_PIPELINE_KEY = "face_edit_atlas_placeholders"
LOCALLY_AUTHORED_UVS_PIPELINE_KEY = "locally_authored_uvs"
VISIBILITY_UV_UNWRAP_PIPELINE_KEY = "visibility_uv_unwrap"
SCAN_PROJECTION_PIPELINE_KEY = "weighted_camera_scan_projection"
FACE_EDIT_INVALIDATED_UV_PROVENANCE_PIPELINE_KEYS = (
    VISIBILITY_UV_UNWRAP_PIPELINE_KEY,
    SCAN_PROJECTION_PIPELINE_KEY,
    "texture_regeneration_uv_fingerprint_version",
    "texture_regeneration_submitted_uv_fingerprint",
    "texture_regeneration_final_uv_fingerprint",
    "texture_regeneration_uv_face_count",
    "texture_regeneration_submitted_uv_face_count",
    "texture_regeneration_final_uv_face_count",
)
LAST_TEXTURE_FACE_REMOVAL_DETAIL_PIPELINE_KEYS = (
    "last_texture_minimum_face_visibility_percentage",
    "last_texture_face_removal_original_face_count",
    "last_texture_face_removal_retained_face_count",
    "last_texture_face_removal_removed_face_count",
    "last_texture_face_removal_visibility_removed_face_count",
    "last_texture_face_removal_stacked_face_removed_count",
    "last_texture_retexture_topology_changed",
)
GENERATION_JOB_KIND_MODEL = "Object generation"
GENERATION_JOB_KIND_TEXTURE = "Object texture generation"
GENERATION_JOB_KIND_FACE_EDIT = "Object face editing"
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


# ### Projection camera percentage helpers ###
def update_projection_camera_percentage(
    current_percentages: Sequence[int],
    camera_id: str,
    requested_percentage: int,
) -> tuple[int, ...]:
    """Update one camera without allowing the six-value total above 100%."""

    percentages = _normalize_editable_projection_camera_percentages(
        current_percentages
    )
    if camera_id not in ALL_CAMERA_IDS:
        raise ValueError(f"Unknown projection camera: {camera_id!r}.")
    if isinstance(requested_percentage, bool) or not isinstance(
        requested_percentage,
        int,
    ):
        raise ValueError("A projection camera percentage must be an integer.")

    target_index = ALL_CAMERA_IDS.index(camera_id)
    maximum_percentage = 100 - (len(ALL_CAMERA_IDS) - 1)
    target_percentage = min(
        maximum_percentage,
        max(1, int(requested_percentage)),
    )
    current_percentage = percentages[target_index]
    if target_percentage == current_percentage:
        return percentages

    updated = list(percentages)
    updated[target_index] = target_percentage
    if target_percentage < current_percentage:
        return tuple(updated)

    overflow = max(0, sum(updated) - 100)
    if overflow == 0:
        return tuple(updated)

    other_indices = tuple(
        index for index in range(len(updated)) if index != target_index
    )
    reducible_percentages = tuple(
        percentages[index] - 1 for index in other_indices
    )
    reductions = _apportion_integer_percentage(
        overflow,
        reducible_percentages,
    )
    for index, reduction in zip(other_indices, reductions, strict=True):
        updated[index] -= reduction
    return tuple(updated)


def _normalize_editable_projection_camera_percentages(
    values: Sequence[int],
) -> tuple[int, ...]:
    """Validate a live control state, which may intentionally total below 100%."""

    if isinstance(values, str | bytes | bytearray):
        raise ValueError("Projection camera percentages must be a sequence.")
    try:
        percentages = tuple(values)
    except TypeError as error:
        raise ValueError(
            "Projection camera percentages must be a sequence."
        ) from error
    if len(percentages) != len(ALL_CAMERA_IDS):
        raise ValueError("Projection camera percentages require six values.")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in percentages
    ):
        raise ValueError("Projection camera percentages must be integers.")
    normalized = tuple(int(value) for value in percentages)
    if any(value < 1 for value in normalized):
        raise ValueError("Every projection camera requires at least 1%.")
    if sum(normalized) > 100:
        raise ValueError("Projection camera percentages cannot exceed 100%.")
    return normalized


def _apportion_integer_percentage(
    amount: int,
    weights: Sequence[int],
) -> tuple[int, ...]:
    """Apportion integer points by weight with deterministic remainder ties."""

    normalized_weights = tuple(max(0, int(weight)) for weight in weights)
    if amount <= 0:
        return tuple(0 for _weight in normalized_weights)
    total_weight = sum(normalized_weights)
    if amount > total_weight:
        raise ValueError("The percentage reduction exceeds available capacity.")

    allocations: list[int] = []
    remainders: list[int] = []
    for weight in normalized_weights:
        allocation, remainder = divmod(amount * weight, total_weight)
        allocations.append(allocation)
        remainders.append(remainder)
    points_left = amount - sum(allocations)
    remainder_order = sorted(
        range(len(normalized_weights)),
        key=lambda index: (
            -remainders[index],
            -normalized_weights[index],
            index,
        ),
    )
    for index in remainder_order[:points_left]:
        allocations[index] += 1
    return tuple(allocations)


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
    failed = Signal(str, str)
    progress = Signal(str, str)
    thread_finished = Signal(str)

    def __init__(self, operation_id: str, parent: QObject) -> None:
        super().__init__(parent)
        self._operation_id = str(operation_id)

    @Slot(object, object)
    def forward_pair_succeeded(self, first: object, second: object) -> None:
        self.pair_succeeded.emit(self._operation_id, first, second)

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
    worker: GenerationWorker | TextureRegenerationWorker | ObjectFaceDeletionWorker
    relay: _ObjectJobSignalRelay
    generation_request: GenerationRequest | None = None
    requested_name: str = ""
    managed_job_id: str | None = None
    record_snapshot: GeneratedObjectRecord | None = None
    source_asset_revision: tuple[object, ...] | None = None

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
        geometry_only: bool = False,
        symmetric_division_enabled: bool = False,
        symmetric_division_orientation: str = (
            SYMMETRIC_DIVISION_ORIENTATION_VERTICAL
        ),
        projection_camera_percentages: Sequence[int] = (
            DEFAULT_PROJECTION_CAMERA_PERCENTAGES
        ),
        enabled_pbr_maps: Sequence[str] = (),
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
        self.projection_camera_percentages = (
            normalize_projection_camera_percentages(
                projection_camera_percentages
            )
        )
        self.enabled_pbr_maps = _normalize_enabled_pbr_maps(
            enabled_pbr_maps
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
    projection_camera_percentages: tuple[int, ...] = (
        DEFAULT_PROJECTION_CAMERA_PERCENTAGES
    )
    enabled_pbr_maps: tuple[str, ...] = ()
    glass_face_indices: tuple[int, ...] = ()
    glass_double_sided: bool | None = None
    preserve_existing_glass: bool = False

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
        object.__setattr__(
            self,
            "projection_camera_percentages",
            normalize_projection_camera_percentages(
                self.projection_camera_percentages
            ),
        )
        object.__setattr__(
            self,
            "enabled_pbr_maps",
            _normalize_enabled_pbr_maps(self.enabled_pbr_maps),
        )
        glass_faces = tuple(
            sorted(set(int(index) for index in self.glass_face_indices))
        )
        if any(index < 0 for index in glass_faces):
            raise ValueError("Glass face indices cannot be negative.")
        if self.preserve_existing_glass and not self.enable_original_uv:
            raise ValueError(
                "Existing glass texture regeneration must preserve original "
                "UVs."
            )
        object.__setattr__(self, "glass_face_indices", glass_faces)
        object.__setattr__(
            self,
            "glass_double_sided",
            (
                None
                if self.glass_double_sided is None
                else bool(self.glass_double_sided)
            ),
        )
        object.__setattr__(
            self,
            "preserve_existing_glass",
            bool(self.preserve_existing_glass),
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
    projection_camera_percentages: tuple[int, ...] = (
        DEFAULT_PROJECTION_CAMERA_PERCENTAGES
    )
    enabled_pbr_maps: tuple[str, ...] = ()
    glass_face_indices: tuple[int, ...] = ()
    glass_double_sided: bool | None = None
    preserve_existing_glass: bool = False

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
        object.__setattr__(
            self,
            "projection_camera_percentages",
            normalize_projection_camera_percentages(
                self.projection_camera_percentages
            ),
        )
        object.__setattr__(
            self,
            "enabled_pbr_maps",
            _normalize_enabled_pbr_maps(self.enabled_pbr_maps),
        )
        glass_faces = tuple(
            sorted(set(int(index) for index in self.glass_face_indices))
        )
        if any(index < 0 for index in glass_faces):
            raise ValueError("Glass face indices cannot be negative.")
        if self.preserve_existing_glass and not self.enable_original_uv:
            raise ValueError(
                "Existing glass texture regeneration must preserve original "
                "UVs."
            )
        object.__setattr__(self, "glass_face_indices", glass_faces)
        object.__setattr__(
            self,
            "glass_double_sided",
            (
                None
                if self.glass_double_sided is None
                else bool(self.glass_double_sided)
            ),
        )
        object.__setattr__(
            self,
            "preserve_existing_glass",
            bool(self.preserve_existing_glass),
        )


@dataclass(frozen=True)
class _MaterializedTextureRegeneration:
    """Worker-owned paid request plus its stable source revision."""

    request: TextureRegenerationRequest
    source_asset_path: str | None = None
    source_asset_revision: tuple[object, ...] | None = None


@dataclass(frozen=True)
class ObjectFaceDeletionRequest:
    """Contained GLB revisions and logical face IDs for one local edit job."""

    object_id: str
    reference_asset_path: str
    source_asset_paths: tuple[str, ...]
    selected_face_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        normalized_object_id = str(self.object_id).strip()
        reference_asset_path = _normalize_face_edit_asset_path(
            self.reference_asset_path
        )
        source_asset_paths = tuple(
            dict.fromkeys(
                _normalize_face_edit_asset_path(path)
                for path in self.source_asset_paths
            )
        )
        selected = tuple(int(index) for index in self.selected_face_indices)
        if not normalized_object_id:
            raise ValueError("Face deletion requires an object ID.")
        if (
            not source_asset_paths
            or reference_asset_path not in source_asset_paths
        ):
            raise ValueError(
                "Face deletion requires its displayed GLB revision."
            )
        if not selected or any(index < 0 for index in selected):
            raise ValueError("Face deletion requires selected face indices.")
        object.__setattr__(self, "object_id", normalized_object_id)
        object.__setattr__(
            self,
            "reference_asset_path",
            reference_asset_path,
        )
        object.__setattr__(self, "source_asset_paths", source_asset_paths)
        object.__setattr__(
            self,
            "selected_face_indices",
            tuple(sorted(set(selected))),
        )


@dataclass(frozen=True)
class PreparedObjectFaceDeletion:
    """Every edited GLB revision plus reference face-count metadata."""

    request: ObjectFaceDeletionRequest
    reference_result: ObjectFaceDeletionResult
    edited_glbs: tuple[tuple[str, bytes], ...]
    preview_model: GeneratedModel


@dataclass(frozen=True)
class TextureRegenerationOutcome:
    """Provider result plus the immutable request and verified final UVs."""

    request: TextureRegenerationRequest
    result: MeshyGenerationResult
    preserved_uv_fingerprint: UvFingerprint | None = None
    final_uv_fingerprint: UvFingerprint | None = None
    scan_projection_stats: ScanProjectionStats | None = None
    final_face_removal_applied: bool = False
    minimum_face_visibility_percentage: int = 0
    final_original_face_count: int = 0
    final_retained_face_count: int = 0
    final_removed_face_count: int = 0
    final_visibility_removed_face_count: int = 0
    final_stacked_face_removed_count: int = 0
    retexture_topology_changed: bool = False
    safe_duplicate_removed_face_count: int = 0
    safe_duplicate_group_count: int = 0


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
    variant_metadata: dict[str, dict[str, object]]
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
class SafeDuplicateProcessedMeshyGenerationResult(MeshyGenerationResult):
    """Provider result after conservative coincident-triangle cleanup."""

    safe_duplicate_removed_face_count: int = 0
    safe_duplicate_group_count: int = 0


@dataclass(frozen=True)
class StagedMeshyGenerationResult(SafeDuplicateProcessedMeshyGenerationResult):
    """Final textured result plus auditable geometry-processing revisions."""

    geometry_safe_duplicate_removed_face_count: int = 0
    geometry_safe_duplicate_group_count: int = 0
    retextured_safe_duplicate_removed_face_count: int = 0
    retextured_safe_duplicate_group_count: int = 0
    geometry_task_id: str = ""
    source_glb_bytes: bytes = b""
    postprocessed_glb_bytes: bytes = b""
    original_face_count: int = 0
    retained_face_count: int = 0
    removed_face_count: int = 0
    protected_face_count: int = 0
    visibility_removed_face_count: int = 0
    stacked_face_removed_count: int = 0
    unused_face_removal_applied: bool = False
    minimum_face_visibility_percentage: int = 0
    final_face_removal_applied: bool = False
    final_original_face_count: int = 0
    final_retained_face_count: int = 0
    final_removed_face_count: int = 0
    final_visibility_removed_face_count: int = 0
    final_stacked_face_removed_count: int = 0
    retexture_topology_changed: bool = False
    visibility_uv_stats: VisibilityUvUnwrapStats | None = None
    scan_projection_stats: ScanProjectionStats | None = None
    geometry_only: bool = False


@dataclass(frozen=True)
class ScanProjectedMeshyGenerationResult(
    SafeDuplicateProcessedMeshyGenerationResult
):
    """One directly textured result rebuilt with weighted camera UVs."""

    scan_projection_stats: ScanProjectionStats | None = None


@dataclass(frozen=True)
class _GeometryFingerprint:
    """Order-independent world-space triangle identity for one GLB."""

    face_count: int
    sha256: str


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
    map_texture_asset_relative_paths: Mapping[str, str] = field(
        default_factory=dict
    )
    map_texture_asset_paths: Mapping[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class ObjectTextureImageVariant:
    """One safe exact-resolution PNG, independent of its selectable GLB."""

    object_id: str
    object_name: str
    resolution: int
    texture_asset_relative_path: str
    texture_asset_path: Path
    map_texture_asset_relative_paths: Mapping[str, str] = field(
        default_factory=dict
    )
    map_texture_asset_paths: Mapping[str, Path] = field(default_factory=dict)


# ### Generation-pipeline decisions ###
def _normalize_enabled_pbr_maps(values: Sequence[str]) -> tuple[str, ...]:
    """Return supported Meshy PBR map IDs in stable UI order."""

    if isinstance(values, str | bytes | bytearray):
        raise ValueError("Enabled PBR maps must be a sequence.")
    try:
        normalized = {str(value).strip().lower() for value in values}
    except TypeError as error:
        raise ValueError("Enabled PBR maps must be a sequence.") from error
    unknown = normalized - set(PBR_MAP_TYPES)
    if unknown:
        raise ValueError(
            "Unknown PBR map selection: " + ", ".join(sorted(unknown))
        )
    return tuple(map_type for map_type in PBR_MAP_TYPES if map_type in normalized)


def _available_variant_pbr_maps(
    variants: PersistableObjectTextureVariants,
) -> tuple[str, ...]:
    raw_maps = getattr(variants, "map_png_by_resolution", None)
    if not isinstance(raw_maps, Mapping):
        return ()
    return tuple(map_type for map_type in PBR_MAP_TYPES if map_type in raw_maps)


def _build_pbr_pipeline_metadata(
    enabled_maps: Sequence[str],
    variants: PersistableObjectTextureVariants,
) -> dict[str, object]:
    """Persist requested contributions separately from provider availability."""

    return {
        PBR_MAPS_ENABLED_PIPELINE_KEY: list(
            _normalize_enabled_pbr_maps(enabled_maps)
        ),
        PBR_MAPS_AVAILABLE_PIPELINE_KEY: list(
            _available_variant_pbr_maps(variants)
        ),
    }


def _resolve_glass_double_sided(
    pipeline: Mapping[str, object],
    requested_value: bool | None,
) -> bool:
    """Resolve new user intent or one persisted legacy glass setting."""

    if requested_value is not None:
        return bool(requested_value)
    raw_metadata = pipeline.get(GLASS_CONVERSION_PIPELINE_KEY)
    if isinstance(raw_metadata, Mapping):
        raw_value = raw_metadata.get("double_sided")
        if isinstance(raw_value, bool):
            return raw_value
    if isinstance(raw_metadata, Mapping):
        # Every HouseMaker glass material created before this setting existed
        # was double-sided, so this is the lossless legacy fallback.
        return True
    return DEFAULT_HOUSEMAKER_GLASS_DOUBLE_SIDED


def _iter_material_leaves(material: object) -> tuple[object, ...]:
    """Flatten one trimesh material tree in face-material index order."""

    if material is None:
        return ()
    nested_materials = getattr(material, "materials", None)
    if isinstance(nested_materials, list | tuple):
        return tuple(
            leaf
            for nested_material in nested_materials
            for leaf in _iter_material_leaves(nested_material)
        )
    return (material,)


def _collect_scene_glass_face_indices(
    scene: object,
) -> frozenset[int] | None:
    """Return exact selected-face IDs using HouseMaker's canonical node order."""

    try:
        node_names = sorted(scene.graph.nodes_geometry, key=str)
        geometry_by_name = scene.geometry
    except (AttributeError, TypeError):
        return None
    glass_faces: set[int] = set()
    face_offset = 0
    for node_name in node_names:
        try:
            _transform, geometry_name = scene.graph.get(node_name)
            geometry = geometry_by_name.get(geometry_name)
            faces = np.asarray(getattr(geometry, "faces", None), dtype=np.int64)
            vertices = np.asarray(
                getattr(geometry, "vertices", None),
                dtype=float,
            )
        except (AttributeError, TypeError, ValueError):
            return None
        if (
            faces.ndim != 2
            or faces.shape[1:] != (3,)
            or vertices.ndim != 2
            or vertices.shape[1:] != (3,)
            or not len(faces)
            or not len(vertices)
        ):
            continue
        material = getattr(getattr(geometry, "visual", None), "material", None)
        material_leaves = _iter_material_leaves(material)
        raw_face_materials = getattr(
            getattr(geometry, "visual", None),
            "face_materials",
            None,
        )
        if raw_face_materials is None:
            face_materials = np.zeros(len(faces), dtype=np.int64)
        else:
            try:
                face_materials = np.asarray(
                    raw_face_materials,
                    dtype=np.int64,
                )
            except (TypeError, ValueError):
                return None
            if face_materials.shape != (len(faces),):
                return None
        for local_face_index, material_index in enumerate(face_materials):
            normalized_material_index = int(material_index)
            if (
                normalized_material_index < 0
                or normalized_material_index >= len(material_leaves)
            ):
                return None
            if is_housemaker_glass_material(
                material_leaves[normalized_material_index]
            ):
                glass_faces.add(face_offset + local_face_index)
        face_offset += len(faces)
    return frozenset(glass_faces)


def _available_metadata_pbr_maps(
    variant_metadata: Mapping[str, Mapping[str, object]],
) -> tuple[str, ...]:
    available: set[str] = set(PBR_MAP_TYPES)
    found_variant = False
    for raw_variant in variant_metadata.values():
        raw_paths = raw_variant.get(TEXTURE_VARIANT_MAP_PNG_PATHS_KEY)
        if not isinstance(raw_paths, Mapping):
            return ()
        found_variant = True
        available &= {
            str(map_type)
            for map_type, raw_path in raw_paths.items()
            if isinstance(raw_path, str) and raw_path.strip()
        }
    if not found_variant:
        return ()
    return tuple(map_type for map_type in PBR_MAP_TYPES if map_type in available)


def _build_unused_face_removal_options(
    settings: GenerationServiceSettings,
) -> UnusedFaceRemovalOptions:
    """Translate the persisted whole-percent setting into purge options."""

    return UnusedFaceRemovalOptions(
        minimum_visible_fraction=(
            settings.minimum_face_visibility_percentage / 100.0
        ),
    )


def _uses_weighted_camera_projection(request: GenerationRequest) -> bool:
    """Whether this textured generation uses the weighted scan atlas."""

    return bool(
        request.settings.use_uv_raycast_for_object_generation
        and not request.geometry_only
    )


def _texture_regeneration_scan_target(
    request: TextureRegenerationRequest,
    symmetry: ObjectSymmetricDivisionMetadata | None,
) -> str | None:
    """Return the safe atlas region for one regenerated object's new UVs."""

    if (
        not request.settings.use_uv_raycast_for_object_generation
        and not request.glass_face_indices
        and not request.preserve_existing_glass
    ):
        return None
    if symmetry is None:
        return SCAN_PROJECTION_TARGET_FULL
    if symmetry.version == SYMMETRIC_QUARTER_METADATA_VERSION:
        return SCAN_PROJECTION_TARGET_TOP_LEFT_QUARTER
    return SCAN_PROJECTION_TARGET_LEFT_HALF


def _remove_safe_duplicates_from_meshy_result(
    result: MeshyGenerationResult,
    progress_callback: Callable[[str], None] | None,
    cancel_event: threading.Event | None,
) -> tuple[MeshyGenerationResult, SafeDuplicateFaceRemovalResult]:
    """Remove only lossless same-facing duplicate triangles from one result."""

    _raise_if_generation_cancelled(cancel_event)
    if progress_callback is not None:
        progress_callback("Removing coincident duplicate faces...")
    try:
        cleanup = remove_safe_duplicate_faces_from_glb(
            result.glb_bytes,
            cancel_requested=(
                None if cancel_event is None else cancel_event.is_set
            ),
        )
    except SafeDuplicateFaceRemovalCancelled as error:
        raise _GenerationCancelled from error
    _raise_if_generation_cancelled(cancel_event)
    if cleanup.removed_face_count == 0:
        return result, cleanup
    if progress_callback is not None:
        progress_callback(
            f"Removed {cleanup.removed_face_count} coincident duplicate "
            + (
                "face."
                if cleanup.removed_face_count == 1
                else "faces."
            )
        )
    previous_removed = int(
        getattr(result, "safe_duplicate_removed_face_count", 0)
    )
    previous_groups = int(
        getattr(result, "safe_duplicate_group_count", 0)
    )
    if isinstance(result, SafeDuplicateProcessedMeshyGenerationResult):
        cleaned_result = replace(
            result,
            glb_bytes=cleanup.glb_bytes,
            safe_duplicate_removed_face_count=(
                previous_removed + cleanup.removed_face_count
            ),
            safe_duplicate_group_count=(
                previous_groups + cleanup.duplicate_group_count
            ),
        )
    else:
        cleaned_result = SafeDuplicateProcessedMeshyGenerationResult(
            task_id=result.task_id,
            glb_bytes=cleanup.glb_bytes,
            name=result.name,
            safe_duplicate_removed_face_count=cleanup.removed_face_count,
            safe_duplicate_group_count=cleanup.duplicate_group_count,
        )
    return cleaned_result, cleanup


def _scan_project_provider_result(
    request: GenerationRequest,
    provider_result: MeshyGenerationResult,
    progress_callback: Callable[[str], None] | None,
    cancel_event: threading.Event | None,
) -> MeshyGenerationResult:
    """Apply a full scan atlas after Meshy, or defer it until symmetry clips."""

    if (
        not _uses_weighted_camera_projection(request)
        or request.symmetric_division_enabled
    ):
        return provider_result
    _raise_if_generation_cancelled(cancel_event)
    if progress_callback is not None:
        progress_callback(
            "Scanning the six weighted camera projections line by line..."
        )
    try:
        projected = scan_project_textured_glb(
            provider_result.glb_bytes,
            request.projection_camera_percentages,
            target_domain=SCAN_PROJECTION_TARGET_FULL,
            cancellation_check=(
                None if cancel_event is None else cancel_event.is_set
            ),
        )
    except ScanProjectionCancelled as error:
        raise _GenerationCancelled from error
    if not isinstance(projected, ScanProjectionResult):
        raise TypeError("The weighted camera projection returned no result.")
    _raise_if_generation_cancelled(cancel_event)
    return ScanProjectedMeshyGenerationResult(
        task_id=provider_result.task_id,
        glb_bytes=projected.glb_bytes,
        name=provider_result.name,
        safe_duplicate_removed_face_count=int(
            getattr(provider_result, "safe_duplicate_removed_face_count", 0)
        ),
        safe_duplicate_group_count=int(
            getattr(provider_result, "safe_duplicate_group_count", 0)
        ),
        scan_projection_stats=projected.stats,
    )


# ### Geometry integrity helpers ###
def _build_geometry_fingerprint(glb_bytes: bytes) -> _GeometryFingerprint:
    """Hash triangle positions independent of primitive, face, and winding order."""

    model = import_generated_glb(glb_bytes)
    triangles = np.asarray(model.mesh.triangles, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not len(triangles):
        raise ValueError("Geometry validation requires triangle faces.")
    rounded = np.round(triangles, decimals=GEOMETRY_FINGERPRINT_DECIMALS)
    rounded[rounded == 0.0] = 0.0
    corner_order = np.lexsort(
        (rounded[:, :, 2], rounded[:, :, 1], rounded[:, :, 0]),
        axis=1,
    )
    canonical_corners = np.take_along_axis(
        rounded,
        corner_order[:, :, np.newaxis],
        axis=1,
    )
    flattened_faces = canonical_corners.reshape((-1, 9))
    face_order = np.lexsort(
        tuple(flattened_faces[:, index] for index in range(8, -1, -1))
    )
    canonical_faces = np.ascontiguousarray(
        flattened_faces[face_order],
        dtype="<f8",
    )
    digest = hashlib.sha256()
    digest.update(b"housemaker-world-triangles-v1\0")
    digest.update(len(canonical_faces).to_bytes(8, "little"))
    digest.update(canonical_faces.tobytes())
    return _GeometryFingerprint(
        face_count=len(canonical_faces),
        sha256=digest.hexdigest(),
    )


def _remap_faces_by_world_geometry(
    source_glb: bytes,
    target_glb: bytes,
    source_face_indices: Sequence[int],
) -> tuple[int, ...]:
    """Map selected source triangles onto a retextured GLB's face order."""

    selected = tuple(sorted(set(int(index) for index in source_face_indices)))
    if not selected:
        return ()
    source = load_object_face_geometry(source_glb)
    target = load_object_face_geometry(target_glb)
    if selected[0] < 0 or selected[-1] >= source.face_count:
        raise ValueError(
            "The selected glass faces no longer belong to the source model."
        )
    source_triangles = _canonicalize_world_triangles(
        source.vertices[source.faces[np.asarray(selected, dtype=np.int64)]]
    )
    target_triangles = _canonicalize_world_triangles(
        target.vertices[target.faces]
    )
    if not len(target_triangles):
        raise ValueError("Meshy Retexture returned no triangle faces.")
    coordinate_span = np.ptp(source.vertices, axis=0)
    model_scale = max(float(np.max(coordinate_span)), np.finfo(float).tiny)
    coordinate_magnitude = max(
        float(np.max(np.abs(source.vertices))),
        float(np.max(np.abs(target.vertices))),
        1.0,
    )
    tolerance = max(
        model_scale * 1e-6,
        np.finfo(float).eps * coordinate_magnitude * 256.0,
    )
    mapped: set[int] = set()
    for triangle in source_triangles:
        maximum_deviation = np.max(
            np.abs(target_triangles - triangle[np.newaxis, :, :]),
            axis=(1, 2),
        )
        matches = np.flatnonzero(maximum_deviation <= tolerance)
        if not len(matches):
            raise ValueError(
                "Meshy Retexture changed the selected geometry, so its glass "
                "faces could not be identified safely. The existing object "
                "was kept."
            )
        mapped.update(int(index) for index in matches)
    return tuple(sorted(mapped))


def _canonicalize_world_triangles(triangles: np.ndarray) -> np.ndarray:
    """Sort every triangle's corners so winding and vertex order do not matter."""

    normalized = np.asarray(triangles, dtype=np.float64)
    if normalized.ndim != 3 or normalized.shape[1:] != (3, 3):
        raise ValueError("Face remapping requires triangle geometry.")
    if not np.all(np.isfinite(normalized)):
        raise ValueError("Face remapping requires finite triangle geometry.")
    corner_order = np.lexsort(
        (normalized[:, :, 2], normalized[:, :, 1], normalized[:, :, 0]),
        axis=1,
    )
    return np.ascontiguousarray(
        np.take_along_axis(
            normalized,
            corner_order[:, :, np.newaxis],
            axis=1,
        ),
        dtype=np.float64,
    )


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
        face_removal_options = (
            _build_unused_face_removal_options(request.settings)
            if use_unused_face_removal
            else None
        )
        if not request.geometry_only and not use_unused_face_removal:
            _raise_if_generation_cancelled(cancel_event)
            provider_result = request_image_to_3d_model(
                api_key=request.settings.meshy_api_key,
                image_png=image_png,
                target_polycount=request.settings.meshy_target_polycount,
                progress_callback=report_generation_progress,
                cancel_event=cancel_event,
                enable_pbr=bool(request.enabled_pbr_maps),
            )
            provider_result, _duplicate_cleanup = (
                _remove_safe_duplicates_from_meshy_result(
                    provider_result,
                    progress_callback,
                    cancel_event,
                )
            )
            return _scan_project_provider_result(
                request,
                provider_result,
                progress_callback,
                cancel_event,
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

        cleaned_geometry_result, geometry_duplicate_cleanup = (
            _remove_safe_duplicates_from_meshy_result(
                geometry_result,
                progress_callback,
                cancel_event,
            )
        )
        processed_glb_bytes = cleaned_geometry_result.glb_bytes
        original_face_count = 0
        retained_face_count = 0
        removed_face_count = 0
        protected_face_count = 0
        visibility_removed_face_count = 0
        stacked_face_removed_count = 0
        if use_unused_face_removal:
            assert face_removal_options is not None

            def report_face_removal(
                update: UnusedFaceRemovalProgress,
                *,
                final_texture: bool = False,
            ) -> None:
                if progress_callback is None:
                    return
                phase_label = "final textured" if final_texture else "generated"
                if update.stage == "capturing":
                    camera_suffix = (
                        ""
                        if update.camera_id is None
                        else f" ({update.camera_id})"
                    )
                    progress_callback(
                        f"Capturing {phase_label} unused-face views"
                        + camera_suffix
                        + "..."
                    )
                elif update.stage == "checking":
                    progress_callback(
                        f"Checking {phase_label} faces: "
                        f"{update.completed_face_count}/"
                        f"{update.total_face_count}"
                    )
                elif update.stage == "exporting":
                    progress_callback(f"Saving {phase_label} visible geometry...")

            removed = remove_unused_faces_from_glb(
                processed_glb_bytes,
                options=face_removal_options,
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
            visibility_removed_face_count = (
                removed.visibility_removed_face_count
            )
            stacked_face_removed_count = removed.stacked_face_removed_count

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
                visibility_removed_face_count=visibility_removed_face_count,
                stacked_face_removed_count=stacked_face_removed_count,
                safe_duplicate_removed_face_count=(
                    geometry_duplicate_cleanup.removed_face_count
                ),
                safe_duplicate_group_count=(
                    geometry_duplicate_cleanup.duplicate_group_count
                ),
                geometry_safe_duplicate_removed_face_count=(
                    geometry_duplicate_cleanup.removed_face_count
                ),
                geometry_safe_duplicate_group_count=(
                    geometry_duplicate_cleanup.duplicate_group_count
                ),
                unused_face_removal_applied=use_unused_face_removal,
                minimum_face_visibility_percentage=(
                    request.settings.minimum_face_visibility_percentage
                ),
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
        submitted_geometry_fingerprint = _build_geometry_fingerprint(
            processed_glb_bytes
        )
        textured_result = request_retextured_model(
            api_key=request.settings.meshy_api_key,
            model_glb=processed_glb_bytes,
            reference_images_png=(image_png,),
            enable_original_uv=False,
            progress_callback=report_texture_progress,
            cancel_event=cancel_event,
            enable_pbr=bool(request.enabled_pbr_maps),
        )
        _raise_if_generation_cancelled(cancel_event)
        textured_result, final_duplicate_cleanup = (
            _remove_safe_duplicates_from_meshy_result(
                textured_result,
                progress_callback,
                cancel_event,
            )
        )
        returned_geometry_fingerprint = _build_geometry_fingerprint(
            textured_result.glb_bytes
        )
        retexture_topology_changed = (
            submitted_geometry_fingerprint != returned_geometry_fingerprint
        )
        if progress_callback is not None:
            progress_callback("Verifying final textured face visibility...")
        assert face_removal_options is not None
        final_removed = remove_unused_faces_from_glb(
            textured_result.glb_bytes,
            options=face_removal_options,
            cancel_requested=(
                None if cancel_event is None else cancel_event.is_set
            ),
            progress_callback=lambda update: report_face_removal(
                update,
                final_texture=True,
            ),
        )
        textured_result = MeshyGenerationResult(
            task_id=textured_result.task_id,
            glb_bytes=final_removed.glb_bytes,
            name=textured_result.name,
        )
        final_result = _scan_project_provider_result(
            request,
            textured_result,
            progress_callback,
            cancel_event,
        )
        return StagedMeshyGenerationResult(
            task_id=final_result.task_id,
            glb_bytes=final_result.glb_bytes,
            name=final_result.name,
            geometry_task_id=geometry_result.task_id,
            source_glb_bytes=geometry_result.glb_bytes,
            postprocessed_glb_bytes=processed_glb_bytes,
            original_face_count=original_face_count,
            retained_face_count=retained_face_count,
            removed_face_count=removed_face_count,
            protected_face_count=protected_face_count,
            visibility_removed_face_count=visibility_removed_face_count,
            stacked_face_removed_count=stacked_face_removed_count,
            safe_duplicate_removed_face_count=(
                geometry_duplicate_cleanup.removed_face_count
                + final_duplicate_cleanup.removed_face_count
            ),
            safe_duplicate_group_count=(
                geometry_duplicate_cleanup.duplicate_group_count
                + final_duplicate_cleanup.duplicate_group_count
            ),
            geometry_safe_duplicate_removed_face_count=(
                geometry_duplicate_cleanup.removed_face_count
            ),
            geometry_safe_duplicate_group_count=(
                geometry_duplicate_cleanup.duplicate_group_count
            ),
            retextured_safe_duplicate_removed_face_count=(
                final_duplicate_cleanup.removed_face_count
            ),
            retextured_safe_duplicate_group_count=(
                final_duplicate_cleanup.duplicate_group_count
            ),
            unused_face_removal_applied=use_unused_face_removal,
            minimum_face_visibility_percentage=(
                request.settings.minimum_face_visibility_percentage
            ),
            final_face_removal_applied=True,
            final_original_face_count=final_removed.original_face_count,
            final_retained_face_count=final_removed.retained_face_count,
            final_removed_face_count=final_removed.removed_face_count,
            final_visibility_removed_face_count=(
                final_removed.visibility_removed_face_count
            ),
            final_stacked_face_removed_count=(
                final_removed.stacked_face_removed_count
            ),
            retexture_topology_changed=retexture_topology_changed,
            scan_projection_stats=(
                final_result.scan_projection_stats
                if isinstance(
                    final_result,
                    ScanProjectedMeshyGenerationResult,
                )
                else None
            ),
        )


# ### Texture regeneration adapter ###
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
            enable_pbr=bool(request.enabled_pbr_maps),
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

    projection_camera_percentages_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._is_external_presentation_active = False
        self._layout = QBoxLayout(QBoxLayout.Direction.TopToBottom, self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)

        self.viewer = GlbViewerWidget(
            wireframe_enabled=False,
            face_editing_enabled=True,
        )
        self.viewer.set_projection_camera_indicators_visible(True)
        self.viewer.projection_camera_percentage_step_requested.connect(
            self.adjust_projection_camera_percentage
        )
        self._layout.addWidget(self.viewer, 1)

        self.details_panel = QWidget()
        self.details_panel.setObjectName("object_generation_details_panel")
        details_layout = QVBoxLayout(self.details_panel)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(4)

        self.projection_camera_controls = QWidget()
        self.projection_camera_controls.setObjectName(
            "projection_camera_controls"
        )
        camera_layout = QGridLayout(self.projection_camera_controls)
        camera_layout.setContentsMargins(4, 2, 4, 2)
        camera_layout.setSpacing(6)
        camera_title = QLabel("Projection texture allocation")
        camera_title.setToolTip(
            "Allocate the available texture area between the six fixed "
            "projection cameras."
        )
        camera_layout.addWidget(camera_title, 0, 0, 1, 4)
        self.projection_camera_percentage_spinboxes: dict[str, QSpinBox] = {}
        for index, ((camera_id, label), default_percentage) in enumerate(
            zip(
                CAMERA_OPTIONS,
                DEFAULT_PROJECTION_CAMERA_PERCENTAGES,
                strict=True,
            )
        ):
            row = 1 + index // 2
            column = (index % 2) * 2
            camera_label = QLabel(label)
            percentage_spinbox = QSpinBox()
            percentage_spinbox.setObjectName(
                f"projection_camera_{camera_id}_percentage"
            )
            percentage_spinbox.setRange(
                1,
                100 - (len(ALL_CAMERA_IDS) - 1),
            )
            percentage_spinbox.setSuffix("%")
            percentage_spinbox.setValue(default_percentage)
            percentage_spinbox.setKeyboardTracking(False)
            percentage_spinbox.setToolTip(
                f"The {label} camera receives this percentage of the UV "
                "texture area. Increases never let the total exceed 100%, "
                "and every camera keeps at least 1%."
            )
            percentage_spinbox.valueChanged.connect(
                lambda value, selected_camera_id=camera_id: (
                    self.set_projection_camera_percentage(
                        selected_camera_id,
                        value,
                    )
                )
            )
            self.projection_camera_percentage_spinboxes[camera_id] = (
                percentage_spinbox
            )
            camera_layout.addWidget(camera_label, row, column)
            camera_layout.addWidget(percentage_spinbox, row, column + 1)
        self.projection_camera_total_label = QLabel()
        self.projection_camera_total_label.setObjectName(
            "projection_camera_percentage_total"
        )
        camera_layout.addWidget(
            self.projection_camera_total_label,
            4,
            0,
            1,
            4,
        )
        details_layout.addWidget(self.projection_camera_controls)
        self._projection_camera_percentages = tuple(
            DEFAULT_PROJECTION_CAMERA_PERCENTAGES
        )
        self.viewer.set_projection_camera_percentages(
            self._projection_camera_percentages
        )
        self._sync_projection_camera_percentage_total()

        self.object_list = QListWidget()
        self.object_list.setObjectName("generated_objects_list")
        self.object_list.setMaximumHeight(OBJECT_LIST_MAXIMUM_HEIGHT)
        self.object_list.setAlternatingRowColors(True)
        self.object_list.setToolTip(
            "Select which generated Meshy object is shown in the 3D view."
        )
        details_layout.addWidget(self.object_list, 1)

        self.face_selection_help_label = QLabel(
            "Toggle faces with Ctrl+click in 3D or a click in Texture "
            "resolution. Ctrl+drag adds faces in 3D. All methods share one "
            "selection; deletion retains existing UVs and textures."
        )
        self.face_selection_help_label.setObjectName(
            "object_face_selection_help"
        )
        self.face_selection_help_label.setWordWrap(True)
        details_layout.addWidget(self.face_selection_help_label)

        self.face_selection_count_label = QLabel("No faces selected")
        self.face_selection_count_label.setObjectName(
            "object_face_selection_count"
        )
        details_layout.addWidget(self.face_selection_count_label)

        self.delete_faces_button = QPushButton("Delete selected faces")
        self.delete_faces_button.setObjectName(
            "delete_generated_object_faces_button"
        )
        self.delete_faces_button.setToolTip(
            "Delete the selected triangles without changing the remaining "
            "UVs or texture."
        )
        self.delete_faces_button.setEnabled(False)
        details_layout.addWidget(self.delete_faces_button)

        self.convert_faces_to_glass_button = QPushButton(
            "Convert faces to glass"
        )
        self.convert_faces_to_glass_button.setObjectName(
            "convert_generated_object_faces_to_glass_button"
        )
        self.convert_faces_to_glass_button.setToolTip(
            "Join the selection into one best-fit two-triangle rectangle, "
            "rebuild the object's UVs, generate its PBR maps, and assign an "
            "atlas-independent reflective glass material."
        )
        self.convert_faces_to_glass_button.setEnabled(False)
        self.glass_double_sided_checkbox = QCheckBox("Double-sided")
        self.glass_double_sided_checkbox.setObjectName(
            "glass_double_sided_checkbox"
        )
        self.glass_double_sided_checkbox.setChecked(
            DEFAULT_HOUSEMAKER_GLASS_DOUBLE_SIDED
        )
        self.glass_double_sided_checkbox.setToolTip(
            "Render converted glass from both sides. Uncheck this to cull "
            "the panel's back face."
        )
        self.glass_conversion_controls = QWidget()
        self.glass_conversion_controls.setObjectName(
            "glass_conversion_controls"
        )
        glass_conversion_layout = QHBoxLayout(
            self.glass_conversion_controls
        )
        glass_conversion_layout.setContentsMargins(0, 0, 0, 0)
        glass_conversion_layout.setSpacing(6)
        glass_conversion_layout.addWidget(
            self.convert_faces_to_glass_button,
            1,
        )
        glass_conversion_layout.addWidget(
            self.glass_double_sided_checkbox,
        )
        details_layout.addWidget(self.glass_conversion_controls)

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

    # ### Projection camera controls ###
    def get_projection_camera_percentages(self) -> tuple[int, ...]:
        """Return the six percentages in canonical camera order."""

        return self._projection_camera_percentages

    def set_projection_camera_percentage(
        self,
        camera_id: str,
        percentage: int,
    ) -> tuple[int, ...]:
        """Apply one field or viewport adjustment through the shared pipeline."""

        updated = update_projection_camera_percentage(
            self._projection_camera_percentages,
            camera_id,
            percentage,
        )
        if updated == self._projection_camera_percentages:
            return updated
        self._projection_camera_percentages = updated
        previous_signal_states: list[tuple[QSpinBox, bool]] = []
        try:
            for selected_camera_id, value in zip(
                ALL_CAMERA_IDS,
                updated,
                strict=True,
            ):
                spinbox = self.projection_camera_percentage_spinboxes[
                    selected_camera_id
                ]
                previous_signal_states.append(
                    (spinbox, spinbox.blockSignals(True))
                )
                spinbox.setValue(value)
        finally:
            for spinbox, was_blocked in previous_signal_states:
                spinbox.blockSignals(was_blocked)
        self.viewer.set_projection_camera_percentages(updated)
        self._sync_projection_camera_percentage_total()
        self.projection_camera_percentages_changed.emit()
        return updated

    def adjust_projection_camera_percentage(
        self,
        camera_id: str,
        step_count: int,
    ) -> tuple[int, ...]:
        """Adjust one camera by one percentage point per viewport wheel tick."""

        if isinstance(step_count, bool) or not isinstance(step_count, int):
            raise ValueError("A projection camera step count must be an integer.")
        if camera_id not in ALL_CAMERA_IDS:
            raise ValueError(f"Unknown projection camera: {camera_id!r}.")
        camera_index = ALL_CAMERA_IDS.index(camera_id)
        return self.set_projection_camera_percentage(
            camera_id,
            self._projection_camera_percentages[camera_index] + step_count,
        )

    def projection_camera_percentages_are_valid(self) -> bool:
        return sum(self.get_projection_camera_percentages()) == 100

    def _sync_projection_camera_percentage_total(self) -> None:
        total = sum(self.get_projection_camera_percentages())
        is_valid = total == 100
        self.projection_camera_total_label.setText(
            f"Total: {total}%"
            + ("" if is_valid else " — must equal 100%")
        )
        self.projection_camera_total_label.setStyleSheet(
            "color: #aeb7c5;" if is_valid else "color: #ff6b6b;"
        )

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
        self._uses_staged_pipeline = bool(
            request.geometry_only
            or request.settings.unused_face_removal
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
                _run_interruptible_stage(
                    prepare_executor,
                    self._cancel_event,
                )
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
                ),
                self._cancel_event,
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
                lambda: _invoke_executor(self._executor, result),
                self._cancel_event,
            )
            if not isinstance(generated_model, GeneratedModel):
                raise TypeError("The Meshy executor returned an invalid model.")
            _raise_if_generation_cancelled(self._cancel_event)
            success_payload: object = result
            if self._asset_directory is not None and self._object_id is not None:
                self.progress.emit(
                    (
                        "Scanning weighted camera projections and saving "
                        "assets (94%)"
                    )
                    if (
                        self._request.symmetric_division_enabled
                        and _uses_weighted_camera_projection(self._request)
                    )
                    else (
                        "Compacting existing symmetric UVs and saving assets "
                        "(94%)"
                        if self._request.symmetric_division_enabled
                        else "Saving local object assets (94%)"
                    )
                )
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
                _run_interruptible_stage(
                    prepare_executor,
                    self._cancel_event,
                )
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
                ),
                self._cancel_event,
            )
            if not isinstance(result, MeshyGenerationResult):
                raise TypeError("Meshy returned an invalid texture result.")
            _raise_if_generation_cancelled(self._cancel_event)
            safe_duplicate_cleanup: SafeDuplicateFaceRemovalResult | None = None
            if not request.enable_original_uv:
                result, safe_duplicate_cleanup = (
                    _remove_safe_duplicates_from_meshy_result(
                        result,
                        self.progress.emit,
                        self._cancel_event,
                    )
                )
            scan_glass_face_indices = request.glass_face_indices
            if request.glass_face_indices and not request.enable_original_uv:
                self.progress.emit(
                    "Matching selected faces to the newly textured model (81%)"
                )
                scan_glass_face_indices = _remap_faces_by_world_geometry(
                    request.model_glb,
                    result.glb_bytes,
                    request.glass_face_indices,
                )
                _raise_if_generation_cancelled(self._cancel_event)
            if request.enable_original_uv:
                # Locally retained UVs and geometry remain authoritative.
                # Meshy contributes only its new atlas, so deleted faces and
                # provider retriangulation can never return to the object.
                self.progress.emit(
                    "Applying texture to preserved local geometry (82%)"
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
            final_face_removal = None
            retexture_topology_changed = False
            if (
                request.settings.unused_face_removal
                and not request.enable_original_uv
                and not request.glass_face_indices
            ):
                self.progress.emit(
                    "Verifying regenerated texture face visibility (82%)"
                )
                submitted_geometry_fingerprint = _build_geometry_fingerprint(
                    request.model_glb
                )
                returned_geometry_fingerprint = _build_geometry_fingerprint(
                    result.glb_bytes
                )
                retexture_topology_changed = (
                    submitted_geometry_fingerprint
                    != returned_geometry_fingerprint
                )

                def report_final_face_removal(
                    update: UnusedFaceRemovalProgress,
                ) -> None:
                    if update.stage == "capturing":
                        camera_suffix = (
                            ""
                            if update.camera_id is None
                            else f" ({update.camera_id})"
                        )
                        self.progress.emit(
                            "Capturing regenerated texture faces"
                            + camera_suffix
                            + "..."
                        )
                    elif update.stage == "checking":
                        self.progress.emit(
                            "Checking regenerated texture faces: "
                            f"{update.completed_face_count}/"
                            f"{update.total_face_count}"
                        )
                    elif update.stage == "exporting":
                        self.progress.emit(
                            "Saving regenerated visible geometry..."
                        )

                final_face_removal = remove_unused_faces_from_glb(
                    result.glb_bytes,
                    options=_build_unused_face_removal_options(request.settings),
                    cancel_requested=self._cancel_event.is_set,
                    progress_callback=report_final_face_removal,
                )
                result = MeshyGenerationResult(
                    task_id=result.task_id,
                    glb_bytes=final_face_removal.glb_bytes,
                    name=result.name,
                )
                _raise_if_generation_cancelled(self._cancel_event)
            preserved_uv_fingerprint = self._validate_final_uvs(
                result,
                request,
            )
            scan_projection_stats = None
            scan_target = _texture_regeneration_scan_target(
                request,
                self._symmetry,
            )
            if scan_target is not None:
                projection_stage = (
                    "Joining selected glass faces and rebuilding UVs (84%)"
                    if request.glass_face_indices
                    else "Rebuilding UVs from the current weighted cameras "
                    "(84%)"
                )
                self.progress.emit(projection_stage)
                try:
                    projected = scan_project_textured_glb(
                        result.glb_bytes,
                        request.projection_camera_percentages,
                        target_domain=scan_target,
                        glass_face_indices=scan_glass_face_indices,
                        glass_double_sided=request.glass_double_sided,
                        cancellation_check=self._cancel_event.is_set,
                    )
                except ScanProjectionCancelled as error:
                    raise _GenerationCancelled from error
                if not isinstance(projected, ScanProjectionResult):
                    raise TypeError(
                        "The weighted camera projection returned no result."
                    )
                result = MeshyGenerationResult(
                    task_id=result.task_id,
                    glb_bytes=projected.glb_bytes,
                    name=result.name,
                )
                scan_projection_stats = projected.stats
                _raise_if_generation_cancelled(self._cancel_event)
            final_uv_fingerprint = (
                build_uv_fingerprint(result.glb_bytes)
                if (
                    request.submitted_uv_fingerprint is not None
                    or scan_projection_stats is not None
                )
                else None
            )
            self.progress.emit(
                "Preparing local 512, 1024 and 2048 texture variants (87%)"
            )
            generated_model = _run_interruptible_stage(
                lambda: _invoke_executor(self._executor, result),
                self._cancel_event,
            )
            if not isinstance(generated_model, GeneratedModel):
                raise TypeError("The Meshy executor returned an invalid model.")
            _raise_if_generation_cancelled(self._cancel_event)
            self.progress.emit("Local texture preparation complete. Saving...")
            outcome = TextureRegenerationOutcome(
                request=request,
                result=result,
                preserved_uv_fingerprint=preserved_uv_fingerprint,
                final_uv_fingerprint=final_uv_fingerprint,
                scan_projection_stats=scan_projection_stats,
                final_face_removal_applied=(final_face_removal is not None),
                minimum_face_visibility_percentage=(
                    request.settings.minimum_face_visibility_percentage
                    if final_face_removal is not None
                    else 0
                ),
                final_original_face_count=(
                    final_face_removal.original_face_count
                    if final_face_removal is not None
                    else 0
                ),
                final_retained_face_count=(
                    final_face_removal.retained_face_count
                    if final_face_removal is not None
                    else 0
                ),
                final_removed_face_count=(
                    final_face_removal.removed_face_count
                    if final_face_removal is not None
                    else 0
                ),
                final_visibility_removed_face_count=(
                    final_face_removal.visibility_removed_face_count
                    if final_face_removal is not None
                    else 0
                ),
                final_stacked_face_removed_count=(
                    final_face_removal.stacked_face_removed_count
                    if final_face_removal is not None
                    else 0
                ),
                retexture_topology_changed=retexture_topology_changed,
                safe_duplicate_removed_face_count=(
                    0
                    if safe_duplicate_cleanup is None
                    else safe_duplicate_cleanup.removed_face_count
                ),
                safe_duplicate_group_count=(
                    0
                    if safe_duplicate_cleanup is None
                    else safe_duplicate_cleanup.duplicate_group_count
                ),
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
        _validate_preserved_uv_retexture_integrity(
            submitted,
            final_fingerprint,
        )
        return final_fingerprint


# ### Local face-edit preparation ###
def _normalize_face_edit_asset_path(raw_path: object) -> str:
    """Normalize one contained relative GLB path without touching disk."""

    normalized = str(raw_path).strip()
    path = Path(normalized)
    if (
        not normalized
        or path.is_absolute()
        or ".." in path.parts
        or path.suffix.lower() != ".glb"
    ):
        raise ValueError("A face-edit source GLB path is unsafe.")
    return path.as_posix()


def _get_face_edit_glb_asset_paths(
    record: GeneratedObjectRecord,
) -> tuple[str, ...]:
    """Return every GLB revision that may later display this object."""

    raw_paths: list[object] = [record.asset_path]
    raw_postprocessed_path = record.pipeline.get("postprocessed_asset_path")
    if isinstance(raw_postprocessed_path, str):
        raw_paths.append(raw_postprocessed_path)
    raw_variants = record.pipeline.get(TEXTURE_VARIANTS_PIPELINE_KEY)
    if isinstance(raw_variants, Mapping):
        for raw_variant in raw_variants.values():
            if not isinstance(raw_variant, Mapping):
                continue
            raw_glb_path = raw_variant.get(TEXTURE_VARIANT_GLB_PATH_KEY)
            if isinstance(raw_glb_path, str):
                raw_paths.append(raw_glb_path)
    return tuple(
        dict.fromkeys(
            _normalize_face_edit_asset_path(raw_path)
            for raw_path in raw_paths
        )
    )


def _resolve_face_edit_source_path(
    asset_directory: Path,
    raw_path: str,
) -> Path:
    """Resolve one face-edit GLB inside the generated-asset directory."""

    asset_root = Path(asset_directory).resolve()
    candidate = (
        asset_root / _normalize_face_edit_asset_path(raw_path)
    ).resolve()
    try:
        candidate.relative_to(asset_root)
    except ValueError as error:
        raise ValueError("A face-edit source GLB is outside the project.") from error
    if candidate.suffix.lower() != ".glb" or not candidate.is_file():
        raise ValueError("A face-edit source GLB is missing.")
    return candidate


def _build_face_edit_source_revisions(
    asset_directory: Path,
    source_asset_paths: Sequence[str],
) -> tuple[tuple[object, ...], ...]:
    """Snapshot every source file so a stale worker result cannot commit."""

    return tuple(
        (
            _normalize_face_edit_asset_path(raw_path),
            *_build_generation_asset_revision(asset_directory, raw_path),
        )
        for raw_path in source_asset_paths
    )


def _prepare_object_face_deletion(
    asset_directory: Path,
    request: ObjectFaceDeletionRequest,
) -> PreparedObjectFaceDeletion:
    """Read, validate, and edit every geometry-bearing saved GLB revision."""

    source_glbs = tuple(
        (
            raw_path,
            _resolve_face_edit_source_path(
                asset_directory,
                raw_path,
            ).read_bytes(),
        )
        for raw_path in request.source_asset_paths
    )
    reference_geometry: ObjectFaceGeometry | None = None
    edited_glbs: list[tuple[str, bytes]] = []
    reference_result: ObjectFaceDeletionResult | None = None
    retained_face_counts: list[int] = []
    ordered_source_glbs = tuple(
        sorted(
            source_glbs,
            key=lambda pair: pair[0] != request.reference_asset_path,
        )
    )
    for raw_path, source_glb in ordered_source_glbs:
        result, source_geometry = (
            _delete_object_faces_preserving_uvs_with_geometry(
                source_glb,
                request.selected_face_indices,
                validate_export=False,
            )
        )
        if reference_geometry is None:
            reference_geometry = source_geometry
        elif not _face_edit_geometry_matches(
            reference_geometry,
            source_geometry,
        ):
            raise ValueError(
                "Saved texture revisions do not share the displayed face "
                "index layout. The existing object was kept."
            )
        if reference_result is None and raw_path == request.reference_asset_path:
            reference_result = result
        edited_glbs.append((raw_path, result.glb_bytes))
        retained_face_counts.append(result.retained_face_count)
    if reference_result is None:
        raise ValueError("The displayed face-edit revision is unavailable.")
    if any(
        retained_face_count != reference_result.retained_face_count
        for retained_face_count in retained_face_counts
    ):
        raise ValueError("Face deletion changed inconsistent saved revisions.")
    edited_by_path = dict(edited_glbs)
    preview_model = import_generated_glb(
        edited_by_path[request.reference_asset_path]
    )
    if len(preview_model.mesh.faces) != reference_result.retained_face_count:
        raise ValueError("Face deletion exported an unexpected face count.")
    return PreparedObjectFaceDeletion(
        request=request,
        reference_result=reference_result,
        edited_glbs=tuple(edited_glbs),
        preview_model=preview_model,
    )


def _face_edit_geometry_matches(
    reference: ObjectFaceGeometry,
    candidate: ObjectFaceGeometry,
) -> bool:
    """Require identical ordered world triangles across texture revisions."""

    if reference.faces.shape != candidate.faces.shape:
        return False
    reference_triangles = reference.vertices[reference.faces]
    candidate_triangles = candidate.vertices[candidate.faces]
    if reference_triangles.shape != candidate_triangles.shape:
        return False
    span = np.ptp(reference.vertices, axis=0)
    scale = max(float(np.max(span)), np.finfo(float).tiny)
    magnitude = max(
        float(np.max(np.abs(reference.vertices))),
        1.0,
    )
    tolerance = max(
        scale * 1e-7,
        np.finfo(float).eps * magnitude * 64.0,
    )
    return bool(
        np.all(np.isfinite(candidate_triangles))
        and np.allclose(
            reference_triangles,
            candidate_triangles,
            rtol=0.0,
            atol=tolerance,
        )
    )


def _rewrite_face_edit_glb_paths(
    record: GeneratedObjectRecord,
    pipeline: dict[str, object],
    replacement_paths: Mapping[str, str],
) -> str:
    """Point every saved geometry revision at its edited GLB counterpart."""

    try:
        next_asset_path = replacement_paths[
            _normalize_face_edit_asset_path(record.asset_path)
        ]
    except KeyError as error:
        raise ValueError("The displayed face-edit GLB was not saved.") from error
    for pipeline_key in MESHY_REVISION_ASSET_PIPELINE_KEYS:
        raw_path = pipeline.get(pipeline_key)
        normalized_path = (
            _normalize_face_edit_asset_path(raw_path)
            if isinstance(raw_path, str)
            else None
        )
        if normalized_path in replacement_paths:
            pipeline[pipeline_key] = replacement_paths[normalized_path]
    raw_variants = pipeline.get(TEXTURE_VARIANTS_PIPELINE_KEY)
    if isinstance(raw_variants, dict):
        for raw_variant in raw_variants.values():
            if not isinstance(raw_variant, dict):
                continue
            raw_glb_path = raw_variant.get(TEXTURE_VARIANT_GLB_PATH_KEY)
            normalized_path = (
                _normalize_face_edit_asset_path(raw_glb_path)
                if isinstance(raw_glb_path, str)
                else None
            )
            if normalized_path in replacement_paths:
                raw_variant[TEXTURE_VARIANT_GLB_PATH_KEY] = (
                    replacement_paths[normalized_path]
                )
    if not isinstance(pipeline.get("postprocessed_asset_path"), str):
        canonical_resolution = _canonical_texture_resolution(record)
        canonical_variant = (
            raw_variants.get(str(canonical_resolution))
            if isinstance(raw_variants, dict)
            else None
        )
        pipeline["postprocessed_asset_path"] = (
            canonical_variant.get(TEXTURE_VARIANT_GLB_PATH_KEY)
            if isinstance(canonical_variant, dict)
            else next_asset_path
        )
    return next_asset_path


# ### Local face-edit worker ###
class ObjectFaceDeletionWorker(QObject):
    """Delete faces from every selectable GLB without blocking Qt."""

    succeeded = Signal(object, object)
    failed = Signal(str)
    finished = Signal()
    progress = Signal(str)

    def __init__(
        self,
        request: ObjectFaceDeletionRequest,
        asset_directory: Path,
    ) -> None:
        super().__init__()
        self._request = request
        self._asset_directory = Path(asset_directory)
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    @Slot()
    def run(self) -> None:
        try:
            _raise_if_generation_cancelled(self._cancel_event)
            self.progress.emit("Deleting selected faces (10%)")
            request = self._request
            result = _run_interruptible_stage(
                lambda: _prepare_object_face_deletion(
                    self._asset_directory,
                    request,
                ),
                self._cancel_event,
            )
            _raise_if_generation_cancelled(self._cancel_event)
            self.progress.emit("Existing UVs and textures retained (90%)")
        except _GenerationCancelled:
            return
        except Exception as error:
            if not self._cancel_event.is_set():
                self.failed.emit(str(error).strip() or error.__class__.__name__)
            return
        else:
            self.succeeded.emit(result, None)
        finally:
            self.finished.emit()


# ### Generation workspace ###
class GenerationWorkspace(QWidget):
    """Manual video selection and Meshy Image-to-3D workspace."""

    data_changed = Signal(object)
    generated_object_deleted = Signal(str)
    generation_completed = Signal(object, object)
    texture_regeneration_completed = Signal(object, object)
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
            | ObjectFaceDeletionWorker
            | None
        ) = None
        self._generated_model: GeneratedModel | None = None
        self._generated_model_cache: dict[str, GeneratedModel] = {}
        self._generated_model_cache_revisions: dict[
            str,
            tuple[object, ...],
        ] = {}
        self._object_face_geometry_cache: dict[
            tuple[object, ...],
            ObjectFaceGeometry,
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
        selection_context_token = (
            _build_object_face_selection_context_token(
                record,
                self._asset_directory,
            )
        )
        if (
            display_snapshot == self._displayed_object_snapshot
            and self._generated_model is not None
            and self.result_view.model is self._generated_model
            and self.result_view.face_edit_face_count > 0
            and self.texture_view.uv_face_selection_context_token
            == selection_context_token
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
        self._object_face_geometry_cache.clear()
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
                (
                    ()
                    if variant is None
                    else tuple(
                        (
                            map_type,
                            raw_path,
                            _build_generation_asset_revision(
                                self._asset_directory,
                                raw_path,
                            ),
                        )
                        for map_type, raw_path in sorted(
                            variant[
                                TEXTURE_VARIANT_MAP_PNG_PATHS_KEY
                            ].items()
                        )
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
            map_texture_asset_relative_paths=(
                image_variant.map_texture_asset_relative_paths
            ),
            map_texture_asset_paths=image_variant.map_texture_asset_paths,
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
        if record.pipeline.get(FACE_EDIT_TEXTURE_STALE_PIPELINE_KEY) is True:
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
        return self._resolve_texture_image_variant_path(
            record,
            normalized_resolution,
            variant[TEXTURE_VARIANT_PNG_PATH_KEY],
            variant[TEXTURE_VARIANT_MAP_PNG_PATHS_KEY],
        )

    def get_atlas_texture_image_variant(
        self,
        object_id: str,
        resolution: int,
    ) -> ObjectTextureImageVariant | None:
        """Resolve a current texture or a face-edit Atlas placeholder."""

        current_variant = self.get_texture_image_variant(
            object_id,
            resolution,
        )
        if current_variant is not None:
            return current_variant
        record = self._find_generated_object_record(object_id)
        if record is None:
            return None
        return self.resolve_atlas_texture_image_variant_for_record(
            record,
            resolution,
        )

    def resolve_atlas_texture_image_variant_for_record(
        self,
        record: GeneratedObjectRecord,
        resolution: int,
    ) -> ObjectTextureImageVariant | None:
        """Resolve prepared Atlas pixels without exposing stale UI variants."""

        if not isinstance(record, GeneratedObjectRecord):
            return None
        if record.pipeline.get(FACE_EDIT_TEXTURE_STALE_PIPELINE_KEY) is not True:
            return self.resolve_texture_image_variant_for_record(
                record,
                resolution,
            )
        try:
            normalized_resolution = int(resolution)
        except (TypeError, ValueError):
            return None
        if normalized_resolution not in _selectable_texture_resolutions(record):
            return None
        raw_placeholders = record.pipeline.get(
            FACE_EDIT_ATLAS_PLACEHOLDERS_PIPELINE_KEY
        )
        if not isinstance(raw_placeholders, Mapping):
            return None
        texture_relative_path = raw_placeholders.get(
            str(normalized_resolution)
        )
        if not isinstance(texture_relative_path, str):
            return None
        return self._resolve_texture_image_variant_path(
            record,
            normalized_resolution,
            texture_relative_path,
        )

    def _resolve_texture_image_variant_path(
        self,
        record: GeneratedObjectRecord,
        resolution: int,
        texture_relative_path: str,
        map_texture_relative_paths: object = None,
    ) -> ObjectTextureImageVariant | None:
        """Resolve contained base-color and optional PBR PNG paths safely."""

        try:
            texture_path = self._resolve_generated_asset_path(
                texture_relative_path,
                allowed_suffixes=frozenset({".png"}),
            )
        except (OSError, RuntimeError, ValueError):
            return None
        if not texture_path.is_file():
            return None
        resolved_map_relative_paths: dict[str, str] = {
            ATLAS_MAP_BASE_COLOR: texture_relative_path
        }
        resolved_map_paths: dict[str, Path] = {
            ATLAS_MAP_BASE_COLOR: texture_path
        }
        if isinstance(map_texture_relative_paths, Mapping):
            for map_type in ATLAS_MAP_TYPES:
                raw_map_path = map_texture_relative_paths.get(map_type)
                if not isinstance(raw_map_path, str) or not raw_map_path.strip():
                    continue
                try:
                    map_path = self._resolve_generated_asset_path(
                        raw_map_path,
                        allowed_suffixes=frozenset({".png"}),
                    )
                except (OSError, RuntimeError, ValueError):
                    continue
                if not map_path.is_file():
                    continue
                resolved_map_relative_paths[map_type] = raw_map_path
                resolved_map_paths[map_type] = map_path
        return ObjectTextureImageVariant(
            object_id=record.object_id,
            object_name=record.object_name,
            resolution=resolution,
            texture_asset_relative_path=texture_relative_path,
            texture_asset_path=texture_path,
            map_texture_asset_relative_paths=resolved_map_relative_paths,
            map_texture_asset_paths=resolved_map_paths,
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

    def remove_generated_object_placement(self, object_id: str) -> bool:
        """Remove one Canvas placement without deleting its object or assets."""

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
        request = self._existing_object_placement_request
        if request is not None and request.object_id == normalized_object_id:
            self._finish_existing_object_placement_request()

        record = self._data.generated_objects[record_index]
        replacement = replace(record, placement=None)
        self._data.generated_objects[record_index] = replacement
        self.status_label.setText(f"Removed from Canvas: {record.object_name}")
        self._emit_data_changed()
        self.generated_object_placement_changed.emit(replacement)
        self._sync_controls()
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

    def delete_selected_object_faces(self) -> bool:
        """Start a local face deletion for the current viewer selection."""

        self.result_view.cancel_transient_pointer_interactions()
        self._sync_face_selection_outputs()
        record = self._find_generated_object_record(self._selected_object_id)
        selected_faces = self.result_view.get_selected_face_indices()
        if record is None or not selected_faces:
            return False
        if self._object_has_active_mutation_job(record.object_id):
            self.status_label.setText(
                "Wait for this object's active job to finish before editing it."
            )
            return False
        generated_model = self._generated_model
        if (
            generated_model is None
            or self.result_view.model is not generated_model
        ):
            return False
        request = ObjectFaceDeletionRequest(
            object_id=record.object_id,
            reference_asset_path=record.asset_path,
            source_asset_paths=_get_face_edit_glb_asset_paths(record),
            selected_face_indices=selected_faces,
        )
        return self._start_object_face_deletion(request, record)

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

    def convert_selected_faces_to_glass(self) -> bool:
        """Run a PBR texture job for the authoritative selected faces."""

        self.result_view.cancel_transient_pointer_interactions()
        self._sync_face_selection_outputs()
        selected_faces = self.result_view.get_selected_face_indices()
        if not selected_faces:
            return False
        generated_model = self._generated_model
        existing_glass_faces = (
            None
            if generated_model is None
            else _collect_scene_glass_face_indices(generated_model.scene)
        )
        face_count = self.result_view.face_edit_face_count
        if (
            face_count > 0
            and (
                len(selected_faces) >= face_count
                or (
                    existing_glass_faces is not None
                    and len(existing_glass_faces.union(selected_faces))
                    >= face_count
                )
            )
        ):
            self.status_label.setText(
                "Glass conversion must leave at least one non-glass face."
            )
            return False
        request = self._build_texture_regeneration_request(
            glass_face_indices=selected_faces,
            glass_double_sided=(
                self.glass_double_sided_checkbox.isChecked()
            ),
        )
        if request is None:
            return False
        for checkbox in self.pbr_map_checkboxes.values():
            checkbox.setChecked(True)
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
            OBJECT_OPERATION_DELETE_FACES: "face deletion",
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
        self._discard_object_face_geometry_cache(object_id)
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

    def _start_object_face_deletion(
        self,
        request: ObjectFaceDeletionRequest,
        record: GeneratedObjectRecord,
    ) -> bool:
        """Run one exact face-edit snapshot on an independent worker."""

        if (
            request.object_id != record.object_id
            or self._object_has_active_mutation_job(record.object_id)
        ):
            return False
        self.result_view.cancel_transient_pointer_interactions()
        operation = _ActiveObjectOperation(
            kind=OBJECT_OPERATION_DELETE_FACES,
            target_object_id=record.object_id,
        )
        thread = QThread(self)
        worker = ObjectFaceDeletionWorker(request, self._asset_directory)
        relay = _ObjectJobSignalRelay(operation.operation_id, self)
        managed_job_id = self._create_managed_job(
            operation,
            kind=GENERATION_JOB_KIND_FACE_EDIT,
            requested_name=None,
            default_name=f"Face edit: {record.object_name}",
            stage="Preparing face deletion...",
        )
        runtime = _ObjectJobRuntime(
            operation=operation,
            thread=thread,
            worker=worker,
            relay=relay,
            managed_job_id=managed_job_id,
            record_snapshot=replace(
                record,
                pipeline=copy.deepcopy(record.pipeline),
            ),
            source_asset_revision=_build_face_edit_source_revisions(
                self._asset_directory,
                request.source_asset_paths,
            ),
        )
        self._register_object_job_runtime(runtime)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.succeeded.connect(relay.forward_pair_succeeded)
        worker.failed.connect(relay.forward_failed)
        worker.progress.connect(relay.forward_progress)
        relay.pair_succeeded.connect(
            self._handle_job_face_deletion_succeeded
        )
        relay.failed.connect(self._handle_job_face_deletion_failed)
        relay.progress.connect(self._handle_job_generation_progress)
        relay.thread_finished.connect(
            self._handle_object_job_thread_finished
        )
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(thread.quit)
        thread.finished.connect(relay.forward_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self.status_label.setText("Preparing face deletion...")
        thread.start()
        self._sync_controls()
        return True

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
            OBJECT_GENERATION_AMBIENT_LIGHT_INTENSITY
        )
        self.generated_objects_list = self.object_3d_panel.object_list
        self.face_selection_count_label = (
            self.object_3d_panel.face_selection_count_label
        )
        self.delete_selected_faces_button = (
            self.object_3d_panel.delete_faces_button
        )
        self.convert_faces_to_glass_button = (
            self.object_3d_panel.convert_faces_to_glass_button
        )
        self.glass_double_sided_checkbox = (
            self.object_3d_panel.glass_double_sided_checkbox
        )
        self.delete_generated_object_button = (
            self.object_3d_panel.delete_object_button
        )
        self.model_statistics_label = self.object_3d_panel.statistics_label
        self.object_3d_panel.projection_camera_percentages_changed.connect(
            self._sync_controls
        )
        self.generated_objects_list.currentItemChanged.connect(
            self._handle_generated_object_selection_changed
        )
        self.delete_generated_object_button.clicked.connect(
            self._handle_delete_generated_object_clicked
        )
        self.result_view.face_selection_changed.connect(
            self._handle_face_selection_changed
        )
        self.delete_selected_faces_button.clicked.connect(
            self._handle_delete_selected_faces_clicked
        )
        self.convert_faces_to_glass_button.clicked.connect(
            self.convert_selected_faces_to_glass
        )
        self.result_view.delete_requested.connect(
            self._handle_delete_selected_faces_clicked
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
        self.texture_view.uv_face_selection_requested.connect(
            self._handle_texture_uv_face_selection_requested
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

        self.pbr_map_control = QWidget()
        self.pbr_map_control.setObjectName("object_generation_pbr_map_control")
        pbr_map_layout = QGridLayout(self.pbr_map_control)
        pbr_map_layout.setContentsMargins(0, 0, 0, 0)
        pbr_map_layout.setHorizontalSpacing(6)
        pbr_map_layout.setVerticalSpacing(0)
        self.pbr_map_checkboxes: dict[str, QCheckBox] = {}
        pbr_labels = {
            PBR_MAP_NORMAL: "Normal",
            PBR_MAP_ROUGHNESS: "Roughness",
            PBR_MAP_METALLIC: "Metallic",
        }
        for index, map_type in enumerate(PBR_MAP_TYPES):
            checkbox = QCheckBox(pbr_labels[map_type])
            checkbox.setObjectName(f"pbr_{map_type}_checkbox")
            checkbox.setToolTip(
                f"Request Meshy PBR maps and dynamically apply the "
                f"{pbr_labels[map_type].lower()} map in the 3D preview."
            )
            checkbox.toggled.connect(self._handle_pbr_map_toggled)
            self.pbr_map_checkboxes[map_type] = checkbox
            pbr_map_layout.addWidget(checkbox, index % 3, index // 3)
        buttons_layout.addWidget(self.pbr_map_control)

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
        self.mask_mode_control = QWidget()
        self.mask_mode_control.setObjectName(
            "object_generation_mask_mode_control"
        )
        mask_mode_layout = QVBoxLayout(self.mask_mode_control)
        mask_mode_layout.setContentsMargins(0, 0, 0, 0)
        mask_mode_layout.setSpacing(0)
        mask_mode_layout.addWidget(self.paint_mask_button)
        mask_mode_layout.addWidget(self.erase_mask_button)
        buttons_layout.addWidget(self.mask_mode_control)

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
            "When weighted camera projection is enabled, rebuild its UVs "
            "from the current geometry and current camera allocations. "
            "Meshy tasks consume account credits."
        )
        self.generate_texture_button.clicked.connect(
            self.generate_selected_object_texture
        )
        buttons_layout.addWidget(self.generate_texture_button)
        self.regenerate_texture_button = self.generate_texture_button

        buttons_layout.addSpacing(30)
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
            "texture generation or face deletion."
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
            "Cancel the current object operation and restore the object "
            "state from before it started."
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

    @Slot(object)
    def _handle_face_selection_changed(self, _raw_indices: object) -> None:
        """Push the authoritative viewer selection to every output."""

        self._sync_face_selection_outputs()

    def _sync_face_selection_outputs(self) -> None:
        """Synchronize the UV view, count, and deletion state from one source."""

        selected_indices = self.result_view.get_selected_face_indices()
        selected_count = len(selected_indices)
        self.texture_view.set_selected_uv_face_indices(selected_indices)
        self.face_selection_count_label.setText(
            "No faces selected"
            if selected_count == 0
            else f"{selected_count:,} face"
            + ("" if selected_count == 1 else "s")
            + " selected"
        )
        self._sync_controls()

    @Slot(object)
    def _handle_texture_uv_face_selection_requested(
        self,
        raw_request: object,
    ) -> None:
        """Apply a 2D UV hit through the authoritative 3D face selection."""

        if not isinstance(raw_request, UvFaceSelectionRequest):
            return
        record = self._find_generated_object_record(self._selected_object_id)
        expected_context_token = (
            None
            if record is None
            else _build_object_face_selection_context_token(
                record,
                self._asset_directory,
            )
        )
        if (
            record is None
            or self._object_has_active_mutation_job(record.object_id)
            or not self.texture_view.uv_face_selection_enabled
            or raw_request.context_token != expected_context_token
            or self.texture_view.uv_face_selection_context_token
            != expected_context_token
        ):
            self._sync_face_selection_outputs()
            return
        face_count = self.result_view.face_edit_face_count
        hits = set(raw_request.face_indices)
        if any(index < 0 or index >= face_count for index in hits):
            self._sync_face_selection_outputs()
            return
        self.result_view.cancel_face_selection_interaction()
        if not hits:
            self._sync_face_selection_outputs()
            return
        self.result_view.update_face_selection(
            hits,
            mode=FACE_SELECTION_TOGGLE,
        )

    @Slot()
    def _handle_delete_selected_faces_clicked(self) -> None:
        self.delete_selected_object_faces()

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

    def _get_enabled_pbr_maps(self) -> tuple[str, ...]:
        """Snapshot PBR checkbox state for one immutable async request."""

        return tuple(
            map_type
            for map_type in PBR_MAP_TYPES
            if self.pbr_map_checkboxes[map_type].isChecked()
        )

    @Slot(bool)
    def _handle_pbr_map_toggled(self, _enabled: bool) -> None:
        """Apply PBR contribution changes without reloading the model."""

        self.result_view.set_pbr_maps_enabled(self._get_enabled_pbr_maps())
        self._sync_controls()

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

    @Slot(str, object, object)
    def _handle_job_face_deletion_succeeded(
        self,
        operation_id: str,
        raw_result: object,
        _unused_preview: object,
    ) -> None:
        """Commit one background face edit only if its source is unchanged."""

        if self._should_ignore_operation_result(
            OBJECT_OPERATION_DELETE_FACES,
            operation_id,
        ):
            return
        runtime = self._object_job_runtimes.get(str(operation_id))
        if runtime is None:
            return
        self._set_legacy_active_job_runtime(runtime)
        prepared = raw_result
        snapshot = runtime.record_snapshot
        if (
            not isinstance(prepared, PreparedObjectFaceDeletion)
            or snapshot is None
        ):
            self._handle_job_face_deletion_failed(
                operation_id,
                "The local face-edit result is invalid.",
            )
            return
        record = self._find_generated_object_record(snapshot.object_id)
        if (
            record is None
            or record != snapshot
            or _build_face_edit_source_revisions(
                self._asset_directory,
                prepared.request.source_asset_paths,
            )
            != runtime.source_asset_revision
        ):
            self._handle_job_face_deletion_failed(
                operation_id,
                "The object changed before its face edit could be applied.",
            )
            return

        persisted_paths: list[str] = []
        try:
            if prepared.request.object_id != record.object_id:
                raise ValueError(
                    "The face-edit result targets a different object."
                )
            result = prepared.reference_result
            face_edit_id = uuid.uuid4().hex
            replacement_paths: dict[str, str] = {}
            for source_index, (source_path, edited_glb) in enumerate(
                prepared.edited_glbs,
                start=1,
            ):
                persisted_path = self._persist_meshy_named_asset(
                    f"{record.object_id}.face-edit-{face_edit_id}."
                    f"revision-{source_index}.glb",
                    edited_glb,
                )
                persisted_paths.append(persisted_path)
                replacement_paths[source_path] = persisted_path

            next_pipeline = copy.deepcopy(record.pipeline)
            next_asset_path = _rewrite_face_edit_glb_paths(
                record,
                next_pipeline,
                replacement_paths,
            )
            preview_model = prepared.preview_model
            preview_revision = _build_generation_asset_revision(
                self._asset_directory,
                next_asset_path,
            )
            if preview_revision[1] is None:
                raise OSError("The saved face-edit revision is unavailable.")
            raw_revision = record.pipeline.get(
                FACE_EDIT_REVISION_PIPELINE_KEY,
                0,
            )
            try:
                face_edit_revision = max(int(raw_revision), 0) + 1
            except (TypeError, ValueError):
                face_edit_revision = 1
            next_pipeline.pop(FACE_EDIT_TEXTURE_STALE_PIPELINE_KEY, None)
            next_pipeline.pop(
                FACE_EDIT_ATLAS_PLACEHOLDERS_PIPELINE_KEY,
                None,
            )
            next_pipeline.pop("face_edit_uv_utilization", None)
            for pipeline_key in FACE_EDIT_INVALIDATED_UV_PROVENANCE_PIPELINE_KEYS:
                next_pipeline.pop(pipeline_key, None)
            next_pipeline.update(
                {
                    FACE_EDIT_REVISION_PIPELINE_KEY: face_edit_revision,
                    LOCALLY_AUTHORED_UVS_PIPELINE_KEY: True,
                    "face_edit_texture_preserved": (
                        result.preserved_textured_uvs
                    ),
                    "face_edit_original_face_count": (
                        result.original_face_count
                    ),
                    "face_edit_retained_face_count": (
                        result.retained_face_count
                    ),
                    "face_edit_deleted_face_count": (
                        result.deleted_face_count
                    ),
                }
            )
            replacement = replace(
                record,
                asset_path=next_asset_path,
                pipeline=_push_object_operation_undo_snapshot(
                    record,
                    next_pipeline,
                    operation=OBJECT_OPERATION_DELETE_FACES,
                ),
            )
        except Exception as error:
            self._remove_newly_persisted_assets(persisted_paths)
            self._handle_job_face_deletion_failed(
                operation_id,
                f"The edited object could not be saved: {error}",
            )
            return

        if not self._request_object_packing_change(
            record,
            replacement,
            preview_model,
            preview_asset_revision=preview_revision,
        ):
            self._remove_newly_persisted_assets(persisted_paths)
            self._handle_job_face_deletion_failed(
                operation_id,
                "The Atlas packing change could not be committed.",
            )
            return

        self._record_operation_commit(
            OBJECT_OPERATION_DELETE_FACES,
            record.object_id,
            operation_id,
        )
        cleanup_failed = self._delete_unreferenced_object_assets(record)
        status = (
            f"Deleted {result.deleted_face_count:,} face"
            + ("" if result.deleted_face_count == 1 else "s")
            + f" from {record.object_name}. "
            + (
                "Its existing UVs and texture were retained."
                if result.preserved_textured_uvs
                else "Its remaining material data was retained."
            )
            + (
                " Some superseded files could not be removed."
                if cleanup_failed
                else ""
            )
        )
        self.status_label.setText(status)
        self._emit_data_changed()
        self.generated_object_changed.emit(replacement, preview_model)
        self._complete_managed_job(runtime, status)

    @Slot(str, str)
    def _handle_job_face_deletion_failed(
        self,
        operation_id: str,
        error_message: str,
    ) -> None:
        if self._should_ignore_operation_result(
            OBJECT_OPERATION_DELETE_FACES,
            operation_id,
        ):
            return
        runtime = self._object_job_runtimes.get(str(operation_id))
        if runtime is None:
            return
        self._set_legacy_active_job_runtime(runtime)
        message = f"Face deletion failed: {str(error_message)}"
        self.status_label.setText(message)
        self._fail_managed_job(runtime, message)

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
        pipeline: dict[str, object] = (
            _build_safe_duplicate_removal_pipeline_metadata(result)
        )
        persisted_asset_paths: list[str] = []
        symmetry: ObjectSymmetricDivisionMetadata | None = None
        scan_projection_stats: ScanProjectionStats | None = None
        variant_metadata: dict[str, dict[str, object]] | None = None
        try:
            if isinstance(result, ScanProjectedMeshyGenerationResult):
                scan_projection_stats = result.scan_projection_stats
                if scan_projection_stats is not None:
                    pipeline.update(
                        _build_scan_projection_pipeline_metadata(
                            scan_projection_stats
                        )
                    )
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
                if geometry_only:
                    raise ValueError(
                        "Symmetric division requires a newly generated "
                        "textured model."
                    )
                if not isinstance(texture_variants, ObjectTextureVariants):
                    raise ValueError(
                        "Symmetric division requires a newly generated "
                        "textured model."
                    )
                symmetric_builder_kwargs: dict[str, object] = {}
                if _uses_weighted_camera_projection(generation_request):
                    symmetric_builder_kwargs.update(
                        {
                            "projection_camera_percentages": (
                                generation_request.projection_camera_percentages
                            ),
                            "cancellation_check": None,
                        }
                    )
                division_result = build_automatic_symmetric_object_variants(
                    texture_variants.glb_by_resolution[
                        TEXTURE_RESOLUTION_2048
                    ],
                    generation_request.symmetric_division_orientation,
                    **symmetric_builder_kwargs,
                )
                symmetry = _validate_automatic_symmetric_division_result(
                    division_result,
                    generation_request.symmetric_division_orientation,
                )
                texture_variants = division_result.variants
                scan_projection_stats = (
                    division_result.scan_projection_stats
                )
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
                        **_build_pbr_pipeline_metadata(
                            (
                                ()
                                if generation_request is None
                                else generation_request.enabled_pbr_maps
                            ),
                            texture_variants,
                        ),
                    }
                )
                persisted_asset_paths.extend(
                    _iter_variant_metadata_asset_paths(variant_metadata)
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
                staged_pipeline = _build_staged_generation_pipeline_metadata(
                    result,
                    source_asset_path,
                )
                if symmetry is None:
                    postprocessed_asset_path = (
                        _resolve_staged_postprocessed_asset_path(
                            result,
                            asset_path,
                            variant_metadata,
                        )
                    )
                    if postprocessed_asset_path is None:
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
            if symmetry is not None:
                assert isinstance(
                    texture_variants,
                    SymmetricSquarePairTextureVariants,
                )
                pipeline = _build_automatic_symmetric_generation_pipeline(
                    pipeline,
                    symmetry,
                    variant_metadata,
                    scan_projection_stats=scan_projection_stats,
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
        elif isinstance(result, ScanProjectedMeshyGenerationResult):
            self.status_label.setText(
                f"Generated: {object_name}. Applied weighted camera "
                "scan-projection UVs."
            )
        else:
            self.status_label.setText(f"Generated: {object_name}")
        if not isinstance(result, StagedMeshyGenerationResult):
            duplicate_status = _format_safe_duplicate_removal_status(result)
            if duplicate_status:
                self.status_label.setText(
                    self.status_label.text().rstrip(".")
                    + ". "
                    + duplicate_status
                )
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
        elif isinstance(result, ScanProjectedMeshyGenerationResult):
            status = (
                f"Generated: {object_name}. Applied weighted camera "
                "scan-projection UVs."
            )
        else:
            status = f"Generated: {object_name}"
        if not isinstance(result, StagedMeshyGenerationResult):
            duplicate_status = _format_safe_duplicate_removal_status(result)
            if duplicate_status:
                status = status.rstrip(".") + ". " + duplicate_status
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
                _validate_symmetric_texture_regeneration_uvs(
                    outcome,
                    symmetry,
                )
                canonical_provider_glb = texture_variants.glb_by_resolution[
                    TEXTURE_RESOLUTION_2048
                ]
                texture_variants = _rebuild_symmetric_texture_variants(
                    canonical_provider_glb,
                    symmetry,
                )
            outcome = _with_persisted_canonical_uv_fingerprint(
                outcome,
                record,
                texture_variants,
            )
            variant_metadata = self._persist_object_texture_variants(
                record.object_id,
                texture_variants,
                asset_stem=f"regenerated-{uuid.uuid4().hex}",
            )
            persisted_asset_paths.extend(
                _iter_variant_metadata_asset_paths(variant_metadata)
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
                f"Generated texture: {record.object_name}."
                + _format_texture_face_cleanup_status(outcome)
                + status_suffix,
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
            f"Generated texture: {record.object_name}."
            + _format_texture_face_cleanup_status(outcome)
            + status_suffix,
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
            OBJECT_OPERATION_DELETE_FACES,
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
        if operation.kind == OBJECT_OPERATION_DELETE_FACES:
            outcome = (
                "The previous geometry was restored."
                if had_commit
                else "The existing geometry was kept."
            )
            self.status_label.setText("Face deletion cancelled. " + outcome)
            return
        self.status_label.setText("Operation cancelled.")

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
        if operation.kind == OBJECT_OPERATION_DELETE_FACES:
            self.result_view.cancel_transient_pointer_interactions()
            self._sync_face_selection_outputs()
        else:
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
        projection_camera_percentages = (
            self.object_3d_panel.get_projection_camera_percentages()
        )
        projection_camera_percentages_are_valid = (
            sum(projection_camera_percentages) == 100
        )
        if (
            not geometry_only
            and self._settings.use_uv_raycast_for_object_generation
            and not projection_camera_percentages_are_valid
        ):
            self.status_label.setText(
                "Projection camera texture allocations must total 100%."
            )
            return None
        if not projection_camera_percentages_are_valid:
            projection_camera_percentages = (
                DEFAULT_PROJECTION_CAMERA_PERCENTAGES
            )
        return GenerationRequest(
            frame_index=self._data.current_frame_index,
            selected_object_bgra=selected_crop,
            settings=self._settings,
            geometry_only=geometry_only,
            symmetric_division_enabled=symmetric_division_enabled,
            symmetric_division_orientation=(
                symmetric_division_orientation
            ),
            projection_camera_percentages=(
                projection_camera_percentages
            ),
            enabled_pbr_maps=self._get_enabled_pbr_maps(),
        )

    def _build_texture_regeneration_request(
        self,
        *,
        glass_face_indices: Sequence[int] = (),
        glass_double_sided: bool | None = None,
    ) -> _TextureRegenerationPreflight | None:
        record = self._find_generated_object_record(self._selected_object_id)
        if not self._can_regenerate_object_texture(record):
            self.status_label.setText(
                "Select a generated object and paint a current video reference "
                "before generating its texture."
            )
            return None
        assert record is not None
        normalized_glass_faces = tuple(
            sorted(set(int(index) for index in glass_face_indices))
        )
        if any(index < 0 for index in normalized_glass_faces):
            self.status_label.setText("Selected glass faces are invalid.")
            return None
        selected_crop = self.video_view.build_selected_object_crop()
        if selected_crop.size == 0:
            self.status_label.setText("The selected texture reference is empty.")
            return None
        projection_camera_percentages = (
            self.object_3d_panel.get_projection_camera_percentages()
        )
        projection_camera_percentages_are_valid = (
            sum(projection_camera_percentages) == 100
        )
        if (
            self._settings.use_uv_raycast_for_object_generation
            and not projection_camera_percentages_are_valid
        ):
            self.status_label.setText(
                "Projection camera texture allocations must total 100%."
            )
            return None
        if not projection_camera_percentages_are_valid:
            projection_camera_percentages = (
                DEFAULT_PROJECTION_CAMERA_PERCENTAGES
            )
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
            preserve_existing_glass = isinstance(
                record.pipeline.get(GLASS_CONVERSION_PIPELINE_KEY),
                Mapping,
            )
            has_complete_texture_uvs = (
                self._selected_object_has_complete_texture_uvs(record)
            )
            enable_original_uv = bool(
                preserve_symmetric_uvs
                or record.pipeline.get(LOCALLY_AUTHORED_UVS_PIPELINE_KEY)
                or preserve_existing_glass
                or (normalized_glass_faces and has_complete_texture_uvs)
            )
            enabled_pbr_maps = (
                PBR_MAP_TYPES
                if normalized_glass_faces
                else self._get_enabled_pbr_maps()
            )
            request = _TextureRegenerationPreflight(
                object_id=record.object_id,
                reference_frame_index=self._data.current_frame_index,
                reference_image_bgra=selected_crop,
                source_asset_path=source_asset_path,
                source_asset_revision=source_asset_revision,
                settings=self._settings,
                enable_original_uv=enable_original_uv,
                preserve_symmetric_uvs=preserve_symmetric_uvs,
                projection_camera_percentages=(
                    projection_camera_percentages
                ),
                enabled_pbr_maps=enabled_pbr_maps,
                glass_face_indices=normalized_glass_faces,
                glass_double_sided=glass_double_sided,
                preserve_existing_glass=preserve_existing_glass,
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
        self.clear_mask_button.setEnabled(
            has_video and has_mask and not has_untracked_legacy_job
        )
        required_key_is_available = bool(self._settings.meshy_api_key)
        projection_camera_percentages_are_valid = (
            not self._settings.use_uv_raycast_for_object_generation
            or self.object_3d_panel.projection_camera_percentages_are_valid()
        )
        self.meshy_target_polycount_control.setVisible(True)
        self.meshy_target_polycount_spinbox.setEnabled(
            not has_untracked_legacy_job
        )
        self.textures_checkbox.setEnabled(not has_untracked_legacy_job)
        self.pbr_map_control.setEnabled(not has_untracked_legacy_job)
        self.wireframe_checkbox.setEnabled(not has_untracked_legacy_job)
        self.generated_objects_list.setEnabled(not has_untracked_legacy_job)
        self.texture_view.setEnabled(not has_untracked_legacy_job)
        self.delete_generated_object_button.setEnabled(
            selected_record is not None
            and not selected_object_is_busy
        )
        selected_face_count = len(
            self.result_view.get_selected_face_indices()
        )
        face_selection_is_available = (
            selected_record is not None
            and self.result_view.face_edit_face_count > 0
            and not selected_object_is_busy
        )
        self.delete_selected_faces_button.setEnabled(
            selected_record is not None
            and selected_face_count > 0
            and not selected_object_is_busy
        )
        self.convert_faces_to_glass_button.setEnabled(
            selected_record is not None
            and selected_face_count > 0
            and not selected_object_is_busy
            and required_key_is_available
            and self._video_source is not None
            and self.video_view.get_frame_bgr() is not None
            and self.video_view.has_selection()
            and projection_camera_percentages_are_valid
        )
        self.result_view.set_face_editing_enabled(
            face_selection_is_available
        )
        self.texture_view.set_uv_face_selection_enabled(
            face_selection_is_available
        )
        self.regenerate_texture_button.setEnabled(
            self._can_regenerate_object_texture(selected_record)
            and projection_camera_percentages_are_valid
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
            and projection_camera_percentages_are_valid
            and not has_untracked_legacy_job
        )
        self.generate_geometry_button.setEnabled(
            has_video
            and has_mask
            and required_key_is_available
            and not self.symmetric_division_checkbox.isChecked()
            and not has_untracked_legacy_job
        )
        self.video_view.set_interaction_enabled(
            has_video and not has_untracked_legacy_job
        )

    def _selected_object_has_complete_texture_uvs(
        self,
        record: GeneratedObjectRecord | None,
    ) -> bool:
        """Return whether every selected-model face has a stable finite UV."""

        model = self._generated_model
        if (
            record is None
            or model is None
            or self._selected_object_id != record.object_id
            or self.result_view.model is not model
        ):
            return False
        try:
            geometry = self._load_object_face_geometry(record, model)
        except Exception:
            return False
        uv_face_indices = np.asarray(
            geometry.uv_face_indices,
            dtype=np.int64,
        )
        return bool(
            geometry.face_count > 0
            and len(uv_face_indices) == geometry.face_count
            and np.array_equal(
                np.sort(uv_face_indices),
                np.arange(geometry.face_count, dtype=np.int64),
            )
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
        self.glass_double_sided_checkbox.setChecked(
            _resolve_glass_double_sided(record.pipeline, None)
        )
        next_snapshot = _build_generated_object_display_snapshot(
            record,
            self._asset_directory,
        )
        selection_context_token = (
            _build_object_face_selection_context_token(
                record,
                self._asset_directory,
            )
        )
        if (
            next_snapshot == self._displayed_object_snapshot
            and self._generated_model is not None
            and self.result_view.model is self._generated_model
            and self.result_view.face_edit_face_count > 0
            and self.texture_view.uv_face_selection_context_token
            == selection_context_token
        ):
            self._refresh_object_texture_atlases(record.object_id)
            self._sync_face_selection_outputs()
            return
        self._displayed_object_snapshot = None
        self._generated_model = None
        self.result_view.cancel_transient_pointer_interactions()
        self.result_view.clear_model()
        self._sync_model_statistics(None)
        self.texture_view.clear()
        try:
            generated_model = self._load_generated_object_model(record)
        except Exception as error:
            self.status_label.setText(
                f"Saved generated object could not be rebuilt: {error}"
            )
            self._refresh_object_texture_atlases(record.object_id)
            self._sync_face_selection_outputs()
            return
        next_snapshot = _build_generated_object_display_snapshot(
            record,
            self._asset_directory,
        )
        selection_context_token = (
            _build_object_face_selection_context_token(
                record,
                self._asset_directory,
            )
        )
        self._generated_model = generated_model
        self.result_view.set_model(generated_model)
        try:
            face_geometry = self._load_object_face_geometry(
                record,
                generated_model,
            )
            self.result_view.set_face_edit_geometry(
                face_geometry.vertices,
                face_geometry.faces,
            )
            self.texture_view.set_uv_overlay_triangles(
                face_geometry.uv_triangles
            )
            self.texture_view.set_uv_face_selection_geometry(
                face_geometry.uv_triangles,
                face_geometry.uv_face_indices,
                context_token=selection_context_token,
            )
        except Exception as error:
            self.status_label.setText(
                f"This object's faces cannot be edited: {error}"
            )
        symmetry = _get_object_symmetric_division_metadata(record)
        self.result_view.set_symmetric_division_preview(
            None if symmetry is None else symmetry.orientation,
            None if symmetry is None else symmetry.plane_coordinate,
        )
        self._sync_model_statistics(generated_model)
        self._refresh_object_texture_atlases(
            record.object_id,
        )
        self._displayed_object_snapshot = next_snapshot
        self._sync_face_selection_outputs()

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
        self.result_view.cancel_transient_pointer_interactions()
        if (
            self._displayed_object_snapshot is None
            and self._generated_model is None
            and self.result_view.model is None
            and not self.texture_view.entries
        ):
            self._sync_face_selection_outputs()
            return
        self._displayed_object_snapshot = None
        self._generated_model = None
        self.result_view.clear_model()
        self._sync_model_statistics(None)
        self.texture_view.clear()
        self._sync_face_selection_outputs()

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
            or record.pipeline.get(FACE_EDIT_TEXTURE_STALE_PIPELINE_KEY) is True
            or replacement.pipeline.get(FACE_EDIT_TEXTURE_STALE_PIPELINE_KEY)
            is True
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

    def _load_object_face_geometry(
        self,
        record: GeneratedObjectRecord,
        model: GeneratedModel,
    ) -> ObjectFaceGeometry:
        """Reuse immutable face/UV geometry for one exact asset revision."""

        asset_revision = _build_generation_asset_revision(
            self._asset_directory,
            record.asset_path,
        )
        cache_key = (record.object_id, *asset_revision)
        cached = self._object_face_geometry_cache.pop(cache_key, None)
        if cached is not None:
            self._object_face_geometry_cache[cache_key] = cached
            return cached
        geometry = load_object_face_geometry_from_scene(model.scene)
        for array in (
            geometry.vertices,
            geometry.faces,
            geometry.uv_triangles,
            geometry.uv_face_indices,
        ):
            array.setflags(write=False)
        self._object_face_geometry_cache[cache_key] = geometry
        while (
            len(self._object_face_geometry_cache)
            > OBJECT_FACE_GEOMETRY_CACHE_MAX_ENTRIES
        ):
            oldest_key = next(iter(self._object_face_geometry_cache))
            self._object_face_geometry_cache.pop(oldest_key, None)
        return geometry

    def _discard_object_face_geometry_cache(self, object_id: str) -> None:
        """Remove every cached asset revision owned by one object."""

        normalized_id = str(object_id)
        stale_keys = tuple(
            cache_key
            for cache_key in self._object_face_geometry_cache
            if cache_key and cache_key[0] == normalized_id
        )
        for cache_key in stale_keys:
            self._object_face_geometry_cache.pop(cache_key, None)

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
    ) -> dict[str, dict[str, object]]:
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
def _staged_result_used_face_removal(
    result: StagedMeshyGenerationResult,
) -> bool:
    return bool(
        result.unused_face_removal_applied
        or result.final_face_removal_applied
        or result.original_face_count
        or result.retained_face_count
        or result.removed_face_count
        or result.protected_face_count
        or result.final_original_face_count
        or result.final_retained_face_count
        or result.final_removed_face_count
    )


def _staged_result_used_visibility_uv(
    result: StagedMeshyGenerationResult,
) -> bool:
    return result.visibility_uv_stats is not None


def _staged_result_used_scan_projection(
    result: StagedMeshyGenerationResult,
) -> bool:
    return result.scan_projection_stats is not None


def _staged_generation_mode(result: StagedMeshyGenerationResult) -> str:
    stages: list[str] = []
    if result.safe_duplicate_removed_face_count > 0:
        stages.append("safe_duplicate_face_removal")
    if _staged_result_used_face_removal(result):
        stages.append("unused_face_removal")
    if _staged_result_used_visibility_uv(result):
        stages.append("visibility_uv_raycast")
    if _staged_result_used_scan_projection(result):
        stages.append("weighted_camera_scan_projection")
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
    duplicate_status = _format_safe_duplicate_removal_status(result)
    if duplicate_status:
        messages.append(duplicate_status)
    if _staged_result_used_face_removal(result):
        messages.append(
            f"Removed {result.removed_face_count} of "
            f"{result.original_face_count} faces."
        )
    if result.final_face_removal_applied:
        messages.append(
            f"Final texture cleanup removed {result.final_removed_face_count} "
            f"of {result.final_original_face_count} faces."
        )
    if result.retexture_topology_changed:
        messages.append("Retexture topology changed and was revalidated.")
    if _staged_result_used_visibility_uv(result):
        messages.append("Applied visibility-optimized UVs.")
    if _staged_result_used_scan_projection(result):
        messages.append("Applied weighted camera scan-projection UVs.")
    return " ".join(messages)


def _format_safe_duplicate_removal_status(result: object) -> str:
    """Describe a conservative cleanup only when it changed the model."""

    removed_face_count = int(
        getattr(result, "safe_duplicate_removed_face_count", 0)
    )
    if removed_face_count <= 0:
        return ""
    if isinstance(result, StagedMeshyGenerationResult):
        geometry_count = result.geometry_safe_duplicate_removed_face_count
        retextured_count = (
            result.retextured_safe_duplicate_removed_face_count
        )
        if geometry_count > 0 and retextured_count > 0:
            return (
                f"Removed {removed_face_count} coincident duplicate face "
                "occurrences across preprocessing passes "
                f"({geometry_count} from geometry and {retextured_count} "
                "from the retextured model)."
            )
        if geometry_count > 0:
            return (
                f"Removed {geometry_count} coincident duplicate "
                + ("face" if geometry_count == 1 else "faces")
                + " from generated geometry."
            )
        if retextured_count > 0:
            return (
                f"Removed {retextured_count} coincident duplicate "
                + ("face" if retextured_count == 1 else "faces")
                + " from the retextured model."
            )
    return (
        f"Removed {removed_face_count} coincident duplicate "
        + ("face." if removed_face_count == 1 else "faces.")
    )


# ### Texture-regeneration status helpers ###
def _format_texture_face_cleanup_status(
    outcome: TextureRegenerationOutcome,
) -> str:
    """Describe optional final geometry verification without hiding counts."""

    status = ""
    duplicate_status = _format_safe_duplicate_removal_status(outcome)
    if duplicate_status:
        status = f" {duplicate_status}"
    if outcome.final_face_removal_applied:
        status += (
            f" Final cleanup removed {outcome.final_removed_face_count} of "
            f"{outcome.final_original_face_count} faces."
        )
        if outcome.retexture_topology_changed:
            status += " Retexture topology changed and was revalidated."
    if outcome.scan_projection_stats is not None:
        status += " Rebuilt UVs from the current weighted cameras."
    return status


def _build_scan_projection_pipeline_metadata(
    stats: ScanProjectionStats,
) -> dict[str, object]:
    """Build stable UV-authority metadata for one scan-projected object."""

    if not isinstance(stats, ScanProjectionStats):
        raise TypeError("Scan-projection statistics are invalid.")
    return {
        LOCALLY_AUTHORED_UVS_PIPELINE_KEY: True,
        SCAN_PROJECTION_PIPELINE_KEY: stats.to_pipeline_dict(),
    }


def _build_safe_duplicate_removal_pipeline_metadata(
    result: object,
) -> dict[str, object]:
    """Persist only a cleanup that removed proven duplicate triangles."""

    removed_face_count = int(
        getattr(result, "safe_duplicate_removed_face_count", 0)
    )
    duplicate_group_count = int(
        getattr(result, "safe_duplicate_group_count", 0)
    )
    if removed_face_count <= 0:
        return {}
    cleanup_metadata: dict[str, object] = {
        "removed_face_count": removed_face_count,
        "duplicate_group_count": duplicate_group_count,
    }
    if isinstance(result, StagedMeshyGenerationResult):
        cleanup_metadata["count_scope"] = "preprocessing_pass_occurrences"
        passes: dict[str, object] = {}
        if result.geometry_safe_duplicate_removed_face_count > 0:
            passes["generated_geometry"] = {
                "removed_face_count": (
                    result.geometry_safe_duplicate_removed_face_count
                ),
                "duplicate_group_count": (
                    result.geometry_safe_duplicate_group_count
                ),
            }
        if result.retextured_safe_duplicate_removed_face_count > 0:
            passes["retextured_model"] = {
                "removed_face_count": (
                    result.retextured_safe_duplicate_removed_face_count
                ),
                "duplicate_group_count": (
                    result.retextured_safe_duplicate_group_count
                ),
            }
        cleanup_metadata["passes"] = passes
    return {
        SAFE_DUPLICATE_REMOVAL_PIPELINE_KEY: cleanup_metadata
    }


def _build_staged_generation_pipeline_metadata(
    result: StagedMeshyGenerationResult,
    source_asset_path: str,
) -> dict[str, object]:
    """Build one shared persisted description of local generation stages."""

    pipeline = _build_safe_duplicate_removal_pipeline_metadata(result)
    pipeline.update(
        {
            "mode": _staged_generation_mode(result),
            "geometry_task_id": result.geometry_task_id,
            "source_asset_path": source_asset_path,
            "unused_face_removal_applied": (
                _staged_result_used_face_removal(result)
            ),
            "geometry_only": result.geometry_only,
        }
    )
    if _staged_result_used_face_removal(result):
        pipeline.update(
            {
                "original_face_count": result.original_face_count,
                "retained_face_count": result.retained_face_count,
                "removed_face_count": result.removed_face_count,
                "protected_face_count": result.protected_face_count,
                "minimum_face_visibility_percentage": (
                    result.minimum_face_visibility_percentage
                ),
                "visibility_removed_face_count": (
                    result.visibility_removed_face_count
                ),
                "stacked_face_removed_count": (
                    result.stacked_face_removed_count
                ),
                "final_face_removal_applied": (
                    result.final_face_removal_applied
                ),
                "final_original_face_count": result.final_original_face_count,
                "final_retained_face_count": result.final_retained_face_count,
                "final_removed_face_count": result.final_removed_face_count,
                "final_visibility_removed_face_count": (
                    result.final_visibility_removed_face_count
                ),
                "final_stacked_face_removed_count": (
                    result.final_stacked_face_removed_count
                ),
                "retexture_topology_changed": (
                    result.retexture_topology_changed
                ),
            }
        )
    stats = result.visibility_uv_stats
    if stats is not None:
        pipeline.update(
            {
                LOCALLY_AUTHORED_UVS_PIPELINE_KEY: True,
                VISIBILITY_UV_UNWRAP_PIPELINE_KEY: {
                    "version": VISIBILITY_UV_UNWRAP_VERSION,
                    "face_count": stats.face_count,
                    "instance_face_count": stats.instance_face_count,
                    "chart_count": stats.chart_count,
                    "exterior_face_count": stats.exterior_face_count,
                    "hidden_face_count": stats.hidden_face_count,
                    "camera_count": stats.camera_count,
                    "ray_sample_count": stats.ray_sample_count,
                    "texture_resolution": stats.texture_resolution,
                    "gutter_pixels": stats.gutter_pixels,
                    "effective_gutter_pixels": (
                        stats.effective_gutter_pixels
                    ),
                    "effective_horizontal_gutter_pixels": (
                        stats.effective_horizontal_gutter_pixels
                    ),
                    "effective_vertical_gutter_pixels": (
                        stats.effective_vertical_gutter_pixels
                    ),
                    "atlas_width": stats.atlas_width,
                    "atlas_height": stats.atlas_height,
                    "atlas_utilization": stats.atlas_utilization,
                    "target_domain": stats.target_domain,
                    "packing_strategy": stats.packing_strategy,
                    "requested_exterior_uv_share": (
                        stats.requested_exterior_uv_share
                    ),
                    "achieved_exterior_uv_share": (
                        stats.achieved_exterior_uv_share
                    ),
                    "uv_triangle_occupancy": stats.uv_triangle_occupancy,
                },
            }
        )
    scan_stats = result.scan_projection_stats
    if scan_stats is not None:
        pipeline.update(
            _build_scan_projection_pipeline_metadata(scan_stats)
        )
    return pipeline


def _resolve_staged_postprocessed_asset_path(
    result: StagedMeshyGenerationResult,
    asset_path: str,
    variant_metadata: Mapping[str, Mapping[str, str]] | None,
) -> str | None:
    """Reuse an authoritative saved UV GLB instead of writing a duplicate."""

    if not (
        _staged_result_used_visibility_uv(result)
        or _staged_result_used_scan_projection(result)
    ):
        return None
    if result.geometry_only:
        return asset_path
    canonical_variant = (
        None
        if variant_metadata is None
        else variant_metadata.get(str(TEXTURE_RESOLUTION_2048))
    )
    if canonical_variant is None:
        raise ValueError(
            "Locally authored UV generation has no canonical texture variant."
        )
    canonical_path = canonical_variant.get(TEXTURE_VARIANT_GLB_PATH_KEY)
    if not isinstance(canonical_path, str) or not canonical_path:
        raise ValueError(
            "Locally authored UV generation has no canonical GLB path."
        )
    return canonical_path


# ### Symmetric-division helpers ###
def _validate_symmetric_texture_regeneration_uvs(
    outcome: TextureRegenerationOutcome,
    symmetry: ObjectSymmetricDivisionMetadata | None = None,
) -> None:
    """Require proof that symmetric geometry survived texture regeneration."""

    request = outcome.request
    submitted_fingerprint = request.submitted_uv_fingerprint
    preserved_fingerprint = outcome.preserved_uv_fingerprint
    final_fingerprint = outcome.final_uv_fingerprint
    if preserved_fingerprint is None and outcome.scan_projection_stats is None:
        preserved_fingerprint = final_fingerprint
    if (
        not request.enable_original_uv
        or not request.preserve_symmetric_uvs
        or submitted_fingerprint is None
    ):
        raise UvIntegrityError(
            "Symmetric texture generation must preserve the existing packed "
            "UV layout. The existing texture was kept."
        )
    if preserved_fingerprint is None:
        raise UvIntegrityError(
            "The symmetric texture result has no verified preserved UV "
            "fingerprint. The existing texture was kept."
        )
    _validate_preserved_uv_retexture_integrity(
        submitted_fingerprint,
        preserved_fingerprint,
    )
    if final_fingerprint is None:
        raise UvIntegrityError(
            "The symmetric texture result has no final UV fingerprint. "
            "The existing texture was kept."
        )
    if outcome.scan_projection_stats is None:
        _validate_preserved_uv_retexture_integrity(
            submitted_fingerprint,
            final_fingerprint,
        )
    else:
        expected_target = (
            SCAN_PROJECTION_TARGET_TOP_LEFT_QUARTER
            if (
                symmetry is not None
                and symmetry.version == SYMMETRIC_QUARTER_METADATA_VERSION
            )
            else SCAN_PROJECTION_TARGET_LEFT_HALF
        )
        if outcome.scan_projection_stats.target_domain == expected_target:
            return
        raise UvIntegrityError(
            "Weighted camera UV rebuilding used an invalid symmetric texture "
            "region. The existing texture was kept."
        )


def _validate_preserved_uv_retexture_integrity(
    submitted: UvFingerprint,
    returned: UvFingerprint,
) -> None:
    """Reject topology or authored-UV changes before replacing the object."""

    if submitted.version != returned.version:
        raise UvIntegrityError(
            "The preserved UV integrity versions do not match. The existing "
            "texture was kept; retry texture generation."
        )
    if submitted.face_count != returned.face_count:
        raise UvIntegrityError(
            "HouseMaker could not preserve the edited object's face count "
            "while applying its new texture. The existing texture was kept."
        )
    if submitted.sha256 != returned.sha256:
        raise UvIntegrityError(
            "HouseMaker could not preserve the edited object's packed UV "
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


def _with_persisted_canonical_uv_fingerprint(
    outcome: TextureRegenerationOutcome,
    record: GeneratedObjectRecord,
    variants: PersistableObjectTextureVariants,
) -> TextureRegenerationOutcome:
    """Fingerprint the exact canonical GLB retained for the next operation."""

    if (
        outcome.final_uv_fingerprint is None
        or outcome.scan_projection_stats is None
    ):
        return outcome
    canonical_resolution = _canonical_texture_resolution(record)
    try:
        canonical_glb = variants.glb_by_resolution[canonical_resolution]
    except KeyError as error:
        raise ValueError(
            "The regenerated texture variants have no canonical UV source."
        ) from error
    return replace(
        outcome,
        final_uv_fingerprint=build_uv_fingerprint(canonical_glb),
    )


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
    variant_metadata: dict[str, dict[str, object]],
    *,
    scan_projection_stats: ScanProjectionStats | None = None,
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
    if scan_projection_stats is not None:
        pipeline.update(
            _build_scan_projection_pipeline_metadata(
                scan_projection_stats
            )
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
            OBJECT_OPERATION_DELETE_FACES,
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
        OBJECT_OPERATION_DELETE_FACES,
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
    variant_metadata: dict[str, dict[str, object]],
) -> dict[str, object]:
    request = outcome.request
    result = outcome.result
    pipeline: dict[str, object] = dict(record.pipeline)
    previous_glass_metadata = pipeline.get(GLASS_CONVERSION_PIPELINE_KEY)
    resolved_glass_double_sided = _resolve_glass_double_sided(
        pipeline,
        request.glass_double_sided,
    )
    pipeline.pop(TEXTURE_INPAINT_STROKES_PIPELINE_KEY, None)
    pipeline.pop(FACE_EDIT_TEXTURE_STALE_PIPELINE_KEY, None)
    pipeline.pop(FACE_EDIT_ATLAS_PLACEHOLDERS_PIPELINE_KEY, None)
    for key in LAST_TEXTURE_FACE_REMOVAL_DETAIL_PIPELINE_KEYS:
        pipeline.pop(key, None)
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
        "preserve_existing_glass": request.preserve_existing_glass,
        "enabled_pbr_maps": list(request.enabled_pbr_maps),
    }
    if request.glass_face_indices:
        history_entry["glass_source_face_count"] = len(
            request.glass_face_indices
        )
    if request.glass_face_indices or request.preserve_existing_glass:
        history_entry["glass_double_sided"] = (
            resolved_glass_double_sided
        )
    scan_projection_stats = outcome.scan_projection_stats
    if scan_projection_stats is not None:
        history_entry["projection_camera_percentages"] = dict(
            scan_projection_stats.to_pipeline_dict()["camera_percentages"]
        )
    if request.submitted_uv_fingerprint is not None:
        history_entry["submitted_uv_face_count"] = (
            request.submitted_uv_fingerprint.face_count
        )
    if outcome.final_uv_fingerprint is not None:
        history_entry["final_uv_face_count"] = (
            outcome.final_uv_fingerprint.face_count
        )
    if outcome.final_face_removal_applied:
        history_entry.update(
            {
                "minimum_face_visibility_percentage": (
                    outcome.minimum_face_visibility_percentage
                ),
                "face_removal_original_face_count": (
                    outcome.final_original_face_count
                ),
                "face_removal_retained_face_count": (
                    outcome.final_retained_face_count
                ),
                "face_removal_removed_face_count": (
                    outcome.final_removed_face_count
                ),
                "face_removal_visibility_removed_face_count": (
                    outcome.final_visibility_removed_face_count
                ),
                "face_removal_stacked_face_removed_count": (
                    outcome.final_stacked_face_removed_count
                ),
                "retexture_topology_changed": (
                    outcome.retexture_topology_changed
                ),
            }
        )
    if outcome.safe_duplicate_removed_face_count > 0:
        history_entry[SAFE_DUPLICATE_REMOVAL_PIPELINE_KEY] = {
            "removed_face_count": (
                outcome.safe_duplicate_removed_face_count
            ),
            "duplicate_group_count": outcome.safe_duplicate_group_count,
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
            "last_texture_face_removal_applied": (
                outcome.final_face_removal_applied
            ),
            "last_texture_safe_duplicate_removed_face_count": (
                outcome.safe_duplicate_removed_face_count
            ),
            "last_texture_safe_duplicate_group_count": (
                outcome.safe_duplicate_group_count
            ),
            "texture_regeneration_history": history[-25:],
            PBR_MAPS_ENABLED_PIPELINE_KEY: list(request.enabled_pbr_maps),
            PBR_MAPS_AVAILABLE_PIPELINE_KEY: list(
                _available_metadata_pbr_maps(variant_metadata)
            ),
        }
    )
    if scan_projection_stats is not None:
        pipeline.pop(VISIBILITY_UV_UNWRAP_PIPELINE_KEY, None)
        pipeline.pop(GLASS_CONVERSION_PIPELINE_KEY, None)
        pipeline.update(
            _build_scan_projection_pipeline_metadata(scan_projection_stats)
        )
        if scan_projection_stats.glass_face_count:
            source_face_count = len(request.glass_face_indices)
            if (
                source_face_count == 0
                and isinstance(previous_glass_metadata, Mapping)
            ):
                raw_source_face_count = previous_glass_metadata.get(
                    "source_face_count",
                    0,
                )
                if (
                    isinstance(raw_source_face_count, int)
                    and not isinstance(raw_source_face_count, bool)
                ):
                    source_face_count = max(raw_source_face_count, 0)
            prefab_glass_material = build_housemaker_glass_material(
                resolved_glass_double_sided
            )
            runtime_material_key = get_housemaker_glass_runtime_key(
                prefab_glass_material
            )
            if runtime_material_key is None:
                raise RuntimeError(
                    "The prefab glass runtime key could not be resolved."
                )
            pipeline[GLASS_CONVERSION_PIPELINE_KEY] = {
                "material_name": HOUSEMAKER_GLASS_MATERIAL_NAME,
                "material_source": GLASS_MATERIAL_SOURCE_PREFAB,
                "material_profile": HOUSEMAKER_GLASS_MATERIAL_PROFILE,
                "atlas_independent": True,
                "runtime_material_key": runtime_material_key,
                "double_sided": resolved_glass_double_sided,
                "source_face_count": source_face_count,
                "output_face_count": scan_projection_stats.glass_face_count,
                "atlas_pixel_count": scan_projection_stats.glass_pixel_count,
            }
    if outcome.final_face_removal_applied:
        pipeline.update(
            {
                "last_texture_minimum_face_visibility_percentage": (
                    outcome.minimum_face_visibility_percentage
                ),
                "last_texture_face_removal_original_face_count": (
                    outcome.final_original_face_count
                ),
                "last_texture_face_removal_retained_face_count": (
                    outcome.final_retained_face_count
                ),
                "last_texture_face_removal_removed_face_count": (
                    outcome.final_removed_face_count
                ),
                "last_texture_face_removal_visibility_removed_face_count": (
                    outcome.final_visibility_removed_face_count
                ),
                "last_texture_face_removal_stacked_face_removed_count": (
                    outcome.final_stacked_face_removed_count
                ),
                "last_texture_retexture_topology_changed": (
                    outcome.retexture_topology_changed
                ),
            }
        )
    if (
        scan_projection_stats is not None
        or record.pipeline.get(LOCALLY_AUTHORED_UVS_PIPELINE_KEY) is True
    ):
        canonical_resolution = _canonical_texture_resolution(record)
        canonical_variant = variant_metadata.get(str(canonical_resolution))
        if canonical_variant is None:
            raise ValueError(
                "The regenerated authored-UV source variant is unavailable."
            )
        pipeline["postprocessed_asset_path"] = canonical_variant[
            TEXTURE_VARIANT_GLB_PATH_KEY
        ]
    final_fingerprint = outcome.final_uv_fingerprint
    if final_fingerprint is not None:
        pipeline.update(
            {
                "texture_regeneration_uv_fingerprint_version": (
                    UV_FINGERPRINT_VERSION
                ),
                "texture_regeneration_final_uv_fingerprint": (
                    final_fingerprint.sha256
                ),
                "texture_regeneration_uv_face_count": (
                    final_fingerprint.face_count
                ),
                "texture_regeneration_final_uv_face_count": (
                    final_fingerprint.face_count
                ),
            }
        )
    if request.submitted_uv_fingerprint is not None:
        if final_fingerprint is None:
            raise UvIntegrityError(
                "The regenerated texture UV fingerprint is unavailable."
            )
        pipeline.update(
            {
                "retexture_enable_original_uv": True,
                "texture_regeneration_submitted_uv_fingerprint": (
                    request.submitted_uv_fingerprint.sha256
                ),
                "texture_regeneration_submitted_uv_face_count": (
                    request.submitted_uv_fingerprint.face_count
                ),
            }
        )
    else:
        pipeline["retexture_enable_original_uv"] = False
        pipeline.pop("texture_regeneration_submitted_uv_fingerprint", None)
        pipeline.pop("texture_regeneration_submitted_uv_face_count", None)
    return pipeline


# ### UV preview helpers ###
def _collect_model_uv_triangles(
    model: GeneratedModel,
) -> tuple[UvTriangle, ...]:
    """Return UV triangles in the canonical editable-face traversal order."""

    geometry = load_object_face_geometry(model.glb_bytes)
    return tuple(
        (
            (float(triangle[0, 0]), float(triangle[0, 1])),
            (float(triangle[1, 0]), float(triangle[1, 1])),
            (float(triangle[2, 0]), float(triangle[2, 1])),
        )
        for triangle in geometry.uv_triangles
    )


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
        if isinstance(raw_variants, dict):
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
                raw_map_paths = raw_variant.get(
                    TEXTURE_VARIANT_MAP_PNG_PATHS_KEY
                )
                if isinstance(raw_map_paths, Mapping):
                    for raw_path in raw_map_paths.values():
                        if isinstance(raw_path, str) and raw_path.strip():
                            raw_paths.append(raw_path)
        raw_placeholders = pipeline.get(
            FACE_EDIT_ATLAS_PLACEHOLDERS_PIPELINE_KEY
        )
        if isinstance(raw_placeholders, Mapping):
            for raw_path in raw_placeholders.values():
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
) -> dict[str, dict[str, object]]:
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
    metadata: dict[str, dict[str, object]] = {}
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
            map_paths: dict[str, str] = {
                ATLAS_MAP_BASE_COLOR: png_path
            }
            raw_map_pngs = getattr(
                variants,
                "map_png_by_resolution",
                None,
            )
            if isinstance(raw_map_pngs, Mapping):
                for map_type in PBR_MAP_TYPES:
                    raw_resolution_map = raw_map_pngs.get(map_type)
                    if not isinstance(raw_resolution_map, Mapping):
                        continue
                    map_png = raw_resolution_map.get(resolution)
                    if not isinstance(map_png, bytes | bytearray | memoryview):
                        continue
                    map_path = persist(
                        f"{normalized_asset_stem}.texture-{resolution}."
                        f"{map_type}.png",
                        bytes(map_png),
                    )
                    created_paths.append(map_path)
                    map_paths[map_type] = map_path
            metadata[str(resolution)] = {
                TEXTURE_VARIANT_GLB_PATH_KEY: glb_path,
                TEXTURE_VARIANT_PNG_PATH_KEY: png_path,
                TEXTURE_VARIANT_MAP_PNG_PATHS_KEY: map_paths,
            }
        return metadata
    except Exception:
        _discard_generated_asset_paths(asset_directory, created_paths)
        raise


def _iter_variant_metadata_asset_paths(
    variant_metadata: Mapping[str, Mapping[str, object]],
) -> tuple[str, ...]:
    """Flatten GLB, base-color, and PBR paths without treating maps as paths."""

    paths: list[str] = []
    for raw_variant in variant_metadata.values():
        for path_key in (
            TEXTURE_VARIANT_GLB_PATH_KEY,
            TEXTURE_VARIANT_PNG_PATH_KEY,
        ):
            raw_path = raw_variant.get(path_key)
            if isinstance(raw_path, str) and raw_path.strip():
                paths.append(raw_path)
        raw_map_paths = raw_variant.get(TEXTURE_VARIANT_MAP_PNG_PATHS_KEY)
        if isinstance(raw_map_paths, Mapping):
            for raw_path in raw_map_paths.values():
                if isinstance(raw_path, str) and raw_path.strip():
                    paths.append(raw_path)
    return tuple(dict.fromkeys(paths))


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
        projection_camera_percentages=(
            raw_request.projection_camera_percentages
        ),
        enabled_pbr_maps=raw_request.enabled_pbr_maps,
        glass_face_indices=raw_request.glass_face_indices,
        glass_double_sided=raw_request.glass_double_sided,
        preserve_existing_glass=raw_request.preserve_existing_glass,
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

    pipeline: dict[str, object] = (
        _build_safe_duplicate_removal_pipeline_metadata(result)
    )
    persisted_asset_paths: list[str] = []
    symmetry: ObjectSymmetricDivisionMetadata | None = None
    scan_projection_stats: ScanProjectionStats | None = None
    variant_metadata: dict[str, dict[str, object]] | None = None
    try:
        _raise_if_generation_cancelled(cancel_event)
        if isinstance(result, ScanProjectedMeshyGenerationResult):
            scan_projection_stats = result.scan_projection_stats
            if scan_projection_stats is not None:
                pipeline.update(
                    _build_scan_projection_pipeline_metadata(
                        scan_projection_stats
                    )
                )
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
            if geometry_only:
                raise ValueError(
                    "Symmetric division requires a newly generated textured "
                    "model."
                )
            if not isinstance(texture_variants, ObjectTextureVariants):
                raise ValueError(
                    "Symmetric division requires a newly generated "
                    "textured model."
                )
            symmetric_builder_kwargs: dict[str, object] = {}
            if _uses_weighted_camera_projection(request):
                symmetric_builder_kwargs.update(
                    {
                        "projection_camera_percentages": (
                            request.projection_camera_percentages
                        ),
                        "cancellation_check": (
                            None
                            if cancel_event is None
                            else cancel_event.is_set
                        ),
                    }
                )
            division_result = build_automatic_symmetric_object_variants(
                texture_variants.glb_by_resolution[TEXTURE_RESOLUTION_2048],
                request.symmetric_division_orientation,
                **symmetric_builder_kwargs,
            )
            symmetry = _validate_automatic_symmetric_division_result(
                division_result,
                request.symmetric_division_orientation,
            )
            texture_variants = division_result.variants
            scan_projection_stats = division_result.scan_projection_stats
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
                _iter_variant_metadata_asset_paths(variant_metadata)
            )
            pipeline.update(
                {
                    TEXTURE_VARIANTS_PIPELINE_KEY: variant_metadata,
                    SELECTED_TEXTURE_RESOLUTION_PIPELINE_KEY: (
                        DEFAULT_TEXTURE_RESOLUTION
                    ),
                    **_build_pbr_pipeline_metadata(
                        request.enabled_pbr_maps,
                        texture_variants,
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
            staged_pipeline = _build_staged_generation_pipeline_metadata(
                result,
                source_asset_path,
            )
            if symmetry is None:
                postprocessed_asset_path = (
                    _resolve_staged_postprocessed_asset_path(
                        result,
                        asset_path,
                        variant_metadata,
                    )
                )
                if postprocessed_asset_path is None:
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

        if symmetry is not None:
            if variant_metadata is None:
                raise ValueError("Symmetric texture variants were not saved.")
            pipeline = _build_automatic_symmetric_generation_pipeline(
                pipeline,
                symmetry,
                variant_metadata,
                scan_projection_stats=scan_projection_stats,
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
            _validate_symmetric_texture_regeneration_uvs(outcome, symmetry)
            canonical_provider_glb = texture_variants.glb_by_resolution[
                TEXTURE_RESOLUTION_2048
            ]
            texture_variants = _rebuild_symmetric_texture_variants(
                canonical_provider_glb,
                symmetry,
            )
        outcome = _with_persisted_canonical_uv_fingerprint(
            outcome,
            record_snapshot,
            texture_variants,
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
            _iter_variant_metadata_asset_paths(variant_metadata)
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


def _run_interruptible_stage(
    operation: Callable[[], object],
    cancel_event: threading.Event | None = None,
) -> object:
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
        if (
            (cancel_event is not None and cancel_event.is_set())
            or qt_thread.isInterruptionRequested()
        ):
            raise _GenerationCancelled
    if (
        (cancel_event is not None and cancel_event.is_set())
        or qt_thread.isInterruptionRequested()
    ):
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
def _build_object_face_selection_context_token(
    record: GeneratedObjectRecord,
    asset_directory: Path,
) -> tuple[object, ...]:
    """Identify the exact object revision behind one selectable UV map."""

    return (
        record.object_id,
        record.asset_path,
        _build_generation_asset_revision(
            asset_directory,
            record.asset_path,
        ),
    )


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
        raw_map_paths = variant[TEXTURE_VARIANT_MAP_PNG_PATHS_KEY]
        assert isinstance(raw_map_paths, Mapping)
        for map_type, raw_path in sorted(raw_map_paths.items()):
            try:
                map_asset_path = (asset_directory / str(raw_path)).resolve()
                map_asset_path.relative_to(asset_root)
                map_stat = map_asset_path.stat()
                map_revision = (
                    map_stat.st_size,
                    map_stat.st_mtime_ns,
                    map_stat.st_ctime_ns,
                )
            except (OSError, RuntimeError, ValueError):
                map_revision = None
            file_revisions.append(
                (
                    resolution,
                    TEXTURE_VARIANT_MAP_PNG_PATHS_KEY,
                    map_type,
                    raw_path,
                    map_revision,
                )
            )
    return (
        record.pipeline.get(FACE_EDIT_TEXTURE_STALE_PIPELINE_KEY) is True,
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
) -> dict[str, object] | None:
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
    raw_map_paths = raw_variant.get(TEXTURE_VARIANT_MAP_PNG_PATHS_KEY)
    map_paths: dict[str, str] = {ATLAS_MAP_BASE_COLOR: png_path}
    if isinstance(raw_map_paths, Mapping):
        for map_type in ATLAS_MAP_TYPES:
            raw_path = raw_map_paths.get(map_type)
            if isinstance(raw_path, str) and raw_path.strip():
                map_paths[map_type] = raw_path
    return {
        TEXTURE_VARIANT_GLB_PATH_KEY: glb_path,
        TEXTURE_VARIANT_PNG_PATH_KEY: png_path,
        TEXTURE_VARIANT_MAP_PNG_PATHS_KEY: map_paths,
    }


def _build_texture_resolution_entries(
    record: GeneratedObjectRecord,
    asset_directory: Path,
) -> list[TextureAtlasEntry]:
    if record.pipeline.get(FACE_EDIT_TEXTURE_STALE_PIPELINE_KEY) is True:
        return []
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
