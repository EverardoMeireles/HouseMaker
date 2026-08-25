# ### Imports ###
from __future__ import annotations

import copy
import hashlib
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
    QFileDialog,
    QGridLayout,
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

from housemaker.camera_uv_integrity import (
    CAMERA_UV_FINGERPRINT_VERSION,
    CAMERA_UV_PROJECTION_VERSION,
    CameraUvFingerprint,
    CameraUvIntegrityError,
    build_camera_uv_fingerprint,
    validate_camera_uv_retexture_integrity,
)
from housemaker.camera_uv_projection import (
    project_uvs_from_camera_views_from_glb,
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
    request_retextured_model,
)
from housemaker.object_texture_variants import (
    DEFAULT_TEXTURE_RESOLUTION,
    TEXTURE_RESOLUTION_2048,
    TEXTURE_RESOLUTIONS,
    ObjectTextureVariants,
    build_object_texture_variants,
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
OBJECT_OPERATION_GENERATE_TEXTURE = "generate_texture"
OBJECT_OPERATION_PURGE_FACES = "purge_faces"
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
    ) -> None:
        self.frame_index = int(frame_index)
        self.selected_object_bgra = np.ascontiguousarray(
            selected_object_bgra
        ).copy()
        self.settings = settings
        self.geometry_only = bool(geometry_only)
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
    submitted_uv_fingerprint: CameraUvFingerprint | None = None

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


@dataclass(frozen=True)
class TextureRegenerationOutcome:
    """Provider result plus the immutable request and verified final UVs."""

    request: TextureRegenerationRequest
    result: MeshyGenerationResult
    final_uv_fingerprint: CameraUvFingerprint | None = None


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
    camera_uv_projection_applied: bool = False
    camera_uv_face_counts: tuple[tuple[str, int], ...] = ()
    camera_uv_leftover_face_count: int = 0
    camera_uv_invisible_face_count: int = 0
    camera_uv_quality_fallback_face_count: int = 0
    camera_uv_conflict_fallback_face_count: int = 0
    camera_uv_projection_version: str = ""
    camera_uv_fingerprint_version: str = ""
    camera_uv_submitted_fingerprint: str = ""
    camera_uv_final_fingerprint: str = ""
    camera_uv_integrity_face_count: int = 0
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
        use_camera_uv_projection = (
            request.settings.project_uvs_from_camera_views
        )
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
            and not use_camera_uv_projection
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

        camera_uv_face_counts: tuple[tuple[str, int], ...] = ()
        camera_uv_leftover_face_count = 0
        camera_uv_invisible_face_count = 0
        camera_uv_quality_fallback_face_count = 0
        camera_uv_conflict_fallback_face_count = 0
        camera_uv_submitted_fingerprint: CameraUvFingerprint | None = None
        if use_camera_uv_projection:
            if progress_callback is not None:
                progress_callback(
                    "Projecting UVs from six fixed camera views..."
                )
            try:
                projected = project_uvs_from_camera_views_from_glb(
                    processed_glb_bytes,
                    cancel_requested=(
                        None if cancel_event is None else cancel_event.is_set
                    ),
                )
            except Exception:
                # Translate the projection core's cancellation exception into
                # the worker's silent cancellation control flow.
                _raise_if_generation_cancelled(cancel_event)
                raise
            processed_glb_bytes = projected.glb_bytes
            _raise_if_generation_cancelled(cancel_event)
            try:
                camera_uv_submitted_fingerprint = (
                    build_camera_uv_fingerprint(processed_glb_bytes)
                )
            except CameraUvIntegrityError as error:
                raise CameraUvIntegrityError(
                    "Camera projection produced a GLB whose UV layout could "
                    "not be verified, so Meshy Retexture was not submitted. "
                    "Retry generation or disable 'Project UVs from camera "
                    f"views' for this object. Detail: {error}"
                ) from error
            camera_uv_face_counts = tuple(
                (
                    camera_id,
                    int(projected.camera_face_counts.get(camera_id, 0)),
                )
                for camera_id in ALL_CAMERA_IDS
            )
            camera_uv_leftover_face_count = int(
                projected.leftover_face_count
            )
            camera_uv_invisible_face_count = int(
                projected.invisible_face_count
            )
            camera_uv_quality_fallback_face_count = int(
                projected.quality_fallback_face_count
            )
            camera_uv_conflict_fallback_face_count = int(
                projected.conflict_fallback_face_count
            )

        if request.geometry_only:
            final_geometry_fingerprint = camera_uv_submitted_fingerprint
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
                camera_uv_projection_applied=use_camera_uv_projection,
                camera_uv_face_counts=camera_uv_face_counts,
                camera_uv_leftover_face_count=camera_uv_leftover_face_count,
                camera_uv_invisible_face_count=(
                    camera_uv_invisible_face_count
                ),
                camera_uv_quality_fallback_face_count=(
                    camera_uv_quality_fallback_face_count
                ),
                camera_uv_conflict_fallback_face_count=(
                    camera_uv_conflict_fallback_face_count
                ),
                camera_uv_projection_version=(
                    CAMERA_UV_PROJECTION_VERSION
                    if use_camera_uv_projection
                    else ""
                ),
                camera_uv_fingerprint_version=(
                    CAMERA_UV_FINGERPRINT_VERSION
                    if use_camera_uv_projection
                    else ""
                ),
                camera_uv_submitted_fingerprint=(
                    ""
                    if final_geometry_fingerprint is None
                    else final_geometry_fingerprint.sha256
                ),
                camera_uv_final_fingerprint=(
                    ""
                    if final_geometry_fingerprint is None
                    else final_geometry_fingerprint.sha256
                ),
                camera_uv_integrity_face_count=(
                    0
                    if final_geometry_fingerprint is None
                    else final_geometry_fingerprint.face_count
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
        textured_result = request_retextured_model(
            api_key=request.settings.meshy_api_key,
            model_glb=processed_glb_bytes,
            reference_images_png=(image_png,),
            enable_original_uv=use_camera_uv_projection,
            progress_callback=report_texture_progress,
            cancel_event=cancel_event,
        )
        camera_uv_final_fingerprint: CameraUvFingerprint | None = None
        if use_camera_uv_projection:
            _raise_if_generation_cancelled(cancel_event)
            submitted_fingerprint = camera_uv_submitted_fingerprint
            if submitted_fingerprint is None:
                raise CameraUvIntegrityError(
                    "The submitted camera UV fingerprint is unavailable. No "
                    "texture variants were saved; retry generation."
                )
            try:
                camera_uv_final_fingerprint = build_camera_uv_fingerprint(
                    textured_result.glb_bytes
                )
            except CameraUvIntegrityError as error:
                raise CameraUvIntegrityError(
                    "Meshy Retexture returned a GLB whose camera UV layout "
                    "could not be verified. No texture variants were saved; "
                    "retry or disable 'Project UVs from camera views' for "
                    f"this object. Detail: {error}"
                ) from error
            validate_camera_uv_retexture_integrity(
                submitted_fingerprint,
                camera_uv_final_fingerprint,
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
            camera_uv_projection_applied=use_camera_uv_projection,
            camera_uv_face_counts=camera_uv_face_counts,
            camera_uv_leftover_face_count=camera_uv_leftover_face_count,
            camera_uv_invisible_face_count=(
                camera_uv_invisible_face_count
            ),
            camera_uv_quality_fallback_face_count=(
                camera_uv_quality_fallback_face_count
            ),
            camera_uv_conflict_fallback_face_count=(
                camera_uv_conflict_fallback_face_count
            ),
            camera_uv_projection_version=(
                CAMERA_UV_PROJECTION_VERSION
                if use_camera_uv_projection
                else ""
            ),
            camera_uv_fingerprint_version=(
                CAMERA_UV_FINGERPRINT_VERSION
                if use_camera_uv_projection
                else ""
            ),
            camera_uv_submitted_fingerprint=(
                ""
                if camera_uv_submitted_fingerprint is None
                else camera_uv_submitted_fingerprint.sha256
            ),
            camera_uv_final_fingerprint=(
                ""
                if camera_uv_final_fingerprint is None
                else camera_uv_final_fingerprint.sha256
            ),
            camera_uv_integrity_face_count=(
                0
                if camera_uv_submitted_fingerprint is None
                else camera_uv_submitted_fingerprint.face_count
            ),
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
            prepare_executor = getattr(self._executor, "prepare", None)
            if callable(prepare_executor):
                self.progress.emit("Preparing model processor...")
                _run_interruptible_stage(prepare_executor)
            result = _run_interruptible_stage(
                lambda: _invoke_planner(
                    self._planner,
                    self._request,
                    self.progress.emit,
                    self._cancel_event,
                )
            )
            self.progress.emit(
                "Validating generated geometry..."
                if self._request.geometry_only
                else "Creating 512, 1024 and 2048 texture variants..."
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
        request: TextureRegenerationRequest,
    ) -> None:
        super().__init__()
        self._regenerator = regenerator
        self._executor = executor
        self._request = request
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    @Slot()
    def run(self) -> None:
        try:
            prepare_executor = getattr(self._executor, "prepare", None)
            if callable(prepare_executor):
                self.progress.emit("Preparing model processor...")
                _run_interruptible_stage(prepare_executor)
            result = _run_interruptible_stage(
                lambda: _invoke_texture_regenerator(
                    self._regenerator,
                    self._request,
                    self.progress.emit,
                    self._cancel_event,
                )
            )
            if not isinstance(result, MeshyGenerationResult):
                raise TypeError("Meshy returned an invalid texture result.")
            _raise_if_generation_cancelled(self._cancel_event)
            final_uv_fingerprint = self._validate_final_uvs(result)
            self.progress.emit(
                "Creating 512, 1024 and 2048 texture variants..."
            )
            generated_model = _run_interruptible_stage(
                lambda: _invoke_executor(self._executor, result)
            )
            if not isinstance(generated_model, GeneratedModel):
                raise TypeError("The Meshy executor returned an invalid model.")
            _raise_if_generation_cancelled(self._cancel_event)
            outcome = TextureRegenerationOutcome(
                request=self._request,
                result=result,
                final_uv_fingerprint=final_uv_fingerprint,
            )
        except _GenerationCancelled:
            return
        except Exception as error:
            self.failed.emit(_safe_error_message(error, self._request.settings))
            return
        else:
            self.succeeded.emit(outcome, generated_model)
        finally:
            self.finished.emit()

    def _validate_final_uvs(
        self,
        result: MeshyGenerationResult,
    ) -> CameraUvFingerprint | None:
        submitted = self._request.submitted_uv_fingerprint
        if submitted is None:
            return None
        try:
            final_fingerprint = build_camera_uv_fingerprint(result.glb_bytes)
        except CameraUvIntegrityError as error:
            raise CameraUvIntegrityError(
                "Meshy Retexture returned a GLB whose camera UV layout could "
                "not be verified. The existing texture was kept. Detail: "
                f"{error}"
            ) from error
        validate_camera_uv_retexture_integrity(
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
        self._generation_thread: QThread | None = None
        self._generation_worker: (
            GenerationWorker
            | TextureRegenerationWorker
            | UncheckedCameraFacePurgeWorker
            | None
        ) = None
        self._generated_model: GeneratedModel | None = None
        self._generated_model_cache: dict[str, GeneratedModel] = {}
        self._is_syncing_texture_resolution_view = False
        self._is_rebuilding_generation_data = False
        self._is_emitting_texture_repair = False
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

        try:
            normalized_resolution = int(resolution)
        except (TypeError, ValueError):
            return None
        record = self._find_generated_object_record(object_id)
        if record is None or normalized_resolution not in TEXTURE_RESOLUTIONS:
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

        if self._generation_thread is not None:
            return False
        record = self._find_generated_object_record(object_id)
        if record is None:
            return False
        variant = self.get_texture_variant(record.object_id, resolution)
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
        if self._selected_object_id == record.object_id:
            self._generated_model_cache[record.object_id] = preview_model
            self._display_generated_object(replacement)
            self.status_label.setText(
                f"Selected {normalized_resolution} x {normalized_resolution} "
                f"texture for {record.object_name}."
            )
        self._emit_data_changed()
        return True

    def set_external_3d_viewer_active(self, is_active: bool) -> None:
        """Show the local atlas inspector while the 3D panel is external."""

        self.object_3d_panel.set_external_presentation_active(is_active)
        self.right_view_stack.setCurrentWidget(
            self.texture_view_page if is_active else self.object_3d_page
        )

    @property
    def is_generating(self) -> bool:
        return self._generation_thread is not None

    def delete_selected_generated_object(self) -> bool:
        """Delete the selected object without showing an interactive prompt."""

        if self._selected_object_id is None:
            return False
        return self.delete_generated_object(self._selected_object_id)

    def purge_selected_object_faces(self) -> bool:
        """Delete faces visible from every currently unchecked camera."""

        if self._generation_thread is not None:
            return False
        request = self._build_unchecked_camera_face_purge_request()
        if request is None:
            return False
        self._start_unchecked_camera_face_purge(request)
        return True

    def regenerate_selected_object_texture(self) -> bool:
        """Compatibility alias for :meth:`generate_selected_object_texture`."""

        return self.generate_selected_object_texture()

    def generate_selected_object_texture(self) -> bool:
        """Generate the selected object's texture from the current mask."""

        if self._generation_thread is not None:
            return False
        request = self._build_texture_regeneration_request()
        if request is None:
            return False
        self._start_texture_regeneration(request)
        return True

    def undo_selected_object_change(self) -> bool:
        """Undo the selected object's latest texture generation or face purge."""

        if self._generation_thread is not None:
            return False
        record = self._find_generated_object_record(self._selected_object_id)
        if record is None:
            return False
        undo_stack = _get_object_operation_undo_stack(record)
        if not undo_stack:
            return False
        snapshot = undo_stack[-1]
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

        record_index = self._data.generated_objects.index(record)
        self._data.generated_objects[record_index] = replacement
        self._generated_model_cache.pop(record.object_id, None)
        selected_object_id = self._selected_object_id
        if selected_object_id == record.object_id:
            self._generated_model_cache[record.object_id] = preview_model
        self._refresh_generated_objects_list(selected_object_id)
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

    def delete_generated_object(self, object_id: str) -> bool:
        """Remove one generated object and its unreferenced GLB/PNG assets.

        This programmatic seam deliberately does not show a confirmation
        dialog. UI callers confirm first. Missing records, active generation,
        unsafe asset paths, and filesystem cleanup failures never partially
        remove another object.
        """

        if self._generation_thread is not None:
            self.status_label.setText(
                "Wait for generation to finish before deleting an object."
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

        deleted_record = self._data.generated_objects.pop(record_index)
        self._generated_model_cache.pop(object_id, None)
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

    def generate_geometry(self) -> None:
        """Generate and locally process geometry without submitting Retexture."""

        if self._generation_thread is not None:
            return
        request = self._build_generation_request(geometry_only=True)
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

    def _start_texture_regeneration(
        self,
        request: TextureRegenerationRequest,
    ) -> None:
        """Start a texture-only task for one immutable selected-object snapshot."""

        self._generation_thread = QThread(self)
        self._generation_worker = TextureRegenerationWorker(
            self._meshy_texture_regenerator,
            self._meshy_executor,
            request,
        )
        self._generation_worker.moveToThread(self._generation_thread)
        self._generation_thread.started.connect(self._generation_worker.run)
        self._generation_worker.succeeded.connect(
            self._handle_texture_regeneration_succeeded
        )
        self._generation_worker.failed.connect(
            self._handle_texture_regeneration_failed
        )
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
        self.status_label.setText("Preparing texture generation...")
        self._generation_thread.start()
        self._sync_controls()

    def _start_unchecked_camera_face_purge(
        self,
        request: UncheckedCameraFacePurgeRequest,
    ) -> None:
        """Start one local face-purge request off the UI thread."""

        self._generation_thread = QThread(self)
        self._generation_worker = UncheckedCameraFacePurgeWorker(request)
        self._generation_worker.moveToThread(self._generation_thread)
        self._generation_thread.started.connect(self._generation_worker.run)
        self._generation_worker.succeeded.connect(
            self._handle_unchecked_camera_face_purge_succeeded
        )
        self._generation_worker.failed.connect(
            self._handle_unchecked_camera_face_purge_failed
        )
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
        self.status_label.setText("Preparing unchecked-camera face purge...")
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
                if isinstance(worker, TextureRegenerationWorker):
                    worker_slots = (
                        (
                            worker.succeeded,
                            self._handle_texture_regeneration_succeeded,
                        ),
                        (
                            worker.failed,
                            self._handle_texture_regeneration_failed,
                        ),
                    )
                elif isinstance(worker, UncheckedCameraFacePurgeWorker):
                    worker_slots = (
                        (
                            worker.succeeded,
                            self._handle_unchecked_camera_face_purge_succeeded,
                        ),
                        (
                            worker.failed,
                            self._handle_unchecked_camera_face_purge_failed,
                        ),
                    )
                else:
                    worker_slots = (
                        (worker.succeeded, self._handle_generation_succeeded),
                        (worker.failed, self._handle_generation_failed),
                    )
                for signal, slot in worker_slots:
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
        if record is None or self._generation_thread is not None:
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
        pipeline: dict[str, object] = {}
        persisted_asset_paths: list[str] = []
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
                postprocessed_asset_path = self._persist_meshy_revision_asset(
                    object_id,
                    MESHY_REVISION_POSTPROCESSED,
                    result.postprocessed_glb_bytes,
                )
                persisted_asset_paths.append(postprocessed_asset_path)
                pipeline.update(
                    {
                        "mode": _staged_generation_mode(result),
                        "geometry_task_id": result.geometry_task_id,
                        "source_asset_path": source_asset_path,
                        "postprocessed_asset_path": postprocessed_asset_path,
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
                        "camera_uv_projection_applied": (
                            result.camera_uv_projection_applied
                        ),
                        "geometry_only": result.geometry_only,
                    }
                )
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
                if result.camera_uv_projection_applied:
                    projected_face_count = sum(
                        count
                        for _camera_id, count in result.camera_uv_face_counts
                    )
                    pipeline.update(
                        {
                            "camera_uv_projection_camera_ids": list(
                                ALL_CAMERA_IDS
                            ),
                            "camera_uv_face_counts": {
                                camera_id: count
                                for camera_id, count in (
                                    result.camera_uv_face_counts
                                )
                            },
                            "camera_uv_projected_face_count": (
                                projected_face_count
                            ),
                            "camera_uv_leftover_face_count": (
                                result.camera_uv_leftover_face_count
                            ),
                            "camera_uv_invisible_face_count": (
                                result.camera_uv_invisible_face_count
                            ),
                            "camera_uv_quality_fallback_face_count": (
                                result.camera_uv_quality_fallback_face_count
                            ),
                            "camera_uv_conflict_fallback_face_count": (
                                result.camera_uv_conflict_fallback_face_count
                            ),
                            "retexture_enable_original_uv": True,
                            "camera_uv_projection_version": (
                                result.camera_uv_projection_version
                            ),
                            "camera_uv_fingerprint_version": (
                                result.camera_uv_fingerprint_version
                            ),
                            "camera_uv_submitted_fingerprint": (
                                result.camera_uv_submitted_fingerprint
                            ),
                            "camera_uv_final_fingerprint": (
                                result.camera_uv_final_fingerprint
                            ),
                            "camera_uv_integrity_face_count": (
                                result.camera_uv_integrity_face_count
                            ),
                        }
                    )
        except Exception as error:
            self._remove_newly_persisted_assets(persisted_asset_paths)
            self._handle_generation_failed(
                f"The Meshy texture variants could not be prepared locally: {error}"
            )
            return
        record = GeneratedObjectRecord(
            object_id=object_id,
            frame_index=self._data.current_frame_index,
            object_name=object_name,
            pipeline=pipeline,
            provider=GENERATION_BACKEND_MESHY,
            provider_task_id=result.task_id,
            asset_path=asset_path,
        )
        self._data.generated_objects.append(record)
        self._generated_model_cache[object_id] = generated_model
        self._selected_object_id = object_id
        self._generated_model = generated_model
        self._refresh_generated_objects_list(object_id)
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
        self._emit_data_changed()
        self.generation_completed.emit(record, generated_model)

    @Slot(object, object)
    def _handle_texture_regeneration_succeeded(
        self,
        raw_outcome: object,
        generated_model: GeneratedModel,
    ) -> None:
        if not isinstance(raw_outcome, TextureRegenerationOutcome):
            self._handle_texture_regeneration_failed(
                "Meshy returned an invalid texture-regeneration result."
            )
            return
        outcome = raw_outcome
        request = outcome.request
        result = outcome.result
        record = self._find_generated_object_record(request.object_id)
        if record is None:
            self._handle_texture_regeneration_failed(
                "The target generated object no longer exists."
            )
            return

        persisted_asset_paths: list[str] = []
        try:
            texture_variants = generated_model.object_texture_variants
            if texture_variants is None:
                raise ValueError(
                    "The regenerated model has no selectable texture variants."
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
            if selected_resolution not in TEXTURE_RESOLUTIONS:
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
                f"{error}"
            )
            return

        record_index = self._data.generated_objects.index(record)
        self._data.generated_objects[record_index] = replacement
        self._generated_model_cache.pop(record.object_id, None)
        selected_object_id = self._selected_object_id
        if selected_object_id == record.object_id:
            self._generated_model_cache[record.object_id] = preview_model
        self._refresh_generated_objects_list(selected_object_id)
        cleanup_failed = self._delete_unreferenced_object_assets(record)
        status_suffix = (
            " Some superseded texture files could not be removed."
            if cleanup_failed
            else ""
        )
        self.status_label.setText(
            f"Generated texture: {record.object_name}." + status_suffix
        )
        self._emit_data_changed()
        self.texture_regeneration_completed.emit(replacement, preview_model)
        self.generated_object_changed.emit(replacement, preview_model)

    @Slot(object)
    def _handle_unchecked_camera_face_purge_succeeded(
        self,
        raw_outcome: object,
    ) -> None:
        if not isinstance(raw_outcome, UncheckedCameraFacePurgeOutcome):
            self._handle_unchecked_camera_face_purge_failed(
                "The face-purge worker returned an invalid result."
            )
            return
        outcome = raw_outcome
        record = self._find_generated_object_record(outcome.request.object_id)
        if record is None:
            self._handle_unchecked_camera_face_purge_failed(
                "The target generated object no longer exists."
            )
            return

        persisted_asset_paths: list[str] = []
        try:
            texture_variants = build_object_texture_variants(
                outcome.result.glb_bytes
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
                if selected_resolution not in TEXTURE_RESOLUTIONS:
                    selected_resolution = DEFAULT_TEXTURE_RESOLUTION
                selected_variant = variant_metadata[str(selected_resolution)]
                asset_path = selected_variant[TEXTURE_VARIANT_GLB_PATH_KEY]
                preview_model = import_generated_glb(
                    texture_variants.glb_by_resolution[selected_resolution]
                )
            next_pipeline = _build_unchecked_camera_face_purge_pipeline(
                record,
                outcome,
                variant_metadata,
                postprocessed_asset_path=asset_path,
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
                f"The purged object could not be saved locally: {error}"
            )
            return

        record_index = self._data.generated_objects.index(record)
        self._data.generated_objects[record_index] = replacement
        self._generated_model_cache.pop(record.object_id, None)
        selected_object_id = self._selected_object_id
        if selected_object_id == record.object_id:
            self._generated_model_cache[record.object_id] = preview_model
        self._refresh_generated_objects_list(selected_object_id)
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

    @Slot(str)
    def _handle_generation_failed(self, error_message: str) -> None:
        self.status_label.setText(f"Generation failed: {error_message}")
        QMessageBox.warning(self, "Generation failed", error_message)

    @Slot(str)
    def _handle_texture_regeneration_failed(self, error_message: str) -> None:
        self.status_label.setText(
            f"Texture generation failed: {error_message}"
        )
        QMessageBox.warning(
            self,
            "Texture generation failed",
            error_message,
        )

    @Slot(str)
    def _handle_unchecked_camera_face_purge_failed(
        self,
        error_message: str,
    ) -> None:
        self.status_label.setText(f"Face purge failed: {error_message}")
        QMessageBox.warning(self, "Face purge failed", error_message)

    @Slot()
    def _handle_generation_thread_finished(self) -> None:
        if self.sender() is not self._generation_thread:
            return
        self._generation_worker = None
        self._generation_thread = None
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
        return GenerationRequest(
            frame_index=self._data.current_frame_index,
            selected_object_bgra=selected_crop,
            settings=self._settings,
            enabled_camera_ids=(
                self.object_3d_panel.get_enabled_postprocess_camera_ids()
            ),
            geometry_only=geometry_only,
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
                TEXTURE_RESOLUTION_2048,
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
    ) -> TextureRegenerationRequest | None:
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
            source_path = self._resolve_texture_regeneration_source_path(record)
            model_glb = source_path.read_bytes()
            import_generated_glb(model_glb)
            preserve_camera_uvs = (
                self._settings.project_uvs_from_camera_views
                and _record_uses_camera_projected_uvs(record)
            )
            submitted_fingerprint = None
            if preserve_camera_uvs:
                submitted_fingerprint = build_camera_uv_fingerprint(model_glb)
                _validate_stored_camera_uv_fingerprint(
                    record,
                    submitted_fingerprint,
                )
            request = TextureRegenerationRequest(
                object_id=record.object_id,
                reference_frame_index=self._data.current_frame_index,
                reference_image_bgra=selected_crop,
                model_glb=model_glb,
                settings=self._settings,
                enable_original_uv=preserve_camera_uvs,
                submitted_uv_fingerprint=submitted_fingerprint,
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

        raw_postprocessed_path = record.pipeline.get(
            "postprocessed_asset_path"
        )
        if isinstance(raw_postprocessed_path, str) and raw_postprocessed_path:
            source_path = self._resolve_meshy_asset_path(
                raw_postprocessed_path
            )
        else:
            if _record_uses_camera_projected_uvs(record):
                raise ValueError(
                    "The processed camera-UV model revision is unavailable."
                )
            canonical_variant = _get_texture_variant_metadata(
                record,
                DEFAULT_TEXTURE_RESOLUTION,
            )
            raw_source_path = (
                record.asset_path
                if canonical_variant is None
                else canonical_variant[TEXTURE_VARIANT_GLB_PATH_KEY]
            )
            source_path = self._resolve_meshy_asset_path(raw_source_path)
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
        is_generating = self._generation_thread is not None
        has_mask = self.video_view.has_selection()
        self.load_video_button.setEnabled(not is_generating)
        self.seekbar.setEnabled(has_video and not is_generating)
        mask_tool_is_available = has_video and not is_generating
        self.paint_mask_button.setEnabled(mask_tool_is_available)
        self.erase_mask_button.setEnabled(mask_tool_is_available)
        self.brush_size_spinbox.setEnabled(mask_tool_is_available)
        self.undo_mask_button.setEnabled(
            has_video
            and bool(self.video_view.get_strokes())
            and not is_generating
        )
        self.clear_mask_button.setEnabled(
            has_video and has_mask and not is_generating
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
            not is_generating
        )
        self.result_view.set_enabled_unused_face_camera_ids(
            enabled_camera_ids
        )
        self.result_view.set_unused_face_camera_indicators_visible(
            True
        )
        self.meshy_target_polycount_control.setVisible(True)
        self.meshy_target_polycount_spinbox.setEnabled(
            not is_generating
        )
        self.ambient_light_slider.setEnabled(not is_generating)
        self.textures_checkbox.setEnabled(not is_generating)
        self.wireframe_checkbox.setEnabled(not is_generating)
        self.delete_generated_object_button.setEnabled(
            not is_generating
            and self._find_generated_object_record(
                self._selected_object_id
            )
            is not None
        )
        self.regenerate_texture_button.setEnabled(
            self._can_regenerate_object_texture(
                self._find_generated_object_record(self._selected_object_id)
            )
        )
        self.undo_object_change_button.setEnabled(
            not is_generating
            and bool(
                _get_object_operation_undo_stack(
                    self._find_generated_object_record(
                        self._selected_object_id
                    )
                )
            )
        )
        self.purge_faces_button.setEnabled(
            not is_generating
            and len(enabled_camera_ids) < len(ALL_CAMERA_IDS)
            and self._find_generated_object_record(
                self._selected_object_id
            )
            is not None
        )
        self.generate_button.setEnabled(
            has_video
            and has_mask
            and required_key_is_available
            and camera_selection_is_valid
            and not is_generating
        )
        self.generate_geometry_button.setEnabled(
            has_video
            and has_mask
            and required_key_is_available
            and camera_selection_is_valid
            and not is_generating
        )
        self.video_view.set_interaction_enabled(
            has_video and not is_generating
        )

    def _can_regenerate_object_texture(
        self,
        record: GeneratedObjectRecord | None,
    ) -> bool:
        if (
            record is None
            or self._generation_thread is not None
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
        record = self._repair_missing_active_texture_variant(record)
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
        self._sync_model_statistics(generated_model)
        self.texture_view.set_uv_overlay_triangles(
            _collect_model_uv_triangles(generated_model)
        )
        self._refresh_object_texture_atlases(
            record.object_id,
        )

    def _repair_missing_active_texture_variant(
        self,
        record: GeneratedObjectRecord,
    ) -> GeneratedObjectRecord:
        """Select the first complete variant if the saved active one vanished."""

        raw_variants = record.pipeline.get(TEXTURE_VARIANTS_PIPELINE_KEY)
        if not isinstance(raw_variants, dict):
            return record
        if self.get_active_texture_variant(record.object_id) is not None:
            return record
        missing_resolution = _get_selected_texture_resolution(record)
        fallback_resolutions = sorted(
            TEXTURE_RESOLUTIONS,
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
        entries = _build_texture_resolution_entries(
            record,
            self._asset_directory,
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
        self.select_object_texture_resolution(record.object_id, resolution)

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
        variants: ObjectTextureVariants,
        *,
        asset_stem: str | None = None,
    ) -> dict[str, dict[str, str]]:
        """Atomically persist each selectable GLB and its atlas-ready PNG."""

        normalized_asset_stem = (
            str(object_id) if asset_stem is None else str(asset_stem)
        )
        if (
            not normalized_asset_stem
            or Path(normalized_asset_stem).name != normalized_asset_stem
            or normalized_asset_stem in {".", ".."}
        ):
            raise ValueError("Texture variant asset stem is unsafe.")
        metadata: dict[str, dict[str, str]] = {}
        created_paths: list[str] = []
        try:
            for resolution in TEXTURE_RESOLUTIONS:
                glb_path = self._persist_meshy_named_asset(
                    f"{normalized_asset_stem}.texture-{resolution}.glb",
                    variants.glb_by_resolution[resolution],
                )
                created_paths.append(glb_path)
                png_path = self._persist_meshy_named_asset(
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
            self._remove_newly_persisted_assets(created_paths)
            raise

    def _remove_newly_persisted_assets(self, raw_paths: Sequence[str]) -> None:
        """Best-effort rollback for files written by one failed generation."""

        for raw_path in raw_paths:
            try:
                path = self._resolve_generated_asset_path(
                    raw_path,
                    allowed_suffixes=frozenset({".glb", ".png"}),
                )
                path.unlink(missing_ok=True)
            except (OSError, ValueError):
                continue

    def _persist_meshy_named_asset(
        self,
        file_name: str,
        glb_bytes: bytes,
    ) -> str:
        self._asset_directory.mkdir(parents=True, exist_ok=True)
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
    if result.camera_uv_projection_applied:
        stages.append("camera_uv_projection")
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
    if result.camera_uv_projection_applied:
        projected_face_count = sum(
            count for _camera_id, count in result.camera_uv_face_counts
        )
        categorized_fallback_count = (
            result.camera_uv_invisible_face_count
            + result.camera_uv_quality_fallback_face_count
            + result.camera_uv_conflict_fallback_face_count
        )
        if categorized_fallback_count == result.camera_uv_leftover_face_count:
            messages.append(
                f"Projected {projected_face_count} faces from six fixed "
                "camera views; fallback UV islands contain "
                + _format_fallback_face_count(
                    result.camera_uv_invisible_face_count,
                    "invisible",
                )
                + ", "
                + _format_fallback_face_count(
                    result.camera_uv_quality_fallback_face_count,
                    "quality-rejected depth-visible",
                )
                + ", and "
                + _format_fallback_face_count(
                    result.camera_uv_conflict_fallback_face_count,
                    "projection-conflict",
                )
                + "."
            )
        else:
            messages.append(
                f"Projected {projected_face_count} faces from six fixed "
                f"camera views; {result.camera_uv_leftover_face_count} faces "
                "use fallback UV islands."
            )
    return " ".join(messages)


def _format_fallback_face_count(count: int, description: str) -> str:
    face_label = "face" if count == 1 else "faces"
    return f"{count} {description} {face_label}"


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
    if selected_resolution not in TEXTURE_RESOLUTIONS:
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
            str(TEXTURE_RESOLUTION_2048)
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
    if _record_uses_camera_projected_uvs(record):
        fingerprint = build_camera_uv_fingerprint(result.glb_bytes)
        pipeline.update(
            {
                "camera_uv_fingerprint_version": (
                    CAMERA_UV_FINGERPRINT_VERSION
                ),
                "camera_uv_submitted_fingerprint": fingerprint.sha256,
                "camera_uv_final_fingerprint": fingerprint.sha256,
                "camera_uv_integrity_face_count": fingerprint.face_count,
            }
        )
    return pipeline


# ### Texture-regeneration helpers ###
def _record_uses_camera_projected_uvs(
    record: GeneratedObjectRecord,
) -> bool:
    if record.pipeline.get("camera_uv_projection_applied"):
        return True
    raw_mode = record.pipeline.get("mode")
    return isinstance(raw_mode, str) and "camera_uv_projection" in raw_mode


def _validate_stored_camera_uv_fingerprint(
    record: GeneratedObjectRecord,
    actual: CameraUvFingerprint,
) -> None:
    stored_version = record.pipeline.get("camera_uv_fingerprint_version")
    if stored_version != CAMERA_UV_FINGERPRINT_VERSION:
        raise CameraUvIntegrityError(
            "The saved camera-UV fingerprint version is unavailable or "
            "unsupported. The existing texture was kept."
        )
    stored_sha256 = record.pipeline.get("camera_uv_submitted_fingerprint")
    if not isinstance(stored_sha256, str) or len(stored_sha256) != 64:
        raise CameraUvIntegrityError(
            "The saved camera-UV fingerprint is unavailable. The existing "
            "texture was kept."
        )
    try:
        int(stored_sha256, 16)
    except ValueError as error:
        raise CameraUvIntegrityError(
            "The saved camera-UV fingerprint is malformed. The existing "
            "texture was kept."
        ) from error
    if stored_sha256 != actual.sha256:
        raise CameraUvIntegrityError(
            "The saved processed model no longer matches its camera-UV "
            "fingerprint. The existing texture was kept."
        )
    raw_face_count = record.pipeline.get("camera_uv_integrity_face_count")
    if (
        isinstance(raw_face_count, bool)
        or not isinstance(raw_face_count, int)
        or raw_face_count <= 0
    ):
        raise CameraUvIntegrityError(
            "The saved camera-UV face count is unavailable. The existing "
            "texture was kept."
        )
    if raw_face_count != actual.face_count:
        raise CameraUvIntegrityError(
            "The saved processed model face count no longer matches its "
            "camera-UV provenance. The existing texture was kept."
        )


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
    if selected_resolution not in TEXTURE_RESOLUTIONS:
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
            "texture_regeneration_history": history[-25:],
        }
    )
    if request.submitted_uv_fingerprint is not None:
        final_fingerprint = outcome.final_uv_fingerprint
        if final_fingerprint is None:
            raise CameraUvIntegrityError(
                "The regenerated texture UV fingerprint is unavailable."
            )
        pipeline.update(
            {
                "retexture_enable_original_uv": True,
                "texture_regeneration_uv_fingerprint_version": (
                    CAMERA_UV_FINGERPRINT_VERSION
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
        if _record_uses_camera_projected_uvs(record):
            pipeline.update(
                {
                    "camera_uv_fingerprint_version": (
                        CAMERA_UV_FINGERPRINT_VERSION
                    ),
                    "camera_uv_submitted_fingerprint": (
                        request.submitted_uv_fingerprint.sha256
                    ),
                    "camera_uv_final_fingerprint": (
                        final_fingerprint.sha256
                    ),
                    "camera_uv_integrity_face_count": (
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


# ### Texture-atlas helpers ###
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
    for resolution in TEXTURE_RESOLUTIONS:
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
