# ### Imports ###
from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath, PureWindowsPath

from housemaker.camera_models import CameraPose
from housemaker.generation_state import (
    MAX_MASK_STROKES_PER_FRAME,
    MaskStroke,
)
from housemaker.video_source import VideoMetadata


# ### Constants ###
SURFACE_TEXTURE_SCHEMA_VERSION = 4
SURFACE_TYPE_WALL = "wall"
SURFACE_TYPE_FLOOR = "floor"
SURFACE_TYPE_CEILING = "ceiling"
SURFACE_TYPES = frozenset(
    {
        SURFACE_TYPE_WALL,
        SURFACE_TYPE_FLOOR,
        SURFACE_TYPE_CEILING,
    }
)
MAX_VIDEO_FRAME_INDEX = 2_147_483_647
MAX_MASKED_FRAMES = 100_000
MAX_SELECTED_SURFACES = 10_000
MAX_SURFACE_TEXTURE_ASSIGNMENTS = 10_000
MAX_REFERENCE_FRAMES_PER_ASSIGNMENT = 1_024
MAX_SURFACE_ID_LENGTH = 512
MAX_ASSIGNMENT_ID_LENGTH = 256
MAX_PROVIDER_NAME_LENGTH = 128
MAX_PROVIDER_TASK_ID_LENGTH = 512
MAX_ASSET_PATH_LENGTH = 2_048
MAX_AREA_DESCRIPTION_LENGTH = 4_096
MAX_COMBINED_AREA_M2 = 1_000_000_000.0
MAX_TEXTURE_DIMENSION_PIXELS = 16_384
SURFACE_TEXTURE_RESOLUTIONS = (512, 1024, 2048)
DEFAULT_SURFACE_TEXTURE_RESOLUTION = 1024
MAX_LOCALIZED_INPAINT_UNDO_HISTORY = 10
MAX_SURFACE_OVERLAY_PLANES = 10_000
MAX_SURFACE_OVERLAY_OFFSET_METERS = 0.05
SURFACE_OVERLAY_ID_SUFFIX = "/overlay:1"

_SURFACE_ID_PATTERN = re.compile(
    r"^level:(?P<level_index>0|[1-9]\d*)/"
    r"(?:room:(?P<room_identity>0|[1-9]\d*)/)?"
    r"(?:(?P<wall>wall):(?P<wall_key>[1-9]\d*:[1-9]\d*)|"
    r"(?P<plane>floor|ceiling))"
    r"(?P<overlay>/overlay:(?P<overlay_key>[1-9]\d*))?$"
)


# ### Surface-overlay models ###
@dataclass(frozen=True)
class SurfaceTextureOverlayPlane:
    """One persisted close-offset plane derived from a fixed surface."""

    parent_surface_id: str
    normal_offset_meters: float

    def __post_init__(self) -> None:
        parent_surface_id = _normalize_surface_id(self.parent_surface_id)
        if _is_overlay_surface_id(parent_surface_id):
            raise ValueError("A surface overlay cannot be placed on another overlay.")
        normal_offset_meters = _normalize_surface_overlay_offset(
            self.normal_offset_meters
        )
        object.__setattr__(self, "parent_surface_id", parent_surface_id)
        object.__setattr__(self, "normal_offset_meters", normal_offset_meters)

    @property
    def surface_id(self) -> str:
        return f"{self.parent_surface_id}{SURFACE_OVERLAY_ID_SUFFIX}"

    @property
    def surface_type(self) -> str:
        return _surface_type_for_id(self.parent_surface_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "surface_id": self.surface_id,
            "parent_surface_id": self.parent_surface_id,
            "normal_offset_meters": float(self.normal_offset_meters),
        }

    @classmethod
    def from_dict(cls, payload: object) -> "SurfaceTextureOverlayPlane":
        if not isinstance(payload, dict):
            raise ValueError("A surface overlay plane must contain an object.")
        plane = cls(
            parent_surface_id=str(payload.get("parent_surface_id", "")),
            normal_offset_meters=payload.get("normal_offset_meters", 0.0),
        )
        raw_surface_id = payload.get("surface_id")
        if raw_surface_id is not None and str(raw_surface_id) != plane.surface_id:
            raise ValueError("A surface overlay plane has an inconsistent ID.")
        return plane


# ### Generated-texture models ###
@dataclass(frozen=True)
class SurfaceTextureVariant:
    """One exact square PNG belonging to a generated surface texture."""

    resolution: int
    asset_path: str

    def __post_init__(self) -> None:
        resolution = _normalize_surface_texture_resolution(self.resolution)
        asset_path = _normalize_safe_relative_asset_path(self.asset_path)
        object.__setattr__(self, "resolution", resolution)
        object.__setattr__(self, "asset_path", asset_path)

    def to_dict(self) -> dict[str, object]:
        return {
            "resolution": self.resolution,
            "asset_path": self.asset_path,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "SurfaceTextureVariant":
        if not isinstance(payload, dict):
            raise ValueError("A surface texture variant must contain an object.")
        return cls(
            resolution=payload.get("resolution", payload.get("size")),
            asset_path=str(payload.get("asset_path", payload.get("path", ""))),
        )


@dataclass(frozen=True)
class SurfaceTextureAssignment:
    """One generated texture assigned to a homogeneous group of surfaces."""

    assignment_id: str
    surface_type: str
    surface_ids: tuple[str, ...]
    provider: str
    asset_path: str
    provider_task_id: str | None = None
    combined_area_m2: float = 0.0
    area_description: str = ""
    reference_frame_indices: tuple[int, ...] = ()
    texture_width: int | None = None
    texture_height: int | None = None
    texture_variants: tuple[SurfaceTextureVariant, ...] = ()
    selected_texture_resolution: int | None = None

    def __post_init__(self) -> None:
        assignment_id = _normalize_required_text(
            self.assignment_id,
            "Surface texture assignment ID",
            MAX_ASSIGNMENT_ID_LENGTH,
        )
        provider = _normalize_required_text(
            self.provider,
            "Surface texture provider",
            MAX_PROVIDER_NAME_LENGTH,
        )
        surface_type = _normalize_surface_type(self.surface_type)
        surface_ids = _normalize_surface_ids(
            self.surface_ids,
            expected_type=surface_type,
            allow_empty=False,
            maximum_count=MAX_SELECTED_SURFACES,
        )
        asset_path = _normalize_safe_relative_asset_path(self.asset_path)
        provider_task_id = _normalize_optional_text(
            self.provider_task_id,
            "Surface texture provider task ID",
            MAX_PROVIDER_TASK_ID_LENGTH,
        )
        combined_area_m2 = _normalize_combined_area(self.combined_area_m2)
        area_description = _normalize_optional_text(
            self.area_description,
            "Surface texture area description",
            MAX_AREA_DESCRIPTION_LENGTH,
        ) or ""
        reference_frame_indices = _normalize_reference_frame_indices(
            self.reference_frame_indices
        )
        texture_width, texture_height = _normalize_texture_dimensions(
            self.texture_width,
            self.texture_height,
        )
        texture_variants = _normalize_surface_texture_variants(
            self.texture_variants
        )
        selected_texture_resolution = _normalize_selected_texture_resolution(
            self.selected_texture_resolution,
            texture_variants,
            asset_path,
        )
        if selected_texture_resolution is not None:
            selected_variant = next(
                variant
                for variant in texture_variants
                if variant.resolution == selected_texture_resolution
            )
            if selected_variant.asset_path != asset_path:
                raise ValueError(
                    "The active surface texture asset does not match its "
                    "selected resolution."
                )
            if texture_width is None and texture_height is None:
                texture_width = selected_texture_resolution
                texture_height = selected_texture_resolution
            elif (texture_width, texture_height) != (
                selected_texture_resolution,
                selected_texture_resolution,
            ):
                raise ValueError(
                    "The active surface texture dimensions do not match its "
                    "selected resolution."
                )

        object.__setattr__(self, "assignment_id", assignment_id)
        object.__setattr__(self, "surface_type", surface_type)
        object.__setattr__(self, "surface_ids", surface_ids)
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "asset_path", asset_path)
        object.__setattr__(self, "provider_task_id", provider_task_id)
        object.__setattr__(self, "combined_area_m2", combined_area_m2)
        object.__setattr__(self, "area_description", area_description)
        object.__setattr__(
            self,
            "reference_frame_indices",
            reference_frame_indices,
        )
        object.__setattr__(self, "texture_width", texture_width)
        object.__setattr__(self, "texture_height", texture_height)
        object.__setattr__(self, "texture_variants", texture_variants)
        object.__setattr__(
            self,
            "selected_texture_resolution",
            selected_texture_resolution,
        )

    def texture_variant_for_resolution(
        self,
        resolution: int,
    ) -> SurfaceTextureVariant | None:
        """Return one exact persisted variant without changing selection."""

        try:
            normalized_resolution = _normalize_surface_texture_resolution(
                resolution
            )
        except ValueError:
            return None
        return next(
            (
                variant
                for variant in self.texture_variants
                if variant.resolution == normalized_resolution
            ),
            None,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "assignment_id": self.assignment_id,
            "surface_type": self.surface_type,
            "surface_ids": list(self.surface_ids),
            "provider": self.provider,
            "provider_task_id": self.provider_task_id,
            "asset_path": self.asset_path,
            "combined_area_m2": float(self.combined_area_m2),
            "area_description": self.area_description,
            "reference_frame_indices": list(self.reference_frame_indices),
            "texture_width": self.texture_width,
            "texture_height": self.texture_height,
            "texture_variants": [
                variant.to_dict() for variant in self.texture_variants
            ],
            "selected_texture_resolution": self.selected_texture_resolution,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "SurfaceTextureAssignment":
        if not isinstance(payload, dict):
            raise ValueError("Surface texture assignment data must contain an object.")

        raw_surface_ids = payload.get("surface_ids", payload.get("surfaces", ()))
        if not isinstance(raw_surface_ids, list | tuple):
            raise ValueError("Surface texture assignment surfaces must contain a list.")
        surface_ids = tuple(str(surface_id) for surface_id in raw_surface_ids)
        raw_surface_type = payload.get("surface_type", payload.get("type"))
        surface_type = (
            _infer_surface_type(surface_ids)
            if raw_surface_type is None
            else str(raw_surface_type)
        )

        raw_reference_frames = payload.get(
            "reference_frame_indices",
            payload.get("frame_indices", ()),
        )
        if not isinstance(raw_reference_frames, list | tuple):
            raise ValueError("Surface texture reference frames must contain a list.")
        raw_texture_variants = payload.get(
            "texture_variants",
            payload.get("variants", ()),
        )
        if not isinstance(raw_texture_variants, list | tuple):
            raise ValueError("Surface texture variants must contain a list.")
        texture_variants = tuple(
            SurfaceTextureVariant.from_dict(raw_variant)
            for raw_variant in raw_texture_variants
        )

        return cls(
            assignment_id=str(payload.get("assignment_id", payload.get("id", ""))),
            surface_type=surface_type,
            surface_ids=surface_ids,
            provider=str(payload.get("provider", "unknown")),
            provider_task_id=(
                payload.get("provider_task_id", payload.get("task_id"))
            ),
            asset_path=str(payload.get("asset_path", payload.get("path", ""))),
            combined_area_m2=payload.get(
                "combined_area_m2",
                payload.get("area_m2", 0.0),
            ),
            area_description=str(payload.get("area_description", "")),
            reference_frame_indices=tuple(raw_reference_frames),
            texture_width=payload.get("texture_width", payload.get("width")),
            texture_height=payload.get("texture_height", payload.get("height")),
            texture_variants=texture_variants,
            selected_texture_resolution=payload.get(
                "selected_texture_resolution",
                payload.get("texture_resolution"),
            ),
        )


# ### Localized-inpaint undo models ###
@dataclass(frozen=True)
class SurfaceTextureInpaintUndoSnapshot:
    """One complete pre-inpaint assignment state plus its painted masks."""

    previous_assignments: tuple[SurfaceTextureAssignment, ...]
    replacement_assignment_ids: tuple[str, ...]
    affected_surface_ids: tuple[str, ...]
    previous_texture_mask_strokes: dict[str, tuple[MaskStroke, ...]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        previous_assignments = tuple(self.previous_assignments)
        if len(previous_assignments) > MAX_SURFACE_TEXTURE_ASSIGNMENTS:
            raise ValueError("An inpaint undo snapshot has too many assignments.")
        if not all(
            isinstance(assignment, SurfaceTextureAssignment)
            for assignment in previous_assignments
        ):
            raise ValueError("An inpaint undo snapshot has invalid assignments.")
        previous_ids = [
            assignment.assignment_id for assignment in previous_assignments
        ]
        if len(previous_ids) != len(set(previous_ids)):
            raise ValueError("An inpaint undo snapshot has duplicate assignments.")

        replacement_ids = _normalize_assignment_ids(
            self.replacement_assignment_ids,
            allow_empty=False,
        )
        affected_ids = _normalize_undo_surface_ids(self.affected_surface_ids)
        previous_strokes: dict[str, tuple[MaskStroke, ...]] = {}
        if not isinstance(self.previous_texture_mask_strokes, dict):
            raise ValueError("An inpaint undo snapshot has invalid mask strokes.")
        for raw_surface_id, raw_strokes in (
            self.previous_texture_mask_strokes.items()
        ):
            surface_id = _normalize_surface_id(raw_surface_id)
            if surface_id not in affected_ids:
                raise ValueError(
                    "An inpaint undo mask targets an unaffected surface."
                )
            strokes = tuple(_normalize_frame_strokes(raw_strokes))
            if strokes:
                previous_strokes[surface_id] = strokes

        object.__setattr__(self, "previous_assignments", previous_assignments)
        object.__setattr__(self, "replacement_assignment_ids", replacement_ids)
        object.__setattr__(self, "affected_surface_ids", affected_ids)
        object.__setattr__(
            self,
            "previous_texture_mask_strokes",
            previous_strokes,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "previous_assignments": [
                assignment.to_dict() for assignment in self.previous_assignments
            ],
            "replacement_assignment_ids": list(self.replacement_assignment_ids),
            "affected_surface_ids": list(self.affected_surface_ids),
            "previous_texture_mask_strokes": {
                surface_id: [stroke.to_dict() for stroke in strokes]
                for surface_id, strokes in sorted(
                    self.previous_texture_mask_strokes.items()
                )
            },
        }

    @classmethod
    def from_dict(cls, payload: object) -> "SurfaceTextureInpaintUndoSnapshot":
        if not isinstance(payload, dict):
            raise ValueError("An inpaint undo snapshot must contain an object.")
        raw_previous = payload.get("previous_assignments", ())
        if not isinstance(raw_previous, list | tuple):
            raise ValueError("Inpaint undo assignments must contain a list.")
        previous_assignments = tuple(
            SurfaceTextureAssignment.from_dict(raw_assignment)
            for raw_assignment in raw_previous
        )
        raw_replacement_ids = payload.get("replacement_assignment_ids", ())
        raw_affected_ids = payload.get("affected_surface_ids", ())
        if not isinstance(raw_replacement_ids, list | tuple) or not isinstance(
            raw_affected_ids,
            list | tuple,
        ):
            raise ValueError("Inpaint undo surface and assignment IDs need lists.")
        previous_strokes = _load_texture_mask_strokes(
            payload.get("previous_texture_mask_strokes", {})
        )
        return cls(
            previous_assignments=previous_assignments,
            replacement_assignment_ids=tuple(
                str(value) for value in raw_replacement_ids
            ),
            affected_surface_ids=tuple(str(value) for value in raw_affected_ids),
            previous_texture_mask_strokes={
                surface_id: tuple(strokes)
                for surface_id, strokes in previous_strokes.items()
            },
        )


# ### Workspace state ###
@dataclass
class SurfaceTextureData:
    """Project-safe state for surface selection and texture generation."""

    video_metadata: VideoMetadata | None = None
    current_frame_index: int = 0
    frame_strokes: dict[int, list[MaskStroke]] = field(default_factory=dict)
    texture_mask_strokes: dict[str, list[MaskStroke]] = field(default_factory=dict)
    camera_pose: CameraPose | None = None
    selected_surface_type: str | None = None
    selected_surface_ids: tuple[str, ...] = ()
    assignments: list[SurfaceTextureAssignment] = field(default_factory=list)
    overlay_planes: list[SurfaceTextureOverlayPlane] = field(default_factory=list)
    localized_inpaint_undo_stack: list[
        SurfaceTextureInpaintUndoSnapshot
    ] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.video_metadata is not None and not isinstance(
            self.video_metadata,
            VideoMetadata,
        ):
            raise ValueError("Surface texture video metadata has an invalid type.")
        if self.camera_pose is not None and not isinstance(self.camera_pose, CameraPose):
            raise ValueError("Surface texture camera pose has an invalid type.")

        current_frame_index = _normalize_frame_index(self.current_frame_index)
        if not _frame_index_is_in_video(current_frame_index, self.video_metadata):
            raise ValueError("Current surface texture frame is outside the video.")

        normalized_frame_strokes: dict[int, list[MaskStroke]] = {}
        if len(self.frame_strokes) > MAX_MASKED_FRAMES:
            raise ValueError("Surface texture data contains too many masked frames.")
        for raw_frame_index, raw_strokes in self.frame_strokes.items():
            frame_index = _normalize_frame_index(raw_frame_index)
            if not _frame_index_is_in_video(frame_index, self.video_metadata):
                raise ValueError("A masked surface texture frame is outside the video.")
            strokes = _normalize_frame_strokes(raw_strokes)
            if strokes:
                normalized_frame_strokes[frame_index] = strokes

        normalized_texture_mask_strokes = _normalize_texture_mask_strokes(
            self.texture_mask_strokes
        )

        selected_surface_type, selected_surface_ids = _normalize_selection(
            self.selected_surface_type,
            self.selected_surface_ids,
        )
        overlay_planes = list(self.overlay_planes)
        if len(overlay_planes) > MAX_SURFACE_OVERLAY_PLANES:
            raise ValueError("Surface texture data contains too many overlay planes.")
        if not all(
            isinstance(plane, SurfaceTextureOverlayPlane)
            for plane in overlay_planes
        ):
            raise ValueError(
                "Surface texture overlay planes have invalid values."
            )
        parent_surface_ids = [plane.parent_surface_id for plane in overlay_planes]
        if len(set(parent_surface_ids)) != len(parent_surface_ids):
            raise ValueError("Only one overlay plane may exist per fixed surface.")
        assignments = list(self.assignments)
        if len(assignments) > MAX_SURFACE_TEXTURE_ASSIGNMENTS:
            raise ValueError("Surface texture data contains too many assignments.")
        if not all(
            isinstance(assignment, SurfaceTextureAssignment)
            for assignment in assignments
        ):
            raise ValueError(
                "Surface texture assignments must be SurfaceTextureAssignment values."
            )
        assignment_ids = [assignment.assignment_id for assignment in assignments]
        if len(set(assignment_ids)) != len(assignment_ids):
            raise ValueError("Surface texture assignment IDs must be unique.")
        undo_stack = list(self.localized_inpaint_undo_stack)
        if len(undo_stack) > MAX_LOCALIZED_INPAINT_UNDO_HISTORY:
            raise ValueError("Surface texture data has too much inpaint undo history.")
        if not all(
            isinstance(snapshot, SurfaceTextureInpaintUndoSnapshot)
            for snapshot in undo_stack
        ):
            raise ValueError("Surface texture inpaint undo history is invalid.")

        self.current_frame_index = current_frame_index
        self.frame_strokes = normalized_frame_strokes
        self.texture_mask_strokes = normalized_texture_mask_strokes
        self.selected_surface_type = selected_surface_type
        self.selected_surface_ids = selected_surface_ids
        self.overlay_planes = overlay_planes
        self.assignments = assignments
        self.localized_inpaint_undo_stack = undo_stack

    @property
    def generated_assignments(self) -> list[SurfaceTextureAssignment]:
        """Compatibility name for callers that describe assignments as generated."""

        return self.assignments

    def clone(self) -> "SurfaceTextureData":
        return copy.deepcopy(self)

    def strokes_for_frame(self, frame_index: int) -> list[MaskStroke]:
        return list(self.frame_strokes.get(_normalize_frame_index(frame_index), []))

    def set_frame_strokes(
        self,
        frame_index: int,
        strokes: list[MaskStroke],
    ) -> None:
        normalized_index = _normalize_frame_index(frame_index)
        if not _frame_index_is_in_video(normalized_index, self.video_metadata):
            raise ValueError("A masked surface texture frame is outside the video.")
        normalized_strokes = _normalize_frame_strokes(strokes)
        if normalized_strokes:
            if (
                normalized_index not in self.frame_strokes
                and len(self.frame_strokes) >= MAX_MASKED_FRAMES
            ):
                raise ValueError("Surface texture data contains too many masked frames.")
            self.frame_strokes[normalized_index] = normalized_strokes
        else:
            self.frame_strokes.pop(normalized_index, None)

    def strokes_for_surface(self, surface_id: str) -> list[MaskStroke]:
        normalized_id = _normalize_surface_id(surface_id)
        return list(self.texture_mask_strokes.get(normalized_id, []))

    def set_surface_strokes(
        self,
        surface_id: str,
        strokes: list[MaskStroke],
    ) -> None:
        normalized_id = _normalize_surface_id(surface_id)
        normalized_strokes = _normalize_frame_strokes(strokes)
        if normalized_strokes:
            if (
                normalized_id not in self.texture_mask_strokes
                and len(self.texture_mask_strokes) >= MAX_SELECTED_SURFACES
            ):
                raise ValueError("Surface texture data contains too many 3D masks.")
            self.texture_mask_strokes[normalized_id] = normalized_strokes
        else:
            self.texture_mask_strokes.pop(normalized_id, None)

    def set_selection(
        self,
        surface_type: str | None,
        surface_ids: tuple[str, ...] | list[str],
    ) -> None:
        normalized_type, normalized_ids = _normalize_selection(
            surface_type,
            surface_ids,
        )
        self.selected_surface_type = normalized_type
        self.selected_surface_ids = normalized_ids

    def assignments_for_surface(self, surface_id: str) -> list[SurfaceTextureAssignment]:
        normalized_id = _normalize_surface_id(surface_id)
        return [
            assignment
            for assignment in self.assignments
            if normalized_id in assignment.surface_ids
        ]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SURFACE_TEXTURE_SCHEMA_VERSION,
            "video_metadata": (
                None
                if self.video_metadata is None
                else self.video_metadata.to_dict()
            ),
            "current_frame_index": int(self.current_frame_index),
            "frame_strokes": {
                str(frame_index): [stroke.to_dict() for stroke in strokes]
                for frame_index, strokes in sorted(self.frame_strokes.items())
            },
            "texture_mask_strokes": {
                surface_id: [stroke.to_dict() for stroke in strokes]
                for surface_id, strokes in sorted(self.texture_mask_strokes.items())
            },
            "camera_pose": (
                None if self.camera_pose is None else self.camera_pose.to_dict()
            ),
            "selected_surface_type": self.selected_surface_type,
            "selected_surface_ids": list(self.selected_surface_ids),
            "overlay_planes": [plane.to_dict() for plane in self.overlay_planes],
            "assignments": [assignment.to_dict() for assignment in self.assignments],
            "localized_inpaint_undo_stack": [
                snapshot.to_dict()
                for snapshot in self.localized_inpaint_undo_stack
            ],
        }

    @classmethod
    def from_dict(cls, payload: object) -> "SurfaceTextureData":
        """Load state defensively so one malformed optional record is isolated."""

        if not isinstance(payload, dict):
            return cls()

        video_metadata = _load_video_metadata(
            payload.get("video_metadata", payload.get("video"))
        )
        current_frame_index = _load_current_frame_index(
            payload.get("current_frame_index", 0),
            video_metadata,
        )
        frame_strokes = _load_frame_strokes(
            payload.get("frame_strokes", {}),
            video_metadata,
        )
        texture_mask_strokes = _load_texture_mask_strokes(
            payload.get(
                "texture_mask_strokes",
                payload.get("surface_mask_strokes", {}),
            )
        )
        camera_pose = _load_camera_pose(
            payload.get("camera_pose", payload.get("first_person_camera_pose"))
        )
        selected_surface_type, selected_surface_ids = _load_selection(payload)
        overlay_planes = _load_overlay_planes(payload.get("overlay_planes", ()))
        assignments = _load_assignments(
            payload.get("assignments", payload.get("generated_assignments", ()))
        )
        undo_stack = _load_localized_inpaint_undo_stack(
            payload.get("localized_inpaint_undo_stack", ())
        )
        return cls(
            video_metadata=video_metadata,
            current_frame_index=current_frame_index,
            frame_strokes=frame_strokes,
            texture_mask_strokes=texture_mask_strokes,
            camera_pose=camera_pose,
            selected_surface_type=selected_surface_type,
            selected_surface_ids=selected_surface_ids,
            overlay_planes=overlay_planes,
            assignments=assignments,
            localized_inpaint_undo_stack=undo_stack,
        )


# ### Deserialization helpers ###
def _load_video_metadata(raw_video_metadata: object) -> VideoMetadata | None:
    if not isinstance(raw_video_metadata, dict):
        return None
    try:
        return VideoMetadata.from_dict(raw_video_metadata)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def _load_current_frame_index(
    raw_frame_index: object,
    video_metadata: VideoMetadata | None,
) -> int:
    try:
        frame_index = _normalize_frame_index(raw_frame_index)
    except ValueError:
        frame_index = 0
    if video_metadata is None or video_metadata.frame_count <= 0:
        return frame_index
    return min(frame_index, video_metadata.frame_count - 1)


def _load_frame_strokes(
    raw_frame_strokes: object,
    video_metadata: VideoMetadata | None,
) -> dict[int, list[MaskStroke]]:
    if not isinstance(raw_frame_strokes, dict):
        return {}

    loaded: dict[int, list[MaskStroke]] = {}
    for raw_frame_index, raw_strokes in list(raw_frame_strokes.items())[
        :MAX_MASKED_FRAMES
    ]:
        if not isinstance(raw_strokes, list):
            continue
        try:
            frame_index = _normalize_frame_index(raw_frame_index)
        except ValueError:
            continue
        if not _frame_index_is_in_video(frame_index, video_metadata):
            continue
        if len(raw_strokes) > MAX_MASK_STROKES_PER_FRAME:
            continue

        strokes: list[MaskStroke] = []
        for raw_stroke in raw_strokes:
            try:
                strokes.append(MaskStroke.from_dict(raw_stroke))
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
        if strokes:
            loaded[frame_index] = strokes
    return loaded


def _load_texture_mask_strokes(
    raw_texture_mask_strokes: object,
) -> dict[str, list[MaskStroke]]:
    if not isinstance(raw_texture_mask_strokes, dict):
        return {}

    loaded: dict[str, list[MaskStroke]] = {}
    for raw_surface_id, raw_strokes in list(raw_texture_mask_strokes.items())[
        :MAX_SELECTED_SURFACES
    ]:
        if not isinstance(raw_strokes, list):
            continue
        try:
            surface_id = _normalize_surface_id(raw_surface_id)
        except ValueError:
            continue
        if len(raw_strokes) > MAX_MASK_STROKES_PER_FRAME:
            continue
        strokes: list[MaskStroke] = []
        for raw_stroke in raw_strokes:
            try:
                strokes.append(MaskStroke.from_dict(raw_stroke))
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
        if strokes:
            loaded[surface_id] = strokes
    return loaded


def _load_camera_pose(raw_camera_pose: object) -> CameraPose | None:
    if not isinstance(raw_camera_pose, dict):
        return None
    try:
        return CameraPose.from_dict(raw_camera_pose)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def _load_selection(payload: dict[object, object]) -> tuple[str | None, tuple[str, ...]]:
    raw_surface_ids = payload.get("selected_surface_ids", ())
    if not isinstance(raw_surface_ids, list | tuple):
        return None, ()
    raw_surface_type = payload.get("selected_surface_type")
    try:
        return _normalize_selection(
            None if raw_surface_type is None else str(raw_surface_type),
            tuple(str(surface_id) for surface_id in raw_surface_ids),
        )
    except ValueError:
        return None, ()


def _load_overlay_planes(
    raw_overlay_planes: object,
) -> list[SurfaceTextureOverlayPlane]:
    if not isinstance(raw_overlay_planes, list | tuple):
        return []

    planes: list[SurfaceTextureOverlayPlane] = []
    seen_parent_ids: set[str] = set()
    for raw_plane in raw_overlay_planes[:MAX_SURFACE_OVERLAY_PLANES]:
        try:
            plane = SurfaceTextureOverlayPlane.from_dict(raw_plane)
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if plane.parent_surface_id in seen_parent_ids:
            continue
        seen_parent_ids.add(plane.parent_surface_id)
        planes.append(plane)
    return planes


def _load_assignments(raw_assignments: object) -> list[SurfaceTextureAssignment]:
    if not isinstance(raw_assignments, list | tuple):
        return []

    assignments: list[SurfaceTextureAssignment] = []
    seen_assignment_ids: set[str] = set()
    for raw_assignment in raw_assignments[:MAX_SURFACE_TEXTURE_ASSIGNMENTS]:
        try:
            assignment = SurfaceTextureAssignment.from_dict(raw_assignment)
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if assignment.assignment_id in seen_assignment_ids:
            continue
        seen_assignment_ids.add(assignment.assignment_id)
        assignments.append(assignment)
    return assignments


def _load_localized_inpaint_undo_stack(
    raw_snapshots: object,
) -> list[SurfaceTextureInpaintUndoSnapshot]:
    if not isinstance(raw_snapshots, list | tuple):
        return []
    loaded: list[SurfaceTextureInpaintUndoSnapshot] = []
    for raw_snapshot in raw_snapshots[-MAX_LOCALIZED_INPAINT_UNDO_HISTORY:]:
        try:
            loaded.append(SurfaceTextureInpaintUndoSnapshot.from_dict(raw_snapshot))
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    return loaded


# ### Validation helpers ###
def _normalize_frame_index(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("Surface texture frame indices must be integers.")
    try:
        frame_index = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Surface texture frame indices must be integers.") from error
    if frame_index < 0 or frame_index > MAX_VIDEO_FRAME_INDEX:
        raise ValueError("Surface texture frame index is outside the supported range.")
    return frame_index


def _frame_index_is_in_video(
    frame_index: int,
    video_metadata: VideoMetadata | None,
) -> bool:
    if video_metadata is None or video_metadata.frame_count <= 0:
        return True
    return frame_index < video_metadata.frame_count


def _normalize_frame_strokes(raw_strokes: object) -> list[MaskStroke]:
    try:
        strokes = list(raw_strokes)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError("Surface texture frame strokes must contain a list.") from error
    if len(strokes) > MAX_MASK_STROKES_PER_FRAME:
        raise ValueError("A surface texture frame contains too many mask strokes.")
    if not all(isinstance(stroke, MaskStroke) for stroke in strokes):
        raise ValueError("Surface texture frame strokes must be MaskStroke values.")
    return strokes


def _normalize_texture_mask_strokes(
    raw_texture_mask_strokes: object,
) -> dict[str, list[MaskStroke]]:
    if not isinstance(raw_texture_mask_strokes, dict):
        raise ValueError("3D texture masks must contain a surface mapping.")
    if len(raw_texture_mask_strokes) > MAX_SELECTED_SURFACES:
        raise ValueError("Surface texture data contains too many 3D masks.")
    normalized: dict[str, list[MaskStroke]] = {}
    for raw_surface_id, raw_strokes in raw_texture_mask_strokes.items():
        surface_id = _normalize_surface_id(raw_surface_id)
        strokes = _normalize_frame_strokes(raw_strokes)
        if strokes:
            normalized[surface_id] = strokes
    return normalized


def _normalize_selection(
    surface_type: str | None,
    surface_ids: object,
) -> tuple[str | None, tuple[str, ...]]:
    try:
        raw_ids = tuple(surface_ids)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError("Selected surface IDs must contain a sequence.") from error
    if not raw_ids:
        return None, ()
    if surface_type is None:
        surface_type = _infer_surface_type(raw_ids)
    normalized_type = _normalize_surface_type(surface_type)
    normalized_ids = _normalize_surface_ids(
        raw_ids,
        expected_type=normalized_type,
        allow_empty=False,
        maximum_count=MAX_SELECTED_SURFACES,
    )
    return normalized_type, normalized_ids


def _normalize_surface_type(surface_type: object) -> str:
    normalized_type = str(surface_type).strip().lower()
    if normalized_type not in SURFACE_TYPES:
        raise ValueError(f"Unknown surface type: {surface_type!r}.")
    return normalized_type


def _normalize_surface_ids(
    raw_surface_ids: object,
    *,
    expected_type: str,
    allow_empty: bool,
    maximum_count: int,
) -> tuple[str, ...]:
    try:
        surface_ids = tuple(raw_surface_ids)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError("Surface IDs must contain a sequence.") from error
    if not surface_ids and not allow_empty:
        raise ValueError("At least one surface ID is required.")
    if len(surface_ids) > maximum_count:
        raise ValueError("Too many surfaces are selected.")

    normalized_ids: list[str] = []
    seen_ids: set[str] = set()
    for raw_surface_id in surface_ids:
        surface_id = _normalize_surface_id(raw_surface_id)
        if _surface_type_for_id(surface_id) != expected_type:
            raise ValueError("All selected surfaces must have the selected type.")
        if surface_id in seen_ids:
            continue
        seen_ids.add(surface_id)
        normalized_ids.append(surface_id)
    if not normalized_ids and not allow_empty:
        raise ValueError("At least one surface ID is required.")
    return tuple(normalized_ids)


def _normalize_undo_surface_ids(raw_surface_ids: object) -> tuple[str, ...]:
    try:
        values = tuple(raw_surface_ids)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError("Inpaint undo surface IDs must contain a sequence.") from error
    if not values or len(values) > MAX_SELECTED_SURFACES:
        raise ValueError("An inpaint undo snapshot has invalid surface IDs.")
    return tuple(dict.fromkeys(_normalize_surface_id(value) for value in values))


def _normalize_assignment_ids(
    raw_assignment_ids: object,
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    try:
        values = tuple(raw_assignment_ids)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError("Assignment IDs must contain a sequence.") from error
    normalized = tuple(
        dict.fromkeys(
            _normalize_required_text(
                value,
                "Surface texture assignment ID",
                MAX_ASSIGNMENT_ID_LENGTH,
            )
            for value in values
        )
    )
    if not normalized and not allow_empty:
        raise ValueError("At least one replacement assignment ID is required.")
    if len(normalized) > MAX_SURFACE_TEXTURE_ASSIGNMENTS:
        raise ValueError("An inpaint undo snapshot has too many replacement IDs.")
    return normalized


def _normalize_surface_id(surface_id: object) -> str:
    normalized_id = str(surface_id).strip()
    if not normalized_id or len(normalized_id) > MAX_SURFACE_ID_LENGTH:
        raise ValueError("Surface ID is empty or too long.")
    if _SURFACE_ID_PATTERN.fullmatch(normalized_id) is None:
        raise ValueError(f"Invalid fixed-surface ID: {normalized_id!r}.")
    return normalized_id


def _is_overlay_surface_id(surface_id: str) -> bool:
    match = _SURFACE_ID_PATTERN.fullmatch(surface_id)
    return match is not None and match.group("overlay") is not None


def _surface_type_for_id(surface_id: str) -> str:
    match = _SURFACE_ID_PATTERN.fullmatch(surface_id)
    if match is None:
        raise ValueError(f"Invalid fixed-surface ID: {surface_id!r}.")
    return SURFACE_TYPE_WALL if match.group("wall") else str(match.group("plane"))


def _normalize_surface_overlay_offset(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("A surface overlay offset must be a number.")
    try:
        offset = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("A surface overlay offset must be a number.") from error
    if (
        not math.isfinite(offset)
        or abs(offset) <= 1e-8
        or abs(offset) > MAX_SURFACE_OVERLAY_OFFSET_METERS
    ):
        raise ValueError(
            "A surface overlay offset must be finite and no more than 5 cm."
        )
    return offset


def _infer_surface_type(surface_ids: object) -> str:
    try:
        first_surface_id = next(iter(surface_ids))  # type: ignore[arg-type]
    except (StopIteration, TypeError) as error:
        raise ValueError("A surface type cannot be inferred without surfaces.") from error
    return _surface_type_for_id(_normalize_surface_id(first_surface_id))


def _normalize_reference_frame_indices(raw_indices: object) -> tuple[int, ...]:
    try:
        indices = tuple(raw_indices)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError("Reference frame indices must contain a sequence.") from error
    if len(indices) > MAX_REFERENCE_FRAMES_PER_ASSIGNMENT:
        raise ValueError("A surface texture assignment has too many reference frames.")
    normalized_indices: list[int] = []
    seen_indices: set[int] = set()
    for raw_index in indices:
        frame_index = _normalize_frame_index(raw_index)
        if frame_index in seen_indices:
            continue
        seen_indices.add(frame_index)
        normalized_indices.append(frame_index)
    return tuple(normalized_indices)


def _normalize_safe_relative_asset_path(raw_path: object) -> str:
    path_text = _normalize_required_text(
        raw_path,
        "Surface texture asset path",
        MAX_ASSET_PATH_LENGTH,
    )
    if "\x00" in path_text:
        raise ValueError("Surface texture asset path contains an invalid character.")

    windows_path = PureWindowsPath(path_text)
    normalized_path = PurePosixPath(path_text.replace("\\", "/"))
    if windows_path.drive or windows_path.root or normalized_path.is_absolute():
        raise ValueError("Surface texture asset path must be relative.")
    if any(part in ("", ".", "..") for part in normalized_path.parts):
        raise ValueError("Surface texture asset path contains an unsafe segment.")
    normalized_text = normalized_path.as_posix()
    if normalized_text in ("", "."):
        raise ValueError("Surface texture asset path cannot be empty.")
    return normalized_text


def _normalize_combined_area(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("Combined surface area must be a number.")
    try:
        area = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Combined surface area must be a number.") from error
    if not math.isfinite(area) or not 0.0 <= area <= MAX_COMBINED_AREA_M2:
        raise ValueError("Combined surface area is outside the supported range.")
    return area


def _normalize_texture_dimensions(
    raw_width: object,
    raw_height: object,
) -> tuple[int | None, int | None]:
    if raw_width is None and raw_height is None:
        return None, None
    if raw_width is None or raw_height is None:
        raise ValueError("Surface texture width and height must be stored together.")
    width = _normalize_texture_dimension(raw_width)
    height = _normalize_texture_dimension(raw_height)
    return width, height


def _normalize_texture_dimension(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("Surface texture dimensions must be integers.")
    if not 1 <= value <= MAX_TEXTURE_DIMENSION_PIXELS:
        raise ValueError("Surface texture dimension is outside the supported range.")
    return value


def _normalize_surface_texture_resolution(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("A surface texture resolution must be an integer.")
    resolution = int(value)
    if resolution not in SURFACE_TEXTURE_RESOLUTIONS:
        supported = ", ".join(str(item) for item in SURFACE_TEXTURE_RESOLUTIONS)
        raise ValueError(
            f"A surface texture resolution must be one of: {supported}."
        )
    return resolution


def _normalize_surface_texture_variants(
    raw_variants: object,
) -> tuple[SurfaceTextureVariant, ...]:
    try:
        variants = tuple(raw_variants)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError("Surface texture variants must contain a sequence.") from error
    if not variants:
        return ()
    if not all(isinstance(variant, SurfaceTextureVariant) for variant in variants):
        raise ValueError(
            "Surface texture variants must be SurfaceTextureVariant values."
        )
    resolutions = tuple(variant.resolution for variant in variants)
    if len(resolutions) != len(set(resolutions)):
        raise ValueError("Surface texture variant resolutions must be unique.")
    if set(resolutions) != set(SURFACE_TEXTURE_RESOLUTIONS):
        raise ValueError(
            "A generated surface texture needs 512, 1024 and 2048 variants."
        )
    asset_paths = tuple(variant.asset_path for variant in variants)
    if len(asset_paths) != len(set(asset_paths)):
        raise ValueError("Surface texture variant asset paths must be unique.")
    return tuple(sorted(variants, key=lambda variant: variant.resolution))


def _normalize_selected_texture_resolution(
    raw_resolution: object,
    variants: tuple[SurfaceTextureVariant, ...],
    active_asset_path: str,
) -> int | None:
    if not variants:
        if raw_resolution is not None:
            raise ValueError(
                "A selected surface texture resolution requires exact variants."
            )
        return None
    if raw_resolution is None:
        matching_variants = tuple(
            variant
            for variant in variants
            if variant.asset_path == active_asset_path
        )
        if len(matching_variants) != 1:
            raise ValueError(
                "The active surface texture asset has no exact variant."
            )
        return matching_variants[0].resolution
    resolution = _normalize_surface_texture_resolution(raw_resolution)
    if not any(variant.resolution == resolution for variant in variants):
        raise ValueError("The selected surface texture variant is unavailable.")
    return resolution


def _normalize_required_text(value: object, label: str, maximum_length: int) -> str:
    normalized_text = str(value).strip()
    if not normalized_text:
        raise ValueError(f"{label} cannot be empty.")
    if len(normalized_text) > maximum_length:
        raise ValueError(f"{label} is too long.")
    return normalized_text


def _normalize_optional_text(
    value: object,
    label: str,
    maximum_length: int,
) -> str | None:
    if value is None:
        return None
    normalized_text = str(value).strip()
    if not normalized_text:
        return None
    if len(normalized_text) > maximum_length:
        raise ValueError(f"{label} is too long.")
    return normalized_text
