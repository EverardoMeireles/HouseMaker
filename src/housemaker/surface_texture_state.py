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
SURFACE_TEXTURE_SCHEMA_VERSION = 2
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

_SURFACE_ID_PATTERN = re.compile(
    r"^level:(?P<level_index>0|[1-9]\d*)/"
    r"(?:room:(?P<room_identity>0|[1-9]\d*)/)?"
    r"(?:(?P<wall>wall):(?P<wall_key>[1-9]\d*:[1-9]\d*)|"
    r"(?P<plane>floor|ceiling))$"
)


# ### Generated-texture models ###
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

        self.current_frame_index = current_frame_index
        self.frame_strokes = normalized_frame_strokes
        self.texture_mask_strokes = normalized_texture_mask_strokes
        self.selected_surface_type = selected_surface_type
        self.selected_surface_ids = selected_surface_ids
        self.assignments = assignments

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
            "assignments": [assignment.to_dict() for assignment in self.assignments],
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
        assignments = _load_assignments(
            payload.get("assignments", payload.get("generated_assignments", ()))
        )
        return cls(
            video_metadata=video_metadata,
            current_frame_index=current_frame_index,
            frame_strokes=frame_strokes,
            texture_mask_strokes=texture_mask_strokes,
            camera_pose=camera_pose,
            selected_surface_type=selected_surface_type,
            selected_surface_ids=selected_surface_ids,
            assignments=assignments,
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


def _normalize_surface_id(surface_id: object) -> str:
    normalized_id = str(surface_id).strip()
    if not normalized_id or len(normalized_id) > MAX_SURFACE_ID_LENGTH:
        raise ValueError("Surface ID is empty or too long.")
    if _SURFACE_ID_PATTERN.fullmatch(normalized_id) is None:
        raise ValueError(f"Invalid fixed-surface ID: {normalized_id!r}.")
    return normalized_id


def _surface_type_for_id(surface_id: str) -> str:
    match = _SURFACE_ID_PATTERN.fullmatch(surface_id)
    if match is None:
        raise ValueError(f"Invalid fixed-surface ID: {surface_id!r}.")
    return SURFACE_TYPE_WALL if match.group("wall") else str(match.group("plane"))


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
