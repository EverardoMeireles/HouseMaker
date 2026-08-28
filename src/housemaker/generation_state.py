# ### Imports ###
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

from housemaker.video_source import VideoMetadata


# ### Constants ###
MASK_MODE_PAINT = "paint"
MASK_MODE_ERASE = "erase"
MASK_MODES = frozenset({MASK_MODE_PAINT, MASK_MODE_ERASE})
MAX_MASK_STROKES_PER_FRAME = 10_000
MAX_MASK_POINTS_PER_STROKE = 100_000
GENERATION_PIPELINE_SCHEMA_VERSION = 1
MESHY_GENERATION_PROVIDER = "meshy"
GENERATION_PROVIDERS = frozenset({MESHY_GENERATION_PROVIDER})


# ### Mask models ###
@dataclass(frozen=True)
class MaskPoint:
    """One normalized source-frame position in the inclusive range [0, 1]."""

    x: float
    y: float

    def __post_init__(self) -> None:
        if not _is_finite_number(self.x) or not _is_finite_number(self.y):
            raise ValueError("Mask point coordinates must be finite numbers.")
        if not 0.0 <= float(self.x) <= 1.0:
            raise ValueError("Mask point X must be in [0, 1].")
        if not 0.0 <= float(self.y) <= 1.0:
            raise ValueError("Mask point Y must be in [0, 1].")

    def to_dict(self) -> dict[str, float]:
        return {"x": float(self.x), "y": float(self.y)}

    @classmethod
    def from_dict(cls, payload: object) -> "MaskPoint":
        if not isinstance(payload, dict):
            raise ValueError("Mask point data must contain an object.")
        return cls(x=float(payload["x"]), y=float(payload["y"]))


@dataclass(frozen=True)
class MaskStroke:
    """A paint or erase operation stored independently from display resolution.

    ``is_fill`` records one right-click paint-bucket action.  Its single point
    is the normalized fill seed; the enclosed region is replayed against the
    preceding mask operations when the mask is rebuilt.
    """

    mode: str
    radius_normalized: float
    points: tuple[MaskPoint, ...]
    is_fill: bool = False

    def __post_init__(self) -> None:
        if self.mode not in MASK_MODES:
            raise ValueError(f"Unknown mask stroke mode: {self.mode!r}.")
        if not _is_finite_number(self.radius_normalized):
            raise ValueError("Mask stroke radius must be finite.")
        if not 0.0 < float(self.radius_normalized) <= 1.0:
            raise ValueError("Mask stroke radius must be in (0, 1].")
        if not self.points:
            raise ValueError("Mask strokes must contain at least one point.")
        if len(self.points) > MAX_MASK_POINTS_PER_STROKE:
            raise ValueError("Mask stroke contains too many points.")
        if not all(isinstance(point, MaskPoint) for point in self.points):
            raise ValueError("Mask stroke points must be MaskPoint values.")
        if not isinstance(self.is_fill, bool):
            raise ValueError("Mask fill flag must be a boolean.")
        if self.is_fill and len(self.points) != 1:
            raise ValueError("Mask fills must contain exactly one seed point.")

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "radius_normalized": float(self.radius_normalized),
            "points": [point.to_dict() for point in self.points],
            "is_fill": self.is_fill,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "MaskStroke":
        if not isinstance(payload, dict):
            raise ValueError("Mask stroke data must contain an object.")
        raw_points = payload.get("points")
        if not isinstance(raw_points, list):
            raise ValueError("Mask stroke points must contain a list.")
        return cls(
            mode=str(payload["mode"]),
            radius_normalized=float(payload["radius_normalized"]),
            points=tuple(MaskPoint.from_dict(point) for point in raw_points),
            is_fill=payload.get("is_fill", False),
        )


# ### Generated-object models ###
@dataclass(frozen=True)
class GeneratedObjectPlacement:
    """One generated object's floor-relative location on a Canvas level."""

    level_index: int
    image_x: float
    image_y: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.level_index, bool)
            or not isinstance(self.level_index, int)
            or self.level_index < 0
        ):
            raise ValueError(
                "Generated-object placement level index must be a "
                "non-negative integer."
            )
        if not _is_strict_finite_number(self.image_x):
            raise ValueError(
                "Generated-object placement X coordinate must be finite."
            )
        if not _is_strict_finite_number(self.image_y):
            raise ValueError(
                "Generated-object placement Y coordinate must be finite."
            )
        object.__setattr__(self, "image_x", float(self.image_x))
        object.__setattr__(self, "image_y", float(self.image_y))

    def to_dict(self) -> dict[str, int | float]:
        return {
            "level_index": self.level_index,
            "image_x": self.image_x,
            "image_y": self.image_y,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "GeneratedObjectPlacement":
        if not isinstance(payload, dict):
            raise ValueError(
                "Generated-object placement data must contain an object."
            )
        return cls(
            level_index=payload.get("level_index"),
            image_x=payload.get("image_x"),
            image_y=payload.get("image_y"),
        )


@dataclass(frozen=True)
class GeneratedObjectRecord:
    """Serializable provenance for a generated object.

    Meshy meshes use a validated local GLB asset path instead of embedding bytes
    in project JSON. ``pipeline`` is retained as an empty compatibility field.
    """

    object_id: str
    frame_index: int
    object_name: str
    pipeline: dict[str, Any]
    provider: str = MESHY_GENERATION_PROVIDER
    provider_task_id: str | None = None
    asset_path: str | None = None
    placement: GeneratedObjectPlacement | None = None

    def __post_init__(self) -> None:
        if not str(self.object_id).strip():
            raise ValueError("Generated object ID cannot be empty.")
        if int(self.frame_index) < 0:
            raise ValueError("Generated object frame index cannot be negative.")
        if not str(self.object_name).strip():
            raise ValueError("Generated object name cannot be empty.")
        if not isinstance(self.pipeline, dict):
            raise ValueError("Generated object pipeline must contain an object.")
        if self.provider not in GENERATION_PROVIDERS:
            raise ValueError(f"Unknown generation provider: {self.provider!r}.")
        if self.provider == MESHY_GENERATION_PROVIDER:
            if not str(self.provider_task_id or "").strip():
                raise ValueError("Meshy generated objects require a task ID.")
            if not str(self.asset_path or "").strip():
                raise ValueError("Meshy generated objects require an asset path.")
        if self.placement is not None and not isinstance(
            self.placement,
            GeneratedObjectPlacement,
        ):
            raise ValueError(
                "Generated-object placement must be a "
                "GeneratedObjectPlacement value."
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "object_id": self.object_id,
            "frame_index": int(self.frame_index),
            "object_name": self.object_name,
            "pipeline": copy.deepcopy(self.pipeline),
            "provider": self.provider,
            "provider_task_id": self.provider_task_id,
            "asset_path": self.asset_path,
            "placement": (
                None if self.placement is None else self.placement.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, payload: object) -> "GeneratedObjectRecord":
        if not isinstance(payload, dict):
            raise ValueError("Generated object data must contain an object.")
        raw_pipeline = payload.get("pipeline")
        if not isinstance(raw_pipeline, dict):
            raise ValueError("Generated object pipeline must contain an object.")
        return cls(
            object_id=str(payload["object_id"]),
            frame_index=int(payload["frame_index"]),
            object_name=str(payload["object_name"]),
            pipeline=copy.deepcopy(raw_pipeline),
            provider=str(payload.get("provider", MESHY_GENERATION_PROVIDER)),
            provider_task_id=(
                None
                if payload.get("provider_task_id") is None
                else str(payload.get("provider_task_id"))
            ),
            asset_path=(
                None
                if payload.get("asset_path") is None
                else str(payload.get("asset_path"))
            ),
            placement=(
                None
                if payload.get("placement") is None
                else GeneratedObjectPlacement.from_dict(
                    payload.get("placement")
                )
            ),
        )


# ### Workspace state ###
@dataclass
class GenerationData:
    """Project-safe Generation-tab state without credentials or mesh bytes."""

    video_metadata: VideoMetadata | None = None
    current_frame_index: int = 0
    frame_strokes: dict[int, list[MaskStroke]] = field(default_factory=dict)
    generated_objects: list[GeneratedObjectRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        if int(self.current_frame_index) < 0:
            raise ValueError("Current video frame cannot be negative.")
        normalized_strokes: dict[int, list[MaskStroke]] = {}
        for raw_frame_index, strokes in self.frame_strokes.items():
            frame_index = int(raw_frame_index)
            if frame_index < 0:
                raise ValueError("Mask frame index cannot be negative.")
            normalized = list(strokes)
            if len(normalized) > MAX_MASK_STROKES_PER_FRAME:
                raise ValueError("A video frame contains too many mask strokes.")
            if not all(isinstance(stroke, MaskStroke) for stroke in normalized):
                raise ValueError("Frame strokes must be MaskStroke values.")
            if normalized:
                normalized_strokes[frame_index] = normalized
        self.frame_strokes = normalized_strokes
        if not all(
            isinstance(record, GeneratedObjectRecord)
            for record in self.generated_objects
        ):
            raise ValueError(
                "Generated objects must be GeneratedObjectRecord values."
            )

    def clone(self) -> "GenerationData":
        return copy.deepcopy(self)

    def strokes_for_frame(self, frame_index: int) -> list[MaskStroke]:
        return list(self.frame_strokes.get(int(frame_index), []))

    def set_frame_strokes(
        self,
        frame_index: int,
        strokes: list[MaskStroke],
    ) -> None:
        normalized_index = int(frame_index)
        if normalized_index < 0:
            raise ValueError("Mask frame index cannot be negative.")
        normalized_strokes = list(strokes)
        if len(normalized_strokes) > MAX_MASK_STROKES_PER_FRAME:
            raise ValueError("A video frame contains too many mask strokes.")
        if not all(
            isinstance(stroke, MaskStroke) for stroke in normalized_strokes
        ):
            raise ValueError("Frame strokes must be MaskStroke values.")
        if normalized_strokes:
            self.frame_strokes[normalized_index] = normalized_strokes
        else:
            self.frame_strokes.pop(normalized_index, None)

    def to_dict(self) -> dict[str, object]:
        return {
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
            "generated_objects": [
                record.to_dict() for record in self.generated_objects
            ],
        }

    @classmethod
    def from_dict(cls, payload: object) -> "GenerationData":
        if not isinstance(payload, dict):
            raise ValueError("Generation data must contain an object.")
        raw_video = payload.get("video_metadata", payload.get("video"))
        raw_frame_strokes = payload.get("frame_strokes", {})
        if not isinstance(raw_frame_strokes, dict):
            raise ValueError("Generation frame strokes must contain an object.")
        raw_generated_objects = payload.get("generated_objects", [])
        if not isinstance(raw_generated_objects, list):
            raise ValueError("Generated objects must contain a list.")
        return cls(
            video_metadata=(
                None
                if raw_video is None
                else VideoMetadata.from_dict(raw_video)
            ),
            current_frame_index=int(payload.get("current_frame_index", 0)),
            frame_strokes={
                int(frame_index): [
                    MaskStroke.from_dict(raw_stroke)
                    for raw_stroke in raw_strokes
                ]
                for frame_index, raw_strokes in raw_frame_strokes.items()
                if isinstance(raw_strokes, list)
            },
            generated_objects=_load_meshy_generated_objects(raw_generated_objects),
        )


# ### Validation helpers ###
def _load_meshy_generated_objects(
    raw_generated_objects: list[object],
) -> list[GeneratedObjectRecord]:
    """Load Meshy records and silently retire legacy procedural records."""

    records: list[GeneratedObjectRecord] = []
    for raw_record in raw_generated_objects:
        if not isinstance(raw_record, dict):
            raise ValueError("Generated object data must contain an object.")
        provider = str(raw_record.get("provider", "procedural"))
        if provider == "procedural":
            continue
        records.append(GeneratedObjectRecord.from_dict(raw_record))
    return records


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _is_strict_finite_number(value: object) -> bool:
    """Reject coercible text and booleans while accepting JSON numbers."""

    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
    )
