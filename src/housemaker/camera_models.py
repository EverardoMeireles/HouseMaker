# ### Imports ###
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any


# ### Constants ###
DEFAULT_FIRST_PERSON_LIGHT_INTENSITY = 1.0
MIN_FIRST_PERSON_LIGHT_INTENSITY = 0.0
MAX_FIRST_PERSON_LIGHT_INTENSITY = 2.0


# ### Camera models ###
@dataclass(frozen=True)
class CameraPose:
    """Immutable world-space Z-up camera pose used by the Canvas marker."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw_degrees: float = 0.0
    pitch_degrees: float = 0.0
    roll_degrees: float = 0.0
    fov_degrees: float = 70.0

    def __post_init__(self) -> None:
        values = (
            self.x,
            self.y,
            self.z,
            self.yaw_degrees,
            self.pitch_degrees,
            self.roll_degrees,
            self.fov_degrees,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Camera pose values must be finite.")
        if not 1.0 <= float(self.fov_degrees) < 179.0:
            raise ValueError("Camera field of view must be in [1, 179) degrees.")

    def to_dict(self) -> dict[str, float]:
        return {
            "x": float(self.x),
            "y": float(self.y),
            "z": float(self.z),
            "yaw_degrees": float(self.yaw_degrees),
            "pitch_degrees": float(self.pitch_degrees),
            "roll_degrees": float(self.roll_degrees),
            "fov_degrees": float(self.fov_degrees),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CameraPose":
        return cls(
            x=float(payload["x"]),
            y=float(payload["y"]),
            z=float(payload["z"]),
            yaw_degrees=float(payload["yaw_degrees"]),
            pitch_degrees=float(payload["pitch_degrees"]),
            roll_degrees=float(payload["roll_degrees"]),
            fov_degrees=float(payload["fov_degrees"]),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "CameraPose":
        decoded = json.loads(payload)
        if not isinstance(decoded, dict):
            raise ValueError("Camera pose JSON must contain an object.")
        return cls.from_dict(decoded)


@dataclass(frozen=True)
class InitialFirstPersonCamera:
    """The single project camera placed on one blueprint level."""

    level_index: int
    pose: CameraPose
    light_intensity: float = DEFAULT_FIRST_PERSON_LIGHT_INTENSITY

    def __post_init__(self) -> None:
        if isinstance(self.level_index, bool) or not isinstance(
            self.level_index,
            int,
        ):
            raise ValueError("Initial camera level index must be an integer.")
        if self.level_index < 0:
            raise ValueError("Initial camera level index cannot be negative.")
        if not isinstance(self.pose, CameraPose):
            raise ValueError("Initial camera pose must be a CameraPose.")
        object.__setattr__(
            self,
            "light_intensity",
            normalize_first_person_light_intensity(self.light_intensity),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "level_index": int(self.level_index),
            "pose": self.pose.to_dict(),
            "light_intensity": float(self.light_intensity),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "InitialFirstPersonCamera":
        if not isinstance(payload, dict):
            raise ValueError("Initial camera JSON must contain an object.")
        pose_payload = payload.get("pose")
        if not isinstance(pose_payload, dict):
            raise ValueError("Initial camera pose must contain an object.")
        return cls(
            level_index=payload.get("level_index"),
            pose=CameraPose.from_dict(pose_payload),
            light_intensity=payload.get(
                "light_intensity",
                DEFAULT_FIRST_PERSON_LIGHT_INTENSITY,
            ),
        )


# ### Validation helpers ###
def normalize_first_person_light_intensity(value: object) -> float:
    """Validate and normalize the camera-mounted light intensity."""

    if isinstance(value, bool):
        raise ValueError("First person light intensity must be a number.")
    try:
        intensity = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("First person light intensity must be a number.") from error
    if not math.isfinite(intensity):
        raise ValueError("First person light intensity must be finite.")
    if not (
        MIN_FIRST_PERSON_LIGHT_INTENSITY
        <= intensity
        <= MAX_FIRST_PERSON_LIGHT_INTENSITY
    ):
        raise ValueError(
            "First person light intensity must be between "
            f"{MIN_FIRST_PERSON_LIGHT_INTENSITY:g} and "
            f"{MAX_FIRST_PERSON_LIGHT_INTENSITY:g}."
        )
    return intensity
