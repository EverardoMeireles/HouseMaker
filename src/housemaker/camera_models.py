# ### Imports ###
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any


# ### Camera models ###
@dataclass(frozen=True)
class CameraPose:
    """Immutable world-space Z-up camera pose used by 3D viewers."""

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
