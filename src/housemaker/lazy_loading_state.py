"""Project-persisted camera start and optional material compression choices."""

# ### Imports ###
from __future__ import annotations

import math
from dataclasses import dataclass


# ### Authored camera start ###
@dataclass(frozen=True)
class CameraStartPoint:
    """Floor-relative, image-anchored point which follows level transforms."""

    level_index: int
    image_x: float
    image_y: float
    height_offset_meters: float = 0.0
    yaw_degrees: float = 0.0
    pitch_degrees: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.level_index, bool) or not isinstance(self.level_index, int):
            raise TypeError("The camera starting level must be an integer.")
        for name in (
            "image_x",
            "image_y",
            "height_offset_meters",
            "yaw_degrees",
            "pitch_degrees",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise TypeError("Camera starting-point coordinates must be numbers.")
            if not math.isfinite(value):
                raise ValueError("Camera starting-point coordinates must be finite.")
            object.__setattr__(self, name, float(value))
        if not -89.9 <= self.pitch_degrees <= 89.9:
            raise ValueError(
                "Camera starting pitch must be between -89.9 and 89.9 degrees."
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "level_index": self.level_index,
            "image_x": self.image_x,
            "image_y": self.image_y,
            "height_offset_meters": self.height_offset_meters,
            "yaw_degrees": self.yaw_degrees,
            "pitch_degrees": self.pitch_degrees,
        }

    @classmethod
    def from_dict(cls, payload) -> CameraStartPoint:
        if not isinstance(payload, dict):
            raise TypeError("The camera starting point must be a JSON object.")
        return cls(
            payload["level_index"],
            payload["image_x"],
            payload["image_y"],
            payload.get("height_offset_meters", 0.0),
            payload.get("yaw_degrees", 0.0),
            payload.get("pitch_degrees", 0.0),
        )


# ### Project export preferences ###
@dataclass(frozen=True)
class RuntimeLoadingData:
    start_point: CameraStartPoint | None = None
    use_ktx2_basis_compression: bool = False

    def __post_init__(self) -> None:
        if self.start_point is not None and not isinstance(
            self.start_point, CameraStartPoint
        ):
            raise TypeError("The FPS starting point must be validated camera data.")
        if not isinstance(self.use_ktx2_basis_compression, bool):
            raise TypeError("Use KTX2/Basis compression must be a boolean.")

    def to_dict(self) -> dict[str, object]:
        return {
            "start_point": None
            if self.start_point is None
            else self.start_point.to_dict(),
            "use_ktx2_basis_compression": self.use_ktx2_basis_compression,
        }

    @classmethod
    def from_dict(cls, payload) -> RuntimeLoadingData:
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise TypeError("Runtime loading settings must be a JSON object.")
        point = payload.get("start_point")
        return cls(
            None if point is None else CameraStartPoint.from_dict(point),
            payload.get("use_ktx2_basis_compression", False),
        )
