# ### Imports ###
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, replace

# ### Constants ###
MIN_GLOBAL_MULTIPLIER = 0.01
MAX_GLOBAL_MULTIPLIER = 100.0
GLOBAL_SCALE_AXES = ("x", "y", "z")


# ### Global scale model ###
@dataclass(frozen=True)
class GlobalScale:
    """Project-wide scale relative to the unchanged authored coordinates.

    X and Y follow the plan axes; Z is vertical. Keeping the factors separate
    from geometry makes changing them reversible without accumulating error.
    """

    x: float = 1.0
    y: float = 1.0
    z: float = 1.0
    linked: bool = True

    def __post_init__(self) -> None:
        for axis in GLOBAL_SCALE_AXES:
            object.__setattr__(self, axis, _scale_factor(getattr(self, axis), axis))
        if not isinstance(self.linked, bool):
            raise TypeError("Global multiplier linked state must be a boolean.")

    @property
    def factors(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z

    @property
    def matrix(self) -> tuple[tuple[float, ...], ...]:
        return _scale_matrix(self.factors)

    @property
    def inverse_matrix(self) -> tuple[tuple[float, ...], ...]:
        return _scale_matrix(tuple(1.0 / value for value in self.factors))

    def with_axis(self, axis: str, value: float) -> GlobalScale:
        if axis not in GLOBAL_SCALE_AXES:
            raise ValueError("Global multiplier axis must be x, y, or z.")
        value = _scale_factor(value, axis)
        if not self.linked:
            return replace(self, **{axis: value})
        ratio = value / getattr(self, axis)
        factors = {name: getattr(self, name) * ratio for name in GLOBAL_SCALE_AXES}
        factors[axis] = value
        # Construction validates every axis before returning any changed state.
        return replace(self, **factors)

    def with_linked(self, linked: bool) -> GlobalScale:
        return replace(self, linked=linked)

    def apply_point(self, point: Iterable[float]) -> tuple[float, float, float]:
        x, y, z = _point_coordinates(point)
        return x * self.x, y * self.y, z * self.z

    def inverse_point(self, point: Iterable[float]) -> tuple[float, float, float]:
        x, y, z = _point_coordinates(point)
        return x / self.x, y / self.y, z / self.z

    def to_dict(self) -> dict[str, object]:
        return {"x": self.x, "y": self.y, "z": self.z, "linked": self.linked}

    @classmethod
    def from_dict(cls, payload: object) -> GlobalScale:
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise TypeError("Global multiplier JSON must contain an object.")
        return cls(
            x=payload.get("x", 1.0),
            y=payload.get("y", 1.0),
            z=payload.get("z", 1.0),
            linked=payload.get("linked", True),
        )


# ### Validation and coordinate helpers ###
def _scale_factor(raw_value: object, axis: str) -> float:
    if isinstance(raw_value, bool):
        raise TypeError(f"Global {axis.upper()} multiplier must be a number.")
    try:
        value = float(raw_value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"Global {axis.upper()} multiplier must be a number."
        ) from error
    if (
        not math.isfinite(value)
        or not MIN_GLOBAL_MULTIPLIER <= value <= MAX_GLOBAL_MULTIPLIER
    ):
        raise ValueError(
            f"Global {axis.upper()} multiplier must be between "
            f"{MIN_GLOBAL_MULTIPLIER:g} and {MAX_GLOBAL_MULTIPLIER:g}."
        )
    return value


def _point_coordinates(point: Iterable[float]) -> tuple[float, float, float]:
    try:
        x, y, z = (float(value) for value in point)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Global scale requires three finite coordinates.") from error
    if not all(math.isfinite(value) for value in (x, y, z)):
        raise ValueError("Global scale requires three finite coordinates.")
    return x, y, z


def _scale_matrix(factors: tuple[float, float, float]) -> tuple[tuple[float, ...], ...]:
    x, y, z = factors
    return (
        (x, 0.0, 0.0, 0.0),
        (0.0, y, 0.0, 0.0),
        (0.0, 0.0, z, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
