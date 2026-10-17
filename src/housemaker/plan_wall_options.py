# ### Imports ###
from __future__ import annotations

import math
from dataclasses import asdict, dataclass


# ### Public data models ###
class PlanWallDetectionError(ValueError):
    """A safe, user-displayable plan-wall detection failure."""


@dataclass(frozen=True, slots=True)
class PlanWallDetectionOptions:
    """Persistable reconstruction controls expressed in native image pixels."""

    minimum_wall_separation_pixels: float = 6.0
    maximum_wall_separation_pixels: float = 48.0
    parallel_angle_tolerance_degrees: float = 7.5
    minimum_parallel_overlap_ratio: float = 0.55
    maximum_gap_bridge_pixels: float = 36.0
    endpoint_snap_distance_pixels: float = 8.0
    maximum_vertex_distance_pixels: float = 5.0
    minimum_wall_length_pixels: float = 20.0
    confidence_threshold: float = 0.45
    endpoint_alignment_tolerance_pixels: float = 16.0
    endpoint_alignment_search_distance_pixels: float = 1024.0

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in asdict(self).values()):
            raise PlanWallDetectionError("Wall reconstruction options must be finite.")
        if self.minimum_wall_separation_pixels <= 0.0:
            raise PlanWallDetectionError(
                "Minimum parallel separation must be positive."
            )
        if self.maximum_wall_separation_pixels < self.minimum_wall_separation_pixels:
            raise PlanWallDetectionError(
                "Maximum parallel separation cannot be smaller than the minimum."
            )
        if not 0.0 < self.parallel_angle_tolerance_degrees < 45.0:
            raise PlanWallDetectionError(
                "Parallel angle tolerance must be between 0 and 45 degrees."
            )
        if not 0.0 < self.minimum_parallel_overlap_ratio <= 1.0:
            raise PlanWallDetectionError(
                "Minimum overlap ratio must be greater than 0 and at most 1."
            )
        if self.maximum_gap_bridge_pixels < 0.0:
            raise PlanWallDetectionError("Maximum collinear gap cannot be negative.")
        if self.endpoint_snap_distance_pixels < 0.0:
            raise PlanWallDetectionError("Endpoint snap distance cannot be negative.")
        if self.endpoint_alignment_tolerance_pixels < 0.0:
            raise PlanWallDetectionError(
                "Endpoint alignment tolerance cannot be negative."
            )
        if self.endpoint_alignment_search_distance_pixels < 0.0:
            raise PlanWallDetectionError(
                "Endpoint alignment search distance cannot be negative."
            )
        if self.maximum_vertex_distance_pixels < 0.0:
            raise PlanWallDetectionError("Maximum vertex distance cannot be negative.")
        if self.minimum_wall_length_pixels <= 0.0:
            raise PlanWallDetectionError("Minimum wall length must be positive.")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise PlanWallDetectionError(
                "Confidence threshold must be between 0 and 1."
            )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw_options: object) -> PlanWallDetectionOptions:
        """Restore old or malformed projects with valid reconstruction defaults."""

        defaults = cls()
        if not isinstance(raw_options, dict):
            return defaults
        values = defaults.to_dict()
        try:
            for name in values:
                if name in raw_options:
                    raw_value = raw_options[name]
                    if isinstance(raw_value, bool):
                        return defaults
                    values[name] = float(raw_value)
            return cls(**values)
        except (TypeError, ValueError, OverflowError):
            return defaults
