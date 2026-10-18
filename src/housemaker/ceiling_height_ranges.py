# ### Imports ###
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


# ### Public data ###
@dataclass(frozen=True)
class CeilingHeightVideoRange:
    """Inclusive video frames explicitly assigned to one project level."""

    level_index: int
    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        _require_integer(self.level_index, "Level index")
        _require_integer(self.start_frame, "Start frame")
        _require_integer(self.end_frame, "End frame")
        if self.start_frame < 0 or self.end_frame < 0:
            raise ValueError("Marked frame positions cannot be negative.")
        if self.end_frame < self.start_frame:
            raise ValueError("A marked range cannot end before it begins.")


# ### Public sampling API ###
def sample_marked_video_frames(
    ranges: Iterable[CeilingHeightVideoRange],
    interval: int,
    frame_count: int,
) -> tuple[tuple[int, int], ...]:
    """Sample marked ranges only, retaining both ends and their level IDs.

    Overlapping ranges for the same level are allowed; duplicate samples are
    removed. Any overlap between different levels is ambiguous and rejected,
    even when the requested sampling interval would skip the overlap.
    """

    _require_integer(interval, "Frame analysis interval")
    _require_integer(frame_count, "Video frame count")
    if interval < 1:
        raise ValueError("The frame analysis interval must be at least one.")
    if frame_count < 0:
        raise ValueError("Video frame count cannot be negative.")

    normalized = tuple(ranges)
    for item in normalized:
        if not isinstance(item, CeilingHeightVideoRange):
            raise TypeError("Each marked area must be a ceiling-height video range.")
        if item.end_frame >= frame_count:
            raise ValueError("A marked range extends beyond the loaded video.")

    ordered = sorted(normalized, key=lambda item: item.start_frame)
    active: list[CeilingHeightVideoRange] = []
    samples: dict[int, int] = {}
    for item in ordered:
        active = [other for other in active if other.end_frame >= item.start_frame]
        if any(other.level_index != item.level_index for other in active):
            raise ValueError(
                "Marked areas for different levels cannot overlap. "
                "Move their boundaries so every frame belongs to one level."
            )
        active.append(item)
        for frame_index in range(item.start_frame, item.end_frame + 1, interval):
            samples[frame_index] = item.level_index
        samples[item.end_frame] = item.level_index
    return tuple(sorted(samples.items()))


# ### Validation helpers ###
def _require_integer(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer.")
