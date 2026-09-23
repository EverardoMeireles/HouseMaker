# ### Imports ###
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# ### Detection constants ###
_INK_THRESHOLD = 127
_MIN_SUPPORT_FRACTION = 0.008
_MIN_GAP_FRACTION = 0.004
_MAX_GAP_FRACTION = 0.18
_MAX_WALL_SEPARATION_FRACTION = 0.05
_MIN_GAP_TO_WALL_SEPARATION_RATIO = 1.15
_MIN_INTERVAL_OVERLAP_RATIO = 0.75


# ### Internal data models ###
@dataclass(frozen=True)
class _DetectionLimits:
    minimum_support: int
    minimum_gap: int
    maximum_gap: int
    maximum_wall_separation: int
    cluster_endpoint_tolerance: int


@dataclass
class _MutableGapCluster:
    normals: list[int] = field(default_factory=list)
    starts: list[int] = field(default_factory=list)
    ends: list[int] = field(default_factory=list)

    @property
    def last_normal(self) -> int:
        return self.normals[-1]

    @property
    def representative_start(self) -> int:
        return round(float(np.median(self.starts)))

    @property
    def representative_end(self) -> int:
        return round(float(np.median(self.ends)))

    def append(self, normal: int, start: int, end: int) -> None:
        self.normals.append(normal)
        self.starts.append(start)
        self.ends.append(end)

    def freeze(self) -> _GapCandidate:
        return _GapCandidate(
            normal_start=min(self.normals),
            normal_end=max(self.normals),
            gap_start=self.representative_start,
            gap_end=self.representative_end,
        )


@dataclass(frozen=True)
class _GapCandidate:
    normal_start: int
    normal_end: int
    gap_start: int
    gap_end: int

    @property
    def normal_center(self) -> float:
        return (self.normal_start + self.normal_end) / 2.0

    @property
    def normal_thickness(self) -> int:
        return self.normal_end - self.normal_start + 1

    @property
    def gap_length(self) -> int:
        return self.gap_end - self.gap_start + 1


# ### Public API ###
def make_doorway_walls_continuous(image: np.ndarray) -> np.ndarray:
    """Bridge paired wall-boundary gaps while leaving the input array unchanged.

    The detector examines horizontal and vertical ink independently. A gap is
    filled only when another gap with matching endpoints exists on a nearby,
    parallel line. This represents the two visible boundaries of a wall around a
    doorway and intentionally leaves isolated line breaks untouched.
    """

    _validate_image(image)
    output = image.copy()
    ink = np.where(image <= _INK_THRESHOLD, 255, 0).astype(np.uint8)
    limits = _detection_limits(image.shape)
    horizontal_candidates = _find_gap_candidates(ink, limits)
    vertical_candidates = _find_gap_candidates(ink.T, limits)

    for first, second in _pair_wall_boundaries(horizontal_candidates, limits):
        _fill_horizontal_gap(output, first)
        _fill_horizontal_gap(output, second)
    for first, second in _pair_wall_boundaries(vertical_candidates, limits):
        _fill_vertical_gap(output, first)
        _fill_vertical_gap(output, second)
    return output


# ### Input validation ###
def _validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("The corrected plan must be a NumPy array.")
    if image.dtype != np.uint8:
        raise TypeError("The corrected plan must use the uint8 data type.")
    if image.ndim != 2:
        raise ValueError("The corrected plan must be a two-dimensional image.")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("The corrected plan cannot be empty.")


# ### Adaptive detection limits ###
def _detection_limits(shape: tuple[int, int]) -> _DetectionLimits:
    short_edge = min(shape)
    minimum_support = max(6, round(short_edge * _MIN_SUPPORT_FRACTION))
    minimum_gap = max(4, round(short_edge * _MIN_GAP_FRACTION))
    maximum_gap = max(24, round(short_edge * _MAX_GAP_FRACTION))
    maximum_wall_separation = max(
        12,
        round(short_edge * _MAX_WALL_SEPARATION_FRACTION),
    )
    return _DetectionLimits(
        minimum_support=minimum_support,
        minimum_gap=minimum_gap,
        maximum_gap=maximum_gap,
        maximum_wall_separation=maximum_wall_separation,
        cluster_endpoint_tolerance=max(2, round(short_edge * 0.002)),
    )


# ### Directional gap detection ###
def _find_gap_candidates(
    ink: np.ndarray,
    limits: _DetectionLimits,
) -> tuple[_GapCandidate, ...]:
    line_kernel = np.ones((1, limits.minimum_support), dtype=np.uint8)
    straight_ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, line_kernel)
    raw_by_normal = [
        _row_gaps(row, limits)
        for row in straight_ink > 0
    ]
    return _cluster_gap_rows(raw_by_normal, limits.cluster_endpoint_tolerance)


def _row_gaps(
    row: np.ndarray,
    limits: _DetectionLimits,
) -> tuple[tuple[int, int], ...]:
    padded = np.pad(row.astype(np.int8), (1, 1), constant_values=0)
    transitions = np.diff(padded)
    run_starts = np.flatnonzero(transitions == 1)
    run_ends = np.flatnonzero(transitions == -1) - 1
    gaps: list[tuple[int, int]] = []
    for index in range(len(run_starts) - 1):
        left_length = int(run_ends[index] - run_starts[index] + 1)
        right_length = int(run_ends[index + 1] - run_starts[index + 1] + 1)
        gap_start = int(run_ends[index] + 1)
        gap_end = int(run_starts[index + 1] - 1)
        gap_length = gap_end - gap_start + 1
        if left_length < limits.minimum_support:
            continue
        if right_length < limits.minimum_support:
            continue
        if not limits.minimum_gap <= gap_length <= limits.maximum_gap:
            continue
        gaps.append((gap_start, gap_end))
    return tuple(gaps)


def _cluster_gap_rows(
    raw_by_normal: list[tuple[tuple[int, int], ...]],
    endpoint_tolerance: int,
) -> tuple[_GapCandidate, ...]:
    active: list[_MutableGapCluster] = []
    completed: list[_MutableGapCluster] = []
    for normal, row_gaps in enumerate(raw_by_normal):
        still_active = [
            cluster for cluster in active if cluster.last_normal >= normal - 1
        ]
        completed.extend(cluster for cluster in active if cluster not in still_active)
        active = still_active
        used_cluster_ids: set[int] = set()
        for start, end in row_gaps:
            matches = [
                cluster
                for cluster in active
                if id(cluster) not in used_cluster_ids
                and abs(cluster.representative_start - start) <= endpoint_tolerance
                and abs(cluster.representative_end - end) <= endpoint_tolerance
            ]
            if matches:
                cluster = min(
                    matches,
                    key=lambda candidate: (
                        abs(candidate.representative_start - start)
                        + abs(candidate.representative_end - end)
                    ),
                )
            else:
                cluster = _MutableGapCluster()
                active.append(cluster)
            cluster.append(normal, start, end)
            used_cluster_ids.add(id(cluster))
    completed.extend(active)
    return tuple(cluster.freeze() for cluster in completed)


# ### Wall-boundary pairing ###
def _pair_wall_boundaries(
    candidates: tuple[_GapCandidate, ...],
    limits: _DetectionLimits,
) -> tuple[tuple[_GapCandidate, _GapCandidate], ...]:
    ordered = sorted(candidates, key=lambda candidate: candidate.normal_center)
    possible_pairs: list[tuple[float, int, int]] = []
    for first_index, first in enumerate(ordered):
        for second_index in range(first_index + 1, len(ordered)):
            second = ordered[second_index]
            separation = second.normal_center - first.normal_center
            if separation > limits.maximum_wall_separation:
                break
            if not _can_be_wall_boundaries(first, second, separation):
                continue
            endpoint_error = abs(first.gap_start - second.gap_start) + abs(
                first.gap_end - second.gap_end
            )
            score = endpoint_error + separation / limits.maximum_wall_separation
            possible_pairs.append((score, first_index, second_index))

    used: set[int] = set()
    pairs: list[tuple[_GapCandidate, _GapCandidate]] = []
    for _score, first_index, second_index in sorted(possible_pairs):
        if first_index in used or second_index in used:
            continue
        used.update((first_index, second_index))
        pairs.append((ordered[first_index], ordered[second_index]))
    return tuple(pairs)


def _can_be_wall_boundaries(
    first: _GapCandidate,
    second: _GapCandidate,
    separation: float,
) -> bool:
    minimum_separation = max(first.normal_thickness, second.normal_thickness) + 1
    if separation < minimum_separation:
        return False
    shorter_gap = min(first.gap_length, second.gap_length)
    if shorter_gap < separation * _MIN_GAP_TO_WALL_SEPARATION_RATIO:
        return False
    overlap = min(first.gap_end, second.gap_end) - max(
        first.gap_start,
        second.gap_start,
    ) + 1
    if overlap <= 0:
        return False
    if overlap / max(first.gap_length, second.gap_length) < _MIN_INTERVAL_OVERLAP_RATIO:
        return False
    endpoint_tolerance = max(2, round(max(first.gap_length, second.gap_length) * 0.15))
    return (
        abs(first.gap_start - second.gap_start) <= endpoint_tolerance
        and abs(first.gap_end - second.gap_end) <= endpoint_tolerance
    )


# ### Gap rendering ###
def _fill_horizontal_gap(output: np.ndarray, candidate: _GapCandidate) -> None:
    output[
        candidate.normal_start : candidate.normal_end + 1,
        max(0, candidate.gap_start - 1) : min(
            output.shape[1],
            candidate.gap_end + 2,
        ),
    ] = 0


def _fill_vertical_gap(output: np.ndarray, candidate: _GapCandidate) -> None:
    output[
        max(0, candidate.gap_start - 1) : min(
            output.shape[0],
            candidate.gap_end + 2,
        ),
        candidate.normal_start : candidate.normal_end + 1,
    ] = 0
