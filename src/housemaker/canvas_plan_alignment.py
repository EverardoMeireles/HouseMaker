# ### Imports ###
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .automatic_wall_detection import (
    PlanWallAnalysis,
    PlanWallDetectionError,
    WallLineEvidence,
    analyze_plan_wall_image,
)
from .models import (
    MAX_CANVAS_LEVEL_SCALE,
    MAX_CANVAS_OFFSET_PIXELS,
    MIN_CANVAS_LEVEL_SCALE,
    MIN_CANVAS_OFFSET_PIXELS,
)

# ### Alignment constants ###
_ANALYSIS_MAXIMUM_DIMENSION = 384
_ORIENTATION_BIN_COUNT = 4
_MINIMUM_EVIDENCE_CONFIDENCE = 0.35
_MINIMUM_EVIDENCE_LENGTH_RATIO = 0.01
_CORRELATION_PEAK_COUNT = 8
_CORRELATION_PEAK_SEPARATION_RATIO = 0.05
_MINIMUM_FEATURE_SCALE = 0.25
_MAXIMUM_FEATURE_SCALE = 4.0
_MINIMUM_VISUAL_PRECISION = 0.35
_MINIMUM_VISUAL_REVERSE_COVERAGE = 0.30
_MINIMUM_VISUAL_WALL_SCORE = 0.35
_MINIMUM_MATCHED_WALL_PIXELS = 40.0
_MINIMUM_MATCHED_WALL_RATIO = 0.15
_AMBIGUOUS_ALIGNMENT_SCORE_MARGIN = 0.02
_DISTINCT_ALIGNMENT_DISTANCE_PIXELS = 16.0
_STAIR_SCORE_WEIGHT = 0.08
_MINIMUM_STAIR_SEGMENT_COUNT = 4
_MINIMUM_STAIR_PIXELS = 12.0
_GEOMETRY_EPSILON = 1e-9

# ### Public tolerance limits ###
MIN_CANVAS_ALIGNMENT_TOLERANCE_PIXELS = 0.5
MAX_CANVAS_ALIGNMENT_TOLERANCE_PIXELS = 12.0
DEFAULT_CANVAS_ALIGNMENT_TOLERANCE_PIXELS = 2.25


# ### Public data models ###
class CanvasPlanAlignmentError(ValueError):
    """A safe, user-displayable automatic Canvas-alignment failure."""


class CanvasPlanAlignmentCancelled(CanvasPlanAlignmentError):
    """Raised when a background Canvas-alignment request is cancelled."""


@dataclass(frozen=True, slots=True)
class CanvasPlanAlignmentLevel:
    """One corrected plan raster and its current Canvas-only transform."""

    level_index: int
    image_path: str
    canvas_level_scale: float
    canvas_offset_x_pixels: float
    canvas_offset_y_pixels: float


@dataclass(frozen=True, slots=True)
class CanvasPlanAlignmentTransform:
    """One Canvas transform aligned through a direct or intermediate plan."""

    level_index: int
    canvas_offset_x_pixels: float
    canvas_offset_y_pixels: float
    reference_level_index: int
    score: float
    wall_score: float
    stair_score: float


@dataclass(frozen=True, slots=True)
class CanvasPlanAlignmentResult:
    """Successful transforms plus images that could not be aligned safely."""

    anchor_level_index: int
    transforms: tuple[CanvasPlanAlignmentTransform, ...]
    skipped_levels: tuple[tuple[int, str], ...] = ()


# ### Internal data models ###
@dataclass(frozen=True, slots=True)
class _PlanFeatureMap:
    level: CanvasPlanAlignmentLevel
    image_width: int
    image_height: int
    native_to_feature_scale: float
    alignment_tolerance_pixels: float
    wall_masks: tuple[np.ndarray, ...]
    wall_heat_maps: tuple[np.ndarray, ...]
    stair_mask: np.ndarray
    stair_heat_map: np.ndarray


@dataclass(frozen=True, slots=True)
class _PairRegistration:
    native_scale: float
    native_translation: tuple[float, float]
    score: float
    wall_score: float
    stair_score: float


@dataclass(frozen=True, slots=True)
class _WallAlignmentScore:
    score: float
    precision: float
    reverse_coverage: float
    matched_wall_pixels: float
    target_wall_pixels: float
    reference_roi_wall_pixels: float
    matched_orientation_count: int


@dataclass(frozen=True, slots=True)
class _RegistrationCandidate:
    score: float
    feature_scale: float
    feature_translation: tuple[float, float]
    wall_alignment: _WallAlignmentScore
    stair_score: float


# ### Public alignment API ###
def align_canvas_plan_levels(
    levels: Sequence[CanvasPlanAlignmentLevel],
    *,
    anchor_level_index: int,
    viewport_size: tuple[float, float],
    alignment_tolerance_pixels: float = DEFAULT_CANVAS_ALIGNMENT_TOLERANCE_PIXELS,
    cancellation_check: Callable[[], bool] | None = None,
) -> CanvasPlanAlignmentResult:
    """Align corrected plans by changing Canvas X/Y offsets only.

    Wall-line agreement is the required evidence. Repeated short parallel lines,
    such as stair treads, provide only a small tie-breaking score and can never
    make an otherwise weak wall registration pass validation. Every level keeps
    its existing Canvas scale; image-fit differences are accounted for while
    solving the translation.
    """

    _raise_if_cancelled(cancellation_check)
    normalized_levels = _validate_alignment_inputs(levels, viewport_size)
    normalized_tolerance = _validate_alignment_tolerance(alignment_tolerance_pixels)
    if len(normalized_levels) < 2:
        raise CanvasPlanAlignmentError(
            "At least two levels with readable plan images are required."
        )
    levels_by_index = {level.level_index: level for level in normalized_levels}
    anchor_level = levels_by_index.get(int(anchor_level_index))
    if anchor_level is None:
        raise CanvasPlanAlignmentError(
            "The alignment reference level does not have a plan image."
        )

    feature_maps: dict[int, _PlanFeatureMap] = {}
    skipped: list[tuple[int, str]] = []
    for level in normalized_levels:
        _raise_if_cancelled(cancellation_check)
        try:
            feature_maps[level.level_index] = _build_plan_feature_map(
                level,
                cancellation_check,
                normalized_tolerance,
            )
        except (OSError, PlanWallDetectionError, CanvasPlanAlignmentError) as error:
            _raise_if_cancelled(cancellation_check)
            skipped.append((level.level_index, str(error)))

    if len(feature_maps) < 2:
        raise CanvasPlanAlignmentError(
            "Fewer than two plan images contain usable wall-line evidence."
        )
    if anchor_level.level_index not in feature_maps:
        fallback_level_index = min(
            feature_maps,
            key=lambda level_index: (
                abs(level_index - anchor_level.level_index),
                abs(level_index),
                level_index,
            ),
        )
        anchor_level = levels_by_index[fallback_level_index]

    native_to_anchor: dict[int, tuple[float, np.ndarray]] = {
        anchor_level.level_index: (1.0, np.zeros(2, dtype=np.float64))
    }
    registrations: dict[int, tuple[int, _PairRegistration]] = {}
    remaining = set(feature_maps).difference(native_to_anchor)
    deferred_failures: dict[int, str] = {}
    while remaining:
        _raise_if_cancelled(cancellation_check)
        available_targets = remaining.difference(deferred_failures)
        if not available_targets:
            skipped.extend(sorted(deferred_failures.items()))
            break
        target_index, reference_indices = _choose_next_alignment_references(
            available_targets,
            native_to_anchor,
        )
        successful: list[tuple[_PairRegistration, int]] = []
        failure_messages: list[str] = []
        for reference_index in reference_indices:
            try:
                registration = _register_feature_maps(
                    feature_maps[reference_index],
                    feature_maps[target_index],
                    viewport_size,
                    cancellation_check,
                )
            except CanvasPlanAlignmentError as error:
                _raise_if_cancelled(cancellation_check)
                failure_messages.append(str(error))
                continue
            successful.append((registration, reference_index))
        if not successful:
            deferred_failures[target_index] = (
                failure_messages[0]
                if failure_messages
                else "No aligned reference level was available."
            )
            continue

        registration, reference_index = max(
            successful,
            key=lambda item: (
                item[0].score,
                item[0].wall_score,
                -abs(target_index - item[1]),
            ),
        )
        reference_scale, reference_translation = native_to_anchor[reference_index]
        pair_translation = np.asarray(
            registration.native_translation,
            dtype=np.float64,
        )
        native_to_anchor[target_index] = (
            reference_scale * registration.native_scale,
            reference_scale * pair_translation + reference_translation,
        )
        registrations[target_index] = (reference_index, registration)
        remaining.remove(target_index)
        deferred_failures.clear()

    transforms = _build_canvas_transforms(
        anchor_level,
        feature_maps,
        native_to_anchor,
        registrations,
        viewport_size,
        skipped,
    )
    if len(transforms) < 2:
        raise CanvasPlanAlignmentError(
            "No other plan image could be aligned reliably with the reference."
        )
    return CanvasPlanAlignmentResult(
        anchor_level_index=anchor_level.level_index,
        transforms=tuple(sorted(transforms, key=lambda item: item.level_index)),
        skipped_levels=tuple(sorted(set(skipped))),
    )


# ### Input validation ###
def _validate_alignment_inputs(
    levels: Sequence[CanvasPlanAlignmentLevel],
    viewport_size: tuple[float, float],
) -> tuple[CanvasPlanAlignmentLevel, ...]:
    try:
        viewport_width = float(viewport_size[0])
        viewport_height = float(viewport_size[1])
    except (IndexError, TypeError, ValueError) as error:
        raise CanvasPlanAlignmentError(
            "The Canvas viewport size is invalid."
        ) from error
    if not (
        math.isfinite(viewport_width)
        and math.isfinite(viewport_height)
        and viewport_width > 0.0
        and viewport_height > 0.0
    ):
        raise CanvasPlanAlignmentError("The Canvas viewport must have a positive size.")

    normalized: list[CanvasPlanAlignmentLevel] = []
    seen_indices: set[int] = set()
    for level in levels:
        if not isinstance(level, CanvasPlanAlignmentLevel):
            raise CanvasPlanAlignmentError(
                "Canvas alignment received invalid level data."
            )
        if level.level_index in seen_indices:
            raise CanvasPlanAlignmentError(
                "Canvas alignment levels must have unique indices."
            )
        seen_indices.add(level.level_index)
        numeric_values = (
            level.canvas_level_scale,
            level.canvas_offset_x_pixels,
            level.canvas_offset_y_pixels,
        )
        if not all(math.isfinite(float(value)) for value in numeric_values):
            raise CanvasPlanAlignmentError(
                f"Level {level.level_index} contains non-finite Canvas data."
            )
        if not (
            MIN_CANVAS_LEVEL_SCALE <= level.canvas_level_scale <= MAX_CANVAS_LEVEL_SCALE
        ):
            raise CanvasPlanAlignmentError(
                f"Level {level.level_index} has an invalid Canvas scale."
            )
        if not (
            MIN_CANVAS_OFFSET_PIXELS
            <= level.canvas_offset_x_pixels
            <= MAX_CANVAS_OFFSET_PIXELS
            and MIN_CANVAS_OFFSET_PIXELS
            <= level.canvas_offset_y_pixels
            <= MAX_CANVAS_OFFSET_PIXELS
        ):
            raise CanvasPlanAlignmentError(
                f"Level {level.level_index} has an invalid Canvas offset."
            )
        if not str(level.image_path).strip():
            raise CanvasPlanAlignmentError(
                f"Level {level.level_index} does not have a plan image."
            )
        normalized.append(level)
    return tuple(normalized)


def _validate_alignment_tolerance(value: float) -> float:
    if isinstance(value, bool):
        raise CanvasPlanAlignmentError("Alignment tolerance must be numeric.")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as error:
        raise CanvasPlanAlignmentError(
            "Alignment tolerance must be numeric."
        ) from error
    if not (
        math.isfinite(normalized)
        and MIN_CANVAS_ALIGNMENT_TOLERANCE_PIXELS
        <= normalized
        <= MAX_CANVAS_ALIGNMENT_TOLERANCE_PIXELS
    ):
        raise CanvasPlanAlignmentError(
            "Alignment tolerance must be between "
            f"{MIN_CANVAS_ALIGNMENT_TOLERANCE_PIXELS:g} and "
            f"{MAX_CANVAS_ALIGNMENT_TOLERANCE_PIXELS:g} pixels."
        )
    return normalized


def _raise_if_cancelled(
    cancellation_check: Callable[[], bool] | None,
) -> None:
    if cancellation_check is not None and cancellation_check():
        raise CanvasPlanAlignmentCancelled("Canvas plan alignment was cancelled.")


# ### Wall-line feature preparation ###
def _build_plan_feature_map(
    level: CanvasPlanAlignmentLevel,
    cancellation_check: Callable[[], bool] | None,
    alignment_tolerance_pixels: float,
) -> _PlanFeatureMap:
    analysis = analyze_plan_wall_image(
        Path(level.image_path),
        cancellation_check=cancellation_check,
    )
    return _feature_map_from_analysis(
        level,
        analysis,
        alignment_tolerance_pixels,
    )


def _feature_map_from_analysis(
    level: CanvasPlanAlignmentLevel,
    analysis: PlanWallAnalysis,
    alignment_tolerance_pixels: float = DEFAULT_CANVAS_ALIGNMENT_TOLERANCE_PIXELS,
) -> _PlanFeatureMap:
    maximum_dimension = max(analysis.image_width, analysis.image_height)
    if maximum_dimension <= 0:
        raise CanvasPlanAlignmentError("The plan image has an invalid size.")
    native_to_feature_scale = min(
        1.0,
        _ANALYSIS_MAXIMUM_DIMENSION / float(maximum_dimension),
    )
    feature_width = max(2, round(analysis.image_width * native_to_feature_scale))
    feature_height = max(2, round(analysis.image_height * native_to_feature_scale))
    wall_masks = [
        np.zeros((feature_height, feature_width), dtype=np.uint8)
        for _unused in range(_ORIENTATION_BIN_COUNT)
    ]
    minimum_length = max(
        5.0,
        min(analysis.image_width, analysis.image_height)
        * _MINIMUM_EVIDENCE_LENGTH_RATIO,
    )
    accepted: list[WallLineEvidence] = []
    for evidence in analysis.line_evidence:
        if (
            evidence.confidence < _MINIMUM_EVIDENCE_CONFIDENCE
            or evidence.length < minimum_length
        ):
            continue
        start = np.asarray(evidence.start, dtype=np.float64)
        end = np.asarray(evidence.end, dtype=np.float64)
        orientation_index = _orientation_bin_for_segment(start, end)
        intensity = round(min(1.0, max(0.25, evidence.confidence)) * 255.0)
        scaled_start = tuple(np.rint(start * native_to_feature_scale).astype(int))
        scaled_end = tuple(np.rint(end * native_to_feature_scale).astype(int))
        cv2.line(
            wall_masks[orientation_index],
            scaled_start,
            scaled_end,
            color=intensity,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        accepted.append(evidence)
    if not accepted or not any(np.any(mask) for mask in wall_masks):
        raise CanvasPlanAlignmentError("No usable wall lines were found in the plan.")

    normalized_tolerance = _validate_alignment_tolerance(alignment_tolerance_pixels)
    stair_mask = _build_stair_cue_mask(
        accepted,
        feature_width,
        feature_height,
        native_to_feature_scale,
        min(analysis.image_width, analysis.image_height),
    )
    stair_exclusion = cv2.dilate(
        np.where(stair_mask > 0, 255, 0).astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )
    wall_masks = [
        np.where(stair_exclusion > 0, 0, mask).astype(np.uint8) for mask in wall_masks
    ]
    if not any(np.any(mask) for mask in wall_masks):
        raise CanvasPlanAlignmentError(
            "No structural wall lines remained after separating stair cues."
        )
    return _PlanFeatureMap(
        level=level,
        image_width=int(analysis.image_width),
        image_height=int(analysis.image_height),
        native_to_feature_scale=native_to_feature_scale,
        alignment_tolerance_pixels=normalized_tolerance,
        wall_masks=tuple(wall_masks),
        wall_heat_maps=tuple(
            _build_line_heat_map(mask, normalized_tolerance) for mask in wall_masks
        ),
        stair_mask=stair_mask,
        stair_heat_map=_build_line_heat_map(stair_mask, normalized_tolerance),
    )


def _orientation_bin_for_segment(start: np.ndarray, end: np.ndarray) -> int:
    direction = end - start
    angle = math.atan2(float(direction[1]), float(direction[0])) % math.pi
    return min(
        _ORIENTATION_BIN_COUNT - 1,
        int(round(angle / math.pi * _ORIENTATION_BIN_COUNT) % _ORIENTATION_BIN_COUNT),
    )


def _build_line_heat_map(
    mask: np.ndarray,
    tolerance_pixels: float = DEFAULT_CANVAS_ALIGNMENT_TOLERANCE_PIXELS,
) -> np.ndarray:
    binary = mask > 0
    if not np.any(binary):
        return np.zeros(mask.shape, dtype=np.float32)
    distance = cv2.distanceTransform((~binary).astype(np.uint8), cv2.DIST_L2, 3)
    heat = np.exp(-0.5 * np.square(distance / tolerance_pixels)).astype(np.float32)
    return heat


# ### Stair-cue preparation ###
def _build_stair_cue_mask(
    evidence: Sequence[WallLineEvidence],
    feature_width: int,
    feature_height: int,
    native_to_feature_scale: float,
    minimum_image_dimension: int,
) -> np.ndarray:
    """Rasterize only short parallel-line clusters resembling stair treads."""

    mask = np.zeros((feature_height, feature_width), dtype=np.uint8)
    minimum_length = max(4.0, minimum_image_dimension * 0.008)
    maximum_length = max(minimum_length, minimum_image_dimension * 0.22)
    candidates = tuple(
        item for item in evidence if minimum_length <= item.length <= maximum_length
    )
    accepted_indices: set[int] = set()
    for seed_index, seed in enumerate(candidates):
        seed_start = np.asarray(seed.start, dtype=np.float64)
        seed_end = np.asarray(seed.end, dtype=np.float64)
        seed_vector = seed_end - seed_start
        seed_length = float(np.linalg.norm(seed_vector))
        if seed_length <= _GEOMETRY_EPSILON:
            continue
        direction = seed_vector / seed_length
        if direction[0] < 0.0 or (
            abs(float(direction[0])) <= _GEOMETRY_EPSILON and direction[1] < 0.0
        ):
            direction = -direction
        normal = np.asarray((-direction[1], direction[0]), dtype=np.float64)
        seed_center = 0.5 * (seed_start + seed_end)
        cluster: list[tuple[int, float]] = []
        for candidate_index, candidate in enumerate(candidates):
            start = np.asarray(candidate.start, dtype=np.float64)
            end = np.asarray(candidate.end, dtype=np.float64)
            vector = end - start
            length = float(np.linalg.norm(vector))
            if length <= _GEOMETRY_EPSILON:
                continue
            candidate_direction = vector / length
            if abs(float(np.dot(direction, candidate_direction))) < math.cos(
                math.radians(8.0)
            ):
                continue
            if not 0.50 <= length / seed_length <= 2.0:
                continue
            center_delta = 0.5 * (start + end) - seed_center
            along_distance = abs(float(np.dot(center_delta, direction)))
            across_distance = float(np.dot(center_delta, normal))
            if along_distance > max(seed_length, length) * 0.75:
                continue
            if abs(across_distance) > max(
                minimum_image_dimension * 0.16,
                seed_length * 5.0,
            ):
                continue
            cluster.append((candidate_index, across_distance))
        distinct_rows = _distinct_sorted_values(
            (distance for _index, distance in cluster),
            minimum_separation=max(1.5, minimum_image_dimension * 0.002),
        )
        if len(
            distinct_rows
        ) >= _MINIMUM_STAIR_SEGMENT_COUNT and _has_regular_stair_spacing(distinct_rows):
            accepted_indices.update(index for index, _distance in cluster)

    for index in accepted_indices:
        item = candidates[index]
        start = tuple(
            np.rint(np.asarray(item.start) * native_to_feature_scale).astype(int)
        )
        end = tuple(np.rint(np.asarray(item.end) * native_to_feature_scale).astype(int))
        cv2.line(mask, start, end, color=255, thickness=2, lineType=cv2.LINE_AA)
    return mask


def _distinct_sorted_values(
    values: Sequence[float],
    *,
    minimum_separation: float,
) -> tuple[float, ...]:
    distinct: list[float] = []
    for value in sorted(values):
        if not distinct or value - distinct[-1] >= minimum_separation:
            distinct.append(float(value))
    return tuple(distinct)


def _has_regular_stair_spacing(rows: Sequence[float]) -> bool:
    """Reject arbitrary parallel marks whose spacing does not resemble treads."""

    gaps = np.diff(np.asarray(rows, dtype=np.float64))
    if len(gaps) < _MINIMUM_STAIR_SEGMENT_COUNT - 1:
        return False
    median_gap = float(np.median(gaps))
    if median_gap <= _GEOMETRY_EPSILON:
        return False
    normalized_error = np.abs(gaps - median_gap) / median_gap
    return bool(
        float(np.median(normalized_error)) <= 0.25
        and float(np.max(normalized_error)) <= 0.75
    )


# ### Alignment ordering ###
def _choose_next_alignment_references(
    remaining: set[int],
    native_to_anchor: dict[int, tuple[float, np.ndarray]],
) -> tuple[int, tuple[int, ...]]:
    """Prefer nearby levels while retaining every aligned plan as a reference."""

    if not remaining or not native_to_anchor:
        raise CanvasPlanAlignmentError("No plan images remain available to align.")
    target_index = min(
        remaining,
        key=lambda index: (
            min(abs(index - reference) for reference in native_to_anchor),
            abs(index),
            index,
        ),
    )
    references = tuple(
        sorted(
            native_to_anchor,
            key=lambda reference: (
                abs(target_index - reference),
                abs(reference),
                reference,
            ),
        )
    )
    return target_index, references


# ### Pairwise registration ###
def _register_feature_maps(
    reference: _PlanFeatureMap,
    target: _PlanFeatureMap,
    viewport_size: tuple[float, float],
    cancellation_check: Callable[[], bool] | None,
) -> _PairRegistration:
    initial_native_scale = _initial_native_scale(reference, target, viewport_size)
    initial_feature_scale = (
        reference.native_to_feature_scale
        * initial_native_scale
        / target.native_to_feature_scale
    )
    if not _MINIMUM_FEATURE_SCALE <= initial_feature_scale <= _MAXIMUM_FEATURE_SCALE:
        raise CanvasPlanAlignmentError(
            "The plans cannot be aligned without changing their Canvas scales."
        )
    evaluated_candidates: list[_RegistrationCandidate] = []
    _find_best_registration_candidate(
        reference,
        target,
        (initial_feature_scale,),
        cancellation_check,
        candidate_sink=evaluated_candidates,
    )
    reliable_candidates = tuple(
        candidate
        for candidate in evaluated_candidates
        if _has_reliable_visual_alignment(candidate)
    )
    if not reliable_candidates:
        raise CanvasPlanAlignmentError(
            "The plan does not share enough aligned wall-line evidence with "
            "the available reference levels."
        )
    best = max(
        reliable_candidates,
        key=lambda candidate: (
            candidate.score,
            candidate.wall_alignment.score,
        ),
    )
    ambiguous_runner_up = _find_ambiguous_runner_up(
        best,
        evaluated_candidates,
        target.wall_masks[0].shape,
    )
    if ambiguous_runner_up is not None:
        raise CanvasPlanAlignmentError(
            "The plan has multiple similarly plausible wall alignments."
        )

    native_translation = (
        np.asarray(
            best.feature_translation,
            dtype=np.float64,
        )
        / reference.native_to_feature_scale
    )
    return _PairRegistration(
        native_scale=float(initial_native_scale),
        native_translation=(
            float(native_translation[0]),
            float(native_translation[1]),
        ),
        score=float(best.score),
        wall_score=float(best.wall_alignment.score),
        stair_score=float(best.stair_score),
    )


def _has_reliable_visual_alignment(
    candidate: _RegistrationCandidate,
) -> bool:
    """Validate only wall evidence; stair cues must never authorize a match."""

    wall_alignment = candidate.wall_alignment
    matched_wall_requirement = max(
        _MINIMUM_MATCHED_WALL_PIXELS,
        _MINIMUM_MATCHED_WALL_RATIO
        * min(
            wall_alignment.target_wall_pixels,
            wall_alignment.reference_roi_wall_pixels,
        ),
    )
    return (
        wall_alignment.precision >= _MINIMUM_VISUAL_PRECISION
        and wall_alignment.reverse_coverage >= _MINIMUM_VISUAL_REVERSE_COVERAGE
        and wall_alignment.score >= _MINIMUM_VISUAL_WALL_SCORE
        and wall_alignment.matched_wall_pixels >= matched_wall_requirement
        and wall_alignment.matched_orientation_count >= 2
    )


def _find_ambiguous_runner_up(
    best: _RegistrationCandidate,
    candidates: Sequence[_RegistrationCandidate],
    target_shape: tuple[int, int],
) -> _RegistrationCandidate | None:
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if candidate is best:
            continue
        if best.score - candidate.score > _AMBIGUOUS_ALIGNMENT_SCORE_MARGIN:
            break
        if not _has_reliable_visual_alignment(candidate):
            continue
        if (
            _alignment_mapping_distance(best, candidate, target_shape)
            >= _DISTINCT_ALIGNMENT_DISTANCE_PIXELS
        ):
            return candidate
    return None


def _alignment_mapping_distance(
    first: _RegistrationCandidate,
    second: _RegistrationCandidate,
    target_shape: tuple[int, int],
) -> float:
    target_height, target_width = target_shape
    sample_points = np.asarray(
        (
            (0.0, 0.0),
            (float(target_width), 0.0),
            (0.0, float(target_height)),
            (float(target_width), float(target_height)),
            (float(target_width) / 2.0, float(target_height) / 2.0),
        ),
        dtype=np.float64,
    )
    first_points = first.feature_scale * sample_points + np.asarray(
        first.feature_translation,
        dtype=np.float64,
    )
    second_points = second.feature_scale * sample_points + np.asarray(
        second.feature_translation,
        dtype=np.float64,
    )
    return float(np.sqrt(np.mean(np.square(first_points - second_points))))


def _initial_native_scale(
    reference: _PlanFeatureMap,
    target: _PlanFeatureMap,
    viewport_size: tuple[float, float],
) -> float:
    reference_fit = _image_fit_scale(
        reference.image_width,
        reference.image_height,
        viewport_size,
    )
    target_fit = _image_fit_scale(
        target.image_width,
        target.image_height,
        viewport_size,
    )
    reference_display_scale = reference_fit * reference.level.canvas_level_scale
    target_display_scale = target_fit * target.level.canvas_level_scale
    return max(
        _GEOMETRY_EPSILON,
        target_display_scale / reference_display_scale,
    )


def _find_best_registration_candidate(
    reference: _PlanFeatureMap,
    target: _PlanFeatureMap,
    feature_scales: Sequence[float],
    cancellation_check: Callable[[], bool] | None,
    *,
    candidate_sink: list[_RegistrationCandidate] | None = None,
) -> _RegistrationCandidate:
    best: _RegistrationCandidate | None = None
    has_stair_evidence = (
        float(np.sum(reference.stair_mask > 0)) >= _MINIMUM_STAIR_PIXELS
        and float(np.sum(target.stair_mask > 0)) >= _MINIMUM_STAIR_PIXELS
    )
    for feature_scale in feature_scales:
        _raise_if_cancelled(cancellation_check)
        resized_masks = _resize_line_masks(target.wall_masks, feature_scale)
        resized_heat_maps = tuple(
            _build_line_heat_map(
                np.where(mask > 0.0, 255, 0).astype(np.uint8),
                reference.alignment_tolerance_pixels,
            )
            for mask in resized_masks
        )
        resized_stair_mask = _resize_single_mask(target.stair_mask, feature_scale)
        resized_stair_heat = _build_line_heat_map(
            np.where(resized_stair_mask > 0.0, 255, 0).astype(np.uint8),
            reference.alignment_tolerance_pixels,
        )
        support_mask = _build_plan_support_mask(resized_masks)
        correlation_candidates = _best_correlation_translations(
            reference.wall_heat_maps,
            resized_masks,
        )
        translations = [translation for translation, _score in correlation_candidates]
        if has_stair_evidence and np.any(resized_stair_mask):
            translations.extend(
                translation
                for translation, _score in _best_correlation_translations(
                    (reference.stair_heat_map,),
                    (resized_stair_mask,),
                )
            )
        if not translations:
            translations.append((0.0, 0.0))
        unique_translations = tuple(
            dict.fromkeys(
                (round(float(x), 4), round(float(y), 4)) for x, y in translations
            )
        )
        for translation in unique_translations:
            wall_alignment = _score_bidirectional_wall_alignment(
                reference.wall_masks,
                reference.wall_heat_maps,
                resized_masks,
                resized_heat_maps,
                support_mask,
                translation,
            )
            stair_score = (
                _score_stair_alignment(
                    reference.stair_mask,
                    reference.stair_heat_map,
                    resized_stair_mask,
                    resized_stair_heat,
                    translation,
                )
                if has_stair_evidence
                else 0.0
            )
            score = (
                1.0 - _STAIR_SCORE_WEIGHT
            ) * wall_alignment.score + _STAIR_SCORE_WEIGHT * stair_score
            candidate = _RegistrationCandidate(
                score=float(score),
                feature_scale=float(feature_scale),
                feature_translation=(
                    float(translation[0]),
                    float(translation[1]),
                ),
                wall_alignment=wall_alignment,
                stair_score=float(stair_score),
            )
            if candidate_sink is not None:
                candidate_sink.append(candidate)
            if best is None or candidate.score > best.score:
                best = candidate
    if best is None:
        raise CanvasPlanAlignmentError(
            "The plan did not produce any registration candidates."
        )
    return best


def _resize_line_masks(
    line_masks: Sequence[np.ndarray],
    scale: float,
) -> tuple[np.ndarray, ...]:
    return tuple(_resize_single_mask(mask, scale) for mask in line_masks)


def _resize_single_mask(mask: np.ndarray, scale: float) -> np.ndarray:
    source_height, source_width = mask.shape
    target_width = max(2, round(source_width * scale))
    target_height = max(2, round(source_height * scale))
    return cv2.resize(
        mask.astype(np.float32) / 255.0,
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST,
    )


def _best_correlation_translations(
    reference_heat_maps: Sequence[np.ndarray],
    resized_masks: Sequence[np.ndarray],
) -> tuple[tuple[tuple[float, float], float], ...]:
    target_height, target_width = resized_masks[0].shape
    reference_height, reference_width = reference_heat_maps[0].shape
    pad_x = target_width
    pad_y = target_height
    response_sum: np.ndarray | None = None
    target_weight = 0.0
    for heat_map, target_mask in zip(
        reference_heat_maps,
        resized_masks,
        strict=True,
    ):
        channel_weight = float(np.sum(target_mask))
        if channel_weight <= _GEOMETRY_EPSILON:
            continue
        padded_heat = cv2.copyMakeBorder(
            heat_map,
            pad_y,
            pad_y,
            pad_x,
            pad_x,
            cv2.BORDER_CONSTANT,
            value=0.0,
        )
        response = cv2.matchTemplate(padded_heat, target_mask, cv2.TM_CCORR)
        response_sum = response if response_sum is None else response_sum + response
        target_weight += channel_weight
    if response_sum is None or target_weight <= _GEOMETRY_EPSILON:
        return ()
    if reference_width <= 0 or reference_height <= 0:
        return ()

    working_response = response_sum.astype(np.float32, copy=True)
    suppression_radius = max(
        6,
        round(min(target_width, target_height) * _CORRELATION_PEAK_SEPARATION_RATIO),
    )
    peaks: list[tuple[tuple[float, float], float]] = []
    rejected_value = np.finfo(np.float32).min
    for _unused in range(_CORRELATION_PEAK_COUNT):
        _minimum, maximum, _minimum_location, maximum_location = cv2.minMaxLoc(
            working_response
        )
        if not math.isfinite(maximum) or maximum <= _GEOMETRY_EPSILON:
            break
        peaks.append(
            (
                (
                    float(maximum_location[0] - pad_x),
                    float(maximum_location[1] - pad_y),
                ),
                max(0.0, min(1.0, float(maximum) / target_weight)),
            )
        )
        x_start = max(0, maximum_location[0] - suppression_radius)
        x_end = min(
            working_response.shape[1],
            maximum_location[0] + suppression_radius + 1,
        )
        y_start = max(0, maximum_location[1] - suppression_radius)
        y_end = min(
            working_response.shape[0],
            maximum_location[1] + suppression_radius + 1,
        )
        working_response[y_start:y_end, x_start:x_end] = rejected_value
    return tuple(peaks)


def _build_plan_support_mask(
    resized_masks: Sequence[np.ndarray],
) -> np.ndarray:
    combined = np.maximum.reduce(resized_masks) > 0.0
    support = np.zeros(combined.shape, dtype=np.float32)
    y_coordinates, x_coordinates = np.nonzero(combined)
    if len(x_coordinates) < 3:
        support[combined] = 1.0
        return support
    points = np.column_stack((x_coordinates, y_coordinates)).astype(np.int32)
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(support, hull, 1.0)
    return support


def _score_bidirectional_wall_alignment(
    reference_line_masks: Sequence[np.ndarray],
    reference_heat_maps: Sequence[np.ndarray],
    resized_masks: Sequence[np.ndarray],
    resized_heat_maps: Sequence[np.ndarray],
    support_mask: np.ndarray,
    translation: tuple[float, float],
) -> _WallAlignmentScore:
    translation_x = round(translation[0])
    translation_y = round(translation[1])
    reference_height, reference_width = reference_heat_maps[0].shape
    target_wall_pixels = 0.0
    matched_target_pixels = 0.0
    reference_roi_wall_pixels = 0.0
    matched_reference_pixels = 0.0
    matched_orientation_count = 0
    for reference_mask, reference_heat, target_mask, target_heat in zip(
        reference_line_masks,
        reference_heat_maps,
        resized_masks,
        resized_heat_maps,
        strict=True,
    ):
        target_height, target_width = target_mask.shape
        target_x_start = max(0, -translation_x)
        target_y_start = max(0, -translation_y)
        target_x_end = min(target_width, reference_width - translation_x)
        target_y_end = min(target_height, reference_height - translation_y)
        channel_target_pixels = float(np.sum(target_mask))
        target_wall_pixels += channel_target_pixels
        if target_x_start >= target_x_end or target_y_start >= target_y_end:
            continue
        reference_heat_slice = reference_heat[
            translation_y + target_y_start : translation_y + target_y_end,
            translation_x + target_x_start : translation_x + target_x_end,
        ]
        reference_mask_slice = (
            reference_mask[
                translation_y + target_y_start : translation_y + target_y_end,
                translation_x + target_x_start : translation_x + target_x_end,
            ].astype(np.float32)
            / 255.0
        )
        target_slice = target_mask[
            target_y_start:target_y_end,
            target_x_start:target_x_end,
        ]
        target_heat_slice = target_heat[
            target_y_start:target_y_end,
            target_x_start:target_x_end,
        ]
        support_slice = support_mask[
            target_y_start:target_y_end,
            target_x_start:target_x_end,
        ]
        channel_matched_target = float(np.sum(reference_heat_slice * target_slice))
        reference_roi_slice = reference_mask_slice * support_slice
        channel_reference_pixels = float(np.sum(reference_roi_slice))
        channel_matched_reference = float(
            np.sum(target_heat_slice * reference_roi_slice)
        )
        matched_target_pixels += channel_matched_target
        reference_roi_wall_pixels += channel_reference_pixels
        matched_reference_pixels += channel_matched_reference
        if min(channel_matched_target, channel_matched_reference) >= 5.0:
            matched_orientation_count += 1

    precision = (
        matched_target_pixels / target_wall_pixels
        if target_wall_pixels > _GEOMETRY_EPSILON
        else 0.0
    )
    reverse_coverage = (
        matched_reference_pixels / reference_roi_wall_pixels
        if reference_roi_wall_pixels > _GEOMETRY_EPSILON
        else 0.0
    )
    score = (
        2.0 * precision * reverse_coverage / (precision + reverse_coverage)
        if precision + reverse_coverage > _GEOMETRY_EPSILON
        else 0.0
    )
    return _WallAlignmentScore(
        score=max(0.0, min(1.0, float(score))),
        precision=max(0.0, min(1.0, float(precision))),
        reverse_coverage=max(0.0, min(1.0, float(reverse_coverage))),
        matched_wall_pixels=min(matched_target_pixels, matched_reference_pixels),
        target_wall_pixels=target_wall_pixels,
        reference_roi_wall_pixels=reference_roi_wall_pixels,
        matched_orientation_count=matched_orientation_count,
    )


def _score_stair_alignment(
    reference_mask: np.ndarray,
    reference_heat: np.ndarray,
    target_mask: np.ndarray,
    target_heat: np.ndarray,
    translation: tuple[float, float],
) -> float:
    if not np.any(reference_mask) or not np.any(target_mask):
        return 0.0
    support = _build_plan_support_mask((target_mask,))
    score = _score_bidirectional_wall_alignment(
        (reference_mask,),
        (reference_heat,),
        (target_mask,),
        (target_heat,),
        support,
        translation,
    )
    return score.score


# ### Canvas transform conversion ###
def _build_canvas_transforms(
    anchor_level: CanvasPlanAlignmentLevel,
    feature_maps: dict[int, _PlanFeatureMap],
    native_to_anchor: dict[int, tuple[float, np.ndarray]],
    registrations: dict[int, tuple[int, _PairRegistration]],
    viewport_size: tuple[float, float],
    skipped: list[tuple[int, str]],
) -> list[CanvasPlanAlignmentTransform]:
    anchor_features = feature_maps[anchor_level.level_index]
    anchor_fit = _image_fit_scale(
        anchor_features.image_width,
        anchor_features.image_height,
        viewport_size,
    )
    anchor_display_scale = anchor_fit * anchor_level.canvas_level_scale
    anchor_center = np.asarray(
        (anchor_features.image_width / 2.0, anchor_features.image_height / 2.0),
        dtype=np.float64,
    )
    anchor_offset = np.asarray(
        (
            anchor_level.canvas_offset_x_pixels,
            anchor_level.canvas_offset_y_pixels,
        ),
        dtype=np.float64,
    )
    transforms = [
        CanvasPlanAlignmentTransform(
            level_index=anchor_level.level_index,
            canvas_offset_x_pixels=anchor_level.canvas_offset_x_pixels,
            canvas_offset_y_pixels=anchor_level.canvas_offset_y_pixels,
            reference_level_index=anchor_level.level_index,
            score=1.0,
            wall_score=1.0,
            stair_score=1.0 if np.any(anchor_features.stair_mask) else 0.0,
        )
    ]
    for level_index, (native_scale, native_translation) in native_to_anchor.items():
        if level_index == anchor_level.level_index:
            continue
        features = feature_maps[level_index]
        target_center = np.asarray(
            (features.image_width / 2.0, features.image_height / 2.0),
            dtype=np.float64,
        )
        offset = anchor_offset + anchor_display_scale * (
            native_translation - anchor_center + native_scale * target_center
        )
        if not (
            np.all(np.isfinite(offset))
            and MIN_CANVAS_OFFSET_PIXELS <= float(offset[0]) <= MAX_CANVAS_OFFSET_PIXELS
            and MIN_CANVAS_OFFSET_PIXELS <= float(offset[1]) <= MAX_CANVAS_OFFSET_PIXELS
        ):
            skipped.append(
                (
                    level_index,
                    "The required Canvas transform is outside the supported range.",
                )
            )
            continue
        reference_index, registration = registrations[level_index]
        transforms.append(
            CanvasPlanAlignmentTransform(
                level_index=level_index,
                canvas_offset_x_pixels=float(offset[0]),
                canvas_offset_y_pixels=float(offset[1]),
                reference_level_index=reference_index,
                score=registration.score,
                wall_score=registration.wall_score,
                stair_score=registration.stair_score,
            )
        )
    return transforms


def _image_fit_scale(
    image_width: int,
    image_height: int,
    viewport_size: tuple[float, float],
) -> float:
    return min(
        float(viewport_size[0]) / float(image_width),
        float(viewport_size[1]) / float(image_height),
    )
