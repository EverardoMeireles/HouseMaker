# ### Imports ###
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import cv2
import numpy as np

from housemaker.plan_metric_state import PlanMeasurementEvidence

# ### Constants ###
MAX_ANALYSIS_IMAGE_EDGE = 3_072
MAX_STRUCTURAL_LINE_COUNT = 8_000
MIN_STRUCTURAL_LINE_LENGTH = 20
MAX_MEASUREMENT_ANGLE_DEGREES = 5.0
MAX_REFINED_SPAN_CHANGE = 0.025
MIN_GEOMETRY_CHECK_SPAN = 80.0
MIN_NONUNIFORM_SPAN_CHANGE = 0.06


# ### Public results and cancellation ###
@dataclass(frozen=True)
class PlanMeasurementRefinementResult:
    measurements: tuple[PlanMeasurementEvidence, ...]
    refined_count: int
    geometry_warning: str = ""
    rejected_count: int = 0


class PlanMeasurementRefinementCancelled(RuntimeError):
    """Raised when the supplied cancellation callback returns true."""


# ### Internal line evidence ###
@dataclass(frozen=True)
class _StructuralLine:
    coordinate: float
    start: float
    end: float
    half_width: float = 0.0
    can_refine: bool = True


@dataclass(frozen=True)
class _PlanLines:
    width: int
    height: int
    vertical: tuple[_StructuralLine, ...]
    horizontal: tuple[_StructuralLine, ...]

    @property
    def snap_radius(self) -> float:
        return min(6.0, max(2.0, max(self.width, self.height) * 0.002))


@dataclass(frozen=True)
class _SupportedSpan:
    start: tuple[float, float]
    end: tuple[float, float]
    both_endpoints_supported: bool


# ### Public refinement ###
def refine_plan_measurements(
    measurements: Sequence[PlanMeasurementEvidence],
    corrected_image: np.ndarray,
    *,
    measurement_reference_image: np.ndarray | None = None,
    cancellation_check: Callable[[], bool] | None = None,
) -> PlanMeasurementRefinementResult:
    """Refine small, unambiguous endpoint errors without choosing a metric scale.

    Only approximately horizontal/vertical dimensions are refined. Long, thin
    perpendicular ink strokes support their measured coordinate; two nearby
    strokes are deliberately ambiguous, even when one is closer. All printed
    values, source coordinates and confidence levels remain unchanged.

    An optional *rectified annotated* reference allows a conservative comparison
    of independently supported spans before and after annotation removal. A raw
    perspective photograph must not be supplied as that reference. A warning is
    evidence to review/retry, never a reason to reject the corrected image.

    Cancellation callbacks returning true raise PlanMeasurementRefinementCancelled;
    exceptions raised by callbacks propagate unchanged.
    """
    _check_cancelled(cancellation_check)
    evidence = tuple(measurements)
    if not all(isinstance(item, PlanMeasurementEvidence) for item in evidence):
        raise TypeError("Plan measurements contain invalid evidence.")
    if not evidence:
        return PlanMeasurementRefinementResult((), 0)
    final_lines = _extract_structural_lines(corrected_image, cancellation_check)
    reference_lines = (
        _extract_structural_lines(measurement_reference_image, cancellation_check)
        if measurement_reference_image is not None
        else None
    )
    refined: list[PlanMeasurementEvidence] = []
    checked_spans: list[tuple[PlanMeasurementEvidence, float, float]] = []
    refined_count = 0
    for item in evidence:
        _check_cancelled(cancellation_check)
        final_span = _refine_span(
            final_lines,
            item.corrected_start_normalized,
            item.corrected_end_normalized,
        )
        updated = replace(
            item,
            corrected_start_normalized=final_span.start,
            corrected_end_normalized=final_span.end,
        )
        refined.append(updated)
        refined_count += int(updated != item)
        if reference_lines is None or not final_span.both_endpoints_supported:
            continue
        reference_span = _refine_span(
            reference_lines, item.start_normalized, item.end_normalized
        )
        if not reference_span.both_endpoints_supported:
            continue
        reference_length = _pixel_length(reference_lines, reference_span)
        final_length = _pixel_length(final_lines, final_span)
        if min(reference_length, final_length) < MIN_GEOMETRY_CHECK_SPAN:
            continue
        if any(
            _same_span(updated, previous) for previous, _ratio, _error in checked_spans
        ):
            continue
        # Bound rasterization/endpoint uncertainty before diagnosing deformation.
        uncertainty = (
            2.0 * final_lines.snap_radius / final_length
            + 2.0 * reference_lines.snap_radius / reference_length
        )
        checked_spans.append((updated, final_length / reference_length, uncertainty))
    return PlanMeasurementRefinementResult(
        measurements=tuple(refined),
        refined_count=refined_count,
        geometry_warning=_geometry_warning(checked_spans),
    )


# ### Bounded structural line extraction ###
def _extract_structural_lines(
    image: np.ndarray,
    cancellation_check: Callable[[], bool] | None,
) -> _PlanLines:
    _check_cancelled(cancellation_check)
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("Plan refinement requires a nonempty image.")
    if image.dtype != np.uint8:
        raise ValueError("Plan refinement requires an 8-bit plan image.")
    height, width = image.shape[:2]
    factor = min(1.0, MAX_ANALYSIS_IMAGE_EDGE / max(width, height))
    if factor < 1.0:
        image = cv2.resize(
            image,
            (max(1, round(width * factor)), max(1, round(height * factor))),
            interpolation=cv2.INTER_AREA,
        )
    if image.ndim == 3:
        if image.shape[2] not in (3, 4):
            raise ValueError("Plan refinement requires a grayscale, BGR or BGRA image.")
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY,
        )
    height, width = image.shape
    if int(image.max()) - int(image.min()) < 32:
        return _PlanLines(width, height, (), ())
    _threshold, ink = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    line_length = max(MIN_STRUCTURAL_LINE_LENGTH, round(max(width, height) * 0.015))
    max_stroke_width = max(3, min(7, round(max(width, height) * 0.004)))
    orientations: list[tuple[_StructuralLine, ...]] = []
    for vertical in (True, False):
        _check_cancelled(cancellation_check)
        kernel = np.ones((line_length, 1) if vertical else (1, line_length), np.uint8)
        structural_ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            structural_ink, connectivity=8
        )
        lines: list[_StructuralLine] = []
        for index in range(1, count):
            if index % 256 == 0:
                _check_cancelled(cancellation_check)
            x, y, component_width, component_height, area = stats[index]
            thickness = component_width if vertical else component_height
            length = component_height if vertical else component_width
            if length < line_length:
                continue
            lines.append(
                _StructuralLine(
                    coordinate=float(x + (component_width - 1) / 2)
                    if vertical
                    else float(y + (component_height - 1) / 2),
                    start=float(y if vertical else x),
                    end=float(
                        y + component_height - 1
                        if vertical
                        else x + component_width - 1
                    ),
                    half_width=float(thickness - 1) / 2.0,
                    can_refine=(
                        thickness <= max_stroke_width
                        and area >= 0.65 * component_width * component_height
                    ),
                )
            )
        # Very busy/noisy images are not a basis for confident snapping. Do not
        # truncate away competing lines and accidentally make an anchor unique.
        orientations.append(
            tuple(lines) if len(lines) <= MAX_STRUCTURAL_LINE_COUNT else ()
        )
    return _PlanLines(width, height, orientations[0], orientations[1])


# ### Endpoint support and geometry checks ###
def _refine_span(
    plan: _PlanLines,
    start: tuple[float, float],
    end: tuple[float, float],
) -> _SupportedSpan:
    start_px = (start[0] * plan.width, start[1] * plan.height)
    end_px = (end[0] * plan.width, end[1] * plan.height)
    dx, dy = end_px[0] - start_px[0], end_px[1] - start_px[1]
    original_length = math.hypot(dx, dy)
    tangent = math.tan(math.radians(MAX_MEASUREMENT_ANGLE_DEGREES))
    if abs(dx) > 0 and abs(dy) <= abs(dx) * tangent:
        axis, lines = 0, plan.vertical
    elif abs(dy) > 0 and abs(dx) <= abs(dy) * tangent:
        axis, lines = 1, plan.horizontal
    else:
        return _SupportedSpan(start, end, False)
    first = _supported_coordinate(start_px, axis, lines, plan.snap_radius)
    second = _supported_coordinate(end_px, axis, lines, plan.snap_radius)
    new_start, new_end = list(start_px), list(end_px)
    if first is not None:
        new_start[axis] = first
    if second is not None:
        new_end[axis] = second
    new_length = math.dist(new_start, new_end)
    if (
        original_length <= 0
        or abs(new_length / original_length - 1.0) > MAX_REFINED_SPAN_CHANGE
    ):
        return _SupportedSpan(start, end, False)
    return _SupportedSpan(
        (new_start[0] / plan.width, new_start[1] / plan.height),
        (new_end[0] / plan.width, new_end[1] / plan.height),
        first is not None and second is not None,
    )


def _supported_coordinate(
    point: tuple[float, float],
    axis: int,
    lines: tuple[_StructuralLine, ...],
    radius: float,
) -> float | None:
    guard_radius = max(6.0, radius * 2.0)
    candidates = [
        line
        for line in lines
        if abs(line.coordinate - point[axis]) <= guard_radius + line.half_width
        and line.start - 1.0 <= point[1 - axis] <= line.end + 1.0
    ]
    if len(candidates) != 1 or not candidates[0].can_refine:
        return None
    return (
        candidates[0].coordinate
        if abs(candidates[0].coordinate - point[axis]) <= radius
        else None
    )


def _pixel_length(plan: _PlanLines, span: _SupportedSpan) -> float:
    return math.hypot(
        (span.end[0] - span.start[0]) * plan.width,
        (span.end[1] - span.start[1]) * plan.height,
    )


def _same_span(first: PlanMeasurementEvidence, second: PlanMeasurementEvidence) -> bool:
    first_points = (first.corrected_start_normalized, first.corrected_end_normalized)
    second_points = (second.corrected_start_normalized, second.corrected_end_normalized)
    return any(
        all(math.dist(a, b) < 0.01 for a, b in zip(first_points, ordered))
        for ordered in (second_points, second_points[::-1])
    )


def _geometry_warning(
    spans: list[tuple[PlanMeasurementEvidence, float, float]],
) -> str:
    if len(spans) < 4:
        return ""
    ordered = sorted(spans, key=lambda item: item[1])
    lower = ordered[1]
    upper = ordered[-2]
    relative_difference = upper[1] / lower[1] - 1.0
    tolerance = max(MIN_NONUNIFORM_SPAN_CHANGE, lower[2] + upper[2])
    if relative_difference <= tolerance:
        return ""
    return (
        "Measured spans changed by different proportions after annotation removal "
        f"({relative_difference:.0%} discrepancy across {len(spans)} supported spans). "
        "Review the corrected geometry and endpoint correspondences; "
        "a single scale may not fit the whole plan."
    )


# ### Cancellation helper ###
def _check_cancelled(cancellation_check: Callable[[], bool] | None) -> None:
    if cancellation_check is not None and cancellation_check():
        raise PlanMeasurementRefinementCancelled(
            "Plan measurement refinement was cancelled."
        )
