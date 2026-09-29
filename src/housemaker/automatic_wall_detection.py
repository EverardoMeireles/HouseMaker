# ### Imports ###
from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

from .models import Edge, Vertex, VertexData

# ### Detection constants ###
_MIN_ANALYSIS_LINE_LENGTH_PX = 5.0
_AXIS_SNAP_LIMIT_DEGREES = 4.0
_GEOMETRY_EPSILON = 1e-6
_POINT_KEY_PRECISION = 6
_MINIMUM_STRUCTURAL_INK_RETENTION = 0.45
_EXISTING_WALL_COLLINEAR_TOLERANCE_PX = 0.5
_EXISTING_WALL_COLLINEAR_ANGLE_TOLERANCE_DEGREES = 0.5


# ### Public data models ###
class PlanWallDetectionError(ValueError):
    """A safe, user-displayable plan-wall detection failure."""


class PlanWallDetectionCancelled(PlanWallDetectionError):
    """Raised when an asynchronous plan-wall analysis is cancelled."""


@dataclass(frozen=True, slots=True)
class WallLineEvidence:
    """One possible wall-face line in native plan-image coordinates."""

    start: tuple[float, float]
    end: tuple[float, float]
    confidence: float

    @property
    def length(self) -> float:
        return math.dist(self.start, self.end)


@dataclass(frozen=True, slots=True)
class PlanWallAnalysis:
    """Reusable result of the comparatively expensive image-analysis stage."""

    image_width: int
    image_height: int
    line_evidence: tuple[WallLineEvidence, ...]


@dataclass(frozen=True, slots=True)
class PlanWallDetectionOptions:
    """Fast reconstruction controls expressed in native image pixels."""

    minimum_wall_separation_pixels: float = 6.0
    maximum_wall_separation_pixels: float = 48.0
    parallel_angle_tolerance_degrees: float = 7.5
    minimum_parallel_overlap_ratio: float = 0.55
    maximum_gap_bridge_pixels: float = 36.0
    endpoint_snap_distance_pixels: float = 8.0
    maximum_vertex_distance_pixels: float = 5.0
    minimum_wall_length_pixels: float = 20.0
    confidence_threshold: float = 0.45

    def __post_init__(self) -> None:
        numeric_values = (
            self.minimum_wall_separation_pixels,
            self.maximum_wall_separation_pixels,
            self.parallel_angle_tolerance_degrees,
            self.minimum_parallel_overlap_ratio,
            self.maximum_gap_bridge_pixels,
            self.endpoint_snap_distance_pixels,
            self.maximum_vertex_distance_pixels,
            self.minimum_wall_length_pixels,
            self.confidence_threshold,
        )
        if not all(math.isfinite(value) for value in numeric_values):
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
        if self.maximum_vertex_distance_pixels < 0.0:
            raise PlanWallDetectionError("Maximum vertex distance cannot be negative.")
        if self.minimum_wall_length_pixels <= 0.0:
            raise PlanWallDetectionError("Minimum wall length must be positive.")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise PlanWallDetectionError(
                "Confidence threshold must be between 0 and 1."
            )


@dataclass(frozen=True, slots=True)
class PlanWallDetectionResult:
    """One complete preview graph and the number of newly detected edges."""

    vertex_data: VertexData
    added_edge_count: int


# ### Internal data models ###
@dataclass(frozen=True, slots=True)
class _Segment:
    start: np.ndarray
    end: np.ndarray
    confidence: float

    @property
    def vector(self) -> np.ndarray:
        return self.end - self.start

    @property
    def length(self) -> float:
        return math.hypot(
            float(self.end[0] - self.start[0]),
            float(self.end[1] - self.start[1]),
        )


@dataclass(frozen=True, slots=True)
class _PairCandidate:
    first_index: int
    second_index: int
    confidence: float
    separation: float
    interior_indices: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class _WallFacePair:
    """Two detected line segments representing opposite faces of one wall."""

    first: _Segment
    second: _Segment


@dataclass(frozen=True, slots=True)
class _ExistingWallMatch:
    """One generated face's deterministic match to an existing wall edge."""

    anchor_index: int
    angle_error: float
    endpoint_distance: float
    gap: float
    overlap: float


@dataclass(slots=True)
class _VertexRepresentative:
    """One shared coordinate inferred from nearby generated endpoints."""

    point: np.ndarray
    source_points: list[np.ndarray]
    incident_segments: list[_Segment]
    existing_vertex_id: int | None = None


# ### Public analysis API ###
def analyze_plan_wall_image(
    image_path: str | Path,
    cancellation_check: Callable[[], bool] | None = None,
) -> PlanWallAnalysis:
    """Load and analyse a corrected plan image without mutating project data."""

    _raise_if_cancelled(cancellation_check)
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise PlanWallDetectionError(f"Plan image was not found: {path}")
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    except (OSError, cv2.error) as error:
        raise PlanWallDetectionError(
            f"The plan image could not be read: {error}"
        ) from error
    if image is None:
        raise PlanWallDetectionError("The selected file is not a readable plan image.")
    _raise_if_cancelled(cancellation_check)
    result = analyze_plan_wall_evidence(image, cancellation_check)
    _raise_if_cancelled(cancellation_check)
    return result


def analyze_plan_wall_evidence(
    image: np.ndarray,
    cancellation_check: Callable[[], bool] | None = None,
) -> PlanWallAnalysis:
    """Find reusable straight-line evidence in a corrected black-and-white plan.

    This is the expensive stage and is intended to run once per image revision.
    Slider changes should call :func:`reconstruct_plan_walls` with the returned
    analysis instead of analysing the image again.
    """

    grayscale = _validated_grayscale_image(image)
    normalized = _normalized_binary_plan(grayscale)
    structural = _structural_binary_plan(normalized)
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected, _widths, _precisions, _nfa = detector.detect(normalized)
    if detected is None:
        return PlanWallAnalysis(
            image_width=int(grayscale.shape[1]),
            image_height=int(grayscale.shape[0]),
            line_evidence=(),
        )

    minimum_dimension = float(min(grayscale.shape))
    evidence: list[WallLineEvidence] = []
    for line_index, coordinates in enumerate(detected.reshape(-1, 4)):
        if line_index % 128 == 0:
            _raise_if_cancelled(cancellation_check)
        start = np.asarray(coordinates[:2], dtype=np.float64)
        end = np.asarray(coordinates[2:], dtype=np.float64)
        length = float(np.linalg.norm(end - start))
        if length < _MIN_ANALYSIS_LINE_LENGTH_PX:
            continue
        start, end = _canonical_endpoints(start, end)
        confidence = _line_image_confidence(
            normalized,
            structural,
            start,
            end,
            minimum_dimension,
        )
        evidence.append(
            WallLineEvidence(
                start=(float(start[0]), float(start[1])),
                end=(float(end[0]), float(end[1])),
                confidence=confidence,
            )
        )

    return PlanWallAnalysis(
        image_width=int(grayscale.shape[1]),
        image_height=int(grayscale.shape[0]),
        line_evidence=_deduplicate_analysis_evidence(evidence),
    )


# ### Public reconstruction API ###
def reconstruct_plan_walls(
    analysis: PlanWallAnalysis,
    options: PlanWallDetectionOptions | None = None,
    existing_vertex_data: VertexData | None = None,
) -> PlanWallDetectionResult:
    """Convert analysed line evidence into a deterministic wall-face graph.

    Existing vertices and edges are treated as immutable anchors. New endpoints
    can snap to those vertices, but existing IDs, coordinates, edge records, and
    ordering are preserved.
    """

    resolved_options = options or PlanWallDetectionOptions()
    _validate_analysis(analysis)
    result = _validated_existing_graph(
        existing_vertex_data,
        analysis.image_width,
        analysis.image_height,
    )
    existing_vertex_ids = frozenset(vertex.id for vertex in result.vertices)
    segments = _segments_from_analysis(analysis, resolved_options)
    segments = _merge_collinear_segments(segments, resolved_options)
    wall_face_pairs = _select_wall_face_pairs(
        segments,
        resolved_options,
        result,
    )
    wall_face_pairs = _align_wall_face_pairs_to_existing_walls(
        wall_face_pairs,
        result,
        resolved_options,
    )
    paired_face_segments = [
        segment
        for pair in wall_face_pairs
        for segment in (pair.first, pair.second)
    ]
    preliminary_graph = result.clone()
    _append_segments_to_graph(
        preliminary_graph,
        _node_segments(
            _deduplicate_segments(paired_face_segments),
            resolved_options,
        ),
        analysis.image_width,
        analysis.image_height,
        resolved_options,
    )
    wall_face_pairs, end_connectors = _close_wall_face_pair_ends(
        wall_face_pairs,
        result,
        resolved_options,
        endpoint_topology=preliminary_graph,
    )
    paired_segments = [
        segment
        for pair in wall_face_pairs
        for segment in (pair.first, pair.second)
    ]
    paired_segments.extend(end_connectors)
    paired_segments = _deduplicate_segments(paired_segments)
    pieces = _node_segments(paired_segments, resolved_options)
    _append_segments_to_graph(
        result,
        pieces,
        analysis.image_width,
        analysis.image_height,
        resolved_options,
    )
    _validate_vertex_graph(
        result,
        analysis.image_width,
        analysis.image_height,
        permitted_out_of_bounds_vertex_ids=existing_vertex_ids,
    )
    existing_edge_count = len(existing_vertex_data.edges) if existing_vertex_data else 0
    return PlanWallDetectionResult(
        vertex_data=result,
        added_edge_count=len(result.edges) - existing_edge_count,
    )


def detect_plan_walls(
    image: np.ndarray,
    options: PlanWallDetectionOptions | None = None,
    existing_vertex_data: VertexData | None = None,
) -> PlanWallDetectionResult:
    """Analyse an image and reconstruct its paired wall faces in one call."""

    return reconstruct_plan_walls(
        analyze_plan_wall_evidence(image),
        options=options,
        existing_vertex_data=existing_vertex_data,
    )


def _raise_if_cancelled(
    cancellation_check: Callable[[], bool] | None,
) -> None:
    if cancellation_check is not None and cancellation_check():
        raise PlanWallDetectionCancelled("Plan wall detection was cancelled.")


# ### Image validation and normalization ###
def _validated_grayscale_image(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise PlanWallDetectionError("The architectural plan must be a NumPy array.")
    if image.dtype != np.uint8:
        raise PlanWallDetectionError(
            "The architectural plan must use the uint8 data type."
        )
    if image.ndim != 2:
        raise PlanWallDetectionError(
            "The architectural plan must be a two-dimensional image."
        )
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise PlanWallDetectionError("The architectural plan cannot be empty.")
    return image


def _normalized_binary_plan(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (3, 3), 0.0)
    _threshold, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )
    if float(np.mean(binary)) < 127.5:
        binary = cv2.bitwise_not(binary)
    return binary


def _structural_binary_plan(binary: np.ndarray) -> np.ndarray:
    """Suppress resolution-relative thin annotations while retaining walls."""

    short_edge = min(binary.shape)
    requested_kernel_size = 2 if short_edge < 500 else 3 if short_edge < 900 else 4
    ink = cv2.bitwise_not(binary)
    ink_count = int(np.count_nonzero(ink))
    if ink_count == 0:
        return binary.copy()
    for kernel_size in range(requested_kernel_size, 1, -1):
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        structural_ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
        retained_fraction = float(np.count_nonzero(structural_ink)) / ink_count
        if retained_fraction >= _MINIMUM_STRUCTURAL_INK_RETENTION:
            return cv2.bitwise_not(structural_ink)
    return binary.copy()


def _line_image_confidence(
    binary: np.ndarray,
    structural_binary: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    minimum_dimension: float,
) -> float:
    vector = end - start
    length = float(np.linalg.norm(vector))
    sample_count = max(2, math.ceil(length) + 1)
    positions = np.linspace(0.0, 1.0, sample_count)
    points = start[None, :] + positions[:, None] * vector[None, :]
    normal = np.asarray((-vector[1], vector[0]), dtype=np.float64) / length
    ink_samples: list[np.ndarray] = []
    structural_ink_samples: list[np.ndarray] = []
    for offset in (-2.0, -1.0, 0.0, 1.0, 2.0):
        offset_points = points + normal[None, :] * offset
        x_values = np.clip(
            np.rint(offset_points[:, 0]).astype(np.int32),
            0,
            binary.shape[1] - 1,
        )
        y_values = np.clip(
            np.rint(offset_points[:, 1]).astype(np.int32),
            0,
            binary.shape[0] - 1,
        )
        ink_samples.append(binary[y_values, x_values] < 128)
        structural_ink_samples.append(structural_binary[y_values, x_values] < 128)
    ink_fraction = float(np.mean(np.logical_or.reduce(ink_samples)))
    structural_ink_fraction = float(
        np.mean(np.logical_or.reduce(structural_ink_samples))
    )
    reference_length = max(12.0, minimum_dimension * 0.15)
    length_score = min(1.0, length / reference_length)
    base_confidence = 0.7 * ink_fraction + 0.3 * length_score
    structural_multiplier = 0.3 + 0.7 * structural_ink_fraction
    return float(np.clip(base_confidence * structural_multiplier, 0.0, 1.0))


def _deduplicate_analysis_evidence(
    evidence: Iterable[WallLineEvidence],
) -> tuple[WallLineEvidence, ...]:
    ordered = sorted(
        evidence,
        key=lambda item: (
            round(item.start[0], 3),
            round(item.start[1], 3),
            round(item.end[0], 3),
            round(item.end[1], 3),
            -item.confidence,
        ),
    )
    kept: list[WallLineEvidence] = []
    for candidate in ordered:
        duplicate_index = next(
            (
                index
                for index, current in enumerate(kept)
                if math.dist(candidate.start, current.start) <= 1.0
                and math.dist(candidate.end, current.end) <= 1.0
            ),
            None,
        )
        if duplicate_index is None:
            kept.append(candidate)
            continue
        if candidate.confidence > kept[duplicate_index].confidence:
            kept[duplicate_index] = candidate
    return tuple(kept)


# ### Analysis and graph validation ###
def _validate_analysis(analysis: PlanWallAnalysis) -> None:
    if not isinstance(analysis, PlanWallAnalysis):
        raise PlanWallDetectionError("Wall reconstruction requires a PlanWallAnalysis.")
    if analysis.image_width <= 0 or analysis.image_height <= 0:
        raise PlanWallDetectionError("Wall analysis image dimensions must be positive.")
    for evidence in analysis.line_evidence:
        values = (*evidence.start, *evidence.end, evidence.confidence)
        if not all(math.isfinite(value) for value in values):
            raise PlanWallDetectionError(
                "Wall line evidence must contain finite values."
            )
        if not 0.0 <= evidence.confidence <= 1.0:
            raise PlanWallDetectionError(
                "Wall line confidence must be between 0 and 1."
            )


def _validated_existing_graph(
    existing: VertexData | None,
    image_width: int,
    image_height: int,
) -> VertexData:
    result = existing.clone() if existing is not None else VertexData()
    _validate_vertex_graph(
        result,
        image_width,
        image_height,
        permitted_out_of_bounds_vertex_ids={
            vertex.id for vertex in result.vertices
        },
    )
    result.normalize_next_vertex_id()
    return result


def _validate_vertex_graph(
    vertex_data: VertexData,
    image_width: int,
    image_height: int,
    *,
    permitted_out_of_bounds_vertex_ids: Iterable[int] = (),
) -> None:
    permitted_out_of_bounds_ids = set(permitted_out_of_bounds_vertex_ids)
    vertex_ids: set[int] = set()
    for vertex in vertex_data.vertices:
        if vertex.id <= 0 or vertex.id in vertex_ids:
            raise PlanWallDetectionError("Vertex IDs must be unique positive integers.")
        if not math.isfinite(vertex.x) or not math.isfinite(vertex.y):
            raise PlanWallDetectionError("Vertex coordinates must be finite.")
        if (
            vertex.id not in permitted_out_of_bounds_ids
            and not 0.0 <= vertex.x <= image_width - 1
        ):
            raise PlanWallDetectionError(
                "Vertex X coordinate is outside the plan image."
            )
        if (
            vertex.id not in permitted_out_of_bounds_ids
            and not 0.0 <= vertex.y <= image_height - 1
        ):
            raise PlanWallDetectionError(
                "Vertex Y coordinate is outside the plan image."
            )
        vertex_ids.add(vertex.id)

    edge_keys: set[tuple[int, int]] = set()
    for edge in vertex_data.edges:
        if (
            edge.start_vertex_id not in vertex_ids
            or edge.end_vertex_id not in vertex_ids
        ):
            raise PlanWallDetectionError("Every edge must reference existing vertices.")
        if edge.start_vertex_id == edge.end_vertex_id:
            raise PlanWallDetectionError("Self-referencing edges are not valid walls.")
        key = tuple(sorted((edge.start_vertex_id, edge.end_vertex_id)))
        if key in edge_keys:
            raise PlanWallDetectionError("Duplicate wall edges are not allowed.")
        edge_keys.add(key)


# ### Evidence normalization ###
def _segments_from_analysis(
    analysis: PlanWallAnalysis,
    options: PlanWallDetectionOptions,
) -> list[_Segment]:
    segments: list[_Segment] = []
    minimum_input_length = max(
        _MIN_ANALYSIS_LINE_LENGTH_PX,
        options.minimum_wall_length_pixels * 0.2,
    )
    for evidence in analysis.line_evidence:
        start = np.asarray(evidence.start, dtype=np.float64)
        end = np.asarray(evidence.end, dtype=np.float64)
        start = _clipped_point(start, analysis.image_width, analysis.image_height)
        end = _clipped_point(end, analysis.image_width, analysis.image_height)
        start, end = _canonical_endpoints(start, end)
        segment = _straightened_segment(
            _Segment(start=start, end=end, confidence=evidence.confidence),
            options.parallel_angle_tolerance_degrees,
        )
        if segment.length >= minimum_input_length:
            segments.append(segment)
    return sorted(segments, key=_segment_sort_key)


def _straightened_segment(
    segment: _Segment,
    angle_tolerance_degrees: float,
) -> _Segment:
    angle = _segment_angle(segment)
    snap_limit = math.radians(
        min(_AXIS_SNAP_LIMIT_DEGREES, angle_tolerance_degrees * 0.75)
    )
    horizontal_error = min(angle, math.pi - angle)
    vertical_error = abs(angle - math.pi / 2.0)
    start = segment.start.copy()
    end = segment.end.copy()
    if horizontal_error <= snap_limit:
        center_y = (start[1] + end[1]) / 2.0
        start[1] = center_y
        end[1] = center_y
    elif vertical_error <= snap_limit:
        center_x = (start[0] + end[0]) / 2.0
        start[0] = center_x
        end[0] = center_x
    start, end = _canonical_endpoints(start, end)
    return _Segment(start=start, end=end, confidence=segment.confidence)


# ### Collinear line merging ###
def _merge_collinear_segments(
    segments: Sequence[_Segment],
    options: PlanWallDetectionOptions,
) -> list[_Segment]:
    offset_tolerance = min(
        4.5,
        max(1.25, options.minimum_wall_separation_pixels * 0.8),
    )
    horizontal = [segment for segment in segments if _is_horizontal(segment)]
    vertical = [segment for segment in segments if _is_vertical(segment)]
    oblique = [
        segment
        for segment in segments
        if not _is_horizontal(segment) and not _is_vertical(segment)
    ]
    merged = [
        *_merge_axis_aligned_segments(
            horizontal,
            horizontal=True,
            offset_tolerance=offset_tolerance,
            maximum_gap=options.maximum_gap_bridge_pixels,
        ),
        *_merge_axis_aligned_segments(
            vertical,
            horizontal=False,
            offset_tolerance=offset_tolerance,
            maximum_gap=options.maximum_gap_bridge_pixels,
        ),
        *_merge_oblique_segments(oblique, options, offset_tolerance),
    ]
    return sorted(merged, key=_segment_sort_key)


def _merge_axis_aligned_segments(
    segments: Sequence[_Segment],
    *,
    horizontal: bool,
    offset_tolerance: float,
    maximum_gap: float,
) -> list[_Segment]:
    if not segments:
        return []
    normal_axis = 1 if horizontal else 0
    tangent_axis = 0 if horizontal else 1
    normal_groups: list[list[_Segment]] = []
    group_centers: list[float] = []
    for segment in sorted(
        segments,
        key=lambda item: (
            float((item.start[normal_axis] + item.end[normal_axis]) / 2.0),
            float(item.start[tangent_axis]),
        ),
    ):
        normal = float((segment.start[normal_axis] + segment.end[normal_axis]) / 2.0)
        matching_indices = [
            index
            for index, center in enumerate(group_centers)
            if abs(center - normal) <= offset_tolerance
        ]
        if not matching_indices:
            normal_groups.append([segment])
            group_centers.append(normal)
            continue
        group_index = min(
            matching_indices,
            key=lambda index: (abs(group_centers[index] - normal), index),
        )
        normal_groups[group_index].append(segment)
        weights = [item.length for item in normal_groups[group_index]]
        normals = [
            float((item.start[normal_axis] + item.end[normal_axis]) / 2.0)
            for item in normal_groups[group_index]
        ]
        group_centers[group_index] = float(np.average(normals, weights=weights))

    output: list[_Segment] = []
    for center, group in zip(group_centers, normal_groups, strict=True):
        ordered = sorted(group, key=lambda item: float(item.start[tangent_axis]))
        runs: list[list[_Segment]] = []
        run_end = -math.inf
        for segment in ordered:
            start_value = float(segment.start[tangent_axis])
            end_value = float(segment.end[tangent_axis])
            if not runs or start_value - run_end > maximum_gap:
                runs.append([segment])
                run_end = end_value
                continue
            runs[-1].append(segment)
            run_end = max(run_end, end_value)
        for run in runs:
            start_value = min(float(item.start[tangent_axis]) for item in run)
            end_value = max(float(item.end[tangent_axis]) for item in run)
            total_length = sum(item.length for item in run)
            confidence = sum(item.confidence * item.length for item in run) / max(
                total_length,
                _GEOMETRY_EPSILON,
            )
            if horizontal:
                start = np.asarray((start_value, center), dtype=np.float64)
                end = np.asarray((end_value, center), dtype=np.float64)
            else:
                start = np.asarray((center, start_value), dtype=np.float64)
                end = np.asarray((center, end_value), dtype=np.float64)
            output.append(_Segment(start=start, end=end, confidence=confidence))
    return output


def _merge_oblique_segments(
    segments: Sequence[_Segment],
    options: PlanWallDetectionOptions,
    offset_tolerance: float,
) -> list[_Segment]:
    merged: list[_Segment] = []
    angle_tolerance = math.radians(options.parallel_angle_tolerance_degrees)
    for segment in sorted(segments, key=_segment_sort_key):
        possible: list[tuple[float, float, int]] = []
        for index, current in enumerate(merged):
            angle_error = _angle_difference(
                _segment_angle(segment),
                _segment_angle(current),
            )
            if angle_error > angle_tolerance:
                continue
            offset, gap = _parallel_offset_and_gap(segment, current)
            if offset <= offset_tolerance and gap <= options.maximum_gap_bridge_pixels:
                possible.append((gap, offset, index))
        if not possible:
            merged.append(segment)
            continue
        _gap, _offset, merge_index = min(possible)
        merged[merge_index] = _combined_collinear_segment(
            merged[merge_index],
            segment,
        )
    return merged


def _parallel_offset_and_gap(
    first: _Segment,
    second: _Segment,
) -> tuple[float, float]:
    direction = _unit_direction(first)
    normal = np.asarray((-direction[1], direction[0]), dtype=np.float64)
    first_center = (first.start + first.end) / 2.0
    second_center = (second.start + second.end) / 2.0
    offset = abs(float(np.dot(second_center - first_center, normal)))
    first_interval = _projection_interval(first, direction)
    second_interval = _projection_interval(second, direction)
    gap = max(
        0.0,
        second_interval[0] - first_interval[1],
        first_interval[0] - second_interval[1],
    )
    return offset, gap


def _combined_collinear_segment(first: _Segment, second: _Segment) -> _Segment:
    first_direction = _unit_direction(first)
    second_direction = _unit_direction(second)
    if float(np.dot(first_direction, second_direction)) < 0.0:
        second_direction = -second_direction
    weighted_direction = (
        first_direction * first.length + second_direction * second.length
    )
    direction = weighted_direction / np.linalg.norm(weighted_direction)
    normal = np.asarray((-direction[1], direction[0]), dtype=np.float64)
    total_length = first.length + second.length
    normal_offset = (
        float(np.dot((first.start + first.end) / 2.0, normal)) * first.length
        + float(np.dot((second.start + second.end) / 2.0, normal)) * second.length
    ) / total_length
    projections = [
        float(np.dot(point, direction))
        for point in (first.start, first.end, second.start, second.end)
    ]
    start = direction * min(projections) + normal * normal_offset
    end = direction * max(projections) + normal * normal_offset
    confidence = (
        first.confidence * first.length + second.confidence * second.length
    ) / total_length
    start, end = _canonical_endpoints(start, end)
    return _Segment(start=start, end=end, confidence=confidence)


# ### Parallel wall-face pairing ###
def _select_wall_face_pairs(
    segments: Sequence[_Segment],
    options: PlanWallDetectionOptions,
    anchors: VertexData,
) -> list[_WallFacePair]:
    candidates: list[_PairCandidate] = []
    tolerance_radians = math.radians(options.parallel_angle_tolerance_degrees)
    angles = tuple(_segment_angle(segment) for segment in segments)
    for first_index, second_index in _parallel_candidate_index_pairs(
        segments,
        angles,
        tolerance_radians,
        options.minimum_wall_length_pixels,
    ):
        first = segments[first_index]
        second = segments[second_index]
        angle_error = _angle_difference(
            angles[first_index],
            angles[second_index],
        )
        if angle_error > tolerance_radians:
            continue
        separation, overlap_ratio = _parallel_separation_and_overlap(
            first,
            second,
        )
        if not (
            options.minimum_wall_separation_pixels
            <= separation
            <= options.maximum_wall_separation_pixels
        ):
            continue
        if overlap_ratio < options.minimum_parallel_overlap_ratio:
            continue
        confidence = _pair_confidence(
            first,
            second,
            separation,
            overlap_ratio,
            angle_error,
            tolerance_radians,
            options,
            anchors,
        )
        interior_indices = _interior_parallel_indices(
            first_index,
            second_index,
            segments,
            angles,
            tolerance_radians,
            options.minimum_parallel_overlap_ratio,
        )
        confidence = min(1.0, confidence + min(0.2, 0.11 * len(interior_indices)))
        if confidence < options.confidence_threshold:
            continue
        candidates.append(
            _PairCandidate(
                first_index=first_index,
                second_index=second_index,
                confidence=confidence,
                separation=separation,
                interior_indices=interior_indices,
            )
        )

    used_indices: set[int] = set()
    selected: list[_WallFacePair] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            -item.confidence,
            -len(item.interior_indices),
            -item.separation,
            _segment_sort_key(segments[item.first_index]),
            _segment_sort_key(segments[item.second_index]),
        ),
    ):
        claimed_indices = {
            candidate.first_index,
            candidate.second_index,
            *candidate.interior_indices,
        }
        if claimed_indices & used_indices:
            continue
        used_indices.update(claimed_indices)
        selected.append(
            _WallFacePair(
                first=segments[candidate.first_index],
                second=segments[candidate.second_index],
            )
        )
    return sorted(
        selected,
        key=lambda pair: (
            _segment_sort_key(pair.first),
            _segment_sort_key(pair.second),
        ),
    )


def _interior_parallel_indices(
    first_index: int,
    second_index: int,
    segments: Sequence[_Segment],
    angles: Sequence[float],
    tolerance_radians: float,
    minimum_overlap_ratio: float,
) -> tuple[int, ...]:
    first = segments[first_index]
    second = segments[second_index]
    direction_x = float(first.end[0] - first.start[0]) / first.length
    direction_y = float(first.end[1] - first.start[1]) / first.length
    normal_x = -direction_y
    normal_y = direction_x
    first_center_x = float(first.start[0] + first.end[0]) / 2.0
    first_center_y = float(first.start[1] + first.end[1]) / 2.0
    second_center_x = float(second.start[0] + second.end[0]) / 2.0
    second_center_y = float(second.start[1] + second.end[1]) / 2.0
    signed_pair_separation = (second_center_x - first_center_x) * normal_x + (
        second_center_y - first_center_y
    ) * normal_y
    lower_offset, upper_offset = sorted((0.0, signed_pair_separation))
    interior: list[int] = []
    for index, candidate in enumerate(segments):
        if index in (first_index, second_index):
            continue
        if _angle_difference(angles[first_index], angles[index]) > tolerance_radians:
            continue
        candidate_center_x = float(candidate.start[0] + candidate.end[0]) / 2.0
        candidate_center_y = float(candidate.start[1] + candidate.end[1]) / 2.0
        signed_offset = (candidate_center_x - first_center_x) * normal_x + (
            candidate_center_y - first_center_y
        ) * normal_y
        if not lower_offset + 0.5 < signed_offset < upper_offset - 0.5:
            continue
        _separation, first_overlap = _parallel_separation_and_overlap(
            first,
            candidate,
        )
        _separation, second_overlap = _parallel_separation_and_overlap(
            second,
            candidate,
        )
        if min(first_overlap, second_overlap) < minimum_overlap_ratio:
            continue
        interior.append(index)
    return tuple(interior)


def _parallel_candidate_index_pairs(
    segments: Sequence[_Segment],
    angles: Sequence[float],
    tolerance_radians: float,
    minimum_length: float,
) -> list[tuple[int, int]]:
    bucket_count = max(1, math.ceil(math.pi / tolerance_radians))
    bucket_width = math.pi / bucket_count
    buckets: dict[int, list[int]] = {}
    for index, (segment, angle) in enumerate(zip(segments, angles, strict=True)):
        if segment.length < minimum_length:
            continue
        bucket = min(bucket_count - 1, int(angle / bucket_width))
        buckets.setdefault(bucket, []).append(index)

    pairs: set[tuple[int, int]] = set()
    for bucket, indices in buckets.items():
        neighbor_buckets = {
            (bucket - 1) % bucket_count,
            bucket,
            (bucket + 1) % bucket_count,
        }
        for first_index in indices:
            for neighbor_bucket in neighbor_buckets:
                for second_index in buckets.get(neighbor_bucket, ()):
                    if second_index <= first_index:
                        continue
                    pairs.add((first_index, second_index))
    return sorted(pairs)


def _parallel_separation_and_overlap(
    first: _Segment,
    second: _Segment,
) -> tuple[float, float]:
    inverse_length = 1.0 / first.length
    direction_x = float(first.end[0] - first.start[0]) * inverse_length
    direction_y = float(first.end[1] - first.start[1]) * inverse_length
    first_center_x = float(first.start[0] + first.end[0]) / 2.0
    first_center_y = float(first.start[1] + first.end[1]) / 2.0
    second_center_x = float(second.start[0] + second.end[0]) / 2.0
    second_center_y = float(second.start[1] + second.end[1]) / 2.0
    separation = abs(
        (second_center_x - first_center_x) * -direction_y
        + (second_center_y - first_center_y) * direction_x
    )
    first_interval = _scalar_projection_interval(
        first,
        direction_x,
        direction_y,
    )
    second_interval = _scalar_projection_interval(
        second,
        direction_x,
        direction_y,
    )
    overlap = max(
        0.0,
        min(first_interval[1], second_interval[1])
        - max(first_interval[0], second_interval[0]),
    )
    longer_length = max(
        first_interval[1] - first_interval[0],
        second_interval[1] - second_interval[0],
    )
    ratio = overlap / longer_length if longer_length > _GEOMETRY_EPSILON else 0.0
    return separation, ratio


def _scalar_projection_interval(
    segment: _Segment,
    direction_x: float,
    direction_y: float,
) -> tuple[float, float]:
    start_value = (
        float(segment.start[0]) * direction_x + float(segment.start[1]) * direction_y
    )
    end_value = (
        float(segment.end[0]) * direction_x + float(segment.end[1]) * direction_y
    )
    return min(start_value, end_value), max(start_value, end_value)


def _pair_confidence(
    first: _Segment,
    second: _Segment,
    separation: float,
    overlap_ratio: float,
    angle_error: float,
    tolerance_radians: float,
    options: PlanWallDetectionOptions,
    anchors: VertexData,
) -> float:
    evidence_score = min(first.confidence, second.confidence)
    overlap_score = min(1.0, overlap_ratio)
    angle_score = 1.0 - angle_error / max(tolerance_radians, _GEOMETRY_EPSILON)
    separation_range = max(
        _GEOMETRY_EPSILON,
        options.maximum_wall_separation_pixels - options.minimum_wall_separation_pixels,
    )
    separation_score = 1.0 - 0.15 * (
        (separation - options.minimum_wall_separation_pixels) / separation_range
    )
    base_score = (
        evidence_score
        * (0.55 + 0.3 * overlap_score + 0.15 * angle_score)
        * separation_score
    )
    anchor_score = max(
        _anchor_alignment(first, anchors, options),
        _anchor_alignment(second, anchors, options),
    )
    return float(np.clip(base_score + 0.12 * anchor_score, 0.0, 1.0))


def _anchor_alignment(
    segment: _Segment,
    anchors: VertexData,
    options: PlanWallDetectionOptions,
) -> float:
    vertices = {vertex.id: vertex for vertex in anchors.vertices}
    best = 0.0
    tolerance = math.radians(options.parallel_angle_tolerance_degrees)
    for edge in anchors.edges:
        start_vertex = vertices[edge.start_vertex_id]
        end_vertex = vertices[edge.end_vertex_id]
        anchor = _Segment(
            start=np.asarray((start_vertex.x, start_vertex.y), dtype=np.float64),
            end=np.asarray((end_vertex.x, end_vertex.y), dtype=np.float64),
            confidence=1.0,
        )
        if anchor.length <= _GEOMETRY_EPSILON:
            continue
        angle_error = _angle_difference(
            _segment_angle(segment),
            _segment_angle(anchor),
        )
        if angle_error > tolerance:
            continue
        offset, gap = _parallel_offset_and_gap(segment, anchor)
        distance = math.hypot(offset, gap)
        influence_distance = max(
            options.maximum_wall_separation_pixels * 2.0,
            options.endpoint_snap_distance_pixels,
            1.0,
        )
        best = max(best, 1.0 - min(1.0, distance / influence_distance))
    return best


# ### Existing-wall anchor alignment ###
def _align_wall_face_pairs_to_existing_walls(
    wall_face_pairs: Sequence[_WallFacePair],
    existing_graph: VertexData,
    options: PlanWallDetectionOptions,
) -> list[_WallFacePair]:
    """Align detected wall pairs with immutable existing wall edges.

    Individual faces use a tolerance just below the minimum wall thickness and
    cannot share an existing edge. The full user-selected vertex distance is
    allowed only when both faces coherently match two distinct existing faces.
    This removes shifted duplicates without collapsing a genuine opposite face.
    """

    full_tolerance = options.maximum_vertex_distance_pixels
    safe_tolerance = min(
        full_tolerance,
        max(0.0, options.minimum_wall_separation_pixels - 0.25),
    )
    if full_tolerance <= _GEOMETRY_EPSILON or not existing_graph.edges:
        return list(wall_face_pairs)

    vertices_by_id = {vertex.id: vertex for vertex in existing_graph.vertices}
    anchors = sorted(
        (
            _segment_from_vertices(
                vertices_by_id[edge.start_vertex_id],
                vertices_by_id[edge.end_vertex_id],
            )
            for edge in existing_graph.edges
        ),
        key=_segment_sort_key,
    )
    anchors = [anchor for anchor in anchors if anchor.length > _GEOMETRY_EPSILON]
    if not anchors:
        return list(wall_face_pairs)

    return [
        _aligned_wall_face_pair_to_existing_walls(
            pair,
            anchors,
            full_tolerance,
            safe_tolerance,
            options,
        )
        for pair in wall_face_pairs
    ]


def _aligned_wall_face_pair_to_existing_walls(
    pair: _WallFacePair,
    anchors: Sequence[_Segment],
    full_tolerance: float,
    safe_tolerance: float,
    options: PlanWallDetectionOptions,
) -> _WallFacePair:
    """Align one wall pair while preserving its two distinct faces."""

    coherent_match = _coherent_existing_wall_pair_match(
        pair,
        anchors,
        full_tolerance,
        safe_tolerance,
        options,
    )
    if coherent_match is not None:
        first_match, second_match = coherent_match
        return _WallFacePair(
            first=_project_segment_onto_existing_wall(
                pair.first,
                anchors[first_match.anchor_index],
                options.endpoint_snap_distance_pixels,
            ),
            second=_project_segment_onto_existing_wall(
                pair.second,
                anchors[second_match.anchor_index],
                options.endpoint_snap_distance_pixels,
            ),
        )

    first_match, second_match = _best_injective_existing_wall_matches(
        pair,
        anchors,
        safe_tolerance,
        options,
    )
    return _WallFacePair(
        first=(
            pair.first
            if first_match is None
            else _project_segment_onto_existing_wall(
                pair.first,
                anchors[first_match.anchor_index],
                options.endpoint_snap_distance_pixels,
            )
        ),
        second=(
            pair.second
            if second_match is None
            else _project_segment_onto_existing_wall(
                pair.second,
                anchors[second_match.anchor_index],
                options.endpoint_snap_distance_pixels,
            )
        ),
    )


def _coherent_existing_wall_pair_match(
    pair: _WallFacePair,
    anchors: Sequence[_Segment],
    full_tolerance: float,
    safe_tolerance: float,
    options: PlanWallDetectionOptions,
) -> tuple[_ExistingWallMatch, _ExistingWallMatch] | None:
    """Match both generated faces to one translated existing wall pair."""

    if full_tolerance <= safe_tolerance + _GEOMETRY_EPSILON:
        return None
    first_matches = _existing_wall_matches(
        pair.first,
        anchors,
        full_tolerance,
        options,
    )
    second_matches = _existing_wall_matches(
        pair.second,
        anchors,
        full_tolerance,
        options,
    )
    assignments: list[
        tuple[tuple[object, ...], _ExistingWallMatch, _ExistingWallMatch]
    ] = []
    for first_match in first_matches:
        for second_match in second_matches:
            if first_match.anchor_index == second_match.anchor_index:
                continue
            if (
                first_match.endpoint_distance <= safe_tolerance
                and second_match.endpoint_distance <= safe_tolerance
            ):
                continue
            first_anchor = anchors[first_match.anchor_index]
            second_anchor = anchors[second_match.anchor_index]
            if not _wall_pair_assignment_is_coherent(
                pair,
                first_anchor,
                second_anchor,
                safe_tolerance,
                options,
            ):
                continue
            score = (
                round(
                    first_match.endpoint_distance
                    + second_match.endpoint_distance,
                    _POINT_KEY_PRECISION,
                ),
                round(
                    first_match.angle_error + second_match.angle_error,
                    _POINT_KEY_PRECISION,
                ),
                round(first_match.gap + second_match.gap, _POINT_KEY_PRECISION),
                -round(
                    first_match.overlap + second_match.overlap,
                    _POINT_KEY_PRECISION,
                ),
                _segment_sort_key(first_anchor),
                _segment_sort_key(second_anchor),
            )
            assignments.append((score, first_match, second_match))
    if not assignments:
        return None
    _score, first_match, second_match = min(
        assignments,
        key=lambda item: item[0],
    )
    return first_match, second_match


def _existing_wall_matches(
    segment: _Segment,
    anchors: Sequence[_Segment],
    tolerance: float,
    options: PlanWallDetectionOptions,
) -> list[_ExistingWallMatch]:
    """Return every nearby collinear existing edge in deterministic order."""

    angle_tolerance = math.radians(options.parallel_angle_tolerance_degrees)
    matches: list[_ExistingWallMatch] = []
    for anchor_index, anchor in enumerate(anchors):
        angle_error = _angle_difference(
            _segment_angle(segment),
            _segment_angle(anchor),
        )
        if angle_error > angle_tolerance:
            continue
        endpoint_distance = max(
            _point_line_distance(segment.start, anchor),
            _point_line_distance(segment.end, anchor),
        )
        if endpoint_distance > tolerance + _GEOMETRY_EPSILON:
            continue
        gap, overlap = _segment_gap_and_overlap_on_anchor(segment, anchor)
        if gap > options.endpoint_snap_distance_pixels + _GEOMETRY_EPSILON:
            continue
        matches.append(
            _ExistingWallMatch(
                anchor_index=anchor_index,
                angle_error=angle_error,
                endpoint_distance=endpoint_distance,
                gap=gap,
                overlap=overlap,
            )
        )
    return sorted(
        matches,
        key=lambda match: (
            round(match.endpoint_distance, _POINT_KEY_PRECISION),
            round(match.angle_error, _POINT_KEY_PRECISION),
            round(match.gap, _POINT_KEY_PRECISION),
            -round(match.overlap, _POINT_KEY_PRECISION),
            _segment_sort_key(anchors[match.anchor_index]),
        ),
    )


def _best_injective_existing_wall_matches(
    pair: _WallFacePair,
    anchors: Sequence[_Segment],
    tolerance: float,
    options: PlanWallDetectionOptions,
) -> tuple[_ExistingWallMatch | None, _ExistingWallMatch | None]:
    """Match nearby faces without assigning both to the same wall edge."""

    first_options: list[_ExistingWallMatch | None] = [
        None,
        *_existing_wall_matches(pair.first, anchors, tolerance, options),
    ]
    second_options: list[_ExistingWallMatch | None] = [
        None,
        *_existing_wall_matches(pair.second, anchors, tolerance, options),
    ]
    assignments: list[
        tuple[
            tuple[object, ...],
            _ExistingWallMatch | None,
            _ExistingWallMatch | None,
        ]
    ] = []
    for first_match in first_options:
        for second_match in second_options:
            if (
                first_match is not None
                and second_match is not None
                and first_match.anchor_index == second_match.anchor_index
            ):
                continue
            present_matches = tuple(
                match
                for match in (first_match, second_match)
                if match is not None
            )
            score = (
                -len(present_matches),
                round(
                    sum(match.endpoint_distance for match in present_matches),
                    _POINT_KEY_PRECISION,
                ),
                round(
                    sum(match.angle_error for match in present_matches),
                    _POINT_KEY_PRECISION,
                ),
                round(
                    sum(match.gap for match in present_matches),
                    _POINT_KEY_PRECISION,
                ),
                -round(
                    sum(match.overlap for match in present_matches),
                    _POINT_KEY_PRECISION,
                ),
                (
                    len(anchors)
                    if first_match is None
                    else first_match.anchor_index
                ),
                (
                    len(anchors)
                    if second_match is None
                    else second_match.anchor_index
                ),
            )
            assignments.append((score, first_match, second_match))
    _score, first_match, second_match = min(
        assignments,
        key=lambda item: item[0],
    )
    return first_match, second_match


def _wall_pair_assignment_is_coherent(
    pair: _WallFacePair,
    first_anchor: _Segment,
    second_anchor: _Segment,
    safe_tolerance: float,
    options: PlanWallDetectionOptions,
) -> bool:
    """Reject unrelated existing edges that merely happen to be nearby."""

    angle_tolerance = math.radians(options.parallel_angle_tolerance_degrees)
    if (
        _angle_difference(
            _segment_angle(first_anchor),
            _segment_angle(second_anchor),
        )
        > angle_tolerance
    ):
        return False

    generated_separation, _overlap = _parallel_separation_and_overlap(
        pair.first,
        pair.second,
    )
    anchor_separation, _overlap = _parallel_separation_and_overlap(
        first_anchor,
        second_anchor,
    )
    coherence_tolerance = max(1.0, safe_tolerance)
    if abs(generated_separation - anchor_separation) > coherence_tolerance:
        return False

    direction = _unit_direction(pair.first)
    normal = np.asarray((-direction[1], direction[0]), dtype=np.float64)
    generated_order = float(
        np.dot(_segment_center(pair.second) - _segment_center(pair.first), normal)
    )
    anchor_order = float(
        np.dot(_segment_center(second_anchor) - _segment_center(first_anchor), normal)
    )
    if generated_order * anchor_order <= _GEOMETRY_EPSILON:
        return False

    first_offset = float(
        np.dot(_segment_center(first_anchor) - _segment_center(pair.first), normal)
    )
    second_offset = float(
        np.dot(_segment_center(second_anchor) - _segment_center(pair.second), normal)
    )
    return abs(first_offset - second_offset) <= coherence_tolerance


def _segment_center(segment: _Segment) -> np.ndarray:
    """Return one segment midpoint without exposing mutable endpoint arrays."""

    return (segment.start + segment.end) / 2.0


def _segment_gap_and_overlap_on_anchor(
    segment: _Segment,
    anchor: _Segment,
) -> tuple[float, float]:
    """Return longitudinal gap and overlap in the anchor's coordinate frame."""

    direction = _unit_direction(anchor)
    segment_interval = _projection_interval(segment, direction)
    anchor_interval = _projection_interval(anchor, direction)
    gap = max(
        0.0,
        segment_interval[0] - anchor_interval[1],
        anchor_interval[0] - segment_interval[1],
    )
    overlap = max(
        0.0,
        min(segment_interval[1], anchor_interval[1])
        - max(segment_interval[0], anchor_interval[0]),
    )
    return gap, overlap


def _project_segment_onto_existing_wall(
    segment: _Segment,
    anchor: _Segment,
    endpoint_snap_distance: float,
) -> _Segment:
    """Project a segment onto an anchor and reuse nearby anchor endpoints."""

    direction = _unit_direction(anchor)
    anchor_start_projection = float(np.dot(anchor.start, direction))
    anchor_end_projection = float(np.dot(anchor.end, direction))
    original_projected_values = [
        float(np.dot(point, direction))
        for point in (segment.start, segment.end)
    ]
    projected_values = list(original_projected_values)
    for index, value in enumerate(projected_values):
        endpoint_candidates = (
            anchor_start_projection,
            anchor_end_projection,
        )
        nearest_endpoint = min(
            endpoint_candidates,
            key=lambda endpoint: (abs(value - endpoint), endpoint),
        )
        if abs(value - nearest_endpoint) <= endpoint_snap_distance:
            projected_values[index] = nearest_endpoint

    # A short detection can sit near one anchor endpoint or straddle a junction
    # shared by two existing edges. Both independently chosen endpoints may then
    # snap to the same point. Keep its projected span so coverage/noding never
    # receives a zero-length segment.
    if abs(projected_values[1] - projected_values[0]) <= 0.25:
        projected_values = original_projected_values

    normal = np.asarray((-direction[1], direction[0]), dtype=np.float64)
    normal_offset = float(np.dot(anchor.start, normal))
    start = direction * projected_values[0] + normal * normal_offset
    end = direction * projected_values[1] + normal * normal_offset
    start, end = _canonical_endpoints(start, end)
    return _Segment(start=start, end=end, confidence=segment.confidence)


# ### Wall-pair endpoint closure ###
def _close_wall_face_pair_ends(
    wall_face_pairs: Sequence[_WallFacePair],
    existing_graph: VertexData,
    options: PlanWallDetectionOptions,
    *,
    endpoint_topology: VertexData | None = None,
) -> tuple[list[_WallFacePair], list[_Segment]]:
    """Close supported paired wall ends with perpendicular cap segments.

    Short end caps are commonly omitted by wall-face pairing because they have
    no parallel partner. Pair provenance identifies the only safe counterpart
    for each endpoint, avoiding nearest-neighbor links between unrelated walls.
    """

    ordered_pairs = sorted(
        wall_face_pairs,
        key=lambda pair: (
            _segment_sort_key(pair.first),
            _segment_sort_key(pair.second),
        ),
    )
    paired_obstructions = [
        (pair_index, segment)
        for pair_index, pair in enumerate(ordered_pairs)
        for segment in (pair.first, pair.second)
    ]
    existing_obstructions = _existing_graph_segments(existing_graph)
    adjusted_pairs: list[_WallFacePair] = []
    connectors: list[_Segment] = []

    for pair_index, pair in enumerate(ordered_pairs):
        if (
            min(pair.first.length, pair.second.length)
            <= options.maximum_vertex_distance_pixels + _GEOMETRY_EPSILON
            or not _wall_face_pair_has_uncovered_geometry(pair, existing_graph)
        ):
            adjusted_pairs.append(pair)
            continue
        first_points = [pair.first.start.copy(), pair.first.end.copy()]
        second_points = [pair.second.start.copy(), pair.second.end.copy()]
        direction = _shared_wall_pair_direction(pair)
        first_indices = sorted(
            range(2),
            key=lambda index: float(np.dot(first_points[index], direction)),
        )
        second_indices = sorted(
            range(2),
            key=lambda index: float(np.dot(second_points[index], direction)),
        )

        for first_index, second_index in zip(
            first_indices,
            second_indices,
            strict=True,
        ):
            if endpoint_topology is not None and not _wall_pair_end_is_dangling(
                first_points[first_index],
                pair.first,
                second_points[second_index],
                pair.second,
                endpoint_topology,
                options,
            ):
                continue
            closure = _aligned_wall_pair_end_closure(
                pair,
                first_points[first_index],
                second_points[second_index],
                direction,
                existing_obstructions,
                options,
            )
            if closure is None:
                continue
            first_point, second_point, connector = closure
            other_obstructions = [
                segment
                for obstruction_pair_index, segment in paired_obstructions
                if obstruction_pair_index != pair_index
            ]
            if _connector_crosses_unrelated_geometry(
                connector,
                (*other_obstructions, *existing_obstructions, *connectors),
            ):
                continue
            first_points[first_index] = first_point
            second_points[second_index] = second_point
            connectors.append(connector)

        first_start, first_end = _canonical_endpoints(
            first_points[0],
            first_points[1],
        )
        second_start, second_end = _canonical_endpoints(
            second_points[0],
            second_points[1],
        )
        adjusted_pairs.append(
            _WallFacePair(
                first=_Segment(
                    start=first_start,
                    end=first_end,
                    confidence=pair.first.confidence,
                ),
                second=_Segment(
                    start=second_start,
                    end=second_end,
                    confidence=pair.second.confidence,
                ),
            )
        )

    return adjusted_pairs, _deduplicate_segments(connectors)


def _wall_pair_end_is_dangling(
    first_endpoint: np.ndarray,
    first_face: _Segment,
    second_endpoint: np.ndarray,
    second_face: _Segment,
    topology: VertexData,
    options: PlanWallDetectionOptions,
) -> bool:
    """Require both proposed cap endpoints to be degree-one face vertices."""

    vertices_by_id = {vertex.id: vertex for vertex in topology.vertices}
    incident_edges: dict[int, list[Edge]] = {
        vertex_id: [] for vertex_id in vertices_by_id
    }
    for edge in topology.edges:
        incident_edges[edge.start_vertex_id].append(edge)
        incident_edges[edge.end_vertex_id].append(edge)

    first_id = _matching_topology_endpoint_id(
        first_endpoint,
        first_face,
        vertices_by_id,
        incident_edges,
        options,
    )
    second_id = _matching_topology_endpoint_id(
        second_endpoint,
        second_face,
        vertices_by_id,
        incident_edges,
        options,
    )
    return (
        first_id is not None
        and second_id is not None
        and first_id != second_id
        and len(incident_edges[first_id]) == 1
        and len(incident_edges[second_id]) == 1
    )


def _matching_topology_endpoint_id(
    endpoint: np.ndarray,
    face: _Segment,
    vertices_by_id: dict[int, Vertex],
    incident_edges: dict[int, list[Edge]],
    options: PlanWallDetectionOptions,
) -> int | None:
    """Find the preliminary graph vertex supported by one wall-face endpoint."""

    search_distance = max(1.5, options.maximum_vertex_distance_pixels)
    face_angle = _segment_angle(face)
    angle_tolerance = math.radians(options.parallel_angle_tolerance_degrees)
    matches: list[tuple[float, int]] = []
    for vertex_id, vertex in vertices_by_id.items():
        distance = math.dist(endpoint, (vertex.x, vertex.y))
        if distance > search_distance + _GEOMETRY_EPSILON:
            continue
        for edge in incident_edges[vertex_id]:
            other_id = (
                edge.end_vertex_id
                if edge.start_vertex_id == vertex_id
                else edge.start_vertex_id
            )
            other = vertices_by_id[other_id]
            incident = _segment_from_vertices(vertex, other)
            if (
                _angle_difference(face_angle, _segment_angle(incident))
                <= angle_tolerance + _GEOMETRY_EPSILON
            ):
                matches.append((distance, vertex_id))
                break
    if not matches:
        return None
    return min(matches)[1]


def _aligned_wall_pair_end_closure(
    pair: _WallFacePair,
    first_endpoint: np.ndarray,
    second_endpoint: np.ndarray,
    shared_direction: np.ndarray,
    existing_segments: Sequence[_Segment],
    options: PlanWallDetectionOptions,
) -> tuple[np.ndarray, np.ndarray, _Segment] | None:
    """Align one corresponding endpoint pair and build its end connector."""

    original_connector = _Segment(
        start=first_endpoint,
        end=second_endpoint,
        confidence=min(pair.first.confidence, pair.second.confidence),
    )
    if original_connector.length <= _GEOMETRY_EPSILON:
        return None

    first_projection = float(np.dot(first_endpoint, shared_direction))
    second_projection = float(np.dot(second_endpoint, shared_direction))
    if (
        abs(first_projection - second_projection)
        > options.endpoint_snap_distance_pixels + _GEOMETRY_EPSILON
    ):
        return None

    target_projection = _wall_pair_end_target_projection(
        first_endpoint,
        second_endpoint,
        first_projection,
        second_projection,
        pair,
        existing_segments,
    )
    if target_projection is None:
        return None
    first_point = _point_on_segment_line_at_projection(
        first_endpoint,
        pair.first,
        shared_direction,
        target_projection,
    )
    second_point = _point_on_segment_line_at_projection(
        second_endpoint,
        pair.second,
        shared_direction,
        target_projection,
    )
    if first_point is None or second_point is None:
        return None
    if (
        math.dist(first_endpoint, first_point)
        > options.endpoint_snap_distance_pixels + _GEOMETRY_EPSILON
        or math.dist(second_endpoint, second_point)
        > options.endpoint_snap_distance_pixels + _GEOMETRY_EPSILON
    ):
        return None

    connector_start, connector_end = _canonical_endpoints(
        first_point.copy(),
        second_point.copy(),
    )
    connector = _Segment(
        start=connector_start,
        end=connector_end,
        confidence=min(pair.first.confidence, pair.second.confidence),
    )
    if not (
        options.minimum_wall_separation_pixels - _GEOMETRY_EPSILON
        <= connector.length
        <= options.maximum_wall_separation_pixels + _GEOMETRY_EPSILON
    ):
        return None
    if not _connector_is_perpendicular_to_pair(
        connector,
        pair,
        options.parallel_angle_tolerance_degrees,
    ):
        return None
    return first_point, second_point, connector


def _shared_wall_pair_direction(pair: _WallFacePair) -> np.ndarray:
    """Return a deterministic length-weighted direction for two wall faces."""

    first_direction = _unit_direction(pair.first)
    second_direction = _unit_direction(pair.second)
    if float(np.dot(first_direction, second_direction)) < 0.0:
        second_direction = -second_direction
    combined = (
        first_direction * pair.first.length
        + second_direction * pair.second.length
    )
    length = float(np.linalg.norm(combined))
    if length <= _GEOMETRY_EPSILON:
        return first_direction
    return combined / length


def _connector_is_perpendicular_to_pair(
    connector: _Segment,
    pair: _WallFacePair,
    tolerance_degrees: float,
) -> bool:
    """Return whether a connector meets both wall faces at about 90 degrees."""

    tolerance = math.radians(tolerance_degrees)
    connector_angle = _segment_angle(connector)
    return all(
        abs(
            math.pi / 2.0
            - _angle_difference(connector_angle, _segment_angle(face))
        )
        <= tolerance + _GEOMETRY_EPSILON
        for face in (pair.first, pair.second)
    )


def _wall_pair_end_target_projection(
    first_endpoint: np.ndarray,
    second_endpoint: np.ndarray,
    first_projection: float,
    second_projection: float,
    pair: _WallFacePair,
    existing_segments: Sequence[_Segment],
) -> float | None:
    """Choose an anchored or confidence-weighted shared endpoint coordinate."""

    first_is_anchored = _point_is_covered_by_existing_wall(
        first_endpoint,
        existing_segments,
    )
    second_is_anchored = _point_is_covered_by_existing_wall(
        second_endpoint,
        existing_segments,
    )
    if first_is_anchored and not second_is_anchored:
        return first_projection
    if second_is_anchored and not first_is_anchored:
        return second_projection
    if first_is_anchored and second_is_anchored:
        return None

    first_weight = pair.first.length * max(
        pair.first.confidence,
        _GEOMETRY_EPSILON,
    )
    second_weight = pair.second.length * max(
        pair.second.confidence,
        _GEOMETRY_EPSILON,
    )
    return (
        first_projection * first_weight
        + second_projection * second_weight
    ) / (first_weight + second_weight)


def _point_is_covered_by_existing_wall(
    point: np.ndarray,
    existing_segments: Sequence[_Segment],
) -> bool:
    """Return whether a point lies on an immutable existing wall edge."""

    for segment in existing_segments:
        if (
            _point_line_distance(point, segment)
            > _EXISTING_WALL_COLLINEAR_TOLERANCE_PX
        ):
            continue
        squared_length = segment.length**2
        parameter = float(
            np.dot(point - segment.start, segment.vector) / squared_length
        )
        if -_GEOMETRY_EPSILON <= parameter <= 1.0 + _GEOMETRY_EPSILON:
            return True
    return False


def _wall_face_pair_has_uncovered_geometry(
    pair: _WallFacePair,
    existing_graph: VertexData,
) -> bool:
    """Return whether either detected face adds geometry beyond existing walls."""

    if not existing_graph.edges:
        return True
    return any(
        _subtract_existing_wall_geometry((face,), existing_graph)
        for face in (pair.first, pair.second)
    )


def _point_on_segment_line_at_projection(
    endpoint: np.ndarray,
    segment: _Segment,
    shared_direction: np.ndarray,
    target_projection: float,
) -> np.ndarray | None:
    """Move an endpoint along its supporting line to one shared projection."""

    segment_direction = _unit_direction(segment)
    denominator = float(np.dot(segment_direction, shared_direction))
    if abs(denominator) <= _GEOMETRY_EPSILON:
        return None
    distance = (
        target_projection - float(np.dot(endpoint, shared_direction))
    ) / denominator
    point = endpoint + segment_direction * distance
    if not np.all(np.isfinite(point)):
        return None
    return np.round(point, decimals=_POINT_KEY_PRECISION)


def _existing_graph_segments(graph: VertexData) -> list[_Segment]:
    """Return immutable graph edges as deterministic geometric segments."""

    vertices_by_id = {vertex.id: vertex for vertex in graph.vertices}
    return sorted(
        (
            _segment_from_vertices(
                vertices_by_id[edge.start_vertex_id],
                vertices_by_id[edge.end_vertex_id],
            )
            for edge in graph.edges
        ),
        key=_segment_sort_key,
    )


def _connector_crosses_unrelated_geometry(
    connector: _Segment,
    obstructions: Sequence[_Segment],
) -> bool:
    """Reject a proposed cap whose interior crosses unrelated linework."""

    endpoint_tolerance = 0.25
    for obstruction in obstructions:
        intersection = _bounded_segment_intersection(connector, obstruction)
        if intersection is None:
            continue
        if min(
            math.dist(intersection, connector.start),
            math.dist(intersection, connector.end),
        ) > endpoint_tolerance:
            return True
    return False


# ### Segment de-duplication and noding ###
def _deduplicate_segments(segments: Sequence[_Segment]) -> list[_Segment]:
    kept: list[_Segment] = []
    for candidate in sorted(segments, key=_segment_sort_key):
        if any(_same_segment(candidate, current) for current in kept):
            continue
        kept.append(candidate)
    return kept


def _node_segments(
    segments: Sequence[_Segment],
    options: PlanWallDetectionOptions,
) -> list[_Segment]:
    mutable = [
        _Segment(
            start=segment.start.copy(),
            end=segment.end.copy(),
            confidence=segment.confidence,
        )
        for segment in segments
    ]
    mutable = _extend_near_intersections(
        mutable,
        options.endpoint_snap_distance_pixels,
    )
    split_points: list[list[np.ndarray]] = [
        [segment.start.copy(), segment.end.copy()] for segment in mutable
    ]
    for first_index, first in enumerate(mutable):
        for second_index in range(first_index + 1, len(mutable)):
            second = mutable[second_index]
            if not _segment_bounds_overlap(first, second, 0.0):
                continue
            intersection = _bounded_segment_intersection(first, second)
            if intersection is None:
                continue
            split_points[first_index].append(intersection)
            split_points[second_index].append(intersection)

    pieces: list[_Segment] = []
    for segment, points in zip(mutable, split_points, strict=True):
        direction = _unit_direction(segment)
        ordered_points = _unique_points(
            sorted(points, key=lambda point: float(np.dot(point, direction)))
        )
        for start, end in pairwise(ordered_points):
            if float(np.linalg.norm(end - start)) <= 0.25:
                continue
            start, end = _canonical_endpoints(start.copy(), end.copy())
            pieces.append(_Segment(start=start, end=end, confidence=segment.confidence))
    return _deduplicate_segments(pieces)


def _extend_near_intersections(
    segments: list[_Segment],
    snap_distance: float,
) -> list[_Segment]:
    if snap_distance <= 0.0:
        return segments
    result = segments
    for _iteration in range(2):
        changed = False
        updated = list(result)
        for first_index, first in enumerate(result):
            for second_index in range(first_index + 1, len(result)):
                second = result[second_index]
                if not _segment_bounds_overlap(first, second, snap_distance):
                    continue
                if _angle_difference(
                    _segment_angle(first),
                    _segment_angle(second),
                ) < math.radians(1.0):
                    continue
                intersection_parameters = _line_intersection_parameters(
                    first,
                    second,
                )
                if intersection_parameters is None:
                    continue
                intersection, first_parameter, second_parameter = (
                    intersection_parameters
                )
                first_distance = _outside_parameter_distance(
                    first_parameter,
                    first.length,
                )
                second_distance = _outside_parameter_distance(
                    second_parameter,
                    second.length,
                )
                if first_distance > snap_distance or second_distance > snap_distance:
                    continue
                new_first = _extended_to_intersection(
                    updated[first_index],
                    intersection,
                    first_parameter,
                )
                new_second = _extended_to_intersection(
                    updated[second_index],
                    intersection,
                    second_parameter,
                )
                changed = changed or not _same_segment(new_first, updated[first_index])
                changed = changed or not _same_segment(
                    new_second,
                    updated[second_index],
                )
                updated[first_index] = new_first
                updated[second_index] = new_second
        result = updated
        if not changed:
            break
    return result


def _line_intersection_parameters(
    first: _Segment,
    second: _Segment,
) -> tuple[np.ndarray, float, float] | None:
    first_dx = float(first.end[0] - first.start[0])
    first_dy = float(first.end[1] - first.start[1])
    second_dx = float(second.end[0] - second.start[0])
    second_dy = float(second.end[1] - second.start[1])
    denominator = first_dx * second_dy - first_dy * second_dx
    if abs(denominator) <= _GEOMETRY_EPSILON:
        return None
    delta_x = float(second.start[0] - first.start[0])
    delta_y = float(second.start[1] - first.start[1])
    first_parameter = (delta_x * second_dy - delta_y * second_dx) / denominator
    second_parameter = (delta_x * first_dy - delta_y * first_dx) / denominator
    intersection = np.asarray(
        (
            float(first.start[0]) + first_parameter * first_dx,
            float(first.start[1]) + first_parameter * first_dy,
        ),
        dtype=np.float64,
    )
    return intersection, first_parameter, second_parameter


def _bounded_segment_intersection(
    first: _Segment,
    second: _Segment,
) -> np.ndarray | None:
    result = _line_intersection_parameters(first, second)
    if result is None:
        return None
    intersection, first_parameter, second_parameter = result
    if not -_GEOMETRY_EPSILON <= first_parameter <= 1.0 + _GEOMETRY_EPSILON:
        return None
    if not -_GEOMETRY_EPSILON <= second_parameter <= 1.0 + _GEOMETRY_EPSILON:
        return None
    return intersection


def _outside_parameter_distance(parameter: float, length: float) -> float:
    if parameter < 0.0:
        return -parameter * length
    if parameter > 1.0:
        return (parameter - 1.0) * length
    return 0.0


def _extended_to_intersection(
    segment: _Segment,
    intersection: np.ndarray,
    parameter: float,
) -> _Segment:
    start = segment.start.copy()
    end = segment.end.copy()
    if parameter < 0.0:
        start = intersection.copy()
    elif parameter > 1.0:
        end = intersection.copy()
    start, end = _canonical_endpoints(start, end)
    return _Segment(start=start, end=end, confidence=segment.confidence)


# ### VertexData graph construction ###
def _append_segments_to_graph(
    graph: VertexData,
    segments: Sequence[_Segment],
    image_width: int,
    image_height: int,
    options: PlanWallDetectionOptions,
) -> None:
    clipped_segments: list[_Segment] = []
    for segment in segments:
        clipped = _clip_segment_to_image(segment, image_width, image_height)
        if clipped is not None and clipped.length > 0.25:
            clipped_segments.append(clipped)
    clipped_segments = _subtract_existing_wall_geometry(
        clipped_segments,
        graph,
    )
    ordered_segments = sorted(clipped_segments, key=_segment_sort_key)
    generated_points = _unique_points(
        sorted(
            [
                point
                for segment in ordered_segments
                for point in (segment.start, segment.end)
            ],
            key=lambda point: (float(point[0]), float(point[1])),
        )
    )
    representatives, representative_indices = _resolve_vertex_representatives(
        graph,
        generated_points,
        ordered_segments,
        options,
    )
    mapped_segment_indices: list[tuple[int, int]] = []
    mapped_edge_keys: set[tuple[int, int]] = set()
    for segment in ordered_segments:
        start_index = representative_indices[_point_key(segment.start)]
        end_index = representative_indices[_point_key(segment.end)]
        if start_index == end_index:
            continue
        edge_key = tuple(sorted((start_index, end_index)))
        if edge_key in mapped_edge_keys:
            continue
        mapped_edge_keys.add(edge_key)
        mapped_segment_indices.append((start_index, end_index))
    used_representative_indices = {
        index for segment_indices in mapped_segment_indices for index in segment_indices
    }

    representative_vertex_ids: dict[int, int] = {}
    for representative_index in sorted(used_representative_indices):
        representative = representatives[representative_index]
        if representative.existing_vertex_id is not None:
            representative_vertex_ids[representative_index] = (
                representative.existing_vertex_id
            )
            continue
        point = representative.point
        vertex = graph.add_vertex(float(point[0]), float(point[1]))
        representative_vertex_ids[representative_index] = vertex.id

    additions: set[tuple[int, int]] = set()
    for start_index, end_index in mapped_segment_indices:
        start_id = representative_vertex_ids[start_index]
        end_id = representative_vertex_ids[end_index]
        if start_id == end_id or graph.has_edge(start_id, end_id):
            continue
        key = tuple(sorted((start_id, end_id)))
        additions.add(key)
    for start_id, end_id in sorted(additions):
        graph.add_edge(start_id, end_id)


def _resolve_vertex_representatives(
    graph: VertexData,
    points: Sequence[np.ndarray],
    generated_segments: Sequence[_Segment],
    options: PlanWallDetectionOptions,
) -> tuple[list[_VertexRepresentative], dict[tuple[float, float], int]]:
    """Resolve generated endpoints without merging opposite wall faces."""

    generated_incidents = _incident_segments_by_point(generated_segments)
    representatives = _existing_vertex_representatives(graph)
    representative_indices: dict[tuple[float, float], int] = {}
    generated_distance = options.maximum_vertex_distance_pixels
    existing_distance = max(
        generated_distance,
        min(
            options.endpoint_snap_distance_pixels,
            options.minimum_wall_separation_pixels * 0.45,
        ),
    )
    search_distance = max(generated_distance, existing_distance)
    if search_distance <= 0.0:
        representatives.clear()
        for point in points:
            point_key = _point_key(point)
            representative_indices[point_key] = len(representatives)
            representatives.append(
                _VertexRepresentative(
                    point=point,
                    source_points=[point],
                    incident_segments=list(generated_incidents.get(point_key, ())),
                )
            )
        return representatives, representative_indices

    buckets: dict[tuple[int, int], list[int]] = {}
    for index, representative in enumerate(representatives):
        cell = _vertex_grid_cell(representative.point, search_distance)
        buckets.setdefault(cell, []).append(index)

    for point in points:
        point_key = _point_key(point)
        point_incidents = list(generated_incidents.get(point_key, ()))
        cell_x, cell_y = _vertex_grid_cell(point, search_distance)
        matches: list[tuple[float, bool, int, int]] = []
        for neighbor_x in range(cell_x - 1, cell_x + 2):
            for neighbor_y in range(cell_y - 1, cell_y + 2):
                for index in buckets.get((neighbor_x, neighbor_y), ()):
                    representative = representatives[index]
                    distance_limit = (
                        existing_distance
                        if representative.existing_vertex_id is not None
                        else generated_distance
                    )
                    if distance_limit <= 0.0:
                        continue
                    distance = math.dist(point, representative.point)
                    if distance > distance_limit:
                        continue
                    if _incidents_span_wall_face_separation(
                        point_incidents,
                        representative.incident_segments,
                        options,
                    ):
                        continue
                    existing_id = representative.existing_vertex_id
                    matches.append(
                        (
                            distance,
                            existing_id is None,
                            existing_id if existing_id is not None else index,
                            index,
                        )
                    )
        if matches:
            representative_index = min(matches)[3]
            representative_indices[point_key] = representative_index
            representatives[representative_index].source_points.append(point)
            representatives[representative_index].incident_segments.extend(
                point_incidents
            )
            continue

        representative_index = len(representatives)
        representative_indices[point_key] = representative_index
        representatives.append(
            _VertexRepresentative(
                point=point,
                source_points=[point],
                incident_segments=point_incidents,
            )
        )
        buckets.setdefault((cell_x, cell_y), []).append(representative_index)
    _refine_generated_vertex_representatives(
        representatives,
        generated_distance,
    )
    return representatives, representative_indices


def _refine_generated_vertex_representatives(
    representatives: Sequence[_VertexRepresentative],
    maximum_adjustment: float,
) -> None:
    """Fit merged endpoints to all plan lines that support each junction."""

    if maximum_adjustment <= _GEOMETRY_EPSILON:
        return
    for representative in representatives:
        if representative.existing_vertex_id is not None:
            continue
        refined = _best_fit_representative_point(representative)
        if refined is None:
            continue
        if math.dist(refined, representative.point) > maximum_adjustment:
            continue
        if any(
            math.dist(refined, source_point) > maximum_adjustment
            for source_point in representative.source_points
        ):
            continue
        representative.point = refined


def _best_fit_representative_point(
    representative: _VertexRepresentative,
) -> np.ndarray | None:
    """Return the stable least-squares intersection of supporting lines."""

    segments = _deduplicate_segments(representative.incident_segments)
    if not segments or not representative.source_points:
        return None

    normal_matrix = np.zeros((2, 2), dtype=np.float64)
    projected_points = np.zeros(2, dtype=np.float64)
    for segment in segments:
        if segment.length <= _GEOMETRY_EPSILON:
            continue
        direction = _unit_direction(segment)
        normal = np.asarray((-direction[1], direction[0]), dtype=np.float64)
        projection = np.outer(normal, normal)
        weight = segment.length * max(segment.confidence, _GEOMETRY_EPSILON)
        normal_matrix += weight * projection
        projected_points += weight * (projection @ segment.start)

    if not np.any(normal_matrix):
        return None
    centroid = np.mean(
        np.asarray(representative.source_points, dtype=np.float64),
        axis=0,
    )
    try:
        correction = np.linalg.pinv(normal_matrix) @ (
            projected_points - normal_matrix @ centroid
        )
    except np.linalg.LinAlgError:
        return None
    refined = centroid + correction
    if not np.all(np.isfinite(refined)):
        return None
    return np.round(refined, decimals=_POINT_KEY_PRECISION)


def _incident_segments_by_point(
    segments: Sequence[_Segment],
) -> dict[tuple[float, float], list[_Segment]]:
    incidents: dict[tuple[float, float], list[_Segment]] = {}
    for segment in segments:
        for point in (segment.start, segment.end):
            incidents.setdefault(_point_key(point), []).append(segment)
    return incidents


def _existing_vertex_representatives(
    graph: VertexData,
) -> list[_VertexRepresentative]:
    vertices_by_id = {vertex.id: vertex for vertex in graph.vertices}
    incidents_by_id: dict[int, list[_Segment]] = {}
    for edge in graph.edges:
        segment = _segment_from_vertices(
            vertices_by_id[edge.start_vertex_id],
            vertices_by_id[edge.end_vertex_id],
        )
        incidents_by_id.setdefault(edge.start_vertex_id, []).append(segment)
        incidents_by_id.setdefault(edge.end_vertex_id, []).append(segment)
    return [
        _VertexRepresentative(
            point=np.asarray((vertex.x, vertex.y), dtype=np.float64),
            source_points=[
                np.asarray((vertex.x, vertex.y), dtype=np.float64)
            ],
            incident_segments=incidents_by_id[vertex.id],
            existing_vertex_id=vertex.id,
        )
        for vertex in sorted(graph.vertices, key=lambda item: item.id)
        if vertex.id in incidents_by_id
    ]


def _vertex_grid_cell(point: np.ndarray, cell_size: float) -> tuple[int, int]:
    return (
        math.floor(float(point[0]) / cell_size),
        math.floor(float(point[1]) / cell_size),
    )


def _incidents_span_wall_face_separation(
    first: Sequence[_Segment],
    second: Sequence[_Segment],
    options: PlanWallDetectionOptions,
) -> bool:
    angle_tolerance = math.radians(options.parallel_angle_tolerance_degrees)
    for first_segment in first:
        first_angle = _segment_angle(first_segment)
        for second_segment in second:
            if (
                _angle_difference(first_angle, _segment_angle(second_segment))
                > angle_tolerance
            ):
                continue
            separation, _gap = _parallel_offset_and_gap(
                first_segment,
                second_segment,
            )
            if (
                options.minimum_wall_separation_pixels - _GEOMETRY_EPSILON
                <= separation
                <= options.maximum_wall_separation_pixels + _GEOMETRY_EPSILON
            ):
                return True
    return False


# ### Existing-wall geometric subtraction ###
def _subtract_existing_wall_geometry(
    generated_segments: Sequence[_Segment],
    existing_graph: VertexData,
) -> list[_Segment]:
    """Keep only generated linework not already covered by immutable walls."""

    existing_vertices = {vertex.id: vertex for vertex in existing_graph.vertices}
    existing_segments = [
        _segment_from_vertices(
            existing_vertices[edge.start_vertex_id],
            existing_vertices[edge.end_vertex_id],
        )
        for edge in existing_graph.edges
    ]
    if not existing_segments:
        return list(generated_segments)

    uncovered: list[_Segment] = []
    for generated in generated_segments:
        breakpoints = {0.0, 1.0}
        covered_intervals: list[tuple[float, float]] = []
        for existing in existing_segments:
            if not _segment_bounds_overlap(
                generated,
                existing,
                _EXISTING_WALL_COLLINEAR_TOLERANCE_PX,
            ):
                continue
            if _segments_are_collinear(generated, existing):
                interval = _projected_overlap_parameters(generated, existing)
                if interval is None:
                    continue
                interval_start, interval_end = interval
                breakpoints.update((interval_start, interval_end))
                if interval_end - interval_start > _GEOMETRY_EPSILON:
                    covered_intervals.append(interval)
                continue
            intersection = _line_intersection_parameters(generated, existing)
            if intersection is None:
                continue
            _point, generated_parameter, existing_parameter = intersection
            if not (
                -_GEOMETRY_EPSILON <= generated_parameter <= 1.0 + _GEOMETRY_EPSILON
            ):
                continue
            if not (
                -_GEOMETRY_EPSILON <= existing_parameter <= 1.0 + _GEOMETRY_EPSILON
            ):
                continue
            breakpoints.add(float(np.clip(generated_parameter, 0.0, 1.0)))

        merged_coverage = _merged_parameter_intervals(covered_intervals)
        ordered_breakpoints = sorted(breakpoints)
        for parameter_start, parameter_end in pairwise(ordered_breakpoints):
            if parameter_end - parameter_start <= _GEOMETRY_EPSILON:
                continue
            midpoint = (parameter_start + parameter_end) / 2.0
            if any(
                interval_start - _GEOMETRY_EPSILON
                <= midpoint
                <= interval_end + _GEOMETRY_EPSILON
                for interval_start, interval_end in merged_coverage
            ):
                continue
            start = generated.start + parameter_start * generated.vector
            end = generated.start + parameter_end * generated.vector
            if math.dist(start, end) <= 0.25:
                continue
            start, end = _canonical_endpoints(start.copy(), end.copy())
            uncovered.append(
                _Segment(
                    start=start,
                    end=end,
                    confidence=generated.confidence,
                )
            )
    return _deduplicate_segments(uncovered)


def _segment_from_vertices(first: Vertex, second: Vertex) -> _Segment:
    start = np.asarray((first.x, first.y), dtype=np.float64)
    end = np.asarray((second.x, second.y), dtype=np.float64)
    start, end = _canonical_endpoints(start, end)
    return _Segment(start=start, end=end, confidence=1.0)


def _segments_are_collinear(first: _Segment, second: _Segment) -> bool:
    if _angle_difference(
        _segment_angle(first),
        _segment_angle(second),
    ) > math.radians(_EXISTING_WALL_COLLINEAR_ANGLE_TOLERANCE_DEGREES):
        return False
    return (
        _point_line_distance(second.start, first)
        <= _EXISTING_WALL_COLLINEAR_TOLERANCE_PX
        and _point_line_distance(second.end, first)
        <= _EXISTING_WALL_COLLINEAR_TOLERANCE_PX
    )


def _point_line_distance(point: np.ndarray, line: _Segment) -> float:
    return abs(_cross_2d(point - line.start, line.vector)) / line.length


def _projected_overlap_parameters(
    generated: _Segment,
    existing: _Segment,
) -> tuple[float, float] | None:
    squared_length = generated.length**2
    first_parameter = float(
        np.dot(existing.start - generated.start, generated.vector) / squared_length
    )
    second_parameter = float(
        np.dot(existing.end - generated.start, generated.vector) / squared_length
    )
    overlap_start = max(0.0, min(first_parameter, second_parameter))
    overlap_end = min(1.0, max(first_parameter, second_parameter))
    if overlap_end < overlap_start - _GEOMETRY_EPSILON:
        return None
    return (
        float(np.clip(overlap_start, 0.0, 1.0)),
        float(np.clip(overlap_end, 0.0, 1.0)),
    )


def _merged_parameter_intervals(
    intervals: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1] + _GEOMETRY_EPSILON:
            merged.append((start, end))
            continue
        previous_start, previous_end = merged[-1]
        merged[-1] = (previous_start, max(previous_end, end))
    return merged


def _clip_segment_to_image(
    segment: _Segment,
    image_width: int,
    image_height: int,
) -> _Segment | None:
    success, clipped_start, clipped_end = cv2.clipLine(
        (0, 0, image_width, image_height),
        tuple(np.rint(segment.start).astype(int)),
        tuple(np.rint(segment.end).astype(int)),
    )
    if not success:
        return None
    start = _clipped_point(
        np.asarray(clipped_start, dtype=np.float64),
        image_width,
        image_height,
    )
    end = _clipped_point(
        np.asarray(clipped_end, dtype=np.float64),
        image_width,
        image_height,
    )
    start, end = _canonical_endpoints(start, end)
    return _Segment(start=start, end=end, confidence=segment.confidence)


# ### Geometry helpers ###
def _canonical_endpoints(
    start: np.ndarray,
    end: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if (float(end[0]), float(end[1])) < (float(start[0]), float(start[1])):
        return end, start
    return start, end


def _clipped_point(
    point: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    return np.asarray(
        (
            np.clip(point[0], 0.0, image_width - 1.0),
            np.clip(point[1], 0.0, image_height - 1.0),
        ),
        dtype=np.float64,
    )


def _segment_angle(segment: _Segment) -> float:
    angle = (
        math.atan2(
            float(segment.end[1] - segment.start[1]),
            float(segment.end[0] - segment.start[0]),
        )
        % math.pi
    )
    return angle


def _is_horizontal(segment: _Segment) -> bool:
    return abs(float(segment.end[1] - segment.start[1])) <= _GEOMETRY_EPSILON


def _is_vertical(segment: _Segment) -> bool:
    return abs(float(segment.end[0] - segment.start[0])) <= _GEOMETRY_EPSILON


def _angle_difference(first: float, second: float) -> float:
    difference = abs(first - second) % math.pi
    return min(difference, math.pi - difference)


def _unit_direction(segment: _Segment) -> np.ndarray:
    inverse_length = 1.0 / segment.length
    return np.asarray(
        (
            float(segment.end[0] - segment.start[0]) * inverse_length,
            float(segment.end[1] - segment.start[1]) * inverse_length,
        ),
        dtype=np.float64,
    )


def _projection_interval(
    segment: _Segment,
    direction: np.ndarray,
) -> tuple[float, float]:
    values = (
        float(np.dot(segment.start, direction)),
        float(np.dot(segment.end, direction)),
    )
    return min(values), max(values)


def _cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _segment_sort_key(
    segment: _Segment,
) -> tuple[float, float, float, float, float]:
    return (
        round(float(segment.start[0]), _POINT_KEY_PRECISION),
        round(float(segment.start[1]), _POINT_KEY_PRECISION),
        round(float(segment.end[0]), _POINT_KEY_PRECISION),
        round(float(segment.end[1]), _POINT_KEY_PRECISION),
        round(float(segment.confidence), _POINT_KEY_PRECISION),
    )


def _same_segment(first: _Segment, second: _Segment) -> bool:
    return (
        math.hypot(
            float(first.start[0] - second.start[0]),
            float(first.start[1] - second.start[1]),
        )
        <= 0.25
        and math.hypot(
            float(first.end[0] - second.end[0]),
            float(first.end[1] - second.end[1]),
        )
        <= 0.25
    )


def _segment_bounds_overlap(
    first: _Segment,
    second: _Segment,
    margin: float,
) -> bool:
    first_minimum_x = min(float(first.start[0]), float(first.end[0])) - margin
    first_maximum_x = max(float(first.start[0]), float(first.end[0])) + margin
    first_minimum_y = min(float(first.start[1]), float(first.end[1])) - margin
    first_maximum_y = max(float(first.start[1]), float(first.end[1])) + margin
    second_minimum_x = min(float(second.start[0]), float(second.end[0]))
    second_maximum_x = max(float(second.start[0]), float(second.end[0]))
    second_minimum_y = min(float(second.start[1]), float(second.end[1]))
    second_maximum_y = max(float(second.start[1]), float(second.end[1]))
    return (
        first_minimum_x <= second_maximum_x
        and first_maximum_x >= second_minimum_x
        and first_minimum_y <= second_maximum_y
        and first_maximum_y >= second_minimum_y
    )


def _unique_points(points: Sequence[np.ndarray]) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    keys: set[tuple[float, float]] = set()
    for point in points:
        key = _point_key(point)
        if key in keys:
            continue
        keys.add(key)
        unique.append(point.copy())
    return unique


def _point_key(point: np.ndarray) -> tuple[float, float]:
    return (
        round(float(point[0]), _POINT_KEY_PRECISION),
        round(float(point[1]), _POINT_KEY_PRECISION),
    )
