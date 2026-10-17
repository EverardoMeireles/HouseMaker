# ### Imports ###
from __future__ import annotations

import heapq
import math
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

from .models import Edge, Vertex, VertexData
from .plan_wall_options import PlanWallDetectionError, PlanWallDetectionOptions
from .plan_wall_pixel_tracing import PixelWallTracingCancelled, trace_plan_ink_paths

# ### Detection constants ###
_MIN_ANALYSIS_LINE_LENGTH_PX = 5.0
_AXIS_SNAP_LIMIT_DEGREES = 4.0
_GEOMETRY_EPSILON = 1e-6
_POINT_KEY_PRECISION = 6
_MINIMUM_STRUCTURAL_INK_RETENTION = 0.45
_EXISTING_WALL_COLLINEAR_TOLERANCE_PX = 0.5
_EXISTING_WALL_COLLINEAR_ANGLE_TOLERANCE_DEGREES = 0.5
_REPAIR_MINIMUM_INK_COVERAGE = 0.92
_REPAIR_MAXIMUM_INK_GAP_PIXELS = 2.0
_REPAIR_MAXIMUM_TURNS = 4
_PIXEL_REPAIR_MAXIMUM_TURNS = 16
_REPAIR_MAXIMUM_SEARCH_STATES = 20_000


# ### Public data models ###
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
    structural_ink: np.ndarray | None = field(default=None, compare=False, repr=False)
    normalized_ink: np.ndarray | None = field(default=None, compare=False, repr=False)
    pixel_component_labels: np.ndarray | None = field(
        default=None, compare=False, repr=False
    )
    pixel_line_evidence: tuple[WallLineEvidence, ...] = ()


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
    structural_ink = structural < 128
    structural_ink.setflags(write=False)
    # Tracing keeps thin genuine wall outlines that structural filtering or
    # the line detector can lose. Cache it here, never on a slider update.
    normalized_ink = _normalized_binary_plan(grayscale, blur=False) < 128
    normalized_ink.setflags(write=False)
    # Neighborhood labels only prune unrelated text before reconstruction.
    # The actual traced paths and all final support tests use undilated ink.
    nearby_ink = cv2.dilate(
        normalized_ink.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)
    )
    _component_count, pixel_component_labels = cv2.connectedComponents(
        nearby_ink, connectivity=8
    )
    pixel_component_labels.setflags(write=False)
    try:
        pixel_paths = trace_plan_ink_paths(normalized_ink, cancellation_check)
    except PixelWallTracingCancelled as error:
        raise PlanWallDetectionCancelled(str(error)) from error
    pixel_lines = tuple(
        WallLineEvidence(start=start, end=end, confidence=1.0)
        for path in pixel_paths
        for start, end in pairwise(path)
        if math.dist(start, end) > 0.25
    )
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected, _widths, _precisions, _nfa = detector.detect(normalized)
    if detected is None:
        return PlanWallAnalysis(
            image_width=int(grayscale.shape[1]),
            image_height=int(grayscale.shape[0]),
            line_evidence=(),
            structural_ink=structural_ink,
            normalized_ink=normalized_ink,
            pixel_component_labels=pixel_component_labels,
            pixel_line_evidence=pixel_lines,
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
        structural_ink=structural_ink,
        normalized_ink=normalized_ink,
        pixel_component_labels=pixel_component_labels,
        pixel_line_evidence=pixel_lines,
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
    aligned_pairs = _align_opposite_wall_endpoints(
        analysis,
        wall_face_pairs,
        preliminary_graph,
        result,
        resolved_options,
    )
    if any(
        aligned is not original
        for aligned, original in zip(aligned_pairs, wall_face_pairs, strict=True)
    ):
        wall_face_pairs = aligned_pairs
        preliminary_graph = result.clone()
        _append_segments_to_graph(
            preliminary_graph,
            _node_segments(
                _deduplicate_segments(
                    [
                        face
                        for pair in wall_face_pairs
                        for face in (pair.first, pair.second)
                    ]
                ),
                resolved_options,
            ),
            analysis.image_width,
            analysis.image_height,
            resolved_options,
        )
    # Repair before capping ends: an artificial cap would hide the dangling
    # vertices that identify incomplete wall continuations.
    repaired_segments = _repair_image_supported_wall_lines(
        analysis, preliminary_graph, result, wall_face_pairs, resolved_options,
    )
    if repaired_segments:
        _append_segments_to_graph(
            preliminary_graph, repaired_segments,
            analysis.image_width, analysis.image_height, resolved_options,
        )
    pixel_repairs = _repair_pixel_supported_wall_lines(
        analysis, preliminary_graph, result, wall_face_pairs, resolved_options,
    )
    if pixel_repairs:
        repaired_segments.extend(pixel_repairs)
        _append_segments_to_graph(
            preliminary_graph, pixel_repairs,
            analysis.image_width, analysis.image_height, resolved_options,
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
    paired_segments.extend(repaired_segments)
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


def _normalized_binary_plan(image: np.ndarray, *, blur: bool = True) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (3, 3), 0.0) if blur else image
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
    for ink in (analysis.structural_ink, analysis.normalized_ink):
        if ink is not None and (
            not isinstance(ink, np.ndarray)
            or ink.dtype != np.bool_
            or ink.shape != (analysis.image_height, analysis.image_width)
        ):
            raise PlanWallDetectionError(
                "Wall analysis ink must match the plan image dimensions."
            )
    labels = analysis.pixel_component_labels
    if labels is not None and (
        not isinstance(labels, np.ndarray)
        or labels.dtype != np.int32
        or labels.shape != (analysis.image_height, analysis.image_width)
    ):
        raise PlanWallDetectionError(
            "Wall tracing components must match the plan image dimensions."
        )
    for evidence in (*analysis.line_evidence, *analysis.pixel_line_evidence):
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


# ### Opposite wall endpoint alignment ###
def _align_opposite_wall_endpoints(
    analysis: PlanWallAnalysis,
    wall_face_pairs: Sequence[_WallFacePair],
    topology: VertexData,
    existing_graph: VertexData,
    options: PlanWallDetectionOptions,
) -> list[_WallFacePair]:
    """Extend unfinished faces to nearby opposing wall-end coordinates.

    The short extension limit is deliberately independent of the much longer
    search span across a room. No segment is ever added between opposing tips:
    subsequent image-supported repair alone decides whether a connection exists.
    """

    tolerance = options.endpoint_alignment_tolerance_pixels
    search_distance = options.endpoint_alignment_search_distance_pixels
    if tolerance <= 0.0 or search_distance <= 0.0 or not topology.vertices:
        return list(wall_face_pairs)

    vertices_by_id = {vertex.id: vertex for vertex in topology.vertices}
    incident_edges: dict[int, list[Edge]] = {
        vertex_id: [] for vertex_id in vertices_by_id
    }
    for edge in topology.edges:
        incident_edges[edge.start_vertex_id].append(edge)
        incident_edges[edge.end_vertex_id].append(edge)
    existing_ids = {vertex.id for vertex in existing_graph.vertices}
    immutable_segments = _existing_graph_segments(existing_graph)
    obstruction_records = sorted(
        (
            (
                edge,
                _segment_from_vertices(
                    vertices_by_id[edge.start_vertex_id],
                    vertices_by_id[edge.end_vertex_id],
                ),
            )
            for edge in topology.edges
        ),
        key=lambda record: _segment_sort_key(record[1]),
    )
    obstructions = [segment for _edge, segment in obstruction_records]
    obstruction_bounds = _build_segment_bounds(obstructions)
    vertices = list(topology.vertices)
    points = np.asarray([(vertex.x, vertex.y) for vertex in vertices])
    cosine_tolerance = math.cos(math.radians(options.parallel_angle_tolerance_degrees))
    adjusted_pairs = list(wall_face_pairs)
    accepted_extensions: list[_Segment] = []

    ordered_pairs = sorted(
        enumerate(wall_face_pairs),
        key=lambda indexed_pair: sorted(
            _segment_sort_key(face)
            for face in (indexed_pair[1].first, indexed_pair[1].second)
        ),
    )
    for pair_index, pair in ordered_pairs:
        own_ids = {
            vertex_id
            for face in (pair.first, pair.second)
            for endpoint in (face.start, face.end)
            if (
                vertex_id := _matching_topology_endpoint_id(
                    endpoint,
                    face,
                    vertices_by_id,
                    incident_edges,
                    options,
                )
            )
            is not None
        }
        adjusted_faces = [pair.first, pair.second]
        for face_index, face in sorted(
            enumerate((pair.first, pair.second)),
            key=lambda indexed_face: _segment_sort_key(indexed_face[1]),
        ):
            face_points = [face.start.copy(), face.end.copy()]
            for endpoint_index, endpoint in enumerate((face.start, face.end)):
                vertex_id = _matching_topology_endpoint_id(
                    endpoint,
                    face,
                    vertices_by_id,
                    incident_edges,
                    options,
                )
                if (
                    vertex_id is None
                    or vertex_id in existing_ids
                    or len(incident_edges[vertex_id]) != 1
                    or _point_is_covered_by_existing_wall(endpoint, immutable_segments)
                ):
                    continue
                source_incident_edge = incident_edges[vertex_id][0]
                outward = _unit_direction(face) * (-1.0 if endpoint_index == 0 else 1.0)
                normal = np.asarray((-outward[1], outward[0]))
                offsets = points - endpoint
                extensions = offsets @ outward
                spans = np.abs(offsets @ normal)
                candidate_indices = np.flatnonzero(
                    (extensions > _GEOMETRY_EPSILON)
                    & (extensions <= tolerance + _GEOMETRY_EPSILON)
                    & (
                        spans
                        >= np.maximum(
                            options.minimum_wall_separation_pixels, 2.0 * extensions
                        )
                    )
                    & (np.sum(offsets * offsets, axis=1) <= search_distance**2)
                )
                # Bound the expensive evidence/intersection checks per tip. The
                # closest opposite wall is more relevant than one behind it.
                candidate_indices = sorted(
                    candidate_indices,
                    key=lambda index: (
                        float(spans[index]),
                        float(extensions[index]),
                        vertices[index].id,
                    ),
                )[:64]
                candidates: list[tuple[float, float, np.ndarray, _Segment]] = []
                for candidate_index in candidate_indices:
                    target = vertices[candidate_index]
                    if target.id in own_ids or not _opposite_endpoint_has_inward_wall(
                        target.id,
                        outward,
                        vertices_by_id,
                        incident_edges,
                        cosine_tolerance,
                    ):
                        continue
                    movement = float(extensions[candidate_index])
                    point = np.round(
                        endpoint + outward * movement, decimals=_POINT_KEY_PRECISION
                    )
                    if not (
                        0.0 <= point[0] <= analysis.image_width - 1.0
                        and 0.0 <= point[1] <= analysis.image_height - 1.0
                    ):
                        continue
                    extension = _Segment(endpoint, point, face.confidence)
                    # Alignment compensates for detector-shortened line ends;
                    # the small extension may intentionally lie beyond the ink.
                    # Only a later, separate connection repair needs continuous
                    # ink across the much longer gap between opposing vertices.
                    # The graph rounds detector coordinates: its own incident
                    # edge can overlap this extension by a fraction of a pixel.
                    # Exempt that exact edge, never unrelated nearby geometry.
                    nearby_obstructions = [
                        obstructions[int(index)]
                        for index in _overlapping_bounds_indices(
                            extension, obstruction_bounds, 0.25
                        )
                        if obstruction_records[int(index)][0] is not source_incident_edge
                    ]
                    if _alignment_extension_crosses_geometry(
                        extension,
                        (*nearby_obstructions, *accepted_extensions),
                    ):
                        continue
                    candidates.append(
                        (float(spans[candidate_index]), movement, point, extension)
                    )

                if not candidates:
                    continue
                best = candidates[0]
                # Two faces at the same end coordinate support the same target,
                # not an ambiguity. Different nearly-equidistant end coordinates
                # are unsafe: leave them for manual editing instead of guessing.
                comparable_span = best[0] + max(2.0, best[0] * 0.1)
                if any(
                    candidate[0] <= comparable_span
                    and abs(candidate[1] - best[1]) > 0.5
                    for candidate in candidates[1:]
                ):
                    continue
                face_points[endpoint_index] = best[2]
                accepted_extensions.append(best[3])
            if all(
                np.array_equal(before, after)
                for before, after in zip(
                    (face.start, face.end),
                    face_points,
                    strict=True,
                )
            ):
                adjusted_faces[face_index] = face
            else:
                start, end = _canonical_endpoints(*face_points)
                adjusted_faces[face_index] = _Segment(start, end, face.confidence)
        if adjusted_faces[0] is not pair.first or adjusted_faces[1] is not pair.second:
            adjusted_pairs[pair_index] = _WallFacePair(*adjusted_faces)
    return adjusted_pairs


def _alignment_extension_crosses_geometry(
    extension: _Segment,
    obstructions: Sequence[_Segment],
) -> bool:
    """Do not extend through a crossing or past an overlapping wall segment."""

    if _connector_crosses_unrelated_geometry(extension, obstructions):
        return True
    direction = _unit_direction(extension)
    start, end = _projection_interval(extension, direction)
    for obstruction in obstructions:
        if not _segments_are_collinear(extension, obstruction):
            continue
        other_start, other_end = _projection_interval(obstruction, direction)
        if min(end, other_end) - max(start, other_start) > _GEOMETRY_EPSILON:
            return True
    return False


def _opposite_endpoint_has_inward_wall(
    vertex_id: int,
    source_outward: np.ndarray,
    vertices_by_id: dict[int, Vertex],
    incident_edges: dict[int, list[Edge]],
    cosine_tolerance: float,
) -> bool:
    """Require an opposite wall running inward from the proposed anchor."""

    vertex = vertices_by_id[vertex_id]
    point = np.asarray((vertex.x, vertex.y))
    for edge in incident_edges[vertex_id]:
        other_id = (
            edge.end_vertex_id
            if edge.start_vertex_id == vertex_id
            else edge.start_vertex_id
        )
        other = vertices_by_id[other_id]
        inward = np.asarray((other.x, other.y)) - point
        length = float(np.linalg.norm(inward))
        if (
            length > _GEOMETRY_EPSILON
            and float(np.dot(inward / length, -source_outward)) >= cosine_tolerance
        ):
            return True
    return False


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
    buckets: dict[tuple[int, int], list[_Segment]] = {}
    for candidate in sorted(segments, key=_segment_sort_key):
        cell_x, cell_y = _vertex_grid_cell(candidate.start, 0.25)
        if any(
            _same_segment(candidate, current)
            for x in range(cell_x - 1, cell_x + 2)
            for y in range(cell_y - 1, cell_y + 2)
            for current in buckets.get((x, y), ())
        ):
            continue
        kept.append(candidate)
        buckets.setdefault((cell_x, cell_y), []).append(candidate)
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
    for first_index, second_index in _overlapping_segment_pairs(mutable, 0.0):
        intersection = _bounded_segment_intersection(
            mutable[first_index], mutable[second_index]
        )
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
        for first_index, second_index in _overlapping_segment_pairs(
            result, snap_distance
        ):
            first, second = result[first_index], result[second_index]
            if _angle_difference(
                _segment_angle(first),
                _segment_angle(second),
            ) < math.radians(1.0):
                continue
            intersection_parameters = _line_intersection_parameters(first, second)
            if intersection_parameters is None:
                continue
            intersection, first_parameter, second_parameter = intersection_parameters
            first_distance = _outside_parameter_distance(first_parameter, first.length)
            second_distance = _outside_parameter_distance(
                second_parameter, second.length
            )
            if first_distance > snap_distance or second_distance > snap_distance:
                continue
            new_first = _extended_to_intersection(
                updated[first_index], intersection, first_parameter
            )
            new_second = _extended_to_intersection(
                updated[second_index], intersection, second_parameter
            )
            changed = changed or not _same_segment(new_first, updated[first_index])
            changed = changed or not _same_segment(new_second, updated[second_index])
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
                    if (
                        representative.existing_vertex_id is None
                        and _incidents_have_distinct_axis_support(
                            point_incidents,
                            representative.incident_segments,
                            generated_distance,
                        )
                    ):
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
        refined = _best_fit_representative_point(
            representative,
            maximum_adjustment,
        )
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
    minimum_axis_support_length: float,
) -> np.ndarray | None:
    """Fit line intersections without bending substantial exact axis supports."""

    segments = _deduplicate_segments(representative.incident_segments)
    if not segments or not representative.source_points:
        return None

    normal_matrix = np.zeros((2, 2), dtype=np.float64)
    projected_points = np.zeros(2, dtype=np.float64)
    axis_coordinates: list[list[float]] = [[], []]
    for segment in segments:
        if segment.length <= _GEOMETRY_EPSILON:
            continue
        direction = _unit_direction(segment)
        normal = np.asarray((-direction[1], direction[0]), dtype=np.float64)
        projection = np.outer(normal, normal)
        weight = segment.length * max(segment.confidence, _GEOMETRY_EPSILON)
        normal_matrix += weight * projection
        projected_points += weight * (projection @ segment.start)
        if segment.length > minimum_axis_support_length:
            if _is_vertical(segment):
                axis_coordinates[0].append(float(segment.start[0]))
            elif _is_horizontal(segment):
                axis_coordinates[1].append(float(segment.start[1]))

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
    fixed_axes = {
        axis: values[0]
        for axis, values in enumerate(axis_coordinates)
        if values and max(values) - min(values) <= _GEOMETRY_EPSILON
    }
    for axis, coordinate in fixed_axes.items():
        refined[axis] = coordinate
    if len(fixed_axes) == 1:
        fixed_axis, coordinate = next(iter(fixed_axes.items()))
        free_axis = 1 - fixed_axis
        if normal_matrix[free_axis, free_axis] > _GEOMETRY_EPSILON:
            refined[free_axis] = (
                projected_points[free_axis]
                - normal_matrix[free_axis, fixed_axis] * coordinate
            ) / normal_matrix[free_axis, free_axis]
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


def _incidents_have_distinct_axis_support(
    first: Sequence[_Segment],
    second: Sequence[_Segment],
    minimum_length: float,
) -> bool:
    """Preserve short rectangular steps between substantial straight walls.

    Merging the two ends of a small step averages different wall coordinates
    and bends both walls. The wall-thickness check below intentionally only
    protects larger separations, so these smaller real corners need their own
    constraint. Short fragments can still collapse, and collinear gap endpoints
    remain free to merge.
    """

    for first_segment in first:
        if first_segment.length <= minimum_length:
            continue
        axis = (
            0 if _is_vertical(first_segment)
            else 1 if _is_horizontal(first_segment)
            else None
        )
        if axis is None:
            continue
        for second_segment in second:
            if second_segment.length <= minimum_length:
                continue
            same_axis = (
                _is_vertical(second_segment)
                if axis == 0 else _is_horizontal(second_segment)
            )
            if not same_axis or abs(
                float(first_segment.start[axis] - second_segment.start[axis])
            ) <= _GEOMETRY_EPSILON:
                continue
            tangent_axis = 1 - axis
            first_interval = sorted((
                float(first_segment.start[tangent_axis]),
                float(first_segment.end[tangent_axis]),
            ))
            second_interval = sorted((
                float(second_segment.start[tangent_axis]),
                float(second_segment.end[tangent_axis]),
            ))
            overlap = min(first_interval[1], second_interval[1]) - max(
                first_interval[0], second_interval[0]
            )
            # Nearby duplicate detections of the same stroke overlap. A step
            # instead joins consecutive spans at different line coordinates.
            if overlap <= minimum_length:
                return True
    return False


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


# ### Image-supported wall-line repair ###
def _repair_image_supported_wall_lines(
    analysis: PlanWallAnalysis,
    graph: VertexData,
    immutable_graph: VertexData,
    wall_pairs: Sequence[_WallFacePair],
    options: PlanWallDetectionOptions,
) -> list[_Segment]:
    """Trace unused ink between supported wall anchors, including corners.

    A path must start at an unfinished wall and reach another wall. Its entire
    length needs structural ink; the gap-bridging slider does not authorize this
    pass to fill white doorway openings. Older analyses without pixels retain
    their original reconstruction behavior.
    """

    if analysis.structural_ink is None:
        return []
    guides = [
        segment
        for segment in _segments_from_analysis(analysis, options)
        if segment.confidence >= max(0.5, options.confidence_threshold)
    ]
    return _repair_supported_wall_guides(
        analysis,
        graph,
        immutable_graph,
        wall_pairs,
        options,
        guides,
        analysis.structural_ink,
    )


def _repair_pixel_supported_wall_lines(
    analysis: PlanWallAnalysis,
    graph: VertexData,
    immutable_graph: VertexData,
    wall_pairs: Sequence[_WallFacePair],
    options: PlanWallDetectionOptions,
) -> list[_Segment]:
    """Recover missing detector segments from independently traced ink paths."""

    if analysis.normalized_ink is None or not analysis.pixel_line_evidence:
        return []
    incidents = _repair_graph_incidents(graph)
    source_lines = [items[0] for items in incidents.values() if len(items) == 1]
    if not source_lines:
        return []
    components = _repair_pixel_source_components(analysis, graph, incidents)
    tolerance = math.radians(min(options.parallel_angle_tolerance_degrees, 15.0))
    # Keep even tiny corner pieces; applying the detector's minimum length here
    # would discard precisely the connectors this fallback needs to recover.
    guides = [
        _straightened_segment(
            _Segment(
                start=np.asarray(line.start, dtype=np.float64),
                end=np.asarray(line.end, dtype=np.float64),
                confidence=line.confidence,
            ),
            options.parallel_angle_tolerance_degrees,
        )
        for line in analysis.pixel_line_evidence
        if line.length > 0.25
        and _repair_pixel_line_is_relevant(
            line, analysis, components, source_lines, tolerance
        )
    ]
    guides = _repair_subtract_nearby_wall_faces(guides, graph)
    return _repair_supported_wall_guides(
        analysis,
        graph,
        immutable_graph,
        wall_pairs,
        options,
        guides,
        analysis.normalized_ink,
        maximum_turns=_PIXEL_REPAIR_MAXIMUM_TURNS,
        prune_unanchored=True,
    )


def _repair_pixel_source_components(
    analysis: PlanWallAnalysis,
    graph: VertexData,
    incidents: dict[int, list[_Segment]],
) -> set[int] | None:
    """Keep only ink neighborhoods reachable from unfinished wall endpoints."""

    labels = analysis.pixel_component_labels
    if labels is None:
        return None
    components: set[int] = set()
    for vertex in graph.vertices:
        if len(incidents.get(vertex.id, ())) != 1:
            continue
        x, y = round(vertex.x), round(vertex.y)
        if (
            x + 3 < 0
            or y + 3 < 0
            or x - 3 >= labels.shape[1]
            or y - 3 >= labels.shape[0]
        ):
            continue
        local = labels[
            max(0, y - 3) : min(labels.shape[0], y + 4),
            max(0, x - 3) : min(labels.shape[1], x + 4),
        ]
        components.update(int(value) for value in np.unique(local) if value)
    return components


def _repair_pixel_line_is_relevant(
    line: WallLineEvidence,
    analysis: PlanWallAnalysis,
    components: set[int] | None,
    source_lines: Sequence[_Segment],
    tolerance: float,
) -> bool:
    labels = analysis.pixel_component_labels
    if labels is not None and components is not None:
        x, y = round(line.start[0]), round(line.start[1])
        if not (0 <= x < labels.shape[1] and 0 <= y < labels.shape[0]):
            return False
        if int(labels[y, x]) not in components:
            return False
    angle = (
        math.atan2(line.end[1] - line.start[1], line.end[0] - line.start[0]) % math.pi
    )
    return any(
        (difference := _angle_difference(angle, _segment_angle(source))) <= tolerance
        or abs(difference - math.pi / 2.0) <= tolerance
        for source in source_lines
    )


def _repair_supported_wall_guides(
    analysis: PlanWallAnalysis,
    graph: VertexData,
    immutable_graph: VertexData,
    wall_pairs: Sequence[_WallFacePair],
    options: PlanWallDetectionOptions,
    guides: Sequence[_Segment],
    ink: np.ndarray,
    *,
    maximum_turns: int = _REPAIR_MAXIMUM_TURNS,
    prune_unanchored: bool = False,
) -> list[_Segment]:
    """Reconnect supported routes while keeping the authored graph immutable."""

    if not guides or not graph.edges:
        return []
    graph_incidents = _repair_graph_incidents(graph)
    sources = sorted(
        vertex_id for vertex_id, items in graph_incidents.items() if len(items) == 1
    )
    if not sources:
        return []
    repair_options = replace(
        options,
        maximum_gap_bridge_pixels=min(
            options.maximum_gap_bridge_pixels, _REPAIR_MAXIMUM_INK_GAP_PIXELS
        ),
        maximum_vertex_distance_pixels=min(options.maximum_vertex_distance_pixels, 2.0),
    )
    guides = _repair_remove_wall_interiors(guides, wall_pairs, options)
    if prune_unanchored:
        # Skeleton branches still share explicit endpoints here. Merging first
        # would hide a T-junction in the middle of a longer guide.
        guides = _repair_prune_unanchored_guides(
            guides,
            graph,
            anchor_tolerance=max(3.0, options.endpoint_snap_distance_pixels),
        )
    guides = _merge_collinear_segments(guides, repair_options)
    guides = [segment for segment in guides if _repair_line_has_ink(segment, ink)]
    # Subtraction eliminates already accepted faces before the more expensive
    # intersection work, while retaining their endpoints as exact anchors.
    guides = _subtract_existing_wall_geometry(guides, graph)
    if not guides:
        return []
    graph_segments = _existing_graph_segments(graph)
    pieces = _node_segments([*guides, *graph_segments], repair_options)
    # Generated wall endpoints are provisional, not authored anchors. Resolve
    # their intersections with the guides together: snapping a straight guide
    # independently onto two rounded generated tips can bend it off its ink.
    network = immutable_graph.clone()
    _append_segments_to_graph(
        network,
        pieces,
        analysis.image_width,
        analysis.image_height,
        repair_options,
    )
    points = {
        vertex.id: np.asarray((vertex.x, vertex.y), dtype=np.float64)
        for vertex in network.vertices
    }
    graph_segments, graph_incidents = _repair_rebased_graph_geometry(graph, network)
    sources = sorted(
        vertex_id for vertex_id, items in graph_incidents.items() if len(items) == 1
    )
    baseline_bounds = _build_segment_bounds(graph_segments)
    adjacency: dict[int, list[tuple[int, _Segment]]] = {}
    for edge in network.edges:
        start_id, end_id = edge.start_vertex_id, edge.end_vertex_id
        segment = _segment_from_vertices(
            network.get_vertex(start_id), network.get_vertex(end_id)
        )
        if any(
            _repair_point_on_segment(segment.start, graph_segments[int(index)])
            and _repair_point_on_segment(segment.end, graph_segments[int(index)])
            for index in _overlapping_bounds_indices(segment, baseline_bounds, 1e-4)
        ):
            continue
        if not _repair_line_has_ink(segment, ink):
            continue
        adjacency.setdefault(start_id, []).append((end_id, segment))
        adjacency.setdefault(end_id, []).append((start_id, segment))
    if not adjacency:
        return []

    anchors = set(graph_incidents)
    obstructions = set(anchors)
    immutable_segments = _existing_graph_segments(immutable_graph)
    for vertex_id, point in points.items():
        if vertex_id in anchors:
            continue
        touching = [
            segment
            for segment in graph_segments
            if _repair_point_on_segment(point, segment)
        ]
        if not touching:
            continue
        obstructions.add(vertex_id)
        # Existing edge records are immutable. Only freshly generated target
        # edges can be split by the final graph-noding pass.
        if not any(
            _repair_point_on_segment(point, segment) for segment in immutable_segments
        ):
            anchors.add(vertex_id)
            graph_incidents[vertex_id] = touching

    accepted: dict[tuple[int, int], _Segment] = {}
    degrees = {vertex_id: len(items) for vertex_id, items in graph_incidents.items()}
    # Each successful route consumes at least one dangling source, so iteration
    # is bounded even when later routes attach to an earlier repaired segment.
    for _iteration in range(len(sources)):
        changed = False
        for source_id in sources:
            if degrees.get(source_id) != 1 or source_id not in adjacency:
                continue
            path = _repair_best_supported_path(
                source_id,
                points,
                adjacency,
                graph_incidents,
                anchors,
                obstructions,
                accepted,
                options,
                maximum_turns=maximum_turns,
            )
            if not path:
                continue
            for start_id, end_id in pairwise(path):
                key = tuple(sorted((start_id, end_id)))
                if key in accepted:
                    continue
                start, end = _canonical_endpoints(points[start_id], points[end_id])
                repaired = _Segment(start=start, end=end, confidence=1.0)
                accepted[key] = repaired
                degrees[start_id] = degrees.get(start_id, 0) + 1
                degrees[end_id] = degrees.get(end_id, 0) + 1
                graph_incidents.setdefault(start_id, []).append(repaired)
                graph_incidents.setdefault(end_id, []).append(repaired)
                anchors.update((start_id, end_id))
                obstructions.update((start_id, end_id))
            changed = True
        if not changed:
            break
    return sorted(accepted.values(), key=_segment_sort_key)


def _repair_rebased_graph_geometry(
    graph: VertexData,
    network: VertexData,
) -> tuple[list[_Segment], dict[int, list[_Segment]]]:
    """Locate provisional wall anchors in the jointly resolved repair network."""

    if not network.vertices:
        return [], {}
    vertices = list(network.vertices)
    vertices_by_id = {vertex.id: vertex for vertex in vertices}
    coordinates = np.asarray([(vertex.x, vertex.y) for vertex in vertices])
    remapped_ids: dict[int, int] = {}
    for vertex in graph.vertices:
        retained = vertices_by_id.get(vertex.id)
        if retained is not None and (retained.x, retained.y) == (vertex.x, vertex.y):
            # Coincident authored vertices remain separate immutable anchors.
            remapped_ids[vertex.id] = vertex.id
            continue
        nearest = int(
            np.argmin(np.sum((coordinates - (vertex.x, vertex.y)) ** 2, axis=1))
        )
        remapped_ids[vertex.id] = vertices[nearest].id
    segments: list[_Segment] = []
    incidents: dict[int, list[_Segment]] = {}
    seen_edges: set[tuple[int, int]] = set()
    for edge in graph.edges:
        first_id = remapped_ids[edge.start_vertex_id]
        second_id = remapped_ids[edge.end_vertex_id]
        key = tuple(sorted((first_id, second_id)))
        if first_id == second_id or key in seen_edges:
            continue
        seen_edges.add(key)
        segment = _segment_from_vertices(
            network.get_vertex(first_id), network.get_vertex(second_id)
        )
        segments.append(segment)
        incidents.setdefault(first_id, []).append(segment)
        incidents.setdefault(second_id, []).append(segment)
    return segments, incidents


def _repair_graph_incidents(graph: VertexData) -> dict[int, list[_Segment]]:
    """Index connected anchors; isolated room-center vertices are not anchors."""

    vertices = {vertex.id: vertex for vertex in graph.vertices}
    incidents: dict[int, list[_Segment]] = {}
    for edge in graph.edges:
        segment = _segment_from_vertices(
            vertices[edge.start_vertex_id], vertices[edge.end_vertex_id]
        )
        incidents.setdefault(edge.start_vertex_id, []).append(segment)
        incidents.setdefault(edge.end_vertex_id, []).append(segment)
    return incidents


def _repair_subtract_nearby_wall_faces(
    guides: Sequence[_Segment],
    graph: VertexData,
) -> list[_Segment]:
    """Avoid duplicating detector stroke edges with pixel stroke centerlines."""

    existing = _existing_graph_segments(graph)
    existing_bounds = _build_segment_bounds(existing)
    retained: list[_Segment] = []
    for guide in guides:
        covered: list[tuple[float, float]] = []
        for index in _overlapping_bounds_indices(guide, existing_bounds, 2.0):
            wall = existing[int(index)]
            if _angle_difference(
                _segment_angle(guide), _segment_angle(wall)
            ) > math.radians(0.5):
                continue
            if (
                max(
                    _point_line_distance(guide.start, wall),
                    _point_line_distance(guide.end, wall),
                )
                > 2.0
            ):
                continue
            interval = _projected_overlap_parameters(guide, wall)
            if interval is not None:
                covered.append(interval)
        previous_end = 0.0
        for lower, upper in [*_merged_parameter_intervals(covered), (1.0, 1.0)]:
            if (lower - previous_end) * guide.length > 0.25:
                retained.append(
                    _Segment(
                        start=guide.start + previous_end * guide.vector,
                        end=guide.start + lower * guide.vector,
                        confidence=guide.confidence,
                    )
                )
            previous_end = upper
    return retained


def _repair_prune_unanchored_guides(
    guides: Sequence[_Segment],
    graph: VertexData,
    *,
    anchor_tolerance: float,
) -> list[_Segment]:
    """Trim text spurs before expensive geometric intersection processing.

    Only the temporary connectivity map groups nearby endpoints. Final wall
    coordinates and ink tests are unchanged by this conservative pruning.
    """

    if not guides:
        return []
    points: list[np.ndarray] = []
    buckets: dict[tuple[int, int], list[int]] = {}
    nodes_by_point: dict[tuple[float, float], int] = {}
    for point in sorted(
        (point for guide in guides for point in (guide.start, guide.end)),
        key=lambda item: (float(item[0]), float(item[1])),
    ):
        key = _point_key(point)
        if key in nodes_by_point:
            continue
        cell_x, cell_y = _vertex_grid_cell(point, 2.0)
        neighbors = [
            index
            for x in range(cell_x - 1, cell_x + 2)
            for y in range(cell_y - 1, cell_y + 2)
            for index in buckets.get((x, y), ())
            if math.dist(point, points[index]) <= 2.0
        ]
        if neighbors:
            nodes_by_point[key] = min(
                neighbors, key=lambda index: (math.dist(point, points[index]), index)
            )
        else:
            index = len(points)
            nodes_by_point[key] = index
            points.append(point)
            buckets.setdefault((cell_x, cell_y), []).append(index)
    ends = [
        (nodes_by_point[_point_key(guide.start)], nodes_by_point[_point_key(guide.end)])
        for guide in guides
    ]
    edges_by_node: list[set[int]] = [set() for _point in points]
    for index, (start_id, end_id) in enumerate(ends):
        edges_by_node[start_id].add(index)
        edges_by_node[end_id].add(index)
    walls = _existing_graph_segments(graph)
    wall_starts = np.asarray([wall.start for wall in walls])
    wall_vectors = np.asarray([wall.vector for wall in walls])
    anchored = np.zeros(len(points), dtype=bool)
    wall_lengths_squared = np.sum(wall_vectors**2, axis=1)
    # Bound temporary arrays even for large, annotation-heavy plan images.
    for start in range(0, len(points), 256):
        deltas = (
            np.asarray(points[start : start + 256])[:, None, :]
            - wall_starts[None, :, :]
        )
        fractions = np.clip(
            np.sum(deltas * wall_vectors, axis=2) / wall_lengths_squared,
            0.0,
            1.0,
        )
        distances = deltas - fractions[:, :, None] * wall_vectors
        anchored[start : start + 256] = np.any(
            np.sum(distances**2, axis=2) <= anchor_tolerance**2,
            axis=1,
        )
    pending = deque(
        index
        for index, items in enumerate(edges_by_node)
        if len(items) <= 1 and not anchored[index]
    )
    retained = set(range(len(guides)))
    while pending:
        node = pending.popleft()
        if anchored[node] or len(edges_by_node[node]) > 1:
            continue
        for edge_id in tuple(edges_by_node[node]):
            if edge_id not in retained:
                continue
            retained.remove(edge_id)
            for endpoint in ends[edge_id]:
                edges_by_node[endpoint].discard(edge_id)
                if not anchored[endpoint] and len(edges_by_node[endpoint]) <= 1:
                    pending.append(endpoint)
    return [guide for index, guide in enumerate(guides) if index in retained]


def _repair_remove_wall_interiors(
    segments: Sequence[_Segment],
    pairs: Sequence[_WallFacePair],
    options: PlanWallDetectionOptions,
) -> list[_Segment]:
    """Clip decorative layers out of wall strips, even on longer guides."""

    tolerance = math.radians(options.parallel_angle_tolerance_degrees)
    retained: list[_Segment] = []
    for segment in segments:
        interiors: list[tuple[float, float]] = []
        for pair in pairs:
            if (
                _angle_difference(_segment_angle(segment), _segment_angle(pair.first))
                > tolerance
            ):
                continue
            direction = _unit_direction(pair.first)
            normal = np.asarray((-direction[1], direction[0]))
            bounds = sorted(
                (
                    float(np.dot(pair.first.start, normal)),
                    float(np.dot(pair.second.start, normal)),
                )
            )
            offsets = [
                float(np.dot(point, normal)) for point in (segment.start, segment.end)
            ]
            if min(offsets) <= bounds[0] + 0.5 or max(offsets) >= bounds[1] - 0.5:
                continue
            first_interval = _projection_interval(pair.first, direction)
            second_interval = _projection_interval(pair.second, direction)
            origin = float(np.dot(segment.start, direction))
            span = float(np.dot(segment.vector, direction))
            lower = (max(first_interval[0], second_interval[0]) - origin) / span
            upper = (min(first_interval[1], second_interval[1]) - origin) / span
            lower, upper = max(0.0, lower), min(1.0, upper)
            if upper > lower:
                interiors.append((lower, upper))
        previous_end = 0.0
        for lower, upper in [*_merged_parameter_intervals(interiors), (1.0, 1.0)]:
            if (lower - previous_end) * segment.length > 0.25:
                start = segment.start + previous_end * segment.vector
                end = segment.start + lower * segment.vector
                retained.append(
                    _Segment(start=start, end=end, confidence=segment.confidence)
                )
            previous_end = upper
    return retained


def _repair_line_has_ink(segment: _Segment, ink: np.ndarray) -> bool:
    """Check coverage and the longest white interruption, at pixel spacing."""

    length = segment.length
    if length <= _GEOMETRY_EPSILON:
        return False
    sample_count = max(2, math.ceil(length) + 1)
    points = (
        segment.start + np.linspace(0.0, 1.0, sample_count)[:, None] * segment.vector
    )
    direction = _unit_direction(segment)
    normal = np.asarray((-direction[1], direction[0]))
    supported = np.zeros(sample_count, dtype=bool)
    for offset in (-1.0, 0.0, 1.0):
        samples = np.rint(points + offset * normal).astype(np.int32)
        inside = (
            (samples[:, 0] >= 0)
            & (samples[:, 0] < ink.shape[1])
            & (samples[:, 1] >= 0)
            & (samples[:, 1] < ink.shape[0])
        )
        supported[inside] |= ink[samples[inside, 1], samples[inside, 0]]
    if float(np.mean(supported)) < _REPAIR_MINIMUM_INK_COVERAGE:
        return False
    white = np.concatenate(([False], ~supported, [False])).astype(np.int8)
    transitions = np.diff(white)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    longest_gap = int(np.max(ends - starts)) if len(starts) else 0
    return longest_gap * length / (sample_count - 1) <= _REPAIR_MAXIMUM_INK_GAP_PIXELS


def _repair_point_on_segment(point: np.ndarray, segment: _Segment) -> bool:
    if _point_line_distance(point, segment) > 1e-4:
        return False
    parameter = float(np.dot(point - segment.start, segment.vector)) / segment.length**2
    return -_GEOMETRY_EPSILON <= parameter <= 1.0 + _GEOMETRY_EPSILON


def _repair_turn_is_supported(
    first: np.ndarray, second: np.ndarray, tolerance: float
) -> bool:
    cosine = float(np.dot(first, second))
    return cosine >= math.cos(tolerance) or abs(cosine) <= math.sin(tolerance)


def _repair_best_supported_path(
    source_id: int,
    points: dict[int, np.ndarray],
    adjacency: dict[int, list[tuple[int, _Segment]]],
    incidents: dict[int, list[_Segment]],
    anchors: set[int],
    obstructions: set[int],
    accepted: dict[tuple[int, int], _Segment],
    options: PlanWallDetectionOptions,
    *,
    maximum_turns: int = _REPAIR_MAXIMUM_TURNS,
) -> tuple[int, ...]:
    """Prefer the shortest straight continuation, then supported right turns."""

    tolerance = math.radians(min(options.parallel_angle_tolerance_degrees, 15.0))
    source_segment = incidents[source_id][0]
    other_point = (
        source_segment.end
        if math.dist(source_segment.start, points[source_id]) < 1e-4
        else source_segment.start
    )
    initial_direction = points[source_id] - other_point
    initial_direction /= np.linalg.norm(initial_direction)
    queue: list[tuple[int, float, tuple[int, ...]]] = [(0, 0.0, (source_id,))]
    best_costs: dict[tuple[int, int, int], float] = {}
    searched_states = 0
    while queue and searched_states < _REPAIR_MAXIMUM_SEARCH_STATES:
        searched_states += 1
        turns, cost, path = heapq.heappop(queue)
        vertex_id = path[-1]
        if len(path) > 1 and vertex_id in obstructions:
            incoming = points[vertex_id] - points[path[-2]]
            incoming /= np.linalg.norm(incoming)
            target_incidents = incidents.get(vertex_id, ())
            if vertex_id in anchors and any(
                _repair_turn_is_supported(
                    incoming, sign * _unit_direction(segment), tolerance
                )
                for segment in target_incidents
                for sign in (-1, 1)
            ):
                return path
            continue
        previous_direction = initial_direction
        if len(path) > 1:
            previous_direction = points[vertex_id] - points[path[-2]]
            previous_direction /= np.linalg.norm(previous_direction)
        for next_id, segment in sorted(
            adjacency.get(vertex_id, ()), key=lambda item: item[0]
        ):
            if next_id in path or tuple(sorted((vertex_id, next_id))) in accepted:
                continue
            direction = points[next_id] - points[vertex_id]
            direction /= np.linalg.norm(direction)
            if not _repair_turn_is_supported(previous_direction, direction, tolerance):
                continue
            is_turn = abs(float(np.dot(previous_direction, direction))) <= math.sin(
                tolerance
            )
            next_turns = turns + int(is_turn)
            if next_turns > maximum_turns:
                continue
            next_cost = cost + segment.length
            key = (vertex_id, next_id, next_turns)
            if next_cost >= best_costs.get(key, math.inf):
                continue
            best_costs[key] = next_cost
            heapq.heappush(queue, (next_turns, next_cost, (*path, next_id)))
    return ()


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
    existing_bounds = _build_segment_bounds(existing_segments)

    uncovered: list[_Segment] = []
    for generated in generated_segments:
        breakpoints = {0.0, 1.0}
        covered_intervals: list[tuple[float, float]] = []
        for index in _overlapping_bounds_indices(
            generated, existing_bounds, _EXISTING_WALL_COLLINEAR_TOLERANCE_PX,
        ):
            existing = existing_segments[int(index)]
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


def _build_segment_bounds(segments: Sequence[_Segment]) -> np.ndarray:
    """Cache box coordinates before repeated intersection candidate checks."""

    return np.asarray(
        [
            (
                min(segment.start[0], segment.end[0]),
                min(segment.start[1], segment.end[1]),
                max(segment.start[0], segment.end[0]),
                max(segment.start[1], segment.end[1]),
            )
            for segment in segments
        ],
        dtype=np.float64,
    ).reshape(-1, 4)


def _overlapping_bounds_indices(
    segment: _Segment,
    bounds: np.ndarray,
    margin: float,
) -> np.ndarray:
    lower = np.minimum(segment.start, segment.end) - margin
    upper = np.maximum(segment.start, segment.end) + margin
    return np.flatnonzero(
        (bounds[:, 0] <= upper[0])
        & (bounds[:, 2] >= lower[0])
        & (bounds[:, 1] <= upper[1])
        & (bounds[:, 3] >= lower[1])
    )


def _overlapping_segment_pairs(
    segments: Sequence[_Segment],
    margin: float,
) -> Iterable[tuple[int, int]]:
    """Keep pair order stable while rejecting distant boxes in NumPy."""

    bounds = _build_segment_bounds(segments)
    for first_index, first in enumerate(segments[:-1]):
        for offset in _overlapping_bounds_indices(
            first, bounds[first_index + 1 :], margin
        ):
            yield first_index, first_index + 1 + int(offset)


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
