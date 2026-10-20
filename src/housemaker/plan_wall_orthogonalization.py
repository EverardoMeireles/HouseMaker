# ### Imports ###
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np
import shapely

from housemaker.models import Vertex, VertexData

# ### Constants ###
_EPSILON = 1e-6
_COORDINATE_SEARCH_STEP = 0.25
_MAX_INK_DISTANCE_PIXELS = 3.0
_MAX_INK_ERROR_INCREASE_PIXELS = 0.35
_MOVEMENT_COST_WEIGHT = 0.025
_MAX_EDGE_SAMPLES = 192
_MIN_SPACING_RATIO = 0.5


# ### Constraint models ###
@dataclass(frozen=True)
class _CoordinateGroup:
    axis: int
    indices: np.ndarray
    edge_indices: np.ndarray
    lower: float
    upper: float
    coordinate: float


class _DisjointSets:
    def __init__(self, count: int) -> None:
        self.parents = list(range(count))

    def find(self, index: int) -> int:
        while self.parents[index] != index:
            self.parents[index] = self.parents[self.parents[index]]
            index = self.parents[index]
        return index

    def union(self, first: int, second: int) -> None:
        first, second = self.find(first), self.find(second)
        if first != second:
            self.parents[max(first, second)] = min(first, second)


# ### Public correction pass ###
def orthogonalize_wall_graph(
    graph: VertexData,
    *,
    fixed_vertex_ids: Iterable[int] = (),
    ink: np.ndarray | None = None,
    image_size: tuple[int, int] | None = None,
    maximum_adjustment_pixels: float = 6.0,
    angle_tolerance_degrees: float = 8.0,
) -> bool:
    """Fit generated near-axis runs to ink without changing graph topology.

    Coordinate constraints are shared by the entire run. Authored vertices
    and intentional diagonal endpoints stay fixed. Unsafe constraints are
    discarded together, leaving the remaining problem visible for review.
    """

    if not math.isfinite(maximum_adjustment_pixels) or maximum_adjustment_pixels < 0.0:
        raise ValueError("Wall correction distance must be finite and nonnegative.")
    if (
        not math.isfinite(angle_tolerance_degrees)
        or not 0.0 < angle_tolerance_degrees < 45.0
    ):
        raise ValueError("Wall correction angle must be between 0 and 45 degrees.")
    if not graph.edges or maximum_adjustment_pixels <= _EPSILON:
        return False
    points = np.asarray(
        [(vertex.x, vertex.y) for vertex in graph.vertices], dtype=float
    )
    indices = {vertex.id: index for index, vertex in enumerate(graph.vertices)}
    edges = np.asarray(
        [
            (indices[edge.start_vertex_id], indices[edge.end_vertex_id])
            for edge in graph.edges
        ],
        dtype=np.int64,
    )
    fixed_ids = frozenset(fixed_vertex_ids)
    fixed = np.asarray(
        [vertex.id in fixed_ids for vertex in graph.vertices], dtype=bool
    )
    distances = _ink_distance_map(ink)
    if image_size is None and distances is not None:
        image_size = (distances.shape[1], distances.shape[0])
    groups = _build_coordinate_groups(
        points,
        edges,
        fixed,
        maximum_adjustment_pixels,
        angle_tolerance_degrees,
        image_size,
    )
    if not groups:
        return False
    proposed = points.copy()
    group_by_coordinate = np.full(points.shape, -1, dtype=np.int64)
    for group_index, group in enumerate(groups):
        proposed[group.indices, group.axis] = group.coordinate
        group_by_coordinate[group.indices, group.axis] = group_index
    if distances is not None:
        # A second sweep lets each perpendicular run account for its newly
        # aligned neighbours while retaining the same bounded constraints.
        for _sweep in range(2):
            for group in groups:
                coordinate = _best_ink_coordinate(
                    group, points, proposed, edges, distances
                )
                proposed[group.indices, group.axis] = coordinate
    active = set(range(len(groups)))
    while active:
        rejected = (
            _unsafe_groups(
                points,
                proposed,
                edges,
                group_by_coordinate,
                maximum_adjustment_pixels,
                distances,
            )
            & active
        )
        if not rejected:
            break
        for group_index in rejected:
            group = groups[group_index]
            proposed[group.indices, group.axis] = points[group.indices, group.axis]
        active.difference_update(rejected)
    # Commit every member of an accepted coordinate group, including tiny
    # adjustments, so the returned run shares exactly the same coordinate.
    changed = np.any(proposed != points, axis=1)
    if not np.any(changed):
        return False
    graph.vertices = [
        Vertex(vertex.id, float(proposed[index, 0]), float(proposed[index, 1]))
        if changed[index]
        else vertex
        for index, vertex in enumerate(graph.vertices)
    ]
    graph.mark_changed()
    return True


# ### Connected coordinate constraints ###
def _build_coordinate_groups(
    points: np.ndarray,
    edges: np.ndarray,
    fixed: np.ndarray,
    maximum_adjustment: float,
    angle_tolerance: float,
    image_size: tuple[int, int] | None,
) -> list[_CoordinateGroup]:
    sets = (_DisjointSets(len(points)), _DisjointSets(len(points)))
    weights = np.zeros(points.shape, dtype=float)
    limit = math.tan(math.radians(angle_tolerance))
    protected = fixed.copy()
    for first, second in edges:
        delta = np.abs(points[second] - points[first])
        if max(delta) <= _EPSILON:
            protected[[first, second]] = True
            continue
        axis = (
            0
            if delta[0] <= delta[1] * limit
            else (1 if delta[1] <= delta[0] * limit else None)
        )
        if axis is None:
            protected[[first, second]] = True
            continue
        sets[axis].union(int(first), int(second))
        weights[[first, second], axis] += float(np.linalg.norm(delta))
    result: list[_CoordinateGroup] = []
    for axis, disjoint in enumerate(sets):
        members: dict[int, list[int]] = {}
        for index in range(len(points)):
            members.setdefault(disjoint.find(index), []).append(index)
        for member_indices in members.values():
            if len(member_indices) < 2:
                continue
            selected = np.asarray(member_indices, dtype=np.int64)
            values = points[selected, axis]
            if float(np.ptp(values)) <= _EPSILON:
                continue
            lower = float(np.max(values - maximum_adjustment))
            upper = float(np.min(values + maximum_adjustment))
            if image_size is not None:
                lower, upper = max(0.0, lower), min(image_size[axis] - 1.0, upper)
            anchors = values[protected[selected]]
            if len(anchors):
                if np.any(anchors != anchors[0]):
                    continue
                lower, upper = (
                    max(lower, float(anchors[0])),
                    min(upper, float(anchors[0])),
                )
            if lower > upper + _EPSILON:
                continue
            coordinate = float(
                np.clip(
                    np.average(
                        values, weights=np.maximum(weights[selected, axis], _EPSILON)
                    ),
                    lower,
                    upper,
                )
            )
            incident = np.flatnonzero(np.any(np.isin(edges, selected), axis=1))
            result.append(
                _CoordinateGroup(axis, selected, incident, lower, upper, coordinate)
            )
    return result


# ### Black-line fitting ###
def _ink_distance_map(ink: np.ndarray | None) -> np.ndarray | None:
    if ink is None:
        return None
    mask = np.asarray(ink, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("Wall correction ink must be a two-dimensional mask.")
    if not np.any(mask):
        return None
    return cv2.distanceTransform(
        (~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )


def _sample_ink_distances(points: np.ndarray, distances: np.ndarray) -> np.ndarray:
    x = np.clip(points[..., 0], 0.0, distances.shape[1] - 1.0)
    y = np.clip(points[..., 1], 0.0, distances.shape[0] - 1.0)
    left, top = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    right = np.minimum(left + 1, distances.shape[1] - 1)
    bottom = np.minimum(top + 1, distances.shape[0] - 1)
    dx, dy = x - left, y - top
    sampled = (
        distances[top, left] * (1.0 - dx) * (1.0 - dy)
        + distances[top, right] * dx * (1.0 - dy)
        + distances[bottom, left] * (1.0 - dx) * dy
        + distances[bottom, right] * dx * dy
    )
    return np.minimum(sampled, _MAX_INK_DISTANCE_PIXELS)


def _edge_ink_costs(segments: np.ndarray, distances: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    sample_count = min(_MAX_EDGE_SAMPLES, max(3, math.ceil(float(np.max(lengths))) + 1))
    ratios = np.linspace(0.0, 1.0, sample_count)
    samples = (
        segments[:, :1] + (segments[:, 1:] - segments[:, :1]) * ratios[None, :, None]
    )
    errors = _sample_ink_distances(samples, distances)
    endpoints = errors[:, (0, -1)]
    return (
        np.mean(errors, axis=1)
        + 0.5 * np.mean(endpoints, axis=1)
        + 0.5 * np.max(endpoints, axis=1)
    )


def _best_ink_coordinate(
    group: _CoordinateGroup,
    original: np.ndarray,
    proposed: np.ndarray,
    edges: np.ndarray,
    distances: np.ndarray,
) -> float:
    if group.upper - group.lower <= _EPSILON:
        return group.coordinate
    candidates = np.unique(
        np.concatenate(
            (
                np.arange(
                    math.ceil(group.lower / _COORDINATE_SEARCH_STEP)
                    * _COORDINATE_SEARCH_STEP,
                    group.upper + _EPSILON,
                    _COORDINATE_SEARCH_STEP,
                ),
                [group.coordinate, float(proposed[group.indices[0], group.axis])],
                original[group.indices, group.axis],
            )
        )
    )
    candidates = candidates[(candidates >= group.lower) & (candidates <= group.upper)]
    relevant_edges = edges[group.edge_indices]
    segments = proposed[relevant_edges]
    original_segments = original[relevant_edges]
    weights = np.maximum(
        np.linalg.norm(original_segments[:, 1] - original_segments[:, 0], axis=1), 1.0
    )
    affected = np.isin(relevant_edges, group.indices)
    best: tuple[float, float, float] | None = None
    for coordinate in candidates:
        candidate_segments = segments.copy()
        candidate_segments[:, :, group.axis] = np.where(
            affected, coordinate, candidate_segments[:, :, group.axis]
        )
        movement = float(
            np.mean((coordinate - original[group.indices, group.axis]) ** 2)
        )
        score = float(
            np.average(_edge_ink_costs(candidate_segments, distances), weights=weights)
        )
        ranking = (
            score + _MOVEMENT_COST_WEIGHT * movement,
            movement,
            float(coordinate),
        )
        if best is None or ranking < best:
            best = ranking
    return group.coordinate if best is None else best[2]


# ### Movement, topology and image safeguards ###
def _unsafe_groups(
    original: np.ndarray,
    proposed: np.ndarray,
    edges: np.ndarray,
    group_by_coordinate: np.ndarray,
    maximum_adjustment: float,
    distances: np.ndarray | None,
) -> set[int]:
    rejected: set[int] = set()
    changed = np.any(np.abs(proposed - original) > _EPSILON, axis=1)

    def reject_vertices(indices: np.ndarray) -> None:
        coordinates = group_by_coordinate[indices]
        moving = np.abs(proposed[indices] - original[indices]) > _EPSILON
        rejected.update(int(value) for value in coordinates[moving] if value >= 0)

    too_far = np.flatnonzero(
        np.linalg.norm(proposed - original, axis=1) > maximum_adjustment + _EPSILON
    )
    reject_vertices(too_far)
    for index in np.flatnonzero(changed):
        coincident = np.flatnonzero(
            (np.linalg.norm(proposed - proposed[index], axis=1) <= _EPSILON)
            & (np.linalg.norm(original - original[index], axis=1) > _EPSILON)
        )
        if len(coincident):
            reject_vertices(np.concatenate(([index], coincident)))
    before, after = original[edges], proposed[edges]
    vectors = before[:, 1] - before[:, 0]
    next_vectors = after[:, 1] - after[:, 0]
    major_axes = np.argmax(np.abs(vectors), axis=1)
    major = vectors[np.arange(len(edges)), major_axes]
    next_major = next_vectors[np.arange(len(edges)), major_axes]
    collapsed = np.flatnonzero(
        (major * next_major <= _EPSILON)
        | (np.linalg.norm(next_vectors, axis=1) < np.linalg.norm(vectors, axis=1) * 0.5)
    )
    reject_vertices(edges[collapsed].reshape(-1))
    moved_edges = np.any(changed[edges], axis=1)
    if distances is not None and np.any(moved_edges):
        selected = np.flatnonzero(moved_edges)
        old_costs = _edge_ink_costs(before[selected], distances)
        new_costs = _edge_ink_costs(after[selected], distances)
        unsupported = selected[new_costs > old_costs + _MAX_INK_ERROR_INCREASE_PIXELS]
        reject_vertices(edges[unsupported].reshape(-1))
    old_lines = shapely.linestrings(before)
    new_lines = shapely.linestrings(after)
    lower, upper = np.min(after, axis=1), np.max(after, axis=1)
    for first in np.flatnonzero(moved_edges):
        nearby = np.flatnonzero(
            np.all(
                (upper >= lower[first] - _EPSILON) & (lower <= upper[first] + _EPSILON),
                axis=1,
            )
        )
        for second in nearby:
            if second == first or (moved_edges[second] and second < first):
                continue
            old_intersection = old_lines[first].intersection(old_lines[second])
            new_intersection = new_lines[first].intersection(new_lines[second])
            if (
                not new_intersection.is_empty and old_intersection.is_empty
            ) or new_intersection.length > old_intersection.length + _EPSILON:
                reject_vertices(edges[[first, second]].reshape(-1))
    # Nearby parallel runs must not converge onto one black stroke. Checking
    # distance also protects narrow wall faces that do not actually intersect.
    # Both runs may move towards each other, so include every original gap
    # that could shrink below the retained-spacing ratio within that budget.
    spacing_search_radius = 2.0 * maximum_adjustment / (1.0 - _MIN_SPACING_RATIO)
    for first in np.flatnonzero(moved_edges):
        candidates = np.flatnonzero(
            np.all(
                (
                    np.max(before, axis=1) + spacing_search_radius
                    >= np.min(before[first], axis=0)
                )
                & (
                    np.min(before, axis=1) - spacing_search_radius
                    <= np.max(before[first], axis=0)
                ),
                axis=1,
            )
        )
        for second in candidates:
            if second == first or np.intersect1d(edges[first], edges[second]).size:
                continue
            first_length, second_length = np.linalg.norm(
                vectors[[first, second]], axis=1
            )
            if first_length <= _EPSILON or second_length <= _EPSILON:
                continue
            if (
                abs(
                    float(
                        np.dot(vectors[first], vectors[second])
                        / (first_length * second_length)
                    )
                )
                < 0.99
            ):
                continue
            old_gap = old_lines[first].distance(old_lines[second])
            if (
                old_gap > _EPSILON
                and new_lines[first].distance(new_lines[second])
                < old_gap * _MIN_SPACING_RATIO
            ):
                reject_vertices(edges[[first, second]].reshape(-1))
    return rejected
