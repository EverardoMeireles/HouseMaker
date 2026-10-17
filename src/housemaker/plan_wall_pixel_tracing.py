# ### Imports ###
from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise

import cv2
import numpy as np

# ### Constants ###
_DIRECTIONS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
)
_OPPOSITE_DIRECTION = (4, 5, 6, 7, 0, 1, 2, 3)
_BIT_COUNTS = np.array([value.bit_count() for value in range(256)], dtype=np.uint8)
_FIRST_DIRECTION = tuple(
    (value & -value).bit_length() - 1 if value else -1 for value in range(256)
)
_SIMPLIFICATION_TOLERANCE = 1.25
_MAX_CORNER_BEVEL = 24.0
_NEIGHBOR_CODE_KERNEL = np.array(
    ((128, 1, 2), (64, 0, 4), (32, 16, 8)), dtype=np.float32
)


# ### Skeleton topology lookup ###
def _build_thinning_lookup_tables() -> tuple[np.ndarray, np.ndarray]:
    codes = np.arange(256, dtype=np.uint16)
    neighbors = tuple((codes >> shift) & 1 for shift in range(8))
    transitions = np.zeros(256, dtype=np.uint8)
    for previous, following in zip(neighbors, neighbors[1:] + neighbors[:1]):
        transitions += (previous == 0) & (following != 0)
    candidates = (_BIT_COUNTS >= 2) & (_BIT_COUNTS <= 6) & (transitions == 1)
    north, _, east, _, south, _, west, _ = neighbors
    first = candidates & ((north * east * south) == 0) & ((east * south * west) == 0)
    second = candidates & ((north * east * west) == 0) & ((north * south * west) == 0)
    return first.astype(np.uint8), second.astype(np.uint8)


_THINNING_REMOVAL_TABLES = _build_thinning_lookup_tables()


# ### Cancellation ###
class PixelWallTracingCancelled(Exception):
    """Raised when a caller cancels plan-ink tracing."""


def _check_cancelled(cancellation_check: Callable[[], bool] | None) -> None:
    if cancellation_check is not None and cancellation_check():
        raise PixelWallTracingCancelled("Plan wall pixel tracing was cancelled.")


# ### Public API ###
def trace_plan_ink_paths(
    ink: np.ndarray,
    cancellation_check: Callable[[], bool] | None = None,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    """Trace actual plan ink into connected, simplified image-coordinate paths.

    True pixels represent black drawing ink. No gap closing, dilation, or line
    detector is used: thin strokes survive and disconnected strokes remain
    disconnected. Paths terminate at endpoints and branches; closed loops are
    returned with their first point repeated at the end.
    """

    if not isinstance(ink, np.ndarray) or ink.ndim != 2:
        raise ValueError("Plan ink must be a two-dimensional NumPy array.")
    _check_cancelled(cancellation_check)
    if ink.size == 0 or not np.any(ink):
        return ()

    occupied_rows = np.flatnonzero(np.any(ink, axis=1))
    occupied_columns = np.flatnonzero(np.any(ink, axis=0))
    top, bottom = int(occupied_rows[0]), int(occupied_rows[-1]) + 1
    left, right = int(occupied_columns[0]), int(occupied_columns[-1]) + 1
    # A white border also makes strokes touching the image boundary traceable.
    original = np.pad(
        np.asarray(ink[top:bottom, left:right], dtype=bool).astype(np.uint8), 1
    )
    skeleton = _thin_ink(original, cancellation_check)
    adjacency = _skeleton_adjacency(skeleton)
    raw_paths = _trace_graph(adjacency, cancellation_check)
    paths = []
    for index, path in enumerate(raw_paths):
        if index % 128 == 0:
            _check_cancelled(cancellation_check)
        simplified = _simplify_path(path, original)
        paths.append(
            tuple((float(x + left - 1), float(y + top - 1)) for x, y in simplified)
        )
    return tuple(paths)


# ### Topology-preserving skeletonization ###
def _thin_ink(
    original: np.ndarray,
    cancellation_check: Callable[[], bool] | None,
) -> np.ndarray:
    """Zhang–Suen thinning retains thin runs, intersections, and white gaps."""

    skeleton = original.copy()
    while True:
        _check_cancelled(cancellation_check)
        changed = False
        for lookup in _THINNING_REMOVAL_TABLES:
            # Encoding the eight neighbors lets OpenCV perform the costly image
            # passes; the lookup applies the exact Zhang–Suen topology rules.
            codes = cv2.filter2D(
                skeleton, -1, _NEIGHBOR_CODE_KERNEL, borderType=cv2.BORDER_CONSTANT
            )
            removable = cv2.LUT(codes, lookup) & skeleton
            if np.any(removable):
                skeleton -= removable
                changed = True
        if not changed:
            return skeleton


# ### Skeleton graph tracing ###
def _skeleton_adjacency(skeleton: np.ndarray) -> np.ndarray:
    adjacency = np.zeros(skeleton.shape, dtype=np.uint8)
    center = skeleton[1:-1, 1:-1] != 0
    target = adjacency[1:-1, 1:-1]
    height, width = center.shape
    for direction, (dy, dx) in enumerate(_DIRECTIONS):
        connected = center & (
            skeleton[1 + dy:1 + dy + height, 1 + dx:1 + dx + width] != 0
        )
        if dy and dx:
            # A diagonal must not short-circuit the two arms of a square corner.
            connected &= skeleton[1:1 + height, 1 + dx:1 + dx + width] == 0
            connected &= skeleton[1 + dy:1 + dy + height, 1:1 + width] == 0
        target |= connected.astype(np.uint8) << direction
    return adjacency


def _trace_graph(
    adjacency: np.ndarray,
    cancellation_check: Callable[[], bool] | None,
) -> list[np.ndarray]:
    width = adjacency.shape[1]
    links = adjacency.ravel()
    visited = np.zeros(links.shape, dtype=np.uint8)
    degrees = _BIT_COUNTS[links]
    offsets = tuple(dy * width + dx for dy, dx in _DIRECTIONS)
    paths = []
    traced_count = 0

    def trace(start: int, direction: int) -> None:
        nonlocal traced_count
        nodes = [start]
        current = start
        while True:
            following = current + offsets[direction]
            visited[current] |= 1 << direction
            visited[following] |= 1 << _OPPOSITE_DIRECTION[direction]
            nodes.append(following)
            traced_count += 1
            if traced_count % 2048 == 0:
                _check_cancelled(cancellation_check)
            if following == start or degrees[following] != 2:
                break
            available = int(links[following]) & ~int(visited[following])
            if not available:
                break
            current = following
            direction = _FIRST_DIRECTION[available]
        points = np.asarray(nodes, dtype=np.int64)
        paths.append(np.column_stack((points % width, points // width)))

    # First split at graph branches/endpoints, then trace remaining closed loops.
    for candidates in (
        np.flatnonzero((degrees != 0) & (degrees != 2)),
        np.flatnonzero(degrees == 2),
    ):
        for index, candidate in enumerate(candidates):
            if index % 2048 == 0:
                _check_cancelled(cancellation_check)
            start = int(candidate)
            while available := int(links[start]) & ~int(visited[start]):
                trace(start, _FIRST_DIRECTION[available])
    return paths


# ### Supported line and corner simplification ###
def _segment_has_ink(start: np.ndarray, end: np.ndarray, ink: np.ndarray) -> bool:
    steps = int(np.ceil(np.max(np.abs(end - start)))) + 1
    samples = np.rint(np.linspace(start, end, max(2, steps))).astype(np.int32)
    return bool(np.all(ink[samples[:, 1], samples[:, 0]] != 0))


def _simplify_path(path: np.ndarray, ink: np.ndarray) -> np.ndarray:
    if len(path) <= 2:
        return path
    approximate = cv2.approxPolyDP(
        path.astype(np.float32).reshape(-1, 1, 2),
        _SIMPLIFICATION_TOLERANCE,
        False,
    ).reshape(-1, 2).astype(np.int64)
    if len(approximate) < 2:
        return path
    node_indices = {tuple(point): index for index, point in enumerate(path)}
    node_indices[tuple(path[0])] = 0
    indices = [node_indices[tuple(point)] for point in approximate]
    indices[0], indices[-1] = 0, len(path) - 1
    supported = [path[0]]

    def append_supported(first: int, last: int) -> None:
        if last - first <= 1 or _segment_has_ink(path[first], path[last], ink):
            supported.append(path[last])
            return
        middle = (first + last) // 2
        append_supported(first, middle)
        append_supported(middle, last)

    for first, last in pairwise(indices):
        append_supported(first, last)
    straightened = _straighten_supported_runs(np.asarray(supported), path, ink)
    return _restore_square_corners(straightened, ink)


def _straighten_supported_runs(
    points: np.ndarray, raw_path: np.ndarray, ink: np.ndarray
) -> np.ndarray:
    """Fit almost-axis-aligned skeleton runs without moving off drawing ink."""

    raw_indices = {tuple(point): index for index, point in enumerate(raw_path)}
    raw_indices[tuple(raw_path[0])] = 0
    runs: list[tuple[int, int] | None] = []
    for index, (start, end) in enumerate(pairwise(points)):
        dx, dy = np.abs(end - start)
        first = raw_indices[tuple(start)]
        last = len(raw_path) - 1 if index == len(points) - 2 else raw_indices[tuple(end)]
        if dx >= 4 and dy <= 2:
            runs.append((1, round(float(np.median(raw_path[first:last + 1, 1])))))
        elif dy >= 4 and dx <= 2:
            runs.append((0, round(float(np.median(raw_path[first:last + 1, 0])))))
        else:
            runs.append(None)
    adjusted = points.copy()
    closed = bool(np.array_equal(points[0], points[-1]))
    for index in range(len(points)):
        previous = runs[index - 1] if index else (runs[-1] if closed else None)
        following = runs[index] if index < len(runs) else (runs[0] if closed else None)
        for run in (previous, following):
            if run is not None:
                coordinate, value = run
                adjusted[index, coordinate] = value
    # Reject any local straightening that crosses even a one-pixel white break.
    while True:
        invalid = set()
        for index, (start, end) in enumerate(pairwise(adjusted)):
            if not _segment_has_ink(start, end, ink):
                invalid.update((index, index + 1))
        changed = False
        for index in invalid:
            if not np.array_equal(adjusted[index], points[index]):
                adjusted[index] = points[index]
                changed = True
        if closed and (0 in invalid or len(points) - 1 in invalid):
            adjusted[0], adjusted[-1] = points[0], points[-1]
        if not changed:
            return adjusted


def _restore_square_corners(path: np.ndarray, ink: np.ndarray) -> np.ndarray:
    """Remove short diagonal bevels introduced by thinning thick right angles."""

    points = list(path)
    index = 0
    while index + 3 < len(points):
        first, before, after, last = points[index:index + 4]
        incoming, outgoing = before - first, last - after
        horizontal_then_vertical = (
            abs(incoming[0]) >= 3 and abs(incoming[1]) <= 1
            and abs(outgoing[1]) >= 3 and abs(outgoing[0]) <= 1
        )
        vertical_then_horizontal = (
            abs(incoming[1]) >= 3 and abs(incoming[0]) <= 1
            and abs(outgoing[0]) >= 3 and abs(outgoing[1]) <= 1
        )
        if not (horizontal_then_vertical or vertical_then_horizontal):
            index += 1
            continue
        if np.linalg.norm(after - before) > _MAX_CORNER_BEVEL:
            index += 1
            continue
        corner = np.array(
            (after[0], before[1]) if horizontal_then_vertical else (before[0], after[1])
        )
        if _segment_has_ink(first, corner, ink) and _segment_has_ink(corner, last, ink):
            points[index + 1:index + 3] = [corner]
        else:
            index += 1
    return np.asarray(points, dtype=np.int64)
