"""Resolve and move four-corner window frames without stretching their host."""

# ### Imports ###
from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace

from shapely.geometry import LineString

from .models import LevelData, Vertex, WindowData

# ### Constants ###
_EPSILON = 1e-7
_ALIGNMENT_TOLERANCE_PIXELS = 0.5
_MAXIMUM_INFERRED_DEPTH_METERS = 1.5
_WALL_ID_PATTERN = re.compile(r"level:(\d+)/(?:room:\d+/)?wall:(\d+):(\d+)")


# ### Linked frame model ###
@dataclass(frozen=True)
class LinkedWindowFrame:
    """An opening inside the available image-space span shared by both faces."""

    corner_vertex_ids: tuple[int, int, int, int]
    start_point: tuple[float, float]
    end_point: tuple[float, float]
    start_ratio: float
    end_ratio: float


# ### Public topology helpers ###
def resolve_linked_window_frame(
    level: LevelData, window: WindowData
) -> LinkedWindowFrame | None:
    """Find the frame's four corners and the run available for slide/resize.

    Existing complete edge-bound windows may be inferred conservatively.
    Ordinary partial wall openings, slopes and bare room rectangles stay
    unlinked. A run stops before the nearest fixed outside node on either
    face, so moving a frame never reorders an adjacent authored wall edge.
    """

    vertices, neighbors = _wall_graph(level)
    corners = _resolve_corner_ids(level, window, (vertices, neighbors))
    if corners is None:
        return None
    first, second = (vertices[key] for key in corners[:2])
    axis = _unit(_subtract(_point(second), _point(first)))
    left_distances = [
        _outward_distance(key, axis, -1.0, vertices, neighbors, set(corners))
        for key in (corners[0], corners[2])
    ]
    right_distances = [
        _outward_distance(key, axis, 1.0, vertices, neighbors, set(corners))
        for key in (corners[1], corners[3])
    ]
    if any(value is None for value in (*left_distances, *right_distances)):
        return None
    width = math.dist(_point(first), _point(second))
    minimum, maximum = -min(left_distances), width + min(right_distances)
    available = maximum - minimum
    return LinkedWindowFrame(
        corner_vertex_ids=corners,
        start_point=_add_scaled(_point(first), axis, minimum),
        end_point=_add_scaled(_point(first), axis, maximum),
        start_ratio=-minimum / available,
        end_ratio=(width - minimum) / available,
    )


def move_linked_window_vertices(
    level: LevelData,
    window: WindowData,
    start_point: tuple[float, float],
    end_point: tuple[float, float],
) -> WindowData:
    """Atomically slide/resize all four corners and return the linked record.

    No level window list is changed. Endpoint order, wall depth, vertex IDs,
    graph connectivity and the opening's vertical ratios are preserved.
    A stale/invalid frame or a move through a neighbor fails without mutation.
    """

    frame = resolve_linked_window_frame(level, window)
    if frame is None:
        raise ValueError("The window frame's four wall corners are no longer valid.")
    start_point, end_point = _finite_point(start_point), _finite_point(end_point)
    graph = level.vertex_data
    corners = frame.corner_vertex_ids
    old_points = tuple(_point(graph.get_vertex(key)) for key in corners)
    axis = _unit(_subtract(old_points[1], old_points[0]))
    new_width = _dot(_subtract(end_point, start_point), axis)
    if new_width <= _EPSILON or any(
        abs(_cross(_subtract(point, old_points[0]), axis)) > _ALIGNMENT_TOLERANCE_PIXELS
        for point in (start_point, end_point)
    ):
        raise ValueError("The window frame must retain a positive, aligned width.")
    # Project tiny pointer roundoff back onto the physical front face.
    start_point, end_point = (
        _add_scaled(old_points[0], axis, _dot(_subtract(point, old_points[0]), axis))
        for point in (start_point, end_point)
    )
    available = math.dist(frame.start_point, frame.end_point)
    positions = tuple(
        _dot(_subtract(point, frame.start_point), axis)
        for point in (start_point, end_point)
    )
    if positions[0] <= _EPSILON or positions[1] >= available - _EPSILON:
        raise ValueError("The window frame cannot cross an adjacent wall corner.")
    protected = _other_opening_corner_ids(level, window)
    if protected.intersection(corners):
        raise ValueError("A window frame cannot move another opening's wall corners.")
    depth = _subtract(old_points[2], old_points[0])
    new_points = (
        start_point,
        end_point,
        _add(start_point, depth),
        _add(end_point, depth),
    )
    new_points = tuple(
        old if math.dist(old, new) <= _EPSILON else new
        for old, new in zip(old_points, new_points)
    )
    replacements = {
        vertex_id: Vertex(vertex_id, *point)
        for vertex_id, point in zip(corners, new_points)
    }
    old_vertices = {vertex.id: vertex for vertex in graph.vertices}
    new_vertices = {
        key: replacements.get(key, vertex) for key, vertex in old_vertices.items()
    }
    result = replace(
        window,
        corner_vertex_ids=corners,
        start_ratio=0.0,
        end_ratio=1.0,
    )
    moved_ids = {key for key in corners if old_vertices[key] != new_vertices[key]}
    if moved_ids:
        _validate_moved_edges(graph.edges, old_vertices, new_vertices, moved_ids)
        # All validation and allocation finish before the one live assignment.
        # Immutable vertex records and unchanged edge records need no deep copy
        # on each pointer move.
        staged_vertices = [new_vertices[vertex.id] for vertex in graph.vertices]
        graph.vertices = staged_vertices
        graph.mark_changed()
    return result


# ### Conservative four-corner recognition ###
def _resolve_corner_ids(level, window, wall_graph=None):
    if window.sloped_ceiling_id is not None:
        return None
    match = _WALL_ID_PATTERN.fullmatch(window.wall_surface_id)
    if match is None or int(match[1]) != level.index:
        return None
    first_id, second_id = int(match[2]), int(match[3])
    vertices, neighbors = _wall_graph(level) if wall_graph is None else wall_graph
    if (
        first_id not in vertices
        or second_id not in vertices
        or second_id not in neighbors[first_id]
    ):
        return None
    explicit = window.corner_vertex_ids
    if explicit is not None:
        return (
            explicit
            if _valid_corners(explicit, first_id, second_id, vertices, neighbors)
            else None
        )
    if abs(window.start_ratio) > _EPSILON or abs(window.end_ratio - 1.0) > _EPSILON:
        return None
    first, second = _point(vertices[first_id]), _point(vertices[second_id])
    if math.dist(first, second) <= _EPSILON:
        return None
    axis = _unit(_subtract(second, first))
    maximum_depth = _MAXIMUM_INFERRED_DEPTH_METERS / level.meters_per_pixel
    candidates = []
    for back_first_id in neighbors[first_id]:
        if back_first_id == second_id:
            continue
        depth = _subtract(_point(vertices[back_first_id]), first)
        if (
            not _ALIGNMENT_TOLERANCE_PIXELS < math.hypot(*depth) <= maximum_depth
            or abs(_dot(depth, axis)) > _ALIGNMENT_TOLERANCE_PIXELS
        ):
            continue
        for back_second_id in neighbors[second_id]:
            ids = first_id, second_id, back_first_id, back_second_id
            if _valid_corners(ids, first_id, second_id, vertices, neighbors):
                candidates.append(ids)
    return min(
        candidates,
        key=lambda ids: (
            math.dist(_point(vertices[ids[0]]), _point(vertices[ids[2]])),
            ids,
        ),
        default=None,
    )


def _valid_corners(corners, first_id, second_id, vertices, neighbors):
    if (
        len(set(corners)) != 4
        or corners[:2] != (first_id, second_id)
        or any(key not in vertices for key in corners)
        or any(
            second not in neighbors[first]
            for first, second in (
                (corners[0], corners[1]),
                (corners[2], corners[3]),
                (corners[0], corners[2]),
                (corners[1], corners[3]),
            )
        )
    ):
        return False
    first, second, back_first, back_second = (_point(vertices[key]) for key in corners)
    width, depth = _subtract(second, first), _subtract(back_first, first)
    if math.hypot(*width) <= _EPSILON or math.hypot(*depth) <= _EPSILON:
        return False
    axis = _unit(width)
    if (
        abs(_dot(depth, axis)) > _ALIGNMENT_TOLERANCE_PIXELS
        or math.dist(back_second, _add(second, depth)) > _ALIGNMENT_TOLERANCE_PIXELS
    ):
        return False
    return all(
        _outward_distance(key, axis, sign, vertices, neighbors, set(corners))
        is not None
        for key, sign in zip(corners, (-1.0, 1.0, -1.0, 1.0))
    )


def _wall_graph(level):
    excluded = {room.center_vertex_id for room in level.rooms}
    vertices = {
        vertex.id: vertex
        for vertex in level.vertex_data.vertices
        if vertex.id not in excluded
    }
    neighbors = {key: set() for key in vertices}
    for edge in level.vertex_data.edges:
        first, second = edge.start_vertex_id, edge.end_vertex_id
        if first in vertices and second in vertices and first != second:
            neighbors[first].add(second)
            neighbors[second].add(first)
    return vertices, neighbors


def _outward_distance(vertex_id, axis, sign, vertices, neighbors, own_corners):
    point = _point(vertices[vertex_id])
    distances = []
    for other_id in neighbors[vertex_id] - own_corners:
        delta = _subtract(_point(vertices[other_id]), point)
        distance = sign * _dot(delta, axis)
        if (
            distance > _EPSILON
            and abs(_cross(delta, axis)) <= _ALIGNMENT_TOLERANCE_PIXELS
        ):
            distances.append(distance)
    return min(distances, default=None)


def _other_opening_corner_ids(level, window):
    protected = {
        key for doorway in level.doorways for key in doorway.corner_vertex_ids or ()
    }
    for other in level.windows:
        if other.window_id == window.window_id:
            continue
        if other.corner_vertex_ids is not None:
            protected.update(other.corner_vertex_ids)
        elif (
            other.sloped_ceiling_id is None
            and abs(other.start_ratio) <= _EPSILON
            and abs(other.end_ratio - 1.0) <= _EPSILON
        ):
            protected.update(_resolve_corner_ids(level, other) or ())
    return protected


# ### Atomic movement safeguards ###
def _validate_moved_edges(edges, old_vertices, new_vertices, moved_ids):
    for key in moved_ids:
        point = _point(new_vertices[key])
        if any(
            other_id != key
            and abs(point[0] - vertex.x) <= _EPSILON
            and abs(point[1] - vertex.y) <= _EPSILON
            and math.hypot(point[0] - vertex.x, point[1] - vertex.y) <= _EPSILON
            for other_id, vertex in new_vertices.items()
        ):
            raise ValueError("The window frame would overlap another wall vertex.")
    records = []
    for edge in edges:
        first, second = edge.start_vertex_id, edge.end_vertex_id
        if first not in old_vertices or second not in old_vertices:
            continue
        old_points = _point(old_vertices[first]), _point(old_vertices[second])
        new_points = _point(new_vertices[first]), _point(new_vertices[second])
        changed = bool({first, second} & moved_ids)
        if changed and (
            math.dist(*new_points) <= _EPSILON
            or _dot(_subtract(*old_points), _subtract(*new_points)) <= 0.0
        ):
            raise ValueError("The window frame would collapse or reverse a wall edge.")
        records.append(
            ((first, second), old_points, new_points, _bounds(new_points), changed)
        )
    lines = {}

    def line(index, *, old=False):
        cache_key = index, old
        if cache_key not in lines:
            lines[cache_key] = LineString(records[index][1 if old else 2])
        return lines[cache_key]

    # Only edges incident to the four moved corners can introduce a crossing.
    # Cheap bounds reject distant pairs before allocating any Shapely objects.
    for index, (key, _old_points, _new_points, bounds, changed) in enumerate(records):
        if not changed:
            continue
        for other_index, (other_key, _, _, other_bounds, other_changed) in enumerate(
            records
        ):
            if (
                index == other_index
                or other_changed
                and other_index < index
                or key[0] in other_key
                or key[1] in other_key
                or not _bounds_overlap(bounds, other_bounds)
            ):
                continue
            if (
                line(index).distance(line(other_index)) <= _EPSILON
                and line(index, old=True).distance(line(other_index, old=True))
                > _EPSILON
            ):
                raise ValueError("The window frame would cross another wall edge.")


# ### Coordinate helpers ###
def _bounds(points):
    first, second = points
    return (
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[0], second[0]),
        max(first[1], second[1]),
    )


def _bounds_overlap(first, second):
    return (
        first[0] <= second[2] + _EPSILON
        and second[0] <= first[2] + _EPSILON
        and first[1] <= second[3] + _EPSILON
        and second[1] <= first[3] + _EPSILON
    )


def _finite_point(value):
    try:
        point = tuple(float(component) for component in value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "Window endpoints must contain finite image coordinates."
        ) from error
    if len(point) != 2 or not all(math.isfinite(component) for component in point):
        raise ValueError("Window endpoints must contain finite image coordinates.")
    return point


def _point(vertex):
    return vertex.x, vertex.y


def _subtract(first, second):
    return first[0] - second[0], first[1] - second[1]


def _add(first, second):
    return first[0] + second[0], first[1] + second[1]


def _add_scaled(point, axis, amount):
    return point[0] + axis[0] * amount, point[1] + axis[1] * amount


def _dot(first, second):
    return first[0] * second[0] + first[1] * second[1]


def _cross(first, second):
    return first[0] * second[1] - first[1] * second[0]


def _unit(vector):
    length = math.hypot(*vector)
    return vector[0] / length, vector[1] / length
