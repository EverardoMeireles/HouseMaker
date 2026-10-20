"""Fit doorways and window frames to four real wall corners atomically."""

# ### Imports ###
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, replace
from itertools import combinations, pairwise

from .models import (
    MAX_DOORWAY_WIDTH_METERS,
    MIN_DOORWAY_WIDTH_METERS,
    PIXEL_TO_METER,
    DoorwayData,
    Edge,
    Vertex,
    VertexData,
)

# ### Constants ###
_EPSILON = 1e-7
_ALIGNMENT_TOLERANCE_PIXELS = 0.5
_PARALLEL_ANGLE_TOLERANCE = math.sin(math.radians(1.0))
_DEFAULT_MAXIMUM_DEPTH_METERS = 1.5


# ### Fit models ###
@dataclass(frozen=True)
class DoorwayFit:
    """One pure placement preview, committed separately by apply_doorway_fit.

    Corners are front-left, front-right, back-left, back-right. Front means
    the negative local depth side; jamb edges join (0, 2) and (1, 3).
    """

    doorway: DoorwayData
    corner_positions: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]
    corner_vertex_ids: tuple[int, int, int, int] | None
    source_edge_keys: tuple[tuple[int, int], ...] | None
    source_vertices: tuple[Vertex, ...]
    source_graph_signature: tuple[tuple[Vertex, ...], tuple[Edge, ...]] | None = None
    excluded_vertex_ids: frozenset[int] = frozenset()


@dataclass(frozen=True)
class _WallRun:
    axis: tuple[float, float]
    origin: tuple[float, float]
    minimum: float
    maximum: float
    edge_keys: tuple[tuple[int, int], ...]


# ### Public fitting helpers ###
def doorway_corner_positions(
    doorway: DoorwayData,
    *,
    meters_per_pixel: float = PIXEL_TO_METER,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]:
    """Return ordered corners using the canvas width/depth convention."""

    _validate_scale(meters_per_pixel)
    angle = math.radians(doorway.rotation_degrees)
    depth_axis = math.cos(angle), math.sin(angle)
    width_axis = -depth_axis[1], depth_axis[0]
    half_width = doorway.width_meters / meters_per_pixel * 0.5
    half_depth = doorway.depth_meters / meters_per_pixel * 0.5
    return tuple(
        (
            doorway.center_x
            + width_sign * half_width * width_axis[0]
            + depth_sign * half_depth * depth_axis[0],
            doorway.center_y
            + width_sign * half_width * width_axis[1]
            + depth_sign * half_depth * depth_axis[1],
        )
        for depth_sign, width_sign in ((-1, -1), (-1, 1), (1, -1), (1, 1))
    )


def find_doorway_fit(
    vertex_data: VertexData,
    raw_center: tuple[float, float],
    doorway: DoorwayData,
    center_tolerance: float,
    *,
    room_center_vertex_ids: Iterable[int] = (),
    meters_per_pixel: float = PIXEL_TO_METER,
    maximum_depth_meters: float = _DEFAULT_MAXIMUM_DEPTH_METERS,
    allow_solid_wall_fit: bool = True,
    allow_unconnected_jambs: bool = False,
) -> DoorwayFit | None:
    """Fit a gap's four corners, or plan a cut through two parallel faces.

    Preview never mutates the graph or input doorway. Single wall lines,
    nonparallel faces and unrelated distant room boundaries are not valid.
    Disable solid-wall fitting to inspect existing four-corner gaps only.
    Window frames may infer missing jambs between four supported endpoints;
    regular doorways still require the existing connected jamb pairs.
    """

    _validate_scale(meters_per_pixel)
    if (
        not all(math.isfinite(value) for value in raw_center)
        or not math.isfinite(center_tolerance)
        or center_tolerance < 0.0
        or not math.isfinite(maximum_depth_meters)
        or maximum_depth_meters <= 0.0
        or not MIN_DOORWAY_WIDTH_METERS
        <= doorway.width_meters
        <= MAX_DOORWAY_WIDTH_METERS
    ):
        return None
    excluded = set(room_center_vertex_ids)
    vertices = {
        vertex.id: vertex
        for vertex in vertex_data.vertices
        if vertex.id not in excluded
    }
    edges = {
        _edge_key(edge.start_vertex_id, edge.end_vertex_id): edge
        for edge in vertex_data.edges
        if edge.start_vertex_id in vertices
        and edge.end_vertex_id in vertices
        and edge.start_vertex_id != edge.end_vertex_id
        and math.dist(
            _point(vertices[edge.start_vertex_id]), _point(vertices[edge.end_vertex_id])
        )
        > _EPSILON
    }
    if len(edges) < 2:
        return None
    maximum_depth = maximum_depth_meters / meters_per_pixel
    fits = _existing_gap_fits(
        vertices,
        edges,
        raw_center,
        doorway,
        center_tolerance,
        meters_per_pixel,
        maximum_depth,
        allow_unconnected_jambs=allow_unconnected_jambs,
    )
    if fits:
        selected = min(
            fits, key=lambda fit: _fit_score(fit, raw_center, doorway, meters_per_pixel)
        )
        return replace(
            selected,
            source_graph_signature=_graph_signature(vertex_data),
            excluded_vertex_ids=frozenset(excluded),
        )
    if not allow_solid_wall_fit:
        return None
    fits = _solid_wall_fits(
        vertices,
        edges,
        raw_center,
        doorway,
        center_tolerance,
        meters_per_pixel,
        maximum_depth,
    )
    if not fits:
        return None
    selected = min(
        fits, key=lambda fit: _fit_score(fit, raw_center, doorway, meters_per_pixel)
    )
    return replace(
        selected,
        source_graph_signature=_graph_signature(vertex_data),
        excluded_vertex_ids=frozenset(excluded),
    )


def apply_doorway_fit(vertex_data: VertexData, fit: DoorwayFit) -> DoorwayData:
    """Atomically cut the opening and attach its four corner IDs.

    A stale preview fails without modifying the live graph. No longitudinal
    bridge edges are stored: geometry/floor builders derive them from the
    doorway's corner IDs instead.
    """

    if fit.source_graph_signature != _graph_signature(vertex_data):
        raise ValueError("The wall layout changed; place the doorway again.")
    for expected in fit.source_vertices:
        if vertex_data.get_vertex(expected.id) != expected:
            raise ValueError("The wall corners changed; place the doorway again.")
    for start, end in fit.source_edge_keys or ():
        if not vertex_data.has_edge(start, end):
            raise ValueError("The wall connections changed; place the doorway again.")
    staged = vertex_data.clone()
    if fit.corner_vertex_ids is not None:
        corner_ids = fit.corner_vertex_ids
    else:
        corner_ids = tuple(
            _reuse_or_add_vertex(staged, point, fit.excluded_vertex_ids)
            for point in fit.corner_positions
        )
    # Jamb corners can already exist on longer non-noded face edges too.
    # Cut those face spans rather than only removing direct corner links.
    _cut_wall_faces(staged, fit, corner_ids)
    if len(set(corner_ids)) != 4:
        raise ValueError("A doorway requires four distinct wall vertices.")
    left_ids, right_ids = (
        {corner_ids[0], corner_ids[2]},
        {corner_ids[1], corner_ids[3]},
    )
    staged.edges = [
        edge
        for edge in staged.edges
        if not (
            edge.start_vertex_id in left_ids
            and edge.end_vertex_id in right_ids
            or edge.end_vertex_id in left_ids
            and edge.start_vertex_id in right_ids
        )
    ]
    staged.add_edge(corner_ids[0], corner_ids[2])
    staged.add_edge(corner_ids[1], corner_ids[3])
    result = replace(fit.doorway, corner_vertex_ids=corner_ids)
    vertex_data.copy_from(staged)
    return result


def apply_window_fit(vertex_data: VertexData, fit: DoorwayFit) -> DoorwayData:
    """Attach four window corners while retaining both physical wall faces.

    The window opening is cut vertically by the surface builder, not by
    removing the wall footprint. Longitudinal edges therefore remain real
    connections for the wall portions below and above the window.
    """

    staged = vertex_data.clone()
    result = apply_doorway_fit(staged, fit)
    front_left, front_right, back_left, back_right = result.corner_vertex_ids
    staged.add_edge(front_left, front_right)
    staged.add_edge(back_left, back_right)
    vertex_data.copy_from(staged)
    return result


def move_doorway_corner_vertices(
    vertex_data: VertexData,
    doorway: DoorwayData,
    *,
    meters_per_pixel: float = PIXEL_TO_METER,
) -> bool:
    """Move an attached opening and its four wall corners as one edit."""

    corner_ids = doorway.corner_vertex_ids
    if corner_ids is None or len(set(corner_ids)) != 4:
        raise ValueError("A doorway requires four distinct wall corners.")
    positions = doorway_corner_positions(doorway, meters_per_pixel=meters_per_pixel)
    if (
        doorway.width_meters <= 0.0
        or doorway.depth_meters <= 0.0
        or not all(math.isfinite(value) for point in positions for value in point)
        or any(vertex_data.get_vertex(vertex_id) is None for vertex_id in corner_ids)
    ):
        raise ValueError("The doorway's four wall corners are no longer valid.")
    replacements = {
        vertex_id: Vertex(vertex_id, *point)
        for vertex_id, point in zip(corner_ids, positions)
    }
    vertices = {
        vertex.id: replacements.get(vertex.id, vertex)
        for vertex in vertex_data.vertices
    }
    for edge in vertex_data.edges:
        if not {edge.start_vertex_id, edge.end_vertex_id} & set(corner_ids):
            continue
        first, second = (
            vertices.get(edge.start_vertex_id),
            vertices.get(edge.end_vertex_id),
        )
        if (
            first is not None
            and second is not None
            and math.dist(_point(first), _point(second)) <= _EPSILON
        ):
            raise ValueError("The doorway would collapse an adjacent wall edge.")
    changed = any(
        vertex_data.get_vertex(vertex_id) != replacement
        for vertex_id, replacement in replacements.items()
    )
    if changed:
        vertex_data.vertices = [
            replacements.get(vertex.id, vertex) for vertex in vertex_data.vertices
        ]
        vertex_data.mark_changed()
    return changed


# ### Existing four-corner gaps ###
def _existing_gap_fits(
    vertices,
    edges,
    raw_center,
    doorway,
    tolerance,
    scale,
    maximum_depth,
    *,
    allow_unconnected_jambs=False,
):
    neighbors: dict[int, list[int]] = {vertex_id: [] for vertex_id in vertices}
    for first, second in edges:
        neighbors[first].append(second)
        neighbors[second].append(first)
    jambs = {
        key
        for key in edges
        if math.dist(_point(vertices[key[0]]), _point(vertices[key[1]]))
        <= maximum_depth
    }
    if allow_unconnected_jambs:
        jambs.update(_supported_missing_jambs(vertices, neighbors, maximum_depth))
    fits: list[DoorwayFit] = []
    for first_key, second_key in combinations(sorted(jambs), 2):
        if set(first_key) & set(second_key):
            continue
        first_points = tuple(_point(vertices[vertex_id]) for vertex_id in first_key)
        second_points = tuple(_point(vertices[vertex_id]) for vertex_id in second_key)
        depth_axis = _unit(_subtract(first_points[1], first_points[0]))
        other_axis = _unit(_subtract(second_points[1], second_points[0]))
        if abs(_cross(depth_axis, other_axis)) > _PARALLEL_ANGLE_TOLERANCE:
            continue
        width_axis = -depth_axis[1], depth_axis[0]
        first_mid, second_mid = _midpoint(*first_points), _midpoint(*second_points)
        if (
            abs(_dot(_subtract(second_mid, first_mid), depth_axis))
            > _ALIGNMENT_TOLERANCE_PIXELS
        ):
            continue
        if (
            abs(math.dist(*first_points) - math.dist(*second_points))
            > _ALIGNMENT_TOLERANCE_PIXELS
        ):
            continue
        if _dot(_subtract(second_mid, first_mid), width_axis) < 0.0:
            first_key, second_key = second_key, first_key
        left_front, left_back = sorted(
            first_key,
            key=lambda vertex_id: _dot(_point(vertices[vertex_id]), depth_axis),
        )
        right_front, right_back = sorted(
            second_key,
            key=lambda vertex_id: _dot(_point(vertices[vertex_id]), depth_axis),
        )
        corner_ids = left_front, right_front, left_back, right_back
        corners = tuple(_point(vertices[vertex_id]) for vertex_id in corner_ids)
        width_pixels = _dot(_subtract(corners[1], corners[0]), width_axis)
        if (
            not MIN_DOORWAY_WIDTH_METERS
            <= width_pixels * scale
            <= MAX_DOORWAY_WIDTH_METERS
        ):
            continue
        if not all(
            _has_outward_branch(
                vertex_id, width_axis, -1 if index % 2 == 0 else 1, vertices, neighbors
            )
            for index, vertex_id in enumerate(corner_ids)
        ):
            continue
        fitted = _make_fit(
            doorway,
            corners,
            width_axis,
            scale,
            corner_ids,
            tuple(key for key in (first_key, second_key) if key in edges),
            vertices,
        )
        if _rectangle_distance(raw_center, fitted, scale) > tolerance:
            continue
        if not _opening_obstructed(fitted, vertices, edges):
            fits.append(fitted)
    return fits


def _supported_missing_jambs(vertices, neighbors, maximum_depth):
    """Infer caps only where both corners have perpendicular wall support.

    Four floating points do not define a wall opening. Each inferred cap
    needs two real outward wall-face branches pointing in the same direction;
    the final rectangle check verifies the opposite cap points the other way.
    """

    supported_ids = sorted(
        (vertex_id for vertex_id in vertices if neighbors[vertex_id]),
        key=lambda vertex_id: (vertices[vertex_id].x, vertices[vertex_id].y, vertex_id),
    )
    jambs = set()
    for index, first_id in enumerate(supported_ids):
        first = vertices[first_id]
        for second_id in supported_ids[index + 1 :]:
            second = vertices[second_id]
            # The canvas recomputes hover previews frequently. Restrict pair
            # searches to the wall-depth neighborhood before checking branches.
            if second.x - first.x > maximum_depth:
                break
            delta = second.x - first.x, second.y - first.y
            length = math.hypot(*delta)
            if not _ALIGNMENT_TOLERANCE_PIXELS < length <= maximum_depth:
                continue
            depth_axis = delta[0] / length, delta[1] / length
            width_axis = -depth_axis[1], depth_axis[0]
            if any(
                all(
                    _has_outward_branch(
                        vertex_id, width_axis, sign, vertices, neighbors
                    )
                    for vertex_id in (first_id, second_id)
                )
                for sign in (-1, 1)
            ):
                jambs.add(_edge_key(first_id, second_id))
    return jambs


def _has_outward_branch(vertex_id, axis, sign, vertices, neighbors):
    point = _point(vertices[vertex_id])
    for neighbor_id in neighbors[vertex_id]:
        delta = _subtract(_point(vertices[neighbor_id]), point)
        length = math.hypot(*delta)
        if sign * _dot(delta, axis) > _ALIGNMENT_TOLERANCE_PIXELS and abs(
            _cross(delta, axis)
        ) <= min(_ALIGNMENT_TOLERANCE_PIXELS, length * _PARALLEL_ANGLE_TOLERANCE):
            return True
    return False


# ### Solid wall faces and connected collinear runs ###
def _solid_wall_fits(
    vertices, edges, raw_center, doorway, tolerance, scale, maximum_depth
):
    runs = _collinear_runs(vertices, edges)
    requested_width = doorway.width_meters / scale
    fits: list[DoorwayFit] = []
    for first, second in combinations(runs, 2):
        if (
            set(first.edge_keys) & set(second.edge_keys)
            or abs(_cross(first.axis, second.axis)) > _PARALLEL_ANGLE_TOLERANCE
        ):
            continue
        width_axis = first.axis
        depth_axis = width_axis[1], -width_axis[0]
        first_depth, second_depth = (
            _dot(first.origin, depth_axis),
            _dot(second.origin, depth_axis),
        )
        depth = abs(second_depth - first_depth)
        if depth <= _ALIGNMENT_TOLERANCE_PIXELS or depth > maximum_depth:
            continue
        # Both ends of each run must belong to genuinely parallel face lines.
        second_ends = tuple(
            _add_scaled(second.origin, second.axis, value)
            for value in (second.minimum, second.maximum)
        )
        if (
            abs(_dot(_subtract(second_ends[1], second_ends[0]), depth_axis))
            > _ALIGNMENT_TOLERANCE_PIXELS
        ):
            continue
        first_range = tuple(
            _dot(_add_scaled(first.origin, first.axis, value), width_axis)
            for value in (first.minimum, first.maximum)
        )
        second_range = sorted(_dot(point, width_axis) for point in second_ends)
        minimum, maximum = (
            max(first_range[0], second_range[0]),
            min(first_range[1], second_range[1]),
        )
        # A cut needs a surviving outward branch on both jambs, not merely
        # four points at the very ends of two isolated parallel segments.
        minimum += 2.0 * _ALIGNMENT_TOLERANCE_PIXELS
        maximum -= 2.0 * _ALIGNMENT_TOLERANCE_PIXELS
        available_width = maximum - minimum
        if available_width * scale < MIN_DOORWAY_WIDTH_METERS - _EPSILON:
            continue
        # Opposite jambs can also look like short parallel faces. Do not turn
        # an existing gap sideways just to squeeze a much wider template in.
        # Explicitly requested narrow/deep openings retain their old behavior.
        if (
            available_width < requested_width - _EPSILON
            and available_width < depth - _EPSILON
        ):
            continue
        # Templates describe a preferred width, not a reason to suppress the
        # wall hover entirely. Keep four outward branches and contract a wide
        # template to the usable span shared by both physical wall faces.
        half_width = min(requested_width, available_width) * 0.5
        width_center = min(
            max(_dot(raw_center, width_axis), minimum + half_width),
            maximum - half_width,
        )
        front_depth, back_depth = sorted((first_depth, second_depth))
        corners = tuple(
            (
                width * width_axis[0] + depth_value * depth_axis[0],
                width * width_axis[1] + depth_value * depth_axis[1],
            )
            for depth_value in (front_depth, back_depth)
            for width in (width_center - half_width, width_center + half_width)
        )
        fitted = _make_fit(
            doorway,
            corners,
            width_axis,
            scale,
            None,
            first.edge_keys + second.edge_keys,
            vertices,
        )
        if _rectangle_distance(raw_center, fitted, scale) > tolerance:
            continue
        if not _opening_obstructed(fitted, vertices, edges):
            fits.append(fitted)
    return fits


def _collinear_runs(vertices, edges):
    remaining = set(edges)
    neighbors: dict[int, set[tuple[int, int]]] = {}
    for key in edges:
        for vertex_id in key:
            neighbors.setdefault(vertex_id, set()).add(key)
    runs: list[_WallRun] = []
    while remaining:
        seed = min(remaining)
        origin = _point(vertices[seed[0]])
        axis = _unit(_subtract(_point(vertices[seed[1]]), origin))
        if axis[0] < -_EPSILON or abs(axis[0]) <= _EPSILON and axis[1] < 0.0:
            axis = -axis[0], -axis[1]
        members, pending = set(), [seed]
        while pending:
            key = pending.pop()
            if key not in remaining:
                continue
            points = tuple(_point(vertices[vertex_id]) for vertex_id in key)
            if any(
                abs(_cross(_subtract(point, origin), axis))
                > _ALIGNMENT_TOLERANCE_PIXELS
                for point in points
            ):
                continue
            if (
                abs(_cross(_unit(_subtract(points[1], points[0])), axis))
                > _PARALLEL_ANGLE_TOLERANCE
            ):
                continue
            remaining.remove(key)
            members.add(key)
            for vertex_id in key:
                pending.extend(neighbors[vertex_id] & remaining)
        projections = [
            _dot(_subtract(_point(vertices[vertex_id]), origin), axis)
            for key in members
            for vertex_id in key
        ]
        runs.append(
            _WallRun(
                axis, origin, min(projections), max(projections), tuple(sorted(members))
            )
        )
    return runs


# ### Atomic edge surgery ###
def _cut_wall_faces(graph, fit, corner_ids):
    doorway = fit.doorway
    angle = math.radians(doorway.rotation_degrees)
    width_axis = -math.sin(angle), math.cos(angle)
    depth_axis = math.cos(angle), math.sin(angle)
    center = doorway.center_x, doorway.center_y
    corner_coordinates = [
        _dot(_subtract(point, center), width_axis) for point in fit.corner_positions
    ]
    minimum, maximum = min(corner_coordinates), max(corner_coordinates)
    candidate_orphans: set[int] = set()
    for edge in tuple(graph.edges):
        start_id, end_id = edge.start_vertex_id, edge.end_vertex_id
        if fit.excluded_vertex_ids.intersection((start_id, end_id)):
            continue
        first, second = graph.get_vertex(start_id), graph.get_vertex(end_id)
        first_point, second_point = _point(first), _point(second)
        if math.dist(first_point, second_point) <= _EPSILON:
            continue
        # Only the two boundary face lines belong to this cut. Keep jambs
        # and unrelated perpendicular or interior wall branches intact.
        face_depth = _dot(_subtract(first_point, center), depth_axis)
        end_depth = _dot(_subtract(second_point, center), depth_axis)
        boundary_depths = (
            _dot(_subtract(fit.corner_positions[index], center), depth_axis)
            for index in (0, 2)
        )
        if abs(face_depth - end_depth) > _ALIGNMENT_TOLERANCE_PIXELS or not any(
            abs(face_depth - boundary) <= _ALIGNMENT_TOLERANCE_PIXELS
            and abs(end_depth - boundary) <= _ALIGNMENT_TOLERANCE_PIXELS
            for boundary in boundary_depths
        ):
            continue
        first_width = _dot(_subtract(first_point, center), width_axis)
        second_width = _dot(_subtract(second_point, center), width_axis)
        if (
            max(first_width, second_width) <= minimum + _EPSILON
            or min(first_width, second_width) >= maximum - _EPSILON
        ):
            continue
        graph.remove_edge(start_id, end_id)
        candidate_orphans.update((start_id, end_id))
        cuts = [(0.0, start_id), (1.0, end_id)]
        delta = _subtract(second_point, first_point)
        denominator = _dot(delta, delta)
        for corner_id, point in zip(corner_ids, fit.corner_positions):
            ratio = _dot(_subtract(point, first_point), delta) / denominator
            if (
                -_EPSILON <= ratio <= 1.0 + _EPSILON
                and abs(_cross(_subtract(point, first_point), _unit(delta)))
                <= _ALIGNMENT_TOLERANCE_PIXELS
            ):
                cuts.append((min(max(ratio, 0.0), 1.0), corner_id))
        cuts = sorted(set(cuts))
        for (_, left_id), (_, right_id) in pairwise(cuts):
            if left_id == right_id:
                continue
            midpoint = _midpoint(
                _point(graph.get_vertex(left_id)), _point(graph.get_vertex(right_id))
            )
            local_width = _dot(_subtract(midpoint, center), width_axis)
            local_depth = abs(_dot(_subtract(midpoint, center), depth_axis))
            corner_depth = max(
                abs(_dot(_subtract(point, center), depth_axis))
                for point in fit.corner_positions
            )
            if not (
                minimum + _EPSILON < local_width < maximum - _EPSILON
                and local_depth <= corner_depth + _ALIGNMENT_TOLERANCE_PIXELS
            ):
                graph.add_edge(left_id, right_id)
    remaining_ids = {
        vertex_id
        for edge in graph.edges
        for vertex_id in (edge.start_vertex_id, edge.end_vertex_id)
    }
    orphan_ids = candidate_orphans - remaining_ids - set(corner_ids)
    if orphan_ids:
        graph.delete_vertices(orphan_ids)


def _reuse_or_add_vertex(graph, point, excluded_ids):
    for vertex in graph.vertices:
        if (
            vertex.id not in excluded_ids
            and math.dist(_point(vertex), point) <= _EPSILON
        ):
            return vertex.id
    return graph.add_vertex(*point).id


# ### Rectangle validity and ranking ###
def _make_fit(doorway, corners, width_axis, scale, corner_ids, source_edges, vertices):
    center = tuple(sum(point[axis] for point in corners) * 0.25 for axis in (0, 1))
    depth_axis = width_axis[1], -width_axis[0]
    fitted_doorway = replace(
        doorway,
        center_x=center[0],
        center_y=center[1],
        width_meters=math.dist(corners[0], corners[1]) * scale,
        depth_meters=math.dist(corners[0], corners[2]) * scale,
        rotation_degrees=math.degrees(math.atan2(depth_axis[1], depth_axis[0])) % 360.0,
        corner_vertex_ids=corner_ids,
    )
    source_ids = {vertex_id for edge in source_edges for vertex_id in edge}
    return DoorwayFit(
        fitted_doorway,
        corners,
        corner_ids,
        source_edges,
        tuple(vertices[vertex_id] for vertex_id in sorted(source_ids)),
    )


def _opening_obstructed(fit, vertices, edges):
    # Existing links across the four corners are deliberately removable.
    source_keys = set(fit.source_edge_keys or ())
    corner_ids = set(fit.corner_vertex_ids or ())
    width_axis = _unit(_subtract(fit.corner_positions[1], fit.corner_positions[0]))
    depth_axis = _unit(_subtract(fit.corner_positions[2], fit.corner_positions[0]))
    width, depth = (
        math.dist(fit.corner_positions[0], fit.corner_positions[1]),
        math.dist(fit.corner_positions[0], fit.corner_positions[2]),
    )
    origin = fit.corner_positions[0]
    for key in edges:
        if key in source_keys or set(key) <= corner_ids:
            continue
        points = [
            (
                _dot(_subtract(_point(vertices[vertex_id]), origin), width_axis),
                _dot(_subtract(_point(vertices[vertex_id]), origin), depth_axis),
            )
            for vertex_id in key
        ]
        # Intersect the segment with the strict rectangle interior (Liang-Barsky).
        lower, upper = 0.0, 1.0
        for axis, limit in ((0, width), (1, depth)):
            delta = points[1][axis] - points[0][axis]
            if abs(delta) <= _EPSILON:
                if (
                    not _ALIGNMENT_TOLERANCE_PIXELS
                    < points[0][axis]
                    < limit - _ALIGNMENT_TOLERANCE_PIXELS
                ):
                    lower, upper = 1.0, 0.0
                    break
            else:
                bounds = sorted(
                    (
                        (_ALIGNMENT_TOLERANCE_PIXELS - points[0][axis]) / delta,
                        (limit - _ALIGNMENT_TOLERANCE_PIXELS - points[0][axis]) / delta,
                    )
                )
                lower, upper = max(lower, bounds[0]), min(upper, bounds[1])
        if lower < upper - _EPSILON:
            return True
    return False


def _rectangle_distance(point, fit, scale):
    doorway = fit.doorway
    angle = math.radians(doorway.rotation_degrees)
    depth_axis = math.cos(angle), math.sin(angle)
    width_axis = -depth_axis[1], depth_axis[0]
    delta = point[0] - doorway.center_x, point[1] - doorway.center_y
    return math.hypot(
        max(0.0, abs(_dot(delta, width_axis)) - doorway.width_meters / scale * 0.5),
        max(0.0, abs(_dot(delta, depth_axis)) - doorway.depth_meters / scale * 0.5),
    )


def _fit_score(fit, raw_center, doorway, scale):
    return (
        _rectangle_distance(raw_center, fit, scale),
        math.dist(raw_center, (fit.doorway.center_x, fit.doorway.center_y)),
        abs(fit.doorway.width_meters - doorway.width_meters) / scale,
        fit.doorway.depth_meters,
        tuple(fit.source_edge_keys or ()),
    )


# ### Basic vector helpers ###
def _graph_signature(graph: VertexData) -> tuple[tuple[Vertex, ...], tuple[Edge, ...]]:
    """Catch new obstacles and removed supports without rerunning semantic fits."""

    return tuple(graph.vertices), tuple(graph.edges)


def _point(vertex):
    return vertex.x, vertex.y


def _subtract(first, second):
    return first[0] - second[0], first[1] - second[1]


def _dot(first, second):
    return first[0] * second[0] + first[1] * second[1]


def _cross(first, second):
    return first[0] * second[1] - first[1] * second[0]


def _unit(delta):
    length = math.hypot(*delta)
    return delta[0] / length, delta[1] / length


def _midpoint(first, second):
    return (first[0] + second[0]) * 0.5, (first[1] + second[1]) * 0.5


def _add_scaled(origin, direction, amount):
    return origin[0] + direction[0] * amount, origin[1] + direction[1] * amount


def _edge_key(first, second):
    return min(first, second), max(first, second)


def _validate_scale(scale):
    if isinstance(scale, bool) or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Plan meters per pixel must be a finite positive number.")
