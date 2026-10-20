"""Fit window frames across both real wall faces and retain sill supports."""

# ### Imports ###
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from shapely import Polygon

from housemaker.doorway_topology import (
    DoorwayFit,
    apply_window_fit,
    doorway_corner_positions,
    find_doorway_fit,
)
from housemaker.glb import _build_window_wall_target_lookup
from housemaker.models import DoorwayData, Edge, LevelData, VertexData
from housemaker.surface_geometry import (
    MIN_WINDOW_SIZE_METERS,
    WallWindowPlacement,
    build_wall_surface_id,
)


# ### Transaction models ###
@dataclass(frozen=True)
class WindowFrameBridgeTransaction:
    """A reversible wall cut, including new corners and both physical faces."""

    level: LevelData
    added_edges: tuple[Edge, ...]
    previous_vertex_data: VertexData
    primary_edge: Edge


# ### Window bridge previews ###
def build_window_frame_bridge_placement(
    level: LevelData,
    pointer: tuple[float, float],
    center_tolerance: float,
    *,
    preferred_width_meters: float,
    height_meters: float,
    wall_margin_meters: float,
) -> WallWindowPlacement | None:
    """Preview four existing corners or a cut through both parallel faces.

    The projected corner IDs are allocated on a clone only. Hovering cannot
    change the live graph, its revision, or its next-vertex allocator.
    """

    available_height = float(level.height_meters) - 2.0 * wall_margin_meters
    if available_height < MIN_WINDOW_SIZE_METERS:
        return None
    fit = _find_window_fit(level, pointer, center_tolerance, preferred_width_meters)
    if fit is None:
        return None
    half_height_ratio = (
        min(height_meters, available_height) / float(level.height_meters) * 0.5
    )
    return _placement_from_fit(
        level, fit, pointer, 0.5 - half_height_ratio, 0.5 + half_height_ratio
    )


def normalize_wall_window_placement(
    levels: Sequence[LevelData],
    placement: WallWindowPlacement,
) -> WallWindowPlacement:
    """Give a rectangle drawn in 3D the same four-corner fit as the canvas.

    Keep the user's drawn width and absolute sill/lintel heights. A lone
    legacy wall face keeps its old placement, but an occupied paired-wall
    target must not fall back to cutting only one face.
    """

    if (
        placement.opening_fit is not None
        or placement.bridge_corner_vertex_ids is not None
    ):
        return placement
    for level in levels:
        if not placement.wall_surface_id.startswith(f"level:{level.index}/"):
            continue
        target = _build_window_wall_target_lookup(level).get(placement.wall_surface_id)
        if target is None:
            continue
        center_ratio = (placement.start_ratio + placement.end_ratio) * 0.5
        pointer = tuple(
            start + center_ratio * (end - start)
            for start, end in zip(target.start_point, target.end_point)
        )
        width = (
            math.dist(target.start_point, target.end_point)
            * (placement.end_ratio - placement.start_ratio)
            * level.meters_per_pixel
            * float(level.scale)
        )
        fit = _find_window_fit(level, pointer, 0.5, width, reject_occupied=True)
        if fit is None:
            return placement
        height_ratio = target.height_meters / level.height_meters
        return _placement_from_fit(
            level,
            fit,
            pointer,
            placement.bottom_ratio * height_ratio,
            placement.top_ratio * height_ratio,
        )
    return placement


def _placement_from_fit(
    level: LevelData,
    fit: DoorwayFit,
    pointer: tuple[float, float],
    bottom_ratio: float,
    top_ratio: float,
) -> WallWindowPlacement:
    projected = level.vertex_data.clone()
    attached = apply_window_fit(projected, fit)
    corner_ids = _placement_corner_ids(fit, pointer, attached.corner_vertex_ids)
    vertex_ids = corner_ids[:2]
    return WallWindowPlacement(
        wall_surface_id=_surface_id(level, vertex_ids),
        start_ratio=0.0,
        end_ratio=1.0,
        bottom_ratio=bottom_ratio,
        top_ratio=top_ratio,
        bridge_vertex_ids=vertex_ids,
        bridge_corner_vertex_ids=corner_ids,
        opening_fit=fit,
    )


# ### Transaction helpers ###
def add_window_frame_bridge(
    levels: Sequence[LevelData],
    placement: WallWindowPlacement,
) -> WindowFrameBridgeTransaction | None:
    """Revalidate and atomically attach all four corners of a window frame.

    Windows retain the two longitudinal wall edges below the sill and above
    the lintel. The opening therefore cuts only its selected vertical range,
    unlike a doorway's floor-to-lintel cut.
    """

    corner_ids = placement.bridge_corner_vertex_ids
    if corner_ids is None:
        if placement.bridge_vertex_ids is not None:
            raise ValueError("A window gap requires four wall corner vertices.")
        return None
    level = next(
        (
            candidate
            for candidate in levels
            if placement.wall_surface_id == _surface_id(candidate, corner_ids[:2])
        ),
        None,
    )
    if level is None:
        raise ValueError("The window bridge has no owning level.")
    fit = placement.opening_fit
    if fit is None:
        fit = _legacy_placement_fit(level, placement)
    if (
        fit is not None
        and fit.corner_vertex_ids is not None
        and any(
            level.vertex_data.get_vertex(vertex_id) is None
            for vertex_id in fit.corner_vertex_ids
        )
    ):
        raise ValueError("The window bridge's four corners no longer exist.")
    if fit is None or _fit_is_occupied(level, fit):
        raise ValueError(
            "The window bridge requires four aligned, supported, unoccupied wall corners."
        )
    previous = level.vertex_data.clone()
    staged = previous.clone()
    try:
        attached = apply_window_fit(staged, fit)
    except ValueError as exc:
        raise ValueError(
            "The window bridge requires four aligned, supported, unoccupied wall corners; "
            "the wall layout changed. Place the window again."
        ) from exc
    fitted_ids = attached.corner_vertex_ids
    assert fitted_ids is not None
    if set(fitted_ids) != set(corner_ids) or not _corner_order_matches(
        fit, fitted_ids, corner_ids
    ):
        raise ValueError("The window bridge requires four aligned wall corners.")
    previous_edge_keys = {_edge_key(edge) for edge in previous.edges}
    added_edges = tuple(
        edge for edge in staged.edges if _edge_key(edge) not in previous_edge_keys
    )
    front_pair = set(corner_ids[:2])
    primary_edge = next(
        edge
        for edge in staged.edges
        if {edge.start_vertex_id, edge.end_vertex_id} == front_pair
    )
    level.vertex_data.copy_from(staged)
    return WindowFrameBridgeTransaction(level, added_edges, previous, primary_edge)


# ### Four-corner validation ###
def _placement_corner_ids(
    fit: DoorwayFit,
    pointer: tuple[float, float],
    corner_vertex_ids: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int]:
    """Use the hovered face and the endpoint order of its stable surface."""

    corner_ids = corner_vertex_ids or fit.corner_vertex_ids
    assert corner_ids is not None
    centers = tuple(
        tuple(
            sum(fit.corner_positions[index][axis] for index in pair) * 0.5
            for axis in (0, 1)
        )
        for pair in ((0, 1), (2, 3))
    )
    if math.dist(pointer, centers[1]) < math.dist(pointer, centers[0]):
        corner_ids = corner_ids[2], corner_ids[3], corner_ids[0], corner_ids[1]
    if corner_ids[0] > corner_ids[1]:
        corner_ids = corner_ids[1], corner_ids[0], corner_ids[3], corner_ids[2]
    return corner_ids


def _find_window_fit(
    level: LevelData,
    pointer: tuple[float, float],
    center_tolerance: float,
    preferred_width_meters: float,
    *,
    reject_occupied: bool = False,
) -> DoorwayFit | None:
    # Existing four-corner gaps take priority over a new cut beside them.
    for allow_solid in (False, True):
        fit = find_doorway_fit(
            level.vertex_data,
            pointer,
            DoorwayData(
                center_x=pointer[0],
                center_y=pointer[1],
                width_meters=preferred_width_meters,
                height_meters=1.0,
                doorway_id="window-fit-preview",
            ),
            center_tolerance,
            room_center_vertex_ids=(room.center_vertex_id for room in level.rooms),
            meters_per_pixel=level.meters_per_pixel * float(level.scale),
            allow_solid_wall_fit=allow_solid,
            allow_unconnected_jambs=True,
        )
        if fit is not None:
            if _fit_is_occupied(level, fit):
                if reject_occupied:
                    raise ValueError(
                        "The window's four-corner wall area is occupied, or its cut "
                        "would detach an existing window. Choose another wall area."
                    )
                return None
            return fit
    return None


def _legacy_placement_fit(
    level: LevelData,
    placement: WallWindowPlacement,
) -> DoorwayFit | None:
    """Validate older callers without allowing arbitrary four-point rectangles."""

    corner_ids = placement.bridge_corner_vertex_ids
    assert corner_ids is not None
    corners = tuple(level.vertex_data.get_vertex(vertex_id) for vertex_id in corner_ids)
    if any(vertex is None for vertex in corners):
        raise ValueError("The window bridge's four corners no longer exist.")
    center = tuple(
        sum(getattr(vertex, axis) for vertex in corners) * 0.25 for axis in ("x", "y")
    )
    width = math.hypot(corners[1].x - corners[0].x, corners[1].y - corners[0].y)
    fit = _find_window_fit(
        level, center, 1e-6, width * level.meters_per_pixel * float(level.scale)
    )
    if fit is None or fit.corner_vertex_ids is None:
        return None
    front_center = (
        (corners[0].x + corners[1].x) * 0.5,
        (corners[0].y + corners[1].y) * 0.5,
    )
    return fit if _placement_corner_ids(fit, front_center) == corner_ids else None


def _corner_order_matches(fit, fitted_ids, placed_ids):
    positions = dict(zip(fitted_ids, fit.corner_positions))
    front_center = tuple(
        (positions[placed_ids[0]][axis] + positions[placed_ids[1]][axis]) * 0.5
        for axis in (0, 1)
    )
    return _placement_corner_ids(fit, front_center, fitted_ids) == placed_ids


# ### Existing opening ownership ###
def _fit_is_occupied(level: LevelData, fit: DoorwayFit) -> bool:
    scale = level.meters_per_pixel * float(level.scale)
    polygon = _corner_polygon(fit.corner_positions)
    for doorway in level.doorways:
        if (
            polygon.intersection(
                _corner_polygon(
                    doorway_corner_positions(doorway, meters_per_pixel=scale)
                )
            ).area
            > 1e-7
        ):
            return True
    corners = fit.corner_positions
    origin = corners[0]
    width = math.dist(corners[0], corners[1])
    depth = math.dist(corners[0], corners[2])
    width_axis = tuple((corners[1][axis] - origin[axis]) / width for axis in (0, 1))
    depth_axis = tuple((corners[2][axis] - origin[axis]) / depth for axis in (0, 1))
    for window in level.windows:
        if window.sloped_ceiling_id is not None:
            continue
        # Cutting its old parent edge would orphan even a non-overlapping
        # window. Keep that attachment safe until split remapping is supported.
        endpoints = _window_edge_endpoints(level, window.wall_surface_id)
        if endpoints is None:
            continue
        if _window_parent_would_be_cut(fit, endpoints):
            return True
        start, end = endpoints
        points = tuple(
            (
                start.x + ratio * (end.x - start.x) - origin[0],
                start.y + ratio * (end.y - start.y) - origin[1],
            )
            for ratio in (window.start_ratio, window.end_ratio)
        )
        depths = tuple(
            sum(point[axis] * depth_axis[axis] for axis in (0, 1)) for point in points
        )
        if any(value < -0.5 or value > depth + 0.5 for value in depths):
            continue
        widths = sorted(
            sum(point[axis] * width_axis[axis] for axis in (0, 1)) for point in points
        )
        if min(width, widths[1]) - max(0.0, widths[0]) > 1e-7:
            return True
    return False


def _window_parent_would_be_cut(fit: DoorwayFit, endpoints) -> bool:
    """Do not orphan an existing window when its parent edge must be split."""

    corner_ids = fit.corner_vertex_ids
    parent_ids = {point.id for point in endpoints}
    if corner_ids is not None and parent_ids in (
        set(corner_ids[:2]),
        set(corner_ids[2:]),
    ):
        return False
    origin = fit.corner_positions[0]
    width = math.dist(origin, fit.corner_positions[1])
    depth = math.dist(origin, fit.corner_positions[2])
    width_axis = tuple(
        (fit.corner_positions[1][axis] - origin[axis]) / width for axis in (0, 1)
    )
    depth_axis = tuple(
        (fit.corner_positions[2][axis] - origin[axis]) / depth for axis in (0, 1)
    )
    offsets = tuple((point.x - origin[0], point.y - origin[1]) for point in endpoints)
    depths = tuple(
        sum(point[axis] * depth_axis[axis] for axis in (0, 1)) for point in offsets
    )
    if not any(
        all(abs(value - boundary) <= 0.5 for value in depths)
        for boundary in (0.0, depth)
    ):
        return False
    widths = sorted(
        sum(point[axis] * width_axis[axis] for axis in (0, 1)) for point in offsets
    )
    return min(width, widths[1]) - max(0.0, widths[0]) > 1e-7


def _window_edge_endpoints(level: LevelData, surface_id: str):
    prefix = f"level:{level.index}/wall:"
    if not surface_id.startswith(prefix):
        return None
    try:
        first_id, second_id = map(int, surface_id[len(prefix) :].split(":"))
    except ValueError:
        return None
    endpoints = tuple(
        level.vertex_data.get_vertex(value) for value in (first_id, second_id)
    )
    return None if any(point is None for point in endpoints) else endpoints


# ### Stable topology helpers ###
def _corner_polygon(corners):
    return Polygon((corners[0], corners[1], corners[3], corners[2]))


def _edge_key(edge: Edge) -> tuple[int, int]:
    return min(edge.start_vertex_id, edge.end_vertex_id), max(
        edge.start_vertex_id, edge.end_vertex_id
    )


def _surface_id(level: LevelData, vertex_ids: tuple[int, int]) -> str:
    return build_wall_surface_id(level.index, f"{min(vertex_ids)}:{max(vertex_ids)}")
