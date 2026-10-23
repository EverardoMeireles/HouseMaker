"""Derive conservative streaming rooms without changing authored wall topology.

All geometry is in canonical HouseMaker world XY/Z-up coordinates, before
the export's global multipliers. Doorway face spans are temporary boundaries;
portals retain connectivity across them. Unresolved layouts are marked as
fallback zones, never evidence that an unclassified area can be hidden.
"""

# ### Imports ###
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import partial

import numpy as np
import shapely
from shapely import LineString, Point, Polygon, box
from shapely.geometry.base import BaseGeometry

from housemaker.doorway_connections import get_level_structural_edges
from housemaker.doorway_topology import doorway_corner_positions
from housemaker.floor_geometry import (
    build_level_floor_footprint,
    build_level_open_space_geometry,
    filter_automatic_slab_opening_geometry,
)
from housemaker.landing_geometry import resolve_stair_section_image_points
from housemaker.level_coordinates import (
    build_level_base_z_lookup,
    level_image_to_world_xy,
    level_world_to_image_xy,
)
from housemaker.models import LevelData, StairData

# ### Constants ###
ROOM_GEOMETRY_EPSILON = 1e-7
ROOM_ID_COORDINATE_DECIMALS = 6
MIN_ROOM_AREA_SQUARE_METERS = 0.4
# Parallel wall skins produce narrow polygonized strips, not walkable rooms.
# A layout with no supported interior is retained as one fail-open zone.
MIN_ROOM_INTERIOR_RADIUS_METERS = 0.2
SURFACE_ROOM_PROBE_METERS = 0.025


# ### Immutable room and portal descriptions ###
@dataclass(frozen=True)
class RoomZone:
    room_id: str
    level_index: int
    polygon: Polygon
    floor_z_meters: float
    top_z_meters: float
    is_fallback: bool = False
    name: str = ""

    @property
    def room_name(self) -> str:
        return self.name


@dataclass(frozen=True)
class RoomPortal:
    portal_id: str
    kind: str
    room_ids: tuple[str, ...]
    walkable: bool
    level_indices: tuple[int, ...]


@dataclass(frozen=True)
class RoomZoneLayout:
    rooms: tuple[RoomZone, ...] = ()
    portals: tuple[RoomPortal, ...] = ()

    def rooms_on_level(self, level_index: int) -> tuple[RoomZone, ...]:
        return tuple(room for room in self.rooms if room.level_index == level_index)


# ### Room extraction and read-only floor clipping ###
def build_room_zones(
    levels: Sequence[LevelData],
    stairs: Sequence[StairData] = (),
    *,
    include_excluded: bool = False,
) -> RoomZoneLayout:
    """Polygonize confirmed walls, preserve slab holes, and connect portals.

    Only included levels become export rooms. All levels still contribute to
    the anchored vertical positions, just as they do in structural export.
    Room IDs depend on image-space boundaries, not list order or translations.
    """

    levels, stairs = tuple(levels), tuple(stairs)
    if not all(isinstance(level, LevelData) for level in levels):
        raise TypeError("Room zones require LevelData values.")
    if len({level.index for level in levels}) != len(levels):
        raise ValueError("Room zones require unique level indices.")
    if not all(isinstance(stair, StairData) for stair in stairs):
        raise TypeError("Room portals require StairData values.")
    from housemaker.glb import build_stair_slab_openings_by_level

    floor_z = build_level_base_z_lookup(levels)
    opening_levels = (
        tuple(replace(level, include_in_export=True) for level in levels)
        if include_excluded
        else levels
    )
    automatic_openings = build_stair_slab_openings_by_level(opening_levels, stairs)
    rooms: list[RoomZone] = []
    for level in sorted(levels, key=lambda item: item.index):
        if not include_excluded and not level.include_in_export:
            continue
        rooms.extend(
            _build_level_rooms(level, floor_z[level.index], automatic_openings)
        )
    named_rooms = []
    room_numbers: dict[int, int] = {}
    for room in sorted(rooms, key=lambda room: room.room_id):
        room_numbers[room.level_index] = room_numbers.get(room.level_index, 0) + 1
        label = (
            "Unassigned"
            if room.is_fallback
            else f"Room {room_numbers[room.level_index]}"
        )
        named_rooms.append(replace(room, name=f"L{room.level_index} — {label}"))
    layout = RoomZoneLayout(tuple(named_rooms))
    portals = [
        *_doorway_portals(levels, layout),
        *_stair_portals(levels, stairs, layout),
    ]
    return RoomZoneLayout(
        layout.rooms, tuple(sorted(portals, key=lambda portal: portal.portal_id))
    )


def _world_point(level: LevelData, point, _size=None) -> np.ndarray:
    return np.asarray(level_image_to_world_xy(level, *point), dtype=float)


def _polygon_parts(geometry: BaseGeometry | None) -> tuple[Polygon, ...]:
    if geometry is None or geometry.is_empty:
        return ()
    return tuple(
        part
        for part in shapely.get_parts(shapely.make_valid(geometry))
        if isinstance(part, Polygon) and part.area > ROOM_GEOMETRY_EPSILON
    )


def _build_level_rooms(level, floor_z, automatic_openings) -> tuple[RoomZone, ...]:
    point_to_world = partial(_world_point, level)
    # A level transform scales wall thickness and room dimensions together;
    # it must not turn the same narrow wall strip into a newly detected room.
    interior_radius = MIN_ROOM_INTERIOR_RADIUS_METERS * float(level.scale)
    minimum_area = MIN_ROOM_AREA_SQUARE_METERS * float(level.scale) ** 2
    vertices = {vertex.id: vertex for vertex in level.vertex_data.vertices}
    lines = []
    for edge in get_level_structural_edges(level):
        first, second = (
            vertices.get(edge.start_vertex_id),
            vertices.get(edge.end_vertex_id),
        )
        if first is None or second is None:
            continue
        points = tuple(
            point_to_world((vertex.x, vertex.y)) for vertex in (first, second)
        )
        if np.isfinite(points).all() and math.dist(*points) > ROOM_GEOMETRY_EPSILON:
            lines.append(LineString(points))
    explicit = build_level_open_space_geometry(
        level, level.image_size_pixels, point_to_world
    )
    automatic = filter_automatic_slab_opening_geometry(
        explicit, automatic_openings.get(level.index)
    )
    holes = shapely.union_all(
        [item for item in (explicit, automatic) if item is not None]
    )
    footprint = build_level_floor_footprint(
        level,
        level.image_size_pixels,
        point_to_world,
        additional_coherency_geometry=automatic,
    )
    if not lines:
        return _fallback_rooms(level, floor_z, footprint, holes, vertices)
    noded = shapely.union_all(lines)
    polygons, cuts, dangles, invalid = shapely.polygonize_full(shapely.get_parts(noded))
    # Preserve proven rooms even when unrelated unfinished walls remain.
    unresolved = any(not geometry.is_empty for geometry in (cuts, dangles, invalid))
    candidates = [
        polygon
        for polygon in _polygon_parts(polygons)
        if polygon.area >= minimum_area
        and not polygon.buffer(-interior_radius).is_empty
    ]
    if not candidates:
        return _fallback_rooms(level, floor_z, footprint, holes, vertices)
    result = []
    for polygon in candidates:
        if footprint is not None:
            polygon = polygon.intersection(footprint)
        if not holes.is_empty:
            polygon = polygon.difference(holes)
        for part in _polygon_parts(polygon):
            result.append(_room(level, part, floor_z))
    if unresolved and result and footprint is not None:
        residual = footprint.difference(
            shapely.union_all([room.polygon for room in result])
        )
        if not holes.is_empty:
            residual = residual.difference(holes)
        for part in _polygon_parts(residual):
            if not part.buffer(-interior_radius).is_empty:
                result.append(_room(level, part, floor_z, fallback=True))
    return tuple(result) or _fallback_rooms(level, floor_z, footprint, holes, vertices)


def _fallback_rooms(level, floor_z, footprint, holes, vertices) -> tuple[RoomZone, ...]:
    if footprint is None:
        points = [
            _world_point(level, (vertex.x, vertex.y)) for vertex in vertices.values()
        ]
        if points:
            bounds = np.asarray(points, dtype=float)
            low, high = bounds.min(axis=0), bounds.max(axis=0)
            footprint = box(*np.minimum(low, high - 0.5), *np.maximum(high, low + 0.5))
        else:
            return ()
    if not holes.is_empty:
        footprint = footprint.difference(holes)
    parts = _polygon_parts(footprint)
    return tuple(_room(level, polygon, floor_z, fallback=True) for polygon in parts)


def _canonical_ring(points) -> tuple[tuple[float, float], ...]:
    ring = tuple(
        tuple(round(float(value), ROOM_ID_COORDINATE_DECIMALS) for value in point[:2])
        for point in points
    )
    if ring and ring[0] == ring[-1]:
        ring = ring[:-1]
    if not ring:
        return ()
    variants = (ring, tuple(reversed(ring)))
    return min(
        variant[index:] + variant[:index]
        for variant in variants
        for index in range(len(variant))
    )


def _room(level, polygon, floor_z, *, fallback=False) -> RoomZone:
    image_rings = (
        (
            _canonical_ring(
                level_world_to_image_xy(level, *point)
                for point in polygon.exterior.coords
            ),
            sorted(
                _canonical_ring(
                    level_world_to_image_xy(level, *point) for point in ring.coords
                )
                for ring in polygon.interiors
            ),
        )
        if not polygon.is_empty
        else ((), [])
    )
    signature = json.dumps((level.index, fallback, image_rings), separators=(",", ":"))
    room_id = (
        f"room:{level.index}:" + hashlib.sha256(signature.encode()).hexdigest()[:16]
    )
    return RoomZone(
        room_id,
        level.index,
        polygon,
        float(floor_z),
        float(floor_z + level.height_meters),
        fallback,
    )


# ### Conservative object and surface ownership ###
def classify_object_room_ids(
    layout: RoomZoneLayout,
    level_index: int,
    anchor_xy: Sequence[float],
    footprint: BaseGeometry | Sequence[Sequence[float]] | None = None,
) -> tuple[str, ...]:
    """Return every occupied room; empty/multiple IDs belong in a shared bucket."""

    anchor = _finite_points((anchor_xy,), 2)[0]
    geometry = None
    if footprint is not None:
        geometry = (
            footprint
            if isinstance(footprint, BaseGeometry)
            else Polygon(_finite_points(footprint, 2))
        )
        if not geometry.is_valid:
            geometry = shapely.make_valid(geometry)
    rooms = layout.rooms_on_level(level_index)
    if (
        geometry is not None
        and not geometry.is_empty
        and geometry.area > ROOM_GEOMETRY_EPSILON
    ):
        matched = tuple(
            room.room_id
            for room in rooms
            if not room.is_fallback
            and room.polygon.intersection(geometry).area > ROOM_GEOMETRY_EPSILON
        )
    else:
        matched = tuple(
            room.room_id
            for room in rooms
            if not room.is_fallback
            and room.polygon.buffer(ROOM_GEOMETRY_EPSILON).covers(Point(anchor))
        )
    return matched or tuple(room.room_id for room in rooms if room.is_fallback)


def room_ids_for_surface(
    layout: RoomZoneLayout,
    level_index: int,
    points_xyz: Sequence[Sequence[float]],
    normal_xyz: Sequence[float] | None = None,
) -> tuple[str, ...]:
    """Classify complete surfaces, probing the visible side of vertical walls."""

    points = _finite_points(points_xyz, 3)
    if not len(points):
        return ()
    center = points.mean(axis=0)
    if normal_xyz is not None:
        normal = _finite_points((normal_xyz,), 3)[0]
        length = float(np.linalg.norm(normal[:2]))
        if length > ROOM_GEOMETRY_EPSILON and abs(float(normal[2])) <= length * 0.1:
            probe = center[:2] + normal[:2] / length * SURFACE_ROOM_PROBE_METERS
            return classify_object_room_ids(layout, level_index, probe)
    footprint = shapely.MultiPoint(points[:, :2]).convex_hull
    if footprint.area > ROOM_GEOMETRY_EPSILON:
        return classify_object_room_ids(layout, level_index, center[:2], footprint)
    return tuple(
        room.room_id
        for room in layout.rooms_on_level(level_index)
        if room.is_fallback or room.polygon.distance(footprint) <= ROOM_GEOMETRY_EPSILON
    )


def _finite_points(points, dimensions) -> np.ndarray:
    values = np.asarray(tuple(points), dtype=float)
    if not len(values):
        return np.empty((0, dimensions))
    if (
        values.ndim != 2
        or values.shape[1] != dimensions
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            f"Room coordinates require finite {dimensions}-component points."
        )
    return values


# ### Doorway and stair portals (not structural wall edits) ###
def _doorway_portals(levels, layout):
    for level in levels:
        if not layout.rooms_on_level(level.index):
            continue
        for doorway in level.doorways:
            corners = doorway.corner_vertex_ids
            vertices = tuple(
                level.vertex_data.get_vertex(value) for value in corners or ()
            )
            if len(vertices) == 4 and all(vertex is not None for vertex in vertices):
                image_points = tuple((vertex.x, vertex.y) for vertex in vertices)
            else:
                image_points = doorway_corner_positions(
                    doorway, meters_per_pixel=level.meters_per_pixel
                )
            world = np.asarray([_world_point(level, point) for point in image_points])
            front, back = world[:2].mean(axis=0), world[2:].mean(axis=0)
            direction = back - front
            length = float(np.linalg.norm(direction))
            if length <= ROOM_GEOMETRY_EPSILON:
                continue
            direction /= length
            room_ids = tuple(
                sorted(
                    {
                        *classify_object_room_ids(
                            layout,
                            level.index,
                            front - direction * SURFACE_ROOM_PROBE_METERS,
                        ),
                        *classify_object_room_ids(
                            layout,
                            level.index,
                            back + direction * SURFACE_ROOM_PROBE_METERS,
                        ),
                    }
                )
            )
            if room_ids:
                yield RoomPortal(
                    f"doorway:{level.index}:{doorway.doorway_id}",
                    "doorway",
                    room_ids,
                    True,
                    (level.index,),
                )


def _stair_portals(levels, stairs, layout):
    lookup = {level.index: level for level in levels}
    for stair in stairs:
        room_ids, level_indices = set(), set()
        for section in stair.sections:
            level = lookup.get(section.level_index)
            if level is None or not layout.rooms_on_level(level.index):
                continue
            points = np.asarray(
                [
                    _world_point(level, point)
                    for point in resolve_stair_section_image_points(level, section)
                ]
            )
            center = points.mean(axis=0)
            matched = classify_object_room_ids(layout, level.index, center)
            if not matched:
                # An upper endpoint can sit inside its correctly cut slab hole.
                # Nearby floor ownership connects it without filling that hole.
                rooms = layout.rooms_on_level(level.index)
                distances = [
                    (room.polygon.distance(Point(center)), room.room_id)
                    for room in rooms
                ]
                nearest = min(
                    (distance for distance, _id in distances), default=math.inf
                )
                radius = (
                    float(np.linalg.norm(points[1] - points[0]))
                    + SURFACE_ROOM_PROBE_METERS
                )
                matched = tuple(
                    room_id
                    for distance, room_id in distances
                    if distance <= min(nearest + ROOM_GEOMETRY_EPSILON, radius)
                )
            room_ids.update(matched)
            level_indices.add(level.index)
        if room_ids and len(level_indices) > 1:
            yield RoomPortal(
                f"stair:{stair.stair_id}",
                "stair",
                tuple(sorted(room_ids)),
                True,
                tuple(sorted(level_indices)),
            )


# ### Canvas conversion (exterior and hole rings stay separate) ###
def room_zone_image_rings(room: RoomZone, level: LevelData):
    """Return image-space exterior and holes, without modifying either owner."""

    if room.level_index != level.index:
        raise ValueError("The room and image level must match.")
    return (
        tuple(
            level_world_to_image_xy(level, *point)
            for point in room.polygon.exterior.coords
        ),
        tuple(
            tuple(level_world_to_image_xy(level, *point) for point in ring.coords)
            for ring in room.polygon.interiors
        ),
    )
