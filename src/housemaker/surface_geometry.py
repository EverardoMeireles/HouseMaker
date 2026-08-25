# ### Imports ###
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import shapely
import trimesh
from shapely import LineString, Point, Polygon

from housemaker.glb import (
    WALL_OPENING_EPSILON,
    WallOpening,
    _build_visible_wall_pieces,
    _build_wall_openings,
    _clip_wall_segment_to_opening,
    _interpolate_2d_point,
)
from housemaker.level_coordinates import (
    build_level_base_z_lookup,
    level_image_to_world_xy,
)
from housemaker.models import Edge, LevelData, RoomData
from housemaker.uv_layout import build_room_walls


# ### Constants ###
SURFACE_TYPE_WALL = "wall"
SURFACE_TYPE_FLOOR = "floor"
SURFACE_TYPE_CEILING = "ceiling"
SURFACE_TYPES = frozenset(
    (SURFACE_TYPE_WALL, SURFACE_TYPE_FLOOR, SURFACE_TYPE_CEILING)
)
SURFACE_GEOMETRY_EPSILON = 1e-8
DEFAULT_SURFACE_OVERLAY_OFFSET_METERS = 0.003
MAX_SURFACE_OVERLAY_OFFSET_METERS = 0.05
SURFACE_OVERLAY_COPLANAR_DECIMALS = 6
DOORWAY_REVEAL_PARALLEL_COSINE = math.cos(math.radians(10.0))


# ### Surface models ###
@dataclass(frozen=True)
class FixedSurface:
    """One selectable, stable fixed surface in world-space Z-up coordinates."""

    surface_id: str
    surface_type: str
    level_index: int
    # Positional metadata only; persistent identity lives in ``surface_id``.
    room_index: int | None
    mesh: trimesh.Trimesh
    area_square_meters: float
    wall_key: str | None = None
    overlay_parent_surface_id: str | None = None

    def __post_init__(self) -> None:
        if not self.surface_id:
            raise ValueError("A fixed surface requires a stable ID.")
        if self.surface_type not in SURFACE_TYPES:
            raise ValueError(f"Unknown fixed surface type: {self.surface_type!r}.")
        if not isinstance(self.mesh, trimesh.Trimesh):
            raise TypeError("A fixed surface mesh must be a trimesh.Trimesh.")
        if (
            not math.isfinite(float(self.area_square_meters))
            or float(self.area_square_meters) <= 0.0
        ):
            raise ValueError("A fixed surface area must be finite and positive.")
        if self.overlay_parent_surface_id == self.surface_id:
            raise ValueError("A surface overlay cannot be its own parent.")


@dataclass(frozen=True)
class _SurfaceWallDefinition:
    surface_id: str
    start_point: tuple[float, float]
    end_point: tuple[float, float]


# ### Public surface builders ###
def build_fixed_surfaces(levels: Sequence[LevelData]) -> list[FixedSurface]:
    """Build walls, floors, and ceilings with stable semantic identities."""

    level_base_z = build_level_base_z_lookup(levels)
    surfaces: list[FixedSurface] = []
    for level in sorted(levels, key=lambda item: item.index):
        if not level.include_in_export:
            continue
        base_z_meters = level_base_z.get(level.index, 0.0)
        reveal_owner_by_opening_index = _build_doorway_reveal_owner_lookup(level)
        surfaces.extend(
            _build_plain_level_wall_surfaces(
                level,
                base_z_meters,
                reveal_owner_by_opening_index,
            )
        )
        for room_index, room in enumerate(level.rooms):
            surfaces.extend(
                _build_room_surfaces(
                    level=level,
                    room=room,
                    room_index=room_index,
                    base_z_meters=base_z_meters,
                    reveal_owner_by_opening_index=(
                        reveal_owner_by_opening_index
                    ),
                )
            )

        surfaces.extend(
            _build_level_residual_horizontal_surfaces(level, base_z_meters)
        )

    return sorted(surfaces, key=_get_surface_sort_key)


def get_combined_surface_area(
    surfaces: Iterable[FixedSurface],
    surface_ids: Iterable[str],
) -> float:
    """Return the one-sided physical area of the requested unique surfaces."""

    requested_ids = {str(surface_id) for surface_id in surface_ids}
    return float(
        sum(
            surface.area_square_meters
            for surface in surfaces
            if surface.surface_id in requested_ids
        )
    )


def build_surface_overlay_plane(
    parent_surface: FixedSurface,
    overlay_surface_id: str,
    normal_offset_meters: float,
) -> FixedSurface:
    """Copy the parent's dominant plane at a small signed normal offset."""

    if not isinstance(parent_surface, FixedSurface):
        raise TypeError("A surface overlay requires a fixed parent surface.")
    if parent_surface.overlay_parent_surface_id is not None:
        raise ValueError("A surface overlay cannot be created on another overlay.")
    normalized_id = str(overlay_surface_id).strip()
    if not normalized_id:
        raise ValueError("A surface overlay requires a stable ID.")
    offset = _normalize_surface_overlay_offset(normal_offset_meters)
    face_indices, plane_normal = _get_dominant_coplanar_face_patch(
        parent_surface.mesh
    )
    source_vertices = np.asarray(parent_surface.mesh.vertices, dtype=float)
    source_faces = np.asarray(parent_surface.mesh.faces, dtype=np.int64)
    face_vertices = source_vertices[source_faces[face_indices]].copy()
    face_vertices += plane_normal[np.newaxis, np.newaxis, :] * offset
    vertices = face_vertices.reshape(-1, 3)
    faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    area = float(mesh.area)
    if area <= SURFACE_GEOMETRY_EPSILON:
        raise ValueError("A surface overlay requires a usable planar patch.")
    return FixedSurface(
        surface_id=normalized_id,
        surface_type=parent_surface.surface_type,
        level_index=parent_surface.level_index,
        room_index=parent_surface.room_index,
        wall_key=parent_surface.wall_key,
        mesh=mesh,
        area_square_meters=area,
        overlay_parent_surface_id=parent_surface.surface_id,
    )


def get_surface_overlay_offset_toward_point(
    parent_surface: FixedSurface,
    world_point: Sequence[float],
    distance_meters: float = DEFAULT_SURFACE_OVERLAY_OFFSET_METERS,
) -> float:
    """Choose the signed offset that puts an overlay toward a viewer point."""

    if not isinstance(parent_surface, FixedSurface):
        raise TypeError("A surface overlay requires a fixed parent surface.")
    distance = abs(_normalize_surface_overlay_offset(distance_meters))
    face_indices, plane_normal = _get_dominant_coplanar_face_patch(
        parent_surface.mesh
    )
    try:
        point = np.asarray(tuple(world_point), dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "An overlay viewpoint must contain three coordinates."
        ) from error
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("An overlay viewpoint must contain finite XYZ coordinates.")
    faces = np.asarray(parent_surface.mesh.faces, dtype=np.int64)[face_indices]
    vertices = np.asarray(parent_surface.mesh.vertices, dtype=float)
    centroid = np.mean(vertices[faces].reshape(-1, 3), axis=0)
    direction = point - centroid
    return distance if float(np.dot(direction, plane_normal)) >= 0.0 else -distance


def build_wall_surface_id(
    level_index: int,
    wall_key: str,
    room_identity: int | None = None,
) -> str:
    """Build an ID using the room's persisted center-vertex identity."""

    if room_identity is None:
        return f"level:{int(level_index)}/wall:{wall_key}"
    return (
        f"level:{int(level_index)}/room:{int(room_identity)}/wall:{wall_key}"
    )


def build_horizontal_surface_id(
    level_index: int,
    surface_type: str,
    room_identity: int | None = None,
) -> str:
    """Build an ID using the room's persisted center-vertex identity."""

    if surface_type not in {SURFACE_TYPE_FLOOR, SURFACE_TYPE_CEILING}:
        raise ValueError("Horizontal surfaces must be a floor or ceiling.")
    if room_identity is None:
        return f"level:{int(level_index)}/{surface_type}"
    return (
        f"level:{int(level_index)}/room:{int(room_identity)}/{surface_type}"
    )


# ### Room surface builders ###
def _build_room_surfaces(
    level: LevelData,
    room: RoomData,
    room_index: int,
    base_z_meters: float,
    reveal_owner_by_opening_index: Mapping[int, str],
) -> list[FixedSurface]:
    surfaces: list[FixedSurface] = []
    room_identity = room.center_vertex_id
    doorway_openings = _build_wall_openings(level.doorways)
    for wall in build_room_walls(room, level.vertex_data):
        wall_surface = _build_wall_surface(
            level=level,
            start_point=wall.start_point,
            end_point=wall.end_point,
            wall_key=wall.key,
            wall_height_meters=room.height_meters,
            base_z_meters=base_z_meters,
            doorway_openings=doorway_openings,
            room_index=room_index,
            room_identity=room_identity,
            reveal_owner_by_opening_index=reveal_owner_by_opening_index,
        )
        if wall_surface is not None:
            surfaces.append(wall_surface)

    room_polygon = _build_room_world_polygon(level, room)
    if room_polygon is None:
        return surfaces
    floor_surface = _build_horizontal_surface(
        polygon=room_polygon,
        surface_id=build_horizontal_surface_id(
            level.index,
            SURFACE_TYPE_FLOOR,
            room_identity,
        ),
        surface_type=SURFACE_TYPE_FLOOR,
        level_index=level.index,
        room_index=room_index,
        z_meters=base_z_meters,
        normal_points_up=True,
    )
    ceiling_surface = _build_horizontal_surface(
        polygon=room_polygon,
        surface_id=build_horizontal_surface_id(
            level.index,
            SURFACE_TYPE_CEILING,
            room_identity,
        ),
        surface_type=SURFACE_TYPE_CEILING,
        level_index=level.index,
        room_index=room_index,
        z_meters=base_z_meters + room.height_meters,
        normal_points_up=False,
    )
    if floor_surface is not None:
        surfaces.append(floor_surface)
    if ceiling_surface is not None:
        surfaces.append(ceiling_surface)
    return surfaces


def _build_plain_level_wall_surfaces(
    level: LevelData,
    base_z_meters: float,
    reveal_owner_by_opening_index: Mapping[int, str],
) -> list[FixedSurface]:
    room_vertex_sets = [set(room.vertex_ids) for room in level.rooms]
    ignored_vertex_ids = {room.center_vertex_id for room in level.rooms}
    doorway_openings = _build_wall_openings(level.doorways)
    vertex_lookup = {vertex.id: vertex for vertex in level.vertex_data.vertices}
    surfaces: list[FixedSurface] = []
    for edge in sorted(level.vertex_data.edges, key=_get_edge_sort_key):
        if (
            edge.start_vertex_id in ignored_vertex_ids
            or edge.end_vertex_id in ignored_vertex_ids
            or any(
                edge.start_vertex_id in vertex_ids
                and edge.end_vertex_id in vertex_ids
                for vertex_ids in room_vertex_sets
            )
        ):
            continue
        start_vertex = vertex_lookup.get(edge.start_vertex_id)
        end_vertex = vertex_lookup.get(edge.end_vertex_id)
        if start_vertex is None or end_vertex is None:
            continue
        wall_key = (
            f"{min(edge.start_vertex_id, edge.end_vertex_id)}:"
            f"{max(edge.start_vertex_id, edge.end_vertex_id)}"
        )
        surface = _build_wall_surface(
            level=level,
            start_point=(start_vertex.x, start_vertex.y),
            end_point=(end_vertex.x, end_vertex.y),
            wall_key=wall_key,
            wall_height_meters=level.height_meters,
            base_z_meters=base_z_meters,
            doorway_openings=doorway_openings,
            room_index=None,
            room_identity=None,
            reveal_owner_by_opening_index=reveal_owner_by_opening_index,
        )
        if surface is not None:
            surfaces.append(surface)
    return surfaces


def _build_doorway_reveal_owner_lookup(level: LevelData) -> dict[int, str]:
    """Assign each reveal tunnel to one stable intersecting wall surface."""

    openings = _build_wall_openings(level.doorways)
    wall_definitions = _build_level_surface_wall_definitions(level)
    owner_by_opening_index: dict[int, str] = {}
    for opening_index, opening in enumerate(openings):
        candidates = [
            wall
            for wall in wall_definitions
            if _wall_is_parallel_to_doorway_width(
                wall.start_point,
                wall.end_point,
                opening,
            )
            and _clip_wall_segment_to_opening(
                start_point=wall.start_point,
                end_point=wall.end_point,
                doorway_opening=opening,
            )
            is not None
        ]
        if candidates:
            owner_by_opening_index[opening_index] = min(
                candidates,
                key=lambda wall: wall.surface_id,
            ).surface_id
    return owner_by_opening_index


def _build_level_surface_wall_definitions(
    level: LevelData,
) -> list[_SurfaceWallDefinition]:
    definitions: list[_SurfaceWallDefinition] = []
    room_vertex_sets = [set(room.vertex_ids) for room in level.rooms]
    ignored_vertex_ids = {room.center_vertex_id for room in level.rooms}
    vertex_lookup = {vertex.id: vertex for vertex in level.vertex_data.vertices}
    for edge in level.vertex_data.edges:
        if (
            edge.start_vertex_id in ignored_vertex_ids
            or edge.end_vertex_id in ignored_vertex_ids
            or any(
                edge.start_vertex_id in vertex_ids
                and edge.end_vertex_id in vertex_ids
                for vertex_ids in room_vertex_sets
            )
        ):
            continue
        start_vertex = vertex_lookup.get(edge.start_vertex_id)
        end_vertex = vertex_lookup.get(edge.end_vertex_id)
        if start_vertex is None or end_vertex is None:
            continue
        wall_key = (
            f"{min(edge.start_vertex_id, edge.end_vertex_id)}:"
            f"{max(edge.start_vertex_id, edge.end_vertex_id)}"
        )
        definitions.append(
            _SurfaceWallDefinition(
                surface_id=build_wall_surface_id(level.index, wall_key),
                start_point=(start_vertex.x, start_vertex.y),
                end_point=(end_vertex.x, end_vertex.y),
            )
        )
    for room in level.rooms:
        definitions.extend(
            _SurfaceWallDefinition(
                surface_id=build_wall_surface_id(
                    level.index,
                    wall.key,
                    room.center_vertex_id,
                ),
                start_point=wall.start_point,
                end_point=wall.end_point,
            )
            for wall in build_room_walls(room, level.vertex_data)
        )
    return definitions


# ### Wall geometry helpers ###
def _build_wall_surface(
    level: LevelData,
    start_point: tuple[float, float],
    end_point: tuple[float, float],
    wall_key: str,
    wall_height_meters: float,
    base_z_meters: float,
    doorway_openings: Sequence[WallOpening],
    room_index: int | None,
    room_identity: int | None,
    reveal_owner_by_opening_index: Mapping[int, str],
) -> FixedSurface | None:
    if (
        not math.isfinite(float(wall_height_meters))
        or wall_height_meters <= SURFACE_GEOMETRY_EPSILON
    ):
        return None

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for wall_piece in _build_visible_wall_pieces(
        start_point=start_point,
        end_point=end_point,
        wall_height_meters=wall_height_meters,
        doorway_openings=doorway_openings,
    ):
        piece_start = _interpolate_2d_point(
            start_point,
            end_point,
            wall_piece.start_ratio,
        )
        piece_end = _interpolate_2d_point(
            start_point,
            end_point,
            wall_piece.end_ratio,
        )
        start_world = level_image_to_world_xy(level, *piece_start)
        end_world = level_image_to_world_xy(level, *piece_end)
        _append_quad(
            vertices,
            faces,
            (
                (*start_world, base_z_meters + wall_piece.bottom_height_meters),
                (*end_world, base_z_meters + wall_piece.bottom_height_meters),
                (*end_world, base_z_meters + wall_piece.top_height_meters),
                (*start_world, base_z_meters + wall_piece.top_height_meters),
            ),
        )

    surface_id = build_wall_surface_id(level.index, wall_key, room_identity)
    for opening_index, doorway_opening in enumerate(doorway_openings):
        if reveal_owner_by_opening_index.get(opening_index) != surface_id:
            continue
        _append_connected_doorway_reveals(
            vertices=vertices,
            faces=faces,
            level=level,
            wall_start=start_point,
            wall_end=end_point,
            wall_height_meters=wall_height_meters,
            base_z_meters=base_z_meters,
            doorway_opening=doorway_opening,
        )

    mesh = _build_mesh(vertices, faces)
    if mesh is None:
        return None
    area = float(mesh.area)
    if area <= SURFACE_GEOMETRY_EPSILON:
        return None
    return FixedSurface(
        surface_id=surface_id,
        surface_type=SURFACE_TYPE_WALL,
        level_index=level.index,
        room_index=room_index,
        wall_key=wall_key,
        mesh=mesh,
        area_square_meters=area,
    )


def _append_connected_doorway_reveals(
    vertices: list[list[float]],
    faces: list[list[int]],
    level: LevelData,
    wall_start: tuple[float, float],
    wall_end: tuple[float, float],
    wall_height_meters: float,
    base_z_meters: float,
    doorway_opening: WallOpening,
) -> None:
    interval = _clip_wall_segment_to_opening(
        start_point=wall_start,
        end_point=wall_end,
        doorway_opening=doorway_opening,
    )
    if interval is None or not _wall_is_parallel_to_doorway_width(
        wall_start,
        wall_end,
        doorway_opening,
    ):
        return

    opening_height = min(doorway_opening.height_meters, wall_height_meters)
    if opening_height <= SURFACE_GEOMETRY_EPSILON:
        return
    low_width = -doorway_opening.half_width_pixels
    high_width = doorway_opening.half_width_pixels
    negative_depth = -doorway_opening.half_depth_pixels
    positive_depth = doorway_opening.half_depth_pixels
    low_negative = _doorway_local_to_world(
        level,
        doorway_opening,
        low_width,
        negative_depth,
    )
    low_positive = _doorway_local_to_world(
        level,
        doorway_opening,
        low_width,
        positive_depth,
    )
    high_negative = _doorway_local_to_world(
        level,
        doorway_opening,
        high_width,
        negative_depth,
    )
    high_positive = _doorway_local_to_world(
        level,
        doorway_opening,
        high_width,
        positive_depth,
    )
    bottom_z = base_z_meters
    top_z = base_z_meters + opening_height
    _append_quad(
        vertices,
        faces,
        (
            (*low_negative, bottom_z),
            (*low_positive, bottom_z),
            (*low_positive, top_z),
            (*low_negative, top_z),
        ),
    )
    _append_quad(
        vertices,
        faces,
        (
            (*high_positive, bottom_z),
            (*high_negative, bottom_z),
            (*high_negative, top_z),
            (*high_positive, top_z),
        ),
    )
    _append_quad(
        vertices,
        faces,
        (
            (*low_negative, top_z),
            (*high_negative, top_z),
            (*high_positive, top_z),
            (*low_positive, top_z),
        ),
    )


def _wall_is_parallel_to_doorway_width(
    wall_start: tuple[float, float],
    wall_end: tuple[float, float],
    doorway_opening: WallOpening,
) -> bool:
    delta_x = wall_end[0] - wall_start[0]
    delta_y = wall_end[1] - wall_start[1]
    length = math.hypot(delta_x, delta_y)
    if length <= WALL_OPENING_EPSILON:
        return False
    alignment = abs(
        delta_x / length * doorway_opening.width_direction_x
        + delta_y / length * doorway_opening.width_direction_y
    )
    return alignment >= DOORWAY_REVEAL_PARALLEL_COSINE


def _doorway_local_to_world(
    level: LevelData,
    doorway_opening: WallOpening,
    width_position: float,
    depth_position: float,
) -> tuple[float, float]:
    image_x = (
        doorway_opening.center_x
        + doorway_opening.width_direction_x * width_position
        + doorway_opening.depth_direction_x * depth_position
    )
    image_y = (
        doorway_opening.center_y
        + doorway_opening.width_direction_y * width_position
        + doorway_opening.depth_direction_y * depth_position
    )
    return level_image_to_world_xy(level, image_x, image_y)


# ### Horizontal geometry helpers ###
def _build_level_residual_horizontal_surfaces(
    level: LevelData,
    base_z_meters: float,
) -> list[FixedSurface]:
    level_polygon = _build_level_contour_world_polygon(level)
    if level_polygon is None:
        return []
    room_polygons = [
        polygon
        for room in level.rooms
        if (polygon := _build_room_world_polygon(level, room)) is not None
    ]
    residual_geometry = level_polygon
    if room_polygons:
        residual_geometry = level_polygon.difference(shapely.union_all(room_polygons))
    if residual_geometry.is_empty or residual_geometry.area <= SURFACE_GEOMETRY_EPSILON:
        return []
    surfaces: list[FixedSurface] = []
    for surface_type, z_meters, points_up in (
        (SURFACE_TYPE_FLOOR, base_z_meters, True),
        (
            SURFACE_TYPE_CEILING,
            base_z_meters + level.height_meters,
            False,
        ),
    ):
        surface = _build_horizontal_surface(
            polygon=residual_geometry,
            surface_id=build_horizontal_surface_id(level.index, surface_type),
            surface_type=surface_type,
            level_index=level.index,
            room_index=None,
            z_meters=z_meters,
            normal_points_up=points_up,
        )
        if surface is not None:
            surfaces.append(surface)
    return surfaces


def _build_horizontal_surface(
    polygon,
    surface_id: str,
    surface_type: str,
    level_index: int,
    room_index: int | None,
    z_meters: float,
    normal_points_up: bool,
) -> FixedSurface | None:
    vertices_2d: list[tuple[float, float]] = []
    vertex_index_by_point: dict[tuple[float, float], int] = {}
    faces: list[list[int]] = []
    for polygon_part in shapely.get_parts(polygon):
        if not isinstance(polygon_part, Polygon) or polygon_part.is_empty:
            continue
        triangles = shapely.constrained_delaunay_triangles(polygon_part)
        for triangle in shapely.get_parts(triangles):
            if not isinstance(triangle, Polygon) or triangle.is_empty:
                continue
            if not shapely.covers(
                polygon_part,
                triangle.representative_point(),
            ):
                continue
            points = [
                (float(point_x), float(point_y))
                for point_x, point_y in list(triangle.exterior.coords)[:-1]
            ]
            if len(points) != 3:
                continue
            if _signed_triangle_area(points) < 0.0:
                points.reverse()
            indices: list[int] = []
            for point in points:
                if point not in vertex_index_by_point:
                    vertex_index_by_point[point] = len(vertices_2d)
                    vertices_2d.append(point)
                indices.append(vertex_index_by_point[point])
            faces.append(indices if normal_points_up else list(reversed(indices)))

    if not vertices_2d or not faces:
        return None
    mesh = trimesh.Trimesh(
        vertices=np.asarray(
            [(point_x, point_y, z_meters) for point_x, point_y in vertices_2d],
            dtype=float,
        ),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    return FixedSurface(
        surface_id=surface_id,
        surface_type=surface_type,
        level_index=level_index,
        room_index=room_index,
        mesh=mesh,
        area_square_meters=float(polygon.area),
    )


def _build_room_world_polygon(
    level: LevelData,
    room: RoomData,
) -> Polygon | None:
    room_vertex_ids = set(room.vertex_ids)
    lines: list[LineString] = []
    for edge in level.vertex_data.edges:
        if (
            edge.start_vertex_id not in room_vertex_ids
            or edge.end_vertex_id not in room_vertex_ids
        ):
            continue
        start_vertex = level.vertex_data.get_vertex(edge.start_vertex_id)
        end_vertex = level.vertex_data.get_vertex(edge.end_vertex_id)
        if start_vertex is None or end_vertex is None:
            continue
        lines.append(
            LineString(
                (
                    level_image_to_world_xy(level, start_vertex.x, start_vertex.y),
                    level_image_to_world_xy(level, end_vertex.x, end_vertex.y),
                )
            )
        )
    polygons = [
        candidate
        for candidate in shapely.get_parts(shapely.polygonize(lines))
        if isinstance(candidate, Polygon)
        and candidate.area > SURFACE_GEOMETRY_EPSILON
    ]
    if polygons:
        center_vertex = level.vertex_data.get_vertex(room.center_vertex_id)
        if center_vertex is not None:
            center = Point(
                level_image_to_world_xy(
                    level,
                    center_vertex.x,
                    center_vertex.y,
                )
            )
            containing = [polygon for polygon in polygons if polygon.covers(center)]
            if containing:
                return max(containing, key=lambda item: item.area)
        return max(polygons, key=lambda item: item.area)
    return _build_polygon_from_vertices(level, room.vertex_ids, room.center_vertex_id)


def _build_level_contour_world_polygon(level: LevelData) -> Polygon | None:
    return _build_polygon_from_vertices(
        level,
        level.floor_contour_vertex_ids,
        None,
        preserve_order=True,
    )


def _build_polygon_from_vertices(
    level: LevelData,
    vertex_ids: Sequence[int],
    center_vertex_id: int | None,
    preserve_order: bool = False,
) -> Polygon | None:
    vertices = [
        vertex
        for vertex_id in vertex_ids
        if (vertex := level.vertex_data.get_vertex(vertex_id)) is not None
        and vertex.id != center_vertex_id
    ]
    if len(vertices) < 3:
        return None
    if not preserve_order:
        center_x = sum(vertex.x for vertex in vertices) / len(vertices)
        center_y = sum(vertex.y for vertex in vertices) / len(vertices)
        vertices.sort(
            key=lambda vertex: math.atan2(
                vertex.y - center_y,
                vertex.x - center_x,
            )
        )
    polygon = Polygon(
        [level_image_to_world_xy(level, vertex.x, vertex.y) for vertex in vertices]
    )
    if not polygon.is_valid:
        repaired = polygon.buffer(0)
        if not isinstance(repaired, Polygon):
            return None
        polygon = repaired
    if polygon.area <= SURFACE_GEOMETRY_EPSILON:
        return None
    return polygon


# ### Mesh helpers ###
def _append_quad(
    vertices: list[list[float]],
    faces: list[list[int]],
    corners: Sequence[tuple[float, float, float]],
) -> None:
    if len(corners) != 4:
        return
    corner_array = np.asarray(corners, dtype=float)
    if not np.all(np.isfinite(corner_array)):
        return
    first_area = np.linalg.norm(
        np.cross(corner_array[1] - corner_array[0], corner_array[2] - corner_array[0])
    )
    second_area = np.linalg.norm(
        np.cross(corner_array[2] - corner_array[0], corner_array[3] - corner_array[0])
    )
    if first_area + second_area <= SURFACE_GEOMETRY_EPSILON:
        return
    vertex_offset = len(vertices)
    vertices.extend(corner_array.tolist())
    faces.extend(
        (
            [vertex_offset, vertex_offset + 1, vertex_offset + 2],
            [vertex_offset, vertex_offset + 2, vertex_offset + 3],
        )
    )


def _build_mesh(
    vertices: list[list[float]],
    faces: list[list[int]],
) -> trimesh.Trimesh | None:
    if not vertices or not faces:
        return None
    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )


def _get_dominant_coplanar_face_patch(
    mesh: trimesh.Trimesh,
) -> tuple[np.ndarray, np.ndarray]:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=float)
    normals = np.asarray(mesh.face_normals, dtype=float)
    areas = np.asarray(mesh.area_faces, dtype=float)
    if (
        faces.ndim != 2
        or faces.shape[1:] != (3,)
        or not len(faces)
        or vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or normals.shape != (len(faces), 3)
        or areas.shape != (len(faces),)
    ):
        raise ValueError("A surface overlay requires triangular parent geometry.")

    grouped_indices: dict[tuple[float, ...], list[int]] = {}
    group_normals: dict[tuple[float, ...], np.ndarray] = {}
    for face_index, normal in enumerate(normals):
        normal_length = float(np.linalg.norm(normal))
        if normal_length <= SURFACE_GEOMETRY_EPSILON:
            continue
        unit_normal = normal / normal_length
        plane_offset = float(np.dot(unit_normal, vertices[faces[face_index, 0]]))
        key = tuple(
            float(value)
            for value in np.round(
                np.append(unit_normal, plane_offset),
                SURFACE_OVERLAY_COPLANAR_DECIMALS,
            )
        )
        grouped_indices.setdefault(key, []).append(face_index)
        group_normals.setdefault(key, unit_normal)
    if not grouped_indices:
        raise ValueError("A surface overlay requires a usable planar parent patch.")
    dominant_key = max(
        grouped_indices,
        key=lambda key: (
            float(np.sum(areas[grouped_indices[key]])),
            len(grouped_indices[key]),
            key,
        ),
    )
    return (
        np.asarray(grouped_indices[dominant_key], dtype=np.int64),
        group_normals[dominant_key].copy(),
    )


def _normalize_surface_overlay_offset(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("A surface overlay offset must be a number.")
    try:
        offset = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("A surface overlay offset must be a number.") from error
    if (
        not math.isfinite(offset)
        or abs(offset) <= SURFACE_GEOMETRY_EPSILON
        or abs(offset) > MAX_SURFACE_OVERLAY_OFFSET_METERS
    ):
        raise ValueError(
            "A surface overlay offset must be finite and no more than 5 cm."
        )
    return offset


# ### Numeric and ordering helpers ###
def _signed_triangle_area(points: Sequence[tuple[float, float]]) -> float:
    first, second, third = points
    return (
        (second[0] - first[0]) * (third[1] - first[1])
        - (second[1] - first[1]) * (third[0] - first[0])
    ) / 2.0


def _get_edge_sort_key(edge: Edge) -> tuple[int, int]:
    return tuple(sorted((edge.start_vertex_id, edge.end_vertex_id)))


def _get_surface_sort_key(
    surface: FixedSurface,
) -> tuple[int, int, int, str]:
    type_order = {
        SURFACE_TYPE_FLOOR: 0,
        SURFACE_TYPE_CEILING: 1,
        SURFACE_TYPE_WALL: 2,
    }
    return (
        surface.level_index,
        -1 if surface.room_index is None else surface.room_index,
        type_order[surface.surface_type],
        surface.surface_id,
    )
