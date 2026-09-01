# ### Imports ###
from __future__ import annotations

import math
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import shapely
import trimesh
from shapely import LineString, Point, Polygon

from housemaker.glb import (
    DoorwayReveal,
    WALL_OPENING_EPSILON,
    WallOpening,
    WindowReveal,
    _build_level_doorway_reveals,
    _build_level_window_reveals,
    _build_wall_opening_reveal_quads,
    _build_visible_wall_pieces,
    _build_level_wall_openings,
    _interpolate_2d_point,
)
from housemaker.level_coordinates import (
    build_level_base_z_lookup,
    level_image_to_world_xy,
)
from housemaker.models import Edge, LevelData, RoomData, WindowData
from housemaker.uv_layout import build_room_walls


# ### Constants ###
SURFACE_TYPE_WALL = "wall"
SURFACE_TYPE_FLOOR = "floor"
SURFACE_TYPE_CEILING = "ceiling"
SURFACE_TYPES = frozenset(
    (SURFACE_TYPE_WALL, SURFACE_TYPE_FLOOR, SURFACE_TYPE_CEILING)
)
SURFACE_GEOMETRY_EPSILON = 1e-8
MIN_WINDOW_SIZE_METERS = 0.05
WINDOW_PLANE_DISTANCE_TOLERANCE_METERS = 0.01
WINDOW_PATCH_COVERAGE_TOLERANCE_METERS = 1e-7
SURFACE_INTERIOR_PROBE_RATIOS = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3)


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
    wall_start_world: tuple[float, float, float] | None = None
    wall_end_world: tuple[float, float, float] | None = None
    wall_height_meters: float | None = None

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


@dataclass(frozen=True)
class WallWindowPlacement:
    """A validated rectangle in one stable wall's normalized local frame."""

    wall_surface_id: str
    start_ratio: float
    end_ratio: float
    bottom_ratio: float
    top_ratio: float

    def __post_init__(self) -> None:
        validated = WindowData(
            window_id="placement",
            wall_surface_id=self.wall_surface_id,
            start_ratio=self.start_ratio,
            end_ratio=self.end_ratio,
            bottom_ratio=self.bottom_ratio,
            top_ratio=self.top_ratio,
        )
        for field_name in (
            "wall_surface_id",
            "start_ratio",
            "end_ratio",
            "bottom_ratio",
            "top_ratio",
        ):
            object.__setattr__(self, field_name, getattr(validated, field_name))


# ### Public surface builders ###
def build_fixed_surfaces(levels: Sequence[LevelData]) -> list[FixedSurface]:
    """Build walls, floors, and ceilings with stable semantic identities."""

    level_base_z = build_level_base_z_lookup(levels)
    surfaces: list[FixedSurface] = []
    for level in sorted(levels, key=lambda item: item.index):
        if not level.include_in_export:
            continue
        base_z_meters = level_base_z.get(level.index, 0.0)
        doorway_reveals_by_surface_id = _group_doorway_reveals_by_surface_id(
            _build_level_doorway_reveals(
                level,
                include_single_wall_fallback=True,
            )
        )
        window_reveals_by_surface_id = _group_window_reveals_by_surface_id(
            _build_level_window_reveals(level)
        )
        surfaces.extend(
            _build_plain_level_wall_surfaces(
                level,
                base_z_meters,
                doorway_reveals_by_surface_id,
                window_reveals_by_surface_id,
            )
        )
        for room_index, room in enumerate(level.rooms):
            surfaces.extend(
                _build_room_surfaces(
                    level=level,
                    room=room,
                    room_index=room_index,
                    base_z_meters=base_z_meters,
                    doorway_reveals_by_surface_id=(
                        doorway_reveals_by_surface_id
                    ),
                    window_reveals_by_surface_id=(
                        window_reveals_by_surface_id
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


def build_wall_window_placement(
    parent_surface: FixedSurface,
    first_world_point: Sequence[float],
    second_world_point: Sequence[float],
) -> WallWindowPlacement:
    """Validate a dragged wall rectangle and return stable local bounds."""

    wall_start, wall_end, wall_height = _get_window_wall_frame(parent_surface)
    first = _normalize_window_world_point(first_world_point)
    second = _normalize_window_world_point(second_world_point)
    wall_axis = wall_end - wall_start
    wall_width = float(np.linalg.norm(wall_axis))
    tangent = wall_axis / wall_width
    plane_normal = np.cross(tangent, np.asarray((0.0, 0.0, 1.0)))
    for point in (first, second):
        plane_distance = abs(float(np.dot(point - wall_start, plane_normal)))
        if plane_distance > WINDOW_PLANE_DISTANCE_TOLERANCE_METERS:
            raise ValueError("A window rectangle must stay on its selected wall.")

    horizontal_ratios = sorted(
        float(np.dot(point - wall_start, tangent)) / wall_width
        for point in (first, second)
    )
    vertical_ratios = sorted(
        (float(point[2]) - float(wall_start[2])) / wall_height
        for point in (first, second)
    )
    start_ratio, end_ratio = _clamp_window_ratio_pair(
        horizontal_ratios,
        "horizontal",
    )
    bottom_ratio, top_ratio = _clamp_window_ratio_pair(
        vertical_ratios,
        "vertical",
    )
    if (end_ratio - start_ratio) * wall_width < MIN_WINDOW_SIZE_METERS:
        raise ValueError("A window must be at least 5 cm wide.")
    if (top_ratio - bottom_ratio) * wall_height < MIN_WINDOW_SIZE_METERS:
        raise ValueError("A window must be at least 5 cm tall.")

    placement = WallWindowPlacement(
        wall_surface_id=parent_surface.surface_id,
        start_ratio=start_ratio,
        end_ratio=end_ratio,
        bottom_ratio=bottom_ratio,
        top_ratio=top_ratio,
    )
    _validate_window_patch_coverage(parent_surface, placement)
    return placement


def get_wall_window_world_corners(
    parent_surface: FixedSurface,
    placement: WallWindowPlacement | WindowData,
) -> tuple[tuple[float, float, float], ...]:
    """Resolve one normalized window rectangle into four world corners."""

    wall_start, wall_end, wall_height = _get_window_wall_frame(parent_surface)
    if placement.wall_surface_id != parent_surface.surface_id:
        raise ValueError("A window placement belongs to a different wall.")
    bottom_start = _interpolate_window_world_point(
        wall_start,
        wall_end,
        placement.start_ratio,
        placement.bottom_ratio,
        wall_height,
    )
    bottom_end = _interpolate_window_world_point(
        wall_start,
        wall_end,
        placement.end_ratio,
        placement.bottom_ratio,
        wall_height,
    )
    top_end = _interpolate_window_world_point(
        wall_start,
        wall_end,
        placement.end_ratio,
        placement.top_ratio,
        wall_height,
    )
    top_start = _interpolate_window_world_point(
        wall_start,
        wall_end,
        placement.start_ratio,
        placement.top_ratio,
        wall_height,
    )
    return (bottom_start, bottom_end, top_end, top_start)


def add_wall_window(
    levels: Sequence[LevelData],
    placement: WallWindowPlacement,
    *,
    window_id: str | None = None,
) -> WindowData:
    """Atomically validate and append a stable window to its owning level."""

    if not isinstance(placement, WallWindowPlacement):
        raise TypeError("A wall window commit requires a validated placement.")
    surfaces = {
        surface.surface_id: surface for surface in build_fixed_surfaces(levels)
    }
    parent_surface = surfaces.get(placement.wall_surface_id)
    if parent_surface is None:
        raise ValueError("The selected wall no longer exists.")
    owning_level = next(
        (
            level
            for level in levels
            if level.index == parent_surface.level_index
        ),
        None,
    )
    if owning_level is None:
        raise ValueError("The selected wall has no owning level.")

    corners = get_wall_window_world_corners(parent_surface, placement)
    current_placement = build_wall_window_placement(
        parent_surface,
        corners[0],
        corners[2],
    )
    normalized_id = (
        f"window-{uuid.uuid4().hex}"
        if window_id is None
        else str(window_id).strip()
    )
    if any(
        existing.window_id == normalized_id
        for level in levels
        for existing in level.windows
    ):
        raise ValueError("A window with this stable ID already exists.")
    window = WindowData(
        window_id=normalized_id,
        wall_surface_id=current_placement.wall_surface_id,
        start_ratio=current_placement.start_ratio,
        end_ratio=current_placement.end_ratio,
        bottom_ratio=current_placement.bottom_ratio,
        top_ratio=current_placement.top_ratio,
    )
    owning_level.windows.append(window)
    try:
        updated_surfaces = {
            surface.surface_id: surface
            for surface in build_fixed_surfaces(levels)
        }
        if window.wall_surface_id not in updated_surfaces:
            raise ValueError("The window would remove its complete owning wall.")
    except Exception:
        if owning_level.windows and owning_level.windows[-1] is window:
            owning_level.windows.pop()
        else:
            owning_level.windows.remove(window)
        raise
    return window


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
    doorway_reveals_by_surface_id: Mapping[
        str,
        Sequence[DoorwayReveal],
    ],
    window_reveals_by_surface_id: Mapping[str, Sequence[WindowReveal]],
) -> list[FixedSurface]:
    surfaces: list[FixedSurface] = []
    room_identity = room.center_vertex_id
    wall_openings = _build_level_wall_openings(level)
    room_polygon = _build_room_world_polygon(level, room)
    for wall in build_room_walls(room, level.vertex_data):
        wall_surface = _build_wall_surface(
            level=level,
            start_point=wall.start_point,
            end_point=wall.end_point,
            wall_key=wall.key,
            wall_height_meters=room.height_meters,
            base_z_meters=base_z_meters,
            wall_openings=wall_openings,
            room_index=room_index,
            room_identity=room_identity,
            doorway_reveals_by_surface_id=doorway_reveals_by_surface_id,
            interior_polygon=room_polygon,
            window_reveals_by_surface_id=window_reveals_by_surface_id,
        )
        if wall_surface is not None:
            surfaces.append(wall_surface)

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
    doorway_reveals_by_surface_id: Mapping[
        str,
        Sequence[DoorwayReveal],
    ],
    window_reveals_by_surface_id: Mapping[str, Sequence[WindowReveal]],
) -> list[FixedSurface]:
    room_vertex_sets = [set(room.vertex_ids) for room in level.rooms]
    ignored_vertex_ids = {room.center_vertex_id for room in level.rooms}
    wall_openings = _build_level_wall_openings(level)
    interior_polygon = _build_level_contour_world_polygon(level)
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
            wall_openings=wall_openings,
            room_index=None,
            room_identity=None,
            doorway_reveals_by_surface_id=doorway_reveals_by_surface_id,
            interior_polygon=interior_polygon,
            window_reveals_by_surface_id=window_reveals_by_surface_id,
        )
        if surface is not None:
            surfaces.append(surface)
    return surfaces


def _group_window_reveals_by_surface_id(
    reveals: Sequence[WindowReveal],
) -> dict[str, tuple[WindowReveal, ...]]:
    grouped: dict[str, list[WindowReveal]] = {}
    for reveal in reveals:
        grouped.setdefault(reveal.owner_surface_id, []).append(reveal)
    return {
        surface_id: tuple(surface_reveals)
        for surface_id, surface_reveals in grouped.items()
    }


def _group_doorway_reveals_by_surface_id(
    reveals: Sequence[DoorwayReveal],
) -> dict[str, tuple[DoorwayReveal, ...]]:
    grouped: dict[str, list[DoorwayReveal]] = {}
    for reveal in reveals:
        grouped.setdefault(reveal.owner_surface_id, []).append(reveal)
    return {
        surface_id: tuple(surface_reveals)
        for surface_id, surface_reveals in grouped.items()
    }


# ### Window placement helpers ###
def _get_window_wall_frame(
    surface: FixedSurface,
) -> tuple[np.ndarray, np.ndarray, float]:
    if not isinstance(surface, FixedSurface):
        raise TypeError("A window requires a fixed wall surface.")
    if surface.surface_type != SURFACE_TYPE_WALL:
        raise ValueError("Windows can only be added to wall surfaces.")
    if (
        surface.wall_start_world is None
        or surface.wall_end_world is None
        or surface.wall_height_meters is None
    ):
        raise ValueError("The selected wall has no placement frame.")
    wall_start = np.asarray(surface.wall_start_world, dtype=float)
    wall_end = np.asarray(surface.wall_end_world, dtype=float)
    wall_height = float(surface.wall_height_meters)
    if (
        wall_start.shape != (3,)
        or wall_end.shape != (3,)
        or not np.isfinite(wall_start).all()
        or not np.isfinite(wall_end).all()
        or not math.isfinite(wall_height)
        or wall_height <= SURFACE_GEOMETRY_EPSILON
    ):
        raise ValueError("The selected wall has an invalid placement frame.")
    wall_axis = wall_end - wall_start
    if (
        abs(float(wall_axis[2])) > SURFACE_GEOMETRY_EPSILON
        or float(np.linalg.norm(wall_axis)) <= SURFACE_GEOMETRY_EPSILON
    ):
        raise ValueError("The selected wall has an invalid placement frame.")
    return wall_start, wall_end, wall_height


def _normalize_window_world_point(value: Sequence[float]) -> np.ndarray:
    try:
        point = np.asarray(tuple(value), dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("A window point must contain finite XYZ values.") from error
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("A window point must contain finite XYZ values.")
    return point


def _clamp_window_ratio_pair(
    ratios: Sequence[float],
    axis_name: str,
) -> tuple[float, float]:
    if len(ratios) != 2:
        raise ValueError("A window rectangle requires two points.")
    low_ratio, high_ratio = (float(ratios[0]), float(ratios[1]))
    tolerance = SURFACE_GEOMETRY_EPSILON * 10.0
    if low_ratio < -tolerance or high_ratio > 1.0 + tolerance:
        raise ValueError(
            f"A window must remain inside the wall's {axis_name} bounds."
        )
    return max(0.0, low_ratio), min(1.0, high_ratio)


def _interpolate_window_world_point(
    wall_start: np.ndarray,
    wall_end: np.ndarray,
    horizontal_ratio: float,
    vertical_ratio: float,
    wall_height: float,
) -> tuple[float, float, float]:
    point = wall_start + (wall_end - wall_start) * float(horizontal_ratio)
    point[2] = wall_start[2] + wall_height * float(vertical_ratio)
    return tuple(float(value) for value in point)


def _validate_window_patch_coverage(
    surface: FixedSurface,
    placement: WallWindowPlacement,
) -> None:
    wall_start, wall_end, wall_height = _get_window_wall_frame(surface)
    wall_axis = wall_end - wall_start
    wall_width = float(np.linalg.norm(wall_axis))
    tangent = wall_axis / wall_width
    plane_normal = np.cross(tangent, np.asarray((0.0, 0.0, 1.0)))
    mesh_vertices = np.asarray(surface.mesh.vertices, dtype=float)
    mesh_faces = np.asarray(surface.mesh.faces, dtype=np.int64)
    polygons: list[Polygon] = []
    for face in mesh_faces:
        triangle = mesh_vertices[face]
        distances = np.dot(triangle - wall_start, plane_normal)
        if np.max(np.abs(distances)) > WINDOW_PATCH_COVERAGE_TOLERANCE_METERS:
            continue
        local_points = [
            (
                float(np.dot(vertex - wall_start, tangent)),
                float(vertex[2] - wall_start[2]),
            )
            for vertex in triangle
        ]
        polygon = Polygon(local_points)
        if polygon.is_valid and polygon.area > SURFACE_GEOMETRY_EPSILON:
            polygons.append(polygon)
    if not polygons:
        raise ValueError("The selected wall has no visible window patch.")
    visible_patch = shapely.union_all(polygons)
    candidate = Polygon(
        (
            (
                placement.start_ratio * wall_width,
                placement.bottom_ratio * wall_height,
            ),
            (
                placement.end_ratio * wall_width,
                placement.bottom_ratio * wall_height,
            ),
            (
                placement.end_ratio * wall_width,
                placement.top_ratio * wall_height,
            ),
            (
                placement.start_ratio * wall_width,
                placement.top_ratio * wall_height,
            ),
        )
    )
    if not visible_patch.buffer(
        WINDOW_PATCH_COVERAGE_TOLERANCE_METERS
    ).covers(candidate):
        raise ValueError(
            "A window must fit the visible wall without crossing another opening."
        )


# ### Wall geometry helpers ###
def _build_wall_surface(
    level: LevelData,
    start_point: tuple[float, float],
    end_point: tuple[float, float],
    wall_key: str,
    wall_height_meters: float,
    base_z_meters: float,
    wall_openings: Sequence[WallOpening],
    room_index: int | None,
    room_identity: int | None,
    doorway_reveals_by_surface_id: Mapping[
        str,
        Sequence[DoorwayReveal],
    ],
    interior_polygon: Polygon | None,
    window_reveals_by_surface_id: Mapping[str, Sequence[WindowReveal]],
) -> FixedSurface | None:
    if (
        not math.isfinite(float(wall_height_meters))
        or wall_height_meters <= SURFACE_GEOMETRY_EPSILON
    ):
        return None

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    duplicate_face_indices: list[int] = []
    for wall_piece in _build_visible_wall_pieces(
        start_point=start_point,
        end_point=end_point,
        wall_height_meters=wall_height_meters,
        doorway_openings=wall_openings,
        wall_key=wall_key,
    ):
        corners = tuple(
            (
                *level_image_to_world_xy(
                    level,
                    *_interpolate_2d_point(
                        start_point,
                        end_point,
                        wall_ratio,
                    ),
                ),
                base_z_meters + height_meters,
            )
            for wall_ratio, height_meters in wall_piece.points
        )
        _append_polygon(
            vertices,
            faces,
            corners,
            reverse_winding=(
                _quad_front_points_outside_polygon(
                    corners,
                    interior_polygon,
                )
            ),
        )

    surface_id = build_wall_surface_id(level.index, wall_key, room_identity)
    for doorway_reveal in doorway_reveals_by_surface_id.get(surface_id, ()):
        _append_connected_doorway_reveals(
            vertices=vertices,
            faces=faces,
            level=level,
            base_z_meters=base_z_meters,
            doorway_reveal=doorway_reveal,
        )

    for window_reveal in window_reveals_by_surface_id.get(surface_id, ()):
        _append_connected_window_reveals(
            vertices=vertices,
            faces=faces,
            duplicate_face_indices=duplicate_face_indices,
            level=level,
            base_z_meters=base_z_meters,
            window_reveal=window_reveal,
        )

    mesh = _build_mesh(vertices, faces)
    if mesh is None:
        return None
    duplicate_area = float(
        np.sum(mesh.area_faces[duplicate_face_indices])
        if duplicate_face_indices
        else 0.0
    )
    area = float(mesh.area) - duplicate_area
    if area <= SURFACE_GEOMETRY_EPSILON:
        return None
    return FixedSurface(
        surface_id=surface_id,
        surface_type=SURFACE_TYPE_WALL,
        level_index=level.index,
        room_index=room_index,
        wall_key=wall_key,
        wall_start_world=(*level_image_to_world_xy(level, *start_point), base_z_meters),
        wall_end_world=(*level_image_to_world_xy(level, *end_point), base_z_meters),
        wall_height_meters=wall_height_meters,
        mesh=mesh,
        area_square_meters=area,
    )


def _append_connected_window_reveals(
    vertices: list[list[float]],
    faces: list[list[int]],
    duplicate_face_indices: list[int],
    level: LevelData,
    base_z_meters: float,
    window_reveal: WindowReveal,
) -> None:
    for image_quad in _build_wall_opening_reveal_quads(
        window_reveal.reveal_pair,
        window_reveal.opening,
        include_sill=True,
    ):
        world_quad = tuple(
            (
                *level_image_to_world_xy(
                    level,
                    local_point[0],
                    local_point[1],
                ),
                base_z_meters + local_point[2],
            )
            for local_point in image_quad
        )
        _append_quad(vertices, faces, world_quad)
        duplicate_start = len(faces)
        _append_quad(
            vertices,
            faces,
            world_quad,
            reverse_winding=True,
        )
        duplicate_face_indices.extend(range(duplicate_start, len(faces)))


def _append_connected_doorway_reveals(
    vertices: list[list[float]],
    faces: list[list[int]],
    level: LevelData,
    base_z_meters: float,
    doorway_reveal: DoorwayReveal,
) -> None:
    for image_quad in _build_wall_opening_reveal_quads(
        doorway_reveal.reveal_pair,
        doorway_reveal.opening,
        include_sill=(
            doorway_reveal.opening.bottom_height_meters
            > WALL_OPENING_EPSILON
        ),
    ):
        world_quad = tuple(
            (
                *level_image_to_world_xy(
                    level,
                    local_point[0],
                    local_point[1],
                ),
                base_z_meters + local_point[2],
            )
            for local_point in image_quad
        )
        _append_quad(vertices, faces, world_quad)


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
def _append_polygon(
    vertices: list[list[float]],
    faces: list[list[int]],
    corners: Sequence[tuple[float, float, float]],
    *,
    reverse_winding: bool = False,
) -> None:
    """Append one finite convex polygon using a triangle fan."""

    if len(corners) < 3:
        return
    corner_array = np.asarray(corners, dtype=float)
    if not np.all(np.isfinite(corner_array)):
        return
    vertex_offset = len(vertices)
    forward_faces = tuple(
        [
            vertex_offset,
            vertex_offset + point_index,
            vertex_offset + point_index + 1,
        ]
        for point_index in range(1, len(corners) - 1)
    )
    if not any(
        np.linalg.norm(
            np.cross(
                corner_array[face[1] - vertex_offset] - corner_array[0],
                corner_array[face[2] - vertex_offset] - corner_array[0],
            )
        )
        > SURFACE_GEOMETRY_EPSILON
        for face in forward_faces
    ):
        return
    vertices.extend(corner_array.tolist())
    faces.extend(
        [list(reversed(face)) for face in forward_faces]
        if reverse_winding
        else forward_faces
    )


def _append_quad(
    vertices: list[list[float]],
    faces: list[list[int]],
    corners: Sequence[tuple[float, float, float]],
    *,
    reverse_winding: bool = False,
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
    forward_faces = (
        [vertex_offset, vertex_offset + 1, vertex_offset + 2],
        [vertex_offset, vertex_offset + 2, vertex_offset + 3],
    )
    faces.extend(
        [list(reversed(face)) for face in forward_faces]
        if reverse_winding
        else forward_faces
    )


def _quad_front_points_outside_polygon(
    corners: Sequence[tuple[float, float, float]],
    interior_polygon: Polygon | None,
) -> bool:
    """Return whether one planar quad must flip to face its local interior."""

    if interior_polygon is None or interior_polygon.is_empty or len(corners) < 3:
        return False
    corner_array = np.asarray(corners, dtype=float)
    face_normal = np.cross(
        corner_array[1] - corner_array[0],
        corner_array[2] - corner_array[0],
    )
    normal_xy = face_normal[:2]
    normal_length = float(np.linalg.norm(normal_xy))
    wall_length = max(
        float(
            np.linalg.norm(
                corner_array[second_index, :2]
                - corner_array[first_index, :2]
            )
        )
        for first_index in range(len(corner_array))
        for second_index in range(first_index + 1, len(corner_array))
    )
    if (
        normal_length <= SURFACE_GEOMETRY_EPSILON
        or wall_length <= SURFACE_GEOMETRY_EPSILON
    ):
        return False
    normal_xy /= normal_length
    midpoint_xy = np.mean(corner_array[:, :2], axis=0)
    for probe_ratio in SURFACE_INTERIOR_PROBE_RATIOS:
        probe_distance = max(
            SURFACE_GEOMETRY_EPSILON * 10.0,
            wall_length * probe_ratio,
        )
        front_is_inside = interior_polygon.contains(
            Point(*(midpoint_xy + normal_xy * probe_distance))
        )
        back_is_inside = interior_polygon.contains(
            Point(*(midpoint_xy - normal_xy * probe_distance))
        )
        if front_is_inside != back_is_inside:
            return not front_is_inside
    return False


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
