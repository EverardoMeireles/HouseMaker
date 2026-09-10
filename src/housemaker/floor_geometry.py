# ### Imports ###
from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import shapely
import trimesh
from shapely import LineString, Polygon
from shapely.geometry.base import BaseGeometry

from housemaker.models import LevelData


# ### Constants ###
GEOMETRY_EPSILON = 1e-12
EDGE_KEY_DECIMALS = 9
OUTER_ENVELOPE_CLOSING_RADIUS_METERS = 2.0
OUTER_ENVELOPE_MITRE_LIMIT = 100.0
LINE_COVERAGE_TOLERANCE_METERS = 1e-8


# ### Type aliases ###
Point2D = tuple[float, float]
PointToWorld = Callable[
    [tuple[float, float], tuple[float, float] | None],
    np.ndarray,
]


# ### Public floor builders ###
def build_level_floor_mesh(
    level: LevelData,
    floor_base_z_meters: float,
    blueprint_size_pixels: tuple[float, float] | None,
    point_to_world_xy: PointToWorld,
) -> trimesh.Trimesh | None:
    """Build one upward slab from the level's anchored structural base."""

    footprint = build_level_floor_footprint(
        level,
        blueprint_size_pixels,
        point_to_world_xy,
    )
    if footprint is None:
        return None

    thickness_meters = _get_valid_floor_thickness(level)
    return _build_floor_prism_mesh(
        house_footprint=footprint,
        floor_bottom_z_meters=floor_base_z_meters,
        floor_surface_z_meters=floor_base_z_meters + thickness_meters,
    )


def build_level_floor_footprint(
    level: LevelData,
    blueprint_size_pixels: tuple[float, float] | None,
    point_to_world_xy: PointToWorld,
    *,
    closing_radius_meters: float = OUTER_ENVELOPE_CLOSING_RADIUS_METERS,
) -> BaseGeometry | None:
    """Return the filled outer envelope of the level's structural walls."""

    structural_lines = _build_structural_lines(
        level,
        blueprint_size_pixels,
        point_to_world_xy,
    )
    if not structural_lines:
        return None
    return _build_outer_wall_envelope(
        structural_lines,
        closing_radius_meters,
    )


# ### Structural footprint helpers ###
def _build_structural_lines(
    level: LevelData,
    blueprint_size_pixels: tuple[float, float] | None,
    point_to_world_xy: PointToWorld,
) -> tuple[LineString, ...]:
    """Build unique, non-degenerate structural wall centerlines."""

    ignored_vertex_ids = {room.center_vertex_id for room in level.rooms}
    line_by_edge_key: dict[tuple[Point2D, Point2D], LineString] = {}
    for edge in level.vertex_data.edges:
        if (
            edge.start_vertex_id in ignored_vertex_ids
            or edge.end_vertex_id in ignored_vertex_ids
        ):
            continue
        start_vertex = level.vertex_data.get_vertex(edge.start_vertex_id)
        end_vertex = level.vertex_data.get_vertex(edge.end_vertex_id)
        if start_vertex is None or end_vertex is None:
            continue

        start_point = _convert_vertex_point(
            level,
            start_vertex.id,
            (start_vertex.x, start_vertex.y),
            blueprint_size_pixels,
            point_to_world_xy,
        )
        end_point = _convert_vertex_point(
            level,
            end_vertex.id,
            (end_vertex.x, end_vertex.y),
            blueprint_size_pixels,
            point_to_world_xy,
        )
        if _get_squared_distance(start_point, end_point) <= GEOMETRY_EPSILON:
            continue
        edge_key = _build_undirected_edge_key(start_point, end_point)
        line_by_edge_key.setdefault(
            edge_key,
            LineString((start_point, end_point)),
        )

    return tuple(line_by_edge_key.values())


def _build_outer_wall_envelope(
    structural_lines: Sequence[LineString],
    closing_radius_meters: float,
) -> BaseGeometry | None:
    """Close ordinary wall openings and retain only filled exterior shells."""

    closing_radius = _get_valid_closing_radius(closing_radius_meters)
    noded_linework = shapely.union_all(structural_lines)
    exact_footprint = _build_exact_closed_footprint(noded_linework)
    if (
        exact_footprint is not None
        and shapely.buffer(
            exact_footprint,
            LINE_COVERAGE_TOLERANCE_METERS,
        ).covers(noded_linework)
    ):
        return exact_footprint

    expanded_linework = shapely.buffer(
        noded_linework,
        closing_radius,
        cap_style="square",
        join_style="mitre",
        mitre_limit=OUTER_ENVELOPE_MITRE_LIMIT,
    )
    filled_envelope = _fill_polygon_exteriors(expanded_linework)
    if filled_envelope is None:
        return None

    footprint = shapely.buffer(
        filled_envelope,
        -closing_radius,
        join_style="mitre",
        mitre_limit=OUTER_ENVELOPE_MITRE_LIMIT,
    )
    if (
        footprint.is_empty
        or float(footprint.area) <= GEOMETRY_EPSILON
        or not shapely.buffer(
            footprint,
            LINE_COVERAGE_TOLERANCE_METERS,
        ).covers(noded_linework)
    ):
        return None
    return footprint


def _build_exact_closed_footprint(
    noded_linework: BaseGeometry,
) -> BaseGeometry | None:
    line_parts = tuple(shapely.get_parts(noded_linework))
    if not line_parts:
        return None
    closed_regions = tuple(
        part
        for part in shapely.get_parts(shapely.polygonize(line_parts))
        if isinstance(part, Polygon)
        and not part.is_empty
        and float(part.area) > GEOMETRY_EPSILON
    )
    if not closed_regions:
        return None
    return _fill_polygon_exteriors(shapely.union_all(closed_regions))


def _fill_polygon_exteriors(
    geometry: BaseGeometry,
) -> BaseGeometry | None:
    exterior_polygons = tuple(
        Polygon(part.exterior)
        for part in shapely.get_parts(geometry)
        if isinstance(part, Polygon)
        and not part.is_empty
        and float(part.area) > GEOMETRY_EPSILON
    )
    if not exterior_polygons:
        return None
    filled_geometry = shapely.union_all(exterior_polygons)
    if (
        filled_geometry.is_empty
        or float(filled_geometry.area) <= GEOMETRY_EPSILON
    ):
        return None
    return filled_geometry


def _convert_vertex_point(
    level: LevelData,
    vertex_id: int,
    image_point: tuple[float, float],
    blueprint_size_pixels: tuple[float, float] | None,
    point_to_world_xy: PointToWorld,
) -> Point2D:
    world_point = point_to_world_xy(image_point, blueprint_size_pixels)
    normalized_point = _to_finite_point(world_point)
    if normalized_point is None:
        raise ValueError(
            f"Level {level.index} floor edge vertex {vertex_id} has an "
            "invalid position."
        )
    return normalized_point


# ### Mesh helpers ###
def _build_floor_prism_mesh(
    house_footprint: BaseGeometry,
    floor_bottom_z_meters: float,
    floor_surface_z_meters: float,
) -> trimesh.Trimesh | None:
    """Triangulate and extrude only the dissolved footprint boundary."""

    floor_height_meters = floor_surface_z_meters - floor_bottom_z_meters
    if (
        not math.isfinite(floor_height_meters)
        or floor_height_meters <= GEOMETRY_EPSILON
    ):
        return None

    footprint_parts = tuple(
        part
        for part in shapely.get_parts(house_footprint)
        if isinstance(part, Polygon)
    )
    vertices: list[Point2D] = []
    vertex_index_by_point: dict[Point2D, int] = {}
    triangle_indices: list[tuple[int, int, int]] = []
    triangle_keys: set[tuple[int, int, int]] = set()
    for footprint_part in footprint_parts:
        triangulation = shapely.constrained_delaunay_triangles(footprint_part)
        for triangle in shapely.get_parts(triangulation):
            if not isinstance(triangle, Polygon) or triangle.is_empty:
                continue
            if not shapely.covers(
                footprint_part,
                triangle.representative_point(),
            ):
                continue
            triangle_points = [
                (float(point_x), float(point_y))
                for point_x, point_y in list(triangle.exterior.coords)[:-1]
            ]
            if len(triangle_points) != 3:
                continue
            if _get_triangle_signed_area(triangle_points) < 0.0:
                triangle_points.reverse()

            indices: list[int] = []
            for point in triangle_points:
                if point not in vertex_index_by_point:
                    vertex_index_by_point[point] = len(vertices)
                    vertices.append(point)
                indices.append(vertex_index_by_point[point])
            triangle_key = tuple(sorted(indices))
            if triangle_key in triangle_keys:
                continue
            triangle_keys.add(triangle_key)
            triangle_indices.append(tuple(indices))

    if not triangle_indices:
        return None

    transform = np.eye(4, dtype=float)
    transform[2, 3] = floor_bottom_z_meters
    return trimesh.creation.extrude_triangulation(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(triangle_indices, dtype=np.int64),
        height=floor_height_meters,
        transform=transform,
    )


# ### Numeric helpers ###
def _get_valid_closing_radius(closing_radius_meters: float) -> float:
    try:
        closing_radius = float(closing_radius_meters)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Floor closing radius must be greater than zero.") from error
    if not math.isfinite(closing_radius) or closing_radius <= 0.0:
        raise ValueError("Floor closing radius must be greater than zero.")
    return closing_radius


def _get_valid_floor_thickness(level: LevelData) -> float:
    raw_thickness = level.floor_thickness_meters
    if isinstance(raw_thickness, bool):
        raise ValueError(
            f"Level {level.index} floor thickness must be greater than zero."
        )
    try:
        thickness_meters = float(raw_thickness)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"Level {level.index} floor thickness must be greater than zero."
        ) from error
    if not math.isfinite(thickness_meters) or thickness_meters <= 0.0:
        raise ValueError(
            f"Level {level.index} floor thickness must be greater than zero."
        )
    return thickness_meters


def _to_finite_point(point: object) -> Point2D | None:
    try:
        point_array = np.asarray(point, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return None
    if point_array.shape != (2,) or not np.all(np.isfinite(point_array)):
        return None
    return float(point_array[0]), float(point_array[1])


def _get_squared_distance(first: Point2D, second: Point2D) -> float:
    return (second[0] - first[0]) ** 2 + (second[1] - first[1]) ** 2


def _build_undirected_edge_key(
    first: Point2D,
    second: Point2D,
) -> tuple[Point2D, Point2D]:
    rounded_points = tuple(
        (
            round(point[0], EDGE_KEY_DECIMALS),
            round(point[1], EDGE_KEY_DECIMALS),
        )
        for point in (first, second)
    )
    return tuple(sorted(rounded_points))


def _get_triangle_signed_area(triangle_points: Sequence[Point2D]) -> float:
    first_point, second_point, third_point = triangle_points
    return (
        (second_point[0] - first_point[0])
        * (third_point[1] - first_point[1])
        - (second_point[1] - first_point[1])
        * (third_point[0] - first_point[0])
    ) / 2.0
