# ### Imports ###
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import shapely
import trimesh
from shapely import Polygon

from housemaker.models import LevelData

# ### Constants ###
GEOMETRY_EPSILON = 1e-12

# ### Type aliases ###
Point2D = tuple[float, float]
PointToWorld = Callable[
    [tuple[float, float], tuple[float, float] | None],
    np.ndarray,
]


# ### Public helpers ###
def build_level_floor_mesh(
    level: LevelData,
    floor_surface_z_meters: float,
    blueprint_size_pixels: tuple[float, float] | None,
    point_to_world_xy: PointToWorld,
) -> trimesh.Trimesh | None:
    house_footprint = _build_house_footprint(
        level=level,
        blueprint_size_pixels=blueprint_size_pixels,
        point_to_world_xy=point_to_world_xy,
    )
    if house_footprint is None:
        return None

    return _build_floor_prism_mesh(
        house_footprint=house_footprint,
        floor_bottom_z_meters=(
            floor_surface_z_meters - level.floor_thickness_meters
        ),
        floor_surface_z_meters=floor_surface_z_meters,
    )


# ### House contour helpers ###
def _build_house_footprint(
    level: LevelData,
    blueprint_size_pixels: tuple[float, float] | None,
    point_to_world_xy: PointToWorld,
) -> Polygon | None:
    contour_vertex_ids = level.floor_contour_vertex_ids
    if not contour_vertex_ids:
        return None
    if (
        len(contour_vertex_ids) < 3
        or len(set(contour_vertex_ids)) != len(contour_vertex_ids)
    ):
        raise ValueError(
            f"Level {level.index} floor contour must contain at least three "
            "unique vertices."
        )

    vertex_lookup = {
        vertex.id: vertex
        for vertex in level.vertex_data.vertices
    }
    contour_points: list[Point2D] = []
    for vertex_id in contour_vertex_ids:
        vertex = vertex_lookup.get(vertex_id)
        if vertex is None:
            raise ValueError(
                f"Level {level.index} floor contour references missing "
                f"vertex {vertex_id}."
            )

        world_point = point_to_world_xy(
            (vertex.x, vertex.y),
            blueprint_size_pixels,
        )
        contour_point = _to_finite_point(world_point)
        if contour_point is None:
            raise ValueError(
                f"Level {level.index} floor contour contains non-finite "
                "coordinates."
            )
        contour_points.append(contour_point)

    house_footprint = Polygon(contour_points)
    if not house_footprint.is_valid:
        validation_reason = shapely.is_valid_reason(house_footprint)
        raise ValueError(
            f"Level {level.index} floor contour is invalid: "
            f"{validation_reason}."
        )
    if float(house_footprint.area) <= GEOMETRY_EPSILON:
        raise ValueError(
            f"Level {level.index} floor contour must enclose a positive area."
        )

    return house_footprint


def _to_finite_point(point: np.ndarray) -> Point2D | None:
    point_x = float(point[0])
    point_y = float(point[1])
    if not math.isfinite(point_x) or not math.isfinite(point_y):
        return None

    return point_x, point_y


# ### Mesh helpers ###
def _build_floor_prism_mesh(
    house_footprint: Polygon,
    floor_bottom_z_meters: float,
    floor_surface_z_meters: float,
) -> trimesh.Trimesh | None:
    floor_height_meters = floor_surface_z_meters - floor_bottom_z_meters
    if floor_height_meters <= GEOMETRY_EPSILON:
        return None

    vertices: list[Point2D] = []
    vertex_index_by_point: dict[Point2D, int] = {}
    triangle_indices: list[tuple[int, int, int]] = []
    triangulation = shapely.constrained_delaunay_triangles(house_footprint)

    for triangle in shapely.get_parts(triangulation):
        if not isinstance(triangle, Polygon) or triangle.is_empty:
            continue
        if not shapely.covers(
            house_footprint,
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

        triangle_vertex_indices: list[int] = []
        for point in triangle_points:
            if point not in vertex_index_by_point:
                vertex_index_by_point[point] = len(vertices)
                vertices.append(point)
            triangle_vertex_indices.append(vertex_index_by_point[point])

        triangle_indices.append(tuple(triangle_vertex_indices))

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
def _get_triangle_signed_area(triangle_points: list[Point2D]) -> float:
    first_point, second_point, third_point = triangle_points
    return (
        (second_point[0] - first_point[0])
        * (third_point[1] - first_point[1])
        - (second_point[1] - first_point[1])
        * (third_point[0] - first_point[0])
    ) / 2.0
