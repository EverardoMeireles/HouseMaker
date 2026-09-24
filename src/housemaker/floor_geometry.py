# ### Imports ###
from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import shapely
import trimesh
from shapely import LineString, Point, Polygon, STRtree
from shapely.geometry.base import BaseGeometry

from housemaker.models import LevelData

# ### Constants ###
GEOMETRY_EPSILON = 1e-12
EDGE_KEY_DECIMALS = 9
OUTER_ENVELOPE_CLOSING_RADIUS_METERS = 2.0
OUTER_ENVELOPE_MITRE_LIMIT = 100.0
LINE_COVERAGE_TOLERANCE_METERS = 1e-8
LINE_ATTACHMENT_TOLERANCE_METERS = LINE_COVERAGE_TOLERANCE_METERS * 2.0
MINIMUM_RECOVERED_FOOTPRINT_AREA_SQUARE_METERS = 1.0
MINIMUM_RECOVERED_BOUNDARY_SUPPORT_RATIO = 0.8
WALL_GAP_ALIGNMENT_MINIMUM_DOT = math.cos(math.radians(10.0))


# ### Type aliases ###
Point2D = tuple[float, float]
PointToWorld = Callable[
    [tuple[float, float], tuple[float, float] | None],
    np.ndarray,
]
WallEndpoint = tuple[np.ndarray, np.ndarray, int]


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

    open_space_geometry = build_level_open_space_geometry(
        level,
        blueprint_size_pixels,
        point_to_world_xy,
    )
    if open_space_geometry is not None:
        footprint = footprint.difference(open_space_geometry)
        if (
            footprint.is_empty
            or float(footprint.area) <= GEOMETRY_EPSILON
        ):
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


def build_level_open_space_geometry(
    level: LevelData,
    blueprint_size_pixels: tuple[float, float] | None,
    point_to_world_xy: PointToWorld,
) -> BaseGeometry | None:
    """Return the union of one level's rectangular slab openings."""

    polygons: list[Polygon] = []
    for open_space in level.open_spaces:
        image_corners = (
            (open_space.minimum_x, open_space.minimum_y),
            (open_space.maximum_x, open_space.minimum_y),
            (open_space.maximum_x, open_space.maximum_y),
            (open_space.minimum_x, open_space.maximum_y),
        )
        world_corners: list[Point2D] = []
        for image_corner in image_corners:
            normalized_corner = _to_finite_point(
                point_to_world_xy(image_corner, blueprint_size_pixels)
            )
            if normalized_corner is None:
                raise ValueError(
                    f"Level {level.index} open space has an invalid position."
                )
            world_corners.append(normalized_corner)
        polygon = Polygon(world_corners)
        if not polygon.is_valid:
            polygon = shapely.make_valid(polygon)
        polygons.extend(
            part
            for part in shapely.get_parts(polygon)
            if isinstance(part, Polygon)
            and not part.is_empty
            and float(part.area) > GEOMETRY_EPSILON
        )

    if not polygons:
        return None
    geometry = shapely.union_all(polygons)
    if geometry.is_empty or float(geometry.area) <= GEOMETRY_EPSILON:
        return None
    return geometry


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
    """Return every recoverable shell without requiring every wall to belong."""

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

    cluster_footprints: list[Polygon] = []
    for cluster_linework in _build_wall_linework_clusters(
        structural_lines,
        maximum_gap=closing_radius * 2.0,
    ):
        cluster_footprint = _build_linework_footprint(
            cluster_linework,
            closing_radius,
        )
        if cluster_footprint is None:
            continue
        cluster_footprints.extend(
            part
            for part in shapely.get_parts(cluster_footprint)
            if isinstance(part, Polygon)
            and not part.is_empty
            and float(part.area) > GEOMETRY_EPSILON
        )
    if not cluster_footprints:
        return None
    return _fill_polygon_exteriors(shapely.union_all(cluster_footprints))


def _build_linework_footprint(
    structural_linework: BaseGeometry,
    closing_radius: float,
) -> BaseGeometry | None:
    """Recover valid shells from one related wall-line cluster."""

    exact_footprint = _build_exact_closed_footprint(structural_linework)
    recovery_linework = structural_linework
    if exact_footprint is not None:
        unresolved_linework = structural_linework.difference(
            shapely.buffer(
                exact_footprint,
                LINE_COVERAGE_TOLERANCE_METERS,
            )
        )
        unresolved_lines = tuple(
            part
            for part in shapely.get_parts(unresolved_linework)
            if isinstance(part, LineString) and not part.is_empty
        )
        if not unresolved_lines:
            return exact_footprint
        retained_components = tuple(
            component
            for component in _build_touching_linework_components(
                unresolved_lines
            )
            if _count_exact_footprint_attachments(
                component,
                exact_footprint,
            )
            != 1
        )
        if not retained_components:
            return exact_footprint
        recovery_linework = shapely.union_all(
            (exact_footprint.boundary, *retained_components)
        )

    expanded_linework = shapely.buffer(
        recovery_linework,
        closing_radius,
        cap_style="square",
        join_style="mitre",
        mitre_limit=OUTER_ENVELOPE_MITRE_LIMIT,
    )
    filled_envelope = _fill_polygon_exteriors(expanded_linework)
    if filled_envelope is None:
        return _build_recoverable_footprint(
            exact_footprint,
            None,
            recovery_linework,
        )

    footprint = shapely.buffer(
        filled_envelope,
        -closing_radius,
        join_style="mitre",
        mitre_limit=OUTER_ENVELOPE_MITRE_LIMIT,
    )
    if (
        footprint.is_empty
        or float(footprint.area) <= GEOMETRY_EPSILON
    ):
        footprint = None
    return _build_recoverable_footprint(
        exact_footprint,
        footprint,
        recovery_linework,
    )


def _count_exact_footprint_attachments(
    linework: BaseGeometry,
    exact_footprint: BaseGeometry,
) -> int:
    """Count distinct line endpoints attached to an exact footprint shell."""

    attachment_points = {
        (
            round(float(point.x), EDGE_KEY_DECIMALS),
            round(float(point.y), EDGE_KEY_DECIMALS),
        )
        for point in shapely.get_parts(linework.boundary)
        if isinstance(point, Point)
        and point.distance(exact_footprint)
        <= LINE_ATTACHMENT_TOLERANCE_METERS
    }
    return len(attachment_points)


def _build_wall_linework_clusters(
    structural_lines: Sequence[LineString],
    *,
    maximum_gap: float,
) -> tuple[BaseGeometry, ...]:
    """Group connected walls without merging nearby detached buildings."""

    linework = shapely.union_all(structural_lines)
    closure_lines = _build_straight_gap_closures(linework, maximum_gap)
    return _build_touching_linework_components(
        structural_lines,
        connector_lines=closure_lines,
    )


def _build_touching_linework_components(
    lines: Sequence[LineString],
    *,
    connector_lines: Sequence[LineString] = (),
) -> tuple[BaseGeometry, ...]:
    """Dissolve source lines connected directly or by approved gap lines."""

    if not lines:
        return ()
    line_tree = STRtree(lines)
    parent_indices = list(range(len(lines)))
    for first_index, line in enumerate(lines):
        nearby_indices = line_tree.query(
            line,
            predicate="dwithin",
            distance=LINE_COVERAGE_TOLERANCE_METERS,
        )
        for second_index in (
            int(index) for index in nearby_indices if int(index) < first_index
        ):
            _join_group_indices(parent_indices, first_index, second_index)

    closed_group_roots: set[int] = set()
    if connector_lines:
        directly_grouped_lines = _group_lines_by_root(lines, parent_indices)
        closed_group_roots = {
            root_index
            for root_index, component_lines in directly_grouped_lines.items()
            if _build_exact_closed_footprint(
                shapely.union_all(component_lines)
            )
            is not None
        }
    for connector_line in connector_lines:
        connector_coordinates = tuple(connector_line.coords)
        if len(connector_coordinates) < 2:
            continue
        start_indices = tuple(
            int(index)
            for index in line_tree.query(
                Point(connector_coordinates[0]),
                predicate="dwithin",
                distance=LINE_COVERAGE_TOLERANCE_METERS,
            )
        )
        end_indices = tuple(
            int(index)
            for index in line_tree.query(
                Point(connector_coordinates[-1]),
                predicate="dwithin",
                distance=LINE_COVERAGE_TOLERANCE_METERS,
            )
        )
        if not start_indices or not end_indices:
            continue
        start_roots = {
            _find_group_root(parent_indices, index) for index in start_indices
        }
        end_roots = {
            _find_group_root(parent_indices, index) for index in end_indices
        }
        if (
            start_roots.isdisjoint(end_roots)
            and start_roots & closed_group_roots
            and end_roots & closed_group_roots
        ):
            continue

        connector_indices = (*start_indices, *end_indices)
        connector_roots = start_roots | end_roots
        merged_group_is_closed = bool(connector_roots & closed_group_roots)
        first_index = connector_indices[0]
        for second_index in connector_indices[1:]:
            _join_group_indices(parent_indices, first_index, second_index)
        closed_group_roots.difference_update(connector_roots)
        if merged_group_is_closed:
            closed_group_roots.add(
                _find_group_root(parent_indices, first_index)
            )

    grouped_lines = _group_lines_by_root(lines, parent_indices)
    return tuple(
        shapely.union_all(component_lines)
        for _root_index, component_lines in sorted(grouped_lines.items())
    )


def _group_lines_by_root(
    lines: Sequence[LineString],
    parent_indices: list[int],
) -> dict[int, list[LineString]]:
    """Collect source lines by their current union-find root."""

    grouped_lines: dict[int, list[LineString]] = {}
    for line_index, line in enumerate(lines):
        root_index = _find_group_root(parent_indices, line_index)
        grouped_lines.setdefault(root_index, []).append(line)
    return grouped_lines


def _build_straight_gap_closures(
    linework: BaseGeometry,
    maximum_gap: float,
) -> tuple[LineString, ...]:
    """Pair nearby dangling endpoints whose wall tangents face each other."""

    endpoints = _get_dangling_endpoints(linework)
    if len(endpoints) < 2:
        return ()
    endpoint_points = tuple(
        Point(*point) for point, _direction, _chain in endpoints
    )
    endpoint_tree = STRtree(endpoint_points)
    candidates: list[tuple[int, float, Point2D, Point2D, int, int]] = []
    for first_index, (
        first_point,
        first_direction,
        first_chain_index,
    ) in enumerate(endpoints):
        nearby_indices = endpoint_tree.query(
            endpoint_points[first_index],
            predicate="dwithin",
            distance=maximum_gap,
        )
        for second_index in (
            int(index) for index in nearby_indices if int(index) > first_index
        ):
            second_point, second_direction, second_chain_index = endpoints[
                second_index
            ]
            gap_delta = second_point - first_point
            gap_length = float(np.linalg.norm(gap_delta))
            if gap_length <= GEOMETRY_EPSILON or gap_length > maximum_gap:
                continue
            gap_direction = gap_delta / gap_length
            if (
                float(np.dot(first_direction, gap_direction))
                < WALL_GAP_ALIGNMENT_MINIMUM_DOT
                or float(np.dot(second_direction, -gap_direction))
                < WALL_GAP_ALIGNMENT_MINIMUM_DOT
            ):
                continue
            candidates.append(
                (
                    int(first_chain_index != second_chain_index),
                    gap_length,
                    _array_to_point(first_point),
                    _array_to_point(second_point),
                    first_index,
                    second_index,
                )
            )

    matched_indices: set[int] = set()
    closures: list[LineString] = []
    for (
        _different_chain,
        _gap_length,
        first_point,
        second_point,
        first_index,
        second_index,
    ) in sorted(candidates):
        if first_index in matched_indices or second_index in matched_indices:
            continue
        matched_indices.update((first_index, second_index))
        closures.append(LineString((first_point, second_point)))
    return tuple(closures)


def _get_dangling_endpoints(
    linework: BaseGeometry,
) -> tuple[WallEndpoint, ...]:
    """Return degree-one endpoints and their outward continuation vectors."""

    occurrences_by_point: dict[Point2D, list[WallEndpoint]] = {}
    merged_linework = shapely.line_merge(linework)
    for chain_index, line in enumerate(shapely.get_parts(merged_linework)):
        if not isinstance(line, LineString):
            continue
        coordinates = np.asarray(line.coords, dtype=float)
        if len(coordinates) < 2:
            continue
        for endpoint_index, neighbor_index in ((0, 1), (-1, -2)):
            point = coordinates[endpoint_index]
            outward = point - coordinates[neighbor_index]
            outward_length = float(np.linalg.norm(outward))
            if outward_length <= GEOMETRY_EPSILON:
                continue
            endpoint = (point, outward / outward_length, chain_index)
            occurrences_by_point.setdefault(
                _array_to_point(point),
                [],
            ).append(endpoint)
    return tuple(
        occurrences[0]
        for _point, occurrences in sorted(occurrences_by_point.items())
        if len(occurrences) == 1
    )


def _find_group_root(parent_indices: list[int], item_index: int) -> int:
    """Return and path-compress one union-find root."""

    while parent_indices[item_index] != item_index:
        parent_indices[item_index] = parent_indices[parent_indices[item_index]]
        item_index = parent_indices[item_index]
    return item_index


def _join_group_indices(
    parent_indices: list[int],
    first_index: int,
    second_index: int,
) -> None:
    """Join two union-find groups."""

    first_root = _find_group_root(parent_indices, first_index)
    second_root = _find_group_root(parent_indices, second_index)
    if first_root != second_root:
        parent_indices[second_root] = first_root


def _build_recoverable_footprint(
    exact_footprint: BaseGeometry | None,
    gap_closed_footprint: BaseGeometry | None,
    structural_linework: BaseGeometry,
) -> BaseGeometry | None:
    """Combine independently valid shells and ignore unresolved wall lines."""

    candidates: list[Polygon] = []
    if gap_closed_footprint is not None:
        candidates.extend(
            part
            for part in shapely.get_parts(gap_closed_footprint)
            if _is_supported_recovered_footprint(part, structural_linework)
        )
    if exact_footprint is not None:
        candidates.extend(
            part
            for part in shapely.get_parts(exact_footprint)
            if isinstance(part, Polygon)
            and not part.is_empty
            and float(part.area) > GEOMETRY_EPSILON
        )
    if not candidates:
        return None
    return _fill_polygon_exteriors(shapely.union_all(candidates))


def _is_supported_recovered_footprint(
    candidate: BaseGeometry,
    structural_linework: BaseGeometry,
) -> bool:
    """Reject broad buffer bridges whose boundary is mostly unsupported."""

    if (
        not isinstance(candidate, Polygon)
        or candidate.is_empty
        or float(candidate.area) + GEOMETRY_EPSILON
        < MINIMUM_RECOVERED_FOOTPRINT_AREA_SQUARE_METERS
    ):
        return False
    boundary_length = float(candidate.boundary.length)
    if boundary_length <= GEOMETRY_EPSILON:
        return False
    supported_boundary = candidate.boundary.intersection(
        shapely.buffer(
            structural_linework,
            LINE_COVERAGE_TOLERANCE_METERS,
        )
    )
    support_ratio = float(supported_boundary.length) / boundary_length
    return support_ratio >= MINIMUM_RECOVERED_BOUNDARY_SUPPORT_RATIO


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
    """Extrude each footprint island without welding point-touching parts."""

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
    part_meshes: list[trimesh.Trimesh] = []
    for footprint_part in footprint_parts:
        part_mesh = _build_floor_footprint_part_mesh(
            footprint_part,
            floor_bottom_z_meters,
            floor_height_meters,
        )
        if part_mesh is not None:
            part_meshes.append(part_mesh)
    if not part_meshes:
        return None
    if len(part_meshes) == 1:
        return part_meshes[0]
    return trimesh.util.concatenate(part_meshes)


def _build_floor_footprint_part_mesh(
    footprint_part: Polygon,
    floor_bottom_z_meters: float,
    floor_height_meters: float,
) -> trimesh.Trimesh | None:
    """Triangulate and extrude one connected floor polygon."""

    cleaned_footprint = shapely.remove_repeated_points(
        footprint_part,
        tolerance=LINE_COVERAGE_TOLERANCE_METERS,
    )
    if not isinstance(cleaned_footprint, Polygon):
        return None
    footprint_part = cleaned_footprint

    vertices: list[Point2D] = []
    vertex_index_by_point: dict[Point2D, int] = {}
    triangle_indices: list[tuple[int, int, int]] = []
    triangle_keys: set[tuple[int, int, int]] = set()
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


def _array_to_point(point: np.ndarray) -> Point2D:
    """Convert one known finite two-dimensional array into a stable key."""

    return float(point[0]), float(point[1])


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
