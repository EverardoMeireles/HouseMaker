# ### Imports ###
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely import LineString, Point, Polygon, STRtree
from shapely.geometry.base import BaseGeometry

from housemaker.floor_geometry import (
    OUTER_ENVELOPE_CLOSING_RADIUS_METERS,
    _build_linework_footprint,
    _build_straight_gap_closures,
    build_level_floor_footprint,
    build_level_open_space_geometry,
)
from housemaker.level_coordinates import (
    _get_valid_level_scale,
    level_image_to_world_xy,
)
from housemaker.models import Edge, LevelData

# ### Constants ###
WALL_ORIENTATION_EPSILON = 1e-8
WALL_SIDE_PROBE_RATIOS = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3)
WALL_PAIR_ALIGNMENT_MINIMUM_DOT = math.cos(math.radians(5.0))
WALL_PAIR_MAXIMUM_DISTANCE_METERS = 0.6
WALL_PAIR_MINIMUM_OVERLAP_RATIO = 0.5
WALL_PAIR_DISTANCE_GROUP_TOLERANCE_METERS = 0.005
WALL_PAIR_SPAN_GAP_TOLERANCE_METERS = 0.01
OPEN_SPACE_CELL_COVERAGE_RATIO = 0.9

# ### Type aliases ###
Point2D = tuple[float, float]
LevelPointTransform = Callable[[float, float], Point2D]
WorldDirectionToImage = Callable[[Point2D], Point2D | None]
WallSideCandidate = tuple[np.ndarray, Polygon]
WallFrame = tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]
ParallelWallSpan = tuple[float, Point2D, float, float]


# ### Orientation model ###
@dataclass(frozen=True)
class WallOrientationResolver:
    """Resolve stable exposed-side normals for an undirected wall graph."""

    bounded_cells: tuple[Polygon, ...]
    floor_footprint: BaseGeometry | None
    open_space_geometry: BaseGeometry | None
    _orientation_footprint: BaseGeometry | None = field(
        repr=False,
        compare=False,
    )
    _cell_tree: STRtree | None = field(repr=False, compare=False)
    _wall_lines: tuple[LineString, ...] = field(repr=False, compare=False)
    _wall_tree: STRtree | None = field(repr=False, compare=False)
    _wall_ray_length: float = field(repr=False, compare=False)
    _paired_wall_maximum_distance: float = field(repr=False, compare=False)
    _paired_wall_distance_group_tolerance: float = field(
        repr=False,
        compare=False,
    )
    _paired_wall_span_gap_tolerance: float = field(
        repr=False,
        compare=False,
    )
    _point_to_world: LevelPointTransform = field(repr=False, compare=False)
    _world_direction_to_image: WorldDirectionToImage = field(
        repr=False,
        compare=False,
    )

    def resolve_image_wall_facing_normal(
        self,
        start_point: Point2D,
        end_point: Point2D,
    ) -> Point2D | None:
        """Resolve a wall normal from image points and return it in world XY."""

        return self.resolve_facing_normal(
            self._point_to_world(*start_point),
            self._point_to_world(*end_point),
        )

    def resolve_image_paired_wall_facing_normal(
        self,
        start_point: Point2D,
        end_point: Point2D,
    ) -> Point2D | None:
        """Resolve only a confident paired-plane orientation correction."""

        return self.resolve_paired_wall_facing_normal(
            self._point_to_world(*start_point),
            self._point_to_world(*end_point),
        )

    def resolve_image_wall_exterior_direction(
        self,
        start_point: Point2D,
        end_point: Point2D,
    ) -> Point2D | None:
        """Return the image-space direction behind a wall's visible front."""

        facing_normal = self.resolve_image_wall_facing_normal(
            start_point,
            end_point,
        )
        if facing_normal is None:
            return None
        return self._world_direction_to_image((-facing_normal[0], -facing_normal[1]))

    def resolve_image_paired_wall_exterior_direction(
        self,
        start_point: Point2D,
        end_point: Point2D,
    ) -> Point2D | None:
        """Return image-space depth direction behind a paired wall's front."""

        facing_normal = self.resolve_image_paired_wall_facing_normal(
            start_point,
            end_point,
        )
        if facing_normal is None:
            return None
        return self._world_direction_to_image((-facing_normal[0], -facing_normal[1]))

    def resolve_facing_normal(
        self,
        start_point: Point2D,
        end_point: Point2D,
    ) -> Point2D | None:
        """Point a wall away from sealed rings and into exposed space."""

        wall_frame = _build_wall_frame(start_point, end_point)
        if wall_frame is None:
            return None
        start, end, wall_delta, wall_length, wall_normal = wall_frame
        midpoint = (start + end) / 2.0
        candidates = self._build_side_candidates(
            midpoint,
            wall_normal,
            wall_length,
        )
        topology_normal: Point2D | None
        if len(candidates) == 1:
            side_normal, polygon = candidates[0]
            if not _cell_is_exposed(polygon, self.open_space_geometry):
                side_normal = -side_normal
            topology_normal = _normal_to_tuple(side_normal)
        elif candidates:
            side_normal, _polygon = max(
                candidates,
                key=lambda candidate: _get_side_score(
                    candidate,
                    self.open_space_geometry,
                ),
            )
            topology_normal = _normal_to_tuple(side_normal)
        else:
            topology_normal = _resolve_floor_footprint_normal(
                midpoint,
                wall_normal,
                wall_length,
                self._orientation_footprint,
            )

        paired_wall_normal = self._resolve_paired_wall_normal(
            midpoint,
            wall_delta / wall_length,
            wall_normal,
            wall_length,
        )
        if paired_wall_normal is not None and _parallel_postprocess_can_override(
            candidates,
            self.open_space_geometry,
        ):
            return paired_wall_normal
        return topology_normal

    def resolve_paired_wall_facing_normal(
        self,
        start_point: Point2D,
        end_point: Point2D,
    ) -> Point2D | None:
        """Return a paired-plane correction without a topology fallback."""

        wall_frame = _build_wall_frame(start_point, end_point)
        if wall_frame is None:
            return None
        start, end, wall_delta, wall_length, wall_normal = wall_frame
        midpoint = (start + end) / 2.0
        candidates = self._build_side_candidates(
            midpoint,
            wall_normal,
            wall_length,
        )
        if not _parallel_postprocess_can_override(
            candidates,
            self.open_space_geometry,
        ):
            return None
        return self._resolve_paired_wall_normal(
            midpoint,
            wall_delta / wall_length,
            wall_normal,
            wall_length,
        )

    def _build_side_candidates(
        self,
        midpoint: np.ndarray,
        wall_normal: np.ndarray,
        wall_length: float,
    ) -> tuple[WallSideCandidate, ...]:
        return tuple(
            candidate
            for side_normal in (wall_normal, -wall_normal)
            if (
                candidate := self._build_side_candidate(
                    midpoint,
                    side_normal,
                    wall_length,
                )
            )
            is not None
        )

    def _resolve_paired_wall_normal(
        self,
        midpoint: np.ndarray,
        wall_direction: np.ndarray,
        wall_normal: np.ndarray,
        wall_length: float,
    ) -> Point2D | None:
        """Face away from a nearby parallel wall plane toward greater clearance."""

        side_distances = self._find_parallel_wall_distances(
            midpoint,
            wall_direction,
            wall_normal,
            wall_length,
        )
        finite_distances = tuple(
            distance for distance in side_distances if math.isfinite(distance)
        )
        if (
            not finite_distances
            or min(finite_distances) > self._paired_wall_maximum_distance
        ):
            return None
        negative_distance, positive_distance = side_distances
        distance_tolerance = max(
            WALL_ORIENTATION_EPSILON * 10.0,
            min(finite_distances) * 1e-7,
        )
        if math.isclose(
            negative_distance,
            positive_distance,
            rel_tol=0.0,
            abs_tol=distance_tolerance,
        ):
            return None
        if positive_distance > negative_distance:
            return _normal_to_tuple(wall_normal)
        return _normal_to_tuple(-wall_normal)

    def _find_parallel_wall_distances(
        self,
        midpoint: np.ndarray,
        wall_direction: np.ndarray,
        wall_normal: np.ndarray,
        wall_length: float,
    ) -> tuple[float, float]:
        """Find the first directly opposing parallel plane on either side."""

        if self._wall_tree is None:
            return math.inf, math.inf
        half_wall = wall_direction * (wall_length / 2.0)
        ray_extent = wall_normal * self._wall_ray_length
        search_region = Polygon(
            (
                _normal_to_tuple(midpoint - half_wall - ray_extent),
                _normal_to_tuple(midpoint + half_wall - ray_extent),
                _normal_to_tuple(midpoint + half_wall + ray_extent),
                _normal_to_tuple(midpoint - half_wall + ray_extent),
            )
        )
        spans: list[ParallelWallSpan] = []
        for raw_index in self._wall_tree.query(search_region):
            candidate = self._wall_lines[int(raw_index)]
            coordinates = np.asarray(candidate.coords, dtype=float)
            if len(coordinates) < 2:
                continue
            candidate_start = coordinates[0]
            candidate_end = coordinates[-1]
            candidate_delta = candidate_end - candidate_start
            candidate_length = float(np.linalg.norm(candidate_delta))
            if candidate_length <= WALL_ORIENTATION_EPSILON:
                continue
            candidate_direction = candidate_delta / candidate_length
            if (
                abs(float(np.dot(wall_direction, candidate_direction)))
                < WALL_PAIR_ALIGNMENT_MINIMUM_DOT
            ):
                continue
            span = _build_parallel_wall_span(
                midpoint,
                wall_direction,
                wall_normal,
                wall_length,
                candidate_start,
                candidate_end,
            )
            if span is not None:
                spans.append(span)
        return _resolve_grouped_parallel_wall_distances(
            spans,
            wall_length,
            distance_tolerance=self._paired_wall_distance_group_tolerance,
            gap_tolerance=self._paired_wall_span_gap_tolerance,
        )

    def _build_side_candidate(
        self,
        midpoint: np.ndarray,
        side_normal: np.ndarray,
        wall_length: float,
    ) -> WallSideCandidate | None:
        polygon = self._find_adjacent_polygon(
            midpoint,
            side_normal,
            wall_length,
        )
        if polygon is None:
            return None
        return side_normal, polygon

    def _find_adjacent_polygon(
        self,
        midpoint: np.ndarray,
        side_normal: np.ndarray,
        wall_length: float,
    ) -> Polygon | None:
        """Return the exact bounded cell immediately beside one wall side."""

        if self._cell_tree is None:
            return None
        for probe_ratio in WALL_SIDE_PROBE_RATIOS:
            probe = Point(
                *(
                    midpoint
                    + side_normal * _get_probe_distance(wall_length, probe_ratio)
                )
            )
            containing_cells = tuple(
                self.bounded_cells[int(index)]
                for index in self._cell_tree.query(probe)
                if self.bounded_cells[int(index)].contains(probe)
            )
            if containing_cells:
                return max(containing_cells, key=_get_cell_score)
        return None


# ### Resolver builders ###
def build_level_wall_orientation_resolver(
    level: LevelData,
    edges: Sequence[Edge] | None = None,
    *,
    floor_footprint: BaseGeometry | None = None,
) -> WallOrientationResolver:
    """Build one indexed world-space resolver shared by walls and windows."""

    if not isinstance(level, LevelData):
        raise TypeError("Wall orientation requires LevelData.")
    ignored_vertex_ids = {room.center_vertex_id for room in level.rooms}
    source_edges = tuple(level.vertex_data.edges if edges is None else edges)
    point_to_world, world_direction_to_image = _build_level_transforms(level)
    lines: list[LineString] = []
    for edge in source_edges:
        if (
            edge.start_vertex_id in ignored_vertex_ids
            or edge.end_vertex_id in ignored_vertex_ids
        ):
            continue
        start_vertex = level.vertex_data.get_vertex(edge.start_vertex_id)
        end_vertex = level.vertex_data.get_vertex(edge.end_vertex_id)
        if start_vertex is None or end_vertex is None:
            continue
        start_point = point_to_world(start_vertex.x, start_vertex.y)
        end_point = point_to_world(end_vertex.x, end_vertex.y)
        if math.dist(start_point, end_point) <= WALL_ORIENTATION_EPSILON:
            continue
        lines.append(LineString((start_point, end_point)))

    closing_radius = OUTER_ENVELOPE_CLOSING_RADIUS_METERS * _get_valid_level_scale(
        level
    )
    polygons = _build_orientation_polygons(
        lines,
        maximum_gap=closing_radius * 2.0,
    )
    resolved_footprint = (
        floor_footprint
        if floor_footprint is not None
        else _build_level_world_floor_footprint(level, point_to_world)
    )
    orientation_footprint = resolved_footprint
    if orientation_footprint is None and lines:
        # Preserve the legacy broad topology hint only for wall orientation.
        # Floor and ceiling generation continue to use clustered footprints.
        orientation_footprint = _build_linework_footprint(
            shapely.union_all(lines),
            closing_radius,
        )
    open_space_geometry = _build_level_world_open_space_geometry(
        level,
        point_to_world,
    )
    wall_lines = tuple(lines)
    return WallOrientationResolver(
        bounded_cells=polygons,
        floor_footprint=resolved_footprint,
        open_space_geometry=open_space_geometry,
        _orientation_footprint=orientation_footprint,
        _cell_tree=STRtree(polygons) if polygons else None,
        _wall_lines=wall_lines,
        _wall_tree=STRtree(wall_lines) if wall_lines else None,
        _wall_ray_length=_get_wall_ray_length(wall_lines),
        _paired_wall_maximum_distance=(
            WALL_PAIR_MAXIMUM_DISTANCE_METERS * _get_valid_level_scale(level)
        ),
        _paired_wall_distance_group_tolerance=(
            WALL_PAIR_DISTANCE_GROUP_TOLERANCE_METERS
            * _get_valid_level_scale(level)
        ),
        _paired_wall_span_gap_tolerance=(
            WALL_PAIR_SPAN_GAP_TOLERANCE_METERS * _get_valid_level_scale(level)
        ),
        _point_to_world=point_to_world,
        _world_direction_to_image=world_direction_to_image,
    )


def _build_orientation_polygons(
    lines: Sequence[LineString],
    *,
    maximum_gap: float,
) -> tuple[Polygon, ...]:
    """Polygonize exact walls plus conservative straight gap closures."""

    if not lines:
        return ()
    linework = shapely.union_all(lines)
    closure_lines = _build_straight_gap_closures(linework, maximum_gap)
    if closure_lines:
        linework = shapely.union_all((linework, *closure_lines))
    return tuple(
        candidate
        for candidate in shapely.get_parts(
            shapely.polygonize(tuple(shapely.get_parts(linework)))
        )
        if isinstance(candidate, Polygon) and candidate.area > WALL_ORIENTATION_EPSILON
    )


def _build_level_transforms(
    level: LevelData,
) -> tuple[LevelPointTransform, WorldDirectionToImage]:
    """Resolve the level's affine image-to-world transform only once."""

    origin = np.asarray(level_image_to_world_xy(level, 0.0, 0.0), dtype=float)
    x_step = np.asarray(level_image_to_world_xy(level, 1.0, 0.0), dtype=float) - origin
    y_step = np.asarray(level_image_to_world_xy(level, 0.0, 1.0), dtype=float) - origin

    def point_to_world(image_x: float, image_y: float) -> Point2D:
        world = origin + x_step * float(image_x) + y_step * float(image_y)
        return float(world[0]), float(world[1])

    def world_direction_to_image(direction: Point2D) -> Point2D | None:
        direction_array = np.asarray(direction, dtype=float)
        image_direction = np.asarray(
            (
                np.dot(direction_array, x_step) / np.dot(x_step, x_step),
                np.dot(direction_array, y_step) / np.dot(y_step, y_step),
            ),
            dtype=float,
        )
        length = float(np.linalg.norm(image_direction))
        if length <= WALL_ORIENTATION_EPSILON:
            return None
        return _normal_to_tuple(image_direction / length)

    return point_to_world, world_direction_to_image


def _build_level_world_floor_footprint(
    level: LevelData,
    point_transform: LevelPointTransform,
) -> BaseGeometry | None:
    """Build the gap-closed floor footprint in transformed level coordinates."""

    def point_to_world(
        image_point: Point2D,
        _blueprint_size_pixels: tuple[float, float] | None,
    ) -> np.ndarray:
        return np.asarray(point_transform(*image_point), dtype=float)

    return build_level_floor_footprint(
        level=level,
        blueprint_size_pixels=level.image_size_pixels,
        point_to_world_xy=point_to_world,
        closing_radius_meters=(
            OUTER_ENVELOPE_CLOSING_RADIUS_METERS * _get_valid_level_scale(level)
        ),
    )


def _build_level_world_open_space_geometry(
    level: LevelData,
    point_transform: LevelPointTransform,
) -> BaseGeometry | None:
    """Build slab-opening geometry for conservative void-cell recognition."""

    def point_to_world(
        image_point: Point2D,
        _blueprint_size_pixels: tuple[float, float] | None,
    ) -> np.ndarray:
        return np.asarray(point_transform(*image_point), dtype=float)

    return build_level_open_space_geometry(
        level,
        level.image_size_pixels,
        point_to_world,
    )


# ### Candidate ordering helpers ###
def _get_cell_score(polygon: Polygon) -> tuple[float, float]:
    """Prefer spacious cells when overlapping polygonized cells are present."""

    perimeter = max(float(polygon.length), WALL_ORIENTATION_EPSILON)
    area = float(polygon.area)
    return area / perimeter, area


def _get_side_score(
    candidate: WallSideCandidate,
    open_space_geometry: BaseGeometry | None,
) -> tuple[int, float, float, float, float, float, float]:
    """Prefer exposed cells to annular wall cavities, then break ties stably."""

    side_normal, polygon = candidate
    openness, area = _get_cell_score(polygon)
    centroid = polygon.centroid
    return (
        int(_cell_is_exposed(polygon, open_space_geometry)),
        openness,
        area,
        float(centroid.x),
        float(centroid.y),
        float(side_normal[0]),
        float(side_normal[1]),
    )


def _cell_is_exposed(
    polygon: Polygon,
    open_space_geometry: BaseGeometry | None,
) -> bool:
    """Distinguish ordinary rooms and open-space rings from sealed cavities."""

    if _geometry_is_mostly_open_space(polygon, open_space_geometry):
        return False
    if not polygon.interiors:
        return True
    return any(
        _geometry_is_mostly_open_space(
            Polygon(interior),
            open_space_geometry,
        )
        for interior in polygon.interiors
    )


def _geometry_is_mostly_open_space(
    geometry: BaseGeometry,
    open_space_geometry: BaseGeometry | None,
) -> bool:
    if (
        open_space_geometry is None
        or open_space_geometry.is_empty
        or geometry.is_empty
        or float(geometry.area) <= WALL_ORIENTATION_EPSILON
    ):
        return False
    overlap_area = float(geometry.intersection(open_space_geometry).area)
    return overlap_area / float(geometry.area) >= OPEN_SPACE_CELL_COVERAGE_RATIO


# ### Parallel wall postprocessing helpers ###
def _parallel_postprocess_can_override(
    candidates: Sequence[WallSideCandidate],
    open_space_geometry: BaseGeometry | None,
) -> bool:
    """Keep explicit open-space semantics ahead of geometric wall pairing."""

    return not any(
        _cell_has_explicit_open_space(polygon, open_space_geometry)
        for _normal, polygon in candidates
    )


def _cell_has_explicit_open_space(
    polygon: Polygon,
    open_space_geometry: BaseGeometry | None,
) -> bool:
    """Return whether a cell or one of its holes is a declared slab opening."""

    return _geometry_is_mostly_open_space(
        polygon,
        open_space_geometry,
    ) or any(
        _geometry_is_mostly_open_space(
            Polygon(interior),
            open_space_geometry,
        )
        for interior in polygon.interiors
    )


def _build_parallel_wall_span(
    midpoint: np.ndarray,
    wall_direction: np.ndarray,
    wall_normal: np.ndarray,
    wall_length: float,
    candidate_start: np.ndarray,
    candidate_end: np.ndarray,
) -> ParallelWallSpan | None:
    """Project one compatible candidate onto the target wall's local frame."""

    candidate_delta = candidate_end - candidate_start
    candidate_length = float(np.linalg.norm(candidate_delta))
    if candidate_length <= WALL_ORIENTATION_EPSILON:
        return None
    candidate_direction = candidate_delta / candidate_length
    denominator = _cross_2d(wall_normal, candidate_direction)
    if abs(denominator) <= WALL_ORIENTATION_EPSILON:
        return None
    relative_start = candidate_start - midpoint
    signed_distance = _cross_2d(relative_start, candidate_direction) / denominator
    if abs(signed_distance) <= WALL_ORIENTATION_EPSILON * 10.0:
        return None

    if float(np.dot(candidate_direction, wall_direction)) < 0.0:
        candidate_direction = -candidate_direction
    target_minimum = -wall_length / 2.0
    target_maximum = wall_length / 2.0
    candidate_projections = (
        float(np.dot(candidate_start - midpoint, wall_direction)),
        float(np.dot(candidate_end - midpoint, wall_direction)),
    )
    overlap_minimum = max(target_minimum, min(candidate_projections))
    overlap_maximum = min(target_maximum, max(candidate_projections))
    if overlap_maximum - overlap_minimum <= WALL_ORIENTATION_EPSILON:
        return None
    return (
        signed_distance,
        _normal_to_tuple(candidate_direction),
        overlap_minimum,
        overlap_maximum,
    )


def _resolve_grouped_parallel_wall_distances(
    spans: Sequence[ParallelWallSpan],
    wall_length: float,
    *,
    distance_tolerance: float,
    gap_tolerance: float,
) -> tuple[float, float]:
    """Merge one physical plane's split edges before measuring its coverage."""

    groups: list[list[ParallelWallSpan]] = []
    for span in sorted(
        spans,
        key=lambda candidate: (
            candidate[0],
            candidate[2],
            candidate[3],
            candidate[1],
        ),
    ):
        signed_distance, direction, _start, _end = span
        matching_group = next(
            (
                group
                for group in groups
                if math.copysign(1.0, group[0][0])
                == math.copysign(1.0, signed_distance)
                and abs(group[0][0] - signed_distance) <= distance_tolerance
                and float(np.dot(group[0][1], direction))
                >= WALL_PAIR_ALIGNMENT_MINIMUM_DOT
            ),
            None,
        )
        if matching_group is None:
            groups.append([span])
        else:
            matching_group.append(span)

    negative_distance = math.inf
    positive_distance = math.inf
    required_coverage = wall_length * WALL_PAIR_MINIMUM_OVERLAP_RATIO
    for group in groups:
        merged_spans = _merge_parallel_wall_intervals(
            tuple((span[2], span[3]) for span in group),
            gap_tolerance=gap_tolerance,
        )
        directly_opposing = any(
            start < -WALL_ORIENTATION_EPSILON
            and end > WALL_ORIENTATION_EPSILON
            and end - start + WALL_ORIENTATION_EPSILON >= required_coverage
            for start, end in merged_spans
        )
        if not directly_opposing:
            continue
        signed_distance = min(
            (span[0] for span in group),
            key=abs,
        )
        if signed_distance < 0.0:
            negative_distance = min(negative_distance, -signed_distance)
        else:
            positive_distance = min(positive_distance, signed_distance)
    return negative_distance, positive_distance


def _merge_parallel_wall_intervals(
    intervals: Sequence[tuple[float, float]],
    *,
    gap_tolerance: float,
) -> tuple[tuple[float, float], ...]:
    """Merge only overlapping or physically contiguous projected segments."""

    if not intervals:
        return ()
    ordered = sorted(intervals)
    merged: list[tuple[float, float]] = [ordered[0]]
    for start, end in ordered[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end + gap_tolerance:
            merged[-1] = previous_start, max(previous_end, end)
        else:
            merged.append((start, end))
    return tuple(merged)


def _get_wall_ray_length(lines: Sequence[LineString]) -> float:
    """Return a finite bidirectional ray length spanning all source walls."""

    if not lines:
        return 1.0
    minimum_x, minimum_y, maximum_x, maximum_y = shapely.total_bounds(lines)
    diagonal = math.hypot(maximum_x - minimum_x, maximum_y - minimum_y)
    return max(1.0, float(diagonal) + 1.0)


def _cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


# ### Floor fallback helpers ###
def _resolve_floor_footprint_normal(
    midpoint: np.ndarray,
    wall_normal: np.ndarray,
    wall_length: float,
    floor_footprint: BaseGeometry | None,
) -> Point2D | None:
    """Use the gap-closed floor envelope when exact loops are unavailable."""

    if floor_footprint is None or floor_footprint.is_empty:
        return None
    side_candidates = tuple(
        (
            side_normal,
            _measure_normal_clearance(floor_footprint, midpoint, side_normal),
        )
        for side_normal in (wall_normal, -wall_normal)
        if _geometry_contains_wall_side(
            floor_footprint,
            midpoint,
            side_normal,
            wall_length,
        )
    )
    if not side_candidates:
        return None
    side_normal, _clearance = max(
        side_candidates,
        key=lambda candidate: (
            candidate[1],
            float(candidate[0][0]),
            float(candidate[0][1]),
        ),
    )
    return _normal_to_tuple(side_normal)


def _geometry_contains_wall_side(
    geometry: BaseGeometry,
    midpoint: np.ndarray,
    side_normal: np.ndarray,
    wall_length: float,
) -> bool:
    """Return whether a geometry begins immediately beside one wall side."""

    return any(
        geometry.contains(
            Point(
                *(
                    midpoint
                    + side_normal * _get_probe_distance(wall_length, probe_ratio)
                )
            )
        )
        for probe_ratio in WALL_SIDE_PROBE_RATIOS
    )


def _measure_normal_clearance(
    geometry: BaseGeometry,
    midpoint: np.ndarray,
    side_normal: np.ndarray,
) -> float:
    """Measure exposed depth from a wall to the next boundary on one side."""

    minimum_x, minimum_y, maximum_x, maximum_y = geometry.bounds
    corners = np.asarray(
        (
            (minimum_x, minimum_y),
            (minimum_x, maximum_y),
            (maximum_x, minimum_y),
            (maximum_x, maximum_y),
        ),
        dtype=float,
    )
    farthest_corner_distance = max(
        float(np.linalg.norm(corner - midpoint)) for corner in corners
    )
    ray_length = max(1.0, farthest_corner_distance * 2.0)
    ray = LineString(
        (
            _normal_to_tuple(midpoint),
            _normal_to_tuple(midpoint + side_normal * ray_length),
        )
    )
    boundary_hits = shapely.get_coordinates(geometry.boundary.intersection(ray))
    positive_distances = tuple(
        distance
        for hit in np.asarray(boundary_hits, dtype=float)
        if (distance := float(np.dot(hit - midpoint, side_normal)))
        > WALL_ORIENTATION_EPSILON * 10.0
    )
    return min(positive_distances, default=ray_length)


# ### Numeric helpers ###
def _build_wall_frame(
    start_point: Point2D,
    end_point: Point2D,
) -> WallFrame | None:
    """Validate a wall segment and return its normalized planar frame."""

    start = np.asarray(start_point, dtype=float)
    end = np.asarray(end_point, dtype=float)
    if (
        start.shape != (2,)
        or end.shape != (2,)
        or not np.all(np.isfinite((start, end)))
    ):
        return None
    wall_delta = end - start
    wall_length = float(np.linalg.norm(wall_delta))
    if wall_length <= WALL_ORIENTATION_EPSILON:
        return None
    wall_normal = np.asarray((-wall_delta[1], wall_delta[0]), dtype=float)
    wall_normal /= wall_length
    return start, end, wall_delta, wall_length, wall_normal


def _get_probe_distance(wall_length: float, probe_ratio: float) -> float:
    return max(WALL_ORIENTATION_EPSILON * 10.0, wall_length * probe_ratio)


def _normal_to_tuple(normal: np.ndarray) -> Point2D:
    return float(normal[0]), float(normal[1])
