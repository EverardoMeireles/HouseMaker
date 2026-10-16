# ### Imports ###
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import shapely
from shapely import LineString, Point, Polygon

from housemaker.models import (
    MAX_LEVEL_INDEX,
    MIN_LEVEL_INDEX,
    LevelData,
    VertexData,
)

# ### Constants ###
OUTER_BOUNDARY_TOLERANCE_PIXELS = 1e-6


# ### Data models ###
@dataclass(frozen=True, order=True)
class OuterWallMark:
    """Persistent reference to one marked structural wall vertex."""

    level_index: int
    vertex_id: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "level_index",
            _normalize_level_index(self.level_index),
        )
        object.__setattr__(
            self,
            "vertex_id",
            _normalize_vertex_id(self.vertex_id),
        )


# ### Mark normalization and queries ###
def normalize_outer_wall_marks(
    marks: Iterable[OuterWallMark],
) -> tuple[OuterWallMark, ...]:
    """Validate, deduplicate, and deterministically order wall marks."""

    if isinstance(marks, str | bytes | Mapping):
        raise TypeError("Outer-wall marks must contain an iterable of marks.")
    normalized: set[OuterWallMark] = set()
    for raw_mark in marks:
        if not isinstance(raw_mark, OuterWallMark):
            raise TypeError("Outer-wall marks must contain OuterWallMark values.")
        normalized.add(
            OuterWallMark(
                level_index=raw_mark.level_index,
                vertex_id=raw_mark.vertex_id,
            )
        )
    return tuple(sorted(normalized))


def get_outer_wall_mark_vertex_ids(
    marks: Iterable[OuterWallMark],
    level_index: int,
) -> frozenset[int]:
    """Return marked vertex IDs owned by one level."""

    normalized_level_index = _normalize_level_index(level_index)
    return frozenset(
        mark.vertex_id
        for mark in normalize_outer_wall_marks(marks)
        if mark.level_index == normalized_level_index
    )


def reconcile_outer_wall_marks(
    levels: Sequence[LevelData],
    marks: Iterable[OuterWallMark],
) -> tuple[OuterWallMark, ...]:
    """Drop marks that no longer identify a structural wall vertex."""

    level_lookup = _build_level_lookup(levels)
    structural_vertex_ids_by_level = {
        level.index: _get_structural_wall_vertex_ids(level) for level in levels
    }
    return tuple(
        mark
        for mark in normalize_outer_wall_marks(marks)
        if (
            mark.level_index in level_lookup
            and mark.vertex_id in structural_vertex_ids_by_level[mark.level_index]
        )
    )


# ### Serialization helpers ###
def outer_wall_marks_to_dicts(
    marks: Iterable[OuterWallMark],
) -> list[dict[str, int]]:
    """Serialize canonical marks into project-safe dictionaries."""

    return [
        {
            "level_index": mark.level_index,
            "vertex_id": mark.vertex_id,
        }
        for mark in normalize_outer_wall_marks(marks)
    ]


def outer_wall_marks_from_payload(
    payload: object,
    *,
    levels: Sequence[LevelData] = (),
) -> tuple[OuterWallMark, ...]:
    """Load valid marks while ignoring malformed or stale entries."""

    if not isinstance(payload, list | tuple):
        return ()
    accepted: list[OuterWallMark] = []
    for item in payload:
        if not isinstance(item, Mapping):
            continue
        try:
            accepted.append(
                OuterWallMark(
                    level_index=item.get("level_index"),
                    vertex_id=item.get("vertex_id"),
                )
            )
        except (TypeError, ValueError, OverflowError):
            continue
    normalized = normalize_outer_wall_marks(accepted)
    return reconcile_outer_wall_marks(levels, normalized) if levels else normalized


# ### Legacy migration helpers ###
def outer_wall_marks_from_legacy_links(
    payload: object,
    *,
    levels: Sequence[LevelData] = (),
) -> tuple[OuterWallMark, ...]:
    """Convert legacy mirror-link sources into non-materialized marks.

    Only link sources are migrated. Legacy target vertices already serialized
    inside a level remain ordinary geometry and are never removed or recreated.
    """

    if not isinstance(payload, Iterable) or isinstance(
        payload,
        str | bytes | Mapping,
    ):
        return ()
    marks: list[OuterWallMark] = []
    for item in payload:
        if isinstance(item, Mapping):
            raw_level_index = item.get("source_level_index")
            raw_vertex_id = item.get("source_vertex_id")
        else:
            raw_level_index = getattr(item, "source_level_index", None)
            raw_vertex_id = getattr(item, "source_vertex_id", None)
        try:
            marks.append(
                OuterWallMark(
                    level_index=raw_level_index,
                    vertex_id=raw_vertex_id,
                )
            )
        except (TypeError, ValueError, OverflowError):
            continue
    normalized = normalize_outer_wall_marks(marks)
    return reconcile_outer_wall_marks(levels, normalized) if levels else normalized


def outer_wall_marks_from_legacy_vertex_payload(
    raw_vertex_data: object,
    *,
    source_level_index: int,
    known_vertex_ids: Iterable[int],
) -> tuple[OuterWallMark, ...]:
    """Convert former per-vertex destination flags into source marks."""

    if not isinstance(raw_vertex_data, Mapping):
        return ()
    raw_levels_by_vertex_id = raw_vertex_data.get("wall_mirror_levels")
    if not isinstance(raw_levels_by_vertex_id, Mapping):
        return ()

    normalized_source_level_index = _normalize_level_index(source_level_index)
    known_ids = {_normalize_vertex_id(vertex_id) for vertex_id in known_vertex_ids}
    marks: list[OuterWallMark] = []
    for raw_vertex_id, raw_target_indices in raw_levels_by_vertex_id.items():
        try:
            vertex_id = _normalize_vertex_id(int(raw_vertex_id))
        except (TypeError, ValueError, OverflowError):
            continue
        if vertex_id not in known_ids:
            continue
        if not isinstance(raw_target_indices, list | tuple | set | frozenset):
            continue
        has_valid_target = False
        for raw_target_index in raw_target_indices:
            try:
                target_index = _normalize_level_index(raw_target_index)
            except (TypeError, ValueError, OverflowError):
                continue
            if target_index != normalized_source_level_index:
                has_valid_target = True
                break
        if has_valid_target:
            marks.append(
                OuterWallMark(
                    level_index=normalized_source_level_index,
                    vertex_id=vertex_id,
                )
            )
    return normalize_outer_wall_marks(marks)


# ### Cross-level coordinate projection ###
def project_world_points_to_level_image(
    level: LevelData,
    world_points: Iterable[tuple[float, float]],
    *,
    excluded_vertex_ids: Iterable[int] = (),
) -> tuple[tuple[float, float], ...]:
    """Project world points under the final bounds pivot they will create.

    A level's scale is applied around the bounds center of all its vertices.
    Adding reference points can therefore move that pivot. This solves the
    final pivot for the complete batch before returning image coordinates.
    """

    if not isinstance(level, LevelData):
        raise TypeError("Outer-wall projection requires LevelData.")
    points = tuple(
        (
            _normalize_finite_float(point[0], "world X"),
            _normalize_finite_float(point[1], "world Y"),
        )
        for point in world_points
    )
    if not points:
        return ()
    excluded_ids = set(_normalize_vertex_ids(excluded_vertex_ids))
    scale = _normalize_positive_float(level.scale, "level scale")
    offset_x = _normalize_finite_float(level.offset_x_meters, "level X offset")
    offset_y = _normalize_finite_float(level.offset_y_meters, "level Y offset")
    existing_raw_points = tuple(
        _image_to_raw_level_point(level, vertex.x, vertex.y)
        for vertex in level.vertex_data.vertices
        if vertex.id not in excluded_ids
    )
    q_x_values = tuple((point[0] - offset_x) / scale for point in points)
    q_y_values = tuple((point[1] - offset_y) / scale for point in points)
    existing_x_values = tuple(point[0] for point in existing_raw_points)
    existing_y_values = tuple(point[1] for point in existing_raw_points)
    pivot_x = _solve_final_axis_pivot(existing_x_values, q_x_values, scale)
    pivot_y = _solve_final_axis_pivot(existing_y_values, q_y_values, scale)
    pivot_factor = (1.0 - scale) / scale
    raw_points = tuple(
        (
            q_x - pivot_factor * pivot_x,
            q_y - pivot_factor * pivot_y,
        )
        for q_x, q_y in zip(q_x_values, q_y_values, strict=True)
    )
    image_points = tuple(
        _raw_level_to_image_point(level, raw_x, raw_y) for raw_x, raw_y in raw_points
    )
    _validate_projected_batch(
        level,
        points,
        image_points,
        existing_raw_points,
        scale,
        offset_x,
        offset_y,
    )
    return image_points


# ### Automatic outer-boundary detection ###
def find_outer_wall_vertex_ids(level: LevelData) -> frozenset[int]:
    """Find vertices belonging to the exterior of every closed wall island."""

    if not isinstance(level, LevelData):
        raise TypeError("Outer-wall detection requires LevelData.")
    ignored_vertex_ids = {room.center_vertex_id for room in level.rooms}
    return find_outer_wall_vertex_ids_from_graph(
        level.vertex_data,
        ignored_vertex_ids=ignored_vertex_ids,
    )


def find_outer_wall_vertex_ids_from_graph(
    vertex_data: VertexData,
    *,
    ignored_vertex_ids: Iterable[int] = (),
) -> frozenset[int]:
    """Find original graph vertices lying on polygonized exterior shells.

    Noding and polygonizing the complete linework creates room cells. Their
    union removes shared internal partitions, leaving the concave exterior of
    each disconnected closed construction. Dangling wall branches do not form
    cells and therefore cannot become part of an exterior shell.
    """

    if not isinstance(vertex_data, VertexData):
        raise TypeError("Outer-wall detection requires VertexData.")
    ignored_ids = {_normalize_vertex_id(vertex_id) for vertex_id in ignored_vertex_ids}
    edge_lines: list[tuple[int, int, LineString]] = []
    for edge in vertex_data.edges:
        if edge.start_vertex_id in ignored_ids or edge.end_vertex_id in ignored_ids:
            continue
        start = vertex_data.get_vertex(edge.start_vertex_id)
        end = vertex_data.get_vertex(edge.end_vertex_id)
        if start is None or end is None:
            continue
        coordinates = (
            (_normalize_coordinate(start.x), _normalize_coordinate(start.y)),
            (_normalize_coordinate(end.x), _normalize_coordinate(end.y)),
        )
        if _points_are_close(coordinates[0], coordinates[1]):
            continue
        edge_lines.append(
            (
                edge.start_vertex_id,
                edge.end_vertex_id,
                LineString(coordinates),
            )
        )
    if not edge_lines:
        return frozenset()

    noded_linework = shapely.union_all(
        tuple(line for _start_id, _end_id, line in edge_lines)
    )
    polygons = tuple(
        part
        for part in shapely.get_parts(
            shapely.polygonize(tuple(shapely.get_parts(noded_linework)))
        )
        if isinstance(part, Polygon)
        and not part.is_empty
        and float(part.area) > OUTER_BOUNDARY_TOLERANCE_PIXELS
    )
    if not polygons:
        return frozenset()

    occupied_area = shapely.union_all(polygons)
    if occupied_area.is_empty:
        return frozenset()
    exterior_boundary = occupied_area.boundary
    structural_vertex_ids = {
        vertex_id
        for start_vertex_id, end_vertex_id, _line in edge_lines
        for vertex_id in (start_vertex_id, end_vertex_id)
    }
    marked_vertex_ids: set[int] = set()
    for vertex_id in structural_vertex_ids:
        vertex = vertex_data.get_vertex(vertex_id)
        if vertex is None:
            continue
        point = Point(
            _normalize_coordinate(vertex.x),
            _normalize_coordinate(vertex.y),
        )
        if (
            float(exterior_boundary.distance(point))
            <= OUTER_BOUNDARY_TOLERANCE_PIXELS
        ):
            marked_vertex_ids.add(vertex_id)
    return frozenset(marked_vertex_ids)


# ### Validation helpers ###
def _build_level_lookup(levels: Sequence[LevelData]) -> dict[int, LevelData]:
    level_lookup: dict[int, LevelData] = {}
    for level in levels:
        if not isinstance(level, LevelData):
            raise TypeError("Outer-wall levels must contain LevelData values.")
        if level.index in level_lookup:
            raise ValueError("Outer-wall levels must have unique indices.")
        level_lookup[level.index] = level
    return level_lookup


def _get_structural_wall_vertex_ids(level: LevelData) -> frozenset[int]:
    """Return non-helper vertices participating in at least one valid wall."""

    ignored_vertex_ids = {room.center_vertex_id for room in level.rooms}
    structural_vertex_ids: set[int] = set()
    for edge in level.vertex_data.edges:
        if (
            edge.start_vertex_id in ignored_vertex_ids
            or edge.end_vertex_id in ignored_vertex_ids
            or level.vertex_data.get_vertex(edge.start_vertex_id) is None
            or level.vertex_data.get_vertex(edge.end_vertex_id) is None
        ):
            continue
        structural_vertex_ids.update((edge.start_vertex_id, edge.end_vertex_id))
    return frozenset(structural_vertex_ids)


def _solve_final_axis_pivot(
    existing_values: tuple[float, ...],
    q_values: tuple[float, ...],
    scale: float,
) -> float:
    q_min = min(q_values)
    q_max = max(q_values)
    pivot_factor = (1.0 - scale) / scale
    if not existing_values:
        return (q_min + q_max) / (2.0 + 2.0 * pivot_factor)

    existing_min = min(existing_values)
    existing_max = max(existing_values)
    candidates = (
        (existing_min + existing_max) / 2.0,
        (existing_min + q_max) / (2.0 + pivot_factor),
        (q_min + existing_max) / (2.0 + pivot_factor),
        (q_min + q_max) / (2.0 + 2.0 * pivot_factor),
    )
    best_candidate = candidates[0]
    best_error = math.inf
    for candidate in candidates:
        new_min = q_min - pivot_factor * candidate
        new_max = q_max - pivot_factor * candidate
        actual_pivot = (min(existing_min, new_min) + max(existing_max, new_max)) / 2.0
        error = abs(actual_pivot - candidate)
        if error < best_error:
            best_candidate = candidate
            best_error = error
    if best_error > 1e-9 * max(1.0, abs(best_candidate)):
        raise RuntimeError("Unable to solve the outer-wall reference level pivot.")
    return best_candidate


def _validate_projected_batch(
    level: LevelData,
    desired_world_points: tuple[tuple[float, float], ...],
    image_points: tuple[tuple[float, float], ...],
    existing_raw_points: tuple[tuple[float, float], ...],
    scale: float,
    offset_x: float,
    offset_y: float,
) -> None:
    new_raw_points = tuple(
        _image_to_raw_level_point(level, image_x, image_y)
        for image_x, image_y in image_points
    )
    all_raw_points = (*existing_raw_points, *new_raw_points)
    pivot_x = (
        min(point[0] for point in all_raw_points)
        + max(point[0] for point in all_raw_points)
    ) / 2.0
    pivot_y = (
        min(point[1] for point in all_raw_points)
        + max(point[1] for point in all_raw_points)
    ) / 2.0
    for raw_point, desired_point in zip(
        new_raw_points,
        desired_world_points,
        strict=True,
    ):
        actual_point = (
            pivot_x + (raw_point[0] - pivot_x) * scale + offset_x,
            pivot_y + (raw_point[1] - pivot_y) * scale + offset_y,
        )
        if not _points_are_close(actual_point, desired_point, tolerance=1e-7):
            raise RuntimeError(
                "Outer-wall reference projection could not preserve world coordinates."
            )


def _image_to_raw_level_point(
    level: LevelData,
    image_x: float,
    image_y: float,
) -> tuple[float, float]:
    centered_x = float(image_x)
    centered_y = float(image_y)
    if level.image_size_pixels is not None:
        image_width, image_height = level.image_size_pixels
        centered_x -= float(image_width) / 2.0
        centered_y -= float(image_height) / 2.0
    return centered_x * level.meters_per_pixel, -centered_y * level.meters_per_pixel


def _raw_level_to_image_point(
    level: LevelData,
    raw_x: float,
    raw_y: float,
) -> tuple[float, float]:
    image_x = raw_x / level.meters_per_pixel
    image_y = -raw_y / level.meters_per_pixel
    if level.image_size_pixels is not None:
        image_width, image_height = level.image_size_pixels
        image_x += float(image_width) / 2.0
        image_y += float(image_height) / 2.0
    return image_x, image_y


def _normalize_level_index(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("Outer-wall level index must be an integer.")
    if not MIN_LEVEL_INDEX <= value <= MAX_LEVEL_INDEX:
        raise ValueError(
            "Outer-wall level index must be between "
            f"{MIN_LEVEL_INDEX} and {MAX_LEVEL_INDEX}."
        )
    return value


def _normalize_vertex_id(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("Outer-wall vertex ID must be an integer.")
    if value <= 0:
        raise ValueError("Outer-wall vertex ID must be positive.")
    return value


def _normalize_vertex_ids(values: Iterable[int]) -> tuple[int, ...]:
    if isinstance(values, str | bytes | Mapping):
        raise TypeError("Outer-wall vertex IDs must contain an iterable.")
    return tuple(sorted({_normalize_vertex_id(value) for value in values}))


def _normalize_coordinate(value: object) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Outer-wall coordinates must be numeric.") from error
    if not math.isfinite(normalized):
        raise ValueError("Outer-wall coordinates must be finite.")
    return normalized


def _normalize_finite_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"Outer-wall {name} must be a number.")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Outer-wall {name} must be a number.") from error
    if not math.isfinite(normalized):
        raise ValueError(f"Outer-wall {name} must be finite.")
    return normalized


def _normalize_positive_float(value: object, name: str) -> float:
    normalized = _normalize_finite_float(value, name)
    if normalized <= 0.0:
        raise ValueError(f"Outer-wall {name} must be greater than zero.")
    return normalized


def _points_are_close(
    first: tuple[float, float],
    second: tuple[float, float],
    *,
    tolerance: float = OUTER_BOUNDARY_TOLERANCE_PIXELS,
) -> bool:
    return math.isclose(first[0], second[0], rel_tol=0.0, abs_tol=tolerance) and (
        math.isclose(first[1], second[1], rel_tol=0.0, abs_tol=tolerance)
    )
