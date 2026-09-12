# ### Imports ###
from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from housemaker.level_coordinates import level_image_to_world_xy
from housemaker.models import (
    MAX_LEVEL_INDEX,
    MIN_LEVEL_INDEX,
    PIXEL_TO_METER,
    LevelData,
)


# ### Data models ###
@dataclass(frozen=True)
class WallMirrorVertexLink:
    """Persistent ownership of one materialized cross-level vertex copy."""

    source_level_index: int
    source_vertex_id: int
    target_level_index: int
    target_vertex_id: int

    def __post_init__(self) -> None:
        source_level_index = _normalize_level_index(
            self.source_level_index,
            "source",
        )
        target_level_index = _normalize_level_index(
            self.target_level_index,
            "target",
        )
        if source_level_index == target_level_index:
            raise ValueError(
                "A wall-mirror source and target must be on different levels."
            )
        object.__setattr__(self, "source_level_index", source_level_index)
        object.__setattr__(
            self,
            "source_vertex_id",
            _normalize_vertex_id(self.source_vertex_id, "source"),
        )
        object.__setattr__(self, "target_level_index", target_level_index)
        object.__setattr__(
            self,
            "target_vertex_id",
            _normalize_vertex_id(self.target_vertex_id, "target"),
        )

    @property
    def source_target_key(self) -> tuple[int, int, int]:
        """Return the unique source vertex and destination-level key."""

        return (
            self.source_level_index,
            self.source_vertex_id,
            self.target_level_index,
        )

    @property
    def target_key(self) -> tuple[int, int]:
        """Return the uniquely owned target vertex key."""

        return self.target_level_index, self.target_vertex_id


@dataclass(frozen=True)
class WallMirrorTopologyResult:
    """Canonical links and affected levels produced by one topology action."""

    links: tuple[WallMirrorVertexLink, ...]
    changed_level_indices: tuple[int, ...] = ()
    target_level_index: int | None = None
    source_to_target_vertex_ids: tuple[tuple[int, int], ...] = ()
    links_changed: bool = False

    @property
    def changed(self) -> bool:
        """Return whether links or any level topology changed."""

        return self.links_changed or bool(self.changed_level_indices)


# ### Link normalization and serialization ###
def normalize_wall_mirror_links(
    links: Iterable[WallMirrorVertexLink],
) -> tuple[WallMirrorVertexLink, ...]:
    """Validate, deduplicate, and deterministically order mirror links."""

    normalized_by_source_target: dict[
        tuple[int, int, int],
        WallMirrorVertexLink,
    ] = {}
    source_by_target: dict[tuple[int, int], tuple[int, int]] = {}
    for raw_link in links:
        if not isinstance(raw_link, WallMirrorVertexLink):
            raise TypeError("Wall-mirror links must contain link objects.")
        link = WallMirrorVertexLink(
            source_level_index=raw_link.source_level_index,
            source_vertex_id=raw_link.source_vertex_id,
            target_level_index=raw_link.target_level_index,
            target_vertex_id=raw_link.target_vertex_id,
        )
        existing_link = normalized_by_source_target.get(
            link.source_target_key
        )
        if existing_link is not None and existing_link != link:
            raise ValueError(
                "A wall vertex cannot own two mirrors on the same level."
            )
        expected_source = (
            link.source_level_index,
            link.source_vertex_id,
        )
        existing_source = source_by_target.get(link.target_key)
        if existing_source is not None and existing_source != expected_source:
            raise ValueError(
                "A mirrored target vertex cannot have two source vertices."
            )
        normalized_by_source_target[link.source_target_key] = link
        source_by_target[link.target_key] = expected_source

    return tuple(
        sorted(
            normalized_by_source_target.values(),
            key=_wall_mirror_link_sort_key,
        )
    )


def wall_mirror_links_to_dicts(
    links: Iterable[WallMirrorVertexLink],
) -> list[dict[str, int]]:
    """Serialize canonical links into project-safe dictionaries."""

    return [
        {
            "source_level_index": link.source_level_index,
            "source_vertex_id": link.source_vertex_id,
            "target_level_index": link.target_level_index,
            "target_vertex_id": link.target_vertex_id,
        }
        for link in normalize_wall_mirror_links(links)
    ]


def wall_mirror_links_from_payload(
    payload: object,
    *,
    levels: Sequence[LevelData] = (),
) -> tuple[WallMirrorVertexLink, ...]:
    """Load valid non-conflicting links while ignoring malformed entries."""

    if not isinstance(payload, list | tuple):
        return ()
    known_level_indices = {level.index for level in levels}
    accepted_links: list[WallMirrorVertexLink] = []
    for item in payload:
        if not isinstance(item, Mapping):
            continue
        try:
            link = WallMirrorVertexLink(
                source_level_index=item.get("source_level_index"),
                source_vertex_id=item.get("source_vertex_id"),
                target_level_index=item.get("target_level_index"),
                target_vertex_id=item.get("target_vertex_id"),
            )
            if known_level_indices and (
                link.source_level_index not in known_level_indices
                or link.target_level_index not in known_level_indices
            ):
                continue
            accepted_links = list(
                normalize_wall_mirror_links((*accepted_links, link))
            )
        except (TypeError, ValueError, OverflowError):
            continue
    return tuple(accepted_links)


# ### Link queries ###
def get_wall_mirror_vertex_ids(
    links: Iterable[WallMirrorVertexLink],
    level_index: int,
) -> frozenset[int]:
    """Return source and target vertices displayed green on one level."""

    normalized_level_index = _normalize_level_index(level_index, "display")
    vertex_ids: set[int] = set()
    for link in normalize_wall_mirror_links(links):
        if link.source_level_index == normalized_level_index:
            vertex_ids.add(link.source_vertex_id)
        if link.target_level_index == normalized_level_index:
            vertex_ids.add(link.target_vertex_id)
    return frozenset(vertex_ids)


def get_outgoing_wall_mirror_links(
    links: Iterable[WallMirrorVertexLink],
    source_level_index: int,
    source_vertex_ids: Iterable[int],
) -> tuple[WallMirrorVertexLink, ...]:
    """Return mirrors owned by the requested source vertices."""

    normalized_level_index = _normalize_level_index(
        source_level_index,
        "source",
    )
    normalized_vertex_ids = set(
        _normalize_vertex_ids(source_vertex_ids)
    )
    return tuple(
        link
        for link in normalize_wall_mirror_links(links)
        if link.source_level_index == normalized_level_index
        and link.source_vertex_id in normalized_vertex_ids
    )


def find_next_wall_mirror_target_level_index(
    levels: Sequence[LevelData],
    links: Iterable[WallMirrorVertexLink],
    source_level_index: int,
    source_vertex_ids: Iterable[int],
    direction: int,
) -> int | None:
    """Find the nearest directional level missing any selected mirror."""

    if isinstance(direction, bool) or direction not in (-1, 1):
        raise ValueError("Wall-mirror direction must be -1 or 1.")
    level_lookup = _build_level_lookup(levels)
    normalized_source_level_index = _normalize_level_index(
        source_level_index,
        "source",
    )
    source_level = level_lookup.get(normalized_source_level_index)
    if source_level is None:
        raise ValueError("The wall-mirror source level does not exist.")
    selected_vertex_ids = _normalize_vertex_ids(source_vertex_ids)
    if not selected_vertex_ids:
        return None
    for vertex_id in selected_vertex_ids:
        if source_level.vertex_data.get_vertex(vertex_id) is None:
            raise ValueError("A selected wall-mirror vertex does not exist.")

    canonical_links = normalize_wall_mirror_links(links)
    mirrored_keys = {
        link.source_target_key
        for link in canonical_links
    }
    candidates = sorted(
        (
            level.index
            for level in levels
            if (level.index - normalized_source_level_index) * direction > 0
        ),
        key=lambda level_index: (
            level_index - normalized_source_level_index
        )
        * direction,
    )
    for candidate_index in candidates:
        if any(
            (
                normalized_source_level_index,
                vertex_id,
                candidate_index,
            )
            not in mirrored_keys
            for vertex_id in selected_vertex_ids
        ):
            return candidate_index
    return None


# ### Topology mutations ###
def mirror_wall_vertex_group(
    levels: Sequence[LevelData],
    links: Iterable[WallMirrorVertexLink],
    source_level_index: int,
    source_vertex_ids: Iterable[int],
    target_level_index: int,
) -> WallMirrorTopologyResult:
    """Materialize selected vertices and their induced edges on one level."""

    level_lookup = _build_level_lookup(levels)
    normalized_source_level_index = _normalize_level_index(
        source_level_index,
        "source",
    )
    normalized_target_level_index = _normalize_level_index(
        target_level_index,
        "target",
    )
    if normalized_source_level_index == normalized_target_level_index:
        raise ValueError(
            "A wall-mirror source and target must be on different levels."
        )
    source_level = level_lookup.get(normalized_source_level_index)
    target_level = level_lookup.get(normalized_target_level_index)
    if source_level is None:
        raise ValueError("The wall-mirror source level does not exist.")
    if target_level is None:
        raise ValueError("The wall-mirror target level does not exist.")

    selected_vertex_ids = _normalize_vertex_ids(source_vertex_ids)
    if not selected_vertex_ids:
        return WallMirrorTopologyResult(
            links=normalize_wall_mirror_links(links),
            target_level_index=normalized_target_level_index,
        )
    source_vertices = []
    for vertex_id in selected_vertex_ids:
        vertex = source_level.vertex_data.get_vertex(vertex_id)
        if vertex is None:
            raise ValueError("A selected wall-mirror vertex does not exist.")
        source_vertices.append(vertex)

    canonical_links = normalize_wall_mirror_links(links)
    links_by_source_target = {
        link.source_target_key: link
        for link in canonical_links
    }
    mapped_target_ids: dict[int, int] = {}
    for vertex_id in selected_vertex_ids:
        existing_link = links_by_source_target.get(
            (
                normalized_source_level_index,
                vertex_id,
                normalized_target_level_index,
            )
        )
        if existing_link is None:
            continue
        if (
            target_level.vertex_data.get_vertex(existing_link.target_vertex_id)
            is None
        ):
            raise ValueError(
                "A wall-mirror link points to a missing target vertex; "
                "reconcile it before mirroring."
            )
        mapped_target_ids[vertex_id] = existing_link.target_vertex_id

    target_snapshot = target_level.vertex_data.clone()
    try:
        missing_source_vertices = tuple(
            vertex
            for vertex in source_vertices
            if vertex.id not in mapped_target_ids
        )
        if missing_source_vertices:
            target_level.vertex_data.normalize_next_vertex_id()
        desired_world_points = tuple(
            level_image_to_world_xy(source_level, vertex.x, vertex.y)
            for vertex in missing_source_vertices
        )
        projected_image_points = project_world_points_to_level_image(
            target_level,
            desired_world_points,
        )
        new_links = list(canonical_links)
        topology_changed = False
        links_changed = False
        for source_vertex, image_point in zip(
            missing_source_vertices,
            projected_image_points,
            strict=True,
        ):
            target_vertex = target_level.vertex_data.add_vertex(*image_point)
            target_vertex_id = target_vertex.id
            mapped_target_ids[source_vertex.id] = target_vertex_id
            new_links.append(
                WallMirrorVertexLink(
                    source_level_index=normalized_source_level_index,
                    source_vertex_id=source_vertex.id,
                    target_level_index=normalized_target_level_index,
                    target_vertex_id=target_vertex_id,
                )
            )
            topology_changed = True
            links_changed = True

        selected_vertex_id_set = set(selected_vertex_ids)
        for edge in source_level.vertex_data.edges:
            if not {
                edge.start_vertex_id,
                edge.end_vertex_id,
            }.issubset(selected_vertex_id_set):
                continue
            created_edge = target_level.vertex_data.add_edge(
                mapped_target_ids[edge.start_vertex_id],
                mapped_target_ids[edge.end_vertex_id],
            )
            topology_changed = topology_changed or created_edge is not None

        normalized_links = normalize_wall_mirror_links(new_links)
        _validate_projected_target_world_points(
            target_level,
            mapped_target_ids,
            tuple(vertex.id for vertex in missing_source_vertices),
            desired_world_points,
        )
    except Exception:
        target_level.vertex_data.copy_from(target_snapshot)
        raise

    changed_levels = (
        tuple(
            sorted(
                {
                    normalized_source_level_index,
                    normalized_target_level_index,
                }
            )
        )
        if topology_changed or links_changed
        else ()
    )
    return WallMirrorTopologyResult(
        links=normalized_links,
        changed_level_indices=changed_levels,
        target_level_index=normalized_target_level_index,
        source_to_target_vertex_ids=tuple(
            (vertex_id, mapped_target_ids[vertex_id])
            for vertex_id in selected_vertex_ids
        ),
        links_changed=links_changed,
    )


def remove_wall_vertex_mirrors(
    levels: Sequence[LevelData],
    links: Iterable[WallMirrorVertexLink],
    level_index: int,
    vertex_ids: Iterable[int],
    *,
    remove_outgoing: bool = True,
    remove_incoming: bool = True,
) -> WallMirrorTopologyResult:
    """Remove selected mirror relations and all owned descendants."""

    if not remove_outgoing and not remove_incoming:
        return WallMirrorTopologyResult(
            links=normalize_wall_mirror_links(links)
        )
    normalized_level_index = _normalize_level_index(level_index, "selected")
    selected_vertex_ids = set(_normalize_vertex_ids(vertex_ids))
    canonical_links = normalize_wall_mirror_links(links)
    if not selected_vertex_ids:
        return WallMirrorTopologyResult(links=canonical_links)

    seed_links = tuple(
        link
        for link in canonical_links
        if (
            remove_outgoing
            and link.source_level_index == normalized_level_index
            and link.source_vertex_id in selected_vertex_ids
        )
        or (
            remove_incoming
            and link.target_level_index == normalized_level_index
            and link.target_vertex_id in selected_vertex_ids
        )
    )
    return _remove_wall_mirror_link_closure(
        levels,
        canonical_links,
        seed_links,
    )


def reconcile_wall_mirror_topology(
    levels: Sequence[LevelData],
    links: Iterable[WallMirrorVertexLink],
) -> WallMirrorTopologyResult:
    """Remove invalid links and cascade copies whose sources disappeared."""

    level_lookup = _build_level_lookup(levels)
    canonical_links = normalize_wall_mirror_links(links)
    source_missing_links = tuple(
        link
        for link in canonical_links
        if not _vertex_exists(
            level_lookup,
            link.source_level_index,
            link.source_vertex_id,
        )
    )
    removal_result = _remove_wall_mirror_link_closure(
        levels,
        canonical_links,
        source_missing_links,
    )
    remaining_links = tuple(
        link
        for link in removal_result.links
        if _vertex_exists(
            level_lookup,
            link.target_level_index,
            link.target_vertex_id,
        )
    )
    normalized_remaining_links = normalize_wall_mirror_links(remaining_links)
    dropped_target_links = len(normalized_remaining_links) != len(
        removal_result.links
    )
    changed_levels = set(removal_result.changed_level_indices)
    if dropped_target_links:
        for link in removal_result.links:
            if link not in normalized_remaining_links:
                changed_levels.add(link.source_level_index)
    return WallMirrorTopologyResult(
        links=normalized_remaining_links,
        changed_level_indices=tuple(sorted(changed_levels)),
        links_changed=(
            removal_result.links_changed
            or dropped_target_links
            or normalized_remaining_links != canonical_links
        ),
    )


def materialize_legacy_wall_mirrors(
    levels: Sequence[LevelData],
    legacy_targets_by_source: Mapping[
        tuple[int, int],
        Iterable[int],
    ],
    links: Iterable[WallMirrorVertexLink] = (),
) -> WallMirrorTopologyResult:
    """Convert legacy source flags into owned target vertices and edges."""

    canonical_links = normalize_wall_mirror_links(links)
    changed_levels: set[int] = set()
    links_changed = False
    mappings: dict[tuple[int, int], int] = {}
    grouped_source_ids: dict[tuple[int, int], set[int]] = {}
    for raw_source_key, raw_target_indices in legacy_targets_by_source.items():
        if (
            not isinstance(raw_source_key, tuple)
            or len(raw_source_key) != 2
        ):
            continue
        try:
            source_level_index = _normalize_level_index(
                raw_source_key[0],
                "legacy source",
            )
            source_vertex_id = _normalize_vertex_id(
                raw_source_key[1],
                "legacy source",
            )
        except (TypeError, ValueError, OverflowError):
            continue
        if not isinstance(raw_target_indices, Iterable) or isinstance(
            raw_target_indices,
            str | bytes | Mapping,
        ):
            continue
        for raw_target_index in raw_target_indices:
            try:
                target_level_index = _normalize_level_index(
                    raw_target_index,
                    "legacy target",
                )
            except (TypeError, ValueError, OverflowError):
                continue
            if target_level_index == source_level_index:
                continue
            grouped_source_ids.setdefault(
                (source_level_index, target_level_index),
                set(),
            ).add(source_vertex_id)

    for (source_level_index, target_level_index), source_ids in sorted(
        grouped_source_ids.items()
    ):
        try:
            result = mirror_wall_vertex_group(
                levels,
                canonical_links,
                source_level_index,
                source_ids,
                target_level_index,
            )
        except (TypeError, ValueError, RuntimeError, OverflowError):
            continue
        canonical_links = result.links
        changed_levels.update(result.changed_level_indices)
        links_changed = links_changed or result.links_changed
        mappings.update(
            {
                (source_level_index, source_vertex_id): target_vertex_id
                for source_vertex_id, target_vertex_id in (
                    result.source_to_target_vertex_ids
                )
            }
        )

    return WallMirrorTopologyResult(
        links=canonical_links,
        changed_level_indices=tuple(sorted(changed_levels)),
        source_to_target_vertex_ids=tuple(
            (source_key[1], target_vertex_id)
            for source_key, target_vertex_id in sorted(mappings.items())
        ),
        links_changed=links_changed,
    )


# ### Exact batch projection ###
def project_world_points_to_level_image(
    level: LevelData,
    world_points: Iterable[tuple[float, float]],
    *,
    excluded_vertex_ids: Iterable[int] = (),
) -> tuple[tuple[float, float], ...]:
    """Project a batch exactly under the bounds pivot it will create."""

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
        _raw_level_to_image_point(level, raw_x, raw_y)
        for raw_x, raw_y in raw_points
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


# ### Internal topology helpers ###
def _remove_wall_mirror_link_closure(
    levels: Sequence[LevelData],
    canonical_links: tuple[WallMirrorVertexLink, ...],
    seed_links: Iterable[WallMirrorVertexLink],
) -> WallMirrorTopologyResult:
    level_lookup = _build_level_lookup(levels)
    links_by_source: dict[tuple[int, int], list[WallMirrorVertexLink]] = {}
    for link in canonical_links:
        links_by_source.setdefault(
            (link.source_level_index, link.source_vertex_id),
            [],
        ).append(link)

    queue = deque(seed_links)
    removed_links: set[WallMirrorVertexLink] = set()
    while queue:
        link = queue.popleft()
        if link in removed_links:
            continue
        removed_links.add(link)
        queue.extend(links_by_source.get(link.target_key, ()))
    if not removed_links:
        return WallMirrorTopologyResult(links=canonical_links)

    level_snapshots = {
        level.index: level.vertex_data.clone()
        for level in levels
    }
    changed_levels: set[int] = set()
    try:
        for link in sorted(
            removed_links,
            key=_wall_mirror_link_sort_key,
            reverse=True,
        ):
            target_level = level_lookup.get(link.target_level_index)
            if target_level is None:
                changed_levels.add(link.source_level_index)
                continue
            if target_level.vertex_data.delete_vertex(link.target_vertex_id):
                changed_levels.add(link.target_level_index)
            changed_levels.add(link.source_level_index)
    except Exception:
        for level in levels:
            level.vertex_data.copy_from(level_snapshots[level.index])
        raise

    remaining_links = normalize_wall_mirror_links(
        link
        for link in canonical_links
        if link not in removed_links
    )
    return WallMirrorTopologyResult(
        links=remaining_links,
        changed_level_indices=tuple(sorted(changed_levels)),
        links_changed=True,
    )


def _validate_projected_target_world_points(
    target_level: LevelData,
    mapped_target_ids: Mapping[int, int],
    selected_source_vertex_ids: tuple[int, ...],
    desired_world_points: tuple[tuple[float, float], ...],
) -> None:
    for source_vertex_id, desired_point in zip(
        selected_source_vertex_ids,
        desired_world_points,
        strict=True,
    ):
        target_vertex = target_level.vertex_data.get_vertex(
            mapped_target_ids[source_vertex_id]
        )
        if target_vertex is None:
            raise RuntimeError("A materialized wall-mirror vertex is missing.")
        actual_point = level_image_to_world_xy(
            target_level,
            target_vertex.x,
            target_vertex.y,
        )
        if not _points_are_close(actual_point, desired_point, tolerance=1e-7):
            raise RuntimeError(
                "Wall-mirror projection could not preserve world coordinates."
            )


def _build_level_lookup(
    levels: Sequence[LevelData],
) -> dict[int, LevelData]:
    level_lookup: dict[int, LevelData] = {}
    for level in levels:
        if not isinstance(level, LevelData):
            raise TypeError("Wall-mirror levels must contain LevelData values.")
        if level.index in level_lookup:
            raise ValueError("Wall-mirror levels must have unique indices.")
        level_lookup[level.index] = level
    return level_lookup


def _vertex_exists(
    level_lookup: Mapping[int, LevelData],
    level_index: int,
    vertex_id: int,
) -> bool:
    level = level_lookup.get(level_index)
    return (
        level is not None
        and level.vertex_data.get_vertex(vertex_id) is not None
    )


# ### Projection helpers ###
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
        actual_pivot = (
            min(existing_min, new_min) + max(existing_max, new_max)
        ) / 2.0
        error = abs(actual_pivot - candidate)
        if error < best_error:
            best_candidate = candidate
            best_error = error
    if best_error > 1e-9 * max(1.0, abs(best_candidate)):
        raise RuntimeError("Unable to solve the wall-mirror level pivot.")
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
                "Wall-mirror projection could not preserve world coordinates."
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
    return centered_x * PIXEL_TO_METER, -centered_y * PIXEL_TO_METER


def _raw_level_to_image_point(
    level: LevelData,
    raw_x: float,
    raw_y: float,
) -> tuple[float, float]:
    image_x = raw_x / PIXEL_TO_METER
    image_y = -raw_y / PIXEL_TO_METER
    if level.image_size_pixels is not None:
        image_width, image_height = level.image_size_pixels
        image_x += float(image_width) / 2.0
        image_y += float(image_height) / 2.0
    return image_x, image_y


# ### Validation helpers ###
def _normalize_level_index(value: object, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Wall-mirror {role} level index must be an integer.")
    if not MIN_LEVEL_INDEX <= value <= MAX_LEVEL_INDEX:
        raise ValueError(
            f"Wall-mirror {role} level index must be between "
            f"{MIN_LEVEL_INDEX} and {MAX_LEVEL_INDEX}."
        )
    return value


def _normalize_vertex_id(value: object, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Wall-mirror {role} vertex ID must be an integer.")
    if value <= 0:
        raise ValueError(
            f"Wall-mirror {role} vertex ID must be a positive integer."
        )
    return value


def _normalize_vertex_ids(values: Iterable[int]) -> tuple[int, ...]:
    if isinstance(values, str | bytes | Mapping):
        raise TypeError("Wall-mirror vertex IDs must contain an iterable.")
    return tuple(
        sorted({_normalize_vertex_id(value, "selected") for value in values})
    )


def _normalize_finite_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"Wall-mirror {name} must be a number.")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Wall-mirror {name} must be a number.") from error
    if not math.isfinite(normalized):
        raise ValueError(f"Wall-mirror {name} must be finite.")
    return normalized


def _normalize_positive_float(value: object, name: str) -> float:
    normalized = _normalize_finite_float(value, name)
    if normalized <= 0.0:
        raise ValueError(f"Wall-mirror {name} must be greater than zero.")
    return normalized


def _wall_mirror_link_sort_key(
    link: WallMirrorVertexLink,
) -> tuple[int, int, int, int]:
    return (
        link.source_level_index,
        link.source_vertex_id,
        link.target_level_index,
        link.target_vertex_id,
    )


def _points_are_close(
    first: tuple[float, float],
    second: tuple[float, float],
    *,
    tolerance: float = 1e-10,
) -> bool:
    return math.isclose(first[0], second[0], rel_tol=0.0, abs_tol=tolerance) and (
        math.isclose(first[1], second[1], rel_tol=0.0, abs_tol=tolerance)
    )
