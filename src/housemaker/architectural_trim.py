# ### Imports ###
from __future__ import annotations

import math
import re
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import trimesh

from housemaker.models import (
    ARCHITECTURAL_TRIM_KINDS,
    DEFAULT_ARCHITECTURAL_TRIM_CORNER_RADIUS_METERS,
    DEFAULT_ARCHITECTURAL_TRIM_DEPTH_METERS,
    DEFAULT_ARCHITECTURAL_TRIM_HEIGHT_METERS,
    DEFAULT_ARCHITECTURAL_TRIM_WIDTH_METERS,
    TRIM_KIND_CORNICE,
    TRIM_KIND_EDGING_STRIP,
    TRIM_KIND_SKIRTING_BOARD,
    ArchitecturalTrimData,
    LevelData,
)

# ### Constants ###
TRIM_PART_FRONT = "front"
TRIM_PART_TOP = "top"
TRIM_PART_BOTTOM = "bottom"
TRIM_PART_SIDES = "sides"
TRIM_PART_KINDS = frozenset(
    {
        TRIM_PART_FRONT,
        TRIM_PART_TOP,
        TRIM_PART_BOTTOM,
        TRIM_PART_SIDES,
    }
)
TRIM_SURFACE_TYPE = "wall"
TRIM_HANDLE_WIDTH = "width"
TRIM_HANDLE_HEIGHT = "height"
TRIM_HANDLE_DEPTH = "depth"
TRIM_HANDLE_CORNER_RADIUS = "corner_radius"
TRIM_HANDLE_KINDS = frozenset(
    {
        TRIM_HANDLE_WIDTH,
        TRIM_HANDLE_HEIGHT,
        TRIM_HANDLE_DEPTH,
        TRIM_HANDLE_CORNER_RADIUS,
    }
)
TRIM_CORNER_ANGLE_TOLERANCE_DEGREES = 7.5
TRIM_ENDPOINT_TOLERANCE_METERS = 1e-5
TRIM_GEOMETRY_EPSILON = 1e-8
TRIM_ROUNDED_PROFILE_SEGMENTS = 8
TRIM_CORNICE_ROUNDED_PROFILE_SEGMENTS = 16
TRIM_MITER_BOUNDARY_MINIMUM_X = "minimum_x"
TRIM_MITER_BOUNDARY_MAXIMUM_X = "maximum_x"
TRIM_MITER_BOUNDARIES = frozenset(
    {
        TRIM_MITER_BOUNDARY_MINIMUM_X,
        TRIM_MITER_BOUNDARY_MAXIMUM_X,
    }
)
_TRIM_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_TRIM_SURFACE_ID_PATTERN = re.compile(
    r"^trim:(?P<trim_id>[0-9a-f]{32})/"
    r"part:(?P<part_kind>front|top|bottom|sides):wall$"
)
__all__ = [
    "ARCHITECTURAL_TRIM_KINDS",
    "TRIM_KIND_CORNICE",
    "TRIM_KIND_EDGING_STRIP",
    "TRIM_KIND_SKIRTING_BOARD",
    "TRIM_MITER_BOUNDARY_MAXIMUM_X",
    "TRIM_MITER_BOUNDARY_MINIMUM_X",
    "TRIM_PART_BOTTOM",
    "TRIM_PART_FRONT",
    "TRIM_PART_KINDS",
    "TRIM_PART_SIDES",
    "TRIM_PART_TOP",
    "ArchitecturalTrimCorniceMiterDescriptor",
    "ArchitecturalTrimEditHandle",
    "ArchitecturalTrimEditTarget",
    "ArchitecturalTrimEndpointMiter",
    "ArchitecturalTrimGeometry",
    "ArchitecturalTrimPart",
    "ArchitecturalTrimPlacementRequest",
    "ArchitecturalTrimRun",
    "add_architectural_trim",
    "build_architectural_trim_edit_targets",
    "build_architectural_trim_geometry",
    "build_architectural_trim_parts",
    "build_architectural_trim_placement_preview_meshes",
    "build_architectural_trim_surface_id",
    "build_cornice_miter_descriptors",
    "get_architectural_trim",
    "is_architectural_trim_surface_id",
    "parse_architectural_trim_surface_id",
    "remove_architectural_trim",
    "update_architectural_trim",
    "validate_architectural_trim_placement",
]


# ### Public data models ###
@dataclass(frozen=True)
class ArchitecturalTrimPlacementRequest:
    """Validated values required to insert one level-owned trim component."""

    kind: str
    wall_surface_ids: tuple[str, ...]
    corner_vertex_id: int | None = None
    width_meters: float = DEFAULT_ARCHITECTURAL_TRIM_WIDTH_METERS
    height_meters: float = DEFAULT_ARCHITECTURAL_TRIM_HEIGHT_METERS
    depth_meters: float = DEFAULT_ARCHITECTURAL_TRIM_DEPTH_METERS
    corner_radius_meters: float = DEFAULT_ARCHITECTURAL_TRIM_CORNER_RADIUS_METERS
    trim_id: str | None = None

    def __post_init__(self) -> None:
        normalized_id = (
            uuid.uuid4().hex
            if self.trim_id is None
            else str(self.trim_id).strip().lower()
        )
        validated = ArchitecturalTrimData(
            trim_id=normalized_id,
            kind=self.kind,
            wall_surface_ids=tuple(self.wall_surface_ids),
            corner_vertex_id=self.corner_vertex_id,
            width_meters=self.width_meters,
            height_meters=self.height_meters,
            depth_meters=self.depth_meters,
            corner_radius_meters=self.corner_radius_meters,
        )
        for field_name in (
            "kind",
            "wall_surface_ids",
            "corner_vertex_id",
            "width_meters",
            "height_meters",
            "depth_meters",
            "corner_radius_meters",
        ):
            object.__setattr__(self, field_name, getattr(validated, field_name))
        object.__setattr__(self, "trim_id", validated.trim_id)

    def to_data(self) -> ArchitecturalTrimData:
        return ArchitecturalTrimData(
            trim_id=str(self.trim_id),
            kind=self.kind,
            wall_surface_ids=self.wall_surface_ids,
            corner_vertex_id=self.corner_vertex_id,
            width_meters=self.width_meters,
            height_meters=self.height_meters,
            depth_meters=self.depth_meters,
            corner_radius_meters=self.corner_radius_meters,
        )


@dataclass(frozen=True)
class ArchitecturalTrimPart:
    """One stable selectable face group from a joined trim run."""

    trim_id: str
    run_id: str
    semantic_id: str
    part_kind: str
    surface_type: str
    mesh: trimesh.Trimesh
    level_index: int
    source_wall_surface_id: str | None = None

    def __post_init__(self) -> None:
        if _TRIM_ID_PATTERN.fullmatch(str(self.trim_id)) is None:
            raise ValueError("Architectural trim parts require a valid trim ID.")
        if not str(self.run_id).strip():
            raise ValueError("Architectural trim parts require a run ID.")
        if self.part_kind not in TRIM_PART_KINDS:
            raise ValueError("Architectural trim parts require a known part kind.")
        if self.surface_type != TRIM_SURFACE_TYPE:
            raise ValueError("Architectural trim parts must use the wall surface type.")
        if self.semantic_id != build_architectural_trim_surface_id(
            self.trim_id,
            self.part_kind,
        ):
            raise ValueError("Architectural trim part semantic ID is inconsistent.")
        if not isinstance(self.mesh, trimesh.Trimesh):
            raise TypeError("Architectural trim parts require triangle meshes.")

    @property
    def surface_id(self) -> str:
        return self.semantic_id

    @property
    def key(self) -> str:
        return self.semantic_id


@dataclass(frozen=True)
class ArchitecturalTrimRun:
    """One connected, seamless export mesh composed from compatible trims."""

    run_id: str
    trim_ids: tuple[str, ...]
    kind: str
    level_index: int
    mesh: trimesh.Trimesh
    parts: tuple[ArchitecturalTrimPart, ...]

    @property
    def trim_id(self) -> str:
        """Compatibility owner for integrations that expect one primary ID."""

        return self.trim_ids[0]


@dataclass(frozen=True)
class ArchitecturalTrimGeometry:
    """Complete procedural trim output for preview or GLB construction."""

    runs: tuple[ArchitecturalTrimRun, ...] = ()

    @property
    def parts(self) -> tuple[ArchitecturalTrimPart, ...]:
        return tuple(part for run in self.runs for part in run.parts)


@dataclass(frozen=True)
class ArchitecturalTrimEndpointMiter:
    """One canonical local-X cornice boundary and its optional miter shear."""

    boundary: str
    wall_point_world: tuple[float, float, float]
    world_shift_per_depth: tuple[float, float, float]
    local_x_shift_per_depth: float
    neighbor_trim_id: str | None = None

    def __post_init__(self) -> None:
        if self.boundary not in TRIM_MITER_BOUNDARIES:
            raise ValueError("Unknown architectural trim miter boundary.")
        wall_point = _normalize_vector3(self.wall_point_world, "miter wall point")
        world_shift = _normalize_vector3(
            self.world_shift_per_depth,
            "miter world shift",
        )
        local_shift = float(self.local_x_shift_per_depth)
        if not math.isfinite(local_shift):
            raise ValueError("Architectural trim miter shifts must be finite.")
        neighbor_trim_id = (
            None
            if self.neighbor_trim_id is None
            else str(self.neighbor_trim_id).strip().lower()
        )
        if neighbor_trim_id is not None and (
            _TRIM_ID_PATTERN.fullmatch(neighbor_trim_id) is None
        ):
            raise ValueError("Architectural trim miter neighbors require valid IDs.")
        if neighbor_trim_id is None and (
            abs(local_shift) > TRIM_GEOMETRY_EPSILON
            or np.linalg.norm(world_shift) > TRIM_GEOMETRY_EPSILON
        ):
            raise ValueError("Open architectural trim ends cannot have miter shifts.")
        object.__setattr__(self, "wall_point_world", wall_point)
        object.__setattr__(self, "world_shift_per_depth", world_shift)
        object.__setattr__(self, "local_x_shift_per_depth", local_shift)
        object.__setattr__(self, "neighbor_trim_id", neighbor_trim_id)

    @property
    def is_joined(self) -> bool:
        return self.neighbor_trim_id is not None

    def miter_point_world(self, profile_offset_meters: float) -> tuple[float, float, float]:
        """Return the procedural miter point for one wall-normal profile offset."""

        offset = float(profile_offset_meters)
        if not math.isfinite(offset):
            raise ValueError("Architectural trim profile offsets must be finite.")
        point = np.asarray(self.wall_point_world, dtype=float) + (
            np.asarray(self.world_shift_per_depth, dtype=float) * offset
        )
        return tuple(float(value) for value in point)


@dataclass(frozen=True)
class ArchitecturalTrimCorniceMiterDescriptor:
    """Miter shears for both canonical local-X boundaries of one cornice."""

    level_index: int
    trim_id: str
    minimum_x: ArchitecturalTrimEndpointMiter
    maximum_x: ArchitecturalTrimEndpointMiter

    def __post_init__(self) -> None:
        normalized_id = str(self.trim_id).strip().lower()
        if _TRIM_ID_PATTERN.fullmatch(normalized_id) is None:
            raise ValueError("Architectural trim miter descriptors require valid IDs.")
        if self.minimum_x.boundary != TRIM_MITER_BOUNDARY_MINIMUM_X:
            raise ValueError("The minimum-X cornice miter is assigned incorrectly.")
        if self.maximum_x.boundary != TRIM_MITER_BOUNDARY_MAXIMUM_X:
            raise ValueError("The maximum-X cornice miter is assigned incorrectly.")
        object.__setattr__(self, "level_index", int(self.level_index))
        object.__setattr__(self, "trim_id", normalized_id)


@dataclass(frozen=True)
class ArchitecturalTrimEditHandle:
    """One world-space scalar gizmo axis for a selected trim component."""

    handle_kind: str
    origin_world: tuple[float, float, float]
    axis_world: tuple[float, float, float]
    value_meters: float

    def __post_init__(self) -> None:
        if self.handle_kind not in TRIM_HANDLE_KINDS:
            raise ValueError("Unknown architectural trim edit handle.")
        origin = _normalize_vector3(self.origin_world, "handle origin")
        axis = np.asarray(_normalize_vector3(self.axis_world, "handle axis"))
        length = float(np.linalg.norm(axis))
        if length <= TRIM_GEOMETRY_EPSILON:
            raise ValueError("Architectural trim handle axes cannot be zero.")
        object.__setattr__(self, "origin_world", origin)
        object.__setattr__(
            self,
            "axis_world",
            tuple(float(value) for value in axis / length),
        )
        if not math.isfinite(float(self.value_meters)):
            raise ValueError("Architectural trim handle values must be finite.")


@dataclass(frozen=True)
class ArchitecturalTrimEditTarget:
    """Selected trim ownership and the gizmos that can edit its profile."""

    trim_id: str
    run_id: str
    semantic_ids: tuple[str, ...]
    kind: str
    level_index: int
    anchor_world: tuple[float, float, float]
    handles: tuple[ArchitecturalTrimEditHandle, ...]


# ### Stable identity helpers ###
def build_architectural_trim_surface_id(trim_id: str, part_kind: str) -> str:
    normalized_id = str(trim_id).strip().lower()
    normalized_part = str(part_kind).strip().lower()
    if _TRIM_ID_PATTERN.fullmatch(normalized_id) is None:
        raise ValueError("Architectural trim IDs must be 32-character UUID hex.")
    if normalized_part not in TRIM_PART_KINDS:
        raise ValueError(f"Unknown architectural trim part: {part_kind!r}.")
    return f"trim:{normalized_id}/part:{normalized_part}:wall"


def parse_architectural_trim_surface_id(
    surface_id: object,
) -> tuple[str, str] | None:
    match = _TRIM_SURFACE_ID_PATTERN.fullmatch(str(surface_id).strip().lower())
    if match is None:
        return None
    return match.group("trim_id"), match.group("part_kind")


def is_architectural_trim_surface_id(surface_id: object) -> bool:
    return parse_architectural_trim_surface_id(surface_id) is not None


# ### Placement lifecycle ###
def validate_architectural_trim_placement(
    level: LevelData,
    placement: ArchitecturalTrimData | ArchitecturalTrimPlacementRequest,
    wall_surfaces: Sequence[object] | None = None,
) -> ArchitecturalTrimData:
    """Validate wall ownership and 90-degree corner requirements."""

    trim = (
        placement.to_data()
        if isinstance(
            placement,
            ArchitecturalTrimPlacementRequest,
        )
        else placement
    )
    if not isinstance(level, LevelData) or not isinstance(
        trim,
        ArchitecturalTrimData,
    ):
        raise TypeError("Architectural trim placement requires level and trim data.")
    if any(
        not surface_id.startswith(f"level:{level.index}/")
        for surface_id in trim.wall_surface_ids
    ):
        raise ValueError("Architectural trim walls must belong to their owner level.")
    surfaces = _resolve_wall_surfaces((level,), wall_surfaces)
    selected = []
    for surface_id in trim.wall_surface_ids:
        surface = surfaces.get(surface_id)
        if surface is None:
            raise ValueError(f"Architectural trim wall does not exist: {surface_id!r}.")
        selected.append(surface)
    if trim.kind == TRIM_KIND_EDGING_STRIP:
        _validate_edging_corner(level, trim, selected)
    elif len(selected) > 1:
        _validate_linear_wall_chain(selected)
    return trim


def add_architectural_trim(
    level: LevelData,
    request: ArchitecturalTrimPlacementRequest | ArchitecturalTrimData,
    wall_surfaces: Sequence[object] | None = None,
) -> ArchitecturalTrimData:
    trim = validate_architectural_trim_placement(level, request, wall_surfaces)
    if any(existing.trim_id == trim.trim_id for existing in level.architectural_trims):
        raise ValueError("An architectural trim with this ID already exists.")
    if any(
        _architectural_trims_conflict(existing, trim)
        for existing in level.architectural_trims
    ):
        raise ValueError("This location already has the selected trim component.")
    level.architectural_trims.append(trim)
    return trim


def update_architectural_trim(
    level: LevelData,
    trim_id: str,
    *,
    kind: str | None = None,
    wall_surface_ids: Sequence[str] | None = None,
    corner_vertex_id: int | None | object = ...,
    width_meters: float | None = None,
    height_meters: float | None = None,
    depth_meters: float | None = None,
    corner_radius_meters: float | None = None,
    wall_surfaces: Sequence[object] | None = None,
) -> ArchitecturalTrimData:
    """Replace one trim atomically while retaining its stable identity."""

    index = _find_trim_index(level, trim_id)
    current = level.architectural_trims[index]
    next_corner_vertex_id = (
        current.corner_vertex_id if corner_vertex_id is ... else corner_vertex_id
    )
    candidate = ArchitecturalTrimData(
        trim_id=current.trim_id,
        kind=current.kind if kind is None else kind,
        wall_surface_ids=(
            current.wall_surface_ids
            if wall_surface_ids is None
            else tuple(wall_surface_ids)
        ),
        corner_vertex_id=next_corner_vertex_id,  # type: ignore[arg-type]
        width_meters=(current.width_meters if width_meters is None else width_meters),
        height_meters=(
            current.height_meters if height_meters is None else height_meters
        ),
        depth_meters=(current.depth_meters if depth_meters is None else depth_meters),
        corner_radius_meters=(
            current.corner_radius_meters
            if corner_radius_meters is None
            else corner_radius_meters
        ),
    )
    validate_architectural_trim_placement(level, candidate, wall_surfaces)
    if any(
        other_index != index and _architectural_trims_conflict(existing, candidate)
        for other_index, existing in enumerate(level.architectural_trims)
    ):
        raise ValueError("This location already has the selected trim component.")
    level.architectural_trims[index] = candidate
    return candidate


def remove_architectural_trim(
    level: LevelData,
    trim_id: str,
) -> ArchitecturalTrimData | None:
    normalized_id = str(trim_id).strip().lower()
    for index, trim in enumerate(level.architectural_trims):
        if trim.trim_id == normalized_id:
            return level.architectural_trims.pop(index)
    return None


def get_architectural_trim(
    level: LevelData,
    trim_id: str,
) -> ArchitecturalTrimData | None:
    normalized_id = str(trim_id).strip().lower()
    return next(
        (trim for trim in level.architectural_trims if trim.trim_id == normalized_id),
        None,
    )


def _architectural_trims_conflict(
    first: ArchitecturalTrimData,
    second: ArchitecturalTrimData,
) -> bool:
    """Return whether two same-kind components occupy the same anchor."""

    if first.kind != second.kind:
        return False
    if first.kind == TRIM_KIND_EDGING_STRIP:
        return first.corner_vertex_id == second.corner_vertex_id
    return bool(set(first.wall_surface_ids).intersection(second.wall_surface_ids))


# ### Geometry construction ###
@dataclass(frozen=True)
class _WallSpan:
    trim: ArchitecturalTrimData
    surface: object
    start: np.ndarray
    end: np.ndarray
    normal: np.ndarray
    minimum_z: float
    maximum_z: float

    @property
    def direction(self) -> np.ndarray:
        delta = self.end[:2] - self.start[:2]
        return delta / np.linalg.norm(delta)


def build_architectural_trim_geometry(
    levels: Sequence[LevelData],
    wall_surfaces: Sequence[object] | None = None,
) -> ArchitecturalTrimGeometry:
    """Build joined run meshes and stable selectable face groups."""

    level_sequence = tuple(levels)
    surface_by_id = _resolve_wall_surfaces(level_sequence, wall_surfaces)
    runs: list[ArchitecturalTrimRun] = []
    for level in level_sequence:
        valid_trims: list[ArchitecturalTrimData] = []
        for trim in level.architectural_trims:
            try:
                valid_trims.append(
                    validate_architectural_trim_placement(
                        level,
                        trim,
                        tuple(surface_by_id.values()),
                    )
                )
            except (TypeError, ValueError):
                continue
        edging_trims = [
            trim for trim in valid_trims if trim.kind == TRIM_KIND_EDGING_STRIP
        ]
        linear_trims = [
            trim for trim in valid_trims if trim.kind != TRIM_KIND_EDGING_STRIP
        ]
        runs.extend(_build_linear_trim_runs(level, linear_trims, surface_by_id))
        runs.extend(
            run
            for trim in edging_trims
            if (run := _build_edging_strip_run(level, trim, surface_by_id)) is not None
        )
    return ArchitecturalTrimGeometry(
        runs=tuple(sorted(runs, key=lambda run: (run.level_index, run.run_id)))
    )


def build_architectural_trim_parts(
    levels: Sequence[LevelData],
    wall_surfaces: Sequence[object] | None = None,
) -> tuple[ArchitecturalTrimPart, ...]:
    return build_architectural_trim_geometry(levels, wall_surfaces).parts


def build_cornice_miter_descriptors(
    levels: Sequence[LevelData],
    wall_surfaces: Sequence[object] | None = None,
) -> tuple[ArchitecturalTrimCorniceMiterDescriptor, ...]:
    """Describe exact procedural corner miters in each cornice's local frame.

    Generated cornices use X along their length and centered Y across their
    depth. A consumer can shear either local-X boundary by
    ``local_x_shift_per_depth * (y + depth / 2)``. Open ends report zero.
    """

    level_sequence = tuple(levels)
    surface_by_id = _resolve_wall_surfaces(level_sequence, wall_surfaces)
    descriptors: list[ArchitecturalTrimCorniceMiterDescriptor] = []
    for level in level_sequence:
        spans: list[_WallSpan] = []
        descriptor_span_indices: list[int] = []
        for trim in level.architectural_trims:
            if trim.kind != TRIM_KIND_CORNICE:
                continue
            try:
                validated = validate_architectural_trim_placement(
                    level,
                    trim,
                    tuple(surface_by_id.values()),
                )
            except (TypeError, ValueError):
                continue
            trim_spans = tuple(
                span
                for surface_id in validated.wall_surface_ids
                if (surface := surface_by_id.get(surface_id)) is not None
                if (span := _build_wall_span(validated, surface)) is not None
            )
            if len(trim_spans) == 1:
                descriptor_span_indices.append(len(spans))
            spans.extend(trim_spans)
        adjacency = _build_linear_span_adjacency(spans)
        for span_index in descriptor_span_indices:
            descriptors.append(
                _build_cornice_miter_descriptor(
                    level.index,
                    span_index,
                    spans,
                    adjacency,
                )
            )
    return tuple(
        sorted(
            descriptors,
            key=lambda descriptor: (descriptor.level_index, descriptor.trim_id),
        )
    )


def build_architectural_trim_placement_preview_meshes(
    request: ArchitecturalTrimPlacementRequest,
    wall_surfaces: Sequence[object],
) -> tuple[trimesh.Trimesh, ...]:
    """Build a placement preview with the same profiles as persisted trims."""

    if not isinstance(request, ArchitecturalTrimPlacementRequest):
        raise TypeError("Architectural trim previews require a placement request.")
    trim = request.to_data()
    surface_by_id = {
        str(getattr(surface, "surface_id", "")): surface for surface in wall_surfaces
    }
    spans = tuple(
        span
        for surface_id in trim.wall_surface_ids
        if (surface := surface_by_id.get(surface_id)) is not None
        if (span := _build_wall_span(trim, surface)) is not None
    )
    if len(spans) != len(trim.wall_surface_ids):
        return ()
    level_index = int(getattr(spans[0].surface, "level_index", 0))
    if trim.kind == TRIM_KIND_EDGING_STRIP:
        if len(spans) != 2:
            return ()
        result = _build_edging_strip_mesh(trim, spans[0], spans[1])
        return () if result is None else (result[0],)
    run = _build_swept_linear_run(level_index, spans)
    return () if run is None else (run.mesh,)


def build_architectural_trim_edit_targets(
    levels: Sequence[LevelData],
    wall_surfaces: Sequence[object] | None = None,
) -> tuple[ArchitecturalTrimEditTarget, ...]:
    """Build one gizmo target per persisted trim, independent of run joining."""

    surface_by_id = _resolve_wall_surfaces(levels, wall_surfaces)
    geometry = build_architectural_trim_geometry(
        levels,
        tuple(surface_by_id.values()),
    )
    run_by_trim_id = {trim_id: run for run in geometry.runs for trim_id in run.trim_ids}
    targets: list[ArchitecturalTrimEditTarget] = []
    for level in levels:
        for trim in level.architectural_trims:
            run = run_by_trim_id.get(trim.trim_id)
            if run is None:
                continue
            owned_parts = tuple(
                part for part in run.parts if part.trim_id == trim.trim_id
            )
            if not owned_parts:
                continue
            bounds = np.asarray(
                trimesh.util.concatenate([part.mesh for part in owned_parts]).bounds,
                dtype=float,
            )
            anchor = (bounds[0] + bounds[1]) * 0.5
            extent = np.maximum(bounds[1] - bounds[0], TRIM_GEOMETRY_EPSILON)
            width_axis, depth_axis = _build_trim_handle_horizontal_axes(
                trim,
                surface_by_id,
                fallback_extent=extent,
            )
            height_axis = np.asarray((0.0, 0.0, 1.0), dtype=float)
            height_origin = anchor.copy()
            height_origin[2] = bounds[1, 2]
            if trim.kind == TRIM_KIND_CORNICE:
                height_axis *= -1.0
                height_origin[2] = bounds[0, 2]
            handles = [
                ArchitecturalTrimEditHandle(
                    handle_kind=TRIM_HANDLE_HEIGHT,
                    origin_world=tuple(height_origin),
                    axis_world=tuple(height_axis),
                    value_meters=float(extent[2]),
                ),
                ArchitecturalTrimEditHandle(
                    handle_kind=TRIM_HANDLE_DEPTH,
                    origin_world=tuple(anchor),
                    axis_world=depth_axis,
                    value_meters=trim.depth_meters,
                ),
            ]
            if trim.kind == TRIM_KIND_EDGING_STRIP:
                handles.insert(
                    0,
                    ArchitecturalTrimEditHandle(
                        handle_kind=TRIM_HANDLE_WIDTH,
                        origin_world=tuple(anchor),
                        axis_world=width_axis,
                        value_meters=trim.width_meters,
                    ),
                )
            else:
                handles.append(
                    ArchitecturalTrimEditHandle(
                        handle_kind=TRIM_HANDLE_CORNER_RADIUS,
                        origin_world=tuple(anchor),
                        axis_world=(0.0, 0.0, 1.0),
                        value_meters=trim.corner_radius_meters,
                    )
                )
            targets.append(
                ArchitecturalTrimEditTarget(
                    trim_id=trim.trim_id,
                    run_id=run.run_id,
                    semantic_ids=tuple(part.semantic_id for part in owned_parts),
                    kind=trim.kind,
                    level_index=level.index,
                    anchor_world=tuple(float(value) for value in anchor),
                    handles=tuple(handles),
                )
            )
    return tuple(targets)


def _build_trim_handle_horizontal_axes(
    trim: ArchitecturalTrimData,
    surface_by_id: Mapping[str, object],
    *,
    fallback_extent: np.ndarray,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return profile-local width and depth axes for rotated trim geometry."""

    spans = tuple(
        span
        for surface_id in trim.wall_surface_ids
        if (surface := surface_by_id.get(surface_id)) is not None
        if (
            span := _build_wall_span(
                trim,
                surface,
            )
        )
        is not None
    )
    if spans:
        if trim.kind == TRIM_KIND_EDGING_STRIP and len(spans) == 2:
            depth_axis = spans[0].normal[:2] + spans[1].normal[:2]
            depth_length = float(np.linalg.norm(depth_axis))
            if depth_length > TRIM_GEOMETRY_EPSILON:
                depth_axis /= depth_length
                width_axis = np.asarray(
                    (-depth_axis[1], depth_axis[0]),
                    dtype=float,
                )
                return (
                    (float(width_axis[0]), float(width_axis[1]), 0.0),
                    (float(depth_axis[0]), float(depth_axis[1]), 0.0),
                )
        depth_axis = spans[0].normal[:2]
        width_axis = spans[0].direction
        return (
            (float(width_axis[0]), float(width_axis[1]), 0.0),
            (float(depth_axis[0]), float(depth_axis[1]), 0.0),
        )

    axes = np.eye(3, dtype=float)
    return (
        tuple(axes[int(np.argmax(fallback_extent[:2]))]),
        tuple(axes[int(np.argmin(fallback_extent[:2]))]),
    )


def _build_linear_trim_runs(
    level: LevelData,
    trims: Sequence[ArchitecturalTrimData],
    surface_by_id: Mapping[str, object],
) -> list[ArchitecturalTrimRun]:
    spans: list[_WallSpan] = []
    for trim in trims:
        for surface_id in trim.wall_surface_ids:
            span = _build_wall_span(trim, surface_by_id[surface_id])
            if span is not None:
                spans.extend(_split_span_at_base_openings(span))
    if not spans:
        return []

    adjacency = _build_linear_span_adjacency(spans)
    runs: list[ArchitecturalTrimRun] = []
    unseen = set(range(len(spans)))
    while unseen:
        seed = min(unseen)
        component: set[int] = set()
        pending = [seed]
        while pending:
            index = pending.pop()
            if index in component:
                continue
            component.add(index)
            pending.extend(adjacency[index] - component)
        unseen.difference_update(component)
        ordered_indices = _order_span_component(component, adjacency, spans)
        run = _build_swept_linear_run(
            level.index,
            [spans[index] for index in ordered_indices],
        )
        if run is not None:
            runs.append(run)
    return _merge_runs_with_shared_trim_owners(runs)


def _build_linear_span_adjacency(
    spans: Sequence[_WallSpan],
) -> dict[int, set[int]]:
    """Resolve the same safe, style-compatible joins used by procedural runs."""

    adjacency: dict[int, set[int]] = {index: set() for index in range(len(spans))}
    grouped_by_style: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for index, span in enumerate(spans):
        grouped_by_style[_trim_style_signature(span.trim)].append(index)
    for indices in grouped_by_style.values():
        endpoint_incidence: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index in indices:
            span = spans[index]
            endpoint_incidence[_point_key(span.start[:2])].append(index)
            endpoint_incidence[_point_key(span.end[:2])].append(index)
        for incident_indices in endpoint_incidence.values():
            unique_indices = tuple(dict.fromkeys(incident_indices))
            for first_index, second_index in _select_safe_incident_joins(
                unique_indices,
                spans,
            ):
                adjacency[first_index].add(second_index)
                adjacency[second_index].add(first_index)
    return adjacency


def _build_cornice_miter_descriptor(
    level_index: int,
    span_index: int,
    spans: Sequence[_WallSpan],
    adjacency: Mapping[int, set[int]],
) -> ArchitecturalTrimCorniceMiterDescriptor:
    span = spans[span_index]
    width_axis = np.cross(span.normal, np.asarray((0.0, 0.0, 1.0), dtype=float))
    width_axis /= np.linalg.norm(width_axis)
    ordered_endpoints = (span.start, span.end)
    if float(np.dot(span.start[:2], width_axis[:2])) > float(
        np.dot(span.end[:2], width_axis[:2])
    ):
        ordered_endpoints = tuple(reversed(ordered_endpoints))
    endpoint_data = (
        (ordered_endpoints[0], TRIM_MITER_BOUNDARY_MINIMUM_X),
        (ordered_endpoints[1], TRIM_MITER_BOUNDARY_MAXIMUM_X),
    )
    endpoints = tuple(
        _build_cornice_endpoint_miter(
            span_index,
            point,
            boundary,
            spans,
            adjacency,
            width_axis,
        )
        for point, boundary in endpoint_data
    )
    minimum_x, maximum_x = endpoints
    return ArchitecturalTrimCorniceMiterDescriptor(
        level_index=level_index,
        trim_id=span.trim.trim_id,
        minimum_x=minimum_x,
        maximum_x=maximum_x,
    )


def _build_cornice_endpoint_miter(
    span_index: int,
    endpoint: np.ndarray,
    boundary: str,
    spans: Sequence[_WallSpan],
    adjacency: Mapping[int, set[int]],
    width_axis: np.ndarray,
) -> ArchitecturalTrimEndpointMiter:
    span = spans[span_index]
    wall_point = np.asarray(
        (endpoint[0], endpoint[1], span.maximum_z),
        dtype=float,
    )
    joined_index = next(
        (
            candidate_index
            for candidate_index in sorted(adjacency[span_index])
            if (
                shared_point := _shared_span_point(
                    span,
                    spans[candidate_index],
                )
            )
            is not None
            and np.linalg.norm(shared_point - endpoint[:2])
            <= TRIM_ENDPOINT_TOLERANCE_METERS
        ),
        None,
    )
    if joined_index is None:
        return ArchitecturalTrimEndpointMiter(
            boundary=boundary,
            wall_point_world=tuple(float(value) for value in wall_point),
            world_shift_per_depth=(0.0, 0.0, 0.0),
            local_x_shift_per_depth=0.0,
        )

    joined_span = spans[joined_index]
    shared_point = _shared_span_point(span, joined_span)
    assert shared_point is not None
    unit_miter_point = _mitered_offset_point(
        shared_point,
        span,
        joined_span,
        1.0,
    )
    if unit_miter_point is None:
        return ArchitecturalTrimEndpointMiter(
            boundary=boundary,
            wall_point_world=tuple(float(value) for value in wall_point),
            world_shift_per_depth=(0.0, 0.0, 0.0),
            local_x_shift_per_depth=0.0,
        )
    world_shift = np.asarray(
        (
            unit_miter_point[0] - shared_point[0],
            unit_miter_point[1] - shared_point[1],
            0.0,
        ),
        dtype=float,
    )
    return ArchitecturalTrimEndpointMiter(
        boundary=boundary,
        wall_point_world=(
            float(shared_point[0]),
            float(shared_point[1]),
            float(span.maximum_z),
        ),
        world_shift_per_depth=tuple(float(value) for value in world_shift),
        local_x_shift_per_depth=float(np.dot(world_shift, width_axis)),
        neighbor_trim_id=joined_span.trim.trim_id,
    )


def _select_safe_incident_joins(
    incident_indices: Sequence[int],
    spans: Sequence[_WallSpan],
) -> tuple[tuple[int, int], ...]:
    """Select only unambiguous pairs at a multi-wall junction."""

    remaining = set(incident_indices)
    selected: list[tuple[int, int]] = []
    while len(remaining) >= 2:
        candidates = {
            index: {
                other
                for other in remaining - {index}
                if _wall_spans_form_join(spans[index], spans[other])
            }
            for index in remaining
        }
        forced_pair = next(
            (
                tuple(sorted((index, next(iter(neighbors)))))
                for index, neighbors in sorted(candidates.items())
                if len(neighbors) == 1 and len(candidates[next(iter(neighbors))]) == 1
            ),
            None,
        )
        if forced_pair is None:
            break
        first_index, second_index = forced_pair
        selected.append((first_index, second_index))
        remaining.difference_update(forced_pair)
    return tuple(selected)


def _split_span_at_base_openings(span: _WallSpan) -> tuple[_WallSpan, ...]:
    """Split skirting at real holes that reach the host wall's base."""

    if span.trim.kind != TRIM_KIND_SKIRTING_BOARD:
        return (span,)
    intervals = _get_wall_base_visible_intervals(span)
    if intervals is None:
        return (span,)
    direction = span.direction
    result: list[_WallSpan] = []
    for start_distance, end_distance in intervals:
        if end_distance - start_distance <= TRIM_ENDPOINT_TOLERANCE_METERS:
            continue
        segment_start = span.start.copy()
        segment_end = span.start.copy()
        segment_start[:2] += direction * start_distance
        segment_end[:2] += direction * end_distance
        result.append(replace(span, start=segment_start, end=segment_end))
    return tuple(result)


def _get_wall_base_visible_intervals(
    span: _WallSpan,
) -> tuple[tuple[float, float], ...] | None:
    """Read visible wall intervals from a horizontal slice of its real mesh."""

    vertices = np.asarray(span.surface.mesh.vertices, dtype=float)  # type: ignore[attr-defined]
    faces = np.asarray(span.surface.mesh.faces, dtype=np.int64)  # type: ignore[attr-defined]
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or faces.ndim != 2
        or faces.shape[1:] != (3,)
    ):
        return None
    available_height = span.maximum_z - span.minimum_z
    probe_offset = min(
        max(span.trim.height_meters * 0.37, TRIM_ENDPOINT_TOLERANCE_METERS * 2.0),
        available_height * 0.5,
    )
    if probe_offset <= TRIM_GEOMETRY_EPSILON:
        return None
    probe_z = span.minimum_z + probe_offset
    wall_length = float(np.linalg.norm(span.end[:2] - span.start[:2]))
    plane_tolerance = TRIM_ENDPOINT_TOLERANCE_METERS * 2.0
    intervals: list[tuple[float, float]] = []
    has_primary_wall_faces = False
    for face in faces:
        triangle = vertices[face]
        plane_distances = np.dot(
            triangle[:, :2] - span.start[:2],
            span.normal[:2],
        )
        if np.max(np.abs(plane_distances)) > plane_tolerance:
            continue
        projected = np.column_stack(
            (
                np.dot(triangle[:, :2] - span.start[:2], span.direction),
                triangle[:, 2],
            )
        )
        first_edge = projected[1] - projected[0]
        second_edge = projected[2] - projected[0]
        projected_area = first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0]
        if abs(float(projected_area)) <= TRIM_GEOMETRY_EPSILON:
            continue
        has_primary_wall_faces = True
        intersection = _intersect_projected_triangle_at_height(projected, probe_z)
        if intersection is None:
            continue
        start_distance, end_distance = intersection
        start_distance = max(0.0, min(wall_length, start_distance))
        end_distance = max(0.0, min(wall_length, end_distance))
        if end_distance - start_distance > TRIM_GEOMETRY_EPSILON:
            intervals.append((start_distance, end_distance))
    if not has_primary_wall_faces:
        return None
    return _merge_linear_intervals(intervals)


def _intersect_projected_triangle_at_height(
    triangle: np.ndarray,
    height: float,
) -> tuple[float, float] | None:
    intersections: list[float] = []
    for start_index, end_index in ((0, 1), (1, 2), (2, 0)):
        start_u, start_z = triangle[start_index]
        end_u, end_z = triangle[end_index]
        if math.isclose(start_z, end_z, abs_tol=TRIM_GEOMETRY_EPSILON):
            if math.isclose(start_z, height, abs_tol=TRIM_GEOMETRY_EPSILON):
                intersections.extend((float(start_u), float(end_u)))
            continue
        ratio = (height - start_z) / (end_z - start_z)
        if -TRIM_GEOMETRY_EPSILON <= ratio <= 1.0 + TRIM_GEOMETRY_EPSILON:
            intersections.append(float(start_u + (end_u - start_u) * ratio))
    if len(intersections) < 2:
        return None
    return min(intersections), max(intersections)


def _merge_linear_intervals(
    intervals: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1] + TRIM_ENDPOINT_TOLERANCE_METERS:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return tuple((start, end) for start, end in merged)


def _merge_runs_with_shared_trim_owners(
    runs: Sequence[ArchitecturalTrimRun],
) -> list[ArchitecturalTrimRun]:
    """Keep clipped pieces owned by one trim in one logical export run."""

    unseen = set(range(len(runs)))
    merged_runs: list[ArchitecturalTrimRun] = []
    while unseen:
        seed = min(unseen)
        component = {seed}
        owner_ids = set(runs[seed].trim_ids)
        changed = True
        while changed:
            changed = False
            for index in sorted(unseen - component):
                if not owner_ids.intersection(runs[index].trim_ids):
                    continue
                component.add(index)
                owner_ids.update(runs[index].trim_ids)
                changed = True
        unseen.difference_update(component)
        selected = tuple(runs[index] for index in sorted(component))
        merged_runs.append(
            selected[0]
            if len(selected) == 1
            else _merge_architectural_trim_run_group(selected)
        )
    return merged_runs


def _merge_architectural_trim_run_group(
    runs: Sequence[ArchitecturalTrimRun],
) -> ArchitecturalTrimRun:
    trim_ids = tuple(dict.fromkeys(trim_id for run in runs for trim_id in run.trim_ids))
    run_id = trim_ids[0]
    mesh = trimesh.util.concatenate(tuple(run.mesh for run in runs))
    mesh.metadata.update(
        {
            "housemaker_trim_run_id": run_id,
            "housemaker_trim_ids": trim_ids,
        }
    )
    grouped_parts: dict[tuple[str, str], list[ArchitecturalTrimPart]] = defaultdict(
        list
    )
    for run in runs:
        for part in run.parts:
            grouped_parts[(part.trim_id, part.part_kind)].append(part)
    parts: list[ArchitecturalTrimPart] = []
    for (trim_id, part_kind), source_parts in sorted(grouped_parts.items()):
        part_mesh = trimesh.util.concatenate(tuple(part.mesh for part in source_parts))
        semantic_id = build_architectural_trim_surface_id(trim_id, part_kind)
        part_mesh.metadata.update(
            {
                "housemaker_trim_id": trim_id,
                "housemaker_trim_run_id": run_id,
                "housemaker_trim_part": part_kind,
                "housemaker_surface_id": semantic_id,
            }
        )
        source_ids = {
            part.source_wall_surface_id
            for part in source_parts
            if part.source_wall_surface_id is not None
        }
        parts.append(
            ArchitecturalTrimPart(
                trim_id=trim_id,
                run_id=run_id,
                semantic_id=semantic_id,
                part_kind=part_kind,
                surface_type=TRIM_SURFACE_TYPE,
                mesh=part_mesh,
                level_index=runs[0].level_index,
                source_wall_surface_id=(
                    next(iter(source_ids)) if len(source_ids) == 1 else None
                ),
            )
        )
    return ArchitecturalTrimRun(
        run_id=run_id,
        trim_ids=trim_ids,
        kind=runs[0].kind,
        level_index=runs[0].level_index,
        mesh=mesh,
        parts=tuple(parts),
    )


def _build_swept_linear_run(
    level_index: int,
    spans: Sequence[_WallSpan],
) -> ArchitecturalTrimRun | None:
    if not spans:
        return None
    oriented = _orient_span_chain(spans)
    if not oriented:
        return None
    is_closed = bool(
        len(oriented) > 2
        and np.linalg.norm(oriented[-1].end[:2] - oriented[0].start[:2])
        <= TRIM_ENDPOINT_TOLERANCE_METERS
    )
    effective_height = min(
        oriented[0].trim.height_meters,
        *(span.maximum_z - span.minimum_z for span in oriented),
    )
    if effective_height <= TRIM_GEOMETRY_EPSILON:
        return None
    source_trim = oriented[0].trim
    effective_radius = min(
        source_trim.corner_radius_meters,
        min(source_trim.depth_meters, effective_height) * 0.5,
    )
    profile = _build_linear_profile(
        replace(
            source_trim,
            height_meters=effective_height,
            corner_radius_meters=effective_radius,
        )
    )
    ring_positions = _build_swept_ring_positions(
        oriented,
        profile,
        is_closed=is_closed,
        height_meters=effective_height,
    )
    if ring_positions is None:
        return None

    vertices = np.asarray(
        [position for ring in ring_positions for position in ring],
        dtype=float,
    )
    ring_size = len(profile)
    faces: list[tuple[int, int, int]] = []
    face_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for span_index, span in enumerate(oriented):
        start_offset = span_index * ring_size
        end_offset = (
            ((span_index + 1) % len(ring_positions)) * ring_size
            if is_closed
            else (span_index + 1) * ring_size
        )
        for profile_index, (_offset, _height, part_kind) in enumerate(profile):
            next_index = (profile_index + 1) % ring_size
            a = start_offset + profile_index
            b = end_offset + profile_index
            c = end_offset + next_index
            d = start_offset + next_index
            group_key = (span.trim.trim_id, part_kind)
            next_offset, next_height, _next_part_kind = profile[next_index]
            expected_normal = span.normal * (next_height - _height)
            expected_normal[2] -= next_offset - _offset
            face_groups[group_key].append(len(faces))
            faces.append(
                _orient_triangle_to_normal(
                    vertices,
                    (a, b, c),
                    expected_normal,
                )
            )
            face_groups[group_key].append(len(faces))
            faces.append(
                _orient_triangle_to_normal(
                    vertices,
                    (a, c, d),
                    expected_normal,
                )
            )
    if not is_closed:
        _append_profile_cap(
            vertices,
            faces,
            face_groups,
            owner_trim_id=oriented[0].trim.trim_id,
            ring_start=0,
            ring_size=ring_size,
            expected_normal=np.asarray(
                (-oriented[0].direction[0], -oriented[0].direction[1], 0.0),
                dtype=float,
            ),
        )
        _append_profile_cap(
            vertices,
            faces,
            face_groups,
            owner_trim_id=oriented[-1].trim.trim_id,
            ring_start=len(oriented) * ring_size,
            ring_size=ring_size,
            expected_normal=np.asarray(
                (oriented[-1].direction[0], oriented[-1].direction[1], 0.0),
                dtype=float,
            ),
        )
    oriented_faces = _orient_faces_consistently_outward(
        vertices,
        np.asarray(faces, dtype=np.int64),
    )
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=oriented_faces,
        process=False,
    )
    trim_ids = tuple(dict.fromkeys(span.trim.trim_id for span in oriented))
    run_id = trim_ids[0]
    mesh.metadata.update(
        {
            "housemaker_trim_run_id": run_id,
            "housemaker_trim_ids": trim_ids,
        }
    )
    source_ids_by_trim: dict[str, set[str]] = defaultdict(set)
    for span in oriented:
        source_ids_by_trim[span.trim.trim_id].add(
            str(span.surface.surface_id)  # type: ignore[attr-defined]
        )
    parts = _build_parts_from_face_groups(
        mesh,
        face_groups,
        run_id=run_id,
        level_index=level_index,
        source_ids_by_trim=source_ids_by_trim,
    )
    return ArchitecturalTrimRun(
        run_id=run_id,
        trim_ids=trim_ids,
        kind=oriented[0].trim.kind,
        level_index=level_index,
        mesh=mesh,
        parts=parts,
    )


def _build_edging_strip_run(
    level: LevelData,
    trim: ArchitecturalTrimData,
    surface_by_id: Mapping[str, object],
) -> ArchitecturalTrimRun | None:
    spans = [
        _build_wall_span(trim, surface_by_id[surface_id])
        for surface_id in trim.wall_surface_ids
    ]
    if any(span is None for span in spans):
        return None
    first, second = spans
    assert first is not None and second is not None
    result = _build_edging_strip_mesh(trim, first, second)
    if result is None:
        return None
    mesh, face_groups = result
    mesh.metadata.update(
        {
            "housemaker_trim_run_id": trim.trim_id,
            "housemaker_trim_ids": (trim.trim_id,),
        }
    )
    source_ids = set(trim.wall_surface_ids)
    parts = _build_parts_from_face_groups(
        mesh,
        face_groups,
        run_id=trim.trim_id,
        level_index=level.index,
        source_ids_by_trim={trim.trim_id: source_ids},
    )
    return ArchitecturalTrimRun(
        run_id=trim.trim_id,
        trim_ids=(trim.trim_id,),
        kind=trim.kind,
        level_index=level.index,
        mesh=mesh,
        parts=parts,
    )


def _build_edging_strip_mesh(
    trim: ArchitecturalTrimData,
    first: _WallSpan,
    second: _WallSpan,
) -> (
    tuple[
        trimesh.Trimesh,
        dict[tuple[str, str], list[int]],
    ]
    | None
):
    """Extrude an equal-leg L profile around a perpendicular wall corner."""

    corner = _shared_span_point(first, second)
    if corner is None:
        return None
    first_away = _span_direction_away_from_point(first, corner)
    second_away = _span_direction_away_from_point(second, corner)
    if first_away is None or second_away is None:
        return None
    base_z = max(first.minimum_z, second.minimum_z)
    available_height = min(first.maximum_z, second.maximum_z) - base_z
    height = min(trim.height_meters, available_height)
    if height <= TRIM_GEOMETRY_EPSILON:
        return None

    first_width = min(
        trim.width_meters,
        float(np.linalg.norm(first.end[:2] - first.start[:2])),
    )
    second_width = min(
        trim.width_meters,
        float(np.linalg.norm(second.end[:2] - second.start[:2])),
    )
    first_offset = first.normal[:2] * trim.depth_meters
    second_offset = second.normal[:2] * trim.depth_meters
    footprint = np.asarray(
        (
            corner,
            corner + first_away * first_width,
            corner + first_away * first_width + first_offset,
            corner + first_offset,
            corner + first_offset + second_offset,
            corner + second_offset,
            corner + second_away * second_width + second_offset,
            corner + second_away * second_width,
        ),
        dtype=float,
    )
    if abs(_polygon_signed_area(footprint)) <= TRIM_GEOMETRY_EPSILON:
        return None

    bottom = np.column_stack((footprint, np.full(len(footprint), base_z, dtype=float)))
    top = np.column_stack(
        (footprint, np.full(len(footprint), base_z + height, dtype=float))
    )
    vertices = np.vstack((bottom, top))
    cap_triangles = (
        (0, 1, 2),
        (0, 2, 3),
        (0, 3, 4),
        (0, 4, 5),
        (0, 5, 6),
        (0, 6, 7),
    )
    faces: list[tuple[int, int, int]] = []
    face_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    side_face_indices: list[int] = []
    for triangle in cap_triangles:
        face_groups[(trim.trim_id, TRIM_PART_BOTTOM)].append(len(faces))
        faces.append(triangle[::-1])
        face_groups[(trim.trim_id, TRIM_PART_TOP)].append(len(faces))
        faces.append(tuple(index + len(footprint) for index in triangle))
    for index in range(len(footprint)):
        following = (index + 1) % len(footprint)
        bottom_start = index
        bottom_end = following
        top_start = index + len(footprint)
        top_end = following + len(footprint)
        side_face_indices.extend((len(faces), len(faces) + 1))
        faces.extend(
            (
                (bottom_start, bottom_end, top_end),
                (bottom_start, top_end, top_start),
            )
        )
    oriented_faces = _orient_faces_consistently_outward(
        vertices,
        np.asarray(faces, dtype=np.int64),
    )
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=oriented_faces,
        process=False,
    )
    for face_index in side_face_indices:
        normal = np.asarray(mesh.face_normals[face_index, :2], dtype=float)
        is_front = (
            max(
                float(np.dot(normal, first.normal[:2])),
                float(np.dot(normal, second.normal[:2])),
            )
            > 0.5
        )
        part_kind = TRIM_PART_FRONT if is_front else TRIM_PART_SIDES
        face_groups[(trim.trim_id, part_kind)].append(face_index)
    return mesh, face_groups


def _build_parts_from_face_groups(
    mesh: trimesh.Trimesh,
    face_groups: Mapping[tuple[str, str], Sequence[int]],
    *,
    run_id: str,
    level_index: int,
    source_ids_by_trim: Mapping[str, set[str]],
) -> tuple[ArchitecturalTrimPart, ...]:
    parts: list[ArchitecturalTrimPart] = []
    for (trim_id, part_kind), face_indices in sorted(face_groups.items()):
        if not face_indices:
            continue
        part_mesh = mesh.submesh(
            [np.asarray(face_indices, dtype=np.int64)],
            append=True,
            repair=False,
        )
        if not isinstance(part_mesh, trimesh.Trimesh) or not len(part_mesh.faces):
            continue
        semantic_id = build_architectural_trim_surface_id(trim_id, part_kind)
        part_mesh.metadata.update(
            {
                "housemaker_trim_id": trim_id,
                "housemaker_trim_run_id": run_id,
                "housemaker_trim_part": part_kind,
                "housemaker_surface_id": semantic_id,
            }
        )
        source_ids = source_ids_by_trim.get(trim_id, set())
        parts.append(
            ArchitecturalTrimPart(
                trim_id=trim_id,
                run_id=run_id,
                semantic_id=semantic_id,
                part_kind=part_kind,
                surface_type=TRIM_SURFACE_TYPE,
                mesh=part_mesh,
                level_index=level_index,
                source_wall_surface_id=(
                    next(iter(source_ids)) if len(source_ids) == 1 else None
                ),
            )
        )
    return tuple(parts)


# ### Linear sweep helpers ###
def _build_linear_profile(
    trim: ArchitecturalTrimData,
) -> tuple[tuple[float, float, str], ...]:
    if trim.kind == TRIM_KIND_CORNICE:
        return _build_cornice_profile(trim)
    return _build_skirting_profile(trim)


def _build_skirting_profile(
    trim: ArchitecturalTrimData,
) -> tuple[tuple[float, float, str], ...]:
    depth = trim.depth_meters
    height = trim.height_meters
    radius = min(trim.corner_radius_meters, depth, height)
    profile: list[tuple[float, float, str]] = [
        (0.0, 0.0, TRIM_PART_BOTTOM),
        (depth, 0.0, TRIM_PART_FRONT),
    ]
    if radius <= TRIM_GEOMETRY_EPSILON:
        profile.extend(
            (
                (depth, height, TRIM_PART_TOP),
                (0.0, height, TRIM_PART_SIDES),
            )
        )
        return tuple(profile)

    center_offset = depth - radius
    center_height = height - radius
    for segment_index in range(TRIM_ROUNDED_PROFILE_SEGMENTS + 1):
        angle = math.pi * 0.5 * segment_index / TRIM_ROUNDED_PROFILE_SEGMENTS
        profile.append(
            (
                center_offset + radius * math.cos(angle),
                center_height + radius * math.sin(angle),
                (
                    TRIM_PART_TOP
                    if segment_index == TRIM_ROUNDED_PROFILE_SEGMENTS
                    else TRIM_PART_FRONT
                ),
            )
        )
    profile.append((0.0, height, TRIM_PART_SIDES))
    return tuple(profile)


def _build_cornice_profile(
    trim: ArchitecturalTrimData,
) -> tuple[tuple[float, float, str], ...]:
    """Build an angled wall-to-ceiling cornice with optional curvature."""

    depth = trim.depth_meters
    height = trim.height_meters
    lower_lip = depth * 0.08
    profile: list[tuple[float, float, str]] = [
        (0.0, 0.0, TRIM_PART_BOTTOM),
        (lower_lip, 0.0, TRIM_PART_FRONT),
    ]
    radius = trim.corner_radius_meters
    if radius <= TRIM_GEOMETRY_EPSILON:
        profile.extend(
            (
                (depth, height, TRIM_PART_TOP),
                (0.0, height, TRIM_PART_SIDES),
            )
        )
        return tuple(profile)

    maximum_radius = min(depth, height) * 0.5
    roundness = min(1.0, radius / maximum_radius)
    for segment_index in range(1, TRIM_CORNICE_ROUNDED_PROFILE_SEGMENTS + 1):
        ratio = segment_index / TRIM_CORNICE_ROUNDED_PROFILE_SEGMENTS
        angle = math.pi * 0.5 * ratio
        straight_offset = lower_lip + (depth - lower_lip) * ratio
        straight_height = height * ratio
        rounded_offset = lower_lip + (depth - lower_lip) * math.sin(angle)
        rounded_height = height * (1.0 - math.cos(angle))
        profile.append(
            (
                straight_offset + (rounded_offset - straight_offset) * roundness,
                straight_height + (rounded_height - straight_height) * roundness,
                (
                    TRIM_PART_TOP
                    if segment_index == TRIM_CORNICE_ROUNDED_PROFILE_SEGMENTS
                    else TRIM_PART_FRONT
                ),
            )
        )
    profile.append((0.0, height, TRIM_PART_SIDES))
    return tuple(profile)


def _build_swept_ring_positions(
    spans: Sequence[_WallSpan],
    profile: Sequence[tuple[float, float, str]],
    *,
    is_closed: bool,
    height_meters: float,
) -> list[list[tuple[float, float, float]]] | None:
    if is_closed:
        return _build_closed_swept_ring_positions(
            spans,
            profile,
            height_meters=height_meters,
        )
    vertices = [spans[0].start, *(span.end for span in spans)]
    rings: list[list[tuple[float, float, float]]] = []
    for vertex_index, vertex in enumerate(vertices):
        ring: list[tuple[float, float, float]] = []
        for offset, local_height, _part_kind in profile:
            if vertex_index == 0:
                point_xy = vertex[:2] + spans[0].normal[:2] * offset
            elif vertex_index == len(vertices) - 1:
                point_xy = vertex[:2] + spans[-1].normal[:2] * offset
            else:
                point_xy = _mitered_offset_point(
                    vertex[:2],
                    spans[vertex_index - 1],
                    spans[vertex_index],
                    offset,
                )
                if point_xy is None:
                    return None
            base_z = (
                spans[min(vertex_index, len(spans) - 1)].minimum_z
                if spans[0].trim.kind == TRIM_KIND_SKIRTING_BOARD
                else spans[min(vertex_index, len(spans) - 1)].maximum_z - height_meters
            )
            ring.append((float(point_xy[0]), float(point_xy[1]), base_z + local_height))
        rings.append(ring)
    return rings


def _build_closed_swept_ring_positions(
    spans: Sequence[_WallSpan],
    profile: Sequence[tuple[float, float, str]],
    *,
    height_meters: float,
) -> list[list[tuple[float, float, float]]] | None:
    """Build one shared mitered ring at every vertex of a closed run."""

    rings: list[list[tuple[float, float, float]]] = []
    for span_index, following in enumerate(spans):
        previous = spans[(span_index - 1) % len(spans)]
        shared_point = following.start
        base_z = (
            following.minimum_z
            if following.trim.kind == TRIM_KIND_SKIRTING_BOARD
            else following.maximum_z - height_meters
        )
        ring: list[tuple[float, float, float]] = []
        for offset, local_height, _part_kind in profile:
            point_xy = _mitered_offset_point(
                shared_point[:2],
                previous,
                following,
                offset,
            )
            if point_xy is None:
                return None
            ring.append(
                (
                    float(point_xy[0]),
                    float(point_xy[1]),
                    base_z + local_height,
                )
            )
        rings.append(ring)
    return rings


def _mitered_offset_point(
    shared_point: np.ndarray,
    previous: _WallSpan,
    following: _WallSpan,
    offset: float,
) -> np.ndarray | None:
    if offset <= TRIM_GEOMETRY_EPSILON:
        return shared_point.copy()
    first_origin = shared_point + previous.normal[:2] * offset
    second_origin = shared_point + following.normal[:2] * offset
    return _line_intersection(
        first_origin,
        previous.direction,
        second_origin,
        following.direction,
    )


def _line_intersection(
    first_origin: np.ndarray,
    first_direction: np.ndarray,
    second_origin: np.ndarray,
    second_direction: np.ndarray,
) -> np.ndarray | None:
    matrix = np.column_stack((first_direction, -second_direction))
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) <= TRIM_GEOMETRY_EPSILON:
        return None
    parameters = np.linalg.solve(matrix, second_origin - first_origin)
    return first_origin + first_direction * float(parameters[0])


def _append_profile_cap(
    vertices: np.ndarray,
    faces: list[tuple[int, int, int]],
    face_groups: dict[tuple[str, str], list[int]],
    *,
    owner_trim_id: str,
    ring_start: int,
    ring_size: int,
    expected_normal: np.ndarray,
) -> None:
    group_key = (owner_trim_id, TRIM_PART_SIDES)
    for profile_index in range(1, ring_size - 1):
        face = _orient_triangle_to_normal(
            vertices,
            (
                ring_start,
                ring_start + profile_index,
                ring_start + profile_index + 1,
            ),
            expected_normal,
        )
        face_groups[group_key].append(len(faces))
        faces.append(face)


def _orient_triangle_to_normal(
    vertices: np.ndarray,
    face: tuple[int, int, int],
    expected_normal: np.ndarray,
) -> tuple[int, int, int]:
    first, second, third = vertices[np.asarray(face, dtype=np.int64)]
    actual_normal = np.cross(second - first, third - first)
    if float(np.dot(actual_normal, expected_normal)) < 0.0:
        return face[0], face[2], face[1]
    return face


def _orient_faces_consistently_outward(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """Orient disconnected watertight bodies without a graph dependency."""

    edge_incidence: dict[
        tuple[int, int],
        list[tuple[int, bool]],
    ] = defaultdict(list)
    face_edges: list[tuple[tuple[int, int], ...]] = []
    for face_index, face in enumerate(faces):
        edges: list[tuple[int, int]] = []
        for first, second in (
            (int(face[0]), int(face[1])),
            (int(face[1]), int(face[2])),
            (int(face[2]), int(face[0])),
        ):
            edge = (min(first, second), max(first, second))
            edge_incidence[edge].append((face_index, first < second))
            edges.append(edge)
        face_edges.append(tuple(edges))

    flips: dict[int, bool] = {}
    components: list[list[int]] = []
    for seed in range(len(faces)):
        if seed in flips:
            continue
        flips[seed] = False
        component: list[int] = []
        pending = [seed]
        while pending:
            face_index = pending.pop()
            component.append(face_index)
            incidence_by_edge = {
                edge: direction
                for edge in face_edges[face_index]
                for candidate, direction in edge_incidence[edge]
                if candidate == face_index
            }
            for edge, face_direction in incidence_by_edge.items():
                for neighbor, neighbor_direction in edge_incidence[edge]:
                    if neighbor == face_index:
                        continue
                    expected_flip = flips[face_index] ^ (
                        face_direction == neighbor_direction
                    )
                    if neighbor not in flips:
                        flips[neighbor] = expected_flip
                        pending.append(neighbor)
        components.append(component)

    oriented = faces.copy()
    for face_index, should_flip in flips.items():
        if should_flip:
            oriented[face_index] = oriented[face_index, ::-1]
    for component in components:
        component_faces = oriented[np.asarray(component, dtype=np.int64)]
        triangles = vertices[component_faces]
        signed_volume = float(
            np.sum(
                np.einsum(
                    "ij,ij->i",
                    triangles[:, 0],
                    np.cross(triangles[:, 1], triangles[:, 2]),
                )
            )
            / 6.0
        )
        if signed_volume < -TRIM_GEOMETRY_EPSILON:
            indices = np.asarray(component, dtype=np.int64)
            oriented[indices] = oriented[indices, ::-1]
    return oriented


# ### Wall topology helpers ###
def _resolve_wall_surfaces(
    levels: Sequence[LevelData],
    wall_surfaces: Sequence[object] | None,
) -> dict[str, object]:
    if wall_surfaces is None:
        from housemaker.surface_geometry import build_base_fixed_surfaces

        candidates = build_base_fixed_surfaces(levels)
    else:
        candidates = tuple(wall_surfaces)
    resolved: dict[str, object] = {}
    for surface in candidates:
        if str(getattr(surface, "surface_type", "")) != "wall":
            continue
        surface_id = str(getattr(surface, "surface_id", "")).strip()
        wall_start = getattr(surface, "wall_start_world", None)
        wall_end = getattr(surface, "wall_end_world", None)
        if surface_id and wall_start is not None and wall_end is not None:
            resolved[surface_id] = surface
    return resolved


def _build_wall_span(
    trim: ArchitecturalTrimData,
    surface: object,
) -> _WallSpan | None:
    try:
        start = np.asarray(surface.wall_start_world, dtype=float)  # type: ignore[attr-defined]
        end = np.asarray(surface.wall_end_world, dtype=float)  # type: ignore[attr-defined]
        mesh = surface.mesh  # type: ignore[attr-defined]
    except (TypeError, ValueError):
        return None
    if (
        start.shape != (3,)
        or end.shape != (3,)
        or not np.all(np.isfinite((start, end)))
        or not isinstance(mesh, trimesh.Trimesh)
        or float(np.linalg.norm(end[:2] - start[:2])) <= TRIM_GEOMETRY_EPSILON
    ):
        return None
    bounds = np.asarray(mesh.bounds, dtype=float)
    if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
        return None
    tangent = end[:2] - start[:2]
    tangent /= np.linalg.norm(tangent)
    normal = _get_wall_surface_normal(mesh, tangent)
    return _WallSpan(
        trim=trim,
        surface=surface,
        start=start,
        end=end,
        normal=np.asarray((normal[0], normal[1], 0.0), dtype=float),
        minimum_z=float(bounds[0, 2]),
        maximum_z=float(bounds[1, 2]),
    )


def _get_wall_surface_normal(
    mesh: trimesh.Trimesh,
    tangent: np.ndarray,
) -> np.ndarray:
    right_normal = np.asarray((tangent[1], -tangent[0]), dtype=float)
    normals = np.asarray(mesh.face_normals, dtype=float)
    areas = np.asarray(mesh.area_faces, dtype=float)
    horizontal_mask = np.abs(normals[:, 2]) <= 0.25
    alignment = np.dot(normals[:, :2], right_normal)
    primary_mask = horizontal_mask & (np.abs(alignment) >= 0.75)
    if np.any(primary_mask):
        orientation_score = float(np.sum(alignment[primary_mask] * areas[primary_mask]))
        if orientation_score < -TRIM_GEOMETRY_EPSILON:
            return -right_normal
    return right_normal


def _validate_linear_wall_chain(surfaces: Sequence[object]) -> None:
    frames = []
    validation_trim = ArchitecturalTrimData(
        trim_id="0" * 32,
        kind=TRIM_KIND_SKIRTING_BOARD,
        wall_surface_ids=tuple(
            str(item.surface_id)  # type: ignore[attr-defined]
            for item in surfaces
        ),
    )
    for surface in surfaces:
        span = _build_wall_span(validation_trim, surface)
        if span is None:
            raise ValueError("Architectural trim wall geometry is invalid.")
        frames.append(span)
    adjacency = {index: set() for index in range(len(frames))}
    for first_index in range(len(frames)):
        for second_index in range(first_index + 1, len(frames)):
            if _wall_spans_form_join(frames[first_index], frames[second_index]):
                adjacency[first_index].add(second_index)
                adjacency[second_index].add(first_index)
    if any(len(neighbors) > 2 for neighbors in adjacency.values()):
        raise ValueError("Architectural trim wall chains cannot branch.")
    visited: set[int] = set()
    pending = [0]
    while pending:
        index = pending.pop()
        if index in visited:
            continue
        visited.add(index)
        pending.extend(adjacency[index] - visited)
    if len(visited) != len(frames):
        raise ValueError(
            "Architectural trim walls must form one connected 90-degree chain."
        )


def _validate_edging_corner(
    level: LevelData,
    trim: ArchitecturalTrimData,
    surfaces: Sequence[object],
) -> None:
    if trim.corner_vertex_id is None:
        raise ValueError("Edging strips require a shared corner vertex.")
    vertex = level.vertex_data.get_vertex(trim.corner_vertex_id)
    if vertex is None:
        raise ValueError("The edging-strip corner vertex does not exist.")
    spans = [_build_wall_span(trim, surface) for surface in surfaces]
    if any(span is None for span in spans):
        raise ValueError("Edging-strip wall geometry is invalid.")
    first, second = spans
    assert first is not None and second is not None
    shared = _shared_span_point(first, second)
    if shared is None or not _directions_are_perpendicular(
        first.direction,
        second.direction,
    ):
        raise ValueError("Edging strips require a shared 90-degree wall corner.")


def _wall_spans_form_join(first: _WallSpan, second: _WallSpan) -> bool:
    if _shared_span_point(first, second) is None:
        return False
    if not _directions_are_perpendicular(first.direction, second.direction):
        return False
    if not _wall_normals_form_continuous_side(first, second):
        return False
    return math.isclose(
        first.minimum_z,
        second.minimum_z,
        abs_tol=TRIM_ENDPOINT_TOLERANCE_METERS,
    ) and math.isclose(
        first.maximum_z,
        second.maximum_z,
        abs_tol=TRIM_ENDPOINT_TOLERANCE_METERS,
    )


def _wall_normals_form_continuous_side(
    first: _WallSpan,
    second: _WallSpan,
) -> bool:
    """Reject corner joins whose host faces occupy opposing path sides."""

    shared = _shared_span_point(first, second)
    if shared is None:
        return False
    first_away = (
        first.direction
        if np.linalg.norm(first.start[:2] - shared) <= TRIM_ENDPOINT_TOLERANCE_METERS
        else -first.direction
    )
    second_away = (
        second.direction
        if np.linalg.norm(second.start[:2] - shared) <= TRIM_ENDPOINT_TOLERANCE_METERS
        else -second.direction
    )
    first_side = _cross_2d(-first_away, first.normal[:2])
    second_side = _cross_2d(second_away, second.normal[:2])
    return first_side * second_side > TRIM_GEOMETRY_EPSILON


def _cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _polygon_signed_area(points: np.ndarray) -> float:
    following = np.roll(points, -1, axis=0)
    return float(
        0.5 * np.sum(points[:, 0] * following[:, 1] - points[:, 1] * following[:, 0])
    )


def _span_direction_away_from_point(
    span: _WallSpan,
    point: np.ndarray,
) -> np.ndarray | None:
    if np.linalg.norm(span.start[:2] - point) <= TRIM_ENDPOINT_TOLERANCE_METERS:
        return span.direction
    if np.linalg.norm(span.end[:2] - point) <= TRIM_ENDPOINT_TOLERANCE_METERS:
        return -span.direction
    return None


def _directions_are_perpendicular(
    first: np.ndarray,
    second: np.ndarray,
) -> bool:
    angle_error = abs(float(np.dot(first, second)))
    return angle_error <= math.sin(math.radians(TRIM_CORNER_ANGLE_TOLERANCE_DEGREES))


def _shared_span_point(
    first: _WallSpan,
    second: _WallSpan,
) -> np.ndarray | None:
    for first_point in (first.start, first.end):
        for second_point in (second.start, second.end):
            if float(np.linalg.norm(first_point[:2] - second_point[:2])) <= (
                TRIM_ENDPOINT_TOLERANCE_METERS
            ):
                return (first_point[:2] + second_point[:2]) * 0.5
    return None


def _order_span_component(
    component: set[int],
    adjacency: Mapping[int, set[int]],
    spans: Sequence[_WallSpan],
) -> list[int]:
    endpoints = sorted(
        index for index in component if len(adjacency[index] & component) <= 1
    )
    current = endpoints[0] if endpoints else min(component)
    ordered: list[int] = []
    previous: int | None = None
    while current not in ordered:
        ordered.append(current)
        candidates = sorted((adjacency[current] & component) - {previous})
        if not candidates:
            break
        previous, current = current, candidates[0]
    if len(ordered) != len(component):
        ordered.extend(sorted(component - set(ordered)))
    return ordered


def _orient_span_chain(spans: Sequence[_WallSpan]) -> list[_WallSpan]:
    if len(spans) == 1:
        return list(spans)
    oriented: list[_WallSpan] = []
    first = spans[0]
    shared = _shared_span_point(first, spans[1])
    if shared is None:
        return []
    if np.linalg.norm(first.end[:2] - shared) > TRIM_ENDPOINT_TOLERANCE_METERS:
        first = replace(first, start=first.end, end=first.start)
    oriented.append(first)
    current_xy = first.end[:2]
    for span in spans[1:]:
        if np.linalg.norm(span.start[:2] - current_xy) <= (
            TRIM_ENDPOINT_TOLERANCE_METERS
        ):
            oriented_span = span
        elif np.linalg.norm(span.end[:2] - current_xy) <= (
            TRIM_ENDPOINT_TOLERANCE_METERS
        ):
            oriented_span = replace(span, start=span.end, end=span.start)
        else:
            return []
        oriented.append(oriented_span)
        current_xy = oriented_span.end[:2]
    return oriented


def _trim_style_signature(trim: ArchitecturalTrimData) -> tuple[object, ...]:
    return (
        trim.kind,
        round(trim.width_meters, 9),
        round(trim.height_meters, 9),
        round(trim.depth_meters, 9),
        round(trim.corner_radius_meters, 9),
    )


def _point_key(point: np.ndarray) -> tuple[int, int]:
    return tuple(
        round(float(value) / TRIM_ENDPOINT_TOLERANCE_METERS) for value in point
    )  # type: ignore[return-value]


# ### General validation helpers ###
def _find_trim_index(level: LevelData, trim_id: str) -> int:
    normalized_id = str(trim_id).strip().lower()
    for index, trim in enumerate(level.architectural_trims):
        if trim.trim_id == normalized_id:
            return index
    raise ValueError(f"Unknown architectural trim ID: {trim_id!r}.")


def _normalize_vector3(
    values: Sequence[float],
    field_name: str,
) -> tuple[float, float, float]:
    try:
        normalized = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"Architectural trim {field_name} must be a vector."
        ) from error
    if len(normalized) != 3 or not all(math.isfinite(value) for value in normalized):
        raise ValueError(f"Architectural trim {field_name} must be a finite vector.")
    return normalized  # type: ignore[return-value]
