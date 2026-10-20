# ### Imports ###
from __future__ import annotations

import math
import uuid
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
import shapely
import trimesh
from shapely import LineString, Polygon
from shapely.geometry.base import BaseGeometry

from housemaker.level_coordinates import build_level_base_z_lookup
from housemaker.models import (
    DEFAULT_SLOPED_CEILING_WALL_DROP_METERS,
    MAX_SLOPED_CEILING_DEPTH_METERS,
    MAX_SLOPED_CEILING_TRANSITION_RADIUS_METERS,
    MIN_SLOPED_CEILING_DEPTH_METERS,
    MIN_SLOPED_CEILING_RATIO_SPAN,
    MIN_SLOPED_CEILING_TRANSITION_RADIUS_METERS,
    LevelData,
    SlopedCeilingData,
    WindowData,
)

if TYPE_CHECKING:
    from housemaker.canvas_openings import CanvasOpeningTarget

# ### Constants ###
SLOPED_CEILING_HANDLE_CREATE_DEPTH = "create_depth"
SLOPED_CEILING_HANDLE_DEPTH = "depth"
SLOPED_CEILING_HANDLE_WALL_DROP = "wall_drop"
SLOPED_CEILING_HANDLE_ANGLE = SLOPED_CEILING_HANDLE_WALL_DROP
SLOPED_CEILING_HANDLE_CEILING_TRANSITION_RADIUS = (
    "ceiling_transition_radius"
)
SLOPED_CEILING_HANDLE_ROUNDING = (
    SLOPED_CEILING_HANDLE_CEILING_TRANSITION_RADIUS
)
SLOPED_CEILING_HANDLE_TRANSITION_RADIUS = SLOPED_CEILING_HANDLE_ROUNDING
SLOPED_CEILING_HANDLE_WALL_TRANSITION_RADIUS = "wall_transition_radius"
SLOPED_CEILING_HANDLE_SIDE_TRANSITION_RADIUS = "side_transition_radius"
SLOPED_CEILING_HANDLE_START_TRANSITION_RADIUS = (
    SLOPED_CEILING_HANDLE_SIDE_TRANSITION_RADIUS
)
SLOPED_CEILING_HANDLE_END_TRANSITION_RADIUS = (
    SLOPED_CEILING_HANDLE_SIDE_TRANSITION_RADIUS
)
SLOPED_CEILING_HANDLE_START = "start"
SLOPED_CEILING_HANDLE_END = "end"
SLOPED_CEILING_HANDLE_KINDS = frozenset(
    (
        SLOPED_CEILING_HANDLE_CREATE_DEPTH,
        SLOPED_CEILING_HANDLE_DEPTH,
        SLOPED_CEILING_HANDLE_WALL_DROP,
        SLOPED_CEILING_HANDLE_CEILING_TRANSITION_RADIUS,
        SLOPED_CEILING_HANDLE_WALL_TRANSITION_RADIUS,
        SLOPED_CEILING_HANDLE_SIDE_TRANSITION_RADIUS,
        SLOPED_CEILING_HANDLE_START,
        SLOPED_CEILING_HANDLE_END,
    )
)

SLOPED_CEILING_GEOMETRY_EPSILON = 1e-7
SLOPED_CEILING_MIN_HEADROOM_METERS = 0.05
SLOPED_CEILING_PROFILE_SAMPLE_COUNT = 33
SLOPED_CEILING_CURVE_SAMPLE_COUNT = 10
SLOPED_CEILING_CLEARANCE_METERS = 0.001
SLOPED_CEILING_BOUNDARY_PROBE_METERS = 1e-5
SLOPED_CEILING_PROFILE_SOLVER_STEPS = 256
SLOPED_CEILING_SIDE_ENDPOINT_FEATHER_FRACTION = 1.0 / 32.0
SLOPED_CEILING_WINDOW_MIN_SIZE_METERS = 0.05
SLOPED_CEILING_WINDOW_PLANE_TOLERANCE_METERS = 0.01
SLOPED_CEILING_WINDOW_REVEAL_DEPTH_METERS = 0.5
SLOPED_CEILING_PART_MAIN = "main"
SLOPED_CEILING_PART_START_RETURN = "start-return"
SLOPED_CEILING_PART_END_RETURN = "end-return"
SLOPED_CEILING_PART_WALL_RETURN = "wall-return"
SLOPED_CEILING_PART_WINDOW_REVEALS = "window-reveals"
SLOPED_CEILING_PARTS = frozenset(
    (
        SLOPED_CEILING_PART_MAIN,
        SLOPED_CEILING_PART_START_RETURN,
        SLOPED_CEILING_PART_END_RETURN,
        SLOPED_CEILING_PART_WALL_RETURN,
        SLOPED_CEILING_PART_WINDOW_REVEALS,
    )
)


# ### Public edit models ###
@dataclass(frozen=True)
class SlopedCeilingEditHandle:
    """One world-space gizmo handle for a sloped-ceiling edit."""

    handle_kind: str
    origin_world: tuple[float, float, float]
    axis_world: tuple[float, float, float]
    value: float
    minimum_value: float
    maximum_value: float
    value_per_meter: float = 1.0

    def __post_init__(self) -> None:
        if self.handle_kind not in SLOPED_CEILING_HANDLE_KINDS:
            raise ValueError(f"Unknown sloped-ceiling handle: {self.handle_kind!r}.")
        _validate_finite_vector(self.origin_world, "handle origin")
        axis = np.asarray(self.axis_world, dtype=float)
        _validate_finite_vector(axis, "handle axis")
        axis_length = float(np.linalg.norm(axis))
        if axis_length <= SLOPED_CEILING_GEOMETRY_EPSILON:
            raise ValueError("A sloped-ceiling handle axis cannot be zero.")
        normalized_axis = axis / axis_length
        minimum = _finite_number(self.minimum_value, "handle minimum")
        maximum = _finite_number(self.maximum_value, "handle maximum")
        value = _finite_number(self.value, "handle value")
        value_per_meter = _finite_number(
            self.value_per_meter,
            "handle value-per-meter",
        )
        if maximum < minimum:
            raise ValueError("A sloped-ceiling handle range is inverted.")
        if abs(value_per_meter) <= SLOPED_CEILING_GEOMETRY_EPSILON:
            raise ValueError("Handle value-per-meter must be non-zero.")
        object.__setattr__(
            self,
            "origin_world",
            tuple(float(value) for value in self.origin_world),
        )
        object.__setattr__(
            self,
            "axis_world",
            tuple(float(value) for value in normalized_axis),
        )
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "minimum_value", minimum)
        object.__setattr__(self, "maximum_value", maximum)
        object.__setattr__(self, "value_per_meter", value_per_meter)


@dataclass(frozen=True)
class SlopedCeilingEditTarget:
    """All slope handles and stable frame data for one selectable wall."""

    slope_id: str | None
    wall_surface_id: str
    level_index: int
    handles: tuple[SlopedCeilingEditHandle, ...]
    wall_start_world: tuple[float, float, float]
    wall_end_world: tuple[float, float, float]
    inward_world: tuple[float, float, float]
    ceiling_z_meters: float
    selection_edge_start_world: tuple[float, float, float]
    selection_edge_end_world: tuple[float, float, float]
    creation_start_ratio: float = 0.0
    creation_end_ratio: float = 1.0
    joined_group_id: str | None = None

    @property
    def target_id(self) -> str:
        """Return one stable identity for an existing slope or free interval."""

        if self.slope_id is not None:
            return self.slope_id
        return (
            "create:"
            f"{self.wall_surface_id}:"
            f"{self.creation_start_ratio:.12f}:"
            f"{self.creation_end_ratio:.12f}"
        )


@dataclass(frozen=True)
class SlopedCeilingEdit:
    """One absolute-value controller edit, suitable for delayed previews."""

    slope_id: str | None
    wall_surface_id: str
    level_index: int
    handle_kind: str
    value: float
    creation_start_ratio: float = 0.0
    creation_end_ratio: float = 1.0

    def __post_init__(self) -> None:
        if self.handle_kind not in SLOPED_CEILING_HANDLE_KINDS:
            raise ValueError(f"Unknown sloped-ceiling handle: {self.handle_kind!r}.")
        if not str(self.wall_surface_id).strip():
            raise ValueError("A sloped-ceiling edit requires a wall surface ID.")
        object.__setattr__(self, "wall_surface_id", str(self.wall_surface_id).strip())
        object.__setattr__(self, "level_index", int(self.level_index))
        object.__setattr__(self, "value", _finite_number(self.value, "edit value"))
        creation_start_ratio = _normalize_creation_ratio(
            self.creation_start_ratio,
            "start",
        )
        creation_end_ratio = _normalize_creation_ratio(
            self.creation_end_ratio,
            "end",
        )
        if (
            self.slope_id is None
            and self.handle_kind == SLOPED_CEILING_HANDLE_CREATE_DEPTH
            and creation_end_ratio - creation_start_ratio
            <= MIN_SLOPED_CEILING_RATIO_SPAN
        ):
            raise ValueError(
                "A sloped-ceiling creation interval must have positive width."
            )
        object.__setattr__(
            self,
            "creation_start_ratio",
            creation_start_ratio,
        )
        object.__setattr__(
            self,
            "creation_end_ratio",
            creation_end_ratio,
        )


# ### Public sloped-window models ###
@dataclass(frozen=True)
class SlopedCeilingWindowHost:
    """Stable straight planar band that may contain one or more windows."""

    slope_id: str
    wall_surface_id: str
    level_index: int
    room_index: int | None
    plane_start_world: tuple[float, float, float]
    wall_tangent_world: tuple[float, float, float]
    wall_normal_world: tuple[float, float, float]
    vertical_axis_world: tuple[float, float, float]
    wall_width_meters: float
    wall_height_meters: float

    def __post_init__(self) -> None:
        for value, label in (
            (self.plane_start_world, "window-host origin"),
            (self.wall_tangent_world, "window-host tangent"),
            (self.wall_normal_world, "window-host normal"),
            (self.vertical_axis_world, "window-host vertical axis"),
        ):
            _validate_finite_vector(value, label)
        tangent = _normalized_vector(self.wall_tangent_world, "window-host tangent")
        normal = _normalized_vector(self.wall_normal_world, "window-host normal")
        vertical = _normalized_vector(
            self.vertical_axis_world,
            "window-host vertical axis",
        )
        if any(
            abs(float(np.dot(first, second))) > 1e-6
            for first, second in (
                (tangent, normal),
                (tangent, vertical),
                (normal, vertical),
            )
        ):
            raise ValueError("Sloped-window host axes must be perpendicular.")
        width = _positive_finite_number(
            self.wall_width_meters,
            "window-host width",
        )
        height = _positive_finite_number(
            self.wall_height_meters,
            "window-host height",
        )
        slope_id = str(self.slope_id).strip()
        wall_surface_id = str(self.wall_surface_id).strip()
        if not slope_id or not wall_surface_id:
            raise ValueError("A sloped-window host requires stable IDs.")
        object.__setattr__(self, "slope_id", slope_id)
        object.__setattr__(self, "wall_surface_id", wall_surface_id)
        object.__setattr__(self, "level_index", int(self.level_index))
        object.__setattr__(
            self,
            "plane_start_world",
            _vector_tuple(self.plane_start_world),
        )
        object.__setattr__(self, "wall_tangent_world", _vector_tuple(tangent))
        object.__setattr__(self, "wall_normal_world", _vector_tuple(normal))
        object.__setattr__(self, "vertical_axis_world", _vector_tuple(vertical))
        object.__setattr__(self, "wall_width_meters", width)
        object.__setattr__(self, "wall_height_meters", height)

    @property
    def target_id(self) -> str:
        """Return the persistent slope identity used by viewer targets."""

        return self.slope_id

    def local_to_world(
        self,
        horizontal_ratio: float,
        vertical_ratio: float,
    ) -> tuple[float, float, float]:
        """Resolve one normalized host coordinate into world space."""

        point = (
            np.asarray(self.plane_start_world, dtype=float)
            + np.asarray(self.wall_tangent_world, dtype=float)
            * float(horizontal_ratio)
            * self.wall_width_meters
            + np.asarray(self.vertical_axis_world, dtype=float)
            * float(vertical_ratio)
            * self.wall_height_meters
        )
        return _vector_tuple(point)

    def world_to_local(
        self,
        point_world: Sequence[float],
    ) -> tuple[float, float]:
        """Project one world point onto normalized planar host coordinates."""

        point = _finite_world_point(point_world, "sloped-window point")
        offset = point - np.asarray(self.plane_start_world, dtype=float)
        horizontal_ratio = float(
            np.dot(offset, np.asarray(self.wall_tangent_world, dtype=float))
            / self.wall_width_meters
        )
        vertical_ratio = float(
            np.dot(offset, np.asarray(self.vertical_axis_world, dtype=float))
            / self.wall_height_meters
        )
        return horizontal_ratio, vertical_ratio


@dataclass(frozen=True)
class SlopedCeilingWindowPlacement:
    """Validated normalized rectangle on one straight sloped-ceiling band."""

    slope_id: str
    wall_surface_id: str
    level_index: int
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
            sloped_ceiling_id=self.slope_id,
        )
        object.__setattr__(self, "slope_id", str(validated.sloped_ceiling_id))
        object.__setattr__(self, "wall_surface_id", validated.wall_surface_id)
        object.__setattr__(self, "level_index", int(self.level_index))
        for field_name in (
            "start_ratio",
            "end_ratio",
            "bottom_ratio",
            "top_ratio",
        ):
            object.__setattr__(self, field_name, getattr(validated, field_name))


# ### Internal geometry models ###
@dataclass(frozen=True)
class _SlopeFrame:
    data: SlopedCeilingData
    level_index: int
    room_index: int | None
    wall_start: np.ndarray
    wall_end: np.ndarray
    tangent_xy: np.ndarray
    inward_xy: np.ndarray
    wall_length: float
    wall_base_z: float
    ceiling_z: float
    maximum_depth_meters: float
    profile: _ResolvedSlopeProfile

    @property
    def start_distance(self) -> float:
        return self.data.start_ratio * self.wall_length

    @property
    def end_distance(self) -> float:
        return self.data.end_ratio * self.wall_length

    @property
    def maximum_drop(self) -> float:
        return self.profile.wall_drop_meters


@dataclass(frozen=True)
class _ResolvedSlopeProfile:
    """One feasible wall-to-ceiling profile with exact circular fillets."""

    depth_meters: float
    wall_drop_meters: float
    angle_radians: float
    wall_radius_meters: float
    ceiling_radius_meters: float
    wall_tangent_depth_meters: float
    wall_tangent_drop_meters: float
    ceiling_tangent_depth_meters: float
    ceiling_tangent_drop_meters: float


@dataclass(frozen=True)
class _SlopedCeilingWindowOpening:
    """Resolved world-space opening used by the ceiling mesh rebuild."""

    host: SlopedCeilingWindowHost
    window: WindowData
    corners_world: tuple[tuple[float, float, float], ...]
    footprint_xy: Polygon


@dataclass
class _JoinedSideReturnGroup:
    """Projected geometry and semantic owner for one joined physical side."""

    owner_slope_id: str
    owner_part: str
    desired_normal: np.ndarray
    plane_normal_xy: np.ndarray
    plane_tangent_xy: np.ndarray
    plane_offset: float
    projected_polygons: list[Polygon]
    contributor_parts: set[tuple[str, str]]


# ### Public edit API ###
def build_sloped_ceiling_edit_targets(
    levels: Sequence[LevelData],
    wall_surfaces: Sequence[object],
) -> tuple[SlopedCeilingEditTarget, ...]:
    """Build deterministic wall-bound targets for the viewer/controller."""

    level_lookup = {level.index: level for level in levels}
    surfaces = tuple(wall_surfaces)
    ceiling_z_lookup = _build_nominal_ceiling_z_lookup(levels, surfaces)
    ceiling_geometry_lookup = _build_host_ceiling_geometry_lookup(surfaces)
    ceiling_geometry_lookup = _restore_sloped_window_host_geometry(
        levels,
        surfaces,
        ceiling_z_lookup,
        ceiling_geometry_lookup,
    )
    targets: list[SlopedCeilingEditTarget] = []
    for surface in surfaces:
        if getattr(surface, "surface_type", None) != "wall":
            continue
        # Authored child faces are not stable wall hosts.  Skipping them keeps
        # one edited surface from disabling the controls for every other wall.
        if getattr(surface, "source_surface_id", None) is not None:
            continue
        level_index = int(surface.level_index)
        level = level_lookup.get(level_index)
        if level is None:
            continue
        surface_id = str(getattr(surface, "surface_id", ""))
        records = tuple(
            sorted(
                (
                    slope
                    for slope in getattr(level, "sloped_ceilings", ())
                    if slope.wall_surface_id == surface_id
                ),
                key=_get_slope_interval_sort_key,
            )
        )
        _validate_non_overlapping_slope_records(records)
        for start_ratio, end_ratio in _get_free_slope_intervals(records):
            frame = _build_frame_for_surface(
                level_index,
                surface,
                SlopedCeilingData(
                    slope_id=uuid.uuid4().hex,
                    wall_surface_id=surface_id,
                    start_ratio=start_ratio,
                    end_ratio=end_ratio,
                ),
                ceiling_z_override=ceiling_z_lookup.get(
                    (level_index, getattr(surface, "room_index", None))
                ),
                host_ceiling_geometry=ceiling_geometry_lookup.get(
                    (level_index, getattr(surface, "room_index", None))
                ),
            )
            if frame is not None:
                targets.append(
                    _build_edit_target(
                        frame,
                        slope_id=None,
                        creation_start_ratio=start_ratio,
                        creation_end_ratio=end_ratio,
                    )
                )
        for record_index, record in enumerate(records):
            frame = _build_frame_for_surface(
                level_index,
                surface,
                record,
                ceiling_z_override=ceiling_z_lookup.get(
                    (level_index, getattr(surface, "room_index", None))
                ),
                host_ceiling_geometry=ceiling_geometry_lookup.get(
                    (level_index, getattr(surface, "room_index", None))
                ),
            )
            if frame is not None:
                targets.append(
                    _build_edit_target(
                        frame,
                        slope_id=record.slope_id,
                        minimum_start_ratio=(
                            records[record_index - 1].end_ratio
                            if record_index > 0
                            else 0.0
                        ),
                        maximum_end_ratio=(
                            records[record_index + 1].start_ratio
                            if record_index + 1 < len(records)
                            else 1.0
                        ),
                    )
                )
    return tuple(
        sorted(
            targets,
            key=lambda item: (
                item.level_index,
                item.wall_surface_id,
                item.slope_id is None,
                item.selection_edge_start_world,
                item.target_id,
            ),
        )
    )


def apply_sloped_ceiling_edit(
    levels: Sequence[LevelData],
    edit: SlopedCeilingEdit,
) -> SlopedCeilingData:
    """Apply one absolute edit to persistent level data and return the record."""

    if not isinstance(edit, SlopedCeilingEdit):
        raise TypeError("Sloped-ceiling edits require SlopedCeilingEdit values.")
    level = next((item for item in levels if item.index == edit.level_index), None)
    if level is None:
        raise ValueError("The sloped-ceiling edit references an unknown level.")
    records = level.sloped_ceilings
    record_index = next(
        (
            index
            for index, record in enumerate(records)
            if record.slope_id == edit.slope_id
        ),
        None,
    )
    if record_index is None:
        if edit.slope_id is not None:
            raise ValueError("The selected sloped ceiling no longer exists.")
        if edit.handle_kind != SLOPED_CEILING_HANDLE_CREATE_DEPTH:
            raise ValueError("Create a sloped ceiling before editing its handles.")
        record = _build_created_slope_record(edit)
        _validate_slope_record_interval_available(records, record)
        records.append(record)
        return record

    current = records[record_index]
    if current.wall_surface_id != edit.wall_surface_id:
        raise ValueError("The sloped-ceiling edit targets a different wall.")
    if (
        edit.handle_kind == SLOPED_CEILING_HANDLE_DEPTH
        and edit.value < MIN_SLOPED_CEILING_DEPTH_METERS
    ):
        records.pop(record_index)
        return current
    field_name = _get_edit_field_name(edit.handle_kind)
    updated = replace(current, **{field_name: edit.value})
    _validate_slope_record_interval_available(
        records,
        updated,
        excluded_slope_id=current.slope_id,
    )
    records[record_index] = updated
    return updated


def build_sloped_ceiling_preview_meshes(
    levels: Sequence[LevelData],
    wall_surfaces: Sequence[object],
    edit: SlopedCeilingEdit | None = None,
    *,
    wall_surface_id: str | None = None,
) -> tuple[trimesh.Trimesh, ...]:
    """Build lightweight translucent-preview skins without mutating levels."""

    level_lookup = {level.index: level for level in levels}
    ceiling_z_lookup = _build_nominal_ceiling_z_lookup(levels, wall_surfaces)
    ceiling_geometry_lookup = _build_host_ceiling_geometry_lookup(wall_surfaces)
    ceiling_geometry_lookup = _restore_sloped_window_host_geometry(
        levels,
        wall_surfaces,
        ceiling_z_lookup,
        ceiling_geometry_lookup,
    )
    surfaces_by_id = {
        str(getattr(surface, "surface_id", "")): surface
        for surface in wall_surfaces
        if getattr(surface, "surface_type", None) == "wall"
    }
    records: list[tuple[int, SlopedCeilingData]] = [
        (level.index, record)
        for level in levels
        for record in getattr(level, "sloped_ceilings", ())
    ]
    if edit is not None:
        records = _preview_records_with_edit(records, edit)
    frames: list[_SlopeFrame] = []
    for level_index, record in records:
        level = level_lookup.get(level_index)
        surface = surfaces_by_id.get(record.wall_surface_id)
        if level is None or surface is None:
            continue
        frame = _build_frame_for_surface(
            level_index,
            surface,
            record,
            ceiling_z_override=ceiling_z_lookup.get(
                (level_index, getattr(surface, "room_index", None))
            ),
            host_ceiling_geometry=ceiling_geometry_lookup.get(
                (level_index, getattr(surface, "room_index", None))
            ),
        )
        if frame is not None:
            frames.append(frame)
    frames_by_host = _group_frames_by_host(frames)
    normalized_wall_surface_id = (
        None if wall_surface_id is None else str(wall_surface_id).strip()
    )
    meshes: list[trimesh.Trimesh] = []
    for frame in frames:
        if (
            normalized_wall_surface_id is not None
            and frame.data.wall_surface_id != normalized_wall_surface_id
        ):
            continue
        mesh = _build_preview_mesh(
            frame,
            frames_by_host[(frame.level_index, frame.room_index)],
            host_ceiling_geometry=ceiling_geometry_lookup.get(
                (frame.level_index, frame.room_index)
            ),
        )
        if mesh is not None:
            meshes.append(mesh)
    return tuple(meshes)


# ### Public sloped-window API ###
def build_sloped_ceiling_window_hosts(
    levels: Sequence[LevelData],
    surfaces: Sequence[object],
) -> tuple[SlopedCeilingWindowHost, ...]:
    """Build one stable window host for each measurable straight slope band."""

    hosts = tuple(
        host
        for frame in _build_slope_frames(levels, surfaces)
        if (host := _build_sloped_ceiling_window_host(frame)) is not None
    )
    return tuple(
        sorted(
            hosts,
            key=lambda item: (
                item.level_index,
                item.wall_surface_id,
                item.slope_id,
            ),
        )
    )


def preserve_sloped_ceiling_window_dimensions(
    levels: Sequence[LevelData],
    surfaces: Sequence[object],
    previous_hosts: Sequence[SlopedCeilingWindowHost],
    edits: Sequence[SlopedCeilingEdit],
) -> tuple[str, ...]:
    """Rebase affected windows without changing their physical dimensions.

    The caller applies the slope edits to a staged level copy first.  This
    function prepares every affected level before replacing any window
    records, so one multi-wall edit remains atomic.  A frame that no longer
    fits after a slope boundary crosses it is removed from the staged level;
    callers can use the returned stable IDs to detach installed window
    fixtures without deleting their reusable assets.
    """

    affected_slope_keys = {
        (int(edit.level_index), str(edit.slope_id))
        for edit in edits
        if edit.slope_id is not None
    }
    if not affected_slope_keys:
        return ()
    if not any(
        (level.index, str(window.sloped_ceiling_id)) in affected_slope_keys
        for level in levels
        for window in level.windows
        if window.sloped_ceiling_id is not None
    ):
        return ()
    previous_hosts_by_key = {
        (host.level_index, host.slope_id): host for host in previous_hosts
    }
    edited_hosts_by_key = {
        (host.level_index, host.slope_id): host
        for host in build_sloped_ceiling_window_hosts(levels, surfaces)
    }
    replacements_by_level: dict[int, dict[str, WindowData]] = {}
    removed_ids_by_level: dict[int, set[str]] = {}
    for level in levels:
        for window in level.windows:
            slope_id = window.sloped_ceiling_id
            slope_key = (level.index, str(slope_id))
            if slope_id is None or slope_key not in affected_slope_keys:
                continue
            previous_host = previous_hosts_by_key.get(slope_key)
            edited_host = edited_hosts_by_key.get(slope_key)
            if previous_host is None or edited_host is None:
                removed_ids_by_level.setdefault(level.index, set()).add(
                    window.window_id
                )
                continue
            if (
                previous_host.wall_surface_id != window.wall_surface_id
                or edited_host.wall_surface_id != window.wall_surface_id
            ):
                removed_ids_by_level.setdefault(level.index, set()).add(
                    window.window_id
                )
                continue
            replacement = _rebase_sloped_ceiling_window(
                window,
                previous_host,
                edited_host,
            )
            if replacement is None:
                removed_ids_by_level.setdefault(level.index, set()).add(
                    window.window_id
                )
                continue
            replacements_by_level.setdefault(level.index, {})[
                window.window_id
            ] = replacement

    for level in levels:
        replacements = replacements_by_level.get(level.index, {})
        removed_ids = removed_ids_by_level.get(level.index, set())
        if not replacements and not removed_ids:
            continue
        level.windows[:] = [
            replacements.get(window.window_id, window)
            for window in level.windows
            if window.window_id not in removed_ids
        ]
    return tuple(
        sorted(
            window_id
            for removed_ids in removed_ids_by_level.values()
            for window_id in removed_ids
        )
    )


def build_sloped_ceiling_window_opening_targets(
    levels: Sequence[LevelData],
    surfaces: Sequence[object],
) -> tuple[CanvasOpeningTarget, ...]:
    """Build Canvas opening targets for windows bound to straight slope bands."""

    # Local imports avoid a surface_geometry -> sloped_ceiling ->
    # canvas_openings -> surface_geometry import cycle.
    from housemaker.canvas_openings import (
        CANVAS_OPENING_WINDOW,
        CanvasOpeningBounds,
        CanvasOpeningReference,
        CanvasOpeningTarget,
    )

    hosts_by_key = {
        (host.level_index, host.slope_id): host
        for host in build_sloped_ceiling_window_hosts(levels, surfaces)
    }
    targets: list[CanvasOpeningTarget] = []
    for level in levels:
        for window_index, window in enumerate(level.windows):
            if window.sloped_ceiling_id is None:
                continue
            host = hosts_by_key.get((level.index, window.sloped_ceiling_id))
            if host is None or host.wall_surface_id != window.wall_surface_id:
                continue
            targets.append(
                CanvasOpeningTarget(
                    reference=CanvasOpeningReference(
                        kind=CANVAS_OPENING_WINDOW,
                        level_index=level.index,
                        item_index=window_index,
                        stable_id=window.window_id,
                    ),
                    wall_surface_id=host.wall_surface_id,
                    plane_start_world=host.plane_start_world,
                    wall_tangent_world=host.wall_tangent_world,
                    wall_normal_world=host.wall_normal_world,
                    vertical_axis_world=host.vertical_axis_world,
                    wall_width_meters=host.wall_width_meters,
                    wall_height_meters=host.wall_height_meters,
                    minimum_width_meters=(
                        SLOPED_CEILING_WINDOW_MIN_SIZE_METERS
                    ),
                    minimum_height_meters=(
                        SLOPED_CEILING_WINDOW_MIN_SIZE_METERS
                    ),
                    bounds=CanvasOpeningBounds(
                        start_ratio=window.start_ratio,
                        end_ratio=window.end_ratio,
                        bottom_ratio=window.bottom_ratio,
                        top_ratio=window.top_ratio,
                    ),
                    sloped_ceiling_id=host.slope_id,
                )
            )
    return tuple(sorted(targets, key=lambda target: target.key))


def build_sloped_ceiling_window_targets(
    levels: Sequence[LevelData],
    surfaces: Sequence[object],
) -> tuple[CanvasOpeningTarget, ...]:
    """Compatibility alias for the explicit opening-target builder."""

    return build_sloped_ceiling_window_opening_targets(levels, surfaces)


def build_sloped_ceiling_window_placement(
    host: SlopedCeilingWindowHost,
    first_world_point: Sequence[float],
    second_world_point: Sequence[float],
) -> SlopedCeilingWindowPlacement:
    """Project and validate one dragged rectangle on a straight slope host."""

    if not isinstance(host, SlopedCeilingWindowHost):
        raise TypeError("A sloped window requires a validated host.")
    first = _finite_world_point(first_world_point, "first sloped-window point")
    second = _finite_world_point(second_world_point, "second sloped-window point")
    origin = np.asarray(host.plane_start_world, dtype=float)
    normal = np.asarray(host.wall_normal_world, dtype=float)
    if any(
        abs(float(np.dot(point - origin, normal)))
        > SLOPED_CEILING_WINDOW_PLANE_TOLERANCE_METERS
        for point in (first, second)
    ):
        raise ValueError("A sloped window rectangle must stay on its host plane.")
    first_horizontal, first_vertical = host.world_to_local(first)
    second_horizontal, second_vertical = host.world_to_local(second)
    start_ratio, end_ratio = _clamp_sloped_window_ratio_pair(
        first_horizontal,
        second_horizontal,
        "horizontal",
    )
    bottom_ratio, top_ratio = _clamp_sloped_window_ratio_pair(
        first_vertical,
        second_vertical,
        "vertical",
    )
    if (
        (end_ratio - start_ratio) * host.wall_width_meters
        < SLOPED_CEILING_WINDOW_MIN_SIZE_METERS
    ):
        raise ValueError("A sloped window must be at least 5 cm wide.")
    if (
        (top_ratio - bottom_ratio) * host.wall_height_meters
        < SLOPED_CEILING_WINDOW_MIN_SIZE_METERS
    ):
        raise ValueError("A sloped window must be at least 5 cm tall.")
    return SlopedCeilingWindowPlacement(
        slope_id=host.slope_id,
        wall_surface_id=host.wall_surface_id,
        level_index=host.level_index,
        start_ratio=start_ratio,
        end_ratio=end_ratio,
        bottom_ratio=bottom_ratio,
        top_ratio=top_ratio,
    )


def get_sloped_ceiling_window_world_corners(
    host: SlopedCeilingWindowHost,
    placement: SlopedCeilingWindowPlacement | WindowData,
) -> tuple[tuple[float, float, float], ...]:
    """Return bottom-start, bottom-end, top-end, and top-start corners."""

    if not isinstance(host, SlopedCeilingWindowHost):
        raise TypeError("Sloped-window corners require a validated host.")
    if isinstance(placement, SlopedCeilingWindowPlacement):
        slope_id = placement.slope_id
        level_index = placement.level_index
    elif isinstance(placement, WindowData):
        slope_id = placement.sloped_ceiling_id
        level_index = host.level_index
    else:
        raise TypeError("Sloped-window corners require a validated placement.")
    if (
        slope_id != host.slope_id
        or placement.wall_surface_id != host.wall_surface_id
        or level_index != host.level_index
    ):
        raise ValueError("The sloped-window placement belongs to another host.")
    return tuple(
        host.local_to_world(horizontal_ratio, vertical_ratio)
        for horizontal_ratio, vertical_ratio in (
            (placement.start_ratio, placement.bottom_ratio),
            (placement.end_ratio, placement.bottom_ratio),
            (placement.end_ratio, placement.top_ratio),
            (placement.start_ratio, placement.top_ratio),
        )
    )


def add_sloped_ceiling_window(
    levels: Sequence[LevelData],
    placement: SlopedCeilingWindowPlacement,
    *,
    window_id: str | None = None,
) -> WindowData:
    """Persist one validated slope-bound window in its owning level."""

    if not isinstance(placement, SlopedCeilingWindowPlacement):
        raise TypeError("A sloped-window commit requires a validated placement.")
    level = next(
        (item for item in levels if item.index == placement.level_index),
        None,
    )
    if level is None:
        raise ValueError("The sloped-window placement references an unknown level.")
    slope = next(
        (
            item
            for item in level.sloped_ceilings
            if item.slope_id == placement.slope_id
        ),
        None,
    )
    if slope is None or slope.wall_surface_id != placement.wall_surface_id:
        raise ValueError("The sloped-window placement host no longer exists.")
    normalized_id = (
        f"window-{uuid.uuid4().hex}"
        if window_id is None
        else str(window_id).strip()
    )
    if any(
        existing.window_id == normalized_id
        for item in levels
        for existing in item.windows
    ):
        raise ValueError("A window with this stable ID already exists.")
    window = WindowData(
        window_id=normalized_id,
        wall_surface_id=placement.wall_surface_id,
        start_ratio=placement.start_ratio,
        end_ratio=placement.end_ratio,
        bottom_ratio=placement.bottom_ratio,
        top_ratio=placement.top_ratio,
        sloped_ceiling_id=placement.slope_id,
    )
    level.windows.append(window)
    return window


# ### Sloped-window edit helpers ###
def _rebase_sloped_ceiling_window(
    window: WindowData,
    previous_host: SlopedCeilingWindowHost,
    edited_host: SlopedCeilingWindowHost,
) -> WindowData | None:
    """Keep one window's size, or return ``None`` when its slope crosses it."""

    previous_center_vertical = (window.bottom_ratio + window.top_ratio) * 0.5
    previous_start_world = previous_host.local_to_world(
        window.start_ratio,
        previous_center_vertical,
    )
    previous_end_world = previous_host.local_to_world(
        window.end_ratio,
        previous_center_vertical,
    )
    edited_start_ratio = edited_host.world_to_local(previous_start_world)[0]
    edited_end_ratio = edited_host.world_to_local(previous_end_world)[0]
    start_ratio = min(edited_start_ratio, edited_end_ratio)
    end_ratio = max(edited_start_ratio, edited_end_ratio)

    previous_center_world = previous_host.local_to_world(
        (window.start_ratio + window.end_ratio) * 0.5,
        previous_center_vertical,
    )
    edited_center_vertical = edited_host.world_to_local(previous_center_world)[1]
    physical_height = (
        window.top_ratio - window.bottom_ratio
    ) * previous_host.wall_height_meters
    vertical_half_span = physical_height / (
        edited_host.wall_height_meters * 2.0
    )
    bottom_ratio = edited_center_vertical - vertical_half_span
    top_ratio = edited_center_vertical + vertical_half_span

    tolerance = SLOPED_CEILING_GEOMETRY_EPSILON * 100.0
    if (
        start_ratio < -tolerance
        or end_ratio > 1.0 + tolerance
        or bottom_ratio < -tolerance
        or top_ratio > 1.0 + tolerance
    ):
        return None
    start_ratio = float(np.clip(start_ratio, 0.0, 1.0))
    end_ratio = float(np.clip(end_ratio, 0.0, 1.0))
    bottom_ratio = float(np.clip(bottom_ratio, 0.0, 1.0))
    top_ratio = float(np.clip(top_ratio, 0.0, 1.0))
    if (
        (end_ratio - start_ratio) * edited_host.wall_width_meters
        < SLOPED_CEILING_WINDOW_MIN_SIZE_METERS
        or (top_ratio - bottom_ratio) * edited_host.wall_height_meters
        < SLOPED_CEILING_WINDOW_MIN_SIZE_METERS
    ):
        return None
    return replace(
        window,
        start_ratio=start_ratio,
        end_ratio=end_ratio,
        bottom_ratio=bottom_ratio,
        top_ratio=top_ratio,
    )


# ### Semantic slope surface helpers ###
def _coalesce_sloped_ceiling_surfaces(
    surfaces: Sequence[object],
) -> list[object]:
    """Reassemble prior semantic slope partitions before rebuilding them."""

    surface_items = list(surfaces)
    children_by_host_id: dict[str, list[object]] = {}
    for surface in surface_items:
        slope_id = getattr(surface, "sloped_ceiling_id", None)
        host_id = getattr(surface, "sloped_ceiling_host_surface_id", None)
        if slope_id is None or host_id is None:
            continue
        children_by_host_id.setdefault(str(host_id), []).append(surface)
    if not children_by_host_id:
        return surface_items

    roots_by_id = {
        str(getattr(surface, "surface_id", "")): surface
        for surface in surface_items
        if getattr(surface, "sloped_ceiling_id", None) is None
    }
    coalesced_by_host_id: dict[str, object] = {}
    for host_id, children in children_by_host_id.items():
        root = roots_by_id.get(host_id)
        template = root or children[0]
        component_surfaces = [
            *((root,) if root is not None else ()),
            *children,
        ]
        combined_mesh = trimesh.util.concatenate(
            [surface.mesh.copy() for surface in component_surfaces]
        )
        coalesced_by_host_id[host_id] = replace(
            template,
            surface_id=host_id,
            mesh=combined_mesh,
            area_square_meters=float(combined_mesh.area),
            sloped_ceiling_id=None,
            sloped_ceiling_wall_surface_id=None,
            sloped_ceiling_host_surface_id=None,
            sloped_ceiling_part=None,
        )

    resolved: list[object] = []
    emitted_host_ids: set[str] = set()
    for surface in surface_items:
        slope_host_id = getattr(
            surface,
            "sloped_ceiling_host_surface_id",
            None,
        )
        surface_id = str(getattr(surface, "surface_id", ""))
        host_id = str(slope_host_id) if slope_host_id is not None else surface_id
        coalesced = coalesced_by_host_id.get(host_id)
        if coalesced is None:
            resolved.append(surface)
            continue
        if host_id in emitted_host_ids:
            continue
        resolved.append(coalesced)
        emitted_host_ids.add(host_id)
    return resolved


def _build_wall_linework_by_level(
    surfaces: Sequence[object],
) -> dict[int, BaseGeometry]:
    """Return wall centerlines that already close vertical slope sides."""

    lines_by_level: dict[int, list[LineString]] = {}
    for surface in surfaces:
        if getattr(surface, "surface_type", None) != "wall":
            continue
        wall_start = getattr(surface, "wall_start_world", None)
        wall_end = getattr(surface, "wall_end_world", None)
        if wall_start is None or wall_end is None:
            continue
        start = np.asarray(wall_start, dtype=float)
        end = np.asarray(wall_end, dtype=float)
        if (
            start.shape != (3,)
            or end.shape != (3,)
            or not np.all(np.isfinite((start, end)))
            or float(np.linalg.norm(end[:2] - start[:2]))
            <= SLOPED_CEILING_GEOMETRY_EPSILON
        ):
            continue
        lines_by_level.setdefault(int(surface.level_index), []).append(
            LineString((tuple(start[:2]), tuple(end[:2])))
        )
    return {
        level_index: shapely.union_all(lines)
        for level_index, lines in lines_by_level.items()
        if lines
    }


# ### Public geometry API ###
def apply_sloped_ceiling_geometry(
    levels: Sequence[LevelData],
    surfaces: Sequence[object],
) -> list[object]:
    """Return distinct slope surfaces with wall tops sharing their profile."""

    surface_items = _coalesce_sloped_ceiling_surfaces(surfaces)
    frames = _build_slope_frames(levels, surface_items)
    if not frames:
        return surface_items

    frames_by_host = _group_frames_by_host(frames)
    ceiling_z_by_host = _build_frame_ceiling_z_lookup(frames_by_host)
    wall_linework_by_level = _build_wall_linework_by_level(surface_items)
    windows_by_slope = _build_sloped_window_lookup(levels)
    resolved: list[object] = []
    for surface in surface_items:
        surface_type = str(getattr(surface, "surface_type", ""))
        host_key = (
            int(surface.level_index),
            getattr(surface, "room_index", None),
        )
        host_frames = frames_by_host.get(host_key, ())
        if not host_frames:
            resolved.append(surface)
            continue
        if surface_type == "ceiling":
            resolved.extend(
                _rebuild_ceiling_surfaces(
                    surface,
                    host_frames,
                    windows_by_slope=windows_by_slope,
                    wall_linework=wall_linework_by_level.get(
                        int(surface.level_index)
                    ),
                )
            )
            continue
        elif surface_type == "wall":
            ceiling_z = ceiling_z_by_host.get(host_key)
            rebuilt = _rebuild_wall_surface(surface, host_frames, ceiling_z)
        else:
            rebuilt = surface
        resolved.append(rebuilt)
    return resolved


def build_sloped_ceiling_surface_id(
    level_index: int,
    slope_id: str,
    part: str = SLOPED_CEILING_PART_MAIN,
) -> str:
    """Return the stable semantic ID for one logical face of a slope."""

    normalized_level_index = int(level_index)
    normalized_slope_id = str(slope_id).strip().lower()
    if (
        normalized_level_index < 0
        or len(normalized_slope_id) != 32
        or any(character not in "0123456789abcdef" for character in normalized_slope_id)
    ):
        raise ValueError("A sloped-ceiling surface requires stable level and slope IDs.")
    normalized_part = str(part).strip().lower()
    if normalized_part not in SLOPED_CEILING_PARTS:
        raise ValueError(f"Unknown sloped-ceiling surface part: {part!r}.")
    part_segment = (
        ""
        if normalized_part == SLOPED_CEILING_PART_MAIN
        else f"part:{normalized_part}:"
    )
    return (
        f"level:{normalized_level_index}/"
        f"sloped-ceiling:{normalized_slope_id}:{part_segment}ceiling"
    )


def get_sloped_ceiling_affected_surface_ids(
    levels: Sequence[LevelData],
    surfaces: Sequence[object],
) -> frozenset[str]:
    """Return semantic IDs whose meshes differ from the flat source geometry."""

    surface_items = list(surfaces)
    frames = _build_slope_frames(levels, surface_items)
    frames_by_host = _group_frames_by_host(frames)
    ceiling_z_by_host = _build_frame_ceiling_z_lookup(frames_by_host)
    affected: set[str] = set()
    for surface in surface_items:
        host_key = (
            int(surface.level_index),
            getattr(surface, "room_index", None),
        )
        host_frames = frames_by_host.get(host_key, ())
        if not host_frames:
            continue
        if getattr(surface, "surface_type", None) == "ceiling":
            affected.add(str(surface.surface_id))
            continue
        if getattr(surface, "surface_type", None) != "wall":
            continue
        ceiling_z = ceiling_z_by_host.get(host_key)
        if ceiling_z is not None and _wall_intersects_any_slope(
            surface,
            host_frames,
            ceiling_z,
        ):
            affected.add(str(surface.surface_id))
    return frozenset(affected)


# ### Target construction helpers ###
def _build_edit_target(
    frame: _SlopeFrame,
    *,
    slope_id: str | None,
    minimum_start_ratio: float = 0.0,
    maximum_end_ratio: float = 1.0,
    creation_start_ratio: float = 0.0,
    creation_end_ratio: float = 1.0,
) -> SlopedCeilingEditTarget:
    data = frame.data
    profile = frame.profile
    start = frame.wall_start + np.append(
        frame.tangent_xy * frame.start_distance,
        frame.ceiling_z - frame.maximum_drop - frame.wall_start[2],
    )
    end = frame.wall_start + np.append(
        frame.tangent_xy * frame.end_distance,
        frame.ceiling_z - frame.maximum_drop - frame.wall_start[2],
    )
    center = (start + end) / 2.0
    inward_3d = np.asarray((*frame.inward_xy, 0.0), dtype=float)
    tangent_3d = np.asarray((*frame.tangent_xy, 0.0), dtype=float)
    depth_origin = center + inward_3d * data.depth_meters
    depth_origin[2] = frame.ceiling_z
    side_start = start.copy()
    side_end = end.copy()
    side_radius = _get_lateral_transition_radius(frame)
    if slope_id is None:
        selection_start = start.copy()
        selection_end = end.copy()
        selection_start[2] = frame.ceiling_z
        selection_end[2] = frame.ceiling_z
        center[2] = frame.ceiling_z
    else:
        selection_start = start.copy()
        selection_end = end.copy()
        selection_start[2] = frame.ceiling_z - frame.maximum_drop
        selection_end[2] = frame.ceiling_z - frame.maximum_drop
    handles: list[SlopedCeilingEditHandle] = []
    if slope_id is None:
        handles.append(
            SlopedCeilingEditHandle(
                handle_kind=SLOPED_CEILING_HANDLE_CREATE_DEPTH,
                origin_world=tuple(float(value) for value in center),
                axis_world=tuple(float(value) for value in inward_3d),
                value=0.0,
                minimum_value=0.0,
                maximum_value=frame.maximum_depth_meters,
            )
        )
    else:
        handles.extend(
            (
                SlopedCeilingEditHandle(
                    SLOPED_CEILING_HANDLE_DEPTH,
                    tuple(float(value) for value in depth_origin),
                    tuple(float(value) for value in inward_3d),
                    data.depth_meters,
                    0.0,
                    frame.maximum_depth_meters,
                ),
                SlopedCeilingEditHandle(
                    SLOPED_CEILING_HANDLE_WALL_DROP,
                    tuple(float(value) for value in center),
                    (0.0, 0.0, 1.0),
                    frame.maximum_drop,
                    0.0,
                    max(
                        0.0,
                        frame.ceiling_z
                        - frame.wall_base_z
                        - SLOPED_CEILING_MIN_HEADROOM_METERS,
                    ),
                    -1.0,
                ),
                SlopedCeilingEditHandle(
                    SLOPED_CEILING_HANDLE_CEILING_TRANSITION_RADIUS,
                    tuple(float(value) for value in depth_origin),
                    tuple(float(value) for value in -inward_3d),
                    profile.ceiling_radius_meters,
                    MIN_SLOPED_CEILING_TRANSITION_RADIUS_METERS,
                    min(
                        MAX_SLOPED_CEILING_TRANSITION_RADIUS_METERS,
                        data.depth_meters,
                    ),
                ),
                SlopedCeilingEditHandle(
                    SLOPED_CEILING_HANDLE_WALL_TRANSITION_RADIUS,
                    tuple(float(value) for value in center),
                    tuple(float(value) for value in inward_3d),
                    profile.wall_radius_meters,
                    MIN_SLOPED_CEILING_TRANSITION_RADIUS_METERS,
                    min(
                        MAX_SLOPED_CEILING_TRANSITION_RADIUS_METERS,
                        frame.maximum_drop,
                    ),
                ),
                SlopedCeilingEditHandle(
                    SLOPED_CEILING_HANDLE_START,
                    tuple(float(value) for value in side_start),
                    tuple(float(value) for value in -tangent_3d),
                    data.start_ratio,
                    minimum_start_ratio,
                    data.end_ratio - MIN_SLOPED_CEILING_RATIO_SPAN,
                    -1.0 / frame.wall_length,
                ),
                SlopedCeilingEditHandle(
                    SLOPED_CEILING_HANDLE_END,
                    tuple(float(value) for value in side_end),
                    tuple(float(value) for value in tangent_3d),
                    data.end_ratio,
                    data.start_ratio + MIN_SLOPED_CEILING_RATIO_SPAN,
                    maximum_end_ratio,
                    1.0 / frame.wall_length,
                ),
            )
        )
        selected_width = frame.end_distance - frame.start_distance
        profile_drop = _profile_drop_at_depth(
            profile,
            data.depth_meters * 0.5,
        )
        profile_midpoint = frame.wall_start + np.append(
            frame.inward_xy * data.depth_meters * 0.5,
            frame.ceiling_z - frame.wall_start[2],
        )
        start_is_internal = _start_side_is_internal(frame)
        end_is_internal = _end_side_is_internal(frame)
        if data.joined_group_id is not None and not (
            start_is_internal or end_is_internal
        ):
            # A physically normalized peer can end at this wall's endpoint
            # while the same joined lateral edge lies inside the longer wall.
            # Keep both records' shared-control layouts structurally equal.
            start_is_internal = True
        if start_is_internal or end_is_internal:
            if start_is_internal:
                tangent_distance = frame.start_distance + side_radius
                side_axis = tangent_3d
            else:
                tangent_distance = frame.end_distance - side_radius
                side_axis = -tangent_3d
            side_radius_origin = profile_midpoint + np.append(
                frame.tangent_xy * tangent_distance,
                0.0,
            )
            side_radius_origin[2] -= profile_drop
            maximum_side_radius = selected_width / (
                2.0 if start_is_internal and end_is_internal else 1.0
            )
            handles.append(
                SlopedCeilingEditHandle(
                    SLOPED_CEILING_HANDLE_SIDE_TRANSITION_RADIUS,
                    tuple(float(value) for value in side_radius_origin),
                    tuple(float(value) for value in side_axis),
                    side_radius,
                    MIN_SLOPED_CEILING_TRANSITION_RADIUS_METERS,
                    min(
                        MAX_SLOPED_CEILING_TRANSITION_RADIUS_METERS,
                        max(0.0, maximum_side_radius),
                    ),
                )
            )
    return SlopedCeilingEditTarget(
        slope_id=slope_id,
        wall_surface_id=data.wall_surface_id,
        level_index=frame.level_index,
        handles=tuple(handles),
        wall_start_world=tuple(float(value) for value in frame.wall_start),
        wall_end_world=tuple(float(value) for value in frame.wall_end),
        inward_world=(float(frame.inward_xy[0]), float(frame.inward_xy[1]), 0.0),
        ceiling_z_meters=frame.ceiling_z,
        selection_edge_start_world=tuple(float(value) for value in selection_start),
        selection_edge_end_world=tuple(float(value) for value in selection_end),
        creation_start_ratio=creation_start_ratio,
        creation_end_ratio=creation_end_ratio,
        joined_group_id=data.joined_group_id,
    )


def _get_edit_field_name(handle_kind: str) -> str:
    field_by_handle = {
        SLOPED_CEILING_HANDLE_CREATE_DEPTH: "depth_meters",
        SLOPED_CEILING_HANDLE_DEPTH: "depth_meters",
        SLOPED_CEILING_HANDLE_WALL_DROP: "wall_drop_meters",
        SLOPED_CEILING_HANDLE_CEILING_TRANSITION_RADIUS: (
            "transition_radius_meters"
        ),
        SLOPED_CEILING_HANDLE_WALL_TRANSITION_RADIUS: (
            "wall_transition_radius_meters"
        ),
        SLOPED_CEILING_HANDLE_SIDE_TRANSITION_RADIUS: (
            "side_transition_radius_meters"
        ),
        SLOPED_CEILING_HANDLE_START: "start_ratio",
        SLOPED_CEILING_HANDLE_END: "end_ratio",
    }
    return field_by_handle[handle_kind]


def _build_created_slope_record(edit: SlopedCeilingEdit) -> SlopedCeilingData:
    """Create a slope whose green handle changes depth, not wall drop."""

    return SlopedCeilingData(
        slope_id=uuid.uuid4().hex,
        wall_surface_id=edit.wall_surface_id,
        depth_meters=max(MIN_SLOPED_CEILING_DEPTH_METERS, edit.value),
        wall_drop_meters=DEFAULT_SLOPED_CEILING_WALL_DROP_METERS,
        start_ratio=edit.creation_start_ratio,
        end_ratio=edit.creation_end_ratio,
    )


def _preview_records_with_edit(
    records: list[tuple[int, SlopedCeilingData]],
    edit: SlopedCeilingEdit,
) -> list[tuple[int, SlopedCeilingData]]:
    updated = list(records)
    record_index = next(
        (
            index
            for index, (level_index, record) in enumerate(updated)
            if level_index == edit.level_index and record.slope_id == edit.slope_id
        ),
        None,
    )
    if record_index is None:
        if edit.slope_id is not None:
            return updated
        if edit.handle_kind != SLOPED_CEILING_HANDLE_CREATE_DEPTH:
            return updated
        created = _build_created_slope_record(edit)
        _validate_slope_record_interval_available(
            (record for _level_index, record in updated),
            created,
        )
        updated.append(
            (
                edit.level_index,
                created,
            )
        )
        return updated
    level_index, record = updated[record_index]
    if (
        edit.handle_kind == SLOPED_CEILING_HANDLE_DEPTH
        and edit.value < MIN_SLOPED_CEILING_DEPTH_METERS
    ):
        updated.pop(record_index)
        return updated
    replacement = replace(
        record,
        **{_get_edit_field_name(edit.handle_kind): edit.value},
    )
    _validate_slope_record_interval_available(
        (
            candidate
            for candidate_level_index, candidate in updated
            if candidate_level_index == level_index
        ),
        replacement,
        excluded_slope_id=record.slope_id,
    )
    updated[record_index] = (
        level_index,
        replacement,
    )
    return updated


# ### Slope interval helpers ###
def _normalize_creation_ratio(value: object, field_name: str) -> float:
    """Normalize one transient creation bound without changing project data."""

    normalized = _finite_number(value, f"creation {field_name} ratio")
    if normalized < 0.0 or normalized > 1.0:
        raise ValueError(
            f"Sloped-ceiling creation {field_name} ratio must be between 0 and 1."
        )
    return normalized


def _get_slope_interval_sort_key(
    record: SlopedCeilingData,
) -> tuple[float, float, str]:
    """Sort persistent slopes by their wall-local interval and stable ID."""

    return (record.start_ratio, record.end_ratio, record.slope_id)


def _slope_intervals_overlap(
    first: SlopedCeilingData,
    second: SlopedCeilingData,
) -> bool:
    """Return whether two same-wall slope interiors overlap measurably."""

    if first.wall_surface_id != second.wall_surface_id:
        return False
    return bool(
        first.start_ratio
        < second.end_ratio - MIN_SLOPED_CEILING_RATIO_SPAN
        and second.start_ratio
        < first.end_ratio - MIN_SLOPED_CEILING_RATIO_SPAN
    )


def _validate_non_overlapping_slope_records(
    records: Iterable[SlopedCeilingData],
) -> None:
    """Reject ambiguous same-wall slope ownership before building controls."""

    records_by_wall: dict[str, list[SlopedCeilingData]] = {}
    for record in records:
        records_by_wall.setdefault(record.wall_surface_id, []).append(record)
    for wall_records in records_by_wall.values():
        ordered = sorted(wall_records, key=_get_slope_interval_sort_key)
        for first, second in pairwise(ordered):
            if _slope_intervals_overlap(first, second):
                raise ValueError(
                    "Sloped-ceiling sections on one wall cannot overlap."
                )


def _validate_slope_record_interval_available(
    records: Iterable[SlopedCeilingData],
    candidate: SlopedCeilingData,
    *,
    excluded_slope_id: str | None = None,
) -> None:
    """Reject a created or resized slope that enters another slope interval."""

    for record in records:
        if record.slope_id == excluded_slope_id:
            continue
        if _slope_intervals_overlap(record, candidate):
            raise ValueError(
                "The sloped-ceiling section overlaps an existing section "
                "on this wall."
            )


def _get_free_slope_intervals(
    records: Sequence[SlopedCeilingData],
) -> tuple[tuple[float, float], ...]:
    """Return every positive-width unsloped interval along one wall."""

    free_intervals: list[tuple[float, float]] = []
    occupied_end = 0.0
    for record in sorted(records, key=_get_slope_interval_sort_key):
        if record.start_ratio - occupied_end > MIN_SLOPED_CEILING_RATIO_SPAN:
            free_intervals.append((occupied_end, record.start_ratio))
        occupied_end = max(occupied_end, record.end_ratio)
    if 1.0 - occupied_end > MIN_SLOPED_CEILING_RATIO_SPAN:
        free_intervals.append((occupied_end, 1.0))
    return tuple(free_intervals)


# ### Frame construction helpers ###
def _build_slope_frames(
    levels: Sequence[LevelData],
    surfaces: Sequence[object],
) -> tuple[_SlopeFrame, ...]:
    surface_by_id = {
        str(getattr(surface, "surface_id", "")): surface
        for surface in surfaces
        if getattr(surface, "surface_type", None) == "wall"
    }
    frames: list[_SlopeFrame] = []
    ceiling_z_lookup = _build_nominal_ceiling_z_lookup(levels, surfaces)
    ceiling_geometry_lookup = _build_host_ceiling_geometry_lookup(surfaces)
    ceiling_geometry_lookup = _restore_sloped_window_host_geometry(
        levels,
        surfaces,
        ceiling_z_lookup,
        ceiling_geometry_lookup,
    )
    for level in levels:
        for record in sorted(
            getattr(level, "sloped_ceilings", ()),
            key=lambda item: item.slope_id,
        ):
            surface = surface_by_id.get(record.wall_surface_id)
            if surface is None or int(surface.level_index) != level.index:
                continue
            frame = _build_frame_for_surface(
                level.index,
                surface,
                record,
                ceiling_z_override=ceiling_z_lookup.get(
                    (level.index, getattr(surface, "room_index", None))
                ),
                host_ceiling_geometry=ceiling_geometry_lookup.get(
                    (level.index, getattr(surface, "room_index", None))
                ),
            )
            if frame is not None:
                frames.append(frame)
    return tuple(frames)


def _build_frame_for_surface(
    level_index: int,
    surface: object,
    record: SlopedCeilingData,
    *,
    ceiling_z_override: float | None = None,
    host_ceiling_geometry: BaseGeometry | None = None,
) -> _SlopeFrame | None:
    raw_start = getattr(surface, "wall_start_world", None)
    raw_end = getattr(surface, "wall_end_world", None)
    if raw_start is None or raw_end is None:
        return None
    wall_start = np.asarray(raw_start, dtype=float)
    wall_end = np.asarray(raw_end, dtype=float)
    delta_xy = wall_end[:2] - wall_start[:2]
    wall_length = float(np.linalg.norm(delta_xy))
    if wall_length <= SLOPED_CEILING_GEOMETRY_EPSILON:
        return None
    tangent = delta_xy / wall_length
    inward = _resolve_surface_inward(
        surface,
        tangent,
        host_ceiling_geometry=host_ceiling_geometry,
    )
    maximum_depth = _resolve_maximum_slope_depth(
        wall_start,
        tangent,
        inward,
        wall_length,
        record,
        host_ceiling_geometry,
    )
    effective_record = (
        record
        if record.depth_meters <= maximum_depth
        else replace(
            record,
            depth_meters=maximum_depth,
            wall_drop_meters=record.wall_drop_meters,
        )
    )
    wall_height = float(getattr(surface, "wall_height_meters", 0.0) or 0.0)
    mesh_vertices = np.asarray(surface.mesh.vertices, dtype=float)
    ceiling_z = (
        float(ceiling_z_override)
        if ceiling_z_override is not None
        else (
            float(np.max(mesh_vertices[:, 2]))
            if len(mesh_vertices)
            else float(wall_start[2] + wall_height)
        )
    )
    available_drop = max(
        0.0,
        ceiling_z
        - float(wall_start[2])
        - SLOPED_CEILING_MIN_HEADROOM_METERS,
    )
    profile = _build_resolved_slope_profile(
        depth_meters=effective_record.depth_meters,
        wall_drop_meters=min(
            float(effective_record.wall_drop_meters or 0.0),
            available_drop,
        ),
        wall_radius_meters=effective_record.wall_transition_radius_meters,
        ceiling_radius_meters=effective_record.transition_radius_meters,
    )
    return _SlopeFrame(
        data=effective_record,
        level_index=int(level_index),
        room_index=getattr(surface, "room_index", None),
        wall_start=wall_start,
        wall_end=wall_end,
        tangent_xy=tangent,
        inward_xy=inward,
        wall_length=wall_length,
        wall_base_z=float(wall_start[2]),
        ceiling_z=ceiling_z,
        maximum_depth_meters=maximum_depth,
        profile=profile,
    )


def _build_sloped_ceiling_window_host(
    frame: _SlopeFrame,
) -> SlopedCeilingWindowHost | None:
    """Resolve the straight profile and lateral bands of one slope."""

    side_radius = _get_lateral_transition_radius(frame)
    tangent_start = frame.start_distance + (
        side_radius if _start_side_is_internal(frame) else 0.0
    )
    tangent_end = frame.end_distance - (
        side_radius if _end_side_is_internal(frame) else 0.0
    )
    width = tangent_end - tangent_start
    depth_start = frame.profile.wall_tangent_depth_meters
    depth_end = frame.profile.ceiling_tangent_depth_meters
    depth_span = depth_end - depth_start
    if (
        width < SLOPED_CEILING_WINDOW_MIN_SIZE_METERS
        or depth_span <= SLOPED_CEILING_GEOMETRY_EPSILON
    ):
        return None

    start_xy = (
        frame.wall_start[:2]
        + frame.tangent_xy * tangent_start
        + frame.inward_xy * depth_start
    )
    start_z = frame.ceiling_z - _profile_drop_at_depth(
        frame.profile,
        depth_start,
    )
    end_z = frame.ceiling_z - _profile_drop_at_depth(
        frame.profile,
        depth_end,
    )
    tangent_world = np.asarray((*frame.tangent_xy, 0.0), dtype=float)
    incline_delta = np.asarray(
        (
            frame.inward_xy[0] * depth_span,
            frame.inward_xy[1] * depth_span,
            end_z - start_z,
        ),
        dtype=float,
    )
    height = float(np.linalg.norm(incline_delta))
    if height < SLOPED_CEILING_WINDOW_MIN_SIZE_METERS:
        return None
    vertical_axis = incline_delta / height
    normal = np.cross(tangent_world, vertical_axis)
    normal_length = float(np.linalg.norm(normal))
    if normal_length <= SLOPED_CEILING_GEOMETRY_EPSILON:
        return None
    normal /= normal_length
    if normal[2] < 0.0:
        normal *= -1.0
    return SlopedCeilingWindowHost(
        slope_id=frame.data.slope_id,
        wall_surface_id=frame.data.wall_surface_id,
        level_index=frame.level_index,
        room_index=frame.room_index,
        plane_start_world=(float(start_xy[0]), float(start_xy[1]), float(start_z)),
        wall_tangent_world=_vector_tuple(tangent_world),
        wall_normal_world=_vector_tuple(normal),
        vertical_axis_world=_vector_tuple(vertical_axis),
        wall_width_meters=float(width),
        wall_height_meters=height,
    )


def _resolve_surface_inward(
    surface: object,
    tangent_xy: np.ndarray,
    *,
    host_ceiling_geometry: BaseGeometry | None = None,
) -> np.ndarray:
    if host_ceiling_geometry is not None and not host_ceiling_geometry.is_empty:
        wall_start = np.asarray(surface.wall_start_world, dtype=float)
        wall_end = np.asarray(surface.wall_end_world, dtype=float)
        midpoint = (wall_start[:2] + wall_end[:2]) * 0.5
        probe_distance = min(
            0.05,
            max(
                SLOPED_CEILING_BOUNDARY_PROBE_METERS,
                float(np.linalg.norm(wall_end[:2] - wall_start[:2])) * 0.02,
            ),
        )
        left = np.asarray((-tangent_xy[1], tangent_xy[0]), dtype=float)
        right = -left
        buffered = host_ceiling_geometry.buffer(
            SLOPED_CEILING_GEOMETRY_EPSILON
        )
        left_is_inside = buffered.covers(
            shapely.Point(tuple(midpoint + left * probe_distance))
        )
        right_is_inside = buffered.covers(
            shapely.Point(tuple(midpoint + right * probe_distance))
        )
        if left_is_inside != right_is_inside:
            return left if left_is_inside else right
    mesh = surface.mesh
    normals = np.asarray(mesh.face_normals, dtype=float)
    areas = np.asarray(mesh.area_faces, dtype=float)
    if len(normals) and len(areas) == len(normals):
        weighted = np.sum(normals[:, :2] * areas[:, np.newaxis], axis=0)
        weighted_length = float(np.linalg.norm(weighted))
        if weighted_length > SLOPED_CEILING_GEOMETRY_EPSILON:
            return weighted / weighted_length
    return np.asarray((-tangent_xy[1], tangent_xy[0]), dtype=float)


def _resolve_maximum_slope_depth(
    wall_start: np.ndarray,
    tangent_xy: np.ndarray,
    inward_xy: np.ndarray,
    wall_length: float,
    record: SlopedCeilingData,
    host_ceiling_geometry: BaseGeometry | None,
) -> float:
    """Stop a slope at the first opposite edge of its ceiling footprint."""

    if host_ceiling_geometry is None or host_ceiling_geometry.is_empty:
        return MAX_SLOPED_CEILING_DEPTH_METERS
    valid_host = (
        host_ceiling_geometry
        if host_ceiling_geometry.is_valid
        else shapely.make_valid(host_ceiling_geometry)
    )
    if valid_host.is_empty:
        return MAX_SLOPED_CEILING_DEPTH_METERS

    selected_start = record.start_ratio * wall_length
    selected_end = record.end_ratio * wall_length
    selected_width = selected_end - selected_start
    probe_depth = max(
        SLOPED_CEILING_GEOMETRY_EPSILON * 100.0,
        min(1e-4, wall_length * 1e-6),
    )
    center = wall_start[:2] + tangent_xy * (
        selected_start + selected_width * 0.5
    )
    probe = center + inward_xy * probe_depth
    host_component = next(
        (
            part
            for part in shapely.get_parts(valid_host)
            if isinstance(part, Polygon)
            and part.buffer(probe_depth).covers(shapely.Point(tuple(probe)))
        ),
        None,
    )
    if host_component is None:
        return MAX_SLOPED_CEILING_DEPTH_METERS

    ray_end = center + inward_xy * MAX_SLOPED_CEILING_DEPTH_METERS
    center_intersection = host_component.intersection(
        LineString((tuple(probe), tuple(ray_end)))
    )
    depth_limits: list[float] = []
    for part in shapely.get_parts(center_intersection):
        if not isinstance(part, LineString) or part.is_empty:
            continue
        coordinates = np.asarray(part.coords, dtype=float)
        depths = (coordinates - center[np.newaxis, :]) @ inward_xy
        if (
            float(np.min(depths))
            <= probe_depth + SLOPED_CEILING_GEOMETRY_EPSILON
            and float(np.max(depths))
            >= probe_depth - SLOPED_CEILING_GEOMETRY_EPSILON
        ):
            depth_limits.append(float(np.max(depths)))
            break
    if not depth_limits:
        return MAX_SLOPED_CEILING_DEPTH_METERS

    if host_component.interiors:
        tangent_margin = min(
            selected_width * 0.25,
            max(
                SLOPED_CEILING_GEOMETRY_EPSILON * 100.0,
                min(1e-4, selected_width * 1e-6),
            ),
        )
        sweep_start = wall_start[:2] + tangent_xy * (
            selected_start + tangent_margin
        )
        sweep_end = wall_start[:2] + tangent_xy * (
            selected_end - tangent_margin
        )
        swept_width = Polygon(
            (
                tuple(sweep_start + inward_xy * probe_depth),
                tuple(sweep_end + inward_xy * probe_depth),
                tuple(
                    sweep_end
                    + inward_xy * MAX_SLOPED_CEILING_DEPTH_METERS
                ),
                tuple(
                    sweep_start
                    + inward_xy * MAX_SLOPED_CEILING_DEPTH_METERS
                ),
            )
        )
        for interior in host_component.interiors:
            if Polygon(interior).area <= SLOPED_CEILING_GEOMETRY_EPSILON:
                continue
            hole_hits = LineString(interior.coords).intersection(swept_width)
            coordinates = np.asarray(
                shapely.get_coordinates(hole_hits),
                dtype=float,
            )
            if not len(coordinates):
                continue
            depths = (coordinates - wall_start[:2][np.newaxis, :]) @ inward_xy
            positive_depths = depths[
                depths >= probe_depth - SLOPED_CEILING_GEOMETRY_EPSILON
            ]
            if len(positive_depths):
                depth_limits.append(float(np.min(positive_depths)))
    return float(
        np.clip(
            min(depth_limits),
            MIN_SLOPED_CEILING_DEPTH_METERS,
            MAX_SLOPED_CEILING_DEPTH_METERS,
        )
    )


def _group_frames_by_host(
    frames: Sequence[_SlopeFrame],
) -> dict[tuple[int, int | None], tuple[_SlopeFrame, ...]]:
    grouped: dict[tuple[int, int | None], list[_SlopeFrame]] = {}
    for frame in frames:
        grouped.setdefault((frame.level_index, frame.room_index), []).append(frame)
    return {
        key: tuple(sorted(items, key=lambda frame: frame.data.slope_id))
        for key, items in grouped.items()
    }


def _build_sloped_window_lookup(
    levels: Sequence[LevelData],
) -> dict[tuple[int, str], tuple[WindowData, ...]]:
    """Group valid slope-bound window records by level and stable slope ID."""

    grouped: dict[tuple[int, str], list[WindowData]] = {}
    for level in levels:
        for window in level.windows:
            if window.sloped_ceiling_id is None:
                continue
            grouped.setdefault(
                (level.index, window.sloped_ceiling_id),
                [],
            ).append(window)
    return {
        key: tuple(sorted(items, key=lambda item: item.window_id))
        for key, items in grouped.items()
    }


def _build_sloped_window_openings(
    frames: Sequence[_SlopeFrame],
    windows_by_slope: dict[tuple[int, str], tuple[WindowData, ...]],
) -> tuple[_SlopedCeilingWindowOpening, ...]:
    """Resolve only windows whose owner and safe planar host still match."""

    openings: list[_SlopedCeilingWindowOpening] = []
    for frame in frames:
        host = _build_sloped_ceiling_window_host(frame)
        if host is None:
            continue
        for window in windows_by_slope.get(
            (frame.level_index, frame.data.slope_id),
            (),
        ):
            if window.wall_surface_id != frame.data.wall_surface_id:
                continue
            corners = get_sloped_ceiling_window_world_corners(host, window)
            footprint = Polygon(
                [(point[0], point[1]) for point in corners]
            )
            if (
                not footprint.is_valid
                or footprint.area <= SLOPED_CEILING_GEOMETRY_EPSILON
            ):
                continue
            openings.append(
                _SlopedCeilingWindowOpening(
                    host=host,
                    window=window,
                    corners_world=corners,
                    footprint_xy=footprint,
                )
            )
    return tuple(openings)


def _build_frame_ceiling_z_lookup(
    frames_by_host: dict[tuple[int, int | None], tuple[_SlopeFrame, ...]],
) -> dict[tuple[int, int | None], float]:
    """Resolve immutable nominal ceiling heights from grouped slope frames."""

    return {
        host_key: _resolve_nominal_host_ceiling_z(host_frames)
        for host_key, host_frames in frames_by_host.items()
    }


def _build_nominal_ceiling_z_lookup(
    levels: Sequence[LevelData],
    surfaces: Sequence[object],
) -> dict[tuple[int, int | None], float]:
    """Resolve immutable flat ceiling elevations, even from sloped surfaces."""

    level_by_index = {level.index: level for level in levels}
    base_z_by_index = build_level_base_z_lookup(levels)
    host_keys = {
        (
            int(surface.level_index),
            getattr(surface, "room_index", None),
        )
        for surface in surfaces
        if getattr(surface, "surface_type", None) in {"wall", "ceiling"}
    }
    result: dict[tuple[int, int | None], float] = {}
    for level_index, room_index in host_keys:
        level = level_by_index.get(level_index)
        if level is None:
            continue
        height = level.height_meters
        if room_index is not None and 0 <= room_index < len(level.rooms):
            height = level.rooms[room_index].height_meters
        result[(level_index, room_index)] = (
            float(base_z_by_index.get(level_index, 0.0))
            + float(height)
            - SLOPED_CEILING_CLEARANCE_METERS
        )
    return result


def _build_host_ceiling_geometry_lookup(
    surfaces: Sequence[object],
) -> dict[tuple[int, int | None], BaseGeometry]:
    """Group ceiling footprints so inward axes ignore manual face winding."""

    grouped: dict[tuple[int, int | None], list[BaseGeometry]] = {}
    for surface in surfaces:
        if getattr(surface, "surface_type", None) != "ceiling":
            continue
        polygon = _mesh_xy_polygon(surface.mesh)
        if polygon is None or polygon.is_empty:
            continue
        grouped.setdefault(
            (
                int(surface.level_index),
                getattr(surface, "room_index", None),
            ),
            [],
        ).append(polygon)
    return {
        key: shapely.union_all(polygons)
        for key, polygons in grouped.items()
    }


def _restore_sloped_window_host_geometry(
    levels: Sequence[LevelData],
    surfaces: Sequence[object],
    ceiling_z_lookup: dict[tuple[int, int | None], float],
    ceiling_geometry_lookup: dict[tuple[int, int | None], BaseGeometry],
) -> dict[tuple[int, int | None], BaseGeometry]:
    """Fill only known slope-window holes before resolving slope depth.

    Rebuilding from an already committed ceiling must not mistake its own
    window holes for structural stair/open-space voids.  A preliminary frame
    uses only the component exteriors to recover each persisted window's
    expected footprint; only those rectangles are restored in the real host.
    """

    walls_by_id = {
        str(getattr(surface, "surface_id", "")): surface
        for surface in surfaces
        if getattr(surface, "surface_type", None) == "wall"
        and getattr(surface, "source_surface_id", None) is None
    }
    restored = dict(ceiling_geometry_lookup)
    for level in levels:
        windows_by_slope: dict[str, list[WindowData]] = {}
        for window in level.windows:
            if window.sloped_ceiling_id is not None:
                windows_by_slope.setdefault(
                    window.sloped_ceiling_id,
                    [],
                ).append(window)
        if not windows_by_slope:
            continue
        for record in level.sloped_ceilings:
            windows = windows_by_slope.get(record.slope_id, ())
            surface = walls_by_id.get(record.wall_surface_id)
            if not windows or surface is None:
                continue
            host_key = (
                level.index,
                getattr(surface, "room_index", None),
            )
            geometry = restored.get(host_key)
            if geometry is None or geometry.is_empty:
                continue
            preliminary_frame = _build_frame_for_surface(
                level.index,
                surface,
                record,
                ceiling_z_override=ceiling_z_lookup.get(host_key),
                host_ceiling_geometry=_geometry_outer_shell(geometry),
            )
            if preliminary_frame is None:
                continue
            host = _build_sloped_ceiling_window_host(preliminary_frame)
            if host is None:
                continue
            footprints = []
            for window in windows:
                if window.wall_surface_id != record.wall_surface_id:
                    continue
                corners = get_sloped_ceiling_window_world_corners(host, window)
                footprint = Polygon(
                    [(point[0], point[1]) for point in corners]
                )
                if (
                    footprint.is_valid
                    and footprint.area > SLOPED_CEILING_GEOMETRY_EPSILON
                ):
                    footprints.append(footprint)
            if footprints:
                restored[host_key] = shapely.union_all(
                    (geometry, *footprints)
                )
    return restored


# ### Cross-section profile helpers ###
def _build_resolved_slope_profile(
    *,
    depth_meters: float,
    wall_drop_meters: float,
    wall_radius_meters: float,
    ceiling_radius_meters: float,
) -> _ResolvedSlopeProfile:
    """Fit two circular transitions while preserving both contact points."""

    depth = max(MIN_SLOPED_CEILING_DEPTH_METERS, float(depth_meters))
    wall_drop = max(0.0, float(wall_drop_meters))
    if wall_drop <= SLOPED_CEILING_GEOMETRY_EPSILON:
        return _ResolvedSlopeProfile(
            depth_meters=depth,
            wall_drop_meters=0.0,
            angle_radians=0.0,
            wall_radius_meters=0.0,
            ceiling_radius_meters=0.0,
            wall_tangent_depth_meters=0.0,
            wall_tangent_drop_meters=0.0,
            ceiling_tangent_depth_meters=depth,
            ceiling_tangent_drop_meters=0.0,
        )

    requested_wall_radius = min(
        max(0.0, float(wall_radius_meters)),
        wall_drop,
    )
    requested_ceiling_radius = min(
        max(0.0, float(ceiling_radius_meters)),
        depth,
    )
    profile = _solve_slope_profile(
        depth,
        wall_drop,
        requested_wall_radius,
        requested_ceiling_radius,
    )
    if profile is not None:
        return profile

    # Oversized fillets can consume the whole straight segment.  Scale them
    # together so their relative visual importance is retained.
    minimum_scale = 0.0
    maximum_scale = 1.0
    resolved = _solve_slope_profile(depth, wall_drop, 0.0, 0.0)
    if resolved is None:  # Defensive: positive depth/drop always has a line.
        raise ValueError("The sloped-ceiling cross-section could not be resolved.")
    for _iteration in range(48):
        candidate_scale = (minimum_scale + maximum_scale) * 0.5
        candidate = _solve_slope_profile(
            depth,
            wall_drop,
            requested_wall_radius * candidate_scale,
            requested_ceiling_radius * candidate_scale,
        )
        if candidate is None:
            maximum_scale = candidate_scale
        else:
            minimum_scale = candidate_scale
            resolved = candidate
    return resolved


def _solve_slope_profile(
    depth: float,
    wall_drop: float,
    wall_radius: float,
    ceiling_radius: float,
) -> _ResolvedSlopeProfile | None:
    """Solve the common tangent between wall and ceiling circular fillets."""

    if (
        wall_radius <= SLOPED_CEILING_GEOMETRY_EPSILON
        and ceiling_radius <= SLOPED_CEILING_GEOMETRY_EPSILON
    ):
        angle = math.atan2(wall_drop, depth)
        return _ResolvedSlopeProfile(
            depth_meters=depth,
            wall_drop_meters=wall_drop,
            angle_radians=angle,
            wall_radius_meters=0.0,
            ceiling_radius_meters=0.0,
            wall_tangent_depth_meters=0.0,
            wall_tangent_drop_meters=wall_drop,
            ceiling_tangent_depth_meters=depth,
            ceiling_tangent_drop_meters=0.0,
        )

    def equation(angle: float) -> float:
        return (
            (depth - wall_radius) * math.sin(angle)
            - (wall_drop - ceiling_radius) * math.cos(angle)
            + wall_radius
            - ceiling_radius
        )

    minimum_angle = 1e-9
    maximum_angle = math.pi * 0.5 - 1e-9
    samples = np.linspace(
        minimum_angle,
        maximum_angle,
        SLOPED_CEILING_PROFILE_SOLVER_STEPS + 1,
    )
    values = [equation(float(angle)) for angle in samples]
    equation_tolerance = (
        max(depth, wall_drop, wall_radius, ceiling_radius, 1.0) * 1e-10
    )
    roots: list[float] = []
    for index, first_value in enumerate(values):
        first_angle = float(samples[index])
        if abs(first_value) <= equation_tolerance:
            roots.append(first_angle)
        if index + 1 >= len(values):
            continue
        second_value = values[index + 1]
        if first_value * second_value >= 0.0:
            continue
        lower_angle = first_angle
        upper_angle = float(samples[index + 1])
        lower_value = first_value
        for _iteration in range(56):
            middle_angle = (lower_angle + upper_angle) * 0.5
            middle_value = equation(middle_angle)
            if lower_value * middle_value <= 0.0:
                upper_angle = middle_angle
            else:
                lower_angle = middle_angle
                lower_value = middle_value
        roots.append((lower_angle + upper_angle) * 0.5)

    base_angle = math.atan2(wall_drop, depth)
    candidates: list[_ResolvedSlopeProfile] = []
    for angle in roots:
        sine = math.sin(angle)
        cosine = math.cos(angle)
        wall_tangent_depth = wall_radius * (1.0 - sine)
        wall_tangent_drop = wall_drop - wall_radius * cosine
        ceiling_tangent_depth = depth - ceiling_radius * sine
        ceiling_tangent_drop = ceiling_radius * (1.0 - cosine)
        delta_depth = ceiling_tangent_depth - wall_tangent_depth
        delta_drop = ceiling_tangent_drop - wall_tangent_drop
        straight_length = delta_depth * cosine - delta_drop * sine
        coordinate_tolerance = max(depth, wall_drop, 1.0) * 1e-8
        if (
            straight_length < -coordinate_tolerance
            or wall_tangent_depth < -coordinate_tolerance
            or ceiling_tangent_depth > depth + coordinate_tolerance
            or wall_tangent_depth
            > ceiling_tangent_depth + coordinate_tolerance
            or wall_tangent_drop
            < ceiling_tangent_drop - coordinate_tolerance
            or wall_tangent_drop > wall_drop + coordinate_tolerance
            or ceiling_tangent_drop < -coordinate_tolerance
        ):
            continue
        candidates.append(
            _ResolvedSlopeProfile(
                depth_meters=depth,
                wall_drop_meters=wall_drop,
                angle_radians=angle,
                wall_radius_meters=wall_radius,
                ceiling_radius_meters=ceiling_radius,
                wall_tangent_depth_meters=max(0.0, wall_tangent_depth),
                wall_tangent_drop_meters=min(
                    wall_drop,
                    max(0.0, wall_tangent_drop),
                ),
                ceiling_tangent_depth_meters=min(
                    depth,
                    max(0.0, ceiling_tangent_depth),
                ),
                ceiling_tangent_drop_meters=min(
                    wall_drop,
                    max(0.0, ceiling_tangent_drop),
                ),
            )
        )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda candidate: abs(candidate.angle_radians - base_angle),
    )


def _profile_drop_at_depth(
    profile: _ResolvedSlopeProfile,
    inward_distance: float,
) -> float:
    """Evaluate the exact circular/linear cross-section at one depth."""

    depth = min(
        profile.depth_meters,
        max(0.0, float(inward_distance)),
    )
    wall_radius = profile.wall_radius_meters
    if (
        wall_radius > SLOPED_CEILING_GEOMETRY_EPSILON
        and depth
        <= profile.wall_tangent_depth_meters
        + SLOPED_CEILING_GEOMETRY_EPSILON
    ):
        radicand = max(
            0.0,
            wall_radius * wall_radius - (depth - wall_radius) ** 2,
        )
        return profile.wall_drop_meters - math.sqrt(radicand)

    ceiling_radius = profile.ceiling_radius_meters
    if (
        ceiling_radius > SLOPED_CEILING_GEOMETRY_EPSILON
        and depth
        >= profile.ceiling_tangent_depth_meters
        - SLOPED_CEILING_GEOMETRY_EPSILON
    ):
        radicand = max(
            0.0,
            ceiling_radius * ceiling_radius
            - (depth - profile.depth_meters) ** 2,
        )
        return ceiling_radius - math.sqrt(radicand)

    tangent = math.tan(profile.angle_radians)
    drop = profile.wall_tangent_drop_meters - tangent * (
        depth - profile.wall_tangent_depth_meters
    )
    return min(profile.wall_drop_meters, max(0.0, drop))


# ### Ceiling geometry helpers ###
def _rebuild_ceiling_surfaces(
    surface: object,
    frames: Sequence[_SlopeFrame],
    *,
    windows_by_slope: (
        dict[tuple[int, str], tuple[WindowData, ...]] | None
    ) = None,
    wall_linework: BaseGeometry | None = None,
) -> tuple[object, ...]:
    """Partition one rebuilt host into a residual ceiling and stable slopes."""

    mesh = surface.mesh
    polygon = _mesh_xy_polygon(mesh)
    if polygon is None:
        return (surface,)
    openings = _build_sloped_window_openings(
        frames,
        windows_by_slope or {},
    )
    outer_shell = _geometry_outer_shell(polygon)
    openings = tuple(
        opening
        for opening in openings
        if outer_shell.buffer(SLOPED_CEILING_GEOMETRY_EPSILON).covers(
            opening.footprint_xy
        )
    )
    # Restore only this feature's prior holes before subtracting the current
    # records.  This keeps rebuilds idempotent without filling stair/open-space
    # holes that belong to another geometry system.
    if openings:
        opening_geometry = shapely.union_all(
            [opening.footprint_xy for opening in openings]
        )
        polygon = shapely.union_all((polygon, opening_geometry))
        visible_polygon = polygon.difference(opening_geometry)
    else:
        visible_polygon = polygon
    cells = _partition_ceiling_polygon(visible_polygon, frames)
    vertices: list[tuple[float, float, float]] = []
    vertex_index: dict[tuple[float, float, float], int] = {}
    faces: list[tuple[int, int, int]] = []
    face_parts: dict[int, tuple[str, str]] = {}
    surface_alias_parts: dict[
        tuple[str, str],
        set[tuple[str, str]],
    ] = {}
    ceiling_z = _resolve_nominal_host_ceiling_z(frames)
    for cell in cells:
        representative = np.asarray(
            cell.representative_point().coords[0],
            dtype=float,
        )
        buffered_cell = cell.buffer(SLOPED_CEILING_GEOMETRY_EPSILON)
        triangles = shapely.constrained_delaunay_triangles(cell)
        drop_by_xy: dict[tuple[float, float], float] = {}
        for triangle in shapely.get_parts(triangles):
            if not isinstance(triangle, Polygon) or triangle.is_empty:
                continue
            if not buffered_cell.covers(triangle):
                continue
            points = list(triangle.exterior.coords)[:-1]
            if len(points) != 3:
                continue
            world_points: list[tuple[float, float, float]] = []
            for point in points:
                point_xy = (float(point[0]), float(point[1]))
                drop = drop_by_xy.get(point_xy)
                if drop is None:
                    drop = _ceiling_drop_at_xy(
                        point,
                        frames,
                        lateral_reference_xy=representative,
                    )
                    drop_by_xy[point_xy] = drop
                world_points.append(
                    (point_xy[0], point_xy[1], ceiling_z - drop)
                )
            indices = [
                _append_indexed_vertex(vertices, vertex_index, point)
                for point in world_points
            ]
            candidate = np.asarray(world_points, dtype=float)
            normal = np.cross(candidate[1] - candidate[0], candidate[2] - candidate[0])
            if normal[2] > 0.0:
                indices.reverse()
            faces.append(tuple(indices))
    main_face_count = len(faces)
    _append_exposed_side_returns(
        visible_polygon,
        frames,
        ceiling_z,
        wall_linework,
        vertices,
        vertex_index,
        faces,
        face_parts,
        surface_alias_parts,
    )
    _append_internal_wall_contact_returns(
        visible_polygon,
        frames,
        ceiling_z,
        vertices,
        vertex_index,
        faces,
        face_parts,
    )
    _append_sloped_window_reveals(
        openings,
        vertices,
        vertex_index,
        faces,
        face_parts,
    )
    rebuilt = _build_mesh(vertices, faces)
    if rebuilt is None:
        return (surface,)
    return _partition_rebuilt_ceiling_surface(
        surface,
        rebuilt,
        frames,
        main_face_count=main_face_count,
        face_parts=face_parts,
        surface_alias_parts=surface_alias_parts,
    )


def _partition_rebuilt_ceiling_surface(
    host_surface: object,
    rebuilt_mesh: trimesh.Trimesh,
    frames: Sequence[_SlopeFrame],
    *,
    main_face_count: int,
    face_parts: dict[int, tuple[str, str]],
    surface_alias_parts: Mapping[
        tuple[str, str],
        Collection[tuple[str, str]],
    ],
) -> tuple[object, ...]:
    """Assign triangles to stable, independently texturable slope parts."""

    triangles = np.asarray(rebuilt_mesh.triangles, dtype=float)
    face_indices_by_part: dict[tuple[str, str], list[int]] = {}
    residual_face_indices: list[int] = []
    for face_index, triangle in enumerate(triangles):
        explicit_part = face_parts.get(face_index)
        if explicit_part is not None:
            face_indices_by_part.setdefault(explicit_part, []).append(face_index)
            continue
        frame = _resolve_triangle_slope_owner(
            triangle,
            frames,
            allow_nearest=face_index >= main_face_count,
        )
        if frame is None:
            residual_face_indices.append(face_index)
            continue
        face_indices_by_part.setdefault(
            (frame.data.slope_id, SLOPED_CEILING_PART_MAIN),
            [],
        ).append(face_index)

    surfaces: list[object] = []
    if residual_face_indices:
        residual_mesh = _extract_mesh_faces(rebuilt_mesh, residual_face_indices)
        surfaces.append(
            replace(
                host_surface,
                mesh=residual_mesh,
                area_square_meters=float(residual_mesh.area),
                sloped_ceiling_id=None,
                sloped_ceiling_wall_surface_id=None,
                sloped_ceiling_host_surface_id=None,
                sloped_ceiling_part=None,
                surface_alias_ids=(),
            )
        )

    host_surface_id = str(host_surface.surface_id)
    frame_by_slope_id = {frame.data.slope_id: frame for frame in frames}
    for slope_id, part in sorted(face_indices_by_part):
        face_indices = face_indices_by_part[(slope_id, part)]
        if not face_indices:
            continue
        frame = frame_by_slope_id[slope_id]
        slope_mesh = _extract_mesh_faces(rebuilt_mesh, face_indices)
        surface_id = build_sloped_ceiling_surface_id(
            int(host_surface.level_index),
            slope_id,
            part,
        )
        alias_ids = tuple(
            sorted(
                build_sloped_ceiling_surface_id(
                    int(host_surface.level_index),
                    alias_slope_id,
                    alias_part,
                )
                for alias_slope_id, alias_part in surface_alias_parts.get(
                    (slope_id, part),
                    (),
                )
                if (alias_slope_id, alias_part) != (slope_id, part)
            )
        )
        surfaces.append(
            replace(
                host_surface,
                surface_id=surface_id,
                mesh=slope_mesh,
                area_square_meters=float(slope_mesh.area),
                source_surface_id=None,
                is_directly_drawn=False,
                sloped_ceiling_id=slope_id,
                sloped_ceiling_wall_surface_id=frame.data.wall_surface_id,
                sloped_ceiling_host_surface_id=host_surface_id,
                sloped_ceiling_part=part,
                surface_alias_ids=alias_ids,
            )
        )
    return tuple(surfaces) if surfaces else (host_surface,)


def _resolve_triangle_slope_owner(
    triangle: np.ndarray,
    frames: Sequence[_SlopeFrame],
    *,
    allow_nearest: bool,
) -> _SlopeFrame | None:
    """Return the deterministic slope responsible for one rebuilt triangle."""

    centroid_xy = np.mean(np.asarray(triangle, dtype=float)[:, :2], axis=0)
    ranked_drops = sorted(
        (
            (
                _single_slope_drop_at_xy(centroid_xy, frame),
                frame.data.slope_id,
                frame,
            )
            for frame in frames
        ),
        key=lambda item: (item[0], item[1]),
        reverse=True,
    )
    if ranked_drops and ranked_drops[0][0] > SLOPED_CEILING_GEOMETRY_EPSILON:
        return ranked_drops[0][2]
    if not allow_nearest:
        return None
    return min(
        frames,
        key=lambda frame: (
            _distance_to_slope_footprint(centroid_xy, frame),
            frame.data.slope_id,
        ),
        default=None,
    )


def _distance_to_slope_footprint(
    point_xy: np.ndarray,
    frame: _SlopeFrame,
) -> float:
    """Return planar distance from a point to one slope's authored rectangle."""

    relative = np.asarray(point_xy, dtype=float) - frame.wall_start[:2]
    tangent_distance = float(np.dot(relative, frame.tangent_xy))
    inward_distance = float(np.dot(relative, frame.inward_xy))
    tangent_gap = max(
        frame.start_distance - tangent_distance,
        tangent_distance - frame.end_distance,
        0.0,
    )
    inward_gap = max(
        -inward_distance,
        inward_distance - frame.data.depth_meters,
        0.0,
    )
    return float(math.hypot(tangent_gap, inward_gap))


def _extract_mesh_faces(
    mesh: trimesh.Trimesh,
    face_indices: Sequence[int],
) -> trimesh.Trimesh:
    """Copy an exact face subset without allowing trimesh to retessellate it."""

    source_faces = np.asarray(mesh.faces, dtype=np.int64)
    selected_faces = source_faces[np.asarray(face_indices, dtype=np.int64)]
    used_vertices, inverse = np.unique(
        selected_faces.reshape(-1),
        return_inverse=True,
    )
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=float)[used_vertices],
        faces=inverse.reshape((-1, 3)),
        process=False,
    )


def _resolve_nominal_host_ceiling_z(frames: Sequence[_SlopeFrame]) -> float:
    """Return the immutable level ceiling height for one rebuilt host.

    The input surface may already contain a committed slope.  Reusing that
    mutable mesh's highest vertex would make later radius edits progressively
    rebase the whole ceiling.  Frames are created from level coordinates, so
    their ceiling height remains stable across preview and committed rebuilds.
    """

    ceiling_values = np.asarray(
        [float(frame.ceiling_z) for frame in frames],
        dtype=float,
    )
    if not len(ceiling_values) or not np.all(np.isfinite(ceiling_values)):
        raise ValueError("A sloped ceiling requires a finite host ceiling height.")
    return float(np.max(ceiling_values))


def _mesh_xy_polygon(mesh: trimesh.Trimesh) -> BaseGeometry | None:
    vertices = np.asarray(mesh.vertices, dtype=float)
    polygons = []
    for face in np.asarray(mesh.faces, dtype=np.int64):
        triangle = Polygon(vertices[face, :2])
        if triangle.is_valid and triangle.area > SLOPED_CEILING_GEOMETRY_EPSILON:
            polygons.append(triangle)
    if not polygons:
        return None
    geometry = shapely.union_all(polygons)
    return None if geometry.is_empty else geometry


def _geometry_outer_shell(geometry: BaseGeometry) -> BaseGeometry:
    """Return filled component exteriors while preserving disconnected hosts."""

    shells = [
        Polygon(part.exterior)
        for part in shapely.get_parts(geometry)
        if isinstance(part, Polygon) and not part.is_empty
    ]
    return shapely.union_all(shells) if shells else geometry


def _append_sloped_window_reveals(
    openings: Sequence[_SlopedCeilingWindowOpening],
    vertices: list[tuple[float, float, float]],
    vertex_index: dict[tuple[float, float, float], int],
    faces: list[tuple[int, int, int]],
    face_parts: dict[int, tuple[str, str]],
) -> None:
    """Seal every slope opening with four inward-facing tunnel sides."""

    for opening in openings:
        inner = np.asarray(opening.corners_world, dtype=float)
        normal = np.asarray(opening.host.wall_normal_world, dtype=float)
        outer = inner + normal * SLOPED_CEILING_WINDOW_REVEAL_DEPTH_METERS
        center = np.mean(inner, axis=0)
        for index in range(4):
            next_index = (index + 1) % 4
            first_inner = inner[index]
            second_inner = inner[next_index]
            first_outer = outer[index]
            second_outer = outer[next_index]
            desired_normal = center - (first_inner + second_inner) * 0.5
            first_face_index = _append_oriented_triangle(
                vertices,
                vertex_index,
                faces,
                (first_inner, second_inner, second_outer),
                desired_normal,
            )
            second_face_index = _append_oriented_triangle(
                vertices,
                vertex_index,
                faces,
                (first_inner, second_outer, first_outer),
                desired_normal,
            )
            for face_index in (first_face_index, second_face_index):
                if face_index is not None:
                    face_parts[face_index] = (
                        opening.host.slope_id,
                        SLOPED_CEILING_PART_WINDOW_REVEALS,
                    )


def _partition_ceiling_polygon(
    polygon: BaseGeometry,
    frames: Sequence[_SlopeFrame],
) -> tuple[Polygon, ...]:
    minimum_x, minimum_y, maximum_x, maximum_y = polygon.bounds
    span = max(maximum_x - minimum_x, maximum_y - minimum_y, 1.0) * 4.0
    lines: list[BaseGeometry] = [polygon.boundary]
    for frame in frames:
        tangent_values = _slope_tangent_breakpoints(frame)
        depth_values = _slope_depth_breakpoints(frame)
        for tangent_distance in tangent_values:
            center = frame.wall_start[:2] + frame.tangent_xy * tangent_distance
            lines.append(
                LineString(
                    (
                        center - frame.inward_xy * span,
                        center + frame.inward_xy * span,
                    )
                )
            )
        for depth in depth_values:
            center = frame.wall_start[:2] + frame.inward_xy * depth
            lines.append(
                LineString(
                    (
                        center - frame.tangent_xy * span,
                        center + frame.tangent_xy * span,
                    )
                )
            )
    noded = shapely.union_all(lines)
    cells = [
        part
        for part in shapely.get_parts(shapely.polygonize(shapely.get_parts(noded)))
        if isinstance(part, Polygon)
        and part.area > SLOPED_CEILING_GEOMETRY_EPSILON
        and polygon.buffer(SLOPED_CEILING_GEOMETRY_EPSILON).covers(
            part.representative_point()
        )
    ]
    return (
        tuple(cells)
        if cells
        else tuple(
            part for part in shapely.get_parts(polygon) if isinstance(part, Polygon)
        )
    )


def _slope_tangent_breakpoints(frame: _SlopeFrame) -> tuple[float, ...]:
    side_radius = _get_lateral_transition_radius(frame)
    values = {frame.start_distance, frame.end_distance}
    if (
        _start_side_is_internal(frame)
        and side_radius > SLOPED_CEILING_GEOMETRY_EPSILON
    ):
        for index in range(SLOPED_CEILING_CURVE_SAMPLE_COUNT + 1):
            angle = (
                math.pi
                * index
                / (2.0 * SLOPED_CEILING_CURVE_SAMPLE_COUNT)
            )
            offset = side_radius * (1.0 - math.cos(angle))
            values.add(frame.start_distance + offset)
    if (
        _end_side_is_internal(frame)
        and side_radius > SLOPED_CEILING_GEOMETRY_EPSILON
    ):
        for index in range(SLOPED_CEILING_CURVE_SAMPLE_COUNT + 1):
            angle = (
                math.pi
                * index
                / (2.0 * SLOPED_CEILING_CURVE_SAMPLE_COUNT)
            )
            offset = side_radius * (1.0 - math.cos(angle))
            values.add(frame.end_distance - offset)
    return tuple(sorted(values))


def _slope_depth_breakpoints(frame: _SlopeFrame) -> tuple[float, ...]:
    profile = frame.profile
    values = {
        0.0,
        profile.wall_tangent_depth_meters,
        profile.ceiling_tangent_depth_meters,
        profile.depth_meters,
    }
    if profile.wall_radius_meters > SLOPED_CEILING_GEOMETRY_EPSILON:
        span = profile.wall_tangent_depth_meters
        for index in range(SLOPED_CEILING_CURVE_SAMPLE_COUNT + 1):
            values.add(span * index / SLOPED_CEILING_CURVE_SAMPLE_COUNT)
    if profile.ceiling_radius_meters > SLOPED_CEILING_GEOMETRY_EPSILON:
        start = profile.ceiling_tangent_depth_meters
        span = profile.depth_meters - start
        for index in range(SLOPED_CEILING_CURVE_SAMPLE_COUNT + 1):
            values.add(
                start + span * index / SLOPED_CEILING_CURVE_SAMPLE_COUNT
            )
    envelope_start, envelope_end, envelope_ramp = (
        _side_rounding_depth_span(frame)
    )
    side_rounding_is_active = (
        _get_lateral_transition_radius(frame)
        > SLOPED_CEILING_GEOMETRY_EPSILON
    )
    if (
        side_rounding_is_active
        and envelope_ramp > SLOPED_CEILING_GEOMETRY_EPSILON
    ):
        for index in range(SLOPED_CEILING_CURVE_SAMPLE_COUNT + 1):
            offset = envelope_ramp * index / SLOPED_CEILING_CURVE_SAMPLE_COUNT
            values.add(envelope_start + offset)
            values.add(envelope_end - offset)
    if side_rounding_is_active:
        for index in range(SLOPED_CEILING_PROFILE_SAMPLE_COUNT):
            depth = (
                profile.depth_meters
                * index
                / (SLOPED_CEILING_PROFILE_SAMPLE_COUNT - 1)
            )
            if envelope_start <= depth <= envelope_end:
                values.add(depth)
    return tuple(sorted(values))


def _ceiling_drop_at_xy(
    point_xy: Sequence[float],
    frames: Sequence[_SlopeFrame],
    *,
    lateral_reference_xy: Sequence[float] | None = None,
) -> float:
    return max(
        (
            _single_slope_drop_at_xy(
                point_xy,
                frame,
                lateral_reference_xy=lateral_reference_xy,
            )
            for frame in frames
        ),
        default=0.0,
    )


def _single_slope_drop_at_xy(
    point_xy: Sequence[float],
    frame: _SlopeFrame,
    *,
    lateral_reference_xy: Sequence[float] | None = None,
) -> float:
    point = np.asarray(point_xy, dtype=float)
    relative = point - frame.wall_start[:2]
    tangent_distance = float(np.dot(relative, frame.tangent_xy))
    inward_distance = float(np.dot(relative, frame.inward_xy))
    depth = frame.data.depth_meters
    if (
        inward_distance < -SLOPED_CEILING_GEOMETRY_EPSILON
        or inward_distance > depth + SLOPED_CEILING_GEOMETRY_EPSILON
        or tangent_distance
        < frame.start_distance - SLOPED_CEILING_GEOMETRY_EPSILON
        or tangent_distance
        > frame.end_distance + SLOPED_CEILING_GEOMETRY_EPSILON
    ):
        return 0.0
    lateral_reference_distance = None
    inward_reference_distance = None
    if lateral_reference_xy is not None:
        reference = np.asarray(lateral_reference_xy, dtype=float)
        if reference.shape == (2,) and np.all(np.isfinite(reference)):
            reference_relative = reference - frame.wall_start[:2]
            lateral_reference_distance = float(
                np.dot(reference_relative, frame.tangent_xy)
            )
            inward_reference_distance = float(
                np.dot(reference_relative, frame.inward_xy)
            )
    if (
        inward_reference_distance is not None
        and math.isclose(
            inward_distance,
            0.0,
            abs_tol=SLOPED_CEILING_GEOMETRY_EPSILON,
        )
        and inward_reference_distance < -SLOPED_CEILING_GEOMETRY_EPSILON
    ):
        # An interior host wall partitions one ceiling polygon into a sloped
        # side and a flat side.  Their cells share the same XY boundary, so
        # the cell representative decides which coincident height belongs to
        # each side.  Without this distinction the outward cell becomes one
        # giant ramp down to the slope's wall contact.
        return 0.0
    if (
        lateral_reference_distance is not None
        and (
            (
                _start_side_is_internal(frame)
                and math.isclose(
                    tangent_distance,
                    frame.start_distance,
                    abs_tol=SLOPED_CEILING_GEOMETRY_EPSILON,
                )
            )
            or (
                _end_side_is_internal(frame)
                and math.isclose(
                    tangent_distance,
                    frame.end_distance,
                    abs_tol=SLOPED_CEILING_GEOMETRY_EPSILON,
                )
            )
        )
        and not (
            frame.start_distance
            < lateral_reference_distance
            < frame.end_distance
        )
    ):
        # Partition cells on the flat side of an exposed boundary need their
        # own coincident vertex.  The side-return skin joins that vertex to
        # the rounded boundary used by the selected side.
        return 0.0
    profile_drop = _profile_drop_at_depth(frame.profile, inward_distance)
    edge_lift = _lateral_edge_fillet_lift(
        tangent_distance,
        inward_distance,
        profile_drop,
        frame,
    )
    return min(
        frame.maximum_drop,
        max(0.0, profile_drop - edge_lift),
    )


# ### Lateral edge fillet helpers ###
def _lateral_edge_fillet_lift(
    tangent_distance: float,
    inward_distance: float,
    profile_drop: float,
    frame: _SlopeFrame,
) -> float:
    """Lift only the narrow lower-edge strip beside an exposed side."""

    effective_radius = _effective_lateral_edge_radius(
        inward_distance,
        profile_drop,
        frame,
    )
    if effective_radius <= SLOPED_CEILING_GEOMETRY_EPSILON:
        return 0.0

    distances: list[float] = []
    if _start_side_is_internal(frame):
        distances.append(tangent_distance - frame.start_distance)
    if _end_side_is_internal(frame):
        distances.append(frame.end_distance - tangent_distance)
    eligible_distances = [
        max(0.0, distance)
        for distance in distances
        if distance >= -SLOPED_CEILING_GEOMETRY_EPSILON
    ]
    if not eligible_distances:
        return 0.0
    edge_distance = min(eligible_distances)
    if edge_distance >= effective_radius:
        return 0.0

    offset_from_center = edge_distance - effective_radius
    radicand = max(
        0.0,
        effective_radius * effective_radius
        - offset_from_center * offset_from_center,
    )
    return float(effective_radius - math.sqrt(radicand))


def _effective_lateral_edge_radius(
    inward_distance: float,
    profile_drop: float,
    frame: _SlopeFrame,
) -> float:
    """Return a depth-local radius without touching either profile fillet."""

    requested_radius = _get_lateral_transition_radius(frame)
    if requested_radius <= SLOPED_CEILING_GEOMETRY_EPSILON:
        return 0.0
    depth_envelope = _side_rounding_depth_envelope(
        inward_distance,
        frame,
    )
    return float(
        min(
            max(0.0, profile_drop),
            requested_radius * depth_envelope,
        )
    )


def _get_lateral_transition_radius(frame: _SlopeFrame) -> float:
    """Resolve one shared radius for every exposed selected-width side."""

    exposed_side_count = int(_start_side_is_internal(frame)) + int(
        _end_side_is_internal(frame)
    )
    if exposed_side_count == 0:
        return 0.0
    width = max(0.0, frame.end_distance - frame.start_distance)
    width_limit = width / (2.0 if exposed_side_count == 2 else 1.0)
    return float(
        min(
            max(0.0, float(frame.data.side_transition_radius_meters or 0.0)),
            width_limit,
        )
    )


def _side_rounding_depth_span(
    frame: _SlopeFrame,
) -> tuple[float, float, float]:
    """Return profile-controlled contact bounds and a fixed side feather."""

    profile = frame.profile
    start = profile.wall_tangent_depth_meters
    end = profile.ceiling_tangent_depth_meters
    straight_span = max(0.0, end - start)
    profile_feather = (
        profile.depth_meters
        * SLOPED_CEILING_SIDE_ENDPOINT_FEATHER_FRACTION
    )
    feather = min(straight_span * 0.5, profile_feather)
    return float(start), float(end), float(feather)


def _side_rounding_depth_envelope(
    inward_distance: float,
    frame: _SlopeFrame,
) -> float:
    """Keep side rounding out of both profile-controlled transition bands."""

    start, end, ramp = _side_rounding_depth_span(frame)
    if (
        ramp <= SLOPED_CEILING_GEOMETRY_EPSILON
        or inward_distance <= start
        or inward_distance >= end
    ):
        return 0.0
    return _smoothstep((inward_distance - start) / ramp) * _smoothstep(
        (end - inward_distance) / ramp
    )


def _start_side_is_internal(frame: _SlopeFrame) -> bool:
    return frame.data.start_ratio > MIN_SLOPED_CEILING_RATIO_SPAN


def _end_side_is_internal(frame: _SlopeFrame) -> bool:
    return frame.data.end_ratio < 1.0 - MIN_SLOPED_CEILING_RATIO_SPAN


# ### Exposed side-return geometry ###
def _append_exposed_side_returns(
    ceiling_polygon: BaseGeometry,
    frames: Sequence[_SlopeFrame],
    ceiling_z: float,
    wall_linework: BaseGeometry | None,
    vertices: list[tuple[float, float, float]],
    vertex_index: dict[tuple[float, float, float], int],
    faces: list[tuple[int, int, int]],
    face_parts: dict[int, tuple[str, str]],
    surface_alias_parts: dict[
        tuple[str, str],
        set[tuple[str, str]],
    ],
) -> None:
    """Close each partial-width side without moving either contact base."""

    boundary = ceiling_polygon.boundary
    joined_groups_by_plane: dict[
        tuple[str, float, float, float],
        _JoinedSideReturnGroup,
    ] = {}
    for frame in sorted(frames, key=lambda item: item.data.slope_id):
        for tangent_distance, direction_sign, is_internal in (
            (
                frame.start_distance,
                -1.0,
                _start_side_is_internal(frame),
            ),
            (
                frame.end_distance,
                1.0,
                _end_side_is_internal(frame),
            ),
        ):
            if not is_internal:
                continue
            anchor = (
                frame.wall_start[:2]
                + frame.tangent_xy * tangent_distance
            )
            line_end = anchor + frame.inward_xy * frame.data.depth_meters
            clipped = ceiling_polygon.intersection(
                LineString((tuple(anchor), tuple(line_end)))
            )
            for part in shapely.get_parts(clipped):
                if not isinstance(part, LineString) or part.is_empty:
                    continue
                coordinates = np.asarray(part.coords, dtype=float)
                if len(coordinates) < 2:
                    continue
                depths = [
                    float(np.dot(point - anchor, frame.inward_xy))
                    for point in (coordinates[0], coordinates[-1])
                ]
                minimum_depth, maximum_depth = sorted(depths)
                samples = {
                    minimum_depth,
                    maximum_depth,
                    *(
                        depth
                        for depth in _slope_depth_breakpoints(frame)
                        if minimum_depth
                        < depth
                        < maximum_depth
                    ),
                }
                wall_owned_depth_intervals: tuple[
                    tuple[float, float], ...
                ] = ()
                if wall_linework is not None and not wall_linework.is_empty:
                    wall_owned_depth_intervals = (
                        _get_wall_overlap_depth_intervals(
                            wall_linework,
                            anchor,
                            frame.inward_xy,
                            minimum_depth,
                            maximum_depth,
                        )
                    )
                    samples.update(
                        depth
                        for interval in wall_owned_depth_intervals
                        for depth in interval
                    )
                ordered_depths = sorted(samples)
                for first_depth, second_depth in pairwise(ordered_depths):
                    middle_depth = (first_depth + second_depth) * 0.5
                    midpoint = anchor + frame.inward_xy * (
                        middle_depth
                    )
                    midpoint_geometry = shapely.Point(tuple(midpoint))
                    if (
                        boundary.distance(midpoint_geometry)
                        <= SLOPED_CEILING_GEOMETRY_EPSILON
                    ):
                        # A perpendicular host wall already closes this edge.
                        continue
                    if any(
                        start - SLOPED_CEILING_GEOMETRY_EPSILON
                        <= middle_depth
                        <= end + SLOPED_CEILING_GEOMETRY_EPSILON
                        for start, end in wall_owned_depth_intervals
                    ):
                        # An interior wall owns this vertical skin.  Emitting a
                        # second selectable return would z-fight with it.
                        continue
                    first = anchor + frame.inward_xy * first_depth
                    second = anchor + frame.inward_xy * second_depth
                    inward_sign = -direction_sign
                    inside_reference_offset = (
                        frame.tangent_xy
                        * inward_sign
                        * SLOPED_CEILING_BOUNDARY_PROBE_METERS
                    )
                    outside_reference_offset = -inside_reference_offset
                    outside_first_z = ceiling_z - _ceiling_drop_at_xy(
                        first,
                        frames,
                        lateral_reference_xy=(
                            first + outside_reference_offset
                        ),
                    )
                    outside_second_z = ceiling_z - _ceiling_drop_at_xy(
                        second,
                        frames,
                        lateral_reference_xy=(
                            second + outside_reference_offset
                        ),
                    )
                    inside_first_z = ceiling_z - _ceiling_drop_at_xy(
                        first,
                        frames,
                        lateral_reference_xy=(first + inside_reference_offset),
                    )
                    inside_second_z = ceiling_z - _ceiling_drop_at_xy(
                        second,
                        frames,
                        lateral_reference_xy=(second + inside_reference_offset),
                    )
                    desired_normal = np.asarray(
                        (
                            frame.tangent_xy[0] * direction_sign,
                            frame.tangent_xy[1] * direction_sign,
                            0.0,
                        ),
                        dtype=float,
                    )
                    part = (
                        SLOPED_CEILING_PART_START_RETURN
                        if direction_sign < 0.0
                        else SLOPED_CEILING_PART_END_RETURN
                    )
                    outside_first = (
                        float(first[0]),
                        float(first[1]),
                        float(outside_first_z),
                    )
                    outside_second = (
                        float(second[0]),
                        float(second[1]),
                        float(outside_second_z),
                    )
                    inside_first = (
                        float(first[0]),
                        float(first[1]),
                        float(inside_first_z),
                    )
                    inside_second = (
                        float(second[0]),
                        float(second[1]),
                        float(inside_second_z),
                    )
                    triangles = (
                        (outside_first, outside_second, inside_second),
                        (outside_first, inside_second, inside_first),
                    )
                    joined_group_id = frame.data.joined_group_id
                    if joined_group_id is not None:
                        _collect_joined_side_return_triangles(
                            triangles,
                            desired_normal,
                            joined_group_id,
                            frame.data.slope_id,
                            part,
                            joined_groups_by_plane,
                        )
                        continue
                    _append_semantic_triangles(
                        vertices,
                        vertex_index,
                        faces,
                        face_parts,
                        triangles,
                        desired_normal,
                        frame.data.slope_id,
                        part,
                    )
    _append_joined_side_return_groups(
        joined_groups_by_plane,
        vertices,
        vertex_index,
        faces,
        face_parts,
        surface_alias_parts,
    )


def _get_wall_overlap_depth_intervals(
    wall_linework: BaseGeometry,
    anchor: np.ndarray,
    inward_xy: np.ndarray,
    minimum_depth: float,
    maximum_depth: float,
) -> tuple[tuple[float, float], ...]:
    """Return depth intervals owned by collinear interior walls."""

    tolerance = SLOPED_CEILING_GEOMETRY_EPSILON * 10.0
    side_normal = np.asarray((-inward_xy[1], inward_xy[0]), dtype=float)
    intervals: list[tuple[float, float]] = []
    for part in shapely.get_parts(wall_linework):
        if not isinstance(part, LineString) or part.is_empty:
            continue
        coordinates = np.asarray(part.coords, dtype=float)
        for first, second in pairwise(coordinates):
            segment = second - first
            segment_length = float(np.linalg.norm(segment))
            if segment_length <= SLOPED_CEILING_GEOMETRY_EPSILON:
                continue
            segment_direction = segment / segment_length
            cross_product = (
                segment_direction[0] * inward_xy[1]
                - segment_direction[1] * inward_xy[0]
            )
            if abs(float(cross_product)) > tolerance:
                continue
            if max(
                abs(float(np.dot(first - anchor, side_normal))),
                abs(float(np.dot(second - anchor, side_normal))),
            ) > tolerance:
                continue
            segment_depths = (
                float(np.dot(first - anchor, inward_xy)),
                float(np.dot(second - anchor, inward_xy)),
            )
            start = max(minimum_depth, min(segment_depths))
            end = min(maximum_depth, max(segment_depths))
            if end - start > SLOPED_CEILING_GEOMETRY_EPSILON:
                intervals.append((start, end))
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + tolerance:
            merged[-1][1] = max(merged[-1][1], end)
            continue
        merged.append([start, end])
    return tuple(
        (round(start, 12), round(end, 12)) for start, end in merged
    )


def _collect_joined_side_return_triangles(
    triangles: Sequence[Sequence[Sequence[float]]],
    desired_normal: np.ndarray,
    joined_group_id: str,
    slope_id: str,
    part: str,
    groups_by_plane: dict[
        tuple[str, float, float, float],
        _JoinedSideReturnGroup,
    ],
) -> None:
    """Collect all patches that form one physical joined-slope side.

    Projection into the shared vertical plane lets the final pass union every
    member before triangulation.  That removes overlaps and tiny detached
    slivers while preserving the first stable real slope as semantic owner.
    """

    normal_xy = np.asarray(desired_normal[:2], dtype=float)
    normal_xy /= float(np.linalg.norm(normal_xy))
    if (
        normal_xy[0] < -SLOPED_CEILING_GEOMETRY_EPSILON
        or (
            abs(float(normal_xy[0])) <= SLOPED_CEILING_GEOMETRY_EPSILON
            and normal_xy[1] < 0.0
        )
    ):
        normal_xy = -normal_xy
    plane_tangent = np.asarray((-normal_xy[1], normal_xy[0]), dtype=float)
    first_point = np.asarray(triangles[0][0], dtype=float)
    plane_offset = float(np.dot(first_point[:2], normal_xy))
    plane_key = (
        joined_group_id,
        round(float(normal_xy[0]), 8),
        round(float(normal_xy[1]), 8),
        round(plane_offset, 8),
    )

    projected_polygons: list[Polygon] = []
    for triangle in triangles:
        projected = Polygon(
            [
                (
                    float(
                        np.dot(
                            np.asarray(point, dtype=float)[:2],
                            plane_tangent,
                        )
                    ),
                    float(point[2]),
                )
                for point in triangle
            ]
        )
        if projected.is_valid and projected.area > SLOPED_CEILING_GEOMETRY_EPSILON:
            projected_polygons.append(projected)
    if not projected_polygons:
        return

    group = groups_by_plane.get(plane_key)
    if group is None:
        groups_by_plane[plane_key] = _JoinedSideReturnGroup(
            owner_slope_id=slope_id,
            owner_part=part,
            desired_normal=np.asarray(desired_normal, dtype=float),
            plane_normal_xy=normal_xy,
            plane_tangent_xy=plane_tangent,
            plane_offset=plane_offset,
            projected_polygons=projected_polygons,
            contributor_parts={(slope_id, part)},
        )
        return
    group.projected_polygons.extend(projected_polygons)
    group.contributor_parts.add((slope_id, part))


def _append_joined_side_return_groups(
    groups_by_plane: dict[
        tuple[str, float, float, float],
        _JoinedSideReturnGroup,
    ],
    vertices: list[tuple[float, float, float]],
    vertex_index: dict[tuple[float, float, float], int],
    faces: list[tuple[int, int, int]],
    face_parts: dict[int, tuple[str, str]],
    surface_alias_parts: dict[
        tuple[str, str],
        set[tuple[str, str]],
    ],
) -> None:
    """Union and emit one selectable mesh for every joined physical side."""

    for plane_key in sorted(groups_by_plane):
        group = groups_by_plane[plane_key]
        united = shapely.union_all(group.projected_polygons)
        world_triangles: list[tuple[tuple[float, float, float], ...]] = []
        for polygon in _iter_polygon_parts(united):
            if polygon.area <= SLOPED_CEILING_GEOMETRY_EPSILON:
                continue
            buffered_polygon = polygon.buffer(
                SLOPED_CEILING_GEOMETRY_EPSILON
            )
            triangulation = shapely.constrained_delaunay_triangles(polygon)
            for triangle in shapely.get_parts(triangulation):
                if (
                    not isinstance(triangle, Polygon)
                    or not buffered_polygon.covers(triangle)
                ):
                    continue
                projected_coordinates = tuple(triangle.exterior.coords)[:-1]
                if len(projected_coordinates) != 3:
                    continue
                world_triangles.append(
                    tuple(
                        (
                            float(
                                (
                                    group.plane_normal_xy * group.plane_offset
                                    + group.plane_tangent_xy * along
                                )[0]
                            ),
                            float(
                                (
                                    group.plane_normal_xy * group.plane_offset
                                    + group.plane_tangent_xy * along
                                )[1]
                            ),
                            float(height),
                        )
                        for along, height in projected_coordinates
                    )
                )
        if not world_triangles:
            continue
        owner = (group.owner_slope_id, group.owner_part)
        aliases = group.contributor_parts.difference((owner,))
        if aliases:
            surface_alias_parts.setdefault(owner, set()).update(aliases)
        _append_semantic_triangles(
            vertices,
            vertex_index,
            faces,
            face_parts,
            world_triangles,
            group.desired_normal,
            group.owner_slope_id,
            group.owner_part,
        )


def _iter_polygon_parts(geometry: BaseGeometry) -> Iterable[Polygon]:
    """Yield every polygon nested in one Shapely result."""

    if isinstance(geometry, Polygon):
        yield geometry
        return
    if geometry.geom_type not in {"GeometryCollection", "MultiPolygon"}:
        return
    for part in shapely.get_parts(geometry):
        yield from _iter_polygon_parts(part)


def _append_semantic_triangles(
    vertices: list[tuple[float, float, float]],
    vertex_index: dict[tuple[float, float, float], int],
    faces: list[tuple[int, int, int]],
    face_parts: dict[int, tuple[str, str]],
    triangles: Sequence[Sequence[Sequence[float]]],
    desired_normal: np.ndarray,
    slope_id: str,
    part: str,
) -> None:
    """Append triangles while retaining one semantic slope-part owner."""

    for triangle in triangles:
        face_index = _append_oriented_triangle(
            vertices,
            vertex_index,
            faces,
            triangle,
            desired_normal,
        )
        if face_index is not None:
            face_parts[face_index] = (slope_id, part)


def _append_internal_wall_contact_returns(
    ceiling_polygon: BaseGeometry,
    frames: Sequence[_SlopeFrame],
    ceiling_z: float,
    vertices: list[tuple[float, float, float]],
    vertex_index: dict[tuple[float, float, float], int],
    faces: list[tuple[int, int, int]],
    face_parts: dict[int, tuple[str, str]],
) -> None:
    """Close the height seam where a slope starts on an interior host wall."""

    buffered_polygon = ceiling_polygon.buffer(SLOPED_CEILING_GEOMETRY_EPSILON)
    for frame in frames:
        contact_start = (
            frame.wall_start[:2]
            + frame.tangent_xy * frame.start_distance
        )
        contact_end = (
            frame.wall_start[:2]
            + frame.tangent_xy * frame.end_distance
        )
        clipped = ceiling_polygon.intersection(
            LineString((tuple(contact_start), tuple(contact_end)))
        )
        for part in shapely.get_parts(clipped):
            if not isinstance(part, LineString) or part.is_empty:
                continue
            coordinates = np.asarray(part.coords, dtype=float)
            if len(coordinates) < 2:
                continue
            tangent_distances = (
                coordinates[[0, -1]] - frame.wall_start[:2]
            ) @ frame.tangent_xy
            minimum_tangent, maximum_tangent = sorted(tangent_distances)
            samples = {
                float(minimum_tangent),
                float(maximum_tangent),
                *(
                    float(distance)
                    for distance in _slope_tangent_breakpoints(frame)
                    if minimum_tangent
                    < distance
                    < maximum_tangent
                ),
            }
            for first_tangent, second_tangent in pairwise(sorted(samples)):
                middle_tangent = (first_tangent + second_tangent) * 0.5
                midpoint = (
                    frame.wall_start[:2]
                    + frame.tangent_xy * middle_tangent
                )
                probe = frame.inward_xy * SLOPED_CEILING_BOUNDARY_PROBE_METERS
                if not (
                    buffered_polygon.covers(shapely.Point(tuple(midpoint + probe)))
                    and buffered_polygon.covers(
                        shapely.Point(tuple(midpoint - probe))
                    )
                ):
                    continue
                first = (
                    frame.wall_start[:2]
                    + frame.tangent_xy * first_tangent
                )
                second = (
                    frame.wall_start[:2]
                    + frame.tangent_xy * second_tangent
                )
                tangent_probe = (
                    frame.tangent_xy
                    * SLOPED_CEILING_BOUNDARY_PROBE_METERS
                )
                inside_first_z = ceiling_z - _ceiling_drop_at_xy(
                    first,
                    frames,
                    lateral_reference_xy=first + probe + tangent_probe,
                )
                inside_second_z = ceiling_z - _ceiling_drop_at_xy(
                    second,
                    frames,
                    lateral_reference_xy=second + probe - tangent_probe,
                )
                outside_first_z = ceiling_z - _ceiling_drop_at_xy(
                    first,
                    frames,
                    lateral_reference_xy=first - probe + tangent_probe,
                )
                outside_second_z = ceiling_z - _ceiling_drop_at_xy(
                    second,
                    frames,
                    lateral_reference_xy=second - probe - tangent_probe,
                )
                outside_first = (
                    float(first[0]),
                    float(first[1]),
                    float(outside_first_z),
                )
                outside_second = (
                    float(second[0]),
                    float(second[1]),
                    float(outside_second_z),
                )
                inside_first = (
                    float(first[0]),
                    float(first[1]),
                    float(inside_first_z),
                )
                inside_second = (
                    float(second[0]),
                    float(second[1]),
                    float(inside_second_z),
                )
                desired_normal = np.asarray(
                    (
                        -frame.inward_xy[0],
                        -frame.inward_xy[1],
                        0.0,
                    ),
                    dtype=float,
                )
                first_face_index = _append_oriented_triangle(
                    vertices,
                    vertex_index,
                    faces,
                    (outside_first, outside_second, inside_second),
                    desired_normal,
                )
                second_face_index = _append_oriented_triangle(
                    vertices,
                    vertex_index,
                    faces,
                    (outside_first, inside_second, inside_first),
                    desired_normal,
                )
                for face_index in (first_face_index, second_face_index):
                    if face_index is not None:
                        face_parts[face_index] = (
                            frame.data.slope_id,
                            SLOPED_CEILING_PART_WALL_RETURN,
                        )


def _append_oriented_triangle(
    vertices: list[tuple[float, float, float]],
    vertex_index: dict[tuple[float, float, float], int],
    faces: list[tuple[int, int, int]],
    points: Sequence[Sequence[float]],
    desired_normal: np.ndarray,
) -> int | None:
    """Append one measurable triangle facing the requested world direction."""

    triangle = np.asarray(points, dtype=float)
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    if float(np.linalg.norm(normal)) <= SLOPED_CEILING_GEOMETRY_EPSILON:
        return None
    indices = [
        _append_indexed_vertex(vertices, vertex_index, point)
        for point in triangle
    ]
    if float(np.dot(normal, desired_normal)) < 0.0:
        indices.reverse()
    faces.append(tuple(indices))
    return len(faces) - 1


# ### Wall geometry helpers ###
def _rebuild_wall_surface(
    surface: object,
    frames: Sequence[_SlopeFrame],
    ceiling_z: float | None,
) -> object:
    if ceiling_z is None or not _wall_intersects_any_slope(surface, frames, ceiling_z):
        return surface
    wall_start = np.asarray(surface.wall_start_world, dtype=float)
    wall_end = np.asarray(surface.wall_end_world, dtype=float)
    wall_delta = wall_end[:2] - wall_start[:2]
    wall_length = float(np.linalg.norm(wall_delta))
    if wall_length <= SLOPED_CEILING_GEOMETRY_EPSILON:
        return surface
    tangent = wall_delta / wall_length
    plane_normal = np.asarray((-tangent[1], tangent[0]), dtype=float)
    mesh = surface.mesh
    source_vertices = np.asarray(mesh.vertices, dtype=float)
    main_polygons: list[Polygon] = []
    passthrough_triangles: list[np.ndarray] = []
    source_normals: list[np.ndarray] = []
    for face, normal in zip(np.asarray(mesh.faces, dtype=np.int64), mesh.face_normals):
        triangle = source_vertices[face]
        plane_distances = (triangle[:, :2] - wall_start[:2]) @ plane_normal
        if np.max(np.abs(plane_distances)) > SLOPED_CEILING_GEOMETRY_EPSILON * 10.0:
            passthrough_triangles.append(triangle.copy())
            continue
        local = [
            (
                float(np.dot(vertex[:2] - wall_start[:2], tangent) / wall_length),
                float(vertex[2] - wall_start[2]),
            )
            for vertex in triangle
        ]
        polygon = Polygon(local)
        if polygon.area > SLOPED_CEILING_GEOMETRY_EPSILON:
            main_polygons.append(polygon)
            source_normals.append(np.asarray(normal, dtype=float))
    if not main_polygons:
        return surface

    visible_geometry = shapely.union_all(main_polygons)
    profile = _build_wall_top_profile(surface, frames, ceiling_z)
    top_polygon = Polygon([(0.0, 0.0), (1.0, 0.0), *reversed(profile)])
    clipped = visible_geometry.intersection(top_polygon)
    target_normal = _resolve_main_wall_normal(source_normals, plane_normal)
    vertices: list[tuple[float, float, float]] = []
    vertex_index: dict[tuple[float, float, float], int] = {}
    faces: list[tuple[int, int, int]] = []
    for part in shapely.get_parts(clipped):
        if not isinstance(part, Polygon) or part.is_empty:
            continue
        buffered_part = part.buffer(SLOPED_CEILING_GEOMETRY_EPSILON)
        for triangle in shapely.get_parts(shapely.constrained_delaunay_triangles(part)):
            if not isinstance(triangle, Polygon) or triangle.is_empty:
                continue
            if not buffered_part.covers(triangle):
                continue
            points = list(triangle.exterior.coords)[:-1]
            if len(points) != 3:
                continue
            world_points = [
                (
                    float(wall_start[0] + tangent[0] * ratio * wall_length),
                    float(wall_start[1] + tangent[1] * ratio * wall_length),
                    float(wall_start[2] + height),
                )
                for ratio, height in points
            ]
            indices = [
                _append_indexed_vertex(vertices, vertex_index, point)
                for point in world_points
            ]
            triangle_normal = np.cross(
                np.asarray(world_points[1]) - np.asarray(world_points[0]),
                np.asarray(world_points[2]) - np.asarray(world_points[0]),
            )
            if float(np.dot(triangle_normal, target_normal)) < 0.0:
                indices.reverse()
            faces.append(tuple(indices))

    for triangle in passthrough_triangles:
        indices = [
            _append_indexed_vertex(vertices, vertex_index, tuple(point))
            for point in triangle
        ]
        faces.append(tuple(indices))
    rebuilt = _build_mesh(vertices, faces)
    if rebuilt is None:
        return surface
    return replace(
        surface,
        mesh=rebuilt,
        area_square_meters=float(rebuilt.area),
    )


def _build_wall_top_profile(
    surface: object,
    frames: Sequence[_SlopeFrame],
    ceiling_z: float,
) -> tuple[tuple[float, float], ...]:
    wall_start = np.asarray(surface.wall_start_world, dtype=float)
    wall_end = np.asarray(surface.wall_end_world, dtype=float)
    ratios = _build_wall_profile_ratios(surface, frames)
    wall_delta = wall_end[:2] - wall_start[:2]
    points: list[tuple[float, float]] = []
    for ratio_index, ratio in enumerate(ratios):
        point_xy = wall_start[:2] + wall_delta * ratio
        if (
            ratio_index > 0
            and ratio_index + 1 < len(ratios)
            and _point_is_exposed_lateral_boundary(point_xy, frames)
        ):
            probe_ratio = min(
                1e-6,
                (ratio - ratios[ratio_index - 1]) * 0.25,
                (ratios[ratio_index + 1] - ratio) * 0.25,
            )
            left_point = wall_start[:2] + wall_delta * (ratio - probe_ratio)
            right_point = wall_start[:2] + wall_delta * (ratio + probe_ratio)
            left_height = max(
                0.0,
                float(
                    ceiling_z
                    - _ceiling_drop_at_xy(left_point, frames)
                    - wall_start[2]
                ),
            )
            right_height = max(
                0.0,
                float(
                    ceiling_z
                    - _ceiling_drop_at_xy(right_point, frames)
                    - wall_start[2]
                ),
            )
            if not math.isclose(
                left_height,
                right_height,
                abs_tol=SLOPED_CEILING_GEOMETRY_EPSILON,
                rel_tol=0.0,
            ):
                points.extend(
                    (
                        (float(ratio), left_height),
                        (float(ratio), right_height),
                    )
                )
                continue
        top_z = ceiling_z - _ceiling_drop_at_xy(point_xy, frames)
        points.append((float(ratio), max(0.0, float(top_z - wall_start[2]))))
    return tuple(points)


def _point_is_exposed_lateral_boundary(
    point_xy: Sequence[float],
    frames: Sequence[_SlopeFrame],
) -> bool:
    """Return whether a point lies on one free selected-width boundary."""

    point = np.asarray(point_xy, dtype=float)
    for frame in frames:
        relative = point - frame.wall_start[:2]
        depth = float(np.dot(relative, frame.inward_xy))
        if (
            depth < -SLOPED_CEILING_GEOMETRY_EPSILON
            or depth
            > frame.data.depth_meters + SLOPED_CEILING_GEOMETRY_EPSILON
        ):
            continue
        tangent = float(np.dot(relative, frame.tangent_xy))
        if (
            _start_side_is_internal(frame)
            and math.isclose(
                tangent,
                frame.start_distance,
                abs_tol=SLOPED_CEILING_GEOMETRY_EPSILON,
            )
        ):
            return True
        if (
            _end_side_is_internal(frame)
            and math.isclose(
                tangent,
                frame.end_distance,
                abs_tol=SLOPED_CEILING_GEOMETRY_EPSILON,
            )
        ):
            return True
    return False


def _wall_intersects_any_slope(
    surface: object,
    frames: Sequence[_SlopeFrame],
    ceiling_z: float,
) -> bool:
    wall_start = getattr(surface, "wall_start_world", None)
    wall_end = getattr(surface, "wall_end_world", None)
    if wall_start is None or wall_end is None:
        return False
    start = np.asarray(wall_start, dtype=float)
    end = np.asarray(wall_end, dtype=float)
    for ratio in _build_wall_profile_ratios(surface, frames):
        point = start[:2] + (end[:2] - start[:2]) * ratio
        if _ceiling_drop_at_xy(point, frames) > SLOPED_CEILING_GEOMETRY_EPSILON:
            return True
    return False


def _build_wall_profile_ratios(
    surface: object,
    frames: Sequence[_SlopeFrame],
) -> tuple[float, ...]:
    """Include every sampled slope-grid crossing on one wall segment."""

    start = np.asarray(surface.wall_start_world, dtype=float)[:2]
    end = np.asarray(surface.wall_end_world, dtype=float)[:2]
    delta = end - start
    ratios = {
        round(float(value), 12)
        for value in np.linspace(0.0, 1.0, SLOPED_CEILING_PROFILE_SAMPLE_COUNT)
    }
    for frame in frames:
        relative_start = start - frame.wall_start[:2]
        for axis, values in (
            (frame.tangent_xy, _slope_tangent_breakpoints(frame)),
            (frame.inward_xy, _slope_depth_breakpoints(frame)),
        ):
            denominator = float(np.dot(delta, axis))
            if abs(denominator) <= SLOPED_CEILING_GEOMETRY_EPSILON:
                continue
            start_coordinate = float(np.dot(relative_start, axis))
            for coordinate in values:
                ratio = (coordinate - start_coordinate) / denominator
                if (
                    -SLOPED_CEILING_GEOMETRY_EPSILON
                    <= ratio
                    <= (1.0 + SLOPED_CEILING_GEOMETRY_EPSILON)
                ):
                    ratios.add(round(min(1.0, max(0.0, ratio)), 12))
    return tuple(sorted(ratios))


def _resolve_main_wall_normal(
    normals: Sequence[np.ndarray],
    fallback_xy: np.ndarray,
) -> np.ndarray:
    if normals:
        summed = np.sum(np.asarray(normals, dtype=float), axis=0)
        length = float(np.linalg.norm(summed))
        if length > SLOPED_CEILING_GEOMETRY_EPSILON:
            return summed / length
    return np.asarray((fallback_xy[0], fallback_xy[1], 0.0), dtype=float)


# ### Preview geometry helpers ###
def _build_preview_sampling_values(
    frame: _SlopeFrame,
    host_frames: Sequence[_SlopeFrame],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Include parallel peers' profile bends in the lightweight preview grid."""

    tangent_values = set(_slope_tangent_breakpoints(frame))
    depth_values = set(_slope_depth_breakpoints(frame))
    for peer in host_frames:
        if peer is frame or abs(float(np.dot(frame.tangent_xy, peer.tangent_xy))) < (
            1.0 - SLOPED_CEILING_GEOMETRY_EPSILON
        ):
            continue
        offset = peer.wall_start[:2] - frame.wall_start[:2]
        tangent_offset = float(np.dot(offset, frame.tangent_xy))
        tangent_sign = float(np.dot(peer.tangent_xy, frame.tangent_xy))
        peer_tangent_values = tuple(
            tangent_offset + tangent_sign * value
            for value in _slope_tangent_breakpoints(peer)
        )
        if (
            max(peer_tangent_values) < frame.start_distance
            or min(peer_tangent_values) > frame.end_distance
        ):
            continue
        depth_offset = float(np.dot(offset, frame.inward_xy))
        depth_sign = float(np.dot(peer.inward_xy, frame.inward_xy))
        peer_depth_values = tuple(
            depth_offset + depth_sign * value
            for value in _slope_depth_breakpoints(peer)
        )
        if max(peer_depth_values) < 0.0 or min(peer_depth_values) > frame.data.depth_meters:
            continue
        tangent_values.update(
            value
            for value in peer_tangent_values
            if frame.start_distance + SLOPED_CEILING_GEOMETRY_EPSILON
            < value
            < frame.end_distance - SLOPED_CEILING_GEOMETRY_EPSILON
        )
        depth_values.update(
            value
            for value in peer_depth_values
            if SLOPED_CEILING_GEOMETRY_EPSILON
            < value
            < frame.data.depth_meters - SLOPED_CEILING_GEOMETRY_EPSILON
        )
    return (
        _merge_preview_sample_values(tangent_values),
        _merge_preview_sample_values(depth_values),
    )


def _merge_preview_sample_values(values: Collection[float]) -> tuple[float, ...]:
    """Avoid near-zero-width grid strips caused by projected float roundoff."""

    ordered = sorted(values)
    merged: list[float] = []
    for value in ordered:
        if not merged or value - merged[-1] > SLOPED_CEILING_GEOMETRY_EPSILON:
            merged.append(value)
    if len(merged) > 1:
        merged[-1] = ordered[-1]
    return tuple(merged)


def _build_preview_mesh(
    frame: _SlopeFrame,
    host_frames: Sequence[_SlopeFrame],
    *,
    host_ceiling_geometry: BaseGeometry | None = None,
) -> trimesh.Trimesh | None:
    """Preview the shared slope only where the committed ceiling can exist."""

    clipped_footprint: BaseGeometry | None = None
    if host_ceiling_geometry is not None:
        start = frame.wall_start[:2] + frame.tangent_xy * frame.start_distance
        end = frame.wall_start[:2] + frame.tangent_xy * frame.end_distance
        offset = frame.inward_xy * frame.data.depth_meters
        footprint = Polygon((start, end, end + offset, start + offset))
        if not host_ceiling_geometry.covers(footprint):
            clipped_footprint = host_ceiling_geometry.intersection(footprint)
            if clipped_footprint.area <= SLOPED_CEILING_GEOMETRY_EPSILON:
                return None

    tangent_values, depth_values = _build_preview_sampling_values(frame, host_frames)
    if len(tangent_values) < 2 or len(depth_values) < 2:
        return None
    vertices: list[tuple[float, float, float]] = []
    tangent_offsets = np.asarray(tangent_values)[:, np.newaxis] * frame.tangent_xy
    for depth in depth_values:
        points_xy = frame.wall_start[:2] + tangent_offsets + frame.inward_xy * depth
        drops = np.zeros(len(points_xy), dtype=float)
        for peer in host_frames:
            if abs(float(np.dot(frame.tangent_xy, peer.inward_xy))) <= 1e-12:
                peer_drops = _get_parallel_preview_row_drops(points_xy, peer)
            else:
                peer_drops = np.fromiter(
                    (_single_slope_drop_at_xy(point, peer) for point in points_xy),
                    dtype=float,
                    count=len(points_xy),
                )
            np.maximum(drops, peer_drops, out=drops)
        vertices.extend(
            (float(point[0]), float(point[1]), float(frame.ceiling_z - drop))
            for point, drop in zip(points_xy, drops)
        )
    row_width = len(tangent_values)
    faces: list[tuple[int, int, int]] = []
    for row in range(len(depth_values) - 1):
        for column in range(row_width - 1):
            first = row * row_width + column
            second = first + 1
            fourth = first + row_width
            third = fourth + 1
            faces.extend(((first, third, second), (first, fourth, third)))
    mesh = _build_mesh(vertices, faces)
    if mesh is not None and clipped_footprint is not None:
        return _clip_preview_mesh_to_footprint(
            mesh, clipped_footprint, frame, host_frames
        )
    return mesh


def _clip_preview_mesh_to_footprint(
    mesh: trimesh.Trimesh,
    footprint: BaseGeometry,
    frame: _SlopeFrame,
    host_frames: Sequence[_SlopeFrame],
) -> trimesh.Trimesh | None:
    """Retain the grid and tessellate only triangles crossing the host boundary."""

    triangle_polygons = shapely.polygons(np.asarray(mesh.triangles)[:, :, :2])
    inside = shapely.covers(footprint, triangle_polygons)
    crossing = ~inside & shapely.intersects(footprint, triangle_polygons)
    retained_meshes: list[trimesh.Trimesh] = []
    if np.any(inside):
        retained_meshes.append(_extract_mesh_faces(mesh, np.flatnonzero(inside)))
    boundary_parts = shapely.intersection(triangle_polygons[crossing], footprint)
    vertices: list[tuple[float, float, float]] = []
    vertex_index: dict[tuple[float, float, float], int] = {}
    faces: list[tuple[int, int, int]] = []
    normal = np.asarray(mesh.face_normals[0], dtype=float)
    for intersection in boundary_parts:
        for part in _iter_polygon_parts(intersection):
            if part.area <= SLOPED_CEILING_GEOMETRY_EPSILON:
                continue
            representative = part.representative_point().coords[0]
            buffered_part = part.buffer(SLOPED_CEILING_GEOMETRY_EPSILON)
            for triangle in shapely.get_parts(shapely.constrained_delaunay_triangles(part)):
                if not isinstance(triangle, Polygon) or not buffered_part.covers(triangle):
                    continue
                world_points = tuple(
                    (
                        float(x),
                        float(y),
                        frame.ceiling_z
                        - _ceiling_drop_at_xy(
                            (x, y), host_frames, lateral_reference_xy=representative
                        ),
                    )
                    for x, y in tuple(triangle.exterior.coords)[:-1]
                )
                _append_oriented_triangle(
                    vertices, vertex_index, faces, world_points, normal
                )
    boundary_mesh = _build_mesh(vertices, faces)
    if boundary_mesh is not None:
        retained_meshes.append(boundary_mesh)
    if not retained_meshes:
        return None
    return (
        retained_meshes[0]
        if len(retained_meshes) == 1
        else trimesh.util.concatenate(retained_meshes)
    )


def _get_parallel_preview_row_drops(
    points_xy: np.ndarray,
    frame: _SlopeFrame,
) -> np.ndarray:
    """Reuse one exact profile evaluation across a parallel preview row."""

    relative = points_xy - frame.wall_start[:2]
    tangent_distances = relative @ frame.tangent_xy
    inward_distance = float(np.dot(relative[0], frame.inward_xy))
    if (
        inward_distance < -SLOPED_CEILING_GEOMETRY_EPSILON
        or inward_distance > frame.data.depth_meters + SLOPED_CEILING_GEOMETRY_EPSILON
    ):
        return np.zeros(len(points_xy), dtype=float)
    profile_drop = _profile_drop_at_depth(frame.profile, inward_distance)
    radius = _effective_lateral_edge_radius(inward_distance, profile_drop, frame)
    drops = np.full(len(points_xy), profile_drop, dtype=float)
    if radius > SLOPED_CEILING_GEOMETRY_EPSILON:
        edge_distances = np.full(len(points_xy), np.inf, dtype=float)
        if _start_side_is_internal(frame):
            np.minimum(
                edge_distances,
                np.maximum(0.0, tangent_distances - frame.start_distance),
                out=edge_distances,
            )
        if _end_side_is_internal(frame):
            np.minimum(
                edge_distances,
                np.maximum(0.0, frame.end_distance - tangent_distances),
                out=edge_distances,
            )
        near_edge = edge_distances < radius
        offsets = edge_distances[near_edge] - radius
        drops[near_edge] -= radius - np.sqrt(
            np.maximum(0.0, radius * radius - offsets * offsets)
        )
    np.clip(drops, 0.0, frame.maximum_drop, out=drops)
    outside_span = (
        tangent_distances < frame.start_distance - SLOPED_CEILING_GEOMETRY_EPSILON
    ) | (tangent_distances > frame.end_distance + SLOPED_CEILING_GEOMETRY_EPSILON)
    drops[outside_span] = 0.0
    return drops


# ### Generic helpers ###
def _append_indexed_vertex(
    vertices: list[tuple[float, float, float]],
    index_by_point: dict[tuple[float, float, float], int],
    point: Sequence[float],
) -> int:
    normalized = tuple(round(float(value), 10) for value in point)
    index = index_by_point.get(normalized)
    if index is not None:
        return index
    index = len(vertices)
    vertices.append(tuple(float(value) for value in point))
    index_by_point[normalized] = index
    return index


def _build_mesh(
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
) -> trimesh.Trimesh | None:
    if not vertices or not faces:
        return None
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    if not len(mesh.faces) or not math.isfinite(float(mesh.area)):
        return None
    return mesh


def _smoothstep(value: float) -> float:
    clamped = min(1.0, max(0.0, float(value)))
    return clamped * clamped * (3.0 - 2.0 * clamped)


def _clamp_sloped_window_ratio_pair(
    first_value: float,
    second_value: float,
    axis_name: str,
) -> tuple[float, float]:
    """Validate two ratios and clamp harmless pointer precision overflow."""

    minimum, maximum = sorted((float(first_value), float(second_value)))
    tolerance = 1e-6
    if minimum < -tolerance or maximum > 1.0 + tolerance:
        raise ValueError(
            f"A sloped window must remain inside the host's {axis_name} bounds."
        )
    return max(0.0, minimum), min(1.0, maximum)


def _finite_world_point(value: object, label: str) -> np.ndarray:
    """Return one finite XYZ vector suitable for planar projection."""

    try:
        point = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"{label.capitalize()} must contain finite XYZ values."
        ) from error
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{label.capitalize()} must contain finite XYZ values.")
    return point


def _normalized_vector(value: object, label: str) -> np.ndarray:
    """Return one finite non-zero unit vector."""

    vector = _finite_world_point(value, label)
    length = float(np.linalg.norm(vector))
    if length <= SLOPED_CEILING_GEOMETRY_EPSILON:
        raise ValueError(f"{label.capitalize()} cannot be zero.")
    return vector / length


def _positive_finite_number(value: object, label: str) -> float:
    """Return one finite strictly positive scalar."""

    number = _finite_number(value, label)
    if number <= SLOPED_CEILING_GEOMETRY_EPSILON:
        raise ValueError(f"{label.capitalize()} must be positive.")
    return number


def _vector_tuple(value: object) -> tuple[float, float, float]:
    """Return a normalized Python tuple without retaining NumPy scalars."""

    vector = np.asarray(value, dtype=float)
    return tuple(float(item) for item in vector)


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"Sloped-ceiling {label} must be numeric.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"Sloped-ceiling {label} must be numeric.") from error
    if not math.isfinite(result):
        raise ValueError(f"Sloped-ceiling {label} must be finite.")
    return result


def _validate_finite_vector(value: object, label: str) -> None:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"Sloped-ceiling {label} must contain finite XYZ values.")
