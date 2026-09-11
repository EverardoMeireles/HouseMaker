# ### Imports ###
from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, replace
import math
import re

import numpy as np

from housemaker.level_coordinates import (
    get_level_world_pivot,
    level_image_to_world_xy,
)
from housemaker.models import (
    MAX_FLOOR_THICKNESS_METERS,
    MAX_LEVEL_OFFSET_METERS,
    MIN_FLOOR_THICKNESS_METERS,
    MIN_LEVEL_OFFSET_METERS,
    PIXEL_TO_METER,
    LevelData,
    RoomData,
    Vertex,
)
from housemaker.surface_geometry import (
    SURFACE_TYPE_CEILING,
    SURFACE_TYPE_FLOOR,
    SURFACE_TYPE_WALL,
    FixedSurface,
    build_fixed_surfaces,
)


# ### Constants ###
CANVAS_SURFACE_EDIT_WALL_VERTEX = "wall_vertex"
CANVAS_SURFACE_EDIT_WALL_TRANSLATION = "wall_translation"
CANVAS_SURFACE_EDIT_LEVEL_HEIGHT = "level_height"
CANVAS_SURFACE_EDIT_ROOM_HEIGHT = "room_height"
CANVAS_SURFACE_EDIT_FLOOR_THICKNESS = "floor_thickness"
CANVAS_SURFACE_EDIT_WALL_KINDS = frozenset(
    (
        CANVAS_SURFACE_EDIT_WALL_VERTEX,
        CANVAS_SURFACE_EDIT_WALL_TRANSLATION,
    )
)
CANVAS_SURFACE_EDIT_KINDS = frozenset(
    (
        *CANVAS_SURFACE_EDIT_WALL_KINDS,
        CANVAS_SURFACE_EDIT_LEVEL_HEIGHT,
        CANVAS_SURFACE_EDIT_ROOM_HEIGHT,
        CANVAS_SURFACE_EDIT_FLOOR_THICKNESS,
    )
)
CANVAS_SURFACE_EDIT_AXIS_X = 0
CANVAS_SURFACE_EDIT_AXIS_Y = 1
CANVAS_SURFACE_EDIT_AXIS_Z = 2
CANVAS_SURFACE_EDIT_AXES = frozenset(
    (
        CANVAS_SURFACE_EDIT_AXIS_X,
        CANVAS_SURFACE_EDIT_AXIS_Y,
        CANVAS_SURFACE_EDIT_AXIS_Z,
    )
)
CANVAS_SURFACE_EDIT_AXIS_NAMES = ("x", "y", "z")
MIN_CANVAS_WALL_LENGTH_METERS = 0.05
MIN_CANVAS_SURFACE_HEIGHT_METERS = 0.1
MAX_CANVAS_SURFACE_HEIGHT_METERS = 100.0
MAX_CANVAS_WALL_EDIT_DELTA_METERS = 10_000.0
CANVAS_SURFACE_EDIT_EPSILON = 1e-9
CANVAS_WALL_CHAIN_RELATIVE_TOLERANCE = 1e-7
CANVAS_WALL_KEY_PATTERN = re.compile(
    r"^(?P<first>[1-9]\d*):(?P<second>[1-9]\d*)$"
)


# ### Edit models ###
@dataclass(frozen=True)
class CanvasSurfaceEditReference:
    """Persistent project property controlled by one Canvas axis handle."""

    kind: str
    level_index: int
    axis_index: int
    vertex_id: int | None = None
    room_center_vertex_id: int | None = None
    wall_vertex_ids: tuple[int, int] = ()

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        if kind not in CANVAS_SURFACE_EDIT_KINDS:
            raise ValueError(f"Unknown Canvas surface edit kind: {kind!r}.")
        level_index = _normalize_integer(self.level_index, "level index")
        axis_index = _normalize_integer(self.axis_index, "axis index")
        if axis_index not in CANVAS_SURFACE_EDIT_AXES:
            raise ValueError("Canvas surface edit axes must be X, Y, or Z.")
        vertex_id = _normalize_optional_positive_integer(
            self.vertex_id,
            "vertex ID",
        )
        room_center_vertex_id = _normalize_optional_positive_integer(
            self.room_center_vertex_id,
            "room center vertex ID",
        )
        wall_vertex_ids = tuple(
            sorted(
                _normalize_positive_integer(value, "wall vertex ID")
                for value in self.wall_vertex_ids
            )
        )

        if kind == CANVAS_SURFACE_EDIT_WALL_VERTEX:
            if (
                vertex_id is None
                or room_center_vertex_id is not None
                or wall_vertex_ids
            ):
                raise ValueError(
                    "Wall edits require one persistent vertex ID only."
                )
            if axis_index not in {
                CANVAS_SURFACE_EDIT_AXIS_X,
                CANVAS_SURFACE_EDIT_AXIS_Y,
            }:
                raise ValueError("Wall vertices can only move on X or Y.")
        elif kind == CANVAS_SURFACE_EDIT_WALL_TRANSLATION:
            if (
                vertex_id is not None
                or room_center_vertex_id is not None
                or len(wall_vertex_ids) != 2
                or len(set(wall_vertex_ids)) != 2
            ):
                raise ValueError(
                    "Wall translation edits require two distinct endpoint IDs."
                )
            if axis_index not in {
                CANVAS_SURFACE_EDIT_AXIS_X,
                CANVAS_SURFACE_EDIT_AXIS_Y,
            }:
                raise ValueError("Walls can only move on X or Y.")
        elif kind == CANVAS_SURFACE_EDIT_ROOM_HEIGHT:
            if (
                room_center_vertex_id is None
                or vertex_id is not None
                or wall_vertex_ids
            ):
                raise ValueError(
                    "Room-height edits require one room center vertex ID only."
                )
            if axis_index != CANVAS_SURFACE_EDIT_AXIS_Z:
                raise ValueError("Room height can only change on Z.")
        else:
            if (
                vertex_id is not None
                or room_center_vertex_id is not None
                or wall_vertex_ids
            ):
                raise ValueError(
                    "Level-wide edits have no item owner ID."
                )
            if axis_index != CANVAS_SURFACE_EDIT_AXIS_Z:
                raise ValueError("Level-wide values can only change on Z.")

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "level_index", level_index)
        object.__setattr__(self, "axis_index", axis_index)
        object.__setattr__(self, "vertex_id", vertex_id)
        object.__setattr__(
            self,
            "room_center_vertex_id",
            room_center_vertex_id,
        )
        object.__setattr__(self, "wall_vertex_ids", wall_vertex_ids)

    @property
    def key(self) -> str:
        """Return a stable key shared by duplicate views of one property."""

        axis_name = CANVAS_SURFACE_EDIT_AXIS_NAMES[self.axis_index]
        if self.kind == CANVAS_SURFACE_EDIT_WALL_VERTEX:
            return (
                f"level:{self.level_index}/vertex:{self.vertex_id}/"
                f"axis:{axis_name}"
            )
        if self.kind == CANVAS_SURFACE_EDIT_WALL_TRANSLATION:
            first_id, second_id = self.wall_vertex_ids
            return (
                f"level:{self.level_index}/wall:{first_id}:{second_id}/"
                f"translation/axis:{axis_name}"
            )
        if self.kind == CANVAS_SURFACE_EDIT_ROOM_HEIGHT:
            return (
                f"level:{self.level_index}/"
                f"room:{self.room_center_vertex_id}/height"
            )
        if self.kind == CANVAS_SURFACE_EDIT_FLOOR_THICKNESS:
            return f"level:{self.level_index}/floor_thickness"
        assert self.kind == CANVAS_SURFACE_EDIT_LEVEL_HEIGHT
        return f"level:{self.level_index}/height"


@dataclass(frozen=True)
class CanvasSurfaceEditHandleTarget:
    """One axis handle plus everything needed to replay its drag baseline."""

    reference: CanvasSurfaceEditReference
    surface_id: str
    origin_world: tuple[float, float, float]
    axis_world: tuple[float, float, float]
    minimum_delta_meters: float
    maximum_delta_meters: float
    baseline_value_meters: float = 0.0
    level_scale: float = 1.0
    level_offset_world: tuple[float, float] = (0.0, 0.0)
    level_pivot_world: tuple[float, float] = (0.0, 0.0)
    chain_vertex_ids: tuple[int, ...] = ()
    chain_vertex_ratios: tuple[float, ...] = ()
    chain_vertex_image_positions: tuple[tuple[float, float], ...] = ()
    required_surface_ids: tuple[str, ...] = ()
    level_image_size_pixels: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reference, CanvasSurfaceEditReference):
            raise TypeError("Canvas surface targets require an edit reference.")
        surface_id = str(self.surface_id).strip()
        if not surface_id:
            raise ValueError("Canvas surface targets require a surface ID.")
        origin = _normalize_vector(self.origin_world, "handle origin", unit=False)
        axis = _normalize_vector(self.axis_world, "handle axis", unit=True)
        expected_axis = np.eye(3, dtype=float)[self.reference.axis_index]
        if abs(float(np.dot(axis, expected_axis))) < 1.0 - 1e-7:
            raise ValueError("A Canvas handle axis must match its reference axis.")
        minimum_delta = _normalize_finite_number(
            self.minimum_delta_meters,
            "minimum edit delta",
        )
        maximum_delta = _normalize_finite_number(
            self.maximum_delta_meters,
            "maximum edit delta",
        )
        if minimum_delta > maximum_delta:
            raise ValueError("Canvas edit delta bounds are reversed.")
        if minimum_delta > 0.0 or maximum_delta < 0.0:
            raise ValueError("Canvas edit delta bounds must include zero.")
        baseline_value = _normalize_finite_number(
            self.baseline_value_meters,
            "baseline value",
        )
        level_scale = _normalize_positive_number(
            self.level_scale,
            "level scale",
        )
        level_offset = _normalize_pair(
            self.level_offset_world,
            "level offset",
        )
        level_pivot = _normalize_pair(
            self.level_pivot_world,
            "level pivot",
        )
        chain_vertex_ids = tuple(
            _normalize_positive_integer(value, "chain vertex ID")
            for value in self.chain_vertex_ids
        )
        chain_vertex_ratios = tuple(
            _normalize_finite_number(value, "chain vertex ratio")
            for value in self.chain_vertex_ratios
        )
        chain_positions = tuple(
            _normalize_pair(value, "chain vertex image position")
            for value in self.chain_vertex_image_positions
        )
        required_surface_ids = tuple(
            dict.fromkeys(
                normalized
                for value in self.required_surface_ids
                if (normalized := str(value).strip())
            )
        )
        image_size = _normalize_optional_positive_pair(
            self.level_image_size_pixels,
            "level image size",
        )
        _validate_target_chain(
            self.reference,
            chain_vertex_ids,
            chain_vertex_ratios,
            chain_positions,
        )

        object.__setattr__(self, "surface_id", surface_id)
        object.__setattr__(self, "origin_world", origin)
        object.__setattr__(self, "axis_world", axis)
        object.__setattr__(self, "minimum_delta_meters", minimum_delta)
        object.__setattr__(self, "maximum_delta_meters", maximum_delta)
        object.__setattr__(self, "baseline_value_meters", baseline_value)
        object.__setattr__(self, "level_scale", level_scale)
        object.__setattr__(self, "level_offset_world", level_offset)
        object.__setattr__(self, "level_pivot_world", level_pivot)
        object.__setattr__(self, "chain_vertex_ids", chain_vertex_ids)
        object.__setattr__(self, "chain_vertex_ratios", chain_vertex_ratios)
        object.__setattr__(
            self,
            "chain_vertex_image_positions",
            chain_positions,
        )
        object.__setattr__(self, "required_surface_ids", required_surface_ids)
        object.__setattr__(self, "level_image_size_pixels", image_size)

    def get_preview_origin(
        self,
        delta_meters: float,
    ) -> tuple[float, float, float]:
        """Return the handle origin after one absolute drag delta."""

        delta = _validate_target_delta(self, delta_meters)
        origin = np.asarray(self.origin_world, dtype=float)
        axis = np.asarray(self.axis_world, dtype=float)
        return tuple(float(value) for value in origin + axis * delta)


@dataclass(frozen=True)
class CanvasSurfaceEdit:
    """An absolute-from-drag-start change for one persistent property."""

    reference: CanvasSurfaceEditReference
    surface_id: str
    delta_meters: float

    def __post_init__(self) -> None:
        if not isinstance(self.reference, CanvasSurfaceEditReference):
            raise TypeError("Canvas surface edits require a reference.")
        surface_id = str(self.surface_id).strip()
        if not surface_id:
            raise ValueError("Canvas surface edits require a surface ID.")
        delta = _normalize_finite_number(self.delta_meters, "edit delta")
        object.__setattr__(self, "surface_id", surface_id)
        object.__setattr__(self, "delta_meters", delta)


@dataclass(frozen=True)
class AppliedCanvasSurfaceEdit:
    """The previous and current absolute values of one accepted edit."""

    reference: CanvasSurfaceEditReference
    level: LevelData
    previous: CanvasSurfaceEdit
    current: CanvasSurfaceEdit


@dataclass(frozen=True)
class _LevelMutationSnapshot:
    """Mutable level fields restored if authoritative validation fails."""

    vertices: tuple[Vertex, ...]
    offset_x_meters: float
    offset_y_meters: float
    height_meters: float
    floor_thickness_meters: float
    room_heights: tuple[tuple[int, float], ...]


# ### Target builders ###
def build_canvas_surface_edit_targets(
    levels: Sequence[LevelData],
    surfaces: Sequence[FixedSurface],
) -> tuple[CanvasSurfaceEditHandleTarget, ...]:
    """Build deterministic handles for editable structural surfaces."""

    level_sequence = _normalize_levels(levels)
    surface_sequence = _normalize_surfaces(surfaces)
    level_by_index = {level.index: level for level in level_sequence}
    required_ids_by_level: dict[int, tuple[str, ...]] = {}
    for level_index in level_by_index:
        required_ids_by_level[level_index] = tuple(
            sorted(
                surface.surface_id
                for surface in surface_sequence
                if surface.level_index == level_index
            )
        )

    targets: list[CanvasSurfaceEditHandleTarget] = []
    for surface in sorted(
        surface_sequence,
        key=lambda candidate: candidate.surface_id,
    ):
        # Persistent topology faces have their own normal extrusion control.
        # Structural wall/level handles belong only to their authoritative
        # source surface, never to one of its editable descendants.
        if getattr(surface, "source_surface_id", None) is not None:
            continue
        level = level_by_index.get(surface.level_index)
        if level is None:
            continue
        if surface.surface_type not in {
            SURFACE_TYPE_WALL,
            SURFACE_TYPE_CEILING,
            SURFACE_TYPE_FLOOR,
        }:
            continue
        common = _build_common_target_values(
            level,
            surface,
            required_ids_by_level[level.index],
        )
        if surface.surface_type == SURFACE_TYPE_WALL:
            targets.extend(_build_wall_targets(level, surface, common))
            continue
        target = (
            _build_floor_target(level, surface, common)
            if surface.surface_type == SURFACE_TYPE_FLOOR
            else _build_ceiling_target(level, surface, common)
        )
        if target is not None:
            targets.append(target)
    return tuple(
        sorted(
            targets,
            key=lambda target: (target.surface_id, target.reference.key),
        )
    )


def rebase_canvas_wall_edit_targets(
    levels: Sequence[LevelData],
    targets: Sequence[CanvasSurfaceEditHandleTarget],
) -> tuple[CanvasSurfaceEditHandleTarget, ...]:
    """Rebase one wall's handles onto current project data without meshing."""

    level_sequence = _normalize_levels(levels)
    try:
        target_sequence = tuple(targets)
    except TypeError as error:
        raise TypeError(
            "Canvas wall target rebasing requires a target sequence."
        ) from error
    if not target_sequence:
        return ()
    if not all(
        isinstance(target, CanvasSurfaceEditHandleTarget)
        for target in target_sequence
    ):
        raise TypeError(
            "Canvas wall target rebasing requires handle targets."
        )

    first_target = target_sequence[0]
    surface_id = first_target.surface_id
    level_index = first_target.reference.level_index
    if any(
        target.reference.kind not in CANVAS_SURFACE_EDIT_WALL_KINDS
        or target.surface_id != surface_id
        or target.reference.level_index != level_index
        for target in target_sequence
    ):
        raise ValueError(
            "Canvas wall target rebasing requires handles from one wall."
        )

    level = _find_level(level_sequence, level_index)
    return tuple(
        _rebase_canvas_wall_edit_target(level, target)
        for target in target_sequence
    )


def rebase_canvas_wall_edit_targets_batch(
    levels: Sequence[LevelData],
    targets: Sequence[CanvasSurfaceEditHandleTarget],
) -> tuple[CanvasSurfaceEditHandleTarget, ...]:
    """Rebase wall handles from any number of walls and levels."""

    level_sequence = _normalize_levels(levels)
    target_sequence = _normalize_wall_target_sequence(
        targets,
        "Canvas wall batch target rebasing",
        allow_empty=True,
    )
    return tuple(
        _rebase_canvas_wall_edit_target(
            _find_level(level_sequence, target.reference.level_index),
            target,
        )
        for target in target_sequence
    )


def rebase_canvas_floor_edit_target(
    levels: Sequence[LevelData],
    target: CanvasSurfaceEditHandleTarget,
) -> CanvasSurfaceEditHandleTarget:
    """Rebase a floor handle onto its current delayed thickness preview."""

    level_sequence = _normalize_levels(levels)
    if not isinstance(target, CanvasSurfaceEditHandleTarget):
        raise TypeError("Canvas floor target rebasing requires a handle target.")
    if target.reference.kind != CANVAS_SURFACE_EDIT_FLOOR_THICKNESS:
        raise ValueError("Canvas floor target rebasing requires a floor handle.")

    level = _find_level(level_sequence, target.reference.level_index)
    _validate_level_baseline(level, target)
    current_thickness = _validate_range(
        level.floor_thickness_meters,
        MIN_FLOOR_THICKNESS_METERS,
        MAX_FLOOR_THICKNESS_METERS,
        "Floor thickness",
    )
    thickness_delta = current_thickness - target.baseline_value_meters
    current_origin = (
        np.asarray(target.origin_world, dtype=float)
        + np.asarray(target.axis_world, dtype=float) * thickness_delta
    )
    return replace(
        target,
        origin_world=tuple(float(value) for value in current_origin),
        minimum_delta_meters=(
            MIN_FLOOR_THICKNESS_METERS - current_thickness
        ),
        maximum_delta_meters=(
            MAX_FLOOR_THICKNESS_METERS - current_thickness
        ),
        baseline_value_meters=current_thickness,
    )


def _build_common_target_values(
    level: LevelData,
    surface: FixedSurface,
    required_surface_ids: tuple[str, ...],
) -> dict[str, object]:
    return {
        "surface_id": surface.surface_id,
        "level_scale": float(level.scale),
        "level_offset_world": (
            float(level.offset_x_meters),
            float(level.offset_y_meters),
        ),
        "level_pivot_world": tuple(
            float(value) for value in get_level_world_pivot(level)
        ),
        "required_surface_ids": required_surface_ids,
        "level_image_size_pixels": level.image_size_pixels,
    }


def _build_wall_targets(
    level: LevelData,
    surface: FixedSurface,
    common: dict[str, object],
) -> tuple[CanvasSurfaceEditHandleTarget, ...]:
    chain = _build_wall_vertex_chain(level, surface)
    if chain is None:
        return ()
    chain_vertex_ids, chain_ratios, chain_positions = chain
    base_z = _get_wall_base_z(surface)
    targets: list[CanvasSurfaceEditHandleTarget] = []
    for vertex_id, image_position in (
        (chain_vertex_ids[0], chain_positions[0]),
        (chain_vertex_ids[-1], chain_positions[-1]),
    ):
        world_x, world_y = level_image_to_world_xy(level, *image_position)
        for axis_index in (
            CANVAS_SURFACE_EDIT_AXIS_X,
            CANVAS_SURFACE_EDIT_AXIS_Y,
        ):
            targets.append(
                CanvasSurfaceEditHandleTarget(
                    reference=CanvasSurfaceEditReference(
                        CANVAS_SURFACE_EDIT_WALL_VERTEX,
                        level.index,
                        axis_index,
                        vertex_id=vertex_id,
                    ),
                    origin_world=(world_x, world_y, base_z),
                    axis_world=tuple(
                        float(value)
                        for value in np.eye(3, dtype=float)[axis_index]
                    ),
                    minimum_delta_meters=(
                        -MAX_CANVAS_WALL_EDIT_DELTA_METERS
                    ),
                    maximum_delta_meters=MAX_CANVAS_WALL_EDIT_DELTA_METERS,
                    chain_vertex_ids=chain_vertex_ids,
                    chain_vertex_ratios=chain_ratios,
                    chain_vertex_image_positions=chain_positions,
                    **common,
                )
            )
    midpoint_image = (
        (chain_positions[0][0] + chain_positions[-1][0]) * 0.5,
        (chain_positions[0][1] + chain_positions[-1][1]) * 0.5,
    )
    midpoint_world = level_image_to_world_xy(level, *midpoint_image)
    wall_vertex_ids = (chain_vertex_ids[0], chain_vertex_ids[-1])
    for axis_index in (
        CANVAS_SURFACE_EDIT_AXIS_X,
        CANVAS_SURFACE_EDIT_AXIS_Y,
    ):
        targets.append(
            CanvasSurfaceEditHandleTarget(
                reference=CanvasSurfaceEditReference(
                    CANVAS_SURFACE_EDIT_WALL_TRANSLATION,
                    level.index,
                    axis_index,
                    wall_vertex_ids=wall_vertex_ids,
                ),
                origin_world=(*midpoint_world, base_z),
                axis_world=tuple(
                    float(value)
                    for value in np.eye(3, dtype=float)[axis_index]
                ),
                minimum_delta_meters=-MAX_CANVAS_WALL_EDIT_DELTA_METERS,
                maximum_delta_meters=MAX_CANVAS_WALL_EDIT_DELTA_METERS,
                chain_vertex_ids=chain_vertex_ids,
                chain_vertex_ratios=chain_ratios,
                chain_vertex_image_positions=chain_positions,
                **common,
            )
        )
    return tuple(targets)


def _build_ceiling_target(
    level: LevelData,
    surface: FixedSurface,
    common: dict[str, object],
) -> CanvasSurfaceEditHandleTarget | None:
    origin = _get_surface_centroid(surface)
    if origin is None:
        return None
    if surface.room_index is None:
        kind = CANVAS_SURFACE_EDIT_LEVEL_HEIGHT
        room_center_vertex_id = None
        baseline_height = float(level.height_meters)
    else:
        room = _get_surface_room(level, surface)
        if room is None:
            return None
        kind = CANVAS_SURFACE_EDIT_ROOM_HEIGHT
        room_center_vertex_id = room.center_vertex_id
        baseline_height = float(room.height_meters)
    return CanvasSurfaceEditHandleTarget(
        reference=CanvasSurfaceEditReference(
            kind,
            level.index,
            CANVAS_SURFACE_EDIT_AXIS_Z,
            room_center_vertex_id=room_center_vertex_id,
        ),
        origin_world=origin,
        axis_world=(0.0, 0.0, 1.0),
        minimum_delta_meters=(
            MIN_CANVAS_SURFACE_HEIGHT_METERS - baseline_height
        ),
        maximum_delta_meters=(
            MAX_CANVAS_SURFACE_HEIGHT_METERS - baseline_height
        ),
        baseline_value_meters=baseline_height,
        **common,
    )


def _build_floor_target(
    level: LevelData,
    surface: FixedSurface,
    common: dict[str, object],
) -> CanvasSurfaceEditHandleTarget | None:
    origin = _get_surface_centroid(surface)
    if origin is None:
        return None
    baseline_thickness = float(level.floor_thickness_meters)
    return CanvasSurfaceEditHandleTarget(
        reference=CanvasSurfaceEditReference(
            CANVAS_SURFACE_EDIT_FLOOR_THICKNESS,
            level.index,
            CANVAS_SURFACE_EDIT_AXIS_Z,
        ),
        origin_world=origin,
        axis_world=(0.0, 0.0, 1.0),
        minimum_delta_meters=(
            MIN_FLOOR_THICKNESS_METERS - baseline_thickness
        ),
        maximum_delta_meters=(
            MAX_FLOOR_THICKNESS_METERS - baseline_thickness
        ),
        baseline_value_meters=baseline_thickness,
        **common,
    )


# ### Project edit application ###
def apply_canvas_surface_edit(
    levels: Sequence[LevelData],
    target: CanvasSurfaceEditHandleTarget,
    edit: CanvasSurfaceEdit,
    *,
    validate_project_geometry: bool = True,
) -> AppliedCanvasSurfaceEdit:
    """Rewrite authoritative state from a target's immutable drag baseline."""

    level_sequence = _normalize_levels(levels)
    if not isinstance(target, CanvasSurfaceEditHandleTarget):
        raise TypeError("Canvas surface edits require a handle target.")
    if not isinstance(edit, CanvasSurfaceEdit):
        raise TypeError("Canvas surface edits require an edit payload.")
    if edit.reference != target.reference:
        raise ValueError("The Canvas surface edit handle changed during the drag.")
    if edit.surface_id != target.surface_id:
        raise ValueError("The Canvas surface edit moved to another surface.")
    delta = _validate_target_delta(target, edit.delta_meters)
    level = _find_level(level_sequence, edit.reference.level_index)
    _validate_level_baseline(level, target)

    previous_delta = _measure_current_delta(level, target)
    previous = CanvasSurfaceEdit(
        target.reference,
        target.surface_id,
        previous_delta,
    )
    snapshot = _capture_level_mutation_snapshot(level)
    try:
        _apply_target_delta(level, target, delta)
        _validate_applied_values(level, target)
        if validate_project_geometry:
            _validate_generated_surfaces(level_sequence, target)
    except Exception as error:
        _restore_level_mutation_snapshot(level, snapshot)
        if isinstance(error, (TypeError, ValueError)):
            raise
        raise ValueError(
            "The Canvas surface edit would create invalid project geometry."
        ) from error

    current = CanvasSurfaceEdit(target.reference, target.surface_id, delta)
    return AppliedCanvasSurfaceEdit(
        reference=target.reference,
        level=level,
        previous=previous,
        current=current,
    )


def apply_canvas_wall_edit_batch(
    levels: Sequence[LevelData],
    targets: Sequence[CanvasSurfaceEditHandleTarget],
    edit: CanvasSurfaceEdit,
    *,
    validate_project_geometry: bool = True,
) -> tuple[AppliedCanvasSurfaceEdit, ...]:
    """Apply one absolute drag delta to a compatible wall-target batch."""

    level_sequence = _normalize_levels(levels)
    if not isinstance(edit, CanvasSurfaceEdit):
        raise TypeError("Canvas wall batch edits require an edit payload.")
    target_sequence = _normalize_wall_edit_batch_targets(targets, edit)
    delta = _normalize_finite_number(edit.delta_meters, "edit delta")
    levels_by_index = {level.index: level for level in level_sequence}
    affected_levels: dict[int, LevelData] = {}
    previous_edits: list[CanvasSurfaceEdit] = []

    for target in target_sequence:
        target_delta = _validate_target_delta(target, delta)
        if not math.isclose(
            target_delta,
            delta,
            rel_tol=0.0,
            abs_tol=CANVAS_SURFACE_EDIT_EPSILON,
        ):
            raise ValueError(
                "The Canvas wall batch delta was clamped inconsistently."
            )
        level = levels_by_index.get(target.reference.level_index)
        if level is None:
            raise ValueError("An edited Canvas level no longer exists.")
        _validate_level_baseline(level, target)
        _validate_wall_target_vertices_exist(level, target)
        affected_levels[level.index] = level
        previous_edits.append(
            CanvasSurfaceEdit(
                target.reference,
                target.surface_id,
                _measure_current_delta(level, target),
            )
        )

    snapshots = {
        level_index: _capture_level_mutation_snapshot(level)
        for level_index, level in affected_levels.items()
    }
    try:
        positions_by_level = _collect_wall_batch_positions(
            target_sequence,
            delta,
        )
        _write_wall_batch_positions(affected_levels, positions_by_level)
        _reconcile_wall_batch_level_transforms(
            affected_levels,
            target_sequence,
        )
        _validate_wall_batch_lengths(affected_levels, target_sequence)
        if validate_project_geometry:
            _validate_generated_surfaces_batch(
                level_sequence,
                target_sequence,
            )
    except Exception as error:
        for level_index, level in affected_levels.items():
            _restore_level_mutation_snapshot(level, snapshots[level_index])
        if isinstance(error, (TypeError, ValueError)):
            raise
        raise ValueError(
            "The Canvas wall batch edit would create invalid project geometry."
        ) from error

    return tuple(
        AppliedCanvasSurfaceEdit(
            reference=target.reference,
            level=affected_levels[target.reference.level_index],
            previous=previous,
            current=CanvasSurfaceEdit(
                target.reference,
                target.surface_id,
                delta,
            ),
        )
        for target, previous in zip(
            target_sequence,
            previous_edits,
            strict=True,
        )
    )


def restore_canvas_surface_edit(
    levels: Sequence[LevelData],
    target: CanvasSurfaceEditHandleTarget,
    *,
    validate_project_geometry: bool = True,
) -> AppliedCanvasSurfaceEdit:
    """Restore exactly the immutable baseline captured by one handle target."""

    if not isinstance(target, CanvasSurfaceEditHandleTarget):
        raise TypeError("Canvas surface restoration requires a handle target.")
    return apply_canvas_surface_edit(
        levels,
        target,
        CanvasSurfaceEdit(target.reference, target.surface_id, 0.0),
        validate_project_geometry=validate_project_geometry,
    )


def restore_canvas_wall_edit_batch(
    levels: Sequence[LevelData],
    targets: Sequence[CanvasSurfaceEditHandleTarget],
    *,
    validate_project_geometry: bool = True,
) -> tuple[AppliedCanvasSurfaceEdit, ...]:
    """Atomically restore every wall target to its drag baseline."""

    target_sequence = _normalize_wall_target_sequence(
        targets,
        "Canvas wall batch restoration",
    )
    primary = target_sequence[0]
    return apply_canvas_wall_edit_batch(
        levels,
        target_sequence,
        CanvasSurfaceEdit(primary.reference, primary.surface_id, 0.0),
        validate_project_geometry=validate_project_geometry,
    )


def validate_canvas_surface_edit_geometry(
    levels: Sequence[LevelData],
    target: CanvasSurfaceEditHandleTarget,
) -> None:
    """Validate current generated geometry without replaying an edit."""

    level_sequence = _normalize_levels(levels)
    if not isinstance(target, CanvasSurfaceEditHandleTarget):
        raise TypeError("Canvas surface validation requires a handle target.")
    level = _find_level(level_sequence, target.reference.level_index)
    _validate_level_baseline(level, target)
    _validate_applied_values(level, target)
    _validate_generated_surfaces(level_sequence, target)


def canvas_surface_edit_targets_are_at_baseline(
    levels: Sequence[LevelData],
    targets: Sequence[CanvasSurfaceEditHandleTarget],
) -> bool:
    """Return whether every edited property still equals its saved baseline."""

    level_sequence = _normalize_levels(levels)
    try:
        target_sequence = tuple(targets)
    except TypeError as error:
        raise TypeError(
            "Canvas surface baseline comparison requires a target sequence."
        ) from error
    if not target_sequence or not all(
        isinstance(target, CanvasSurfaceEditHandleTarget)
        for target in target_sequence
    ):
        raise ValueError(
            "Canvas surface baseline comparison requires handle targets."
        )
    for target in target_sequence:
        level = _find_level(level_sequence, target.reference.level_index)
        _validate_level_baseline(level, target)
        if not math.isclose(
            _measure_current_delta(level, target),
            0.0,
            rel_tol=0.0,
            abs_tol=CANVAS_SURFACE_EDIT_EPSILON,
        ):
            return False
    return True


def validate_canvas_wall_edit_batch_geometry(
    levels: Sequence[LevelData],
    targets: Sequence[CanvasSurfaceEditHandleTarget],
) -> None:
    """Validate a current multi-wall preview at its commit boundary."""

    level_sequence = _normalize_levels(levels)
    target_sequence = _normalize_wall_edit_batch_targets(targets)
    affected_levels: dict[int, LevelData] = {}
    for target in target_sequence:
        level = _find_level(level_sequence, target.reference.level_index)
        _validate_level_baseline(level, target)
        _validate_wall_target_vertices_exist(level, target)
        affected_levels[level.index] = level
    _validate_wall_batch_lengths(affected_levels, target_sequence)
    _validate_generated_surfaces_batch(level_sequence, target_sequence)


# ### Wall batch helpers ###
def _normalize_wall_target_sequence(
    targets: Sequence[CanvasSurfaceEditHandleTarget],
    operation_name: str,
    *,
    allow_empty: bool = False,
) -> tuple[CanvasSurfaceEditHandleTarget, ...]:
    try:
        target_sequence = tuple(targets)
    except TypeError as error:
        raise TypeError(f"{operation_name} requires a target sequence.") from error
    if not target_sequence and not allow_empty:
        raise ValueError(f"{operation_name} requires at least one target.")
    if not all(
        isinstance(target, CanvasSurfaceEditHandleTarget)
        for target in target_sequence
    ):
        raise TypeError(f"{operation_name} requires wall handle targets.")
    if any(
        target.reference.kind not in CANVAS_SURFACE_EDIT_WALL_KINDS
        for target in target_sequence
    ):
        raise ValueError(f"{operation_name} only accepts wall handle targets.")
    return target_sequence


def _normalize_wall_edit_batch_targets(
    targets: Sequence[CanvasSurfaceEditHandleTarget],
    edit: CanvasSurfaceEdit | None = None,
) -> tuple[CanvasSurfaceEditHandleTarget, ...]:
    target_sequence = _normalize_wall_target_sequence(
        targets,
        "Canvas wall batch editing",
    )
    unique_targets: dict[
        tuple[str, str],
        CanvasSurfaceEditHandleTarget,
    ] = {}
    for target in target_sequence:
        key = (target.surface_id, target.reference.key)
        existing = unique_targets.get(key)
        if existing is not None and existing != target:
            raise ValueError(
                "Duplicate Canvas wall batch handles have conflicting baselines."
            )
        unique_targets.setdefault(key, target)
    normalized = tuple(unique_targets.values())

    primary_reference = normalized[0].reference
    if edit is not None:
        if edit.reference.kind not in CANVAS_SURFACE_EDIT_WALL_KINDS:
            raise ValueError("Canvas wall batch edits require a wall reference.")
        primary_matches = tuple(
            target
            for target in normalized
            if target.reference == edit.reference
            and target.surface_id == edit.surface_id
        )
        if len(primary_matches) != 1:
            raise ValueError(
                "The primary Canvas wall handle is not in the edit batch."
            )
        primary_reference = edit.reference

    if any(
        target.reference.axis_index != primary_reference.axis_index
        for target in normalized
    ):
        raise ValueError("Canvas wall batch handles must share one axis.")
    if primary_reference.kind == CANVAS_SURFACE_EDIT_WALL_TRANSLATION:
        if any(
            target.reference.kind != CANVAS_SURFACE_EDIT_WALL_TRANSLATION
            for target in normalized
        ):
            raise ValueError(
                "Canvas wall translation batches require translation handles."
            )
    elif any(
        target.reference != primary_reference
        for target in normalized
    ):
        raise ValueError(
            "Canvas endpoint batches require one shared persistent reference."
        )
    return normalized


def _collect_wall_batch_positions(
    targets: tuple[CanvasSurfaceEditHandleTarget, ...],
    delta: float,
) -> dict[int, dict[int, tuple[float, float]]]:
    positions_by_level: dict[int, dict[int, tuple[float, float]]] = {}
    for target in targets:
        level_positions = positions_by_level.setdefault(
            target.reference.level_index,
            {},
        )
        target_positions = _build_wall_target_positions(target, delta)
        for vertex_id, position in zip(
            target.chain_vertex_ids,
            target_positions,
            strict=True,
        ):
            existing = level_positions.get(vertex_id)
            if existing is not None and not _pairs_are_close(
                existing,
                position,
            ):
                raise ValueError(
                    "Selected Canvas walls require conflicting positions for "
                    f"vertex {vertex_id}."
                )
            level_positions.setdefault(vertex_id, position)
    return positions_by_level


def _write_wall_batch_positions(
    levels_by_index: dict[int, LevelData],
    positions_by_level: dict[int, dict[int, tuple[float, float]]],
) -> None:
    for level_index, vertex_positions in positions_by_level.items():
        level = levels_by_index[level_index]
        for vertex_id, position in vertex_positions.items():
            if level.vertex_data.move_vertex(vertex_id, *position) is None:
                raise ValueError("An edited Canvas wall vertex disappeared.")


def _reconcile_wall_batch_level_transforms(
    levels_by_index: dict[int, LevelData],
    targets: tuple[CanvasSurfaceEditHandleTarget, ...],
) -> None:
    targets_by_level: dict[int, list[CanvasSurfaceEditHandleTarget]] = {}
    for target in targets:
        targets_by_level.setdefault(
            target.reference.level_index,
            [],
        ).append(target)

    for level_index, level_targets in targets_by_level.items():
        baseline = level_targets[0]
        if any(
            not _wall_level_baselines_match(baseline, target)
            for target in level_targets[1:]
        ):
            raise ValueError(
                "Selected Canvas wall handles have inconsistent level baselines."
            )
        level = levels_by_index[level_index]
        level.offset_x_meters, level.offset_y_meters = (
            baseline.level_offset_world
        )
        next_pivot = np.asarray(get_level_world_pivot(level), dtype=float)
        baseline_pivot = np.asarray(
            baseline.level_pivot_world,
            dtype=float,
        )
        compensation = (
            (1.0 - baseline.level_scale) * (baseline_pivot - next_pivot)
        )
        level.offset_x_meters += float(compensation[0])
        level.offset_y_meters += float(compensation[1])
        _validate_level_offsets(level)


def _wall_level_baselines_match(
    first: CanvasSurfaceEditHandleTarget,
    second: CanvasSurfaceEditHandleTarget,
) -> bool:
    return bool(
        math.isclose(
            first.level_scale,
            second.level_scale,
            rel_tol=0.0,
            abs_tol=CANVAS_SURFACE_EDIT_EPSILON,
        )
        and _pairs_are_close(
            first.level_offset_world,
            second.level_offset_world,
        )
        and _pairs_are_close(
            first.level_pivot_world,
            second.level_pivot_world,
        )
        and first.level_image_size_pixels == second.level_image_size_pixels
    )


def _pairs_are_close(
    first: tuple[float, float],
    second: tuple[float, float],
) -> bool:
    return all(
        math.isclose(
            first_value,
            second_value,
            rel_tol=0.0,
            abs_tol=CANVAS_WALL_CHAIN_RELATIVE_TOLERANCE,
        )
        for first_value, second_value in zip(first, second, strict=True)
    )


# ### Single-target application helpers ###
def _apply_target_delta(
    level: LevelData,
    target: CanvasSurfaceEditHandleTarget,
    delta: float,
) -> None:
    kind = target.reference.kind
    if kind in CANVAS_SURFACE_EDIT_WALL_KINDS:
        _apply_wall_delta(level, target, delta)
        return
    new_value = target.baseline_value_meters + delta
    if kind == CANVAS_SURFACE_EDIT_LEVEL_HEIGHT:
        level.height_meters = new_value
        return
    if kind == CANVAS_SURFACE_EDIT_FLOOR_THICKNESS:
        level.floor_thickness_meters = new_value
        return
    assert kind == CANVAS_SURFACE_EDIT_ROOM_HEIGHT
    room = _find_room_by_center_vertex_id(
        level,
        target.reference.room_center_vertex_id,
    )
    if room is None:
        raise ValueError("The edited Canvas room no longer exists.")
    room.height_meters = new_value


def _rebase_canvas_wall_edit_target(
    level: LevelData,
    target: CanvasSurfaceEditHandleTarget,
) -> CanvasSurfaceEditHandleTarget:
    """Copy one wall handle with the current chain as its new baseline."""

    _validate_level_baseline(level, target)
    vertex_lookup = {
        vertex.id: vertex for vertex in level.vertex_data.vertices
    }
    if any(
        vertex_id not in vertex_lookup
        for vertex_id in target.chain_vertex_ids
    ):
        raise ValueError("The edited Canvas wall vertices no longer exist.")
    chain_positions = tuple(
        (
            float(vertex_lookup[vertex_id].x),
            float(vertex_lookup[vertex_id].y),
        )
        for vertex_id in target.chain_vertex_ids
    )
    if target.reference.kind == CANVAS_SURFACE_EDIT_WALL_TRANSLATION:
        reference_image_position = (
            (chain_positions[0][0] + chain_positions[-1][0]) * 0.5,
            (chain_positions[0][1] + chain_positions[-1][1]) * 0.5,
        )
    else:
        reference_vertex_id = target.reference.vertex_id
        assert reference_vertex_id is not None
        reference_vertex = vertex_lookup.get(reference_vertex_id)
        if reference_vertex is None:
            raise ValueError("The edited Canvas wall vertex no longer exists.")
        reference_image_position = (
            float(reference_vertex.x),
            float(reference_vertex.y),
        )
    world_x, world_y = level_image_to_world_xy(
        level,
        *reference_image_position,
    )
    return replace(
        target,
        origin_world=(world_x, world_y, target.origin_world[2]),
        level_offset_world=(
            float(level.offset_x_meters),
            float(level.offset_y_meters),
        ),
        level_pivot_world=tuple(
            float(value) for value in get_level_world_pivot(level)
        ),
        chain_vertex_image_positions=chain_positions,
    )


def _apply_wall_delta(
    level: LevelData,
    target: CanvasSurfaceEditHandleTarget,
    delta: float,
) -> None:
    _validate_wall_target_vertices_exist(level, target)
    new_positions = _build_wall_target_positions(target, delta)
    for vertex_id, position in zip(
        target.chain_vertex_ids,
        new_positions,
        strict=True,
    ):
        if level.vertex_data.move_vertex(vertex_id, *position) is None:
            raise ValueError("The edited Canvas wall vertex disappeared.")

    level.offset_x_meters, level.offset_y_meters = target.level_offset_world
    next_pivot = np.asarray(get_level_world_pivot(level), dtype=float)
    baseline_pivot = np.asarray(target.level_pivot_world, dtype=float)
    compensation = (
        (1.0 - target.level_scale) * (baseline_pivot - next_pivot)
    )
    level.offset_x_meters += float(compensation[0])
    level.offset_y_meters += float(compensation[1])
    _validate_level_offsets(level)

    _validate_wall_target_length(level, target)


def _build_wall_target_positions(
    target: CanvasSurfaceEditHandleTarget,
    delta: float,
) -> tuple[tuple[float, float], ...]:
    reference = target.reference
    if abs(delta) <= CANVAS_SURFACE_EDIT_EPSILON:
        return target.chain_vertex_image_positions

    image_delta = np.zeros(2, dtype=float)
    image_distance = delta / (PIXEL_TO_METER * target.level_scale)
    if reference.axis_index == CANVAS_SURFACE_EDIT_AXIS_X:
        image_delta[0] = image_distance
    else:
        image_delta[1] = -image_distance
    if reference.kind == CANVAS_SURFACE_EDIT_WALL_TRANSLATION:
        return tuple(
            tuple(
                float(value)
                for value in np.asarray(position, dtype=float) + image_delta
            )
            for position in target.chain_vertex_image_positions
        )

    start = np.asarray(target.chain_vertex_image_positions[0], dtype=float)
    end = np.asarray(target.chain_vertex_image_positions[-1], dtype=float)
    if reference.vertex_id == target.chain_vertex_ids[0]:
        start = start + image_delta
    elif reference.vertex_id == target.chain_vertex_ids[-1]:
        end = end + image_delta
    else:
        raise ValueError("A Canvas wall handle must own one chain endpoint.")
    return tuple(
        tuple(float(value) for value in start + (end - start) * ratio)
        for ratio in target.chain_vertex_ratios
    )


def _validate_wall_target_vertices_exist(
    level: LevelData,
    target: CanvasSurfaceEditHandleTarget,
) -> None:
    vertex_ids = {vertex.id for vertex in level.vertex_data.vertices}
    if any(
        vertex_id not in vertex_ids
        for vertex_id in target.chain_vertex_ids
    ):
        raise ValueError("The edited Canvas wall vertices no longer exist.")


def _validate_wall_batch_lengths(
    levels_by_index: dict[int, LevelData],
    targets: tuple[CanvasSurfaceEditHandleTarget, ...],
) -> None:
    validated_walls: set[tuple[int, frozenset[int]]] = set()
    for target in targets:
        wall_key = (
            target.reference.level_index,
            frozenset(
                (
                    target.chain_vertex_ids[0],
                    target.chain_vertex_ids[-1],
                )
            ),
        )
        if wall_key in validated_walls:
            continue
        validated_walls.add(wall_key)
        _validate_wall_target_length(
            levels_by_index[target.reference.level_index],
            target,
        )


def _validate_wall_target_length(
    level: LevelData,
    target: CanvasSurfaceEditHandleTarget,
) -> None:
    first = level.vertex_data.get_vertex(target.chain_vertex_ids[0])
    last = level.vertex_data.get_vertex(target.chain_vertex_ids[-1])
    if first is None or last is None:
        raise ValueError("The edited Canvas wall vertices no longer exist.")
    first_world = level_image_to_world_xy(level, first.x, first.y)
    last_world = level_image_to_world_xy(level, last.x, last.y)
    wall_length = math.dist(first_world, last_world)
    if wall_length < MIN_CANVAS_WALL_LENGTH_METERS:
        raise ValueError(
            "Canvas wall endpoints must remain at least "
            f"{MIN_CANVAS_WALL_LENGTH_METERS:g} meters apart."
        )


# ### Authoritative validation helpers ###
def _validate_applied_values(
    level: LevelData,
    target: CanvasSurfaceEditHandleTarget,
) -> None:
    kind = target.reference.kind
    if kind in {
        CANVAS_SURFACE_EDIT_LEVEL_HEIGHT,
        CANVAS_SURFACE_EDIT_ROOM_HEIGHT,
        CANVAS_SURFACE_EDIT_FLOOR_THICKNESS,
    }:
        if kind == CANVAS_SURFACE_EDIT_LEVEL_HEIGHT:
            value = level.height_meters
        elif kind == CANVAS_SURFACE_EDIT_ROOM_HEIGHT:
            value = _get_required_room_height(level, target.reference)
        else:
            value = level.floor_thickness_meters
        minimum_value = (
            MIN_FLOOR_THICKNESS_METERS
            if kind == CANVAS_SURFACE_EDIT_FLOOR_THICKNESS
            else MIN_CANVAS_SURFACE_HEIGHT_METERS
        )
        maximum_value = (
            MAX_FLOOR_THICKNESS_METERS
            if kind == CANVAS_SURFACE_EDIT_FLOOR_THICKNESS
            else MAX_CANVAS_SURFACE_HEIGHT_METERS
        )
        _validate_range(
            value,
            minimum_value,
            maximum_value,
            (
                "Floor thickness"
                if kind == CANVAS_SURFACE_EDIT_FLOOR_THICKNESS
                else "Canvas surface height"
            ),
        )


def _validate_generated_surfaces(
    levels: tuple[LevelData, ...],
    target: CanvasSurfaceEditHandleTarget,
) -> None:
    """Run the expensive topology check only at a requested commit boundary."""

    _validate_generated_surfaces_batch(levels, (target,))


def _validate_generated_surfaces_batch(
    levels: tuple[LevelData, ...],
    targets: tuple[CanvasSurfaceEditHandleTarget, ...],
) -> None:
    """Build once and validate all surfaces required by a wall edit batch."""

    try:
        rebuilt_surfaces = tuple(build_fixed_surfaces(levels))
    except Exception as error:
        raise ValueError(
            "The Canvas surface edit would create invalid generated surfaces."
        ) from error
    rebuilt_ids = {surface.surface_id for surface in rebuilt_surfaces}
    required_ids = {
        surface_id
        for target in targets
        for surface_id in (
            target.surface_id,
            *target.required_surface_ids,
        )
    }
    missing_ids = sorted(
        surface_id for surface_id in required_ids if surface_id not in rebuilt_ids
    )
    if missing_ids:
        raise ValueError(
            "The Canvas surface edit would remove required generated surfaces."
        )


def _validate_level_baseline(
    level: LevelData,
    target: CanvasSurfaceEditHandleTarget,
) -> None:
    if not math.isclose(
        float(level.scale),
        target.level_scale,
        rel_tol=0.0,
        abs_tol=CANVAS_SURFACE_EDIT_EPSILON,
    ):
        raise ValueError("The Canvas level scale changed during the edit.")
    current_image_size = _normalize_optional_positive_pair(
        level.image_size_pixels,
        "level image size",
    )
    if current_image_size != target.level_image_size_pixels:
        raise ValueError("The Canvas level image size changed during the edit.")


def _validate_target_delta(
    target: CanvasSurfaceEditHandleTarget,
    value: object,
) -> float:
    delta = _normalize_finite_number(value, "edit delta")
    if (
        delta < target.minimum_delta_meters - CANVAS_SURFACE_EDIT_EPSILON
        or delta > target.maximum_delta_meters + CANVAS_SURFACE_EDIT_EPSILON
    ):
        raise ValueError("The Canvas surface edit exceeds its allowed range.")
    return min(
        max(delta, target.minimum_delta_meters),
        target.maximum_delta_meters,
    )


def _validate_level_offsets(level: LevelData) -> None:
    _validate_range(
        level.offset_x_meters,
        MIN_LEVEL_OFFSET_METERS,
        MAX_LEVEL_OFFSET_METERS,
        "Canvas level X offset",
    )
    _validate_range(
        level.offset_y_meters,
        MIN_LEVEL_OFFSET_METERS,
        MAX_LEVEL_OFFSET_METERS,
        "Canvas level Y offset",
    )


def _validate_range(
    value: object,
    minimum: float,
    maximum: float,
    field_name: str,
) -> float:
    normalized = _normalize_finite_number(value, field_name)
    if not minimum <= normalized <= maximum:
        raise ValueError(
            f"{field_name} must be between {minimum:g} and {maximum:g}."
        )
    return normalized


# ### Wall-chain helpers ###
def _build_wall_vertex_chain(
    level: LevelData,
    surface: FixedSurface,
) -> tuple[
    tuple[int, ...],
    tuple[float, ...],
    tuple[tuple[float, float], ...],
] | None:
    endpoint_ids = _parse_wall_endpoint_ids(surface.wall_key)
    if endpoint_ids is None:
        return None
    vertex_lookup = {
        vertex.id: vertex for vertex in level.vertex_data.vertices
    }
    first = vertex_lookup.get(endpoint_ids[0])
    second = vertex_lookup.get(endpoint_ids[1])
    if first is None or second is None:
        return None
    first_point = np.asarray((first.x, first.y), dtype=float)
    second_point = np.asarray((second.x, second.y), dtype=float)
    wall_vector = second_point - first_point
    length_squared = float(np.dot(wall_vector, wall_vector))
    if length_squared <= CANVAS_SURFACE_EDIT_EPSILON:
        return None
    length = math.sqrt(length_squared)
    tolerance = max(
        CANVAS_SURFACE_EDIT_EPSILON,
        length * CANVAS_WALL_CHAIN_RELATIVE_TOLERANCE,
    )

    allowed_ids = _get_surface_wall_vertex_ids(level, surface)
    candidate_ratios: dict[int, float] = {}
    for vertex_id in allowed_ids:
        vertex = vertex_lookup.get(vertex_id)
        if vertex is None:
            continue
        point = np.asarray((vertex.x, vertex.y), dtype=float)
        offset = point - first_point
        ratio = float(np.dot(offset, wall_vector) / length_squared)
        projected = first_point + wall_vector * ratio
        if float(np.linalg.norm(point - projected)) > tolerance:
            continue
        ratio_tolerance = tolerance / length
        if not -ratio_tolerance <= ratio <= 1.0 + ratio_tolerance:
            continue
        candidate_ratios[vertex_id] = min(max(ratio, 0.0), 1.0)
    if not all(vertex_id in candidate_ratios for vertex_id in endpoint_ids):
        return None

    edge_keys = {
        frozenset((edge.start_vertex_id, edge.end_vertex_id))
        for edge in level.vertex_data.edges
    }
    ordered_candidates = tuple(
        sorted(
            candidate_ratios,
            key=lambda vertex_id: (candidate_ratios[vertex_id], vertex_id),
        )
    )
    if _is_connected_ordered_chain(ordered_candidates, edge_keys):
        chain_ids = ordered_candidates
    else:
        chain_ids = _find_monotonic_wall_path(
            endpoint_ids,
            candidate_ratios,
            edge_keys,
        )
        if chain_ids is None and surface.room_index is not None:
            # Rooms without explicit edges use their angularly ordered boundary.
            chain_ids = ordered_candidates
    if chain_ids is None or len(chain_ids) < 2:
        return None

    ratios = tuple(candidate_ratios[vertex_id] for vertex_id in chain_ids)
    positions = tuple(
        (float(vertex_lookup[vertex_id].x), float(vertex_lookup[vertex_id].y))
        for vertex_id in chain_ids
    )
    if not _surface_matches_chain_endpoints(level, surface, chain_ids):
        return None
    return chain_ids, ratios, positions


def _get_surface_wall_vertex_ids(
    level: LevelData,
    surface: FixedSurface,
) -> set[int]:
    if surface.room_index is None:
        return {vertex.id for vertex in level.vertex_data.vertices}
    room = _get_surface_room(level, surface)
    if room is None:
        return set()
    return set(room.vertex_ids)


def _find_monotonic_wall_path(
    endpoint_ids: tuple[int, int],
    ratios: dict[int, float],
    edge_keys: set[frozenset[int]],
) -> tuple[int, ...] | None:
    first_id, last_id = endpoint_ids
    adjacency: dict[int, list[int]] = {vertex_id: [] for vertex_id in ratios}
    for edge_key in edge_keys:
        if len(edge_key) != 2:
            continue
        first, second = tuple(edge_key)
        if first not in adjacency or second not in adjacency:
            continue
        adjacency[first].append(second)
        adjacency[second].append(first)
    queue: deque[tuple[int, tuple[int, ...]]] = deque(((first_id, (first_id,)),))
    while queue:
        current_id, path = queue.popleft()
        if current_id == last_id:
            return path
        current_ratio = ratios[current_id]
        for next_id in sorted(
            adjacency[current_id],
            key=lambda vertex_id: (ratios[vertex_id], vertex_id),
        ):
            if next_id in path:
                continue
            if ratios[next_id] <= current_ratio + CANVAS_SURFACE_EDIT_EPSILON:
                continue
            queue.append((next_id, (*path, next_id)))
    return None


def _is_connected_ordered_chain(
    vertex_ids: tuple[int, ...],
    edge_keys: set[frozenset[int]],
) -> bool:
    return len(vertex_ids) >= 2 and all(
        frozenset((first, second)) in edge_keys
        for first, second in zip(vertex_ids, vertex_ids[1:])
    )


def _surface_matches_chain_endpoints(
    level: LevelData,
    surface: FixedSurface,
    chain_vertex_ids: tuple[int, ...],
) -> bool:
    if surface.wall_start_world is None or surface.wall_end_world is None:
        return False
    vertices = (
        level.vertex_data.get_vertex(chain_vertex_ids[0]),
        level.vertex_data.get_vertex(chain_vertex_ids[-1]),
    )
    if any(vertex is None for vertex in vertices):
        return False
    endpoint_world = tuple(
        np.asarray(level_image_to_world_xy(level, vertex.x, vertex.y), dtype=float)
        for vertex in vertices
        if vertex is not None
    )
    surface_world = (
        np.asarray(surface.wall_start_world[:2], dtype=float),
        np.asarray(surface.wall_end_world[:2], dtype=float),
    )
    tolerance = max(
        MIN_CANVAS_WALL_LENGTH_METERS * 0.01,
        CANVAS_SURFACE_EDIT_EPSILON,
    )
    direct = all(
        np.linalg.norm(first - second) <= tolerance
        for first, second in zip(endpoint_world, surface_world)
    )
    reverse = all(
        np.linalg.norm(first - second) <= tolerance
        for first, second in zip(endpoint_world, reversed(surface_world))
    )
    return direct or reverse


def _parse_wall_endpoint_ids(wall_key: object) -> tuple[int, int] | None:
    match = CANVAS_WALL_KEY_PATTERN.fullmatch(str(wall_key or "").strip())
    if match is None:
        return None
    first = int(match.group("first"))
    second = int(match.group("second"))
    if first == second:
        return None
    return first, second


# ### State snapshot helpers ###
def _capture_level_mutation_snapshot(level: LevelData) -> _LevelMutationSnapshot:
    return _LevelMutationSnapshot(
        vertices=tuple(level.vertex_data.vertices),
        offset_x_meters=float(level.offset_x_meters),
        offset_y_meters=float(level.offset_y_meters),
        height_meters=float(level.height_meters),
        floor_thickness_meters=float(level.floor_thickness_meters),
        room_heights=tuple(
            (room.center_vertex_id, float(room.height_meters))
            for room in level.rooms
        ),
    )


def _restore_level_mutation_snapshot(
    level: LevelData,
    snapshot: _LevelMutationSnapshot,
) -> None:
    level.vertex_data.vertices = list(snapshot.vertices)
    level.offset_x_meters = snapshot.offset_x_meters
    level.offset_y_meters = snapshot.offset_y_meters
    level.height_meters = snapshot.height_meters
    level.floor_thickness_meters = snapshot.floor_thickness_meters
    room_height_by_id = dict(snapshot.room_heights)
    for room in level.rooms:
        if room.center_vertex_id in room_height_by_id:
            room.height_meters = room_height_by_id[room.center_vertex_id]


def _measure_current_delta(
    level: LevelData,
    target: CanvasSurfaceEditHandleTarget,
) -> float:
    reference = target.reference
    if reference.kind in CANVAS_SURFACE_EDIT_WALL_KINDS:
        if reference.kind == CANVAS_SURFACE_EDIT_WALL_TRANSLATION:
            endpoints = tuple(
                level.vertex_data.get_vertex(vertex_id)
                for vertex_id in (
                    target.chain_vertex_ids[0],
                    target.chain_vertex_ids[-1],
                )
            )
            if any(vertex is None for vertex in endpoints):
                raise ValueError("The edited Canvas wall vertices no longer exist.")
            first, second = endpoints
            assert first is not None and second is not None
            image_position = (
                (first.x + second.x) * 0.5,
                (first.y + second.y) * 0.5,
            )
        else:
            assert reference.vertex_id is not None
            vertex = level.vertex_data.get_vertex(reference.vertex_id)
            if vertex is None:
                raise ValueError("The edited Canvas wall vertex no longer exists.")
            image_position = (vertex.x, vertex.y)
        world_xy = level_image_to_world_xy(level, *image_position)
        current = np.asarray(
            (world_xy[0], world_xy[1], target.origin_world[2]),
            dtype=float,
        )
        return float(
            np.dot(
                current - np.asarray(target.origin_world, dtype=float),
                np.asarray(target.axis_world, dtype=float),
            )
        )
    if reference.kind == CANVAS_SURFACE_EDIT_LEVEL_HEIGHT:
        value = float(level.height_meters)
    elif reference.kind == CANVAS_SURFACE_EDIT_FLOOR_THICKNESS:
        value = float(level.floor_thickness_meters)
    else:
        assert reference.kind == CANVAS_SURFACE_EDIT_ROOM_HEIGHT
        value = _get_required_room_height(level, reference)
    return value - target.baseline_value_meters


# ### Lookup and geometry helpers ###
def _normalize_levels(levels: Sequence[LevelData]) -> tuple[LevelData, ...]:
    try:
        normalized = tuple(levels)
    except TypeError as error:
        raise TypeError("Canvas surface edits require a level sequence.") from error
    if not all(isinstance(level, LevelData) for level in normalized):
        raise TypeError("Canvas surface edit levels must contain LevelData values.")
    indices = tuple(level.index for level in normalized)
    if len(indices) != len(set(indices)):
        raise ValueError("Canvas surface edit level indices must be unique.")
    return normalized


def _normalize_surfaces(
    surfaces: Sequence[FixedSurface],
) -> tuple[FixedSurface, ...]:
    try:
        normalized = tuple(surfaces)
    except TypeError as error:
        raise TypeError("Canvas surface edits require a surface sequence.") from error
    if not all(isinstance(surface, FixedSurface) for surface in normalized):
        raise TypeError(
            "Canvas surface edit targets must contain FixedSurface values."
        )
    ids = tuple(surface.surface_id for surface in normalized)
    if len(ids) != len(set(ids)):
        raise ValueError("Canvas surface edit IDs must be unique.")
    return normalized


def _find_level(
    levels: Sequence[LevelData],
    level_index: int,
) -> LevelData:
    level = next(
        (candidate for candidate in levels if candidate.index == level_index),
        None,
    )
    if level is None:
        raise ValueError("The edited Canvas level no longer exists.")
    return level


def _get_surface_room(
    level: LevelData,
    surface: FixedSurface,
) -> RoomData | None:
    room_index = surface.room_index
    if room_index is None or not 0 <= room_index < len(level.rooms):
        return None
    return level.rooms[room_index]


def _find_room_by_center_vertex_id(
    level: LevelData,
    room_center_vertex_id: int | None,
) -> RoomData | None:
    return next(
        (
            room
            for room in level.rooms
            if room.center_vertex_id == room_center_vertex_id
        ),
        None,
    )


def _get_required_room_height(
    level: LevelData,
    reference: CanvasSurfaceEditReference,
) -> float:
    room = _find_room_by_center_vertex_id(
        level,
        reference.room_center_vertex_id,
    )
    if room is None:
        raise ValueError("The edited Canvas room no longer exists.")
    return float(room.height_meters)


def _get_surface_centroid(
    surface: FixedSurface,
) -> tuple[float, float, float] | None:
    try:
        centroid = np.asarray(surface.mesh.centroid, dtype=float)
    except (AttributeError, TypeError, ValueError):
        return None
    if centroid.shape != (3,) or not np.all(np.isfinite(centroid)):
        return None
    return tuple(float(value) for value in centroid)


def _get_wall_base_z(surface: FixedSurface) -> float:
    if surface.wall_start_world is None:
        raise ValueError("A Canvas wall target requires its world frame.")
    return _normalize_finite_number(surface.wall_start_world[2], "wall base Z")


# ### Model validation helpers ###
def _validate_target_chain(
    reference: CanvasSurfaceEditReference,
    vertex_ids: tuple[int, ...],
    ratios: tuple[float, ...],
    positions: tuple[tuple[float, float], ...],
) -> None:
    is_wall = reference.kind in CANVAS_SURFACE_EDIT_WALL_KINDS
    if not is_wall:
        if vertex_ids or ratios or positions:
            raise ValueError("Only Canvas wall targets may carry a vertex chain.")
        return
    if len(vertex_ids) < 2:
        raise ValueError("Canvas wall targets require at least two chain vertices.")
    if len(set(vertex_ids)) != len(vertex_ids):
        raise ValueError("Canvas wall chain vertex IDs must be unique.")
    if not (len(vertex_ids) == len(ratios) == len(positions)):
        raise ValueError("Canvas wall chain baseline arrays must have equal sizes.")
    endpoint_ids = (vertex_ids[0], vertex_ids[-1])
    if reference.kind == CANVAS_SURFACE_EDIT_WALL_TRANSLATION:
        if set(reference.wall_vertex_ids) != set(endpoint_ids):
            raise ValueError(
                "Canvas wall translation handles must reference both chain endpoints."
            )
    elif reference.vertex_id not in set(endpoint_ids):
        raise ValueError("Canvas wall handles must reference a chain endpoint.")
    if not math.isclose(ratios[0], 0.0, abs_tol=1e-7):
        raise ValueError("Canvas wall chain ratios must start at zero.")
    if not math.isclose(ratios[-1], 1.0, abs_tol=1e-7):
        raise ValueError("Canvas wall chain ratios must end at one.")
    if any(
        next_ratio <= ratio + CANVAS_SURFACE_EDIT_EPSILON
        for ratio, next_ratio in zip(ratios, ratios[1:])
    ):
        raise ValueError("Canvas wall chain ratios must increase strictly.")


def _normalize_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Canvas surface {field_name} must be an integer.")
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"Canvas surface {field_name} must be an integer."
        ) from error
    if normalized != value:
        raise ValueError(f"Canvas surface {field_name} must be an integer.")
    return normalized


def _normalize_positive_integer(value: object, field_name: str) -> int:
    normalized = _normalize_integer(value, field_name)
    if normalized <= 0:
        raise ValueError(f"Canvas surface {field_name} must be positive.")
    return normalized


def _normalize_optional_positive_integer(
    value: object,
    field_name: str,
) -> int | None:
    if value is None:
        return None
    return _normalize_positive_integer(value, field_name)


def _normalize_finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Canvas surface {field_name} must be a number.")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"Canvas surface {field_name} must be a number."
        ) from error
    if not math.isfinite(normalized):
        raise ValueError(f"Canvas surface {field_name} must be finite.")
    return normalized


def _normalize_positive_number(value: object, field_name: str) -> float:
    normalized = _normalize_finite_number(value, field_name)
    if normalized <= 0.0:
        raise ValueError(f"Canvas surface {field_name} must be positive.")
    return normalized


def _normalize_vector(
    value: object,
    field_name: str,
    *,
    unit: bool,
) -> tuple[float, float, float]:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"Canvas surface {field_name} must contain numeric XYZ values."
        ) from error
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(
            f"Canvas surface {field_name} must contain finite XYZ values."
        )
    if unit:
        length = float(np.linalg.norm(vector))
        if length <= CANVAS_SURFACE_EDIT_EPSILON:
            raise ValueError("A Canvas surface handle axis cannot be zero.")
        vector = vector / length
    return tuple(float(component) for component in vector)


def _normalize_pair(value: object, field_name: str) -> tuple[float, float]:
    try:
        pair = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(
            f"Canvas surface {field_name} must contain two numbers."
        ) from error
    if len(pair) != 2:
        raise ValueError(f"Canvas surface {field_name} must contain two numbers.")
    return tuple(
        _normalize_finite_number(component, field_name) for component in pair
    )  # type: ignore[return-value]


def _normalize_optional_positive_pair(
    value: object,
    field_name: str,
) -> tuple[float, float] | None:
    if value is None:
        return None
    pair = _normalize_pair(value, field_name)
    if pair[0] <= 0.0 or pair[1] <= 0.0:
        raise ValueError(f"Canvas surface {field_name} must be positive.")
    return pair
