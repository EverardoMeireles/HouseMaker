# ### Imports ###
from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from housemaker.level_coordinates import (
    level_image_to_world_xy,
    level_world_to_image_xy,
)
from housemaker.models import (
    DOORWAY_SHAPE_ARCH,
    MIN_DOORWAY_HEIGHT_METERS,
    MIN_DOORWAY_WIDTH_METERS,
    DoorwayData,
    LevelData,
    WindowData,
    normalize_doorway_arch_amount,
)
from housemaker.surface_geometry import (
    MIN_WINDOW_SIZE_METERS,
    SURFACE_TYPE_WALL,
    FixedSurface,
)

# ### Constants ###
CANVAS_OPENING_DOORWAY = "doorway"
CANVAS_OPENING_WINDOW = "window"
CANVAS_OPENING_KINDS = frozenset(
    (CANVAS_OPENING_DOORWAY, CANVAS_OPENING_WINDOW)
)
OPENING_FRAME_EPSILON = 1e-9
DOORWAY_WALL_ALIGNMENT_MINIMUM = 0.75
SLOPED_CEILING_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


# ### Opening edit models ###
@dataclass(frozen=True)
class CanvasOpeningReference:
    """Stable-enough project identity for one Canvas opening edit session."""

    kind: str
    level_index: int
    item_index: int
    stable_id: str | None = None

    def __post_init__(self) -> None:
        normalized_kind = str(self.kind).strip().lower()
        if normalized_kind not in CANVAS_OPENING_KINDS:
            raise ValueError("Canvas openings must be doorways or windows.")
        normalized_level_index = int(self.level_index)
        normalized_item_index = int(self.item_index)
        if normalized_item_index < 0:
            raise ValueError("Canvas opening indices cannot be negative.")
        normalized_stable_id = (
            None if self.stable_id is None else str(self.stable_id).strip()
        )
        if normalized_kind == CANVAS_OPENING_WINDOW and not normalized_stable_id:
            raise ValueError("Canvas window edits require a stable window ID.")
        object.__setattr__(self, "kind", normalized_kind)
        object.__setattr__(self, "level_index", normalized_level_index)
        object.__setattr__(self, "item_index", normalized_item_index)
        object.__setattr__(self, "stable_id", normalized_stable_id)

    @property
    def key(self) -> str:
        if self.kind == CANVAS_OPENING_WINDOW:
            return f"window:{self.stable_id}"
        return f"doorway:{self.level_index}:{self.item_index}"


@dataclass(frozen=True)
class CanvasOpeningBounds:
    """One rectangular opening in normalized wall-local coordinates."""

    start_ratio: float
    end_ratio: float
    bottom_ratio: float
    top_ratio: float

    def __post_init__(self) -> None:
        values = tuple(
            _normalize_finite_number(value, field_name)
            for value, field_name in (
                (self.start_ratio, "start ratio"),
                (self.end_ratio, "end ratio"),
                (self.bottom_ratio, "bottom ratio"),
                (self.top_ratio, "top ratio"),
            )
        )
        start_ratio, end_ratio, bottom_ratio, top_ratio = values
        if end_ratio - start_ratio <= OPENING_FRAME_EPSILON:
            raise ValueError("Canvas opening width must be positive.")
        if top_ratio - bottom_ratio <= OPENING_FRAME_EPSILON:
            raise ValueError("Canvas opening height must be positive.")
        object.__setattr__(self, "start_ratio", start_ratio)
        object.__setattr__(self, "end_ratio", end_ratio)
        object.__setattr__(self, "bottom_ratio", bottom_ratio)
        object.__setattr__(self, "top_ratio", top_ratio)

    @property
    def horizontal_span(self) -> float:
        return self.end_ratio - self.start_ratio

    @property
    def vertical_span(self) -> float:
        return self.top_ratio - self.bottom_ratio


@dataclass(frozen=True)
class CanvasOpeningTarget:
    """World-space planar frame and current bounds for one editable opening."""

    reference: CanvasOpeningReference
    wall_surface_id: str
    plane_start_world: tuple[float, float, float]
    wall_tangent_world: tuple[float, float, float]
    wall_normal_world: tuple[float, float, float]
    wall_width_meters: float
    wall_height_meters: float
    minimum_width_meters: float
    minimum_height_meters: float
    bounds: CanvasOpeningBounds
    arch_amount: float | None = None
    vertical_axis_world: tuple[float, float, float] = (0.0, 0.0, 1.0)
    sloped_ceiling_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reference, CanvasOpeningReference):
            raise TypeError("Canvas opening targets require a reference.")
        normalized_surface_id = str(self.wall_surface_id).strip()
        if not normalized_surface_id:
            raise ValueError("Canvas opening targets require a wall surface ID.")
        plane_start = _normalize_world_vector(
            self.plane_start_world,
            "opening plane start",
        )
        tangent = _normalize_unit_vector(
            self.wall_tangent_world,
            "opening wall tangent",
        )
        normal = _normalize_unit_vector(
            self.wall_normal_world,
            "opening wall normal",
        )
        vertical_axis = _normalize_unit_vector(
            self.vertical_axis_world,
            "opening vertical axis",
        )
        if (
            abs(float(np.dot(tangent, normal))) > 1e-6
            or abs(float(np.dot(tangent, vertical_axis))) > 1e-6
            or abs(float(np.dot(normal, vertical_axis))) > 1e-6
        ):
            raise ValueError("Canvas opening frame axes must be perpendicular.")
        wall_width = _normalize_positive_number(
            self.wall_width_meters,
            "opening wall width",
        )
        wall_height = _normalize_positive_number(
            self.wall_height_meters,
            "opening wall height",
        )
        minimum_width = _normalize_positive_number(
            self.minimum_width_meters,
            "minimum opening width",
        )
        minimum_height = _normalize_positive_number(
            self.minimum_height_meters,
            "minimum opening height",
        )
        if not isinstance(self.bounds, CanvasOpeningBounds):
            raise TypeError("Canvas opening targets require wall-local bounds.")
        arch_amount = _normalize_optional_doorway_arch_amount(
            self.reference,
            self.arch_amount,
        )
        sloped_ceiling_id = _normalize_optional_sloped_ceiling_id(
            self.sloped_ceiling_id
        )
        if (
            sloped_ceiling_id is not None
            and self.reference.kind != CANVAS_OPENING_WINDOW
        ):
            raise ValueError("Only Canvas windows can target a sloped ceiling.")
        object.__setattr__(self, "wall_surface_id", normalized_surface_id)
        object.__setattr__(self, "plane_start_world", plane_start)
        object.__setattr__(self, "wall_tangent_world", tangent)
        object.__setattr__(self, "wall_normal_world", normal)
        object.__setattr__(self, "vertical_axis_world", vertical_axis)
        object.__setattr__(self, "wall_width_meters", wall_width)
        object.__setattr__(self, "wall_height_meters", wall_height)
        object.__setattr__(self, "minimum_width_meters", minimum_width)
        object.__setattr__(self, "minimum_height_meters", minimum_height)
        object.__setattr__(self, "arch_amount", arch_amount)
        object.__setattr__(self, "sloped_ceiling_id", sloped_ceiling_id)

    @property
    def key(self) -> str:
        return self.reference.key

    @property
    def minimum_horizontal_span(self) -> float:
        return min(self.minimum_width_meters / self.wall_width_meters, 1.0)

    @property
    def minimum_vertical_span(self) -> float:
        return min(self.minimum_height_meters / self.wall_height_meters, 1.0)

    def get_world_corners(self) -> tuple[tuple[float, float, float], ...]:
        """Return bottom-start, bottom-end, top-end, and top-start."""

        return tuple(
            self.local_to_world(horizontal_ratio, vertical_ratio)
            for horizontal_ratio, vertical_ratio in (
                (self.bounds.start_ratio, self.bounds.bottom_ratio),
                (self.bounds.end_ratio, self.bounds.bottom_ratio),
                (self.bounds.end_ratio, self.bounds.top_ratio),
                (self.bounds.start_ratio, self.bounds.top_ratio),
            )
        )

    def get_world_center(self) -> tuple[float, float, float]:
        return self.local_to_world(
            (self.bounds.start_ratio + self.bounds.end_ratio) * 0.5,
            (self.bounds.bottom_ratio + self.bounds.top_ratio) * 0.5,
        )

    def local_to_world(
        self,
        horizontal_ratio: float,
        vertical_ratio: float,
    ) -> tuple[float, float, float]:
        origin = np.asarray(self.plane_start_world, dtype=float)
        tangent = np.asarray(self.wall_tangent_world, dtype=float)
        vertical_axis = np.asarray(self.vertical_axis_world, dtype=float)
        point = (
            origin
            + tangent * float(horizontal_ratio) * self.wall_width_meters
            + vertical_axis
            * float(vertical_ratio)
            * self.wall_height_meters
        )
        return tuple(float(value) for value in point)

    def world_to_local(self, point: object) -> tuple[float, float]:
        world_point = np.asarray(
            _normalize_world_vector(point, "opening wall point"),
            dtype=float,
        )
        offset = world_point - np.asarray(self.plane_start_world, dtype=float)
        horizontal_ratio = float(
            np.dot(offset, np.asarray(self.wall_tangent_world, dtype=float))
            / self.wall_width_meters
        )
        vertical_ratio = float(
            np.dot(offset, np.asarray(self.vertical_axis_world, dtype=float))
            / self.wall_height_meters
        )
        return horizontal_ratio, vertical_ratio

    def with_bounds(self, bounds: CanvasOpeningBounds) -> CanvasOpeningTarget:
        return replace(self, bounds=bounds)

    def with_arch_amount(
        self,
        arch_amount: float | None,
    ) -> CanvasOpeningTarget:
        """Return this stable wall target with a new optional arch control."""

        return replace(self, arch_amount=arch_amount)

    def with_edit(self, edit: CanvasOpeningEdit) -> CanvasOpeningTarget:
        """Return the live target state represented by one matching edit."""

        if edit.reference != self.reference:
            raise ValueError("The Canvas opening edit targets another opening.")
        if edit.destination_target is not None:
            if self.reference.kind != CANVAS_OPENING_WINDOW:
                raise ValueError("Only Canvas windows can move to another host.")
            return edit.destination_target
        if edit.wall_surface_id != self.wall_surface_id:
            raise ValueError("The Canvas opening edit targets another wall.")
        if (edit.arch_amount is None) != (self.arch_amount is None):
            raise ValueError("The Canvas doorway arch mode changed during the edit.")
        return replace(
            self,
            bounds=edit.bounds,
            arch_amount=edit.arch_amount,
        )


@dataclass(frozen=True)
class CanvasOpeningEdit:
    """One live wall-local opening shape emitted by the 3D scene viewer."""

    reference: CanvasOpeningReference
    wall_surface_id: str
    bounds: CanvasOpeningBounds
    arch_amount: float | None = None
    destination_target: CanvasOpeningTarget | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reference, CanvasOpeningReference):
            raise TypeError("Canvas opening edits require a reference.")
        normalized_surface_id = str(self.wall_surface_id).strip()
        if not normalized_surface_id:
            raise ValueError("Canvas opening edits require a wall surface ID.")
        if not isinstance(self.bounds, CanvasOpeningBounds):
            raise TypeError("Canvas opening edits require bounds.")
        arch_amount = _normalize_optional_doorway_arch_amount(
            self.reference,
            self.arch_amount,
        )
        destination_target = self.destination_target
        if destination_target is not None:
            if not isinstance(destination_target, CanvasOpeningTarget):
                raise TypeError(
                    "A Canvas opening destination must be an opening target."
                )
            if self.reference.kind != CANVAS_OPENING_WINDOW:
                raise ValueError("Only Canvas windows can move to another host.")
            if destination_target.reference != self.reference:
                raise ValueError(
                    "The Canvas opening destination targets another opening."
                )
            if destination_target.wall_surface_id != normalized_surface_id:
                raise ValueError(
                    "The Canvas opening destination targets another wall."
                )
            if destination_target.bounds != self.bounds:
                raise ValueError(
                    "The Canvas opening destination bounds do not match the edit."
                )
            if destination_target.arch_amount != arch_amount:
                raise ValueError(
                    "The Canvas opening destination arch does not match the edit."
                )
        object.__setattr__(self, "wall_surface_id", normalized_surface_id)
        object.__setattr__(self, "arch_amount", arch_amount)


@dataclass(frozen=True)
class AppliedCanvasOpeningEdit:
    """The project objects replaced by one accepted live opening edit."""

    reference: CanvasOpeningReference
    level: LevelData
    previous: DoorwayData | WindowData
    current: DoorwayData | WindowData


@dataclass(frozen=True)
class CanvasWindowDimensionSnapshot:
    """One window's immutable physical size before its host is edited."""

    level_index: int
    window: WindowData
    horizontal_center_ratio: float
    width_meters: float
    bottom_meters: float
    height_meters: float

    def __post_init__(self) -> None:
        level_index = int(self.level_index)
        if not isinstance(self.window, WindowData):
            raise TypeError("Window dimension snapshots require WindowData.")
        center = _normalize_finite_number(
            self.horizontal_center_ratio,
            "window center ratio",
        )
        width = _normalize_positive_number(self.width_meters, "window width")
        bottom = _normalize_finite_number(self.bottom_meters, "window bottom")
        height = _normalize_positive_number(self.height_meters, "window height")
        object.__setattr__(self, "level_index", level_index)
        object.__setattr__(self, "horizontal_center_ratio", center)
        object.__setattr__(self, "width_meters", width)
        object.__setattr__(self, "bottom_meters", bottom)
        object.__setattr__(self, "height_meters", height)


# ### Target builders ###
def build_canvas_opening_targets(
    levels: Sequence[LevelData],
    surfaces: Sequence[FixedSurface],
) -> tuple[CanvasOpeningTarget, ...]:
    """Build explicit hole-plane targets because openings have no render faces."""

    level_sequence = tuple(levels)
    all_surfaces = tuple(surfaces)
    surface_sequence = tuple(
        surface
        for surface in all_surfaces
        if isinstance(surface, FixedSurface)
        and surface.surface_type == SURFACE_TYPE_WALL
    )
    surface_by_id = {surface.surface_id: surface for surface in surface_sequence}
    surfaces_by_level: dict[int, list[FixedSurface]] = {}
    for surface in surface_sequence:
        surfaces_by_level.setdefault(surface.level_index, []).append(surface)

    targets: list[CanvasOpeningTarget] = []
    for level in level_sequence:
        if not isinstance(level, LevelData):
            continue
        for doorway_index, doorway in enumerate(level.doorways):
            target = _build_doorway_target(
                level,
                doorway,
                doorway_index,
                surfaces_by_level.get(level.index, ()),
            )
            if target is not None:
                targets.append(target)
        for window_index, window in enumerate(level.windows):
            if window.sloped_ceiling_id is not None:
                continue
            surface = surface_by_id.get(window.wall_surface_id)
            target = _build_window_target(
                window,
                window_index,
                surface,
            )
            if target is not None:
                targets.append(target)
    # Imported lazily to avoid the surface_geometry/sloped_ceiling module
    # cycle while still returning one unified target stream to the viewer.
    from housemaker.sloped_ceiling import (
        build_sloped_ceiling_window_opening_targets,
    )

    targets.extend(
        build_sloped_ceiling_window_opening_targets(
            level_sequence,
            all_surfaces,
        )
    )
    return tuple(sorted(targets, key=lambda target: target.key))


def capture_canvas_window_dimensions(
    levels: Sequence[LevelData],
    surfaces: Sequence[FixedSurface],
    *,
    level_indices: Sequence[int] | None = None,
) -> tuple[CanvasWindowDimensionSnapshot, ...]:
    """Capture physical window rectangles before their hosts are reshaped."""

    normalized_level_indices = (
        None if level_indices is None else {int(value) for value in level_indices}
    )
    levels_by_index = {
        level.index: level for level in levels if isinstance(level, LevelData)
    }
    targets = build_canvas_opening_targets(levels, surfaces)
    snapshots: list[CanvasWindowDimensionSnapshot] = []
    for target in targets:
        reference = target.reference
        if reference.kind != CANVAS_OPENING_WINDOW:
            continue
        if (
            normalized_level_indices is not None
            and reference.level_index not in normalized_level_indices
        ):
            continue
        level = levels_by_index.get(reference.level_index)
        window = (
            None
            if level is None
            else next(
                (
                    candidate
                    for candidate in level.windows
                    if candidate.window_id == reference.stable_id
                ),
                None,
            )
        )
        if window is None:
            continue
        bounds = target.bounds
        snapshots.append(
            CanvasWindowDimensionSnapshot(
                level_index=level.index,
                window=window,
                horizontal_center_ratio=(
                    bounds.start_ratio + bounds.end_ratio
                )
                * 0.5,
                width_meters=(
                    bounds.horizontal_span * target.wall_width_meters
                ),
                bottom_meters=(bounds.bottom_ratio * target.wall_height_meters),
                height_meters=(
                    bounds.vertical_span * target.wall_height_meters
                ),
            )
        )
    return tuple(
        sorted(
            snapshots,
            key=lambda snapshot: (
                snapshot.level_index,
                snapshot.window.window_id,
            ),
        )
    )


def preserve_canvas_window_dimensions(
    levels: Sequence[LevelData],
    surfaces: Sequence[FixedSurface],
    snapshots: Sequence[CanvasWindowDimensionSnapshot],
    *,
    require_every_window: bool = True,
) -> None:
    """Re-normalize windows against edited hosts without resizing them."""

    snapshot_sequence = tuple(snapshots)
    if not snapshot_sequence:
        return
    if not all(
        isinstance(snapshot, CanvasWindowDimensionSnapshot)
        for snapshot in snapshot_sequence
    ):
        raise TypeError("Window preservation requires dimension snapshots.")

    targets_by_key = {
        (target.reference.level_index, target.reference.stable_id): target
        for target in build_canvas_opening_targets(levels, surfaces)
        if target.reference.kind == CANVAS_OPENING_WINDOW
    }
    levels_by_index = {
        level.index: level for level in levels if isinstance(level, LevelData)
    }
    replacements_by_level: dict[int, dict[str, WindowData]] = {}
    for snapshot in snapshot_sequence:
        window_id = snapshot.window.window_id
        level = levels_by_index.get(snapshot.level_index)
        target = targets_by_key.get((snapshot.level_index, window_id))
        current_window = (
            None
            if level is None
            else next(
                (
                    window
                    for window in level.windows
                    if window.window_id == window_id
                ),
                None,
            )
        )
        if level is None or target is None or current_window is None:
            if require_every_window:
                raise ValueError(
                    "A window frame lost its wall while preserving its dimensions."
                )
            continue
        if (
            current_window.wall_surface_id != snapshot.window.wall_surface_id
            or current_window.sloped_ceiling_id
            != snapshot.window.sloped_ceiling_id
            or target.wall_surface_id != snapshot.window.wall_surface_id
            or target.sloped_ceiling_id != snapshot.window.sloped_ceiling_id
        ):
            if require_every_window:
                raise ValueError(
                    "A window frame changed hosts while preserving its dimensions."
                )
            continue

        half_width_ratio = (
            snapshot.width_meters / target.wall_width_meters * 0.5
        )
        bottom_ratio = snapshot.bottom_meters / target.wall_height_meters
        bounds = CanvasOpeningBounds(
            start_ratio=(
                snapshot.horizontal_center_ratio - half_width_ratio
            ),
            end_ratio=(snapshot.horizontal_center_ratio + half_width_ratio),
            bottom_ratio=bottom_ratio,
            top_ratio=(
                bottom_ratio
                + snapshot.height_meters / target.wall_height_meters
            ),
        )
        try:
            _validate_committed_bounds(target, bounds)
        except ValueError as error:
            raise ValueError(
                "The edited wall or ceiling is too small to keep an existing "
                "window frame at its current dimensions."
            ) from error
        replacements_by_level.setdefault(level.index, {})[window_id] = replace(
            current_window,
            start_ratio=bounds.start_ratio,
            end_ratio=bounds.end_ratio,
            bottom_ratio=bounds.bottom_ratio,
            top_ratio=bounds.top_ratio,
        )

    for level_index, replacements in replacements_by_level.items():
        level = levels_by_index[level_index]
        level.windows[:] = [
            replacements.get(window.window_id, window)
            for window in level.windows
        ]


def _build_window_target(
    window: WindowData,
    window_index: int,
    surface: FixedSurface | None,
) -> CanvasOpeningTarget | None:
    if surface is None:
        return None
    frame = _get_surface_frame(surface)
    if frame is None:
        return None
    wall_start, tangent, normal, wall_width, wall_height = frame
    return CanvasOpeningTarget(
        reference=CanvasOpeningReference(
            kind=CANVAS_OPENING_WINDOW,
            level_index=surface.level_index,
            item_index=window_index,
            stable_id=window.window_id,
        ),
        wall_surface_id=surface.surface_id,
        plane_start_world=tuple(float(value) for value in wall_start),
        wall_tangent_world=tuple(float(value) for value in tangent),
        wall_normal_world=tuple(float(value) for value in normal),
        wall_width_meters=wall_width,
        wall_height_meters=wall_height,
        minimum_width_meters=MIN_WINDOW_SIZE_METERS,
        minimum_height_meters=MIN_WINDOW_SIZE_METERS,
        bounds=CanvasOpeningBounds(
            start_ratio=window.start_ratio,
            end_ratio=window.end_ratio,
            bottom_ratio=window.bottom_ratio,
            top_ratio=window.top_ratio,
        ),
        sloped_ceiling_id=None,
    )


def _build_doorway_target(
    level: LevelData,
    doorway: DoorwayData,
    doorway_index: int,
    surfaces: Sequence[FixedSurface],
) -> CanvasOpeningTarget | None:
    selected = _select_doorway_surface(level, doorway, surfaces)
    if selected is None:
        return None
    surface, frame, center_world = selected
    wall_start, tangent, normal, wall_width, wall_height = frame
    normal_offset = float(np.dot(center_world - wall_start, normal))
    plane_start = wall_start + normal * normal_offset
    center_ratio = float(np.dot(center_world - wall_start, tangent) / wall_width)
    level_scale = _normalize_positive_number(level.scale, "level scale")
    half_width_ratio = (
        float(doorway.width_meters) * level_scale / wall_width * 0.5
    )
    bottom_height = float(getattr(doorway, "bottom_height_meters", 0.0))
    return CanvasOpeningTarget(
        reference=CanvasOpeningReference(
            kind=CANVAS_OPENING_DOORWAY,
            level_index=level.index,
            item_index=doorway_index,
            stable_id=doorway.doorway_id,
        ),
        wall_surface_id=surface.surface_id,
        plane_start_world=tuple(float(value) for value in plane_start),
        wall_tangent_world=tuple(float(value) for value in tangent),
        wall_normal_world=tuple(float(value) for value in normal),
        wall_width_meters=wall_width,
        wall_height_meters=wall_height,
        minimum_width_meters=MIN_DOORWAY_WIDTH_METERS * level_scale,
        minimum_height_meters=MIN_DOORWAY_HEIGHT_METERS,
        bounds=CanvasOpeningBounds(
            start_ratio=center_ratio - half_width_ratio,
            end_ratio=center_ratio + half_width_ratio,
            bottom_ratio=bottom_height / wall_height,
            top_ratio=(bottom_height + float(doorway.height_meters))
            / wall_height,
        ),
        arch_amount=(
            doorway.arch_amount
            if doorway.shape == DOORWAY_SHAPE_ARCH
            else None
        ),
        sloped_ceiling_id=None,
    )


def _select_doorway_surface(
    level: LevelData,
    doorway: DoorwayData,
    surfaces: Sequence[FixedSurface],
) -> tuple[
    FixedSurface,
    tuple[np.ndarray, np.ndarray, np.ndarray, float, float],
    np.ndarray,
] | None:
    center_xy = level_image_to_world_xy(
        level,
        doorway.center_x,
        doorway.center_y,
    )
    center_world = np.asarray((center_xy[0], center_xy[1], 0.0), dtype=float)
    rotation_radians = math.radians(float(doorway.rotation_degrees))
    width_image_direction = np.asarray(
        (-math.sin(rotation_radians), math.cos(rotation_radians)),
        dtype=float,
    )
    sample_xy = level_image_to_world_xy(
        level,
        doorway.center_x + float(width_image_direction[0]),
        doorway.center_y + float(width_image_direction[1]),
    )
    width_world_direction = np.asarray(
        (sample_xy[0] - center_xy[0], sample_xy[1] - center_xy[1], 0.0),
        dtype=float,
    )
    width_length = float(np.linalg.norm(width_world_direction))
    if width_length <= OPENING_FRAME_EPSILON:
        return None
    width_world_direction /= width_length

    candidates: list[
        tuple[
            tuple[float, float, str],
            FixedSurface,
            tuple[np.ndarray, np.ndarray, np.ndarray, float, float],
        ]
    ] = []
    for surface in surfaces:
        frame = _get_surface_frame(surface)
        if frame is None:
            continue
        wall_start, tangent, _normal, wall_width, _wall_height = frame
        center_at_wall_height = center_world.copy()
        center_at_wall_height[2] = wall_start[2]
        alignment = abs(float(np.dot(tangent, width_world_direction)))
        projected_distance = float(
            np.dot(center_at_wall_height - wall_start, tangent)
        )
        clamped_distance = min(max(projected_distance, 0.0), wall_width)
        nearest_point = wall_start + tangent * clamped_distance
        wall_distance = float(np.linalg.norm(center_at_wall_height - nearest_point))
        alignment_penalty = max(0.0, 1.0 - alignment) * max(wall_width, 1.0)
        if alignment < DOORWAY_WALL_ALIGNMENT_MINIMUM:
            alignment_penalty += max(wall_width, 1.0) * 10.0
        candidates.append(
            (
                (alignment_penalty + wall_distance, wall_distance, surface.surface_id),
                surface,
                frame,
            )
        )
    if not candidates:
        return None
    _score, surface, frame = min(candidates, key=lambda candidate: candidate[0])
    center_world[2] = frame[0][2]
    return surface, frame, center_world


def _get_surface_frame(
    surface: FixedSurface,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float] | None:
    if (
        surface.wall_start_world is None
        or surface.wall_end_world is None
        or surface.wall_height_meters is None
    ):
        return None
    wall_start = np.asarray(surface.wall_start_world, dtype=float)
    wall_end = np.asarray(surface.wall_end_world, dtype=float)
    wall_height = float(surface.wall_height_meters)
    if (
        wall_start.shape != (3,)
        or wall_end.shape != (3,)
        or not np.all(np.isfinite(wall_start))
        or not np.all(np.isfinite(wall_end))
        or not math.isfinite(wall_height)
        or wall_height <= OPENING_FRAME_EPSILON
    ):
        return None
    wall_axis = wall_end - wall_start
    wall_axis[2] = 0.0
    wall_width = float(np.linalg.norm(wall_axis))
    if wall_width <= OPENING_FRAME_EPSILON:
        return None
    tangent = wall_axis / wall_width
    normal = np.cross(tangent, np.asarray((0.0, 0.0, 1.0), dtype=float))
    normal_length = float(np.linalg.norm(normal))
    if normal_length <= OPENING_FRAME_EPSILON:
        return None
    return wall_start, tangent, normal / normal_length, wall_width, wall_height


# ### Project edit application ###
def apply_canvas_opening_edit(
    levels: Sequence[LevelData],
    target: CanvasOpeningTarget,
    edit: CanvasOpeningEdit,
) -> AppliedCanvasOpeningEdit:
    """Apply validated wall-local bounds and arch state to one opening."""

    if not isinstance(target, CanvasOpeningTarget):
        raise TypeError("Canvas opening edits require their current target frame.")
    if not isinstance(edit, CanvasOpeningEdit):
        raise TypeError("Canvas opening edits require an edit payload.")
    if edit.reference != target.reference:
        raise ValueError("The Canvas opening edit target changed during the drag.")
    destination_target = edit.destination_target
    applied_target = target if destination_target is None else destination_target
    if destination_target is None and edit.wall_surface_id != target.wall_surface_id:
        raise ValueError("The Canvas opening edit moved to a different wall.")
    if (edit.arch_amount is None) != (applied_target.arch_amount is None):
        raise ValueError("The Canvas doorway arch mode changed during the drag.")
    if (
        edit.reference.kind == CANVAS_OPENING_DOORWAY
        and applied_target.sloped_ceiling_id is not None
    ):
        raise ValueError("Canvas doorways cannot move to sloped ceilings.")
    _validate_committed_bounds(applied_target, edit.bounds)
    level = next(
        (
            candidate
            for candidate in levels
            if candidate.index == edit.reference.level_index
        ),
        None,
    )
    if level is None:
        raise ValueError("The edited Canvas opening level no longer exists.")

    if edit.reference.kind == CANVAS_OPENING_WINDOW:
        return _apply_window_edit(level, applied_target, edit)
    return _apply_doorway_edit(level, applied_target, edit)


def _apply_window_edit(
    level: LevelData,
    target: CanvasOpeningTarget,
    edit: CanvasOpeningEdit,
) -> AppliedCanvasOpeningEdit:
    stable_id = edit.reference.stable_id
    window_index = next(
        (
            index
            for index, window in enumerate(level.windows)
            if window.window_id == stable_id
        ),
        None,
    )
    if window_index is None:
        raise ValueError("The edited Canvas window no longer exists.")
    previous = level.windows[window_index]
    current = WindowData(
        window_id=previous.window_id,
        wall_surface_id=target.wall_surface_id,
        start_ratio=edit.bounds.start_ratio,
        end_ratio=edit.bounds.end_ratio,
        bottom_ratio=edit.bounds.bottom_ratio,
        top_ratio=edit.bounds.top_ratio,
        sloped_ceiling_id=target.sloped_ceiling_id,
    )
    level.windows[window_index] = current
    return AppliedCanvasOpeningEdit(
        reference=replace(edit.reference, item_index=window_index),
        level=level,
        previous=previous,
        current=current,
    )


def _apply_doorway_edit(
    level: LevelData,
    target: CanvasOpeningTarget,
    edit: CanvasOpeningEdit,
) -> AppliedCanvasOpeningEdit:
    doorway_index = edit.reference.item_index
    if not 0 <= doorway_index < len(level.doorways):
        raise ValueError("The edited Canvas doorway no longer exists.")
    previous = level.doorways[doorway_index]
    center_ratio = (edit.bounds.start_ratio + edit.bounds.end_ratio) * 0.5
    center_world = target.local_to_world(center_ratio, 0.0)
    center_x, center_y = level_world_to_image_xy(
        level,
        center_world[0],
        center_world[1],
    )
    level_scale = _normalize_positive_number(level.scale, "level scale")
    current = replace(
        previous,
        center_x=center_x,
        center_y=center_y,
        width_meters=(
            edit.bounds.horizontal_span
            * target.wall_width_meters
            / level_scale
        ),
        bottom_height_meters=(
            edit.bounds.bottom_ratio * target.wall_height_meters
        ),
        height_meters=(
            edit.bounds.vertical_span * target.wall_height_meters
        ),
        arch_amount=(
            previous.arch_amount
            if edit.arch_amount is None
            else edit.arch_amount
        ),
    )
    level.doorways[doorway_index] = current
    return AppliedCanvasOpeningEdit(
        reference=edit.reference,
        level=level,
        previous=previous,
        current=current,
    )


def _validate_committed_bounds(
    target: CanvasOpeningTarget,
    bounds: CanvasOpeningBounds,
) -> None:
    tolerance = OPENING_FRAME_EPSILON * 10.0
    if (
        bounds.start_ratio < -tolerance
        or bounds.end_ratio > 1.0 + tolerance
        or bounds.bottom_ratio < -tolerance
        or bounds.top_ratio > 1.0 + tolerance
    ):
        raise ValueError("Canvas opening edits must remain inside their wall.")
    if bounds.horizontal_span + tolerance < target.minimum_horizontal_span:
        raise ValueError("The edited Canvas opening is too narrow.")
    if bounds.vertical_span + tolerance < target.minimum_vertical_span:
        raise ValueError("The edited Canvas opening is too short.")


# ### Validation helpers ###
def _normalize_optional_sloped_ceiling_id(value: object | None) -> str | None:
    """Normalize one optional persistent slope host identity."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("Sloped-ceiling target IDs must be strings.")
    normalized = value.strip().lower()
    if SLOPED_CEILING_ID_PATTERN.fullmatch(normalized) is None:
        raise ValueError(
            "Sloped-ceiling target IDs must be 32-character UUID hex."
        )
    return normalized


def _normalize_optional_doorway_arch_amount(
    reference: CanvasOpeningReference,
    value: object | None,
) -> float | None:
    """Normalize an active doorway arch, leaving other openings unarched."""

    if value is None:
        return None
    if reference.kind != CANVAS_OPENING_DOORWAY:
        raise ValueError("Only Canvas doorways can expose an arch control.")
    return normalize_doorway_arch_amount(value)


def _normalize_finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name.capitalize()} must be a finite number.")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"{field_name.capitalize()} must be a finite number."
        ) from error
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name.capitalize()} must be a finite number.")
    return normalized


def _normalize_positive_number(value: object, field_name: str) -> float:
    normalized = _normalize_finite_number(value, field_name)
    if normalized <= 0.0:
        raise ValueError(f"{field_name.capitalize()} must be positive.")
    return normalized


def _normalize_world_vector(
    value: object,
    field_name: str,
) -> tuple[float, float, float]:
    try:
        vector = np.asarray(tuple(value), dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"{field_name.capitalize()} must contain XYZ values."
        ) from error
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{field_name.capitalize()} must contain finite XYZ values.")
    return tuple(float(component) for component in vector)


def _normalize_unit_vector(
    value: object,
    field_name: str,
) -> tuple[float, float, float]:
    vector = np.asarray(_normalize_world_vector(value, field_name), dtype=float)
    length = float(np.linalg.norm(vector))
    if length <= OPENING_FRAME_EPSILON:
        raise ValueError(f"{field_name.capitalize()} cannot be zero.")
    vector /= length
    return tuple(float(component) for component in vector)
