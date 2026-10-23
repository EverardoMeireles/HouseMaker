# ### Imports ###
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from statistics import median

from housemaker.canvas_openings import CANVAS_OPENING_WINDOW, CanvasOpeningTarget
from housemaker.global_scale import (
    MAX_GLOBAL_MULTIPLIER,
    MIN_GLOBAL_MULTIPLIER,
    GlobalScale,
)
from housemaker.models import DOORWAY_SHAPE_RECTANGULAR, DoorwayData, WindowData
from housemaker.window_calibration_state import (
    MANUAL_CALIBRATION_MODEL,
    WindowCalibrationPair,
    confirmed_vertical_guide,
    confirmed_window_outline,
    has_confirmed_calibration_handles,
)

# ### Calibration application models ###
MIN_MEASUREMENT_CONFIDENCE = 0.65
CONSENSUS_RELATIVE_TOLERANCE = 0.12
JOINT_PROPORTION_METHOD = "manual_joint_scene_proportions"
# Compare authored distances, not rounded display values or relative percentages.
DIMENSION_CHANGE_EPSILON_METERS = 1e-7


@dataclass(frozen=True)
class WindowCalibrationAnchor:
    window: WindowData | DoorwayData
    target: CanvasOpeningTarget
    level_height: float

    def physical_width(self, scale: GlobalScale) -> float:
        tangent = self.target.wall_tangent_world
        stretch = math.sqrt(
            sum(
                (component * factor) ** 2
                for component, factor in zip(tangent, scale.factors)
            )
        )
        return (
            self.target.wall_width_meters * self.target.bounds.horizontal_span * stretch
        )


@dataclass(frozen=True)
class WindowCalibrationApplication:
    windows: tuple[WindowData, ...]
    scale: GlobalScale
    inferred_z: float | None
    pairs: tuple[WindowCalibrationPair, ...]
    summary: str
    doorways: tuple[DoorwayData, ...] = ()


# ### Stable opening identity ###
def calibration_anchor_for_pair(
    pair: WindowCalibrationPair,
    anchors: Mapping[str, WindowCalibrationAnchor],
) -> WindowCalibrationAnchor | None:
    """Resolve typed identities while accepting pre-existing window callers."""
    anchor = anchors.get(pair.opening_key)
    if anchor is None and pair.opening_kind == CANVAS_OPENING_WINDOW:
        anchor = anchors.get(pair.window_id)
    if (
        anchor is None
        or anchor.target.reference.kind != pair.opening_kind
        or anchor.target.reference.stable_id != pair.opening_id
    ):
        return None
    if pair.opening_kind == CANVAS_OPENING_WINDOW:
        if (
            not isinstance(anchor.window, WindowData)
            or anchor.window.window_id != pair.opening_id
        ):
            return None
    elif (
        not isinstance(anchor.window, DoorwayData)
        or anchor.window.doorway_id != pair.opening_id
    ):
        return None
    return anchor


# ### Robust aggregation ###
def _consensus(values: Sequence[float]) -> float | None:
    """Require a strict agreeing majority; two conflicting views cannot win."""
    if not values:
        return None
    center = median(values)
    agreeing = [
        value
        for value in values
        if abs(value - center) <= max(abs(center), 1e-9) * CONSENSUS_RELATIVE_TOLERANCE
    ]
    if len(agreeing) * 2 <= len(values):
        return None
    return sum(agreeing) / len(agreeing)


def _shared_level_scale(values: Sequence[float | None]) -> float | None:
    """A project-wide Z must suit every measured level, not just the majority."""
    if not values or any(value is None for value in values):
        return None
    center = median(values)
    if any(
        abs(value - center) > abs(center) * CONSENSUS_RELATIVE_TOLERANCE
        for value in values
    ):
        return None
    return sum(values) / len(values)


def _with_calculation_details(pair: WindowCalibrationPair, message: str) -> str:
    """Retain geometric uncertainty alongside the applied dimension, not just in a tooltip."""
    measurement = pair.measurement
    if (
        measurement is None
        or measurement.landmarks.get("solver") != MANUAL_CALIBRATION_MODEL
    ):
        return message
    details = measurement.reason.strip()
    return f"{message} {details}" if details else message


def _agreeing_view_ids(
    views: Sequence[WindowCalibrationPair], field: str
) -> set[str] | None:
    """Return a strict agreeing majority among views measuring this field."""
    measured = [
        (pair.pair_id, getattr(pair.measurement, field))
        for pair in views
        if getattr(pair.measurement, field) is not None
    ]
    if not measured:
        return None
    center = median(value for _pair_id, value in measured)
    agreeing = {
        pair_id
        for pair_id, value in measured
        if abs(value - center) <= max(abs(center), 1e-9) * CONSENSUS_RELATIVE_TOLERANCE
    }
    return agreeing if len(agreeing) * 2 > len(measured) else None


def _coherent_view_measurements(
    views: Sequence[WindowCalibrationPair],
) -> tuple[list[WindowCalibrationPair], float | None, float | None, float | None, bool]:
    """Do not construct a window from unrelated per-field outlier subsets.

    First reject views whose opening aspect disagrees with the majority. Only
    these views can supply sill/ceiling measurements. Optional measurements
    must also agree; conflicting room evidence blocks Z, but does not prevent
    a reliable opening-height fit. Finally use one common subset for every
    accepted ratio, so an angle rejected for one ratio cannot affect another.
    """
    height_ids = _agreeing_view_ids(views, "height_to_width")
    if height_ids is None:
        return [], None, None, None, False
    height_views = [pair for pair in views if pair.pair_id in height_ids]
    optional_fields = ("sill_to_width", "room_height_to_width")
    optional_ids = {
        field: _agreeing_view_ids(height_views, field) for field in optional_fields
    }
    room_present = any(
        pair.measurement.room_height_to_width is not None for pair in height_views
    )
    room_conflict = room_present and optional_ids["room_height_to_width"] is None
    accepted = [
        pair
        for pair in height_views
        if all(
            ids is None
            or getattr(pair.measurement, field) is None
            or pair.pair_id in ids
            for field, ids in optional_ids.items()
        )
    ]
    if len(accepted) * 2 <= len(views):
        # Different majorities for sill and room may have no coherent majority
        # together. Retain the reliable opening height, not a synthetic blend.
        accepted = height_views
        optional_ids = {field: None for field in optional_fields}
        room_conflict = room_present

    def average(field: str) -> float | None:
        values = [
            getattr(pair.measurement, field)
            for pair in accepted
            if getattr(pair.measurement, field) is not None
        ]
        return sum(values) / len(values) if values else None

    return (
        accepted,
        average("height_to_width"),
        average("sill_to_width") if optional_ids["sill_to_width"] else None,
        average("room_height_to_width")
        if optional_ids["room_height_to_width"]
        else None,
        room_conflict,
    )


# ### Shared scene-fit results ###
def _joint_metadata(pair: WindowCalibrationPair) -> dict | None:
    """Read explicit scene-fit provenance, including failures without a measurement."""
    metadata = dict(pair.measurement.landmarks) if pair.measurement is not None else {}
    metadata.update(pair.calculation_diagnostics)
    method = metadata.get("geometry_method")
    if method is None and pair.measurement is not None:
        method = pair.measurement.method
    return metadata if method == JOINT_PROPORTION_METHOD else None


def _joint_scale_proposal(
    pairs: Sequence[WindowCalibrationPair],
) -> tuple[bool, float | None, str]:
    """Consume one jointly fitted scale; never re-average its derived room heights."""
    evidence = [value for pair in pairs if (value := _joint_metadata(pair)) is not None]
    if not evidence:
        return False, None, ""
    if any(value.get("scale_status_code") != "ok" for value in evidence):
        contradictory = any(
            value.get("scale_status_code") == "inconsistent_scale" for value in evidence
        )
        return (
            True,
            None,
            "Global Z unchanged: opening-balanced scene fitting found incompatible floor-to-ceiling proportions. Review the individual opening outcomes."
            if contradictory
            else "Global Z unchanged: the joint scene fit could not determine a reliable vertical scale from these views.",
        )
    values = [value.get("global_z") for value in evidence]
    if any(
        not isinstance(value, (float, int))
        or not math.isfinite(value)
        or not MIN_GLOBAL_MULTIPLIER <= value <= MAX_GLOBAL_MULTIPLIER
        for value in values
    ):
        return (
            True,
            None,
            "Global Z unchanged: the joint scale result is missing or invalid; calculate proportions again.",
        )
    fitted_z = float(values[0])
    if any(
        not math.isclose(value, fitted_z, rel_tol=1e-6, abs_tol=1e-8)
        for value in values
    ):
        return (
            True,
            None,
            "Global Z unchanged: these results do not belong to one consistent joint fit; calculate proportions again.",
        )
    return True, fitted_z, ""


def _joint_opening_fractions(
    views: Sequence[WindowCalibrationPair], fitted_z: float
) -> tuple[float, float] | None:
    """Fractions are usable only with the exact scale they were fitted alongside."""
    fractions = []
    for pair in views:
        metadata = _joint_metadata(pair)
        if (
            metadata is None
            or metadata.get("scale_status_code") != "ok"
            or pair.measurement is None
            or pair.measurement.sill_to_width is None
        ):
            continue
        z = metadata.get("global_z")
        bottom, top = metadata.get("bottom_fraction"), metadata.get("top_fraction")
        if bottom is None and top is None:
            continue  # Independently fitted fallback, not part of this joint solution.
        if (
            not all(
                isinstance(value, (float, int)) and math.isfinite(value)
                for value in (z, bottom, top)
            )
            or not math.isclose(z, fitted_z, rel_tol=1e-6, abs_tol=1e-8)
            or not 0 <= bottom < top <= 1
        ):
            return None
        fractions.append((float(bottom), float(top)))
    if not fractions:
        return None
    first = fractions[0]
    if any(
        not all(
            math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-8) for a, b in zip(first, value)
        )
        for value in fractions[1:]
    ):
        return None
    return first


# ### Physical dimensions and absolute Z fitting ###
def plan_window_calibration_application(
    pairs: Sequence[WindowCalibrationPair],
    anchors: Mapping[str, WindowCalibrationAnchor],
    scale: GlobalScale,
    *,
    require_edited_handles: bool = False,
    preserve_global_scale: bool = False,
) -> WindowCalibrationApplication:
    """Preflight the whole transaction without changing a scene or wall topology.

    The joint fit preserves the supplied horizontal widths. Each opening votes once and
    each level votes once for Z, even when it has many photographs. Inferred Z
    is an absolute factor against authored height, never a correction applied
    repeatedly to the previous factor. X/Y and the Link axes setting are kept.
    Live workspaces also require explicit handle edits before accepting cached
    measurements; legacy standalone callers can still read older projects.
    """
    grouped: dict[str, list[WindowCalibrationPair]] = defaultdict(list)
    resolved_anchors: dict[str, WindowCalibrationAnchor] = {}
    statuses: dict[str, str] = {}
    contradictory_room_evidence = False
    for pair in pairs:
        if not pair.enabled:
            continue
        anchor = calibration_anchor_for_pair(pair, anchors)
        if anchor is None or anchor.target.reference.level_index != pair.level_index:
            statuses[pair.pair_id] = "Not applied: this opening no longer exists."
        elif any(
            not math.isfinite(value) or value <= 0
            for value in (anchor.level_height, anchor.target.wall_height_meters)
        ):
            statuses[pair.pair_id] = (
                "Not applied: the level and wall heights must be positive and finite."
            )
        elif isinstance(anchor.window, DoorwayData) and (
            anchor.window.shape != DOORWAY_SHAPE_RECTANGULAR
        ):
            statuses[pair.pair_id] = (
                "Not applied: arched doorways are not supported yet."
            )
        elif anchor.target.sloped_ceiling_id is not None:
            statuses[pair.pair_id] = (
                "Not applied: sloped window frames are not supported yet."
            )
        elif (
            require_edited_handles and not has_confirmed_calibration_handles(pair)
        ) or (
            pair.vertical_guide_lines
            and (
                not confirmed_vertical_guide(pair) or not confirmed_window_outline(pair)
            )
        ):
            statuses[pair.pair_id] = (
                "Not applied: move all four opening corners and all four circular guide endpoints first."
            )
        elif (
            pair.measurement is not None
            and pair.measurement.confidence >= MIN_MEASUREMENT_CONFIDENCE
        ):
            measurement = pair.measurement
            if (
                measurement.room_height_to_width is not None
                and measurement.height_to_width + (measurement.sill_to_width or 0)
                > measurement.room_height_to_width + 1e-6
            ):
                statuses[pair.pair_id] = (
                    "Not applied: opening/sill exceeds the measured room height. Review the boundaries."
                )
            else:
                grouped[pair.opening_key].append(pair)
                resolved_anchors[pair.opening_key] = anchor
        elif pair.measurement is not None:
            statuses[pair.pair_id] = (
                "Not applied: calculation is too uncertain. "
                if pair.measurement.landmarks.get("solver") == MANUAL_CALIBRATION_MODEL
                else "Not applied: insufficient geometric confidence. "
            ) + pair.measurement.reason

        # An explicitly contradictory level must not disappear from the Z vote
        # just because its unreliable room ratio (or whole measurement) was withheld.
        if (
            anchor is not None
            and anchor.target.reference.level_index == pair.level_index
            and (
                pair.calculation_diagnostics.get("wall_evidence_conflict") is True
                or (
                    pair.measurement is not None
                    and pair.measurement.landmarks.get("wall_evidence_conflict") is True
                )
            )
        ):
            contradictory_room_evidence = True

    joint_mode, joint_z, joint_detail = _joint_scale_proposal(
        [pair for pair in pairs if pair.enabled and pair.pair_id not in statuses]
    )
    if preserve_global_scale:
        # After scene calibration, opening measurements are physical dimensions
        # within the accepted scene. Cached joint fractions cannot change it.
        joint_mode, joint_z, joint_detail = False, None, ""
    dimensions: dict[str, tuple[float, float | None]] = {}
    level_z: dict[int, list[float]] = defaultdict(list)
    room_evidence = 0
    accepted_by_opening: dict[str, list[WindowCalibrationPair]] = {}
    for opening_key, views in grouped.items():
        anchor = resolved_anchors[opening_key]
        fitted_views = (
            [
                pair
                for pair in views
                if (metadata := _joint_metadata(pair)) is not None
                and metadata.get("status_code") == "ok"
            ]
            if joint_z is not None
            else []
        )
        accepted, height_ratio, sill_ratio, room_ratio, room_conflict = (
            _coherent_view_measurements(fitted_views or views)
        )
        if height_ratio is None:
            for pair in views:
                statuses[pair.pair_id] = (
                    "Not applied: photographs disagree on opening proportions."
                )
            continue
        accepted_by_opening[opening_key] = accepted
        accepted_ids = {pair.pair_id for pair in accepted}
        for pair in views:
            if pair.pair_id not in accepted_ids:
                statuses[pair.pair_id] = (
                    "Not applied: this view disagrees with the other views of this opening."
                )
        width = anchor.physical_width(scale)
        dimensions[opening_key] = (
            width * height_ratio,
            None if sill_ratio is None else width * sill_ratio,
        )
        if any(pair.measurement.room_height_to_width is not None for pair in accepted):
            room_evidence += 1
            if room_conflict or room_ratio is None:
                contradictory_room_evidence = True
        if room_ratio is not None:
            candidate_z = width * room_ratio / anchor.level_height
            if MIN_GLOBAL_MULTIPLIER <= candidate_z <= MAX_GLOBAL_MULTIPLIER:
                level_z[anchor.target.reference.level_index].append(candidate_z)
            else:
                contradictory_room_evidence = True

    # Do not quietly discard an internally contradictory level and let another
    # level determine a project-wide scale for it.
    per_level = [_consensus(values) for values in level_z.values()]
    inferred_z = (
        None
        if preserve_global_scale
        else (
            joint_z
            if joint_mode
            else (
                None
                if contradictory_room_evidence
                or any(value is None for value in per_level)
                else _shared_level_scale(per_level)
            )
        )
    )
    next_scale = scale if inferred_z is None else replace(scale, z=inferred_z)
    updates: list[WindowData] = []
    doorway_updates: list[DoorwayData] = []
    matching_openings: set[str] = set()
    blocked_by_wall: set[str] = set()
    for opening_key, (height, sill) in dimensions.items():
        anchor = resolved_anchors[opening_key]
        physical_wall_height = anchor.target.wall_height_meters * next_scale.z
        is_doorway = isinstance(anchor.window, DoorwayData)
        fractions = (
            _joint_opening_fractions(accepted_by_opening[opening_key], inferred_z)
            if joint_mode and inferred_z is not None
            else None
        )
        if fractions is not None:
            # Fit the opening and its floor-to-ceiling envelope together. Only
            # convert to scene units AFTER choosing the shared new scale.
            room_height = anchor.level_height * next_scale.z
            height = (fractions[1] - fractions[0]) * room_height
            sill = fractions[0] * room_height
        if is_doorway:
            # Doorways keep their authored floor/threshold offset. Only their
            # height is inferred; plan dimensions and bridge topology stay fixed.
            bottom = (
                anchor.window.bottom_height_meters / anchor.target.wall_height_meters
            )
        else:
            bottom = (
                anchor.window.bottom_ratio
                if sill is None
                else sill / physical_wall_height
            )
        top = bottom + height / physical_wall_height
        if (
            height / next_scale.z < anchor.target.minimum_height_meters
            or bottom < -1e-6
            or top > 1.0 + 1e-6
        ):
            blocked_by_wall.add(opening_key)
            message = (
                "Not applied: inferred opening does not fit its wall. "
                f"Calculated height {height:.3f} m + "
                f"{'threshold' if is_doorway else 'sill'} {bottom * physical_wall_height:.3f} m "
                f"places its top at {top * physical_wall_height:.3f} m; "
                f"available wall height is {physical_wall_height:.3f} m. "
            )
            if inferred_z is None and top > 1.0 + 1e-6 and not preserve_global_scale:
                message += (
                    "Global Z could not be recalculated, so the existing ceiling "
                    "still blocks this fit. Changing the opening's current height "
                    "will not resolve the conflicting scale evidence. "
                )
            message += "Review the marked views and fixed plan widths."
        elif is_doorway:
            authored_height = height / next_scale.z
            matches = math.isclose(
                anchor.window.height_meters,
                authored_height,
                rel_tol=0.0,
                abs_tol=DIMENSION_CHANGE_EPSILON_METERS,
            )
            if matches:
                matching_openings.add(opening_key)
            else:
                doorway_updates.append(
                    replace(anchor.window, height_meters=authored_height)
                )
            message = (
                "Already matches:" if matches else "Applied:"
            ) + f" height {height:.3f} m; threshold unchanged."
        else:
            fitted = replace(
                anchor.window,
                bottom_ratio=max(0.0, bottom),
                top_ratio=min(1.0, top),
            )
            matches = all(
                abs(previous - updated) * anchor.target.wall_height_meters
                <= DIMENSION_CHANGE_EPSILON_METERS
                for previous, updated in (
                    (anchor.window.bottom_ratio, fitted.bottom_ratio),
                    (anchor.window.top_ratio, fitted.top_ratio),
                )
            )
            if matches:
                matching_openings.add(opening_key)
            else:
                updates.append(fitted)
            message = ("Already matches:" if matches else "Applied:") + (
                f" height {height:.3f} m"
            )
            message += "; sill unchanged." if sill is None else f", sill {sill:.3f} m."
        for pair in accepted_by_opening[opening_key]:
            statuses[pair.pair_id] = _with_calculation_details(pair, message)

    if preserve_global_scale:
        detail = "Scene-calibrated Global X/Y/Z preserved; only opening dimensions were calculated."
    elif inferred_z is None:
        detail = (
            joint_detail
            if joint_mode
            else (
                "Global Z unchanged: inconsistent floor-to-ceiling evidence; the measured levels/openings do not agree on one Z multiplier."
                if room_evidence or contradictory_room_evidence
                else "Global Z unchanged: no reliable floor-to-ceiling measurement."
            )
        )
    else:
        detail = (
            f"Global Z {'already matches' if next_scale == scale else 'set to'} "
            f"{inferred_z:.4f} (all levels); X/Y unchanged."
        )
        if joint_mode:
            detail += " Opening proportions and Global Z fitted together."
        if room_evidence == 1:
            detail += " Single-opening estimate; add another opening to cross-check."
    if matching_openings:
        detail += f" {len(matching_openings)} opening(s) already match; not reapplied."
    if blocked_by_wall:
        detail += (
            f" {len(blocked_by_wall)} opening(s) could not fit the wall height "
            "and were left unchanged; see their calculated and available dimensions."
        )
    fit_weights = [
        weight
        for pair in pairs
        if pair.enabled
        and (metadata := _joint_metadata(pair)) is not None
        and isinstance(weight := metadata.get("sample_weight"), (float, int))
        and not isinstance(weight, bool)
        and math.isfinite(weight)
        and 0 <= weight <= 1
    ]
    if fit_weights and joint_z is not None:
        detail += (
            f" Shared fit: {sum(weight > 0 for weight in fit_weights)} sample(s) retained"
            f" ({sum(0 < weight < 0.99 for weight in fit_weights)} downweighted), "
            f"{sum(weight == 0 for weight in fit_weights)} excluded."
        )
    review_count = sum(
        pair.enabled
        and not statuses.get(pair.pair_id, pair.status).startswith(
            ("Applied:", "Already matches:")
        )
        for pair in pairs
    )
    if review_count:
        detail += f" {review_count} sample(s) need review; see the individual sample outcomes."
    return WindowCalibrationApplication(
        windows=tuple(updates),
        scale=next_scale,
        inferred_z=inferred_z,
        pairs=tuple(
            replace(pair, status=statuses[pair.pair_id])
            if pair.pair_id in statuses
            else pair
            for pair in pairs
        ),
        summary=(
            f"Updated {len(updates)} window frame(s)"
            + (
                f" and {len(doorway_updates)} doorway(s)"
                if any(
                    pair.enabled and pair.opening_kind != CANVAS_OPENING_WINDOW
                    for pair in pairs
                )
                else ""
            )
            + f". {detail}"
        ),
        doorways=tuple(doorway_updates),
    )
