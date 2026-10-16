# ### Imports ###
from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from .models import (
    DEFAULT_LEVEL_SCALE,
    MAX_CANVAS_OFFSET_PIXELS,
    MIN_CANVAS_OFFSET_PIXELS,
    LevelData,
)
from .plan_correction_exchange import ExternalPlanCorrection
from .plan_metric_state import PlanMetricCalibration


# ### Image binding ###
def matches_external_plan_image(
    correction: ExternalPlanCorrection,
    path: str | Path | None,
    size: tuple[float, float] | None,
) -> bool:
    """Match a manually loaded result, never an arbitrary original photograph."""
    return bool(
        path
        and Path(path).name.casefold() == correction.image_filename.casefold()
        and size is not None
        and tuple(size) == correction.image_size
    )


def external_plan_uses_loaded_coordinates(
    level: LevelData,
    correction: ExternalPlanCorrection,
    path: str | Path | None,
    *,
    loaded_revision: tuple[object, ...] | None = None,
    current_revision: tuple[object, ...] | None = None,
) -> bool:
    """Distinguish metadata for the loaded raster from image replacement.

    Old projects may have no recorded revision or image fingerprint. For those,
    an unchanged loaded path and pixel dimensions are the available binding;
    this does not assert that the drawing itself was correctly registered.
    """
    if (
        not level.image_path
        or not path
        or correction.level_index != level.index
        or not matches_external_plan_image(
            correction, level.image_path, level.image_size_pixels,
        )
    ):
        return False
    loaded_path = Path(level.image_path).expanduser().resolve()
    candidate_path = Path(path).expanduser().resolve()
    if candidate_path != loaded_path:
        return False

    changed_message = (
        f"{level.display_name}'s plan image changed outside the app. Its JSON "
        "cannot be applied as a scale-only correction to the existing geometry. "
        "Restore the image used to create that geometry, or remove/remap the "
        "geometry before loading a replacement image."
    )
    if loaded_revision is not None and loaded_revision != current_revision:
        raise ValueError(changed_message)
    calibration = level.plan_metric_calibration
    expected_hash = (
        calibration.normalized_image_sha256 if calibration is not None else ""
    )
    if expected_hash:
        actual_hash = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(changed_message)
    return True


def prepare_external_plan_calibration(
    correction: ExternalPlanCorrection,
    path: str | Path,
) -> PlanMetricCalibration | None:
    """Check the file contract; retain optional evidence without a quality gate."""
    image_path = Path(path)
    with Image.open(image_path) as image:
        size = image.size
        image.verify()
    if not matches_external_plan_image(correction, image_path, size):
        raise ValueError(
            f"L{correction.level_index} expects {correction.image_filename} "
            f"at {correction.image_size[0]} x {correction.image_size[1]} pixels. "
            "Load that corrected image without resizing it, or import updated JSON."
        )
    density = correction.meters_per_pixel
    if density is None or len(correction.measurements) < 2:
        return None
    residuals = []
    for measurement in correction.measurements:
        start = measurement.corrected_start_normalized
        end = measurement.corrected_end_normalized
        pixel_length = math.hypot(
            (end[0] - start[0]) * size[0], (end[1] - start[1]) * size[1]
        )
        residuals.append(abs(pixel_length * density / measurement.value_meters - 1))
    try:
        return PlanMetricCalibration(
            estimated_meters_per_pixel=density,
            target_meters_per_pixel=density,
            normalization_scale=1.0,
            relative_residual=sum(residuals) / len(residuals),
            confidence=correction.confidence,
            source_image_size=size,
            pre_normalization_size=size,
            normalized_image_size=size,
            measurements=correction.measurements,
            normalized_image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
        )
    except ValueError:
        # Repeated evidence is not an independent calibration. Still accept the
        # explicitly imported density, without fabricating validated evidence.
        return None


# ### Metadata application ###
def apply_external_plan_metadata(
    level: LevelData,
    correction: ExternalPlanCorrection,
    calibration: PlanMetricCalibration | None,
) -> None:
    level.external_plan_correction = correction
    level.plan_metric_calibration = calibration
    level.plan_meters_per_pixel = correction.meters_per_pixel
    if correction.meters_per_pixel is not None:
        level.scale = DEFAULT_LEVEL_SCALE


def align_external_plan_levels(
    levels: Sequence[LevelData],
    viewport_size: tuple[float, float],
) -> None:
    """Align shared landmarks using X/Y offsets only, independent of zoom/pan."""
    bound_levels = {
        level.index: level
        for level in levels
        if level.external_plan_correction is not None
        and matches_external_plan_image(
            level.external_plan_correction, level.image_path, level.image_size_pixels
        )
    }
    for level in bound_levels.values():
        correction = level.external_plan_correction
        assert correction is not None
        reference = bound_levels.get(correction.reference_level_index)
        if reference is None or reference is level:
            continue
        reference_correction = reference.external_plan_correction
        assert reference_correction is not None
        if (
            reference_correction.reference_level_index != correction.reference_level_index
            or correction.alignment_point_normalized is None
            or reference_correction.alignment_point_normalized is None
        ):
            continue
        reference_anchor = _display_anchor(reference, viewport_size)
        target_anchor = _display_anchor(level, viewport_size)
        offset = (
            reference.canvas_offset_x_pixels + reference_anchor[0] - target_anchor[0],
            reference.canvas_offset_y_pixels + reference_anchor[1] - target_anchor[1],
        )
        if all(MIN_CANVAS_OFFSET_PIXELS <= value <= MAX_CANVAS_OFFSET_PIXELS for value in offset):
            level.canvas_offset_x_pixels, level.canvas_offset_y_pixels = offset


# ### Canvas coordinate helpers ###
def _display_anchor(
    level: LevelData,
    viewport_size: tuple[float, float],
) -> tuple[float, float]:
    correction = level.external_plan_correction
    assert correction is not None and correction.alignment_point_normalized is not None
    width, height = correction.image_size
    fit = min(viewport_size[0] / width, viewport_size[1] / height)
    scale = fit * level.canvas_level_scale
    x, y = correction.alignment_point_normalized
    return ((x - 0.5) * width * scale, (y - 0.5) * height * scale)
