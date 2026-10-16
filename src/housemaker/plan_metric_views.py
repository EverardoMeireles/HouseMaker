# ### Imports ###
from __future__ import annotations

import base64
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

# ### Constants ###
MAX_PLAN_MEASUREMENT_DETAIL_IMAGES = 4
MAX_PLAN_MEASUREMENT_DETAIL_EDGE = 2_048
INITIAL_DETAIL_BOUNDS = (
    (0.0, 0.0, 0.55, 0.55),
    (0.45, 0.0, 1.0, 0.55),
    (0.0, 0.45, 0.55, 1.0),
    (0.45, 0.45, 1.0, 1.0),
)
RAW_PLAN_DETAIL_ROLE = "original"
RECTIFIED_PLAN_DETAIL_ROLE = "measurement_reference"


# ### Public models ###
@dataclass(frozen=True)
class PlanMeasurementDetailImage:
    """A crop with explicit location in its full source image."""

    label: str
    image_data_uri: str
    source_role: str
    bounds_normalized: tuple[float, float, float, float]


class PlanMeasurementDetailCancelled(RuntimeError):
    """Crop preparation was cancelled before the next inference request."""


# ### Detail preparation ###
def build_plan_measurement_detail_images(
    original_image: np.ndarray,
    *,
    measurement_reference_image: np.ndarray | None = None,
    focus_regions: Sequence[tuple[float, float, float, float]] = (),
    cancellation_check: Callable[[], bool] | None = None,
) -> tuple[PlanMeasurementDetailImage, ...]:
    """Read original-resolution regions before applying any request-size limit.

    Initial overlapping quadrants enlarge small printed labels without trusting
    generated text. Retries keep at least two original-image crops for reading
    authoritative digits, alongside at most two rectified reference crops. Focus
    bounds are never transferred to the raw photograph, which may be warped.
    """

    _raise_if_cancelled(cancellation_check)
    raw_regions = _valid_crop_regions(original_image, INITIAL_DETAIL_BOUNDS)
    reference_regions = (
        _valid_crop_regions(measurement_reference_image, focus_regions)[:2]
        if focus_regions and measurement_reference_image is not None
        else ()
    )
    if reference_regions:
        # Prioritize raw quadrants rich in fine image detail without pretending
        # that their coordinates correspond to the rectified focus regions.
        raw_regions = tuple(sorted(
            raw_regions,
            key=lambda bounds: -_fine_detail_score(original_image, bounds),
        ))[:MAX_PLAN_MEASUREMENT_DETAIL_IMAGES - len(reference_regions)]
    crops = [
        (original_image, RAW_PLAN_DETAIL_ROLE, "ORIGINAL PLAN (RAW)", bounds)
        for bounds in raw_regions
    ]
    if measurement_reference_image is not None:
        crops.extend(
            (measurement_reference_image, RECTIFIED_PLAN_DETAIL_ROLE,
             "RECTIFIED ANNOTATED PLAN", bounds)
            for bounds in reference_regions
        )
    details: list[PlanMeasurementDetailImage] = []
    for source, source_role, source_label, pixel_bounds in crops:
        _raise_if_cancelled(cancellation_check)
        height, width = source.shape[:2]
        left, top, right, bottom = pixel_bounds
        crop = source[top:bottom, left:right]
        # Crop the native raster first; never crop the downsized overview.
        longest_edge = max(crop.shape[:2])
        if longest_edge > MAX_PLAN_MEASUREMENT_DETAIL_EDGE:
            factor = MAX_PLAN_MEASUREMENT_DETAIL_EDGE / longest_edge
            crop = cv2.resize(
                crop,
                (max(1, round(crop.shape[1] * factor)),
                 max(1, round(crop.shape[0] * factor))),
                interpolation=cv2.INTER_AREA,
            )
        success, encoded = cv2.imencode(".png", crop)
        if not success:
            raise ValueError("A plan measurement detail could not be encoded.")
        bounds = (left / width, top / height, right / width, bottom / height)
        bounds_label = ", ".join(f"{value:.6f}" for value in bounds)
        details.append(
            PlanMeasurementDetailImage(
                label=(
                    f"DETAIL {len(details) + 1} of {source_label}; full-source "
                    f"normalized bounds [left, top, right, bottom] = [{bounds_label}]. "
                    "This is a close-up reading aid, not a separate measurement. "
                    "Report endpoint coordinates in the FULL reference/final "
                    "images specified by the schema, NEVER crop-local coordinates. "
                    "Printed values must remain verified against the raw original."
                ),
                image_data_uri=(
                    "data:image/png;base64,"
                    + base64.b64encode(encoded.tobytes()).decode("ascii")
                ),
                source_role=source_role,
                bounds_normalized=bounds,
            )
        )
        if len(details) >= MAX_PLAN_MEASUREMENT_DETAIL_IMAGES:
            break
    _raise_if_cancelled(cancellation_check)
    return tuple(details)


# ### Validation helpers ###
def _valid_crop_regions(
    source: np.ndarray,
    proposed_regions: Sequence[tuple[float, float, float, float]],
) -> tuple[tuple[int, int, int, int], ...]:
    if not isinstance(source, np.ndarray) or source.ndim not in (2, 3):
        raise ValueError("Plan measurement detail source is not an image.")
    height, width = source.shape[:2]
    if height < 2 or width < 2:
        return ()
    regions: list[tuple[int, int, int, int]] = []
    for proposed in proposed_regions:
        bounds = _pixel_bounds(proposed, width, height)
        if bounds is not None and bounds not in regions:
            regions.append(bounds)
    return tuple(regions)


def _fine_detail_score(
    source: np.ndarray, bounds: tuple[int, int, int, int],
) -> float:
    """Favor printed detail over blank paper; no raw/reference mapping implied."""

    left, top, right, bottom = bounds
    crop = source[top:bottom, left:right]
    if crop.ndim == 3:
        crop = crop[..., :3].mean(axis=2).astype(np.uint8)
    if max(crop.shape[:2]) > 512:
        factor = 512 / max(crop.shape[:2])
        crop = cv2.resize(
            crop,
            (max(1, round(crop.shape[1] * factor)),
             max(1, round(crop.shape[0] * factor))),
            interpolation=cv2.INTER_AREA,
        )
    return float(np.mean(np.abs(cv2.Laplacian(crop, cv2.CV_32F))))


def _pixel_bounds(
    bounds: Sequence[float], width: int, height: int,
) -> tuple[int, int, int, int] | None:
    if len(bounds) != 4:
        return None
    try:
        left, top, right, bottom = (float(value) for value in bounds)
    except (ValueError, TypeError):
        return None
    if not all(math.isfinite(value) for value in (left, top, right, bottom)):
        return None
    x0 = max(0, min(width, math.floor(round(left * width, 9))))
    y0 = max(0, min(height, math.floor(round(top * height, 9))))
    x1 = max(0, min(width, math.ceil(round(right * width, 9))))
    y1 = max(0, min(height, math.ceil(round(bottom * height, 9))))
    return (x0, y0, x1, y1) if x1 - x0 >= 2 and y1 - y0 >= 2 else None


def _raise_if_cancelled(cancellation_check: Callable[[], bool] | None) -> None:
    if cancellation_check is not None and cancellation_check():
        raise PlanMeasurementDetailCancelled(
            "Plan measurement detail preparation was cancelled."
        )
