# ### Imports ###
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# ### Constants ###
_MINIMUM_EDGE_PIXELS = 2.0
_MINIMUM_AREA_PIXELS = 16.0
_MINIMUM_FOCAL_WIDTH_RATIO = 0.15
_MAXIMUM_FOCAL_WIDTH_RATIO = 4.0
_MAXIMUM_ASPECT_UNCERTAINTY = 0.15
_FRONT_VIEW_CONVERGENCE_TOLERANCE = 0.02
_FRONT_VIEW_ORTHOGONALITY_TOLERANCE = 0.025
_PERSPECTIVE_WARNING = (
    "Perspective cannot be uniquely corrected from this outline; "
    "use a more frontal frame. This is an apparent image-ratio estimate."
)


# ### Public result model ###
@dataclass(frozen=True)
class OpeningProportionsResult:
    """Dimensionless proportions; the caller supplies the trusted opening width."""

    height_to_width: float
    arch_height_to_width: float
    method: str
    warning: str = ""

    @property
    def arch_amount(self) -> float:
        """Convert the rise to HouseMaker's elliptical-arch parameter."""

        return self.arch_height_to_width / min(0.5, self.height_to_width)


# ### Public measurement entry point ###
def measure_opening_proportions(
    outline: Sequence[Sequence[float]],
    image_size: Sequence[int],
    *,
    arch_apex: Sequence[float] | None = None,
    is_doorway: bool = False,
) -> OpeningProportionsResult:
    """Measure manually marked TL, TR, BR, BL corners in normalized coordinates.

    A rectangular opening constrains focal length only under centered principal
    point, square-pixel, pinhole-camera assumptions, and only when both edge
    directions have sufficient perspective. Unobservable or unstable cases use
    an explicitly labelled apparent-ratio approximation, never an invented FOV.
    For an arch, the top two corners mark the springline and the fifth point
    marks its apex. No image recognition, existing height, or global scale enters
    this calculation.
    """

    size = _validate_image_size(image_size)
    points = _validate_outline(outline, size)
    homography = _square_to_image_homography(points)
    apparent_ratio = _apparent_aspect(points)
    rectified_ratio = _perspective_aspect(points, size)
    if rectified_ratio is not None and _stable_perspective_aspect(
        points, size, rectified_ratio
    ):
        rectangle_ratio = rectified_ratio
        method = "perspective-rectified"
        warning = (
            "Perspective correction assumes a centered, square-pixel camera "
            "and negligible lens distortion. Review the result before applying."
        )
    elif _is_nearly_frontal(points, homography):
        rectangle_ratio = apparent_ratio
        method = "front-view-ratio"
        warning = (
            "Nearly parallel edges: this ratio assumes a straight-on view. "
            "An oblique view can still distort the proportions."
        )
    else:
        rectangle_ratio = apparent_ratio
        method = "apparent-image-ratio"
        warning = _PERSPECTIVE_WARNING

    arch_ratio = 0.0
    if arch_apex is not None:
        if not is_doorway:
            raise ValueError("Only doorway outlines can include an arch apex.")
        arch_ratio = _measure_arch_rise(arch_apex, size, homography, rectangle_ratio)
    total_ratio = rectangle_ratio + arch_ratio
    if arch_ratio > min(0.5, total_ratio) + 1e-7:
        raise ValueError(
            "The marked arch rises more than half the opening width. "
            "Move the top corners to the arch's springline and check the apex."
        )
    return OpeningProportionsResult(total_ratio, arch_ratio, method, warning)


# ### Input validation ###
def _validate_image_size(image_size: Sequence[int]) -> np.ndarray:
    try:
        values = tuple(image_size)
        size = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "The reference image must have a valid width and height."
        ) from error
    if (
        size.shape != (2,)
        or any(isinstance(value, (bool, np.bool_)) for value in values)
        or not np.isfinite(size).all()
        or np.any(size < 2)
        or np.any(size != np.floor(size))
    ):
        raise ValueError("The reference image must have a valid width and height.")
    return size


def _normalized_point(value: Sequence[float], label: str) -> np.ndarray:
    try:
        point = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{label} must be a normalized point inside the image."
        ) from error
    if (
        point.shape != (2,)
        or not np.isfinite(point).all()
        or np.any(point < 0)
        or np.any(point > 1)
    ):
        raise ValueError(f"{label} must be a normalized point inside the image.")
    return point


def _validate_outline(
    outline: Sequence[Sequence[float]], size: np.ndarray
) -> np.ndarray:
    try:
        values = tuple(outline)
    except TypeError as error:
        raise ValueError("Mark all four opening corners.") from error
    if len(values) != 4:
        raise ValueError("Mark all four opening corners.")
    points = np.asarray(
        [_normalized_point(value, "Each opening corner") * size for value in values]
    )
    edges = np.roll(points, -1, axis=0) - points
    if np.min(np.linalg.norm(edges, axis=1)) < _MINIMUM_EDGE_PIXELS:
        raise ValueError("The opening outline is too small or has overlapping corners.")
    crosses = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(
        edges[:, 0], -1
    )
    if not np.all(crosses > 1e-7):
        raise ValueError(
            "The opening outline is crossed or folded. "
            "Mark top-left, top-right, bottom-right, then bottom-left."
        )
    area = 0.5 * abs(
        np.dot(points[:, 0], np.roll(points[:, 1], -1))
        - np.dot(points[:, 1], np.roll(points[:, 0], -1))
    )
    if area < _MINIMUM_AREA_PIXELS:
        raise ValueError("The opening outline is too small to measure.")
    return points


# ### Rectangle camera geometry ###
def _square_to_image_homography(points: np.ndarray) -> np.ndarray:
    square = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    rows: list[list[float]] = []
    values: list[float] = []
    for (u, v), (x, y) in zip(square, points):
        rows.extend(
            (
                [u, v, 1.0, 0.0, 0.0, 0.0, -x * u, -x * v],
                [0.0, 0.0, 0.0, u, v, 1.0, -y * u, -y * v],
            )
        )
        values.extend((float(x), float(y)))
    try:
        solution = np.linalg.solve(np.asarray(rows), np.asarray(values))
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "The opening outline cannot form a valid rectangle."
        ) from error
    return np.append(solution, 1.0).reshape(3, 3)


def _apparent_aspect(points: np.ndarray) -> float:
    edges = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    return float((edges[1] + edges[3]) / (edges[0] + edges[2]))


def _perspective_aspect(points: np.ndarray, size: np.ndarray) -> float | None:
    centered = (points - size / 2.0) / size.max()
    homography = _square_to_image_homography(centered)
    horizontal, vertical = homography[:, 0], homography[:, 1]
    denominator = horizontal[2] * vertical[2]
    if abs(denominator) < 1e-10:
        return None
    focal_squared = -float(np.dot(horizontal[:2], vertical[:2])) / denominator
    if not math.isfinite(focal_squared) or focal_squared <= 0:
        return None
    focal = math.sqrt(focal_squared)
    focal_width_ratio = focal * size.max() / size[0]
    if (
        not _MINIMUM_FOCAL_WIDTH_RATIO
        <= focal_width_ratio
        <= _MAXIMUM_FOCAL_WIDTH_RATIO
    ):
        return None
    horizontal_camera = horizontal.copy()
    vertical_camera = vertical.copy()
    horizontal_camera[:2] /= focal
    vertical_camera[:2] /= focal
    return float(np.linalg.norm(vertical_camera) / np.linalg.norm(horizontal_camera))


def _stable_perspective_aspect(
    points: np.ndarray, size: np.ndarray, aspect: float
) -> bool:
    """Do not amplify a half-pixel manual marking error into a large height edit."""

    for corner in range(4):
        for axis in range(2):
            for direction in (-0.5, 0.5):
                perturbed = points.copy()
                perturbed[corner, axis] += direction
                try:
                    alternative = _perspective_aspect(perturbed, size)
                except ValueError:
                    return False
                if (
                    alternative is None
                    or abs(alternative / aspect - 1.0) > _MAXIMUM_ASPECT_UNCERTAINTY
                ):
                    return False
    return True


def _is_nearly_frontal(points: np.ndarray, homography: np.ndarray) -> bool:
    if np.max(np.abs(homography[2, :2])) > _FRONT_VIEW_CONVERGENCE_TOLERANCE:
        return False
    horizontal = (points[1] - points[0]) + (points[2] - points[3])
    vertical = (points[3] - points[0]) + (points[2] - points[1])
    cosine = abs(float(np.dot(horizontal, vertical))) / (
        np.linalg.norm(horizontal) * np.linalg.norm(vertical)
    )
    return cosine <= _FRONT_VIEW_ORTHOGONALITY_TOLERANCE


# ### Arch measurement ###
def _measure_arch_rise(
    arch_apex: Sequence[float],
    size: np.ndarray,
    homography: np.ndarray,
    rectangle_ratio: float,
) -> float:
    apex = _normalized_point(arch_apex, "The arch apex") * size
    try:
        canonical = np.linalg.solve(homography, (*apex, 1.0))
    except np.linalg.LinAlgError as error:
        raise ValueError("The opening outline cannot locate the arch apex.") from error
    if abs(canonical[2]) < 1e-9:
        raise ValueError("The arch apex is outside the opening's plane.")
    canonical /= canonical[2]
    if not 0.15 <= canonical[0] <= 0.85 or canonical[1] > 1e-5:
        raise ValueError("Place the arch apex above the middle of the two top corners.")
    return max(0.0, float(-canonical[1] * rectangle_ratio))
