# ### Imports ###
from __future__ import annotations

import numpy as np

# ### Constants ###
_MIP_SCALES = (1, 4, 16, 64)
_MIN_VARIATION = 0.025
_JOIN_ALLOWANCE = 1.10
_RIDGE_ALLOWANCE = 0.80
_CORNER_RADIUS_FRACTION = 0.06
TILING_FIX_SCORE_THRESHOLD = 1.0
_SCALAR_MAP_TYPES = frozenset(
    {
        "ambient_occlusion",
        "ao",
        "displacement",
        "glossiness",
        "height",
        "metallic",
        "occlusion",
        "roughness",
        "specular",
        "specular_glossiness",
    }
)


# ### Public scoring API ###
def score_variant_sheet(
    sheet: np.ndarray,
    tile_height: int,
    tile_width: int,
    *,
    map_type: str = "base_color",
) -> float:
    """Estimate visibility of a two-by-two sheet's internal and wrap seams.

    Lower is better. The score compares every join with adjacent texture detail
    at native and mip-like box-filtered scales. A centered stripe and each of
    the four toroidal junctions get separate penalties; a good average cannot
    hide one conspicuous corner. This is a ranking heuristic, not a guarantee
    that two semantic features (such as bricks) line up.
    """

    _validate_sheet(sheet, tile_height, tile_width)
    normalized_type = map_type.strip().lower()
    scale_scores: list[float] = []
    for scale in _MIP_SCALES:
        if scale > min(tile_height, tile_width) // 4:
            continue
        seam_lines = {
            (axis, split): _seam_line_scores(
                sheet, axis=axis, split=split, scale=scale, map_type=normalized_type
            )
            for axis, splits in ((1, (0, tile_width)), (0, (0, tile_height)))
            for split in splits
        }
        seam_scores = [_summarize_line(values) for values in seam_lines.values()]
        corner_scores = _corner_scores(
            seam_lines, tile_height, tile_width, scale
        )
        scale_scores.append(
            0.30 * float(np.mean(seam_scores))
            + 0.35 * max(seam_scores)
            + 0.35 * max(corner_scores)
        )
    if not scale_scores:
        return 0.0
    # A broad seam visible in a mip must not be drowned out by a good native
    # pixel score (or vice versa).
    return float(0.4 * np.mean(scale_scores) + 0.6 * max(scale_scores))


def score_repeated_texture_tile(
    tile: np.ndarray,
    *,
    map_type: str = "base_color",
) -> float:
    """Score the visible seams produced by repeating one texture tile."""

    _validate_tile(tile)
    tile_height, tile_width = tile.shape[:2]
    repeated_sheet = np.tile(tile, (2, 2, 1))
    return score_variant_sheet(
        repeated_sheet,
        tile_height,
        tile_width,
        map_type=map_type,
    )


def texture_needs_tiling_fix(
    tile: np.ndarray,
    *,
    map_type: str = "base_color",
) -> bool:
    """Return whether one repeated texture tile has a visible seam."""

    return (
        score_repeated_texture_tile(tile, map_type=map_type)
        >= TILING_FIX_SCORE_THRESHOLD
    )


# ### Input validation ###
def _validate_tile(tile: np.ndarray) -> None:
    if not isinstance(tile, np.ndarray):
        raise TypeError("The repeated texture tile must be a NumPy array.")
    if tile.dtype != np.uint8 or tile.ndim != 3 or tile.shape[2] not in (1, 2, 3, 4):
        raise ValueError(
            "The repeated texture tile must contain uint8 pixels with 1–4 channels."
        )
    if min(tile.shape[:2]) < 4:
        raise ValueError(
            "Repeated texture tiles must be at least four pixels wide and high."
        )


def _validate_sheet(sheet: np.ndarray, tile_height: int, tile_width: int) -> None:
    if not isinstance(sheet, np.ndarray):
        raise TypeError("The variant sheet must be a NumPy array.")
    if sheet.dtype != np.uint8 or sheet.ndim != 3 or sheet.shape[2] not in (1, 2, 3, 4):
        raise ValueError(
            "The variant sheet must contain uint8 pixels with 1–4 channels."
        )
    if tile_height < 4 or tile_width < 4:
        raise ValueError("Variant tiles must be at least four pixels wide and high.")
    if sheet.shape[:2] != (2 * tile_height, 2 * tile_width):
        raise ValueError("The sheet must contain exactly two by two variant tiles.")


# ### Multi-scale seam measurements ###
def _seam_line_scores(
    sheet: np.ndarray,
    *,
    axis: int,
    split: int,
    scale: int,
    map_type: str,
) -> np.ndarray:
    """Return one relative seam-visibility score per pooled line position."""

    longitudinal = sheet.shape[1 - axis]
    transverse = sheet.shape[axis]
    indices = (np.arange(split - 4 * scale, split + 4 * scale) % transverse).astype(
        np.intp
    )
    strip = np.take(sheet, indices, axis=axis)
    if axis == 0:
        strip = np.swapaxes(strip, 0, 1)
    pooled_length = longitudinal // scale
    strip = strip[: pooled_length * scale]
    strip = strip.reshape(pooled_length, scale, 8, scale, sheet.shape[2])
    pooled = strip.mean(axis=(1, 3), dtype=np.float32) / 255.0
    features = _visual_features(pooled, map_type)

    left_inner = features[:, 3]
    right_inner = features[:, 4]
    interior = 0.25 * (
        _distance(features[:, 0], features[:, 1])
        + _distance(features[:, 1], features[:, 2])
        + _distance(features[:, 5], features[:, 6])
        + _distance(features[:, 6], features[:, 7])
    )
    jump = _distance(left_inner, right_inner)
    ridge = _distance(
        (left_inner + right_inner) * 0.5,
        (features[:, 2] + features[:, 5]) * 0.5,
    )
    denominator = interior + _MIN_VARIATION
    jump_excess = np.maximum(jump - _JOIN_ALLOWANCE * interior, 0.0)
    ridge_excess = np.maximum(ridge - _RIDGE_ALLOWANCE * interior, 0.0)
    return (jump_excess + 0.65 * ridge_excess) / denominator


def _visual_features(pooled: np.ndarray, map_type: str) -> np.ndarray:
    if map_type in _SCALAR_MAP_TYPES or pooled.shape[2] < 3:
        return pooled[:, :, :1]
    if map_type == "normal":
        return pooled[:, :, :3] * 2.0 - 1.0
    rgb = pooled[:, :, :3]
    luminance = (
        0.2126 * rgb[:, :, 0]
        + 0.7152 * rgb[:, :, 1]
        + 0.0722 * rgb[:, :, 2]
    )
    red_chroma = (rgb[:, :, 0] - rgb[:, :, 1]) * 0.35
    blue_chroma = (rgb[:, :, 2] - rgb[:, :, 1]) * 0.35
    if pooled.shape[2] == 4:
        return np.stack(
            (luminance, red_chroma, blue_chroma, pooled[:, :, 3]), axis=2
        )
    return np.stack((luminance, red_chroma, blue_chroma), axis=2)


def _distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.linalg.norm(first - second, axis=1)


# ### Worst-seam and junction summaries ###
def _summarize_line(scores: np.ndarray) -> float:
    if scores.size == 0:
        return 0.0
    return float(
        0.40 * np.mean(scores)
        + 0.35 * np.quantile(scores, 0.90)
        + 0.25 * np.quantile(scores, 0.98)
    )


def _corner_scores(
    seam_lines: dict[tuple[int, int], np.ndarray],
    tile_height: int,
    tile_width: int,
    scale: int,
) -> list[float]:
    """Score all four junctions, including the three involving sheet wrap."""

    scores: list[float] = []
    for y in (0, tile_height):
        for x in (0, tile_width):
            vertical = seam_lines[(1, x)]
            horizontal = seam_lines[(0, y)]
            y_radius = max(1, round(tile_height * _CORNER_RADIUS_FRACTION / scale))
            x_radius = max(1, round(tile_width * _CORNER_RADIUS_FRACTION / scale))
            y_index = y // scale
            x_index = x // scale
            vertical_near = vertical[
                (np.arange(y_index - y_radius, y_index + y_radius) % vertical.size)
            ]
            horizontal_near = horizontal[
                (np.arange(x_index - x_radius, x_index + x_radius) % horizontal.size)
            ]
            scores.append(
                0.5
                * (_summarize_line(vertical_near) + _summarize_line(horizontal_near))
            )
    return scores


# ### Public exports ###
__all__ = [
    "TILING_FIX_SCORE_THRESHOLD",
    "score_repeated_texture_tile",
    "score_variant_sheet",
    "texture_needs_tiling_fix",
]
