"""Content-aware curved joins for a two-by-two sheet of rotated texture tiles.

The input tiles include genuine overscan on every edge. A seam can therefore
move through a small overlap without wrapping or stretching an individual tile.
The same paths compose every aligned PBR map.
"""

# ### Imports ###
from __future__ import annotations

from collections.abc import Callable, Mapping

import cv2
import numpy as np

from housemaker.pbr_maps import ATLAS_MAP_BASE_COLOR, PBR_MAP_METALLIC, PBR_MAP_NORMAL

# ### Constants ###
_MIN_NORMAL_LENGTH = 1e-6
_PATH_SMOOTH_SIGMA = 7.0
_PATH_MOVEMENT_COST = 0.012
_SEAM_COLOR_WEIGHT = 1.0
_SEAM_GRADIENT_WEIGHT = 0.3
_SEAM_EDGE_WEIGHT = 0.12
_TONE_EDGE_BAND_FRACTION = 0.025
_TONE_ALONG_EDGE_SIGMA_FRACTION = 0.015
_MAX_TONE_DELTA_RANGE_FRACTION = 0.45
_MAX_TONE_OUTPUT_EXTENSION_FRACTION = 0.1


# ### Public API ###
def build_curved_variant_sheet(
    texture_maps: Mapping[str, np.ndarray],
    quarter_turns: tuple[int, int, int, int],
    *,
    tile_height: int,
    tile_width: int,
    overlap: int,
    cancellation_check: Callable[[], bool] | None = None,
) -> dict[str, np.ndarray]:
    """Compose four rotated overscan images with shared toroidal curved joins.

    Each input is ``(tile_height + 2*overlap, tile_width + 2*overlap)``.
    The returned sheet is exactly ``(2*tile_height, 2*tile_width)``. Internal
    and outer joins use the same winding paths, so the sheet remains tileable.
    """

    _check_cancelled(cancellation_check)
    maps = _validate_maps(texture_maps, quarter_turns, tile_height, tile_width, overlap)
    reference_type = (
        ATLAS_MAP_BASE_COLOR if ATLAS_MAP_BASE_COLOR in maps else next(iter(maps))
    )
    reference_variants = _reconcile_tile_tones(
        _rotated_variants(maps[reference_type], quarter_turns, reference_type),
        reference_type,
        tile_height,
        tile_width,
        overlap,
    )
    vertical, horizontal = _plan_paths(
        reference_variants, tile_height, tile_width, overlap, cancellation_check
    )

    results: dict[str, np.ndarray] = {}
    for map_type, pixels in maps.items():
        _check_cancelled(cancellation_check)
        variants = (
            reference_variants
            if map_type == reference_type
            else _reconcile_tile_tones(
                _rotated_variants(pixels, quarter_turns, map_type),
                map_type,
                tile_height,
                tile_width,
                overlap,
            )
        )
        results[map_type] = _compose_map(
            variants,
            map_type,
            tile_height,
            tile_width,
            overlap,
            vertical,
            horizontal,
            cancellation_check,
        )
    _check_cancelled(cancellation_check)
    return results


# ### Validation and rotation ###
def _validate_maps(
    texture_maps: Mapping[str, np.ndarray],
    quarter_turns: tuple[int, int, int, int],
    tile_height: int,
    tile_width: int,
    overlap: int,
) -> dict[str, np.ndarray]:
    if not isinstance(texture_maps, Mapping) or not texture_maps:
        raise ValueError("Curved tiling needs at least one aligned texture map.")
    if len(quarter_turns) != 4 or any(
        turn not in (0, 1, 2, 3) for turn in quarter_turns
    ):
        raise ValueError("Four valid quarter-turns are required.")
    if (
        tile_height < 8
        or tile_width < 8
        or not 1 <= overlap < min(tile_height, tile_width) // 3
    ):
        raise ValueError(
            "Curved tiling needs a bounded overlap and valid tile dimensions."
        )
    if tile_height != tile_width and any(turn % 2 for turn in quarter_turns):
        raise ValueError("Rectangular tiles cannot use 90-degree rotations.")
    expected = (tile_height + overlap * 2, tile_width + overlap * 2)
    maps: dict[str, np.ndarray] = {}
    for map_type, pixels in texture_maps.items():
        if not isinstance(map_type, str) or not map_type:
            raise ValueError("Texture map names must be nonempty strings.")
        if not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8:
            raise ValueError("Curved tiling requires uint8 NumPy texture maps.")
        if pixels.ndim != 3 or pixels.shape[2] not in (1, 2, 3, 4):
            raise ValueError("Curved tiling requires one to four texture channels.")
        if pixels.shape[:2] != expected:
            raise ValueError(
                "Aligned overscan maps must match the tile and overlap dimensions."
            )
        if map_type == PBR_MAP_NORMAL and pixels.shape[2] < 3:
            raise ValueError("Normal maps require at least three channels.")
        maps[map_type] = pixels
    return maps


def _rotated_variants(
    pixels: np.ndarray, quarter_turns: tuple[int, int, int, int], map_type: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rotated = []
    for turns in quarter_turns:
        variant = np.ascontiguousarray(np.rot90(pixels, turns))
        if map_type == PBR_MAP_NORMAL and turns:
            variant = _rotate_tangent_normals(variant, turns)
        rotated.append(variant)
    return tuple(rotated)  # type: ignore[return-value]


def _rotate_tangent_normals(pixels: np.ndarray, turns: int) -> np.ndarray:
    vectors = pixels[:, :, :3].astype(np.float32) / 127.5 - 1.0
    x, y = vectors[:, :, 0].copy(), vectors[:, :, 1].copy()
    if turns == 1:
        vectors[:, :, 0], vectors[:, :, 1] = -y, x
    elif turns == 2:
        vectors[:, :, 0], vectors[:, :, 1] = -x, -y
    else:
        vectors[:, :, 0], vectors[:, :, 1] = y, -x
    pixels[:, :, :3] = np.clip(np.rint((vectors * 0.5 + 0.5) * 255), 0, 255).astype(
        np.uint8
    )
    return pixels


# ### Low-frequency edge-tone reconciliation ###
def _reconcile_tile_tones(
    variants: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    map_type: str,
    tile_height: int,
    tile_width: int,
    overlap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Match joins with a bounded correction spread across complete tiles.

    Comparing edge *bands* and smoothing along each join prevents grain from
    becoming a full-height stripe. The correction extends into overscan, so a
    curved path samples the same coherent tone field on both sides. Metallic
    values and tangent normals are left intact rather than globally tinted.
    """

    if map_type in {PBR_MAP_METALLIC, PBR_MAP_NORMAL}:
        return variants
    reconciled = [np.empty_like(variant) for variant in variants]
    channels = variants[0].shape[2]
    band = max(
        1,
        min(overlap, round(min(tile_height, tile_width) * _TONE_EDGE_BAND_FRACTION)),
    )
    x_fraction = np.clip(
        (np.arange(tile_width + 2 * overlap, dtype=np.float32) - overlap)
        / max(tile_width - 1, 1),
        0.0,
        1.0,
    )
    y_fraction = np.clip(
        (np.arange(tile_height + 2 * overlap, dtype=np.float32) - overlap)
        / max(tile_height - 1, 1),
        0.0,
        1.0,
    )
    y_lookup = np.clip(
        np.arange(tile_height + 2 * overlap) - overlap, 0, tile_height - 1
    )
    x_lookup = np.clip(np.arange(tile_width + 2 * overlap) - overlap, 0, tile_width - 1)

    for channel in range(channels):
        # Base-color alpha is coverage, not tone. Keep it exactly aligned with
        # the curved seam but never change its value across a whole cell.
        if map_type == ATLAS_MAP_BASE_COLOR and channel >= 3:
            for result, source in zip(reconciled, variants, strict=True):
                result[:, :, channel] = source[:, :, channel]
            continue
        linear = map_type == ATLAS_MAP_BASE_COLOR and channel < 3
        values = [
            (
                _srgb_to_linear(source[:, :, channel])
                if linear
                else source[:, :, channel].astype(np.float32) / 255.0
            )
            for source in variants
        ]
        source_min = min(float(value.min()) for value in values)
        source_max = max(float(value.max()) for value in values)
        span = source_max - source_min
        if span <= 1e-6:
            for result, source in zip(reconciled, variants, strict=True):
                result[:, :, channel] = source[:, :, channel]
            continue
        delta_limit = span * _MAX_TONE_DELTA_RANGE_FRACTION

        vertical_delta = np.empty((2, 2, tile_height), dtype=np.float32)
        for seam in range(2):
            for row in range(2):
                left = values[row * 2 + (seam - 1) % 2]
                right = values[row * 2 + seam]
                left_band = left[
                    overlap : overlap + tile_height,
                    overlap + tile_width - band : overlap + tile_width,
                ].mean(axis=1)
                right_band = right[
                    overlap : overlap + tile_height, overlap : overlap + band
                ].mean(axis=1)
                vertical_delta[seam, row] = _smooth_edge_delta(
                    (right_band - left_band) * 0.5, tile_height, delta_limit
                )
        for row in range(2):
            for column in range(2):
                left_delta = -vertical_delta[column, row, y_lookup]
                right_delta = vertical_delta[(column + 1) % 2, row, y_lookup]
                values[row * 2 + column] += (
                    left_delta[:, None] * (1.0 - x_fraction[None, :])
                    + right_delta[:, None] * x_fraction[None, :]
                )

        horizontal_delta = np.empty((2, 2, tile_width), dtype=np.float32)
        for seam in range(2):
            for column in range(2):
                top = values[((seam - 1) % 2) * 2 + column]
                bottom = values[seam * 2 + column]
                top_band = top[
                    overlap + tile_height - band : overlap + tile_height,
                    overlap : overlap + tile_width,
                ].mean(axis=0)
                bottom_band = bottom[
                    overlap : overlap + band, overlap : overlap + tile_width
                ].mean(axis=0)
                horizontal_delta[seam, column] = _smooth_edge_delta(
                    (bottom_band - top_band) * 0.5, tile_width, delta_limit
                )
        for row in range(2):
            for column in range(2):
                top_delta = -horizontal_delta[row, column, x_lookup]
                bottom_delta = horizontal_delta[(row + 1) % 2, column, x_lookup]
                value = values[row * 2 + column]
                value += (
                    top_delta[None, :] * (1.0 - y_fraction[:, None])
                    + bottom_delta[None, :] * y_fraction[:, None]
                )
                extension = span * _MAX_TONE_OUTPUT_EXTENSION_FRACTION
                np.clip(
                    value, source_min - extension, source_max + extension, out=value
                )
                encoded = (
                    _linear_to_srgb(value)
                    if linear
                    else np.clip(np.rint(value * 255.0), 0, 255)
                )
                reconciled[row * 2 + column][:, :, channel] = encoded.astype(np.uint8)
    return tuple(reconciled)  # type: ignore[return-value]


def _smooth_edge_delta(delta: np.ndarray, length: int, limit: float) -> np.ndarray:
    sigma = max(1.0, length * _TONE_ALONG_EDGE_SIGMA_FRACTION)
    smoothed = cv2.GaussianBlur(
        delta.astype(np.float32)[:, None], (1, 0), sigmaX=0.0, sigmaY=sigma
    ).ravel()
    return np.clip(smoothed, -limit, limit)


# ### Shared curved-seam planning ###
def _plan_paths(
    variants: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tile_height: int,
    tile_width: int,
    overlap: int,
    cancellation_check: Callable[[], bool] | None,
) -> tuple[np.ndarray, np.ndarray]:
    features = tuple(_planning_features(variant) for variant in variants)
    vertical = np.empty((2, tile_height * 2), dtype=np.float32)
    horizontal = np.empty((2, tile_width * 2), dtype=np.float32)
    # Keep the path at least one blending radius away from the overscan edge.
    # Otherwise the last strip pixel could still contain the other tile and
    # make a second straight seam where the strip meets the untouched center.
    margin = int(np.ceil(_seam_feather(overlap)))
    offsets = np.arange(-overlap + margin, overlap - margin, dtype=np.int32)
    if offsets.size == 0:
        offsets = np.array([0], dtype=np.int32)
    for seam in range(2):
        for row in range(2):
            _check_cancelled(cancellation_check)
            left = features[row * 2 + (seam - 1) % 2]
            right = features[row * 2 + seam]
            y = overlap + np.arange(tile_height)
            left_strip = left[y[:, None], (overlap + tile_width + offsets)[None, :]]
            right_strip = right[y[:, None], (overlap + offsets)[None, :]]
            cost = _seam_cost(left_strip, right_strip)
            vertical[seam, row * tile_height : (row + 1) * tile_height] = _find_path(
                cost, offsets
            )
        for column in range(2):
            _check_cancelled(cancellation_check)
            top = features[((seam - 1) % 2) * 2 + column]
            bottom = features[seam * 2 + column]
            x = overlap + np.arange(tile_width)
            top_strip = top[(overlap + tile_height + offsets)[:, None], x[None, :]]
            bottom_strip = bottom[(overlap + offsets)[:, None], x[None, :]]
            cost = _seam_cost(
                np.swapaxes(top_strip, 0, 1), np.swapaxes(bottom_strip, 0, 1)
            )
            horizontal[seam, column * tile_width : (column + 1) * tile_width] = (
                _find_path(cost, offsets)
            )
    return vertical, horizontal


def _planning_features(pixels: np.ndarray) -> np.ndarray:
    rgb = pixels[:, :, :3].astype(np.float32) / 255.0
    if rgb.shape[2] == 1:
        rgb = np.repeat(rgb, 3, axis=2)
    elif rgb.shape[2] == 2:
        rgb = np.concatenate((rgb, rgb[:, :, :1]), axis=2)
    rgb = cv2.GaussianBlur(rgb, (0, 0), sigmaX=2.0, sigmaY=2.0)
    luminance = rgb[:, :, 0] * 0.2126 + rgb[:, :, 1] * 0.7152 + rgb[:, :, 2] * 0.0722
    return np.stack(
        (luminance, rgb[:, :, 0] - luminance, rgb[:, :, 2] - luminance), axis=2
    )


def _seam_cost(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Prefer matching low-frequency tone while avoiding high-contrast features."""

    color = np.mean(np.abs(left - right), axis=2)
    left_gradient = np.mean(np.abs(np.diff(left, axis=1, prepend=left[:, :1])), axis=2)
    right_gradient = np.mean(
        np.abs(np.diff(right, axis=1, prepend=right[:, :1])), axis=2
    )
    gradient_mismatch = np.abs(left_gradient - right_gradient)
    cost = (
        _SEAM_COLOR_WEIGHT * color
        + _SEAM_GRADIENT_WEIGHT * gradient_mismatch
        + _SEAM_EDGE_WEIGHT * (left_gradient + right_gradient)
    )
    return np.ascontiguousarray(cost, dtype=np.float32)


def _find_path(cost: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Find a bounded path with pinned endpoints and at most 1 px row steps."""

    length, states = cost.shape
    center = int(np.flatnonzero(offsets == 0)[0])
    previous = np.full(states, np.inf, dtype=np.float32)
    previous[center] = cost[0, center]
    parents = np.zeros((length, states), dtype=np.int16)
    for row in range(1, length):
        stay = previous
        from_left = np.roll(previous, 1) + _PATH_MOVEMENT_COST
        from_right = np.roll(previous, -1) + _PATH_MOVEMENT_COST
        from_left[0] = np.inf
        from_right[-1] = np.inf
        choices = np.stack((from_left, stay, from_right), axis=0)
        selection = np.argmin(choices, axis=0)
        parents[row] = (
            np.arange(states, dtype=np.int16) + selection.astype(np.int16) - 1
        )
        previous = cost[row] + np.min(choices, axis=0)
    indices = np.empty(length, dtype=np.int16)
    indices[-1] = center
    for row in range(length - 1, 0, -1):
        indices[row - 1] = parents[row, indices[row]]
    path = offsets[indices].astype(np.float32)
    sigma = min(_PATH_SMOOTH_SIGMA, max(1.0, length / 32.0))
    path = cv2.GaussianBlur(path.reshape(-1, 1), (0, 0), sigmaX=0, sigmaY=sigma).ravel()
    path[0] = path[-1] = 0.0
    return path


# ### Map composition ###
def _compose_map(
    variants: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    map_type: str,
    tile_height: int,
    tile_width: int,
    overlap: int,
    vertical: np.ndarray,
    horizontal: np.ndarray,
    cancellation_check: Callable[[], bool] | None,
) -> np.ndarray:
    height, width = tile_height * 2, tile_width * 2
    output = np.empty((height, width, variants[0].shape[2]), dtype=np.uint8)
    for index, variant in enumerate(variants):
        row, column = divmod(index, 2)
        output[
            row * tile_height : (row + 1) * tile_height,
            column * tile_width : (column + 1) * tile_width,
        ] = variant[overlap : overlap + tile_height, overlap : overlap + tile_width]

    # The right/bottom endpoint is exclusive: the left/top overscan ends at
    # ``tile + 2*overlap - 1``.
    strip_offsets = np.arange(-overlap, overlap, dtype=np.int32)
    feather = _seam_feather(overlap)
    for seam in range(2):
        _check_cancelled(cancellation_check)
        global_x = (seam * tile_width + strip_offsets) % width
        for row in range(2):
            global_y = row * tile_height + np.arange(tile_height)
            src_y = overlap + np.arange(tile_height)
            left = variants[row * 2 + (seam - 1) % 2][
                src_y[:, None], (overlap + tile_width + strip_offsets)[None, :]
            ]
            right = variants[row * 2 + seam][
                src_y[:, None], (overlap + strip_offsets)[None, :]
            ]
            weight = _smoothstep(
                (strip_offsets[None, :] - vertical[seam, global_y, None]) / feather
            )
            output[global_y[:, None], global_x[None, :]] = _blend_pair(
                left, right, weight, map_type
            )

    for seam in range(2):
        _check_cancelled(cancellation_check)
        global_y = (seam * tile_height + strip_offsets) % height
        for column in range(2):
            global_x = column * tile_width + np.arange(tile_width)
            src_x = overlap + np.arange(tile_width)
            top = variants[((seam - 1) % 2) * 2 + column][
                (overlap + tile_height + strip_offsets)[:, None], src_x[None, :]
            ]
            bottom = variants[seam * 2 + column][
                (overlap + strip_offsets)[:, None], src_x[None, :]
            ]
            weight = _smoothstep(
                (strip_offsets[:, None] - horizontal[seam, global_x][None, :]) / feather
            )
            output[global_y[:, None], global_x[None, :]] = _blend_pair(
                top, bottom, weight, map_type
            )

    # Recompose four-way intersections in one pass. The two seam weights form
    # a partition of unity; neither pass can overwrite the other at a corner.
    dy, dx = np.meshgrid(strip_offsets, strip_offsets, indexing="ij")
    for y_seam in range(2):
        global_y = (y_seam * tile_height + strip_offsets) % height
        for x_seam in range(2):
            _check_cancelled(cancellation_check)
            global_x = (x_seam * tile_width + strip_offsets) % width
            top_row, bottom_row = (y_seam - 1) % 2, y_seam
            left_col, right_col = (x_seam - 1) % 2, x_seam
            samples = []
            for row, y_local in (
                (top_row, overlap + tile_height + dy),
                (bottom_row, overlap + dy),
            ):
                for col, x_local in (
                    (left_col, overlap + tile_width + dx),
                    (right_col, overlap + dx),
                ):
                    samples.append(variants[row * 2 + col][y_local, x_local])
            weight_x = _smoothstep((dx - vertical[x_seam, global_y][:, None]) / feather)
            weight_y = _smoothstep(
                (dy - horizontal[y_seam, global_x][None, :]) / feather
            )
            output[global_y[:, None], global_x[None, :]] = _blend_four(
                samples, weight_x, weight_y, map_type
            )
    return np.ascontiguousarray(output)


# ### PBR-aware pixel blending ###
def _seam_feather(overlap: int) -> float:
    return max(1.0, min(4.0, overlap / 5.0))


def _smoothstep(distance: np.ndarray) -> np.ndarray:
    fraction = np.clip(distance * 0.5 + 0.5, 0.0, 1.0)
    return fraction * fraction * (3.0 - 2.0 * fraction)


def _blend_pair(
    left: np.ndarray, right: np.ndarray, weight: np.ndarray, map_type: str
) -> np.ndarray:
    if map_type == PBR_MAP_METALLIC:
        return np.where(weight[:, :, None] >= 0.5, right, left)
    return _weighted_pixels((left, right), (1.0 - weight, weight), map_type)


def _blend_four(
    pixels: list[np.ndarray], weight_x: np.ndarray, weight_y: np.ndarray, map_type: str
) -> np.ndarray:
    weights = (
        (1.0 - weight_y) * (1.0 - weight_x),
        (1.0 - weight_y) * weight_x,
        weight_y * (1.0 - weight_x),
        weight_y * weight_x,
    )
    if map_type == PBR_MAP_METALLIC:
        return np.take_along_axis(
            np.stack(pixels, axis=2),
            np.argmax(np.stack(weights, axis=2), axis=2)[:, :, None, None],
            axis=2,
        )[:, :, 0]
    return _weighted_pixels(tuple(pixels), weights, map_type)


def _weighted_pixels(
    pixels: tuple[np.ndarray, ...], weights: tuple[np.ndarray, ...], map_type: str
) -> np.ndarray:
    channels = pixels[0].shape[2]
    result = np.zeros(pixels[0].shape, dtype=np.float32)
    linear_channels = 3 if map_type == ATLAS_MAP_BASE_COLOR and channels >= 3 else 0
    normal_channels = 3 if map_type == PBR_MAP_NORMAL else 0
    for source, weight in zip(pixels, weights, strict=True):
        if linear_channels:
            result[:, :, :3] += _srgb_to_linear(source[:, :, :3]) * weight[:, :, None]
            if channels > 3:
                result[:, :, 3:] += (
                    source[:, :, 3:].astype(np.float32) * weight[:, :, None]
                )
        elif normal_channels:
            result[:, :, :3] += (
                source[:, :, :3].astype(np.float32) / 127.5 - 1.0
            ) * weight[:, :, None]
            if channels > 3:
                result[:, :, 3:] += (
                    source[:, :, 3:].astype(np.float32) * weight[:, :, None]
                )
        else:
            result += source.astype(np.float32) * weight[:, :, None]
    if linear_channels:
        result[:, :, :3] = _linear_to_srgb(result[:, :, :3])
    elif normal_channels:
        normals = result[:, :, :3]
        lengths = np.linalg.norm(normals, axis=2, keepdims=True)
        normals = normals / np.maximum(lengths, _MIN_NORMAL_LENGTH)
        normals[lengths[:, :, 0] <= _MIN_NORMAL_LENGTH] = (0.0, 0.0, 1.0)
        result[:, :, :3] = (normals * 0.5 + 0.5) * 255.0
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def _srgb_to_linear(pixels: np.ndarray) -> np.ndarray:
    values = pixels.astype(np.float32) / 255.0
    return np.where(
        values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4
    )


def _linear_to_srgb(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    encoded = np.where(
        values <= 0.0031308, values * 12.92, 1.055 * values ** (1.0 / 2.4) - 0.055
    )
    return np.clip(np.rint(encoded * 255.0), 0, 255)


# ### Cancellation ###
def _check_cancelled(cancellation_check: Callable[[], bool] | None) -> None:
    if cancellation_check is not None and cancellation_check():
        raise InterruptedError("Texture tiling repair cancelled.")


# ### Public exports ###
__all__ = ["build_curved_variant_sheet"]
