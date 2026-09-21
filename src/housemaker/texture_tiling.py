# ### Imports ###
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from housemaker.pbr_maps import (
    ATLAS_MAP_BASE_COLOR,
    PBR_MAP_METALLIC,
    PBR_MAP_NORMAL,
    PBR_MAP_ROUGHNESS,
)

# ### Constants ###
_MIN_ROTATION_SIZE = 16
_MAX_ROTATION_SIZE = 256
_ROTATION_GRID_DIVISOR = 4
_PATCH_FEATHER_FRACTION = 0.08
_BORDER_BLEND_FRACTION = 0.04
_MIN_NORMAL_LENGTH = 1e-6
_MIN_MEANINGFUL_VARIATION = 1e-5
_INTERNAL_SEAM_ALLOWANCE = 1.5
_LOCAL_PBR_SEAM_CORRECTION_FRACTION = 0.04
_LOCAL_PBR_SEAM_CORRECTION_MAX_WIDTH = 32
_MAX_EDGE_CORRECTION_RANGE_FRACTION = 0.5
_MAX_OUTPUT_RANGE_EXTENSION_FRACTION = 0.1
_EDGE_NEIGHBORHOOD_FRACTION = 0.03
_MAX_EDGE_NEIGHBORHOOD_WIDTH = 64
_SCALAR_MAP_TYPES = frozenset(
    {
        PBR_MAP_ROUGHNESS,
        "ambient_occlusion",
        "ao",
        "occlusion",
        "specular",
        "specular_glossiness",
        "glossiness",
        "height",
        "displacement",
    }
)


# ### Public data models ###
@dataclass(frozen=True)
class TilingRepairOperation:
    """One seeded quarter-turn of a square region of an aligned texture set."""

    top: int
    left: int
    size: int
    quarter_turns: int
    feather: int

    def __post_init__(self) -> None:
        if self.top < 0 or self.left < 0 or self.size <= 0:
            raise ValueError("A tiling rotation requires a positive in-bounds square.")
        if self.quarter_turns not in {1, 2, 3}:
            raise ValueError("A tiling rotation must use one to three quarter-turns.")
        if not 0 < self.feather < self.size // 2:
            raise ValueError("A tiling rotation needs a valid feather width.")


@dataclass(frozen=True)
class TilingRepairPlan:
    """Shared, deterministic rotations and border blending for aligned maps."""

    height: int
    width: int
    offset_y: int
    offset_x: int
    estimated_period_y: int | None
    estimated_period_x: int | None
    operations: tuple[TilingRepairOperation, ...]
    edge_blend_width_y: int = 0
    edge_blend_width_x: int = 0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ValueError("A tiling repair plan requires non-empty dimensions.")
        if self.offset_y != 0 or self.offset_x != 0:
            raise ValueError("Rotation plans cannot offset texture coordinates.")
        if not 0 <= self.edge_blend_width_y <= self.height // 2:
            raise ValueError("The vertical border blend width is invalid.")
        if not 0 <= self.edge_blend_width_x <= self.width // 2:
            raise ValueError("The horizontal border blend width is invalid.")
        for operation in self.operations:
            if operation.top + operation.size > self.height:
                raise ValueError("A tiling rotation exceeds the image height.")
            if operation.left + operation.size > self.width:
                raise ValueError("A tiling rotation exceeds the image width.")


@dataclass(frozen=True)
class TilingRepairResult:
    """Repaired maps and diagnostics returned without changing the sources."""

    maps: dict[str, np.ndarray]
    plan: TilingRepairPlan
    seam_score_before: float
    seam_score_after: float


# ### Edge-compatible variant data model ###
@dataclass(frozen=True)
class EdgeCompatibleVariantResult:
    """A two-by-two sheet of full-tile rotations with reconciled joins."""

    maps: dict[str, np.ndarray]
    tile_height: int
    tile_width: int
    quarter_turns: tuple[int, int, int, int]
    seed: int


# ### Public repair API ###
def repair_texture_tiling(
    texture_maps: Mapping[str, np.ndarray],
    *,
    reference_map_type: str = ATLAS_MAP_BASE_COLOR,
    seed: int = 0,
    cancellation_check: Callable[[], bool] | None = None,
) -> TilingRepairResult:
    """Rotate aligned square patches and make their outer borders repeat.

    A fixed seed gives stable results across previews and exports. The plan is
    applied identically to every PBR map; no pixels are resampled or cropped.
    Color blends in linear light, normal vectors rotate and renormalize, and
    metallic values are never interpolated.
    """

    _raise_if_tiling_repair_cancelled(cancellation_check)
    normalized_maps = _normalize_texture_maps(texture_maps)
    normalized_reference = _normalize_map_type(reference_map_type)
    if normalized_reference not in normalized_maps:
        raise ValueError(
            f"The tiling repair reference map {normalized_reference!r} is missing."
        )
    _validate_seed(seed)

    # Choose a map with real structure even when the base color is flat.
    planning_map_type = normalized_reference
    planning_strength = -1.0
    for map_type, source in normalized_maps.items():
        _raise_if_tiling_repair_cancelled(cancellation_check)
        features = _map_features(source, map_type)
        strength = _feature_variation(features) + _texture_tiling_seam_severity(
            source, map_type
        )
        if strength > planning_strength:
            planning_map_type = map_type
            planning_strength = strength

    _raise_if_tiling_repair_cancelled(cancellation_check)
    plan = build_tiling_repair_plan(
        normalized_maps[planning_map_type],
        map_type=planning_map_type,
        seed=seed,
    )
    _raise_if_tiling_repair_cancelled(cancellation_check)
    seam_score_before = _aggregate_tiling_seam_score(normalized_maps)
    repaired: dict[str, np.ndarray] = {}
    for map_type, source in normalized_maps.items():
        _raise_if_tiling_repair_cancelled(cancellation_check)
        repaired[map_type] = apply_tiling_repair_plan(source, map_type, plan)
        _raise_if_tiling_repair_cancelled(cancellation_check)

    seam_score_after = _aggregate_tiling_seam_score(repaired)
    changed = any(
        not np.array_equal(repaired[map_type], source)
        for map_type, source in normalized_maps.items()
    )
    if not changed or seam_score_after > seam_score_before + 1e-7:
        plan = _identity_tiling_repair_plan((plan.height, plan.width), seed=seed)
        repaired = {
            map_type: source.copy() for map_type, source in normalized_maps.items()
        }
        seam_score_after = seam_score_before
    _raise_if_tiling_repair_cancelled(cancellation_check)
    return TilingRepairResult(
        maps=repaired,
        plan=plan,
        seam_score_before=seam_score_before,
        seam_score_after=seam_score_after,
    )


# ### Edge-compatible variant API ###
def create_edge_compatible_variants(
    texture_maps: Mapping[str, np.ndarray],
    *,
    reference_map_type: str = ATLAS_MAP_BASE_COLOR,
    seed: int = 0,
    cancellation_check: Callable[[], bool] | None = None,
) -> EdgeCompatibleVariantResult:
    """Rotate whole tiles and reconcile their seams without an inset patch.

    Each cell contains a complete source-image rotation. Color and scalar-map
    corrections at internal and wrapped joins are distributed across each
    whole cell; normal and metallic corrections stay local to their joins.
    There is no shared unrotated border or square rotation boundary. The seam
    lies between texels, and its gradient follows neighboring texels. A
    consumer must select one sheet cell per whole texture repeat.
    """

    _raise_if_tiling_repair_cancelled(cancellation_check)
    normalized_maps = _normalize_texture_maps(texture_maps)
    normalized_reference = _normalize_map_type(reference_map_type)
    if normalized_reference not in normalized_maps:
        raise ValueError(
            f"The tiling repair reference map {normalized_reference!r} is missing."
        )
    _validate_seed(seed)
    tile_height, tile_width = next(iter(normalized_maps.values())).shape[:2]
    generator = np.random.default_rng(seed)
    possible_turns = (0, 1, 2, 3) if tile_height == tile_width else (0, 2, 0, 2)
    quarter_turns = tuple(int(value) for value in generator.permutation(possible_turns))

    sheets: dict[str, np.ndarray] = {}
    for map_type, source in normalized_maps.items():
        _raise_if_tiling_repair_cancelled(cancellation_check)
        sheet = np.empty(
            (tile_height * 2, tile_width * 2, source.shape[2]), dtype=np.uint8
        )
        for index, turns in enumerate(quarter_turns):
            variant = np.rot90(source, turns).copy()
            if map_type == PBR_MAP_NORMAL and turns:
                variant = _rotate_tangent_normals(variant, turns)
            row, column = divmod(index, 2)
            sheet[
                row * tile_height : (row + 1) * tile_height,
                column * tile_width : (column + 1) * tile_width,
            ] = variant
        sheets[map_type] = _stitch_whole_tile_sheet(
            sheet, map_type, tile_height, tile_width, cancellation_check
        )
        _raise_if_tiling_repair_cancelled(cancellation_check)

    return EdgeCompatibleVariantResult(
        maps=sheets,
        tile_height=tile_height,
        tile_width=tile_width,
        quarter_turns=quarter_turns,
        seed=seed,
    )


def build_tiling_repair_plan(
    reference_map: np.ndarray,
    *,
    map_type: str = ATLAS_MAP_BASE_COLOR,
    seed: int = 0,
) -> TilingRepairPlan:
    """Build reproducible quarter-turns of in-place square source patches."""

    reference = _validate_texture_array(reference_map, "Reference texture")
    normalized_type = _normalize_map_type(map_type)
    if normalized_type == PBR_MAP_NORMAL and reference.shape[2] < 3:
        raise ValueError("A tangent normal map requires at least three channels.")
    _validate_seed(seed)
    height, width = reference.shape[:2]
    features = _map_features(reference, normalized_type)
    if (
        min(height, width) < _MIN_ROTATION_SIZE
        or _feature_variation(features) <= _MIN_MEANINGFUL_VARIATION
    ):
        return _identity_tiling_repair_plan((height, width), seed=seed)

    size = min(
        _MAX_ROTATION_SIZE,
        max(_MIN_ROTATION_SIZE, min(height, width) // _ROTATION_GRID_DIVISOR),
    )
    feather = max(2, min(size // 3, round(size * _PATCH_FEATHER_FRACTION)))
    top_margin = (height % size) // 2
    left_margin = (width % size) // 2
    generator = np.random.default_rng(seed)
    operations: list[TilingRepairOperation] = []
    for top in range(top_margin, height - size + 1, size):
        for left in range(left_margin, width - size + 1, size):
            quarter_turns = int(generator.integers(0, 4))
            if quarter_turns:
                operations.append(
                    TilingRepairOperation(
                        top=top,
                        left=left,
                        size=size,
                        quarter_turns=quarter_turns,
                        feather=feather,
                    )
                )
    if not operations:
        operations.append(
            TilingRepairOperation(
                top=top_margin,
                left=left_margin,
                size=size,
                quarter_turns=int(generator.integers(1, 4)),
                feather=feather,
            )
        )

    return TilingRepairPlan(
        height=height,
        width=width,
        offset_y=0,
        offset_x=0,
        estimated_period_y=None,
        estimated_period_x=None,
        operations=tuple(operations),
        edge_blend_width_y=_border_blend_width(height),
        edge_blend_width_x=_border_blend_width(width),
        seed=seed,
    )


def apply_tiling_repair_plan(
    source_map: np.ndarray,
    map_type: str,
    plan: TilingRepairPlan,
) -> np.ndarray:
    """Apply one shared geometric plan with map-specific pixel semantics."""

    source = _validate_texture_array(source_map, "Texture map")
    if source.shape[:2] != (plan.height, plan.width):
        raise ValueError("A texture map does not match its tiling repair plan.")
    normalized_type = _normalize_map_type(map_type)
    if normalized_type == PBR_MAP_NORMAL and source.shape[2] < 3:
        raise ValueError("A tangent normal map requires at least three channels.")
    repaired = source.copy()
    for operation in plan.operations:
        square = source[
            operation.top : operation.top + operation.size,
            operation.left : operation.left + operation.size,
        ]
        rotated = np.rot90(square, operation.quarter_turns).copy()
        if normalized_type == PBR_MAP_NORMAL:
            rotated = _rotate_tangent_normals(rotated, operation.quarter_turns)
        mask = _rotation_feather_mask(operation.size, operation.feather)
        target = repaired[
            operation.top : operation.top + operation.size,
            operation.left : operation.left + operation.size,
        ]
        target[:] = _blend_texture_pixels(target, rotated, mask, normalized_type)
    _close_repeating_borders(repaired, normalized_type, plan)
    return np.ascontiguousarray(repaired, dtype=np.uint8)


def texture_tiling_seam_score(
    source_map: np.ndarray,
    *,
    map_type: str = ATLAS_MAP_BASE_COLOR,
) -> float:
    """Return the mean normalized mismatch across both repeating borders."""

    source = _validate_texture_array(source_map, "Texture map")
    features = _map_features(source, _normalize_map_type(map_type))
    scores = _feature_border_scores(features)
    return float(np.mean(scores)) if scores else 0.0


# ### Plan and cancellation helpers ###
def _raise_if_tiling_repair_cancelled(
    cancellation_check: Callable[[], bool] | None,
) -> None:
    if cancellation_check is not None and cancellation_check():
        raise InterruptedError("Texture tiling repair cancelled.")


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("A tiling rotation seed must be a non-negative integer.")


def _identity_tiling_repair_plan(
    shape: tuple[int, int], *, seed: int = 0
) -> TilingRepairPlan:
    height, width = shape
    return TilingRepairPlan(height, width, 0, 0, None, None, (), seed=seed)


def _border_blend_width(dimension: int) -> int:
    return min(dimension // 2, max(2, round(dimension * _BORDER_BLEND_FRACTION)))


def _feature_variation(features: np.ndarray) -> float:
    """Measure spatial structure, not differences between feature channels."""

    return float(np.sqrt(np.mean(np.var(features, axis=(0, 1)))))


def _rotation_feather_mask(size: int, feather: int) -> np.ndarray:
    coordinates = np.arange(size, dtype=np.float32)
    edge_distance = np.minimum(coordinates, coordinates[::-1])
    distance = np.minimum(edge_distance[:, None], edge_distance[None, :])
    fraction = np.clip(distance / feather, 0.0, 1.0)
    return np.ascontiguousarray(fraction * fraction * (3.0 - 2.0 * fraction))


# ### Full-tile variant seam helpers ###
def _stitch_whole_tile_sheet(
    sheet: np.ndarray,
    map_type: str,
    tile_height: int,
    tile_width: int,
    cancellation_check: Callable[[], bool] | None,
) -> np.ndarray:
    """Match the two joins on each axis with bounded, tile-wide corrections.

    Each scalar channel is processed alone to keep 2048-pixel inputs within a
    practical memory footprint. Metallic and tangent-normal corrections stay
    near the join so their material values and directions do not drift across
    an entire object surface.
    """

    stitched = np.empty_like(sheet)
    for channel in range(sheet.shape[2]):
        _raise_if_tiling_repair_cancelled(cancellation_check)
        values = sheet[:, :, channel].astype(np.float32)
        if map_type == ATLAS_MAP_BASE_COLOR and channel < 3:
            values = _srgb_to_linear(values)
        elif map_type == PBR_MAP_NORMAL and channel < 3:
            values = values / 127.5 - 1.0
        source_min = float(values.min())
        source_max = float(values.max())
        local_width = None
        if map_type in {PBR_MAP_METALLIC, PBR_MAP_NORMAL} and channel < 3:
            local_width = _local_pbr_seam_correction_width
        values = _distribute_variant_seam_correction(
            values, axis=1, period=tile_width, local_width=local_width
        )
        values = _distribute_variant_seam_correction(
            values, axis=0, period=tile_height, local_width=local_width
        )
        extension = (source_max - source_min) * _MAX_OUTPUT_RANGE_EXTENSION_FRACTION
        np.clip(values, source_min - extension, source_max + extension, out=values)
        if map_type == ATLAS_MAP_BASE_COLOR and channel < 3:
            stitched[:, :, channel] = _linear_to_srgb(values)
        elif map_type == PBR_MAP_NORMAL and channel < 3:
            stitched[:, :, channel] = np.clip(
                np.rint((values * 0.5 + 0.5) * 255.0), 0, 255
            ).astype(np.uint8)
        else:
            stitched[:, :, channel] = np.clip(np.rint(values), 0, 255).astype(
                np.uint8
            )
    if map_type == PBR_MAP_NORMAL:
        _renormalize_normal_sheet(stitched)
    return np.ascontiguousarray(stitched)


def _distribute_variant_seam_correction(
    values: np.ndarray,
    *,
    axis: int,
    period: int,
    local_width: Callable[[int], int] | None,
) -> np.ndarray:
    """Align edge neighborhoods while preserving source grain in the cells.

    A single edge texel is an unstable guide for a noisy texture: its random
    value would become a stripe when correction is spread across the cell.
    Scalar and color maps therefore compare neighboring *bands*, sized for
    mip sampling. Vector/discrete PBR maps retain a short local correction.
    """

    moved = np.moveaxis(values, axis, 0)
    size = moved.shape[0]
    deltas = np.zeros_like(moved)
    band = min(
        _MAX_EDGE_NEIGHBORHOOD_WIDTH,
        max(2, round(period * _EDGE_NEIGHBORHOOD_FRACTION)),
        max(1, period // 4),
    )
    for split in (0, period):
        a = (split - 1) % size
        b = split % size
        if local_width is None and band > 1:
            edge_left = np.take(
                moved, np.arange(split - band, split), axis=0, mode="wrap"
            ).mean(axis=0)
            edge_right = np.take(
                moved, np.arange(split, split + band), axis=0, mode="wrap"
            ).mean(axis=0)
            before = np.take(
                moved, np.arange(split - band * 2, split - band),
                axis=0, mode="wrap"
            ).mean(axis=0)
            after = np.take(
                moved, np.arange(split + band, split + band * 2),
                axis=0, mode="wrap"
            ).mean(axis=0)
        else:
            before = moved[(split - 2) % size]
            after = moved[(split + 1) % size]
            edge_left = moved[a]
            edge_right = moved[b]
        slope = ((edge_left - before) + (after - edge_right)) * 0.5
        midpoint = (edge_left + edge_right) * 0.5
        deltas[a] = midpoint - slope * 0.5 - edge_left
        deltas[b] = midpoint + slope * 0.5 - edge_right

    span = float(moved.max() - moved.min())
    correction_limit = span * _MAX_EDGE_CORRECTION_RANGE_FRACTION
    np.clip(deltas, -correction_limit, correction_limit, out=deltas)

    corrected = moved.copy()
    for tile_index in range(2):
        start = tile_index * period
        end = start + period
        left_delta = deltas[start]
        right_delta = deltas[end - 1]
        if local_width is None:
            fraction = np.linspace(0.0, 1.0, period, dtype=np.float32)
            profile = (
                (1.0 - fraction).reshape((-1,) + (1,) * left_delta.ndim)
                * left_delta
                + fraction.reshape((-1,) + (1,) * right_delta.ndim) * right_delta
            )
        else:
            band = local_width(period)
            distance = np.arange(period, dtype=np.float32)
            left_weight = np.clip(1.0 - distance / band, 0.0, 1.0)
            right_weight = left_weight[::-1]
            left_weight = left_weight.reshape((-1,) + (1,) * left_delta.ndim)
            right_weight = right_weight.reshape((-1,) + (1,) * right_delta.ndim)
            profile = left_weight * left_delta + right_weight * right_delta
        corrected[start:end] += profile
    return np.moveaxis(corrected, 0, axis)


def _local_pbr_seam_correction_width(period: int) -> int:
    return min(
        _LOCAL_PBR_SEAM_CORRECTION_MAX_WIDTH,
        max(2, round(period * _LOCAL_PBR_SEAM_CORRECTION_FRACTION)),
    )


def _renormalize_normal_sheet(sheet: np.ndarray) -> None:
    for top in range(0, sheet.shape[0], 256):
        chunk = sheet[top : top + 256, :, :3]
        vectors = chunk.astype(np.float32) / 127.5 - 1.0
        lengths = np.linalg.norm(vectors, axis=2, keepdims=True)
        vectors = vectors / np.maximum(lengths, _MIN_NORMAL_LENGTH)
        vectors[lengths[:, :, 0] <= _MIN_NORMAL_LENGTH] = (0.0, 0.0, 1.0)
        chunk[:] = np.clip(
            np.rint((vectors * 0.5 + 0.5) * 255.0), 0, 255
        ).astype(np.uint8)




# ### Patch rotation and repeating borders ###
def _rotate_tangent_normals(patch: np.ndarray, quarter_turns: int) -> np.ndarray:
    """Rotate tangent XY in UV space as the image is rotated in array space."""

    vectors = patch[:, :, :3].astype(np.float32) / 127.5 - 1.0
    x = vectors[:, :, 0].copy()
    y = vectors[:, :, 1].copy()
    if quarter_turns == 1:
        vectors[:, :, 0], vectors[:, :, 1] = -y, x
    elif quarter_turns == 2:
        vectors[:, :, 0], vectors[:, :, 1] = -x, -y
    else:
        vectors[:, :, 0], vectors[:, :, 1] = y, -x
    patch[:, :, :3] = np.clip(np.rint((vectors * 0.5 + 0.5) * 255.0), 0, 255).astype(
        np.uint8
    )
    return patch


def _close_repeating_borders(
    pixels: np.ndarray, map_type: str, plan: TilingRepairPlan
) -> None:
    if map_type == PBR_MAP_METALLIC:
        if plan.edge_blend_width_x:
            pixels[:, -1] = pixels[:, 0]
        if plan.edge_blend_width_y:
            pixels[-1] = pixels[0]
        return
    if plan.edge_blend_width_x:
        width = plan.edge_blend_width_x
        left = pixels[:, :width].copy()
        right = pixels[:, -width:][:, ::-1].copy()
        weights = _border_pair_weights(width, pixels.shape[0], axis=1)
        pixels[:, :width] = _blend_texture_pixels(left, right, weights, map_type)
        pixels[:, -width:] = _blend_texture_pixels(right, left, weights, map_type)[
            :, ::-1
        ]
    if plan.edge_blend_width_y:
        width = plan.edge_blend_width_y
        top = pixels[:width].copy()
        bottom = pixels[-width:][::-1].copy()
        weights = _border_pair_weights(width, pixels.shape[1], axis=0)
        pixels[:width] = _blend_texture_pixels(top, bottom, weights, map_type)
        pixels[-width:] = _blend_texture_pixels(bottom, top, weights, map_type)[::-1]


def _border_pair_weights(width: int, other_dimension: int, *, axis: int) -> np.ndarray:
    distance = np.arange(width, dtype=np.float32)
    fraction = distance / max(width - 1, 1)
    smooth = fraction * fraction * (3.0 - 2.0 * fraction)
    half_weight = 0.5 * (1.0 - smooth)
    if axis == 1:
        return np.broadcast_to(half_weight[None, :], (other_dimension, width))
    return np.broadcast_to(half_weight[:, None], (width, other_dimension))


# ### Pixel blending ###
def _blend_texture_pixels(
    target: np.ndarray,
    patch: np.ndarray,
    blend_mask: np.ndarray,
    map_type: str,
) -> np.ndarray:
    weights = blend_mask[:, :, None].astype(np.float32)
    if map_type == PBR_MAP_METALLIC:
        return np.where(weights >= 0.5, patch, target).astype(np.uint8)
    if map_type == PBR_MAP_NORMAL:
        return _blend_normal_pixels(target, patch, weights)
    if map_type == ATLAS_MAP_BASE_COLOR and target.shape[2] >= 3:
        result = np.empty_like(target)
        target_linear = _srgb_to_linear(target[:, :, :3])
        patch_linear = _srgb_to_linear(patch[:, :, :3])
        result[:, :, :3] = _linear_to_srgb(
            target_linear * (1.0 - weights) + patch_linear * weights
        )
        if target.shape[2] > 3:
            result[:, :, 3:] = _blend_uint8_channels(
                target[:, :, 3:], patch[:, :, 3:], weights
            )
        return result
    return _blend_uint8_channels(target, patch, weights)


def _blend_normal_pixels(
    target: np.ndarray, patch: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    target_vectors = target[:, :, :3].astype(np.float32) / 127.5 - 1.0
    patch_vectors = patch[:, :, :3].astype(np.float32) / 127.5 - 1.0
    vectors = target_vectors * (1.0 - weights) + patch_vectors * weights
    lengths = np.linalg.norm(vectors, axis=2, keepdims=True)
    fallback = np.where(weights >= 0.5, patch_vectors, target_vectors)
    invalid = lengths[:, :, 0] <= _MIN_NORMAL_LENGTH
    vectors = vectors / np.maximum(lengths, _MIN_NORMAL_LENGTH)
    if np.any(invalid):
        fallback_lengths = np.linalg.norm(fallback, axis=2, keepdims=True)
        fallback = fallback / np.maximum(fallback_lengths, _MIN_NORMAL_LENGTH)
        fallback[fallback_lengths[:, :, 0] <= _MIN_NORMAL_LENGTH] = (0, 0, 1)
        vectors[invalid] = fallback[invalid]
    result = np.empty_like(target)
    result[:, :, :3] = np.clip(np.rint((vectors * 0.5 + 0.5) * 255.0), 0, 255).astype(
        np.uint8
    )
    if target.shape[2] > 3:
        result[:, :, 3:] = _blend_uint8_channels(
            target[:, :, 3:], patch[:, :, 3:], weights
        )
    return result


def _blend_uint8_channels(
    target: np.ndarray, patch: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    return np.clip(
        np.rint(
            target.astype(np.float32) * (1.0 - weights)
            + patch.astype(np.float32) * weights
        ),
        0,
        255,
    ).astype(np.uint8)


# ### Seam diagnostics ###
def _aggregate_tiling_seam_score(texture_maps: Mapping[str, np.ndarray]) -> float:
    scores = [
        texture_tiling_seam_score(source, map_type=map_type)
        for map_type, source in texture_maps.items()
    ]
    return float(np.mean(scores)) if scores else 0.0


def _texture_tiling_seam_severity(source: np.ndarray, map_type: str) -> float:
    features = _map_features(source, map_type)
    axis_severities: list[float] = []
    if source.shape[1] > 1:
        border_score = float(np.mean(np.abs(features[:, 0] - features[:, -1])))
        internal_scores = np.mean(np.abs(np.diff(features, axis=1)), axis=(0, 2))
        axis_severities.append(_excess_border_score(border_score, internal_scores))
    if source.shape[0] > 1:
        border_score = float(np.mean(np.abs(features[0] - features[-1])))
        internal_scores = np.mean(np.abs(np.diff(features, axis=0)), axis=(1, 2))
        axis_severities.append(_excess_border_score(border_score, internal_scores))
    return float(np.mean(axis_severities)) if axis_severities else 0.0


def _excess_border_score(border_score: float, internal_scores: np.ndarray) -> float:
    if internal_scores.size == 0:
        return max(border_score, 0.0)
    allowed_transition = float(np.median(internal_scores)) * _INTERNAL_SEAM_ALLOWANCE
    return max(0.0, border_score - allowed_transition)


def _feature_border_scores(features: np.ndarray) -> list[float]:
    scores: list[float] = []
    if features.shape[1] > 1:
        scores.append(float(np.mean(np.abs(features[:, 0] - features[:, -1]))))
    if features.shape[0] > 1:
        scores.append(float(np.mean(np.abs(features[0] - features[-1]))))
    return scores


# ### Input validation ###
def _normalize_texture_maps(
    texture_maps: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if not isinstance(texture_maps, Mapping):
        raise TypeError("Texture tiling repair requires a map-type mapping.")
    if not texture_maps:
        raise ValueError("Texture tiling repair requires at least one map.")
    normalized: dict[str, np.ndarray] = {}
    expected_shape: tuple[int, int] | None = None
    for raw_map_type, raw_source in texture_maps.items():
        map_type = _normalize_map_type(raw_map_type)
        if map_type in normalized:
            raise ValueError("Texture tiling repair contains duplicate map types.")
        source = _validate_texture_array(raw_source, f"{map_type} texture")
        if expected_shape is None:
            expected_shape = source.shape[:2]
        elif source.shape[:2] != expected_shape:
            raise ValueError("Aligned texture maps must have identical dimensions.")
        if map_type == PBR_MAP_NORMAL and source.shape[2] < 3:
            raise ValueError("A tangent normal map requires at least three channels.")
        normalized[map_type] = source
    return normalized


def _normalize_map_type(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("A texture map type must be non-empty text.")
    return value.strip().lower()


def _validate_texture_array(source: np.ndarray, label: str) -> np.ndarray:
    if not isinstance(source, np.ndarray):
        raise TypeError(f"{label} must be a NumPy array.")
    if source.dtype != np.uint8:
        raise ValueError(f"{label} must use uint8 pixels.")
    if source.ndim != 3 or source.shape[2] not in {1, 2, 3, 4}:
        raise ValueError(f"{label} must have one to four channels.")
    if source.shape[0] <= 0 or source.shape[1] <= 0:
        raise ValueError(f"{label} cannot be empty.")
    return np.ascontiguousarray(source, dtype=np.uint8)


# ### Map feature helpers ###
def _map_features(source: np.ndarray, map_type: str) -> np.ndarray:
    if map_type == PBR_MAP_NORMAL:
        vectors = source[:, :, :3].astype(np.float32) / 127.5 - 1.0
        lengths = np.linalg.norm(vectors, axis=2, keepdims=True)
        vectors = vectors / np.maximum(lengths, _MIN_NORMAL_LENGTH)
        vectors[lengths[:, :, 0] <= _MIN_NORMAL_LENGTH] = (0.0, 0.0, 1.0)
        return np.ascontiguousarray(vectors * 0.5, dtype=np.float32)
    if map_type in _SCALAR_MAP_TYPES or map_type == PBR_MAP_METALLIC:
        return np.ascontiguousarray(
            source[:, :, :1].astype(np.float32) / 255.0, dtype=np.float32
        )
    return _reference_features(source)


def _reference_features(source: np.ndarray) -> np.ndarray:
    if source.shape[2] >= 3:
        rgb = _srgb_to_linear(source[:, :, :3])
    else:
        gray = source[:, :, 0].astype(np.float32) / 255.0
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
    luminance = rgb[:, :, 0] * 0.2126 + rgb[:, :, 1] * 0.7152 + rgb[:, :, 2] * 0.0722
    chroma_red = rgb[:, :, 0] - luminance
    chroma_blue = rgb[:, :, 2] - luminance
    return np.ascontiguousarray(
        np.stack((luminance, chroma_red * 0.5, chroma_blue * 0.5), axis=2),
        dtype=np.float32,
    )


def _srgb_to_linear(source: np.ndarray) -> np.ndarray:
    values = source.astype(np.float32) / 255.0
    return np.where(
        values <= 0.04045,
        values / 12.92,
        np.power((values + 0.055) / 1.055, 2.4),
    ).astype(np.float32)


def _linear_to_srgb(source: np.ndarray) -> np.ndarray:
    values = np.clip(source, 0.0, 1.0)
    encoded = np.where(
        values <= 0.0031308,
        values * 12.92,
        1.055 * np.power(values, 1.0 / 2.4) - 0.055,
    )
    return np.clip(np.rint(encoded * 255.0), 0, 255).astype(np.uint8)


# ### Public exports ###
__all__ = [
    "EdgeCompatibleVariantResult",
    "TilingRepairOperation",
    "TilingRepairPlan",
    "TilingRepairResult",
    "apply_tiling_repair_plan",
    "build_tiling_repair_plan",
    "create_edge_compatible_variants",
    "repair_texture_tiling",
    "texture_tiling_seam_score",
]
