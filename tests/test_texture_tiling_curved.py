# ### Imports ###
from __future__ import annotations

import cv2
import numpy as np
import pytest

from housemaker.pbr_maps import ATLAS_MAP_BASE_COLOR, PBR_MAP_METALLIC, PBR_MAP_NORMAL
from housemaker.texture_tiling_curved import (
    _plan_paths,
    _reconcile_tile_tones,
    _rotated_variants,
    build_curved_variant_sheet,
)


# ### Fixtures ###
def _pattern(size: int) -> np.ndarray:
    yy, xx = np.indices((size, size))
    gray = (70 + xx * 2 + yy + ((xx // 5 + yy // 7) % 2) * 20).astype(np.uint8)
    return np.stack((gray, gray // 2, 255 - gray), axis=2)


# ### Geometry and color tests ###
def test_curved_sheet_keeps_full_rotated_tile_interiors() -> None:
    tile, overlap = 64, 8
    rgb = _pattern(tile + overlap * 2)
    yy, xx = np.indices(rgb.shape[:2])
    source = np.concatenate(
        (rgb, ((xx * 3 + yy * 5) % 256).astype(np.uint8)[:, :, None]), axis=2
    )
    result = build_curved_variant_sheet(
        {ATLAS_MAP_BASE_COLOR: source},
        (0, 1, 2, 3),
        tile_height=tile,
        tile_width=tile,
        overlap=overlap,
    )[ATLAS_MAP_BASE_COLOR]
    assert result.shape == (tile * 2, tile * 2, 4)
    for index, turns in enumerate((0, 1, 2, 3)):
        row, column = divmod(index, 2)
        expected = np.rot90(source, turns)[
            overlap + 16 : overlap + 48, overlap + 16 : overlap + 48
        ]
        actual = result[
            row * tile + 16 : row * tile + 48, column * tile + 16 : column * tile + 48
        ]
        # Broad edge-tone matching may change RGB, but never the actual
        # interior texel positions or coverage channel.
        assert np.array_equal(actual[:, :, 3], expected[:, :, 3])


def test_curved_paths_bend_and_join_periodically() -> None:
    tile, overlap = 128, 12
    source = np.random.default_rng(15).integers(
        80, 180, (tile + 2 * overlap, tile + 2 * overlap, 3), dtype=np.uint8
    )
    variants = _reconcile_tile_tones(
        _rotated_variants(source, (0, 1, 2, 3), ATLAS_MAP_BASE_COLOR),
        ATLAS_MAP_BASE_COLOR,
        tile,
        tile,
        overlap,
    )
    vertical, horizontal = _plan_paths(variants, tile, tile, overlap, None)
    assert np.ptp(vertical) > 3.0
    assert np.ptp(horizontal) > 3.0
    for paths in (vertical, horizontal):
        assert np.all(paths[:, [0, tile - 1, tile, 2 * tile - 1]] == 0.0)


def test_curved_strip_rejoins_untouched_tiles_at_overlap_limits() -> None:
    tile, overlap = 64, 8
    rgb = _pattern(tile + overlap * 2)
    yy, xx = np.indices(rgb.shape[:2])
    alpha = ((xx * 3 + yy * 5) % 256).astype(np.uint8)
    source = np.dstack((rgb, alpha))
    output = build_curved_variant_sheet(
        {ATLAS_MAP_BASE_COLOR: source},
        (0, 1, 2, 3),
        tile_height=tile,
        tile_width=tile,
        overlap=overlap,
    )[ATLAS_MAP_BASE_COLOR]
    left, right, bottom = (np.rot90(source, turn) for turn in (0, 1, 2))
    y = tile // 2
    assert output[y, tile - overlap, 3] == left[overlap + y, tile, 3]
    assert output[y, tile + overlap - 1, 3] == right[overlap + y, 2 * overlap - 1, 3]
    x = tile // 2
    assert output[tile - overlap, x, 3] == left[tile, overlap + x, 3]
    assert output[tile + overlap - 1, x, 3] == bottom[2 * overlap - 1, overlap + x, 3]


def test_curved_tone_matching_reduces_mip_level_cross_on_noisy_ramp() -> None:
    tile, overlap = 128, 12
    yy, xx = np.indices((tile + 2 * overlap, tile + 2 * overlap))
    grain = np.random.default_rng(41).integers(-6, 7, xx.shape)
    gray = np.clip(110 + 0.27 * xx + 0.15 * yy + grain, 0, 255).astype(np.uint8)
    source = np.repeat(gray[:, :, None], 3, axis=2)
    variants = [
        np.rot90(source, turns)[overlap : overlap + tile, overlap : overlap + tile]
        for turns in (0, 1, 2, 3)
    ]
    raw = np.block([[variants[0], variants[1]], [variants[2], variants[3]]])
    repaired = build_curved_variant_sheet(
        {ATLAS_MAP_BASE_COLOR: source},
        (0, 1, 2, 3),
        tile_height=tile,
        tile_width=tile,
        overlap=overlap,
    )[ATLAS_MAP_BASE_COLOR]

    def worst_mip_seam(image: np.ndarray) -> float:
        mip = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(
            np.float32
        )
        return max(
            float(np.abs(mip[:, 0] - mip[:, -1]).mean()),
            float(np.abs(mip[:, 16] - mip[:, 15]).mean()),
            float(np.abs(mip[0] - mip[-1]).mean()),
            float(np.abs(mip[16] - mip[15]).mean()),
        )

    assert worst_mip_seam(repaired) < worst_mip_seam(raw)


def test_curved_sheet_uses_same_shared_seam_for_scalar_map() -> None:
    tile, overlap = 48, 6
    source = _pattern(tile + overlap * 2)
    gray = source[:, :, :1].copy()
    maps = build_curved_variant_sheet(
        {ATLAS_MAP_BASE_COLOR: source, "roughness": gray},
        (0, 1, 2, 3),
        tile_height=tile,
        tile_width=tile,
        overlap=overlap,
    )
    # A single-channel map can use the same geometric seam without RGB input.
    assert maps["roughness"].shape == (tile * 2, tile * 2, 1)
    assert maps["roughness"].dtype == np.uint8
    assert np.isfinite(maps["roughness"]).all()


# ### PBR and control tests ###
def test_curved_sheet_preserves_metallic_values_and_normal_lengths() -> None:
    tile, overlap = 48, 6
    size = tile + 2 * overlap
    metallic = np.zeros((size, size, 1), dtype=np.uint8)
    metallic[:, size // 2 :] = 255
    normals = np.empty((size, size, 3), dtype=np.uint8)
    normals[:] = (170, 120, 245)
    result = build_curved_variant_sheet(
        {PBR_MAP_METALLIC: metallic, PBR_MAP_NORMAL: normals},
        (0, 1, 2, 3),
        tile_height=tile,
        tile_width=tile,
        overlap=overlap,
    )
    assert set(np.unique(result[PBR_MAP_METALLIC])) <= {0, 255}
    vectors = result[PBR_MAP_NORMAL].astype(np.float32) / 127.5 - 1.0
    lengths = np.linalg.norm(vectors, axis=2)
    assert np.max(np.abs(lengths - 1.0)) < 0.02


def test_curved_sheet_checks_cancellation() -> None:
    with pytest.raises(InterruptedError, match="cancelled"):
        build_curved_variant_sheet(
            {ATLAS_MAP_BASE_COLOR: _pattern(64)},
            (0, 1, 2, 3),
            tile_height=48,
            tile_width=48,
            overlap=8,
            cancellation_check=lambda: True,
        )


def test_curved_sheet_rejects_wrong_overscan_shape() -> None:
    with pytest.raises(ValueError, match="overscan"):
        build_curved_variant_sheet(
            {ATLAS_MAP_BASE_COLOR: _pattern(64)},
            (0, 1, 2, 3),
            tile_height=64,
            tile_width=64,
            overlap=8,
        )
