# ### Imports ###
import numpy as np
import pytest

from housemaker.texture_tiling_scoring import (
    TILING_FIX_SCORE_THRESHOLD,
    score_repeated_texture_tile,
    score_variant_sheet,
    texture_needs_tiling_fix,
)


# ### Synthetic sheet helpers ###
def _periodic_tile(size: int = 128) -> np.ndarray:
    axis = np.arange(size, dtype=np.float32) * (2.0 * np.pi / size)
    values = 125.0 + 18.0 * np.sin(axis[:, None]) + 15.0 * np.cos(axis[None, :])
    tile = np.clip(np.rint(values), 0, 255).astype(np.uint8)
    return np.repeat(tile[:, :, None], 3, axis=2)


def _periodic_sheet(size: int = 128) -> np.ndarray:
    return np.tile(_periodic_tile(size), (2, 2, 1))


# ### Seam visibility tests ###
def test_obvious_center_cross_scores_worse_than_continuous_sheet() -> None:
    clean = _periodic_sheet()
    crossed = clean.copy()
    crossed[:, 127:130, :3] = 230
    crossed[127:130, :, :3] = 230

    assert score_variant_sheet(crossed, 128, 128) > (
        score_variant_sheet(clean, 128, 128) + 0.4
    )


def test_one_bad_corner_is_not_hidden_by_three_good_corners() -> None:
    clean = _periodic_sheet()
    one_bad_corner = clean.copy()
    one_bad_corner[128:141, 128:141] = 230

    assert score_variant_sheet(one_bad_corner, 128, 128) > (
        score_variant_sheet(clean, 128, 128) + 0.15
    )


def test_bad_corner_on_wrapped_sheet_border_is_detected() -> None:
    clean = _periodic_sheet()
    wrapped_corner = clean.copy()
    wrapped_corner[:14, -14:] = 230

    assert score_variant_sheet(wrapped_corner, 128, 128) > (
        score_variant_sheet(clean, 128, 128) + 0.15
    )


def test_coherent_noise_is_not_scored_like_a_high_contrast_seam() -> None:
    generator = np.random.default_rng(17)
    tile = generator.integers(105, 151, size=(128, 128), dtype=np.uint8)
    noisy = np.repeat(np.tile(tile, (2, 2))[:, :, None], 3, axis=2)
    with_seam = noisy.copy()
    with_seam[:, 127:130] = 230

    assert score_variant_sheet(noisy, 128, 128) < (
        score_variant_sheet(with_seam, 128, 128) * 0.5
    )


def test_broad_mip_visible_seam_is_detected_when_immediate_edge_matches() -> None:
    clean = np.full((256, 256, 3), 128, dtype=np.uint8)
    broad = clean.copy()
    broad[:, 112:127] = 180
    broad[:, 129:144] = 180
    assert np.array_equal(broad[:, 127], broad[:, 128])

    assert score_variant_sheet(broad, 128, 128) > (
        score_variant_sheet(clean, 128, 128) + 0.2
    )


# ### Repeated-tile diagnostic tests ###
def test_clean_periodic_and_noise_tiles_do_not_request_a_tiling_fix() -> None:
    generator = np.random.default_rng(23)
    noise = generator.integers(105, 151, size=(128, 128, 3), dtype=np.uint8)

    assert TILING_FIX_SCORE_THRESHOLD == 1.0
    assert score_repeated_texture_tile(_periodic_tile()) < TILING_FIX_SCORE_THRESHOLD
    assert score_repeated_texture_tile(noise) < TILING_FIX_SCORE_THRESHOLD
    assert not texture_needs_tiling_fix(_periodic_tile())
    assert not texture_needs_tiling_fix(noise)


def test_gradient_and_broad_repeat_seams_request_a_tiling_fix() -> None:
    size = 128
    gradient_channel = np.broadcast_to(
        np.linspace(0, 255, size, dtype=np.uint8),
        (size, size),
    )
    gradient = np.repeat(gradient_channel[:, :, None], 3, axis=2)
    broad_seam = np.full((size, size, 3), 128, dtype=np.uint8)
    broad_seam[:, 8:16] = 200
    broad_seam[:, -16:-8] = 200
    assert np.array_equal(broad_seam[:, 0], broad_seam[:, -1])

    assert score_repeated_texture_tile(gradient) >= TILING_FIX_SCORE_THRESHOLD
    assert score_repeated_texture_tile(broad_seam) >= TILING_FIX_SCORE_THRESHOLD
    assert texture_needs_tiling_fix(gradient)
    assert texture_needs_tiling_fix(broad_seam)


# ### Validation tests ###
def test_invalid_sheet_geometry_is_rejected() -> None:
    with pytest.raises(ValueError, match="two by two"):
        score_variant_sheet(np.zeros((128, 256, 3), dtype=np.uint8), 128, 128)


def test_repeated_tile_diagnostic_requires_uint8_pixels() -> None:
    with pytest.raises(ValueError, match="uint8"):
        texture_needs_tiling_fix(np.zeros((8, 8, 3), dtype=np.float32))
