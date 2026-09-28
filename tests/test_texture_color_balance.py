# ### Imports ###
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from housemaker.texture_color_balance import (
    ColorBalanceAdjustment,
    ColorBalanceTone,
    TextureColorBalanceSettings,
    apply_texture_color_balance_rgba,
)


# ### Test helpers ###
def _settings_for_tone(
    tone: ColorBalanceTone,
    adjustment: ColorBalanceAdjustment,
    *,
    preserve_luminosity: bool = True,
) -> TextureColorBalanceSettings:
    values = {
        ColorBalanceTone.SHADOWS: ColorBalanceAdjustment(),
        ColorBalanceTone.MIDTONES: ColorBalanceAdjustment(),
        ColorBalanceTone.HIGHLIGHTS: ColorBalanceAdjustment(),
    }
    values[tone] = adjustment
    return TextureColorBalanceSettings(
        shadows=values[ColorBalanceTone.SHADOWS],
        midtones=values[ColorBalanceTone.MIDTONES],
        highlights=values[ColorBalanceTone.HIGHLIGHTS],
        preserve_luminosity=preserve_luminosity,
    )


def _rgb_delta_magnitude(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    return np.sum(
        np.abs(
            after[..., :3].astype(np.int16) - before[..., :3].astype(np.int16)
        ),
        axis=2,
    )


# ### Settings model tests ###
def test_settings_are_frozen_typed_and_neutral_by_default() -> None:
    settings = TextureColorBalanceSettings()

    assert settings.is_neutral
    assert settings.adjustment_for_tone(ColorBalanceTone.SHADOWS).is_neutral
    assert settings.adjustment_for_tone("midtones") == settings.midtones
    with pytest.raises(FrozenInstanceError):
        settings.preserve_luminosity = False  # type: ignore[misc]


@pytest.mark.parametrize("value", (-101, 101))
def test_adjustment_rejects_out_of_range_controls(value: int) -> None:
    with pytest.raises(ValueError, match="between -100 and 100"):
        ColorBalanceAdjustment(cyan_red=value)


@pytest.mark.parametrize("value", (True, 1.5, "1"))
def test_adjustment_rejects_non_integer_controls(value: object) -> None:
    with pytest.raises(TypeError, match="whole numbers"):
        ColorBalanceAdjustment(magenta_green=value)  # type: ignore[arg-type]


def test_settings_reject_invalid_nested_values_and_preserve_flag() -> None:
    with pytest.raises(TypeError, match="ColorBalanceAdjustment"):
        TextureColorBalanceSettings(shadows=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="boolean"):
        TextureColorBalanceSettings(preserve_luminosity=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown color balance tone"):
        TextureColorBalanceSettings().adjustment_for_tone("whites")  # type: ignore[arg-type]


# ### Pixel integrity tests ###
def test_neutral_settings_are_an_exact_non_mutating_copy() -> None:
    source = np.asarray(
        [
            [[0, 0, 0, 0], [25, 90, 180, 1]],
            [[210, 50, 110, 127], [255, 255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    original = source.copy()

    result = apply_texture_color_balance_rgba(
        source,
        TextureColorBalanceSettings(),
    )

    assert np.array_equal(result, original)
    assert np.array_equal(source, original)
    assert result is not source
    assert result.flags.c_contiguous


def test_alpha_is_preserved_bit_exact_and_input_is_not_mutated() -> None:
    source = np.asarray(
        [[[96, 112, 128, alpha] for alpha in (0, 1, 127, 255)]],
        dtype=np.uint8,
    )
    original = source.copy()
    settings = _settings_for_tone(
        ColorBalanceTone.MIDTONES,
        ColorBalanceAdjustment(cyan_red=70, magenta_green=-45, yellow_blue=20),
    )

    result = apply_texture_color_balance_rgba(source, settings)

    assert np.array_equal(result[..., 3], original[..., 3])
    assert np.array_equal(source, original)
    assert not np.array_equal(result[..., :3], original[..., :3])


@pytest.mark.parametrize("preserve_luminosity", (False, True))
def test_exact_black_padding_always_stays_black(
    preserve_luminosity: bool,
) -> None:
    source = np.asarray(
        [[[0, 0, 0, 0], [0, 0, 0, 255], [1, 1, 1, 255]]],
        dtype=np.uint8,
    )
    settings = _settings_for_tone(
        ColorBalanceTone.SHADOWS,
        ColorBalanceAdjustment(cyan_red=100, magenta_green=-100, yellow_blue=100),
        preserve_luminosity=preserve_luminosity,
    )

    result = apply_texture_color_balance_rgba(source, settings)

    assert np.array_equal(result[0, :2, :3], source[0, :2, :3])


def test_repeated_application_to_same_source_is_deterministic() -> None:
    generator = np.random.default_rng(937)
    source = generator.integers(0, 256, size=(31, 29, 4), dtype=np.uint8)
    settings = TextureColorBalanceSettings(
        shadows=ColorBalanceAdjustment(-30, 20, 45),
        midtones=ColorBalanceAdjustment(60, -50, 25),
        highlights=ColorBalanceAdjustment(15, 40, -75),
    )

    first = apply_texture_color_balance_rgba(source, settings)
    second = apply_texture_color_balance_rgba(source, settings)

    assert np.array_equal(first, second)


# ### Opponent color tests ###
@pytest.mark.parametrize(
    ("adjustment", "raised_channel"),
    (
        (ColorBalanceAdjustment(cyan_red=100), 0),
        (ColorBalanceAdjustment(magenta_green=100), 1),
        (ColorBalanceAdjustment(yellow_blue=100), 2),
    ),
)
def test_positive_midtones_move_toward_the_named_rgb_channel(
    adjustment: ColorBalanceAdjustment,
    raised_channel: int,
) -> None:
    source = np.full((1, 1, 4), 128, dtype=np.uint8)
    source[..., 3] = 255
    settings = _settings_for_tone(ColorBalanceTone.MIDTONES, adjustment)

    result = apply_texture_color_balance_rgba(source, settings)

    assert result[0, 0, raised_channel] > source[0, 0, raised_channel]
    other_channels = {0, 1, 2} - {raised_channel}
    assert all(
        result[0, 0, channel] < source[0, 0, channel]
        for channel in other_channels
    )


@pytest.mark.parametrize(
    ("adjustment", "lowered_channel"),
    (
        (ColorBalanceAdjustment(cyan_red=-100), 0),
        (ColorBalanceAdjustment(magenta_green=-100), 1),
        (ColorBalanceAdjustment(yellow_blue=-100), 2),
    ),
)
def test_negative_midtones_move_toward_the_named_opponent(
    adjustment: ColorBalanceAdjustment,
    lowered_channel: int,
) -> None:
    source = np.full((1, 1, 4), 128, dtype=np.uint8)
    source[..., 3] = 255
    settings = _settings_for_tone(ColorBalanceTone.MIDTONES, adjustment)

    result = apply_texture_color_balance_rgba(source, settings)

    assert result[0, 0, lowered_channel] < source[0, 0, lowered_channel]
    other_channels = {0, 1, 2} - {lowered_channel}
    assert all(
        result[0, 0, channel] > source[0, 0, channel]
        for channel in other_channels
    )


# ### Tone and luminosity tests ###
@pytest.mark.parametrize(
    ("tone", "strongest_index"),
    (
        (ColorBalanceTone.SHADOWS, 0),
        (ColorBalanceTone.MIDTONES, 1),
        (ColorBalanceTone.HIGHLIGHTS, 2),
    ),
)
def test_tonal_controls_target_their_expected_luminance_range(
    tone: ColorBalanceTone,
    strongest_index: int,
) -> None:
    source = np.asarray(
        [[[48, 48, 48, 255], [128, 128, 128, 255], [208, 208, 208, 255]]],
        dtype=np.uint8,
    )
    settings = _settings_for_tone(
        tone,
        ColorBalanceAdjustment(cyan_red=80),
    )

    result = apply_texture_color_balance_rgba(source, settings)
    magnitudes = _rgb_delta_magnitude(source, result)[0]

    assert magnitudes[strongest_index] == np.max(magnitudes)
    assert np.count_nonzero(magnitudes == magnitudes[strongest_index]) == 1


def test_preserve_luminosity_keeps_midrange_luma_within_one_byte() -> None:
    source = np.asarray(
        [
            [
                [72, 110, 164, 255],
                [180, 125, 70, 255],
                [95, 170, 115, 255],
            ]
        ],
        dtype=np.uint8,
    )
    settings = TextureColorBalanceSettings(
        shadows=ColorBalanceAdjustment(50, -25, 35),
        midtones=ColorBalanceAdjustment(-40, 65, -30),
        highlights=ColorBalanceAdjustment(25, -50, 45),
        preserve_luminosity=True,
    )

    result = apply_texture_color_balance_rgba(source, settings)
    weights = np.asarray((0.2126, 0.7152, 0.0722))
    before_luma = np.sum(source[..., :3] * weights, axis=2)
    after_luma = np.sum(result[..., :3] * weights, axis=2)

    assert np.max(np.abs(after_luma - before_luma)) <= 1.0


# ### Input validation tests ###
def test_operation_requires_rgba_uint8_pixels_and_typed_settings() -> None:
    valid = np.zeros((2, 2, 4), dtype=np.uint8)
    with pytest.raises(TypeError, match="numpy array"):
        apply_texture_color_balance_rgba([], TextureColorBalanceSettings())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="uint8"):
        apply_texture_color_balance_rgba(
            valid.astype(np.float32),
            TextureColorBalanceSettings(),
        )
    with pytest.raises(ValueError, match="H x W x 4"):
        apply_texture_color_balance_rgba(
            valid[..., :3],
            TextureColorBalanceSettings(),
        )
    with pytest.raises(TypeError, match="settings"):
        apply_texture_color_balance_rgba(valid, object())  # type: ignore[arg-type]
