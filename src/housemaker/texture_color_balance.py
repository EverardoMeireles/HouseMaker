# ### Imports ###
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral

import numpy as np

# ### Public constants ###
MIN_COLOR_BALANCE_VALUE = -100
MAX_COLOR_BALANCE_VALUE = 100
MAXIMUM_COLOR_BALANCE_CHANNEL_SHIFT = 0.25


# ### Internal constants ###
_MIDDLE_LUMINANCE = 0.5
_LUMINANCE_WEIGHTS = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float64)
_DELTA_EPSILON = np.finfo(np.float64).eps


# ### Public settings models ###
class ColorBalanceTone(str, Enum):
    """One tonal range exposed by Photoshop-style color balance controls."""

    SHADOWS = "shadows"
    MIDTONES = "midtones"
    HIGHLIGHTS = "highlights"


@dataclass(frozen=True)
class ColorBalanceAdjustment:
    """Three signed opponent-color controls for one tonal range.

    Positive values move toward red, green, or blue respectively. Negative
    values move toward the named opponent: cyan, magenta, or yellow.
    """

    cyan_red: int = 0
    magenta_green: int = 0
    yellow_blue: int = 0

    def __post_init__(self) -> None:
        for field_name in ("cyan_red", "magenta_green", "yellow_blue"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError("Color balance controls must be whole numbers.")
            normalized_value = int(value)
            if not MIN_COLOR_BALANCE_VALUE <= normalized_value <= MAX_COLOR_BALANCE_VALUE:
                raise ValueError(
                    "Color balance controls must be between "
                    f"{MIN_COLOR_BALANCE_VALUE} and {MAX_COLOR_BALANCE_VALUE}."
                )
            object.__setattr__(self, field_name, normalized_value)

    @property
    def is_neutral(self) -> bool:
        """Return whether this tonal range leaves every channel unchanged."""

        return not (self.cyan_red or self.magenta_green or self.yellow_blue)


@dataclass(frozen=True)
class TextureColorBalanceSettings:
    """Immutable controls for one deterministic base-color adjustment."""

    shadows: ColorBalanceAdjustment = field(default_factory=ColorBalanceAdjustment)
    midtones: ColorBalanceAdjustment = field(default_factory=ColorBalanceAdjustment)
    highlights: ColorBalanceAdjustment = field(default_factory=ColorBalanceAdjustment)
    preserve_luminosity: bool = True

    def __post_init__(self) -> None:
        for field_name in ("shadows", "midtones", "highlights"):
            if not isinstance(getattr(self, field_name), ColorBalanceAdjustment):
                raise TypeError(
                    "Color balance tonal settings must be ColorBalanceAdjustment "
                    "values."
                )
        if not isinstance(self.preserve_luminosity, bool):
            raise TypeError("Preserve luminosity must be a boolean.")

    @property
    def is_neutral(self) -> bool:
        """Return whether applying these settings is an exact pixel no-op."""

        return bool(
            self.shadows.is_neutral
            and self.midtones.is_neutral
            and self.highlights.is_neutral
        )

    def adjustment_for_tone(
        self,
        tone: ColorBalanceTone,
    ) -> ColorBalanceAdjustment:
        """Return the immutable channel controls for one tonal range."""

        if not isinstance(tone, ColorBalanceTone):
            try:
                tone = ColorBalanceTone(tone)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Unknown color balance tone: {tone!r}.") from error
        return getattr(self, tone.value)


# ### Public color-balance operation ###
def apply_texture_color_balance_rgba(
    rgba: np.ndarray,
    settings: TextureColorBalanceSettings,
) -> np.ndarray:
    """Return color-balanced RGBA pixels without changing the input array.

    The operation intentionally works only on gamma-encoded base-color RGB.
    Alpha is copied byte-for-byte, and exact black pixels stay black so unused
    symmetric-texture regions and padding cannot become tinted.
    """

    source = _validate_rgba(rgba)
    if not isinstance(settings, TextureColorBalanceSettings):
        raise TypeError("Texture color balance settings are required.")
    if settings.is_neutral:
        return np.array(source, dtype=np.uint8, copy=True, order="C")

    source_rgb = np.asarray(source[..., :3], dtype=np.float64) / 255.0
    luminance = np.sum(source_rgb * _LUMINANCE_WEIGHTS, axis=2)
    shadow_weight, midtone_weight, highlight_weight = _build_tone_weights(
        luminance
    )
    channel_delta = _build_channel_delta(
        settings,
        shadow_weight,
        midtone_weight,
        highlight_weight,
    )
    if settings.preserve_luminosity:
        channel_delta = _remove_luminance_delta(channel_delta)
        channel_delta = _limit_delta_to_rgb_gamut(source_rgb, channel_delta)

    balanced_rgb = np.clip(source_rgb + channel_delta, 0.0, 1.0)
    black_padding = np.all(source[..., :3] == 0, axis=2)
    balanced_rgb[black_padding] = 0.0

    output = np.empty(source.shape, dtype=np.uint8)
    output[..., :3] = np.floor((balanced_rgb * 255.0) + 0.5).astype(np.uint8)
    output[..., 3] = source[..., 3]
    return output


# ### Tone weighting helpers ###
def _build_tone_weights(
    luminance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build smooth tonal weights that add to exactly one per pixel."""

    shadow_position = np.clip(luminance / _MIDDLE_LUMINANCE, 0.0, 1.0)
    highlight_position = np.clip(
        (luminance - _MIDDLE_LUMINANCE) / (1.0 - _MIDDLE_LUMINANCE),
        0.0,
        1.0,
    )
    shadow_weight = 1.0 - _smoothstep(shadow_position)
    highlight_weight = _smoothstep(highlight_position)
    midtone_weight = 1.0 - shadow_weight - highlight_weight
    return shadow_weight, midtone_weight, highlight_weight


def _smoothstep(values: np.ndarray) -> np.ndarray:
    """Return a continuous cubic transition for normalized values."""

    return values * values * (3.0 - (2.0 * values))


def _build_channel_delta(
    settings: TextureColorBalanceSettings,
    shadow_weight: np.ndarray,
    midtone_weight: np.ndarray,
    highlight_weight: np.ndarray,
) -> np.ndarray:
    """Combine tonal controls into one signed RGB delta per pixel."""

    tone_weights = (shadow_weight, midtone_weight, highlight_weight)
    adjustments = (settings.shadows, settings.midtones, settings.highlights)
    channel_controls = np.asarray(
        [
            (
                adjustment.cyan_red,
                adjustment.magenta_green,
                adjustment.yellow_blue,
            )
            for adjustment in adjustments
        ],
        dtype=np.float64,
    )
    stacked_weights = np.stack(tone_weights, axis=2)
    normalized_delta = np.einsum(
        "hwt,tc->hwc",
        stacked_weights,
        channel_controls,
        optimize=False,
    )
    return normalized_delta * (
        MAXIMUM_COLOR_BALANCE_CHANNEL_SHIFT / MAX_COLOR_BALANCE_VALUE
    )


# ### Luminosity and gamut helpers ###
def _remove_luminance_delta(channel_delta: np.ndarray) -> np.ndarray:
    """Project adjustments onto the zero-luminance opponent-color plane."""

    luminance_delta = np.sum(channel_delta * _LUMINANCE_WEIGHTS, axis=2)
    return channel_delta - luminance_delta[..., np.newaxis]


def _limit_delta_to_rgb_gamut(
    source_rgb: np.ndarray,
    channel_delta: np.ndarray,
) -> np.ndarray:
    """Uniformly reduce chroma only where a channel would leave RGB gamut."""

    channel_limits = np.ones_like(channel_delta)
    positive = channel_delta > _DELTA_EPSILON
    negative = channel_delta < -_DELTA_EPSILON
    np.divide(
        1.0 - source_rgb,
        channel_delta,
        out=channel_limits,
        where=positive,
    )
    np.divide(
        -source_rgb,
        channel_delta,
        out=channel_limits,
        where=negative,
    )
    pixel_scale = np.clip(np.min(channel_limits, axis=2), 0.0, 1.0)
    return channel_delta * pixel_scale[..., np.newaxis]


# ### Validation helpers ###
def _validate_rgba(rgba: np.ndarray) -> np.ndarray:
    """Return a validated RGBA byte array without taking ownership."""

    if not isinstance(rgba, np.ndarray):
        raise TypeError("Texture color balance requires a numpy array.")
    if rgba.dtype != np.uint8:
        raise ValueError("Texture color balance requires uint8 pixels.")
    if rgba.ndim != 3 or rgba.shape[2] != 4 or not rgba.shape[0] or not rgba.shape[1]:
        raise ValueError("Texture color balance requires a non-empty H x W x 4 image.")
    return rgba


# ### Public exports ###
__all__ = [
    "MAX_COLOR_BALANCE_VALUE",
    "MAXIMUM_COLOR_BALANCE_CHANNEL_SHIFT",
    "MIN_COLOR_BALANCE_VALUE",
    "ColorBalanceAdjustment",
    "ColorBalanceTone",
    "TextureColorBalanceSettings",
    "apply_texture_color_balance_rgba",
]
