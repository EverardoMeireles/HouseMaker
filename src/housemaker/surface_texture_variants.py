# ### Imports ###
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from housemaker.surface_texture_state import (
    DEFAULT_SURFACE_TEXTURE_RESOLUTION,
    SURFACE_TEXTURE_RESOLUTIONS,
)


# ### Constants ###
CANONICAL_SURFACE_TEXTURE_RESOLUTION = 2048
MAX_SURFACE_TEXTURE_SOURCE_BYTES = 64 * 1024 * 1024
MAX_SURFACE_TEXTURE_SOURCE_DIMENSION = 16_384
MAX_SURFACE_TEXTURE_SOURCE_PIXELS = 64_000_000
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


# ### Variant data model ###
@dataclass(frozen=True)
class SurfaceTextureVariants:
    """Owned PNG payloads for every supported square texture resolution."""

    texture_png_by_resolution: dict[int, bytes]

    def __post_init__(self) -> None:
        raw_variants = self.texture_png_by_resolution
        if not isinstance(raw_variants, dict):
            raise TypeError("Surface texture variants must contain a mapping.")
        if set(raw_variants) != set(SURFACE_TEXTURE_RESOLUTIONS):
            raise ValueError(
                "Surface texture variants require 512, 1024 and 2048 PNGs."
            )
        normalized: dict[int, bytes] = {}
        for resolution in SURFACE_TEXTURE_RESOLUTIONS:
            payload = raw_variants[resolution]
            if not isinstance(payload, bytes | bytearray | memoryview):
                raise TypeError("A surface texture variant must be PNG bytes.")
            normalized[resolution] = bytes(payload)
        object.__setattr__(self, "texture_png_by_resolution", normalized)


# ### Public variant generation ###
def build_surface_texture_variants(texture_png: bytes) -> SurfaceTextureVariants:
    """Normalize one PNG to 2048 and directly derive its smaller variants."""

    source_rgba = _decode_png_rgba(texture_png)
    canonical_rgba = _normalize_to_canonical_resolution(source_rgba)
    variants_rgba = {
        CANONICAL_SURFACE_TEXTURE_RESOLUTION: canonical_rgba,
        1024: _resize_rgba(canonical_rgba, 1024, cv2.INTER_AREA),
        512: _resize_rgba(canonical_rgba, 512, cv2.INTER_AREA),
    }
    return SurfaceTextureVariants(
        texture_png_by_resolution={
            resolution: _encode_rgba_png(variants_rgba[resolution])
            for resolution in SURFACE_TEXTURE_RESOLUTIONS
        }
    )


# ### PNG helpers ###
def _decode_png_rgba(texture_png: bytes) -> np.ndarray:
    if not isinstance(texture_png, bytes | bytearray | memoryview):
        raise TypeError("A generated surface texture must contain PNG bytes.")
    payload = bytes(texture_png)
    if not payload.startswith(PNG_SIGNATURE):
        raise ValueError("A generated surface texture must be a PNG image.")
    if len(payload) > MAX_SURFACE_TEXTURE_SOURCE_BYTES:
        raise ValueError("The generated surface texture PNG is too large.")
    try:
        with Image.open(BytesIO(payload)) as image:
            if str(image.format or "").upper() != "PNG":
                raise ValueError(
                    "A generated surface texture must be a PNG image."
                )
            if int(getattr(image, "n_frames", 1)) != 1:
                raise ValueError(
                    "A generated surface texture must be a static PNG image."
                )
            width, height = image.size
            _validate_source_dimensions(width, height)
            image.load()
            rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    except ValueError:
        raise
    except (OSError, SyntaxError) as error:
        raise ValueError(
            "The generated surface texture is not a valid PNG image."
        ) from error
    return np.ascontiguousarray(rgba)


def _validate_source_dimensions(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("A generated surface texture cannot be empty.")
    if (
        width > MAX_SURFACE_TEXTURE_SOURCE_DIMENSION
        or height > MAX_SURFACE_TEXTURE_SOURCE_DIMENSION
        or width * height > MAX_SURFACE_TEXTURE_SOURCE_PIXELS
    ):
        raise ValueError("The generated surface texture dimensions are too large.")


def _normalize_to_canonical_resolution(source_rgba: np.ndarray) -> np.ndarray:
    height, width = source_rgba.shape[:2]
    canonical_size = (
        CANONICAL_SURFACE_TEXTURE_RESOLUTION,
        CANONICAL_SURFACE_TEXTURE_RESOLUTION,
    )
    if (width, height) == canonical_size:
        return source_rgba.copy()
    interpolation = (
        cv2.INTER_AREA
        if width >= CANONICAL_SURFACE_TEXTURE_RESOLUTION
        and height >= CANONICAL_SURFACE_TEXTURE_RESOLUTION
        else cv2.INTER_LANCZOS4
    )
    return _resize_rgba(
        source_rgba,
        CANONICAL_SURFACE_TEXTURE_RESOLUTION,
        interpolation,
    )


def _resize_rgba(
    source_rgba: np.ndarray,
    resolution: int,
    interpolation: int,
) -> np.ndarray:
    return np.ascontiguousarray(
        cv2.resize(
            source_rgba,
            (int(resolution), int(resolution)),
            interpolation=interpolation,
        ),
        dtype=np.uint8,
    )


def _encode_rgba_png(source_rgba: np.ndarray) -> bytes:
    bgra = cv2.cvtColor(source_rgba, cv2.COLOR_RGBA2BGRA)
    did_encode, encoded = cv2.imencode(".png", bgra)
    if not did_encode:
        raise ValueError("A generated surface texture variant could not be encoded.")
    return bytes(encoded)


# ### Public exports ###
__all__ = [
    "CANONICAL_SURFACE_TEXTURE_RESOLUTION",
    "DEFAULT_SURFACE_TEXTURE_RESOLUTION",
    "SURFACE_TEXTURE_RESOLUTIONS",
    "SurfaceTextureVariants",
    "build_surface_texture_variants",
]
