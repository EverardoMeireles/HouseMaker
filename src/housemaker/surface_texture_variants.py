# ### Imports ###
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from housemaker.pbr_maps import (
    ATLAS_MAP_BASE_COLOR,
    ATLAS_MAP_TYPES,
    PBR_MAP_NORMAL,
    normalize_pbr_map_types,
)
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
BASE_NORMAL_BLUR_SIGMA = 1.0
BASE_NORMAL_MIN_GRADIENT_RMS = 0.004
BASE_NORMAL_MIN_LUMINANCE_STD = 0.015
DERIVED_NORMAL_MAX_GRADIENT_SCALE = 12.0
DERIVED_NORMAL_TARGET_SLOPE = 0.3
FLAT_NORMAL_MAX_ANGULAR_DEVIATION = 0.035
FLAT_NORMAL_MAX_DETAILED_FRACTION = 0.05
NORMAL_CLASSIFICATION_ROWS_PER_BATCH = 256
NORMAL_GRADIENT_REFERENCE_RESOLUTION = 512.0
NORMAL_VECTOR_EPSILON = 1e-6


# ### Variant data model ###
@dataclass(frozen=True)
class SurfaceTextureVariants:
    """Owned PNG payloads for every supported square texture resolution."""

    texture_png_by_resolution: dict[int, bytes]
    map_png_by_resolution: dict[str, dict[int, bytes]] | None = None

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
        raw_map_pngs = dict(self.map_png_by_resolution or {})
        raw_map_pngs.setdefault(ATLAS_MAP_BASE_COLOR, dict(normalized))
        if any(map_type not in ATLAS_MAP_TYPES for map_type in raw_map_pngs):
            raise ValueError("Surface texture variants contain an unknown map.")
        normalized_maps: dict[str, dict[int, bytes]] = {}
        for map_type in ATLAS_MAP_TYPES:
            if map_type not in raw_map_pngs:
                continue
            resolution_map = raw_map_pngs[map_type]
            if not isinstance(resolution_map, Mapping):
                raise TypeError("A surface texture map must contain a mapping.")
            if set(resolution_map) != set(SURFACE_TEXTURE_RESOLUTIONS):
                raise ValueError(
                    f"The {map_type} map requires every texture resolution."
                )
            normalized_maps[map_type] = {}
            for resolution in SURFACE_TEXTURE_RESOLUTIONS:
                map_png = resolution_map[resolution]
                if not isinstance(map_png, bytes | bytearray | memoryview):
                    raise TypeError("A surface texture map must contain PNG bytes.")
                normalized_maps[map_type][resolution] = bytes(map_png)
        if normalized_maps[ATLAS_MAP_BASE_COLOR] != normalized:
            raise ValueError(
                "Surface base-color map variants must match their textures."
            )
        object.__setattr__(self, "texture_png_by_resolution", normalized)
        object.__setattr__(self, "map_png_by_resolution", normalized_maps)

    @property
    def available_map_types(self) -> tuple[str, ...]:
        """Return prepared texture maps in canonical display order."""

        assert self.map_png_by_resolution is not None
        return tuple(
            map_type
            for map_type in ATLAS_MAP_TYPES
            if map_type in self.map_png_by_resolution
        )


# ### Public variant generation ###
def build_surface_texture_variants(
    texture_png: bytes,
    pbr_texture_pngs: Mapping[str, bytes] | None = None,
) -> SurfaceTextureVariants:
    """Normalize aligned color/PBR PNGs and derive each smaller variant."""

    source_rgba = _decode_png_rgba(texture_png, "surface base-color texture")
    pbr_sources = _normalize_pbr_texture_pngs(pbr_texture_pngs)
    source_maps = {
        ATLAS_MAP_BASE_COLOR: source_rgba,
        **pbr_sources,
    }
    canonical_maps = {
        map_type: _normalize_to_canonical_resolution(map_type, source_map)
        for map_type, source_map in source_maps.items()
    }
    if PBR_MAP_NORMAL in canonical_maps:
        canonical_maps[PBR_MAP_NORMAL] = _repair_flat_normal_map(
            canonical_maps[ATLAS_MAP_BASE_COLOR],
            canonical_maps[PBR_MAP_NORMAL],
        )
    maps_by_resolution = {
        CANONICAL_SURFACE_TEXTURE_RESOLUTION: canonical_maps,
        1024: {
            map_type: _resize_texture_map(
                map_type,
                source_map,
                1024,
                cv2.INTER_AREA,
            )
            for map_type, source_map in canonical_maps.items()
        },
        512: {
            map_type: _resize_texture_map(
                map_type,
                source_map,
                512,
                cv2.INTER_AREA,
            )
            for map_type, source_map in canonical_maps.items()
        },
    }
    encoded_maps = {
        map_type: {
            resolution: _encode_rgba_png(maps_by_resolution[resolution][map_type])
            for resolution in SURFACE_TEXTURE_RESOLUTIONS
        }
        for map_type in source_maps
    }
    return SurfaceTextureVariants(
        texture_png_by_resolution=encoded_maps[ATLAS_MAP_BASE_COLOR],
        map_png_by_resolution=encoded_maps,
    )


# ### PNG helpers ###
def _normalize_pbr_texture_pngs(
    raw_maps: Mapping[str, bytes] | None,
) -> dict[str, np.ndarray]:
    if raw_maps is None:
        return {}
    if not isinstance(raw_maps, Mapping):
        raise TypeError("Surface PBR textures must contain a mapping.")
    normalized_types = normalize_pbr_map_types(
        tuple(raw_maps),
        label="Surface PBR textures",
    )
    if len(normalized_types) != len(raw_maps):
        raise ValueError("Surface PBR textures contain duplicate map IDs.")
    normalized_maps: dict[str, np.ndarray] = {}
    for raw_map_type, texture_png in raw_maps.items():
        map_type = str(raw_map_type).strip().lower()
        decoded = _decode_png_rgba(
            texture_png,
            f"surface {map_type} texture",
        )
        normalized_maps[map_type] = decoded
    return {
        map_type: normalized_maps[map_type]
        for map_type in normalized_types
    }


def _decode_png_rgba(texture_png: bytes, label: str) -> np.ndarray:
    if not isinstance(texture_png, bytes | bytearray | memoryview):
        raise TypeError(f"The generated {label} must contain PNG bytes.")
    payload = bytes(texture_png)
    if not payload.startswith(PNG_SIGNATURE):
        raise ValueError(f"The generated {label} must be a PNG image.")
    if len(payload) > MAX_SURFACE_TEXTURE_SOURCE_BYTES:
        raise ValueError(f"The generated {label} PNG is too large.")
    try:
        with Image.open(BytesIO(payload)) as image:
            if str(image.format or "").upper() != "PNG":
                raise ValueError(f"The generated {label} must be a PNG image.")
            if int(getattr(image, "n_frames", 1)) != 1:
                raise ValueError(f"The generated {label} must be a static PNG image.")
            width, height = image.size
            _validate_source_dimensions(width, height)
            image.load()
            rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    except ValueError:
        raise
    except (OSError, SyntaxError) as error:
        raise ValueError(
            f"The generated {label} is not a valid PNG image."
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


def _normalize_to_canonical_resolution(
    map_type: str,
    source_rgba: np.ndarray,
) -> np.ndarray:
    height, width = source_rgba.shape[:2]
    canonical_size = (
        CANONICAL_SURFACE_TEXTURE_RESOLUTION,
        CANONICAL_SURFACE_TEXTURE_RESOLUTION,
    )
    if (width, height) == canonical_size:
        if map_type == PBR_MAP_NORMAL:
            return _normalize_tangent_normal_rgba(source_rgba)
        return source_rgba.copy()
    interpolation = (
        cv2.INTER_AREA
        if width >= CANONICAL_SURFACE_TEXTURE_RESOLUTION
        and height >= CANONICAL_SURFACE_TEXTURE_RESOLUTION
        else cv2.INTER_LANCZOS4
    )
    return _resize_texture_map(
        map_type,
        source_rgba,
        CANONICAL_SURFACE_TEXTURE_RESOLUTION,
        interpolation,
    )


def _resize_texture_map(
    map_type: str,
    source_rgba: np.ndarray,
    resolution: int,
    interpolation: int,
) -> np.ndarray:
    if map_type == PBR_MAP_NORMAL:
        return _resize_tangent_normal_rgba(
            source_rgba,
            resolution,
            interpolation,
        )
    return _resize_rgba(source_rgba, resolution, interpolation)


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


# ### Normal-map helpers ###
def _repair_flat_normal_map(
    base_color_rgba: np.ndarray,
    provider_normal_rgba: np.ndarray,
) -> np.ndarray:
    """Replace an unusably flat provider normal only when color detail exists."""

    normalized_provider = provider_normal_rgba
    if not _is_overwhelmingly_flat_normal(normalized_provider):
        return normalized_provider

    luminance = _base_color_luminance(base_color_rgba)
    blurred = cv2.GaussianBlur(
        luminance,
        (0, 0),
        BASE_NORMAL_BLUR_SIGMA,
        borderType=cv2.BORDER_REFLECT101,
    )
    height, width = luminance.shape
    gradient_x = cv2.Scharr(blurred, cv2.CV_32F, 1, 0) / 16.0
    gradient_y = cv2.Scharr(blurred, cv2.CV_32F, 0, 1) / 16.0
    gradient_x *= width / NORMAL_GRADIENT_REFERENCE_RESOLUTION
    gradient_y *= height / NORMAL_GRADIENT_REFERENCE_RESOLUTION
    gradient_magnitude = cv2.magnitude(gradient_x, gradient_y)
    if not _has_meaningful_luminance_detail(luminance, gradient_magnitude):
        return normalized_provider

    meaningful_gradients = gradient_magnitude[
        gradient_magnitude > BASE_NORMAL_MIN_GRADIENT_RMS
    ]
    if meaningful_gradients.size == 0:
        return normalized_provider
    reference_gradient = float(np.percentile(meaningful_gradients, 95.0))
    gradient_scale = min(
        DERIVED_NORMAL_TARGET_SLOPE / reference_gradient,
        DERIVED_NORMAL_MAX_GRADIENT_SCALE,
    )
    tangent_vectors = np.empty((*luminance.shape, 3), dtype=np.float32)
    tangent_vectors[:, :, 0] = -gradient_x * gradient_scale
    tangent_vectors[:, :, 1] = gradient_y * gradient_scale
    tangent_vectors[:, :, 2] = 1.0
    return _encode_tangent_vectors(
        tangent_vectors,
        normalized_provider[:, :, 3],
    )


def _is_overwhelmingly_flat_normal(normal_rgba: np.ndarray) -> bool:
    median_rgb = np.asarray(
        [
            _lower_median_uint8(normal_rgba[:, :, channel])
            for channel in range(3)
        ],
        dtype=np.float32,
    )
    median_vector = median_rgb / 127.5 - 1.0
    median_length = float(np.linalg.norm(median_vector))
    if median_length <= NORMAL_VECTOR_EPSILON:
        median_vector = np.asarray((0.0, 0.0, 1.0), dtype=np.float32)
    else:
        median_vector = median_vector / median_length

    height, width = normal_rgba.shape[:2]
    maximum_detailed_pixels = int(
        height * width * FLAT_NORMAL_MAX_DETAILED_FRACTION
    )
    detailed_pixels = 0
    maximum_squared_deviation = FLAT_NORMAL_MAX_ANGULAR_DEVIATION**2
    for row_start in range(0, height, NORMAL_CLASSIFICATION_ROWS_PER_BATCH):
        row_end = min(row_start + NORMAL_CLASSIFICATION_ROWS_PER_BATCH, height)
        vectors = _decode_tangent_vectors(normal_rgba[row_start:row_end])
        squared_deviation = np.sum(
            np.square(vectors - median_vector),
            axis=2,
        )
        detailed_pixels += int(
            np.count_nonzero(squared_deviation > maximum_squared_deviation)
        )
        if detailed_pixels > maximum_detailed_pixels:
            return False
    return True


def _lower_median_uint8(channel: np.ndarray) -> int:
    histogram = np.bincount(channel.reshape(-1), minlength=256)
    lower_middle_index = (int(channel.size) - 1) // 2
    return int(np.searchsorted(np.cumsum(histogram), lower_middle_index + 1))


def _base_color_luminance(base_color_rgba: np.ndarray) -> np.ndarray:
    rgb = base_color_rgba[:, :, :3].astype(np.float32) / 255.0
    return np.ascontiguousarray(
        rgb[:, :, 0] * 0.2126
        + rgb[:, :, 1] * 0.7152
        + rgb[:, :, 2] * 0.0722,
        dtype=np.float32,
    )


def _has_meaningful_luminance_detail(
    luminance: np.ndarray,
    gradient_magnitude: np.ndarray,
) -> bool:
    luminance_std = float(np.std(luminance))
    gradient_rms = float(np.sqrt(np.mean(np.square(gradient_magnitude))))
    return bool(
        luminance_std >= BASE_NORMAL_MIN_LUMINANCE_STD
        and gradient_rms >= BASE_NORMAL_MIN_GRADIENT_RMS
    )


def _resize_tangent_normal_rgba(
    source_rgba: np.ndarray,
    resolution: int,
    interpolation: int,
) -> np.ndarray:
    tangent_vectors = _decode_tangent_vectors(source_rgba)
    resized_vectors = cv2.resize(
        tangent_vectors,
        (int(resolution), int(resolution)),
        interpolation=interpolation,
    )
    resized_alpha = cv2.resize(
        source_rgba[:, :, 3],
        (int(resolution), int(resolution)),
        interpolation=interpolation,
    )
    return _encode_tangent_vectors(resized_vectors, resized_alpha)


def _normalize_tangent_normal_rgba(source_rgba: np.ndarray) -> np.ndarray:
    return _encode_tangent_vectors(
        _decode_tangent_vectors(source_rgba),
        source_rgba[:, :, 3],
    )


def _decode_tangent_vectors(source_rgba: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        source_rgba[:, :, :3].astype(np.float32) / 127.5 - 1.0,
        dtype=np.float32,
    )


def _encode_tangent_vectors(
    tangent_vectors: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    lengths = np.linalg.norm(tangent_vectors, axis=2, keepdims=True)
    safe_lengths = np.where(lengths > NORMAL_VECTOR_EPSILON, lengths, 1.0)
    normalized_vectors = tangent_vectors / safe_lengths
    invalid_pixels = lengths[:, :, 0] <= NORMAL_VECTOR_EPSILON
    normalized_vectors[invalid_pixels] = (0.0, 0.0, 1.0)
    encoded_rgba = np.empty((*normalized_vectors.shape[:2], 4), dtype=np.uint8)
    encoded_rgba[:, :, :3] = np.clip(
        np.rint((normalized_vectors * 0.5 + 0.5) * 255.0),
        0,
        255,
    ).astype(np.uint8)
    encoded_rgba[:, :, 3] = np.clip(np.rint(alpha), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(encoded_rgba)


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
