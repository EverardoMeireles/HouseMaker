# ### Imports ###
from __future__ import annotations

import math
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from io import BytesIO
from typing import Protocol

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from housemaker.surface_texture_providers import (
    SurfaceTextureResult,
    SurfaceTextureTaskError,
    request_surface_texture,
)


# ### Constants ###
OBJECT_TEXTURE_INPAINT_RESOLUTION = 2048
TEXTURE_UV_MODE_PAINT = "paint"
TEXTURE_UV_MODE_ERASE = "erase"
TEXTURE_UV_MODES = frozenset({TEXTURE_UV_MODE_PAINT, TEXTURE_UV_MODE_ERASE})
MAX_TEXTURE_UV_POINTS_PER_STROKE = 100_000
RAY_INTERSECTION_EPSILON = 1e-9
DEFAULT_SCREEN_BRUSH_SAMPLE_SPACING_PIXELS = 4.0
DEFAULT_MAX_SCREEN_BRUSH_RAY_SAMPLES = 257
MIN_SCREEN_BRUSH_RAY_SAMPLES = 5
MAX_SCREEN_BRUSH_RAY_SAMPLES = 4_096
DEFAULT_OBJECT_TEXTURE_INPAINT_FEATHER_RADIUS_PIXELS = 8.0
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_INPUT_EDGE_PIXELS = 32_768
MAX_INPUT_PIXEL_COUNT = 64_000_000


# ### UV mask models ###
@dataclass(frozen=True)
class TextureUvPoint:
    """One normalized atlas coordinate, with V measured from the bottom."""

    u: float
    v: float

    def __post_init__(self) -> None:
        if not _is_finite_number(self.u) or not _is_finite_number(self.v):
            raise ValueError("Texture UV coordinates must be finite numbers.")
        if not 0.0 <= float(self.u) <= 1.0:
            raise ValueError("Texture U must be in [0, 1].")
        if not 0.0 <= float(self.v) <= 1.0:
            raise ValueError("Texture V must be in [0, 1].")
        object.__setattr__(self, "u", float(self.u))
        object.__setattr__(self, "v", float(self.v))

    def to_dict(self) -> dict[str, float]:
        return {"u": self.u, "v": self.v}

    @classmethod
    def from_dict(cls, payload: object) -> "TextureUvPoint":
        if not isinstance(payload, dict):
            raise ValueError("Texture UV point data must contain an object.")
        return cls(u=float(payload["u"]), v=float(payload["v"]))


@dataclass(frozen=True)
class TextureUvStroke:
    """One replayable paint or erase stroke in normalized UV coordinates."""

    mode: str
    radius_normalized: float
    points: tuple[TextureUvPoint, ...]
    connect_points: bool = True

    def __post_init__(self) -> None:
        if self.mode not in TEXTURE_UV_MODES:
            raise ValueError(f"Unknown texture UV stroke mode: {self.mode!r}.")
        if not _is_finite_number(self.radius_normalized):
            raise ValueError("Texture UV stroke radius must be finite.")
        if not 0.0 < float(self.radius_normalized) <= 1.0:
            raise ValueError("Texture UV stroke radius must be in (0, 1].")
        normalized_points = tuple(self.points)
        if not normalized_points:
            raise ValueError("Texture UV strokes must contain at least one point.")
        if len(normalized_points) > MAX_TEXTURE_UV_POINTS_PER_STROKE:
            raise ValueError("Texture UV stroke contains too many points.")
        if not all(
            isinstance(point, TextureUvPoint) for point in normalized_points
        ):
            raise ValueError(
                "Texture UV stroke points must be TextureUvPoint values."
            )
        if not isinstance(self.connect_points, bool):
            raise ValueError("Texture UV stroke connection mode must be a boolean.")
        object.__setattr__(
            self,
            "radius_normalized",
            float(self.radius_normalized),
        )
        object.__setattr__(self, "points", normalized_points)

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "radius_normalized": self.radius_normalized,
            "points": [point.to_dict() for point in self.points],
            "connect_points": self.connect_points,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "TextureUvStroke":
        if not isinstance(payload, dict):
            raise ValueError("Texture UV stroke data must contain an object.")
        raw_points = payload.get("points")
        if not isinstance(raw_points, list):
            raise ValueError("Texture UV stroke points must contain a list.")
        return cls(
            mode=str(payload["mode"]),
            radius_normalized=float(payload["radius_normalized"]),
            points=tuple(TextureUvPoint.from_dict(point) for point in raw_points),
            connect_points=payload.get("connect_points", True),
        )


@dataclass(frozen=True)
class TextureUvHit:
    """Nearest triangle hit and its interpolated atlas coordinate."""

    distance: float
    face_index: int
    point: TextureUvPoint
    barycentric_weights: tuple[float, float, float]
    is_back_facing: bool


TextureScreenRayBuilder = Callable[
    [tuple[float, float]],
    tuple[Sequence[float], Sequence[float]] | None,
]


# ### Inpaint request models ###
@dataclass(frozen=True)
class ObjectTextureInpaintRequest:
    """Owned canonical atlas, mask, references, and provider selection."""

    object_id: str
    provider: str
    api_key: str = field(repr=False)
    reference_pngs: tuple[bytes, ...]
    prompt: str
    existing_texture_png: bytes
    edit_mask_png: bytes

    def __post_init__(self) -> None:
        object_id = str(self.object_id).strip()
        provider = str(self.provider).strip()
        api_key = str(self.api_key).strip()
        prompt = str(self.prompt).strip()
        if not object_id:
            raise ValueError("Object texture inpainting requires an object ID.")
        if not provider:
            raise ValueError("Object texture inpainting requires a provider.")
        if not api_key:
            raise ValueError("Object texture inpainting requires an API key.")
        if not prompt:
            raise ValueError("Object texture inpainting requires a prompt.")
        try:
            reference_pngs = tuple(bytes(image) for image in self.reference_pngs)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Object texture references must contain PNG byte strings."
            ) from error
        if not reference_pngs or any(not image for image in reference_pngs):
            raise ValueError(
                "Object texture inpainting requires at least one reference PNG."
            )
        for image_index, reference_png in enumerate(reference_pngs, start=1):
            _load_png(reference_png, f"Object texture reference {image_index}")
        existing_rgba = _decode_exact_texture_png(
            self.existing_texture_png,
            "Existing object texture",
        )
        edit_mask = _decode_exact_edit_mask(self.edit_mask_png)
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "api_key", api_key)
        object.__setattr__(self, "reference_pngs", reference_pngs)
        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(
            self,
            "existing_texture_png",
            _encode_rgba_png(existing_rgba),
        )
        object.__setattr__(self, "edit_mask_png", _encode_mask_png(edit_mask))


@dataclass(frozen=True)
class ObjectTextureInpaintResult:
    """Provider result after HouseMaker enforces the exact editable mask."""

    object_id: str
    provider: str
    texture_png: bytes
    task_id: str | None = None


# ### Provider interface ###
ObjectTextureInpaintProgress = Callable[[str, int], None]


class ObjectTextureInpaintProvider(Protocol):
    def inpaint(
        self,
        request: ObjectTextureInpaintRequest,
        progress_callback: ObjectTextureInpaintProgress | None = None,
        cancel_event: threading.Event | None = None,
    ) -> ObjectTextureInpaintResult:
        """Generate only the selected atlas pixels."""


class SurfaceTextureRequester(Protocol):
    def __call__(
        self,
        provider: str,
        api_key: str,
        reference_pngs: Sequence[bytes],
        prompt: str,
        *,
        existing_texture_png: bytes | None = None,
        edit_mask_png: bytes | None = None,
        progress_callback: ObjectTextureInpaintProgress | None = None,
        cancel_event: threading.Event | None = None,
    ) -> SurfaceTextureResult:
        ...


class ObjectTextureInpaintCancelled(RuntimeError):
    """Raised when local cancellation wins over a provider result."""


class DefaultObjectTextureInpaintProvider:
    """Use the existing documented image provider adapters for atlas edits."""

    def __init__(
        self,
        requester: SurfaceTextureRequester = request_surface_texture,
    ) -> None:
        self._requester = requester

    def inpaint(
        self,
        request: ObjectTextureInpaintRequest,
        progress_callback: ObjectTextureInpaintProgress | None = None,
        cancel_event: threading.Event | None = None,
    ) -> ObjectTextureInpaintResult:
        if not isinstance(request, ObjectTextureInpaintRequest):
            raise TypeError("A valid object texture inpaint request is required.")
        _raise_if_cancelled(cancel_event)
        try:
            provider_result = self._requester(
                request.provider,
                request.api_key,
                request.reference_pngs,
                request.prompt,
                existing_texture_png=request.existing_texture_png,
                edit_mask_png=request.edit_mask_png,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )
        except SurfaceTextureTaskError as error:
            if cancel_event is not None and cancel_event.is_set():
                raise ObjectTextureInpaintCancelled(
                    "Object texture inpainting was canceled locally."
                ) from error
            raise
        if not isinstance(provider_result, SurfaceTextureResult):
            raise TypeError("The texture inpaint provider returned an invalid result.")
        _raise_if_cancelled(cancel_event)
        texture_png = composite_object_texture_inpaint(
            request.existing_texture_png,
            request.edit_mask_png,
            provider_result.texture_png,
        )
        validate_object_texture_inpaint_outside_mask(
            request.existing_texture_png,
            request.edit_mask_png,
            texture_png,
        )
        _raise_if_cancelled(cancel_event)
        return ObjectTextureInpaintResult(
            object_id=request.object_id,
            provider=provider_result.provider,
            texture_png=texture_png,
            task_id=provider_result.task_id,
        )


def inpaint_object_texture(
    request: ObjectTextureInpaintRequest,
    progress_callback: ObjectTextureInpaintProgress | None = None,
    cancel_event: threading.Event | None = None,
    *,
    provider: ObjectTextureInpaintProvider | None = None,
) -> ObjectTextureInpaintResult:
    """Run one provider-backed edit with exact outside-mask preservation."""

    active_provider = provider or DefaultObjectTextureInpaintProvider()
    return active_provider.inpaint(request, progress_callback, cancel_event)


# ### Mask rasterization ###
def rasterize_texture_uv_strokes(
    texture_size: tuple[int, int],
    strokes: Sequence[TextureUvStroke],
) -> np.ndarray:
    """Rasterize bottom-origin normalized UV strokes to a binary image mask."""

    width = max(0, int(texture_size[0]))
    height = max(0, int(texture_size[1]))
    if width <= 0 or height <= 0:
        return np.empty((0, 0), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    shortest_side = max(1, min(width, height))
    for stroke in strokes:
        if not isinstance(stroke, TextureUvStroke):
            raise TypeError("Texture UV masks require TextureUvStroke values.")
        radius = max(1, int(round(stroke.radius_normalized * shortest_side)))
        value = 0 if stroke.mode == TEXTURE_UV_MODE_ERASE else 255
        pixel_points = [
            (
                min(max(int(round(point.u * (width - 1))), 0), width - 1),
                min(
                    max(int(round((1.0 - point.v) * (height - 1))), 0),
                    height - 1,
                ),
            )
            for point in stroke.points
        ]
        cv2.circle(
            mask,
            pixel_points[0],
            radius,
            value,
            thickness=-1,
            lineType=cv2.LINE_8,
        )
        for start_point, end_point in zip(pixel_points, pixel_points[1:]):
            if stroke.connect_points:
                cv2.line(
                    mask,
                    start_point,
                    end_point,
                    value,
                    thickness=radius * 2,
                    lineType=cv2.LINE_8,
                )
            cv2.circle(
                mask,
                end_point,
                radius,
                value,
                thickness=-1,
                lineType=cv2.LINE_8,
            )
    return mask


# ### Mask enforcement ###
def composite_object_texture_inpaint(
    existing_texture_png: bytes,
    edit_mask_png: bytes,
    generated_texture_png: bytes,
    *,
    feather_radius_pixels: float = (
        DEFAULT_OBJECT_TEXTURE_INPAINT_FEATHER_RADIUS_PIXELS
    ),
) -> bytes:
    """Blend inward at the edit edge while preserving every outside pixel."""

    existing_rgba = _decode_exact_texture_png(
        existing_texture_png,
        "Existing object texture",
    )
    edit_mask = _decode_exact_edit_mask(edit_mask_png)
    generated_rgba = _decode_png_rgba(
        generated_texture_png,
        "Generated object texture",
    )
    if generated_rgba.shape[:2] != existing_rgba.shape[:2]:
        generated_rgba = cv2.resize(
            generated_rgba,
            (OBJECT_TEXTURE_INPAINT_RESOLUTION, OBJECT_TEXTURE_INPAINT_RESOLUTION),
            interpolation=cv2.INTER_LANCZOS4,
        )
    result = existing_rgba.copy()
    editable = edit_mask > 0
    feather_radius = _normalize_nonnegative_number(
        feather_radius_pixels,
        "Texture inpaint feather radius",
    )
    if feather_radius == 0.0:
        result[editable] = generated_rgba[editable]
    else:
        feather_weights = _build_inward_feather_weights(
            editable,
            feather_radius,
        )
        editable_rows, editable_columns = np.nonzero(editable)
        row_start = int(editable_rows.min())
        row_end = int(editable_rows.max()) + 1
        column_start = int(editable_columns.min())
        column_end = int(editable_columns.max()) + 1
        edit_crop = editable[row_start:row_end, column_start:column_end]
        weight_crop = feather_weights[
            row_start:row_end,
            column_start:column_end,
        ]
        result_crop = result[row_start:row_end, column_start:column_end]
        existing_crop = existing_rgba[
            row_start:row_end,
            column_start:column_end,
        ]
        generated_crop = generated_rgba[
            row_start:row_end,
            column_start:column_end,
        ]
        for channel_index in range(4):
            base_channel = existing_crop[:, :, channel_index].astype(
                np.float32
            )
            generated_channel = generated_crop[:, :, channel_index].astype(
                np.float32
            )
            blended_channel = np.rint(
                base_channel
                + (generated_channel - base_channel) * weight_crop
            ).astype(np.uint8)
            result_channel = result_crop[:, :, channel_index]
            result_channel[edit_crop] = blended_channel[edit_crop]
    if not np.array_equal(result[~editable], existing_rgba[~editable]):
        raise ValueError("Object texture inpainting changed pixels outside its mask.")
    return _encode_rgba_png(result)


def _build_inward_feather_weights(
    editable: np.ndarray,
    feather_radius_pixels: float,
) -> np.ndarray:
    padded_mask = np.pad(
        np.asarray(editable, dtype=np.uint8),
        ((1, 1), (1, 1)),
        mode="constant",
    )
    distance = cv2.distanceTransform(
        padded_mask,
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )[1:-1, 1:-1]
    normalized = np.clip(
        distance / float(feather_radius_pixels),
        0.0,
        1.0,
    ).astype(np.float32)
    smooth = normalized * normalized * (3.0 - 2.0 * normalized)
    smooth[~editable] = 0.0
    return np.ascontiguousarray(smooth, dtype=np.float32)


def validate_object_texture_inpaint_outside_mask(
    existing_texture_png: bytes,
    edit_mask_png: bytes,
    inpainted_texture_png: bytes,
) -> None:
    """Reject an output that changes any pixel outside the editable mask."""

    existing_rgba = _decode_exact_texture_png(
        existing_texture_png,
        "Existing object texture",
    )
    edit_mask = _decode_exact_edit_mask(edit_mask_png)
    inpainted_rgba = _decode_exact_texture_png(
        inpainted_texture_png,
        "Inpainted object texture",
    )
    if not np.array_equal(
        inpainted_rgba[edit_mask == 0],
        existing_rgba[edit_mask == 0],
    ):
        raise ValueError("Object texture inpainting changed pixels outside its mask.")


# ### Ray-to-UV picking ###
@dataclass(frozen=True)
class _PreparedUvRayMesh:
    faces: np.ndarray
    uv: np.ndarray
    triangles: np.ndarray
    first_edges: np.ndarray
    second_edges: np.ndarray


def pick_texture_uv_from_ray(
    mesh: object,
    ray_origin: Sequence[float],
    ray_direction: Sequence[float],
) -> TextureUvHit | None:
    """Return the nearest triangle UV hit without optional spatial indexes."""

    origin = _normalize_vector3(ray_origin, "Ray origin", normalize=False)
    direction = _normalize_vector3(
        ray_direction,
        "Ray direction",
        normalize=True,
    )
    prepared_mesh = _prepare_uv_ray_mesh(mesh)
    if prepared_mesh is None:
        return None
    return _pick_texture_uv_from_prepared_mesh(
        prepared_mesh,
        origin,
        direction,
    )


def _prepare_uv_ray_mesh(mesh: object) -> _PreparedUvRayMesh | None:
    """Validate and cache triangle arrays once for a group of screen rays."""

    try:
        vertices = np.asarray(getattr(mesh, "vertices"), dtype=float)
        faces = np.asarray(getattr(mesh, "faces"), dtype=np.int64)
        uv = np.asarray(getattr(getattr(mesh, "visual"), "uv"), dtype=float)
    except (AttributeError, TypeError, ValueError):
        return None
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or faces.ndim != 2
        or faces.shape[1:] != (3,)
        or uv.ndim != 2
        or uv.shape[1:] != (2,)
        or len(uv) != len(vertices)
        or not len(faces)
        or not np.all(np.isfinite(vertices))
        or not np.all(np.isfinite(uv))
        or np.any(faces < 0)
        or np.any(faces >= len(vertices))
    ):
        return None

    triangles = vertices[faces]
    first_edges = triangles[:, 1] - triangles[:, 0]
    second_edges = triangles[:, 2] - triangles[:, 0]
    return _PreparedUvRayMesh(
        faces=faces,
        uv=uv,
        triangles=triangles,
        first_edges=first_edges,
        second_edges=second_edges,
    )


def _pick_texture_uv_from_prepared_mesh(
    prepared_mesh: _PreparedUvRayMesh,
    ray_origin: Sequence[float],
    ray_direction: Sequence[float],
) -> TextureUvHit | None:
    """Return one nearest UV hit while reusing validated triangle arrays."""

    origin = _normalize_vector3(ray_origin, "Ray origin", normalize=False)
    direction = _normalize_vector3(ray_direction, "Ray direction", normalize=True)
    faces = prepared_mesh.faces
    uv = prepared_mesh.uv
    triangles = prepared_mesh.triangles
    first_edges = prepared_mesh.first_edges
    second_edges = prepared_mesh.second_edges
    perpendicular = np.cross(direction, second_edges)
    determinants = np.einsum("ij,ij->i", first_edges, perpendicular)
    valid = np.abs(determinants) > RAY_INTERSECTION_EPSILON
    safe_determinants = np.where(valid, determinants, 1.0)
    inverse_determinants = 1.0 / safe_determinants
    origin_deltas = origin - triangles[:, 0]
    first_coordinates = (
        np.einsum("ij,ij->i", origin_deltas, perpendicular)
        * inverse_determinants
    )
    valid &= first_coordinates >= -RAY_INTERSECTION_EPSILON
    valid &= first_coordinates <= 1.0 + RAY_INTERSECTION_EPSILON
    cross_deltas = np.cross(origin_deltas, first_edges)
    second_coordinates = (
        np.einsum("j,ij->i", direction, cross_deltas)
        * inverse_determinants
    )
    valid &= second_coordinates >= -RAY_INTERSECTION_EPSILON
    valid &= (
        first_coordinates + second_coordinates
        <= 1.0 + RAY_INTERSECTION_EPSILON
    )
    distances = (
        np.einsum("ij,ij->i", second_edges, cross_deltas)
        * inverse_determinants
    )
    valid &= distances > RAY_INTERSECTION_EPSILON
    if not np.any(valid):
        return None

    valid_indices = np.flatnonzero(valid)
    face_index = int(valid_indices[np.argmin(distances[valid_indices])])
    first_weight = float(first_coordinates[face_index])
    second_weight = float(second_coordinates[face_index])
    origin_weight = 1.0 - first_weight - second_weight
    barycentric_weights = (origin_weight, first_weight, second_weight)
    face_uv = uv[faces[face_index]]
    interpolated_uv = np.einsum(
        "i,ij->j",
        np.asarray(barycentric_weights, dtype=float),
        face_uv,
    )
    face_normal = np.cross(first_edges[face_index], second_edges[face_index])
    is_back_facing = float(np.dot(face_normal, direction)) >= 0.0
    return TextureUvHit(
        distance=float(distances[face_index]),
        face_index=face_index,
        point=TextureUvPoint(
            u=float(np.clip(interpolated_uv[0], 0.0, 1.0)),
            v=float(np.clip(interpolated_uv[1], 0.0, 1.0)),
        ),
        barycentric_weights=barycentric_weights,
        is_back_facing=is_back_facing,
    )


# ### Screen-space brush sampling ###
def sample_texture_uv_hits_from_screen_brush(
    mesh: object,
    cursor_position: Sequence[float],
    radius_pixels: float,
    ray_builder: TextureScreenRayBuilder,
    *,
    sample_spacing_pixels: float = (
        DEFAULT_SCREEN_BRUSH_SAMPLE_SPACING_PIXELS
    ),
    maximum_sample_count: int = DEFAULT_MAX_SCREEN_BRUSH_RAY_SAMPLES,
) -> tuple[TextureUvHit, ...]:
    """Sample nearest visible UVs throughout one circular screen brush.

    The center ray is only one sample. A brush can therefore reach nearby
    faces even when its cursor center lies over background. The deterministic
    lattice is capped before any rays are built.
    """

    cursor = _normalize_vector2(cursor_position, "Brush cursor")
    radius = _normalize_nonnegative_number(radius_pixels, "Brush radius")
    spacing = _normalize_positive_number(
        sample_spacing_pixels,
        "Brush sample spacing",
    )
    maximum_samples = _normalize_screen_brush_sample_count(
        maximum_sample_count
    )
    if not callable(ray_builder):
        raise TypeError("The screen brush ray builder must be callable.")
    prepared_mesh = _prepare_uv_ray_mesh(mesh)
    if prepared_mesh is None:
        return ()

    hits: list[TextureUvHit] = []
    seen_hits: set[tuple[int, float, float]] = set()
    for offset_x, offset_y in _build_screen_brush_offsets(
        radius,
        spacing,
        maximum_samples,
    ):
        sample_position = (
            float(cursor[0] + offset_x),
            float(cursor[1] + offset_y),
        )
        if not all(math.isfinite(value) for value in sample_position):
            raise ValueError("A screen brush sample exceeded the numeric range.")
        ray = ray_builder(sample_position)
        if ray is None:
            continue
        if not isinstance(ray, (tuple, list)) or len(ray) != 2:
            raise ValueError(
                "The screen brush ray builder returned an invalid ray."
            )
        hit = _pick_texture_uv_from_prepared_mesh(
            prepared_mesh,
            ray[0],
            ray[1],
        )
        if hit is None:
            continue
        hit_key = (
            hit.face_index,
            round(hit.point.u, 12),
            round(hit.point.v, 12),
        )
        if hit_key in seen_hits:
            continue
        seen_hits.add(hit_key)
        hits.append(hit)
    return tuple(hits)


def build_texture_uv_stamp_stroke_from_screen_brush(
    mesh: object,
    cursor_position: Sequence[float],
    radius_pixels: float,
    ray_builder: TextureScreenRayBuilder,
    *,
    mode: str,
    stamp_radius_normalized: float,
    sample_spacing_pixels: float = (
        DEFAULT_SCREEN_BRUSH_SAMPLE_SPACING_PIXELS
    ),
    maximum_sample_count: int = DEFAULT_MAX_SCREEN_BRUSH_RAY_SAMPLES,
) -> TextureUvStroke | None:
    """Build disconnected UV stamps for all faces under a screen brush."""

    if mode not in TEXTURE_UV_MODES:
        raise ValueError(f"Unknown texture UV stroke mode: {mode!r}.")
    stamp_radius = _normalize_positive_number(
        stamp_radius_normalized,
        "Texture UV stamp radius",
    )
    if stamp_radius > 1.0:
        raise ValueError("Texture UV stamp radius must be in (0, 1].")
    hits = sample_texture_uv_hits_from_screen_brush(
        mesh,
        cursor_position,
        radius_pixels,
        ray_builder,
        sample_spacing_pixels=sample_spacing_pixels,
        maximum_sample_count=maximum_sample_count,
    )
    if not hits:
        return None
    return TextureUvStroke(
        mode=mode,
        radius_normalized=stamp_radius,
        points=tuple(hit.point for hit in hits),
        connect_points=False,
    )


def _build_screen_brush_offsets(
    radius_pixels: float,
    sample_spacing_pixels: float,
    maximum_sample_count: int,
) -> tuple[tuple[float, float], ...]:
    if radius_pixels == 0.0:
        return ((0.0, 0.0),)
    maximum_steps = max(
        1,
        (math.isqrt(maximum_sample_count) - 1) // 2,
    )
    if sample_spacing_pixels >= radius_pixels:
        steps = 1
    else:
        requested_ratio = radius_pixels / sample_spacing_pixels
        steps = (
            maximum_steps
            if not math.isfinite(requested_ratio)
            or requested_ratio >= maximum_steps
            else max(1, int(math.ceil(requested_ratio)))
        )
    step_size = radius_pixels / steps
    offsets = [
        (column * step_size, row * step_size)
        for row in range(-steps, steps + 1)
        for column in range(-steps, steps + 1)
        if column * column + row * row <= steps * steps
    ]
    offsets.sort(
        key=lambda point: (
            point[0] ** 2 + point[1] ** 2,
            point[1],
            point[0],
        )
    )
    if len(offsets) > maximum_sample_count:
        raise RuntimeError("The bounded screen brush sampler exceeded its limit.")
    return tuple(offsets)


# ### Image helpers ###
def _decode_exact_texture_png(payload: bytes, label: str) -> np.ndarray:
    expected_size = (
        OBJECT_TEXTURE_INPAINT_RESOLUTION,
        OBJECT_TEXTURE_INPAINT_RESOLUTION,
    )
    image = _load_png(payload, label, expected_size=expected_size)
    return np.ascontiguousarray(
        np.asarray(image.convert("RGBA"), dtype=np.uint8)
    )


def _decode_exact_edit_mask(payload: bytes) -> np.ndarray:
    expected_size = (
        OBJECT_TEXTURE_INPAINT_RESOLUTION,
        OBJECT_TEXTURE_INPAINT_RESOLUTION,
    )
    image = _load_png(
        payload,
        "Object texture edit mask",
        expected_size=expected_size,
    )
    mask = np.asarray(image.convert("L"), dtype=np.uint8)
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    if not np.any(mask):
        raise ValueError("Object texture edit mask has no editable pixels.")
    return np.ascontiguousarray(mask)


def _decode_png_rgba(payload: bytes, label: str) -> np.ndarray:
    return np.ascontiguousarray(
        np.asarray(_load_png(payload, label).convert("RGBA"), dtype=np.uint8)
    )


def _load_png(
    payload: bytes,
    label: str,
    *,
    expected_size: tuple[int, int] | None = None,
) -> Image.Image:
    normalized = bytes(payload)
    if not normalized.startswith(PNG_SIGNATURE):
        raise ValueError(f"{label} must be a PNG image.")
    try:
        with Image.open(BytesIO(normalized)) as image:
            if str(image.format or "").upper() != "PNG":
                raise ValueError(f"{label} must be a PNG image.")
            if int(getattr(image, "n_frames", 1)) != 1:
                raise ValueError(f"{label} must be a static PNG image.")
            if expected_size is not None and image.size != expected_size:
                raise ValueError(
                    f"{label} must be {expected_size[0]} x {expected_size[1]}."
                )
            if (
                image.width <= 0
                or image.height <= 0
                or image.width > MAX_INPUT_EDGE_PIXELS
                or image.height > MAX_INPUT_EDGE_PIXELS
                or image.width * image.height > MAX_INPUT_PIXEL_COUNT
            ):
                raise ValueError(f"{label} dimensions are outside the supported limit.")
            image.load()
            return image.copy()
    except ValueError:
        raise
    except (OSError, SyntaxError, UnidentifiedImageError) as error:
        raise ValueError(f"{label} is not a valid PNG image.") from error


def _encode_rgba_png(rgba: np.ndarray) -> bytes:
    normalized = np.ascontiguousarray(rgba, dtype=np.uint8)
    did_encode, encoded = cv2.imencode(
        ".png",
        cv2.cvtColor(normalized, cv2.COLOR_RGBA2BGRA),
    )
    if not did_encode:
        raise ValueError("Object texture PNG encoding failed.")
    return bytes(encoded)


def _encode_mask_png(mask: np.ndarray) -> bytes:
    did_encode, encoded = cv2.imencode(
        ".png",
        np.ascontiguousarray(mask, dtype=np.uint8),
    )
    if not did_encode:
        raise ValueError("Object texture mask PNG encoding failed.")
    return bytes(encoded)


# ### Validation helpers ###
def _normalize_vector2(value: Sequence[float], label: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain two numbers.") from error
    if vector.shape != (2,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain two finite numbers.")
    return vector


def _normalize_vector3(
    value: Sequence[float],
    label: str,
    *,
    normalize: bool,
) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain three numbers.") from error
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain three finite numbers.")
    if normalize:
        length = float(np.linalg.norm(vector))
        if length <= RAY_INTERSECTION_EPSILON:
            raise ValueError(f"{label} cannot be zero length.")
        vector = vector / length
    return vector


def _normalize_nonnegative_number(value: object, label: str) -> float:
    if not _is_finite_number(value):
        raise ValueError(f"{label} must be a finite number.")
    normalized = float(value)
    if normalized < 0.0:
        raise ValueError(f"{label} cannot be negative.")
    return normalized


def _normalize_positive_number(value: object, label: str) -> float:
    normalized = _normalize_nonnegative_number(value, label)
    if normalized <= 0.0:
        raise ValueError(f"{label} must be positive.")
    return normalized


def _normalize_screen_brush_sample_count(value: object) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("The screen brush sample limit must be an integer.") from error
    if isinstance(value, bool) or normalized != value:
        raise ValueError("The screen brush sample limit must be an integer.")
    if not (
        MIN_SCREEN_BRUSH_RAY_SAMPLES
        <= normalized
        <= MAX_SCREEN_BRUSH_RAY_SAMPLES
    ):
        raise ValueError(
            "The screen brush sample limit must be between "
            f"{MIN_SCREEN_BRUSH_RAY_SAMPLES} and "
            f"{MAX_SCREEN_BRUSH_RAY_SAMPLES}."
        )
    return normalized


def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise ObjectTextureInpaintCancelled(
            "Object texture inpainting was canceled locally."
        )


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False
