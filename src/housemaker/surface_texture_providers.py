# ### Imports ###
from __future__ import annotations

import base64
import binascii
import json
import math
import struct
import threading
import time
import uuid
import warnings
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from io import BytesIO
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, UnidentifiedImageError

from housemaker.meshy_generation import (
    MESHY_RETEXTURE_ENDPOINT,
    build_retexture_request_body,
)
from housemaker.pbr_maps import (
    PBR_MAP_NORMAL,
    PBR_MAP_TYPES,
    normalize_pbr_map_types,
)


# ### Provider constants ###
MESHY_IMAGE_TO_IMAGE_ENDPOINT = (
    "https://api.meshy.ai/openapi/v1/image-to-image"
)
OPENAI_RESPONSES_ENDPOINT = "https://api.openai.com/v1/responses"
OPENAI_IMAGE_EDITS_ENDPOINT = "https://api.openai.com/v1/images/edits"

MESHY_PROVIDER = "meshy"
OPENAI_PROVIDER = "openai"
SUPPORTED_PROVIDERS = frozenset({MESHY_PROVIDER, OPENAI_PROVIDER})

DEFAULT_MESHY_IMAGE_MODEL = "nano-banana-2"
SUPPORTED_MESHY_IMAGE_MODELS = frozenset(
    {
        "nano-banana",
        "nano-banana-2",
        "nano-banana-pro",
        "gpt-image-2",
    }
)
DEFAULT_OPENAI_IMAGE_MODEL = "gpt-5.6-luna"
OPENAI_MINI_ANALYSIS_MODEL = "gpt-4o-mini"
OPENAI_IMAGE_RENDER_MODEL = "gpt-image-2"
OPENAI_DIRECT_IMAGE_MODELS = frozenset(
    {"gpt-5.6-luna", "gpt-5.6-terra"}
)
SUPPORTED_OPENAI_IMAGE_MODELS = (
    OPENAI_DIRECT_IMAGE_MODELS | {OPENAI_MINI_ANALYSIS_MODEL}
)
# ### Request limits ###
MIN_REFERENCE_IMAGE_COUNT = 1
MAX_REFERENCE_IMAGE_COUNT = 5
MAX_REFERENCE_PNG_BYTES = 20 * 1024 * 1024
MAX_TOTAL_REFERENCE_BYTES = 50 * 1024 * 1024
MAX_OUTPUT_PNG_BYTES = 48 * 1024 * 1024
MAX_PROMPT_CHARACTERS = 4_000
MAX_ANALYSIS_CHARACTERS = 8_000
MAX_PNG_EDGE_PIXELS = 32_768
MAX_PNG_PIXELS = 100_000_000

# GPT Image requests can legitimately take around two minutes to complete.
DEFAULT_REQUEST_TIMEOUT_SECONDS = 180.0
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
DEFAULT_MAX_POLLS = 360
MAX_METADATA_JSON_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_IMAGE_JSON_RESPONSE_BYTES = 72 * 1024 * 1024
MAX_ERROR_RESPONSE_BYTES = 64 * 1024
RETRYABLE_HTTP_STATUS_CODES = frozenset(
    {408, 409, 425, 429, 500, 502, 503, 504}
)
ACTIVE_TASK_STATUSES = frozenset({"PENDING", "IN_PROGRESS"})
TERMINAL_TASK_STATUSES = frozenset({"SUCCEEDED", "FAILED", "CANCELED"})
KNOWN_TASK_STATUSES = ACTIVE_TASK_STATUSES | TERMINAL_TASK_STATUSES
MESHY_TASK_RETRY_DELAYS_SECONDS = (5.0, 15.0)
RETRYABLE_MESHY_TASK_ERROR_TYPES = frozenset(
    {"server_error", "service_unavailable", "timeout"}
)
GENERIC_MESHY_INVALID_INPUT_MESSAGE = (
    "the input file or parameters could not be processed"
)
SURFACE_PBR_PRIMARY_U_MAX = 0.95


# ### Image constants ###
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_VALID_BIT_DEPTHS = {
    0: frozenset({1, 2, 4, 8, 16}),
    2: frozenset({8, 16}),
    3: frozenset({1, 2, 4, 8}),
    4: frozenset({8, 16}),
    6: frozenset({8, 16}),
}
SUPPORTED_DOWNLOADED_IMAGE_FORMATS = frozenset({"PNG", "JPEG", "WEBP"})
SUPPORTED_DOWNLOADED_IMAGE_CONTENT_TYPES = frozenset(
    {
        "application/octet-stream",
        "binary/octet-stream",
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
        "image/x-png",
    }
)
SURFACE_TEXTURE_PROMPT_PREFIX = (
    "Create one square, seamless, tileable, front-facing architectural "
    "surface texture from the reference images. Preserve the material's "
    "identity, colors, pattern, and believable physical scale. Remove "
    "perspective distortion, lighting gradients, shadows, reflections, "
    "objects, borders, and visible seams. Fill the image edge-to-edge with "
    "only the flat material. User direction: "
)
OPENAI_ANALYSIS_INSTRUCTION = (
    "Analyze every reference image and write one precise image-generation "
    "prompt for a square, seamless, tileable, flat architectural material "
    "texture. Reconcile the views into one consistent material, preserve its "
    "colors and pattern scale, and exclude perspective, lighting, shadows, "
    "reflections, objects, borders, and seams. Return only the final prompt "
    "without markdown. User direction: "
)
# ### Data models ###
@dataclass(frozen=True)
class SurfaceTextureProviderSettings:
    """Select one provider/model while keeping both service keys out of repr."""

    provider: str
    model: str = ""
    meshy_api_key: str = field(default="", repr=False)
    openai_api_key: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        provider = _normalize_provider(self.provider)
        meshy_key = _normalize_optional_api_key(
            self.meshy_api_key,
            "Meshy",
        )
        openai_key = _normalize_optional_api_key(
            self.openai_api_key,
            "OpenAI",
        )
        if provider == MESHY_PROVIDER:
            model = _normalize_model(
                self.model or DEFAULT_MESHY_IMAGE_MODEL,
                SUPPORTED_MESHY_IMAGE_MODELS,
                "Meshy",
            )
            if not meshy_key:
                raise ValueError("A Meshy API key is required.")
        else:
            model = _normalize_model(
                self.model or DEFAULT_OPENAI_IMAGE_MODEL,
                SUPPORTED_OPENAI_IMAGE_MODELS,
                "OpenAI",
            )
            if not openai_key:
                raise ValueError("An OpenAI API key is required.")

        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "meshy_api_key", meshy_key)
        object.__setattr__(self, "openai_api_key", openai_key)

    @property
    def active_api_key(self) -> str:
        if self.provider == MESHY_PROVIDER:
            return self.meshy_api_key
        return self.openai_api_key


@dataclass(frozen=True)
class SurfaceTextureRequest:
    """Owned PNG references and prompt for one generated surface texture."""

    reference_pngs: tuple[bytes, ...]
    prompt: str
    settings: SurfaceTextureProviderSettings
    enabled_pbr_maps: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.settings, SurfaceTextureProviderSettings):
            raise TypeError("Surface texture settings are invalid.")
        object.__setattr__(
            self,
            "reference_pngs",
            _normalize_reference_pngs(self.reference_pngs),
        )
        object.__setattr__(self, "prompt", _normalize_prompt(self.prompt))
        enabled_pbr_maps = normalize_pbr_map_types(
            self.enabled_pbr_maps,
            label="Enabled surface texture PBR maps",
        )
        if enabled_pbr_maps and self.settings.provider != MESHY_PROVIDER:
            raise ValueError(
                "Surface PBR map generation requires the Meshy provider."
            )
        object.__setattr__(self, "enabled_pbr_maps", enabled_pbr_maps)


@dataclass(frozen=True)
class SurfaceTextureResult:
    """Provider-neutral texture result consumed by the workspace."""

    provider: str
    texture_png: bytes
    task_id: str | None = None
    pbr_texture_pngs: Mapping[str, bytes] = field(default_factory=dict)
    pbr_task_id: str | None = None

    def __post_init__(self) -> None:
        provider = str(self.provider).strip()
        if not provider:
            raise ValueError("A surface texture result provider is required.")
        texture_png = _validate_png(
            self.texture_png,
            "Surface base-color texture",
            MAX_OUTPUT_PNG_BYTES,
        )
        pbr_texture_pngs = _normalize_result_pbr_texture_pngs(
            self.pbr_texture_pngs
        )
        task_id = _normalize_optional_task_id(self.task_id)
        pbr_task_id = _normalize_optional_task_id(self.pbr_task_id)
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "texture_png", texture_png)
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "pbr_texture_pngs", pbr_texture_pngs)
        object.__setattr__(self, "pbr_task_id", pbr_task_id)

    @property
    def available_pbr_maps(self) -> tuple[str, ...]:
        """Return downloaded PBR artifacts in canonical display order."""

        return tuple(
            map_type
            for map_type in PBR_MAP_TYPES
            if map_type in self.pbr_texture_pngs
        )


# ### Exceptions ###
class SurfaceTextureProviderError(RuntimeError):
    """Base exception safe to display in the application UI."""


class SurfaceTextureRequestError(SurfaceTextureProviderError):
    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class SurfaceTextureTaskError(SurfaceTextureProviderError):
    """A completed provider task failed or returned unusable output."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        task_id: str | None = None,
        error_type: str | None = None,
        error_code: str | None = None,
        doc_url: str | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = bool(retryable)
        self.task_id = task_id
        self.error_type = error_type
        self.error_code = error_code
        self.doc_url = doc_url


# ### Transport protocols ###
class ResponseStream(Protocol):
    def read(self, amount: int = -1) -> bytes:
        ...

    def __enter__(self) -> "ResponseStream":
        ...

    def __exit__(self, *args: object) -> object:
        ...


UrlOpenFunction = Callable[..., ResponseStream]
SleepFunction = Callable[[float], None]
ProgressCallback = Callable[[str, int], None]


# ### Request builders ###
def build_meshy_image_to_image_request_body(
    request: SurfaceTextureRequest,
) -> dict[str, Any]:
    """Build Meshy's documented multi-reference Image-to-Image payload."""

    _require_request_provider(request, MESHY_PROVIDER)
    return {
        "ai_model": request.settings.model,
        "prompt": _build_surface_texture_prompt(request.prompt),
        "reference_image_urls": [
            _png_data_uri(image_png) for image_png in request.reference_pngs
        ],
        "generate_multi_view": False,
        "aspect_ratio": "1:1",
    }


def build_openai_responses_request_body(
    request: SurfaceTextureRequest,
) -> dict[str, Any]:
    """Build a direct GPT-5.6 Responses image-generation request."""

    _require_request_provider(request, OPENAI_PROVIDER)
    if request.settings.model not in OPENAI_DIRECT_IMAGE_MODELS:
        raise ValueError(
            "Direct OpenAI texture generation requires GPT-5.6 Luna or Terra."
        )
    image_tool: dict[str, Any] = {
        "type": "image_generation",
        "action": "generate",
        "quality": "high",
        "size": "2048x2048",
    }
    return {
        "model": request.settings.model,
        "store": False,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": _build_surface_texture_prompt(
                            request.prompt,
                        ),
                    },
                    *[
                        {
                            "type": "input_image",
                            "image_url": _png_data_uri(image_png),
                            "detail": "high",
                        }
                        for image_png in request.reference_pngs
                    ],
                ],
            }
        ],
        "tools": [image_tool],
    }


def build_meshy_surface_pbr_request_body(
    base_color_png: bytes,
) -> dict[str, Any]:
    """Build the documented Retexture payload for an aligned surface map set."""

    normalized_base_color = _validate_png(
        base_color_png,
        "Generated surface base-color texture",
        MAX_OUTPUT_PNG_BYTES,
    )
    body = build_retexture_request_body(
        model_glb=_build_surface_pbr_slab_glb(),
        reference_images_png=(normalized_base_color,),
        enable_original_uv=True,
        enable_pbr=True,
    )
    return body


def build_openai_analysis_request_body(
    request: SurfaceTextureRequest,
) -> dict[str, Any]:
    """Build the GPT-4o mini analysis stage used before GPT Image 2."""

    _require_request_provider(request, OPENAI_PROVIDER)
    if request.settings.model != OPENAI_MINI_ANALYSIS_MODEL:
        raise ValueError("This analysis request requires GPT-4o mini.")
    return {
        "model": OPENAI_MINI_ANALYSIS_MODEL,
        "store": False,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            OPENAI_ANALYSIS_INSTRUCTION + request.prompt
                        ),
                    },
                    *[
                        {
                            "type": "input_image",
                            "image_url": _png_data_uri(image_png),
                            "detail": "high",
                        }
                        for image_png in request.reference_pngs
                    ],
                ],
            }
        ],
        "max_output_tokens": 800,
    }


def build_openai_image_edit_multipart(
    reference_pngs: Sequence[bytes],
    prompt: str,
) -> tuple[bytes, str]:
    """Build the GPT Image 2 multipart edit request used after mini analysis."""

    normalized_images = _normalize_reference_pngs(reference_pngs)
    normalized_prompt = _normalize_analysis(prompt)
    fields = (
        ("model", OPENAI_IMAGE_RENDER_MODEL),
        ("prompt", normalized_prompt),
        ("size", "2048x2048"),
        ("quality", "high"),
    )
    boundary = _new_multipart_boundary(
        [value.encode("utf-8") for _, value in fields]
        + list(normalized_images)
    )
    boundary_bytes = boundary.encode("ascii")
    chunks: list[bytes] = []

    for name, value in fields:
        chunks.extend(
            (
                b"--" + boundary_bytes + b"\r\n",
                (
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                ).encode("ascii"),
                value.encode("utf-8"),
                b"\r\n",
            )
        )
    for image_index, image_png in enumerate(normalized_images, start=1):
        chunks.extend(
            (
                b"--" + boundary_bytes + b"\r\n",
                (
                    "Content-Disposition: form-data; name=\"image[]\"; "
                    f'filename="reference_{image_index}.png"\r\n'
                ).encode("ascii"),
                b"Content-Type: image/png\r\n\r\n",
                image_png,
                b"\r\n",
            )
        )
    chunks.append(b"--" + boundary_bytes + b"--\r\n")
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


# ### Surface PBR slab ###
@lru_cache(maxsize=1)
def _build_surface_pbr_slab_glb() -> bytes:
    """Build a thin slab with a large, isolated primary surface UV island.

    Meshy's Retexture backend rejects zero-volume planes intermittently. A
    closed slab gives the provider valid three-dimensional geometry. The front
    face owns 95% of the atlas while the five support faces share the remainder
    without overlap, preventing conflicting PBR pixels from being baked over
    one another.
    """

    half_depth = 0.01
    faces = (
        # Front, back, right, left, top, bottom. Each face owns its vertices so
        # it can retain a flat normal and an independent UV island.
        (
            ((-0.5, -0.5, half_depth), (0.5, -0.5, half_depth),
             (0.5, 0.5, half_depth), (-0.5, 0.5, half_depth)),
            (0.0, 0.0, 1.0),
        ),
        (
            ((0.5, -0.5, -half_depth), (-0.5, -0.5, -half_depth),
             (-0.5, 0.5, -half_depth), (0.5, 0.5, -half_depth)),
            (0.0, 0.0, -1.0),
        ),
        (
            ((0.5, -0.5, half_depth), (0.5, -0.5, -half_depth),
             (0.5, 0.5, -half_depth), (0.5, 0.5, half_depth)),
            (1.0, 0.0, 0.0),
        ),
        (
            ((-0.5, -0.5, -half_depth), (-0.5, -0.5, half_depth),
             (-0.5, 0.5, half_depth), (-0.5, 0.5, -half_depth)),
            (-1.0, 0.0, 0.0),
        ),
        (
            ((-0.5, 0.5, half_depth), (0.5, 0.5, half_depth),
             (0.5, 0.5, -half_depth), (-0.5, 0.5, -half_depth)),
            (0.0, 1.0, 0.0),
        ),
        (
            ((-0.5, -0.5, -half_depth), (0.5, -0.5, -half_depth),
             (0.5, -0.5, half_depth), (-0.5, -0.5, half_depth)),
            (0.0, -1.0, 0.0),
        ),
    )
    primary_uvs = (
        (0.0, 0.0),
        (SURFACE_PBR_PRIMARY_U_MAX, 0.0),
        (SURFACE_PBR_PRIMARY_U_MAX, 1.0),
        (0.0, 1.0),
    )
    support_uvs = tuple(
        (
            (SURFACE_PBR_PRIMARY_U_MAX, support_index / 5.0),
            (1.0, support_index / 5.0),
            (1.0, (support_index + 1) / 5.0),
            (SURFACE_PBR_PRIMARY_U_MAX, (support_index + 1) / 5.0),
        )
        for support_index in range(5)
    )
    face_uvs = (primary_uvs, *support_uvs)
    position_values = tuple(
        component
        for vertices, _normal in faces
        for vertex in vertices
        for component in vertex
    )
    normal_values = tuple(
        component
        for _vertices, normal in faces
        for _vertex_index in range(4)
        for component in normal
    )
    texture_coordinate_values = tuple(
        component
        for coordinates in face_uvs
        for coordinate in coordinates
        for component in coordinate
    )
    index_values = tuple(
        face_index * 4 + local_index
        for face_index in range(len(faces))
        for local_index in (0, 1, 2, 0, 2, 3)
    )
    positions = struct.pack(f"<{len(position_values)}f", *position_values)
    normals = struct.pack(f"<{len(normal_values)}f", *normal_values)
    texture_coordinates = struct.pack(
        f"<{len(texture_coordinate_values)}f",
        *texture_coordinate_values,
    )
    indices = struct.pack(f"<{len(index_values)}H", *index_values)
    binary_payload = positions + normals + texture_coordinates + indices
    position_offset = 0
    normal_offset = len(positions)
    texture_coordinate_offset = normal_offset + len(normals)
    index_offset = texture_coordinate_offset + len(texture_coordinates)
    document = {
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 24,
                "max": [0.5, 0.5, half_depth],
                "min": [-0.5, -0.5, -half_depth],
                "type": "VEC3",
            },
            {
                "bufferView": 1,
                "componentType": 5126,
                "count": 24,
                "type": "VEC3",
            },
            {
                "bufferView": 2,
                "componentType": 5126,
                "count": 24,
                "max": [1.0, 1.0],
                "min": [0.0, 0.0],
                "type": "VEC2",
            },
            {
                "bufferView": 3,
                "componentType": 5123,
                "count": 36,
                "max": [23],
                "min": [0],
                "type": "SCALAR",
            },
        ],
        "asset": {"generator": "HouseMaker", "version": "2.0"},
        "bufferViews": [
            {
                "buffer": 0,
                "byteLength": len(positions),
                "byteOffset": position_offset,
                "target": 34962,
            },
            {
                "buffer": 0,
                "byteLength": len(normals),
                "byteOffset": normal_offset,
                "target": 34962,
            },
            {
                "buffer": 0,
                "byteLength": len(texture_coordinates),
                "byteOffset": texture_coordinate_offset,
                "target": 34962,
            },
            {
                "buffer": 0,
                "byteLength": len(indices),
                "byteOffset": index_offset,
                "target": 34963,
            },
        ],
        "buffers": [{"byteLength": len(binary_payload)}],
        "materials": [
            {
                "doubleSided": False,
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0,
                },
            }
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {
                            "NORMAL": 1,
                            "POSITION": 0,
                            "TEXCOORD_0": 2,
                        },
                        "indices": 3,
                        "material": 0,
                        "mode": 4,
                    }
                ]
            }
        ],
        "nodes": [{"mesh": 0}],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
    }
    json_payload = json.dumps(
        document,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    json_payload += b" " * ((-len(json_payload)) % 4)
    binary_payload += b"\x00" * ((-len(binary_payload)) % 4)
    total_length = 12 + 8 + len(json_payload) + 8 + len(binary_payload)
    return b"".join(
        (
            struct.pack("<4sII", b"glTF", 2, total_length),
            struct.pack("<I4s", len(json_payload), b"JSON"),
            json_payload,
            struct.pack("<I4s", len(binary_payload), b"BIN\x00"),
            binary_payload,
        )
    )


# ### Public generation API ###
def request_surface_texture(
    provider: str,
    api_key: str,
    reference_pngs: Sequence[bytes],
    prompt: str,
    *,
    enabled_pbr_maps: Sequence[str] = (),
    progress_callback: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
    opener: UrlOpenFunction | None = None,
    sleep: SleepFunction = time.sleep,
) -> SurfaceTextureResult:
    """Generate a texture using a workspace provider choice identifier."""

    provider_choice = _normalize_provider_choice(provider)
    if provider_choice == MESHY_PROVIDER:
        settings = SurfaceTextureProviderSettings(
            provider=MESHY_PROVIDER,
            meshy_api_key=api_key,
        )
    else:
        settings = SurfaceTextureProviderSettings(
            provider=OPENAI_PROVIDER,
            model=provider_choice,
            openai_api_key=api_key,
        )
    request = SurfaceTextureRequest(
        reference_pngs=tuple(reference_pngs),
        prompt=prompt,
        settings=settings,
        enabled_pbr_maps=tuple(enabled_pbr_maps),
    )
    _raise_if_cancelled(cancel_event)
    if provider_choice == MESHY_PROVIDER:
        return _generate_meshy_result(
            request=request,
            timeout_seconds=DEFAULT_REQUEST_TIMEOUT_SECONDS,
            poll_interval_seconds=DEFAULT_POLL_INTERVAL_SECONDS,
            max_polls=DEFAULT_MAX_POLLS,
            opener=opener,
            sleep=sleep,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
    texture_png = _generate_with_openai(
        request=request,
        timeout_seconds=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        opener=opener,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )
    return SurfaceTextureResult(
        provider=provider_choice,
        texture_png=texture_png,
    )


def generate_surface_texture(
    request: SurfaceTextureRequest,
    *,
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    max_polls: int = DEFAULT_MAX_POLLS,
    opener: UrlOpenFunction | None = None,
    sleep: SleepFunction = time.sleep,
    progress_callback: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> bytes:
    """Generate one PNG through the provider selected by the request."""

    if not isinstance(request, SurfaceTextureRequest):
        raise TypeError("A valid surface texture request is required.")
    timeout = _normalize_positive_float(timeout_seconds, "request timeout")
    _raise_if_cancelled(cancel_event)
    if request.settings.provider == MESHY_PROVIDER:
        return _generate_with_meshy(
            request=request,
            timeout_seconds=timeout,
            poll_interval_seconds=poll_interval_seconds,
            max_polls=max_polls,
            opener=opener,
            sleep=sleep,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
    return _generate_with_openai(
        request=request,
        timeout_seconds=timeout,
        opener=opener,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )


# ### Meshy adapter ###
def _generate_with_meshy(
    request: SurfaceTextureRequest,
    timeout_seconds: float,
    poll_interval_seconds: float,
    max_polls: int,
    opener: UrlOpenFunction | None,
    sleep: SleepFunction,
    progress_callback: ProgressCallback | None,
    cancel_event: threading.Event | None,
) -> bytes:
    return _generate_meshy_result(
        request=request,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        max_polls=max_polls,
        opener=opener,
        sleep=sleep,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    ).texture_png


def _generate_meshy_result(
    request: SurfaceTextureRequest,
    timeout_seconds: float,
    poll_interval_seconds: float,
    max_polls: int,
    opener: UrlOpenFunction | None,
    sleep: SleepFunction,
    progress_callback: ProgressCallback | None,
    cancel_event: threading.Event | None,
) -> SurfaceTextureResult:
    api_key = request.settings.meshy_api_key
    has_pbr_stage = bool(request.enabled_pbr_maps)
    image_progress_callback = (
        progress_callback
        if not has_pbr_stage
        else _build_stage_progress_callback(
            progress_callback,
            "IMAGE_TO_IMAGE",
            start_percent=0,
            span_percent=45,
        )
    )
    task_id, task = _create_and_wait_for_meshy_task(
        api_key=api_key,
        create_task=lambda: _create_meshy_image_task(
            request=request,
            timeout_seconds=timeout_seconds,
            opener=opener,
        ),
        endpoint=MESHY_IMAGE_TO_IMAGE_ENDPOINT,
        task_label="image",
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        max_polls=max_polls,
        opener=opener,
        sleep=sleep,
        progress_callback=image_progress_callback,
        cancel_event=cancel_event,
    )
    _raise_if_cancelled(cancel_event)
    image_url = _get_meshy_image_url(task, task_id)
    image_png = _download_image_as_png(
        image_url,
        provider_name="Meshy",
        timeout_seconds=timeout_seconds,
        opener=opener,
    )
    if not has_pbr_stage:
        return SurfaceTextureResult(
            provider=MESHY_PROVIDER,
            texture_png=image_png,
            task_id=task_id,
        )

    _raise_if_cancelled(cancel_event)
    _notify_progress(progress_callback, "PBR_RETEXTURE_SUBMITTING", 48)
    pbr_task_id, pbr_task = _create_and_wait_for_meshy_task(
        api_key=api_key,
        create_task=lambda: _create_meshy_pbr_task(
            api_key=api_key,
            base_color_png=image_png,
            timeout_seconds=timeout_seconds,
            opener=opener,
        ),
        endpoint=MESHY_RETEXTURE_ENDPOINT,
        task_label="surface PBR retexture",
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        max_polls=max_polls,
        opener=opener,
        sleep=sleep,
        progress_callback=_build_stage_progress_callback(
            progress_callback,
            "PBR_RETEXTURE",
            start_percent=50,
            span_percent=45,
        ),
        cancel_event=cancel_event,
    )
    _base_color_url, pbr_urls = _get_meshy_texture_urls(
        pbr_task,
        pbr_task_id,
    )
    missing_requested_maps = set(request.enabled_pbr_maps) - set(pbr_urls)
    if missing_requested_maps:
        raise SurfaceTextureTaskError(
            "Meshy surface PBR task "
            f"{pbr_task_id} omitted requested maps: "
            + ", ".join(sorted(missing_requested_maps))
            + "."
        )
    _raise_if_cancelled(cancel_event)
    _notify_progress(progress_callback, "DOWNLOADING_PBR_MAPS", 96)
    downloaded_pbr_maps: dict[str, bytes] = {}
    for map_type in PBR_MAP_TYPES:
        map_url = pbr_urls.get(map_type)
        if map_url is None:
            continue
        _raise_if_cancelled(cancel_event)
        downloaded_map = _download_image_as_png(
            map_url,
            provider_name=f"Meshy {map_type}",
            timeout_seconds=timeout_seconds,
            opener=opener,
        )
        downloaded_pbr_maps[map_type] = _align_surface_pbr_primary_region(
            downloaded_map,
            map_type=map_type,
            label=f"Meshy {map_type} texture image",
        )
    _notify_progress(progress_callback, "SUCCEEDED", 100)
    return SurfaceTextureResult(
        provider=MESHY_PROVIDER,
        # Retexture's helper-slab base color is less faithful than the detailed
        # image-to-image result that was supplied to it as the style source.
        texture_png=image_png,
        task_id=task_id,
        pbr_texture_pngs=downloaded_pbr_maps,
        pbr_task_id=pbr_task_id,
    )


def _create_and_wait_for_meshy_task(
    *,
    api_key: str,
    create_task: Callable[[], str],
    endpoint: str,
    task_label: str,
    timeout_seconds: float,
    poll_interval_seconds: float,
    max_polls: int,
    opener: UrlOpenFunction | None,
    sleep: SleepFunction,
    progress_callback: ProgressCallback | None,
    cancel_event: threading.Event | None,
) -> tuple[str, dict[str, Any]]:
    """Run one Meshy task with bounded backoff for classified failures."""

    latest_progress = 0

    def report_progress(status: str, progress: int) -> None:
        nonlocal latest_progress
        latest_progress = max(latest_progress, min(max(int(progress), 0), 100))
        _notify_progress(progress_callback, status, latest_progress)

    retry_delays = MESHY_TASK_RETRY_DELAYS_SECONDS
    for attempt_index in range(len(retry_delays) + 1):
        _raise_if_cancelled(cancel_event)
        try:
            task_id = create_task()
            task = _wait_for_meshy_task(
                api_key=api_key,
                task_id=task_id,
                endpoint=endpoint,
                task_label=task_label,
                timeout_seconds=timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
                max_polls=max_polls,
                opener=opener,
                sleep=sleep,
                progress_callback=report_progress,
                cancel_event=cancel_event,
            )
            return task_id, task
        except (SurfaceTextureRequestError, SurfaceTextureTaskError) as error:
            is_retryable = bool(getattr(error, "retryable", False))
            if not is_retryable or attempt_index >= len(retry_delays):
                raise
            retry_number = attempt_index + 1
            report_progress(f"RETRYING_{retry_number}", latest_progress)
            _interruptible_sleep(
                retry_delays[attempt_index],
                sleep,
                cancel_event,
            )

    raise AssertionError("The Meshy retry loop ended without a result.")


def _create_meshy_image_task(
    request: SurfaceTextureRequest,
    timeout_seconds: float,
    opener: UrlOpenFunction | None,
) -> str:
    api_key = request.settings.meshy_api_key
    return _create_meshy_task(
        api_key=api_key,
        endpoint=MESHY_IMAGE_TO_IMAGE_ENDPOINT,
        request_body=build_meshy_image_to_image_request_body(request),
        task_label="image",
        timeout_seconds=timeout_seconds,
        opener=opener,
    )


def _create_meshy_pbr_task(
    api_key: str,
    base_color_png: bytes,
    timeout_seconds: float,
    opener: UrlOpenFunction | None,
) -> str:
    return _create_meshy_task(
        api_key=api_key,
        endpoint=MESHY_RETEXTURE_ENDPOINT,
        request_body=build_meshy_surface_pbr_request_body(base_color_png),
        task_label="surface PBR retexture",
        timeout_seconds=timeout_seconds,
        opener=opener,
    )


def _create_meshy_task(
    api_key: str,
    endpoint: str,
    request_body: Mapping[str, Any],
    task_label: str,
    timeout_seconds: float,
    opener: UrlOpenFunction | None,
) -> str:
    response = _request_json(
        Request(
            endpoint,
            data=_encode_json(request_body),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        ),
        provider_name="Meshy",
        timeout_seconds=timeout_seconds,
        opener=opener,
        secrets=(api_key,),
        max_bytes=MAX_METADATA_JSON_RESPONSE_BYTES,
    )
    task_id = response.get("result")
    if not isinstance(task_id, str) or not task_id.strip():
        raise SurfaceTextureRequestError(
            f"Meshy returned an invalid {task_label} task identifier."
        )
    return _normalize_task_id(task_id)


def _get_meshy_task(
    api_key: str,
    task_id: str,
    endpoint: str,
    timeout_seconds: float,
    opener: UrlOpenFunction | None,
) -> dict[str, Any]:
    response = _request_json(
        Request(
            f"{endpoint}/{quote(task_id, safe='')}",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            },
            method="GET",
        ),
        provider_name="Meshy",
        timeout_seconds=timeout_seconds,
        opener=opener,
        secrets=(api_key,),
        max_bytes=MAX_METADATA_JSON_RESPONSE_BYTES,
    )
    return response


def _wait_for_meshy_task(
    api_key: str,
    task_id: str,
    endpoint: str,
    task_label: str,
    timeout_seconds: float,
    poll_interval_seconds: float,
    max_polls: int,
    opener: UrlOpenFunction | None,
    sleep: SleepFunction,
    progress_callback: ProgressCallback | None,
    cancel_event: threading.Event | None,
) -> dict[str, Any]:
    poll_interval = _normalize_non_negative_float(
        poll_interval_seconds,
        "poll interval",
    )
    poll_count = _normalize_positive_int(max_polls, "maximum poll count")

    for poll_index in range(poll_count):
        _raise_if_cancelled(cancel_event)
        task = _get_meshy_task(
            api_key=api_key,
            task_id=task_id,
            endpoint=endpoint,
            timeout_seconds=timeout_seconds,
            opener=opener,
        )
        status = _get_task_status(task, task_id)
        progress = _get_task_progress(task, status, task_id)
        _notify_progress(progress_callback, status, progress)
        if status == "SUCCEEDED":
            return task
        if status in {"FAILED", "CANCELED"}:
            error_message, error_type, error_code, doc_url = (
                _get_task_error_details(task)
            )
            detail = _redact(error_message, (api_key,))
            classification = "/".join(
                value for value in (error_type, error_code) if value
            )
            if classification:
                detail += f" [{classification}]"
            retryable = _is_retryable_meshy_task_failure(
                task_label,
                status,
                error_message,
                error_type,
                error_code,
            )
            raise SurfaceTextureTaskError(
                f"Meshy {task_label} task {task_id} "
                f"{status.lower()}: {detail}",
                retryable=retryable,
                task_id=task_id,
                error_type=error_type,
                error_code=error_code,
                doc_url=doc_url,
            )
        if poll_index + 1 < poll_count:
            _interruptible_sleep(
                poll_interval,
                sleep,
                cancel_event,
            )

    raise SurfaceTextureTaskError(
        f"Meshy {task_label} task {task_id} timed out before completion."
    )


# ### OpenAI adapter ###
def _generate_with_openai(
    request: SurfaceTextureRequest,
    timeout_seconds: float,
    opener: UrlOpenFunction | None,
    progress_callback: ProgressCallback | None,
    cancel_event: threading.Event | None,
) -> bytes:
    if request.settings.model == OPENAI_MINI_ANALYSIS_MODEL:
        generated_png = _generate_with_openai_mini_pipeline(
            request=request,
            timeout_seconds=timeout_seconds,
            opener=opener,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
    else:
        generated_png = _generate_with_openai_responses(
            request=request,
            timeout_seconds=timeout_seconds,
            opener=opener,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
    return generated_png


def _generate_with_openai_responses(
    request: SurfaceTextureRequest,
    timeout_seconds: float,
    opener: UrlOpenFunction | None,
    progress_callback: ProgressCallback | None,
    cancel_event: threading.Event | None,
) -> bytes:
    api_key = request.settings.openai_api_key
    _notify_progress(progress_callback, "IN_PROGRESS", 10)
    response = _request_json(
        Request(
            OPENAI_RESPONSES_ENDPOINT,
            data=_encode_json(build_openai_responses_request_body(request)),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        ),
        provider_name="OpenAI",
        timeout_seconds=timeout_seconds,
        opener=opener,
        secrets=(api_key,),
        max_bytes=MAX_IMAGE_JSON_RESPONSE_BYTES,
    )
    _raise_if_cancelled(cancel_event)
    result = _extract_openai_responses_png(response)
    _notify_progress(progress_callback, "SUCCEEDED", 100)
    return result


def _generate_with_openai_mini_pipeline(
    request: SurfaceTextureRequest,
    timeout_seconds: float,
    opener: UrlOpenFunction | None,
    progress_callback: ProgressCallback | None,
    cancel_event: threading.Event | None,
) -> bytes:
    api_key = request.settings.openai_api_key
    _notify_progress(progress_callback, "IN_PROGRESS", 10)
    analysis_response = _request_json(
        Request(
            OPENAI_RESPONSES_ENDPOINT,
            data=_encode_json(build_openai_analysis_request_body(request)),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        ),
        provider_name="OpenAI",
        timeout_seconds=timeout_seconds,
        opener=opener,
        secrets=(api_key,),
        max_bytes=MAX_METADATA_JSON_RESPONSE_BYTES,
    )
    analysis_prompt = _extract_openai_output_text(analysis_response)
    _raise_if_cancelled(cancel_event)
    _notify_progress(progress_callback, "IN_PROGRESS", 45)
    multipart_body, content_type = build_openai_image_edit_multipart(
        request.reference_pngs,
        analysis_prompt,
    )
    image_response = _request_json(
        Request(
            OPENAI_IMAGE_EDITS_ENDPOINT,
            data=multipart_body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": content_type,
                "Accept": "application/json",
            },
            method="POST",
        ),
        provider_name="OpenAI",
        timeout_seconds=timeout_seconds,
        opener=opener,
        secrets=(api_key,),
        max_bytes=MAX_IMAGE_JSON_RESPONSE_BYTES,
    )
    _raise_if_cancelled(cancel_event)
    result = _extract_openai_image_api_png(image_response)
    _notify_progress(progress_callback, "SUCCEEDED", 100)
    return result


# ### Artifact helpers ###
def _align_surface_pbr_primary_region(
    texture_png: bytes,
    *,
    map_type: str,
    label: str,
) -> bytes:
    """Restore and align Meshy's deterministic mirrored slab artifacts."""

    isolated_png = _extract_surface_pbr_primary_region(
        texture_png,
        label=label,
    )
    return align_surface_pbr_map_png(
        isolated_png,
        map_type=map_type,
        label=label,
    )


def align_surface_pbr_map_png(
    texture_png: bytes,
    *,
    map_type: str,
    label: str = "Surface PBR texture",
) -> bytes:
    """Mirror one Meshy surface map into HouseMaker's base-color orientation."""

    normalized_map_type = str(map_type).strip().lower()
    if normalized_map_type not in PBR_MAP_TYPES:
        raise ValueError("Surface PBR alignment requires a supported map type.")
    normalized_png = _validate_png(
        texture_png,
        label,
        MAX_OUTPUT_PNG_BYTES,
    )
    try:
        with Image.open(BytesIO(normalized_png)) as source_image:
            source_image.load()
            aligned_rgba = np.asarray(
                source_image.convert("RGBA").transpose(
                    Image.Transpose.FLIP_LEFT_RIGHT
                ),
                dtype=np.uint8,
            ).copy()
    except (OSError, SyntaxError, UnidentifiedImageError) as error:
        raise SurfaceTextureTaskError(
            f"{label} could not be aligned with the base color."
        ) from error

    if normalized_map_type == PBR_MAP_NORMAL:
        # Mirroring U reverses the tangent-space X axis. Negating the encoded
        # red channel keeps highlights and relief aligned with base-color pixels.
        aligned_rgba[:, :, 0] = 255 - aligned_rgba[:, :, 0]

    output = BytesIO()
    try:
        Image.fromarray(aligned_rgba, mode="RGBA").save(
            output,
            format="PNG",
            compress_level=6,
        )
    except (OSError, ValueError) as error:
        raise SurfaceTextureTaskError(
            f"{label} could not be aligned with the base color."
        ) from error
    return _validate_png(
        output.getvalue(),
        label,
        MAX_OUTPUT_PNG_BYTES,
    )


def _extract_surface_pbr_primary_region(
    texture_png: bytes,
    *,
    label: str,
) -> bytes:
    """Remove helper-face pixels and restore the primary island to a square."""

    normalized_png = _validate_png(
        texture_png,
        label,
        MAX_OUTPUT_PNG_BYTES,
    )
    try:
        with Image.open(BytesIO(normalized_png)) as source_image:
            source_image.load()
            width, height = source_image.size
            primary_width = max(
                1,
                min(width, round(width * SURFACE_PBR_PRIMARY_U_MAX)),
            )
            if primary_width == width:
                return normalized_png
            primary_region = source_image.convert("RGBA").crop(
                (0, 0, primary_width, height)
            )
            restored = primary_region.resize(
                (width, height),
                resample=Image.Resampling.LANCZOS,
            )
    except (OSError, SyntaxError, UnidentifiedImageError) as error:
        raise SurfaceTextureTaskError(
            f"{label} could not be isolated from the helper atlas."
        ) from error

    output = BytesIO()
    try:
        restored.save(output, format="PNG", compress_level=6)
    except (OSError, ValueError) as error:
        raise SurfaceTextureTaskError(
            f"{label} could not be restored to a square texture."
        ) from error
    return _validate_png(
        output.getvalue(),
        label,
        MAX_OUTPUT_PNG_BYTES,
    )


def _download_image_as_png(
    url: str,
    provider_name: str,
    timeout_seconds: float,
    opener: UrlOpenFunction | None,
) -> bytes:
    normalized_url = str(url).strip()
    parsed_url = urlparse(normalized_url)
    if parsed_url.scheme.lower() != "https" or not parsed_url.netloc:
        raise SurfaceTextureTaskError(
            f"{provider_name} image downloads must use an HTTPS URL."
        )
    request = Request(
        normalized_url,
        headers={
            "Accept": (
                "image/png,image/jpeg,image/webp;q=0.9,"
                "application/octet-stream;q=0.5"
            )
        },
        method="GET",
    )
    url_opener = urlopen if opener is None else opener
    try:
        with url_opener(request, timeout=timeout_seconds) as response:
            content_type = _get_response_content_type(response)
            payload = response.read(MAX_OUTPUT_PNG_BYTES + 1)
    except HTTPError as error:
        raise _build_http_error(
            error,
            provider_name=provider_name,
            secrets=(),
        ) from error
    except (URLError, TimeoutError, OSError) as error:
        raise SurfaceTextureRequestError(
            f"Unable to download the {provider_name} texture image.",
            retryable=True,
        ) from error

    if len(payload) > MAX_OUTPUT_PNG_BYTES:
        raise SurfaceTextureTaskError(
            f"{provider_name} returned an oversized texture image."
        )
    try:
        return _normalize_downloaded_image_to_png(
            payload,
            f"{provider_name} texture image",
            content_type,
        )
    except ValueError as error:
        raise SurfaceTextureTaskError(str(error)) from error


def _get_response_content_type(response: object) -> str | None:
    """Read a response MIME type without depending on one HTTP implementation."""

    headers = getattr(response, "headers", None)
    if headers is not None:
        get_header = getattr(headers, "get", None)
        if callable(get_header):
            try:
                content_type = get_header("Content-Type")
            except (AttributeError, TypeError, ValueError):
                content_type = None
            if isinstance(content_type, str) and content_type.strip():
                return content_type.split(";", maxsplit=1)[0].strip().lower()
            return None
        get_content_type = getattr(headers, "get_content_type", None)
        if callable(get_content_type):
            try:
                content_type = get_content_type()
            except (AttributeError, TypeError, ValueError):
                content_type = None
            if isinstance(content_type, str) and content_type.strip():
                return content_type.strip().lower()

    get_header = getattr(response, "getheader", None)
    if callable(get_header):
        try:
            content_type = get_header("Content-Type")
        except (AttributeError, TypeError, ValueError):
            content_type = None
        if isinstance(content_type, str) and content_type.strip():
            return content_type.split(";", maxsplit=1)[0].strip().lower()
    return None


def _normalize_downloaded_image_to_png(
    payload: bytes,
    label: str,
    content_type: str | None,
) -> bytes:
    """Validate a static PNG/JPEG/WebP artifact and return safe PNG bytes."""

    _validate_downloaded_image_content_type(content_type, label)
    if not payload:
        raise ValueError(f"{label} is empty.")
    if len(payload) > MAX_OUTPUT_PNG_BYTES:
        raise ValueError(f"{label} is too large.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as source_image:
                image_format = str(source_image.format or "").upper()
                if image_format not in SUPPORTED_DOWNLOADED_IMAGE_FORMATS:
                    raise ValueError(
                        f"{label} must be a PNG, JPEG, or WebP image."
                    )
                _validate_downloaded_image_dimensions(
                    source_image.width,
                    source_image.height,
                    label,
                )
                if int(getattr(source_image, "n_frames", 1)) != 1:
                    raise ValueError(f"{label} must be a static image.")
                source_image.load()
                if image_format == "PNG":
                    return _validate_png(
                        payload,
                        label,
                        MAX_OUTPUT_PNG_BYTES,
                    )
                normalized_image = source_image.convert("RGBA")
    except ValueError:
        raise
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        OSError,
        SyntaxError,
        UnidentifiedImageError,
    ) as error:
        raise ValueError(
            f"{label} is not a valid PNG, JPEG, or WebP image."
        ) from error

    output = BytesIO()
    try:
        normalized_image.save(output, format="PNG", compress_level=6)
    except (OSError, ValueError) as error:
        raise ValueError(f"{label} could not be normalized to PNG.") from error
    return _validate_png(
        output.getvalue(),
        label,
        MAX_OUTPUT_PNG_BYTES,
    )


def _validate_downloaded_image_content_type(
    content_type: str | None,
    label: str,
) -> None:
    if content_type is None:
        return
    normalized_type = content_type.split(";", maxsplit=1)[0].strip().lower()
    if normalized_type in SUPPORTED_DOWNLOADED_IMAGE_CONTENT_TYPES:
        return
    if normalized_type.startswith("image/"):
        raise ValueError(f"{label} has an unsupported image content type.")
    raise ValueError(f"{label} returned non-image content.")


def _validate_downloaded_image_dimensions(
    width: int,
    height: int,
    label: str,
) -> None:
    if (
        width <= 0
        or height <= 0
        or width > MAX_PNG_EDGE_PIXELS
        or height > MAX_PNG_EDGE_PIXELS
        or width * height > MAX_PNG_PIXELS
    ):
        raise ValueError(f"{label} dimensions are outside the supported limit.")


# ### HTTP helpers ###
def _request_json(
    request: Request,
    *,
    provider_name: str,
    timeout_seconds: float,
    opener: UrlOpenFunction | None,
    secrets: tuple[str, ...],
    max_bytes: int,
) -> dict[str, Any]:
    url_opener = urlopen if opener is None else opener
    try:
        with url_opener(request, timeout=timeout_seconds) as response:
            payload = response.read(max_bytes + 1)
    except HTTPError as error:
        raise _build_http_error(
            error,
            provider_name=provider_name,
            secrets=secrets,
        ) from error
    except (URLError, TimeoutError, OSError) as error:
        reason = str(getattr(error, "reason", None) or error).strip()
        message = (
            f"Unable to reach {provider_name}: "
            f"{reason or 'network unavailable'}"
        )
        raise SurfaceTextureRequestError(
            _redact(message, secrets),
            retryable=True,
        ) from error

    if len(payload) > max_bytes:
        raise SurfaceTextureRequestError(
            f"{provider_name} returned an oversized response."
        )
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SurfaceTextureRequestError(
            f"{provider_name} returned invalid JSON."
        ) from error
    if not isinstance(decoded, dict):
        raise SurfaceTextureRequestError(
            f"{provider_name} returned an invalid response object."
        )
    return decoded


def _build_http_error(
    error: HTTPError,
    *,
    provider_name: str,
    secrets: tuple[str, ...],
) -> SurfaceTextureRequestError:
    status_code = int(error.code)
    detail = "request rejected"
    try:
        payload = error.read(MAX_ERROR_RESPONSE_BYTES + 1)
        if len(payload) <= MAX_ERROR_RESPONSE_BYTES:
            decoded = json.loads(payload.decode("utf-8"))
            extracted_detail = _get_error_message(decoded)
            if extracted_detail:
                detail = extracted_detail
    except (OSError, UnicodeError, json.JSONDecodeError, AttributeError):
        pass
    message = _redact(
        f"{provider_name} request failed ({status_code}): {detail}",
        secrets,
    )
    return SurfaceTextureRequestError(
        message,
        status_code=status_code,
        retryable=status_code in RETRYABLE_HTTP_STATUS_CODES,
    )


def _encode_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


# ### Provider response helpers ###
def _extract_openai_responses_png(response: Mapping[str, Any]) -> bytes:
    output = response.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "image_generation_call":
                continue
            result = item.get("result")
            if isinstance(result, str) and result.strip():
                return _decode_png_base64(
                    result,
                    "OpenAI image generation result",
                )
    raise SurfaceTextureTaskError(
        "OpenAI completed without a generated texture image."
    )


def _extract_openai_image_api_png(response: Mapping[str, Any]) -> bytes:
    data = response.get("data")
    if isinstance(data, list) and data:
        first_image = data[0]
        if isinstance(first_image, dict):
            encoded = first_image.get("b64_json")
            if isinstance(encoded, str) and encoded.strip():
                return _decode_png_base64(
                    encoded,
                    "OpenAI image edit result",
                )
    raise SurfaceTextureTaskError(
        "OpenAI completed without a rendered texture image."
    )


def _extract_openai_output_text(response: Mapping[str, Any]) -> str:
    top_level_text = response.get("output_text")
    if isinstance(top_level_text, str) and top_level_text.strip():
        return _normalize_analysis(top_level_text)

    fragments: list[str] = []
    output = response.get("output")
    if isinstance(output, list):
        for item in output:
            content = item.get("content") if isinstance(item, dict) else None
            if not isinstance(content, list):
                continue
            for content_item in content:
                if not isinstance(content_item, dict):
                    continue
                if content_item.get("type") != "output_text":
                    continue
                text = content_item.get("text")
                if isinstance(text, str) and text.strip():
                    fragments.append(text.strip())
    if fragments:
        return _normalize_analysis("\n".join(fragments))
    raise SurfaceTextureTaskError(
        "GPT-4o mini completed without a usable texture prompt."
    )


def _get_meshy_image_url(task: Mapping[str, Any], task_id: str) -> str:
    image_urls = task.get("image_urls")
    if not isinstance(image_urls, list) or not image_urls:
        raise SurfaceTextureTaskError(
            f"Meshy image task {task_id} succeeded without an image artifact."
        )
    image_url = image_urls[0]
    if not isinstance(image_url, str) or not image_url.strip():
        raise SurfaceTextureTaskError(
            f"Meshy image task {task_id} returned an invalid image artifact."
        )
    return image_url.strip()


def _get_meshy_texture_urls(
    task: Mapping[str, Any],
    task_id: str,
) -> tuple[str, dict[str, str]]:
    """Return one aligned Meshy base color and every supported PBR URL."""

    texture_urls = task.get("texture_urls")
    if not isinstance(texture_urls, list) or not texture_urls:
        raise SurfaceTextureTaskError(
            f"Meshy surface PBR task {task_id} succeeded without textures."
        )
    first_texture_set = texture_urls[0]
    if not isinstance(first_texture_set, Mapping):
        raise SurfaceTextureTaskError(
            f"Meshy surface PBR task {task_id} returned invalid textures."
        )
    raw_base_color_url = first_texture_set.get("base_color")
    if not isinstance(raw_base_color_url, str) or not raw_base_color_url.strip():
        raise SurfaceTextureTaskError(
            f"Meshy surface PBR task {task_id} omitted its base color."
        )
    pbr_urls: dict[str, str] = {}
    for map_type in PBR_MAP_TYPES:
        raw_url = first_texture_set.get(map_type)
        if raw_url is None:
            continue
        if not isinstance(raw_url, str) or not raw_url.strip():
            raise SurfaceTextureTaskError(
                f"Meshy surface PBR task {task_id} returned an invalid "
                f"{map_type} artifact."
            )
        pbr_urls[map_type] = raw_url.strip()
    return raw_base_color_url.strip(), pbr_urls


def _get_task_status(task: Mapping[str, Any], task_id: str) -> str:
    status = task.get("status")
    if not isinstance(status, str):
        raise SurfaceTextureTaskError(
            f"Meshy image task {task_id} returned an invalid status."
        )
    normalized = status.upper()
    if normalized not in KNOWN_TASK_STATUSES:
        raise SurfaceTextureTaskError(
            f"Meshy image task {task_id} returned an invalid status."
        )
    return normalized


def _get_task_progress(
    task: Mapping[str, Any],
    status: str,
    task_id: str,
) -> int:
    default_progress = 100 if status == "SUCCEEDED" else 0
    progress = task.get("progress", default_progress)
    if isinstance(progress, bool) or not isinstance(progress, int):
        raise SurfaceTextureTaskError(
            f"Meshy image task {task_id} returned invalid progress."
        )
    if not 0 <= progress <= 100:
        raise SurfaceTextureTaskError(
            f"Meshy image task {task_id} returned invalid progress."
        )
    return progress


def _get_task_error_details(
    task: Mapping[str, Any],
) -> tuple[str, str | None, str | None, str | None]:
    """Return safe, structured Meshy task failure metadata."""

    task_error = task.get("task_error")
    if isinstance(task_error, dict):
        message = _normalize_optional_provider_text(task_error.get("message"))
        error_type = _normalize_optional_provider_text(task_error.get("type"))
        error_code = _normalize_optional_provider_text(task_error.get("code"))
        doc_url = _normalize_optional_provider_url(task_error.get("doc_url"))
        return (
            message or "unknown provider error",
            error_type,
            error_code,
            doc_url,
        )
    return "unknown provider error", None, None, None


def _is_retryable_meshy_task_failure(
    task_label: str,
    status: str,
    message: str,
    error_type: str | None,
    error_code: str | None,
) -> bool:
    """Classify documented transient failures and Meshy's generic image fault."""

    if status == "CANCELED":
        return False
    normalized_type = str(error_type or "").strip().lower()
    if normalized_type in RETRYABLE_MESHY_TASK_ERROR_TYPES:
        return True
    normalized_code = str(error_code or "").strip().lower()
    normalized_message = str(message).strip().lower()
    return (
        task_label == "image"
        and normalized_type == "invalid_input"
        and normalized_code == "invalid_input"
        and GENERIC_MESHY_INVALID_INPUT_MESSAGE in normalized_message
    )


def _normalize_optional_provider_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized[:1_000] or None


def _normalize_optional_provider_url(value: object) -> str | None:
    normalized = _normalize_optional_provider_text(value)
    if normalized is None or len(normalized) > 2_048:
        return None
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return normalized


def _get_error_message(decoded: object) -> str:
    if not isinstance(decoded, dict):
        return ""
    error_object = decoded.get("error")
    if isinstance(error_object, dict):
        message = error_object.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    message = decoded.get("message")
    if isinstance(message, str) and message.strip():
        return message.strip()
    return ""


# ### PNG helpers ###
def _normalize_reference_pngs(
    reference_pngs: Sequence[bytes],
) -> tuple[bytes, ...]:
    if isinstance(reference_pngs, (bytes, bytearray, memoryview, str)):
        raise TypeError("Reference PNGs must be provided as a sequence.")
    try:
        images = tuple(reference_pngs)
    except TypeError as error:
        raise TypeError("Reference PNGs must be provided as a sequence.") from error
    if not MIN_REFERENCE_IMAGE_COUNT <= len(images) <= MAX_REFERENCE_IMAGE_COUNT:
        raise ValueError(
            "Surface texture generation requires between "
            f"{MIN_REFERENCE_IMAGE_COUNT} and {MAX_REFERENCE_IMAGE_COUNT} "
            "reference PNGs."
        )

    normalized_images = tuple(
        _validate_png(
            image,
            f"Reference image {index}",
            MAX_REFERENCE_PNG_BYTES,
        )
        for index, image in enumerate(images, start=1)
    )
    if sum(map(len, normalized_images)) > MAX_TOTAL_REFERENCE_BYTES:
        raise ValueError("The combined reference PNG data is too large.")
    return normalized_images


def _validate_png(
    image: bytes,
    label: str,
    max_bytes: int,
) -> bytes:
    if not isinstance(image, (bytes, bytearray, memoryview)):
        raise TypeError(f"{label} must contain PNG bytes.")
    payload = bytes(image)
    if not payload:
        raise ValueError(f"{label} is empty.")
    if len(payload) > max_bytes:
        raise ValueError(f"{label} is too large.")
    if not payload.startswith(PNG_SIGNATURE):
        raise ValueError(f"{label} is not a PNG image.")

    offset = len(PNG_SIGNATURE)
    seen_ihdr = False
    seen_idat = False
    seen_iend = False
    while offset < len(payload):
        if offset + 12 > len(payload):
            raise ValueError(f"{label} is a truncated PNG image.")
        chunk_length = struct.unpack(">I", payload[offset : offset + 4])[0]
        chunk_type = payload[offset + 4 : offset + 8]
        chunk_end = offset + 12 + chunk_length
        if chunk_end > len(payload):
            raise ValueError(f"{label} is a truncated PNG image.")
        chunk_data = payload[offset + 8 : offset + 8 + chunk_length]
        expected_crc = struct.unpack(">I", payload[chunk_end - 4 : chunk_end])[0]
        actual_crc = zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ValueError(f"{label} has an invalid PNG checksum.")

        if not seen_ihdr and chunk_type != b"IHDR":
            raise ValueError(f"{label} has an invalid PNG header.")
        if chunk_type == b"IHDR":
            if seen_ihdr or chunk_length != 13:
                raise ValueError(f"{label} has an invalid PNG header.")
            _validate_png_header(chunk_data, label)
            seen_ihdr = True
        elif chunk_type == b"IDAT":
            seen_idat = True
        elif chunk_type == b"IEND":
            if chunk_length != 0 or chunk_end != len(payload):
                raise ValueError(f"{label} has an invalid PNG ending.")
            seen_iend = True
            offset = chunk_end
            break
        offset = chunk_end

    if not (seen_ihdr and seen_idat and seen_iend) or offset != len(payload):
        raise ValueError(f"{label} is an incomplete PNG image.")
    return payload


def _validate_png_header(header: bytes, label: str) -> None:
    width, height, bit_depth, color_type, compression, filter_method, interlace = (
        struct.unpack(">IIBBBBB", header)
    )
    valid_depths = PNG_VALID_BIT_DEPTHS.get(color_type, frozenset())
    if (
        width <= 0
        or height <= 0
        or width > MAX_PNG_EDGE_PIXELS
        or height > MAX_PNG_EDGE_PIXELS
        or width * height > MAX_PNG_PIXELS
        or bit_depth not in valid_depths
        or compression != 0
        or filter_method != 0
        or interlace not in {0, 1}
    ):
        raise ValueError(f"{label} has an unsupported PNG header.")


def _decode_png_base64(encoded: str, label: str) -> bytes:
    normalized = encoded.strip()
    prefix = "data:image/png;base64,"
    if normalized.startswith(prefix):
        normalized = normalized[len(prefix) :]
    maximum_encoded_length = ((MAX_OUTPUT_PNG_BYTES + 2) // 3) * 4 + 4
    if len(normalized) > maximum_encoded_length:
        raise SurfaceTextureTaskError(f"{label} is too large.")
    try:
        payload = base64.b64decode(normalized, validate=True)
    except (binascii.Error, ValueError) as error:
        raise SurfaceTextureTaskError(
            f"{label} contains invalid base64 data."
        ) from error
    try:
        return _validate_png(payload, label, MAX_OUTPUT_PNG_BYTES)
    except (TypeError, ValueError) as error:
        raise SurfaceTextureTaskError(str(error)) from error


def _png_data_uri(image_png: bytes) -> str:
    encoded = base64.b64encode(image_png).decode("ascii")
    return f"data:image/png;base64,{encoded}"


# ### Validation helpers ###
def _normalize_provider(provider: str) -> str:
    if not isinstance(provider, str):
        raise TypeError("Surface texture provider must be a string.")
    normalized = provider.strip().lower()
    if normalized not in SUPPORTED_PROVIDERS:
        raise ValueError("Surface texture provider must be Meshy or OpenAI.")
    return normalized


def _normalize_provider_choice(provider: str) -> str:
    if not isinstance(provider, str):
        raise TypeError("Surface texture provider must be a string.")
    normalized = provider.strip().lower()
    allowed_choices = {MESHY_PROVIDER, *SUPPORTED_OPENAI_IMAGE_MODELS}
    if normalized not in allowed_choices:
        allowed = ", ".join(sorted(allowed_choices))
        raise ValueError(
            "Unsupported surface texture provider. "
            f"Available choices: {allowed}."
        )
    return normalized


def _normalize_model(
    model: str,
    allowed_models: frozenset[str],
    provider_name: str,
) -> str:
    if not isinstance(model, str):
        raise TypeError(f"{provider_name} model must be a string.")
    normalized = model.strip().lower()
    if normalized not in allowed_models:
        allowed = ", ".join(sorted(allowed_models))
        raise ValueError(
            f"Unsupported {provider_name} surface texture model. "
            f"Available models: {allowed}."
        )
    return normalized


def _normalize_optional_api_key(api_key: str, provider_name: str) -> str:
    if not isinstance(api_key, str):
        raise TypeError(f"{provider_name} API key must be a string.")
    return api_key.strip()


def _normalize_prompt(prompt: str) -> str:
    if not isinstance(prompt, str):
        raise TypeError("Surface texture prompt must be a string.")
    normalized = prompt.strip()
    if not normalized:
        raise ValueError("A surface texture prompt is required.")
    if len(normalized) > MAX_PROMPT_CHARACTERS:
        raise ValueError("The surface texture prompt is too long.")
    return normalized


def _normalize_analysis(prompt: str) -> str:
    if not isinstance(prompt, str):
        raise TypeError("The generated texture prompt must be a string.")
    normalized = prompt.strip()
    if not normalized:
        raise SurfaceTextureTaskError(
            "GPT-4o mini returned an empty texture prompt."
        )
    if len(normalized) > MAX_ANALYSIS_CHARACTERS:
        raise SurfaceTextureTaskError(
            "GPT-4o mini returned an oversized texture prompt."
        )
    return normalized


def _normalize_task_id(task_id: str) -> str:
    normalized = task_id.strip()
    if not normalized or len(normalized) > 256:
        raise SurfaceTextureRequestError(
            "Meshy returned an invalid image task identifier."
        )
    return normalized


def _normalize_optional_task_id(task_id: object) -> str | None:
    if task_id is None:
        return None
    if not isinstance(task_id, str):
        raise ValueError("A surface texture task identifier must be a string.")
    try:
        return _normalize_task_id(task_id)
    except SurfaceTextureRequestError as error:
        raise ValueError(str(error)) from error


def _normalize_result_pbr_texture_pngs(
    raw_maps: object,
) -> dict[str, bytes]:
    if not isinstance(raw_maps, Mapping):
        raise ValueError("Surface PBR texture results must contain a mapping.")
    normalized_types = normalize_pbr_map_types(
        tuple(raw_maps),
        label="Surface PBR texture results",
    )
    if len(normalized_types) != len(raw_maps):
        raise ValueError("Surface PBR texture results contain duplicate map IDs.")
    normalized_payloads: dict[str, bytes] = {}
    for raw_map_type, raw_payload in raw_maps.items():
        map_type = str(raw_map_type).strip().lower()
        normalized_payloads[map_type] = _validate_png(
            raw_payload,
            f"Surface {map_type} texture",
            MAX_OUTPUT_PNG_BYTES,
        )
    return {
        map_type: normalized_payloads[map_type]
        for map_type in normalized_types
    }


def _normalize_positive_float(value: float, label: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"The {label} must be positive and finite.") from error
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"The {label} must be positive and finite.")
    return normalized


def _normalize_non_negative_float(value: float, label: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"The {label} must be non-negative and finite."
        ) from error
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"The {label} must be non-negative and finite.")
    return normalized


def _normalize_positive_int(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"The {label} must be a positive integer.")
    return value


def _require_request_provider(
    request: SurfaceTextureRequest,
    provider: str,
) -> None:
    if not isinstance(request, SurfaceTextureRequest):
        raise TypeError("A valid surface texture request is required.")
    if request.settings.provider != provider:
        raise ValueError(f"This request does not use the {provider} provider.")


def _build_surface_texture_prompt(prompt: str) -> str:
    return SURFACE_TEXTURE_PROMPT_PREFIX + prompt


def _new_multipart_boundary(values: Sequence[bytes]) -> str:
    for _attempt in range(10):
        boundary = f"----HouseMakerTexture{uuid.uuid4().hex}"
        encoded_boundary = boundary.encode("ascii")
        if all(encoded_boundary not in value for value in values):
            return boundary
    raise SurfaceTextureRequestError(
        "Unable to safely encode the OpenAI image request."
    )


# ### Progress and cancellation helpers ###
def _build_stage_progress_callback(
    callback: ProgressCallback | None,
    stage_name: str,
    *,
    start_percent: int,
    span_percent: int,
) -> ProgressCallback | None:
    if callback is None:
        return None

    def report(status: str, progress: int) -> None:
        stage_progress = int(start_percent) + round(
            min(max(int(progress), 0), 100) * int(span_percent) / 100.0
        )
        callback(f"{stage_name}_{status}", stage_progress)

    return report


def _notify_progress(
    callback: ProgressCallback | None,
    status: str,
    progress: int,
) -> None:
    if callback is not None:
        callback(status, progress)


def _interruptible_sleep(
    seconds: float,
    sleep: SleepFunction,
    cancel_event: threading.Event | None,
) -> None:
    if cancel_event is None or sleep is not time.sleep:
        sleep(seconds)
        _raise_if_cancelled(cancel_event)
        return
    if cancel_event.wait(seconds):
        _raise_if_cancelled(cancel_event)


def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise SurfaceTextureTaskError(
            "Surface texture generation was canceled locally."
        )


# ### Redaction helpers ###
def _redact(message: str, secrets: tuple[str, ...]) -> str:
    redacted = str(message)
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "[redacted]")
    return redacted
