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
from io import BytesIO
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from PIL import Image, UnidentifiedImageError


# ### Provider constants ###
MESHY_IMAGE_TO_IMAGE_ENDPOINT = (
    "https://api.meshy.ai/openapi/v1/image-to-image"
)
OPENAI_RESPONSES_ENDPOINT = "https://api.openai.com/v1/responses"
OPENAI_IMAGE_EDITS_ENDPOINT = "https://api.openai.com/v1/images/edits"
OPENAI_FILES_ENDPOINT = "https://api.openai.com/v1/files"

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
MAX_PACKED_REFERENCE_EDGE_PIXELS = 4_096

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
PARTIAL_TEXTURE_PROMPT_PREFIX = (
    "Edit only the marked portion of the existing texture. Keep its scale, "
    "layout, and continuity at the edit boundary; leave every unmarked pixel "
    "unchanged. Use the painted video references as the source for the new "
    "material detail. "
)
MESHY_PARTIAL_TEXTURE_PROMPT_PREFIX = (
    "This is a localized texture edit. The first reference is the existing "
    "texture. The second is a control image of that texture with the editable "
    "region highlighted in bright magenta. Only replace the magenta-marked "
    "region, using the remaining video references as visual evidence. Preserve "
    "the original texture everywhere else and blend the edit naturally at its "
    "boundary. "
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
    existing_texture_png: bytes | None = None
    edit_mask_png: bytes | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.settings, SurfaceTextureProviderSettings):
            raise TypeError("Surface texture settings are invalid.")
        object.__setattr__(
            self,
            "reference_pngs",
            _normalize_reference_pngs(self.reference_pngs),
        )
        object.__setattr__(self, "prompt", _normalize_prompt(self.prompt))
        existing_texture_png, edit_mask_png = _normalize_partial_edit_images(
            self.existing_texture_png,
            self.edit_mask_png,
        )
        object.__setattr__(self, "existing_texture_png", existing_texture_png)
        object.__setattr__(self, "edit_mask_png", edit_mask_png)

    @property
    def is_partial_edit(self) -> bool:
        return self.existing_texture_png is not None


@dataclass(frozen=True)
class SurfaceTextureResult:
    """Provider-neutral texture result consumed by the workspace."""

    provider: str
    texture_png: bytes
    task_id: str | None = None


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
    pass


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
    reference_pngs = list(request.reference_pngs)
    prompt_prefix = ""
    if request.is_partial_edit:
        assert request.existing_texture_png is not None
        assert request.edit_mask_png is not None
        control_png = _build_meshy_partial_edit_control_png(
            request.existing_texture_png,
            request.edit_mask_png,
        )
        packed_video_references = _pack_meshy_partial_video_references(
            request.reference_pngs,
            MAX_REFERENCE_IMAGE_COUNT - 2,
        )
        reference_pngs = [
            request.existing_texture_png,
            control_png,
            *packed_video_references,
        ]
        prompt_prefix = MESHY_PARTIAL_TEXTURE_PROMPT_PREFIX
    return {
        "ai_model": request.settings.model,
        "prompt": prompt_prefix + _build_surface_texture_prompt(
            request.prompt,
            partial_edit=request.is_partial_edit,
        ),
        "reference_image_urls": [
            _png_data_uri(image_png) for image_png in reference_pngs
        ],
        "generate_multi_view": False,
        "aspect_ratio": "1:1",
    }


def build_openai_responses_request_body(
    request: SurfaceTextureRequest,
    *,
    existing_texture_file_id: str | None = None,
    edit_mask_file_id: str | None = None,
) -> dict[str, Any]:
    """Build a direct GPT-5.6 Responses image-generation request."""

    _require_request_provider(request, OPENAI_PROVIDER)
    if request.settings.model not in OPENAI_DIRECT_IMAGE_MODELS:
        raise ValueError(
            "Direct OpenAI texture generation requires GPT-5.6 Luna or Terra."
        )
    if request.is_partial_edit:
        existing_file_id = _normalize_openai_file_id(
            existing_texture_file_id,
            "existing texture",
        )
        mask_file_id = _normalize_openai_file_id(
            edit_mask_file_id,
            "texture edit mask",
        )
        existing_content = [
            {
                "type": "input_image",
                "file_id": existing_file_id,
                "detail": "high",
            }
        ]
        image_tool: dict[str, Any] = {
            "type": "image_generation",
            "action": "edit",
            "quality": "high",
            "size": "2048x2048",
            "input_image_mask": {"file_id": mask_file_id},
        }
    else:
        if existing_texture_file_id is not None or edit_mask_file_id is not None:
            raise ValueError("OpenAI edit file IDs require a partial texture edit.")
        existing_content = []
        image_tool = {
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
                            partial_edit=request.is_partial_edit,
                        ),
                    },
                    *existing_content,
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
                            OPENAI_ANALYSIS_INSTRUCTION
                            + (
                                PARTIAL_TEXTURE_PROMPT_PREFIX
                                if request.is_partial_edit
                                else ""
                            )
                            + request.prompt
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
    *,
    existing_texture_png: bytes | None = None,
    edit_mask_png: bytes | None = None,
) -> tuple[bytes, str]:
    """Build the GPT Image 2 multipart edit request used after mini analysis."""

    normalized_images = _normalize_reference_pngs(reference_pngs)
    normalized_prompt = _normalize_analysis(prompt)
    normalized_existing, normalized_mask = _normalize_partial_edit_images(
        existing_texture_png,
        edit_mask_png,
    )
    if normalized_existing is not None:
        normalized_prompt = (
            PARTIAL_TEXTURE_PROMPT_PREFIX + normalized_prompt
        )
        normalized_images = (normalized_existing, *normalized_images)
        normalized_mask = _build_openai_edit_mask_png(normalized_mask)
    fields = (
        ("model", OPENAI_IMAGE_RENDER_MODEL),
        ("prompt", normalized_prompt),
        ("size", "2048x2048"),
        ("quality", "high"),
    )
    boundary = _new_multipart_boundary(
        [value.encode("utf-8") for _, value in fields]
        + list(normalized_images)
        + ([] if normalized_mask is None else [normalized_mask])
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
    if normalized_mask is not None:
        chunks.extend(
            (
                b"--" + boundary_bytes + b"\r\n",
                (
                    "Content-Disposition: form-data; name=\"mask\"; "
                    'filename="texture_edit_mask.png"\r\n'
                ).encode("ascii"),
                b"Content-Type: image/png\r\n\r\n",
                normalized_mask,
                b"\r\n",
            )
        )
    chunks.append(b"--" + boundary_bytes + b"--\r\n")
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def build_openai_file_upload_multipart(
    image_png: bytes,
    filename: str,
) -> tuple[bytes, str]:
    """Build one documented ``purpose=vision`` OpenAI file upload."""

    normalized_image = _validate_png(
        image_png,
        "OpenAI vision upload",
        MAX_OUTPUT_PNG_BYTES,
    )
    if filename not in {"existing_texture.png", "texture_edit_mask.png"}:
        raise ValueError("OpenAI vision upload filename is invalid.")
    purpose = b"vision"
    boundary = _new_multipart_boundary([purpose, normalized_image])
    boundary_bytes = boundary.encode("ascii")
    body = b"".join(
        (
            b"--" + boundary_bytes + b"\r\n",
            b'Content-Disposition: form-data; name="purpose"\r\n\r\n',
            purpose,
            b"\r\n",
            b"--" + boundary_bytes + b"\r\n",
            (
                "Content-Disposition: form-data; name=\"file\"; "
                f'filename="{filename}"\r\n'
            ).encode("ascii"),
            b"Content-Type: image/png\r\n\r\n",
            normalized_image,
            b"\r\n",
            b"--" + boundary_bytes + b"--\r\n",
        )
    )
    return body, f"multipart/form-data; boundary={boundary}"


# ### Public generation API ###
def request_surface_texture(
    provider: str,
    api_key: str,
    reference_pngs: Sequence[bytes],
    prompt: str,
    *,
    existing_texture_png: bytes | None = None,
    edit_mask_png: bytes | None = None,
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
        existing_texture_png=existing_texture_png,
        edit_mask_png=edit_mask_png,
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
    task_id = _create_meshy_task(
        request=request,
        timeout_seconds=timeout_seconds,
        opener=opener,
    )
    task = _wait_for_meshy_task(
        api_key=api_key,
        task_id=task_id,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        max_polls=max_polls,
        opener=opener,
        sleep=sleep,
        progress_callback=progress_callback,
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
    image_png = composite_partial_texture_edit(
        image_png,
        request.existing_texture_png,
        request.edit_mask_png,
    )
    return SurfaceTextureResult(
        provider=MESHY_PROVIDER,
        texture_png=image_png,
        task_id=task_id,
    )


def _create_meshy_task(
    request: SurfaceTextureRequest,
    timeout_seconds: float,
    opener: UrlOpenFunction | None,
) -> str:
    api_key = request.settings.meshy_api_key
    response = _request_json(
        Request(
            MESHY_IMAGE_TO_IMAGE_ENDPOINT,
            data=_encode_json(build_meshy_image_to_image_request_body(request)),
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
            "Meshy returned an invalid image task identifier."
        )
    return _normalize_task_id(task_id)


def _get_meshy_task(
    api_key: str,
    task_id: str,
    timeout_seconds: float,
    opener: UrlOpenFunction | None,
) -> dict[str, Any]:
    response = _request_json(
        Request(
            f"{MESHY_IMAGE_TO_IMAGE_ENDPOINT}/"
            f"{quote(task_id, safe='')}",
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
            timeout_seconds=timeout_seconds,
            opener=opener,
        )
        status = _get_task_status(task, task_id)
        progress = _get_task_progress(task, status, task_id)
        _notify_progress(progress_callback, status, progress)
        if status == "SUCCEEDED":
            _get_meshy_image_url(task, task_id)
            return task
        if status in {"FAILED", "CANCELED"}:
            detail = _redact(
                _get_task_error_message(task),
                (api_key,),
            )
            raise SurfaceTextureTaskError(
                f"Meshy image task {task_id} {status.lower()}: {detail}"
            )
        if poll_index + 1 < poll_count:
            _interruptible_sleep(
                poll_interval,
                sleep,
                cancel_event,
            )

    raise SurfaceTextureTaskError(
        f"Meshy image task {task_id} timed out before completion."
    )


# ### OpenAI adapter ###
def _upload_openai_vision_png(
    api_key: str,
    image_png: bytes,
    filename: str,
    timeout_seconds: float,
    opener: UrlOpenFunction | None,
) -> str:
    multipart_body, content_type = build_openai_file_upload_multipart(
        image_png,
        filename,
    )
    response = _request_json(
        Request(
            OPENAI_FILES_ENDPOINT,
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
        max_bytes=MAX_METADATA_JSON_RESPONSE_BYTES,
    )
    return _normalize_openai_file_id(response.get("id"), filename)


def _best_effort_delete_openai_file(
    api_key: str,
    file_id: str,
    timeout_seconds: float,
    opener: UrlOpenFunction | None,
) -> None:
    try:
        _request_json(
            Request(
                f"{OPENAI_FILES_ENDPOINT}/{quote(file_id, safe='')}",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Accept": "application/json",
                },
                method="DELETE",
            ),
            provider_name="OpenAI",
            timeout_seconds=min(timeout_seconds, 15.0),
            opener=opener,
            secrets=(api_key,),
            max_bytes=MAX_METADATA_JSON_RESPONSE_BYTES,
        )
    except Exception:
        # Cleanup failure must not discard an otherwise valid generated texture.
        return


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
    return composite_partial_texture_edit(
        generated_png,
        request.existing_texture_png,
        request.edit_mask_png,
    )


def _generate_with_openai_responses(
    request: SurfaceTextureRequest,
    timeout_seconds: float,
    opener: UrlOpenFunction | None,
    progress_callback: ProgressCallback | None,
    cancel_event: threading.Event | None,
) -> bytes:
    api_key = request.settings.openai_api_key
    _notify_progress(progress_callback, "IN_PROGRESS", 10)
    uploaded_file_ids: list[str] = []
    try:
        existing_file_id: str | None = None
        mask_file_id: str | None = None
        if request.is_partial_edit:
            assert request.existing_texture_png is not None
            assert request.edit_mask_png is not None
            existing_file_id = _upload_openai_vision_png(
                api_key,
                request.existing_texture_png,
                "existing_texture.png",
                timeout_seconds,
                opener,
            )
            uploaded_file_ids.append(existing_file_id)
            _raise_if_cancelled(cancel_event)
            mask_file_id = _upload_openai_vision_png(
                api_key,
                _build_openai_edit_mask_png(request.edit_mask_png),
                "texture_edit_mask.png",
                timeout_seconds,
                opener,
            )
            uploaded_file_ids.append(mask_file_id)
            _raise_if_cancelled(cancel_event)
            _notify_progress(progress_callback, "IN_PROGRESS", 30)
        response = _request_json(
            Request(
                OPENAI_RESPONSES_ENDPOINT,
                data=_encode_json(
                    build_openai_responses_request_body(
                        request,
                        existing_texture_file_id=existing_file_id,
                        edit_mask_file_id=mask_file_id,
                    )
                ),
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
    finally:
        for file_id in reversed(uploaded_file_ids):
            _best_effort_delete_openai_file(
                api_key,
                file_id,
                timeout_seconds,
                opener,
            )


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
        existing_texture_png=request.existing_texture_png,
        edit_mask_png=request.edit_mask_png,
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


def _get_task_error_message(task: Mapping[str, Any]) -> str:
    task_error = task.get("task_error")
    if isinstance(task_error, dict):
        message = task_error.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    return "unknown provider error"


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


# ### Partial texture edit helpers ###
def composite_partial_texture_edit(
    generated_texture_png: bytes,
    existing_texture_png: bytes | None,
    edit_mask_png: bytes | None,
) -> bytes:
    """Composite a generated texture only where the white edit mask permits."""

    existing, edit_mask = _normalize_partial_edit_images(
        existing_texture_png,
        edit_mask_png,
    )
    if existing is None or edit_mask is None:
        try:
            return _validate_png(
                generated_texture_png,
                "Generated texture",
                MAX_OUTPUT_PNG_BYTES,
            )
        except (TypeError, ValueError) as error:
            raise SurfaceTextureTaskError(str(error)) from error

    try:
        base_image = _load_static_png(existing, "Existing texture").convert("RGBA")
        generated_image = _load_static_png(
            generated_texture_png,
            "Generated texture",
        ).convert("RGBA")
        editable_mask = _load_static_png(edit_mask, "Texture edit mask").convert(
            "L"
        )
        if generated_image.size != base_image.size:
            generated_image = Image.merge(
                "RGBA",
                tuple(
                    channel.resize(base_image.size, Image.Resampling.LANCZOS)
                    for channel in generated_image.split()
                ),
            )
        composited = Image.composite(
            generated_image,
            base_image,
            editable_mask,
        )
        return _encode_pil_png(composited, "Composited texture")
    except SurfaceTextureTaskError:
        raise
    except (OSError, ValueError) as error:
        raise SurfaceTextureTaskError(
            "The partial texture edit could not be composited."
        ) from error


def _normalize_partial_edit_images(
    existing_texture_png: bytes | None,
    edit_mask_png: bytes | None,
) -> tuple[bytes | None, bytes | None]:
    if existing_texture_png is None and edit_mask_png is None:
        return None, None
    if existing_texture_png is None or edit_mask_png is None:
        raise ValueError(
            "An existing texture and its edit mask must be provided together."
        )
    existing = _validate_png(
        existing_texture_png,
        "Existing texture",
        MAX_OUTPUT_PNG_BYTES,
    )
    mask = _validate_png(
        edit_mask_png,
        "Texture edit mask",
        MAX_REFERENCE_PNG_BYTES,
    )
    existing_image = _load_static_png(existing, "Existing texture")
    mask_image = _load_static_png(mask, "Texture edit mask").convert("L")
    if existing_image.size != mask_image.size:
        raise ValueError(
            "The existing texture and edit mask must have identical dimensions."
        )
    if mask_image.getbbox() is None:
        raise ValueError("The texture edit mask has no editable pixels.")
    return existing, _encode_pil_png(mask_image, "Texture edit mask")


def _build_openai_edit_mask_png(edit_mask_png: bytes | None) -> bytes:
    if edit_mask_png is None:
        raise ValueError("An OpenAI texture edit requires a mask.")
    editable_mask = _load_static_png(
        edit_mask_png,
        "Texture edit mask",
    ).convert("L")
    preserved_alpha = editable_mask.point(lambda value: 255 - value)
    openai_mask = Image.new("RGBA", editable_mask.size, (255, 255, 255, 255))
    openai_mask.putalpha(preserved_alpha)
    return _encode_pil_png(openai_mask, "OpenAI texture edit mask")


def _build_meshy_partial_edit_control_png(
    existing_texture_png: bytes,
    edit_mask_png: bytes,
) -> bytes:
    base_image = _load_static_png(
        existing_texture_png,
        "Existing texture",
    ).convert("RGBA")
    editable_mask = _load_static_png(
        edit_mask_png,
        "Texture edit mask",
    ).convert("L")
    highlight_strength = editable_mask.point(
        lambda value: round(value * 0.82)
    )
    magenta = Image.new("RGBA", base_image.size, (255, 0, 255, 255))
    control_image = Image.composite(magenta, base_image, highlight_strength)
    return _encode_pil_png(control_image, "Meshy texture edit control")


def _pack_meshy_partial_video_references(
    reference_pngs: Sequence[bytes],
    maximum_images: int,
) -> tuple[bytes, ...]:
    if len(reference_pngs) <= maximum_images:
        return tuple(reference_pngs)
    groups: list[list[bytes]] = [[] for _ in range(maximum_images)]
    for index, reference_png in enumerate(reference_pngs):
        groups[index % maximum_images].append(reference_png)
    return tuple(
        _build_horizontal_reference_sheet(group)
        for group in groups
        if group
    )


def _build_horizontal_reference_sheet(reference_pngs: Sequence[bytes]) -> bytes:
    images = [
        _load_static_png(reference_png, "Video texture reference").convert("RGBA")
        for reference_png in reference_pngs
    ]
    scale = min(
        1.0,
        MAX_PACKED_REFERENCE_EDGE_PIXELS / sum(image.width for image in images),
        MAX_PACKED_REFERENCE_EDGE_PIXELS / max(image.height for image in images),
    )
    if scale < 1.0:
        images = [
            image.resize(
                (
                    max(1, int(image.width * scale)),
                    max(1, int(image.height * scale)),
                ),
                Image.Resampling.LANCZOS,
            )
            for image in images
        ]
    sheet_width = sum(image.width for image in images)
    sheet_height = max(image.height for image in images)
    sheet = Image.new("RGBA", (sheet_width, sheet_height), (0, 0, 0, 0))
    left = 0
    for image in images:
        top = (sheet_height - image.height) // 2
        sheet.alpha_composite(image, (left, top))
        left += image.width
    return _encode_pil_png(sheet, "Packed video texture references")


def _load_static_png(image_png: bytes, label: str) -> Image.Image:
    try:
        validated = _validate_png(image_png, label, MAX_OUTPUT_PNG_BYTES)
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(validated)) as source_image:
                if int(getattr(source_image, "n_frames", 1)) != 1:
                    raise ValueError(f"{label} must be a static image.")
                source_image.load()
                return source_image.copy()
    except ValueError:
        raise
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        OSError,
        SyntaxError,
        UnidentifiedImageError,
    ) as error:
        raise ValueError(f"{label} is not a valid PNG image.") from error


def _encode_pil_png(image: Image.Image, label: str) -> bytes:
    output = BytesIO()
    try:
        image.save(output, format="PNG", compress_level=6)
        return _validate_png(output.getvalue(), label, MAX_OUTPUT_PNG_BYTES)
    except (OSError, ValueError) as error:
        raise ValueError(f"{label} could not be encoded as PNG.") from error


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


def _normalize_openai_file_id(file_id: object, label: str) -> str:
    if not isinstance(file_id, str):
        raise SurfaceTextureRequestError(
            f"OpenAI returned an invalid {label} file identifier."
        )
    normalized = file_id.strip()
    if (
        not normalized
        or len(normalized) > 256
        or any(ord(character) < 33 for character in normalized)
    ):
        raise SurfaceTextureRequestError(
            f"OpenAI returned an invalid {label} file identifier."
        )
    return normalized


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


def _build_surface_texture_prompt(
    prompt: str,
    *,
    partial_edit: bool = False,
) -> str:
    prefix = PARTIAL_TEXTURE_PROMPT_PREFIX if partial_edit else ""
    return prefix + SURFACE_TEXTURE_PROMPT_PREFIX + prompt


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
