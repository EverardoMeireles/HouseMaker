# ### Imports ###
from __future__ import annotations

import base64
import json
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen


# ### Constants ###
MESHY_IMAGE_TO_3D_ENDPOINT = "https://api.meshy.ai/openapi/v1/image-to-3d"
MESHY_RETEXTURE_ENDPOINT = "https://api.meshy.ai/openapi/v1/retexture"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
DEFAULT_MAX_POLLS = 360
DEFAULT_SMART_TOPOLOGY_TARGET_POLYCOUNT = 4_000
MIN_SMART_TOPOLOGY_TARGET_POLYCOUNT = 100
MAX_SMART_TOPOLOGY_TARGET_POLYCOUNT = 15_000
MAX_JSON_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_GLB_DOWNLOAD_BYTES = 128 * 1024 * 1024
TERMINAL_TASK_STATUSES = frozenset({"SUCCEEDED", "FAILED", "CANCELED"})
ACTIVE_TASK_STATUSES = frozenset({"PENDING", "IN_PROGRESS"})
KNOWN_TASK_STATUSES = TERMINAL_TASK_STATUSES | ACTIVE_TASK_STATUSES
RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
MAX_RETEXTURE_REFERENCE_IMAGES = 4
SINGLE_IMAGE_RETEXTURE_AI_MODEL = "meshy-6"
MULTIVIEW_RETEXTURE_AI_MODEL = "meshy-7"


# ### Data models ###
@dataclass(frozen=True)
class MeshyGenerationResult:
    task_id: str
    glb_bytes: bytes
    name: str = "Meshy object"


# ### Exceptions ###
class MeshyGenerationError(RuntimeError):
    """Base exception for safe, user-displayable Meshy failures."""


class MeshyRequestError(MeshyGenerationError):
    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class MeshyTaskError(MeshyGenerationError):
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
TaskGetter = Callable[..., dict[str, Any]]


# ### Request builders ###
def build_image_to_3d_request_body(
    image_png: bytes,
    target_polycount: int = DEFAULT_SMART_TOPOLOGY_TARGET_POLYCOUNT,
    should_texture: bool = True,
    enable_pbr: bool = False,
) -> dict[str, Any]:
    """Build a Smart Topology Meshy Image-to-3D request."""

    normalized_image = _normalize_binary_payload(
        image_png,
        empty_message="The selected object image is empty.",
    )
    if not isinstance(should_texture, bool):
        raise ValueError("Meshy should_texture must be a boolean.")
    if not isinstance(enable_pbr, bool):
        raise ValueError("Meshy enable_pbr must be a boolean.")
    if enable_pbr and not should_texture:
        raise ValueError(
            "Meshy PBR maps require texture generation to be enabled."
        )
    normalized_polycount = _normalize_smart_topology_target_polycount(
        target_polycount
    )
    body: dict[str, Any] = {
        "image_url": _build_data_uri("image/png", normalized_image),
        "model_type": "smart-topology",
        "ai_model": "meshy-t2",
        "should_texture": should_texture,
        "target_polycount": normalized_polycount,
        "moderation": False,
        "target_formats": ["glb"],
        "auto_size": True,
        "origin_at": "bottom",
    }
    if should_texture:
        body.update(
            {
                "enable_pbr": enable_pbr,
                "texture_resolution": "2k",
            }
        )
    return body


def build_retexture_request_body(
    model_glb: bytes,
    reference_images_png: Sequence[bytes],
    enable_original_uv: bool = False,
    enable_pbr: bool = False,
) -> dict[str, Any]:
    """Build a Retexture request for a locally post-processed GLB."""

    normalized_model = _normalize_binary_payload(
        model_glb,
        empty_message="The post-processed GLB is empty.",
    )
    if not isinstance(enable_original_uv, bool):
        raise ValueError("Meshy enable_original_uv must be a boolean.")
    if not isinstance(enable_pbr, bool):
        raise ValueError("Meshy enable_pbr must be a boolean.")
    image_data_uris = _normalize_retexture_reference_images(reference_images_png)
    body: dict[str, Any] = {
        "model_url": _build_data_uri("application/octet-stream", normalized_model),
        "enable_original_uv": enable_original_uv,
        "enable_pbr": enable_pbr,
        "texture_resolution": "2k",
        "target_formats": ["glb"],
    }
    if len(image_data_uris) == 1:
        body.update(
            {
                "image_style_url": image_data_uris[0],
                "ai_model": SINGLE_IMAGE_RETEXTURE_AI_MODEL,
            }
        )
    else:
        body.update(
            {
                "multiview_image_urls": image_data_uris,
                "ai_model": MULTIVIEW_RETEXTURE_AI_MODEL,
            }
        )
    return body


def _normalize_retexture_reference_images(
    reference_images_png: Sequence[bytes],
) -> list[str]:
    if isinstance(reference_images_png, (str, bytes, bytearray, memoryview)):
        raise ValueError("Meshy reference images must be provided as a sequence.")
    try:
        images = list(reference_images_png)
    except TypeError as error:
        raise ValueError(
            "Meshy reference images must be provided as a sequence."
        ) from error
    if not 1 <= len(images) <= MAX_RETEXTURE_REFERENCE_IMAGES:
        raise ValueError(
            "Meshy Retexture requires between 1 and "
            f"{MAX_RETEXTURE_REFERENCE_IMAGES} reference images."
        )
    return [
        _build_data_uri(
            "image/png",
            _normalize_binary_payload(
                image,
                empty_message=f"Meshy reference image {index} is empty.",
            ),
        )
        for index, image in enumerate(images, start=1)
    ]


def _normalize_smart_topology_target_polycount(target_polycount: int) -> int:
    """Validate Meshy's current Smart Topology T2 face-count range."""

    if isinstance(target_polycount, bool) or not isinstance(target_polycount, int):
        raise ValueError("Meshy target polycount must be an integer.")
    if not (
        MIN_SMART_TOPOLOGY_TARGET_POLYCOUNT
        <= target_polycount
        <= MAX_SMART_TOPOLOGY_TARGET_POLYCOUNT
    ):
        raise ValueError(
            "Meshy Smart Topology target polycount must be between "
            f"{MIN_SMART_TOPOLOGY_TARGET_POLYCOUNT:,} and "
            f"{MAX_SMART_TOPOLOGY_TARGET_POLYCOUNT:,}."
        )
    return target_polycount


# ### Task API ###
def create_image_to_3d_task(
    api_key: str,
    image_png: bytes,
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    opener: UrlOpenFunction | None = None,
    target_polycount: int = DEFAULT_SMART_TOPOLOGY_TARGET_POLYCOUNT,
    should_texture: bool = True,
    enable_pbr: bool = False,
) -> str:
    normalized_key = _require_api_key(api_key)
    payload = build_image_to_3d_request_body(
        image_png,
        target_polycount=target_polycount,
        should_texture=should_texture,
        enable_pbr=enable_pbr,
    )
    response = _request_json(
        Request(
            MESHY_IMAGE_TO_3D_ENDPOINT,
            data=_encode_json(payload),
            headers={
                "Authorization": f"Bearer {normalized_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        ),
        timeout_seconds=timeout_seconds,
        opener=opener,
        secrets=(normalized_key,),
    )
    task_id = response.get("result")
    if not isinstance(task_id, str) or not task_id.strip():
        raise MeshyRequestError("Meshy returned an invalid task identifier.")
    return task_id.strip()


def get_image_to_3d_task(
    api_key: str,
    task_id: str,
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    opener: UrlOpenFunction | None = None,
) -> dict[str, Any]:
    normalized_key = _require_api_key(api_key)
    normalized_task_id = _require_task_id(task_id)
    response = _request_json(
        Request(
            f"{MESHY_IMAGE_TO_3D_ENDPOINT}/{quote(normalized_task_id, safe='')}",
            headers={"Authorization": f"Bearer {normalized_key}"},
            method="GET",
        ),
        timeout_seconds=timeout_seconds,
        opener=opener,
        secrets=(normalized_key,),
    )
    return response


def wait_for_image_to_3d_task(
    api_key: str,
    task_id: str,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    max_polls: int = DEFAULT_MAX_POLLS,
    opener: UrlOpenFunction | None = None,
    sleep: SleepFunction = time.sleep,
    progress_callback: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    return _wait_for_model_task(
        api_key=api_key,
        task_id=task_id,
        get_task=get_image_to_3d_task,
        poll_interval_seconds=poll_interval_seconds,
        max_polls=max_polls,
        opener=opener,
        sleep=sleep,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )


# ### Retexture task API ###
def create_retexture_task(
    api_key: str,
    model_glb: bytes,
    reference_images_png: Sequence[bytes],
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    opener: UrlOpenFunction | None = None,
    enable_original_uv: bool = False,
    enable_pbr: bool = False,
) -> str:
    normalized_key = _require_api_key(api_key)
    payload = build_retexture_request_body(
        model_glb=model_glb,
        reference_images_png=reference_images_png,
        enable_original_uv=enable_original_uv,
        enable_pbr=enable_pbr,
    )
    response = _request_json(
        Request(
            MESHY_RETEXTURE_ENDPOINT,
            data=_encode_json(payload),
            headers={
                "Authorization": f"Bearer {normalized_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        ),
        timeout_seconds=timeout_seconds,
        opener=opener,
        secrets=(normalized_key,),
    )
    task_id = response.get("result")
    if not isinstance(task_id, str) or not task_id.strip():
        raise MeshyRequestError("Meshy returned an invalid task identifier.")
    return task_id.strip()


def get_retexture_task(
    api_key: str,
    task_id: str,
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    opener: UrlOpenFunction | None = None,
) -> dict[str, Any]:
    normalized_key = _require_api_key(api_key)
    normalized_task_id = _require_task_id(task_id)
    return _request_json(
        Request(
            f"{MESHY_RETEXTURE_ENDPOINT}/{quote(normalized_task_id, safe='')}",
            headers={"Authorization": f"Bearer {normalized_key}"},
            method="GET",
        ),
        timeout_seconds=timeout_seconds,
        opener=opener,
        secrets=(normalized_key,),
    )


def wait_for_retexture_task(
    api_key: str,
    task_id: str,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    max_polls: int = DEFAULT_MAX_POLLS,
    opener: UrlOpenFunction | None = None,
    sleep: SleepFunction = time.sleep,
    progress_callback: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Poll a Retexture task until its final GLB is ready."""

    return _wait_for_model_task(
        api_key=api_key,
        task_id=task_id,
        get_task=get_retexture_task,
        poll_interval_seconds=poll_interval_seconds,
        max_polls=max_polls,
        opener=opener,
        sleep=sleep,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )


# ### Artifact API ###
def download_glb(
    url: str,
    max_bytes: int = MAX_GLB_DOWNLOAD_BYTES,
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    opener: UrlOpenFunction | None = None,
) -> bytes:
    normalized_url = str(url).strip()
    parsed_url = urlparse(normalized_url)
    if parsed_url.scheme.lower() != "https" or not parsed_url.netloc:
        raise ValueError("Meshy GLB downloads must use an HTTPS URL.")
    normalized_max_bytes = int(max_bytes)
    if normalized_max_bytes <= 0:
        raise ValueError("The Meshy GLB size limit must be positive.")

    request = Request(
        normalized_url,
        headers={"Accept": "model/gltf-binary,application/octet-stream"},
        method="GET",
    )
    url_opener = urlopen if opener is None else opener
    try:
        with url_opener(request, timeout=float(timeout_seconds)) as response:
            payload = response.read(normalized_max_bytes + 1)
    except HTTPError as error:
        raise _build_http_error(error, secrets=()) from error
    except (URLError, TimeoutError, OSError) as error:
        raise MeshyRequestError(
            "Unable to download the Meshy GLB artifact.",
            retryable=True,
        ) from error

    if len(payload) > normalized_max_bytes:
        raise MeshyRequestError("The Meshy GLB artifact is too large.")
    if not payload:
        raise MeshyRequestError("Meshy returned an empty GLB artifact.")
    return bytes(payload)


def request_image_to_3d_model(
    api_key: str,
    image_png: bytes,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    max_polls: int = DEFAULT_MAX_POLLS,
    opener: UrlOpenFunction | None = None,
    sleep: SleepFunction = time.sleep,
    progress_callback: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
    target_polycount: int = DEFAULT_SMART_TOPOLOGY_TARGET_POLYCOUNT,
    should_texture: bool = True,
    enable_pbr: bool = False,
) -> MeshyGenerationResult:
    task_id = create_image_to_3d_task(
        api_key=api_key,
        image_png=image_png,
        opener=opener,
        target_polycount=target_polycount,
        should_texture=should_texture,
        enable_pbr=enable_pbr,
    )
    task = wait_for_image_to_3d_task(
        api_key=api_key,
        task_id=task_id,
        poll_interval_seconds=poll_interval_seconds,
        max_polls=max_polls,
        opener=opener,
        sleep=sleep,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )
    _raise_if_cancelled(cancel_event, task_id)
    glb_bytes = download_glb(_get_glb_url(task, task_id), opener=opener)
    return MeshyGenerationResult(task_id=task_id, glb_bytes=glb_bytes)


def request_retextured_model(
    api_key: str,
    model_glb: bytes,
    reference_images_png: Sequence[bytes],
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    max_polls: int = DEFAULT_MAX_POLLS,
    opener: UrlOpenFunction | None = None,
    sleep: SleepFunction = time.sleep,
    progress_callback: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
    enable_original_uv: bool = False,
    enable_pbr: bool = False,
) -> MeshyGenerationResult:
    if cancel_event is not None and cancel_event.is_set():
        raise MeshyTaskError(
            "Meshy Retexture was canceled locally before task submission."
        )
    task_id = create_retexture_task(
        api_key=api_key,
        model_glb=model_glb,
        reference_images_png=reference_images_png,
        opener=opener,
        enable_original_uv=enable_original_uv,
        enable_pbr=enable_pbr,
    )
    task = wait_for_retexture_task(
        api_key=api_key,
        task_id=task_id,
        poll_interval_seconds=poll_interval_seconds,
        max_polls=max_polls,
        opener=opener,
        sleep=sleep,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )
    _raise_if_cancelled(cancel_event, task_id)
    glb_bytes = download_glb(_get_glb_url(task, task_id), opener=opener)
    return MeshyGenerationResult(
        task_id=task_id,
        glb_bytes=glb_bytes,
        name="Meshy textured object",
    )


# ### HTTP helpers ###
def _request_json(
    request: Request,
    timeout_seconds: float,
    opener: UrlOpenFunction | None,
    secrets: tuple[str, ...],
) -> dict[str, Any]:
    url_opener = urlopen if opener is None else opener
    try:
        with url_opener(request, timeout=float(timeout_seconds)) as response:
            payload = response.read(MAX_JSON_RESPONSE_BYTES + 1)
    except HTTPError as error:
        raise _build_http_error(error, secrets=secrets) from error
    except (URLError, TimeoutError, OSError) as error:
        reason = str(getattr(error, "reason", None) or error).strip()
        message = f"Unable to reach Meshy: {reason or 'network unavailable'}"
        raise MeshyRequestError(
            _redact(message, secrets),
            retryable=True,
        ) from error

    if len(payload) > MAX_JSON_RESPONSE_BYTES:
        raise MeshyRequestError("Meshy returned an oversized response.")
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise MeshyRequestError("Meshy returned invalid JSON.") from error
    if not isinstance(decoded, dict):
        raise MeshyRequestError("Meshy returned an invalid response object.")
    return decoded


def _build_http_error(
    error: HTTPError,
    secrets: tuple[str, ...],
) -> MeshyRequestError:
    status_code = int(error.code)
    detail = "request rejected"
    try:
        payload = error.read(MAX_JSON_RESPONSE_BYTES + 1)
        decoded = json.loads(payload.decode("utf-8"))
        if isinstance(decoded, dict):
            raw_message = decoded.get("message")
            if isinstance(raw_message, str) and raw_message.strip():
                detail = raw_message.strip()
    except (OSError, UnicodeError, json.JSONDecodeError, AttributeError):
        pass
    return MeshyRequestError(
        _redact(f"Meshy request failed ({status_code}): {detail}", secrets),
        status_code=status_code,
        retryable=status_code in RETRYABLE_HTTP_STATUS_CODES,
    )


def _encode_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


# ### Task helpers ###
def _wait_for_model_task(
    api_key: str,
    task_id: str,
    get_task: TaskGetter,
    poll_interval_seconds: float,
    max_polls: int,
    opener: UrlOpenFunction | None,
    sleep: SleepFunction,
    progress_callback: ProgressCallback | None,
    cancel_event: threading.Event | None,
) -> dict[str, Any]:
    normalized_task_id = _require_task_id(task_id)
    normalized_max_polls = int(max_polls)
    if normalized_max_polls <= 0:
        raise ValueError("Meshy max polls must be positive.")
    poll_interval = float(poll_interval_seconds)
    if not math.isfinite(poll_interval) or poll_interval < 0.0:
        raise ValueError("Meshy poll interval must be a finite non-negative value.")

    for poll_index in range(normalized_max_polls):
        _raise_if_cancelled(cancel_event, normalized_task_id)
        task = get_task(
            api_key=api_key,
            task_id=normalized_task_id,
            opener=opener,
        )
        status = _get_task_status(task, normalized_task_id)
        progress = _get_task_progress(task)
        if progress_callback is not None:
            progress_callback(status, progress)

        if status == "SUCCEEDED":
            _get_glb_url(task, normalized_task_id)
            return task
        if status in {"FAILED", "CANCELED"}:
            detail = _redact(
                _get_task_error_message(task),
                (str(api_key).strip(),),
            )
            raise MeshyTaskError(
                f"Meshy task {normalized_task_id} {status.lower()}: {detail}"
            )
        if poll_index + 1 < normalized_max_polls:
            _interruptible_sleep(
                poll_interval,
                sleep,
                cancel_event,
                normalized_task_id,
            )

    raise MeshyTaskError(
        f"Meshy task {normalized_task_id} timed out before completion."
    )


def _require_api_key(api_key: str) -> str:
    normalized_key = str(api_key).strip()
    if not normalized_key:
        raise ValueError("A Meshy API key is required.")
    return normalized_key


def _require_task_id(task_id: str) -> str:
    normalized_task_id = str(task_id).strip()
    if not normalized_task_id or len(normalized_task_id) > 256:
        raise ValueError("The Meshy task identifier is invalid.")
    return normalized_task_id


def _get_task_status(task: Mapping[str, Any], task_id: str) -> str:
    status = task.get("status")
    if not isinstance(status, str) or status.upper() not in KNOWN_TASK_STATUSES:
        raise MeshyTaskError(f"Meshy task {task_id} returned an invalid status.")
    return status.upper()


def _get_task_progress(task: Mapping[str, Any]) -> int:
    progress = task.get("progress", 0)
    if isinstance(progress, bool):
        return 0
    try:
        return min(max(int(progress), 0), 100)
    except (TypeError, ValueError, OverflowError):
        return 0


def _get_task_error_message(task: Mapping[str, Any]) -> str:
    task_error = task.get("task_error")
    if isinstance(task_error, dict):
        message = task_error.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    return "unknown provider error"


def _get_glb_url(task: Mapping[str, Any], task_id: str) -> str:
    model_urls = task.get("model_urls")
    glb_url = model_urls.get("glb") if isinstance(model_urls, dict) else None
    if not isinstance(glb_url, str) or not glb_url.strip():
        raise MeshyTaskError(
            f"Meshy task {task_id} succeeded without a GLB artifact."
        )
    return glb_url.strip()


def _interruptible_sleep(
    seconds: float,
    sleep: SleepFunction,
    cancel_event: threading.Event | None,
    task_id: str,
) -> None:
    if cancel_event is None or sleep is not time.sleep:
        sleep(seconds)
        _raise_if_cancelled(cancel_event, task_id)
        return
    if cancel_event.wait(seconds):
        _raise_if_cancelled(cancel_event, task_id)


def _raise_if_cancelled(
    cancel_event: threading.Event | None,
    task_id: str,
) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise MeshyTaskError(f"Meshy task {task_id} was canceled locally.")


def _redact(message: str, secrets: tuple[str, ...]) -> str:
    redacted = str(message)
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "[redacted]")
    return redacted


# ### Binary payload helpers ###
def _normalize_binary_payload(payload: bytes, empty_message: str) -> bytes:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ValueError("Meshy binary inputs must contain bytes.")
    normalized_payload = bytes(payload)
    if not normalized_payload:
        raise ValueError(empty_message)
    return normalized_payload


def _build_data_uri(mime_type: str, payload: bytes) -> str:
    encoded_payload = base64.b64encode(payload).decode("ascii")
    return f"data:{mime_type};base64,{encoded_payload}"
