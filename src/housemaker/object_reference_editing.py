# ### Imports ###
from __future__ import annotations

import base64
import json
import queue
import threading
import uuid
from collections.abc import Callable
from typing import Any, Protocol
from urllib.request import Request, urlopen

import cv2
import numpy as np

from housemaker.plan_image_correction import (
    PlanImageCorrectionCancelled,
    PlanImageCorrectionInferenceError,
    openai_responses_image_edit,
)
from housemaker.qwen_plan_image_correction import (
    QwenPlanCorrectionCancelled,
    QwenPlanCorrectionError,
    create_default_qwen_object_reference_editor,
)

# ### Provider constants ###
OPENAI_IMAGE_EDIT_URL = "https://api.openai.com/v1/images/edits"
OBJECT_REFERENCE_EDIT_MODEL_GPT_IMAGE_2_5_SUNBURST = (
    "gpt-image-2.5-sunburst"
)
OBJECT_REFERENCE_EDIT_MODEL_GPT_IMAGE_2_5_FLARE = "gpt-image-2.5-flare"
OBJECT_REFERENCE_EDIT_MODEL_GPT_IMAGE_2 = "gpt-image-2"
OBJECT_REFERENCE_EDIT_MODEL_GPT_5_6_TERRA = "gpt-5.6-terra"
OBJECT_REFERENCE_EDIT_MODEL_GPT_5_6_LUNA = "gpt-5.6-luna"
OBJECT_REFERENCE_EDIT_MODEL_QWEN_IMAGE_2_1 = "Qwen-Image-2.1"
OBJECT_REFERENCE_EDIT_MODEL_OPTIONS = (
    (
        "GPT Image 2.5 Sunburst",
        OBJECT_REFERENCE_EDIT_MODEL_GPT_IMAGE_2_5_SUNBURST,
    ),
    ("GPT Image 2.5 Flare", OBJECT_REFERENCE_EDIT_MODEL_GPT_IMAGE_2_5_FLARE),
    ("GPT Image 2", OBJECT_REFERENCE_EDIT_MODEL_GPT_IMAGE_2),
    ("GPT-5.6 Terra", OBJECT_REFERENCE_EDIT_MODEL_GPT_5_6_TERRA),
    ("GPT-5.6 Luna", OBJECT_REFERENCE_EDIT_MODEL_GPT_5_6_LUNA),
    ("Qwen-Image-2.1", OBJECT_REFERENCE_EDIT_MODEL_QWEN_IMAGE_2_1),
)
SUPPORTED_OBJECT_REFERENCE_EDIT_MODELS = tuple(
    model_id for _label, model_id in OBJECT_REFERENCE_EDIT_MODEL_OPTIONS
)
OPENAI_IMAGE_API_REFERENCE_EDIT_MODELS = frozenset(
    {
        OBJECT_REFERENCE_EDIT_MODEL_GPT_IMAGE_2_5_SUNBURST,
        OBJECT_REFERENCE_EDIT_MODEL_GPT_IMAGE_2_5_FLARE,
        OBJECT_REFERENCE_EDIT_MODEL_GPT_IMAGE_2,
    }
)
OPENAI_RESPONSES_REFERENCE_EDIT_MODELS = frozenset(
    {
        OBJECT_REFERENCE_EDIT_MODEL_GPT_5_6_TERRA,
        OBJECT_REFERENCE_EDIT_MODEL_GPT_5_6_LUNA,
    }
)
DEFAULT_OBJECT_REFERENCE_EDIT_MODEL = OBJECT_REFERENCE_EDIT_MODEL_GPT_IMAGE_2
OBJECT_REFERENCE_EDIT_QUALITY = "high"
OBJECT_REFERENCE_EDIT_BACKGROUND = "transparent"
OBJECT_REFERENCE_EDIT_OUTPUT_FORMAT = "png"
OBJECT_REFERENCE_EDIT_OUTPUT_SIZE = (1024, 1024)
OBJECT_REFERENCE_EDIT_OUTPUT_SHAPE = (1024, 1024, 4)
SUPPORTED_OUTPUT_SIZES = (OBJECT_REFERENCE_EDIT_OUTPUT_SIZE,)
OPENAI_NETWORK_TIMEOUT_SECONDS = 300.0
CANCELLATION_POLL_SECONDS = 0.05
MAX_PROMPT_CHARACTERS = 4_000
MAX_SOURCE_PIXELS = 25_000_000
MAX_PNG_BYTES = 50 * 1024 * 1024
MAX_RESPONSE_BYTES = 64 * 1024 * 1024
QWEN_OBJECT_REFERENCE_CONTENT_SCALE = 0.65

OBJECT_REFERENCE_GENERATION_INSTRUCTION = """The supplied image contains one foreground object isolated with the user's mask.
Use that object as the visual reference and generate a complete new 1024x1024 image
that applies this requested semantic change: {request}

Keep the same object, viewpoint, perspective, pose, proportions, scale, lighting,
and every source detail not explicitly affected by the request. Plausibly
reconstruct any object surface hidden by the removed or changed item. Transparent
source pixels are background, not an edit boundary. Center the complete object and
keep it fully inside the square. Return one isolated object on a transparent
background, with no text, border, scenery, or additional object."""
OPENAI_RESPONSES_OBJECT_REFERENCE_GENERATION_INSTRUCTION = """The supplied image contains one foreground object isolated with the user's mask.
Use that object as the visual reference and generate a complete new 1024x1024 image
that applies this requested semantic change: {request}

Keep the same object, viewpoint, perspective, pose, proportions, scale, lighting,
and every source detail not explicitly affected by the request. Plausibly
reconstruct any object surface hidden by the removed or changed item. Transparent
source pixels are background, not an edit boundary. Center the complete object and
keep it fully inside the square. Return one isolated object on a plain neutral
opaque background, with no text, border, scenery, or additional object."""
QWEN_OBJECT_REFERENCE_GENERATION_INSTRUCTION = """This is an RGBA image with transparency.
The non-transparent pixels are a user selection and may show only one visible
surface or part of an object. Treat that selection as visual evidence, not as the
requested output boundary, crop, position, scale, or silhouette. Infer and
reconstruct the complete object beyond the selection, then apply these instructions:

{request}

Generate a genuinely new 1024x1024 composition rather than returning the selected
fragment. Center the complete object, keep every part fully visible, and leave
comfortable transparent space around it. The image has an alpha channel and the
background is transparent."""


# ### Public exceptions ###
class ObjectReferenceEditingError(RuntimeError):
    """Failure text safe to display in the application UI."""


class ObjectReferenceEditingCancelled(ObjectReferenceEditingError):
    pass


# ### Public protocols ###
CancellationCheck = Callable[[], bool]


class ObjectReferenceImageEditor(Protocol):
    def __call__(
        self,
        source_png: bytes,
        *,
        api_key: str,
        model: str,
        prompt: str,
        output_size: tuple[int, int],
        cancellation_check: CancellationCheck | None,
    ) -> bytes: ...


class UrlOpenFunction(Protocol):
    def __call__(self, request: Request, **kwargs: Any) -> object: ...


# ### Public editing API ###
def edit_object_reference(
    source_bgra: np.ndarray,
    prompt: str,
    *,
    api_key: str,
    model: str = DEFAULT_OBJECT_REFERENCE_EDIT_MODEL,
    cancellation_check: CancellationCheck | None = None,
    image_editor: ObjectReferenceImageEditor | None = None,
) -> np.ndarray:
    """Generate one square reference from the object isolated by source alpha."""

    normalized_model = _normalize_model(model)
    source, normalized_prompt, key = _validate_edit_request(
        source_bgra,
        prompt,
        api_key,
        require_api_key=object_reference_edit_model_requires_api_key(
            normalized_model
        ),
    )
    _raise_if_cancelled(cancellation_check)
    uses_qwen = normalized_model == OBJECT_REFERENCE_EDIT_MODEL_QWEN_IMAGE_2_1
    canvas_source = _crop_to_visible_alpha(source) if uses_qwen else source
    source_canvas = _build_square_reference_canvas(
        canvas_source,
        maximum_content_scale=(
            QWEN_OBJECT_REFERENCE_CONTENT_SCALE
            if uses_qwen
            else 1.0
        ),
    )
    source_png = _encode_png(source_canvas)
    _raise_if_cancelled(cancellation_check)

    editor = image_editor or _default_image_editor(normalized_model)
    prompt_template = (
        OPENAI_RESPONSES_OBJECT_REFERENCE_GENERATION_INSTRUCTION
        if normalized_model in OPENAI_RESPONSES_REFERENCE_EDIT_MODELS
        else OBJECT_REFERENCE_GENERATION_INSTRUCTION
    )
    provider_prompt = prompt_template.format(
        request=normalized_prompt,
    )
    try:
        editor_arguments = {
            "api_key": key,
            "model": normalized_model,
            "prompt": provider_prompt,
            "output_size": OBJECT_REFERENCE_EDIT_OUTPUT_SIZE,
            "cancellation_check": cancellation_check,
        }
        if (
            image_editor is None
            and normalized_model
            == OBJECT_REFERENCE_EDIT_MODEL_QWEN_IMAGE_2_1
        ):
            # The local backend cooperatively stops its child process. Keep it
            # on this worker so cancellation also finishes cleanup and releases
            # the global Qwen inference lock before the worker exits.
            result_png = editor(
                source_png,
                **editor_arguments,
            )
        else:
            result_png = _invoke_editor_cancellably(
                editor,
                source_png,
                **editor_arguments,
            )
    except ObjectReferenceEditingCancelled:
        raise
    except ObjectReferenceEditingError:
        raise
    except Exception:  # noqa: BLE001 - provider details and secrets stay private.
        raise ObjectReferenceEditingError(
            "The AI object reference edit failed."
        ) from None

    _raise_if_cancelled(cancellation_check)
    provider_canvas = _decode_provider_png(
        result_png,
        OBJECT_REFERENCE_EDIT_OUTPUT_SIZE,
    )
    _raise_if_cancelled(cancellation_check)
    return np.ascontiguousarray(provider_canvas.copy())


def object_reference_edit_model_requires_api_key(model: str) -> bool:
    """Return whether a supported reference editor needs OpenAI credentials."""

    return _normalize_model(model) != OBJECT_REFERENCE_EDIT_MODEL_QWEN_IMAGE_2_1


def openai_object_reference_edit(
    source_png: bytes,
    *,
    api_key: str,
    model: str = DEFAULT_OBJECT_REFERENCE_EDIT_MODEL,
    prompt: str,
    output_size: tuple[int, int],
    cancellation_check: CancellationCheck | None,
    opener: UrlOpenFunction | None = None,
) -> bytes:
    """Generate a square reference from one isolated GPT Image input."""

    key = api_key.strip() if isinstance(api_key, str) else ""
    if not key:
        raise ObjectReferenceEditingError(
            "Add an OpenAI API key in Settings before editing a reference."
        )
    normalized_model = _normalize_openai_image_api_model(model)
    normalized_prompt = _normalize_prompt(prompt)
    size = _validate_supported_output_size(output_size)
    source_bytes = _validate_png_payload(source_png, "source")
    _raise_if_cancelled(cancellation_check)

    body, content_type = build_openai_object_reference_edit_multipart(
        source_bytes,
        normalized_prompt,
        model=normalized_model,
        output_size=size,
    )
    request = Request(
        OPENAI_IMAGE_EDIT_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": content_type,
            "User-Agent": "HouseMaker/1.0",
        },
    )
    open_request = opener or urlopen
    try:
        with open_request(
            request,
            timeout=OPENAI_NETWORK_TIMEOUT_SECONDS,
        ) as response:
            raw_response = _read_response_limited(response, cancellation_check)
        parsed = json.loads(raw_response.decode("utf-8"))
        encoded = parsed["data"][0]["b64_json"]
        if not isinstance(encoded, str) or not encoded:
            raise ValueError
        maximum_encoded_characters = ((MAX_RESPONSE_BYTES + 2) // 3) * 4
        if len(encoded) > maximum_encoded_characters:
            raise ValueError
        decoded = base64.b64decode(encoded, validate=True)
        if not decoded or len(decoded) > MAX_RESPONSE_BYTES:
            raise ValueError
        return decoded
    except ObjectReferenceEditingCancelled:
        raise
    except Exception:  # noqa: BLE001 - never expose transport/provider diagnostics.
        raise ObjectReferenceEditingError(
            "The OpenAI image edit did not return a usable image."
        ) from None


def openai_responses_object_reference_edit(
    source_png: bytes,
    *,
    api_key: str,
    model: str,
    prompt: str,
    output_size: tuple[int, int],
    cancellation_check: CancellationCheck | None,
) -> bytes:
    """Edit an isolated reference through a GPT-5.6 Responses model."""

    normalized_model = _normalize_openai_responses_model(model)
    source_bytes = _validate_png_payload(source_png, "source")
    size = _validate_supported_output_size(output_size)
    _raise_if_cancelled(cancellation_check)
    try:
        return openai_responses_image_edit(
            source_bytes,
            model=normalized_model,
            api_key=api_key,
            prompt=_normalize_prompt(prompt),
            output_size=size,
            cancellation_check=cancellation_check,
            background="opaque",
            action="generate",
        )
    except PlanImageCorrectionCancelled:
        raise ObjectReferenceEditingCancelled(
            "Object reference editing was cancelled."
        ) from None
    except PlanImageCorrectionInferenceError as error:
        raise ObjectReferenceEditingError(str(error)) from None
    except ObjectReferenceEditingCancelled:
        raise
    except Exception:  # noqa: BLE001 - provider details remain private.
        raise ObjectReferenceEditingError(
            "The OpenAI image edit did not return a usable image."
        ) from None


def qwen_object_reference_edit(
    source_png: bytes,
    *,
    api_key: str,
    model: str,
    prompt: str,
    output_size: tuple[int, int],
    cancellation_check: CancellationCheck | None,
) -> bytes:
    """Edit an isolated reference with the installed local Qwen model."""

    del api_key
    normalized_model = _normalize_model(model)
    if normalized_model != OBJECT_REFERENCE_EDIT_MODEL_QWEN_IMAGE_2_1:
        raise ObjectReferenceEditingError(
            "Choose the Qwen-Image-2.1 object reference editing model."
        )
    source_bytes = _validate_png_payload(source_png, "source")
    size = _validate_supported_output_size(output_size)
    normalized_prompt = _normalize_prompt(prompt)
    _raise_if_cancelled(cancellation_check)
    try:
        editor = create_default_qwen_object_reference_editor()
        return editor(
            source_bytes,
            api_key="",
            prompt=QWEN_OBJECT_REFERENCE_GENERATION_INSTRUCTION.format(
                request=normalized_prompt,
            ),
            output_size=size,
            cancellation_check=cancellation_check,
        )
    except QwenPlanCorrectionCancelled:
        raise ObjectReferenceEditingCancelled(
            "Object reference editing was cancelled."
        ) from None
    except QwenPlanCorrectionError as error:
        raise ObjectReferenceEditingError(str(error)) from None
    except ObjectReferenceEditingCancelled:
        raise
    except Exception:  # noqa: BLE001 - local runtime details remain private.
        raise ObjectReferenceEditingError(
            "The local Qwen-Image-2.1 reference edit failed."
        ) from None


def build_openai_object_reference_edit_multipart(
    source_png: bytes,
    prompt: str,
    *,
    model: str = DEFAULT_OBJECT_REFERENCE_EDIT_MODEL,
    output_size: tuple[int, int],
) -> tuple[bytes, str]:
    """Build the multipart body used by the selected GPT Image model."""

    source_bytes = _validate_png_payload(source_png, "source")
    normalized_model = _normalize_openai_image_api_model(model)
    normalized_prompt = _normalize_prompt(prompt)
    width, height = _validate_supported_output_size(output_size)
    fields = (
        ("model", normalized_model),
        ("prompt", normalized_prompt),
        ("size", f"{width}x{height}"),
        ("quality", OBJECT_REFERENCE_EDIT_QUALITY),
        ("background", OBJECT_REFERENCE_EDIT_BACKGROUND),
        ("output_format", OBJECT_REFERENCE_EDIT_OUTPUT_FORMAT),
    )
    boundary = _new_multipart_boundary(source_bytes)
    marker = boundary.encode("ascii")
    chunks: list[bytes] = []
    for name, value in fields:
        chunks.extend(
            (
                b"--" + marker + b"\r\n",
                (
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                ).encode("ascii"),
                value.encode("utf-8"),
                b"\r\n",
            )
        )
    chunks.extend(
        _multipart_png_part(marker, "image", "object_reference.png", source_bytes)
    )
    chunks.append(b"--" + marker + b"--\r\n")
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


# ### Provider selection ###
def _default_image_editor(model: str) -> ObjectReferenceImageEditor:
    if model in OPENAI_IMAGE_API_REFERENCE_EDIT_MODELS:
        return openai_object_reference_edit
    if model in OPENAI_RESPONSES_REFERENCE_EDIT_MODELS:
        return openai_responses_object_reference_edit
    if model == OBJECT_REFERENCE_EDIT_MODEL_QWEN_IMAGE_2_1:
        return qwen_object_reference_edit
    raise ObjectReferenceEditingError(
        "Choose a supported object reference editing model."
    )


# ### Input validation ###
def _validate_edit_request(
    source_bgra: np.ndarray,
    prompt: str,
    api_key: str,
    *,
    require_api_key: bool,
) -> tuple[np.ndarray, str, str]:
    if (
        not isinstance(source_bgra, np.ndarray)
        or source_bgra.dtype != np.uint8
        or source_bgra.ndim != 3
        or source_bgra.shape[2] != 4
    ):
        raise ObjectReferenceEditingError(
            "The object reference must be a uint8 BGRA image."
        )
    height, width = source_bgra.shape[:2]
    if (
        width < 1
        or height < 1
        or source_bgra.size == 0
        or width * height > MAX_SOURCE_PIXELS
    ):
        raise ObjectReferenceEditingError(
            "The object reference is empty or too large to edit."
        )
    source = np.ascontiguousarray(source_bgra.copy())
    subject = source[:, :, 3] > 0
    if not np.any(subject):
        raise ObjectReferenceEditingError(
            "The object reference has no visible subject pixels."
        )
    source[~subject, :3] = 0

    normalized_prompt = _normalize_prompt(prompt)
    key = api_key.strip() if isinstance(api_key, str) else ""
    if require_api_key and not key:
        raise ObjectReferenceEditingError(
            "Add an OpenAI API key in Settings before editing a reference."
        )
    return source, normalized_prompt, key


def _normalize_prompt(prompt: str) -> str:
    if not isinstance(prompt, str):
        raise ObjectReferenceEditingError("Describe the requested object edit.")
    normalized = prompt.strip()
    if not normalized:
        raise ObjectReferenceEditingError("Describe the requested object edit.")
    if len(normalized) > MAX_PROMPT_CHARACTERS:
        raise ObjectReferenceEditingError(
            f"The object edit description must be {MAX_PROMPT_CHARACTERS} "
            "characters or fewer."
        )
    return normalized


def _normalize_model(model: str) -> str:
    normalized = model.strip() if isinstance(model, str) else ""
    if normalized not in SUPPORTED_OBJECT_REFERENCE_EDIT_MODELS:
        raise ObjectReferenceEditingError(
            "Choose a supported object reference editing model."
        )
    return normalized


def _normalize_openai_image_api_model(model: str) -> str:
    normalized = _normalize_model(model)
    if normalized not in OPENAI_IMAGE_API_REFERENCE_EDIT_MODELS:
        raise ObjectReferenceEditingError(
            "Choose a GPT Image model for the direct image-edit endpoint."
        )
    return normalized


def _normalize_openai_responses_model(model: str) -> str:
    normalized = _normalize_model(model)
    if normalized not in OPENAI_RESPONSES_REFERENCE_EDIT_MODELS:
        raise ObjectReferenceEditingError(
            "Choose GPT-5.6 Terra or GPT-5.6 Luna for this reference edit."
        )
    return normalized


def _validate_supported_output_size(
    output_size: tuple[int, int],
) -> tuple[int, int]:
    try:
        size = tuple(int(value) for value in output_size)
    except (TypeError, ValueError):
        size = ()
    if len(size) != 2 or size not in SUPPORTED_OUTPUT_SIZES:
        raise ObjectReferenceEditingError(
            "The requested AI image size is unsupported."
        )
    return size[0], size[1]


def _validate_png_payload(payload: bytes, label: str) -> bytes:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ObjectReferenceEditingError(
            f"The prepared {label} image is invalid."
        )
    binary = bytes(payload)
    if not binary or len(binary) > MAX_PNG_BYTES or not binary.startswith(b"\x89PNG"):
        raise ObjectReferenceEditingError(
            f"The prepared {label} image is invalid."
        )
    return binary


# ### Square reference preparation ###
def _crop_to_visible_alpha(source_bgra: np.ndarray) -> np.ndarray:
    visible_rows, visible_columns = np.nonzero(source_bgra[:, :, 3] > 0)
    if visible_rows.size == 0:
        return source_bgra.copy()
    top = int(visible_rows.min())
    bottom = int(visible_rows.max()) + 1
    left = int(visible_columns.min())
    right = int(visible_columns.max()) + 1
    return np.ascontiguousarray(source_bgra[top:bottom, left:right].copy())


def _build_square_reference_canvas(
    source_bgra: np.ndarray,
    *,
    maximum_content_scale: float = 1.0,
) -> np.ndarray:
    canvas_width, canvas_height = OBJECT_REFERENCE_EDIT_OUTPUT_SIZE
    source_height, source_width = source_bgra.shape[:2]
    scale = min(canvas_width / source_width, canvas_height / source_height)
    scale *= float(maximum_content_scale)
    content_width = min(canvas_width, max(1, round(source_width * scale)))
    content_height = min(canvas_height, max(1, round(source_height * scale)))
    content_x = (canvas_width - content_width) // 2
    content_y = (canvas_height - content_height) // 2
    source_canvas = np.zeros(
        OBJECT_REFERENCE_EDIT_OUTPUT_SHAPE,
        dtype=np.uint8,
    )
    source_resized = _resize_image(
        source_bgra,
        content_width,
        content_height,
    )
    source_canvas[
        content_y : content_y + content_height,
        content_x : content_x + content_width,
    ] = source_resized
    return source_canvas


def _resize_image(
    image: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    if image.shape[:2] == (height, width):
        return image.copy()
    shrinking = width < image.shape[1] or height < image.shape[0]
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LANCZOS4
    return cv2.resize(image, (width, height), interpolation=interpolation)


# ### Provider result decoding ###
def _decode_provider_png(
    payload: bytes,
    expected_size: tuple[int, int],
) -> np.ndarray:
    binary = _validate_png_payload(payload, "edited")
    decoded = cv2.imdecode(np.frombuffer(binary, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.size == 0:
        raise ObjectReferenceEditingError(
            "The AI object reference edit returned an invalid PNG."
        )
    if decoded.ndim == 2:
        decoded = cv2.cvtColor(decoded, cv2.COLOR_GRAY2BGRA)
    elif decoded.ndim == 3 and decoded.shape[2] == 3:
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2BGRA)
    elif decoded.ndim != 3 or decoded.shape[2] != 4:
        raise ObjectReferenceEditingError(
            "The AI object reference edit returned an invalid PNG."
        )
    expected_width, expected_height = expected_size
    if decoded.shape[:2] != (expected_height, expected_width):
        raise ObjectReferenceEditingError(
            "The AI object reference edit returned an unexpected image size."
        )
    return np.ascontiguousarray(decoded)


# ### Cancellable provider invocation ###
def _invoke_editor_cancellably(
    editor: ObjectReferenceImageEditor,
    source_png: bytes,
    *,
    api_key: str,
    model: str,
    prompt: str,
    output_size: tuple[int, int],
    cancellation_check: CancellationCheck | None,
) -> bytes:
    if cancellation_check is None:
        return editor(
            source_png,
            api_key=api_key,
            model=model,
            prompt=prompt,
            output_size=output_size,
            cancellation_check=None,
        )

    result_queue: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)
    transport_cancelled = threading.Event()

    def invoke() -> None:
        try:
            result = editor(
                source_png,
                api_key=api_key,
                model=model,
                prompt=prompt,
                output_size=output_size,
                cancellation_check=transport_cancelled.is_set,
            )
        except Exception as error:  # noqa: BLE001 - returned to the caller safely.
            result_queue.put((False, error))
        else:
            result_queue.put((True, result))

    threading.Thread(
        target=invoke,
        name="object-reference-edit-request",
        daemon=True,
    ).start()
    while True:
        if cancellation_check():
            transport_cancelled.set()
            raise ObjectReferenceEditingCancelled(
                "Object reference editing was cancelled."
            )
        try:
            succeeded, payload = result_queue.get(
                timeout=CANCELLATION_POLL_SECONDS
            )
        except queue.Empty:
            continue
        if not succeeded:
            assert isinstance(payload, Exception)
            raise payload
        if not isinstance(payload, bytes):
            raise ObjectReferenceEditingError(
                "The AI object reference edit returned no image."
            )
        return payload


# ### Multipart helpers ###
def _new_multipart_boundary(source_png: bytes) -> str:
    while True:
        boundary = f"----HouseMaker{uuid.uuid4().hex}"
        marker = boundary.encode("ascii")
        if marker not in source_png:
            return boundary


def _multipart_png_part(
    marker: bytes,
    name: str,
    filename: str,
    payload: bytes,
) -> list[bytes]:
    return [
        b"--" + marker + b"\r\n",
        (
            f'Content-Disposition: form-data; name="{name}"; '
            f'filename="{filename}"\r\n'
        ).encode("ascii"),
        b"Content-Type: image/png\r\n\r\n",
        payload,
        b"\r\n",
    ]


# ### Encoding and response helpers ###
def _encode_png(image: np.ndarray) -> bytes:
    succeeded, encoded = cv2.imencode(
        ".png",
        image,
        [cv2.IMWRITE_PNG_COMPRESSION, 3],
    )
    if not succeeded:
        raise ObjectReferenceEditingError(
            "The object reference could not be prepared for editing."
        )
    payload = encoded.tobytes()
    if len(payload) > MAX_PNG_BYTES:
        raise ObjectReferenceEditingError(
            "The prepared object reference is too large to edit."
        )
    return payload


def _read_response_limited(
    response: object,
    cancellation_check: CancellationCheck | None,
) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        _raise_if_cancelled(cancellation_check)
        chunk = response.read(64 * 1024)  # type: ignore[attr-defined]
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_RESPONSE_BYTES:
            raise ValueError("Response is too large.")
        chunks.append(chunk)
    return b"".join(chunks)


def _raise_if_cancelled(
    cancellation_check: CancellationCheck | None,
) -> None:
    if cancellation_check is not None and bool(cancellation_check()):
        raise ObjectReferenceEditingCancelled(
            "Object reference editing was cancelled."
        )
