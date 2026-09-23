# ### Imports ###
from __future__ import annotations

import base64
import json
import math
import os
import queue
import tempfile
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Protocol
from urllib.request import Request, urlopen

import cv2
import numpy as np
from PIL import Image, ImageOps

from housemaker.plan_correction_models import (
    DEFAULT_PLAN_CORRECTION_MODEL,
    OPENAI_PLAN_CORRECTION_MODELS,
    PLAN_CORRECTION_MODEL_GPT_5_6_LUNA,
    PLAN_CORRECTION_MODEL_GPT_5_6_TERRA,
    PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
    PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1,
    PLAN_CORRECTION_MODELS,
)
from housemaker.plan_wall_continuity import make_doorway_walls_continuous
from housemaker.qwen_plan_image_correction import (
    QwenPlanCorrectionCancelled,
    QwenPlanCorrectionError,
    create_default_qwen_plan_image_editor,
)

# ### OpenAI image-edit constants ###
OPENAI_IMAGE_EDIT_URL = "https://api.openai.com/v1/images/edits"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPENAI_IMAGE_MODEL = PLAN_CORRECTION_MODEL_GPT_IMAGE_2
OPENAI_IMAGE_QUALITY = "high"
OPENAI_NETWORK_TIMEOUT_SECONDS = 300.0
MAX_API_RESPONSE_BYTES = 64 * 1024 * 1024
MAX_API_INPUT_BYTES = 50 * 1024 * 1024
OUTPUT_DIMENSION_MULTIPLE = 32
MAX_OUTPUT_EDGE_PIXELS = 2048
MIN_OUTPUT_PIXEL_COUNT = 655_360
MAX_OUTPUT_ASPECT_RATIO = 3.0
EDITOR_SAFE_MARGIN_FRACTION = 0.03
EDITOR_CANCELLATION_POLL_SECONDS = 0.1

PLAN_CORRECTION_PROMPT = """Use case: precise-object-edit
Asset type: preprocessing input for automatic architectural wall extraction
Input image: the supplied top-down photograph is the edit target.
Primary request: convert this photographed architectural floor plan into a clean,
flat, orthographic black-and-white scanned plan.
Preserve exactly: every wall, doorway, window, stair, room boundary,
non-measurement symbol, room or feature label, relative position, proportion, and
the complete outer perimeter. Do not redesign, infer new architecture, omit
architectural geometry, or change retained text.
Remove every measurement annotation from the output: dimension and measurement
lines, extension or witness lines, their arrowheads or tick marks, and all numeric
values or unit text belonging to those measurements. Preserve numbers only when
they are part of a non-measurement room or feature label.
Correct only the page photography: straighten overall rotation and perspective,
gently correct fold-induced page deformation, remove fold creases, shadows, paper
grain, stains, speckles, color casts, and photographic background.
Composition: retain the entire sheet and complete floor plan with a safe white
margin on all four sides. Nothing may touch or be clipped by an output edge.
Style: crisp technical drawing, solid black ink lines on a uniform pure-white
background; preserve architectural fine lines and dashed lines without thickening
them, while omitting all measurement graphics.
Avoid: cropping, warped walls, curved straight lines, hallucinated details, missing
plan sections, retained measurement annotations, gray texture, black smudges, blur,
decorative styling, or a watermark."""


# ### Image constants ###
MAX_INPUT_EDGE_PIXELS = 16_384
MAX_INPUT_PIXEL_COUNT = 20_000_000
MIN_INPUT_EDGE_PIXELS = 16
PNG_COMPRESSION_LEVEL = 3


# ### Method and stage constants ###
CORRECTION_METHOD_OPENAI = "openai"
CORRECTION_METHOD_QWEN = "qwen"

STAGE_LOADING = "loading"
STAGE_PREPARING = "preparing"
STAGE_AI_CORRECTION = "ai_correction"
STAGE_BINARIZATION = "binarization"
STAGE_WALL_CONTINUITY = "wall_continuity"
STAGE_SAVING = "saving"
STAGE_COMPLETE = "complete"


# ### Public protocols and data models ###
CancellationCheck = Callable[[], bool]


@dataclass(frozen=True)
class PlanCorrectionProgress:
    stage: str
    percent: int
    message: str


ProgressCallback = Callable[[PlanCorrectionProgress], None]


class ImageEditor(Protocol):
    def __call__(
        self,
        image_bytes: bytes,
        *,
        api_key: str,
        prompt: str,
        output_size: tuple[int, int],
        cancellation_check: CancellationCheck | None,
    ) -> bytes: ...


@dataclass(frozen=True)
class PlanImageCorrectionResult:
    output_path: Path
    method: str
    input_size: tuple[int, int]
    output_size: tuple[int, int]
    stages: tuple[str, ...]


# ### Public exceptions ###
class PlanImageCorrectionError(RuntimeError):
    """Failure text safe to display in the application UI."""


class PlanImageCorrectionCancelled(PlanImageCorrectionError):
    pass


class PlanImageCorrectionInferenceError(PlanImageCorrectionError):
    pass


# ### Progress reporting ###
class _ProgressReporter:
    def __init__(self, callback: ProgressCallback | None) -> None:
        self._callback = callback
        self._stages: list[str] = []

    @property
    def stages(self) -> tuple[str, ...]:
        return tuple(self._stages)

    def emit(self, stage: str, percent: int, message: str) -> None:
        if stage not in self._stages:
            self._stages.append(stage)
        if self._callback is not None:
            self._callback(
                PlanCorrectionProgress(stage, max(0, min(100, int(percent))), message)
            )


# ### Backend selection ###
def _create_plan_image_editor(model: str) -> ImageEditor:
    if model == PLAN_CORRECTION_MODEL_GPT_IMAGE_2:
        return openai_image_edit
    if model in {
        PLAN_CORRECTION_MODEL_GPT_5_6_LUNA,
        PLAN_CORRECTION_MODEL_GPT_5_6_TERRA,
    }:
        return partial(openai_responses_image_edit, model=model)
    if model == PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1:
        try:
            return create_default_qwen_plan_image_editor()
        except QwenPlanCorrectionError as error:
            raise PlanImageCorrectionInferenceError(str(error)) from None
        except Exception:  # noqa: BLE001 - redact factory/runtime diagnostics.
            raise PlanImageCorrectionInferenceError(
                "The local Qwen-Image-2.1 editor could not be created."
            ) from None
    raise PlanImageCorrectionError(f"Unsupported plan correction model: {model}.")


# ### Public correction API ###
def correct_plan_image(
    input_path: str | Path,
    output_path: str | Path,
    *,
    api_key: str | None = None,
    model: str = DEFAULT_PLAN_CORRECTION_MODEL,
    image_editor: ImageEditor | None = None,
    progress_callback: ProgressCallback | None = None,
    cancellation_check: CancellationCheck | None = None,
    make_walls_continuous: bool = False,
) -> PlanImageCorrectionResult:
    """Create a safe black-and-white plan without ever overwriting the source."""

    selected_model = str(model).strip()
    if selected_model not in PLAN_CORRECTION_MODELS:
        raise PlanImageCorrectionError(
            f"Unsupported plan correction model: {selected_model or '(empty)'}."
        )
    if not isinstance(make_walls_continuous, bool):
        raise PlanImageCorrectionError(
            "Make walls continuous must be enabled or disabled."
        )
    source_path = Path(input_path).expanduser()
    destination = Path(output_path).expanduser()
    if destination.suffix.lower() != ".png":
        raise PlanImageCorrectionError("The corrected plan must be saved as PNG.")
    same_path = False
    try:
        same_path = os.path.normcase(str(source_path.resolve())) == os.path.normcase(
            str(destination.resolve())
        )
        same_file = destination.exists() and os.path.samefile(source_path, destination)
    except OSError:
        same_file = False
    if same_path or same_file:
        raise PlanImageCorrectionError(
            "The corrected plan must be saved separately from its source image."
        )

    reporter = _ProgressReporter(progress_callback)
    reporter.emit(STAGE_LOADING, 0, "Reading the architectural plan.")
    _raise_if_cancelled(cancellation_check)
    source = _load_plan_image(source_path)
    source_height, source_width = source.shape[:2]
    output_size = calculate_output_size(source_width, source_height)

    reporter.emit(STAGE_PREPARING, 8, "Preparing the complete plan sheet.")
    prepared = _prepare_editor_image(source, output_size)
    _raise_if_cancelled(cancellation_check)
    editor_input = _encode_png(prepared)
    _raise_if_cancelled(cancellation_check)
    if len(editor_input) >= MAX_API_INPUT_BYTES:
        raise PlanImageCorrectionInferenceError(
            "The prepared plan is too large for image correction."
        )
    key = str(api_key or "").strip()
    active_editor = image_editor or _create_plan_image_editor(selected_model)
    if (
        selected_model in OPENAI_PLAN_CORRECTION_MODELS
        and not key
        and image_editor is None
    ):
        raise PlanImageCorrectionInferenceError(
            "OpenAI image correction is not configured."
        )
    reporter.emit(
        STAGE_AI_CORRECTION,
        20,
        f"Correcting the photographed plan with {selected_model}.",
    )
    _raise_if_cancelled(cancellation_check)
    try:
        editor_arguments = {
            "api_key": key,
            "prompt": PLAN_CORRECTION_PROMPT,
            "output_size": output_size,
            "cancellation_check": cancellation_check,
        }
        if (
            selected_model == PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1
            and image_editor is None
        ):
            # sd-cli owns a child process and a global inference lock. Keep it
            # in this QThread so cancellation cannot finish the worker before
            # the child has terminated and the lock is released.
            payload = active_editor(editor_input, **editor_arguments)
        else:
            # Remote OpenAI calls and explicitly injected editors can be
            # detached so a stalled transport cannot block application exit.
            payload = _invoke_editor_cancellably(
                active_editor,
                editor_input,
                **editor_arguments,
            )
    except PlanImageCorrectionCancelled:
        raise
    except QwenPlanCorrectionCancelled:
        raise PlanImageCorrectionCancelled(
            "Image correction was cancelled."
        ) from None
    except PlanImageCorrectionInferenceError:
        raise
    except QwenPlanCorrectionError as error:
        raise PlanImageCorrectionInferenceError(str(error)) from None
    except Exception:  # noqa: BLE001 - redact every untrusted editor failure.
        # Do not forward untrusted exception text: it can contain credentials,
        # request headers, or provider response bodies.
        raise PlanImageCorrectionInferenceError(
            f"The {selected_model} image correction failed."
        ) from None
    _raise_if_cancelled(cancellation_check)
    corrected = _decode_editor_image(payload, expected_size=output_size)

    _raise_if_cancelled(cancellation_check)
    reporter.emit(STAGE_BINARIZATION, 82, "Creating the black-and-white plan.")
    binary = binarize_ai_plan(corrected)
    if make_walls_continuous:
        _raise_if_cancelled(cancellation_check)
        reporter.emit(
            STAGE_WALL_CONTINUITY,
            88,
            "Connecting walls across doorways.",
        )
        _raise_if_cancelled(cancellation_check)
        binary = make_doorway_walls_continuous(binary)
        _raise_if_cancelled(cancellation_check)
    method = (
        CORRECTION_METHOD_OPENAI
        if selected_model in OPENAI_PLAN_CORRECTION_MODELS
        else CORRECTION_METHOD_QWEN
    )

    reporter.emit(STAGE_SAVING, 94, "Saving the corrected plan.")
    _write_png_atomically(destination, binary, cancellation_check)
    reporter.emit(STAGE_COMPLETE, 100, "Image correction complete.")
    return PlanImageCorrectionResult(
        output_path=destination,
        method=method,
        input_size=(source_width, source_height),
        output_size=(binary.shape[1], binary.shape[0]),
        stages=reporter.stages,
    )


def calculate_output_size(width: int, height: int) -> tuple[int, int]:
    """Return a backend-compatible size without cropping or stretching."""

    if width < 1 or height < 1:
        raise ValueError("Image dimensions must be positive.")

    # Very long sheets are letterboxed onto the widest supported canvas rather
    # than distorted to satisfy the API's 3:1 aspect-ratio limit.
    canvas_width = float(width)
    canvas_height = float(height)
    if canvas_width / canvas_height > MAX_OUTPUT_ASPECT_RATIO:
        canvas_height = canvas_width / MAX_OUTPUT_ASPECT_RATIO
    elif canvas_height / canvas_width > MAX_OUTPUT_ASPECT_RATIO:
        canvas_width = canvas_height / MAX_OUTPUT_ASPECT_RATIO

    scale = min(1.0, MAX_OUTPUT_EDGE_PIXELS / max(canvas_width, canvas_height))
    if canvas_width * scale * canvas_height * scale < MIN_OUTPUT_PIXEL_COUNT:
        scale = (MIN_OUTPUT_PIXEL_COUNT / (canvas_width * canvas_height)) ** 0.5
    scale = min(scale, MAX_OUTPUT_EDGE_PIXELS / max(canvas_width, canvas_height))
    scaled_width = canvas_width * scale
    scaled_height = canvas_height * scale

    def rounded_multiple(value: float) -> int:
        rounded = round(value / OUTPUT_DIMENSION_MULTIPLE) * OUTPUT_DIMENSION_MULTIPLE
        return max(OUTPUT_DIMENSION_MULTIPLE, min(MAX_OUTPUT_EDGE_PIXELS, rounded))

    def enforce_aspect_limit(output_width: int, output_height: int) -> tuple[int, int]:
        if output_width > output_height * MAX_OUTPUT_ASPECT_RATIO:
            minimum_height = (
                math.ceil(
                    output_width
                    / MAX_OUTPUT_ASPECT_RATIO
                    / OUTPUT_DIMENSION_MULTIPLE
                )
                * OUTPUT_DIMENSION_MULTIPLE
            )
            output_height = max(output_height, minimum_height)
        elif output_height > output_width * MAX_OUTPUT_ASPECT_RATIO:
            minimum_width = (
                math.ceil(
                    output_height
                    / MAX_OUTPUT_ASPECT_RATIO
                    / OUTPUT_DIMENSION_MULTIPLE
                )
                * OUTPUT_DIMENSION_MULTIPLE
            )
            output_width = max(output_width, minimum_width)
        return output_width, output_height

    output_width = rounded_multiple(scaled_width)
    output_height = rounded_multiple(scaled_height)
    output_width, output_height = enforce_aspect_limit(output_width, output_height)
    target_ratio = canvas_width / canvas_height
    while output_width * output_height < MIN_OUTPUT_PIXEL_COUNT:
        width_error = abs(
            ((output_width + OUTPUT_DIMENSION_MULTIPLE) / output_height)
            - target_ratio
        )
        height_error = abs(
            (output_width / (output_height + OUTPUT_DIMENSION_MULTIPLE))
            - target_ratio
        )
        if width_error <= height_error and output_width < MAX_OUTPUT_EDGE_PIXELS:
            output_width += OUTPUT_DIMENSION_MULTIPLE
        elif output_height < MAX_OUTPUT_EDGE_PIXELS:
            output_height += OUTPUT_DIMENSION_MULTIPLE
        else:  # Defensive: valid <=3:1 canvases always fit the minimum area.
            raise ValueError("Image aspect ratio cannot satisfy output constraints.")
    output_width, output_height = enforce_aspect_limit(output_width, output_height)
    return output_width, output_height


# ### OpenAI image editing ###
def _invoke_editor_cancellably(
    editor: ImageEditor,
    image_bytes: bytes,
    *,
    api_key: str,
    prompt: str,
    output_size: tuple[int, int],
    cancellation_check: CancellationCheck | None,
) -> bytes:
    """Keep a stalled network request from blocking job cancellation or exit."""

    if cancellation_check is None:
        return editor(
            image_bytes,
            api_key=api_key,
            prompt=prompt,
            output_size=output_size,
            cancellation_check=None,
        )

    result_queue: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)
    transport_cancelled = threading.Event()

    def invoke() -> None:
        try:
            result = editor(
                image_bytes,
                api_key=api_key,
                prompt=prompt,
                output_size=output_size,
                cancellation_check=transport_cancelled.is_set,
            )
        except Exception as error:  # noqa: BLE001 - handed back to caller safely.
            result_queue.put((False, error))
        else:
            result_queue.put((True, result))

    threading.Thread(
        target=invoke,
        name="plan-image-correction-request",
        daemon=True,
    ).start()
    while True:
        if cancellation_check():
            transport_cancelled.set()
            raise PlanImageCorrectionCancelled("Image correction was cancelled.")
        try:
            succeeded, payload = result_queue.get(
                timeout=EDITOR_CANCELLATION_POLL_SECONDS
            )
        except queue.Empty:
            continue
        if not succeeded:
            assert isinstance(payload, Exception)
            raise payload
        if not isinstance(payload, bytes):
            raise PlanImageCorrectionInferenceError(
                "The AI image edit returned no image."
            )
        return payload


def openai_image_edit(
    image_bytes: bytes,
    *,
    api_key: str,
    prompt: str,
    output_size: tuple[int, int],
    cancellation_check: CancellationCheck | None,
) -> bytes:
    """Call the Images edit endpoint and return decoded PNG bytes."""

    key = str(api_key).strip()
    if not key:
        raise PlanImageCorrectionInferenceError(
            "OpenAI image correction is not configured."
        )
    _validate_output_size(output_size)
    _raise_if_cancelled(cancellation_check)
    boundary = f"----HouseMaker{uuid.uuid4().hex}"
    fields = {
        "model": OPENAI_IMAGE_MODEL,
        "prompt": prompt,
        "quality": OPENAI_IMAGE_QUALITY,
        "size": f"{output_size[0]}x{output_size[1]}",
        "output_format": "png",
    }
    body = _build_multipart_body(boundary, fields, image_bytes)
    request = Request(
        OPENAI_IMAGE_EDIT_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "HouseMaker/1.0",
        },
    )
    try:
        with urlopen(request, timeout=OPENAI_NETWORK_TIMEOUT_SECONDS) as response:
            raw_response = _read_response_limited(response, cancellation_check)
        parsed = json.loads(raw_response.decode("utf-8"))
        encoded = parsed["data"][0]["b64_json"]
        if not isinstance(encoded, str):
            raise TypeError
        decoded = base64.b64decode(encoded, validate=True)
        if not decoded or len(decoded) > MAX_API_RESPONSE_BYTES:
            raise ValueError
        return decoded
    except PlanImageCorrectionCancelled:
        raise
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        # Provider response bodies and transport exception strings are deliberately
        # omitted to guarantee that secrets never reach the UI or job log.
        raise PlanImageCorrectionInferenceError(
            "The OpenAI image edit did not return a usable image."
        ) from None


def openai_responses_image_edit(
    image_bytes: bytes,
    *,
    model: str,
    api_key: str,
    prompt: str,
    output_size: tuple[int, int],
    cancellation_check: CancellationCheck | None,
) -> bytes:
    """Use a GPT-5.6 model to direct an image-generation edit."""

    if model not in {
        PLAN_CORRECTION_MODEL_GPT_5_6_LUNA,
        PLAN_CORRECTION_MODEL_GPT_5_6_TERRA,
    }:
        raise PlanImageCorrectionInferenceError(
            "The selected OpenAI plan correction model is unsupported."
        )
    key = str(api_key).strip()
    if not key:
        raise PlanImageCorrectionInferenceError(
            "OpenAI image correction is not configured."
        )
    _validate_output_size(output_size)
    _raise_if_cancelled(cancellation_check)
    encoded_input = base64.b64encode(bytes(image_bytes)).decode("ascii")
    request_body = json.dumps(
        {
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{encoded_input}",
                            "detail": "original",
                        },
                    ],
                }
            ],
            "tools": [
                {
                    "type": "image_generation",
                    "model": OPENAI_IMAGE_MODEL,
                    "action": "edit",
                    "quality": OPENAI_IMAGE_QUALITY,
                    "size": f"{output_size[0]}x{output_size[1]}",
                    "output_format": "png",
                }
            ],
            "tool_choice": {"type": "image_generation"},
            "store": False,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    request = Request(
        OPENAI_RESPONSES_URL,
        data=request_body,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "HouseMaker/1.0",
        },
    )
    try:
        with urlopen(request, timeout=OPENAI_NETWORK_TIMEOUT_SECONDS) as response:
            raw_response = _read_response_limited(response, cancellation_check)
        parsed = json.loads(raw_response.decode("utf-8"))
        return _decode_openai_image_generation_result(parsed)
    except PlanImageCorrectionCancelled:
        raise
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ):
        raise PlanImageCorrectionInferenceError(
            f"The {model} image correction did not return a usable image."
        ) from None


def _decode_openai_image_generation_result(payload: object) -> bytes:
    """Decode the first completed image-generation call in a response."""

    if not isinstance(payload, dict):
        raise TypeError
    output = payload.get("output")
    if not isinstance(output, list):
        raise TypeError
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "image_generation_call":
            continue
        encoded = item.get("result")
        if not isinstance(encoded, str) or not encoded:
            raise ValueError
        maximum_encoded_characters = ((MAX_API_RESPONSE_BYTES + 2) // 3) * 4
        if len(encoded) > maximum_encoded_characters:
            raise ValueError
        decoded = base64.b64decode(encoded, validate=True)
        if not decoded or len(decoded) > MAX_API_RESPONSE_BYTES:
            raise ValueError
        return decoded
    raise ValueError


def _build_multipart_body(
    boundary: str,
    fields: dict[str, str],
    image_bytes: bytes,
) -> bytes:
    chunks: list[bytes] = []
    marker = boundary.encode("ascii")
    for name, value in fields.items():
        chunks.extend(
            (
                b"--" + marker + b"\r\n",
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode("utf-8"),
                b"\r\n",
            )
        )
    chunks.extend(
        (
            b"--" + marker + b"\r\n",
            b'Content-Disposition: form-data; name="image[]"; filename="plan.png"\r\n',
            b"Content-Type: image/png\r\n\r\n",
            bytes(image_bytes),
            b"\r\n--" + marker + b"--\r\n",
        )
    )
    return b"".join(chunks)


def _validate_output_size(output_size: tuple[int, int]) -> None:
    try:
        width, height = (int(value) for value in output_size)
    except (TypeError, ValueError):
        raise PlanImageCorrectionInferenceError(
            "The requested AI image size is invalid."
        ) from None
    if (
        width < OUTPUT_DIMENSION_MULTIPLE
        or height < OUTPUT_DIMENSION_MULTIPLE
        or width > MAX_OUTPUT_EDGE_PIXELS
        or height > MAX_OUTPUT_EDGE_PIXELS
        or width % OUTPUT_DIMENSION_MULTIPLE
        or height % OUTPUT_DIMENSION_MULTIPLE
        or width * height < MIN_OUTPUT_PIXEL_COUNT
        or max(width, height) / min(width, height) > MAX_OUTPUT_ASPECT_RATIO
    ):
        raise PlanImageCorrectionInferenceError(
            "The requested AI image size is outside the supported range."
        )


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
        if total > MAX_API_RESPONSE_BYTES:
            raise ValueError("Response is too large.")
        chunks.append(chunk)
    return b"".join(chunks)


# ### Accepted-image binarization ###
def binarize_ai_plan(image_bgr: np.ndarray) -> np.ndarray:
    """Globally binarize an accepted AI result without local noise amplification."""

    _validate_color_image(image_bgr)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0.45)
    otsu, _unused = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    threshold = max(90, min(205, round(otsu)))
    return np.where(gray <= threshold, 0, 255).astype(np.uint8)


# ### Image input and output ###
def _load_plan_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise PlanImageCorrectionError(f"Plan image does not exist: {path}")
    try:
        with Image.open(path) as opened:
            oriented = ImageOps.exif_transpose(opened)
            width, height = oriented.size
            if (
                min(width, height) < MIN_INPUT_EDGE_PIXELS
                or max(width, height) > MAX_INPUT_EDGE_PIXELS
                or width * height > MAX_INPUT_PIXEL_COUNT
            ):
                raise PlanImageCorrectionError(
                    "The plan image dimensions are outside the supported range."
                )
            rgba = oriented.convert("RGBA")
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            rgb = Image.alpha_composite(white, rgba).convert("RGB")
            array = np.array(rgb, dtype=np.uint8, copy=True)
    except PlanImageCorrectionError:
        raise
    except (OSError, Image.DecompressionBombError) as error:
        raise PlanImageCorrectionError(
            "The selected file is not a readable plan image."
        ) from error
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def _decode_editor_image(
    payload: object,
    *,
    expected_size: tuple[int, int],
) -> np.ndarray:
    if not isinstance(payload, (bytes, bytearray)) or not payload:
        raise PlanImageCorrectionInferenceError(
            "The AI image edit returned no image."
        )
    if len(payload) > MAX_API_RESPONSE_BYTES:
        raise PlanImageCorrectionInferenceError(
            "The AI image edit was unexpectedly large."
        )
    try:
        with Image.open(BytesIO(bytes(payload))) as opened:
            width, height = opened.size
            if (width, height) != expected_size:
                raise PlanImageCorrectionInferenceError(
                    "The AI result changed the requested plan dimensions."
                )
            rgb = opened.convert("RGB")
            array = np.array(rgb, dtype=np.uint8, copy=True)
    except PlanImageCorrectionInferenceError:
        raise
    except (OSError, Image.DecompressionBombError):
        raise PlanImageCorrectionInferenceError(
            "The AI image edit returned an unreadable image."
        ) from None
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def _fit_complete_image(
    image_bgr: np.ndarray, output_size: tuple[int, int]
) -> np.ndarray:
    """Fit the whole image on a white canvas without cropping or distortion."""

    _validate_color_image(image_bgr)
    if (image_bgr.shape[1], image_bgr.shape[0]) == output_size:
        return image_bgr.copy()
    output_width, output_height = output_size
    scale = min(
        output_width / image_bgr.shape[1],
        output_height / image_bgr.shape[0],
    )
    resized_width = max(1, min(output_width, round(image_bgr.shape[1] * scale)))
    resized_height = max(1, min(output_height, round(image_bgr.shape[0] * scale)))
    shrinking = scale < 1.0
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
    resized = cv2.resize(
        image_bgr,
        (resized_width, resized_height),
        interpolation=interpolation,
    )
    canvas = np.full((output_height, output_width, 3), 255, dtype=np.uint8)
    left = (output_width - resized_width) // 2
    top = (output_height - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def _prepare_editor_image(
    image_bgr: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    """Fit the entire source inside a white guard margin for AI editing."""

    output_width, output_height = output_size
    horizontal_margin = max(
        2,
        round(output_width * EDITOR_SAFE_MARGIN_FRACTION),
    )
    vertical_margin = max(
        2,
        round(output_height * EDITOR_SAFE_MARGIN_FRACTION),
    )
    inner_size = (
        max(1, output_width - horizontal_margin * 2),
        max(1, output_height - vertical_margin * 2),
    )
    fitted = _fit_complete_image(image_bgr, inner_size)
    canvas = np.full((output_height, output_width, 3), 255, dtype=np.uint8)
    canvas[
        vertical_margin : vertical_margin + inner_size[1],
        horizontal_margin : horizontal_margin + inner_size[0],
    ] = fitted
    return canvas


def _encode_png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(
        ".png", image, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION_LEVEL]
    )
    if not ok:
        raise PlanImageCorrectionError("The plan image could not be encoded.")
    return encoded.tobytes()


def _write_png_atomically(
    destination: Path,
    binary: np.ndarray,
    cancellation_check: CancellationCheck | None,
) -> None:
    _raise_if_cancelled(cancellation_check)
    encoded = _encode_png(binary)
    _raise_if_cancelled(cancellation_check)
    temporary: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, raw_path = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
        )
        temporary = Path(raw_path)
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        _raise_if_cancelled(cancellation_check)
        os.replace(temporary, destination)
        temporary = None
    except PlanImageCorrectionCancelled:
        raise
    except OSError as error:
        raise PlanImageCorrectionError(
            "The corrected plan could not be saved."
        ) from error
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


# ### Validation and cancellation helpers ###
def _validate_color_image(image: np.ndarray) -> None:
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
        or min(image.shape[:2]) < 2
    ):
        raise PlanImageCorrectionError(
            "Plan correction requires a non-empty 8-bit color image."
        )


def _raise_if_cancelled(cancellation_check: CancellationCheck | None) -> None:
    if cancellation_check is not None and bool(cancellation_check()):
        raise PlanImageCorrectionCancelled("Image correction was cancelled.")
