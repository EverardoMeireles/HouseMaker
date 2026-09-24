# ### Imports ###
from __future__ import annotations

import base64
import json
import math
import queue
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.request import Request, urlopen

import cv2
import numpy as np

from housemaker.video_source import normalize_video_frame


# ### Constants ###
CEILING_HEIGHT_INFERENCE_MODEL = "gpt-5.6-luna"
CEILING_HEIGHT_JOB_KIND = "Ceiling height inference"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPENAI_NETWORK_TIMEOUT_SECONDS = 120.0
MAX_INPUT_EDGE_PIXELS = 1_536
MAX_INPUT_IMAGE_BYTES = 16 * 1024 * 1024
MAX_RESPONSE_BYTES = 1 * 1024 * 1024
CANCELLATION_POLL_SECONDS = 0.05
MIN_PLAUSIBLE_HEIGHT_METERS = 1.5
MAX_PLAUSIBLE_HEIGHT_METERS = 15.0
MAX_BASIS_CHARACTERS = 320
MINIMUM_RANGE_BY_CONFIDENCE = {
    "low": 0.6,
    "medium": 0.4,
    "high": 0.2,
}
_OPENAI_TRANSPORT_LOCK = threading.Lock()

CEILING_HEIGHT_PROMPT = """Estimate the finished-floor to finished-ceiling vertical height in the supplied single indoor video frame.

Use only evidence visible in this frame. Consider the visible floor/ceiling boundaries, perspective and vanishing points, and recognizable standard-size objects such as doors, windows, cabinets, people, or furniture. Treat object dimensions as uncertain priors rather than exact measurements, and cross-check more than one independent cue whenever possible. Do not mistake camera height, wall width, or an opening height for ceiling height.

Absolute scale from one uncalibrated image is inherently ambiguous. Return status "insufficient_evidence" unless the frame provides a usable floor-to-ceiling relationship and at least one credible scale cue. Never invent hidden geometry or false precision. For an estimate, return values rounded to 0.1 metre and a conservative likely interval: at least 0.6 metre wide for low confidence, 0.4 metre for medium, or 0.2 metre for high. Keep basis short, nonempty, and explain the strongest visible cues or why the evidence is insufficient."""

CEILING_HEIGHT_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["estimated", "insufficient_evidence"],
        },
        "estimate_m": {
            "anyOf": [{"type": "number"}, {"type": "null"}],
        },
        "minimum_m": {
            "anyOf": [{"type": "number"}, {"type": "null"}],
        },
        "maximum_m": {
            "anyOf": [{"type": "number"}, {"type": "null"}],
        },
        "confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "basis": {"type": "string", "pattern": ".*\\S.*"},
    },
    "required": [
        "status",
        "estimate_m",
        "minimum_m",
        "maximum_m",
        "confidence",
        "basis",
    ],
    "additionalProperties": False,
}


# ### Public data ###
@dataclass(frozen=True)
class CeilingHeightEstimate:
    """Validated metric estimate for one immutable video frame."""

    status: str
    estimate_m: float | None
    minimum_m: float | None
    maximum_m: float | None
    confidence: str
    basis: str

    @property
    def has_estimate(self) -> bool:
        return self.status == "estimated"


# ### Public exceptions ###
class CeilingHeightEstimationError(RuntimeError):
    """Failure text that is safe to display in the application UI."""


class CeilingHeightEstimationCancelled(CeilingHeightEstimationError):
    pass


# ### Public protocols ###
class CeilingHeightEstimator(Protocol):
    def __call__(
        self,
        frame_bgr: np.ndarray,
        *,
        api_key: str,
        cancellation_check: Callable[[], bool] | None = None,
    ) -> CeilingHeightEstimate: ...


class UrlOpenFunction(Protocol):
    def __call__(self, request: Request, **kwargs: Any) -> object: ...


# ### Request construction ###
def build_ceiling_height_request_body(
    image_data_uri: str,
    *,
    model: str = CEILING_HEIGHT_INFERENCE_MODEL,
) -> dict[str, Any]:
    """Build one strict Responses API vision-analysis request."""

    normalized_uri = str(image_data_uri).strip()
    if not normalized_uri.startswith("data:image/"):
        raise CeilingHeightEstimationError(
            "The current video frame could not be prepared for analysis."
        )
    return {
        "model": str(model).strip() or CEILING_HEIGHT_INFERENCE_MODEL,
        "store": False,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": CEILING_HEIGHT_PROMPT},
                    {
                        "type": "input_image",
                        "image_url": normalized_uri,
                        "detail": "high",
                    },
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "ceiling_height_estimate",
                "strict": True,
                "schema": CEILING_HEIGHT_RESULT_SCHEMA,
            }
        },
        "max_output_tokens": 500,
    }


# ### Public inference API ###
def infer_ceiling_height(
    frame_bgr: np.ndarray,
    *,
    api_key: str,
    cancellation_check: Callable[[], bool] | None = None,
    model: str = CEILING_HEIGHT_INFERENCE_MODEL,
    opener: UrlOpenFunction | None = None,
) -> CeilingHeightEstimate:
    """Estimate ceiling height without exposing provider diagnostics or secrets."""

    key = str(api_key).strip()
    if not key:
        raise CeilingHeightEstimationError(
            "Add an OpenAI API key in Settings before inferring ceiling height."
        )
    _raise_if_cancelled(cancellation_check)
    image_data_uri = _encode_frame_data_uri(frame_bgr)
    request_body = build_ceiling_height_request_body(
        image_data_uri,
        model=model,
    )

    transport_cancelled = threading.Event()

    def request_estimate() -> CeilingHeightEstimate:
        return _request_ceiling_height_estimate(
            request_body,
            api_key=key,
            cancellation_check=transport_cancelled.is_set,
            opener=opener,
        )

    if cancellation_check is None:
        return request_estimate()
    return _run_cancellably(
        request_estimate,
        cancellation_check,
        transport_cancelled,
    )


def format_ceiling_height_estimate(estimate: CeilingHeightEstimate) -> str:
    """Return a compact, uncertainty-aware label for the Generation tab."""

    if not estimate.has_estimate:
        suffix = f" {estimate.basis}" if estimate.basis else ""
        return f"Ceiling height: Could not estimate reliably.{suffix}"
    assert estimate.estimate_m is not None
    assert estimate.minimum_m is not None
    assert estimate.maximum_m is not None
    result = (
        "Estimated ceiling height: "
        f"{estimate.estimate_m:.1f} m "
        f"(likely {estimate.minimum_m:.1f}-"
        f"{estimate.maximum_m:.1f} m, "
        f"{estimate.confidence} confidence)."
    )
    if estimate.basis:
        result += f" {estimate.basis}"
    return result


# ### Frame encoding ###
def _encode_frame_data_uri(frame_bgr: np.ndarray) -> str:
    try:
        normalized = normalize_video_frame(np.asarray(frame_bgr))
    except (TypeError, ValueError, cv2.error):
        raise CeilingHeightEstimationError(
            "The current video frame could not be prepared for analysis."
        ) from None
    if normalized.size == 0 or normalized.shape[0] < 2 or normalized.shape[1] < 2:
        raise CeilingHeightEstimationError(
            "The current video frame is too small to analyze."
        )
    if normalized.dtype != np.uint8:
        if not np.issubdtype(normalized.dtype, np.number):
            raise CeilingHeightEstimationError(
                "The current video frame has an unsupported format."
            )
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    height, width = normalized.shape[:2]
    scale = min(1.0, MAX_INPUT_EDGE_PIXELS / max(height, width))
    if scale < 1.0:
        normalized = cv2.resize(
            normalized,
            (max(2, round(width * scale)), max(2, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    encoded, buffer = cv2.imencode(
        ".jpg",
        normalized,
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )
    if not encoded:
        raise CeilingHeightEstimationError(
            "The current video frame could not be encoded for analysis."
        )
    payload = buffer.tobytes()
    if not payload or len(payload) > MAX_INPUT_IMAGE_BYTES:
        raise CeilingHeightEstimationError(
            "The current video frame is too large to analyze."
        )
    encoded_payload = base64.b64encode(payload).decode("ascii")
    return f"data:image/jpeg;base64,{encoded_payload}"


# ### OpenAI transport ###
def _request_ceiling_height_estimate(
    request_body: Mapping[str, Any],
    *,
    api_key: str,
    cancellation_check: Callable[[], bool] | None,
    opener: UrlOpenFunction | None,
) -> CeilingHeightEstimate:
    request = Request(
        OPENAI_RESPONSES_URL,
        data=json.dumps(request_body, separators=(",", ":")).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "HouseMaker/1.0",
        },
    )
    open_request = urlopen if opener is None else opener
    while not _OPENAI_TRANSPORT_LOCK.acquire(
        timeout=CANCELLATION_POLL_SECONDS
    ):
        _raise_if_cancelled(cancellation_check)
    try:
        _raise_if_cancelled(cancellation_check)
        with open_request(
            request,
            timeout=OPENAI_NETWORK_TIMEOUT_SECONDS,
        ) as response:
            raw_response = _read_response_limited(response, cancellation_check)
        payload = json.loads(raw_response.decode("utf-8"))
        output_text = _extract_openai_output_text(payload)
        result_payload = json.loads(output_text)
        return _parse_ceiling_height_estimate(result_payload)
    except CeilingHeightEstimationCancelled:
        raise
    except CeilingHeightEstimationError:
        raise
    except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError):
        # Provider bodies and transport messages can contain request details.
        raise CeilingHeightEstimationError(
            "OpenAI did not return a usable ceiling-height estimate."
        ) from None
    finally:
        _OPENAI_TRANSPORT_LOCK.release()


def _read_response_limited(
    response: object,
    cancellation_check: Callable[[], bool] | None,
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
            raise CeilingHeightEstimationError(
                "OpenAI returned an unexpectedly large estimation response."
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _run_cancellably(
    operation: Callable[[], CeilingHeightEstimate],
    cancellation_check: Callable[[], bool],
    transport_cancelled: threading.Event,
) -> CeilingHeightEstimate:
    """Allow the Qt worker to stop even if a network transport stalls."""

    outcome: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

    def invoke() -> None:
        try:
            result = operation()
        except Exception as error:  # noqa: BLE001 - transferred to caller.
            outcome.put((False, error))
        else:
            outcome.put((True, result))

    threading.Thread(
        target=invoke,
        name="ceiling-height-openai-request",
        daemon=True,
    ).start()
    while True:
        if cancellation_check():
            transport_cancelled.set()
            raise CeilingHeightEstimationCancelled(
                "Ceiling height inference was cancelled."
            )
        try:
            succeeded, value = outcome.get(timeout=CANCELLATION_POLL_SECONDS)
        except queue.Empty:
            continue
        if cancellation_check():
            transport_cancelled.set()
            raise CeilingHeightEstimationCancelled(
                "Ceiling height inference was cancelled."
            )
        if succeeded and isinstance(value, CeilingHeightEstimate):
            return value
        if isinstance(value, Exception):
            raise value
        raise CeilingHeightEstimationError(
            "Ceiling height inference finished without a result."
        )


# ### Response parsing ###
def _extract_openai_output_text(payload: object) -> str:
    if not isinstance(payload, Mapping):
        raise CeilingHeightEstimationError(
            "OpenAI returned an invalid estimation response."
        )
    status = payload.get("status")
    if status == "incomplete":
        raise CeilingHeightEstimationError(
            "OpenAI could not finish the ceiling-height estimate."
        )
    if status != "completed":
        raise CeilingHeightEstimationError(
            "OpenAI did not complete the ceiling-height estimate."
        )

    fragments: list[str] = []
    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            content = item.get("content") if isinstance(item, Mapping) else None
            if not isinstance(content, list):
                continue
            for content_item in content:
                if not isinstance(content_item, Mapping):
                    continue
                if content_item.get("type") == "refusal":
                    raise CeilingHeightEstimationError(
                        "OpenAI declined to analyze the current frame."
                    )
                if content_item.get("type") != "output_text":
                    continue
                text = content_item.get("text")
                if isinstance(text, str) and text.strip():
                    fragments.append(text.strip())
    if fragments:
        return "\n".join(fragments)
    top_level_text = payload.get("output_text")
    if isinstance(top_level_text, str) and top_level_text.strip():
        return top_level_text.strip()
    raise CeilingHeightEstimationError(
        "OpenAI completed without a ceiling-height estimate."
    )


def _parse_ceiling_height_estimate(payload: object) -> CeilingHeightEstimate:
    if not isinstance(payload, Mapping):
        raise CeilingHeightEstimationError(
            "OpenAI returned an invalid ceiling-height estimate."
        )
    status = payload.get("status")
    confidence = payload.get("confidence")
    if status not in {"estimated", "insufficient_evidence"}:
        raise CeilingHeightEstimationError(
            "OpenAI returned an invalid ceiling-height status."
        )
    if confidence not in {"low", "medium", "high"}:
        raise CeilingHeightEstimationError(
            "OpenAI returned an invalid confidence value."
        )
    basis = _normalize_basis(payload.get("basis"))
    estimate = _optional_finite_number(payload.get("estimate_m"))
    minimum = _optional_finite_number(payload.get("minimum_m"))
    maximum = _optional_finite_number(payload.get("maximum_m"))

    if status == "insufficient_evidence":
        if any(value is not None for value in (estimate, minimum, maximum)):
            raise CeilingHeightEstimationError(
                "OpenAI returned inconsistent insufficient-evidence data."
            )
        return CeilingHeightEstimate(
            status=status,
            estimate_m=None,
            minimum_m=None,
            maximum_m=None,
            confidence="low",
            basis=basis,
        )

    if estimate is None or minimum is None or maximum is None:
        raise CeilingHeightEstimationError(
            "OpenAI omitted part of the ceiling-height range."
        )
    if not (
        MIN_PLAUSIBLE_HEIGHT_METERS
        <= minimum
        <= estimate
        <= maximum
        <= MAX_PLAUSIBLE_HEIGHT_METERS
    ):
        raise CeilingHeightEstimationError(
            "OpenAI returned an implausible ceiling-height range."
        )
    estimate, minimum, maximum = _normalize_estimate_precision(
        estimate,
        minimum,
        maximum,
        str(confidence),
    )
    return CeilingHeightEstimate(
        status=status,
        estimate_m=estimate,
        minimum_m=minimum,
        maximum_m=maximum,
        confidence=str(confidence),
        basis=basis,
    )


# ### Validation helpers ###
def _optional_finite_number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CeilingHeightEstimationError(
            "OpenAI returned a non-numeric ceiling height."
        )
    normalized = float(value)
    if not math.isfinite(normalized):
        raise CeilingHeightEstimationError(
            "OpenAI returned a non-finite ceiling height."
        )
    return normalized


def _normalize_basis(value: object) -> str:
    if not isinstance(value, str):
        raise CeilingHeightEstimationError(
            "OpenAI omitted the ceiling-height explanation."
        )
    normalized = " ".join(value.split())
    if not normalized:
        raise CeilingHeightEstimationError(
            "OpenAI omitted the ceiling-height explanation."
        )
    if len(normalized) <= MAX_BASIS_CHARACTERS:
        return normalized
    return normalized[: MAX_BASIS_CHARACTERS - 3].rstrip() + "..."


def _normalize_estimate_precision(
    estimate: float,
    minimum: float,
    maximum: float,
    confidence: str,
) -> tuple[float, float, float]:
    """Keep a single-image estimate conservative and tenth-metre precise."""

    rounded_estimate = round(estimate, 1)
    rounded_minimum = math.floor(minimum * 10.0) / 10.0
    rounded_maximum = math.ceil(maximum * 10.0) / 10.0
    target_width = MINIMUM_RANGE_BY_CONFIDENCE[confidence]
    half_width = target_width / 2.0
    rounded_minimum = min(
        rounded_minimum,
        math.floor((rounded_estimate - half_width) * 10.0) / 10.0,
    )
    rounded_maximum = max(
        rounded_maximum,
        math.ceil((rounded_estimate + half_width) * 10.0) / 10.0,
    )
    rounded_minimum = max(MIN_PLAUSIBLE_HEIGHT_METERS, rounded_minimum)
    rounded_maximum = min(MAX_PLAUSIBLE_HEIGHT_METERS, rounded_maximum)
    if rounded_maximum - rounded_minimum < target_width:
        if rounded_minimum <= MIN_PLAUSIBLE_HEIGHT_METERS:
            rounded_maximum = min(
                MAX_PLAUSIBLE_HEIGHT_METERS,
                rounded_minimum + target_width,
            )
        elif rounded_maximum >= MAX_PLAUSIBLE_HEIGHT_METERS:
            rounded_minimum = max(
                MIN_PLAUSIBLE_HEIGHT_METERS,
                rounded_maximum - target_width,
            )
    return tuple(
        round(value, 1)
        for value in (rounded_estimate, rounded_minimum, rounded_maximum)
    )


def _raise_if_cancelled(
    cancellation_check: Callable[[], bool] | None,
) -> None:
    if cancellation_check is not None and cancellation_check():
        raise CeilingHeightEstimationCancelled(
            "Ceiling height inference was cancelled."
        )


