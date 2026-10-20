# ### Imports ###
from __future__ import annotations

import copy
import json
import queue
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from housemaker.openai_batch import (
    LUNA_BATCH_MODEL,
    OpenAIBatchCancelled,
    OpenAIBatchError,
    request_luna_batch_response,
    to_openai_model,
)
from housemaker.plan_correction_models import (
    PLAN_CORRECTION_MODEL_GPT_5_6_LUNA,
    PLAN_CORRECTION_MODEL_GPT_5_6_TERRA,
    PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
    PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1,
)
from housemaker.plan_metric_batch import (
    MULTI_PLAN_MEASUREMENT_REQUEST_NAME,
    EncodedPlanMeasurementPair,
    MultiPlanMeasurementAnalyzer,
    plan_measurement_schema_for_id,
    validate_encoded_plan_pairs,
)
from housemaker.plan_metric_calibration import (
    PLAN_MEASUREMENT_PROMPT,
    CancellationCheck,
    PlanMeasurementAnalyzer,
    PlanMetricCalibrationCancelled,
    PlanMetricCalibrationError,
    PlanMetricCalibrationInferenceError,
)
from housemaker.plan_metric_views import PlanMeasurementDetailImage
from housemaker.qwen_ceiling_height_estimation import (
    QwenCeilingHeightCancelled,
    QwenCeilingHeightError,
    QwenCeilingHeightServer,
)

# ### Provider constants ###
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPENAI_NETWORK_TIMEOUT_SECONDS = 120.0
OPENAI_MAX_RESPONSE_BYTES = 1 * 1024 * 1024
OPENAI_MAX_OUTPUT_TOKENS = 6_144
OPENAI_MULTI_PLAN_MAX_OUTPUT_TOKENS = 16_000
CANCELLATION_POLL_SECONDS = 0.05
LOCAL_QWEN_MAX_MEASUREMENTS = 12
LOCAL_QWEN_CONTEXT_TOKENS = 32_768
LOCAL_QWEN_MAX_OUTPUT_TOKENS = 6_144
PLAN_MEASUREMENT_REQUEST_NAME = "architectural_plan_measurements"


# ### Public protocols ###
class UrlOpenFunction(Protocol):
    def __call__(self, request: Request, **kwargs: Any) -> object: ...


class LocalQwenServer(Protocol):
    def __enter__(self) -> LocalQwenServer: ...  # noqa: PYI034

    def __exit__(self, *_args: object) -> None: ...

    def estimate(
        self,
        image_data_uri: str,
        *,
        previous_image_data_uri: str | None,
        plan_image_data_uri: str | None = None,
        additional_images: Sequence[tuple[str, str]] = (),
        max_output_tokens: int = LOCAL_QWEN_MAX_OUTPUT_TOKENS,
        prompt: str,
        result_schema: Mapping[str, Any],
        cancellation_check: CancellationCheck | None,
    ) -> object: ...


LocalQwenServerFactory = Callable[..., LocalQwenServer]


# ### Provider selection ###
def plan_measurement_model_for_correction_model(
    correction_model: str,
) -> str:
    """Return the vision model used to read dimensions after correction."""

    selected_model = str(correction_model).strip()
    if selected_model == PLAN_CORRECTION_MODEL_GPT_IMAGE_2:
        return PLAN_CORRECTION_MODEL_GPT_5_6_LUNA
    if selected_model in {
        PLAN_CORRECTION_MODEL_GPT_5_6_LUNA,
        LUNA_BATCH_MODEL,
        PLAN_CORRECTION_MODEL_GPT_5_6_TERRA,
        PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1,
    }:
        return selected_model
    raise PlanMetricCalibrationInferenceError(
        "The selected plan correction model cannot transcribe measurements."
    )


def create_plan_measurement_analyzer(
    correction_model: str,
    *,
    api_key: str = "",
    opener: UrlOpenFunction | None = None,
    qwen_server_factory: LocalQwenServerFactory | None = None,
) -> PlanMeasurementAnalyzer:
    """Create a provider-neutral analyzer without starting local inference."""

    measurement_model = plan_measurement_model_for_correction_model(
        correction_model
    )
    if measurement_model == PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1:
        return LocalQwenPlanMeasurementAnalyzer(
            server_factory=qwen_server_factory,
        )
    return OpenAIPlanMeasurementAnalyzer(
        measurement_model,
        api_key=api_key,
        opener=opener,
    )


def create_multi_plan_measurement_analyzer(
    correction_model: str,
    *,
    api_key: str = "",
    opener: UrlOpenFunction | None = None,
    qwen_server_factory: LocalQwenServerFactory | None = None,
) -> MultiPlanMeasurementAnalyzer:
    """Create one grouped analyzer without starting network or local inference."""

    measurement_model = plan_measurement_model_for_correction_model(
        correction_model
    )
    if measurement_model == PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1:
        return LocalQwenMultiPlanMeasurementAnalyzer(
            server_factory=qwen_server_factory,
        )
    return OpenAIMultiPlanMeasurementAnalyzer(
        measurement_model,
        api_key=api_key,
        opener=opener,
    )


# ### OpenAI analyzer ###
class OpenAIPlanMeasurementAnalyzer:
    """Read explicit plan dimensions through a strict Responses request."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        opener: UrlOpenFunction | None = None,
    ) -> None:
        self._model = str(model).strip()
        self._api_key = str(api_key).strip()
        self._opener = urlopen if opener is None else opener

    def __call__(
        self,
        original_image_data_uri: str,
        corrected_image_data_uri: str,
        *,
        measurement_reference_image_data_uri: str | None = None,
        detail_images: Sequence[PlanMeasurementDetailImage] = (),
        prompt: str,
        result_schema: Mapping[str, Any],
        cancellation_check: CancellationCheck | None,
    ) -> object:
        if not self._api_key:
            raise PlanMetricCalibrationInferenceError(
                "Add an OpenAI API key in Settings before correcting the plan."
            )
        _raise_if_cancelled(cancellation_check)
        request_body = _build_openai_request_body(
            original_image_data_uri,
            corrected_image_data_uri,
            measurement_reference_image_data_uri=(
                measurement_reference_image_data_uri
            ),
            detail_images=detail_images,
            prompt=prompt,
            result_schema=result_schema,
            model=self._model,
        )
        if self._model == LUNA_BATCH_MODEL:
            return _request_batch_measurements(
                request_body,
                api_key=self._api_key,
                cancellation_check=cancellation_check,
                opener=self._opener,
            )
        transport_cancelled = threading.Event()

        def request_measurements() -> object:
            return _request_openai_measurements(
                request_body,
                api_key=self._api_key,
                cancellation_check=transport_cancelled.is_set,
                opener=self._opener,
            )

        if cancellation_check is None:
            return request_measurements()
        return _run_cancellably(
            request_measurements,
            cancellation_check,
            transport_cancelled,
        )


class OpenAIMultiPlanMeasurementAnalyzer:
    """Read every labeled plan pair in one shared Responses request."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        opener: UrlOpenFunction | None = None,
    ) -> None:
        self._model = str(model).strip()
        self._api_key = str(api_key).strip()
        self._opener = urlopen if opener is None else opener

    def __call__(
        self,
        plan_pairs: Sequence[EncodedPlanMeasurementPair],
        *,
        prompt: str,
        result_schema: Mapping[str, Any],
        cancellation_check: CancellationCheck | None,
    ) -> object:
        pairs = validate_encoded_plan_pairs(plan_pairs)
        if not self._api_key:
            raise PlanMetricCalibrationInferenceError(
                "Add an OpenAI API key in Settings before correcting the plans."
            )
        _raise_if_cancelled(cancellation_check)
        request_body = _build_openai_multi_plan_request_body(
            pairs,
            prompt=prompt,
            result_schema=result_schema,
            model=self._model,
        )
        if self._model == LUNA_BATCH_MODEL:
            return _request_batch_measurements(
                request_body,
                api_key=self._api_key,
                cancellation_check=cancellation_check,
                opener=self._opener,
            )
        transport_cancelled = threading.Event()

        def request_measurements() -> object:
            return _request_openai_measurements(
                request_body,
                api_key=self._api_key,
                cancellation_check=transport_cancelled.is_set,
                opener=self._opener,
            )

        if cancellation_check is None:
            return request_measurements()
        return _run_cancellably(
            request_measurements,
            cancellation_check,
            transport_cancelled,
        )


# ### Local Qwen analyzer ###
class LocalQwenPlanMeasurementAnalyzer:
    """Lazily start Qwen3-VL for one post-correction transcription."""

    def __init__(
        self,
        *,
        server_factory: LocalQwenServerFactory | None = None,
    ) -> None:
        self._server_factory = (
            QwenCeilingHeightServer
            if server_factory is None
            else server_factory
        )

    def __call__(
        self,
        original_image_data_uri: str,
        corrected_image_data_uri: str,
        *,
        measurement_reference_image_data_uri: str | None = None,
        detail_images: Sequence[PlanMeasurementDetailImage] = (),
        prompt: str,
        result_schema: Mapping[str, Any],
        cancellation_check: CancellationCheck | None,
    ) -> object:
        _raise_if_cancelled(cancellation_check)
        has_measurement_reference = (
            measurement_reference_image_data_uri is not None
        )
        local_prompt = build_local_qwen_plan_measurement_prompt(
            prompt,
            has_measurement_reference=has_measurement_reference,
        )
        local_schema = cap_local_qwen_measurement_schema(result_schema)
        try:
            with self._server_factory(
                cancellation_check=cancellation_check,
                context_tokens=LOCAL_QWEN_CONTEXT_TOKENS,
            ) as server:
                _raise_if_cancelled(cancellation_check)
                estimate_arguments: dict[str, object] = {
                    "previous_image_data_uri": (
                        measurement_reference_image_data_uri
                        if has_measurement_reference
                        else original_image_data_uri
                    ),
                    "prompt": local_prompt,
                    "result_schema": local_schema,
                    "cancellation_check": cancellation_check,
                    "max_output_tokens": LOCAL_QWEN_MAX_OUTPUT_TOKENS,
                }
                if detail_images:
                    estimate_arguments["additional_images"] = _local_detail_images(
                        detail_images
                    )
                if has_measurement_reference:
                    estimate_arguments["plan_image_data_uri"] = (
                        original_image_data_uri
                    )
                return server.estimate(
                    corrected_image_data_uri,
                    **estimate_arguments,
                )
        except (PlanMetricCalibrationCancelled, QwenCeilingHeightCancelled):
            raise PlanMetricCalibrationCancelled(
                "Plan metric calibration was cancelled."
            ) from None
        except (PlanMetricCalibrationError, QwenCeilingHeightError):
            raise PlanMetricCalibrationInferenceError(
                "The local Qwen plan measurement transcription failed."
            ) from None
        except Exception:  # noqa: BLE001 - never expose local process details.
            raise PlanMetricCalibrationInferenceError(
                "The local Qwen plan measurement transcription failed."
            ) from None


class LocalQwenMultiPlanMeasurementAnalyzer:
    """Process grouped pairs sequentially through one local Qwen server."""

    def __init__(
        self,
        *,
        server_factory: LocalQwenServerFactory | None = None,
    ) -> None:
        self._server_factory = (
            QwenCeilingHeightServer
            if server_factory is None
            else server_factory
        )

    def __call__(
        self,
        plan_pairs: Sequence[EncodedPlanMeasurementPair],
        *,
        prompt: str,
        result_schema: Mapping[str, Any],
        cancellation_check: CancellationCheck | None,
    ) -> object:
        pairs = validate_encoded_plan_pairs(plan_pairs)
        if not str(prompt).strip():
            raise PlanMetricCalibrationInferenceError(
                "The multi-plan measurement transcription prompt is empty."
            )
        _raise_if_cancelled(cancellation_check)
        grouped_results: dict[str, object] = {}
        try:
            with self._server_factory(
                cancellation_check=cancellation_check,
                context_tokens=LOCAL_QWEN_CONTEXT_TOKENS,
            ) as server:
                for pair in pairs:
                    _raise_if_cancelled(cancellation_check)
                    single_schema = plan_measurement_schema_for_id(
                        result_schema,
                        pair.plan_id,
                    )
                    local_schema = cap_local_qwen_measurement_schema(
                        single_schema
                    )
                    local_prompt = build_local_qwen_sequential_batch_prompt(
                        pair.plan_id,
                        prompt=prompt,
                        has_measurement_reference=(
                            pair.measurement_reference_image_data_uri
                            is not None
                        ),
                    )
                    has_measurement_reference = (
                        pair.measurement_reference_image_data_uri is not None
                    )
                    estimate_arguments: dict[str, object] = {
                        "previous_image_data_uri": (
                            pair.measurement_reference_image_data_uri
                            if has_measurement_reference
                            else pair.original_image_data_uri
                        ),
                        "prompt": local_prompt,
                        "result_schema": local_schema,
                        "cancellation_check": cancellation_check,
                        "max_output_tokens": LOCAL_QWEN_MAX_OUTPUT_TOKENS,
                    }
                    if pair.detail_images:
                        estimate_arguments["additional_images"] = _local_detail_images(
                            pair.detail_images, plan_id=pair.plan_id
                        )
                    if has_measurement_reference:
                        estimate_arguments["plan_image_data_uri"] = (
                            pair.original_image_data_uri
                        )
                    grouped_results[pair.plan_id] = server.estimate(
                        pair.corrected_image_data_uri,
                        **estimate_arguments,
                    )
        except (PlanMetricCalibrationCancelled, QwenCeilingHeightCancelled):
            raise PlanMetricCalibrationCancelled(
                "Plan metric calibration was cancelled."
            ) from None
        except (PlanMetricCalibrationError, QwenCeilingHeightError):
            raise PlanMetricCalibrationInferenceError(
                "The local Qwen multi-plan measurement transcription failed."
            ) from None
        except Exception:  # noqa: BLE001 - never expose local process details.
            raise PlanMetricCalibrationInferenceError(
                "The local Qwen multi-plan measurement transcription failed."
            ) from None
        return {"plans": grouped_results}


def build_local_qwen_plan_measurement_prompt(
    prompt: str,
    *,
    has_measurement_reference: bool = False,
) -> str:
    """Bound local output while retaining the strongest independent evidence."""

    normalized_prompt = str(prompt).strip()
    if not normalized_prompt:
        raise PlanMetricCalibrationInferenceError(
            "The plan measurement transcription prompt is empty."
        )
    if has_measurement_reference:
        image_roles = (
            "PLAN REFERENCE is the raw ORIGINAL PLAN and is authoritative "
            "for every printed value. PREVIOUS is the RECTIFIED ANNOTATED "
            "PLAN and is only a reading/correspondence aid. CURRENT is the "
            "CORRECTED PLAN (FINAL). Omit a measurement unless its printed "
            "value agrees in PLAN REFERENCE and PREVIOUS."
        )
    else:
        image_roles = (
            "The first image, labeled PREVIOUS, is the MEASUREMENT REFERENCE "
            "PLAN and raw ORIGINAL PLAN; it is authoritative for every "
            "printed value. The second image, labeled CURRENT, is the "
            "CORRECTED PLAN."
        )
    return (
        f"{normalized_prompt}\n\n"
        f"{image_roles} Return no more than "
        f"{LOCAL_QWEN_MAX_MEASUREMENTS} measurements. Keep only the strongest "
        "independent evidence: aim for 8–12 when genuinely readable, including "
        "at least two horizontal and two vertical spans when available, spread "
        "over different regions of the plan. Never invent evidence or duplicate "
        "the same span to reach the target. Prefer long spans with unambiguous printed "
        "values, endpoints, units, and corrected-plan correspondences. Rank "
        "strongest evidence first. Do not spend output tokens explaining the "
        "selection."
    )


def build_local_qwen_sequential_batch_prompt(
    plan_id: str,
    *,
    prompt: str = "",
    has_measurement_reference: bool = False,
) -> str:
    """Describe the local fallback accurately without claiming shared vision."""

    return build_local_qwen_plan_measurement_prompt(
        f"{PLAN_MEASUREMENT_PROMPT}\n\n"
        f"Additional grouped analysis instructions: {str(prompt).strip()}\n\n"
        f"This request contains only PLAN ID {plan_id}. Process this pair "
        "independently. Other plan pairs are not visible in the local fallback, "
        "so do not claim that their notation, measurements, or geometry were "
        "cross-checked. Apply any retry instructions for this plan ID. Return "
        "only this plan's result in the supplied single-plan schema, not a "
        "wrapper containing a plans mapping.",
        has_measurement_reference=has_measurement_reference,
    )


def cap_local_qwen_measurement_schema(
    result_schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy a result schema and bound its measurement array for local context."""

    if not isinstance(result_schema, Mapping):
        raise PlanMetricCalibrationInferenceError(
            "The plan measurement result schema is invalid."
        )
    schema = copy.deepcopy(dict(result_schema))
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return schema
    mutable_properties = dict(properties)
    measurements = mutable_properties.get("measurements")
    if not isinstance(measurements, Mapping):
        schema["properties"] = mutable_properties
        return schema
    mutable_measurements = dict(measurements)
    configured_limit = mutable_measurements.get("maxItems")
    if isinstance(configured_limit, int) and not isinstance(
        configured_limit,
        bool,
    ):
        limit = min(LOCAL_QWEN_MAX_MEASUREMENTS, configured_limit)
    else:
        limit = LOCAL_QWEN_MAX_MEASUREMENTS
    mutable_measurements["maxItems"] = max(1, limit)
    mutable_properties["measurements"] = mutable_measurements
    schema["properties"] = mutable_properties
    return schema


# ### OpenAI request construction ###
def _build_openai_request_body(
    original_image_data_uri: str,
    corrected_image_data_uri: str,
    *,
    measurement_reference_image_data_uri: str | None = None,
    detail_images: Sequence[PlanMeasurementDetailImage] = (),
    prompt: str,
    result_schema: Mapping[str, Any],
    model: str,
) -> dict[str, Any]:
    original_uri = _validated_image_data_uri(
        original_image_data_uri,
        "original plan",
    )
    corrected_uri = _validated_image_data_uri(
        corrected_image_data_uri,
        "corrected plan",
    )
    reference_uri = (
        None
        if measurement_reference_image_data_uri is None
        else _validated_image_data_uri(
            measurement_reference_image_data_uri,
            "rectified annotated plan",
        )
    )
    normalized_prompt = str(prompt).strip()
    if not normalized_prompt:
        raise PlanMetricCalibrationInferenceError(
            "The plan measurement transcription prompt is empty."
        )
    if not isinstance(result_schema, Mapping):
        raise PlanMetricCalibrationInferenceError(
            "The plan measurement result schema is invalid."
        )
    original_label = (
        "ORIGINAL PLAN (RAW):"
        if reference_uri is not None
        else "MEASUREMENT REFERENCE PLAN:"
    )
    corrected_label = (
        "CORRECTED PLAN (FINAL):"
        if reference_uri is not None
        else "CORRECTED PLAN:"
    )
    content: list[dict[str, object]] = [
        {"type": "input_text", "text": normalized_prompt},
        {"type": "input_text", "text": original_label},
        {
            "type": "input_image",
            "image_url": original_uri,
            "detail": "original",
        },
    ]
    if reference_uri is not None:
        content.extend(
            (
                {
                    "type": "input_text",
                    "text": "RECTIFIED ANNOTATED PLAN:",
                },
                {
                    "type": "input_image",
                    "image_url": reference_uri,
                    "detail": "original",
                },
            )
        )
    content.extend(
        (
            {"type": "input_text", "text": corrected_label},
            {
                "type": "input_image",
                "image_url": corrected_uri,
                "detail": "original",
            },
        )
    )
    _append_openai_detail_images(content, detail_images)
    return {
        "model": to_openai_model(model),
        "store": False,
        "reasoning": {"effort": "none"},
        "input": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": PLAN_MEASUREMENT_REQUEST_NAME,
                "strict": True,
                "schema": copy.deepcopy(dict(result_schema)),
            }
        },
        "max_output_tokens": OPENAI_MAX_OUTPUT_TOKENS,
    }


def _build_openai_multi_plan_request_body(
    plan_pairs: Sequence[EncodedPlanMeasurementPair],
    *,
    prompt: str,
    result_schema: Mapping[str, Any],
    model: str,
) -> dict[str, Any]:
    """Build one shared request while preserving every plan-pair boundary."""

    pairs = validate_encoded_plan_pairs(plan_pairs)
    normalized_prompt = str(prompt).strip()
    if not normalized_prompt:
        raise PlanMetricCalibrationInferenceError(
            "The multi-plan measurement transcription prompt is empty."
        )
    if not isinstance(result_schema, Mapping):
        raise PlanMetricCalibrationInferenceError(
            "The multi-plan measurement result schema is invalid."
        )
    content: list[dict[str, object]] = [
        {"type": "input_text", "text": normalized_prompt}
    ]
    for pair in pairs:
        original_uri = _validated_image_data_uri(
            pair.original_image_data_uri,
            f"original plan {pair.plan_id}",
        )
        corrected_uri = _validated_image_data_uri(
            pair.corrected_image_data_uri,
            f"corrected plan {pair.plan_id}",
        )
        reference_uri = (
            None
            if pair.measurement_reference_image_data_uri is None
            else _validated_image_data_uri(
                pair.measurement_reference_image_data_uri,
                f"rectified annotated plan {pair.plan_id}",
            )
        )
        original_label = (
            f"PLAN ID {pair.plan_id} - ORIGINAL PLAN (RAW):"
            if reference_uri is not None
            else f"PLAN ID {pair.plan_id} - MEASUREMENT REFERENCE PLAN:"
        )
        corrected_label = (
            f"PLAN ID {pair.plan_id} - CORRECTED PLAN (FINAL):"
            if reference_uri is not None
            else f"PLAN ID {pair.plan_id} - CORRECTED PLAN:"
        )
        content.extend(
            (
                {
                    "type": "input_text",
                    "text": original_label,
                },
                {
                    "type": "input_image",
                    "image_url": original_uri,
                    "detail": "original",
                },
            )
        )
        if reference_uri is not None:
            content.extend(
                (
                    {
                        "type": "input_text",
                        "text": (
                            f"PLAN ID {pair.plan_id} - "
                            "RECTIFIED ANNOTATED PLAN:"
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": reference_uri,
                        "detail": "original",
                    },
                )
            )
        content.extend(
            (
                {
                    "type": "input_text",
                    "text": corrected_label,
                },
                {
                    "type": "input_image",
                    "image_url": corrected_uri,
                    "detail": "original",
                },
            )
        )
        _append_openai_detail_images(content, pair.detail_images, plan_id=pair.plan_id)
    return {
        "model": to_openai_model(str(model).strip()),
        "store": False,
        "reasoning": {"effort": "none"},
        "input": [{"role": "user", "content": content}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": MULTI_PLAN_MEASUREMENT_REQUEST_NAME,
                "strict": True,
                "schema": copy.deepcopy(dict(result_schema)),
            }
        },
        "max_output_tokens": min(
            OPENAI_MULTI_PLAN_MAX_OUTPUT_TOKENS,
            OPENAI_MAX_OUTPUT_TOKENS * len(pairs),
        ),
    }


# ### Detail-image request helpers ###
def _local_detail_images(
    details: Sequence[PlanMeasurementDetailImage], *, plan_id: str = "",
) -> tuple[tuple[str, str], ...]:
    """Keep crop provenance attached in both remote and local requests."""

    images: list[tuple[str, str]] = []
    for detail in details:
        if not isinstance(detail, PlanMeasurementDetailImage) or not detail.label.strip():
            raise PlanMetricCalibrationInferenceError(
                "The plan measurement detail image is invalid."
            )
        label = f"PLAN ID {plan_id} - {detail.label}" if plan_id else detail.label
        images.append(
            (label, _validated_image_data_uri(detail.image_data_uri, "plan detail"))
        )
    return tuple(images)


def _append_openai_detail_images(
    content: list[dict[str, object]],
    details: Sequence[PlanMeasurementDetailImage],
    *,
    plan_id: str = "",
) -> None:
    for label, image_uri in _local_detail_images(details, plan_id=plan_id):
        content.extend(
            (
                {"type": "input_text", "text": label},
                {"type": "input_image", "image_url": image_uri, "detail": "original"},
            )
        )


# ### OpenAI transport ###
def _request_batch_measurements(
    request_body: Mapping[str, Any],
    *,
    api_key: str,
    cancellation_check: CancellationCheck | None,
    opener: UrlOpenFunction,
) -> object:
    """Keep the Batch wait cancellable without a regular transport lock."""

    try:
        envelope = request_luna_batch_response(
            request_body,
            api_key=api_key,
            cancellation_check=cancellation_check,
            opener=opener,
        )
        result = json.loads(_extract_openai_output_text(envelope))
        if not isinstance(result, Mapping):
            raise PlanMetricCalibrationInferenceError(
                "GPT-5.6 Luna Batch API returned an invalid plan transcription."
            )
        return result
    except OpenAIBatchCancelled as error:
        raise PlanMetricCalibrationCancelled(str(error)) from None
    except OpenAIBatchError as error:
        raise PlanMetricCalibrationInferenceError(str(error)) from None
    except (ValueError, TypeError):
        raise PlanMetricCalibrationInferenceError(
            "GPT-5.6 Luna Batch API returned unusable plan transcription JSON."
        ) from None


def _request_openai_measurements(
    request_body: Mapping[str, Any],
    *,
    api_key: str,
    cancellation_check: CancellationCheck | None,
    opener: UrlOpenFunction,
) -> object:
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
    try:
        _raise_if_cancelled(cancellation_check)
        with opener(
            request,
            timeout=OPENAI_NETWORK_TIMEOUT_SECONDS,
        ) as response:
            raw_response = _read_response_limited(
                response,
                cancellation_check,
            )
        response_payload = json.loads(raw_response.decode("utf-8"))
        output_text = _extract_openai_output_text(response_payload)
        result = json.loads(output_text)
        if not isinstance(result, Mapping):
            raise PlanMetricCalibrationInferenceError(
                "OpenAI returned an invalid plan measurement transcription."
            )
        return result
    except PlanMetricCalibrationCancelled:
        raise
    except PlanMetricCalibrationError:
        raise
    except HTTPError as error:
        raise PlanMetricCalibrationInferenceError(
            "OpenAI plan measurement transcription was rejected "
            f"({int(error.code)})."
        ) from None
    except (URLError, TimeoutError, OSError):
        raise PlanMetricCalibrationInferenceError(
            "Unable to reach OpenAI for plan measurement transcription."
        ) from None
    except (UnicodeError, ValueError, TypeError, json.JSONDecodeError):
        raise PlanMetricCalibrationInferenceError(
            "OpenAI did not return a usable plan measurement transcription."
        ) from None


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
        if total > OPENAI_MAX_RESPONSE_BYTES:
            raise PlanMetricCalibrationInferenceError(
                "OpenAI returned an unexpectedly large transcription response."
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _run_cancellably(
    operation: Callable[[], object],
    cancellation_check: CancellationCheck,
    transport_cancelled: threading.Event,
) -> object:
    """Return promptly when a caller cancels a blocked network request."""

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
        name="plan-measurement-openai-request",
        daemon=True,
    ).start()
    while True:
        if cancellation_check():
            transport_cancelled.set()
            raise PlanMetricCalibrationCancelled(
                "Plan metric calibration was cancelled."
            )
        try:
            succeeded, value = outcome.get(
                timeout=CANCELLATION_POLL_SECONDS
            )
        except queue.Empty:
            continue
        if cancellation_check():
            transport_cancelled.set()
            raise PlanMetricCalibrationCancelled(
                "Plan metric calibration was cancelled."
            )
        if succeeded:
            return value
        if isinstance(value, Exception):
            raise value
        raise PlanMetricCalibrationInferenceError(
            "Plan measurement transcription finished without a result."
        )


# ### OpenAI response parsing ###
def _extract_openai_output_text(payload: object) -> str:
    if not isinstance(payload, Mapping):
        raise PlanMetricCalibrationInferenceError(
            "OpenAI returned an invalid plan measurement response."
        )
    if payload.get("status") != "completed":
        raise PlanMetricCalibrationInferenceError(
            "OpenAI did not complete plan measurement transcription."
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
                    raise PlanMetricCalibrationInferenceError(
                        "OpenAI declined to transcribe plan measurements."
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
    raise PlanMetricCalibrationInferenceError(
        "OpenAI completed without a plan measurement transcription."
    )


# ### Validation helpers ###
def _validated_image_data_uri(value: object, label: str) -> str:
    uri = str(value).strip()
    if not uri.startswith("data:image/") or ";base64," not in uri:
        raise PlanMetricCalibrationInferenceError(
            f"The {label} could not be prepared for measurement transcription."
        )
    return uri


def _raise_if_cancelled(
    cancellation_check: CancellationCheck | None,
) -> None:
    if cancellation_check is not None and bool(cancellation_check()):
        raise PlanMetricCalibrationCancelled(
            "Plan metric calibration was cancelled."
        )
