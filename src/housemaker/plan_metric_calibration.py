# ### Imports ###
from __future__ import annotations

import base64
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

import cv2
import numpy as np

from housemaker.models import PIXEL_TO_METER
from housemaker.plan_metric_refinement import (
    PlanMeasurementRefinementCancelled,
    PlanMeasurementRefinementResult,
    refine_plan_measurements,
)
from housemaker.plan_metric_state import (
    MAX_PLAN_MEASUREMENT_COUNT,
    PlanMeasurementEvidence,
    PlanMetricCalibration,
    PlanMetricDiagnostics,
)
from housemaker.plan_metric_views import (
    PlanMeasurementDetailCancelled,
    PlanMeasurementDetailImage,
    build_plan_measurement_detail_images,
)

# ### Calibration constants ###
PLAN_MEASUREMENT_PROMPT = """Analyze two or three images of the same architectural
floor plan.

The ORIGINAL PLAN is the raw photograph and the authoritative source for every
printed measurement value. A RECTIFIED ANNOTATED PLAN may also be supplied as an
AI-produced reading and correspondence aid. The CORRECTED PLAN is the final clean,
orthographic plan in which measurement annotations may have been removed.

Never trust a value only because it appears in the rectified annotated plan. When
that optional image is present, verify every printed number and unit against the raw
original. If the two readings disagree, or the value cannot be confirmed in the raw
original, omit the measurement. Do not repair or guess a conflicting value.

Transcribe only explicit dimensions that describe distances in the horizontal floor
plane from the ORIGINAL PLAN. Include valid dimensions whose text or dimension line
is vertical on the page. Ignore room names, reference numbers,
quantities, angles, elevations, ceiling heights, and vertical section dimensions.
Convert every retained distance to metres. For an unlabeled number, infer
centimetres, millimetres, or metres only when the drawing convention and
surrounding measurements make that interpretation clear; otherwise omit it.

For each measurement, identify the two endpoints of the measured span and the
corresponding anchors in the corrected plan. When a RECTIFIED ANNOTATED PLAN is
present, start_normalized and end_normalized must be relative to that image;
otherwise they must be relative to the ORIGINAL PLAN. corrected_start_normalized
and corrected_end_normalized must always be relative to the CORRECTED PLAN. Use
dimension ticks, arrowheads, or witness-line intersections for the annotated
endpoints. Both endpoint pairs must refer to exactly the same physical span.
Coordinates are normalized from 0 to 1 relative to their image, with [0, 0] at the
top-left and [1, 1] at the bottom-right.

For label, write a concise semantic description of that span, such as "Salon
interior width between the left and right walls". Use nearby room or feature names
and architectural anchors when available; do not merely repeat the printed number.

Infer the drawing's unit convention from all of its dimensions before converting
any value. Cross-check the conversion against physically plausible room and building
sizes. In particular, do not blindly treat every bare integer as millimetres: on a
residential plan, bare values such as 610 and 390 may denote centimetres (6.10 m and
3.90 m), while values such as 6100 and 3900 may denote millimetres. If no single
unit convention produces a plausible and internally consistent floor plan, omit the
ambiguous measurements instead of guessing.

Aim for 8–12 independent long measurements when the drawing provides them,
including at least two horizontal and two vertical spans spread across different
rooms and regions. Prefer overall dimensions as well as room dimensions. A repeated
reading of the same span is not independent evidence. Do not invent measurements
to meet a quota. Explicitly distinguish inside wall faces, outside wall faces, and
wall centerlines: the endpoints must match the printed dimension's meaning.
Detail crops, when supplied, are magnifications of the labeled source rectangle.
Return coordinates in the full annotated/reference and final images described
above, never relative to a crop. Use the overviews to associate each crop with its
correct physical span.

Do not estimate distances from standard object sizes. Do not invent obscured values
or correspondences. Assign high confidence only when the printed value, units or drawing
convention, endpoints, and corrected-image correspondence are all unambiguous."""

PLAN_MEASUREMENT_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "measurements": {
            "type": "array",
            "maxItems": MAX_PLAN_MEASUREMENT_COUNT,
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "value_meters": {"type": "number", "exclusiveMinimum": 0},
                    "start_normalized": {
                        "type": "array",
                        "items": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "end_normalized": {
                        "type": "array",
                        "items": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "corrected_start_normalized": {
                        "type": "array",
                        "items": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "corrected_end_normalized": {
                        "type": "array",
                        "items": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                },
                "required": [
                    "label",
                    "value_meters",
                    "start_normalized",
                    "end_normalized",
                    "corrected_start_normalized",
                    "corrected_end_normalized",
                    "confidence",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["measurements"],
    "additionalProperties": False,
}

PLAN_MEASUREMENT_REQUEST_NAME = "architectural_plan_measurements"
DEFAULT_PLAN_MEASUREMENT_MODEL = "gpt-5.6-luna"
MAX_TRANSCRIPTION_IMAGE_EDGE_PIXELS = 2_048
MAX_TRANSCRIPTION_IMAGE_BYTES = 16 * 1024 * 1024
MAX_NORMALIZED_PLAN_EDGE_PIXELS = 16_384
MAX_NORMALIZED_PLAN_PIXEL_COUNT = 100_000_000
MIN_NORMALIZED_PLAN_EDGE_PIXELS = 2
MIN_CORRECTED_MEASUREMENT_SPAN_PIXELS = 24.0
CONSENSUS_RELATIVE_TOLERANCE = 0.12
MAX_ACCEPTED_RELATIVE_RESIDUAL = 0.15
MAX_AXIS_SCALE_DIFFERENCE = 0.10
MIN_RELIABLE_PLAN_MEASUREMENTS = 2
MAX_DUPLICATE_ENDPOINT_DISTANCE = 0.02
MAX_DUPLICATE_VALUE_DIFFERENCE = 0.05
MAX_CROSS_PLAN_VALUE_DIFFERENCE = 0.05
MAX_CROSS_PLAN_SUPPORT_MULTIPLIER = 2.0
CROSS_PLAN_SUPPORT_WEIGHT_STEP = 0.25
CROSS_PLAN_UNIT_RATIO_TOLERANCE = 0.08
MIN_CROSS_PLAN_UNIT_CONSENSUS = 2
MIN_CROSS_PLAN_LABEL_WORDS = 3
MIN_RELIABLE_REFERENCE_SPAN_METERS = 0.9
MIN_PLAUSIBLE_PLAN_LONG_EDGE_METERS = 2.0
MIN_PLAUSIBLE_PLAN_AREA_SQUARE_METERS = 2.0
MAX_PLAUSIBLE_PLAN_LONG_EDGE_METERS = 500.0
MAX_PLAUSIBLE_PLAN_AREA_SQUARE_METERS = 100_000.0
_GEOMETRY_EPSILON = 1e-12
_CONFIDENCE_WEIGHTS = {
    "low": 0.35,
    "medium": 0.7,
    "high": 1.0,
}
_CROSS_PLAN_LABEL_IGNORED_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "between",
        "from",
        "in",
        "of",
        "on",
        "plan",
        "the",
        "to",
    }
)


# ### Public protocols and models ###
CancellationCheck = Callable[[], bool]


class PlanMeasurementAnalyzer(Protocol):
    """Provider-neutral structured vision analyzer used by plan correction."""

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
    ) -> object: ...


@dataclass(frozen=True)
class NormalizedPlanMetricResult:
    """A corrected raster and its own metric density.

    The historical result name is retained for correction-worker compatibility.
    By default the raster is unchanged; calibration controls coordinate conversion
    instead of reducing every plan to a shared pixel density.
    """

    image: np.ndarray
    calibration: PlanMetricCalibration


@dataclass(frozen=True)
class PlanMetricEvidenceSet:
    """Measurements and corrected-raster dimensions for one plan."""

    plan_key: str
    corrected_image_size: tuple[int, int]
    measurements: tuple[PlanMeasurementEvidence, ...]

    def __post_init__(self) -> None:
        plan_key = str(self.plan_key).strip()
        if not plan_key:
            raise ValueError("A plan metric evidence key is required.")
        image_size = _validated_image_size(
            self.corrected_image_size,
            "corrected plan",
        )
        measurements = tuple(self.measurements)
        if not measurements:
            raise ValueError("Plan metric evidence requires measurements.")
        if len(measurements) > MAX_PLAN_MEASUREMENT_COUNT:
            raise ValueError("Plan metric evidence contains too many measurements.")
        if not all(isinstance(item, PlanMeasurementEvidence) for item in measurements):
            raise TypeError("Plan metric evidence contains invalid measurements.")
        object.__setattr__(self, "plan_key", plan_key)
        object.__setattr__(self, "corrected_image_size", image_size)
        object.__setattr__(self, "measurements", measurements)


@dataclass(frozen=True)
class PlanMetricScaleSolution:
    """One locally solved plan scale after deterministic batch reconciliation."""

    plan_key: str
    estimated_meters_per_pixel: float
    relative_residual: float
    confidence: str
    measurements: tuple[PlanMeasurementEvidence, ...]
    corroborated_measurement_count: int = 0
    rejected_unit_mismatch_count: int = 0
    diagnostics: PlanMetricDiagnostics | None = None


@dataclass(frozen=True)
class PlanMetricScaleFailure:
    """One plan that could not be calibrated safely within a batch."""

    plan_key: str
    reason: str
    rejected_unit_mismatch_count: int = 0


@dataclass(frozen=True)
class PlanMetricBatchResult:
    """Independent plan-scale solutions and safe per-plan failures."""

    solutions: tuple[PlanMetricScaleSolution, ...]
    failures: tuple[PlanMetricScaleFailure, ...]


@dataclass(frozen=True)
class _MeasurementCandidate:
    evidence: PlanMeasurementEvidence
    meters_per_pixel: float
    weight: float
    axis: str | None


# ### Public exceptions ###
class PlanMetricCalibrationError(RuntimeError):
    """Failure text safe to display in the application UI."""


class PlanMetricCalibrationCancelled(PlanMetricCalibrationError):
    pass


class PlanMetricCalibrationInferenceError(PlanMetricCalibrationError):
    pass


# ### Structured request construction ###
def build_plan_measurement_request_body(
    original_image_data_uri: str,
    corrected_image_data_uri: str,
    *,
    measurement_reference_image_data_uri: str | None = None,
    model: str = DEFAULT_PLAN_MEASUREMENT_MODEL,
) -> dict[str, Any]:
    """Build one strict raw/reference/final transcription request."""

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
    selected_model = str(model).strip()
    if not selected_model:
        raise PlanMetricCalibrationError(
            "A plan measurement transcription model is required."
        )
    original_label = (
        "ORIGINAL PLAN (RAW):"
        if reference_uri is not None
        else "MEASUREMENT REFERENCE PLAN:"
    )
    corrected_label = (
        "CORRECTED PLAN (FINAL):" if reference_uri is not None else "CORRECTED PLAN:"
    )
    content: list[dict[str, object]] = [
        {"type": "input_text", "text": PLAN_MEASUREMENT_PROMPT},
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
    return {
        "model": selected_model,
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
                "schema": PLAN_MEASUREMENT_RESULT_SCHEMA,
            }
        },
        "max_output_tokens": 4_000,
    }


# ### Public calibration pipeline ###
def calibrate_and_normalize_plan(
    original_image: np.ndarray,
    corrected_image: np.ndarray,
    analyzer: PlanMeasurementAnalyzer,
    *,
    measurement_reference_image: np.ndarray | None = None,
    cancellation_check: CancellationCheck | None = None,
    target_meters_per_pixel: float | None = None,
) -> NormalizedPlanMetricResult:
    """Calibrate the native corrected raster while retaining raw-source metadata.

    No resampling occurs unless a caller explicitly requests a target density.
    Plan correction leaves this unset so every level retains its image detail.
    """

    source = _validated_image(original_image, "original plan")
    corrected = _validated_image(corrected_image, "corrected plan")
    measurements = transcribe_plan_measurements(
        source,
        corrected,
        analyzer,
        measurement_reference_image=measurement_reference_image,
        cancellation_check=cancellation_check,
    )
    prepared = prepare_plan_metric_measurements(
        measurements,
        corrected,
        measurement_reference_image=measurement_reference_image,
        cancellation_check=cancellation_check,
    )
    image_size = (corrected.shape[1], corrected.shape[0])
    solution, failure = solve_reviewed_plan_metrics("plan", prepared, image_size)
    if solution is None or solution.confidence != "high":
        retry_prompt = build_plan_metric_retry_prompt(
            "plan",
            prepared.measurements,
            solution,
            image_size,
            geometry_warning=prepared.geometry_warning,
        )
        try:
            retry = transcribe_plan_measurements(
                source,
                corrected,
                analyzer,
                measurement_reference_image=measurement_reference_image,
                cancellation_check=cancellation_check,
                prompt=PLAN_MEASUREMENT_PROMPT + "\n\n" + retry_prompt,
                focus_regions=plan_metric_retry_regions(
                    prepared.measurements,
                    solution,
                ),
            )
        except PlanMetricCalibrationCancelled:
            raise
        except PlanMetricCalibrationError:
            if solution is not None:
                solution = add_metric_review_note(
                    solution,
                    "Additional measurement analysis was unavailable.",
                )
        else:
            fresh, combined, conflicts = prepare_plan_metric_retry(
                prepared,
                retry,
                corrected,
                measurement_reference_image=measurement_reference_image,
                cancellation_check=cancellation_check,
            )
            fresh_solution, fresh_failure = solve_reviewed_plan_metrics(
                "plan", fresh, image_size, retry_count=1
            )
            combined_solution, combined_failure = solve_reviewed_plan_metrics(
                "plan",
                combined,
                image_size,
                retry_count=1,
            )
            candidate = select_plan_metric_review(
                fresh_solution,
                combined_solution,
                conflicts,
                geometry_warning=combined.geometry_warning,
            )
            solution = select_plan_metric_review(
                solution,
                candidate,
                conflicts,
                geometry_warning=combined.geometry_warning,
            )
            if solution is None:
                failure = fresh_failure or combined_failure or failure
    if solution is None:
        raise PlanMetricCalibrationInferenceError(failure)
    return normalize_plan_with_scale_solution(
        source,
        corrected,
        solution,
        cancellation_check=cancellation_check,
        target_meters_per_pixel=target_meters_per_pixel,
    )


def transcribe_plan_measurements(
    original_image: np.ndarray,
    corrected_image: np.ndarray,
    analyzer: PlanMeasurementAnalyzer,
    *,
    measurement_reference_image: np.ndarray | None = None,
    cancellation_check: CancellationCheck | None = None,
    prompt: str = PLAN_MEASUREMENT_PROMPT,
    focus_regions: Sequence[tuple[float, float, float, float]] = (),
) -> tuple[PlanMeasurementEvidence, ...]:
    """Transcribe one plan from its best annotation-preserving reference."""

    source = _validated_image(original_image, "original plan")
    corrected = _validated_image(corrected_image, "corrected plan")
    measurement_reference = (
        None
        if measurement_reference_image is None
        else _validated_image(
            measurement_reference_image,
            "rectified annotated plan",
        )
    )
    if not callable(analyzer):
        raise TypeError("The plan measurement analyzer must be callable.")
    _raise_if_cancelled(cancellation_check)
    original_uri = encode_plan_measurement_image(source)
    measurement_reference_uri = (
        None
        if measurement_reference is None
        else encode_plan_measurement_image(measurement_reference)
    )
    corrected_uri = encode_plan_measurement_image(corrected)
    details = prepare_plan_metric_detail_images(
        source,
        measurement_reference_image=measurement_reference,
        focus_regions=focus_regions,
        cancellation_check=cancellation_check,
    )
    _raise_if_cancelled(cancellation_check)
    try:
        analyzer_arguments: dict[str, object] = {
            "prompt": prompt,
            "result_schema": PLAN_MEASUREMENT_RESULT_SCHEMA,
            "cancellation_check": cancellation_check,
        }
        if measurement_reference_uri is not None:
            analyzer_arguments["measurement_reference_image_data_uri"] = (
                measurement_reference_uri
            )
        if details:
            analyzer_arguments["detail_images"] = details
        payload = analyzer(original_uri, corrected_uri, **analyzer_arguments)
    except PlanMetricCalibrationCancelled:
        raise
    except PlanMetricCalibrationError:
        raise
    except Exception:  # noqa: BLE001 - never expose provider internals or secrets.
        raise PlanMetricCalibrationInferenceError(
            "Plan measurement transcription failed."
        ) from None
    _raise_if_cancelled(cancellation_check)
    return parse_plan_measurement_result(payload, allow_empty=True)


def normalize_plan_with_scale_solution(
    original_image: np.ndarray,
    corrected_image: np.ndarray,
    solution: PlanMetricScaleSolution,
    *,
    cancellation_check: CancellationCheck | None = None,
    target_meters_per_pixel: float | None = None,
) -> NormalizedPlanMetricResult:
    """Attach a local scale solution without resampling the raster by default."""

    source = _validated_image(original_image, "original plan")
    corrected = _validated_image(corrected_image, "corrected plan")
    if not isinstance(solution, PlanMetricScaleSolution):
        raise TypeError("A plan metric scale solution is required.")
    _raise_if_cancelled(cancellation_check)
    if target_meters_per_pixel is None:
        target_density = solution.estimated_meters_per_pixel
        normalized = corrected.copy()
        normalization_scale = 1.0
    else:
        target_density = _positive_finite_number(
            target_meters_per_pixel,
            "Target plan metres per pixel",
        )
        normalized, normalization_scale = normalize_plan_raster(
            corrected,
            estimated_meters_per_pixel=solution.estimated_meters_per_pixel,
            target_meters_per_pixel=target_density,
        )
    _raise_if_cancelled(cancellation_check)
    calibration = PlanMetricCalibration(
        estimated_meters_per_pixel=solution.estimated_meters_per_pixel,
        target_meters_per_pixel=target_density,
        normalization_scale=normalization_scale,
        relative_residual=solution.relative_residual,
        confidence=solution.confidence,
        source_image_size=(source.shape[1], source.shape[0]),
        pre_normalization_size=(corrected.shape[1], corrected.shape[0]),
        normalized_image_size=(normalized.shape[1], normalized.shape[0]),
        measurements=solution.measurements,
        diagnostics=solution.diagnostics,
    )
    return NormalizedPlanMetricResult(
        image=normalized,
        calibration=calibration,
    )


def encode_plan_measurement_image(image: np.ndarray) -> str:
    """Encode one bounded plan image as a lossless data URI for transcription."""

    validated = _validated_image(image, "plan transcription")
    height, width = validated.shape[:2]
    scale = min(1.0, MAX_TRANSCRIPTION_IMAGE_EDGE_PIXELS / max(width, height))
    if scale < 1.0:
        resized = cv2.resize(
            validated,
            (
                max(2, round(width * scale)),
                max(2, round(height * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )
    else:
        resized = validated
    try:
        encoded_ok, encoded = cv2.imencode(
            ".png",
            resized,
            [cv2.IMWRITE_PNG_COMPRESSION, 3],
        )
    except cv2.error:
        encoded_ok = False
        encoded = None
    if not encoded_ok or encoded is None:
        raise PlanMetricCalibrationError(
            "A plan image could not be prepared for measurement transcription."
        )
    payload = encoded.tobytes()
    if not payload or len(payload) > MAX_TRANSCRIPTION_IMAGE_BYTES:
        raise PlanMetricCalibrationError(
            "A plan image is too large for measurement transcription."
        )
    return "data:image/png;base64," + base64.b64encode(payload).decode("ascii")


def parse_plan_measurement_result(
    payload: object,
    *,
    allow_empty: bool = False,
) -> tuple[PlanMeasurementEvidence, ...]:
    """Validate a strict structured transcription payload."""

    parsed = payload
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError:
            raise PlanMetricCalibrationInferenceError(
                "The plan measurement transcription was not valid JSON."
            ) from None
    if not isinstance(parsed, Mapping):
        raise PlanMetricCalibrationInferenceError(
            "The plan measurement transcription was invalid."
        )
    raw_measurements = parsed.get("measurements")
    if not isinstance(raw_measurements, Sequence) or isinstance(
        raw_measurements,
        (str, bytes, bytearray),
    ):
        raise PlanMetricCalibrationInferenceError(
            "The plan measurement transcription omitted its measurements."
        )
    if len(raw_measurements) > MAX_PLAN_MEASUREMENT_COUNT:
        raise PlanMetricCalibrationInferenceError(
            "The plan measurement transcription returned too many measurements."
        )
    measurements: list[PlanMeasurementEvidence] = []
    for raw_measurement in raw_measurements:
        try:
            measurement = PlanMeasurementEvidence.from_dict(raw_measurement)
        except (TypeError, ValueError, OverflowError):
            raise PlanMetricCalibrationInferenceError(
                "The plan measurement transcription contained invalid evidence."
            ) from None
        if measurement not in measurements:
            measurements.append(measurement)
    if not measurements and not allow_empty:
        raise PlanMetricCalibrationInferenceError(
            "No explicit usable floor-plan measurements were found."
        )
    return tuple(measurements)


def solve_plan_metric_scale(
    measurements: Sequence[PlanMeasurementEvidence],
    *,
    corrected_image_size: tuple[int, int],
) -> tuple[
    float,
    float,
    str,
    tuple[PlanMeasurementEvidence, ...],
]:
    """Return a robust isotropic metres-per-pixel solution and its inliers."""

    width, height = _validated_image_size(
        corrected_image_size,
        "corrected plan",
    )
    candidates = _build_measurement_candidates(measurements, width, height)
    return _solve_plan_metric_candidates(
        candidates,
        width=width,
        height=height,
    )


def solve_plan_metric_scale_batch(
    evidence_sets: Sequence[PlanMetricEvidenceSet],
) -> PlanMetricBatchResult:
    """Reconcile corroborating evidence while preserving each local scale.

    Raw metres-per-pixel values are never combined across plans because photos
    may use unrelated crops and resolutions. Repeated semantic measurements can
    only weight a local consensus or reject a corroborated power-of-ten unit
    mistake. Every successful plan still requires two reliable local spans.
    """

    if isinstance(evidence_sets, (str, bytes, bytearray)):
        raise TypeError("Plan metric evidence sets must form a sequence.")
    plans = tuple(evidence_sets)
    if not plans:
        raise ValueError("At least one plan metric evidence set is required.")
    if not all(isinstance(item, PlanMetricEvidenceSet) for item in plans):
        raise TypeError("Plan metric evidence sets contain an invalid item.")
    plan_keys = tuple(item.plan_key for item in plans)
    if len(set(plan_keys)) != len(plan_keys):
        raise ValueError("Plan metric evidence keys must be unique.")

    candidates_by_plan = {
        plan.plan_key: _build_measurement_candidates(
            plan.measurements,
            *plan.corrected_image_size,
        )
        for plan in plans
    }
    rejected_by_plan = _cross_plan_unit_mismatches(candidates_by_plan)
    corroboration = _cross_plan_corroboration(candidates_by_plan)
    solutions: list[PlanMetricScaleSolution] = []
    failures: list[PlanMetricScaleFailure] = []
    for plan in plans:
        rejected = rejected_by_plan.get(plan.plan_key, frozenset())
        has_cross_plan_support = any(
            corroboration.get((plan.plan_key, candidate.evidence), 0) > 0
            for candidate in candidates_by_plan[plan.plan_key]
            if candidate.evidence not in rejected
        )
        candidates = tuple(
            _candidate_with_cross_plan_support(
                candidate,
                corroboration.get((plan.plan_key, candidate.evidence), 0),
            )
            for candidate in candidates_by_plan[plan.plan_key]
            if candidate.evidence not in rejected
        )
        try:
            estimate, residual, confidence, accepted = _solve_plan_metric_candidates(
                candidates,
                width=plan.corrected_image_size[0],
                height=plan.corrected_image_size[1],
                allow_weighted_tie_break=has_cross_plan_support,
            )
        except PlanMetricCalibrationInferenceError as error:
            failures.append(
                PlanMetricScaleFailure(
                    plan_key=plan.plan_key,
                    reason=str(error),
                    rejected_unit_mismatch_count=len(rejected),
                )
            )
            continue
        corroborated_count = sum(
            corroboration.get((plan.plan_key, measurement), 0) > 0
            for measurement in accepted
        )
        solutions.append(
            PlanMetricScaleSolution(
                plan_key=plan.plan_key,
                estimated_meters_per_pixel=estimate,
                relative_residual=residual,
                confidence=confidence,
                measurements=accepted,
                corroborated_measurement_count=corroborated_count,
                rejected_unit_mismatch_count=len(rejected),
            )
        )
    return PlanMetricBatchResult(
        solutions=tuple(solutions),
        failures=tuple(failures),
    )


def _solve_plan_metric_candidates(
    candidates: Sequence[_MeasurementCandidate],
    *,
    width: int,
    height: int,
    allow_weighted_tie_break: bool = False,
) -> tuple[
    float,
    float,
    str,
    tuple[PlanMeasurementEvidence, ...],
]:
    """Solve already validated candidates for one corrected raster only."""

    if not candidates:
        raise PlanMetricCalibrationInferenceError(
            "The transcribed plan measurements have no measurable corrected spans."
        )
    accepted = _select_consensus_candidates(
        candidates,
        allow_weighted_tie_break=allow_weighted_tie_break,
    )
    if not accepted:
        raise PlanMetricCalibrationInferenceError(
            "The transcribed plan measurements do not agree on a reliable scale. "
            + _consensus_failure_details(candidates)
        )
    reliable_count = sum(
        candidate.evidence.confidence in {"medium", "high"} for candidate in accepted
    )
    if reliable_count < MIN_RELIABLE_PLAN_MEASUREMENTS:
        raise PlanMetricCalibrationInferenceError(
            "At least two consistent, reliable plan measurements are required "
            "before the corrected plan can be rescaled."
        )
    estimate = _weighted_geometric_mean(
        tuple(candidate.meters_per_pixel for candidate in accepted),
        tuple(candidate.weight for candidate in accepted),
    )
    residual = _relative_root_mean_square_error(accepted, estimate)
    if residual > MAX_ACCEPTED_RELATIVE_RESIDUAL:
        raise PlanMetricCalibrationInferenceError(
            "The transcribed plan measurements do not agree on a reliable scale."
        )
    _validate_isotropic_scale(accepted)
    _validate_physical_plan_scale(
        accepted,
        estimate,
        width=width,
        height=height,
    )
    confidence = _calibration_confidence(accepted, residual)
    if len(accepted) * 2 <= len(candidates) and confidence == "high":
        # A strong minority can establish a usable scale, but the unresolved
        # readings must remain visible in its confidence rather than disappear.
        confidence = "medium"
    return (
        estimate,
        residual,
        confidence,
        tuple(candidate.evidence for candidate in accepted),
    )


def normalize_plan_raster(
    image: np.ndarray,
    *,
    estimated_meters_per_pixel: float,
    target_meters_per_pixel: float = PIXEL_TO_METER,
) -> tuple[np.ndarray, float]:
    """Uniformly resample a complete raster to the requested metric density."""

    validated = _validated_image(image, "corrected plan")
    estimated_density = _positive_finite_number(
        estimated_meters_per_pixel,
        "Estimated plan metres per pixel",
    )
    target_density = _positive_finite_number(
        target_meters_per_pixel,
        "Target plan metres per pixel",
    )
    normalization_scale = estimated_density / target_density
    output_width = max(1, round(validated.shape[1] * normalization_scale))
    output_height = max(1, round(validated.shape[0] * normalization_scale))
    if min(output_width, output_height) < MIN_NORMALIZED_PLAN_EDGE_PIXELS:
        raise PlanMetricCalibrationError(
            "Metric normalization would make the corrected plan too small."
        )
    if (
        max(output_width, output_height) > MAX_NORMALIZED_PLAN_EDGE_PIXELS
        or output_width * output_height > MAX_NORMALIZED_PLAN_PIXEL_COUNT
    ):
        raise PlanMetricCalibrationError(
            "Metric normalization would make the corrected plan too large."
        )
    if (output_width, output_height) == (
        validated.shape[1],
        validated.shape[0],
    ):
        return validated.copy(), normalization_scale
    interpolation = cv2.INTER_AREA if normalization_scale < 1.0 else cv2.INTER_CUBIC
    try:
        normalized = cv2.resize(
            validated,
            (output_width, output_height),
            interpolation=interpolation,
        )
    except cv2.error:
        raise PlanMetricCalibrationError(
            "The corrected plan could not be normalized to metric scale."
        ) from None
    return np.ascontiguousarray(normalized), normalization_scale


# ### Measurement review and bounded retries ###
def prepare_plan_metric_detail_images(
    original_image: np.ndarray,
    *,
    measurement_reference_image: np.ndarray | None = None,
    focus_regions: Sequence[tuple[float, float, float, float]] = (),
    cancellation_check: CancellationCheck | None = None,
) -> tuple[PlanMeasurementDetailImage, ...]:
    try:
        return build_plan_measurement_detail_images(
            original_image,
            measurement_reference_image=measurement_reference_image,
            focus_regions=focus_regions,
            cancellation_check=cancellation_check,
        )
    except PlanMeasurementDetailCancelled:
        raise PlanMetricCalibrationCancelled(
            "Plan metric calibration was cancelled."
        ) from None


def prepare_plan_metric_measurements(
    measurements: Sequence[PlanMeasurementEvidence],
    corrected_image: np.ndarray,
    *,
    measurement_reference_image: np.ndarray | None = None,
    cancellation_check: CancellationCheck | None = None,
) -> PlanMeasurementRefinementResult:
    """Refine geometric anchors without changing transcribed values."""

    try:
        return refine_plan_measurements(
            measurements,
            corrected_image,
            measurement_reference_image=measurement_reference_image,
            cancellation_check=cancellation_check,
        )
    except PlanMeasurementRefinementCancelled:
        raise PlanMetricCalibrationCancelled(
            "Plan metric calibration was cancelled."
        ) from None


def prepare_plan_metric_retry(
    previous: PlanMeasurementRefinementResult,
    measurements: Sequence[PlanMeasurementEvidence],
    corrected_image: np.ndarray,
    *,
    measurement_reference_image: np.ndarray | None = None,
    cancellation_check: CancellationCheck | None = None,
) -> tuple[
    PlanMeasurementRefinementResult,
    PlanMeasurementRefinementResult,
    int,
]:
    """Keep the complete retry independent from a possibly failed first read.

    Solve both the fresh response and the deduplicated union. Otherwise stale
    endpoint mistakes can outvote a successful retry, or a corrected value can
    disappear solely because it contradicted the rejected first response.
    """

    fresh = prepare_plan_metric_measurements(
        measurements,
        corrected_image,
        measurement_reference_image=measurement_reference_image,
        cancellation_check=cancellation_check,
    )
    fresh = preserve_metric_refinement_context(previous, fresh)
    merged, conflicts = merge_plan_metric_measurements(
        previous.measurements, fresh.measurements
    )
    combined = replace(
        fresh,
        measurements=merged,
        refined_count=min(len(merged), fresh.refined_count),
    )
    return fresh, combined, conflicts


def preserve_metric_refinement_context(
    previous: PlanMeasurementRefinementResult,
    current: PlanMeasurementRefinementResult,
) -> PlanMeasurementRefinementResult:
    """A retry with fewer supported anchors cannot clear an observed deformation."""

    return replace(
        current,
        geometry_warning=current.geometry_warning or previous.geometry_warning,
        refined_count=min(
            len(current.measurements),
            max(previous.refined_count, current.refined_count),
        ),
    )


def attach_plan_metric_diagnostics(
    solution: PlanMetricScaleSolution,
    prepared: PlanMeasurementRefinementResult,
    image_size: tuple[int, int],
    *,
    retry_count: int = 0,
) -> PlanMetricScaleSolution:
    """Explain confidence using the same counts and thresholds as the solver."""

    candidates = _build_measurement_candidates(
        solution.measurements,
        *image_size,
    )
    horizontal = sum(
        item.axis == "horizontal" and item.evidence.confidence != "low"
        for item in candidates
    )
    vertical = sum(
        item.axis == "vertical" and item.evidence.confidence != "low"
        for item in candidates
    )
    reasons: list[str] = []
    if horizontal < 2:
        reasons.append("Fewer than two reliable horizontal measurements.")
    if vertical < 2:
        reasons.append("Fewer than two reliable vertical measurements.")
    if solution.relative_residual > 0.03:
        reasons.append("Measurement disagreement exceeds 3%.")
    if prepared.geometry_warning:
        reasons.append(prepared.geometry_warning)
    count = len(prepared.measurements) + prepared.rejected_count
    if solution.confidence != "high" and len(solution.measurements) * 2 <= count:
        reasons.append(
            "Scale uses a consistent subset; at least half of the readings were rejected."
        )
    diagnostics = PlanMetricDiagnostics(
        candidate_count=count,
        accepted_count=len(solution.measurements),
        rejected_count=count - len(solution.measurements),
        horizontal_count=horizontal,
        vertical_count=vertical,
        relative_disagreement=solution.relative_residual,
        refined_count=prepared.refined_count,
        retry_count=retry_count,
        reasons=tuple(reasons),
        geometry_warning=prepared.geometry_warning,
    )
    return replace(
        solution,
        confidence=(
            "medium"
            if prepared.geometry_warning and solution.confidence == "high"
            else solution.confidence
        ),
        diagnostics=diagnostics,
    )


def solve_reviewed_plan_metrics(
    plan_key: str,
    prepared: PlanMeasurementRefinementResult,
    image_size: tuple[int, int],
    *,
    retry_count: int = 0,
) -> tuple[PlanMetricScaleSolution | None, str]:
    if not prepared.measurements:
        return None, "No explicit usable floor-plan measurements were found."
    try:
        density, residual, confidence, accepted = solve_plan_metric_scale(
            prepared.measurements,
            corrected_image_size=image_size,
        )
    except PlanMetricCalibrationError as error:
        return None, str(error)
    solution = PlanMetricScaleSolution(
        plan_key,
        density,
        residual,
        confidence,
        accepted,
    )
    return attach_plan_metric_diagnostics(
        solution,
        prepared,
        image_size,
        retry_count=retry_count,
    ), ""


def build_plan_metric_retry_prompt(
    plan_id: str,
    measurements: Sequence[PlanMeasurementEvidence],
    solution: PlanMetricScaleSolution | None,
    image_size: tuple[int, int],
    *,
    geometry_warning: str = "",
) -> str:
    """Ask for missing evidence, without suggesting a desired numeric scale."""

    candidates = _build_measurement_candidates(
        () if solution is None else solution.measurements,
        *image_size,
    )
    horizontal = sum(item.axis == "horizontal" for item in candidates)
    vertical = sum(item.axis == "vertical" for item in candidates)
    requests = [
        f"Targeted verification for PLAN ID {plan_id}.",
        (
            "For this retry, replace the initial 8-12 measurement goal with only "
            "the 4-6 clearest independently verifiable long dimensions, or fewer "
            "if necessary. Prioritize accurate endpoints over measurement count; "
            "do not repeat weak readings just because they were listed before."
        ),
        (
            f"The first analysis accepted {horizontal} horizontal and {vertical} "
            "vertical spans. Read additional independent long spans in missing "
            "directions and different rooms; aim for at least two in each direction."
        ),
        (
            "Inspect the raw-photo close-ups and distinguish dimension ticks from "
            "wall edges. Recheck units and inside/outside wall-face endpoints. "
            "Return a complete corrected measurement set, including reliable "
            "previously read spans. Do not adjust values to make them agree."
        ),
    ]
    if solution is None or len(solution.measurements) < len(measurements):
        requests.append(
            "Some spans were inconsistent or unusable. Verify their printed "
            "values and corresponding architectural anchors independently."
        )
    if geometry_warning:
        requests.append(
            "The final cleanup may have changed relative span lengths. Check "
            "the annotated and final images carefully; use actual final-image "
            "anchors and omit an uncertain correspondence."
        )
    spans = [
        {
            "label": item.label,
            "annotated_start": item.start_normalized,
            "annotated_end": item.end_normalized,
        }
        for item in measurements[:12]
    ]
    if spans:
        requests.append(
            "Locations to recheck (full annotated-image coordinates): "
            + json.dumps(spans, ensure_ascii=True, separators=(",", ":"))
        )
    return "\n".join(requests)


def plan_metric_retry_regions(
    measurements: Sequence[PlanMeasurementEvidence],
    solution: PlanMetricScaleSolution | None,
) -> tuple[tuple[float, float, float, float], ...]:
    accepted = set(() if solution is None else solution.measurements)
    questionable = [item for item in measurements if item not in accepted]
    regions = []
    for item in questionable[:4]:
        points = (item.start_normalized, item.end_normalized)
        regions.append(
            (
                max(0.0, min(point[0] for point in points) - 0.08),
                max(0.0, min(point[1] for point in points) - 0.08),
                min(1.0, max(point[0] for point in points) + 0.08),
                min(1.0, max(point[1] for point in points) + 0.08),
            )
        )
    return tuple(regions)


def merge_plan_metric_measurements(
    previous: Sequence[PlanMeasurementEvidence],
    additional: Sequence[PlanMeasurementEvidence],
) -> tuple[tuple[PlanMeasurementEvidence, ...], int]:
    """Re-reading a span cannot count as another independent observation."""

    merged = list(previous)
    conflicts: list[PlanMeasurementEvidence] = []
    for item in additional:
        if any(_measurement_spans_match(item, other) for other in conflicts):
            continue
        matches = [
            index
            for index, other in enumerate(merged)
            if _measurement_spans_match(item, other)
        ]
        if not matches:
            merged.append(item)
            continue
        prior = merged[matches[0]]
        if not _measurement_evidence_is_duplicate(prior, item):
            conflicts.append(item)
            merged = [
                other for other in merged if not _measurement_spans_match(other, item)
            ]
        elif (
            _CONFIDENCE_WEIGHTS[item.confidence]
            >= _CONFIDENCE_WEIGHTS[prior.confidence]
        ):
            merged[matches[0]] = item
    return tuple(merged[:MAX_PLAN_MEASUREMENT_COUNT]), len(conflicts)


def add_metric_review_note(
    solution: PlanMetricScaleSolution,
    note: str,
    *,
    limit_confidence: bool = False,
) -> PlanMetricScaleSolution:
    diagnostics = solution.diagnostics
    if diagnostics is not None:
        diagnostics = replace(
            diagnostics,
            retry_count=1,
            reasons=tuple(dict.fromkeys((*diagnostics.reasons, note))),
        )
    return replace(
        solution,
        diagnostics=diagnostics,
        confidence=(
            "medium"
            if limit_confidence and solution.confidence == "high"
            else solution.confidence
        ),
    )


def select_plan_metric_review(
    previous: PlanMetricScaleSolution | None,
    candidate: PlanMetricScaleSolution | None,
    conflicts: int = 0,
    *,
    geometry_warning: str = "",
) -> PlanMetricScaleSolution | None:
    """Keep usable evidence when an optional retry fails or becomes ambiguous."""

    selected = previous
    if candidate is not None:
        if previous is None:
            selected = candidate
        elif (
            _relative_scale_difference(
                previous.estimated_meters_per_pixel,
                candidate.estimated_meters_per_pixel,
            )
            <= CONSENSUS_RELATIVE_TOLERANCE
        ):

            def quality(item: PlanMetricScaleSolution) -> tuple[float, ...]:
                diagnostics = item.diagnostics
                coverage = (
                    min(2, diagnostics.horizontal_count)
                    + min(2, diagnostics.vertical_count)
                    if diagnostics
                    else 0
                )
                return (
                    _CONFIDENCE_WEIGHTS[item.confidence],
                    coverage,
                    len(item.measurements),
                    -item.relative_residual,
                )

            materially_less_consistent = candidate.relative_residual > max(
                0.03,
                previous.relative_residual + 0.01,
            )
            if not materially_less_consistent and quality(candidate) >= quality(
                previous
            ):
                selected = candidate
        else:
            selected = add_metric_review_note(
                previous,
                "Additional analysis disagreed with the established scale.",
                limit_confidence=True,
            )
    if selected is not None and selected is previous:
        selected = add_metric_review_note(
            selected,
            "Additional analysis did not improve the retained evidence.",
        )
    if selected is not None and conflicts:
        selected = add_metric_review_note(
            selected,
            f"Additional analysis gave conflicting values for {conflicts} spans.",
            limit_confidence=True,
        )
    # A rejected scale estimate must not hide newly detected image deformation.
    warning = geometry_warning or (
        candidate.diagnostics.geometry_warning
        if candidate is not None and candidate.diagnostics is not None
        else ""
    )
    if selected is not None and warning:
        if selected.diagnostics is not None:
            selected = replace(
                selected,
                diagnostics=replace(selected.diagnostics, geometry_warning=warning),
            )
        selected = add_metric_review_note(selected, warning, limit_confidence=True)
    return selected


# ### Robust scale helpers ###
def _build_measurement_candidates(
    measurements: Sequence[PlanMeasurementEvidence],
    width: int,
    height: int,
) -> tuple[_MeasurementCandidate, ...]:
    if isinstance(measurements, (str, bytes, bytearray)):
        raise TypeError("Plan measurements must contain measurement evidence.")
    candidates: list[_MeasurementCandidate] = []
    for measurement in measurements:
        if not isinstance(measurement, PlanMeasurementEvidence):
            raise TypeError("Plan measurements contain invalid evidence.")
        start_x = measurement.corrected_start_normalized[0] * width
        start_y = measurement.corrected_start_normalized[1] * height
        end_x = measurement.corrected_end_normalized[0] * width
        end_y = measurement.corrected_end_normalized[1] * height
        pixel_distance = math.hypot(end_x - start_x, end_y - start_y)
        if pixel_distance < MIN_CORRECTED_MEASUREMENT_SPAN_PIXELS:
            continue
        meters_per_pixel = measurement.value_meters / pixel_distance
        if not math.isfinite(meters_per_pixel) or meters_per_pixel <= 0.0:
            continue
        if any(
            _measurement_evidence_is_duplicate(
                measurement,
                candidate.evidence,
            )
            for candidate in candidates
        ):
            continue
        candidates.append(
            _MeasurementCandidate(
                evidence=measurement,
                meters_per_pixel=meters_per_pixel,
                weight=(
                    _CONFIDENCE_WEIGHTS[measurement.confidence]
                    * min(4.0, math.sqrt(pixel_distance / 24.0))
                ),
                axis=_dominant_axis(end_x - start_x, end_y - start_y),
            )
        )
    return tuple(candidates)


# ### Cross-plan reconciliation helpers ###
def _cross_plan_corroboration(
    candidates_by_plan: Mapping[str, Sequence[_MeasurementCandidate]],
) -> dict[tuple[str, PlanMeasurementEvidence], int]:
    """Count distinct other plans that corroborate a semantic measurement."""

    grouped = _cross_plan_candidate_groups(candidates_by_plan)
    corroboration: dict[tuple[str, PlanMeasurementEvidence], int] = {}
    for records in grouped.values():
        for plan_key, candidate in records:
            supporting_plans = {
                other_plan_key
                for other_plan_key, other_candidate in records
                if other_plan_key != plan_key
                and _relative_scale_difference(
                    candidate.evidence.value_meters,
                    other_candidate.evidence.value_meters,
                )
                <= MAX_CROSS_PLAN_VALUE_DIFFERENCE
            }
            if supporting_plans:
                corroboration[(plan_key, candidate.evidence)] = len(supporting_plans)
    return corroboration


def _cross_plan_unit_mismatches(
    candidates_by_plan: Mapping[str, Sequence[_MeasurementCandidate]],
) -> dict[str, frozenset[PlanMeasurementEvidence]]:
    """Reject only strongly corroborated power-of-ten transcription mistakes."""

    rejected: dict[str, set[PlanMeasurementEvidence]] = {}
    for records in _cross_plan_candidate_groups(candidates_by_plan).values():
        clusters = tuple(
            (
                {
                    plan_key
                    for plan_key, candidate in records
                    if _relative_scale_difference(
                        seed.evidence.value_meters,
                        candidate.evidence.value_meters,
                    )
                    <= MAX_CROSS_PLAN_VALUE_DIFFERENCE
                },
                tuple(
                    (plan_key, candidate)
                    for plan_key, candidate in records
                    if _relative_scale_difference(
                        seed.evidence.value_meters,
                        candidate.evidence.value_meters,
                    )
                    <= MAX_CROSS_PLAN_VALUE_DIFFERENCE
                ),
            )
            for _seed_plan_key, seed in records
        )
        if not clusters:
            continue
        largest_support = max(len(plan_keys) for plan_keys, _items in clusters)
        if largest_support < MIN_CROSS_PLAN_UNIT_CONSENSUS:
            continue
        dominant_clusters = tuple(
            items for plan_keys, items in clusters if len(plan_keys) == largest_support
        )
        dominant_values = {
            tuple(sorted(item.evidence.value_meters for _key, item in cluster))
            for cluster in dominant_clusters
        }
        if len(dominant_values) != 1:
            continue
        dominant = dominant_clusters[0]
        dominant_evidence = {
            (plan_key, candidate.evidence) for plan_key, candidate in dominant
        }
        consensus_value = _weighted_geometric_mean(
            tuple(candidate.evidence.value_meters for _key, candidate in dominant),
            tuple(candidate.weight for _key, candidate in dominant),
        )
        for plan_key, candidate in records:
            if (plan_key, candidate.evidence) in dominant_evidence:
                continue
            if not _is_power_of_ten_unit_mismatch(
                candidate.evidence.value_meters,
                consensus_value,
            ):
                continue
            rejected.setdefault(plan_key, set()).add(candidate.evidence)
    return {
        plan_key: frozenset(measurements) for plan_key, measurements in rejected.items()
    }


def _cross_plan_candidate_groups(
    candidates_by_plan: Mapping[str, Sequence[_MeasurementCandidate]],
) -> dict[str, tuple[tuple[str, _MeasurementCandidate], ...]]:
    grouped: dict[str, list[tuple[str, _MeasurementCandidate]]] = {}
    for plan_key, candidates in candidates_by_plan.items():
        for candidate in candidates:
            signature = _cross_plan_measurement_signature(candidate.evidence.label)
            if signature is None:
                continue
            grouped.setdefault(signature, []).append((plan_key, candidate))
    return {
        signature: tuple(records)
        for signature, records in grouped.items()
        if len({plan_key for plan_key, _candidate in records}) > 1
    }


def _cross_plan_measurement_signature(label: str) -> str | None:
    normalized = "".join(
        character.casefold() if character.isalnum() else " " for character in label
    )
    words = tuple(
        word
        for word in normalized.split()
        if word not in _CROSS_PLAN_LABEL_IGNORED_WORDS
    )
    if len(words) < MIN_CROSS_PLAN_LABEL_WORDS:
        return None
    return " ".join(sorted(set(words)))


def _candidate_with_cross_plan_support(
    candidate: _MeasurementCandidate,
    supporting_plan_count: int,
) -> _MeasurementCandidate:
    if supporting_plan_count <= 0:
        return candidate
    multiplier = min(
        MAX_CROSS_PLAN_SUPPORT_MULTIPLIER,
        1.0 + supporting_plan_count * CROSS_PLAN_SUPPORT_WEIGHT_STEP,
    )
    return _MeasurementCandidate(
        evidence=candidate.evidence,
        meters_per_pixel=candidate.meters_per_pixel,
        weight=candidate.weight * multiplier,
        axis=candidate.axis,
    )


def _is_power_of_ten_unit_mismatch(value: float, consensus: float) -> bool:
    logarithmic_ratio = abs(math.log10(value / consensus))
    nearest_power = round(logarithmic_ratio)
    return (
        1 <= nearest_power <= 3
        and abs(logarithmic_ratio - nearest_power) <= CROSS_PLAN_UNIT_RATIO_TOLERANCE
    )


# ### Local candidate helpers ###
def _measurement_evidence_is_duplicate(
    first: PlanMeasurementEvidence,
    second: PlanMeasurementEvidence,
) -> bool:
    """Return whether two records describe the same corrected-image span."""

    value_difference = abs(first.value_meters - second.value_meters) / max(
        first.value_meters,
        second.value_meters,
    )
    if value_difference > MAX_DUPLICATE_VALUE_DIFFERENCE:
        return False
    return _measurement_spans_match(first, second)


def _measurement_spans_match(
    first: PlanMeasurementEvidence,
    second: PlanMeasurementEvidence,
) -> bool:
    """Compare geometry independently of possibly conflicting printed values."""

    first_start = first.corrected_start_normalized
    first_end = first.corrected_end_normalized
    second_start = second.corrected_start_normalized
    second_end = second.corrected_end_normalized
    same_direction = (
        math.dist(first_start, second_start) <= MAX_DUPLICATE_ENDPOINT_DISTANCE
        and math.dist(first_end, second_end) <= MAX_DUPLICATE_ENDPOINT_DISTANCE
    )
    reversed_direction = (
        math.dist(first_start, second_end) <= MAX_DUPLICATE_ENDPOINT_DISTANCE
        and math.dist(first_end, second_start) <= MAX_DUPLICATE_ENDPOINT_DISTANCE
    )
    return same_direction or reversed_direction


def _select_consensus_candidates(
    candidates: Sequence[_MeasurementCandidate],
    *,
    allow_weighted_tie_break: bool = False,
) -> tuple[_MeasurementCandidate, ...]:
    if len(candidates) == 1:
        return (candidates[0],)
    groups = _measurement_consensus_groups(candidates)
    clusters: list[tuple[float, int, float, tuple[_MeasurementCandidate, ...]]] = []
    for seed, inliers in zip(candidates, groups, strict=True):
        cluster_weight = sum(candidate.weight for candidate in inliers)
        cluster_spread = sum(
            _relative_scale_difference(
                candidate.meters_per_pixel,
                seed.meters_per_pixel,
            )
            * candidate.weight
            for candidate in inliers
        ) / max(cluster_weight, _GEOMETRY_EPSILON)
        clusters.append(
            (
                cluster_weight,
                len(inliers),
                -cluster_spread,
                inliers,
            )
        )
    best_weight, best_count, _spread, best = max(
        clusters,
        key=lambda item: (item[1], item[0], item[2]),
    )
    competing = tuple(
        item
        for item in clusters
        if item[3] != best
        and item[1] == best_count
        and math.isclose(item[0], best_weight, rel_tol=0.05, abs_tol=1e-9)
    )
    rejected = bool(competing) and any(
        not _candidate_groups_overlap(best, item[3]) for item in competing
    )
    if len(candidates) > 1 and best_count < 2:
        rejected = True
    if len(candidates) >= 3 and best_count * 2 <= len(candidates):
        equally_large_disjoint = tuple(
            item
            for item in clusters
            if item[1] == best_count and not _candidate_groups_overlap(best, item[3])
        )
        strongest_competing_weight = max(
            (item[0] for item in equally_large_disjoint),
            default=0.0,
        )
        if (
            not allow_weighted_tie_break
            or strongest_competing_weight <= 0.0
            or best_weight <= strongest_competing_weight * 1.10
        ):
            rejected = True
    if rejected:
        best = _select_strong_minority_consensus(groups)
        if not best:
            return ()
    return tuple(
        sorted(
            best,
            key=lambda candidate: (
                -candidate.weight,
                candidate.evidence.label,
                candidate.meters_per_pixel,
            ),
        )
    )


def _measurement_consensus_groups(
    candidates: Sequence[_MeasurementCandidate],
) -> tuple[tuple[_MeasurementCandidate, ...], ...]:
    return tuple(
        tuple(
            candidate
            for candidate in candidates
            if _relative_scale_difference(
                candidate.meters_per_pixel, seed.meters_per_pixel
            )
            <= CONSENSUS_RELATIVE_TOLERANCE
        )
        for seed in candidates
    )


def _select_strong_minority_consensus(
    groups: Sequence[tuple[_MeasurementCandidate, ...]],
) -> tuple[_MeasurementCandidate, ...]:
    """Distinguish dispersed bad readings from a competing coherent scale.

    Expanding the transcription must not invalidate four excellent independent
    measurements merely because it also found unrelated uncertain dimensions.
    Require the existing high-confidence precision and two spans per axis, plus
    at least twice the independent reliable support outside this hypothesis.
    A shared borderline reading must not hide a separate coherent competitor.
    This is not permission to pick between two similarly plausible scales.
    """

    verified: list[tuple[int, int, float, tuple[_MeasurementCandidate, ...]]] = []
    for group in groups:
        reliable = _independent_reliable_consensus_support(group)
        if len(reliable) < 4:
            continue
        if any(
            sum(item.axis == axis for item in reliable) < 2
            for axis in ("horizontal", "vertical")
        ):
            continue
        estimate = _weighted_geometric_mean(
            tuple(item.meters_per_pixel for item in group),
            tuple(item.weight for item in group),
        )
        residual = _relative_root_mean_square_error(group, estimate)
        if residual > 0.03:
            continue
        competing_support = max(
            (
                len(
                    _independent_reliable_consensus_support(
                        other, excluding=group
                    )
                )
                for other in groups
            ),
            default=0,
        )
        if len(reliable) < 2 * competing_support:
            continue
        verified.append((len(reliable), len(group), -residual, group))
    if not verified:
        return ()
    return max(verified, key=lambda item: item[:3])[3]


def _independent_reliable_consensus_support(
    candidates: Sequence[_MeasurementCandidate],
    *,
    excluding: Sequence[_MeasurementCandidate] = (),
) -> tuple[_MeasurementCandidate, ...]:
    independent: list[_MeasurementCandidate] = []
    excluded_ids = {id(candidate) for candidate in excluding}
    for candidate in candidates:
        if candidate.evidence.confidence == "low" or id(candidate) in excluded_ids:
            continue
        if any(
            _measurement_spans_match(candidate.evidence, previous.evidence)
            for previous in independent
        ):
            continue
        independent.append(candidate)
    return tuple(independent)


def _consensus_failure_details(candidates: Sequence[_MeasurementCandidate]) -> str:
    groups = _measurement_consensus_groups(candidates)
    largest = max(
        groups,
        key=lambda group: (
            len(group),
            sum(item.evidence.confidence != "low" for item in group),
        ),
        default=(),
    )
    reliable_count = sum(item.evidence.confidence != "low" for item in largest)
    return (
        f"Largest agreeing group: {len(largest)} of {len(candidates)} spans "
        f"({reliable_count} reliable); no clearly supported scale was found."
    )


def _candidate_groups_overlap(
    first: Sequence[_MeasurementCandidate],
    second: Sequence[_MeasurementCandidate],
) -> bool:
    first_ids = {id(candidate) for candidate in first}
    return bool(first_ids & {id(candidate) for candidate in second})


def _dominant_axis(delta_x: float, delta_y: float) -> str | None:
    absolute_x = abs(delta_x)
    absolute_y = abs(delta_y)
    if absolute_x >= absolute_y * 2.0:
        return "horizontal"
    if absolute_y >= absolute_x * 2.0:
        return "vertical"
    return None


def _validate_isotropic_scale(
    accepted: Sequence[_MeasurementCandidate],
) -> None:
    horizontal = tuple(
        candidate for candidate in accepted if candidate.axis == "horizontal"
    )
    vertical = tuple(
        candidate for candidate in accepted if candidate.axis == "vertical"
    )
    if len(horizontal) < 2 or len(vertical) < 2:
        return
    horizontal_scale = _weighted_geometric_mean(
        tuple(candidate.meters_per_pixel for candidate in horizontal),
        tuple(candidate.weight for candidate in horizontal),
    )
    vertical_scale = _weighted_geometric_mean(
        tuple(candidate.meters_per_pixel for candidate in vertical),
        tuple(candidate.weight for candidate in vertical),
    )
    if (
        _relative_scale_difference(horizontal_scale, vertical_scale)
        > MAX_AXIS_SCALE_DIFFERENCE
    ):
        raise PlanMetricCalibrationInferenceError(
            "The corrected plan has inconsistent horizontal and vertical scale."
        )


def _validate_physical_plan_scale(
    accepted: Sequence[_MeasurementCandidate],
    meters_per_pixel: float,
    *,
    width: int,
    height: int,
) -> None:
    """Reject coherent OCR unit mistakes before they destructively resize a plan."""

    reliable_spans = tuple(
        candidate.evidence.value_meters
        for candidate in accepted
        if candidate.evidence.confidence in {"medium", "high"}
    )
    if not reliable_spans or max(reliable_spans) < MIN_RELIABLE_REFERENCE_SPAN_METERS:
        raise PlanMetricCalibrationInferenceError(
            "The transcribed dimensions contain no reliable room-scale span; "
            "their centimetre or millimetre interpretation may be wrong."
        )

    physical_width = width * meters_per_pixel
    physical_height = height * meters_per_pixel
    long_edge = max(physical_width, physical_height)
    area = physical_width * physical_height
    if (
        long_edge < MIN_PLAUSIBLE_PLAN_LONG_EDGE_METERS
        or area < MIN_PLAUSIBLE_PLAN_AREA_SQUARE_METERS
    ):
        raise PlanMetricCalibrationInferenceError(
            "The transcribed dimensions imply an implausibly small floor plan; "
            "their centimetre or millimetre interpretation may be wrong."
        )
    if (
        long_edge > MAX_PLAUSIBLE_PLAN_LONG_EDGE_METERS
        or area > MAX_PLAUSIBLE_PLAN_AREA_SQUARE_METERS
    ):
        raise PlanMetricCalibrationInferenceError(
            "The transcribed dimensions imply an implausibly large floor plan; "
            "their units may be wrong."
        )


def _weighted_geometric_mean(
    values: Sequence[float],
    weights: Sequence[float],
) -> float:
    total_weight = sum(weights)
    if total_weight <= 0.0 or len(values) != len(weights) or not values:
        raise PlanMetricCalibrationInferenceError(
            "The transcribed plan measurements do not have usable confidence."
        )
    return math.exp(
        sum(
            weight * math.log(value)
            for value, weight in zip(values, weights, strict=True)
        )
        / total_weight
    )


def _relative_root_mean_square_error(
    candidates: Sequence[_MeasurementCandidate],
    estimate: float,
) -> float:
    total_weight = sum(candidate.weight for candidate in candidates)
    squared_error = sum(
        candidate.weight * ((estimate / candidate.meters_per_pixel) - 1.0) ** 2
        for candidate in candidates
    )
    return math.sqrt(squared_error / max(total_weight, _GEOMETRY_EPSILON))


def _relative_scale_difference(first: float, second: float) -> float:
    return abs(math.log(first / second))


def _calibration_confidence(
    accepted: Sequence[_MeasurementCandidate],
    residual: float,
) -> str:
    reliable_count = sum(
        candidate.evidence.confidence in {"medium", "high"} for candidate in accepted
    )
    horizontal_count = sum(
        candidate.axis == "horizontal" and candidate.evidence.confidence != "low"
        for candidate in accepted
    )
    vertical_count = sum(
        candidate.axis == "vertical" and candidate.evidence.confidence != "low"
        for candidate in accepted
    )
    if horizontal_count >= 2 and vertical_count >= 2 and residual <= 0.03:
        return "high"
    if len(accepted) >= 2 and reliable_count >= 1 and residual <= 0.08:
        return "medium"
    return "low"


# ### Validation and cancellation helpers ###
def _validated_image(image: object, label: str) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise PlanMetricCalibrationError(
            f"The {label} image must be an 8-bit NumPy array."
        )
    if image.ndim == 2 or image.ndim == 3 and image.shape[2] in {1, 3, 4}:
        pass
    else:
        raise PlanMetricCalibrationError(
            f"The {label} image has an unsupported format."
        )
    if min(image.shape[:2]) < 2:
        raise PlanMetricCalibrationError(f"The {label} image is empty.")
    return image


def _validated_image_size(
    value: object,
    label: str,
) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(f"The {label} size must contain width and height.")
    if any(isinstance(dimension, bool) for dimension in value):
        raise TypeError(f"The {label} size must contain integers.")
    try:
        width, height = (int(dimension) for dimension in value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"The {label} size must contain integers.") from None
    if width < 1 or height < 1:
        raise ValueError(f"The {label} size must be positive.")
    return width, height


def _positive_finite_number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a number.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{label} must be a number.") from None
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be finite and greater than zero.")
    return number


def _validated_image_data_uri(value: object, label: str) -> str:
    uri = str(value).strip()
    if not uri.startswith("data:image/") or ";base64," not in uri:
        raise PlanMetricCalibrationError(
            f"The {label} could not be prepared for measurement transcription."
        )
    return uri


def _raise_if_cancelled(cancellation_check: CancellationCheck | None) -> None:
    if cancellation_check is not None and bool(cancellation_check()):
        raise PlanMetricCalibrationCancelled("Plan metric calibration was cancelled.")
