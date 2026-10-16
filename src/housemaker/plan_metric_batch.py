# ### Imports ###
from __future__ import annotations

import copy
import json
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from housemaker.plan_metric_calibration import (
    PLAN_MEASUREMENT_RESULT_SCHEMA,
    CancellationCheck,
    NormalizedPlanMetricResult,
    PlanMetricCalibrationCancelled,
    PlanMetricCalibrationError,
    PlanMetricCalibrationInferenceError,
    PlanMetricEvidenceSet,
    PlanMetricScaleSolution,
    add_metric_review_note,
    attach_plan_metric_diagnostics,
    build_plan_metric_retry_prompt,
    encode_plan_measurement_image,
    normalize_plan_with_scale_solution,
    parse_plan_measurement_result,
    plan_metric_retry_regions,
    prepare_plan_metric_detail_images,
    prepare_plan_metric_measurements,
    prepare_plan_metric_retry,
    select_plan_metric_review,
    solve_plan_metric_scale_batch,
)
from housemaker.plan_metric_refinement import PlanMeasurementRefinementResult
from housemaker.plan_metric_state import PlanMeasurementEvidence
from housemaker.plan_metric_views import PlanMeasurementDetailImage

# ### Batch transcription constants ###
MULTI_PLAN_MEASUREMENT_PROMPT = """Analyze every labeled architectural floor-plan
set supplied in this request. Each set always has an ORIGINAL PLAN (RAW) and a
CORRECTED PLAN (FINAL), and may also have a RECTIFIED ANNOTATED PLAN. All sets
belong to the same property, but every image has its own independent pixel
coordinate system and may have been photographed or corrected at a different
scale.

First infer the shared printed dimension notation and unit convention from all
raw ORIGINAL plans together. Use repeated notation and compatible dimensions
across levels to disambiguate centimetres, millimetres, and metres. Then
transcribe the strongest explicit dimensions separately for each exact PLAN ID.
The raw original is authoritative for printed numbers and units. A rectified
annotated plan is only a reading and correspondence aid. When it is present,
omit any measurement whose printed value disagrees between raw and rectified
images or cannot be confirmed in the raw original.

For each plan, include only explicit dimensions that describe distances in the
horizontal floor plane. Include valid dimensions whose text or dimension line is
vertical on the page. Ignore room names, reference numbers, quantities, angles,
elevations, ceiling heights, and vertical section dimensions. Convert every
retained distance to metres. Infer an omitted unit only when the shared drawing
convention and surrounding measurements make the interpretation clear.

For each measurement, identify its annotated endpoints and the corresponding
architectural anchors in that plan's final corrected image. When a RECTIFIED
ANNOTATED PLAN is present, start_normalized and end_normalized are relative to
that image; otherwise they are relative to the raw ORIGINAL PLAN.
corrected_start_normalized and corrected_end_normalized are always relative to
the CORRECTED PLAN (FINAL). Coordinates are normalized from 0 to 1 relative to
their image, with [0, 0] at the top-left and [1, 1] at the bottom-right. Never
copy coordinates between PLAN IDs, average pixel distances across plans, or
apply one plan's pixel scale to another plan.

For label, write a concise semantic description of the measured span. Do not
invent obscured values, endpoints, or cross-level correspondences. Other plans may
help interpret the shared unit convention, but a measurement belongs to a PLAN ID
only when its value and endpoints are supported by that plan's images. Prefer
8–12 independent long measurements per plan when available, with at least two
horizontal and two vertical spans distributed across rooms and regions. Do not
invent evidence to meet a quota or count repeated readings as independent spans.
Distinguish inside wall faces, outside wall faces, and centerlines according to
each dimension's meaning. Detail crops are magnifications of labeled source
rectangles; return full annotated/reference and final image coordinates, never
crop-local coordinates. Assign high
confidence only when the printed value, unit interpretation, endpoints, and
corrected-image correspondence are all unambiguous."""

MULTI_PLAN_MEASUREMENT_REQUEST_NAME = "architectural_multi_plan_measurements"
MAX_MULTI_PLAN_MEASUREMENT_COUNT = 32
MAX_BATCH_MEASUREMENTS_PER_PLAN = 12
MAX_PLAN_MEASUREMENT_ID_CHARACTERS = 64
_PLAN_MEASUREMENT_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}\Z")


# ### Provider-neutral models ###
@dataclass(frozen=True)
class EncodedPlanMeasurementPair:
    """One raw/reference/final plan set prepared for a vision provider."""

    plan_id: str
    original_image_data_uri: str
    corrected_image_data_uri: str
    measurement_reference_image_data_uri: str | None = None
    detail_images: tuple[PlanMeasurementDetailImage, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "plan_id",
            validate_plan_measurement_id(self.plan_id),
        )
        _validate_image_data_uri(self.original_image_data_uri, "original plan")
        _validate_image_data_uri(self.corrected_image_data_uri, "corrected plan")
        if self.measurement_reference_image_data_uri is not None:
            _validate_image_data_uri(
                self.measurement_reference_image_data_uri,
                "rectified annotated plan",
            )
        details = tuple(self.detail_images)
        if len(details) > 4 or not all(
            isinstance(item, PlanMeasurementDetailImage) for item in details
        ):
            raise ValueError("Plan measurement detail images are invalid.")
        object.__setattr__(self, "detail_images", details)


@dataclass(frozen=True)
class PlanMeasurementRasterPair:
    """Raw source, corrected plan, and optional transcription reference."""

    plan_id: str
    original_image: np.ndarray
    corrected_image: np.ndarray
    measurement_reference_image: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "plan_id",
            validate_plan_measurement_id(self.plan_id),
        )


@dataclass(frozen=True)
class PlanMeasurementGroup:
    """Validated measurement evidence belonging to exactly one plan."""

    plan_id: str
    measurements: tuple[PlanMeasurementEvidence, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "plan_id",
            validate_plan_measurement_id(self.plan_id),
        )
        measurements = tuple(self.measurements)
        if not all(isinstance(item, PlanMeasurementEvidence) for item in measurements):
            raise TypeError("Plan measurement evidence is invalid.")
        object.__setattr__(self, "measurements", measurements)


class MultiPlanMeasurementAnalyzer(Protocol):
    """Analyze several labeled plan pairs without coupling their pixel scales."""

    def __call__(
        self,
        plan_pairs: Sequence[EncodedPlanMeasurementPair],
        *,
        prompt: str,
        result_schema: Mapping[str, Any],
        cancellation_check: CancellationCheck | None,
    ) -> object: ...


@dataclass(frozen=True)
class _StoredCalibrationError:
    """A safely repeatable error shared with more than one worker thread."""

    error_type: type[PlanMetricCalibrationError]
    message: str

    def raise_new(self) -> None:
        raise self.error_type(self.message)


# ### Coordinated native-resolution calibration ###
class PlanMetricBatchCoordinator:
    """Barrier and one-shot batch solver for independently corrected plans.

    Each correction worker submits its own original/corrected raster pair. The
    final arrival performs one grouped transcription and deterministic batch
    solve, then publishes a separate calibrated result or error for every plan.
    Pixel scales remain local to their raster; the images are never resampled.
    """

    _WAIT_POLL_SECONDS = 0.05

    def __init__(
        self,
        plan_ids: Sequence[str],
        analyzer: MultiPlanMeasurementAnalyzer,
    ) -> None:
        self._plan_ids = validate_plan_measurement_ids(plan_ids)
        if not callable(analyzer):
            raise TypeError("The multi-plan measurement analyzer must be callable.")
        self._analyzer = analyzer
        self._condition = threading.Condition()
        self._submissions: dict[str, PlanMeasurementRasterPair] = {}
        self._cancellation_checks: dict[str, CancellationCheck] = {}
        self._results: dict[str, NormalizedPlanMetricResult] = {}
        self._errors: dict[str, _StoredCalibrationError] = {}
        self._finished = False

    def normalize(
        self,
        plan_id: str,
        original_image: np.ndarray,
        corrected_image: np.ndarray,
        *,
        measurement_reference_image: np.ndarray | None = None,
        cancellation_check: CancellationCheck | None = None,
    ) -> NormalizedPlanMetricResult:
        """Submit one plan and retain its native raster after the batch solves.

        The historical method name remains compatible with correction workers.
        """

        normalized_id = validate_plan_measurement_id(plan_id)
        if normalized_id not in self._plan_ids:
            raise ValueError(
                f"Plan measurement ID '{normalized_id}' is not in this batch."
            )
        if cancellation_check is not None and not callable(cancellation_check):
            raise TypeError("The cancellation check must be callable.")
        if self._check_cancelled(cancellation_check):
            self.cancel()
            raise PlanMetricCalibrationCancelled(
                "Plan metric calibration was cancelled."
            )
        submission = PlanMeasurementRasterPair(
            normalized_id,
            original_image,
            corrected_image,
            measurement_reference_image,
        )

        batch_pairs: tuple[PlanMeasurementRasterPair, ...] | None = None
        with self._condition:
            if self._finished:
                return self._resolved_result(normalized_id)
            if normalized_id in self._submissions:
                raise PlanMetricCalibrationError(
                    f"Plan '{normalized_id}' was submitted more than once."
                )
            self._submissions[normalized_id] = submission
            if cancellation_check is not None:
                self._cancellation_checks[normalized_id] = cancellation_check
            if len(self._submissions) == len(self._plan_ids):
                batch_pairs = tuple(self._submissions[item] for item in self._plan_ids)

        if batch_pairs is not None:
            self._process_batch(batch_pairs)
        return self._wait_for_result(normalized_id, cancellation_check)

    def cancel(self) -> None:
        """Cancel the whole batch and wake every waiting correction worker."""

        with self._condition:
            if self._finished:
                return
            error = _stored_error(
                PlanMetricCalibrationCancelled("Plan metric calibration was cancelled.")
            )
            self._errors = {plan_id: error for plan_id in self._plan_ids}
            self._finished = True
            self._condition.notify_all()

    def _process_batch(
        self,
        pairs: tuple[PlanMeasurementRasterPair, ...],
    ) -> None:
        try:
            results, errors = self._calculate_batch_outcome(pairs)
        except BaseException as error:  # noqa: BLE001 - always release the barrier.
            stored = _stored_error(error)
            results = {}
            errors = {plan_id: stored for plan_id in self._plan_ids}
        with self._condition:
            if self._finished:
                return
            self._results = results
            self._errors = errors
            self._finished = True
            self._condition.notify_all()

    def _calculate_batch_outcome(
        self,
        pairs: tuple[PlanMeasurementRasterPair, ...],
    ) -> tuple[
        dict[str, NormalizedPlanMetricResult],
        dict[str, _StoredCalibrationError],
    ]:
        groups = transcribe_multi_plan_measurements(
            pairs,
            self._analyzer,
            cancellation_check=self._batch_cancellation_requested,
        )
        self._raise_if_batch_cancelled()
        prepared = {
            pair.plan_id: prepare_plan_metric_measurements(
                group.measurements,
                pair.corrected_image,
                measurement_reference_image=pair.measurement_reference_image,
                cancellation_check=self._batch_cancellation_requested,
            )
            for pair, group in zip(pairs, groups, strict=True)
        }
        solutions, failures = _solve_prepared_metric_batch(pairs, prepared)
        retry_pairs = tuple(
            pair
            for pair in pairs
            if pair.plan_id not in solutions
            or solutions[pair.plan_id].confidence != "high"
        )
        if retry_pairs:
            requests = [
                build_plan_metric_retry_prompt(
                    pair.plan_id,
                    prepared[pair.plan_id].measurements,
                    solutions.get(pair.plan_id),
                    (pair.corrected_image.shape[1], pair.corrected_image.shape[0]),
                    geometry_warning=prepared[pair.plan_id].geometry_warning,
                )
                for pair in retry_pairs
            ]
            try:
                retry_groups = transcribe_multi_plan_measurements(
                    retry_pairs,
                    self._analyzer,
                    cancellation_check=self._batch_cancellation_requested,
                    prompt=MULTI_PLAN_MEASUREMENT_PROMPT
                    + "\n\n"
                    + "\n\n".join(requests),
                    focus_regions_by_plan={
                        pair.plan_id: plan_metric_retry_regions(
                            prepared[pair.plan_id].measurements,
                            solutions.get(pair.plan_id),
                        )
                        for pair in retry_pairs
                    },
                )
            except PlanMetricCalibrationCancelled:
                raise
            except PlanMetricCalibrationError:
                for pair in retry_pairs:
                    if pair.plan_id in solutions:
                        solutions[pair.plan_id] = add_metric_review_note(
                            solutions[pair.plan_id],
                            "Additional measurement analysis was unavailable.",
                        )
            else:
                fresh = dict(prepared)
                updated = dict(prepared)
                conflicts = {}
                for pair, group in zip(retry_pairs, retry_groups, strict=True):
                    (
                        fresh[pair.plan_id],
                        updated[pair.plan_id],
                        conflicts[pair.plan_id],
                    ) = prepare_plan_metric_retry(
                        prepared[pair.plan_id],
                        group.measurements,
                        pair.corrected_image,
                        measurement_reference_image=pair.measurement_reference_image,
                        cancellation_check=self._batch_cancellation_requested,
                    )
                # Keep other levels as supporting evidence for the shared unit
                # convention, without changing a level already rated high.
                fresh_solutions, fresh_failures = _solve_prepared_metric_batch(
                    pairs,
                    fresh,
                    retry_count=1,
                )
                reviewed, review_failures = _solve_prepared_metric_batch(
                    pairs,
                    updated,
                    retry_count=1,
                )
                for pair in retry_pairs:
                    plan_id = pair.plan_id
                    candidate = select_plan_metric_review(
                        fresh_solutions.get(plan_id),
                        reviewed.get(plan_id),
                        conflicts.get(plan_id, 0),
                        geometry_warning=updated[plan_id].geometry_warning,
                    )
                    chosen = select_plan_metric_review(
                        solutions.get(plan_id),
                        candidate,
                        conflicts.get(plan_id, 0),
                        geometry_warning=updated[plan_id].geometry_warning,
                    )
                    if chosen is not None:
                        solutions[plan_id] = chosen
                        failures.pop(plan_id, None)
                    elif plan_id in fresh_failures:
                        failures[plan_id] = fresh_failures[plan_id]
                    elif plan_id in review_failures:
                        failures[plan_id] = review_failures[plan_id]

        self._raise_if_batch_cancelled()
        errors = {
            plan_id: _stored_error(PlanMetricCalibrationInferenceError(reason))
            for plan_id, reason in failures.items()
        }
        results: dict[str, NormalizedPlanMetricResult] = {}
        pairs_by_id = {pair.plan_id: pair for pair in pairs}
        for plan_id, solution in solutions.items():
            self._raise_if_batch_cancelled()
            pair = pairs_by_id[plan_id]
            try:
                results[plan_id] = normalize_plan_with_scale_solution(
                    pair.original_image,
                    pair.corrected_image,
                    solution,
                    cancellation_check=self._batch_cancellation_requested,
                )
            except PlanMetricCalibrationCancelled:
                raise
            except PlanMetricCalibrationError as error:
                errors[plan_id] = _stored_error(error)

        for plan_id in self._plan_ids:
            if plan_id not in results and plan_id not in errors:
                errors[plan_id] = _stored_error(
                    PlanMetricCalibrationInferenceError(
                        "The plan measurements could not establish a reliable scale."
                    )
                )
        return results, errors

    def _wait_for_result(
        self,
        plan_id: str,
        cancellation_check: CancellationCheck | None,
    ) -> NormalizedPlanMetricResult:
        while True:
            with self._condition:
                if self._finished:
                    return self._resolved_result(plan_id)
                self._condition.wait(self._WAIT_POLL_SECONDS)
            if self._check_cancelled(cancellation_check):
                self.cancel()

    def _resolved_result(self, plan_id: str) -> NormalizedPlanMetricResult:
        error = self._errors.get(plan_id)
        if error is not None:
            error.raise_new()
        result = self._results.get(plan_id)
        if result is None:
            raise PlanMetricCalibrationError(
                "The coordinated plan metric result is unavailable."
            )
        return result

    def _batch_cancellation_requested(self) -> bool:
        with self._condition:
            if self._finished:
                return True
            checks = tuple(self._cancellation_checks.values())
        for check in checks:
            if self._check_cancelled(check):
                self.cancel()
                return True
        return False

    def _raise_if_batch_cancelled(self) -> None:
        if self._batch_cancellation_requested():
            raise PlanMetricCalibrationCancelled(
                "Plan metric calibration was cancelled."
            )

    @staticmethod
    def _check_cancelled(
        cancellation_check: CancellationCheck | None,
    ) -> bool:
        if cancellation_check is None:
            return False
        try:
            return bool(cancellation_check())
        except Exception:  # noqa: BLE001 - a broken check must not deadlock peers.
            return True


def _stored_error(error: BaseException) -> _StoredCalibrationError:
    """Convert one failure into a safe exception reusable by every waiter."""

    if isinstance(error, PlanMetricCalibrationCancelled):
        error_type = PlanMetricCalibrationCancelled
    elif isinstance(error, PlanMetricCalibrationInferenceError):
        error_type = PlanMetricCalibrationInferenceError
    elif isinstance(error, PlanMetricCalibrationError):
        error_type = PlanMetricCalibrationError
    else:
        return _StoredCalibrationError(
            PlanMetricCalibrationError,
            "Coordinated plan metric calibration failed.",
        )
    message = str(error).strip() or "Coordinated plan metric calibration failed."
    return _StoredCalibrationError(error_type, message)


def _solve_prepared_metric_batch(
    pairs: Sequence[PlanMeasurementRasterPair],
    prepared: Mapping[str, PlanMeasurementRefinementResult],
    *,
    retry_count: int = 0,
) -> tuple[dict[str, PlanMetricScaleSolution], dict[str, str]]:
    """Preserve cross-level unit checks while reviewing each raster locally."""

    evidence_sets = []
    failures = {}
    image_sizes = {}
    for pair in pairs:
        image_sizes[pair.plan_id] = (
            int(pair.corrected_image.shape[1]),
            int(pair.corrected_image.shape[0]),
        )
        evidence = prepared[pair.plan_id].measurements
        if not evidence:
            failures[pair.plan_id] = (
                "No explicit usable floor-plan measurements were found."
            )
            continue
        evidence_sets.append(
            PlanMetricEvidenceSet(
                pair.plan_id,
                image_sizes[pair.plan_id],
                evidence,
            )
        )
    solutions = {}
    if evidence_sets:
        result = solve_plan_metric_scale_batch(evidence_sets)
        for solution in result.solutions:
            solutions[solution.plan_key] = attach_plan_metric_diagnostics(
                solution,
                prepared[solution.plan_key],
                image_sizes[solution.plan_key],
                retry_count=retry_count,
            )
        failures.update({item.plan_key: item.reason for item in result.failures})
    return solutions, failures


# ### Public batch transcription API ###
def transcribe_multi_plan_measurements(
    raster_pairs: Sequence[PlanMeasurementRasterPair],
    analyzer: MultiPlanMeasurementAnalyzer,
    *,
    cancellation_check: CancellationCheck | None = None,
    prompt: str = MULTI_PLAN_MEASUREMENT_PROMPT,
    focus_regions_by_plan: Mapping[str, Sequence[tuple[float, float, float, float]]]
    | None = None,
) -> tuple[PlanMeasurementGroup, ...]:
    """Transcribe all pairs together while returning per-plan evidence."""

    pairs = validate_raster_plan_pairs(raster_pairs)
    if not callable(analyzer):
        raise TypeError("The multi-plan measurement analyzer must be callable.")
    _raise_if_cancelled(cancellation_check)
    encoded_pairs = tuple(
        EncodedPlanMeasurementPair(
            plan_id=pair.plan_id,
            original_image_data_uri=encode_plan_measurement_image(pair.original_image),
            corrected_image_data_uri=encode_plan_measurement_image(
                pair.corrected_image
            ),
            measurement_reference_image_data_uri=(
                None
                if pair.measurement_reference_image is None
                else encode_plan_measurement_image(pair.measurement_reference_image)
            ),
            detail_images=prepare_plan_metric_detail_images(
                pair.original_image,
                measurement_reference_image=pair.measurement_reference_image,
                focus_regions=(focus_regions_by_plan or {}).get(pair.plan_id, ()),
                cancellation_check=cancellation_check,
            ),
        )
        for pair in pairs
    )
    _raise_if_cancelled(cancellation_check)
    result_schema = build_multi_plan_measurement_result_schema(
        tuple(pair.plan_id for pair in encoded_pairs)
    )
    try:
        payload = analyzer(
            encoded_pairs,
            prompt=prompt,
            result_schema=result_schema,
            cancellation_check=cancellation_check,
        )
    except PlanMetricCalibrationCancelled:
        raise
    except PlanMetricCalibrationError:
        raise
    except Exception:  # noqa: BLE001 - never expose provider internals or secrets.
        raise PlanMetricCalibrationInferenceError(
            "Multi-plan measurement transcription failed."
        ) from None
    _raise_if_cancelled(cancellation_check)
    return parse_multi_plan_measurement_result(
        payload,
        expected_plan_ids=tuple(pair.plan_id for pair in encoded_pairs),
    )


def build_multi_plan_measurement_result_schema(
    plan_ids: Sequence[str],
) -> dict[str, Any]:
    """Build one strict result schema keyed by the supplied stable plan IDs."""

    normalized_ids = validate_plan_measurement_ids(plan_ids)
    grouped_properties = {
        plan_id: _bounded_plan_measurement_schema() for plan_id in normalized_ids
    }
    return {
        "type": "object",
        "properties": {
            "plans": {
                "type": "object",
                "properties": grouped_properties,
                "required": list(normalized_ids),
                "additionalProperties": False,
            }
        },
        "required": ["plans"],
        "additionalProperties": False,
    }


def plan_measurement_schema_for_id(
    grouped_schema: Mapping[str, Any],
    plan_id: str,
) -> dict[str, Any]:
    """Extract one plan's ordinary schema for a sequential provider fallback."""

    normalized_id = validate_plan_measurement_id(plan_id)
    try:
        plans_schema = grouped_schema["properties"]["plans"]
        schema = plans_schema["properties"][normalized_id]
    except (KeyError, TypeError):
        raise PlanMetricCalibrationInferenceError(
            "The grouped plan measurement result schema is invalid."
        ) from None
    if not isinstance(schema, Mapping):
        raise PlanMetricCalibrationInferenceError(
            "The grouped plan measurement result schema is invalid."
        )
    return copy.deepcopy(dict(schema))


def _bounded_plan_measurement_schema() -> dict[str, Any]:
    """Bound each group so many loaded levels cannot exhaust output tokens."""

    schema = copy.deepcopy(PLAN_MEASUREMENT_RESULT_SCHEMA)
    measurements = schema["properties"]["measurements"]
    measurements["maxItems"] = min(
        int(measurements["maxItems"]),
        MAX_BATCH_MEASUREMENTS_PER_PLAN,
    )
    return schema


def parse_multi_plan_measurement_result(
    payload: object,
    *,
    expected_plan_ids: Sequence[str],
) -> tuple[PlanMeasurementGroup, ...]:
    """Validate grouped evidence without combining cross-raster coordinates."""

    normalized_ids = validate_plan_measurement_ids(expected_plan_ids)
    parsed = payload
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError:
            raise PlanMetricCalibrationInferenceError(
                "The multi-plan measurement transcription was not valid JSON."
            ) from None
    if not isinstance(parsed, Mapping):
        raise PlanMetricCalibrationInferenceError(
            "The multi-plan measurement transcription was invalid."
        )
    raw_plans = parsed.get("plans")
    if not isinstance(raw_plans, Mapping):
        raise PlanMetricCalibrationInferenceError(
            "The multi-plan measurement transcription omitted its plan groups."
        )
    if set(raw_plans) != set(normalized_ids):
        raise PlanMetricCalibrationInferenceError(
            "The multi-plan measurement transcription returned the wrong plan IDs."
        )
    groups: list[PlanMeasurementGroup] = []
    for plan_id in normalized_ids:
        raw_group = raw_plans[plan_id]
        if not isinstance(raw_group, Mapping):
            raise PlanMetricCalibrationInferenceError(
                "The multi-plan measurement transcription contained an invalid group."
            )
        raw_measurements = raw_group.get("measurements")
        if not isinstance(raw_measurements, list):
            raise PlanMetricCalibrationInferenceError(
                "The multi-plan measurement transcription omitted measurements."
            )
        if not raw_measurements:
            groups.append(PlanMeasurementGroup(plan_id, ()))
            continue
        measurements = parse_plan_measurement_result(raw_group)
        groups.append(PlanMeasurementGroup(plan_id, measurements))
    return tuple(groups)


# ### Input validation ###
def validate_plan_measurement_id(value: object) -> str:
    """Return one bounded identifier safe for a strict schema property name."""

    plan_id = str(value).strip()
    if (
        not plan_id
        or len(plan_id) > MAX_PLAN_MEASUREMENT_ID_CHARACTERS
        or _PLAN_MEASUREMENT_ID_PATTERN.fullmatch(plan_id) is None
    ):
        raise ValueError(
            "Plan measurement IDs must use letters, numbers, '.', '_', ':', "
            "or '-' and begin with a letter or number."
        )
    return plan_id


def validate_plan_measurement_ids(
    values: Sequence[str],
) -> tuple[str, ...]:
    """Validate a non-empty, bounded sequence of distinct plan IDs."""

    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError("Plan measurement IDs must form a sequence.")
    plan_ids = tuple(validate_plan_measurement_id(value) for value in values)
    if not plan_ids:
        raise ValueError("At least one plan measurement ID is required.")
    if len(plan_ids) > MAX_MULTI_PLAN_MEASUREMENT_COUNT:
        raise ValueError("Too many plans were supplied for one measurement request.")
    if len(set(plan_ids)) != len(plan_ids):
        raise ValueError("Plan measurement IDs must be unique.")
    return plan_ids


def validate_encoded_plan_pairs(
    values: Sequence[EncodedPlanMeasurementPair],
) -> tuple[EncodedPlanMeasurementPair, ...]:
    """Validate distinct encoded plan pairs without altering their order."""

    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError("Encoded plan pairs must form a sequence.")
    pairs = tuple(values)
    if not all(isinstance(pair, EncodedPlanMeasurementPair) for pair in pairs):
        raise TypeError("Encoded plan pairs contain an invalid item.")
    validate_plan_measurement_ids(tuple(pair.plan_id for pair in pairs))
    return pairs


def validate_raster_plan_pairs(
    values: Sequence[PlanMeasurementRasterPair],
) -> tuple[PlanMeasurementRasterPair, ...]:
    """Validate distinct raster pairs without solving any plan scale."""

    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError("Plan raster pairs must form a sequence.")
    pairs = tuple(values)
    if not all(isinstance(pair, PlanMeasurementRasterPair) for pair in pairs):
        raise TypeError("Plan raster pairs contain an invalid item.")
    validate_plan_measurement_ids(tuple(pair.plan_id for pair in pairs))
    return pairs


def _validate_image_data_uri(value: object, label: str) -> str:
    uri = str(value).strip()
    if not uri.startswith("data:image/") or ";base64," not in uri:
        raise ValueError(f"The {label} must be an encoded image data URI.")
    return uri


def _raise_if_cancelled(
    cancellation_check: CancellationCheck | None,
) -> None:
    if cancellation_check is not None and bool(cancellation_check()):
        raise PlanMetricCalibrationCancelled("Plan metric calibration was cancelled.")
