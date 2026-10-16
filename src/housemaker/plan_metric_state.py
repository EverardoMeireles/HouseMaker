# ### Imports ###
from __future__ import annotations

import math
from dataclasses import dataclass

# ### Constants ###
PLAN_METRIC_CALIBRATION_VERSION = 1
PLAN_MEASUREMENT_CONFIDENCE_LEVELS = frozenset({"low", "medium", "high"})
PLAN_CALIBRATION_CONFIDENCE_LEVELS = PLAN_MEASUREMENT_CONFIDENCE_LEVELS
MAX_PLAN_MEASUREMENT_LABEL_CHARACTERS = 160
MAX_PLAN_MEASUREMENT_COUNT = 64
MAX_PLAN_METRIC_DIAGNOSTIC_REASONS = 8
MAX_PLAN_METRIC_DIAGNOSTIC_CHARACTERS = 320
MIN_PLAN_CALIBRATION_MEASUREMENT_COUNT = 2
PLAN_CALIBRATION_SCALE_RELATIVE_TOLERANCE = 1e-6
PLAN_MEASUREMENT_ENDPOINT_TOLERANCE = 0.005


# ### Measurement models ###
@dataclass(frozen=True)
class PlanMeasurementEvidence:
    """One metric dimension mapped from an annotated plan to the final plan."""

    label: str
    value_meters: float
    start_normalized: tuple[float, float]
    end_normalized: tuple[float, float]
    corrected_start_normalized: tuple[float, float]
    corrected_end_normalized: tuple[float, float]
    confidence: str

    def __post_init__(self) -> None:
        label = " ".join(str(self.label).split())[
            :MAX_PLAN_MEASUREMENT_LABEL_CHARACTERS
        ]
        value_meters = _positive_finite_number(
            self.value_meters,
            "Plan measurement value",
        )
        start = _normalized_point(
            self.start_normalized,
            "Plan measurement start",
        )
        end = _normalized_point(
            self.end_normalized,
            "Plan measurement end",
        )
        if math.dist(start, end) <= 1e-6:
            raise ValueError("Plan measurement endpoints must be distinct.")
        corrected_start = _normalized_point(
            self.corrected_start_normalized,
            "Corrected plan measurement start",
        )
        corrected_end = _normalized_point(
            self.corrected_end_normalized,
            "Corrected plan measurement end",
        )
        if math.dist(corrected_start, corrected_end) <= 1e-6:
            raise ValueError(
                "Corrected plan measurement endpoints must be distinct."
            )
        confidence = str(self.confidence).strip().lower()
        if confidence not in PLAN_MEASUREMENT_CONFIDENCE_LEVELS:
            raise ValueError("Plan measurement confidence is invalid.")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "value_meters", value_meters)
        object.__setattr__(self, "start_normalized", start)
        object.__setattr__(self, "end_normalized", end)
        object.__setattr__(
            self,
            "corrected_start_normalized",
            corrected_start,
        )
        object.__setattr__(
            self,
            "corrected_end_normalized",
            corrected_end,
        )
        object.__setattr__(self, "confidence", confidence)

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "value_meters": self.value_meters,
            "start_normalized": list(self.start_normalized),
            "end_normalized": list(self.end_normalized),
            "corrected_start_normalized": list(
                self.corrected_start_normalized
            ),
            "corrected_end_normalized": list(
                self.corrected_end_normalized
            ),
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, payload: object) -> PlanMeasurementEvidence:
        if not isinstance(payload, dict):
            raise TypeError("Plan measurement JSON must contain an object.")
        return cls(
            label=payload.get("label", ""),
            value_meters=payload.get("value_meters"),
            start_normalized=payload.get("start_normalized"),
            end_normalized=payload.get("end_normalized"),
            corrected_start_normalized=payload.get(
                "corrected_start_normalized"
            ),
            corrected_end_normalized=payload.get(
                "corrected_end_normalized"
            ),
            confidence=payload.get("confidence", "low"),
        )


# ### Calibration diagnostics ###
@dataclass(frozen=True)
class PlanMetricDiagnostics:
    """Explain scale consistency without claiming absolute measurement accuracy."""

    candidate_count: int
    accepted_count: int
    rejected_count: int
    horizontal_count: int
    vertical_count: int
    relative_disagreement: float
    refined_count: int = 0
    retry_count: int = 0
    reasons: tuple[str, ...] = ()
    geometry_warning: str = ""

    def __post_init__(self) -> None:
        for name in (
            "candidate_count",
            "accepted_count",
            "rejected_count",
            "horizontal_count",
            "vertical_count",
            "refined_count",
            "retry_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"Plan metric diagnostic {name} must be an integer.")
            if not 0 <= value <= MAX_PLAN_MEASUREMENT_COUNT:
                raise ValueError(f"Plan metric diagnostic {name} is out of range.")
        if self.accepted_count + self.rejected_count != self.candidate_count:
            raise ValueError("Plan metric diagnostic measurement counts disagree.")
        if self.horizontal_count + self.vertical_count > self.accepted_count:
            raise ValueError("Plan metric diagnostic axis counts exceed accepted spans.")
        if self.refined_count > self.candidate_count:
            raise ValueError("Plan metric diagnostic refined count exceeds candidates.")
        if not isinstance(self.reasons, (list, tuple)):
            raise TypeError("Plan metric diagnostic reasons must be a sequence.")
        if len(self.reasons) > MAX_PLAN_METRIC_DIAGNOSTIC_REASONS:
            raise ValueError("Too many plan metric diagnostic reasons.")
        reasons = tuple(
            _diagnostic_text(value, "reason") for value in self.reasons
        )
        object.__setattr__(self, "reasons", tuple(value for value in reasons if value))
        object.__setattr__(
            self,
            "geometry_warning",
            _diagnostic_text(self.geometry_warning, "geometry warning"),
        )
        object.__setattr__(
            self,
            "relative_disagreement",
            _non_negative_finite_number(
                self.relative_disagreement,
                "Plan metric diagnostic disagreement",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_count": self.candidate_count,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "horizontal_count": self.horizontal_count,
            "vertical_count": self.vertical_count,
            "relative_disagreement": self.relative_disagreement,
            "refined_count": self.refined_count,
            "retry_count": self.retry_count,
            "reasons": list(self.reasons),
            "geometry_warning": self.geometry_warning,
        }

    @classmethod
    def from_dict(cls, payload: object) -> PlanMetricDiagnostics:
        if not isinstance(payload, dict):
            raise TypeError("Plan metric diagnostics JSON must contain an object.")
        return cls(
            candidate_count=payload.get("candidate_count"),
            accepted_count=payload.get("accepted_count"),
            rejected_count=payload.get("rejected_count"),
            horizontal_count=payload.get("horizontal_count"),
            vertical_count=payload.get("vertical_count"),
            relative_disagreement=payload.get("relative_disagreement"),
            refined_count=payload.get("refined_count", 0),
            retry_count=payload.get("retry_count", 0),
            reasons=payload.get("reasons", ()),
            geometry_warning=payload.get("geometry_warning", ""),
        )


# ### Calibration model ###
@dataclass(frozen=True)
class PlanMetricCalibration:
    """Metric density of a corrected raster with backward-compatible history.

    ``target_meters_per_pixel`` describes the image currently stored on the level.
    New calibrations preserve that raster, so target equals estimated density and
    normalization_scale is 1. Legacy version-1 records may describe a resampled
    image; retain their target density and historical sizes exactly as saved.
    """

    estimated_meters_per_pixel: float
    target_meters_per_pixel: float
    normalization_scale: float
    relative_residual: float
    confidence: str
    source_image_size: tuple[int, int]
    pre_normalization_size: tuple[int, int]
    normalized_image_size: tuple[int, int]
    measurements: tuple[PlanMeasurementEvidence, ...]
    version: int = PLAN_METRIC_CALIBRATION_VERSION
    normalized_image_sha256: str = ""
    diagnostics: PlanMetricDiagnostics | None = None

    def __post_init__(self) -> None:
        version = int(self.version)
        if version != PLAN_METRIC_CALIBRATION_VERSION:
            raise ValueError("Plan metric calibration version is unsupported.")
        estimated = _positive_finite_number(
            self.estimated_meters_per_pixel,
            "Estimated plan meters per pixel",
        )
        target = _positive_finite_number(
            self.target_meters_per_pixel,
            "Target plan meters per pixel",
        )
        scale = _positive_finite_number(
            self.normalization_scale,
            "Plan normalization scale",
        )
        residual = _non_negative_finite_number(
            self.relative_residual,
            "Plan calibration residual",
        )
        confidence = str(self.confidence).strip().lower()
        if confidence not in PLAN_CALIBRATION_CONFIDENCE_LEVELS:
            raise ValueError("Plan calibration confidence is invalid.")
        source_size = _image_size(self.source_image_size, "source plan")
        pre_normalization_size = _image_size(
            self.pre_normalization_size,
            "corrected plan",
        )
        normalized_size = _image_size(
            self.normalized_image_size,
            "normalized plan",
        )
        measurements = tuple(self.measurements)
        if self.diagnostics is not None and not isinstance(
            self.diagnostics, PlanMetricDiagnostics
        ):
            raise TypeError("Plan calibration diagnostics are invalid.")
        image_fingerprint = str(self.normalized_image_sha256).strip().lower()
        if image_fingerprint and (
            len(image_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in image_fingerprint
            )
        ):
            raise ValueError("Normalized plan fingerprint is invalid.")
        if not (
            MIN_PLAN_CALIBRATION_MEASUREMENT_COUNT
            <= len(measurements)
            <= MAX_PLAN_MEASUREMENT_COUNT
        ):
            raise ValueError(
                "Plan calibration requires between "
                f"{MIN_PLAN_CALIBRATION_MEASUREMENT_COUNT} and "
                f"{MAX_PLAN_MEASUREMENT_COUNT} measurements."
            )
        if not all(
            isinstance(item, PlanMeasurementEvidence) for item in measurements
        ):
            raise TypeError("Plan calibration measurements are invalid.")
        expected_scale = estimated / target
        if not math.isclose(
            scale,
            expected_scale,
            rel_tol=PLAN_CALIBRATION_SCALE_RELATIVE_TOLERANCE,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "Plan normalization scale must match the calibrated and "
                "target pixel densities."
            )
        expected_normalized_size = tuple(
            max(1, round(dimension * scale))
            for dimension in pre_normalization_size
        )
        if normalized_size != expected_normalized_size:
            raise ValueError(
                "Normalized plan size must match the corrected plan size "
                "and normalization scale."
            )
        if _independent_measurement_count(measurements) < 2:
            raise ValueError(
                "Plan calibration requires at least two independent "
                "measurement spans."
            )
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "estimated_meters_per_pixel", estimated)
        object.__setattr__(self, "target_meters_per_pixel", target)
        object.__setattr__(self, "normalization_scale", scale)
        object.__setattr__(self, "relative_residual", residual)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "source_image_size", source_size)
        object.__setattr__(
            self,
            "pre_normalization_size",
            pre_normalization_size,
        )
        object.__setattr__(self, "normalized_image_size", normalized_size)
        object.__setattr__(self, "measurements", measurements)
        object.__setattr__(
            self,
            "normalized_image_sha256",
            image_fingerprint,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "estimated_meters_per_pixel": self.estimated_meters_per_pixel,
            "target_meters_per_pixel": self.target_meters_per_pixel,
            "normalization_scale": self.normalization_scale,
            "relative_residual": self.relative_residual,
            "confidence": self.confidence,
            "source_image_size": list(self.source_image_size),
            "pre_normalization_size": list(self.pre_normalization_size),
            "normalized_image_size": list(self.normalized_image_size),
            "normalized_image_sha256": self.normalized_image_sha256,
            "measurements": [item.to_dict() for item in self.measurements],
            "diagnostics": (
                self.diagnostics.to_dict() if self.diagnostics is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, payload: object) -> PlanMetricCalibration:
        if not isinstance(payload, dict):
            raise TypeError("Plan metric calibration JSON must contain an object.")
        raw_measurements = payload.get("measurements")
        if not isinstance(raw_measurements, list):
            raise TypeError("Plan calibration measurements must be a list.")
        return cls(
            version=payload.get(
                "version",
                PLAN_METRIC_CALIBRATION_VERSION,
            ),
            estimated_meters_per_pixel=payload.get(
                "estimated_meters_per_pixel"
            ),
            target_meters_per_pixel=payload.get("target_meters_per_pixel"),
            normalization_scale=payload.get("normalization_scale"),
            relative_residual=payload.get("relative_residual"),
            confidence=payload.get("confidence", "low"),
            source_image_size=payload.get("source_image_size"),
            pre_normalization_size=payload.get("pre_normalization_size"),
            normalized_image_size=payload.get("normalized_image_size"),
            normalized_image_sha256=payload.get(
                "normalized_image_sha256",
                "",
            ),
            measurements=tuple(
                PlanMeasurementEvidence.from_dict(item)
                for item in raw_measurements
            ),
            diagnostics=(
                PlanMetricDiagnostics.from_dict(payload["diagnostics"])
                if payload.get("diagnostics") is not None
                else None
            ),
        )


# ### Presentation helpers ###
def ceiling_height_plan_measurements(
    calibration: PlanMetricCalibration | None,
) -> tuple[PlanMeasurementEvidence, ...]:
    """Return the strongest plan spans in a stable annotation order."""

    if calibration is None:
        return ()
    return tuple(
        sorted(
            calibration.measurements,
            key=lambda item: (
                {"high": 0, "medium": 1, "low": 2}[item.confidence],
                -item.value_meters,
                item.label.casefold(),
            ),
        )[:12]
    )


def build_ceiling_height_plan_context(
    level_name: str,
    calibration: PlanMetricCalibration | None,
) -> str:
    """Describe validated horizontal plan evidence without implying Z scale."""

    if calibration is None:
        return ""
    measurements = ceiling_height_plan_measurements(calibration)
    evidence = ", ".join(
        (
            f"M{index} - {item.label}: {item.value_meters:g} m"
            if item.label
            else f"M{index}: {item.value_meters:g} m"
        )
        for index, item in enumerate(measurements, start=1)
    )
    return (
        f"Metric floor-plan context for {str(level_name).strip()}: {evidence}. "
        "The supplied PLAN REFERENCE image marks those same spans as M1, M2, "
        "and so on. "
        f"The plan calibration is {calibration.confidence} confidence with "
        f"{calibration.relative_residual * 100.0:.1f}% relative disagreement. "
        "These are horizontal plan distances only, never a ceiling-height "
        "measurement. Use them only when a matching visible architectural span "
        "can be identified in the video frame."
    )


# ### Validation helpers ###
def _diagnostic_text(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Plan metric diagnostic {label} must be text.")
    return " ".join(value.split())[:MAX_PLAN_METRIC_DIAGNOSTIC_CHARACTERS]


def _positive_finite_number(value: object, label: str) -> float:
    number = _non_negative_finite_number(value, label)
    if number <= 0.0:
        raise ValueError(f"{label} must be greater than zero.")
    return number


def _non_negative_finite_number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a number.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be a number.") from error
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return number


def _normalized_point(value: object, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError(f"{label} must contain two coordinates.")
    coordinates = tuple(
        _non_negative_finite_number(coordinate, label)
        for coordinate in value
    )
    if any(coordinate > 1.0 for coordinate in coordinates):
        raise ValueError(f"{label} coordinates must be between zero and one.")
    return coordinates  # type: ignore[return-value]


def _image_size(value: object, label: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError(f"The {label} size must contain width and height.")
    if any(isinstance(dimension, bool) for dimension in value):
        raise TypeError(f"The {label} size must contain integers.")
    try:
        width, height = (int(dimension) for dimension in value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"The {label} size must contain integers.") from error
    if width < 1 or height < 1:
        raise ValueError(f"The {label} size must be positive.")
    return width, height


def _independent_measurement_count(
    measurements: tuple[PlanMeasurementEvidence, ...],
) -> int:
    """Count distinct corrected-image spans without trusting labels or OCR."""

    independent_spans: list[
        tuple[tuple[float, float], tuple[float, float]]
    ] = []
    for measurement in measurements:
        candidate = _ordered_measurement_span(
            measurement.corrected_start_normalized,
            measurement.corrected_end_normalized,
        )
        if any(
            math.dist(candidate[0], existing[0])
            <= PLAN_MEASUREMENT_ENDPOINT_TOLERANCE
            and math.dist(candidate[1], existing[1])
            <= PLAN_MEASUREMENT_ENDPOINT_TOLERANCE
            for existing in independent_spans
        ):
            continue
        independent_spans.append(candidate)
    return len(independent_spans)


def _ordered_measurement_span(
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return an orientation-independent endpoint pair for comparisons."""

    return (start, end) if start <= end else (end, start)
