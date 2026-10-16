# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import unittest
from dataclasses import replace

from housemaker.main import (
    _build_plan_metric_evidence_tooltip,
    _build_plan_metric_status_text,
)
from housemaker.plan_metric_state import (
    MAX_PLAN_METRIC_DIAGNOSTIC_CHARACTERS,
    PlanMeasurementEvidence,
    PlanMetricCalibration,
    PlanMetricDiagnostics,
)


# ### Test fixtures ###
def _diagnostics() -> PlanMetricDiagnostics:
    return PlanMetricDiagnostics(
        candidate_count=5,
        accepted_count=2,
        rejected_count=3,
        horizontal_count=2,
        vertical_count=0,
        relative_disagreement=0.021,
        refined_count=2,
        retry_count=1,
        reasons=("Only horizontal measurements were accepted.",),
        geometry_warning="Annotation removal moved one measured wall boundary.",
    )


def _calibration() -> PlanMetricCalibration:
    return PlanMetricCalibration(
        estimated_meters_per_pixel=0.02,
        target_meters_per_pixel=0.02,
        normalization_scale=1.0,
        relative_residual=0.021,
        confidence="medium",
        source_image_size=(1000, 800),
        pre_normalization_size=(1000, 800),
        normalized_image_size=(1000, 800),
        measurements=tuple(
            PlanMeasurementEvidence(
                label=label,
                value_meters=8.0,
                start_normalized=(0.1, y),
                end_normalized=(0.5, y),
                corrected_start_normalized=(0.1, y),
                corrected_end_normalized=(0.5, y),
                confidence="high",
            )
            for label, y in (("room width", 0.2), ("hall width", 0.6))
        ),
        diagnostics=_diagnostics(),
    )


# ### Persistence and validation tests ###
class PlanMetricDiagnosticsTests(unittest.TestCase):
    def test_diagnostics_round_trip_with_calibration(self) -> None:
        original = _calibration()

        restored = PlanMetricCalibration.from_dict(original.to_dict())

        self.assertEqual(restored, original)
        self.assertEqual(restored.diagnostics.reasons, _diagnostics().reasons)

    def test_legacy_calibration_without_diagnostics_remains_loadable(self) -> None:
        payload = _calibration().to_dict()
        del payload["diagnostics"]

        restored = PlanMetricCalibration.from_dict(payload)

        self.assertIsNone(restored.diagnostics)
        self.assertEqual(len(restored.measurements), 2)

    def test_invalid_diagnostic_payloads_are_rejected(self) -> None:
        invalid_fields = (
            {"candidate_count": True},
            {"retry_count": -1},
            {"accepted_count": 1.5},
            {"rejected_count": 1},
            {"horizontal_count": 3},
            {"refined_count": 6},
            {"relative_disagreement": float("nan")},
            {"relative_disagreement": -0.1},
            {"reasons": "not a sequence of reasons"},
            {"reasons": [42]},
            {"geometry_warning": False},
        )
        for fields in invalid_fields:
            with self.subTest(fields=fields):
                payload = _diagnostics().to_dict() | fields
                with self.assertRaises((TypeError, ValueError)):
                    PlanMetricDiagnostics.from_dict(payload)

    def test_diagnostic_text_is_bounded_and_normalized(self) -> None:
        diagnostics = replace(
            _diagnostics(),
            reasons=("  Only\n horizontal measurements.  ",),
            geometry_warning="g" * 1000,
        )

        self.assertEqual(diagnostics.reasons, ("Only horizontal measurements.",))
        self.assertEqual(
            len(diagnostics.geometry_warning),
            MAX_PLAN_METRIC_DIAGNOSTIC_CHARACTERS,
        )


# ### User-facing evidence tests ###
class PlanMetricDiagnosticsPresentationTests(unittest.TestCase):
    def test_status_explains_medium_confidence_and_measurement_consistency(self) -> None:
        text = _build_plan_metric_status_text(_calibration(), include_density=True)

        self.assertIn("medium confidence", text)
        self.assertIn("2 measurements accepted, 2.1% disagreement", text)
        self.assertIn("Only horizontal measurements were accepted", text)
        self.assertIn("50 px/m", text)

    def test_tooltip_includes_rejected_evidence_coverage_and_geometry_warning(self) -> None:
        text = _build_plan_metric_evidence_tooltip(_calibration())

        self.assertIn("2 accepted, 3 rejected, 5 candidates", text)
        self.assertIn("2 horizontal, 0 vertical", text)
        self.assertIn("Endpoints refined: 2 measurements", text)
        self.assertIn("additional analysis passes: 1", text)
        self.assertIn(_diagnostics().geometry_warning, text)
        self.assertIn("not guaranteed real-world measurement accuracy", text)

    def test_legacy_calibration_can_be_presented_without_diagnostics(self) -> None:
        calibration = replace(_calibration(), diagnostics=None)

        self.assertIn(
            "2 measurements accepted",
            _build_plan_metric_status_text(calibration, include_density=False),
        )
        self.assertIn("room width: 8 m", _build_plan_metric_evidence_tooltip(calibration))


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
