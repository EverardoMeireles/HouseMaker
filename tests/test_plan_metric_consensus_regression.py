# ### Imports ###
from __future__ import annotations

import unittest
from dataclasses import replace

from housemaker.plan_metric_calibration import (
    PlanMetricCalibrationInferenceError,
    solve_plan_metric_scale,
)
from housemaker.plan_metric_state import PlanMeasurementEvidence


# ### Independent measurement fixtures ###
def _span(label, scale, start, end, *, confidence="high"):
    distance = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
    return PlanMeasurementEvidence(
        label, scale * distance * 1000, start, end, start, end, confidence
    )


def _balanced_spans(scale=0.01):
    return tuple(
        _span(f"Verified {index}", scale, start, end)
        for index, (start, end) in enumerate(
            (
                ((0.05, 0.1), (0.45, 0.1)),
                ((0.05, 0.3), (0.55, 0.3)),
                ((0.65, 0.1), (0.65, 0.5)),
                ((0.85, 0.1), (0.85, 0.5)),
            )
        )
    )


def _rival_spans():
    return tuple(
        _span(f"Rival {index}", 0.025, start, end)
        for index, (start, end) in enumerate(
            (
                ((0.05, 0.45), (0.45, 0.45)),
                ((0.05, 0.5), (0.55, 0.5)),
                ((0.75, 0.55), (0.75, 0.95)),
                ((0.95, 0.55), (0.95, 0.95)),
            )
        )
    )


def _scattered_spans(scales, *, confidence="high"):
    return tuple(
        _span(
            f"Scattered {index}",
            scale,
            (0.1, 0.6 + index * 0.04),
            (0.4, 0.6 + index * 0.04),
            confidence=confidence,
        )
        for index, scale in enumerate(scales)
    )


def _solve(spans):
    return solve_plan_metric_scale(spans, corrected_image_size=(1000, 1000))


# ### Strong minority consensus regressions ###
class PlanMetricConsensusRegressionTests(unittest.TestCase):
    def test_scattered_low_confidence_readings_do_not_erase_verified_scale(self):
        good = _balanced_spans()
        outliers = _scattered_spans((0.016, 0.022, 0.030, 0.040), confidence="low")
        density, residual, confidence, accepted = _solve(good + outliers)
        self.assertAlmostEqual(density, 0.01)
        self.assertLess(residual, 1e-10)
        self.assertEqual(confidence, "medium")
        self.assertEqual(set(accepted), set(good))

    def test_disconnected_high_confidence_outliers_do_not_form_a_rival_scale(self):
        good = _balanced_spans()
        outliers = _scattered_spans(
            (0.001, 0.002, 0.003, 0.005, 0.018, 0.025, 0.04, 0.06)
        )
        density, _residual, confidence, accepted = _solve(good + outliers)
        self.assertAlmostEqual(density, 0.01)
        self.assertEqual(confidence, "medium")
        self.assertEqual(set(accepted), set(good))

    def test_equally_supported_balanced_scales_remain_rejected(self):
        with self.assertRaisesRegex(
            PlanMetricCalibrationInferenceError, "Largest agreeing group: 4 of 8"
        ):
            _solve(_balanced_spans() + _rival_spans())

    def test_conflicting_values_for_same_spans_remain_competing_evidence(self):
        original = _balanced_spans()
        contradictory = tuple(
            replace(item, value_meters=item.value_meters * 2) for item in original
        )
        with self.assertRaisesRegex(
            PlanMetricCalibrationInferenceError, "reliable scale"
        ):
            _solve(original + contradictory)

    def test_three_reliable_rival_readings_block_four_reading_minority(self):
        evidence = _balanced_spans() + _rival_spans()[:3] + _scattered_spans((0.06,))
        with self.assertRaisesRegex(
            PlanMetricCalibrationInferenceError, "reliable scale"
        ):
            _solve(evidence)

    def test_shared_borderline_reading_cannot_hide_competing_scale(self):
        rival = tuple(
            replace(item, value_meters=item.value_meters * 0.012 / 0.025)
            for item in _rival_spans()
        )
        outliers = _scattered_spans(
            (0.001, 0.002, 0.003, 0.0045, 0.02, 0.03, 0.05, 0.075, 0.12, 0.18)
        )
        for confidence in ("low", "high"):
            with self.subTest(bridge_confidence=confidence):
                bridge = _span(
                    "Borderline",
                    0.0112,
                    (0.08, 0.56),
                    (0.13, 0.56),
                    confidence=confidence,
                )
                evidence = _balanced_spans() + rival + (bridge,) + outliers
                with self.assertRaisesRegex(
                    PlanMetricCalibrationInferenceError, "reliable scale"
                ):
                    _solve(evidence)

    def test_three_measurements_do_not_qualify_as_a_strong_minority(self):
        evidence = _balanced_spans()[:3] + _scattered_spans((0.002, 0.025, 0.06))
        with self.assertRaisesRegex(
            PlanMetricCalibrationInferenceError, "reliable scale"
        ):
            _solve(evidence)

    def test_conflicting_rereads_of_two_spans_cannot_claim_four_independent_spans(self):
        original = (_balanced_spans()[0], _balanced_spans()[2])
        repeated = tuple(
            replace(
                item,
                label="Reread " + item.label,
                value_meters=item.value_meters * 1.058,
            )
            for item in original
        )
        evidence = original + repeated + _scattered_spans((0.002, 0.025, 0.04, 0.06))
        with self.assertRaisesRegex(
            PlanMetricCalibrationInferenceError, "reliable scale"
        ):
            _solve(evidence)

    def test_single_axis_minority_remains_rejected(self):
        horizontal = _scattered_spans((0.01,) * 4)
        outliers = tuple(
            replace(
                item,
                corrected_start_normalized=(0.5, item.start_normalized[1]),
                corrected_end_normalized=(0.8, item.end_normalized[1]),
            )
            for item in _scattered_spans((0.002, 0.025, 0.04, 0.06))
        )
        with self.assertRaisesRegex(
            PlanMetricCalibrationInferenceError, "reliable scale"
        ):
            _solve(horizontal + outliers)

    def test_minority_must_meet_existing_high_precision_requirement(self):
        good = tuple(
            replace(item, value_meters=item.value_meters * adjustment)
            for item, adjustment in zip(
                _balanced_spans(), (0.95, 1.0, 1.05, 1.0), strict=True
            )
        )
        with self.assertRaisesRegex(
            PlanMetricCalibrationInferenceError, "reliable scale"
        ):
            _solve(good + _scattered_spans((0.002, 0.025, 0.04, 0.06)))

    def test_strong_minority_does_not_bypass_physical_units_validation(self):
        with self.assertRaisesRegex(
            PlanMetricCalibrationInferenceError, "room-scale span"
        ):
            _solve(
                _balanced_spans(0.00001) + _scattered_spans((0.002, 0.025, 0.04, 0.06))
            )

    def test_clean_balanced_evidence_still_gets_high_confidence(self):
        self.assertEqual(_solve(_balanced_spans())[2], "high")


# ### Standalone test entry point ###
if __name__ == "__main__":
    unittest.main()
