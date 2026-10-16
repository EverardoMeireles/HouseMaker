# ### Imports ###
from __future__ import annotations

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from housemaker.plan_metric_batch import PlanMetricBatchCoordinator
from housemaker.plan_metric_calibration import (
    PlanMetricCalibrationCancelled,
    PlanMetricCalibrationInferenceError,
    calibrate_and_normalize_plan,
    merge_plan_metric_measurements,
    select_plan_metric_review,
    solve_plan_metric_scale,
    solve_reviewed_plan_metrics,
)
from housemaker.plan_metric_refinement import PlanMeasurementRefinementResult
from housemaker.plan_metric_state import PlanMeasurementEvidence


# ### Independent evidence fixtures ###
def _spans() -> tuple[PlanMeasurementEvidence, ...]:
    return tuple(
        PlanMeasurementEvidence(label, value, start, end, start, end, "high")
        for label, value, start, end in (
            ("Salon inner width", 4.0, (0.1, 0.1), (0.5, 0.1)),
            ("Hall inner width", 5.0, (0.2, 0.4), (0.7, 0.4)),
            ("Salon inner depth", 3.0, (0.2, 0.2), (0.2, 0.5)),
            ("Bedroom inner depth", 4.0, (0.8, 0.2), (0.8, 0.6)),
        )
    )


def _payload(spans: tuple[PlanMeasurementEvidence, ...]) -> dict[str, object]:
    return {"measurements": [item.to_dict() for item in spans]}


def _unrelated_bad_spans() -> tuple[PlanMeasurementEvidence, ...]:
    return tuple(
        PlanMeasurementEvidence(
            f"Uncertain span {index}",
            value,
            (0.1, row),
            (0.5, row),
            (0.1, row),
            (0.5, row),
            "high",
        )
        for index, (value, row) in enumerate(
            ((8.0, 0.65), (12.0, 0.72), (20.0, 0.79), (32.0, 0.86))
        )
    )


# ### Adaptive calibration tests ###
class PlanMetricReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.full((1000, 1000), 255, dtype=np.uint8)

    def test_missing_axis_gets_one_targeted_retry_and_becomes_verified(self) -> None:
        calls = []

        def analyzer(*_args, **kwargs):
            calls.append(kwargs)
            return _payload(_spans()[:2] if len(calls) == 1 else _spans())

        result = calibrate_and_normalize_plan(self.image, self.image, analyzer)
        self.assertEqual(len(calls), 2)
        self.assertIn("0 vertical", calls[1]["prompt"])
        self.assertNotIn("0.01", calls[1]["prompt"])
        self.assertEqual(result.calibration.confidence, "high")
        self.assertAlmostEqual(result.calibration.estimated_meters_per_pixel, 0.01)
        diagnostics = result.calibration.diagnostics
        self.assertEqual(diagnostics.accepted_count, 4)
        self.assertEqual(diagnostics.horizontal_count, 2)
        self.assertEqual(diagnostics.vertical_count, 2)
        self.assertEqual(diagnostics.retry_count, 1)

    def test_balanced_success_needs_no_retry(self) -> None:
        calls = []

        def analyzer(*_args, **kwargs):
            calls.append(kwargs)
            return _payload(_spans())

        result = calibrate_and_normalize_plan(self.image, self.image, analyzer)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.calibration.diagnostics.retry_count, 0)

    def test_empty_first_read_can_recover_on_retry(self) -> None:
        calls = []

        def analyzer(*_args, **kwargs):
            calls.append(kwargs)
            return _payload(() if len(calls) == 1 else _spans())

        result = calibrate_and_normalize_plan(self.image, self.image, analyzer)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result.calibration.confidence, "high")

    def test_failed_optional_retry_preserves_usable_scale(self) -> None:
        calls = []

        def analyzer(*_args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 2:
                raise PlanMetricCalibrationInferenceError("Safe test error.")
            return _payload(_spans()[:2])

        result = calibrate_and_normalize_plan(self.image, self.image, analyzer)
        self.assertAlmostEqual(result.calibration.estimated_meters_per_pixel, 0.01)
        self.assertEqual(result.calibration.diagnostics.retry_count, 1)
        self.assertIn("unavailable", " ".join(result.calibration.diagnostics.reasons))

    def test_fresh_retry_is_not_outvoted_by_failed_first_read(self) -> None:
        reads = iter((_unrelated_bad_spans(), _spans()[:3]))
        result = calibrate_and_normalize_plan(
            self.image,
            self.image,
            lambda *_args, **_kwargs: _payload(next(reads)),
        )
        self.assertAlmostEqual(result.calibration.estimated_meters_per_pixel, 0.01)
        self.assertEqual(result.calibration.diagnostics.accepted_count, 3)
        self.assertEqual(result.calibration.diagnostics.retry_count, 1)
        self.assertEqual(result.calibration.confidence, "medium")

    def test_corrected_retry_values_are_not_deleted_by_failed_first_read(self) -> None:
        bad = tuple(
            replace(item, value_meters=item.value_meters * multiplier)
            for item, multiplier in zip(_spans(), (2.0, 3.0, 5.0, 8.0), strict=True)
        )
        reads = iter((bad, _spans()))
        result = calibrate_and_normalize_plan(
            self.image,
            self.image,
            lambda *_args, **_kwargs: _payload(next(reads)),
        )
        self.assertAlmostEqual(result.calibration.estimated_meters_per_pixel, 0.01)
        self.assertEqual(result.calibration.diagnostics.accepted_count, 4)
        self.assertEqual(result.calibration.confidence, "medium")
        self.assertIn("conflicting values", " ".join(result.calibration.diagnostics.reasons))

    def test_repeated_span_does_not_raise_confidence(self) -> None:
        merged, conflicts = merge_plan_metric_measurements(_spans()[:2], _spans()[:2])
        self.assertEqual(conflicts, 0)
        self.assertEqual(len(merged), 2)
        self.assertEqual(
            solve_plan_metric_scale(merged, corrected_image_size=(1000, 1000))[2],
            "medium",
        )

    def test_three_spans_with_only_one_vertical_cannot_claim_high(self) -> None:
        result = solve_plan_metric_scale(
            _spans()[:3], corrected_image_size=(1000, 1000)
        )
        self.assertEqual(result[2], "medium")

    def test_contradictory_reread_is_not_an_independent_measurement(self) -> None:
        changed = replace(_spans()[0], value_meters=8.0)
        merged, conflicts = merge_plan_metric_measurements(_spans(), (changed,))
        self.assertEqual(conflicts, 1)
        self.assertEqual(len(merged), 3)
        self.assertNotIn(_spans()[0], merged)
        self.assertNotIn(changed, merged)

    def test_retry_cannot_trade_large_disagreement_for_an_extra_span(self) -> None:
        prepared = PlanMeasurementRefinementResult(_spans()[:2], 0)
        baseline, _failure = solve_reviewed_plan_metrics("plan", prepared, (1000, 1000))
        worse = replace(baseline, relative_residual=0.079, measurements=_spans()[:3])
        selected = select_plan_metric_review(baseline, worse)
        self.assertEqual(selected.relative_residual, baseline.relative_residual)
        self.assertEqual(selected.measurements, baseline.measurements)

    def test_geometry_warning_cannot_disappear_when_retry_has_less_support(
        self,
    ) -> None:
        prepared = [
            PlanMeasurementRefinementResult(_spans(), 2, "Span proportions changed."),
            PlanMeasurementRefinementResult(_spans(), 0),
        ]
        with patch(
            "housemaker.plan_metric_calibration.refine_plan_measurements",
            side_effect=prepared,
        ):
            result = calibrate_and_normalize_plan(
                self.image,
                self.image,
                lambda *_args, **_kwargs: _payload(_spans()),
            )
        self.assertEqual(result.calibration.confidence, "medium")
        self.assertEqual(result.calibration.diagnostics.refined_count, 2)
        self.assertEqual(
            result.calibration.diagnostics.geometry_warning, "Span proportions changed."
        )

    def test_rejected_retry_keeps_new_geometry_warning(self) -> None:
        prepared = PlanMeasurementRefinementResult(_spans(), 0)
        baseline, _failure = solve_reviewed_plan_metrics("plan", prepared, (1000, 1000))
        warning = "Span proportions changed."
        candidate = replace(
            baseline,
            relative_residual=0.079,
            diagnostics=replace(baseline.diagnostics, geometry_warning=warning),
        )
        for retry in (candidate, None):
            with self.subTest(retry_solved=retry is not None):
                selected = select_plan_metric_review(
                    baseline,
                    retry,
                    geometry_warning=warning if retry is None else "",
                )
                self.assertEqual(selected.relative_residual, baseline.relative_residual)
                self.assertEqual(selected.confidence, "medium")
                self.assertEqual(selected.diagnostics.geometry_warning, warning)
                self.assertIn(warning, selected.diagnostics.reasons)


# ### Coordinated retry tests ###
class PlanMetricBatchReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.full((1000, 1000), 255, dtype=np.uint8)

    def test_only_weak_level_is_sent_again(self) -> None:
        calls = []

        def analyzer(pairs, **kwargs):
            calls.append((tuple(pair.plan_id for pair in pairs), kwargs["prompt"]))
            return {
                "plans": {
                    pair.plan_id: _payload(
                        _spans()[:2]
                        if pair.plan_id == "L3" and len(calls) == 1
                        else _spans()
                    )
                    for pair in pairs
                }
            }

        coordinator = PlanMetricBatchCoordinator(("L2", "L3"), analyzer)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(coordinator.normalize, level, self.image, self.image)
                for level in ("L2", "L3")
            ]
            results = [future.result(timeout=5) for future in futures]
        self.assertEqual([item[0] for item in calls], [("L2", "L3"), ("L3",)])
        self.assertIn("PLAN ID L3", calls[1][1])
        self.assertEqual(
            [result.calibration.diagnostics.retry_count for result in results], [0, 1]
        )
        self.assertTrue(
            all(result.calibration.confidence == "high" for result in results)
        )

    def test_failed_levels_recover_from_independent_retry(self) -> None:
        calls = []

        def analyzer(pairs, **_kwargs):
            calls.append(tuple(pair.plan_id for pair in pairs))
            return {
                "plans": {
                    pair.plan_id: _payload(
                        _unrelated_bad_spans() if len(calls) == 1 else _spans()[:3]
                    )
                    for pair in pairs
                }
            }

        coordinator = PlanMetricBatchCoordinator(("L2", "L3"), analyzer)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(coordinator.normalize, level, self.image, self.image)
                for level in ("L2", "L3")
            ]
            results = [future.result(timeout=5) for future in futures]
        self.assertEqual(calls, [("L2", "L3"), ("L2", "L3")])
        for result in results:
            self.assertAlmostEqual(result.calibration.estimated_meters_per_pixel, 0.01)
            self.assertEqual(result.calibration.diagnostics.accepted_count, 3)
            self.assertEqual(result.calibration.diagnostics.retry_count, 1)

    def test_cancellation_during_retry_releases_every_waiting_level(self) -> None:
        retry_started = threading.Event()
        release = threading.Event()
        calls = []

        def analyzer(pairs, **_kwargs):
            calls.append(pairs)
            if len(calls) == 2:
                retry_started.set()
                release.wait(3)
            return {"plans": {pair.plan_id: _payload(_spans()[:2]) for pair in pairs}}

        coordinator = PlanMetricBatchCoordinator(("L2", "L3"), analyzer)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(coordinator.normalize, level, self.image, self.image)
                for level in ("L2", "L3")
            ]
            try:
                self.assertTrue(retry_started.wait(3))
                coordinator.cancel()
            finally:
                release.set()
            for future in futures:
                with self.assertRaises(PlanMetricCalibrationCancelled):
                    future.result(timeout=3)


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
