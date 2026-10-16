# ### Imports ###
from __future__ import annotations

import unittest
from dataclasses import replace

import cv2
import numpy as np

from housemaker.plan_metric_refinement import (
    PlanMeasurementRefinementCancelled,
    refine_plan_measurements,
)
from housemaker.plan_metric_state import PlanMeasurementEvidence


# ### Synthetic plan helpers ###
def _measurement(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    size: int = 600,
    label: str = "Living room interior width",
) -> PlanMeasurementEvidence:
    return PlanMeasurementEvidence(
        label=label,
        value_meters=4.25,
        start_normalized=(start[0] / size, start[1] / size),
        end_normalized=(end[0] / size, end[1] / size),
        corrected_start_normalized=(start[0] / size, start[1] / size),
        corrected_end_normalized=(end[0] / size, end[1] / size),
        confidence="medium",
    )


def _blank(size: int = 600) -> np.ndarray:
    return np.full((size, size), 255, dtype=np.uint8)


def _anchors(image: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    for x, y in (start, end):
        cv2.line(image, (x, y - 25), (x, y + 25), 0, 1)


# ### Endpoint refinement tests ###
class PlanMetricRefinementTests(unittest.TestCase):
    def test_refines_small_errors_without_changing_source_values_or_confidence(
        self,
    ) -> None:
        image = _blank()
        _anchors(image, (100, 150), (400, 150))
        evidence = _measurement((101.5, 150), (398.5, 150))

        result = refine_plan_measurements((evidence,), image)

        self.assertEqual(result.refined_count, 1)
        self.assertEqual(result.rejected_count, 0)
        self.assertEqual(
            result.measurements[0],
            replace(
                evidence,
                corrected_start_normalized=(100 / 600, 150 / 600),
                corrected_end_normalized=(400 / 600, 150 / 600),
            ),
        )
        self.assertEqual(result.geometry_warning, "")

    def test_vertical_span_uses_horizontal_wall_lines(self) -> None:
        image = _blank()
        cv2.line(image, (125, 100), (175, 100), 0, 1)
        cv2.line(image, (125, 450), (175, 450), 0, 1)
        evidence = _measurement((150, 101), (150, 449))

        result = refine_plan_measurements((evidence,), image)

        self.assertEqual(result.refined_count, 1)
        self.assertEqual(
            result.measurements[0].corrected_start_normalized, (0.25, 100 / 600)
        )
        self.assertEqual(result.measurements[0].corrected_end_normalized, (0.25, 0.75))

    def test_ambiguity_prevents_jumping_between_nearby_wall_faces(self) -> None:
        image = _blank()
        _anchors(image, (100, 150), (400, 150))
        _anchors(image, (105, 150), (395, 150))
        evidence = _measurement((101, 150), (399, 150))

        result = refine_plan_measurements((evidence,), image)

        self.assertEqual(result.measurements, (evidence,))
        self.assertEqual(result.refined_count, 0)

    def test_wide_filled_wall_is_not_ignored_as_a_competing_boundary(self) -> None:
        image = _blank()
        _anchors(image, (100, 150), (400, 150))
        cv2.rectangle(image, (105, 125), (125, 175), 0, -1)
        evidence = _measurement((101, 150), (400, 150))

        result = refine_plan_measurements((evidence,), image)

        self.assertEqual(result.measurements, (evidence,))

    def test_blank_ambiguous_diagonal_and_distant_evidence_are_unchanged(self) -> None:
        evidence = _measurement((101, 150), (399, 150))
        blank_result = refine_plan_measurements((evidence,), _blank())
        self.assertEqual(blank_result.measurements, (evidence,))
        self.assertEqual(blank_result.refined_count, 0)
        image = _blank()
        _anchors(image, (100, 150), (400, 250))
        diagonal = _measurement((101, 150), (399, 250))
        self.assertEqual(
            refine_plan_measurements((diagonal,), image).measurements, (diagonal,)
        )
        distant = _measurement((110, 150), (390, 250))
        self.assertEqual(
            refine_plan_measurements((distant,), image).measurements, (distant,)
        )

    def test_large_relative_correction_is_not_applied_to_short_span(self) -> None:
        image = _blank()
        _anchors(image, (100, 150), (130, 150))
        evidence = _measurement((101, 150), (129, 150))

        result = refine_plan_measurements((evidence,), image)

        self.assertEqual(result.measurements, (evidence,))


# ### Annotation-removal geometry tests ###
class PlanMetricGeometryCheckTests(unittest.TestCase):
    def _images_and_evidence(
        self, *, deform: bool
    ) -> tuple[np.ndarray, np.ndarray, tuple[PlanMeasurementEvidence, ...]]:
        reference, final = _blank(), _blank()
        measurements = []
        for index, y in enumerate((80, 200, 320, 440)):
            start, end = (100, y), (400, y)
            final_end = (460, y) if deform and index >= 2 else end
            _anchors(reference, start, end)
            _anchors(final, start, final_end)
            evidence = _measurement(start, end, label=f"Room {index} interior width")
            measurements.append(
                replace(
                    evidence,
                    corrected_end_normalized=(final_end[0] / 600, final_end[1] / 600),
                )
            )
        return reference, final, tuple(measurements)

    def test_warns_for_multiple_supported_nonuniform_span_changes(self) -> None:
        reference, final, evidence = self._images_and_evidence(deform=True)

        result = refine_plan_measurements(
            evidence, final, measurement_reference_image=reference
        )

        self.assertIn("different proportions", result.geometry_warning)
        self.assertIn("20%", result.geometry_warning)
        self.assertEqual(result.measurements, evidence)
        self.assertEqual(result.rejected_count, 0)

    def test_uniform_scaling_and_image_resizing_do_not_trigger_warning(self) -> None:
        reference, final, evidence = self._images_and_evidence(deform=False)
        final = cv2.resize(final, (900, 900), interpolation=cv2.INTER_NEAREST)

        result = refine_plan_measurements(
            evidence, final, measurement_reference_image=reference
        )

        self.assertEqual(result.geometry_warning, "")

    def test_blank_reference_and_single_disagreement_do_not_establish_deformation(
        self,
    ) -> None:
        reference, final, evidence = self._images_and_evidence(deform=True)
        blank_result = refine_plan_measurements(
            evidence, final, measurement_reference_image=_blank()
        )
        self.assertEqual(blank_result.geometry_warning, "")
        insufficient = refine_plan_measurements(
            evidence[:3], final, measurement_reference_image=reference
        )
        self.assertEqual(insufficient.geometry_warning, "")

    def test_duplicate_readings_cannot_supply_independent_geometry_support(
        self,
    ) -> None:
        reference, final, evidence = self._images_and_evidence(deform=True)
        duplicated = (
            evidence[0],
            replace(evidence[0], label="Repeated label"),
            evidence[2],
            replace(evidence[2], label="Another repeated label"),
        )

        result = refine_plan_measurements(
            duplicated, final, measurement_reference_image=reference
        )

        self.assertEqual(result.geometry_warning, "")


# ### Cancellation tests ###
class PlanMetricRefinementCancellationTests(unittest.TestCase):
    def test_cancellation_is_checked_before_image_work_and_during_extraction(
        self,
    ) -> None:
        evidence = (_measurement((101, 150), (399, 150)),)
        with self.assertRaises(PlanMeasurementRefinementCancelled):
            refine_plan_measurements(
                evidence, _blank(), cancellation_check=lambda: True
            )
        calls = 0

        def cancelled() -> bool:
            nonlocal calls
            calls += 1
            return calls == 3

        image = _blank()
        _anchors(image, (100, 150), (400, 150))
        with self.assertRaises(PlanMeasurementRefinementCancelled):
            refine_plan_measurements(evidence, image, cancellation_check=cancelled)
        self.assertEqual(calls, 3)

    def test_callback_exception_propagates(self) -> None:
        def cancelled() -> bool:
            raise LookupError("Caller cancelled the batch")

        with self.assertRaisesRegex(LookupError, "Caller cancelled"):
            refine_plan_measurements((), _blank(), cancellation_check=cancelled)


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
