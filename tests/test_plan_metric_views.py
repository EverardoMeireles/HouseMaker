# ### Imports ###
from __future__ import annotations

import base64
import unittest

import cv2
import numpy as np

from housemaker.plan_metric_views import (
    PlanMeasurementDetailCancelled,
    PlanMeasurementDetailImage,
    build_plan_measurement_detail_images,
)


# ### Test helpers ###
def _decoded(detail: PlanMeasurementDetailImage) -> np.ndarray:
    encoded = base64.b64decode(detail.image_data_uri.split(",", 1)[1])
    return cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_UNCHANGED)


# ### Detail crop tests ###
class PlanMeasurementDetailTests(unittest.TestCase):
    def test_initial_crops_preserve_native_pixels_and_cover_overlapping_quadrants(self):
        source = np.full((2400, 3000), 255, dtype=np.uint8)
        source[100:120, 100:120] = 23
        details = build_plan_measurement_detail_images(source)

        self.assertEqual(len(details), 4)
        self.assertEqual(details[0].source_role, "original")
        self.assertEqual(details[0].bounds_normalized, (0.0, 0.0, 0.55, 0.55))
        self.assertEqual(_decoded(details[0]).shape, (1320, 1650))
        np.testing.assert_array_equal(_decoded(details[0]), source[:1320, :1650])
        self.assertLess(details[1].bounds_normalized[0], details[0].bounds_normalized[2])
        self.assertEqual(details[-1].bounds_normalized, (0.45, 0.45, 1.0, 1.0))
        self.assertIn("NEVER crop-local coordinates", details[0].label)
        self.assertIn("raw original", details[0].label)

    def test_focus_bounds_belong_to_rectified_reference_not_raw_photo(self):
        raw = np.zeros((1200, 1800), dtype=np.uint8)
        rectified = np.full((1000, 2000), 233, dtype=np.uint8)
        rectified[200:600, 500:1500] = 47
        details = build_plan_measurement_detail_images(
            raw, measurement_reference_image=rectified,
            focus_regions=((0.25, 0.2, 0.75, 0.6),),
        )

        self.assertEqual(len(details), 4)
        self.assertEqual([detail.source_role for detail in details],
                         ["original", "original", "original", "measurement_reference"])
        focus = details[-1]
        self.assertEqual(focus.bounds_normalized, (0.25, 0.2, 0.75, 0.6))
        np.testing.assert_array_equal(_decoded(focus), rectified[200:600, 500:1500])
        self.assertIn("RECTIFIED ANNOTATED PLAN", focus.label)

    def test_focus_regions_without_rectified_reference_are_not_reused_on_raw(self):
        image = np.zeros((200, 300), dtype=np.uint8)
        details = build_plan_measurement_detail_images(
            image, focus_regions=((0.2, 0.2, 0.8, 0.8),),
        )
        self.assertEqual(details, build_plan_measurement_detail_images(image))
        self.assertEqual(len(details), 4)

    def test_invalid_repeated_regions_are_skipped_and_actual_bounds_are_clamped(self):
        image = np.full((100, 200), 255, dtype=np.uint8)
        details = build_plan_measurement_detail_images(
            image, measurement_reference_image=image,
            focus_regions=(
                (0.8, 0.8, 0.2, 0.2),
                (float("nan"), 0, 1, 1),
                (-0.2, 0.1, 1.2, 0.8),
                (-0.2, 0.1, 1.2, 0.8),
            ),
        )
        self.assertEqual(len(details), 4)
        self.assertEqual(details[-1].bounds_normalized, (0.0, 0.1, 1.0, 0.8))
        self.assertEqual(sum(item.source_role == "measurement_reference" for item in details), 1)

    def test_focused_retry_keeps_two_raw_quadrants_with_most_detail_and_at_most_two_focuses(self):
        raw = np.full((400, 400), 255, dtype=np.uint8)
        # Fine printed-like detail appears only in the lower quadrants; the
        # rectified retry requests happen to be near the top and do not map it.
        raw[280:340:4, 40:100] = 0
        raw[280:340:4, 300:360] = 0
        reference = np.full((300, 600), 127, dtype=np.uint8)
        details = build_plan_measurement_detail_images(
            raw, measurement_reference_image=reference,
            focus_regions=((0, 0, 0.25, 0.25), (0.5, 0, 0.75, 0.25),
                           (0.2, 0.2, 0.5, 0.5)),
        )

        self.assertEqual(len(details), 4)
        self.assertEqual([item.source_role for item in details],
                         ["original", "original", "measurement_reference", "measurement_reference"])
        self.assertEqual(
            {item.bounds_normalized for item in details[:2]},
            {(0, 0.45, 0.55, 1), (0.45, 0.45, 1, 1)},
        )
        for detail in details[:2]:
            left, top, right, bottom = detail.bounds_normalized
            np.testing.assert_array_equal(
                _decoded(detail), raw[round(top * 400):round(bottom * 400),
                                     round(left * 400):round(right * 400)],
            )

    def test_invalid_focuses_fall_back_to_all_four_raw_quadrants(self):
        image = np.full((100, 200), 255, dtype=np.uint8)
        details = build_plan_measurement_detail_images(
            image, measurement_reference_image=image,
            focus_regions=((0.8, 0.8, 0.2, 0.2),),
        )
        self.assertEqual(details, build_plan_measurement_detail_images(image))

    def test_cancellation_interrupts_crop_preparation(self):
        with self.assertRaises(PlanMeasurementDetailCancelled):
            build_plan_measurement_detail_images(
                np.zeros((100, 100), dtype=np.uint8), cancellation_check=lambda: True,
            )


# ### Entry point ###
if __name__ == "__main__":
    unittest.main()
