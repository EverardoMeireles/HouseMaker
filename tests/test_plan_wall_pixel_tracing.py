# ### Imports ###
from __future__ import annotations

import unittest

import cv2
import numpy as np

from housemaker.plan_wall_pixel_tracing import (
    PixelWallTracingCancelled,
    trace_plan_ink_paths,
)


# ### Pixel tracing regression tests ###
class PlanWallPixelTracingTests(unittest.TestCase):
    def test_traces_thin_stepped_wall_without_a_line_detector(self) -> None:
        ink = np.zeros((130, 220), dtype=np.uint8)
        points = ((10, 110), (10, 70), (70, 70), (70, 15), (200, 15))
        cv2.polylines(ink, [np.asarray(points, dtype=np.int32)], False, 1, 1)

        paths = trace_plan_ink_paths(ink.astype(bool))

        self.assertEqual(len(paths), 1)
        self.assertIn(paths[0], (points, points[::-1]))

    def test_restores_square_corners_in_thick_strokes(self) -> None:
        ink = np.zeros((130, 220), dtype=np.uint8)
        points = np.array(((10, 110), (10, 70), (70, 70), (70, 15), (200, 15)))
        cv2.polylines(ink, [points.astype(np.int32)], False, 1, 7)

        paths = trace_plan_ink_paths(ink.astype(bool))

        self.assertEqual(len(paths), 1)
        self.assertEqual(set(paths[0][1:-1]), {(10, 70), (70, 70), (70, 15)})

    def test_keeps_white_gap_and_parallel_strokes_disconnected(self) -> None:
        ink = np.zeros((100, 180), dtype=np.uint8)
        cv2.line(ink, (10, 40), (70, 40), 1, 1)
        cv2.line(ink, (80, 40), (160, 40), 1, 1)
        cv2.line(ink, (10, 44), (160, 44), 1, 1)

        paths = trace_plan_ink_paths(ink.astype(bool))

        self.assertEqual(len(paths), 3)
        self.assertEqual(set(paths), {
            ((10, 40), (70, 40)),
            ((80, 40), (160, 40)),
            ((10, 44), (160, 44)),
        })

    def test_preserves_closed_loop_and_its_right_angles(self) -> None:
        ink = np.zeros((90, 180), dtype=np.uint8)
        cv2.rectangle(ink, (10, 10), (160, 70), 1, 3)

        paths = trace_plan_ink_paths(ink.astype(bool))

        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0][0], paths[0][-1])
        self.assertEqual(set(paths[0]), {(10, 10), (160, 10), (160, 70), (10, 70)})

    def test_traces_strokes_touching_image_boundary(self) -> None:
        ink = np.zeros((50, 50), dtype=bool)
        ink[0, :] = True

        self.assertEqual(trace_plan_ink_paths(ink), (((0, 0), (49, 0)),))

    def test_branches_terminate_at_the_same_junction(self) -> None:
        ink = np.zeros((90, 120), dtype=np.uint8)
        cv2.line(ink, (10, 40), (100, 40), 1, 1)
        cv2.line(ink, (60, 40), (60, 80), 1, 1)

        paths = trace_plan_ink_paths(ink.astype(bool))

        self.assertEqual(len(paths), 3)
        self.assertTrue(all((60, 40) in path for path in paths))

    def test_results_are_deterministic_and_input_is_unchanged(self) -> None:
        ink = np.zeros((90, 120), dtype=np.uint8)
        cv2.rectangle(ink, (10, 10), (80, 70), 1, 3)
        before = ink.copy()

        self.assertEqual(trace_plan_ink_paths(ink), trace_plan_ink_paths(ink))
        np.testing.assert_array_equal(ink, before)


# ### Empty input and cancellation tests ###
class PlanWallPixelTracingValidationTests(unittest.TestCase):
    def test_empty_images_have_no_paths(self) -> None:
        self.assertEqual(trace_plan_ink_paths(np.zeros((90, 120), dtype=bool)), ())
        self.assertEqual(trace_plan_ink_paths(np.zeros((0, 0), dtype=bool)), ())

    def test_invalid_dimensions_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            trace_plan_ink_paths(np.zeros((10, 10, 3), dtype=bool))

    def test_cancellation_is_checked_during_skeletonization(self) -> None:
        calls = 0

        def cancel_after_start() -> bool:
            nonlocal calls
            calls += 1
            return calls >= 2

        with self.assertRaises(PixelWallTracingCancelled):
            trace_plan_ink_paths(np.ones((100, 100), dtype=bool), cancel_after_start)
        self.assertEqual(calls, 2)


# ### Standalone test runner ###
if __name__ == "__main__":
    unittest.main()
