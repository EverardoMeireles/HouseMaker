# ### Imports ###
from __future__ import annotations

import unittest

import cv2
import numpy as np

from housemaker.plan_wall_continuity import make_doorway_walls_continuous


# ### Fixture helpers ###
def _blank_plan() -> np.ndarray:
    return np.full((140, 180), 255, dtype=np.uint8)


def _draw_horizontal_boundary_pair(image: np.ndarray) -> None:
    for y in (48, 60):
        cv2.line(image, (12, y), (68, y), 0, 2)
        cv2.line(image, (91, y), (168, y), 0, 2)


def _draw_vertical_boundary_pair(image: np.ndarray) -> None:
    for x in (72, 84):
        cv2.line(image, (x, 10), (x, 58), 0, 2)
        cv2.line(image, (x, 81), (x, 130), 0, 2)


# ### Continuity tests ###
class PlanWallContinuityTests(unittest.TestCase):
    def test_connects_both_boundaries_of_a_horizontal_doorway(self) -> None:
        image = _blank_plan()
        _draw_horizontal_boundary_pair(image)

        result = make_doorway_walls_continuous(image)

        self.assertTrue(np.all(result[47:50, 69:91] == 0))
        self.assertTrue(np.all(result[59:62, 69:91] == 0))

    def test_connects_both_boundaries_of_a_vertical_doorway(self) -> None:
        image = _blank_plan()
        _draw_vertical_boundary_pair(image)

        result = make_doorway_walls_continuous(image)

        self.assertTrue(np.all(result[59:81, 71:74] == 0))
        self.assertTrue(np.all(result[59:81, 83:86] == 0))

    def test_leaves_an_unpaired_gap_unchanged(self) -> None:
        image = _blank_plan()
        cv2.line(image, (12, 48), (68, 48), 0, 2)
        cv2.line(image, (91, 48), (168, 48), 0, 2)
        cv2.line(image, (12, 60), (168, 60), 0, 2)

        result = make_doorway_walls_continuous(image)

        np.testing.assert_array_equal(result, image)

    def test_does_not_modify_the_input_array(self) -> None:
        image = _blank_plan()
        _draw_horizontal_boundary_pair(image)
        original = image.copy()

        result = make_doorway_walls_continuous(image)

        np.testing.assert_array_equal(image, original)
        self.assertFalse(np.array_equal(result, original))


# ### Validation tests ###
class PlanWallContinuityValidationTests(unittest.TestCase):
    def test_rejects_non_array_input(self) -> None:
        with self.assertRaisesRegex(TypeError, "NumPy array"):
            make_doorway_walls_continuous([[0, 255]])  # type: ignore[arg-type]

    def test_rejects_non_uint8_input(self) -> None:
        image = np.full((20, 20), 255, dtype=np.float32)

        with self.assertRaisesRegex(TypeError, "uint8"):
            make_doorway_walls_continuous(image)

    def test_rejects_color_input(self) -> None:
        image = np.full((20, 20, 3), 255, dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            make_doorway_walls_continuous(image)

    def test_rejects_empty_input(self) -> None:
        image = np.empty((0, 20), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "empty"):
            make_doorway_walls_continuous(image)


if __name__ == "__main__":
    unittest.main()
