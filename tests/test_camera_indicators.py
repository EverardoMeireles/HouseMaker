# ### Imports ###
from __future__ import annotations

import unittest

import numpy as np

from housemaker.camera_indicators import (
    CAMERA_INDICATOR_LABELS,
    INDICATOR_COLOR,
    INDICATOR_LABEL_COLOR,
    build_unused_face_camera_indicator_geometries,
    create_unused_face_camera_indicator_items,
    normalize_unused_face_camera_ids,
)
from housemaker.unused_face_removal import (
    ALL_CAMERA_IDS,
    CAMERA_ID_BOTTOM,
    CAMERA_ID_NEG_X,
    CAMERA_ID_NEG_Y,
    CAMERA_ID_POS_X,
    CAMERA_ID_POS_Y,
    CAMERA_ID_TOP,
)


# ### Constants ###
MODEL_BOUNDS = np.array(
    (
        (-2.0, -1.0, 0.5),
        (4.0, 3.0, 6.5),
    ),
    dtype=float,
)
OUTWARD_BY_CAMERA_ID = {
    CAMERA_ID_POS_X: np.array((1.0, 0.0, 0.0)),
    CAMERA_ID_NEG_X: np.array((-1.0, 0.0, 0.0)),
    CAMERA_ID_POS_Y: np.array((0.0, 1.0, 0.0)),
    CAMERA_ID_NEG_Y: np.array((0.0, -1.0, 0.0)),
    CAMERA_ID_TOP: np.array((0.0, 0.0, 1.0)),
    CAMERA_ID_BOTTOM: np.array((0.0, 0.0, -1.0)),
}


# ### Tests ###
class CameraIndicatorGeometryTests(unittest.TestCase):
    def test_builds_six_cameras_outside_bounds_aimed_at_the_object(self) -> None:
        geometries = build_unused_face_camera_indicator_geometries(MODEL_BOUNDS)
        center = MODEL_BOUNDS.mean(axis=0)

        self.assertEqual(tuple(geometries), ALL_CAMERA_IDS)
        for camera_id, outward in OUTWARD_BY_CAMERA_ID.items():
            with self.subTest(camera_id=camera_id):
                geometry = geometries[camera_id]
                depth_axis = int(np.argmax(np.abs(outward)))
                outside_limit = (
                    MODEL_BOUNDS[1, depth_axis]
                    if outward[depth_axis] > 0.0
                    else MODEL_BOUNDS[0, depth_axis]
                )
                self.assertGreater(
                    (geometry.camera_position[depth_axis] - outside_limit)
                    * outward[depth_axis],
                    0.0,
                )
                self.assertLess(
                    float(
                        np.dot(
                            geometry.aim_endpoint - geometry.camera_position,
                            outward,
                        )
                    ),
                    0.0,
                )
                self.assertLess(
                    np.linalg.norm(geometry.aim_endpoint - center),
                    np.linalg.norm(geometry.camera_position - center),
                )
                self.assertTrue(np.all(np.isfinite(geometry.label_position)))
                self.assertGreater(
                    np.linalg.norm(
                        geometry.label_position - geometry.camera_position
                    ),
                    0.0,
                )
                self.assertLess(
                    np.linalg.norm(
                        geometry.label_position - geometry.camera_position
                    ),
                    float(np.max(MODEL_BOUNDS[1] - MODEL_BOUNDS[0])) * 0.25,
                )
                self.assertEqual(geometry.line_positions.ndim, 2)
                self.assertEqual(geometry.line_positions.shape[1], 3)
                self.assertEqual(len(geometry.line_positions) % 2, 0)
                self.assertTrue(
                    np.any(
                        np.all(
                            np.isclose(
                                geometry.line_positions,
                                geometry.aim_endpoint,
                            ),
                            axis=1,
                        )
                    )
                )

    def test_items_use_geometry_labels_and_indicator_style(self) -> None:
        geometries = build_unused_face_camera_indicator_geometries(MODEL_BOUNDS)
        items = create_unused_face_camera_indicator_items(MODEL_BOUNDS)

        self.assertEqual(tuple(items), ALL_CAMERA_IDS)
        for camera_id in ALL_CAMERA_IDS:
            with self.subTest(camera_id=camera_id):
                self.assertEqual(len(items[camera_id]), 2)
                item, label = items[camera_id]
                np.testing.assert_allclose(
                    item.pos,
                    geometries[camera_id].line_positions,
                )
                self.assertEqual(item.mode, "lines")
                self.assertEqual(tuple(item.color), INDICATOR_COLOR)
                self.assertEqual(label.text, CAMERA_INDICATOR_LABELS[camera_id])
                np.testing.assert_allclose(
                    label.pos,
                    geometries[camera_id].label_position,
                )
                self.assertEqual(label.color, INDICATOR_LABEL_COLOR)
                self.assertTrue(label.font.bold())

    def test_camera_id_normalization_is_canonical_and_allows_none(self) -> None:
        self.assertEqual(
            normalize_unused_face_camera_ids(
                (CAMERA_ID_BOTTOM, CAMERA_ID_POS_X, CAMERA_ID_BOTTOM)
            ),
            (CAMERA_ID_POS_X, CAMERA_ID_BOTTOM),
        )
        self.assertEqual(normalize_unused_face_camera_ids(()), ())

        with self.assertRaisesRegex(ValueError, "Unknown unused-face camera"):
            normalize_unused_face_camera_ids(("diagonal",))

    def test_rejects_non_finite_or_reversed_bounds(self) -> None:
        invalid_bounds = (
            np.array(((0.0, 0.0, 0.0), (-1.0, 1.0, 1.0))),
            np.array(((0.0, 0.0, 0.0), (1.0, np.nan, 1.0))),
            np.zeros((3, 3)),
        )
        for bounds in invalid_bounds:
            with self.subTest(bounds=bounds):
                with self.assertRaisesRegex(ValueError, "two finite 3D corners"):
                    build_unused_face_camera_indicator_geometries(bounds)


if __name__ == "__main__":
    unittest.main()
