# ### Environment setup ###
from __future__ import annotations

import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtWidgets import QApplication

from housemaker.camera_indicators import (
    DEFAULT_CAMERA_INDICATOR_PERCENTAGES,
    INDICATOR_BAR_FILL_SELECTED_COLOR,
    INDICATOR_COLOR,
    INDICATOR_LABEL_COLOR,
    INDICATOR_SELECTED_COLOR,
    INDICATOR_SELECTED_LABEL_COLOR,
    PROJECTION_CAMERA_INDICATOR_LABELS,
    build_projection_camera_indicator_geometries,
    create_projection_camera_indicator_items,
    update_projection_camera_indicator_items,
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


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Tests ###
class ProjectionCameraIndicatorGeometryTests(unittest.TestCase):
    def test_builds_six_cameras_outside_bounds_aimed_at_object(self) -> None:
        geometries = build_projection_camera_indicator_geometries(MODEL_BOUNDS)
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
                self.assertEqual(geometry.line_positions.ndim, 2)
                self.assertEqual(geometry.line_positions.shape[1], 3)
                self.assertEqual(len(geometry.line_positions) % 2, 0)
                self.assertEqual(
                    geometry.selection_line_positions.shape[1],
                    3,
                )
                self.assertLess(
                    len(geometry.selection_line_positions),
                    len(geometry.line_positions),
                )
                self.assertFalse(
                    np.allclose(
                        geometry.percentage_bar_start,
                        geometry.percentage_bar_end,
                    )
                )

    def test_items_use_axis_labels_and_indicator_style(self) -> None:
        geometries = build_projection_camera_indicator_geometries(MODEL_BOUNDS)
        items = create_projection_camera_indicator_items(MODEL_BOUNDS)

        self.assertEqual(tuple(items), ALL_CAMERA_IDS)
        for camera_id in ALL_CAMERA_IDS:
            with self.subTest(camera_id=camera_id):
                self.assertEqual(len(items[camera_id]), 4)
                line_item, track_item, fill_item, label_item = items[camera_id]
                np.testing.assert_allclose(
                    line_item.pos,
                    geometries[camera_id].line_positions,
                )
                self.assertEqual(line_item.mode, "lines")
                self.assertEqual(tuple(line_item.color), INDICATOR_COLOR)
                self.assertEqual(track_item.mode, "lines")
                self.assertEqual(fill_item.mode, "lines")
                self.assertLess(
                    fill_item.depthValue(),
                    track_item.depthValue(),
                )
                percentage_index = ALL_CAMERA_IDS.index(camera_id)
                expected_percentage = (
                    DEFAULT_CAMERA_INDICATOR_PERCENTAGES[percentage_index]
                )
                self.assertEqual(
                    label_item.text,
                    f"{PROJECTION_CAMERA_INDICATOR_LABELS[camera_id]} "
                    f"{expected_percentage}%",
                )
                np.testing.assert_allclose(
                    label_item.pos,
                    geometries[camera_id].label_position,
                )
                self.assertEqual(label_item.color, INDICATOR_LABEL_COLOR)
                self.assertTrue(label_item.font.bold())

    def test_percentage_bars_and_highlight_update_in_place(self) -> None:
        geometries = build_projection_camera_indicator_geometries(MODEL_BOUNDS)
        percentages = (40, 20, 10, 10, 10, 10)
        items = create_projection_camera_indicator_items(
            MODEL_BOUNDS,
            percentages,
            selected_camera_id=CAMERA_ID_POS_X,
        )

        line_item, _track_item, fill_item, label_item = items[CAMERA_ID_POS_X]
        self.assertEqual(tuple(line_item.color), INDICATOR_SELECTED_COLOR)
        self.assertEqual(tuple(fill_item.color), INDICATOR_BAR_FILL_SELECTED_COLOR)
        self.assertEqual(label_item.color, INDICATOR_SELECTED_LABEL_COLOR)
        self.assertEqual(label_item.text, "+X 40%")
        start = geometries[CAMERA_ID_POS_X].percentage_bar_start
        end = geometries[CAMERA_ID_POS_X].percentage_bar_end
        np.testing.assert_allclose(fill_item.pos[0], start)
        np.testing.assert_allclose(fill_item.pos[1], start + (end - start) * 0.4)

        updated = (39, 21, 10, 10, 10, 10)
        update_projection_camera_indicator_items(
            items,
            geometries,
            updated,
            selected_camera_id=CAMERA_ID_NEG_X,
        )

        self.assertEqual(tuple(line_item.color), INDICATOR_COLOR)
        self.assertEqual(label_item.color, INDICATOR_LABEL_COLOR)
        self.assertEqual(label_item.text, "+X 39%")
        selected_line, _track, selected_fill, selected_label = items[
            CAMERA_ID_NEG_X
        ]
        self.assertEqual(tuple(selected_line.color), INDICATOR_SELECTED_COLOR)
        self.assertEqual(
            tuple(selected_fill.color),
            INDICATOR_BAR_FILL_SELECTED_COLOR,
        )
        self.assertEqual(selected_label.text, "-X 21%")

    def test_percentage_display_rejects_unsafe_ui_values(self) -> None:
        invalid_percentages = (
            (0, 20, 20, 20, 20, 20),
            (96, 1, 1, 1, 1, 0),
            (20, 20, 20, 20, 20, 1),
        )
        for percentages in invalid_percentages:
            with self.subTest(percentages=percentages):
                with self.assertRaises(ValueError):
                    create_projection_camera_indicator_items(
                        MODEL_BOUNDS,
                        percentages,
                    )

    def test_rejects_non_finite_or_reversed_bounds(self) -> None:
        invalid_bounds = (
            np.array(((0.0, 0.0, 0.0), (-1.0, 1.0, 1.0))),
            np.array(((0.0, 0.0, 0.0), (1.0, np.nan, 1.0))),
            np.zeros((3, 3)),
        )
        for bounds in invalid_bounds:
            with self.subTest(bounds=bounds):
                with self.assertRaisesRegex(ValueError, "two finite 3D corners"):
                    build_projection_camera_indicator_geometries(bounds)


if __name__ == "__main__":
    unittest.main()
