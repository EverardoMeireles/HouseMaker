"""Per-plan pixel density must preserve metric dimensions without image resizing."""

# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ### Imports ###
import unittest

import numpy as np
from PySide6.QtCore import QPointF
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from housemaker.blueprint_canvas import BlueprintCanvas, WindowWallFrame
from housemaker.doorway_bridge import find_doorway_bridge_target
from housemaker.doorway_geometry import doorway_indices_on_removed_wall_edges
from housemaker.level_coordinates import (
    build_doorway_world_outline_positions,
    build_level_base_z_lookup,
    get_level_world_pivot,
    level_image_to_world_xy,
    level_world_to_image_xy,
)
from housemaker.models import DoorwayData, LevelData, VertexData
from housemaker.plan_metric_state import PlanMeasurementEvidence, PlanMetricCalibration

# ### Test fixtures ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


def _calibration(density: float) -> PlanMetricCalibration:
    return PlanMetricCalibration(
        estimated_meters_per_pixel=density,
        target_meters_per_pixel=density,
        normalization_scale=1.0,
        relative_residual=0.0,
        confidence="medium",
        source_image_size=(1000, 1000),
        pre_normalization_size=(1000, 1000),
        normalized_image_size=(1000, 1000),
        measurements=tuple(
            PlanMeasurementEvidence(
                label=f"span {y}",
                value_meters=500 * density,
                start_normalized=(0.1, y),
                end_normalized=(0.6, y),
                corrected_start_normalized=(0.1, y),
                corrected_end_normalized=(0.6, y),
                confidence="high",
            )
            for y in (0.2, 0.7)
        ),
    )


def _level(density: float = 0.01) -> LevelData:
    return LevelData(
        index=2,
        name="Ground",
        image_size_pixels=(1000.0, 1000.0),
        plan_metric_calibration=_calibration(density),
    )


# ### Coordinate regression tests ###
class MetricLevelCoordinateTests(unittest.TestCase):
    def test_different_image_densities_produce_matching_metric_lengths(self) -> None:
        for density, pixel_length in ((0.01, 500.0), (0.025, 200.0)):
            level = _level(density)
            origin = level_image_to_world_xy(level, 500.0, 500.0)
            endpoint = level_image_to_world_xy(level, 500.0 + pixel_length, 500.0)
            self.assertEqual(origin, (0.0, 0.0))
            self.assertEqual(endpoint, (5.0, 0.0))
            self.assertEqual(
                level_world_to_image_xy(level, *endpoint),
                (500.0 + pixel_length, 500.0),
            )

    def test_density_change_invalidates_cached_pivot(self) -> None:
        level = _level()
        level.vertex_data.add_vertex(100.0, 100.0)
        level.vertex_data.add_vertex(300.0, 300.0)
        self.assertEqual(get_level_world_pivot(level), (-3.0, 3.0))
        level.plan_metric_calibration = _calibration(0.03)
        self.assertEqual(get_level_world_pivot(level), (-9.0, 9.0))

    def test_manual_scale_and_offsets_remain_independent_of_density(self) -> None:
        level = _level()
        level.vertex_data.add_vertex(100.0, 100.0)
        level.vertex_data.add_vertex(300.0, 300.0)
        level.scale = 2.0
        level.offset_x_meters = 4.0
        level.offset_y_meters = -3.0
        self.assertEqual(level_image_to_world_xy(level, 300.0, 300.0), (3.0, -2.0))
        self.assertEqual(level_world_to_image_xy(level, 3.0, -2.0), (300.0, 300.0))

    def test_doorway_outline_is_metric_and_height_is_not_rescaled(self) -> None:
        level = _level(0.005)
        level.scale = 1.5
        doorway = DoorwayData(
            center_x=500.0,
            center_y=500.0,
            width_meters=1.0,
            height_meters=2.1,
            depth_meters=0.2,
            rotation_degrees=90.0,
        )
        points = np.asarray(
            build_doorway_world_outline_positions([level], level, doorway)
        )
        np.testing.assert_allclose(np.ptp(points, axis=0), (1.5, 0.3, 2.1))
        self.assertAlmostEqual(
            points[:, 2].min(), build_level_base_z_lookup([level])[2]
        )

    def test_bridge_width_uses_calibrated_density(self) -> None:
        vertices = VertexData()
        vertices.add_vertex(0.0, 0.0)
        vertices.add_vertex(100.0, 0.0)
        target = find_doorway_bridge_target(
            vertices, set(), (50.0, 0.0), 1.0, 5.0, meters_per_pixel=0.01
        )
        self.assertIsNotNone(target)
        self.assertAlmostEqual(target.width_meters, 1.0)

    def test_wall_removal_uses_calibrated_doorway_footprint(self) -> None:
        vertices = VertexData()
        first = vertices.add_vertex(0.0, 10.0)
        second = vertices.add_vertex(100.0, 10.0)
        edge = vertices.add_edge(first.id, second.id)
        doorway = DoorwayData(
            center_x=50.0,
            center_y=0.0,
            width_meters=1.0,
            height_meters=2.1,
            depth_meters=0.3,
            rotation_degrees=90.0,
        )
        self.assertEqual(
            doorway_indices_on_removed_wall_edges(
                vertices, [doorway], [edge], meters_per_pixel=0.01
            ),
            {0},
        )
        self.assertEqual(
            doorway_indices_on_removed_wall_edges(vertices, [doorway], [edge]), set()
        )


# ### Canvas regression tests ###
class MetricCanvasOpeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.level = _level()
        self.canvas = BlueprintCanvas()
        self.canvas.set_level_data(self.level.vertex_data, [], None, doorways=[])
        self.canvas.set_stair_context([], self.level)
        self.canvas.blueprint_image = QImage(1000, 1000, QImage.Format.Format_RGB32)

    def tearDown(self) -> None:
        self.canvas.close()
        self.canvas.deleteLater()
        _qt_application.processEvents()

    def _doorway(self) -> DoorwayData:
        return DoorwayData(
            center_x=500.0,
            center_y=500.0,
            width_meters=1.0,
            height_meters=2.1,
            depth_meters=0.4,
            rotation_degrees=90.0,
        )

    def test_doorway_outline_and_hit_testing_use_plan_density(self) -> None:
        doorway = self._doorway()
        points = np.asarray(self.canvas._get_doorway_corners(doorway))
        np.testing.assert_allclose(np.ptp(points, axis=0), (100.0, 40.0))
        left_handle = self.canvas._get_doorway_width_handle_image_position(doorway, 1.0)
        self.assertAlmostEqual(left_handle[0], 450.0)

    def test_doorway_resize_converts_pixels_to_calibrated_meters(self) -> None:
        doorway = self._doorway()
        vertices = self.level.vertex_data
        first = vertices.add_vertex(100.0, 500.0)
        second = vertices.add_vertex(900.0, 500.0)
        edge = vertices.add_edge(first.id, second.id)
        self.canvas.doorways.append(doorway)
        self.canvas.drag_doorway_index = 0
        self.canvas.doorway_drag_initial_doorway = doorway
        self.canvas.doorway_drag_width_side_sign = 1.0
        self.canvas.doorway_drag_wall_edge = edge
        self.canvas._resize_dragged_doorway_width(QPointF(400.0, 500.0))
        resized = self.canvas.doorways[0]
        self.assertAlmostEqual(resized.width_meters, 1.5)
        self.assertAlmostEqual(resized.center_x, 475.0)
        self.assertEqual(resized.height_meters, 2.1)

    def test_window_minimum_width_uses_calibrated_wall_length(self) -> None:
        frame = WindowWallFrame((0.0, 0.0), (500.0, 0.0), height_meters=2.5)
        ratio = self.canvas._get_window_minimum_width_ratio(frame)
        self.level.plan_metric_calibration = _calibration(0.02)
        self.assertAlmostEqual(
            self.canvas._get_window_minimum_width_ratio(frame), ratio / 2.0
        )


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
