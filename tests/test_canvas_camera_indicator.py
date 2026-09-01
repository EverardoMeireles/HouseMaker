# ### Environment setup ###
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.blueprint_canvas import (
    CAMERA_INDICATOR_COLOR,
    CAMERA_INDICATOR_DIRECTION_LENGTH_SCREEN,
    BlueprintCanvas,
)
from housemaker.camera_models import CameraPose
from housemaker.main import BlueprintWorkspace
from housemaker.models import LevelData


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Blueprint Canvas tests ###
class BlueprintCanvasCameraIndicatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas = BlueprintCanvas()
        self.canvas.resize(640, 480)
        self.canvas.blueprint_image = QImage(
            200,
            100,
            QImage.Format.Format_RGB32,
        )
        self.canvas.blueprint_image.fill(Qt.GlobalColor.white)
        self.canvas.level_context = LevelData(
            index=0,
            name="Ground",
            image_size_pixels=(200.0, 100.0),
        )

    def tearDown(self) -> None:
        self.canvas.close()
        self.canvas.deleteLater()
        _qt_application.processEvents()

    def test_indicator_maps_world_position_and_cardinal_yaws(self) -> None:
        expected_center = self.canvas._image_to_widget(100.0, 50.0)
        direction_expectations = (
            (0.0, 1, 0),
            (90.0, 0, -1),
            (180.0, -1, 0),
            (-90.0, 0, 1),
        )

        for yaw_degrees, expected_x_sign, expected_y_sign in direction_expectations:
            with self.subTest(yaw_degrees=yaw_degrees):
                self.canvas.set_camera_indicator_pose(
                    CameraPose(yaw_degrees=yaw_degrees)
                )
                geometry = self.canvas._get_camera_indicator_geometry()

                self.assertIsNotNone(geometry)
                assert geometry is not None
                center, tip = geometry
                direction_x = tip.x() - center.x()
                direction_y = tip.y() - center.y()
                self.assertAlmostEqual(center.x(), expected_center.x())
                self.assertAlmostEqual(center.y(), expected_center.y())
                self.assertAlmostEqual(
                    (direction_x**2 + direction_y**2) ** 0.5,
                    CAMERA_INDICATOR_DIRECTION_LENGTH_SCREEN,
                )
                if expected_x_sign == 0:
                    self.assertAlmostEqual(direction_x, 0.0)
                else:
                    self.assertGreater(direction_x * expected_x_sign, 0.0)
                if expected_y_sign == 0:
                    self.assertAlmostEqual(direction_y, 0.0)
                else:
                    self.assertGreater(direction_y * expected_y_sign, 0.0)

    def test_indicator_reprojects_with_canvas_zoom_and_pan(self) -> None:
        pose = CameraPose(x=0.25, y=-0.15, yaw_degrees=35.0)
        self.canvas.set_camera_indicator_pose(pose)
        initial_geometry = self.canvas._get_camera_indicator_geometry()
        assert initial_geometry is not None

        self.canvas.zoom_scale = 2.0
        self.canvas.view_offset.setX(45.0)
        zoomed_geometry = self.canvas._get_camera_indicator_geometry()
        assert zoomed_geometry is not None

        self.assertEqual(self.canvas.get_camera_indicator_pose(), pose)
        self.assertNotEqual(zoomed_geometry[0], initial_geometry[0])
        self.assertAlmostEqual(
            (zoomed_geometry[1] - zoomed_geometry[0]).manhattanLength(),
            (initial_geometry[1] - initial_geometry[0]).manhattanLength(),
        )

    def test_indicator_is_hidden_without_a_blueprint_or_level(self) -> None:
        self.canvas.set_camera_indicator_pose(CameraPose())
        self.canvas.blueprint_image = None
        self.assertIsNone(self.canvas._get_camera_indicator_geometry())

        self.canvas.blueprint_image = QImage(100, 100, QImage.Format.Format_RGB32)
        self.canvas.level_context = None
        self.assertIsNone(self.canvas._get_camera_indicator_geometry())

    def test_indicator_is_painted_over_the_blueprint(self) -> None:
        self.canvas.set_camera_indicator_pose(CameraPose(yaw_degrees=45.0))
        geometry = self.canvas._get_camera_indicator_geometry()
        assert geometry is not None
        center, _direction_tip = geometry
        rendered = QImage(
            self.canvas.size(),
            QImage.Format.Format_ARGB32,
        )
        rendered.fill(Qt.GlobalColor.transparent)

        self.canvas.render(rendered)

        center_color = QColor.fromRgba(
            rendered.pixel(round(center.x()), round(center.y()))
        )
        self.assertEqual(center_color.name(), CAMERA_INDICATOR_COLOR.name())


# ### Main-workspace integration tests ###
class CanvasCameraIndicatorIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(
                Path(self._temporary_directory.name) / "settings.json"
            )
        )
        self.workspace.resize(1400, 850)
        self.workspace.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.viewer.exit_first_person_mode()
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self._temporary_directory.cleanup()

    def test_live_first_person_movement_updates_canvas_pose(self) -> None:
        initial_pose = CameraPose(
            x=1.0,
            y=2.0,
            z=1.7,
            yaw_degrees=0.0,
        )
        self.workspace.viewer.set_first_person_camera_pose(initial_pose)

        self.assertEqual(
            self.workspace.canvas.get_camera_indicator_pose(),
            initial_pose,
        )

        self.workspace.viewer.enter_first_person_mode()
        QTest.keyPress(self.workspace.viewer.view, Qt.Key.Key_Z)
        self.workspace.viewer.view.step_first_person_movement(0.4)
        QTest.keyRelease(self.workspace.viewer.view, Qt.Key.Key_Z)
        moved_pose = self.workspace.viewer.get_first_person_camera_pose()

        self.assertGreater(moved_pose.x, initial_pose.x)
        self.assertEqual(
            self.workspace.canvas.get_camera_indicator_pose(),
            moved_pose,
        )


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
