# ### Environment setup ###
from __future__ import annotations

import os
import unittest
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QImage, QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.blueprint_canvas import BlueprintCanvas
from housemaker.camera_models import (
    CameraPose,
    InitialFirstPersonCamera,
)
from housemaker.level_coordinates import level_image_to_world_xy
from housemaker.main import (
    DEFAULT_FIRST_PERSON_CAMERA_HEIGHT_METERS,
    BlueprintWorkspace,
)
from housemaker.models import GROUND_LEVEL_INDEX, LevelData


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_level(level_index: int = GROUND_LEVEL_INDEX) -> LevelData:
    level = LevelData(
        index=level_index,
        name=f"Level {level_index}",
        image_size_pixels=(100.0, 100.0),
    )
    level.vertex_data.add_vertex(10.0, 10.0)
    level.vertex_data.add_vertex(90.0, 90.0)
    return level


def _build_canvas(
    level: LevelData,
    camera: InitialFirstPersonCamera | None = None,
) -> BlueprintCanvas:
    canvas = BlueprintCanvas()
    canvas.resize(640, 520)
    canvas.set_level_data(
        vertex_data=level.vertex_data,
        rooms=level.rooms,
        image_path=None,
        image_scale=level.image_scale,
        image_offset_x=level.image_offset_x,
        image_offset_y=level.image_offset_y,
        floor_contour_vertex_ids=level.floor_contour_vertex_ids,
        doorways=level.doorways,
    )
    canvas.blueprint_image = QImage(100, 100, QImage.Format.Format_RGB32)
    canvas.blueprint_image.fill(Qt.GlobalColor.white)
    canvas.set_initial_first_person_camera_context(camera, level)
    canvas.show()
    _qt_application.processEvents()
    return canvas


def _image_position(canvas: BlueprintCanvas, x: float, y: float) -> QPointF:
    return canvas._image_to_widget(x, y)


def _send_drag_move(canvas: BlueprintCanvas, widget_position: QPointF) -> None:
    global_position = QPointF(canvas.mapToGlobal(widget_position.toPoint()))
    event = QMouseEvent(
        QEvent.Type.MouseMove,
        widget_position,
        global_position,
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(canvas, event)


def _camera_for_image_point(
    level: LevelData,
    image_x: float = 50.0,
    image_y: float = 50.0,
) -> InitialFirstPersonCamera:
    world_x, world_y = level_image_to_world_xy(level, image_x, image_y)
    return InitialFirstPersonCamera(
        level_index=level.index,
        pose=CameraPose(
            x=world_x,
            y=world_y,
            z=1.65,
            yaw_degrees=18.0,
            pitch_degrees=-4.0,
            roll_degrees=2.0,
            fov_degrees=74.0,
        ),
    )


# ### Canvas behavior tests ###
class FirstPersonCameraCanvasTests(unittest.TestCase):
    def setUp(self) -> None:
        self.widgets: list[object] = []

    def tearDown(self) -> None:
        for widget in reversed(self.widgets):
            shutdown = getattr(widget, "shutdown", None)
            if callable(shutdown):
                shutdown()
            generation_workspace = getattr(widget, "generation", None)
            if generation_workspace is not None:
                generation_workspace.shutdown()
            close = getattr(widget, "close", None)
            if callable(close):
                close()
        _qt_application.processEvents()

    def _track_widget(self, widget: object) -> object:
        self.widgets.append(widget)
        return widget

    def test_one_shot_placement_creates_then_replaces_singleton_without_vertices(
        self,
    ) -> None:
        level = _build_level()
        canvas = self._track_widget(_build_canvas(level))
        self.assertIsInstance(canvas, BlueprintCanvas)
        original_vertices = tuple(canvas.vertex_data.vertices)
        emitted_cameras: list[object] = []
        canvas.first_person_camera_changed.connect(emitted_cameras.append)

        canvas.start_first_person_camera_placement(1.65)
        first_position = _image_position(canvas, 25.0, 35.0)
        QTest.mouseClick(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=first_position.toPoint(),
        )
        first_camera = canvas.initial_first_person_camera

        self.assertIsNotNone(first_camera)
        self.assertIsNone(canvas.pending_first_person_camera_z)
        self.assertEqual(tuple(canvas.vertex_data.vertices), original_vertices)
        self.assertEqual(emitted_cameras, [first_camera])

        canvas.start_first_person_camera_placement(2.1)
        second_position = _image_position(canvas, 75.0, 65.0)
        QTest.mouseClick(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=second_position.toPoint(),
        )
        second_camera = canvas.initial_first_person_camera

        self.assertIsNotNone(second_camera)
        self.assertNotEqual(second_camera, first_camera)
        self.assertEqual(second_camera.level_index, level.index)  # type: ignore[union-attr]
        self.assertAlmostEqual(second_camera.pose.z, 2.1)  # type: ignore[union-attr]
        self.assertEqual(tuple(canvas.vertex_data.vertices), original_vertices)
        self.assertEqual(emitted_cameras, [first_camera, second_camera])

    def test_escape_cancels_replacement_and_preserves_existing_camera(self) -> None:
        level = _build_level()
        existing_camera = _camera_for_image_point(level)
        canvas = self._track_widget(_build_canvas(level, existing_camera))
        self.assertIsInstance(canvas, BlueprintCanvas)
        emitted_cameras: list[object] = []
        canvas.first_person_camera_changed.connect(emitted_cameras.append)

        canvas.start_first_person_camera_placement(2.4)
        QTest.keyClick(canvas, Qt.Key.Key_Escape)
        _qt_application.processEvents()

        self.assertIsNone(canvas.pending_first_person_camera_z)
        self.assertEqual(canvas.initial_first_person_camera, existing_camera)
        self.assertEqual(emitted_cameras, [])

    def test_repositioning_camera_preserves_its_light_intensity(self) -> None:
        level = _build_level()
        existing_camera = replace(
            _camera_for_image_point(level),
            light_intensity=0.45,
        )
        canvas = self._track_widget(_build_canvas(level, existing_camera))
        self.assertIsInstance(canvas, BlueprintCanvas)

        canvas.start_first_person_camera_placement(2.2)
        QTest.mouseClick(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=_image_position(canvas, 75.0, 30.0).toPoint(),
        )
        repositioned_camera = canvas.initial_first_person_camera

        self.assertIsNotNone(repositioned_camera)
        self.assertEqual(
            repositioned_camera.light_intensity,  # type: ignore[union-attr]
            existing_camera.light_intensity,
        )
        self.assertEqual(  # type: ignore[union-attr]
            repositioned_camera.pose.pitch_degrees,
            existing_camera.pose.pitch_degrees,
        )
        self.assertEqual(  # type: ignore[union-attr]
            repositioned_camera.pose.roll_degrees,
            existing_camera.pose.roll_degrees,
        )
        self.assertEqual(  # type: ignore[union-attr]
            repositioned_camera.pose.fov_degrees,
            existing_camera.pose.fov_degrees,
        )

    def test_body_drag_changes_only_world_xy(self) -> None:
        level = _build_level()
        existing_camera = _camera_for_image_point(level)
        canvas = self._track_widget(_build_canvas(level, existing_camera))
        self.assertIsInstance(canvas, BlueprintCanvas)
        camera_points = canvas._get_initial_first_person_camera_widget_points()
        self.assertIsNotNone(camera_points)
        center, _direction_tip = camera_points  # type: ignore[misc]
        target = _image_position(canvas, 70.0, 72.0)
        target_image = canvas._widget_to_image_clamped(target)
        expected_x, expected_y = level_image_to_world_xy(
            level,
            target_image.x(),
            target_image.y(),
        )

        QTest.mousePress(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=center.toPoint(),
        )
        _send_drag_move(canvas, target)
        QTest.mouseRelease(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=target.toPoint(),
        )
        moved_camera = canvas.initial_first_person_camera

        self.assertIsNotNone(moved_camera)
        self.assertAlmostEqual(moved_camera.pose.x, expected_x)  # type: ignore[union-attr]
        self.assertAlmostEqual(moved_camera.pose.y, expected_y)  # type: ignore[union-attr]
        self.assertEqual(moved_camera.pose.z, existing_camera.pose.z)  # type: ignore[union-attr]
        self.assertEqual(  # type: ignore[union-attr]
            moved_camera.pose.yaw_degrees,
            existing_camera.pose.yaw_degrees,
        )
        self.assertEqual(  # type: ignore[union-attr]
            moved_camera.pose.pitch_degrees,
            existing_camera.pose.pitch_degrees,
        )
        self.assertEqual(  # type: ignore[union-attr]
            moved_camera.pose.roll_degrees,
            existing_camera.pose.roll_degrees,
        )
        self.assertEqual(  # type: ignore[union-attr]
            moved_camera.pose.fov_degrees,
            existing_camera.pose.fov_degrees,
        )

    def test_direction_handle_drag_changes_only_yaw(self) -> None:
        level = _build_level()
        existing_camera = _camera_for_image_point(level)
        canvas = self._track_widget(_build_canvas(level, existing_camera))
        self.assertIsInstance(canvas, BlueprintCanvas)
        camera_points = canvas._get_initial_first_person_camera_widget_points()
        self.assertIsNotNone(camera_points)
        center, direction_tip = camera_points  # type: ignore[misc]
        target = center + QPointF(0.0, -90.0)

        QTest.mousePress(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=direction_tip.toPoint(),
        )
        _send_drag_move(canvas, target)
        QTest.mouseRelease(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=target.toPoint(),
        )
        rotated_camera = canvas.initial_first_person_camera

        self.assertIsNotNone(rotated_camera)
        self.assertAlmostEqual(rotated_camera.pose.yaw_degrees, 90.0)  # type: ignore[union-attr]
        self.assertEqual(rotated_camera.pose.x, existing_camera.pose.x)  # type: ignore[union-attr]
        self.assertEqual(rotated_camera.pose.y, existing_camera.pose.y)  # type: ignore[union-attr]
        self.assertEqual(rotated_camera.pose.z, existing_camera.pose.z)  # type: ignore[union-attr]
        self.assertEqual(  # type: ignore[union-attr]
            rotated_camera.pose.pitch_degrees,
            existing_camera.pose.pitch_degrees,
        )
        self.assertEqual(  # type: ignore[union-attr]
            rotated_camera.pose.roll_degrees,
            existing_camera.pose.roll_degrees,
        )
        self.assertEqual(  # type: ignore[union-attr]
            rotated_camera.pose.fov_degrees,
            existing_camera.pose.fov_degrees,
        )

    def test_z_api_changes_only_z(self) -> None:
        level = _build_level()
        existing_camera = _camera_for_image_point(level)
        canvas = self._track_widget(_build_canvas(level, existing_camera))
        self.assertIsInstance(canvas, BlueprintCanvas)

        canvas.update_first_person_camera_z(2.35)
        updated_camera = canvas.initial_first_person_camera

        self.assertIsNotNone(updated_camera)
        self.assertEqual(updated_camera.pose, replace(existing_camera.pose, z=2.35))  # type: ignore[union-attr]
        self.assertEqual(updated_camera.level_index, existing_camera.level_index)  # type: ignore[union-attr]

    def test_light_intensity_api_changes_only_light_intensity(self) -> None:
        level = _build_level()
        existing_camera = replace(
            _camera_for_image_point(level),
            light_intensity=0.4,
        )
        canvas = self._track_widget(_build_canvas(level, existing_camera))
        self.assertIsInstance(canvas, BlueprintCanvas)

        canvas.update_first_person_camera_light_intensity(1.35)
        updated_camera = canvas.initial_first_person_camera

        self.assertEqual(
            updated_camera,
            replace(existing_camera, light_intensity=1.35),
        )

    def test_light_intensity_drag_creates_one_undo_step(self) -> None:
        level = _build_level()
        existing_camera = replace(
            _camera_for_image_point(level),
            light_intensity=0.4,
        )
        canvas = self._track_widget(_build_canvas(level, existing_camera))
        self.assertIsInstance(canvas, BlueprintCanvas)

        canvas.begin_first_person_camera_light_intensity_adjustment()
        canvas.update_first_person_camera_light_intensity(1.1)
        canvas.update_first_person_camera_light_intensity(0.65)
        canvas.end_first_person_camera_light_intensity_adjustment()

        self.assertEqual(len(canvas.undo_stack), 1)
        self.assertAlmostEqual(
            canvas.initial_first_person_camera.light_intensity,  # type: ignore[union-attr]
            0.65,
        )
        canvas.undo_last_step()
        self.assertEqual(canvas.initial_first_person_camera, existing_camera)

    def test_camera_is_visible_only_in_its_owner_level_context(self) -> None:
        owner_level = _build_level(GROUND_LEVEL_INDEX)
        other_level = _build_level(GROUND_LEVEL_INDEX + 1)
        camera = _camera_for_image_point(owner_level)
        canvas = self._track_widget(_build_canvas(owner_level, camera))
        self.assertIsInstance(canvas, BlueprintCanvas)

        self.assertTrue(canvas._initial_first_person_camera_is_visible())
        self.assertIsNotNone(
            canvas._get_initial_first_person_camera_widget_points()
        )

        canvas.set_initial_first_person_camera_context(camera, other_level)
        self.assertFalse(canvas._initial_first_person_camera_is_visible())
        self.assertIsNone(canvas._get_initial_first_person_camera_widget_points())

        canvas.set_initial_first_person_camera_context(camera, owner_level)
        self.assertTrue(canvas._initial_first_person_camera_is_visible())

    def test_external_alignment_rebases_camera_in_existing_undo_history(
        self,
    ) -> None:
        level = _build_level()
        original_camera = _camera_for_image_point(level)
        canvas = self._track_widget(_build_canvas(level, original_camera))
        self.assertIsInstance(canvas, BlueprintCanvas)
        canvas._push_undo_state()
        first_vertex = level.vertex_data.vertices[0]
        canvas.vertex_data.move_vertex(
            first_vertex.id,
            first_vertex.x + 5.0,
            first_vertex.y,
        )
        aligned_camera = replace(
            original_camera,
            pose=replace(
                original_camera.pose,
                x=original_camera.pose.x + 1.0,
                yaw_degrees=60.0,
            ),
        )

        canvas.set_initial_first_person_camera_context(aligned_camera, level)
        canvas.undo_last_step()

        self.assertEqual(canvas.initial_first_person_camera, aligned_camera)

    def test_blueprint_workspace_button_and_camera_controls_stay_synchronized(
        self,
    ) -> None:
        workspace = self._track_widget(BlueprintWorkspace())
        self.assertIsInstance(workspace, BlueprintWorkspace)
        workspace.resize(1500, 900)
        level = workspace.current_level
        level.image_size_pixels = (100.0, 100.0)
        workspace.canvas.blueprint_image = QImage(
            100,
            100,
            QImage.Format.Format_RGB32,
        )
        workspace.canvas.blueprint_image.fill(Qt.GlobalColor.white)
        workspace.canvas.set_initial_first_person_camera_context(None, level)
        workspace.show()
        _qt_application.processEvents()

        self.assertEqual(
            workspace.set_first_person_camera_button.text(),
            "Set first person camera",
        )
        self.assertFalse(workspace.first_person_camera_z_spinbox.isEnabled())
        self.assertFalse(workspace.first_person_camera_light_slider.isEnabled())
        self.assertFalse(workspace.clear_first_person_camera_button.isEnabled())
        self.assertEqual(workspace.first_person_camera_light_slider.minimum(), 0)
        self.assertEqual(workspace.first_person_camera_light_slider.maximum(), 200)
        self.assertEqual(workspace.first_person_camera_light_slider.value(), 100)

        QTest.mouseClick(
            workspace.set_first_person_camera_button,
            Qt.MouseButton.LeftButton,
        )
        self.assertIs(
            workspace.workspace_tabs.currentWidget(),
            workspace.canvas_viewer_workspace,
        )
        self.assertAlmostEqual(
            workspace.canvas.pending_first_person_camera_z,
            DEFAULT_FIRST_PERSON_CAMERA_HEIGHT_METERS,
        )

        QTest.mouseClick(
            workspace.canvas,
            Qt.MouseButton.LeftButton,
            pos=_image_position(workspace.canvas, 60.0, 40.0).toPoint(),
        )
        _qt_application.processEvents()
        placed_camera = workspace.initial_first_person_camera

        self.assertIsNotNone(placed_camera)
        self.assertTrue(workspace.first_person_camera_z_spinbox.isEnabled())
        self.assertTrue(workspace.first_person_camera_light_slider.isEnabled())
        self.assertTrue(workspace.clear_first_person_camera_button.isEnabled())
        self.assertIn("Camera: L2 Ground", workspace.first_person_camera_status_label.text())
        self.assertEqual(workspace.first_person_camera_light_slider.value(), 100)

        workspace.first_person_camera_z_spinbox.setValue(2.2)
        _qt_application.processEvents()
        self.assertAlmostEqual(
            workspace.initial_first_person_camera.pose.z,  # type: ignore[union-attr]
            2.2,
        )

        workspace.first_person_camera_light_slider.setValue(135)
        _qt_application.processEvents()
        self.assertAlmostEqual(
            workspace.initial_first_person_camera.light_intensity,  # type: ignore[union-attr]
            1.35,
        )
        self.assertEqual(
            workspace.first_person_camera_light_value_label.text(),
            "135%",
        )

        QTest.mouseClick(
            workspace.clear_first_person_camera_button,
            Qt.MouseButton.LeftButton,
        )
        _qt_application.processEvents()
        self.assertIsNone(workspace.initial_first_person_camera)
        self.assertFalse(workspace.first_person_camera_z_spinbox.isEnabled())
        self.assertFalse(workspace.first_person_camera_light_slider.isEnabled())
        self.assertFalse(workspace.clear_first_person_camera_button.isEnabled())
        self.assertEqual(workspace.first_person_camera_light_slider.value(), 100)


# ### Test runner ###
if __name__ == "__main__":
    unittest.main()
