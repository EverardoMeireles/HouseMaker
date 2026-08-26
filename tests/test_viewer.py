# ### Environment setup ###
from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, patch

import numpy as np
from OpenGL import GL
import pyqtgraph.opengl as gl
import trimesh

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QFocusEvent, QKeyEvent
from PySide6.QtWidgets import QApplication
from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.camera_models import CameraPose
from housemaker.glb import GeneratedModel
from housemaker.unused_face_removal import ALL_CAMERA_IDS
from housemaker.viewer import (
    NAVIGATION_MODE_FIRST_PERSON,
    NAVIGATION_MODE_ORBIT,
    GlbViewerWidget,
    SelectableGLViewWidget,
)


# ### Test doubles ###
class FakeMouseMoveEvent:
    def __init__(
        self,
        *,
        position: QPointF,
        buttons: Qt.MouseButton,
        modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
    ) -> None:
        self._position = position
        self._buttons = buttons
        self._modifiers = modifiers
        self.was_accepted = False

    def position(self) -> QPointF:
        return self._position

    def buttons(self) -> Qt.MouseButton:
        return self._buttons

    def modifiers(self) -> Qt.KeyboardModifier:
        return self._modifiers

    def accept(self) -> None:
        self.was_accepted = True


class FakeWheelEvent:
    def __init__(self, delta: int) -> None:
        self._delta = int(delta)
        self.was_accepted = False

    def angleDelta(self) -> QPoint:
        return QPoint(0, self._delta)

    def accept(self) -> None:
        self.was_accepted = True


class FakeMousePressEvent:
    def __init__(
        self,
        *,
        button: Qt.MouseButton,
        position: QPointF = QPointF(),
    ) -> None:
        self._button = button
        self._position = position
        self.was_accepted = False

    def button(self) -> Qt.MouseButton:
        return self._button

    def position(self) -> QPointF:
        return self._position

    def accept(self) -> None:
        self.was_accepted = True


# ### Fixture helpers ###
def _build_generated_model(*, textured: bool = False) -> GeneratedModel:
    if not textured:
        mesh = trimesh.creation.box()
        return GeneratedModel(
            mesh=mesh,
            scene=trimesh.Scene(mesh),
            glb_bytes=b"",
        )

    mesh = trimesh.Trimesh(
        vertices=np.array(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            ),
            dtype=float,
        ),
        faces=np.array(((0, 1, 2),), dtype=int),
        process=False,
    )
    texture = Image.fromarray(
        np.full((2, 2, 4), (220, 90, 40, 255), dtype=np.uint8),
        mode="RGBA",
    )
    mesh.visual = TextureVisuals(
        uv=np.array(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=float),
        material=PBRMaterial(baseColorTexture=texture),
    )
    return GeneratedModel(
        mesh=mesh,
        scene=trimesh.Scene(mesh),
        glb_bytes=b"",
    )


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Test constants ###
_EXPECTED_UNUSED_FACE_CAMERA_LABELS = {
    "pos_x": "+X",
    "neg_x": "-X",
    "pos_y": "+Y",
    "neg_y": "-Y",
    "top": "Top",
    "bottom": "Bottom",
}


# ### Tests ###
class GlbViewerRenderingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.widgets: list[GlbViewerWidget] = []

    def tearDown(self) -> None:
        for widget in self.widgets:
            widget.close()
            widget.deleteLater()
        _qt_application.processEvents()

    def _build_viewer(self, **options: bool) -> GlbViewerWidget:
        viewer = GlbViewerWidget(**options)
        self.widgets.append(viewer)
        return viewer

    def test_textures_can_be_hidden_without_hiding_geometry(self) -> None:
        viewer = self._build_viewer()
        viewer.set_model(_build_generated_model(textured=True))

        self.assertIsNotNone(viewer.textured_mesh_item)
        self.assertIsNotNone(viewer.mesh_item)
        assert viewer.textured_mesh_item is not None
        assert viewer.mesh_item is not None
        self.assertTrue(viewer.get_textures_enabled())
        self.assertTrue(viewer.textured_mesh_item.visible())
        self.assertFalse(viewer.mesh_item.opts["drawFaces"])

        viewer.set_textures_enabled(False)

        self.assertFalse(viewer.get_textures_enabled())
        self.assertFalse(viewer.textured_mesh_item.visible())
        self.assertTrue(viewer.mesh_item.opts["drawFaces"])

    def test_texture_edit_mask_is_previewed_without_mutating_base_texture(
        self,
    ) -> None:
        viewer = self._build_viewer()
        viewer.set_model(_build_generated_model(textured=True))
        assert viewer.textured_mesh_item is not None
        textured_item = viewer.textured_mesh_item
        base_texture = textured_item._texture_rgba.copy()
        edit_mask = np.zeros((2, 2), dtype=np.uint8)
        edit_mask[0, 0] = 255

        viewer.set_texture_edit_mask(edit_mask)

        self.assertTrue(textured_item._edit_mask_enabled)
        np.testing.assert_array_equal(textured_item._edit_mask, edit_mask)
        np.testing.assert_array_equal(textured_item._texture_rgba, base_texture)

        viewer.set_texture_edit_mask(None)

        self.assertFalse(textured_item._edit_mask_enabled)
        np.testing.assert_array_equal(textured_item._texture_rgba, base_texture)

    def test_wireframe_only_forces_edges_and_hides_all_surfaces(self) -> None:
        viewer = self._build_viewer()
        viewer.set_model(_build_generated_model(textured=True))

        self.assertIsNotNone(viewer.textured_mesh_item)
        self.assertIsNotNone(viewer.mesh_item)
        assert viewer.textured_mesh_item is not None
        assert viewer.mesh_item is not None
        viewer.set_wireframe_enabled(False)
        viewer.set_wireframe_only(True)

        self.assertTrue(viewer.get_wireframe_only())
        self.assertFalse(viewer.textured_mesh_item.visible())
        self.assertFalse(viewer.mesh_item.opts["drawFaces"])
        self.assertTrue(viewer.mesh_item.opts["drawEdges"])

        viewer.set_wireframe_only(False)

        self.assertFalse(viewer.get_wireframe_only())
        self.assertTrue(viewer.textured_mesh_item.visible())
        self.assertFalse(viewer.mesh_item.opts["drawFaces"])
        self.assertFalse(viewer.mesh_item.opts["drawEdges"])

    def test_wireframe_can_be_disabled_for_a_non_textured_model(self) -> None:
        viewer = self._build_viewer(wireframe_enabled=False)
        viewer.set_model(_build_generated_model())

        self.assertIsNotNone(viewer.mesh_item)
        assert viewer.mesh_item is not None
        self.assertTrue(viewer.mesh_item.opts["drawFaces"])
        self.assertFalse(viewer.get_wireframe_enabled())
        self.assertTrue(viewer.mesh_item.opts["drawFaces"])
        self.assertFalse(viewer.mesh_item.opts["drawEdges"])

        viewer.set_wireframe_enabled(True)

        self.assertTrue(viewer.get_wireframe_enabled())
        self.assertTrue(viewer.mesh_item.opts["drawEdges"])

    def test_hidden_initial_wireframe_prepares_edges_for_later_toggle(
        self,
    ) -> None:
        viewer = self._build_viewer(wireframe_enabled=False)
        viewer.set_model(_build_generated_model(textured=True))

        assert viewer.mesh_item is not None
        self.assertFalse(viewer.mesh_item.opts["drawEdges"])

        viewer.mesh_item.parseMeshData()

        self.assertFalse(viewer.mesh_item.opts["drawEdges"])
        self.assertIsNotNone(viewer.mesh_item.edges)
        self.assertIsNotNone(viewer.mesh_item.edgeVerts)

        viewer.set_wireframe_enabled(True)

        self.assertTrue(viewer.mesh_item.opts["drawEdges"])
        self.assertGreater(viewer.mesh_item.edges.size, 0)

    def test_wireframe_edges_accept_coplanar_depth_then_restore_gl_state(
        self,
    ) -> None:
        viewer = self._build_viewer(wireframe_enabled=False)
        viewer.set_model(_build_generated_model(textured=True))
        viewer.set_wireframe_enabled(True)

        assert viewer.mesh_item is not None
        with (
            patch.object(
                GL,
                "glGetIntegerv",
                return_value=GL.GL_LESS,
            ),
            patch.object(GL, "glDepthFunc") as depth_function,
            patch.object(gl.GLMeshItem, "paint") as base_paint,
        ):
            viewer.mesh_item.paint()

        base_paint.assert_called_once_with()
        gl_options = viewer.mesh_item._GLGraphicsItem__glOpts
        self.assertFalse(gl_options[GL.GL_CULL_FACE])
        self.assertEqual(
            [call.args[0] for call in depth_function.call_args_list],
            [GL.GL_LEQUAL, GL.GL_LESS],
        )

    def test_viewer_forwards_first_person_navigation_apis(self) -> None:
        viewer = self._build_viewer()
        pose = CameraPose(x=1.0, y=2.0, z=1.7, yaw_degrees=45.0)

        self.assertTrue(viewer.first_person_crosshair_label.isHidden())

        viewer.set_first_person_camera_pose(pose)
        viewer.set_navigation_mode(NAVIGATION_MODE_FIRST_PERSON)

        self.assertEqual(viewer.get_navigation_mode(), NAVIGATION_MODE_FIRST_PERSON)
        self.assertEqual(viewer.get_first_person_camera_pose(), pose)
        self.assertTrue(viewer.is_first_person_pointer_captured)
        self.assertFalse(viewer.first_person_crosshair_label.isHidden())
        viewer.release_first_person_pointer_capture()
        self.assertFalse(viewer.is_first_person_pointer_captured)
        self.assertEqual(viewer.get_navigation_mode(), NAVIGATION_MODE_FIRST_PERSON)
        self.assertEqual(viewer.toggle_navigation_mode(), NAVIGATION_MODE_ORBIT)
        self.assertEqual(viewer.get_navigation_mode(), NAVIGATION_MODE_ORBIT)
        self.assertTrue(viewer.first_person_crosshair_label.isHidden())

    def test_first_person_pose_survives_a_model_scene_refresh(self) -> None:
        viewer = self._build_viewer()
        pose = CameraPose(x=1.0, y=2.0, z=1.7, yaw_degrees=45.0)
        viewer.set_first_person_camera_pose(pose)
        viewer.enter_first_person_mode()

        viewer.set_model(_build_generated_model())

        self.assertEqual(viewer.get_navigation_mode(), NAVIGATION_MODE_FIRST_PERSON)
        self.assertEqual(viewer.get_first_person_camera_pose(), pose)
        self.assertAlmostEqual(float(viewer.view.opts["distance"]), 1.0)

    def test_unused_face_camera_indicators_follow_selected_camera_ids(
        self,
    ) -> None:
        viewer = self._build_viewer()
        viewer.set_unused_face_camera_indicators_visible(True)
        viewer.set_model(_build_generated_model())

        indicators = viewer.unused_face_camera_indicator_items
        labels = viewer.unused_face_camera_indicator_labels
        self.assertEqual(tuple(indicators), ALL_CAMERA_IDS)
        self.assertEqual(tuple(labels), ALL_CAMERA_IDS)
        self.assertEqual(
            viewer.get_enabled_unused_face_camera_ids(),
            ALL_CAMERA_IDS,
        )
        self.assertTrue(viewer.get_unused_face_camera_indicators_visible())
        self.assertTrue(all(indicators.values()))
        self.assertTrue(
            all(
                item.visible() and item in viewer.view.items
                for camera_items in indicators.values()
                for item in camera_items
            )
        )
        self.assertEqual(
            {
                camera_id: label.text
                for camera_id, label in labels.items()
            },
            _EXPECTED_UNUSED_FACE_CAMERA_LABELS,
        )
        for camera_id, label in labels.items():
            label_position = np.asarray(label.pos, dtype=float)
            self.assertEqual(label_position.shape, (3,))
            self.assertTrue(np.all(np.isfinite(label_position)))
            self.assertIn(label, indicators[camera_id])
            camera_line_positions = np.concatenate(
                [
                    np.asarray(item.pos, dtype=float)
                    for item in indicators[camera_id]
                    if item is not label
                ],
                axis=0,
            )
            nearest_line_distance = float(
                np.min(
                    np.linalg.norm(
                        camera_line_positions - label_position,
                        axis=1,
                    )
                )
            )
            self.assertLess(nearest_line_distance, 0.5)

        enabled_camera_ids = ("neg_x", "pos_y", "bottom")
        viewer.set_enabled_unused_face_camera_ids(enabled_camera_ids)

        self.assertEqual(
            viewer.get_enabled_unused_face_camera_ids(),
            enabled_camera_ids,
        )
        for camera_id, camera_items in indicators.items():
            expected_visibility = camera_id in enabled_camera_ids
            self.assertTrue(
                all(
                    item.visible() == expected_visibility
                    for item in camera_items
                )
            )
            self.assertEqual(
                labels[camera_id].visible(),
                expected_visibility,
            )

        viewer.set_unused_face_camera_indicators_visible(False)

        self.assertFalse(viewer.get_unused_face_camera_indicators_visible())
        self.assertEqual(
            viewer.get_enabled_unused_face_camera_ids(),
            enabled_camera_ids,
        )
        self.assertEqual(viewer.unused_face_camera_indicator_items, {})
        self.assertEqual(viewer.unused_face_camera_indicator_labels, {})
        removed_items = tuple(
            item
            for camera_items in indicators.values()
            for item in camera_items
        )
        self.assertTrue(
            all(
                not item.visible() and item not in viewer.view.items
                for item in removed_items
            )
        )

        viewer.set_unused_face_camera_indicators_visible(True)

        rebuilt_indicators = viewer.unused_face_camera_indicator_items
        rebuilt_labels = viewer.unused_face_camera_indicator_labels
        self.assertEqual(tuple(rebuilt_indicators), ALL_CAMERA_IDS)
        self.assertEqual(
            {
                camera_id: label.text
                for camera_id, label in rebuilt_labels.items()
            },
            _EXPECTED_UNUSED_FACE_CAMERA_LABELS,
        )
        for camera_id, camera_items in rebuilt_indicators.items():
            expected_visibility = camera_id in enabled_camera_ids
            self.assertTrue(
                all(
                    item.visible() == expected_visibility
                    and item in viewer.view.items
                    for item in camera_items
                )
            )

    def test_unused_face_camera_indicator_state_survives_model_refresh(
        self,
    ) -> None:
        viewer = self._build_viewer()
        enabled_camera_ids = ("pos_x", "top")
        viewer.set_enabled_unused_face_camera_ids(enabled_camera_ids)
        viewer.set_unused_face_camera_indicators_visible(True)

        self.assertEqual(viewer.unused_face_camera_indicator_items, {})
        self.assertEqual(viewer.unused_face_camera_indicator_labels, {})
        viewer.set_model(_build_generated_model())
        original_indicators = viewer.unused_face_camera_indicator_items
        original_labels = viewer.unused_face_camera_indicator_labels
        self.assertEqual(tuple(original_indicators), ALL_CAMERA_IDS)
        self.assertEqual(tuple(original_labels), ALL_CAMERA_IDS)
        original_items = tuple(
            item
            for camera_items in original_indicators.values()
            for item in camera_items
        )

        viewer.clear_model()

        self.assertEqual(viewer.unused_face_camera_indicator_items, {})
        self.assertEqual(viewer.unused_face_camera_indicator_labels, {})
        self.assertTrue(
            all(item not in viewer.view.items for item in original_items)
        )
        self.assertEqual(
            viewer.get_enabled_unused_face_camera_ids(),
            enabled_camera_ids,
        )
        self.assertTrue(viewer.get_unused_face_camera_indicators_visible())

        viewer.set_model(_build_generated_model())

        rebuilt_indicators = viewer.unused_face_camera_indicator_items
        rebuilt_labels = viewer.unused_face_camera_indicator_labels
        self.assertEqual(tuple(rebuilt_indicators), ALL_CAMERA_IDS)
        self.assertEqual(tuple(rebuilt_labels), ALL_CAMERA_IDS)
        self.assertEqual(
            {
                camera_id: label.text
                for camera_id, label in rebuilt_labels.items()
            },
            _EXPECTED_UNUSED_FACE_CAMERA_LABELS,
        )
        self.assertTrue(
            all(
                rebuilt_labels[camera_id] is not original_labels[camera_id]
                for camera_id in ALL_CAMERA_IDS
            )
        )
        self.assertTrue(
            all(
                item not in original_items
                for camera_items in rebuilt_indicators.values()
                for item in camera_items
            )
        )
        for camera_id, camera_items in rebuilt_indicators.items():
            expected_visibility = camera_id in enabled_camera_ids
            self.assertTrue(
                all(
                    item.visible() == expected_visibility
                    for item in camera_items
                )
            )
            self.assertEqual(
                rebuilt_labels[camera_id].visible(),
                expected_visibility,
            )


class BlenderNavigationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.view = SelectableGLViewWidget()

    def tearDown(self) -> None:
        self.view.close()
        self.view.deleteLater()
        _qt_application.processEvents()

    def test_middle_drag_orbits_and_shift_middle_drag_pans(self) -> None:
        orbit = Mock()
        pan = Mock()
        self.view.orbit = orbit
        self.view.pan = pan
        self.view.mousePos = QPointF(10.0, 20.0)

        orbit_event = FakeMouseMoveEvent(
            position=QPointF(16.0, 25.0),
            buttons=Qt.MouseButton.MiddleButton,
        )
        self.view.mouseMoveEvent(orbit_event)

        orbit.assert_called_once_with(-6.0, 5.0)
        self.assertTrue(orbit_event.was_accepted)
        self.assertFalse(pan.called)

        self.view.mousePos = QPointF(16.0, 25.0)
        pan_event = FakeMouseMoveEvent(
            position=QPointF(19.0, 21.0),
            buttons=Qt.MouseButton.MiddleButton,
            modifiers=Qt.KeyboardModifier.ShiftModifier,
        )
        self.view.mouseMoveEvent(pan_event)

        pan.assert_called_once_with(3.0, -4.0, 0.0, relative="view")
        self.assertTrue(pan_event.was_accepted)

    def test_left_drag_does_not_orbit_and_wheel_zooms(self) -> None:
        orbit = Mock()
        pan = Mock()
        self.view.orbit = orbit
        self.view.pan = pan
        self.view.mousePos = QPointF(1.0, 1.0)

        left_drag_event = FakeMouseMoveEvent(
            position=QPointF(8.0, 5.0),
            buttons=Qt.MouseButton.LeftButton,
        )
        self.view.mouseMoveEvent(left_drag_event)

        self.assertTrue(left_drag_event.was_accepted)
        self.assertFalse(orbit.called)
        self.assertFalse(pan.called)

        original_distance = float(self.view.opts["distance"])
        wheel_event = FakeWheelEvent(120)
        self.view.wheelEvent(wheel_event)

        self.assertTrue(wheel_event.was_accepted)
        self.assertLess(float(self.view.opts["distance"]), original_distance)


class FirstPersonNavigationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.view = SelectableGLViewWidget()
        self.view.resize(320, 180)

    def tearDown(self) -> None:
        self.view.exit_first_person_mode()
        self.view.close()
        self.view.deleteLater()
        _qt_application.processEvents()

    def test_toggle_switches_between_orbit_and_first_person(self) -> None:
        self.assertEqual(self.view.get_navigation_mode(), NAVIGATION_MODE_ORBIT)

        self.assertEqual(
            self.view.toggle_navigation_mode(),
            NAVIGATION_MODE_FIRST_PERSON,
        )
        self.assertTrue(self.view.is_first_person_active)
        self.assertEqual(
            self.view.toggle_navigation_mode(),
            NAVIGATION_MODE_ORBIT,
        )
        self.assertFalse(self.view.is_first_person_active)

    def test_zqsd_movement_uses_yaw_and_preserves_camera_z(self) -> None:
        self.view.set_first_person_camera_pose(
            CameraPose(x=1.0, y=2.0, z=1.7)
        )
        self.view.enter_first_person_mode()
        self.view.keyPressEvent(
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_Z,
                Qt.KeyboardModifier.NoModifier,
            )
        )

        self.view.step_first_person_movement(0.4)
        moved_pose = self.view.get_first_person_camera_pose()

        self.assertAlmostEqual(moved_pose.x, 2.0)
        self.assertAlmostEqual(moved_pose.y, 2.0)
        self.assertAlmostEqual(moved_pose.z, 1.7)

    def test_q_moves_left_and_d_moves_right_on_french_layout(self) -> None:
        self.view.enter_first_person_mode()
        cases = (
            (0.0, Qt.Key.Key_Q, (0.0, 1.0)),
            (0.0, Qt.Key.Key_D, (0.0, -1.0)),
            (90.0, Qt.Key.Key_Q, (-1.0, 0.0)),
            (90.0, Qt.Key.Key_D, (1.0, 0.0)),
        )
        for yaw_degrees, key, expected_xy in cases:
            with self.subTest(yaw_degrees=yaw_degrees, key=key):
                self.view.set_first_person_camera_pose(
                    CameraPose(z=1.7, yaw_degrees=yaw_degrees)
                )
                self.view.keyPressEvent(
                    QKeyEvent(
                        QEvent.Type.KeyPress,
                        key,
                        Qt.KeyboardModifier.NoModifier,
                    )
                )
                self.view.step_first_person_movement(0.4)
                self.view.keyReleaseEvent(
                    QKeyEvent(
                        QEvent.Type.KeyRelease,
                        key,
                        Qt.KeyboardModifier.NoModifier,
                    )
                )

                moved_pose = self.view.get_first_person_camera_pose()
                self.assertAlmostEqual(moved_pose.x, expected_xy[0])
                self.assertAlmostEqual(moved_pose.y, expected_xy[1])
                self.assertAlmostEqual(moved_pose.z, 1.7)

    def test_r_moves_down_and_f_moves_up_without_gravity(self) -> None:
        self.view.enter_first_person_mode()
        cases = (
            (Qt.Key.Key_R, 0.7),
            (Qt.Key.Key_F, 2.7),
        )
        for key, expected_z in cases:
            with self.subTest(key=key):
                self.view.set_first_person_camera_pose(
                    CameraPose(x=1.0, y=2.0, z=1.7)
                )
                self.view.keyPressEvent(
                    QKeyEvent(
                        QEvent.Type.KeyPress,
                        key,
                        Qt.KeyboardModifier.NoModifier,
                    )
                )
                self.view.step_first_person_movement(0.4)
                self.view.keyReleaseEvent(
                    QKeyEvent(
                        QEvent.Type.KeyRelease,
                        key,
                        Qt.KeyboardModifier.NoModifier,
                    )
                )

                moved_pose = self.view.get_first_person_camera_pose()
                self.assertAlmostEqual(moved_pose.x, 1.0)
                self.assertAlmostEqual(moved_pose.y, 2.0)
                self.assertAlmostEqual(moved_pose.z, expected_z)

    def test_repeated_vertical_key_press_keeps_moving_until_release(self) -> None:
        self.view.set_first_person_camera_pose(CameraPose(z=1.7))
        self.view.enter_first_person_mode()
        press_event = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_F,
            Qt.KeyboardModifier.NoModifier,
        )

        self.view.keyPressEvent(press_event)
        self.view.keyPressEvent(press_event)
        self.view.step_first_person_movement(0.2)
        self.view.step_first_person_movement(0.2)

        self.assertAlmostEqual(
            self.view.get_first_person_camera_pose().z,
            2.7,
        )
        self.view.keyReleaseEvent(
            QKeyEvent(
                QEvent.Type.KeyRelease,
                Qt.Key.Key_F,
                Qt.KeyboardModifier.NoModifier,
            )
        )
        self.view.step_first_person_movement(0.4)
        self.assertAlmostEqual(
            self.view.get_first_person_camera_pose().z,
            2.7,
        )

    def test_right_click_releases_pointer_but_preserves_first_person_view(
        self,
    ) -> None:
        self.view.set_first_person_camera_pose(CameraPose(z=1.7))
        self.view.enter_first_person_mode()
        orbit = Mock()
        self.view.orbit = orbit
        center = QPointF(self.view.rect().center())
        move_event = FakeMouseMoveEvent(
            position=center + QPointF(10.0, 5.0),
            buttons=Qt.MouseButton.NoButton,
        )

        self.view.mouseMoveEvent(move_event)
        pose = self.view.get_first_person_camera_pose()

        self.assertTrue(move_event.was_accepted)
        self.assertFalse(orbit.called)
        self.assertAlmostEqual(pose.yaw_degrees, -1.6)
        self.assertAlmostEqual(pose.pitch_degrees, -0.8)

        first_person_distance = float(self.view.opts["distance"])
        release_event = FakeMousePressEvent(button=Qt.MouseButton.RightButton)
        self.view.mousePressEvent(release_event)

        self.assertTrue(release_event.was_accepted)
        self.assertEqual(
            self.view.get_navigation_mode(),
            NAVIGATION_MODE_FIRST_PERSON,
        )
        self.assertFalse(self.view.is_first_person_pointer_captured)
        self.assertAlmostEqual(
            float(self.view.opts["distance"]),
            first_person_distance,
        )

    def test_released_first_person_pointer_can_select_viewport_content(
        self,
    ) -> None:
        self.view.enter_first_person_mode()
        self.view.release_first_person_pointer_capture()
        clicked_positions: list[QPointF] = []
        clicked_items: list[object] = []
        self.view.viewport_clicked.connect(clicked_positions.append)
        self.view.items_clicked.connect(clicked_items.append)
        self.view._get_clicked_items = Mock(return_value=["wall"])
        position = QPointF(42.0, 24.0)

        self.view.mousePressEvent(
            FakeMousePressEvent(
                button=Qt.MouseButton.LeftButton,
                position=position,
            )
        )
        self.view.mouseReleaseEvent(
            FakeMousePressEvent(
                button=Qt.MouseButton.LeftButton,
                position=position,
            )
        )

        self.assertEqual(clicked_positions, [position])
        self.assertEqual(clicked_items, [["wall"]])
        self.assertEqual(
            self.view.get_navigation_mode(),
            NAVIGATION_MODE_FIRST_PERSON,
        )

    def test_focus_loss_releases_capture_without_leaving_first_person(
        self,
    ) -> None:
        self.view.enter_first_person_mode()
        self.view.keyPressEvent(
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_Z,
                Qt.KeyboardModifier.NoModifier,
            )
        )

        self.view.focusOutEvent(QFocusEvent(QEvent.Type.FocusOut))

        self.assertEqual(
            self.view.get_navigation_mode(),
            NAVIGATION_MODE_FIRST_PERSON,
        )
        self.assertFalse(self.view.is_first_person_pointer_captured)
        self.assertEqual(self.view._pressed_movement_keys, set())

    def test_rectangle_tool_preserves_first_person_camera_mode(self) -> None:
        self.view.enter_first_person_mode()

        self.view.set_rectangle_drawing_enabled(True)

        self.assertTrue(self.view.is_rectangle_drawing_enabled)
        self.assertFalse(self.view.is_first_person_pointer_captured)
        self.assertEqual(
            self.view.get_navigation_mode(),
            NAVIGATION_MODE_FIRST_PERSON,
        )
