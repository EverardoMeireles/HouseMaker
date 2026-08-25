# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import unittest
from unittest.mock import patch

import numpy as np
import trimesh
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QVector3D
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication
from trimesh.visual.texture import TextureVisuals

from housemaker.glb import import_generated_glb
from housemaker.object_texture_inpaint import pick_texture_uv_from_ray
from housemaker.viewer import (
    NAVIGATION_MODE_FIRST_PERSON,
    NAVIGATION_MODE_ORBIT,
    SelectableGLViewWidget,
)


# ### Test application ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Event doubles ###
class _PointerEvent:
    def __init__(
        self,
        position: QPointF,
        *,
        button: Qt.MouseButton = Qt.MouseButton.LeftButton,
        buttons: Qt.MouseButton = Qt.MouseButton.LeftButton,
    ) -> None:
        self._position = position
        self._button = button
        self._buttons = buttons
        self.was_accepted = False

    def position(self) -> QPointF:
        return self._position

    def button(self) -> Qt.MouseButton:
        return self._button

    def buttons(self) -> Qt.MouseButton:
        return self._buttons

    def accept(self) -> None:
        self.was_accepted = True


# ### Camera ray tests ###
class ViewerCameraRayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.view = SelectableGLViewWidget()
        self.view.resize(800, 400)
        self.view.setCameraPosition(
            pos=QVector3D(0.0, 0.0, 0.0),
            distance=10.0,
            elevation=0.0,
            azimuth=0.0,
        )

    def tearDown(self) -> None:
        self.view.exit_first_person_mode()
        self.view.close()
        self.view.deleteLater()
        _qt_application.processEvents()

    def test_ray_uses_full_viewport_projection_and_qt_widget_coordinates(
        self,
    ) -> None:
        viewport = self.view.getViewport()
        original_projection = self.view.projectionMatrix

        with patch.object(
            self.view,
            "projectionMatrix",
            wraps=original_projection,
        ) as projection:
            center_ray = self.view.build_camera_ray(QPointF(400.0, 200.0))
            right_ray = self.view.build_camera_ray(QPointF(800.0, 200.0))
            top_ray = self.view.build_camera_ray(QPointF(400.0, 0.0))

        self.assertEqual(
            projection.call_args_list[0].args,
            (viewport, viewport),
        )
        self.assertIsNotNone(center_ray)
        self.assertIsNotNone(right_ray)
        self.assertIsNotNone(top_ray)
        assert center_ray is not None
        assert right_ray is not None
        assert top_ray is not None
        center_origin, center_direction = center_ray
        _right_origin, right_direction = right_ray
        _top_origin, top_direction = top_ray
        np.testing.assert_allclose(
            center_origin,
            (9.99, 0.0, 0.0),
            atol=1e-4,
        )
        np.testing.assert_allclose(
            center_direction,
            (-1.0, 0.0, 0.0),
            atol=1e-6,
        )
        self.assertGreater(right_direction[1], 0.0)
        self.assertGreater(top_direction[2], 0.0)
        self.assertAlmostEqual(float(np.linalg.norm(right_direction)), 1.0)
        self.assertAlmostEqual(float(np.linalg.norm(top_direction)), 1.0)

    def test_center_ray_stays_in_the_shifted_world_coordinate_frame(self) -> None:
        expected_target = np.asarray((3.0, -2.0, 5.0), dtype=float)
        self.view.setCameraPosition(
            pos=QVector3D(*expected_target.tolist()),
            distance=10.0,
            elevation=0.0,
            azimuth=0.0,
        )

        ray = self.view.build_camera_ray(QPointF(400.0, 200.0))

        self.assertIsNotNone(ray)
        assert ray is not None
        origin, direction = ray
        target_offset = expected_target - origin
        distance_along_ray = float(np.dot(target_offset, direction))
        closest_point = origin + direction * distance_along_ray
        np.testing.assert_allclose(closest_point, expected_target, atol=2e-4)
        self.assertGreater(distance_along_ray, 0.0)

    def test_ray_hits_uvs_after_glb_node_and_axis_transforms(self) -> None:
        mesh = trimesh.Trimesh(
            vertices=np.asarray(
                (
                    (0.0, -1.0, -1.0),
                    (0.0, 1.0, -1.0),
                    (0.0, 0.0, 1.0),
                ),
                dtype=float,
            ),
            faces=np.asarray(((0, 1, 2),), dtype=np.int64),
            process=False,
        )
        mesh.visual = TextureVisuals(
            uv=np.asarray(((0.0, 0.0), (1.0, 0.0), (0.5, 1.0)))
        )
        scene = trimesh.Scene()
        node_transform = np.eye(4, dtype=float)
        node_transform[:3, 3] = (3.0, 4.0, 5.0)
        scene.add_geometry(
            mesh,
            geom_name="triangle",
            node_name="translated-triangle",
            transform=node_transform,
        )
        model = import_generated_glb(bytes(scene.export(file_type="glb")))
        target = np.asarray(model.mesh.centroid, dtype=float)
        self.view.setCameraPosition(
            pos=QVector3D(*target.tolist()),
            distance=10.0,
            elevation=0.0,
            azimuth=0.0,
        )

        ray = self.view.build_camera_ray(QPointF(400.0, 200.0))

        self.assertIsNotNone(ray)
        assert ray is not None
        hit = pick_texture_uv_from_ray(model.mesh, *ray)
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.face_index, 0)
        self.assertAlmostEqual(hit.point.u, 0.5, places=5)
        self.assertAlmostEqual(hit.point.v, 1.0 / 3.0, places=4)


# ### Retired inpaint pointer tests ###
class ViewerRetiredInpaintPointerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.view = SelectableGLViewWidget()

    def tearDown(self) -> None:
        self.view.exit_first_person_mode()
        self.view.close()
        self.view.deleteLater()
        _qt_application.processEvents()

    def test_object_texture_inpaint_pointer_api_is_removed(self) -> None:
        self.assertFalse(hasattr(self.view, "texture_inpaint_enabled"))
        self.assertFalse(hasattr(self.view, "set_texture_inpaint_enabled"))
        self.assertFalse(
            hasattr(self.view, "texture_inpaint_pointer_pressed")
        )


# ### Test runner ###
if __name__ == "__main__":
    unittest.main()
