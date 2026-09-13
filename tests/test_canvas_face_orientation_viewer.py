# ### Environment setup ###
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import numpy as np
import trimesh
from OpenGL import GL
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from housemaker.glb import GeneratedModel
from housemaker.surface_geometry import SURFACE_TYPE_WALL, FixedSurface
from housemaker.viewer import (
    CANVAS_FACE_ORIENTATION_DEPTH_VALUE,
    GlbViewerWidget,
    _build_canvas_face_orientation_geometry,
    _FaceOrientationOverlayMeshItem,
)

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_wall() -> FixedSurface:
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (4.0, 0.0, 0.0),
            (4.0, 0.0, 3.0),
            (0.0, 0.0, 3.0),
        ),
        dtype=float,
    )
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        process=False,
    )
    return FixedSurface(
        surface_id="level:0/wall:1:2",
        surface_type=SURFACE_TYPE_WALL,
        level_index=0,
        room_index=None,
        mesh=mesh,
        area_square_meters=12.0,
        wall_key="1:2",
        wall_start_world=(0.0, 0.0, 0.0),
        wall_end_world=(4.0, 0.0, 0.0),
        wall_height_meters=3.0,
    )


def _build_model() -> GeneratedModel:
    mesh = trimesh.creation.box(extents=(4.0, 0.2, 3.0))
    mesh.apply_translation((2.0, 0.1, 1.5))
    return GeneratedModel(mesh=mesh, scene=trimesh.Scene(mesh), glb_bytes=b"")


# ### Face-orientation viewer tests ###
class CanvasFaceOrientationViewerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.viewer = GlbViewerWidget(window_editing_enabled=True)
        self.wall = _build_wall()
        self.viewer.set_wall_targets((self.wall,))
        self.viewer.set_model(_build_model())

    def tearDown(self) -> None:
        self.viewer.close()
        self.viewer.deleteLater()
        _qt_application.processEvents()

    def test_mode_renders_depth_tested_front_and_back_passes(self) -> None:
        self.assertTrue(self.viewer.set_canvas_face_orientation_visible(True))

        items = self.viewer._canvas_face_orientation_items
        self.assertEqual(len(items), 2)
        self.assertTrue(
            all(isinstance(item, _FaceOrientationOverlayMeshItem) for item in items)
        )
        culled_faces = {
            item._GLGraphicsItem__glOpts["glCullFace"][0] for item in items
        }
        self.assertEqual(culled_faces, {GL.GL_FRONT, GL.GL_BACK})
        self.assertTrue(
            all(
                item._GLGraphicsItem__glOpts[GL.GL_DEPTH_TEST]
                for item in items
            )
        )
        self.assertTrue(
            all(
                item._GLGraphicsItem__glOpts["glDepthMask"] == (True,)
                for item in items
            )
        )
        self.assertTrue(
            all(
                item.depthValue() == CANVAS_FACE_ORIENTATION_DEPTH_VALUE
                for item in items
            )
        )

        self.assertTrue(self.viewer.set_canvas_face_orientation_visible(False))
        self.assertEqual(self.viewer._canvas_face_orientation_items, [])

    def test_click_emits_stable_surface_id_and_selects_surface(self) -> None:
        emitted: list[str] = []
        self.viewer.canvas_surface_orientation_flip_requested.connect(
            emitted.append
        )
        self.viewer.set_canvas_face_orientation_visible(True)
        ray = (
            np.asarray((2.0, -2.0, 1.5), dtype=float),
            np.asarray((0.0, 1.0, 0.0), dtype=float),
        )

        with patch.object(self.viewer.view, "build_camera_ray", return_value=ray):
            self.viewer._handle_canvas_face_orientation_pick_requested(
                QPointF(10.0, 10.0)
            )

        self.assertEqual(emitted, [self.wall.surface_id])
        self.assertEqual(
            self.viewer.get_selected_canvas_surface_ids(),
            (self.wall.surface_id,),
        )

    def test_placed_object_in_front_blocks_surface_flip(self) -> None:
        emitted: list[str] = []
        self.viewer.canvas_surface_orientation_flip_requested.connect(
            emitted.append
        )
        self.viewer.set_canvas_face_orientation_visible(True)
        ray = (
            np.asarray((2.0, -2.0, 1.5), dtype=float),
            np.asarray((0.0, 1.0, 0.0), dtype=float),
        )

        with (
            patch.object(self.viewer.view, "build_camera_ray", return_value=ray),
            patch(
                "housemaker.viewer._get_nearest_preview_placed_object_ray_hit",
                return_value=(object(), np.zeros(3, dtype=float), 1.0),
            ),
        ):
            self.viewer._handle_canvas_face_orientation_pick_requested(
                QPointF(10.0, 10.0)
            )

        self.assertEqual(emitted, [])

    def test_overlay_geometry_keeps_source_triangle_winding(self) -> None:
        geometry = _build_canvas_face_orientation_geometry((self.wall,))

        assert geometry is not None
        vertices, faces = geometry
        np.testing.assert_allclose(vertices, self.wall.mesh.vertices)
        np.testing.assert_array_equal(faces, self.wall.mesh.faces)


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
