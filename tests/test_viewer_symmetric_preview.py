# ### Environment setup ###
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import numpy as np
import trimesh
from PIL import Image
from PySide6.QtWidgets import QApplication
from trimesh.visual.material import SimpleMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.glb import GeneratedModel
from housemaker.viewer import GlbViewerWidget


# ### Test application ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _textured_half_model() -> GeneratedModel:
    vertices = np.asarray(
        [
            (-1.0, -0.5, 0.0),
            (0.0, -0.5, 0.0),
            (0.0, 0.5, 1.0),
            (-1.0, 0.5, 1.0),
        ],
        dtype=float,
    )
    faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    texture = Image.new("RGBA", (8, 8), (120, 70, 30, 255))
    mesh.visual = TextureVisuals(
        uv=np.asarray(
            ((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)),
            dtype=float,
        ),
        material=SimpleMaterial(image=texture),
    )
    return GeneratedModel(
        mesh=mesh,
        scene=trimesh.Scene(mesh.copy()),
        glb_bytes=b"retained-half-glb",
    )


# ### Symmetric viewer tests ###
class SymmetricViewerPreviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.viewer = GlbViewerWidget()
        self.viewer.resize(640, 480)
        self.viewer.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.viewer.close()
        self.viewer.deleteLater()
        _qt_application.processEvents()

    def test_mirror_fades_without_mutating_retained_or_export_data(self) -> None:
        model = _textured_half_model()
        vertices_before = np.asarray(model.mesh.vertices).copy()
        faces_before = np.asarray(model.mesh.faces).copy()
        glb_before = bytes(model.glb_bytes)
        scene_bounds_before = np.asarray(model.scene.bounds).copy()

        self.viewer.set_model(model)
        self.viewer.set_symmetric_division_preview("vertical", 0.0)

        self.assertIsNotNone(self.viewer.symmetric_preview_textured_mesh_item)
        self.assertIsNotNone(self.viewer.symmetric_preview_mesh_item)
        mirrored_vertices = self.viewer._symmetric_preview_vertices
        mirrored_faces = self.viewer._symmetric_preview_faces
        assert mirrored_vertices is not None and mirrored_faces is not None
        np.testing.assert_allclose(
            mirrored_vertices[:, 0],
            -vertices_before[:, 0],
        )
        np.testing.assert_array_equal(
            mirrored_faces,
            faces_before[:, (0, 2, 1)],
        )
        ghost = self.viewer.symmetric_preview_textured_mesh_item
        assert ghost is not None
        opacity_before = ghost._opacity
        self.viewer._advance_symmetric_preview_fade()
        self.assertNotEqual(ghost._opacity, opacity_before)
        np.testing.assert_array_equal(model.mesh.vertices, vertices_before)
        np.testing.assert_array_equal(model.mesh.faces, faces_before)
        np.testing.assert_array_equal(model.scene.bounds, scene_bounds_before)
        self.assertEqual(model.glb_bytes, glb_before)

    def test_texture_wireframe_visibility_and_timer_lifecycle(self) -> None:
        self.viewer.set_model(_textured_half_model())
        self.viewer.set_symmetric_division_preview("vertical", 0.0)
        textured = self.viewer.symmetric_preview_textured_mesh_item
        fallback = self.viewer.symmetric_preview_mesh_item
        assert textured is not None and fallback is not None
        self.assertTrue(self.viewer._symmetric_preview_timer.isActive())

        self.viewer.set_textures_enabled(False)
        self.assertFalse(textured.visible())
        self.assertTrue(fallback.opts["drawFaces"])
        self.viewer.set_wireframe_only(True)
        self.assertFalse(fallback.opts["drawFaces"])
        self.assertTrue(fallback.opts["drawEdges"])

        self.viewer.hide()
        _qt_application.processEvents()
        self.assertFalse(self.viewer._symmetric_preview_timer.isActive())
        self.viewer.show()
        _qt_application.processEvents()
        self.assertTrue(self.viewer._symmetric_preview_timer.isActive())
        self.viewer.clear_model()
        self.assertFalse(self.viewer._symmetric_preview_timer.isActive())
        self.assertIsNone(self.viewer.symmetric_preview_textured_mesh_item)
        self.assertIsNone(self.viewer.symmetric_preview_mesh_item)


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
