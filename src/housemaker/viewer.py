# ### Imports ###
from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl
from PySide6.QtGui import QVector3D
from PySide6.QtWidgets import QVBoxLayout, QWidget

from housemaker.glb import GeneratedModel

# ### Constants ###
EDGE_COLOR = (0.12, 0.12, 0.16, 1.0)
FACE_COLOR = np.array([0.78, 0.80, 0.84, 1.0], dtype=float)

# ### Widgets ###
class GlbViewerWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model: GeneratedModel | None = None
        self.grid_item: gl.GLGridItem | None = None
        self.mesh_item: gl.GLMeshItem | None = None

        self._build_ui()
        self._populate_scene()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor((24, 24, 28))
        layout.addWidget(self.view, 1)

    def set_model(self, model: GeneratedModel) -> None:
        self.model = model
        self._populate_scene()

    def clear_model(self) -> None:
        self.model = None
        self._populate_scene()

    def _populate_scene(self) -> None:
        self._clear_scene()
        self._add_grid()
        if self.model is None:
            self._set_default_camera()
            return

        vertices = np.asarray(self.model.mesh.vertices, dtype=np.float32)
        faces = np.asarray(self.model.mesh.faces, dtype=np.int32)
        if vertices.size == 0 or faces.size == 0:
            self._set_default_camera()
            return

        face_colors = np.tile(FACE_COLOR, (faces.shape[0], 1))

        self.mesh_item = gl.GLMeshItem(
            vertexes=vertices,
            faces=faces,
            faceColors=face_colors,
            smooth=False,
            drawEdges=True,
            edgeColor=EDGE_COLOR,
            shader="shaded",
        )
        self.view.addItem(self.mesh_item)

        bounding_box = self.model.mesh.bounding_box
        center = np.asarray(bounding_box.centroid, dtype=float)
        extent = float(max(bounding_box.extents.max(), 1.0))

        self.view.opts["center"] = QVector3D(
            float(center[0]),
            float(center[1]),
            float(center[2]),
        )
        self.view.setCameraPosition(distance=extent * 3.0, elevation=28.0, azimuth=-40.0)
        self.view.update()

    def _add_grid(self) -> None:
        self.grid_item = gl.GLGridItem()
        self.grid_item.setSize(x=20.0, y=20.0)
        self.grid_item.setSpacing(x=1.0, y=1.0)
        self.view.addItem(self.grid_item)

    def _set_default_camera(self) -> None:
        self.view.opts["center"] = QVector3D(0.0, 0.0, 0.0)
        self.view.setCameraPosition(distance=18.0, elevation=28.0, azimuth=-40.0)

    def _clear_scene(self) -> None:
        if not hasattr(self, "view"):
            return

        self.view.clear()
        self.grid_item = None
        self.mesh_item = None
