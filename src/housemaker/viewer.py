# ### Imports ###
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph.opengl as gl
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCloseEvent, QVector3D
from PySide6.QtWidgets import QFileDialog, QMessageBox, QPushButton, QVBoxLayout, QWidget

from housemaker.glb import GeneratedModel, export_glb_file

# ### Constants ###
EDGE_COLOR = (0.12, 0.12, 0.16, 1.0)
FACE_COLOR = np.array([0.78, 0.80, 0.84, 1.0], dtype=float)

# ### Windows ###
class GlbViewerWindow(QWidget):
    closed = Signal(object)

    def __init__(
        self,
        model: GeneratedModel,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.model = model
        self.grid_item: gl.GLGridItem | None = None
        self.mesh_item: gl.GLMeshItem | None = None

        self._build_ui()
        self._populate_scene()

    def _build_ui(self) -> None:
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowTitle("HouseMaker 3D Viewer")
        self.resize(1100, 720)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor((24, 24, 28))
        layout.addWidget(self.view, 1)

        self.export_button = QPushButton("export glb")
        self.export_button.setMinimumHeight(48)
        self.export_button.clicked.connect(self._export_glb)
        layout.addWidget(self.export_button)

    def _populate_scene(self) -> None:
        self.grid_item = gl.GLGridItem()
        self.grid_item.setSize(x=20.0, y=20.0)
        self.grid_item.setSpacing(x=1.0, y=1.0)
        self.view.addItem(self.grid_item)

        vertices = np.asarray(self.model.mesh.vertices, dtype=np.float32)
        faces = np.asarray(self.model.mesh.faces, dtype=np.int32)
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

    def closeEvent(self, event: QCloseEvent) -> None:
        self._clear_scene()
        self.closed.emit(self)
        super().closeEvent(event)

    def _export_glb(self) -> None:
        default_path = Path.cwd() / "housemaker_export.glb"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "export glb",
            str(default_path),
            "GLB Files (*.glb)",
        )
        if not file_path:
            return

        export_glb_file(self.model, file_path)
        QMessageBox.information(self, "GLB exported", f"Saved GLB to:\n{file_path}")

    def _clear_scene(self) -> None:
        if not hasattr(self, "view"):
            return

        self.view.clear()
        self.grid_item = None
        self.mesh_item = None
