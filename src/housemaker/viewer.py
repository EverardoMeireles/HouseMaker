# ### Imports ###
from __future__ import annotations

import numpy as np
from pyqtgraph import Transform3D
import pyqtgraph.opengl as gl
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QVector3D
from PySide6.QtWidgets import QVBoxLayout, QWidget

from housemaker.glb import GeneratedModel, PreviewTexturedWall

# ### Constants ###
EDGE_COLOR = (0.12, 0.12, 0.16, 1.0)
FACE_COLOR = np.array([0.78, 0.80, 0.84, 1.0], dtype=float)
TEXTURE_PREVIEW_OFFSET_METERS = 0.01
CAMERA_STATE_KEYS = ("center", "distance", "elevation", "azimuth", "fov")
CLICK_SELECTION_TOLERANCE = 4.0

# ### Widgets ###
class SelectableGLViewWidget(gl.GLViewWidget):
    items_clicked = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.click_press_position = QPointF()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        self.click_press_position = event.position()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if (
            event.button() == Qt.MouseButton.LeftButton
            and _get_point_distance(self.click_press_position, event.position())
            <= CLICK_SELECTION_TOLERANCE
        ):
            clicked_items = self._get_clicked_items(event.position())
            if clicked_items:
                self.items_clicked.emit(clicked_items)

        super().mouseReleaseEvent(event)

    def _get_clicked_items(self, position: QPointF) -> list[object]:
        region_size = int(CLICK_SELECTION_TOLERANCE * 2.0)
        region = (
            int(position.x() - CLICK_SELECTION_TOLERANCE),
            int(position.y() - CLICK_SELECTION_TOLERANCE),
            region_size,
            region_size,
        )
        try:
            return list(self.itemsAt(region))
        except Exception:
            return []


class GlbViewerWidget(QWidget):
    wall_selected = Signal(int, int, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model: GeneratedModel | None = None
        self.grid_item: gl.GLGridItem | None = None
        self.mesh_item: gl.GLMeshItem | None = None
        self.textured_wall_items: list[gl.GLImageItem] = []
        self.wall_by_item_id: dict[int, PreviewTexturedWall] = {}

        self._build_ui()
        self._populate_scene()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.view = SelectableGLViewWidget()
        self.view.setBackgroundColor((24, 24, 28))
        self.view.items_clicked.connect(self._handle_view_items_clicked)
        layout.addWidget(self.view, 1)

    def set_model(self, model: GeneratedModel, preserve_camera: bool = False) -> None:
        camera_state = self._capture_camera_state() if preserve_camera else None
        self.model = model
        self._populate_scene()
        if camera_state is not None:
            self._restore_camera_state(camera_state)

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
        self._add_textured_wall_items()

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

    def _add_textured_wall_items(self) -> None:
        if self.model is None:
            return

        for textured_wall in self.model.preview_textured_walls:
            self._add_textured_wall_item(textured_wall, offset_sign=1.0)
            self._add_textured_wall_item(textured_wall, offset_sign=-1.0)

    def _add_textured_wall_item(
        self,
        textured_wall: PreviewTexturedWall,
        offset_sign: float,
    ) -> None:
        texture_rgba = np.asarray(textured_wall.texture_rgba, dtype=np.ubyte)
        if texture_rgba.ndim != 3 or texture_rgba.shape[2] != 4:
            return

        image_item = gl.GLImageItem(
            texture_rgba,
            smooth=True,
            glOptions="opaque",
        )
        image_item.setTransform(
            _build_textured_wall_transform(
                textured_wall=textured_wall,
                offset_sign=offset_sign,
            )
        )
        self.view.addItem(image_item)
        self.textured_wall_items.append(image_item)
        self.wall_by_item_id[id(image_item)] = textured_wall

    def _handle_view_items_clicked(self, clicked_items: list[object]) -> None:
        for clicked_item in clicked_items:
            textured_wall = self.wall_by_item_id.get(id(clicked_item))
            if textured_wall is None:
                continue

            self.wall_selected.emit(
                textured_wall.level_index,
                textured_wall.room_index,
                textured_wall.wall_key,
            )
            return

    def _set_default_camera(self) -> None:
        self.view.opts["center"] = QVector3D(0.0, 0.0, 0.0)
        self.view.setCameraPosition(distance=18.0, elevation=28.0, azimuth=-40.0)

    def _clear_scene(self) -> None:
        if not hasattr(self, "view"):
            return

        self.view.clear()
        self.grid_item = None
        self.mesh_item = None
        self.textured_wall_items = []
        self.wall_by_item_id = {}

    def _capture_camera_state(self) -> dict[str, object]:
        camera_state: dict[str, object] = {}
        for key in CAMERA_STATE_KEYS:
            if key not in self.view.opts:
                continue

            value = self.view.opts[key]
            if isinstance(value, QVector3D):
                camera_state[key] = QVector3D(value)
            else:
                camera_state[key] = value

        return camera_state

    def _restore_camera_state(self, camera_state: dict[str, object]) -> None:
        self.view.opts.update(camera_state)
        self.view.update()


# ### Transform helpers ###
def _build_textured_wall_transform(
    textured_wall: PreviewTexturedWall,
    offset_sign: float,
) -> Transform3D:
    texture_width = max(1.0, float(textured_wall.texture_rgba.shape[0]))
    texture_height = max(1.0, float(textured_wall.texture_rgba.shape[1]))
    start_point = np.asarray(textured_wall.start_point, dtype=float)
    end_point = np.asarray(textured_wall.end_point, dtype=float)
    wall_vector = end_point - start_point
    wall_length = float(np.linalg.norm(wall_vector[:2]))
    if wall_length <= 1e-6:
        return Transform3D()

    wall_normal = np.array(
        [-wall_vector[1] / wall_length, wall_vector[0] / wall_length, 0.0],
        dtype=float,
    )
    if offset_sign >= 0.0:
        origin = start_point + wall_normal * TEXTURE_PREVIEW_OFFSET_METERS
        z_axis = wall_normal
    else:
        origin = start_point - wall_normal * TEXTURE_PREVIEW_OFFSET_METERS
        z_axis = -wall_normal

    x_axis = wall_vector / texture_width
    y_axis = np.array(
        [0.0, 0.0, float(textured_wall.height_meters) / texture_height],
        dtype=float,
    )

    return Transform3D(
        [
            [x_axis[0], y_axis[0], z_axis[0], origin[0]],
            [x_axis[1], y_axis[1], z_axis[1], origin[1]],
            [x_axis[2], y_axis[2], z_axis[2], origin[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _get_point_distance(first_point: QPointF, second_point: QPointF) -> float:
    delta = first_point - second_point
    return float((delta.x() ** 2 + delta.y() ** 2) ** 0.5)
