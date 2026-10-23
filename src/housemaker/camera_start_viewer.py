"""Independent one-click FPS spawn placement and non-interactive markers."""

# ### Imports ###
from __future__ import annotations

import math
import weakref

import numpy as np
import pyqtgraph.opengl as gl
from OpenGL import GL
from PySide6.QtCore import QEvent, QObject, Qt
from shiboken6 import isValid as is_valid_qt_object


# ### Placement event filter ###
class _CameraStartPlacementFilter(QObject):
    """Consume only this temporary tool's clicks, before normal viewer tools."""

    def __init__(self, viewer):
        super().__init__(viewer)
        self._viewer_reference = weakref.ref(viewer)

    def eventFilter(self, watched, event) -> bool:  # type: ignore[override]
        # Deferred Qt events can outlive Python wrapper cleanup. The filter
        # must also avoid retaining the owning widget in a reference cycle.
        reference = getattr(self, "_viewer_reference", None)
        viewer = reference() if reference is not None else None
        if (
            viewer is None
            or not is_valid_qt_object(viewer)
            or not getattr(viewer, "_camera_start_placement_active", False)
        ):
            return False
        kind = event.type()
        if kind == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
            viewer.cancel_camera_start_placement()
            return True
        if kind == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.RightButton:
                viewer.cancel_camera_start_placement()
                return True
            if event.button() == Qt.MouseButton.LeftButton:
                viewer._update_camera_start_hover(event.position())
                viewer._camera_start_pressed = True
                return True
        if kind == QEvent.Type.MouseButtonRelease:
            if event.button() == Qt.MouseButton.LeftButton:
                if viewer._camera_start_pressed:
                    viewer._update_camera_start_hover(event.position())
                    point = viewer._camera_start_hover
                    viewer._camera_start_pressed = False
                    if point is not None:
                        viewer.cancel_camera_start_placement(notify=False)
                        viewer.camera_start_placed.emit(point)
                return True
            if event.button() == Qt.MouseButton.RightButton:
                return True
        if kind == QEvent.Type.MouseMove:
            viewer._update_camera_start_hover(event.position())
            return True
        if kind == QEvent.Type.Leave:
            viewer._camera_start_hover = None
            viewer._refresh_camera_start_items()
        return False


# ### Viewer API ###
class CameraStartViewerMixin:
    def _init_camera_start_viewer(self) -> None:
        self._camera_start_placement_active = False
        self._camera_start_plane_z = 0.0
        self._camera_start_hover = None
        self._camera_start_pressed = False
        self._camera_start_marker = None
        self._camera_start_marker_yaw = 0.0
        self._camera_start_marker_scale = None
        self._camera_start_items = []

    def _connect_camera_start_input(self) -> None:
        self._camera_start_event_filter = _CameraStartPlacementFilter(self)
        self.view.installEventFilter(self._camera_start_event_filter)

    @property
    def is_camera_start_placement_active(self) -> bool:
        return self._camera_start_placement_active

    def begin_camera_start_placement(self, *, fallback_plane_z: float = 0.0) -> bool:
        if not math.isfinite(fallback_plane_z):
            raise ValueError("The FPS starting floor must have a finite height.")
        self.cancel_landing_placement()
        self.cancel_directional_light_placement()
        self.cancel_tour_point_placement()
        self.cancel_object_placement()
        self.cancel_door_placement()
        self.cancel_architectural_trim_placement()
        self.cancel_window_placement(status_message=None)
        self.cancel_surface_vertex_placement()
        self._cancel_canvas_gizmo_drag()
        self.release_first_person_pointer_capture()
        self._camera_start_plane_z = float(fallback_plane_z)
        self._camera_start_placement_active = True
        self._camera_start_pressed = False
        self._camera_start_hover = None
        self.view.setCursor(Qt.CursorShape.CrossCursor)
        self.view.setToolTip(
            "Click a floor to set the FPS starting point. Right-click or Escape cancels."
        )
        return True

    def cancel_camera_start_placement(self, *, notify: bool = True) -> bool:
        if not self._camera_start_placement_active:
            return False
        self._camera_start_placement_active = False
        self._camera_start_hover = None
        self._camera_start_pressed = False
        self.view.unsetCursor()
        self.view.setToolTip("")
        self._refresh_camera_start_items()
        if notify:
            self.camera_start_placement_cancelled.emit()
        return True

    def set_camera_start_marker(self, position, yaw_degrees: float = 0.0) -> None:
        point = None if position is None else tuple(float(value) for value in position)
        if point is not None and (
            len(point) != 3 or not all(map(math.isfinite, point))
        ):
            raise ValueError("The FPS starting marker needs three finite coordinates.")
        if not math.isfinite(yaw_degrees):
            raise ValueError("The FPS starting direction must be finite.")
        scale = self.view.get_global_scale().factors
        if (
            point == self._camera_start_marker
            and float(yaw_degrees) == self._camera_start_marker_yaw
            and scale == self._camera_start_marker_scale
            and all(item in self.view.items for item in self._camera_start_items)
        ):
            return
        self._camera_start_marker = point
        self._camera_start_marker_yaw = float(yaw_degrees)
        self._camera_start_marker_scale = scale
        self._refresh_camera_start_items()

    # ### Scene hit and marker drawing ###
    def _update_camera_start_hover(self, position) -> None:
        ray = self.view.build_camera_ray(position)
        point = (
            None
            if ray is None
            else self._pick_scene_world_point(
                *ray, fallback_plane_z=self._camera_start_plane_z
            )
        )
        self._camera_start_hover = None if point is None else tuple(map(float, point))
        self._refresh_camera_start_items()

    def _refresh_camera_start_items(self) -> None:
        if not hasattr(self, "view"):
            return
        for item in self._camera_start_items:
            if item in self.view.items:
                self.view.removeItem(item)
        self._camera_start_items = []
        point = self._camera_start_hover or self._camera_start_marker
        if point is None:
            return
        center = np.asarray(point, dtype=float)
        factors = self.view.get_global_scale().factors
        radians = math.radians(self._camera_start_marker_yaw)
        tip = center + np.asarray(
            (
                math.cos(radians) * 0.5 / factors[0],
                math.sin(radians) * 0.5 / factors[1],
                0.0,
            )
        )
        color = (0.3, 0.85, 1.0, 1.0)
        items = (
            gl.GLScatterPlotItem(
                pos=np.asarray([center]), color=color, size=14, pxMode=True
            ),
            gl.GLLinePlotItem(pos=np.asarray([center, tip]), color=color, width=4.0),
        )
        for item in items:
            item.setGLOptions({GL.GL_DEPTH_TEST: False, GL.GL_BLEND: True})
            item.setDepthValue(10_030.0)
            self.view.addItem(item)
            self._camera_start_items.append(item)
        self.view.update()
