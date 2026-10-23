"""Persistent floor-relative FPS starting points for progressive exports."""

# ### Imports ###
from __future__ import annotations

from dataclasses import replace

from PySide6.QtWidgets import QHBoxLayout, QPushButton

from housemaker.lazy_loading_state import CameraStartPoint
from housemaker.level_coordinates import (
    build_level_base_z_lookup,
    level_image_to_world_xy,
    level_world_to_image_xy,
)


# ### Starting-point controls ###
class CameraStartWorkspaceMixin:
    def _init_camera_start_state(self) -> None:
        self._placing_camera_start = False

    def _add_camera_start_controls(self, layout) -> None:
        self.set_fps_start_button = QPushButton("Set FPS starting point")
        self.set_fps_start_button.setObjectName("set-fps-starting-point-button")
        self.set_fps_start_button.setCheckable(True)
        self.set_fps_start_button.setToolTip(
            "Click the plan or a floor in the 3D window. The initial facing direction "
            "uses the current FPS camera; right-click or Escape cancels."
        )
        self.set_fps_start_button.toggled.connect(self._handle_camera_start_toggled)
        layout.addWidget(self.set_fps_start_button)
        row = QHBoxLayout()
        self.use_current_fps_start_button = QPushButton("Use current FPS position")
        self.use_current_fps_start_button.setObjectName("use-current-fps-start-button")
        self.use_current_fps_start_button.setToolTip(
            "Store the current FPS position and viewing direction as the export start."
        )
        self.use_current_fps_start_button.clicked.connect(self._use_current_fps_start)
        self.clear_fps_start_button = QPushButton("Clear")
        self.clear_fps_start_button.setObjectName("clear-fps-start-button")
        self.clear_fps_start_button.setToolTip(
            "Use automatic starting-room selection again."
        )
        self.clear_fps_start_button.clicked.connect(self._clear_camera_start)
        row.addWidget(self.use_current_fps_start_button)
        row.addWidget(self.clear_fps_start_button)
        layout.addLayout(row)
        self.canvas.camera_start_placement_requested.connect(
            self._place_canvas_camera_start
        )
        self.viewer.camera_start_placed.connect(self._place_world_camera_start)
        self.canvas.camera_start_placement_cancelled.connect(
            self._cancel_camera_start_placement
        )
        self.viewer.camera_start_placement_cancelled.connect(
            self._cancel_camera_start_placement
        )
        self._sync_camera_start_marker()

    # ### One-shot placement ###
    def _handle_camera_start_toggled(self, checked: bool) -> None:
        if not checked:
            self._cancel_camera_start_placement()
            return
        self._placing_camera_start = True
        self.canvas.start_camera_start_placement()
        floor_z = build_level_base_z_lookup(self.levels).get(
            self.current_level.index, 0.0
        )
        self.viewer.begin_camera_start_placement(fallback_plane_z=floor_z)
        self.viewer.set_surface_tools_status(
            "Click the plan or a floor in the 3D window to set the FPS starting point. "
            "Right-click or Escape cancels."
        )

    def _cancel_camera_start_placement(self) -> None:
        self._placing_camera_start = False
        self.canvas.cancel_camera_start_placement(notify=False)
        self.viewer.cancel_camera_start_placement(notify=False)
        if hasattr(self, "set_fps_start_button"):
            blocked = self.set_fps_start_button.blockSignals(True)
            self.set_fps_start_button.setChecked(False)
            self.set_fps_start_button.blockSignals(blocked)

    def _place_canvas_camera_start(self, image_x: float, image_y: float) -> None:
        pose = self.viewer.get_first_person_camera_pose()
        self._store_camera_start(
            CameraStartPoint(
                self.current_level.index,
                image_x,
                image_y,
                yaw_degrees=pose.yaw_degrees,
                pitch_degrees=max(-89.9, min(89.9, pose.pitch_degrees)),
            )
        )

    def _place_world_camera_start(self, point) -> None:
        x, y, z = (float(value) for value in point)
        level = self._camera_start_level_for_height(z)
        if level is None:
            return
        image_x, image_y = level_world_to_image_xy(level, x, y)
        pose = self.viewer.get_first_person_camera_pose()
        floor_z = build_level_base_z_lookup(self.levels)[level.index]
        self._store_camera_start(
            CameraStartPoint(
                level.index,
                image_x,
                image_y,
                z - floor_z,
                pose.yaw_degrees,
                max(-89.9, min(89.9, pose.pitch_degrees)),
            )
        )

    def _camera_start_level_for_height(self, z: float):
        levels = tuple(level for level in self.levels if level.include_in_export)
        if not levels:
            return None
        floor_z = build_level_base_z_lookup(self.levels)
        # On stairs or a landing, bind to the closest floor below the point.
        below = tuple(level for level in levels if floor_z[level.index] <= z + 0.05)
        if below:
            return max(below, key=lambda level: floor_z[level.index])
        return min(levels, key=lambda level: abs(floor_z[level.index] - z))

    def _use_current_fps_start(self) -> None:
        pose = self.viewer.get_first_person_camera_pose()
        eye_height = self._generation_settings.first_person_eye_height_meters
        self._place_world_camera_start(
            (
                pose.x,
                pose.y,
                pose.z - eye_height / self.global_scale.z,
            )
        )

    def _clear_camera_start(self) -> None:
        self._cancel_camera_start_placement()
        self.runtime_loading = replace(self.runtime_loading, start_point=None)
        self._sync_camera_start_marker()

    def _store_camera_start(self, point: CameraStartPoint) -> None:
        self._cancel_camera_start_placement()
        previous = self.runtime_loading
        self.runtime_loading = replace(previous, start_point=point)
        self._sync_camera_start_marker()
        self.viewer.set_surface_tools_status("FPS starting point saved for export.")

    # ### Lightweight marker synchronization ###
    def _sync_camera_start_marker(self) -> None:
        if not hasattr(self, "viewer") or not hasattr(self, "canvas"):
            return
        point = self.runtime_loading.start_point
        level = next(
            (
                level
                for level in self.levels
                if point is not None and level.index == point.level_index
            ),
            None,
        )
        world = None
        if level is not None:
            xy = level_image_to_world_xy(level, point.image_x, point.image_y)
            floor_z = build_level_base_z_lookup(self.levels)[level.index]
            world = (*xy, floor_z + point.height_offset_meters)
        self.canvas.set_camera_start_marker(
            (point.image_x, point.image_y)
            if level is not None and level.index == self.current_level.index
            else None,
            point.yaw_degrees if point is not None else 0.0,
        )
        self.viewer.set_camera_start_marker(
            world,
            point.yaw_degrees if point is not None else 0.0,
        )
        if hasattr(self, "clear_fps_start_button"):
            self.clear_fps_start_button.setEnabled(point is not None)
