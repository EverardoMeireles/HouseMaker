"""Default sunlight, initial-direction editing and persisted orbit controls."""

# ### Imports ###
from __future__ import annotations

from dataclasses import replace

from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
)

from housemaker.directional_light_state import (
    MAX_SUN_ORBIT_SPEED_DEGREES_PER_SECOND,
    MIN_SUN_ORBIT_SPEED_DEGREES_PER_SECOND,
    SunOrbitData,
    create_default_sun_light,
)
from housemaker.level_coordinates import (
    build_level_base_z_lookup,
    level_image_to_world_xy,
)

# ### Constants ###
SUNLIGHT_SCENE_CLEARANCE_METERS = 2.0


# ### Default scene and controls ###
class SunlightWorkspaceMixin:
    def _init_sunlight_state(self) -> None:
        self.directional_lights = (create_default_sun_light(),)
        self.sun_orbit = SunOrbitData()
        self._is_editing_sunlight_direction = False

    def _ensure_default_sunlight(self) -> None:
        if not self.directional_lights:
            self.directional_lights = (create_default_sun_light(),)

    def _position_sunlight_above_levels(self) -> bool:
        """Keep the primary marker above the built house without rotating it."""

        if not self.directional_lights:
            return False
        occupied_levels = tuple(
            level
            for level in self.levels
            if level.vertex_data.vertices
            or level.rooms
            or level.editable_surfaces
            or level.landings
        )
        if not occupied_levels:
            return False
        base_z = build_level_base_z_lookup(self.levels)
        ceiling_z = max(
            base_z[level.index]
            + max(
                (
                    level.height_meters,
                    *(room.height_meters for room in level.rooms),
                )
            )
            for level in occupied_levels
        )
        sun = self.directional_lights[0]
        position_x, position_y, _position_z = sun.position
        # Center a newly created/default marker once; later plan dragging and
        # saved custom XY positions remain user-controlled.
        if (
            sun.light_id == "directional-light-1"
            and sun.name == "Sunlight"
            and sun.position == (0.0, 0.0, 3.0)
        ):
            points = tuple(
                level_image_to_world_xy(level, vertex.x, vertex.y)
                for level in occupied_levels
                for vertex in level.vertex_data.vertices
            )
            if points:
                position_x = (
                    min(x for x, _y in points) + max(x for x, _y in points)
                ) / 2.0
                position_y = (
                    min(y for _x, y in points) + max(y for _x, y in points)
                ) / 2.0
        position_z = ceiling_z + SUNLIGHT_SCENE_CLEARANCE_METERS / self.global_scale.z
        moved = sun.with_position((position_x, position_y, position_z))
        if moved == sun:
            return False
        self.directional_lights = (moved, *self.directional_lights[1:])
        return True

    def _add_sunlight_controls(self, layout) -> None:
        self.add_directional_light_button = QPushButton(
            "Edit sunlight initial direction"
        )
        self.add_directional_light_button.setObjectName(
            "edit-sunlight-direction-button"
        )
        self.add_directional_light_button.setCheckable(True)
        self.add_directional_light_button.setMinimumHeight(40)
        self.add_directional_light_button.toggled.connect(
            self._handle_add_directional_light_clicked
        )
        self.edit_sunlight_direction_button = self.add_directional_light_button
        layout.addWidget(self.add_directional_light_button)

        row = QHBoxLayout()
        self.enable_sun_orbit_checkbox = QCheckBox("Enable sun orbit")
        self.enable_sun_orbit_checkbox.setObjectName("enable-sun-orbit-checkbox")
        self.enable_sun_orbit_checkbox.toggled.connect(
            self._handle_sun_orbit_enabled_changed
        )
        row.addWidget(self.enable_sun_orbit_checkbox)
        self.sun_orbit_speed_label = QLabel("Orbit speed")
        row.addWidget(self.sun_orbit_speed_label)
        self.sun_orbit_speed_spinbox = QDoubleSpinBox()
        self.sun_orbit_speed_spinbox.setObjectName("sun-orbit-speed-spinbox")
        self.sun_orbit_speed_spinbox.setRange(
            MIN_SUN_ORBIT_SPEED_DEGREES_PER_SECOND,
            MAX_SUN_ORBIT_SPEED_DEGREES_PER_SECOND,
        )
        self.sun_orbit_speed_spinbox.setDecimals(1)
        self.sun_orbit_speed_spinbox.setSingleStep(0.5)
        self.sun_orbit_speed_spinbox.setSuffix(" °/s")
        self.sun_orbit_speed_spinbox.setToolTip(
            "Sun direction rotation around the vertical Z axis, in degrees per second."
        )
        self.sun_orbit_speed_spinbox.valueChanged.connect(
            self._handle_sun_orbit_speed_changed
        )
        row.addWidget(self.sun_orbit_speed_spinbox)
        row.addStretch(1)
        layout.addLayout(row)
        self._sync_sunlight_controls()

    def _sync_sunlight_controls(self) -> None:
        checkbox = self.enable_sun_orbit_checkbox
        spinbox = self.sun_orbit_speed_spinbox
        for control, value in (
            (checkbox, self.sun_orbit.enabled),
            (spinbox, self.sun_orbit.speed_degrees_per_second),
        ):
            blocked = control.blockSignals(True)
            if control is checkbox:
                checkbox.setChecked(value)
            else:
                spinbox.setValue(value)
            control.blockSignals(blocked)
        self.sun_orbit_speed_label.setVisible(self.sun_orbit.enabled)
        spinbox.setVisible(self.sun_orbit.enabled)
        self._refresh_sunlight_control_tooltips()

    # ### Initial direction editing and live orbit configuration ###
    def _handle_sunlight_preview_requested(self, light_id: str) -> None:
        """A real marker click previews orbit instead of editing its start."""

        if not any(light.light_id == light_id for light in self.directional_lights):
            return
        if self._is_editing_sunlight_direction:
            self._set_sunlight_direction_editing(False)
        self._handle_directional_light_selection_changed(light_id)

    def _set_sunlight_direction_editing(self, enabled: bool) -> None:
        self._is_editing_sunlight_direction = bool(enabled)
        if enabled:
            self._ensure_default_sunlight()
            self._selected_directional_light_id = self.directional_lights[0].light_id
        self._set_add_directional_light_button_checked(bool(enabled))
        self.viewer.set_sun_direction_editing(bool(enabled))
        self._sync_directional_light_views()
        if enabled:
            if not self._external_scene_3d_host.is_active:
                self.workspace_tabs.setCurrentWidget(self.scene_3d_workspace)
            self.viewer.release_first_person_pointer_capture()
            self.viewer.view.setFocus()

    def _sync_sunlight_viewer_configuration(self) -> None:
        self.viewer.set_sun_orbit(self.sun_orbit)
        self.viewer.set_sun_direction_editing(self._is_editing_sunlight_direction)

    def _handle_sun_orbit_enabled_changed(self, enabled: bool) -> None:
        if enabled:
            self._ensure_default_sunlight()
        self.sun_orbit = replace(self.sun_orbit, enabled=bool(enabled))
        self._sync_sunlight_controls()
        self._sync_directional_light_views()

    def _handle_sun_orbit_speed_changed(self, speed: float) -> None:
        self.sun_orbit = replace(self.sun_orbit, speed_degrees_per_second=float(speed))
        self._sync_sunlight_viewer_configuration()

    def _refresh_sunlight_control_tooltips(self) -> None:
        self.add_directional_light_button.setToolTip(
            "Edit the existing sun's initial direction with the RGB rotation rings. "
            "Orbit pauses and returns to the initial direction while editing; "
            "click again to resume it. Ray guides show where the sunlight travels. "
            "The editing handle's position does not affect global sunlight."
        )
        self.enable_sun_orbit_checkbox.setToolTip(
            "Rotate the primary sun around the house's vertical Z axis, keeping its elevation. "
            "Click the sun marker to preview its orbit at the configured speed. "
            "The initial direction and orbit settings are saved with the project."
        )
