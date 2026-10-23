# ### Imports ###
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QWidget,
)

from housemaker.global_scale import (
    GLOBAL_SCALE_AXES,
    MAX_GLOBAL_MULTIPLIER,
    MIN_GLOBAL_MULTIPLIER,
    GlobalScale,
)
from housemaker.level_coordinates import build_level_base_z_lookup
from housemaker.models import (
    MAX_FLOOR_THICKNESS_METERS,
    MIN_FLOOR_THICKNESS_METERS,
)


# ### Undo state ###
@dataclass(frozen=True)
class CanvasGlobalScaleUndoState:
    scale: GlobalScale


# ### Project-wide multiplier controls ###
class GlobalScaleWorkspaceMixin:
    """Scale the displayed/exported building without rewriting its source data."""

    def _build_global_scale_controls(self, layout: QFormLayout) -> None:
        row = QWidget(self.level_dimensions_group)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)
        self.global_multiplier_spinboxes: dict[str, QDoubleSpinBox] = {}
        for axis in GLOBAL_SCALE_AXES:
            row_layout.addWidget(QLabel(axis.upper(), row))
            spinbox = QDoubleSpinBox(row)
            spinbox.setObjectName(f"global_{axis}_multiplier")
            spinbox.setRange(MIN_GLOBAL_MULTIPLIER, MAX_GLOBAL_MULTIPLIER)
            spinbox.setDecimals(4)
            spinbox.setSingleStep(0.05)
            spinbox.setValue(getattr(self.global_scale, axis))
            spinbox.setKeyboardTracking(False)
            spinbox.setToolTip(
                "Scale every level and its contents. X/Y follow the plan; Z is "
                "vertical. Original geometry and plan calibration are preserved."
            )
            spinbox.valueChanged.connect(
                lambda value, axis=axis: self._handle_global_multiplier_changed(
                    axis, value
                )
            )
            self.global_multiplier_spinboxes[axis] = spinbox
            row_layout.addWidget(spinbox, 1)
        layout.addRow("Global multiplier", row)
        self.global_multiplier_link_checkbox = QCheckBox("Link axes", self)
        self.global_multiplier_link_checkbox.setChecked(self.global_scale.linked)
        self.global_multiplier_link_checkbox.setToolTip(
            "Change all axes proportionally. Uncheck to adjust plan proportions "
            "and height independently. Applies to the entire project."
        )
        self.global_multiplier_link_checkbox.toggled.connect(
            self._handle_global_multiplier_link_changed
        )
        layout.addRow(self.global_multiplier_link_checkbox)

    def _handle_global_multiplier_changed(self, axis: str, value: float) -> None:
        try:
            scale = self.global_scale.with_axis(axis, value)
        except (TypeError, ValueError) as error:
            self._sync_global_scale_controls()
            self.viewer.set_surface_tools_status(str(error))
            return
        self._change_global_scale(scale)

    def _handle_global_multiplier_link_changed(self, checked: bool) -> None:
        self._change_global_scale(self.global_scale.with_linked(checked))

    def _change_global_scale(self, scale: GlobalScale) -> None:
        if scale == self.global_scale:
            return
        self._record_canvas_undo_state(CanvasGlobalScaleUndoState(self.global_scale))
        self._set_global_scale(scale)

    def _set_global_scale(self, scale: GlobalScale) -> None:
        self.global_scale = scale
        self._sync_global_scale_controls()
        self._sync_global_scale_views()

    def _restore_global_scale_undo_state(self, state: CanvasGlobalScaleUndoState) -> None:
        self._set_global_scale(state.scale)

    # ### Synchronization ###
    def _sync_global_scale_controls(self) -> None:
        for axis, spinbox in self.global_multiplier_spinboxes.items():
            with QSignalBlocker(spinbox):
                spinbox.setValue(getattr(self.global_scale, axis))
        with QSignalBlocker(self.global_multiplier_link_checkbox):
            self.global_multiplier_link_checkbox.setChecked(self.global_scale.linked)
        self._sync_scaled_level_dimensions()

    def _sync_scaled_level_dimensions(self) -> None:
        """The two meter fields show physical dimensions, not source units."""
        z = self.global_scale.z
        decimals = 4 if z < 1.0 else 2
        with QSignalBlocker(self.height_level_spinbox):
            self.height_level_spinbox.setDecimals(decimals)
            self.height_level_spinbox.setRange(0.1 * z, 100.0 * z)
            self.height_level_spinbox.setValue(self.current_level.height_meters * z)
        with QSignalBlocker(self.floor_thickness_spinbox):
            self.floor_thickness_spinbox.setDecimals(decimals)
            self.floor_thickness_spinbox.setRange(
                MIN_FLOOR_THICKNESS_METERS * z,
                MAX_FLOOR_THICKNESS_METERS * z,
            )
            self.floor_thickness_spinbox.setValue(
                self.current_level.floor_thickness_meters * z
            )

    def _sync_global_scale_views(self) -> None:
        self.canvas.set_global_scale(self.global_scale)
        self.viewer.set_global_scale(
            self.global_scale,
            floor_z=build_level_base_z_lookup(self.levels)[self.current_level.index],
        )
        if self._position_sunlight_above_levels():
            self._sync_directional_light_views()
        if self.tour_preview_viewer is not None:
            self.tour_preview_viewer.set_global_scale(self.global_scale)
        stair_index = self._editing_stair_index
        if stair_index is not None and 0 <= stair_index < len(self.stairs):
            self._sync_stair_calculated_values(
                self._staged_stair or self.stairs[stair_index]
            )

