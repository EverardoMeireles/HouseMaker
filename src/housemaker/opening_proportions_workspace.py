"""Manual opening proportions, height-only scene edits and shared undo."""

# ### Imports ###
from __future__ import annotations

import math
from dataclasses import dataclass, replace

from PySide6.QtCore import QTimer
from PySide6.QtGui import QCursor, QKeySequence, QShortcut

from housemaker.canvas_openings import (
    CANVAS_OPENING_DOORWAY,
    CANVAS_OPENING_WINDOW,
    CanvasOpeningReference,
)
from housemaker.models import (
    DOORWAY_SHAPE_ARCH,
    DOORWAY_SHAPE_RECTANGULAR,
    DoorwayData,
    WindowData,
)
from housemaker.opening_proportions_geometry import measure_opening_proportions


# ### Height-only application and undo data ###
@dataclass(frozen=True)
class OpeningProportionsUndoState:
    level_index: int
    opening_kind: str
    opening: WindowData | DoorwayData


def _axis_stretch(axis, scale) -> float:
    return math.sqrt(
        sum((value * factor) ** 2 for value, factor in zip(axis, scale.factors))
    )


def opening_physical_dimensions(target, scale) -> tuple[float, float]:
    return (
        target.wall_width_meters
        * target.bounds.horizontal_span
        * _axis_stretch(target.wall_tangent_world, scale),
        target.wall_height_meters
        * target.bounds.vertical_span
        * _axis_stretch(target.vertical_axis_world, scale),
    )


def build_opening_proportions_update(opening, target, measurement, scale):
    """Preserve width, sill, placement and topology; never change project scale."""
    ratio, arch_ratio = measurement.height_to_width, measurement.arch_height_to_width
    if (
        isinstance(ratio, bool)
        or isinstance(arch_ratio, bool)
        or not math.isfinite(ratio)
        or ratio <= 0
        or not math.isfinite(arch_ratio)
        or arch_ratio < 0
        or arch_ratio >= ratio
    ):
        raise ValueError("The marked opening proportions are invalid.")
    width, _old_height = opening_physical_dimensions(target, scale)
    vertical_stretch = _axis_stretch(target.vertical_axis_world, scale)
    height = width * ratio / vertical_stretch
    remaining = target.wall_height_meters * (1 - target.bounds.bottom_ratio)
    if height < target.minimum_height_meters - 1e-7:
        raise ValueError("The marked opening would be too short.")
    if height > remaining + 1e-7:
        raise ValueError(
            "The requested height does not fit above the current sill or threshold. Lower the opening or adjust the wall height first."
        )
    height = min(height, remaining)
    if isinstance(opening, WindowData):
        if (
            target.reference.kind != CANVAS_OPENING_WINDOW
            or target.reference.stable_id != opening.window_id
        ):
            raise ValueError("The selected window frame no longer matches its wall.")
        if arch_ratio:
            raise ValueError("Window frame outlines cannot include an arch.")
        return replace(
            opening, top_ratio=opening.bottom_ratio + height / target.wall_height_meters
        )
    if (
        not isinstance(opening, DoorwayData)
        or target.reference.kind != CANVAS_OPENING_DOORWAY
        or target.reference.stable_id != opening.doorway_id
    ):
        raise ValueError("The selected doorway no longer matches its wall.")
    arch_rise = width * arch_ratio / vertical_stretch
    amount = opening.arch_amount
    if arch_rise > 1e-8:
        # Doorway cross sections use the original width; level scale and global
        # X/Y stretch its horizontal coordinates, not its vertical arch rise.
        amount = arch_rise / min(opening.width_meters / 2, height)
        if amount > 1 + 1e-7:
            raise ValueError(
                "The requested arch exceeds this doorway profile's supported rise. Reduce the arch peak or check the current horizontal and vertical scales."
            )
        amount = min(amount, 1.0)
    return replace(
        opening,
        height_meters=height,
        shape=DOORWAY_SHAPE_ARCH if arch_rise > 1e-8 else DOORWAY_SHAPE_RECTANGULAR,
        arch_amount=amount,
    )


def opening_target_with_proportions(opening, target):
    """Describe an edited profile without changing scene data or its width."""
    if isinstance(opening, WindowData):
        return replace(
            target,
            bounds=replace(
                target.bounds,
                bottom_ratio=opening.bottom_ratio,
                top_ratio=opening.top_ratio,
            ),
        )
    return replace(
        target,
        bounds=replace(
            target.bounds,
            top_ratio=target.bounds.bottom_ratio
            + opening.height_meters / target.wall_height_meters,
        ),
        arch_amount=opening.arch_amount
        if opening.shape == DOORWAY_SHAPE_ARCH
        else None,
    )


def _opening_reference_key(reference):
    if not isinstance(reference, CanvasOpeningReference):
        return None
    return reference.kind, reference.level_index, reference.stable_id


# ### Stable opening resolution and lightweight preview ###
class OpeningProportionsWorkspaceMixin:
    def _init_opening_proportions(self) -> None:
        self._opening_proportions_dialog = None
        self._opening_proportions_reference = None
        self._opening_proportions_target = None

    def _find_proportions_opening(self, reference):
        if not isinstance(reference, CanvasOpeningReference) or not reference.stable_id:
            raise ValueError("Select an existing doorway or window frame.")
        level = next(
            (level for level in self.levels if level.index == reference.level_index),
            None,
        )
        if level is None:
            raise ValueError("The selected opening's level no longer exists.")
        items = (
            level.windows if reference.kind == CANVAS_OPENING_WINDOW else level.doorways
        )
        for index, opening in enumerate(items):
            identity = (
                opening.window_id
                if isinstance(opening, WindowData)
                else opening.doorway_id
            )
            if identity == reference.stable_id:
                return level, index, opening
        raise ValueError("The selected opening no longer exists.")

    def _resolve_proportions_target(self, reference):
        level, index, opening = self._find_proportions_opening(reference)
        targets = self._build_opening_target_lookup_for_levels(
            self.levels, include_excluded_levels=True
        )
        target = next(
            (
                target
                for target in targets.values()
                if target.reference.kind == reference.kind
                and target.reference.level_index == reference.level_index
                and target.reference.stable_id == reference.stable_id
            ),
            None,
        )
        if target is None:
            raise ValueError(
                "This opening no longer has a usable wall or slope surface."
            )
        return level, index, opening, target

    def _settle_opening_proportions_edits(self) -> None:
        self._commit_pending_level_transform_update()
        self._commit_pending_canvas_surface_mesh_update()
        self._commit_pending_sloped_ceiling_mesh_update()
        self._commit_pending_architectural_trim_mesh_update()
        self._commit_pending_wall_vertex_update()
        self._commit_pending_doorway_mesh_update()

    def _open_opening_proportions(self, reference: object) -> None:
        dialog = self._opening_proportions_dialog
        if (
            dialog is not None
            and dialog.isVisible()
            and _opening_reference_key(self._opening_proportions_reference)
            == _opening_reference_key(reference)
        ):
            self._focus_opening_proportions_dialog(dialog)
            return
        try:
            self._settle_opening_proportions_edits()
            level, index, opening, target = self._resolve_proportions_target(reference)
        except (TypeError, ValueError) as error:
            self.viewer.set_window_tools_status(str(error))
            return
        self.viewer.clear_opening_proportions_preview()
        previous_source = dialog.current_measurement() if dialog is not None else {}
        was_visible = dialog is not None and dialog.isVisible()
        if dialog is None:
            from housemaker.opening_proportions_dialog import OpeningProportionsDialog

            dialog = OpeningProportionsDialog(self)
            self._opening_proportions_dialog = dialog
            dialog.outline_changed.connect(self._preview_opening_proportions)
            dialog.apply_requested.connect(self._apply_opening_proportions)
            dialog.finished.connect(
                lambda _result: self._opening_proportions_dialog_closed(dialog)
            )
            undo = QShortcut(QKeySequence.StandardKey.Undo, dialog)
            undo.activated.connect(self._handle_canvas_undo_requested)
            dialog._canvas_undo_shortcut = undo
        self._opening_proportions_reference = target.reference
        self._opening_proportions_target = target
        width, height = opening_physical_dimensions(target, self.global_scale)
        arch_rise = 0.0
        if isinstance(opening, DoorwayData) and opening.shape == DOORWAY_SHAPE_ARCH:
            arch_rise = (
                min(opening.width_meters / 2, opening.height_meters)
                * opening.arch_amount
                * _axis_stretch(target.vertical_axis_world, self.global_scale)
            )
        dialog.set_target(
            f"{level.display_name} — {'Doorway' if isinstance(opening, DoorwayData) else 'Window frame'} {index + 1}",
            isinstance(opening, DoorwayData),
            width,
            height,
            arch_rise,
            target_key=_opening_reference_key(target.reference),
        )
        if not was_visible:
            path = previous_source.get("source_path")
            frame = previous_source.get("frame_index", 0)
            if not path:
                source = self.merged_generation_workspace._current_video_source()
                if source is not None:
                    path = source.metadata.path
                    frame = (
                        self.merged_generation_workspace._current_video_frame_index()
                    )
            if path:
                try:
                    dialog.load_source(path, frame_index=frame or 0)
                except (OSError, ValueError, IndexError) as error:
                    dialog.show_error(
                        f"The video could not be opened: {error}. "
                        "Load another video to continue."
                    )
        self._focus_opening_proportions_dialog(dialog)

    # ### Reusable editor focus and cursor handoff ###
    def _focus_opening_proportions_dialog(self, dialog) -> None:
        # Release mouse-look before warping the cursor, even if native focus
        # activation arrives later than the queued cursor handoff.
        self.viewer.view.release_first_person_pointer_capture()
        if dialog.isMinimized():
            dialog.showNormal()
        else:
            dialog.show()
        dialog.raise_()
        dialog.activateWindow()

        def center_cursor():
            if dialog is not self._opening_proportions_dialog or not dialog.isVisible():
                return
            dialog.video_view.setFocus()
            QCursor.setPos(dialog.mapToGlobal(dialog.rect().center()))

        QTimer.singleShot(0, center_cursor)

    def _opening_proportions_dialog_closed(self, dialog) -> None:
        if dialog is not self._opening_proportions_dialog:
            return
        self.viewer.clear_opening_proportions_preview()

        def return_to_viewer():
            if dialog is not self._opening_proportions_dialog or dialog.isVisible():
                return
            view = self.viewer.view
            host = view.window()
            if host.isMinimized():
                host.showNormal()
            host.raise_()
            host.activateWindow()
            view.focus_navigation()
            QCursor.setPos(view.mapToGlobal(view.rect().center()))

        QTimer.singleShot(0, return_to_viewer)

    def _opening_proportions_measurement(self, payload):
        if not isinstance(payload, dict):
            raise TypeError("Outline an opening on the video first.")
        reference = self._opening_proportions_reference
        return measure_opening_proportions(
            payload.get("outline", ()),
            payload.get("frame_size", ()),
            arch_apex=payload.get("arch_apex"),
            is_doorway=reference is not None
            and reference.kind == CANVAS_OPENING_DOORWAY,
        )

    def _preview_opening_proportions(self, payload) -> None:
        dialog, target = (
            self._opening_proportions_dialog,
            self._opening_proportions_target,
        )
        if dialog is None or target is None:
            return
        try:
            _level, _index, opening = self._find_proportions_opening(
                self._opening_proportions_reference
            )
            measurement = self._opening_proportions_measurement(payload)
            updated = build_opening_proportions_update(
                opening, target, measurement, self.global_scale
            )
            if payload.get("outline_edited", False):
                self.viewer.set_opening_proportions_preview(
                    opening_target_with_proportions(updated, target)
                )
            else:
                self.viewer.clear_opening_proportions_preview()
            dialog.set_result_text(measurement.warning, valid=True)
        except (TypeError, ValueError) as error:
            self.viewer.clear_opening_proportions_preview()
            dialog.show_error(str(error))

    # ### Explicit application, normal scene synchronization and undo ###
    def _publish_proportions_opening(self, level, index, opening, target) -> None:
        if isinstance(opening, WindowData):
            level.windows[index] = opening
            self._sync_viewer_window_snapshot(level)
            self._refresh_canvas_windows_for_level(level)
        else:
            level.doorways[index] = opening
            self._viewer_doorways_by_level_index[level.index] = self._copy_doorways(
                level.doorways
            )
            if level is self.current_level:
                self.canvas.doorways = level.doorways
                self._handle_canvas_doorway_selection_changed(index)
                self.canvas.update()
        updated_target = opening_target_with_proportions(opening, target)
        self._canvas_opening_targets_by_key[updated_target.key] = updated_target
        self.viewer.update_canvas_opening_target(updated_target)
        reference = self._opening_proportions_reference
        if (
            reference is not None
            and reference.kind == updated_target.reference.kind
            and reference.level_index == updated_target.reference.level_index
            and reference.stable_id == updated_target.reference.stable_id
        ):
            # Undo may restore another opening while this editor stays open.
            # Its cached frame must continue to describe the edited opening.
            self._opening_proportions_target = updated_target
        self._schedule_viewer_preview_refresh(preserve_camera=True)

    def _apply_opening_proportions(self, payload) -> None:
        dialog = self._opening_proportions_dialog
        if dialog is None:
            return
        try:
            self._settle_opening_proportions_edits()
            level, index, opening, target = self._resolve_proportions_target(
                self._opening_proportions_reference
            )
            measurement = self._opening_proportions_measurement(payload)
            updated = build_opening_proportions_update(
                opening, target, measurement, self.global_scale
            )
            if updated == opening:
                self.viewer.clear_opening_proportions_preview()
                dialog.set_result_text("The opening already matches these proportions.")
                return
            self._record_canvas_undo_state(
                OpeningProportionsUndoState(
                    level.index, target.reference.kind, replace(opening)
                )
            )
            try:
                self._publish_proportions_opening(level, index, updated, target)
            except Exception:
                self._publish_proportions_opening(level, index, opening, target)
                self._canvas_undo_stack.pop()
                raise
            self._preview_opening_proportions(payload)
            self.viewer.clear_opening_proportions_preview()
            self.viewer.set_window_tools_status(
                "Opening proportions applied. Ctrl+Z undoes this change."
            )
        except (TypeError, ValueError, RuntimeError) as error:
            dialog.show_error(str(error))

    def _restore_opening_proportions_undo_state(self, state) -> None:
        identity = (
            state.opening.window_id
            if isinstance(state.opening, WindowData)
            else state.opening.doorway_id
        )
        reference = CanvasOpeningReference(
            state.opening_kind, state.level_index, 0, identity
        )
        level, index, _opening, target = self._resolve_proportions_target(reference)
        self._publish_proportions_opening(level, index, replace(state.opening), target)
        self.viewer.clear_opening_proportions_preview()
        self.viewer.set_window_tools_status("Opening height and arch restored.")

    # ### Project replacement and media cleanup ###
    def _reset_opening_proportions(self) -> None:
        dialog = self._opening_proportions_dialog
        self._opening_proportions_dialog = None
        self._opening_proportions_reference = None
        self._opening_proportions_target = None
        if hasattr(self, "viewer"):
            self.viewer.clear_opening_proportions_preview()
        if dialog is not None:
            dialog.dispose()
            dialog.close()
            dialog.deleteLater()
