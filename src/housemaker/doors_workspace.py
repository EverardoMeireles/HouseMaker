# ### Imports ###
from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pyqtgraph.opengl as gl
from PySide6.QtCore import QSignalBlocker, Qt, Signal, Slot
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from housemaker.door_state import (
    DOOR_SLOT_BACK_BODY,
    DOOR_SLOT_BACK_KNOB,
    DOOR_SLOT_BACK_LEFT_BODY,
    DOOR_SLOT_BACK_LEFT_KNOB,
    DOOR_SLOT_BODY,
    DOOR_SLOT_KNOB,
    DOOR_SLOT_LEFT_BODY,
    DOOR_SLOT_LEFT_KNOB,
    DoorDefinition,
    DoorLibraryData,
    DoorSlotData,
    GlassAreaData,
    WindowDefinition,
    door_slot_component,
    door_slot_counterpart,
    door_slot_display_name,
    door_slot_side,
    is_door_body_slot,
)
from housemaker.glb import GeneratedModel
from housemaker.viewer import GlbViewerWidget

# ### Constants ###
ITEM_KIND_ROLE = Qt.ItemDataRole.UserRole
ITEM_DOOR_ID_ROLE = Qt.ItemDataRole.UserRole + 1
ITEM_SLOT_ID_ROLE = Qt.ItemDataRole.UserRole + 2
ITEM_ASSET_KIND_ROLE = Qt.ItemDataRole.UserRole + 3
ITEM_GLASS_AREA_ID_ROLE = Qt.ItemDataRole.UserRole + 4

ITEM_KIND_DOOR = "door"
ITEM_KIND_WINDOW = "window"
ITEM_KIND_FACE = "face"
ITEM_KIND_SLOT = "slot"
ITEM_KIND_GLASS_AREA = "glass_area"

ASSET_KIND_DOOR = "door"
ASSET_KIND_WINDOW = "window"

SINGLE_SIDED_SLOT_ORDER = (
    DOOR_SLOT_BODY,
    DOOR_SLOT_KNOB,
)
DOUBLE_LEAF_SLOT_ORDER = (
    DOOR_SLOT_BODY,
    DOOR_SLOT_LEFT_BODY,
    DOOR_SLOT_KNOB,
    DOOR_SLOT_LEFT_KNOB,
)
DOUBLE_SIDED_SLOT_ORDER = (
    DOOR_SLOT_BODY,
    DOOR_SLOT_KNOB,
    DOOR_SLOT_BACK_BODY,
    DOOR_SLOT_BACK_KNOB,
)
DOUBLE_SIDED_DOUBLE_LEAF_SLOT_ORDER = (
    DOOR_SLOT_BODY,
    DOOR_SLOT_LEFT_BODY,
    DOOR_SLOT_KNOB,
    DOOR_SLOT_LEFT_KNOB,
    DOOR_SLOT_BACK_BODY,
    DOOR_SLOT_BACK_LEFT_BODY,
    DOOR_SLOT_BACK_KNOB,
    DOOR_SLOT_BACK_LEFT_KNOB,
)
DOOR_FACE_ORDER = ("front", "back")
DOOR_FACE_DISPLAY_NAMES = {
    "front": "Front",
    "back": "Back",
}
OPENING_PREVIEW_FRONT_ARROW_COLOR = (0.12, 0.88, 1.0, 1.0)
OPENING_PREVIEW_FRONT_ARROW_WIDTH_PIXELS = 5.0
OPENING_PREVIEW_FRONT_ARROW_LENGTH_RATIO = 0.4
OPENING_PREVIEW_FRONT_ARROW_MINIMUM_LENGTH_METERS = 0.35
OPENING_PREVIEW_FRONT_ARROW_MAXIMUM_LENGTH_METERS = 1.25
OPENING_PREVIEW_FRONT_ARROW_CLEARANCE_RATIO = 0.04
OPENING_PREVIEW_FRONT_ARROW_MINIMUM_CLEARANCE_METERS = 0.04
OPENING_PREVIEW_FRONT_ARROW_HEAD_LENGTH_RATIO = 0.3
OPENING_PREVIEW_FRONT_ARROW_HEAD_WIDTH_RATIO = 0.18
OPENING_PREVIEW_FRONT_ARROW_ROLE = "front_direction"


# ### Preview arrow geometry ###
def _build_preview_front_direction_arrow_positions(
    model: GeneratedModel,
) -> np.ndarray | None:
    """Build a floor-level arrow pointing toward canonical local front, -Y."""

    bounds = np.asarray(model.mesh.bounds, dtype=float)
    if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
        return None
    minimum, maximum = bounds
    extents = maximum - minimum
    if np.any(extents < 0.0):
        return None
    model_scale = float(np.max(extents))
    if not math.isfinite(model_scale) or model_scale <= 0.0:
        return None

    arrow_length = float(
        np.clip(
            model_scale * OPENING_PREVIEW_FRONT_ARROW_LENGTH_RATIO,
            OPENING_PREVIEW_FRONT_ARROW_MINIMUM_LENGTH_METERS,
            OPENING_PREVIEW_FRONT_ARROW_MAXIMUM_LENGTH_METERS,
        )
    )
    clearance = max(
        model_scale * OPENING_PREVIEW_FRONT_ARROW_CLEARANCE_RATIO,
        OPENING_PREVIEW_FRONT_ARROW_MINIMUM_CLEARANCE_METERS,
    )
    center_x = float((minimum[0] + maximum[0]) * 0.5)
    height = max(model_scale * 0.01, 0.01)
    origin = np.asarray(
        (center_x, float(minimum[1] - clearance), float(minimum[2] + height)),
        dtype=float,
    )
    tip = origin + np.asarray((0.0, -arrow_length, 0.0), dtype=float)
    head_center = tip + np.asarray(
        (
            0.0,
            arrow_length * OPENING_PREVIEW_FRONT_ARROW_HEAD_LENGTH_RATIO,
            0.0,
        ),
        dtype=float,
    )
    head_half_width = (
        arrow_length * OPENING_PREVIEW_FRONT_ARROW_HEAD_WIDTH_RATIO
    )
    return np.ascontiguousarray(
        (
            origin,
            tip,
            tip,
            head_center + np.asarray((-head_half_width, 0.0, 0.0)),
            tip,
            head_center + np.asarray((head_half_width, 0.0, 0.0)),
        ),
        dtype=np.float32,
    )


# ### Doors workspace ###
class DoorsWorkspace(QWidget):
    """Display reusable doors/windows and coordinate component edits."""

    # Legacy door-only signals remain stable for existing integrations.
    selection_changed = Signal(object, object)
    slot_edit_requested = Signal(str, str)
    slot_confirmation_requested = Signal(str, str)
    slot_transform_changed = Signal(str, str, object, object)
    slot_axis_scales_changed = Signal(str, str, object)
    make_double_changed = Signal(str, bool)
    make_double_sided_changed = Signal(str, bool)
    slot_mesh_paste_requested = Signal(str, str, str)

    # Asset-aware signals cover both doors and windows.
    asset_selection_changed = Signal(str, str, object, object)
    asset_slot_edit_requested = Signal(str, str, str)
    asset_slot_confirmation_requested = Signal(str, str, str)
    asset_slot_transform_changed = Signal(str, str, str, object, object)
    asset_slot_axis_scales_changed = Signal(str, str, str, object)
    asset_make_double_changed = Signal(str, str, bool)
    asset_make_double_sided_changed = Signal(str, str, bool)
    asset_slot_mesh_paste_requested = Signal(str, str, str, str)
    asset_slot_horizontal_flip_requested = Signal(str, str, str)
    asset_delete_requested = Signal(str, str)
    glass_area_creation_requested = Signal(str, str, str)
    face_creation_requested = Signal(str, str, str, object)
    face_deletion_requested = Signal(str, str, str, object)
    face_glass_conversion_requested = Signal(str, str, str, object)
    undo_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("doors_workspace")
        self._data = DoorLibraryData()
        self._pending_slot_positions: set[tuple[str, str, str]] = set()
        self._previewed_door_id: str | None = None
        self._previewed_asset_kind: str | None = None
        self._previewed_asset_id: str | None = None
        self._preview_face_edit_object_id: str | None = None
        self._preview_front_direction_arrow_item: gl.GLLinePlotItem | None = None
        self._copied_asset_slot: tuple[str, str, str] | None = None
        self._refreshing = False
        self._build_ui()
        self._connect_signals()
        self._refresh_tree()

    # ### Public model API ###
    def set_data(self, data: DoorLibraryData) -> None:
        """Replace the library while preserving a still-valid selection."""

        if not isinstance(data, DoorLibraryData):
            raise TypeError("DoorsWorkspace.set_data expects DoorLibraryData.")
        self._data = data.clone()
        self._discard_stale_pending_slots()
        self._refresh_tree()

    def data(self) -> DoorLibraryData:
        """Return a detached, persistence-ready copy of the door library."""

        return self._data.clone()

    def set_doors(self, doors: Iterable[DoorDefinition]) -> None:
        """Replace definitions without disturbing the library's placements."""

        prepared = list(doors)
        if any(not isinstance(door, DoorDefinition) for door in prepared):
            raise TypeError("DoorsWorkspace.set_doors expects DoorDefinition values.")
        known_door_ids = {door.door_id for door in prepared}
        self.set_data(
            DoorLibraryData(
                doors=prepared,
                placements=[
                    placement
                    for placement in self._data.placements
                    if placement.door_id in known_door_ids
                ],
                windows=list(self._data.windows),
                window_placements=list(self._data.window_placements),
            )
        )

    def doors(self) -> tuple[DoorDefinition, ...]:
        return tuple(self._data.doors)

    def upsert_door(self, door: DoorDefinition, *, select_body: bool = False) -> None:
        """Insert or replace one definition and optionally select its body."""

        if not isinstance(door, DoorDefinition):
            raise TypeError("DoorsWorkspace.upsert_door expects DoorDefinition.")
        doors = list(self._data.doors)
        for index, existing in enumerate(doors):
            if existing.door_id == door.door_id:
                doors[index] = door
                break
        else:
            doors.append(door)
        self._data = DoorLibraryData(
            doors=doors,
            placements=list(self._data.placements),
            windows=list(self._data.windows),
            window_placements=list(self._data.window_placements),
        )
        self._discard_stale_pending_slots()
        self._refresh_tree(
            select_door_id=door.door_id if select_body else None,
            select_slot_id=DOOR_SLOT_BODY if select_body else None,
        )

    def remove_door(self, door_id: str) -> bool:
        """Remove one reusable door and all of its bound placements."""

        normalized_id = str(door_id).strip()
        retained_doors = [
            door for door in self._data.doors if door.door_id != normalized_id
        ]
        if len(retained_doors) == len(self._data.doors):
            return False
        self._data = DoorLibraryData(
            doors=retained_doors,
            placements=[
                placement
                for placement in self._data.placements
                if placement.door_id != normalized_id
            ],
            windows=list(self._data.windows),
            window_placements=list(self._data.window_placements),
        )
        self._discard_stale_pending_slots()
        self._refresh_tree()
        return True

    def replace_slot(self, door_id: str, slot: DoorSlotData) -> bool:
        """Replace one component slot while retaining the current selection."""

        if not isinstance(slot, DoorSlotData):
            raise TypeError("DoorsWorkspace.replace_slot expects DoorSlotData.")
        door = self._data.get_door(str(door_id).strip())
        if door is None or door.get_slot(slot.slot_id) is None:
            return False
        self.upsert_door(door.replace_slot(slot))
        return True

    # ### Window model API ###
    def set_windows(self, windows: Iterable[WindowDefinition]) -> None:
        """Replace window definitions without disturbing door data."""

        prepared = list(windows)
        if any(not isinstance(window, WindowDefinition) for window in prepared):
            raise TypeError(
                "DoorsWorkspace.set_windows expects WindowDefinition values."
            )
        known_window_ids = {window.window_id for window in prepared}
        self.set_data(
            DoorLibraryData(
                doors=list(self._data.doors),
                placements=list(self._data.placements),
                windows=prepared,
                window_placements=[
                    placement
                    for placement in self._data.window_placements
                    if placement.window_id in known_window_ids
                ],
            )
        )

    def windows(self) -> tuple[WindowDefinition, ...]:
        return tuple(self._data.windows)

    def upsert_window(
        self,
        window: WindowDefinition,
        *,
        select_body: bool = False,
    ) -> None:
        """Insert or replace one reusable window definition."""

        if not isinstance(window, WindowDefinition):
            raise TypeError(
                "DoorsWorkspace.upsert_window expects WindowDefinition."
            )
        windows = list(self._data.windows)
        for index, existing in enumerate(windows):
            if existing.window_id == window.window_id:
                windows[index] = window
                break
        else:
            windows.append(window)
        self._data = DoorLibraryData(
            doors=list(self._data.doors),
            placements=list(self._data.placements),
            windows=windows,
            window_placements=list(self._data.window_placements),
        )
        self._discard_stale_pending_slots()
        self._refresh_tree(
            select_asset_kind=ASSET_KIND_WINDOW if select_body else None,
            select_asset_id=window.window_id if select_body else None,
            select_slot_id=DOOR_SLOT_BODY if select_body else None,
        )

    def remove_window(self, window_id: str) -> bool:
        """Remove one reusable window and all of its placements."""

        normalized_id = str(window_id).strip()
        retained_windows = [
            window
            for window in self._data.windows
            if window.window_id != normalized_id
        ]
        if len(retained_windows) == len(self._data.windows):
            return False
        self._data = DoorLibraryData(
            doors=list(self._data.doors),
            placements=list(self._data.placements),
            windows=retained_windows,
            window_placements=[
                placement
                for placement in self._data.window_placements
                if placement.window_id != normalized_id
            ],
        )
        self._discard_stale_pending_slots()
        self._refresh_tree()
        return True

    def replace_window_slot(
        self,
        window_id: str,
        slot: DoorSlotData,
    ) -> bool:
        """Replace one window component slot and preserve selection."""

        if not isinstance(slot, DoorSlotData):
            raise TypeError(
                "DoorsWorkspace.replace_window_slot expects DoorSlotData."
            )
        window = self._data.get_window(str(window_id).strip())
        if window is None or window.get_slot(slot.slot_id) is None:
            return False
        self.upsert_window(window.replace_slot(slot))
        return True

    def selected_door_id(self) -> str | None:
        if self.selected_asset_kind() != ASSET_KIND_DOOR:
            return None
        return self.selected_asset_id()

    def selected_window_id(self) -> str | None:
        if self.selected_asset_kind() != ASSET_KIND_WINDOW:
            return None
        return self.selected_asset_id()

    def selected_asset_kind(self) -> str | None:
        item = self.door_tree.currentItem()
        if item is None:
            return None
        value = item.data(0, ITEM_ASSET_KIND_ROLE)
        return str(value) if value in {ASSET_KIND_DOOR, ASSET_KIND_WINDOW} else None

    def selected_asset_id(self) -> str | None:
        item = self.door_tree.currentItem()
        if item is None:
            return None
        value = item.data(0, ITEM_DOOR_ID_ROLE)
        return str(value) if value else None

    def selected_slot_id(self) -> str | None:
        item = self.door_tree.currentItem()
        if item is None or item.data(0, ITEM_KIND_ROLE) != ITEM_KIND_SLOT:
            return None
        value = item.data(0, ITEM_SLOT_ID_ROLE)
        return str(value) if value else None

    def selected_door(self) -> DoorDefinition | None:
        door_id = self.selected_door_id()
        return self._data.get_door(door_id) if door_id is not None else None

    def selected_window(self) -> WindowDefinition | None:
        window_id = self.selected_window_id()
        return self._data.get_window(window_id) if window_id is not None else None

    def selected_asset(self) -> DoorDefinition | WindowDefinition | None:
        """Return the currently selected reusable opening asset."""

        if self.selected_asset_kind() == ASSET_KIND_DOOR:
            return self.selected_door()
        if self.selected_asset_kind() == ASSET_KIND_WINDOW:
            return self.selected_window()
        return None

    def selected_slot(self) -> DoorSlotData | None:
        """Return a selected door slot for backward compatibility."""

        door = self.selected_door()
        slot_id = self.selected_slot_id()
        if door is None or slot_id is None:
            return None
        return door.get_slot(slot_id)

    def selected_asset_slot(self) -> DoorSlotData | None:
        asset = self.selected_asset()
        slot_id = self.selected_slot_id()
        if asset is None or slot_id is None:
            return None
        return asset.get_slot(slot_id)

    def selected_window_slot(self) -> DoorSlotData | None:
        window = self.selected_window()
        slot_id = self.selected_slot_id()
        if window is None or slot_id is None:
            return None
        return window.get_slot(slot_id)

    def selected_glass_area_id(self) -> str | None:
        item = self.door_tree.currentItem()
        if item is None or item.data(0, ITEM_KIND_ROLE) != ITEM_KIND_GLASS_AREA:
            return None
        value = item.data(0, ITEM_GLASS_AREA_ID_ROLE)
        return str(value) if value else None

    def selected_glass_area(self) -> GlassAreaData | None:
        asset = self.selected_asset()
        glass_area_id = self.selected_glass_area_id()
        if asset is None or glass_area_id is None:
            return None
        return asset.get_glass_area(glass_area_id)

    def select_door_slot(self, door_id: str, slot_id: str | None = None) -> bool:
        """Select a door or one of its slots by persistent identifiers."""

        item = self._find_item(str(door_id).strip(), slot_id)
        if item is None:
            return False
        self.door_tree.setCurrentItem(item)
        item.setSelected(True)
        self.door_tree.scrollToItem(item)
        return True

    def select_window_slot(
        self,
        window_id: str,
        slot_id: str | None = None,
    ) -> bool:
        """Select a window or one of its component slots."""

        return self.select_asset_part(
            ASSET_KIND_WINDOW,
            window_id,
            slot_id=slot_id,
        )

    def select_glass_area(
        self,
        asset_kind: str,
        asset_id: str,
        glass_area_id: str,
    ) -> bool:
        """Select one glass-area leaf without enabling component gizmos."""

        return self.select_asset_part(
            asset_kind,
            asset_id,
            glass_area_id=glass_area_id,
        )

    def select_asset_part(
        self,
        asset_kind: str,
        asset_id: str,
        *,
        slot_id: str | None = None,
        glass_area_id: str | None = None,
    ) -> bool:
        """Select an asset root, component slot, or glass-area leaf."""

        item = self._find_asset_item(
            self._normalize_asset_kind(asset_kind),
            str(asset_id).strip(),
            slot_id=slot_id,
            glass_area_id=glass_area_id,
        )
        if item is None:
            return False
        self.door_tree.setCurrentItem(item)
        item.setSelected(True)
        self.door_tree.scrollToItem(item)
        return True

    def set_slot_position_pending(
        self,
        door_id: str,
        slot_id: str,
        pending: bool = True,
    ) -> bool:
        """Record whether an editable part has an unconfirmed gizmo transform."""

        return self.set_asset_slot_position_pending(
            ASSET_KIND_DOOR,
            door_id,
            slot_id,
            pending,
        )

    def set_asset_slot_position_pending(
        self,
        asset_kind: str,
        asset_id: str,
        slot_id: str,
        pending: bool = True,
    ) -> bool:
        """Track an unconfirmed hardware transform for either asset type."""

        normalized_kind = self._normalize_asset_kind(asset_kind)
        normalized_asset_id = str(asset_id).strip()
        normalized_slot_id = str(slot_id).strip().lower()
        asset = self._get_asset(normalized_kind, normalized_asset_id)
        if asset is None or asset.get_slot(normalized_slot_id) is None:
            return False
        key = (normalized_kind, normalized_asset_id, normalized_slot_id)
        if pending and not is_door_body_slot(normalized_slot_id):
            self._pending_slot_positions.add(key)
        else:
            self._pending_slot_positions.discard(key)
        self._refresh_slot_details()
        return True

    def is_slot_position_pending(self, door_id: str, slot_id: str) -> bool:
        return self.is_asset_slot_position_pending(
            ASSET_KIND_DOOR,
            door_id,
            slot_id,
        )

    def is_asset_slot_position_pending(
        self,
        asset_kind: str,
        asset_id: str,
        slot_id: str,
    ) -> bool:
        return (
            self._normalize_asset_kind(asset_kind),
            str(asset_id).strip(),
            str(slot_id).strip().lower(),
        ) in self._pending_slot_positions

    def clear_pending_slot_positions(self) -> None:
        self._pending_slot_positions.clear()
        self._refresh_tree()

    # ### Complete asset preview API ###
    def set_preview_model(self, model: GeneratedModel | None) -> None:
        """Display the assembled door/window and selected editable part."""

        self.clear_preview_face_edit_geometry()
        if model is None:
            self.preview_viewer.set_selected_placed_object_ids(())
            self.preview_viewer.clear_model()
            self._refresh_preview_front_direction_arrow()
            self._previewed_door_id = None
            self._previewed_asset_kind = None
            self._previewed_asset_id = None
            return
        if not isinstance(model, GeneratedModel):
            raise TypeError("Opening previews require a GeneratedModel or None.")
        selected_asset_kind = self.selected_asset_kind()
        selected_asset_id = self.selected_asset_id()
        self.preview_viewer.set_model(
            model,
            preserve_camera=bool(
                self.preview_viewer.model is not None
                and selected_asset_kind == self._previewed_asset_kind
                and selected_asset_id is not None
                and selected_asset_id == self._previewed_asset_id
            ),
        )
        self._refresh_preview_front_direction_arrow()
        self._previewed_asset_kind = selected_asset_kind
        self._previewed_asset_id = selected_asset_id
        self._previewed_door_id = (
            selected_asset_id
            if selected_asset_kind == ASSET_KIND_DOOR
            else None
        )
        self._sync_preview_slot_gizmo_selection(preserve_mode=True)

    def _refresh_preview_front_direction_arrow(self) -> None:
        """Rebuild the non-interactive local-front marker for this viewer only."""

        previous_item = self._preview_front_direction_arrow_item
        self._preview_front_direction_arrow_item = None
        if (
            previous_item is not None
            and previous_item in self.preview_viewer.view.items
        ):
            self.preview_viewer.view.removeItem(previous_item)

        model = self.preview_viewer.model
        if model is None:
            self.preview_viewer.view.update()
            return
        positions = _build_preview_front_direction_arrow_positions(model)
        if positions is None:
            self.preview_viewer.view.update()
            return

        arrow_item = gl.GLLinePlotItem(
            pos=positions,
            color=OPENING_PREVIEW_FRONT_ARROW_COLOR,
            width=OPENING_PREVIEW_FRONT_ARROW_WIDTH_PIXELS,
            antialias=True,
            mode="lines",
        )
        arrow_item.setGLOptions("translucent")
        arrow_item._housemaker_doors_preview_role = (
            OPENING_PREVIEW_FRONT_ARROW_ROLE
        )
        self.preview_viewer.view.addItem(arrow_item)
        self._preview_front_direction_arrow_item = arrow_item
        self.preview_viewer.view.update()

    def set_preview_face_edit_geometry(
        self,
        object_id: str,
        vertices: object,
        faces: object,
    ) -> bool:
        """Enable source-indexed face editing for the selected preview slot."""

        normalized_object_id = str(object_id).strip()
        slot = self.selected_asset_slot()
        model = self.preview_viewer.model
        if (
            not normalized_object_id
            or slot is None
            or slot.source_object_id != normalized_object_id
            or model is None
            or not any(
                preview.object_id == normalized_object_id
                for preview in model.preview_placed_objects
            )
        ):
            self.clear_preview_face_edit_geometry()
            return False
        try:
            self.preview_viewer.set_face_edit_geometry(vertices, faces)
        except (TypeError, ValueError):
            self.clear_preview_face_edit_geometry()
            return False
        self._preview_face_edit_object_id = normalized_object_id
        self.preview_viewer.set_face_editing_enabled(True)
        self._sync_glass_action_button()
        return True

    def clear_preview_face_edit_geometry(self) -> None:
        """Disable face editing without changing the selected component slot."""

        self._preview_face_edit_object_id = None
        self.preview_viewer.clear_face_edit_geometry()
        self.preview_viewer.set_face_editing_enabled(False)
        self._sync_glass_action_button()

    def _sync_preview_slot_gizmo_selection(
        self,
        *,
        preserve_mode: bool = False,
    ) -> None:
        """Select the current preview part and show gizmos only for slots."""

        slot = self.selected_asset_slot()
        glass_area = self.selected_glass_area()
        model = self.preview_viewer.model
        editable_object_id = None
        if slot is not None and model is not None:
            has_preview = any(
                preview.object_id == slot.source_object_id
                for preview in model.preview_placed_objects
            )
            if has_preview:
                editable_object_id = slot.source_object_id
        selected_object_id = editable_object_id
        if glass_area is not None and model is not None:
            has_glass_preview = any(
                preview.object_id == glass_area.source_object_id
                for preview in model.preview_placed_objects
            )
            if has_glass_preview:
                selected_object_id = glass_area.source_object_id
        if editable_object_id != self._preview_face_edit_object_id:
            self.clear_preview_face_edit_geometry()
        body_selected = bool(
            editable_object_id is not None
            and slot is not None
            and is_door_body_slot(slot.slot_id)
        )
        preserve_axis_scale = bool(
            preserve_mode
            and editable_object_id is not None
            and self.preview_viewer.get_selected_placed_object_id()
            == editable_object_id
            and self.preview_viewer.get_placed_object_gizmo_mode() == "scale"
        )
        self.preview_viewer.set_placed_object_gizmo_capabilities(
            transform_enabled=(
                editable_object_id is not None and not body_selected
            ),
            axis_scale_enabled=editable_object_id is not None,
            prefer_axis_scale=body_selected or preserve_axis_scale,
        )
        self.preview_viewer.set_selected_placed_object_ids(
            () if selected_object_id is None else (selected_object_id,),
            active_object_id=selected_object_id,
        )

    # ### UI construction ###
    def _build_ui(self) -> None:
        controls_widget = QWidget(self)
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)

        self.door_tree = QTreeWidget(self)
        self.door_tree.setObjectName("doors_tree")
        self.door_tree.setHeaderLabels(("Door, window / part",))
        self.door_tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.door_tree.setRootIsDecorated(True)
        self.door_tree.setUniformRowHeights(True)
        self.door_tree.setToolTip(
            "Double-click a component slot to open it in Generation."
        )
        self.door_tree.header().setStretchLastSection(True)
        self.door_tree.setColumnWidth(0, 190)

        self.delete_asset_button = QPushButton("Delete", self)
        self.delete_asset_button.setObjectName("delete_opening_asset_button")
        self.delete_asset_button.setToolTip(
            "Delete the selected door or window and all of its placements. "
            "Press Ctrl+Z to restore it."
        )
        self.delete_asset_button.setEnabled(False)

        self.rotation_status_label = QLabel("Rotation: -", self)
        self.rotation_status_label.setObjectName("opening_part_rotation_label")

        self.make_double_checkbox = QCheckBox("Make double", self)
        self.make_double_checkbox.setObjectName("make_door_double_checkbox")
        self.make_double_checkbox.setToolTip(
            "Split the selected door or window into independently editable "
            "left and right Body and Handle slots."
        )
        self.make_double_sided_checkbox = QCheckBox(
            "Make double sided",
            self,
        )
        self.make_double_sided_checkbox.setObjectName(
            "make_door_double_sided_checkbox"
        )
        self.make_double_sided_checkbox.setToolTip(
            "Create independently generated Front and Back Body and Handle "
            "slots for the selected door or window."
        )
        door_options_widget = QWidget(self)
        door_options_layout = QHBoxLayout(door_options_widget)
        door_options_layout.setContentsMargins(0, 0, 0, 0)
        door_options_layout.setSpacing(8)
        door_options_layout.addWidget(self.make_double_checkbox)
        door_options_layout.addWidget(self.make_double_sided_checkbox)
        door_options_layout.addStretch(1)

        self.add_glass_area_button = QPushButton("Add glass area", self)
        self.add_glass_area_button.setObjectName("add_opening_glass_area_button")
        self.add_glass_area_button.setToolTip(
            "Drag a rectangle through the selected door or window body and "
            "fill the cutout with the shared glass material."
        )
        self.add_glass_area_button.setEnabled(False)

        self.flip_mesh_horizontally_button = QPushButton(
            "Flip mesh horizontally",
            self,
        )
        self.flip_mesh_horizontally_button.setObjectName(
            "flip_opening_slot_mesh_horizontally_button"
        )
        self.flip_mesh_horizontally_button.setToolTip(
            "Flip the selected Body or Handle across its local horizontal axis."
        )
        self.flip_mesh_horizontally_button.setEnabled(False)

        self.confirm_slot_position_button = QPushButton(
            "Confirm slot transform",
            self,
        )
        self.confirm_slot_position_button.setObjectName(
            "confirm_door_slot_position_button"
        )
        self.confirm_slot_position_button.setEnabled(False)

        controls_layout.addWidget(self.door_tree, 1)
        controls_layout.addWidget(self.rotation_status_label)
        controls_layout.addWidget(door_options_widget)
        controls_layout.addWidget(self.delete_asset_button)
        controls_layout.addWidget(self.flip_mesh_horizontally_button)
        controls_layout.addWidget(self.add_glass_area_button)
        controls_layout.addWidget(self.confirm_slot_position_button)

        preview_group = QGroupBox("Complete door / window preview", self)
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(4, 4, 4, 4)
        self.wireframe_checkbox = QCheckBox("Wireframe", preview_group)
        self.wireframe_checkbox.setObjectName(
            "door_preview_wireframe_checkbox"
        )
        self.wireframe_checkbox.setChecked(False)
        self.wireframe_checkbox.setToolTip(
            "Show mesh edges over the complete door or window preview."
        )
        self.cursor_light_checkbox = QCheckBox("Cursor light", preview_group)
        self.cursor_light_checkbox.setObjectName(
            "door_preview_cursor_light_checkbox"
        )
        self.cursor_light_checkbox.setChecked(True)
        self.cursor_light_checkbox.setToolTip(
            "Move the preview point light with the cursor over the door or "
            "window."
        )
        self.proportional_editing_checkbox = QCheckBox(
            "Proportional editing",
            preview_group,
        )
        self.proportional_editing_checkbox.setObjectName(
            "door_preview_proportional_editing_checkbox"
        )
        self.proportional_editing_checkbox.setChecked(True)
        self.proportional_editing_checkbox.setToolTip(
            "Mirror position and rotation edits to the opposite leaf, and "
            "apply the same scale to both leaf parts."
        )
        self.proportional_editing_checkbox.setVisible(False)
        preview_options_widget = QWidget(preview_group)
        preview_options_layout = QHBoxLayout(preview_options_widget)
        preview_options_layout.setContentsMargins(0, 0, 0, 0)
        preview_options_layout.setSpacing(8)
        preview_options_layout.addWidget(self.wireframe_checkbox)
        preview_options_layout.addWidget(self.cursor_light_checkbox)
        preview_options_layout.addWidget(self.proportional_editing_checkbox)
        preview_options_layout.addStretch(1)
        self.preview_viewer = GlbViewerWidget(
            preview_group,
            wireframe_enabled=False,
            placed_object_editing_enabled=True,
            placed_object_click_selection_enabled=True,
            placed_object_auxiliary_controls_enabled=False,
            placed_object_axis_scale_gizmos_enabled=True,
            face_editing_enabled=False,
            symmetric_preview_fade_enabled=False,
            orbiting_point_light_with_pbr=True,
            hover_point_light_enabled=True,
        )
        self.preview_viewer.setObjectName("door_complete_preview_viewer")
        preview_layout.addWidget(preview_options_widget)
        preview_layout.addWidget(self.preview_viewer, 1)

        self.workspace_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.workspace_splitter.setObjectName("doors_workspace_splitter")
        self.workspace_splitter.setChildrenCollapsible(False)
        self.workspace_splitter.addWidget(controls_widget)
        self.workspace_splitter.addWidget(preview_group)
        self.workspace_splitter.setStretchFactor(0, 0)
        self.workspace_splitter.setStretchFactor(1, 1)
        self.workspace_splitter.setSizes((360, 840))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.workspace_splitter, 1)

        self.undo_shortcut = QShortcut(
            QKeySequence.StandardKey.Undo,
            self,
        )
        self.undo_shortcut.setContext(
            Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        self.copy_slot_shortcut = QShortcut(
            QKeySequence.StandardKey.Copy,
            self,
        )
        self.copy_slot_shortcut.setContext(
            Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        self.paste_slot_shortcut = QShortcut(
            QKeySequence.StandardKey.Paste,
            self,
        )
        self.paste_slot_shortcut.setContext(
            Qt.ShortcutContext.WidgetWithChildrenShortcut
        )

    def _connect_signals(self) -> None:
        self.door_tree.currentItemChanged.connect(
            self._handle_current_item_changed
        )
        self.door_tree.itemDoubleClicked.connect(
            self._handle_item_double_clicked
        )
        self.make_double_checkbox.toggled.connect(
            self._handle_make_double_changed
        )
        self.make_double_sided_checkbox.toggled.connect(
            self._handle_make_double_sided_changed
        )
        self.add_glass_area_button.clicked.connect(
            self._handle_add_glass_area_clicked
        )
        self.delete_asset_button.clicked.connect(
            self._handle_delete_asset_clicked
        )
        self.flip_mesh_horizontally_button.clicked.connect(
            self._handle_flip_mesh_horizontally_clicked
        )
        self.confirm_slot_position_button.clicked.connect(
            self._handle_confirm_slot_position_clicked
        )
        self.wireframe_checkbox.toggled.connect(
            self.preview_viewer.set_wireframe_enabled
        )
        self.cursor_light_checkbox.toggled.connect(
            self.preview_viewer.set_hover_point_light_enabled
        )
        self.preview_viewer.placed_object_transform_changed.connect(
            self._handle_preview_slot_transform_changed
        )
        self.preview_viewer.placed_object_transform_preview_changed.connect(
            self._handle_preview_slot_transform_preview_changed
        )
        self.preview_viewer.placed_object_rotation_preview_changed.connect(
            self._handle_preview_slot_rotation_preview_changed
        )
        self.preview_viewer.placed_object_axis_scales_changed.connect(
            self._handle_preview_slot_axis_scales_changed
        )
        self.preview_viewer.placed_object_selection_changed.connect(
            self._handle_preview_object_selection_changed
        )
        self.preview_viewer.placed_object_double_clicked.connect(
            self._handle_preview_object_double_clicked
        )
        self.preview_viewer.face_selection_changed.connect(
            self._handle_preview_face_selection_changed
        )
        self.preview_viewer.object_face_creation_requested.connect(
            self._handle_preview_face_creation_requested
        )
        self.preview_viewer.delete_requested.connect(
            self._handle_preview_face_delete_requested
        )
        self.undo_shortcut.activated.connect(self._emit_undo_requested)
        self.copy_slot_shortcut.activated.connect(
            self._handle_copy_slot_shortcut
        )
        self.paste_slot_shortcut.activated.connect(
            self._handle_paste_slot_shortcut
        )
        self.preview_viewer.undo_requested.connect(self._emit_undo_requested)

    @Slot()
    def _emit_undo_requested(self) -> None:
        """Route Ctrl+Z from any Doors control through one public signal."""

        self.undo_requested.emit()

    # ### Tree synchronization ###
    def _refresh_tree(
        self,
        *,
        select_door_id: str | None = None,
        select_asset_kind: str | None = None,
        select_asset_id: str | None = None,
        select_slot_id: str | None = None,
        select_glass_area_id: str | None = None,
    ) -> None:
        previous_selection = (
            self.selected_asset_kind(),
            self.selected_asset_id(),
            self.selected_slot_id(),
            self.selected_glass_area_id(),
        )
        if select_door_id is not None:
            select_asset_kind = ASSET_KIND_DOOR
            select_asset_id = select_door_id
        requested_selection = (
            select_asset_kind,
            select_asset_id,
            select_slot_id,
            select_glass_area_id,
        )
        if select_asset_id is None:
            requested_selection = previous_selection

        self._refreshing = True
        blocker = QSignalBlocker(self.door_tree)
        self.door_tree.clear()
        for door in self._data.doors:
            door_item = self._build_door_item(door)
            self.door_tree.addTopLevelItem(door_item)
            self._expand_asset_item(door_item)
        for window in self._data.windows:
            window_item = self._build_window_item(window)
            self.door_tree.addTopLevelItem(window_item)
            self._expand_asset_item(window_item)
        selected_item = self._find_asset_item(
            requested_selection[0],
            requested_selection[1],
            slot_id=requested_selection[2],
            glass_area_id=requested_selection[3],
        )
        if selected_item is None and self.door_tree.topLevelItemCount() > 0:
            first_asset_item = self.door_tree.topLevelItem(0)
            selected_item = self._find_asset_item(
                first_asset_item.data(0, ITEM_ASSET_KIND_ROLE),
                first_asset_item.data(0, ITEM_DOOR_ID_ROLE),
                slot_id=DOOR_SLOT_BODY,
            )
            if selected_item is None:
                selected_item = first_asset_item
        self.door_tree.setCurrentItem(selected_item)
        del blocker
        self._refreshing = False
        self._refresh_slot_details()
        self._refresh_rotation_status()

        current_selection = (
            self.selected_asset_kind(),
            self.selected_asset_id(),
            self.selected_slot_id(),
            self.selected_glass_area_id(),
        )
        if current_selection != previous_selection:
            self._emit_selection_changed()

    def _build_door_item(self, door: DoorDefinition) -> QTreeWidgetItem:
        return self._build_opening_asset_item(
            door,
            asset_kind=ASSET_KIND_DOOR,
            item_kind=ITEM_KIND_DOOR,
        )

    def _build_window_item(self, window: WindowDefinition) -> QTreeWidgetItem:
        return self._build_opening_asset_item(
            window,
            asset_kind=ASSET_KIND_WINDOW,
            item_kind=ITEM_KIND_WINDOW,
        )

    def _build_opening_asset_item(
        self,
        asset: DoorDefinition | WindowDefinition,
        *,
        asset_kind: str,
        item_kind: str,
    ) -> QTreeWidgetItem:
        """Build the shared single/double and front/back component hierarchy."""

        asset_id = self._asset_id(asset)
        item = QTreeWidgetItem((asset.name,))
        item.setData(0, ITEM_KIND_ROLE, item_kind)
        item.setData(0, ITEM_DOOR_ID_ROLE, asset_id)
        item.setData(0, ITEM_ASSET_KIND_ROLE, asset_kind)
        slots_by_id = {slot.slot_id: slot for slot in asset.slots}
        if not asset.make_double_sided:
            slot_order = (
                DOUBLE_LEAF_SLOT_ORDER
                if asset.make_double
                else SINGLE_SIDED_SLOT_ORDER
            )
            for slot_id in slot_order:
                slot = slots_by_id.get(slot_id)
                if slot is not None:
                    item.addChild(
                        self._build_slot_item(
                            asset_kind,
                            asset_id,
                            slot,
                            make_double=asset.make_double,
                        )
                    )
            self._append_glass_area_items(item, asset_kind, asset)
            return item

        for face in DOOR_FACE_ORDER:
            face_item = QTreeWidgetItem((DOOR_FACE_DISPLAY_NAMES[face],))
            face_item.setData(0, ITEM_KIND_ROLE, ITEM_KIND_FACE)
            face_item.setData(0, ITEM_DOOR_ID_ROLE, asset_id)
            face_item.setData(0, ITEM_ASSET_KIND_ROLE, asset_kind)
            slot_order = (
                DOUBLE_SIDED_DOUBLE_LEAF_SLOT_ORDER
                if asset.make_double
                else DOUBLE_SIDED_SLOT_ORDER
            )
            for slot_id in slot_order:
                if door_slot_side(slot_id) != face:
                    continue
                slot = slots_by_id.get(slot_id)
                if slot is not None:
                    face_item.addChild(
                        self._build_slot_item(
                            asset_kind,
                            asset_id,
                            slot,
                            make_double=asset.make_double,
                        )
                    )
            item.addChild(face_item)
        self._append_glass_area_items(item, asset_kind, asset)
        return item

    def _build_slot_item(
        self,
        asset_kind: str,
        asset_id: str,
        slot: DoorSlotData,
        *,
        make_double: bool = False,
    ) -> QTreeWidgetItem:
        """Build one component leaf shared by door and window trees."""

        slot_item = QTreeWidgetItem(
            (
                door_slot_display_name(
                    slot.slot_id,
                    make_double=make_double,
                ),
            )
        )
        slot_item.setData(0, ITEM_KIND_ROLE, ITEM_KIND_SLOT)
        slot_item.setData(0, ITEM_DOOR_ID_ROLE, asset_id)
        slot_item.setData(0, ITEM_SLOT_ID_ROLE, slot.slot_id)
        slot_item.setData(0, ITEM_ASSET_KIND_ROLE, asset_kind)
        return slot_item

    def _append_glass_area_items(
        self,
        parent: QTreeWidgetItem,
        asset_kind: str,
        asset: DoorDefinition | WindowDefinition,
    ) -> None:
        asset_id = self._asset_id(asset)
        for glass_area in asset.glass_areas:
            item = QTreeWidgetItem((glass_area.name,))
            item.setData(0, ITEM_KIND_ROLE, ITEM_KIND_GLASS_AREA)
            item.setData(0, ITEM_DOOR_ID_ROLE, asset_id)
            item.setData(0, ITEM_ASSET_KIND_ROLE, asset_kind)
            item.setData(
                0,
                ITEM_GLASS_AREA_ID_ROLE,
                glass_area.glass_area_id,
            )
            parent.addChild(item)

    @staticmethod
    def _expand_asset_item(item: QTreeWidgetItem) -> None:
        item.setExpanded(True)
        for child_index in range(item.childCount()):
            child = item.child(child_index)
            if child.data(0, ITEM_KIND_ROLE) == ITEM_KIND_FACE:
                child.setExpanded(True)

    @staticmethod
    def _find_slot_item(
        parent: QTreeWidgetItem,
        slot_id: str,
    ) -> QTreeWidgetItem | None:
        """Find one slot leaf below a door regardless of grouping depth."""

        for child_index in range(parent.childCount()):
            child = parent.child(child_index)
            if child.data(0, ITEM_SLOT_ID_ROLE) == slot_id:
                return child
            descendant = DoorsWorkspace._find_slot_item(child, slot_id)
            if descendant is not None:
                return descendant
        return None

    def _find_item(
        self,
        door_id: str | None,
        slot_id: str | None,
    ) -> QTreeWidgetItem | None:
        """Find a legacy door item by persistent identifiers."""

        return self._find_asset_item(
            ASSET_KIND_DOOR,
            door_id,
            slot_id=slot_id,
        )

    def _find_asset_item(
        self,
        asset_kind: str | None,
        asset_id: str | None,
        *,
        slot_id: str | None = None,
        glass_area_id: str | None = None,
    ) -> QTreeWidgetItem | None:
        if not asset_kind or not asset_id:
            return None
        normalized_slot_id = str(slot_id).strip().lower() if slot_id else None
        normalized_glass_id = (
            str(glass_area_id).strip() if glass_area_id else None
        )
        for index in range(self.door_tree.topLevelItemCount()):
            asset_item = self.door_tree.topLevelItem(index)
            if (
                asset_item.data(0, ITEM_ASSET_KIND_ROLE) != asset_kind
                or asset_item.data(0, ITEM_DOOR_ID_ROLE) != asset_id
            ):
                continue
            if normalized_slot_id is not None:
                return self._find_slot_item(asset_item, normalized_slot_id)
            if normalized_glass_id is not None:
                return self._find_glass_area_item(
                    asset_item,
                    normalized_glass_id,
                )
            return asset_item
        return None

    @staticmethod
    def _find_glass_area_item(
        parent: QTreeWidgetItem,
        glass_area_id: str,
    ) -> QTreeWidgetItem | None:
        for child_index in range(parent.childCount()):
            child = parent.child(child_index)
            if child.data(0, ITEM_GLASS_AREA_ID_ROLE) == glass_area_id:
                return child
        return None

    # ### Selection and confirmation ###
    def _handle_current_item_changed(
        self,
        _current: QTreeWidgetItem | None,
        _previous: QTreeWidgetItem | None,
    ) -> None:
        if self._refreshing:
            return
        self._refresh_slot_details()
        self._refresh_rotation_status()
        self._sync_preview_slot_gizmo_selection()
        self._emit_selection_changed()

    @Slot(object)
    def _handle_preview_object_selection_changed(
        self,
        object_id: object,
    ) -> None:
        """Select the slot or glass area picked in the complete preview."""

        asset_kind = self._previewed_asset_kind
        asset_id = self._previewed_asset_id
        if asset_kind is None or asset_id is None or object_id is None:
            return
        asset = self._get_asset(asset_kind, asset_id)
        if asset is None:
            return
        slot = self._find_preview_slot(asset, object_id)
        if slot is not None:
            if (
                self.selected_asset_kind() == asset_kind
                and self.selected_asset_id() == asset_id
                and self.selected_slot_id() == slot.slot_id
            ):
                return
            self.select_asset_part(
                asset_kind,
                asset_id,
                slot_id=slot.slot_id,
            )
            return
        glass_area = self._find_preview_glass_area(asset, object_id)
        if glass_area is None or (
            self.selected_asset_kind() == asset_kind
            and self.selected_asset_id() == asset_id
            and self.selected_glass_area_id() == glass_area.glass_area_id
        ):
            return
        self.select_glass_area(
            asset_kind,
            asset_id,
            glass_area.glass_area_id,
        )

    @Slot(str)
    def _handle_preview_object_double_clicked(self, object_id: str) -> None:
        """Open Generation only for concrete component slots."""

        asset_kind = self._previewed_asset_kind
        asset_id = self._previewed_asset_id
        if asset_kind is None or asset_id is None:
            return
        asset = self._get_asset(asset_kind, asset_id)
        if asset is None:
            return
        slot = self._find_preview_slot(asset, object_id)
        if slot is None:
            return
        self.select_asset_part(asset_kind, asset_id, slot_id=slot.slot_id)
        self._emit_slot_edit_requested(asset_kind, asset_id, slot.slot_id)

    @staticmethod
    def _find_preview_slot(
        asset: DoorDefinition | WindowDefinition,
        object_id: object,
    ) -> DoorSlotData | None:
        """Resolve one preview-object identity to its component slot."""

        normalized_object_id = str(object_id).strip()
        return next(
            (
                slot
                for slot in asset.slots
                if slot.source_object_id == normalized_object_id
            ),
            None,
        )

    @staticmethod
    def _find_preview_glass_area(
        asset: DoorDefinition | WindowDefinition,
        object_id: object,
    ) -> GlassAreaData | None:
        """Resolve one preview-object identity to its glass-area modifier."""

        normalized_object_id = str(object_id).strip()
        return next(
            (
                area
                for area in asset.glass_areas
                if area.source_object_id == normalized_object_id
            ),
            None,
        )

    @Slot(str, object, object)
    def _handle_preview_slot_transform_changed(
        self,
        object_id: str,
        world_position: object,
        rotation_degrees: object,
    ) -> None:
        """Forward transforms only for the selected generated hardware slot."""

        asset_kind = self.selected_asset_kind()
        asset_id = self.selected_asset_id()
        slot = self.selected_asset_slot()
        if (
            asset_kind is None
            or asset_id is None
            or slot is None
            or is_door_body_slot(slot.slot_id)
            or str(object_id) != slot.source_object_id
        ):
            return
        self.asset_slot_transform_changed.emit(
            asset_kind,
            asset_id,
            slot.slot_id,
            world_position,
            rotation_degrees,
        )
        if asset_kind == ASSET_KIND_DOOR:
            self.slot_transform_changed.emit(
                asset_id,
                slot.slot_id,
                world_position,
                rotation_degrees,
            )

    @Slot(str, object, object, object)
    def _handle_preview_slot_transform_preview_changed(
        self,
        object_id: str,
        world_position: object,
        rotation_degrees: object,
        axis_scales: object,
    ) -> None:
        """Mirror a paired leaf part continuously during its gizmo drag."""

        asset = self.selected_asset()
        slot = self.selected_asset_slot()
        if (
            asset is None
            or slot is None
            or str(object_id) != slot.source_object_id
            or not self.is_proportional_editing_enabled()
        ):
            return
        try:
            counterpart_slot_id = door_slot_counterpart(slot.slot_id)
            counterpart = asset.get_slot(counterpart_slot_id)
            preview_slot = slot.with_transform(
                position_meters=world_position,
                rotation_degrees=rotation_degrees,
                axis_scales=axis_scales,
            )
        except (TypeError, ValueError):
            return
        if counterpart is None:
            return
        self.preview_viewer.apply_placed_object_full_transform_preview(
            counterpart.source_object_id,
            (
                -preview_slot.position_meters[0],
                preview_slot.position_meters[1],
                preview_slot.position_meters[2],
            ),
            (
                preview_slot.rotation_degrees[0],
                -preview_slot.rotation_degrees[1],
                -preview_slot.rotation_degrees[2],
            ),
            preview_slot.axis_scales,
        )

    @Slot(str, object)
    def _handle_preview_slot_rotation_preview_changed(
        self,
        object_id: str,
        rotation_degrees: object,
    ) -> None:
        """Display the selected slot's live gizmo rotation."""

        slot = self.selected_asset_slot()
        if slot is None or str(object_id) != slot.source_object_id:
            return
        self.rotation_status_label.setText(
            self._format_rotation_status(rotation_degrees)
        )

    @Slot(str, object)
    def _handle_preview_slot_axis_scales_changed(
        self,
        object_id: str,
        axis_scales: object,
    ) -> None:
        """Forward local XYZ scales for any selected component slot."""

        asset_kind = self.selected_asset_kind()
        asset_id = self.selected_asset_id()
        slot = self.selected_asset_slot()
        if (
            asset_kind is None
            or asset_id is None
            or slot is None
            or str(object_id) != slot.source_object_id
        ):
            return
        self.asset_slot_axis_scales_changed.emit(
            asset_kind,
            asset_id,
            slot.slot_id,
            axis_scales,
        )
        if asset_kind == ASSET_KIND_DOOR:
            self.slot_axis_scales_changed.emit(
                asset_id,
                slot.slot_id,
                axis_scales,
            )

    def _emit_selection_changed(self) -> None:
        asset_kind = self.selected_asset_kind()
        asset_id = self.selected_asset_id()
        slot_id = self.selected_slot_id()
        glass_area_id = self.selected_glass_area_id()
        self.asset_selection_changed.emit(
            "" if asset_kind is None else asset_kind,
            "" if asset_id is None else asset_id,
            slot_id,
            glass_area_id,
        )
        if asset_kind == ASSET_KIND_DOOR:
            self.selection_changed.emit(asset_id, slot_id)
        else:
            self.selection_changed.emit(None, None)

    def _handle_item_double_clicked(
        self,
        item: QTreeWidgetItem,
        _column: int,
    ) -> None:
        """Request Generation editing only for a concrete component slot."""

        if item.data(0, ITEM_KIND_ROLE) != ITEM_KIND_SLOT:
            return
        asset_kind = item.data(0, ITEM_ASSET_KIND_ROLE)
        asset_id = item.data(0, ITEM_DOOR_ID_ROLE)
        slot_id = item.data(0, ITEM_SLOT_ID_ROLE)
        if (
            asset_kind not in {ASSET_KIND_DOOR, ASSET_KIND_WINDOW}
            or not isinstance(asset_id, str)
            or not isinstance(slot_id, str)
        ):
            return
        if not asset_id.strip() or not slot_id.strip():
            return
        self._emit_slot_edit_requested(asset_kind, asset_id, slot_id)

    def _handle_confirm_slot_position_clicked(self) -> None:
        asset_kind = self.selected_asset_kind()
        asset_id = self.selected_asset_id()
        slot_id = self.selected_slot_id()
        if (
            asset_kind is None
            or asset_id is None
            or slot_id is None
            or is_door_body_slot(slot_id)
            or not self.is_asset_slot_position_pending(
                asset_kind,
                asset_id,
                slot_id,
            )
        ):
            return
        self.asset_slot_confirmation_requested.emit(
            asset_kind,
            asset_id,
            slot_id,
        )
        if asset_kind == ASSET_KIND_DOOR:
            self.slot_confirmation_requested.emit(asset_id, slot_id)

    def _emit_slot_edit_requested(
        self,
        asset_kind: str,
        asset_id: str,
        slot_id: str,
    ) -> None:
        self.asset_slot_edit_requested.emit(asset_kind, asset_id, slot_id)
        if asset_kind == ASSET_KIND_DOOR:
            self.slot_edit_requested.emit(asset_id, slot_id)

    def _handle_add_glass_area_clicked(self) -> None:
        """Convert selected faces or request a rectangular glass cutout."""

        asset_kind = self.selected_asset_kind()
        asset_id = self.selected_asset_id()
        asset = self.selected_asset()
        if asset_kind is None or asset_id is None or asset is None:
            return
        selected_slot = self.selected_asset_slot()
        selected_faces = self.preview_viewer.get_selected_face_indices()
        if (
            selected_slot is not None
            and selected_faces
            and self._preview_face_edit_object_id
            == selected_slot.source_object_id
        ):
            self.face_glass_conversion_requested.emit(
                asset_kind,
                asset_id,
                selected_slot.source_object_id,
                selected_faces,
            )
            return
        body = (
            selected_slot
            if selected_slot is not None
            and is_door_body_slot(selected_slot.slot_id)
            else asset.get_slot(DOOR_SLOT_BODY)
        )
        if body is None:
            return
        self.glass_area_creation_requested.emit(
            asset_kind,
            asset_id,
            body.source_object_id,
        )

    @Slot(object)
    def _handle_preview_face_selection_changed(
        self,
        _face_indices: object,
    ) -> None:
        """Keep the shared glass button synchronized with face selection."""

        self._sync_glass_action_button()

    @Slot()
    def _handle_preview_face_delete_requested(self) -> None:
        """Request deletion only for faces owned by the selected source slot."""

        asset_kind = self.selected_asset_kind()
        asset_id = self.selected_asset_id()
        slot = self.selected_asset_slot()
        selected_faces = self.preview_viewer.get_selected_face_indices()
        if (
            asset_kind is None
            or asset_id is None
            or slot is None
            or not selected_faces
            or self._preview_face_edit_object_id != slot.source_object_id
        ):
            return
        self.face_deletion_requested.emit(
            asset_kind,
            asset_id,
            slot.source_object_id,
            selected_faces,
        )

    @Slot(object)
    def _handle_preview_face_creation_requested(
        self,
        raw_vertex_indices: object,
    ) -> None:
        """Request a face fill only in the selected source slot's index space."""

        asset_kind = self.selected_asset_kind()
        asset_id = self.selected_asset_id()
        slot = self.selected_asset_slot()
        try:
            vertex_indices = tuple(int(index) for index in raw_vertex_indices)  # type: ignore[arg-type]
        except (TypeError, ValueError, OverflowError):
            return
        if (
            asset_kind is None
            or asset_id is None
            or slot is None
            or len(vertex_indices) < 3
            or self._preview_face_edit_object_id != slot.source_object_id
        ):
            return
        self.face_creation_requested.emit(
            asset_kind,
            asset_id,
            slot.source_object_id,
            vertex_indices,
        )

    def _handle_delete_asset_clicked(self) -> None:
        """Request deletion of the selected door or window owner."""

        asset_kind = self.selected_asset_kind()
        asset_id = self.selected_asset_id()
        if asset_kind is None or asset_id is None:
            return
        self.asset_delete_requested.emit(asset_kind, asset_id)

    def _handle_flip_mesh_horizontally_clicked(self) -> None:
        """Request a persistent horizontal flip for one concrete slot."""

        asset_kind = self.selected_asset_kind()
        asset_id = self.selected_asset_id()
        slot = self.selected_asset_slot()
        if asset_kind is None or asset_id is None or slot is None:
            return
        self.asset_slot_horizontal_flip_requested.emit(
            asset_kind,
            asset_id,
            slot.slot_id,
        )

    def _handle_make_double_changed(self, checked: bool) -> None:
        """Request one model-owned single/double-leaf transition."""

        if self._refreshing:
            return
        asset_kind = self.selected_asset_kind()
        asset_id = self.selected_asset_id()
        asset = self.selected_asset()
        if asset_kind is None or asset_id is None or asset is None:
            self._sync_make_double_checkbox(None)
            return
        requested = bool(checked)
        if requested == asset.make_double:
            return
        self.asset_make_double_changed.emit(asset_kind, asset_id, requested)
        if asset_kind == ASSET_KIND_DOOR:
            self.make_double_changed.emit(asset_id, requested)

    def _handle_copy_slot_shortcut(self) -> None:
        """Remember one explicit double-opening slot as the mesh copy source."""

        asset_kind = self.selected_asset_kind()
        asset_id = self.selected_asset_id()
        asset = self.selected_asset()
        slot = self.selected_asset_slot()
        if (
            asset_kind is None
            or asset_id is None
            or asset is None
            or slot is None
            or not asset.make_double
        ):
            return
        self._copied_asset_slot = (asset_kind, asset_id, slot.slot_id)

    def _handle_paste_slot_shortcut(self) -> None:
        """Request a mirrored mesh copy into the selected counterpart slot."""

        copied = self._copied_asset_slot
        asset_kind = self.selected_asset_kind()
        asset_id = self.selected_asset_id()
        asset = self.selected_asset()
        target = self.selected_asset_slot()
        if (
            copied is None
            or asset_kind is None
            or asset_id is None
            or asset is None
            or target is None
            or not asset.make_double
        ):
            return
        source_kind, source_asset_id, source_slot_id = copied
        if source_kind != asset_kind or source_asset_id != asset_id:
            return
        if door_slot_counterpart(source_slot_id) != target.slot_id:
            return
        if door_slot_component(source_slot_id) != door_slot_component(
            target.slot_id
        ):
            return
        self.asset_slot_mesh_paste_requested.emit(
            asset_kind,
            asset_id,
            source_slot_id,
            target.slot_id,
        )
        if asset_kind == ASSET_KIND_DOOR:
            self.slot_mesh_paste_requested.emit(
                asset_id,
                source_slot_id,
                target.slot_id,
            )

    def _handle_make_double_sided_changed(self, checked: bool) -> None:
        """Request one model-owned single/double-sided asset transition."""

        if self._refreshing:
            return
        asset_kind = self.selected_asset_kind()
        asset_id = self.selected_asset_id()
        asset = self.selected_asset()
        if asset_kind is None or asset_id is None or asset is None:
            self._sync_make_double_sided_checkbox(None)
            return
        requested = bool(checked)
        if requested == asset.make_double_sided:
            return
        self.asset_make_double_sided_changed.emit(
            asset_kind,
            asset_id,
            requested,
        )
        if asset_kind == ASSET_KIND_DOOR:
            self.make_double_sided_changed.emit(asset_id, requested)

    # ### Detail presentation ###
    def _refresh_rotation_status(self) -> None:
        """Show the persisted XYZ rotation for the selected component slot."""

        slot = self.selected_asset_slot()
        self.rotation_status_label.setText(
            "Rotation: -"
            if slot is None
            else self._format_rotation_status(slot.rotation_degrees)
        )

    @staticmethod
    def _format_rotation_status(rotation_degrees: object) -> str:
        """Return compact, validated XYZ rotation text for the controls."""

        try:
            rotation = tuple(
                float(value) for value in rotation_degrees  # type: ignore[union-attr]
            )
        except (TypeError, ValueError):
            return "Rotation: -"
        if len(rotation) != 3 or not all(
            math.isfinite(value) for value in rotation
        ):
            return "Rotation: -"
        rotation = tuple(0.0 if abs(value) < 5e-10 else value for value in rotation)
        return (
            f"Rotation: X {rotation[0]:.1f} degrees, "
            f"Y {rotation[1]:.1f} degrees, Z {rotation[2]:.1f} degrees"
        )

    def _refresh_slot_details(self) -> None:
        asset_kind = self.selected_asset_kind()
        asset_id = self.selected_asset_id()
        asset = self.selected_asset()
        slot = self.selected_asset_slot()
        glass_area = self.selected_glass_area()
        self._sync_make_double_checkbox(asset)
        self._sync_make_double_sided_checkbox(asset)
        self.proportional_editing_checkbox.setVisible(
            bool(asset is not None and asset.make_double)
        )
        self.delete_asset_button.setEnabled(asset is not None)
        self.flip_mesh_horizontally_button.setEnabled(slot is not None)
        self._sync_glass_action_button()
        if asset is None or asset_kind is None or asset_id is None:
            self.confirm_slot_position_button.setEnabled(False)
            return

        if glass_area is not None:
            self.confirm_slot_position_button.setEnabled(False)
            return
        if slot is None:
            self.confirm_slot_position_button.setEnabled(False)
            return

        pending = self.is_asset_slot_position_pending(
            asset_kind,
            asset_id,
            slot.slot_id,
        )
        self.confirm_slot_position_button.setEnabled(
            not is_door_body_slot(slot.slot_id) and pending
        )

    def _sync_glass_action_button(self) -> None:
        """Present one unambiguous glass action for the current selection."""

        asset = self.selected_asset()
        slot = self.selected_asset_slot()
        selected_faces = self.preview_viewer.get_selected_face_indices()
        converts_faces = bool(
            asset is not None
            and slot is not None
            and selected_faces
            and self._preview_face_edit_object_id == slot.source_object_id
        )
        self.add_glass_area_button.setText(
            "Convert faces to glass" if converts_faces else "Add glass area"
        )
        self.add_glass_area_button.setToolTip(
            (
                "Convert the selected component faces to the shared glass "
                "material."
            )
            if converts_faces
            else (
                "Drag a rectangle through the selected door or window body "
                "and fill the cutout with the shared glass material."
            )
        )
        self.add_glass_area_button.setEnabled(asset is not None)

    def _sync_make_double_checkbox(
        self,
        asset: DoorDefinition | WindowDefinition | None,
    ) -> None:
        """Reflect the selected asset's leaf structure without a request."""

        blocker = QSignalBlocker(self.make_double_checkbox)
        self.make_double_checkbox.setChecked(
            False if asset is None else asset.make_double
        )
        self.make_double_checkbox.setEnabled(asset is not None)
        del blocker

    def is_proportional_editing_enabled(self) -> bool:
        """Return whether paired leaf parts should transform together."""

        asset = self.selected_asset()
        return bool(
            self.proportional_editing_checkbox.isChecked()
            and asset is not None
            and asset.make_double
        )

    def _sync_make_double_sided_checkbox(
        self,
        asset: DoorDefinition | WindowDefinition | None,
    ) -> None:
        """Reflect selected-asset sidedness without emitting an edit request."""

        blocker = QSignalBlocker(self.make_double_sided_checkbox)
        self.make_double_sided_checkbox.setChecked(
            False if asset is None else asset.make_double_sided
        )
        self.make_double_sided_checkbox.setEnabled(asset is not None)
        del blocker

    @staticmethod
    def _asset_id(asset: DoorDefinition | WindowDefinition) -> str:
        if isinstance(asset, DoorDefinition):
            return asset.door_id
        return asset.window_id

    @staticmethod
    def _normalize_asset_kind(asset_kind: str) -> str:
        normalized = str(asset_kind).strip().lower()
        if normalized not in {ASSET_KIND_DOOR, ASSET_KIND_WINDOW}:
            raise ValueError("Asset kind must be 'door' or 'window'.")
        return normalized

    def _get_asset(
        self,
        asset_kind: str,
        asset_id: str,
    ) -> DoorDefinition | WindowDefinition | None:
        if asset_kind == ASSET_KIND_DOOR:
            return self._data.get_door(asset_id)
        if asset_kind == ASSET_KIND_WINDOW:
            return self._data.get_window(asset_id)
        return None

    # ### State cleanup ###
    def _discard_stale_pending_slots(self) -> None:
        valid_slots = {
            (ASSET_KIND_DOOR, door.door_id, slot.slot_id)
            for door in self._data.doors
            for slot in door.slots
            if not is_door_body_slot(slot.slot_id)
        }
        valid_slots.update(
            {
                (ASSET_KIND_WINDOW, window.window_id, slot.slot_id)
                for window in self._data.windows
                for slot in window.slots
                if not is_door_body_slot(slot.slot_id)
            }
        )
        self._pending_slot_positions.intersection_update(valid_slots)

