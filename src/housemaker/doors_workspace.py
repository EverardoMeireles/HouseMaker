# ### Imports ###
from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import QSignalBlocker, Qt, Signal, Slot
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFormLayout,
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
    DOOR_SLOT_BODY,
    DOOR_SLOT_KNOB,
    DoorDefinition,
    DoorLibraryData,
    DoorSlotData,
    door_slot_side,
    is_door_body_slot,
)
from housemaker.glb import GeneratedModel
from housemaker.viewer import GlbViewerWidget

# ### Constants ###
ITEM_KIND_ROLE = Qt.ItemDataRole.UserRole
ITEM_DOOR_ID_ROLE = Qt.ItemDataRole.UserRole + 1
ITEM_SLOT_ID_ROLE = Qt.ItemDataRole.UserRole + 2

ITEM_KIND_DOOR = "door"
ITEM_KIND_FACE = "face"
ITEM_KIND_SLOT = "slot"

SINGLE_SIDED_SLOT_ORDER = (
    DOOR_SLOT_BODY,
    DOOR_SLOT_KNOB,
)
DOUBLE_SIDED_SLOT_ORDER = (
    DOOR_SLOT_BODY,
    DOOR_SLOT_KNOB,
    DOOR_SLOT_BACK_BODY,
    DOOR_SLOT_BACK_KNOB,
)
DOOR_FACE_ORDER = ("front", "back")
DOOR_FACE_DISPLAY_NAMES = {
    "front": "Front",
    "back": "Back",
}


# ### Doors workspace ###
class DoorsWorkspace(QWidget):
    """Display reusable doors and coordinate edits to their component slots."""

    selection_changed = Signal(object, object)
    slot_edit_requested = Signal(str, str)
    slot_confirmation_requested = Signal(str, str)
    slot_transform_changed = Signal(str, str, object, object)
    slot_axis_scales_changed = Signal(str, str, object)
    make_double_sided_changed = Signal(str, bool)
    generate_displacement_changed = Signal(str, bool)
    undo_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("doors_workspace")
        self._data = DoorLibraryData()
        self._pending_slot_positions: set[tuple[str, str]] = set()
        self._previewed_door_id: str | None = None
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

    def selected_door_id(self) -> str | None:
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

    def selected_slot(self) -> DoorSlotData | None:
        door = self.selected_door()
        slot_id = self.selected_slot_id()
        if door is None or slot_id is None:
            return None
        return door.get_slot(slot_id)

    def select_door_slot(self, door_id: str, slot_id: str | None = None) -> bool:
        """Select a door or one of its slots by persistent identifiers."""

        item = self._find_item(str(door_id).strip(), slot_id)
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

        normalized_door_id = str(door_id).strip()
        normalized_slot_id = str(slot_id).strip().lower()
        door = self._data.get_door(normalized_door_id)
        if door is None or door.get_slot(normalized_slot_id) is None:
            return False
        key = (normalized_door_id, normalized_slot_id)
        if pending and not is_door_body_slot(normalized_slot_id):
            self._pending_slot_positions.add(key)
        else:
            self._pending_slot_positions.discard(key)
        self._refresh_slot_details()
        return True

    def is_slot_position_pending(self, door_id: str, slot_id: str) -> bool:
        return (str(door_id).strip(), str(slot_id).strip().lower()) in (
            self._pending_slot_positions
        )

    def clear_pending_slot_positions(self) -> None:
        self._pending_slot_positions.clear()
        self._refresh_tree()

    # ### Complete-door preview API ###
    def set_preview_model(self, model: GeneratedModel | None) -> None:
        """Display the assembled door and edit the selected hardware slot."""

        if model is None:
            self.preview_viewer.set_selected_placed_object_ids(())
            self.preview_viewer.clear_model()
            self._previewed_door_id = None
            return
        if not isinstance(model, GeneratedModel):
            raise TypeError("Door previews require a GeneratedModel or None.")
        selected_door_id = self.selected_door_id()
        self.preview_viewer.set_model(
            model,
            preserve_camera=bool(
                self.preview_viewer.model is not None
                and selected_door_id is not None
                and selected_door_id == self._previewed_door_id
            ),
        )
        self._previewed_door_id = selected_door_id
        self._sync_preview_slot_gizmo_selection(preserve_mode=True)

    def _sync_preview_slot_gizmo_selection(
        self,
        *,
        preserve_mode: bool = False,
    ) -> None:
        """Show transform and scale gizmos for the selected generated slot."""

        slot = self.selected_slot()
        model = self.preview_viewer.model
        editable_object_id = None
        if slot is not None and model is not None:
            has_preview = any(
                preview.object_id == slot.source_object_id
                for preview in model.preview_placed_objects
            )
            if has_preview:
                editable_object_id = slot.source_object_id
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
            transform_enabled=not body_selected,
            axis_scale_enabled=True,
            prefer_axis_scale=body_selected or preserve_axis_scale,
        )
        self.preview_viewer.set_selected_placed_object_ids(
            () if editable_object_id is None else (editable_object_id,),
            active_object_id=editable_object_id,
        )

    # ### UI construction ###
    def _build_ui(self) -> None:
        controls_widget = QWidget(self)
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)

        self.door_tree = QTreeWidget(self)
        self.door_tree.setObjectName("doors_tree")
        self.door_tree.setHeaderLabels(("Door / slot",))
        self.door_tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.door_tree.setRootIsDecorated(True)
        self.door_tree.setUniformRowHeights(True)
        self.door_tree.setToolTip(
            "Double-click a component slot to open it in Generation."
        )
        self.door_tree.header().setStretchLastSection(True)
        self.door_tree.setColumnWidth(0, 190)

        self.make_double_sided_checkbox = QCheckBox(
            "Make double sided",
            self,
        )
        self.make_double_sided_checkbox.setObjectName(
            "make_door_double_sided_checkbox"
        )
        self.make_double_sided_checkbox.setToolTip(
            "Create independently generated Front and Back body and "
            "door knob/handle slots."
        )
        self.generate_displacement_checkbox = QCheckBox(
            "Generate displacement",
            self,
        )
        self.generate_displacement_checkbox.setObjectName(
            "generate_door_displacement_checkbox"
        )
        self.generate_displacement_checkbox.setToolTip(
            "Make Generate replace door-body geometry through Meshy so raised "
            "and recessed details are retained instead of only retexturing "
            "the fitted body."
        )
        door_options_widget = QWidget(self)
        door_options_layout = QHBoxLayout(door_options_widget)
        door_options_layout.setContentsMargins(0, 0, 0, 0)
        door_options_layout.setSpacing(8)
        door_options_layout.addWidget(self.make_double_sided_checkbox)
        door_options_layout.addWidget(self.generate_displacement_checkbox)
        door_options_layout.addStretch(1)

        self.profile_group = QGroupBox("Door profile", self)
        profile_layout = QFormLayout(self.profile_group)
        self.profile_name_label = QLabel("-", self.profile_group)
        self.profile_shape_label = QLabel("-", self.profile_group)
        self.profile_dimensions_label = QLabel("-", self.profile_group)
        self.profile_source_label = QLabel("-", self.profile_group)
        self.selected_slot_label = QLabel("-", self.profile_group)
        profile_layout.addRow("Name", self.profile_name_label)
        profile_layout.addRow("Silhouette", self.profile_shape_label)
        profile_layout.addRow("Dimensions", self.profile_dimensions_label)
        profile_layout.addRow("Source doorway", self.profile_source_label)
        profile_layout.addRow("Selected slot", self.selected_slot_label)

        self.confirm_slot_position_button = QPushButton(
            "Confirm slot transform",
            self,
        )
        self.confirm_slot_position_button.setObjectName(
            "confirm_door_slot_position_button"
        )
        self.confirm_slot_position_button.setEnabled(False)

        controls_layout.addWidget(door_options_widget)
        controls_layout.addWidget(self.door_tree, 1)
        controls_layout.addWidget(self.profile_group)
        controls_layout.addWidget(self.confirm_slot_position_button)

        preview_group = QGroupBox("Complete door preview", self)
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(4, 4, 4, 4)
        self.wireframe_checkbox = QCheckBox("Wireframe", preview_group)
        self.wireframe_checkbox.setObjectName(
            "door_preview_wireframe_checkbox"
        )
        self.wireframe_checkbox.setChecked(False)
        self.wireframe_checkbox.setToolTip(
            "Show mesh edges over the complete door preview."
        )
        self.preview_viewer = GlbViewerWidget(
            preview_group,
            wireframe_enabled=False,
            placed_object_editing_enabled=True,
            placed_object_click_selection_enabled=True,
            placed_object_auxiliary_controls_enabled=False,
            placed_object_axis_scale_gizmos_enabled=True,
            face_editing_enabled=False,
            symmetric_preview_fade_enabled=False,
        )
        self.preview_viewer.setObjectName("door_complete_preview_viewer")
        preview_layout.addWidget(self.wireframe_checkbox)
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

    def _connect_signals(self) -> None:
        self.door_tree.currentItemChanged.connect(
            self._handle_current_item_changed
        )
        self.door_tree.itemDoubleClicked.connect(
            self._handle_item_double_clicked
        )
        self.make_double_sided_checkbox.toggled.connect(
            self._handle_make_double_sided_changed
        )
        self.generate_displacement_checkbox.toggled.connect(
            self._handle_generate_displacement_changed
        )
        self.confirm_slot_position_button.clicked.connect(
            self._handle_confirm_slot_position_clicked
        )
        self.wireframe_checkbox.toggled.connect(
            self.preview_viewer.set_wireframe_enabled
        )
        self.preview_viewer.placed_object_transform_changed.connect(
            self._handle_preview_slot_transform_changed
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
        self.undo_shortcut.activated.connect(self._emit_undo_requested)
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
        select_slot_id: str | None = None,
    ) -> None:
        previous_selection = (
            self.selected_door_id(),
            self.selected_slot_id(),
        )
        requested_selection = (
            select_door_id,
            select_slot_id,
        )
        if select_door_id is None:
            requested_selection = previous_selection

        self._refreshing = True
        blocker = QSignalBlocker(self.door_tree)
        self.door_tree.clear()
        for door in self._data.doors:
            door_item = self._build_door_item(door)
            self.door_tree.addTopLevelItem(door_item)
            door_item.setExpanded(True)
            for child_index in range(door_item.childCount()):
                child = door_item.child(child_index)
                if child.data(0, ITEM_KIND_ROLE) == ITEM_KIND_FACE:
                    child.setExpanded(True)
        selected_item = self._find_item(*requested_selection)
        if selected_item is None and self.door_tree.topLevelItemCount() > 0:
            first_door_item = self.door_tree.topLevelItem(0)
            selected_item = self._find_item(
                first_door_item.data(0, ITEM_DOOR_ID_ROLE),
                DOOR_SLOT_BODY,
            )
            if selected_item is None:
                selected_item = first_door_item
        self.door_tree.setCurrentItem(selected_item)
        del blocker
        self._refreshing = False
        self._refresh_slot_details()

        current_selection = (
            self.selected_door_id(),
            self.selected_slot_id(),
        )
        if current_selection != previous_selection:
            self._emit_selection_changed()

    def _build_door_item(self, door: DoorDefinition) -> QTreeWidgetItem:
        item = QTreeWidgetItem((door.name,))
        item.setData(0, ITEM_KIND_ROLE, ITEM_KIND_DOOR)
        item.setData(0, ITEM_DOOR_ID_ROLE, door.door_id)
        slots_by_id = {slot.slot_id: slot for slot in door.slots}
        if not door.make_double_sided:
            for slot_id in SINGLE_SIDED_SLOT_ORDER:
                slot = slots_by_id.get(slot_id)
                if slot is not None:
                    item.addChild(self._build_slot_item(door, slot))
            return item

        for face in DOOR_FACE_ORDER:
            face_item = QTreeWidgetItem((DOOR_FACE_DISPLAY_NAMES[face],))
            face_item.setData(0, ITEM_KIND_ROLE, ITEM_KIND_FACE)
            face_item.setData(0, ITEM_DOOR_ID_ROLE, door.door_id)
            for slot_id in DOUBLE_SIDED_SLOT_ORDER:
                if door_slot_side(slot_id) != face:
                    continue
                slot = slots_by_id.get(slot_id)
                if slot is not None:
                    face_item.addChild(self._build_slot_item(door, slot))
            item.addChild(face_item)
        return item

    def _build_slot_item(
        self,
        door: DoorDefinition,
        slot: DoorSlotData,
    ) -> QTreeWidgetItem:
        """Build one leaf shared by flat and face-grouped door trees."""

        slot_item = QTreeWidgetItem((slot.display_name,))
        slot_item.setData(0, ITEM_KIND_ROLE, ITEM_KIND_SLOT)
        slot_item.setData(0, ITEM_DOOR_ID_ROLE, door.door_id)
        slot_item.setData(0, ITEM_SLOT_ID_ROLE, slot.slot_id)
        return slot_item

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
        if not door_id:
            return None
        normalized_slot_id = str(slot_id).strip().lower() if slot_id else None
        for index in range(self.door_tree.topLevelItemCount()):
            door_item = self.door_tree.topLevelItem(index)
            if door_item.data(0, ITEM_DOOR_ID_ROLE) != door_id:
                continue
            if normalized_slot_id is None:
                return door_item
            return self._find_slot_item(door_item, normalized_slot_id)
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
        self._sync_preview_slot_gizmo_selection()
        self._emit_selection_changed()

    @Slot(object)
    def _handle_preview_object_selection_changed(
        self,
        object_id: object,
    ) -> None:
        """Select the door slot whose mesh was picked in the 3D preview."""

        previewed_door_id = self._previewed_door_id
        if previewed_door_id is None or object_id is None:
            return
        door = self._data.get_door(previewed_door_id)
        if door is None:
            return
        slot = self._find_preview_slot(door, object_id)
        if slot is None or (
            self.selected_door_id() == door.door_id
            and self.selected_slot_id() == slot.slot_id
        ):
            return
        self.select_door_slot(door.door_id, slot.slot_id)

    @Slot(str)
    def _handle_preview_object_double_clicked(self, object_id: str) -> None:
        """Open Generation for the concrete door part double-clicked in 3D."""

        previewed_door_id = self._previewed_door_id
        if previewed_door_id is None:
            return
        door = self._data.get_door(previewed_door_id)
        if door is None:
            return
        slot = self._find_preview_slot(door, object_id)
        if slot is None:
            return
        self.select_door_slot(door.door_id, slot.slot_id)
        self.slot_edit_requested.emit(door.door_id, slot.slot_id)

    @staticmethod
    def _find_preview_slot(
        door: DoorDefinition,
        object_id: object,
    ) -> DoorSlotData | None:
        """Resolve one preview-object identity to its persistent door slot."""

        normalized_object_id = str(object_id).strip()
        return next(
            (
                slot
                for slot in door.slots
                if slot.source_object_id == normalized_object_id
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

        door = self.selected_door()
        slot = self.selected_slot()
        if (
            door is None
            or slot is None
            or is_door_body_slot(slot.slot_id)
            or str(object_id) != slot.source_object_id
        ):
            return
        self.slot_transform_changed.emit(
            door.door_id,
            slot.slot_id,
            world_position,
            rotation_degrees,
        )

    @Slot(str, object)
    def _handle_preview_slot_axis_scales_changed(
        self,
        object_id: str,
        axis_scales: object,
    ) -> None:
        """Forward local XYZ scales for any selected generated door slot."""

        door = self.selected_door()
        slot = self.selected_slot()
        if (
            door is None
            or slot is None
            or str(object_id) != slot.source_object_id
        ):
            return
        self.slot_axis_scales_changed.emit(
            door.door_id,
            slot.slot_id,
            axis_scales,
        )

    def _emit_selection_changed(self) -> None:
        door_id = self.selected_door_id()
        slot_id = self.selected_slot_id()
        self.selection_changed.emit(door_id, slot_id)

    def _handle_item_double_clicked(
        self,
        item: QTreeWidgetItem,
        _column: int,
    ) -> None:
        """Request Generation editing only for a concrete component slot."""

        if item.data(0, ITEM_KIND_ROLE) != ITEM_KIND_SLOT:
            return
        door_id = item.data(0, ITEM_DOOR_ID_ROLE)
        slot_id = item.data(0, ITEM_SLOT_ID_ROLE)
        if not isinstance(door_id, str) or not isinstance(slot_id, str):
            return
        if not door_id.strip() or not slot_id.strip():
            return
        self.slot_edit_requested.emit(door_id, slot_id)

    def _handle_confirm_slot_position_clicked(self) -> None:
        door_id = self.selected_door_id()
        slot_id = self.selected_slot_id()
        if (
            door_id is None
            or slot_id is None
            or is_door_body_slot(slot_id)
            or not self.is_slot_position_pending(door_id, slot_id)
        ):
            return
        self.slot_confirmation_requested.emit(door_id, slot_id)

    def _handle_make_double_sided_changed(self, checked: bool) -> None:
        """Request one model-owned single/double-sided door transition."""

        if self._refreshing:
            return
        door = self.selected_door()
        if door is None:
            self._sync_make_double_sided_checkbox(None)
            return
        requested = bool(checked)
        if requested == door.make_double_sided:
            return
        self.make_double_sided_changed.emit(door.door_id, requested)

    def _handle_generate_displacement_changed(self, checked: bool) -> None:
        """Request a model-owned door displacement-mode update."""

        if self._refreshing:
            return
        door = self.selected_door()
        if door is None:
            self._sync_generate_displacement_checkbox(None)
            return
        requested = bool(checked)
        if requested == door.generate_displacement:
            return
        self.generate_displacement_changed.emit(door.door_id, requested)

    # ### Detail presentation ###
    def _refresh_slot_details(self) -> None:
        door = self.selected_door()
        slot = self.selected_slot()
        self._sync_make_double_sided_checkbox(door)
        self._sync_generate_displacement_checkbox(door)
        if door is None:
            self.profile_name_label.setText("-")
            self.profile_shape_label.setText("-")
            self.profile_dimensions_label.setText("-")
            self.profile_source_label.setText("-")
            self.selected_slot_label.setText("-")
            self.confirm_slot_position_button.setEnabled(False)
            return

        self.profile_name_label.setText(door.name)
        self.profile_shape_label.setText(self._door_profile_text(door))
        self.profile_dimensions_label.setText(self._door_dimensions_text(door))
        self.profile_source_label.setText(door.source_doorway_id)
        if slot is None:
            self.selected_slot_label.setText("Door")
            self.confirm_slot_position_button.setEnabled(False)
            return

        pending = self.is_slot_position_pending(door.door_id, slot.slot_id)
        status = "transform pending" if pending else (
            "joined" if slot.joined else "editable"
        )
        self.selected_slot_label.setText(f"{slot.display_name} - {status}")
        self.confirm_slot_position_button.setEnabled(
            not is_door_body_slot(slot.slot_id) and pending
        )

    def _sync_make_double_sided_checkbox(
        self,
        door: DoorDefinition | None,
    ) -> None:
        """Reflect selected-door sidedness without emitting an edit request."""

        blocker = QSignalBlocker(self.make_double_sided_checkbox)
        self.make_double_sided_checkbox.setChecked(
            False if door is None else door.make_double_sided
        )
        self.make_double_sided_checkbox.setEnabled(door is not None)
        del blocker

    def _sync_generate_displacement_checkbox(
        self,
        door: DoorDefinition | None,
    ) -> None:
        """Reflect the selected door's displacement generation preference."""

        blocker = QSignalBlocker(self.generate_displacement_checkbox)
        self.generate_displacement_checkbox.setChecked(
            False if door is None else door.generate_displacement
        )
        self.generate_displacement_checkbox.setEnabled(door is not None)
        del blocker

    @staticmethod
    def _door_profile_text(door: DoorDefinition) -> str:
        if door.shape == "arch":
            return f"Arched ({door.arch_amount * 100.0:.0f}%)"
        return "Rectangular"

    @staticmethod
    def _door_dimensions_text(door: DoorDefinition) -> str:
        return (
            f"{door.width_meters:.3f} m W x "
            f"{door.height_meters:.3f} m H x "
            f"{door.thickness_meters:.3f} m thick"
        )

    # ### State cleanup ###
    def _discard_stale_pending_slots(self) -> None:
        valid_slots = {
            (door.door_id, slot.slot_id)
            for door in self._data.doors
            for slot in door.slots
            if not is_door_body_slot(slot.slot_id)
        }
        self._pending_slot_positions.intersection_update(valid_slots)

