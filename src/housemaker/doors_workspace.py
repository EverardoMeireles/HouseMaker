# ### Imports ###
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from housemaker.door_state import (
    DOOR_SIDE_DUPLICATION_BACK,
    DOOR_SIDE_DUPLICATION_FRONT,
    DOOR_SLOT_BODY,
    DOOR_SLOT_HINGES,
    DOOR_SLOT_KNOB,
    DoorDefinition,
    DoorLibraryData,
    DoorSideDuplication,
    DoorSlotData,
)
from housemaker.glb import GeneratedModel
from housemaker.viewer import GlbViewerWidget

# ### Constants ###
ITEM_KIND_ROLE = Qt.ItemDataRole.UserRole
ITEM_DOOR_ID_ROLE = Qt.ItemDataRole.UserRole + 1
ITEM_SLOT_ID_ROLE = Qt.ItemDataRole.UserRole + 2

ITEM_KIND_DOOR = "door"
ITEM_KIND_SLOT = "slot"

DOOR_SLOT_ORDER = (
    DOOR_SLOT_BODY,
    DOOR_SLOT_HINGES,
    DOOR_SLOT_KNOB,
)
SIDE_DUPLICATION_TOOLTIP = (
    "Keep only the selected front/back half of the generated door body "
    "and reconstruct the omitted side in the R3F app."
)
SIDE_DUPLICATION_LOCKED_TOOLTIP = (
    "Side duplication is already applied to this generated door body. "
    "Its saved side is locked so the controls match the geometry and export."
)


# ### Doors workspace ###
class DoorsWorkspace(QWidget):
    """Display reusable doors and coordinate edits to their component slots."""

    selection_changed = Signal(object, object)
    slot_edit_requested = Signal(str, str)
    slot_confirmation_requested = Signal(str, str)
    door_definition_changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("doors_workspace")
        self._data = DoorLibraryData()
        self._pending_slot_positions: set[tuple[str, str]] = set()
        self._previewed_door_id: str | None = None
        self._persisted_side_duplication_by_door_id: dict[
            str,
            DoorSideDuplication,
        ] = {}
        self._refreshing = False
        self._syncing_side_duplication_controls = False
        self._build_ui()
        self._connect_signals()
        self._refresh_tree()

    # ### Public model API ###
    def set_data(self, data: DoorLibraryData) -> None:
        """Replace the library while preserving a still-valid selection."""

        if not isinstance(data, DoorLibraryData):
            raise TypeError("DoorsWorkspace.set_data expects DoorLibraryData.")
        self._data = data.clone()
        self._persisted_side_duplication_by_door_id.clear()
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

    def set_persisted_side_duplication(
        self,
        door_id: str,
        side_duplication: DoorSideDuplication | None,
    ) -> DoorDefinition | None:
        """Lock a generated body's applied side and reconcile displayed state.

        Passing ``None`` removes the lock while retaining the door's editable
        preference. A concrete value is authoritative because the generated
        mesh and export metadata can no longer be changed by these controls.
        """

        normalized_door_id = str(door_id).strip()
        door = self._data.get_door(normalized_door_id)
        if door is None:
            self._persisted_side_duplication_by_door_id.pop(
                normalized_door_id,
                None,
            )
            return None
        if side_duplication is None:
            self._persisted_side_duplication_by_door_id.pop(
                normalized_door_id,
                None,
            )
            if self.selected_door_id() == normalized_door_id:
                self._refresh_slot_details()
            return door
        if not isinstance(side_duplication, DoorSideDuplication):
            raise TypeError(
                "Persisted door side duplication requires DoorSideDuplication."
            )
        self._persisted_side_duplication_by_door_id[normalized_door_id] = (
            side_duplication
        )
        if door.side_duplication != side_duplication:
            door = replace(door, side_duplication=side_duplication)
            self.upsert_door(door)
        elif self.selected_door_id() == normalized_door_id:
            self._refresh_slot_details()
        return door

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
        self._persisted_side_duplication_by_door_id.pop(normalized_id, None)
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
        if pending and normalized_slot_id != DOOR_SLOT_BODY:
            self._pending_slot_positions.add(key)
        else:
            self._pending_slot_positions.discard(key)
        item = self._find_item(normalized_door_id, normalized_slot_id)
        slot = door.get_slot(normalized_slot_id)
        if item is not None and slot is not None:
            item.setText(1, self._slot_summary(door, slot))
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
        """Display or clear the assembled door without enabling edit gizmos."""

        if model is None:
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

    # ### UI construction ###
    def _build_ui(self) -> None:
        controls_widget = QWidget(self)
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)

        self.door_tree = QTreeWidget(self)
        self.door_tree.setObjectName("doors_tree")
        self.door_tree.setHeaderLabels(("Door / slot", "Details"))
        self.door_tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.door_tree.setRootIsDecorated(True)
        self.door_tree.setUniformRowHeights(True)
        self.door_tree.setToolTip(
            "Double-click a component slot to open it in Generation."
        )
        self.door_tree.header().setStretchLastSection(True)
        self.door_tree.setColumnWidth(0, 190)

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

        self.side_duplication_control = QWidget(self.profile_group)
        side_duplication_layout = QHBoxLayout(self.side_duplication_control)
        side_duplication_layout.setContentsMargins(0, 0, 0, 0)
        side_duplication_layout.setSpacing(8)
        self.side_duplication_checkbox = QCheckBox(
            "Side duplication",
            self.side_duplication_control,
        )
        self.side_duplication_checkbox.setObjectName(
            "door_side_duplication_checkbox"
        )
        self.side_duplication_checkbox.setToolTip(
            SIDE_DUPLICATION_TOOLTIP
        )
        self.side_duplication_front_radio = QRadioButton(
            "Front",
            self.side_duplication_control,
        )
        self.side_duplication_front_radio.setObjectName(
            "door_side_duplication_front_radio"
        )
        self.side_duplication_back_radio = QRadioButton(
            "Back",
            self.side_duplication_control,
        )
        self.side_duplication_back_radio.setObjectName(
            "door_side_duplication_back_radio"
        )
        self.side_duplication_button_group = QButtonGroup(self)
        self.side_duplication_button_group.setExclusive(True)
        self.side_duplication_button_group.addButton(
            self.side_duplication_front_radio
        )
        self.side_duplication_button_group.addButton(
            self.side_duplication_back_radio
        )
        self.side_duplication_front_radio.setChecked(True)
        side_duplication_layout.addWidget(self.side_duplication_checkbox)
        side_duplication_layout.addWidget(self.side_duplication_front_radio)
        side_duplication_layout.addWidget(self.side_duplication_back_radio)
        side_duplication_layout.addStretch(1)
        profile_layout.addRow(self.side_duplication_control)

        self.confirm_slot_position_button = QPushButton(
            "Confirm slot position",
            self,
        )
        self.confirm_slot_position_button.setObjectName(
            "confirm_door_slot_position_button"
        )
        self.confirm_slot_position_button.setEnabled(False)

        controls_layout.addWidget(self.door_tree, 1)
        controls_layout.addWidget(self.profile_group)
        controls_layout.addWidget(self.confirm_slot_position_button)

        preview_group = QGroupBox("Complete door preview", self)
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(4, 4, 4, 4)
        self.preview_viewer = GlbViewerWidget(
            preview_group,
            wireframe_enabled=False,
            placed_object_editing_enabled=False,
            placed_object_auxiliary_controls_enabled=False,
            face_editing_enabled=False,
        )
        self.preview_viewer.setObjectName("door_complete_preview_viewer")
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

    def _connect_signals(self) -> None:
        self.door_tree.currentItemChanged.connect(
            self._handle_current_item_changed
        )
        self.door_tree.itemDoubleClicked.connect(
            self._handle_item_double_clicked
        )
        self.confirm_slot_position_button.clicked.connect(
            self._handle_confirm_slot_position_clicked
        )
        self.side_duplication_checkbox.toggled.connect(
            self._handle_side_duplication_changed
        )
        self.side_duplication_front_radio.toggled.connect(
            self._handle_side_duplication_changed
        )
        self.side_duplication_back_radio.toggled.connect(
            self._handle_side_duplication_changed
        )

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
        selected_item = self._find_item(*requested_selection)
        if selected_item is None and self.door_tree.topLevelItemCount() > 0:
            first_door_item = self.door_tree.topLevelItem(0)
            selected_item = (
                first_door_item.child(0)
                if first_door_item.childCount() > 0
                else first_door_item
            )
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
        item = QTreeWidgetItem((door.name, self._door_profile_text(door)))
        item.setData(0, ITEM_KIND_ROLE, ITEM_KIND_DOOR)
        item.setData(0, ITEM_DOOR_ID_ROLE, door.door_id)
        slots_by_id = {slot.slot_id: slot for slot in door.slots}
        for slot_id in DOOR_SLOT_ORDER:
            slot = slots_by_id.get(slot_id)
            if slot is None:
                continue
            slot_item = QTreeWidgetItem(
                (slot.display_name, self._slot_summary(door, slot))
            )
            slot_item.setData(0, ITEM_KIND_ROLE, ITEM_KIND_SLOT)
            slot_item.setData(0, ITEM_DOOR_ID_ROLE, door.door_id)
            slot_item.setData(0, ITEM_SLOT_ID_ROLE, slot.slot_id)
            item.addChild(slot_item)
        return item

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
            for child_index in range(door_item.childCount()):
                child = door_item.child(child_index)
                if child.data(0, ITEM_SLOT_ID_ROLE) == normalized_slot_id:
                    return child
            return None
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
        self._emit_selection_changed()

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
            or slot_id == DOOR_SLOT_BODY
            or not self.is_slot_position_pending(door_id, slot_id)
        ):
            return
        self.slot_confirmation_requested.emit(door_id, slot_id)

    # ### Detail presentation ###
    def _refresh_slot_details(self) -> None:
        door = self.selected_door()
        slot = self.selected_slot()
        self._sync_side_duplication_controls(door)
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
        status = "position pending" if pending else (
            "joined" if slot.joined else "editable"
        )
        self.selected_slot_label.setText(f"{slot.display_name} - {status}")
        self.confirm_slot_position_button.setEnabled(
            slot.slot_id != DOOR_SLOT_BODY and pending
        )

    def _sync_side_duplication_controls(
        self,
        door: DoorDefinition | None,
    ) -> None:
        """Reflect the selected door's body reconstruction settings."""

        self._syncing_side_duplication_controls = True
        blockers = (
            QSignalBlocker(self.side_duplication_checkbox),
            QSignalBlocker(self.side_duplication_front_radio),
            QSignalBlocker(self.side_duplication_back_radio),
        )
        try:
            side_duplication = None if door is None else door.side_duplication
            enabled = side_duplication is not None
            kept_side = (
                DOOR_SIDE_DUPLICATION_FRONT
                if side_duplication is None
                else side_duplication.kept_side
            )
            self.side_duplication_checkbox.setChecked(enabled)
            self.side_duplication_front_radio.setChecked(
                kept_side == DOOR_SIDE_DUPLICATION_FRONT
            )
            self.side_duplication_back_radio.setChecked(
                kept_side == DOOR_SIDE_DUPLICATION_BACK
            )
            locked = bool(
                door is not None
                and door.door_id
                in self._persisted_side_duplication_by_door_id
            )
            controls_tooltip = (
                SIDE_DUPLICATION_LOCKED_TOOLTIP
                if locked
                else SIDE_DUPLICATION_TOOLTIP
            )
            self.side_duplication_checkbox.setToolTip(controls_tooltip)
            self.side_duplication_front_radio.setToolTip(controls_tooltip)
            self.side_duplication_back_radio.setToolTip(controls_tooltip)
            self.side_duplication_checkbox.setEnabled(
                door is not None and not locked
            )
            self.side_duplication_front_radio.setEnabled(
                door is not None and enabled and not locked
            )
            self.side_duplication_back_radio.setEnabled(
                door is not None and enabled and not locked
            )
        finally:
            del blockers
            self._syncing_side_duplication_controls = False

    def _handle_side_duplication_changed(self, _checked: bool) -> None:
        """Persist checkbox/radio changes on the selected door definition."""

        if self._syncing_side_duplication_controls:
            return
        door = self.selected_door()
        if door is None:
            return
        if door.door_id in self._persisted_side_duplication_by_door_id:
            self._sync_side_duplication_controls(door)
            return
        side_duplication = None
        if self.side_duplication_checkbox.isChecked():
            kept_side = (
                DOOR_SIDE_DUPLICATION_BACK
                if self.side_duplication_back_radio.isChecked()
                else DOOR_SIDE_DUPLICATION_FRONT
            )
            side_duplication = DoorSideDuplication(kept_side=kept_side)
        updated = replace(door, side_duplication=side_duplication)
        if updated == door:
            self._sync_side_duplication_controls(door)
            return
        self.upsert_door(updated)
        self.door_definition_changed.emit(updated)

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

    def _slot_summary(
        self,
        door: DoorDefinition,
        slot: DoorSlotData,
    ) -> str:
        if slot.slot_id == DOOR_SLOT_BODY:
            return f"{self._door_dimensions_text(door)}, {self._door_profile_text(door)}"
        if self.is_slot_position_pending(door.door_id, slot.slot_id):
            return "Position pending"
        return "Joined" if slot.joined else "Editable part"

    # ### State cleanup ###
    def _discard_stale_pending_slots(self) -> None:
        valid_slots = {
            (door.door_id, slot.slot_id)
            for door in self._data.doors
            for slot in door.slots
            if slot.slot_id != DOOR_SLOT_BODY
        }
        self._pending_slot_positions.intersection_update(valid_slots)

