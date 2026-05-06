# ### Imports ###
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSplitter,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from housemaker.blueprint_canvas import BlueprintCanvas
from housemaker.glb import DEFAULT_WALL_HEIGHT_METERS, convert_to_glb
from housemaker.models import (
    DEFAULT_IMAGE_OFFSET,
    DEFAULT_IMAGE_SCALE,
    DEFAULT_UV_MAP_HEIGHT,
    DEFAULT_UV_MAP_WIDTH,
    DEFAULT_WALL_UV_ROTATION_DEGREES,
    DEFAULT_WALL_UV_SCALE,
    GROUND_LEVEL_INDEX,
    LevelData,
    RoomData,
    create_default_levels,
)
from housemaker.project_io import ProjectData, load_project, save_project
from housemaker.uv_canvas import UvCanvas, calculate_unoccupied_uv_pixels
from housemaker.viewer import GlbViewerWindow

# ### Widgets ###
class HomePage(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(24)

        title_label = QLabel("HouseMaker")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 28px; font-weight: 600;")
        layout.addWidget(title_label)
        layout.addStretch(1)

        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(32)

        self.blueprint_button = QPushButton("Blueprint")
        self.blueprint_button.setMinimumSize(280, 220)
        self.blueprint_button.setStyleSheet("font-size: 24px; font-weight: 600;")
        buttons_layout.addWidget(self.blueprint_button)

        self.projection_button = QPushButton("Projection")
        self.projection_button.setMinimumSize(280, 220)
        self.projection_button.setEnabled(False)
        self.projection_button.setStyleSheet("font-size: 24px; font-weight: 600;")
        buttons_layout.addWidget(self.projection_button)

        layout.addLayout(buttons_layout)
        layout.addStretch(2)


class PowerOfTwoSpinBox(QSpinBox):
    def setValue(self, value: int) -> None:  # type: ignore[override]
        super().setValue(_nearest_power_of_two_value(value))

    def stepBy(self, steps: int) -> None:  # type: ignore[override]
        self.setValue(_step_power_of_two_value(self.value(), steps))

    def valueFromText(self, text: str) -> int:  # type: ignore[override]
        try:
            return _nearest_power_of_two_value(int(text or self.minimum()))
        except ValueError:
            return self.value()

    def textFromValue(self, value: int) -> str:  # type: ignore[override]
        return str(_nearest_power_of_two_value(value))


class RightAngleSpinBox(QSpinBox):
    def setValue(self, value: int) -> None:  # type: ignore[override]
        super().setValue(_nearest_right_angle_value(value))

    def stepBy(self, steps: int) -> None:  # type: ignore[override]
        self.setValue(_step_right_angle_value(self.value(), steps))

    def valueFromText(self, text: str) -> int:  # type: ignore[override]
        try:
            return _nearest_right_angle_value(int(text or self.minimum()))
        except ValueError:
            return self.value()

    def textFromValue(self, value: int) -> str:  # type: ignore[override]
        return str(_nearest_right_angle_value(value))


class BlueprintWorkspace(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer_windows: list[GlbViewerWindow] = []
        self.levels: list[LevelData] = create_default_levels()
        self.current_level_index = GROUND_LEVEL_INDEX
        self._is_syncing_level_controls = False
        self._is_syncing_uv_controls = False
        self._build_ui()

    @property
    def vertex_data(self):
        return self.current_level.vertex_data

    @property
    def current_level(self) -> LevelData:
        return self.levels[self.current_level_index]

    def _build_ui(self) -> None:
        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        root_layout.addWidget(splitter, 1)

        self.canvas = BlueprintCanvas()
        splitter.addWidget(self.canvas)

        side_panel = QWidget()
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(16, 16, 16, 16)
        side_layout.setSpacing(12)

        self.side_tabs = QTabWidget()
        side_layout.addWidget(self.side_tabs, 1)

        generals_tab = QWidget()
        generals_layout = QVBoxLayout(generals_tab)
        generals_layout.setContentsMargins(10, 12, 10, 10)
        generals_layout.setSpacing(12)
        self.side_tabs.addTab(generals_tab, "Generals")
        side_layout = generals_layout

        height_label = QLabel("Height level")
        height_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        side_layout.addWidget(height_label)

        self.height_level_spinbox = QDoubleSpinBox()
        self.height_level_spinbox.setRange(0.1, 100.0)
        self.height_level_spinbox.setDecimals(2)
        self.height_level_spinbox.setSingleStep(0.1)
        self.height_level_spinbox.setValue(DEFAULT_WALL_HEIGHT_METERS)
        self.height_level_spinbox.setSuffix(" m")
        self.height_level_spinbox.setMinimumHeight(40)
        self.height_level_spinbox.valueChanged.connect(self._handle_height_level_changed)
        side_layout.addWidget(self.height_level_spinbox)

        self.load_image_button = QPushButton("Load image")
        self.load_image_button.setMinimumHeight(44)
        self.load_image_button.clicked.connect(self._handle_load_image_clicked)
        side_layout.addWidget(self.load_image_button)

        image_transform_layout = QFormLayout()
        image_transform_layout.setContentsMargins(0, 0, 0, 0)
        image_transform_layout.setSpacing(8)

        self.image_scale_spinbox = self._build_image_transform_spinbox(
            value=DEFAULT_IMAGE_SCALE,
            minimum=0.01,
            maximum=20.0,
            decimals=3,
            single_step=0.05,
        )
        self.image_scale_spinbox.valueChanged.connect(self._handle_image_scale_changed)
        image_transform_layout.addRow("Scale", self.image_scale_spinbox)

        self.image_x_offset_spinbox = self._build_image_transform_spinbox(
            value=DEFAULT_IMAGE_OFFSET,
            minimum=-100000.0,
            maximum=100000.0,
            decimals=1,
            single_step=1.0,
            suffix=" px",
        )
        self.image_x_offset_spinbox.valueChanged.connect(
            self._handle_image_x_offset_changed
        )
        image_transform_layout.addRow("X offset", self.image_x_offset_spinbox)

        self.image_y_offset_spinbox = self._build_image_transform_spinbox(
            value=DEFAULT_IMAGE_OFFSET,
            minimum=-100000.0,
            maximum=100000.0,
            decimals=1,
            single_step=1.0,
            suffix=" px",
        )
        self.image_y_offset_spinbox.valueChanged.connect(
            self._handle_image_y_offset_changed
        )
        image_transform_layout.addRow("Y offset", self.image_y_offset_spinbox)

        include_widget = QWidget()
        include_layout = QHBoxLayout(include_widget)
        include_layout.setContentsMargins(0, 0, 0, 0)
        include_layout.setSpacing(12)

        self.include_button_group = QButtonGroup(self)
        self.include_yes_radio = QRadioButton("Yes")
        self.include_no_radio = QRadioButton("No")
        self.include_button_group.addButton(self.include_yes_radio)
        self.include_button_group.addButton(self.include_no_radio)
        self.include_yes_radio.toggled.connect(self._handle_include_toggled)
        self.include_no_radio.toggled.connect(self._handle_include_toggled)
        include_layout.addWidget(self.include_yes_radio)
        include_layout.addWidget(self.include_no_radio)
        include_layout.addStretch(1)
        image_transform_layout.addRow("Include", include_widget)
        side_layout.addLayout(image_transform_layout)

        snap_label = QLabel("Snap")
        snap_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        side_layout.addWidget(snap_label)

        self.snap_middle_equal_angle_radio = QRadioButton(
            "Snap to middle equal angle only"
        )
        self.snap_middle_equal_angle_radio.setAutoExclusive(False)
        self.snap_middle_equal_angle_radio.setChecked(True)
        self.snap_middle_equal_angle_radio.toggled.connect(
            self._handle_snap_middle_equal_angle_toggled
        )
        side_layout.addWidget(self.snap_middle_equal_angle_radio)

        self.blueprint_name_label = QLabel("Image: none for this level")
        self.blueprint_name_label.setWordWrap(True)
        side_layout.addWidget(self.blueprint_name_label)

        levels_label = QLabel("Levels")
        levels_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        side_layout.addWidget(levels_label)

        self.levels_list = QListWidget()
        self.levels_list.currentRowChanged.connect(self._handle_level_selection_changed)
        side_layout.addWidget(self.levels_list, 1)

        rooms_label = QLabel("Rooms")
        rooms_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        side_layout.addWidget(rooms_label)

        self.rooms_list = QListWidget()
        side_layout.addWidget(self.rooms_list, 1)

        self.delete_room_shortcut = QShortcut(
            QKeySequence.StandardKey.Delete,
            self.rooms_list,
        )
        self.delete_room_shortcut.setContext(
            Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        self.delete_room_shortcut.activated.connect(self._delete_selected_room)

        room_layout = QFormLayout()
        room_layout.setContentsMargins(0, 0, 0, 0)
        room_layout.setSpacing(8)

        self.room_name_field = QLineEdit()
        self.room_name_field.setPlaceholderText("Room name")
        self.room_name_field.setMinimumHeight(34)
        room_layout.addRow("Room name", self.room_name_field)
        side_layout.addLayout(room_layout)

        self.designate_room_button = QPushButton("Designate room")
        self.designate_room_button.setMinimumHeight(44)
        self.designate_room_button.clicked.connect(self._handle_designate_room_clicked)
        side_layout.addWidget(self.designate_room_button)

        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)

        self.save_button = QPushButton("Save")
        self.save_button.setMinimumHeight(56)
        self.save_button.clicked.connect(self._handle_save_clicked)
        buttons_layout.addWidget(self.save_button)

        self.load_button = QPushButton("Load")
        self.load_button.setMinimumHeight(56)
        self.load_button.clicked.connect(self._handle_load_clicked)
        buttons_layout.addWidget(self.load_button)

        self.convert_button = QPushButton("Convert")
        self.convert_button.setMinimumHeight(56)
        self.convert_button.setStyleSheet("font-size: 18px; font-weight: 600;")
        self.convert_button.clicked.connect(self._handle_convert_clicked)
        buttons_layout.addWidget(self.convert_button, 1)
        side_layout.addLayout(buttons_layout)

        self.side_tabs.addTab(self._build_uvs_tab(), "UVs")

        splitter.addWidget(side_panel)
        splitter.setStretchFactor(0, 9)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([1160, 440])

        self.canvas.rooms_changed.connect(self._refresh_room_lists)
        self._refresh_levels_list()
        self._refresh_room_lists()
        self._sync_level_controls()
        self._sync_canvas_to_current_level()

    def _build_uvs_tab(self) -> QWidget:
        uvs_tab = QWidget()
        uvs_layout = QVBoxLayout(uvs_tab)
        uvs_layout.setContentsMargins(10, 12, 10, 10)
        uvs_layout.setSpacing(12)

        uv_map_label = QLabel("UV Map")
        uv_map_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        uvs_layout.addWidget(uv_map_label)

        self.uv_canvas = UvCanvas()
        self.uv_canvas.selected_wall_changed.connect(self._handle_uv_wall_selected)
        uvs_layout.addWidget(self.uv_canvas, 2)

        uv_controls_layout = QFormLayout()
        uv_controls_layout.setContentsMargins(0, 0, 0, 0)
        uv_controls_layout.setSpacing(8)

        self.unoccupied_uv_pixels_label = QLabel("Unoccupied pixels: 0")
        self.unoccupied_uv_pixels_label.setMinimumHeight(24)
        uv_controls_layout.addRow("", self.unoccupied_uv_pixels_label)

        self.uv_map_width_spinbox = PowerOfTwoSpinBox()
        self.uv_map_width_spinbox.setRange(64, 8192)
        self.uv_map_width_spinbox.valueChanged.connect(
            self._handle_uv_map_width_changed
        )
        uv_controls_layout.addRow("Map X", self.uv_map_width_spinbox)

        self.uv_map_height_spinbox = PowerOfTwoSpinBox()
        self.uv_map_height_spinbox.setRange(64, 8192)
        self.uv_map_height_spinbox.valueChanged.connect(
            self._handle_uv_map_height_changed
        )
        uv_controls_layout.addRow("Map Y", self.uv_map_height_spinbox)

        self.uv_wall_scale_spinbox = QDoubleSpinBox()
        self.uv_wall_scale_spinbox.setRange(0.01, 100.0)
        self.uv_wall_scale_spinbox.setDecimals(3)
        self.uv_wall_scale_spinbox.setSingleStep(0.05)
        self.uv_wall_scale_spinbox.valueChanged.connect(
            self._handle_uv_wall_scale_changed
        )
        uv_controls_layout.addRow("Wall scale", self.uv_wall_scale_spinbox)

        self.uv_wall_rotation_spinbox = RightAngleSpinBox()
        self.uv_wall_rotation_spinbox.setRange(0, 270)
        self.uv_wall_rotation_spinbox.setSuffix(" deg")
        self.uv_wall_rotation_spinbox.valueChanged.connect(
            self._handle_uv_wall_rotation_changed
        )
        uv_controls_layout.addRow("Wall rotation", self.uv_wall_rotation_spinbox)
        uvs_layout.addLayout(uv_controls_layout)

        uv_rooms_label = QLabel("Rooms")
        uv_rooms_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        uvs_layout.addWidget(uv_rooms_label)

        self.uv_rooms_list = QListWidget()
        self.uv_rooms_list.currentRowChanged.connect(
            self._handle_uv_room_selection_changed
        )
        uvs_layout.addWidget(self.uv_rooms_list, 1)
        return uvs_tab

    @staticmethod
    def _build_image_transform_spinbox(
        value: float,
        minimum: float,
        maximum: float,
        decimals: int,
        single_step: float,
        suffix: str = "",
    ) -> QDoubleSpinBox:
        spinbox = QDoubleSpinBox()
        spinbox.setRange(minimum, maximum)
        spinbox.setDecimals(decimals)
        spinbox.setSingleStep(single_step)
        spinbox.setValue(value)
        spinbox.setSuffix(suffix)
        spinbox.setMinimumHeight(34)
        return spinbox

    def load_blueprint(self, file_path: str) -> None:
        self._set_current_level_image(file_path)

    def _handle_convert_clicked(self) -> None:
        try:
            generated_model = convert_to_glb(self.levels)
        except ValueError as error:
            QMessageBox.warning(self, "Convert failed", str(error))
            return

        viewer_window = GlbViewerWindow(generated_model)
        viewer_window.closed.connect(self._handle_viewer_closed)
        viewer_window.show()
        viewer_window.raise_()
        viewer_window.activateWindow()
        self.viewer_windows.append(viewer_window)

    def _handle_viewer_closed(self, closed_viewer: GlbViewerWindow) -> None:
        self.viewer_windows = [
            viewer_window
            for viewer_window in self.viewer_windows
            if viewer_window is not closed_viewer
        ]

    def _handle_save_clicked(self) -> None:
        default_path = Path.cwd() / "housemaker_project.json"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "save project",
            str(default_path),
            "JSON Files (*.json)",
        )
        if not file_path:
            return

        try:
            save_project(
                path=file_path,
                current_level_index=self.current_level_index,
                levels=self.levels,
            )
        except ValueError as error:
            QMessageBox.critical(self, "Save failed", str(error))
            return

        QMessageBox.information(self, "Project saved", f"Saved project to:\n{file_path}")

    def _handle_load_clicked(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "load project",
            str(Path.cwd()),
            "JSON Files (*.json)",
        )
        if not file_path:
            return

        try:
            project_data = load_project(file_path)
            self._apply_loaded_project(project_data)
        except ValueError as error:
            QMessageBox.critical(self, "Project load failed", str(error))

    def _handle_load_image_clicked(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "load image",
            str(Path.home()),
            "Image Files (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if not file_path:
            return

        try:
            self._set_current_level_image(file_path)
        except ValueError as error:
            QMessageBox.critical(self, "Image load failed", str(error))

    def _refresh_levels_list(self) -> None:
        self._is_syncing_level_controls = True
        self.levels_list.clear()
        for level in self.levels:
            self.levels_list.addItem(self._build_level_item(level))
        self.levels_list.setCurrentRow(self.current_level_index)
        self._is_syncing_level_controls = False

    def _build_level_item(self, level: LevelData) -> QListWidgetItem:
        return QListWidgetItem(level.display_name)

    def _refresh_rooms_list(self) -> None:
        selected_room_index = self._get_selected_room_index()
        self.rooms_list.clear()
        for room_index, room in enumerate(self.current_level.rooms):
            room_item = self._build_room_item(room_index, room.name)
            self.rooms_list.addItem(room_item)

        if (
            selected_room_index is not None
            and selected_room_index < self.rooms_list.count()
        ):
            self.rooms_list.setCurrentRow(selected_room_index)

    def _refresh_room_lists(self) -> None:
        self._refresh_rooms_list()
        self._refresh_uv_rooms_list()
        self._sync_uv_controls()

    def _refresh_uv_rooms_list(self) -> None:
        selected_room_index = self._get_selected_uv_room_index()
        self.uv_rooms_list.clear()
        for room_index, room in enumerate(self.current_level.rooms):
            room_item = self._build_room_item(room_index, room.name)
            self.uv_rooms_list.addItem(room_item)

        if self.uv_rooms_list.count() == 0:
            self._sync_uv_controls()
            return

        if selected_room_index is None:
            self.uv_rooms_list.setCurrentRow(0)
            return

        if selected_room_index < self.uv_rooms_list.count():
            self.uv_rooms_list.setCurrentRow(selected_room_index)
            return

        self.uv_rooms_list.setCurrentRow(self.uv_rooms_list.count() - 1)

    def _build_room_item(self, room_index: int, room_name: str) -> QListWidgetItem:
        room_item = QListWidgetItem(room_name)
        room_item.setData(Qt.ItemDataRole.UserRole, room_index)
        return room_item

    def _get_selected_room_index(self) -> int | None:
        selected_item = self.rooms_list.currentItem()
        if selected_item is None:
            return None

        room_index = selected_item.data(Qt.ItemDataRole.UserRole)
        if room_index is None:
            return None

        return int(room_index)

    def _get_selected_uv_room_index(self) -> int | None:
        selected_item = self.uv_rooms_list.currentItem()
        if selected_item is None:
            return None

        room_index = selected_item.data(Qt.ItemDataRole.UserRole)
        if room_index is None:
            return None

        return int(room_index)

    def _get_selected_uv_room(self) -> RoomData | None:
        room_index = self._get_selected_uv_room_index()
        if room_index is None or room_index >= len(self.current_level.rooms):
            return None

        return self.current_level.rooms[room_index]

    def _sync_level_controls(self) -> None:
        self._is_syncing_level_controls = True
        self.height_level_spinbox.setValue(self.current_level.height_meters)
        self.image_scale_spinbox.setValue(self.current_level.image_scale)
        self.image_x_offset_spinbox.setValue(self.current_level.image_offset_x)
        self.image_y_offset_spinbox.setValue(self.current_level.image_offset_y)
        self.include_yes_radio.setChecked(self.current_level.include_in_export)
        self.include_no_radio.setChecked(not self.current_level.include_in_export)
        if self.levels_list.currentRow() != self.current_level_index:
            self.levels_list.setCurrentRow(self.current_level_index)
        self._update_blueprint_name_label()
        self._is_syncing_level_controls = False

    def _sync_uv_controls(self) -> None:
        self._is_syncing_uv_controls = True
        selected_room = self._get_selected_uv_room()
        has_room = selected_room is not None
        self.uv_map_width_spinbox.setEnabled(has_room)
        self.uv_map_height_spinbox.setEnabled(has_room)
        self.unoccupied_uv_pixels_label.setEnabled(has_room)

        if selected_room is None:
            self.uv_canvas.set_room_context(None, None, self.current_level.height_meters)
            self.uv_map_width_spinbox.setValue(DEFAULT_UV_MAP_WIDTH)
            self.uv_map_height_spinbox.setValue(DEFAULT_UV_MAP_HEIGHT)
            self.uv_wall_scale_spinbox.setValue(DEFAULT_WALL_UV_SCALE)
            self.uv_wall_scale_spinbox.setEnabled(False)
            self.uv_wall_rotation_spinbox.setValue(DEFAULT_WALL_UV_ROTATION_DEGREES)
            self.uv_wall_rotation_spinbox.setEnabled(False)
            self.unoccupied_uv_pixels_label.setText("Unoccupied pixels: 0")
            self._is_syncing_uv_controls = False
            return

        self.uv_canvas.set_room_context(
            selected_room,
            self.current_level.vertex_data,
            self.current_level.height_meters,
        )
        self.uv_map_width_spinbox.setValue(selected_room.uv_map_width)
        self.uv_map_height_spinbox.setValue(selected_room.uv_map_height)

        selected_wall_key = self.uv_canvas.get_selected_wall_key()
        has_selected_wall = selected_wall_key is not None
        self.uv_wall_scale_spinbox.setEnabled(has_selected_wall)
        self.uv_wall_rotation_spinbox.setEnabled(has_selected_wall)
        if selected_wall_key is None:
            self.uv_wall_scale_spinbox.setValue(DEFAULT_WALL_UV_SCALE)
            self.uv_wall_rotation_spinbox.setValue(DEFAULT_WALL_UV_ROTATION_DEGREES)
        else:
            self.uv_wall_scale_spinbox.setValue(
                selected_room.wall_uv_scales.get(
                    selected_wall_key,
                    DEFAULT_WALL_UV_SCALE,
                )
            )
            self.uv_wall_rotation_spinbox.setValue(
                selected_room.wall_uv_rotations.get(
                    selected_wall_key,
                    DEFAULT_WALL_UV_ROTATION_DEGREES,
                )
            )

        self._update_unoccupied_uv_pixels_label()
        self._is_syncing_uv_controls = False

    def _update_unoccupied_uv_pixels_label(self) -> None:
        selected_room = self._get_selected_uv_room()
        if selected_room is None:
            self.unoccupied_uv_pixels_label.setText("Unoccupied pixels: 0")
            return

        unoccupied_pixels = calculate_unoccupied_uv_pixels(
            room=selected_room,
            vertex_data=self.current_level.vertex_data,
            wall_height_meters=self.current_level.height_meters,
        )
        self.unoccupied_uv_pixels_label.setText(
            f"Unoccupied pixels: {unoccupied_pixels:,}"
        )

    def _handle_level_selection_changed(self, level_index: int) -> None:
        if self._is_syncing_level_controls or level_index < 0 or level_index >= len(self.levels):
            return

        self.current_level_index = level_index
        self._sync_level_controls()
        self._sync_canvas_to_current_level()
        self._refresh_room_lists()

    def _handle_height_level_changed(self, value: float) -> None:
        if self._is_syncing_level_controls:
            return

        self.current_level.height_meters = value
        self._sync_uv_controls()

    def _handle_image_scale_changed(self, value: float) -> None:
        if self._is_syncing_level_controls:
            return

        self.current_level.image_scale = value
        self._sync_canvas_image_transform()

    def _handle_image_x_offset_changed(self, value: float) -> None:
        if self._is_syncing_level_controls:
            return

        self.current_level.image_offset_x = value
        self._sync_canvas_image_transform()

    def _handle_image_y_offset_changed(self, value: float) -> None:
        if self._is_syncing_level_controls:
            return

        self.current_level.image_offset_y = value
        self._sync_canvas_image_transform()

    def _handle_include_toggled(self, checked: bool) -> None:
        if self._is_syncing_level_controls or not checked:
            return

        self.current_level.include_in_export = self.include_yes_radio.isChecked()

    def _handle_snap_middle_equal_angle_toggled(self, checked: bool) -> None:
        self.canvas.set_snap_middle_equal_angle_only(checked)

    def _handle_uv_room_selection_changed(self, room_index: int) -> None:
        self._sync_uv_controls()

    def _handle_uv_map_width_changed(self, value: int) -> None:
        if self._is_syncing_uv_controls:
            return

        selected_room = self._get_selected_uv_room()
        if selected_room is None:
            return

        selected_room.uv_map_width = int(value)
        self.uv_canvas.update()
        self._update_unoccupied_uv_pixels_label()

    def _handle_uv_map_height_changed(self, value: int) -> None:
        if self._is_syncing_uv_controls:
            return

        selected_room = self._get_selected_uv_room()
        if selected_room is None:
            return

        selected_room.uv_map_height = int(value)
        self.uv_canvas.update()
        self._update_unoccupied_uv_pixels_label()

    def _handle_uv_wall_selected(self, wall_key: str) -> None:
        self._sync_uv_controls()

    def _handle_uv_wall_scale_changed(self, value: float) -> None:
        if self._is_syncing_uv_controls:
            return

        selected_room = self._get_selected_uv_room()
        selected_wall_key = self.uv_canvas.get_selected_wall_key()
        if selected_room is None or selected_wall_key is None:
            return

        selected_room.wall_uv_scales[selected_wall_key] = float(value)
        self.uv_canvas.update()
        self._update_unoccupied_uv_pixels_label()

    def _handle_uv_wall_rotation_changed(self, value: int) -> None:
        if self._is_syncing_uv_controls:
            return

        selected_room = self._get_selected_uv_room()
        selected_wall_key = self.uv_canvas.get_selected_wall_key()
        if selected_room is None or selected_wall_key is None:
            return

        selected_room.wall_uv_rotations[selected_wall_key] = _nearest_right_angle_value(
            value
        )
        self.uv_canvas.update()
        self._update_unoccupied_uv_pixels_label()

    def _handle_designate_room_clicked(self) -> None:
        room_name = self.room_name_field.text().strip()
        if not room_name:
            QMessageBox.warning(self, "Room name required", "Enter a room name first.")
            return

        selected_vertex_ids = self.canvas.get_selected_vertex_ids()
        if len(selected_vertex_ids) < 3:
            QMessageBox.warning(
                self,
                "Select room vertices",
                "Shift-click at least three vertices before designating a room.",
            )
            return

        self.canvas.start_room_designation(
            room_name,
            selected_vertex_ids,
            self.current_level.height_meters,
        )
        QMessageBox.information(
            self,
            "Set room center",
            "Click the vertex that marks the center of the room.",
        )

    def _delete_selected_room(self) -> None:
        room_index = self._get_selected_room_index()
        if room_index is None:
            return

        self.canvas.delete_room_at_index(room_index)

    def _apply_loaded_project(self, project_data: ProjectData) -> None:
        self._apply_project_state(
            levels=project_data.levels,
            current_level_index=project_data.current_level_index,
        )

    def _apply_project_state(
        self,
        levels: list[LevelData],
        current_level_index: int,
    ) -> None:
        self.levels = levels
        self.current_level_index = min(max(current_level_index, 0), len(self.levels) - 1)

        self._refresh_levels_list()
        self._sync_level_controls()
        self._sync_canvas_to_current_level()
        self._refresh_room_lists()

    def _set_current_level_image(self, file_path: str) -> None:
        normalized_path = str(Path(file_path).resolve())
        self.canvas.load_blueprint(
            file_path=normalized_path,
            vertex_data=self.current_level.vertex_data,
            rooms=self.current_level.rooms,
            image_scale=self.current_level.image_scale,
            image_offset_x=self.current_level.image_offset_x,
            image_offset_y=self.current_level.image_offset_y,
        )
        self.current_level.image_path = normalized_path
        self.current_level.image_size_pixels = self.canvas.get_image_size_pixels()
        self._update_blueprint_name_label()

    def _sync_canvas_to_current_level(self) -> None:
        self.canvas.set_level_data(
            vertex_data=self.current_level.vertex_data,
            rooms=self.current_level.rooms,
            image_path=self.current_level.image_path,
            image_scale=self.current_level.image_scale,
            image_offset_x=self.current_level.image_offset_x,
            image_offset_y=self.current_level.image_offset_y,
        )
        if self.canvas.blueprint_image is not None:
            self.current_level.image_size_pixels = self.canvas.get_image_size_pixels()
        self._update_blueprint_name_label()

    def _sync_canvas_image_transform(self) -> None:
        self.canvas.set_image_transform(
            image_scale=self.current_level.image_scale,
            image_offset_x=self.current_level.image_offset_x,
            image_offset_y=self.current_level.image_offset_y,
        )

    def _update_blueprint_name_label(self) -> None:
        image_path = self.current_level.image_path
        if image_path is None:
            self.blueprint_name_label.setText("Image: none for this level")
            return

        if self.canvas.blueprint_image is None:
            self.blueprint_name_label.setText(f"Image missing: {image_path}")
            return

        self.blueprint_name_label.setText(f"Image: {Path(image_path).name}")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._build_ui()

    def _build_ui(self) -> None:
        self.setWindowTitle("HouseMaker")
        self.resize(1600, 900)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.home_page = HomePage()
        self.blueprint_workspace = BlueprintWorkspace()

        self.stack.addWidget(self.home_page)
        self.stack.addWidget(self.blueprint_workspace)
        self.stack.setCurrentWidget(self.home_page)

        self.home_page.blueprint_button.clicked.connect(self._open_blueprint_mode)

    def _open_blueprint_mode(self) -> None:
        self.stack.setCurrentWidget(self.blueprint_workspace)


# ### Numeric helpers ###
def _nearest_power_of_two_value(value: int) -> int:
    clamped_value = min(max(int(value), 64), 8192)
    lower_power = 64
    while lower_power * 2 <= clamped_value:
        lower_power *= 2

    upper_power = min(lower_power * 2, 8192)
    if clamped_value - lower_power <= upper_power - clamped_value:
        return lower_power

    return upper_power


def _step_power_of_two_value(value: int, steps: int) -> int:
    power_value = _nearest_power_of_two_value(value)
    for _ in range(abs(steps)):
        if steps > 0:
            power_value = min(power_value * 2, 8192)
        else:
            power_value = max(power_value // 2, 64)

    return power_value


def _nearest_right_angle_value(value: int) -> int:
    clamped_value = min(max(int(value), 0), 270)
    return (round(clamped_value / 90) * 90) % 360


def _step_right_angle_value(value: int, steps: int) -> int:
    right_angle_value = _nearest_right_angle_value(value)
    for _ in range(abs(steps)):
        if steps > 0:
            right_angle_value = min(right_angle_value + 90, 270)
        else:
            right_angle_value = max(right_angle_value - 90, 0)

    return right_angle_value


# ### Entrypoint ###
def main() -> int:
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication(sys.argv)
    app.setApplicationName("HouseMaker")
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()
    return app.exec()


# ### Direct execution ###
if __name__ == "__main__":
    raise SystemExit(main())
