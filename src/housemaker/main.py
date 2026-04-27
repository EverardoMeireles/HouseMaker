# ### Imports ###
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from housemaker.blueprint_canvas import BlueprintCanvas
from housemaker.glb import DEFAULT_WALL_HEIGHT_METERS, convert_to_glb
from housemaker.models import (
    DEFAULT_IMAGE_OFFSET,
    DEFAULT_IMAGE_SCALE,
    GROUND_LEVEL_INDEX,
    LevelData,
    create_default_levels,
)
from housemaker.project_io import ProjectData, load_project, save_project
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


class BlueprintWorkspace(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer_windows: list[GlbViewerWindow] = []
        self.levels: list[LevelData] = create_default_levels()
        self.current_level_index = GROUND_LEVEL_INDEX
        self._is_syncing_level_controls = False
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

        self.blueprint_name_label = QLabel("Image: none for this level")
        self.blueprint_name_label.setWordWrap(True)
        side_layout.addWidget(self.blueprint_name_label)

        levels_label = QLabel("Levels")
        levels_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        side_layout.addWidget(levels_label)

        self.levels_list = QListWidget()
        self.levels_list.currentRowChanged.connect(self._handle_level_selection_changed)
        side_layout.addWidget(self.levels_list, 1)

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

        splitter.addWidget(side_panel)
        splitter.setStretchFactor(0, 9)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([1380, 220])

        self._refresh_levels_list()
        self._sync_level_controls()
        self._sync_canvas_to_current_level()

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

    def _handle_level_selection_changed(self, level_index: int) -> None:
        if self._is_syncing_level_controls or level_index < 0 or level_index >= len(self.levels):
            return

        self.current_level_index = level_index
        self._sync_level_controls()
        self._sync_canvas_to_current_level()

    def _handle_height_level_changed(self, value: float) -> None:
        if self._is_syncing_level_controls:
            return

        self.current_level.height_meters = value

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

    def _set_current_level_image(self, file_path: str) -> None:
        normalized_path = str(Path(file_path).resolve())
        self.canvas.load_blueprint(
            file_path=normalized_path,
            vertex_data=self.current_level.vertex_data,
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
