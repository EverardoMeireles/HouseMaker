# ### Imports ###
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from housemaker.blueprint_canvas import BlueprintCanvas
from housemaker.glb import DEFAULT_WALL_HEIGHT_METERS, convert_to_glb
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
        self._build_ui()

    @property
    def vertex_data(self):
        return self.canvas.vertex_data

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
        side_layout.addWidget(self.height_level_spinbox)

        levels_label = QLabel("Levels")
        levels_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        side_layout.addWidget(levels_label)

        self.blueprint_name_label = QLabel("No blueprint selected")
        self.blueprint_name_label.setWordWrap(True)
        side_layout.addWidget(self.blueprint_name_label)

        self.levels_list = QListWidget()
        self.levels_list.addItem("Ground Floor")
        self.levels_list.addItem("More levels later")
        side_layout.addWidget(self.levels_list, 1)

        self.convert_button = QPushButton("Convert")
        self.convert_button.setMinimumHeight(56)
        self.convert_button.setStyleSheet("font-size: 18px; font-weight: 600;")
        self.convert_button.clicked.connect(self._handle_convert_clicked)
        side_layout.addWidget(self.convert_button)

        splitter.addWidget(side_panel)
        splitter.setStretchFactor(0, 9)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([1350, 150])

    def load_blueprint(self, file_path: str) -> None:
        self.canvas.load_blueprint(file_path)
        self.blueprint_name_label.setText(f"Loaded: {Path(file_path).name}")

    def _handle_convert_clicked(self) -> None:
        try:
            generated_model = convert_to_glb(
                self.vertex_data,
                wall_height_meters=self.height_level_spinbox.value(),
            )
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
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "select blueprint",
            str(Path.home()),
            "Image Files (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if not file_path:
            return

        try:
            self.blueprint_workspace.load_blueprint(file_path)
        except ValueError as error:
            QMessageBox.critical(self, "Blueprint load failed", str(error))
            return

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
