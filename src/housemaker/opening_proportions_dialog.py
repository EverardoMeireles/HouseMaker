# ### Imports ###
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from housemaker.opening_proportions_view import OpeningProportionsView
from housemaker.video_source import VideoFrameSource

# ### Constants ###
SOURCE_FILE_FILTER = (
    "Videos and images (*.mp4 *.mov *.avi *.mkv *.webm *.m4v "
    "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff);;All files (*)"
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
FRAME_ARROW_STEP = 2
OPENING_INSTRUCTIONS = (
    "Drag the four corners onto the opening. Scroll to zoom; middle-drag to pan."
)
OpeningDraft = tuple[tuple, tuple | None, bool]


# ### Opening proportions dialog ###
class OpeningProportionsDialog(QDialog):
    """A single opening, source video and manually editable image outline."""

    apply_requested = Signal(object)
    outline_changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Opening proportions")
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowMinimizeButtonHint
        )
        self.resize(1050, 760)
        self._source: VideoFrameSource | None = None
        self._source_path = ""
        self._frame_index = 0
        self._is_doorway = False
        self._width_meters = 1.0
        self._height_meters = 2.0
        self._arch_height_meters = 0.0
        self._target_key: object | None = None
        self._target_drafts: dict[object, dict[str, dict[int, OpeningDraft]]] = {}
        self._drafts: dict[int, OpeningDraft] = {}
        self._outline_edited = False
        self._loading_frame = False
        self._result_valid = False
        self._shortcuts: list[QShortcut] = []
        self._build_ui()
        self._configure_shortcuts()

    # ### Controls ###
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.target_label = QLabel()
        self.target_label.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.target_label)
        toolbar = QHBoxLayout()
        self.load_video_button = QPushButton("Load video")
        self.load_video_button.clicked.connect(self._browse_source)
        toolbar.addWidget(self.load_video_button)
        self.frame_label = QLabel("No video loaded")
        toolbar.addWidget(self.frame_label)
        toolbar.addStretch(1)
        layout.addLayout(toolbar)
        self.video_view = OpeningProportionsView(self)
        self.video_view.outline_changed.connect(self._on_manual_outline_changed)
        layout.addWidget(self.video_view, 1)
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setEnabled(False)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.setSingleStep(FRAME_ARROW_STEP)
        self.seek_slider.valueChanged.connect(self._seek_frame)
        layout.addWidget(self.seek_slider)
        self.instructions_label = QLabel(OPENING_INSTRUCTIONS)
        self.instructions_label.setWordWrap(True)
        layout.addWidget(self.instructions_label)
        self.result_label = QLabel("Load a video or image, then outline the opening.")
        self.result_label.setWordWrap(True)
        self.result_label.setTextFormat(Qt.TextFormat.PlainText)
        self.result_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
        )
        self.result_label.setFixedHeight(
            self.result_label.fontMetrics().lineSpacing() * 3 + 6
        )
        layout.addWidget(self.result_label)
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.Close
        )
        self.apply_button = self.button_box.button(
            QDialogButtonBox.StandardButton.Apply
        )
        self.apply_button.setText("Apply proportions")
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self._request_apply)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def _configure_shortcuts(self) -> None:
        for key, offset in (
            (Qt.Key.Key_Left, -FRAME_ARROW_STEP),
            (Qt.Key.Key_Right, FRAME_ARROW_STEP),
        ):
            shortcut = QShortcut(key, self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(lambda delta=offset: self.step_frame(delta))
            self._shortcuts.append(shortcut)

    # ### Public target and result interface ###
    def set_target(
        self,
        label: str,
        is_doorway: bool,
        width_meters: float,
        current_height_meters: float,
        arch_height_meters: float = 0.0,
        *,
        target_key: object | None = None,
    ) -> None:
        self._save_draft()
        self._target_key = target_key
        self._is_doorway = bool(is_doorway)
        self._width_meters = max(1e-6, float(width_meters))
        self._height_meters = max(1e-6, float(current_height_meters))
        self._arch_height_meters = max(0.0, float(arch_height_meters))
        self.target_label.setText(str(label))
        self.video_view.set_doorway(is_doorway)
        self._select_target_drafts()
        self._result_valid = False
        self.apply_button.setEnabled(False)
        if self.video_view.has_frame():
            self._restore_draft()
            self._on_outline_changed()

    def set_result_text(self, text: str, valid: bool = True) -> None:
        self.result_label.setText(str(text))
        self.result_label.setToolTip(str(text))
        self._result_valid = bool(valid)
        self.apply_button.setEnabled(self._result_valid and self.video_view.has_frame())

    def show_error(self, text: str) -> None:
        self.set_result_text(text, valid=False)

    def current_measurement(self) -> dict[str, object]:
        return {
            "frame_size": self.video_view.frame_size,
            "outline": self.video_view.outline,
            "arch_apex": self.video_view.arch_apex,
            "is_doorway": self._is_doorway,
            "source_path": self._source_path,
            "frame_index": self._frame_index,
            "outline_edited": self._outline_edited,
        }

    # ### Video and image loading ###
    def load_source(self, path: str | Path, frame_index: int = 0) -> None:
        """Open a separate reader; never seek or close the Generation tab's reader."""
        normalized_path = str(Path(path).resolve())
        new_source: VideoFrameSource | None = None
        if Path(path).suffix.lower() in IMAGE_EXTENSIONS:
            encoded = np.fromfile(normalized_path, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
            if image is None:
                raise ValueError("The selected image could not be opened.")
            selected_frame = 0
            frame_count = 1
        else:
            new_source = VideoFrameSource(normalized_path)
            try:
                frame_count = new_source.metadata.frame_count
                selected_frame = min(max(0, int(frame_index)), frame_count - 1)
                image = new_source.get_frame(selected_frame)
            except Exception:
                new_source.close()
                raise
        self._save_draft()
        self.dispose()
        self._source = new_source
        self._source_path = normalized_path
        self._frame_index = selected_frame
        self._select_target_drafts()
        self.seek_slider.blockSignals(True)
        self.seek_slider.setRange(0, frame_count - 1)
        self.seek_slider.setValue(selected_frame)
        self.seek_slider.setEnabled(new_source is not None and frame_count > 1)
        self.seek_slider.blockSignals(False)
        self.video_view.reset_zoom()
        self._show_frame(image)

    def _browse_source(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Load video or image",
            self._source_path,
            SOURCE_FILE_FILTER,
        )
        if not path:
            return
        try:
            self.load_source(path)
        except (OSError, ValueError, IndexError) as error:
            QMessageBox.warning(self, "Reference unavailable", str(error))

    def _seek_frame(self, frame_index: int) -> None:
        if self._source is None:
            return
        try:
            image = self._source.get_frame(frame_index)
        except (OSError, ValueError, IndexError) as error:
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(self._frame_index)
            self.seek_slider.blockSignals(False)
            self.show_error(str(error))
            return
        self._save_draft()
        self._frame_index = frame_index
        self._show_frame(image)

    def step_frame(self, offset: int) -> None:
        if self._source is not None:
            self.seek_slider.setValue(self._frame_index + int(offset))

    def _show_frame(self, image: np.ndarray) -> None:
        self._loading_frame = True
        try:
            self.video_view.set_frame(image)
            self._restore_draft()
            count = self._source.metadata.frame_count if self._source else 1
            self.frame_label.setText(f"Frame {self._frame_index + 1} / {count}")
        finally:
            self._loading_frame = False
        self._on_outline_changed()

    def _set_default_outline(self) -> None:
        self._outline_edited = False
        width, height = self.video_view.frame_size
        if width <= 0 or height <= 0:
            return
        pixel_aspect = self._width_meters / self._height_meters
        box_height = min(0.7, 0.7 * width / max(height * pixel_aspect, 1e-6))
        box_width = box_height * height * pixel_aspect / width
        arch_fraction = min(0.8, self._arch_height_meters / self._height_meters)
        top = 0.5 - box_height * 0.5
        spring = top + box_height * arch_fraction
        self.video_view.set_outline(
            (
                (0.5 - box_width * 0.5, spring),
                (0.5 + box_width * 0.5, spring),
                (0.5 + box_width * 0.5, top + box_height),
                (0.5 - box_width * 0.5, top + box_height),
            ),
            (0.5, top) if self._is_doorway else None,
        )

    # ### Annotation and application signals ###
    def _select_target_drafts(self) -> None:
        if self._target_key is None:
            self._drafts = {}
        else:
            sources = self._target_drafts.setdefault(self._target_key, {})
            self._drafts = sources.setdefault(self._source_path, {})

    def _restore_draft(self) -> None:
        draft = self._drafts.get(self._frame_index)
        if draft is None:
            self._set_default_outline()
        else:
            outline, apex, self._outline_edited = draft
            self.video_view.set_outline(outline, apex)

    def _save_draft(self) -> None:
        if self.video_view.has_frame():
            self._drafts[self._frame_index] = (
                self.video_view.outline,
                self.video_view.arch_apex,
                self._outline_edited,
            )

    def _on_manual_outline_changed(self) -> None:
        self._outline_edited = True
        self._on_outline_changed()

    def _on_outline_changed(self) -> None:
        if self._loading_frame or not self.video_view.has_frame():
            return
        self._save_draft()
        self._result_valid = False
        self.apply_button.setEnabled(False)
        self.outline_changed.emit(self.current_measurement())

    def _request_apply(self) -> None:
        if self._result_valid and self.video_view.has_frame():
            self.apply_requested.emit(self.current_measurement())

    # ### Reader lifetime ###
    def dispose(self) -> None:
        if self._source is not None:
            self._source.close()
            self._source = None

    def done(self, result: int) -> None:
        self.dispose()
        super().done(result)

    def closeEvent(self, event) -> None:
        self.dispose()
        super().closeEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            self.step_frame(
                -FRAME_ARROW_STEP
                if event.key() == Qt.Key.Key_Left
                else FRAME_ARROW_STEP
            )
            event.accept()
            return
        super().keyPressEvent(event)
