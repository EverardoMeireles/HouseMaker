# ### Imports ###
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, QTimer, Qt
from PySide6.QtGui import (
    QColor,
    QImage,
    QKeySequence,
    QPen,
    QPixmap,
    QShortcut,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

# ### Constants ###
DEFAULT_MANUAL_PLAYBACK_INTERVAL_MS = 33
MANUAL_WINDOW_SCREEN_RATIO = 0.95
VIDEO_FILE_FILTER = (
    "Video Files (*.mp4 *.mov *.avi *.mkv *.webm *.m4v);;All Files (*)"
)
VIEWPORT_FIT_PADDING = 12.0
STITCHED_IMAGE_NORMAL_OPACITY = 1.0
STITCHED_IMAGE_DRAG_OPACITY = 0.35
KEYBOARD_NUDGE_OPACITY_MS = 180
ALIGNMENT_GUIDE_TOLERANCE_PIXELS = 0.01
ALIGNMENT_GUIDE_Z_VALUE = 3.0

# ### Graphics items ###
class HorizontalWheelGraphicsView(QGraphicsView):
    def wheelEvent(self, event: QWheelEvent) -> None:  # type: ignore[override]
        wheel_delta = event.angleDelta().x() or event.angleDelta().y()
        if wheel_delta == 0:
            wheel_delta = event.pixelDelta().x() or event.pixelDelta().y()
        if wheel_delta == 0:
            super().wheelEvent(event)
            return

        horizontal_scrollbar = self.horizontalScrollBar()
        horizontal_scrollbar.setValue(horizontal_scrollbar.value() - wheel_delta)
        event.accept()


class MovableFramePixmapItem(QGraphicsPixmapItem):
    def __init__(
        self,
        position_changed_callback: Callable[[], None],
        drag_started_callback: Callable[[], None],
        drag_finished_callback: Callable[[], None],
        snap_requested_callback: Callable[[], None],
    ) -> None:
        super().__init__()
        self.position_changed_callback = position_changed_callback
        self.drag_started_callback = drag_started_callback
        self.drag_finished_callback = drag_finished_callback
        self.snap_requested_callback = snap_requested_callback

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.RightButton:
            self.snap_requested_callback()
            event.accept()
            return

        self.drag_started_callback()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        super().mouseReleaseEvent(event)
        self.drag_finished_callback()

    def itemChange(self, change, value):  # type: ignore[override]
        result = super().itemChange(change, value)
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.position_changed_callback()

        return result


# ### Dialogs ###
class ManualVideoStitchDialog(QDialog):
    def __init__(self, video_path: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.video_path = str(Path(video_path).resolve())
        self.frames, self.playback_interval_ms = _load_video_frames(self.video_path)
        if not self.frames:
            raise ValueError(f"Unable to load video frames: {self.video_path}")

        self.current_frame_index = 0
        self.stitched_image: np.ndarray | None = None
        self.stitched_frame_count = 0
        self.view_scale = 1.0
        self._is_syncing_frame_progress = False
        self._last_fit_frame_height: float | None = None
        self._last_fit_viewport_height: int | None = None
        self._build_ui()
        self._sync_scene_items()

    def get_stitched_image(self) -> np.ndarray | None:
        if self.stitched_image is None:
            return None

        return self.stitched_image.copy()

    def _build_ui(self) -> None:
        self.setWindowTitle("Manual video stitching")
        self._resize_to_screen()

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        self.scene = QGraphicsScene(self)
        self.view = HorizontalWheelGraphicsView(self.scene)
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.view.customContextMenuRequested.connect(
            lambda _position: self._snap_current_frame_to_nearest_alignment_guide()
        )
        root_layout.addWidget(self.view, 19)

        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)

        self.frame_progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_progress_slider.setRange(0, max(0, len(self.frames) - 1))
        self.frame_progress_slider.setSingleStep(1)
        self.frame_progress_slider.setPageStep(max(1, len(self.frames) // 20))
        self.frame_progress_slider.setValue(self.current_frame_index)
        self.frame_progress_slider.valueChanged.connect(
            self._handle_frame_progress_changed
        )
        self.frame_progress_slider.sliderPressed.connect(
            self._handle_frame_progress_pressed
        )
        self.frame_progress_slider.sliderReleased.connect(
            self._handle_frame_progress_released
        )
        controls_layout.addWidget(self.frame_progress_slider)

        playback_controls_layout = QHBoxLayout()
        playback_controls_layout.setContentsMargins(0, 0, 0, 0)
        playback_controls_layout.setSpacing(10)

        self.back_button = QPushButton("Go back a frame")
        self.back_button.setMinimumHeight(40)
        self.back_button.clicked.connect(self._go_back_frame)
        playback_controls_layout.addWidget(self.back_button)

        self.play_button = QPushButton("Play")
        self.play_button.setMinimumHeight(40)
        self.play_button.clicked.connect(self._toggle_playback)
        playback_controls_layout.addWidget(self.play_button)

        self.forward_button = QPushButton("Go forward a frame")
        self.forward_button.setMinimumHeight(40)
        self.forward_button.clicked.connect(self._go_forward_frame)
        playback_controls_layout.addWidget(self.forward_button)

        self.stitch_button = QPushButton("Stitch")
        self.stitch_button.setMinimumHeight(40)
        self.stitch_button.clicked.connect(self._stitch_current_frame)
        playback_controls_layout.addWidget(self.stitch_button)

        self.make_image_button = QPushButton("Make image")
        self.make_image_button.setMinimumHeight(40)
        self.make_image_button.setEnabled(False)
        self.make_image_button.clicked.connect(self._handle_make_image_clicked)
        playback_controls_layout.addWidget(self.make_image_button)

        controls_layout.addLayout(playback_controls_layout)

        root_layout.addWidget(controls_widget, 1)

        self.playback_timer = QTimer(self)
        self.playback_timer.setInterval(self.playback_interval_ms)
        self.playback_timer.timeout.connect(self._advance_playback_frame)

        self.keyboard_nudge_opacity_timer = QTimer(self)
        self.keyboard_nudge_opacity_timer.setSingleShot(True)
        self.keyboard_nudge_opacity_timer.setInterval(KEYBOARD_NUDGE_OPACITY_MS)
        self.keyboard_nudge_opacity_timer.timeout.connect(
            self._handle_current_frame_drag_finished
        )

        self.final_image_item = QGraphicsPixmapItem()
        self.final_image_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.final_image_item.setZValue(2.0)
        self.scene.addItem(self.final_image_item)

        self.current_frame_item = MovableFramePixmapItem(
            self._handle_current_frame_moved,
            self._handle_current_frame_drag_started,
            self._handle_current_frame_drag_finished,
            self._snap_current_frame_to_nearest_alignment_guide,
        )
        self.current_frame_item.setFlags(
            QGraphicsPixmapItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsPixmapItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsPixmapItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.current_frame_item.setZValue(1.0)
        self.current_frame_item.setCursor(Qt.CursorShape.OpenHandCursor)
        self.scene.addItem(self.current_frame_item)
        self._build_alignment_guides()
        self._build_keyboard_shortcuts()

    def _build_alignment_guides(self) -> None:
        guide_pen = _build_alignment_guide_pen()
        self.top_alignment_guide_item = QGraphicsLineItem()
        self.top_alignment_guide_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.top_alignment_guide_item.setPen(guide_pen)
        self.top_alignment_guide_item.setZValue(ALIGNMENT_GUIDE_Z_VALUE)
        self.top_alignment_guide_item.hide()
        self.scene.addItem(self.top_alignment_guide_item)

        self.bottom_alignment_guide_item = QGraphicsLineItem()
        self.bottom_alignment_guide_item.setAcceptedMouseButtons(
            Qt.MouseButton.NoButton
        )
        self.bottom_alignment_guide_item.setPen(guide_pen)
        self.bottom_alignment_guide_item.setZValue(ALIGNMENT_GUIDE_Z_VALUE)
        self.bottom_alignment_guide_item.hide()
        self.scene.addItem(self.bottom_alignment_guide_item)

    def _build_keyboard_shortcuts(self) -> None:
        self.keyboard_nudge_shortcuts: list[QShortcut] = []
        shortcut_specs = (
            ("Left", QPointF(-1.0, 0.0)),
            ("Right", QPointF(1.0, 0.0)),
            ("Up", QPointF(0.0, -1.0)),
            ("Down", QPointF(0.0, 1.0)),
        )
        for key_sequence, offset in shortcut_specs:
            shortcut = QShortcut(QKeySequence(key_sequence), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(
                lambda offset=offset: self._nudge_current_frame(offset)
            )
            self.keyboard_nudge_shortcuts.append(shortcut)

    def _resize_to_screen(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1200, 800)
            return

        available_geometry = screen.availableGeometry()
        self.resize(
            int(available_geometry.width() * MANUAL_WINDOW_SCREEN_RATIO),
            int(available_geometry.height() * MANUAL_WINDOW_SCREEN_RATIO),
        )

    def _sync_scene_items(self, preserve_view: bool = False) -> None:
        if self.stitched_image is None:
            self.final_image_item.setPixmap(QPixmap())
        else:
            self.final_image_item.setPixmap(_bgr_image_to_qpixmap(self.stitched_image))

        self._sync_current_frame_pixmap()
        self._sync_scene_rect()
        self._sync_frame_progress_slider()
        self._sync_alignment_guides()
        self.make_image_button.setEnabled(self.stitched_image is not None)
        if preserve_view:
            return

        self._fit_image_height_in_view()

    def _sync_current_frame_pixmap(self) -> None:
        current_frame = self.frames[self.current_frame_index]
        alpha_mask = None
        if self._should_cull_current_frame_overlap():
            alpha_mask = _build_frame_visibility_alpha_mask(
                frame=current_frame,
                frame_x=int(round(self.current_frame_item.pos().x())),
                frame_y=int(round(self.current_frame_item.pos().y())),
                stitched_image=self.stitched_image,
            )
        self.current_frame_item.setPixmap(
            _bgr_image_to_qpixmap(current_frame, alpha_mask=alpha_mask)
        )

    def _should_cull_current_frame_overlap(self) -> bool:
        if self.stitched_image is None or not hasattr(self, "final_image_item"):
            return False

        return self.final_image_item.opacity() >= STITCHED_IMAGE_NORMAL_OPACITY

    def _handle_current_frame_moved(self) -> None:
        if not hasattr(self, "current_frame_item"):
            return

        self._sync_current_frame_pixmap()
        self._sync_alignment_guides()

    def _handle_current_frame_drag_started(self) -> None:
        if not hasattr(self, "final_image_item") or self.stitched_image is None:
            return

        self.final_image_item.setOpacity(STITCHED_IMAGE_DRAG_OPACITY)
        self._sync_current_frame_pixmap()

    def _handle_current_frame_drag_finished(self) -> None:
        if not hasattr(self, "final_image_item"):
            return

        self.final_image_item.setOpacity(STITCHED_IMAGE_NORMAL_OPACITY)
        self._sync_current_frame_pixmap()

    def _nudge_current_frame(self, offset: QPointF) -> None:
        if not hasattr(self, "current_frame_item"):
            return

        if self.stitched_image is not None:
            self._handle_current_frame_drag_started()
            self.keyboard_nudge_opacity_timer.start()

        self.current_frame_item.setPos(self.current_frame_item.pos() + offset)
        self._sync_scene_rect()
        self._sync_alignment_guides()

    def _sync_frame_progress_slider(self) -> None:
        self._is_syncing_frame_progress = True
        self.frame_progress_slider.setValue(self.current_frame_index)
        self._is_syncing_frame_progress = False

    def _handle_frame_progress_changed(self, frame_index: int) -> None:
        if self._is_syncing_frame_progress:
            return

        if self.stitched_image is not None:
            self._handle_current_frame_drag_started()
            if not self.frame_progress_slider.isSliderDown():
                self.keyboard_nudge_opacity_timer.start()

        self.current_frame_index = min(max(0, frame_index), len(self.frames) - 1)
        self._sync_scene_items(preserve_view=True)

    def _handle_frame_progress_pressed(self) -> None:
        if self.stitched_image is None:
            return

        self.keyboard_nudge_opacity_timer.stop()
        self._handle_current_frame_drag_started()

    def _handle_frame_progress_released(self) -> None:
        self.keyboard_nudge_opacity_timer.stop()
        self._handle_current_frame_drag_finished()

    def _sync_alignment_guides(self) -> None:
        if not hasattr(self, "top_alignment_guide_item"):
            return
        if self.stitched_image is None:
            self.top_alignment_guide_item.hide()
            self.bottom_alignment_guide_item.hide()
            return

        stitched_rect = self._get_stitched_image_pixel_rect()
        current_rect = self._get_current_frame_pixel_rect()
        if stitched_rect.isEmpty() or current_rect.isEmpty():
            self.top_alignment_guide_item.hide()
            self.bottom_alignment_guide_item.hide()
            return

        line_left = min(stitched_rect.left(), current_rect.left())
        line_right = max(stitched_rect.right(), current_rect.right())
        self._sync_alignment_guide_item(
            guide_item=self.top_alignment_guide_item,
            is_aligned=_are_scene_values_aligned(stitched_rect.top(), current_rect.top()),
            line_left=line_left,
            line_right=line_right,
            line_y=stitched_rect.top(),
        )
        self._sync_alignment_guide_item(
            guide_item=self.bottom_alignment_guide_item,
            is_aligned=_are_scene_values_aligned(
                stitched_rect.bottom(),
                current_rect.bottom(),
            ),
            line_left=line_left,
            line_right=line_right,
            line_y=stitched_rect.bottom(),
        )

    def _sync_alignment_guide_item(
        self,
        guide_item: QGraphicsLineItem,
        is_aligned: bool,
        line_left: float,
        line_right: float,
        line_y: float,
    ) -> None:
        if not is_aligned:
            guide_item.setLine(0.0, 0.0, 0.0, 0.0)
            guide_item.hide()
            return

        guide_item.setLine(line_left, line_y, line_right, line_y)
        guide_item.show()

    def _snap_current_frame_to_nearest_alignment_guide(self) -> None:
        if self.stitched_image is None or not hasattr(self, "current_frame_item"):
            return

        stitched_rect = self._get_stitched_image_pixel_rect()
        current_rect = self._get_current_frame_pixel_rect()
        if stitched_rect.isEmpty() or current_rect.isEmpty():
            return

        target_top_for_top_alignment = stitched_rect.top()
        target_top_for_bottom_alignment = stitched_rect.bottom() - current_rect.height()
        current_top = current_rect.top()
        top_distance = abs(current_top - target_top_for_top_alignment)
        bottom_distance = abs(current_top - target_top_for_bottom_alignment)
        if top_distance <= bottom_distance:
            snapped_y = target_top_for_top_alignment
        else:
            snapped_y = target_top_for_bottom_alignment

        current_position = self.current_frame_item.pos()
        self.current_frame_item.setPos(QPointF(current_position.x(), snapped_y))
        self._sync_scene_rect()
        self._sync_alignment_guides()

    def _get_stitched_image_pixel_rect(self) -> QRectF:
        if self.stitched_image is None:
            return QRectF()

        image_height, image_width = self.stitched_image.shape[:2]
        image_position = self.final_image_item.pos()
        return QRectF(
            image_position.x(),
            image_position.y(),
            float(image_width),
            float(image_height),
        )

    def _get_current_frame_pixel_rect(self) -> QRectF:
        current_frame = self.frames[self.current_frame_index]
        frame_height, frame_width = current_frame.shape[:2]
        frame_position = self.current_frame_item.pos()
        return QRectF(
            frame_position.x(),
            frame_position.y(),
            float(frame_width),
            float(frame_height),
        )

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if hasattr(self, "view"):
            self._fit_image_height_in_view(force=True)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.playback_timer.stop()
        super().closeEvent(event)

    def _sync_scene_rect(self) -> None:
        scene_rect = self._get_content_rect()
        margin = max(80.0, scene_rect.width() * 0.1, scene_rect.height() * 0.1)
        self.scene.setSceneRect(scene_rect.adjusted(-margin, -margin, margin, margin))

    def _fit_image_height_in_view(self, force: bool = False) -> None:
        content_rect = self._get_content_rect()
        if content_rect.isEmpty():
            return

        viewport_height = max(
            1.0,
            float(self.view.viewport().height()) - VIEWPORT_FIT_PADDING,
        )
        frame_height = self._get_frame_display_height()
        viewport_height_int = int(round(viewport_height))
        if (
            force
            or self._last_fit_viewport_height != viewport_height_int
            or self._last_fit_frame_height != frame_height
        ):
            self.view_scale = viewport_height / max(1.0, frame_height)
            self._last_fit_viewport_height = viewport_height_int
            self._last_fit_frame_height = frame_height

        self._apply_view_scale(center_on_content=True)

    def _apply_view_scale(self, center_on_content: bool = False) -> None:
        self.view.resetTransform()
        self.view.scale(self.view_scale, self.view_scale)
        if center_on_content:
            self.view.centerOn(self._get_content_rect().center())

    def _get_content_rect(self) -> QRectF:
        content_rect = QRectF(self.current_frame_item.sceneBoundingRect())
        if self.stitched_image is not None:
            content_rect = content_rect.united(self.final_image_item.sceneBoundingRect())

        return content_rect

    def _get_frame_display_height(self) -> float:
        current_frame = self.frames[self.current_frame_index]
        return float(current_frame.shape[0])

    def _go_back_frame(self) -> None:
        self.current_frame_index = max(0, self.current_frame_index - 1)
        self._sync_scene_items(preserve_view=True)

    def _go_forward_frame(self) -> None:
        self.current_frame_index = min(
            len(self.frames) - 1,
            self.current_frame_index + 1,
        )
        self._sync_scene_items(preserve_view=True)

    def _toggle_playback(self) -> None:
        if self.playback_timer.isActive():
            self.playback_timer.stop()
            self.play_button.setText("Play")
            return

        self.playback_timer.start()
        self.play_button.setText("Pause")

    def _advance_playback_frame(self) -> None:
        if self.current_frame_index >= len(self.frames) - 1:
            self.playback_timer.stop()
            self.play_button.setText("Play")
            return

        self.current_frame_index += 1
        self._sync_scene_items(preserve_view=True)

    def _stitch_current_frame(self) -> None:
        current_frame = self.frames[self.current_frame_index]
        current_position = self.current_frame_item.pos()
        if self.stitched_image is None:
            self.stitched_image = current_frame.copy()
        else:
            self.stitched_image, normalized_position = _add_frame_to_stitched_image(
                stitched_image=self.stitched_image,
                frame=current_frame,
                frame_x=int(round(current_position.x())),
                frame_y=int(round(current_position.y())),
            )
            self.current_frame_item.setPos(normalized_position)

        self.stitched_frame_count += 1
        self._advance_to_next_frame_after_stitch()
        self._sync_scene_items(preserve_view=True)

    def _handle_make_image_clicked(self) -> None:
        if self.stitched_image is None:
            return

        self.accept()

    def _advance_to_next_frame_after_stitch(self) -> None:
        if self.current_frame_index < len(self.frames) - 1:
            self.current_frame_index += 1

        self._place_current_frame_to_right_of_stitched_image()

    def _place_current_frame_to_right_of_stitched_image(self) -> None:
        if self.stitched_image is None:
            return

        stitched_width = self.stitched_image.shape[1]
        self.current_frame_item.setPos(QPointF(float(stitched_width), 0.0))


# ### Visual helpers ###
def _build_alignment_guide_pen() -> QPen:
    guide_pen = QPen(QColor("#22c55e"))
    guide_pen.setStyle(Qt.PenStyle.DotLine)
    guide_pen.setWidthF(1.5)
    guide_pen.setCosmetic(True)
    return guide_pen


def _are_scene_values_aligned(first_value: float, second_value: float) -> bool:
    return abs(first_value - second_value) <= ALIGNMENT_GUIDE_TOLERANCE_PIXELS


# ### Video helpers ###
def _load_video_frames(video_path: str) -> tuple[list[np.ndarray], int]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    try:
        frames: list[np.ndarray] = []
        while True:
            did_read, frame = capture.read()
            if not did_read or frame is None:
                break

            frames.append(_normalize_video_frame(frame))

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        capture.release()

    if fps <= 0.0:
        return frames, DEFAULT_MANUAL_PLAYBACK_INTERVAL_MS

    return frames, max(1, int(round(1000.0 / fps)))


def _normalize_video_frame(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    return frame


# ### Stitch helpers ###
def _add_frame_to_stitched_image(
    stitched_image: np.ndarray,
    frame: np.ndarray,
    frame_x: int,
    frame_y: int,
) -> tuple[np.ndarray, QPointF]:
    stitched_height, stitched_width = stitched_image.shape[:2]
    frame_height, frame_width = frame.shape[:2]
    min_x = min(0, frame_x)
    min_y = min(0, frame_y)
    max_x = max(stitched_width, frame_x + frame_width)
    max_y = max(stitched_height, frame_y + frame_height)

    output_width = max_x - min_x
    output_height = max_y - min_y
    output_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)

    stitched_x = -min_x
    stitched_y = -min_y
    output_image[
        stitched_y : stitched_y + stitched_height,
        stitched_x : stitched_x + stitched_width,
    ] = stitched_image

    normalized_frame_x = frame_x - min_x
    normalized_frame_y = frame_y - min_y
    frame_target = output_image[
        normalized_frame_y : normalized_frame_y + frame_height,
        normalized_frame_x : normalized_frame_x + frame_width,
    ]
    visibility_mask = _build_non_intersecting_frame_mask(
        frame_shape=frame.shape,
        frame_x=normalized_frame_x,
        frame_y=normalized_frame_y,
        stitched_x=stitched_x,
        stitched_y=stitched_y,
        stitched_width=stitched_width,
        stitched_height=stitched_height,
    )
    frame_target[visibility_mask] = frame[visibility_mask]

    return output_image, QPointF(float(normalized_frame_x), float(normalized_frame_y))


def _build_frame_visibility_alpha_mask(
    frame: np.ndarray,
    frame_x: int,
    frame_y: int,
    stitched_image: np.ndarray | None,
) -> np.ndarray | None:
    if stitched_image is None:
        return None

    stitched_height, stitched_width = stitched_image.shape[:2]
    visibility_mask = _build_non_intersecting_frame_mask(
        frame_shape=frame.shape,
        frame_x=frame_x,
        frame_y=frame_y,
        stitched_x=0,
        stitched_y=0,
        stitched_width=stitched_width,
        stitched_height=stitched_height,
    )
    return np.where(visibility_mask, 255, 0).astype(np.uint8)


def _build_non_intersecting_frame_mask(
    frame_shape: tuple[int, ...],
    frame_x: int,
    frame_y: int,
    stitched_x: int,
    stitched_y: int,
    stitched_width: int,
    stitched_height: int,
) -> np.ndarray:
    frame_height, frame_width = frame_shape[:2]
    visibility_mask = np.ones((frame_height, frame_width), dtype=bool)

    intersection_left = max(frame_x, stitched_x)
    intersection_top = max(frame_y, stitched_y)
    intersection_right = min(frame_x + frame_width, stitched_x + stitched_width)
    intersection_bottom = min(frame_y + frame_height, stitched_y + stitched_height)
    if intersection_left >= intersection_right:
        return visibility_mask
    if intersection_top >= intersection_bottom:
        return visibility_mask

    visibility_mask[
        intersection_top - frame_y : intersection_bottom - frame_y,
        intersection_left - frame_x : intersection_right - frame_x,
    ] = False
    return visibility_mask


# ### Image helpers ###
def _bgr_image_to_qpixmap(
    image_bgr: np.ndarray,
    alpha_mask: np.ndarray | None = None,
) -> QPixmap:
    image_array = np.ascontiguousarray(image_bgr.astype(np.uint8))
    if alpha_mask is not None:
        image_rgba = _build_rgba_image_array(image_array, alpha_mask)
        image_height, image_width, channel_count = image_rgba.shape
        image = QImage(
            image_rgba.data,
            image_width,
            image_height,
            channel_count * image_width,
            QImage.Format.Format_RGBA8888,
        ).copy()
        return QPixmap.fromImage(image)

    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    image_height, image_width, channel_count = image_rgb.shape
    image = QImage(
        image_rgb.data,
        image_width,
        image_height,
        channel_count * image_width,
        QImage.Format.Format_RGB888,
    ).copy()
    return QPixmap.fromImage(image)


def _build_rgba_image_array(
    image_bgr: np.ndarray,
    alpha_mask: np.ndarray,
) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    normalized_alpha = np.ascontiguousarray(alpha_mask.astype(np.uint8))
    return np.dstack((image_rgb, normalized_alpha))


# ### File helpers ###
def build_unique_stitched_output_path(video_path: str) -> Path:
    source_path = Path(video_path)
    output_path = source_path.with_name(f"{source_path.stem}_stitched.png")
    suffix_index = 1
    while output_path.exists():
        output_path = source_path.with_name(
            f"{source_path.stem}_stitched_{suffix_index}.png"
        )
        suffix_index += 1

    return output_path


def save_stitched_image(image_bgr: np.ndarray, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image_bgr):
        raise OSError(f"Unable to save stitched image: {output_path}")

    return output_path
