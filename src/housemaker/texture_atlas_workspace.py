# ### Imports ###
from __future__ import annotations

import copy
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from PySide6.QtCore import (
    QByteArray,
    QLineF,
    QMimeData,
    QPointF,
    QRectF,
    QStandardPaths,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QDrag,
    QDragEnterEvent,
    QDragLeaveEvent,
    QDragMoveEvent,
    QDropEvent,
    QImage,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QShortcut,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from housemaker.texture_atlas_state import (
    ATLAS_BASE_CELL_SIZE,
    ATLAS_DOUBLE_SIZED_PACKING_MODES,
    ATLAS_HALF_SLOT_PACKING_MODES,
    ATLAS_LIMITED_RESOLUTION_PACKING_MODES,
    ATLAS_PACKING_MODE_FULL,
    ATLAS_PACKING_MODE_SYMMETRIC_HALF,
    ATLAS_PACKING_MODE_SYMMETRIC_PAIR,
    ATLAS_PACKING_MODE_SYMMETRIC_QUARTER,
    ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR,
    ATLAS_PACKING_MODES,
    ATLAS_RESOLUTIONS,
    ATLAS_SLOT_HALF_LEFT,
    ATLAS_SLOT_HALF_RIGHT,
    ATLAS_SLOT_QUADRANT_BOTTOM_LEFT,
    ATLAS_SLOT_QUADRANT_BOTTOM_RIGHT,
    ATLAS_SLOT_QUADRANT_ORDER,
    ATLAS_SLOT_QUADRANT_TOP_RIGHT,
    SYMMETRIC_PACKED_TEXTURE_RESOLUTIONS,
    TextureAtlasData,
    TextureAtlasPlacement,
    TextureAtlasRecord,
    write_texture_atlas_png,
)


# ### Constants ###
ATLAS_ID_ROLE = Qt.ItemDataRole.UserRole
OBJECT_ID_ROLE = Qt.ItemDataRole.UserRole
OBJECT_MISSING_ROLE = Qt.ItemDataRole.UserRole + 1
PREVIEW_MARGIN_PIXELS = 16.0
PREVIEW_BACKGROUND_COLOR = QColor(31, 34, 39)
PREVIEW_EMPTY_COLOR = QColor(50, 54, 61)
PREVIEW_GRID_COLOR = QColor(76, 82, 92)
PREVIEW_MISSING_COLOR = QColor(105, 59, 52)
PREVIEW_BORDER_COLOR = QColor(220, 224, 230)
PREVIEW_SELECTED_BORDER_COLOR = QColor(255, 139, 31)
PREVIEW_LABEL_COLOR = QColor(245, 247, 250)
PREVIEW_DRAG_VALID_FILL_COLOR = QColor(38, 190, 95, 72)
PREVIEW_DRAG_VALID_BORDER_COLOR = QColor(77, 255, 142, 235)
PREVIEW_DRAG_BLOCKED_FILL_COLOR = QColor(225, 56, 64, 88)
PREVIEW_DRAG_BLOCKED_BORDER_COLOR = QColor(255, 104, 111, 240)
PREVIEW_DRAG_IMAGE_OPACITY = 0.46
MAX_SOURCE_THUMBNAIL_SIZE = 256
MAX_VARIANT_SOURCE_CACHE_ENTRIES = 256
TEXTURE_RESOLUTION_ORDER = (512, 1024, 2048)
WALL_TEXTURE_SOURCE_ID_PREFIX = "surface-wall-texture:"
ATLAS_TEXTURE_SOURCE_MIME_TYPE = (
    "application/x-housemaker-texture-atlas-source"
)
MAX_DRAG_SOURCE_ID_BYTES = 4_096


# ### Public texture-source model ###
@dataclass(frozen=True, eq=False)
class AtlasObjectTextureSource:
    """One generated object or wall texture that can be packed into an atlas."""

    object_id: str
    object_name: str
    texture_path: str
    texture_resolution: int
    physical_texture_path: Path
    preview_rgba: np.ndarray
    fit_to_square: bool = False
    supports_resolution_changes: bool = True
    supports_3d_preview: bool = True
    packing_mode: str = ATLAS_PACKING_MODE_FULL
    symmetric_preview_orientation: str | None = None
    symmetric_preview_plane_coordinate: float | None = None

    def __post_init__(self) -> None:
        object_id = str(self.object_id).strip()
        object_name = str(self.object_name).strip()
        texture_path = str(self.texture_path).strip()
        if not object_id:
            raise ValueError("Atlas texture source ID cannot be empty.")
        if not object_name:
            raise ValueError("Atlas texture source name cannot be empty.")
        if not texture_path:
            raise ValueError("Atlas texture path cannot be empty.")
        if int(self.texture_resolution) not in {512, 1024, 2048}:
            raise ValueError("Atlas source texture must be 512, 1024, or 2048.")
        physical_texture_path = Path(self.physical_texture_path)
        rgba = _normalize_preview_rgba(self.preview_rgba)
        rgba.setflags(write=False)
        if max(rgba.shape[:2]) > MAX_SOURCE_THUMBNAIL_SIZE:
            raise ValueError(
                "Atlas source previews cannot exceed 256 pixels per side."
            )
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "object_name", object_name)
        object.__setattr__(self, "texture_path", texture_path)
        object.__setattr__(self, "texture_resolution", int(self.texture_resolution))
        object.__setattr__(self, "physical_texture_path", physical_texture_path)
        object.__setattr__(self, "preview_rgba", rgba)
        object.__setattr__(self, "fit_to_square", bool(self.fit_to_square))
        object.__setattr__(
            self,
            "supports_resolution_changes",
            bool(self.supports_resolution_changes),
        )
        object.__setattr__(
            self,
            "supports_3d_preview",
            bool(self.supports_3d_preview),
        )
        packing_mode = str(self.packing_mode).strip().lower()
        if packing_mode not in ATLAS_PACKING_MODES:
            raise ValueError("Unknown Atlas source packing mode.")
        if (
            packing_mode in ATLAS_LIMITED_RESOLUTION_PACKING_MODES
            and int(self.texture_resolution)
            not in SYMMETRIC_PACKED_TEXTURE_RESOLUTIONS
        ):
            raise ValueError(
                "High-density symmetric Atlas sources must use 512 or 1024 "
                "content."
            )
        if packing_mode != ATLAS_PACKING_MODE_FULL and bool(self.fit_to_square):
            raise ValueError(
                "Symmetric Atlas sources require an exact square texture."
            )
        orientation = (
            None
            if self.symmetric_preview_orientation is None
            else str(self.symmetric_preview_orientation).strip().lower()
        )
        plane_coordinate = self.symmetric_preview_plane_coordinate
        if packing_mode == ATLAS_PACKING_MODE_FULL:
            if orientation is not None or plane_coordinate is not None:
                raise ValueError(
                    "Full Atlas sources cannot define a symmetric preview."
                )
        else:
            if orientation not in {"vertical", "horizontal"}:
                raise ValueError(
                    "Symmetric Atlas sources require a preview orientation."
                )
            if plane_coordinate is None or not math.isfinite(
                float(plane_coordinate)
            ):
                raise ValueError(
                    "Symmetric Atlas sources require a finite preview plane."
                )
            plane_coordinate = float(plane_coordinate)
        object.__setattr__(self, "packing_mode", packing_mode)
        object.__setattr__(self, "symmetric_preview_orientation", orientation)
        object.__setattr__(
            self,
            "symmetric_preview_plane_coordinate",
            plane_coordinate,
        )

    def get_preview_image(self) -> QImage:
        """Return an owned Qt image for atlas preview painting."""

        rgba = self.preview_rgba
        image = QImage(
            rgba.data,
            rgba.shape[1],
            rgba.shape[0],
            rgba.strides[0],
            QImage.Format.Format_RGBA8888,
        )
        return image.copy()

    def load_texture_rgba(self) -> np.ndarray:
        """Decode and validate the exact PNG only when atlas output needs it."""

        with Image.open(self.physical_texture_path) as image:
            if self.fit_to_square:
                loaded_image = _fit_image_to_square(
                    image,
                    self.texture_resolution,
                )
            else:
                loaded_image = image.convert("RGBA")
            rgba = np.asarray(loaded_image, dtype=np.uint8)
        physical_resolution = _physical_texture_resolution(
            self.packing_mode,
            self.texture_resolution,
        )
        expected_shape = (physical_resolution, physical_resolution)
        if rgba.shape[:2] != expected_shape:
            raise ValueError(
                f"Texture {self.texture_path!r} must be "
                f"{physical_resolution} x {physical_resolution}; "
                f"received {rgba.shape[1]} x {rgba.shape[0]}."
            )
        return np.ascontiguousarray(rgba)


@dataclass(frozen=True)
class AtlasDragSlotPreview:
    """One non-mutating, snapped placement shown during an Atlas drag."""

    object_id: str
    x: int
    y: int
    size: int
    slot_half: str | None
    slot_quadrant: str | None
    is_valid: bool


TextureVariantResolver = Callable[
    [str, int],
    AtlasObjectTextureSource | None,
]
TextureVariantSelectabilityResolver = Callable[[str, int], bool]
TextureResolutionCommitCallback = Callable[[], bool]


def load_atlas_object_texture_source(
    *,
    object_id: str,
    object_name: str,
    texture_path: str,
    texture_resolution: int,
    physical_texture_path: str | Path,
    fit_to_square: bool = False,
    supports_resolution_changes: bool = True,
    supports_3d_preview: bool = True,
    packing_mode: str = ATLAS_PACKING_MODE_FULL,
    symmetric_preview_orientation: str | None = None,
    symmetric_preview_plane_coordinate: float | None = None,
) -> AtlasObjectTextureSource:
    """Load one texture-source descriptor for the Atlas workspace."""

    normalized_physical_path = Path(physical_texture_path)
    with Image.open(normalized_physical_path) as image:
        physical_resolution = _physical_texture_resolution(
            packing_mode,
            int(texture_resolution),
        )
        if not fit_to_square and image.size != (
            physical_resolution,
            physical_resolution,
        ):
            raise ValueError(
                f"Texture {str(texture_path)!r} must be "
                f"{physical_resolution} x {physical_resolution}; "
                f"received {image.width} x {image.height}."
            )
        thumbnail = (
            _fit_image_to_square(image, int(texture_resolution))
            if fit_to_square
            else image.convert("RGBA")
        )
        thumbnail.thumbnail(
            (MAX_SOURCE_THUMBNAIL_SIZE, MAX_SOURCE_THUMBNAIL_SIZE),
            Image.Resampling.LANCZOS,
        )
        rgba = np.asarray(thumbnail, dtype=np.uint8)
    return AtlasObjectTextureSource(
        object_id=object_id,
        object_name=object_name,
        texture_path=texture_path,
        texture_resolution=texture_resolution,
        physical_texture_path=normalized_physical_path,
        preview_rgba=rgba,
        fit_to_square=fit_to_square,
        supports_resolution_changes=supports_resolution_changes,
        supports_3d_preview=supports_3d_preview,
        packing_mode=packing_mode,
        symmetric_preview_orientation=symmetric_preview_orientation,
        symmetric_preview_plane_coordinate=symmetric_preview_plane_coordinate,
    )


def choose_atlas_texture_resolution(width: int, height: int) -> int:
    """Choose the smallest supported square that contains an image."""

    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or int(width) <= 0
        or int(height) <= 0
    ):
        raise ValueError("Atlas texture dimensions must be positive integers.")
    longest_side = max(int(width), int(height))
    return next(
        (
            resolution
            for resolution in TEXTURE_RESOLUTION_ORDER
            if longest_side <= resolution
        ),
        TEXTURE_RESOLUTION_ORDER[-1],
    )


def build_atlas_wall_texture_source_id(assignment_id: str) -> str:
    """Return the reserved Atlas identity for one generated wall texture."""

    normalized_id = str(assignment_id).strip()
    if not normalized_id:
        raise ValueError("Wall texture assignment ID cannot be empty.")
    return f"{WALL_TEXTURE_SOURCE_ID_PREFIX}{normalized_id}"


def is_atlas_wall_texture_source_id(source_id: object) -> bool:
    """Return whether an Atlas source ID belongs to a wall texture."""

    return str(source_id).startswith(WALL_TEXTURE_SOURCE_ID_PREFIX)


def get_atlas_wall_texture_assignment_id(source_id: object) -> str | None:
    """Return the surface assignment ID encoded in an Atlas source ID."""

    normalized_id = str(source_id)
    if not normalized_id.startswith(WALL_TEXTURE_SOURCE_ID_PREFIX):
        return None
    assignment_id = normalized_id[len(WALL_TEXTURE_SOURCE_ID_PREFIX) :]
    return assignment_id or None


# ### Interactive object list ###
class TextureAtlasObjectList(QListWidget):
    """Selectable, draggable texture sources with wheel-only resizing."""

    object_clicked = Signal(str, object)
    object_wheeled = Signal(str, int)

    def set_wheel_resize_object_ids(self, object_ids: set[str]) -> None:
        """Limit wheel capture to packed sources that can actually resize."""

        self._wheel_resize_object_ids = frozenset(
            str(object_id) for object_id in object_ids
        )

    def startDrag(self, supported_actions: Qt.DropAction) -> None:  # type: ignore[override]
        del supported_actions
        item = self.currentItem()
        if item is None or bool(item.data(OBJECT_MISSING_ROLE)):
            return
        object_id = item.data(OBJECT_ID_ROLE)
        if object_id is None:
            return
        try:
            mime_data = _build_texture_source_mime_data(str(object_id))
        except ValueError:
            return
        drag = QDrag(self)
        drag.setMimeData(mime_data)
        drag.exec(Qt.DropAction.CopyAction)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        item = self.itemAt(event.position().toPoint())
        button = event.button()
        is_supported_click = button in {
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.RightButton,
        }
        self._mouse_button_in_progress = button if is_supported_click else None
        try:
            super().mousePressEvent(event)
        finally:
            self._mouse_button_in_progress = None
        if item is None or not is_supported_click:
            return
        object_id = item.data(OBJECT_ID_ROLE)
        if object_id is not None:
            self.object_clicked.emit(str(object_id), button)

    def wheelEvent(self, event: QWheelEvent) -> None:  # type: ignore[override]
        """Resize the selected texture instead of the hovered list row."""

        item = self.currentItem()
        wheel_delta = int(event.angleDelta().y())
        if item is None or wheel_delta == 0:
            super().wheelEvent(event)
            return
        object_id = item.data(OBJECT_ID_ROLE)
        if object_id is None or str(object_id) not in getattr(
            self,
            "_wheel_resize_object_ids",
            frozenset(),
        ):
            super().wheelEvent(event)
            return
        self.object_wheeled.emit(
            str(object_id),
            1 if wheel_delta > 0 else -1,
        )
        event.accept()

    @property
    def mouse_button_in_progress(self) -> object | None:
        return getattr(self, "_mouse_button_in_progress", None)


# ### Atlas preview ###
class TextureAtlasPreview(QWidget):
    """Scalable atlas preview that preserves unavailable source allocations."""

    object_clicked = Signal(str, object)
    object_wheeled = Signal(str, int)
    object_dropped = Signal(str, int, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._atlas: TextureAtlasRecord | None = None
        self._sources: dict[str, AtlasObjectTextureSource] = {}
        self._source_preview_images: dict[str, QImage] = {}
        self._content_signature: tuple[object, ...] | None = None
        self._selected_object_id: str | None = None
        self._drag_start_position: QPointF | None = None
        self._drag_object_id: str | None = None
        self._drag_slot_preview: AtlasDragSlotPreview | None = None
        self._wheel_resize_object_ids: frozenset[str] = frozenset()
        self.setMinimumSize(360, 360)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_content(
        self,
        atlas: TextureAtlasRecord | None,
        sources: dict[str, AtlasObjectTextureSource],
    ) -> None:
        content_signature = _build_atlas_preview_content_signature(
            atlas,
            sources,
        )
        if content_signature == self._content_signature:
            self._clear_drag_slot_preview()
            return

        previous_sources = self._sources
        previous_images = self._source_preview_images
        source_preview_images: dict[str, QImage] = {}
        for object_id, source in sources.items():
            if (
                previous_sources.get(object_id) is source
                and object_id in previous_images
            ):
                source_preview_images[object_id] = previous_images[object_id]
            else:
                source_preview_images[object_id] = source.get_preview_image()

        self._atlas = atlas
        self._sources = dict(sources)
        self._source_preview_images = source_preview_images
        self._content_signature = content_signature
        self._drag_slot_preview = None
        self.update()

    @property
    def drag_slot_preview(self) -> AtlasDragSlotPreview | None:
        """Return the currently painted drag feedback without atlas mutation."""

        return self._drag_slot_preview

    def set_selected_object_id(self, object_id: str | None) -> None:
        """Highlight one placement without changing atlas data."""

        normalized_id = None if object_id is None else str(object_id)
        if normalized_id == self._selected_object_id:
            return
        self._selected_object_id = normalized_id
        self.update()

    def set_wheel_resize_object_ids(self, object_ids: set[str]) -> None:
        """Limit wheel capture to packed sources that can resize."""

        self._wheel_resize_object_ids = frozenset(
            str(object_id) for object_id in object_ids
        )

    def object_id_at(self, position: QPointF) -> str | None:
        """Return the placement at one widget-space position, if any."""

        atlas = self._atlas
        if atlas is None:
            return None
        atlas_rect = _aspect_fit_square(self.width(), self.height())
        if (
            atlas_rect.width() <= 0.0
            or position.x() < atlas_rect.left()
            or position.y() < atlas_rect.top()
            or position.x() >= atlas_rect.right()
            or position.y() >= atlas_rect.bottom()
        ):
            return None
        scale = float(atlas.resolution) / atlas_rect.width()
        atlas_x = (position.x() - atlas_rect.left()) * scale
        atlas_y = (position.y() - atlas_rect.top()) * scale
        for placement in atlas.placements:
            content_x = placement.x
            content_y = placement.y
            content_width = placement.size
            content_height = placement.size
            if placement.packing_mode in ATLAS_HALF_SLOT_PACKING_MODES:
                content_width = placement.size / 2.0
                if placement.slot_half == ATLAS_SLOT_HALF_RIGHT:
                    content_x += content_width
            elif (
                placement.packing_mode
                == ATLAS_PACKING_MODE_SYMMETRIC_QUARTER
            ):
                content_width = placement.texture_resolution
                content_height = placement.texture_resolution
                offset_x, offset_y = _quarter_region_offset(
                    placement.slot_quadrant,
                    placement.texture_resolution,
                )
                content_x += offset_x
                content_y += offset_y
            if (
                content_x <= atlas_x < content_x + content_width
                and content_y <= atlas_y < content_y + content_height
            ):
                return placement.object_id
        return None

    def atlas_slot_at(
        self,
        object_id: str,
        position: QPointF,
    ) -> tuple[int, int] | None:
        """Return the exact 512-pixel grid slot beneath a widget point."""

        atlas = self._atlas
        source = self._sources.get(str(object_id))
        if atlas is None or source is None:
            return None
        atlas_rect = _aspect_fit_square(self.width(), self.height())
        if (
            atlas_rect.width() <= 0.0
            or position.x() < atlas_rect.left()
            or position.y() < atlas_rect.top()
            or position.x() >= atlas_rect.right()
            or position.y() >= atlas_rect.bottom()
        ):
            return None
        scale = float(atlas.resolution) / atlas_rect.width()
        atlas_x = int((position.x() - atlas_rect.left()) * scale)
        atlas_y = int((position.y() - atlas_rect.top()) * scale)
        return (
            (atlas_x // ATLAS_BASE_CELL_SIZE) * ATLAS_BASE_CELL_SIZE,
            (atlas_y // ATLAS_BASE_CELL_SIZE) * ATLAS_BASE_CELL_SIZE,
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        super().mousePressEvent(event)
        if event.button() not in {
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.RightButton,
        }:
            return
        object_id = self.object_id_at(event.position())
        if object_id is not None:
            self.object_clicked.emit(object_id, event.button())
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_position = QPointF(event.position())
            self._drag_object_id = object_id

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        """Drag an existing placement to another exact Atlas slot."""

        if not bool(event.buttons() & Qt.MouseButton.LeftButton):
            super().mouseMoveEvent(event)
            return
        start_position = self._drag_start_position
        object_id = self._drag_object_id
        if start_position is None or object_id is None:
            super().mouseMoveEvent(event)
            return
        distance = (
            event.position().toPoint() - start_position.toPoint()
        ).manhattanLength()
        if distance < QApplication.startDragDistance():
            super().mouseMoveEvent(event)
            return

        self._drag_start_position = None
        self._drag_object_id = None
        try:
            mime_data = _build_texture_source_mime_data(object_id)
        except ValueError:
            return
        drag = QDrag(self)
        drag.setMimeData(mime_data)
        drag.exec(Qt.DropAction.CopyAction)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        """Clear a pending placement drag when the left button is released."""

        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_position = None
            self._drag_object_id = None
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:  # type: ignore[override]
        """Resize the selected placement regardless of pointer position."""

        object_id = self._selected_object_id
        wheel_delta = int(event.angleDelta().y())
        if (
            object_id is None
            or object_id not in self._wheel_resize_object_ids
            or wheel_delta == 0
        ):
            super().wheelEvent(event)
            return
        self.object_wheeled.emit(
            object_id,
            1 if wheel_delta > 0 else -1,
        )
        event.accept()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        object_id = _read_texture_source_mime_data(event.mimeData())
        if object_id is not None and self._update_drag_slot_preview(
            object_id,
            QPointF(event.position()),
        ):
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
            return
        self._clear_drag_slot_preview()
        event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:  # type: ignore[override]
        object_id = _read_texture_source_mime_data(event.mimeData())
        if object_id is not None and self._update_drag_slot_preview(
            object_id,
            QPointF(event.position()),
        ):
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
            return
        self._clear_drag_slot_preview()
        event.ignore()

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:  # type: ignore[override]
        """Remove transient slot feedback when a drag leaves the preview."""

        self._clear_drag_slot_preview()
        event.accept()

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        object_id = _read_texture_source_mime_data(event.mimeData())
        has_preview = (
            object_id is not None
            and self._update_drag_slot_preview(object_id, event.position())
        )
        slot_preview = self._drag_slot_preview if has_preview else None
        if slot_preview is None or not slot_preview.is_valid:
            self._clear_drag_slot_preview()
            event.ignore()
            return
        self.object_dropped.emit(
            slot_preview.object_id,
            slot_preview.x,
            slot_preview.y,
        )
        self._clear_drag_slot_preview()
        event.setDropAction(Qt.DropAction.CopyAction)
        event.accept()

    def paintEvent(self, event: QPaintEvent) -> None:  # type: ignore[override]
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.fillRect(self.rect(), PREVIEW_BACKGROUND_COLOR)
        atlas = self._atlas
        if atlas is None:
            painter.setPen(PREVIEW_LABEL_COLOR)
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Create or select a texture atlas",
            )
            return

        atlas_rect = _aspect_fit_square(self.width(), self.height())
        painter.fillRect(atlas_rect, PREVIEW_EMPTY_COLOR)
        self._paint_quadtree_grid(painter, atlas_rect, atlas.resolution)
        painted_symmetric_slots: set[tuple[int, int, int]] = set()
        for placement in atlas.placements:
            logical_rect = _placement_preview_rect(
                placement,
                atlas.resolution,
                atlas_rect,
            )
            if placement.packing_mode in {
                ATLAS_PACKING_MODE_SYMMETRIC_HALF,
                ATLAS_PACKING_MODE_SYMMETRIC_PAIR,
                ATLAS_PACKING_MODE_SYMMETRIC_QUARTER,
                ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR,
            }:
                slot_key = (placement.x, placement.y, placement.size)
                if slot_key not in painted_symmetric_slots:
                    painter.fillRect(logical_rect, QColor(0, 0, 0, 255))
                    painted_symmetric_slots.add(slot_key)
            placement_rect = _placement_content_preview_rect(
                placement,
                atlas.resolution,
                atlas_rect,
            )
            source = self._sources.get(placement.object_id)
            if source is None:
                painter.fillRect(placement_rect, PREVIEW_MISSING_COLOR)
                label = f"Missing\n{placement.object_id}"
            else:
                preview_image = self._source_preview_images[placement.object_id]
                if placement.packing_mode in ATLAS_HALF_SLOT_PACKING_MODES:
                    painter.drawImage(
                        placement_rect,
                        preview_image,
                        QRectF(
                            0.0,
                            0.0,
                            preview_image.width() / 2.0,
                            float(preview_image.height()),
                        ),
                    )
                elif (
                    placement.packing_mode
                    == ATLAS_PACKING_MODE_SYMMETRIC_QUARTER
                ):
                    painter.drawImage(
                        placement_rect,
                        preview_image,
                        QRectF(
                            0.0,
                            0.0,
                            preview_image.width() / 2.0,
                            preview_image.height() / 2.0,
                        ),
                    )
                else:
                    painter.drawImage(placement_rect, preview_image)
                label = (
                    f"{source.object_name}\n"
                    f"{placement.texture_resolution} x "
                    f"{placement.texture_resolution}"
                )
            is_selected = placement.object_id == self._selected_object_id
            painter.setPen(
                QPen(
                    (
                        PREVIEW_SELECTED_BORDER_COLOR
                        if is_selected
                        else PREVIEW_BORDER_COLOR
                    ),
                    3.0 if is_selected else 1.0,
                )
            )
            painter.drawRect(placement_rect)
            painter.setPen(PREVIEW_LABEL_COLOR)
            painter.drawText(
                placement_rect.adjusted(5.0, 5.0, -5.0, -5.0),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                label,
            )

        self._paint_drag_slot_preview(painter, atlas_rect)

        painter.setPen(QPen(PREVIEW_BORDER_COLOR, 2.0))
        painter.drawRect(atlas_rect)

    def _drag_slot_preview_at(
        self,
        object_id: str,
        position: QPointF,
    ) -> AtlasDragSlotPreview | None:
        """Build snapped feedback, including collision and bounds validity."""

        atlas = self._atlas
        source = self._sources.get(str(object_id))
        if atlas is None or source is None:
            return None
        atlas_rect = _aspect_fit_square(self.width(), self.height())
        if atlas_rect.width() <= 0.0:
            return None

        pointer_is_inside = (
            position.x() >= atlas_rect.left()
            and position.y() >= atlas_rect.top()
            and position.x() < atlas_rect.right()
            and position.y() < atlas_rect.bottom()
        )
        scale = float(atlas.resolution) / atlas_rect.width()
        raw_x = int((position.x() - atlas_rect.left()) * scale)
        raw_y = int((position.y() - atlas_rect.top()) * scale)
        atlas_x = min(max(raw_x, 0), max(0, atlas.resolution - 1))
        atlas_y = min(max(raw_y, 0), max(0, atlas.resolution - 1))
        slot_size = _physical_texture_resolution(
            source.packing_mode,
            source.texture_resolution,
        )
        existing = atlas.placement_for_object(str(object_id))
        if (
            source.packing_mode
            in {
                ATLAS_PACKING_MODE_SYMMETRIC_HALF,
                ATLAS_PACKING_MODE_SYMMETRIC_PAIR,
                ATLAS_PACKING_MODE_SYMMETRIC_QUARTER,
                ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR,
            }
            and existing is not None
            and existing.packing_mode == source.packing_mode
            and existing.texture_resolution == source.texture_resolution
            and existing.x <= atlas_x < existing.x + existing.size
            and existing.y <= atlas_y < existing.y + existing.size
        ):
            return AtlasDragSlotPreview(
                object_id=str(object_id),
                x=existing.x,
                y=existing.y,
                size=existing.size,
                slot_half=existing.slot_half,
                slot_quadrant=existing.slot_quadrant,
                is_valid=pointer_is_inside,
            )
        compatible_group: list[TextureAtlasPlacement] | None = None
        if source.packing_mode in {
            ATLAS_PACKING_MODE_SYMMETRIC_HALF,
            ATLAS_PACKING_MODE_SYMMETRIC_PAIR,
            ATLAS_PACKING_MODE_SYMMETRIC_QUARTER,
            ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR,
        }:
            visited_slots: set[tuple[int, int, int]] = set()
            for placement in atlas.placements:
                slot_key = (placement.x, placement.y, placement.size)
                if slot_key in visited_slots:
                    continue
                visited_slots.add(slot_key)
                if (
                    placement.object_id == object_id
                    or placement.packing_mode != source.packing_mode
                    or placement.texture_resolution
                    != source.texture_resolution
                    or not (
                        placement.x <= atlas_x < placement.x + placement.size
                        and placement.y
                        <= atlas_y
                        < placement.y + placement.size
                    )
                ):
                    continue
                slot_members = [
                    candidate
                    for candidate in atlas.placements
                    if candidate.object_id != object_id
                    and candidate.x == placement.x
                    and candidate.y == placement.y
                    and candidate.size == placement.size
                ]
                capacity = (
                    2
                    if source.packing_mode
                    in ATLAS_HALF_SLOT_PACKING_MODES
                    else 4
                )
                if 0 < len(slot_members) < capacity:
                    compatible_group = slot_members
                    break
        slot_x = (
            compatible_group[0].x
            if compatible_group is not None
            else (atlas_x // ATLAS_BASE_CELL_SIZE) * ATLAS_BASE_CELL_SIZE
        )
        slot_y = (
            compatible_group[0].y
            if compatible_group is not None
            else (atlas_y // ATLAS_BASE_CELL_SIZE) * ATLAS_BASE_CELL_SIZE
        )
        slot_half = (
            ATLAS_SLOT_HALF_RIGHT
            if compatible_group is not None
            and source.packing_mode
            in ATLAS_HALF_SLOT_PACKING_MODES
            else (
                ATLAS_SLOT_HALF_LEFT
                if source.packing_mode
                in ATLAS_HALF_SLOT_PACKING_MODES
                else None
            )
        )
        occupied_quadrants = (
            {
                placement.slot_quadrant
                for placement in compatible_group
            }
            if compatible_group is not None
            else set()
        )
        slot_quadrant = (
            next(
                quadrant
                for quadrant in ATLAS_SLOT_QUADRANT_ORDER
                if quadrant not in occupied_quadrants
            )
            if source.packing_mode == ATLAS_PACKING_MODE_SYMMETRIC_QUARTER
            else None
        )
        fits_bounds = (
            slot_x + slot_size <= atlas.resolution
            and slot_y + slot_size <= atlas.resolution
        )
        overlaps_other = any(
            placement.object_id != object_id
            and not (
                compatible_group is not None
                and placement.x == compatible_group[0].x
                and placement.y == compatible_group[0].y
                and placement.size == compatible_group[0].size
            )
            and slot_x < placement.x + placement.size
            and slot_x + slot_size > placement.x
            and slot_y < placement.y + placement.size
            and slot_y + slot_size > placement.y
            for placement in atlas.placements
        )
        return AtlasDragSlotPreview(
            object_id=str(object_id),
            x=slot_x,
            y=slot_y,
            size=slot_size,
            slot_half=slot_half,
            slot_quadrant=slot_quadrant,
            is_valid=(
                pointer_is_inside and fits_bounds and not overlaps_other
            ),
        )

    def _update_drag_slot_preview(
        self,
        object_id: str,
        position: QPointF,
    ) -> bool:
        """Update live drag feedback and schedule only necessary repaints."""

        slot_preview = self._drag_slot_preview_at(object_id, position)
        if slot_preview != self._drag_slot_preview:
            self._drag_slot_preview = slot_preview
            self.update()
        return slot_preview is not None

    def _clear_drag_slot_preview(self) -> None:
        """Clear transient drag state without touching persisted placements."""

        if self._drag_slot_preview is None:
            return
        self._drag_slot_preview = None
        self.update()

    def _paint_drag_slot_preview(
        self,
        painter: QPainter,
        atlas_rect: QRectF,
    ) -> None:
        """Paint a translucent source footprint and valid/blocked feedback."""

        slot_preview = self._drag_slot_preview
        atlas = self._atlas
        if slot_preview is None or atlas is None:
            return
        preview_image = self._source_preview_images.get(slot_preview.object_id)
        if preview_image is None:
            return
        logical_rect = _atlas_slot_preview_rect(
            slot_preview.x,
            slot_preview.y,
            slot_preview.size,
            atlas.resolution,
            atlas_rect,
        )
        preview_rect = _slot_content_preview_rect(
            logical_rect,
            slot_preview.slot_half,
            slot_preview.slot_quadrant,
        )
        fill_color = (
            PREVIEW_DRAG_VALID_FILL_COLOR
            if slot_preview.is_valid
            else PREVIEW_DRAG_BLOCKED_FILL_COLOR
        )
        border_color = (
            PREVIEW_DRAG_VALID_BORDER_COLOR
            if slot_preview.is_valid
            else PREVIEW_DRAG_BLOCKED_BORDER_COLOR
        )

        painter.save()
        painter.setClipRect(atlas_rect)
        painter.setOpacity(PREVIEW_DRAG_IMAGE_OPACITY)
        if slot_preview.slot_half is not None:
            painter.drawImage(
                preview_rect,
                preview_image,
                QRectF(
                    0.0,
                    0.0,
                    preview_image.width() / 2.0,
                    float(preview_image.height()),
                ),
            )
        elif slot_preview.slot_quadrant is not None:
            painter.drawImage(
                preview_rect,
                preview_image,
                QRectF(
                    0.0,
                    0.0,
                    preview_image.width() / 2.0,
                    preview_image.height() / 2.0,
                ),
            )
        else:
            painter.drawImage(preview_rect, preview_image)
        painter.restore()
        clipped_rect = preview_rect.intersected(atlas_rect)
        painter.fillRect(clipped_rect, fill_color)
        border_pen = QPen(border_color, 3.0)
        if not slot_preview.is_valid:
            border_pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(border_pen)
        painter.drawRect(clipped_rect)
        if not slot_preview.is_valid:
            painter.drawLine(
                QLineF(clipped_rect.topLeft(), clipped_rect.bottomRight())
            )
            painter.drawLine(
                QLineF(clipped_rect.topRight(), clipped_rect.bottomLeft())
            )

    @staticmethod
    def _paint_quadtree_grid(
        painter: QPainter,
        atlas_rect: QRectF,
        atlas_resolution: int,
    ) -> None:
        painter.setPen(QPen(PREVIEW_GRID_COLOR, 1.0))
        smallest_cell_count = max(
            1,
            int(atlas_resolution) // ATLAS_BASE_CELL_SIZE,
        )
        cell_size = atlas_rect.width() / smallest_cell_count
        for index in range(1, smallest_cell_count):
            position = index * cell_size
            painter.drawLine(
                QLineF(
                    atlas_rect.left() + position,
                    atlas_rect.top(),
                    atlas_rect.left() + position,
                    atlas_rect.bottom(),
                )
            )
            painter.drawLine(
                QLineF(
                    atlas_rect.left(),
                    atlas_rect.top() + position,
                    atlas_rect.right(),
                    atlas_rect.top() + position,
                )
            )


# ### Atlas workspace ###
class TextureAtlasWorkspace(QWidget):
    """Create atlases and assign generated object or wall textures."""

    data_changed = Signal(object)
    object_preview_requested = Signal(str, int)
    object_preview_clear_requested = Signal()
    object_texture_resolution_changed = Signal(str, int)

    def __init__(
        self,
        asset_directory: str | Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._asset_directory = (
            _default_texture_atlas_asset_directory()
            if asset_directory is None
            else Path(asset_directory).expanduser()
        )
        self._data = TextureAtlasData()
        self._sources_by_object_id: dict[str, AtlasObjectTextureSource] = {}
        self._texture_variant_resolver: TextureVariantResolver | None = None
        self._texture_variant_selectability_resolver: (
            TextureVariantSelectabilityResolver | None
        ) = None
        self._variant_source_cache: dict[
            tuple[str, int, str, str],
            AtlasObjectTextureSource,
        ] = {}
        self._lazy_materialization_error: tuple[str, str] | None = None
        self._is_syncing = False
        self._is_handling_object_click = False
        self._is_coalescing_preview_requests = False
        self._coalesced_preview_request_key: tuple[str, int] | None = None
        self._build_ui()
        self._refresh_all()

    def get_data(self) -> TextureAtlasData:
        return copy.deepcopy(self._data)

    def set_data(self, data: TextureAtlasData | None) -> None:
        if data is not None and not isinstance(data, TextureAtlasData):
            raise TypeError("Texture atlas data has an invalid type.")
        self._data = copy.deepcopy(data or TextureAtlasData())
        self._lazy_materialization_error = None
        self._refresh_all()

    def set_object_texture_sources(
        self,
        sources: list[AtlasObjectTextureSource]
        | tuple[AtlasObjectTextureSource, ...],
        *,
        variant_resolver: TextureVariantResolver | None = None,
        selectability_resolver: (
            TextureVariantSelectabilityResolver | None
        ) = None,
    ) -> None:
        """Expose active PNGs plus exact and globally selectable variants.

        ``variant_resolver`` may resolve a PNG-only Atlas source. The optional
        ``selectability_resolver`` is stricter and prevents a packed resize
        unless the application can also assign the matching 3D model variant.
        """

        normalized_sources = list(sources)
        if not all(
            isinstance(source, AtlasObjectTextureSource)
            for source in normalized_sources
        ):
            raise TypeError(
                "Atlas texture sources must be AtlasObjectTextureSource values."
            )
        source_ids = [source.object_id for source in normalized_sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("Atlas texture source IDs must be unique.")
        selected_object_id = self._selected_object_id()
        self._sources_by_object_id = {
            source.object_id: source for source in normalized_sources
        }
        self._texture_variant_resolver = variant_resolver
        self._texture_variant_selectability_resolver = selectability_resolver
        self._variant_source_cache.clear()
        self._refresh_object_list(selected_object_id)
        self._refresh_preview()

    def materialize_missing_atlases(self) -> int:
        """Materialize the selected non-empty atlas PNG when it is missing."""

        if not self._materialize_selected_atlas_if_missing():
            if self._lazy_materialization_error is not None:
                self._refresh_preview()
            return 0
        self._refresh_all()
        self._emit_data_changed()
        return 1

    def refresh_texture_source_content(
        self,
        source_ids: tuple[str, ...] | list[str],
    ) -> bool:
        """Atomically rewrite atlases whose source pixels changed in place."""

        normalized_ids = tuple(
            dict.fromkeys(
                source_id
                for source_id in (str(value).strip() for value in source_ids)
                if source_id
            )
        )
        if not normalized_ids:
            return True
        source_id_lookup = set(normalized_ids)
        affected_atlases = tuple(
            atlas
            for atlas in self._data.atlases
            if any(
                placement.object_id in source_id_lookup
                for placement in atlas.placements
            )
        )
        if not affected_atlases:
            return True

        previous_data = self._data.clone()
        previous_lazy_error = self._lazy_materialization_error
        affected_atlas_ids = tuple(
            atlas.atlas_id for atlas in affected_atlases
        )
        png_snapshots: dict[Path, bytes | None] = {}
        try:
            png_snapshots = self._snapshot_atlas_pngs(affected_atlas_ids)
            for atlas in affected_atlases:
                self._materialize_atlas(atlas)
        except (OSError, TypeError, ValueError) as error:
            self._data = previous_data
            self._lazy_materialization_error = previous_lazy_error
            restore_failures = _restore_atlas_png_snapshots(png_snapshots)
            self._refresh_all()
            self.status_label.setText(
                "Texture content refresh blocked; existing atlas PNGs were "
                f"kept: {error}"
            )
            if restore_failures:
                self.status_label.setText(
                    self.status_label.text()
                    + f" {restore_failures} atlas PNG file(s) could not be "
                    "restored."
                )
            return False

        self._refresh_all()
        self._emit_data_changed()
        atlas_count = len(affected_atlases)
        self.status_label.setText(
            f"Refreshed source pixels in {atlas_count} texture atlas"
            f"{'es' if atlas_count != 1 else ''}."
        )
        return True

    def remove_deleted_object(self, object_id: str) -> int:
        """Purge one explicitly deleted object from every atlas.

        Ordinary missing sources remain pinned. This operation is reserved for
        the generated-object deletion signal, where retaining the old atlas PNG
        would also retain pixels belonging to an object the user deleted.
        """

        if not isinstance(object_id, str) or not object_id:
            return 0
        return self._remove_deleted_texture_sources(
            (object_id,),
            status_subject="the deleted object",
        )

    def remove_deleted_wall_texture_assignments(
        self,
        assignment_ids: tuple[str, ...] | list[str],
    ) -> int:
        """Purge fully replaced wall textures from every atlas atomically."""

        if not isinstance(assignment_ids, tuple | list):
            return 0
        try:
            source_ids = tuple(
                build_atlas_wall_texture_source_id(assignment_id)
                for assignment_id in assignment_ids
            )
        except (TypeError, ValueError):
            return 0
        return self._remove_deleted_texture_sources(
            source_ids,
            status_subject=(
                "the deleted wall texture"
                if len(source_ids) == 1
                else f"{len(source_ids)} deleted wall textures"
            ),
        )

    def _remove_deleted_texture_sources(
        self,
        source_ids: tuple[str, ...],
        *,
        status_subject: str,
    ) -> int:
        """Remove a set of unavailable sources while rebuilding each atlas once."""

        normalized_ids = tuple(
            dict.fromkeys(
                source_id
                for source_id in (str(value).strip() for value in source_ids)
                if source_id
            )
        )
        if not normalized_ids:
            return 0
        source_id_set = set(normalized_ids)
        affected_atlases = [
            atlas
            for atlas in self._data.atlases
            if any(
                placement.object_id in source_id_set
                for placement in atlas.placements
            )
        ]
        for source_id in normalized_ids:
            self._sources_by_object_id.pop(source_id, None)
        if not affected_atlases:
            self._refresh_object_list(self._selected_object_id())
            self._refresh_preview()
            return 0

        detached_png_count = 0
        for atlas in affected_atlases:
            previous_image_path = atlas.image_path
            for source_id in normalized_ids:
                self._data.unassign_object(atlas.atlas_id, source_id)
            _rebuilt, cleanup_failed = self._rebuild_or_detach_atlas_image(
                atlas,
                previous_image_path,
            )
            detached_png_count += int(cleanup_failed)

        self._refresh_all()
        self._emit_data_changed()
        atlas_count = len(affected_atlases)
        self.status_label.setText(
            f"Removed {status_subject} from {atlas_count} texture atlas"
            f"{'es' if atlas_count != 1 else ''}."
        )
        if detached_png_count:
            self.status_label.setText(
                self.status_label.text()
                + " Some obsolete PNG files could not be removed, but are no "
                "longer referenced by the project."
            )
        return atlas_count

    def refresh_regenerated_object_texture(self, object_id: str) -> int:
        """Replace pinned paths after an object's texture variants change."""

        if not isinstance(object_id, str) or not object_id:
            return 0
        affected_atlases = [
            atlas
            for atlas in self._data.atlases
            if atlas.placement_for_object(object_id) is not None
        ]
        if not affected_atlases:
            return 0

        replacement_sources: dict[int, AtlasObjectTextureSource] = {}
        for atlas in affected_atlases:
            placement = atlas.placement_for_object(object_id)
            assert placement is not None
            source = self._resolve_current_object_texture_source(
                object_id,
                placement.texture_resolution,
            )
            if source is None:
                self.status_label.setText(
                    "The regenerated texture could not replace an atlas "
                    f"placement at {placement.texture_resolution} x "
                    f"{placement.texture_resolution}."
                )
                return 0
            replacement_sources[placement.texture_resolution] = source

        if all(
            (
                (placement := atlas.placement_for_object(object_id))
                is not None
                and (
                    source := replacement_sources[placement.texture_resolution]
                ).texture_path
                == placement.texture_path
                and source.texture_resolution == placement.texture_resolution
                and source.packing_mode == placement.packing_mode
            )
            for atlas in affected_atlases
        ):
            return 0

        previous_image_paths = {
            atlas.atlas_id: atlas.image_path for atlas in affected_atlases
        }
        next_data = self._data.clone()
        for atlas in affected_atlases:
            placement = atlas.placement_for_object(object_id)
            assert placement is not None
            source = replacement_sources[placement.texture_resolution]
            next_data.assign_object(
                atlas.atlas_id,
                object_id,
                source.texture_path,
                source.texture_resolution,
                source.packing_mode,
            )
        self._data = next_data

        detached_png_count = 0
        for atlas in affected_atlases:
            updated_atlas = self._data.atlas_by_id(atlas.atlas_id)
            assert updated_atlas is not None
            _rebuilt, cleanup_failed = self._rebuild_or_detach_atlas_image(
                updated_atlas,
                previous_image_paths[atlas.atlas_id],
            )
            detached_png_count += int(cleanup_failed)

        self._refresh_all()
        self._emit_data_changed()
        atlas_count = len(affected_atlases)
        self.status_label.setText(
            f"Updated the regenerated texture in {atlas_count} texture atlas"
            f"{'es' if atlas_count != 1 else ''}."
        )
        if detached_png_count:
            self.status_label.setText(
                self.status_label.text()
                + " Some obsolete PNG files could not be removed, but are no "
                "longer referenced by the project."
            )
        return atlas_count

    def _resolve_current_object_texture_source(
        self,
        object_id: str,
        resolution: int,
    ) -> AtlasObjectTextureSource | None:
        active_source = self._sources_by_object_id.get(object_id)
        if (
            active_source is not None
            and active_source.texture_resolution == int(resolution)
        ):
            return active_source
        resolver = self._texture_variant_resolver
        if resolver is None:
            return None
        source = resolver(object_id, int(resolution))
        if (
            source is None
            or source.object_id != object_id
            or source.texture_resolution != int(resolution)
        ):
            return None
        return source

    @property
    def selected_atlas(self) -> TextureAtlasRecord | None:
        if self._data.selected_atlas_id is None:
            return None
        return self._data.atlas_by_id(self._data.selected_atlas_id)

    @property
    def selected_object_id(self) -> str | None:
        """Return the texture source currently selected in the Atlas tab."""

        return self._selected_object_id()

    def get_selected_object_texture_resolution(self) -> int | None:
        """Return the selected atlas allocation, or its active resolution."""

        object_id = self._selected_object_id()
        if object_id is None:
            return None
        atlas = self.selected_atlas
        if atlas is not None:
            placement = atlas.placement_for_object(object_id)
            if placement is not None:
                return placement.texture_resolution
        source = self._sources_by_object_id.get(object_id)
        return None if source is None else source.texture_resolution

    def can_set_object_texture_resolution(
        self,
        object_id: str,
        texture_resolution: int,
    ) -> bool:
        """Preflight one exact variant across every atlas without writing."""

        try:
            self._build_global_resolution_candidate(
                object_id,
                texture_resolution,
            )
        except (OSError, TypeError, ValueError):
            return False
        return True

    def transition_object_packing(
        self,
        object_id: str,
        candidate_sources: list[AtlasObjectTextureSource]
        | tuple[AtlasObjectTextureSource, ...],
        *,
        commit_callback: TextureResolutionCommitCallback,
    ) -> bool:
        """Atomically reconcile a generated object's texture packing."""

        normalized_id = str(object_id).strip()
        sources = list(candidate_sources)
        if not normalized_id or not sources or not callable(commit_callback):
            return False
        if any(source.object_id != normalized_id for source in sources):
            return False
        sources_by_resolution = {
            source.texture_resolution: source for source in sources
        }
        if len(sources_by_resolution) != len(sources):
            return False
        affected_atlas_ids = tuple(
            atlas.atlas_id
            for atlas in self._data.atlases
            if atlas.placement_for_object(normalized_id) is not None
        )
        if not affected_atlas_ids:
            return self._accept_texture_resolution_commit(commit_callback)

        previous_data = self._data.clone()
        previous_lazy_error = self._lazy_materialization_error
        next_data = self._data.clone()
        try:
            for atlas_id in affected_atlas_ids:
                atlas = next_data.atlas_by_id(atlas_id)
                assert atlas is not None
                placement = atlas.placement_for_object(normalized_id)
                assert placement is not None
                source = sources_by_resolution.get(placement.texture_resolution)
                if source is None:
                    source = next(
                        (
                            candidate
                            for candidate in sources
                            if candidate.packing_mode
                            in {
                                ATLAS_PACKING_MODE_SYMMETRIC_QUARTER,
                                ATLAS_PACKING_MODE_SYMMETRIC_PAIR,
                                ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR,
                            }
                            and candidate.texture_resolution * 2
                            == placement.size
                        ),
                        None,
                    )
                if source is None:
                    raise ValueError(
                        "The candidate object is missing an exact texture "
                        f"variant at {placement.texture_resolution} x "
                        f"{placement.texture_resolution}."
                    )
                next_data.assign_object(
                    atlas_id,
                    normalized_id,
                    source.texture_path,
                    source.texture_resolution,
                    source.packing_mode,
                )
            for atlas_id in affected_atlas_ids:
                atlas = next_data.atlas_by_id(atlas_id)
                assert atlas is not None
                for placement in atlas.placements:
                    source = (
                        sources_by_resolution.get(placement.texture_resolution)
                        if placement.object_id == normalized_id
                        else self._resolve_placement_source(placement)
                    )
                    if (
                        source is None
                        or source.texture_path != placement.texture_path
                        or source.packing_mode != placement.packing_mode
                    ):
                        raise ValueError(
                            "An exact texture needed to rebuild an affected "
                            "atlas is unavailable."
                        )
        except (OSError, TypeError, ValueError) as error:
            self.status_label.setText(
                "Object packing change blocked; all Atlas placements and "
                f"PNGs remain unchanged: {error}"
            )
            return False

        png_snapshots: dict[Path, bytes | None] = {}
        source_overrides = {
            (normalized_id, resolution): source
            for resolution, source in sources_by_resolution.items()
        }
        try:
            png_snapshots = self._snapshot_atlas_pngs(affected_atlas_ids)
            for atlas_id in affected_atlas_ids:
                atlas = next_data.atlas_by_id(atlas_id)
                assert atlas is not None
                self._materialize_atlas(
                    atlas,
                    source_overrides=source_overrides,
                )
        except (OSError, TypeError, ValueError) as error:
            _restore_atlas_png_snapshots(png_snapshots)
            self._lazy_materialization_error = previous_lazy_error
            self.status_label.setText(
                "Object packing change blocked; all Atlas placements and "
                f"PNGs remain unchanged: {error}"
            )
            return False

        self._data = next_data
        if not self._accept_texture_resolution_commit(commit_callback):
            self._data = previous_data
            restore_failures = _restore_atlas_png_snapshots(png_snapshots)
            self._lazy_materialization_error = previous_lazy_error
            self._refresh_all()
            self.status_label.setText(
                "Object packing change blocked; Generation rejected the "
                "candidate, so every Atlas placement and PNG was restored."
            )
            if restore_failures:
                self.status_label.setText(
                    self.status_label.text()
                    + f" {restore_failures} prior Atlas PNG file(s) could not "
                    "be restored."
                )
            return False

        try:
            self._refresh_all()
            self._emit_data_changed()
        except Exception as error:
            self._data = previous_data
            restore_failures = _restore_atlas_png_snapshots(png_snapshots)
            self._lazy_materialization_error = previous_lazy_error
            try:
                self._refresh_all()
            except Exception:
                pass
            self.status_label.setText(
                "Object packing change blocked; Atlas could not publish the "
                f"candidate, so every placement and PNG was restored: {error}"
            )
            if restore_failures:
                self.status_label.setText(
                    self.status_label.text()
                    + f" {restore_failures} prior Atlas PNG file(s) could not "
                    "be restored."
                )
            return False
        self.status_label.setText(
            f"Updated {len(affected_atlas_ids)} Atlas layout"
            f"{'s' if len(affected_atlas_ids) != 1 else ''} for the "
            "object's texture packing."
        )
        return True

    def set_object_texture_resolution(
        self,
        object_id: str,
        texture_resolution: int,
        *,
        commit_callback: TextureResolutionCommitCallback | None = None,
    ) -> bool:
        """Transactionally update one source in every atlas and globally.

        ``commit_callback`` lets the application commit another workspace in
        the same transaction. A rejection restores the exact prior layout and
        every affected atlas PNG without emitting either Atlas signal.
        """

        try:
            next_data, target_source, affected_atlas_ids = (
                self._build_global_resolution_candidate(
                    object_id,
                    texture_resolution,
                )
            )
        except (OSError, TypeError, ValueError) as error:
            self.status_label.setText(
                f"Texture size change blocked; all atlas PNGs remain "
                f"unchanged: {error}"
            )
            return False

        if not affected_atlas_ids:
            if not self._accept_texture_resolution_commit(commit_callback):
                self.status_label.setText(
                    "Texture size change blocked; the global assignment "
                    "rejected the selected resolution."
                )
                return False
            self.status_label.setText(
                f"Selected {target_source.texture_resolution} x "
                f"{target_source.texture_resolution} for "
                f"{target_source.object_name}."
            )
            if commit_callback is None:
                self.object_texture_resolution_changed.emit(
                    target_source.object_id,
                    target_source.texture_resolution,
                )
            return True

        previous_data = self._data.clone()
        previous_lazy_error = self._lazy_materialization_error
        png_snapshots: dict[Path, bytes | None] = {}
        try:
            png_snapshots = self._snapshot_atlas_pngs(affected_atlas_ids)
            for atlas_id in affected_atlas_ids:
                candidate_atlas = next_data.atlas_by_id(atlas_id)
                assert candidate_atlas is not None
                self._materialize_atlas(candidate_atlas)
        except (OSError, TypeError, ValueError) as error:
            restore_failures = _restore_atlas_png_snapshots(png_snapshots)
            self._lazy_materialization_error = previous_lazy_error
            self.status_label.setText(
                "Texture size change blocked; all placements remain "
                f"unchanged: {error}"
            )
            if restore_failures:
                self.status_label.setText(
                    self.status_label.text()
                    + f" {restore_failures} prior atlas PNG file(s) could not "
                    "be restored."
                )
            return False

        self._data = next_data
        if not self._accept_texture_resolution_commit(commit_callback):
            self._data = previous_data
            restore_failures = _restore_atlas_png_snapshots(png_snapshots)
            self._lazy_materialization_error = previous_lazy_error
            self._refresh_all()
            self.status_label.setText(
                "Texture size change blocked; the global assignment rejected "
                "the selected resolution, so every Atlas placement and PNG "
                "was restored."
            )
            if restore_failures:
                self.status_label.setText(
                    self.status_label.text()
                    + f" {restore_failures} prior atlas PNG file(s) could not "
                    "be restored."
                )
            return False

        self._refresh_all()
        self._emit_data_changed()
        atlas_count = len(affected_atlas_ids)
        self.status_label.setText(
            f"Changed {target_source.object_name} to "
            f"{target_source.texture_resolution} x "
            f"{target_source.texture_resolution} in {atlas_count} atlas"
            f"{'es' if atlas_count != 1 else ''}."
        )
        if commit_callback is None:
            self.object_texture_resolution_changed.emit(
                target_source.object_id,
                target_source.texture_resolution,
            )
        return True

    @staticmethod
    def _accept_texture_resolution_commit(
        commit_callback: TextureResolutionCommitCallback | None,
    ) -> bool:
        """Return whether an optional cross-workspace commit was accepted."""

        if commit_callback is None:
            return True
        try:
            return bool(commit_callback())
        except Exception:
            return False

    def _build_global_resolution_candidate(
        self,
        object_id: str,
        texture_resolution: int,
    ) -> tuple[TextureAtlasData, AtlasObjectTextureSource, tuple[str, ...]]:
        normalized_id = str(object_id).strip()
        if not normalized_id:
            raise ValueError("Atlas texture source ID cannot be empty.")
        try:
            normalized_resolution = int(texture_resolution)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Texture resolution is invalid.") from error
        if normalized_resolution not in TEXTURE_RESOLUTION_ORDER:
            raise ValueError("Texture resolution must be 512, 1024, or 2048.")

        active_source = self._sources_by_object_id.get(normalized_id)
        if active_source is None:
            raise ValueError("The texture source is unavailable.")
        if (
            normalized_resolution != active_source.texture_resolution
            and not active_source.supports_resolution_changes
        ):
            raise ValueError("This texture source keeps its generated resolution.")
        target_source = self._resolve_object_variant(
            normalized_id,
            normalized_resolution,
        )
        if target_source is None:
            raise ValueError(
                f"The exact {normalized_resolution} x {normalized_resolution} "
                "texture variant is unavailable."
            )
        next_data = self._data.clone()
        affected_atlas_ids: list[str] = []
        for atlas in self._data.atlases:
            if atlas.placement_for_object(normalized_id) is None:
                continue
            next_data.assign_object(
                atlas.atlas_id,
                normalized_id,
                target_source.texture_path,
                target_source.texture_resolution,
                target_source.packing_mode,
                allow_pairing=False,
            )
            affected_atlas_ids.append(atlas.atlas_id)

        for atlas_id in affected_atlas_ids:
            candidate_atlas = next_data.atlas_by_id(atlas_id)
            assert candidate_atlas is not None
            for placement in candidate_atlas.placements:
                source = (
                    target_source
                    if placement.object_id == normalized_id
                    else self._resolve_placement_source(placement)
                )
                if source is None:
                    raise ValueError(
                        "An exact texture needed to rebuild an affected atlas "
                        "is unavailable."
                    )
        selectability_resolver = self._texture_variant_selectability_resolver
        if selectability_resolver is not None:
            try:
                is_selectable = bool(
                    selectability_resolver(
                        normalized_id,
                        normalized_resolution,
                    )
                )
            except Exception:
                is_selectable = False
            if not is_selectable:
                raise ValueError(
                    f"The exact {normalized_resolution} x "
                    f"{normalized_resolution} 3D texture variant is "
                    "unavailable."
                )
        return next_data, target_source, tuple(affected_atlas_ids)

    def _snapshot_atlas_pngs(
        self,
        atlas_ids: tuple[str, ...],
    ) -> dict[Path, bytes | None]:
        snapshots: dict[Path, bytes | None] = {}
        for atlas_id in atlas_ids:
            output_path = self._resolve_owned_atlas_path(f"{atlas_id}.png")
            if output_path is None:
                raise ValueError("An atlas output path is unsafe.")
            snapshots[output_path] = (
                output_path.read_bytes() if output_path.is_file() else None
            )
        return snapshots

    def request_selected_object_preview(self) -> bool:
        """Request a 3D preview for the selected Atlas texture source."""

        object_id = self._selected_object_id()
        source = (
            None
            if object_id is None
            else self._sources_by_object_id.get(str(object_id))
        )
        resolution = self.get_selected_object_texture_resolution()
        if (
            object_id is None
            or source is None
            or not source.supports_3d_preview
            or resolution is None
        ):
            self.object_preview_clear_requested.emit()
            return False
        request_key = (object_id, resolution)
        if (
            self._is_coalescing_preview_requests
            and request_key == self._coalesced_preview_request_key
        ):
            return True
        if self._is_coalescing_preview_requests:
            self._coalesced_preview_request_key = request_key
        self.object_preview_requested.emit(object_id, resolution)
        return True

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        creation_widget = QWidget()
        creation_layout = QFormLayout(creation_widget)
        creation_layout.setContentsMargins(0, 0, 0, 0)
        self.atlas_name_edit = QLineEdit()
        self.atlas_name_edit.setObjectName("texture_atlas_name_edit")
        self.atlas_name_edit.setPlaceholderText("Atlas name")
        creation_layout.addRow("Name", self.atlas_name_edit)
        self.atlas_resolution_combo = QComboBox()
        self.atlas_resolution_combo.setObjectName("texture_atlas_resolution_combo")
        for resolution in sorted(ATLAS_RESOLUTIONS):
            self.atlas_resolution_combo.addItem(
                f"{resolution} x {resolution}",
                resolution,
            )
        creation_layout.addRow("Resolution", self.atlas_resolution_combo)
        self.create_atlas_button = QPushButton("Create atlas")
        self.create_atlas_button.setObjectName("create_texture_atlas_button")
        self.create_atlas_button.clicked.connect(self._create_atlas)
        creation_layout.addRow("", self.create_atlas_button)
        root_layout.addWidget(creation_widget)

        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        content_splitter.setChildrenCollapsible(False)
        selectors = QWidget()
        selectors_layout = QVBoxLayout(selectors)
        selectors_layout.setContentsMargins(0, 0, 0, 0)

        selectors_layout.addWidget(QLabel("Atlases"))
        self.atlas_list = QListWidget()
        self.atlas_list.setObjectName("texture_atlas_list")
        self.atlas_list.currentItemChanged.connect(
            self._handle_atlas_selection_changed
        )
        selectors_layout.addWidget(self.atlas_list, 1)
        self.remove_atlas_button = QPushButton("Delete atlas")
        self.remove_atlas_button.setObjectName("delete_texture_atlas_button")
        self.remove_atlas_button.clicked.connect(self._remove_selected_atlas)
        selectors_layout.addWidget(self.remove_atlas_button)

        selectors_layout.addWidget(QLabel("Texture sources"))
        self.object_list = TextureAtlasObjectList()
        self.object_list.setObjectName("texture_atlas_object_list")
        self.object_list.setDragEnabled(True)
        self.object_list.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.object_list.setDefaultDropAction(Qt.DropAction.CopyAction)
        self.object_list.currentItemChanged.connect(
            self._handle_object_selection_changed
        )
        self.object_list.object_clicked.connect(self._handle_object_mouse_click)
        self.object_list.object_wheeled.connect(self._handle_object_wheel)
        selectors_layout.addWidget(self.object_list, 1)
        self.delete_object_list_shortcut = QShortcut(
            QKeySequence.StandardKey.Delete,
            self.object_list,
        )
        self.delete_object_list_shortcut.setContext(
            Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        self.delete_object_list_shortcut.activated.connect(
            self.remove_selected_texture_from_atlas
        )

        assignment_buttons = QHBoxLayout()
        self.assign_object_button = QPushButton("Add selected texture")
        self.assign_object_button.setObjectName("assign_texture_atlas_object_button")
        self.assign_object_button.clicked.connect(self._assign_selected_object)
        assignment_buttons.addWidget(self.assign_object_button)
        self.unassign_object_button = QPushButton("Remove from atlas")
        self.unassign_object_button.setObjectName(
            "unassign_texture_atlas_object_button"
        )
        self.unassign_object_button.clicked.connect(
            self.remove_selected_texture_from_atlas
        )
        assignment_buttons.addWidget(self.unassign_object_button)
        selectors_layout.addLayout(assignment_buttons)

        content_splitter.addWidget(selectors)
        self.preview = TextureAtlasPreview()
        self.preview.setObjectName("texture_atlas_workspace_preview")
        self.preview.object_clicked.connect(self._handle_object_mouse_click)
        self.preview.object_wheeled.connect(self._handle_object_wheel)
        self.preview.object_dropped.connect(self._handle_object_drop)
        content_splitter.addWidget(self.preview)
        self.delete_preview_shortcut = QShortcut(
            QKeySequence.StandardKey.Delete,
            self.preview,
        )
        self.delete_preview_shortcut.setContext(
            Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        self.delete_preview_shortcut.activated.connect(
            self.remove_selected_texture_from_atlas
        )
        content_splitter.setStretchFactor(0, 1)
        content_splitter.setStretchFactor(1, 3)
        root_layout.addWidget(content_splitter, 1)

        self.status_label = QLabel()
        self.status_label.setObjectName("texture_atlas_status_label")
        self.status_label.setWordWrap(True)
        root_layout.addWidget(self.status_label)

    def _create_atlas(self) -> None:
        name = self.atlas_name_edit.text().strip()
        if not name:
            self.status_label.setText("Enter a name for the texture atlas.")
            return
        resolution = int(self.atlas_resolution_combo.currentData())
        previous_data = self._data.clone()
        try:
            atlas = self._data.create_atlas(name, resolution)
            self._data.select_atlas(atlas.atlas_id)
        except (OSError, TypeError, ValueError) as error:
            self._data = previous_data
            self._refresh_all()
            self.status_label.setText(str(error))
            return
        self.atlas_name_edit.clear()
        self._refresh_all()
        self._emit_data_changed()
        self.status_label.setText(
            f"Created {name} at {resolution} x {resolution}."
        )

    def _remove_selected_atlas(self) -> None:
        atlas = self.selected_atlas
        if atlas is None:
            return
        name = atlas.name
        image_path = atlas.image_path
        if not self._data.remove_atlas(atlas.atlas_id):
            return
        cleanup_error: OSError | None = None
        owned_image_path = self._resolve_owned_atlas_path(image_path)
        if owned_image_path is not None:
            try:
                owned_image_path.unlink(missing_ok=True)
            except OSError as error:
                cleanup_error = error
        self._refresh_all()
        self._emit_data_changed()
        if cleanup_error is None:
            self.status_label.setText(f"Deleted atlas: {name}.")
        else:
            self.status_label.setText(
                f"Deleted atlas: {name}. Its PNG could not be removed: "
                f"{cleanup_error}"
            )

    def _assign_selected_object(self) -> None:
        atlas = self.selected_atlas
        source = self._selected_object_source()
        if atlas is None or source is None:
            return
        previous_data = self._data.clone()
        try:
            placement = self._data.assign_object(
                atlas.atlas_id,
                source.object_id,
                source.texture_path,
                source.texture_resolution,
                source.packing_mode,
            )
            self._materialize_atlas(atlas)
        except (OSError, TypeError, ValueError) as error:
            self._data = previous_data
            self._refresh_all()
            self.status_label.setText(str(error))
            return
        self._refresh_atlas_list(atlas.atlas_id)
        self._refresh_object_list(source.object_id)
        self._refresh_preview()
        self._sync_controls()
        self._emit_data_changed()
        self.status_label.setText(
            f"Added {source.object_name} at ({placement.x}, {placement.y}) "
            f"using its {placement.texture_resolution} x "
            f"{placement.texture_resolution} texture."
        )

    def remove_selected_texture_from_atlas(self) -> None:
        """Remove the selected source from only the selected atlas."""

        atlas = self.selected_atlas
        object_id = self._selected_object_id()
        if atlas is None or object_id is None:
            return
        previous_image_path = atlas.image_path
        try:
            if not self._data.unassign_object(atlas.atlas_id, object_id):
                return
        except (OSError, TypeError, ValueError) as error:
            self._refresh_all()
            self.status_label.setText(str(error))
            return
        rebuilt, cleanup_failed = self._rebuild_or_detach_atlas_image(
            atlas,
            previous_image_path,
        )
        self._refresh_atlas_list(atlas.atlas_id)
        self._refresh_object_list(None)
        self._refresh_preview()
        self._sync_controls()
        self._emit_data_changed()
        self.status_label.setText("Removed the selected texture from this atlas.")
        if not rebuilt:
            if atlas.placements:
                detail = (
                    " Its PNG was detached because a remaining exact texture "
                    "source is unavailable."
                )
            else:
                detail = " Its now-empty derived PNG was removed."
            self.status_label.setText(self.status_label.text() + detail)
        if cleanup_failed:
            self.status_label.setText(
                self.status_label.text()
                + " The obsolete PNG file could not be removed, but is no "
                "longer referenced by the project."
            )

    def _handle_object_selection_changed(
        self,
        _current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        self.preview.set_selected_object_id(self._selected_object_id())
        self._sync_controls()
        if (
            not self._is_syncing
            and not self._is_handling_object_click
            and self.object_list.mouse_button_in_progress is None
        ):
            self.request_selected_object_preview()

    def _handle_object_mouse_click(
        self,
        object_id: str,
        button: object,
    ) -> None:
        if button not in {
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.RightButton,
        }:
            return
        self._is_handling_object_click = True
        try:
            self._select_object_row(object_id)
            self.status_label.setText(
                "Selected the texture. Use the mouse wheel to change the "
                "size of a packed texture."
            )
        finally:
            self._is_handling_object_click = False
        self.request_selected_object_preview()

    def _handle_object_drop(self, object_id: str, x: int, y: int) -> None:
        """Place one dragged exact source without moving other allocations."""

        atlas = self.selected_atlas
        source = self._manual_drop_source(str(object_id), atlas)
        if atlas is None or source is None:
            self.status_label.setText(
                "Texture placement blocked: the atlas or source is unavailable."
            )
            return
        previous_data = self._data.clone()
        try:
            placement = self._data.place_object_at(
                atlas.atlas_id,
                source.object_id,
                source.texture_path,
                source.texture_resolution,
                x,
                y,
                source.packing_mode,
                self.preview.drag_slot_preview.slot_half
                if self.preview.drag_slot_preview is not None
                else None,
                self.preview.drag_slot_preview.slot_quadrant
                if self.preview.drag_slot_preview is not None
                else None,
            )
            self._materialize_atlas(atlas)
        except (OSError, TypeError, ValueError) as error:
            self._data = previous_data
            self._refresh_all()
            self.status_label.setText(
                "Texture placement blocked; its previous placement and PNG "
                f"remain unchanged: {error}"
            )
            return

        self._refresh_atlas_list(atlas.atlas_id)
        self._refresh_object_list(source.object_id)
        self._refresh_preview()
        self._sync_controls()
        self._emit_data_changed()
        self.status_label.setText(
            f"Placed {source.object_name} at ({placement.x}, {placement.y}) "
            f"using its {placement.texture_resolution} x "
            f"{placement.texture_resolution} texture."
        )

    def _handle_object_wheel(
        self,
        object_id: str,
        direction: int,
    ) -> None:
        """Resize the currently selected texture for every wheel event."""

        if int(direction) == 0:
            return
        selected_object_id = self._selected_object_id()
        if selected_object_id is None:
            return
        object_id = selected_object_id
        self._is_handling_object_click = True
        self._is_coalescing_preview_requests = True
        self._coalesced_preview_request_key = None
        try:
            atlas = self.selected_atlas
            placement = (
                None if atlas is None else atlas.placement_for_object(object_id)
            )
            if placement is None:
                self.status_label.setText(
                    "Selected the texture. Add its active texture "
                    "to the selected atlas before changing its packed size."
                )
            elif self._source_has_fixed_resolution(object_id):
                self.status_label.setText(
                    "This texture source keeps its generated resolution."
                )
            else:
                self._cycle_object_texture_resolution(
                    object_id,
                    1 if int(direction) > 0 else -1,
                )
        except Exception:
            self._is_coalescing_preview_requests = False
            self._coalesced_preview_request_key = None
            raise
        finally:
            self._is_handling_object_click = False
        try:
            self.request_selected_object_preview()
        finally:
            self._is_coalescing_preview_requests = False
            self._coalesced_preview_request_key = None

    def _cycle_object_texture_resolution(
        self,
        object_id: str,
        direction: int,
    ) -> bool:
        """Transactionally cycle one source across every atlas placement."""

        atlas = self.selected_atlas
        if atlas is None:
            return False
        if self._source_has_fixed_resolution(object_id):
            self.status_label.setText(
                "Generated wall textures keep their generated resolution."
            )
            return False
        placement = atlas.placement_for_object(object_id)
        if placement is None:
            return False
        current_resolution = placement.texture_resolution
        resolution_order = (
            (512, 1024)
            if placement.packing_mode
            in {
                ATLAS_PACKING_MODE_SYMMETRIC_QUARTER,
                ATLAS_PACKING_MODE_SYMMETRIC_PAIR,
                ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR,
            }
            else TEXTURE_RESOLUTION_ORDER
        )
        current_index = resolution_order.index(current_resolution)
        target_resolution = resolution_order[
            (current_index + direction) % len(resolution_order)
        ]
        object_name = self._object_display_name(object_id)
        if self.set_object_texture_resolution(object_id, target_resolution):
            return True

        failure_message = self.status_label.text()
        if "has no space" in failure_message:
            self._report_texture_resolution_change_failure(
                atlas,
                object_name,
                current_resolution,
                target_resolution,
                ValueError(failure_message),
            )
        elif "Keeping" not in failure_message:
            self.status_label.setText(
                failure_message
                + f" Keeping {current_resolution} x {current_resolution}."
            )
        return False

    def _report_texture_resolution_change_failure(
        self,
        atlas: TextureAtlasRecord,
        object_name: str,
        current_resolution: int,
        target_resolution: int,
        error: Exception,
    ) -> None:
        """Explain a rejected Atlas resize without changing global state."""

        if "has no space" in str(error):
            self.status_label.setText(
                "Texture size change blocked by atlas capacity: "
                f"{atlas.name} has no room for {object_name} at "
                f"{target_resolution} x {target_resolution}. Keeping "
                f"{current_resolution} x {current_resolution}."
            )
            return
        self.status_label.setText(
            "Texture size change blocked; its previous placement and PNG "
            f"remain unchanged. Keeping {object_name} at "
            f"{current_resolution} x {current_resolution}: {error}"
        )

    def _handle_atlas_selection_changed(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        if self._is_syncing:
            return
        atlas_id = (
            None if current is None else str(current.data(ATLAS_ID_ROLE))
        )
        try:
            self._data.select_atlas(atlas_id)
        except ValueError:
            self._data.select_atlas(None)
        self._materialize_selected_atlas_if_missing()
        self._refresh_object_list(self._selected_object_id())
        self._refresh_preview()
        self._sync_controls()
        self._emit_data_changed()
        self.request_selected_object_preview()

    def _refresh_all(self) -> None:
        self._refresh_atlas_list(self._data.selected_atlas_id)
        self._refresh_object_list(self._selected_object_id())
        self._refresh_preview()
        self._sync_controls()

    def _refresh_atlas_list(self, selected_atlas_id: str | None) -> None:
        self._is_syncing = True
        self.atlas_list.clear()
        selected_row = -1
        for row, atlas in enumerate(self._data.atlases):
            item = QListWidgetItem(
                f"{atlas.name} · {atlas.resolution} x {atlas.resolution} · "
                f"{len(atlas.placements)} texture"
                f"{'s' if len(atlas.placements) != 1 else ''}"
            )
            item.setData(ATLAS_ID_ROLE, atlas.atlas_id)
            self.atlas_list.addItem(item)
            if atlas.atlas_id == selected_atlas_id:
                selected_row = row
        if selected_row >= 0:
            self.atlas_list.setCurrentRow(selected_row)
        self._is_syncing = False

    def _refresh_object_list(self, selected_object_id: str | None) -> None:
        was_syncing = self._is_syncing
        atlas = self.selected_atlas
        self._is_syncing = True
        try:
            self.object_list.clear()
            selected_row = -1
            row = 0
            for source in self._sources_by_object_id.values():
                placement = (
                    None
                    if atlas is None
                    else atlas.placement_for_object(source.object_id)
                )
                displayed_resolution = (
                    source.texture_resolution
                    if placement is None
                    else placement.texture_resolution
                )
                item = QListWidgetItem(
                    f"{source.object_name} · {displayed_resolution} x "
                    f"{displayed_resolution}"
                )
                item.setData(OBJECT_ID_ROLE, source.object_id)
                item.setData(OBJECT_MISSING_ROLE, False)
                item.setFlags(
                    item.flags() | Qt.ItemFlag.ItemIsDragEnabled
                )
                item.setToolTip(
                    source.texture_path
                    if placement is None
                    else (
                        f"Pinned {placement.texture_resolution} x "
                        f"{placement.texture_resolution}: "
                        f"{placement.texture_path}"
                    )
                )
                self.object_list.addItem(item)
                if source.object_id == selected_object_id:
                    selected_row = row
                row += 1

            if atlas is not None:
                for placement in atlas.placements:
                    if placement.object_id in self._sources_by_object_id:
                        continue
                    item = QListWidgetItem(f"[Missing] {placement.object_id}")
                    item.setData(OBJECT_ID_ROLE, placement.object_id)
                    item.setData(OBJECT_MISSING_ROLE, True)
                    item.setFlags(
                        item.flags() & ~Qt.ItemFlag.ItemIsDragEnabled
                    )
                    item.setToolTip(
                        f"Pinned {placement.texture_resolution} x "
                        f"{placement.texture_resolution}: "
                        f"{placement.texture_path}"
                    )
                    self.object_list.addItem(item)
                    if placement.object_id == selected_object_id:
                        selected_row = row
                    row += 1
            if selected_row >= 0:
                self.object_list.setCurrentRow(selected_row)
            elif self.object_list.count() > 0:
                self.object_list.setCurrentRow(0)
        finally:
            self._is_syncing = was_syncing
        wheel_resize_object_ids = {
            placement.object_id
            for placement in (() if atlas is None else atlas.placements)
            if not self._source_has_fixed_resolution(placement.object_id)
        }
        self.object_list.set_wheel_resize_object_ids(wheel_resize_object_ids)
        self.preview.set_wheel_resize_object_ids(wheel_resize_object_ids)
        self._sync_controls()

    def _refresh_preview(self) -> None:
        resolved_sources = self._resolve_placement_sources(self.selected_atlas)
        preview_sources = dict(self._sources_by_object_id)
        atlas = self.selected_atlas
        for placement in (() if atlas is None else atlas.placements):
            exact_source = resolved_sources.get(placement.object_id)
            if exact_source is None:
                preview_sources.pop(placement.object_id, None)
            else:
                preview_sources[placement.object_id] = exact_source
        self.preview.set_content(
            atlas,
            preview_sources,
        )
        self.preview.set_selected_object_id(self._selected_object_id())
        atlas = self.selected_atlas
        if atlas is None:
            self.status_label.setText("Create an atlas to begin packing textures.")
            return
        missing_count = sum(
            placement.object_id not in resolved_sources
            for placement in atlas.placements
        )
        if missing_count:
            self.status_label.setText(
                f"{missing_count} atlas placement"
                f"{'s' if missing_count != 1 else ''} retained without an "
                "available source texture."
            )
        elif (
            self._lazy_materialization_error is not None
            and self._lazy_materialization_error[0] == atlas.atlas_id
        ):
            self.status_label.setText(
                "The atlas layout preview is ready, but its PNG could not be "
                f"built: {self._lazy_materialization_error[1]}."
            )
        else:
            self.status_label.setText(
                f"{len(atlas.placements)} texture"
                f"{'s' if len(atlas.placements) != 1 else ''} packed into "
                f"{atlas.name}."
            )

    def _sync_controls(self) -> None:
        atlas = self.selected_atlas
        object_id = self._selected_object_id()
        source = self._selected_object_source()
        self.remove_atlas_button.setEnabled(atlas is not None)
        self.assign_object_button.setEnabled(atlas is not None and source is not None)
        self.unassign_object_button.setEnabled(
            atlas is not None
            and object_id is not None
            and atlas.placement_for_object(object_id) is not None
        )

    def _selected_object_id(self) -> str | None:
        item = self.object_list.currentItem()
        return None if item is None else str(item.data(OBJECT_ID_ROLE))

    def _selected_object_source(self) -> AtlasObjectTextureSource | None:
        object_id = self._selected_object_id()
        return (
            None
            if object_id is None
            else self._sources_by_object_id.get(object_id)
        )

    def _select_object_row(self, object_id: str) -> bool:
        normalized_id = str(object_id)
        for row in range(self.object_list.count()):
            item = self.object_list.item(row)
            if str(item.data(OBJECT_ID_ROLE)) == normalized_id:
                self.object_list.setCurrentRow(row)
                self.preview.set_selected_object_id(normalized_id)
                return True
        return False

    def _object_display_name(self, object_id: str) -> str:
        source = self._sources_by_object_id.get(object_id)
        return object_id if source is None else source.object_name

    def _source_has_fixed_resolution(self, source_id: str) -> bool:
        source = self._sources_by_object_id.get(str(source_id))
        if source is not None:
            return not source.supports_resolution_changes
        return is_atlas_wall_texture_source_id(source_id)

    def _manual_drop_source(
        self,
        object_id: str,
        atlas: TextureAtlasRecord | None,
    ) -> AtlasObjectTextureSource | None:
        if atlas is not None:
            placement = atlas.placement_for_object(object_id)
            if placement is not None:
                return self._resolve_placement_source(placement)
        return self._sources_by_object_id.get(object_id)

    def _emit_data_changed(self) -> None:
        self.data_changed.emit(self.get_data())

    def _materialize_selected_atlas_if_missing(self) -> bool:
        """Lazily rebuild one selected derived PNG without blocking project load."""

        atlas = self.selected_atlas
        self._lazy_materialization_error = None
        if atlas is None or not atlas.placements:
            return False
        if atlas.image_path:
            candidate = self._resolve_owned_atlas_path(atlas.image_path)
            if candidate is not None and candidate.is_file():
                return False
        if any(
            self._resolve_placement_source(placement) is None
            for placement in atlas.placements
        ):
            return False
        try:
            self._materialize_atlas(atlas)
        except (OSError, TypeError, ValueError) as error:
            self._lazy_materialization_error = (atlas.atlas_id, str(error))
            return False
        return True

    def _materialize_atlas(
        self,
        atlas: TextureAtlasRecord,
        *,
        source_overrides: dict[
            tuple[str, int],
            AtlasObjectTextureSource,
        ]
        | None = None,
    ) -> None:
        """Write a complete atlas before exposing its updated state."""

        self._asset_directory.mkdir(parents=True, exist_ok=True)
        relative_path = f"{atlas.atlas_id}.png"
        output_path = self._resolve_owned_atlas_path(relative_path)
        if output_path is None:
            raise ValueError(
                "Texture atlas output path escapes its asset directory."
            )

        def load_source(placement: TextureAtlasPlacement) -> np.ndarray:
            source = (
                None
                if source_overrides is None
                else source_overrides.get(
                    (placement.object_id, placement.texture_resolution)
                )
            )
            if source is None:
                source = self._resolve_placement_source(placement)
            if source is None:
                raise ValueError(
                    "The atlas cannot be updated because the exact "
                    f"{placement.texture_resolution} x "
                    f"{placement.texture_resolution} texture for source "
                    f"{placement.object_id!r} is unavailable."
                )
            if (
                source.texture_path != placement.texture_path
                or source.packing_mode != placement.packing_mode
            ):
                raise ValueError(
                    "The exact texture source does not match its Atlas placement."
                )
            rgba = source.load_texture_rgba()
            return np.ascontiguousarray(rgba[:, :, (2, 1, 0, 3)])

        write_texture_atlas_png(
            atlas,
            output_path,
            source_loader=load_source,
            project_relative_image_path=relative_path,
        )
        if (
            self._lazy_materialization_error is not None
            and self._lazy_materialization_error[0] == atlas.atlas_id
        ):
            self._lazy_materialization_error = None

    def _rebuild_or_detach_atlas_image(
        self,
        atlas: TextureAtlasRecord,
        previous_image_path: str | None,
    ) -> tuple[bool, bool]:
        """Rebuild complete output or detach every stale reference safely."""

        can_rebuild = bool(atlas.placements) and all(
            self._resolve_placement_source(placement) is not None
            for placement in atlas.placements
        )
        rebuilt = False
        if can_rebuild:
            try:
                self._materialize_atlas(atlas)
            except (OSError, TypeError, ValueError):
                pass
            else:
                rebuilt = True
        if not rebuilt:
            atlas.image_path = None

        previous_owned_path = self._resolve_owned_atlas_path(
            previous_image_path
        )
        current_owned_path = self._resolve_owned_atlas_path(atlas.image_path)
        cleanup_failed = False
        if (
            previous_owned_path is not None
            and previous_owned_path != current_owned_path
        ):
            try:
                previous_owned_path.unlink(missing_ok=True)
            except OSError:
                cleanup_failed = True
        return rebuilt, cleanup_failed

    def _resolve_placement_sources(
        self,
        atlas: TextureAtlasRecord | None,
    ) -> dict[str, AtlasObjectTextureSource]:
        if atlas is None:
            return {}
        resolved: dict[str, AtlasObjectTextureSource] = {}
        for placement in atlas.placements:
            source = self._resolve_placement_source(placement)
            if source is not None:
                resolved[placement.object_id] = source
        return resolved

    def _resolve_placement_source(
        self,
        placement: TextureAtlasPlacement,
    ) -> AtlasObjectTextureSource | None:
        active_source = self._sources_by_object_id.get(placement.object_id)
        if (
            active_source is not None
            and active_source.texture_resolution == placement.texture_resolution
            and active_source.texture_path == placement.texture_path
            and active_source.packing_mode == placement.packing_mode
        ):
            return active_source

        cache_key = (
            placement.object_id,
            placement.texture_resolution,
            placement.texture_path,
            placement.packing_mode,
        )
        cached_source = self._variant_source_cache.get(cache_key)
        if cached_source is not None:
            if cached_source.physical_texture_path.is_file():
                return cached_source
            self._variant_source_cache.pop(cache_key, None)

        resolver = self._texture_variant_resolver
        if resolver is not None:
            source = resolver(
                placement.object_id,
                placement.texture_resolution,
            )
            if source is not None:
                if (
                    source.object_id == placement.object_id
                    and source.texture_resolution
                    == placement.texture_resolution
                    and source.texture_path == placement.texture_path
                    and source.packing_mode == placement.packing_mode
                ):
                    if (
                        len(self._variant_source_cache)
                        >= MAX_VARIANT_SOURCE_CACHE_ENTRIES
                    ):
                        oldest_key = next(iter(self._variant_source_cache))
                        self._variant_source_cache.pop(oldest_key, None)
                    self._variant_source_cache[cache_key] = source
                    return source
                return None
        return None

    def _resolve_object_variant(
        self,
        object_id: str,
        texture_resolution: int,
    ) -> AtlasObjectTextureSource | None:
        """Resolve a requested variant before mutating a packed placement."""

        active_source = self._sources_by_object_id.get(object_id)
        if (
            active_source is not None
            and active_source.texture_resolution == texture_resolution
        ):
            return active_source

        for cache_key, cached_source in list(
            self._variant_source_cache.items()
        ):
            if (
                cache_key[0] == object_id
                and cache_key[1] == texture_resolution
            ):
                if cached_source.physical_texture_path.is_file():
                    return cached_source
                self._variant_source_cache.pop(cache_key, None)

        resolver = self._texture_variant_resolver
        if resolver is None:
            return None
        source = resolver(object_id, texture_resolution)
        if source is None:
            return None
        if (
            source.object_id != object_id
            or source.texture_resolution != texture_resolution
        ):
            return None
        cache_key = (
            source.object_id,
            source.texture_resolution,
            source.texture_path,
            source.packing_mode,
        )
        if len(self._variant_source_cache) >= MAX_VARIANT_SOURCE_CACHE_ENTRIES:
            oldest_key = next(iter(self._variant_source_cache))
            self._variant_source_cache.pop(oldest_key, None)
        self._variant_source_cache[cache_key] = source
        return source

    def _resolve_owned_atlas_path(
        self,
        raw_relative_path: str | None,
    ) -> Path | None:
        if not raw_relative_path:
            return None
        try:
            root = self._asset_directory.resolve()
            candidate = (root / raw_relative_path).resolve()
            candidate.relative_to(root)
        except (OSError, RuntimeError, ValueError):
            return None
        if candidate.suffix.lower() != ".png":
            return None
        return candidate


# ### Atlas preview cache helpers ###
def _build_atlas_preview_content_signature(
    atlas: TextureAtlasRecord | None,
    sources: dict[str, AtlasObjectTextureSource],
) -> tuple[object, ...]:
    """Snapshot layout state and immutable source identities for repainting."""

    atlas_signature: tuple[object, ...] | None = None
    if atlas is not None:
        atlas_signature = (
            atlas.atlas_id,
            atlas.name,
            int(atlas.resolution),
            tuple(atlas.placements),
        )
    source_signature = tuple(
        sorted(
            (str(object_id), id(source))
            for object_id, source in sources.items()
        )
    )
    return atlas_signature, source_signature


# ### Atlas PNG transaction helpers ###
def _restore_atlas_png_snapshots(
    snapshots: dict[Path, bytes | None],
) -> int:
    failure_count = 0
    for destination, payload in snapshots.items():
        try:
            if payload is None:
                destination.unlink(missing_ok=True)
            else:
                _write_bytes_atomically(destination, payload)
        except OSError:
            failure_count += 1
    return failure_count


def _write_bytes_atomically(destination: Path, payload: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


# ### Drag-and-drop helpers ###
def _build_texture_source_mime_data(object_id: str) -> QMimeData:
    encoded_id = str(object_id).encode("utf-8")
    if not encoded_id or len(encoded_id) > MAX_DRAG_SOURCE_ID_BYTES:
        raise ValueError("Atlas texture source drag ID is invalid.")
    mime_data = QMimeData()
    mime_data.setData(
        ATLAS_TEXTURE_SOURCE_MIME_TYPE,
        QByteArray(encoded_id),
    )
    return mime_data


def _read_texture_source_mime_data(mime_data: QMimeData) -> str | None:
    if not isinstance(mime_data, QMimeData):
        return None
    if not mime_data.hasFormat(ATLAS_TEXTURE_SOURCE_MIME_TYPE):
        return None
    encoded_id = bytes(mime_data.data(ATLAS_TEXTURE_SOURCE_MIME_TYPE))
    if not encoded_id or len(encoded_id) > MAX_DRAG_SOURCE_ID_BYTES:
        return None
    try:
        object_id = encoded_id.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return object_id if object_id.strip() == object_id and object_id else None


# ### Preview helpers ###
def _physical_texture_resolution(packing_mode: str, resolution: int) -> int:
    """Return the source PNG and logical Atlas slot side length."""

    multiplier = 2 if packing_mode in ATLAS_DOUBLE_SIZED_PACKING_MODES else 1
    return int(resolution) * multiplier


def _fit_image_to_square(image: Image.Image, resolution: int) -> Image.Image:
    """Aspect-fit all source pixels into a transparent square allocation."""

    rgba = image.convert("RGBA")
    scale = min(
        float(resolution) / max(1, rgba.width),
        float(resolution) / max(1, rgba.height),
    )
    fitted_size = (
        max(1, min(int(resolution), round(rgba.width * scale))),
        max(1, min(int(resolution), round(rgba.height * scale))),
    )
    if rgba.size != fitted_size:
        rgba = rgba.resize(fitted_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (int(resolution), int(resolution)), (0, 0, 0, 0))
    canvas.alpha_composite(
        rgba,
        (
            (int(resolution) - rgba.width) // 2,
            (int(resolution) - rgba.height) // 2,
        ),
    )
    return canvas


def _normalize_preview_rgba(source: np.ndarray) -> np.ndarray:
    rgba = np.asarray(source)
    if rgba.dtype != np.uint8:
        raise ValueError("Atlas preview pixels must use uint8 values.")
    if rgba.ndim != 3 or rgba.shape[2] not in {3, 4}:
        raise ValueError("Atlas preview pixels must contain RGB or RGBA data.")
    if rgba.shape[0] <= 0 or rgba.shape[1] <= 0:
        raise ValueError("Atlas preview pixels cannot be empty.")
    if rgba.shape[2] == 3:
        rgba = np.dstack(
            (rgba, np.full(rgba.shape[:2], 255, dtype=np.uint8))
        )
    return np.ascontiguousarray(rgba).copy()


def _aspect_fit_square(width: int, height: int) -> QRectF:
    available_width = max(0.0, float(width) - PREVIEW_MARGIN_PIXELS * 2.0)
    available_height = max(0.0, float(height) - PREVIEW_MARGIN_PIXELS * 2.0)
    side = min(available_width, available_height)
    return QRectF(
        (float(width) - side) / 2.0,
        (float(height) - side) / 2.0,
        side,
        side,
    )


def _placement_preview_rect(
    placement: TextureAtlasPlacement,
    atlas_resolution: int,
    atlas_rect: QRectF,
) -> QRectF:
    scale = atlas_rect.width() / float(atlas_resolution)
    return QRectF(
        atlas_rect.left() + placement.x * scale,
        atlas_rect.top() + placement.y * scale,
        placement.size * scale,
        placement.size * scale,
    )


def _placement_content_preview_rect(
    placement: TextureAtlasPlacement,
    atlas_resolution: int,
    atlas_rect: QRectF,
) -> QRectF:
    """Return one independently selectable packed-content rectangle."""

    logical_rect = _placement_preview_rect(
        placement,
        atlas_resolution,
        atlas_rect,
    )
    if placement.packing_mode == ATLAS_PACKING_MODE_FULL:
        return logical_rect
    if placement.packing_mode == ATLAS_PACKING_MODE_SYMMETRIC_QUARTER:
        return _slot_content_preview_rect(
            logical_rect,
            None,
            placement.slot_quadrant,
        )
    half_width = logical_rect.width() / 2.0
    return QRectF(
        (
            logical_rect.left() + half_width
            if placement.slot_half == ATLAS_SLOT_HALF_RIGHT
            else logical_rect.left()
        ),
        logical_rect.top(),
        half_width,
        logical_rect.height(),
    )


def _slot_content_preview_rect(
    logical_rect: QRectF,
    slot_half: str | None,
    slot_quadrant: str | None,
) -> QRectF:
    if slot_quadrant in ATLAS_SLOT_QUADRANT_ORDER:
        half_width = logical_rect.width() / 2.0
        half_height = logical_rect.height() / 2.0
        offset_x = (
            half_width
            if slot_quadrant
            in {
                ATLAS_SLOT_QUADRANT_TOP_RIGHT,
                ATLAS_SLOT_QUADRANT_BOTTOM_RIGHT,
            }
            else 0.0
        )
        offset_y = (
            half_height
            if slot_quadrant
            in {
                ATLAS_SLOT_QUADRANT_BOTTOM_LEFT,
                ATLAS_SLOT_QUADRANT_BOTTOM_RIGHT,
            }
            else 0.0
        )
        return QRectF(
            logical_rect.left() + offset_x,
            logical_rect.top() + offset_y,
            half_width,
            half_height,
        )
    if slot_half not in {ATLAS_SLOT_HALF_LEFT, ATLAS_SLOT_HALF_RIGHT}:
        return logical_rect
    half_width = logical_rect.width() / 2.0
    return QRectF(
        (
            logical_rect.left() + half_width
            if slot_half == ATLAS_SLOT_HALF_RIGHT
            else logical_rect.left()
        ),
        logical_rect.top(),
        half_width,
        logical_rect.height(),
    )


def _quarter_region_offset(
    slot_quadrant: str | None,
    content_resolution: int,
) -> tuple[int, int]:
    if slot_quadrant not in ATLAS_SLOT_QUADRANT_ORDER:
        return (0, 0)
    offset_x = (
        int(content_resolution)
        if slot_quadrant
        in {
            ATLAS_SLOT_QUADRANT_TOP_RIGHT,
            ATLAS_SLOT_QUADRANT_BOTTOM_RIGHT,
        }
        else 0
    )
    offset_y = (
        int(content_resolution)
        if slot_quadrant
        in {
            ATLAS_SLOT_QUADRANT_BOTTOM_LEFT,
            ATLAS_SLOT_QUADRANT_BOTTOM_RIGHT,
        }
        else 0
    )
    return offset_x, offset_y


def _atlas_slot_preview_rect(
    x: int,
    y: int,
    size: int,
    atlas_resolution: int,
    atlas_rect: QRectF,
) -> QRectF:
    """Map one snapped Atlas slot into widget preview coordinates."""

    scale = atlas_rect.width() / float(atlas_resolution)
    return QRectF(
        atlas_rect.left() + int(x) * scale,
        atlas_rect.top() + int(y) * scale,
        int(size) * scale,
        int(size) * scale,
    )


def _default_texture_atlas_asset_directory() -> Path:
    root = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppDataLocation
    )
    base_directory = Path(root) if root else Path.home() / ".housemaker"
    return base_directory / "texture_atlases"
