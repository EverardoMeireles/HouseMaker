# ### Imports ###
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import (
    QByteArray,
    QBuffer,
    QIODevice,
    QLineF,
    QPointF,
    QSize,
    Qt,
    Signal,
    QTimer,
)
from PySide6.QtGui import (
    QColor,
    QIcon,
    QImage,
    QImageReader,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
    QResizeEvent,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)


# ### Constants ###
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_ATLAS_FILE_BYTES = 64 * 1024 * 1024
MAX_ATLAS_DIMENSION_PIXELS = 16_384
MAX_ATLAS_PIXEL_COUNT = 64_000_000
MAX_ATLAS_ID_LENGTH = 1_024
MAX_ATLAS_DISPLAY_NAME_LENGTH = 1_024
MAX_ATLAS_OWNER_ID_LENGTH = 2_048
THUMBNAIL_SIZE = QSize(84, 84)
THUMBNAIL_GRID_SIZE = QSize(124, 112)
THUMBNAIL_LIST_HEIGHT = 128
EMPTY_PREVIEW_TEXT = "No generated texture atlases"
UNSELECTED_PREVIEW_TEXT = "Select a texture atlas"
UV_EDGE_COLOR = QColor("#ff8a00")
UV_EDGE_UNDERLAY_COLOR = QColor(8, 10, 14, 210)
UV_VERTEX_COLOR = QColor("#fff0c2")
UV_VERTEX_OUTLINE_COLOR = QColor("#ff8a00")
UV_EDGE_WIDTH_PIXELS = 2.0
UV_EDGE_UNDERLAY_WIDTH_PIXELS = 4.0
UV_VERTEX_RADIUS_PIXELS = 2.5
UV_VERTEX_OUTLINE_WIDTH_PIXELS = 1.25
UV_OVERLAY_RESIZE_DEBOUNCE_MS = 100
EDIT_MASK_OVERLAY_COLOR = QColor(255, 104, 24, 148)


# ### Atlas data model ###
AtlasImageSource = (
    QImage
    | np.ndarray
    | bytes
    | bytearray
    | memoryview
    | str
    | Path
)
UvPoint = tuple[float, float]
UvTriangle = tuple[UvPoint, UvPoint, UvPoint]
UvEdge = tuple[UvPoint, UvPoint]


@dataclass(frozen=True)
class _UvOverlayGeometry:
    """Deduplicated UV geometry shared by every preview size."""

    edges: tuple[UvEdge, ...]
    vertices: tuple[UvPoint, ...]


@dataclass(frozen=True)
class _ScaledUvOverlayGeometry:
    """Screen-space draw batches cached for one preview size."""

    lines: tuple[QLineF, ...]
    points: QPolygonF


@dataclass(frozen=True, eq=False)
class TextureAtlasEntry:
    """One immutable atlas descriptor with an owned RGBA image."""

    atlas_id: str
    display_name: str
    image: AtlasImageSource
    owner_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "atlas_id",
            _normalize_required_text(
                self.atlas_id,
                "Texture atlas ID",
                MAX_ATLAS_ID_LENGTH,
            ),
        )
        object.__setattr__(
            self,
            "display_name",
            _normalize_required_text(
                self.display_name,
                "Texture atlas display name",
                MAX_ATLAS_DISPLAY_NAME_LENGTH,
            ),
        )
        object.__setattr__(
            self,
            "owner_id",
            _normalize_optional_text(
                self.owner_id,
                "Texture atlas owner ID",
                MAX_ATLAS_OWNER_ID_LENGTH,
            ),
        )
        object.__setattr__(self, "image", _decode_atlas_image(self.image))

    def get_image(self) -> QImage:
        """Return a detached image so callers cannot mutate the entry."""

        assert isinstance(self.image, QImage)
        return self.image.copy()


# ### Aspect-fit preview ###
class _AspectFitPreviewLabel(QLabel):
    """A label that retains its source image while the widget is resized."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._source_pixmap = QPixmap()
        self._scaled_base_pixmap = QPixmap()
        self._edit_mask_enabled = False
        self._edit_mask_image = QImage()
        self._edit_mask_overlay_image = QImage()
        self._uv_overlay_enabled = False
        self._uv_overlay_triangles: tuple[UvTriangle, ...] = ()
        self._uv_overlay_geometry = _UvOverlayGeometry((), ())
        self._scaled_uv_overlay_geometry: (
            _ScaledUvOverlayGeometry | None
        ) = None
        self._scaled_uv_overlay_size = QSize()
        self._overlay_revision = 0
        self._pending_overlay_revision: int | None = None
        self._uv_overlay_timer = QTimer(self)
        self._uv_overlay_timer.setSingleShot(True)
        self._uv_overlay_timer.setInterval(UV_OVERLAY_RESIZE_DEBOUNCE_MS)
        self._uv_overlay_timer.timeout.connect(self.flush_pending_uv_overlay)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(180, 180)
        self.setStyleSheet(
            "border: 1px solid #4b5563; "
            "background: #18181c; color: #aeb7c5;"
        )
        self.show_message(EMPTY_PREVIEW_TEXT)

    def set_atlas_image(self, image: QImage) -> None:
        self._source_pixmap = QPixmap.fromImage(image)
        self.setText("")
        self._sync_scaled_pixmap(defer_overlay=False)

    @property
    def uv_overlay_enabled(self) -> bool:
        return self._uv_overlay_enabled

    @property
    def uv_overlay_triangles(self) -> tuple[UvTriangle, ...]:
        return self._uv_overlay_triangles

    @property
    def uv_overlay_render_pending(self) -> bool:
        return self._pending_overlay_revision is not None

    @property
    def edit_mask_enabled(self) -> bool:
        return self._edit_mask_enabled

    @property
    def edit_mask_image(self) -> QImage:
        return self._edit_mask_image.copy()

    def set_edit_mask_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._edit_mask_enabled:
            return
        self._edit_mask_enabled = enabled
        self._sync_scaled_pixmap(defer_overlay=False)

    def set_edit_mask(self, mask: np.ndarray | None) -> None:
        mask_image, overlay_image = _build_edit_mask_images(mask)
        if mask_image == self._edit_mask_image:
            return
        self._edit_mask_image = mask_image
        self._edit_mask_overlay_image = overlay_image
        self._sync_scaled_pixmap(defer_overlay=False)

    def set_uv_overlay_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._uv_overlay_enabled:
            return
        self._uv_overlay_enabled = enabled
        self._sync_scaled_pixmap(defer_overlay=False)

    def set_uv_overlay_triangles(
        self,
        triangles: Sequence[Sequence[Sequence[float]]],
    ) -> None:
        normalized = _normalize_uv_triangles(triangles)
        if normalized == self._uv_overlay_triangles:
            return
        self._uv_overlay_triangles = normalized
        self._uv_overlay_geometry = _build_uv_overlay_geometry(normalized)
        self._invalidate_scaled_uv_overlay_geometry()
        self._sync_scaled_pixmap(defer_overlay=False)

    def flush_pending_uv_overlay(self) -> None:
        """Compose the latest coalesced resize overlay, if one is pending."""

        pending_revision = self._pending_overlay_revision
        self._uv_overlay_timer.stop()
        self._pending_overlay_revision = None
        if pending_revision is None or pending_revision != self._overlay_revision:
            return
        self._compose_current_uv_overlay()

    def show_message(self, message: str) -> None:
        self._cancel_pending_uv_overlay()
        self._source_pixmap = QPixmap()
        self._scaled_base_pixmap = QPixmap()
        self.setPixmap(QPixmap())
        self.setText(str(message))

    def resizeEvent(self, event: QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._sync_scaled_pixmap(defer_overlay=True)

    def _sync_scaled_pixmap(self, *, defer_overlay: bool) -> None:
        self._cancel_pending_uv_overlay()
        if self._source_pixmap.isNull():
            return
        target_size = self.contentsRect().size() - QSize(12, 12)
        if target_size.width() <= 0 or target_size.height() <= 0:
            return
        self._scaled_base_pixmap = self._source_pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(self._scaled_base_pixmap)
        if not self._should_compose_any_overlay():
            return
        if defer_overlay:
            self._pending_overlay_revision = self._overlay_revision
            self._uv_overlay_timer.start()
            return
        self._compose_current_uv_overlay()

    def _cancel_pending_uv_overlay(self) -> None:
        self._overlay_revision += 1
        self._pending_overlay_revision = None
        self._uv_overlay_timer.stop()

    def _should_compose_uv_overlay(self) -> bool:
        return (
            self._uv_overlay_enabled
            and bool(self._uv_overlay_geometry.vertices)
            and not self._scaled_base_pixmap.isNull()
        )

    def _should_compose_edit_mask(self) -> bool:
        return (
            self._edit_mask_enabled
            and not self._edit_mask_overlay_image.isNull()
            and not self._scaled_base_pixmap.isNull()
        )

    def _should_compose_any_overlay(self) -> bool:
        return (
            self._should_compose_edit_mask()
            or self._should_compose_uv_overlay()
        )

    def _compose_current_uv_overlay(self) -> None:
        if not self._should_compose_any_overlay():
            return
        composed_pixmap = self._scaled_base_pixmap.copy()
        if self._should_compose_edit_mask():
            _compose_edit_mask_overlay(
                composed_pixmap,
                self._edit_mask_overlay_image,
            )
        if self._should_compose_uv_overlay():
            scaled_geometry = self._get_scaled_uv_overlay_geometry(
                composed_pixmap.size()
            )
            _compose_uv_overlay(composed_pixmap, scaled_geometry)
        self.setPixmap(composed_pixmap)

    def _get_scaled_uv_overlay_geometry(
        self,
        size: QSize,
    ) -> _ScaledUvOverlayGeometry:
        cached = self._scaled_uv_overlay_geometry
        if cached is not None and self._scaled_uv_overlay_size == size:
            return cached
        scaled = _scale_uv_overlay_geometry(self._uv_overlay_geometry, size)
        self._scaled_uv_overlay_geometry = scaled
        self._scaled_uv_overlay_size = QSize(size)
        return scaled

    def _invalidate_scaled_uv_overlay_geometry(self) -> None:
        self._scaled_uv_overlay_geometry = None
        self._scaled_uv_overlay_size = QSize()


# ### Texture atlas view ###
class TextureAtlasView(QWidget):
    """Large atlas preview with a horizontal selectable thumbnail strip."""

    atlas_selected = Signal(object)
    atlas_activated = Signal(object)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        empty_preview_text: str = EMPTY_PREVIEW_TEXT,
        unselected_preview_text: str = UNSELECTED_PREVIEW_TEXT,
    ) -> None:
        super().__init__(parent)
        self._empty_preview_text = str(empty_preview_text)
        self._unselected_preview_text = str(unselected_preview_text)
        self._entries: list[TextureAtlasEntry] = []
        self._entry_by_id: dict[str, TextureAtlasEntry] = {}
        self._selected_atlas_id: str | None = None
        self._is_rebuilding_list = False
        self._build_ui()
        self.preview_label.show_message(self._empty_preview_text)

    @property
    def entries(self) -> tuple[TextureAtlasEntry, ...]:
        return tuple(self._entries)

    @property
    def selected_atlas_id(self) -> str | None:
        return self._selected_atlas_id

    @property
    def selected_entry(self) -> TextureAtlasEntry | None:
        if self._selected_atlas_id is None:
            return None
        return self._entry_by_id.get(self._selected_atlas_id)

    @property
    def uv_overlay_enabled(self) -> bool:
        return self.preview_label.uv_overlay_enabled

    @property
    def uv_overlay_triangles(self) -> tuple[UvTriangle, ...]:
        return self.preview_label.uv_overlay_triangles

    @property
    def uv_overlay_render_pending(self) -> bool:
        return self.preview_label.uv_overlay_render_pending

    @property
    def edit_mask_enabled(self) -> bool:
        return self.preview_label.edit_mask_enabled

    @property
    def edit_mask_image(self) -> QImage:
        return self.preview_label.edit_mask_image

    def set_edit_mask_enabled(self, enabled: bool) -> None:
        """Show or hide the non-destructive object-texture edit mask."""

        self.preview_label.set_edit_mask_enabled(enabled)

    def set_edit_mask(self, mask: np.ndarray | None) -> None:
        """Replace the editable-white atlas mask shown over the preview."""

        self.preview_label.set_edit_mask(mask)

    def set_uv_overlay_enabled(self, enabled: bool) -> None:
        """Show or hide the optional UV edge-and-vertex preview overlay."""

        self.preview_label.set_uv_overlay_enabled(enabled)

    def set_uv_overlay_triangles(
        self,
        triangles: Sequence[Sequence[Sequence[float]]],
    ) -> None:
        """Replace the normalized UV triangles used by the preview overlay."""

        self.preview_label.set_uv_overlay_triangles(triangles)

    def flush_pending_uv_overlay(self) -> None:
        """Render the terminal coalesced overlay without waiting for its timer."""

        self.preview_label.flush_pending_uv_overlay()

    def set_atlases(
        self,
        entries: list[TextureAtlasEntry] | tuple[TextureAtlasEntry, ...],
        selected_atlas_id: str | None = None,
    ) -> None:
        """Replace all entries while retaining a surviving stable selection."""

        normalized_entries = list(entries)
        if not all(
            isinstance(entry, TextureAtlasEntry) for entry in normalized_entries
        ):
            raise TypeError("Texture atlases must be TextureAtlasEntry values.")
        entry_by_id = {entry.atlas_id: entry for entry in normalized_entries}
        if len(entry_by_id) != len(normalized_entries):
            raise ValueError("Texture atlas IDs must be unique.")

        target_id = self._resolve_replacement_selection(
            entry_by_id,
            selected_atlas_id,
        )
        previous_id = self._selected_atlas_id
        self._entries = normalized_entries
        self._entry_by_id = entry_by_id
        self._rebuild_thumbnail_list(target_id)
        self._apply_selection(target_id, emit_signal=previous_id != target_id)

    def select_atlas(self, atlas_id: str | None) -> bool:
        """Select one atlas by stable ID, or clear selection with ``None``."""

        if atlas_id is None:
            target_id = None
        else:
            target_id = str(atlas_id)
            if target_id not in self._entry_by_id:
                return False
        if target_id == self._selected_atlas_id:
            return True

        signals_were_blocked = self.atlas_list.blockSignals(True)
        try:
            self._set_thumbnail_current_id(target_id)
        finally:
            self.atlas_list.blockSignals(signals_were_blocked)
        self._apply_selection(target_id, emit_signal=True)
        return True

    def clear(self) -> None:
        self.set_edit_mask(None)
        self.set_uv_overlay_triangles(())
        self.set_atlases(())

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.preview_label = _AspectFitPreviewLabel()
        self.preview_label.setObjectName("texture_atlas_preview")
        layout.addWidget(self.preview_label, 1)

        self.atlas_list = QListWidget()
        self.atlas_list.setObjectName("texture_atlas_thumbnail_list")
        self.atlas_list.setViewMode(QListView.ViewMode.IconMode)
        self.atlas_list.setFlow(QListView.Flow.LeftToRight)
        self.atlas_list.setWrapping(False)
        self.atlas_list.setMovement(QListView.Movement.Static)
        self.atlas_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.atlas_list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.atlas_list.setIconSize(THUMBNAIL_SIZE)
        self.atlas_list.setGridSize(THUMBNAIL_GRID_SIZE)
        self.atlas_list.setFixedHeight(THUMBNAIL_LIST_HEIGHT)
        self.atlas_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.atlas_list.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.atlas_list.currentItemChanged.connect(
            self._handle_current_item_changed
        )
        self.atlas_list.itemDoubleClicked.connect(
            self._handle_item_double_clicked
        )
        layout.addWidget(self.atlas_list)

    def _resolve_replacement_selection(
        self,
        entry_by_id: dict[str, TextureAtlasEntry],
        selected_atlas_id: str | None,
    ) -> str | None:
        if selected_atlas_id is not None:
            target_id = str(selected_atlas_id)
            if target_id not in entry_by_id:
                raise ValueError(f"Unknown texture atlas ID: {target_id!r}.")
            return target_id
        if self._selected_atlas_id in entry_by_id:
            return self._selected_atlas_id
        return next(iter(entry_by_id), None)

    def _rebuild_thumbnail_list(self, selected_atlas_id: str | None) -> None:
        self._is_rebuilding_list = True
        signals_were_blocked = self.atlas_list.blockSignals(True)
        try:
            self.atlas_list.clear()
            for entry in self._entries:
                item = QListWidgetItem(
                    QIcon(
                        QPixmap.fromImage(entry.get_image()).scaled(
                            THUMBNAIL_SIZE,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                    ),
                    entry.display_name,
                )
                item.setData(Qt.ItemDataRole.UserRole, entry.atlas_id)
                tooltip = entry.display_name
                if entry.owner_id is not None:
                    tooltip += f"\nOwner: {entry.owner_id}"
                item.setToolTip(tooltip)
                self.atlas_list.addItem(item)
            self._set_thumbnail_current_id(selected_atlas_id)
        finally:
            self.atlas_list.blockSignals(signals_were_blocked)
            self._is_rebuilding_list = False

    def _set_thumbnail_current_id(self, atlas_id: str | None) -> None:
        if atlas_id is None:
            self.atlas_list.setCurrentRow(-1)
            self.atlas_list.clearSelection()
            return
        for row in range(self.atlas_list.count()):
            item = self.atlas_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == atlas_id:
                self.atlas_list.setCurrentRow(row)
                return

    def _handle_current_item_changed(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        if self._is_rebuilding_list:
            return
        atlas_id = (
            None
            if current is None
            else str(current.data(Qt.ItemDataRole.UserRole))
        )
        self._apply_selection(atlas_id, emit_signal=True)

    def _handle_item_double_clicked(self, item: QListWidgetItem) -> None:
        atlas_id = str(item.data(Qt.ItemDataRole.UserRole) or "")
        entry = self._entry_by_id.get(atlas_id)
        if entry is not None:
            self.atlas_activated.emit(entry)

    def _apply_selection(
        self,
        atlas_id: str | None,
        *,
        emit_signal: bool,
    ) -> None:
        entry = None if atlas_id is None else self._entry_by_id.get(atlas_id)
        self._selected_atlas_id = None if entry is None else entry.atlas_id
        if entry is None:
            message = (
                self._empty_preview_text
                if not self._entries
                else self._unselected_preview_text
            )
            self.preview_label.setToolTip("")
            self.preview_label.show_message(message)
        else:
            self.preview_label.setToolTip(entry.display_name)
            self.preview_label.set_atlas_image(entry.get_image())
        if emit_signal:
            self.atlas_selected.emit(entry)


# ### Edit-mask overlay helpers ###
def _build_edit_mask_images(
    mask: np.ndarray | None,
) -> tuple[QImage, QImage]:
    if mask is None:
        return QImage(), QImage()
    mask_array = np.asarray(mask)
    if mask_array.dtype == np.bool_:
        mask_array = np.where(mask_array, 255, 0).astype(np.uint8)
    elif mask_array.dtype != np.uint8:
        raise ValueError("Texture edit masks must use bool or uint8 pixels.")
    if (
        mask_array.ndim != 2
        or mask_array.shape[0] <= 0
        or mask_array.shape[1] <= 0
    ):
        raise ValueError("Texture edit masks must contain a 2D pixel array.")
    _validate_image_dimensions(mask_array.shape[1], mask_array.shape[0])
    binary_mask = np.ascontiguousarray(
        np.where(mask_array > 0, 255, 0).astype(np.uint8)
    )
    mask_image = QImage(
        binary_mask.data,
        binary_mask.shape[1],
        binary_mask.shape[0],
        int(binary_mask.strides[0]),
        QImage.Format.Format_Grayscale8,
    ).copy()
    if not np.any(binary_mask):
        return mask_image, QImage()

    overlay = np.empty(binary_mask.shape + (4,), dtype=np.uint8)
    overlay[:, :, 0] = EDIT_MASK_OVERLAY_COLOR.red()
    overlay[:, :, 1] = EDIT_MASK_OVERLAY_COLOR.green()
    overlay[:, :, 2] = EDIT_MASK_OVERLAY_COLOR.blue()
    overlay[:, :, 3] = np.where(
        binary_mask > 0,
        EDIT_MASK_OVERLAY_COLOR.alpha(),
        0,
    ).astype(np.uint8)
    overlay = np.ascontiguousarray(overlay)
    overlay_image = QImage(
        overlay.data,
        overlay.shape[1],
        overlay.shape[0],
        int(overlay.strides[0]),
        QImage.Format.Format_RGBA8888,
    ).copy()
    return mask_image, overlay_image


def _compose_edit_mask_overlay(
    pixmap: QPixmap,
    overlay_image: QImage,
) -> None:
    if pixmap.isNull() or overlay_image.isNull():
        return
    scaled_overlay = QPixmap.fromImage(overlay_image).scaled(
        pixmap.size(),
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    painter = QPainter(pixmap)
    painter.drawPixmap(0, 0, scaled_overlay)
    painter.end()


# ### UV overlay helpers ###
def _normalize_uv_triangles(
    triangles: Sequence[Sequence[Sequence[float]]],
) -> tuple[UvTriangle, ...]:
    if len(triangles) == 0:
        return ()
    try:
        coordinates = np.asarray(triangles, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "UV overlay triangles must contain numeric coordinates."
        ) from error
    if coordinates.ndim != 3 or coordinates.shape[1:] != (3, 2):
        raise ValueError("UV overlay data must contain three points per triangle.")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("UV overlay coordinates must be finite.")
    return tuple(
        (
            (float(triangle[0, 0]), float(triangle[0, 1])),
            (float(triangle[1, 0]), float(triangle[1, 1])),
            (float(triangle[2, 0]), float(triangle[2, 1])),
        )
        for triangle in coordinates
    )


def _paint_uv_overlay(
    pixmap: QPixmap,
    triangles: Sequence[UvTriangle],
) -> None:
    if pixmap.isNull() or not triangles:
        return
    geometry = _build_uv_overlay_geometry(triangles)
    scaled_geometry = _scale_uv_overlay_geometry(geometry, pixmap.size())
    _compose_uv_overlay(pixmap, scaled_geometry)


def _build_uv_overlay_geometry(
    triangles: Sequence[UvTriangle],
) -> _UvOverlayGeometry:
    vertices: dict[UvPoint, None] = {}
    edges: dict[UvEdge, None] = {}
    for triangle in triangles:
        for uv in triangle:
            vertices.setdefault(uv, None)
        for first_index, second_index in ((0, 1), (1, 2), (2, 0)):
            first_uv = triangle[first_index]
            second_uv = triangle[second_index]
            edge_key = (
                (first_uv, second_uv)
                if first_uv <= second_uv
                else (second_uv, first_uv)
            )
            edges.setdefault(edge_key, None)
    return _UvOverlayGeometry(tuple(edges), tuple(vertices))


def _scale_uv_overlay_geometry(
    geometry: _UvOverlayGeometry,
    size: QSize,
) -> _ScaledUvOverlayGeometry:
    width_scale = max(size.width() - 1, 1)
    height_scale = max(size.height() - 1, 1)

    def scale_point(uv: UvPoint) -> QPointF:
        return QPointF(
            uv[0] * width_scale,
            (1.0 - uv[1]) * height_scale,
        )

    return _ScaledUvOverlayGeometry(
        lines=tuple(
            QLineF(scale_point(first_uv), scale_point(second_uv))
            for first_uv, second_uv in geometry.edges
        ),
        points=QPolygonF(
            [scale_point(uv) for uv in geometry.vertices]
        ),
    )


def _compose_uv_overlay(
    pixmap: QPixmap,
    geometry: _ScaledUvOverlayGeometry,
) -> None:
    if pixmap.isNull() or not geometry.points:
        return

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    underlay_pen = QPen(
        UV_EDGE_UNDERLAY_COLOR,
        UV_EDGE_UNDERLAY_WIDTH_PIXELS,
    )
    underlay_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    underlay_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(underlay_pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawLines(geometry.lines)

    edge_pen = QPen(UV_EDGE_COLOR, UV_EDGE_WIDTH_PIXELS)
    edge_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    edge_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(edge_pen)
    painter.drawLines(geometry.lines)

    vertex_outline_pen = QPen(
        UV_VERTEX_OUTLINE_COLOR,
        2.0
        * (UV_VERTEX_RADIUS_PIXELS + UV_VERTEX_OUTLINE_WIDTH_PIXELS),
    )
    vertex_outline_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(vertex_outline_pen)
    painter.drawPoints(geometry.points)
    vertex_fill_pen = QPen(
        UV_VERTEX_COLOR,
        2.0 * UV_VERTEX_RADIUS_PIXELS,
    )
    vertex_fill_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(vertex_fill_pen)
    painter.drawPoints(geometry.points)
    painter.end()


# ### Image decoding helpers ###
def _decode_atlas_image(source: AtlasImageSource) -> QImage:
    if isinstance(source, QImage):
        return _normalize_qimage(source)
    if isinstance(source, np.ndarray):
        return _qimage_from_array(source)
    if isinstance(source, bytes | bytearray | memoryview):
        return _qimage_from_png_bytes(bytes(source))
    if isinstance(source, str | Path):
        return _qimage_from_path(Path(source))
    raise TypeError(
        "Texture atlas images must be QImage, uint8 RGB/RGBA arrays, PNG "
        "bytes, or file paths."
    )


def _qimage_from_array(source: np.ndarray) -> QImage:
    image_array = np.asarray(source)
    if image_array.dtype != np.uint8:
        raise ValueError("Texture atlas pixel arrays must use uint8 values.")
    if (
        image_array.ndim != 3
        or image_array.shape[2] not in {3, 4}
        or image_array.shape[0] <= 0
        or image_array.shape[1] <= 0
    ):
        raise ValueError("Texture atlas arrays must contain RGB or RGBA pixels.")
    _validate_image_dimensions(image_array.shape[1], image_array.shape[0])
    contiguous = np.ascontiguousarray(image_array)
    image_format = (
        QImage.Format.Format_RGB888
        if contiguous.shape[2] == 3
        else QImage.Format.Format_RGBA8888
    )
    image = QImage(
        contiguous.data,
        contiguous.shape[1],
        contiguous.shape[0],
        contiguous.strides[0],
        image_format,
    )
    return _normalize_qimage(image.copy())


def _qimage_from_png_bytes(payload: bytes) -> QImage:
    if not payload.startswith(PNG_SIGNATURE):
        raise ValueError("Texture atlas data must be a PNG image.")
    if len(payload) > MAX_ATLAS_FILE_BYTES:
        raise ValueError("Texture atlas PNG data is too large.")

    buffer = QBuffer()
    buffer.setData(QByteArray(payload))
    if not buffer.open(QIODevice.OpenModeFlag.ReadOnly):
        raise ValueError("Texture atlas PNG data could not be opened.")
    try:
        reader = QImageReader(buffer, QByteArray(b"PNG"))
        reader.setAutoTransform(False)
        if not reader.canRead():
            raise ValueError("Texture atlas data is not a valid PNG image.")
        image_size = reader.size()
        _validate_image_dimensions(image_size.width(), image_size.height())
        image = reader.read()
        if image.isNull():
            raise ValueError("Texture atlas data is not a valid PNG image.")
    finally:
        buffer.close()
    return _normalize_qimage(image)


def _qimage_from_path(path: Path) -> QImage:
    normalized_path = path.expanduser()
    if not normalized_path.is_file():
        raise ValueError(f"Texture atlas image is missing: {normalized_path}")
    try:
        file_size = normalized_path.stat().st_size
    except OSError as error:
        raise ValueError(
            f"Texture atlas image could not be inspected: {normalized_path}"
        ) from error
    if file_size > MAX_ATLAS_FILE_BYTES:
        raise ValueError("Texture atlas PNG file is too large.")
    try:
        payload = normalized_path.read_bytes()
    except OSError as error:
        raise ValueError(
            f"Texture atlas image could not be read: {normalized_path}"
        ) from error
    return _qimage_from_png_bytes(payload)


def _normalize_qimage(image: QImage) -> QImage:
    if image.isNull():
        raise ValueError("Texture atlas image cannot be empty.")
    _validate_image_dimensions(image.width(), image.height())
    return image.convertToFormat(QImage.Format.Format_RGBA8888).copy()


def _validate_image_dimensions(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("Texture atlas image cannot be empty.")
    if width > MAX_ATLAS_DIMENSION_PIXELS or height > MAX_ATLAS_DIMENSION_PIXELS:
        raise ValueError("Texture atlas image dimensions are too large.")
    if width * height > MAX_ATLAS_PIXEL_COUNT:
        raise ValueError("Texture atlas image contains too many pixels.")


# ### Text normalization helpers ###
def _normalize_required_text(
    raw_value: object,
    label: str,
    maximum_length: int,
) -> str:
    value = str(raw_value).strip()
    if not value:
        raise ValueError(f"{label} cannot be empty.")
    if len(value) > maximum_length:
        raise ValueError(f"{label} is too long.")
    return value


def _normalize_optional_text(
    raw_value: object | None,
    label: str,
    maximum_length: int,
) -> str | None:
    if raw_value is None:
        return None
    return _normalize_required_text(raw_value, label, maximum_length)
