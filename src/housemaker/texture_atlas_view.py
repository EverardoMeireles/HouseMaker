# ### Imports ###
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import QByteArray, QBuffer, QIODevice, QSize, Qt, Signal
from PySide6.QtGui import QIcon, QImage, QImageReader, QPixmap, QResizeEvent
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
        self._sync_scaled_pixmap()

    def show_message(self, message: str) -> None:
        self._source_pixmap = QPixmap()
        self.setPixmap(QPixmap())
        self.setText(str(message))

    def resizeEvent(self, event: QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._sync_scaled_pixmap()

    def _sync_scaled_pixmap(self) -> None:
        if self._source_pixmap.isNull():
            return
        target_size = self.contentsRect().size() - QSize(12, 12)
        if target_size.width() <= 0 or target_size.height() <= 0:
            return
        self.setPixmap(
            self._source_pixmap.scaled(
                target_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


# ### Texture atlas view ###
class TextureAtlasView(QWidget):
    """Large atlas preview with a horizontal selectable thumbnail strip."""

    atlas_selected = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._entries: list[TextureAtlasEntry] = []
        self._entry_by_id: dict[str, TextureAtlasEntry] = {}
        self._selected_atlas_id: str | None = None
        self._is_rebuilding_list = False
        self._build_ui()

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
                EMPTY_PREVIEW_TEXT
                if not self._entries
                else UNSELECTED_PREVIEW_TEXT
            )
            self.preview_label.setToolTip("")
            self.preview_label.show_message(message)
        else:
            self.preview_label.setToolTip(entry.display_name)
            self.preview_label.set_atlas_image(entry.get_image())
        if emit_signal:
            self.atlas_selected.emit(entry)


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
