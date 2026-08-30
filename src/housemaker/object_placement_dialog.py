# ### Imports ###
from __future__ import annotations

import copy
from collections.abc import Sequence
from pathlib import Path

import cv2
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from housemaker.blueprint_canvas import BlueprintCanvas
from housemaker.generation_state import GeneratedObjectPlacement
from housemaker.models import LevelData

# ### Constants ###
PLACEMENT_DIALOG_MINIMUM_WIDTH = 900
PLACEMENT_DIALOG_MINIMUM_HEIGHT = 620
PLACEMENT_LEVEL_LIST_WIDTH = 220
LEVEL_INDEX_ITEM_ROLE = Qt.ItemDataRole.UserRole


# ### Read-only placement canvas ###
class ObjectPlacementCanvas(BlueprintCanvas):
    """Render one blueprint while reserving left-clicks for placement."""

    image_position_clicked = Signal(float, float)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            image_point = self._widget_to_image(event.position())
            if image_point is not None:
                self.image_position_clicked.emit(
                    float(image_point.x()),
                    float(image_point.y()),
                )
            event.accept()
            return
        if event.button() == Qt.MouseButton.RightButton:
            event.accept()
            return
        super().mousePressEvent(event)


# ### Placement dialog ###
class ObjectPlacementDialog(QDialog):
    """Choose one Canvas blueprint and one position inside its image."""

    placement_selected = Signal(object)

    def __init__(
        self,
        levels: Sequence[LevelData],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._levels_by_index = _snapshot_levels_with_usable_images(levels)
        self._selected_placement: GeneratedObjectPlacement | None = None
        self._build_ui()
        self._populate_levels()

    @property
    def selected_placement(self) -> GeneratedObjectPlacement | None:
        """Return the accepted immutable placement, if one was selected."""

        return self._selected_placement

    def select_level(self, level_index: int) -> bool:
        """Select an available level by its stable project index."""

        for row in range(self.level_list.count()):
            item = self.level_list.item(row)
            if item.data(LEVEL_INDEX_ITEM_ROLE) == level_index:
                self.level_list.setCurrentRow(row)
                return True
        return False

    def _build_ui(self) -> None:
        self.setWindowTitle("Place generated object")
        self.setModal(False)
        self.setMinimumSize(
            PLACEMENT_DIALOG_MINIMUM_WIDTH,
            PLACEMENT_DIALOG_MINIMUM_HEIGHT,
        )

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        instructions = QLabel(
            "Choose a level with a blueprint, then click its image to place the "
            "generated object."
        )
        instructions.setWordWrap(True)
        root_layout.addWidget(instructions)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(10)

        self.canvas = ObjectPlacementCanvas()
        self.canvas.setObjectName("object_placement_canvas")
        self.canvas.setCursor(Qt.CursorShape.CrossCursor)
        self.canvas.image_position_clicked.connect(
            self._handle_image_position_clicked
        )
        content_layout.addWidget(self.canvas, 1)

        level_panel = QWidget()
        level_panel.setObjectName("object_placement_level_panel")
        level_panel.setFixedWidth(PLACEMENT_LEVEL_LIST_WIDTH)
        level_layout = QVBoxLayout(level_panel)
        level_layout.setContentsMargins(0, 0, 0, 0)
        level_layout.setSpacing(6)
        level_layout.addWidget(QLabel("Levels with blueprints"))

        self.level_list = QListWidget()
        self.level_list.setObjectName("object_placement_level_list")
        self.level_list.currentItemChanged.connect(
            self._handle_level_selection_changed
        )
        level_layout.addWidget(self.level_list, 1)

        self.status_label = QLabel()
        self.status_label.setObjectName("object_placement_status_label")
        self.status_label.setWordWrap(True)
        level_layout.addWidget(self.status_label)
        content_layout.addWidget(level_panel)
        root_layout.addLayout(content_layout, 1)

    def _populate_levels(self) -> None:
        for level in self._levels_by_index.values():
            item = QListWidgetItem(level.display_name)
            item.setData(LEVEL_INDEX_ITEM_ROLE, level.index)
            self.level_list.addItem(item)

        if self.level_list.count() == 0:
            self.status_label.setText(
                "No Canvas levels with usable blueprint images are available."
            )
            return
        self.level_list.setCurrentRow(0)

    def _handle_level_selection_changed(
        self,
        current_item: QListWidgetItem | None,
        _previous_item: QListWidgetItem | None,
    ) -> None:
        if current_item is None:
            self.status_label.setText("Choose a level with a blueprint image.")
            return
        level_index = current_item.data(LEVEL_INDEX_ITEM_ROLE)
        level = self._levels_by_index.get(level_index)
        if level is None:
            self.status_label.setText("The selected level is unavailable.")
            return

        self.canvas.set_level_data(
            vertex_data=level.vertex_data,
            rooms=level.rooms,
            image_path=level.image_path,
            floor_contour_vertex_ids=level.floor_contour_vertex_ids,
            doorways=level.doorways,
        )
        if self.canvas.blueprint_image is None:
            self.status_label.setText(
                "This level has no available blueprint image."
            )
            return
        self.status_label.setText("Click anywhere inside the blueprint image.")

    def _handle_image_position_clicked(
        self,
        image_x: float,
        image_y: float,
    ) -> None:
        current_item = self.level_list.currentItem()
        if current_item is None:
            return
        level_index = current_item.data(LEVEL_INDEX_ITEM_ROLE)
        if level_index not in self._levels_by_index:
            return

        placement = GeneratedObjectPlacement(
            level_index=level_index,
            image_x=image_x,
            image_y=image_y,
        )
        self._selected_placement = placement
        self.placement_selected.emit(placement)
        self.accept()


# ### Level helpers ###
def _snapshot_levels_with_usable_images(
    levels: Sequence[LevelData],
) -> dict[int, LevelData]:
    """Copy exactly the levels whose associated image can be opened."""

    available_levels: dict[int, LevelData] = {}
    for level in levels:
        if not isinstance(level, LevelData):
            raise TypeError("Object placement levels must be LevelData values.")
        if not _has_usable_blueprint_image(level):
            continue
        if level.index in available_levels:
            raise ValueError(
                f"Duplicate Canvas level index with an image: {level.index}."
            )
        available_levels[level.index] = copy.deepcopy(level)
    return available_levels


def _has_usable_blueprint_image(level: LevelData) -> bool:
    """Return whether the level points at a readable, non-empty image."""

    image_path = level.image_path
    if not isinstance(image_path, str) or not image_path.strip():
        return False
    resolved_path = Path(image_path).expanduser()
    if not resolved_path.is_file():
        return False
    image = cv2.imread(str(resolved_path), cv2.IMREAD_COLOR)
    return image is not None and image.size > 0
