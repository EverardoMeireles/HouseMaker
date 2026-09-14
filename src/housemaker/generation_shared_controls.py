# ### Imports ###
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QWidget,
)

from housemaker.generation_views import VideoInpaintView


# ### Shared control model ###
@dataclass(frozen=True)
class GenerationSharedControls:
    """The single widget set consumed by both generation pipelines."""

    video_view: VideoInpaintView
    seekbar: QSlider
    load_video_button: QPushButton
    paint_mask_button: QRadioButton
    erase_mask_button: QRadioButton
    mask_mode_button_group: QButtonGroup
    mask_mode_control: QWidget
    brush_size_spinbox: QSpinBox
    clear_mask_button: QPushButton
    pbr_map_control: QWidget
    pbr_map_checkboxes: dict[str, QCheckBox]
    ai_prompt_edit: QLineEdit
    cancel_button: QPushButton
