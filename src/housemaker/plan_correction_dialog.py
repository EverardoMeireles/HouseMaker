# ### Imports ###
from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


# ### Dialog ###
class PlanCorrectionDialog(QDialog):
    """Choose how to correct plans without starting any work on dismissal."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("plan_correction_dialog")
        self.setWindowTitle("Image correction")
        self._chosen_action: str | None = None

        layout = QVBoxLayout(self)
        self.instructions_label = QLabel(
            "Use the configured correction model, or copy a prompt into ChatGPT "
            "with your plan images named L1, L2, L3, etc. Import its returned "
            "JSON and manually load the corresponding corrected images.",
            self,
        )
        self.instructions_label.setWordWrap(True)
        layout.addWidget(self.instructions_label)

        buttons_layout = QHBoxLayout()
        self.api_button = QPushButton("Use api", self)
        self.copy_prompt_button = QPushButton("Copy prompt", self)
        self.browse_json_button = QPushButton("Browse json", self)
        for button, action in (
            (self.api_button, "api"),
            (self.copy_prompt_button, "copy_prompt"),
            (self.browse_json_button, "browse_json"),
        ):
            button.setObjectName(f"plan_correction_{action}")
            button.setAutoDefault(False)
            button.clicked.connect(
                lambda _checked=False, selected=action: self._choose(selected)
            )
            buttons_layout.addWidget(button)
        layout.addLayout(buttons_layout)
        self.resize(480, 145)

    # ### Public state ###
    @property
    def chosen_action(self) -> str | None:
        return self._chosen_action

    # ### Choice handling ###
    def _choose(self, action: str) -> None:
        self._chosen_action = action
        self.accept()

    def reject(self) -> None:
        self._chosen_action = None
        super().reject()
