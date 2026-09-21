# ### Imports ###
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from housemaker.surface_texture_state import (
    SURFACE_TILING_MODE_EDGE_VARIANTS,
    SURFACE_TILING_MODE_WHOLE_REPEATS,
)

# ### Constants ###
PREVIEW_EDGE_PIXELS = 360


# ### Preview helpers ###
def _build_preview_label(png_bytes: bytes) -> QLabel:
    """Build one bounded image label from an in-memory PNG."""

    pixmap = QPixmap()
    if not pixmap.loadFromData(bytes(png_bytes), "PNG"):
        raise ValueError("The texture tiling preview is not a valid PNG image.")
    label = QLabel()
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setMinimumSize(PREVIEW_EDGE_PIXELS, PREVIEW_EDGE_PIXELS)
    label.setSizePolicy(
        QSizePolicy.Policy.Expanding,
        QSizePolicy.Policy.Expanding,
    )
    label.setPixmap(
        pixmap.scaled(
            PREVIEW_EDGE_PIXELS,
            PREVIEW_EDGE_PIXELS,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    )
    return label


def _build_labeled_preview(title: str, png_bytes: bytes) -> QWidget:
    """Wrap one tiled image in a compact labeled column."""

    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    title_label = QLabel(title)
    title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(title_label)
    layout.addWidget(_build_preview_label(png_bytes), 1)
    return container


# ### Tiling preview dialog ###
class SurfaceTextureTilingPreviewDialog(QDialog):
    """Compare a repeating source before committing its repaired revision."""

    def __init__(
        self,
        before_preview_png: bytes,
        after_preview_png: bytes,
        *,
        before_seam_score: float | None = None,
        after_seam_score: float | None = None,
        method: str = SURFACE_TILING_MODE_WHOLE_REPEATS,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("surface_texture_tiling_preview_dialog")
        self.setWindowTitle(
            "Fix tiling 2"
            if method == SURFACE_TILING_MODE_EDGE_VARIANTS
            else "Fix tiling"
        )
        self.setModal(True)

        root_layout = QVBoxLayout(self)
        if method == SURFACE_TILING_MODE_EDGE_VARIANTS:
            explanation_text = (
                "The previews repeat the texture 3 x 3. Four edge-compatible "
                "rotated variants share the existing Atlas slot and are chosen "
                "across the surface. Each variant has half the original linear "
                "pixel resolution. The preview shows how their joins look."
            )
        else:
            explanation_text = (
                "The previews repeat the texture 3 x 3. Whole repetitions are "
                "rotated to break up repetition without changing texture pixels "
                "or Atlas size. Existing seams may still remain."
            )
        explanation = QLabel(explanation_text)
        explanation.setWordWrap(True)
        root_layout.addWidget(explanation)

        preview_layout = QHBoxLayout()
        preview_layout.addWidget(
            _build_labeled_preview("Before", before_preview_png),
            1,
        )
        preview_layout.addWidget(
            _build_labeled_preview("After", after_preview_png),
            1,
        )
        root_layout.addLayout(preview_layout, 1)

        if before_seam_score is not None and after_seam_score is not None:
            score_label = QLabel(
                "Edge mismatch: "
                f"{float(before_seam_score):.4f} -> "
                f"{float(after_seam_score):.4f} (lower is better)"
            )
            score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            root_layout.addWidget(score_label)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.Cancel
        )
        apply_button = button_box.button(
            QDialogButtonBox.StandardButton.Apply
        )
        apply_button.clicked.connect(self.accept)
        button_box.rejected.connect(self.reject)
        root_layout.addWidget(button_box)

        self.resize(820, 520)


# ### Public exports ###
__all__ = ["SurfaceTextureTilingPreviewDialog"]
