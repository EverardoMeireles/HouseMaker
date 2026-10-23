"""Place normal top-level windows without exceeding a monitor's work area."""

# ### Imports ###
from __future__ import annotations

from PySide6.QtCore import QMargins, QPoint
from PySide6.QtGui import QScreen
from PySide6.QtWidgets import QWidget


# ### Window-frame helpers ###
def window_frame_margins(window: QWidget) -> QMargins:
    """Use the target display's logical-pixel decorations, not fixed offsets."""

    handle = window.windowHandle()
    if handle is not None:
        margins = handle.frameMargins()
        return QMargins(
            max(0, margins.left()),
            max(0, margins.top()),
            max(0, margins.right()),
            max(0, margins.bottom()),
        )
    frame = window.frameGeometry()
    client = window.geometry()
    return QMargins(
        max(0, client.left() - frame.left()),
        max(0, client.top() - frame.top()),
        max(0, frame.right() - client.right()),
        max(0, frame.bottom() - client.bottom()),
    )


# ### Work-area fitting ###
def fit_normal_window_to_screen(
    window: QWidget,
    screen: QScreen,
    *,
    edge_offset: int = 0,
    align_right: bool = False,
) -> None:
    """Fit client size and place the complete frame inside the screen.

    ``resize`` receives client dimensions, while ``move`` receives the framed
    top-left. All values remain in Qt's logical coordinate system, including
    negative monitor origins. Call again after a display/DPI change settles.
    Existing content minimum constraints are deliberately not altered here.
    """

    if window.isMaximized() or window.isFullScreen():
        return
    available = screen.availableGeometry()
    if not available.isValid():
        raise ValueError("The display has no usable window work area.")
    margins = window_frame_margins(window)
    client_width = min(
        window.width(), max(1, available.width() - margins.left() - margins.right())
    )
    client_height = min(
        window.height(), max(1, available.height() - margins.top() - margins.bottom())
    )
    if (client_width, client_height) != (window.width(), window.height()):
        window.resize(client_width, client_height)
    # Read back the actual size: a content minimum can reject a requested fit.
    frame_width = window.width() + margins.left() + margins.right()
    frame_height = window.height() + margins.top() + margins.bottom()
    horizontal_space = max(0, available.width() - frame_width)
    vertical_space = max(0, available.height() - frame_height)
    if align_right:
        offset = max(0, int(edge_offset))
        x = available.left() + max(0, horizontal_space - offset)
        y = available.top() + min(offset, vertical_space)
    else:
        x = available.left() + horizontal_space // 2
        y = available.top() + vertical_space // 2
    position = QPoint(x, y)
    if window.frameGeometry().topLeft() != position:
        window.move(position)
