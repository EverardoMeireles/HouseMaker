# ### Imports ###
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QTextBrowser, QWidget

# ### Constants ###
TOUR_TOOLTIP_OVERLAY_WIDTH_PIXELS = 360
TOUR_TOOLTIP_OVERLAY_MINIMUM_HEIGHT_PIXELS = 96
TOUR_TOOLTIP_OVERLAY_MAXIMUM_VIEWPORT_HEIGHT_RATIO = 0.72
TOUR_TOOLTIP_OVERLAY_MARGIN_PIXELS = 18
TOUR_TOOLTIP_POSITION_LEFT = "left"
TOUR_TOOLTIP_POSITION_RIGHT = "right"
TOUR_TOOLTIP_POSITION_OPPOSITE = "opposite"
TOUR_TOOLTIP_POSITIONS = frozenset(
    (
        TOUR_TOOLTIP_POSITION_LEFT,
        TOUR_TOOLTIP_POSITION_RIGHT,
        TOUR_TOOLTIP_POSITION_OPPOSITE,
    )
)

_TOUR_TOOLTIP_BASE_STYLE = """
html, body {
    margin: 0;
    padding: 0;
    color: #f5f7fb;
    background: transparent;
    font-family: sans-serif;
    font-size: 14px;
}
.housemaker-tour-tooltip {
    box-sizing: border-box;
    padding: 14px 16px;
    border: 1px solid rgba(255, 255, 255, 0.24);
    border-radius: 8px;
    background: rgba(24, 27, 34, 0.94);
}
.housemaker-tour-tooltip img {
    max-width: 100%;
    height: auto;
}
"""


# ### Tooltip widget ###
class TourTooltipOverlay(QTextBrowser):
    """A screen-space rich-HTML tooltip anchored to a projected 3D point."""

    def __init__(self, viewport: QWidget) -> None:
        super().__init__(viewport)
        self.setObjectName("tour-floating-tooltip-overlay")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFrameShape(QTextBrowser.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet("background: transparent; border: none;")
        self.setVisible(False)

    def show_tooltip(
        self,
        *,
        anchor_screen_position: QPointF,
        tooltip_position: str,
        html_body: str,
        style: str,
    ) -> None:
        """Render and position one tooltip beside its projected hotspot."""

        viewport = self.parentWidget()
        if viewport is None or viewport.width() <= 0 or viewport.height() <= 0:
            self.hide_tooltip()
            return
        normalized_position = str(tooltip_position).strip().casefold()
        if normalized_position not in TOUR_TOOLTIP_POSITIONS:
            normalized_position = TOUR_TOOLTIP_POSITION_OPPOSITE
        normalized_html = str(html_body).strip()
        if not normalized_html:
            self.hide_tooltip()
            return
        self.setHtml(_build_tooltip_document(normalized_html, str(style)))
        width = min(
            TOUR_TOOLTIP_OVERLAY_WIDTH_PIXELS,
            max(viewport.width() - TOUR_TOOLTIP_OVERLAY_MARGIN_PIXELS * 2, 1),
        )
        self.document().setTextWidth(float(width))
        content_height = math.ceil(self.document().size().height())
        maximum_height = max(
            TOUR_TOOLTIP_OVERLAY_MINIMUM_HEIGHT_PIXELS,
            int(
                viewport.height()
                * TOUR_TOOLTIP_OVERLAY_MAXIMUM_VIEWPORT_HEIGHT_RATIO
            ),
        )
        height = min(
            max(content_height, TOUR_TOOLTIP_OVERLAY_MINIMUM_HEIGHT_PIXELS),
            maximum_height,
        )
        side = resolve_tooltip_side(
            normalized_position,
            anchor_x=float(anchor_screen_position.x()),
            viewport_width=float(viewport.width()),
        )
        x = (
            TOUR_TOOLTIP_OVERLAY_MARGIN_PIXELS
            if side == TOUR_TOOLTIP_POSITION_LEFT
            else viewport.width() - width - TOUR_TOOLTIP_OVERLAY_MARGIN_PIXELS
        )
        maximum_y = max(
            TOUR_TOOLTIP_OVERLAY_MARGIN_PIXELS,
            viewport.height() - height - TOUR_TOOLTIP_OVERLAY_MARGIN_PIXELS,
        )
        y = min(
            max(
                round(anchor_screen_position.y() - height * 0.5),
                TOUR_TOOLTIP_OVERLAY_MARGIN_PIXELS,
            ),
            maximum_y,
        )
        self.setGeometry(int(x), int(y), int(width), int(height))
        self.raise_()
        self.show()

    def hide_tooltip(self) -> None:
        """Hide the overlay without discarding the last rendered document."""

        self.hide()


# ### Helper functions ###
def resolve_tooltip_side(
    tooltip_position: str,
    *,
    anchor_x: float,
    viewport_width: float,
) -> str:
    """Resolve an explicit or opposite-side placement to left or right."""

    normalized_position = str(tooltip_position).strip().casefold()
    if normalized_position == TOUR_TOOLTIP_POSITION_LEFT:
        return TOUR_TOOLTIP_POSITION_LEFT
    if normalized_position == TOUR_TOOLTIP_POSITION_RIGHT:
        return TOUR_TOOLTIP_POSITION_RIGHT
    return (
        TOUR_TOOLTIP_POSITION_RIGHT
        if float(anchor_x) <= float(viewport_width) * 0.5
        else TOUR_TOOLTIP_POSITION_LEFT
    )


def _build_tooltip_document(html_body: str, style: str) -> str:
    """Wrap authored HTML and CSS in the editor's safe document shell."""

    authored_style = str(style).strip()
    if authored_style and "{" not in authored_style:
        authored_style = (
            ".housemaker-tour-tooltip {" + authored_style + "}"
        )
    return (
        "<!doctype html><html><head><meta charset='utf-8'><style>"
        f"{_TOUR_TOOLTIP_BASE_STYLE}\n{authored_style}"
        "</style></head><body><table width='100%' cellspacing='0' "
        "cellpadding='0'><tr><td class='housemaker-tour-tooltip'>"
        f"{html_body}"
        "</td></tr></table></body></html>"
    )
