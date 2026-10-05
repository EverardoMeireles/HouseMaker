# ### Environment setup ###
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication, QWidget

from housemaker.tour_tooltip_overlay import (
    TOUR_TOOLTIP_OVERLAY_MARGIN_PIXELS,
    TOUR_TOOLTIP_POSITION_LEFT,
    TOUR_TOOLTIP_POSITION_RIGHT,
    TourTooltipOverlay,
    resolve_tooltip_side,
)

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])


# ### Tests ###
class TourTooltipOverlayTests(unittest.TestCase):
    def test_opposite_position_uses_the_other_half_of_the_viewport(self) -> None:
        self.assertEqual(
            resolve_tooltip_side(
                "opposite",
                anchor_x=100.0,
                viewport_width=800.0,
            ),
            TOUR_TOOLTIP_POSITION_RIGHT,
        )
        self.assertEqual(
            resolve_tooltip_side(
                "opposite",
                anchor_x=700.0,
                viewport_width=800.0,
            ),
            TOUR_TOOLTIP_POSITION_LEFT,
        )

    def test_explicit_position_does_not_depend_on_anchor(self) -> None:
        self.assertEqual(
            resolve_tooltip_side("left", anchor_x=0.0, viewport_width=800.0),
            TOUR_TOOLTIP_POSITION_LEFT,
        )
        self.assertEqual(
            resolve_tooltip_side("right", anchor_x=800.0, viewport_width=800.0),
            TOUR_TOOLTIP_POSITION_RIGHT,
        )

    def test_widget_renders_html_style_and_positions_on_requested_side(self) -> None:
        viewport = QWidget()
        self.addCleanup(viewport.close)
        viewport.resize(900, 600)
        tooltip = TourTooltipOverlay(viewport)

        tooltip.show_tooltip(
            anchor_screen_position=QPointF(120.0, 300.0),
            tooltip_position="right",
            html_body="<h1>Kitchen</h1><img src='example.png'>",
            style="h1 { color: #ff0000; }",
        )

        self.assertTrue(tooltip.isVisibleTo(viewport))
        self.assertEqual(
            tooltip.x() + tooltip.width(),
            viewport.width() - TOUR_TOOLTIP_OVERLAY_MARGIN_PIXELS,
        )
        rendered_html = tooltip.toHtml().casefold()
        self.assertIn("kitchen", rendered_html)
        self.assertIn("#ff0000", rendered_html)

    def test_style_declarations_are_applied_to_the_tooltip_container(self) -> None:
        viewport = QWidget()
        self.addCleanup(viewport.close)
        viewport.resize(800, 500)
        tooltip = TourTooltipOverlay(viewport)

        tooltip.show_tooltip(
            anchor_screen_position=QPointF(400.0, 250.0),
            tooltip_position="left",
            html_body="<p>Details</p>",
            style="color: #00ff00; padding: 24px;",
        )

        rendered_html = tooltip.toHtml().casefold()
        self.assertIn("#00ff00", rendered_html)


if __name__ == "__main__":
    unittest.main()
