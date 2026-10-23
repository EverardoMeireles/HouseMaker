"""Detected-room fills are stable, optional and never interactive geometry."""

# ### Environment and imports ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.blueprint_canvas import CanvasRoomProfile
from tests.test_canvas_directional_lights import _build_canvas, _render_canvas


# ### Canvas helpers ###
@pytest.fixture
def canvas():
    widget = _build_canvas()
    yield widget
    widget.close()
    widget.deleteLater()
    QApplication.processEvents()


def _profiles():
    return (
        CanvasRoomProfile(
            "left-room",
            ((10, 10), (90, 10), (90, 90), (10, 90)),
            (((30, 30), (50, 30), (50, 50), (30, 50)),),
        ),
        CanvasRoomProfile("right-room", ((110, 10), (190, 10), (190, 90), (110, 90))),
    )


def _pixel(canvas, image_x, image_y):
    point = canvas._image_to_widget(image_x, image_y).toPoint()
    rendered = _render_canvas(canvas)
    return rendered.pixelColor(point)


# ### Visible fills and hole preservation ###
def test_hidden_by_default_then_distinct_translucent_room_colors_preserve_holes(canvas):
    canvas.set_room_overlays(_profiles())
    white = QColor(Qt.GlobalColor.white)
    assert _pixel(canvas, 20, 60) == white
    assert canvas.set_room_overlays_visible(True)

    left, right = _pixel(canvas, 20, 60), _pixel(canvas, 130, 60)
    assert left != white and right != white and left != right
    assert min(left.red(), left.green(), left.blue()) > 150
    assert _pixel(canvas, 40, 40) == white
    assert not canvas.set_room_overlays(_profiles())
    assert not canvas.set_room_overlays_visible(True)
    canvas.set_room_overlays_visible(False)
    assert _pixel(canvas, 20, 60) == white


def test_room_color_is_stable_after_reordering_zoom_and_pan(canvas):
    canvas.set_room_overlays(_profiles())
    canvas.set_room_overlays_visible(True)
    first = _pixel(canvas, 20, 60)
    canvas.set_room_overlays(tuple(reversed(_profiles())))
    canvas.zoom_scale = 1.3
    canvas.view_offset = QPointF(12, 8)
    assert _pixel(canvas, 20, 60) == first


def test_overlay_updates_and_clicks_do_not_create_wall_geometry(canvas):
    vertex = canvas.vertex_data.add_vertex(20, 60)
    canvas.set_room_overlays(_profiles())
    canvas.set_room_overlays_visible(True)
    before = canvas.vertex_data.clone()
    target = canvas._image_to_widget(20, 60).toPoint()
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=target)
    assert canvas.vertex_data == before
    assert canvas.selected_vertex_ids == (vertex.id,)
    canvas.set_room_overlays(())
    assert canvas.vertex_data == before


@pytest.mark.parametrize(
    "profile",
    (
        ("", ((0, 0), (1, 0), (1, 1))),
        ("room", ((0, 0), (1, 0))),
        ("room", ((0, 0), (1, 0), (1, float("nan")))),
    ),
)
def test_invalid_profiles_cannot_reach_canvas_painter(profile):
    with pytest.raises(ValueError):
        CanvasRoomProfile(*profile)
