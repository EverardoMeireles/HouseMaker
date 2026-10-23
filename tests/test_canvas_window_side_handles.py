"""Window jamb-side controls keep four linked wall corners coherent."""

# ### Environment and imports ###
from __future__ import annotations

import os
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest

from housemaker.blueprint_canvas import (
    VERTEX_RADIUS_SCREEN,
    WINDOW_WIDTH_HANDLE_RADIUS_SCREEN,
)
from housemaker.window_topology import move_linked_window_vertices
from tests import test_canvas_linked_windows_2d as linked_window_fixtures
from tests.test_canvas_linked_windows_2d import (
    _bind_live_edits,
    _drag,
    _move_held_pointer,
    _rotate,
)


# ### Fixture adapters ###
@pytest.fixture
def linked_canvas():
    yield from linked_window_fixtures.linked_canvas.__wrapped__()


# ### Geometry helpers ###
def _corners(level):
    return tuple(
        (
            level.vertex_data.get_vertex(vertex_id).x,
            level.vertex_data.get_vertex(vertex_id).y,
        )
        for vertex_id in level.windows[0].corner_vertex_ids
    )


def _jamb_midpoint(level, side):
    corners = _corners(level)
    return tuple(
        (corners[side][axis] + corners[side + 2][axis]) * 0.5 for axis in (0, 1)
    )


# ### Rendering and hit testing ###
def test_handles_are_side_midpoints_and_all_four_vertices_keep_the_same_radius(
    linked_canvas,
):
    canvas, level = linked_canvas()
    window = level.windows[0]
    canvas._set_selected_window_id(window.window_id)
    painter = Mock()
    canvas._paint_window_resize_handles(painter, window, None)
    calls = painter.drawEllipse.call_args_list
    assert len(calls) == 2
    expected = tuple(
        canvas._image_to_widget(*_jamb_midpoint(level, side)) for side in (0, 1)
    )
    corner_centers = tuple(canvas._image_to_widget(*point) for point in _corners(level))
    for call, center in zip(calls, expected):
        assert call.args == (
            center,
            WINDOW_WIDTH_HANDLE_RADIUS_SCREEN,
            WINDOW_WIDTH_HANDLE_RADIUS_SCREEN,
        )
        assert center not in corner_centers
    vertex_painter = Mock()
    canvas._paint_vertices(vertex_painter)
    for center in corner_centers:
        markers = [
            call
            for call in vertex_painter.drawEllipse.call_args_list
            if call.args[0] == center
        ]
        assert len(markers) == 1
        assert markers[0].args[1:] == (VERTEX_RADIUS_SCREEN, VERTEX_RADIUS_SCREEN)


@pytest.mark.parametrize("angle", (0.0, 90.0))
@pytest.mark.parametrize("side", (0, 1))
@pytest.mark.parametrize("depth_fraction", (0.0, 0.25, 0.5, 0.75, 1.0))
def test_entire_lateral_jamb_including_both_wall_faces_is_resizable(
    linked_canvas, angle, side, depth_fraction
):
    canvas, level = linked_canvas(angle)
    canvas._set_selected_window_id(level.windows[0].window_id)
    corners = _corners(level)
    point = tuple(
        corners[side][axis]
        + depth_fraction * (corners[side + 2][axis] - corners[side][axis])
        for axis in (0, 1)
    )
    hit = canvas._find_window_hit(canvas._image_to_widget(*point))
    assert hit is not None
    assert hit.window_index == 0
    assert hit.width_side_sign == (-1.0 if side == 0 else 1.0)


# ### Linked four-corner transforms ###
@pytest.mark.parametrize("angle", (0.0, 37.0, 90.0))
@pytest.mark.parametrize("side", (0, 1))
@pytest.mark.parametrize("distance", (-20.0, 20.0))
def test_side_resize_moves_both_corners_of_one_jamb_and_anchors_the_other(
    linked_canvas, angle, side, distance
):
    canvas, level = linked_canvas(angle)
    before = _corners(level)
    started, previews, finished = _bind_live_edits(canvas, level)
    canvas._set_selected_window_id(level.windows[0].window_id)
    press = _jamb_midpoint(level, side)
    origin, moved = _rotate((0.0, 0.0), angle), _rotate((distance, 0.0), angle)
    delta = tuple(moved[axis] - origin[axis] for axis in (0, 1))
    target = tuple(press[axis] + delta[axis] for axis in (0, 1))
    _drag(canvas, press, [target])
    assert len(started) == len(finished) == 1
    assert finished[0] == (previews[-1], True)
    after = _corners(level)
    for index in (side, side + 2):
        assert after[index] == pytest.approx(
            tuple(before[index][axis] + delta[axis] for axis in (0, 1)), abs=0.5
        )
    for index in (1 - side, 3 - side):
        assert after[index] == before[index]


def test_side_resize_cancel_restores_all_four_corners(linked_canvas):
    canvas, level = linked_canvas()
    before = _corners(level)
    _bind_live_edits(canvas, level)
    canvas._set_selected_window_id(level.windows[0].window_id)
    press = _jamb_midpoint(level, 1)
    _drag(canvas, press, [(press[0] + 20.0, press[1])], cancel=True)
    assert _corners(level) == before
    assert canvas.window_drag_current_edit is None


def test_small_opening_handles_remain_separated_and_drag_without_an_initial_jump(
    linked_canvas,
):
    canvas, level = linked_canvas()
    before = _corners(level)
    middle = (before[0][0] + before[1][0]) * 0.5
    level.windows[0] = move_linked_window_vertices(
        level,
        level.windows[0],
        (middle - 4.0, before[0][1]),
        (middle + 4.0, before[0][1]),
    )
    before = _corners(level)
    _bind_live_edits(canvas, level)
    canvas._set_selected_window_id(level.windows[0].window_id)
    handles = canvas._get_window_width_handle_widget_centers(level.windows[0])
    assert handles is not None
    assert handles[1].x() - handles[0].x() >= 36.0 - 1e-6
    press_widget = handles[1].toPoint()
    target_widget = (QPointF(press_widget) + QPointF(6.0, 0.0)).toPoint()
    press_image = canvas._widget_to_image(QPointF(press_widget))
    target_image = canvas._widget_to_image(QPointF(target_widget))
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=press_widget)
    _move_held_pointer(canvas, (target_image.x(), target_image.y()))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=target_widget)
    after = _corners(level)
    delta = target_image.x() - press_image.x()
    assert after[0] == before[0]
    assert after[2] == before[2]
    assert after[1][0] == pytest.approx(before[1][0] + delta)
    assert after[3][0] == pytest.approx(before[3][0] + delta)
