"""Four-corner window footprints and full-wall-run pointer editing."""

# ### Environment setup ###
from __future__ import annotations

import math
import os
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ### Imports ###
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QImage, QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.blueprint_canvas import BlueprintCanvas
from housemaker.models import LevelData, VertexData
from housemaker.surface_geometry import add_wall_window
from housemaker.window_frame_bridge import (
    add_window_frame_bridge,
    build_window_frame_bridge_placement,
)
from housemaker.window_topology import move_linked_window_vertices

# ### Fixtures ###
_application = QApplication.instance() or QApplication([])
_application.setQuitOnLastWindowClosed(False)


def _rotate(point, angle):
    radians = math.radians(angle)
    return (
        250.0 + point[0] * math.cos(radians) - point[1] * math.sin(radians),
        250.0 + point[0] * math.sin(radians) + point[1] * math.cos(radians),
    )


@pytest.fixture
def linked_canvas():
    canvases = []

    def build(angle=0.0):
        graph = VertexData()
        for point in ((10.0, 50.0), (210.0, 50.0), (10.0, 70.0), (210.0, 70.0)):
            graph.add_vertex(*_rotate(point, angle))
        for edge in ((1, 2), (3, 4), (1, 3), (2, 4)):
            graph.add_edge(*edge)
        level = LevelData(index=2, name="Ground", vertex_data=graph)
        placement = build_window_frame_bridge_placement(
            level,
            _rotate((90.0, 50.0), angle),
            1.0,
            preferred_width_meters=1.0,
            height_meters=1.0,
            wall_margin_meters=0.025,
        )
        assert placement is not None
        add_window_frame_bridge([level], placement)
        window = add_wall_window([level], placement)
        window = replace(window, corner_vertex_ids=placement.bridge_corner_vertex_ids)
        level.windows[:] = [window]
        canvas = BlueprintCanvas()
        canvas.resize(900, 900)
        canvas.set_level_data(
            vertex_data=graph,
            rooms=level.rooms,
            image_path=None,
            doorways=level.doorways,
            windows=level.windows,
        )
        canvas.blueprint_image = QImage(500, 500, QImage.Format.Format_RGB32)
        canvas.blueprint_image.fill(Qt.GlobalColor.white)
        canvas.set_stair_context((), level)
        canvas.show()
        _application.processEvents()
        canvases.append(canvas)
        return canvas, level

    yield build
    for canvas in canvases:
        canvas.close()
        canvas.deleteLater()
    _application.processEvents()


def _bind_live_edits(canvas, level):
    """Model the shared project edit owner without building a 3D scene."""

    state = {}
    started, previews, finished = [], [], []

    def begin(edit):
        state["frame"] = canvas.window_drag_wall_frame
        started.append(edit)

    def apply(edit):
        frame = state["frame"]
        points = tuple(
            tuple(
                frame.start_point[axis]
                + ratio * (frame.end_point[axis] - frame.start_point[axis])
                for axis in (0, 1)
            )
            for ratio in (edit.bounds.start_ratio, edit.bounds.end_ratio)
        )
        window = move_linked_window_vertices(level, level.windows[0], *points)
        level.windows[0] = replace(
            window,
            bottom_ratio=edit.bounds.bottom_ratio,
            top_ratio=edit.bounds.top_ratio,
        )
        canvas.windows = level.windows

    canvas.canvas_opening_edit_started.connect(begin)
    canvas.canvas_opening_edit_preview_changed.connect(previews.append)
    canvas.canvas_opening_edit_preview_changed.connect(apply)
    canvas.canvas_opening_edit_cancelled.connect(apply)
    canvas.canvas_opening_edit_finished.connect(
        lambda edit, changed: finished.append((edit, changed))
    )
    return started, previews, finished


def _move_held_pointer(canvas, image_point):
    position = canvas._image_to_widget(*image_point)
    event = QMouseEvent(
        QEvent.Type.MouseMove,
        position,
        QPointF(canvas.mapToGlobal(position.toPoint())),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(canvas, event)


def _drag(canvas, press_image, destinations, *, cancel=False):
    press = canvas._image_to_widget(*press_image).toPoint()
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=press)
    for point in destinations:
        _move_held_pointer(canvas, point)
    if cancel:
        QTest.keyClick(canvas, Qt.Key.Key_Escape)
    else:
        QTest.mouseRelease(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=canvas._image_to_widget(*destinations[-1]).toPoint(),
        )


def _front_positions(level):
    window = level.windows[0]
    return tuple(
        (level.vertex_data.get_vertex(value).x, level.vertex_data.get_vertex(value).y)
        for value in window.corner_vertex_ids[:2]
    )


# ### Full-depth footprint tests ###
def test_committed_window_polygon_covers_all_four_vertices_and_back_pair(linked_canvas):
    canvas, level = linked_canvas()
    window = level.windows[0]
    polygon = canvas._get_window_widget_strip(window)
    assert polygon is not None
    actual = {(point.x(), point.y()) for point in polygon}
    expected = {
        (
            canvas._image_to_widget(vertex.x, vertex.y).x(),
            canvas._image_to_widget(vertex.x, vertex.y).y(),
        )
        for value in window.corner_vertex_ids
        for vertex in (level.vertex_data.get_vertex(value),)
    }
    assert actual == expected
    back_center = canvas._image_to_widget(*_rotate((90.0, 69.0), 0.0))
    hit = canvas._find_window_hit(back_center)
    assert hit is not None and hit.window_index == 0
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=back_center.toPoint())
    assert canvas.selected_window_id == window.window_id


# ### Full-run movement and resizing tests ###
@pytest.mark.parametrize("angle", (0.0, 90.0, 180.0, 270.0))
def test_dragging_linked_body_moves_four_vertices_beyond_its_original_edge(
    linked_canvas, angle
):
    canvas, level = linked_canvas(angle)
    before = _front_positions(level)
    started, previews, finished = _bind_live_edits(canvas, level)
    _drag(
        canvas,
        _rotate((90.0, 60.0), angle),
        [_rotate((110.0, 90.0), angle), _rotate((130.0, 90.0), angle)],
    )
    assert len(started) == len(finished) == 1
    assert finished[0] == (previews[-1], True)
    assert started[0].bounds.horizontal_span < 1.0
    assert level.windows[0].start_ratio == 0.0
    assert level.windows[0].end_ratio == 1.0
    delta = tuple(
        after - before_value
        for after, before_value in zip(_front_positions(level)[0], before[0])
    )
    expected = tuple(
        end - start
        for end, start in zip(
            _rotate((130.0, 50.0), angle), _rotate((90.0, 50.0), angle)
        )
    )
    assert delta == pytest.approx(expected, abs=0.5)
    segment = canvas._get_window_widget_segment(level.windows[0])
    assert segment is not None
    for actual, point in zip(segment, _front_positions(level)):
        expected_point = canvas._image_to_widget(*point)
        assert (actual.x(), actual.y()) == pytest.approx(
            (expected_point.x(), expected_point.y())
        )


@pytest.mark.parametrize("endpoint_index", (0, 1))
def test_linked_resize_moves_one_jamb_and_keeps_the_other_fixed(
    linked_canvas, endpoint_index
):
    canvas, level = linked_canvas()
    before = _front_positions(level)
    started, previews, finished = _bind_live_edits(canvas, level)
    canvas._set_selected_window_id(level.windows[0].window_id)
    target = (
        before[endpoint_index][0] + (-20.0 if endpoint_index == 0 else 20.0),
        before[endpoint_index][1],
    )
    _drag(canvas, before[endpoint_index], [target])
    assert len(started) == len(finished) == 1
    assert finished[0] == (previews[-1], True)
    assert _front_positions(level)[1 - endpoint_index] == pytest.approx(
        before[1 - endpoint_index]
    )
    assert _front_positions(level)[endpoint_index] == pytest.approx(target)
    assert (level.windows[0].start_ratio, level.windows[0].end_ratio) == (0.0, 1.0)


def test_repeated_drags_rebase_the_full_run_and_cancel_restores_corners(linked_canvas):
    canvas, level = linked_canvas()
    _bind_live_edits(canvas, level)
    _drag(canvas, _rotate((90.0, 60.0), 0.0), [_rotate((110.0, 60.0), 0.0)])
    after_first = _front_positions(level)
    _drag(canvas, _rotate((110.0, 60.0), 0.0), [_rotate((130.0, 60.0), 0.0)])
    after_second = _front_positions(level)
    assert after_second[0][0] - after_first[0][0] == pytest.approx(20.0, abs=0.5)
    _drag(
        canvas, _rotate((130.0, 60.0), 0.0), [_rotate((150.0, 60.0), 0.0)], cancel=True
    )
    assert _front_positions(level) == after_second
    assert canvas.window_drag_initial_bounds is None
    assert canvas.window_drag_wall_frame is None


@pytest.mark.parametrize("mode", ("body_left", "body_right", "start", "end"))
def test_linked_window_clamps_without_collapsing_adjacent_wall_edges(
    linked_canvas, mode
):
    canvas, level = linked_canvas()
    before = _front_positions(level)
    _bind_live_edits(canvas, level)
    canvas._set_selected_window_id(level.windows[0].window_id)
    if mode.startswith("body"):
        press = _rotate((90.0, 60.0), 0.0)
        destination = (0.0 if mode.endswith("left") else 1000.0, press[1])
    else:
        index = 0 if mode == "start" else 1
        press = before[index]
        destination = (0.0 if index == 0 else 1000.0, press[1])
    _drag(canvas, press, [destination])
    first, second = _front_positions(level)
    assert 260.0 < first[0] < second[0] < 460.0
    if mode in ("start", "body_left"):
        assert first[0] == pytest.approx(260.0, abs=1e-3)
    else:
        assert second[0] == pytest.approx(460.0, abs=1e-3)
    if mode == "start":
        assert second == before[1]
    elif mode == "end":
        assert first == before[0]
    for edge in level.vertex_data.edges:
        first_vertex = level.vertex_data.get_vertex(edge.start_vertex_id)
        second_vertex = level.vertex_data.get_vertex(edge.end_vertex_id)
        assert (
            math.dist(
                (first_vertex.x, first_vertex.y), (second_vertex.x, second_vertex.y)
            )
            > 0.0
        )


def test_legacy_four_corner_window_infers_full_depth_and_remains_movable(linked_canvas):
    canvas, level = linked_canvas()
    level.windows[0] = replace(level.windows[0], corner_vertex_ids=None)
    before = _front_positions_from_surface(level)
    canvas.windows = level.windows
    assert canvas._get_linked_window_widget_polygon(level.windows[0]) is not None
    _bind_live_edits(canvas, level)
    _drag(canvas, _rotate((90.0, 60.0), 0.0), [_rotate((110.0, 60.0), 0.0)])
    assert level.windows[0].corner_vertex_ids is not None
    assert _front_positions(level)[0][0] - before[0][0] == pytest.approx(20.0, abs=0.5)


def _front_positions_from_surface(level):
    pair = tuple(
        int(value) for value in level.windows[0].wall_surface_id.rsplit(":", 2)[1:]
    )
    return tuple(
        (level.vertex_data.get_vertex(value).x, level.vertex_data.get_vertex(value).y)
        for value in pair
    )
