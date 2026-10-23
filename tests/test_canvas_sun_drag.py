"""Sun markers translate freely on the plan without editing wall geometry."""

# ### Environment and imports ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.blueprint_canvas import CanvasDirectionalLightProfile
from tests.test_canvas_directional_lights import _build_canvas


# ### Actual pointer events ###
@pytest.fixture
def canvas():
    widget = _build_canvas()
    widget.set_directional_light_profiles(
        (CanvasDirectionalLightProfile("sun", 80, 35),)
    )
    try:
        yield widget
    finally:
        widget.close()
        widget.deleteLater()
        QApplication.processEvents()


def _move_with_left(canvas, point):
    event = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(point),
        QPointF(canvas.mapToGlobal(point.toPoint())),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(canvas, event)


def _press_and_move(canvas, delta=None):
    if delta is None:
        delta = QPointF(45, 30)
    start = canvas._image_to_widget(80, 35)
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=start.toPoint())
    end = start + delta
    _move_with_left(canvas, end)
    return end


# ### Free plan translation and signal lifecycle ###
def test_drag_emits_preview_and_commit_preserving_selection_and_walls(canvas):
    starts, previews, finished = [], [], []
    canvas.directional_light_move_started.connect(starts.append)
    canvas.directional_light_move_preview_changed.connect(
        lambda *args: previews.append(args)
    )
    canvas.directional_light_move_finished.connect(lambda *args: finished.append(args))

    end = _press_and_move(canvas)
    assert starts == ["sun"]
    assert previews
    assert canvas.get_directional_light_profiles()[0].image_x > 80
    assert not finished
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=end.toPoint())

    assert finished == [("sun", False)]
    assert canvas.get_selected_directional_light_id() == "sun"
    assert not canvas.vertex_data.vertices
    assert not canvas.vertex_data.edges
    assert canvas._directional_light_drag is None


def test_off_center_press_preserves_pointer_offset_when_zoomed_and_panned(canvas):
    canvas.zoom_scale = 1.5
    canvas.view_offset = QPointF(35, 20)
    start = canvas._image_to_widget(80, 35) + QPointF(3, -2)
    delta = QPointF(35, -25)
    rect = canvas._image_display_rect()
    expected_x = 80 + delta.x() * canvas.blueprint_image.width() / rect.width()
    expected_y = 35 + delta.y() * canvas.blueprint_image.height() / rect.height()
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=start.toPoint())
    # Use the actual rounded press point, just as a real mouse delivers it.
    end = QPointF(start.toPoint()) + delta
    _move_with_left(canvas, end)

    profile = canvas.get_directional_light_profiles()[0]
    assert profile.image_x == pytest.approx(expected_x)
    assert profile.image_y == pytest.approx(expected_y)
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=end.toPoint())


def test_drag_can_move_beyond_the_printed_plan_without_clamping(canvas):
    start = canvas._image_to_widget(80, 35)
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=start.toPoint())
    end = QPointF(canvas._image_display_rect().left() - 10, start.y())
    _move_with_left(canvas, end)
    assert canvas.get_directional_light_profiles()[0].image_x < 0
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=end.toPoint())


@pytest.mark.parametrize("cancel", ("right_click", "escape", "hide"))
def test_cancel_restores_initial_marker_and_does_not_delete_the_sun(canvas, cancel):
    initial = canvas.get_directional_light_profiles()
    finished = []
    canvas.directional_light_move_finished.connect(lambda *args: finished.append(args))
    end = _press_and_move(canvas)
    if cancel == "right_click":
        QTest.mouseClick(canvas, Qt.MouseButton.RightButton, pos=end.toPoint())
    elif cancel == "escape":
        QTest.keyClick(canvas, Qt.Key.Key_Escape)
    else:
        canvas.hide()

    assert canvas.get_directional_light_profiles() == initial
    assert finished == [("sun", True)]
    assert canvas.get_selected_directional_light_id() == "sun"
    assert canvas._directional_light_drag is None
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=end.toPoint())
    assert not canvas.vertex_data.vertices


def test_click_only_selects_without_a_move_or_commit(canvas):
    starts, finished = [], []
    canvas.directional_light_move_started.connect(starts.append)
    canvas.directional_light_move_finished.connect(lambda *args: finished.append(args))
    QTest.mouseClick(
        canvas, Qt.MouseButton.LeftButton, pos=canvas._image_to_widget(80, 35).toPoint()
    )
    assert not starts and not finished
    assert canvas.get_selected_directional_light_id() == "sun"


def test_republishing_current_profiles_during_preview_does_not_reset_drag(canvas):
    canvas.directional_light_move_preview_changed.connect(
        lambda *_args: canvas.set_directional_light_profiles(
            canvas.get_directional_light_profiles()
        )
    )
    end = _press_and_move(canvas)
    first = canvas.get_directional_light_profiles()[0]
    _move_with_left(canvas, end + QPointF(30, 0))
    assert canvas._directional_light_drag.active
    assert canvas.get_directional_light_profiles()[0].image_x > first.image_x
    QTest.mouseRelease(
        canvas, Qt.MouseButton.LeftButton, pos=(end + QPointF(30, 0)).toPoint()
    )
