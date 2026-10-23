"""Real Canvas pointer handlers must not trap first-person focus re-entry."""

# ### Environment setup ###
from __future__ import annotations

import os
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ### Imports ###
import numpy as np
import pytest
import trimesh
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QFocusEvent, QMouseEvent, QVector3D
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton

from housemaker.canvas_openings import (
    CanvasOpeningBounds,
    CanvasOpeningReference,
    CanvasOpeningTarget,
)
from housemaker.glb import GeneratedModel
from housemaker.surface_geometry import FixedSurface
from housemaker.viewer import GlbViewerWidget

# ### Application and scene fixtures ###
_application = QApplication.instance() or QApplication([])
_application.setQuitOnLastWindowClosed(False)


@pytest.fixture
def viewer():
    widget = GlbViewerWidget(window_editing_enabled=True)
    mesh = trimesh.creation.box(extents=(4.0, 0.2, 3.0))
    mesh.apply_translation((2.0, 0.0, 1.5))
    widget.set_model(
        GeneratedModel(mesh=mesh, scene=trimesh.Scene(mesh), glb_bytes=b"")
    )
    widget.view.resize(640, 480)
    yield widget
    widget.exit_first_person_mode()
    QTest.keyRelease(widget.view, Qt.Key.Key_Control)
    widget.close()
    widget.deleteLater()
    _application.processEvents()


def _mouse(
    kind, *, button=Qt.MouseButton.LeftButton, modifiers=Qt.KeyboardModifier.NoModifier
):
    point = QPointF(8.0, 8.0)
    return QMouseEvent(
        kind,
        point,
        point,
        button,
        Qt.MouseButton.NoButton if kind == QEvent.Type.MouseButtonRelease else button,
        modifiers,
    )


def _left_click(view, *, modifiers=Qt.KeyboardModifier.NoModifier):
    view.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, modifiers=modifiers))
    view.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, modifiers=modifiers))


def _right_release(view):
    view.mousePressEvent(
        _mouse(QEvent.Type.MouseButtonPress, button=Qt.MouseButton.RightButton)
    )


def _lose_focus_to_control(view, *, separate_window=False):
    control = QPushButton(None if separate_window else view.parentWidget())
    view.focusOutEvent(QFocusEvent(QEvent.Type.FocusOut))
    QTest.mouseClick(control, Qt.MouseButton.LeftButton)
    return control


def _select_opening(widget):
    mesh = trimesh.Trimesh(
        vertices=[(0, 0, 0), (4, 0, 0), (4, 0, 3), (0, 0, 3)],
        faces=[(0, 1, 2), (0, 2, 3)],
        process=False,
    )
    wall = FixedSurface(
        surface_id="level:0/wall:1:2",
        surface_type="wall",
        level_index=0,
        room_index=None,
        mesh=mesh,
        area_square_meters=12.0,
        wall_key="1:2",
        wall_start_world=(0, 0, 0),
        wall_end_world=(4, 0, 0),
        wall_height_meters=3.0,
    )
    target = CanvasOpeningTarget(
        reference=CanvasOpeningReference("window", 0, 0, "opening-a"),
        wall_surface_id=wall.surface_id,
        plane_start_world=(0, 0, 0),
        wall_tangent_world=(1, 0, 0),
        wall_normal_world=(0, -1, 0),
        wall_width_meters=4,
        wall_height_meters=3,
        minimum_width_meters=0.2,
        minimum_height_meters=0.3,
        bounds=CanvasOpeningBounds(0.25, 0.75, 0.2, 0.8),
    )
    widget.set_wall_targets((wall,))
    widget.set_canvas_opening_targets((target,))
    widget.select_canvas_opening(target.reference)
    return target


# ### Background click recovery with real Canvas handlers ###
def test_right_release_then_panel_click_returns_to_mouse_look_without_rectangle_selection(
    viewer,
):
    view = viewer.view
    viewer.enter_first_person_mode()
    pose = viewer.get_first_person_camera_pose()
    _right_release(view)
    control = _lose_focus_to_control(view)
    clicked = Mock()
    view.viewport_clicked.connect(clicked)
    _left_click(view)
    assert viewer.is_first_person_active
    assert viewer.is_first_person_pointer_captured
    assert viewer.get_first_person_camera_pose() == pose
    assert not view.is_primary_pointer_drag_reserved
    assert viewer._canvas_rectangle_selection_press_position is None
    assert viewer._canvas_rectangle_selection_cancel_event is None
    clicked.assert_not_called()
    control.deleteLater()


def test_focus_loss_while_captured_does_not_require_a_navigation_mode_toggle(viewer):
    view = viewer.view
    viewer.enter_first_person_mode()
    pose = viewer.get_first_person_camera_pose()
    control = _lose_focus_to_control(view)
    assert not viewer.is_first_person_pointer_captured
    _left_click(view)
    assert viewer.is_first_person_pointer_captured
    assert viewer.is_first_person_active
    assert viewer.get_first_person_camera_pose() == pose
    assert not view.is_primary_pointer_drag_reserved
    control.deleteLater()


def test_ctrl_released_in_another_top_level_window_cannot_leave_stale_interaction_state(
    viewer,
):
    view = viewer.view
    viewer.enter_first_person_mode()
    pose = viewer.get_first_person_camera_pose()
    QTest.keyPress(view, Qt.Key.Key_Control)
    assert view.is_first_person_ctrl_interaction_active
    control = _lose_focus_to_control(view, separate_window=True)
    assert control.window() is not view.window()
    QTest.keyRelease(control, Qt.Key.Key_Control)
    _left_click(view)
    assert not view.is_first_person_ctrl_interaction_active
    assert not view.is_control_modifier_pressed
    assert viewer.is_first_person_pointer_captured
    assert viewer.get_first_person_camera_pose() == pose
    assert not view.is_primary_pointer_drag_reserved
    control.deleteLater()


# ### Intentional Ctrl and asterisk interactions remain available ###
def test_actual_held_ctrl_keeps_the_cursor_and_canvas_selection_available(viewer):
    view = viewer.view
    viewer.enter_first_person_mode()
    QTest.keyPress(view, Qt.Key.Key_Control)
    control = _lose_focus_to_control(view)
    _left_click(view, modifiers=Qt.KeyboardModifier.ControlModifier)
    assert view.is_first_person_ctrl_interaction_active
    assert not viewer.is_first_person_pointer_captured
    assert viewer.is_first_person_active
    QTest.keyRelease(view, Qt.Key.Key_Control)
    assert viewer.is_first_person_pointer_captured
    control.deleteLater()


def test_held_ctrl_can_still_grab_an_opening_gizmo(viewer):
    target = _select_opening(viewer)
    view = viewer.view
    viewer.enter_first_person_mode()
    QTest.keyPress(view, Qt.Key.Key_Control)
    point = np.asarray(target.get_world_center())
    ray = point + np.asarray((0, -2, 0)), np.asarray((0, 1, 0))
    with patch.object(view, "build_camera_ray", return_value=ray):
        view.mousePressEvent(
            _mouse(
                QEvent.Type.MouseButtonPress,
                modifiers=Qt.KeyboardModifier.ControlModifier,
            )
        )
        assert viewer._canvas_opening_edit_drag is not None
        assert view.is_primary_pointer_drag_reserved
        assert not viewer.is_first_person_pointer_captured
        viewer._cancel_canvas_gizmo_drag()


def test_right_released_cursor_can_still_activate_the_opening_star(viewer):
    target = _select_opening(viewer)
    view = viewer.view
    viewer.enter_first_person_mode()
    _right_release(view)
    viewer._update_opening_proportions_control_geometry()
    center = viewer._opening_proportions_center.copy()
    inverse, valid = view.viewMatrix().inverted()
    assert valid
    vector = inverse.mapVector(QVector3D(0, 0, 1))
    normal = np.asarray((vector.x(), vector.y(), vector.z()))
    ray = center + normal * 3, -normal
    requests = []
    viewer.opening_proportions_requested.connect(requests.append)
    with patch.object(view, "build_camera_ray", return_value=ray):
        _left_click(view)
    assert requests == [target.reference]
    assert viewer.is_first_person_active
    assert not view.is_primary_pointer_drag_reserved
