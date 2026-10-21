"""Landing gizmo presses stay responsive when an axis points at the camera."""

# ### Environment ###
from __future__ import annotations

import os
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ### Imports ###
import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QVector3D
from PySide6.QtWidgets import QApplication

from housemaker.glb import convert_to_glb
from housemaker.models import LandingData, LevelData
from housemaker.viewer import (
    PLACED_OBJECT_GIZMO_SCALE,
    TRANSFORM_GIZMO_ROTATE,
    TRANSFORM_GIZMO_TRANSLATE,
    GlbViewerWidget,
    _get_landing_uniform_scale_handle_position,
    _TransformGizmoHandle,
)

# ### Fixtures and camera rays ###
_application = QApplication.instance() or QApplication([])
_application.setQuitOnLastWindowClosed(False)


@pytest.fixture
def selected_landing_viewer():
    level = LevelData(index=2, name="Ground")
    landing = LandingData(center_x=100.0, center_y=100.0)
    level.landings = [landing]
    viewer = GlbViewerWidget(window_editing_enabled=True)
    viewer.set_model(convert_to_glb([level]))
    viewer.set_landing_context([level])
    assert viewer.select_landing(landing.landing_id)
    yield viewer, landing
    viewer.close()
    viewer.deleteLater()
    _application.processEvents()


def _group_and_pivot(viewer, landing):
    group = viewer._placed_object_render_groups[f"landing:{landing.landing_id}"]
    return group, np.asarray(group.preview.world_position, dtype=float)


def _pointer_event(event_type, position):
    return QMouseEvent(
        event_type,
        position,
        position,
        Qt.MouseButton.LeftButton,
        (
            Qt.MouseButton.LeftButton
            if event_type == QEvent.Type.MouseButtonPress
            else Qt.MouseButton.NoButton
        ),
        Qt.KeyboardModifier.NoModifier,
    )


# ### Axis picking and draggable visible shafts ###
@pytest.mark.parametrize("view_axis_index", (0, 1, 2))
def test_pivot_pick_never_chooses_collapsed_camera_aligned_axis(
    selected_landing_viewer, view_axis_index
):
    viewer, landing = selected_landing_viewer
    _group, pivot = _group_and_pivot(viewer, landing)
    view_axis = np.eye(3)[view_axis_index]
    ray = pivot + view_axis * 5.0, -view_axis

    handle = viewer._pick_transform_gizmo_handle(*ray)

    assert handle is not None
    assert handle.kind == TRANSFORM_GIZMO_TRANSLATE
    assert handle.axis_index != view_axis_index
    with patch.object(viewer.view, "build_camera_ray", return_value=ray):
        assert viewer._begin_placed_object_gizmo_drag(handle, QPointF())
    assert viewer.view.is_primary_pointer_drag_reserved


@pytest.mark.parametrize("scale_mode", (False, True))
def test_nearly_camera_aligned_axis_cannot_steal_visible_shaft_pick(
    selected_landing_viewer, scale_mode
):
    viewer, landing = selected_landing_viewer
    if scale_mode:
        assert viewer._toggle_placed_object_gizmo_mode()
        assert viewer._placed_object_gizmo_mode == PLACED_OBJECT_GIZMO_SCALE
    _group, pivot = _group_and_pivot(viewer, landing)
    point = pivot + np.asarray((0.6 * viewer._transform_gizmo_size, 0.0, 0.0))
    direction = np.asarray((-1.0, 0.04, 0.0))
    direction /= np.linalg.norm(direction)
    ray = point - direction * 5.0, direction

    handle = viewer._pick_transform_gizmo_handle(*ray)

    assert handle is not None
    assert handle.axis_index != 0
    with patch.object(viewer.view, "build_camera_ray", return_value=ray):
        assert viewer._begin_placed_object_gizmo_drag(handle, QPointF())


def test_top_down_rotation_ring_remains_pickable(selected_landing_viewer):
    viewer, landing = selected_landing_viewer
    _group, pivot = _group_and_pivot(viewer, landing)
    radius = viewer._transform_gizmo_size * 0.72
    offset = radius / np.sqrt(2.0)
    point = pivot + np.asarray((offset, offset, 0.0))
    ray = point + (0.0, 0.0, 5.0), np.asarray((0.0, 0.0, -1.0))

    handle = viewer._pick_transform_gizmo_handle(*ray)

    assert handle == _TransformGizmoHandle(TRANSFORM_GIZMO_ROTATE, 2)
    with patch.object(viewer.view, "build_camera_ray", return_value=ray):
        assert viewer._begin_placed_object_gizmo_drag(handle, QPointF())


# ### Failed presses must not fall through to expensive scene selection ###
def test_failed_gizmo_press_suppresses_scene_click_on_actual_mouse_release(
    selected_landing_viewer,
):
    viewer, _landing = selected_landing_viewer
    position = QPointF(30.0, 30.0)
    ray = np.asarray((5.0, 0.0, 0.0)), np.asarray((-1.0, 0.0, 0.0))
    clicks = []
    viewer.view.viewport_clicked.connect(clicks.append)
    handle = _TransformGizmoHandle(TRANSFORM_GIZMO_TRANSLATE, 0)

    with (
        patch.object(viewer.view, "build_camera_ray", return_value=ray),
        patch.object(viewer, "_pick_landing_generation_control", return_value=False),
        patch.object(viewer, "_pick_transform_gizmo_handle", return_value=handle),
        patch.object(viewer, "_begin_placed_object_gizmo_drag", return_value=False),
        patch.object(viewer.view, "_get_clicked_items", return_value=[]) as item_pick,
    ):
        viewer.view.mousePressEvent(
            _pointer_event(QEvent.Type.MouseButtonPress, position)
        )
        viewer.view.mouseReleaseEvent(
            _pointer_event(QEvent.Type.MouseButtonRelease, position)
        )

    assert clicks == []
    item_pick.assert_not_called()
    assert not viewer.view.is_primary_pointer_drag_reserved
    assert viewer._placed_object_transform_drag is None


# ### Uniform scale still works when its diagonal faces the camera ###
def test_view_aligned_uniform_scale_uses_draggable_screen_axis(
    selected_landing_viewer,
):
    viewer, landing = selected_landing_viewer
    assert viewer._toggle_placed_object_gizmo_mode()
    group, pivot = _group_and_pivot(viewer, landing)
    viewer.view.setCameraPosition(
        pos=QVector3D(*pivot),
        elevation=float(np.degrees(np.arctan(1.0 / np.sqrt(2.0)))),
        azimuth=45.0,
        distance=5.0,
    )
    center = _get_landing_uniform_scale_handle_position(
        group, viewer._transform_gizmo_size
    )
    direction = pivot - center
    direction /= np.linalg.norm(direction)
    origin = center - direction * 5.0
    ray = origin, direction
    handle = _TransformGizmoHandle("uniform_scale", -1)

    with patch.object(viewer.view, "build_camera_ray", return_value=ray):
        assert viewer._begin_placed_object_gizmo_drag(handle, QPointF())
    drag = viewer._placed_object_transform_drag
    assert drag is not None
    assert abs(float(np.dot(drag.axis, direction))) <= 1e-6
    move = drag.axis * drag.start_gizmo_size / np.sqrt(3.0) * 0.25
    with patch.object(
        viewer.view, "build_camera_ray", return_value=(origin + move, direction)
    ):
        assert viewer._update_placed_object_gizmo_drag(QPointF())

    np.testing.assert_allclose(
        drag.preview_axis_scales,
        np.asarray(drag.start_axis_scales) * 1.25,
        atol=1e-6,
    )

