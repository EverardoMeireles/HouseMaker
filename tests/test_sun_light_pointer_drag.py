# ### Environment setup ###
from __future__ import annotations

import os
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ### Imports ###
import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.directional_light_state import create_directional_light
from housemaker.global_scale import GlobalScale
from housemaker.main import BlueprintWorkspace
from housemaker.viewer import GlbViewerWidget

# ### Fixtures and real pointer helpers ###
_application = QApplication.instance() or QApplication([])


@pytest.fixture
def viewer():
    widget = GlbViewerWidget(window_editing_enabled=True)
    widget.resize(900, 650)
    widget.show()
    _application.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    _application.processEvents()


def _send_mouse(viewer, event_type, position):
    moving = event_type == QEvent.Type.MouseMove
    releasing = event_type == QEvent.Type.MouseButtonRelease
    event = QMouseEvent(
        event_type,
        position,
        position,
        Qt.MouseButton.NoButton if moving else Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton if releasing else Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(viewer.view, event)


# ### Sun gizmo pointer regressions ###
@pytest.mark.parametrize("orientation_visible", (False, True))
@pytest.mark.parametrize("axis_index", (0, 1, 2))
@pytest.mark.parametrize("scaled", (False, True))
def test_sun_translation_uses_actual_axis_pointer_events(
    viewer, orientation_visible, axis_index, scaled
):
    if scaled:
        viewer.set_global_scale(GlobalScale(1.8, 1.9, 1.3, linked=False))
    light = create_directional_light((), position=(0.0, 0.0, 1.0))
    viewer.set_directional_lights((light,))
    viewer.set_selected_directional_light_id(light.light_id)
    viewer.set_canvas_face_orientation_visible(orientation_visible)
    axis = np.eye(3)[axis_index]
    endpoint = np.asarray(light.position) + axis * viewer._directional_light_gizmo_size
    start = viewer._project_tour_world_point(endpoint)[0]
    end = viewer._project_tour_world_point(endpoint + axis * 0.4)[0]
    preview_events = []
    final_events = []
    viewer.directional_light_transform_preview_changed.connect(
        lambda *payload: preview_events.append(payload)
    )
    viewer.directional_light_transform_changed.connect(
        lambda *payload: final_events.append(payload)
    )
    original_sun = viewer._get_sun_light_snapshot()

    _send_mouse(viewer, QEvent.Type.MouseButtonPress, start)
    assert viewer._directional_light_drag is not None
    assert viewer.view.is_primary_pointer_drag_reserved
    assert viewer._canvas_rectangle_selection_press_position is None
    _send_mouse(viewer, QEvent.Type.MouseMove, end)
    assert preview_events
    assert viewer._get_sun_light_snapshot() == original_sun
    _send_mouse(viewer, QEvent.Type.MouseButtonRelease, end)

    assert viewer._directional_light_drag is None
    assert not viewer.view.is_primary_pointer_drag_reserved
    assert len(final_events) == 1
    assert final_events[0][0] == light.light_id
    np.testing.assert_allclose(
        final_events[0][1], np.asarray(light.position) + axis * 0.4, atol=1e-4
    )
    np.testing.assert_allclose(
        final_events[0][2], np.asarray(light.target) + axis * 0.4, atol=1e-4
    )


def test_face_orientation_still_receives_clicks_outside_the_sun_gizmo(viewer):
    light = create_directional_light((), position=(0.0, 0.0, 1.0))
    viewer.set_directional_lights((light,))
    viewer.set_selected_directional_light_id(light.light_id)
    viewer.set_canvas_face_orientation_visible(True)
    with patch.object(viewer, "_begin_canvas_rectangle_selection") as begin_rectangle:
        _send_mouse(viewer, QEvent.Type.MouseButtonPress, QPointF(10.0, 10.0))
    begin_rectangle.assert_called_once()
    assert viewer._directional_light_drag is None


# ### Canonical scene-state integration ###
def test_pointer_translation_commits_sun_position_and_preserves_its_direction(tmp_path):
    workspace = BlueprintWorkspace(
        application_settings=ApplicationSettingsStore(tmp_path / "settings.json")
    )
    try:
        widget = workspace.viewer
        widget.resize(900, 650)
        widget.show()
        _application.processEvents()
        light = create_directional_light((), position=(0.0, 0.0, 1.0))
        workspace.directional_lights = (light,)
        workspace._selected_directional_light_id = light.light_id
        workspace._sync_directional_light_views()
        widget.set_canvas_face_orientation_visible(True)
        endpoint = np.asarray(light.position) + np.asarray(
            (widget._directional_light_gizmo_size, 0.0, 0.0)
        )
        start = widget._project_tour_world_point(endpoint)[0]
        end = widget._project_tour_world_point(endpoint + (0.4, 0.0, 0.0))[0]

        _send_mouse(widget, QEvent.Type.MouseButtonPress, start)
        _send_mouse(widget, QEvent.Type.MouseMove, end)
        _send_mouse(widget, QEvent.Type.MouseButtonRelease, end)

        actual = workspace.directional_lights[0]
        np.testing.assert_allclose(actual.position, (0.4, 0.0, 1.0), atol=1e-4)
        np.testing.assert_allclose(actual.target, (0.4, 0.0, 0.0), atol=1e-4)
        assert widget._directional_lights == (actual,)
        assert widget.get_selected_directional_light_id() == light.light_id
        assert widget._directional_light_drag is None
    finally:
        workspace.shutdown()
        workspace.close()
        workspace.deleteLater()
        _application.processEvents()
