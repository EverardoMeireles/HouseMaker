"""Plan sun movements preserve physical direction and survive editor lifecycle."""

# ### Environment and imports ###
from __future__ import annotations

import os
from dataclasses import replace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.directional_light_state import DirectionalLightData
from housemaker.global_scale import GlobalScale
from housemaker.level_coordinates import level_image_to_world_xy
from housemaker.main import BlueprintWorkspace
from housemaker.models import GROUND_LEVEL_INDEX, create_default_levels
from housemaker.project_io import ProjectData
from tests.test_canvas_sun_drag import _move_with_left

# ### Temporary authored sun and plan ###
_application = QApplication.instance() or QApplication([])
_application.setQuitOnLastWindowClosed(False)


@pytest.fixture
def workspace(tmp_path):
    widget = BlueprintWorkspace(
        application_settings=ApplicationSettingsStore(tmp_path / "settings.json")
    )
    _application.processEvents()
    canvas = widget.canvas
    canvas.resize(640, 480)
    canvas.blueprint_image = QImage(200, 100, QImage.Format.Format_RGB32)
    canvas.blueprint_image.fill(Qt.GlobalColor.white)
    level = widget.current_level
    level.image_size_pixels = (200, 100)
    level.plan_meters_per_pixel = 0.04
    level.scale = 1.7
    level.offset_x_meters = 2.5
    level.offset_y_meters = -1.4
    widget.global_scale = GlobalScale(1.8, 2.3, 1.2, linked=False)
    widget.viewer.set_global_scale(widget.global_scale)
    x, y = level_image_to_world_xy(level, 80, 35)
    light = DirectionalLightData(
        "sun", "Sun", (x, y, 4), (x + 0.5, y - 0.4, 2), level_index=level.index
    )
    widget.directional_lights = (light,)
    widget._selected_directional_light_id = light.light_id
    widget._sync_directional_light_views()
    try:
        yield widget
    finally:
        widget.shutdown()
        widget.close()
        _application.processEvents()


def _drag_sun(workspace):
    canvas = workspace.canvas
    start = canvas._image_to_widget(80, 35).toPoint()
    end = QPointF(start) + QPointF(45, 30)
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=start)
    _move_with_left(canvas, end)
    return end


# ### Shared canonical coordinates and no scene rebuilds ###
def test_plan_drag_previews_xy_with_current_level_transform_and_nonuniform_global_scale(
    workspace,
):
    initial = workspace.directional_lights[0]
    rect = workspace.canvas._image_display_rect()
    expected_x, expected_y = level_image_to_world_xy(
        workspace.current_level,
        80 + 45 * 200 / rect.width(),
        35 + 30 * 100 / rect.height(),
    )
    with (
        patch.object(workspace, "_schedule_viewer_preview_refresh") as rebuild,
        patch.object(workspace.viewer, "set_model") as replace_mesh,
    ):
        end = _drag_sun(workspace)
        moved = workspace.directional_lights[0]
        np.testing.assert_allclose(
            moved.position, (expected_x, expected_y, initial.position[2])
        )
        np.testing.assert_allclose(
            np.subtract(moved.target, moved.position),
            np.subtract(initial.target, initial.position),
        )
        assert workspace.viewer._directional_lights == (moved,)
        assert workspace.viewer.get_global_scale() == workspace.global_scale
        QTest.mouseRelease(
            workspace.canvas, Qt.MouseButton.LeftButton, pos=end.toPoint()
        )
        assert workspace.directional_lights == (moved,)
        assert workspace.canvas.get_selected_directional_light_id() == "sun"
        assert workspace._canvas_directional_light_move_initial is None
        rebuild.assert_not_called()
        replace_mesh.assert_not_called()


@pytest.mark.parametrize("cancel", ("right_click", "escape"))
def test_cancel_restores_the_exact_original_scene_light_and_viewer(workspace, cancel):
    original = workspace.directional_lights
    end = _drag_sun(workspace)
    assert workspace.directional_lights != original
    if cancel == "right_click":
        QTest.mouseClick(
            workspace.canvas, Qt.MouseButton.RightButton, pos=end.toPoint()
        )
    else:
        QTest.keyClick(workspace.canvas, Qt.Key.Key_Escape)
    assert workspace.directional_lights == original
    assert workspace.viewer._directional_lights == original
    assert workspace._canvas_directional_light_move_initial is None
    QTest.mouseRelease(workspace.canvas, Qt.MouseButton.LeftButton, pos=end.toPoint())


def test_level_change_cancels_using_original_world_coordinates(workspace):
    original = workspace.directional_lights
    _drag_sun(workspace)
    workspace._handle_level_list_row_changed(
        workspace._level_list_row_for_position(GROUND_LEVEL_INDEX + 1)
    )
    assert workspace.current_level_index == GROUND_LEVEL_INDEX + 1
    assert workspace.directional_lights == original
    assert workspace._canvas_directional_light_move_initial is None


def test_loading_another_project_with_the_same_light_id_does_not_restore_the_old_sun(
    workspace,
):
    _drag_sun(workspace)
    new_sun = replace(
        workspace.directional_lights[0], position=(20, 30, 5), target=(20, 30, 2)
    )
    project = ProjectData(
        None, GROUND_LEVEL_INDEX, create_default_levels(), directional_lights=(new_sun,)
    )
    with patch.object(workspace, "_queue_viewer_preview_refresh"):
        workspace._apply_loaded_project(project)
    assert workspace.directional_lights == (new_sun,)
    assert workspace.viewer._directional_lights == (new_sun,)
    assert workspace._canvas_directional_light_move_initial is None
