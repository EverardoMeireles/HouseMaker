"""Camera spawn authoring does not mutate geometry or steal ordinary FPS input."""

# ### Environment and imports ###
from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

from housemaker.camera_models import CameraPose
from housemaker.camera_start_workspace import CameraStartWorkspaceMixin
from housemaker.global_scale import GlobalScale
from housemaker.lazy_loading_state import CameraStartPoint, RuntimeLoadingData
from housemaker.level_coordinates import (
    build_level_base_z_lookup,
    level_image_to_world_xy,
)
from housemaker.models import LevelData
from housemaker.viewer import GlbViewerWidget
from tests.test_canvas_directional_lights import _build_canvas, _render_canvas


# ### Fixtures ###
class _Workspace(CameraStartWorkspaceMixin, QWidget):
    def __init__(self):
        super().__init__()
        self.runtime_loading = RuntimeLoadingData()
        self._init_camera_start_state()
        self.canvas = _build_canvas()
        self.viewer = GlbViewerWidget(window_editing_enabled=True)
        self.levels = [self.canvas.level_context]
        self.current_level = self.levels[0]
        self.global_scale = GlobalScale(x=2.0, y=3.0, z=4.0, linked=False)
        self._generation_settings = SimpleNamespace(first_person_eye_height_meters=1.7)
        self.viewer.set_global_scale(self.global_scale)
        self._add_camera_start_controls(QVBoxLayout(self))


@pytest.fixture
def workspace():
    widget = _Workspace()
    yield widget
    widget._cancel_camera_start_placement()
    for child in (widget.canvas, widget.viewer):
        child.close()
        child.deleteLater()
    widget.close()
    widget.deleteLater()
    QApplication.processEvents()


# ### Canvas placement and cancellation ###
def test_canvas_placement_is_one_shot_and_never_edits_walls(workspace):
    canvas = workspace.canvas
    before = canvas.vertex_data.to_dict()
    workspace.viewer.set_first_person_camera_pose(
        CameraPose(yaw_degrees=37, pitch_degrees=-12)
    )
    workspace.set_fps_start_button.click()
    assert canvas._camera_start_placement_active
    assert workspace.viewer.is_camera_start_placement_active
    QTest.mouseClick(
        canvas,
        Qt.MouseButton.LeftButton,
        pos=canvas._image_to_widget(75, 40).toPoint(),
    )
    point = workspace.runtime_loading.start_point
    assert point.level_index == canvas.level_context.index
    assert point.image_x == pytest.approx(75, abs=0.5)
    assert point.image_y == pytest.approx(40, abs=0.5)
    assert point.yaw_degrees == 37
    assert point.pitch_degrees == -12
    assert point.height_offset_meters == 0
    assert not canvas._camera_start_placement_active
    assert not workspace.viewer.is_camera_start_placement_active
    assert not workspace.set_fps_start_button.isChecked()
    assert canvas.vertex_data.to_dict() == before


@pytest.mark.parametrize("escape", [False, True])
def test_canvas_right_click_or_escape_cancels_both_tools(workspace, escape):
    workspace.set_fps_start_button.click()
    if escape:
        QTest.keyClick(workspace.canvas, Qt.Key.Key_Escape)
    else:
        QTest.mouseClick(workspace.canvas, Qt.MouseButton.RightButton)
    assert workspace.runtime_loading.start_point is None
    assert not workspace.canvas._camera_start_placement_active
    assert not workspace.viewer.is_camera_start_placement_active
    assert not workspace.set_fps_start_button.isChecked()


def test_switching_to_other_tools_cancels_spawn_in_both_views(workspace):
    workspace.set_fps_start_button.click()
    workspace.canvas.begin_landing_placement()
    assert workspace.canvas.is_landing_placement_active()
    assert not workspace.viewer.is_camera_start_placement_active
    assert not workspace.set_fps_start_button.isChecked()
    workspace.canvas.cancel_landing_placement()
    workspace.set_fps_start_button.click()
    workspace.viewer.begin_landing_placement(fallback_plane_z=0)
    assert workspace.viewer.is_landing_placement_active()
    assert not workspace.canvas._camera_start_placement_active
    assert not workspace.set_fps_start_button.isChecked()


# ### 3D placement, eye height and source coordinates ###
def test_3d_click_bypasses_regular_selection_without_claiming_sun_or_tour_state(
    workspace,
):
    viewer = workspace.viewer
    floor_z = build_level_base_z_lookup(workspace.levels)[workspace.current_level.index]
    world_xy = level_image_to_world_xy(workspace.current_level, 70, 45)
    point = (*world_xy, floor_z + 0.3)
    pressed = QSignalSpy(viewer.view.primary_pointer_pressed)
    requested = QSignalSpy(viewer.camera_start_placed)
    workspace.set_fps_start_button.click()
    assert not viewer.is_directional_light_placement_active
    assert not viewer.is_tour_point_placement_active
    with (
        patch.object(
            viewer.view, "build_camera_ray", return_value=((0, 0, 1), (0, 0, -1))
        ),
        patch.object(viewer, "_pick_scene_world_point", return_value=point),
    ):
        QTest.mouseClick(
            viewer.view, Qt.MouseButton.LeftButton, pos=QPointF(20, 20).toPoint()
        )
    assert requested.count() == 1
    assert pressed.count() == 0
    stored = workspace.runtime_loading.start_point
    assert stored.image_x == pytest.approx(70)
    assert stored.image_y == pytest.approx(45)
    assert stored.height_offset_meters == pytest.approx(0.3)
    assert not viewer.is_camera_start_placement_active


@pytest.mark.parametrize("escape", [False, True])
def test_viewer_right_click_or_escape_cancels_both_modes(workspace, escape):
    workspace.set_fps_start_button.click()
    if escape:
        QTest.keyClick(workspace.viewer.view, Qt.Key.Key_Escape)
    else:
        QTest.mouseClick(workspace.viewer.view, Qt.MouseButton.RightButton)
    assert workspace.runtime_loading.start_point is None
    assert not workspace.canvas._camera_start_placement_active
    assert not workspace.viewer.is_camera_start_placement_active


def test_current_fps_position_subtracts_physical_eye_height_once(workspace):
    level = workspace.current_level
    floor_z = build_level_base_z_lookup(workspace.levels)[level.index]
    x, y = level_image_to_world_xy(level, 90, 50)
    workspace.viewer.set_first_person_camera_pose(
        CameraPose(
            x=x,
            y=y,
            z=floor_z + 0.2 + 1.7 / 4,
            yaw_degrees=45,
            pitch_degrees=10,
        )
    )
    workspace.use_current_fps_start_button.click()
    point = workspace.runtime_loading.start_point
    assert point.image_x == pytest.approx(90)
    assert point.image_y == pytest.approx(50)
    assert point.height_offset_meters == pytest.approx(0.2)
    assert point.yaw_degrees == 45
    assert point.pitch_degrees == 10


# ### Marker persistence and level transforms ###
def test_marker_follows_level_transform_and_is_hidden_on_other_canvas_levels(workspace):
    level = workspace.current_level
    workspace._store_camera_start(CameraStartPoint(level.index, 60, 30, yaw_degrees=90))
    initial = workspace.viewer._camera_start_marker
    level.offset_x_meters = 2
    workspace._sync_camera_start_marker()
    expected_xy = level_image_to_world_xy(level, 60, 30)
    assert workspace.viewer._camera_start_marker[:2] == expected_xy
    workspace.current_level = LevelData(index=level.index + 1, name="Other")
    workspace._sync_camera_start_marker()
    assert workspace.canvas._camera_start_marker is None
    assert workspace.viewer._camera_start_marker is not None
    workspace.viewer._clear_scene()
    assert len(workspace.viewer._camera_start_items) == 2
    assert all(
        item in workspace.viewer.view.items
        for item in workspace.viewer._camera_start_items
    )
    assert initial is not None


def test_marker_renders_and_clear_preserves_compression_preference(workspace):
    workspace.runtime_loading = RuntimeLoadingData(use_ktx2_basis_compression=True)
    before = _render_canvas(workspace.canvas)
    workspace._place_canvas_camera_start(60, 30)
    after = _render_canvas(workspace.canvas)
    assert before != after
    assert workspace.clear_fps_start_button.isEnabled()
    workspace.clear_fps_start_button.click()
    assert workspace.runtime_loading.start_point is None
    assert workspace.runtime_loading.use_ktx2_basis_compression
    assert workspace.canvas._camera_start_marker is None
    assert workspace.viewer._camera_start_marker is None
    assert not workspace.clear_fps_start_button.isEnabled()


def test_marker_refresh_is_idempotent_but_follows_floor_height(workspace):
    workspace._place_canvas_camera_start(60, 30)
    items = tuple(workspace.viewer._camera_start_items)
    previous = workspace.viewer._camera_start_marker
    workspace._sync_camera_start_marker()
    assert tuple(workspace.viewer._camera_start_items) == items
    workspace.current_level.floor_thickness_meters += 0.2
    workspace._sync_camera_start_marker()
    assert workspace.viewer._camera_start_marker[2] == pytest.approx(previous[2] + 0.2)
    assert tuple(workspace.viewer._camera_start_items) != items
