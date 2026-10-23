"""Gravity changes FPS movement only and uses cached scene support geometry."""

# ### Environment ###
from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ### Imports ###
import pytest
import trimesh
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from housemaker.camera_models import CameraPose
from housemaker.first_person_gravity import FirstPersonGroundIndex
from housemaker.glb import (
    GeneratedModel,
    PreviewPlacedObject,
    PreviewStairPart,
    PreviewTexturedSurface,
)
from housemaker.global_scale import GlobalScale
from housemaker.viewer import GlbViewerWidget, SelectableGLViewWidget

# ### Fixtures and geometry helpers ###
_application = QApplication.instance() or QApplication([])
_application.setQuitOnLastWindowClosed(False)


@pytest.fixture
def view():
    widget = SelectableGLViewWidget()
    try:
        yield widget
    finally:
        widget.exit_first_person_mode()
        widget.close()
        widget.deleteLater()
        _application.processEvents()


@pytest.fixture
def viewer(request):
    widget = GlbViewerWidget(window_editing_enabled=getattr(request, "param", False))
    try:
        yield widget
    finally:
        widget.exit_first_person_mode()
        widget.close()
        widget.deleteLater()
        _application.processEvents()


def _floor(z=0.0, *, x0=-20.0, x1=20.0):
    return trimesh.Trimesh(
        vertices=((x0, -20, z), (x1, -20, z), (x1, 20, z), (x0, 20, z)),
        faces=((0, 1, 2), (0, 2, 3)), process=False,
    )


def _model(mesh):
    return GeneratedModel(mesh=mesh, scene=trimesh.Scene(mesh), glb_bytes=b"")


def _start(view, provider, pose=None):
    view.set_first_person_ground_height_provider(provider)
    view.set_first_person_camera_pose(pose if pose is not None else CameraPose(z=1.7))
    view.set_first_person_gravity_enabled(True)
    view.enter_first_person_mode()
    view._movement_timer.stop()


# ### Falling, physical eye height and movement ###
def test_gravity_is_opt_in_and_does_not_change_orbit_camera(view):
    assert not view.is_first_person_gravity_enabled()
    view.set_first_person_camera_pose(CameraPose(z=4.0))
    view.set_first_person_gravity_enabled(True)
    view.step_first_person_movement(1.0)
    assert view.get_first_person_camera_pose().z == 4.0


def test_stationary_camera_falls_and_stops_at_real_floor_eye_height(view):
    index = FirstPersonGroundIndex(_floor(1.25))
    _start(view, index.ground_height, CameraPose(z=6.0))
    view.step_first_person_movement(0.2)
    assert 2.95 < view.get_first_person_camera_pose().z < 6.0
    view.step_first_person_movement(2.0)
    assert view.get_first_person_camera_pose().z == pytest.approx(2.95)
    view.set_first_person_eye_height(1.85)
    assert view.get_first_person_camera_pose().z == pytest.approx(3.10)
    view.step_first_person_movement(1.0)
    assert view.get_first_person_camera_pose().z == pytest.approx(3.10)


def test_gravity_overrides_flight_pitch_and_ignores_vertical_keys(view):
    view.set_first_person_movement_mode("noclip")
    index = FirstPersonGroundIndex(_floor())
    _start(view, index.ground_height, CameraPose(z=1.7, pitch_degrees=60.0))
    view._pressed_movement_keys = {Qt.Key.Key_Z, Qt.Key.Key_F}
    view.step_first_person_movement(0.4)
    assert view.get_first_person_camera_pose().x == pytest.approx(1.0)
    assert view.get_first_person_camera_pose().z == pytest.approx(1.7)
    assert view.get_first_person_movement_mode() == "noclip"
    view.set_first_person_gravity_enabled(False)
    view._pressed_movement_keys = {Qt.Key.Key_F}
    view.step_first_person_movement(0.4)
    assert view.get_first_person_camera_pose().z == pytest.approx(2.7)


def test_walks_up_and_down_steps_without_jitter(view):
    treads = tuple(_floor(step * 0.2, x0=step * 0.5, x1=(step + 1) * 0.5)
                   for step in range(6))
    index = FirstPersonGroundIndex(trimesh.util.concatenate(treads))
    _start(view, index.ground_height, CameraPose(x=0.1, z=1.7))
    view._pressed_movement_keys = {Qt.Key.Key_Z}
    view.step_first_person_movement(0.8)
    assert view.get_first_person_camera_pose().z == pytest.approx(2.5)
    view._pressed_movement_keys = {Qt.Key.Key_S}
    view.step_first_person_movement(0.8)
    assert view.get_first_person_camera_pose().z == pytest.approx(1.7)


def test_floor_gap_allows_falling_to_the_next_level(view):
    index = FirstPersonGroundIndex(trimesh.util.concatenate((
        _floor(0.0), _floor(3.0, x0=-20.0, x1=0.0),
    )))
    _start(view, index.ground_height, CameraPose(x=-0.2, z=4.7))
    view._pressed_movement_keys = {Qt.Key.Key_Z}
    view.step_first_person_movement(0.2)
    assert view.get_first_person_camera_pose().z < 4.7
    view._pressed_movement_keys.clear()
    view.step_first_person_movement(2.0)
    assert view.get_first_person_camera_pose().z == pytest.approx(1.7)


def test_physical_eye_height_does_not_scale_with_building(view):
    view.set_global_scale(GlobalScale(x=1.8, y=1.9, z=2.0))
    index = FirstPersonGroundIndex(_floor(1.0))
    _start(view, index.ground_height, CameraPose(z=1.85))
    view.step_first_person_movement(1.0)
    assert view.get_first_person_camera_pose().z == pytest.approx(1.85)
    view.set_first_person_eye_height(2.0)
    assert view.get_first_person_camera_pose().z == pytest.approx(2.0)


def test_teleport_resets_fall_velocity(view):
    index = FirstPersonGroundIndex(_floor())
    _start(view, index.ground_height, CameraPose(z=8.0))
    view.step_first_person_movement(0.5)
    assert view._first_person_vertical_velocity < 0.0
    view.set_first_person_camera_pose(CameraPose(z=1.7))
    assert view._first_person_vertical_velocity == 0.0
    view.step_first_person_movement(0.1)
    assert view.get_first_person_camera_pose().z == pytest.approx(1.7)


def test_floor_above_head_is_not_used_as_support(view):
    index = FirstPersonGroundIndex(trimesh.util.concatenate((_floor(), _floor(2.5))))
    _start(view, index.ground_height)
    view.step_first_person_movement(0.5)
    assert view.get_first_person_camera_pose().z == pytest.approx(1.7)


def test_slab_between_feet_and_eyes_does_not_teleport_camera_upward(view):
    index = FirstPersonGroundIndex(trimesh.util.concatenate((_floor(), _floor(1.0))))
    _start(view, index.ground_height)
    view.step_first_person_movement(0.5)
    assert view.get_first_person_camera_pose().z == pytest.approx(1.7)


# ### Render-independent lazy support cache ###
def test_scene_ground_index_is_lazy_cached_and_replaced_with_model(viewer):
    viewer.set_model(_model(_floor()))
    assert viewer._first_person_ground_index is None
    assert viewer._get_first_person_ground_height(0, 0, 2) == 0.0
    index = viewer._first_person_ground_index
    assert viewer._get_first_person_ground_height(1, 1, 2) == 0.0
    assert viewer._first_person_ground_index is index
    viewer.set_model(_model(_floor(1.0)), preserve_camera=True)
    assert viewer._first_person_ground_index is None
    assert viewer._get_first_person_ground_height(0, 0, 2) == 1.0
    viewer.clear_model()
    assert viewer._get_first_person_ground_height(0, 0, 2) is None


def test_textured_floor_is_walkable_when_excluded_from_untextured_base(viewer):
    model = _model(_floor(1.0))
    model.preview_untextured_mesh = trimesh.Trimesh(process=False)
    model.preview_textured_surfaces = [PreviewTexturedSurface(
        "level:2/floor", "floor", _floor(1.0), level_index=2,
    )]
    viewer.set_model(model)
    assert viewer._get_first_person_ground_height(0, 0, 2) == 1.0


def test_landing_support_uses_its_current_preview_transform(viewer):
    viewer.set_model(_model(_floor()))
    transform = trimesh.transformations.translation_matrix((0, 0, 1.0))
    viewer._landing_previews = {"landing:test": PreviewPlacedObject(
        "landing:test", (_floor(),), transform, (0, 0, 1.0), (0, 0, 0),
    )}
    assert viewer._get_first_person_ground_height(0, 0, 2) == 1.0


def test_direct_landing_transform_invalidates_old_support(viewer):
    viewer.set_model(_model(_floor()))
    transform = trimesh.transformations.translation_matrix((0, 0, 1.0))
    preview = PreviewPlacedObject(
        "landing:test", (_floor(x0=0, x1=1),), transform, (0, 0, 1.0), (0, 0, 0),
    )
    viewer._landing_previews = {preview.object_id: preview}
    assert viewer._get_first_person_ground_height(0.5, 0, 2) == 1.0
    group = SimpleNamespace(
        preview=preview,
        current_transform=trimesh.transformations.translation_matrix((2, 0, 1.0)),
    )
    viewer._remember_placed_object_preview_transform(
        group, (2, 0, 1.0), (0, 0, 0), (1, 1, 1),
    )
    assert viewer._first_person_ground_index is None
    assert viewer._get_first_person_ground_height(0.5, 0, 2) == 0.0
    assert viewer._get_first_person_ground_height(2.5, 0, 2) == 1.0


def test_staged_stair_support_is_reindexed_when_changed_and_cleared(viewer):
    viewer.set_model(_model(_floor()))
    assert viewer._get_first_person_ground_height(0, 0, 2) == 0.0
    stair_id = "a" * 32
    part = PreviewStairPart(
        stair_id, 0, f"stair:{stair_id}/part:treads:floor", "treads", "floor",
        _floor(0.2), (2,),
    )
    assert viewer.set_canvas_stair_preview(0, (part,))
    assert viewer._get_first_person_ground_height(0, 0, 2) == pytest.approx(0.2)
    viewer.clear_canvas_stair_preview()
    assert viewer._get_first_person_ground_height(0, 0, 2) == 0.0


@pytest.mark.parametrize("viewer", (True,), indirect=True)
def test_hidden_levels_do_not_provide_invisible_support(viewer):
    model = _model(_floor())
    model.preview_untextured_mesh = trimesh.Trimesh(process=False)
    model.preview_textured_surfaces = [
        PreviewTexturedSurface("level:2/floor", "floor", _floor(1.0), level_index=2),
        PreviewTexturedSurface("level:3/floor", "floor", _floor(3.0), level_index=3),
    ]
    viewer.set_canvas_scene_levels(((2, "Ground"), (3, "First")))
    viewer.set_model(model)
    assert viewer._get_first_person_ground_height(0, 0, 4) == 3.0
    viewer.set_visible_canvas_level_indices((2,))
    assert viewer._get_first_person_ground_height(0, 0, 4) == 1.0


def test_disabled_gravity_never_builds_support_index(viewer):
    viewer.set_model(_model(_floor()))
    viewer.enter_first_person_mode()
    viewer.view._movement_timer.stop()
    with patch.object(viewer, "_get_display_mesh", side_effect=AssertionError):
        viewer.view.step_first_person_movement(0.2)
    assert viewer._first_person_ground_index is None
