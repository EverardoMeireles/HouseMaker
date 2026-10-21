# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from dataclasses import replace

import numpy as np
import pytest
import trimesh
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QVector3D
from PySide6.QtWidgets import QApplication

from housemaker.camera_models import CameraPose
from housemaker.glb import GeneratedModel
from housemaker.global_scale import GlobalScale
from housemaker.viewer import GlbViewerWidget, SelectableGLViewWidget

# ### Fixtures and helpers ###
APP = QApplication.instance() or QApplication([])
APP.setQuitOnLastWindowClosed(False)


@pytest.fixture
def view():
    widget = SelectableGLViewWidget()
    widget.resize(800, 600)
    yield widget
    widget.exit_first_person_mode()
    widget.close()
    widget.deleteLater()
    APP.processEvents()


@pytest.fixture
def viewer():
    widget = GlbViewerWidget()
    widget.view.resize(800, 600)
    yield widget
    widget.view.exit_first_person_mode()
    widget.close()
    widget.deleteLater()
    APP.processEvents()


def _xyz(point):
    return np.asarray((point.x(), point.y(), point.z()), dtype=float)


def _matrix(matrix):
    return np.asarray(matrix.data()).reshape(4, 4, order="F")


# ### Camera, rendering, and picking ###
def test_scaled_scene_projection_matches_a_physical_scene_and_rays_stay_canonical(view):
    scale = GlobalScale(2.0, 0.5, 3.0, linked=False)
    target = np.asarray((2.0, -4.0, 1.2))
    view.setCameraPosition(
        pos=QVector3D(*target), distance=12.0, elevation=25, azimuth=35
    )
    physical = SelectableGLViewWidget()
    physical.resize(800, 600)
    try:
        physical.setCameraPosition(
            pos=QVector3D(*scale.apply_point(target)),
            distance=12.0,
            elevation=25,
            azimuth=35,
        )
        view.set_global_scale(scale)
        np.testing.assert_allclose(
            _matrix(view.viewMatrix()),
            _matrix(physical.viewMatrix()) @ np.asarray(scale.matrix),
            atol=2e-6,
        )
        np.testing.assert_allclose(
            _xyz(view.cameraPosition()) * scale.factors,
            _xyz(physical.cameraPosition()),
            atol=2e-6,
        )
        origin, direction = view.build_camera_ray(QPointF(400, 300))
        hit = origin + direction * np.dot(target - origin, direction)
        np.testing.assert_allclose(hit, target, atol=1e-4)
    finally:
        physical.close()
        physical.deleteLater()


def test_global_scale_does_not_mutate_editable_mesh_or_overlay_points(viewer):
    mesh = trimesh.creation.box(extents=(3.0, 4.0, 2.0))
    model = GeneratedModel(mesh=mesh, scene=trimesh.Scene(mesh), glb_bytes=b"")
    vertices = np.asarray(mesh.vertices).copy()
    viewer.set_model(model)
    viewer.set_tour_overlay(
        trigger_point=(1, 2, 0),
        curve_points=((0, 0, 1), (4, 4, 2)),
    )
    viewer.set_global_scale(GlobalScale(2, 3, 4, linked=False))
    np.testing.assert_array_equal(model.mesh.vertices, vertices)
    np.testing.assert_array_equal(viewer._tour_curve_points, ((0, 0, 1), (4, 4, 2)))
    viewer.set_global_scale(GlobalScale())
    np.testing.assert_array_equal(model.mesh.vertices, vertices)


@pytest.mark.parametrize("relative", ["view", "view-upright"])
def test_pan_keeps_the_same_physical_screen_motion(view, relative):
    scale = GlobalScale(2, 0.5, 3, linked=False)
    view.setCameraPosition(
        pos=QVector3D(2, 3, 4), distance=10, elevation=25, azimuth=30
    )
    physical = SelectableGLViewWidget()
    physical.resize(view.size())
    try:
        physical.setCameraPosition(
            pos=QVector3D(*scale.apply_point((2, 3, 4))),
            distance=10,
            elevation=25,
            azimuth=30,
        )
        view.set_global_scale(scale)
        view.pan(20, 15, 0, relative=relative)
        physical.pan(20, 15, 0, relative=relative)
        np.testing.assert_allclose(
            _xyz(view.opts["center"]) * scale.factors,
            _xyz(physical.opts["center"]),
            atol=2e-6,
        )
    finally:
        physical.close()
        physical.deleteLater()


# ### Physical first-person movement and authored tour cameras ###
def test_first_person_speed_and_direction_remain_physical_after_anisotropic_scaling(
    view,
):
    scale = GlobalScale(2, 3, 4, linked=False)
    view.set_global_scale(scale)
    view.set_first_person_speed(2.0)
    pose = CameraPose(x=1, y=2, z=0.4, yaw_degrees=30)
    view.set_first_person_camera_pose(pose)
    view.enter_first_person_mode()
    view._pressed_movement_keys = {Qt.Key.Key_Z}
    view.step_first_person_movement(0.5)
    moved = view.get_first_person_camera_pose()
    physical_motion = (
        np.asarray((moved.x, moved.y, moved.z)) - (pose.x, pose.y, pose.z)
    ) * scale.factors
    np.testing.assert_allclose(physical_motion, (np.sqrt(3) / 2, 0.5, 0), atol=1e-6)
    np.testing.assert_allclose(
        _xyz(view.cameraPosition()), (moved.x, moved.y, moved.z), atol=1e-6
    )
    _, direction = view.build_camera_ray(QPointF(400, 300))
    physical_direction = direction * scale.factors
    physical_direction /= np.linalg.norm(physical_direction)
    np.testing.assert_allclose(physical_direction, (np.sqrt(3) / 2, 0.5, 0), atol=2e-5)


def test_scale_change_preserves_camera_height_above_current_level_floor(viewer):
    pose = CameraPose(x=2, y=3, z=4.6)
    viewer.set_first_person_camera_pose(pose)
    viewer.set_global_scale(GlobalScale(2, 2, 3, linked=False), floor_z=3.0)
    moved = viewer.get_first_person_camera_pose()
    assert (moved.z - 3.0) * 3.0 == pytest.approx(1.6)
    viewer.set_global_scale(GlobalScale(), floor_z=3.0)
    assert viewer.get_first_person_camera_pose() == replace(pose)


def test_tour_camera_stays_on_canonical_curve_and_aims_at_scaled_target(viewer):
    scale = GlobalScale(3, 0.5, 2, linked=False)
    viewer.set_global_scale(scale)
    viewer.set_tour_overlay(
        trigger_point=(0, 0, 0),
        curve_points=((1, 2, 3), (5, 6, 4)),
    )
    camera = np.asarray((2.0, 3.0, 3.5))
    target = np.asarray((-3.0, 6.0, 2.0))
    assert viewer.set_tour_preview(0.5, target, camera_position=camera)
    np.testing.assert_allclose(_xyz(viewer.view.cameraPosition()), camera, atol=2e-6)
    near, direction = viewer.view.build_camera_ray(QPointF(400, 300))
    hit = near + np.dot(target - near, direction) * direction
    np.testing.assert_allclose(hit, target, atol=1e-4)
    forward = np.asarray(viewer.get_camera_forward_world_point(1.0))
    assert np.linalg.norm((forward - camera) * scale.factors) == pytest.approx(
        1.0, abs=1e-6
    )


def test_rescaling_cached_tour_camera_preserves_manual_viewpoint_without_scrubbing(
    viewer,
):
    viewer.set_tour_overlay(
        trigger_point=(0, 0, 0),
        curve_points=((1, 2, 3), (5, 6, 4)),
    )
    assert viewer.set_tour_preview(0.5, (-3, 6, 2), camera_position=(2, 3, 3.5))
    viewer.view.orbit(12, 4)
    camera = _xyz(viewer.view.cameraPosition())
    target = _xyz(viewer.view.opts["center"])
    viewer.set_global_scale(GlobalScale(3, 0.5, 2, linked=False))
    np.testing.assert_allclose(_xyz(viewer.view.cameraPosition()), camera, atol=2e-6)
    near, direction = viewer.view.build_camera_ray(QPointF(400, 300))
    hit = near + np.dot(target - near, direction) * direction
    np.testing.assert_allclose(hit, target, atol=1e-4)


def test_scaling_does_not_reset_orbit_navigation_or_field_of_view(view):
    view.setCameraPosition(
        pos=QVector3D(3, 4, 5), distance=12, elevation=17, azimuth=23
    )
    view.opts["fov"] = 72.0
    expected = dict(view.opts)
    view.set_global_scale(GlobalScale(3, 0.5, 2, linked=False))
    assert view.opts == expected
