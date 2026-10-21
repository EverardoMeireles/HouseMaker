"""The purple landing cube scales all axes by one positive common factor."""

# ### Environment setup ###
from __future__ import annotations

import os
from dataclasses import replace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import numpy as np
import pyqtgraph.opengl as gl
import pytest
import trimesh
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from housemaker import viewer as viewer_module
from housemaker.glb import PreviewPlacedObject, convert_to_glb
from housemaker.landing_geometry import resolve_stair_section_image_points
from housemaker.models import LandingData, LevelData
from housemaker.viewer import (
    PLACED_OBJECT_MAX_SCALE,
    PLACED_OBJECT_MIN_SCALE,
    TRANSFORM_GIZMO_SCALE,
    GlbViewerWidget,
    _get_placed_object_scale_world_axis,
    _TransformGizmoHandle,
)
from tests import test_landing_main_integration as landing_main_tests

# ### Fixtures and radial camera rays ###
_application = QApplication.instance() or QApplication([])
_application.setQuitOnLastWindowClosed(False)
_UNIFORM_KIND = "uniform_scale"


@pytest.fixture
def landing_viewer():
    level = LevelData(index=2, name="Ground")
    landing = LandingData(
        center_x=100.0,
        center_y=120.0,
        rotation_degrees=27.0,
        scale_x=1.8,
        scale_y=0.6,
        scale_z=1.3,
        elevation_meters=0.5,
    )
    level.landings = [landing]
    viewer = GlbViewerWidget(window_editing_enabled=True)
    viewer.set_model(convert_to_glb([level]))
    viewer.set_landing_context([level])
    viewer.select_landing(landing.landing_id)
    assert viewer._toggle_placed_object_gizmo_mode()
    yield viewer, level, landing
    viewer.close()
    viewer.deleteLater()
    _application.processEvents()


def _uniform_geometry(viewer, landing):
    group = viewer._placed_object_render_groups[f"landing:{landing.landing_id}"]
    pivot = np.asarray(group.preview.world_position, dtype=float)
    axes = np.asarray(
        [_get_placed_object_scale_world_axis(group, index) for index in range(3)]
    )
    tips = pivot + axes * viewer._transform_gizmo_size
    center = tips.mean(axis=0)
    radius = float(np.linalg.norm(center - pivot))
    direction = (center - pivot) / radius
    return group, pivot, center, radius, direction


def _ray(point):
    offset = np.asarray((3.0, -5.0, 4.0))
    return np.asarray(point) + offset, -offset / np.linalg.norm(offset)


def _begin(viewer, landing):
    geometry = _uniform_geometry(viewer, landing)
    with patch.object(viewer.view, "build_camera_ray", return_value=_ray(geometry[2])):
        assert viewer._begin_placed_object_gizmo_drag(
            _TransformGizmoHandle(_UNIFORM_KIND, -1), QPointF()
        )
    return geometry


def _move(viewer, geometry, factor, *, finish=False):
    _group, _pivot, center, radius, direction = geometry
    point = center + direction * radius * (factor - 1.0)
    with patch.object(viewer.view, "build_camera_ray", return_value=_ray(point)):
        method = (
            viewer._finish_placed_object_gizmo_drag
            if finish
            else viewer._update_placed_object_gizmo_drag
        )
        return method(QPointF())


def _assert_common_factor(payload, original, factor):
    for name in ("scale_x", "scale_y", "scale_z"):
        assert payload[name] == pytest.approx(getattr(original, name) * factor)
    for name in ("center_x", "center_y", "rotation_degrees", "elevation_meters"):
        assert payload[name] == pytest.approx(getattr(original, name))


# ### Rendering, handle validation and picking ###
def test_uniform_scale_handle_is_the_only_axisless_transform_handle():
    assert viewer_module.TRANSFORM_GIZMO_UNIFORM_SCALE == _UNIFORM_KIND
    assert _TransformGizmoHandle(_UNIFORM_KIND, -1).axis_index == -1
    with pytest.raises(ValueError):
        _TransformGizmoHandle(TRANSFORM_GIZMO_SCALE, -1)
    with pytest.raises(ValueError):
        _TransformGizmoHandle(_UNIFORM_KIND, 0)


def test_purple_cube_is_at_centroid_of_three_axis_cube_tips(landing_viewer):
    viewer, _level, landing = landing_viewer
    assert len(viewer._transform_gizmo_items) == 7
    cubes = [
        item
        for item in viewer._transform_gizmo_items
        if isinstance(item, gl.GLMeshItem)
    ]
    assert len(cubes) == 4
    centers = [
        np.asarray(item.opts["meshdata"].vertexes()).mean(axis=0) for item in cubes
    ]
    np.testing.assert_allclose(centers[-1], np.mean(centers[:3], axis=0), atol=1e-6)
    color = cubes[-1].opts["color"]
    assert color[0] > color[1] and color[2] > color[1]
    _group, _pivot, center, _radius, _direction = _uniform_geometry(viewer, landing)
    np.testing.assert_allclose(centers[-1], center, atol=1e-6)


def test_centroid_cube_is_clickable_as_uniform_scale_not_a_single_axis(landing_viewer):
    viewer, _level, landing = landing_viewer
    _group, _pivot, center, _radius, _direction = _uniform_geometry(viewer, landing)
    handle = viewer._pick_transform_gizmo_handle(*_ray(center))
    assert handle == _TransformGizmoHandle(_UNIFORM_KIND, -1)


def test_xyz_cube_tips_keep_their_individual_axis_scale_picking(landing_viewer):
    viewer, _level, landing = landing_viewer
    group, pivot, _center, _radius, _direction = _uniform_geometry(viewer, landing)
    for index in range(3):
        tip = (
            pivot
            + _get_placed_object_scale_world_axis(group, index)
            * viewer._transform_gizmo_size
        )
        assert viewer._pick_transform_gizmo_handle(*_ray(tip)) == _TransformGizmoHandle(
            TRANSFORM_GIZMO_SCALE, index
        )


def test_normal_placed_objects_do_not_gain_the_landing_uniform_handle(landing_viewer):
    viewer, _level, _landing = landing_viewer
    preview = PreviewPlacedObject(
        "ordinary-chair",
        (trimesh.creation.box(),),
        np.eye(4),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    viewer.model.preview_placed_objects.append(preview)
    viewer._add_placed_object_items()
    assert viewer.select_placed_object(preview.object_id)
    assert viewer._toggle_placed_object_gizmo_mode()
    assert len(viewer._transform_gizmo_items) == 6
    assert (
        sum(isinstance(item, gl.GLMeshItem) for item in viewer._transform_gizmo_items)
        == 3
    )
    with patch.object(
        viewer.view, "build_camera_ray", return_value=_ray((0.2, 0.2, 0.2))
    ):
        assert not viewer._begin_placed_object_gizmo_drag(
            _TransformGizmoHandle(_UNIFORM_KIND, -1), QPointF()
        )


# ### Uniform drag factors, existing aspect ratios and bounds ###
def test_doubling_radial_handle_distance_doubles_every_existing_axis_scale(
    landing_viewer,
):
    viewer, _level, landing = landing_viewer
    edits, finished = [], []
    viewer.landing_transform_preview_changed.connect(
        lambda *values: edits.append(values)
    )
    viewer.landing_edit_finished.connect(lambda *values: finished.append(values))
    geometry = _begin(viewer, landing)

    assert _move(viewer, geometry, 2.0, finish=True)

    _assert_common_factor(edits[-1][1], landing, 2.0)
    assert finished == [(landing.landing_id, True)]
    assert not viewer.view.is_primary_pointer_drag_reserved


def test_repeated_uniform_previews_do_not_compound_the_scale_factor(landing_viewer):
    viewer, _level, landing = landing_viewer
    edits = []
    viewer.landing_transform_preview_changed.connect(
        lambda *values: edits.append(values)
    )
    geometry = _begin(viewer, landing)
    for factor in (1.25, 1.5, 2.0):
        assert _move(viewer, geometry, factor)
        _assert_common_factor(edits[-1][1], landing, factor)

    assert _move(viewer, geometry, 2.0, finish=True)

    _assert_common_factor(edits[-1][1], landing, 2.0)


@pytest.mark.parametrize("requested", [-100.0, 1_000_000.0])
def test_uniform_clamps_keep_positive_axes_and_preserve_nonuniform_aspect(
    landing_viewer, requested
):
    viewer, _level, landing = landing_viewer
    edits = []
    viewer.landing_transform_preview_changed.connect(
        lambda *values: edits.append(values)
    )
    geometry = _begin(viewer, landing)
    assert _move(viewer, geometry, requested, finish=True)
    payload = edits[-1][1]
    scales = [payload[name] for name in ("scale_x", "scale_y", "scale_z")]
    assert min(scales) >= PLACED_OBJECT_MIN_SCALE - 1e-9
    assert max(scales) <= PLACED_OBJECT_MAX_SCALE + 1e-9
    factor = scales[0] / landing.scale_x
    _assert_common_factor(payload, landing, factor)


def test_uniform_drag_freezes_initial_gizmo_radius_for_its_entire_gesture(
    landing_viewer,
):
    viewer, _level, landing = landing_viewer
    edits = []
    viewer.landing_transform_preview_changed.connect(
        lambda *values: edits.append(values)
    )
    geometry = _begin(viewer, landing)
    viewer._transform_gizmo_size *= 10.0

    assert _move(viewer, geometry, 2.0, finish=True)

    _assert_common_factor(edits[-1][1], landing, 2.0)


def test_cancel_uniform_drag_restores_original_transform_and_emits_cancel(
    landing_viewer,
):
    viewer, _level, landing = landing_viewer
    cancelled = []
    viewer.landing_edit_cancelled.connect(cancelled.append)
    geometry = _begin(viewer, landing)
    before = geometry[0].current_transform.copy()
    assert _move(viewer, geometry, 1.5)

    viewer._cancel_placed_object_gizmo_drag()

    np.testing.assert_array_equal(geometry[0].current_transform, before)
    assert cancelled == [landing.landing_id]
    assert not viewer.view.is_primary_pointer_drag_reserved


# ### Full scene transactions and landing-bound stairs ###
@pytest.fixture
def main_landing():
    fixture = landing_main_tests.LandingMainIntegrationTests()
    fixture.setUp()
    landing = replace(
        fixture._add(), scale_x=1.6, scale_y=1.2, scale_z=0.8, rotation_degrees=25.0
    )
    fixture.level.landings[0] = landing
    stair = fixture._attach_stair(landing)
    fixture.workspace._reset_viewer_landing_snapshots()
    viewer = fixture.workspace.viewer
    viewer.set_model(
        convert_to_glb(fixture.workspace.levels, stairs=fixture.workspace.stairs)
    )
    viewer.set_landing_context(fixture.workspace.levels, landing.landing_id)
    viewer.select_landing(landing.landing_id)
    assert viewer._toggle_placed_object_gizmo_mode()
    fixture.rebuild.reset_mock()
    yield fixture, landing, stair
    fixture.tearDown()


def test_uniform_scale_updates_bound_stairs_live_then_delays_mesh_and_undo_restores(
    main_landing,
):
    fixture, landing, stair = main_landing
    workspace, viewer = fixture.workspace, fixture.workspace.viewer
    before_points = resolve_stair_section_image_points(
        fixture.level, stair.start_section
    )
    geometry = _begin(viewer, landing)
    assert _move(viewer, geometry, 1.5)
    updated = fixture.level.landings[0]
    for name in ("scale_x", "scale_y", "scale_z"):
        assert getattr(updated, name) == pytest.approx(getattr(landing, name) * 1.5)
    after_points = resolve_stair_section_image_points(
        fixture.level, stair.start_section
    )
    assert after_points != before_points
    assert not workspace._landing_mesh_update_timer.isActive()
    fixture.rebuild.assert_not_called()

    assert _move(viewer, geometry, 1.5, finish=True)

    assert workspace._landing_mesh_update_timer.isActive()
    assert len(workspace._canvas_undo_stack) == 1
    fixture.rebuild.assert_not_called()
    workspace._commit_landing_mesh_update()
    fixture.rebuild.assert_called_once()
    workspace._handle_canvas_undo_requested()
    assert fixture.level.landings == [landing]
    assert workspace.stairs == [stair]
    assert (
        resolve_stair_section_image_points(fixture.level, stair.start_section)
        == before_points
    )


def test_cancel_uniform_scale_restores_bound_stairs_without_a_mesh_rebuild(
    main_landing,
):
    fixture, landing, stair = main_landing
    before_points = resolve_stair_section_image_points(
        fixture.level, stair.start_section
    )
    viewer = fixture.workspace.viewer
    geometry = _begin(viewer, landing)
    assert _move(viewer, geometry, 1.5)

    viewer._cancel_placed_object_gizmo_drag()

    assert fixture.level.landings == [landing]
    assert fixture.workspace.stairs == [stair]
    assert (
        resolve_stair_section_image_points(fixture.level, stair.start_section)
        == before_points
    )
    assert not fixture.workspace._landing_mesh_update_timer.isActive()
    assert fixture.workspace._canvas_undo_stack == []
    fixture.rebuild.assert_not_called()
