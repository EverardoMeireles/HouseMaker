"""Delayed object placements remain editable without rebuilding the house model."""

# ### Environment setup ###
from __future__ import annotations

import os
from dataclasses import replace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ### Imports ###
import numpy as np
import pytest
import trimesh
from PIL import Image
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from housemaker.camera_models import CameraPose
from housemaker.glb import GeneratedModel, PreviewPlacedObject
from housemaker.viewer import (
    OBJECT_PLACEMENT_PREVIEW_COLOR,
    GlbViewerWidget,
    _build_placed_object_scaled_rotation,
    _TransformGizmoHandle,
)

# ### Fixtures ###
_application = QApplication.instance() or QApplication([])
_application.setQuitOnLastWindowClosed(False)


def _preview(
    object_id="object-a",
    *,
    position=(0.0, 0.0, 0.0),
    rotation=(0.0, 0.0, 0.0),
    scale=1.0,
    axis_scales=(1.0, 1.0, 1.0),
    symmetric=False,
    textured=False,
):
    mesh = trimesh.creation.box(extents=(0.8, 1.0, 1.0))
    mesh.apply_translation((0.4 if symmetric else 0.0, 0.0, 0.5))
    if textured:
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=np.zeros((len(mesh.vertices), 2)),
            image=Image.new("RGBA", (2, 2), (20, 30, 220, 255)),
        )
    transform = np.eye(4)
    transform[:3, :3] = _build_placed_object_scaled_rotation(
        rotation, scale, axis_scales
    )
    transform[:3, 3] = position
    return PreviewPlacedObject(
        object_id=object_id,
        meshes=(mesh,),
        placement_transform=transform,
        world_position=position,
        rotation_degrees=rotation,
        scale=scale,
        axis_scales=axis_scales,
        source_object_id="source-a",
        symmetric_preview_orientation="vertical" if symmetric else None,
        symmetric_preview_plane_coordinate=0.0 if symmetric else None,
    )


def _model(*objects):
    mesh = trimesh.creation.box(extents=(0.01, 0.01, 0.01))
    mesh.apply_translation((100, 100, 100))
    return GeneratedModel(
        mesh=mesh,
        scene=trimesh.Scene(mesh),
        glb_bytes=b"",
        preview_base_mesh=mesh.copy(),
        preview_placed_objects=list(objects),
    )


def _ray(x=0.0):
    return np.asarray((x, -5.0, 0.0)), np.asarray((0.0, 1.0, 0.0))


@pytest.fixture
def viewer():
    widget = GlbViewerWidget(window_editing_enabled=True)
    yield widget
    widget.exit_first_person_mode()
    widget.close()
    widget.deleteLater()
    _application.processEvents()


# ### Independent editable retained models ###
def test_pending_object_without_house_model_has_green_geometry_and_real_gizmos(viewer):
    original = _preview(
        position=(2.0, 3.0, 1.0),
        rotation=(15.0, -20.0, 37.0),
        scale=1.4,
        axis_scales=(1.2, 0.7, 1.8),
        textured=True,
    )
    original_vertices = original.meshes[0].vertices.copy()

    assert viewer.set_pending_placed_object_previews((original,))
    assert viewer.model is None
    assert viewer.get_pending_placed_object_ids() == (original.object_id,)
    assert viewer.get_visible_placed_object_ids() == (original.object_id,)
    assert viewer.select_placed_object(original.object_id)
    assert viewer._transform_gizmo_items
    assert viewer._placed_object_instance_gizmo_items
    group = viewer._placed_object_render_groups[original.object_id]
    np.testing.assert_allclose(group.current_transform, original.placement_transform)
    assert group.preview.scale == original.scale
    assert group.preview.axis_scales == original.axis_scales
    assert group.preview.source_object_id == original.source_object_id
    assert all(part.textured_item is None for part in group.retained_parts)
    np.testing.assert_allclose(
        group.preview.meshes[0].visual.face_colors / 255.0,
        np.tile(OBJECT_PLACEMENT_PREVIEW_COLOR, (len(original.meshes[0].faces), 1)),
        atol=1.0 / 255.0,
    )
    assert original.meshes[0].visual.kind == "texture"
    np.testing.assert_array_equal(original.meshes[0].vertices, original_vertices)


def test_pending_models_are_incremental_and_preserve_fps_camera(viewer):
    original = _preview()
    viewer.enter_first_person_mode()
    pose = CameraPose(2.0, -3.0, 1.7, yaw_degrees=23.0, pitch_degrees=5.0)
    viewer.set_first_person_camera_pose(pose)
    with patch.object(
        viewer, "_populate_scene", side_effect=AssertionError("house rebuilt")
    ):
        viewer.set_pending_placed_object_previews((original,))
        first_root = viewer._placed_object_render_groups[original.object_id].root_item
        viewer.select_placed_object(original.object_id)
        second = _preview("instance-a", position=(3.0, 0.0, 0.0))
        viewer.set_pending_placed_object_previews(
            (*viewer.get_pending_placed_object_previews(), second)
        )
    assert (
        viewer._placed_object_render_groups[original.object_id].root_item is first_root
    )
    assert viewer.get_selected_placed_object_ids() == (original.object_id,)
    assert viewer.get_pending_placed_object_ids() == (
        original.object_id,
        second.object_id,
    )
    assert viewer.is_first_person_active
    assert viewer.get_first_person_camera_pose() == pose


def test_pending_override_never_mutates_committed_model_and_clears_without_duplicates(
    viewer,
):
    committed = _preview(position=(-2.0, 0.0, 0.0), textured=True)
    untouched = _preview("other-object", position=(5.0, 0.0, 0.0))
    model = _model(committed, untouched)
    viewer.set_model(model)
    other_root = viewer._placed_object_render_groups[untouched.object_id].root_item
    pending_transform = committed.placement_transform.copy()
    pending_transform[:3, 3] = (2.0, 0.0, 0.0)
    pending = replace(
        committed, world_position=(2.0, 0.0, 0.0), placement_transform=pending_transform
    )

    viewer.set_pending_placed_object_previews((pending,))

    assert len(viewer._placed_object_render_groups) == 2
    assert (
        viewer._placed_object_render_groups[untouched.object_id].root_item is other_root
    )
    assert model.preview_placed_objects[0] is committed
    assert viewer.select_placed_object(committed.object_id)
    assert viewer.clear_pending_placed_object_previews()
    assert viewer.get_pending_placed_object_ids() == ()
    assert len(viewer._placed_object_render_groups) == 2
    assert viewer._placed_object_render_groups[committed.object_id].preview is committed
    assert viewer.get_selected_placed_object_ids() == (committed.object_id,)


def test_pending_transform_edit_survives_model_refresh_and_final_material_handoff(
    viewer,
):
    original = _preview(scale=1.4, textured=True)
    committed_model = _model(original)
    viewer.set_model(committed_model)
    viewer.set_pending_placed_object_previews((original,))
    viewer.select_placed_object(original.object_id)
    position, rotation, scales = (4.0, -2.0, 1.5), (15.0, 23.0, -47.0), (0.8, 1.7, 1.2)

    assert viewer.apply_placed_object_full_transform_preview(
        original.object_id, position, rotation, scales
    )
    pending = viewer.get_pending_placed_object_previews()[0]
    assert pending.world_position == position
    assert pending.rotation_degrees == rotation
    assert pending.scale == original.scale
    assert pending.axis_scales == scales
    assert committed_model.preview_placed_objects[0] is original
    final_preview = replace(
        original,
        placement_transform=pending.placement_transform,
        world_position=position,
        rotation_degrees=rotation,
        axis_scales=scales,
    )
    final_model = _model(final_preview)
    viewer.set_model(final_model, preserve_camera=True)

    assert viewer.get_pending_placed_object_ids() == (original.object_id,)
    group = viewer._placed_object_render_groups[original.object_id]
    np.testing.assert_allclose(group.current_transform, pending.placement_transform)
    assert all(part.textured_item is None for part in group.retained_parts)
    assert viewer.get_selected_placed_object_ids() == (original.object_id,)
    viewer.clear_pending_placed_object_previews()
    group = viewer._placed_object_render_groups[original.object_id]
    assert group.preview is final_preview
    assert any(part.textured_item is not None for part in group.retained_parts)
    assert len(viewer._placed_object_render_groups) == 1


def test_pending_transform_edits_preserve_a_non_origin_local_pivot(viewer):
    original = _preview(
        position=(2.0, 3.0, 1.0),
        rotation=(10.0, 20.0, 30.0),
        scale=1.4,
        axis_scales=(1.2, 0.8, 1.5),
    )
    pivot = np.asarray((0.2, -0.3, 0.4))
    transform = original.placement_transform.copy()
    transform[:3, 3] -= transform[:3, :3] @ pivot
    original = replace(original, placement_transform=transform)
    viewer.set_pending_placed_object_previews((original,))
    position, rotation, scales = (4.0, -1.0, 2.0), (17.0, -29.0, 45.0), (0.6, 1.3, 1.7)

    assert viewer.apply_placed_object_full_transform_preview(
        original.object_id, position, rotation, scales
    )

    expected = np.eye(4)
    expected[:3, :3] = _build_placed_object_scaled_rotation(
        rotation, original.scale, scales
    )
    expected[:3, 3] = np.asarray(position) - expected[:3, :3] @ pivot
    np.testing.assert_allclose(
        viewer.get_pending_placed_object_previews()[0].placement_transform,
        expected,
    )


# ### Existing edit, pick and clone paths apply to pending objects ###
def test_pending_model_uses_normal_cpu_object_selection(viewer):
    pending = _preview()
    viewer.set_pending_placed_object_previews((pending,))
    with patch.object(viewer.view, "build_camera_ray", return_value=_ray()):
        viewer._handle_window_wall_pick_requested(QPointF())
    assert viewer.get_selected_placed_object_ids() == (pending.object_id,)


def test_pending_translation_gizmo_emits_commit_and_keeps_edited_baseline(viewer):
    pending = _preview()
    viewer.set_pending_placed_object_previews((pending,))
    viewer.select_placed_object(pending.object_id)
    emitted = []
    viewer.placed_object_transform_changed.connect(lambda *args: emitted.append(args))
    handle = _TransformGizmoHandle("translate", 0)
    with patch.object(viewer.view, "build_camera_ray", return_value=_ray()):
        assert viewer._begin_placed_object_gizmo_drag(handle, QPointF())
    assert viewer.is_placed_object_interaction_active
    with patch.object(viewer.view, "build_camera_ray", return_value=_ray(2.0)):
        assert viewer._finish_placed_object_gizmo_drag(QPointF())
    assert not viewer.is_placed_object_interaction_active
    assert len(emitted) == 1
    assert emitted[0][0] == pending.object_id
    assert viewer.get_pending_placed_object_previews()[0].world_position == (
        2.0,
        0.0,
        0.0,
    )
    viewer._populate_scene()
    assert viewer.get_pending_placed_object_previews()[0].world_position == (
        2.0,
        0.0,
        0.0,
    )
    assert viewer.get_selected_placed_object_ids() == (pending.object_id,)


def test_pending_instance_drag_reports_active_without_moving_source(viewer):
    pending = _preview(rotation=(10.0, 20.0, 30.0), axis_scales=(0.7, 1.1, 1.5))
    viewer.set_pending_placed_object_previews((pending,))
    viewer.select_placed_object(pending.object_id)
    ray = (np.asarray((1.0, 1.0, 5.0)), np.asarray((0.0, 0.0, -1.0)))
    with patch.object(viewer.view, "build_camera_ray", return_value=ray):
        assert viewer._begin_placed_object_instance_drag(QPointF())
    assert viewer.is_placed_object_interaction_active
    viewer._cancel_placed_object_instance_drag()
    assert not viewer.is_placed_object_interaction_active
    np.testing.assert_allclose(
        viewer.get_pending_placed_object_previews()[0].placement_transform,
        pending.placement_transform,
    )


def test_pending_half_mesh_shows_the_mirrored_part_without_texture_or_fade(viewer):
    pending = _preview(symmetric=True, textured=True)
    viewer.set_pending_placed_object_previews((pending,))
    group = viewer._placed_object_render_groups[pending.object_id]
    assert len(group.symmetric_groups) == 1
    assert len(group.pick_meshes) == 2
    mirror = group.symmetric_groups[0]
    assert mirror.textured_item is None
    assert not mirror.fade_enabled
    assert not viewer._symmetric_preview_timer.isActive()
    assert min(mesh.bounds[0, 0] for mesh in group.pick_meshes) < 0.0


# ### Authoritative map validation and retirement ###
def test_pending_map_idempotence_removal_and_reset(viewer):
    first, second = _preview(), _preview("instance-a", position=(2.0, 0.0, 0.0))
    viewer.set_pending_placed_object_previews((first, second))
    first_root = viewer._placed_object_render_groups[first.object_id].root_item
    assert not viewer.set_pending_placed_object_previews((first, second))
    assert not viewer.set_pending_placed_object_previews(
        viewer.get_pending_placed_object_previews()
    )
    assert viewer._placed_object_render_groups[first.object_id].root_item is first_root
    assert viewer.set_pending_placed_object_previews((first,))
    assert second.object_id not in viewer._placed_object_render_groups
    viewer.clear_model()
    assert viewer.get_pending_placed_object_ids() == ()
    assert viewer._placed_object_render_groups == {}


def test_pending_map_validation_never_partially_replaces_existing_objects(viewer):
    pending = _preview()
    viewer.set_pending_placed_object_previews((pending,))
    with pytest.raises(ValueError, match="unique"):
        viewer.set_pending_placed_object_previews((pending, pending))
    with pytest.raises(TypeError, match="PreviewPlacedObject"):
        viewer.set_pending_placed_object_previews((pending, object()))
    assert viewer.get_pending_placed_object_ids() == (pending.object_id,)
