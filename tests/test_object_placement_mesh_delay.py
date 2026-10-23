# ### Environment ###
from __future__ import annotations

import os
from contextlib import ExitStack
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ### Imports ###
import numpy as np
import pytest
import trimesh
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.generation_state import GeneratedObjectPlacement, GenerationData
from housemaker.glb import GeneratedModel, compose_placed_generated_models_preview
from housemaker.main import BlueprintWorkspace
from housemaker.viewer import (
    OBJECT_PLACEMENT_PREVIEW_COLOR,
    TRANSFORM_GIZMO_TRANSLATE,
    PlacedObjectInstanceRequest,
    SceneObjectPlacementCandidate,
    _TransformGizmoHandle,
)
from tests import test_generated_object_placement_main as placement_fixtures

# ### Fixtures and assertions ###
_application = QApplication.instance() or QApplication([])
_application.setQuitOnLastWindowClosed(False)


@pytest.fixture
def workspace(tmp_path):
    """Use real controllers and tiny models, never jobs or a scene rebuild."""

    with ExitStack() as stack:
        refresh = stack.enter_context(
            patch.object(BlueprintWorkspace, "_schedule_viewer_preview_refresh")
        )
        widget = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(tmp_path / "settings.json")
        )
        widget._set_mesh_edit_update_delay_seconds(60.0)
        widget.levels = [placement_fixtures._level(2)]
        widget.current_level_index = 0
        widget.canvas.set_stair_context((), widget.current_level)
        widget.generation._asset_directory = tmp_path / "generated"
        model = placement_fixtures._generated_model((0.6, 0.8, 1.0))
        for method in ("get_generated_object_model", "get_generated_object_preview_model"):
            stack.enter_context(patch.object(widget.generation, method, return_value=model))
        widget.generation.set_data(
            GenerationData(
                generated_objects=[
                    placement_fixtures._record("chair", None),
                    placement_fixtures._record("table", None),
                ]
            )
        )
        refresh.reset_mock()
        try:
            yield widget
        finally:
            widget.shutdown()
            widget.close()
            widget.deleteLater()
            _application.processEvents()


def _candidate(count=1):
    return SceneObjectPlacementCandidate(
        level_index=2,
        world_positions=tuple((0.5 + index, 0.75, 0.2) for index in range(count)),
    )


def _place_group(workspace, object_ids=("chair",)):
    assert workspace._commit_placeable_object_group(object_ids, _candidate(len(object_ids)))


def _assert_pending(workspace, object_ids):
    assert workspace._pending_object_placement_ids == set(object_ids)
    assert set(workspace.viewer.get_pending_placed_object_ids()) == set(object_ids)
    expected_color = np.rint(np.asarray(OBJECT_PLACEMENT_PREVIEW_COLOR) * 255).astype(
        np.uint8
    )
    for preview in workspace.viewer.get_pending_placed_object_previews():
        for mesh in preview.meshes:
            assert mesh.visual.kind == "face"
            assert np.all(mesh.visual.face_colors == expected_color)
    assert workspace._object_placement_mesh_update_timer.isActive()
    assert workspace._has_pending_mesh_updates()
    workspace._refresh_confirm_mesh_updates_button()
    assert workspace.confirm_mesh_updates_button.isEnabled()
    assert workspace.confirm_mesh_updates_button.property("mesh_updates_pending") is True


def _assert_no_pending_preview(workspace):
    assert not workspace._pending_object_placement_ids
    assert not workspace._committing_object_placement_revisions
    assert workspace.viewer.get_pending_placed_object_ids() == ()
    assert not workspace._object_placement_mesh_update_timer.isActive()


# ### Placement transactions ###
@pytest.mark.parametrize("object_ids", (("chair",), ("chair", "table")))
def test_place_persists_immediately_with_editable_pending_preview(workspace, object_ids):
    undo_count = len(workspace._canvas_undo_stack)

    _place_group(workspace, object_ids)

    workspace._schedule_viewer_preview_refresh.assert_not_called()
    _assert_pending(workspace, object_ids)
    assert len(workspace._canvas_undo_stack) == undo_count + 1
    saved = workspace.generation.get_data()
    saved_records = {record.object_id: record for record in saved.generated_objects}
    for object_id in object_ids:
        placement = workspace.generation.get_scene_object_placement(object_id)
        assert placement is not None
        assert saved_records[object_id].placement == placement
        assert object_id in workspace.viewer._placed_object_render_groups
    workspace.viewer.set_selected_placed_object_ids(
        object_ids, active_object_id=object_ids[-1]
    )
    assert workspace.viewer.get_selected_placed_object_ids() == object_ids
    assert workspace.viewer._transform_gizmo_items
    assert {
        profile.object_id for profile in workspace.canvas.get_placed_object_profiles()
    } == set(object_ids)


def test_generation_owned_atlas_place_uses_the_same_delay(workspace):
    with patch.object(workspace, "_ensure_viewer_preview_current"):
        assert workspace.generation.request_generated_object_placement("chair")
    session = workspace._direct_object_placement_session
    assert session is not None
    assert session.generation_request_token is not None
    placement = GeneratedObjectPlacement(2, 50.0, 40.0)
    workspace._schedule_viewer_preview_refresh.reset_mock()

    assert workspace._commit_generation_owned_placement_request(
        session.generation_request_token, placement
    )

    assert workspace.generation.get_scene_object_placement("chair") == placement
    workspace._schedule_viewer_preview_refresh.assert_not_called()
    _assert_pending(workspace, ("chair",))


@pytest.mark.parametrize("path", ("atlas", "gizmo"))
def test_instance_place_is_persistent_but_waits_for_mesh_update(workspace, path):
    source_placement = GeneratedObjectPlacement(
        2, 10.0, 20.0, rotation_degrees=(0.0, 0.0, 35.0), scale=1.2
    )
    assert workspace.generation.restore_generated_object_placement(
        "chair", source_placement, emit_change_signals=False
    )
    if path == "atlas":
        assert workspace._commit_generated_object_instance_placement("chair", _candidate())
    else:
        workspace._handle_placed_object_instance_requested(
            PlacedObjectInstanceRequest(
                source_scene_object_id="chair",
                world_position=(0.5, 0.75, 0.2),
                rotation_degrees=source_placement.rotation_degrees,
                scale=source_placement.scale,
                axis_scales=source_placement.axis_scales,
                world_transform=trimesh.transformations.translation_matrix((0.5, 0.75, 0.2)),
            )
        )

    instances = workspace.generation.get_generated_object_instances()
    assert len(instances) == 1
    instance = instances[0]
    assert instance.source_object_id == "chair"
    assert workspace.generation.get_data().object_instances == [instance]
    workspace._schedule_viewer_preview_refresh.assert_not_called()
    _assert_pending(workspace, (instance.instance_id,))
    assert workspace.viewer.get_selected_placed_object_ids() == (instance.instance_id,)
    assert workspace.viewer._transform_gizmo_items


# ### Quiet-period restart and immediate confirmation ###
@pytest.mark.parametrize("edit", ("position", "canvas", "scale", "axis_scale"))
def test_editing_pending_placement_restarts_delay_without_a_heavy_refresh(workspace, edit):
    _place_group(workspace)
    timer = workspace._object_placement_mesh_update_timer
    previous = workspace.generation.get_scene_object_placement("chair")
    assert previous is not None
    with patch.object(timer, "start", wraps=timer.start) as restart:
        if edit == "position":
            workspace._handle_placed_object_transform_changed(
                "chair", (1.5, 1.75, 0.4), (0.0, 0.0, 45.0)
            )
        elif edit == "canvas":
            workspace._handle_blueprint_placed_object_transform_committed(
                "chair", previous.image_x + 20.0, previous.image_y + 10.0, 35.0
            )
        elif edit == "scale":
            workspace._handle_placed_object_scales_changed((("chair", 1.5),))
        else:
            workspace._handle_placed_object_axis_scales_changed("chair", (1.2, 1.3, 1.4))

    assert workspace.generation.get_scene_object_placement("chair") != previous
    assert restart.call_count >= 1
    workspace._schedule_viewer_preview_refresh.assert_not_called()
    _assert_pending(workspace, ("chair",))


def test_shared_delay_setting_restarts_the_object_timer(workspace):
    _place_group(workspace)
    timer = workspace._object_placement_mesh_update_timer

    workspace._set_mesh_edit_update_delay_seconds(12.5)

    assert timer.interval() == 12500
    assert timer.isActive()
    assert workspace._mesh_edit_update_delay_seconds == 12.5
    workspace._schedule_viewer_preview_refresh.assert_not_called()


def test_timeout_waits_while_a_real_pending_object_gizmo_drag_is_active(workspace):
    _place_group(workspace)
    viewer = workspace.viewer
    preview = viewer.get_pending_placed_object_previews()[0]
    ray = (
        np.asarray(preview.world_position) + (0.0, 0.0, 10.0),
        np.asarray((0.0, 0.0, -1.0)),
    )
    with patch.object(viewer.view, "build_camera_ray", return_value=ray):
        assert viewer._begin_placed_object_gizmo_drag(
            _TransformGizmoHandle(TRANSFORM_GIZMO_TRANSLATE, 0), QPointF(20.0, 20.0)
        )
    assert viewer._placed_object_transform_drag is not None
    assert viewer.is_placed_object_interaction_active

    try:
        workspace._object_placement_mesh_update_timer.timeout.emit()
        workspace._schedule_viewer_preview_refresh.assert_not_called()
        _assert_pending(workspace, ("chair",))
    finally:
        viewer._cancel_placed_object_gizmo_drag()

    workspace._object_placement_mesh_update_timer.timeout.emit()
    workspace._schedule_viewer_preview_refresh.assert_called_once_with(preserve_camera=True)


@pytest.mark.parametrize("trigger", ("timer", "confirm"))
def test_pending_mesh_is_rebuilt_only_on_timeout_or_confirm(workspace, trigger):
    _place_group(workspace, ("chair", "table"))
    workspace._schedule_viewer_preview_refresh.assert_not_called()

    if trigger == "timer":
        workspace._object_placement_mesh_update_timer.timeout.emit()
    else:
        workspace._refresh_confirm_mesh_updates_button()
        workspace.confirm_mesh_updates_button.click()

    workspace._schedule_viewer_preview_refresh.assert_called_once_with(preserve_camera=True)
    assert not workspace._pending_object_placement_ids
    assert not workspace._object_placement_mesh_update_timer.isActive()
    # The green objects must survive while the asynchronous scene is building.
    assert set(workspace.viewer.get_pending_placed_object_ids()) == {"chair", "table"}
    assert set(workspace._committing_object_placement_revisions) == {"chair", "table"}
    assert workspace._mesh_edit_update_delay_seconds == 60.0

    workspace._canvas_viewer_preview_revision = workspace._viewer_preview_revision
    workspace._clear_committed_object_placement_previews_if_displayed()

    _assert_no_pending_preview(workspace)


def test_final_model_replaces_green_preview_only_at_its_displayed_revision(workspace):
    _place_group(workspace, ("chair", "table"))
    workspace._commit_pending_object_placement_mesh_update()
    target_revision = max(workspace._committing_object_placement_revisions.values())
    workspace._canvas_viewer_preview_revision = target_revision - 1
    workspace._clear_committed_object_placement_previews_if_displayed()
    assert set(workspace.viewer.get_pending_placed_object_ids()) == {"chair", "table"}
    placements = workspace._build_placed_generated_models(
        include_instances=True,
        include_door_placements=False,
        preview_only=True,
        precomputed_architectural_trim_models=(),
    )
    final_model = compose_placed_generated_models_preview(
        GeneratedModel(mesh=trimesh.Trimesh(), scene=trimesh.Scene(), glb_bytes=b""),
        placements,
    )

    workspace.viewer.set_model(final_model, preserve_camera=True)
    workspace._canvas_viewer_preview_revision = target_revision
    workspace._clear_committed_object_placement_previews_if_displayed()

    _assert_no_pending_preview(workspace)
    assert set(workspace.viewer._placed_object_render_groups) == {"chair", "table"}
    for group in workspace.viewer._placed_object_render_groups.values():
        assert group.preview in final_model.preview_placed_objects


# ### Removal, undo, and lifecycle ###
@pytest.mark.parametrize("object_ids", (("chair",), ("chair", "table")))
def test_undo_before_timeout_removes_every_pending_object_without_ghosts(
    workspace, object_ids
):
    _place_group(workspace, object_ids)

    workspace._handle_canvas_undo_requested()

    assert all(
        workspace.generation.get_scene_object_placement(object_id) is None
        for object_id in object_ids
    )
    _assert_no_pending_preview(workspace)
    assert not set(object_ids).intersection(workspace.viewer._placed_object_render_groups)


def test_remove_pending_placement_cancels_preview_and_keeps_reusable_asset(workspace):
    _place_group(workspace)

    workspace._handle_placed_object_removal_requested("chair")

    assert workspace.generation.get_scene_object_placement("chair") is None
    assert "chair" in workspace.generation.get_generated_object_ids()
    _assert_no_pending_preview(workspace)
    assert "chair" not in workspace.viewer._placed_object_render_groups


def test_undo_pending_instance_does_not_remove_its_source(workspace):
    source = GeneratedObjectPlacement(2, 10.0, 20.0)
    assert workspace.generation.restore_generated_object_placement(
        "chair", source, emit_change_signals=False
    )
    assert workspace._commit_generated_object_instance_placement("chair", _candidate())
    instance_id = workspace.generation.get_generated_object_instances()[0].instance_id

    workspace._handle_canvas_undo_requested()

    assert workspace.generation.get_scene_object_placement("chair") == source
    assert workspace.generation.get_generated_object_instance(instance_id) is None
    _assert_no_pending_preview(workspace)
    assert instance_id not in workspace.viewer._placed_object_render_groups


def test_shutdown_stops_pending_placement_timer(workspace):
    _place_group(workspace)

    workspace.shutdown()

    assert not workspace._object_placement_mesh_update_timer.isActive()
    assert not workspace._pending_object_placement_ids
    assert workspace.viewer.get_pending_placed_object_ids() == ()
