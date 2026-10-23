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
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.generation_state import (
    GeneratedObjectInstance,
    GeneratedObjectPlacement,
    GenerationData,
)
from housemaker.global_scale import GlobalScale
from housemaker.level_coordinates import level_image_to_world_xy
from housemaker.main import BlueprintWorkspace
from housemaker.object_symmetry import SymmetricDivisionMetadata
from housemaker.surface_geometry import SURFACE_TYPE_FLOOR, FixedSurface
from tests import test_generated_object_placement_main as fixtures

# ### Fixtures and pointer helpers ###
_application = QApplication.instance() or QApplication([])
_application.setQuitOnLastWindowClosed(False)


@pytest.fixture
def workspace(tmp_path):
    with ExitStack() as stack:
        stack.enter_context(patch.object(BlueprintWorkspace, "_schedule_viewer_preview_refresh"))
        stack.enter_context(patch.object(BlueprintWorkspace, "_ensure_viewer_preview_current"))
        widget = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(tmp_path / "settings.json")
        )
        widget.resize(1600, 1000)
        widget._set_mesh_edit_update_delay_seconds(60.0)
        widget.levels = [fixtures._level(2)]
        widget.current_level_index = 0
        widget.canvas.set_stair_context((), widget.current_level)
        widget.generation._asset_directory = tmp_path / "generated"
        widget.generation._asset_directory.mkdir()
        models = {
            "chair": fixtures._generated_model((2.0, 1.0, 1.0)),
            "desk": fixtures._generated_model((1.0, 3.0, 1.0)),
            "lamp": fixtures._generated_model((0.5, 0.5, 2.0)),
            "table": fixtures._generated_model((3.0, 2.0, 1.0)),
        }
        for object_id, model in models.items():
            (widget.generation._asset_directory / f"{object_id}.glb").write_bytes(model.glb_bytes)
        for method in ("get_generated_object_model", "get_generated_object_preview_model"):
            stack.enter_context(patch.object(
                widget.generation, method, side_effect=lambda object_id, *_args: models.get(object_id)
            ))
        stack.enter_context(patch.object(
            widget, "_get_scene_generated_object_model",
            side_effect=lambda object_id, **_kwargs: models.get(object_id),
        ))
        widget.generation.set_data(GenerationData(
            generated_objects=[fixtures._record(object_id, None) for object_id in models]
        ))
        widget.texture_atlas_workspace.set_object_texture_sources(
            (), placeable_objects={object_id: object_id for object_id in models}
        )
        widget.show()
        widget.workspace_tabs.setCurrentIndex(widget.atlas_workspace_tab_index)
        _application.processEvents()
        widget._schedule_viewer_preview_refresh.reset_mock()
        try:
            yield widget
        finally:
            widget.shutdown()
            widget.close()
            widget.deleteLater()
            _application.processEvents()


def _select_atlas_rows(workspace, last_row, modifier):
    source_list = workspace.texture_atlas_workspace.object_list
    QTest.mouseClick(source_list.viewport(), Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier,
                     source_list.visualItemRect(source_list.item(0)).center())
    QTest.mouseClick(source_list.viewport(), Qt.MouseButton.LeftButton, modifier,
                     source_list.visualItemRect(source_list.item(last_row)).center())
    return workspace.texture_atlas_workspace.selected_object_texture_ids


def _floor_click(workspace):
    mesh = trimesh.Trimesh(
        vertices=((-50, -50, 0), (50, -50, 0), (50, 50, 0), (-50, 50, 0)),
        faces=((0, 1, 2), (0, 2, 3)), process=False,
    )
    floor = FixedSurface(
        surface_id="level:2/floor:test", surface_type=SURFACE_TYPE_FLOOR,
        level_index=2, room_index=None, mesh=mesh, area_square_meters=10000,
    )
    workspace.viewer.set_canvas_scene_levels(((2, "Level 2"),), reset_visibility=True)
    workspace.viewer._canvas_surface_targets = {floor.surface_id: floor}
    with patch.object(workspace.viewer.view, "build_camera_ray", return_value=(
        np.asarray((10.0, 10.0, 5.0)), np.asarray((0.0, 0.0, -1.0)),
    )):
        for kind, buttons in (
            (QEvent.Type.MouseButtonPress, Qt.MouseButton.LeftButton),
            (QEvent.Type.MouseButtonRelease, Qt.MouseButton.NoButton),
        ):
            event = QMouseEvent(kind, QPointF(100, 100), QPointF(100, 100),
                                Qt.MouseButton.LeftButton, buttons,
                                Qt.KeyboardModifier.NoModifier)
            QApplication.sendEvent(workspace.viewer.view, event)


# ### Real Atlas selection, floor placement and undo ###
@pytest.mark.parametrize("modifier,last_row,count", (
    (Qt.KeyboardModifier.ControlModifier, 2, 2),
    (Qt.KeyboardModifier.ShiftModifier, 3, 4),
))
def test_atlas_multi_select_places_one_grid_with_pending_meshes_and_one_undo(
    workspace, modifier, last_row, count
):
    selected = _select_atlas_rows(workspace, last_row, modifier)
    assert len(selected) == count
    before = workspace.generation.get_data()
    undo_count = len(workspace._canvas_undo_stack)
    workspace.texture_atlas_workspace.place_assign_button.click()
    session = workspace._direct_object_placement_session
    assert session is not None and session.placeable_ids == selected
    assert workspace.viewer.is_object_placement_active
    assert workspace.generation.get_data() == before
    assert len(workspace.viewer._object_placement_preview_meshes) == count
    offsets = tuple(value.copy() for value in workspace.viewer._object_placement_preview_offsets)
    _floor_click(workspace)
    assert workspace._direct_object_placement_session is None
    assert not workspace.viewer.is_object_placement_active
    assert set(workspace.viewer.get_pending_placed_object_ids()) == set(selected)
    assert set(workspace.viewer.get_selected_placed_object_ids()) == set(selected)
    assert workspace._pending_object_placement_ids == set(selected)
    assert workspace._object_placement_mesh_update_timer.isActive()
    assert len(workspace._canvas_undo_stack) == undo_count + 1
    for object_id, offset in zip(selected, offsets, strict=True):
        placement = workspace.generation.get_scene_object_placement(object_id)
        actual = level_image_to_world_xy(workspace.current_level, placement.image_x, placement.image_y)
        np.testing.assert_allclose(actual, np.asarray((10.0, 10.0)) + offset[:2])
    workspace._schedule_viewer_preview_refresh.assert_not_called()
    workspace._handle_canvas_undo_requested()
    assert all(workspace.generation.get_scene_object_placement(value) is None for value in selected)
    assert not workspace.viewer.get_pending_placed_object_ids()
    assert len(workspace._canvas_undo_stack) == undo_count


def test_cancelled_grid_changes_no_placements_or_pending_meshes(workspace):
    before = workspace.generation.get_data()
    undo_count = len(workspace._canvas_undo_stack)
    workspace._handle_atlas_objects_place_requested(("chair", "desk"))
    workspace.viewer.cancel_object_placement()
    assert workspace._direct_object_placement_session is None
    assert workspace.generation.get_data() == before
    assert len(workspace._canvas_undo_stack) == undo_count
    assert not workspace._pending_object_placement_ids


def test_invalid_group_does_not_replace_an_existing_picker_or_load_geometry(workspace):
    workspace._handle_atlas_objects_place_requested(("chair", "desk"))
    session = workspace._direct_object_placement_session
    before = workspace.generation.get_data()
    with patch.object(workspace, "_get_scene_generated_object_model") as load:
        workspace._handle_atlas_objects_place_requested(("chair", "missing"))
    load.assert_not_called()
    assert workspace._direct_object_placement_session is session
    assert workspace.generation.get_data() == before
    assert not workspace._pending_object_placement_ids


@pytest.mark.parametrize("start_after_arming", (False, True))
def test_active_mutation_blocks_the_entire_group_without_partial_changes(workspace, start_after_arming):
    before = workspace.generation.get_data()
    if start_after_arming:
        workspace._handle_atlas_objects_place_requested(("chair", "desk"))
    with patch.object(workspace.generation, "_object_has_active_mutation_job",
                      side_effect=lambda object_id: object_id == "desk"):
        if start_after_arming:
            _floor_click(workspace)
        else:
            workspace._handle_atlas_objects_place_requested(("chair", "desk"))
    assert workspace._direct_object_placement_session is None
    assert workspace.generation.get_data() == before
    assert not workspace._pending_object_placement_ids


def test_grid_keeps_existing_instance_identity_rotation_and_scale(workspace):
    previous = GeneratedObjectPlacement(
        2, 30, 40, rotation_degrees=(15, 25, 40), scale=1.3, axis_scales=(1.2, 0.8, 1.1)
    )
    instance = GeneratedObjectInstance("chair-copy", "chair", previous)
    assert workspace.generation.restore_generated_object_instance(instance, emit_change_signals=False)
    undo_count = len(workspace._canvas_undo_stack)
    workspace._handle_atlas_objects_place_requested(("desk", instance.instance_id))
    _floor_click(workspace)
    actual = workspace.generation.get_generated_object_instance(instance.instance_id)
    assert actual.source_object_id == instance.source_object_id
    assert actual.placement.rotation_degrees == previous.rotation_degrees
    assert actual.placement.scale == previous.scale
    assert actual.placement.axis_scales == previous.axis_scales
    assert actual.placement != previous
    assert len(workspace._canvas_undo_stack) == undo_count + 1
    workspace._handle_canvas_undo_requested()
    assert workspace.generation.get_generated_object_instance(instance.instance_id) == instance
    assert workspace.generation.get_generated_object_placement("desk") is None


def test_physical_grid_spacing_respects_global_scale_and_current_rotations(workspace):
    previous = GeneratedObjectPlacement(2, 30, 40, rotation_degrees=(0, 0, 90), scale=2.0)
    assert workspace.generation.restore_generated_object_placement("chair", previous, emit_change_signals=False)
    workspace.global_scale = GlobalScale(1.8, 1.9, 1.3, linked=False)
    workspace.viewer.set_global_scale(workspace.global_scale)
    workspace._handle_atlas_objects_place_requested(("chair", "desk"))
    previews = workspace.viewer._object_placement_preview_meshes
    factors = np.asarray(workspace.global_scale.factors)
    physical = [mesh.bounds * factors for mesh in previews]
    assert physical[1][0, 0] - physical[0][1, 0] == pytest.approx(0.25)
    assert physical[0][1, 0] - physical[0][0, 0] == pytest.approx(3.6)
    assert physical[0][1, 1] - physical[0][0, 1] == pytest.approx(7.6)


def test_grid_footprints_include_off_center_mirrored_halves(workspace):
    metadata = SymmetricDivisionMetadata(
        orientation="vertical", kept_side="left", plane_coordinate=1.0
    )
    with patch.object(workspace.generation, "resolve_symmetric_division_for_record",
                      side_effect=lambda record: metadata if record.object_id == "chair" else None):
        workspace._handle_atlas_objects_place_requested(("chair", "desk"))
    previews = workspace.viewer._object_placement_preview_meshes
    offsets = workspace.viewer._object_placement_preview_offsets
    assert previews[0].extents[0] == pytest.approx(4.0)
    assert previews[1].bounds[0, 0] - previews[0].bounds[1, 0] == pytest.approx(0.25)
    assert (previews[0].bounds[0, 0] + previews[0].bounds[1, 0]) / 2.0 == pytest.approx(
        offsets[0][0] + 1.0
    )


def test_failed_group_transaction_rolls_back_every_previous_member(workspace):
    before = workspace.generation.get_data()
    undo_count = len(workspace._canvas_undo_stack)
    workspace._handle_atlas_objects_place_requested(("chair", "desk"))
    original = workspace._restore_group_member_placement

    def restore(object_id, placement, **kwargs):
        if object_id == "desk" and placement is not None:
            return False
        return original(object_id, placement, **kwargs)

    with patch.object(workspace, "_restore_group_member_placement", side_effect=restore):
        _floor_click(workspace)
    assert workspace.generation.get_data() == before
    assert len(workspace._canvas_undo_stack) == undo_count
    assert not workspace._pending_object_placement_ids
    assert not workspace.viewer.get_pending_placed_object_ids()
