"""Sunlight defaults, initial-direction editing and saved orbit controls."""

# ### Environment and imports ###
from __future__ import annotations

import os
from dataclasses import replace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.directional_light_state import DirectionalLightData, SunOrbitData
from housemaker.global_scale import GlobalScale
from housemaker.level_coordinates import (
    build_level_base_z_lookup,
    level_image_to_world_xy,
)
from housemaker.main import BlueprintWorkspace
from tests.runtime_export_helpers import mock_material_export
from housemaker.models import (
    GROUND_LEVEL_INDEX,
    Edge,
    RoomData,
    Vertex,
    VertexData,
    create_default_levels,
)

# ### Fixtures and legacy-project helpers ###
_application = QApplication.instance() or QApplication([])
_application.setQuitOnLastWindowClosed(False)


@pytest.fixture
def workspace(tmp_path):
    widget = BlueprintWorkspace(
        application_settings=ApplicationSettingsStore(tmp_path / "settings.json"),
    )
    try:
        yield widget
    finally:
        widget.shutdown()
        widget.close()
        widget.deleteLater()
        _application.processEvents()


def _legacy_sun(identifier="legacy-sun", *, intensity=2.3):
    return DirectionalLightData(
        identifier, "My existing sunlight", (4.0, 5.0, 6.0), (0.0, 1.0, 2.0),
        color="#ffcc88", intensity=intensity,
    )


def _apply(workspace, *, lights=None, orbit=None, levels=None):
    workspace._apply_project_state(
        levels=create_default_levels() if levels is None else levels,
        current_level_index=GROUND_LEVEL_INDEX,
        directional_lights=lights, sun_orbit=orbit,
    )


def _two_occupied_levels():
    levels = create_default_levels()
    for offset in (0, 1):
        level = levels[GROUND_LEVEL_INDEX + offset]
        level.height_meters = 2.5 + offset * 0.6
        level.floor_thickness_meters = 0.2 + offset * 0.15
        level.vertex_data = VertexData(
            vertices=[Vertex(1, 100.0, 50.0), Vertex(2, 500.0, 350.0)],
            edges=[Edge(1, 2)],
        )
        level.scale = 1.0 + offset * 0.5
        level.offset_x_meters = 10.0 - offset * 4.0
        level.offset_y_meters = -2.0 + offset * 5.0
    # Empty placeholders must not move the Sun dozens of metres above the house.
    levels[-1].height_meters = 80.0
    return levels


def _occupied_ceiling_top(levels):
    base_z = build_level_base_z_lookup(levels)
    return max(
        base_z[level.index]
        + max((level.height_meters, *(room.height_meters for room in level.rooms)))
        for level in levels
        if level.vertex_data.vertices
    )


# ### Default sunlight and backward-compatible project installation ###
def test_new_workspace_has_one_active_sun_without_placing_an_object(workspace):
    assert len(workspace.directional_lights) == 1
    sun = workspace.directional_lights[0]
    assert sun.intensity > 0.0
    assert sun.position != sun.target
    assert tuple(target - position for target, position in zip(sun.target, sun.position)) == (
        1.0, 0.0, -1.0,
    )
    assert "sun" in sun.name.casefold()
    assert not workspace.viewer.is_directional_light_placement_active
    assert not workspace.canvas.is_directional_light_placement_active()
    assert workspace.generation.get_generated_object_records() == ()


@pytest.mark.parametrize("legacy_lights", (None, ()))
def test_old_project_without_lights_gets_exactly_one_default_sun(workspace, legacy_lights):
    _apply(workspace, lights=legacy_lights)
    assert len(workspace.directional_lights) == 1
    assert workspace.directional_lights[0].intensity > 0.0
    assert workspace.sun_orbit == SunOrbitData()
    _apply(workspace, lights=legacy_lights)
    assert len(workspace.directional_lights) == 1


def test_old_nonempty_light_records_are_preserved_including_disabled_intensity(workspace):
    lights = (_legacy_sun(), _legacy_sun("custom-off-sun", intensity=0.0))
    _apply(workspace, lights=lights)
    assert workspace.directional_lights == lights
    assert workspace.directional_lights[0].name == "My existing sunlight"
    assert workspace.directional_lights[1].intensity == 0.0


# ### Checkable initial-direction editing never creates another light ###
def test_edit_button_selects_existing_primary_sun_without_point_placement(workspace):
    initial = workspace.directional_lights
    button = workspace.add_directional_light_button
    assert button.text() == "Edit sunlight initial direction"
    assert button.isCheckable()
    with (
        patch.object(workspace.canvas, "start_directional_light_placement") as canvas_placement,
        patch.object(workspace.viewer, "begin_directional_light_placement") as viewer_placement,
        patch.object(workspace.viewer, "set_sun_direction_editing",
                     wraps=workspace.viewer.set_sun_direction_editing) as editing,
    ):
        button.click()
        assert button.isChecked()
        assert workspace.directional_lights == initial
        assert workspace.viewer.get_selected_directional_light_id() == initial[0].light_id
        assert editing.call_args.args == (True,)
        canvas_placement.assert_not_called()
        viewer_placement.assert_not_called()
        button.click()
        assert not button.isChecked()
        assert editing.call_args.args == (False,)
    assert workspace.directional_lights == initial


def test_edit_button_always_uses_first_existing_legacy_sun_not_another_selected_light(workspace):
    lights = (_legacy_sun("first"), _legacy_sun("second"))
    _apply(workspace, lights=lights)
    workspace._selected_directional_light_id = lights[1].light_id
    workspace._sync_directional_light_views()
    workspace.add_directional_light_button.click()
    assert workspace.viewer.get_selected_directional_light_id() == lights[0].light_id
    assert workspace.directional_lights == lights


# ### Marker activation exits initial-direction editing ###
@pytest.mark.parametrize("origin", ("viewer", "canvas"))
def test_clicking_already_selected_sun_resumes_preview_orbit(workspace, origin):
    sun = workspace.directional_lights[0]
    workspace.add_directional_light_button.click()
    assert workspace._is_editing_sunlight_direction
    assert workspace.viewer.get_selected_directional_light_id() == sun.light_id
    initial = workspace.directional_lights
    with patch.object(
        workspace.viewer, "set_sun_direction_editing",
        wraps=workspace.viewer.set_sun_direction_editing,
    ) as editing:
        if origin == "viewer":
            workspace.viewer.directional_light_preview_requested.emit(sun.light_id)
        else:
            workspace.canvas.directional_light_selection_requested.emit(sun.light_id)
    assert not workspace.add_directional_light_button.isChecked()
    assert not workspace._is_editing_sunlight_direction
    assert editing.call_args.args == (False,)
    assert workspace.viewer.get_selected_directional_light_id() == sun.light_id
    assert workspace.sun_orbit.enabled
    assert workspace.directional_lights == initial


def test_programmatic_sun_selection_does_not_exit_direction_editing(workspace):
    workspace.add_directional_light_button.click()
    sun = workspace.directional_lights[0]
    workspace._handle_directional_light_selection_changed(sun.light_id)
    assert workspace.add_directional_light_button.isChecked()
    assert workspace._is_editing_sunlight_direction
    assert workspace.viewer._sun_direction_editing


# ### Authoring handle placement above occupied scene levels ###
def test_default_handle_centers_above_all_occupied_levels_with_physical_clearance(workspace):
    workspace.levels = _two_occupied_levels()
    workspace.global_scale = GlobalScale(1.7, 0.8, 2.0)
    initial = workspace.directional_lights[0]
    with (
        patch("housemaker.main.build_fixed_surfaces") as build_surfaces,
        patch("housemaker.main.convert_to_glb") as build_model,
    ):
        assert workspace._position_sunlight_above_levels()
    build_surfaces.assert_not_called()
    build_model.assert_not_called()
    sun = workspace.directional_lights[0]
    plan_points = [
        level_image_to_world_xy(level, vertex.x, vertex.y)
        for level in workspace.levels
        for vertex in level.vertex_data.vertices
    ]
    assert sun.position[:2] == pytest.approx(
        tuple((min(axis) + max(axis)) / 2.0 for axis in zip(*plan_points))
    )
    top = _occupied_ceiling_top(workspace.levels)
    assert (sun.position[2] - top) * workspace.global_scale.z == pytest.approx(2.0)
    assert tuple(a - b for a, b in zip(sun.target, sun.position)) == pytest.approx(
        tuple(a - b for a, b in zip(initial.target, initial.position))
    )
    assert not workspace._position_sunlight_above_levels()


def test_height_changes_preserve_moved_xy_and_translate_target_without_rotation(workspace):
    workspace.levels = _two_occupied_levels()
    workspace._position_sunlight_above_levels()
    initial = workspace.directional_lights[0]
    delta = (4.0, -7.0, 0.0)
    moved = replace(
        initial,
        position=tuple(value + change for value, change in zip(initial.position, delta)),
        target=tuple(value + change for value, change in zip(initial.target, delta)),
    )
    workspace.directional_lights = (moved,)
    workspace.levels[GROUND_LEVEL_INDEX + 1].height_meters += 2.0
    assert workspace._position_sunlight_above_levels()
    sun = workspace.directional_lights[0]
    assert sun.position[:2] == moved.position[:2]
    assert sun.position[2] == pytest.approx(moved.position[2] + 2.0)
    assert sun.target[:2] == moved.target[:2]
    assert sun.target[2] == pytest.approx(moved.target[2] + 2.0)


def test_taller_room_and_nonexported_level_still_raise_primary_sun_only(workspace):
    workspace.levels = _two_occupied_levels()
    top_level = workspace.levels[GROUND_LEVEL_INDEX + 1]
    top_level.include_in_export = False
    top_level.rooms = [RoomData("Tall room", (1, 2), 1, (50, 60, 70), height_meters=5.0)]
    first = _legacy_sun("first")
    second = _legacy_sun("secondary")
    workspace.directional_lights = (first, second)
    assert workspace._position_sunlight_above_levels()
    sun = workspace.directional_lights[0]
    assert sun.position[:2] == first.position[:2]
    assert sun.position[2] == pytest.approx(_occupied_ceiling_top(workspace.levels) + 2.0)
    assert workspace.directional_lights[1] == second


def test_global_z_changes_keep_clearance_at_two_physical_metres(workspace):
    workspace.levels = _two_occupied_levels()
    workspace._position_sunlight_above_levels()
    xy = workspace.directional_lights[0].position[:2]
    workspace.global_scale = GlobalScale(2.0, 1.5, 4.0)
    assert workspace._position_sunlight_above_levels()
    sun = workspace.directional_lights[0]
    assert sun.position[:2] == xy
    assert sun.position[2] == pytest.approx(_occupied_ceiling_top(workspace.levels) + 0.5)


def test_global_z_spinbox_repositions_marker_without_geometry_rebuild(workspace):
    workspace.levels = _two_occupied_levels()
    workspace._sync_viewer_scene_levels()
    initial = workspace.directional_lights[0]
    workspace.global_multiplier_link_checkbox.setChecked(False)
    with (
        patch("housemaker.main.build_fixed_surfaces") as build_surfaces,
        patch("housemaker.main.convert_to_glb") as build_model,
        patch.object(
            workspace.viewer, "set_directional_lights",
            wraps=workspace.viewer.set_directional_lights,
        ) as update_lights,
    ):
        workspace.global_multiplier_spinboxes["z"].setValue(4.0)
    assert workspace.global_scale.z == 4.0
    sun = workspace.directional_lights[0]
    assert sun.position[:2] == initial.position[:2]
    assert (sun.position[2] - _occupied_ceiling_top(workspace.levels)) * 4.0 == pytest.approx(2.0)
    assert update_lights.call_args.args == (workspace.directional_lights,)
    build_surfaces.assert_not_called()
    build_model.assert_not_called()


def test_scene_level_sync_updates_light_marker_after_floor_thickness_change(workspace):
    workspace.levels = _two_occupied_levels()
    workspace._sync_viewer_scene_levels()
    previous_z = workspace.directional_lights[0].position[2]
    workspace.levels[GROUND_LEVEL_INDEX].floor_thickness_meters += 0.4
    workspace._sync_viewer_scene_levels()
    assert workspace.directional_lights[0].position[2] == pytest.approx(previous_z + 0.4)
    assert workspace.viewer._directional_lights == workspace.directional_lights


def test_project_install_positions_default_sun_above_loaded_levels(workspace):
    _apply(workspace, levels=_two_occupied_levels())
    sun = workspace.directional_lights[0]
    assert sun.position[2] == pytest.approx(_occupied_ceiling_top(workspace.levels) + 2.0)
    assert workspace.viewer._directional_lights == workspace.directional_lights


# ### Orbit visibility, physical speed and viewer plumbing ###
def test_orbit_is_on_by_default_and_speed_field_hides_only_when_disabled(workspace):
    assert workspace.sun_orbit == SunOrbitData(
        enabled=True, speed_degrees_per_second=0.5,
    )
    checkbox = workspace.enable_sun_orbit_checkbox
    speed = workspace.sun_orbit_speed_spinbox
    assert checkbox.isChecked()
    assert speed.value() == 0.5
    assert speed.isVisibleTo(workspace)
    assert workspace.sun_orbit.enabled
    checkbox.setChecked(False)
    assert not speed.isVisibleTo(workspace)
    assert not workspace.sun_orbit.enabled
    checkbox.setChecked(True)
    assert speed.isVisibleTo(workspace)
    assert workspace.sun_orbit.enabled


def test_orbit_speed_changes_are_sent_to_viewer_without_mutating_initial_lights(workspace):
    initial = workspace.directional_lights
    with patch.object(workspace.viewer, "set_sun_orbit",
                      wraps=workspace.viewer.set_sun_orbit) as update:
        workspace.enable_sun_orbit_checkbox.setChecked(True)
        workspace.sun_orbit_speed_spinbox.setValue(17.5)
        assert workspace.sun_orbit == SunOrbitData(True, 17.5)
        assert update.call_args.args == (workspace.sun_orbit,)
    assert workspace.directional_lights == initial


def test_editing_pauses_orbit_without_disabling_saved_orbit_setting(workspace):
    workspace.enable_sun_orbit_checkbox.setChecked(True)
    workspace.sun_orbit_speed_spinbox.setValue(8.0)
    configured = workspace.sun_orbit
    with patch.object(workspace.viewer, "set_sun_direction_editing",
                      wraps=workspace.viewer.set_sun_direction_editing) as editing:
        workspace.add_directional_light_button.click()
        assert editing.call_args.args == (True,)
        assert workspace.viewer.get_sun_orbit_phase_degrees() == 0.0
        assert workspace.sun_orbit == configured
        assert workspace.enable_sun_orbit_checkbox.isChecked()
        workspace.add_directional_light_button.click()
        assert editing.call_args.args == (False,)
        assert workspace.sun_orbit == configured


def test_project_apply_restores_orbit_config_controls_and_viewer(workspace):
    config = SunOrbitData(True, 13.5)
    lights = (_legacy_sun(),)
    with patch.object(workspace.viewer, "set_sun_orbit",
                      wraps=workspace.viewer.set_sun_orbit) as update:
        _apply(workspace, lights=lights, orbit=config)
    assert workspace.sun_orbit == config
    assert workspace.enable_sun_orbit_checkbox.isChecked()
    assert workspace.sun_orbit_speed_spinbox.value() == 13.5
    assert update.call_args.args == (config,)
    assert workspace.directional_lights == lights


# ### Save and runtime export completion boundaries ###
def test_save_passes_initial_sun_and_orbit_config_to_project_writer(workspace, tmp_path):
    workspace.enable_sun_orbit_checkbox.setChecked(True)
    workspace.sun_orbit_speed_spinbox.setValue(7.5)
    initial = workspace.directional_lights
    save_path = tmp_path / "sunlight.json"
    with (
        patch("housemaker.main.QFileDialog.getSaveFileName",
              return_value=(str(save_path), "JSON Files (*.json)")),
        patch("housemaker.main.save_project") as save,
        patch("housemaker.main.QMessageBox.information"),
    ):
        workspace._handle_save_clicked()
    assert save.call_count == 1
    assert save.call_args.kwargs["sun_orbit"] == SunOrbitData(True, 7.5)
    assert save.call_args.kwargs["directional_lights"] == initial


def test_runtime_export_passes_orbit_config_without_baking_current_phase_into_lights(
    workspace, tmp_path,
):
    workspace.enable_sun_orbit_checkbox.setChecked(True)
    workspace.sun_orbit_speed_spinbox.setValue(11.5)
    initial = workspace.directional_lights
    export_path = tmp_path / "sunlight.glb"
    generated_model = object()
    with (
        mock_material_export(workspace, generated_model),
        patch.object(workspace, "_sync_atlas_object_texture_sources"),
        patch.object(workspace, "_show_unpacked_scene_texture_export_error", return_value=False),
        patch.object(workspace, "_show_mixed_scene_atlas_export_error", return_value=False),
        patch("housemaker.main.QFileDialog.getSaveFileName",
              return_value=(str(export_path), "GLB Files (*.glb)")),
        patch.object(workspace, "_build_model_with_stable_dependencies",
                     return_value=(generated_model, ("revision",))),
        patch.object(workspace, "_build_export_placed_models", return_value=((), ())),
        patch("housemaker.main.export_glb_file", return_value=export_path),
        patch("housemaker.main.write_runtime_scene_manifest",
              return_value=export_path.with_suffix(".json")) as write_manifest,
        patch.object(workspace, "_ensure_viewer_preview_current"),
        patch("housemaker.main.QMessageBox.information"),
    ):
        workspace._handle_glb_export_clicked()
    assert write_manifest.call_count == 1
    assert write_manifest.call_args.kwargs["sun_orbit"] == SunOrbitData(True, 11.5)
    assert write_manifest.call_args.kwargs["directional_lights"] == initial
