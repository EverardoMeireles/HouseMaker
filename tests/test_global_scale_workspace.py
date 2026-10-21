# ### Environment setup ###
from __future__ import annotations

import gc
import json
import os
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import pytest
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.global_scale import GlobalScale
from housemaker.main import BlueprintWorkspace
from housemaker.models import StairData, create_default_levels
from housemaker.project_io import load_project, save_project

# ### Qt lifetime and workspace fixtures ###
_application = QApplication.instance() or QApplication([])
_application.setQuitOnLastWindowClosed(False)
_retained_workspaces: list[BlueprintWorkspace] = []


@pytest.fixture(scope="module", autouse=True)
def _stable_qt_lifetimes():
    """Keep offscreen Qt/OpenGL wrappers alive until interpreter shutdown."""

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


@pytest.fixture
def workspace(tmp_path: Path):
    value = BlueprintWorkspace(
        application_settings=ApplicationSettingsStore(tmp_path / "settings.json")
    )
    _retained_workspaces.append(value)
    value._set_mesh_edit_update_delay_seconds(60.0)
    value._canvas_undo_stack.clear()
    # Test real UI signals and state changes without building geometry or
    # opening an OpenGL context on the offscreen platform.
    with ExitStack() as mocks:
        for name in (
            "_schedule_viewer_preview_refresh",
            "_refresh_viewer_preview",
            "_ensure_viewer_preview_current",
            "_sync_atlas_object_texture_sources",
        ):
            mocks.enter_context(patch.object(value, name))
        try:
            yield value
        finally:
            value.shutdown()
            value.close()


def _control_factors(workspace: BlueprintWorkspace) -> tuple[float, ...]:
    return tuple(workspace.global_multiplier_spinboxes[axis].value() for axis in "xyz")


# ### Linked controls, authoring preservation and undo ###
def test_controls_start_linked_below_floor_thickness_and_share_scale(workspace) -> None:
    assert workspace.global_scale == GlobalScale()
    assert _control_factors(workspace) == (1.0, 1.0, 1.0)
    assert workspace.global_multiplier_link_checkbox.isChecked()
    layout = workspace.level_dimensions_group.layout()
    floor_row, _ = layout.getWidgetPosition(workspace.floor_thickness_spinbox)
    multiplier_row, _ = layout.getWidgetPosition(
        workspace.global_multiplier_spinboxes["x"].parentWidget()
    )
    assert multiplier_row == floor_row + 1
    assert workspace.canvas.get_global_scale() == GlobalScale()
    assert workspace.viewer.get_global_scale() == GlobalScale()


def test_linked_and_independent_controls_never_rewrite_authored_levels(
    workspace,
) -> None:
    for index, level in enumerate(workspace.levels):
        level.vertex_data.add_vertex(100.0 + index, 200.0 + index)
        level.height_meters = 2.45 + index * 0.1
        level.canvas_level_scale = 1.25
        level.canvas_offset_x_pixels = 12.0
        level.plan_meters_per_pixel = 0.00843
    before = [asdict(level) for level in workspace.levels]

    workspace.global_multiplier_spinboxes["x"].setValue(2.0)
    assert _control_factors(workspace) == (2.0, 2.0, 2.0)
    workspace.global_multiplier_link_checkbox.setChecked(False)
    workspace.global_multiplier_spinboxes["y"].setValue(3.0)

    expected = GlobalScale(2.0, 3.0, 2.0, linked=False)
    assert workspace.global_scale == expected
    assert _control_factors(workspace) == expected.factors
    assert workspace.canvas.get_global_scale() == expected
    assert workspace.viewer.get_global_scale() == expected
    assert [asdict(level) for level in workspace.levels] == before
    workspace._refresh_viewer_preview.assert_not_called()
    assert workspace.height_level_spinbox.value() == pytest.approx(
        workspace.current_level.height_meters * 2.0
    )


def test_canvas_undo_restores_each_multiplier_and_link_change(workspace) -> None:
    workspace.global_multiplier_spinboxes["x"].setValue(2.0)
    workspace.global_multiplier_link_checkbox.setChecked(False)
    workspace.global_multiplier_spinboxes["y"].setValue(3.0)
    assert len(workspace._canvas_undo_stack) == 3

    for expected in (
        GlobalScale(2.0, 2.0, 2.0, linked=False),
        GlobalScale(2.0, 2.0, 2.0),
        GlobalScale(),
    ):
        workspace._handle_canvas_undo_requested()
        assert workspace.global_scale == expected
        assert _control_factors(workspace) == expected.factors
        assert workspace.global_multiplier_link_checkbox.isChecked() == expected.linked
        assert workspace.canvas.get_global_scale() == expected
        assert workspace.viewer.get_global_scale() == expected
    assert not workspace._canvas_undo_stack


def test_rejected_linked_edit_restores_controls_without_partial_state_or_undo(
    workspace,
) -> None:
    original = GlobalScale(2.0, 90.0, 1.0)
    workspace._set_global_scale(original)
    workspace._canvas_undo_stack.clear()
    with patch.object(workspace.viewer, "set_surface_tools_status") as status:
        workspace.global_multiplier_spinboxes["x"].setValue(4.0)

    assert workspace.global_scale == original
    assert _control_factors(workspace) == original.factors
    assert workspace.canvas.get_global_scale() == original
    assert workspace.viewer.get_global_scale() == original
    assert not workspace._canvas_undo_stack
    status.assert_called_once()
    assert "Global Y multiplier" in status.call_args.args[0]


# ### Physical dimension edits ###
def test_actual_stair_rise_displays_physical_height_without_changing_step_count(
    workspace,
) -> None:
    stair = StairData(
        start_level_index=workspace.levels[2].index,
        end_level_index=workspace.levels[3].index,
        start_x=0.0,
        start_y=0.0,
        end_x=0.0,
        end_y=200.0,
    )
    workspace.stairs = [stair]
    workspace._editing_stair_index = 0
    workspace._sync_stair_calculated_values(stair)
    count = workspace.stair_calculated_step_count_label.text()
    rise = float(workspace.stair_actual_rise_label.text().split()[0])

    workspace.global_multiplier_spinboxes["z"].setValue(2.0)

    assert workspace.stair_calculated_step_count_label.text() == count
    assert float(workspace.stair_actual_rise_label.text().split()[0]) == pytest.approx(
        rise * 2.0, abs=0.1
    )
    assert workspace.stairs == [stair]


def test_small_vertical_multiplier_preserves_subcentimeter_field_precision(
    workspace,
) -> None:
    workspace.current_level.height_meters = 2.45
    workspace.current_level.floor_thickness_meters = 0.3
    workspace.global_multiplier_link_checkbox.setChecked(False)
    workspace.global_multiplier_spinboxes["z"].setValue(0.01)

    assert workspace.height_level_spinbox.value() == pytest.approx(0.0245)
    assert workspace.floor_thickness_spinbox.value() == pytest.approx(0.003)
    assert workspace.current_level.floor_thickness_meters == pytest.approx(0.3)
    workspace.floor_thickness_spinbox.setValue(0.0035)
    assert workspace.current_level.floor_thickness_meters == pytest.approx(0.35)
    assert workspace.floor_thickness_spinbox.value() == pytest.approx(0.0035)


def test_height_and_floor_fields_edit_physical_meters_once(workspace) -> None:
    workspace.global_multiplier_link_checkbox.setChecked(False)
    workspace.global_multiplier_spinboxes["z"].setValue(2.0)

    workspace.height_level_spinbox.setValue(6.0)
    assert workspace.current_level.height_meters == pytest.approx(3.0)
    assert workspace.height_level_spinbox.value() == pytest.approx(6.0)
    workspace.floor_thickness_spinbox.setValue(0.8)
    assert workspace.current_level.floor_thickness_meters == pytest.approx(0.4)
    assert workspace.floor_thickness_spinbox.value() == pytest.approx(0.8)
    assert workspace._pending_canvas_surface_mesh_update
    assert workspace._canvas_surface_mesh_update_timer.isActive()

    workspace._sync_level_controls()
    assert workspace.height_level_spinbox.value() == pytest.approx(6.0)
    assert workspace.floor_thickness_spinbox.value() == pytest.approx(0.8)
    assert workspace.current_level.height_meters == pytest.approx(3.0)
    assert workspace.current_level.floor_thickness_meters == pytest.approx(0.4)


def test_switching_levels_preserves_scale_and_displays_their_physical_dimensions(
    workspace,
) -> None:
    workspace.global_multiplier_spinboxes["z"].setValue(1.5)
    other_index = (workspace.current_level_index + 1) % len(workspace.levels)
    other = workspace.levels[other_index]
    other.height_meters = 2.2
    other.floor_thickness_meters = 0.2

    workspace._handle_level_selection_changed(other_index)

    assert workspace.current_level is other
    assert workspace.global_scale == GlobalScale(1.5, 1.5, 1.5)
    assert workspace.height_level_spinbox.value() == pytest.approx(3.3)
    assert workspace.floor_thickness_spinbox.value() == pytest.approx(0.3)
    assert workspace.canvas.get_global_scale() == workspace.global_scale
    assert workspace.viewer.get_global_scale() == workspace.global_scale


# ### Persistence and final export boundaries ###
def test_save_and_load_restore_multiplier_without_baking_authored_dimensions(
    workspace, tmp_path: Path
) -> None:
    workspace.current_level.height_meters = 2.45
    workspace.current_level.floor_thickness_meters = 0.25
    workspace.global_multiplier_link_checkbox.setChecked(False)
    workspace.global_multiplier_spinboxes["x"].setValue(1.25)
    workspace.global_multiplier_spinboxes["z"].setValue(2.0)
    expected = workspace.global_scale
    path = tmp_path / "saved-project.json"
    with (
        patch(
            "housemaker.main.QFileDialog.getSaveFileName", return_value=(str(path), "")
        ),
        patch("housemaker.main.QMessageBox.information"),
    ):
        workspace._handle_save_clicked()
    loaded = load_project(path)
    assert loaded.global_scale == expected
    loaded_level = loaded.levels[loaded.current_level_index]
    assert loaded_level.height_meters == pytest.approx(2.45)
    assert loaded_level.floor_thickness_meters == pytest.approx(0.25)

    workspace.global_multiplier_spinboxes["x"].setValue(3.0)
    workspace._load_project_path(path)

    assert workspace.global_scale == expected
    assert _control_factors(workspace) == expected.factors
    assert not workspace.global_multiplier_link_checkbox.isChecked()
    assert workspace.canvas.get_global_scale() == expected
    assert workspace.viewer.get_global_scale() == expected
    assert workspace.height_level_spinbox.value() == pytest.approx(4.9)
    assert workspace.floor_thickness_spinbox.value() == pytest.approx(0.5)
    assert not workspace._canvas_undo_stack


def test_loading_legacy_project_resets_previous_scale_and_link_state(
    workspace, tmp_path: Path
) -> None:
    path = tmp_path / "legacy.json"
    save_project(path, workspace.current_level_index, create_default_levels())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("global_scale")
    path.write_text(json.dumps(payload), encoding="utf-8")
    workspace.global_multiplier_link_checkbox.setChecked(False)
    workspace.global_multiplier_spinboxes["x"].setValue(4.0)

    workspace._load_project_path(path)

    assert workspace.global_scale == GlobalScale()
    assert _control_factors(workspace) == (1.0, 1.0, 1.0)
    assert workspace.global_multiplier_link_checkbox.isChecked()
    assert workspace.canvas.get_global_scale() == GlobalScale()
    assert workspace.viewer.get_global_scale() == GlobalScale()


def test_export_receives_canonical_model_and_multiplier_separately(
    workspace, tmp_path: Path
) -> None:
    workspace.global_multiplier_spinboxes["x"].setValue(2.0)
    export_path = tmp_path / "scaled.glb"
    canonical_model = object()
    with (
        patch.object(
            workspace, "_show_unpacked_scene_texture_export_error", return_value=False
        ),
        patch(
            "housemaker.main.QFileDialog.getSaveFileName",
            return_value=(str(export_path), ""),
        ),
        patch.object(
            workspace,
            "_build_model_with_stable_dependencies",
            return_value=(canonical_model, ("revision",)),
        ),
        patch.object(workspace, "_build_export_placed_models", return_value=((), ())),
        patch(
            "housemaker.main.export_glb_file", return_value=export_path
        ) as export_glb,
        patch(
            "housemaker.main.write_runtime_scene_manifest",
            return_value=export_path.with_suffix(".json"),
        ) as write_manifest,
        patch("housemaker.main.QMessageBox.information"),
    ):
        workspace._handle_glb_export_clicked()

    export_glb.assert_called_once_with(canonical_model, export_path)
    assert write_manifest.call_count == 1
    assert write_manifest.call_args.args == (export_path,)
    assert write_manifest.call_args.kwargs["global_scale"] == GlobalScale(2.0, 2.0, 2.0)


# ### Lazy viewers ###
def test_lazy_tour_viewer_inherits_scale_and_tracks_subsequent_changes(
    workspace,
) -> None:
    assert workspace.tour_preview_viewer is None
    workspace.global_multiplier_spinboxes["x"].setValue(1.5)

    viewer = workspace._ensure_tour_preview_viewer()

    assert viewer.get_global_scale() == GlobalScale(1.5, 1.5, 1.5)
    workspace.global_multiplier_link_checkbox.setChecked(False)
    workspace.global_multiplier_spinboxes["y"].setValue(2.5)
    assert viewer.get_global_scale() == GlobalScale(1.5, 2.5, 1.5, linked=False)
    assert workspace._ensure_tour_preview_viewer() is viewer
