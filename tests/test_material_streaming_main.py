"""Authoring controls persist and export the declarative streaming instructions."""

# ### Environment and imports ###
from __future__ import annotations

import json
from dataclasses import replace
from unittest.mock import patch

from housemaker.ktx2_compression import Ktx2CompressionError
from housemaker.lazy_loading_state import CameraStartPoint, RuntimeLoadingData
from housemaker.models import GROUND_LEVEL_INDEX, create_default_levels
from housemaker.project_io import ProjectData
from tests.test_room_export_workspace import _textured_model
from tests.test_room_export_workspace import workspace as _workspace_fixture

# ### Shared, isolated Qt workspace fixture ###
workspace = _workspace_fixture


# ### Controls, placement and persistence ###
def test_compression_checkbox_is_immediately_below_bake_and_has_no_mesh_side_effects(
    workspace,
):
    checkbox = workspace.use_ktx2_basis_compression_checkbox
    assert checkbox.text() == "Use KTX2/Basis compression"
    assert not checkbox.isChecked()
    layout = workspace.levels_group.layout()
    row = layout.itemAt(layout.indexOf(checkbox) - 1).layout()
    assert row.indexOf(workspace.bake_visibility_button) >= 0
    revision = workspace._viewer_preview_revision
    checkbox.setChecked(True)
    assert workspace.runtime_loading.use_ktx2_basis_compression
    assert workspace._viewer_preview_revision == revision


def test_save_and_load_forward_starting_point_and_compression_choice(
    workspace, tmp_path
):
    settings = RuntimeLoadingData(CameraStartPoint(2, 45, 60, 0.1, 25, -10), True)
    workspace.runtime_loading = settings
    with (
        patch(
            "housemaker.main.QFileDialog.getSaveFileName",
            return_value=(str(tmp_path / "save.json"), ""),
        ),
        patch("housemaker.main.save_project") as save,
        patch("housemaker.main.QMessageBox.information"),
    ):
        workspace._handle_save_clicked()
    assert save.call_args.kwargs["runtime_loading"] == settings
    workspace._apply_loaded_project(
        ProjectData(
            None, GROUND_LEVEL_INDEX, create_default_levels(), runtime_loading=settings
        )
    )
    assert workspace.runtime_loading == settings
    assert workspace.use_ktx2_basis_compression_checkbox.isChecked()
    assert workspace.clear_fps_start_button.isEnabled()
    workspace._apply_loaded_project(
        ProjectData(None, GROUND_LEVEL_INDEX, create_default_levels())
    )
    assert workspace.runtime_loading == RuntimeLoadingData()
    assert not workspace.use_ktx2_basis_compression_checkbox.isChecked()


# ### Main export boundary ###
def _mock_scene(workspace, model):
    from contextlib import ExitStack

    stack = ExitStack()
    for name in (
        "_sync_atlas_object_texture_sources",
        "_ensure_viewer_preview_current",
    ):
        stack.enter_context(patch.object(workspace, name))
    for name in (
        "_show_unpacked_scene_texture_export_error",
        "_show_mixed_scene_atlas_export_error",
    ):
        stack.enter_context(patch.object(workspace, name, return_value=False))
    stack.enter_context(
        patch.object(
            workspace, "_build_model_with_stable_dependencies", return_value=(model, ())
        )
    )
    stack.enter_context(
        patch.object(workspace, "_build_export_placed_models", return_value=((), ()))
    )
    stack.enter_context(patch("housemaker.main.QMessageBox.information"))
    return stack


def test_main_exports_preview_and_full_cubes_and_start_pose_in_json(
    workspace, tmp_path
):
    workspace.runtime_loading = RuntimeLoadingData(
        CameraStartPoint(2, 45, 60, 0, 25, -10)
    )
    with (
        _mock_scene(workspace, _textured_model()),
        patch("housemaker.main.QMessageBox.critical") as failure,
    ):
        workspace._handle_glb_export_clicked()
    failure.assert_not_called()
    path = tmp_path / "public" / "models" / "housemaker_export.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    policy = payload["materialStreaming"]
    assert policy["initialCamera"]["source"] == "authored"
    assert (
        payload["firstPersonCamera"]["position"] == policy["initialCamera"]["position"]
    )
    assert policy["textureCompression"]["requiredLoader"] is None
    for material in payload["roomAssets"]["materials"]:
        for tier in ("preview", "full"):
            assert (path.parent / material["variants"][tier]["file"]).is_file()


def test_missing_optional_encoder_fails_before_writing_any_runtime_files(
    workspace, tmp_path
):
    workspace.runtime_loading = replace(
        workspace.runtime_loading, use_ktx2_basis_compression=True
    )
    with (
        _mock_scene(workspace, _textured_model()),
        patch(
            "housemaker.ktx2_compression.require_ktx2_encoder",
            side_effect=Ktx2CompressionError("Encoder unavailable"),
        ),
        patch("housemaker.main.QMessageBox.critical") as failure,
    ):
        workspace._handle_glb_export_clicked()
    failure.assert_called_once()
    assert "Encoder unavailable" in failure.call_args.args[-1]
    assert not (tmp_path / "public").exists()
