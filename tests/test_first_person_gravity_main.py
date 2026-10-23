"""Main's gravity control, eye-height settings and runtime export stay aligned."""

# ### Environment and imports ###
from __future__ import annotations

import os
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.main import BlueprintWorkspace
from housemaker.settings_widget import FIRST_PERSON_EYE_HEIGHT_SETTING_KEY
from tests.runtime_export_helpers import mock_material_export

# ### Temporary Main workspaces ###
_application = QApplication.instance() or QApplication([])
_application.setQuitOnLastWindowClosed(False)


@pytest.fixture
def create_workspace(tmp_path):
    created = []

    def create(*, eye_height=None):
        settings = ApplicationSettingsStore(tmp_path / f"settings-{len(created)}.json")
        if eye_height is not None:
            settings.set(FIRST_PERSON_EYE_HEIGHT_SETTING_KEY, eye_height)
        widget = BlueprintWorkspace(application_settings=settings)
        created.append(widget)
        # Complete constructor-scheduled layout callbacks before interacting.
        _application.processEvents()
        return widget, settings

    try:
        yield create
    finally:
        for widget in reversed(created):
            widget.shutdown()
            widget.close()
            _application.processEvents()


# ### Shared 3D scene controls and external hosting ###
def test_activate_gravity_starts_off_and_controls_the_shared_scene_viewer(
    create_workspace,
):
    workspace, _settings = create_workspace()
    checkbox = workspace.activate_gravity_checkbox
    scene = workspace.scene_3d_workspace

    assert checkbox.text() == "Activate gravity"
    assert scene.isAncestorOf(checkbox)
    assert scene.isAncestorOf(workspace.confirm_mesh_updates_button)
    assert (
        checkbox.parentWidget() is workspace.confirm_mesh_updates_button.parentWidget()
    )
    assert checkbox.focusPolicy() == Qt.FocusPolicy.NoFocus
    assert not checkbox.isChecked()
    assert not workspace.viewer.get_first_person_gravity_enabled()

    checkbox.setChecked(True)
    assert workspace.viewer.get_first_person_gravity_enabled()
    checkbox.setChecked(False)
    assert not workspace.viewer.get_first_person_gravity_enabled()


def test_gravity_checkbox_follows_scene_into_external_host_and_back(create_workspace):
    workspace, _settings = create_workspace()
    screen = _application.primaryScreen()
    if screen is None:
        pytest.skip("A Qt screen is required for the external host")
    checkbox = workspace.activate_gravity_checkbox

    with (
        patch(
            "housemaker.main.resolve_fullscreen_3d_viewer_screen", return_value=screen
        ),
        patch.object(workspace, "_ensure_viewer_preview_current"),
    ):
        workspace._apply_scene_3d_display_screen("test-scene-screen")
        host = workspace._external_scene_3d_host
        assert host.is_active
        assert host.viewer is workspace.scene_3d_workspace
        assert host.window.isAncestorOf(checkbox)
        assert host.window.isAncestorOf(workspace.confirm_mesh_updates_button)
        checkbox.setChecked(True)
        assert workspace.viewer.get_first_person_gravity_enabled()

        workspace._apply_scene_3d_display_screen(None)

    assert not host.is_active
    assert workspace.workspace_tabs.indexOf(workspace.scene_3d_workspace) >= 0
    assert workspace.scene_3d_workspace.isAncestorOf(checkbox)
    assert checkbox.isChecked()
    assert workspace.viewer.get_first_person_gravity_enabled()


# ### Eye height uses persisted application settings, not project geometry ###
def test_default_eye_height_and_settings_changes_reach_the_scene_viewer(
    create_workspace,
):
    workspace, settings = create_workspace()
    spinbox = workspace.settings_widget.first_person_eye_height_spinbox

    assert spinbox.value() == 1.70
    assert workspace.viewer.get_first_person_eye_height() == 1.70
    assert workspace._generation_settings.first_person_eye_height_meters == 1.70

    spinbox.setValue(1.88)

    assert workspace.viewer.get_first_person_eye_height() == 1.88
    assert workspace._generation_settings.first_person_eye_height_meters == 1.88
    assert settings.get(FIRST_PERSON_EYE_HEIGHT_SETTING_KEY) == 1.88
    assert (
        ApplicationSettingsStore(settings.path).get(FIRST_PERSON_EYE_HEIGHT_SETTING_KEY)
        == 1.88
    )


def test_startup_restores_custom_eye_height_but_gravity_remains_opt_in(
    create_workspace,
):
    workspace, _settings = create_workspace(eye_height=1.93)

    assert workspace.settings_widget.first_person_eye_height_spinbox.value() == 1.93
    assert workspace._generation_settings.first_person_eye_height_meters == 1.93
    assert workspace.viewer.get_first_person_eye_height() == 1.93
    assert not workspace.activate_gravity_checkbox.isChecked()
    assert not workspace.viewer.get_first_person_gravity_enabled()


# ### Export forwards the active eye-height setting without building meshes ###
@pytest.mark.parametrize("eye_height", (1.70, 1.84))
def test_export_forwards_current_eye_height_to_runtime_manifest(
    create_workspace, tmp_path, eye_height
):
    workspace, _settings = create_workspace()
    workspace.settings_widget.first_person_eye_height_spinbox.setValue(eye_height)
    export_path = tmp_path / "house.glb"
    generated_model = object()
    with (
        mock_material_export(workspace, generated_model),
        patch.object(workspace, "_sync_atlas_object_texture_sources"),
        patch.object(
            workspace, "_show_unpacked_scene_texture_export_error", return_value=False
        ),
        patch.object(
            workspace, "_show_mixed_scene_atlas_export_error", return_value=False
        ),
        patch(
            "housemaker.main.QFileDialog.getSaveFileName",
            return_value=(str(export_path), ""),
        ),
        patch.object(
            workspace,
            "_build_model_with_stable_dependencies",
            return_value=(generated_model, ("revision",)),
        ),
        patch.object(workspace, "_build_export_placed_models", return_value=((), ())),
        patch("housemaker.main.export_glb_file", return_value=export_path),
        patch(
            "housemaker.main.write_runtime_scene_manifest",
            return_value=export_path.with_suffix(".json"),
        ) as write_manifest,
        patch.object(workspace, "_ensure_viewer_preview_current"),
        patch("housemaker.main.QMessageBox.information"),
        patch("housemaker.main.QMessageBox.critical") as failure,
    ):
        workspace._handle_glb_export_clicked()

    failure.assert_not_called()
    write_manifest.assert_called_once()
    assert write_manifest.call_args.args == (export_path,)
    assert (
        write_manifest.call_args.kwargs["first_person_eye_height_meters"] == eye_height
    )
    assert workspace.viewer.get_first_person_eye_height() == eye_height
