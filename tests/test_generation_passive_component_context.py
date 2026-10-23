"""Restored library previews must never become component-generation jobs."""

# ### Environment and imports ###
from __future__ import annotations

import os
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from housemaker.generation_state import GeneratedObjectRecord, GenerationData
from housemaker.generation_workspace import (
    ARCHITECTURAL_TRIM_COMPONENT_PIPELINE_KEY,
    DOOR_COMPONENT_PIPELINE_KEY,
    GenerationWorkspace,
)
from housemaker.settings_widget import GenerationServiceSettings
from tests.test_manual_inpaint_queue import _stroke, _VideoSource

# ### Lightweight Qt and component fixtures ###
_application = QApplication.instance() or QApplication([])


@pytest.fixture
def workspace(tmp_path):
    widget = GenerationWorkspace(asset_directory=tmp_path)
    widget.set_runtime_settings(GenerationServiceSettings(meshy_api_key="test-only"))
    try:
        yield widget
    finally:
        widget.shutdown()
        widget.close()
        widget.deleteLater()
        _application.processEvents()


def _reference():
    return np.full((20, 32, 4), (30, 80, 160, 255), dtype=np.uint8)


def _select_component(workspace, kind, *, ready, rebuild_body=True):
    is_trim = kind == "trim"
    slot = "door_knob" if kind == "handle" else "body"
    metadata = (
        {ARCHITECTURAL_TRIM_COMPONENT_PIPELINE_KEY: {"level_index": 0, "trim_id": "trim"}}
        if is_trim else
        {DOOR_COMPONENT_PIPELINE_KEY: {"door_id": "opening", "slot_id": slot}}
    )
    record = GeneratedObjectRecord(
        object_id="library-component", frame_index=0, object_name="Library component",
        pipeline=metadata, provider_task_id="existing-task", asset_path="existing.glb",
    )
    workspace._data.generated_objects.append(record)
    if is_trim:
        assert workspace.set_architectural_trim_editing_target(
            level_index=0, trim_id="trim", object_id=record.object_id,
            object_name=record.object_name, fit_dimensions=(4.0, 0.1, 0.3),
            generation_ready=ready, defer_file_backed_preview=True,
        )
        target = workspace._architectural_trim_editing_target
    else:
        assert workspace.set_door_slot_editing_target(
            door_id="opening", slot_id=slot, object_id=record.object_id,
            object_name=record.object_name, slot_display_name=slot,
            asset_kind="window" if kind == "window" else "door",
            generate_displacement=slot == "body" and rebuild_body,
            door_body_fit_dimensions=(0.9, 0.1, 2.0) if slot == "body" else None,
            generation_ready=ready, defer_file_backed_preview=True,
        )
        target = workspace._door_slot_editing_target
    return record, target


# ### Ordinary reference generation is independent of passive library selection ###
@pytest.mark.parametrize("kind", ("door", "window", "handle", "trim"))
@pytest.mark.parametrize("geometry_only", (False, True))
def test_pasted_reference_generates_an_ordinary_object_not_the_passive_component(
    workspace, kind, geometry_only,
):
    record, target = _select_component(workspace, kind, ready=False)
    workspace.set_temporary_object_reference(_reference())
    action = workspace.generate_geometry if geometry_only else workspace.generate
    button = workspace.generate_geometry_button if geometry_only else workspace.generate_button

    assert not workspace.component_generation_active
    assert button.isEnabled()
    assert workspace.active_opening_component_kind() is None
    with patch.object(workspace, "_start_generation", return_value="ordinary-job") as start:
        action()

    start.assert_called_once()
    request = start.call_args.args[0]
    assert request.geometry_only == geometry_only
    assert request.door_body_fit_dimensions is None
    assert request.door_side_duplication is None
    assert request.architectural_trim_fit_dimensions is None
    assert request.settings is workspace._settings
    assert start.call_args.kwargs.get("door_slot_target") is None
    assert start.call_args.kwargs.get("architectural_trim_target") is None
    assert start.call_args.kwargs.get("replaced_object_record") is None
    np.testing.assert_array_equal(request.selected_object_bgra, _reference())
    assert not target.generation_ready
    assert workspace._find_generated_object_record(record.object_id) is record
    assert workspace._generation_component_targets() == (None, None)


@pytest.mark.parametrize("kind", ("door", "window", "trim"))
def test_passive_preview_preserves_ordinary_blob_keys_references_and_batch(workspace, kind):
    source = _VideoSource()
    workspace._video_source = source
    workspace._data = GenerationData(video_metadata=source.metadata)
    workspace.show_frame(0)
    workspace.video_view.set_strokes([_stroke(0.2, 0.25), _stroke(0.8, 0.75)])
    workspace._handle_video_strokes_changed(workspace.video_view.get_strokes())
    key = workspace.current_inpaint_reference_key()
    _select_component(workspace, kind, ready=False)

    assert workspace.current_inpaint_reference_key() == key
    workspace.set_temporary_object_reference(_reference())
    inputs = workspace.snapshot_inpaint_reference_inputs()
    requests = workspace._build_generation_requests(geometry_only=True)

    assert len(inputs) == len(requests) == 2
    assert requests[0].inpaint_key == key
    np.testing.assert_array_equal(requests[0].selected_object_bgra, _reference())
    assert requests[1].selected_object_bgra.shape != _reference().shape
    assert all(request.door_body_fit_dimensions is None for request in requests)
    assert all(request.architectural_trim_fit_dimensions is None for request in requests)


# ### Explicit component generation survives external references ###
@pytest.mark.parametrize("kind", ("door", "window", "handle", "trim"))
@pytest.mark.parametrize("geometry_only", (False, True))
def test_explicit_component_target_remains_active_after_pasting(workspace, kind, geometry_only):
    record, target = _select_component(
        workspace, kind, ready=True, rebuild_body=not geometry_only,
    )
    workspace.set_temporary_object_reference(_reference())
    action = workspace.generate_geometry if geometry_only else workspace.generate

    assert workspace.component_generation_active
    with patch.object(workspace, "_start_generation", return_value="component-job") as start:
        action()

    start.assert_called_once()
    request = start.call_args.args[0]
    assert start.call_args.kwargs["replaced_object_record"] is record
    assert start.call_args.kwargs[
        "architectural_trim_target" if kind == "trim" else "door_slot_target"
    ] is target
    assert target.generation_ready
    np.testing.assert_array_equal(request.selected_object_bgra, _reference())


# ### Defensive direct dispatch and texture mutation guards ###
def test_direct_door_dispatch_cannot_upgrade_a_passive_target(workspace):
    _record, target = _select_component(workspace, "door", ready=False)
    workspace.set_temporary_object_reference(_reference())

    with patch.object(workspace, "_start_generation") as start:
        assert not workspace._start_door_slot_model_generation(target, geometry_only=True)

    start.assert_not_called()
    assert workspace._door_slot_editing_target is target
    assert not target.generation_ready


@pytest.mark.parametrize("kind", ("door", "window", "trim"))
def test_passive_library_component_cannot_be_retextured_as_an_ordinary_object(workspace, kind):
    record, _target = _select_component(workspace, kind, ready=False)
    workspace.set_temporary_object_reference(_reference())

    assert not workspace._can_regenerate_object_texture(record)
    assert not workspace.regenerate_texture_button.isEnabled()
    with patch.object(workspace, "_start_texture_regeneration") as start:
        assert not workspace.generate_selected_object_texture()

    start.assert_not_called()
