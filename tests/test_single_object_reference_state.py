"""Ordinary inpaints and safe imports after removing multi-view sessions."""

# ### Environment ###
from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ### Imports ###
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from housemaker import generation_state
from housemaker.generation_state import (
    GeneratedObjectRecord,
    GenerationData,
)
from housemaker.generation_workspace import GenerationWorkspace
from housemaker.settings_widget import GenerationServiceSettings
from tests.test_manual_inpaint_queue import _reference, _stroke, _VideoSource

# ### Application and fixtures ###
_application = QApplication.instance() or QApplication([])


@pytest.fixture
def workspace(tmp_path):
    widget = GenerationWorkspace(asset_directory=tmp_path)
    source = _VideoSource()
    widget._video_source = source
    widget._data = GenerationData(video_metadata=source.metadata)
    widget.set_runtime_settings(GenerationServiceSettings(meshy_api_key="test-only"))
    widget.show_frame(0)
    widget.video_view.set_strokes([_stroke(0.2, 0.25), _stroke(0.8, 0.75)])
    widget._handle_video_strokes_changed(widget.video_view.get_strokes())
    widget._data.set_frame_strokes(3, [_stroke(0.5, 0.5)])
    try:
        yield widget
    finally:
        widget.shutdown()
        widget.close()
        widget.deleteLater()
        _application.processEvents()


def _bank(*indices):
    return {
        str(index): [_stroke(0.2 + index * 0.01, 0.3).to_dict()]
        for index in indices
    }


def _legacy_payload(**session):
    return {
        "frame_strokes": _bank(2),
        "single_object_reference": {
            "geometry_model": "meshy-7.1",
            "geometry_resolution": "2k",
            "ordinary_frame_strokes": _bank(0, 3),
            "frame_strokes": _bank(2),
            # Removed images/slot metadata must not be decoded or reactivated.
            "reference_pngs": ["not-a-png"],
            "view_assignments": {"front": 999},
            **session,
        },
    }


# ### Legacy saved-mask recovery ###
@pytest.mark.parametrize(
    "mode",
    (
        {"multi_view_generation": True},
        {"multi_inpaint": True},
        {"multi_view": True},
    ),
)
def test_active_legacy_mode_restores_only_the_hidden_ordinary_masks(mode):
    restored = GenerationData.from_dict(_legacy_payload(**mode))

    assert set(restored.frame_strokes) == {0, 3}
    assert restored.geometry_model == "meshy-7.1"
    assert restored.geometry_resolution == "2k"
    assert not hasattr(restored, "single_object_reference")
    assert "single_object_reference" not in restored.to_dict()
    assert GenerationData.from_dict(json.loads(json.dumps(restored.to_dict()))) == restored


def test_disabled_legacy_mode_preserves_current_masks_not_its_stale_hidden_bank():
    restored = GenerationData.from_dict(
        _legacy_payload(multi_view_generation=False, multi_inpaint=True)
    )

    assert set(restored.frame_strokes) == {2}


def test_legacy_multiview_flag_on_an_unsupported_provider_was_not_an_active_bank():
    restored = GenerationData.from_dict(
        _legacy_payload(multi_view=True, geometry_model="meshy-t2")
    )

    assert set(restored.frame_strokes) == {2}
    assert restored.geometry_model == "meshy-t2"


def test_explicit_empty_ordinary_bank_does_not_restore_removed_mode_masks():
    restored = GenerationData.from_dict(
        _legacy_payload(multi_view_generation=True, ordinary_frame_strokes={})
    )

    assert restored.frame_strokes == {}


def test_legacy_save_without_a_hidden_bank_keeps_its_available_strokes():
    payload = _legacy_payload(multi_view_generation=True)
    del payload["single_object_reference"]["ordinary_frame_strokes"]

    assert set(GenerationData.from_dict(payload).frame_strokes) == {2}


def test_unknown_removed_reference_metadata_is_ignored():
    payload = _legacy_payload(multi_view_generation=False)
    payload["single_object_reference"].update(
        reference_pngs=object(), view_assignments=False, revision="invalid"
    )

    assert set(GenerationData.from_dict(payload).frame_strokes) == {2}


def test_legacy_import_preserves_generated_assets_and_placements(tmp_path):
    asset = tmp_path / "existing.glb"
    asset.write_bytes(b"untouched-existing-generated-file")
    record = GeneratedObjectRecord(
        object_id="existing", frame_index=0, object_name="Existing",
        pipeline={}, provider_task_id="task", asset_path=asset.name,
    )
    payload = _legacy_payload(multi_view_generation=True)
    payload["generated_objects"] = [record.to_dict()]

    restored = GenerationData.from_dict(payload)

    assert restored.generated_objects == [record]
    assert asset.read_bytes() == b"untouched-existing-generated-file"


def test_loading_recovered_masks_does_not_overwrite_them_when_video_is_missing(workspace):
    data = GenerationData.from_dict(_legacy_payload(multi_view_generation=True))

    workspace.set_data(data, defer_file_backed_preview=True)

    assert workspace.get_data() == data
    assert set(workspace.get_data().frame_strokes) == {0, 3}
    assert workspace.get_temporary_object_reference() is None


# ### Independent single-image geometry options ###
def test_geometry_options_round_trip_without_mode_or_reference_fields():
    data = GenerationData(geometry_model="meshy-7.1", geometry_resolution="2k")
    payload = data.to_dict()

    assert payload["geometry_model"] == "meshy-7.1"
    assert payload["geometry_resolution"] == "2k"
    assert "single_object_reference" not in payload
    assert data.clone() == data
    assert GenerationData.from_dict(payload) == data
    assert not hasattr(generation_state, "SingleObjectReferenceData")


def test_new_geometry_options_override_legacy_nested_provider_values():
    payload = _legacy_payload(multi_view_generation=False)
    payload.update(geometry_model="meshy-t2.5", geometry_resolution="standard")

    restored = GenerationData.from_dict(payload)

    assert restored.geometry_model == "meshy-t2.5"
    assert restored.geometry_resolution == "standard"


def test_provider_changes_never_hide_or_replace_ordinary_masks(workspace):
    queue = workspace.inpaint_queue()
    before = workspace.get_data().frame_strokes

    workspace.set_geometry_generation_options("meshy-7.1", "2k")
    workspace.set_geometry_generation_options("meshy-t2", "standard")
    workspace.set_geometry_generation_options("meshy-t2.5", "standard")

    assert workspace.geometry_generation_options() == ("meshy-t2.5", "standard")
    assert workspace.inpaint_queue() == queue
    assert workspace.get_data().frame_strokes == before


def test_71_generates_each_ordinary_blob_from_one_image_without_front_slots(workspace):
    workspace.set_geometry_generation_options("meshy-7.1", "2k")
    keys = workspace._current_inpaint_keys()
    assert len(keys) == 3
    for key, value in zip(keys, (20, 40, 80), strict=True):
        assert workspace.set_inpaint_reference(key, _reference(value))

    requests = workspace._build_generation_requests()

    assert len(requests) == 3
    assert [request.inpaint_key for request in requests] == list(keys)
    for request, value in zip(requests, (20, 40, 80), strict=True):
        assert request.geometry_model == "meshy-7.1"
        assert request.geometry_resolution == "2k"
        assert request.reference_images_bgra == ()
        assert not hasattr(request, "geometry_reference_images_bgra")
        np.testing.assert_array_equal(request.selected_object_bgra, _reference(value))
    assert workspace._data.current_frame_index == 0


# ### Ordinary per-blob editing and navigation ###
def test_pasting_again_replaces_only_the_current_blobs_reference(workspace):
    queue = workspace.inpaint_queue()
    workspace.set_temporary_object_reference(_reference(20))
    workspace.set_temporary_object_reference(_reference(40))

    assert workspace.inpaint_queue() == queue
    np.testing.assert_array_equal(workspace.get_current_object_generation_reference(), _reference(40))
    assert workspace.navigate_inpaints(1)
    assert workspace.get_temporary_object_reference() is None
    assert workspace.navigate_inpaints(-1)
    np.testing.assert_array_equal(workspace.get_current_object_generation_reference(), _reference(40))


def test_clear_mask_removes_only_selected_blob_and_preserves_other_frames(workspace):
    assert workspace.inpaint_queue() == ((0, 0), (0, 1), (3, 0))

    workspace.clear_mask_button.click()

    assert workspace.inpaint_queue() == ((0, 0), (3, 0))
    assert workspace._data.strokes_for_frame(3) == [_stroke(0.5, 0.5)]


def test_reference_prompts_symmetry_and_pbr_maps_remain_individual(workspace):
    first, second, _third = workspace._current_inpaint_keys()
    assert workspace.set_inpaint_reference_view_preferences(first, "front", "First object")
    assert workspace.set_inpaint_reference_view_preferences(second, "three-quarter", "Second object")
    assert workspace.set_inpaint_symmetric_division(first, True)
    assert workspace.set_inpaint_symmetric_division(second, False)
    assert workspace.set_inpaint_pbr_maps(first, ("normal",))
    assert workspace.set_inpaint_pbr_maps(second, ("roughness", "metallic"))

    assert workspace.navigate_inpaints(1)

    assert workspace.current_inpaint_reference_key() == second
    assert not workspace.symmetric_division_checkbox.isChecked()
    assert workspace._get_enabled_pbr_maps() == ("roughness", "metallic")
    assert workspace.get_inpaint_reference_view_preferences(second) == ("three-quarter", "Second object")
    assert workspace.get_inpaint_reference_view_preferences(first) == ("front", "First object")
    assert workspace.navigate_inpaints(-1)
    assert workspace.symmetric_division_checkbox.isChecked()
    assert workspace._get_enabled_pbr_maps() == ("normal",)


def test_removed_workflow_has_no_remaining_workspace_controls_or_session_apis(workspace):
    for name in (
        "multi_view_generation_enabled", "single_object_reference_active",
        "set_multi_view_generation_enabled", "set_single_object_reference_modes",
        "append_single_object_reference", "navigate_single_object_reference",
        "assign_single_object_reference_view", "get_single_object_geometry_views",
        "get_single_object_inpaint_references", "clear_single_object_reference_session",
    ):
        assert not hasattr(workspace, name)
