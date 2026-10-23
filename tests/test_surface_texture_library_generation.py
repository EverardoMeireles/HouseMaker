"""Ordinary materials can be generated first and assigned to any surface later."""

# ### Environment and imports ###
from __future__ import annotations

import json
import os
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.generation_jobs import JOB_STATUS_COMPLETED
from housemaker.generation_workspace import GenerationWorkspace
from housemaker.glb import convert_to_glb
from housemaker.settings_widget import GenerationServiceSettings
from housemaker.surface_geometry import build_fixed_surfaces
from housemaker.surface_texture_providers import SurfaceTextureResult
from housemaker.surface_texture_state import (
    SURFACE_TYPE_GENERIC,
    SURFACE_TYPES,
    SurfaceTextureAssignment,
    SurfaceTextureData,
    surface_id_accepts_texture_type,
)
from housemaker.surface_texture_variants import build_surface_texture_variants
from housemaker.surface_texture_workspace import (
    SURFACE_TEXTURE_JOB_KIND,
    SurfaceTextureGenerationWorkspace,
    _build_material_prompt,
    _PreparedSurfaceTextureOutput,
)
from tests.test_surface_texture_workspace import (
    _FakeProvider,
    _surface_assignment_with_variants,
    _test_level,
    _test_stroke,
    _texture_png,
    _write_test_video,
)

# ### Local fixtures ###
_application = QApplication.instance() or QApplication([])


@pytest.fixture
def workspace(tmp_path):
    objects = GenerationWorkspace(asset_directory=tmp_path / "objects")
    widget = SurfaceTextureGenerationWorkspace(
        provider=_FakeProvider(), asset_directory=tmp_path / "surfaces",
        application_settings=ApplicationSettingsStore(tmp_path / "settings.json"),
        shared_controls=objects.get_shared_controls(),
    )
    widget.set_runtime_settings(GenerationServiceSettings(meshy_api_key="test-only"))
    widget.set_levels((_test_level(),))
    widget.set_temporary_reference(np.full((24, 32, 4), (30, 80, 120, 255), np.uint8))
    try:
        yield widget
    finally:
        widget.shutdown()
        objects.shutdown()
        widget.close()
        objects.close()
        widget.deleteLater()
        objects.deleteLater()
        _application.processEvents()


def _surfaces():
    return tuple(build_fixed_surfaces((_test_level(),)))


def _prepared(request):
    return (
        _PreparedSurfaceTextureOutput(
            request.surface_ids, build_surface_texture_variants(_texture_png()),
        ),
    )


def _result():
    return SurfaceTextureResult(provider="meshy", texture_png=_texture_png(), task_id="fake")


# ### Reusable material metadata and legacy state ###
@pytest.mark.parametrize("texture_type", ("wall", "floor", "ceiling", SURFACE_TYPE_GENERIC))
@pytest.mark.parametrize("target", (
    "level:2/room:0/wall:1:2", "level:2/room:0/floor", "level:2/room:0/ceiling",
    f"level:2/edit-face:{'a' * 32}:floor",
    f"level:2/sloped-ceiling:{'b' * 32}:part:left:ceiling",
    "level:2/opening:window:MixedCase/reveal:header",
    f"stair:{'c' * 32}/part:treads:floor",
))
def test_ordinary_texture_origin_never_restricts_ordinary_surface_targets(texture_type, target):
    assert surface_id_accepts_texture_type(target, texture_type)


def test_generic_metadata_does_not_change_structural_surface_types():
    assert SURFACE_TYPE_GENERIC not in SURFACE_TYPES
    with pytest.raises(ValueError):
        surface_id_accepts_texture_type("level:2/floor", "unknown")
    assert not surface_id_accepts_texture_type(
        f"trim:{'a' * 32}/part:front:wall", SURFACE_TYPE_GENERIC,
    )


@pytest.mark.parametrize("texture_type", ("wall", "floor", "ceiling", SURFACE_TYPE_GENERIC))
def test_mixed_assignment_round_trips_without_relabelling_legacy_origin(texture_type):
    assignment = SurfaceTextureAssignment(
        assignment_id="material", surface_type=texture_type,
        surface_ids=("level:2/room:0/wall:1:2", "level:2/room:0/floor", "level:2/room:0/ceiling"),
        provider="meshy", asset_path="material.png", display_name="Reusable plaster",
    )
    data = SurfaceTextureData(assignments=[assignment])

    restored = SurfaceTextureData.from_dict(json.loads(json.dumps(data.to_dict())))

    assert restored.assignments == [assignment]
    assert restored.assignments[0].surface_type == texture_type


# ### Optional target snapshots and reference sources ###
def test_pasted_reference_without_targets_is_ready_for_generic_generation(workspace):
    assert workspace.generate_button.isEnabled()
    request = workspace._build_request()

    assert request is not None
    assert request.surface_type == SURFACE_TYPE_GENERIC
    assert request.surface_ids == ()
    assert request.combined_area_m2 == 0.0
    assert len(request.reference_pngs) == 1
    assert "a wall" not in request.prompt
    assert "0.00 square" not in request.prompt


def test_video_inpaint_without_targets_builds_a_source_only_request(workspace, tmp_path):
    video = tmp_path / "source.avi"
    _write_test_video(video)
    workspace.load_video(str(video))
    workspace.video_view.set_strokes([_test_stroke()])
    workspace._handle_video_strokes_changed(workspace.video_view.get_strokes())

    request = workspace._build_request()

    assert workspace.generate_button.isEnabled()
    assert request is not None and request.surface_ids == ()
    assert request.reference_pngs
    assert request.reference_frame_indices == (0,)


def test_missing_reference_or_key_still_prevents_generation(workspace):
    workspace.clear_temporary_reference()
    assert not workspace.generate_button.isEnabled()
    assert workspace._build_request() is None
    workspace.set_temporary_reference(np.full((8, 8, 4), 255, np.uint8))
    workspace.set_runtime_settings(GenerationServiceSettings())
    assert not workspace.generate_button.isEnabled()
    assert workspace._build_request() is None


def test_mixed_selected_targets_are_optional_autoapply_not_a_material_type(workspace):
    surfaces = _surfaces()
    targets = tuple(next(surface for surface in surfaces if surface.surface_type == kind)
                    for kind in ("wall", "floor", "ceiling"))
    workspace.set_scene_surface_selection(targets)

    request = workspace._build_request()

    assert request is not None
    assert request.surface_type == SURFACE_TYPE_GENERIC
    assert request.surface_ids == tuple(target.surface_id for target in targets)
    assert request.combined_area_m2 == pytest.approx(sum(target.area_square_meters for target in targets))
    assert workspace.generate_button.isEnabled()
    assert workspace.get_data().selected_surface_ids == ()


def test_material_prompt_remains_generic_even_for_legacy_typed_callers():
    prompt = _build_material_prompt("wall", 12.0, "Pale plaster")
    assert "a wall" not in prompt
    assert "reusable on architectural surfaces" in prompt
    assert "Pale plaster" in prompt


# ### Completion retains source-only intent and stale targets safely ###
def test_source_only_completion_does_not_paint_later_selection(workspace):
    request = workspace._build_request()
    wall = next(surface for surface in _surfaces() if surface.surface_type == "wall")
    workspace.set_scene_surface_selection((wall,))
    completed = QSignalSpy(workspace.generation_completed)
    changed = QSignalSpy(workspace.surface_content_changed)

    assert workspace._handle_generation_succeeded(request, _result(), _prepared(request))

    assignment = workspace.get_assignments()[0]
    assert assignment.surface_ids == ()
    assert assignment.surface_type == SURFACE_TYPE_GENERIC
    assert len(assignment.texture_variants) == 3
    assert all(workspace._resolve_asset_path(variant.asset_path).is_file()
               for variant in assignment.texture_variants)
    assert completed.count() == 1 and changed.count() == 0
    workspace.set_scene_surface_selection(_surfaces())
    assert workspace.get_assignments()[0].surface_ids == ()
    assert changed.count() == 0
    assert "library" in workspace.status_label.text()


def test_completion_applies_only_the_original_target_snapshot(workspace):
    surfaces = _surfaces()
    wall = next(surface for surface in surfaces if surface.surface_type == "wall")
    floor = next(surface for surface in surfaces if surface.surface_type == "floor")
    workspace.set_scene_surface_selection((wall,))
    request = workspace._build_request()
    workspace.set_scene_surface_selection((floor,))
    changed = QSignalSpy(workspace.surface_content_changed)

    assert workspace._handle_generation_succeeded(request, _result(), _prepared(request))

    assert workspace.get_assignments()[0].surface_ids == (wall.surface_id,)
    assert changed.count() == 1


def test_source_only_request_cannot_gain_targets_from_a_mismatched_output(workspace):
    request = workspace._build_request()
    wall = next(surface for surface in _surfaces() if surface.surface_type == "wall")
    output = _PreparedSurfaceTextureOutput(
        (wall.surface_id,), build_surface_texture_variants(_texture_png()),
    )

    assert workspace._handle_generation_succeeded(request, _result(), (output,))

    assert workspace.get_assignments()[0].surface_ids == ()


def test_generic_completion_can_autoapply_its_mixed_original_targets(workspace):
    targets = _surfaces()
    workspace.set_scene_surface_selection(targets)
    request = workspace._build_request()
    changed = QSignalSpy(workspace.surface_content_changed)

    assert workspace._handle_generation_succeeded(request, _result(), _prepared(request))

    assignment = workspace.get_assignments()[0]
    assert assignment.surface_ids == request.surface_ids
    assert assignment.surface_type == SURFACE_TYPE_GENERIC
    assert changed.count() == 1
    assert SurfaceTextureData.from_dict(workspace.get_data().to_dict()).assignments == [assignment]


def test_stale_async_targets_save_library_assets_instead_of_failing_or_retargeting(workspace):
    wall = next(surface for surface in _surfaces() if surface.surface_type == "wall")
    request = replace(workspace._build_request(), surface_ids=(wall.surface_id, "level:2/wall:99:100"))
    output = workspace._persist_texture_variants(
        request.surface_ids, "stale-result", build_surface_texture_variants(_texture_png()),
    )
    manager = workspace._job_manager
    job = manager.create_job(
        kind=SURFACE_TEXTURE_JOB_KIND, requested_name="", default_name="Material", stage="Generating",
    )
    worker = SimpleNamespace(claim_saved_outputs=Mock(), discard_unclaimed_outputs=Mock())
    workspace._generation_workers[job.job_id] = worker
    workspace._generation_submission_order[job.job_id] = 1
    workspace.set_scene_surface_selection((wall,))
    changed = QSignalSpy(workspace.surface_content_changed)

    workspace._handle_generation_job_succeeded(job.job_id, request, _result(), (output,))

    assert manager.get_job(job.job_id).status == JOB_STATUS_COMPLETED
    worker.claim_saved_outputs.assert_called_once()
    worker.discard_unclaimed_outputs.assert_not_called()
    assert workspace.get_assignments()[0].surface_ids == ()
    assert all(workspace._resolve_asset_path(variant.asset_path).is_file() for variant in output.variants)
    assert changed.count() == 0
    assert "original target surfaces changed" in workspace.status_label.text()


# ### Explicit reuse is cross-category and undoable ###
@pytest.mark.parametrize("texture_type", ("wall", "floor", "ceiling", SURFACE_TYPE_GENERIC))
def test_library_texture_can_be_explicitly_assigned_across_ordinary_categories(workspace, texture_type):
    assignment = _surface_assignment_with_variants(
        workspace._asset_directory, "library", (), surface_type=texture_type,
    )
    workspace._data.assignments = [assignment]
    target_ids = tuple(surface.surface_id for surface in _surfaces())
    before = workspace.snapshot_assignments()
    changed = QSignalSpy(workspace.surface_content_changed)

    assert workspace.assignment_targets_are_valid(assignment.assignment_id, target_ids)
    assert workspace.apply_assignment_texture(assignment.assignment_id, target_ids)

    assert set(workspace.get_assignments()[0].surface_ids) == set(target_ids)
    assert changed.count() == 1
    assert workspace.restore_assignment_snapshot(before)
    assert workspace.get_assignments() == before
    assert workspace._resolve_asset_path(assignment.asset_path).is_file()


def test_generic_planar_generation_rejects_specialized_trim_targets(workspace):
    request = replace(workspace._build_request(), surface_ids=(f"trim:{'a' * 32}/part:front:wall",))
    with patch.object(workspace, "_get_surface_selection_snapshot", return_value=SimpleNamespace(
        surface_ids=request.surface_ids, combined_area_m2=1.0,
    )):
        assert workspace._build_request() is None
    assert "architectural trim" in workspace.status_label.text()


# ### Generic library metadata never controls projection or export geometry ###
def test_generic_mixed_assignment_exports_using_each_targets_structural_type(workspace):
    targets = tuple(
        next(surface for surface in _surfaces() if surface.surface_type == kind)
        for kind in ("wall", "floor", "ceiling")
    )
    assignment = _surface_assignment_with_variants(
        workspace._asset_directory, "generic-export", tuple(target.surface_id for target in targets),
        surface_type=SURFACE_TYPE_GENERIC,
    )
    workspace._data.assignments = [assignment]

    model = convert_to_glb(
        (_test_level(),), surface_materials=workspace.get_surface_material_sources(),
    )

    assert model.glb_bytes
    assert {surface.surface_id: surface.surface_type for surface in model.preview_textured_surfaces} == {
        target.surface_id: target.surface_type for target in targets
    }
    for surface in model.preview_textured_surfaces:
        assert surface.surface_type != SURFACE_TYPE_GENERIC
        assert np.all(np.isfinite(surface.mesh.visual.uv))
