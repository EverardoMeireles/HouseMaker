"""Visibility baking stays background-only, cancellable and tied to the scene."""

# ### Environment and imports ###
from __future__ import annotations

import os
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.generation_jobs import (
    JOB_STATUS_CANCELLED,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JOB_STATUS_RUNNING,
)
from housemaker.main import BlueprintWorkspace
from housemaker.models import (
    GROUND_LEVEL_INDEX,
    RoomData,
    VertexData,
    create_default_levels,
)
from housemaker.project_io import ProjectData
from housemaker.pvs_workspace import _build_snapshot_model
from housemaker.surface_geometry import build_fixed_surfaces
from tests.test_pvs_state import _bake
from tests.test_scene_export import _source_model

# ### Temporary workspaces ###
_application = QApplication.instance() or QApplication([])
_application.setQuitOnLastWindowClosed(False)


@pytest.fixture
def workspace(tmp_path):
    widget = BlueprintWorkspace(
        application_settings=ApplicationSettingsStore(tmp_path / "settings.json")
    )
    _application.processEvents()
    try:
        yield widget
    finally:
        widget.shutdown()
        widget.close()
        _application.processEvents()


def _matching_bake(workspace):
    return replace(
        _bake(),
        authoring_signature=workspace._build_pvs_authoring_signature(),
    )


def _add_room(level, *, offset_pixels=0):
    vertices = VertexData()
    boundary = tuple(
        vertices.add_vertex(x + offset_pixels, y).id
        for x, y in ((0, 0), (200, 0), (200, 200), (0, 200))
    )
    for start, end in zip(boundary, (*boundary[1:], boundary[0]), strict=True):
        vertices.add_edge(start, end)
    center = vertices.add_vertex(100 + offset_pixels, 100)
    level.vertex_data = vertices
    level.plan_meters_per_pixel = 0.01
    level.rooms = [RoomData("Tiny room", boundary, center.id, (120, 140, 160))]


class _ControlledBakeThread(QObject):
    """A manual worker which never starts native threads or constructs meshes."""

    progress = Signal(str)
    finished = Signal()

    def __init__(self, snapshot, parent=None):
        super().__init__(parent)
        self.snapshot = snapshot
        self.result = None
        self.error_message = None
        self.was_cancelled = False
        self.interrupted = False
        self.started = False
        self.waited = False

    def start(self):
        self.started = True

    def requestInterruption(self):
        self.interrupted = True

    def isInterruptionRequested(self):
        return self.interrupted

    def wait(self):
        self.waited = True
        return True

    def complete(self, result=None, *, error=None):
        self.result = result
        self.error_message = error
        self.finished.emit()


@pytest.fixture
def fake_bake(workspace, monkeypatch):
    snapshot = SimpleNamespace(
        authoring_signature=workspace._build_pvs_authoring_signature()
    )
    monkeypatch.setattr("housemaker.pvs_workspace.PvsBakeThread", _ControlledBakeThread)
    monkeypatch.setattr(workspace, "_capture_pvs_snapshot", lambda: snapshot)

    def start():
        workspace.bake_visibility_button.click()
        _application.processEvents()
        worker = workspace._pvs_bake_thread
        assert isinstance(worker, _ControlledBakeThread)
        assert worker.started
        return worker, workspace._pvs_job_id

    return start


# ### Levels UI and inexpensive stale detection ###
def test_bake_button_and_status_are_inside_the_canvas_levels_section(workspace):
    button = workspace.bake_visibility_button

    assert button.text() == "Bake visibility(PVS)"
    assert workspace.levels_group.isAncestorOf(button)
    assert workspace.levels_group.isAncestorOf(workspace.pvs_status_label)
    assert workspace._pvs_visibility is None
    assert workspace._pvs_bake_thread is None


def test_authoring_fingerprint_tracks_structure_not_camera_or_current_level(workspace):
    initial = workspace._build_pvs_authoring_signature()
    assert len(initial) == 64
    workspace.viewer.select_wall_target(None)
    workspace.viewer.set_first_person_camera_pose(
        replace(
            workspace.viewer.get_first_person_camera_pose(), x=4.0, yaw_degrees=30.0
        )
    )
    workspace.current_level_index = 0
    assert workspace._build_pvs_authoring_signature() == initial

    workspace.levels[GROUND_LEVEL_INDEX].height_meters += 0.2
    assert workspace._build_pvs_authoring_signature() != initial


def test_eye_height_is_part_of_the_bake_validity_fingerprint(workspace):
    initial = workspace._build_pvs_authoring_signature()
    workspace.settings_widget.first_person_eye_height_spinbox.setValue(1.88)
    assert workspace._build_pvs_authoring_signature() != initial


# ### Background jobs, cancellation and stale results ###
def test_preparation_publishes_one_job_before_capture_and_cannot_start_twice(
    workspace, monkeypatch
):
    capture = []
    monkeypatch.setattr("housemaker.pvs_workspace.PvsBakeThread", _ControlledBakeThread)

    def snapshot():
        capture.append(True)
        return SimpleNamespace(
            authoring_signature=workspace._build_pvs_authoring_signature()
        )

    monkeypatch.setattr(workspace, "_capture_pvs_snapshot", snapshot)
    workspace._handle_bake_visibility_clicked()
    assert not capture
    assert len(workspace.job_manager.jobs()) == 1
    assert workspace.job_manager.jobs()[0].status == JOB_STATUS_RUNNING
    assert not workspace.bake_visibility_button.isEnabled()
    workspace._refresh_pvs_status()
    assert not workspace.bake_visibility_button.isEnabled()
    workspace._handle_bake_visibility_clicked()
    assert len(workspace.job_manager.jobs()) == 1

    _application.processEvents()
    assert capture == [True]


def test_background_progress_and_completion_update_jobs_and_install_the_bake(
    workspace, fake_bake
):
    worker, job_id = fake_bake()
    assert workspace.jobs_window.isVisible()
    assert not workspace.bake_visibility_button.isEnabled()
    worker.progress.emit("Baking potentially visible sets (63%)")
    assert workspace.job_manager.get_job(job_id).progress == 63
    assert "63%" in workspace.pvs_status_label.text()

    bake = _matching_bake(workspace)
    worker.complete(bake)

    assert workspace._pvs_visibility == bake
    assert workspace._pvs_bake_thread is None
    assert workspace.job_manager.get_job(job_id).status == JOB_STATUS_COMPLETED
    assert workspace.bake_visibility_button.isEnabled()
    assert "baked" in workspace.pvs_status_label.text().lower()


def test_cancelling_a_worker_does_not_install_its_late_result(workspace, fake_bake):
    previous = _matching_bake(workspace)
    workspace._pvs_visibility = previous
    worker, job_id = fake_bake()
    assert workspace._cancel_pvs_bake()
    assert worker.interrupted

    worker.complete(replace(previous, source_signature="b" * 64))

    assert workspace._pvs_visibility == previous
    assert workspace.job_manager.get_job(job_id).status == JOB_STATUS_CANCELLED
    assert workspace.bake_visibility_button.isEnabled()


def test_late_progress_from_a_finished_worker_cannot_overwrite_a_new_job(
    workspace, fake_bake
):
    old_worker, old_job_id = fake_bake()
    old_worker.complete(_matching_bake(workspace))
    new_worker, new_job_id = fake_bake()
    new_worker.progress.emit("Building export geometry (20%)")
    current_job = workspace.job_manager.get_job(new_job_id)
    current_status = workspace.pvs_status_label.text()

    workspace._handle_pvs_progress("Old job finishing (99%)", old_worker)

    assert workspace.job_manager.get_job(new_job_id) == current_job
    assert workspace.pvs_status_label.text() == current_status
    assert workspace.job_manager.get_job(old_job_id).status == JOB_STATUS_COMPLETED


def test_late_cancel_is_remembered_even_when_qthread_interruption_is_a_noop(
    workspace, fake_bake
):
    previous = _matching_bake(workspace)
    workspace._pvs_visibility = previous
    worker, job_id = fake_bake()
    worker.started = False  # The native run finished, but finished is still queued.
    with patch.object(worker, "requestInterruption"):
        assert workspace._cancel_pvs_bake()
    assert not worker.isInterruptionRequested()

    worker.complete(replace(previous, source_signature="b" * 64))

    assert workspace._pvs_visibility == previous
    assert workspace.job_manager.get_job(job_id).status == JOB_STATUS_CANCELLED
    assert workspace.bake_visibility_button.isEnabled()


def test_cancel_before_capture_does_not_create_a_worker(workspace, monkeypatch):
    created = []
    monkeypatch.setattr(
        workspace, "_capture_pvs_snapshot", lambda: created.append(True)
    )
    workspace._handle_bake_visibility_clicked()
    job_id = workspace._pvs_job_id
    assert workspace._cancel_pvs_bake()
    _application.processEvents()

    assert not created
    assert workspace._pvs_bake_thread is None
    assert workspace.job_manager.get_job(job_id).status == JOB_STATUS_CANCELLED
    assert workspace.bake_visibility_button.isEnabled()


def test_scene_changes_discard_completed_worker_data_and_keep_the_previous_bake(
    workspace, fake_bake
):
    previous = _matching_bake(workspace)
    workspace._pvs_visibility = previous
    worker, job_id = fake_bake()
    workspace.levels[GROUND_LEVEL_INDEX].height_meters += 0.2
    worker.complete(replace(previous, source_signature="b" * 64))

    assert workspace._pvs_visibility == previous
    assert workspace.job_manager.get_job(job_id).status == JOB_STATUS_FAILED
    assert "changed" in workspace.pvs_status_label.text().lower()


def test_failed_bake_preserves_previous_data_and_explains_the_error(
    workspace, fake_bake
):
    previous = _matching_bake(workspace)
    workspace._pvs_visibility = previous
    worker, job_id = fake_bake()
    worker.complete(error="No walkable camera cells")

    assert workspace._pvs_visibility == previous
    assert workspace.job_manager.get_job(job_id).status == JOB_STATUS_FAILED
    assert "No walkable camera cells" in workspace.pvs_status_label.text()
    assert workspace.bake_visibility_button.isEnabled()


def test_shutdown_requests_cancellation_and_joins_the_worker(workspace, fake_bake):
    worker, job_id = fake_bake()
    workspace.shutdown()

    assert worker.interrupted
    assert worker.waited
    assert workspace._pvs_bake_thread is None
    assert workspace.job_manager.get_job(job_id).status == JOB_STATUS_CANCELLED


@pytest.mark.parametrize("upper_included", (False, True))
def test_tiny_rooms_bake_on_a_real_worker_and_match_only_included_export_geometry(
    workspace, upper_included
):
    level = workspace.levels[GROUND_LEVEL_INDEX]
    _add_room(level)
    upper = workspace.levels[GROUND_LEVEL_INDEX + 1]
    _add_room(upper, offset_pixels=1000)
    upper_floor = next(
        surface
        for surface in build_fixed_surfaces(workspace.levels)
        if surface.level_index == upper.index and surface.surface_type == "floor"
    )
    upper_mid_x = float(upper_floor.mesh.bounds[:, 0].mean())
    upper.include_in_export = upper_included

    workspace.bake_visibility_button.click()
    job_id = workspace._pvs_job_id
    deadline = time.monotonic() + 15
    while workspace._pvs_job_id is not None and time.monotonic() < deadline:
        _application.processEvents()
        time.sleep(0.005)

    job = workspace.job_manager.get_job(job_id)
    assert workspace._pvs_bake_thread is None, (
        "Tiny-room visibility worker did not finish"
    )
    assert job.status == JOB_STATUS_COMPLETED, job.stage
    bake = workspace._pvs_visibility
    assert bake is not None
    assert bake.cells and bake.clusters
    assert bake.coordinate_system == "gltf-y-up"
    assert bake.authoring_signature == workspace._build_pvs_authoring_signature()
    assert (
        any(cell.minimum[0] <= upper_mid_x <= cell.maximum[0] for cell in bake.cells)
        is upper_included
    )

    snapshot = workspace._capture_pvs_snapshot()
    _base, model = _build_snapshot_model(snapshot)
    partitioned, exported_visibility = workspace._prepare_pvs_export_model(
        model, (), ()
    )

    assert exported_visibility == bake
    assert partitioned.glb_bytes
    assert exported_visibility.source_signature == bake.source_signature


# ### Missing or outdated visibility never removes exported geometry ###
def test_export_without_a_bake_keeps_the_original_model(workspace):
    model = _source_model()
    with patch("housemaker.pvs_workspace.prepare_pvs_geometry") as partition:
        exported, visibility = workspace._prepare_pvs_export_model(model, (), ())

    assert exported is model
    assert visibility is None
    partition.assert_not_called()


def test_unsupported_partition_validation_fails_open_without_blocking_export(workspace):
    bake = _matching_bake(workspace)
    workspace._pvs_visibility = bake
    model = _source_model()
    with patch(
        "housemaker.pvs_workspace.prepare_pvs_geometry",
        side_effect=ValueError("Ambiguous GLB node names"),
    ):
        exported, visibility = workspace._prepare_pvs_export_model(model, (), ())

    assert exported is model
    assert visibility is None
    assert workspace._pvs_visibility == bake
    assert "without pvs" in workspace._pvs_export_note.lower()
    assert "Ambiguous GLB node names" in workspace._pvs_export_note


@pytest.mark.parametrize("changed_membership", (False, True))
def test_export_with_changed_geometry_or_cluster_membership_fails_open(
    workspace, changed_membership
):
    bake = _matching_bake(workspace)
    workspace._pvs_visibility = bake
    model = _source_model()
    clusters = (
        (replace(bake.clusters[0], maximum=(7, 3, 0)),)
        if changed_membership
        else bake.clusters
    )
    signature = bake.source_signature if changed_membership else "b" * 64
    with patch(
        "housemaker.pvs_workspace.prepare_pvs_geometry",
        return_value=(b"partitioned", clusters, (), signature),
    ):
        exported, visibility = workspace._prepare_pvs_export_model(model, (), ())

    assert exported is model
    assert visibility is None
    assert workspace._pvs_visibility == bake
    assert "outdated" in workspace._pvs_export_note.lower()
    assert (
        "rebake" in workspace._pvs_export_note.lower()
        or "bake visibility" in workspace._pvs_export_note.lower()
    )


def test_unchanged_export_geometry_can_revalidate_a_changed_authoring_fingerprint(
    workspace,
):
    bake = replace(_matching_bake(workspace), authoring_signature="b" * 64)
    workspace._pvs_visibility = bake
    model = _source_model()
    with patch(
        "housemaker.pvs_workspace.prepare_pvs_geometry",
        return_value=(b"partitioned", bake.clusters, (), bake.source_signature),
    ):
        exported, visibility = workspace._prepare_pvs_export_model(model, (), ())

    assert exported.glb_bytes == b"partitioned"
    assert visibility.source_signature == bake.source_signature
    assert visibility.authoring_signature == workspace._build_pvs_authoring_signature()
    assert workspace._pvs_visibility == visibility


def test_snapshot_owns_level_geometry_instead_of_reading_later_edits(workspace):
    _add_room(workspace.levels[GROUND_LEVEL_INDEX])
    snapshot = workspace._capture_pvs_snapshot()
    original_height = snapshot.levels[GROUND_LEVEL_INDEX].height_meters
    original_x = snapshot.levels[GROUND_LEVEL_INDEX].vertex_data.vertices[0].x
    workspace.levels[GROUND_LEVEL_INDEX].height_meters += 0.5
    vertices = workspace.levels[GROUND_LEVEL_INDEX].vertex_data.vertices
    vertices[0] = replace(vertices[0], x=original_x + 30)

    assert snapshot.levels[GROUND_LEVEL_INDEX].height_meters == original_height
    assert snapshot.levels[GROUND_LEVEL_INDEX].vertex_data.vertices[0].x == original_x


# ### Project persistence and runtime export integration ###
def test_loaded_project_restores_its_bake_and_old_projects_clear_previous_bakes(
    workspace,
):
    bake = _matching_bake(workspace)
    project = ProjectData(
        None, GROUND_LEVEL_INDEX, create_default_levels(), visibility=bake
    )
    workspace._apply_loaded_project(project)
    assert workspace._pvs_visibility == bake
    workspace._apply_loaded_project(
        ProjectData(None, GROUND_LEVEL_INDEX, create_default_levels())
    )
    assert workspace._pvs_visibility is None


def test_save_forwards_the_existing_valid_bake_to_project_persistence(
    workspace, tmp_path
):
    bake = _matching_bake(workspace)
    workspace._pvs_visibility = bake
    with (
        patch(
            "housemaker.main.QFileDialog.getSaveFileName",
            return_value=(str(tmp_path / "house.json"), ""),
        ),
        patch("housemaker.main.save_project") as save,
        patch("housemaker.main.QMessageBox.information"),
        patch("housemaker.main.QMessageBox.critical") as failure,
    ):
        workspace._handle_save_clicked()

    failure.assert_not_called()
    assert save.call_args.kwargs["visibility"] == bake


def test_export_forwards_the_validated_partitioned_bake_to_runtime_json(
    workspace, tmp_path
):
    bake = _matching_bake(workspace)
    export_path = tmp_path / "house.glb"
    model, partitioned_model = object(), object()
    with (
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
            return_value=(model, ("revision",)),
        ),
        patch.object(workspace, "_build_export_placed_models", return_value=((), ())),
        patch.object(
            workspace,
            "_prepare_pvs_export_model",
            return_value=(partitioned_model, bake),
        ) as prepare,
        patch("housemaker.main.export_glb_file", return_value=export_path) as export,
        patch(
            "housemaker.main.write_runtime_scene_manifest",
            return_value=export_path.with_suffix(".json"),
        ) as manifest,
        patch.object(workspace, "_ensure_viewer_preview_current"),
        patch("housemaker.main.QMessageBox.information"),
        patch("housemaker.main.QMessageBox.critical") as failure,
    ):
        workspace._handle_glb_export_clicked()

    failure.assert_not_called()
    prepare.assert_called_once()
    assert export.call_args.args[0] is partitioned_model
    assert manifest.call_args.kwargs["visibility"] == bake
