# ### Environment ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ### Imports ###
from dataclasses import replace
from unittest.mock import patch

import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.main import BlueprintWorkspace
from housemaker.surface_texture_state import SurfaceTextureAssignment

# ### Fixtures and helpers ###
_application = QApplication.instance() or QApplication([])
_application.setQuitOnLastWindowClosed(False)
WALL = "level:2/wall:1:2"
FLOOR = "level:2/floor"


@pytest.fixture
def workspace(tmp_path):
    with patch.object(BlueprintWorkspace, "_queue_viewer_preview_refresh"):
        widget = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(tmp_path / "settings.json")
        )
        widget._set_mesh_edit_update_delay_seconds(60.0)
        _application.processEvents()
        widget._queue_viewer_preview_refresh.reset_mock()
        try:
            yield widget
        finally:
            widget.shutdown()
            widget.close()
            # Settle responsive-layout callbacks before deleting their widgets.
            _application.processEvents()
            _application.processEvents()
            widget.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _assignment(surface_ids=(WALL,), assignment_id="paint"):
    return SurfaceTextureAssignment(
        assignment_id=assignment_id,
        surface_type="surface",
        surface_ids=surface_ids,
        provider="test",
        asset_path=f"{assignment_id}.png",
    )


def _publish(workspace, *assignments):
    with patch.object(
        workspace.surface_texture_generation, "get_assignments", return_value=assignments
    ):
        workspace._handle_surface_texture_content_changed()
        _application.processEvents()


# ### Assignment quiet period and preview lifetime ###
def test_assignments_wait_with_all_target_faces_marked(workspace):
    revision = workspace._viewer_preview_revision
    _publish(workspace, _assignment((WALL, FLOOR)))

    assert workspace._pending_surface_texture_ids == {WALL, FLOOR}
    assert set(workspace.viewer.get_pending_surface_texture_preview_ids()) == {WALL, FLOOR}
    assert workspace._surface_texture_mesh_update_timer.isActive()
    assert workspace._viewer_preview_revision > revision
    workspace._queue_viewer_preview_refresh.assert_not_called()
    assert workspace._has_pending_mesh_updates()
    assert workspace.confirm_mesh_updates_button.isEnabled()

    workspace._commit_pending_surface_texture_mesh_update()
    assert not workspace._pending_surface_texture_ids
    assert not workspace._surface_texture_mesh_update_timer.isActive()
    assert set(workspace._committing_surface_texture_revisions) == {WALL, FLOOR}
    workspace._queue_viewer_preview_refresh.assert_called_once()
    assert set(workspace.viewer.get_pending_surface_texture_preview_ids()) == {WALL, FLOOR}

    workspace._canvas_viewer_preview_revision = workspace._viewer_preview_revision - 1
    workspace._clear_committed_surface_texture_previews_if_displayed()
    assert workspace.viewer.get_pending_surface_texture_preview_ids()
    workspace._canvas_viewer_preview_revision = workspace._viewer_preview_revision
    workspace._clear_committed_surface_texture_previews_if_displayed()
    assert workspace.viewer.get_pending_surface_texture_preview_ids() == ()


def test_later_assignments_restart_delay_and_keep_other_pending_faces(workspace):
    _publish(workspace, _assignment())
    timer = workspace._surface_texture_mesh_update_timer
    with patch.object(timer, "start", wraps=timer.start) as start:
        _publish(workspace, _assignment(), _assignment((FLOOR,), "tile"))
        start.assert_called_once()
    assert workspace._pending_surface_texture_ids == {WALL, FLOOR}


def test_reassigning_while_rebuild_pending_does_not_clear_new_preview(workspace):
    _publish(workspace, _assignment())
    workspace._commit_pending_surface_texture_mesh_update()
    old_revision = workspace._viewer_preview_revision
    _publish(workspace, _assignment(assignment_id="new-paint"))
    workspace._canvas_viewer_preview_revision = old_revision
    workspace._clear_committed_surface_texture_previews_if_displayed()
    assert workspace.viewer.get_pending_surface_texture_preview_ids() == (WALL,)
    assert workspace._pending_surface_texture_ids == {WALL}


def test_timer_uses_mesh_edit_delay_and_confirm_bypasses_it(workspace):
    _publish(workspace, _assignment())
    workspace._set_mesh_edit_update_delay_seconds(12.0)
    assert workspace._surface_texture_mesh_update_timer.interval() == 12000
    assert workspace._surface_texture_mesh_update_timer.isActive()
    workspace.confirm_mesh_updates_button.click()
    assert not workspace._pending_surface_texture_ids
    assert not workspace._surface_texture_mesh_update_timer.isActive()
    assert workspace._mesh_edit_update_delay_seconds == 12.0


def test_timer_commits_without_an_explicit_confirm(workspace):
    workspace._set_mesh_edit_update_delay_seconds(0.01)
    timer = workspace._surface_texture_mesh_update_timer
    timeout = QSignalSpy(timer.timeout)
    _publish(workspace, _assignment())
    assert timer.interval() == 10
    assert timer.isActive()
    # A nested Qt event loop can finish an unrelated queued callback before
    # delivering this due timer; observe delivery, not a wall-clock poll budget.
    assert timeout.count() or timeout.wait(5000), (
        timer.remainingTime(),
        workspace._is_assigning_surface_texture_from_atlas,
    )
    assert timeout.count() == 1
    assert not workspace._pending_surface_texture_ids, (
        timer.remainingTime(),
        workspace._is_assigning_surface_texture_from_atlas,
    )
    assert WALL in workspace._committing_surface_texture_revisions


@pytest.mark.parametrize("method", (
    "_queue_viewer_preview_refresh", "_run_scheduled_viewer_preview_refresh",
    "_refresh_viewer_preview",
))
def test_other_scene_refreshes_cannot_apply_assignment_early(workspace, method):
    _publish(workspace, _assignment())
    with (
        patch.object(workspace, "_active_viewer_preview_needs_refresh", return_value=True),
        patch.object(workspace, "_build_viewer_preview_model") as build,
        patch.object(workspace, "_get_viewer_surface_snapshot") as snapshot,
    ):
        getattr(BlueprintWorkspace, method)(workspace)
    build.assert_not_called()
    snapshot.assert_not_called()
    assert workspace._canvas_interaction_blocks_ambient_occlusion_preview()


def test_window_undo_preview_cannot_install_texture_before_its_delay(workspace):
    _publish(workspace, _assignment())
    with patch.object(workspace.viewer, "set_model") as install:
        assert not workspace._apply_canvas_window_preview(
            object(), dependency_signature=(),
        )
    install.assert_not_called()
    assert workspace._pending_surface_texture_ids == {WALL}
    assert workspace._surface_texture_mesh_update_timer.isActive()


# ### Library-only results and lifecycle ###
def test_library_only_generation_does_not_dirty_scene_or_stage_faces(workspace):
    assignment = _assignment(())
    revision = workspace._viewer_preview_revision
    with patch.object(workspace.texture_atlas_workspace, "mark_sources_new") as mark:
        workspace._handle_surface_texture_generation_completed(assignment)
        _publish(workspace, assignment)
    mark.assert_called_once()
    assert workspace._viewer_preview_revision == revision
    assert not workspace._surface_texture_mesh_update_timer.isActive()
    assert not workspace._pending_surface_texture_ids
    workspace._queue_viewer_preview_refresh.assert_not_called()


def test_renaming_does_not_count_as_a_new_assignment(workspace):
    assignment = _assignment()
    _publish(workspace, assignment)
    workspace._commit_pending_surface_texture_mesh_update()
    _publish(workspace, replace(assignment, display_name="Different name"))
    assert not workspace._pending_surface_texture_ids
    assert not workspace._surface_texture_mesh_update_timer.isActive()


def test_unassigning_delays_the_visible_material_removal(workspace):
    assignment = _assignment()
    _publish(workspace, assignment)
    workspace._commit_pending_surface_texture_mesh_update()
    _publish(workspace, replace(assignment, surface_ids=()))
    assert workspace._pending_surface_texture_ids == {WALL}
    assert workspace.viewer.get_pending_surface_texture_preview_ids() == (WALL,)


def test_project_restore_is_not_treated_as_new_texture_assignment(workspace):
    workspace._is_applying_project_state = True
    try:
        _publish(workspace, _assignment())
    finally:
        workspace._is_applying_project_state = False
    assert not workspace._pending_surface_texture_ids
    with (
        patch.object(workspace.surface_texture_generation, "get_assignments",
                     return_value=(_assignment(),)),
        patch.object(workspace, "_sync_atlas_object_texture_sources"),
        patch.object(workspace, "_schedule_viewer_preview_refresh"),
    ):
        workspace._finalize_project_state_apply()
    assert WALL in workspace._surface_texture_binding_snapshot
    _publish(workspace, _assignment())
    assert not workspace._pending_surface_texture_ids


def test_shutdown_cancels_only_transient_assignment_work(workspace):
    _publish(workspace, _assignment())
    workspace.shutdown()
    assert not workspace._surface_texture_mesh_update_timer.isActive()
    assert not workspace._pending_surface_texture_ids
    assert not workspace._committing_surface_texture_revisions
    assert workspace.viewer.get_pending_surface_texture_preview_ids() == ()


def test_confirm_before_queued_timer_start_cannot_rearm_a_completed_assignment(workspace):
    with patch.object(workspace.surface_texture_generation, "get_assignments",
                      return_value=(_assignment(),)):
        workspace._handle_surface_texture_content_changed()
    assert workspace._pending_surface_texture_ids == {WALL}
    assert not workspace._surface_texture_mesh_update_timer.isActive()

    workspace.confirm_mesh_updates_button.click()
    _application.processEvents()

    assert not workspace._pending_surface_texture_ids
    assert not workspace._surface_texture_mesh_update_timer.isActive()
    assert WALL in workspace._committing_surface_texture_revisions
