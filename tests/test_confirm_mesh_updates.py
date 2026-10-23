# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from contextlib import ExitStack
from unittest.mock import Mock, patch

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.main import BlueprintWorkspace
from housemaker.models import DoorwayData, WindowData

# ### Application and workspace fixtures ###
_application = QApplication.instance() or QApplication([])
_application.setQuitOnLastWindowClosed(False)


@pytest.fixture
def workspace(tmp_path):
    widget = BlueprintWorkspace(
        application_settings=ApplicationSettingsStore(tmp_path / "settings.json")
    )
    widget._set_mesh_edit_update_delay_seconds(60.0)
    try:
        yield widget
    finally:
        widget.shutdown()
        widget.close()
        _application.processEvents()


def _mesh_timers(workspace: BlueprintWorkspace):
    return tuple(
        getattr(workspace, name)
        for name in (
            "_doorway_mesh_update_timer",
            "_canvas_surface_mesh_update_timer",
            "_canvas_face_orientation_update_timer",
            "_sloped_ceiling_mesh_update_timer",
            "_architectural_trim_mesh_update_timer",
            "_level_transform_mesh_update_timer",
            "_stair_point_mesh_update_timer",
            "_wall_vertex_update_timer",
            "_landing_mesh_update_timer",
            "_object_placement_mesh_update_timer",
        )
    )


def _click_confirm(workspace: BlueprintWorkspace) -> None:
    workspace._refresh_confirm_mesh_updates_button()
    assert workspace.confirm_mesh_updates_button.isEnabled()
    workspace.confirm_mesh_updates_button.click()


# ### Pending-edit indication ###
def test_idle_button_is_disabled_without_blinking(workspace) -> None:
    button = workspace.confirm_mesh_updates_button
    assert button.text() == "Confirm mesh updates"
    assert button.focusPolicy() == Qt.FocusPolicy.NoFocus
    assert not workspace._has_pending_mesh_updates()
    for _ in range(3):
        workspace._refresh_confirm_mesh_updates_button()
        assert not button.isEnabled()
        assert button.styleSheet() == ""


@pytest.mark.parametrize(
    ("field", "pending_value"),
    (
        ("_pending_level_transform", object()),
        ("_pending_canvas_face_orientation_undo_state", object()),
        ("_pending_canvas_face_orientation_surface_ids", {"wall"}),
        ("_pending_canvas_surface_mesh_update", True),
        ("_pending_floor_thickness_level_index", 2),
        ("_pending_sloped_ceiling_undo_state", object()),
        ("_pending_architectural_trim_undo_state", object()),
        ("_pending_wall_vertex_mesh_update", True),
        ("_pending_open_space_corner_mesh_update", True),
        ("_pending_stair_point_mesh_update", True),
        ("_pending_landing_mesh_update", True),
        ("_pending_object_placement_ids", {"object-a"}),
        ("_pending_doorway_mesh_level_index", 2),
        ("_pending_window_mesh_level_index", 2),
        ("_staged_canvas_opening_mesh_update", True),
        ("_pending_doorway_bridge_topology_change", True),
    ),
)
def test_each_mesh_edit_queue_blinks_yellow(workspace, field, pending_value) -> None:
    previous = getattr(workspace, field)
    try:
        setattr(workspace, field, pending_value)
        assert workspace._has_pending_mesh_updates()
        workspace._refresh_confirm_mesh_updates_button()
        button = workspace.confirm_mesh_updates_button
        assert button.isEnabled()
        assert button.property("mesh_updates_pending") is True
        assert "#f2ce43" in button.styleSheet()
        workspace._refresh_confirm_mesh_updates_button()
        assert button.styleSheet() == ""
        assert button.isEnabled()
    finally:
        setattr(workspace, field, previous)
        workspace._refresh_confirm_mesh_updates_button()
    assert not button.isEnabled()
    assert button.styleSheet() == ""


def test_pending_indicator_updates_without_a_manual_refresh(workspace) -> None:
    workspace._pending_wall_vertex_mesh_update = True
    QTest.qWait(workspace._confirm_mesh_updates_timer.interval() + 50)
    assert workspace.confirm_mesh_updates_button.isEnabled()
    assert (
        workspace.confirm_mesh_updates_button.property("mesh_updates_pending") is True
    )
    workspace._pending_wall_vertex_mesh_update = False
    QTest.qWait(workspace._confirm_mesh_updates_timer.interval() + 50)
    assert not workspace.confirm_mesh_updates_button.isEnabled()
    assert workspace.confirm_mesh_updates_button.styleSheet() == ""


# ### Immediate edit completion and preserved settings ###
def test_confirm_flushes_every_edit_path_without_changing_delay(workspace) -> None:
    methods = (
        "_finish_level_transform_drag",
        "_commit_pending_level_transform_update",
        "_commit_pending_canvas_face_orientation_update",
        "_commit_pending_canvas_surface_mesh_update",
        "_commit_pending_sloped_ceiling_mesh_update",
        "_commit_pending_architectural_trim_mesh_update",
        "_commit_pending_wall_vertex_update",
        "_commit_pending_stair_point_mesh_update",
        "_commit_landing_mesh_update",
        "_commit_pending_object_placement_mesh_update",
        "_commit_pending_doorway_mesh_update",
        "_queue_viewer_preview_refresh",
    )
    calls = Mock()
    workspace._pending_wall_vertex_mesh_update = True
    intervals = tuple(timer.interval() for timer in _mesh_timers(workspace))
    with ExitStack() as stack:
        for name in methods:
            patched = stack.enter_context(patch.object(workspace, name))
            calls.attach_mock(patched, name)
        _click_confirm(workspace)
    assert [call[0] for call in calls.mock_calls] == list(methods)
    assert workspace._mesh_edit_update_delay_seconds == 60.0
    assert tuple(timer.interval() for timer in _mesh_timers(workspace)) == intervals
    workspace._pending_wall_vertex_mesh_update = False


def test_confirm_without_pending_edits_is_inert(workspace) -> None:
    with patch.object(workspace, "_queue_viewer_preview_refresh") as refresh:
        workspace._handle_confirm_mesh_updates_clicked()
    refresh.assert_not_called()


def test_confirm_commits_real_level_transform_and_keeps_undo(workspace) -> None:
    level = workspace.current_level
    old_offset = level.offset_x_meters
    with patch.object(workspace, "_schedule_viewer_preview_refresh"):
        workspace._handle_level_x_offset_changed(old_offset + 1.25)
        assert level.offset_x_meters == old_offset
        assert workspace._level_transform_mesh_update_timer.isActive()
        _click_confirm(workspace)
        assert level.offset_x_meters == old_offset + 1.25
        assert not workspace._level_transform_mesh_update_timer.isActive()
        assert workspace._pending_level_transform is None
        assert not workspace.confirm_mesh_updates_button.isEnabled()
        assert workspace._canvas_undo_stack
        workspace._handle_canvas_undo_requested()
    assert level.offset_x_meters == old_offset
    assert workspace._mesh_edit_update_delay_seconds == 60.0


def test_confirm_publishes_real_doorway_and_window_snapshots(workspace) -> None:
    level = workspace.current_level
    level.doorways.append(DoorwayData(10.0, 20.0, 1.0, 2.2))
    level.windows.append(
        WindowData("window-a", f"level:{level.index}/wall:1:2", 0.2, 0.5, 0.3, 0.75)
    )
    workspace._pending_doorway_mesh_level_index = level.index
    workspace._pending_window_mesh_level_index = level.index
    workspace._doorway_mesh_update_timer.start()
    with patch.object(workspace, "_schedule_viewer_preview_refresh") as refresh:
        _click_confirm(workspace)
    assert workspace._viewer_doorways_by_level_index[level.index] == tuple(
        level.doorways
    )
    assert workspace._viewer_windows_by_level_index[level.index] == tuple(level.windows)
    assert workspace._pending_doorway_mesh_level_index is None
    assert workspace._pending_window_mesh_level_index is None
    assert not workspace._doorway_mesh_update_timer.isActive()
    assert not workspace.confirm_mesh_updates_button.isEnabled()
    refresh.assert_called_once_with(preserve_camera=True)


def test_confirm_publishes_floor_thickness_and_wall_edits(workspace) -> None:
    level = workspace.current_level
    level.floor_thickness_meters += 0.1
    workspace._pending_floor_thickness_level_index = level.index
    workspace._pending_canvas_surface_mesh_update = True
    workspace._pending_wall_vertex_mesh_update = True
    workspace._pending_open_space_corner_mesh_update = True
    workspace._canvas_surface_mesh_update_timer.start()
    workspace._wall_vertex_update_timer.start()
    with (
        patch.object(workspace, "_schedule_viewer_preview_refresh"),
        patch.object(
            workspace, "_reconcile_surface_assignments_with_scene", return_value=False
        ),
    ):
        _click_confirm(workspace)
    assert (
        workspace._viewer_floor_thickness_by_level_index[level.index]
        == level.floor_thickness_meters
    )
    assert not workspace._has_pending_mesh_updates()
    assert not workspace._canvas_surface_mesh_update_timer.isActive()
    assert not workspace._wall_vertex_update_timer.isActive()


# ### Detached-window ownership and lifecycle ###
def test_button_moves_with_the_dedicated_3d_window(workspace) -> None:
    screen = _application.primaryScreen()
    assert screen is not None
    with patch(
        "housemaker.main.resolve_fullscreen_3d_viewer_screen", return_value=screen
    ):
        workspace._apply_scene_3d_display_screen("screen:test")
    host = workspace._external_scene_3d_host
    button = workspace.confirm_mesh_updates_button
    assert host.is_active
    assert host.window.isAncestorOf(button)
    assert button.isVisible()
    assert workspace.scene_3d_workspace.isAncestorOf(button)
    workspace._apply_scene_3d_display_screen(None)
    assert not host.is_active
    assert workspace.scene_3d_workspace.isAncestorOf(button)


def test_shutdown_stops_pending_edit_indicator(workspace) -> None:
    assert workspace._confirm_mesh_updates_timer.isActive()
    workspace.shutdown()
    assert not workspace._confirm_mesh_updates_timer.isActive()
