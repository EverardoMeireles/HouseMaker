"""New opening controls must not wait for the delayed wall-mesh rebuild."""

# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ### Imports ###
from dataclasses import replace
from unittest.mock import patch

import pytest
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.canvas_openings import (
    CanvasOpeningEdit,
    build_canvas_opening_preview_targets,
    build_canvas_opening_targets,
)
from housemaker.main import BlueprintWorkspace
from housemaker.models import DoorwayData, LevelData, VertexData
from housemaker.surface_geometry import (
    add_wall_window,
    build_fixed_surfaces,
    build_wall_window_placement,
)
from housemaker.viewer import (
    CANVAS_OPENING_GIZMO_SIDE,
    CANVAS_OPENING_PREVIEW_COLOR,
    CANVAS_OPENING_SIDE_TOP,
    _CanvasOpeningGizmoHandle,
)
from housemaker.window_frame_bridge import (
    add_window_frame_bridge,
    normalize_wall_window_placement,
)

# ### Fixtures ###
_application = QApplication.instance() or QApplication([])
_application.setQuitOnLastWindowClosed(False)


def _level():
    graph = VertexData()
    for point in ((0.0, 0.0), (200.0, 0.0), (0.0, 10.0), (200.0, 10.0)):
        graph.add_vertex(*point)
    for edge in ((1, 2), (3, 4), (1, 3), (2, 4)):
        graph.add_edge(*edge)
    return LevelData(index=2, name="Ground", vertex_data=graph)


@pytest.fixture
def workspace(tmp_path):
    widget = BlueprintWorkspace(
        application_settings=ApplicationSettingsStore(tmp_path / "settings.json"),
    )
    widget.levels = [_level()]
    widget.current_level_index = 0
    widget._reset_viewer_doorway_snapshots()
    widget._sync_canvas_to_current_level()
    widget._set_canvas_viewer_targets(tuple(build_fixed_surfaces(widget.levels)))
    widget._sync_viewer_scene_levels()
    widget._set_mesh_edit_update_delay_seconds(60.0)
    yield widget
    widget.shutdown()
    widget.close()
    _application.processEvents()


def _placement(surfaces):
    wall = next(
        surface for surface in surfaces if surface.surface_id == "level:2/wall:1:2"
    )
    return build_wall_window_placement(wall, (1.2, 0.0, 0.8), (2.2, 0.0, 2.0))


# ### Pure target resolution tests ###
def test_pending_window_resolves_new_host_without_rebuilding_surfaces():
    level = _level()
    old_surfaces = tuple(build_fixed_surfaces((level,)))
    placement = normalize_wall_window_placement((level,), _placement(old_surfaces))
    add_window_frame_bridge((level,), placement)
    window = add_wall_window((level,), placement)
    assert window.wall_surface_id not in {
        surface.surface_id for surface in old_surfaces
    }
    assert not build_canvas_opening_targets((level,), old_surfaces)

    with patch(
        "housemaker.surface_geometry.build_fixed_surfaces", side_effect=AssertionError
    ):
        targets = build_canvas_opening_preview_targets((level,), old_surfaces)

    assert len(targets) == 1
    assert targets[0].reference.stable_id == window.window_id
    assert targets[0].wall_surface_id == window.wall_surface_id
    assert targets[0].opposite_wall_plane_world is not None
    assert 0.0 < targets[0].bounds.start_ratio < targets[0].bounds.end_ratio < 1.0


# ### Immediate controls and delayed commit tests ###
def test_window_can_be_selected_and_edited_before_first_mesh_commit(workspace):
    level = workspace.current_level
    surfaces = tuple(workspace._canvas_surface_targets_by_id.values())
    committed = workspace._viewer_windows_by_level_index[level.index]
    with patch.object(workspace, "_schedule_viewer_preview_refresh") as refresh:
        workspace._handle_canvas_window_placement_requested(_placement(surfaces))
        target = workspace.viewer._get_selected_canvas_opening_target()
        assert target is not None
        assert target.key in workspace.viewer._pending_canvas_opening_keys
        assert workspace.viewer._canvas_opening_gizmo_items
        assert (
            tuple(workspace.viewer._canvas_opening_gizmo_items[0].color)
            == CANVAS_OPENING_PREVIEW_COLOR
        )
        assert workspace.canvas.selected_window_id == level.windows[0].window_id
        assert workspace._viewer_windows_by_level_index[level.index] == committed
        refresh.assert_not_called()

        # Exercise a real gizmo press while the host ID is still absent from
        # the committed surface cache, then cancel without losing the frame.
        center = target.local_to_world(
            (target.bounds.start_ratio + target.bounds.end_ratio) * 0.5,
            target.bounds.top_ratio,
        )
        with patch.object(
            workspace.viewer.view,
            "build_camera_ray",
            return_value=((center[0], center[1] - 2.0, center[2]), (0.0, 1.0, 0.0)),
        ):
            assert workspace.viewer._begin_canvas_opening_gizmo_drag(
                _CanvasOpeningGizmoHandle(
                    CANVAS_OPENING_GIZMO_SIDE, CANVAS_OPENING_SIDE_TOP
                ),
                QPointF(10.0, 10.0),
            )
            assert not workspace._doorway_mesh_update_timer.isActive()
            workspace.viewer._cancel_canvas_opening_edit_drag()
        assert workspace.viewer._get_selected_canvas_opening_target() is not None

        original = CanvasOpeningEdit(
            target.reference, target.wall_surface_id, target.bounds
        )
        edited = replace(original, bounds=replace(target.bounds, top_ratio=0.85))
        corners = level.windows[0].corner_vertex_ids
        workspace._handle_canvas_opening_edit_started(original)
        workspace._handle_canvas_opening_edit_preview_changed(edited)
        assert not workspace._doorway_mesh_update_timer.isActive()
        assert level.windows[0].top_ratio == pytest.approx(0.85)
        assert level.windows[0].corner_vertex_ids == corners
        workspace._handle_canvas_opening_edit_finished(edited, True)
        assert workspace._doorway_mesh_update_timer.isActive()
        assert workspace._viewer_windows_by_level_index[level.index] == committed

        workspace._commit_pending_doorway_mesh_update()
        assert workspace._viewer_windows_by_level_index[level.index][
            0
        ].top_ratio == pytest.approx(0.85)
        assert refresh.called

    workspace._set_canvas_viewer_targets(tuple(build_fixed_surfaces(workspace.levels)))
    assert target.key not in workspace.viewer._pending_canvas_opening_keys
    assert workspace.viewer._get_selected_canvas_opening_target() is not None


def test_new_doorway_has_live_controls_without_refresh(workspace):
    level = workspace.current_level
    level.doorways.append(
        DoorwayData(
            center_x=100.0,
            center_y=5.0,
            width_meters=1.0,
            height_meters=2.0,
            rotation_degrees=90.0,
        )
    )
    workspace.canvas.doorways = level.doorways
    workspace.canvas._set_selected_doorway_index(0)
    with patch.object(workspace, "_schedule_viewer_preview_refresh") as refresh:
        workspace._handle_doorways_changed()
    target = workspace.viewer._get_selected_canvas_opening_target()
    assert target is not None
    assert target.reference.kind == "doorway"
    assert target.key in workspace.viewer._pending_canvas_opening_keys
    assert workspace.viewer._canvas_opening_gizmo_items
    assert not workspace._viewer_doorways_by_level_index[level.index]
    assert workspace._doorway_mesh_update_timer.isActive()
    refresh.assert_not_called()


def test_missing_host_is_visible_only_as_explicit_pending_opening(workspace):
    workspace._handle_canvas_window_placement_requested(
        _placement(tuple(workspace._canvas_surface_targets_by_id.values())),
    )
    target = workspace.viewer._get_selected_canvas_opening_target()
    assert target is not None
    assert workspace.viewer._canvas_opening_target_is_visible(target)
    workspace.viewer.set_canvas_surface_focus_type("floor")
    assert not workspace.viewer._canvas_opening_target_is_visible(target)
    workspace.viewer.set_canvas_surface_focus_type(None)
    assert workspace.viewer._canvas_opening_target_is_visible(target)
    workspace.viewer.set_canvas_opening_targets((target,))
    assert not workspace.viewer._canvas_opening_target_is_visible(target)
    assert workspace.viewer._get_selected_canvas_opening_target() is None


def test_click_on_reveal_does_not_select_opening_hole_plane(workspace):
    workspace._handle_canvas_window_placement_requested(
        _placement(tuple(workspace._canvas_surface_targets_by_id.values())),
    )
    target = workspace.viewer._get_selected_canvas_opening_target()
    assert target is not None
    surfaces = tuple(build_fixed_surfaces(workspace.levels))
    reveal = next(
        surface
        for surface in surfaces
        if getattr(surface, "opening_part", None) == "left"
    )
    workspace._set_canvas_viewer_targets(surfaces)
    viewer = workspace.viewer
    assert reveal.surface_id not in viewer._window_wall_targets
    viewer.select_canvas_opening(None)
    with (
        patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=((1.0, -2.0, 1.0), (0.0, 1.0, 0.0)),
        ),
        patch(
            "housemaker.viewer._get_nearest_canvas_opening_ray_hit",
            return_value=(target, (1.0, 0.0, 1.0), 2.0),
        ),
        patch(
            "housemaker.viewer._get_nearest_fixed_surface_ray_hit",
            return_value=(reveal, (1.0, 0.1, 1.0), 2.1),
        ),
        patch(
            "housemaker.viewer._get_nearest_preview_placed_object_ray_hit",
            return_value=None,
        ),
    ):
        viewer._handle_window_wall_pick_requested(QPointF(100.0, 100.0))
    assert viewer.get_selected_canvas_opening_reference() is None
    assert viewer.get_selected_canvas_surface_ids() == (reveal.surface_id,)
    workspace._desired_canvas_surface_ids = (reveal.surface_id,)
    workspace._restore_desired_canvas_scene_selection()
    assert viewer.get_selected_canvas_surface_ids() == (reveal.surface_id,)
    assert not workspace._get_canvas_wall_surface_ids((reveal.surface_id,))
