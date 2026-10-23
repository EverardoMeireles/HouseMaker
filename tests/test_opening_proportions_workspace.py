# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import math
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from PySide6.QtWidgets import QApplication, QWidget

from housemaker.canvas_openings import (
    CanvasOpeningBounds,
    CanvasOpeningReference,
    CanvasOpeningTarget,
)
from housemaker.global_scale import GlobalScale
from housemaker.models import (
    DOORWAY_SHAPE_ARCH,
    DOORWAY_SHAPE_RECTANGULAR,
    DoorwayData,
    LevelData,
    WindowData,
)
from housemaker.opening_proportions_geometry import OpeningProportionsResult
from housemaker.opening_proportions_workspace import (
    OpeningProportionsUndoState,
    OpeningProportionsWorkspaceMixin,
    build_opening_proportions_update,
)

# ### Qt application lifetime ###
_application = QApplication.instance() or QApplication([])
_application.setQuitOnLastWindowClosed(False)
_retained_dialog_hosts = []


# ### Opening fixtures ###
def _window(*, bottom=0.2, top=0.35):
    return WindowData(
        window_id="measured-window",
        wall_surface_id="level:2/wall:1:2",
        start_ratio=0.25,
        end_ratio=0.55,
        bottom_ratio=bottom,
        top_ratio=top,
        corner_vertex_ids=(1, 2, 3, 4),
    )


def _doorway(*, height=0.8, width=1.2):
    return DoorwayData(
        center_x=100.0,
        center_y=120.0,
        width_meters=width,
        height_meters=height,
        depth_meters=0.25,
        rotation_degrees=25.0,
        bottom_height_meters=0.1,
        doorway_id="measured-doorway",
        corner_vertex_ids=(1, 2, 3, 4),
    )


def _target(opening, *, width=4.0, height=4.0, tangent=(1.0, 0.0, 0.0)):
    is_window = isinstance(opening, WindowData)
    return CanvasOpeningTarget(
        reference=CanvasOpeningReference(
            kind="window" if is_window else "doorway",
            level_index=2,
            item_index=0,
            stable_id=opening.window_id if is_window else opening.doorway_id,
        ),
        wall_surface_id="level:2/wall:1:2",
        plane_start_world=(0.0, 0.0, 0.0),
        wall_tangent_world=tangent,
        wall_normal_world=(-tangent[1], tangent[0], 0.0),
        wall_width_meters=width,
        wall_height_meters=height,
        minimum_width_meters=0.1,
        minimum_height_meters=0.1,
        bounds=CanvasOpeningBounds(
            start_ratio=opening.start_ratio if is_window else 0.25,
            end_ratio=opening.end_ratio if is_window else 0.55,
            bottom_ratio=(
                opening.bottom_ratio
                if is_window
                else opening.bottom_height_meters / height
            ),
            top_ratio=(
                opening.top_ratio
                if is_window
                else (opening.bottom_height_meters + opening.height_meters) / height
            ),
        ),
    )


def _measurement(height_to_width=1.5, arch_height_to_width=0.0):
    return OpeningProportionsResult(
        height_to_width=height_to_width,
        arch_height_to_width=arch_height_to_width,
        method="front-view-ratio",
    )


# ### Height updates and authored-width preservation ###
def test_window_height_comes_from_current_width_not_existing_wrong_height():
    opening = _window()
    updated = build_opening_proportions_update(
        opening, _target(opening), _measurement(), GlobalScale()
    )
    assert updated.top_ratio == pytest.approx(0.2 + 1.2 * 1.5 / 4.0)
    assert updated.bottom_ratio == opening.bottom_ratio
    assert updated.start_ratio == opening.start_ratio
    assert updated.end_ratio == opening.end_ratio
    assert updated.corner_vertex_ids == opening.corner_vertex_ids
    assert updated.window_id == opening.window_id
    assert updated.wall_surface_id == opening.wall_surface_id
    assert opening.top_ratio == 0.35


@pytest.mark.parametrize("wrong_top", [0.3, 0.5, 0.9])
def test_manually_incorrect_window_heights_are_restored(wrong_top):
    opening = _window(top=wrong_top)
    updated = build_opening_proportions_update(
        opening, _target(opening), _measurement(), GlobalScale()
    )
    assert updated.top_ratio == pytest.approx(0.65)


def test_window_repeat_is_idempotent_and_does_not_mutate_global_multipliers():
    opening = _window()
    scale = GlobalScale(1.8, 1.9, 2.0, linked=False)
    scale_before = asdict(scale)
    measurement = _measurement()
    first = build_opening_proportions_update(
        opening, _target(opening), measurement, scale
    )
    second = build_opening_proportions_update(first, _target(first), measurement, scale)
    assert second == first
    assert asdict(scale) == scale_before


@pytest.mark.parametrize(
    ("tangent", "expected_horizontal_stretch"),
    [
        ((1.0, 0.0, 0.0), 2.0),
        ((0.0, 1.0, 0.0), 3.0),
        ((0.6, 0.8, 0.0), math.hypot(0.6 * 2.0, 0.8 * 3.0)),
    ],
)
def test_anisotropic_global_xy_uses_actual_wall_direction(
    tangent, expected_horizontal_stretch
):
    opening = _window()
    target = _target(opening, height=8.0, tangent=tangent)
    updated = build_opening_proportions_update(
        opening, target, _measurement(), GlobalScale(2.0, 3.0, 1.7, linked=False)
    )
    expected_height = 1.2 * expected_horizontal_stretch * 1.5 / 1.7
    assert updated.top_ratio == pytest.approx(0.2 + expected_height / 8.0)


def test_sloped_window_uses_scaled_local_vertical_axis():
    opening = _window()
    target = replace(
        _target(opening, height=5.0),
        vertical_axis_world=(0.0, 0.6, 0.8),
        wall_normal_world=(0.0, -0.8, 0.6),
    )
    updated = build_opening_proportions_update(
        opening, target, _measurement(), GlobalScale(2.0, 3.0, 4.0, linked=False)
    )
    local_vertical_stretch = math.hypot(0.6 * 3.0, 0.8 * 4.0)
    expected_height = 1.2 * 2.0 * 1.5 / local_vertical_stretch
    assert updated.top_ratio == pytest.approx(0.2 + expected_height / 5.0)


def test_rectangular_doorway_preserves_position_width_depth_and_topology():
    opening = _doorway()
    before = asdict(opening)
    updated = build_opening_proportions_update(
        opening, _target(opening), _measurement(), GlobalScale()
    )
    assert updated.height_meters == pytest.approx(1.8)
    assert updated.shape == DOORWAY_SHAPE_RECTANGULAR
    for field_name in (
        "center_x",
        "center_y",
        "width_meters",
        "depth_meters",
        "rotation_degrees",
        "bottom_height_meters",
        "doorway_id",
        "corner_vertex_ids",
    ):
        assert getattr(updated, field_name) == getattr(opening, field_name)
    assert asdict(opening) == before


@pytest.mark.parametrize("wrong_height", [0.2, 1.0, 3.5])
def test_manually_incorrect_doorway_heights_are_restored(wrong_height):
    opening = _doorway(height=wrong_height)
    updated = build_opening_proportions_update(
        opening, _target(opening), _measurement(), GlobalScale()
    )
    assert updated.height_meters == pytest.approx(1.8)


# ### Arched doorway model conversion ###
def test_arch_rise_uses_physical_width_and_converts_to_authored_model_amount():
    opening = _doorway()
    updated = build_opening_proportions_update(
        opening,
        _target(opening),
        _measurement(height_to_width=1.8, arch_height_to_width=0.2),
        GlobalScale(2.0, 3.0, 2.0, linked=False),
    )
    assert updated.shape == DOORWAY_SHAPE_ARCH
    assert updated.height_meters == pytest.approx(2.16)
    assert updated.arch_amount == pytest.approx(0.24 / 0.6)


def test_level_scaled_doorway_width_is_not_scaled_twice_for_arch_amount():
    opening = _doorway(width=1.2)
    # Level scale expands plan XY only: world width is twice the intrinsic
    # profile width, while the arch generator still uses intrinsic half-width.
    target = _target(opening, width=8.0, height=6.0)
    updated = build_opening_proportions_update(
        opening,
        target,
        _measurement(height_to_width=1.8, arch_height_to_width=0.1),
        GlobalScale(),
    )
    assert updated.height_meters == pytest.approx(4.32)
    assert updated.arch_amount == pytest.approx(0.24 / 0.6)
    assert updated.width_meters == 1.2


def test_flat_fifth_point_removes_existing_doorway_arch():
    opening = replace(_doorway(), shape=DOORWAY_SHAPE_ARCH, arch_amount=0.75)
    updated = build_opening_proportions_update(
        opening, _target(opening), _measurement(), GlobalScale()
    )
    assert updated.shape == DOORWAY_SHAPE_RECTANGULAR


def test_arch_measurement_that_exceeds_profile_capacity_is_rejected():
    opening = _doorway()
    with pytest.raises(ValueError, match="arch|Arch"):
        build_opening_proportions_update(
            opening,
            _target(opening, width=8.0, height=10.0),
            _measurement(height_to_width=1.8, arch_height_to_width=0.4),
            GlobalScale(),
        )


# ### Fit constraints and invalid measurements ###
@pytest.mark.parametrize("kind", ["window", "doorway"])
def test_opening_height_cannot_extend_through_ceiling(kind):
    opening = _window(bottom=0.8, top=0.9) if kind == "window" else _doorway()
    target = _target(opening, height=1.0 if kind == "doorway" else 4.0)
    with pytest.raises(ValueError):
        build_opening_proportions_update(opening, target, _measurement(), GlobalScale())


@pytest.mark.parametrize("kind", ["window", "doorway"])
def test_opening_height_cannot_fall_below_minimum(kind):
    opening = _window() if kind == "window" else _doorway()
    with pytest.raises(ValueError):
        build_opening_proportions_update(
            opening, _target(opening), _measurement(0.01), GlobalScale()
        )


@pytest.mark.parametrize("ratio", [0.0, -1.0, math.nan, math.inf])
def test_non_positive_or_non_finite_measurement_is_rejected(ratio):
    opening = _window()
    with pytest.raises(ValueError):
        build_opening_proportions_update(
            opening, _target(opening), _measurement(ratio), GlobalScale()
        )


def test_windows_reject_arch_measurement():
    opening = _window()
    with pytest.raises(ValueError):
        build_opening_proportions_update(
            opening, _target(opening), _measurement(1.5, 0.1), GlobalScale()
        )


# ### Workspace harness without OpenGL or generation providers ###
class _Workspace(OpeningProportionsWorkspaceMixin):
    def __init__(self, opening):
        self._init_opening_proportions()
        self.global_scale = GlobalScale()
        self.levels = [LevelData(index=2, name="Ground", height_meters=4.0)]
        self.current_level = self.levels[0]
        is_window = isinstance(opening, WindowData)
        if is_window:
            self.current_level.windows = [opening]
        else:
            self.current_level.doorways = [opening]
        self._opening_proportions_reference = _target(opening).reference
        self._opening_proportions_target = _target(opening)
        self._opening_proportions_dialog = Mock()
        self.viewer = Mock()
        self.canvas = Mock()
        self._canvas_undo_stack = []
        self._viewer_doorways_by_level_index = {}
        self._canvas_opening_targets_by_key = {}
        self._sync_viewer_window_snapshot = Mock()
        self._refresh_canvas_windows_for_level = Mock()
        self._handle_canvas_doorway_selection_changed = Mock()
        self._schedule_viewer_preview_refresh = Mock()
        for name in (
            "_commit_pending_level_transform_update",
            "_commit_pending_canvas_surface_mesh_update",
            "_commit_pending_sloped_ceiling_mesh_update",
            "_commit_pending_architectural_trim_mesh_update",
            "_commit_pending_wall_vertex_update",
            "_commit_pending_doorway_mesh_update",
        ):
            setattr(self, name, Mock())

    def _build_opening_target_lookup_for_levels(self, levels, **_kwargs):
        targets = {}
        for level in levels:
            for opening in (*level.windows, *level.doorways):
                target = _target(opening)
                targets[target.key] = target
        return targets

    def _record_canvas_undo_state(self, state):
        self._canvas_undo_stack.append(state)

    def _copy_doorways(self, doorways):
        return [replace(doorway) for doorway in doorways]


def _payload(*, doorway=False, arch=False):
    return {
        "frame_size": (1000, 1000),
        "outline": ((0.3, 0.3), (0.5, 0.3), (0.5, 0.6), (0.3, 0.6)),
        "arch_apex": (0.4, 0.26) if arch else ((0.4, 0.3) if doorway else None),
        "is_doorway": doorway,
    }


# ### Explicit application, stable identities and one-operation undo ###
@pytest.mark.parametrize("kind", ["window", "doorway"])
def test_apply_records_single_undo_and_repeated_apply_is_noop(kind):
    opening = _window() if kind == "window" else _doorway()
    workspace = _Workspace(opening)
    workspace._apply_opening_proportions(_payload(doorway=kind == "doorway"))
    assert len(workspace._canvas_undo_stack) == 1
    state = workspace._canvas_undo_stack[0]
    assert isinstance(state, OpeningProportionsUndoState)
    assert state.opening == opening
    workspace._apply_opening_proportions(_payload(doorway=kind == "doorway"))
    assert len(workspace._canvas_undo_stack) == 1
    workspace._opening_proportions_dialog.show_error.assert_not_called()
    assert workspace.global_scale == GlobalScale()
    assert workspace.current_level.height_meters == 4.0


@pytest.mark.parametrize("kind", ["window", "doorway"])
def test_undo_restores_height_and_arch_without_changing_other_opening(kind):
    opening = _window() if kind == "window" else _doorway()
    workspace = _Workspace(opening)
    unrelated = replace(_window(), window_id="unrelated-window")
    workspace.current_level.windows.append(unrelated)
    workspace._apply_opening_proportions(
        _payload(doorway=kind == "doorway", arch=kind == "doorway")
    )
    workspace._restore_opening_proportions_undo_state(workspace._canvas_undo_stack[0])
    if kind == "window":
        assert workspace.current_level.windows[0] == opening
    else:
        assert workspace.current_level.doorways[0] == opening
    assert workspace.current_level.windows[-1] is unrelated
    assert workspace.global_scale == GlobalScale()
    workspace._schedule_viewer_preview_refresh.assert_called_with(preserve_camera=True)


def test_apply_uses_stable_window_id_after_list_is_reordered():
    opening = _window()
    workspace = _Workspace(opening)
    unrelated = replace(opening, window_id="other-window", top_ratio=0.9)
    workspace.current_level.windows.insert(0, unrelated)
    workspace._apply_opening_proportions(_payload())
    assert workspace.current_level.windows[0] is unrelated
    assert workspace.current_level.windows[1].top_ratio == pytest.approx(0.65)
    assert workspace._canvas_undo_stack[0].opening.window_id == opening.window_id


def test_apply_uses_fresh_width_after_window_is_resized():
    opening = _window()
    workspace = _Workspace(opening)
    workspace.current_level.windows[0] = replace(opening, end_ratio=0.65)
    workspace._apply_opening_proportions(_payload())
    updated = workspace.current_level.windows[0]
    assert updated.end_ratio == 0.65
    assert updated.top_ratio == pytest.approx(0.2 + 1.6 * 1.5 / 4.0)
    assert workspace._opening_proportions_dialog.show_error.call_count == 0


@pytest.mark.parametrize("kind", ["window", "doorway"])
def test_deleted_opening_cannot_be_replaced_by_another_at_same_index(kind):
    opening = _window() if kind == "window" else _doorway()
    workspace = _Workspace(opening)
    replacement = (
        replace(opening, window_id="replacement-window")
        if kind == "window"
        else replace(opening, doorway_id="replacement-doorway")
    )
    if kind == "window":
        workspace.current_level.windows[0] = replacement
    else:
        workspace.current_level.doorways[0] = replacement
    workspace._apply_opening_proportions(_payload(doorway=kind == "doorway"))
    assert not workspace._canvas_undo_stack
    workspace._opening_proportions_dialog.show_error.assert_called_once()
    workspace._schedule_viewer_preview_refresh.assert_not_called()


def test_invalid_measurement_does_not_push_undo_or_change_scene():
    opening = _window()
    workspace = _Workspace(opening)
    workspace._apply_opening_proportions({"outline": (), "frame_size": (0, 0)})
    assert workspace.current_level.windows[0] is opening
    assert not workspace._canvas_undo_stack
    workspace._opening_proportions_dialog.show_error.assert_called_once()
    workspace._schedule_viewer_preview_refresh.assert_not_called()


def test_undo_restores_original_arch_after_flat_fifth_point_makes_rectangular_door():
    opening = replace(_doorway(height=2.2), shape=DOORWAY_SHAPE_ARCH, arch_amount=0.7)
    workspace = _Workspace(opening)
    workspace._apply_opening_proportions(_payload(doorway=True))
    updated = workspace.current_level.doorways[0]
    assert updated.shape == DOORWAY_SHAPE_RECTANGULAR
    assert updated.height_meters == pytest.approx(1.8)
    state = workspace._canvas_undo_stack[0]
    workspace._restore_opening_proportions_undo_state(state)
    assert workspace.current_level.doorways[0] == opening
    assert workspace.current_level.doorways[0].shape == DOORWAY_SHAPE_ARCH
    assert workspace.current_level.doorways[0].arch_amount == 0.7


@pytest.mark.parametrize("other_kind", ["window", "doorway"])
def test_undo_of_other_opening_preserves_current_editors_cached_target(other_kind):
    selected_opening = _window()
    workspace = _Workspace(selected_opening)
    selected_target = workspace._opening_proportions_target
    if other_kind == "window":
        before = replace(_window(), window_id="other-window", top_ratio=0.45)
        edited = replace(before, top_ratio=0.75)
        workspace.current_level.windows.append(edited)
    else:
        before = _doorway(height=1.4)
        edited = replace(before, height_meters=2.3)
        workspace.current_level.doorways.append(edited)
    workspace._restore_opening_proportions_undo_state(
        OpeningProportionsUndoState(2, other_kind, before)
    )
    assert workspace._opening_proportions_target is selected_target
    assert workspace._opening_proportions_reference == selected_target.reference
    assert workspace.current_level.windows[0] is selected_opening
    items = (
        workspace.current_level.windows
        if other_kind == "window"
        else workspace.current_level.doorways
    )
    assert items[-1] == before
    published = workspace.viewer.update_canvas_opening_target.call_args.args[0]
    assert published.reference.stable_id == (
        before.window_id if other_kind == "window" else before.doorway_id
    )


def test_publish_on_other_level_does_not_replace_cached_editor_target():
    opening = _window()
    workspace = _Workspace(opening)
    selected_target = workspace._opening_proportions_target
    other_level = LevelData(index=3, name="First floor", height_meters=4.0)
    other_level.windows = [opening]
    other_target = replace(
        selected_target,
        reference=replace(selected_target.reference, level_index=3),
    )
    updated = replace(opening, top_ratio=0.6)
    workspace._publish_proportions_opening(other_level, 0, updated, other_target)
    assert other_level.windows[0] == updated
    assert workspace._opening_proportions_target is selected_target


def test_publish_matches_stable_identity_despite_stale_editor_item_index():
    opening = _window()
    workspace = _Workspace(opening)
    workspace._opening_proportions_reference = replace(
        workspace._opening_proportions_reference, item_index=7
    )
    updated = replace(opening, top_ratio=0.6)
    workspace._publish_proportions_opening(
        workspace.current_level, 0, updated, workspace._opening_proportions_target
    )
    assert workspace._opening_proportions_target.bounds.top_ratio == 0.6


# ### Real dialog launch and isolated Generation-video preload ###
class _DialogWorkspace(_Workspace, QWidget):
    def __init__(self, opening):
        QWidget.__init__(self)
        _Workspace.__init__(self, opening)
        self._opening_proportions_dialog = None
        self.merged_generation_workspace = Mock()
        self._handle_canvas_undo_requested = Mock()
        _retained_dialog_hosts.append(self)


def test_open_preloads_generation_video_and_current_frame_without_seeking_its_reader():
    from housemaker.opening_proportions_dialog import OpeningProportionsDialog

    opening = _window()
    workspace = _DialogWorkspace(opening)
    generation_source = Mock()
    generation_source.metadata = SimpleNamespace(path="existing-generation-video.mp4")
    workspace.merged_generation_workspace._current_video_source.return_value = (
        generation_source
    )
    workspace.merged_generation_workspace._current_video_frame_index.return_value = 325
    with patch.object(OpeningProportionsDialog, "load_source") as load_source:
        workspace._open_opening_proportions(_target(opening).reference)
    dialog = workspace._opening_proportions_dialog
    try:
        assert isinstance(dialog, OpeningProportionsDialog)
        assert dialog.isVisible()
        load_source.assert_called_once_with(
            "existing-generation-video.mp4", frame_index=325
        )
        generation_source.get_frame.assert_not_called()
        generation_source.close.assert_not_called()
        assert workspace._opening_proportions_reference.stable_id == opening.window_id
        assert dialog._is_doorway is False
        assert dialog._width_meters == pytest.approx(1.2)
        assert not workspace._canvas_undo_stack
        workspace._schedule_viewer_preview_refresh.assert_not_called()
    finally:
        workspace._reset_opening_proportions()
        workspace.close()


def test_open_without_generation_video_keeps_manual_load_available():
    from housemaker.opening_proportions_dialog import OpeningProportionsDialog

    opening = _doorway()
    workspace = _DialogWorkspace(opening)
    workspace.merged_generation_workspace._current_video_source.return_value = None
    with patch.object(OpeningProportionsDialog, "load_source") as load_source:
        workspace._open_opening_proportions(_target(opening).reference)
    dialog = workspace._opening_proportions_dialog
    try:
        load_source.assert_not_called()
        assert dialog.load_video_button.isEnabled()
        assert dialog._is_doorway is True
        assert "Doorway" in dialog.target_label.text()
        assert not workspace._canvas_undo_stack
    finally:
        workspace._reset_opening_proportions()
        workspace.close()


@pytest.mark.parametrize(
    "error",
    [
        OSError("The video was moved"),
        ValueError("Corrupt video"),
        IndexError("No frame"),
    ],
)
def test_default_video_failure_still_shows_editor_and_allows_another_video(error):
    from housemaker.opening_proportions_dialog import OpeningProportionsDialog

    opening = _window()
    workspace = _DialogWorkspace(opening)
    workspace.merged_generation_workspace._current_video_source.return_value = (
        SimpleNamespace(metadata=SimpleNamespace(path="unavailable-video.mp4"))
    )
    workspace.merged_generation_workspace._current_video_frame_index.return_value = 12
    with patch.object(OpeningProportionsDialog, "load_source", side_effect=error):
        workspace._open_opening_proportions(_target(opening).reference)
    dialog = workspace._opening_proportions_dialog
    try:
        assert dialog.isVisible()
        assert dialog.load_video_button.isEnabled()
        assert "Load another video" in dialog.result_label.text()
        assert str(error) in dialog.result_label.text()
        assert not dialog.apply_button.isEnabled()
        assert not workspace._canvas_undo_stack
    finally:
        workspace._reset_opening_proportions()
        workspace.close()
