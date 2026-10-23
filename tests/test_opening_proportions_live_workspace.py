"""The shared editor previews annotations without committing scene changes."""

# ### Environment and imports ###
from __future__ import annotations

import os
from dataclasses import replace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QTimer
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QApplication

from tests import test_opening_proportions_integration as integration
from tests.test_opening_proportions_workspace import (
    _doorway,
    _payload,
    _window,
    _Workspace,
)

# ### Test image and editor helpers ###
workspace = integration.workspace


@pytest.fixture
def frame_path(tmp_path):
    path = tmp_path / "opening.png"
    success, png = cv2.imencode(".png", np.full((400, 400, 3), 180, np.uint8))
    assert success
    path.write_bytes(png.tobytes())
    return path


def _open(value, target, frame_path):
    value.viewer.opening_proportions_requested.emit(target.reference)
    dialog = value._opening_proportions_dialog
    dialog.load_source(frame_path)
    return dialog


# ### Immediate, reversible display-only previews ###
@pytest.mark.parametrize("kind", ("window", "doorway"))
def test_first_corner_move_previews_without_mutation_or_undo(
    workspace, frame_path, kind
):
    value, level, targets = workspace
    target = next(target for target in targets if target.reference.kind == kind)
    original = replace(level.windows[0] if kind == "window" else level.doorways[0])
    dialog = _open(value, target, frame_path)
    assert value.viewer._opening_proportions_preview_target is None
    left, top = dialog.video_view.outline[0]
    dialog.video_view._move_handle(0, (left + 0.025, top))
    preview = value.viewer._opening_proportions_preview_target
    assert preview is not None
    assert preview.reference == target.reference
    assert preview.bounds.bottom_ratio == target.bounds.bottom_ratio
    assert preview.bounds.start_ratio == target.bounds.start_ratio
    assert preview.bounds.end_ratio == target.bounds.end_ratio
    assert (level.windows[0] if kind == "window" else level.doorways[0]) == original
    assert not value._canvas_undo_stack
    assert "Height:" not in dialog.result_label.text()
    assert "Arch rise:" not in dialog.result_label.text()
    dialog.close()
    assert value.viewer._opening_proportions_preview_target is None
    assert (level.windows[0] if kind == "window" else level.doorways[0]) == original


@pytest.mark.parametrize("kind", ("window", "doorway"))
def test_default_frame_has_no_preview_and_invalid_edit_clears_previous_preview(kind):
    opening = _window() if kind == "window" else _doorway()
    value = _Workspace(opening)
    payload = _payload(doorway=kind == "doorway")
    value._preview_opening_proportions(payload)
    value.viewer.set_opening_proportions_preview.assert_not_called()
    payload["outline_edited"] = True
    value._preview_opening_proportions(payload)
    value.viewer.set_opening_proportions_preview.assert_called_once()
    value.viewer.clear_opening_proportions_preview.reset_mock()
    payload["outline"] = ()
    value._preview_opening_proportions(payload)
    value.viewer.clear_opening_proportions_preview.assert_called_once()
    assert not value._canvas_undo_stack


def test_pink_arch_move_previews_without_dimension_readout_or_scene_change(
    workspace, frame_path
):
    value, level, targets = workspace
    target = next(target for target in targets if target.reference.kind == "doorway")
    original = replace(level.doorways[0])
    dialog = _open(value, target, frame_path)
    width_before, height_before = dialog.size().width(), dialog.size().height()
    x, y = dialog.video_view.arch_apex
    dialog.video_view._move_handle(4, (x, y - 0.02))
    preview = value.viewer._opening_proportions_preview_target
    assert preview.arch_amount > 0
    assert level.doorways[0] == original
    assert not value._canvas_undo_stack
    assert "Height:" not in dialog.result_label.text()
    assert "Arch rise:" not in dialog.result_label.text()
    assert (dialog.width(), dialog.height()) == (width_before, height_before)
    dialog.apply_button.click()
    assert value.viewer._opening_proportions_preview_target is None
    assert len(value._canvas_undo_stack) == 1


# ### One editor instance, independent per-opening annotations ###
def test_switching_openings_reuses_window_and_keeps_source_and_manual_outline(
    workspace, frame_path
):
    value, _level, targets = workspace
    window = next(target for target in targets if target.reference.kind == "window")
    door = next(target for target in targets if target.reference.kind == "doorway")
    dialog = _open(value, window, frame_path)
    point = dialog.video_view.outline[0]
    dialog.video_view._move_handle(0, (point[0] + 0.025, point[1]))
    window_outline = dialog.video_view.outline
    with patch.object(dialog, "load_source", wraps=dialog.load_source) as load:
        value.viewer.opening_proportions_requested.emit(door.reference)
        assert value._opening_proportions_dialog is dialog
        assert dialog.video_view.arch_apex is not None
        assert dialog.current_measurement()["source_path"] == str(frame_path.resolve())
        assert value.viewer._opening_proportions_preview_target is None
        value.viewer.opening_proportions_requested.emit(window.reference)
        assert value._opening_proportions_dialog is dialog
        assert dialog.video_view.arch_apex is None
        assert dialog.video_view.outline == window_outline
        assert (
            value.viewer._opening_proportions_preview_target.reference
            == window.reference
        )
        load.assert_not_called()


def test_reactivating_same_opening_only_returns_cursor_without_resolving_or_reloading(
    workspace, frame_path
):
    value, _level, targets = workspace
    target = targets[0]
    dialog = _open(value, target, frame_path)
    QApplication.processEvents()
    with (
        patch.object(value, "_resolve_proportions_target") as resolve,
        patch.object(dialog, "load_source") as load,
        patch.object(QCursor, "setPos") as move,
    ):
        # A reordered list is still the same stable opening.
        value._open_opening_proportions(replace(target.reference, item_index=9))
        QApplication.processEvents()
        resolve.assert_not_called()
        load.assert_not_called()
        move.assert_called_with(dialog.mapToGlobal(dialog.rect().center()))
    assert value._opening_proportions_dialog is dialog


def test_closed_editor_can_reopen_in_same_instance_with_its_annotations(
    workspace, frame_path
):
    value, _level, targets = workspace
    target = targets[0]
    dialog = _open(value, target, frame_path)
    x, y = dialog.video_view.outline[0]
    dialog.video_view._move_handle(0, (x + 0.025, y))
    outline = dialog.video_view.outline
    dialog.close()
    value._open_opening_proportions(target.reference)
    assert value._opening_proportions_dialog is dialog
    assert dialog.isVisible()
    assert dialog.video_view.outline == outline
    assert dialog.current_measurement()["outline_edited"]


# ### Cursor focus handoff and project cleanup ###
def test_editor_handoff_releases_mouse_look_before_warping_cursor(workspace):
    value, _level, targets = workspace
    value.viewer.enter_first_person_mode()
    pose = value.viewer.get_first_person_camera_pose()
    assert value.viewer.is_first_person_pointer_captured
    with patch.object(QCursor, "setPos"):
        value.viewer.opening_proportions_requested.emit(targets[0].reference)
        assert not value.viewer.is_first_person_pointer_captured
        assert value.viewer.is_first_person_active
        assert value.viewer.get_first_person_camera_pose() == pose
        QApplication.processEvents()


def test_open_and_close_center_the_cursor_in_editor_then_the_actual_view(workspace):
    value, _level, targets = workspace
    with patch.object(QCursor, "setPos") as move:
        value.viewer.opening_proportions_requested.emit(targets[0].reference)
        dialog = value._opening_proportions_dialog
        QApplication.processEvents()
        move.assert_called_with(dialog.mapToGlobal(dialog.rect().center()))
        dialog.close()
        QApplication.processEvents()
        view = value.viewer.view
        move.assert_called_with(view.mapToGlobal(view.rect().center()))


def test_reset_cancels_pending_cursor_handoff(workspace):
    value, _level, targets = workspace
    callbacks = []
    with (
        patch.object(
            QTimer,
            "singleShot",
            side_effect=lambda _delay, call: callbacks.append(call),
        ),
        patch.object(QCursor, "setPos") as move,
    ):
        value.viewer.opening_proportions_requested.emit(targets[0].reference)
        value._reset_opening_proportions()
        for callback in callbacks:
            callback()
        move.assert_not_called()
