"""Real main-window opening entry, deterministic apply and Canvas undo."""

# ### Environment and imports ###
from __future__ import annotations

import os
from dataclasses import replace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.canvas_openings import build_canvas_opening_targets
from housemaker.main import BlueprintWorkspace
from housemaker.models import DOORWAY_SHAPE_ARCH
from tests.test_canvas_opening_main import _build_level_and_wall

# ### A real workspace with a simple authored wall, no providers or GL uploads ###
_application = QApplication.instance() or QApplication([])
_application.setQuitOnLastWindowClosed(False)
_retained = []


@pytest.fixture
def workspace(tmp_path):
    value = BlueprintWorkspace(
        application_settings=ApplicationSettingsStore(tmp_path / "settings.json")
    )
    _retained.append(value)
    level, wall = _build_level_and_wall()
    level.windows[0] = replace(level.windows[0], top_ratio=0.55)
    level.doorways[0] = replace(level.doorways[0], height_meters=0.8)
    value.levels = [level]
    value.current_level_index = 0
    value._reset_viewer_doorway_snapshots()
    value._sync_canvas_to_current_level()
    value._set_canvas_viewer_targets((wall,))
    value._canvas_undo_stack.clear()

    def targets(levels, **_kwargs):
        return {
            target.key: target
            for target in build_canvas_opening_targets(levels, (wall,))
        }

    with (
        patch.object(
            value, "_build_opening_target_lookup_for_levels", side_effect=targets
        ),
        patch.object(value, "_schedule_viewer_preview_refresh"),
        patch.object(value, "_queue_viewer_preview_refresh"),
    ):
        yield value, level, tuple(targets([level]).values())
        value.shutdown()
        value.close()


# ### Click entry and Apply travel through the real connected controls ###
@pytest.mark.parametrize("kind", ("window", "doorway"))
def test_opening_editor_signal_apply_and_shared_undo(workspace, tmp_path, kind):
    value, level, targets = workspace
    target = next(target for target in targets if target.reference.kind == kind)
    previous = replace(level.windows[0] if kind == "window" else level.doorways[0])
    other = replace(level.doorways[0] if kind == "window" else level.windows[0])
    scale = value.global_scale
    vertex_data = level.vertex_data.to_dict()
    value.viewer.opening_proportions_requested.emit(target.reference)
    dialog = value._opening_proportions_dialog
    assert dialog is not None
    frame = tmp_path / "opening.png"
    success, png = cv2.imencode(".png", np.full((400, 400, 3), 180, np.uint8))
    assert success
    frame.write_bytes(png.tobytes())
    dialog.load_source(str(frame))
    assert dialog.video_view.has_frame()
    outline = ((0.2, 0.2), (0.6, 0.2), (0.6, 0.8), (0.2, 0.8))
    apex = (0.4, 0.1) if kind == "doorway" else None
    dialog.video_view.set_outline(outline, apex)
    dialog.video_view.outline_changed.emit()
    assert dialog.apply_button.isEnabled()
    dialog.apply_button.click()
    assert len(value._canvas_undo_stack) == 1
    if kind == "window":
        assert level.windows[0].top_ratio == pytest.approx(0.7)
        assert level.windows[0].bottom_ratio == previous.bottom_ratio
        assert level.doorways[0] == other
    else:
        assert level.doorways[0].height_meters == pytest.approx(1.75)
        assert level.doorways[0].shape == DOORWAY_SHAPE_ARCH
        assert level.doorways[0].arch_amount == pytest.approx(0.5)
        assert level.windows[0] == other
    assert value.global_scale == scale
    assert level.height_meters == 3
    assert level.vertex_data.to_dict() == vertex_data
    value._handle_canvas_undo_requested()
    assert (level.windows[0] if kind == "window" else level.doorways[0]) == previous
    assert value.global_scale == scale


def test_project_replacement_cleanup_closes_the_editor(workspace):
    value, _level, targets = workspace
    value.viewer.opening_proportions_requested.emit(targets[0].reference)
    dialog = value._opening_proportions_dialog
    value._reset_opening_proportions()
    assert value._opening_proportions_dialog is None
    assert value._opening_proportions_reference is None
    assert not dialog.isVisible()
