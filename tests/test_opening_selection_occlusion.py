"""Opening selection and controls respect nearer geometry, including pending cuts."""

# ### Environment setup ###
from __future__ import annotations

import os
from dataclasses import replace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ### Imports ###
import numpy as np
import pytest
import trimesh
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QApplication

from housemaker.canvas_openings import (
    CanvasOpeningBounds,
    CanvasOpeningReference,
    CanvasOpeningTarget,
)
from housemaker.surface_geometry import FixedSurface
from housemaker.viewer import (
    CANVAS_OPENING_GIZMO_ANCHOR,
    CANVAS_OPENING_GIZMO_SIDE,
    CANVAS_OPENING_SIDE_LEFT,
    GlbViewerWidget,
    _get_canvas_opening_creation_local_position,
)

# ### Fixture helpers ###
_application = QApplication.instance() or QApplication([])
_application.setQuitOnLastWindowClosed(False)
_HOST_ID = "level:0/wall:1:2"
_BACK_HOST_ID = "level:0/wall:3:4"
_BLOCKER_ID = "level:0/wall:5:6"


def _wall(surface_id: str, y: float) -> FixedSurface:
    mesh = trimesh.Trimesh(
        vertices=((0, y, 0), (4, y, 0), (4, y, 3), (0, y, 3)),
        faces=((0, 1, 2), (0, 2, 3)),
        process=False,
    )
    return FixedSurface(
        surface_id=surface_id,
        surface_type="wall",
        level_index=0,
        room_index=None,
        mesh=mesh,
        area_square_meters=12.0,
        wall_key=surface_id.split("/wall:", 1)[1],
        wall_start_world=(0.0, y, 0.0),
        wall_end_world=(4.0, y, 0.0),
        wall_height_meters=3.0,
    )


def _target(kind: str, *, pending: bool = False) -> CanvasOpeningTarget:
    return CanvasOpeningTarget(
        reference=CanvasOpeningReference(kind, 0, 0, f"{kind}-a"),
        wall_surface_id=_HOST_ID,
        plane_start_world=(0.0, 0.0, 0.0),
        wall_tangent_world=(1.0, 0.0, 0.0),
        wall_normal_world=(0.0, -1.0, 0.0),
        wall_width_meters=4.0,
        wall_height_meters=3.0,
        minimum_width_meters=0.2,
        minimum_height_meters=0.3,
        bounds=CanvasOpeningBounds(0.25, 0.75, 0.2, 0.8),
        host_wall_planes_world=(
            ((0.0, -0.1, 0.0), (0.0, 0.1, 0.0)) if pending else ((0.0, 0.0, 0.0),)
        ),
    )


def _install(
    viewer: GlbViewerWidget,
    kind: str,
    *,
    pending: bool = False,
    blocked: bool = False,
) -> CanvasOpeningTarget:
    target = _target(kind, pending=pending)
    walls = (
        [_wall(_HOST_ID, -0.1), _wall(_BACK_HOST_ID, 0.1)]
        if pending
        else [_wall(_HOST_ID, 0.0)]
    )
    if blocked:
        walls.append(_wall(_BLOCKER_ID, -1.0))
    viewer.set_wall_targets(tuple(walls))
    viewer.set_canvas_opening_targets((target,))
    if pending:
        viewer.set_pending_canvas_opening_target(target)
    return target


def _ray_at(point: object, *, reverse: bool = False) -> tuple[np.ndarray, np.ndarray]:
    x, _y, z = np.asarray(point, dtype=float)
    return np.asarray((x, 2.0 if reverse else -2.0, z)), np.asarray(
        (0.0, -1.0 if reverse else 1.0, 0.0)
    )


def _select_at_center(
    viewer: GlbViewerWidget, mode: str, *, reverse: bool = False
) -> None:
    with patch.object(
        viewer.view,
        "build_camera_ray",
        return_value=_ray_at((2.0, 0.0, 1.5), reverse=reverse),
    ):
        if mode == "fps":
            viewer._handle_first_person_center_pick_requested(QPointF())
        else:
            modifier = (
                Qt.KeyboardModifier.ShiftModifier
                if mode == "shift"
                else Qt.KeyboardModifier.NoModifier
            )
            with patch(
                "housemaker.viewer.QApplication.keyboardModifiers",
                return_value=modifier,
            ):
                viewer._handle_window_wall_pick_requested(QPointF())


def _control_ray(
    viewer: GlbViewerWidget, target: CanvasOpeningTarget, control: str
) -> tuple[np.ndarray, np.ndarray]:
    if control == "star":
        viewer._update_opening_proportions_control_geometry()
        assert viewer._opening_proportions_center is not None
        point = viewer._opening_proportions_center
    elif control == "plus":
        local = _get_canvas_opening_creation_local_position(target)
        assert local is not None
        point = target.local_to_world(*local)
    elif control == "side":
        point = target.local_to_world(0.25, 0.5)
    else:
        point = target.get_world_center()
    return _ray_at(point)


@pytest.fixture
def viewer():
    widget = GlbViewerWidget(window_editing_enabled=True)
    widget.view.setCameraPosition(elevation=0.0, azimuth=-90.0)
    yield widget
    widget.exit_first_person_mode()
    widget.close()
    widget.deleteLater()
    _application.processEvents()


# ### Opening selection depth ordering ###
@pytest.mark.parametrize("kind", ("window", "doorway"))
@pytest.mark.parametrize("mode", ("normal", "shift", "fps"))
def test_nearer_unrelated_wall_blocks_opening_selection(viewer, kind, mode):
    _install(viewer, kind, blocked=True)

    _select_at_center(viewer, mode)

    assert viewer.get_selected_canvas_opening_reference() is None
    assert viewer.get_selected_wall_surface_id() == _BLOCKER_ID


@pytest.mark.parametrize("kind", ("window", "doorway"))
@pytest.mark.parametrize("mode", ("normal", "shift", "fps"))
def test_visible_opening_wins_its_coplanar_host_wall(viewer, kind, mode):
    target = _install(viewer, kind)

    _select_at_center(viewer, mode)

    assert viewer.get_selected_canvas_opening_reference() == target.reference
    assert viewer.get_selected_wall_surface_id() is None


# ### Pending wall cuts exempt only their own host geometry ###
@pytest.mark.parametrize("kind", ("window", "doorway"))
@pytest.mark.parametrize("mode", ("normal", "shift", "fps"))
def test_pending_opening_can_be_selected_through_its_own_uncut_host(viewer, kind, mode):
    target = _install(viewer, kind, pending=True)

    _select_at_center(viewer, mode)

    assert viewer.get_selected_canvas_opening_reference() == target.reference


@pytest.mark.parametrize("kind", ("window", "doorway"))
@pytest.mark.parametrize("mode", ("normal", "shift", "fps"))
def test_pending_opening_does_not_bypass_an_unrelated_nearer_wall(viewer, kind, mode):
    _install(viewer, kind, pending=True, blocked=True)

    _select_at_center(viewer, mode)

    assert viewer.get_selected_canvas_opening_reference() is None
    assert viewer.get_selected_wall_surface_id() == _BLOCKER_ID


@pytest.mark.parametrize("kind", ("window", "doorway"))
@pytest.mark.parametrize("mode", ("normal", "shift", "fps"))
def test_blocker_between_stale_host_and_pending_opening_is_not_skipped(
    viewer, kind, mode
):
    _install(viewer, kind, pending=True)
    viewer.set_wall_targets(
        (
            _wall(_HOST_ID, -0.1),
            _wall(_BACK_HOST_ID, 0.1),
            _wall(_BLOCKER_ID, -0.05),
        )
    )

    _select_at_center(viewer, mode)

    assert viewer.get_selected_canvas_opening_reference() is None
    assert viewer.get_selected_wall_surface_id() == _BLOCKER_ID


@pytest.mark.parametrize("kind", ("window", "doorway"))
@pytest.mark.parametrize("mode", ("normal", "shift", "fps"))
def test_pending_host_planes_work_when_committed_host_id_is_missing(viewer, kind, mode):
    _install(viewer, kind, pending=True)
    target = replace(_target(kind, pending=True), wall_surface_id="level:0/wall:9:10")
    viewer.set_canvas_opening_targets(())
    viewer.set_pending_canvas_opening_target(target)

    _select_at_center(viewer, mode)

    assert viewer.get_selected_canvas_opening_reference() == target.reference


@pytest.mark.parametrize("kind", ("window", "doorway"))
@pytest.mark.parametrize("mode", ("normal", "shift", "fps"))
def test_pending_own_host_skins_do_not_block_reverse_side_selection(viewer, kind, mode):
    target = _install(viewer, kind, pending=True)

    _select_at_center(viewer, mode, reverse=True)

    assert viewer.get_selected_canvas_opening_reference() == target.reference


@pytest.mark.parametrize("kind", ("window", "doorway"))
def test_pending_opening_host_id_alone_cannot_exempt_a_distant_wall(viewer, kind):
    target = _target(kind, pending=True)
    viewer.set_wall_targets((_wall(_HOST_ID, -1.0),))
    viewer.set_pending_canvas_opening_target(target)

    _select_at_center(viewer, "normal")

    assert viewer.get_selected_canvas_opening_reference() is None
    assert viewer.get_selected_wall_surface_id() == _HOST_ID


# ### Opening controls must not capture a click through another wall ###
@pytest.mark.parametrize("kind", ("window", "doorway"))
@pytest.mark.parametrize("control", ("side", "anchor", "plus", "star"))
@pytest.mark.parametrize("pending", (False, True))
def test_occluded_opening_controls_fall_through_to_normal_selection(
    viewer, kind, control, pending
):
    target = _install(viewer, kind, pending=pending, blocked=True)
    viewer.select_canvas_opening(target.reference)

    with patch.object(viewer.view, "pixelSize", return_value=0.01):
        ray = _control_ray(viewer, target, control)
        with (
            patch.object(viewer.view, "build_camera_ray", return_value=ray),
            patch.object(viewer, "_begin_canvas_rectangle_selection") as fallback,
        ):
            viewer._handle_placed_object_pointer_pressed(QPointF())

    assert viewer._canvas_opening_edit_drag is None
    assert viewer._canvas_opening_creation_pressed_reference is None
    assert viewer._opening_proportions_pressed_reference is None
    assert not viewer.view.is_primary_pointer_drag_reserved
    fallback.assert_called_once()


@pytest.mark.parametrize("kind", ("window", "doorway"))
@pytest.mark.parametrize("control", ("side", "anchor", "plus", "star"))
@pytest.mark.parametrize("pending", (False, True))
def test_visible_opening_controls_remain_usable_before_and_after_wall_cut(
    viewer, kind, control, pending
):
    target = _install(viewer, kind, pending=pending)
    viewer.select_canvas_opening(target.reference)

    with patch.object(viewer.view, "pixelSize", return_value=0.01):
        ray = _control_ray(viewer, target, control)
        with patch.object(viewer.view, "build_camera_ray", return_value=ray):
            viewer._handle_placed_object_pointer_pressed(QPointF())

    assert viewer.view.is_primary_pointer_drag_reserved
    if control == "star":
        assert viewer._opening_proportions_pressed_reference == target.reference
    elif control == "plus":
        assert viewer._canvas_opening_creation_pressed_reference == target.reference
    else:
        drag = viewer._canvas_opening_edit_drag
        assert drag is not None
        assert drag.handle.kind == (
            CANVAS_OPENING_GIZMO_SIDE
            if control == "side"
            else CANVAS_OPENING_GIZMO_ANCHOR
        )
        if control == "side":
            assert drag.handle.side == CANVAS_OPENING_SIDE_LEFT
