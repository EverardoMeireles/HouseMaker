"""Opening proportion entry points do not disturb navigation or scene picking."""

# ### Environment setup ###
from __future__ import annotations

import os
from dataclasses import replace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ### Imports ###
import numpy as np
import pyqtgraph.opengl as gl
import pytest
import trimesh
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QVector3D
from PySide6.QtWidgets import QApplication

from housemaker.canvas_openings import (
    CanvasOpeningBounds,
    CanvasOpeningReference,
    CanvasOpeningTarget,
)
from housemaker.glb import GeneratedModel, PreviewPlacedObject
from housemaker.surface_geometry import FixedSurface
from housemaker.viewer import GlbViewerWidget, SelectableGLViewWidget

# ### Fixture helpers ###
_application = QApplication.instance() or QApplication([])
_application.setQuitOnLastWindowClosed(False)


def _target(kind="window", arch_amount=None):
    return CanvasOpeningTarget(
        reference=CanvasOpeningReference(kind, 0, 0, "opening-a"),
        wall_surface_id="level:0/wall:1:2",
        plane_start_world=(0.0, 0.0, 0.0),
        wall_tangent_world=(1.0, 0.0, 0.0),
        wall_normal_world=(0.0, -1.0, 0.0),
        wall_width_meters=4.0,
        wall_height_meters=3.0,
        minimum_width_meters=0.2,
        minimum_height_meters=0.3,
        bounds=CanvasOpeningBounds(0.25, 0.75, 0.2, 0.8),
        arch_amount=arch_amount,
    )


def _wall(*, y=0.0, surface_id="level:0/wall:1:2"):
    return FixedSurface(
        surface_id=surface_id,
        surface_type="wall",
        level_index=0,
        room_index=None,
        mesh=trimesh.Trimesh(
            vertices=[(0, y, 0), (4, y, 0), (4, y, 3), (0, y, 3)],
            faces=[(0, 1, 2), (0, 2, 3)],
            process=False,
        ),
        area_square_meters=12.0,
        wall_key="1:2",
        wall_start_world=(0, y, 0),
        wall_end_world=(4, y, 0),
        wall_height_meters=3.0,
    )


def _mouse(kind, *, position=None, button=Qt.MouseButton.LeftButton):
    if position is None:
        position = QPointF(13, 17)
    return QMouseEvent(
        kind,
        position,
        position,
        button,
        Qt.MouseButton.NoButton if kind == QEvent.Type.MouseButtonRelease else button,
        Qt.KeyboardModifier.NoModifier,
    )


def _center_ray():
    return np.asarray((2.0, -2.0, 1.5)), np.asarray((0.0, 1.0, 0.0))


def _button_ray(viewer):
    viewer._update_opening_proportions_control_geometry()
    center = viewer._opening_proportions_center.copy()
    inverse, valid = viewer.view.viewMatrix().inverted()
    assert valid
    vector = inverse.mapVector(QVector3D(0, 0, 1))
    normal = np.asarray((vector.x(), vector.y(), vector.z()))
    return center + normal * 3.0, -normal


@pytest.fixture
def viewer():
    widget = GlbViewerWidget(window_editing_enabled=True)
    widget.set_wall_targets((_wall(),))
    widget.set_canvas_opening_targets((_target(),))
    yield widget
    widget.exit_first_person_mode()
    widget.close()
    widget.deleteLater()
    _application.processEvents()


# ### First-person center ray selection ###
def test_fps_captured_click_selects_opening_at_center_without_moving_camera(viewer):
    view = viewer.view
    view.resize(640, 480)
    viewer.enter_first_person_mode()
    pose = viewer.get_first_person_camera_pose()
    with patch.object(view, "build_camera_ray", return_value=_center_ray()) as ray:
        view.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress))
        view.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease))
    assert ray.call_args.args[0] == QPointF(view.rect().center())
    assert viewer.get_selected_canvas_opening_reference() == _target().reference
    assert viewer.get_first_person_camera_pose() == pose
    assert viewer.is_first_person_pointer_captured
    assert viewer.is_first_person_active


def test_fps_center_click_also_selects_plain_walls_and_not_openings_behind_them(viewer):
    blocker = _wall(y=-1.0, surface_id="level:0/wall:3:4")
    viewer.set_wall_targets((_wall(), blocker))
    viewer.enter_first_person_mode()
    with patch.object(viewer.view, "build_camera_ray", return_value=_center_ray()):
        viewer.view.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress))
        viewer.view.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease))
    assert viewer.get_selected_canvas_opening_reference() is None
    assert viewer.get_selected_wall_surface_id() == blocker.surface_id
    assert viewer.is_first_person_pointer_captured


def test_fps_selection_uses_cpu_picking_not_gl_item_selection(viewer):
    viewer.enter_first_person_mode()
    viewer.view._get_clicked_items = Mock(side_effect=AssertionError("No GL picking"))
    with patch.object(viewer.view, "build_camera_ray", return_value=_center_ray()):
        viewer.view.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress))
        viewer.view.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease))
    assert not viewer.view._get_clicked_items.called


def test_double_click_in_fps_uses_crosshair_not_event_position(viewer):
    viewer.view.resize(640, 480)
    viewer.enter_first_person_mode()
    pose = viewer.get_first_person_camera_pose()
    requested = []
    viewer.opening_proportions_requested.connect(requested.append)
    with patch.object(
        viewer.view, "build_camera_ray", return_value=_center_ray()
    ) as ray:
        viewer.view.mouseDoubleClickEvent(_mouse(QEvent.Type.MouseButtonDblClick))
    assert requested == [_target().reference]
    assert ray.call_args.args[0] == QPointF(viewer.view.rect().center())
    assert viewer.get_first_person_camera_pose() == pose
    assert viewer.is_first_person_pointer_captured


def test_disabled_selection_does_not_emit_center_click():
    view = SelectableGLViewWidget()
    view.set_viewport_click_selection_enabled(False)
    events = []
    view.first_person_center_clicked.connect(events.append)
    view.enter_first_person_mode()
    view.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress))
    view.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease))
    assert events == []
    view.exit_first_person_mode()
    view.close()
    view.deleteLater()


# ### Opening activation and occlusion ###
@pytest.mark.parametrize(
    "kind,arch", [("window", None), ("doorway", None), ("doorway", 0.45)]
)
def test_opening_double_click_emits_same_reference_for_every_opening_type(
    viewer, kind, arch
):
    target = _target(kind, arch)
    viewer.set_canvas_opening_targets((target,))
    requested = []
    viewer.opening_proportions_requested.connect(requested.append)
    with patch.object(viewer.view, "build_camera_ray", return_value=_center_ray()):
        viewer.view.mouseDoubleClickEvent(_mouse(QEvent.Type.MouseButtonDblClick))
    assert requested == [target.reference]
    assert viewer.get_selected_canvas_opening_reference() == target.reference


def test_double_click_does_not_open_an_opening_through_another_wall(viewer):
    viewer.set_wall_targets((_wall(), _wall(y=-1.0, surface_id="level:0/wall:3:4")))
    requested = []
    viewer.opening_proportions_requested.connect(requested.append)
    with patch.object(viewer.view, "build_camera_ray", return_value=_center_ray()):
        viewer.view.mouseDoubleClickEvent(_mouse(QEvent.Type.MouseButtonDblClick))
    assert requested == []


def test_nearer_objects_stairs_and_trim_occlude_opening_activation(viewer):
    ray = _center_ray()
    for name in (
        "_get_nearest_preview_placed_object_ray_hit",
        "_get_nearest_preview_stair_part_ray_hit",
        "_get_nearest_architectural_trim_part_ray_hit",
    ):
        with (
            patch.object(viewer.view, "build_camera_ray", return_value=ray),
            patch(
                f"housemaker.viewer.{name}",
                return_value=(object(), np.zeros(3), 0.5),
            ),
        ):
            assert viewer._pick_opening_proportions_target(QPointF()) is None


def test_invisible_opening_is_not_activated(viewer):
    # Hide the wall through the real visibility controls: pending hosts and
    # committed hosts share the same level/surface-type visibility predicate.
    viewer.set_canvas_surface_focus_type("floor")
    assert viewer.get_visible_canvas_surface_ids() == ()
    with patch.object(
        viewer.view,
        "build_camera_ray",
        return_value=_center_ray(),
    ):
        assert viewer._pick_opening_proportions_target(QPointF()) is None


# ### Installed opening fixtures ###
@pytest.mark.parametrize("kind", ["window", "doorway"])
def test_installed_fixtures_redirect_selection_and_activation_to_their_opening(
    viewer, kind
):
    target = _target(kind)
    viewer.set_canvas_opening_targets((target,))
    base = trimesh.creation.box(extents=(0.01, 0.01, 0.01))
    base.apply_translation((100, 100, 100))
    fixture = trimesh.creation.box(extents=(1.5, 0.2, 1.6))
    fixture.apply_translation((2.0, 0.0, 1.5))
    object_id = f"{kind}-placement:fixture-a"
    viewer.set_model(
        GeneratedModel(
            mesh=base,
            scene=trimesh.Scene(base),
            glb_bytes=b"",
            preview_base_mesh=base.copy(),
            preview_placed_objects=[
                PreviewPlacedObject(
                    object_id=object_id,
                    meshes=(fixture,),
                    placement_transform=np.eye(4),
                    world_position=(0, 0, 0),
                    rotation_degrees=(0, 0, 0),
                )
            ],
        )
    )
    assert viewer.set_placed_opening_bindings({object_id: target.reference})
    requested = []
    viewer.opening_proportions_requested.connect(requested.append)
    viewer.enter_first_person_mode()
    with patch.object(viewer.view, "build_camera_ray", return_value=_center_ray()):
        viewer.view.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress))
        viewer.view.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease))
        assert viewer.get_selected_canvas_opening_reference() == target.reference
        viewer.view.mouseDoubleClickEvent(_mouse(QEvent.Type.MouseButtonDblClick))
    assert requested == [target.reference]
    assert viewer.get_selected_placed_object_ids() == ()


# ### Asterisk rendering, hover and gestures ###
def test_asterisk_exists_only_for_selected_opening_and_uses_three_vector_strokes(
    viewer,
):
    assert viewer._opening_proportions_items == []
    viewer.select_canvas_opening(_target().reference)
    background, icon = viewer._opening_proportions_items
    assert isinstance(background, gl.GLScatterPlotItem)
    assert isinstance(icon, gl.GLLinePlotItem)
    assert np.asarray(icon.pos).shape == (6, 3)
    assert background.size >= 24
    previous = tuple(viewer._opening_proportions_items)
    viewer.select_canvas_opening(None)
    assert viewer._opening_proportions_items == []
    assert all(item not in viewer.view.items for item in previous)


def test_star_click_emits_opening_reference_instead_of_fixture_creation(viewer):
    viewer.select_canvas_opening(_target().reference)
    requested = []
    fixtures = []
    viewer.opening_proportions_requested.connect(requested.append)
    viewer.window_fixture_creation_requested.connect(fixtures.append)
    with patch.object(
        viewer.view, "build_camera_ray", return_value=_button_ray(viewer)
    ):
        viewer.view.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress))
        assert viewer.view.is_primary_pointer_drag_reserved
        viewer.view.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease))
    assert requested == [_target().reference]
    assert fixtures == []
    assert not viewer.view.is_primary_pointer_drag_reserved


def test_releasing_away_from_star_does_not_open_editor(viewer):
    viewer.select_canvas_opening(_target().reference)
    requested = []
    viewer.opening_proportions_requested.connect(requested.append)
    with patch.object(
        viewer.view, "build_camera_ray", return_value=_button_ray(viewer)
    ):
        viewer.view.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress))
    with patch.object(viewer.view, "build_camera_ray", return_value=None):
        viewer.view.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease))
    assert requested == []
    assert not viewer.view.is_primary_pointer_drag_reserved


def test_right_click_cancel_star_does_not_activate_editor(viewer):
    viewer.select_canvas_opening(_target().reference)
    requested = []
    viewer.opening_proportions_requested.connect(requested.append)
    with patch.object(
        viewer.view, "build_camera_ray", return_value=_button_ray(viewer)
    ):
        viewer.view.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress))
        viewer.view.mousePressEvent(
            _mouse(QEvent.Type.MouseButtonPress, button=Qt.MouseButton.RightButton)
        )
    assert requested == []
    assert viewer._opening_proportions_pressed_reference is None
    assert not viewer.view.is_primary_pointer_drag_reserved


def test_fps_released_cursor_can_click_star_without_toggling_navigation(viewer):
    viewer.select_canvas_opening(_target().reference)
    viewer.enter_first_person_mode()
    pose = viewer.get_first_person_camera_pose()
    viewer.view.mousePressEvent(
        _mouse(QEvent.Type.MouseButtonPress, button=Qt.MouseButton.RightButton)
    )
    requested = []
    viewer.opening_proportions_requested.connect(requested.append)
    with patch.object(
        viewer.view, "build_camera_ray", return_value=_button_ray(viewer)
    ):
        viewer.view.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress))
        viewer.view.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease))
    assert requested == [_target().reference]
    assert viewer.is_first_person_active
    assert not viewer.is_first_person_pointer_captured
    assert viewer.get_first_person_camera_pose() == pose


def test_star_hover_highlights_and_restores_existing_cursor(viewer):
    viewer.select_canvas_opening(_target().reference)
    viewer.view.setCursor(Qt.CursorShape.CrossCursor)
    background = viewer._opening_proportions_items[0]
    initial_size = background.size
    with patch.object(
        viewer.view, "build_camera_ray", return_value=_button_ray(viewer)
    ):
        viewer.view.primary_pointer_hovered.emit(QPointF())
    assert viewer._opening_proportions_hovered
    assert background.size > initial_size
    assert viewer.view.cursor().shape() == Qt.CursorShape.PointingHandCursor
    viewer.view.primary_pointer_left.emit()
    assert not viewer._opening_proportions_hovered
    assert viewer.view.cursor().shape() == Qt.CursorShape.CrossCursor


def test_scene_clear_removes_stale_star_items(viewer):
    viewer.select_canvas_opening(_target().reference)
    viewer._clear_scene()
    assert viewer._opening_proportions_items == []
    assert viewer._opening_proportions_center is None


def test_target_replacement_cannot_activate_previous_opening_on_release(viewer):
    viewer.select_canvas_opening(_target().reference)
    requested = []
    viewer.opening_proportions_requested.connect(requested.append)
    with patch.object(
        viewer.view, "build_camera_ray", return_value=_button_ray(viewer)
    ):
        viewer.view.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress))
        other = replace(
            _target(), reference=CanvasOpeningReference("window", 0, 1, "other")
        )
        viewer.set_canvas_opening_targets((other,))
        viewer.view.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease))
    assert requested == []
