"""The landing generation asterisk remains visible and clickable from any view."""

# ### Environment setup ###
from __future__ import annotations

import os
from unittest.mock import PropertyMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ### Imports ###
import numpy as np
import pyqtgraph.opengl as gl
import pytest
from OpenGL import GL
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QVector3D
from PySide6.QtWidgets import QApplication

from housemaker.glb import convert_to_glb
from housemaker.models import LandingData, LevelData
from housemaker.viewer import CANVAS_OPENING_OVERLAY_DEPTH_VALUE, GlbViewerWidget

# ### Fixtures and camera helpers ###
_application = QApplication.instance() or QApplication([])
_application.setQuitOnLastWindowClosed(False)


@pytest.fixture
def selected_landing():
    level = LevelData(index=2, name="Ground")
    landing = LandingData(center_x=100.0, center_y=100.0)
    level.landings = [landing]
    viewer = GlbViewerWidget(window_editing_enabled=True)
    viewer.set_model(convert_to_glb([level]))
    viewer.set_landing_context([level])
    assert viewer.select_landing(landing.landing_id)
    yield viewer, landing
    viewer.close()
    viewer.deleteLater()
    _application.processEvents()


def _camera_basis(viewer):
    inverse, valid = viewer.view.viewMatrix().inverted()
    assert valid
    vectors = [
        inverse.mapVector(QVector3D(*point))
        for point in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    ]
    return tuple(
        np.asarray((vector.x(), vector.y(), vector.z()), dtype=float)
        for vector in vectors
    )


def _button_items(viewer):
    items = viewer._landing_generation_items
    assert not any(isinstance(item, gl.GLTextItem) for item in items)
    background = next(item for item in items if isinstance(item, gl.GLScatterPlotItem))
    icon = next(item for item in items if isinstance(item, gl.GLLinePlotItem))
    return background, icon


def _hover_landing_button(viewer, *, hit=True):
    center = viewer._landing_generation_center.copy()
    right, _up, normal = _camera_basis(viewer)
    if not hit:
        center += right * viewer._landing_generation_hit_radius * 3.0
    ray = center + normal * 3.0, -normal
    with patch.object(viewer.view, "build_camera_ray", return_value=ray):
        viewer.view.primary_pointer_hovered.emit(QPointF(10.0, 10.0))


# ### Visible vector control and camera-facing geometry ###
@pytest.mark.parametrize("elevation, azimuth", [(25.0, 35.0), (90.0, 0.0), (0.0, 90.0)])
def test_landing_asterisk_has_large_contrasting_overlay_and_camera_facing_strokes(
    selected_landing, elevation, azimuth
):
    viewer, landing = selected_landing
    viewer.view.setCameraPosition(elevation=elevation, azimuth=azimuth)
    viewer._sync_landing_generation_control()
    background, icon = _button_items(viewer)
    center = viewer._landing_generation_center
    assert center is not None
    assert background in viewer.view.items
    assert icon in viewer.view.items
    assert background.pxMode
    assert background.size >= 24.0
    assert icon.width >= 2.0
    assert not np.allclose(background.color, icon.color)
    np.testing.assert_allclose(background.pos[0], center, atol=1e-6)
    assert icon.mode == "lines"
    segments = np.asarray(icon.pos, dtype=float).reshape(-1, 2, 3)
    assert len(segments) >= 3
    np.testing.assert_allclose(
        segments.mean(axis=1), np.tile(center, (len(segments), 1)), atol=1e-6
    )
    _right, _up, normal = _camera_basis(viewer)
    offsets = np.asarray(icon.pos, dtype=float) - center
    np.testing.assert_allclose(offsets @ normal, 0.0, atol=1e-6)
    assert np.linalg.matrix_rank(offsets, tol=1e-6) == 2
    for item in (background, icon):
        assert item.depthValue() >= CANVAS_OPENING_OVERLAY_DEPTH_VALUE
        assert item._GLGraphicsItem__glOpts[GL.GL_DEPTH_TEST] is False
    group = viewer._placed_object_render_groups[f"landing:{landing.landing_id}"]
    pivot = np.asarray(group.preview.world_position, dtype=float)
    right, up, _normal = _camera_basis(viewer)
    screen_offset = np.asarray(
        (np.dot(center - pivot, right), np.dot(center - pivot, up))
    )
    assert np.linalg.norm(screen_offset) > 0.0


def test_landing_asterisk_size_and_offset_remain_constant_in_pixels_on_zoom(
    selected_landing,
):
    viewer, landing = selected_landing
    group = viewer._placed_object_render_groups[f"landing:{landing.landing_id}"]
    pivot = np.asarray(group.preview.world_position, dtype=float)
    samples = []
    for pixel_size in (0.002, 0.02):
        with patch.object(viewer.view, "pixelSize", return_value=pixel_size):
            viewer._sync_landing_generation_control()
        background, icon = _button_items(viewer)
        center = viewer._landing_generation_center.copy()
        samples.append(
            (
                background.size,
                (np.asarray(icon.pos, dtype=float) - center) / pixel_size,
                (center - pivot) / pixel_size,
            )
        )
    assert samples[0][0] == samples[1][0]
    np.testing.assert_allclose(samples[0][1], samples[1][1], atol=1e-3)
    np.testing.assert_allclose(samples[0][2], samples[1][2], atol=1e-3)


def test_landing_asterisk_tracks_orbit_without_rebuilding_items(selected_landing):
    viewer, _landing = selected_landing
    original_items = tuple(viewer._landing_generation_items)
    old_center = viewer._landing_generation_center.copy()
    old_strokes = _button_items(viewer)[1].pos.copy()
    viewer.view.setCameraPosition(elevation=90.0, azimuth=0.0)
    viewer.view.frame_about_to_render.emit()
    assert tuple(viewer._landing_generation_items) == original_items
    assert not np.allclose(viewer._landing_generation_center, old_center)
    assert not np.allclose(_button_items(viewer)[1].pos, old_strokes)
    right, up, normal = _camera_basis(viewer)
    center = viewer._landing_generation_center
    np.testing.assert_allclose(
        (_button_items(viewer)[1].pos - center) @ normal, 0.0, atol=1e-6
    )
    assert (
        abs(np.dot(center - old_center, right)) + abs(np.dot(center - old_center, up))
        > 0.0
    )


def test_landing_asterisk_unchanged_frame_does_not_schedule_another_repaint(
    selected_landing,
):
    viewer, _landing = selected_landing
    background, icon = _button_items(viewer)
    with (
        patch.object(
            background, "setData", wraps=background.setData
        ) as background_update,
        patch.object(icon, "setData", wraps=icon.setData) as icon_update,
    ):
        viewer.view.frame_about_to_render.emit()
        viewer.view.frame_about_to_render.emit()
    background_update.assert_not_called()
    icon_update.assert_not_called()


# ### Screen-space click target and lifecycle ###
def test_landing_asterisk_click_target_covers_visible_button_after_zoom(
    selected_landing,
):
    viewer, _landing = selected_landing
    for pixel_size in (0.002, 0.02):
        with patch.object(viewer.view, "pixelSize", return_value=pixel_size):
            viewer._sync_landing_generation_control()
            background, _icon = _button_items(viewer)
            center = viewer._landing_generation_center.copy()
            right, _up, normal = _camera_basis(viewer)
            visual_radius = background.size * pixel_size * 0.5
            inside = center + right * visual_radius * 0.9
            outside = center + right * visual_radius * 2.0
            assert viewer._pick_landing_generation_control(
                (inside + normal * 3.0, -normal)
            )
            assert not viewer._pick_landing_generation_control(
                (outside + normal * 3.0, -normal)
            )
            assert not viewer._pick_landing_generation_control(
                (center + normal * 3.0, normal)
            )


def test_landing_asterisk_survives_gizmo_mode_switch_and_is_removed_on_deselect(
    selected_landing,
):
    viewer, _landing = selected_landing
    assert viewer._toggle_placed_object_gizmo_mode()
    background, icon = _button_items(viewer)
    assert background in viewer.view.items
    assert icon in viewer.view.items
    previous_items = tuple(viewer._landing_generation_items)
    assert viewer.select_landing(None)
    assert viewer._landing_generation_center is None
    assert not viewer._landing_generation_items
    assert all(item not in viewer.view.items for item in previous_items)
    viewer.view.frame_about_to_render.emit()
    assert not viewer._landing_generation_items


# ### Lightweight hover feedback and cursor ownership ###
def test_landing_asterisk_hover_highlights_without_rebuilding_geometry(
    selected_landing,
):
    viewer, _landing = selected_landing
    viewer.view.setCursor(Qt.CursorShape.ArrowCursor)
    items = tuple(viewer._landing_generation_items)
    background, icon = _button_items(viewer)
    initial_color = np.asarray(background.color).copy()
    initial_size = background.size
    initial_width = icon.width
    initial_positions = np.asarray(icon.pos).copy()

    with patch.object(viewer, "_sync_landing_generation_control") as rebuild:
        _hover_landing_button(viewer)

    assert viewer._landing_generation_hovered
    assert tuple(viewer._landing_generation_items) == items
    assert not np.allclose(background.color, initial_color)
    assert background.size > initial_size
    assert icon.width > initial_width
    np.testing.assert_allclose(icon.pos, initial_positions)
    assert viewer.view.cursor().shape() == Qt.CursorShape.PointingHandCursor
    rebuild.assert_not_called()

    _hover_landing_button(viewer, hit=False)

    assert not viewer._landing_generation_hovered
    np.testing.assert_allclose(background.color, initial_color)
    assert background.size == initial_size
    assert icon.width == initial_width
    assert viewer.view.cursor().shape() == Qt.CursorShape.ArrowCursor


def test_landing_asterisk_repeated_hover_does_not_schedule_repaints(
    selected_landing,
):
    viewer, _landing = selected_landing
    _hover_landing_button(viewer)
    background, icon = _button_items(viewer)
    with (
        patch.object(background, "setData", wraps=background.setData) as backdrop_update,
        patch.object(icon, "setData", wraps=icon.setData) as icon_update,
    ):
        _hover_landing_button(viewer)
        _hover_landing_button(viewer)
    backdrop_update.assert_not_called()
    icon_update.assert_not_called()


def test_landing_asterisk_pointer_exit_restores_previous_cursor(selected_landing):
    viewer, _landing = selected_landing
    viewer.view.setCursor(Qt.CursorShape.CrossCursor)
    _hover_landing_button(viewer)
    assert viewer.view.cursor().shape() == Qt.CursorShape.PointingHandCursor

    viewer.view.primary_pointer_left.emit()

    assert not viewer._landing_generation_hovered
    assert viewer.view.cursor().shape() == Qt.CursorShape.CrossCursor


def test_landing_asterisk_pointer_exit_restores_inherited_cursor(selected_landing):
    viewer, _landing = selected_landing
    viewer.setCursor(Qt.CursorShape.CrossCursor)
    viewer.view.unsetCursor()
    assert not viewer.view.testAttribute(Qt.WidgetAttribute.WA_SetCursor)

    _hover_landing_button(viewer)
    assert viewer.view.testAttribute(Qt.WidgetAttribute.WA_SetCursor)
    viewer.view.primary_pointer_left.emit()

    assert not viewer._landing_generation_hovered
    assert not viewer.view.testAttribute(Qt.WidgetAttribute.WA_SetCursor)
    assert viewer.view.cursor().shape() == Qt.CursorShape.CrossCursor


def test_landing_asterisk_pointer_exit_does_not_override_new_tool_cursor(
    selected_landing,
):
    viewer, _landing = selected_landing
    _hover_landing_button(viewer)
    viewer.view.setCursor(Qt.CursorShape.CrossCursor)

    viewer.view.primary_pointer_left.emit()

    assert not viewer._landing_generation_hovered
    assert viewer.view.cursor().shape() == Qt.CursorShape.CrossCursor


def test_landing_asterisk_deselection_clears_hover_feedback(selected_landing):
    viewer, _landing = selected_landing
    viewer.view.setCursor(Qt.CursorShape.ArrowCursor)
    _hover_landing_button(viewer)

    assert viewer.select_landing(None)

    assert not viewer._landing_generation_hovered
    assert not viewer._landing_generation_items
    assert viewer.view.cursor().shape() == Qt.CursorShape.ArrowCursor


def test_landing_asterisk_does_not_hover_during_landing_placement(selected_landing):
    viewer, _landing = selected_landing
    assert viewer.begin_landing_placement()
    _hover_landing_button(viewer)

    assert not viewer._landing_generation_hovered
    assert viewer.view.cursor().shape() == Qt.CursorShape.CrossCursor


def test_landing_asterisk_does_not_hover_during_captured_first_person_navigation(
    selected_landing,
):
    viewer, _landing = selected_landing
    with patch.object(
        type(viewer.view),
        "is_first_person_pointer_captured",
        new_callable=PropertyMock,
        return_value=True,
    ):
        _hover_landing_button(viewer)

    assert not viewer._landing_generation_hovered
    assert viewer.view.cursor().shape() != Qt.CursorShape.PointingHandCursor


def test_landing_asterisk_does_not_hover_during_reserved_gizmo_drag(selected_landing):
    viewer, _landing = selected_landing
    viewer.view.reserve_primary_pointer_drag()
    try:
        _hover_landing_button(viewer)
        assert not viewer._landing_generation_hovered
        assert viewer.view.cursor().shape() != Qt.CursorShape.PointingHandCursor
    finally:
        viewer.view.cancel_primary_pointer_drag()


def test_landing_asterisk_capture_change_clears_hover_but_keeps_blank_cursor(
    selected_landing,
):
    viewer, _landing = selected_landing
    _hover_landing_button(viewer)
    assert viewer._landing_generation_hovered
    viewer.view.setCursor(Qt.CursorShape.BlankCursor)

    viewer.view.first_person_pointer_capture_changed.emit(True)

    assert not viewer._landing_generation_hovered
    assert viewer.view.cursor().shape() == Qt.CursorShape.BlankCursor
