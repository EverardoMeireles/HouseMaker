"""Landing placement, polygon selection, structural snapping and scene gizmos."""

# ### Environment ###
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
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.blueprint_canvas import BlueprintCanvas
from housemaker.glb import PreviewPlacedObject, convert_to_glb
from housemaker.landing_geometry import build_landing_mesh, landing_image_footprint
from housemaker.models import LandingData, LevelData
from housemaker.viewer import (
    PLACED_OBJECT_GIZMO_SCALE,
    TRANSFORM_GIZMO_ROTATE,
    TRANSFORM_GIZMO_SCALE,
    TRANSFORM_GIZMO_TRANSLATE,
    GlbViewerWidget,
    _TransformGizmoHandle,
)

# ### Fixtures ###
_application = QApplication.instance() or QApplication([])
_application.setQuitOnLastWindowClosed(False)


@pytest.fixture
def authoring_views():
    widgets = []

    def build():
        level = LevelData(index=2, name="Ground")
        landing = LandingData(
            center_x=100.0,
            center_y=100.0,
            footprint_meters=(
                (-0.8, -0.8),
                (0.8, -0.8),
                (0.8, 0.0),
                (0.0, 0.0),
                (0.0, 0.8),
                (-0.8, 0.8),
            ),
        )
        level.landings.append(landing)
        canvas = BlueprintCanvas()
        canvas.resize(600, 600)
        canvas.set_level_data(
            level.vertex_data, level.rooms, None, level.doorways, windows=level.windows
        )
        canvas.blueprint_image = QImage(200, 200, QImage.Format.Format_RGB32)
        canvas.blueprint_image.fill(Qt.GlobalColor.white)
        canvas.set_stair_context((), level)
        viewer = GlbViewerWidget(window_editing_enabled=True)
        viewer.set_model(convert_to_glb([level]))
        viewer.set_landing_context([level])
        widgets.extend((canvas, viewer))
        return canvas, viewer, level, landing

    yield build
    for widget in widgets:
        widget.close()
        widget.deleteLater()
    _application.processEvents()


# ### Canvas placement, polygons and attachments ###
def test_canvas_landing_placement_is_exclusive_one_shot_and_cancellable(
    authoring_views,
):
    canvas, _, _, _ = authoring_views()
    placed = []
    canvas.landing_placement_requested.connect(lambda x, y: placed.append((x, y)))
    assert canvas.begin_landing_placement()
    QTest.mouseClick(
        canvas,
        Qt.MouseButton.LeftButton,
        pos=canvas._image_to_widget(140.0, 150.0).toPoint(),
    )
    assert placed[0] == pytest.approx((140.0, 150.0), abs=0.5)
    assert not canvas.is_landing_placement_active()
    assert canvas.begin_landing_placement()
    canvas.start_window_placement()
    assert not canvas.is_landing_placement_active()
    canvas.cancel_window_placement()
    assert canvas.begin_landing_placement()
    QTest.keyClick(canvas, Qt.Key.Key_Escape)
    assert not canvas.is_landing_placement_active()
    assert len(placed) == 1


def test_canvas_uses_nonrectangular_landing_polygon_and_emits_live_edits(
    authoring_views,
):
    canvas, _, level, landing = authoring_views()
    polygon = canvas._landing_widget_polygon(landing)
    assert len(polygon) == 6
    assert not polygon.containsPoint(
        canvas._image_to_widget(130.0, 130.0), Qt.FillRule.OddEvenFill
    )
    assert canvas._begin_landing_pointer_drag(canvas._image_to_widget(80.0, 100.0))
    edits, finished = [], []
    canvas.landing_transform_preview_changed.connect(lambda *args: edits.append(args))
    canvas.landing_edit_finished.connect(lambda *args: finished.append(args))
    canvas._update_landing_pointer_drag(canvas._image_to_widget(95.0, 110.0))
    assert edits[-1][0] == landing.landing_id
    assert edits[-1][1]["center_x"] == pytest.approx(115.0)
    assert edits[-1][1]["center_y"] == pytest.approx(110.0)
    assert level.landings == [landing]
    canvas._finish_landing_drag(canvas._image_to_widget(95.0, 110.0))
    assert finished == [(landing.landing_id, True)]


def test_landing_rotation_and_scale_gizmos_emit_canonical_values(authoring_views):
    _, viewer, _, landing = authoring_views()
    viewer.select_landing(landing.landing_id)
    assert len(viewer._transform_gizmo_items) == 7
    group = viewer._placed_object_render_groups[f"landing:{landing.landing_id}"]
    pivot = np.asarray(group.preview.world_position)
    edits = []
    viewer.landing_transform_preview_changed.connect(lambda *args: edits.append(args))
    with patch.object(
        viewer.view,
        "build_camera_ray",
        return_value=(pivot + (1.0, 0.0, 3.0), np.asarray((0.0, 0.0, -1.0))),
    ):
        assert viewer._begin_placed_object_gizmo_drag(
            _TransformGizmoHandle(TRANSFORM_GIZMO_ROTATE, 2), QPointF()
        )
    with patch.object(
        viewer.view,
        "build_camera_ray",
        return_value=(pivot + (0.0, 1.0, 3.0), np.asarray((0.0, 0.0, -1.0))),
    ):
        assert viewer._update_placed_object_gizmo_drag(QPointF())
        assert viewer._finish_placed_object_gizmo_drag(QPointF())
    assert edits[-1][1]["rotation_degrees"] == pytest.approx(-90.0)
    viewer._toggle_placed_object_gizmo_mode()
    assert len(viewer._transform_gizmo_items) == 7
    direction = np.asarray((0.0, 5.0, -2.0))
    direction /= np.linalg.norm(direction)
    with patch.object(
        viewer.view,
        "build_camera_ray",
        return_value=(pivot + (0.0, -5.0, 2.0), direction),
    ):
        assert viewer._begin_placed_object_gizmo_drag(
            _TransformGizmoHandle(TRANSFORM_GIZMO_SCALE, 2), QPointF()
        )
    with patch.object(
        viewer.view,
        "build_camera_ray",
        return_value=(pivot + (0.0, -5.0, 2.3), direction),
    ):
        assert viewer._update_placed_object_gizmo_drag(QPointF())
        assert viewer._finish_placed_object_gizmo_drag(QPointF())
    assert edits[-1][1]["scale_z"] > 1.0
    assert edits[-1][1]["scale_x"] == pytest.approx(1.0)
    assert edits[-1][1]["scale_y"] == pytest.approx(1.0)


def test_landing_selection_does_not_emit_object_events_or_clear_normal_selection(
    authoring_views,
):
    _, viewer, level, landing = authoring_views()
    normal = PreviewPlacedObject(
        "normal-chair",
        (trimesh.creation.box(),),
        np.eye(4),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    viewer.model.preview_placed_objects.append(normal)
    viewer._add_placed_object_items()
    selected, sets = [], []
    viewer.placed_object_selection_changed.connect(selected.append)
    viewer.placed_object_selection_set_changed.connect(sets.append)
    viewer.select_landing(landing.landing_id)
    assert selected == sets == []
    viewer.select_landing(None)
    assert selected == sets == []
    viewer.select_landing(landing.landing_id)
    assert viewer.select_placed_object("normal-chair")
    assert selected == ["normal-chair"]
    assert sets == [("normal-chair",)]
    viewer.set_landing_context([level], None)
    assert viewer._selected_placed_object_id == "normal-chair"


def test_landing_proxy_matches_core_geometry_after_scaled_rotation_and_model_reset(
    authoring_views,
):
    _, viewer, level, landing = authoring_views()
    level.scale = 1.5
    landing = replace(
        landing,
        rotation_degrees=35.0,
        scale_x=1.8,
        scale_y=0.6,
        scale_z=1.4,
        elevation_meters=0.2,
    )
    level.landings[:] = [landing]
    viewer.set_landing_context([level], landing.landing_id)
    viewer.set_model(convert_to_glb([level]), preserve_camera=True)
    viewer.set_landing_context([level], landing.landing_id)
    group = viewer._placed_object_render_groups[f"landing:{landing.landing_id}"]
    proxy_mesh = group.preview.meshes[0].copy()
    proxy_mesh.apply_transform(group.current_transform)
    core_mesh = build_landing_mesh(level, landing, [level])
    assert np.allclose(proxy_mesh.vertices, core_mesh.vertices)
    payload = viewer._landing_transform_payload(
        group.preview.object_id,
        group.preview.world_position,
        group.preview.rotation_degrees,
        group.preview.axis_scales,
    )
    for key in (
        "center_x",
        "center_y",
        "rotation_degrees",
        "elevation_meters",
        "scale_x",
        "scale_y",
        "scale_z",
    ):
        assert payload[key] == pytest.approx(getattr(landing, key))
    assert len(viewer._landing_previews) == 1
    assert len(viewer._placed_object_render_groups) == 1
    assert not viewer.model.preview_placed_objects
    assert viewer._landing_source_geometry
    display = viewer._get_display_mesh()
    assert display is not None
    # Landing-supported floors and ceilings remain; only the original static
    # landing is replaced by its independent editable proxy.
    original = viewer._get_unfiltered_display_mesh()
    assert len(display.faces) == len(original.faces) - len(core_mesh.faces)
    landing_triangles = {
        tuple(sorted(tuple(np.round(point, 7)) for point in triangle))
        for triangle in core_mesh.triangles
    }
    assert not any(
        tuple(sorted(tuple(np.round(point, 7)) for point in triangle))
        in landing_triangles
        for triangle in display.triangles
    )


def test_viewer_landing_placement_uses_floor_plane_and_excludes_other_modes(
    authoring_views,
):
    _, viewer, _, _ = authoring_views()
    placed = []
    viewer.landing_placement_requested.connect(placed.append)
    assert viewer.begin_directional_light_placement(fallback_plane_z=1.25)
    assert viewer.begin_landing_placement(fallback_plane_z=1.25)
    assert not viewer.is_directional_light_placement_active
    ray = np.asarray((0.4, 0.5, 5.0)), np.asarray((0.0, 0.0, -1.0))
    with patch.object(viewer.view, "build_camera_ray", return_value=ray):
        viewer._handle_placed_object_pointer_pressed(QPointF())
        assert viewer._landing_placement_item is not None
        viewer._handle_canvas_gizmo_pointer_released(QPointF())
    assert placed == [(0.4, 0.5, 1.25)]
    assert not viewer.is_landing_placement_active()
    assert viewer.begin_landing_placement()
    assert viewer.begin_directional_light_placement(fallback_plane_z=0.0)
    assert not viewer.is_landing_placement_active()


def test_stair_point_snaps_to_landing_polygon_edge_with_stable_local_anchor(
    authoring_views,
):
    canvas, _, level, landing = authoring_views()
    corner = landing_image_footprint(level, landing)[0]
    raw = QPointF(corner[0] + 0.5, corner[1] - 0.5)
    target = canvas._build_stair_point_target(
        raw, Qt.KeyboardModifier.NoModifier, canvas._image_to_widget(raw.x(), raw.y())
    )
    assert target.point == pytest.approx(corner)
    assert target.landing_attachment is not None
    assert target.landing_attachment.landing_id == landing.landing_id
    assert target.landing_attachment.local_x_meters == pytest.approx(-0.8)
    assert target.landing_attachment.local_y_meters == pytest.approx(-0.8)


# ### Viewer proxies, picking, generation and transforms ###
def test_scene_landing_proxy_is_selectable_and_not_added_to_export_records(
    authoring_views,
):
    _, viewer, _, landing = authoring_views()
    object_id = f"landing:{landing.landing_id}"
    assert object_id in viewer._placed_object_render_groups
    assert all(
        item.object_id != object_id for item in viewer.model.preview_placed_objects
    )
    group = viewer._placed_object_render_groups[object_id]
    point = np.asarray(group.preview.world_position) + np.asarray((-0.4, 0.0, 3.0))
    ray = point, np.asarray((0.0, 0.0, -1.0))
    with patch.object(viewer.view, "build_camera_ray", return_value=ray):
        viewer._handle_window_wall_pick_requested(QPointF(10.0, 10.0))
    assert viewer._selected_landing_id == landing.landing_id
    assert viewer._landing_generation_center is not None
    assert not viewer._placed_object_instance_gizmo_items
    with patch.object(viewer.view, "build_camera_ray", return_value=ray):
        viewer._handle_window_wall_pick_requested(QPointF(10.0, 10.0))
    assert viewer.get_placed_object_gizmo_mode() == PLACED_OBJECT_GIZMO_SCALE


def test_landing_star_control_emits_generation_request(authoring_views):
    _, viewer, _, landing = authoring_views()
    assert viewer.select_landing(landing.landing_id)
    center = viewer._landing_generation_center
    assert center is not None
    ray = center + (0.0, -3.0, 0.0), np.asarray((0.0, 1.0, 0.0))
    generated = []
    viewer.landing_generation_requested.connect(generated.append)
    with patch.object(viewer.view, "build_camera_ray", return_value=ray):
        viewer._handle_placed_object_pointer_pressed(QPointF())
        viewer._handle_canvas_gizmo_pointer_released(QPointF())
    assert generated == [landing.landing_id]


def test_landing_translation_gizmo_emits_image_coordinates_and_finishes(
    authoring_views,
):
    _, viewer, _, landing = authoring_views()
    assert viewer.select_landing(landing.landing_id)
    group = viewer._placed_object_render_groups[f"landing:{landing.landing_id}"]
    pivot = np.asarray(group.preview.world_position)
    direction = np.asarray((0.0, 5.0, -2.0))
    direction /= np.linalg.norm(direction)
    start_ray = pivot + (0.0, -5.0, 2.0), direction
    next_ray = pivot + (0.4, -5.0, 2.0), direction
    edits, finished = [], []
    viewer.landing_transform_preview_changed.connect(lambda *args: edits.append(args))
    viewer.landing_edit_finished.connect(lambda *args: finished.append(args))
    with patch.object(viewer.view, "build_camera_ray", return_value=start_ray):
        assert viewer._begin_placed_object_gizmo_drag(
            _TransformGizmoHandle(TRANSFORM_GIZMO_TRANSLATE, 0), QPointF()
        )
    with patch.object(viewer.view, "build_camera_ray", return_value=next_ray):
        assert viewer._update_placed_object_gizmo_drag(QPointF())
        assert viewer._finish_placed_object_gizmo_drag(QPointF())
    assert edits[-1][0] == landing.landing_id
    assert edits[-1][1]["center_x"] == pytest.approx(landing.center_x + 20.0)
    assert edits[-1][1]["center_y"] == pytest.approx(landing.center_y)
    assert finished == [(landing.landing_id, True)]
