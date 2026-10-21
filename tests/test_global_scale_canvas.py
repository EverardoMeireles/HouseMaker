# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import pytest
from PySide6.QtCore import QPointF
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from housemaker.blueprint_canvas import BlueprintCanvas
from housemaker.camera_models import CameraPose
from housemaker.global_scale import GlobalScale
from housemaker.models import LevelData, VertexData

# ### Qt fixture helpers ###
_application = QApplication.instance() or QApplication([])
_application.setQuitOnLastWindowClosed(False)


@pytest.fixture
def canvas(tmp_path):
    image = QImage(160, 100, QImage.Format.Format_RGB32)
    image.fill(0xFFFFFFFF)
    path = tmp_path / "plan.png"
    assert image.save(str(path))
    vertices = VertexData()
    first = vertices.add_vertex(20.0, 25.0)
    second = vertices.add_vertex(130.0, 75.0)
    vertices.add_edge(first.id, second.id)
    widget = BlueprintCanvas()
    widget.resize(720, 480)
    widget.load_blueprint(str(path), vertex_data=vertices)
    yield widget
    widget.close()
    widget.deleteLater()
    _application.processEvents()


# ### Display and editing coordinate tests ###
def test_nonuniform_scale_preserves_editable_coordinates_and_pan_zoom(canvas) -> None:
    canvas.set_canvas_level_scale(1.25)
    canvas.zoom_scale = 1.5
    canvas.view_offset = QPointF(24.0, -17.0)
    source_geometry = canvas.vertex_data.to_dict()
    image_cache_key = canvas.blueprint_image.cacheKey()
    before = canvas._image_display_rect()
    scale = GlobalScale(x=2.0, y=0.5, z=3.0, linked=False)

    assert canvas.set_global_scale(scale)
    after = canvas._image_display_rect()
    restored = canvas._widget_to_image(canvas._image_to_widget(60.0, 35.0))

    assert after.width() == pytest.approx(before.width() * 2.0)
    assert after.height() == pytest.approx(before.height() * 0.5)
    assert restored.x() == pytest.approx(60.0)
    assert restored.y() == pytest.approx(35.0)
    assert canvas.vertex_data.to_dict() == source_geometry
    assert canvas.blueprint_image.cacheKey() == image_cache_key
    assert canvas.canvas_level_scale == 1.25
    assert canvas.zoom_scale == 1.5
    assert canvas.view_offset == QPointF(24.0, -17.0)
    assert canvas.get_global_scale() == scale
    assert not canvas.set_global_scale(scale)


def test_scaling_plan_and_comparison_offsets_preserves_registration(canvas) -> None:
    canvas.set_canvas_level_offsets(20.0, -30.0)
    comparison = LevelData(
        index=3,
        name="Upper",
        image_path=canvas.blueprint_path,
        canvas_level_scale=0.75,
        canvas_offset_x_pixels=-10.0,
        canvas_offset_y_pixels=15.0,
    )
    assert canvas.set_level_comparison_overlay(comparison)
    overlay = canvas.get_level_comparison_overlay()
    canvas.zoom_scale = 2.0
    canvas.view_offset = QPointF(17.0, 8.0)
    before_current = canvas._image_display_rect()
    before_overlay = canvas._comparison_image_display_rect(overlay)

    canvas.set_global_scale(GlobalScale(x=1.5, y=0.5, linked=False))
    after_current = canvas._image_display_rect()
    after_overlay = canvas._comparison_image_display_rect(overlay)

    assert after_overlay.width() == pytest.approx(before_overlay.width() * 1.5)
    assert after_overlay.height() == pytest.approx(before_overlay.height() * 0.5)
    before_delta = before_overlay.center() - before_current.center()
    after_delta = after_overlay.center() - after_current.center()
    assert after_delta.x() == pytest.approx(before_delta.x() * 1.5)
    assert after_delta.y() == pytest.approx(before_delta.y() * 0.5)
    assert canvas.canvas_offset_x_pixels == 20.0
    assert canvas.canvas_offset_y_pixels == -30.0


def test_scaled_vertex_and_edge_picking_use_the_displayed_geometry(canvas) -> None:
    canvas.set_global_scale(GlobalScale(x=2.0, y=0.5, linked=False))
    first = canvas.vertex_data.vertices[0]
    displayed_vertex = canvas._image_to_widget(first.x, first.y)
    displayed_midpoint = canvas._image_to_widget(75.0, 50.0)

    assert canvas._find_vertex_at(displayed_vertex) == first
    edge_hit = canvas._find_edge_at(displayed_midpoint)

    assert edge_hit is not None
    assert edge_hit.point == pytest.approx((75.0, 50.0))
    assert edge_hit.distance == pytest.approx(0.0)


def test_global_scale_allows_panning_without_extra_zoom(canvas) -> None:
    canvas.set_global_scale(GlobalScale(x=1.0, y=2.0, linked=False))
    canvas._start_panning(QPointF(350.0, 240.0))
    canvas._update_pan(QPointF(375.0, 210.0))

    assert canvas.is_panning
    assert canvas.view_offset == QPointF(25.0, -30.0)


def test_vertical_only_multiplier_does_not_change_the_plan(canvas) -> None:
    before = canvas._image_display_rect()

    canvas.set_global_scale(GlobalScale(z=2.0, linked=False))

    assert canvas._image_display_rect() == before
    canvas._start_panning(QPointF(350.0, 240.0))
    assert not canvas.is_panning


def test_snap_search_distance_covers_the_less_scaled_axis(canvas) -> None:
    canvas.set_global_scale(GlobalScale(x=2.0, y=0.5, linked=False))
    rect = canvas._image_display_rect()
    screen_pixels_per_image_pixel = rect.height() / canvas.blueprint_image.height()

    assert canvas._screen_distance_to_image(10.0) == pytest.approx(
        10.0 / screen_pixels_per_image_pixel
    )


def test_camera_indicator_keeps_physical_heading_when_plan_axes_scale_differently(
    canvas,
) -> None:
    canvas.level_context = LevelData(
        index=2, name="Ground", image_size_pixels=(160.0, 100.0)
    )
    pose = CameraPose(x=0.2, y=0.1, yaw_degrees=30.0)
    canvas.set_camera_indicator_pose(pose)
    center_before, tip_before = canvas._get_camera_indicator_geometry()

    canvas.set_global_scale(GlobalScale(x=3.0, y=0.5, linked=False))
    center_after, tip_after = canvas._get_camera_indicator_geometry()

    before = tip_before - center_before
    after = tip_after - center_after
    assert after.x() == pytest.approx(before.x())
    assert after.y() == pytest.approx(before.y())
    assert center_before != center_after
    assert canvas.get_camera_indicator_pose() == pose
