"""Solid wall hover remains usable when saved doorway templates are wide."""

# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ### Imports ###
import pytest

from housemaker.doorway_topology import apply_doorway_fit, find_doorway_fit
from housemaker.models import (
    DOORWAY_SHAPE_ARCH,
    DoorwayData,
    DoorwayPreset,
    VertexData,
)


# ### Fixture helpers ###
def _solid_wall(*, second_range=(0.0, 100.0)) -> VertexData:
    graph = VertexData()
    for y, span in ((45.0, (0.0, 100.0)), (55.0, second_range)):
        first, second = (graph.add_vertex(x, y) for x in span)
        graph.add_edge(first.id, second.id)
    return graph


def _wide_template() -> DoorwayData:
    return DoorwayData(
        50.0,
        45.0,
        3.065839503566073,
        2.4,
        shape=DOORWAY_SHAPE_ARCH,
        arch_amount=0.625,
    )


def _assert_four_attached_corners(graph, doorway) -> None:
    corners = doorway.corner_vertex_ids
    assert corners is not None
    assert len(set(corners)) == 4
    assert graph.has_edge(corners[0], corners[2])
    assert graph.has_edge(corners[1], corners[3])
    for first in (corners[0], corners[2]):
        for second in (corners[1], corners[3]):
            assert not graph.has_edge(first, second)
    for vertex_id in corners:
        assert (
            sum(
                vertex_id in (edge.start_vertex_id, edge.end_vertex_id)
                for edge in graph.edges
            )
            == 2
        )


# ### Solid-wall width fitting ###
@pytest.mark.parametrize("second_range", [(0.0, 100.0), (20.0, 80.0)])
def test_wide_template_contracts_to_shared_solid_wall_span(second_range) -> None:
    graph = _solid_wall(second_range=second_range)
    before = graph.clone()
    template = _wide_template()

    fit = find_doorway_fit(graph, (50.0, 45.0), template, 8.0)

    assert fit is not None
    assert fit.corner_vertex_ids is None
    assert graph == before
    assert template.width_meters == pytest.approx(3.065839503566073)
    span_width = (second_range[1] - second_range[0]) * 0.02
    assert span_width - 0.06 < fit.doorway.width_meters < span_width
    assert fit.doorway.height_meters == template.height_meters
    assert fit.doorway.shape == template.shape
    assert fit.doorway.arch_amount == template.arch_amount
    assert fit.doorway.center_x == pytest.approx(50.0)
    assert fit.doorway.center_y == pytest.approx(50.0)
    assert fit.doorway.depth_meters == pytest.approx(0.2)

    doorway = apply_doorway_fit(graph, fit)
    _assert_four_attached_corners(graph, doorway)


def test_solid_span_too_small_for_minimum_doorway_remains_invalid() -> None:
    graph = _solid_wall(second_range=(48.0, 52.0))
    before = graph.clone()

    assert find_doorway_fit(graph, (50.0, 45.0), _wide_template(), 8.0) is None
    assert graph == before


def test_contraction_does_not_reinterpret_opposite_jambs_as_solid_wall_faces() -> None:
    graph = VertexData()
    for x in (30.0, 70.0):
        first, second = graph.add_vertex(x, 45.0), graph.add_vertex(x, 55.0)
        graph.add_edge(first.id, second.id)
    before = graph.clone()

    assert find_doorway_fit(graph, (50.0, 50.0), _wide_template(), 8.0) is None
    assert graph == before


def test_an_explicit_narrow_deep_opening_still_fits_without_contraction() -> None:
    graph = _solid_wall(second_range=(40.0, 60.0))
    template = DoorwayData(50.0, 45.0, 0.12, 2.1)

    fit = find_doorway_fit(graph, (50.0, 50.0), template, 8.0)

    assert fit is not None
    assert fit.doorway.width_meters == pytest.approx(0.12)
    assert fit.doorway.depth_meters == pytest.approx(0.2)


def test_gap_only_fitting_does_not_cut_a_solid_wall() -> None:
    graph = _solid_wall()
    before = graph.clone()

    assert (
        find_doorway_fit(
            graph,
            (50.0, 45.0),
            _wide_template(),
            8.0,
            allow_solid_wall_fit=False,
        )
        is None
    )
    assert graph == before
    assert find_doorway_fit(graph, (50.0, 45.0), _wide_template(), 8.0) is not None


def test_gap_only_fitting_keeps_existing_four_corner_openings() -> None:
    graph = _solid_wall()
    fit = find_doorway_fit(graph, (50.0, 45.0), _wide_template(), 8.0)
    placed = apply_doorway_fit(graph, fit)

    existing = find_doorway_fit(
        graph,
        (50.0, 50.0),
        _wide_template(),
        8.0,
        allow_solid_wall_fit=False,
    )

    assert existing is not None
    assert set(existing.corner_vertex_ids) == set(placed.corner_vertex_ids)


# ### Real canvas hover regression ###
def test_mouse_hover_shows_a_wide_template_preview_on_solid_wall() -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from housemaker.blueprint_canvas import BlueprintCanvas

    application = QApplication.instance() or QApplication([])
    canvas = BlueprintCanvas()
    canvas.resize(640, 520)
    graph = _solid_wall()
    canvas.set_level_data(graph, [], image_path=None, doorways=[])
    canvas.blueprint_image = QImage(100, 100, QImage.Format.Format_RGB32)
    canvas.blueprint_image.fill(Qt.GlobalColor.white)
    canvas.show()
    application.processEvents()
    before = graph.clone()
    template = _wide_template()
    try:
        canvas.start_doorway_placement(
            DoorwayPreset(
                template.width_meters,
                template.height_meters,
                template.shape,
                template.arch_amount,
            )
        )
        QTest.mouseMove(canvas, canvas._image_to_widget(50.0, 45.0).toPoint())
        application.processEvents()

        assert canvas.pending_doorway is not None
        assert canvas.pending_doorway.width_meters < 2.0
        assert canvas.pending_doorway.depth_meters == pytest.approx(0.2)
        assert graph == before
        assert not canvas.doorways
        assert not canvas.undo_stack

        QTest.mouseClick(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=canvas._image_to_widget(50.0, 45.0).toPoint(),
        )
        application.processEvents()
        assert len(canvas.doorways) == 1
        _assert_four_attached_corners(graph, canvas.doorways[0])
    finally:
        canvas.close()
        canvas.deleteLater()
        application.processEvents()
