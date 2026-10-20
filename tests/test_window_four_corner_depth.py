"""Snapped window cuts follow their actual four-corner wall thickness."""

# ### Imports ###
from __future__ import annotations

import math

import numpy as np
import pytest

from housemaker.glb import (
    WINDOW_COPLANAR_DEPTH_PIXELS,
    WINDOW_CUT_DEPTH_METERS,
    _build_level_window_reveals,
    _build_visible_wall_pieces,
    _build_window_openings,
)
from housemaker.models import LevelData, WindowData
from housemaker.surface_geometry import build_fixed_surfaces


# ### Fixture helpers ###
def _build_snapped_window_level(
    *,
    angle: float = 0.0,
    reverse_edges: bool = False,
    depth: float = 60.0,
    scale: float = 1.0,
) -> tuple[LevelData, tuple[float, float], tuple[float, float]]:
    level = LevelData(index=2, name="Ground", scale=scale)
    radians = math.radians(angle)
    width_axis = (math.cos(radians), math.sin(radians))
    depth_axis = (-width_axis[1], width_axis[0])
    for width, offset in (
        (0.0, 0.0),
        (100.0, 0.0),
        (0.0, depth),
        (100.0, depth),
        (-40.0, 0.0),
        (140.0, 0.0),
        (-40.0, depth),
        (140.0, depth),
    ):
        level.vertex_data.add_vertex(
            150.0 + width * width_axis[0] + offset * depth_axis[0],
            150.0 + width * width_axis[1] + offset * depth_axis[1],
        )
    for first, second in (
        (1, 2),
        (3, 4),
        (1, 3),
        (2, 4),
        (5, 1),
        (2, 6),
        (7, 3),
        (4, 8),
    ):
        if reverse_edges:
            first, second = second, first
        level.vertex_data.add_edge(first, second)
    level.windows = [
        WindowData(
            window_id="snapped-frame",
            wall_surface_id="level:2/wall:1:2",
            start_ratio=0.0,
            end_ratio=1.0,
            bottom_ratio=0.25,
            top_ratio=0.75,
        )
    ]
    return level, width_axis, depth_axis


# ### Four-corner depth and reveal regressions ###
@pytest.mark.parametrize("angle", [0.0, 90.0, 27.0])
@pytest.mark.parametrize("reverse_edges", [False, True])
@pytest.mark.parametrize("scale", [1.0, 1.5])
def test_four_corner_cut_reaches_thick_paired_face_for_either_edge_orientation(
    angle: float,
    reverse_edges: bool,
    scale: float,
) -> None:
    level, width_axis, depth_axis = _build_snapped_window_level(
        angle=angle,
        reverse_edges=reverse_edges,
        scale=scale,
    )
    level.flipped_surface_ids.add("level:2/wall:1:2")

    (opening,) = _build_window_openings(level)
    front_midpoint = np.array((150.0, 150.0)) + 50.0 * np.array(width_axis)

    np.testing.assert_allclose(
        (opening.depth_direction_x, opening.depth_direction_y),
        depth_axis,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        (opening.center_x, opening.center_y),
        front_midpoint + 30.0 * np.array(depth_axis),
        atol=1e-9,
    )
    assert opening.half_depth_pixels == pytest.approx(
        30.0 + WINDOW_COPLANAR_DEPTH_PIXELS
    )
    for start_id, end_id in ((1, 2), (3, 4)):
        start = level.vertex_data.get_vertex(start_id)
        end = level.vertex_data.get_vertex(end_id)
        assert start is not None and end is not None
        pieces = _build_visible_wall_pieces(
            (start.x, start.y),
            (end.x, end.y),
            level.height_meters,
            [opening],
            wall_key=f"{start_id}:{end_id}",
        )
        visible_ratio_area = sum(
            abs(
                sum(
                    first[0] * second[1] - second[0] * first[1]
                    for first, second in zip(
                        piece.points, (*piece.points[1:], piece.points[0])
                    )
                )
            )
            * 0.5
            for piece in pieces
        )
        assert visible_ratio_area == pytest.approx(level.height_meters * 0.5)
    reveals = _build_level_window_reveals(level)
    assert len(reveals) == 1
    pair = reveals[0].reveal_pair
    assert (
        pair.second_contact.depth_position - pair.first_contact.depth_position
        == pytest.approx(60.0)
    )


def test_four_corner_cut_does_not_cut_farther_parallel_wall() -> None:
    level, _width_axis, _depth_axis = _build_snapped_window_level(depth=10.0)
    first = level.vertex_data.add_vertex(150.0, 170.0)
    second = level.vertex_data.add_vertex(250.0, 170.0)
    level.vertex_data.add_edge(first.id, second.id)

    surfaces = {
        surface.surface_id: surface for surface in build_fixed_surfaces([level])
    }

    assert surfaces["level:2/wall:3:4"].area_square_meters == pytest.approx(3.0)
    assert surfaces[
        f"level:2/wall:{first.id}:{second.id}"
    ].area_square_meters == pytest.approx(6.0)


# ### Legacy and incomplete-topology safeguards ###
@pytest.mark.parametrize("missing_edge", [(3, 4), (2, 4), (5, 1), (4, 8)])
def test_incomplete_four_corner_graph_keeps_legacy_window_depth(
    missing_edge: tuple[int, int],
) -> None:
    level, _width_axis, _depth_axis = _build_snapped_window_level()
    key = set(missing_edge)
    level.vertex_data.edges = [
        edge
        for edge in level.vertex_data.edges
        if {edge.start_vertex_id, edge.end_vertex_id} != key
    ]

    (opening,) = _build_window_openings(level)

    full_depth = 2.0 * opening.half_depth_pixels - WINDOW_COPLANAR_DEPTH_PIXELS
    assert full_depth == pytest.approx(WINDOW_CUT_DEPTH_METERS / level.meters_per_pixel)


def test_ordinary_room_rectangle_is_not_treated_as_window_jambs() -> None:
    level, _width_axis, _depth_axis = _build_snapped_window_level()
    level.vertex_data.edges = [
        edge
        for edge in level.vertex_data.edges
        if edge.start_vertex_id <= 4 and edge.end_vertex_id <= 4
    ]

    (opening,) = _build_window_openings(level)

    full_depth = 2.0 * opening.half_depth_pixels - WINDOW_COPLANAR_DEPTH_PIXELS
    assert full_depth == pytest.approx(WINDOW_CUT_DEPTH_METERS / level.meters_per_pixel)
