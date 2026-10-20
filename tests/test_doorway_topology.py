"""Four-corner doorway fitting, preview purity and wall-graph surgery."""

# ### Imports ###
from __future__ import annotations

import math
from dataclasses import replace
from itertools import pairwise

import pytest

from housemaker.doorway_topology import (
    apply_doorway_fit,
    doorway_corner_positions,
    find_doorway_fit,
    move_doorway_corner_vertices,
)
from housemaker.models import DoorwayData, VertexData


# ### Fixtures and assertions ###
def _template(**changes):
    return DoorwayData(50, 5, 0.8, 2.1, **changes)


def _wall_faces(*, subdivisions=(), depth=10.0):
    graph = VertexData()
    rows = []
    for y in (0.0, depth):
        row = [graph.add_vertex(x, y) for x in (0.0, *subdivisions, 100.0)]
        for first, second in pairwise(row):
            graph.add_edge(first.id, second.id)
        rows.append(row)
    return graph, rows


def _gap_graph(*, bridges=False):
    graph = VertexData()
    corners = [graph.add_vertex(x, y) for y in (0.0, 10.0) for x in (30.0, 70.0)]
    for first, second in ((0, 2), (1, 3)):
        graph.add_edge(corners[first].id, corners[second].id)
    for index, corner in enumerate(corners):
        branch = graph.add_vertex(0.0 if index % 2 == 0 else 100.0, corner.y)
        graph.add_edge(corner.id, branch.id)
    if bridges:
        for first, second in ((0, 1), (0, 3), (2, 1), (2, 3)):
            graph.add_edge(corners[first].id, corners[second].id)
    return graph, corners


def _assert_opening(graph, doorway):
    front_left, front_right, back_left, back_right = doorway.corner_vertex_ids
    assert graph.has_edge(front_left, back_left)
    assert graph.has_edge(front_right, back_right)
    for first in (front_left, back_left):
        for second in (front_right, back_right):
            assert not graph.has_edge(first, second)
    expected = doorway_corner_positions(doorway)
    for vertex_id, position in zip(doorway.corner_vertex_ids, expected):
        vertex = graph.get_vertex(vertex_id)
        assert (vertex.x, vertex.y) == pytest.approx(position)


# ### Existing four-corner openings ###
def test_fitting_existing_jambs_is_pure_and_resizes_to_the_gap():
    graph, corners = _gap_graph()
    before = graph.clone()
    doorway = _template()
    fit = find_doorway_fit(graph, (45.0, 5.0), doorway, 8.0)

    assert fit is not None
    assert graph == before
    assert doorway.corner_vertex_ids is None
    assert fit.doorway.center_x == pytest.approx(50.0)
    assert fit.doorway.center_y == pytest.approx(5.0)
    assert fit.doorway.width_meters == pytest.approx(0.8)
    assert fit.doorway.depth_meters == pytest.approx(0.2)
    assert set(fit.corner_vertex_ids) == {vertex.id for vertex in corners}
    result = apply_doorway_fit(graph, fit)
    _assert_opening(graph, result)
    assert len(graph.vertices) == len(before.vertices)


def test_all_direct_and_diagonal_cross_links_are_removed_but_jambs_remain():
    graph, _ = _gap_graph(bridges=True)
    fit = find_doorway_fit(graph, (50.0, 5.0), _template(), 8.0)
    assert fit is not None
    assert fit.corner_vertex_ids is not None
    _assert_opening(graph, apply_doorway_fit(graph, fit))
    assert len(graph.edges) == 6


def test_four_points_without_perpendicular_connected_jamb_pairs_are_invalid():
    graph, corners = _gap_graph()
    graph.remove_edge(corners[0].id, corners[2].id)
    assert find_doorway_fit(graph, (50.0, 5.0), _template(), 2.0) is None


def test_existing_jambs_need_outward_wall_branches_at_every_corner():
    graph, corners = _gap_graph()
    graph.remove_edge(corners[0].id, 5)
    assert find_doorway_fit(graph, (50.0, 5.0), _template(), 2.0) is None


# ### Solid parallel wall face cuts ###
def test_cutting_two_solid_faces_adds_four_corners_without_longitudinal_bridges():
    graph, _ = _wall_faces()
    fit = find_doorway_fit(graph, (50.0, 0.0), _template(), 8.0)
    assert fit is not None
    assert fit.corner_vertex_ids is None
    assert len(graph.vertices) == 4
    result = apply_doorway_fit(graph, fit)
    _assert_opening(graph, result)
    assert len(graph.vertices) == 8
    assert len(graph.edges) == 6
    assert result.center_y == pytest.approx(5.0)
    assert result.depth_meters == pytest.approx(0.2)


def test_connected_subdivisions_are_fitted_as_whole_wall_face_runs():
    graph, _ = _wall_faces(subdivisions=(20.0, 50.0, 80.0))
    fit = find_doorway_fit(graph, (50.0, 0.0), _template(), 8.0)
    assert fit is not None
    assert len(fit.source_edge_keys) == 8
    result = apply_doorway_fit(graph, fit)
    _assert_opening(graph, result)
    assert not any(vertex.x == 50.0 for vertex in graph.vertices)
    assert sum(vertex.x == 20.0 for vertex in graph.vertices) == 2
    assert sum(vertex.x == 80.0 for vertex in graph.vertices) == 2


def test_existing_vertices_at_doorway_ends_are_reused():
    graph, _ = _wall_faces(subdivisions=(30.0, 70.0))
    before = len(graph.vertices)
    fit = find_doorway_fit(graph, (50.0, 5.0), _template(), 8.0)
    assert fit is not None
    _assert_opening(graph, apply_doorway_fit(graph, fit))
    assert len(graph.vertices) == before


def test_single_wall_face_is_not_a_valid_doorway():
    graph = VertexData()
    first, second = graph.add_vertex(0, 0), graph.add_vertex(100, 0)
    graph.add_edge(first.id, second.id)
    assert find_doorway_fit(graph, (50.0, 0.0), _template(), 8.0) is None


def test_distant_room_boundaries_do_not_count_as_a_wall_pair():
    graph, _ = _wall_faces(depth=200.0)
    assert find_doorway_fit(graph, (50.0, 0.0), _template(), 8.0) is None


def test_solid_cut_leaves_outward_wall_branches_at_all_four_corners():
    graph, _ = _wall_faces()
    fit = find_doorway_fit(graph, (0.0, 0.0), _template(), 30.0)
    assert fit is not None
    result = apply_doorway_fit(graph, fit)
    _assert_opening(graph, result)
    for vertex_id in result.corner_vertex_ids:
        incident = [
            edge
            for edge in graph.edges
            if vertex_id in (edge.start_vertex_id, edge.end_vertex_id)
        ]
        assert len(incident) == 2


def test_nonparallel_face_edges_are_not_paired():
    graph, rows = _wall_faces()
    graph.move_vertex(rows[1][-1].id, 100.0, 20.0)
    assert find_doorway_fit(graph, (50.0, 0.0), _template(), 8.0) is None


def test_perpendicular_cross_wall_inside_opening_is_not_silently_deleted():
    graph, rows = _wall_faces(subdivisions=(50.0,))
    graph.add_edge(rows[0][1].id, rows[1][1].id)
    assert find_doorway_fit(graph, (50.0, 5.0), _template(), 2.0) is None


def test_room_center_spokes_are_excluded_from_wall_face_fitting():
    graph = VertexData()
    first, second, room_center = (
        graph.add_vertex(*point) for point in ((0, 0), (100, 0), (50, 10))
    )
    graph.add_edge(first.id, second.id)
    graph.add_edge(first.id, room_center.id)
    graph.add_edge(second.id, room_center.id)
    assert (
        find_doorway_fit(
            graph,
            (50.0, 0.0),
            _template(),
            8.0,
            room_center_vertex_ids={room_center.id},
        )
        is None
    )


# ### Rotations, calibration and stale previews ###
@pytest.mark.parametrize("angle", [0.0, 45.0, 90.0, 137.0])
def test_rotated_wall_faces_use_the_same_corner_convention(angle):
    graph, _ = _wall_faces()
    radians = math.radians(angle)
    for vertex in list(graph.vertices):
        graph.move_vertex(
            vertex.id,
            vertex.x * math.cos(radians) - vertex.y * math.sin(radians),
            vertex.x * math.sin(radians) + vertex.y * math.cos(radians),
        )
    raw_center = (
        50 * math.cos(radians) - 5 * math.sin(radians),
        50 * math.sin(radians) + 5 * math.cos(radians),
    )
    fit = find_doorway_fit(graph, raw_center, _template(), 8.0)
    assert fit is not None
    result = apply_doorway_fit(graph, fit)
    _assert_opening(graph, result)
    assert result.width_meters == pytest.approx(0.8)
    assert result.depth_meters == pytest.approx(0.2)


def test_metric_calibration_controls_requested_width_and_fitted_depth():
    graph, _ = _wall_faces()
    fit = find_doorway_fit(graph, (50.0, 0.0), _template(), 8.0, meters_per_pixel=0.01)
    assert fit is not None
    result = apply_doorway_fit(graph, fit)
    assert result.width_meters == pytest.approx(0.8)
    assert result.depth_meters == pytest.approx(0.1)
    assert sorted(
        {graph.get_vertex(vertex_id).x for vertex_id in result.corner_vertex_ids}
    ) == pytest.approx((10, 90))


def test_stale_fit_fails_without_partial_mutation():
    graph, rows = _wall_faces()
    fit = find_doorway_fit(graph, (50.0, 0.0), _template(), 8.0)
    graph.move_vertex(rows[0][0].id, -1.0, 0.0)
    before = graph.clone()
    with pytest.raises(ValueError, match="changed"):
        apply_doorway_fit(graph, fit)
    assert graph == before


def test_moving_attached_doorway_moves_all_corners_atomically():
    graph, _ = _gap_graph()
    doorway = apply_doorway_fit(
        graph, find_doorway_fit(graph, (50.0, 5.0), _template(), 8.0)
    )
    moved = replace(doorway, center_x=55.0, width_meters=1.0)
    assert move_doorway_corner_vertices(graph, moved)
    _assert_opening(graph, moved)
    assert not move_doorway_corner_vertices(graph, moved)


def test_missing_attached_corner_cannot_partially_move_remaining_corners():
    graph, _ = _gap_graph()
    doorway = apply_doorway_fit(
        graph, find_doorway_fit(graph, (50.0, 5.0), _template(), 8.0)
    )
    graph.delete_vertex(doorway.corner_vertex_ids[0])
    before = graph.clone()
    with pytest.raises(ValueError, match="valid"):
        move_doorway_corner_vertices(graph, replace(doorway, center_x=60.0))
    assert graph == before
