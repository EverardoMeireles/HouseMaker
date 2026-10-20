"""Window frames infer missing jambs without changing strict doorway fitting."""

# ### Imports ###
from __future__ import annotations

import math

import pytest

from housemaker.doorway_topology import (
    apply_window_fit,
    doorway_corner_positions,
    find_doorway_fit,
)
from housemaker.models import DoorwayData, VertexData


# ### Fixture helpers ###
def _gap_graph(*, caps=0):
    graph = VertexData()
    corners = tuple(
        graph.add_vertex(x, y).id for y in (0.0, 10.0) for x in (30.0, 70.0)
    )
    branches = []
    for index, vertex_id in enumerate(corners):
        corner = graph.get_vertex(vertex_id)
        branch = graph.add_vertex(0.0 if index % 2 == 0 else 100.0, corner.y)
        graph.add_edge(vertex_id, branch.id)
        branches.append(branch.id)
    for first, second in ((0, 2), (1, 3))[:caps]:
        graph.add_edge(corners[first], corners[second])
    return graph, corners, tuple(branches)


def _fit(graph, **options):
    return find_doorway_fit(
        graph,
        (50.0, 5.0),
        DoorwayData(50.0, 5.0, 1.2, 1.0),
        8.0,
        allow_solid_wall_fit=False,
        allow_unconnected_jambs=True,
        **options,
    )


def _assert_window_connections(graph, doorway):
    first, second, third, fourth = doorway.corner_vertex_ids
    assert graph.has_edge(first, second)
    assert graph.has_edge(third, fourth)
    assert graph.has_edge(first, third)
    assert graph.has_edge(second, fourth)
    assert not graph.has_edge(first, fourth)
    assert not graph.has_edge(second, third)
    for vertex_id, point in zip(
        doorway.corner_vertex_ids, doorway_corner_positions(doorway)
    ):
        vertex = graph.get_vertex(vertex_id)
        assert (vertex.x, vertex.y) == pytest.approx(point)


# ### Missing-cap gap inference ###
@pytest.mark.parametrize("caps", [0, 1, 2])
def test_window_gap_reuses_four_supported_corners_with_zero_one_or_two_caps(caps):
    graph, corners, branches = _gap_graph(caps=caps)
    before = graph.clone()
    fit = _fit(graph)
    assert fit is not None
    assert graph == before
    assert graph.revision == before.revision
    assert set(fit.corner_vertex_ids) == set(corners)
    assert fit.doorway.width_meters == pytest.approx(0.8)
    assert fit.doorway.depth_meters == pytest.approx(0.2)

    result = apply_window_fit(graph, fit)
    _assert_window_connections(graph, result)
    assert len(graph.vertices) == 8
    assert len(graph.edges) == 8
    assert graph.revision == before.revision + 1
    for corner_id, branch_id in zip(corners, branches):
        assert graph.has_edge(corner_id, branch_id)


@pytest.mark.parametrize("caps", [0, 1])
def test_doorway_fitting_remains_strict_by_default(caps):
    graph, _, _ = _gap_graph(caps=caps)
    assert (
        find_doorway_fit(
            graph,
            (50.0, 5.0),
            DoorwayData(50.0, 5.0, 1.2, 1.0),
            8.0,
            allow_solid_wall_fit=False,
        )
        is None
    )


def test_four_floating_vertices_do_not_define_a_wall_opening():
    graph, _, _ = _gap_graph()
    graph.edges.clear()
    assert _fit(graph) is None


def test_every_corner_needs_a_real_outward_wall_face_branch():
    graph, corners, branches = _gap_graph()
    graph.remove_edge(corners[0], branches[0])
    assert _fit(graph) is None


def test_nonparallel_or_skewed_endpoint_pairs_are_not_bridged():
    graph, corners, _ = _gap_graph()
    graph.move_vertex(corners[3], 73.0, 13.0)
    assert _fit(graph) is None


def test_cross_wall_through_gap_is_not_removed_to_make_a_window():
    graph, _, _ = _gap_graph()
    first, second = graph.add_vertex(50.0, -20.0), graph.add_vertex(50.0, 30.0)
    graph.add_edge(first.id, second.id)
    assert _fit(graph) is None


def test_excluded_room_center_corner_cannot_support_a_window_gap():
    graph, corners, _ = _gap_graph()
    assert _fit(graph, room_center_vertex_ids={corners[0]}) is None


def test_cross_links_are_removed_but_physical_wall_spans_are_retained():
    graph, corners, _ = _gap_graph()
    graph.add_edge(corners[0], corners[3])
    graph.add_edge(corners[1], corners[2])
    fit = _fit(graph)
    assert fit is not None
    _assert_window_connections(graph, apply_window_fit(graph, fit))


# ### Solid wall cuts and atomic placement ###
def test_solid_wall_fit_creates_four_corners_across_exact_wall_thickness():
    graph = VertexData()
    for y in (0.0, 17.0):
        first, second = graph.add_vertex(0.0, y), graph.add_vertex(100.0, y)
        graph.add_edge(first.id, second.id)
    fit = find_doorway_fit(
        graph,
        (50.0, 0.0),
        DoorwayData(50.0, 0.0, 0.8, 1.0),
        8.0,
        allow_unconnected_jambs=True,
    )
    assert fit is not None
    assert fit.corner_vertex_ids is None
    result = apply_window_fit(graph, fit)
    assert result.depth_meters == pytest.approx(0.34)
    assert len(graph.vertices) == 8
    assert len(graph.edges) == 8
    _assert_window_connections(graph, result)


def test_unrelated_parallel_solid_edge_endpoints_are_not_mistaken_for_gap():
    graph = VertexData()
    for y in (0.0, 10.0):
        first, second = graph.add_vertex(0.0, y), graph.add_vertex(100.0, y)
        graph.add_edge(first.id, second.id)
    assert _fit(graph) is None


@pytest.mark.parametrize("angle", [45.0, 90.0, 137.0])
def test_missing_jamb_inference_is_rotation_independent(angle):
    graph, corners, _ = _gap_graph()
    radians = math.radians(angle)
    for vertex in tuple(graph.vertices):
        graph.move_vertex(
            vertex.id,
            vertex.x * math.cos(radians) - vertex.y * math.sin(radians),
            vertex.x * math.sin(radians) + vertex.y * math.cos(radians),
        )
    center = (
        50.0 * math.cos(radians) - 5.0 * math.sin(radians),
        50.0 * math.sin(radians) + 5.0 * math.cos(radians),
    )
    fit = find_doorway_fit(
        graph,
        center,
        DoorwayData(*center, 1.2, 1.0),
        8.0,
        allow_solid_wall_fit=False,
        allow_unconnected_jambs=True,
    )
    assert fit is not None
    assert set(fit.corner_vertex_ids) == set(corners)
    _assert_window_connections(graph, apply_window_fit(graph, fit))


def test_stale_window_preview_cannot_partially_mutate_graph():
    graph, _, _ = _gap_graph()
    fit = _fit(graph)
    assert fit is not None
    first, second = graph.add_vertex(50.0, -20.0), graph.add_vertex(50.0, 30.0)
    graph.add_edge(first.id, second.id)
    before = graph.clone()
    with pytest.raises(ValueError, match="changed"):
        apply_window_fit(graph, fit)
    assert graph == before
    assert graph.revision == before.revision
