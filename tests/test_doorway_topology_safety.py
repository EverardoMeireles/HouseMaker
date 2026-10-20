# ### Imports ###
from __future__ import annotations

import pytest

from housemaker.doorway_topology import apply_doorway_fit, find_doorway_fit
from housemaker.models import DoorwayData, VertexData


# ### Fixture helpers ###
def _gap_graph() -> tuple[VertexData, tuple[int, ...]]:
    graph = VertexData()
    corners = tuple(
        graph.add_vertex(*point).id
        for point in ((100.0, 0.0), (200.0, 0.0), (100.0, 10.0), (200.0, 10.0))
    )
    graph.add_edge(corners[0], corners[2])
    graph.add_edge(corners[1], corners[3])
    for index, vertex_id in enumerate(corners):
        corner = graph.get_vertex(vertex_id)
        outside = graph.add_vertex(0.0 if index % 2 == 0 else 300.0, corner.y)
        graph.add_edge(vertex_id, outside.id)
    return graph, corners


def _fit(graph: VertexData):
    fit = find_doorway_fit(
        graph,
        (150.0, 5.0),
        DoorwayData(150.0, 5.0, 2.0, 2.1),
        12.0,
    )
    assert fit is not None
    assert fit.corner_vertex_ids is not None
    return fit


# ### Atomic placement safety tests ###
def test_existing_jamb_fit_splits_non_noded_wall_faces_across_the_opening() -> None:
    graph, _corners = _gap_graph()
    for face_y in (0.0, 10.0):
        first = next(
            vertex
            for vertex in graph.vertices
            if vertex.x == 0.0 and vertex.y == face_y
        )
        second = next(
            vertex
            for vertex in graph.vertices
            if vertex.x == 300.0 and vertex.y == face_y
        )
        graph.add_edge(first.id, second.id)
    fit = _fit(graph)
    apply_doorway_fit(graph, fit)
    for edge in graph.edges:
        start, end = (
            graph.get_vertex(edge.start_vertex_id),
            graph.get_vertex(edge.end_vertex_id),
        )
        if start.y == end.y and start.y in (0.0, 10.0):
            assert max(start.x, end.x) <= 100.0 or min(start.x, end.x) >= 200.0


def test_stale_gap_preview_rejects_new_crossing_wall_without_mutating_graph() -> None:
    graph, _corners = _gap_graph()
    fit = _fit(graph)
    first = graph.add_vertex(150.0, -20.0)
    second = graph.add_vertex(150.0, 30.0)
    graph.add_edge(first.id, second.id)
    before = graph.clone()
    with pytest.raises(ValueError):
        apply_doorway_fit(graph, fit)
    assert graph == before
    assert graph.revision == before.revision


def test_stale_gap_preview_rejects_removed_outward_wall_branch() -> None:
    graph, corners = _gap_graph()
    fit = _fit(graph)
    branch = next(
        edge
        for edge in graph.edges
        if corners[0] in (edge.start_vertex_id, edge.end_vertex_id)
        and corners[2] not in (edge.start_vertex_id, edge.end_vertex_id)
    )
    graph.remove_edge(branch.start_vertex_id, branch.end_vertex_id)
    before = graph.clone()
    with pytest.raises(ValueError):
        apply_doorway_fit(graph, fit)
    assert graph == before
    assert graph.revision == before.revision


def test_room_center_at_a_cut_corner_is_not_reused_or_modified() -> None:
    graph = VertexData()
    for y in (0.0, 10.0):
        first, second = graph.add_vertex(0.0, y), graph.add_vertex(300.0, y)
        graph.add_edge(first.id, second.id)
    center = graph.add_vertex(100.0, 0.0)
    guide = graph.add_vertex(150.0, 30.0)
    graph.add_edge(center.id, guide.id)
    fit = find_doorway_fit(
        graph,
        (150.0, 5.0),
        DoorwayData(150.0, 5.0, 2.0, 2.1),
        12.0,
        room_center_vertex_ids={center.id},
    )
    assert fit is not None

    doorway = apply_doorway_fit(graph, fit)

    assert center.id not in doorway.corner_vertex_ids
    assert graph.get_vertex(center.id) == center
    assert graph.has_edge(center.id, guide.id)


def test_degenerate_unrelated_face_edge_does_not_crash_the_cut() -> None:
    graph, _ = _gap_graph()
    first, second = graph.add_vertex(150.0, 0.0), graph.add_vertex(150.0, 0.0)
    graph.add_edge(first.id, second.id)

    doorway = apply_doorway_fit(graph, _fit(graph))

    assert len(doorway.corner_vertex_ids) == 4
    assert graph.has_edge(first.id, second.id)
