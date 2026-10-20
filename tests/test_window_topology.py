"""Persistent four-corner window identity, safe run limits and atomic movement."""

# ### Imports ###
from __future__ import annotations

import math
from dataclasses import replace
from unittest.mock import patch

import pytest
from shapely.geometry import LineString

from housemaker.models import DoorwayData, LevelData, VertexData, WindowData
from housemaker.window_topology import (
    move_linked_window_vertices,
    resolve_linked_window_frame,
)


# ### Fixtures and assertions ###
def _level(*, linked=True):
    graph = VertexData()
    corners = tuple(
        graph.add_vertex(x, y).id for y in (0.0, 10.0) for x in (30.0, 70.0)
    )
    for first, second in ((0, 1), (2, 3), (0, 2), (1, 3)):
        graph.add_edge(corners[first], corners[second])
    outside_ids = []
    for index, corner_id in enumerate(corners):
        corner = graph.get_vertex(corner_id)
        outside = graph.add_vertex(0.0 if index % 2 == 0 else 100.0, corner.y)
        graph.add_edge(corner_id, outside.id)
        outside_ids.append(outside.id)
    window = WindowData(
        window_id="window-linked",
        wall_surface_id=f"level:2/wall:{corners[0]}:{corners[1]}",
        start_ratio=0.0,
        end_ratio=1.0,
        bottom_ratio=0.3,
        top_ratio=0.8,
        corner_vertex_ids=corners if linked else None,
    )
    return (
        LevelData(2, "Ground", vertex_data=graph, windows=[window]),
        window,
        corners,
        tuple(outside_ids),
    )


def _assert_positions(level, corners, points):
    for vertex_id, point in zip(corners, points):
        vertex = level.vertex_data.get_vertex(vertex_id)
        assert (vertex.x, vertex.y) == pytest.approx(point)


# ### Persistent metadata and conservative recognition ###
def test_window_corner_ids_round_trip_and_legacy_payload_remains_supported():
    _, window, corners, _ = _level()
    assert window.to_dict()["corner_vertex_ids"] == list(corners)
    assert WindowData.from_dict(window.to_dict()) == window
    payload = window.to_dict()
    del payload["corner_vertex_ids"]
    assert WindowData.from_dict(payload).corner_vertex_ids is None
    assert "corner_vertex_ids" not in WindowData.from_dict(payload).to_dict()


@pytest.mark.parametrize(
    "invalid",
    [(1, 2, 3), (1, 2, 3, 3), (1, 2, 3, 0), (True, 2, 3, 4), ("1", 2, 3, 4), 123],
)
def test_invalid_persistent_corner_ids_are_rejected(invalid):
    _, window, _, _ = _level()
    with pytest.raises(ValueError, match="four distinct"):
        replace(window, corner_vertex_ids=invalid)


@pytest.mark.parametrize("linked", [False, True])
def test_full_run_and_opening_ratios_are_resolved_on_both_faces(linked):
    level, window, corners, _ = _level(linked=linked)
    frame = resolve_linked_window_frame(level, window)
    assert frame is not None
    assert frame.corner_vertex_ids == corners
    assert frame.start_point == pytest.approx((0.0, 0.0))
    assert frame.end_point == pytest.approx((100.0, 0.0))
    assert frame.start_ratio == pytest.approx(0.3)
    assert frame.end_ratio == pytest.approx(0.7)


def test_legacy_partial_edge_window_is_not_promoted_to_linked_frame():
    level, window, _, _ = _level(linked=False)
    assert (
        resolve_linked_window_frame(
            level, replace(window, start_ratio=0.2, end_ratio=0.8)
        )
        is None
    )


def test_room_rectangle_without_outward_branches_is_not_a_window_frame():
    level, window, corners, outside = _level(linked=False)
    level.vertex_data.delete_vertices(outside)
    assert len(level.vertex_data.vertices) == len(corners)
    assert resolve_linked_window_frame(level, window) is None


def test_sloped_window_keeps_its_existing_unlinked_behavior():
    level, window, _, _ = _level()
    assert (
        resolve_linked_window_frame(level, replace(window, sloped_ceiling_id="a" * 32))
        is None
    )


def test_missing_or_skewed_corners_do_not_produce_an_editable_frame():
    level, window, corners, _ = _level()
    level.vertex_data.move_vertex(corners[3], 73.0, 15.0)
    assert resolve_linked_window_frame(level, window) is None
    level.vertex_data.delete_vertex(corners[3])
    assert resolve_linked_window_frame(level, window) is None


def test_available_run_is_the_shared_span_of_both_parallel_faces():
    level, window, _, outside = _level()
    level.vertex_data.move_vertex(outside[2], 10.0, 10.0)
    level.vertex_data.move_vertex(outside[3], 90.0, 10.0)
    frame = resolve_linked_window_frame(level, window)
    assert frame.start_point == pytest.approx((10.0, 0.0))
    assert frame.end_point == pytest.approx((90.0, 0.0))
    assert frame.start_ratio == pytest.approx(0.25)
    assert frame.end_ratio == pytest.approx(0.75)


# ### Atomic linked movement and resize ###
def test_moving_frame_updates_four_vertices_once_without_mutating_window_list():
    level, window, corners, outside = _level()
    graph = level.vertex_data
    before = graph.clone()
    result = move_linked_window_vertices(level, window, (40.0, 0.0), (80.0, 0.0))
    _assert_positions(level, corners, ((40, 0), (80, 0), (40, 10), (80, 10)))
    assert graph.edges == before.edges
    assert graph.revision == before.revision + 1
    assert graph._next_vertex_id == before._next_vertex_id
    assert level.windows == [window]
    assert result.corner_vertex_ids == corners
    assert (result.start_ratio, result.end_ratio) == (0.0, 1.0)
    assert (result.bottom_ratio, result.top_ratio) == (
        window.bottom_ratio,
        window.top_ratio,
    )
    for vertex_id in outside:
        assert graph.get_vertex(vertex_id) == before.get_vertex(vertex_id)


def test_resizing_moves_paired_jambs_and_retains_parallel_depth():
    level, window, corners, _ = _level()
    result = move_linked_window_vertices(level, window, (20.0, 0.0), (85.0, 0.0))
    _assert_positions(level, corners, ((20, 0), (85, 0), (20, 10), (85, 10)))
    assert result.corner_vertex_ids == corners


def test_repeated_previews_use_current_corners_without_accumulating_motion():
    level, window, corners, _ = _level(linked=False)
    first = move_linked_window_vertices(level, window, (40.0, 0.0), (80.0, 0.0))
    assert first.corner_vertex_ids == corners
    second = move_linked_window_vertices(level, first, (45.0, 0.0), (85.0, 0.0))
    _assert_positions(level, corners, ((45, 0), (85, 0), (45, 10), (85, 10)))
    frame = resolve_linked_window_frame(level, second)
    assert frame.start_point == pytest.approx((0, 0))
    assert frame.end_point == pytest.approx((100, 0))


def test_noop_with_tiny_pointer_roundoff_does_not_change_graph_or_revision():
    level, window, _, _ = _level()
    before = level.vertex_data.clone()
    result = move_linked_window_vertices(
        level, window, (30.0 + 1e-10, 1e-10), (70.0 - 1e-10, 0.0)
    )
    assert result == window
    assert level.vertex_data == before
    assert level.vertex_data.revision == before.revision


@pytest.mark.parametrize(
    "start,end",
    [
        ((80.0, 0), (20.0, 0)),
        ((30.0, 0), (30.0, 0)),
        ((0.0, 0), (40.0, 0)),
        ((50.0, 0), (100.0, 0)),
        ((30.0, 2), (70.0, 2)),
        ((math.nan, 0), (70.0, 0)),
    ],
)
def test_invalid_frame_motion_is_rejected_without_partial_changes(start, end):
    level, window, _, _ = _level()
    before = level.vertex_data.clone()
    with pytest.raises(ValueError):
        move_linked_window_vertices(level, window, start, end)
    assert level.vertex_data == before
    assert level.vertex_data.revision == before.revision


def test_motion_cannot_create_new_crossing_with_unrelated_wall():
    level, window, _, _ = _level()
    graph = level.vertex_data
    first, second = graph.add_vertex(80.0, -5.0), graph.add_vertex(80.0, 15.0)
    graph.add_edge(first.id, second.id)
    before = graph.clone()
    with pytest.raises(ValueError, match="cross"):
        move_linked_window_vertices(level, window, (45.0, 0.0), (85.0, 0.0))
    assert graph == before
    assert graph.revision == before.revision


def test_shared_doorway_corner_is_not_moved_by_linked_window():
    level, window, corners, _ = _level()
    level.doorways.append(DoorwayData(50, 5, 0.8, 2.0, corner_vertex_ids=corners))
    before = level.vertex_data.clone()
    with pytest.raises(ValueError, match="another opening"):
        move_linked_window_vertices(level, window, (40.0, 0.0), (80.0, 0.0))
    assert level.vertex_data == before


def test_other_window_corners_limit_available_run_and_remain_fixed():
    level, window, corners, outside = _level()
    graph = level.vertex_data
    graph.move_vertex(outside[1], 80.0, 0.0)
    graph.move_vertex(outside[3], 80.0, 10.0)
    front_end, back_end = graph.add_vertex(90.0, 0.0), graph.add_vertex(90.0, 10.0)
    far_front, far_back = graph.add_vertex(100.0, 0.0), graph.add_vertex(100.0, 10.0)
    other_ids = outside[1], front_end.id, outside[3], back_end.id
    for first, second in (
        (other_ids[0], other_ids[1]),
        (other_ids[2], other_ids[3]),
        (other_ids[0], other_ids[2]),
        (other_ids[1], other_ids[3]),
        (front_end.id, far_front.id),
        (back_end.id, far_back.id),
    ):
        graph.add_edge(first, second)
    other = replace(
        window,
        window_id="window-other",
        wall_surface_id=f"level:2/wall:{other_ids[0]}:{other_ids[1]}",
        corner_vertex_ids=other_ids,
    )
    level.windows.append(other)
    before = graph.clone()
    frame = resolve_linked_window_frame(level, window)
    assert frame.end_point == pytest.approx((80.0, 0.0))
    with pytest.raises(ValueError, match="adjacent"):
        move_linked_window_vertices(level, window, (35.0, 0.0), (82.0, 0.0))
    assert graph == before
    move_linked_window_vertices(level, window, (35.0, 0.0), (75.0, 0.0))
    _assert_positions(level, corners, ((35, 0), (75, 0), (35, 10), (75, 10)))
    assert all(graph.get_vertex(key) == before.get_vertex(key) for key in other_ids)


def test_large_distant_graph_does_not_construct_shapely_lines_for_every_edge():
    level, window, _, _ = _level()
    graph = level.vertex_data
    for index in range(1000):
        first = graph.add_vertex(200.0 + index * 3.0, 100.0)
        second = graph.add_vertex(201.0 + index * 3.0, 100.0)
        graph.add_edge(first.id, second.id)
    with patch("housemaker.window_topology.LineString", wraps=LineString) as build_line:
        move_linked_window_vertices(level, window, (40.0, 0.0), (80.0, 0.0))
    assert build_line.call_count < 20
    assert len(graph.edges) == 1008


@pytest.mark.parametrize("angle", [45.0, 90.0, 137.0])
def test_rotated_run_and_movement_preserve_the_same_corner_order(angle):
    level, window, corners, _ = _level()
    radians = math.radians(angle)

    def rotated(point):
        x, y = point
        return x * math.cos(radians) - y * math.sin(radians), x * math.sin(
            radians
        ) + y * math.cos(radians)

    for vertex in tuple(level.vertex_data.vertices):
        level.vertex_data.move_vertex(vertex.id, *rotated((vertex.x, vertex.y)))
    frame = resolve_linked_window_frame(level, window)
    assert frame.start_point == pytest.approx(rotated((0, 0)))
    assert frame.end_point == pytest.approx(rotated((100, 0)))
    result = move_linked_window_vertices(
        level, window, rotated((40, 0)), rotated((80, 0))
    )
    _assert_positions(
        level,
        corners,
        tuple(rotated(point) for point in ((40, 0), (80, 0), (40, 10), (80, 10))),
    )
    assert result.corner_vertex_ids == corners
