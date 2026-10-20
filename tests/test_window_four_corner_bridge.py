"""Regressions for window gaps supported by four real wall corners."""

# ### Imports ###
from __future__ import annotations

import math

import pytest

from housemaker.models import DoorwayData, LevelData, VertexData, WindowData
from housemaker.surface_geometry import WallWindowPlacement, add_wall_window
from housemaker.window_frame_bridge import (
    add_window_frame_bridge,
    build_window_frame_bridge_placement,
    normalize_wall_window_placement,
)


# ### Fixtures ###
def _level() -> LevelData:
    graph = VertexData()
    for point in (
        (10.0, 50.0),
        (40.0, 50.0),
        (90.0, 50.0),
        (130.0, 50.0),
        (10.0, 60.0),
        (40.0, 60.0),
        (90.0, 60.0),
        (130.0, 60.0),
    ):
        graph.add_vertex(*point)
    for edge in ((1, 2), (3, 4), (5, 6), (7, 8), (2, 6), (3, 7)):
        graph.add_edge(*edge)
    return LevelData(index=2, name="Ground", vertex_data=graph)


def _placement(level: LevelData, pointer=(65.0, 50.0)) -> WallWindowPlacement | None:
    return build_window_frame_bridge_placement(
        level,
        pointer,
        5.0,
        preferred_width_meters=1.0,
        height_meters=1.0,
        wall_margin_meters=0.025,
    )


def _snapshot(graph: VertexData):
    return tuple(graph.vertices), tuple(graph.edges), graph.revision


def _solid_level() -> LevelData:
    graph = VertexData()
    for point in ((10.0, 50.0), (210.0, 50.0), (10.0, 60.0), (210.0, 60.0)):
        graph.add_vertex(*point)
    for edge in ((1, 2), (3, 4), (1, 3), (2, 4)):
        graph.add_edge(*edge)
    return LevelData(index=2, name="Ground", vertex_data=graph)


# ### Pure preview tests ###
def test_preview_requires_four_corners_not_a_single_face_gap():
    level = _level()
    level.vertex_data.edges = level.vertex_data.edges[:2]
    assert _placement(level) is None


def test_preview_is_pure_and_matches_all_four_corners_exactly():
    level = _level()
    before = _snapshot(level.vertex_data)
    placement = _placement(level)
    assert placement is not None
    assert placement.bridge_vertex_ids == (2, 3)
    assert placement.bridge_corner_vertex_ids == (2, 3, 6, 7)
    assert placement.start_ratio == 0.0
    assert placement.end_ratio == 1.0
    assert _snapshot(level.vertex_data) == before


def test_hovering_opposite_face_uses_that_surface_and_ordered_corners():
    placement = _placement(_level(), (65.0, 60.0))
    assert placement is not None
    assert placement.wall_surface_id == "level:2/wall:6:7"
    assert placement.bridge_corner_vertex_ids == (6, 7, 2, 3)


@pytest.mark.parametrize("angle", (30.0, 90.0, 157.0))
def test_rotated_four_corner_gaps_still_snap_and_commit(angle):
    level = _level()
    sine, cosine = math.sin(math.radians(angle)), math.cos(math.radians(angle))
    for vertex in tuple(level.vertex_data.vertices):
        level.vertex_data.move_vertex(
            vertex.id,
            vertex.x * cosine - vertex.y * sine + 200.0,
            vertex.x * sine + vertex.y * cosine + 200.0,
        )
    pointer = 65.0 * cosine - 50.0 * sine + 200.0, 65.0 * sine + 50.0 * cosine + 200.0
    placement = _placement(level, pointer)
    assert placement is not None
    transaction = add_window_frame_bridge([level], placement)
    assert transaction is not None
    assert len(transaction.added_edges) == 2
    assert len(level.vertex_data.vertices) == 8
    assert level.vertex_data.has_edge(2, 3)
    assert level.vertex_data.has_edge(6, 7)


@pytest.mark.parametrize("missing_edge", ((1, 2), (7, 8)))
def test_incomplete_outward_wall_supports_do_not_snap(missing_edge):
    level = _level()
    level.vertex_data.remove_edge(*missing_edge)
    assert _placement(level) is None


@pytest.mark.parametrize("missing_jambs", (((2, 6),), ((3, 7),), ((2, 6), (3, 7))))
def test_gap_without_jamb_caps_still_fits_and_connects_all_four_corners(missing_jambs):
    level = _level()
    for edge in missing_jambs:
        level.vertex_data.remove_edge(*edge)
    before = _snapshot(level.vertex_data)
    placement = _placement(level)
    assert placement is not None
    assert placement.bridge_corner_vertex_ids == (2, 3, 6, 7)
    assert _snapshot(level.vertex_data) == before
    transaction = add_window_frame_bridge([level], placement)
    assert transaction is not None
    for edge in ((2, 6), (3, 7), (2, 3), (6, 7)):
        assert level.vertex_data.has_edge(*edge)


def test_nonrectangular_gap_does_not_snap():
    level = _level()
    level.vertex_data.move_vertex(7, 92.0, 60.0)
    assert _placement(level) is None


def test_fully_connected_four_corner_wall_reuses_exact_existing_span():
    level = _level()
    level.vertex_data.add_edge(2, 3)
    level.vertex_data.add_edge(6, 7)
    placement = _placement(level)
    assert placement is not None
    assert placement.bridge_corner_vertex_ids == (2, 3, 6, 7)
    before_edges = {_edge_pair(edge) for edge in level.vertex_data.edges}
    transaction = add_window_frame_bridge([level], placement)
    assert transaction is not None
    assert not transaction.added_edges
    assert {_edge_pair(edge) for edge in level.vertex_data.edges} == before_edges


@pytest.mark.parametrize("support", ((1, 4), (5, 8)))
def test_gap_with_non_noded_long_face_is_cut_without_duplicate_wall_support(support):
    level = _level()
    level.vertex_data.add_edge(*support)
    before = _snapshot(level.vertex_data)
    placement = _placement(level)
    assert placement is not None
    assert _snapshot(level.vertex_data) == before
    transaction = add_window_frame_bridge([level], placement)
    assert transaction is not None
    assert not level.vertex_data.has_edge(*support)
    assert level.vertex_data.has_edge(2, 3)
    assert level.vertex_data.has_edge(6, 7)
    assert len(level.vertex_data.edges) == 8


def test_gap_owned_by_a_doorway_is_not_reused_for_a_window():
    level = _level()
    level.doorways.append(
        DoorwayData(
            center_x=65.0,
            center_y=55.0,
            width_meters=1.0,
            height_meters=2.0,
            corner_vertex_ids=(2, 3, 6, 7),
        )
    )
    assert _placement(level) is None


@pytest.mark.parametrize("surface_id", ("level:2/wall:2:3", "level:2/wall:6:7"))
def test_gap_owned_by_either_window_face_is_not_reused(surface_id):
    level = _level()
    level.windows.append(
        WindowData(
            window_id="existing-window",
            wall_surface_id=surface_id,
            start_ratio=0.0,
            end_ratio=1.0,
            bottom_ratio=0.25,
            top_ratio=0.75,
        )
    )
    assert _placement(level) is None


# ### Atomic commit tests ###
@pytest.mark.parametrize("existing_pair", ((2, 3), (6, 7)))
def test_transaction_tracks_only_missing_support_and_retains_existing_one(
    existing_pair,
):
    level = _level()
    original = level.vertex_data.add_edge(*existing_pair)
    placement = _placement(level)
    assert placement is not None
    before_vertices = tuple(level.vertex_data.vertices)
    transaction = add_window_frame_bridge([level], placement)
    assert transaction is not None
    assert len(transaction.added_edges) == 1
    assert transaction.primary_edge != original
    assert transaction.level is level
    assert tuple(level.vertex_data.vertices) == before_vertices
    for edge in transaction.added_edges:
        level.vertex_data.remove_edge(edge.start_vertex_id, edge.end_vertex_id)
    assert level.vertex_data.has_edge(*existing_pair)


@pytest.mark.parametrize(
    "change", ("moved", "missing_jamb", "new_obstacle", "occupied")
)
def test_stale_corner_geometry_rejected_without_partial_wall_changes(change):
    level = _level()
    placement = _placement(level)
    assert placement is not None
    if change == "moved":
        level.vertex_data.move_vertex(7, 92.0, 60.0)
    elif change == "missing_jamb":
        level.vertex_data.remove_edge(2, 6)
    elif change == "new_obstacle":
        first = level.vertex_data.add_vertex(65.0, 40.0)
        second = level.vertex_data.add_vertex(65.0, 70.0)
        level.vertex_data.add_edge(first.id, second.id)
    else:
        level.vertex_data.add_edge(2, 3)
        level.vertex_data.add_edge(6, 7)
    before = _snapshot(level.vertex_data)
    with pytest.raises(ValueError, match="four aligned"):
        add_window_frame_bridge([level], placement)
    assert _snapshot(level.vertex_data) == before
    assert not level.windows


def test_missing_corner_rejected_without_adding_any_support():
    level = _level()
    placement = _placement(level)
    assert placement is not None
    level.vertex_data.vertices = [
        vertex for vertex in level.vertex_data.vertices if vertex.id != 7
    ]
    before = _snapshot(level.vertex_data)
    with pytest.raises(ValueError, match="corners no longer exist"):
        add_window_frame_bridge([level], placement)
    assert _snapshot(level.vertex_data) == before


def test_successful_window_keeps_both_sill_and_lintel_supports():
    level = _level()
    placement = _placement(level)
    assert placement is not None
    transaction = add_window_frame_bridge([level], placement)
    assert transaction is not None
    window = add_wall_window([level], placement)
    assert window in level.windows
    assert level.vertex_data.has_edge(2, 3)
    assert level.vertex_data.has_edge(6, 7)
    assert 0.0 < window.bottom_ratio < window.top_ratio < 1.0


def test_regular_solid_wall_placement_requires_no_gap_transaction():
    level = _level()
    placement = WallWindowPlacement(
        wall_surface_id="level:2/wall:1:2",
        start_ratio=0.1,
        end_ratio=0.9,
        bottom_ratio=0.25,
        top_ratio=0.75,
    )
    before = _snapshot(level.vertex_data)
    assert add_window_frame_bridge([level], placement) is None
    assert _snapshot(level.vertex_data) == before


def test_legacy_two_point_bridge_cannot_bypass_four_corner_validation():
    level = _level()
    placement = WallWindowPlacement(
        wall_surface_id="level:2/wall:2:3",
        start_ratio=0.0,
        end_ratio=1.0,
        bottom_ratio=0.25,
        top_ratio=0.75,
        bridge_vertex_ids=(2, 3),
    )
    before = _snapshot(level.vertex_data)
    with pytest.raises(ValueError, match="four wall corner"):
        add_window_frame_bridge([level], placement)
    assert _snapshot(level.vertex_data) == before


def test_swapped_back_corner_order_cannot_create_twisted_bridge():
    level = _level()
    placement = _placement(level)
    assert placement is not None
    invalid = WallWindowPlacement(
        wall_surface_id=placement.wall_surface_id,
        start_ratio=placement.start_ratio,
        end_ratio=placement.end_ratio,
        bottom_ratio=placement.bottom_ratio,
        top_ratio=placement.top_ratio,
        bridge_vertex_ids=(2, 3),
        bridge_corner_vertex_ids=(2, 3, 7, 6),
    )
    before = _snapshot(level.vertex_data)
    with pytest.raises(ValueError, match="four aligned"):
        add_window_frame_bridge([level], invalid)
    assert _snapshot(level.vertex_data) == before


def test_duplicate_window_commit_rejected_without_duplicate_supports():
    level = _level()
    placement = _placement(level)
    assert placement is not None
    add_window_frame_bridge([level], placement)
    add_wall_window([level], placement)
    before = _snapshot(level.vertex_data)
    with pytest.raises(ValueError, match="four aligned"):
        add_window_frame_bridge([level], placement)
    assert _snapshot(level.vertex_data) == before
    assert len(level.windows) == 1


# ### Solid-wall cuts and rollback ###
def test_solid_wall_hover_reserves_four_corners_without_mutation():
    level = _solid_level()
    before = _snapshot(level.vertex_data)
    next_id = level.vertex_data._next_vertex_id
    placement = _placement(level, (100.0, 50.0))
    assert placement is not None
    assert placement.opening_fit is not None
    assert placement.opening_fit.corner_vertex_ids is None
    assert len(set(placement.bridge_corner_vertex_ids)) == 4
    assert all(
        level.vertex_data.get_vertex(value) is None
        for value in placement.bridge_corner_vertex_ids
    )
    assert placement.opening_fit.doorway.depth_meters == pytest.approx(0.2)
    assert _snapshot(level.vertex_data) == before
    assert level.vertex_data._next_vertex_id == next_id
    repeated = _placement(level, (100.0, 50.0))
    assert repeated == placement


@pytest.mark.parametrize("pointer_y", (50.0, 60.0))
def test_solid_wall_commit_cuts_both_faces_and_uses_the_hovered_parent(pointer_y):
    level = _solid_level()
    placement = _placement(level, (100.0, pointer_y))
    assert placement is not None
    transaction = add_window_frame_bridge([level], placement)
    assert transaction is not None
    assert len(level.vertex_data.vertices) == 8
    assert len(level.vertex_data.edges) == 10
    corners = placement.bridge_corner_vertex_ids
    assert corners is not None
    for edge in (
        (corners[0], corners[1]),
        (corners[2], corners[3]),
        (corners[0], corners[2]),
        (corners[1], corners[3]),
    ):
        assert level.vertex_data.has_edge(*edge)
    assert not level.vertex_data.has_edge(1, 2)
    assert not level.vertex_data.has_edge(3, 4)
    assert level.vertex_data.get_vertex(corners[0]).y == pytest.approx(pointer_y)
    window = add_wall_window([level], placement)
    assert window.wall_surface_id == placement.wall_surface_id


def test_solid_wall_transaction_restores_all_edges_vertices_and_allocator():
    level = _solid_level()
    original = level.vertex_data.clone()
    placement = _placement(level, (100.0, 50.0))
    assert placement is not None
    transaction = add_window_frame_bridge([level], placement)
    assert transaction is not None
    level.vertex_data.copy_from(transaction.previous_vertex_data)
    assert tuple(level.vertex_data.vertices) == tuple(original.vertices)
    assert tuple(level.vertex_data.edges) == tuple(original.edges)
    assert level.vertex_data._next_vertex_id == original._next_vertex_id


def test_solid_wall_stale_hover_rejected_atomically():
    level = _solid_level()
    placement = _placement(level, (100.0, 50.0))
    assert placement is not None
    level.vertex_data.add_vertex(500.0, 500.0)
    before = _snapshot(level.vertex_data)
    next_id = level.vertex_data._next_vertex_id
    with pytest.raises(ValueError, match="wall layout changed"):
        add_window_frame_bridge([level], placement)
    assert _snapshot(level.vertex_data) == before
    assert level.vertex_data._next_vertex_id == next_id


def test_second_window_can_cut_an_unoccupied_part_of_the_same_wall_run():
    level = _solid_level()
    first = _placement(level, (60.0, 50.0))
    assert first is not None
    add_window_frame_bridge([level], first)
    window = add_wall_window([level], first)
    second = _placement(level, (155.0, 50.0))
    assert second is not None
    add_window_frame_bridge([level], second)
    add_wall_window([level], second)
    assert window in level.windows
    assert len(level.windows) == 2


def test_existing_legacy_window_parent_is_not_silently_orphaned_by_a_new_cut():
    level = _solid_level()
    level.windows.append(
        WindowData("existing", "level:2/wall:1:2", 0.05, 0.15, 0.25, 0.75)
    )
    before = _snapshot(level.vertex_data)
    assert _placement(level, (155.0, 50.0)) is None
    assert _snapshot(level.vertex_data) == before


# ### Dedicated 3D-window rectangle normalization ###
def test_drawn_3d_rectangle_gets_the_same_full_wall_cut_and_keeps_vertical_bounds():
    level = _solid_level()
    raw = WallWindowPlacement(
        wall_surface_id="level:2/wall:1:2",
        start_ratio=0.35,
        end_ratio=0.60,
        bottom_ratio=0.2,
        top_ratio=0.7,
    )
    before = _snapshot(level.vertex_data)
    normalized = normalize_wall_window_placement([level], raw)
    assert _snapshot(level.vertex_data) == before
    assert normalized is not raw
    assert normalized.opening_fit is not None
    assert normalized.opening_fit.doorway.width_meters == pytest.approx(1.0)
    assert normalized.opening_fit.doorway.center_x == pytest.approx(105.0)
    assert normalized.opening_fit.doorway.depth_meters == pytest.approx(0.2)
    assert normalized.bottom_ratio == raw.bottom_ratio
    assert normalized.top_ratio == raw.top_ratio
    assert normalized.start_ratio == 0.0
    assert normalized.end_ratio == 1.0
    transaction = add_window_frame_bridge([level], normalized)
    assert transaction is not None
    window = add_wall_window([level], normalized)
    assert window.wall_surface_id == normalized.wall_surface_id
    assert len(level.vertex_data.vertices) == 8
    assert not level.vertex_data.has_edge(1, 2)
    assert not level.vertex_data.has_edge(3, 4)


def test_drawn_single_face_legacy_window_remains_unchanged():
    level = _solid_level()
    level.vertex_data.edges = level.vertex_data.edges[:1]
    raw = WallWindowPlacement("level:2/wall:1:2", 0.35, 0.6, 0.2, 0.7)
    assert normalize_wall_window_placement([level], raw) is raw


def test_normalization_is_idempotent_for_an_existing_four_corner_preview():
    level = _solid_level()
    fitted = _placement(level, (100.0, 50.0))
    assert fitted is not None
    assert normalize_wall_window_placement([level], fitted) is fitted


def test_normalization_does_not_fall_back_to_one_face_when_pair_is_occupied():
    level = _solid_level()
    level.windows.append(
        WindowData("existing", "level:2/wall:1:2", 0.05, 0.15, 0.25, 0.75)
    )
    raw = WallWindowPlacement("level:2/wall:1:2", 0.65, 0.90, 0.2, 0.7)
    before = _snapshot(level.vertex_data)
    with pytest.raises(ValueError, match="four-corner wall area is occupied"):
        normalize_wall_window_placement([level], raw)
    assert _snapshot(level.vertex_data) == before


def test_3d_normalization_uses_scaled_metric_width_not_canvas_transform_scale():
    level = _solid_level()
    level.scale = 1.5
    level.canvas_level_scale = 2.0
    raw = WallWindowPlacement("level:2/wall:1:2", 0.35, 0.60, 0.2, 0.7)
    normalized = normalize_wall_window_placement([level], raw)
    assert normalized.opening_fit is not None
    assert normalized.opening_fit.doorway.width_meters == pytest.approx(1.5)
    assert normalized.opening_fit.doorway.depth_meters == pytest.approx(0.3)
    assert normalized.opening_fit.doorway.center_x == pytest.approx(105.0)


def _edge_pair(edge):
    return tuple(sorted((edge.start_vertex_id, edge.end_vertex_id)))
