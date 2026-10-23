"""Room extraction is read-only, transformed consistently and conservative."""

# ### Imports ###
from __future__ import annotations

import copy
from dataclasses import replace

import pytest
from shapely import Point, Polygon

from housemaker.level_coordinates import level_image_to_world_xy
from housemaker.models import (
    DoorwayData,
    LevelData,
    OpenSpaceData,
    StairData,
    VertexData,
)
from housemaker.room_zones import (
    build_room_zones,
    classify_object_room_ids,
    room_ids_for_surface,
    room_zone_image_rings,
)


# ### Confirmed plan fixtures ###
def _level(index=2, *, divided=False):
    graph = VertexData()
    vertices = {}

    def vertex(point):
        if point not in vertices:
            vertices[point] = graph.add_vertex(*point)
        return vertices[point]

    def edge(first, second):
        graph.add_edge(vertex(first).id, vertex(second).id)

    for first, second in (
        ((0, 0), (10, 0)),
        ((10, 0), (10, 6)),
        ((10, 6), (0, 6)),
        ((0, 6), (0, 0)),
    ):
        edge(first, second)
    doorways = []
    if divided:
        for x in (4.9, 5.1):
            edge((x, 0), (x, 2.5))
            edge((x, 3.5), (x, 6))
        edge((4.9, 2.5), (5.1, 2.5))
        edge((4.9, 3.5), (5.1, 3.5))
        corner_ids = tuple(
            vertex(point).id
            for point in (
                (4.9, 2.5),
                (4.9, 3.5),
                (5.1, 2.5),
                (5.1, 3.5),
            )
        )
        doorways.append(
            DoorwayData(
                500,
                300,
                1,
                2.1,
                depth_meters=0.2,
                doorway_id="middle-door",
                corner_vertex_ids=corner_ids,
            )
        )
    graph.vertices = [
        replace(vertex, x=vertex.x * 100, y=vertex.y * 100) for vertex in graph.vertices
    ]
    return LevelData(
        index,
        f"Level {index}",
        vertex_data=graph,
        doorways=doorways,
        image_size_pixels=(1000, 600),
        plan_meters_per_pixel=0.01,
    )


def _point(level, x, y):
    return level_image_to_world_xy(level, x * 100, y * 100)


# ### Stable room boundaries and doorway portals ###
def test_four_corner_doorway_separates_two_rooms_without_editing_saved_graph():
    level = _level(divided=True)
    before = copy.deepcopy(level)
    layout = build_room_zones((level,))
    assert level == before
    assert len(layout.rooms) == 2
    assert all(not room.is_fallback for room in layout.rooms)
    assert sorted(room.polygon.area for room in layout.rooms) == pytest.approx(
        (29.4, 29.4)
    )
    assert len(layout.portals) == 1
    portal = layout.portals[0]
    assert portal.kind == "doorway" and portal.walkable
    assert set(portal.room_ids) == {room.room_id for room in layout.rooms}
    assert portal.level_indices == (2,)
    assert all(room.name.startswith("L2 — Room ") for room in layout.rooms)


@pytest.mark.parametrize("scale", (0.5, 1.7, 3.0))
def test_ids_ignore_input_edge_order_level_scale_and_translation(scale):
    level = _level(divided=True)
    first = build_room_zones((level,))
    transformed = copy.deepcopy(level)
    transformed.scale = scale
    transformed.offset_x_meters = 13
    transformed.offset_y_meters = -8
    transformed.vertex_data.edges.reverse()
    second = build_room_zones((transformed,))
    assert [room.room_id for room in first.rooms] == [
        room.room_id for room in second.rooms
    ]
    assert sorted(room.polygon.area for room in second.rooms) == pytest.approx(
        (29.4 * scale**2,) * 2
    )
    for room in second.rooms:
        exterior, holes = room_zone_image_rings(room, transformed)
        assert not holes
        assert Polygon(exterior).area == pytest.approx(29.4 * 10000)
        assert room.top_z_meters - room.floor_z_meters == transformed.height_meters


def test_unrelated_dead_end_does_not_destroy_proven_rooms():
    level = _level(divided=True)
    a = level.vertex_data.add_vertex(1, 1)
    b = level.vertex_data.add_vertex(2, 1)
    level.vertex_data.add_edge(a.id, b.id)
    layout = build_room_zones((level,))
    assert len([room for room in layout.rooms if not room.is_fallback]) == 2


def test_open_plan_has_fallback_zone_instead_of_invented_closed_rooms():
    level = _level()
    level.vertex_data.edges.pop()
    layout = build_room_zones((level,))
    assert layout.rooms and all(room.is_fallback for room in layout.rooms)
    assert classify_object_room_ids(layout, 2, _point(level, 3, 3))


# ### Object and oriented surface ownership ###
def test_footprints_across_rooms_are_shared_and_exterior_stays_unassigned():
    level = _level(divided=True)
    layout = build_room_zones((level,))
    left = classify_object_room_ids(layout, 2, _point(level, 2, 3))
    right = classify_object_room_ids(layout, 2, _point(level, 8, 3))
    assert len(left) == len(right) == 1 and left != right
    crossing = Polygon(
        [_point(level, *point) for point in ((4, 2), (6, 2), (6, 4), (4, 4))]
    )
    assert set(
        classify_object_room_ids(layout, 2, _point(level, 5, 3), crossing)
    ) == {*left, *right}
    assert classify_object_room_ids(layout, 2, (1000, 1000)) == ()


def test_wall_surface_ownership_follows_visible_side_and_broad_floor_stays_shared():
    level = _level(divided=True)
    layout = build_room_zones((level,))
    left = classify_object_room_ids(layout, 2, _point(level, 2, 3))
    x, y = _point(level, 4.9, 1)
    wall = ((x, y, 0), (x, y + 1, 0), (x, y + 1, 2))
    assert room_ids_for_surface(layout, 2, wall, (-1, 0, 0)) == left
    floor = tuple(
        (*_point(level, *point), 0) for point in ((0, 0), (10, 0), (10, 6), (0, 6))
    )
    assert len(room_ids_for_surface(layout, 2, floor, (0, 0, 1))) == 2


# ### Slab holes and cross-level stairs ###
def test_explicit_slab_hole_remains_a_hole_in_room_and_canvas_rings():
    level = _level()
    level.open_spaces = [OpenSpaceData("shaft", 200, 200, 300, 400)]
    layout = build_room_zones((level,))
    room = layout.rooms[0]
    assert room.polygon.area == pytest.approx(58)
    assert not room.polygon.covers(Point(_point(level, 2.5, 3)))
    assert len(room_zone_image_rings(room, level)[1]) == 1


def test_stairs_connect_levels_without_filling_the_upper_floor_hole():
    lower, upper = _level(2), _level(3)
    stair = StairData(
        start_level_index=2,
        start_a_x=200,
        start_a_y=200,
        start_b_x=300,
        start_b_y=200,
        end_level_index=3,
        end_a_x=200,
        end_a_y=500,
        end_b_x=300,
        end_b_y=500,
    )
    before = copy.deepcopy((lower, upper, stair))
    layout = build_room_zones((lower, upper), (stair,))
    assert (lower, upper, stair) == before
    assert len(layout.portals) == 1
    assert layout.portals[0].kind == "stair"
    assert layout.portals[0].level_indices == (2, 3)
    assert len(layout.portals[0].room_ids) >= 2
    assert sum(room.polygon.area for room in layout.rooms_on_level(3)) < 60
    assert (
        layout.rooms_on_level(3)[0].floor_z_meters
        > layout.rooms_on_level(2)[0].top_z_meters
    )


def test_overlay_can_include_excluded_levels_without_changing_export_default():
    level = _level()
    level.include_in_export = False
    assert build_room_zones((level,)).rooms == ()
    assert build_room_zones((level,), include_excluded=True).rooms


@pytest.mark.parametrize("points", (((float("nan"), 0, 0),), ((0, 0),)))
def test_surface_classifier_rejects_bad_coordinates(points):
    with pytest.raises(ValueError):
        room_ids_for_surface(build_room_zones((_level(),)), 2, points)
