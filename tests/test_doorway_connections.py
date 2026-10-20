# ### Imports ###
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import shapely
from shapely import Point, Polygon

from housemaker.doorway_connections import (
    get_doorway_structural_edges,
    get_level_structural_edges,
    get_level_structural_vertex_data,
)
from housemaker.floor_geometry import build_level_floor_footprint
from housemaker.glb import (
    _build_level_doorway_reveals,
    _build_level_meshes,
)
from housemaker.models import (
    DOORWAY_SHAPE_ARCH,
    DoorwayData,
    LevelData,
    RoomData,
    VertexData,
)
from housemaker.surface_geometry import (
    SURFACE_TYPE_CEILING,
    SURFACE_TYPE_FLOOR,
    build_base_fixed_surfaces,
)
from housemaker.wall_orientation import build_level_wall_orientation_resolver


# ### Fixture helpers ###
def _cut_level(*, arched: bool = False, with_room: bool = False) -> LevelData:
    graph = VertexData()
    point_ids: dict[tuple[float, float], int] = {}

    def vertex_id(point: tuple[float, float]) -> int:
        if point not in point_ids:
            point_ids[point] = graph.add_vertex(*point).id
        return point_ids[point]

    outer = ((0, 0), (100, 0), (200, 0), (300, 0), (300, 300), (0, 300))
    inner = ((10, 10), (100, 10), (200, 10), (290, 10), (290, 290), (10, 290))
    for ring in (outer, inner):
        for first, second in zip(ring, (*ring[1:], ring[0])):
            first_id, second_id = vertex_id(first), vertex_id(second)
            if first[0] == 100 and second[0] == 200:
                continue
            graph.add_edge(first_id, second_id)
    corners = tuple(
        vertex_id(point) for point in ((100, 10), (200, 10), (100, 0), (200, 0))
    )
    graph.add_edge(corners[0], corners[2])
    graph.add_edge(corners[1], corners[3])
    doorway = DoorwayData(
        center_x=150,
        center_y=5,
        width_meters=2,
        height_meters=2.1,
        depth_meters=0.2,
        rotation_degrees=270,
        shape=DOORWAY_SHAPE_ARCH if arched else "rectangular",
    )
    doorway.corner_vertex_ids = corners
    rooms = []
    if with_room:
        center_id = graph.add_vertex(150, 150).id
        rooms.append(
            RoomData(
                "Room",
                tuple(vertex_id(point) for point in inner),
                center_id,
                (200, 200, 200),
            )
        )
    return LevelData(
        index=2, name="Ground", vertex_data=graph, doorways=[doorway], rooms=rooms
    )


def _span_keys(level: LevelData) -> set[tuple[int, int]]:
    first, second, third, fourth = level.doorways[0].corner_vertex_ids
    return {tuple(sorted((first, second))), tuple(sorted((third, fourth)))}


# ### Structural connection tests ###
def test_connections_are_derived_without_mutating_saved_graph() -> None:
    level = _cut_level()
    before = level.vertex_data.clone()
    keys = {
        tuple(sorted((edge.start_vertex_id, edge.end_vertex_id)))
        for edge in get_level_structural_edges(level)
    }
    assert _span_keys(level).issubset(keys)
    assert len(keys) == len(before.edges) + 2
    assert level.vertex_data == before
    assert level.vertex_data.revision == before.revision


def test_existing_virtual_span_is_not_duplicated() -> None:
    level = _cut_level()
    for first, second in _span_keys(level):
        level.vertex_data.add_edge(first, second)
    assert get_level_structural_edges(level) == tuple(level.vertex_data.edges)
    assert get_level_structural_vertex_data(level) is level.vertex_data


@pytest.mark.parametrize(
    "damage",
    (
        "missing_vertex",
        "duplicate_vertex",
        "missing_jamb",
        "skew",
        "wrong_axis",
        "room_center",
    ),
)
def test_invalid_corner_links_do_not_close_unrelated_wall_gaps(damage: str) -> None:
    level = _cut_level()
    doorway = level.doorways[0]
    first, second, third, fourth = doorway.corner_vertex_ids
    ignored_ids = ()
    if damage == "missing_vertex":
        doorway.corner_vertex_ids = (first, second, third, 999999)
    elif damage == "duplicate_vertex":
        doorway.corner_vertex_ids = (first, second, third, first)
    elif damage == "missing_jamb":
        level.vertex_data.remove_edge(first, third)
    elif damage == "skew":
        level.vertex_data.vertices = [
            replace(vertex, x=vertex.x + 2) if vertex.id == fourth else vertex
            for vertex in level.vertex_data.vertices
        ]
    elif damage == "wrong_axis":
        doorway.rotation_degrees += 30
    elif damage == "room_center":
        ignored_ids = (first,)
    assert get_doorway_structural_edges(
        level.vertex_data, level.doorways, ignored_vertex_ids=ignored_ids
    ) == tuple(level.vertex_data.edges)


def test_legacy_unbound_doorway_does_not_change_structural_topology() -> None:
    level = _cut_level()
    level.doorways[0].corner_vertex_ids = None
    assert get_level_structural_edges(level) == tuple(level.vertex_data.edges)


# ### Floor, ceiling and wall geometry tests ###
def test_doorway_connects_exact_floor_perimeter_even_with_gap_closing_disabled() -> (
    None
):
    level = _cut_level()
    footprint = build_level_floor_footprint(
        level,
        None,
        lambda point, _size: np.asarray(point, dtype=float) * 0.02,
        closing_radius_meters=1e-7,
    )
    assert footprint is not None
    assert math.isclose(footprint.area, 36.0)


@pytest.mark.parametrize("with_room", (False, True))
@pytest.mark.parametrize("arched", (False, True))
def test_cut_graph_preserves_lintel_and_semantic_floor_ceiling(
    with_room: bool, arched: bool
) -> None:
    level = _cut_level(arched=arched, with_room=with_room)
    physical_edges = tuple(level.vertex_data.edges)
    surfaces = build_base_fixed_surfaces([level])
    assert any(surface.surface_type == SURFACE_TYPE_FLOOR for surface in surfaces)
    assert any(surface.surface_type == SURFACE_TYPE_CEILING for surface in surfaces)
    if not with_room:
        surface_ids = {surface.surface_id for surface in surfaces}
        assert all(
            f"level:2/wall:{first}:{second}" in surface_ids
            for first, second in _span_keys(level)
        )
    front_triangles = []
    for surface in surfaces:
        for triangle in surface.mesh.triangles:
            if np.allclose(triangle[:, 1], -0.2):
                polygon = Polygon(triangle[:, (0, 2)])
                if polygon.area > 1e-8:
                    front_triangles.append(polygon)
    front_wall = shapely.union_all(front_triangles)
    base_z = level.floor_thickness_meters
    assert front_wall.covers(Point(3.0, base_z + 2.5))
    assert not front_wall.covers(Point(3.0, base_z + 0.8))
    if arched:
        assert front_wall.covers(Point(2.05, base_z + 1.7))
        assert not front_wall.covers(Point(3.0, base_z + 1.7))
    assert tuple(level.vertex_data.edges) == physical_edges


@pytest.mark.parametrize("arched", (False, True))
def test_derived_spans_have_reveals_and_export_geometry(arched: bool) -> None:
    level = _cut_level(arched=arched)
    reveals = _build_level_doorway_reveals(level)
    assert len(reveals) == 1
    assert reveals[0].owner_surface_id in {
        f"level:2/wall:{first}:{second}" for first, second in _span_keys(level)
    }
    meshes = _build_level_meshes(
        level.vertex_data,
        level.height_meters,
        0,
        None,
        doorways=level.doorways,
        level=level,
    )
    # Two ordinary spans were removed, but their lintels are still rendered.
    assert len(meshes) >= len(level.vertex_data.edges)


def test_wall_orientation_sees_closed_room_cells_through_doorway() -> None:
    level = _cut_level()
    resolver = build_level_wall_orientation_resolver(level)
    assert any(math.isclose(polygon.area, 31.36) for polygon in resolver.bounded_cells)
    assert resolver.floor_footprint is not None
    assert math.isclose(resolver.floor_footprint.area, 36)
