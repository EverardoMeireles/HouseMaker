# ### Imports ###
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from housemaker.canvas_openings import (
    build_canvas_opening_preview_targets,
    build_canvas_opening_targets,
)
from housemaker.level_coordinates import (
    build_level_base_z_lookup,
    level_image_to_world_xy,
)
from housemaker.models import DoorwayData, LevelData, WindowData
from housemaker.surface_geometry import build_fixed_surfaces


# ### Fixture helpers ###
def _gapped_level(angle: float = 0.0, scale: float = 1.0) -> LevelData:
    level = LevelData(index=2, name="Ground", scale=scale)
    radians = math.radians(angle)
    tangent = np.asarray((math.cos(radians), math.sin(radians)))
    normal = np.asarray((-tangent[1], tangent[0]))
    for along, across in (
        (0, 0), (100, 0), (0, 60), (100, 60),
        (-100, 0), (200, 0), (-100, 60), (200, 60),
    ):
        point = np.asarray((200.0, 200.0)) + tangent * along + normal * across
        level.vertex_data.add_vertex(*point)
    for first, second in ((5, 1), (2, 6), (7, 3), (4, 8), (1, 3), (2, 4)):
        level.vertex_data.add_edge(first, second)
    return level


def _world_vertex(level: LevelData, vertex_id: int) -> tuple[float, float, float]:
    vertex = level.vertex_data.get_vertex(vertex_id)
    base_z = build_level_base_z_lookup((level,))[level.index]
    return (*level_image_to_world_xy(level, vertex.x, vertex.y), base_z)


def _plane_offsets(target, points) -> tuple[float, ...]:
    normal = np.asarray(target.wall_normal_world)
    return tuple(sorted(float(np.dot(point, normal)) for point in points))


def _unlinked_target(center_y: float, depth: float):
    level = LevelData(index=2, name="Ground")
    for point in ((0, 0), (100, 0), (0, 10), (100, 10)):
        level.vertex_data.add_vertex(*point)
    level.vertex_data.add_edge(1, 2)
    level.vertex_data.add_edge(3, 4)
    surfaces = build_fixed_surfaces((level,))
    level.doorways = [
        DoorwayData(50, center_y, 0.8, 2.0, depth, rotation_degrees=90)
    ]
    (target,) = build_canvas_opening_targets((level,), surfaces)
    return level, target


# ### Exact host-plane metadata tests ###
@pytest.mark.parametrize("angle", (0.0, 27.0, 90.0))
@pytest.mark.parametrize("scale", (1.0, 1.7))
def test_pending_linked_doorway_uses_actual_corner_planes_despite_missing_host_id(
    angle: float, scale: float,
) -> None:
    level = _gapped_level(angle, scale)
    old_surfaces = build_fixed_surfaces((level,))
    first = level.vertex_data.get_vertex(1)
    last = level.vertex_data.get_vertex(4)
    level.doorways = [
        DoorwayData(
            (first.x + last.x) * 0.5,
            (first.y + last.y) * 0.5,
            2.0,
            2.1,
            depth_meters=0.2,  # Deliberately not the actual linked wall depth.
            rotation_degrees=angle + 90.0,
            corner_vertex_ids=(1, 2, 3, 4),
        )
    ]
    (target,) = build_canvas_opening_preview_targets((level,), old_surfaces)
    assert target.wall_surface_id not in {surface.surface_id for surface in old_surfaces}
    np.testing.assert_allclose(
        target.host_wall_planes_world,
        (_world_vertex(level, 1), _world_vertex(level, 3)),
    )
    offsets = _plane_offsets(target, target.host_wall_planes_world)
    center = float(np.dot(target.plane_start_world, target.wall_normal_world))
    assert center == pytest.approx(sum(offsets) * 0.5)
    assert offsets[1] - offsets[0] == pytest.approx(1.2 * scale)


def test_pending_linked_window_records_both_real_parallel_faces() -> None:
    level = _gapped_level()
    old_surfaces = build_fixed_surfaces((level,))
    level.vertex_data.add_edge(1, 2)
    level.vertex_data.add_edge(3, 4)
    level.windows = [
        WindowData(
            "window-a", "level:2/wall:1:2", 0.0, 1.0, 0.3, 0.7,
            corner_vertex_ids=(1, 2, 3, 4),
        )
    ]
    (target,) = build_canvas_opening_preview_targets((level,), old_surfaces)
    assert len(target.host_wall_planes_world) == 2
    np.testing.assert_allclose(
        _plane_offsets(target, target.host_wall_planes_world),
        _plane_offsets(target, (_world_vertex(level, 1), _world_vertex(level, 3))),
    )
    assert target.opposite_wall_plane_world in target.host_wall_planes_world


def test_unlinked_doorway_reflects_host_only_when_depth_matches_actual_center_offset() -> None:
    level, target = _unlinked_target(center_y=5.0, depth=0.2)
    assert len(target.host_wall_planes_world) == 2
    np.testing.assert_allclose(
        _plane_offsets(target, target.host_wall_planes_world),
        _plane_offsets(target, (_world_vertex(level, 1), _world_vertex(level, 3))),
    )


@pytest.mark.parametrize(("center_y", "depth"), ((0.0, 0.2), (5.0, 0.6)))
def test_unlinked_doorway_does_not_invent_opposite_plane_when_depth_is_unknown(
    center_y: float, depth: float,
) -> None:
    _level, target = _unlinked_target(center_y, depth)
    assert len(target.host_wall_planes_world) == 1


# ### Metadata validation tests ###
def test_host_plane_points_are_normalized_to_immutable_xyz_tuples() -> None:
    _level, target = _unlinked_target(0.0, 0.2)
    updated = replace(target, host_wall_planes_world=([1, 2, 3], [4, 5, 6]))
    assert updated.host_wall_planes_world == ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
    with pytest.raises(ValueError):
        replace(target, host_wall_planes_world=((1.0, math.nan, 3.0),))
    with pytest.raises(ValueError):
        replace(target, host_wall_planes_world=((1.0, 2.0),))
