"""Linked doorway gizmos use the contiguous wall run, not just the lintel."""

# ### Imports ###
from __future__ import annotations

import copy
import math
import unittest
from dataclasses import replace

import numpy as np

from housemaker.canvas_openings import (
    CanvasOpeningEdit,
    apply_canvas_opening_edit,
    build_canvas_opening_targets,
)
from housemaker.doorway_topology import (
    apply_doorway_fit,
    doorway_corner_positions,
    find_doorway_fit,
)
from housemaker.models import DoorwayData, LevelData, VertexData
from housemaker.surface_geometry import FixedSurface, build_base_fixed_surfaces


# ### Fixture helpers ###
def _linked_level() -> LevelData:
    graph = VertexData()
    for y in (45.0, 55.0):
        first = graph.add_vertex(10.0, y)
        second = graph.add_vertex(90.0, y)
        graph.add_edge(first.id, second.id)
    fit = find_doorway_fit(
        graph,
        (50.0, 50.0),
        DoorwayData(50.0, 50.0, width_meters=0.9, height_meters=2.1),
        center_tolerance=5.0,
    )
    assert fit is not None
    doorway = apply_doorway_fit(graph, fit)
    return LevelData(index=2, name="Ground", vertex_data=graph, doorways=[doorway])


def _targets(level: LevelData, extras: tuple[FixedSurface, ...] = ()):
    surfaces = build_base_fixed_surfaces([level])
    target = build_canvas_opening_targets([level], [*surfaces, *extras])[0]
    return target, surfaces


def _extension_surface(
    seed: FixedSurface,
    *,
    start: tuple[float, float, float] = (1.8, -0.9, 0.3),
    end: tuple[float, float, float] = (2.2, -0.9, 0.3),
    height: float = 3.0,
    level_index: int = 2,
) -> FixedSurface:
    return replace(
        seed,
        surface_id="level:2/wall:extra",
        wall_start_world=start,
        wall_end_world=end,
        wall_height_meters=height,
        level_index=level_index,
    )


# ### Linked frame and edit tests ###
class LinkedDoorwayOpeningEditTests(unittest.TestCase):
    def _assert_corners_match(self, level: LevelData) -> None:
        doorway = level.doorways[0]
        for vertex_id, point in zip(
            doorway.corner_vertex_ids,
            doorway_corner_positions(doorway, meters_per_pixel=level.meters_per_pixel),
        ):
            vertex = level.vertex_data.get_vertex(vertex_id)
            np.testing.assert_allclose((vertex.x, vertex.y), point, atol=1e-12)

    def test_linked_target_extends_to_both_wall_ends_and_keeps_surface_id(self) -> None:
        level = _linked_level()
        target, surfaces = _targets(level)
        level.doorways[0] = replace(level.doorways[0], corner_vertex_ids=None)
        legacy = build_canvas_opening_targets([level], surfaces)[0]

        self.assertEqual(target.wall_surface_id, legacy.wall_surface_id)
        self.assertAlmostEqual(target.wall_width_meters, 1.6)
        self.assertAlmostEqual(legacy.wall_width_meters, 0.9)
        self.assertAlmostEqual(target.bounds.start_ratio, 0.21875)
        self.assertAlmostEqual(target.bounds.end_ratio, 0.78125)
        self.assertAlmostEqual(legacy.bounds.start_ratio, 0.0)
        self.assertAlmostEqual(legacy.bounds.end_ratio, 1.0)
        np.testing.assert_allclose(target.plane_start_world, (0.2, -1.0, 0.3))

    def test_move_edit_updates_four_corners_and_reverse_edit_restores_them(
        self,
    ) -> None:
        level = _linked_level()
        target, _surfaces = _targets(level)
        original = copy.deepcopy(level.doorways[0])
        original_vertices = tuple(level.vertex_data.vertices)
        original_edges = tuple(level.vertex_data.edges)
        moved_bounds = replace(
            target.bounds,
            start_ratio=target.bounds.start_ratio + 0.1,
            end_ratio=target.bounds.end_ratio + 0.1,
        )
        applied = apply_canvas_opening_edit(
            [level],
            target,
            CanvasOpeningEdit(target.reference, target.wall_surface_id, moved_bounds),
        )
        self.assertAlmostEqual(applied.current.center_x, original.center_x + 8.0)
        self.assertEqual(tuple(level.vertex_data.edges), original_edges)
        self._assert_corners_match(level)
        rebuilt, _surfaces = _targets(level)
        self.assertAlmostEqual(rebuilt.wall_width_meters, 1.6)
        apply_canvas_opening_edit(
            [level],
            target,
            CanvasOpeningEdit(target.reference, target.wall_surface_id, target.bounds),
        )
        self.assertAlmostEqual(level.doorways[0].center_x, original.center_x)
        self.assertEqual(tuple(level.vertex_data.edges), original_edges)
        np.testing.assert_allclose(
            [(vertex.x, vertex.y) for vertex in level.vertex_data.vertices],
            [(vertex.x, vertex.y) for vertex in original_vertices],
            atol=1e-12,
        )

    def test_width_edit_expands_beyond_lintel_and_preserves_missing_faces(self) -> None:
        level = _linked_level()
        target, _surfaces = _targets(level)
        before_edges = tuple(level.vertex_data.edges)
        bounds = replace(
            target.bounds,
            start_ratio=target.bounds.start_ratio - 0.05,
            end_ratio=target.bounds.end_ratio + 0.05,
        )
        apply_canvas_opening_edit(
            [level],
            target,
            CanvasOpeningEdit(target.reference, target.wall_surface_id, bounds),
        )
        self.assertAlmostEqual(level.doorways[0].width_meters, 1.06)
        self.assertEqual(tuple(level.vertex_data.edges), before_edges)
        self._assert_corners_match(level)
        corners = level.doorways[0].corner_vertex_ids
        self.assertFalse(level.vertex_data.has_edge(corners[0], corners[1]))
        self.assertFalse(level.vertex_data.has_edge(corners[2], corners[3]))

    def test_transform_and_scale_are_respected(self) -> None:
        level = _linked_level()
        level.scale = 2.0
        level.offset_x_meters = 10.0
        level.offset_y_meters = -4.0
        cosine, sine = math.cos(math.radians(37.0)), math.sin(math.radians(37.0))
        level.vertex_data.vertices = [
            replace(
                vertex,
                x=50.0 + (vertex.x - 50.0) * cosine - (vertex.y - 50.0) * sine,
                y=50.0 + (vertex.x - 50.0) * sine + (vertex.y - 50.0) * cosine,
            )
            for vertex in level.vertex_data.vertices
        ]
        level.vertex_data.mark_changed()
        level.doorways[0] = replace(
            level.doorways[0],
            rotation_degrees=level.doorways[0].rotation_degrees + 37.0,
        )
        target, _surfaces = _targets(level)
        self.assertAlmostEqual(target.wall_width_meters, 3.2)
        self.assertAlmostEqual(target.bounds.start_ratio, 0.21875)
        self.assertAlmostEqual(target.bounds.end_ratio, 0.78125)

    def test_reverse_directed_surface_extends_the_same_run(self) -> None:
        level = _linked_level()
        target, surfaces = _targets(level)
        seed = next(
            surface
            for surface in surfaces
            if surface.surface_id == target.wall_surface_id
        )
        extension = _extension_surface(
            seed, start=(2.2, -0.9, 0.3), end=(1.8, -0.9, 0.3)
        )
        extended, _surfaces = _targets(level, (extension,))
        self.assertAlmostEqual(extended.wall_width_meters, 2.0)
        self.assertEqual(extended.wall_surface_id, target.wall_surface_id)

    def test_disconnected_and_non_coplanar_surfaces_do_not_extend_target(self) -> None:
        level = _linked_level()
        target, surfaces = _targets(level)
        seed = next(
            surface
            for surface in surfaces
            if surface.surface_id == target.wall_surface_id
        )
        candidates = (
            _extension_surface(seed, start=(1.81, -0.9, 0.3)),
            _extension_surface(seed, start=(1.8001, -0.9, 0.3)),
            _extension_surface(seed, start=(1.8, -0.89, 0.3), end=(2.2, -0.89, 0.3)),
            _extension_surface(seed, start=(1.8, -0.9, 0.3), end=(2.2, -0.89999, 0.3)),
            _extension_surface(seed, start=(1.8, -0.9, 0.4), end=(2.2, -0.9, 0.4)),
            _extension_surface(seed, height=3.2),
            _extension_surface(seed, level_index=3),
            _extension_surface(seed, start=(1.8, -0.9, 0.3), end=(1.8, -0.5, 0.3)),
        )
        for candidate in candidates:
            with self.subTest(candidate=candidate):
                unchanged, _surfaces = _targets(level, (candidate,))
                self.assertAlmostEqual(
                    unchanged.wall_width_meters, target.wall_width_meters
                )
                self.assertEqual(unchanged.wall_surface_id, target.wall_surface_id)

    def test_touching_interval_extension_is_order_independent_on_both_ends(
        self,
    ) -> None:
        level = _linked_level()
        target, surfaces = _targets(level)
        seed = next(
            surface
            for surface in surfaces
            if surface.surface_id == target.wall_surface_id
        )
        extras = (
            _extension_surface(seed, start=(-0.2, -0.9, 0.3), end=(0.2, -0.9, 0.3)),
            _extension_surface(seed, start=(1.8, -0.9, 0.3), end=(2.2, -0.9, 0.3)),
        )
        for ordered in ([*surfaces, *extras], [*extras, *reversed(surfaces)]):
            extended = build_canvas_opening_targets([level], ordered)[0]
            self.assertAlmostEqual(extended.wall_width_meters, 2.4)
            self.assertEqual(extended.wall_surface_id, target.wall_surface_id)
            np.testing.assert_allclose(extended.plane_start_world, (-0.2, -1.0, 0.3))


if __name__ == "__main__":
    unittest.main()
