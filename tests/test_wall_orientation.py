# ### Imports ###
from __future__ import annotations

import unittest

import numpy as np

from housemaker.models import LevelData, RoomData, VertexData
from housemaker.surface_geometry import build_fixed_surfaces
from housemaker.wall_orientation import build_level_wall_orientation_resolver

# ### Fixtures ###
WallSegment = tuple[tuple[float, float], tuple[float, float]]


def _build_level(segments: tuple[WallSegment, ...]) -> LevelData:
    vertex_data = VertexData()
    vertices_by_point = {}
    for start_point, end_point in segments:
        for point in (start_point, end_point):
            if point not in vertices_by_point:
                vertices_by_point[point] = vertex_data.add_vertex(*point)
        vertex_data.add_edge(
            vertices_by_point[start_point].id,
            vertices_by_point[end_point].id,
        )
    return LevelData(index=2, name="Ground", vertex_data=vertex_data)


def _build_room_owned_outer_wall_level() -> tuple[LevelData, str]:
    """Build a room-owned contour with a nearby inner wall plane."""

    vertex_data = VertexData()
    loops = (
        ((0.0, 0.0), (500.0, 0.0), (500.0, 500.0), (0.0, 500.0)),
        ((20.0, 20.0), (480.0, 20.0), (480.0, 480.0), (20.0, 480.0)),
    )
    loop_vertex_ids = []
    for points in loops:
        vertex_ids = tuple(vertex_data.add_vertex(*point).id for point in points)
        loop_vertex_ids.append(vertex_ids)
        for start_id, end_id in zip(
            vertex_ids,
            (*vertex_ids[1:], vertex_ids[0]),
            strict=True,
        ):
            vertex_data.add_edge(start_id, end_id)
    center = vertex_data.add_vertex(250.0, 250.0)
    outer_ids = loop_vertex_ids[0]
    room = RoomData(
        name="Outer contour room",
        vertex_ids=outer_ids,
        center_vertex_id=center.id,
        color_rgb=(120, 140, 160),
    )
    top_wall_id = (
        f"level:2/room:{center.id}/wall:"
        f"{min(outer_ids[0], outer_ids[1])}:{max(outer_ids[0], outer_ids[1])}"
    )
    return (
        LevelData(
            index=2,
            name="Ground",
            vertex_data=vertex_data,
            rooms=[room],
        ),
        top_wall_id,
    )


# ### Tests ###
class WallOrientationResolverTests(unittest.TestCase):
    def test_parallel_wall_postprocess_faces_toward_the_farther_wall(self) -> None:
        target = ((20.0, 20.0), (480.0, 20.0))
        level = _build_level(
            (
                target,
                ((0.0, 0.0), (500.0, 0.0)),
                ((0.0, 300.0), (500.0, 300.0)),
            )
        )

        normal = build_level_wall_orientation_resolver(
            level
        ).resolve_image_wall_facing_normal(*target)

        np.testing.assert_allclose(normal, (0.0, -1.0), atol=1e-9)

    def test_parallel_wall_postprocess_ignores_target_edge_direction(self) -> None:
        target = ((480.0, 20.0), (20.0, 20.0))
        level = _build_level(
            (
                target,
                ((500.0, 0.0), (0.0, 0.0)),
                ((0.0, 300.0), (500.0, 300.0)),
            )
        )

        normal = build_level_wall_orientation_resolver(
            level
        ).resolve_image_wall_facing_normal(*target)

        np.testing.assert_allclose(normal, (0.0, -1.0), atol=1e-9)

    def test_parallel_wall_postprocess_merges_split_counterpart_plane(self) -> None:
        target = ((20.0, 20.0), (480.0, 20.0))
        split_close_plane = tuple(
            (
                ((end_x, 0.0), (start_x, 0.0))
                if segment_index % 2
                else ((start_x, 0.0), (end_x, 0.0))
            )
            for segment_index, (start_x, end_x) in enumerate(
                zip(
                    (0.0, 125.0, 250.0, 375.0),
                    (125.0, 250.0, 375.0, 500.0),
                    strict=True,
                )
            )
        )
        level = _build_level(
            (
                target,
                *split_close_plane,
                ((0.0, 300.0), (500.0, 300.0)),
            )
        )

        normal = build_level_wall_orientation_resolver(
            level
        ).resolve_image_wall_facing_normal(*target)

        np.testing.assert_allclose(normal, (0.0, -1.0), atol=1e-9)

    def test_short_target_pairs_with_one_long_counterpart_plane(self) -> None:
        target = ((200.0, 20.0), (300.0, 20.0))
        level = _build_level(
            (
                target,
                ((0.0, 0.0), (500.0, 0.0)),
                ((0.0, 300.0), (500.0, 300.0)),
            )
        )

        normal = build_level_wall_orientation_resolver(
            level
        ).resolve_image_wall_facing_normal(*target)

        np.testing.assert_allclose(normal, (0.0, -1.0), atol=1e-9)

    def test_disconnected_or_short_parallel_fragments_do_not_form_a_pair(
        self,
    ) -> None:
        target = ((0.0, 20.0), (500.0, 20.0))
        candidate_sets = (
            (
                ((0.0, 0.0), (200.0, 0.0)),
                ((300.0, 0.0), (500.0, 0.0)),
            ),
            (((150.0, 0.0), (350.0, 0.0)),),
        )
        for candidates in candidate_sets:
            with self.subTest(candidates=candidates):
                resolver = build_level_wall_orientation_resolver(
                    _build_level((target, *candidates))
                )

                self.assertIsNone(
                    resolver.resolve_image_paired_wall_facing_normal(*target)
                )

    def test_close_opposite_room_wall_does_not_override_closed_room(self) -> None:
        top_wall = ((0.0, 0.0), (500.0, 0.0))
        level = _build_level(
            (
                top_wall,
                ((500.0, 0.0), (500.0, 40.0)),
                ((500.0, 40.0), (0.0, 40.0)),
                ((0.0, 40.0), (0.0, 0.0)),
            )
        )

        resolver = build_level_wall_orientation_resolver(level)
        normal = resolver.resolve_image_wall_facing_normal(*top_wall)

        np.testing.assert_allclose(normal, (0.0, -1.0), atol=1e-9)
        self.assertIsNone(
            resolver.resolve_image_paired_wall_facing_normal(*top_wall)
        )

    def test_capped_parallel_planes_face_away_from_the_hidden_cavity(self) -> None:
        top_wall = ((0.0, 0.0), (500.0, 0.0))
        bottom_wall = ((500.0, 20.0), (0.0, 20.0))
        level = _build_level(
            (
                top_wall,
                ((500.0, 0.0), (500.0, 20.0)),
                bottom_wall,
                ((0.0, 20.0), (0.0, 0.0)),
            )
        )
        resolver = build_level_wall_orientation_resolver(level)

        np.testing.assert_allclose(
            resolver.resolve_image_wall_facing_normal(*top_wall),
            (0.0, 1.0),
            atol=1e-9,
        )
        np.testing.assert_allclose(
            resolver.resolve_image_wall_facing_normal(*bottom_wall),
            (0.0, -1.0),
            atol=1e-9,
        )

    def test_room_owned_wall_uses_confident_paired_plane_correction(self) -> None:
        level, top_wall_id = _build_room_owned_outer_wall_level()

        surfaces = {
            surface.surface_id: surface for surface in build_fixed_surfaces([level])
        }

        np.testing.assert_allclose(
            surfaces[top_wall_id].mesh.face_normals,
            np.tile((0.0, 1.0, 0.0), (2, 1)),
            atol=1e-9,
        )

    def test_distant_parallel_wall_is_not_treated_as_a_wall_plane_pair(self) -> None:
        target = ((20.0, 20.0), (480.0, 20.0))
        level = _build_level(
            (
                target,
                ((0.0, 200.0), (500.0, 200.0)),
            )
        )

        normal = build_level_wall_orientation_resolver(
            level
        ).resolve_image_wall_facing_normal(*target)

        self.assertIsNone(normal)

    def test_near_perpendicular_wall_is_not_treated_as_a_pair(self) -> None:
        target = ((20.0, 20.0), (480.0, 20.0))
        level = _build_level(
            (
                target,
                ((250.0, 0.0), (250.0, 100.0)),
                ((0.0, 200.0), (500.0, 200.0)),
            )
        )

        normal = build_level_wall_orientation_resolver(
            level
        ).resolve_image_wall_facing_normal(*target)

        self.assertIsNone(normal)


if __name__ == "__main__":
    unittest.main()
