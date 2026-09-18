# ### Imports ###
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

import housemaker.glb as glb
from housemaker.models import (
    PIXEL_TO_METER,
    STAIR_NOSING_FRONT,
    STAIR_NOSING_LEFT,
    STAIR_NOSING_RIGHT,
    STAIR_STARTING_STEP_CURTAIL,
    STAIR_STRINGER_LEFT,
    STAIR_TREAD_EDGE_ROUNDED,
    LevelData,
    StairData,
    StairSectionData,
    VertexData,
)


# ### Fixture helpers ###
def _build_level(index: int, name: str) -> LevelData:
    vertex_data = VertexData()
    vertex_data.add_vertex(0.0, 0.0)
    vertex_data.add_vertex(300.0, 100.0)
    return LevelData(index=index, name=name, vertex_data=vertex_data)


def _build_multi_guide_stair() -> StairData:
    return StairData(
        start_level_index=2,
        start_a_x=0.0,
        start_a_y=0.0,
        start_b_x=0.0,
        start_b_y=50.0,
        end_level_index=3,
        end_a_x=250.0,
        end_a_y=0.0,
        end_b_x=250.0,
        end_b_y=50.0,
        intermediate_sections=(
            StairSectionData(2, 150.0, 0.0, 150.0, 50.0),
            StairSectionData(2, 50.0, 0.0, 50.0, 50.0),
            StairSectionData(3, 200.0, 0.0, 200.0, 50.0),
            StairSectionData(2, 100.0, 0.0, 100.0, 50.0),
        ),
    )


def _copy_stair_with(stair: StairData, **changes: object) -> StairData:
    payload = stair.to_dict()
    payload.update(changes)
    return StairData.from_dict(payload)


def _build_tight_turn_stair(
    nosing_placements: tuple[str, ...],
) -> StairData:
    centers = tuple(
        np.asarray(center, dtype=float)
        for center in ((0.0, 0.0), (2.0, 0.0), (2.0, 0.5), (0.0, 0.5))
    )
    route_directions: list[np.ndarray] = []
    for center_index, center in enumerate(centers):
        if center_index == 0:
            route_direction = centers[1] - center
        elif center_index == len(centers) - 1:
            route_direction = center - centers[center_index - 1]
        else:
            route_direction = centers[center_index + 1] - centers[center_index - 1]
        route_directions.append(route_direction / np.linalg.norm(route_direction))

    sections: list[StairSectionData] = []
    for section_index, (center, route_direction) in enumerate(
        zip(centers, route_directions)
    ):
        width_direction = np.asarray(
            (-route_direction[1], route_direction[0]),
            dtype=float,
        )
        section_a = center - (width_direction * 0.1)
        section_b = center + (width_direction * 0.1)
        sections.append(
            StairSectionData(
                level_index=2 if section_index < 3 else 3,
                a_x=float(section_a[0] / PIXEL_TO_METER),
                a_y=float(-section_a[1] / PIXEL_TO_METER),
                b_x=float(section_b[0] / PIXEL_TO_METER),
                b_y=float(-section_b[1] / PIXEL_TO_METER),
            )
        )

    start, *guides, end = sections
    return StairData(
        start_level_index=start.level_index,
        start_a_x=start.a_x,
        start_a_y=start.a_y,
        start_b_x=start.b_x,
        start_b_y=start.b_y,
        end_level_index=end.level_index,
        end_a_x=end.a_x,
        end_a_y=end.a_y,
        end_b_x=end.b_x,
        end_b_y=end.b_y,
        intermediate_sections=tuple(guides),
        tread_overhang_meters=0.2,
        nosing_placements=nosing_placements,
        stringer_placement=STAIR_STRINGER_LEFT,
    )


# ### Stair route cache tests ###
class StairRouteCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        glb.clear_stair_route_geometry_cache()

    def tearDown(self) -> None:
        glb.clear_stair_route_geometry_cache()

    def test_parameter_only_preview_edits_reuse_route_topology_and_sampling(
        self,
    ) -> None:
        levels = [_build_level(2, "Ground"), _build_level(3, "Story")]
        stair = _build_multi_guide_stair()
        variants = (
            _copy_stair_with(stair, tread_thickness_meters=0.06),
            _copy_stair_with(stair, tread_overhang_meters=0.12),
            _copy_stair_with(
                stair,
                nosing_placements=[STAIR_NOSING_LEFT, STAIR_NOSING_RIGHT],
            ),
            _copy_stair_with(stair, tread_edge_profile=STAIR_TREAD_EDGE_ROUNDED),
            _copy_stair_with(stair, tread_edge_radius_meters=0.02),
            _copy_stair_with(
                stair,
                starting_step=STAIR_STARTING_STEP_CURTAIL,
            ),
            _copy_stair_with(
                stair,
                starting_step=STAIR_STARTING_STEP_CURTAIL,
                starting_step_edge_radius_meters=0.08,
            ),
            _copy_stair_with(
                stair,
                starting_step=STAIR_STARTING_STEP_CURTAIL,
                starting_step_edge_points=4,
            ),
            _copy_stair_with(stair, stringer_placement=STAIR_STRINGER_LEFT),
        )

        with (
            patch(
                "housemaker.glb._order_stair_route_sections",
                wraps=glb._order_stair_route_sections,
            ) as order_route,
            patch(
                "housemaker.glb._build_smoothed_stair_route_sections",
                wraps=glb._build_smoothed_stair_route_sections,
            ) as sample_route,
        ):
            glb.build_canvas_stair_part_targets(levels, (stair,))
            for variant in variants:
                glb.build_canvas_stair_part_targets(levels, (variant,))

        self.assertEqual(order_route.call_count, 1)
        self.assertEqual(sample_route.call_count, 1)

    def test_tight_turn_rejects_overlapping_lateral_nosing_only(self) -> None:
        levels = [_build_level(2, "Ground"), _build_level(3, "Story")]
        front_only_stair = _build_tight_turn_stair((STAIR_NOSING_FRONT,))
        lateral_stair = _build_tight_turn_stair(
            (STAIR_NOSING_LEFT, STAIR_NOSING_RIGHT)
        )

        front_only_parts = glb.build_canvas_stair_part_targets(
            levels,
            (front_only_stair,),
        )
        rounded_stair = _copy_stair_with(
            front_only_stair,
            tread_edge_profile=STAIR_TREAD_EDGE_ROUNDED,
            tread_edge_radius_meters=1.0,
        )
        rounded_parts = glb.build_canvas_stair_part_targets(
            levels,
            (rounded_stair,),
        )
        rounded_treads = next(
            part.mesh.copy()
            for part in rounded_parts
            if part.part_kind == glb.STAIR_PART_TREADS
        )
        rounded_treads.merge_vertices()
        rounded_risers = next(
            part.mesh
            for part in rounded_parts
            if part.part_kind == glb.STAIR_PART_RISERS
        )

        self.assertTrue(front_only_parts)
        # The first tread's exposed leading shell is a separate selectable
        # riser primitive. Each material primitive is therefore intentionally
        # open, while the surviving tread topology remains consistently wound.
        self.assertGreater(len(rounded_risers.faces), 0)
        self.assertTrue(np.isfinite(rounded_treads.vertices).all())
        self.assertTrue(rounded_treads.is_winding_consistent)
        self.assertTrue((rounded_treads.area_faces > 1e-12).all())
        with self.assertRaisesRegex(
            ValueError,
            "lateral nosing overhang.*self-intersect or collapse",
        ):
            glb.build_canvas_stair_part_targets(levels, (lateral_stair,))

    def test_route_and_level_xy_changes_invalidate_cached_route_geometry(
        self,
    ) -> None:
        ground_level = _build_level(2, "Ground")
        upper_level = _build_level(3, "Story")
        levels = [ground_level, upper_level]
        stair = _build_multi_guide_stair()
        changed_sections = list(stair.to_dict()["intermediate_sections"])
        changed_sections[1] = {
            **changed_sections[1],
            "a_y": 10.0,
            "b_y": 60.0,
        }
        changed_route = _copy_stair_with(
            stair,
            intermediate_sections=changed_sections,
        )

        with (
            patch(
                "housemaker.glb._order_stair_route_sections",
                wraps=glb._order_stair_route_sections,
            ) as order_route,
            patch(
                "housemaker.glb._build_smoothed_stair_route_sections",
                wraps=glb._build_smoothed_stair_route_sections,
            ) as sample_route,
        ):
            glb.build_canvas_stair_part_targets(levels, (stair,))
            glb.build_canvas_stair_part_targets(levels, (changed_route,))
            ground_level.offset_x_meters = 0.25
            glb.build_canvas_stair_part_targets(levels, (changed_route,))

        self.assertEqual(order_route.call_count, 3)
        self.assertEqual(sample_route.call_count, 3)

    def test_vertical_level_change_reuses_xy_route_but_updates_mesh_height(
        self,
    ) -> None:
        ground_level = _build_level(2, "Ground")
        upper_level = _build_level(3, "Story")
        levels = [ground_level, upper_level]
        stair = _build_multi_guide_stair()

        with patch(
            "housemaker.glb._order_stair_route_sections",
            wraps=glb._order_stair_route_sections,
        ) as order_route:
            initial_parts = glb.build_canvas_stair_part_targets(levels, (stair,))
            initial_maximum_z = max(
                float(part.mesh.bounds[1, 2]) for part in initial_parts
            )
            ground_level.height_meters += 0.5
            updated_parts = glb.build_canvas_stair_part_targets(levels, (stair,))
            updated_maximum_z = max(
                float(part.mesh.bounds[1, 2]) for part in updated_parts
            )

        self.assertEqual(order_route.call_count, 1)
        self.assertAlmostEqual(updated_maximum_z - initial_maximum_z, 0.5)

    def test_cached_routes_are_thawed_to_fresh_mutable_arrays(self) -> None:
        levels = [_build_level(2, "Ground"), _build_level(3, "Story")]
        stair = _build_multi_guide_stair()
        level_lookup = {level.index: level for level in levels}

        first_control_route = glb._resolve_stair_route_sections(
            stair,
            level_lookup,
        )
        expected_first_x = float(first_control_route[0][0][0])
        first_control_route[0][0][0] = expected_first_x + 1000.0
        second_control_route = glb._resolve_stair_route_sections(
            stair,
            level_lookup,
        )

        first_sampled_route, first_distances = (
            glb._get_smoothed_stair_route_geometry(second_control_route)
        )
        expected_sampled_x = float(first_sampled_route[0][0][0])
        first_sampled_route[0][0][0] = expected_sampled_x + 1000.0
        first_distances[0] = 1000.0
        second_sampled_route, second_distances = (
            glb._get_smoothed_stair_route_geometry(second_control_route)
        )

        self.assertEqual(float(second_control_route[0][0][0]), expected_first_x)
        self.assertEqual(float(second_sampled_route[0][0][0]), expected_sampled_x)
        self.assertEqual(second_distances[0], 0.0)
        self.assertFalse(
            np.shares_memory(
                first_sampled_route[0][0],
                second_sampled_route[0][0],
            )
        )

    def test_route_caches_are_bounded_and_explicitly_clearable(self) -> None:
        levels = [_build_level(2, "Ground"), _build_level(3, "Story")]
        glb.build_canvas_stair_part_targets(levels, (_build_multi_guide_stair(),))

        order_cache_info = glb._order_stair_route_sections_cached.cache_info()
        sample_cache_info = (
            glb._build_smoothed_stair_route_geometry_cached.cache_info()
        )
        self.assertEqual(order_cache_info.maxsize, glb.STAIR_ROUTE_CACHE_MAX_ENTRIES)
        self.assertEqual(sample_cache_info.maxsize, glb.STAIR_ROUTE_CACHE_MAX_ENTRIES)
        self.assertGreater(order_cache_info.currsize, 0)
        self.assertGreater(sample_cache_info.currsize, 0)

        glb.clear_stair_route_geometry_cache()

        self.assertEqual(
            glb._order_stair_route_sections_cached.cache_info().currsize,
            0,
        )
        self.assertEqual(
            glb._build_smoothed_stair_route_geometry_cached.cache_info().currsize,
            0,
        )


if __name__ == "__main__":
    unittest.main()
