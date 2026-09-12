# ### Imports ###
from __future__ import annotations

import math
import unittest

from housemaker.level_coordinates import level_image_to_world_xy
from housemaker.models import LevelData, VertexData, create_default_levels
from housemaker.wall_mirroring import (
    WallMirrorVertexLink,
    find_next_wall_mirror_target_level_index,
    get_wall_mirror_vertex_ids,
    mirror_wall_vertex_group,
    normalize_wall_mirror_links,
    project_world_points_to_level_image,
    reconcile_wall_mirror_topology,
    remove_wall_vertex_mirrors,
    wall_mirror_links_from_payload,
    wall_mirror_links_to_dicts,
)


# ### Fixtures ###
def _build_three_vertex_source() -> tuple[list[LevelData], int, tuple[int, ...]]:
    levels = create_default_levels()
    source = levels[2]
    first = source.vertex_data.add_vertex(100.0, 100.0)
    second = source.vertex_data.add_vertex(200.0, 100.0)
    third = source.vertex_data.add_vertex(200.0, 200.0)
    source.vertex_data.add_edge(first.id, second.id)
    source.vertex_data.add_edge(second.id, third.id)
    return levels, source.index, (first.id, second.id, third.id)


# ### Link tests ###
class WallMirrorLinkTests(unittest.TestCase):
    def test_links_are_validated_deduplicated_and_sorted(self) -> None:
        second = WallMirrorVertexLink(2, 2, 4, 8)
        first = WallMirrorVertexLink(2, 1, 3, 7)

        self.assertEqual(
            normalize_wall_mirror_links((second, first, first)),
            (first, second),
        )
        with self.assertRaisesRegex(ValueError, "two mirrors"):
            normalize_wall_mirror_links(
                (
                    first,
                    WallMirrorVertexLink(2, 1, 3, 9),
                )
            )
        with self.assertRaisesRegex(ValueError, "two source"):
            normalize_wall_mirror_links(
                (
                    first,
                    WallMirrorVertexLink(1, 3, 3, 7),
                )
            )

    def test_link_payload_helpers_are_additive_and_tolerant(self) -> None:
        levels = create_default_levels()
        link = WallMirrorVertexLink(2, 1, 3, 4)
        payload = wall_mirror_links_to_dicts((link,))
        payload.extend(
            (
                {"source_level_index": True},
                {**payload[0], "target_vertex_id": 9},
                "invalid",
            )
        )

        self.assertEqual(
            wall_mirror_links_from_payload(payload, levels=levels),
            (link,),
        )

    def test_green_vertex_ids_include_both_link_endpoints(self) -> None:
        links = (
            WallMirrorVertexLink(1, 4, 2, 7),
            WallMirrorVertexLink(2, 7, 3, 9),
        )

        self.assertEqual(get_wall_mirror_vertex_ids(links, 1), {4})
        self.assertEqual(get_wall_mirror_vertex_ids(links, 2), {7})
        self.assertEqual(get_wall_mirror_vertex_ids(links, 3), {9})


# ### Materialization tests ###
class WallMirrorMaterializationTests(unittest.TestCase):
    def test_group_materialization_copies_only_induced_edges(self) -> None:
        levels, source_index, (first_id, second_id, third_id) = (
            _build_three_vertex_source()
        )

        result = mirror_wall_vertex_group(
            levels,
            (),
            source_index,
            (first_id, second_id),
            source_index + 1,
        )

        target = levels[source_index + 1]
        mapping = dict(result.source_to_target_vertex_ids)
        self.assertEqual(len(target.vertex_data.vertices), 2)
        self.assertEqual(len(target.vertex_data.edges), 1)
        self.assertTrue(
            target.vertex_data.has_edge(mapping[first_id], mapping[second_id])
        )
        self.assertNotIn(third_id, mapping)

    def test_existing_links_are_reused_and_missing_group_edge_is_added(self) -> None:
        levels, source_index, (first_id, second_id, _third_id) = (
            _build_three_vertex_source()
        )
        first_result = mirror_wall_vertex_group(
            levels,
            (),
            source_index,
            (first_id,),
            source_index + 1,
        )
        first_target_id = dict(
            first_result.source_to_target_vertex_ids
        )[first_id]
        target = levels[source_index + 1]
        target.vertex_data.move_vertex(first_target_id, 777.0, -234.0)

        completed = mirror_wall_vertex_group(
            levels,
            first_result.links,
            source_index,
            (first_id, second_id),
            source_index + 1,
        )

        mapping = dict(completed.source_to_target_vertex_ids)
        self.assertEqual(mapping[first_id], first_target_id)
        self.assertEqual(len(target.vertex_data.vertices), 2)
        retained_target = target.vertex_data.get_vertex(first_target_id)
        assert retained_target is not None
        self.assertEqual(
            (retained_target.x, retained_target.y),
            (777.0, -234.0),
        )
        self.assertTrue(
            target.vertex_data.has_edge(mapping[first_id], mapping[second_id])
        )

    def test_new_mirrors_never_reuse_unrelated_coincident_vertex(self) -> None:
        levels, source_index, (first_id, _second_id, _third_id) = (
            _build_three_vertex_source()
        )
        source = levels[source_index]
        target = levels[source_index + 1]
        source_vertex = source.vertex_data.get_vertex(first_id)
        assert source_vertex is not None
        world_point = level_image_to_world_xy(
            source,
            source_vertex.x,
            source_vertex.y,
        )
        image_point = project_world_points_to_level_image(
            target,
            (world_point,),
        )[0]
        unrelated = target.vertex_data.add_vertex(*image_point)
        target.vertex_data._next_vertex_id = unrelated.id

        result = mirror_wall_vertex_group(
            levels,
            (),
            source_index,
            (first_id,),
            target.index,
        )

        mirrored_id = dict(result.source_to_target_vertex_ids)[first_id]
        self.assertNotEqual(mirrored_id, unrelated.id)
        self.assertEqual(len(target.vertex_data.vertices), 2)

    def test_next_target_requires_the_whole_group_to_already_exist(self) -> None:
        levels, source_index, (first_id, second_id, _third_id) = (
            _build_three_vertex_source()
        )
        one_vertex = mirror_wall_vertex_group(
            levels,
            (),
            source_index,
            (first_id,),
            source_index + 1,
        )
        self.assertEqual(
            find_next_wall_mirror_target_level_index(
                levels,
                one_vertex.links,
                source_index,
                (first_id, second_id),
                1,
            ),
            source_index + 1,
        )
        both_vertices = mirror_wall_vertex_group(
            levels,
            one_vertex.links,
            source_index,
            (first_id, second_id),
            source_index + 1,
        )
        self.assertEqual(
            find_next_wall_mirror_target_level_index(
                levels,
                both_vertices.links,
                source_index,
                (first_id, second_id),
                1,
            ),
            source_index + 2,
        )

    def test_materialized_vertices_can_mirror_onward(self) -> None:
        levels, source_index, (first_id, _second_id, _third_id) = (
            _build_three_vertex_source()
        )
        first_result = mirror_wall_vertex_group(
            levels,
            (),
            source_index,
            (first_id,),
            source_index + 1,
        )
        middle_id = first_result.links[0].target_vertex_id

        second_result = mirror_wall_vertex_group(
            levels,
            first_result.links,
            source_index + 1,
            (middle_id,),
            source_index + 2,
        )

        self.assertEqual(len(second_result.links), 2)
        self.assertEqual(
            second_result.links[-1].source_vertex_id,
            middle_id,
        )


# ### Projection tests ###
class WallMirrorProjectionTests(unittest.TestCase):
    def test_batch_projection_is_exact_after_bounds_change(self) -> None:
        level = LevelData(
            index=3,
            name="Scaled",
            scale=0.35,
            offset_x_meters=2.75,
            offset_y_meters=-1.25,
            image_size_pixels=(800.0, 500.0),
            vertex_data=VertexData(),
        )
        level.vertex_data.add_vertex(360.0, 220.0)
        level.vertex_data.add_vertex(420.0, 280.0)
        desired_world_points = ((-8.0, 6.0), (13.0, -9.0), (4.0, 2.0))

        image_points = project_world_points_to_level_image(
            level,
            desired_world_points,
        )
        new_vertices = tuple(
            level.vertex_data.add_vertex(*image_point)
            for image_point in image_points
        )

        for vertex, desired_point in zip(
            new_vertices,
            desired_world_points,
            strict=True,
        ):
            actual_point = level_image_to_world_xy(level, vertex.x, vertex.y)
            self.assertTrue(
                math.isclose(
                    actual_point[0], desired_point[0], abs_tol=1e-8
                )
            )
            self.assertTrue(
                math.isclose(
                    actual_point[1], desired_point[1], abs_tol=1e-8
                )
            )

    def test_reprojecting_an_existing_mirror_excludes_its_old_bounds(self) -> None:
        level = LevelData(index=3, name="Target", scale=2.4)
        fixed = level.vertex_data.add_vertex(-20.0, 5.0)
        movable = level.vertex_data.add_vertex(1000.0, 900.0)
        desired = ((3.5, -7.25),)

        image_point = project_world_points_to_level_image(
            level,
            desired,
            excluded_vertex_ids=(movable.id,),
        )[0]
        level.vertex_data.move_vertex(movable.id, *image_point)
        moved_vertex = level.vertex_data.get_vertex(movable.id)
        assert moved_vertex is not None

        actual = level_image_to_world_xy(
            level,
            moved_vertex.x,
            moved_vertex.y,
        )
        self.assertAlmostEqual(actual[0], desired[0][0])
        self.assertAlmostEqual(actual[1], desired[0][1])
        self.assertIsNotNone(level.vertex_data.get_vertex(fixed.id))


# ### Removal and reconciliation tests ###
class WallMirrorRemovalTests(unittest.TestCase):
    def test_removal_deletes_owned_targets_edges_and_descendants(self) -> None:
        levels, source_index, (first_id, second_id, _third_id) = (
            _build_three_vertex_source()
        )
        first_result = mirror_wall_vertex_group(
            levels,
            (),
            source_index,
            (first_id, second_id),
            source_index + 1,
        )
        middle_mapping = dict(first_result.source_to_target_vertex_ids)
        second_result = mirror_wall_vertex_group(
            levels,
            first_result.links,
            source_index + 1,
            middle_mapping.values(),
            source_index + 2,
        )

        removed = remove_wall_vertex_mirrors(
            levels,
            second_result.links,
            source_index,
            (first_id, second_id),
            remove_incoming=False,
        )

        self.assertEqual(removed.links, ())
        self.assertEqual(levels[source_index + 1].vertex_data.vertices, [])
        self.assertEqual(levels[source_index + 1].vertex_data.edges, [])
        self.assertEqual(levels[source_index + 2].vertex_data.vertices, [])

    def test_removing_selected_incoming_mirror_keeps_its_source(self) -> None:
        levels, source_index, (first_id, _second_id, _third_id) = (
            _build_three_vertex_source()
        )
        result = mirror_wall_vertex_group(
            levels,
            (),
            source_index,
            (first_id,),
            source_index + 1,
        )
        target_id = result.links[0].target_vertex_id

        removed = remove_wall_vertex_mirrors(
            levels,
            result.links,
            source_index + 1,
            (target_id,),
            remove_outgoing=False,
        )

        self.assertEqual(removed.links, ())
        self.assertIsNotNone(
            levels[source_index].vertex_data.get_vertex(first_id)
        )
        self.assertIsNone(
            levels[source_index + 1].vertex_data.get_vertex(target_id)
        )

    def test_reconcile_missing_target_drops_link_without_source_deletion(
        self,
    ) -> None:
        levels, source_index, (first_id, _second_id, _third_id) = (
            _build_three_vertex_source()
        )
        result = mirror_wall_vertex_group(
            levels,
            (),
            source_index,
            (first_id,),
            source_index + 1,
        )
        target_id = result.links[0].target_vertex_id
        levels[source_index + 1].vertex_data.delete_vertex(target_id)

        reconciled = reconcile_wall_mirror_topology(levels, result.links)

        self.assertEqual(reconciled.links, ())
        self.assertIsNotNone(
            levels[source_index].vertex_data.get_vertex(first_id)
        )

    def test_reconcile_missing_source_cascades_owned_descendants(self) -> None:
        levels, source_index, (first_id, _second_id, _third_id) = (
            _build_three_vertex_source()
        )
        first_result = mirror_wall_vertex_group(
            levels,
            (),
            source_index,
            (first_id,),
            source_index + 1,
        )
        middle_id = first_result.links[0].target_vertex_id
        second_result = mirror_wall_vertex_group(
            levels,
            first_result.links,
            source_index + 1,
            (middle_id,),
            source_index + 2,
        )
        levels[source_index].vertex_data.delete_vertex(first_id)

        reconciled = reconcile_wall_mirror_topology(
            levels,
            second_result.links,
        )

        self.assertEqual(reconciled.links, ())
        self.assertEqual(levels[source_index + 1].vertex_data.vertices, [])
        self.assertEqual(levels[source_index + 2].vertex_data.vertices, [])


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
