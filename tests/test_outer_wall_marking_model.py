# ### Imports ###
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from housemaker.level_coordinates import level_image_to_world_xy
from housemaker.models import GROUND_LEVEL_INDEX, LevelData, RoomData, VertexData
from housemaker.outer_wall_marking import (
    OuterWallMark,
    find_outer_wall_vertex_ids,
    get_outer_wall_mark_vertex_ids,
    normalize_outer_wall_marks,
    outer_wall_marks_from_legacy_links,
    outer_wall_marks_from_legacy_vertex_payload,
    outer_wall_marks_from_payload,
    outer_wall_marks_to_dicts,
    project_world_points_to_level_image,
    reconcile_outer_wall_marks,
)
from housemaker.project_io import load_project, save_project


# ### Fixture helpers ###
def _build_markable_level(level_index: int) -> tuple[LevelData, tuple[int, ...]]:
    """Return a square exterior plus a disconnected interior wall."""

    vertex_data = VertexData()
    boundary_ids = tuple(
        vertex_data.add_vertex(*point).id
        for point in (
            (10.0, 10.0),
            (90.0, 10.0),
            (90.0, 90.0),
            (10.0, 90.0),
        )
    )
    for start_id, end_id in zip(
        boundary_ids,
        (*boundary_ids[1:], boundary_ids[0]),
    ):
        vertex_data.add_edge(start_id, end_id)
    inner_first = vertex_data.add_vertex(35.0, 50.0)
    inner_second = vertex_data.add_vertex(65.0, 50.0)
    vertex_data.add_edge(inner_first.id, inner_second.id)
    center = vertex_data.add_vertex(50.0, 50.0)
    return (
        LevelData(
            index=level_index,
            name=f"Level {level_index}",
            image_size_pixels=(100.0, 100.0),
            vertex_data=vertex_data,
            rooms=[
                RoomData(
                    name="Outer room",
                    vertex_ids=boundary_ids,
                    center_vertex_id=center.id,
                    color_rgb=(120, 140, 160),
                )
            ],
        ),
        boundary_ids,
    )


# ### Mark value tests ###
class OuterWallMarkValueTests(unittest.TestCase):
    def test_marks_are_validated_deduplicated_and_sorted(self) -> None:
        second = OuterWallMark(3, 7)
        first = OuterWallMark(2, 9)

        self.assertEqual(
            normalize_outer_wall_marks((second, first, second)),
            (first, second),
        )
        with self.assertRaises((TypeError, ValueError)):
            OuterWallMark(True, 1)
        with self.assertRaises((TypeError, ValueError)):
            OuterWallMark(2, 0)

    def test_payload_helpers_are_tolerant_and_filter_missing_vertices(self) -> None:
        level, boundary_ids = _build_markable_level(2)
        mark = OuterWallMark(level.index, boundary_ids[0])
        payload: list[object] = outer_wall_marks_to_dicts((mark,))
        payload.extend(
            (
                "invalid",
                {"level_index": True, "vertex_id": boundary_ids[1]},
                {"level_index": level.index, "vertex_id": 999_999},
                {"level_index": 99, "vertex_id": boundary_ids[1]},
            )
        )

        self.assertEqual(
            outer_wall_marks_from_payload(payload, levels=(level,)),
            (mark,),
        )

    def test_reconciliation_drops_deleted_and_non_wall_vertices(self) -> None:
        level, boundary_ids = _build_markable_level(2)
        isolated = level.vertex_data.add_vertex(25.0, 25.0)
        missing = OuterWallMark(level.index, 999_999)

        reconciled = reconcile_outer_wall_marks(
            (level,),
            (
                OuterWallMark(level.index, boundary_ids[0]),
                OuterWallMark(level.index, isolated.id),
                missing,
            ),
        )

        self.assertEqual(
            reconciled,
            (OuterWallMark(level.index, boundary_ids[0]),),
        )

    def test_level_query_returns_only_that_levels_marked_vertices(self) -> None:
        marks = (
            OuterWallMark(1, 4),
            OuterWallMark(2, 7),
            OuterWallMark(2, 9),
        )

        self.assertEqual(
            get_outer_wall_mark_vertex_ids(marks, 2),
            frozenset((7, 9)),
        )

    def test_legacy_mirror_metadata_migrates_only_source_marks(self) -> None:
        level, boundary_ids = _build_markable_level(2)
        first_id, second_id = boundary_ids[:2]

        link_marks = outer_wall_marks_from_legacy_links(
            (
                {
                    "source_level_index": level.index,
                    "source_vertex_id": first_id,
                    "target_level_index": level.index + 1,
                    "target_vertex_id": 99,
                },
            ),
            levels=(level,),
        )
        vertex_marks = outer_wall_marks_from_legacy_vertex_payload(
            {
                "wall_mirror_levels": {
                    str(second_id): [level.index + 1],
                }
            },
            source_level_index=level.index,
            known_vertex_ids=(first_id, second_id),
        )

        self.assertEqual(link_marks, (OuterWallMark(level.index, first_id),))
        self.assertEqual(vertex_marks, (OuterWallMark(level.index, second_id),))


# ### Automatic outer-boundary tests ###
class OuterWallDetectionTests(unittest.TestCase):
    def test_outer_detection_excludes_interior_and_isolated_vertices(self) -> None:
        level, boundary_ids = _build_markable_level(2)
        isolated = level.vertex_data.add_vertex(4.0, 50.0)

        detected_ids = find_outer_wall_vertex_ids(level)

        self.assertEqual(detected_ids, frozenset(boundary_ids))
        self.assertNotIn(isolated.id, detected_ids)

    def test_outer_detection_keeps_each_closed_construction_island(self) -> None:
        vertex_data = VertexData()
        expected_ids: set[int] = set()
        for offset_x in (0.0, 100.0):
            loop_ids = tuple(
                vertex_data.add_vertex(offset_x + x, y).id
                for x, y in (
                    (10.0, 10.0),
                    (50.0, 10.0),
                    (50.0, 50.0),
                    (10.0, 50.0),
                )
            )
            expected_ids.update(loop_ids)
            for start_id, end_id in zip(
                loop_ids,
                (*loop_ids[1:], loop_ids[0]),
            ):
                vertex_data.add_edge(start_id, end_id)
        dangling_start = vertex_data.add_vertex(70.0, 70.0)
        dangling_end = vertex_data.add_vertex(80.0, 80.0)
        vertex_data.add_edge(dangling_start.id, dangling_end.id)
        level = LevelData(
            index=2,
            name="Two islands",
            image_size_pixels=(160.0, 80.0),
            vertex_data=vertex_data,
        )

        detected_ids = find_outer_wall_vertex_ids(level)

        self.assertEqual(detected_ids, frozenset(expected_ids))

    def test_outer_detection_marks_a_partition_endpoint_on_an_unsplit_shell(
        self,
    ) -> None:
        level, boundary_ids = _build_markable_level(2)
        shell_vertex = level.vertex_data.add_vertex(50.0, 10.0)
        interior_vertex = level.vertex_data.add_vertex(50.0, 45.0)
        level.vertex_data.add_edge(shell_vertex.id, interior_vertex.id)

        detected_ids = find_outer_wall_vertex_ids(level)

        self.assertEqual(
            detected_ids,
            frozenset((*boundary_ids, shell_vertex.id)),
        )
        self.assertNotIn(interior_vertex.id, detected_ids)


# ### Coordinate projection tests ###
class OuterWallCoordinateProjectionTests(unittest.TestCase):
    def test_projection_preserves_world_points_after_batch_is_inserted(self) -> None:
        level = LevelData(
            index=2,
            name="Scaled target",
            image_size_pixels=(800.0, 600.0),
            scale=1.75,
            offset_x_meters=2.5,
            offset_y_meters=-1.25,
            vertex_data=VertexData(),
        )
        level.vertex_data.add_vertex(300.0, 250.0)
        level.vertex_data.add_vertex(500.0, 350.0)
        expected_world_points = ((-4.5, 8.0), (12.25, -3.75))

        projected_points = project_world_points_to_level_image(
            level,
            expected_world_points,
        )
        for image_point in projected_points:
            level.vertex_data.add_vertex(*image_point)
        actual_world_points = tuple(
            level_image_to_world_xy(level, *image_point)
            for image_point in projected_points
        )

        for actual, expected in zip(
            actual_world_points,
            expected_world_points,
            strict=True,
        ):
            self.assertAlmostEqual(actual[0], expected[0], places=7)
            self.assertAlmostEqual(actual[1], expected[1], places=7)


# ### Project persistence tests ###
class OuterWallMarkPersistenceTests(unittest.TestCase):
    def test_project_round_trip_preserves_marks_without_copying_geometry(
        self,
    ) -> None:
        source, boundary_ids = _build_markable_level(GROUND_LEVEL_INDEX)
        target = LevelData(
            index=GROUND_LEVEL_INDEX + 1,
            name="Empty target",
            image_size_pixels=(100.0, 100.0),
            vertex_data=VertexData(),
        )
        marks = (
            OuterWallMark(source.index, boundary_ids[0]),
            OuterWallMark(source.index, boundary_ids[1]),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "outer-marks.housemaker"
            save_project(
                project_path,
                GROUND_LEVEL_INDEX,
                [source, target],
                outer_wall_marks=marks,
            )
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            restored = load_project(project_path)

        self.assertEqual(
            payload["outer_wall_marks"],
            outer_wall_marks_to_dicts(marks),
        )
        self.assertNotIn("wall_mirror_links", payload)
        self.assertEqual(restored.outer_wall_marks, marks)
        restored_target = next(
            level for level in restored.levels if level.index == target.index
        )
        self.assertEqual(restored_target.vertex_data.vertices, [])
        self.assertEqual(restored_target.vertex_data.edges, [])

    def test_loading_legacy_links_does_not_materialize_new_target_geometry(
        self,
    ) -> None:
        source, boundary_ids = _build_markable_level(GROUND_LEVEL_INDEX)
        target = LevelData(
            index=GROUND_LEVEL_INDEX + 1,
            name="Empty target",
            image_size_pixels=(100.0, 100.0),
            vertex_data=VertexData(),
        )
        source_id = boundary_ids[0]

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "legacy-mirror.housemaker"
            save_project(project_path, GROUND_LEVEL_INDEX, [source, target])
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            payload.pop("outer_wall_marks", None)
            payload["wall_mirror_links"] = [
                {
                    "source_level_index": source.index,
                    "source_vertex_id": source_id,
                    "target_level_index": target.index,
                    "target_vertex_id": 999_999,
                }
            ]
            project_path.write_text(json.dumps(payload), encoding="utf-8")

            restored = load_project(project_path)

        self.assertEqual(
            restored.outer_wall_marks,
            (OuterWallMark(source.index, source_id),),
        )
        restored_target = next(
            level for level in restored.levels if level.index == target.index
        )
        self.assertEqual(restored_target.vertex_data.vertices, [])
        self.assertEqual(restored_target.vertex_data.edges, [])


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
