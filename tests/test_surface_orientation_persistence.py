# ### Imports ###
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from housemaker.models import DoorwayData, LevelData, RoomData, VertexData
from housemaker.project_io import load_project, save_project
from housemaker.surface_geometry import build_fixed_surfaces
from housemaker.surface_orientation_edits import (
    flip_surface_orientation,
    remap_flipped_surface_ids_with_lineage,
)


# ### Fixture helpers ###
def _build_square_level() -> LevelData:
    vertex_data = VertexData()
    vertex_ids = tuple(
        vertex_data.add_vertex(*point).id
        for point in (
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (0.0, 100.0),
        )
    )
    for start_id, end_id in zip(
        vertex_ids,
        (*vertex_ids[1:], vertex_ids[0]),
    ):
        vertex_data.add_edge(start_id, end_id)
    return LevelData(index=2, name="Ground", vertex_data=vertex_data)


def _build_room_with_doorway_level() -> tuple[LevelData, str]:
    level = _build_square_level()
    boundary_ids = tuple(vertex.id for vertex in level.vertex_data.vertices)
    center = level.vertex_data.add_vertex(50.0, 50.0)
    level.rooms = [
        RoomData(
            name="Room",
            vertex_ids=boundary_ids,
            center_vertex_id=center.id,
            color_rgb=(120, 140, 160),
        )
    ]
    level.doorways = [
        DoorwayData(
            center_x=50.0,
            center_y=0.0,
            width_meters=0.8,
            height_meters=2.0,
            depth_meters=0.2,
            rotation_degrees=90.0,
        )
    ]
    return level, f"level:2/room:{center.id}/wall:1:2"


# ### Persistence tests ###
class SurfaceOrientationPersistenceTests(unittest.TestCase):
    def test_level_defaults_do_not_share_flip_state(self) -> None:
        first = LevelData(index=1, name="First")
        second = LevelData(index=2, name="Second")

        first.flipped_surface_ids.add("level:1/wall:1:2")

        self.assertEqual(second.flipped_surface_ids, set())

    def test_project_round_trip_preserves_manual_surface_flips(self) -> None:
        level = _build_square_level()
        level.flipped_surface_ids = {
            "level:2/wall:3:4",
            "level:2/wall:1:2",
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "orientation.housemaker"
            save_project(project_path, level.index, [level])
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            restored = load_project(project_path)

        self.assertEqual(
            payload["levels"][0]["flipped_surface_ids"],
            ["level:2/wall:1:2", "level:2/wall:3:4"],
        )
        restored_level = next(
            candidate for candidate in restored.levels if candidate.index == level.index
        )
        self.assertEqual(restored_level.flipped_surface_ids, level.flipped_surface_ids)

    def test_project_load_isolates_malformed_and_cross_level_flip_ids(self) -> None:
        level = _build_square_level()
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "orientation.housemaker"
            save_project(project_path, level.index, [level])
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            payload["levels"][0]["flipped_surface_ids"] = [
                " LEVEL:2/WALL:1:2 ",
                "level:2/wall:1:2",
                "level:3/wall:1:2",
                "level:2/",
                42,
            ]
            project_path.write_text(json.dumps(payload), encoding="utf-8")

            restored = load_project(project_path)

        restored_level = next(
            candidate for candidate in restored.levels if candidate.index == level.index
        )
        self.assertEqual(
            restored_level.flipped_surface_ids,
            {"level:2/wall:1:2"},
        )


# ### Orientation operation tests ###
class SurfaceOrientationOperationTests(unittest.TestCase):
    def test_flip_operation_toggles_one_stable_surface_id(self) -> None:
        level = _build_square_level()
        surface_id = "level:2/wall:1:2"

        first_result = flip_surface_orientation((level,), surface_id)
        second_result = flip_surface_orientation((level,), surface_id)

        self.assertEqual(first_result.selected_surface_ids, (surface_id,))
        self.assertTrue(first_result.requires_mesh_refresh)
        self.assertTrue(first_result.state_changed)
        self.assertEqual(second_result.selected_surface_ids, (surface_id,))
        self.assertEqual(level.flipped_surface_ids, set())

    def test_unknown_surface_does_not_change_persistent_state(self) -> None:
        level = _build_square_level()
        existing = {"level:2/wall:1:2"}
        level.flipped_surface_ids = set(existing)

        with self.assertRaisesRegex(ValueError, "no longer exists"):
            flip_surface_orientation((level,), "level:2/wall:999:1000")

        self.assertEqual(level.flipped_surface_ids, existing)

    def test_flip_operation_reverses_rendered_surface_winding(self) -> None:
        level = _build_square_level()
        surface_id = "level:2/wall:1:2"
        original = next(
            surface
            for surface in build_fixed_surfaces((level,))
            if surface.surface_id == surface_id
        )

        flip_surface_orientation((level,), surface_id)
        flipped = next(
            surface
            for surface in build_fixed_surfaces((level,))
            if surface.surface_id == surface_id
        )

        original_normal = np.average(
            original.mesh.face_normals,
            axis=0,
            weights=original.mesh.area_faces,
        )
        flipped_normal = np.average(
            flipped.mesh.face_normals,
            axis=0,
            weights=flipped.mesh.area_faces,
        )
        np.testing.assert_allclose(flipped_normal, -original_normal)

    def test_wall_flip_preserves_connected_doorway_reveal_winding(self) -> None:
        level, surface_id = _build_room_with_doorway_level()
        original = next(
            surface
            for surface in build_fixed_surfaces((level,))
            if surface.surface_id == surface_id
        )
        original_normals = np.asarray(original.mesh.face_normals, dtype=float)
        wall_faces = np.abs(original_normals[:, 1]) > 0.9
        reveal_faces = ~wall_faces
        self.assertTrue(np.any(wall_faces))
        self.assertTrue(np.any(reveal_faces))

        flip_surface_orientation((level,), surface_id)
        flipped = next(
            surface
            for surface in build_fixed_surfaces((level,))
            if surface.surface_id == surface_id
        )
        flipped_normals = np.asarray(flipped.mesh.face_normals, dtype=float)

        np.testing.assert_allclose(
            flipped_normals[wall_faces],
            -original_normals[wall_faces],
        )
        np.testing.assert_allclose(
            flipped_normals[reveal_faces],
            original_normals[reveal_faces],
        )

    def test_horizontal_surface_flip_reverses_every_face(self) -> None:
        level = _build_square_level()
        surface_id = "level:2/floor"
        original = next(
            surface
            for surface in build_fixed_surfaces((level,))
            if surface.surface_id == surface_id
        )

        flip_surface_orientation((level,), surface_id)
        flipped = next(
            surface
            for surface in build_fixed_surfaces((level,))
            if surface.surface_id == surface_id
        )

        np.testing.assert_allclose(
            flipped.mesh.face_normals,
            -original.mesh.face_normals,
        )

    def test_editable_flip_follows_surface_lineage(self) -> None:
        level = _build_square_level()
        source_id = (
            "level:2/edit-face:11111111111111111111111111111111:wall"
        )
        child_ids = (
            "level:2/edit-face:22222222222222222222222222222222:wall",
            "level:2/edit-face:33333333333333333333333333333333:wall",
        )
        root_id = "level:2/wall:1:2"
        level.flipped_surface_ids = {source_id, root_id}

        remap_flipped_surface_ids_with_lineage(
            (level,),
            {source_id: child_ids, root_id: child_ids},
        )

        self.assertEqual(
            level.flipped_surface_ids,
            {root_id, *child_ids},
        )

    def test_editable_flip_resolves_chained_lineage_in_any_mapping_order(
        self,
    ) -> None:
        level = _build_square_level()
        source_id = (
            "level:2/edit-face:11111111111111111111111111111111:wall"
        )
        middle_id = (
            "level:2/edit-face:22222222222222222222222222222222:wall"
        )
        terminal_id = (
            "level:2/edit-face:33333333333333333333333333333333:wall"
        )
        level.flipped_surface_ids = {source_id}

        remap_flipped_surface_ids_with_lineage(
            (level,),
            {
                middle_id: (terminal_id,),
                source_id: (middle_id,),
            },
        )

        self.assertEqual(level.flipped_surface_ids, {terminal_id})


if __name__ == "__main__":
    unittest.main()
