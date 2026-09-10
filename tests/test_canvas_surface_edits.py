# ### Imports ###
from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest
from unittest.mock import patch

import numpy as np

from housemaker.canvas_surface_edits import (
    CANVAS_SURFACE_EDIT_AXIS_X,
    CANVAS_SURFACE_EDIT_AXIS_Y,
    CANVAS_SURFACE_EDIT_FLOOR_THICKNESS,
    CANVAS_SURFACE_EDIT_LEVEL_HEIGHT,
    CANVAS_SURFACE_EDIT_ROOM_HEIGHT,
    CANVAS_SURFACE_EDIT_WALL_VERTEX,
    CanvasSurfaceEdit,
    CanvasSurfaceEditReference,
    apply_canvas_surface_edit,
    build_canvas_surface_edit_targets,
    rebase_canvas_wall_edit_targets,
    restore_canvas_surface_edit,
    validate_canvas_surface_edit_geometry,
)
from housemaker.level_coordinates import level_image_to_world_xy
from housemaker.models import LevelData, RoomData, VertexData
from housemaker.surface_geometry import (
    SURFACE_TYPE_CEILING,
    SURFACE_TYPE_WALL,
    build_fixed_surfaces,
)


# ### Fixture helpers ###
def _build_room_level(
    *,
    scale: float = 1.0,
    offset_x_meters: float = 0.0,
    offset_y_meters: float = 0.0,
) -> LevelData:
    vertex_data = VertexData()
    first = vertex_data.add_vertex(0.0, 0.0)
    middle = vertex_data.add_vertex(50.0, 0.0)
    second = vertex_data.add_vertex(100.0, 0.0)
    third = vertex_data.add_vertex(100.0, 100.0)
    fourth = vertex_data.add_vertex(0.0, 100.0)
    center = vertex_data.add_vertex(50.0, 50.0)
    for start_id, end_id in (
        (first.id, middle.id),
        (middle.id, second.id),
        (second.id, third.id),
        (third.id, fourth.id),
        (fourth.id, first.id),
    ):
        vertex_data.add_edge(start_id, end_id)
    room = RoomData(
        name="Room",
        vertex_ids=(
            first.id,
            middle.id,
            second.id,
            third.id,
            fourth.id,
        ),
        center_vertex_id=center.id,
        color_rgb=(80, 90, 100),
        height_meters=3.0,
    )
    return LevelData(
        index=2,
        name="Ground",
        scale=scale,
        offset_x_meters=offset_x_meters,
        offset_y_meters=offset_y_meters,
        vertex_data=vertex_data,
        rooms=[room],
    )


def _build_plain_level() -> LevelData:
    vertex_data = VertexData()
    vertices = tuple(
        vertex_data.add_vertex(*point)
        for point in (
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (0.0, 100.0),
        )
    )
    for first, second in zip(vertices, (*vertices[1:], vertices[0])):
        vertex_data.add_edge(first.id, second.id)
    return LevelData(
        index=2,
        name="Ground",
        height_meters=3.0,
        vertex_data=vertex_data,
    )


def _build_adjacent_room_level() -> LevelData:
    vertex_data = VertexData()
    first = vertex_data.add_vertex(0.0, 0.0)
    shared_bottom = vertex_data.add_vertex(100.0, 0.0)
    shared_top = vertex_data.add_vertex(100.0, 100.0)
    fourth = vertex_data.add_vertex(0.0, 100.0)
    first_center = vertex_data.add_vertex(50.0, 50.0)
    sixth = vertex_data.add_vertex(200.0, 0.0)
    seventh = vertex_data.add_vertex(200.0, 100.0)
    second_center = vertex_data.add_vertex(150.0, 50.0)
    for start_id, end_id in (
        (first.id, shared_bottom.id),
        (shared_bottom.id, shared_top.id),
        (shared_top.id, fourth.id),
        (fourth.id, first.id),
        (shared_bottom.id, sixth.id),
        (sixth.id, seventh.id),
        (seventh.id, shared_top.id),
    ):
        vertex_data.add_edge(start_id, end_id)
    return LevelData(
        index=2,
        name="Ground",
        vertex_data=vertex_data,
        rooms=[
            RoomData(
                name="Left",
                vertex_ids=(
                    first.id,
                    shared_bottom.id,
                    shared_top.id,
                    fourth.id,
                ),
                center_vertex_id=first_center.id,
                color_rgb=(1, 2, 3),
            ),
            RoomData(
                name="Right",
                vertex_ids=(
                    shared_bottom.id,
                    sixth.id,
                    seventh.id,
                    shared_top.id,
                ),
                center_vertex_id=second_center.id,
                color_rgb=(4, 5, 6),
            ),
        ],
    )


def _build_targets(level: LevelData):
    surfaces = tuple(build_fixed_surfaces((level,)))
    return surfaces, build_canvas_surface_edit_targets((level,), surfaces)


def _find_target(targets, surface_suffix: str, reference_key: str):
    return next(
        target
        for target in targets
        if target.surface_id.endswith(surface_suffix)
        and target.reference.key == reference_key
    )


def _world_positions(level: LevelData) -> dict[int, tuple[float, float]]:
    return {
        vertex.id: level_image_to_world_xy(level, vertex.x, vertex.y)
        for vertex in level.vertex_data.vertices
    }


# ### Target model and builder tests ###
class CanvasSurfaceEditTargetTests(unittest.TestCase):
    def test_reference_is_immutable_and_has_a_stable_owner_key(self) -> None:
        reference = CanvasSurfaceEditReference(
            CANVAS_SURFACE_EDIT_WALL_VERTEX,
            2,
            CANVAS_SURFACE_EDIT_AXIS_X,
            vertex_id=17,
        )

        self.assertEqual(reference.key, "level:2/vertex:17/axis:x")
        with self.assertRaises(FrozenInstanceError):
            reference.vertex_id = 18  # type: ignore[misc]

    def test_room_wall_has_four_handles_and_preserves_collinear_chain(self) -> None:
        level = _build_room_level()
        _surfaces, targets = _build_targets(level)
        wall_targets = tuple(
            target
            for target in targets
            if target.surface_id.endswith("wall:1:3")
        )

        self.assertEqual(len(wall_targets), 4)
        self.assertEqual(
            {
                (target.reference.vertex_id, target.reference.axis_index)
                for target in wall_targets
            },
            {
                (1, CANVAS_SURFACE_EDIT_AXIS_X),
                (1, CANVAS_SURFACE_EDIT_AXIS_Y),
                (3, CANVAS_SURFACE_EDIT_AXIS_X),
                (3, CANVAS_SURFACE_EDIT_AXIS_Y),
            },
        )
        for target in wall_targets:
            self.assertEqual(target.chain_vertex_ids, (1, 2, 3))
            np.testing.assert_allclose(
                target.chain_vertex_ratios,
                (0.0, 0.5, 1.0),
            )

    def test_room_floor_and_ceiling_targets_name_authoritative_properties(
        self,
    ) -> None:
        level = _build_room_level()
        _surfaces, targets = _build_targets(level)
        floor_target = next(
            target
            for target in targets
            if target.surface_id.endswith("/floor")
        )
        ceiling_target = next(
            target
            for target in targets
            if target.surface_id.endswith("/ceiling")
        )

        self.assertEqual(
            floor_target.reference.kind,
            CANVAS_SURFACE_EDIT_FLOOR_THICKNESS,
        )
        self.assertEqual(floor_target.axis_world, (0.0, 0.0, 1.0))
        self.assertEqual(floor_target.baseline_value_meters, 0.3)
        self.assertEqual(floor_target.surface_id, "level:2/room:6/floor")
        self.assertEqual(
            ceiling_target.reference.kind,
            CANVAS_SURFACE_EDIT_ROOM_HEIGHT,
        )
        self.assertEqual(ceiling_target.reference.room_center_vertex_id, 6)
        self.assertEqual(ceiling_target.axis_world, (0.0, 0.0, 1.0))

    def test_inferred_level_ceiling_controls_level_height(self) -> None:
        level = _build_plain_level()
        _surfaces, targets = _build_targets(level)
        target = next(
            candidate
            for candidate in targets
            if candidate.surface_id == "level:2/ceiling"
        )

        self.assertEqual(
            target.reference.kind,
            CANVAS_SURFACE_EDIT_LEVEL_HEIGHT,
        )
        self.assertIsNone(target.reference.room_center_vertex_id)
        self.assertEqual(target.baseline_value_meters, 3.0)

    def test_shared_wall_surfaces_may_repeat_persistent_reference_keys(
        self,
    ) -> None:
        level = _build_adjacent_room_level()
        _surfaces, targets = _build_targets(level)
        shared_targets = tuple(
            target
            for target in targets
            if target.surface_id.endswith("wall:2:3")
        )

        self.assertEqual(len(shared_targets), 8)
        self.assertEqual(len({target.surface_id for target in shared_targets}), 2)
        self.assertEqual(
            len({target.reference.key for target in shared_targets}),
            4,
        )

    def test_wall_rebase_uses_current_data_as_the_next_drag_baseline(
        self,
    ) -> None:
        level = _build_room_level(
            scale=2.0,
            offset_x_meters=3.0,
            offset_y_meters=-2.0,
        )
        _surfaces, targets = _build_targets(level)
        wall_targets = tuple(
            target
            for target in targets
            if target.surface_id.endswith("wall:1:3")
        )
        first_target = _find_target(
            wall_targets,
            "wall:1:3",
            "level:2/vertex:1/axis:x",
        )
        baseline_required_ids = first_target.required_surface_ids
        baseline_ratios = first_target.chain_vertex_ratios

        apply_canvas_surface_edit(
            (level,),
            first_target,
            CanvasSurfaceEdit(
                first_target.reference,
                first_target.surface_id,
                -0.6,
            ),
            validate_project_geometry=False,
        )
        world_after_first_edit = _world_positions(level)

        with patch(
            "housemaker.canvas_surface_edits.build_fixed_surfaces",
        ) as rebuild:
            rebased_targets = rebase_canvas_wall_edit_targets(
                (level,),
                wall_targets,
            )

        rebuild.assert_not_called()
        self.assertEqual(len(rebased_targets), 4)
        current_image_positions = tuple(
            (
                level.vertex_data.get_vertex(vertex_id).x,
                level.vertex_data.get_vertex(vertex_id).y,
            )
            for vertex_id in first_target.chain_vertex_ids
        )
        for rebased_target in rebased_targets:
            self.assertEqual(
                rebased_target.chain_vertex_image_positions,
                current_image_positions,
            )
            self.assertEqual(
                rebased_target.chain_vertex_ratios,
                baseline_ratios,
            )
            self.assertEqual(
                rebased_target.required_surface_ids,
                baseline_required_ids,
            )
            self.assertEqual(
                rebased_target.level_offset_world,
                (
                    level.offset_x_meters,
                    level.offset_y_meters,
                ),
            )
            assert rebased_target.reference.vertex_id is not None
            expected_world = world_after_first_edit[
                rebased_target.reference.vertex_id
            ]
            np.testing.assert_allclose(
                rebased_target.origin_world[:2],
                expected_world,
                atol=1e-9,
            )

        second_target = _find_target(
            rebased_targets,
            "wall:1:3",
            "level:2/vertex:3/axis:y",
        )
        second_result = apply_canvas_surface_edit(
            (level,),
            second_target,
            CanvasSurfaceEdit(
                second_target.reference,
                second_target.surface_id,
                0.4,
            ),
            validate_project_geometry=False,
        )
        world_after_second_edit = _world_positions(level)

        self.assertAlmostEqual(second_result.previous.delta_meters, 0.0)
        np.testing.assert_allclose(
            world_after_second_edit[1],
            world_after_first_edit[1],
            atol=1e-9,
        )
        self.assertAlmostEqual(
            world_after_second_edit[3][1],
            world_after_first_edit[3][1] + 0.4,
        )


# ### Authoritative application tests ###
class CanvasSurfaceEditApplicationTests(unittest.TestCase):
    def test_scaled_wall_edit_is_absolute_and_preserves_other_world_points(
        self,
    ) -> None:
        level = _build_room_level(
            scale=2.0,
            offset_x_meters=3.0,
            offset_y_meters=-2.0,
        )
        _surfaces, targets = _build_targets(level)
        target = _find_target(
            targets,
            "wall:1:3",
            "level:2/vertex:1/axis:x",
        )
        baseline_vertices = tuple(level.vertex_data.vertices)
        baseline_world = _world_positions(level)
        baseline_offsets = (
            level.offset_x_meters,
            level.offset_y_meters,
        )

        first_result = apply_canvas_surface_edit(
            (level,),
            target,
            CanvasSurfaceEdit(target.reference, target.surface_id, -1.0),
            validate_project_geometry=False,
        )

        first_world = _world_positions(level)
        self.assertAlmostEqual(first_world[1][0], baseline_world[1][0] - 1.0)
        self.assertAlmostEqual(first_world[1][1], baseline_world[1][1])
        self.assertAlmostEqual(first_world[2][0], baseline_world[2][0] - 0.5)
        for vertex_id in (3, 4, 5, 6):
            np.testing.assert_allclose(
                first_world[vertex_id],
                baseline_world[vertex_id],
                atol=1e-9,
            )
        self.assertEqual(first_result.previous.delta_meters, 0.0)
        self.assertEqual(first_result.current.delta_meters, -1.0)
        self.assertNotEqual(
            (level.offset_x_meters, level.offset_y_meters),
            baseline_offsets,
        )

        second_result = apply_canvas_surface_edit(
            (level,),
            target,
            CanvasSurfaceEdit(target.reference, target.surface_id, -0.4),
            validate_project_geometry=False,
        )

        second_world = _world_positions(level)
        self.assertAlmostEqual(second_result.previous.delta_meters, -1.0)
        self.assertAlmostEqual(second_world[1][0], baseline_world[1][0] - 0.4)
        self.assertAlmostEqual(second_world[2][0], baseline_world[2][0] - 0.2)

        restore_canvas_surface_edit(
            (level,),
            target,
            validate_project_geometry=False,
        )

        self.assertEqual(tuple(level.vertex_data.vertices), baseline_vertices)
        np.testing.assert_allclose(
            (level.offset_x_meters, level.offset_y_meters),
            baseline_offsets,
            atol=1e-9,
        )

    def test_room_and_level_ceiling_edits_use_stable_owners(self) -> None:
        room_level = _build_room_level()
        _room_surfaces, room_targets = _build_targets(room_level)
        room_target = next(
            target
            for target in room_targets
            if target.reference.kind == CANVAS_SURFACE_EDIT_ROOM_HEIGHT
        )
        room_level.rooms.insert(
            0,
            RoomData(
                name="Other",
                vertex_ids=(1, 3, 4),
                center_vertex_id=5,
                color_rgb=(0, 0, 0),
                height_meters=8.0,
            ),
        )

        apply_canvas_surface_edit(
            (room_level,),
            room_target,
            CanvasSurfaceEdit(
                room_target.reference,
                room_target.surface_id,
                0.75,
            ),
            validate_project_geometry=False,
        )

        edited_room = next(
            room
            for room in room_level.rooms
            if room.center_vertex_id == 6
        )
        self.assertAlmostEqual(edited_room.height_meters, 3.75)
        self.assertEqual(room_level.rooms[0].height_meters, 8.0)

        plain_level = _build_plain_level()
        _plain_surfaces, plain_targets = _build_targets(plain_level)
        level_target = next(
            target
            for target in plain_targets
            if target.reference.kind == CANVAS_SURFACE_EDIT_LEVEL_HEIGHT
        )
        apply_canvas_surface_edit(
            (plain_level,),
            level_target,
            CanvasSurfaceEdit(
                level_target.reference,
                level_target.surface_id,
                1.25,
            ),
        )
        self.assertAlmostEqual(plain_level.height_meters, 4.25)

    def test_wall_minimum_length_failure_is_atomic(self) -> None:
        level = _build_room_level()
        _surfaces, targets = _build_targets(level)
        target = _find_target(
            targets,
            "wall:1:3",
            "level:2/vertex:1/axis:x",
        )
        baseline_vertices = tuple(level.vertex_data.vertices)
        baseline_offsets = (
            level.offset_x_meters,
            level.offset_y_meters,
        )

        with self.assertRaisesRegex(ValueError, "at least"):
            apply_canvas_surface_edit(
                (level,),
                target,
                CanvasSurfaceEdit(target.reference, target.surface_id, 1.96),
                validate_project_geometry=False,
            )

        self.assertEqual(tuple(level.vertex_data.vertices), baseline_vertices)
        self.assertEqual(
            (level.offset_x_meters, level.offset_y_meters),
            baseline_offsets,
        )

    def test_full_surface_validation_is_optional_and_rolls_back_failure(
        self,
    ) -> None:
        level = _build_room_level()
        _surfaces, targets = _build_targets(level)
        target = next(
            candidate
            for candidate in targets
            if candidate.reference.kind == CANVAS_SURFACE_EDIT_ROOM_HEIGHT
        )

        with patch(
            "housemaker.canvas_surface_edits.build_fixed_surfaces",
        ) as rebuild:
            apply_canvas_surface_edit(
                (level,),
                target,
                CanvasSurfaceEdit(target.reference, target.surface_id, 0.2),
                validate_project_geometry=False,
            )
        rebuild.assert_not_called()
        self.assertAlmostEqual(level.rooms[0].height_meters, 3.2)

        with (
            patch(
                "housemaker.canvas_surface_edits.build_fixed_surfaces",
                return_value=(),
            ),
            self.assertRaisesRegex(ValueError, "required generated surfaces"),
        ):
            apply_canvas_surface_edit(
                (level,),
                target,
                CanvasSurfaceEdit(target.reference, target.surface_id, 0.4),
            )
        self.assertAlmostEqual(level.rooms[0].height_meters, 3.2)

    def test_current_geometry_validation_does_not_replay_an_edit(self) -> None:
        level = _build_plain_level()
        _surfaces, targets = _build_targets(level)
        target = _find_target(
            targets,
            "wall:1:2",
            "level:2/vertex:2/axis:x",
        )
        apply_canvas_surface_edit(
            (level,),
            target,
            CanvasSurfaceEdit(target.reference, target.surface_id, 0.25),
            validate_project_geometry=False,
        )
        vertices_before_validation = tuple(level.vertex_data.vertices)
        offsets_before_validation = (
            level.offset_x_meters,
            level.offset_y_meters,
        )

        with patch(
            "housemaker.canvas_surface_edits.build_fixed_surfaces",
            wraps=build_fixed_surfaces,
        ) as rebuild:
            validate_canvas_surface_edit_geometry((level,), target)

        rebuild.assert_called_once_with((level,))
        self.assertEqual(
            tuple(level.vertex_data.vertices),
            vertices_before_validation,
        )
        self.assertEqual(
            (level.offset_x_meters, level.offset_y_meters),
            offsets_before_validation,
        )

        with (
            patch(
                "housemaker.canvas_surface_edits.build_fixed_surfaces",
                return_value=(),
            ),
            self.assertRaisesRegex(ValueError, "required generated surfaces"),
        ):
            validate_canvas_surface_edit_geometry((level,), target)
        self.assertEqual(
            tuple(level.vertex_data.vertices),
            vertices_before_validation,
        )

# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
