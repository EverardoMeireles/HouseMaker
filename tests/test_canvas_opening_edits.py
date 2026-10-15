# ### Imports ###
from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
import trimesh

from housemaker.canvas_openings import (
    CANVAS_OPENING_DOORWAY,
    CANVAS_OPENING_WINDOW,
    CanvasOpeningBounds,
    CanvasOpeningEdit,
    CanvasOpeningReference,
    CanvasOpeningTarget,
    apply_canvas_opening_edit,
    build_canvas_opening_targets,
)
from housemaker.models import (
    DOORWAY_SHAPE_ARCH,
    DoorwayData,
    LevelData,
    WindowData,
)
from housemaker.surface_geometry import SURFACE_TYPE_WALL, FixedSurface


# ### Fixture helpers ###
def _build_wall() -> FixedSurface:
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (4.0, 0.0, 0.0),
            (4.0, 0.0, 3.0),
            (0.0, 0.0, 3.0),
        ),
        dtype=float,
    )
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        process=False,
    )
    return FixedSurface(
        surface_id="level:2/wall:1:2",
        surface_type=SURFACE_TYPE_WALL,
        level_index=2,
        room_index=None,
        mesh=mesh,
        area_square_meters=12.0,
        wall_key="1:2",
        wall_start_world=(0.0, 0.0, 0.0),
        wall_end_world=(4.0, 0.0, 0.0),
        wall_height_meters=3.0,
    )


def _build_level() -> tuple[LevelData, FixedSurface]:
    wall = _build_wall()
    level = LevelData(
        index=2,
        name="Ground",
        height_meters=3.0,
        doorways=[
            DoorwayData(
                center_x=100.0,
                center_y=0.0,
                width_meters=1.0,
                height_meters=1.8,
                rotation_degrees=90.0,
                bottom_height_meters=0.3,
            )
        ],
        windows=[
            WindowData(
                window_id="window-a",
                wall_surface_id=wall.surface_id,
                start_ratio=0.1,
                end_ratio=0.3,
                bottom_ratio=0.3,
                top_ratio=0.7,
            ),
            WindowData(
                window_id="window-b",
                wall_surface_id=wall.surface_id,
                start_ratio=0.6,
                end_ratio=0.8,
                bottom_ratio=0.4,
                top_ratio=0.8,
            ),
        ],
    )
    return level, wall


def _build_window_host_target(
    source: CanvasOpeningTarget,
    *,
    wall_surface_id: str,
    bounds: CanvasOpeningBounds,
    sloped_ceiling_id: str | None,
) -> CanvasOpeningTarget:
    """Build one alternate planar host for a stable window reference."""

    diagonal = float(2**-0.5)
    return CanvasOpeningTarget(
        reference=source.reference,
        wall_surface_id=wall_surface_id,
        plane_start_world=(1.0, 2.0, 2.0),
        wall_tangent_world=(1.0, 0.0, 0.0),
        wall_normal_world=(0.0, -diagonal, diagonal),
        vertical_axis_world=(0.0, diagonal, diagonal),
        wall_width_meters=5.0,
        wall_height_meters=2.0,
        minimum_width_meters=0.05,
        minimum_height_meters=0.05,
        bounds=bounds,
        sloped_ceiling_id=sloped_ceiling_id,
    )


# ### Opening target and application tests ###
class CanvasOpeningEditTests(unittest.TestCase):
    def test_planar_target_uses_its_normalized_vertical_axis(self) -> None:
        target = CanvasOpeningTarget(
            reference=CanvasOpeningReference(
                kind=CANVAS_OPENING_WINDOW,
                level_index=2,
                item_index=0,
                stable_id="sloped-window-a",
            ),
            wall_surface_id="level:2/wall:1:2",
            plane_start_world=(1.0, 2.0, 3.0),
            wall_tangent_world=(2.0, 0.0, 0.0),
            wall_normal_world=(0.0, -1.0, 1.0),
            vertical_axis_world=(0.0, 2.0, 2.0),
            wall_width_meters=4.0,
            wall_height_meters=2.0,
            minimum_width_meters=0.05,
            minimum_height_meters=0.05,
            bounds=CanvasOpeningBounds(0.1, 0.4, 0.2, 0.8),
        )

        expected = np.asarray((2.0, 2.0 + 2**-0.5, 3.0 + 2**-0.5))
        world_point = target.local_to_world(0.25, 0.5)

        np.testing.assert_allclose(target.wall_tangent_world, (1.0, 0.0, 0.0))
        np.testing.assert_allclose(
            target.vertical_axis_world,
            (0.0, 2**-0.5, 2**-0.5),
        )
        np.testing.assert_allclose(world_point, expected)
        np.testing.assert_allclose(target.world_to_local(world_point), (0.25, 0.5))

    def test_vertical_wall_target_keeps_the_legacy_global_z_axis(self) -> None:
        _level, wall = _build_level()
        target = CanvasOpeningTarget(
            reference=CanvasOpeningReference(
                kind=CANVAS_OPENING_WINDOW,
                level_index=2,
                item_index=0,
                stable_id="window-a",
            ),
            wall_surface_id=wall.surface_id,
            plane_start_world=(0.0, 0.0, 0.0),
            wall_tangent_world=(1.0, 0.0, 0.0),
            wall_normal_world=(0.0, -1.0, 0.0),
            wall_width_meters=4.0,
            wall_height_meters=3.0,
            minimum_width_meters=0.05,
            minimum_height_meters=0.05,
            bounds=CanvasOpeningBounds(0.1, 0.4, 0.2, 0.8),
        )

        self.assertEqual(target.vertical_axis_world, (0.0, 0.0, 1.0))
        np.testing.assert_allclose(
            target.local_to_world(0.25, 0.5),
            (1.0, 0.0, 1.5),
        )

    def test_targets_use_one_wall_local_frame_for_doorways_and_windows(
        self,
    ) -> None:
        level, wall = _build_level()

        targets = build_canvas_opening_targets((level,), (wall,))

        doorway_target = next(
            target
            for target in targets
            if target.reference.kind == CANVAS_OPENING_DOORWAY
        )
        window_targets = tuple(
            target
            for target in targets
            if target.reference.kind == CANVAS_OPENING_WINDOW
        )
        self.assertEqual(len(window_targets), 2)
        self.assertEqual(doorway_target.wall_surface_id, wall.surface_id)
        self.assertAlmostEqual(doorway_target.bounds.start_ratio, 0.375)
        self.assertAlmostEqual(doorway_target.bounds.end_ratio, 0.625)
        self.assertAlmostEqual(doorway_target.bounds.bottom_ratio, 0.1)
        self.assertAlmostEqual(doorway_target.bounds.top_ratio, 0.7)
        self.assertTrue(
            np.allclose(
                doorway_target.get_world_center(),
                (2.0, 0.0, 1.2),
            )
        )

    def test_vertical_target_builder_defers_sloped_ceiling_windows(self) -> None:
        level, wall = _build_level()
        level.windows.append(
            WindowData(
                window_id="sloped-window-a",
                wall_surface_id=wall.surface_id,
                start_ratio=0.2,
                end_ratio=0.5,
                bottom_ratio=0.2,
                top_ratio=0.7,
                sloped_ceiling_id="a" * 32,
            )
        )

        targets = build_canvas_opening_targets((level,), (wall,))

        self.assertNotIn(
            "sloped-window-a",
            {
                target.reference.stable_id
                for target in targets
                if target.reference.kind == CANVAS_OPENING_WINDOW
            },
        )

    def test_doorway_edit_persists_all_four_rectangle_edges(self) -> None:
        level, wall = _build_level()
        target = next(
            target
            for target in build_canvas_opening_targets((level,), (wall,))
            if target.reference.kind == CANVAS_OPENING_DOORWAY
        )
        bounds = CanvasOpeningBounds(0.25, 0.7, 0.2, 0.85)

        result = apply_canvas_opening_edit(
            (level,),
            target,
            CanvasOpeningEdit(target.reference, target.wall_surface_id, bounds),
        )

        doorway = level.doorways[0]
        self.assertIs(result.current, doorway)
        self.assertAlmostEqual(doorway.center_x, 95.0)
        self.assertAlmostEqual(doorway.center_y, 0.0)
        self.assertAlmostEqual(doorway.width_meters, 1.8)
        self.assertAlmostEqual(doorway.bottom_height_meters, 0.6)
        self.assertAlmostEqual(doorway.height_meters, 1.95)
        self.assertAlmostEqual(doorway.rotation_degrees, 90.0)

    def test_arch_target_and_edit_preserve_the_shape_and_update_its_amount(
        self,
    ) -> None:
        level, wall = _build_level()
        level.doorways[0] = replace(
            level.doorways[0],
            shape=DOORWAY_SHAPE_ARCH,
            arch_amount=0.35,
        )
        target = next(
            target
            for target in build_canvas_opening_targets((level,), (wall,))
            if target.reference.kind == CANVAS_OPENING_DOORWAY
        )

        result = apply_canvas_opening_edit(
            (level,),
            target,
            CanvasOpeningEdit(
                target.reference,
                target.wall_surface_id,
                target.bounds,
                arch_amount=0.8,
            ),
        )

        self.assertEqual(target.arch_amount, 0.35)
        self.assertEqual(result.current.shape, DOORWAY_SHAPE_ARCH)
        self.assertEqual(result.current.arch_amount, 0.8)

    def test_window_edit_resolves_the_stable_id_after_list_reordering(self) -> None:
        level, wall = _build_level()
        target = next(
            target
            for target in build_canvas_opening_targets((level,), (wall,))
            if target.reference.stable_id == "window-b"
        )
        level.windows.reverse()
        bounds = CanvasOpeningBounds(0.45, 0.75, 0.2, 0.9)

        result = apply_canvas_opening_edit(
            (level,),
            target,
            CanvasOpeningEdit(target.reference, target.wall_surface_id, bounds),
        )

        self.assertEqual(result.reference.item_index, 0)
        self.assertEqual(level.windows[0].window_id, "window-b")
        self.assertEqual(level.windows[0].start_ratio, 0.45)
        self.assertEqual(level.windows[0].top_ratio, 0.9)
        self.assertEqual(level.windows[1].window_id, "window-a")

    def test_window_edit_preserves_its_sloped_ceiling_binding(self) -> None:
        level, wall = _build_level()
        slope_id = "a" * 32
        level.windows = [
            replace(level.windows[0], sloped_ceiling_id=slope_id)
        ]
        target = CanvasOpeningTarget(
            reference=CanvasOpeningReference(
                kind=CANVAS_OPENING_WINDOW,
                level_index=level.index,
                item_index=0,
                stable_id=level.windows[0].window_id,
            ),
            wall_surface_id=wall.surface_id,
            plane_start_world=(0.0, 0.0, 2.0),
            wall_tangent_world=(1.0, 0.0, 0.0),
            wall_normal_world=(0.0, -2**-0.5, 2**-0.5),
            vertical_axis_world=(0.0, 2**-0.5, 2**-0.5),
            wall_width_meters=4.0,
            wall_height_meters=2.0,
            minimum_width_meters=0.05,
            minimum_height_meters=0.05,
            bounds=CanvasOpeningBounds(0.1, 0.3, 0.3, 0.7),
            sloped_ceiling_id=slope_id,
        )
        bounds = CanvasOpeningBounds(0.2, 0.6, 0.15, 0.75)

        result = apply_canvas_opening_edit(
            (level,),
            target,
            CanvasOpeningEdit(target.reference, target.wall_surface_id, bounds),
        )

        self.assertEqual(result.current.sloped_ceiling_id, slope_id)
        self.assertEqual(result.current.start_ratio, 0.2)
        self.assertEqual(result.current.top_ratio, 0.75)

    def test_window_edit_can_move_from_a_wall_to_a_sloped_host(self) -> None:
        level, wall = _build_level()
        source = next(
            target
            for target in build_canvas_opening_targets((level,), (wall,))
            if target.reference.stable_id == "window-a"
        )
        bounds = CanvasOpeningBounds(0.25, 0.55, 0.15, 0.65)
        slope_id = "b" * 32
        destination = _build_window_host_target(
            source,
            wall_surface_id="level:2/wall:3:4",
            bounds=bounds,
            sloped_ceiling_id=slope_id,
        )
        edit = CanvasOpeningEdit(
            reference=source.reference,
            wall_surface_id=destination.wall_surface_id,
            bounds=bounds,
            destination_target=destination,
        )

        result = apply_canvas_opening_edit((level,), source, edit)

        self.assertIs(source.with_edit(edit), destination)
        self.assertEqual(result.previous.window_id, "window-a")
        self.assertEqual(result.current.window_id, "window-a")
        self.assertEqual(result.current.wall_surface_id, destination.wall_surface_id)
        self.assertEqual(result.current.sloped_ceiling_id, slope_id)
        self.assertEqual(result.current.start_ratio, bounds.start_ratio)
        self.assertEqual(result.current.top_ratio, bounds.top_ratio)

    def test_window_edit_can_move_from_a_slope_back_to_a_wall(self) -> None:
        level, wall = _build_level()
        slope_id = "c" * 32
        reference_target = next(
            target
            for target in build_canvas_opening_targets((level,), (wall,))
            if target.reference.stable_id == "window-a"
        )
        level.windows[0] = replace(
            level.windows[0],
            wall_surface_id="level:2/wall:3:4",
            sloped_ceiling_id=slope_id,
        )
        initial_bounds = CanvasOpeningBounds(0.1, 0.3, 0.3, 0.7)
        source = _build_window_host_target(
            reference_target,
            wall_surface_id=level.windows[0].wall_surface_id,
            bounds=initial_bounds,
            sloped_ceiling_id=slope_id,
        )
        destination_bounds = CanvasOpeningBounds(0.35, 0.65, 0.2, 0.8)
        destination = replace(
            source,
            wall_surface_id=wall.surface_id,
            plane_start_world=(0.0, 0.0, 0.0),
            wall_normal_world=(0.0, -1.0, 0.0),
            vertical_axis_world=(0.0, 0.0, 1.0),
            wall_width_meters=4.0,
            wall_height_meters=3.0,
            bounds=destination_bounds,
            sloped_ceiling_id=None,
        )
        edit = CanvasOpeningEdit(
            reference=source.reference,
            wall_surface_id=destination.wall_surface_id,
            bounds=destination_bounds,
            destination_target=destination,
        )

        result = apply_canvas_opening_edit((level,), source, edit)

        self.assertEqual(result.previous.sloped_ceiling_id, slope_id)
        self.assertEqual(result.current.window_id, result.previous.window_id)
        self.assertEqual(result.current.wall_surface_id, wall.surface_id)
        self.assertIsNone(result.current.sloped_ceiling_id)
        self.assertEqual(result.current.bottom_ratio, 0.2)
        self.assertEqual(result.current.top_ratio, 0.8)

    def test_window_rehost_destination_must_exactly_match_the_edit(self) -> None:
        level, wall = _build_level()
        source = next(
            target
            for target in build_canvas_opening_targets((level,), (wall,))
            if target.reference.stable_id == "window-a"
        )
        bounds = CanvasOpeningBounds(0.25, 0.55, 0.15, 0.65)
        destination = _build_window_host_target(
            source,
            wall_surface_id="level:2/wall:3:4",
            bounds=bounds,
            sloped_ceiling_id="d" * 32,
        )
        mismatched_reference = replace(
            destination.reference,
            stable_id="another-window",
        )

        invalid_edits = (
            {
                "wall_surface_id": source.wall_surface_id,
                "bounds": bounds,
                "destination_target": destination,
            },
            {
                "wall_surface_id": destination.wall_surface_id,
                "bounds": source.bounds,
                "destination_target": destination,
            },
            {
                "wall_surface_id": destination.wall_surface_id,
                "bounds": bounds,
                "destination_target": replace(
                    destination,
                    reference=mismatched_reference,
                ),
            },
        )
        for arguments in invalid_edits:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                CanvasOpeningEdit(
                    reference=source.reference,
                    **arguments,
                )

    def test_doorways_reject_destination_hosts_and_sloped_targets(self) -> None:
        level, wall = _build_level()
        doorway_target = next(
            target
            for target in build_canvas_opening_targets((level,), (wall,))
            if target.reference.kind == CANVAS_OPENING_DOORWAY
        )

        with self.assertRaises(ValueError):
            CanvasOpeningEdit(
                reference=doorway_target.reference,
                wall_surface_id=doorway_target.wall_surface_id,
                bounds=doorway_target.bounds,
                arch_amount=doorway_target.arch_amount,
                destination_target=doorway_target,
            )
        with self.assertRaises(ValueError):
            replace(doorway_target, sloped_ceiling_id="e" * 32)

    def test_edits_reject_bounds_outside_the_wall_and_below_minimum_size(
        self,
    ) -> None:
        level, wall = _build_level()
        target = next(
            target
            for target in build_canvas_opening_targets((level,), (wall,))
            if target.reference.stable_id == "window-a"
        )

        for bounds in (
            CanvasOpeningBounds(-0.1, 0.2, 0.2, 0.7),
            CanvasOpeningBounds(0.2, 0.205, 0.2, 0.7),
            CanvasOpeningBounds(0.2, 0.7, 0.2, 0.205),
        ):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                apply_canvas_opening_edit(
                    (level,),
                    target,
                    CanvasOpeningEdit(
                        target.reference,
                        target.wall_surface_id,
                        bounds,
                    ),
                )


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
