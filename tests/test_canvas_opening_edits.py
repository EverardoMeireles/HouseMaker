# ### Imports ###
from __future__ import annotations

import unittest

import numpy as np
import trimesh

from housemaker.canvas_openings import (
    CANVAS_OPENING_DOORWAY,
    CANVAS_OPENING_WINDOW,
    CanvasOpeningBounds,
    CanvasOpeningEdit,
    apply_canvas_opening_edit,
    build_canvas_opening_targets,
)
from housemaker.models import DoorwayData, LevelData, WindowData
from housemaker.surface_geometry import FixedSurface, SURFACE_TYPE_WALL


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


# ### Opening target and application tests ###
class CanvasOpeningEditTests(unittest.TestCase):
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
