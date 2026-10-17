# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import unittest

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from housemaker.door_state import (
    DoorLibraryData,
    WindowPlacement,
    create_window_definition_for_frame,
)
from housemaker.models import WindowData
from housemaker.surface_geometry import SURFACE_TYPE_WALL, build_fixed_surfaces
from tests import test_canvas_connection_severance as severance_tests


# ### Pending connection-cut fixture helpers ###
class ConnectionSeveranceWindowUndoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = severance_tests.CanvasConnectionSeveranceMainTests()
        self.fixture.setUp()
        self.workspace = self.fixture.workspace
        self.level = self.fixture.level
        walls = sorted(
            (
                surface
                for surface in build_fixed_surfaces((self.level,))
                if surface.surface_type == SURFACE_TYPE_WALL
            ),
            key=lambda surface: surface.wall_key,
        )
        self.frames = [
            WindowData(
                window_id=f"frame-{index}",
                wall_surface_id=wall.surface_id,
                start_ratio=0.25,
                end_ratio=0.75,
                bottom_ratio=0.2,
                top_ratio=0.7,
            )
            for index, wall in enumerate(walls)
        ]
        self.level.windows = list(self.frames)
        self.workspace._sync_canvas_to_current_level()
        self.assets = [
            create_window_definition_for_frame(
                frame.window_id,
                1.0,
                1.0,
                name=f"Reusable window {index}",
                window_id=f"window-{index}",
            )
            for index, frame in enumerate(self.frames)
        ]
        self.placements = [
            WindowPlacement(
                placement_id=f"placement-{index}",
                window_id=asset.window_id,
                level_index=self.level.index,
                window_frame_id=frame.window_id,
            )
            for index, (asset, frame) in enumerate(zip(self.assets, self.frames))
        ]
        self.workspace.doors_workspace.set_data(
            DoorLibraryData(
                windows=list(self.assets),
                window_placements=list(self.placements),
            )
        )
        self.workspace._canvas_undo_stack.clear()

    def tearDown(self) -> None:
        self.fixture.tearDown()

    def _cut_wall(self, vertex_ids: tuple[int, int]) -> None:
        self.workspace.canvas.set_selected_vertex_ids(vertex_ids)
        QTest.keyClick(self.workspace.canvas, Qt.Key.Key_Delete)

    def _assert_only_second_wall_restored(self) -> None:
        self.assertFalse(self.level.vertex_data.has_edge(1, 2))
        self.assertTrue(self.level.vertex_data.has_edge(3, 4))
        self._assert_frames([self.frames[1]])
        self.assertEqual(
            self.workspace.doors_workspace.data().window_placements,
            [self.placements[1]],
        )
        self.assertEqual(
            self.workspace.doors_workspace.data().windows,
            self.assets,
        )

    def _assert_both_walls_restored(self) -> None:
        self.assertTrue(self.level.vertex_data.has_edge(1, 2))
        self.assertTrue(self.level.vertex_data.has_edge(3, 4))
        self._assert_frames(self.frames)
        self.assertCountEqual(
            self.workspace.doors_workspace.data().window_placements,
            self.placements,
        )

    def _assert_frames(self, expected: list[WindowData]) -> None:
        self.assertEqual(len(self.level.windows), len(expected))
        for actual, original in zip(self.level.windows, expected):
            self.assertEqual(actual.window_id, original.window_id)
            self.assertEqual(actual.wall_surface_id, original.wall_surface_id)
            for field in (
                "start_ratio", "end_ratio", "bottom_ratio", "top_ratio"
            ):
                self.assertAlmostEqual(
                    getattr(actual, field), getattr(original, field)
                )

    # ### Independent window undo within one delayed cut burst ###
    def test_committed_cut_burst_restores_each_frame_and_fixture_independently(
        self,
    ) -> None:
        self._cut_wall((1, 2))
        self._cut_wall((3, 4))
        self.workspace._commit_pending_wall_vertex_update()
        self.assertEqual(self.level.windows, [])
        self.assertEqual(
            self.workspace.doors_workspace.data().window_placements, []
        )

        self.workspace._handle_canvas_undo_requested()
        self._assert_only_second_wall_restored()

        self.workspace._handle_canvas_undo_requested()
        self._assert_both_walls_restored()

    def test_uncommitted_cut_burst_does_not_restore_an_earlier_cuts_frame(
        self,
    ) -> None:
        self._cut_wall((1, 2))
        self._cut_wall((3, 4))
        self.assertTrue(self.workspace._wall_vertex_update_timer.isActive())

        self.workspace._handle_canvas_undo_requested()
        self._assert_only_second_wall_restored()

        self.workspace._handle_canvas_undo_requested()
        self._assert_both_walls_restored()

    def test_separate_commits_preserve_existing_window_undo_behavior(self) -> None:
        self._cut_wall((1, 2))
        self.workspace._commit_pending_wall_vertex_update()
        self._cut_wall((3, 4))
        self.workspace._commit_pending_wall_vertex_update()

        self.workspace._handle_canvas_undo_requested()
        self._assert_only_second_wall_restored()

        self.workspace._handle_canvas_undo_requested()
        self._assert_both_walls_restored()


# ### Direct test runner ###
if __name__ == "__main__":
    unittest.main()
