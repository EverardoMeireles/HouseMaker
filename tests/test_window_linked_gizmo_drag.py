"""Drag the actual green 3D window handle over its noded wall run."""

# ### Environment setup ###
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import numpy as np
from PySide6.QtCore import QPointF

from housemaker.viewer import (
    CANVAS_OPENING_GIZMO_ANCHOR,
    _CanvasOpeningGizmoHandle,
)
from tests import test_window_linked_opening_edits as linked_tests


# ### Linked scene fixture ###
class LinkedWindowGizmoDragTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = linked_tests.LinkedWindowOpeningEditTests()
        self.fixture.setUp()
        self.workspace = self.fixture.workspace
        self.level = self.fixture.level
        self.viewer = self.workspace.viewer
        self.target = self.fixture.target
        self.window = self.fixture.window
        self.before_graph = self.fixture.before_graph
        self.viewer.select_canvas_opening(None)
        self.assertTrue(self.viewer.select_canvas_opening(self.target.reference))
        self.handle = _CanvasOpeningGizmoHandle(CANVAS_OPENING_GIZMO_ANCHOR)
        self.center = np.asarray(self.target.get_world_center(), dtype=float)

    def tearDown(self) -> None:
        self.fixture.tearDown()

    # ### Pointer rays and assertions ###
    def _ray(self, dx=0.0, dz=0.0):
        point = self.center + np.asarray((dx, 0.0, dz), dtype=float)
        # This camera looks toward the near (y=50px) face from the room side.
        # The ray picker, logical wall rectangles and rehosting remain real.
        return point + np.asarray((0.0, 5.0, 0.0)), np.asarray((0.0, -1.0, 0.0))

    def _begin(self):
        with patch.object(
            self.viewer.view, "build_camera_ray", return_value=self._ray()
        ):
            self.assertTrue(
                self.viewer._begin_canvas_opening_gizmo_drag(self.handle, QPointF())
            )
        self.assertTrue(self.viewer.view.is_primary_pointer_drag_reserved)

    def _update(self, dx=0.0, dz=0.0):
        with patch.object(
            self.viewer.view, "build_camera_ray", return_value=self._ray(dx, dz)
        ):
            return self.viewer._update_canvas_opening_gizmo_drag(QPointF())

    def _finish(self, dx=0.0, dz=0.0):
        with patch.object(
            self.viewer.view, "build_camera_ray", return_value=self._ray(dx, dz)
        ):
            changed = self.viewer._finish_canvas_opening_gizmo_drag(QPointF())
        self.assertFalse(self.viewer.view.is_primary_pointer_drag_reserved)
        return changed

    def _assert_moved(self, dx=0.0, dz=0.0):
        self.fixture._assert_horizontal_movement(dx / self.level.meters_per_pixel)
        window = self.level.windows[0]
        self.assertAlmostEqual(
            window.bottom_ratio,
            self.window.bottom_ratio + dz / self.level.height_meters,
        )
        self.assertAlmostEqual(
            window.top_ratio, self.window.top_ratio + dz / self.level.height_meters
        )
        live_target = self.workspace._canvas_opening_targets_by_key[self.target.key]
        self.assertAlmostEqual(
            live_target.wall_width_meters, self.target.wall_width_meters
        )
        self.assertEqual(live_target.wall_surface_id, self.target.wall_surface_id)
        self.assertIsNone(window.sloped_ceiling_id)

    # ### Actual green-handle dragging ###
    def test_horizontal_drag_inside_original_hole_keeps_full_wall_frame(self):
        self._begin()

        self.assertTrue(self._finish(0.2))

        self._assert_moved(0.2)

    def test_drag_crosses_old_jamb_into_adjacent_connected_wall_segment(self):
        self._begin()

        self.assertTrue(self._finish(0.6))

        self._assert_moved(0.6)
        self.assertEqual(len(self.workspace._canvas_undo_stack), 1)

    def test_diagonal_drag_moves_all_corners_and_preserves_vertical_size(self):
        self._begin()

        self.assertTrue(self._finish(0.2, 0.3))

        self._assert_moved(0.2, 0.3)
        self.assertAlmostEqual(
            (self.level.windows[0].top_ratio - self.level.windows[0].bottom_ratio)
            * self.level.height_meters,
            1.0,
        )

    def test_repeated_live_updates_use_absolute_green_handle_displacement(self):
        self._begin()
        for dx in (0.1, 0.2, 0.3):
            self.assertTrue(self._update(dx, 0.15))
            self._assert_moved(dx, 0.15)

        self.assertTrue(self._finish(0.3, 0.15))

        self._assert_moved(0.3, 0.15)
        self.assertEqual(len(self.workspace._canvas_undo_stack), 1)

    def test_back_face_camera_drag_keeps_linked_source_frame_and_moves_four_corners(
        self,
    ):
        initial_ray = (
            self.center + np.asarray((0.0, -5.0, 0.0)),
            np.asarray((0.0, 1.0, 0.0)),
        )
        with patch.object(
            self.viewer.view, "build_camera_ray", return_value=initial_ray
        ):
            self.assertTrue(
                self.viewer._begin_canvas_opening_gizmo_drag(self.handle, QPointF())
            )
        moved_ray = (
            self.center + np.asarray((0.2, -5.0, 0.0)),
            np.asarray((0.0, 1.0, 0.0)),
        )

        with patch.object(self.viewer.view, "build_camera_ray", return_value=moved_ray):
            self.assertTrue(self.viewer._finish_canvas_opening_gizmo_drag(QPointF()))

        self._assert_moved(0.2)

    def test_unmoved_oblique_back_face_pointer_does_not_jump_window_or_lose_links(self):
        offset = np.asarray((-3.0, -5.0, 1.0))
        ray = self.center + offset, -offset / np.linalg.norm(offset)
        with patch.object(self.viewer.view, "build_camera_ray", return_value=ray):
            self.assertTrue(
                self.viewer._begin_canvas_opening_gizmo_drag(self.handle, QPointF())
            )
            self.viewer._update_canvas_opening_gizmo_drag(QPointF())
            self.viewer._finish_canvas_opening_gizmo_drag(QPointF())

        self._assert_moved()
        self.assertEqual(self.level.vertex_data.to_dict(), self.before_graph)
        self.assertEqual(self.level.windows, [self.window])
        np.testing.assert_allclose(
            self.workspace._canvas_opening_targets_by_key[
                self.target.key
            ].get_world_center(),
            self.target.get_world_center(),
            atol=1e-12,
        )

    # ### Main transaction delay and restoration ###
    def test_green_handle_defers_mesh_until_release_delay_and_undo_restores_graph(self):
        with (
            patch.object(self.workspace, "_build_viewer_preview_model") as build,
            patch.object(
                self.workspace, "_reconcile_canvas_surface_edit_and_refresh"
            ) as refresh,
        ):
            self._begin()
            self.assertTrue(self._update(0.2))
            self._assert_moved(0.2)
            self.assertTrue(self.workspace._is_canvas_opening_drag_active)
            self.assertFalse(self.workspace._doorway_mesh_update_timer.isActive())
            self.workspace._queue_viewer_preview_refresh()
            self.workspace._run_scheduled_viewer_preview_refresh()
            build.assert_not_called()
            refresh.assert_not_called()

            self.assertTrue(self._finish(0.2))

            self.assertFalse(self.workspace._is_canvas_opening_drag_active)
            self.assertTrue(self.workspace._doorway_mesh_update_timer.isActive())
            refresh.assert_not_called()
            self.workspace._commit_pending_doorway_mesh_update()
            refresh.assert_called_once()

        self.workspace._handle_canvas_undo_requested()
        self.assertEqual(self.level.vertex_data.to_dict(), self.before_graph)
        self.assertEqual(self.level.windows, [self.window])
        self.assertEqual(self.workspace._canvas_undo_stack, [])

    def test_cancelling_real_green_handle_restores_exact_graph_and_window(self):
        self._begin()
        self.assertTrue(self._update(0.2, 0.15))
        self._assert_moved(0.2, 0.15)

        self.assertTrue(self.viewer._cancel_canvas_opening_edit_drag())

        self.assertEqual(self.level.vertex_data.to_dict(), self.before_graph)
        self.assertEqual(self.level.windows, [self.window])
        self.assertEqual(self.workspace._canvas_undo_stack, [])
        self.assertFalse(self.viewer.view.is_primary_pointer_drag_reserved)
        self.assertFalse(self.workspace._doorway_mesh_update_timer.isActive())


# ### Test runner ###
if __name__ == "__main__":
    unittest.main()
