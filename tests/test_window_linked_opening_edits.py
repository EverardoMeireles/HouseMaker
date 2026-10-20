"""Four-corner windows move along their full wall run without losing links."""

# ### Environment setup ###
from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.canvas_openings import (
    CanvasOpeningEdit,
    apply_canvas_opening_edit,
    build_canvas_opening_targets,
    capture_canvas_window_dimensions,
    preserve_canvas_window_dimensions,
)
from housemaker.main import BlueprintWorkspace
from housemaker.project_io import load_project, save_project
from housemaker.surface_geometry import build_fixed_surfaces
from tests.test_window_solid_fit_canvas import (
    _application,
    _configure_canvas,
    _hover,
    _solid_level,
)


# ### Fixture helpers ###
def _window_target(level):
    return build_canvas_opening_targets([level], build_fixed_surfaces([level]))[0]


def _edit(target, *, horizontal=0.0, vertical=0.0, widen=0.0):
    bounds = replace(
        target.bounds,
        start_ratio=target.bounds.start_ratio + horizontal - widen,
        end_ratio=target.bounds.end_ratio + horizontal + widen,
        bottom_ratio=target.bounds.bottom_ratio + vertical,
        top_ratio=target.bounds.top_ratio + vertical,
    )
    return CanvasOpeningEdit(target.reference, target.wall_surface_id, bounds)


def _send_drag_move(canvas, point):
    event = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(point),
        QPointF(canvas.mapToGlobal(point.toPoint())),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(canvas, event)


# ### Linked window edit integration tests ###
class LinkedWindowOpeningEditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(
                Path(self.temporary.name) / "settings.json"
            )
        )
        self.workspace._set_mesh_edit_update_delay_seconds(60.0)
        self.level = _solid_level()
        self.workspace.levels = [self.level]
        self.workspace.current_level_index = 0
        self.workspace._sync_canvas_to_current_level()
        self.workspace._reset_viewer_doorway_snapshots()
        _configure_canvas(self.workspace.canvas, self.level)
        self.placement = _hover(self.workspace.canvas)
        self.workspace._handle_canvas_window_placement_requested(self.placement)
        self.workspace._commit_pending_doorway_mesh_update()
        self.workspace.canvas.cancel_window_placement()
        self.workspace._set_canvas_viewer_targets(build_fixed_surfaces([self.level]))
        self.workspace._canvas_undo_stack.clear()
        self.window = self.level.windows[0]
        self.target = _window_target(self.level)
        self.before_graph = self.level.vertex_data.to_dict()
        self.before_edges = tuple(self.level.vertex_data.edges)
        self.corner_ids = self.placement.bridge_corner_vertex_ids
        assert self.corner_ids is not None
        self.before_positions = {
            vertex.id: (vertex.x, vertex.y)
            for vertex in self.level.vertex_data.vertices
        }

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _application.processEvents()
        self.temporary.cleanup()

    def _assert_horizontal_movement(self, delta, *, width_delta=0.0):
        corner_ids = self.corner_ids
        low_ids = {corner_ids[0], corner_ids[2]}
        high_ids = {corner_ids[1], corner_ids[3]}
        if (
            self.before_positions[corner_ids[0]][0]
            > self.before_positions[corner_ids[1]][0]
        ):
            low_ids, high_ids = high_ids, low_ids
        for vertex in self.level.vertex_data.vertices:
            old_x, old_y = self.before_positions[vertex.id]
            expected_x = old_x
            if vertex.id in low_ids:
                expected_x += delta - width_delta
            elif vertex.id in high_ids:
                expected_x += delta + width_delta
            self.assertAlmostEqual(vertex.x, expected_x, places=9)
            self.assertAlmostEqual(vertex.y, old_y, places=9)
        self.assertEqual(tuple(self.level.vertex_data.edges), self.before_edges)
        window = self.level.windows[0]
        self.assertEqual((window.start_ratio, window.end_ratio), (0.0, 1.0))
        self.assertEqual(set(window.corner_vertex_ids or ()), set(corner_ids))
        rebuilt = _window_target(self.level)
        self.assertAlmostEqual(
            rebuilt.bounds.horizontal_span * rebuilt.wall_width_meters,
            1.0 + width_delta * 2.0 * self.level.meters_per_pixel,
        )

    # ### Full run targets and persistent corner links ###
    def test_new_window_target_uses_full_wall_run_not_its_window_span(self):
        self.assertAlmostEqual(self.target.wall_width_meters, 2.4)
        self.assertAlmostEqual(self.target.bounds.start_ratio, 0.25)
        self.assertAlmostEqual(self.target.bounds.end_ratio, 2.0 / 3.0)
        self.assertEqual(set(self.window.corner_vertex_ids or ()), set(self.corner_ids))

    def test_horizontal_3d_edit_moves_all_four_corners_and_retains_connections(self):
        edit = _edit(self.target, horizontal=0.1)

        apply_canvas_opening_edit([self.level], self.target, edit)

        self._assert_horizontal_movement(12.0)
        self.assertEqual(
            (self.level.windows[0].bottom_ratio, self.level.windows[0].top_ratio),
            (self.window.bottom_ratio, self.window.top_ratio),
        )

    def test_vertical_3d_edit_keeps_footprint_and_exact_window_dimensions(self):
        edit = _edit(self.target, vertical=0.1)

        apply_canvas_opening_edit([self.level], self.target, edit)

        self.assertEqual(self.level.vertex_data.to_dict(), self.before_graph)
        self.assertAlmostEqual(
            self.level.windows[0].bottom_ratio, self.window.bottom_ratio + 0.1
        )
        self.assertAlmostEqual(
            self.level.windows[0].top_ratio, self.window.top_ratio + 0.1
        )
        rebuilt = _window_target(self.level)
        self.assertAlmostEqual(
            rebuilt.bounds.horizontal_span * rebuilt.wall_width_meters, 1.0
        )
        self.assertAlmostEqual(
            rebuilt.bounds.vertical_span * rebuilt.wall_height_meters, 1.0
        )

    def test_repeated_previews_use_absolute_bounds_instead_of_accumulating_movement(
        self,
    ):
        for delta in (0.025, 0.05, 0.1):
            apply_canvas_opening_edit(
                [self.level], self.target, _edit(self.target, horizontal=delta)
            )

        self._assert_horizontal_movement(12.0)

    def test_width_resize_moves_front_and_back_endpoints_equally(self):
        apply_canvas_opening_edit(
            [self.level], self.target, _edit(self.target, widen=0.025)
        )

        self._assert_horizontal_movement(0.0, width_delta=3.0)

    def test_height_preservation_keeps_linked_window_ratios_local_to_its_own_span(self):
        snapshots = capture_canvas_window_dimensions(
            [self.level], build_fixed_surfaces([self.level])
        )
        self.level.height_meters = 4.25

        preserve_canvas_window_dimensions(
            [self.level], build_fixed_surfaces([self.level]), snapshots
        )

        window = self.level.windows[0]
        self.assertEqual((window.start_ratio, window.end_ratio), (0.0, 1.0))
        self.assertEqual(window.corner_vertex_ids, self.window.corner_vertex_ids)
        self.assertEqual(self.level.vertex_data.to_dict(), self.before_graph)
        target = _window_target(self.level)
        self.assertAlmostEqual(target.wall_width_meters, 2.4)
        self.assertAlmostEqual(
            target.bounds.horizontal_span * target.wall_width_meters, 1.0
        )
        self.assertAlmostEqual(
            target.bounds.vertical_span * target.wall_height_meters, 1.0
        )
        self.assertAlmostEqual(
            target.bounds.bottom_ratio * target.wall_height_meters, 1.0
        )

    # ### Shared preview, delay, undo and cancel ###
    def test_3d_drag_defers_mesh_rebuild_and_shared_undo_restores_exact_graph(self):
        start = _edit(self.target)
        edit = _edit(self.target, horizontal=0.1)
        with (
            patch.object(self.workspace, "_build_viewer_preview_model") as build,
            patch.object(
                self.workspace, "_reconcile_canvas_surface_edit_and_refresh"
            ) as refresh,
        ):
            self.workspace._handle_canvas_opening_edit_started(start)
            for delta in (0.025, 0.05, 0.1):
                self.workspace._handle_canvas_opening_edit_preview_changed(
                    _edit(self.target, horizontal=delta)
                )
            self._assert_horizontal_movement(12.0)
            self.assertTrue(self.workspace._is_canvas_opening_drag_active)
            self.assertFalse(self.workspace._doorway_mesh_update_timer.isActive())
            self.assertEqual(
                self.workspace._viewer_windows_by_level_index[2], (self.window,)
            )
            self.workspace._queue_viewer_preview_refresh()
            self.workspace._run_scheduled_viewer_preview_refresh()
            build.assert_not_called()
            refresh.assert_not_called()

            self.workspace._handle_canvas_opening_edit_finished(edit, True)

            self.assertTrue(self.workspace._doorway_mesh_update_timer.isActive())
            self.assertTrue(self.workspace._pending_doorway_bridge_topology_change)
            self.assertEqual(len(self.workspace._canvas_undo_stack), 1)
            refresh.assert_not_called()
            self.workspace._commit_pending_doorway_mesh_update()
            refresh.assert_called_once()

        self.workspace._handle_canvas_undo_requested()
        self.assertEqual(self.level.vertex_data.to_dict(), self.before_graph)
        self.assertEqual(self.level.windows, [self.window])
        self.assertEqual(self.workspace._canvas_undo_stack, [])

    def _prepare_canvas_drag(self):
        # Send pointer events directly: showing the complete offscreen
        # workspace would unnecessarily initialize its native OpenGL view.
        canvas = self.workspace.canvas
        segment = canvas._get_window_widget_segment(self.window)
        self.assertIsNotNone(segment)
        assert segment is not None
        center = (segment[0] + segment[1]) * 0.5
        press_position = center.toPoint()
        target = QPointF(canvas._image_to_widget(80.0, 50.0).toPoint())
        press_image = canvas._widget_to_image_clamped(QPointF(press_position))
        target_image = canvas._widget_to_image_clamped(target)
        QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=press_position)
        _send_drag_move(canvas, target)
        return canvas, target, target_image.x() - press_image.x()

    def test_3d_drag_cancel_restores_exact_window_and_graph(self):
        self.assertFalse(self.workspace._doorway_mesh_update_timer.isActive())
        self.assertFalse(self.workspace._pending_doorway_bridge_topology_change)
        start = _edit(self.target)
        self.workspace._handle_canvas_opening_edit_started(start)
        self.workspace._handle_canvas_opening_edit_preview_changed(
            _edit(self.target, horizontal=0.1)
        )
        self._assert_horizontal_movement(12.0)

        self.workspace._handle_canvas_opening_edit_cancelled(start)

        self.assertEqual(self.level.vertex_data.to_dict(), self.before_graph)
        self.assertEqual(self.level.windows, [self.window])
        self.assertEqual(self.workspace._canvas_undo_stack, [])
        self.assertFalse(self.workspace._is_canvas_opening_drag_active)
        self.assertFalse(self.workspace._doorway_mesh_update_timer.isActive())
        self.assertFalse(self.workspace._pending_doorway_bridge_topology_change)
        self.assertIsNone(self.workspace._pending_window_mesh_level_index)

    def _slope_rehost_edits(self):
        diagonal = 2.0**-0.5
        destination = replace(
            self.target,
            plane_start_world=(0.2, -1.0, 2.0),
            wall_normal_world=(0.0, -diagonal, diagonal),
            vertical_axis_world=(0.0, diagonal, diagonal),
            sloped_ceiling_id="a" * 32,
        )
        start = CanvasOpeningEdit(
            self.target.reference,
            self.target.wall_surface_id,
            self.target.bounds,
            destination_target=self.target,
        )
        moved = CanvasOpeningEdit(
            destination.reference,
            destination.wall_surface_id,
            destination.bounds,
            destination_target=destination,
        )
        return start, moved

    def test_slope_rehost_cancel_restores_linked_window_graph_and_target(self):
        start, moved = self._slope_rehost_edits()
        self.workspace._handle_canvas_opening_edit_started(start)
        self.workspace._handle_canvas_opening_edit_preview_changed(
            _edit(self.target, horizontal=0.05)
        )
        self._assert_horizontal_movement(6.0)
        self.workspace._handle_canvas_opening_edit_preview_changed(moved)
        self.assertEqual(self.level.windows[0].sloped_ceiling_id, "a" * 32)
        self.assertIsNone(self.level.windows[0].corner_vertex_ids)

        self.workspace._handle_canvas_opening_edit_cancelled(start)

        self.assertEqual(self.level.windows, [self.window])
        self.assertEqual(self.level.vertex_data.to_dict(), self.before_graph)
        self.assertEqual(
            self.workspace._canvas_opening_targets_by_key[self.target.key], self.target
        )
        self.assertEqual(
            self.workspace.viewer._canvas_opening_targets[self.target.key], self.target
        )
        self.assertEqual(self.workspace._canvas_undo_stack, [])
        self.assertFalse(self.workspace._doorway_mesh_update_timer.isActive())
        self.assertFalse(self.workspace._pending_doorway_bridge_topology_change)

    def test_slope_rehost_undo_restores_linked_window_graph_and_target(self):
        start, moved = self._slope_rehost_edits()
        self.workspace._handle_canvas_opening_edit_started(start)
        self.workspace._handle_canvas_opening_edit_preview_changed(
            _edit(self.target, horizontal=0.05)
        )
        self._assert_horizontal_movement(6.0)
        self.workspace._handle_canvas_opening_edit_preview_changed(moved)
        self.workspace._handle_canvas_opening_edit_finished(moved, True)
        self.assertEqual(len(self.workspace._canvas_undo_stack), 1)

        self.workspace._handle_canvas_undo_requested()

        self.assertEqual(self.level.windows, [self.window])
        self.assertEqual(self.level.vertex_data.to_dict(), self.before_graph)
        self.assertEqual(
            self.workspace._canvas_opening_targets_by_key[self.target.key], self.target
        )
        self.assertEqual(
            self.workspace.viewer._canvas_opening_targets[self.target.key], self.target
        )
        self.assertEqual(self.workspace._canvas_undo_stack, [])

    def test_cancelling_second_drag_keeps_earlier_pending_corner_move(self):
        first_start = _edit(self.target)
        first_edit = _edit(self.target, horizontal=0.05)
        self.workspace._handle_canvas_opening_edit_started(first_start)
        self.workspace._handle_canvas_opening_edit_preview_changed(first_edit)
        self.workspace._handle_canvas_opening_edit_finished(first_edit, True)
        self._assert_horizontal_movement(6.0)
        first_graph = self.level.vertex_data.to_dict()
        self.assertTrue(self.workspace._doorway_mesh_update_timer.isActive())
        target = self.workspace._canvas_opening_targets_by_key[self.target.key]
        second_start = _edit(target)
        self.workspace._handle_canvas_opening_edit_started(second_start)
        self.workspace._handle_canvas_opening_edit_preview_changed(
            _edit(target, horizontal=0.025)
        )
        self._assert_horizontal_movement(9.0)

        self.workspace._handle_canvas_opening_edit_cancelled(second_start)

        self.assertEqual(self.level.vertex_data.to_dict(), first_graph)
        self.assertEqual(len(self.workspace._canvas_undo_stack), 1)
        self.assertTrue(self.workspace._pending_doorway_bridge_topology_change)
        self.assertEqual(self.workspace._pending_window_mesh_level_index, 2)
        self.assertTrue(self.workspace._doorway_mesh_update_timer.isActive())

    def test_2d_body_drag_moves_both_faces_and_records_one_undo(self):
        canvas, target, delta = self._prepare_canvas_drag()
        self.assertGreater(delta, 0.0)
        self._assert_horizontal_movement(delta)
        self.assertFalse(self.workspace._doorway_mesh_update_timer.isActive())

        QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=target.toPoint())

        self.assertTrue(self.workspace._doorway_mesh_update_timer.isActive())
        self.assertEqual(len(self.workspace._canvas_undo_stack), 1)
        self.workspace._handle_canvas_undo_requested()
        self.assertEqual(self.level.vertex_data.to_dict(), self.before_graph)
        self.assertEqual(self.level.windows, [self.window])

    def test_escape_cancels_2d_body_drag_restoring_exact_window_and_graph(self):
        self.assertFalse(self.workspace._doorway_mesh_update_timer.isActive())
        canvas, _target, delta = self._prepare_canvas_drag()
        self.assertGreater(delta, 0.0)
        self._assert_horizontal_movement(delta)

        QTest.keyClick(canvas, Qt.Key.Key_Escape)

        self.assertEqual(self.level.vertex_data.to_dict(), self.before_graph)
        self.assertEqual(self.level.windows, [self.window])
        self.assertEqual(self.workspace._canvas_undo_stack, [])
        self.assertFalse(self.workspace._doorway_mesh_update_timer.isActive())
        self.assertFalse(self.workspace._pending_doorway_bridge_topology_change)

    # ### Save/load and old four-corner windows ###
    def test_save_load_retains_corner_links_and_movable_full_run_target(self):
        apply_canvas_opening_edit(
            [self.level], self.target, _edit(self.target, horizontal=0.1)
        )
        path = Path(self.temporary.name) / "linked-window.json"
        save_project(path, 0, [self.level])

        restored = next(
            level
            for level in load_project(path).levels
            if level.index == self.level.index
        )

        self.assertEqual(
            restored.vertex_data.to_dict(), self.level.vertex_data.to_dict()
        )
        self.assertEqual(restored.windows, self.level.windows)
        self.assertEqual(
            set(restored.windows[0].corner_vertex_ids or ()), set(self.corner_ids)
        )
        target = _window_target(restored)
        self.assertAlmostEqual(target.wall_width_meters, 2.4)
        apply_canvas_opening_edit([restored], target, _edit(target, horizontal=-0.1))
        self.assertEqual(tuple(restored.vertex_data.edges), self.before_edges)
        for vertex in restored.vertex_data.vertices:
            old_x, old_y = self.before_positions[vertex.id]
            self.assertAlmostEqual(vertex.x, old_x, places=9)
            self.assertAlmostEqual(vertex.y, old_y, places=9)

    def test_legacy_four_corner_frame_without_saved_links_remains_movable(self):
        self.level.windows[0] = replace(self.window, corner_vertex_ids=None)
        target = _window_target(self.level)
        self.assertAlmostEqual(target.wall_width_meters, 2.4)

        apply_canvas_opening_edit([self.level], target, _edit(target, horizontal=0.1))

        self._assert_horizontal_movement(12.0)


# ### Test runner ###
if __name__ == "__main__":
    unittest.main()
