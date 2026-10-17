# ### Environment setup ###
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.blueprint_canvas import BlueprintCanvas
from housemaker.main import BlueprintWorkspace
from housemaker.models import DoorwayData, LevelData, VertexData
from housemaker.surface_geometry import add_wall_window, build_fixed_surfaces
from housemaker.window_frame_bridge import (
    add_window_frame_bridge,
    build_window_frame_bridge_placement,
)

# ### Fixtures ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


def _gap_level(*, diagonal_offset: float = 0.0) -> LevelData:
    vertices = VertexData()
    for point in (
        (10.0, 50.0),
        (40.0, 50.0),
        (90.0, 50.0 + diagonal_offset),
        (130.0, 50.0 + diagonal_offset),
    ):
        vertices.add_vertex(*point)
    vertices.add_edge(1, 2)
    vertices.add_edge(3, 4)
    return LevelData(index=2, name="Ground", vertex_data=vertices)


def _bridge_placement(level: LevelData):
    return build_window_frame_bridge_placement(
        level,
        (65.0, 50.0),
        5.0,
        preferred_width_meters=1.0,
        height_meters=1.0,
        wall_margin_meters=0.025,
    )


# ### Bridge validation tests ###
class WindowFrameBridgeTests(unittest.TestCase):
    def test_preview_connects_exact_gap_and_keeps_centered_window_height(self) -> None:
        level = _gap_level()
        placement = _bridge_placement(level)
        self.assertIsNotNone(placement)
        assert placement is not None
        self.assertEqual(placement.bridge_vertex_ids, (2, 3))
        self.assertEqual((placement.start_ratio, placement.end_ratio), (0.0, 1.0))
        self.assertAlmostEqual(placement.bottom_ratio + placement.top_ratio, 1.0)
        self.assertAlmostEqual(
            (placement.top_ratio - placement.bottom_ratio) * level.height_meters,
            1.0,
        )
        self.assertFalse(level.vertex_data.has_edge(2, 3))

        bridge = add_window_frame_bridge([level], placement)
        self.assertIsNotNone(bridge)
        window = add_wall_window([level], placement)
        self.assertTrue(level.vertex_data.has_edge(2, 3))
        self.assertEqual(window.wall_surface_id, placement.wall_surface_id)
        wall = next(
            surface for surface in build_fixed_surfaces([level])
            if surface.surface_id == window.wall_surface_id
        )
        self.assertAlmostEqual(wall.area_square_meters, level.height_meters - 1.0)

    def test_shift_never_bridges_connected_or_non_aligned_vertices(self) -> None:
        for kind in ("connected", "non_aligned"):
            with self.subTest(kind=kind):
                level = _gap_level(
                    diagonal_offset=5.0 if kind == "non_aligned" else 0.0
                )
                if kind == "connected":
                    level.vertex_data.add_edge(2, 3)
                self.assertIsNone(_bridge_placement(level))

    def test_changed_gap_is_rejected_before_adding_an_edge(self) -> None:
        level = _gap_level()
        placement = _bridge_placement(level)
        assert placement is not None
        level.vertex_data.add_edge(2, 3)
        with self.assertRaisesRegex(ValueError, "aligned and unconnected"):
            add_window_frame_bridge([level], placement)
        self.assertEqual(len(level.vertex_data.edges), 3)
        self.assertEqual(level.windows, [])

    def test_shift_press_release_updates_stationary_canvas_preview_and_cancel_is_safe(self) -> None:
        level = _gap_level()
        canvas = BlueprintCanvas()
        canvas.resize(640, 520)
        canvas.set_level_data(
            vertex_data=level.vertex_data,
            rooms=level.rooms,
            image_path=None,
            doorways=level.doorways,
            windows=level.windows,
        )
        canvas.blueprint_image = QImage(150, 100, QImage.Format.Format_RGB32)
        canvas.blueprint_image.fill(Qt.GlobalColor.white)
        canvas.set_stair_context((), level)
        canvas.show()
        _qt_application.processEvents()
        try:
            self.assertTrue(canvas.start_window_placement())
            canvas._update_pending_window(QPointF(65.0, 50.0))
            self.assertIsNone(canvas._pending_window_placement.bridge_vertex_ids)
            QTest.keyPress(canvas, Qt.Key.Key_Shift)
            pending = canvas._pending_window_placement
            assert pending is not None
            self.assertEqual(pending.bridge_vertex_ids, (2, 3))
            segment = canvas._get_window_widget_segment(pending)
            self.assertIsNotNone(segment)
            assert segment is not None
            self.assertEqual(segment[0], canvas._image_to_widget(40.0, 50.0))
            self.assertEqual(segment[1], canvas._image_to_widget(90.0, 50.0))
            QTest.keyRelease(canvas, Qt.Key.Key_Shift)
            self.assertIsNone(canvas._pending_window_placement.bridge_vertex_ids)
            QTest.keyPress(canvas, Qt.Key.Key_Shift)
            QTest.keyClick(canvas, Qt.Key.Key_Escape)
            self.assertFalse(canvas.is_window_placement_active())
            self.assertFalse(level.vertex_data.has_edge(2, 3))
            self.assertEqual(level.windows, [])
        finally:
            canvas.close()
            canvas.deleteLater()
            _qt_application.processEvents()


# ### Main transaction tests ###
class WindowFrameBridgeMainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(
                Path(self.temporary.name) / "settings.json"
            )
        )
        self.workspace._set_mesh_edit_update_delay_seconds(60.0)
        self.level = _gap_level()
        self.workspace.levels = [self.level]
        self.workspace.current_level_index = 0
        self.workspace._sync_canvas_to_current_level()
        self.workspace._reset_viewer_doorway_snapshots()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self.temporary.cleanup()

    def test_bridge_and_window_wait_for_delay_and_undo_as_one_action(self) -> None:
        placement = _bridge_placement(self.level)
        assert placement is not None
        with (
            patch.object(self.workspace, "_build_viewer_preview_model") as build_preview,
            patch.object(
                self.workspace, "_reconcile_canvas_surface_edit_and_refresh"
            ) as refresh,
        ):
            self.workspace._handle_canvas_window_placement_requested(placement)
            self.assertTrue(self.level.vertex_data.has_edge(2, 3))
            self.assertEqual(len(self.level.windows), 1)
            self.assertEqual(len(self.workspace._canvas_undo_stack), 1)
            self.assertTrue(self.workspace._pending_doorway_bridge_topology_change)
            self.assertTrue(self.workspace._doorway_mesh_update_timer.isActive())
            self.workspace._queue_viewer_preview_refresh()
            self.workspace._run_scheduled_viewer_preview_refresh()
            build_preview.assert_not_called()
            refresh.assert_not_called()
            self.workspace._commit_pending_doorway_mesh_update()
            refresh.assert_called_once()
            build_preview.assert_not_called()
        self.workspace._handle_canvas_undo_requested()
        self.assertEqual(self.level.windows, [])
        self.assertFalse(self.level.vertex_data.has_edge(2, 3))
        self.assertEqual(len(self.level.vertex_data.edges), 2)
        self.assertEqual(self.workspace._canvas_undo_stack, [])

    def test_failed_window_commit_leaves_no_bridge_or_undo_entry(self) -> None:
        placement = _bridge_placement(self.level)
        assert placement is not None
        with patch(
            "housemaker.main.add_wall_window",
            side_effect=ValueError("rejected opening"),
        ):
            self.workspace._handle_canvas_window_placement_requested(placement)
        self.assertFalse(self.level.vertex_data.has_edge(2, 3))
        self.assertEqual(self.level.windows, [])
        self.assertEqual(self.workspace._canvas_undo_stack, [])
        self.assertFalse(self.workspace._pending_doorway_bridge_topology_change)

    def test_undo_latest_pending_bridge_keeps_earlier_bridge_and_cutout(self) -> None:
        self.level.vertex_data.add_vertex(180.0, 50.0)
        self.level.vertex_data.add_vertex(220.0, 50.0)
        self.level.vertex_data.add_edge(5, 6)
        first_placement = _bridge_placement(self.level)
        second_placement = build_window_frame_bridge_placement(
            self.level,
            (155.0, 50.0),
            5.0,
            preferred_width_meters=1.0,
            height_meters=1.0,
            wall_margin_meters=0.025,
        )
        assert first_placement is not None
        assert second_placement is not None
        self.workspace._handle_canvas_window_placement_requested(first_placement)
        first_window = self.level.windows[0]
        self.workspace._handle_canvas_window_placement_requested(second_placement)
        self.assertEqual(len(self.level.windows), 2)
        self.assertTrue(self.workspace._doorway_mesh_update_timer.isActive())

        self.workspace._handle_canvas_undo_requested()

        self.assertEqual(self.level.windows, [first_window])
        self.assertTrue(self.level.vertex_data.has_edge(2, 3))
        self.assertFalse(self.level.vertex_data.has_edge(4, 5))
        preview_level = self.workspace._build_viewer_preview_levels()[0]
        self.assertEqual(preview_level.windows, [first_window])
        self.assertIsNotNone(self.workspace.viewer.model)
        first_wall = self.workspace.viewer._window_wall_targets[
            first_window.wall_surface_id
        ]
        self.assertAlmostEqual(
            first_wall.area_square_meters, self.level.height_meters - 1.0
        )
        self.assertNotIn(
            second_placement.wall_surface_id,
            self.workspace.viewer._window_wall_targets,
        )
        self.assertEqual(len(self.workspace._canvas_undo_stack), 1)

        self.workspace._handle_canvas_undo_requested()
        self.assertFalse(self.level.vertex_data.has_edge(2, 3))
        self.assertEqual(self.level.windows, [])
        self.assertEqual(self.workspace._canvas_undo_stack, [])

    def test_undo_pending_window_keeps_older_pending_doorway_in_3d(self) -> None:
        doorway = DoorwayData(
            center_x=110.0,
            center_y=50.0,
            width_meters=0.4,
            height_meters=1.8,
            rotation_degrees=90.0,
        )
        self.workspace.canvas.pending_doorway = doorway
        self.workspace.canvas._commit_pending_doorway()
        self.assertEqual(self.level.doorways, [doorway])
        self.assertEqual(self.workspace._viewer_doorways_by_level_index[2], ())
        self.assertIsNotNone(self.workspace._pending_doorway_mesh_level_index)
        placement = _bridge_placement(self.level)
        assert placement is not None
        self.workspace._handle_canvas_window_placement_requested(placement)

        self.workspace._handle_canvas_undo_requested()

        self.assertEqual(self.level.windows, [])
        self.assertEqual(self.level.doorways, [doorway])
        self.assertEqual(
            self.workspace._viewer_doorways_by_level_index[2], (doorway,)
        )
        rendered_wall = self.workspace.viewer._window_wall_targets[
            "level:2/wall:3:4"
        ]
        expected_wall = next(
            surface
            for surface in build_fixed_surfaces([self.level])
            if surface.surface_id == "level:2/wall:3:4"
        )
        self.assertAlmostEqual(
            rendered_wall.area_square_meters,
            expected_wall.area_square_meters,
        )
        self.assertNotAlmostEqual(
            rendered_wall.area_square_meters, 0.8 * self.level.height_meters
        )
        self.assertFalse(self.workspace._doorway_mesh_update_timer.isActive())
        self.assertFalse(self.level.vertex_data.has_edge(2, 3))
        self.assertEqual(len(self.workspace._canvas_undo_stack), 1)

    def test_unexpected_window_failure_also_rolls_back_bridge(self) -> None:
        placement = _bridge_placement(self.level)
        assert placement is not None
        with (
            patch(
                "housemaker.main.add_wall_window",
                side_effect=RuntimeError("mesh failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "mesh failure"),
        ):
            self.workspace._handle_canvas_window_placement_requested(placement)
        self.assertFalse(self.level.vertex_data.has_edge(2, 3))
        self.assertEqual(self.level.windows, [])
        self.assertEqual(self.workspace._canvas_undo_stack, [])


# ### Test runner ###
if __name__ == "__main__":
    unittest.main()
