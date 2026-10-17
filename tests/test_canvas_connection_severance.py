# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QImage, QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.blueprint_canvas import (
    CANVAS_SNAPSHOT_ACTION_CONNECTION_SEVERANCE,
    BlueprintCanvas,
)
from housemaker.door_state import (
    DoorLibraryData,
    WindowPlacement,
    create_window_definition_for_frame,
)
from housemaker.main import BlueprintWorkspace
from housemaker.models import DoorwayData, LevelData, RoomData, VertexData, WindowData
from housemaker.surface_geometry import SURFACE_TYPE_WALL, build_fixed_surfaces

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture and event helpers ###
def _build_two_wall_level() -> LevelData:
    vertex_data = VertexData()
    first = vertex_data.add_vertex(10.0, 25.0)
    second = vertex_data.add_vertex(90.0, 25.0)
    third = vertex_data.add_vertex(10.0, 75.0)
    fourth = vertex_data.add_vertex(90.0, 75.0)
    vertex_data.add_edge(first.id, second.id)
    vertex_data.add_edge(third.id, fourth.id)
    return LevelData(index=2, name="Ground", vertex_data=vertex_data)


def _send_repeated_delete(canvas: BlueprintCanvas) -> None:
    QApplication.sendEvent(
        canvas,
        QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_Delete,
            Qt.KeyboardModifier.NoModifier,
            "",
            True,
            1,
        ),
    )


# ### Canvas connection severance tests ###
class CanvasConnectionSeveranceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas = BlueprintCanvas()
        self.canvas.resize(640, 520)
        self.canvas.blueprint_image = QImage(
            100, 100, QImage.Format.Format_RGB32
        )
        self.canvas.blueprint_image.fill(Qt.GlobalColor.white)
        self.canvas.vertex_data = _build_two_wall_level().vertex_data
        self.pair = tuple(
            vertex.id for vertex in self.canvas.vertex_data.vertices[:2]
        )
        self.unrelated_pair = tuple(
            vertex.id for vertex in self.canvas.vertex_data.vertices[2:]
        )
        self.canvas.set_selected_vertex_ids(self.pair)
        self.canvas.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.canvas.close()
        self.canvas.deleteLater()
        _qt_application.processEvents()

    def test_setting_is_disabled_by_default(self) -> None:
        QTest.keyClick(self.canvas, Qt.Key.Key_Delete)

        self.assertTrue(
            all(self.canvas.vertex_data.get_vertex(vertex_id) is None
                for vertex_id in self.pair)
        )
        self.assertTrue(self.canvas.vertex_data.has_edge(*self.unrelated_pair))

    def test_delete_cuts_only_selected_edge_and_preserves_selection(self) -> None:
        self.canvas.set_cut_connection_on_double_vertex_deletion(True)
        self.canvas.vertex_data.add_edge(
            self.pair[1], self.unrelated_pair[0]
        )
        events: list[tuple[str, object]] = []
        self.canvas.wall_connections_severed.connect(
            lambda pair: events.append(("severed", pair))
        )
        self.canvas.wall_vertices_deleted.connect(
            lambda vertex_ids: events.append(("deleted", vertex_ids))
        )
        self.canvas.geometry_changed.connect(
            lambda: events.append(("geometry", None))
        )

        QTest.keyClick(self.canvas, Qt.Key.Key_Delete)

        self.assertEqual(
            events, [("severed", self.pair), ("geometry", None)]
        )
        self.assertEqual(len(self.canvas.vertex_data.vertices), 4)
        self.assertFalse(self.canvas.vertex_data.has_edge(*self.pair))
        self.assertTrue(self.canvas.vertex_data.has_edge(*self.unrelated_pair))
        self.assertTrue(
            self.canvas.vertex_data.has_edge(
                self.pair[1], self.unrelated_pair[0]
            )
        )
        self.assertEqual(self.canvas.selected_vertex_ids, self.pair)
        self.assertEqual(len(self.canvas.undo_stack), 1)
        self.assertEqual(
            self.canvas.undo_stack[-1].action_kind,
            CANVAS_SNAPSHOT_ACTION_CONNECTION_SEVERANCE,
        )

    def test_second_deliberate_delete_removes_still_selected_vertices(self) -> None:
        self.canvas.set_cut_connection_on_double_vertex_deletion(True)

        QTest.keyClick(self.canvas, Qt.Key.Key_Delete)
        QTest.keyClick(self.canvas, Qt.Key.Key_Delete)

        self.assertTrue(
            all(self.canvas.vertex_data.get_vertex(vertex_id) is None
                for vertex_id in self.pair)
        )
        self.assertEqual(self.canvas.selected_vertex_ids, ())
        self.assertEqual(len(self.canvas.undo_stack), 2)

    def test_undo_restores_connection_and_keeps_selected_vertices(self) -> None:
        self.canvas.set_cut_connection_on_double_vertex_deletion(True)
        QTest.keyClick(self.canvas, Qt.Key.Key_Delete)

        self.canvas.undo_last_step()

        self.assertTrue(self.canvas.vertex_data.has_edge(*self.pair))
        self.assertTrue(self.canvas.vertex_data.has_edge(*self.unrelated_pair))
        self.assertEqual(self.canvas.selected_vertex_ids, self.pair)
        self.assertEqual(len(self.canvas.vertex_data.vertices), 4)

    def test_repeated_undo_restores_deletion_then_original_connection(self) -> None:
        self.canvas.set_cut_connection_on_double_vertex_deletion(True)
        QTest.keyClick(self.canvas, Qt.Key.Key_Delete)
        QTest.keyClick(self.canvas, Qt.Key.Key_Delete)

        self.canvas.undo_last_step()

        self.assertEqual(len(self.canvas.vertex_data.vertices), 4)
        self.assertFalse(self.canvas.vertex_data.has_edge(*self.pair))
        self.assertEqual(self.canvas.selected_vertex_ids, self.pair)

        self.canvas.undo_last_step()

        self.assertTrue(self.canvas.vertex_data.has_edge(*self.pair))
        self.assertEqual(self.canvas.selected_vertex_ids, self.pair)

    def test_held_delete_cannot_delete_vertices_after_cutting_connection(self) -> None:
        self.canvas.set_cut_connection_on_double_vertex_deletion(True)
        QTest.keyClick(self.canvas, Qt.Key.Key_Delete)

        _send_repeated_delete(self.canvas)
        _send_repeated_delete(self.canvas)

        self.assertFalse(self.canvas.vertex_data.has_edge(*self.pair))
        self.assertEqual(len(self.canvas.vertex_data.vertices), 4)
        self.assertEqual(self.canvas.selected_vertex_ids, self.pair)
        self.assertEqual(len(self.canvas.undo_stack), 1)

    def test_cutting_connection_removes_its_doorway_and_undo_restores_it(self) -> None:
        self.canvas.set_cut_connection_on_double_vertex_deletion(True)
        doorway = DoorwayData(
            center_x=50.0,
            center_y=25.0,
            width_meters=0.3,
            height_meters=2.1,
            rotation_degrees=90.0,
        )
        unrelated_doorway = DoorwayData(
            center_x=50.0,
            center_y=75.0,
            width_meters=0.3,
            height_meters=2.1,
            rotation_degrees=90.0,
        )
        self.canvas.doorways.extend((doorway, unrelated_doorway))

        QTest.keyClick(self.canvas, Qt.Key.Key_Delete)

        self.assertEqual(self.canvas.doorways, [unrelated_doorway])
        self.assertEqual(self.canvas.selected_vertex_ids, self.pair)
        self.canvas.undo_last_step()
        self.assertEqual(self.canvas.doorways, [doorway, unrelated_doorway])
        self.assertTrue(self.canvas.vertex_data.has_edge(*self.pair))

    def test_enabled_setting_still_deletes_unconnected_pair(self) -> None:
        self.canvas.set_cut_connection_on_double_vertex_deletion(True)
        self.canvas.set_selected_vertex_ids(
            (self.pair[0], self.unrelated_pair[0])
        )

        QTest.keyClick(self.canvas, Qt.Key.Key_Delete)

        self.assertIsNone(self.canvas.vertex_data.get_vertex(self.pair[0]))
        self.assertIsNone(
            self.canvas.vertex_data.get_vertex(self.unrelated_pair[0])
        )
        self.assertEqual(len(self.canvas.vertex_data.vertices), 2)

    def test_enabled_setting_still_deletes_more_than_two_vertices(self) -> None:
        self.canvas.set_cut_connection_on_double_vertex_deletion(True)
        self.canvas.set_selected_vertex_ids((*self.pair, self.unrelated_pair[0]))

        QTest.keyClick(self.canvas, Qt.Key.Key_Delete)

        self.assertEqual(
            [vertex.id for vertex in self.canvas.vertex_data.vertices],
            [self.unrelated_pair[1]],
        )

    def test_enabled_setting_still_deletes_single_vertex(self) -> None:
        self.canvas.set_cut_connection_on_double_vertex_deletion(True)
        self.canvas.set_selected_vertex_ids((self.pair[0],))

        QTest.keyClick(self.canvas, Qt.Key.Key_Delete)

        self.assertIsNone(self.canvas.vertex_data.get_vertex(self.pair[0]))
        self.assertIsNotNone(self.canvas.vertex_data.get_vertex(self.pair[1]))
        self.assertEqual(len(self.canvas.vertex_data.vertices), 3)

    def test_disabling_setting_restores_direct_vertex_deletion(self) -> None:
        self.canvas.set_cut_connection_on_double_vertex_deletion(True)
        self.canvas.set_cut_connection_on_double_vertex_deletion(False)

        QTest.keyClick(self.canvas, Qt.Key.Key_Delete)

        self.assertEqual(len(self.canvas.vertex_data.vertices), 2)
        self.assertEqual(self.canvas.selected_vertex_ids, ())

    def test_last_legacy_room_edge_does_not_recreate_its_walls(self) -> None:
        vertices = VertexData()
        boundary = [
            vertices.add_vertex(x, y)
            for x, y in ((10, 10), (90, 10), (90, 90), (10, 90))
        ]
        center = vertices.add_vertex(50, 50)
        pairs = [
            (boundary[index].id, boundary[(index + 1) % 4].id)
            for index in range(4)
        ]
        for pair in pairs:
            vertices.add_edge(*pair)
        room = RoomData(
            name="Legacy room",
            vertex_ids=tuple(vertex.id for vertex in boundary),
            center_vertex_id=center.id,
            color_rgb=(100, 120, 140),
        )
        self.canvas.vertex_data = vertices
        self.canvas.rooms[:] = [room]
        self.canvas.set_cut_connection_on_double_vertex_deletion(True)

        for index, pair in enumerate(pairs):
            self.canvas.set_selected_vertex_ids(pair)
            QTest.keyClick(self.canvas, Qt.Key.Key_Delete)
            level = LevelData(
                index=2, name="Ground", vertex_data=vertices,
                rooms=self.canvas.rooms,
            )
            self.assertEqual(
                sum(
                    surface.surface_type == SURFACE_TYPE_WALL
                    for surface in build_fixed_surfaces([level])
                ),
                3 - index,
            )

        self.assertEqual(len(vertices.vertices), 5)
        self.assertEqual(self.canvas.selected_vertex_ids, pairs[-1])
        self.assertEqual(self.canvas.rooms, [])
        self.canvas.undo_last_step()
        self.assertTrue(vertices.has_edge(*pairs[-1]))
        self.assertEqual(self.canvas.rooms, [room])


# ### Workspace delay and undo integration tests ###
class CanvasConnectionSeveranceMainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(
                Path(self.temporary_directory.name) / "settings.json"
            )
        )
        self.level = _build_two_wall_level()
        self.workspace._apply_project_state([self.level], 0)
        self.workspace.canvas.blueprint_image = QImage(
            100, 100, QImage.Format.Format_RGB32
        )
        self.workspace.canvas.blueprint_image.fill(Qt.GlobalColor.white)
        self.workspace.canvas.set_cut_connection_on_double_vertex_deletion(True)
        self.pair = tuple(
            vertex.id for vertex in self.level.vertex_data.vertices[:2]
        )
        self.workspace.canvas.set_selected_vertex_ids(self.pair)
        self.workspace._canvas_undo_stack.clear()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self.temporary_directory.cleanup()

    def test_checkbox_applies_immediately_and_restores_on_restart(self) -> None:
        checkbox = (
            self.workspace.settings_widget
            .cut_connection_on_double_vertex_deletion_checkbox
        )
        checkbox.setChecked(True)
        self.assertTrue(
            self.workspace.canvas.cut_connection_on_double_vertex_deletion
        )
        checkbox.setChecked(False)
        self.assertFalse(
            self.workspace.canvas.cut_connection_on_double_vertex_deletion
        )
        checkbox.setChecked(True)
        restored = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(
                Path(self.temporary_directory.name) / "settings.json"
            )
        )
        try:
            self.assertTrue(
                restored.canvas.cut_connection_on_double_vertex_deletion
            )
        finally:
            restored.shutdown()
            restored.close()
            _qt_application.processEvents()

    def test_severance_rebuilds_only_after_mesh_edit_delay(self) -> None:
        timer = self.workspace._wall_vertex_update_timer
        with (
            patch.object(
                self.workspace.surface_texture_generation,
                "reconcile_assignments_with_levels",
                return_value=False,
            ) as reconcile,
            patch.object(
                self.workspace, "_schedule_viewer_preview_refresh"
            ) as refresh,
        ):
            QTest.keyClick(self.workspace.canvas, Qt.Key.Key_Delete)

            self.assertFalse(self.level.vertex_data.has_edge(*self.pair))
            self.assertTrue(self.workspace._pending_wall_vertex_mesh_update)
            self.assertTrue(timer.isActive())
            self.assertEqual(timer.interval(), 1_000)
            reconcile.assert_not_called()
            refresh.assert_not_called()

            self.workspace._commit_pending_wall_vertex_update()

        reconcile.assert_called_once_with(self.workspace.levels)
        refresh.assert_called_once_with(preserve_camera=True)
        self.assertFalse(self.workspace._pending_wall_vertex_mesh_update)
        self.assertFalse(timer.isActive())

    def test_cutting_active_chain_ends_it_and_starts_delay(self) -> None:
        self.workspace.canvas.active_vertex_id = self.pair[0]

        QTest.keyClick(self.workspace.canvas, Qt.Key.Key_Delete)

        self.assertIsNone(self.workspace.canvas.active_vertex_id)
        self.assertEqual(self.workspace.canvas.selected_vertex_ids, self.pair)
        self.assertTrue(self.workspace._pending_wall_vertex_mesh_update)
        self.assertTrue(self.workspace._wall_vertex_update_timer.isActive())

    def test_ctrl_z_restores_cut_wall_frame_and_placed_window(self) -> None:
        host_wall = next(
            surface
            for surface in build_fixed_surfaces((self.level,))
            if surface.surface_type == SURFACE_TYPE_WALL
            and surface.wall_key == "1:2"
        )
        frame = WindowData(
            window_id="severed-wall-frame",
            wall_surface_id=host_wall.surface_id,
            start_ratio=0.25,
            end_ratio=0.75,
            bottom_ratio=0.2,
            top_ratio=0.7,
        )
        self.level.windows = [frame]
        self.workspace._sync_canvas_to_current_level()
        window = create_window_definition_for_frame(
            frame.window_id,
            1.0,
            1.0,
            name="Reusable window",
            window_id="severed-wall-window",
        )
        placement = WindowPlacement(
            placement_id="severed-wall-window-placement",
            window_id=window.window_id,
            level_index=self.level.index,
            window_frame_id=frame.window_id,
        )
        self.workspace.doors_workspace.set_data(
            DoorLibraryData(windows=[window], window_placements=[placement])
        )
        self.workspace.canvas.set_selected_vertex_ids(self.pair)
        self.workspace._canvas_undo_stack.clear()

        QTest.keyClick(self.workspace.canvas, Qt.Key.Key_Delete)

        self.assertEqual(self.level.windows, [frame])
        self.assertEqual(
            self.workspace.doors_workspace.data().window_placements, [placement]
        )
        self.workspace._commit_pending_wall_vertex_update()
        self.assertEqual(self.level.windows, [])
        self.assertEqual(
            self.workspace.doors_workspace.data().window_placements, []
        )
        self.workspace.viewer.view.keyPressEvent(
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_Z,
                Qt.KeyboardModifier.ControlModifier,
            )
        )

        self.assertTrue(self.level.vertex_data.has_edge(*self.pair))
        self.assertEqual(self.workspace.canvas.selected_vertex_ids, self.pair)
        self.assertEqual(self.level.windows, [frame])
        self.assertEqual(
            self.workspace.doors_workspace.data().window_placements, [placement]
        )
        self.assertEqual(
            self.workspace.doors_workspace.data().get_window(window.window_id),
            window,
        )


# ### Direct test runner ###
if __name__ == "__main__":
    unittest.main()
