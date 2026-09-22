# ### Environment setup ###
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.level_coordinates import (
    level_image_to_world_xy,
    level_world_to_image_xy,
)
from housemaker.main import BlueprintWorkspace
from housemaker.models import DoorwayData, LevelData, RoomData, VertexData
from housemaker.wall_mirroring import WallMirrorVertexLink

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_level(
    level_index: int,
    *,
    scale: float = 1.0,
    offset_x_meters: float = 0.0,
    offset_y_meters: float = 0.0,
) -> LevelData:
    """Return one square wall loop with a separate room-center vertex."""

    vertex_data = VertexData()
    boundary_ids = tuple(
        vertex_data.add_vertex(*point).id
        for point in (
            (20.0, 30.0),
            (120.0, 30.0),
            (120.0, 130.0),
            (20.0, 130.0),
        )
    )
    for start_id, end_id in zip(
        boundary_ids,
        (*boundary_ids[1:], boundary_ids[0]),
    ):
        vertex_data.add_edge(start_id, end_id)
    room_center = vertex_data.add_vertex(70.0, 80.0)
    vertex_data.add_vertex(170.0, 80.0)
    room = RoomData(
        name=f"Room {level_index}",
        vertex_ids=boundary_ids,
        center_vertex_id=room_center.id,
        color_rgb=(120, 150, 180),
    )
    return LevelData(
        index=level_index,
        name=f"Level {level_index}",
        scale=scale,
        offset_x_meters=offset_x_meters,
        offset_y_meters=offset_y_meters,
        image_size_pixels=(200.0, 160.0),
        vertex_data=vertex_data,
        rooms=[room],
    )


def _send_ctrl_z(workspace: BlueprintWorkspace) -> None:
    workspace.viewer.view.keyPressEvent(
        QKeyEvent(
            QKeyEvent.Type.KeyPress,
            Qt.Key.Key_Z,
            Qt.KeyboardModifier.ControlModifier,
        )
    )
    _qt_application.processEvents()


def _link_key(link: WallMirrorVertexLink) -> tuple[int, int, int, int]:
    """Return the stable source-to-target identity of one mirror link."""

    return (
        link.source_level_index,
        link.source_vertex_id,
        link.target_level_index,
        link.target_vertex_id,
    )


# ### Wall-mirror Main integration tests ###
class WallLevelMirroringMainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(
                Path(self.temporary_directory.name) / "settings.json"
            )
        )
        self.lower_level = _build_level(1)
        self.source_level = _build_level(
            2,
            scale=1.5,
            offset_x_meters=2.75,
            offset_y_meters=-1.25,
        )
        self.upper_level = _build_level(
            3,
            scale=0.75,
            offset_x_meters=-3.5,
            offset_y_meters=4.25,
        )
        self.workspace._apply_project_state(
            [self.lower_level, self.source_level, self.upper_level],
            1,
        )
        self.workspace.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        self.workspace.deleteLater()
        _qt_application.processEvents()
        self.temporary_directory.cleanup()

    def _select_source_wall_vertices(self, *vertex_ids: int) -> None:
        self.assertIs(self.workspace.current_level, self.source_level)
        self.workspace.canvas.set_selected_vertex_ids(vertex_ids)
        _qt_application.processEvents()

    def _find_link(
        self,
        source_level_index: int,
        source_vertex_id: int,
        target_level_index: int,
    ) -> object:
        matches = tuple(
            link
            for link in self.workspace.wall_mirror_links
            if (
                link.source_level_index == source_level_index
                and link.source_vertex_id == source_vertex_id
                and link.target_level_index == target_level_index
            )
        )
        self.assertEqual(len(matches), 1)
        return matches[0]

    def _target_ids(
        self,
        source_vertex_ids: tuple[int, ...],
        target_level_index: int,
    ) -> dict[int, int]:
        return {
            source_vertex_id: int(
                self._find_link(
                    self.source_level.index,
                    source_vertex_id,
                    target_level_index,
                ).target_vertex_id
            )
            for source_vertex_id in source_vertex_ids
        }

    def test_group_up_copies_only_edges_induced_by_the_selection(self) -> None:
        self._select_source_wall_vertices(1, 2, 3)

        self.workspace.wall_mirror_up_button.click()

        target_ids = self._target_ids((1, 2, 3), self.upper_level.index)
        mirrored_ids = set(target_ids.values())
        mirrored_edge_keys = {
            tuple(sorted((edge.start_vertex_id, edge.end_vertex_id)))
            for edge in self.upper_level.vertex_data.edges
            if (
                edge.start_vertex_id in mirrored_ids
                and edge.end_vertex_id in mirrored_ids
            )
        }
        self.assertEqual(
            mirrored_edge_keys,
            {
                tuple(sorted((target_ids[1], target_ids[2]))),
                tuple(sorted((target_ids[2], target_ids[3]))),
            },
        )
        self.assertFalse(
            any(
                (edge.start_vertex_id in mirrored_ids)
                != (edge.end_vertex_id in mirrored_ids)
                for edge in self.upper_level.vertex_data.edges
            ),
            "An edge to an unselected source endpoint was copied.",
        )
        self.assertFalse(
            any(
                link.source_vertex_id == 4
                and link.target_level_index == self.upper_level.index
                for link in self.workspace.wall_mirror_links
            )
        )

    def test_mirrored_vertices_preserve_source_world_positions(self) -> None:
        self._select_source_wall_vertices(1, 2, 3)

        self.workspace.wall_mirror_up_button.click()

        target_ids = self._target_ids((1, 2, 3), self.upper_level.index)
        for source_vertex_id, target_vertex_id in target_ids.items():
            with self.subTest(source_vertex_id=source_vertex_id):
                source_vertex = self.source_level.vertex_data.get_vertex(
                    source_vertex_id
                )
                target_vertex = self.upper_level.vertex_data.get_vertex(
                    target_vertex_id
                )
                assert source_vertex is not None and target_vertex is not None
                source_world = level_image_to_world_xy(
                    self.source_level,
                    source_vertex.x,
                    source_vertex.y,
                )
                expected_target = level_world_to_image_xy(
                    self.upper_level,
                    *source_world,
                )
                target_world = level_image_to_world_xy(
                    self.upper_level,
                    target_vertex.x,
                    target_vertex.y,
                )
                self.assertAlmostEqual(target_vertex.x, expected_target[0])
                self.assertAlmostEqual(target_vertex.y, expected_target[1])
                self.assertAlmostEqual(target_world[0], source_world[0])
                self.assertAlmostEqual(target_world[1], source_world[1])

    def test_partial_group_reuses_the_nearest_existing_target(self) -> None:
        second_upper_level = _build_level(4)
        self.workspace.levels.append(second_upper_level)
        self._select_source_wall_vertices(1)
        self.workspace.wall_mirror_up_button.click()
        first_target_id = self._find_link(2, 1, 3).target_vertex_id

        self._select_source_wall_vertices(1, 2, 3)
        self.workspace.wall_mirror_up_button.click()

        target_ids = self._target_ids((1, 2, 3), 3)
        self.assertEqual(target_ids[1], first_target_id)
        self.assertFalse(
            any(
                link.source_vertex_id in (1, 2, 3)
                and link.target_level_index == 4
                for link in self.workspace.wall_mirror_links
            )
        )
        expected_key = tuple(sorted((target_ids[1], target_ids[2])))
        self.assertEqual(
            sum(
                tuple(sorted((edge.start_vertex_id, edge.end_vertex_id)))
                == expected_key
                for edge in self.upper_level.vertex_data.edges
            ),
            1,
        )

    def test_repeated_group_up_uses_each_level_once_then_is_a_noop(
        self,
    ) -> None:
        second_upper_level = _build_level(4)
        self.workspace.levels.append(second_upper_level)
        self._select_source_wall_vertices(1, 2)

        self.workspace.wall_mirror_up_button.click()
        self.workspace.wall_mirror_up_button.click()

        self.assertEqual(
            {
                (link.source_vertex_id, link.target_level_index)
                for link in self.workspace.wall_mirror_links
                if link.source_level_index == self.source_level.index
            },
            {(1, 3), (2, 3), (1, 4), (2, 4)},
        )
        self.assertEqual(len(self.workspace._canvas_undo_stack), 2)
        self.assertFalse(self.workspace.wall_mirror_up_button.isEnabled())

        self.workspace._handle_wall_mirror_up_clicked()

        self.assertEqual(len(self.workspace.wall_mirror_links), 4)
        self.assertEqual(len(self.workspace._canvas_undo_stack), 2)

    def test_group_mirror_is_one_globally_undoable_topology_action(
        self,
    ) -> None:
        source_before = self.source_level.vertex_data.clone()
        upper_before = self.upper_level.vertex_data.clone()
        self._select_source_wall_vertices(1, 2, 3)

        self.workspace.wall_mirror_up_button.click()

        self.assertEqual(len(self.workspace._canvas_undo_stack), 1)
        self.assertEqual(len(self.workspace.wall_mirror_links), 3)
        self.assertNotEqual(self.upper_level.vertex_data, upper_before)

        _send_ctrl_z(self.workspace)

        self.assertEqual(self.source_level.vertex_data, source_before)
        self.assertEqual(self.upper_level.vertex_data, upper_before)
        self.assertEqual(self.workspace.wall_mirror_links, ())
        self.assertEqual(self.workspace._canvas_undo_stack, [])

    def test_selected_source_group_deletes_and_restores_as_one_action(
        self,
    ) -> None:
        self._select_source_wall_vertices(1, 2)
        self.workspace.wall_mirror_up_button.click()
        target_ids = self._target_ids((1, 2), self.upper_level.index)
        source_before_delete = self.source_level.vertex_data.clone()
        upper_before_delete = self.upper_level.vertex_data.clone()
        links_before_delete = self.workspace.wall_mirror_links
        undo_count_before_delete = len(self.workspace._canvas_undo_stack)

        self.workspace.canvas._delete_selected_vertices()
        _qt_application.processEvents()

        for source_vertex_id in (1, 2):
            self.assertIsNone(
                self.source_level.vertex_data.get_vertex(source_vertex_id)
            )
            self.assertIsNone(
                self.upper_level.vertex_data.get_vertex(
                    target_ids[source_vertex_id]
                )
            )
        self.assertEqual(self.workspace.wall_mirror_links, ())
        self.assertEqual(
            len(self.workspace._canvas_undo_stack),
            undo_count_before_delete + 1,
        )

        _send_ctrl_z(self.workspace)

        self.assertEqual(self.source_level.vertex_data, source_before_delete)
        self.assertEqual(self.upper_level.vertex_data, upper_before_delete)
        self.assertEqual(
            self.workspace.wall_mirror_links,
            links_before_delete,
        )
        self.assertEqual(self.workspace.canvas.selected_vertex_ids, (1, 2))

    def test_removing_mirrored_wall_removes_and_undo_restores_its_doorway(
        self,
    ) -> None:
        self._select_source_wall_vertices(1, 2)
        self.workspace.wall_mirror_up_button.click()
        target_ids = self._target_ids((1, 2), self.upper_level.index)
        first = self.upper_level.vertex_data.get_vertex(target_ids[1])
        second = self.upper_level.vertex_data.get_vertex(target_ids[2])
        assert first is not None and second is not None
        doorway = DoorwayData(
            center_x=(first.x + second.x) / 2.0,
            center_y=(first.y + second.y) / 2.0,
            width_meters=0.9,
            height_meters=2.1,
            rotation_degrees=90.0,
        )
        self.upper_level.doorways.append(doorway)
        self.workspace._reset_viewer_doorway_snapshots()

        self.workspace.wall_mirror_undo_button.click()
        self.assertEqual(self.upper_level.doorways, [])
        self.workspace._commit_pending_wall_vertex_update()
        self.assertEqual(
            self.workspace._viewer_doorways_by_level_index[self.upper_level.index],
            (),
        )

        _send_ctrl_z(self.workspace)
        self.assertEqual(self.upper_level.doorways, [doorway])
        self.assertEqual(
            self.workspace._viewer_doorways_by_level_index[self.upper_level.index],
            (doorway,),
        )

    def test_deleting_source_wall_restores_mirrored_doorway_on_undo(self) -> None:
        self._select_source_wall_vertices(1, 2)
        self.workspace.wall_mirror_up_button.click()
        target_ids = self._target_ids((1, 2), self.upper_level.index)
        first = self.upper_level.vertex_data.get_vertex(target_ids[1])
        second = self.upper_level.vertex_data.get_vertex(target_ids[2])
        assert first is not None and second is not None
        doorway = DoorwayData(
            center_x=(first.x + second.x) / 2.0,
            center_y=(first.y + second.y) / 2.0,
            width_meters=0.9,
            height_meters=2.1,
            rotation_degrees=90.0,
        )
        self.upper_level.doorways.append(doorway)
        self.workspace._reset_viewer_doorway_snapshots()

        self.workspace.canvas._delete_selected_vertices()
        self.workspace._commit_pending_wall_vertex_update()
        self.assertEqual(self.upper_level.doorways, [])

        _send_ctrl_z(self.workspace)
        self.assertEqual(self.upper_level.doorways, [doorway])
        self.assertEqual(
            self.workspace._viewer_doorways_by_level_index[self.upper_level.index],
            (doorway,),
        )

    def test_dedicated_undo_clears_selected_outgoing_mirrors_atomically(
        self,
    ) -> None:
        self._select_source_wall_vertices(1, 2)
        self.workspace.wall_mirror_up_button.click()
        self.workspace.wall_mirror_down_button.click()
        topology_before_undo = tuple(
            (level.index, level.vertex_data.clone())
            for level in self.workspace.levels
        )
        links_before_undo = tuple(self.workspace.wall_mirror_links)

        self.workspace.wall_mirror_undo_button.click()

        self.assertEqual(self.workspace.wall_mirror_links, ())
        self.assertEqual(len(self.workspace._canvas_undo_stack), 3)
        self.assertFalse(self.workspace.wall_mirror_undo_button.isEnabled())

        _send_ctrl_z(self.workspace)

        self.assertEqual(self.workspace.wall_mirror_links, links_before_undo)
        for level_index, expected_vertex_data in topology_before_undo:
            level = next(
                item
                for item in self.workspace.levels
                if item.index == level_index
            )
            self.assertEqual(level.vertex_data, expected_vertex_data)

    def test_dedicated_undo_removes_selected_incoming_mirrors(self) -> None:
        self._select_source_wall_vertices(1, 2)
        self.workspace.wall_mirror_up_button.click()
        links_before_undo = tuple(self.workspace.wall_mirror_links)
        target_ids = self._target_ids((1, 2), self.upper_level.index)
        upper_before_undo = self.upper_level.vertex_data.clone()

        self.workspace._handle_level_selection_changed(2)
        self.assertIs(self.workspace.current_level, self.upper_level)
        self.workspace.canvas.set_selected_vertex_ids(target_ids.values())
        _qt_application.processEvents()
        self.assertTrue(self.workspace.wall_mirror_undo_button.isEnabled())

        self.workspace.wall_mirror_undo_button.click()

        self.assertEqual(self.workspace.wall_mirror_links, ())
        self.assertTrue(
            all(
                self.upper_level.vertex_data.get_vertex(vertex_id) is None
                for vertex_id in target_ids.values()
            )
        )

        _send_ctrl_z(self.workspace)

        self.assertEqual(self.workspace.wall_mirror_links, links_before_undo)
        self.assertEqual(self.upper_level.vertex_data, upper_before_undo)
        self.assertEqual(
            self.workspace.canvas.selected_vertex_ids,
            tuple(target_ids.values()),
        )

    def test_target_mirrors_are_selectable_local_green_vertices(self) -> None:
        self._select_source_wall_vertices(1, 2)
        self.workspace.wall_mirror_up_button.click()
        target_ids = self._target_ids((1, 2), self.upper_level.index)

        self.workspace._handle_level_selection_changed(2)

        self.assertTrue(
            all(
                self.workspace.canvas.vertex_data.get_vertex(vertex_id)
                is not None
                for vertex_id in target_ids.values()
            )
        )
        self.assertTrue(
            set(target_ids.values()).issubset(
                self.workspace.canvas.get_wall_mirror_vertex_ids()
            )
        )
        self.assertTrue(
            self.workspace.canvas.select_vertex(target_ids[1])
        )
        self.assertEqual(
            self.workspace.canvas.selected_vertex_ids,
            (target_ids[1],),
        )
        _qt_application.processEvents()
        self.assertTrue(self.workspace.wall_mirror_undo_button.isEnabled())

    def test_removing_a_clicked_target_clears_the_active_drawing_chain(
        self,
    ) -> None:
        self._select_source_wall_vertices(1)
        self.workspace.wall_mirror_up_button.click()
        target_id = self._target_ids((1,), self.upper_level.index)[1]
        self.workspace._handle_level_selection_changed(2)
        self.workspace.canvas.blueprint_image = QImage(
            200,
            160,
            QImage.Format.Format_RGB32,
        )
        target_vertex = self.upper_level.vertex_data.get_vertex(target_id)
        assert target_vertex is not None

        QTest.mouseClick(
            self.workspace.canvas,
            Qt.MouseButton.LeftButton,
            pos=self.workspace.canvas._image_to_widget(
                target_vertex.x,
                target_vertex.y,
            ).toPoint(),
        )
        self.assertEqual(self.workspace.canvas.active_vertex_id, target_id)

        self.workspace.wall_mirror_undo_button.click()

        self.assertIsNone(self.workspace.canvas.active_vertex_id)
        self.assertIsNone(self.workspace.canvas.pressed_vertex_id)
        self.assertIsNone(self.workspace.canvas.drag_vertex_id)
        QTest.mouseClick(
            self.workspace.canvas,
            Qt.MouseButton.LeftButton,
            pos=self.workspace.canvas._image_to_widget(190.0, 150.0).toPoint(),
        )
        known_vertex_ids = {
            vertex.id for vertex in self.upper_level.vertex_data.vertices
        }
        self.assertTrue(
            all(
                edge.start_vertex_id in known_vertex_ids
                and edge.end_vertex_id in known_vertex_ids
                for edge in self.upper_level.vertex_data.edges
            )
        )

    def test_room_centers_and_isolated_vertices_are_not_mirror_sources(
        self,
    ) -> None:
        room_center_id = self.source_level.rooms[0].center_vertex_id
        isolated_vertex_id = self.source_level.vertex_data.vertices[-1].id

        for vertex_id in (room_center_id, isolated_vertex_id):
            with self.subTest(vertex_id=vertex_id):
                self._select_source_wall_vertices(vertex_id)
                self.assertFalse(self.workspace.wall_mirror_up_button.isEnabled())
                self.assertFalse(
                    self.workspace.wall_mirror_down_button.isEnabled()
                )
                self.assertFalse(
                    self.workspace.wall_mirror_undo_button.isEnabled()
                )

    def test_canonical_links_are_stably_ordered_and_unique(self) -> None:
        self._select_source_wall_vertices(3, 2, 1)
        self.workspace.wall_mirror_up_button.click()

        link_keys = tuple(map(_link_key, self.workspace.wall_mirror_links))

        self.assertEqual(link_keys, tuple(sorted(link_keys)))
        self.assertEqual(len(link_keys), len(set(link_keys)))


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
