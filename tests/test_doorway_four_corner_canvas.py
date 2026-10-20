"""Canvas placement keeps a doorway and its four wall corners together."""

# ### Environment setup ###
from __future__ import annotations

import copy
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ### Imports ###
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.blueprint_canvas import BlueprintCanvas
from housemaker.doorway_topology import doorway_corner_positions
from housemaker.models import (
    DOORWAY_SHAPE_ARCH,
    PIXEL_TO_METER,
    DoorwayPreset,
    LevelData,
    VertexData,
)
from housemaker.project_io import load_project, save_project

# ### Module state ###
_application = QApplication.instance() or QApplication([])
_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _add_edge(graph: VertexData, first: tuple, second: tuple) -> tuple[int, int]:
    start = graph.add_vertex(*first)
    end = graph.add_vertex(*second)
    graph.add_edge(start.id, end.id)
    return start.id, end.id


def _solid_faces() -> VertexData:
    graph = VertexData()
    _add_edge(graph, (10.0, 45.0), (90.0, 45.0))
    _add_edge(graph, (10.0, 55.0), (90.0, 55.0))
    return graph


def _four_corner_gap(*, connected: bool = False) -> tuple[VertexData, set[int]]:
    graph = VertexData()
    corners = {
        point: graph.add_vertex(*point).id
        for point in ((40.0, 45.0), (60.0, 45.0), (40.0, 55.0), (60.0, 55.0))
    }
    for (x, y), corner_id in corners.items():
        outside = graph.add_vertex(10.0 if x == 40.0 else 90.0, y)
        graph.add_edge(outside.id, corner_id)
    for x in (40.0, 60.0):
        graph.add_edge(corners[(x, 45.0)], corners[(x, 55.0)])
    if connected:
        for left_y in (45.0, 55.0):
            for right_y in (45.0, 55.0):
                graph.add_edge(corners[(40.0, left_y)], corners[(60.0, right_y)])
    return graph, set(corners.values())


def _graph_state(graph: VertexData) -> tuple:
    return tuple(graph.vertices), tuple(graph.edges), graph._next_vertex_id


# ### Placement and editing tests ###
class DoorwayFourCornerCanvasTests(unittest.TestCase):
    def setUp(self) -> None:
        self.canvases: list[BlueprintCanvas] = []

    def tearDown(self) -> None:
        for canvas in self.canvases:
            canvas.close()
            canvas.deleteLater()
        _application.processEvents()

    def _canvas(self, graph: VertexData) -> BlueprintCanvas:
        canvas = BlueprintCanvas()
        canvas.resize(640, 520)
        canvas.set_level_data(graph, [], image_path=None, doorways=[])
        canvas.blueprint_image = QImage(100, 100, QImage.Format.Format_RGB32)
        canvas.blueprint_image.fill(Qt.GlobalColor.white)
        canvas.show()
        _application.processEvents()
        self.canvases.append(canvas)
        return canvas

    @staticmethod
    def _preview(
        canvas: BlueprintCanvas,
        modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
    ) -> None:
        canvas.start_doorway_placement(
            DoorwayPreset(
                width_meters=0.9,
                height_meters=2.1,
                shape=DOORWAY_SHAPE_ARCH,
                arch_amount=0.625,
            )
        )
        canvas._update_pending_doorway(QPointF(50.0, 50.0), modifiers)

    def _assert_attached_corners(self, canvas: BlueprintCanvas) -> None:
        doorway = canvas.doorways[0]
        self.assertIsNotNone(doorway.corner_vertex_ids)
        for vertex_id, position in zip(
            doorway.corner_vertex_ids or (), doorway_corner_positions(doorway)
        ):
            vertex = canvas.vertex_data.get_vertex(vertex_id)
            self.assertIsNotNone(vertex)
            self.assertAlmostEqual(vertex.x, position[0])
            self.assertAlmostEqual(vertex.y, position[1])

    def test_single_wall_face_and_free_placement_are_rejected(self) -> None:
        single = VertexData()
        _add_edge(single, (10.0, 50.0), (90.0, 50.0))
        for graph in (VertexData(), single):
            for modifiers in (
                Qt.KeyboardModifier.NoModifier,
                Qt.KeyboardModifier.ShiftModifier,
            ):
                with self.subTest(edges=len(graph.edges), modifiers=modifiers):
                    canvas = self._canvas(graph)
                    before = _graph_state(graph)
                    self._preview(canvas, modifiers)
                    self.assertIsNone(canvas.pending_doorway)
                    canvas._commit_pending_doorway()
                    self.assertEqual(canvas.doorways, [])
                    self.assertEqual(canvas.undo_stack, [])
                    self.assertEqual(_graph_state(graph), before)

    def test_two_unconnected_vertices_are_not_a_valid_doorway(self) -> None:
        graph = VertexData()
        graph.add_vertex(40.0, 50.0)
        graph.add_vertex(60.0, 50.0)
        canvas = self._canvas(graph)
        self._preview(canvas, Qt.KeyboardModifier.ShiftModifier)
        self.assertIsNone(canvas.pending_doorway)
        canvas._commit_pending_doorway()
        self.assertEqual(canvas.doorways, [])

    def test_gap_fits_four_corners_and_removes_all_cross_connections(self) -> None:
        graph, corners = _four_corner_gap(connected=True)
        canvas = self._canvas(graph)
        before = _graph_state(graph)
        self._preview(canvas)
        self.assertIsNotNone(canvas.pending_doorway)
        self.assertEqual(_graph_state(graph), before)
        canvas._commit_pending_doorway()

        self.assertEqual(len(canvas.doorways), 1)
        doorway = canvas.doorways[0]
        self.assertEqual(set(doorway.corner_vertex_ids), corners)
        self.assertAlmostEqual(doorway.width_meters, 20.0 * PIXEL_TO_METER)
        self.assertAlmostEqual(doorway.depth_meters, 10.0 * PIXEL_TO_METER)
        self.assertEqual(doorway.shape, DOORWAY_SHAPE_ARCH)
        self.assertAlmostEqual(doorway.arch_amount, 0.625)
        self.assertEqual(len(graph.vertices), 8)
        self.assertEqual(len(graph.edges), 6)
        for edge in graph.edges:
            first = graph.get_vertex(edge.start_vertex_id)
            second = graph.get_vertex(edge.end_vertex_id)
            if first.id in corners and second.id in corners:
                self.assertAlmostEqual(first.x, second.x)
        self._assert_attached_corners(canvas)

    def test_solid_wall_preview_is_pure_and_commit_creates_four_cuts(self) -> None:
        for modifiers in (
            Qt.KeyboardModifier.NoModifier,
            Qt.KeyboardModifier.ShiftModifier,
        ):
            with self.subTest(modifiers=modifiers):
                graph = _solid_faces()
                canvas = self._canvas(graph)
                before = _graph_state(graph)
                self._preview(canvas, modifiers)
                self.assertIsNotNone(canvas.pending_doorway)
                self.assertAlmostEqual(canvas.pending_doorway.depth_meters, 0.2)
                self.assertEqual(_graph_state(graph), before)
                self.assertEqual(canvas.doorways, [])
                self.assertEqual(canvas.undo_stack, [])
                canvas._commit_pending_doorway()

                self.assertEqual(len(graph.vertices), 8)
                self.assertEqual(len(graph.edges), 6)
                self.assertEqual(len(canvas.doorways), 1)
                doorway = canvas.doorways[0]
                corner_ids = doorway.corner_vertex_ids
                self.assertEqual(len(set(corner_ids)), 4)
                self.assertTrue(graph.has_edge(corner_ids[0], corner_ids[2]))
                self.assertTrue(graph.has_edge(corner_ids[1], corner_ids[3]))
                self.assertFalse(graph.has_edge(corner_ids[0], corner_ids[1]))
                self.assertFalse(graph.has_edge(corner_ids[2], corner_ids[3]))
                self._assert_attached_corners(canvas)

    def test_mouse_click_commits_snapped_solid_wall_cut(self) -> None:
        canvas = self._canvas(_solid_faces())
        self._preview(canvas)
        position = canvas._image_to_widget(50.0, 45.0).toPoint()
        QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=position)
        _application.processEvents()
        self.assertEqual(len(canvas.doorways), 1)
        self.assertIsNone(canvas.pending_doorway_preset)
        self._assert_attached_corners(canvas)

    def test_one_undo_restores_graph_and_removes_doorway(self) -> None:
        graph = _solid_faces()
        canvas = self._canvas(graph)
        before = _graph_state(graph)
        self._preview(canvas)
        canvas._commit_pending_doorway()
        self.assertEqual(len(canvas.undo_stack), 1)
        canvas.undo_last_step()
        self.assertEqual(canvas.doorways, [])
        self.assertEqual(_graph_state(graph), before)
        self.assertEqual(canvas.undo_stack, [])

    def test_saved_doorway_retains_four_corner_links(self) -> None:
        graph = _solid_faces()
        canvas = self._canvas(graph)
        self._preview(canvas)
        canvas._commit_pending_doorway()
        level = LevelData(
            index=2, name="Ground", vertex_data=graph, doorways=canvas.doorways
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "four-corner-project.json"
            save_project(path, current_level_index=2, levels=[level])
            restored = next(
                level for level in load_project(path).levels if level.index == 2
            )
        self.assertEqual(restored.doorways, canvas.doorways)
        self.assertEqual(_graph_state(restored.vertex_data), _graph_state(graph))

    def test_deleting_linked_opening_restores_wall_spans_and_can_be_undone(
        self,
    ) -> None:
        graph = _solid_faces()
        canvas = self._canvas(graph)
        self._preview(canvas)
        canvas._commit_pending_doorway()
        before = _graph_state(graph)
        doorway = copy.deepcopy(canvas.doorways[0])
        corners = doorway.corner_vertex_ids

        self.assertTrue(canvas._delete_selected_doorway())

        self.assertEqual(canvas.doorways, [])
        self.assertTrue(graph.has_edge(corners[0], corners[1]))
        self.assertTrue(graph.has_edge(corners[2], corners[3]))
        canvas.undo_last_step()
        self.assertEqual(_graph_state(graph), before)
        self.assertEqual(canvas.doorways, [doorway])

    def test_deleting_one_corner_removes_its_attached_doorway(self) -> None:
        graph = _solid_faces()
        canvas = self._canvas(graph)
        self._preview(canvas)
        canvas._commit_pending_doorway()
        doorway = copy.deepcopy(canvas.doorways[0])
        canvas.set_selected_vertex_ids((doorway.corner_vertex_ids[0],))

        canvas._delete_selected_vertices()

        self.assertEqual(canvas.doorways, [])
        canvas.undo_last_step()
        self.assertEqual(canvas.doorways, [doorway])

    def test_an_already_owned_four_corner_opening_cannot_be_placed_twice(self) -> None:
        canvas = self._canvas(_solid_faces())
        self._preview(canvas)
        canvas._commit_pending_doorway()

        self._preview(canvas)
        canvas._commit_pending_doorway()

        self.assertIsNone(canvas.pending_doorway)
        self.assertEqual(len(canvas.doorways), 1)
        self.assertEqual(len(canvas.undo_stack), 1)

    def test_depth_resize_moves_linked_corners_during_preview(self) -> None:
        graph = _solid_faces()
        canvas = self._canvas(graph)
        self._preview(canvas)
        canvas._commit_pending_doorway()
        original = copy.deepcopy(canvas.doorways[0])
        canvas.drag_doorway_index = 0
        canvas.doorway_drag_initial_doorway = original
        canvas.doorway_drag_depth_side_sign = 1.0
        depth_axis = canvas._get_doorway_depth_direction(original)
        target = QPointF(
            original.center_x + depth_axis[0] * 20.0,
            original.center_y + depth_axis[1] * 20.0,
        )
        canvas._resize_dragged_doorway_depth(target)
        self.assertGreater(canvas.doorways[0].depth_meters, original.depth_meters)
        self._assert_attached_corners(canvas)

    def test_width_resize_moves_linked_corners_without_reconnecting_gap(self) -> None:
        graph = _solid_faces()
        canvas = self._canvas(graph)
        self._preview(canvas)
        canvas._commit_pending_doorway()
        original = copy.deepcopy(canvas.doorways[0])
        canvas.pressed_doorway_index = 0
        canvas.drag_doorway_index = 0
        canvas.doorway_drag_initial_doorway = original
        canvas.doorway_drag_wall_edge = canvas._find_doorway_wall_span_edge(original)
        canvas.doorway_drag_width_side_sign = 1.0
        self.assertIsNotNone(canvas.doorway_drag_wall_edge)

        canvas._resize_dragged_doorway_width(QPointF(82.0, 50.0))

        self.assertGreater(canvas.doorways[0].width_meters, original.width_meters)
        self._assert_attached_corners(canvas)
        self.assertTrue(canvas._commit_extended_doorway_bridges())
        corners = canvas.doorways[0].corner_vertex_ids
        self.assertFalse(graph.has_edge(corners[0], corners[1]))
        self.assertFalse(graph.has_edge(corners[2], corners[3]))
        self.assertEqual(len(graph.edges), 6)

    def test_moving_doorway_moves_all_four_corner_vertices_immediately(self) -> None:
        graph = _solid_faces()
        canvas = self._canvas(graph)
        self._preview(canvas)
        canvas._commit_pending_doorway()
        original = copy.deepcopy(canvas.doorways[0])
        canvas.drag_doorway_index = 0
        canvas.doorway_drag_initial_doorway = original
        canvas.doorway_drag_press_image_point = (50.0, 50.0)
        canvas.doorway_drag_wall_edge = canvas._find_doorway_wall_span_edge(original)
        self.assertIsNotNone(canvas.doorway_drag_wall_edge)

        canvas._move_dragged_doorway(QPointF(60.0, 50.0))

        moved = canvas.doorways[0]
        self.assertAlmostEqual(moved.center_x, 60.0)
        self.assertAlmostEqual(moved.center_y, 50.0)
        self.assertAlmostEqual(moved.width_meters, original.width_meters)
        self.assertAlmostEqual(moved.depth_meters, original.depth_meters)
        self._assert_attached_corners(canvas)

    def test_stale_preview_does_not_cut_graph_or_add_empty_undo_entry(self) -> None:
        graph = _solid_faces()
        canvas = self._canvas(graph)
        self._preview(canvas)
        graph.add_vertex(10.0, 10.0)
        first_edge = graph.edges[0]
        graph.edges.remove(first_edge)
        graph.mark_changed()
        before = _graph_state(graph)

        canvas._commit_pending_doorway()

        self.assertEqual(_graph_state(graph), before)
        self.assertEqual(canvas.doorways, [])
        self.assertEqual(canvas.undo_stack, [])


if __name__ == "__main__":
    unittest.main()
