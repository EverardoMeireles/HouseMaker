# ### Environment setup ###
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QImage, QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.blueprint_canvas import BlueprintCanvas
from housemaker.models import RoomData, VertexData

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Event helpers ###
def _send_drag_move(
    canvas: BlueprintCanvas,
    position: QPoint,
    button: Qt.MouseButton = Qt.MouseButton.LeftButton,
) -> None:
    event = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(position),
        QPointF(canvas.mapToGlobal(position)),
        Qt.MouseButton.NoButton,
        button,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(canvas, event)


# ### Wall vertex event tests ###
class BlueprintWallVertexEventTests(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas = BlueprintCanvas()
        self.canvas.resize(640, 520)
        self.canvas.blueprint_image = QImage(
            100,
            100,
            QImage.Format.Format_RGB32,
        )
        self.canvas.blueprint_image.fill(Qt.GlobalColor.white)
        self.canvas.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.canvas.close()
        self.canvas.deleteLater()
        _qt_application.processEvents()

    def _record_geometry_events(self) -> list[str]:
        events: list[str] = []
        self.canvas.wall_vertex_added.connect(
            lambda: events.append("wall_vertex_added")
        )
        self.canvas.geometry_changed.connect(
            lambda: events.append("geometry_changed")
        )
        return events

    def test_new_free_vertex_emits_wall_event_before_geometry_event(
        self,
    ) -> None:
        events = self._record_geometry_events()

        self.canvas._handle_new_vertex_click((25.0, 40.0))

        self.assertEqual(
            events,
            ["wall_vertex_added", "geometry_changed"],
        )
        self.assertEqual(len(self.canvas.vertex_data.vertices), 1)
        vertex = self.canvas.vertex_data.vertices[0]
        self.assertEqual((vertex.x, vertex.y), (25.0, 40.0))
        self.assertEqual(self.canvas.active_vertex_id, vertex.id)
        self.assertEqual(self.canvas.selected_vertex_id, vertex.id)

    def test_blank_add_interaction_wraps_the_mutation_events(self) -> None:
        events: list[str] = []
        self.canvas.wall_vertex_interaction_changed.connect(
            lambda active: events.append(f"interaction:{active}")
        )
        self.canvas.wall_vertex_added.connect(
            lambda: events.append("wall_vertex_added")
        )
        self.canvas.geometry_changed.connect(
            lambda: events.append("geometry_changed")
        )
        position = self.canvas._image_to_widget(25.0, 40.0).toPoint()

        QTest.mousePress(
            self.canvas,
            Qt.MouseButton.LeftButton,
            pos=position,
        )

        self.assertEqual(
            events,
            [
                "interaction:True",
                "wall_vertex_added",
                "geometry_changed",
            ],
        )

        QTest.mouseRelease(
            self.canvas,
            Qt.MouseButton.LeftButton,
            pos=position,
        )

        self.assertEqual(events[-1], "interaction:False")

    def test_new_vertex_on_edge_emits_wall_event_before_geometry_event(
        self,
    ) -> None:
        vertex_data = VertexData()
        first = vertex_data.add_vertex(0.0, 0.0)
        second = vertex_data.add_vertex(100.0, 0.0)
        edge = vertex_data.add_edge(first.id, second.id)
        assert edge is not None
        self.canvas.vertex_data = vertex_data
        events = self._record_geometry_events()

        self.canvas._handle_new_vertex_on_edge_click((40.0, 0.0), edge)

        self.assertEqual(
            events,
            ["wall_vertex_added", "geometry_changed"],
        )
        self.assertEqual(len(vertex_data.vertices), 3)
        self.assertEqual(len(vertex_data.edges), 2)
        middle = vertex_data.vertices[-1]
        self.assertTrue(vertex_data.has_edge(first.id, middle.id))
        self.assertTrue(vertex_data.has_edge(middle.id, second.id))

    def test_connecting_existing_vertices_is_only_a_general_change(
        self,
    ) -> None:
        vertex_data = VertexData()
        first = vertex_data.add_vertex(0.0, 0.0)
        second = vertex_data.add_vertex(100.0, 0.0)
        self.canvas.vertex_data = vertex_data
        self.canvas.active_vertex_id = first.id
        events = self._record_geometry_events()

        self.canvas._handle_existing_vertex_click(second)

        self.assertEqual(events, ["geometry_changed"])
        self.assertTrue(vertex_data.has_edge(first.id, second.id))

    def test_reconnecting_an_existing_edge_skips_undo_and_geometry(self) -> None:
        vertex_data = VertexData()
        first = vertex_data.add_vertex(20.0, 40.0)
        second = vertex_data.add_vertex(80.0, 40.0)
        vertex_data.add_edge(second.id, first.id)
        self.canvas.vertex_data = vertex_data
        self.canvas.active_vertex_id = first.id
        self.canvas.selected_vertex_id = first.id
        events = self._record_geometry_events()

        QTest.mouseClick(
            self.canvas,
            Qt.MouseButton.LeftButton,
            pos=self.canvas._image_to_widget(
                second.x,
                second.y,
            ).toPoint(),
        )

        self.assertEqual(events, [])
        self.assertEqual(self.canvas.undo_stack, [])
        self.assertEqual(len(vertex_data.edges), 1)
        self.assertEqual(self.canvas.active_vertex_id, second.id)
        self.assertEqual(self.canvas.selected_vertex_id, second.id)

    def test_deleting_existing_vertex_is_only_a_general_change(self) -> None:
        vertex = self.canvas.vertex_data.add_vertex(10.0, 20.0)
        self.canvas.selected_vertex_id = vertex.id
        events = self._record_geometry_events()

        self.canvas._delete_selected_vertices()

        self.assertEqual(events, ["geometry_changed"])
        self.assertIsNone(self.canvas.vertex_data.get_vertex(vertex.id))

    def test_delete_removes_the_full_vertex_selection_in_one_action(self) -> None:
        vertex_data = VertexData()
        first = vertex_data.add_vertex(10.0, 10.0)
        second = vertex_data.add_vertex(70.0, 10.0)
        third = vertex_data.add_vertex(70.0, 70.0)
        fourth = vertex_data.add_vertex(10.0, 70.0)
        center = vertex_data.add_vertex(40.0, 40.0)
        unrelated_start = vertex_data.add_vertex(80.0, 80.0)
        unrelated_end = vertex_data.add_vertex(90.0, 90.0)
        vertex_data.add_edge(first.id, second.id)
        vertex_data.add_edge(second.id, third.id)
        vertex_data.add_edge(third.id, fourth.id)
        vertex_data.add_edge(fourth.id, first.id)
        vertex_data.add_edge(unrelated_start.id, unrelated_end.id)
        self.canvas.vertex_data = vertex_data
        self.canvas.rooms.append(
            RoomData(
                name="Selected room",
                vertex_ids=(first.id, second.id, third.id, fourth.id),
                center_vertex_id=center.id,
                color_rgb=(100, 120, 140),
            )
        )
        self.canvas.set_selected_vertex_ids((first.id, third.id))
        self.canvas.active_vertex_id = first.id
        self.canvas.preview_point = (25.0, 25.0)
        self.canvas.preview_guides = [object()]  # type: ignore[list-item]
        events = self._record_geometry_events()
        self.canvas.rooms_changed.connect(lambda: events.append("rooms_changed"))

        QTest.keyClick(self.canvas, Qt.Key.Key_Delete)

        self.assertEqual(events, ["rooms_changed", "geometry_changed"])
        self.assertEqual(
            [vertex.id for vertex in vertex_data.vertices],
            [
                second.id,
                fourth.id,
                center.id,
                unrelated_start.id,
                unrelated_end.id,
            ],
        )
        self.assertEqual(len(vertex_data.edges), 1)
        self.assertTrue(
            vertex_data.has_edge(unrelated_start.id, unrelated_end.id)
        )
        self.assertEqual(self.canvas.rooms, [])
        self.assertEqual(self.canvas.selected_vertex_ids, ())
        self.assertIsNone(self.canvas.active_vertex_id)
        self.assertIsNone(self.canvas.preview_point)
        self.assertEqual(self.canvas.preview_guides, [])
        self.assertEqual(len(self.canvas.undo_stack), 1)

        self.canvas.undo_last_step()

        self.assertEqual(
            [vertex.id for vertex in vertex_data.vertices],
            [
                first.id,
                second.id,
                third.id,
                fourth.id,
                center.id,
                unrelated_start.id,
                unrelated_end.id,
            ],
        )
        self.assertEqual(len(vertex_data.edges), 5)
        self.assertEqual(len(self.canvas.rooms), 1)
        self.assertEqual(
            self.canvas.selected_vertex_ids,
            (first.id, third.id),
        )
        self.assertEqual(self.canvas.active_vertex_id, first.id)

    def test_delete_ignores_an_entirely_stale_vertex_selection(self) -> None:
        self.canvas.set_selected_vertex_ids((101, 102))
        events = self._record_geometry_events()

        QTest.keyClick(self.canvas, Qt.Key.Key_Delete)

        self.assertEqual(events, [])
        self.assertEqual(self.canvas.selected_vertex_ids, ())
        self.assertEqual(self.canvas.undo_stack, [])

    def test_dragged_new_vertex_emits_geometry_only_after_release(
        self,
    ) -> None:
        self.canvas._handle_new_vertex_click((25.0, 40.0))
        events = self._record_geometry_events()
        interaction_states: list[bool] = []
        self.canvas.wall_vertex_interaction_changed.connect(
            interaction_states.append
        )
        start = self.canvas._image_to_widget(25.0, 40.0).toPoint()
        target = self.canvas._image_to_widget(60.0, 70.0).toPoint()

        QTest.mousePress(
            self.canvas,
            Qt.MouseButton.LeftButton,
            pos=start,
        )
        _send_drag_move(self.canvas, target)

        self.assertEqual(events, [])
        self.assertEqual(interaction_states, [True])
        moved_vertex = self.canvas.vertex_data.vertices[0]
        self.assertNotEqual((moved_vertex.x, moved_vertex.y), (25.0, 40.0))

        QTest.mouseRelease(
            self.canvas,
            Qt.MouseButton.LeftButton,
            pos=target,
        )

        self.assertEqual(events, ["geometry_changed"])
        self.assertEqual(interaction_states, [True, False])

    def test_new_vertex_press_without_movement_emits_no_geometry_change(
        self,
    ) -> None:
        self.canvas._handle_new_vertex_click((25.0, 40.0))
        events = self._record_geometry_events()
        interaction_states: list[bool] = []
        self.canvas.wall_vertex_interaction_changed.connect(
            interaction_states.append
        )
        position = self.canvas._image_to_widget(25.0, 40.0).toPoint()

        QTest.mousePress(
            self.canvas,
            Qt.MouseButton.LeftButton,
            pos=position,
        )
        QTest.mouseRelease(
            self.canvas,
            Qt.MouseButton.LeftButton,
            pos=position,
        )

        self.assertEqual(events, [])
        self.assertEqual(interaction_states, [True, False])

    def test_shift_click_builds_an_ordered_vertex_selection_without_edges(
        self,
    ) -> None:
        vertex_data = VertexData()
        first = vertex_data.add_vertex(20.0, 20.0)
        second = vertex_data.add_vertex(80.0, 20.0)
        third = vertex_data.add_vertex(80.0, 80.0)
        self.canvas.vertex_data = vertex_data

        QTest.mouseClick(
            self.canvas,
            Qt.MouseButton.LeftButton,
            pos=self.canvas._image_to_widget(first.x, first.y).toPoint(),
        )
        QTest.mouseClick(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.ShiftModifier,
            pos=self.canvas._image_to_widget(second.x, second.y).toPoint(),
        )
        QTest.mouseClick(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.ShiftModifier,
            pos=self.canvas._image_to_widget(third.x, third.y).toPoint(),
        )

        self.assertEqual(
            self.canvas.selected_vertex_ids,
            (first.id, second.id, third.id),
        )
        self.assertEqual(self.canvas.selected_vertex_id, third.id)
        self.assertIsNone(self.canvas.active_vertex_id)
        self.assertEqual(vertex_data.edges, [])

        QTest.mouseClick(
            self.canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.ShiftModifier,
            pos=self.canvas._image_to_widget(second.x, second.y).toPoint(),
        )

        self.assertEqual(
            self.canvas.selected_vertex_ids,
            (first.id, third.id),
        )
        self.assertEqual(self.canvas.selected_vertex_id, third.id)

        QTest.mouseClick(
            self.canvas,
            Qt.MouseButton.LeftButton,
            pos=self.canvas._image_to_widget(second.x, second.y).toPoint(),
        )

        self.assertEqual(self.canvas.selected_vertex_ids, (second.id,))
        self.assertEqual(self.canvas.active_vertex_id, second.id)

    def test_shift_click_on_an_edge_or_blank_space_creates_nothing(self) -> None:
        vertex_data = VertexData()
        first = vertex_data.add_vertex(20.0, 20.0)
        second = vertex_data.add_vertex(80.0, 20.0)
        vertex_data.add_edge(first.id, second.id)
        self.canvas.vertex_data = vertex_data
        self.canvas.active_vertex_id = first.id
        self.canvas.selected_vertex_id = first.id

        for image_point in ((50.0, 20.0), (50.0, 75.0)):
            QTest.mouseClick(
                self.canvas,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.ShiftModifier,
                pos=self.canvas._image_to_widget(*image_point).toPoint(),
            )

        self.assertEqual(len(vertex_data.vertices), 2)
        self.assertEqual(len(vertex_data.edges), 1)
        self.assertEqual(self.canvas.selected_vertex_ids, (first.id,))
        self.assertIsNone(self.canvas.active_vertex_id)

    def test_snapshot_restores_the_full_vertex_selection(self) -> None:
        first = self.canvas.vertex_data.add_vertex(20.0, 20.0)
        second = self.canvas.vertex_data.add_vertex(80.0, 20.0)
        self.canvas.set_selected_vertex_ids((first.id, second.id))
        self.canvas._push_undo_state()
        self.canvas.selected_vertex_id = first.id

        self.canvas.undo_last_step()

        self.assertEqual(
            self.canvas.selected_vertex_ids,
            (first.id, second.id),
        )
        self.assertEqual(self.canvas.selected_vertex_id, second.id)

    def test_center_snap_finds_a_rectangle_center(self) -> None:
        vertex_data = VertexData()
        corners = (
            vertex_data.add_vertex(20.0, 20.0),
            vertex_data.add_vertex(80.0, 20.0),
            vertex_data.add_vertex(80.0, 80.0),
            vertex_data.add_vertex(20.0, 80.0),
        )
        self.canvas.vertex_data = vertex_data

        candidate = self.canvas._find_center_snap_candidate((51.0, 49.0))

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(
            candidate.source_vertex_ids,
            tuple(vertex.id for vertex in corners),
        )
        self.assertEqual(candidate.point, (50.0, 50.0))

    def test_canvas_does_not_paint_an_instruction_overlay(self) -> None:
        self.canvas.blueprint_path = "test-plan.png"
        former_overlay_point = self.canvas._image_to_widget(10.0, 7.0).toPoint()

        rendered = self.canvas.grab().toImage()

        self.assertEqual(
            rendered.pixelColor(former_overlay_point),
            Qt.GlobalColor.white,
        )

    def test_former_instruction_area_accepts_click_and_middle_pan(self) -> None:
        self.canvas.blueprint_path = "test-plan.png"
        self.canvas.zoom_scale = 2.0
        start = QPoint(110, 70)
        target = QPoint(145, 95)
        self.assertIsNotNone(self.canvas._widget_to_image(QPointF(start)))

        QTest.mouseClick(
            self.canvas,
            Qt.MouseButton.LeftButton,
            pos=start,
        )

        self.assertEqual(len(self.canvas.vertex_data.vertices), 1)
        initial_offset = QPointF(self.canvas.view_offset)
        QTest.mousePress(
            self.canvas,
            Qt.MouseButton.MiddleButton,
            pos=start,
        )
        _send_drag_move(
            self.canvas,
            target,
            Qt.MouseButton.MiddleButton,
        )
        QTest.mouseRelease(
            self.canvas,
            Qt.MouseButton.MiddleButton,
            pos=target,
        )

        self.assertEqual(
            self.canvas.view_offset,
            initial_offset + QPointF(target - start),
        )

if __name__ == "__main__":
    unittest.main()
