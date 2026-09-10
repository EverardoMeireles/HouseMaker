# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import unittest

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QImage, QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.blueprint_canvas import BlueprintCanvas
from housemaker.models import VertexData


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Event helpers ###
def _send_drag_move(canvas: BlueprintCanvas, position: QPoint) -> None:
    event = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(position),
        QPointF(canvas.mapToGlobal(position)),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,
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

    def test_deleting_existing_vertex_is_only_a_general_change(self) -> None:
        vertex = self.canvas.vertex_data.add_vertex(10.0, 20.0)
        self.canvas.selected_vertex_id = vertex.id
        events = self._record_geometry_events()

        self.canvas._delete_selected_vertex()

        self.assertEqual(events, ["geometry_changed"])
        self.assertIsNone(self.canvas.vertex_data.get_vertex(vertex.id))

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


if __name__ == "__main__":
    unittest.main()
