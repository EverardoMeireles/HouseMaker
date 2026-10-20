# ### Environment setup ###
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QImage, QKeyEvent, QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.blueprint_canvas import BlueprintCanvas
from housemaker.models import OpenSpaceData

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)
_CTRL = Qt.KeyboardModifier.ControlModifier
_NO_MODIFIERS = Qt.KeyboardModifier.NoModifier


# ### Event helpers ###
def _send_drag_move(
    canvas: BlueprintCanvas,
    point: QPoint,
    modifiers: Qt.KeyboardModifier = _NO_MODIFIERS,
) -> None:
    event = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(point),
        QPointF(canvas.mapToGlobal(point)),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,
        modifiers,
    )
    QApplication.sendEvent(canvas, event)


def _send_control_key(canvas: BlueprintCanvas, pressed: bool) -> None:
    QApplication.sendEvent(
        canvas,
        QKeyEvent(
            QEvent.Type.KeyPress if pressed else QEvent.Type.KeyRelease,
            Qt.Key.Key_Control,
            _CTRL if pressed else _NO_MODIFIERS,
        ),
    )


# ### Connected-only snap preview tests ###
class BlueprintVertexDragCtrlSnapTests(unittest.TestCase):
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

    def _vertex_position(self, vertex_id: int) -> tuple[float, float]:
        vertex = self.canvas.vertex_data.get_vertex(vertex_id)
        self.assertIsNotNone(vertex)
        assert vertex is not None
        return vertex.x, vertex.y

    def test_control_keeps_one_connected_vertex_axis_snap(self) -> None:
        graph = self.canvas.vertex_data
        dragged = graph.add_vertex(12.0, 12.0)
        connected = graph.add_vertex(30.0, 80.0)
        unrelated = graph.add_vertex(61.0, 50.0)
        graph.add_edge(dragged.id, connected.id)

        preview = self.canvas._build_drag_preview(
            QPointF(30.7, 50.8), dragged.id, _CTRL
        )

        self.assertEqual(preview.point, (30.0, 50.8))
        self.assertEqual(
            [(guide.axis, guide.source_vertex_id) for guide in preview.guides],
            [("x", connected.id)],
        )
        self.assertNotIn(
            unrelated.id, [guide.source_vertex_id for guide in preview.guides]
        )

    def test_control_keeps_both_connected_neighbors_in_either_edge_direction(
        self,
    ) -> None:
        graph = self.canvas.vertex_data
        dragged = graph.add_vertex(12.0, 12.0)
        first = graph.add_vertex(40.0, 90.0)
        second = graph.add_vertex(90.0, 60.0)
        graph.add_vertex(40.8, 60.8)
        graph.add_edge(dragged.id, first.id)
        graph.add_edge(second.id, dragged.id)

        preview = self.canvas._build_drag_preview(
            QPointF(40.8, 60.8), dragged.id, _CTRL
        )

        self.assertEqual(preview.point, (40.0, 60.0))
        self.assertEqual(
            [(guide.axis, guide.source_vertex_id) for guide in preview.guides],
            [("x", first.id), ("y", second.id)],
        )

    def test_control_does_not_snap_to_indirectly_connected_vertices(self) -> None:
        graph = self.canvas.vertex_data
        dragged = graph.add_vertex(12.0, 12.0)
        neighbor = graph.add_vertex(25.0, 90.0)
        indirect = graph.add_vertex(60.0, 50.0)
        graph.add_edge(dragged.id, neighbor.id)
        graph.add_edge(neighbor.id, indirect.id)

        preview = self.canvas._build_drag_preview(
            QPointF(60.5, 50.5), dragged.id, _CTRL
        )

        self.assertEqual(preview.point, (60.5, 50.5))
        self.assertEqual(preview.guides, [])

    def test_control_with_no_connections_is_free_of_other_snap_sources(self) -> None:
        graph = self.canvas.vertex_data
        dragged = graph.add_vertex(12.0, 12.0)
        graph.add_vertex(60.0, 40.0)
        self.canvas.set_outer_wall_reference_overlay(((60.0, 40.0),), ())
        self.canvas.open_spaces.append(
            OpenSpaceData("opening-a", 60.0, 40.0, 85.0, 80.0)
        )
        raw = QPointF(60.4, 40.4)

        self.assertEqual(
            self.canvas._build_drag_preview(raw, dragged.id).point,
            (60.0, 40.0),
        )
        with patch.object(
            self.canvas,
            "_find_center_snap_candidate",
            side_effect=AssertionError("Ctrl must disable center snapping"),
        ):
            preview = self.canvas._build_drag_preview(raw, dragged.id, _CTRL)

        self.assertEqual(preview.point, (60.4, 40.4))
        self.assertEqual(preview.guides, [])

    def test_control_ignores_open_space_axis_snapping(self) -> None:
        graph = self.canvas.vertex_data
        dragged = graph.add_vertex(12.0, 12.0)
        self.canvas.open_spaces.append(
            OpenSpaceData("opening-a", 60.0, 40.0, 85.0, 80.0)
        )

        preview = self.canvas._build_drag_preview(
            QPointF(60.5, 40.5), dragged.id, _CTRL
        )

        self.assertEqual(preview.point, (60.5, 40.5))
        self.assertEqual(preview.guides, [])

    # ### Live modifier and release regressions ###
    def test_stationary_control_toggle_recalculates_from_raw_pointer(self) -> None:
        graph = self.canvas.vertex_data
        dragged = graph.add_vertex(15.0, 15.0)
        neighbor = graph.add_vertex(25.0, 90.0)
        graph.add_vertex(60.0, 50.0)
        graph.add_edge(dragged.id, neighbor.id)
        start = self.canvas._image_to_widget(dragged.x, dragged.y).toPoint()
        target = self.canvas._image_to_widget(60.8, 50.8).toPoint()
        raw = self.canvas._widget_to_image_clamped(QPointF(target))
        raw_point = (raw.x(), raw.y())
        events: list[str] = []
        self.canvas.geometry_changed.connect(lambda: events.append("geometry"))
        self.canvas.wall_vertex_interaction_changed.connect(
            lambda active: events.append(f"interaction:{active}")
        )

        QTest.mousePress(self.canvas, Qt.MouseButton.LeftButton, pos=start)
        _send_drag_move(self.canvas, target)
        self.assertEqual(self._vertex_position(dragged.id), (60.0, 50.0))

        _send_control_key(self.canvas, True)
        self.assertEqual(self._vertex_position(dragged.id), raw_point)
        self.assertEqual(self.canvas.preview_guides, [])
        _send_control_key(self.canvas, False)
        self.assertEqual(self._vertex_position(dragged.id), (60.0, 50.0))
        self.assertEqual(len(self.canvas.undo_stack), 1)
        self.assertEqual(events, ["interaction:True"])

        QTest.mouseRelease(self.canvas, Qt.MouseButton.LeftButton, pos=target)

        self.assertEqual(len(self.canvas.undo_stack), 1)
        self.assertEqual(events, ["interaction:True", "geometry", "interaction:False"])

    def test_control_release_does_not_snap_or_join_unrelated_edge(self) -> None:
        graph = self.canvas.vertex_data
        dragged = graph.add_vertex(15.0, 15.0)
        neighbor = graph.add_vertex(20.0, 90.0)
        edge_start = graph.add_vertex(40.0, 50.0)
        edge_end = graph.add_vertex(80.0, 50.0)
        graph.add_edge(dragged.id, neighbor.id)
        graph.add_edge(edge_start.id, edge_end.id)
        original_edges = tuple(graph.edges)
        start = self.canvas._image_to_widget(dragged.x, dragged.y).toPoint()
        target = self.canvas._image_to_widget(60.0, 50.7).toPoint()
        raw = self.canvas._widget_to_image_clamped(QPointF(target))

        QTest.mousePress(self.canvas, Qt.MouseButton.LeftButton, pos=start)
        _send_drag_move(self.canvas, target)
        QTest.mouseRelease(self.canvas, Qt.MouseButton.LeftButton, _CTRL, target)

        self.assertEqual(self._vertex_position(dragged.id), (raw.x(), raw.y()))
        self.assertEqual(tuple(graph.edges), original_edges)
        self.assertFalse(graph.has_edge(dragged.id, edge_start.id))
        self.assertFalse(graph.has_edge(dragged.id, edge_end.id))

    def test_normal_release_still_joins_unrelated_edge(self) -> None:
        graph = self.canvas.vertex_data
        dragged = graph.add_vertex(15.0, 15.0)
        edge_start = graph.add_vertex(40.0, 50.0)
        edge_end = graph.add_vertex(80.0, 50.0)
        graph.add_edge(edge_start.id, edge_end.id)
        start = self.canvas._image_to_widget(dragged.x, dragged.y).toPoint()
        target = self.canvas._image_to_widget(60.0, 50.7).toPoint()

        QTest.mousePress(self.canvas, Qt.MouseButton.LeftButton, pos=start)
        _send_drag_move(self.canvas, target, _CTRL)
        QTest.mouseRelease(self.canvas, Qt.MouseButton.LeftButton, pos=target)

        self.assertEqual(self._vertex_position(dragged.id)[1], 50.0)
        self.assertTrue(graph.has_edge(dragged.id, edge_start.id))
        self.assertTrue(graph.has_edge(dragged.id, edge_end.id))
        self.assertFalse(graph.has_edge(edge_start.id, edge_end.id))
