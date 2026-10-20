# ### Environment setup ###
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication

from housemaker.blueprint_canvas import (
    DEAD_END_VERTEX_HIGHLIGHT_COLOR,
    NON_RIGHT_ANGLE_HIGHLIGHT_COLOR,
    BlueprintCanvas,
)
from housemaker.canvas_wall_angles import find_non_right_angle_wall_parts
from housemaker.models import VertexData

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_graph(*, right_angle: bool = False) -> VertexData:
    graph = VertexData()
    first = graph.add_vertex(20.0, 20.0)
    corner = graph.add_vertex(50.0, 20.0)
    last = graph.add_vertex(50.0 if right_angle else 70.0, 50.0)
    graph.add_edge(first.id, corner.id)
    graph.add_edge(corner.id, last.id)
    return graph


def _render(canvas: BlueprintCanvas) -> QImage:
    image = QImage(canvas.size(), QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    canvas.render(image)
    return image


# ### Canvas highlight tests ###
class CanvasNonRightAngleHighlightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas = BlueprintCanvas()
        self.canvas.resize(600, 500)
        self.canvas.vertex_data = _build_graph()
        self.canvas.blueprint_image = QImage(100, 100, QImage.Format.Format_RGB32)
        self.canvas.blueprint_image.fill(Qt.GlobalColor.white)
        self.canvas.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.canvas.close()
        self.canvas.deleteLater()
        _qt_application.processEvents()

    def test_disabled_highlight_does_no_analysis_or_blinking(self) -> None:
        with patch(
            "housemaker.blueprint_canvas.find_non_right_angle_wall_parts"
        ) as analyze:
            _render(self.canvas)
        self.assertFalse(self.canvas.highlight_non_right_angles_enabled)
        self.assertFalse(self.canvas._non_right_angle_highlight_timer.isActive())
        analyze.assert_not_called()

    def test_purple_edges_and_corner_blink_without_changing_geometry_or_selection(
        self,
    ) -> None:
        canvas = self.canvas
        original = canvas.vertex_data.clone()
        original_revision = canvas.vertex_data.revision
        canvas.set_selected_vertex_ids((2,))
        edge_point = canvas._image_to_widget(35.0, 20.0).toPoint()
        corner_point = canvas._image_to_widget(50.0, 20.0).toPoint()
        ordinary_edge_color = _render(canvas).pixelColor(edge_point)
        canvas.set_highlight_non_right_angles(True)
        shown = _render(canvas)
        for point in (edge_point, corner_point):
            self.assertEqual(shown.pixelColor(point), NON_RIGHT_ANGLE_HIGHLIGHT_COLOR)
        self.assertTrue(canvas._non_right_angle_highlight_timer.isActive())

        canvas._advance_non_right_angle_highlight()
        hidden = _render(canvas)
        self.assertEqual(hidden.pixelColor(edge_point), ordinary_edge_color)
        self.assertNotEqual(
            hidden.pixelColor(corner_point), NON_RIGHT_ANGLE_HIGHLIGHT_COLOR
        )
        canvas._advance_non_right_angle_highlight()
        self.assertEqual(
            _render(canvas).pixelColor(edge_point), NON_RIGHT_ANGLE_HIGHLIGHT_COLOR
        )
        self.assertEqual(canvas.vertex_data, original)
        self.assertEqual(canvas.vertex_data.revision, original_revision)
        self.assertEqual(canvas.selected_vertex_ids, (2,))
        self.assertEqual(canvas.undo_stack, [])

    def test_angle_analysis_is_cached_and_refreshes_during_vertex_moves(self) -> None:
        canvas = self.canvas
        with patch(
            "housemaker.blueprint_canvas.find_non_right_angle_wall_parts",
            wraps=find_non_right_angle_wall_parts,
        ) as analyze:
            canvas.set_highlight_non_right_angles(True)
            _render(canvas)
            canvas._advance_non_right_angle_highlight()
            _render(canvas)
            self.assertEqual(analyze.call_count, 1)
            canvas.vertex_data.move_vertex(3, 50.0, 50.0)
            _render(canvas)
            self.assertEqual(analyze.call_count, 2)
            self.assertFalse(canvas._get_wall_angle_highlights().vertex_ids)
            self.assertEqual(
                canvas._get_wall_angle_highlights().dead_end_vertex_ids, {1, 3}
            )
            self.assertTrue(canvas._non_right_angle_highlight_timer.isActive())
            canvas.vertex_data.move_vertex(3, 70.0, 50.0)
            _render(canvas)
            self.assertEqual(analyze.call_count, 3)
            self.assertEqual(canvas._get_wall_angle_highlights().vertex_ids, {2})
            self.assertTrue(canvas._non_right_angle_highlight_timer.isActive())

    def test_switching_graphs_with_same_revision_does_not_reuse_old_result(
        self,
    ) -> None:
        canvas = self.canvas
        canvas.set_highlight_non_right_angles(True)
        _render(canvas)
        replacement = _build_graph(right_angle=True)
        self.assertEqual(canvas.vertex_data.revision, replacement.revision)
        canvas.vertex_data = replacement
        _render(canvas)
        self.assertFalse(canvas._get_wall_angle_highlights().edge_keys)
        self.assertEqual(
            canvas._get_wall_angle_highlights().dead_end_vertex_ids, {1, 3}
        )
        self.assertTrue(canvas._non_right_angle_highlight_timer.isActive())
        self.assertTrue(canvas.highlight_non_right_angles_enabled)

    def test_dead_end_vertices_blink_brown_without_highlighting_the_edge(self) -> None:
        canvas = self.canvas
        graph = VertexData()
        first = graph.add_vertex(20.0, 30.0)
        second = graph.add_vertex(80.0, 30.0)
        isolated = graph.add_vertex(50.0, 70.0)
        graph.add_edge(first.id, second.id)
        canvas.vertex_data = graph
        canvas.set_highlight_non_right_angles(True)
        shown = _render(canvas)
        for vertex in (first, second):
            point = canvas._image_to_widget(vertex.x, vertex.y).toPoint()
            self.assertEqual(shown.pixelColor(point), DEAD_END_VERTEX_HIGHLIGHT_COLOR)
        self.assertNotEqual(
            shown.pixelColor(canvas._image_to_widget(isolated.x, isolated.y).toPoint()),
            DEAD_END_VERTEX_HIGHLIGHT_COLOR,
        )
        highlights = canvas._get_wall_angle_highlights()
        self.assertEqual(highlights.dead_end_vertex_ids, {first.id, second.id})
        self.assertFalse(highlights.vertex_ids)
        self.assertFalse(highlights.edge_keys)
        self.assertTrue(canvas._non_right_angle_highlight_timer.isActive())
        canvas._advance_non_right_angle_highlight()
        hidden = _render(canvas)
        self.assertNotEqual(
            hidden.pixelColor(canvas._image_to_widget(first.x, first.y).toPoint()),
            DEAD_END_VERTEX_HIGHLIGHT_COLOR,
        )

    def test_closed_right_angle_walls_have_no_blink_timer(self) -> None:
        canvas = self.canvas
        graph = _build_graph(right_angle=True)
        fourth = graph.add_vertex(20.0, 50.0)
        graph.add_edge(3, fourth.id)
        graph.add_edge(fourth.id, 1)
        canvas.vertex_data = graph
        canvas.set_highlight_non_right_angles(True)
        _render(canvas)
        highlights = canvas._get_wall_angle_highlights()
        self.assertFalse(highlights.edge_keys)
        self.assertFalse(highlights.vertex_ids)
        self.assertFalse(highlights.dead_end_vertex_ids)
        self.assertFalse(canvas._non_right_angle_highlight_timer.isActive())

    def test_generated_wall_preview_is_highlighted_without_editing_level(self) -> None:
        canvas = self.canvas
        canvas.vertex_data = _build_graph(right_angle=True)
        baseline = canvas.vertex_data.clone()
        canvas.set_highlight_non_right_angles(True)
        _render(canvas)
        self.assertFalse(canvas._get_wall_angle_highlights().edge_keys)
        candidate = _build_graph()
        canvas.set_generated_wall_preview(candidate)
        self.assertEqual(
            _render(canvas).pixelColor(canvas._image_to_widget(35.0, 20.0).toPoint()),
            NON_RIGHT_ANGLE_HIGHLIGHT_COLOR,
        )
        self.assertEqual(canvas._get_wall_angle_highlights().vertex_ids, {2})
        self.assertEqual(canvas.vertex_data, baseline)
        canvas.clear_generated_wall_preview()
        _render(canvas)
        self.assertFalse(canvas._get_wall_angle_highlights().edge_keys)

    def test_timer_blinks_and_stops_when_hidden_or_disabled(self) -> None:
        canvas = self.canvas
        canvas.set_highlight_non_right_angles(True)
        _render(canvas)
        timer = canvas._non_right_angle_highlight_timer
        timer.setInterval(20)
        spy = QSignalSpy(timer.timeout)
        QTest.qWait(60)
        self.assertGreaterEqual(spy.count(), 1)
        canvas.hide()
        self.assertFalse(timer.isActive())
        canvas.show()
        _qt_application.processEvents()
        self.assertTrue(timer.isActive())
        canvas.set_highlight_non_right_angles(False)
        self.assertFalse(timer.isActive())
        self.assertFalse(canvas._wall_angle_highlight_vertices)


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
