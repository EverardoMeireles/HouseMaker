# ### Environment setup ###
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.blueprint_canvas import BlueprintCanvas
from housemaker.models import VERTEX_HIT_RADIUS_SCREEN, VertexData

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_canvas(vertex_data: VertexData | None = None) -> BlueprintCanvas:
    """Return one visible Canvas with a one-to-one plan image."""

    canvas = BlueprintCanvas()
    canvas.resize(640, 520)
    canvas.blueprint_image = QImage(100, 100, QImage.Format.Format_RGB32)
    canvas.blueprint_image.fill(Qt.GlobalColor.white)
    canvas.vertex_data = vertex_data or VertexData()
    canvas.show()
    _qt_application.processEvents()
    return canvas


def _render_canvas(canvas: BlueprintCanvas) -> QImage:
    """Render one Canvas for guide-color assertions."""

    image = QImage(canvas.size(), QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    canvas.render(image)
    return image


# ### Canvas guide tests ###
class OuterWallMarkingCanvasTests(unittest.TestCase):
    def setUp(self) -> None:
        self.canvases: list[BlueprintCanvas] = []

    def tearDown(self) -> None:
        for canvas in self.canvases:
            canvas.close()
            canvas.deleteLater()
        _qt_application.processEvents()

    def _track(self, canvas: BlueprintCanvas) -> BlueprintCanvas:
        self.canvases.append(canvas)
        return canvas

    def test_vertex_selection_emits_primary_and_group_changes(self) -> None:
        canvas = self._track(_build_canvas())
        primary_changes: list[object] = []
        group_changes: list[object] = []
        canvas.selected_vertex_changed.connect(primary_changes.append)
        canvas.selected_vertices_changed.connect(group_changes.append)

        canvas.selected_vertex_id = 7
        canvas.selected_vertex_id = 7
        canvas.set_selected_vertex_ids((7, 9))
        canvas.set_selected_vertex_ids((9,))
        canvas.selected_vertex_id = None

        self.assertEqual(primary_changes, [7, 9, None])
        self.assertEqual(group_changes, [(7,), (7, 9), (9,), ()])

    def test_local_mark_api_normalizes_ids_and_repaints(self) -> None:
        canvas = self._track(_build_canvas())

        self.assertTrue(canvas.set_outer_wall_mark_vertex_ids((4, 9, 4)))
        self.assertEqual(
            canvas.get_outer_wall_mark_vertex_ids(),
            frozenset((4, 9)),
        )
        self.assertFalse(canvas.set_outer_wall_mark_vertex_ids((9, 4)))
        self.assertTrue(canvas.set_outer_wall_mark_vertex_ids(()))
        self.assertEqual(canvas.get_outer_wall_mark_vertex_ids(), frozenset())

    def test_marked_local_vertex_remains_selectable_local_geometry(self) -> None:
        vertex_data = VertexData()
        vertex = vertex_data.add_vertex(70.0, 60.0)
        canvas = self._track(_build_canvas(vertex_data))
        canvas.set_outer_wall_mark_vertex_ids((vertex.id,))

        QTest.mouseClick(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=canvas._image_to_widget(vertex.x, vertex.y).toPoint(),
        )

        self.assertEqual(len(vertex_data.vertices), 1)
        self.assertEqual(canvas.selected_vertex_ids, (vertex.id,))
        self.assertEqual(canvas.active_vertex_id, vertex.id)

    def test_local_mark_is_green_while_cross_level_guides_are_red_and_yellow(
        self,
    ) -> None:
        vertex_data = VertexData()
        local_vertex = vertex_data.add_vertex(20.0, 20.0)
        canvas = self._track(_build_canvas(vertex_data))
        canvas.set_outer_wall_mark_vertex_ids((local_vertex.id,))
        canvas.set_outer_wall_reference_overlay(
            ((30.0, 70.0), (70.0, 70.0)),
            ((((30.0, 70.0), (70.0, 70.0))),),
        )

        image = _render_canvas(canvas)
        local_color = image.pixelColor(
            canvas._image_to_widget(20.0, 20.0).toPoint()
        )
        reference_color = image.pixelColor(
            canvas._image_to_widget(30.0, 70.0).toPoint()
        )
        edge_color = image.pixelColor(
            canvas._image_to_widget(50.0, 70.0).toPoint()
        )

        self.assertLess(local_color.red(), 120)
        self.assertGreater(local_color.green(), 150)
        self.assertLess(local_color.blue(), 180)
        self.assertGreater(reference_color.red(), 150)
        self.assertLess(reference_color.green(), 140)
        self.assertLess(reference_color.blue(), 140)
        self.assertGreater(edge_color.red(), 150)
        self.assertGreater(edge_color.green(), 130)
        self.assertLess(edge_color.blue(), 140)

    def test_reference_point_participates_in_new_wall_vertex_snapping(self) -> None:
        canvas = self._track(_build_canvas())
        guide_point = (60.0, 45.0)
        canvas.set_outer_wall_reference_overlay((guide_point,), ())

        preview = canvas._build_connection_preview(
            QPointF(61.0, 46.0),
            Qt.KeyboardModifier.NoModifier,
        )

        self.assertEqual(preview.point, guide_point)
        self.assertEqual(preview.guides, [])
        self.assertEqual(len(canvas.vertex_data.vertices), 0)

    def test_reference_point_provides_green_orthogonal_snap_guide(self) -> None:
        canvas = self._track(_build_canvas())
        guide_point = (60.0, 25.0)
        canvas.set_outer_wall_reference_overlay((guide_point,), ())

        preview = canvas._build_connection_preview(
            QPointF(61.0, 80.0),
            Qt.KeyboardModifier.NoModifier,
        )

        self.assertEqual(preview.point, (60.0, 80.0))
        self.assertEqual(len(preview.guides), 1)
        self.assertEqual(preview.guides[0].axis, "x")
        self.assertEqual(preview.guides[0].source_point, guide_point)

        canvas.preview_point = preview.point
        canvas.preview_guides = preview.guides
        image = _render_canvas(canvas)
        widget_x = round(canvas._image_to_widget(guide_point[0], 0.0).x())
        start_y = round(canvas._image_to_widget(0.0, guide_point[1]).y())
        end_y = round(canvas._image_to_widget(0.0, preview.point[1]).y())
        green_guide_pixels = [
            image.pixelColor(widget_x, widget_y)
            for widget_y in range(min(start_y, end_y) + 4, max(start_y, end_y) - 3)
            if 0 <= widget_x < image.width() and 0 <= widget_y < image.height()
        ]
        self.assertTrue(
            any(
                color.green() > color.red() + 30
                and color.green() > color.blue() + 10
                for color in green_guide_pixels
            )
        )
        self.assertTrue(
            any(
                color.red() > 220
                and color.green() > 220
                and color.blue() > 220
                for color in green_guide_pixels
            )
        )

        click_position = canvas._image_to_widget(61.0, 80.0).toPoint()
        clicked_image_point = canvas._widget_to_image(QPointF(click_position))
        self.assertIsNotNone(clicked_image_point)
        committed_preview = canvas._build_connection_preview(
            clicked_image_point,
            Qt.KeyboardModifier.NoModifier,
        )
        QTest.mouseClick(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=click_position,
        )

        created_vertex = canvas.vertex_data.vertices[-1]
        self.assertEqual(
            (created_vertex.x, created_vertex.y),
            committed_preview.point,
        )

    def test_first_wall_vertex_hover_shows_reference_axis_guide(self) -> None:
        canvas = self._track(_build_canvas())
        guide_point = (60.0, 25.0)
        canvas.set_outer_wall_reference_overlay((guide_point,), ())

        QTest.mouseMove(
            canvas,
            canvas._image_to_widget(61.0, 80.0).toPoint(),
        )

        self.assertIsNone(canvas.active_vertex_id)
        self.assertIsNotNone(canvas.preview_point)
        self.assertEqual(canvas.preview_point[0], guide_point[0])
        self.assertEqual(len(canvas.preview_guides), 1)
        self.assertEqual(canvas.preview_guides[0].axis, "x")
        self.assertEqual(canvas.preview_guides[0].source_point, guide_point)

    def test_green_reference_guide_paints_above_yellow_reference_edge(
        self,
    ) -> None:
        canvas = self._track(_build_canvas())
        guide_point = (60.0, 20.0)
        edge_end = (60.0, 85.0)
        canvas.set_outer_wall_reference_overlay(
            (guide_point,),
            ((guide_point, edge_end),),
        )
        preview = canvas._build_connection_preview(
            QPointF(61.0, 65.0),
            Qt.KeyboardModifier.NoModifier,
        )
        canvas.preview_point = preview.point
        canvas.preview_guides = preview.guides

        image = _render_canvas(canvas)
        widget_x = round(canvas._image_to_widget(guide_point[0], 0.0).x())
        start_y = round(canvas._image_to_widget(0.0, guide_point[1]).y())
        end_y = round(canvas._image_to_widget(0.0, preview.point[1]).y())
        guide_colors = [
            image.pixelColor(widget_x, widget_y)
            for widget_y in range(start_y + 4, end_y - 3)
        ]

        self.assertTrue(
            any(
                color.green() > color.red() + 30
                and color.green() > color.blue() + 10
                for color in guide_colors
            )
        )

    def test_dragged_wall_vertex_uses_reference_point_axis_snap(self) -> None:
        vertex_data = VertexData()
        dragged = vertex_data.add_vertex(15.0, 15.0)
        canvas = self._track(_build_canvas(vertex_data))
        guide_point = (70.0, 40.0)
        canvas.set_outer_wall_reference_overlay((guide_point,), ())

        preview = canvas._build_drag_preview(
            QPointF(25.0, 41.0),
            dragged_vertex_id=dragged.id,
        )

        self.assertEqual(preview.point, (25.0, 40.0))
        self.assertEqual(len(preview.guides), 1)
        self.assertEqual(preview.guides[0].axis, "y")
        self.assertEqual(preview.guides[0].source_point, guide_point)

    def test_reference_guide_wins_equal_axis_tie_with_local_vertex(self) -> None:
        vertex_data = VertexData()
        dragged = vertex_data.add_vertex(15.0, 15.0)
        vertex_data.add_vertex(70.0, 80.0)
        canvas = self._track(_build_canvas(vertex_data))
        guide_point = (70.0, 40.0)
        canvas.set_outer_wall_reference_overlay((guide_point,), ())

        preview = canvas._build_drag_preview(
            QPointF(71.0, 60.0),
            dragged_vertex_id=dragged.id,
        )

        self.assertEqual(preview.point, (70.0, 60.0))
        self.assertEqual(len(preview.guides), 1)
        self.assertIsNone(preview.guides[0].source_vertex_id)
        self.assertEqual(preview.guides[0].source_point, guide_point)

    def test_dragged_wall_vertex_exact_snap_precedes_reference_axis_snap(self) -> None:
        vertex_data = VertexData()
        dragged = vertex_data.add_vertex(15.0, 15.0)
        canvas = self._track(_build_canvas(vertex_data))
        guide_point = (70.0, 40.0)
        canvas.set_outer_wall_reference_overlay((guide_point,), ())
        tolerance = canvas._screen_distance_to_image(VERTEX_HIT_RADIUS_SCREEN)

        preview = canvas._build_drag_preview(
            QPointF(guide_point[0] + tolerance * 0.9, guide_point[1]),
            dragged_vertex_id=dragged.id,
        )

        self.assertEqual(preview.point, guide_point)
        self.assertEqual(preview.guides, [])

    def test_reference_point_has_priority_over_a_nearby_editable_edge(self) -> None:
        vertex_data = VertexData()
        first = vertex_data.add_vertex(10.0, 50.0)
        second = vertex_data.add_vertex(90.0, 50.0)
        vertex_data.add_edge(first.id, second.id)
        canvas = self._track(_build_canvas(vertex_data))
        guide_point = (50.0, 45.0)
        canvas.set_outer_wall_reference_overlay((guide_point,), ())

        QTest.mouseClick(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=canvas._image_to_widget(51.0, 46.0).toPoint(),
        )

        added_vertex = canvas.vertex_data.vertices[-1]
        self.assertEqual((added_vertex.x, added_vertex.y), guide_point)
        self.assertEqual(
            canvas.vertex_data.edges,
            [vertex_data.edges[0]],
        )

    def test_reference_axis_snap_has_priority_over_a_nearby_editable_edge(
        self,
    ) -> None:
        vertex_data = VertexData()
        first = vertex_data.add_vertex(40.0, 50.0)
        second = vertex_data.add_vertex(60.0, 60.0)
        vertex_data.add_edge(first.id, second.id)
        canvas = self._track(_build_canvas(vertex_data))
        guide_point = (50.0, 10.0)
        canvas.set_outer_wall_reference_overlay((guide_point,), ())
        click_position = canvas._image_to_widget(51.0, 56.0).toPoint()
        clicked_image_point = canvas._widget_to_image(QPointF(click_position))
        self.assertIsNotNone(clicked_image_point)
        preview = canvas._build_connection_preview(
            clicked_image_point,
            Qt.KeyboardModifier.NoModifier,
        )

        QTest.mouseClick(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=click_position,
        )

        added_vertex = canvas.vertex_data.vertices[-1]
        self.assertEqual((added_vertex.x, added_vertex.y), preview.point)
        self.assertEqual(preview.point[0], guide_point[0])
        self.assertEqual(
            canvas.vertex_data.edges,
            [vertex_data.edges[0]],
        )

    def test_reference_drag_snap_is_preserved_near_an_editable_edge(self) -> None:
        vertex_data = VertexData()
        dragged = vertex_data.add_vertex(20.0, 20.0)
        first = vertex_data.add_vertex(10.0, 50.0)
        second = vertex_data.add_vertex(90.0, 50.0)
        vertex_data.add_edge(first.id, second.id)
        canvas = self._track(_build_canvas(vertex_data))
        guide_point = (50.0, 49.0)
        canvas.set_outer_wall_reference_overlay((guide_point,), ())
        start = canvas._image_to_widget(dragged.x, dragged.y).toPoint()
        end = canvas._image_to_widget(*guide_point).toPoint()

        QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(canvas, end, delay=10)
        QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=end)

        moved_vertex = vertex_data.get_vertex(dragged.id)
        self.assertIsNotNone(moved_vertex)
        self.assertEqual(
            (moved_vertex.x, moved_vertex.y),
            guide_point,
        )
        self.assertEqual(vertex_data.edges, [vertex_data.edges[0]])

    def test_reference_overlay_does_not_create_editable_local_geometry(self) -> None:
        canvas = self._track(_build_canvas())
        points = ((25.0, 25.0), (75.0, 25.0))
        canvas.set_outer_wall_reference_overlay(
            points,
            ((points[0], points[1]),),
        )

        self.assertEqual(canvas.vertex_data.vertices, [])
        self.assertEqual(canvas.vertex_data.edges, [])
        self.assertIsNone(
            canvas._find_vertex_at(canvas._image_to_widget(*points[0]))
        )

    def test_marking_mode_emits_toggle_and_cancel_without_editing_geometry(
        self,
    ) -> None:
        vertex_data = VertexData()
        vertex = vertex_data.add_vertex(50.0, 50.0)
        canvas = self._track(_build_canvas(vertex_data))
        toggled_ids: list[int] = []
        cancel_count: list[bool] = []
        canvas.outer_wall_mark_vertex_toggled.connect(toggled_ids.append)
        canvas.outer_wall_mark_cancel_requested.connect(
            lambda: cancel_count.append(True)
        )
        canvas.set_outer_wall_marking_active(True)

        QTest.mouseClick(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=canvas._image_to_widget(vertex.x, vertex.y).toPoint(),
        )
        QTest.mouseClick(
            canvas,
            Qt.MouseButton.RightButton,
            pos=canvas.rect().center(),
        )

        self.assertEqual(toggled_ids, [vertex.id])
        self.assertEqual(cancel_count, [True])
        self.assertFalse(canvas.is_outer_wall_marking_active())
        self.assertEqual(len(canvas.vertex_data.vertices), 1)
        self.assertEqual(canvas.vertex_data.edges, [])

    def test_delete_is_ignored_while_auto_marking_is_active(self) -> None:
        vertex_data = VertexData()
        first = vertex_data.add_vertex(30.0, 50.0)
        second = vertex_data.add_vertex(70.0, 50.0)
        vertex_data.add_edge(first.id, second.id)
        canvas = self._track(_build_canvas(vertex_data))
        canvas.set_selected_vertex_ids((first.id, second.id))
        canvas.set_outer_wall_marking_active(True)

        QTest.keyClick(canvas, Qt.Key.Key_Delete)

        self.assertEqual(
            {vertex.id for vertex in canvas.vertex_data.vertices},
            {first.id, second.id},
        )
        self.assertEqual(len(canvas.vertex_data.edges), 1)

    def test_loading_a_different_level_clears_stale_marks_and_guides(self) -> None:
        canvas = self._track(_build_canvas())
        canvas.set_outer_wall_mark_vertex_ids((2,))
        canvas.set_outer_wall_reference_overlay(
            ((20.0, 20.0),),
            (),
        )

        canvas.set_level_data(
            vertex_data=VertexData(),
            rooms=[],
            image_path=None,
        )

        self.assertEqual(canvas.get_outer_wall_mark_vertex_ids(), frozenset())
        overlay = canvas.get_outer_wall_reference_overlay()
        self.assertEqual(overlay.points, ())
        self.assertEqual(overlay.edges, ())

    def test_pruning_a_deleted_vertex_clears_its_drawing_state(self) -> None:
        vertex_data = VertexData()
        vertex = vertex_data.add_vertex(30.0, 40.0)
        canvas = self._track(_build_canvas(vertex_data))
        canvas.set_outer_wall_mark_vertex_ids((vertex.id,))
        canvas.selected_vertex_id = vertex.id
        canvas.active_vertex_id = vertex.id
        canvas.pressed_vertex_id = vertex.id
        canvas.drag_vertex_id = vertex.id
        canvas.preview_point = (vertex.x, vertex.y)
        vertex_data.delete_vertex(vertex.id)

        self.assertTrue(canvas.prune_missing_vertex_references())

        self.assertEqual(canvas.selected_vertex_ids, ())
        self.assertEqual(canvas.get_outer_wall_mark_vertex_ids(), frozenset())
        self.assertIsNone(canvas.active_vertex_id)
        self.assertIsNone(canvas.pressed_vertex_id)
        self.assertIsNone(canvas.drag_vertex_id)
        self.assertIsNone(canvas.preview_point)


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
