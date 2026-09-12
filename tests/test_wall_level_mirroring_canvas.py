# ### Environment setup ###
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.blueprint_canvas import (
    SELECTED_VERTEX_FILL_COLOR,
    WALL_MIRROR_VERTEX_FILL_COLOR,
    BlueprintCanvas,
)
from housemaker.models import VertexData

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_canvas(vertex_data: VertexData | None = None) -> BlueprintCanvas:
    canvas = BlueprintCanvas()
    canvas.resize(640, 520)
    canvas.blueprint_image = QImage(100, 100, QImage.Format.Format_RGB32)
    canvas.blueprint_image.fill(Qt.GlobalColor.white)
    canvas.vertex_data = vertex_data or VertexData()
    canvas.show()
    _qt_application.processEvents()
    return canvas


def _render_canvas(canvas: BlueprintCanvas) -> QImage:
    image = QImage(canvas.size(), QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    canvas.render(image)
    return image


# ### Tests ###
class WallLevelMirroringCanvasTests(unittest.TestCase):
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

    def test_vertex_selection_properties_emit_primary_and_group_changes(
        self,
    ) -> None:
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
        self.assertEqual(
            group_changes,
            [(7,), (7, 9), (9,), ()],
        )
        self.assertEqual(canvas.selected_vertex_ids, ())

    def test_wall_mirror_marker_api_normalizes_and_repaints(self) -> None:
        canvas = self._track(_build_canvas())

        self.assertTrue(canvas.set_wall_mirror_vertex_ids(iter((4, 9, 4))))
        self.assertEqual(
            canvas.get_wall_mirror_vertex_ids(),
            frozenset((4, 9)),
        )
        self.assertFalse(canvas.set_wall_mirror_vertex_ids((9, 4)))
        self.assertTrue(canvas.set_wall_mirror_vertex_ids(()))
        self.assertEqual(canvas.get_wall_mirror_vertex_ids(), frozenset())

    def test_flagged_local_vertex_stays_green_when_selected(self) -> None:
        vertex_data = VertexData()
        vertex = vertex_data.add_vertex(30.0, 40.0)
        canvas = self._track(_build_canvas(vertex_data))
        canvas.set_wall_mirror_vertex_ids((vertex.id,))
        canvas.active_vertex_id = vertex.id
        canvas.selected_vertex_id = vertex.id

        image = _render_canvas(canvas)
        marker_center = canvas._image_to_widget(vertex.x, vertex.y).toPoint()

        self.assertEqual(
            image.pixelColor(marker_center).name(),
            WALL_MIRROR_VERTEX_FILL_COLOR.name(),
        )

    def test_unflagged_selected_vertex_retains_existing_blue_color(self) -> None:
        vertex_data = VertexData()
        vertex = vertex_data.add_vertex(30.0, 40.0)
        canvas = self._track(_build_canvas(vertex_data))
        canvas.selected_vertex_id = vertex.id

        image = _render_canvas(canvas)
        marker_center = canvas._image_to_widget(vertex.x, vertex.y).toPoint()

        self.assertEqual(
            image.pixelColor(marker_center).name(),
            SELECTED_VERTEX_FILL_COLOR.name(),
        )

    def test_every_vertex_in_a_group_selection_is_blue(self) -> None:
        vertex_data = VertexData()
        first = vertex_data.add_vertex(25.0, 40.0)
        second = vertex_data.add_vertex(75.0, 40.0)
        canvas = self._track(_build_canvas(vertex_data))
        canvas.set_selected_vertex_ids((first.id, second.id))

        image = _render_canvas(canvas)

        for vertex in (first, second):
            marker_center = canvas._image_to_widget(
                vertex.x,
                vertex.y,
            ).toPoint()
            self.assertEqual(
                image.pixelColor(marker_center).name(),
                SELECTED_VERTEX_FILL_COLOR.name(),
            )

    def test_mirrored_local_vertex_selects_without_creating_a_vertex(self) -> None:
        vertex_data = VertexData()
        vertex = vertex_data.add_vertex(70.0, 60.0)
        canvas = self._track(_build_canvas(vertex_data))
        canvas.set_wall_mirror_vertex_ids((vertex.id,))
        marker_center = canvas._image_to_widget(vertex.x, vertex.y).toPoint()

        QTest.mouseClick(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=marker_center,
        )

        self.assertEqual(len(vertex_data.vertices), 1)
        self.assertEqual(canvas.selected_vertex_ids, (vertex.id,))
        self.assertEqual(canvas.active_vertex_id, vertex.id)

    def test_loading_another_level_clears_stale_mirror_markers(self) -> None:
        canvas = self._track(_build_canvas())
        canvas.set_wall_mirror_vertex_ids((2,))

        canvas.set_level_data(
            vertex_data=VertexData(),
            rooms=[],
            image_path=None,
        )

        self.assertEqual(canvas.get_wall_mirror_vertex_ids(), frozenset())

    def test_pruning_a_deleted_vertex_clears_its_drawing_state(self) -> None:
        vertex_data = VertexData()
        vertex = vertex_data.add_vertex(30.0, 40.0)
        canvas = self._track(_build_canvas(vertex_data))
        canvas.selected_vertex_id = vertex.id
        canvas.active_vertex_id = vertex.id
        canvas.pressed_vertex_id = vertex.id
        canvas.drag_vertex_id = vertex.id
        canvas.preview_point = (vertex.x, vertex.y)
        vertex_data.delete_vertex(vertex.id)

        self.assertTrue(canvas.prune_missing_vertex_references())

        self.assertEqual(canvas.selected_vertex_ids, ())
        self.assertIsNone(canvas.active_vertex_id)
        self.assertIsNone(canvas.pressed_vertex_id)
        self.assertIsNone(canvas.drag_vertex_id)
        self.assertIsNone(canvas.preview_point)


if __name__ == "__main__":
    unittest.main()
