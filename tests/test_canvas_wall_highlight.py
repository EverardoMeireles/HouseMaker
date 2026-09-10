# ### Environment setup ###
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from housemaker.blueprint_canvas import (
    SELECTED_WALL_EDGE_COLOR,
    BlueprintCanvas,
)
from housemaker.models import LevelData, RoomData, VertexData
from housemaker.surface_geometry import build_wall_surface_id


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_level(*, collinear_wall: bool = False) -> LevelData:
    vertex_data = VertexData()
    boundary_points = (
        (
            (10.0, 20.0),
            (50.0, 20.0),
            (90.0, 20.0),
            (90.0, 90.0),
            (10.0, 90.0),
        )
        if collinear_wall
        else (
            (10.0, 20.0),
            (90.0, 20.0),
            (90.0, 90.0),
            (10.0, 90.0),
        )
    )
    boundary_ids = tuple(
        vertex_data.add_vertex(*point).id for point in boundary_points
    )
    for start_id, end_id in zip(
        boundary_ids,
        (*boundary_ids[1:], boundary_ids[0]),
    ):
        vertex_data.add_edge(start_id, end_id)
    center = vertex_data.add_vertex(50.0, 55.0)
    room = RoomData(
        name="Room",
        vertex_ids=boundary_ids,
        center_vertex_id=center.id,
        color_rgb=(150, 180, 210),
    )
    return LevelData(
        index=2,
        name="Ground",
        vertex_data=vertex_data,
        rooms=[room],
    )


def _build_canvas(level: LevelData) -> BlueprintCanvas:
    canvas = BlueprintCanvas()
    canvas.resize(640, 520)
    canvas.set_level_data(
        vertex_data=level.vertex_data,
        rooms=level.rooms,
        image_path=None,
        doorways=level.doorways,
        windows=level.windows,
    )
    canvas.blueprint_image = QImage(100, 100, QImage.Format.Format_RGB32)
    canvas.blueprint_image.fill(Qt.GlobalColor.white)
    canvas.set_stair_context((), level)
    canvas.show()
    _qt_application.processEvents()
    return canvas


def _first_wall_surface_id(level: LevelData, end_vertex_id: int) -> str:
    return build_wall_surface_id(
        level.index,
        f"1:{end_vertex_id}",
        level.rooms[0].center_vertex_id,
    )


def _render_canvas(canvas: BlueprintCanvas) -> QImage:
    rendered = QImage(canvas.size(), QImage.Format.Format_ARGB32)
    rendered.fill(Qt.GlobalColor.transparent)
    canvas.render(rendered)
    return rendered


# ### Tests ###
class CanvasWallHighlightTests(unittest.TestCase):
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

    def test_selected_wall_state_is_normalized_and_clearable(self) -> None:
        canvas = self._track(_build_canvas(_build_level()))
        surface_id = "level:2/room:5/wall:1:2"

        self.assertTrue(canvas.set_selected_wall_surface_id(f"  {surface_id}  "))
        self.assertEqual(canvas.get_selected_wall_surface_id(), surface_id)
        self.assertFalse(canvas.set_selected_wall_surface_id(surface_id))
        self.assertTrue(canvas.set_selected_wall_surface_id(None))
        self.assertIsNone(canvas.get_selected_wall_surface_id())

    def test_selected_wall_is_painted_with_prominent_highlight(self) -> None:
        level = _build_level()
        canvas = self._track(_build_canvas(level))
        surface_id = _first_wall_surface_id(level, end_vertex_id=2)
        sample_point = canvas._image_to_widget(50.0, 20.0).toPoint()

        without_selection = _render_canvas(canvas)
        ordinary_edge_color = without_selection.pixelColor(sample_point)

        canvas.set_selected_wall_surface_id(surface_id)
        with_selection = _render_canvas(canvas)
        self.assertEqual(
            with_selection.pixelColor(sample_point).name(),
            SELECTED_WALL_EDGE_COLOR.name(),
        )

        canvas.set_selected_wall_surface_id(None)
        cleared = _render_canvas(canvas)
        self.assertEqual(
            cleared.pixelColor(sample_point),
            ordinary_edge_color,
        )

    def test_collinear_semantic_wall_is_highlighted_across_all_edges(self) -> None:
        level = _build_level(collinear_wall=True)
        canvas = self._track(_build_canvas(level))
        surface_id = _first_wall_surface_id(level, end_vertex_id=3)

        frame = canvas._build_window_wall_frames().get(surface_id)
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(frame.start_point, (10.0, 20.0))
        self.assertEqual(frame.end_point, (90.0, 20.0))

        canvas.set_selected_wall_surface_id(surface_id)
        rendered = _render_canvas(canvas)
        for image_x in (30.0, 70.0):
            with self.subTest(image_x=image_x):
                sample_point = canvas._image_to_widget(
                    image_x,
                    20.0,
                ).toPoint()
                self.assertEqual(
                    rendered.pixelColor(sample_point).name(),
                    SELECTED_WALL_EDGE_COLOR.name(),
                )

    def test_unknown_wall_id_has_no_paint_overlay(self) -> None:
        level = _build_level()
        canvas = self._track(_build_canvas(level))
        sample_point = canvas._image_to_widget(50.0, 20.0).toPoint()
        ordinary_edge_color = _render_canvas(canvas).pixelColor(sample_point)

        canvas.set_selected_wall_surface_id("level:99/wall:100:101")

        rendered = _render_canvas(canvas)
        self.assertEqual(
            rendered.pixelColor(sample_point),
            ordinary_edge_color,
        )


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
