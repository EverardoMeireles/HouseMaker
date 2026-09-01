# ### Environment setup ###
from __future__ import annotations

import copy
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QImage, QMouseEvent, QPainter, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.blueprint_canvas import BlueprintCanvas
from housemaker.models import LevelData, RoomData, VertexData, WindowData
from housemaker.surface_geometry import build_wall_surface_id


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_square_level() -> LevelData:
    vertex_data = VertexData()
    boundary_ids = tuple(
        vertex_data.add_vertex(*point).id
        for point in (
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (0.0, 100.0),
        )
    )
    for start_id, end_id in zip(
        boundary_ids,
        (*boundary_ids[1:], boundary_ids[0]),
    ):
        vertex_data.add_edge(start_id, end_id)
    center = vertex_data.add_vertex(50.0, 50.0)
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
        floor_contour_vertex_ids=boundary_ids,
    )


def _build_window(
    level: LevelData,
    wall_key: str = "1:2",
    window_id: str = "window-horizontal",
) -> WindowData:
    room = level.rooms[0]
    return WindowData(
        window_id=window_id,
        wall_surface_id=build_wall_surface_id(
            level.index,
            wall_key,
            room.center_vertex_id,
        ),
        start_ratio=0.25,
        end_ratio=0.75,
        bottom_ratio=0.25,
        top_ratio=0.75,
    )


def _build_canvas(level: LevelData) -> BlueprintCanvas:
    canvas = BlueprintCanvas()
    canvas.resize(640, 520)
    canvas.set_level_data(
        vertex_data=level.vertex_data,
        rooms=level.rooms,
        image_path=None,
        floor_contour_vertex_ids=level.floor_contour_vertex_ids,
        doorways=level.doorways,
        windows=level.windows,
    )
    canvas.blueprint_image = QImage(100, 100, QImage.Format.Format_RGB32)
    canvas.blueprint_image.fill(Qt.GlobalColor.white)
    canvas.set_stair_context((), level)
    canvas.show()
    _qt_application.processEvents()
    return canvas


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


def _send_wheel(canvas: BlueprintCanvas, position: QPoint, delta: int) -> None:
    event = QWheelEvent(
        QPointF(position),
        QPointF(canvas.mapToGlobal(position)),
        QPoint(),
        QPoint(0, delta),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )
    QApplication.sendEvent(canvas, event)


def _assert_points_almost_equal(
    test_case: unittest.TestCase,
    actual: QPointF,
    expected: QPointF,
) -> None:
    test_case.assertAlmostEqual(actual.x(), expected.x())
    test_case.assertAlmostEqual(actual.y(), expected.y())


# ### Tests ###
class CanvasWindow2dTests(unittest.TestCase):
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

    def test_windows_resolve_semantic_wall_segments_and_render_as_strips(
        self,
    ) -> None:
        level = _build_square_level()
        horizontal = _build_window(level)
        vertical = _build_window(
            level,
            wall_key="2:3",
            window_id="window-vertical",
        )
        level.windows[:] = [horizontal, vertical]
        canvas = self._track(_build_canvas(level))

        frames = canvas._build_window_wall_frames()
        self.assertEqual(
            frames[horizontal.wall_surface_id].start_point,
            (0.0, 0.0),
        )
        self.assertEqual(
            frames[horizontal.wall_surface_id].end_point,
            (100.0, 0.0),
        )
        self.assertEqual(
            frames[vertical.wall_surface_id].start_point,
            (100.0, 0.0),
        )
        self.assertEqual(
            frames[vertical.wall_surface_id].end_point,
            (100.0, 100.0),
        )

        horizontal_segment = canvas._get_window_widget_segment(horizontal)
        vertical_segment = canvas._get_window_widget_segment(vertical)
        self.assertIsNotNone(horizontal_segment)
        self.assertIsNotNone(vertical_segment)
        assert horizontal_segment is not None
        assert vertical_segment is not None
        _assert_points_almost_equal(
            self,
            horizontal_segment[0],
            canvas._image_to_widget(25.0, 0.0),
        )
        _assert_points_almost_equal(
            self,
            horizontal_segment[1],
            canvas._image_to_widget(75.0, 0.0),
        )
        _assert_points_almost_equal(
            self,
            vertical_segment[0],
            canvas._image_to_widget(100.0, 25.0),
        )
        _assert_points_almost_equal(
            self,
            vertical_segment[1],
            canvas._image_to_widget(100.0, 75.0),
        )

        rendered = QImage(canvas.size(), QImage.Format.Format_ARGB32)
        rendered.fill(Qt.GlobalColor.transparent)
        painter = QPainter(rendered)
        canvas._paint_windows(painter)
        painter.end()
        horizontal_center = (
            (horizontal_segment[0] + horizontal_segment[1]) * 0.5
        ).toPoint()
        vertical_center = (
            (vertical_segment[0] + vertical_segment[1]) * 0.5
        ).toPoint()
        self.assertGreater(rendered.pixelColor(horizontal_center).alpha(), 0)
        self.assertGreater(rendered.pixelColor(vertical_center).alpha(), 0)

    def test_window_endpoints_and_bodies_keep_the_arrow_cursor(self) -> None:
        level = _build_square_level()
        horizontal = _build_window(level)
        vertical = _build_window(
            level,
            wall_key="2:3",
            window_id="window-vertical",
        )
        level.windows[:] = [horizontal, vertical]
        canvas = self._track(_build_canvas(level))

        for window in level.windows:
            segment = canvas._get_window_widget_segment(window)
            self.assertIsNotNone(segment)
            assert segment is not None
            for position in (
                segment[0],
                (segment[0] + segment[1]) * 0.5,
                segment[1],
            ):
                QTest.mouseMove(canvas, position.toPoint())
                _qt_application.processEvents()
                self.assertEqual(
                    canvas.cursor().shape(),
                    Qt.CursorShape.ArrowCursor,
                )

    def test_clicking_and_dragging_a_window_does_not_edit_the_canvas(self) -> None:
        level = _build_square_level()
        window = _build_window(level)
        level.windows.append(window)
        canvas = self._track(_build_canvas(level))
        expected_window = copy.deepcopy(window)
        vertex_count = len(canvas.vertex_data.vertices)
        edge_count = len(canvas.vertex_data.edges)
        segment = canvas._get_window_widget_segment(window)
        self.assertIsNotNone(segment)
        assert segment is not None
        center = ((segment[0] + segment[1]) * 0.5).toPoint()
        drag_target = canvas._image_to_widget(90.0, 0.0).toPoint()

        QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=center)
        QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=center)
        _send_drag_move(canvas, drag_target)
        QTest.mouseRelease(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=drag_target,
        )

        self.assertEqual(canvas.windows, [expected_window])
        self.assertEqual(len(canvas.vertex_data.vertices), vertex_count)
        self.assertEqual(len(canvas.vertex_data.edges), edge_count)
        self.assertIsNone(canvas.active_vertex_id)
        self.assertIsNone(canvas.selected_vertex_id)
        self.assertEqual(canvas.undo_stack, [])

    def test_wheel_over_window_zooms_without_changing_window_data(self) -> None:
        level = _build_square_level()
        window = _build_window(level)
        level.windows.append(window)
        canvas = self._track(_build_canvas(level))
        expected_window = copy.deepcopy(window)
        segment = canvas._get_window_widget_segment(window)
        self.assertIsNotNone(segment)
        assert segment is not None
        center = ((segment[0] + segment[1]) * 0.5).toPoint()
        zoom_before = canvas.zoom_scale

        _send_wheel(canvas, center, 120)

        self.assertGreater(canvas.zoom_scale, zoom_before)
        self.assertEqual(canvas.windows, [expected_window])
        self.assertEqual(canvas.undo_stack, [])


if __name__ == "__main__":
    unittest.main()
