# ### Environment setup ###
from __future__ import annotations

import copy
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.blueprint_canvas import (
    CANVAS_WINDOW_WALL_MARGIN_METERS,
    PENDING_WINDOW_FILL_COLOR,
    WINDOW_FILL_COLOR,
    BlueprintCanvas,
)
from housemaker.canvas_openings import (
    CANVAS_OPENING_WINDOW,
    CanvasOpeningEdit,
    CanvasOpeningReference,
)
from housemaker.models import LevelData, RoomData, VertexData, WindowData
from housemaker.surface_geometry import (
    MIN_WINDOW_SIZE_METERS,
    WallWindowPlacement,
    build_wall_surface_id,
)

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


def _assert_colors_near(
    test_case: unittest.TestCase,
    actual: QColor,
    expected: QColor,
) -> None:
    for actual_channel, expected_channel in zip(
        actual.getRgb(),
        expected.getRgb(),
    ):
        test_case.assertAlmostEqual(
            actual_channel,
            expected_channel,
            delta=2,
        )


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
        vertical_center = ((vertical_segment[0] + vertical_segment[1]) * 0.5).toPoint()
        self.assertGreater(rendered.pixelColor(horizontal_center).alpha(), 0)
        self.assertGreater(rendered.pixelColor(vertical_center).alpha(), 0)

    def test_window_body_selects_by_stable_id_and_exposes_end_handles(
        self,
    ) -> None:
        level = _build_square_level()
        window = _build_window(level)
        level.windows.append(window)
        canvas = self._track(_build_canvas(level))
        selections: list[object] = []
        canvas.canvas_opening_selection_requested.connect(selections.append)
        segment = canvas._get_window_widget_segment(window)
        self.assertIsNotNone(segment)
        assert segment is not None
        center = ((segment[0] + segment[1]) * 0.5).toPoint()

        QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=center)
        _qt_application.processEvents()

        self.assertEqual(canvas.selected_window_id, window.window_id)
        self.assertEqual(len(selections), 1)
        reference = selections[0]
        self.assertIsInstance(reference, CanvasOpeningReference)
        assert isinstance(reference, CanvasOpeningReference)
        self.assertEqual(reference.kind, CANVAS_OPENING_WINDOW)
        self.assertEqual(reference.level_index, level.index)
        self.assertEqual(reference.item_index, 0)
        self.assertEqual(reference.stable_id, window.window_id)

        QTest.mouseMove(canvas, center)
        _qt_application.processEvents()
        self.assertEqual(canvas.cursor().shape(), Qt.CursorShape.OpenHandCursor)
        for endpoint in segment:
            with self.subTest(endpoint=endpoint):
                QTest.mouseMove(canvas, endpoint.toPoint())
                _qt_application.processEvents()
                self.assertEqual(
                    canvas.cursor().shape(),
                    Qt.CursorShape.SizeHorCursor,
                )

    def test_window_body_drag_moves_only_along_its_wall_and_emits_edits(
        self,
    ) -> None:
        cases = (
            ("1:2", "horizontal", (70.0, 35.0), 0),
            ("2:3", "vertical", (65.0, 70.0), 1),
        )
        for wall_key, window_id, target_image, axis_index in cases:
            with self.subTest(wall_key=wall_key):
                level = _build_square_level()
                original = _build_window(
                    level,
                    wall_key=wall_key,
                    window_id=window_id,
                )
                level.windows.append(original)
                canvas = self._track(_build_canvas(level))
                segment = canvas._get_window_widget_segment(original)
                self.assertIsNotNone(segment)
                assert segment is not None
                center = ((segment[0] + segment[1]) * 0.5).toPoint()
                QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=center)

                started: list[CanvasOpeningEdit] = []
                previews: list[CanvasOpeningEdit] = []
                finished: list[tuple[CanvasOpeningEdit, bool]] = []
                canvas.canvas_opening_edit_started.connect(started.append)
                canvas.canvas_opening_edit_preview_changed.connect(previews.append)
                canvas.canvas_opening_edit_finished.connect(
                    lambda edit, changed, finished=finished: finished.append(
                        (edit, changed)
                    )
                )
                target = canvas._image_to_widget(*target_image).toPoint()
                press_image = canvas._widget_to_image_clamped(QPointF(center))
                target_image_point = canvas._widget_to_image_clamped(QPointF(target))

                QTest.mousePress(
                    canvas,
                    Qt.MouseButton.LeftButton,
                    pos=center,
                )
                _send_drag_move(canvas, target)
                QTest.mouseRelease(
                    canvas,
                    Qt.MouseButton.LeftButton,
                    pos=target,
                )

                self.assertEqual(len(started), 1)
                self.assertGreaterEqual(len(previews), 1)
                self.assertEqual(len(finished), 1)
                final_edit, changed = finished[0]
                self.assertTrue(changed)
                self.assertEqual(final_edit, previews[-1])
                self.assertEqual(final_edit.reference.stable_id, window_id)
                self.assertEqual(
                    final_edit.wall_surface_id,
                    original.wall_surface_id,
                )
                self.assertEqual(
                    started[0].bounds.bottom_ratio,
                    original.bottom_ratio,
                )
                self.assertEqual(
                    started[0].bounds.top_ratio,
                    original.top_ratio,
                )
                pointer_delta = (
                    target_image_point.x() - press_image.x(),
                    target_image_point.y() - press_image.y(),
                )[axis_index]
                expected_center = min(
                    max(0.5 + pointer_delta / 100.0, 0.25),
                    0.75,
                )
                self.assertAlmostEqual(
                    final_edit.bounds.start_ratio,
                    expected_center - 0.25,
                    places=6,
                )
                self.assertAlmostEqual(
                    final_edit.bounds.end_ratio,
                    expected_center + 0.25,
                    places=6,
                )
                self.assertEqual(final_edit.bounds.bottom_ratio, 0.25)
                self.assertEqual(final_edit.bounds.top_ratio, 0.75)
                self.assertEqual(original.start_ratio, 0.25)
                self.assertEqual(original.end_ratio, 0.75)
                self.assertEqual(canvas.undo_stack, [])

    def test_each_window_endpoint_resizes_with_the_other_endpoint_anchored(
        self,
    ) -> None:
        cases = (
            (0, 10.0, 0.10, 0.75),
            (1, 90.0, 0.25, 0.90),
        )
        for endpoint_index, target_x, expected_start, expected_end in cases:
            with self.subTest(endpoint_index=endpoint_index):
                level = _build_square_level()
                original = _build_window(level)
                level.windows.append(original)
                canvas = self._track(_build_canvas(level))
                segment = canvas._get_window_widget_segment(original)
                self.assertIsNotNone(segment)
                assert segment is not None
                center = ((segment[0] + segment[1]) * 0.5).toPoint()
                QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=center)
                endpoint = segment[endpoint_index].toPoint()
                target = canvas._image_to_widget(target_x, 35.0).toPoint()
                started: list[CanvasOpeningEdit] = []
                previews: list[CanvasOpeningEdit] = []
                finished: list[tuple[CanvasOpeningEdit, bool]] = []
                canvas.canvas_opening_edit_started.connect(started.append)
                canvas.canvas_opening_edit_preview_changed.connect(previews.append)
                canvas.canvas_opening_edit_finished.connect(
                    lambda edit, changed, finished=finished: finished.append(
                        (edit, changed)
                    )
                )

                QTest.mousePress(
                    canvas,
                    Qt.MouseButton.LeftButton,
                    pos=endpoint,
                )
                _send_drag_move(canvas, target)
                QTest.mouseRelease(
                    canvas,
                    Qt.MouseButton.LeftButton,
                    pos=target,
                )

                self.assertEqual(len(started), 1)
                self.assertGreaterEqual(len(previews), 1)
                self.assertEqual(len(finished), 1)
                final_edit, changed = finished[0]
                self.assertTrue(changed)
                self.assertAlmostEqual(
                    final_edit.bounds.start_ratio,
                    expected_start,
                    places=2,
                )
                self.assertAlmostEqual(
                    final_edit.bounds.end_ratio,
                    expected_end,
                    places=2,
                )
                self.assertEqual(final_edit.bounds.bottom_ratio, 0.25)
                self.assertEqual(final_edit.bounds.top_ratio, 0.75)
                self.assertEqual(final_edit.reference.stable_id, original.window_id)
                self.assertEqual(original.start_ratio, 0.25)
                self.assertEqual(original.end_ratio, 0.75)
                self.assertEqual(canvas.undo_stack, [])

    def test_window_move_and_endpoint_resize_clamp_inside_the_wall(self) -> None:
        cases = (
            ("move", None, 110.0, 0.50, 1.00),
            (
                "resize",
                0,
                110.0,
                0.75 - MIN_WINDOW_SIZE_METERS / (100.0 * 0.02),
                0.75,
            ),
            (
                "resize",
                1,
                -10.0,
                0.25,
                0.25 + MIN_WINDOW_SIZE_METERS / (100.0 * 0.02),
            ),
        )
        for operation, endpoint_index, target_x, expected_start, expected_end in cases:
            with self.subTest(operation=operation, endpoint_index=endpoint_index):
                level = _build_square_level()
                original = _build_window(level)
                level.windows.append(original)
                canvas = self._track(_build_canvas(level))
                segment = canvas._get_window_widget_segment(original)
                self.assertIsNotNone(segment)
                assert segment is not None
                center = ((segment[0] + segment[1]) * 0.5).toPoint()
                QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=center)
                press = (
                    center
                    if endpoint_index is None
                    else segment[endpoint_index].toPoint()
                )
                target = canvas._image_to_widget(target_x, 0.0).toPoint()
                finished: list[tuple[CanvasOpeningEdit, bool]] = []
                canvas.canvas_opening_edit_finished.connect(
                    lambda edit, changed, finished=finished: finished.append(
                        (edit, changed)
                    )
                )

                QTest.mousePress(
                    canvas,
                    Qt.MouseButton.LeftButton,
                    pos=press,
                )
                _send_drag_move(canvas, target)
                QTest.mouseRelease(
                    canvas,
                    Qt.MouseButton.LeftButton,
                    pos=target,
                )

                self.assertEqual(len(finished), 1)
                final_edit, changed = finished[0]
                self.assertTrue(changed)
                self.assertAlmostEqual(
                    final_edit.bounds.start_ratio,
                    expected_start,
                    places=6,
                )
                self.assertAlmostEqual(
                    final_edit.bounds.end_ratio,
                    expected_end,
                    places=6,
                )
                self.assertEqual(final_edit.bounds.bottom_ratio, 0.25)
                self.assertEqual(final_edit.bounds.top_ratio, 0.75)

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
        QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=center)
        edits: list[object] = []
        canvas.canvas_opening_edit_preview_changed.connect(edits.append)
        zoom_before = canvas.zoom_scale

        _send_wheel(canvas, center, 120)

        self.assertGreater(canvas.zoom_scale, zoom_before)
        self.assertEqual(canvas.selected_window_id, window.window_id)
        self.assertEqual(canvas.windows, [expected_window])
        self.assertEqual(edits, [])
        self.assertEqual(canvas.undo_stack, [])

    def test_window_tool_hover_and_click_emit_one_centered_scaled_opening(
        self,
    ) -> None:
        level = _build_square_level()
        level.scale = 2.0
        canvas = self._track(_build_canvas(level))
        placements: list[WallWindowPlacement] = []
        canvas.window_placement_requested.connect(placements.append)

        self.assertTrue(canvas.start_window_placement())
        hover_position = canvas._image_to_widget(50.0, 0.0).toPoint()
        QTest.mouseMove(canvas, hover_position)
        _qt_application.processEvents()

        preview = canvas._pending_window_placement
        self.assertIsInstance(preview, WallWindowPlacement)
        assert preview is not None
        wall_length_meters = 100.0 * 0.02 * level.scale
        self.assertAlmostEqual(
            (preview.end_ratio - preview.start_ratio) * wall_length_meters,
            1.0,
        )
        self.assertAlmostEqual(
            (preview.top_ratio - preview.bottom_ratio) * level.rooms[0].height_meters,
            1.0,
        )
        self.assertAlmostEqual(
            (preview.bottom_ratio + preview.top_ratio) * 0.5,
            0.5,
        )

        QTest.mouseClick(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=hover_position,
        )
        _qt_application.processEvents()

        self.assertEqual(placements, [preview])
        self.assertFalse(canvas.is_window_placement_active())
        self.assertIsNone(canvas._pending_window_placement)

    def test_window_tool_clamps_default_width_inside_wall_endpoints(self) -> None:
        level = _build_square_level()
        level.scale = 2.0
        canvas = self._track(_build_canvas(level))
        wall_length_meters = 100.0 * 0.02 * level.scale
        expected_margin_ratio = CANVAS_WINDOW_WALL_MARGIN_METERS / wall_length_meters

        self.assertTrue(canvas.start_window_placement())
        canvas._update_pending_window(QPointF(0.0, 0.0))
        start_preview = canvas._pending_window_placement
        self.assertIsNotNone(start_preview)
        assert start_preview is not None
        self.assertAlmostEqual(start_preview.start_ratio, expected_margin_ratio)
        self.assertAlmostEqual(
            start_preview.end_ratio - start_preview.start_ratio,
            1.0 / wall_length_meters,
        )

        canvas._update_pending_window(QPointF(100.0, 0.0))
        end_preview = canvas._pending_window_placement
        self.assertIsNotNone(end_preview)
        assert end_preview is not None
        self.assertAlmostEqual(end_preview.end_ratio, 1.0 - expected_margin_ratio)
        self.assertAlmostEqual(
            end_preview.end_ratio - end_preview.start_ratio,
            1.0 / wall_length_meters,
        )

    def test_committed_and_pending_windows_use_purple_fills(self) -> None:
        level = _build_square_level()
        committed = _build_window(level)
        level.windows.append(committed)
        canvas = self._track(_build_canvas(level))
        committed_segment = canvas._get_window_widget_segment(committed)
        self.assertIsNotNone(committed_segment)
        assert committed_segment is not None

        committed_render = QImage(
            canvas.size(),
            QImage.Format.Format_ARGB32,
        )
        committed_render.fill(Qt.GlobalColor.transparent)
        committed_painter = QPainter(committed_render)
        canvas._paint_windows(committed_painter)
        committed_painter.end()
        committed_center = (
            (committed_segment[0] + committed_segment[1]) * 0.5
        ).toPoint()
        _assert_colors_near(
            self,
            committed_render.pixelColor(committed_center),
            WINDOW_FILL_COLOR,
        )

        self.assertTrue(canvas.start_window_placement())
        canvas._update_pending_window(QPointF(50.0, 0.0))
        pending = canvas._pending_window_placement
        self.assertIsNotNone(pending)
        assert pending is not None
        pending_segment = canvas._get_window_widget_segment(pending)
        self.assertIsNotNone(pending_segment)
        assert pending_segment is not None
        pending_render = QImage(
            canvas.size(),
            QImage.Format.Format_ARGB32,
        )
        pending_render.fill(Qt.GlobalColor.transparent)
        pending_painter = QPainter(pending_render)
        canvas._paint_pending_window(pending_painter)
        pending_painter.end()
        pending_center = ((pending_segment[0] + pending_segment[1]) * 0.5).toPoint()
        _assert_colors_near(
            self,
            pending_render.pixelColor(pending_center),
            PENDING_WINDOW_FILL_COLOR,
        )

    def test_escape_and_right_click_cancel_window_placement(self) -> None:
        level = _build_square_level()
        canvas = self._track(_build_canvas(level))
        placements: list[WallWindowPlacement] = []
        canvas.window_placement_requested.connect(placements.append)
        hover_position = canvas._image_to_widget(50.0, 0.0).toPoint()

        self.assertTrue(canvas.start_window_placement())
        QTest.mouseMove(canvas, hover_position)
        QTest.keyClick(canvas, Qt.Key.Key_Escape)
        self.assertFalse(canvas.is_window_placement_active())
        self.assertIsNone(canvas._pending_window_placement)

        self.assertTrue(canvas.start_window_placement())
        QTest.mouseMove(canvas, hover_position)
        QTest.mouseClick(
            canvas,
            Qt.MouseButton.RightButton,
            pos=hover_position,
        )
        self.assertFalse(canvas.is_window_placement_active())
        self.assertIsNone(canvas._pending_window_placement)
        self.assertEqual(placements, [])


if __name__ == "__main__":
    unittest.main()
