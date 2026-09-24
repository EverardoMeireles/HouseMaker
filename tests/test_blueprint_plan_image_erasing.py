# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.blueprint_canvas import BlueprintCanvas, PlanImageEraseCommit
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


def _send_wheel(canvas: BlueprintCanvas, position: QPoint, delta: int) -> None:
    event = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(position),
        QPointF(canvas.mapToGlobal(position)),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(canvas, event)
    QTest.qWait(1)
    wheel_event = QWheelEvent(
        QPointF(position),
        QPointF(canvas.mapToGlobal(position)),
        QPoint(0, 0),
        QPoint(0, delta),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(canvas, wheel_event)


# ### Plan-image erase tests ###
class BlueprintPlanImageErasingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)
        self.source_path = self.directory / "source.png"
        Image.new("RGB", (200, 120), "black").save(self.source_path)
        self.source_bytes = self.source_path.read_bytes()
        self.canvas = BlueprintCanvas()
        self.canvas.resize(640, 520)
        self.canvas.load_blueprint(str(self.source_path))
        self.canvas.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.canvas.close()
        self.canvas.deleteLater()
        _qt_application.processEvents()
        self.temporary_directory.cleanup()

    def _next_output(self, name: str = "erased.png") -> Path:
        return self.directory / name

    def _drag_image_rectangle(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> None:
        start_widget = self.canvas._image_to_widget(*start).toPoint()
        end_widget = self.canvas._image_to_widget(*end).toPoint()
        QTest.mousePress(
            self.canvas,
            Qt.MouseButton.LeftButton,
            pos=start_widget,
        )
        _send_drag_move(self.canvas, end_widget)
        QTest.mouseRelease(
            self.canvas,
            Qt.MouseButton.LeftButton,
            pos=end_widget,
        )
        _qt_application.processEvents()

    def test_changed_selection_creates_one_immutable_revision_and_stays_active(
        self,
    ) -> None:
        output_path = self._next_output()
        mode_changes: list[bool] = []
        commits: list[PlanImageEraseCommit] = []
        self.canvas.plan_image_erase_mode_changed.connect(mode_changes.append)
        self.canvas.plan_image_erase_committed.connect(commits.append)
        vertex_data = self.canvas.vertex_data
        self.canvas.zoom_scale = 1.4
        self.canvas.view_offset = QPointF(13.0, -9.0)
        self.canvas.canvas_level_scale = 1.25
        self.canvas.canvas_offset_x_pixels = 17.0
        self.canvas.canvas_offset_y_pixels = -11.0

        self.assertTrue(self.canvas.start_plan_image_erasing(output_path))
        self._drag_image_rectangle((30.0, 25.0), (170.0, 95.0))

        self.assertEqual(mode_changes, [True])
        self.assertTrue(self.canvas.is_plan_image_erasing())
        self.assertEqual(len(commits), 1)
        commit = commits[0]
        self.assertEqual(commit.previous_path, str(self.source_path.resolve()))
        self.assertEqual(commit.replacement_path, str(output_path.resolve()))
        self.assertEqual(commit.image_size_pixels, (200.0, 120.0))
        self.assertEqual(
            commit.replacement_revision,
            self.canvas.get_blueprint_image_revision(),
        )
        self.assertTrue(output_path.is_file())
        self.assertEqual(self.source_path.read_bytes(), self.source_bytes)
        with Image.open(output_path) as output_image:
            self.assertEqual(output_image.getpixel((100, 60)), (255, 255, 255))
            self.assertEqual(output_image.getpixel((100, 5)), (0, 0, 0))
            self.assertEqual(output_image.getpixel((10, 60)), (0, 0, 0))
        self.assertIs(self.canvas.vertex_data, vertex_data)
        self.assertEqual(self.canvas.undo_stack, [])
        self.assertAlmostEqual(self.canvas.zoom_scale, 1.4)
        self.assertEqual(self.canvas.view_offset, QPointF(13.0, -9.0))

    def test_marquee_is_non_destructive_until_release_and_does_not_add_walls(
        self,
    ) -> None:
        output_path = self._next_output()
        geometry = VertexData()
        first_vertex = geometry.add_vertex(5.0, 5.0)
        second_vertex = geometry.add_vertex(15.0, 5.0)
        geometry.add_edge(first_vertex.id, second_vertex.id)
        self.canvas.vertex_data = geometry
        geometry_before = geometry.clone()
        self.assertTrue(self.canvas.start_plan_image_erasing(output_path))
        start_widget = self.canvas._image_to_widget(20.0, 20.0).toPoint()
        end_widget = self.canvas._image_to_widget(180.0, 100.0).toPoint()

        QTest.mousePress(
            self.canvas,
            Qt.MouseButton.LeftButton,
            pos=start_widget,
        )
        _send_drag_move(self.canvas, end_widget)
        _qt_application.processEvents()

        self.assertIs(self.canvas.vertex_data, geometry)
        self.assertEqual(self.canvas.vertex_data, geometry_before)
        self.assertEqual(self.canvas.selected_vertex_ids, ())
        self.assertEqual(
            self.canvas.blueprint_image.pixelColor(100, 60).name(),
            "#000000",
        )
        preview = self.canvas.grab().toImage()
        preview_center = self.canvas._image_to_widget(100.0, 60.0).toPoint()
        self.assertGreater(preview.pixelColor(preview_center).red(), 0)
        self.assertFalse(output_path.exists())

        QTest.mouseRelease(
            self.canvas,
            Qt.MouseButton.LeftButton,
            pos=end_widget,
        )
        _qt_application.processEvents()

        with Image.open(output_path) as output_image:
            self.assertEqual(output_image.getpixel((100, 60)), (255, 255, 255))

    def test_owner_can_supply_a_new_immutable_path_after_each_commit(self) -> None:
        first_output = self._next_output("first.png")
        second_output = self._next_output("second.png")
        commits: list[PlanImageEraseCommit] = []

        def record_and_replenish(commit: PlanImageEraseCommit) -> None:
            commits.append(commit)
            if len(commits) == 1:
                self.canvas.set_plan_image_erase_output_path(second_output)

        self.canvas.plan_image_erase_committed.connect(record_and_replenish)
        self.assertTrue(self.canvas.start_plan_image_erasing(first_output))

        self._drag_image_rectangle((30.0, 15.0), (170.0, 50.0))
        self._drag_image_rectangle((30.0, 70.0), (170.0, 105.0))

        self.assertEqual(len(commits), 2)
        self.assertEqual(commits[0].replacement_path, str(first_output.resolve()))
        self.assertEqual(commits[1].previous_path, str(first_output.resolve()))
        self.assertEqual(commits[1].replacement_path, str(second_output.resolve()))
        self.assertTrue(first_output.is_file())
        self.assertTrue(second_output.is_file())

    def test_selection_mapping_honors_zoom_pan_and_canvas_transforms(self) -> None:
        self.canvas.zoom_scale = 2.3
        self.canvas.view_offset = QPointF(-37.0, 29.0)
        self.canvas.canvas_level_scale = 1.4
        self.canvas.canvas_offset_x_pixels = 21.0
        self.canvas.canvas_offset_y_pixels = -18.0
        output_path = self._next_output()
        self.assertTrue(self.canvas.start_plan_image_erasing(output_path))

        self._drag_image_rectangle((40.0, 30.0), (160.0, 90.0))

        with Image.open(output_path) as output_image:
            self.assertEqual(output_image.getpixel((100, 60)), (255, 255, 255))
            self.assertEqual(output_image.getpixel((20, 60)), (0, 0, 0))
            self.assertEqual(output_image.getpixel((180, 60)), (0, 0, 0))
            self.assertEqual(output_image.getpixel((100, 15)), (0, 0, 0))
            self.assertEqual(output_image.getpixel((100, 105)), (0, 0, 0))

    def test_drag_is_clamped_to_image_bounds(self) -> None:
        output_path = self._next_output()
        self.assertTrue(self.canvas.start_plan_image_erasing(output_path))
        start = self.canvas._image_to_widget(50.0, 35.0).toPoint()
        outside = QPoint(self.canvas.width() + 300, self.canvas.height() + 300)

        QTest.mousePress(self.canvas, Qt.MouseButton.LeftButton, pos=start)
        _send_drag_move(self.canvas, outside)
        QTest.mouseRelease(self.canvas, Qt.MouseButton.LeftButton, pos=outside)
        _qt_application.processEvents()

        with Image.open(output_path) as output_image:
            self.assertEqual(output_image.getpixel((199, 119)), (255, 255, 255))
            self.assertEqual(output_image.getpixel((25, 90)), (0, 0, 0))

    def test_click_and_thin_drag_are_no_ops_without_consuming_output_path(self) -> None:
        output_path = self._next_output()
        commits: list[object] = []
        self.canvas.plan_image_erase_committed.connect(commits.append)
        self.assertTrue(self.canvas.start_plan_image_erasing(output_path))
        position = self.canvas._image_to_widget(100.0, 60.0).toPoint()

        QTest.mouseClick(self.canvas, Qt.MouseButton.LeftButton, pos=position)
        thin_end = position + QPoint(80, 2)
        QTest.mousePress(self.canvas, Qt.MouseButton.LeftButton, pos=position)
        _send_drag_move(self.canvas, thin_end)
        QTest.mouseRelease(self.canvas, Qt.MouseButton.LeftButton, pos=thin_end)
        _qt_application.processEvents()

        self.assertEqual(commits, [])
        self.assertFalse(output_path.exists())
        self.assertEqual(self.canvas._plan_image_erase_output_path, output_path)
        self.assertEqual(
            self.canvas.blueprint_image.pixelColor(100, 60).name(),
            "#000000",
        )

    def test_unchanged_white_selection_does_not_consume_path_or_emit_commit(self) -> None:
        white_path = self.directory / "white.png"
        Image.new("RGB", (200, 120), "white").save(white_path)
        self.assertTrue(
            self.canvas.load_blueprint_image_preserving_view(str(white_path))
        )
        output_path = self._next_output()
        commits: list[object] = []
        self.canvas.plan_image_erase_committed.connect(commits.append)
        self.assertTrue(self.canvas.start_plan_image_erasing(output_path))

        self._drag_image_rectangle((30.0, 25.0), (170.0, 95.0))

        self.assertEqual(commits, [])
        self.assertFalse(output_path.exists())
        self.assertEqual(self.canvas._plan_image_erase_output_path, output_path)

    def test_escape_and_right_click_exit_and_restore_uncommitted_pixels(self) -> None:
        output_path = self._next_output()
        mode_changes: list[bool] = []
        self.canvas.plan_image_erase_mode_changed.connect(mode_changes.append)
        self.assertTrue(self.canvas.start_plan_image_erasing(output_path))
        position = self.canvas._image_to_widget(100.0, 60.0).toPoint()
        QTest.mousePress(self.canvas, Qt.MouseButton.LeftButton, pos=position)

        escape_event = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_Escape,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(self.canvas, escape_event)

        self.assertFalse(self.canvas.is_plan_image_erasing())
        self.assertEqual(mode_changes, [True, False])
        self.assertFalse(output_path.exists())
        self.assertEqual(
            self.canvas.blueprint_image.pixelColor(100, 60).name(),
            "#000000",
        )

        second_output = self._next_output("second.png")
        self.assertTrue(self.canvas.start_plan_image_erasing(second_output))
        QTest.mouseClick(
            self.canvas,
            Qt.MouseButton.RightButton,
            pos=position,
        )
        self.assertFalse(self.canvas.is_plan_image_erasing())
        self.assertFalse(second_output.exists())

    def test_save_failure_restores_pixels_and_emits_failure_only(self) -> None:
        output_path = self._next_output()
        commits: list[object] = []
        failures: list[str] = []
        self.canvas.plan_image_erase_committed.connect(commits.append)
        self.canvas.plan_image_erase_failed.connect(failures.append)
        self.assertTrue(self.canvas.start_plan_image_erasing(output_path))

        with patch(
            "housemaker.blueprint_canvas._save_plan_image_png_atomically",
            side_effect=OSError("disk full"),
        ):
            self._drag_image_rectangle((30.0, 25.0), (170.0, 95.0))

        self.assertEqual(commits, [])
        self.assertEqual(failures, ["disk full"])
        self.assertTrue(self.canvas.is_plan_image_erasing())
        self.assertIsNone(self.canvas._plan_image_erase_output_path)
        self.assertEqual(
            self.canvas.blueprint_image.pixelColor(100, 60).name(),
            "#000000",
        )
        self.assertFalse(output_path.exists())

    def test_middle_pan_and_wheel_zoom_remain_available_in_erase_mode(self) -> None:
        self.canvas.zoom_scale = 2.0
        self.assertTrue(self.canvas.start_plan_image_erasing(self._next_output()))
        start = QPoint(250, 220)
        end = QPoint(290, 250)

        QTest.mousePress(
            self.canvas,
            Qt.MouseButton.MiddleButton,
            pos=start,
        )
        middle_move = QMouseEvent(
            QEvent.Type.MouseMove,
            QPointF(end),
            QPointF(self.canvas.mapToGlobal(end)),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.MiddleButton,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(self.canvas, middle_move)
        QTest.mouseRelease(
            self.canvas,
            Qt.MouseButton.MiddleButton,
            pos=end,
        )
        offset_after_pan = QPointF(self.canvas.view_offset)
        zoom_before = self.canvas.zoom_scale
        _send_wheel(self.canvas, end, 120)

        self.assertEqual(offset_after_pan, QPointF(40.0, 30.0))
        self.assertGreater(self.canvas.zoom_scale, zoom_before)
        self.assertTrue(self.canvas.is_plan_image_erasing())

    def test_preserving_load_keeps_geometry_view_and_history_but_stops_mode(
        self,
    ) -> None:
        replacement_path = self.directory / "replacement.png"
        Image.new("RGB", (200, 120), "white").save(replacement_path)
        vertex_data = VertexData()
        vertex_data.add_vertex(20.0, 30.0)
        self.canvas.vertex_data = vertex_data
        self.canvas._push_undo_state()
        self.canvas.zoom_scale = 1.8
        self.canvas.view_offset = QPointF(11.0, 17.0)
        self.assertTrue(self.canvas.start_plan_image_erasing(self._next_output()))

        self.assertTrue(
            self.canvas.load_blueprint_image_preserving_view(
                str(replacement_path)
            )
        )

        self.assertFalse(self.canvas.is_plan_image_erasing())
        self.assertIs(self.canvas.vertex_data, vertex_data)
        self.assertEqual(len(self.canvas.undo_stack), 1)
        self.assertAlmostEqual(self.canvas.zoom_scale, 1.8)
        self.assertEqual(self.canvas.view_offset, QPointF(11.0, 17.0))
        self.assertEqual(
            self.canvas.blueprint_path,
            str(replacement_path.resolve()),
        )

    def test_other_placement_modes_stop_erasing(self) -> None:
        mode_changes: list[bool] = []
        self.canvas.plan_image_erase_mode_changed.connect(mode_changes.append)
        self.assertTrue(self.canvas.start_plan_image_erasing(self._next_output()))

        self.assertTrue(self.canvas.start_open_space_placement())

        self.assertFalse(self.canvas.is_plan_image_erasing())
        self.assertTrue(self.canvas.is_open_space_placement_active())
        self.assertEqual(mode_changes, [True, False])

    def test_output_path_must_be_a_new_png(self) -> None:
        with self.assertRaisesRegex(ValueError, "PNG"):
            self.canvas.start_plan_image_erasing(self.directory / "bad.jpg")
        existing_path = self.directory / "existing.png"
        Image.new("RGB", (1, 1), "white").save(existing_path)
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.canvas.start_plan_image_erasing(existing_path)


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
