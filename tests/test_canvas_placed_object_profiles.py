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

from housemaker.blueprint_canvas import (
    BlueprintCanvas,
    CanvasPlacedObjectProfile,
)
from housemaker.models import LevelData

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_level(level_index: int = 2) -> LevelData:
    return LevelData(
        index=level_index,
        name=f"Level {level_index}",
        image_size_pixels=(100.0, 100.0),
    )


def _build_profile(
    object_id: str = "chair",
    *,
    level_index: int = 2,
    rotation_z: float = 0.0,
) -> CanvasPlacedObjectProfile:
    return CanvasPlacedObjectProfile(
        object_id=object_id,
        level_index=level_index,
        anchor_x=40.0,
        anchor_y=50.0,
        corners=((30.0, 40.0), (50.0, 40.0), (50.0, 60.0), (30.0, 60.0)),
        rotation_degrees=(10.0, 20.0, rotation_z),
    )


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


def _render_canvas(canvas: BlueprintCanvas) -> QImage:
    image = QImage(canvas.size(), QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    canvas.render(image)
    return image


# ### Canvas profile tests ###
class CanvasPlacedObjectProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.canvases: list[BlueprintCanvas] = []

    def tearDown(self) -> None:
        for canvas in reversed(self.canvases):
            canvas.close()
            canvas.deleteLater()
        _qt_application.processEvents()

    def _build_canvas(self, level: LevelData | None = None) -> BlueprintCanvas:
        active_level = level or _build_level()
        canvas = BlueprintCanvas()
        canvas.resize(640, 520)
        canvas.set_level_data(
            vertex_data=active_level.vertex_data,
            rooms=active_level.rooms,
            image_path=None,
            doorways=active_level.doorways,
            windows=active_level.windows,
        )
        canvas.blueprint_image = QImage(100, 100, QImage.Format.Format_RGB32)
        canvas.blueprint_image.fill(Qt.GlobalColor.white)
        canvas.set_stair_context((), active_level)
        canvas.show()
        _qt_application.processEvents()
        self.canvases.append(canvas)
        return canvas

    def test_profile_paints_green_and_filters_to_current_level(self) -> None:
        canvas = self._build_canvas()
        canvas.set_placed_object_profiles(
            (_build_profile(), _build_profile("other-level", level_index=3))
        )

        rendered = _render_canvas(canvas)
        center = canvas._image_to_widget(40.0, 50.0).toPoint()
        pixel = rendered.pixelColor(center)

        self.assertGreater(pixel.green(), pixel.red())
        self.assertGreater(pixel.green(), pixel.blue())
        self.assertIsNone(
            canvas._find_placed_object_profile_at(
                canvas._image_to_widget(70.0, 70.0)
            )
        )

    def test_click_selects_profile_without_creating_wall_geometry(self) -> None:
        canvas = self._build_canvas()
        canvas.set_placed_object_profiles((_build_profile(),))
        canvas.set_selected_wall_surface_id("level:2/wall:1:2")
        requests: list[tuple[object, object]] = []
        canvas.placed_object_selection_requested.connect(
            lambda object_ids, active_id: requests.append((object_ids, active_id))
        )

        QTest.mouseClick(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=canvas._image_to_widget(40.0, 50.0).toPoint(),
        )

        self.assertEqual(canvas.get_selected_placed_object_ids(), ("chair",))
        self.assertEqual(requests, [(('chair',), "chair")])
        self.assertIsNone(canvas.get_selected_wall_surface_id())
        self.assertEqual(canvas.vertex_data.vertices, [])
        self.assertEqual(canvas.vertex_data.edges, [])

    def test_body_drag_previews_and_commits_once_on_release(self) -> None:
        canvas = self._build_canvas()
        original = _build_profile()
        canvas.set_placed_object_profiles((original,))
        commits: list[tuple[str, float, float, float]] = []
        canvas.placed_object_transform_committed.connect(
            lambda *values: commits.append(values)
        )
        start = canvas._image_to_widget(40.0, 50.0).toPoint()
        end = canvas._image_to_widget(55.0, 62.0).toPoint()

        QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=start)
        _send_drag_move(canvas, end)
        self.assertEqual(commits, [])
        preview = canvas.get_placed_object_profiles()[0]
        self.assertAlmostEqual(preview.anchor_x, 55.0, places=1)
        self.assertAlmostEqual(preview.anchor_y, 62.0, delta=0.2)
        QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=end)

        self.assertEqual(len(commits), 1)
        object_id, image_x, image_y, rotation_z = commits[0]
        self.assertEqual(object_id, "chair")
        self.assertAlmostEqual(image_x, 55.0, places=1)
        self.assertAlmostEqual(image_y, 62.0, delta=0.2)
        self.assertAlmostEqual(rotation_z, 0.0)

    def test_rotation_handle_changes_only_world_z_rotation(self) -> None:
        canvas = self._build_canvas()
        profile = _build_profile(rotation_z=25.0)
        canvas.set_placed_object_profiles((profile,))
        canvas.set_selected_placed_object_ids(("chair",), active_object_id="chair")
        commits: list[tuple[str, float, float, float]] = []
        canvas.placed_object_transform_committed.connect(
            lambda *values: commits.append(values)
        )
        geometry = canvas._get_placed_object_widget_geometry(profile)
        start = geometry.rotation_handle.toPoint()
        handle_delta = geometry.rotation_handle - geometry.anchor
        clockwise_end = QPointF(
            geometry.anchor.x() - handle_delta.y(),
            geometry.anchor.y() + handle_delta.x(),
        ).toPoint()

        QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=start)
        _send_drag_move(canvas, clockwise_end)
        QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=clockwise_end)

        self.assertEqual(len(commits), 1)
        self.assertAlmostEqual(commits[0][1], profile.anchor_x)
        self.assertAlmostEqual(commits[0][2], profile.anchor_y)
        self.assertAlmostEqual(commits[0][3], -65.0, delta=0.3)
        rotated = canvas.get_placed_object_profiles()[0]
        self.assertEqual(rotated.rotation_degrees[:2], (10.0, 20.0))

    def test_escape_cancels_a_local_move_preview(self) -> None:
        canvas = self._build_canvas()
        original = _build_profile()
        canvas.set_placed_object_profiles((original,))
        commits: list[tuple[str, float, float, float]] = []
        canvas.placed_object_transform_committed.connect(
            lambda *values: commits.append(values)
        )
        start = canvas._image_to_widget(40.0, 50.0).toPoint()
        end = canvas._image_to_widget(58.0, 65.0).toPoint()

        QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=start)
        _send_drag_move(canvas, end)
        self.assertNotEqual(canvas.get_placed_object_profiles()[0], original)
        QTest.keyClick(canvas, Qt.Key.Key_Escape)

        self.assertEqual(canvas.get_placed_object_profiles(), (original,))
        self.assertEqual(commits, [])

    def test_thin_profile_has_screen_space_selection_tolerance(self) -> None:
        canvas = self._build_canvas()
        thin_profile = CanvasPlacedObjectProfile(
            object_id="thin-table",
            level_index=2,
            anchor_x=40.0,
            anchor_y=50.0,
            corners=((30.0, 50.0), (50.0, 50.0), (50.0, 50.0), (30.0, 50.0)),
            rotation_degrees=(0.0, 0.0, 0.0),
        )
        canvas.set_placed_object_profiles((thin_profile,))
        near_edge = canvas._image_to_widget(40.0, 50.0) + QPointF(0.0, 5.0)

        QTest.mouseClick(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=near_edge.toPoint(),
        )

        self.assertEqual(
            canvas.get_selected_placed_object_ids(),
            ("thin-table",),
        )

    def test_async_profile_refresh_does_not_cancel_an_active_drag(self) -> None:
        canvas = self._build_canvas()
        original = _build_profile()
        other = CanvasPlacedObjectProfile(
            object_id="table",
            level_index=2,
            anchor_x=75.0,
            anchor_y=75.0,
            corners=((70.0, 70.0), (80.0, 70.0), (80.0, 80.0), (70.0, 80.0)),
            rotation_degrees=(0.0, 0.0, 0.0),
        )
        canvas.set_placed_object_profiles((original, other))
        start = canvas._image_to_widget(40.0, 50.0).toPoint()
        end = canvas._image_to_widget(58.0, 65.0).toPoint()

        QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=start)
        _send_drag_move(canvas, end)
        local_preview = canvas.get_placed_object_profiles()[0]
        refreshed_other = CanvasPlacedObjectProfile(
            object_id="table",
            level_index=2,
            anchor_x=70.0,
            anchor_y=75.0,
            corners=((65.0, 70.0), (75.0, 70.0), (75.0, 80.0), (65.0, 80.0)),
            rotation_degrees=(0.0, 0.0, 0.0),
        )

        canvas.set_placed_object_profiles((original, refreshed_other))

        self.assertEqual(canvas.get_placed_object_profiles()[0], local_preview)
        self.assertEqual(canvas.get_placed_object_profiles()[1], refreshed_other)
        self.assertIsNotNone(canvas._placed_object_edit_drag)
        QTest.keyClick(canvas, Qt.Key.Key_Escape)
        self.assertEqual(canvas.get_placed_object_profiles()[0], original)

    def test_async_profile_refresh_wins_when_pointer_does_not_drag(self) -> None:
        canvas = self._build_canvas()
        original = _build_profile()
        refreshed = CanvasPlacedObjectProfile(
            object_id="chair",
            level_index=2,
            anchor_x=42.0,
            anchor_y=54.0,
            corners=((32.0, 44.0), (52.0, 44.0), (52.0, 64.0), (32.0, 64.0)),
            rotation_degrees=(10.0, 20.0, 0.0),
        )
        canvas.set_placed_object_profiles((original,))
        start = canvas._image_to_widget(40.0, 50.0).toPoint()

        QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=start)
        canvas.set_placed_object_profiles((refreshed,))
        QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=start)

        self.assertEqual(canvas.get_placed_object_profiles(), (refreshed,))


if __name__ == "__main__":
    unittest.main()
