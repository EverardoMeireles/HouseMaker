# ### Environment setup ###
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.blueprint_canvas import (
    DIRECTIONAL_LIGHT_MARKER_COLOR,
    BlueprintCanvas,
    CanvasDirectionalLightProfile,
)
from housemaker.models import LevelData

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_canvas() -> BlueprintCanvas:
    canvas = BlueprintCanvas()
    canvas.resize(640, 480)
    canvas.blueprint_image = QImage(
        200,
        100,
        QImage.Format.Format_RGB32,
    )
    canvas.blueprint_image.fill(Qt.GlobalColor.white)
    canvas.level_context = LevelData(
        index=2,
        name="Ground",
        image_size_pixels=(200.0, 100.0),
    )
    canvas.show()
    _qt_application.processEvents()
    return canvas


def _render_canvas(canvas: BlueprintCanvas) -> QImage:
    image = QImage(canvas.size(), QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    canvas.render(image)
    return image


# ### Directional-light Canvas tests ###
class CanvasDirectionalLightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas = _build_canvas()

    def tearDown(self) -> None:
        self.canvas.close()
        self.canvas.deleteLater()
        _qt_application.processEvents()

    def test_one_shot_placement_emits_image_coordinates_without_editing_walls(
        self,
    ) -> None:
        requested: list[tuple[float, float]] = []
        changed: list[bool] = []
        self.canvas.directional_light_placement_requested.connect(
            lambda image_x, image_y: requested.append((image_x, image_y))
        )
        self.canvas.directional_light_placement_changed.connect(changed.append)
        target = self.canvas._image_to_widget(75.0, 40.0).toPoint()

        self.assertTrue(self.canvas.start_directional_light_placement())
        QTest.mouseMove(self.canvas, target)
        self.assertIsNotNone(self.canvas._directional_light_hover_image_point)
        QTest.mouseClick(self.canvas, Qt.MouseButton.LeftButton, pos=target)

        self.assertFalse(self.canvas.is_directional_light_placement_active())
        self.assertEqual(changed, [True, False])
        self.assertEqual(len(requested), 1)
        self.assertAlmostEqual(requested[0][0], 75.0, delta=0.5)
        self.assertAlmostEqual(requested[0][1], 40.0, delta=0.5)
        self.assertEqual(self.canvas.vertex_data.vertices, [])
        self.assertEqual(self.canvas.vertex_data.edges, [])

    def test_right_click_and_escape_cancel_without_requesting_a_light(self) -> None:
        requested: list[tuple[float, float]] = []
        cancelled: list[bool] = []
        self.canvas.directional_light_placement_requested.connect(
            lambda image_x, image_y: requested.append((image_x, image_y))
        )
        self.canvas.directional_light_placement_cancelled.connect(
            lambda: cancelled.append(True)
        )
        target = self.canvas._image_to_widget(50.0, 50.0).toPoint()

        self.canvas.start_directional_light_placement()
        QTest.mouseClick(self.canvas, Qt.MouseButton.RightButton, pos=target)
        self.canvas.start_directional_light_placement()
        QTest.keyClick(self.canvas, Qt.Key.Key_Escape)

        self.assertEqual(requested, [])
        self.assertEqual(cancelled, [True, True])
        self.assertFalse(self.canvas.is_directional_light_placement_active())

    def test_profile_click_selects_light_without_creating_a_vertex(self) -> None:
        profile = CanvasDirectionalLightProfile(
            light_id="directional-light-1",
            image_x=80.0,
            image_y=35.0,
        )
        selections: list[object] = []
        self.canvas.set_directional_light_profiles((profile,))
        self.canvas.directional_light_selection_requested.connect(selections.append)

        QTest.mouseClick(
            self.canvas,
            Qt.MouseButton.LeftButton,
            pos=self.canvas._image_to_widget(80.0, 35.0).toPoint(),
        )

        self.assertEqual(selections, ["directional-light-1"])
        self.assertEqual(
            self.canvas.get_selected_directional_light_id(),
            "directional-light-1",
        )
        self.assertEqual(self.canvas.vertex_data.vertices, [])

    def test_marker_reprojects_with_zoom_and_remains_hit_testable(self) -> None:
        profile = CanvasDirectionalLightProfile(
            light_id="directional-light-1",
            image_x=35.0,
            image_y=25.0,
        )
        self.canvas.set_directional_light_profiles((profile,))
        initial_position = self.canvas._image_to_widget(
            profile.image_x,
            profile.image_y,
        )

        self.canvas.zoom_scale = 2.0
        self.canvas.view_offset = QPointF(55.0, 30.0)
        zoomed_position = self.canvas._image_to_widget(
            profile.image_x,
            profile.image_y,
        )

        self.assertNotEqual(initial_position, zoomed_position)
        self.assertEqual(
            self.canvas._find_directional_light_profile_at(zoomed_position),
            profile,
        )

    def test_marker_and_pending_hover_are_painted_above_the_plan(self) -> None:
        profile = CanvasDirectionalLightProfile(
            light_id="directional-light-1",
            image_x=100.0,
            image_y=50.0,
        )
        self.canvas.set_directional_light_profiles((profile,))
        marker_position = self.canvas._image_to_widget(100.0, 50.0).toPoint()

        rendered = _render_canvas(self.canvas)
        self.assertEqual(
            rendered.pixelColor(marker_position).name(),
            DIRECTIONAL_LIGHT_MARKER_COLOR.name(),
        )

        self.canvas.start_directional_light_placement()
        hover_position = self.canvas._image_to_widget(130.0, 65.0).toPoint()
        QTest.mouseMove(self.canvas, hover_position)
        rendered = _render_canvas(self.canvas)
        hover_color = QColor(rendered.pixelColor(hover_position))
        self.assertGreater(hover_color.red(), hover_color.blue())
        self.assertGreater(hover_color.green(), hover_color.blue())

    def test_another_canvas_tool_cancels_light_placement(self) -> None:
        cancelled: list[bool] = []
        self.canvas.directional_light_placement_cancelled.connect(
            lambda: cancelled.append(True)
        )
        self.canvas.start_directional_light_placement()

        self.assertTrue(self.canvas.start_open_space_placement())

        self.assertFalse(self.canvas.is_directional_light_placement_active())
        self.assertTrue(self.canvas.is_open_space_placement_active())
        self.assertEqual(cancelled, [True])


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
