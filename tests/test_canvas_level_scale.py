# ### Environment setup ###
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PIL import Image
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QFocusEvent, QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import housemaker.blueprint_canvas as blueprint_canvas_module
from housemaker.app_settings import ApplicationSettingsStore
from housemaker.blueprint_canvas import BlueprintCanvas
from housemaker.main import (
    CANVAS_LEVEL_SCALE_SLIDER_FACTOR,
    BlueprintWorkspace,
)
from housemaker.models import (
    DEFAULT_CANVAS_LEVEL_SCALE,
    LevelData,
    VertexData,
)
from housemaker.project_io import load_project, save_project

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _write_blueprint(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (120, 80), color).save(path)


def _build_level(index: int, image_path: Path) -> LevelData:
    vertices = VertexData()
    first = vertices.add_vertex(10.0, 10.0)
    second = vertices.add_vertex(110.0, 70.0)
    vertices.add_edge(first.id, second.id)
    return LevelData(
        index=index,
        name="Ground" if index == 2 else f"Level {index}",
        image_path=str(image_path),
        vertex_data=vertices,
    )


# ### Canvas scale rendering tests ###
class CanvasLevelScaleRenderingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.image_path = Path(self.temporary_directory.name) / "plan.png"
        _write_blueprint(self.image_path, (30, 60, 90))
        self.canvas = BlueprintCanvas()
        self.canvas.resize(640, 480)
        self.canvas.load_blueprint(str(self.image_path))

    def tearDown(self) -> None:
        self.canvas.close()
        self.canvas.deleteLater()
        self.temporary_directory.cleanup()
        _qt_application.processEvents()

    def test_canvas_scale_changes_display_mapping_without_changing_geometry(
        self,
    ) -> None:
        base_rect = self.canvas._base_image_display_rect()
        self.assertTrue(self.canvas.set_canvas_level_scale(1.5))
        scaled_rect = self.canvas._base_image_display_rect()

        self.assertAlmostEqual(scaled_rect.width(), base_rect.width() * 1.5)
        self.assertAlmostEqual(scaled_rect.height(), base_rect.height() * 1.5)
        self.assertEqual(scaled_rect.center(), base_rect.center())
        image_point = self.canvas._widget_to_image(
            self.canvas._image_to_widget(30.0, 40.0)
        )
        self.assertIsNotNone(image_point)
        assert image_point is not None
        self.assertAlmostEqual(image_point.x(), 30.0)
        self.assertAlmostEqual(image_point.y(), 40.0)

    def test_comparison_uses_its_own_canvas_scale(self) -> None:
        comparison = _build_level(1, self.image_path)
        comparison.canvas_level_scale = 0.5

        self.assertTrue(self.canvas.set_level_comparison_overlay(comparison))
        overlay = self.canvas.get_level_comparison_overlay()
        self.assertIsNotNone(overlay)
        assert overlay is not None
        self.assertEqual(overlay.level_index, 1)
        self.assertAlmostEqual(overlay.canvas_level_scale, 0.5)
        comparison_rect = self.canvas._comparison_image_display_rect(overlay)
        current_rect = self.canvas._image_display_rect()
        self.assertAlmostEqual(
            comparison_rect.width(),
            current_rect.width() * 0.5,
        )

        self.assertTrue(self.canvas.clear_level_comparison_overlay())
        self.assertIsNone(self.canvas.get_level_comparison_overlay())

    def test_scaled_canvas_can_pan_without_additional_wheel_zoom(self) -> None:
        self.canvas.set_canvas_level_scale(2.0)

        self.canvas._start_panning(self.canvas.rect().center())

        self.assertTrue(self.canvas.is_panning)

    def test_comparison_image_decode_is_cached_by_file_revision(self) -> None:
        comparison_path = Path(self.temporary_directory.name) / "adjacent.png"
        _write_blueprint(comparison_path, (90, 60, 30))
        comparison = _build_level(1, comparison_path)

        with patch(
            "housemaker.blueprint_canvas._load_qimage_from_path",
            wraps=blueprint_canvas_module._load_qimage_from_path,
        ) as load_image:
            self.assertTrue(
                self.canvas.set_level_comparison_overlay(comparison)
            )
            self.canvas.clear_level_comparison_overlay()
            self.assertTrue(
                self.canvas.set_level_comparison_overlay(comparison)
            )

        load_image.assert_called_once_with(str(comparison_path))


# ### Workspace interaction tests ###
class CanvasLevelScaleWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        temporary_path = Path(self.temporary_directory.name)
        self.image_paths = [
            temporary_path / f"level-{index}.png" for index in range(4)
        ]
        for index, image_path in enumerate(self.image_paths):
            _write_blueprint(image_path, (30 + index * 20, 40, 50))
        self.levels = [
            _build_level(index, self.image_paths[index])
            for index in range(4)
        ]
        self.levels[1].canvas_level_scale = 0.75
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(
                temporary_path / "settings.json"
            )
        )
        self.workspace._apply_project_state(self.levels, 2)

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        self.workspace.deleteLater()
        self.temporary_directory.cleanup()
        _qt_application.processEvents()

    def test_slider_drag_is_live_compares_below_and_creates_one_undo_step(
        self,
    ) -> None:
        original_3d_scale = self.workspace.current_level.scale
        self.workspace._handle_canvas_transform_drag_started()
        overlay = self.workspace.canvas.get_level_comparison_overlay()
        self.assertIsNotNone(overlay)
        assert overlay is not None
        self.assertEqual(overlay.level_index, 1)
        self.assertAlmostEqual(overlay.canvas_level_scale, 0.75)

        self.workspace.canvas_level_scale_slider.setValue(
            round(1.5 * CANVAS_LEVEL_SCALE_SLIDER_FACTOR)
        )
        self.assertAlmostEqual(
            self.workspace.current_level.canvas_level_scale,
            1.5,
        )
        self.assertAlmostEqual(self.workspace.canvas.canvas_level_scale, 1.5)
        self.assertAlmostEqual(
            self.workspace.current_level.scale,
            original_3d_scale,
        )
        self.assertEqual(self.workspace._canvas_undo_stack, [])

        self.workspace._handle_canvas_transform_drag_finished()
        self.assertIsNone(
            self.workspace.canvas.get_level_comparison_overlay()
        )
        self.assertEqual(len(self.workspace._canvas_undo_stack), 1)

        self.workspace._handle_canvas_undo_requested()
        self.assertAlmostEqual(
            self.workspace.current_level.canvas_level_scale,
            DEFAULT_CANVAS_LEVEL_SCALE,
        )
        self.assertAlmostEqual(
            self.workspace.canvas.canvas_level_scale,
            DEFAULT_CANVAS_LEVEL_SCALE,
        )

    def test_underground_level_compares_with_level_above(self) -> None:
        self.workspace._handle_level_selection_changed(1)
        self.workspace._handle_canvas_transform_drag_started()

        overlay = self.workspace.canvas.get_level_comparison_overlay()
        self.assertIsNotNone(overlay)
        assert overlay is not None
        self.assertEqual(overlay.level_index, 2)

        self.workspace._handle_canvas_transform_drag_finished()

    def test_selecting_another_level_preserves_zoom_and_view_position(
        self,
    ) -> None:
        self.workspace.canvas.zoom_scale = 2.35
        self.workspace.canvas.view_offset = QPointF(47.0, -31.0)

        self.workspace._handle_level_selection_changed(1)

        self.assertAlmostEqual(self.workspace.canvas.zoom_scale, 2.35)
        self.assertEqual(
            self.workspace.canvas.view_offset,
            QPointF(47.0, -31.0),
        )

        self.workspace._handle_level_selection_changed(3)

        self.assertAlmostEqual(self.workspace.canvas.zoom_scale, 2.35)
        self.assertEqual(
            self.workspace.canvas.view_offset,
            QPointF(47.0, -31.0),
        )

    def test_r_and_f_hold_exact_upper_and_lower_level_previews(self) -> None:
        QTest.keyPress(self.workspace.canvas, Qt.Key.Key_R)
        upper_overlay = self.workspace.canvas.get_level_comparison_overlay()
        self.assertIsNotNone(upper_overlay)
        assert upper_overlay is not None
        self.assertEqual(upper_overlay.level_index, 3)

        QTest.keyRelease(self.workspace.canvas, Qt.Key.Key_R)
        self.assertIsNone(
            self.workspace.canvas.get_level_comparison_overlay()
        )

        QTest.keyPress(self.workspace.canvas, Qt.Key.Key_F)
        lower_overlay = self.workspace.canvas.get_level_comparison_overlay()
        self.assertIsNotNone(lower_overlay)
        assert lower_overlay is not None
        self.assertEqual(lower_overlay.level_index, 1)

        QTest.keyRelease(self.workspace.canvas, Qt.Key.Key_F)
        self.assertIsNone(
            self.workspace.canvas.get_level_comparison_overlay()
        )

    def test_keyboard_preview_ignores_modifiers_and_auto_repeat(self) -> None:
        QTest.keyPress(
            self.workspace.canvas,
            Qt.Key.Key_R,
            Qt.KeyboardModifier.ControlModifier,
        )
        self.assertIsNone(
            self.workspace.canvas.get_level_comparison_overlay()
        )
        QTest.keyRelease(
            self.workspace.canvas,
            Qt.Key.Key_R,
            Qt.KeyboardModifier.ControlModifier,
        )

        repeat_event = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_R,
            Qt.KeyboardModifier.NoModifier,
            "r",
            True,
            2,
        )
        QApplication.sendEvent(self.workspace.canvas, repeat_event)
        self.assertIsNone(
            self.workspace.canvas.get_level_comparison_overlay()
        )

    def test_transform_preview_wins_and_restores_held_keyboard_preview(
        self,
    ) -> None:
        QTest.keyPress(self.workspace.canvas, Qt.Key.Key_R)
        self.workspace._handle_canvas_transform_drag_started()
        transform_overlay = self.workspace.canvas.get_level_comparison_overlay()
        self.assertIsNotNone(transform_overlay)
        assert transform_overlay is not None
        self.assertEqual(transform_overlay.level_index, 1)

        self.workspace._handle_canvas_transform_drag_finished()
        restored_overlay = self.workspace.canvas.get_level_comparison_overlay()
        self.assertIsNotNone(restored_overlay)
        assert restored_overlay is not None
        self.assertEqual(restored_overlay.level_index, 3)

        QTest.keyRelease(self.workspace.canvas, Qt.Key.Key_R)
        self.assertIsNone(
            self.workspace.canvas.get_level_comparison_overlay()
        )

    def test_keyboard_release_cannot_clear_active_transform_preview(self) -> None:
        QTest.keyPress(self.workspace.canvas, Qt.Key.Key_R)
        self.workspace._handle_canvas_transform_drag_started()

        QTest.keyRelease(self.workspace.canvas, Qt.Key.Key_R)
        overlay = self.workspace.canvas.get_level_comparison_overlay()
        self.assertIsNotNone(overlay)
        assert overlay is not None
        self.assertEqual(overlay.level_index, 1)

        self.workspace._handle_canvas_transform_drag_finished()
        self.assertIsNone(
            self.workspace.canvas.get_level_comparison_overlay()
        )

    def test_level_transform_button_restores_held_keyboard_preview(self) -> None:
        QTest.keyPress(self.workspace.canvas, Qt.Key.Key_R)
        self.workspace._handle_level_transform_button_pressed(
            self.workspace.level_scale_slider,
            1,
        )
        transform_overlay = self.workspace.canvas.get_level_comparison_overlay()
        self.assertIsNotNone(transform_overlay)
        assert transform_overlay is not None
        self.assertEqual(transform_overlay.level_index, 1)

        self.workspace._handle_level_transform_button_released()
        restored_overlay = self.workspace.canvas.get_level_comparison_overlay()
        self.assertIsNotNone(restored_overlay)
        assert restored_overlay is not None
        self.assertEqual(restored_overlay.level_index, 3)

        QTest.keyRelease(self.workspace.canvas, Qt.Key.Key_R)

    def test_focus_loss_and_level_switch_clear_keyboard_preview(self) -> None:
        QTest.keyPress(self.workspace.canvas, Qt.Key.Key_R)
        QApplication.sendEvent(
            self.workspace.canvas,
            QFocusEvent(QEvent.Type.FocusOut),
        )
        self.assertIsNone(
            self.workspace.canvas.get_level_comparison_overlay()
        )

        QTest.keyPress(self.workspace.canvas, Qt.Key.Key_F)
        self.workspace._handle_level_selection_changed(1)
        self.assertIsNone(
            self.workspace.canvas.get_level_comparison_overlay()
        )

    def test_keyboard_preview_handles_level_bounds_and_missing_images(
        self,
    ) -> None:
        self.workspace._handle_level_selection_changed(3)
        QTest.keyPress(self.workspace.canvas, Qt.Key.Key_R)
        self.assertIsNone(
            self.workspace.canvas.get_level_comparison_overlay()
        )
        QTest.keyRelease(self.workspace.canvas, Qt.Key.Key_R)

        self.workspace._handle_level_selection_changed(0)
        QTest.keyPress(self.workspace.canvas, Qt.Key.Key_F)
        self.assertIsNone(
            self.workspace.canvas.get_level_comparison_overlay()
        )
        QTest.keyRelease(self.workspace.canvas, Qt.Key.Key_F)

        self.workspace._handle_level_selection_changed(2)
        self.workspace.levels[3].image_path = None
        QTest.keyPress(self.workspace.canvas, Qt.Key.Key_R)
        self.assertIsNone(
            self.workspace.canvas.get_level_comparison_overlay()
        )
        QTest.keyRelease(self.workspace.canvas, Qt.Key.Key_R)


# ### Project compatibility tests ###
class CanvasLevelScaleProjectTests(unittest.TestCase):
    def test_canvas_scale_persists_and_legacy_projects_default_to_one(self) -> None:
        levels = [_build_level(2, Path("missing.png"))]
        levels[0].canvas_level_scale = 1.37

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "project.json"
            save_project(project_path, 0, levels)
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            self.assertAlmostEqual(
                payload["levels"][0]["canvas_level_scale"],
                1.37,
            )
            self.assertAlmostEqual(
                load_project(project_path).levels[2].canvas_level_scale,
                1.37,
            )

            payload["levels"][0].pop("canvas_level_scale")
            project_path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertAlmostEqual(
                load_project(project_path).levels[2].canvas_level_scale,
                DEFAULT_CANVAS_LEVEL_SCALE,
            )


if __name__ == "__main__":
    unittest.main()
