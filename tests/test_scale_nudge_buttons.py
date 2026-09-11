# ### Environment setup ###
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PIL import Image
from PySide6.QtWidgets import QApplication, QPushButton

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.main import (
    CANVAS_LEVEL_SCALE_SLIDER_FACTOR,
    LEVEL_OFFSET_SLIDER_FACTOR,
    LEVEL_SCALE_SLIDER_FACTOR,
    BlueprintWorkspace,
)
from housemaker.models import LevelData, VertexData

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _write_blueprint(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (120, 80), color).save(path)


def _build_level(index: int, image_path: Path) -> LevelData:
    vertices = VertexData()
    corners = tuple(
        vertices.add_vertex(*position)
        for position in (
            (10.0, 10.0),
            (110.0, 10.0),
            (110.0, 70.0),
            (10.0, 70.0),
        )
    )
    for start, end in zip(corners, (*corners[1:], corners[0])):
        vertices.add_edge(start.id, end.id)
    return LevelData(
        index=index,
        name="Ground" if index == 2 else f"Level {index}",
        image_path=str(image_path),
        vertex_data=vertices,
    )


# ### Transform nudge integration tests ###
class TransformNudgeButtonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        temporary_path = Path(self.temporary_directory.name)
        image_paths = [temporary_path / f"level-{index}.png" for index in range(3)]
        for index, image_path in enumerate(image_paths):
            _write_blueprint(image_path, (35 + index * 20, 60, 85))
        self.levels = [_build_level(index, image_paths[index]) for index in range(3)]
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(
                temporary_path / "settings.json"
            )
        )
        self.workspace._apply_project_state(self.levels, 2)
        self.workspace._set_mesh_edit_update_delay_seconds(60.0)
        self.workspace._canvas_undo_stack.clear()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        self.workspace.deleteLater()
        _qt_application.processEvents()
        self.temporary_directory.cleanup()

    def test_all_transform_rows_expose_non_repeating_nudge_buttons(self) -> None:
        button_pairs = (
            (
                "level_scale_decrease_button",
                "level_scale_increase_button",
            ),
            (
                "level_x_offset_decrease_button",
                "level_x_offset_increase_button",
            ),
            (
                "level_y_offset_decrease_button",
                "level_y_offset_increase_button",
            ),
            (
                "canvas_level_scale_decrease_button",
                "canvas_level_scale_increase_button",
            ),
            (
                "canvas_x_offset_decrease_button",
                "canvas_x_offset_increase_button",
            ),
            (
                "canvas_y_offset_decrease_button",
                "canvas_y_offset_increase_button",
            ),
        )

        buttons = tuple(
            getattr(self.workspace, button_name)
            for pair in button_pairs
            for button_name in pair
        )

        self.assertTrue(all(isinstance(button, QPushButton) for button in buttons))
        self.assertEqual(
            tuple(button.text() for button in buttons),
            ("\N{MINUS SIGN}", "+") * len(button_pairs),
        )
        self.assertTrue(all(not button.autoRepeat() for button in buttons))

    def test_level_scale_buttons_stage_exactly_one_slider_step(self) -> None:
        level = self.workspace.current_level
        original_scale = float(level.scale)
        expected_step = 1.0 / LEVEL_SCALE_SLIDER_FACTOR

        self.workspace.level_scale_increase_button.click()

        self.assertAlmostEqual(level.scale, original_scale)
        pending = self.workspace._pending_level_transform
        self.assertIsNotNone(pending)
        assert pending is not None
        self.assertAlmostEqual(pending.scale, original_scale + expected_step)
        self.assertTrue(self.workspace._level_transform_mesh_update_timer.isActive())
        self.workspace._commit_pending_level_transform_update()
        self.assertAlmostEqual(level.scale, original_scale + expected_step)

        self.workspace.level_scale_decrease_button.click()

        self.assertAlmostEqual(level.scale, original_scale + expected_step)
        pending = self.workspace._pending_level_transform
        self.assertIsNotNone(pending)
        assert pending is not None
        self.assertAlmostEqual(pending.scale, original_scale)
        self.assertTrue(self.workspace._level_transform_mesh_update_timer.isActive())
        self.workspace._commit_pending_level_transform_update()
        self.assertAlmostEqual(level.scale, original_scale)

    def test_canvas_level_scale_buttons_apply_one_slider_step(self) -> None:
        level = self.workspace.current_level
        original_scale = float(level.canvas_level_scale)
        expected_step = 1.0 / CANVAS_LEVEL_SCALE_SLIDER_FACTOR

        self.workspace.canvas_level_scale_increase_button.click()

        self.assertAlmostEqual(
            level.canvas_level_scale,
            original_scale + expected_step,
        )
        self.assertAlmostEqual(
            self.workspace.canvas.canvas_level_scale,
            original_scale + expected_step,
        )

        self.workspace.canvas_level_scale_decrease_button.click()

        self.assertAlmostEqual(level.canvas_level_scale, original_scale)
        self.assertAlmostEqual(
            self.workspace.canvas.canvas_level_scale,
            original_scale,
        )

    def test_level_offset_buttons_stage_exactly_one_slider_step(self) -> None:
        level = self.workspace.current_level
        expected_step = 1.0 / LEVEL_OFFSET_SLIDER_FACTOR

        self.workspace.level_x_offset_increase_button.click()
        pending = self.workspace._pending_level_transform
        self.assertIsNotNone(pending)
        assert pending is not None
        self.assertAlmostEqual(pending.offset_x_meters, expected_step)
        self.assertAlmostEqual(pending.offset_y_meters, 0.0)

        self.workspace.level_y_offset_decrease_button.click()
        pending = self.workspace._pending_level_transform
        self.assertIsNotNone(pending)
        assert pending is not None
        self.assertAlmostEqual(pending.offset_x_meters, expected_step)
        self.assertAlmostEqual(pending.offset_y_meters, -expected_step)
        self.assertAlmostEqual(level.offset_x_meters, 0.0)
        self.assertAlmostEqual(level.offset_y_meters, 0.0)

    def test_holding_any_nudge_button_shows_preview_and_steps_only_once(
        self,
    ) -> None:
        def pending_level_value(attribute_name: str) -> float:
            pending = self.workspace._pending_level_transform
            if pending is not None:
                return float(getattr(pending, attribute_name))
            return float(getattr(self.workspace.current_level, attribute_name))

        level = self.workspace.current_level
        button_cases = (
            (
                "Level scale decrease",
                self.workspace.level_scale_decrease_button,
                lambda: pending_level_value("scale"),
                -1.0 / LEVEL_SCALE_SLIDER_FACTOR,
            ),
            (
                "Level scale increase",
                self.workspace.level_scale_increase_button,
                lambda: pending_level_value("scale"),
                1.0 / LEVEL_SCALE_SLIDER_FACTOR,
            ),
            (
                "Level X offset decrease",
                self.workspace.level_x_offset_decrease_button,
                lambda: pending_level_value("offset_x_meters"),
                -1.0 / LEVEL_OFFSET_SLIDER_FACTOR,
            ),
            (
                "Level X offset increase",
                self.workspace.level_x_offset_increase_button,
                lambda: pending_level_value("offset_x_meters"),
                1.0 / LEVEL_OFFSET_SLIDER_FACTOR,
            ),
            (
                "Level Y offset decrease",
                self.workspace.level_y_offset_decrease_button,
                lambda: pending_level_value("offset_y_meters"),
                -1.0 / LEVEL_OFFSET_SLIDER_FACTOR,
            ),
            (
                "Level Y offset increase",
                self.workspace.level_y_offset_increase_button,
                lambda: pending_level_value("offset_y_meters"),
                1.0 / LEVEL_OFFSET_SLIDER_FACTOR,
            ),
            (
                "Canvas level scale decrease",
                self.workspace.canvas_level_scale_decrease_button,
                lambda: float(level.canvas_level_scale),
                -1.0 / CANVAS_LEVEL_SCALE_SLIDER_FACTOR,
            ),
            (
                "Canvas level scale increase",
                self.workspace.canvas_level_scale_increase_button,
                lambda: float(level.canvas_level_scale),
                1.0 / CANVAS_LEVEL_SCALE_SLIDER_FACTOR,
            ),
            (
                "Canvas X offset decrease",
                self.workspace.canvas_x_offset_decrease_button,
                lambda: float(level.canvas_offset_x_pixels),
                -1.0,
            ),
            (
                "Canvas X offset increase",
                self.workspace.canvas_x_offset_increase_button,
                lambda: float(level.canvas_offset_x_pixels),
                1.0,
            ),
            (
                "Canvas Y offset decrease",
                self.workspace.canvas_y_offset_decrease_button,
                lambda: float(level.canvas_offset_y_pixels),
                -1.0,
            ),
            (
                "Canvas Y offset increase",
                self.workspace.canvas_y_offset_increase_button,
                lambda: float(level.canvas_offset_y_pixels),
                1.0,
            ),
        )

        for label, button, read_value, expected_delta in button_cases:
            with self.subTest(button=label):
                value_before_press = read_value()
                button.setDown(True)
                button.pressed.emit()

                overlay = self.workspace.canvas.get_level_comparison_overlay()
                self.assertIsNotNone(overlay)
                assert overlay is not None
                self.assertEqual(overlay.level_index, 1)
                value_while_held = read_value()
                self.assertAlmostEqual(
                    value_while_held,
                    value_before_press + expected_delta,
                )
                _qt_application.processEvents()
                self.assertAlmostEqual(read_value(), value_while_held)

                button.setDown(False)
                button.released.emit()

                self.assertIsNone(self.workspace.canvas.get_level_comparison_overlay())


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
