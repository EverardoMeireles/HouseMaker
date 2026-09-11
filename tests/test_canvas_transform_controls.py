# ### Environment setup ###
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PIL import Image
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt
from PySide6.QtWidgets import QApplication, QGroupBox, QLabel, QSlider

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.blueprint_canvas import BlueprintCanvas
from housemaker.main import (
    CANVAS_OFFSET_SLIDER_FACTOR,
    LEVEL_OFFSET_SLIDER_FACTOR,
    BlueprintWorkspace,
)
from housemaker.models import DEFAULT_CANVAS_LEVEL_SCALE, LevelData, VertexData
from housemaker.project_io import load_project, save_project

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _write_blueprint(path: Path) -> None:
    Image.new("RGB", (160, 100), (40, 70, 100)).save(path)


def _build_level(
    image_path: Path,
    *,
    canvas_offset_x_pixels: float = 0.0,
    canvas_offset_y_pixels: float = 0.0,
) -> LevelData:
    vertices = VertexData()
    first = vertices.add_vertex(20.0, 25.0)
    second = vertices.add_vertex(130.0, 75.0)
    vertices.add_edge(first.id, second.id)
    return LevelData(
        index=2,
        name="Ground",
        image_path=str(image_path),
        vertex_data=vertices,
        canvas_offset_x_pixels=canvas_offset_x_pixels,
        canvas_offset_y_pixels=canvas_offset_y_pixels,
    )


def _map_image_point_to_rect(
    rect: QRectF,
    image_size: tuple[int, int],
    image_point: tuple[float, float],
) -> QPointF:
    """Map one blueprint-space reference point through a display rectangle."""

    image_width, image_height = image_size
    image_x, image_y = image_point
    return QPointF(
        rect.left() + image_x / float(image_width) * rect.width(),
        rect.top() + image_y / float(image_height) * rect.height(),
    )


def _nearest_group_box(widget: object) -> QGroupBox | None:
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QGroupBox):
            return parent
        parent = parent.parentWidget()
    return None


def _assert_slider_column(
    test_case: unittest.TestCase,
    group: QGroupBox,
    sliders: tuple[QSlider, QSlider, QSlider],
) -> None:
    positions = [slider.mapTo(group, QPoint()) for slider in sliders]
    test_case.assertEqual(
        [position.y() for position in positions],
        sorted(position.y() for position in positions),
    )
    test_case.assertLessEqual(
        max(position.x() for position in positions)
        - min(position.x() for position in positions),
        1,
    )
    for slider in sliders:
        test_case.assertIs(_nearest_group_box(slider), group)
        test_case.assertEqual(
            slider.orientation(),
            Qt.Orientation.Horizontal,
        )


# ### Canvas rendering tests ###
class CanvasOffsetRenderingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.image_path = Path(self.temporary_directory.name) / "plan.png"
        _write_blueprint(self.image_path)
        self.level = _build_level(self.image_path)
        self.canvas = BlueprintCanvas()
        self.canvas.resize(720, 480)
        self.canvas.load_blueprint(
            str(self.image_path),
            vertex_data=self.level.vertex_data,
        )

    def tearDown(self) -> None:
        self.canvas.close()
        self.canvas.deleteLater()
        self.temporary_directory.cleanup()
        _qt_application.processEvents()

    def test_canvas_offsets_translate_only_the_plan_display(self) -> None:
        geometry_before = self.canvas.vertex_data.to_dict()
        rect_before = self.canvas._base_image_display_rect()
        point_before = self.canvas._image_to_widget(35.0, 45.0)

        self.assertTrue(self.canvas.set_canvas_level_offsets(42.5, -18.0))

        rect_after = self.canvas._base_image_display_rect()
        point_after = self.canvas._image_to_widget(35.0, 45.0)
        self.assertAlmostEqual(rect_after.width(), rect_before.width())
        self.assertAlmostEqual(rect_after.height(), rect_before.height())
        self.assertAlmostEqual(
            rect_after.center().x() - rect_before.center().x(),
            42.5,
        )
        self.assertAlmostEqual(
            rect_after.center().y() - rect_before.center().y(),
            -18.0,
        )
        self.assertAlmostEqual(point_after.x() - point_before.x(), 42.5)
        self.assertAlmostEqual(point_after.y() - point_before.y(), -18.0)
        image_point = self.canvas._widget_to_image(point_after)
        self.assertIsNotNone(image_point)
        assert image_point is not None
        self.assertAlmostEqual(image_point.x(), 35.0)
        self.assertAlmostEqual(image_point.y(), 45.0)
        self.assertEqual(self.canvas.vertex_data.to_dict(), geometry_before)
        self.assertFalse(self.canvas.set_canvas_level_offsets(42.5, -18.0))

    def test_comparison_overlay_uses_its_own_canvas_offsets(self) -> None:
        self.canvas.set_canvas_level_offsets(12.0, 8.0)
        self.canvas.zoom_scale = 1.4
        self.canvas.view_offset = QPointF(7.0, -3.0)
        comparison = _build_level(
            self.image_path,
            canvas_offset_x_pixels=-28.0,
            canvas_offset_y_pixels=31.0,
        )
        comparison.canvas_level_scale = 0.65

        self.assertTrue(self.canvas.set_level_comparison_overlay(comparison))
        overlay = self.canvas.get_level_comparison_overlay()
        self.assertIsNotNone(overlay)
        assert overlay is not None
        self.assertAlmostEqual(overlay.canvas_offset_x_pixels, -28.0)
        self.assertAlmostEqual(overlay.canvas_offset_y_pixels, 31.0)

        current_rect = self.canvas._image_display_rect()
        comparison_rect = self.canvas._comparison_image_display_rect(overlay)
        self.assertAlmostEqual(
            comparison_rect.center().x() - current_rect.center().x(),
            -40.0 * self.canvas.zoom_scale,
        )
        self.assertAlmostEqual(
            comparison_rect.center().y() - current_rect.center().y(),
            23.0 * self.canvas.zoom_scale,
        )
        self.assertAlmostEqual(
            comparison_rect.width(),
            current_rect.width() * 0.65,
        )

    def test_zoom_anchor_stays_fixed_with_nonzero_current_canvas_offset(
        self,
    ) -> None:
        self.canvas.set_canvas_level_scale(1.27)
        self.canvas.set_canvas_level_offsets(46.0, -31.0)
        reference_point = (73.0, 42.0)
        widget_anchor = self.canvas._image_to_widget(*reference_point)

        for zoom_factor in (1.75, 0.8):
            self.canvas._zoom_around_widget_point(widget_anchor, zoom_factor)
            zoomed_reference_point = self.canvas._image_to_widget(*reference_point)

            self.assertAlmostEqual(
                zoomed_reference_point.x(),
                widget_anchor.x(),
            )
            self.assertAlmostEqual(
                zoomed_reference_point.y(),
                widget_anchor.y(),
            )

    def test_comparison_transform_displacement_scales_with_canvas_zoom(
        self,
    ) -> None:
        self.canvas.set_canvas_level_scale(1.2)
        self.canvas.set_canvas_level_offsets(31.0, -17.0)
        comparison = _build_level(
            self.image_path,
            canvas_offset_x_pixels=-23.0,
            canvas_offset_y_pixels=29.0,
        )
        comparison.canvas_level_scale = 0.8
        self.assertTrue(self.canvas.set_level_comparison_overlay(comparison))
        overlay = self.canvas.get_level_comparison_overlay()
        assert overlay is not None

        initial_current_rect = self.canvas._image_display_rect()
        initial_comparison_rect = self.canvas._comparison_image_display_rect(overlay)
        initial_center_delta = (
            initial_comparison_rect.center() - initial_current_rect.center()
        )

        self.canvas._zoom_around_widget_point(QPointF(173.0, 129.0), 1.8)

        zoomed_current_rect = self.canvas._image_display_rect()
        zoomed_comparison_rect = self.canvas._comparison_image_display_rect(overlay)
        zoom_ratio = self.canvas.zoom_scale
        self.assertAlmostEqual(
            zoomed_comparison_rect.center().x() - zoomed_current_rect.center().x(),
            initial_center_delta.x() * zoom_ratio,
        )
        self.assertAlmostEqual(
            zoomed_comparison_rect.center().y() - zoomed_current_rect.center().y(),
            initial_center_delta.y() * zoom_ratio,
        )
        self.assertAlmostEqual(
            zoomed_comparison_rect.width() / initial_comparison_rect.width(),
            zoom_ratio,
        )

    def test_aligned_plan_reference_points_stay_aligned_after_zoom(self) -> None:
        self.canvas.set_canvas_level_scale(1.15)
        self.canvas.set_canvas_level_offsets(19.0, -11.0)
        current_reference = (48.0, 37.0)
        comparison_reference = (107.0, 66.0)
        comparison_scale = 0.72
        current_widget_point = self.canvas._image_to_widget(*current_reference)
        unshifted_comparison_rect = self.canvas._base_display_rect_for_image(
            self.canvas.blueprint_image,
            comparison_scale,
        )
        unshifted_comparison_point = _map_image_point_to_rect(
            unshifted_comparison_rect,
            (
                self.canvas.blueprint_image.width(),
                self.canvas.blueprint_image.height(),
            ),
            comparison_reference,
        )
        comparison = _build_level(
            self.image_path,
            canvas_offset_x_pixels=(
                current_widget_point.x() - unshifted_comparison_point.x()
            ),
            canvas_offset_y_pixels=(
                current_widget_point.y() - unshifted_comparison_point.y()
            ),
        )
        comparison.canvas_level_scale = comparison_scale
        self.assertTrue(self.canvas.set_level_comparison_overlay(comparison))
        overlay = self.canvas.get_level_comparison_overlay()
        assert overlay is not None

        comparison_widget_point = _map_image_point_to_rect(
            self.canvas._comparison_image_display_rect(overlay),
            (
                overlay.blueprint_image.width(),
                overlay.blueprint_image.height(),
            ),
            comparison_reference,
        )
        self.assertAlmostEqual(
            comparison_widget_point.x(),
            current_widget_point.x(),
        )
        self.assertAlmostEqual(
            comparison_widget_point.y(),
            current_widget_point.y(),
        )

        self.canvas._zoom_around_widget_point(QPointF(211.0, 156.0), 1.65)

        zoomed_current_point = self.canvas._image_to_widget(*current_reference)
        zoomed_comparison_point = _map_image_point_to_rect(
            self.canvas._comparison_image_display_rect(overlay),
            (
                overlay.blueprint_image.width(),
                overlay.blueprint_image.height(),
            ),
            comparison_reference,
        )
        self.assertAlmostEqual(
            zoomed_comparison_point.x(),
            zoomed_current_point.x(),
        )
        self.assertAlmostEqual(
            zoomed_comparison_point.y(),
            zoomed_current_point.y(),
        )


# ### Project persistence tests ###
class CanvasOffsetProjectTests(unittest.TestCase):
    def test_canvas_offsets_round_trip_and_retired_offsets_stay_ignored(
        self,
    ) -> None:
        level = _build_level(
            Path("missing.png"),
            canvas_offset_x_pixels=37.5,
            canvas_offset_y_pixels=-14.25,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "project.json"
            save_project(project_path, 0, [level])
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            saved_level = payload["levels"][0]
            self.assertAlmostEqual(saved_level["canvas_offset_x_pixels"], 37.5)
            self.assertAlmostEqual(saved_level["canvas_offset_y_pixels"], -14.25)

            loaded_level = next(
                candidate
                for candidate in load_project(project_path).levels
                if candidate.index == 2
            )
            self.assertAlmostEqual(loaded_level.canvas_offset_x_pixels, 37.5)
            self.assertAlmostEqual(loaded_level.canvas_offset_y_pixels, -14.25)

            saved_level.pop("canvas_offset_x_pixels")
            saved_level.pop("canvas_offset_y_pixels")
            saved_level.pop("canvas_level_scale")
            saved_level["image_scale"] = 2.5
            saved_level["image_offset_x"] = -91.0
            saved_level["image_offset_y"] = 63.0
            project_path.write_text(json.dumps(payload), encoding="utf-8")
            legacy_level = next(
                candidate
                for candidate in load_project(project_path).levels
                if candidate.index == 2
            )
            self.assertAlmostEqual(
                legacy_level.canvas_level_scale,
                DEFAULT_CANVAS_LEVEL_SCALE,
            )
            self.assertAlmostEqual(legacy_level.canvas_offset_x_pixels, 0.0)
            self.assertAlmostEqual(legacy_level.canvas_offset_y_pixels, 0.0)

    def test_missing_canvas_offsets_default_to_zero(self) -> None:
        level = _build_level(Path("missing.png"))

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "project.json"
            save_project(project_path, 0, [level])
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            saved_level = payload["levels"][0]
            saved_level.pop("canvas_offset_x_pixels")
            saved_level.pop("canvas_offset_y_pixels")
            project_path.write_text(json.dumps(payload), encoding="utf-8")

            loaded_level = next(
                candidate
                for candidate in load_project(project_path).levels
                if candidate.index == 2
            )
            self.assertAlmostEqual(loaded_level.canvas_offset_x_pixels, 0.0)
            self.assertAlmostEqual(loaded_level.canvas_offset_y_pixels, 0.0)


# ### Transform control layout tests ###
class CanvasTransformControlLayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        settings_path = Path(self.temporary_directory.name) / "settings.json"
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(settings_path)
        )
        self.workspace.resize(1600, 900)
        self.workspace.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        self.workspace.deleteLater()
        self.temporary_directory.cleanup()
        _qt_application.processEvents()

    def test_level_and_canvas_transforms_are_boxed_slider_columns(self) -> None:
        level_group = self.workspace.level_transform_group
        canvas_group = self.workspace.canvas_transform_group
        self.assertIsInstance(level_group, QGroupBox)
        self.assertIsInstance(canvas_group, QGroupBox)
        self.assertIsNot(level_group, canvas_group)

        level_sliders = (
            self.workspace.level_scale_slider,
            self.workspace.level_x_offset_slider,
            self.workspace.level_y_offset_slider,
        )
        canvas_sliders = (
            self.workspace.canvas_level_scale_slider,
            self.workspace.canvas_x_offset_slider,
            self.workspace.canvas_y_offset_slider,
        )
        _assert_slider_column(self, level_group, level_sliders)
        _assert_slider_column(self, canvas_group, canvas_sliders)

        level_labels = {label.text() for label in level_group.findChildren(QLabel)}
        canvas_labels = {label.text() for label in canvas_group.findChildren(QLabel)}
        self.assertTrue({"Level scale", "X offset", "Y offset"} <= level_labels)
        self.assertTrue(
            {
                "Canvas level scale",
                "Canvas X offset",
                "Canvas Y offset",
            }
            <= canvas_labels
        )

    def test_each_transform_drag_creates_one_reversible_undo_step(self) -> None:
        self.workspace._handle_level_transform_drag_started()
        self.workspace.level_x_offset_slider.setValue(
            round(1.25 * LEVEL_OFFSET_SLIDER_FACTOR)
        )
        self.assertEqual(self.workspace._canvas_undo_stack, [])
        self.workspace._handle_level_transform_drag_finished()
        self.assertTrue(self.workspace._level_transform_mesh_update_timer.isActive())
        self.assertEqual(self.workspace._canvas_undo_stack, [])
        self.workspace._commit_pending_level_transform_update()
        self.assertEqual(len(self.workspace._canvas_undo_stack), 1)
        self.workspace._handle_canvas_undo_requested()
        self.assertAlmostEqual(
            self.workspace.current_level.offset_x_meters,
            0.0,
        )

        self.workspace._handle_canvas_transform_drag_started()
        self.workspace.canvas_x_offset_slider.setValue(
            round(38.0 * CANVAS_OFFSET_SLIDER_FACTOR)
        )
        self.assertEqual(self.workspace._canvas_undo_stack, [])
        self.workspace._handle_canvas_transform_drag_finished()
        self.assertEqual(len(self.workspace._canvas_undo_stack), 1)
        self.workspace._handle_canvas_undo_requested()
        self.assertAlmostEqual(
            self.workspace.current_level.canvas_offset_x_pixels,
            0.0,
        )
        self.assertAlmostEqual(
            self.workspace.canvas.canvas_offset_x_pixels,
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
