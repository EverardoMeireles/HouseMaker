# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import copy
import math
import tempfile
import unittest
from pathlib import Path

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication

from housemaker.generation_state import (
    GeneratedObjectPlacement,
    GeneratedObjectRecord,
    GenerationData,
)
from housemaker.models import LevelData, VertexData
from housemaker.object_placement_dialog import (
    LEVEL_INDEX_ITEM_ROLE,
    ObjectPlacementDialog,
)

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _write_blueprint(
    directory: Path,
    name: str,
    width: int,
    height: int,
) -> Path:
    path = directory / name
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(QColor("#334155"))
    if not image.save(str(path), "PNG"):
        raise RuntimeError("The placement test blueprint could not be saved.")
    return path


def _build_level(
    index: int,
    image_path: Path | None,
    *,
    include_in_export: bool = True,
) -> LevelData:
    vertex_data = VertexData()
    first = vertex_data.add_vertex(15.0, 18.0)
    second = vertex_data.add_vertex(80.0, 72.0)
    vertex_data.add_edge(first.id, second.id)
    return LevelData(
        index=index,
        name=f"Placement level {index}",
        vertex_data=vertex_data,
        image_path=None if image_path is None else str(image_path),
        include_in_export=include_in_export,
    )


def _record(
    placement: GeneratedObjectPlacement | None,
) -> GeneratedObjectRecord:
    return GeneratedObjectRecord(
        object_id="chair",
        frame_index=4,
        object_name="Chair",
        pipeline={},
        provider="meshy",
        provider_task_id="task-chair",
        asset_path="chair.glb",
        placement=placement,
    )


# ### Placement-state tests ###
class GeneratedObjectPlacementTests(unittest.TestCase):
    def test_placement_and_generation_data_round_trip(self) -> None:
        placement = GeneratedObjectPlacement(
            level_index=5,
            image_x=123.25,
            image_y=67.5,
            height_offset_meters=1.25,
            rotation_degrees=(15.0, -30.0, 225.0),
        )
        record = _record(placement)
        data = GenerationData(generated_objects=[record])

        self.assertEqual(
            GeneratedObjectPlacement.from_dict(placement.to_dict()),
            placement,
        )
        self.assertEqual(GenerationData.from_dict(data.to_dict()), data)
        self.assertEqual(
            record.to_dict()["placement"],
            {
                "level_index": 5,
                "image_x": 123.25,
                "image_y": 67.5,
                "height_offset_meters": 1.25,
                "rotation_degrees": [15.0, -30.0, 225.0],
            },
        )

    def test_legacy_placement_defaults_to_floor_height_and_zero_rotation(
        self,
    ) -> None:
        placement = GeneratedObjectPlacement.from_dict(
            {
                "level_index": 2,
                "image_x": 20.0,
                "image_y": 30.0,
            }
        )

        self.assertEqual(placement.height_offset_meters, 0.0)
        self.assertEqual(placement.rotation_degrees, (0.0, 0.0, 0.0))

    def test_legacy_record_without_placement_loads_unplaced(self) -> None:
        payload = _record(None).to_dict()
        payload.pop("placement")

        loaded = GeneratedObjectRecord.from_dict(payload)

        self.assertIsNone(loaded.placement)

    def test_placement_rejects_invalid_level_and_coordinates(self) -> None:
        invalid_values = (
            {"level_index": True, "image_x": 1.0, "image_y": 2.0},
            {"level_index": 2.0, "image_x": 1.0, "image_y": 2.0},
            {"level_index": -1, "image_x": 1.0, "image_y": 2.0},
            {"level_index": 2, "image_x": True, "image_y": 2.0},
            {"level_index": 2, "image_x": "1.0", "image_y": 2.0},
            {"level_index": 2, "image_x": math.nan, "image_y": 2.0},
            {"level_index": 2, "image_x": 1.0, "image_y": math.inf},
        )

        for values in invalid_values:
            with self.subTest(values=values), self.assertRaises(ValueError):
                GeneratedObjectPlacement(**values)

    def test_placement_rejects_invalid_height_and_rotation(self) -> None:
        invalid_values = (
            {"height_offset_meters": True},
            {"height_offset_meters": "1.0"},
            {"height_offset_meters": math.nan},
            {"height_offset_meters": math.inf},
            {"rotation_degrees": "0,0,0"},
            {"rotation_degrees": (0.0, 0.0)},
            {"rotation_degrees": (0.0, 0.0, True)},
            {"rotation_degrees": (0.0, "1.0", 0.0)},
            {"rotation_degrees": (0.0, math.nan, 0.0)},
            {"rotation_degrees": (0.0, 0.0, math.inf)},
        )

        for extra_values in invalid_values:
            with (
                self.subTest(extra_values=extra_values),
                self.assertRaises((TypeError, ValueError)),
            ):
                GeneratedObjectPlacement(
                    level_index=2,
                    image_x=1.0,
                    image_y=2.0,
                    **extra_values,
                )

    def test_record_rejects_an_untyped_placement(self) -> None:
        with self.assertRaises(ValueError):
            GeneratedObjectRecord(
                object_id="chair",
                frame_index=0,
                object_name="Chair",
                pipeline={},
                provider="meshy",
                provider_task_id="task-chair",
                asset_path="chair.glb",
                placement={"level_index": 2},  # type: ignore[arg-type]
            )


# ### Placement-dialog tests ###
class ObjectPlacementDialogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)
        self.dialogs: list[ObjectPlacementDialog] = []

    def tearDown(self) -> None:
        for dialog in self.dialogs:
            dialog.close()
        _qt_application.processEvents()
        self.temporary_directory.cleanup()

    def _show_dialog(
        self,
        levels: list[LevelData],
    ) -> ObjectPlacementDialog:
        dialog = ObjectPlacementDialog(levels)
        self.dialogs.append(dialog)
        dialog.resize(1000, 700)
        dialog.show()
        _qt_application.processEvents()
        return dialog

    def test_lists_only_levels_with_usable_images_by_stable_index(self) -> None:
        first_image = _write_blueprint(
            self.directory,
            "first.png",
            200,
            100,
        )
        hidden_image = _write_blueprint(
            self.directory,
            "hidden.png",
            180,
            90,
        )
        last_image = _write_blueprint(
            self.directory,
            "last.png",
            160,
            120,
        )
        broken_image = self.directory / "broken.png"
        broken_image.write_text("not an image", encoding="utf-8")
        dialog = self._show_dialog(
            [
                _build_level(2, first_image),
                _build_level(3, hidden_image, include_in_export=False),
                _build_level(4, None),
                _build_level(5, self.directory / "missing.png"),
                _build_level(6, broken_image),
                _build_level(7, last_image),
            ]
        )

        self.assertEqual(dialog.level_list.count(), 3)
        self.assertEqual(
            [
                dialog.level_list.item(row).data(LEVEL_INDEX_ITEM_ROLE)
                for row in range(dialog.level_list.count())
            ],
            [2, 3, 7],
        )
        self.assertTrue(dialog.select_level(7))
        self.assertEqual(
            dialog.level_list.currentItem().data(LEVEL_INDEX_ITEM_ROLE),
            7,
        )
        self.assertTrue(dialog.select_level(3))
        self.assertFalse(dialog.select_level(5))

    def test_inside_left_click_emits_once_and_accepts_placement(self) -> None:
        image_path = _write_blueprint(
            self.directory,
            "ground.png",
            200,
            100,
        )
        dialog = self._show_dialog([_build_level(4, image_path)])
        selected = QSignalSpy(dialog.placement_selected)
        click_position = dialog.canvas._image_to_widget(125.0, 35.0).toPoint()

        QTest.mouseClick(
            dialog.canvas,
            Qt.MouseButton.LeftButton,
            pos=click_position,
        )
        _qt_application.processEvents()

        self.assertEqual(selected.count(), 1)
        placement = selected.at(0)[0]
        self.assertIsInstance(placement, GeneratedObjectPlacement)
        self.assertEqual(placement.level_index, 4)
        self.assertAlmostEqual(placement.image_x, 125.0, delta=0.2)
        self.assertAlmostEqual(placement.image_y, 35.0, delta=0.2)
        self.assertEqual(dialog.selected_placement, placement)
        self.assertEqual(dialog.result(), ObjectPlacementDialog.DialogCode.Accepted)

    def test_outside_and_right_clicks_are_ignored_without_canvas_mutation(
        self,
    ) -> None:
        image_path = _write_blueprint(
            self.directory,
            "readonly.png",
            240,
            140,
        )
        level = _build_level(6, image_path)
        original_level = copy.deepcopy(level)
        dialog = self._show_dialog([level])
        selected = QSignalSpy(dialog.placement_selected)
        inside = dialog.canvas._image_to_widget(80.0, 60.0).toPoint()

        QTest.mouseClick(
            dialog.canvas,
            Qt.MouseButton.LeftButton,
            pos=QPoint(1, 1),
        )
        QTest.mouseClick(
            dialog.canvas,
            Qt.MouseButton.RightButton,
            pos=inside,
        )
        _qt_application.processEvents()

        self.assertEqual(selected.count(), 0)
        self.assertIsNone(dialog.selected_placement)
        self.assertEqual(level, original_level)
        self.assertEqual(len(dialog.canvas.vertex_data.vertices), 2)
        self.assertEqual(len(dialog.canvas.vertex_data.edges), 1)

    def test_switching_levels_uses_copies_and_emits_selected_level(self) -> None:
        first_image = _write_blueprint(
            self.directory,
            "level-two.png",
            200,
            100,
        )
        second_image = _write_blueprint(
            self.directory,
            "level-five.png",
            300,
            180,
        )
        levels = [
            _build_level(2, first_image),
            _build_level(5, second_image),
        ]
        original_levels = copy.deepcopy(levels)
        dialog = self._show_dialog(levels)
        selected = QSignalSpy(dialog.placement_selected)

        dialog.level_list.setCurrentRow(1)
        _qt_application.processEvents()
        click_position = dialog.canvas._image_to_widget(210.0, 90.0).toPoint()
        QTest.mouseClick(
            dialog.canvas,
            Qt.MouseButton.LeftButton,
            pos=click_position,
        )
        _qt_application.processEvents()

        self.assertEqual(selected.count(), 1)
        placement = selected.at(0)[0]
        self.assertEqual(placement.level_index, 5)
        self.assertAlmostEqual(placement.image_x, 210.0, delta=0.2)
        self.assertAlmostEqual(placement.image_y, 90.0, delta=0.2)
        self.assertEqual(levels, original_levels)

    def test_empty_usable_image_list_reports_no_available_levels(self) -> None:
        broken_path = self.directory / "broken-only.png"
        broken_path.write_text("not an image", encoding="utf-8")
        dialog = self._show_dialog(
            [
                _build_level(1, None),
                _build_level(2, self.directory / "missing.png"),
                _build_level(3, broken_path, include_in_export=False),
            ]
        )

        self.assertEqual(dialog.level_list.count(), 0)
        self.assertIsNone(dialog.selected_placement)
        self.assertIn("usable blueprint images", dialog.status_label.text())


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
