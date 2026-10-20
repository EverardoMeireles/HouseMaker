# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication

from housemaker.generation_state import (
    MASK_MODE_ERASE,
    MASK_MODE_PAINT,
    MaskPoint,
    MaskStroke,
)
from housemaker.generation_views import (
    MASK_OVERLAY_RGB,
    SELECTED_MASK_OVERLAY_RGB,
    VideoInpaintView,
    rasterize_mask_strokes,
)

# ### Test helpers ###
_application = QApplication.instance() or QApplication([])


def _dot(x: float, y: float, radius: float = 0.06) -> MaskStroke:
    return MaskStroke(MASK_MODE_PAINT, radius, (MaskPoint(x, y),))


# ### Connected inpaint tests ###
class ManualInpaintBlobTests(unittest.TestCase):
    def setUp(self) -> None:
        self.view = VideoInpaintView()
        self.view.resize(400, 200)
        self.view.show()
        self.frame = np.full((100, 200, 3), (20, 80, 140), dtype=np.uint8)
        self.strokes = [_dot(0.75, 0.75), _dot(0.2, 0.25), _dot(0.7, 0.25)]
        self.view.set_frame(self.frame, self.strokes)
        _application.processEvents()

    def tearDown(self) -> None:
        self.view.close()
        _application.processEvents()

    def test_blobs_are_sorted_and_selected_mask_is_full_frame(self) -> None:
        masks = self.view.get_blob_masks()

        self.assertEqual(self.view.get_mask_shape(), (100, 200))
        self.assertEqual(self.view.get_blob_count(), 3)
        self.assertEqual(len(masks), 3)
        self.assertTrue(masks[0][25, 40])
        self.assertTrue(masks[1][25, 139])
        self.assertTrue(masks[2][74, 149])
        self.assertEqual(self.view.get_selected_blob_index(), 0)
        np.testing.assert_array_equal(self.view.get_selected_blob_mask(), masks[0])
        masks[0][:] = 0
        self.assertTrue(self.view.get_selected_blob_mask()[25, 40])

    def test_selected_crop_does_not_include_neighboring_blobs(self) -> None:
        self.view.select_blob(1)

        crop = self.view.get_selected_blob_crop(padding_ratio=0)

        self.assertEqual(crop.shape, (13, 13, 4))
        self.assertEqual(len(self.view.build_selected_object_crops()), 3)
        np.testing.assert_array_equal(
            crop, self.view.build_selected_object_crop(padding_ratio=0),
        )

    def test_only_selected_blob_is_red(self) -> None:
        self.view.select_blob(1)

        selected_color = self.view._mask_overlay_image.pixelColor(139, 25)
        other_color = self.view._mask_overlay_image.pixelColor(40, 25)

        self.assertEqual(selected_color.getRgb()[:3], SELECTED_MASK_OVERLAY_RGB)
        self.assertEqual(other_color.getRgb()[:3], MASK_OVERLAY_RGB)

    def test_selecting_existing_blob_adds_no_stroke(self) -> None:
        signal = QSignalSpy(self.view.strokes_changed)

        QTest.mouseClick(self.view, Qt.MouseButton.LeftButton, pos=QPoint(280, 50))

        self.assertEqual(self.view.get_strokes(), self.strokes)
        self.assertEqual(self.view.get_selected_blob_index(), 1)
        self.assertEqual(signal.count(), 0)

    def test_new_blob_becomes_selected(self) -> None:
        QTest.mouseClick(self.view, Qt.MouseButton.LeftButton, pos=QPoint(190, 150))

        selected = self.view.get_selected_blob_mask()

        self.assertTrue(selected[75, 95])
        self.assertFalse(selected[25, 40])
        self.assertEqual(len(self.view.get_blob_masks()), 4)

    def test_eraser_can_edit_a_painted_blob_instead_of_only_selecting_it(self) -> None:
        self.view.set_brush_mode(MASK_MODE_ERASE)
        self.view.set_brush_radius_pixels(2)
        self.assertTrue(self.view.get_mask()[25, 139])

        QTest.mouseClick(self.view, Qt.MouseButton.LeftButton, pos=QPoint(280, 50))

        self.assertFalse(self.view.get_mask()[25, 139])
        self.assertEqual(self.view.get_strokes()[-1].mode, MASK_MODE_ERASE)
        self.assertEqual(len(self.view.get_strokes()), 4)

    def test_dragging_from_inside_a_blob_extends_the_painted_selection(self) -> None:
        self.view.set_brush_radius_pixels(4)
        QTest.mousePress(self.view, Qt.MouseButton.LeftButton, pos=QPoint(80, 50))
        QTest.mouseMove(self.view, QPoint(110, 50))
        QTest.mouseRelease(self.view, Qt.MouseButton.LeftButton, pos=QPoint(110, 50))

        self.assertTrue(self.view.get_mask()[25, 55])
        self.assertEqual(len(self.view.get_strokes()), 4)
        self.assertTrue(self.view.get_selected_blob_mask()[25, 55])

    def test_clear_removes_only_selected_and_is_replayable(self) -> None:
        self.view.select_blob(1)
        original_mask = self.view.get_mask()

        self.view.clear_mask()

        self.assertEqual(len(self.view.get_blob_masks()), 2)
        self.assertFalse(self.view.get_mask()[25, 139])
        self.assertTrue(self.view.get_mask()[25, 40])
        deletion = self.view.get_strokes()[-1]
        self.assertEqual(deletion.mode, "erase")
        self.assertTrue(deletion.is_fill)
        reloaded = [MaskStroke.from_dict(stroke.to_dict()) for stroke in self.view.get_strokes()]
        np.testing.assert_array_equal(
            self.view.get_mask(), rasterize_mask_strokes((200, 100), reloaded),
        )
        self.view.set_strokes(self.view.get_strokes()[:-1])
        np.testing.assert_array_equal(self.view.get_mask(), original_mask)

    def test_right_click_removes_clicked_not_selected_blob(self) -> None:
        QTest.mouseClick(self.view, Qt.MouseButton.RightButton, pos=QPoint(280, 50))

        self.assertFalse(self.view.get_mask()[25, 139])
        self.assertTrue(self.view.get_mask()[25, 40])
        self.assertEqual(len(self.view.get_blob_masks()), 2)
        self.assertEqual(self.view.get_selected_blob_index(), 0)

    def test_right_click_fills_enclosed_outline_then_removes_complete_blob(self) -> None:
        outline = MaskStroke(
            MASK_MODE_PAINT, 0.015,
            tuple(MaskPoint(x, y) for x, y in (
                (0.2, 0.2), (0.8, 0.2), (0.8, 0.8),
                (0.2, 0.8), (0.2, 0.2),
            )),
        )
        self.view.set_strokes([outline])
        self.assertFalse(self.view.get_mask()[50, 100])

        QTest.mouseClick(self.view, Qt.MouseButton.RightButton, pos=QPoint(200, 100))

        self.assertTrue(self.view.get_mask()[50, 100])
        np.testing.assert_array_equal(
            self.view.get_mask(), rasterize_mask_strokes((200, 100), self.view.get_strokes()),
        )
        QTest.mouseClick(self.view, Qt.MouseButton.RightButton, pos=QPoint(200, 100))
        self.assertFalse(self.view.has_selection())

    def test_selection_survives_rebuild_and_other_blob_removal_without_signal_loop(self) -> None:
        self.view.select_blob(2)
        signal = QSignalSpy(self.view.selected_blob_changed)

        self.view.set_strokes(self.view.get_strokes())
        self.view.select_blob(2)

        self.assertEqual(signal.count(), 0)
        QTest.mouseClick(self.view, Qt.MouseButton.RightButton, pos=QPoint(80, 50))
        self.assertEqual(self.view.get_selected_blob_index(), 1)
        self.assertTrue(self.view.get_selected_blob_mask()[74, 149])

    def test_new_blob_emits_one_selection_signal_but_rebuilding_it_does_not(self) -> None:
        signal = QSignalSpy(self.view.selected_blob_changed)

        QTest.mouseClick(self.view, Qt.MouseButton.LeftButton, pos=QPoint(190, 150))

        self.assertEqual(signal.count(), 1)
        self.view.set_strokes(self.view.get_strokes())
        self.assertEqual(signal.count(), 1)

    def test_right_click_actions_do_not_exceed_the_stroke_limit(self) -> None:
        signal = QSignalSpy(self.view.strokes_changed)
        original = self.view.get_mask()
        with patch("housemaker.generation_views.MAX_MASK_STROKES_PER_FRAME", 3):
            QTest.mouseClick(self.view, Qt.MouseButton.RightButton, pos=QPoint(80, 50))
            self.view.clear_mask()
            self.assertFalse(self.view._append_enclosed_fill(MaskPoint(0.5, 0.5)))

        np.testing.assert_array_equal(self.view.get_mask(), original)
        self.assertEqual(len(self.view.get_strokes()), 3)
        self.assertEqual(signal.count(), 0)

    def test_connectivity_is_not_recomputed_during_each_paint_move(self) -> None:
        QTest.mousePress(self.view, Qt.MouseButton.LeftButton, pos=QPoint(190, 100))
        with patch("housemaker.generation_views.cv2.connectedComponentsWithStats") as components:
            QTest.mouseMove(self.view, QPoint(190, 130))
            self.assertEqual(components.call_count, 0)
        QTest.mouseRelease(self.view, Qt.MouseButton.LeftButton, pos=QPoint(190, 130))

    def test_empty_selection_and_invalid_indices(self) -> None:
        self.view.select_blob(None)
        self.assertFalse(np.any(self.view.get_selected_blob_mask()))
        self.assertEqual(self.view.get_selected_blob_crop().shape, (0, 0, 4))
        with self.assertRaises(IndexError):
            self.view.select_blob(10)
        self.view.clear_frame()
        self.assertEqual(self.view.get_blob_masks(), ())
        self.assertIsNone(self.view.get_selected_blob_index())


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
