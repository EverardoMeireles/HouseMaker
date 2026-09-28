# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import unittest

import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication

from housemaker.generation_state import (
    MASK_MODE_PAINT,
    MaskPoint,
    MaskStroke,
)
from housemaker.generation_views import MASK_OVERLAY_RGB, VideoInpaintView

# ### Test helpers ###
_qt_application = QApplication.instance() or QApplication([])


def _stroke(x: float, y: float, radius: float = 0.08) -> MaskStroke:
    return MaskStroke(
        mode=MASK_MODE_PAINT,
        radius_normalized=radius,
        points=(MaskPoint(x, y),),
    )


# ### Object-mask tests ###
class ObjectMaskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.view = VideoInpaintView()
        self.view.resize(400, 300)
        self.view.show()
        self.frame = np.full((100, 200, 3), (20, 80, 140), dtype=np.uint8)
        self.view.set_frame(self.frame, [_stroke(0.3, 0.5, 0.12)])
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.view.close()
        _qt_application.processEvents()

    def test_paint_and_clear_use_the_existing_object_mask(self) -> None:
        signal = QSignalSpy(self.view.strokes_changed)

        QTest.mouseClick(
            self.view,
            Qt.MouseButton.LeftButton,
            pos=QPoint(250, 150),
        )
        _qt_application.processEvents()

        self.assertEqual(signal.count(), 1)
        self.assertEqual(len(self.view.get_strokes()), 2)

        self.view.clear_mask()

        self.assertFalse(self.view.has_selection())
        self.assertEqual(signal.count(), 2)

    def test_object_overlay_uses_the_standard_mask_color(self) -> None:
        object_color = self.view._mask_overlay_image.pixelColor(60, 50)

        self.assertEqual(
            (object_color.red(), object_color.green(), object_color.blue()),
            MASK_OVERLAY_RGB,
        )

    def test_frame_restores_the_existing_object_strokes(self) -> None:
        object_stroke = _stroke(0.2, 0.4)

        self.view.set_frame(self.frame, [object_stroke])

        self.assertEqual(self.view.get_strokes(), [object_stroke])
        self.assertGreater(np.count_nonzero(self.view.get_mask()), 0)


# ### Reference-edit input tests ###
class ReferenceEditInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.view = VideoInpaintView()
        frame = np.zeros((80, 120, 3), dtype=np.uint8)
        frame[:, :] = (15, 45, 90)
        self.view.set_frame(frame, [_stroke(0.5, 0.5, 0.2)])

    def tearDown(self) -> None:
        self.view.close()

    def test_build_source_encodes_object_selection_in_its_alpha(self) -> None:
        source_bgra = self.view.build_reference_edit_source(
            padding_ratio=0.1
        )

        self.assertEqual(source_bgra.shape[2], 4)
        self.assertGreater(np.count_nonzero(source_bgra[:, :, 3]), 0)
        self.assertGreater(np.count_nonzero(source_bgra[:, :, 3] == 0), 0)

    def test_build_source_rejects_multiple_object_blobs(self) -> None:
        self.view.set_strokes([_stroke(0.2, 0.5), _stroke(0.8, 0.5)])

        with self.assertRaisesRegex(ValueError, "exactly one connected object"):
            self.view.build_reference_edit_source()

    def test_build_source_requires_an_object_selection(self) -> None:
        self.view.set_strokes([])

        with self.assertRaisesRegex(ValueError, "one selected object"):
            self.view.build_reference_edit_source()


# ### Reference preview tests ###
class ReferencePreviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.view = VideoInpaintView()
        self.view.resize(400, 300)
        self.view.show()
        self.frame = np.full((100, 200, 3), 70, dtype=np.uint8)
        self.view.set_frame(self.frame, [_stroke(0.5, 0.5)])
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.view.close()
        _qt_application.processEvents()

    def test_arbitrary_bgra_preview_is_display_only_and_can_be_cleared(self) -> None:
        source_before = self.view.get_frame_bgr()
        object_strokes_before = self.view.get_strokes()
        frame_signal = QSignalSpy(self.view.frame_changed)
        object_signal = QSignalSpy(self.view.strokes_changed)
        preview = np.zeros((33, 57, 4), dtype=np.uint8)
        preview[:, :, :3] = (10, 20, 240)
        preview[:, :, 3] = 128

        self.view.set_reference_preview_bgra(preview)

        self.assertTrue(self.view.has_reference_preview())
        self.assertEqual(self.view._get_display_image().size().width(), 57)
        self.assertFalse(self.view._can_edit_mask())
        QTest.mouseClick(
            self.view,
            Qt.MouseButton.LeftButton,
            pos=QPoint(200, 150),
        )
        self.assertTrue(np.array_equal(self.view.get_frame_bgr(), source_before))
        self.assertEqual(self.view.get_strokes(), object_strokes_before)
        self.assertEqual(frame_signal.count(), 0)
        self.assertEqual(object_signal.count(), 0)

        self.view.clear_reference_preview()

        self.assertFalse(self.view.has_reference_preview())
        self.assertEqual(self.view._get_display_image().size().width(), 200)
        self.assertTrue(self.view._can_edit_mask())


if __name__ == "__main__":
    unittest.main()
