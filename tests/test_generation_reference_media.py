# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.generation_jobs import GenerationJobManager
from housemaker.generation_state import MASK_MODE_PAINT, MaskPoint, MaskStroke
from housemaker.generation_workspace import GenerationWorkspace
from housemaker.merged_generation_workspace import MergedGenerationWorkspace
from housemaker.surface_texture_workspace import (
    SurfaceTextureGenerationWorkspace,
)

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _stroke(x_position: float, *, radius: float = 0.18) -> MaskStroke:
    return MaskStroke(
        MASK_MODE_PAINT,
        radius,
        (MaskPoint(x_position, 0.5),),
    )


def _video_frame(frame_index: int) -> np.ndarray:
    frame = np.empty((48, 64, 3), dtype=np.uint8)
    frame[:] = (20 + frame_index * 30, 80, 180 - frame_index * 40)
    frame[:, :, 1] += np.arange(64, dtype=np.uint8)[None, :]
    return frame


def _write_test_video(path: Path, *, frame_count: int = 2) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    if not writer.isOpened():
        raise unittest.SkipTest("MJPG video writing is unavailable.")
    try:
        for frame_index in range(frame_count):
            writer.write(_video_frame(frame_index))
    finally:
        writer.release()


def _reference_bgra() -> np.ndarray:
    image = np.zeros((13, 17, 4), dtype=np.uint8)
    image[:, :, 0] = 31
    image[:, :, 1] = np.arange(17, dtype=np.uint8)[None, :] + 90
    image[:, :, 2] = np.arange(13, dtype=np.uint8)[:, None] + 170
    image[:, :, 3] = 220
    image[:3, :4, 3] = 0
    return image


def _bgra_to_qimage(image_bgra: np.ndarray) -> QImage:
    rgba = cv2.cvtColor(image_bgra, cv2.COLOR_BGRA2RGBA)
    height, width = rgba.shape[:2]
    return QImage(
        rgba.data,
        width,
        height,
        int(rgba.strides[0]),
        QImage.Format.Format_RGBA8888,
    ).copy()


def _qimage_to_bgra(image: QImage) -> np.ndarray:
    converted = image.convertToFormat(QImage.Format.Format_RGBA8888)
    height = converted.height()
    width = converted.width()
    rgba = np.frombuffer(
        converted.constBits(),
        dtype=np.uint8,
        count=height * converted.bytesPerLine(),
    ).reshape(height, converted.bytesPerLine())[:, : width * 4]
    return cv2.cvtColor(rgba.reshape(height, width, 4), cv2.COLOR_RGBA2BGRA)


# ### Reference-media workflow tests ###
class GenerationReferenceMediaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.video_path = root / "reference.avi"
        _write_test_video(self.video_path)
        self.job_manager = GenerationJobManager()
        self.objects = GenerationWorkspace(
            asset_directory=root / "objects",
            job_manager=self.job_manager,
        )
        self.surfaces = SurfaceTextureGenerationWorkspace(
            asset_directory=root / "surfaces",
            application_settings=ApplicationSettingsStore(root / "settings.json"),
            job_manager=self.job_manager,
            shared_controls=self.objects.get_shared_controls(),
        )
        self.workspace = MergedGenerationWorkspace(
            self.surfaces,
            self.objects,
            self.job_manager,
        )
        self.workspace.show()
        _qt_application.clipboard().clear()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        _qt_application.clipboard().clear()
        self.workspace.shutdown()
        self.surfaces.shutdown()
        self.objects.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self.temporary_directory.cleanup()

    def _load_video(self) -> None:
        self.objects.load_video(str(self.video_path))
        self.surfaces.load_video(str(self.video_path))
        self.workspace.sync_shared_controls()
        _qt_application.processEvents()

    def _set_clipboard_reference(self, image_bgra: np.ndarray) -> None:
        _qt_application.clipboard().setImage(_bgra_to_qimage(image_bgra))
        self.workspace.sync_shared_controls()
        _qt_application.processEvents()

    def test_reference_media_buttons_share_the_load_video_row(self) -> None:
        expected_buttons = (
            ("load_video_button", "Load video"),
            ("copy_inpaint_button", "Copy inpaint"),
            ("paste_inpaint_button", "Paste"),
            ("load_reference_image_button", "Load image"),
        )
        buttons = []
        for attribute_name, expected_text in expected_buttons:
            with self.subTest(attribute_name=attribute_name):
                button = getattr(self.workspace, attribute_name)
                self.assertEqual(button.text(), expected_text)
                self.assertEqual(button.objectName(), attribute_name)
                buttons.append(button)

        parent = buttons[0].parentWidget()
        self.assertTrue(all(button.parentWidget() is parent for button in buttons))
        layout = parent.layout()
        indices = tuple(layout.indexOf(button) for button in buttons)
        self.assertEqual(indices, tuple(range(indices[0], indices[0] + 4)))

    def test_copy_inpaint_copies_the_masked_bgra_crop(self) -> None:
        self._load_video()
        self.workspace.video_view.set_strokes([_stroke(0.45)])
        self.workspace.sync_shared_controls()
        expected_crop = self.workspace.video_view.build_selected_object_crop()
        self.assertGreater(np.count_nonzero(expected_crop[:, :, 3]), 0)
        self.assertGreater(np.count_nonzero(expected_crop[:, :, 3] == 0), 0)

        self.workspace.copy_inpaint_button.click()

        clipboard_image = _qt_application.clipboard().image()
        self.assertFalse(clipboard_image.isNull())
        np.testing.assert_array_equal(
            _qimage_to_bgra(clipboard_image),
            expected_crop,
        )

    def test_paste_inpaint_is_one_reference_and_preserves_loaded_video(self) -> None:
        self._load_video()
        strokes = [_stroke(0.25, radius=0.08), _stroke(0.75, radius=0.08)]
        self.workspace.video_view.set_strokes(strokes)
        source_before = self.objects._video_source
        metadata_before = self.objects.get_data().video_metadata
        frame_before = self.workspace.video_view.get_frame_bgr()
        mask_before = self.workspace.video_view.get_mask()
        reference = _reference_bgra()
        self._set_clipboard_reference(reference)

        self.workspace.paste_inpaint_button.click()

        self.assertIs(self.objects._video_source, source_before)
        self.assertEqual(self.objects.get_data().video_metadata, metadata_before)
        np.testing.assert_array_equal(
            self.workspace.video_view.get_frame_bgr(),
            frame_before,
        )
        np.testing.assert_array_equal(
            self.workspace.video_view.get_mask(),
            mask_before,
        )
        self.assertEqual(self.workspace.video_view.get_strokes(), strokes)
        np.testing.assert_array_equal(
            self.workspace.video_view.get_reference_preview_bgra(),
            reference,
        )
        requests = self.objects._build_generation_requests()
        self.assertEqual(len(requests), 1)
        np.testing.assert_array_equal(requests[0].selected_object_bgra, reference)

    def test_load_image_is_temporary_and_seek_restores_the_video(self) -> None:
        self._load_video()
        source_before = self.objects._video_source
        metadata_before = self.objects.get_data().video_metadata
        assert source_before is not None
        expected_second_frame = source_before.get_frame(1)
        reference = _reference_bgra()
        image_path = Path(self.temporary_directory.name) / "reference.png"
        self.assertTrue(cv2.imwrite(str(image_path), reference))

        with patch(
            "housemaker.merged_generation_workspace.QFileDialog.getOpenFileName",
            return_value=(str(image_path), "Image files"),
        ):
            self.workspace.load_reference_image_button.click()

        self.assertIs(self.objects._video_source, source_before)
        self.assertEqual(self.objects.get_data().video_metadata, metadata_before)
        np.testing.assert_array_equal(
            self.objects._build_generation_requests()[0].selected_object_bgra,
            reference,
        )
        self.assertTrue(self.workspace.video_view.has_reference_preview())

        self.workspace.seekbar.setValue(1)
        _qt_application.processEvents()

        self.assertFalse(self.workspace.video_view.has_reference_preview())
        self.assertEqual(self.objects._build_generation_requests(), ())
        np.testing.assert_array_equal(
            self.workspace.video_view.get_frame_bgr(),
            expected_second_frame,
        )

    def test_paste_inpaint_without_video_allows_one_object_request(self) -> None:
        reference = _reference_bgra()
        self._set_clipboard_reference(reference)

        self.workspace.paste_inpaint_button.click()

        self.assertIsNone(self.objects._video_source)
        self.assertIsNone(self.surfaces._video_source)
        self.assertIsNone(self.workspace.video_view.get_frame_bgr())
        self.assertTrue(self.workspace.video_view.has_reference_preview())
        requests = self.objects._build_generation_requests()
        self.assertEqual(len(requests), 1)
        np.testing.assert_array_equal(requests[0].selected_object_bgra, reference)
        self.assertFalse(self.workspace.seekbar.isEnabled())
        self.assertFalse(self.workspace.paint_mask_button.isEnabled())
        self.assertFalse(self.workspace.erase_mask_button.isEnabled())


if __name__ == "__main__":
    unittest.main()
