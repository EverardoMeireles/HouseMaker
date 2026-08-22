# ### Environment setup ###
from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication, QListView

from housemaker.texture_atlas_view import (
    EMPTY_PREVIEW_TEXT,
    MAX_ATLAS_DIMENSION_PIXELS,
    TextureAtlasEntry,
    TextureAtlasView,
)


# ### Test application ###
_qt_application = QApplication.instance() or QApplication([])


# ### Fixture helpers ###
def _rgba_array(
    color: tuple[int, int, int, int] = (20, 80, 190, 255),
    *,
    width: int = 12,
    height: int = 8,
) -> np.ndarray:
    pixels = np.empty((height, width, 4), dtype=np.uint8)
    pixels[:, :] = np.asarray(color, dtype=np.uint8)
    return pixels


def _png_bytes(
    color: tuple[int, int, int, int] = (20, 80, 190, 255),
    *,
    width: int = 12,
    height: int = 8,
) -> bytes:
    output = io.BytesIO()
    Image.fromarray(
        _rgba_array(color, width=width, height=height),
        mode="RGBA",
    ).save(output, format="PNG")
    return output.getvalue()


def _entry(
    atlas_id: str,
    color: tuple[int, int, int, int],
    *,
    display_name: str | None = None,
    owner_id: str | None = None,
) -> TextureAtlasEntry:
    return TextureAtlasEntry(
        atlas_id=atlas_id,
        display_name=display_name or atlas_id,
        image=_rgba_array(color),
        owner_id=owner_id,
    )


# ### Entry decoding tests ###
class TextureAtlasEntryTests(unittest.TestCase):
    def test_rgba_array_is_owned_and_normalized(self) -> None:
        source = _rgba_array((10, 20, 30, 40))

        entry = TextureAtlasEntry("atlas-1", "Atlas one", source)
        source[:, :] = (200, 201, 202, 203)

        image = entry.get_image()
        self.assertEqual(image.format(), QImage.Format.Format_RGBA8888)
        self.assertEqual(image.size().toTuple(), (12, 8))
        self.assertEqual(image.pixelColor(0, 0), QColor(10, 20, 30, 40))

    def test_rgb_array_is_converted_to_opaque_rgba(self) -> None:
        source = np.full((3, 5, 3), (15, 25, 35), dtype=np.uint8)

        entry = TextureAtlasEntry("atlas-rgb", "RGB atlas", source)

        self.assertEqual(
            entry.get_image().pixelColor(0, 0),
            QColor(15, 25, 35, 255),
        )

    def test_png_bytes_and_png_path_are_decoded(self) -> None:
        payload = _png_bytes((70, 90, 110, 130))

        byte_entry = TextureAtlasEntry("bytes", "Bytes", payload)
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "atlas.png"
            image_path.write_bytes(payload)
            path_entry = TextureAtlasEntry("path", "Path", image_path)

        for entry in (byte_entry, path_entry):
            with self.subTest(entry=entry.atlas_id):
                self.assertEqual(entry.get_image().size().toTuple(), (12, 8))
                self.assertEqual(
                    entry.get_image().pixelColor(0, 0),
                    QColor(70, 90, 110, 130),
                )

    def test_qimage_is_copied_from_the_caller(self) -> None:
        source = QImage(4, 3, QImage.Format.Format_RGBA8888)
        source.fill(QColor(12, 34, 56, 78))

        entry = TextureAtlasEntry("qimage", "QImage", source)
        source.fill(Qt.GlobalColor.black)

        self.assertEqual(
            entry.get_image().pixelColor(0, 0),
            QColor(12, 34, 56, 78),
        )

    def test_invalid_images_and_metadata_are_rejected(self) -> None:
        invalid_cases = (
            ("", "Atlas", _rgba_array()),
            ("atlas", "", _rgba_array()),
            ("atlas", "Atlas", b"not a png"),
            ("atlas", "Atlas", np.zeros((2, 2), dtype=np.uint8)),
            ("atlas", "Atlas", np.zeros((2, 2, 4), dtype=np.float32)),
        )
        for atlas_id, display_name, image in invalid_cases:
            with self.subTest(atlas_id=atlas_id, display_name=display_name):
                with self.assertRaises(ValueError):
                    TextureAtlasEntry(atlas_id, display_name, image)

        with self.assertRaises(ValueError):
            TextureAtlasEntry(
                "wide",
                "Too wide",
                np.zeros(
                    (1, MAX_ATLAS_DIMENSION_PIXELS + 1, 4),
                    dtype=np.uint8,
                ),
            )
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaises(ValueError):
                TextureAtlasEntry(
                    "missing",
                    "Missing",
                    Path(temporary_directory) / "missing.png",
                )


# ### Atlas view tests ###
class TextureAtlasViewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.view = TextureAtlasView()
        self.view.resize(760, 560)
        self.view.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.view.close()
        self.view.deleteLater()
        _qt_application.processEvents()

    def test_layout_has_large_preview_and_horizontal_thumbnail_strip(self) -> None:
        self.assertEqual(self.view.preview_label.text(), EMPTY_PREVIEW_TEXT)
        self.assertEqual(
            self.view.atlas_list.viewMode(),
            QListView.ViewMode.IconMode,
        )
        self.assertEqual(
            self.view.atlas_list.flow(),
            QListView.Flow.LeftToRight,
        )
        self.assertFalse(self.view.atlas_list.isWrapping())
        self.assertEqual(
            self.view.atlas_list.verticalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )
        self.assertGreater(
            self.view.preview_label.height(),
            self.view.atlas_list.height(),
        )

    def test_set_atlases_selects_first_and_builds_named_thumbnails(self) -> None:
        first = _entry("first", (255, 20, 20, 255), owner_id="chair")
        second = _entry("second", (20, 255, 20, 255))
        selections: list[TextureAtlasEntry | None] = []
        self.view.atlas_selected.connect(selections.append)

        self.view.set_atlases([first, second])

        self.assertEqual(self.view.entries, (first, second))
        self.assertEqual(self.view.atlas_list.count(), 2)
        self.assertEqual(self.view.selected_atlas_id, "first")
        self.assertIs(self.view.selected_entry, first)
        self.assertIs(selections[-1], first)
        first_item = self.view.atlas_list.item(0)
        self.assertEqual(first_item.text(), "first")
        self.assertEqual(
            first_item.data(Qt.ItemDataRole.UserRole),
            "first",
        )
        self.assertIn("Owner: chair", first_item.toolTip())
        self.assertFalse(self.view.preview_label.pixmap().isNull())

    def test_explicit_and_thumbnail_selection_emit_the_selected_entry(self) -> None:
        first = _entry("first", (255, 20, 20, 255))
        second = _entry("second", (20, 255, 20, 255))
        selections: list[TextureAtlasEntry | None] = []
        self.view.atlas_selected.connect(selections.append)
        self.view.set_atlases([first, second])
        selections.clear()

        self.assertTrue(self.view.select_atlas("second"))
        self.assertEqual(self.view.selected_atlas_id, "second")
        self.assertEqual(self.view.atlas_list.currentRow(), 1)
        self.assertEqual(selections, [second])

        self.view.atlas_list.setCurrentRow(0)
        self.assertEqual(self.view.selected_atlas_id, "first")
        self.assertEqual(selections, [second, first])
        self.assertFalse(self.view.select_atlas("unknown"))
        self.assertEqual(self.view.selected_atlas_id, "first")

    def test_stable_id_selection_survives_reordering_and_image_replacement(self) -> None:
        first = _entry("first", (255, 20, 20, 255))
        second = _entry("second", (20, 255, 20, 255))
        self.view.set_atlases([first, second], selected_atlas_id="second")
        replacement = _entry("second", (20, 20, 255, 255), display_name="Updated")

        self.view.set_atlases([replacement, first])

        self.assertEqual(self.view.selected_atlas_id, "second")
        self.assertIs(self.view.selected_entry, replacement)
        self.assertEqual(self.view.atlas_list.currentRow(), 0)
        self.assertEqual(self.view.preview_label.toolTip(), "Updated")

    def test_clear_removes_entries_preview_and_selection(self) -> None:
        entry = _entry("first", (255, 20, 20, 255))
        selections: list[TextureAtlasEntry | None] = []
        self.view.atlas_selected.connect(selections.append)
        self.view.set_atlases([entry])
        selections.clear()

        self.view.clear()

        self.assertEqual(self.view.entries, ())
        self.assertEqual(self.view.atlas_list.count(), 0)
        self.assertIsNone(self.view.selected_atlas_id)
        self.assertIsNone(self.view.selected_entry)
        self.assertEqual(self.view.preview_label.text(), EMPTY_PREVIEW_TEXT)
        self.assertEqual(selections, [None])

    def test_duplicate_or_unknown_selected_ids_do_not_replace_existing_data(self) -> None:
        original = _entry("original", (255, 20, 20, 255))
        self.view.set_atlases([original])
        duplicate_one = _entry("duplicate", (20, 255, 20, 255))
        duplicate_two = _entry("duplicate", (20, 20, 255, 255))

        with self.assertRaises(ValueError):
            self.view.set_atlases([duplicate_one, duplicate_two])
        self.assertEqual(self.view.entries, (original,))

        with self.assertRaises(ValueError):
            self.view.set_atlases([duplicate_one], selected_atlas_id="missing")
        self.assertEqual(self.view.entries, (original,))

    def test_preview_scales_to_fit_without_changing_aspect_ratio(self) -> None:
        wide = TextureAtlasEntry(
            "wide",
            "Wide atlas",
            _rgba_array(width=400, height=100),
        )
        self.view.set_atlases([wide])
        self.view.resize(500, 460)
        _qt_application.processEvents()

        pixmap = self.view.preview_label.pixmap()
        self.assertFalse(pixmap.isNull())
        self.assertLessEqual(pixmap.width(), self.view.preview_label.width())
        self.assertLessEqual(pixmap.height(), self.view.preview_label.height())
        self.assertAlmostEqual(
            pixmap.width() / pixmap.height(),
            4.0,
            delta=0.05,
        )


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
