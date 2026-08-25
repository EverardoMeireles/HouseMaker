# ### Environment setup ###
from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import QApplication, QListView, QListWidgetItem

from housemaker.texture_atlas_view import (
    EDIT_MASK_OVERLAY_COLOR,
    EMPTY_PREVIEW_TEXT,
    MAX_ATLAS_DIMENSION_PIXELS,
    TextureAtlasEntry,
    TextureAtlasView,
    UV_EDGE_COLOR,
    UV_VERTEX_COLOR,
    _compose_uv_overlay,
    _paint_uv_overlay,
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


def _image_has_color_near_uv(
    image: QImage,
    uv: tuple[float, float],
    expected: QColor,
    *,
    radius: int = 3,
) -> bool:
    center_x = round(uv[0] * max(image.width() - 1, 1))
    center_y = round((1.0 - uv[1]) * max(image.height() - 1, 1))
    for y in range(
        max(0, center_y - radius),
        min(image.height(), center_y + radius + 1),
    ):
        for x in range(
            max(0, center_x - radius),
            min(image.width(), center_x + radius + 1),
        ):
            if image.pixelColor(x, y) == expected:
                return True
    return False


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

    def test_thumbnail_double_click_activates_live_entry_but_not_stale_id(
        self,
    ) -> None:
        first = _entry("first", (255, 20, 20, 255))
        activated: list[TextureAtlasEntry] = []
        self.view.atlas_activated.connect(activated.append)
        self.view.set_atlases([first])

        self.view.atlas_list.itemDoubleClicked.emit(
            self.view.atlas_list.item(0)
        )

        self.assertEqual(activated, [first])
        stale_item = QListWidgetItem("stale")
        stale_item.setData(Qt.ItemDataRole.UserRole, first.atlas_id)
        self.view.clear()
        self.view._handle_item_double_clicked(stale_item)
        self.assertEqual(activated, [first])

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

    def test_uv_overlay_is_opt_in_validated_and_cleared_with_the_view(self) -> None:
        triangles = (
            ((0.1, 0.2), (0.8, 0.2), (0.1, 0.9)),
        )

        self.assertFalse(self.view.uv_overlay_enabled)
        self.assertEqual(self.view.uv_overlay_triangles, ())
        self.view.set_uv_overlay_triangles(triangles)
        self.assertEqual(self.view.uv_overlay_triangles, triangles)
        self.view.set_uv_overlay_enabled(True)
        self.assertTrue(self.view.uv_overlay_enabled)

        with self.assertRaisesRegex(ValueError, "three points"):
            self.view.set_uv_overlay_triangles((((0.0, 0.0), (1.0, 1.0)),))
        with self.assertRaisesRegex(ValueError, "finite"):
            self.view.set_uv_overlay_triangles(
                (((0.0, 0.0), (float("nan"), 0.5), (1.0, 1.0)),)
            )
        self.assertEqual(self.view.uv_overlay_triangles, triangles)

        self.view.clear()

        self.assertTrue(self.view.uv_overlay_enabled)
        self.assertEqual(self.view.uv_overlay_triangles, ())

    def test_uv_overlay_draws_flipped_vertices_and_edges_over_preview(self) -> None:
        background = (20, 30, 40, 255)
        entry = TextureAtlasEntry(
            "square",
            "Square",
            _rgba_array(background, width=160, height=160),
        )
        triangle = (
            ((0.2, 0.2), (0.8, 0.2), (0.2, 0.8)),
        )
        self.view.set_uv_overlay_triangles(triangle)
        self.view.set_atlases((entry,))
        without_overlay = self.view.preview_label.pixmap().toImage()

        self.view.set_uv_overlay_enabled(True)
        with_overlay = self.view.preview_label.pixmap().toImage()

        self.assertEqual(
            without_overlay.pixelColor(
                round(0.2 * (without_overlay.width() - 1)),
                round(0.2 * (without_overlay.height() - 1)),
            ),
            QColor(*background),
        )
        for uv in triangle[0]:
            with self.subTest(vertex=uv):
                self.assertTrue(
                    _image_has_color_near_uv(
                        with_overlay,
                        uv,
                        UV_VERTEX_COLOR,
                    )
                )
        self.assertTrue(
            _image_has_color_near_uv(
                with_overlay,
                (0.5, 0.2),
                UV_EDGE_COLOR,
            )
        )

        self.view.set_uv_overlay_enabled(False)
        restored = self.view.preview_label.pixmap().toImage()
        self.assertEqual(restored, without_overlay)

    def test_edit_mask_overlay_is_owned_opt_in_and_cleared(self) -> None:
        background = (20, 30, 40, 255)
        self.view.set_atlases(
            (
                TextureAtlasEntry(
                    "square",
                    "Square",
                    _rgba_array(background, width=160, height=160),
                ),
            )
        )
        original = self.view.preview_label.pixmap().toImage()
        mask = np.zeros((160, 160), dtype=np.uint8)
        mask[48:112, 48:112] = 255

        self.view.set_edit_mask(mask)
        mask[:, :] = 0

        self.assertFalse(self.view.edit_mask_enabled)
        self.assertEqual(self.view.preview_label.pixmap().toImage(), original)
        self.assertFalse(self.view.edit_mask_image.isNull())
        self.assertEqual(
            self.view.edit_mask_image.pixelColor(80, 80).red(),
            255,
        )

        self.view.set_edit_mask_enabled(True)
        overlaid = self.view.preview_label.pixmap().toImage()
        self.assertNotEqual(
            overlaid.pixelColor(
                overlaid.width() // 2,
                overlaid.height() // 2,
            ),
            QColor(*background),
        )
        self.assertEqual(overlaid.pixelColor(2, 2), QColor(*background))

        self.view.set_edit_mask_enabled(False)
        self.assertEqual(self.view.preview_label.pixmap().toImage(), original)
        self.view.set_edit_mask_enabled(True)
        self.view.clear()

        self.assertTrue(self.view.edit_mask_enabled)
        self.assertTrue(self.view.edit_mask_image.isNull())

    def test_edit_mask_is_composed_before_uv_wireframe(self) -> None:
        background = (20, 30, 40, 255)
        triangle = (
            ((0.2, 0.2), (0.8, 0.2), (0.2, 0.8)),
        )
        mask = np.zeros((160, 160), dtype=np.uint8)
        mask[24:136, 24:136] = 255
        self.view.set_atlases(
            (
                TextureAtlasEntry(
                    "square",
                    "Square",
                    _rgba_array(background, width=160, height=160),
                ),
            )
        )
        self.view.set_edit_mask(mask)
        self.view.set_edit_mask_enabled(True)
        self.view.set_uv_overlay_triangles(triangle)
        self.view.set_uv_overlay_enabled(True)

        composed = self.view.preview_label.pixmap().toImage()

        center_color = composed.pixelColor(
            composed.width() // 2,
            composed.height() // 2,
        )
        self.assertNotEqual(center_color, QColor(*background))
        self.assertNotEqual(center_color, EDIT_MASK_OVERLAY_COLOR)
        for uv in triangle[0]:
            with self.subTest(vertex=uv):
                self.assertTrue(
                    _image_has_color_near_uv(
                        composed,
                        uv,
                        UV_VERTEX_COLOR,
                    )
                )

    def test_resize_storm_defers_one_overlay_until_the_final_size(self) -> None:
        background = (20, 30, 40, 255)
        triangle = (
            ((0.2, 0.2), (0.8, 0.2), (0.2, 0.8)),
        )
        self.view.set_uv_overlay_triangles(triangle)
        self.view.set_atlases(
            (
                TextureAtlasEntry(
                    "square",
                    "Square",
                    _rgba_array(background, width=160, height=160),
                ),
            )
        )
        self.view.set_uv_overlay_enabled(True)
        preview = self.view.preview_label

        with patch(
            "housemaker.texture_atlas_view._compose_uv_overlay",
            wraps=_compose_uv_overlay,
        ) as compose_overlay:
            preview.resize(preview.width() - 10, preview.height() - 10)
            first_base = preview.pixmap().toImage()

            self.assertTrue(self.view.uv_overlay_render_pending)
            self.assertTrue(preview._uv_overlay_timer.isActive())
            self.assertTrue(preview._uv_overlay_timer.isSingleShot())
            self.assertEqual(compose_overlay.call_count, 0)
            self.assertFalse(
                _image_has_color_near_uv(
                    first_base,
                    triangle[0][0],
                    UV_VERTEX_COLOR,
                )
            )

            preview.resize(preview.width() - 10, preview.height() - 5)
            final_base_size = preview.pixmap().size()

            self.assertTrue(self.view.uv_overlay_render_pending)
            self.assertTrue(preview._uv_overlay_timer.isActive())
            self.assertEqual(compose_overlay.call_count, 0)

            preview._uv_overlay_timer.timeout.emit()

            self.assertFalse(self.view.uv_overlay_render_pending)
            self.assertFalse(preview._uv_overlay_timer.isActive())
            self.assertEqual(compose_overlay.call_count, 1)
            self.assertEqual(
                compose_overlay.call_args.args[0].size(),
                final_base_size,
            )
            self.assertTrue(
                _image_has_color_near_uv(
                    preview.pixmap().toImage(),
                    triangle[0][0],
                    UV_VERTEX_COLOR,
                )
            )

            self.view.flush_pending_uv_overlay()
            self.assertEqual(compose_overlay.call_count, 1)

    def test_pending_overlay_never_paints_stale_content(self) -> None:
        old_background = (20, 30, 40, 255)
        new_background = (80, 100, 120, 255)
        old_triangle = (
            ((0.1, 0.1), (0.3, 0.1), (0.1, 0.3)),
        )
        new_triangle = (
            ((0.7, 0.7), (0.9, 0.7), (0.7, 0.9)),
        )
        self.view.set_uv_overlay_triangles(old_triangle)
        self.view.set_atlases(
            (
                TextureAtlasEntry(
                    "square",
                    "Square",
                    _rgba_array(old_background, width=160, height=160),
                ),
            )
        )
        self.view.set_uv_overlay_enabled(True)
        preview = self.view.preview_label

        with patch(
            "housemaker.texture_atlas_view._compose_uv_overlay",
            wraps=_compose_uv_overlay,
        ) as compose_overlay:
            preview.resize(preview.width() - 10, preview.height() - 10)
            self.assertTrue(self.view.uv_overlay_render_pending)

            self.view.set_uv_overlay_triangles(new_triangle)

            self.assertFalse(self.view.uv_overlay_render_pending)
            self.assertFalse(preview._uv_overlay_timer.isActive())
            self.assertEqual(compose_overlay.call_count, 1)
            updated_uv_image = preview.pixmap().toImage()
            self.assertFalse(
                _image_has_color_near_uv(
                    updated_uv_image,
                    old_triangle[0][0],
                    UV_VERTEX_COLOR,
                )
            )
            self.assertTrue(
                _image_has_color_near_uv(
                    updated_uv_image,
                    new_triangle[0][0],
                    UV_VERTEX_COLOR,
                )
            )
            self.view.flush_pending_uv_overlay()
            self.assertEqual(compose_overlay.call_count, 1)

            preview.resize(preview.width() - 10, preview.height() - 5)
            self.assertTrue(self.view.uv_overlay_render_pending)
            replacement = TextureAtlasEntry(
                "square",
                "Replacement",
                _rgba_array(new_background, width=160, height=160),
            )

            self.view.set_atlases((replacement,))

            self.assertFalse(self.view.uv_overlay_render_pending)
            self.assertFalse(preview._uv_overlay_timer.isActive())
            self.assertEqual(compose_overlay.call_count, 2)
            replacement_image = compose_overlay.call_args.args[0].toImage()
            self.assertEqual(
                replacement_image.pixelColor(
                    replacement_image.width() // 2,
                    replacement_image.height() // 2,
                ),
                QColor(*new_background),
            )

            preview.resize(preview.width() - 10, preview.height() - 5)
            self.assertTrue(self.view.uv_overlay_render_pending)
            self.view.clear()

            self.assertFalse(self.view.uv_overlay_render_pending)
            self.assertFalse(preview._uv_overlay_timer.isActive())
            self.assertTrue(preview.pixmap().isNull())
            self.view.flush_pending_uv_overlay()
            self.assertEqual(compose_overlay.call_count, 2)

    def test_same_size_toggle_and_source_change_reuse_scaled_geometry(self) -> None:
        first_triangle = (
            ((0.1, 0.1), (0.9, 0.1), (0.1, 0.9)),
        )
        second_triangle = (
            ((0.2, 0.2), (0.8, 0.2), (0.2, 0.8)),
        )
        self.view.set_uv_overlay_triangles(first_triangle)
        self.view.set_atlases(
            (
                TextureAtlasEntry(
                    "square",
                    "Square",
                    _rgba_array(width=160, height=160),
                ),
            )
        )
        preview = self.view.preview_label

        with patch(
            "housemaker.texture_atlas_view._compose_uv_overlay",
            wraps=_compose_uv_overlay,
        ) as compose_overlay:
            self.view.set_uv_overlay_enabled(True)
            first_scaled_geometry = compose_overlay.call_args.args[1]

            self.view.set_uv_overlay_enabled(False)
            self.view.set_uv_overlay_enabled(True)
            toggled_scaled_geometry = compose_overlay.call_args.args[1]

            self.assertEqual(compose_overlay.call_count, 2)
            self.assertIs(toggled_scaled_geometry, first_scaled_geometry)

            replacement = TextureAtlasEntry(
                "square",
                "Replacement",
                _rgba_array((60, 70, 80, 255), width=160, height=160),
            )
            self.view.set_atlases((replacement,))
            source_scaled_geometry = compose_overlay.call_args.args[1]

            self.assertEqual(compose_overlay.call_count, 3)
            self.assertIs(source_scaled_geometry, first_scaled_geometry)

            self.view.set_uv_overlay_triangles(second_triangle)
            model_scaled_geometry = compose_overlay.call_args.args[1]

            self.assertEqual(compose_overlay.call_count, 4)
            self.assertIsNot(model_scaled_geometry, first_scaled_geometry)

            preview.resize(preview.width() - 12, preview.height() - 12)
            self.assertTrue(self.view.uv_overlay_render_pending)
            self.assertEqual(compose_overlay.call_count, 4)
            self.view.flush_pending_uv_overlay()
            resized_scaled_geometry = compose_overlay.call_args.args[1]

            self.assertEqual(compose_overlay.call_count, 5)
            self.assertIsNot(resized_scaled_geometry, model_scaled_geometry)

    def test_uv_overlay_batches_unique_edges_and_vertices(self) -> None:
        with patch("housemaker.texture_atlas_view.QPainter") as painter_type:
            _paint_uv_overlay(
                QPixmap(120, 120),
                (
                    ((0.1, 0.1), (0.9, 0.1), (0.9, 0.9)),
                    ((0.1, 0.1), (0.9, 0.9), (0.1, 0.9)),
                ),
            )

        painter = painter_type.return_value
        self.assertEqual(painter.drawLines.call_count, 2)
        self.assertEqual(painter.drawPoints.call_count, 2)

        line_batches = [
            call.args[0]
            for call in painter.drawLines.call_args_list
        ]
        point_batches = [
            call.args[0]
            for call in painter.drawPoints.call_args_list
        ]
        self.assertEqual([len(lines) for lines in line_batches], [5, 5])
        self.assertEqual([len(points) for points in point_batches], [4, 4])

        def _line_coordinates(lines) -> set[tuple[tuple[float, float], ...]]:
            coordinates = set()
            for line in lines:
                endpoints = tuple(
                    sorted(
                        (
                            (line.p1().x(), line.p1().y()),
                            (line.p2().x(), line.p2().y()),
                        )
                    )
                )
                coordinates.add(endpoints)
            return coordinates

        def _point_coordinates(points) -> set[tuple[float, float]]:
            return {(point.x(), point.y()) for point in points}

        self.assertEqual(
            _line_coordinates(line_batches[0]),
            _line_coordinates(line_batches[1]),
        )
        self.assertEqual(len(_line_coordinates(line_batches[0])), 5)
        self.assertEqual(
            _point_coordinates(point_batches[0]),
            _point_coordinates(point_batches[1]),
        )
        self.assertEqual(len(_point_coordinates(point_batches[0])), 4)

    def test_large_uv_overlay_remains_batched(self) -> None:
        triangle_count = 15_000
        triangles = tuple(
            (
                (float(index * 3), 0.0),
                (float(index * 3 + 1), 0.0),
                (float(index * 3), 1.0),
            )
            for index in range(triangle_count)
        )

        with patch("housemaker.texture_atlas_view.QPainter") as painter_type:
            _paint_uv_overlay(QPixmap(2, 2), triangles)

        painter = painter_type.return_value
        self.assertEqual(painter.drawLines.call_count, 2)
        self.assertEqual(painter.drawPoints.call_count, 2)

        line_batches = [
            call.args[0]
            for call in painter.drawLines.call_args_list
        ]
        point_batches = [
            call.args[0]
            for call in painter.drawPoints.call_args_list
        ]
        self.assertIs(line_batches[0], line_batches[1])
        self.assertIs(point_batches[0], point_batches[1])
        self.assertEqual(len(line_batches[0]), triangle_count * 3)
        self.assertEqual(len(point_batches[0]), triangle_count * 3)


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
