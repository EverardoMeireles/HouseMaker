# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

import housemaker.blueprint_canvas as blueprint_canvas_module
from housemaker.blueprint_canvas import BlueprintCanvas
from housemaker.models import VertexData


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Blueprint image revision tests ###
class BlueprintImageRefreshTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.image_path = Path(self.temporary_directory.name) / "blueprint.png"
        Image.new("RGB", (32, 24), (20, 40, 60)).save(self.image_path)
        self.canvas = BlueprintCanvas()

    def tearDown(self) -> None:
        self.canvas.close()
        self.canvas.deleteLater()
        self.temporary_directory.cleanup()
        _qt_application.processEvents()

    def test_refresh_preserves_geometry_view_and_undo_state(self) -> None:
        vertex_data = VertexData()
        vertex_data.add_vertex(10.0, 12.0)
        self.canvas.load_blueprint(str(self.image_path), vertex_data=vertex_data)
        self.canvas._push_undo_state()
        self.canvas.zoom_scale = 1.75
        self.canvas.view_offset = QPointF(17.0, 23.0)
        previous_stat = self.image_path.stat()
        Image.new("RGB", (48, 20), (180, 40, 20)).save(self.image_path)
        os.utime(
            self.image_path,
            ns=(
                previous_stat.st_atime_ns,
                previous_stat.st_mtime_ns + 1_000_000,
            ),
        )

        self.assertTrue(self.canvas.refresh_blueprint_image_if_stale())

        self.assertIs(self.canvas.vertex_data, vertex_data)
        self.assertEqual(len(self.canvas.undo_stack), 1)
        self.assertEqual(self.canvas.zoom_scale, 1.75)
        self.assertEqual(self.canvas.view_offset, QPointF(17.0, 23.0))
        self.assertEqual(self.canvas.get_image_size_pixels(), (48.0, 20.0))
        self.canvas.resize(640, 480)
        base_rect = self.canvas._base_image_display_rect()
        self.assertAlmostEqual(base_rect.left(), 16.0)
        self.assertAlmostEqual(base_rect.width(), 608.0)
        self.assertAlmostEqual(base_rect.height(), 608.0 / 2.4)
        self.assertEqual(base_rect.center(), QPointF(320.0, 240.0))
        self.assertFalse(hasattr(self.canvas, "image_scale"))
        self.assertFalse(hasattr(self.canvas, "image_offset_x"))
        self.assertFalse(hasattr(self.canvas, "image_offset_y"))
        self.assertFalse(hasattr(self.canvas, "set_image_transform"))
        display_rect = self.canvas._image_display_rect()
        self.assertAlmostEqual(display_rect.width(), base_rect.width() * 1.75)
        self.assertAlmostEqual(display_rect.height(), base_rect.height() * 1.75)
        self.assertEqual(
            display_rect.center(),
            base_rect.center() + QPointF(17.0, 23.0),
        )

    def test_failed_decode_retries_at_the_same_revision(self) -> None:
        self.canvas.blueprint_path = str(self.image_path)
        real_loader = blueprint_canvas_module._load_qimage_from_path
        attempts = 0

        def load_after_failure(file_path: str):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ValueError("temporary decode failure")
            return real_loader(file_path)

        with patch(
            "housemaker.blueprint_canvas._load_qimage_from_path",
            side_effect=load_after_failure,
        ):
            self.assertFalse(self.canvas.refresh_blueprint_image_if_stale())
            self.assertTrue(self.canvas.refresh_blueprint_image_if_stale())

        self.assertEqual(attempts, 2)
        self.assertEqual(self.canvas.get_image_size_pixels(), (32.0, 24.0))

    def test_initial_decode_is_not_bound_to_a_later_file_revision(self) -> None:
        normalized_path = str(self.image_path.resolve())
        revision_before = (normalized_path, 10, 20, 30)
        revision_after = (normalized_path, 11, 21, 31)

        with patch(
            "housemaker.blueprint_canvas._build_blueprint_image_revision",
            side_effect=(revision_before, revision_after),
        ):
            self.canvas.load_blueprint(str(self.image_path))

        self.assertIsNone(self.canvas._blueprint_image_revision)
        with (
            patch(
                "housemaker.blueprint_canvas._build_blueprint_image_revision",
                return_value=revision_after,
            ),
            patch(
                "housemaker.blueprint_canvas._load_qimage_from_path",
                wraps=blueprint_canvas_module._load_qimage_from_path,
            ) as load_image,
        ):
            self.assertTrue(self.canvas.refresh_blueprint_image_if_stale())

        load_image.assert_called_once_with(str(self.image_path))
        self.assertEqual(
            self.canvas._blueprint_image_revision,
            revision_after,
        )

    def test_refresh_does_not_bind_pixels_to_a_mid_decode_replacement(
        self,
    ) -> None:
        self.canvas.load_blueprint(str(self.image_path))
        validated_revision = self.canvas.get_blueprint_image_revision()
        revision_before = (str(self.image_path.resolve()), 11, 21, 31)
        revision_after = (str(self.image_path.resolve()), 12, 22, 32)

        with (
            patch(
                "housemaker.blueprint_canvas._build_blueprint_image_revision",
                side_effect=(revision_before, revision_after),
            ),
            patch(
                "housemaker.blueprint_canvas._load_qimage_from_path",
                wraps=blueprint_canvas_module._load_qimage_from_path,
            ),
        ):
            self.assertFalse(self.canvas.refresh_blueprint_image_if_stale())

        self.assertEqual(
            self.canvas.get_blueprint_image_revision(),
            validated_revision,
        )

        with (
            patch(
                "housemaker.blueprint_canvas._build_blueprint_image_revision",
                side_effect=(revision_after, revision_after),
            ),
            patch(
                "housemaker.blueprint_canvas._load_qimage_from_path",
                wraps=blueprint_canvas_module._load_qimage_from_path,
            ) as load_image,
        ):
            self.assertTrue(self.canvas.refresh_blueprint_image_if_stale())

        load_image.assert_called_once_with(str(self.image_path))
        self.assertEqual(
            self.canvas.get_blueprint_image_revision(),
            revision_after,
        )

    def test_known_missing_blueprint_revision_is_cached(self) -> None:
        missing_path = self.image_path.parent / "missing.png"
        self.canvas.set_level_data(
            vertex_data=VertexData(),
            rooms=[],
            image_path=str(missing_path),
        )
        cached_revision = self.canvas._blueprint_image_revision
        self.assertIsNotNone(cached_revision)

        with patch(
            "housemaker.blueprint_canvas._load_qimage_from_path",
            wraps=blueprint_canvas_module._load_qimage_from_path,
        ) as load_image:
            self.assertFalse(self.canvas.refresh_blueprint_image_if_stale())
            self.assertFalse(self.canvas.refresh_blueprint_image_if_stale())

        load_image.assert_not_called()
        Image.new("RGB", (20, 12), (120, 60, 30)).save(missing_path)
        self.assertTrue(self.canvas.refresh_blueprint_image_if_stale())
        self.assertEqual(self.canvas.get_image_size_pixels(), (20.0, 12.0))


# ### Canvas API tests ###
class BlueprintCanvasApiTests(unittest.TestCase):
    def test_loading_api_has_no_retired_transform_parameters(self) -> None:
        retired_parameters = {
            "image_scale",
            "image_offset_x",
            "image_offset_y",
        }

        self.assertTrue(
            retired_parameters.isdisjoint(
                inspect.signature(BlueprintCanvas.load_blueprint).parameters
            )
        )
        self.assertTrue(
            retired_parameters.isdisjoint(
                inspect.signature(BlueprintCanvas.set_level_data).parameters
            )
        )
        self.assertFalse(hasattr(BlueprintCanvas, "set_image_transform"))


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
