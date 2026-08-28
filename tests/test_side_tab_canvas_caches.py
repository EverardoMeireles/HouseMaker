# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from PySide6.QtWidgets import QApplication

import housemaker.texture_creator_canvas as texture_creator_canvas_module
import housemaker.uv_canvas as uv_canvas_module
from housemaker.models import RoomData, VertexData
from housemaker.texture_creator_canvas import TextureCreatorCanvas
from housemaker.uv_canvas import UvCanvas, _build_uv_layout_cache_key


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_room() -> tuple[RoomData, VertexData]:
    vertex_data = VertexData()
    boundary_ids = tuple(
        vertex_data.add_vertex(*point).id
        for point in (
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (0.0, 100.0),
        )
    )
    for start_id, end_id in zip(
        boundary_ids,
        (*boundary_ids[1:], boundary_ids[0]),
    ):
        vertex_data.add_edge(start_id, end_id)
    center = vertex_data.add_vertex(50.0, 50.0)
    room = RoomData(
        name="Room",
        vertex_ids=boundary_ids,
        center_vertex_id=center.id,
        color_rgb=(120, 160, 200),
    )
    return room, vertex_data


def _write_png(path: Path, size: tuple[int, int]) -> None:
    Image.new("RGBA", size, (80, 120, 180, 255)).save(path, format="PNG")


# ### UV layout cache tests ###
class UvCanvasLayoutCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas = UvCanvas()

    def tearDown(self) -> None:
        self.canvas.close()
        self.canvas.deleteLater()
        _qt_application.processEvents()

    def test_reuses_layout_for_unchanged_and_equivalent_contexts(self) -> None:
        room, vertex_data = _build_room()
        self.canvas.resize(640, 480)
        self.canvas.set_room_context(room, vertex_data, 3.0)

        with patch(
            "housemaker.uv_canvas.build_uv_wall_layout",
            wraps=uv_canvas_module.build_uv_wall_layout,
        ) as build_layout:
            first_render = self.canvas.grab()
            first_layout = self.canvas._get_wall_layout()
            second_render = self.canvas.grab()
            second_layout = self.canvas._get_wall_layout()
            self.canvas.set_selected_wall_key("1:2")
            equivalent_room, equivalent_vertices = _build_room()
            self.canvas.set_room_context(
                equivalent_room,
                equivalent_vertices,
                3.0,
            )
            equivalent_layout = self.canvas._get_wall_layout()

        self.assertFalse(first_render.isNull())
        self.assertFalse(second_render.isNull())
        self.assertIs(second_layout, first_layout)
        self.assertIs(equivalent_layout, first_layout)
        self.assertEqual(self.canvas.get_selected_wall_key(), "1:2")
        build_layout.assert_called_once()

    def test_rebuilds_after_in_place_geometry_uv_and_height_changes(self) -> None:
        room, vertex_data = _build_room()
        self.canvas.set_room_context(room, vertex_data, 3.0)

        with patch(
            "housemaker.uv_canvas.build_uv_wall_layout",
            wraps=uv_canvas_module.build_uv_wall_layout,
        ) as build_layout:
            self.canvas._get_wall_layout()
            room.wall_uv_scales["1:2"] = 1.25
            self.canvas._get_wall_layout()
            vertex_data.move_vertex(2, 120.0, 0.0)
            self.canvas._get_wall_layout()
            vertex_data.move_vertex(room.center_vertex_id, 60.0, 50.0)
            self.canvas._get_wall_layout()
            self.canvas.set_room_context(room, vertex_data, 3.5)
            self.canvas._get_wall_layout()
            room.center_vertex_id = room.vertex_ids[0]
            self.canvas._get_wall_layout()

        self.assertEqual(build_layout.call_count, 6)

    def test_cache_key_covers_every_nested_uv_mapping(self) -> None:
        room, vertex_data = _build_room()
        baseline = _build_uv_layout_cache_key(room, vertex_data, 3.0)
        mutations = (
            lambda: room.wall_uv_rotations.__setitem__("1:2", 90),
            lambda: room.wall_uv_positions.__setitem__("1:2", (12.0, 18.0)),
            lambda: room.wall_subdivisions.__setitem__("1:2", 2),
            lambda: room.wall_subdivision_positions.__setitem__(
                "1:2",
                ((8.0, 9.0), (18.0, 19.0)),
            ),
            lambda: room.wall_subdivision_source_ranges.__setitem__(
                "1:2",
                ((0.0, 0.5), (0.5, 1.0)),
            ),
        )

        for mutation in mutations:
            with self.subTest(mutation=mutation):
                mutation()
                next_key = _build_uv_layout_cache_key(room, vertex_data, 3.0)
                self.assertNotEqual(next_key, baseline)
                baseline = next_key


# ### Texture creator image cache tests ###
class TextureCreatorCanvasCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.image_path = Path(self.temporary_directory.name) / "source.png"
        _write_png(self.image_path, (32, 24))
        self.room, _vertex_data = _build_room()
        self.canvas = TextureCreatorCanvas()

    def tearDown(self) -> None:
        self.canvas.close()
        self.canvas.deleteLater()
        self.temporary_directory.cleanup()
        _qt_application.processEvents()

    def test_identical_context_reuses_decoded_image_and_initialization(self) -> None:
        changes: list[bool] = []
        self.canvas.texture_changed.connect(lambda: changes.append(True))

        with patch(
            "housemaker.texture_creator_canvas._load_source_image",
            wraps=texture_creator_canvas_module._load_source_image,
        ) as load_image:
            self.canvas.set_context(
                self.room,
                "1:2",
                (4.0, 3.0),
                str(self.image_path),
            )
            self.canvas.set_context(
                self.room,
                "1:2",
                (4.0, 3.0),
                str(self.image_path),
            )
            self.canvas.set_context(None, None, None, None)
            self.canvas.set_context(
                self.room,
                "1:2",
                (4.0, 3.0),
                str(self.image_path),
            )

        load_image.assert_called_once_with(str(self.image_path.resolve()))
        self.assertEqual(changes, [True])

    def test_file_stat_change_invalidates_decoded_image(self) -> None:
        with patch(
            "housemaker.texture_creator_canvas._load_source_image",
            wraps=texture_creator_canvas_module._load_source_image,
        ) as load_image:
            self.canvas.set_context(
                self.room,
                "1:2",
                (4.0, 3.0),
                str(self.image_path),
            )
            previous_stat = self.image_path.stat()
            _write_png(self.image_path, (48, 24))
            os.utime(
                self.image_path,
                ns=(
                    previous_stat.st_atime_ns,
                    previous_stat.st_mtime_ns + 1_000_000,
                ),
            )
            self.canvas.set_context(
                self.room,
                "1:2",
                (4.0, 3.0),
                str(self.image_path),
            )

        self.assertEqual(load_image.call_count, 2)
        self.assertEqual(self.canvas.source_image.width(), 48)
        self.assertEqual(self.canvas.source_image.height(), 24)

    def test_known_missing_image_revision_skips_repeat_decode(self) -> None:
        missing_path = self.image_path.parent / "missing.png"

        with patch(
            "housemaker.texture_creator_canvas._load_source_image",
            wraps=texture_creator_canvas_module._load_source_image,
        ) as load_image:
            self.canvas.set_context(
                self.room,
                "1:2",
                (4.0, 3.0),
                str(missing_path),
            )
            self.canvas.set_context(
                self.room,
                "1:2",
                (4.0, 3.0),
                str(missing_path),
            )

        load_image.assert_not_called()
        self.assertTrue(self.canvas.source_image.isNull())
        self.assertIsNotNone(self.canvas._decoded_image_cache_key)

    def test_failed_decode_retries_the_same_file_revision(self) -> None:
        decoded_image = texture_creator_canvas_module.QImage(
            str(self.image_path)
        )
        with patch(
            "housemaker.texture_creator_canvas._load_source_image",
            side_effect=(
                texture_creator_canvas_module.QImage(),
                decoded_image,
            ),
        ) as load_image:
            self.canvas.set_context(
                self.room,
                "1:2",
                (4.0, 3.0),
                str(self.image_path),
            )
            self.assertTrue(self.canvas.source_image.isNull())
            self.canvas.set_context(
                self.room,
                "1:2",
                (4.0, 3.0),
                str(self.image_path),
            )

        self.assertEqual(load_image.call_count, 2)
        self.assertFalse(self.canvas.source_image.isNull())
        self.assertEqual(self.canvas.source_image.size(), decoded_image.size())
        self.assertIn("1:2", self.room.wall_textures)


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
