# ### Environment setup ###
from __future__ import annotations

import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import (
    QDragEnterEvent,
    QDragLeaveEvent,
    QDragMoveEvent,
    QDropEvent,
    QImage,
    QMouseEvent,
    QPainter,
    QWheelEvent,
)
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.texture_atlas_state import (
    ATLAS_PACKING_MODE_SYMMETRIC_HALF,
    ATLAS_PACKING_MODE_SYMMETRIC_PAIR,
    ATLAS_PACKING_MODE_SYMMETRIC_QUARTER,
    ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR,
    ATLAS_SLOT_HALF_LEFT,
    ATLAS_SLOT_HALF_RIGHT,
    ATLAS_SLOT_QUADRANT_BOTTOM_LEFT,
    ATLAS_SLOT_QUADRANT_ORDER,
    ATLAS_SLOT_QUADRANT_TOP_LEFT,
    ATLAS_SLOT_QUADRANT_TOP_RIGHT,
    TextureAtlasData,
)
from housemaker.texture_atlas_workspace import (
    AtlasObjectTextureSource,
    TextureAtlasWorkspace,
    _build_texture_source_mime_data,
    build_atlas_wall_texture_source_id,
    choose_atlas_texture_resolution,
    get_atlas_wall_texture_assignment_id,
    load_atlas_object_texture_source,
)


# ### Test application ###
_qt_application = QApplication.instance() or QApplication([])


# ### Fixture helpers ###
def _source(
    object_id: str,
    *,
    directory: str | Path,
    resolution: int = 512,
    color: tuple[int, int, int, int] = (30, 120, 210, 255),
    symmetric_orientation: str | None = None,
    symmetric_plane_coordinate: float | None = None,
    packing_mode: str | None = None,
) -> AtlasObjectTextureSource:
    pixels = np.empty((8, 8, 4), dtype=np.uint8)
    pixels[:, :] = np.asarray(color, dtype=np.uint8)
    texture_path = Path(directory) / f"{object_id}-{resolution}.png"
    resolved_packing_mode = (
        packing_mode
        if packing_mode is not None
        else (
            ATLAS_PACKING_MODE_SYMMETRIC_HALF
            if symmetric_orientation is not None
            else "full"
        )
    )
    physical_resolution = (
        resolution * 2
        if resolved_packing_mode
        in {
            ATLAS_PACKING_MODE_SYMMETRIC_QUARTER,
            ATLAS_PACKING_MODE_SYMMETRIC_PAIR,
        }
        else resolution
    )
    exact_pixels = np.empty(
        (physical_resolution, physical_resolution, 4),
        dtype=np.uint8,
    )
    exact_pixels[:, :] = np.asarray(color, dtype=np.uint8)
    Image.fromarray(exact_pixels, mode="RGBA").save(texture_path)
    return AtlasObjectTextureSource(
        object_id=object_id,
        object_name=object_id.title(),
        texture_path=f"textures/{object_id}-{resolution}.png",
        texture_resolution=resolution,
        physical_texture_path=texture_path,
        preview_rgba=pixels,
        packing_mode=resolved_packing_mode,
        symmetric_preview_orientation=symmetric_orientation,
        symmetric_preview_plane_coordinate=symmetric_plane_coordinate,
    )


def _variant_resolver(
    variants: dict[tuple[str, int], AtlasObjectTextureSource],
):
    return lambda object_id, resolution: variants.get(
        (object_id, resolution)
    )


def _wall_source(
    assignment_id: str,
    *,
    directory: str | Path,
    size: tuple[int, int] = (12, 8),
    color: tuple[int, int, int, int] = (180, 80, 30, 255),
) -> AtlasObjectTextureSource:
    texture_path = Path(directory) / f"wall-{assignment_id}.png"
    pixels = np.empty((size[1], size[0], 4), dtype=np.uint8)
    pixels[:, :] = np.asarray(color, dtype=np.uint8)
    Image.fromarray(pixels, mode="RGBA").save(texture_path)
    resolution = choose_atlas_texture_resolution(*size)
    return load_atlas_object_texture_source(
        object_id=build_atlas_wall_texture_source_id(assignment_id),
        object_name="Wall texture",
        texture_path=f"surface_textures/{texture_path.name}",
        texture_resolution=resolution,
        physical_texture_path=texture_path,
        fit_to_square=True,
        supports_resolution_changes=False,
        supports_3d_preview=False,
    )


def _resizable_wall_variants(
    assignment_id: str,
    *,
    directory: str | Path,
) -> dict[tuple[str, int], AtlasObjectTextureSource]:
    source_id = build_atlas_wall_texture_source_id(assignment_id)
    variants: dict[tuple[str, int], AtlasObjectTextureSource] = {}
    for resolution, color in (
        (512, (180, 80, 30, 255)),
        (1024, (80, 180, 30, 255)),
        (2048, (30, 80, 180, 255)),
    ):
        texture_path = Path(directory) / (
            f"wall-{assignment_id}-{resolution}.png"
        )
        pixels = np.empty((8, 12, 4), dtype=np.uint8)
        pixels[:, :] = np.asarray(color, dtype=np.uint8)
        Image.fromarray(pixels, mode="RGBA").save(texture_path)
        variants[(source_id, resolution)] = load_atlas_object_texture_source(
            object_id=source_id,
            object_name="Wall texture",
            texture_path=(
                f"surface_textures/wall-{assignment_id}-{resolution}.png"
            ),
            texture_resolution=resolution,
            physical_texture_path=texture_path,
            fit_to_square=True,
            supports_resolution_changes=True,
            supports_3d_preview=False,
        )
    return variants


def _wheel_event(position: QPointF, delta: int) -> QWheelEvent:
    return QWheelEvent(
        position,
        position,
        QPoint(),
        QPoint(0, int(delta)),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )


def _atlas_preview_point(
    preview,
    atlas_resolution: int,
    atlas_x: float,
    atlas_y: float,
) -> QPointF:
    """Map Atlas coordinates to the preview widget for drag tests."""

    preview_side = min(preview.width() - 32.0, preview.height() - 32.0)
    preview_origin = QPointF(
        (preview.width() - preview_side) / 2.0,
        (preview.height() - preview_side) / 2.0,
    )
    return QPointF(
        preview_origin.x() + atlas_x * preview_side / atlas_resolution,
        preview_origin.y() + atlas_y * preview_side / atlas_resolution,
    )


def _paint_drag_feedback(preview) -> QImage:
    """Paint only the transient slot feedback onto a transparent image."""

    preview_side = min(preview.width() - 32.0, preview.height() - 32.0)
    atlas_rect = QRectF(
        (preview.width() - preview_side) / 2.0,
        (preview.height() - preview_side) / 2.0,
        preview_side,
        preview_side,
    )
    image = QImage(
        preview.width(),
        preview.height(),
        QImage.Format.Format_ARGB32,
    )
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    preview._paint_drag_slot_preview(painter, atlas_rect)
    painter.end()
    return image


# ### Workspace tests ###
class TextureAtlasWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = TextureAtlasWorkspace(
            asset_directory=self._temporary_directory.name
        )
        self.workspace.resize(1100, 720)
        self.workspace.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.close()
        self.workspace.deleteLater()
        _qt_application.processEvents()
        self._temporary_directory.cleanup()

    def test_wall_source_identity_round_trips_assignment_id(self) -> None:
        source_id = build_atlas_wall_texture_source_id("plaster:variant-a")

        self.assertEqual(
            get_atlas_wall_texture_assignment_id(source_id),
            "plaster:variant-a",
        )
        self.assertIsNone(get_atlas_wall_texture_assignment_id("chair"))
        self.assertIsNone(
            get_atlas_wall_texture_assignment_id("surface-wall-texture:")
        )

    def test_creates_named_atlas_with_selected_resolution(self) -> None:
        self.workspace.atlas_name_edit.setText("Furniture")
        self.workspace.atlas_resolution_combo.setCurrentIndex(
            self.workspace.atlas_resolution_combo.findData(4096)
        )

        self.workspace.create_atlas_button.click()

        data = self.workspace.get_data()
        self.assertEqual(len(data.atlases), 1)
        self.assertEqual(data.atlases[0].name, "Furniture")
        self.assertEqual(data.atlases[0].resolution, 4096)
        self.assertEqual(data.selected_atlas_id, data.atlases[0].atlas_id)
        self.assertIsNone(data.atlases[0].image_path)
        self.assertEqual(
            list(Path(self._temporary_directory.name).glob("*.png")),
            [],
        )
        self.assertEqual(self.workspace.atlas_list.count(), 1)
        self.assertIn("Furniture", self.workspace.status_label.text())

    def test_adds_active_object_texture_and_uses_core_quadtree_packing(self) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Packed", 4096, atlas_id="atlas-a")
        self.workspace.set_data(data)
        sources = [
            _source(
                f"chair-{index}",
                directory=self._temporary_directory.name,
            )
            for index in range(4)
        ]
        self.workspace.set_object_texture_sources(sources)

        for row in range(4):
            self.workspace.object_list.setCurrentRow(row)
            self.workspace.assign_object_button.click()

        packed = self.workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert packed is not None
        self.assertEqual(
            [(item.x, item.y) for item in packed.placements],
            [(0, 0), (512, 0), (0, 512), (512, 512)],
        )
        self.assertTrue(
            all(item.texture_resolution == 512 for item in packed.placements)
        )
        assert packed.image_path is not None
        with Image.open(
            Path(self._temporary_directory.name) / packed.image_path
        ) as atlas_image:
            self.assertEqual(atlas_image.getpixel((10, 10)), (30, 120, 210, 255))

    def test_adds_rectangular_wall_texture_without_stretching_it(self) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Walls", 2048, atlas_id="atlas-a")
        wall_source = _wall_source(
            "brick",
            directory=self._temporary_directory.name,
        )
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources([wall_source])

        self.workspace.assign_object_button.click()

        packed = self.workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert packed is not None
        placement = packed.placement_for_object(wall_source.object_id)
        assert placement is not None
        self.assertEqual(placement.texture_resolution, 512)
        self.assertTrue(
            placement.object_id.startswith("surface-wall-texture:")
        )
        assert packed.image_path is not None
        with Image.open(
            Path(self._temporary_directory.name) / packed.image_path
        ) as atlas_image:
            self.assertEqual(atlas_image.getpixel((256, 256)), (180, 80, 30, 255))
            self.assertEqual(atlas_image.getpixel((256, 20)), (0, 0, 0, 0))

    def test_wall_texture_click_only_selects_and_clears_3d_preview(
        self,
    ) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Walls", 2048, atlas_id="atlas-a")
        wall_source = _wall_source(
            "plaster",
            directory=self._temporary_directory.name,
        )
        data.assign_object(
            atlas.atlas_id,
            wall_source.object_id,
            wall_source.texture_path,
            wall_source.texture_resolution,
        )
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources([wall_source])
        preview_requests: list[tuple[str, int]] = []
        resolution_changes: list[tuple[str, int]] = []
        clear_requests: list[bool] = []
        self.workspace.object_preview_requested.connect(
            lambda source_id, resolution: preview_requests.append(
                (source_id, resolution)
            )
        )
        self.workspace.object_texture_resolution_changed.connect(
            lambda source_id, resolution: resolution_changes.append(
                (source_id, resolution)
            )
        )
        self.workspace.object_preview_clear_requested.connect(
            lambda: clear_requests.append(True)
        )

        self.workspace._handle_object_mouse_click(
            wall_source.object_id,
            Qt.MouseButton.RightButton,
        )

        placement = self.workspace.get_data().atlas_by_id(
            atlas.atlas_id
        ).placement_for_object(wall_source.object_id)
        assert placement is not None
        self.assertEqual(placement.texture_resolution, 512)
        self.assertEqual(preview_requests, [])
        self.assertEqual(resolution_changes, [])
        self.assertEqual(clear_requests, [True])
        self.assertIn(
            "Selected the texture",
            self.workspace.status_label.text(),
        )

    def test_resizable_wall_texture_wheel_emits_global_assignment(self) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Walls", 2048, atlas_id="atlas-a")
        variants = _resizable_wall_variants(
            "plaster",
            directory=self._temporary_directory.name,
        )
        source_id = build_atlas_wall_texture_source_id("plaster")
        data.assign_object(
            atlas.atlas_id,
            source_id,
            variants[(source_id, 512)].texture_path,
            512,
        )
        selectability_checks: list[tuple[str, int]] = []
        resolution_changes: list[tuple[str, int]] = []
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources(
            [variants[(source_id, 512)]],
            variant_resolver=_variant_resolver(variants),
            selectability_resolver=lambda object_id, resolution: (
                selectability_checks.append((object_id, resolution)) or True
            ),
        )
        self.workspace.object_texture_resolution_changed.connect(
            lambda object_id, resolution: resolution_changes.append(
                (object_id, resolution)
            )
        )

        self.workspace._handle_object_wheel(source_id, 1)

        placement = self.workspace.get_data().atlas_by_id(
            atlas.atlas_id
        ).placement_for_object(source_id)
        assert placement is not None
        self.assertEqual(placement.texture_resolution, 1024)
        self.assertEqual(
            placement.texture_path,
            variants[(source_id, 1024)].texture_path,
        )
        self.assertEqual(selectability_checks, [(source_id, 1024)])
        self.assertEqual(resolution_changes, [(source_id, 1024)])
        self.assertIn("1024 x 1024", self.workspace.object_list.item(0).text())

    def test_mouse_wheel_resizes_wall_texture_from_list_and_atlas(self) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Walls", 2048, atlas_id="atlas-a")
        variants = _resizable_wall_variants(
            "brick",
            directory=self._temporary_directory.name,
        )
        source_id = build_atlas_wall_texture_source_id("brick")
        data.assign_object(
            atlas.atlas_id,
            source_id,
            variants[(source_id, 512)].texture_path,
            512,
        )
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources(
            [variants[(source_id, 512)]],
            variant_resolver=_variant_resolver(variants),
            selectability_resolver=lambda _object_id, _resolution: True,
        )
        resolution_changes: list[tuple[str, int]] = []
        self.workspace.object_texture_resolution_changed.connect(
            lambda object_id, resolution: resolution_changes.append(
                (object_id, resolution)
            )
        )
        _qt_application.processEvents()

        row_center = QPointF(
            self.workspace.object_list.visualItemRect(
                self.workspace.object_list.item(0)
            ).center()
        )
        list_wheel = _wheel_event(row_center, 120)
        self.workspace.object_list.wheelEvent(list_wheel)

        enlarged = self.workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert enlarged is not None
        placement = enlarged.placement_for_object(source_id)
        assert placement is not None
        self.assertTrue(list_wheel.isAccepted())
        self.assertEqual(placement.texture_resolution, 1024)

        preview = self.workspace.preview
        preview_side = min(preview.width() - 32.0, preview.height() - 32.0)
        preview_origin = QPointF(
            (preview.width() - preview_side) / 2.0,
            (preview.height() - preview_side) / 2.0,
        )
        placement_center = QPointF(
            preview_origin.x()
            + (placement.x + placement.size / 2.0)
            * preview_side
            / atlas.resolution,
            preview_origin.y()
            + (placement.y + placement.size / 2.0)
            * preview_side
            / atlas.resolution,
        )
        preview_wheel = _wheel_event(placement_center, -120)
        preview.wheelEvent(preview_wheel)

        restored = self.workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert restored is not None
        placement = restored.placement_for_object(source_id)
        assert placement is not None
        self.assertTrue(preview_wheel.isAccepted())
        self.assertEqual(placement.texture_resolution, 512)
        self.assertEqual(
            resolution_changes,
            [(source_id, 1024), (source_id, 512)],
        )
        self.assertEqual(self.workspace.selected_object_id, source_id)

    def test_mouse_wheel_resizes_selection_instead_of_hovered_texture(
        self,
    ) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Selected", 4096, atlas_id="atlas-a")
        variants: dict[tuple[str, int], AtlasObjectTextureSource] = {}
        for object_id in ("selected", "hovered"):
            for resolution in (512, 1024, 2048):
                variants[(object_id, resolution)] = _source(
                    object_id,
                    directory=self._temporary_directory.name,
                    resolution=resolution,
                )
            source = variants[(object_id, 512)]
            data.assign_object(
                atlas.atlas_id,
                object_id,
                source.texture_path,
                source.texture_resolution,
            )
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources(
            [variants[("selected", 512)], variants[("hovered", 512)]],
            variant_resolver=_variant_resolver(variants),
            selectability_resolver=lambda _object_id, _resolution: True,
        )
        self.workspace.object_list.setCurrentRow(0)
        _qt_application.processEvents()

        hovered_row_center = QPointF(
            self.workspace.object_list.visualItemRect(
                self.workspace.object_list.item(1)
            ).center()
        )
        list_wheel = _wheel_event(hovered_row_center, 120)
        self.workspace.object_list.wheelEvent(list_wheel)

        updated = self.workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert updated is not None
        selected = updated.placement_for_object("selected")
        hovered = updated.placement_for_object("hovered")
        assert selected is not None and hovered is not None
        self.assertEqual(selected.texture_resolution, 1024)
        self.assertEqual(hovered.texture_resolution, 512)
        self.assertEqual(self.workspace.selected_object_id, "selected")

        preview_wheel = _wheel_event(
            QPointF(self.workspace.preview.rect().center()),
            120,
        )
        self.workspace.preview.wheelEvent(preview_wheel)

        updated = self.workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert updated is not None
        selected = updated.placement_for_object("selected")
        hovered = updated.placement_for_object("hovered")
        assert selected is not None and hovered is not None
        self.assertTrue(preview_wheel.isAccepted())
        self.assertEqual(selected.texture_resolution, 2048)
        self.assertEqual(hovered.texture_resolution, 512)

        before = self.workspace.get_data()
        self.workspace.object_list.setCurrentRow(-1)
        unselected_wheel = _wheel_event(
            QPointF(self.workspace.preview.rect().center()),
            -120,
        )
        unselected_wheel.setAccepted(False)
        self.workspace.preview.wheelEvent(unselected_wheel)

        self.assertFalse(unselected_wheel.isAccepted())
        self.assertEqual(self.workspace.get_data(), before)

    def test_dragged_source_is_placed_at_the_exact_preview_grid_slot(self) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Manual", 2048, atlas_id="atlas-a")
        source = _source(
            "chair",
            directory=self._temporary_directory.name,
            resolution=512,
        )
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources([source])
        _qt_application.processEvents()

        preview = self.workspace.preview
        preview_side = min(preview.width() - 32.0, preview.height() - 32.0)
        preview_origin = QPointF(
            (preview.width() - preview_side) / 2.0,
            (preview.height() - preview_side) / 2.0,
        )
        drop_point = QPointF(
            preview_origin.x() + 1300.0 * preview_side / atlas.resolution,
            preview_origin.y() + 700.0 * preview_side / atlas.resolution,
        )
        slot = preview.atlas_slot_at(source.object_id, drop_point)

        self.assertEqual(slot, (1024, 512))
        self.assertTrue(
            bool(
                self.workspace.object_list.item(0).flags()
                & Qt.ItemFlag.ItemIsDragEnabled
            )
        )
        assert slot is not None
        mime_data = _build_texture_source_mime_data(source.object_id)
        drop_event = QDropEvent(
            drop_point,
            Qt.DropAction.CopyAction,
            mime_data,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        preview.dropEvent(drop_event)

        updated = self.workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert updated is not None
        placement = updated.placement_for_object(source.object_id)
        assert placement is not None
        self.assertTrue(drop_event.isAccepted())
        self.assertEqual((placement.x, placement.y), slot)
        self.assertEqual(placement.texture_resolution, 512)
        self.assertIsNotNone(updated.image_path)

    def test_large_drag_preview_and_drop_advance_one_512_pixel_cell(self) -> None:
        for resolution in (1024, 2048):
            with self.subTest(resolution=resolution):
                data = TextureAtlasData()
                atlas = data.create_atlas(
                    "Manual",
                    4096,
                    atlas_id="atlas-a",
                )
                source = _source(
                    f"chair-{resolution}",
                    directory=self._temporary_directory.name,
                    resolution=resolution,
                )
                self.workspace.set_data(data)
                self.workspace.set_object_texture_sources([source])
                preview = self.workspace.preview
                _qt_application.processEvents()
                mime_data = _build_texture_source_mime_data(source.object_id)
                first_point = _atlas_preview_point(
                    preview,
                    atlas.resolution,
                    700,
                    700,
                )
                second_point = _atlas_preview_point(
                    preview,
                    atlas.resolution,
                    700,
                    1200,
                )

                first_event = QDragEnterEvent(
                    first_point.toPoint(),
                    Qt.DropAction.CopyAction,
                    mime_data,
                    Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.NoModifier,
                )
                preview.dragEnterEvent(first_event)
                first_slot = preview.drag_slot_preview
                assert first_slot is not None
                self.assertEqual((first_slot.x, first_slot.y), (512, 512))
                self.assertEqual(first_slot.size, resolution)

                second_event = QDragMoveEvent(
                    second_point.toPoint(),
                    Qt.DropAction.CopyAction,
                    mime_data,
                    Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.NoModifier,
                )
                preview.dragMoveEvent(second_event)
                second_slot = preview.drag_slot_preview
                assert second_slot is not None
                self.assertEqual((second_slot.x, second_slot.y), (512, 1024))
                self.assertEqual(second_slot.size, resolution)
                self.assertTrue(second_slot.is_valid)

                drop_event = QDropEvent(
                    second_point,
                    Qt.DropAction.CopyAction,
                    mime_data,
                    Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.NoModifier,
                )
                preview.dropEvent(drop_event)

                placement = self.workspace.get_data().atlas_by_id(
                    atlas.atlas_id
                ).placement_for_object(source.object_id)
                assert placement is not None
                self.assertTrue(drop_event.isAccepted())
                self.assertEqual((placement.x, placement.y), (512, 1024))
                self.assertEqual(placement.size, resolution)

    def test_large_drag_preview_uses_its_full_footprint_for_validity(self) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Manual", 4096, atlas_id="atlas-a")
        dragged = _source(
            "large",
            directory=self._temporary_directory.name,
            resolution=1024,
        )
        occupied = _source(
            "occupied",
            directory=self._temporary_directory.name,
        )
        data.place_object_at(
            atlas.atlas_id,
            dragged.object_id,
            dragged.texture_path,
            dragged.texture_resolution,
            512,
            512,
        )
        data.place_object_at(
            atlas.atlas_id,
            occupied.object_id,
            occupied.texture_path,
            occupied.texture_resolution,
            1536,
            1024,
        )
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources([dragged, occupied])
        preview = self.workspace.preview
        _qt_application.processEvents()
        mime_data = _build_texture_source_mime_data(dragged.object_id)

        cases = (
            (700, 1200, (512, 1024), True),
            (1200, 1200, (1024, 1024), False),
            (3700, 1200, (3584, 1024), False),
        )
        for atlas_x, atlas_y, expected_slot, is_valid in cases:
            with self.subTest(expected_slot=expected_slot):
                point = _atlas_preview_point(
                    preview,
                    atlas.resolution,
                    atlas_x,
                    atlas_y,
                )
                move_event = QDragMoveEvent(
                    point.toPoint(),
                    Qt.DropAction.CopyAction,
                    mime_data,
                    Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.NoModifier,
                )

                preview.dragMoveEvent(move_event)

                slot = preview.drag_slot_preview
                assert slot is not None
                self.assertEqual((slot.x, slot.y), expected_slot)
                self.assertEqual(slot.size, 1024)
                self.assertEqual(slot.is_valid, is_valid)

    def test_drag_preview_tracks_list_source_without_mutating_until_drop(
        self,
    ) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Manual", 2048, atlas_id="atlas-a")
        source = _source("chair", directory=self._temporary_directory.name)
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources([source])
        preview = self.workspace.preview
        _qt_application.processEvents()
        mime_data = _build_texture_source_mime_data(source.object_id)
        before = self.workspace.get_data()
        first_point = _atlas_preview_point(preview, atlas.resolution, 700, 700)
        second_point = _atlas_preview_point(
            preview,
            atlas.resolution,
            1300,
            700,
        )

        enter_event = QDragEnterEvent(
            first_point.toPoint(),
            Qt.DropAction.CopyAction,
            mime_data,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        preview.dragEnterEvent(enter_event)

        slot_preview = preview.drag_slot_preview
        assert slot_preview is not None
        self.assertTrue(enter_event.isAccepted())
        self.assertEqual(
            (slot_preview.object_id, slot_preview.x, slot_preview.y),
            (source.object_id, 512, 512),
        )
        self.assertTrue(slot_preview.is_valid)
        self.assertEqual(self.workspace.get_data(), before)

        move_event = QDragMoveEvent(
            second_point.toPoint(),
            Qt.DropAction.CopyAction,
            mime_data,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        preview.dragMoveEvent(move_event)

        moved_preview = preview.drag_slot_preview
        assert moved_preview is not None
        self.assertTrue(move_event.isAccepted())
        self.assertEqual((moved_preview.x, moved_preview.y), (1024, 512))
        self.assertTrue(moved_preview.is_valid)
        self.assertEqual(self.workspace.get_data(), before)

        drop_event = QDropEvent(
            second_point,
            Qt.DropAction.CopyAction,
            mime_data,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        preview.dropEvent(drop_event)

        placement = self.workspace.get_data().atlas_by_id(
            atlas.atlas_id
        ).placement_for_object(source.object_id)
        assert placement is not None
        self.assertEqual((placement.x, placement.y), (1024, 512))
        self.assertIsNone(preview.drag_slot_preview)

    def test_drag_preview_marks_collisions_and_outside_points_as_blocked(
        self,
    ) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Manual", 2048, atlas_id="atlas-a")
        occupied = _source(
            "occupied",
            directory=self._temporary_directory.name,
            color=(20, 20, 20, 255),
        )
        dragged = _source(
            "dragged",
            directory=self._temporary_directory.name,
            color=(30, 120, 210, 255),
        )
        data.place_object_at(
            atlas.atlas_id,
            occupied.object_id,
            occupied.texture_path,
            occupied.texture_resolution,
            0,
            0,
        )
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources([occupied, dragged])
        preview = self.workspace.preview
        _qt_application.processEvents()
        mime_data = _build_texture_source_mime_data(dragged.object_id)
        collision_point = _atlas_preview_point(
            preview,
            atlas.resolution,
            200,
            200,
        )
        valid_point = _atlas_preview_point(
            preview,
            atlas.resolution,
            700,
            200,
        )
        before = self.workspace.get_data()

        collision_event = QDragEnterEvent(
            collision_point.toPoint(),
            Qt.DropAction.CopyAction,
            mime_data,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        preview.dragEnterEvent(collision_event)
        blocked_preview = preview.drag_slot_preview
        assert blocked_preview is not None
        self.assertTrue(collision_event.isAccepted())
        self.assertFalse(blocked_preview.is_valid)

        blocked_image = _paint_drag_feedback(preview)
        blocked_pixel = blocked_image.pixelColor(collision_point.toPoint())
        self.assertGreater(blocked_pixel.alpha(), 0)
        self.assertLess(blocked_pixel.alpha(), 255)

        valid_event = QDragMoveEvent(
            valid_point.toPoint(),
            Qt.DropAction.CopyAction,
            mime_data,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        preview.dragMoveEvent(valid_event)
        valid_preview = preview.drag_slot_preview
        assert valid_preview is not None
        self.assertTrue(valid_event.isAccepted())
        self.assertTrue(valid_preview.is_valid)
        valid_image = _paint_drag_feedback(preview)
        valid_pixel = valid_image.pixelColor(valid_point.toPoint())
        self.assertGreater(valid_pixel.alpha(), 0)
        self.assertLess(valid_pixel.alpha(), 255)
        self.assertNotEqual(valid_pixel.rgba(), blocked_pixel.rgba())

        outside_point = QPointF(2.0, preview.height() / 2.0)
        outside_event = QDragMoveEvent(
            outside_point.toPoint(),
            Qt.DropAction.CopyAction,
            mime_data,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        preview.dragMoveEvent(outside_event)
        outside_preview = preview.drag_slot_preview
        assert outside_preview is not None
        self.assertTrue(outside_event.isAccepted())
        self.assertFalse(outside_preview.is_valid)
        self.assertEqual(self.workspace.get_data(), before)

        blocked_drop = QDropEvent(
            outside_point,
            Qt.DropAction.CopyAction,
            mime_data,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        preview.dropEvent(blocked_drop)
        self.assertFalse(blocked_drop.isAccepted())
        self.assertIsNone(preview.drag_slot_preview)
        self.assertEqual(self.workspace.get_data(), before)

    def test_drag_preview_clears_on_leave_and_content_refresh(self) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Manual", 2048, atlas_id="atlas-a")
        source = _source("chair", directory=self._temporary_directory.name)
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources([source])
        preview = self.workspace.preview
        _qt_application.processEvents()
        mime_data = _build_texture_source_mime_data(source.object_id)
        point = _atlas_preview_point(preview, atlas.resolution, 700, 700)

        enter_event = QDragEnterEvent(
            point.toPoint(),
            Qt.DropAction.CopyAction,
            mime_data,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        preview.dragEnterEvent(enter_event)
        self.assertIsNotNone(preview.drag_slot_preview)

        preview.dragLeaveEvent(QDragLeaveEvent())
        self.assertIsNone(preview.drag_slot_preview)

        preview.dragEnterEvent(enter_event)
        self.assertIsNotNone(preview.drag_slot_preview)
        self.workspace._refresh_preview()
        self.assertIsNone(preview.drag_slot_preview)

    def test_existing_preview_texture_can_be_dragged_to_an_exact_slot(self) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Manual", 2048, atlas_id="atlas-a")
        source = _source("chair", directory=self._temporary_directory.name)
        data.place_object_at(
            atlas.atlas_id,
            source.object_id,
            source.texture_path,
            source.texture_resolution,
            0,
            0,
        )
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources([source])
        preview = self.workspace.preview
        _qt_application.processEvents()
        preview_side = min(preview.width() - 32.0, preview.height() - 32.0)
        preview_origin = QPointF(
            (preview.width() - preview_side) / 2.0,
            (preview.height() - preview_side) / 2.0,
        )
        start_point = QPointF(
            preview_origin.x() + 256.0 * preview_side / atlas.resolution,
            preview_origin.y() + 256.0 * preview_side / atlas.resolution,
        )
        drop_point = QPointF(
            preview_origin.x() + 1300.0 * preview_side / atlas.resolution,
            preview_origin.y() + 700.0 * preview_side / atlas.resolution,
        )
        press_event = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            start_point,
            start_point,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        preview.mousePressEvent(press_event)
        move_event = QMouseEvent(
            QEvent.Type.MouseMove,
            drop_point,
            drop_point,
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        with patch("housemaker.texture_atlas_workspace.QDrag") as drag_class:
            preview.mouseMoveEvent(move_event)

        drag = drag_class.return_value
        drag.exec.assert_called_once_with(Qt.DropAction.CopyAction)
        mime_data = drag.setMimeData.call_args.args[0]
        drag_enter_event = QDragEnterEvent(
            drop_point.toPoint(),
            Qt.DropAction.CopyAction,
            mime_data,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        preview.dragEnterEvent(drag_enter_event)
        self.assertTrue(drag_enter_event.isAccepted())
        slot_preview = preview.drag_slot_preview
        assert slot_preview is not None
        self.assertEqual((slot_preview.x, slot_preview.y), (1024, 512))
        self.assertTrue(slot_preview.is_valid)
        drop_event = QDropEvent(
            drop_point,
            Qt.DropAction.CopyAction,
            mime_data,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        preview.dropEvent(drop_event)

        updated = self.workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert updated is not None
        placement = updated.placement_for_object(source.object_id)
        assert placement is not None
        self.assertTrue(drop_event.isAccepted())
        self.assertEqual((placement.x, placement.y), (1024, 512))

    def test_list_row_starts_drag_before_and_after_atlas_assignment(self) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Manual", 2048, atlas_id="atlas-a")
        source = _source("chair", directory=self._temporary_directory.name)
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources([source])

        for should_be_assigned in (False, True):
            if should_be_assigned:
                self.workspace.assign_object_button.click()
            with patch(
                "housemaker.texture_atlas_workspace.QDrag"
            ) as drag_class:
                self.workspace.object_list.startDrag(Qt.DropAction.CopyAction)

            drag = drag_class.return_value
            drag.exec.assert_called_once_with(Qt.DropAction.CopyAction)
            mime_data = drag.setMimeData.call_args.args[0]
            self.assertTrue(
                mime_data.hasFormat(
                    "application/x-housemaker-texture-atlas-source"
                )
            )
            self.assertEqual(
                bytes(
                    mime_data.data(
                        "application/x-housemaker-texture-atlas-source"
                    )
                ).decode("utf-8"),
                source.object_id,
            )

        updated = self.workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert updated is not None
        self.assertIsNotNone(updated.placement_for_object(source.object_id))

    def test_drag_collision_and_png_failure_preserve_state_and_png(self) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Manual", 2048, atlas_id="atlas-a")
        first = _source("first", directory=self._temporary_directory.name)
        second = _source("second", directory=self._temporary_directory.name)
        data.place_object_at(
            atlas.atlas_id,
            first.object_id,
            first.texture_path,
            first.texture_resolution,
            0,
            0,
        )
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources([first, second])
        selected_atlas = self.workspace.selected_atlas
        assert selected_atlas is not None
        self.workspace._materialize_atlas(selected_atlas)
        before = self.workspace.get_data()
        assert selected_atlas.image_path is not None
        png_path = Path(self._temporary_directory.name) / selected_atlas.image_path
        before_png = png_path.read_bytes()
        changes: list[TextureAtlasData] = []
        self.workspace.data_changed.connect(changes.append)

        self.workspace.preview.object_dropped.emit(second.object_id, 0, 0)

        self.assertEqual(self.workspace.get_data(), before)
        self.assertEqual(png_path.read_bytes(), before_png)
        self.assertEqual(changes, [])
        self.assertIn("overlap", self.workspace.status_label.text())

        with patch(
            "housemaker.texture_atlas_workspace.write_texture_atlas_png",
            side_effect=OSError("disk full"),
        ):
            self.workspace.preview.object_dropped.emit(
                second.object_id,
                512,
                0,
            )

        self.assertEqual(self.workspace.get_data(), before)
        self.assertEqual(png_path.read_bytes(), before_png)
        self.assertEqual(changes, [])
        self.assertIn("disk full", self.workspace.status_label.text())

    def test_moving_pinned_variant_preserves_its_exact_path_and_resolution(
        self,
    ) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Pinned", 2048, atlas_id="atlas-a")
        pinned = _source(
            "chair",
            directory=self._temporary_directory.name,
            resolution=512,
        )
        active = _source(
            "chair",
            directory=self._temporary_directory.name,
            resolution=1024,
        )
        data.place_object_at(
            atlas.atlas_id,
            pinned.object_id,
            pinned.texture_path,
            pinned.texture_resolution,
            0,
            0,
        )
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources(
            [active],
            variant_resolver=_variant_resolver({("chair", 512): pinned}),
        )

        self.workspace.preview.object_dropped.emit("chair", 512, 512)

        updated = self.workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert updated is not None
        moved = updated.placement_for_object("chair")
        assert moved is not None
        self.assertEqual((moved.x, moved.y), (512, 512))
        self.assertEqual(moved.texture_resolution, 512)
        self.assertEqual(moved.texture_path, pinned.texture_path)

    def test_manual_slot_survives_other_source_removal_and_resize(self) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Stable", 2048, atlas_id="atlas-a")
        fixed = _source("fixed", directory=self._temporary_directory.name)
        variants = {
            resolution: _source(
                "resizable",
                directory=self._temporary_directory.name,
                resolution=resolution,
            )
            for resolution in (512, 1024)
        }
        data.place_object_at(
            atlas.atlas_id,
            fixed.object_id,
            fixed.texture_path,
            fixed.texture_resolution,
            1536,
            512,
        )
        data.place_object_at(
            atlas.atlas_id,
            "resizable",
            variants[512].texture_path,
            512,
            0,
            0,
        )
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources(
            [fixed, variants[512]],
            variant_resolver=lambda object_id, resolution: (
                variants.get(resolution)
                if object_id == "resizable"
                else None
            ),
            selectability_resolver=lambda _object_id, _resolution: True,
        )

        self.workspace._select_object_row("resizable")
        self.workspace.unassign_object_button.click()
        after_removal = self.workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert after_removal is not None
        fixed_placement = after_removal.placement_for_object("fixed")
        assert fixed_placement is not None
        self.assertEqual((fixed_placement.x, fixed_placement.y), (1536, 512))

        self.workspace.preview.object_dropped.emit("resizable", 0, 0)
        self.workspace._handle_object_wheel("resizable", 1)

        after_resize = self.workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert after_resize is not None
        fixed_placement = after_resize.placement_for_object("fixed")
        resized = after_resize.placement_for_object("resizable")
        assert fixed_placement is not None
        assert resized is not None
        self.assertEqual((fixed_placement.x, fixed_placement.y), (1536, 512))
        self.assertEqual(resized.texture_resolution, 1024)

    def test_global_resolution_without_placements_still_emits_commit(self) -> None:
        data = TextureAtlasData()
        data.create_atlas("Empty", 2048, atlas_id="atlas-a")
        variants = {
            resolution: _source(
                "chair",
                directory=self._temporary_directory.name,
                resolution=resolution,
            )
            for resolution in (512, 1024)
        }
        selectability_checks: list[tuple[str, int]] = []
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources(
            [variants[512]],
            variant_resolver=lambda object_id, resolution: (
                variants.get(resolution) if object_id == "chair" else None
            ),
            selectability_resolver=lambda object_id, resolution: (
                selectability_checks.append((object_id, resolution)) or True
            ),
        )
        changes: list[TextureAtlasData] = []
        resolution_changes: list[tuple[str, int]] = []
        self.workspace.data_changed.connect(changes.append)
        self.workspace.object_texture_resolution_changed.connect(
            lambda object_id, resolution: resolution_changes.append(
                (object_id, resolution)
            )
        )

        self.assertTrue(
            self.workspace.set_object_texture_resolution("chair", 1024)
        )

        self.assertEqual(changes, [])
        self.assertEqual(resolution_changes, [("chair", 1024)])
        self.assertEqual(selectability_checks, [("chair", 1024)])

    def test_zero_placement_callback_commits_once_without_atlas_signals(
        self,
    ) -> None:
        data = TextureAtlasData()
        data.create_atlas("Empty", 2048, atlas_id="atlas-a")
        variants = {
            resolution: _source(
                "chair",
                directory=self._temporary_directory.name,
                resolution=resolution,
            )
            for resolution in (512, 1024)
        }
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources(
            [variants[512]],
            variant_resolver=lambda object_id, resolution: (
                variants.get(resolution) if object_id == "chair" else None
            ),
            selectability_resolver=lambda _object_id, _resolution: True,
        )
        callbacks: list[str] = []
        data_changes: list[TextureAtlasData] = []
        resolution_changes: list[tuple[str, int]] = []
        self.workspace.data_changed.connect(data_changes.append)
        self.workspace.object_texture_resolution_changed.connect(
            lambda object_id, resolution: resolution_changes.append(
                (object_id, resolution)
            )
        )

        changed = self.workspace.set_object_texture_resolution(
            "chair",
            1024,
            commit_callback=lambda: callbacks.append("accepted") or True,
        )

        self.assertTrue(changed)
        self.assertEqual(callbacks, ["accepted"])
        self.assertEqual(self.workspace.get_data(), data)
        self.assertEqual(data_changes, [])
        self.assertEqual(resolution_changes, [])

    def test_rejected_callback_restores_exact_layout_png_and_signals(self) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Staged", 2048, atlas_id="atlas-a")
        variants = {
            resolution: _source(
                "chair",
                directory=self._temporary_directory.name,
                resolution=resolution,
            )
            for resolution in (512, 1024)
        }
        fixed = _source(
            "fixed",
            directory=self._temporary_directory.name,
        )
        data.place_object_at(
            atlas.atlas_id,
            "chair",
            variants[512].texture_path,
            512,
            512,
            0,
        )
        data.place_object_at(
            atlas.atlas_id,
            fixed.object_id,
            fixed.texture_path,
            fixed.texture_resolution,
            1536,
            1536,
        )
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources(
            [variants[512], fixed],
            variant_resolver=lambda object_id, resolution: (
                variants.get(resolution) if object_id == "chair" else None
            ),
            selectability_resolver=lambda _object_id, _resolution: True,
        )
        selected_atlas = self.workspace.selected_atlas
        assert selected_atlas is not None
        self.workspace._materialize_atlas(selected_atlas)
        before = self.workspace.get_data()
        png_path = Path(self._temporary_directory.name) / "atlas-a.png"
        before_png = png_path.read_bytes()
        data_changes: list[TextureAtlasData] = []
        resolution_changes: list[tuple[str, int]] = []
        self.workspace.data_changed.connect(data_changes.append)
        self.workspace.object_texture_resolution_changed.connect(
            lambda object_id, resolution: resolution_changes.append(
                (object_id, resolution)
            )
        )

        for rejection in ("false", "exception"):
            observed_candidates: list[tuple[int, int, int]] = []

            def reject_candidate() -> bool:
                candidate_atlas = self.workspace.selected_atlas
                assert candidate_atlas is not None
                candidate = candidate_atlas.placement_for_object("chair")
                assert candidate is not None
                observed_candidates.append(
                    (candidate.texture_resolution, candidate.x, candidate.y)
                )
                if rejection == "exception":
                    raise RuntimeError("global commit failed")
                return False

            with self.subTest(rejection=rejection):
                changed = self.workspace.set_object_texture_resolution(
                    "chair",
                    1024,
                    commit_callback=reject_candidate,
                )

                self.assertFalse(changed)
                self.assertEqual(observed_candidates, [(1024, 512, 0)])
                self.assertEqual(self.workspace.get_data(), before)
                restored_atlas = self.workspace.selected_atlas
                assert restored_atlas is not None
                restored = restored_atlas.placement_for_object("chair")
                assert restored is not None
                self.assertEqual(
                    (restored.texture_resolution, restored.x, restored.y),
                    (512, 512, 0),
                )
                self.assertEqual(png_path.read_bytes(), before_png)
                self.assertEqual(data_changes, [])
                self.assertEqual(resolution_changes, [])

    def test_global_resolution_failure_restores_every_prior_png(self) -> None:
        data = TextureAtlasData()
        first_atlas = data.create_atlas("First", 2048, atlas_id="atlas-a")
        second_atlas = data.create_atlas("Second", 2048, atlas_id="atlas-b")
        variants = {
            resolution: _source(
                "chair",
                directory=self._temporary_directory.name,
                resolution=resolution,
            )
            for resolution in (512, 1024)
        }
        for atlas in (first_atlas, second_atlas):
            data.assign_object(
                atlas.atlas_id,
                "chair",
                variants[512].texture_path,
                512,
            )
            atlas.image_path = f"{atlas.atlas_id}.png"
            Path(
                self._temporary_directory.name,
                atlas.image_path,
            ).write_bytes(f"old-{atlas.atlas_id}".encode())
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources(
            [variants[512]],
            variant_resolver=lambda object_id, resolution: (
                variants.get(resolution) if object_id == "chair" else None
            ),
            selectability_resolver=lambda _object_id, _resolution: True,
        )
        before = self.workspace.get_data()
        original_materialize = self.workspace._materialize_atlas
        call_count = 0

        def fail_second_materialization(candidate_atlas) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise OSError("second write failed")
            original_materialize(candidate_atlas)

        changes: list[TextureAtlasData] = []
        resolution_changes: list[tuple[str, int]] = []
        self.workspace.data_changed.connect(changes.append)
        self.workspace.object_texture_resolution_changed.connect(
            lambda object_id, resolution: resolution_changes.append(
                (object_id, resolution)
            )
        )
        with patch.object(
            self.workspace,
            "_materialize_atlas",
            side_effect=fail_second_materialization,
        ):
            changed = self.workspace.set_object_texture_resolution(
                "chair",
                1024,
            )

        self.assertFalse(changed)
        self.assertEqual(self.workspace.get_data(), before)
        self.assertEqual(changes, [])
        self.assertEqual(resolution_changes, [])
        for atlas in (first_atlas, second_atlas):
            self.assertEqual(
                Path(
                    self._temporary_directory.name,
                    f"{atlas.atlas_id}.png",
                ).read_bytes(),
                f"old-{atlas.atlas_id}".encode(),
            )

    def test_reassigns_object_when_active_texture_resolution_changes(self) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Variants", 4096, atlas_id="atlas-a")
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources(
            [_source("lamp", directory=self._temporary_directory.name)]
        )
        self.workspace.assign_object_button.click()

        self.workspace.set_object_texture_sources(
            [
                _source(
                    "lamp",
                    directory=self._temporary_directory.name,
                    resolution=2048,
                )
            ]
        )
        self.workspace.assign_object_button.click()

        placement = self.workspace.get_data().atlas_by_id(
            atlas.atlas_id
        ).placement_for_object("lamp")
        assert placement is not None
        self.assertEqual(placement.texture_resolution, 2048)
        self.assertEqual(placement.texture_path, "textures/lamp-2048.png")

    def test_regenerated_texture_remaps_path_and_rematerializes_pixels(self) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Regenerated", 2048, atlas_id="atlas-a")
        self.workspace.set_data(data)
        old_source = _source(
            "chair",
            directory=self._temporary_directory.name,
            color=(30, 120, 210, 255),
        )
        self.workspace.set_object_texture_sources([old_source])
        self.workspace.assign_object_button.click()
        before = self.workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert before is not None
        old_placement = before.placement_for_object("chair")
        assert old_placement is not None
        old_atlas_path = before.image_path
        assert old_atlas_path is not None

        new_physical_path = (
            Path(self._temporary_directory.name) / "chair-regenerated-512.png"
        )
        new_color = (220, 40, 15, 255)
        new_pixels = np.empty((512, 512, 4), dtype=np.uint8)
        new_pixels[:, :] = np.asarray(new_color, dtype=np.uint8)
        Image.fromarray(new_pixels, mode="RGBA").save(new_physical_path)
        new_source = AtlasObjectTextureSource(
            object_id="chair",
            object_name="Chair",
            texture_path="textures/chair-regenerated-512.png",
            texture_resolution=512,
            physical_texture_path=new_physical_path,
            preview_rgba=np.full((8, 8, 4), new_color, dtype=np.uint8),
        )
        self.workspace.set_object_texture_sources(
            [new_source],
            variant_resolver=_variant_resolver({("chair", 512): new_source}),
        )
        old_source.physical_texture_path.unlink()
        changes: list[TextureAtlasData] = []
        self.workspace.data_changed.connect(changes.append)

        affected_count = self.workspace.refresh_regenerated_object_texture(
            "chair"
        )

        self.assertEqual(affected_count, 1)
        self.assertEqual(len(changes), 1)
        updated = self.workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert updated is not None
        placement = updated.placement_for_object("chair")
        assert placement is not None
        self.assertEqual(placement.texture_resolution, 512)
        self.assertEqual(
            placement.texture_path,
            "textures/chair-regenerated-512.png",
        )
        self.assertEqual(
            (placement.x, placement.y, placement.size),
            (old_placement.x, old_placement.y, old_placement.size),
        )
        self.assertEqual(updated.image_path, old_atlas_path)
        with Image.open(
            Path(self._temporary_directory.name) / updated.image_path
        ) as atlas_image:
            self.assertEqual(
                atlas_image.getpixel((placement.x + 10, placement.y + 10)),
                new_color,
            )
        self.assertFalse(old_source.physical_texture_path.exists())
        self.assertIn("Updated", self.workspace.status_label.text())

    def test_regenerated_unassigned_object_is_a_no_op(self) -> None:
        data = TextureAtlasData()
        data.create_atlas("Empty", 2048, atlas_id="atlas-a")
        self.workspace.set_data(data)
        before = self.workspace.get_data()
        changes: list[TextureAtlasData] = []
        self.workspace.data_changed.connect(changes.append)

        affected_count = self.workspace.refresh_regenerated_object_texture(
            "unassigned"
        )

        self.assertEqual(affected_count, 0)
        self.assertEqual(self.workspace.get_data(), before)
        self.assertEqual(changes, [])

    def test_object_row_mouse_buttons_only_select_and_preview(self) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Clickable", 2048, atlas_id="atlas-a")
        variants = {
            ("chair", resolution): _source(
                "chair",
                directory=self._temporary_directory.name,
                resolution=resolution,
            )
            for resolution in (512, 1024, 2048)
        }
        data.assign_object(
            atlas.atlas_id,
            "chair",
            variants[("chair", 512)].texture_path,
            512,
        )
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources(
            [variants[("chair", 512)]],
            variant_resolver=_variant_resolver(variants),
        )
        preview_requests: list[tuple[str, int]] = []
        resolution_changes: list[tuple[str, int]] = []
        click_events: list[tuple[str, int]] = []
        self.workspace.object_preview_requested.connect(
            lambda object_id, resolution: preview_requests.append(
                (object_id, resolution)
            )
        )
        self.workspace.object_preview_requested.connect(
            lambda _object_id, resolution: click_events.append(
                ("preview", resolution)
            )
        )
        self.workspace.object_texture_resolution_changed.connect(
            lambda object_id, resolution: resolution_changes.append(
                (object_id, resolution)
            )
        )
        self.workspace.object_texture_resolution_changed.connect(
            lambda _object_id, resolution: click_events.append(
                ("global", resolution)
            )
        )

        row_position = self.workspace.object_list.visualItemRect(
            self.workspace.object_list.item(0)
        ).center()
        QTest.mouseClick(
            self.workspace.object_list.viewport(),
            Qt.MouseButton.RightButton,
            pos=row_position,
        )
        QTest.mouseClick(
            self.workspace.object_list.viewport(),
            Qt.MouseButton.LeftButton,
            pos=row_position,
        )
        QTest.mouseClick(
            self.workspace.object_list.viewport(),
            Qt.MouseButton.LeftButton,
            pos=row_position,
        )
        QTest.mouseClick(
            self.workspace.object_list.viewport(),
            Qt.MouseButton.RightButton,
            pos=row_position,
        )

        placement = self.workspace.get_data().atlas_by_id(
            atlas.atlas_id
        ).placement_for_object("chair")
        assert placement is not None
        self.assertEqual(placement.texture_resolution, 512)
        self.assertIn("512 x 512", self.workspace.object_list.item(0).text())
        self.assertEqual(
            preview_requests,
            [
                ("chair", 512),
                ("chair", 512),
                ("chair", 512),
                ("chair", 512),
            ],
        )
        self.assertEqual(resolution_changes, [])
        self.assertEqual(
            click_events,
            [
                ("preview", 512),
                ("preview", 512),
                ("preview", 512),
                ("preview", 512),
            ],
        )
        self.assertEqual(self.workspace.selected_object_id, "chair")
        self.assertEqual(
            self.workspace.get_selected_object_texture_resolution(),
            512,
        )

    def test_preview_hit_testing_and_click_select_the_exact_placement(self) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Preview", 2048, atlas_id="atlas-a")
        variants: dict[tuple[str, int], AtlasObjectTextureSource] = {}
        for object_id in ("chair-a", "chair-b"):
            for resolution in (512, 1024, 2048):
                variants[(object_id, resolution)] = _source(
                    object_id,
                    directory=self._temporary_directory.name,
                    resolution=resolution,
                )
            data.assign_object(
                atlas.atlas_id,
                object_id,
                variants[(object_id, 512)].texture_path,
                512,
            )
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources(
            [variants[("chair-a", 512)], variants[("chair-b", 512)]],
            variant_resolver=_variant_resolver(variants),
        )
        _qt_application.processEvents()
        preview = self.workspace.preview
        side = min(preview.width() - 32.0, preview.height() - 32.0)
        left = (preview.width() - side) / 2.0
        top = (preview.height() - side) / 2.0
        scale = side / 2048.0
        shared_edge = QPointF(left + 512.0 * scale, top + 128.0 * scale)
        self.assertEqual(preview.object_id_at(shared_edge), "chair-b")
        second_center = QPointF(
            left + (512.0 + 256.0) * scale,
            top + 256.0 * scale,
        )
        preview_requests: list[tuple[str, int]] = []
        self.workspace.object_preview_requested.connect(
            lambda object_id, resolution: preview_requests.append(
                (object_id, resolution)
            )
        )

        QTest.mouseClick(
            preview,
            Qt.MouseButton.RightButton,
            pos=QPointF(2.0, 2.0).toPoint(),
        )
        self.assertEqual(preview_requests, [])

        QTest.mouseClick(
            preview,
            Qt.MouseButton.RightButton,
            pos=second_center.toPoint(),
        )

        updated = self.workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert updated is not None
        placement = updated.placement_for_object("chair-b")
        assert placement is not None
        self.assertEqual(placement.texture_resolution, 512)
        self.assertEqual(self.workspace.selected_object_id, "chair-b")
        self.assertEqual(preview_requests, [("chair-b", 512)])

    def test_global_resolution_api_repacks_every_atlas_layout(self) -> None:
        data = TextureAtlasData()
        first = data.create_atlas("First", 2048, atlas_id="atlas-a")
        second = data.create_atlas("Second", 2048, atlas_id="atlas-b")
        variants = {
            resolution: _source(
                "chair",
                directory=self._temporary_directory.name,
                resolution=resolution,
            )
            for resolution in (512, 1024)
        }
        for atlas in (first, second):
            data.assign_object(
                atlas.atlas_id,
                "chair",
                variants[512].texture_path,
                512,
            )
        data.select_atlas(first.atlas_id)
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources(
            [variants[512]],
            variant_resolver=lambda object_id, resolution: (
                variants.get(resolution) if object_id == "chair" else None
            ),
            selectability_resolver=lambda _object_id, _resolution: True,
        )
        resolution_changes: list[tuple[str, int]] = []
        preview_requests: list[tuple[str, int]] = []
        self.workspace.object_texture_resolution_changed.connect(
            lambda object_id, resolution: resolution_changes.append(
                (object_id, resolution)
            )
        )
        self.workspace.object_preview_requested.connect(
            lambda object_id, resolution: preview_requests.append(
                (object_id, resolution)
            )
        )

        self.assertTrue(
            self.workspace.set_object_texture_resolution("chair", 1024)
        )
        self.workspace.atlas_list.setCurrentRow(1)

        updated = self.workspace.get_data()
        first_placement = updated.atlas_by_id(
            first.atlas_id
        ).placement_for_object("chair")
        second_placement = updated.atlas_by_id(
            second.atlas_id
        ).placement_for_object("chair")
        assert first_placement is not None
        assert second_placement is not None
        self.assertEqual(first_placement.texture_resolution, 1024)
        self.assertEqual(second_placement.texture_resolution, 1024)
        self.assertEqual(resolution_changes, [("chair", 1024)])
        self.assertEqual(preview_requests, [("chair", 1024)])

    def test_keyboard_selection_previews_without_resizing(self) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Keyboard", 2048, atlas_id="atlas-a")
        first = _source("first", directory=self._temporary_directory.name)
        second = _source(
            "second",
            directory=self._temporary_directory.name,
            resolution=1024,
        )
        data.assign_object(
            atlas.atlas_id,
            first.object_id,
            first.texture_path,
            first.texture_resolution,
        )
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources([first, second])
        preview_requests: list[tuple[str, int]] = []
        changes: list[TextureAtlasData] = []
        self.workspace.object_preview_requested.connect(
            lambda object_id, resolution: preview_requests.append(
                (object_id, resolution)
            )
        )
        self.workspace.data_changed.connect(changes.append)

        self.workspace.object_list.setCurrentRow(1)

        self.assertEqual(preview_requests, [("second", 1024)])
        self.assertEqual(changes, [])
        unchanged = self.workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert unchanged is not None
        self.assertEqual(len(unchanged.placements), 1)

    def test_capacity_failure_keeps_state_png_and_preview_resolution(self) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Full", 2048, atlas_id="atlas-a")
        for object_id in ("chair-a", "chair-b", "chair-c", "chair-d"):
            data.assign_object(
                atlas.atlas_id,
                object_id,
                f"textures/{object_id}-1024.png",
                1024,
            )
        atlas.image_path = "atlas-a.png"
        old_png_path = Path(self._temporary_directory.name) / "atlas-a.png"
        old_png_path.write_bytes(b"existing atlas bytes")
        current = _source(
            "chair-a",
            directory=self._temporary_directory.name,
            resolution=1024,
        )
        larger = _source(
            "chair-a",
            directory=self._temporary_directory.name,
            resolution=2048,
        )
        selectability_checks: list[tuple[str, int]] = []
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources(
            [current],
            variant_resolver=_variant_resolver({("chair-a", 2048): larger}),
            selectability_resolver=lambda object_id, resolution: (
                selectability_checks.append((object_id, resolution)) or True
            ),
        )
        changes: list[TextureAtlasData] = []
        preview_requests: list[tuple[str, int]] = []
        resolution_changes: list[tuple[str, int]] = []
        self.workspace.data_changed.connect(changes.append)
        self.workspace.object_preview_requested.connect(
            lambda object_id, resolution: preview_requests.append(
                (object_id, resolution)
            )
        )
        self.workspace.object_texture_resolution_changed.connect(
            lambda object_id, resolution: resolution_changes.append(
                (object_id, resolution)
            )
        )

        self.workspace._handle_object_wheel("chair-a", 1)

        unchanged = self.workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert unchanged is not None
        placement = unchanged.placement_for_object("chair-a")
        assert placement is not None
        self.assertEqual(placement.texture_resolution, 1024)
        self.assertEqual(old_png_path.read_bytes(), b"existing atlas bytes")
        self.assertEqual(changes, [])
        self.assertEqual(preview_requests, [("chair-a", 1024)])
        self.assertEqual(resolution_changes, [])
        self.assertEqual(selectability_checks, [])
        self.assertIn("blocked by atlas capacity", self.workspace.status_label.text())
        self.assertIn("Keeping 1024 x 1024", self.workspace.status_label.text())

    def test_missing_3d_variant_blocks_global_resolution_commit(self) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Missing GLB", 2048, atlas_id="atlas-a")
        variants = {
            resolution: _source(
                "chair",
                directory=self._temporary_directory.name,
                resolution=resolution,
            )
            for resolution in (512, 1024)
        }
        data.assign_object(
            atlas.atlas_id,
            "chair",
            variants[512].texture_path,
            512,
        )
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources(
            [variants[512]],
            variant_resolver=lambda object_id, resolution: (
                variants.get(resolution) if object_id == "chair" else None
            ),
            selectability_resolver=lambda _object_id, _resolution: False,
        )
        atlas_changes: list[TextureAtlasData] = []
        resolution_changes: list[tuple[str, int]] = []
        self.workspace.data_changed.connect(atlas_changes.append)
        self.workspace.object_texture_resolution_changed.connect(
            lambda object_id, resolution: resolution_changes.append(
                (object_id, resolution)
            )
        )

        changed = self.workspace._cycle_object_texture_resolution("chair", 1)

        unchanged = self.workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert unchanged is not None
        placement = unchanged.placement_for_object("chair")
        assert placement is not None
        self.assertFalse(changed)
        self.assertEqual(placement.texture_resolution, 512)
        self.assertEqual(atlas_changes, [])
        self.assertEqual(resolution_changes, [])
        self.assertIn("3D texture variant", self.workspace.status_label.text())
        self.assertIn("Keeping 512 x 512", self.workspace.status_label.text())

    def test_assigned_and_unassigned_clicks_only_select_without_mutating(self) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Selection", 2048, atlas_id="atlas-a")
        assigned = _source("assigned", directory=self._temporary_directory.name)
        unassigned = _source(
            "unassigned",
            directory=self._temporary_directory.name,
            resolution=1024,
        )
        data.assign_object(
            atlas.atlas_id,
            assigned.object_id,
            assigned.texture_path,
            assigned.texture_resolution,
        )
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources([assigned, unassigned])
        changes: list[TextureAtlasData] = []
        preview_requests: list[tuple[str, int]] = []
        self.workspace.data_changed.connect(changes.append)
        self.workspace.object_preview_requested.connect(
            lambda object_id, resolution: preview_requests.append(
                (object_id, resolution)
            )
        )

        self.workspace._handle_object_mouse_click(
            "assigned",
            Qt.MouseButton.LeftButton,
        )
        self.workspace._handle_object_mouse_click(
            "unassigned",
            Qt.MouseButton.RightButton,
        )

        unchanged = self.workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert unchanged is not None
        self.assertEqual(len(unchanged.placements), 1)
        self.assertEqual(changes, [])
        self.assertEqual(
            preview_requests,
            [("assigned", 512), ("unassigned", 1024)],
        )
        self.assertIn("Selected the texture", self.workspace.status_label.text())

    def test_missing_object_placement_is_retained_and_reported(self) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Loaded", 2048, atlas_id="atlas-a")
        data.assign_object(
            atlas.atlas_id,
            "deleted-chair",
            "textures/deleted-chair-512.png",
            512,
        )

        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources([])

        loaded = self.workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert loaded is not None
        self.assertIsNotNone(loaded.placement_for_object("deleted-chair"))
        self.assertIn("retained", self.workspace.status_label.text())
        self.assertIn("without an available", self.workspace.status_label.text())
        self.assertEqual(self.workspace.object_list.count(), 1)
        self.assertEqual(
            self.workspace.object_list.item(0).text(),
            "[Missing] deleted-chair",
        )
        self.assertFalse(self.workspace.assign_object_button.isEnabled())
        self.assertTrue(self.workspace.unassign_object_button.isEnabled())

    def test_missing_placement_can_be_removed_before_another_assignment(
        self,
    ) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Recoverable", 2048, atlas_id="atlas-a")
        data.assign_object(
            atlas.atlas_id,
            "deleted-chair",
            "textures/deleted-chair-512.png",
            512,
        )
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources([])

        self.workspace.object_list.setCurrentRow(0)
        self.workspace.unassign_object_button.click()

        cleaned = self.workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert cleaned is not None
        self.assertEqual(cleaned.placements, [])
        self.assertEqual(self.workspace.object_list.count(), 0)
        self.assertIsNone(cleaned.image_path)
        self.assertIn("now-empty derived PNG", self.workspace.status_label.text())

        replacement = _source(
            "replacement-table",
            directory=self._temporary_directory.name,
        )
        self.workspace.set_object_texture_sources([replacement])
        self.workspace.object_list.setCurrentRow(0)
        self.workspace.assign_object_button.click()

        updated = self.workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert updated is not None
        self.assertIsNotNone(
            updated.placement_for_object("replacement-table")
        )
        assert updated.image_path is not None
        self.assertTrue(
            (
                Path(self._temporary_directory.name) / updated.image_path
            ).is_file()
        )

    def test_removing_one_of_two_missing_placements_commits_safely(
        self,
    ) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Two missing", 2048, atlas_id="atlas-a")
        for object_id in ("missing-a", "missing-b"):
            data.assign_object(
                atlas.atlas_id,
                object_id,
                f"textures/{object_id}-512.png",
                512,
            )
        atlas.image_path = "atlas-a.png"
        stale_path = Path(self._temporary_directory.name) / atlas.image_path
        Image.new("RGBA", (16, 16), (210, 20, 30, 255)).save(stale_path)
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources([])
        changes: list[TextureAtlasData] = []
        self.workspace.data_changed.connect(changes.append)

        self.workspace.object_list.setCurrentRow(0)
        self.workspace.unassign_object_button.click()

        updated = self.workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert updated is not None
        self.assertEqual(
            [placement.object_id for placement in updated.placements],
            ["missing-b"],
        )
        self.assertIsNone(updated.image_path)
        self.assertFalse(stale_path.exists())
        self.assertEqual(len(changes), 1)
        self.assertIn("detached", self.workspace.status_label.text())

    def test_explicit_object_deletion_preserves_other_atlas_slots(self) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Shared", 2048, atlas_id="atlas-a")
        second_atlas = data.create_atlas(
            "Deleted only",
            2048,
            atlas_id="atlas-b",
        )
        deleted_source = _source(
            "deleted",
            directory=self._temporary_directory.name,
            color=(210, 20, 30, 255),
        )
        remaining_source = _source(
            "remaining",
            directory=self._temporary_directory.name,
            color=(20, 190, 70, 255),
        )
        data.assign_object(
            atlas.atlas_id,
            deleted_source.object_id,
            deleted_source.texture_path,
            deleted_source.texture_resolution,
        )
        data.assign_object(
            atlas.atlas_id,
            remaining_source.object_id,
            remaining_source.texture_path,
            remaining_source.texture_resolution,
        )
        data.assign_object(
            second_atlas.atlas_id,
            deleted_source.object_id,
            deleted_source.texture_path,
            deleted_source.texture_resolution,
        )
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources(
            [deleted_source, remaining_source]
        )
        self.assertEqual(self.workspace.materialize_missing_atlases(), 1)
        selected_only = self.workspace.get_data()
        self.assertIsNotNone(
            selected_only.atlas_by_id(atlas.atlas_id).image_path
        )
        self.assertIsNone(
            selected_only.atlas_by_id(second_atlas.atlas_id).image_path
        )
        self.workspace.atlas_list.setCurrentRow(1)
        lazily_built = self.workspace.get_data().atlas_by_id(
            second_atlas.atlas_id
        )
        assert lazily_built is not None
        self.assertIsNotNone(lazily_built.image_path)
        self.assertEqual(self.workspace.materialize_missing_atlases(), 0)
        self.workspace.atlas_list.setCurrentRow(0)
        changes: list[TextureAtlasData] = []
        self.workspace.data_changed.connect(changes.append)

        removed_count = self.workspace.remove_deleted_object("deleted")

        self.assertEqual(removed_count, 2)
        self.assertEqual(len(changes), 1)
        updated = self.workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert updated is not None
        self.assertIsNone(updated.placement_for_object("deleted"))
        remaining = updated.placement_for_object("remaining")
        assert remaining is not None
        self.assertEqual((remaining.x, remaining.y), (512, 0))
        assert updated.image_path is not None
        output_path = (
            Path(self._temporary_directory.name) / updated.image_path
        )
        with Image.open(output_path) as image:
            self.assertEqual(
                image.getpixel((remaining.x + 10, remaining.y + 10)),
                (20, 190, 70, 255),
            )
        other = self.workspace.get_data().atlas_by_id(second_atlas.atlas_id)
        assert other is not None
        self.assertEqual(other.placements, [])
        self.assertIsNone(other.image_path)
        changes[0].atlases.clear()
        self.assertEqual(len(self.workspace.get_data().atlases), 2)

    def test_explicit_deletion_detaches_stale_png_when_other_source_missing(
        self,
    ) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Incomplete", 2048, atlas_id="atlas-a")
        deleted_source = _source(
            "deleted",
            directory=self._temporary_directory.name,
        )
        data.assign_object(
            atlas.atlas_id,
            deleted_source.object_id,
            deleted_source.texture_path,
            deleted_source.texture_resolution,
        )
        data.assign_object(
            atlas.atlas_id,
            "missing",
            "textures/missing-512.png",
            512,
        )
        atlas.image_path = "atlas-a.png"
        stale_path = Path(self._temporary_directory.name) / atlas.image_path
        Image.new("RGBA", (16, 16), (210, 20, 30, 255)).save(stale_path)
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources([deleted_source])
        changes: list[TextureAtlasData] = []
        self.workspace.data_changed.connect(changes.append)

        removed_count = self.workspace.remove_deleted_object("deleted")

        self.assertEqual(removed_count, 1)
        self.assertEqual(len(changes), 1)
        updated = self.workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert updated is not None
        self.assertIsNone(updated.placement_for_object("deleted"))
        missing = updated.placement_for_object("missing")
        assert missing is not None
        self.assertEqual((missing.x, missing.y), (512, 0))
        self.assertIsNone(updated.image_path)
        self.assertFalse(stale_path.exists())

    def test_atlas_selection_refreshes_its_missing_placement_rows(self) -> None:
        data = TextureAtlasData()
        first = data.create_atlas("First", 2048, atlas_id="atlas-a")
        second = data.create_atlas("Second", 2048, atlas_id="atlas-b")
        data.assign_object(
            first.atlas_id,
            "missing-chair",
            "textures/missing-chair.png",
            512,
        )
        data.assign_object(
            second.atlas_id,
            "missing-table",
            "textures/missing-table.png",
            512,
        )
        data.select_atlas(first.atlas_id)
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources([])
        self.assertEqual(
            self.workspace.object_list.item(0).text(),
            "[Missing] missing-chair",
        )
        preview_requests: list[tuple[str, int]] = []
        self.workspace.object_preview_requested.connect(
            lambda object_id, resolution: preview_requests.append(
                (object_id, resolution)
            )
        )

        self.workspace.atlas_list.setCurrentRow(1)

        self.assertEqual(self.workspace.object_list.count(), 1)
        self.assertEqual(
            self.workspace.object_list.item(0).text(),
            "[Missing] missing-table",
        )
        self.assertEqual(preview_requests, [("missing-table", 512)])

    def test_mutations_emit_detached_state(self) -> None:
        changes: list[TextureAtlasData] = []
        self.workspace.data_changed.connect(changes.append)
        self.workspace.atlas_name_edit.setText("Emitted")

        self.workspace.create_atlas_button.click()

        self.assertEqual(len(changes), 1)
        changes[0].atlases.clear()
        self.assertEqual(len(self.workspace.get_data().atlases), 1)

    def test_exact_persisted_resolution_is_used_instead_of_active_variant(
        self,
    ) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Pinned", 4096, atlas_id="atlas-a")
        exact_source = _source(
            "table",
            directory=self._temporary_directory.name,
            resolution=512,
        )
        active_source = _source(
            "table",
            directory=self._temporary_directory.name,
            resolution=2048,
        )
        data.assign_object(
            atlas.atlas_id,
            "table",
            exact_source.texture_path,
            512,
        )
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources(
            [active_source],
            variant_resolver=lambda object_id, resolution: (
                exact_source
                if object_id == "table" and resolution == 512
                else None
            ),
        )

        self.assertEqual(self.workspace.materialize_missing_atlases(), 1)

        loaded = self.workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert loaded is not None
        self.assertIsNotNone(loaded.image_path)
        self.assertNotIn("retained", self.workspace.status_label.text())

    def test_exact_variant_source_is_cached_for_one_atlas_rebuild(self) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Cached", 2048, atlas_id="atlas-a")
        exact_source = _source(
            "table",
            directory=self._temporary_directory.name,
            resolution=512,
        )
        active_source = _source(
            "table",
            directory=self._temporary_directory.name,
            resolution=2048,
        )
        data.assign_object(
            atlas.atlas_id,
            "table",
            exact_source.texture_path,
            exact_source.texture_resolution,
        )
        resolver_calls = 0

        def resolve_exact(
            object_id: str,
            resolution: int,
        ) -> AtlasObjectTextureSource | None:
            nonlocal resolver_calls
            resolver_calls += 1
            if object_id == "table" and resolution == 512:
                return exact_source
            return None

        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources(
            [active_source],
            variant_resolver=resolve_exact,
        )

        self.assertEqual(self.workspace.materialize_missing_atlases(), 1)
        self.assertEqual(resolver_calls, 1)

    def test_failed_png_write_rolls_back_assignment(self) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Atomic", 2048, atlas_id="atlas-a")
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources(
            [_source("chair", directory=self._temporary_directory.name)]
        )

        with patch(
            "housemaker.texture_atlas_workspace.write_texture_atlas_png",
            side_effect=ValueError("write failed"),
        ):
            self.workspace.assign_object_button.click()

        loaded = self.workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert loaded is not None
        self.assertEqual(loaded.placements, [])
        self.assertEqual(self.workspace.status_label.text(), "write failed")

    def test_lazy_png_failure_and_subsequent_materialization(self) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Retryable", 2048, atlas_id="atlas-a")
        source = _source("chair", directory=self._temporary_directory.name)
        data.assign_object(
            atlas.atlas_id,
            source.object_id,
            source.texture_path,
            source.texture_resolution,
        )
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources([source])

        with patch(
            "housemaker.texture_atlas_workspace.write_texture_atlas_png",
            side_effect=OSError("disk full"),
        ):
            self.assertEqual(self.workspace.materialize_missing_atlases(), 0)

        failed = self.workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert failed is not None
        self.assertIsNone(failed.image_path)
        self.assertIn("PNG could not be built", self.workspace.status_label.text())
        self.assertIn("disk full", self.workspace.status_label.text())

        self.assertEqual(self.workspace.materialize_missing_atlases(), 1)

        materialized = self.workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert materialized is not None
        self.assertIsNotNone(materialized.image_path)

    def test_delete_atlas_removes_its_owned_png(self) -> None:
        self.workspace.atlas_name_edit.setText("Disposable")
        self.workspace.create_atlas_button.click()
        self.workspace.set_object_texture_sources(
            [_source("chair", directory=self._temporary_directory.name)]
        )
        self.workspace.assign_object_button.click()
        atlas = self.workspace.get_data().atlases[0]
        assert atlas.image_path is not None
        image_path = Path(self._temporary_directory.name) / atlas.image_path
        self.assertTrue(image_path.exists())

        self.workspace.remove_atlas_button.click()

        self.assertFalse(image_path.exists())
        self.assertEqual(self.workspace.get_data().atlases, [])

    def test_delete_atlas_keeps_state_deleted_when_png_cleanup_fails(self) -> None:
        self.workspace.atlas_name_edit.setText("Cleanup failure")
        self.workspace.create_atlas_button.click()
        self.workspace.set_object_texture_sources(
            [_source("chair", directory=self._temporary_directory.name)]
        )
        self.workspace.assign_object_button.click()

        with patch.object(Path, "unlink", side_effect=OSError("locked")):
            self.workspace.remove_atlas_button.click()

        self.assertEqual(self.workspace.get_data().atlases, [])
        self.assertIn("could not be removed", self.workspace.status_label.text())
        self.assertIn("locked", self.workspace.status_label.text())

    def test_many_sources_retain_only_small_owned_thumbnails(self) -> None:
        sources = [
            _source(
                f"object-{index}",
                directory=self._temporary_directory.name,
            )
            for index in range(24)
        ]

        self.workspace.set_object_texture_sources(sources)

        self.assertEqual(self.workspace.object_list.count(), 24)
        self.assertTrue(
            all(max(source.preview_rgba.shape[:2]) <= 256 for source in sources)
        )
        self.assertLess(
            sum(source.preview_rgba.nbytes for source in sources),
            24 * 256 * 256 * 4,
        )

    def test_symmetric_sources_pair_across_orientations_and_hit_each_half(
        self,
    ) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Halves", 2048, atlas_id="atlas-a")
        vertical = _source(
            "vertical",
            directory=self._temporary_directory.name,
            symmetric_orientation="vertical",
            symmetric_plane_coordinate=0.0,
        )
        horizontal = _source(
            "horizontal",
            directory=self._temporary_directory.name,
            symmetric_orientation="horizontal",
            symmetric_plane_coordinate=1.5,
        )
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources([vertical, horizontal])
        for row in range(2):
            self.workspace.object_list.setCurrentRow(row)
            self.workspace.assign_object_button.click()

        packed = self.workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert packed is not None
        placements = {
            placement.object_id: placement for placement in packed.placements
        }
        self.assertEqual(
            (placements["vertical"].x, placements["vertical"].y),
            (placements["horizontal"].x, placements["horizontal"].y),
        )
        self.assertEqual(
            placements["vertical"].slot_half,
            ATLAS_SLOT_HALF_LEFT,
        )
        self.assertEqual(
            placements["horizontal"].slot_half,
            ATLAS_SLOT_HALF_RIGHT,
        )
        left_point = _atlas_preview_point(
            self.workspace.preview,
            atlas.resolution,
            placements["vertical"].x + 128,
            placements["vertical"].y + 128,
        )
        right_point = _atlas_preview_point(
            self.workspace.preview,
            atlas.resolution,
            placements["vertical"].x + 384,
            placements["vertical"].y + 128,
        )
        self.assertEqual(
            self.workspace.preview.object_id_at(left_point),
            "vertical",
        )
        self.assertEqual(
            self.workspace.preview.object_id_at(right_point),
            "horizontal",
        )

    def test_half_drag_snaps_to_compatible_unpaired_right_side(self) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Drag halves", 2048, atlas_id="atlas-a")
        first = _source(
            "first",
            directory=self._temporary_directory.name,
            symmetric_orientation="vertical",
            symmetric_plane_coordinate=0.0,
        )
        second = _source(
            "second",
            directory=self._temporary_directory.name,
            symmetric_orientation="horizontal",
            symmetric_plane_coordinate=2.0,
        )
        data.assign_object(
            atlas.atlas_id,
            first.object_id,
            first.texture_path,
            first.texture_resolution,
            first.packing_mode,
        )
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources([first, second])
        hover_point = _atlas_preview_point(
            self.workspace.preview,
            atlas.resolution,
            400,
            200,
        )

        preview = self.workspace.preview._drag_slot_preview_at(
            second.object_id,
            hover_point,
        )

        self.assertIsNotNone(preview)
        assert preview is not None
        self.assertTrue(preview.is_valid)
        self.assertEqual((preview.x, preview.y), (0, 0))
        self.assertEqual(preview.slot_half, ATLAS_SLOT_HALF_RIGHT)

    def test_half_resolution_change_splits_pairs_in_every_atlas(self) -> None:
        data = TextureAtlasData()
        first_512 = _source(
            "first",
            directory=self._temporary_directory.name,
            symmetric_orientation="vertical",
            symmetric_plane_coordinate=0.0,
        )
        first_1024 = _source(
            "first",
            directory=self._temporary_directory.name,
            resolution=1024,
            symmetric_orientation="vertical",
            symmetric_plane_coordinate=0.0,
        )
        partner = _source(
            "partner",
            directory=self._temporary_directory.name,
            symmetric_orientation="horizontal",
            symmetric_plane_coordinate=1.0,
        )
        for atlas_id in ("atlas-a", "atlas-b"):
            atlas = data.create_atlas(atlas_id, 2048, atlas_id=atlas_id)
            for source in (first_512, partner):
                data.assign_object(
                    atlas.atlas_id,
                    source.object_id,
                    source.texture_path,
                    source.texture_resolution,
                    source.packing_mode,
                )
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources(
            [first_512, partner],
            variant_resolver=_variant_resolver(
                {
                    (first_512.object_id, 512): first_512,
                    (first_1024.object_id, 1024): first_1024,
                    (partner.object_id, 512): partner,
                }
            ),
        )

        self.assertTrue(
            self.workspace.set_object_texture_resolution(
                first_512.object_id,
                1024,
                commit_callback=lambda: True,
            )
        )

        for atlas in self.workspace.get_data().atlases:
            resized = atlas.placement_for_object("first")
            survivor = atlas.placement_for_object("partner")
            assert resized is not None and survivor is not None
            self.assertEqual(resized.texture_resolution, 1024)
            self.assertEqual(resized.slot_half, ATLAS_SLOT_HALF_LEFT)
            self.assertEqual(survivor.slot_half, ATLAS_SLOT_HALF_LEFT)
            self.assertNotEqual(
                (resized.x, resized.y),
                (survivor.x, survivor.y),
            )

    def test_full_transition_without_spare_slot_restores_pair_and_png(self) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Full", 2048, atlas_id="atlas-a")
        target = _source(
            "target",
            directory=self._temporary_directory.name,
            resolution=1024,
            symmetric_orientation="vertical",
            symmetric_plane_coordinate=0.0,
        )
        partner = _source(
            "partner",
            directory=self._temporary_directory.name,
            resolution=1024,
            symmetric_orientation="horizontal",
            symmetric_plane_coordinate=1.0,
        )
        fillers = [
            _source(
                f"filler-{index}",
                directory=self._temporary_directory.name,
                resolution=1024,
            )
            for index in range(3)
        ]
        for source in (target, partner):
            data.assign_object(
                atlas.atlas_id,
                source.object_id,
                source.texture_path,
                source.texture_resolution,
                source.packing_mode,
            )
        for source, (x, y) in zip(
            fillers,
            ((1024, 0), (0, 1024), (1024, 1024)),
            strict=True,
        ):
            data.place_object_at(
                atlas.atlas_id,
                source.object_id,
                source.texture_path,
                source.texture_resolution,
                x,
                y,
            )
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources([target, partner, *fillers])
        self.assertEqual(self.workspace.materialize_missing_atlases(), 1)
        before = self.workspace.get_data()
        before_atlas = before.atlas_by_id(atlas.atlas_id)
        assert before_atlas is not None and before_atlas.image_path is not None
        png_path = Path(self._temporary_directory.name) / before_atlas.image_path
        png_before = png_path.read_bytes()
        full_target = _source(
            "target",
            directory=self._temporary_directory.name,
            resolution=1024,
        )
        commit = Mock(return_value=True)

        changed = self.workspace.transition_object_packing(
            target.object_id,
            [full_target],
            commit_callback=commit,
        )

        self.assertFalse(changed)
        commit.assert_not_called()
        self.assertEqual(self.workspace.get_data(), before)
        self.assertEqual(png_path.read_bytes(), png_before)

    def test_quarter_sources_share_row_major_regions_and_hit_independently(
        self,
    ) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Quarters", 2048, atlas_id="atlas-a")
        sources = [
            _source(
                object_id,
                directory=self._temporary_directory.name,
                symmetric_orientation=(
                    "vertical" if index % 2 == 0 else "horizontal"
                ),
                symmetric_plane_coordinate=float(index),
                packing_mode=ATLAS_PACKING_MODE_SYMMETRIC_QUARTER,
            )
            for index, object_id in enumerate(("one", "two", "three", "four"))
        ]
        for source in sources:
            data.assign_object(
                atlas.atlas_id,
                source.object_id,
                source.texture_path,
                source.texture_resolution,
                source.packing_mode,
            )
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources(sources)

        packed = self.workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert packed is not None
        placements = {
            placement.object_id: placement for placement in packed.placements
        }
        self.assertEqual(
            [placements[source.object_id].slot_quadrant for source in sources],
            list(ATLAS_SLOT_QUADRANT_ORDER),
        )
        hit_offsets = ((256, 256), (768, 256), (256, 768), (768, 768))
        for source, (offset_x, offset_y) in zip(
            sources,
            hit_offsets,
            strict=True,
        ):
            point = _atlas_preview_point(
                self.workspace.preview,
                atlas.resolution,
                offset_x,
                offset_y,
            )
            self.assertEqual(
                self.workspace.preview.object_id_at(point),
                source.object_id,
            )

    def test_quarter_drag_fills_first_free_region_and_same_group_is_noop(
        self,
    ) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Drag quarters", 2048, atlas_id="atlas-a")
        first = _source(
            "first",
            directory=self._temporary_directory.name,
            symmetric_orientation="vertical",
            symmetric_plane_coordinate=0.0,
            packing_mode=ATLAS_PACKING_MODE_SYMMETRIC_QUARTER,
        )
        second = _source(
            "second",
            directory=self._temporary_directory.name,
            symmetric_orientation="horizontal",
            symmetric_plane_coordinate=2.0,
            packing_mode=ATLAS_PACKING_MODE_SYMMETRIC_QUARTER,
        )
        data.assign_object(
            atlas.atlas_id,
            first.object_id,
            first.texture_path,
            first.texture_resolution,
            first.packing_mode,
        )
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources([first, second])
        hover_point = _atlas_preview_point(
            self.workspace.preview,
            atlas.resolution,
            800,
            200,
        )

        preview = self.workspace.preview._drag_slot_preview_at(
            second.object_id,
            hover_point,
        )

        assert preview is not None
        self.assertTrue(preview.is_valid)
        self.assertEqual((preview.x, preview.y, preview.size), (0, 0, 1024))
        self.assertEqual(
            preview.slot_quadrant,
            ATLAS_SLOT_QUADRANT_TOP_RIGHT,
        )
        data.assign_object(
            atlas.atlas_id,
            second.object_id,
            second.texture_path,
            second.texture_resolution,
            second.packing_mode,
        )
        self.workspace.set_data(data)
        noop_point = _atlas_preview_point(
            self.workspace.preview,
            atlas.resolution,
            200,
            800,
        )

        noop = self.workspace.preview._drag_slot_preview_at(
            second.object_id,
            noop_point,
        )

        assert noop is not None
        self.assertEqual(noop.slot_quadrant, ATLAS_SLOT_QUADRANT_TOP_RIGHT)

    def test_quarter_resolution_change_breaks_out_and_compacts_survivors(
        self,
    ) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Resize quarters", 4096, atlas_id="atlas-a")
        target_512 = _source(
            "target",
            directory=self._temporary_directory.name,
            symmetric_orientation="vertical",
            symmetric_plane_coordinate=0.0,
            packing_mode=ATLAS_PACKING_MODE_SYMMETRIC_QUARTER,
        )
        target_1024 = _source(
            "target",
            directory=self._temporary_directory.name,
            resolution=1024,
            symmetric_orientation="vertical",
            symmetric_plane_coordinate=0.0,
            packing_mode=ATLAS_PACKING_MODE_SYMMETRIC_QUARTER,
        )
        partners = [
            _source(
                f"partner-{index}",
                directory=self._temporary_directory.name,
                symmetric_orientation="horizontal",
                symmetric_plane_coordinate=float(index + 1),
                packing_mode=ATLAS_PACKING_MODE_SYMMETRIC_QUARTER,
            )
            for index in range(3)
        ]
        for source in (target_512, *partners):
            data.assign_object(
                atlas.atlas_id,
                source.object_id,
                source.texture_path,
                source.texture_resolution,
                source.packing_mode,
            )
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources(
            [target_512, *partners],
            variant_resolver=_variant_resolver(
                {
                    ("target", 512): target_512,
                    ("target", 1024): target_1024,
                    **{
                        (source.object_id, 512): source
                        for source in partners
                    },
                }
            ),
        )

        self.assertTrue(
            self.workspace.set_object_texture_resolution(
                "target",
                1024,
                commit_callback=lambda: True,
            )
        )

        updated = self.workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert updated is not None
        target = updated.placement_for_object("target")
        assert target is not None
        self.assertEqual(target.size, 2048)
        self.assertEqual(target.slot_quadrant, ATLAS_SLOT_QUADRANT_TOP_LEFT)
        survivor_placements = [
            updated.placement_for_object(source.object_id)
            for source in partners
        ]
        self.assertTrue(all(item is not None for item in survivor_placements))
        self.assertEqual(
            [item.slot_quadrant for item in survivor_placements if item is not None],
            [
                ATLAS_SLOT_QUADRANT_TOP_LEFT,
                ATLAS_SLOT_QUADRANT_TOP_RIGHT,
                ATLAS_SLOT_QUADRANT_BOTTOM_LEFT,
            ],
        )
        self.assertNotEqual(
            (target.x, target.y),
            (survivor_placements[0].x, survivor_placements[0].y),
        )

    def test_quarter_resolution_callback_rejection_restores_layout_and_png(
        self,
    ) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Quarter rollback", 2048, atlas_id="atlas-a")
        variants = {
            resolution: _source(
                "target",
                directory=self._temporary_directory.name,
                resolution=resolution,
                symmetric_orientation="vertical",
                symmetric_plane_coordinate=0.0,
                packing_mode=ATLAS_PACKING_MODE_SYMMETRIC_QUARTER,
            )
            for resolution in (512, 1024)
        }
        source = variants[1024]
        data.assign_object(
            atlas.atlas_id,
            source.object_id,
            source.texture_path,
            source.texture_resolution,
            source.packing_mode,
        )
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources(
            [source],
            variant_resolver=lambda object_id, resolution: (
                variants.get(resolution) if object_id == "target" else None
            ),
        )
        self.assertEqual(self.workspace.materialize_missing_atlases(), 1)
        before = self.workspace.get_data()
        before_atlas = before.atlas_by_id(atlas.atlas_id)
        assert before_atlas is not None and before_atlas.image_path is not None
        png_path = Path(self._temporary_directory.name) / before_atlas.image_path
        png_before = png_path.read_bytes()

        changed = self.workspace.set_object_texture_resolution(
            "target",
            512,
            commit_callback=lambda: False,
        )

        self.assertFalse(changed)
        self.assertEqual(self.workspace.get_data(), before)
        self.assertEqual(png_path.read_bytes(), png_before)

    def test_quarter_transition_maps_2048_slot_to_1024_content_variant(
        self,
    ) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Quarter transition", 2048, atlas_id="atlas-a")
        full_source = _source(
            "target",
            directory=self._temporary_directory.name,
            resolution=2048,
        )
        quarter_source = _source(
            "target",
            directory=self._temporary_directory.name,
            resolution=1024,
            symmetric_orientation="vertical",
            symmetric_plane_coordinate=0.0,
            packing_mode=ATLAS_PACKING_MODE_SYMMETRIC_QUARTER,
        )
        data.assign_object(
            atlas.atlas_id,
            full_source.object_id,
            full_source.texture_path,
            full_source.texture_resolution,
            full_source.packing_mode,
        )
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources([full_source])
        commit = Mock(return_value=True)

        changed = self.workspace.transition_object_packing(
            "target",
            [quarter_source],
            commit_callback=commit,
        )

        self.assertTrue(changed)
        commit.assert_called_once_with()
        updated = self.workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert updated is not None
        placement = updated.placement_for_object("target")
        assert placement is not None
        self.assertEqual(placement.texture_resolution, 1024)
        self.assertEqual(placement.size, 2048)
        self.assertEqual(
            placement.packing_mode,
            ATLAS_PACKING_MODE_SYMMETRIC_QUARTER,
        )
        self.assertEqual(
            placement.slot_quadrant,
            ATLAS_SLOT_QUADRANT_TOP_LEFT,
        )

    def test_square_pair_transition_maps_a_full_2048_slot_to_1024(self) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Square transition", 2048, atlas_id="atlas-a")
        full_source = _source(
            "target",
            directory=self._temporary_directory.name,
            resolution=2048,
        )
        square_source = _source(
            "target",
            directory=self._temporary_directory.name,
            resolution=1024,
            symmetric_orientation="vertical",
            symmetric_plane_coordinate=0.0,
            packing_mode=ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR,
        )
        data.assign_object(
            atlas.atlas_id,
            full_source.object_id,
            full_source.texture_path,
            full_source.texture_resolution,
            full_source.packing_mode,
        )
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources([full_source])
        commit = Mock(return_value=True)

        changed = self.workspace.transition_object_packing(
            "target",
            [square_source],
            commit_callback=commit,
        )

        self.assertTrue(changed)
        commit.assert_called_once_with()
        updated = self.workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert updated is not None
        placement = updated.placement_for_object("target")
        assert placement is not None
        self.assertEqual((placement.texture_resolution, placement.size), (1024, 1024))
        self.assertEqual(
            placement.packing_mode,
            ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR,
        )
        self.assertEqual(placement.slot_half, ATLAS_SLOT_HALF_LEFT)

    def test_pair_drag_fills_right_half_and_both_members_hit_full_height(
        self,
    ) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Pair drag", 2048, atlas_id="atlas-a")
        first = _source(
            "first",
            directory=self._temporary_directory.name,
            symmetric_orientation="vertical",
            symmetric_plane_coordinate=0.0,
            packing_mode=ATLAS_PACKING_MODE_SYMMETRIC_PAIR,
        )
        second = _source(
            "second",
            directory=self._temporary_directory.name,
            symmetric_orientation="horizontal",
            symmetric_plane_coordinate=1.0,
            packing_mode=ATLAS_PACKING_MODE_SYMMETRIC_PAIR,
        )
        data.assign_object(
            atlas.atlas_id,
            first.object_id,
            first.texture_path,
            first.texture_resolution,
            first.packing_mode,
        )
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources([first, second])
        right_half_point = _atlas_preview_point(
            self.workspace.preview,
            atlas.resolution,
            768,
            800,
        )

        preview = self.workspace.preview._drag_slot_preview_at(
            second.object_id,
            right_half_point,
        )

        assert preview is not None
        self.assertTrue(preview.is_valid)
        self.assertEqual((preview.x, preview.y, preview.size), (0, 0, 1024))
        self.assertEqual(preview.slot_half, ATLAS_SLOT_HALF_RIGHT)
        data.assign_object(
            atlas.atlas_id,
            second.object_id,
            second.texture_path,
            second.texture_resolution,
            second.packing_mode,
        )
        self.workspace.set_data(data)
        for source, atlas_x in ((first, 256), (second, 768)):
            point = _atlas_preview_point(
                self.workspace.preview,
                atlas.resolution,
                atlas_x,
                900,
            )
            self.assertEqual(
                self.workspace.preview.object_id_at(point),
                source.object_id,
            )
        noop = self.workspace.preview._drag_slot_preview_at(
            second.object_id,
            right_half_point,
        )
        assert noop is not None
        self.assertEqual(noop.slot_half, ATLAS_SLOT_HALF_RIGHT)

    def test_square_pair_uses_an_ordinary_source_and_drag_slot(self) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Square pair drag", 2048, atlas_id="atlas-a")
        first = _source(
            "first-square",
            directory=self._temporary_directory.name,
            symmetric_orientation="vertical",
            symmetric_plane_coordinate=0.0,
            packing_mode=ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR,
        )
        second = _source(
            "second-square",
            directory=self._temporary_directory.name,
            symmetric_orientation="horizontal",
            symmetric_plane_coordinate=1.0,
            packing_mode=ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR,
        )
        self.assertEqual(first.load_texture_rgba().shape, (512, 512, 4))
        data.assign_object(
            atlas.atlas_id,
            first.object_id,
            first.texture_path,
            first.texture_resolution,
            first.packing_mode,
        )
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources([first, second])
        right_half_point = _atlas_preview_point(
            self.workspace.preview,
            atlas.resolution,
            384,
            500,
        )

        preview = self.workspace.preview._drag_slot_preview_at(
            second.object_id,
            right_half_point,
        )

        assert preview is not None
        self.assertTrue(preview.is_valid)
        self.assertEqual((preview.x, preview.y, preview.size), (0, 0, 512))
        self.assertEqual(preview.slot_half, ATLAS_SLOT_HALF_RIGHT)
        data.assign_object(
            atlas.atlas_id,
            second.object_id,
            second.texture_path,
            second.texture_resolution,
            second.packing_mode,
        )
        self.workspace.set_data(data)
        for source, atlas_x in ((first, 128), (second, 384)):
            point = _atlas_preview_point(
                self.workspace.preview,
                atlas.resolution,
                atlas_x,
                500,
            )
            self.assertEqual(
                self.workspace.preview.object_id_at(point),
                source.object_id,
            )

    def test_square_pair_drag_does_not_snap_into_a_legacy_half_slot(self) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Distinct drag modes", 2048, atlas_id="atlas-a")
        legacy = _source(
            "legacy",
            directory=self._temporary_directory.name,
            symmetric_orientation="vertical",
            symmetric_plane_coordinate=0.0,
        )
        square = _source(
            "square",
            directory=self._temporary_directory.name,
            symmetric_orientation="vertical",
            symmetric_plane_coordinate=0.0,
            packing_mode=ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR,
        )
        data.assign_object(
            atlas.atlas_id,
            legacy.object_id,
            legacy.texture_path,
            legacy.texture_resolution,
            legacy.packing_mode,
        )
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources([legacy, square])
        hover_point = _atlas_preview_point(
            self.workspace.preview,
            atlas.resolution,
            384,
            256,
        )

        preview = self.workspace.preview._drag_slot_preview_at(
            square.object_id,
            hover_point,
        )

        assert preview is not None
        self.assertFalse(preview.is_valid)
        self.assertEqual(preview.slot_half, ATLAS_SLOT_HALF_LEFT)

    def test_square_pair_resolution_change_splits_and_rebuilds_black_halves(
        self,
    ) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Square resize", 2048, atlas_id="atlas-a")
        target_512 = _source(
            "square-target",
            directory=self._temporary_directory.name,
            symmetric_orientation="vertical",
            symmetric_plane_coordinate=0.0,
            packing_mode=ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR,
        )
        target_1024 = _source(
            "square-target",
            directory=self._temporary_directory.name,
            resolution=1024,
            symmetric_orientation="vertical",
            symmetric_plane_coordinate=0.0,
            packing_mode=ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR,
        )
        partner = _source(
            "square-partner",
            directory=self._temporary_directory.name,
            symmetric_orientation="horizontal",
            symmetric_plane_coordinate=1.0,
            packing_mode=ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR,
        )
        for source in (target_512, partner):
            data.assign_object(
                atlas.atlas_id,
                source.object_id,
                source.texture_path,
                source.texture_resolution,
                source.packing_mode,
            )
        variants = {
            (target_512.object_id, 512): target_512,
            (target_1024.object_id, 1024): target_1024,
            (partner.object_id, 512): partner,
        }
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources(
            [target_512, partner],
            variant_resolver=_variant_resolver(variants),
        )

        changed = self.workspace.set_object_texture_resolution(
            target_512.object_id,
            1024,
            commit_callback=lambda: True,
        )

        self.assertTrue(changed)
        updated = self.workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert updated is not None
        resized = updated.placement_for_object(target_512.object_id)
        survivor = updated.placement_for_object(partner.object_id)
        assert resized is not None and survivor is not None
        self.assertEqual((resized.texture_resolution, resized.size), (1024, 1024))
        self.assertEqual((survivor.texture_resolution, survivor.size), (512, 512))
        self.assertEqual(resized.slot_half, ATLAS_SLOT_HALF_LEFT)
        self.assertEqual(survivor.slot_half, ATLAS_SLOT_HALF_LEFT)
        self.assertNotEqual((resized.x, resized.y), (survivor.x, survivor.y))
        output_path = (
            Path(self._temporary_directory.name) / str(updated.image_path)
        )
        with Image.open(output_path) as image:
            pixels = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        self.assertTrue(
            np.all(
                pixels[
                    resized.y : resized.y + resized.size,
                    resized.x + resized.size // 2 : resized.x + resized.size,
                ]
                == (0, 0, 0, 255)
            )
        )
        self.assertTrue(
            np.all(
                pixels[
                    survivor.y : survivor.y + survivor.size,
                    survivor.x + survivor.size // 2 : survivor.x + survivor.size,
                ]
                == (0, 0, 0, 255)
            )
        )

    def test_pair_multi_atlas_resize_rejection_restores_layouts_and_pngs(
        self,
    ) -> None:
        data = TextureAtlasData()
        atlases = [
            data.create_atlas(name, 4096, atlas_id=atlas_id)
            for name, atlas_id in (
                ("First pair", "atlas-a"),
                ("Second pair", "atlas-b"),
            )
        ]
        target_variants = {
            resolution: _source(
                "target",
                directory=self._temporary_directory.name,
                resolution=resolution,
                symmetric_orientation="vertical",
                symmetric_plane_coordinate=0.0,
                packing_mode=ATLAS_PACKING_MODE_SYMMETRIC_PAIR,
            )
            for resolution in (512, 1024)
        }
        partners = [
            _source(
                f"partner-{index}",
                directory=self._temporary_directory.name,
                symmetric_orientation="horizontal",
                symmetric_plane_coordinate=float(index),
                packing_mode=ATLAS_PACKING_MODE_SYMMETRIC_PAIR,
            )
            for index in range(2)
        ]
        for atlas, partner in zip(atlases, partners, strict=True):
            for source in (target_variants[512], partner):
                data.assign_object(
                    atlas.atlas_id,
                    source.object_id,
                    source.texture_path,
                    source.texture_resolution,
                    source.packing_mode,
                )
        all_sources = [target_variants[512], *partners]
        variants = {
            (source.object_id, source.texture_resolution): source
            for source in [*target_variants.values(), *partners]
        }
        self.workspace.set_data(data)
        self.workspace.set_object_texture_sources(
            all_sources,
            variant_resolver=_variant_resolver(variants),
        )
        for atlas in self.workspace._data.atlases:
            self.workspace._materialize_atlas(atlas)
        before = self.workspace.get_data()
        pngs_before = {
            atlas.atlas_id: (
                Path(self._temporary_directory.name) / str(atlas.image_path)
            ).read_bytes()
            for atlas in before.atlases
        }
        commit = Mock(return_value=False)

        changed = self.workspace.set_object_texture_resolution(
            "target",
            1024,
            commit_callback=commit,
        )

        self.assertFalse(changed)
        commit.assert_called_once_with()
        self.assertEqual(self.workspace.get_data(), before)
        for atlas in before.atlases:
            png_path = Path(self._temporary_directory.name) / str(atlas.image_path)
            self.assertEqual(png_path.read_bytes(), pngs_before[atlas.atlas_id])

    def test_materialization_rejects_mutated_traversal_id_without_writing(
        self,
    ) -> None:
        data = TextureAtlasData()
        atlas = data.create_atlas("Unsafe", 2048, atlas_id="atlas-a")
        malicious_name = f"outside-{uuid.uuid4().hex}"
        atlas.atlas_id = f"../../{malicious_name}"
        outside_path = (
            Path(self._temporary_directory.name)
            / f"../../{malicious_name}.png"
        ).resolve()
        self.assertFalse(outside_path.exists())

        with self.assertRaisesRegex(ValueError, "escapes"):
            self.workspace._materialize_atlas(atlas)

        self.assertFalse(outside_path.exists())


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
