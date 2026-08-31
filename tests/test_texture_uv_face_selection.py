# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh
from PIL import Image
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.generation_state import (
    GeneratedObjectRecord,
    GenerationData,
)
from housemaker.generation_workspace import (
    TEXTURE_VARIANTS_PIPELINE_KEY,
    GenerationWorkspace,
)
from housemaker.texture_atlas_view import (
    TextureAtlasEntry,
    TextureAtlasView,
    UvFaceSelectionRequest,
    _build_indexed_uv_face_geometry,
    _find_uv_face_indices_at_point,
)


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])


# ### Fixture helpers ###
_LOWER_LEFT_TRIANGLE = (
    ((0.10, 0.10), (0.80, 0.10), (0.10, 0.80))
)
_UPPER_RIGHT_TRIANGLE = (
    ((0.20, 0.90), (0.90, 0.90), (0.90, 0.20))
)


def _atlas_entry(
    *,
    width: int = 160,
    height: int = 160,
    color: tuple[int, int, int, int] = (24, 36, 48, 255),
) -> TextureAtlasEntry:
    pixels = np.empty((height, width, 4), dtype=np.uint8)
    pixels[:, :] = color
    return TextureAtlasEntry("atlas", "Atlas", pixels)


def _preview_position_for_uv(
    view: TextureAtlasView,
    uv: tuple[float, float],
) -> QPoint:
    preview = view.preview_label
    pixmap = preview._scaled_base_pixmap
    if pixmap.isNull():
        raise AssertionError("The test preview has no displayed pixmap.")
    contents = preview.contentsRect()
    left = contents.x() + (contents.width() - pixmap.width()) / 2.0
    top = contents.y() + (contents.height() - pixmap.height()) / 2.0
    return QPoint(
        round(left + uv[0] * max(pixmap.width() - 1, 1)),
        round(top + (1.0 - uv[1]) * max(pixmap.height() - 1, 1)),
    )


def _textured_quad_glb() -> bytes:
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        dtype=float,
    )
    faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    uv = np.asarray(
        (
            (0.1, 0.1),
            (0.9, 0.1),
            (0.9, 0.9),
            (0.1, 0.9),
        ),
        dtype=float,
    )
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    )
    mesh.visual = TextureVisuals(
        uv=uv,
        material=PBRMaterial(
            baseColorTexture=Image.new(
                "RGBA",
                (8, 8),
                (80, 120, 160, 255),
            )
        ),
    )
    return bytes(trimesh.Scene(mesh).export(file_type="glb"))


def _textured_triangle_glb() -> bytes:
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.5, 0.0, 0.0),
            (0.0, 1.5, 0.0),
        ),
        dtype=float,
    )
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(((0, 1, 2),), dtype=np.int64),
        process=False,
    )
    mesh.visual = TextureVisuals(
        uv=np.asarray(
            ((0.55, 0.55), (0.95, 0.55), (0.55, 0.95)),
            dtype=float,
        ),
        material=PBRMaterial(
            baseColorTexture=Image.new(
                "RGBA",
                (8, 8),
                (40, 180, 80, 255),
            )
        ),
    )
    return bytes(trimesh.Scene(mesh).export(file_type="glb"))


# ### Indexed UV hit tests ###
class IndexedUvFaceHitTests(unittest.TestCase):
    def test_hits_preserve_explicit_ids_select_overlaps_and_skip_degenerate(
        self,
    ) -> None:
        degenerate = (
            ((0.10, 0.10), (0.30, 0.30), (0.50, 0.50))
        )
        geometry = _build_indexed_uv_face_geometry(
            (_LOWER_LEFT_TRIANGLE, _LOWER_LEFT_TRIANGLE, degenerate),
            (91, 7, 44),
        )

        self.assertEqual(
            _find_uv_face_indices_at_point(geometry, (0.20, 0.20)),
            (7, 91),
        )
        self.assertEqual(
            _find_uv_face_indices_at_point(geometry, (0.95, 0.95)),
            (),
        )

    def test_shared_edge_returns_both_logical_faces_deterministically(
        self,
    ) -> None:
        lower_right = (
            ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))
        )
        upper_left = (
            ((0.0, 0.0), (1.0, 1.0), (0.0, 1.0))
        )
        geometry = _build_indexed_uv_face_geometry(
            (lower_right, upper_left),
            (30, 12),
        )

        self.assertEqual(
            _find_uv_face_indices_at_point(geometry, (0.5, 0.5)),
            (12, 30),
        )

    def test_subpixel_fallback_selects_only_the_nearest_face(self) -> None:
        nearest = (
            ((0.503, 0.503), (0.506, 0.503), (0.503, 0.506))
        )
        farther = (
            ((0.510, 0.510), (0.513, 0.510), (0.510, 0.513))
        )
        geometry = _build_indexed_uv_face_geometry(
            (nearest, farther),
            (88, 99),
        )

        self.assertEqual(
            _find_uv_face_indices_at_point(geometry, (0.50, 0.50)),
            (),
        )
        self.assertEqual(
            _find_uv_face_indices_at_point(
                geometry,
                (0.50, 0.50),
                pixmap_scale=(100.0, 100.0),
            ),
            (88,),
        )


# ### Interactive texture preview tests ###
class TextureUvFaceInteractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.view = TextureAtlasView()
        self.view.resize(600, 600)
        self.view.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.view.close()
        self.view.deleteLater()
        _qt_application.processEvents()

    def test_aspect_fit_click_flips_v_rejects_letterbox_and_accepts_ctrl(
        self,
    ) -> None:
        top_triangle = (
            ((0.35, 0.65), (0.65, 0.65), (0.50, 0.95))
        )
        bottom_triangle = (
            ((0.35, 0.05), (0.65, 0.05), (0.50, 0.35))
        )
        requests: list[UvFaceSelectionRequest] = []
        self.view.uv_face_selection_requested.connect(requests.append)
        self.view.set_uv_face_selection_geometry(
            (top_triangle, bottom_triangle),
            (101, 202),
        )
        self.view.set_uv_face_selection_enabled(True)
        self.view.set_atlases((_atlas_entry(width=400, height=100),))
        _qt_application.processEvents()

        QTest.mouseClick(
            self.view.preview_label,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            _preview_position_for_uv(self.view, (0.50, 0.78)),
        )
        QTest.mouseClick(
            self.view.preview_label,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.ControlModifier,
            _preview_position_for_uv(self.view, (0.50, 0.18)),
        )

        self.assertEqual(
            requests,
            [
                UvFaceSelectionRequest((101,)),
                UvFaceSelectionRequest((202,)),
            ],
        )

        preview = self.view.preview_label
        pixmap = preview._scaled_base_pixmap
        contents = preview.contentsRect()
        pixmap_top = (
            contents.y() + (contents.height() - pixmap.height()) / 2.0
        )
        self.assertGreater(pixmap_top, contents.top() + 5)
        QTest.mouseClick(
            preview,
            Qt.MouseButton.LeftButton,
            pos=QPoint(contents.center().x(), contents.top() + 2),
        )

        self.assertEqual(len(requests), 2)

    def test_selected_face_fill_is_visible_with_wireframe_disabled(self) -> None:
        self.view.set_uv_face_selection_geometry(
            (_LOWER_LEFT_TRIANGLE,),
            (55,),
        )
        self.view.set_atlases((_atlas_entry(),))
        _qt_application.processEvents()
        self.assertFalse(self.view.uv_overlay_enabled)
        sample_uv = (0.25, 0.25)
        sample_position = _preview_position_for_uv(self.view, sample_uv)
        preview = self.view.preview_label
        pixmap = preview._scaled_base_pixmap
        contents = preview.contentsRect()
        left = contents.x() + (contents.width() - pixmap.width()) / 2.0
        top = contents.y() + (contents.height() - pixmap.height()) / 2.0
        sample_x = round(sample_position.x() - left)
        sample_y = round(sample_position.y() - top)
        before = preview.pixmap().toImage().pixelColor(sample_x, sample_y)

        self.view.set_selected_uv_face_indices((55,))

        after = preview.pixmap().toImage().pixelColor(sample_x, sample_y)
        self.assertNotEqual(after, before)
        self.assertEqual(self.view.selected_uv_face_indices, (55,))
        self.assertEqual(
            self.view.selected_uv_triangles,
            (_LOWER_LEFT_TRIANGLE,),
        )
        self.assertFalse(self.view.uv_overlay_enabled)

    def test_face_selection_cursor_requires_a_displayed_texture(self) -> None:
        self.view.set_uv_face_selection_geometry(
            (_LOWER_LEFT_TRIANGLE,),
            (55,),
        )
        self.view.set_uv_face_selection_enabled(True)

        self.assertNotEqual(
            self.view.preview_label.cursor().shape(),
            Qt.CursorShape.PointingHandCursor,
        )

        self.view.set_atlases((_atlas_entry(),))

        self.assertEqual(
            self.view.preview_label.cursor().shape(),
            Qt.CursorShape.PointingHandCursor,
        )

        self.view.set_atlases(())

        self.assertNotEqual(
            self.view.preview_label.cursor().shape(),
            Qt.CursorShape.PointingHandCursor,
        )


# ### Workspace synchronization tests ###
class TextureUvFaceWorkspaceTests(unittest.TestCase):
    def test_2d_and_3d_face_selections_share_one_authoritative_state(
        self,
    ) -> None:
        glb_bytes = _textured_quad_glb()
        with tempfile.TemporaryDirectory() as temporary_directory:
            asset_directory = Path(temporary_directory) / "assets"
            asset_directory.mkdir()
            (asset_directory / "quad.glb").write_bytes(glb_bytes)
            workspace = GenerationWorkspace(asset_directory=asset_directory)
            try:
                workspace.set_data(
                    GenerationData(
                        generated_objects=[
                            GeneratedObjectRecord(
                                object_id="quad",
                                frame_index=0,
                                object_name="Quad",
                                pipeline={},
                                provider_task_id="quad-task",
                                asset_path="quad.glb",
                            )
                        ]
                    )
                )

                self.assertTrue(
                    workspace.texture_view.uv_face_selection_enabled
                )
                self.assertEqual(workspace.texture_view.uv_face_indices, (0, 1))
                context_token = (
                    workspace.texture_view.uv_face_selection_context_token
                )

                workspace.texture_view.uv_face_selection_requested.emit(
                    UvFaceSelectionRequest(
                        (1,),
                        context_token,
                    )
                )

                self.assertEqual(
                    workspace.result_view.get_selected_face_indices(),
                    (1,),
                )
                self.assertEqual(
                    workspace.texture_view.selected_uv_face_indices,
                    (1,),
                )
                self.assertEqual(
                    workspace.face_selection_count_label.text(),
                    "1 face selected",
                )
                self.assertTrue(workspace.delete_selected_faces_button.isEnabled())

                workspace.result_view.set_selected_face_indices((0,))

                self.assertEqual(
                    workspace.texture_view.selected_uv_face_indices,
                    (0,),
                )
                np.testing.assert_allclose(
                    np.asarray(
                        workspace.texture_view.selected_uv_triangles
                    ),
                    np.asarray(
                        (
                            (
                                (0.1, 0.1),
                                (0.9, 0.1),
                                (0.9, 0.9),
                            ),
                        )
                    ),
                    rtol=0.0,
                    atol=1e-7,
                )

                workspace.texture_view.set_selected_uv_face_indices(())
                workspace.texture_view.uv_face_selection_requested.emit(
                    UvFaceSelectionRequest((), context_token)
                )

                self.assertEqual(
                    workspace.texture_view.selected_uv_face_indices,
                    (0,),
                )

                workspace.texture_view.uv_face_selection_requested.emit(
                    UvFaceSelectionRequest(
                        (1,),
                        context_token,
                    )
                )

                self.assertEqual(
                    workspace.result_view.get_selected_face_indices(),
                    (0, 1),
                )
                self.assertEqual(
                    workspace.texture_view.selected_uv_face_indices,
                    (0, 1),
                )
            finally:
                workspace.shutdown()
                workspace.close()
                workspace.deleteLater()
                _qt_application.processEvents()

    def test_model_switch_replaces_every_selection_output_and_releases_orbit(
        self,
    ) -> None:
        first_glb = _textured_quad_glb()
        second_glb = _textured_triangle_glb()
        with tempfile.TemporaryDirectory() as temporary_directory:
            asset_directory = Path(temporary_directory) / "assets"
            asset_directory.mkdir()
            (asset_directory / "first.glb").write_bytes(first_glb)
            (asset_directory / "second.glb").write_bytes(second_glb)
            Image.new("RGBA", (16, 16), (180, 30, 20, 255)).save(
                asset_directory / "first.png"
            )
            Image.new("RGBA", (16, 16), (20, 180, 30, 255)).save(
                asset_directory / "second.png"
            )

            def build_record(
                object_id: str,
                asset_name: str,
            ) -> GeneratedObjectRecord:
                return GeneratedObjectRecord(
                    object_id=object_id,
                    frame_index=0,
                    object_name=object_id.title(),
                    pipeline={
                        TEXTURE_VARIANTS_PIPELINE_KEY: {
                            "512": {
                                "glb_asset_path": f"{asset_name}.glb",
                                "texture_asset_path": f"{asset_name}.png",
                            }
                        },
                        "selected_texture_resolution": 512,
                    },
                    provider_task_id=f"{object_id}-task",
                    asset_path=f"{asset_name}.glb",
                )

            workspace = GenerationWorkspace(asset_directory=asset_directory)
            workspace.resize(1_000, 700)
            workspace.show()
            try:
                workspace.set_data(
                    GenerationData(
                        generated_objects=[
                            build_record("first", "first"),
                            build_record("second", "second"),
                        ]
                    )
                )
                workspace.generated_objects_list.setCurrentRow(0)
                _qt_application.processEvents()

                first_context = (
                    workspace.texture_view.uv_face_selection_context_token
                )
                self.assertEqual(workspace._selected_object_id, "first")
                self.assertEqual(workspace.texture_view.uv_face_indices, (0, 1))
                self.assertTrue(workspace.texture_view.entries)
                self.assertTrue(
                    all(
                        entry.owner_id == "first"
                        for entry in workspace.texture_view.entries
                    )
                )
                first_entry = workspace.texture_view.selected_entry
                self.assertIsNotNone(first_entry)
                assert first_entry is not None
                self.assertEqual(
                    first_entry.get_image().pixelColor(0, 0).getRgb(),
                    (180, 30, 20, 255),
                )
                QTest.mouseClick(
                    workspace.texture_view.preview_label,
                    Qt.MouseButton.LeftButton,
                    pos=_preview_position_for_uv(
                        workspace.texture_view,
                        (0.75, 0.25),
                    ),
                )
                with patch.object(
                    workspace.result_view,
                    "_pick_editable_face",
                    return_value=1,
                ):
                    QTest.mouseClick(
                        workspace.result_view.view,
                        Qt.MouseButton.LeftButton,
                        Qt.KeyboardModifier.ControlModifier,
                        QPoint(100, 100),
                    )

                self.assertEqual(
                    workspace.result_view.get_selected_face_indices(),
                    (0, 1),
                )
                self.assertEqual(
                    workspace.texture_view.selected_uv_face_indices,
                    (0, 1),
                )

                workspace.texture_view.uv_face_selection_requested.emit(
                    UvFaceSelectionRequest((0,), first_context)
                )

                self.assertEqual(
                    workspace.result_view.get_selected_face_indices(),
                    (1,),
                )
                self.assertEqual(
                    workspace.texture_view.selected_uv_face_indices,
                    (1,),
                )

                with patch.object(
                    workspace.result_view.view,
                    "_get_clicked_items",
                    side_effect=AssertionError(
                        "A plain object-editor click must not run GL picking."
                    ),
                ):
                    QTest.mouseClick(
                        workspace.result_view.view,
                        Qt.MouseButton.LeftButton,
                        pos=QPoint(100, 100),
                    )

                QTest.mousePress(
                    workspace.result_view.view,
                    Qt.MouseButton.MiddleButton,
                    pos=QPoint(100, 100),
                )
                self.assertTrue(
                    workspace.result_view.view.is_middle_navigation_active
                )
                workspace.generated_objects_list.setCurrentRow(1)
                _qt_application.processEvents()
                QTest.mouseRelease(
                    workspace.result_view.view,
                    Qt.MouseButton.MiddleButton,
                    pos=QPoint(100, 100),
                )

                self.assertEqual(workspace._selected_object_id, "second")
                self.assertEqual(workspace.result_view.model.glb_bytes, second_glb)
                self.assertEqual(workspace.result_view.face_edit_face_count, 1)
                self.assertEqual(workspace.texture_view.uv_face_indices, (0,))
                self.assertEqual(
                    len(workspace._object_face_geometry_cache),
                    2,
                )
                self.assertTrue(
                    all(
                        entry.owner_id == "second"
                        for entry in workspace.texture_view.entries
                    )
                )
                second_entry = workspace.texture_view.selected_entry
                self.assertIsNotNone(second_entry)
                assert second_entry is not None
                self.assertEqual(
                    second_entry.get_image().pixelColor(0, 0).getRgb(),
                    (20, 180, 30, 255),
                )
                self.assertEqual(
                    workspace.result_view.get_selected_face_indices(),
                    (),
                )
                self.assertEqual(
                    workspace.texture_view.selected_uv_face_indices,
                    (),
                )
                self.assertFalse(
                    workspace.result_view.view.is_middle_navigation_active
                )
                self.assertIsNot(
                    QWidget.mouseGrabber(),
                    workspace.result_view.view,
                )

                workspace.texture_view.uv_face_selection_requested.emit(
                    UvFaceSelectionRequest((0,), first_context)
                )

                self.assertEqual(
                    workspace.result_view.get_selected_face_indices(),
                    (),
                )

                second_context = (
                    workspace.texture_view.uv_face_selection_context_token
                )
                QTest.mouseClick(
                    workspace.texture_view.preview_label,
                    Qt.MouseButton.LeftButton,
                    pos=_preview_position_for_uv(
                        workspace.texture_view,
                        (0.65, 0.65),
                    ),
                )

                self.assertEqual(
                    workspace.result_view.get_selected_face_indices(),
                    (0,),
                )
                self.assertEqual(
                    workspace.texture_view.selected_uv_face_indices,
                    (0,),
                )
                self.assertEqual(
                    workspace.face_selection_count_label.text(),
                    "1 face selected",
                )
                self.assertTrue(
                    workspace.delete_selected_faces_button.isEnabled()
                )

                workspace.generated_objects_list.setCurrentRow(0)
                _qt_application.processEvents()

                self.assertEqual(workspace._selected_object_id, "first")
                self.assertEqual(workspace.result_view.model.glb_bytes, first_glb)
                self.assertEqual(workspace.texture_view.uv_face_indices, (0, 1))
                self.assertEqual(
                    workspace.result_view.get_selected_face_indices(),
                    (),
                )
                self.assertEqual(
                    workspace.texture_view.selected_uv_face_indices,
                    (),
                )
                restored_entry = workspace.texture_view.selected_entry
                self.assertIsNotNone(restored_entry)
                assert restored_entry is not None
                self.assertEqual(
                    restored_entry.get_image().pixelColor(0, 0).getRgb(),
                    (180, 30, 20, 255),
                )
                self.assertEqual(
                    len(workspace._object_face_geometry_cache),
                    2,
                )

                workspace.result_view.clear_face_edit_geometry()
                workspace.texture_view.clear_uv_face_selection_geometry()

                workspace.refresh_file_backed_previews()

                self.assertEqual(workspace.result_view.face_edit_face_count, 2)
                self.assertEqual(workspace.texture_view.uv_face_indices, (0, 1))
                self.assertEqual(
                    workspace.texture_view.uv_face_selection_context_token,
                    first_context,
                )
            finally:
                workspace.shutdown()
                workspace.close()
                workspace.deleteLater()
                _qt_application.processEvents()


if __name__ == "__main__":
    unittest.main()
