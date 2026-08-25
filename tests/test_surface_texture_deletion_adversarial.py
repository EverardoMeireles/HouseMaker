# ### Environment setup ###
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PIL import Image
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.generation_state import MASK_MODE_PAINT, MaskPoint, MaskStroke
from housemaker.main import BlueprintWorkspace
from housemaker.models import LevelData, RoomData, VertexData
from housemaker.surface_texture_state import (
    SurfaceTextureAssignment,
    SurfaceTextureData,
    SurfaceTextureInpaintUndoSnapshot,
)
from housemaker.surface_texture_workspace import (
    SurfaceTextureGenerationWorkspace,
)
from housemaker.texture_atlas_state import TextureAtlasData
from housemaker.texture_atlas_workspace import (
    build_atlas_wall_texture_source_id,
)


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _test_level() -> LevelData:
    vertex_data = VertexData()
    for point_x, point_y in (
        (0.0, 0.0),
        (100.0, 0.0),
        (100.0, 100.0),
        (0.0, 100.0),
        (50.0, 50.0),
    ):
        vertex_data.add_vertex(point_x, point_y)
    for start_vertex_id, end_vertex_id in ((1, 2), (2, 3), (3, 4), (4, 1)):
        vertex_data.add_edge(start_vertex_id, end_vertex_id)
    return LevelData(
        index=2,
        name="Ground",
        vertex_data=vertex_data,
        rooms=[
            RoomData(
                name="Room",
                vertex_ids=(1, 2, 3, 4),
                center_vertex_id=5,
                color_rgb=(120, 150, 180),
            )
        ],
        image_size_pixels=(100.0, 100.0),
        floor_contour_vertex_ids=(1, 2, 3, 4),
    )


def _write_texture(
    asset_directory: Path,
    file_name: str,
    color: tuple[int, int, int, int],
) -> str:
    asset_directory.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (13, 9), color).save(
        asset_directory / file_name,
        format="PNG",
    )
    return file_name


def _assignment(
    assignment_id: str,
    surface_ids: tuple[str, ...],
    asset_path: str,
) -> SurfaceTextureAssignment:
    return SurfaceTextureAssignment(
        assignment_id=assignment_id,
        surface_type="wall",
        surface_ids=surface_ids,
        provider="test",
        asset_path=asset_path,
    )


def _stroke(x: float) -> MaskStroke:
    return MaskStroke(
        mode=MASK_MODE_PAINT,
        radius_normalized=0.12,
        points=(MaskPoint(x=x, y=0.5),),
    )


# ### Standalone workspace adversarial tests ###
class SurfaceTextureDeletionAdversarialTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self._temporary_path = Path(self._temporary_directory.name)
        self.asset_directory = self._temporary_path / "surface_textures"
        self.workspace = SurfaceTextureGenerationWorkspace(
            asset_directory=self.asset_directory,
            application_settings=ApplicationSettingsStore(
                self._temporary_path / "settings.json"
            ),
        )
        self.workspace.set_levels([_test_level()])
        self.workspace.resize(1000, 700)
        self.workspace.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        self.workspace.deleteLater()
        _qt_application.processEvents()
        self._temporary_directory.cleanup()

    def test_shared_asset_and_unrelated_missing_record_do_not_block_delete(
        self,
    ) -> None:
        deleted_surface = "level:2/room:5/wall:1:2"
        shared_surface = "level:2/room:5/wall:2:3"
        missing_surface = "level:2/room:5/wall:3:4"
        shared_path = _write_texture(
            self.asset_directory,
            "shared.png",
            (80, 150, 210, 255),
        )
        deleted = _assignment(
            "delete-shared",
            (deleted_surface,),
            shared_path,
        )
        shared = _assignment(
            "keep-shared",
            (shared_surface,),
            shared_path,
        )
        missing = _assignment(
            "keep-missing",
            (missing_surface,),
            "missing.png",
        )
        shared_stroke = _stroke(0.45)
        missing_stroke = _stroke(0.75)
        self.workspace.set_data(
            SurfaceTextureData(
                selected_surface_type="wall",
                selected_surface_ids=(deleted_surface,),
                assignments=[deleted, shared, missing],
                texture_mask_strokes={
                    deleted_surface: [_stroke(0.2)],
                    shared_surface: [shared_stroke],
                    missing_surface: [missing_stroke],
                },
            )
        )

        self.assertTrue(
            self.workspace.delete_assignment_texture(deleted.assignment_id)
        )

        data = self.workspace.get_data()
        self.assertEqual(
            [assignment.assignment_id for assignment in data.assignments],
            [shared.assignment_id, missing.assignment_id],
        )
        self.assertEqual(
            data.texture_mask_strokes,
            {
                shared_surface: [shared_stroke],
                missing_surface: [missing_stroke],
            },
        )
        self.assertTrue((self.asset_directory / shared_path).is_file())
        self.assertIsNone(
            self.workspace.surface_view.get_surface_texture_rgba(
                deleted_surface
            )
        )
        self.assertIsNotNone(
            self.workspace.surface_view.get_surface_texture_rgba(shared_surface)
        )
        self.assertIsNone(
            self.workspace.surface_view.get_surface_texture_rgba(missing_surface)
        )

    def test_mask_restore_failure_rolls_back_history_viewer_and_signals(
        self,
    ) -> None:
        target_surface = "level:2/room:5/wall:1:2"
        retained_surface = "level:2/room:5/wall:2:3"
        target = _assignment(
            "rollback-target",
            (target_surface,),
            _write_texture(
                self.asset_directory,
                "rollback-target.png",
                (190, 80, 40, 255),
            ),
        )
        retained = _assignment(
            "rollback-retained",
            (retained_surface,),
            _write_texture(
                self.asset_directory,
                "rollback-retained.png",
                (30, 160, 90, 255),
            ),
        )
        snapshot = SurfaceTextureInpaintUndoSnapshot(
            previous_assignments=(target, retained),
            replacement_assignment_ids=(target.assignment_id,),
            affected_surface_ids=(target_surface,),
            previous_texture_mask_strokes={
                target_surface: (_stroke(0.3),),
            },
        )
        original = SurfaceTextureData(
            selected_surface_type="wall",
            selected_surface_ids=(target_surface,),
            assignments=[retained, target],
            texture_mask_strokes={target_surface: [_stroke(0.6)]},
            localized_inpaint_undo_stack=[snapshot],
        )
        self.workspace.set_data(original)
        removed = QSignalSpy(self.workspace.assignments_removed)
        changed = QSignalSpy(self.workspace.data_changed)
        content_changed = QSignalSpy(self.workspace.surface_content_changed)

        with patch.object(
            self.workspace.surface_view,
            "set_texture_mask_strokes",
            side_effect=[ValueError("mask restore failed"), None],
        ):
            deleted = self.workspace.delete_assignment_texture(
                target.assignment_id
            )

        self.assertFalse(deleted)
        restored = self.workspace.get_data()
        self.assertEqual(restored.assignments, original.assignments)
        self.assertEqual(
            restored.texture_mask_strokes,
            original.texture_mask_strokes,
        )
        self.assertEqual(
            restored.localized_inpaint_undo_stack,
            original.localized_inpaint_undo_stack,
        )
        self.assertEqual(removed.count(), 0)
        self.assertEqual(changed.count(), 0)
        self.assertEqual(content_changed.count(), 0)
        self.assertIsNotNone(
            self.workspace.surface_view.get_surface_texture_rgba(target_surface)
        )
        self.assertTrue(
            (self.asset_directory / target.asset_path).is_file()
        )

    def test_distinct_undo_assets_are_discarded_before_ordered_signals(
        self,
    ) -> None:
        target_surface = "level:2/room:5/wall:1:2"
        retained_surface = "level:2/room:5/wall:2:3"
        current = _assignment(
            "current-edit",
            (target_surface,),
            _write_texture(
                self.asset_directory,
                "current-edit.png",
                (190, 70, 40, 255),
            ),
        )
        historical = _assignment(
            "historical-edit",
            (target_surface,),
            _write_texture(
                self.asset_directory,
                "historical-edit.png",
                (160, 130, 100, 255),
            ),
        )
        retained = _assignment(
            "retained-wall",
            (retained_surface,),
            _write_texture(
                self.asset_directory,
                "retained-wall.png",
                (30, 170, 90, 255),
            ),
        )
        snapshot = SurfaceTextureInpaintUndoSnapshot(
            previous_assignments=(retained, historical),
            replacement_assignment_ids=(current.assignment_id,),
            affected_surface_ids=(target_surface,),
            previous_texture_mask_strokes={
                target_surface: (_stroke(0.25),),
            },
        )
        self.workspace.set_data(
            SurfaceTextureData(
                selected_surface_type="wall",
                selected_surface_ids=(target_surface,),
                assignments=[retained, current],
                texture_mask_strokes={
                    target_surface: [_stroke(0.65)],
                    retained_surface: [_stroke(0.8)],
                },
                localized_inpaint_undo_stack=[snapshot],
            )
        )
        removed = QSignalSpy(self.workspace.assignments_removed)
        signal_order: list[str] = []
        self.workspace.assignments_removed.connect(
            lambda _ids: signal_order.append("removed")
        )
        self.workspace.data_changed.connect(
            lambda _data: signal_order.append("data")
        )
        self.workspace.surface_content_changed.connect(
            lambda: signal_order.append("content")
        )

        self.assertTrue(
            self.workspace.delete_assignment_texture(current.assignment_id)
        )

        data = self.workspace.get_data()
        self.assertEqual(data.assignments, [retained])
        self.assertEqual(data.localized_inpaint_undo_stack, [])
        self.assertEqual(
            tuple(removed.at(0)[0]),
            (current.assignment_id, historical.assignment_id),
        )
        self.assertEqual(signal_order, ["removed", "data", "content"])
        self.assertFalse(
            (self.asset_directory / current.asset_path).exists()
        )
        self.assertFalse(
            (self.asset_directory / historical.asset_path).exists()
        )
        self.assertTrue(
            (self.asset_directory / retained.asset_path).is_file()
        )


# ### Main workspace adversarial integration ###
class SurfaceTextureDeletionAtlasIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self._temporary_path = Path(self._temporary_directory.name)
        self.settings = ApplicationSettingsStore(
            self._temporary_path / "settings.json"
        )
        self.workspace = BlueprintWorkspace(
            application_settings=self.settings
        )
        self.workspace.resize(1200, 760)
        self.workspace.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        self.workspace.deleteLater()
        _qt_application.processEvents()
        self._temporary_directory.cleanup()

    def test_real_delete_purges_packed_wall_before_source_refresh(self) -> None:
        surface_workspace = self.workspace.surface_texture_generation
        atlas_workspace = self.workspace.texture_atlas_workspace
        assignment = _assignment(
            "packed-wall",
            ("level:2/wall:1:2",),
            _write_texture(
                self._temporary_path / "surface_textures",
                "packed-wall.png",
                (170, 70, 25, 255),
            ),
        )
        surface_data = SurfaceTextureData(assignments=[assignment])
        surface_workspace.set_data(surface_data)
        surface_workspace.data_changed.emit(surface_data)
        source_id = build_atlas_wall_texture_source_id(
            assignment.assignment_id
        )
        source = atlas_workspace._sources_by_object_id[source_id]
        atlas_data = TextureAtlasData()
        atlas = atlas_data.create_atlas("Walls", 2048, atlas_id="walls")
        atlas_data.assign_object(
            atlas.atlas_id,
            source_id,
            source.texture_path,
            source.texture_resolution,
        )
        atlas_workspace.set_data(atlas_data)

        self.assertTrue(
            surface_workspace.delete_assignment_texture(
                assignment.assignment_id
            )
        )
        _qt_application.processEvents()

        updated_atlas = atlas_workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert updated_atlas is not None
        self.assertIsNone(updated_atlas.placement_for_object(source_id))
        self.assertNotIn(source_id, atlas_workspace._sources_by_object_id)


if __name__ == "__main__":
    unittest.main()
