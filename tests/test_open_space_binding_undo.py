# ### Environment setup ###
from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PIL import Image
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.main import BlueprintWorkspace
from housemaker.models import LevelData, VertexData
from housemaker.surface_texture_state import (
    SURFACE_TYPE_FLOOR,
    SurfaceTextureAssignment,
    SurfaceTextureData,
)
from housemaker.texture_atlas_state import (
    TextureAtlasData,
    TextureAtlasPlacement,
)
from housemaker.texture_atlas_workspace import (
    build_atlas_wall_texture_source_id,
)

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)
_BLUEPRINT_SIZE_PIXELS = 100
_BLUEPRINT_MAX_COORDINATE = float(_BLUEPRINT_SIZE_PIXELS - 1)
_FLOOR_SURFACE_ID = "level:2/floor"
_UPPER_FLOOR_SURFACE_ID = "level:3/floor"


# ### Fixture helpers ###
def _build_square_level(
    index: int,
    *,
    image_path: Path | None = None,
) -> LevelData:
    vertex_data = VertexData()
    vertex_ids = tuple(
        vertex_data.add_vertex(*point).id
        for point in (
            (0.0, 0.0),
            (_BLUEPRINT_MAX_COORDINATE, 0.0),
            (
                _BLUEPRINT_MAX_COORDINATE,
                _BLUEPRINT_MAX_COORDINATE,
            ),
            (0.0, _BLUEPRINT_MAX_COORDINATE),
        )
    )
    for start_id, end_id in zip(
        vertex_ids,
        (*vertex_ids[1:], vertex_ids[0]),
        strict=True,
    ):
        vertex_data.add_edge(start_id, end_id)
    return LevelData(
        index=index,
        name="Ground" if index == 2 else f"Level {index}",
        vertex_data=vertex_data,
        image_path=None if image_path is None else str(image_path),
        image_size_pixels=(
            float(_BLUEPRINT_SIZE_PIXELS),
            float(_BLUEPRINT_SIZE_PIXELS),
        ),
    )


def _draw_full_floor_open_space(workspace: BlueprintWorkspace) -> None:
    canvas = workspace.canvas
    workspace._handle_add_open_space_clicked()
    if not canvas.is_open_space_placement_active():
        raise AssertionError("The open-space tool did not activate.")

    start = canvas._image_to_widget(0.0, 0.0)
    canvas._open_space_drag_start_image = QPointF(0.0, 0.0)
    canvas._open_space_drag_current_image = QPointF(0.0, 0.0)
    canvas._open_space_drag_press_widget = QPointF(start)
    display_rect = canvas._image_display_rect()
    committed = canvas._finish_open_space_drag(
        QPointF(display_rect.right() + 1.0, display_rect.bottom() + 1.0)
    )
    if not committed:
        raise AssertionError("The full-floor open space did not commit.")
    _qt_application.processEvents()


def _atlas_placement(
    workspace: BlueprintWorkspace,
    atlas_id: str,
    source_id: str,
) -> TextureAtlasPlacement | None:
    atlas = workspace.texture_atlas_workspace.get_data().atlas_by_id(atlas_id)
    if atlas is None:
        return None
    return atlas.placement_for_object(source_id)


# ### Open-space binding undo integration tests ###
class OpenSpaceBindingUndoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.temporary_path = Path(self.temporary_directory.name)
        self.blueprint_path = self.temporary_path / "ground.png"
        Image.new(
            "RGB",
            (_BLUEPRINT_SIZE_PIXELS, _BLUEPRINT_SIZE_PIXELS),
            (30, 50, 70),
        ).save(self.blueprint_path)
        self.settings = ApplicationSettingsStore(
            self.temporary_path / "settings.json"
        )
        self.workspace = BlueprintWorkspace(application_settings=self.settings)
        self.ground = _build_square_level(2, image_path=self.blueprint_path)
        self.upper = _build_square_level(3)
        self.workspace._apply_project_state(
            [self.ground, self.upper],
            0,
        )
        self.workspace.resize(1200, 800)
        self.workspace.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        self.workspace.deleteLater()
        _qt_application.processEvents()
        self.temporary_directory.cleanup()

    def _seed_floor_texture(
        self,
    ) -> tuple[SurfaceTextureAssignment, str, str, TextureAtlasPlacement]:
        asset_directory = self.settings.path.parent / "surface_textures"
        asset_directory.mkdir(parents=True, exist_ok=True)
        asset_path = "ground-floor.png"
        Image.new("RGBA", (512, 512), (110, 90, 70, 255)).save(
            asset_directory / asset_path
        )
        assignment = SurfaceTextureAssignment(
            assignment_id="ground-floor-texture",
            surface_type=SURFACE_TYPE_FLOOR,
            surface_ids=(_FLOOR_SURFACE_ID,),
            provider="test",
            asset_path=asset_path,
            texture_width=512,
            texture_height=512,
        )
        surface_workspace = self.workspace.surface_texture_generation
        surface_workspace.set_data(
            SurfaceTextureData(assignments=[assignment])
        )
        surface_workspace.reconcile_assignments_with_levels(
            self.workspace.levels,
            emit_signals=False,
        )
        assignment = surface_workspace.get_assignments()[0]

        self.workspace._atlas_generation_signature = None
        self.workspace._sync_atlas_object_texture_sources(
            automatically_assign_scene_textures=False
        )
        source_id = build_atlas_wall_texture_source_id(
            assignment.assignment_id
        )
        atlas_data = TextureAtlasData()
        atlas = atlas_data.create_atlas(
            "Original floor atlas",
            2048,
            atlas_id="original-floor-atlas",
        )
        self.workspace.texture_atlas_workspace.set_data(atlas_data)
        if not self.workspace.texture_atlas_workspace.assign_source_to_selected_atlas(
            source_id
        ):
            raise AssertionError("The floor texture could not be packed.")
        placement = _atlas_placement(
            self.workspace,
            atlas.atlas_id,
            source_id,
        )
        if placement is None:
            raise AssertionError("The packed floor texture has no placement.")
        return assignment, source_id, atlas.atlas_id, placement

    def test_ctrl_z_restores_full_floor_target_and_atlas_placement(self) -> None:
        assignment, source_id, atlas_id, original_placement = (
            self._seed_floor_texture()
        )

        _draw_full_floor_open_space(self.workspace)

        self.assertEqual(len(self.ground.open_spaces), 1)
        after_opening = self.workspace.surface_texture_generation.get_assignment(
            assignment.assignment_id
        )
        self.assertIsNotNone(after_opening)
        assert after_opening is not None
        self.assertEqual(after_opening.surface_ids, ())
        self.assertIsNone(_atlas_placement(self.workspace, atlas_id, source_id))

        self.workspace._handle_canvas_undo_requested()
        _qt_application.processEvents()

        restored = self.workspace.surface_texture_generation.get_assignment(
            assignment.assignment_id
        )
        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertEqual(restored.surface_ids, (_FLOOR_SURFACE_ID,))
        self.assertEqual(restored.asset_path, assignment.asset_path)
        self.assertEqual(self.ground.open_spaces, [])
        self.assertEqual(
            _atlas_placement(self.workspace, atlas_id, source_id),
            original_placement,
        )

    def test_ctrl_z_does_not_overwrite_a_newer_surface_or_atlas_binding(
        self,
    ) -> None:
        assignment, source_id, original_atlas_id, _placement = (
            self._seed_floor_texture()
        )
        _draw_full_floor_open_space(self.workspace)
        after_opening = self.workspace.surface_texture_generation.get_assignment(
            assignment.assignment_id
        )
        self.assertIsNotNone(after_opening)
        assert after_opening is not None
        self.assertEqual(after_opening.surface_ids, ())

        atlas_data = self.workspace.texture_atlas_workspace.get_data()
        newer_atlas = atlas_data.create_atlas(
            "Newer floor atlas",
            2048,
            atlas_id="newer-floor-atlas",
        )
        atlas_data.select_atlas(newer_atlas.atlas_id)
        self.workspace.texture_atlas_workspace.set_data(atlas_data)
        newer_assignment = replace(
            after_opening,
            surface_ids=(_UPPER_FLOOR_SURFACE_ID,),
        )
        surface_workspace = self.workspace.surface_texture_generation
        surface_workspace.restore_assignment_snapshot(
            (newer_assignment,),
            emit_signals=False,
        )
        surface_workspace.reconcile_assignments_with_levels(
            self.workspace.levels,
            emit_signals=False,
        )
        newer_assignment = surface_workspace.get_assignments()[0]
        self.workspace._atlas_generation_signature = None
        self.workspace._sync_atlas_object_texture_sources(
            automatically_assign_scene_textures=False
        )
        self.assertTrue(
            self.workspace.texture_atlas_workspace.assign_source_to_selected_atlas(
                source_id
            )
        )
        newer_placement = _atlas_placement(
            self.workspace,
            newer_atlas.atlas_id,
            source_id,
        )
        self.assertIsNotNone(newer_placement)

        self.workspace._handle_canvas_undo_requested()
        _qt_application.processEvents()

        self.assertEqual(
            surface_workspace.get_assignment(assignment.assignment_id),
            newer_assignment,
        )
        self.assertIsNone(
            _atlas_placement(self.workspace, original_atlas_id, source_id)
        )
        self.assertEqual(
            _atlas_placement(
                self.workspace,
                newer_atlas.atlas_id,
                source_id,
            ),
            newer_placement,
        )
        assert self.workspace.viewer.surface_tools_status_label is not None
        self.assertIn(
            "newer texture bindings were kept",
            self.workspace.viewer.surface_tools_status_label.text(),
        )


if __name__ == "__main__":
    unittest.main()
