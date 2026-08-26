# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.generation_state import MASK_MODE_PAINT, MaskPoint, MaskStroke
from housemaker.main import BlueprintWorkspace
from housemaker.models import GROUND_LEVEL_INDEX, LevelData, RoomData, VertexData
from housemaker.surface_texture_providers import SurfaceTextureResult
from housemaker.surface_texture_state import (
    SURFACE_TYPE_WALL,
    SurfaceTextureAssignment,
    SurfaceTextureData,
    SurfaceTextureInpaintUndoSnapshot,
)
from housemaker.surface_texture_workspace import (
    SurfaceTextureGenerationWorkspace,
    SurfaceTextureRequest,
    _decode_png_rgba,
    _encode_png,
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
    return LevelData(
        index=GROUND_LEVEL_INDEX,
        name="Ground",
        vertex_data=vertex_data,
        rooms=[
            RoomData(
                name="Room",
                vertex_ids=boundary_ids,
                center_vertex_id=center.id,
                color_rgb=(120, 150, 180),
            )
        ],
        image_size_pixels=(100.0, 100.0),
        floor_contour_vertex_ids=boundary_ids,
    )


def _texture_png(
    color: tuple[int, int, int, int],
    *,
    size: tuple[int, int] = (10, 7),
) -> bytes:
    output = BytesIO()
    Image.new("RGBA", size, color).save(output, format="PNG")
    return output.getvalue()


def _mask_png(*, width: int = 10, height: int = 7) -> bytes:
    return _encode_png(np.full((height, width), 255, dtype=np.uint8))


def _stroke(x: float = 0.35) -> MaskStroke:
    return MaskStroke(
        mode=MASK_MODE_PAINT,
        radius_normalized=0.12,
        points=(MaskPoint(x=x, y=0.5),),
    )


def _assignment(
    assignment_id: str,
    surface_id: str,
    asset_path: str,
) -> SurfaceTextureAssignment:
    return SurfaceTextureAssignment(
        assignment_id=assignment_id,
        surface_type=SURFACE_TYPE_WALL,
        surface_ids=(surface_id,),
        provider="test",
        asset_path=asset_path,
        texture_width=10,
        texture_height=7,
    )


def _localized_request(
    surface_id: str,
    existing_texture_png: bytes,
) -> SurfaceTextureRequest:
    mask_png = _mask_png()
    return SurfaceTextureRequest(
        provider="meshy",
        api_key="test-key",
        reference_pngs=(existing_texture_png,),
        reference_frame_indices=(0,),
        surface_type=SURFACE_TYPE_WALL,
        surface_ids=(surface_id,),
        combined_area_m2=4.0,
        prompt="Blend the localized repair into the existing material",
        existing_texture_png=existing_texture_png,
        edit_mask_png=mask_png,
        surface_edit_mask_pngs=((surface_id, mask_png),),
    )


def _post_inpaint_state(
    surface_id: str,
    previous_assignment: SurfaceTextureAssignment,
    replacement_assignment: SurfaceTextureAssignment,
) -> SurfaceTextureData:
    return SurfaceTextureData(
        assignments=[replacement_assignment],
        localized_inpaint_undo_stack=[
            SurfaceTextureInpaintUndoSnapshot(
                previous_assignments=(previous_assignment,),
                replacement_assignment_ids=(
                    replacement_assignment.assignment_id,
                ),
                affected_surface_ids=(surface_id,),
                previous_texture_mask_strokes={surface_id: (_stroke(),)},
            )
        ],
    )


# ### Workspace transaction tests ###
class SurfaceTextureInpaintUndoTransactionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.asset_directory = (
            Path(self.temporary_directory.name) / "surface_textures"
        )
        self.asset_directory.mkdir()
        self.workspace = SurfaceTextureGenerationWorkspace(
            asset_directory=self.asset_directory
        )
        self.workspace.set_levels([_test_level()])
        self.wall_id = next(
            surface.surface_id
            for surface in self.workspace.surface_view.get_surfaces()
            if surface.surface_type == SURFACE_TYPE_WALL
        )

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        self.workspace.deleteLater()
        _qt_application.processEvents()
        self.temporary_directory.cleanup()

    def _seed_post_inpaint_state(
        self,
        *,
        previous_bytes: bytes | None,
    ) -> tuple[
        SurfaceTextureAssignment,
        SurfaceTextureAssignment,
        bytes,
    ]:
        previous_path = self.asset_directory / "before.png"
        replacement_path = self.asset_directory / "after.png"
        previous_path.unlink(missing_ok=True)
        if previous_bytes is not None:
            previous_path.write_bytes(previous_bytes)
        replacement_png = _texture_png((25, 170, 90, 255))
        replacement_path.write_bytes(replacement_png)
        previous_assignment = _assignment(
            "before",
            self.wall_id,
            previous_path.name,
        )
        replacement_assignment = _assignment(
            "after",
            self.wall_id,
            replacement_path.name,
        )
        self.workspace.set_data(
            _post_inpaint_state(
                self.wall_id,
                previous_assignment,
                replacement_assignment,
            )
        )
        return previous_assignment, replacement_assignment, replacement_png

    def test_apply_failure_rolls_back_state_view_assets_and_signals(self) -> None:
        previous_png = _texture_png((180, 40, 30, 255))
        _previous, _replacement, replacement_png = (
            self._seed_post_inpaint_state(previous_bytes=previous_png)
        )
        state_before = self.workspace.get_data()
        data_changes = QSignalSpy(self.workspace.data_changed)
        assignment_removals = QSignalSpy(self.workspace.assignments_removed)
        completed_undos = QSignalSpy(self.workspace.localized_inpaint_undone)
        original_apply = self.workspace.surface_view.set_surface_texture

        def reject_only_the_prior_asset(
            surface_ids: tuple[str, ...],
            texture: object,
        ) -> None:
            if isinstance(texture, str | Path) and Path(texture).name == "before.png":
                raise ValueError("simulated prior texture apply failure")
            original_apply(surface_ids, texture)  # type: ignore[arg-type]

        with patch.object(
            self.workspace.surface_view,
            "set_surface_texture",
            side_effect=reject_only_the_prior_asset,
        ):
            self.assertFalse(self.workspace.undo_localized_texture_inpaint())

        self.assertEqual(self.workspace.get_data(), state_before)
        self.assertEqual(data_changes.count(), 0)
        self.assertEqual(assignment_removals.count(), 0)
        self.assertEqual(completed_undos.count(), 0)
        self.assertTrue((self.asset_directory / "before.png").is_file())
        self.assertTrue((self.asset_directory / "after.png").is_file())
        np.testing.assert_array_equal(
            self.workspace.surface_view.get_surface_texture_rgba(self.wall_id),
            _decode_png_rgba(replacement_png, "Replacement texture"),
        )
        self.assertIn("could not be undone", self.workspace.status_label.text())

    def test_missing_or_corrupt_prior_png_blocks_undo_without_mutation(self) -> None:
        cases = (
            ("missing", None, "missing"),
            ("corrupt", b"this is not a PNG", "valid PNG"),
        )
        for label, previous_bytes, expected_status in cases:
            with self.subTest(label=label):
                _previous, _replacement, replacement_png = (
                    self._seed_post_inpaint_state(
                        previous_bytes=previous_bytes,
                    )
                )
                state_before = self.workspace.get_data()
                assignment_removals = QSignalSpy(
                    self.workspace.assignments_removed
                )

                self.assertFalse(
                    self.workspace.undo_localized_texture_inpaint()
                )

                self.assertEqual(self.workspace.get_data(), state_before)
                self.assertEqual(assignment_removals.count(), 0)
                self.assertTrue((self.asset_directory / "after.png").is_file())
                np.testing.assert_array_equal(
                    self.workspace.surface_view.get_surface_texture_rgba(
                        self.wall_id
                    ),
                    _decode_png_rgba(replacement_png, "Replacement texture"),
                )
                self.assertIn(expected_status, self.workspace.status_label.text())

# ### Atlas integration tests ###
class SurfaceTextureInpaintUndoAtlasTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.settings = ApplicationSettingsStore(
            Path(self.temporary_directory.name) / "settings.json"
        )
        self.workspace = BlueprintWorkspace(application_settings=self.settings)
        self.surface_workspace = self.workspace.surface_texture_generation
        self.surface_workspace.set_levels([_test_level()])
        self.wall_id = next(
            surface.surface_id
            for surface in self.surface_workspace.surface_view.get_surfaces()
            if surface.surface_type == SURFACE_TYPE_WALL
        )

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self.temporary_directory.cleanup()

    def test_undo_preserves_pinned_placement_and_rematerializes_missing_atlas(
        self,
    ) -> None:
        old_png = _texture_png((170, 70, 25, 255))
        asset_directory = self.surface_workspace._asset_directory
        asset_directory.mkdir(parents=True, exist_ok=True)
        old_path = asset_directory / "wall-before.png"
        old_path.write_bytes(old_png)
        old_assignment = _assignment(
            "wall-before",
            self.wall_id,
            old_path.name,
        )
        surface_data = SurfaceTextureData(assignments=[old_assignment])
        self.surface_workspace.set_data(surface_data)
        self.surface_workspace.data_changed.emit(surface_data)

        source_id = build_atlas_wall_texture_source_id(
            old_assignment.assignment_id
        )
        atlas_workspace = self.workspace.texture_atlas_workspace
        source = atlas_workspace._sources_by_object_id[source_id]
        atlas_data = TextureAtlasData()
        atlas = atlas_data.create_atlas("Undo atlas", 2048, atlas_id="undo-atlas")
        placement_before = atlas_data.assign_object(
            atlas.atlas_id,
            source_id,
            source.texture_path,
            source.texture_resolution,
        )
        atlas_workspace.set_data(atlas_data)
        self.assertEqual(atlas_workspace.materialize_missing_atlases(), 1)
        packed_before = atlas_workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert packed_before is not None and packed_before.image_path is not None
        atlas_png_path = (
            self.settings.path.parent
            / "texture_atlases"
            / packed_before.image_path
        )
        self.assertTrue(atlas_png_path.is_file())

        self.surface_workspace._handle_generation_succeeded(
            _localized_request(self.wall_id, old_png),
            SurfaceTextureResult(
                provider="meshy",
                texture_png=_texture_png((25, 160, 95, 255)),
                task_id="atlas-inpaint",
            ),
        )
        generated = self.surface_workspace.get_data()
        self.assertEqual(len(generated.localized_inpaint_undo_stack), 1)
        packed_during = atlas_workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert packed_during is not None
        self.assertEqual(
            packed_during.placement_for_object(source_id),
            placement_before,
        )

        atlas_png_path.unlink()
        self.assertEqual(atlas_workspace.materialize_missing_atlases(), 0)
        self.assertFalse(atlas_png_path.exists())

        self.assertTrue(
            self.surface_workspace.undo_localized_texture_inpaint()
        )
        _qt_application.processEvents()

        restored = self.surface_workspace.get_data()
        self.assertEqual(restored.assignments, [old_assignment])
        packed_after = atlas_workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert packed_after is not None
        placement_after = packed_after.placement_for_object(source_id)
        self.assertEqual(placement_after, placement_before)
        self.assertTrue(atlas_png_path.is_file())
        self.assertIn(source_id, atlas_workspace._sources_by_object_id)
        with Image.open(atlas_png_path) as atlas_image:
            sample_x = placement_before.x + placement_before.size // 2
            sample_y = placement_before.y + placement_before.size // 2
            self.assertEqual(
                atlas_image.convert("RGBA").getpixel((sample_x, sample_y)),
                (170, 70, 25, 255),
            )


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
