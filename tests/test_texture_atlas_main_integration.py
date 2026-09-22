# ### Environment setup ###
from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import numpy as np
import trimesh
from PIL import Image
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.generation_state import (
    GeneratedObjectPlacement,
    GeneratedObjectRecord,
    GenerationData,
)
from housemaker.glb import GeneratedModel
from housemaker.main import BlueprintWorkspace, _build_surface_ao_pre_atlas_scene
from housemaker.models import (
    GROUND_LEVEL_INDEX,
    LevelData,
    RoomData,
    VertexData,
    create_default_levels,
)
from housemaker.pbr_maps import (
    ATLAS_MAP_BASE_COLOR,
    PBR_MAP_METALLIC,
    PBR_MAP_NORMAL,
    PBR_MAP_ROUGHNESS,
)
from housemaker.project_io import ProjectData
from housemaker.surface_geometry import build_fixed_surfaces
from housemaker.surface_materials import SurfaceMaterialSourceSpec
from housemaker.surface_texture_state import (
    SURFACE_TEXTURE_RESOLUTIONS,
    SURFACE_TILING_MODE_EDGE_VARIANTS,
    SURFACE_TILING_MODE_WHOLE_REPEATS,
    SURFACE_TYPE_FLOOR,
    SURFACE_TYPE_WALL,
    SurfaceTextureAssignment,
    SurfaceTextureData,
    SurfaceTextureVariant,
)
from housemaker.surface_texture_workspace import (
    PreparedSurfaceTextureTilingRepair,
    PreparedSurfaceTextureTilingVariant,
    _build_surface_asset_revision,
)
from housemaker.texture_atlas_state import (
    ATLAS_PACKING_MODE_SYMMETRIC_HALF,
    ATLAS_PACKING_MODE_SYMMETRIC_PAIR,
    ATLAS_PACKING_MODE_SYMMETRIC_QUARTER,
    ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR,
    TextureAtlasData,
)
from housemaker.texture_atlas_workspace import (
    SCENE_BOUND_SOURCE_COLOR,
    build_atlas_wall_texture_source_id,
    build_texture_atlas_map_image_relative_path,
)

# ### Test application ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _wall_texture_assignment(
    asset_directory: str | Path,
    *,
    assignment_id: str = "brick-wall",
    surface_ids: tuple[str, ...] = ("level:2/wall:1:2",),
    surface_type: str = SURFACE_TYPE_WALL,
    size: tuple[int, int] = (12, 8),
    color: tuple[int, int, int, int] = (170, 70, 25, 255),
) -> SurfaceTextureAssignment:
    directory = Path(asset_directory)
    directory.mkdir(parents=True, exist_ok=True)
    asset_path = f"{assignment_id}.png"
    Image.new("RGBA", size, color).save(directory / asset_path)
    return SurfaceTextureAssignment(
        assignment_id=assignment_id,
        surface_type=surface_type,
        surface_ids=surface_ids,
        provider="test",
        asset_path=asset_path,
        texture_width=size[0],
        texture_height=size[1],
    )


def _wall_texture_assignment_with_variants(
    asset_directory: str | Path,
    *,
    assignment_id: str = "brick-wall-variants",
    surface_ids: tuple[str, ...] = ("level:2/wall:1:2",),
    selected_resolution: int = 512,
) -> SurfaceTextureAssignment:
    directory = Path(asset_directory)
    directory.mkdir(parents=True, exist_ok=True)
    variants: list[SurfaceTextureVariant] = []
    for resolution, color in zip(
        SURFACE_TEXTURE_RESOLUTIONS,
        (
            (170, 70, 25, 255),
            (70, 170, 25, 255),
            (25, 70, 170, 255),
        ),
        strict=True,
    ):
        asset_path = f"{assignment_id}.texture-{resolution}.png"
        Image.new("RGBA", (resolution, resolution), color).save(
            directory / asset_path,
            format="PNG",
        )
        variants.append(SurfaceTextureVariant(resolution, asset_path))
    active_path = next(
        variant.asset_path
        for variant in variants
        if variant.resolution == selected_resolution
    )
    return SurfaceTextureAssignment(
        assignment_id=assignment_id,
        surface_type=SURFACE_TYPE_WALL,
        surface_ids=surface_ids,
        provider="test",
        asset_path=active_path,
        texture_width=selected_resolution,
        texture_height=selected_resolution,
        texture_variants=tuple(variants),
        selected_texture_resolution=selected_resolution,
    )


def _wall_texture_assignment_with_pbr_variants(
    asset_directory: str | Path,
    *,
    assignment_id: str = "pbr-wall-variants",
    surface_ids: tuple[str, ...] = ("level:2/wall:1:2",),
    selected_resolution: int = 512,
) -> SurfaceTextureAssignment:
    directory = Path(asset_directory)
    directory.mkdir(parents=True, exist_ok=True)
    map_colors = {
        ATLAS_MAP_BASE_COLOR: (40, 60, 80, 255),
        PBR_MAP_NORMAL: (110, 120, 230, 255),
        PBR_MAP_ROUGHNESS: (70, 70, 70, 255),
        PBR_MAP_METALLIC: (190, 190, 190, 255),
    }
    variants: list[SurfaceTextureVariant] = []
    for resolution in SURFACE_TEXTURE_RESOLUTIONS:
        map_asset_paths: dict[str, str] = {}
        for map_type, color in map_colors.items():
            asset_path = (
                f"{assignment_id}.{map_type}-{resolution}.png"
            )
            Image.new("RGBA", (resolution, resolution), color).save(
                directory / asset_path,
                format="PNG",
            )
            map_asset_paths[map_type] = asset_path
        variants.append(
            SurfaceTextureVariant(
                resolution=resolution,
                asset_path=map_asset_paths[ATLAS_MAP_BASE_COLOR],
                map_asset_paths=map_asset_paths,
            )
        )
    active_variant = next(
        variant
        for variant in variants
        if variant.resolution == selected_resolution
    )
    return SurfaceTextureAssignment(
        assignment_id=assignment_id,
        surface_type=SURFACE_TYPE_WALL,
        surface_ids=surface_ids,
        provider="test",
        asset_path=active_variant.asset_path,
        texture_width=selected_resolution,
        texture_height=selected_resolution,
        texture_variants=tuple(variants),
        selected_texture_resolution=selected_resolution,
        enabled_pbr_maps=(
            PBR_MAP_NORMAL,
            PBR_MAP_ROUGHNESS,
            PBR_MAP_METALLIC,
        ),
    )


def _png_bytes(
    size: tuple[int, int],
    color: tuple[int, int, int, int],
) -> bytes:
    output = BytesIO()
    Image.new("RGBA", size, color).save(output, format="PNG")
    return output.getvalue()


def _prepared_surface_tiling_repair(
    surface_workspace,
    assignment: SurfaceTextureAssignment,
    *,
    map_colors: dict[str, tuple[int, int, int, int]],
) -> PreparedSurfaceTextureTilingRepair:
    """Build one deterministic accepted repair without running image quilting."""

    variants: list[PreparedSurfaceTextureTilingVariant] = []
    if assignment.texture_variants:
        for variant in assignment.texture_variants:
            variants.append(
                PreparedSurfaceTextureTilingVariant(
                    resolution=variant.resolution,
                    map_pngs=tuple(
                        (
                            map_type,
                            _png_bytes(
                                (variant.resolution, variant.resolution),
                                map_colors[map_type],
                            ),
                        )
                        for map_type in variant.map_asset_paths
                    ),
                )
            )
    else:
        variants.append(
            PreparedSurfaceTextureTilingVariant(
                resolution=None,
                map_pngs=(
                    (
                        ATLAS_MAP_BASE_COLOR,
                        _png_bytes(
                            (
                                assignment.texture_width or 12,
                                assignment.texture_height or 8,
                            ),
                            map_colors[ATLAS_MAP_BASE_COLOR],
                        ),
                    ),
                ),
            )
        )
    asset_directory = surface_workspace._asset_directory
    source_revisions = tuple(
        (
            raw_path,
            _build_surface_asset_revision(asset_directory, raw_path),
        )
        for raw_path in surface_workspace._assignment_asset_relative_paths(
            assignment
        )
    )
    preview = _png_bytes((12, 12), map_colors[ATLAS_MAP_BASE_COLOR])
    return PreparedSurfaceTextureTilingRepair(
        assignment=assignment,
        variants=tuple(variants),
        source_revisions=source_revisions,
        before_preview_png=preview,
        after_preview_png=preview,
        has_changes=True,
        method=SURFACE_TILING_MODE_EDGE_VARIANTS,
    )


def _generated_object_record_with_variants(
    asset_directory: str | Path,
    *,
    object_id: str,
    object_name: str,
    resolutions: tuple[int, ...],
    selected_resolution: int,
    placement: GeneratedObjectPlacement | None = None,
    pbr_map_types: tuple[str, ...] = (),
) -> GeneratedObjectRecord:
    directory = Path(asset_directory)
    directory.mkdir(parents=True, exist_ok=True)
    glb_payload = bytes(
        trimesh.Scene(trimesh.creation.box()).export(file_type="glb")
    )
    variants: dict[str, dict[str, object]] = {}
    for index, resolution in enumerate(resolutions):
        glb_name = f"{object_id}.texture-{resolution}.glb"
        texture_name = f"{object_id}.texture-{resolution}.png"
        (directory / glb_name).write_bytes(glb_payload)
        Image.new(
            "RGBA",
            (resolution, resolution),
            (40 + index * 30, 80, 120, 255),
        ).save(directory / texture_name, format="PNG")
        variant: dict[str, object] = {
            "glb_asset_path": glb_name,
            "texture_asset_path": texture_name,
        }
        if pbr_map_types:
            map_paths = {ATLAS_MAP_BASE_COLOR: texture_name}
            for map_index, map_type in enumerate(pbr_map_types):
                map_name = (
                    f"{object_id}.texture-{resolution}.{map_type}.png"
                )
                Image.new(
                    "RGBA",
                    (resolution, resolution),
                    (80 + map_index * 30, 90, 180, 255),
                ).save(directory / map_name, format="PNG")
                map_paths[map_type] = map_name
            variant["map_texture_asset_paths"] = map_paths
        variants[str(resolution)] = variant
    selected_glb_name = str(
        variants[str(selected_resolution)]["glb_asset_path"]
    )
    return GeneratedObjectRecord(
        object_id=object_id,
        frame_index=0,
        object_name=object_name,
        pipeline={
            "texture_variants": variants,
            "selected_texture_resolution": selected_resolution,
        },
        provider_task_id=f"{object_id}-task",
        asset_path=selected_glb_name,
        placement=placement,
    )


def _add_square_room_to_level(level: LevelData) -> str:
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
        strict=True,
    ):
        vertex_data.add_edge(start_id, end_id)
    center = vertex_data.add_vertex(50.0, 50.0)
    level.vertex_data = vertex_data
    level.rooms = [
        RoomData(
            name="Atlas room",
            vertex_ids=boundary_ids,
            center_vertex_id=center.id,
            color_rgb=(120, 140, 160),
        )
    ]
    return next(
        surface.surface_id
        for surface in build_fixed_surfaces([level])
        if surface.surface_type == SURFACE_TYPE_WALL
    )


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


# ### Main integration tests ###
class TextureAtlasMainIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.settings = ApplicationSettingsStore(
            Path(self._temporary_directory.name) / "settings.json"
        )
        self.workspace = BlueprintWorkspace(
            application_settings=self.settings
        )
        self.workspace.resize(1300, 800)
        self.workspace.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        self.workspace.deleteLater()
        _qt_application.processEvents()
        self._temporary_directory.cleanup()

    def test_atlas_is_after_canvas_and_embeds_its_preview(self) -> None:
        names = [
            self.workspace.workspace_tabs.tabText(index)
            for index in range(self.workspace.workspace_tabs.count())
        ]

        self.assertEqual(names[:3], ["Canvas", "3D scene", "Atlas"])
        self.workspace.workspace_tabs.setCurrentWidget(
            self.workspace.texture_atlas_workspace
        )
        _qt_application.processEvents()
        self.assertFalse(self.workspace.side_panel.isVisible())
        self.assertIsNot(
            self.workspace.workspace_tabs.currentWidget(),
            self.workspace.scene_3d_workspace,
        )
        self.assertEqual(
            self.workspace.atlas_object_preview_viewer
            .get_ambient_light_intensity(),
            1.0,
        )
        self.assertFalse(self.workspace.atlas_object_preview_viewer.isHidden())
        self.assertIs(
            self.workspace.texture_atlas_workspace.object_preview_widget,
            self.workspace.atlas_object_preview_viewer,
        )
        self.assertIs(
            self.workspace.atlas_object_preview_viewer.parentWidget(),
            self.workspace.texture_atlas_workspace.object_preview_container,
        )

    def test_permanent_object_delete_control_is_below_object_texture_list(
        self,
    ) -> None:
        atlas_workspace = self.workspace.texture_atlas_workspace
        texture_column_layout = atlas_workspace.object_list.parentWidget().layout()
        assert texture_column_layout is not None

        self.assertEqual(
            texture_column_layout.indexOf(atlas_workspace.delete_object_button),
            texture_column_layout.indexOf(atlas_workspace.object_list) + 1,
        )
        self.assertFalse(
            hasattr(self.workspace.generation, "delete_generated_object_button")
        )

    def test_save_passes_detached_atlas_state_to_project_io(self) -> None:
        atlas_data = TextureAtlasData()
        atlas_data.create_atlas("Saved Atlas", 2048, atlas_id="atlas-a")
        self.workspace.texture_atlas_workspace.set_data(atlas_data)
        save_path = Path(self._temporary_directory.name) / "project.json"

        with (
            patch(
                "housemaker.main.QFileDialog.getSaveFileName",
                return_value=(str(save_path), "JSON Files (*.json)"),
            ),
            patch("housemaker.main.save_project") as save_project_mock,
            patch("housemaker.main.QMessageBox.information"),
        ):
            self.workspace._handle_save_clicked()

        saved_data = save_project_mock.call_args.kwargs["texture_atlases"]
        self.assertEqual(saved_data, atlas_data)
        self.assertIsNot(saved_data, atlas_data)

    def test_project_load_restores_atlas_state(self) -> None:
        atlas_data = TextureAtlasData()
        atlas_data.create_atlas("Loaded Atlas", 4096, atlas_id="atlas-a")
        project = ProjectData(
            blueprint_path=None,
            current_level_index=GROUND_LEVEL_INDEX,
            levels=create_default_levels(),
            texture_atlases=atlas_data,
        )

        self.workspace.atlas_object_preview_viewer.set_model(
            _generated_box_model()
        )
        self.workspace._atlas_preview_variant_key = (
            "old-object",
            1024,
            "old.glb",
            1,
            1,
        )

        self.workspace._apply_loaded_project(project)

        loaded = self.workspace.texture_atlas_workspace.get_data()
        self.assertEqual(len(loaded.atlases), 1)
        self.assertEqual(loaded.atlases[0].name, "Loaded Atlas")
        self.assertEqual(loaded.atlases[0].resolution, 4096)
        self.assertIsNone(loaded.atlases[0].image_path)
        self.assertIsNone(self.workspace.atlas_object_preview_viewer.model)
        self.assertIsNone(self.workspace._atlas_preview_variant_key)

    def test_project_load_immediately_clears_surface_highlight_state(
        self,
    ) -> None:
        _add_square_room_to_level(self.workspace.current_level)
        surfaces = tuple(build_fixed_surfaces(self.workspace.levels))
        wall_id = next(
            surface.surface_id
            for surface in surfaces
            if surface.surface_type == SURFACE_TYPE_WALL
        )
        self.workspace.surface_texture_generation.set_levels(
            self.workspace.levels
        )
        self.workspace._set_canvas_viewer_targets(surfaces)
        assignment = _wall_texture_assignment(
            self.settings.path.parent / "surface_textures",
            assignment_id="previous-project-highlight",
            surface_ids=(wall_id,),
        )
        self.workspace.surface_texture_generation.set_data(
            SurfaceTextureData(assignments=[assignment])
        )
        source_id = build_atlas_wall_texture_source_id(
            assignment.assignment_id
        )
        self.workspace._handle_atlas_surface_texture_selected(source_id)
        self.assertEqual(
            self.workspace.viewer.get_highlighted_canvas_surface_ids(),
            (wall_id,),
        )
        self.assertEqual(
            self.workspace.texture_atlas_workspace._green_outline_source_ids,
            frozenset({source_id}),
        )
        project = ProjectData(
            blueprint_path=None,
            current_level_index=GROUND_LEVEL_INDEX,
            levels=create_default_levels(),
        )

        self.workspace._apply_loaded_project(project)

        self.assertIsNone(self.workspace._selected_atlas_surface_source_id)
        self.assertEqual(
            self.workspace.viewer.get_highlighted_canvas_surface_ids(),
            (),
        )
        self.assertEqual(
            self.workspace.texture_atlas_workspace._green_outline_source_ids,
            frozenset(),
        )

    def test_atlas_repeat_size_editor_updates_selected_surface_assignment(
        self,
    ) -> None:
        assignment = _wall_texture_assignment(
            self.settings.path.parent / "surface_textures",
            assignment_id="repeat-size-wall",
        )
        surface_workspace = self.workspace.surface_texture_generation
        surface_workspace.set_data(SurfaceTextureData(assignments=[assignment]))
        surface_workspace.data_changed.emit(surface_workspace.get_data())
        signature_before = surface_workspace.get_preview_dependency_signature()
        source_id = build_atlas_wall_texture_source_id(assignment.assignment_id)
        atlas_workspace = self.workspace.texture_atlas_workspace
        source_item = next(
            atlas_workspace.surface_list.item(index)
            for index in range(atlas_workspace.surface_list.count())
            if atlas_workspace.surface_list.item(index).data(
                Qt.ItemDataRole.UserRole
            ) == source_id
        )
        atlas_workspace.surface_list.setCurrentItem(source_item)
        _qt_application.processEvents()
        self.assertTrue(atlas_workspace.surface_texture_repeat_size_spin.isEnabled())
        self.assertEqual(atlas_workspace.surface_texture_repeat_size_spin.value(), 2.0)

        atlas_workspace.surface_texture_repeat_size_spin.setValue(1.25)
        atlas_workspace.surface_texture_repeat_size_spin.editingFinished.emit()
        _qt_application.processEvents()

        updated = surface_workspace.get_assignment(assignment.assignment_id)
        self.assertIsNotNone(updated)
        assert updated is not None
        self.assertEqual(updated.texture_repeat_size_m, 1.25)
        self.assertEqual(
            atlas_workspace._surface_texture_entries_by_id[
                source_id
            ].texture_repeat_size_m,
            1.25,
        )
        self.assertNotEqual(
            surface_workspace.get_preview_dependency_signature(),
            signature_before,
        )
        source = surface_workspace.get_surface_material_sources()[
            assignment.surface_ids[0]
        ]
        self.assertIsInstance(source, SurfaceMaterialSourceSpec)
        self.assertEqual(source.texture_repeat_size_m, 1.25)

    def test_generated_wall_texture_can_be_selected_and_added_to_atlas(
        self,
    ) -> None:
        assignment = _wall_texture_assignment(
            self.settings.path.parent / "surface_textures",
            surface_ids=(),
        )
        surface_data = SurfaceTextureData(assignments=[assignment])
        self.workspace.surface_texture_generation.set_data(surface_data)
        self.workspace.surface_texture_generation.data_changed.emit(surface_data)
        atlas_data = TextureAtlasData()
        atlas = atlas_data.create_atlas("Walls", 2048, atlas_id="walls")
        self.workspace.texture_atlas_workspace.set_data(atlas_data)

        source_id = build_atlas_wall_texture_source_id(
            assignment.assignment_id
        )
        source_items = [
            self.workspace.texture_atlas_workspace.surface_list.item(index)
            for index in range(
                self.workspace.texture_atlas_workspace.surface_list.count()
            )
        ]
        source_item = next(
            item
            for item in source_items
            if item.data(Qt.ItemDataRole.UserRole) == source_id
        )
        self.assertIn("Wall texture", source_item.text())
        self.workspace.texture_atlas_workspace.surface_list.setCurrentItem(
            source_item
        )
        self.assertTrue(
            self.workspace.texture_atlas_workspace
            .assign_source_to_selected_atlas(source_id)
        )

        packed = self.workspace.texture_atlas_workspace.get_data().atlas_by_id(
            atlas.atlas_id
        )
        assert packed is not None
        placement = packed.placement_for_object(source_id)
        assert placement is not None
        self.assertEqual(placement.texture_resolution, 512)
        self.workspace.atlas_object_preview_viewer.set_model(
            _generated_box_model()
        )
        self.assertTrue(
            self.workspace.texture_atlas_workspace
            .request_selected_object_preview()
        )
        preview_model = self.workspace.atlas_object_preview_viewer.model
        self.assertIsNotNone(preview_model)
        assert preview_model is not None
        self.assertEqual(len(preview_model.mesh.faces), 2)
        bounds = np.asarray(preview_model.mesh.bounds, dtype=float)
        np.testing.assert_allclose(bounds[:, 0], (-1.0, 1.0))
        np.testing.assert_allclose(bounds[:, 1], (0.0, 0.0))
        np.testing.assert_allclose(bounds[:, 2], (0.0, 2.0))
        np.testing.assert_allclose(
            np.asarray(preview_model.mesh.visual.uv, dtype=float),
            ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        )
        texture = preview_model.mesh.visual.material.baseColorTexture
        texture_rgba = np.asarray(texture.convert("RGBA"), dtype=np.uint8)
        self.assertEqual(texture_rgba.shape, (512, 512, 4))
        np.testing.assert_array_equal(
            texture_rgba[256, 256],
            np.asarray((170, 70, 25, 255), dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            texture_rgba[0, 0],
            np.asarray((0, 0, 0, 0), dtype=np.uint8),
        )
        self.assertNotIn(
            "3D texture variant is missing",
            self.workspace.texture_atlas_workspace.status_label.text(),
        )

    def test_atlas_object_selection_syncs_canvas_and_place_uses_exact_id(
        self,
    ) -> None:
        record = _generated_object_record_with_variants(
            self.settings.path.parent / "generated",
            object_id="atlas-chair",
            object_name="Atlas chair",
            resolutions=(512,),
            selected_resolution=512,
        )
        generation_data = GenerationData(generated_objects=[record])
        self.workspace.generation.set_data(generation_data)
        self.workspace.generation.data_changed.emit(generation_data)
        atlas_workspace = self.workspace.texture_atlas_workspace

        with (
            patch.object(
                self.workspace.viewer,
                "select_placed_object",
                return_value=True,
            ) as select_object,
            patch.object(
                self.workspace.generation,
                "request_placeable_object_placement",
                return_value=True,
            ) as request_placement,
        ):
            atlas_workspace.object_list.object_clicked.emit(
                record.object_id,
                Qt.MouseButton.LeftButton,
            )
            atlas_workspace.place_assign_button.click()

        self.assertEqual(
            select_object.call_args_list,
            [call(None), call(record.object_id)],
        )
        request_placement.assert_called_once_with(record.object_id)

    def test_atlas_multi_object_selection_syncs_group_to_canvas(self) -> None:
        records = tuple(
            _generated_object_record_with_variants(
                self.settings.path.parent / "generated",
                object_id=object_id,
                object_name=object_id.title(),
                resolutions=(512,),
                selected_resolution=512,
            )
            for object_id in ("chair", "table")
        )
        data = GenerationData(generated_objects=list(records))
        self.workspace.generation.set_data(data)
        self.workspace.generation.data_changed.emit(data)
        atlas_workspace = self.workspace.texture_atlas_workspace
        source_list = atlas_workspace.object_list

        with patch.object(
            self.workspace.viewer,
            "set_selected_placed_object_ids",
        ) as select_objects:
            for row, modifiers in (
                (0, Qt.KeyboardModifier.NoModifier),
                (1, Qt.KeyboardModifier.ControlModifier),
            ):
                QTest.mouseClick(
                    source_list.viewport(),
                    Qt.MouseButton.LeftButton,
                    modifiers,
                    source_list.visualItemRect(source_list.item(row)).center(),
                )

        self.assertEqual(
            atlas_workspace.selected_object_texture_ids,
            ("chair", "table"),
        )
        self.assertEqual(self.workspace._desired_canvas_object_ids, ("chair", "table"))
        select_objects.assert_called_with(
            ("chair", "table"),
            active_object_id="table",
        )

    def test_atlas_multi_surface_selection_highlights_surface_union(self) -> None:
        _add_square_room_to_level(self.workspace.current_level)
        surfaces = tuple(build_fixed_surfaces(self.workspace.levels))
        wall_ids = tuple(
            surface.surface_id
            for surface in surfaces
            if surface.surface_type == SURFACE_TYPE_WALL
        )[:2]
        self.assertEqual(len(wall_ids), 2)
        self.workspace.surface_texture_generation.set_levels(self.workspace.levels)
        self.workspace._set_canvas_viewer_targets(surfaces)
        assignments = tuple(
            _wall_texture_assignment(
                self.settings.path.parent / "surface_textures",
                assignment_id=f"wall-{index}",
                surface_ids=(wall_id,),
            )
            for index, wall_id in enumerate(wall_ids)
        )
        data = SurfaceTextureData(assignments=list(assignments))
        self.workspace.surface_texture_generation.set_data(data)
        self.workspace.surface_texture_generation.data_changed.emit(data)
        atlas_workspace = self.workspace.texture_atlas_workspace
        source_list = atlas_workspace.surface_list

        for row, modifiers in (
            (0, Qt.KeyboardModifier.NoModifier),
            (1, Qt.KeyboardModifier.ControlModifier),
        ):
            QTest.mouseClick(
                source_list.viewport(),
                Qt.MouseButton.LeftButton,
                modifiers,
                source_list.visualItemRect(source_list.item(row)).center(),
            )

        source_ids = tuple(
            build_atlas_wall_texture_source_id(assignment.assignment_id)
            for assignment in assignments
        )
        self.assertEqual(atlas_workspace.selected_surface_texture_ids, source_ids)
        self.assertEqual(
            set(self.workspace.viewer.get_highlighted_canvas_surface_ids()),
            set(wall_ids),
        )
        self.assertEqual(
            atlas_workspace._green_outline_source_ids,
            frozenset(source_ids),
        )

    def test_geometry_only_object_can_be_previewed_and_placed_from_atlas(
        self,
    ) -> None:
        textured_record = _generated_object_record_with_variants(
            self.settings.path.parent / "generated",
            object_id="geometry-only-table",
            object_name="Geometry only table",
            resolutions=(512,),
            selected_resolution=512,
        )
        record = replace(textured_record, pipeline={})
        generation_data = GenerationData(generated_objects=[record])
        self.workspace.generation.set_data(generation_data)
        self.workspace.generation.data_changed.emit(generation_data)
        atlas_workspace = self.workspace.texture_atlas_workspace
        item = next(
            atlas_workspace.object_list.item(row)
            for row in range(atlas_workspace.object_list.count())
            if atlas_workspace.object_list.item(row).data(
                Qt.ItemDataRole.UserRole
            )
            == record.object_id
        )

        self.assertEqual(item.text(), "[No texture] Geometry only table")
        atlas_workspace.object_list.object_clicked.emit(
            record.object_id,
            Qt.MouseButton.LeftButton,
        )
        self.assertIsNotNone(self.workspace.atlas_object_preview_viewer.model)

        with patch.object(
            self.workspace.generation,
            "request_placeable_object_placement",
            return_value=True,
        ) as request_placement:
            atlas_workspace.place_assign_button.click()

        request_placement.assert_called_once_with(record.object_id)

    def test_remove_action_unplaces_object_and_unpacks_its_texture(self) -> None:
        record = _generated_object_record_with_variants(
            self.settings.path.parent / "generated",
            object_id="removable-chair",
            object_name="Removable chair",
            resolutions=(512,),
            selected_resolution=512,
            placement=GeneratedObjectPlacement(
                level_index=self.workspace.current_level.index,
                image_x=30.0,
                image_y=45.0,
            ),
        )
        self.workspace.generation.set_data(
            GenerationData(generated_objects=[record])
        )
        atlas_data = TextureAtlasData()
        atlas = atlas_data.create_atlas("Objects", 2048, atlas_id="objects")
        self.workspace.texture_atlas_workspace.set_data(atlas_data)
        self.workspace._atlas_generation_signature = None
        self.workspace._sync_atlas_object_texture_sources(
            automatically_assign_scene_textures=False
        )
        atlas_workspace = self.workspace.texture_atlas_workspace
        self.assertTrue(
            atlas_workspace.assign_source_to_selected_atlas(record.object_id)
        )
        atlas_workspace.object_list.object_clicked.emit(
            record.object_id,
            Qt.MouseButton.LeftButton,
        )
        selected_item = atlas_workspace.object_list.currentItem()
        assert selected_item is not None
        self.assertEqual(
            selected_item.foreground().color(),
            SCENE_BOUND_SOURCE_COLOR,
        )
        self.assertEqual(
            atlas_workspace._green_outline_source_ids,
            frozenset(),
        )

        atlas_workspace.remove_source_button.click()

        retained = self.workspace.generation.get_data().generated_objects
        self.assertEqual(len(retained), 1)
        self.assertEqual(retained[0].object_id, record.object_id)
        self.assertIsNone(retained[0].placement)
        self.assertTrue(
            (self.settings.path.parent / "generated" / record.asset_path)
            .is_file()
        )
        updated_atlas = atlas_workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert updated_atlas is not None
        self.assertIsNone(updated_atlas.placement_for_object(record.object_id))
        source_item = next(
            atlas_workspace.object_list.item(row)
            for row in range(atlas_workspace.object_list.count())
            if atlas_workspace.object_list.item(row).data(
                Qt.ItemDataRole.UserRole
            )
            == record.object_id
        )
        self.assertEqual(
            source_item.foreground().style(),
            Qt.BrushStyle.NoBrush,
        )

    def test_remove_action_unplaces_geometry_only_object(self) -> None:
        textured_record = _generated_object_record_with_variants(
            self.settings.path.parent / "generated",
            object_id="removable-geometry",
            object_name="Removable geometry",
            resolutions=(512,),
            selected_resolution=512,
            placement=GeneratedObjectPlacement(
                level_index=self.workspace.current_level.index,
                image_x=20.0,
                image_y=25.0,
            ),
        )
        record = replace(textured_record, pipeline={})
        self.workspace.generation.set_data(
            GenerationData(generated_objects=[record])
        )
        self.workspace._atlas_generation_signature = None
        self.workspace._sync_atlas_object_texture_sources(
            automatically_assign_scene_textures=False
        )
        atlas_workspace = self.workspace.texture_atlas_workspace
        atlas_workspace.object_list.object_clicked.emit(
            record.object_id,
            Qt.MouseButton.LeftButton,
        )

        atlas_workspace.remove_source_button.click()

        retained = self.workspace.generation.get_data().generated_objects
        self.assertEqual(len(retained), 1)
        self.assertIsNone(retained[0].placement)

    def test_active_generation_operation_is_placeable_from_atlas(self) -> None:
        operation_id = "active-model-operation"
        atlas_workspace = self.workspace.texture_atlas_workspace
        with patch.object(
            self.workspace.generation,
            "get_placeable_object_names_by_id",
            return_value={operation_id: "Generating cupboard"},
        ):
            self.workspace.generation.placeable_objects_changed.emit(
                {operation_id: "Generating cupboard"}
            )

        item = next(
            atlas_workspace.object_list.item(row)
            for row in range(atlas_workspace.object_list.count())
            if atlas_workspace.object_list.item(row).data(
                Qt.ItemDataRole.UserRole
            )
            == operation_id
        )
        self.assertEqual(item.text(), "[No texture] Generating cupboard")

        with patch.object(
            self.workspace.generation,
            "request_placeable_object_placement",
            return_value=True,
        ) as request_placement:
            atlas_workspace.object_list.object_clicked.emit(
                operation_id,
                Qt.MouseButton.LeftButton,
            )
            atlas_workspace.place_assign_button.click()

        request_placement.assert_called_once_with(operation_id)
        self.assertEqual(
            self.workspace._desired_canvas_object_id,
            operation_id,
        )

        self.workspace.generation.placeable_objects_changed.emit({})

        self.assertIsNone(atlas_workspace.selected_object_id)
        self.assertIsNone(self.workspace._desired_canvas_object_id)

    def test_atlas_surface_assign_replaces_usage_and_removes_empty_source(
        self,
    ) -> None:
        _add_square_room_to_level(self.workspace.current_level)
        surfaces = tuple(build_fixed_surfaces(self.workspace.levels))
        wall_ids = tuple(
            surface.surface_id
            for surface in surfaces
            if surface.surface_type == SURFACE_TYPE_WALL
        )
        first_wall, second_wall = wall_ids[:2]
        self.workspace.surface_texture_generation.set_levels(
            self.workspace.levels
        )
        self.workspace._set_canvas_viewer_targets(surfaces)

        first_assignment = _wall_texture_assignment(
            self.settings.path.parent / "surface_textures",
            assignment_id="first-wall-texture",
            surface_ids=(first_wall,),
        )
        second_assignment = _wall_texture_assignment(
            self.settings.path.parent / "surface_textures",
            assignment_id="second-wall-texture",
            surface_ids=(second_wall,),
            color=(30, 110, 190, 255),
        )
        atlas_data = TextureAtlasData()
        atlas = atlas_data.create_atlas("Surfaces", 2048, atlas_id="surfaces")
        self.workspace.texture_atlas_workspace.set_data(atlas_data)
        surface_data = SurfaceTextureData(
            assignments=[first_assignment, second_assignment]
        )
        self.workspace.surface_texture_generation.set_data(surface_data)
        self.workspace.surface_texture_generation.data_changed.emit(surface_data)

        first_source_id = build_atlas_wall_texture_source_id(
            first_assignment.assignment_id
        )
        second_source_id = build_atlas_wall_texture_source_id(
            second_assignment.assignment_id
        )
        self.assertTrue(
            self.workspace.texture_atlas_workspace
            .is_source_assigned_to_any_atlas(second_source_id)
        )

        self.workspace.viewer.select_wall_target(second_wall)
        self.workspace.texture_atlas_workspace.surface_list.object_clicked.emit(
            first_source_id,
            Qt.MouseButton.LeftButton,
        )
        self.workspace.texture_atlas_workspace.place_assign_button.click()

        assignments = {
            assignment.assignment_id: assignment
            for assignment in self.workspace.surface_texture_generation
            .get_data().assignments
        }
        self.assertEqual(
            assignments[first_assignment.assignment_id].surface_ids,
            (first_wall, second_wall),
        )
        self.assertEqual(
            assignments[second_assignment.assignment_id].surface_ids,
            (),
        )
        packed_atlas = (
            self.workspace.texture_atlas_workspace.get_data().atlas_by_id(
                atlas.atlas_id
            )
        )
        assert packed_atlas is not None
        self.assertIsNotNone(packed_atlas.placement_for_object(first_source_id))
        self.assertIsNone(packed_atlas.placement_for_object(second_source_id))
        surface_labels = tuple(
            self.workspace.texture_atlas_workspace.surface_list.item(row).text()
            for row in range(
                self.workspace.texture_atlas_workspace.surface_list.count()
            )
        )
        self.assertTrue(any("2 surfaces" in label for label in surface_labels))
        self.assertTrue(any("0 surfaces" in label for label in surface_labels))
        self.assertEqual(
            set(self.workspace.viewer.get_selected_canvas_surface_ids()),
            {second_wall},
        )
        self.assertEqual(
            set(self.workspace.viewer.get_highlighted_canvas_surface_ids()),
            {first_wall, second_wall},
        )

    def test_remove_action_unassigns_surfaces_and_unpacks_texture(self) -> None:
        _add_square_room_to_level(self.workspace.current_level)
        surfaces = tuple(build_fixed_surfaces(self.workspace.levels))
        wall_ids = tuple(
            surface.surface_id
            for surface in surfaces
            if surface.surface_type == SURFACE_TYPE_WALL
        )
        assigned_wall, manual_target = wall_ids[:2]
        self.workspace.surface_texture_generation.set_levels(
            self.workspace.levels
        )
        self.workspace._set_canvas_viewer_targets(surfaces)
        assignment = _wall_texture_assignment(
            self.settings.path.parent / "surface_textures",
            assignment_id="removable-surface-texture",
            surface_ids=(assigned_wall,),
        )
        self.workspace.surface_texture_generation.set_data(
            SurfaceTextureData(assignments=[assignment])
        )
        atlas_data = TextureAtlasData()
        atlas = atlas_data.create_atlas("Surfaces", 2048, atlas_id="surfaces")
        self.workspace.texture_atlas_workspace.set_data(atlas_data)
        self.workspace._atlas_generation_signature = None
        self.workspace._sync_atlas_object_texture_sources(
            automatically_assign_scene_textures=False
        )
        source_id = build_atlas_wall_texture_source_id(
            assignment.assignment_id
        )
        atlas_workspace = self.workspace.texture_atlas_workspace
        self.assertTrue(
            atlas_workspace.assign_source_to_selected_atlas(source_id)
        )
        self.workspace.viewer.select_wall_target(manual_target)
        atlas_workspace.surface_list.object_clicked.emit(
            source_id,
            Qt.MouseButton.LeftButton,
        )
        selected_item = atlas_workspace.surface_list.currentItem()
        assert selected_item is not None
        self.assertEqual(
            selected_item.foreground().color(),
            SCENE_BOUND_SOURCE_COLOR,
        )
        self.assertEqual(
            self.workspace.viewer.get_highlighted_canvas_surface_ids(),
            (assigned_wall,),
        )
        self.assertEqual(
            atlas_workspace._green_outline_source_ids,
            frozenset({source_id}),
        )

        atlas_workspace.remove_source_button.click()

        retained = self.workspace.surface_texture_generation.get_assignment(
            assignment.assignment_id
        )
        assert retained is not None
        self.assertEqual(retained.surface_ids, ())
        self.assertTrue(
            (
                self.settings.path.parent
                / "surface_textures"
                / assignment.asset_path
            ).is_file()
        )
        updated_atlas = atlas_workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert updated_atlas is not None
        self.assertIsNone(updated_atlas.placement_for_object(source_id))
        self.assertEqual(
            self.workspace.viewer.get_selected_canvas_surface_ids(),
            (manual_target,),
        )
        self.assertEqual(
            self.workspace.viewer.get_highlighted_canvas_surface_ids(),
            (),
        )
        self.assertEqual(
            atlas_workspace._green_outline_source_ids,
            frozenset(),
        )

    def test_delete_surface_texture_clears_binding_before_atlas_resync(
        self,
    ) -> None:
        _add_square_room_to_level(self.workspace.current_level)
        surfaces = tuple(build_fixed_surfaces(self.workspace.levels))
        wall_id = next(
            surface.surface_id
            for surface in surfaces
            if surface.surface_type == SURFACE_TYPE_WALL
        )
        self.workspace.surface_texture_generation.set_levels(
            self.workspace.levels
        )
        self.workspace._set_canvas_viewer_targets(surfaces)
        assignment = _wall_texture_assignment(
            self.settings.path.parent / "surface_textures",
            assignment_id="delete-key-surface-texture",
            surface_ids=(wall_id,),
        )
        self.workspace.surface_texture_generation.set_data(
            SurfaceTextureData(assignments=[assignment])
        )
        atlas_data = TextureAtlasData()
        first_atlas = atlas_data.create_atlas(
            "First",
            2048,
            atlas_id="first",
        )
        atlas_data.create_atlas("Second", 2048, atlas_id="second")
        atlas_data.select_atlas(first_atlas.atlas_id)
        atlas_workspace = self.workspace.texture_atlas_workspace
        atlas_workspace.set_data(atlas_data)
        self.workspace._atlas_generation_signature = None
        self.workspace._sync_atlas_object_texture_sources(
            automatically_assign_scene_textures=False
        )
        source_id = build_atlas_wall_texture_source_id(
            assignment.assignment_id
        )
        self.assertTrue(
            atlas_workspace.assign_source_to_selected_atlas(source_id)
        )
        atlas_workspace.surface_list.object_clicked.emit(
            source_id,
            Qt.MouseButton.LeftButton,
        )

        atlas_workspace.remove_selected_texture_from_atlas()

        retained_assignment = (
            self.workspace.surface_texture_generation.get_assignment(
                assignment.assignment_id
            )
        )
        assert retained_assignment is not None
        self.assertEqual(retained_assignment.surface_ids, ())
        removed_atlas = atlas_workspace.get_data().atlas_by_id(
            first_atlas.atlas_id
        )
        assert removed_atlas is not None
        self.assertIsNone(removed_atlas.placement_for_object(source_id))

        atlas_workspace.atlas_list.setCurrentRow(1)
        atlas_workspace.atlas_list.setCurrentRow(0)

        resynchronized_atlas = atlas_workspace.get_data().atlas_by_id(
            first_atlas.atlas_id
        )
        assert resynchronized_atlas is not None
        self.assertIsNone(
            resynchronized_atlas.placement_for_object(source_id)
        )

    def test_surface_replacement_reclaims_displaced_full_atlas_slot(
        self,
    ) -> None:
        wall_id = _add_square_room_to_level(self.workspace.current_level)
        surfaces = tuple(build_fixed_surfaces(self.workspace.levels))
        self.workspace.surface_texture_generation.set_levels(
            self.workspace.levels
        )
        self.workspace._set_canvas_viewer_targets(surfaces)
        asset_directory = self.settings.path.parent / "surface_textures"
        old_assignment = _wall_texture_assignment(
            asset_directory,
            assignment_id="full-slot-old",
            surface_ids=(wall_id,),
            size=(2048, 2048),
        )
        replacement_assignment = _wall_texture_assignment(
            asset_directory,
            assignment_id="full-slot-replacement",
            surface_ids=(),
            size=(2048, 2048),
            color=(20, 100, 180, 255),
        )
        self.workspace.surface_texture_generation.set_data(
            SurfaceTextureData(
                assignments=[old_assignment, replacement_assignment]
            )
        )
        atlas_data = TextureAtlasData()
        atlas = atlas_data.create_atlas("Full surface", 2048, atlas_id="full")
        self.workspace.texture_atlas_workspace.set_data(atlas_data)
        self.workspace._atlas_generation_signature = None
        self.workspace._sync_atlas_object_texture_sources(
            automatically_assign_scene_textures=False
        )
        old_source_id = build_atlas_wall_texture_source_id(
            old_assignment.assignment_id
        )
        replacement_source_id = build_atlas_wall_texture_source_id(
            replacement_assignment.assignment_id
        )
        self.assertTrue(
            self.workspace.texture_atlas_workspace
            .assign_source_to_selected_atlas(old_source_id)
        )
        self.workspace.viewer.select_wall_target(wall_id)
        self.workspace.texture_atlas_workspace.surface_list.object_clicked.emit(
            replacement_source_id,
            Qt.MouseButton.LeftButton,
        )

        with patch("housemaker.main.QMessageBox.question") as question:
            self.workspace.texture_atlas_workspace.place_assign_button.click()

        question.assert_not_called()
        updated_atlas_data = self.workspace.texture_atlas_workspace.get_data()
        self.assertEqual(len(updated_atlas_data.atlases), 1)
        updated_atlas = updated_atlas_data.atlas_by_id(atlas.atlas_id)
        assert updated_atlas is not None
        self.assertIsNone(updated_atlas.placement_for_object(old_source_id))
        self.assertIsNotNone(
            updated_atlas.placement_for_object(replacement_source_id)
        )
        old_assignment_after = (
            self.workspace.surface_texture_generation.get_assignment(
                old_assignment.assignment_id
            )
        )
        replacement_after = (
            self.workspace.surface_texture_generation.get_assignment(
                replacement_assignment.assignment_id
            )
        )
        assert old_assignment_after is not None
        assert replacement_after is not None
        self.assertEqual(old_assignment_after.surface_ids, ())
        self.assertEqual(replacement_after.surface_ids, (wall_id,))

    def test_browsing_surface_textures_preserves_manual_canvas_target(
        self,
    ) -> None:
        _add_square_room_to_level(self.workspace.current_level)
        surfaces = tuple(build_fixed_surfaces(self.workspace.levels))
        wall_ids = tuple(
            surface.surface_id
            for surface in surfaces
            if surface.surface_type == SURFACE_TYPE_WALL
        )
        first_wall, second_wall, manual_target = wall_ids[:3]
        self.workspace.surface_texture_generation.set_levels(
            self.workspace.levels
        )
        self.workspace._set_canvas_viewer_targets(surfaces)
        first_assignment = _wall_texture_assignment(
            self.settings.path.parent / "surface_textures",
            assignment_id="first-browsed-texture",
            surface_ids=(first_wall,),
        )
        second_assignment = _wall_texture_assignment(
            self.settings.path.parent / "surface_textures",
            assignment_id="second-browsed-texture",
            surface_ids=(second_wall,),
            color=(20, 90, 180, 255),
        )
        self.workspace.surface_texture_generation.set_data(
            SurfaceTextureData(
                assignments=[first_assignment, second_assignment]
            )
        )

        self.workspace.viewer.select_wall_target(manual_target)
        self.workspace._handle_atlas_surface_texture_selected(
            build_atlas_wall_texture_source_id(first_assignment.assignment_id)
        )
        self.workspace._handle_atlas_surface_texture_selected(
            build_atlas_wall_texture_source_id(second_assignment.assignment_id)
        )

        self.assertEqual(
            self.workspace._atlas_surface_assignment_targets(second_assignment),
            (manual_target,),
        )
        self.assertEqual(
            self.workspace.viewer.get_selected_canvas_surface_ids(),
            (manual_target,),
        )
        self.assertEqual(
            self.workspace.viewer.get_highlighted_canvas_surface_ids(),
            (second_wall,),
        )
        self.assertEqual(
            self.workspace.texture_atlas_workspace._green_outline_source_ids,
            frozenset(
                {
                    build_atlas_wall_texture_source_id(
                        second_assignment.assignment_id
                    )
                }
            ),
        )

    def test_selected_surface_texture_usage_resyncs_canvas_highlight(
        self,
    ) -> None:
        _add_square_room_to_level(self.workspace.current_level)
        surfaces = tuple(build_fixed_surfaces(self.workspace.levels))
        wall_ids = tuple(
            surface.surface_id
            for surface in surfaces
            if surface.surface_type == SURFACE_TYPE_WALL
        )
        first_wall, second_wall = wall_ids[:2]
        self.workspace.surface_texture_generation.set_levels(
            self.workspace.levels
        )
        self.workspace._set_canvas_viewer_targets(surfaces)
        assignment = _wall_texture_assignment(
            self.settings.path.parent / "surface_textures",
            assignment_id="externally-retargeted-texture",
            surface_ids=(first_wall,),
        )
        surface_workspace = self.workspace.surface_texture_generation
        surface_workspace.set_data(
            SurfaceTextureData(assignments=[assignment])
        )
        surface_workspace.data_changed.emit(surface_workspace.get_data())
        source_id = build_atlas_wall_texture_source_id(
            assignment.assignment_id
        )
        self.workspace.texture_atlas_workspace.surface_list.object_clicked.emit(
            source_id,
            Qt.MouseButton.LeftButton,
        )
        self.assertEqual(
            self.workspace.viewer.get_selected_canvas_surface_ids(),
            (),
        )
        self.assertEqual(
            self.workspace.viewer.get_highlighted_canvas_surface_ids(),
            (first_wall,),
        )

        updated_assignment = replace(
            assignment,
            surface_ids=(second_wall,),
        )
        updated_data = SurfaceTextureData(assignments=[updated_assignment])
        surface_workspace.set_data(updated_data)
        surface_workspace.data_changed.emit(updated_data)

        self.assertEqual(
            self.workspace.viewer.get_selected_canvas_surface_ids(),
            (),
        )
        self.assertEqual(
            self.workspace._desired_canvas_surface_ids,
            (),
        )
        self.assertEqual(
            self.workspace.viewer.get_highlighted_canvas_surface_ids(),
            (second_wall,),
        )

    def test_rebuilt_canvas_targets_clear_removed_surface_selection(
        self,
    ) -> None:
        _add_square_room_to_level(self.workspace.current_level)
        surfaces = tuple(build_fixed_surfaces(self.workspace.levels))
        wall_id = next(
            surface.surface_id
            for surface in surfaces
            if surface.surface_type == SURFACE_TYPE_WALL
        )
        self.workspace._set_canvas_viewer_targets(surfaces)
        self.workspace.viewer.select_wall_target(wall_id)
        self.assertEqual(
            self.workspace._atlas_surface_assignment_target_ids,
            (wall_id,),
        )

        self.workspace._set_canvas_viewer_targets(
            tuple(
                surface
                for surface in surfaces
                if surface.surface_id != wall_id
            )
        )

        self.assertEqual(self.workspace._desired_canvas_surface_ids, ())
        self.assertEqual(
            self.workspace._atlas_surface_assignment_target_ids,
            (),
        )
        self.assertEqual(
            self.workspace.viewer.get_selected_canvas_surface_ids(),
            (),
        )

    def test_rebuilt_canvas_targets_clear_stale_atlas_green_outline(
        self,
    ) -> None:
        _add_square_room_to_level(self.workspace.current_level)
        surfaces = tuple(build_fixed_surfaces(self.workspace.levels))
        wall = next(
            surface
            for surface in surfaces
            if surface.surface_type == SURFACE_TYPE_WALL
        )
        self.workspace.surface_texture_generation.set_levels(
            self.workspace.levels
        )
        self.workspace._set_canvas_viewer_targets(surfaces)
        assignment = _wall_texture_assignment(
            self.settings.path.parent / "surface_textures",
            assignment_id="stale-highlight-texture",
            surface_ids=(wall.surface_id,),
        )
        self.workspace.surface_texture_generation.set_data(
            SurfaceTextureData(assignments=[assignment])
        )
        source_id = build_atlas_wall_texture_source_id(
            assignment.assignment_id
        )
        self.workspace._handle_atlas_surface_texture_selected(source_id)
        self.assertEqual(
            self.workspace.texture_atlas_workspace._green_outline_source_ids,
            frozenset({source_id}),
        )

        self.workspace._set_canvas_viewer_targets(
            tuple(
                surface
                for surface in surfaces
                if surface.surface_id != wall.surface_id
            )
        )

        self.assertEqual(
            self.workspace.viewer.get_highlighted_canvas_surface_ids(),
            (),
        )
        self.assertEqual(
            self.workspace.texture_atlas_workspace._green_outline_source_ids,
            frozenset(),
        )

        self.workspace._set_canvas_viewer_targets(surfaces)

        self.assertEqual(
            self.workspace.viewer.get_highlighted_canvas_surface_ids(),
            (wall.surface_id,),
        )
        self.assertEqual(
            self.workspace.texture_atlas_workspace._green_outline_source_ids,
            frozenset({source_id}),
        )

    def test_atlas_assign_ignores_stale_surface_generation_selection(
        self,
    ) -> None:
        wall_id = _add_square_room_to_level(self.workspace.current_level)
        surfaces = tuple(build_fixed_surfaces(self.workspace.levels))
        surface_workspace = self.workspace.surface_texture_generation
        surface_workspace.set_levels(self.workspace.levels)
        self.workspace._set_canvas_viewer_targets(surfaces)
        assignment = _wall_texture_assignment(
            self.settings.path.parent / "surface_textures",
            assignment_id="unused-with-stale-selection",
            surface_ids=(),
        )
        surface_data = SurfaceTextureData(assignments=[assignment])
        surface_workspace.set_data(surface_data)
        surface_workspace.data_changed.emit(surface_data)
        stale_surface = next(
            surface for surface in surfaces if surface.surface_id == wall_id
        )
        surface_workspace.set_scene_surface_selection((stale_surface,))
        source_id = build_atlas_wall_texture_source_id(
            assignment.assignment_id
        )
        self.workspace._handle_atlas_surface_texture_selected(source_id)

        self.workspace._handle_atlas_surface_assign_requested(source_id)

        retained = surface_workspace.get_assignment(assignment.assignment_id)
        assert retained is not None
        self.assertEqual(retained.surface_ids, ())
        self.assertIn(
            "Canvas surface",
            self.workspace.texture_atlas_workspace.status_label.text(),
        )

    def test_failed_unused_surface_atlas_cleanup_is_retried(self) -> None:
        assignment = _wall_texture_assignment(
            self.settings.path.parent / "surface_textures",
            assignment_id="cleanup-retry-texture",
        )
        self.workspace.surface_texture_generation.set_data(
            SurfaceTextureData(assignments=[assignment])
        )
        atlas_data = TextureAtlasData()
        atlas = atlas_data.create_atlas("Cleanup", 2048, atlas_id="cleanup")
        self.workspace.texture_atlas_workspace.set_data(atlas_data)
        self.workspace._atlas_generation_signature = None
        self.workspace._sync_atlas_object_texture_sources(
            automatically_assign_scene_textures=False
        )
        source_id = build_atlas_wall_texture_source_id(
            assignment.assignment_id
        )
        self.assertTrue(
            self.workspace.texture_atlas_workspace
            .assign_source_to_selected_atlas(source_id)
        )
        packed = self.workspace.texture_atlas_workspace.get_data().atlas_by_id(
            atlas.atlas_id
        )
        assert packed is not None
        self.assertIsNotNone(packed.placement_for_object(source_id))
        self.workspace.surface_texture_generation.set_data(
            SurfaceTextureData(
                assignments=[replace(assignment, surface_ids=())]
            )
        )
        self.workspace._atlas_generation_signature = None

        with patch.object(
            self.workspace.texture_atlas_workspace,
            "remove_scene_texture_from_atlases",
            return_value=0,
        ) as remove_texture:
            self.workspace._sync_atlas_object_texture_sources(
                automatically_assign_scene_textures=False
            )
            self.workspace._sync_atlas_object_texture_sources(
                automatically_assign_scene_textures=False
            )

        self.assertEqual(remove_texture.call_count, 2)
        self.assertIsNone(self.workspace._atlas_generation_signature)

    def test_deleted_wall_reconciles_counter_and_removes_atlas_texture(
        self,
    ) -> None:
        wall_id = _add_square_room_to_level(self.workspace.current_level)
        self.workspace.surface_texture_generation.set_levels(
            self.workspace.levels
        )
        assignment = _wall_texture_assignment(
            self.settings.path.parent / "surface_textures",
            assignment_id="deleted-wall-texture",
            surface_ids=(wall_id,),
        )
        surface_workspace = self.workspace.surface_texture_generation
        surface_workspace.set_data(
            SurfaceTextureData(assignments=[assignment])
        )
        surface_workspace.data_changed.emit(surface_workspace.get_data())
        atlas_data = TextureAtlasData()
        atlas = atlas_data.create_atlas(
            "Deleted wall",
            2048,
            atlas_id="deleted-wall",
        )
        atlas_workspace = self.workspace.texture_atlas_workspace
        atlas_workspace.set_data(atlas_data)
        self.workspace._atlas_generation_signature = None
        self.workspace._sync_atlas_object_texture_sources(
            automatically_assign_scene_textures=False
        )
        source_id = build_atlas_wall_texture_source_id(
            assignment.assignment_id
        )
        self.assertTrue(
            atlas_workspace.assign_source_to_selected_atlas(source_id)
        )

        self.workspace.current_level.rooms.clear()
        self.workspace._handle_canvas_surface_geometry_changed()

        retained = surface_workspace.get_assignment(assignment.assignment_id)
        assert retained is not None
        self.assertEqual(retained.surface_ids, ())
        updated_atlas = atlas_workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert updated_atlas is not None
        self.assertIsNone(updated_atlas.placement_for_object(source_id))
        surface_labels = tuple(
            atlas_workspace.surface_list.item(row).text()
            for row in range(atlas_workspace.surface_list.count())
        )
        self.assertTrue(any("0 surfaces" in label for label in surface_labels))

    def test_plain_wall_vertex_deletion_reconciles_atlas_immediately(
        self,
    ) -> None:
        vertex_data = VertexData()
        start = vertex_data.add_vertex(0.0, 0.0)
        end = vertex_data.add_vertex(100.0, 0.0)
        vertex_data.add_edge(start.id, end.id)
        self.workspace.current_level.vertex_data = vertex_data
        self.workspace.current_level.rooms = []
        self.workspace._sync_canvas_to_current_level()
        wall_id = next(
            surface.surface_id
            for surface in build_fixed_surfaces(self.workspace.levels)
            if surface.surface_type == SURFACE_TYPE_WALL
        )
        surface_workspace = self.workspace.surface_texture_generation
        surface_workspace.set_levels(self.workspace.levels)
        assignment = _wall_texture_assignment(
            self.settings.path.parent / "surface_textures",
            assignment_id="plain-wall-delete",
            surface_ids=(wall_id,),
        )
        surface_workspace.set_data(
            SurfaceTextureData(assignments=[assignment])
        )
        surface_workspace.data_changed.emit(surface_workspace.get_data())
        atlas_data = TextureAtlasData()
        atlas = atlas_data.create_atlas(
            "Plain wall",
            2048,
            atlas_id="plain-wall",
        )
        atlas_workspace = self.workspace.texture_atlas_workspace
        atlas_workspace.set_data(atlas_data)
        self.workspace._atlas_generation_signature = None
        self.workspace._sync_atlas_object_texture_sources(
            automatically_assign_scene_textures=False
        )
        source_id = build_atlas_wall_texture_source_id(
            assignment.assignment_id
        )
        self.assertTrue(
            atlas_workspace.assign_source_to_selected_atlas(source_id)
        )

        self.workspace.canvas.selected_vertex_id = start.id
        self.workspace.canvas._delete_selected_vertices()

        retained = surface_workspace.get_assignment(assignment.assignment_id)
        assert retained is not None
        self.assertEqual(retained.surface_ids, ())
        updated_atlas = atlas_workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert updated_atlas is not None
        self.assertIsNone(updated_atlas.placement_for_object(source_id))

        self.workspace._handle_canvas_undo_requested()

        restored = surface_workspace.get_assignment(assignment.assignment_id)
        assert restored is not None
        self.assertEqual(restored.surface_ids, (wall_id,))
        restored_atlas = atlas_workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert restored_atlas is not None
        self.assertIsNotNone(restored_atlas.placement_for_object(source_id))

    def test_project_load_reconciles_stale_surface_assignments(self) -> None:
        assignment = _wall_texture_assignment(
            self.settings.path.parent / "surface_textures",
            assignment_id="stale-loaded-surface",
            surface_ids=("level:2/room:99/wall:1:2",),
        )
        source_id = build_atlas_wall_texture_source_id(
            assignment.assignment_id
        )
        atlas_data = TextureAtlasData()
        atlas = atlas_data.create_atlas(
            "Loaded stale",
            2048,
            atlas_id="loaded-stale",
        )
        atlas_data.assign_object(
            atlas.atlas_id,
            source_id,
            f"surface_textures/{assignment.asset_path}",
            512,
        )

        self.workspace._apply_project_state(
            levels=create_default_levels(),
            current_level_index=0,
            surface_texture_generation=SurfaceTextureData(
                assignments=[assignment]
            ),
            texture_atlases=atlas_data,
        )

        retained = self.workspace.surface_texture_generation.get_assignment(
            assignment.assignment_id
        )
        assert retained is not None
        self.assertEqual(retained.surface_ids, ())
        updated_atlas = (
            self.workspace.texture_atlas_workspace.get_data().atlas_by_id(
                atlas.atlas_id
            )
        )
        assert updated_atlas is not None
        self.assertIsNone(updated_atlas.placement_for_object(source_id))

    def test_full_atlas_surface_assign_requires_confirmation_before_new_atlas(
        self,
    ) -> None:
        wall_id = _add_square_room_to_level(self.workspace.current_level)
        surfaces = tuple(build_fixed_surfaces(self.workspace.levels))
        self.workspace.surface_texture_generation.set_levels(
            self.workspace.levels
        )
        self.workspace._set_canvas_viewer_targets(surfaces)
        assignment = _wall_texture_assignment(
            self.settings.path.parent / "surface_textures",
            assignment_id="unused-wall-texture",
            surface_ids=(),
        )
        surface_data = SurfaceTextureData(assignments=[assignment])
        self.workspace.surface_texture_generation.set_data(surface_data)

        atlas_data = TextureAtlasData()
        full_atlas = atlas_data.create_atlas(
            "Full",
            2048,
            atlas_id="full",
        )
        atlas_data.assign_object(
            full_atlas.atlas_id,
            "occupier",
            "occupier.png",
            2048,
        )
        self.workspace.texture_atlas_workspace.set_data(atlas_data)
        self.workspace.surface_texture_generation.data_changed.emit(surface_data)
        source_id = build_atlas_wall_texture_source_id(assignment.assignment_id)
        self.workspace.viewer.select_wall_target(wall_id)
        self.workspace.texture_atlas_workspace.surface_list.object_clicked.emit(
            source_id,
            Qt.MouseButton.LeftButton,
        )
        before_surface_data = (
            self.workspace.surface_texture_generation.get_data().to_dict()
        )
        before_atlas_data = (
            self.workspace.texture_atlas_workspace.get_data().to_dict()
        )

        with patch(
            "housemaker.main.QMessageBox.question",
            return_value=QMessageBox.StandardButton.No,
        ) as question:
            self.workspace.texture_atlas_workspace.place_assign_button.click()

        question.assert_called_once()
        self.assertEqual(
            question.call_args.args[2],
            "not space in atlas for the texture, create a new atlas?",
        )
        self.assertEqual(
            self.workspace.surface_texture_generation.get_data().to_dict(),
            before_surface_data,
        )
        self.assertEqual(
            self.workspace.texture_atlas_workspace.get_data().to_dict(),
            before_atlas_data,
        )

        with patch(
            "housemaker.main.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Yes,
        ):
            self.workspace.texture_atlas_workspace.place_assign_button.click()

        updated_assignment = (
            self.workspace.surface_texture_generation.get_assignment(
                assignment.assignment_id
            )
        )
        assert updated_assignment is not None
        self.assertEqual(updated_assignment.surface_ids, (wall_id,))
        updated_atlas_data = self.workspace.texture_atlas_workspace.get_data()
        self.assertEqual(len(updated_atlas_data.atlases), 2)
        selected_atlas = updated_atlas_data.atlas_by_id(
            updated_atlas_data.selected_atlas_id or ""
        )
        assert selected_atlas is not None
        self.assertEqual(selected_atlas.name, "Atlas")
        self.assertIsNotNone(selected_atlas.placement_for_object(source_id))

    def test_placed_object_is_automatically_added_at_settings_resolution(
        self,
    ) -> None:
        resolution_combo = (
            self.workspace.settings_widget
            .automatic_atlas_texture_resolution_combo
        )
        resolution_combo.setCurrentIndex(resolution_combo.findData(1024))
        atlas_data = TextureAtlasData()
        atlas = atlas_data.create_atlas(
            "Placed objects",
            2048,
            atlas_id="placed-objects",
        )
        self.workspace.texture_atlas_workspace.set_data(atlas_data)
        record = _generated_object_record_with_variants(
            self.settings.path.parent / "generated",
            object_id="placed-chair",
            object_name="Placed chair",
            resolutions=(512, 1024),
            selected_resolution=512,
            placement=GeneratedObjectPlacement(
                level_index=self.workspace.current_level.index,
                image_x=50.0,
                image_y=60.0,
            ),
        )
        generation_data = GenerationData(generated_objects=[record])

        self.workspace.generation.set_data(generation_data)
        self.workspace.generation.data_changed.emit(generation_data)

        packed = self.workspace.texture_atlas_workspace.get_data().atlas_by_id(
            atlas.atlas_id
        )
        assert packed is not None
        placement = packed.placement_for_object(record.object_id)
        self.assertIsNotNone(placement)
        assert placement is not None
        self.assertEqual(placement.texture_resolution, 1024)
        active_variant = self.workspace.generation.get_active_texture_variant(
            record.object_id
        )
        self.assertIsNotNone(active_variant)
        assert active_variant is not None
        self.assertEqual(active_variant.resolution, 1024)
        self.assertEqual(
            self.workspace.texture_atlas_workspace
            .get_unpacked_scene_texture_source_ids(),
            (),
        )

    def test_placed_object_creates_overflow_atlas_when_enabled(self) -> None:
        asset_directory = self.settings.path.parent / "generated"
        filler_records = tuple(
            _generated_object_record_with_variants(
                asset_directory,
                object_id=f"filler-{index}",
                object_name=f"Filler {index}",
                resolutions=(1024,),
                selected_resolution=1024,
            )
            for index in range(4)
        )
        pending_record = _generated_object_record_with_variants(
            asset_directory,
            object_id="overflow-chair",
            object_name="Overflow chair",
            resolutions=(512,),
            selected_resolution=512,
        )
        initial_generation_data = GenerationData(
            generated_objects=[*filler_records, pending_record]
        )
        self.workspace.generation.set_data(initial_generation_data)
        self.workspace.generation.data_changed.emit(initial_generation_data)
        atlas_workspace = self.workspace.texture_atlas_workspace
        atlas_data = TextureAtlasData()
        selected_atlas = atlas_data.create_atlas(
            "Full Atlas",
            2048,
            atlas_id="full-atlas",
        )
        for record in filler_records:
            source = atlas_workspace._sources_by_object_id[record.object_id]
            atlas_data.assign_object(
                selected_atlas.atlas_id,
                source.object_id,
                source.texture_path,
                source.texture_resolution,
            )
        atlas_workspace.set_data(atlas_data)
        self.workspace.settings_widget \
            .automatic_atlas_creation_checkbox.setChecked(True)
        placed_record = replace(
            pending_record,
            placement=GeneratedObjectPlacement(
                level_index=self.workspace.current_level.index,
                image_x=50.0,
                image_y=60.0,
            ),
        )
        placed_data = GenerationData(
            generated_objects=[*filler_records, placed_record]
        )

        with patch.object(atlas_workspace, "_materialize_atlas"):
            self.workspace.generation.set_data(placed_data)
            self.workspace.generation.data_changed.emit(placed_data)

        packed_data = atlas_workspace.get_data()
        self.assertEqual(len(packed_data.atlases), 2)
        packed_selected = packed_data.atlas_by_id(selected_atlas.atlas_id)
        assert packed_selected is not None
        self.assertIsNone(
            packed_selected.placement_for_object(placed_record.object_id)
        )
        overflow_atlas = next(
            atlas
            for atlas in packed_data.atlases
            if atlas.atlas_id != selected_atlas.atlas_id
        )
        self.assertEqual(overflow_atlas.resolution, selected_atlas.resolution)
        self.assertIsNotNone(
            overflow_atlas.placement_for_object(placed_record.object_id)
        )

    def test_main_passes_half_mesh_prefix_setting_to_auto_assignment(
        self,
    ) -> None:
        checkbox = (
            self.workspace.settings_widget
            .use_half_mesh_texture_prefix_checkbox
        )
        checkbox.setChecked(True)
        self.workspace._last_automatic_atlas_assignment_key = None

        with patch.object(
            self.workspace.texture_atlas_workspace,
            "auto_assign_scene_texture_sources",
            return_value=(),
        ) as auto_assign:
            self.workspace._automatically_assign_scene_textures()

        auto_assign.assert_called_once()
        self.assertTrue(
            auto_assign.call_args.kwargs["use_half_mesh_texture_prefix"]
        )

    def test_main_passes_automatic_atlas_creation_setting_and_retries(
        self,
    ) -> None:
        checkbox = self.workspace.settings_widget.automatic_atlas_creation_checkbox
        self.workspace._last_automatic_atlas_assignment_key = None

        with patch.object(
            self.workspace.texture_atlas_workspace,
            "auto_assign_scene_texture_sources",
            return_value=(),
        ) as auto_assign:
            self.workspace._automatically_assign_scene_textures()
            self.workspace._automatically_assign_scene_textures()
            self.assertEqual(auto_assign.call_count, 1)

            checkbox.setChecked(True)

        self.assertEqual(auto_assign.call_count, 2)
        self.assertFalse(
            auto_assign.call_args_list[0].kwargs["allow_atlas_creation"]
        )
        self.assertTrue(
            auto_assign.call_args_list[1].kwargs["allow_atlas_creation"]
        )

    def test_half_mesh_prefix_setting_change_retries_cached_assignment(
        self,
    ) -> None:
        self.workspace._last_automatic_atlas_assignment_key = None
        checkbox = (
            self.workspace.settings_widget
            .use_half_mesh_texture_prefix_checkbox
        )

        with patch.object(
            self.workspace.texture_atlas_workspace,
            "auto_assign_scene_texture_sources",
            return_value=(),
        ) as auto_assign:
            self.workspace._automatically_assign_scene_textures()
            self.workspace._automatically_assign_scene_textures()
            self.assertEqual(auto_assign.call_count, 1)

            checkbox.setChecked(True)

        self.assertEqual(auto_assign.call_count, 2)
        self.assertFalse(
            auto_assign.call_args_list[0].kwargs[
                "use_half_mesh_texture_prefix"
            ]
        )
        self.assertTrue(
            auto_assign.call_args_list[1].kwargs[
                "use_half_mesh_texture_prefix"
            ]
        )

    def test_placed_half_mesh_uses_prefixed_atlas_and_full_mesh_does_not(
        self,
    ) -> None:
        settings_widget = self.workspace.settings_widget
        settings_widget.automatic_atlas_texture_sort_by_pbr_checkbox.setChecked(
            True
        )
        settings_widget.use_half_mesh_texture_prefix_checkbox.setChecked(True)
        atlas_data = TextureAtlasData()
        selected_atlas = atlas_data.create_atlas(
            "Placed objects",
            2048,
            atlas_id="placed-objects",
        )
        self.workspace.texture_atlas_workspace.set_data(atlas_data)
        asset_directory = self.settings.path.parent / "generated"
        half_record = _generated_object_record_with_variants(
            asset_directory,
            object_id="half-window",
            object_name="Half window",
            resolutions=(512,),
            selected_resolution=512,
            placement=GeneratedObjectPlacement(
                level_index=self.workspace.current_level.index,
                image_x=25.0,
                image_y=30.0,
            ),
        )
        half_record = replace(
            half_record,
            pipeline={
                **half_record.pipeline,
                "symmetric_division": {
                    "version": 1,
                    "orientation": "vertical",
                    "kept_side": "left",
                    "plane_coordinate": 0.0,
                    "texture_content_half": "left",
                },
            },
        )
        full_record = _generated_object_record_with_variants(
            asset_directory,
            object_id="full-chair",
            object_name="Full chair",
            resolutions=(512,),
            selected_resolution=512,
            placement=GeneratedObjectPlacement(
                level_index=self.workspace.current_level.index,
                image_x=50.0,
                image_y=60.0,
            ),
        )
        generation_data = GenerationData(
            generated_objects=[half_record, full_record]
        )

        with patch.object(
            self.workspace.texture_atlas_workspace,
            "_materialize_atlas",
        ):
            self.workspace.generation.set_data(generation_data)
            self.workspace.generation.data_changed.emit(generation_data)

        packed_data = self.workspace.texture_atlas_workspace.get_data()
        self.assertEqual(packed_data.selected_atlas_id, selected_atlas.atlas_id)
        regular_atlas = packed_data.atlas_by_id(selected_atlas.atlas_id)
        assert regular_atlas is not None
        self.assertIsNotNone(
            regular_atlas.placement_for_object(full_record.object_id)
        )
        self.assertIsNone(
            regular_atlas.placement_for_object(half_record.object_id)
        )
        half_atlases = [
            atlas
            for atlas in packed_data.atlases
            if atlas.name.startswith("[HALF]")
        ]
        self.assertEqual(len(half_atlases), 1)
        self.assertEqual(half_atlases[0].resolution, 4096)
        self.assertIsNotNone(
            half_atlases[0].placement_for_object(half_record.object_id)
        )
        self.assertIsNone(
            half_atlases[0].placement_for_object(full_record.object_id)
        )

    def test_pbr_sort_routes_non_pbr_object_to_unique_auxiliary_atlas(
        self,
    ) -> None:
        settings_widget = self.workspace.settings_widget
        settings_widget.automatic_atlas_texture_sort_by_pbr_checkbox.setChecked(
            True
        )
        resolution_combo = (
            settings_widget.automatic_atlas_texture_resolution_combo
        )
        resolution_combo.setCurrentIndex(resolution_combo.findData(1024))
        asset_directory = self.settings.path.parent / "generated"
        pbr_record = _generated_object_record_with_variants(
            asset_directory,
            object_id="pbr-cabinet",
            object_name="PBR cabinet",
            resolutions=(1024,),
            selected_resolution=1024,
            pbr_map_types=(PBR_MAP_NORMAL,),
        )
        pbr_data = GenerationData(generated_objects=[pbr_record])
        self.workspace.generation.set_data(pbr_data)
        self.workspace.generation.data_changed.emit(pbr_data)
        atlas_workspace = self.workspace.texture_atlas_workspace
        pbr_source = atlas_workspace._sources_by_object_id[
            pbr_record.object_id
        ]
        self.assertTrue(pbr_source.has_texture_map(PBR_MAP_NORMAL))

        atlas_data = TextureAtlasData()
        selected_atlas = atlas_data.create_atlas(
            "Selected PBR",
            2048,
            atlas_id="selected-pbr",
        )
        reserved_name_atlas = atlas_data.create_atlas(
            "[NON-PBR] Atlas",
            2048,
            atlas_id="reserved-non-pbr-name",
        )
        for atlas in (selected_atlas, reserved_name_atlas):
            atlas_data.assign_object(
                atlas.atlas_id,
                pbr_source.object_id,
                pbr_source.texture_path,
                pbr_source.texture_resolution,
            )
        atlas_data.select_atlas(selected_atlas.atlas_id)
        atlas_workspace.set_data(atlas_data)

        non_pbr_record = _generated_object_record_with_variants(
            asset_directory,
            object_id="plain-chair",
            object_name="Plain chair",
            resolutions=(512, 1024),
            selected_resolution=512,
            placement=GeneratedObjectPlacement(
                level_index=self.workspace.current_level.index,
                image_x=40.0,
                image_y=55.0,
            ),
        )
        scene_data = GenerationData(
            generated_objects=[pbr_record, non_pbr_record]
        )

        self.workspace.generation.set_data(scene_data)
        self.workspace.generation.data_changed.emit(scene_data)

        packed_data = atlas_workspace.get_data()
        self.assertEqual(packed_data.selected_atlas_id, selected_atlas.atlas_id)
        packed_selected_atlas = packed_data.atlas_by_id(
            selected_atlas.atlas_id
        )
        packed_reserved_atlas = packed_data.atlas_by_id(
            reserved_name_atlas.atlas_id
        )
        assert packed_selected_atlas is not None
        assert packed_reserved_atlas is not None
        self.assertIsNone(
            packed_selected_atlas.placement_for_object(
                non_pbr_record.object_id
            )
        )
        self.assertIsNone(
            packed_reserved_atlas.placement_for_object(
                non_pbr_record.object_id
            )
        )
        auxiliary_atlases = [
            atlas
            for atlas in packed_data.atlases
            if atlas.atlas_id
            not in {selected_atlas.atlas_id, reserved_name_atlas.atlas_id}
        ]
        self.assertEqual(len(auxiliary_atlases), 1)
        auxiliary_atlas = auxiliary_atlases[0]
        self.assertTrue(auxiliary_atlas.name.startswith("[NON-PBR]"))
        self.assertEqual(
            len({atlas.name.casefold() for atlas in packed_data.atlases}),
            len(packed_data.atlases),
        )
        placement = auxiliary_atlas.placement_for_object(
            non_pbr_record.object_id
        )
        self.assertIsNotNone(placement)
        assert placement is not None
        self.assertEqual(placement.texture_resolution, 1024)

    def test_disabled_pbr_sort_keeps_non_pbr_object_in_selected_atlas(
        self,
    ) -> None:
        self.workspace.settings_widget \
            .automatic_atlas_texture_sort_by_pbr_checkbox.setChecked(False)
        asset_directory = self.settings.path.parent / "generated"
        pbr_record = _generated_object_record_with_variants(
            asset_directory,
            object_id="mapped-desk",
            object_name="Mapped desk",
            resolutions=(512,),
            selected_resolution=512,
            pbr_map_types=(PBR_MAP_ROUGHNESS,),
        )
        pbr_data = GenerationData(generated_objects=[pbr_record])
        self.workspace.generation.set_data(pbr_data)
        self.workspace.generation.data_changed.emit(pbr_data)
        atlas_workspace = self.workspace.texture_atlas_workspace
        pbr_source = atlas_workspace._sources_by_object_id[
            pbr_record.object_id
        ]
        atlas_data = TextureAtlasData()
        selected_atlas = atlas_data.create_atlas(
            "Mixed Atlas",
            2048,
            atlas_id="mixed-atlas",
        )
        atlas_data.assign_object(
            selected_atlas.atlas_id,
            pbr_source.object_id,
            pbr_source.texture_path,
            pbr_source.texture_resolution,
        )
        atlas_workspace.set_data(atlas_data)
        non_pbr_record = _generated_object_record_with_variants(
            asset_directory,
            object_id="plain-stool",
            object_name="Plain stool",
            resolutions=(512,),
            selected_resolution=512,
            placement=GeneratedObjectPlacement(
                level_index=self.workspace.current_level.index,
                image_x=35.0,
                image_y=50.0,
            ),
        )
        scene_data = GenerationData(
            generated_objects=[pbr_record, non_pbr_record]
        )

        self.workspace.generation.set_data(scene_data)
        self.workspace.generation.data_changed.emit(scene_data)

        packed_data = atlas_workspace.get_data()
        self.assertEqual(len(packed_data.atlases), 1)
        packed_atlas = packed_data.atlas_by_id(selected_atlas.atlas_id)
        assert packed_atlas is not None
        self.assertIsNotNone(
            packed_atlas.placement_for_object(non_pbr_record.object_id)
        )

    def test_canvas_removal_unassigns_texture_but_retains_generated_source(
        self,
    ) -> None:
        asset_directory = self.settings.path.parent / "generated"
        record = _generated_object_record_with_variants(
            asset_directory,
            object_id="placed-sideboard",
            object_name="Placed sideboard",
            resolutions=(512,),
            selected_resolution=512,
            placement=GeneratedObjectPlacement(
                level_index=self.workspace.current_level.index,
                image_x=30.0,
                image_y=45.0,
            ),
        )
        generation_data = GenerationData(generated_objects=[record])
        self.workspace.generation.set_data(generation_data)
        self.workspace.generation.data_changed.emit(generation_data)
        atlas_workspace = self.workspace.texture_atlas_workspace
        source = atlas_workspace._sources_by_object_id[record.object_id]
        atlas_data = TextureAtlasData()
        first_atlas = atlas_data.create_atlas(
            "First placement",
            2048,
            atlas_id="first-placement",
        )
        second_atlas = atlas_data.create_atlas(
            "Second placement",
            2048,
            atlas_id="second-placement",
        )
        for atlas in (first_atlas, second_atlas):
            atlas_data.assign_object(
                atlas.atlas_id,
                source.object_id,
                source.texture_path,
                source.texture_resolution,
            )
        atlas_workspace.set_data(atlas_data)

        self.workspace.viewer.placed_object_removal_requested.emit(
            record.object_id
        )

        retained_data = self.workspace.generation.get_data()
        self.assertEqual(
            [item.object_id for item in retained_data.generated_objects],
            [record.object_id],
        )
        self.assertIsNone(retained_data.generated_objects[0].placement)
        self.assertIn(record.object_id, atlas_workspace._sources_by_object_id)
        self.assertTrue(source.physical_texture_path.is_file())
        for atlas in atlas_workspace.get_data().atlases:
            with self.subTest(atlas=atlas.name):
                self.assertIsNone(
                    atlas.placement_for_object(record.object_id)
                )
        self.assertEqual(
            atlas_workspace.get_unpacked_scene_texture_source_ids(),
            (),
        )

    def test_included_textured_surface_is_auto_added_at_settings_resolution(
        self,
    ) -> None:
        resolution_combo = (
            self.workspace.settings_widget
            .automatic_atlas_texture_resolution_combo
        )
        resolution_combo.setCurrentIndex(resolution_combo.findData(1024))
        wall_surface_id = _add_square_room_to_level(
            self.workspace.current_level
        )
        self.workspace.surface_texture_generation.set_levels(
            self.workspace.levels
        )
        atlas_data = TextureAtlasData()
        atlas = atlas_data.create_atlas(
            "Included surfaces",
            2048,
            atlas_id="included-surfaces",
        )
        self.workspace.texture_atlas_workspace.set_data(atlas_data)
        assignment = _wall_texture_assignment_with_variants(
            self.settings.path.parent / "surface_textures",
            assignment_id="included-wall",
            surface_ids=(wall_surface_id,),
            selected_resolution=512,
        )
        surface_data = SurfaceTextureData(assignments=[assignment])

        self.workspace.surface_texture_generation.set_data(surface_data)
        self.workspace.surface_texture_generation.data_changed.emit(surface_data)

        source_id = build_atlas_wall_texture_source_id(
            assignment.assignment_id
        )
        packed = self.workspace.texture_atlas_workspace.get_data().atlas_by_id(
            atlas.atlas_id
        )
        assert packed is not None
        placement = packed.placement_for_object(source_id)
        self.assertIsNotNone(placement)
        assert placement is not None
        self.assertEqual(placement.texture_resolution, 1024)
        selected_assignment = (
            self.workspace.surface_texture_generation.get_assignment(
                assignment.assignment_id
            )
        )
        self.assertIsNotNone(selected_assignment)
        assert selected_assignment is not None
        self.assertEqual(selected_assignment.selected_texture_resolution, 1024)
        self.assertEqual(
            self.workspace.texture_atlas_workspace
            .get_unpacked_scene_texture_source_ids(),
            (),
        )

    def test_selected_surface_texture_creates_overflow_atlas_when_enabled(
        self,
    ) -> None:
        asset_directory = self.settings.path.parent / "generated"
        filler_records = tuple(
            _generated_object_record_with_variants(
                asset_directory,
                object_id=f"surface-filler-{index}",
                object_name=f"Surface filler {index}",
                resolutions=(1024,),
                selected_resolution=1024,
            )
            for index in range(4)
        )
        generation_data = GenerationData(generated_objects=list(filler_records))
        self.workspace.generation.set_data(generation_data)
        self.workspace.generation.data_changed.emit(generation_data)
        wall_surface_id = _add_square_room_to_level(self.workspace.current_level)
        self.workspace.surface_texture_generation.set_levels(self.workspace.levels)
        assignment = _wall_texture_assignment_with_variants(
            self.settings.path.parent / "surface_textures",
            assignment_id="overflow-wall",
            surface_ids=(),
            selected_resolution=512,
        )
        empty_surface_data = SurfaceTextureData(assignments=[assignment])
        self.workspace.surface_texture_generation.set_data(empty_surface_data)
        self.workspace.surface_texture_generation.data_changed.emit(
            empty_surface_data
        )
        atlas_workspace = self.workspace.texture_atlas_workspace
        atlas_data = TextureAtlasData()
        selected_atlas = atlas_data.create_atlas(
            "Full Surface Atlas",
            2048,
            atlas_id="full-surface-atlas",
        )
        for record in filler_records:
            source = atlas_workspace._sources_by_object_id[record.object_id]
            atlas_data.assign_object(
                selected_atlas.atlas_id,
                source.object_id,
                source.texture_path,
                source.texture_resolution,
            )
        atlas_workspace.set_data(atlas_data)
        self.workspace.settings_widget \
            .automatic_atlas_creation_checkbox.setChecked(True)
        assigned_surface_data = SurfaceTextureData(
            assignments=[replace(assignment, surface_ids=(wall_surface_id,))]
        )

        with patch.object(atlas_workspace, "_materialize_atlas"):
            self.workspace.surface_texture_generation.set_data(
                assigned_surface_data
            )
            self.workspace.surface_texture_generation.data_changed.emit(
                assigned_surface_data
            )

        source_id = build_atlas_wall_texture_source_id(assignment.assignment_id)
        packed_data = atlas_workspace.get_data()
        self.assertEqual(len(packed_data.atlases), 2)
        packed_selected = packed_data.atlas_by_id(selected_atlas.atlas_id)
        assert packed_selected is not None
        self.assertIsNone(packed_selected.placement_for_object(source_id))
        overflow_atlas = next(
            atlas
            for atlas in packed_data.atlases
            if atlas.atlas_id != selected_atlas.atlas_id
        )
        self.assertIsNotNone(overflow_atlas.placement_for_object(source_id))

    def test_missing_active_surface_variant_can_switch_to_valid_resolution(
        self,
    ) -> None:
        assignment = _wall_texture_assignment_with_variants(
            self.settings.path.parent / "surface_textures",
            assignment_id="recover-missing-variant",
            selected_resolution=512,
        )
        surface_workspace = self.workspace.surface_texture_generation
        surface_workspace.set_data(
            SurfaceTextureData(assignments=[assignment])
        )
        surface_workspace.data_changed.emit(surface_workspace.get_data())
        source_id = build_atlas_wall_texture_source_id(
            assignment.assignment_id
        )
        self.assertIn(source_id, self.workspace._atlas_wall_texture_source_ids)
        active_path = surface_workspace.get_assignment_asset_path(
            assignment.assignment_id,
            512,
        )
        assert active_path is not None
        active_path.unlink()
        self.workspace._atlas_generation_signature = None

        self.workspace._sync_atlas_object_texture_sources(
            automatically_assign_scene_textures=False
        )

        self.assertNotIn(
            source_id,
            self.workspace._atlas_wall_texture_source_ids,
        )
        self.assertTrue(
            surface_workspace.select_assignment_texture_resolution(
                assignment.assignment_id,
                1024,
            )
        )
        updated = surface_workspace.get_assignment(assignment.assignment_id)
        assert updated is not None
        self.assertEqual(updated.selected_texture_resolution, 1024)

    def test_required_texture_stays_unpacked_and_is_highlighted_green(
        self,
    ) -> None:
        atlas_data = TextureAtlasData()
        atlas = atlas_data.create_atlas(
            "Full atlas",
            2048,
            atlas_id="full-atlas",
        )
        atlas_data.assign_object(
            atlas.atlas_id,
            "occupier",
            "occupier.png",
            2048,
        )
        self.workspace.texture_atlas_workspace.set_data(atlas_data)
        record = _generated_object_record_with_variants(
            self.settings.path.parent / "generated",
            object_id="unpacked-table",
            object_name="Unpacked table",
            resolutions=(512,),
            selected_resolution=512,
            placement=GeneratedObjectPlacement(
                level_index=self.workspace.current_level.index,
                image_x=25.0,
                image_y=35.0,
            ),
        )
        generation_data = GenerationData(generated_objects=[record])

        self.workspace.generation.set_data(generation_data)
        self.workspace.generation.data_changed.emit(generation_data)

        self.assertEqual(
            self.workspace.texture_atlas_workspace
            .get_unpacked_scene_texture_source_ids(),
            (record.object_id,),
        )
        packed = self.workspace.texture_atlas_workspace.get_data().atlas_by_id(
            atlas.atlas_id
        )
        assert packed is not None
        self.assertIsNone(packed.placement_for_object(record.object_id))
        source_item = next(
            self.workspace.texture_atlas_workspace.object_list.item(index)
            for index in range(
                self.workspace.texture_atlas_workspace.object_list.count()
            )
            if self.workspace.texture_atlas_workspace.object_list.item(
                index
            ).data(Qt.ItemDataRole.UserRole)
            == record.object_id
        )
        self.assertEqual(
            source_item.foreground().color(),
            SCENE_BOUND_SOURCE_COLOR,
        )

    def test_export_blocks_before_file_dialog_for_unpacked_scene_texture(
        self,
    ) -> None:
        record = _generated_object_record_with_variants(
            self.settings.path.parent / "generated",
            object_id="unatlased-sofa",
            object_name="Unatlased sofa",
            resolutions=(512,),
            selected_resolution=512,
            placement=GeneratedObjectPlacement(
                level_index=self.workspace.current_level.index,
                image_x=15.0,
                image_y=20.0,
            ),
        )
        self.workspace.generation.set_data(
            GenerationData(generated_objects=[record])
        )

        with (
            patch("housemaker.main.QMessageBox.warning") as warning,
            patch(
                "housemaker.main.QFileDialog.getSaveFileName"
            ) as file_dialog,
        ):
            self.workspace._handle_glb_export_clicked()

        file_dialog.assert_not_called()
        warning.assert_called_once()
        self.assertEqual(warning.call_args.args[1], "Export blocked")
        self.assertIn(record.object_name, warning.call_args.args[2])
        self.assertEqual(
            self.workspace.texture_atlas_workspace
            .get_unpacked_scene_texture_source_ids(),
            (record.object_id,),
        )

    def test_missing_used_object_texture_still_blocks_export(self) -> None:
        record = _generated_object_record_with_variants(
            self.settings.path.parent / "generated",
            object_id="missing-textured-object",
            object_name="Missing textured object",
            resolutions=(512,),
            selected_resolution=512,
            placement=GeneratedObjectPlacement(
                level_index=self.workspace.current_level.index,
                image_x=15.0,
                image_y=20.0,
            ),
        )
        texture_path = (
            self.settings.path.parent
            / "generated"
            / "missing-textured-object.texture-512.png"
        )
        texture_path.unlink()
        generation_data = GenerationData(generated_objects=[record])

        self.workspace.generation.set_data(generation_data)
        self.workspace.generation.data_changed.emit(generation_data)

        self.assertEqual(
            self.workspace.texture_atlas_workspace
            .get_unpacked_scene_texture_source_ids(),
            (record.object_id,),
        )
        with patch("housemaker.main.QMessageBox.warning") as warning:
            self.assertTrue(
                self.workspace._show_unpacked_scene_texture_export_error()
            )
        warning.assert_called_once()
        self.assertIn(record.object_name, warning.call_args.args[2])

    def test_export_does_not_readd_a_manually_unpacked_scene_texture(
        self,
    ) -> None:
        atlas_data = TextureAtlasData()
        atlas = atlas_data.create_atlas(
            "Manual removal",
            2048,
            atlas_id="manual-removal",
        )
        self.workspace.texture_atlas_workspace.set_data(atlas_data)
        record = _generated_object_record_with_variants(
            self.settings.path.parent / "generated",
            object_id="removed-lamp",
            object_name="Removed lamp",
            resolutions=(512,),
            selected_resolution=512,
            placement=GeneratedObjectPlacement(
                level_index=self.workspace.current_level.index,
                image_x=15.0,
                image_y=20.0,
            ),
        )
        generation_data = GenerationData(generated_objects=[record])
        self.workspace.generation.set_data(generation_data)
        self.workspace.generation.data_changed.emit(generation_data)
        self.assertTrue(
            self.workspace.texture_atlas_workspace._select_object_row(
                record.object_id
            )
        )
        self.workspace.texture_atlas_workspace \
            .remove_selected_texture_from_atlas()

        with (
            patch("housemaker.main.QMessageBox.warning") as warning,
            patch(
                "housemaker.main.QFileDialog.getSaveFileName"
            ) as file_dialog,
        ):
            self.workspace._handle_glb_export_clicked()

        file_dialog.assert_not_called()
        warning.assert_called_once()
        retained_atlas = (
            self.workspace.texture_atlas_workspace.get_data().atlas_by_id(
                atlas.atlas_id
            )
        )
        assert retained_atlas is not None
        self.assertIsNone(
            retained_atlas.placement_for_object(record.object_id)
        )

    def test_non_wall_surface_texture_is_exposed_to_atlas(self) -> None:
        assignment = _wall_texture_assignment(
            self.settings.path.parent / "surface_textures",
            assignment_id="oak-floor",
            surface_ids=("level:2/room:1/floor",),
            surface_type=SURFACE_TYPE_FLOOR,
        )
        surface_data = SurfaceTextureData(assignments=[assignment])
        self.workspace.surface_texture_generation.set_data(surface_data)
        self.workspace.surface_texture_generation.data_changed.emit(surface_data)

        source_id = build_atlas_wall_texture_source_id(
            assignment.assignment_id
        )
        source_items = [
            self.workspace.texture_atlas_workspace.surface_list.item(index)
            for index in range(
                self.workspace.texture_atlas_workspace.surface_list.count()
            )
        ]
        source_item = next(
            item
            for item in source_items
            if item.data(Qt.ItemDataRole.UserRole) == source_id
        )

        self.assertTrue(source_item.text().startswith("[FLOOR] "))
        self.assertIn("Floor texture", source_item.text())
        source = (
            self.workspace.texture_atlas_workspace
            ._sources_by_object_id[source_id]
        )
        self.assertEqual(
            tuple(source.load_texture_rgba(PBR_MAP_ROUGHNESS)[0, 0]),
            (184, 184, 184, 255),
        )
        self.assertEqual(
            self.workspace._build_atlas_surface_source_ids(),
            {"level:2/room:1/floor": source_id},
        )

    def test_canvas_export_uses_one_atlas_material_for_surface_source(
        self,
    ) -> None:
        assignment = _wall_texture_assignment(
            self.settings.path.parent / "surface_textures",
        )
        surface_data = SurfaceTextureData(assignments=[assignment])
        self.workspace.surface_texture_generation.set_data(surface_data)
        self.workspace.surface_texture_generation.data_changed.emit(surface_data)
        atlas_data = TextureAtlasData()
        atlas_data.create_atlas("Architecture", 2048, atlas_id="architecture")
        self.workspace.texture_atlas_workspace.set_data(atlas_data)
        source_id = build_atlas_wall_texture_source_id(
            assignment.assignment_id
        )
        source_item = next(
            self.workspace.texture_atlas_workspace.surface_list.item(index)
            for index in range(
                self.workspace.texture_atlas_workspace.surface_list.count()
            )
            if self.workspace.texture_atlas_workspace.surface_list.item(
                index
            ).data(Qt.ItemDataRole.UserRole)
            == source_id
        )
        self.workspace.texture_atlas_workspace.surface_list.setCurrentItem(
            source_item
        )
        self.assertTrue(
            self.workspace.texture_atlas_workspace
            .assign_source_to_selected_atlas(source_id)
        )

        with patch(
            "housemaker.main.convert_to_glb",
            return_value=_generated_textured_surface_model(
                assignment.surface_ids[0]
            ),
        ):
            result = self.workspace._build_generated_model(None)

        self.assertIsNotNone(result)
        assert result is not None
        atlas_materials = [
            getattr(getattr(mesh.visual, "material", None), "name", "")
            for mesh in result.scene.geometry.values()
        ]
        self.assertEqual(atlas_materials, ["Architecture"])

    def test_canvas_export_omits_unassigned_materialless_surfaces(self) -> None:
        wall_surface_id = _add_square_room_to_level(
            self.workspace.current_level
        )
        self.workspace.surface_texture_generation.set_levels(
            self.workspace.levels
        )
        assignment = _wall_texture_assignment(
            self.settings.path.parent / "surface_textures",
            assignment_id="exported-wall",
            surface_ids=(wall_surface_id,),
        )
        surface_data = SurfaceTextureData(assignments=[assignment])
        self.workspace.surface_texture_generation.set_data(surface_data)
        self.workspace.surface_texture_generation.data_changed.emit(
            surface_data
        )
        atlas_data = TextureAtlasData()
        atlas = atlas_data.create_atlas(
            "Architecture",
            2048,
            atlas_id="architecture",
        )
        self.workspace.texture_atlas_workspace.set_data(atlas_data)
        source_id = build_atlas_wall_texture_source_id(
            assignment.assignment_id
        )
        source = self.workspace.texture_atlas_workspace \
            ._sources_by_object_id[source_id]
        atlas_data.assign_object(
            atlas.atlas_id,
            source.object_id,
            source.texture_path,
            source.texture_resolution,
        )
        self.workspace.texture_atlas_workspace.set_data(atlas_data)

        result = self.workspace._build_generated_model(None)

        self.assertIsNotNone(result)
        assert result is not None
        materialless_surface_names = [
            name
            for name, mesh in result.scene.geometry.items()
            if (
                getattr(getattr(mesh.visual, "material", None), "name", None)
                in {None, "DefaultMaterial"}
                and "housemaker_object_id"
                not in dict(getattr(mesh, "metadata", {}) or {})
            )
        ]
        self.assertEqual(materialless_surface_names, [])

    def test_wall_preview_uses_exact_pinned_texture_variant(self) -> None:
        assignment = _wall_texture_assignment_with_variants(
            self.settings.path.parent / "surface_textures",
            selected_resolution=512,
        )
        surface_data = SurfaceTextureData(assignments=[assignment])
        self.workspace.surface_texture_generation.set_data(surface_data)
        self.workspace.surface_texture_generation.data_changed.emit(surface_data)
        source_id = build_atlas_wall_texture_source_id(
            assignment.assignment_id
        )
        target_variant = assignment.texture_variant_for_resolution(1024)
        assert target_variant is not None
        atlas_data = TextureAtlasData()
        atlas = atlas_data.create_atlas("Pinned wall", 2048, atlas_id="walls")
        atlas_data.assign_object(
            atlas.atlas_id,
            source_id,
            f"surface_textures/{target_variant.asset_path}",
            1024,
        )
        atlas_workspace = self.workspace.texture_atlas_workspace
        atlas_workspace.set_data(atlas_data)
        self.assertTrue(atlas_workspace._select_object_row(source_id))

        with patch.object(
            self.workspace.generation,
            "get_texture_variant",
        ) as generated_variant:
            self.assertTrue(atlas_workspace.request_selected_object_preview())

        generated_variant.assert_not_called()
        preview_model = self.workspace.atlas_object_preview_viewer.model
        self.assertIsNotNone(preview_model)
        assert preview_model is not None
        texture = preview_model.mesh.visual.material.baseColorTexture
        texture_rgba = np.asarray(texture.convert("RGBA"), dtype=np.uint8)
        self.assertEqual(texture_rgba.shape, (1024, 1024, 4))
        np.testing.assert_array_equal(
            texture_rgba[512, 512],
            np.asarray((70, 170, 25, 255), dtype=np.uint8),
        )
        self.assertEqual(self.workspace._atlas_preview_variant_key[1], 1024)

    def test_wall_list_wheel_does_not_resize_and_preview_updates_globally(
        self,
    ) -> None:
        wall_id = _add_square_room_to_level(self.workspace.current_level)
        self.workspace.surface_texture_generation.set_levels(
            self.workspace.levels
        )
        assignment = _wall_texture_assignment_with_variants(
            self.settings.path.parent / "surface_textures",
            surface_ids=(wall_id,),
        )
        surface_data = SurfaceTextureData(assignments=[assignment])
        self.workspace.surface_texture_generation.set_data(surface_data)
        self.workspace.surface_texture_generation.data_changed.emit(surface_data)
        atlas_workspace = self.workspace.texture_atlas_workspace
        source_id = build_atlas_wall_texture_source_id(
            assignment.assignment_id
        )
        source = atlas_workspace._sources_by_object_id[source_id]
        self.assertTrue(source.supports_resolution_changes)
        self.assertEqual(source.texture_resolution, 512)
        atlas_data = TextureAtlasData()
        atlas = atlas_data.create_atlas(
            "Resizable walls",
            4096,
            atlas_id="resizable-walls",
        )
        atlas_data.assign_object(
            atlas.atlas_id,
            source_id,
            source.texture_path,
            source.texture_resolution,
        )
        atlas_workspace.set_data(atlas_data)
        self.workspace.workspace_tabs.setCurrentWidget(atlas_workspace)
        _qt_application.processEvents()
        source_item = next(
            atlas_workspace.surface_list.item(index)
            for index in range(atlas_workspace.surface_list.count())
            if atlas_workspace.surface_list.item(index).data(
                Qt.ItemDataRole.UserRole
            )
            == source_id
        )
        wheel = _wheel_event(
            QPointF(atlas_workspace.surface_list.visualItemRect(source_item).center()),
            120,
        )

        atlas_workspace.surface_list.wheelEvent(wheel)
        _qt_application.processEvents()

        unchanged_assignment = self.workspace.surface_texture_generation \
            .get_data().assignments[0]
        self.assertEqual(unchanged_assignment.selected_texture_resolution, 512)
        unchanged_atlas = atlas_workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert unchanged_atlas is not None
        unchanged_placement = unchanged_atlas.placement_for_object(source_id)
        assert unchanged_placement is not None
        self.assertEqual(unchanged_placement.texture_resolution, 512)

        preview = atlas_workspace.preview
        preview_side = min(preview.width() - 32.0, preview.height() - 32.0)
        preview_origin = QPointF(
            (preview.width() - preview_side) / 2.0,
            (preview.height() - preview_side) / 2.0,
        )
        placement_center = QPointF(
            preview_origin.x()
            + (unchanged_placement.x + unchanged_placement.size / 2.0)
            * preview_side
            / atlas.resolution,
            preview_origin.y()
            + (unchanged_placement.y + unchanged_placement.size / 2.0)
            * preview_side
            / atlas.resolution,
        )
        preview_wheel = _wheel_event(placement_center, 120)

        preview.wheelEvent(preview_wheel)
        _qt_application.processEvents()

        self.assertTrue(preview_wheel.isAccepted())
        updated_assignment = self.workspace.surface_texture_generation \
            .get_data().assignments[0]
        self.assertEqual(updated_assignment.assignment_id, assignment.assignment_id)
        self.assertEqual(updated_assignment.selected_texture_resolution, 1024)
        self.assertEqual(
            updated_assignment.asset_path,
            f"{assignment.assignment_id}.texture-1024.png",
        )
        updated_atlas = atlas_workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert updated_atlas is not None
        placement = updated_atlas.placement_for_object(source_id)
        assert placement is not None
        self.assertEqual(placement.texture_resolution, 1024)
        self.assertEqual(
            placement.texture_path,
            f"surface_textures/{assignment.assignment_id}.texture-1024.png",
        )

    def test_surface_resolution_selection_repacks_every_atlas_placement(
        self,
    ) -> None:
        wall_id = _add_square_room_to_level(self.workspace.current_level)
        self.workspace.surface_texture_generation.set_levels(
            self.workspace.levels
        )
        assignment = _wall_texture_assignment_with_variants(
            self.settings.path.parent / "surface_textures",
            surface_ids=(wall_id,),
        )
        surface_workspace = self.workspace.surface_texture_generation
        surface_data = SurfaceTextureData(assignments=[assignment])
        surface_workspace.set_data(surface_data)
        surface_workspace.data_changed.emit(surface_data)
        atlas_workspace = self.workspace.texture_atlas_workspace
        source_id = build_atlas_wall_texture_source_id(
            assignment.assignment_id
        )
        source = atlas_workspace._sources_by_object_id[source_id]
        atlas_data = TextureAtlasData()
        first = atlas_data.create_atlas("First", 2048, atlas_id="first")
        second = atlas_data.create_atlas("Second", 2048, atlas_id="second")
        for atlas in (first, second):
            atlas_data.assign_object(
                atlas.atlas_id,
                source_id,
                source.texture_path,
                source.texture_resolution,
            )
        atlas_workspace.set_data(atlas_data)
        resolution_signals: list[tuple[str, int]] = []
        atlas_data_changes: list[TextureAtlasData] = []
        atlas_workspace.object_texture_resolution_changed.connect(
            lambda object_id, resolution: resolution_signals.append(
                (object_id, resolution)
            )
        )
        atlas_workspace.data_changed.connect(atlas_data_changes.append)

        with patch.object(
            surface_workspace,
            "select_assignment_texture_resolution",
            wraps=surface_workspace.select_assignment_texture_resolution,
        ) as select_resolution:
            changed = atlas_workspace.set_object_texture_resolution(
                source_id,
                1024,
            )
        _qt_application.processEvents()

        self.assertTrue(changed)
        self.assertEqual(select_resolution.call_count, 1)
        self.assertEqual(resolution_signals, [(source_id, 1024)])
        self.assertEqual(len(atlas_data_changes), 1)
        selected_assignment = surface_workspace.get_data().assignments[0]
        self.assertEqual(selected_assignment.selected_texture_resolution, 1024)
        self.assertEqual(
            atlas_workspace._sources_by_object_id[
                source_id
            ].texture_resolution,
            1024,
        )
        updated_data = atlas_workspace.get_data()
        for atlas_id in (first.atlas_id, second.atlas_id):
            updated_atlas = updated_data.atlas_by_id(atlas_id)
            assert updated_atlas is not None
            placement = updated_atlas.placement_for_object(source_id)
            assert placement is not None
            self.assertEqual(placement.texture_resolution, 1024)
            self.assertEqual(
                placement.texture_path,
                (
                    "surface_textures/"
                    f"{assignment.assignment_id}.texture-1024.png"
                ),
            )
            self.assertTrue(
                (
                    self.settings.path.parent
                    / "texture_atlases"
                    / f"{atlas_id}.png"
                ).is_file()
            )

    def test_surface_resolution_selection_is_blocked_by_any_atlas_capacity(
        self,
    ) -> None:
        assignment = _wall_texture_assignment_with_variants(
            self.settings.path.parent / "surface_textures"
        )
        filler_assignment = _wall_texture_assignment_with_variants(
            self.settings.path.parent / "surface_textures",
            assignment_id="filler-wall-variants",
            surface_ids=("level:2/wall:2:3",),
        )
        surface_workspace = self.workspace.surface_texture_generation
        surface_data = SurfaceTextureData(
            assignments=[assignment, filler_assignment]
        )
        surface_workspace.set_data(surface_data)
        surface_workspace.data_changed.emit(surface_data)
        atlas_workspace = self.workspace.texture_atlas_workspace
        source_id = build_atlas_wall_texture_source_id(
            assignment.assignment_id
        )
        source = atlas_workspace._sources_by_object_id[source_id]
        filler_source_id = build_atlas_wall_texture_source_id(
            filler_assignment.assignment_id
        )
        filler_source = atlas_workspace._sources_by_object_id[filler_source_id]
        atlas_data = TextureAtlasData()
        atlas = atlas_data.create_atlas("Too small", 2048, atlas_id="small")
        atlas_data.assign_object(
            atlas.atlas_id,
            source_id,
            source.texture_path,
            source.texture_resolution,
        )
        atlas_data.assign_object(
            atlas.atlas_id,
            filler_source_id,
            filler_source.texture_path,
            filler_source.texture_resolution,
        )
        atlas_workspace.set_data(atlas_data)
        previous_atlas_data = atlas_workspace.get_data()

        changed = atlas_workspace.set_object_texture_resolution(
            source_id,
            2048,
        )
        _qt_application.processEvents()

        self.assertFalse(changed)
        selected_assignment = surface_workspace.get_data().assignments[0]
        self.assertEqual(selected_assignment.selected_texture_resolution, 512)
        self.assertEqual(atlas_workspace.get_data(), previous_atlas_data)
        self.assertIn("blocked", atlas_workspace.status_label.text().lower())

    def test_removed_wall_assignment_is_purged_from_every_atlas(self) -> None:
        assignment = _wall_texture_assignment(
            self.settings.path.parent / "surface_textures"
        )
        surface_data = SurfaceTextureData(assignments=[assignment])
        self.workspace.surface_texture_generation.set_data(surface_data)
        self.workspace.surface_texture_generation.data_changed.emit(surface_data)
        source_id = build_atlas_wall_texture_source_id(
            assignment.assignment_id
        )
        source = self.workspace.texture_atlas_workspace._sources_by_object_id[
            source_id
        ]
        atlas_data = TextureAtlasData()
        first = atlas_data.create_atlas("First", 2048, atlas_id="first")
        second = atlas_data.create_atlas("Second", 2048, atlas_id="second")
        for atlas in (first, second):
            atlas_data.assign_object(
                atlas.atlas_id,
                source_id,
                source.texture_path,
                source.texture_resolution,
            )
        self.workspace.texture_atlas_workspace.set_data(atlas_data)
        atlas_changes: list[object] = []
        self.workspace.texture_atlas_workspace.data_changed.connect(
            atlas_changes.append
        )

        self.workspace.surface_texture_generation.set_data(
            SurfaceTextureData()
        )
        self.workspace.surface_texture_generation.assignments_removed.emit(
            (assignment.assignment_id,)
        )

        updated = self.workspace.texture_atlas_workspace.get_data()
        self.assertIsNone(
            updated.atlas_by_id(first.atlas_id).placement_for_object(source_id)
        )
        self.assertIsNone(
            updated.atlas_by_id(second.atlas_id).placement_for_object(source_id)
        )
        self.assertEqual(len(atlas_changes), 1)

    def test_deleted_surface_with_missing_asset_is_purged_from_atlas(
        self,
    ) -> None:
        assignment = _wall_texture_assignment(
            self.settings.path.parent / "surface_textures",
            assignment_id="missing-deleted-surface",
        )
        surface_asset = (
            self.settings.path.parent
            / "surface_textures"
            / assignment.asset_path
        )
        surface_asset.unlink()
        surface_data = SurfaceTextureData(assignments=[assignment])
        self.workspace.surface_texture_generation.set_data(surface_data)
        self.workspace.surface_texture_generation.data_changed.emit(surface_data)
        source_id = build_atlas_wall_texture_source_id(
            assignment.assignment_id
        )
        self.assertNotIn(
            source_id,
            self.workspace._atlas_wall_texture_source_ids,
        )
        missing_items = tuple(
            self.workspace.texture_atlas_workspace.surface_list.item(row)
            for row in range(
                self.workspace.texture_atlas_workspace.surface_list.count()
            )
        )
        missing_item = next(
            item
            for item in missing_items
            if item.data(Qt.ItemDataRole.UserRole) == source_id
        )
        self.assertIn("[Missing texture]", missing_item.text())
        self.assertIn("1 surface", missing_item.text())
        atlas_data = TextureAtlasData()
        atlas = atlas_data.create_atlas("Missing", 2048, atlas_id="missing")
        atlas_data.assign_object(
            atlas.atlas_id,
            source_id,
            f"surface_textures/{assignment.asset_path}",
            512,
        )
        self.workspace.texture_atlas_workspace.set_data(atlas_data)

        self.workspace.surface_texture_generation.set_data(
            SurfaceTextureData()
        )
        self.workspace.surface_texture_generation.assignments_removed.emit(
            (assignment.assignment_id,)
        )

        updated = self.workspace.texture_atlas_workspace.get_data().atlas_by_id(
            atlas.atlas_id
        )
        assert updated is not None
        self.assertIsNone(updated.placement_for_object(source_id))

    def test_deleting_selected_surface_texture_clears_canvas_highlight(
        self,
    ) -> None:
        _add_square_room_to_level(self.workspace.current_level)
        surfaces = tuple(build_fixed_surfaces(self.workspace.levels))
        wall_ids = tuple(
            surface.surface_id
            for surface in surfaces
            if surface.surface_type == SURFACE_TYPE_WALL
        )
        first_wall, second_wall = wall_ids[:2]
        self.workspace.surface_texture_generation.set_levels(
            self.workspace.levels
        )
        self.workspace._set_canvas_viewer_targets(surfaces)
        asset_directory = self.settings.path.parent / "surface_textures"
        first_assignment = _wall_texture_assignment(
            asset_directory,
            assignment_id="selected-deleted",
            surface_ids=(first_wall,),
        )
        second_assignment = _wall_texture_assignment(
            asset_directory,
            assignment_id="replacement-selection",
            surface_ids=(second_wall,),
        )
        surface_workspace = self.workspace.surface_texture_generation
        surface_workspace.set_data(
            SurfaceTextureData(
                assignments=[first_assignment, second_assignment]
            )
        )
        surface_workspace.data_changed.emit(surface_workspace.get_data())
        first_source_id = build_atlas_wall_texture_source_id(
            first_assignment.assignment_id
        )
        self.workspace.texture_atlas_workspace.surface_list.object_clicked.emit(
            first_source_id,
            Qt.MouseButton.LeftButton,
        )
        self.assertEqual(
            self.workspace.viewer.get_selected_canvas_surface_ids(),
            (),
        )
        self.assertEqual(
            self.workspace.viewer.get_highlighted_canvas_surface_ids(),
            (first_wall,),
        )

        surface_workspace.set_data(
            SurfaceTextureData(assignments=[second_assignment])
        )
        surface_workspace.assignments_removed.emit(
            (first_assignment.assignment_id,)
        )

        self.assertEqual(
            self.workspace.viewer.get_selected_canvas_surface_ids(),
            (),
        )
        self.assertEqual(
            self.workspace._desired_canvas_surface_ids,
            (),
        )
        self.assertEqual(
            self.workspace.viewer.get_highlighted_canvas_surface_ids(),
            (second_wall,),
        )

    def test_partial_wall_assignment_keeps_its_packed_identity(self) -> None:
        assignment = _wall_texture_assignment(
            self.settings.path.parent / "surface_textures",
            surface_ids=("level:2/wall:1:2", "level:2/wall:2:3"),
        )
        surface_data = SurfaceTextureData(assignments=[assignment])
        self.workspace.surface_texture_generation.set_data(surface_data)
        self.workspace.surface_texture_generation.data_changed.emit(surface_data)
        source_id = build_atlas_wall_texture_source_id(
            assignment.assignment_id
        )
        source = self.workspace.texture_atlas_workspace._sources_by_object_id[
            source_id
        ]
        atlas_data = TextureAtlasData()
        atlas = atlas_data.create_atlas("Walls", 2048, atlas_id="walls")
        atlas_data.assign_object(
            atlas.atlas_id,
            source_id,
            source.texture_path,
            source.texture_resolution,
        )
        self.workspace.texture_atlas_workspace.set_data(atlas_data)
        retained_assignment = replace(
            assignment,
            surface_ids=("level:2/wall:2:3",),
        )
        retained_data = SurfaceTextureData(
            assignments=[retained_assignment]
        )

        self.workspace.surface_texture_generation.set_data(retained_data)
        self.workspace.surface_texture_generation.data_changed.emit(
            retained_data
        )

        packed = self.workspace.texture_atlas_workspace.get_data().atlas_by_id(
            atlas.atlas_id
        )
        assert packed is not None
        self.assertIsNotNone(packed.placement_for_object(source_id))
        item = self.workspace.texture_atlas_workspace.surface_list.item(0)
        self.assertIn("1 surface", item.text())

    def test_project_load_materializes_persisted_wall_texture_placement(
        self,
    ) -> None:
        levels = create_default_levels()
        wall_id = _add_square_room_to_level(levels[GROUND_LEVEL_INDEX])
        assignment = _wall_texture_assignment(
            self.settings.path.parent / "surface_textures",
            surface_ids=(wall_id,),
        )
        source_id = build_atlas_wall_texture_source_id(
            assignment.assignment_id
        )
        atlas_data = TextureAtlasData()
        atlas = atlas_data.create_atlas("Walls", 2048, atlas_id="walls")
        atlas_data.assign_object(
            atlas.atlas_id,
            source_id,
            f"surface_textures/{assignment.asset_path}",
            512,
        )
        project = ProjectData(
            blueprint_path=None,
            current_level_index=GROUND_LEVEL_INDEX,
            levels=levels,
            surface_texture_generation=SurfaceTextureData(
                assignments=[assignment]
            ),
            texture_atlases=atlas_data,
        )

        self.workspace._apply_loaded_project(project)

        loaded = self.workspace.texture_atlas_workspace.get_data().atlas_by_id(
            atlas.atlas_id
        )
        assert loaded is not None
        self.assertIsNotNone(loaded.image_path)
        assert loaded.image_path is not None
        atlas_png = self.settings.path.parent / "texture_atlases" / loaded.image_path
        self.assertTrue(atlas_png.is_file())
        with Image.open(atlas_png) as image:
            self.assertEqual(image.getpixel((256, 256)), (170, 70, 25, 255))

    def test_generation_packing_callback_routes_candidate_sources_to_atlas(
        self,
    ) -> None:
        old_record = GeneratedObjectRecord(
            object_id="chair",
            frame_index=0,
            object_name="Chair",
            pipeline={},
            provider_task_id="task-old",
            asset_path="chair-old.glb",
        )
        replacement = replace(
            old_record,
            pipeline={"symmetric_division": {"version": 1}},
            asset_path="chair-half.glb",
        )
        symmetry = SimpleNamespace(
            orientation="vertical",
            plane_coordinate=0.0,
        )
        variants = {
            resolution: SimpleNamespace(resolution=resolution)
            for resolution in (512, 1024, 2048)
        }
        sources = {
            resolution: SimpleNamespace(
                object_id="chair",
                texture_resolution=resolution,
            )
            for resolution in variants
        }
        commit = Mock(return_value=True)

        with patch.object(
            self.workspace.generation,
            "resolve_symmetric_division_for_record",
            return_value=symmetry,
        ), patch.object(
            self.workspace.generation,
            "resolve_texture_image_variant_for_record",
            side_effect=lambda _record, resolution: variants[resolution],
        ), patch.object(
            self.workspace,
            "_build_atlas_object_texture_source",
            side_effect=lambda variant, _symmetry: sources[variant.resolution],
        ), patch.object(
            self.workspace.texture_atlas_workspace,
            "transition_object_packing",
            return_value=True,
        ) as transition:
            accepted = (
                self.workspace
                ._handle_generation_object_packing_change_requested(
                    old_record,
                    replacement,
                    GeneratedModel(
                        mesh=trimesh.creation.box(),
                        scene=trimesh.Scene(),
                        glb_bytes=b"candidate",
                    ),
                    commit,
                )
            )

        self.assertTrue(accepted)
        transition.assert_called_once_with(
            "chair",
            [sources[512], sources[1024], sources[2048]],
            commit_callback=commit,
        )

    def test_face_edit_keeps_pinned_atlas_texture_pixels_and_path(
        self,
    ) -> None:
        generation_assets = self.workspace.generation._asset_directory
        generation_assets.mkdir(parents=True, exist_ok=True)
        texture_path = generation_assets / "chair-512.png"
        texture_color = (35, 75, 125, 255)
        Image.new("RGBA", (512, 512), texture_color).save(
            texture_path
        )
        old_record = GeneratedObjectRecord(
            object_id="chair",
            frame_index=0,
            object_name="Chair",
            pipeline={
                "texture_variants": {
                    "512": {
                        "glb_asset_path": "chair-old-512.glb",
                        "texture_asset_path": texture_path.name,
                    }
                },
                "selected_texture_resolution": 512,
            },
            provider_task_id="task-old",
            asset_path="chair-old-512.glb",
        )
        replacement = replace(
            old_record,
            pipeline={
                "texture_variants": {
                    "512": {
                        "glb_asset_path": "chair-face-edit-512.glb",
                        "texture_asset_path": texture_path.name,
                    }
                },
                "selected_texture_resolution": 512,
                "locally_authored_uvs": True,
            },
            asset_path="chair-face-edit-512.glb",
        )
        atlas_data = TextureAtlasData()
        atlas = atlas_data.create_atlas(
            "Edited object",
            2048,
            atlas_id="atlas-a",
        )
        atlas_data.assign_object(
            atlas.atlas_id,
            "chair",
            texture_path.name,
            512,
        )
        self.workspace.texture_atlas_workspace.set_data(atlas_data)
        commit = Mock(return_value=True)

        accepted = self.workspace \
            ._handle_generation_object_packing_change_requested(
                old_record,
                replacement,
                _generated_box_model(),
                commit,
            )

        self.assertTrue(accepted)
        commit.assert_called_once_with()
        updated_atlas = self.workspace.texture_atlas_workspace.selected_atlas
        self.assertIsNotNone(updated_atlas)
        assert updated_atlas is not None
        placement = updated_atlas.placement_for_object("chair")
        self.assertIsNotNone(placement)
        assert placement is not None
        self.assertEqual(placement.texture_path, texture_path.name)
        self.assertIsNotNone(updated_atlas.image_path)
        atlas_png = (
            self.workspace.texture_atlas_workspace._asset_directory
            / str(updated_atlas.image_path)
        )
        with Image.open(atlas_png) as image:
            self.assertEqual(
                image.getpixel((placement.x, placement.y)),
                texture_color,
            )

    def test_generation_quarter_packing_callback_skips_missing_2048_variant(
        self,
    ) -> None:
        old_record = GeneratedObjectRecord(
            object_id="chair",
            frame_index=0,
            object_name="Chair",
            pipeline={},
            provider_task_id="task-old",
            asset_path="chair-old.glb",
        )
        replacement = replace(
            old_record,
            pipeline={"symmetric_division": {"version": 2}},
            asset_path="chair-quarter.glb",
        )
        symmetry = SimpleNamespace(
            version=2,
            orientation="vertical",
            kept_side="left",
            plane_coordinate=0.0,
            packing_mode="symmetric_quarter",
            texture_content_quadrant="top_left",
        )
        variants = {
            resolution: SimpleNamespace(resolution=resolution)
            for resolution in (512, 1024)
        }
        sources = {
            resolution: SimpleNamespace(
                object_id="chair",
                texture_resolution=resolution,
            )
            for resolution in variants
        }
        commit = Mock(return_value=True)

        with patch.object(
            self.workspace.generation,
            "resolve_symmetric_division_for_record",
            return_value=symmetry,
        ), patch.object(
            self.workspace.generation,
            "resolve_texture_image_variant_for_record",
            side_effect=lambda _record, resolution: variants.get(resolution),
        ), patch.object(
            self.workspace,
            "_build_atlas_object_texture_source",
            side_effect=lambda variant, _symmetry: sources[variant.resolution],
        ), patch.object(
            self.workspace.texture_atlas_workspace,
            "transition_object_packing",
            return_value=True,
        ) as transition:
            accepted = (
                self.workspace
                ._handle_generation_object_packing_change_requested(
                    old_record,
                    replacement,
                    _generated_box_model(),
                    commit,
                )
            )

        self.assertTrue(accepted)
        transition.assert_called_once_with(
            "chair",
            [sources[512], sources[1024]],
            commit_callback=commit,
        )

    def test_generation_pair_packing_callback_skips_missing_2048_variant(
        self,
    ) -> None:
        old_record = GeneratedObjectRecord(
            object_id="chair",
            frame_index=0,
            object_name="Chair",
            pipeline={},
            provider_task_id="task-old",
            asset_path="chair-old.glb",
        )
        replacement = replace(
            old_record,
            pipeline={"symmetric_division": {"version": 3}},
            asset_path="chair-pair.glb",
        )
        symmetry = SimpleNamespace(
            version=3,
            orientation="vertical",
            kept_side="left",
            plane_coordinate=0.0,
            packing_mode="symmetric_pair",
            texture_content_half="left",
        )
        variants = {
            resolution: SimpleNamespace(resolution=resolution)
            for resolution in (512, 1024)
        }
        sources = {
            resolution: SimpleNamespace(
                object_id="chair",
                texture_resolution=resolution,
            )
            for resolution in variants
        }
        commit = Mock(return_value=True)

        with patch.object(
            self.workspace.generation,
            "resolve_symmetric_division_for_record",
            return_value=symmetry,
        ), patch.object(
            self.workspace.generation,
            "resolve_texture_image_variant_for_record",
            side_effect=lambda _record, resolution: variants.get(resolution),
        ), patch.object(
            self.workspace,
            "_build_atlas_object_texture_source",
            side_effect=lambda variant, _symmetry: sources[variant.resolution],
        ), patch.object(
            self.workspace.texture_atlas_workspace,
            "transition_object_packing",
            return_value=True,
        ) as transition:
            accepted = (
                self.workspace
                ._handle_generation_object_packing_change_requested(
                    old_record,
                    replacement,
                    _generated_box_model(),
                    commit,
                )
            )

        self.assertTrue(accepted)
        transition.assert_called_once_with(
            "chair",
            [sources[512], sources[1024]],
            commit_callback=commit,
        )

    def test_main_adapts_current_square_and_all_legacy_symmetry_metadata(
        self,
    ) -> None:
        asset_root = Path(self._temporary_directory.name)
        square_pair_png = asset_root / "chair-square-pair-512.png"
        pair_png = asset_root / "chair-pair-512.png"
        quarter_png = asset_root / "chair-quarter-512.png"
        half_png = asset_root / "chair-half-512.png"
        _write_texture_png(square_pair_png, 512, (90, 60, 30, 255))
        _write_texture_png(pair_png, 1024, (30, 60, 90, 255))
        _write_texture_png(quarter_png, 1024, (20, 40, 60, 255))
        _write_texture_png(half_png, 512, (60, 40, 20, 255))
        square_pair_variant = SimpleNamespace(
            object_id="square-pair",
            object_name="Square pair",
            resolution=512,
            texture_asset_relative_path=square_pair_png.name,
            texture_asset_path=square_pair_png,
        )
        pair_variant = SimpleNamespace(
            object_id="pair",
            object_name="Pair",
            resolution=512,
            texture_asset_relative_path=pair_png.name,
            texture_asset_path=pair_png,
        )
        quarter_variant = SimpleNamespace(
            object_id="quarter",
            object_name="Quarter",
            resolution=512,
            texture_asset_relative_path=quarter_png.name,
            texture_asset_path=quarter_png,
        )
        half_variant = SimpleNamespace(
            object_id="half",
            object_name="Half",
            resolution=512,
            texture_asset_relative_path=half_png.name,
            texture_asset_path=half_png,
        )
        square_pair_metadata = SimpleNamespace(
            version=4,
            orientation="vertical",
            plane_coordinate=0.0,
            packing_mode="symmetric_pair",
            texture_content_half="left",
        )
        quarter_metadata = SimpleNamespace(
            version=2,
            orientation="vertical",
            plane_coordinate=0.0,
            packing_mode="symmetric_quarter",
            texture_content_quadrant="top_left",
        )
        pair_metadata = SimpleNamespace(
            version=3,
            orientation="vertical",
            plane_coordinate=0.0,
            packing_mode="symmetric_pair",
            texture_content_half="left",
        )
        legacy_metadata = SimpleNamespace(
            version=1,
            orientation="horizontal",
            plane_coordinate=1.0,
        )

        square_pair_source = self.workspace._build_atlas_object_texture_source(
            square_pair_variant,
            square_pair_metadata,
        )
        pair_source = self.workspace._build_atlas_object_texture_source(
            pair_variant,
            pair_metadata,
        )
        quarter_source = self.workspace._build_atlas_object_texture_source(
            quarter_variant,
            quarter_metadata,
        )
        half_source = self.workspace._build_atlas_object_texture_source(
            half_variant,
            legacy_metadata,
        )

        assert (
            square_pair_source is not None
            and pair_source is not None
            and quarter_source is not None
            and half_source is not None
        )
        self.assertEqual(
            square_pair_source.packing_mode,
            ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR,
        )
        self.assertEqual(
            square_pair_source.load_texture_rgba().shape,
            (512, 512, 4),
        )
        self.assertEqual(
            pair_source.packing_mode,
            ATLAS_PACKING_MODE_SYMMETRIC_PAIR,
        )
        self.assertEqual(
            pair_source.load_texture_rgba().shape,
            (1024, 1024, 4),
        )
        self.assertEqual(
            quarter_source.packing_mode,
            ATLAS_PACKING_MODE_SYMMETRIC_QUARTER,
        )
        self.assertEqual(
            quarter_source.load_texture_rgba().shape,
            (1024, 1024, 4),
        )
        self.assertEqual(
            half_source.packing_mode,
            ATLAS_PACKING_MODE_SYMMETRIC_HALF,
        )

    def test_stroke_only_generation_change_does_not_reload_thumbnails(
        self,
    ) -> None:
        record = SimpleNamespace(object_id="chair")
        variant = SimpleNamespace(
            object_id="chair",
            object_name="Chair",
            resolution=1024,
            texture_asset_relative_path="chair-1024.png",
        )
        self.workspace._atlas_generation_signature = None

        with (
            patch.object(
                self.workspace.generation,
                "get_generated_object_ids",
                return_value=(record.object_id,),
            ),
            patch.object(
                self.workspace.generation,
                "get_placeable_object_names_by_id",
                return_value={record.object_id: "Chair"},
            ),
            patch.object(
                self.workspace.generation,
                "get_active_texture_variant",
                return_value=variant,
            ),
            patch.object(
                self.workspace,
                "_build_atlas_object_texture_source",
                return_value=object(),
            ) as source_builder,
            patch.object(
                self.workspace.texture_atlas_workspace,
                "set_object_texture_sources",
            ),
        ):
            self.workspace._handle_generation_data_changed_for_atlases(
                object()
            )
            self.workspace._handle_generation_data_changed_for_atlases(
                object()
            )

        source_builder.assert_called_once_with(variant, None)

    def test_atlas_activation_reloads_sources_only_after_file_revision_change(
        self,
    ) -> None:
        variant = SimpleNamespace(
            object_id="chair",
            object_name="Chair",
            resolution=1024,
            texture_asset_relative_path="chair-1024.png",
            texture_asset_path=Path(self._temporary_directory.name)
            / "chair-1024.png",
        )
        self.workspace._atlas_generation_signature = None
        texture_revision = [("png-revision-one",)]

        def dependency_signature(_object_id: str):
            return (
                (
                    1024,
                    "chair-1024.glb",
                    ("glb-revision",),
                    "chair-1024.png",
                    texture_revision[0],
                ),
            )

        with (
            patch.object(
                self.workspace.generation,
                "get_generated_object_ids",
                return_value=("chair",),
            ),
            patch.object(
                self.workspace.generation,
                "get_active_texture_variant",
                return_value=variant,
            ),
            patch.object(
                self.workspace.generation,
                "get_texture_variant_dependency_signature",
                side_effect=dependency_signature,
            ),
            patch.object(
                self.workspace,
                "_build_atlas_object_texture_source",
                return_value=object(),
            ) as source_builder,
            patch.object(
                self.workspace.texture_atlas_workspace,
                "set_object_texture_sources",
            ),
        ):
            self.workspace._sync_atlas_object_texture_sources()
            self.workspace._sync_atlas_object_texture_sources()
            source_builder.assert_called_once_with(variant, None)

            texture_revision[0] = ("png-revision-two",)
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.settings_widget
            )
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.texture_atlas_workspace
            )

        self.assertEqual(source_builder.call_count, 2)

    def test_atlas_source_cache_ignores_glb_only_revision_changes(self) -> None:
        variant = SimpleNamespace(
            object_id="chair",
            object_name="Chair",
            resolution=1024,
            texture_asset_relative_path="chair-1024.png",
        )
        glb_revision = [("glb-revision-one",)]

        def dependency_signature(_object_id: str):
            return (
                (
                    1024,
                    "chair-1024.glb",
                    glb_revision[0],
                    "chair-1024.png",
                    ("png-revision",),
                ),
            )

        self.workspace._atlas_generation_signature = None
        with (
            patch.object(
                self.workspace.generation,
                "get_generated_object_ids",
                return_value=("chair",),
            ),
            patch.object(
                self.workspace.generation,
                "get_active_texture_variant",
                return_value=variant,
            ),
            patch.object(
                self.workspace.generation,
                "get_texture_variant_dependency_signature",
                side_effect=dependency_signature,
            ),
            patch.object(
                self.workspace,
                "_build_atlas_object_texture_source",
                return_value=object(),
            ) as source_builder,
            patch.object(
                self.workspace.texture_atlas_workspace,
                "set_object_texture_sources",
            ) as set_sources,
        ):
            self.workspace._sync_atlas_object_texture_sources()
            glb_revision[0] = ("glb-revision-two",)
            self.workspace._sync_atlas_object_texture_sources()

        source_builder.assert_called_once_with(variant, None)
        set_sources.assert_called_once()

    def test_unchanged_wall_source_is_cached_but_file_change_reloads_it(
        self,
    ) -> None:
        assignment = _wall_texture_assignment(
            self.settings.path.parent / "surface_textures"
        )
        surface_data = SurfaceTextureData(assignments=[assignment])
        self.workspace.surface_texture_generation.set_data(surface_data)
        self.workspace._atlas_generation_signature = None

        with patch.object(
            self.workspace,
            "_build_atlas_wall_texture_source",
            wraps=self.workspace._build_atlas_wall_texture_source,
        ) as source_builder:
            self.workspace._handle_surface_texture_data_changed_for_atlases(
                surface_data
            )
            self.workspace._handle_surface_texture_data_changed_for_atlases(
                surface_data
            )
            texture_path = (
                self.settings.path.parent
                / "surface_textures"
                / assignment.asset_path
            )
            previous_stat = texture_path.stat()
            Image.new("RGBA", (13, 8), (20, 90, 160, 255)).save(
                texture_path
            )
            os.utime(
                texture_path,
                ns=(
                    previous_stat.st_atime_ns,
                    previous_stat.st_mtime_ns + 1_000_000_000,
                ),
            )
            self.workspace._handle_surface_texture_data_changed_for_atlases(
                surface_data
            )

        self.assertEqual(source_builder.call_count, 2)

    def test_atlas_activation_rematerializes_same_path_wall_pixels(
        self,
    ) -> None:
        surface_asset_directory = self.settings.path.parent / "surface_textures"
        assignment = _wall_texture_assignment(surface_asset_directory)
        self.workspace.surface_texture_generation.set_data(
            SurfaceTextureData(assignments=[assignment])
        )
        self.workspace._atlas_generation_signature = None
        self.workspace._atlas_source_content_paths = None
        self.workspace._atlas_source_content_revisions = None
        self.workspace._sync_atlas_object_texture_sources()

        source_id = build_atlas_wall_texture_source_id(
            assignment.assignment_id
        )
        source = self.workspace.texture_atlas_workspace \
            ._sources_by_object_id[source_id]
        atlas_data = TextureAtlasData()
        atlas = atlas_data.create_atlas(
            "Same-path wall",
            2048,
            atlas_id="same-path-wall",
        )
        atlas_data.assign_object(
            atlas.atlas_id,
            source.object_id,
            source.texture_path,
            source.texture_resolution,
            source.packing_mode,
        )
        atlas_workspace = self.workspace.texture_atlas_workspace
        atlas_workspace.set_data(atlas_data)
        self.assertEqual(atlas_workspace.materialize_missing_atlases(), 1)
        atlas_path = (
            self.settings.path.parent
            / "texture_atlases"
            / f"{atlas.atlas_id}.png"
        )
        with Image.open(atlas_path) as atlas_image:
            self.assertEqual(
                atlas_image.convert("RGBA").getpixel((256, 256)),
                (170, 70, 25, 255),
            )

        texture_path = surface_asset_directory / assignment.asset_path
        Image.new("RGBA", (12, 8), (20, 90, 160, 255)).save(texture_path)
        current_stat = texture_path.stat()
        os.utime(
            texture_path,
            ns=(
                current_stat.st_atime_ns,
                current_stat.st_mtime_ns + 1_000_000,
            ),
        )
        self.workspace.workspace_tabs.setCurrentWidget(
            self.workspace.settings_widget
        )
        self.workspace.workspace_tabs.setCurrentWidget(atlas_workspace)

        with Image.open(atlas_path) as atlas_image:
            self.assertEqual(
                atlas_image.convert("RGBA").getpixel((256, 256)),
                (20, 90, 160, 255),
            )

    def test_wall_pbr_paths_and_map_only_revision_refresh_atlas_content(
        self,
    ) -> None:
        surface_asset_directory = self.settings.path.parent / "surface_textures"
        assignment = _wall_texture_assignment_with_pbr_variants(
            surface_asset_directory
        )
        surface_data = SurfaceTextureData(assignments=[assignment])
        self.workspace.surface_texture_generation.set_data(surface_data)
        self.workspace._atlas_generation_signature = None
        self.workspace._atlas_source_content_paths = None
        self.workspace._atlas_source_content_revisions = None
        self.workspace._sync_atlas_object_texture_sources()

        source_id = build_atlas_wall_texture_source_id(
            assignment.assignment_id
        )
        atlas_workspace = self.workspace.texture_atlas_workspace
        source = atlas_workspace._sources_by_object_id[source_id]
        active_variant = assignment.texture_variant_for_resolution(512)
        assert active_variant is not None
        expected_logical_paths = {
            map_type: f"surface_textures/{asset_path}"
            for map_type, asset_path in active_variant.map_asset_paths.items()
        }
        expected_physical_paths = {
            map_type: surface_asset_directory / asset_path
            for map_type, asset_path in active_variant.map_asset_paths.items()
        }
        self.assertEqual(dict(source.map_texture_paths), expected_logical_paths)
        self.assertEqual(
            dict(source.physical_map_texture_paths),
            expected_physical_paths,
        )
        np.testing.assert_array_equal(
            source.load_texture_rgba(PBR_MAP_NORMAL)[256, 256],
            np.asarray((110, 120, 230, 255), dtype=np.uint8),
        )

        atlas_data = TextureAtlasData()
        atlas = atlas_data.create_atlas(
            "Surface PBR refresh",
            2048,
            atlas_id="surface-pbr-refresh",
        )
        atlas_data.assign_object(
            atlas.atlas_id,
            source.object_id,
            source.texture_path,
            source.texture_resolution,
            source.packing_mode,
        )
        atlas_workspace.set_data(atlas_data)
        self.assertEqual(atlas_workspace.materialize_missing_atlases(), 1)
        normal_atlas_path = (
            self.settings.path.parent
            / "texture_atlases"
            / build_texture_atlas_map_image_relative_path(
                atlas.atlas_id,
                PBR_MAP_NORMAL,
            )
        )
        base_atlas_path = (
            self.settings.path.parent
            / "texture_atlases"
            / f"{atlas.atlas_id}.png"
        )
        with Image.open(normal_atlas_path) as normal_atlas:
            self.assertEqual(
                normal_atlas.convert("RGBA").getpixel((256, 256)),
                (110, 120, 230, 255),
            )

        normal_path = expected_physical_paths[PBR_MAP_NORMAL]
        Image.new("RGBA", (512, 512), (25, 190, 140, 255)).save(
            normal_path,
            format="PNG",
        )
        current_stat = normal_path.stat()
        os.utime(
            normal_path,
            ns=(
                current_stat.st_atime_ns,
                current_stat.st_mtime_ns + 1_000_000,
            ),
        )
        self.workspace._handle_surface_texture_data_changed_for_atlases(
            surface_data
        )

        refreshed_source = atlas_workspace._sources_by_object_id[source_id]
        self.assertEqual(
            dict(refreshed_source.map_texture_paths),
            expected_logical_paths,
        )
        np.testing.assert_array_equal(
            refreshed_source.load_texture_rgba(PBR_MAP_NORMAL)[256, 256],
            np.asarray((25, 190, 140, 255), dtype=np.uint8),
        )
        with Image.open(normal_atlas_path) as normal_atlas:
            self.assertEqual(
                normal_atlas.convert("RGBA").getpixel((256, 256)),
                (25, 190, 140, 255),
            )
        with Image.open(base_atlas_path) as base_atlas:
            self.assertEqual(
                base_atlas.convert("RGBA").getpixel((256, 256)),
                (40, 60, 80, 255),
            )

    def test_surface_tiling_preview_allows_changed_rotation_with_equal_seam_score(
        self,
    ) -> None:
        asset_directory = self.settings.path.parent / "surface_textures"
        assignment = _wall_texture_assignment(
            asset_directory,
            assignment_id="rotated-preview",
        )
        surface_workspace = self.workspace.surface_texture_generation
        surface_workspace.set_data(SurfaceTextureData(assignments=[assignment]))
        candidate = _prepared_surface_tiling_repair(
            surface_workspace,
            assignment,
            map_colors={ATLAS_MAP_BASE_COLOR: (25, 170, 210, 255)},
        )
        source_id = build_atlas_wall_texture_source_id(assignment.assignment_id)
        thread = Mock(result=candidate, error_message=None, was_cancelled=False)
        self.workspace._surface_texture_tiling_threads[source_id] = thread

        with (
            patch("housemaker.main.SurfaceTextureTilingPreviewDialog") as dialog,
            patch.object(
                self.workspace,
                "_commit_surface_texture_tiling_repair",
            ) as commit,
        ):
            dialog.return_value.exec.return_value = QDialog.DialogCode.Accepted
            self.workspace._handle_surface_texture_tiling_prepared(
                source_id,
                thread,
            )

        dialog.assert_called_once()
        commit.assert_called_once_with(source_id, candidate)
        thread.deleteLater.assert_called_once_with()

    def test_surface_tiling_button_prepares_edge_variants(self) -> None:
        source_id = build_atlas_wall_texture_source_id("dispatch-wall")
        surface_workspace = self.workspace.surface_texture_generation
        with (
            patch.object(
                surface_workspace, "snapshot_assignment_tiling_repair"
            ) as snapshot,
            patch(
                "housemaker.main._SurfaceTextureTilingPreparationThread"
            ) as thread_type,
        ):
            thread_type.return_value.isRunning.return_value = False
            atlas_workspace = self.workspace.texture_atlas_workspace
            atlas_workspace.surface_texture_fix_tiling_requested.emit(source_id)

        snapshot.assert_called_once_with(
            "dispatch-wall", method=SURFACE_TILING_MODE_EDGE_VARIANTS
        )
        thread_type.return_value.start.assert_called_once_with()

    def test_surface_ao_snapshot_preserves_tiling_material_spec(self) -> None:
        assignment = replace(
            _wall_texture_assignment(
                self.settings.path.parent / "surface_textures",
                assignment_id="ao-rotated-wall",
            ),
            tiling_mode=SURFACE_TILING_MODE_WHOLE_REPEATS,
            tiling_seed=73,
            texture_repeat_size_m=1.5,
        )
        self.workspace.surface_texture_generation.set_data(
            SurfaceTextureData(assignments=[assignment])
        )
        snapshot = self.workspace._capture_surface_ao_scene_snapshot(
            ((), (), ())
        )
        source = dict(snapshot.surface_materials)[assignment.surface_ids[0]]
        self.assertIsInstance(source, SurfaceMaterialSourceSpec)
        self.assertEqual(source.tiling_mode, SURFACE_TILING_MODE_WHOLE_REPEATS)
        self.assertEqual(source.tiling_seed, 73)
        self.assertEqual(source.texture_repeat_size_m, 1.5)

        mesh = trimesh.creation.box()
        base_model = GeneratedModel(
            mesh=mesh,
            scene=trimesh.Scene(mesh.copy()),
            glb_bytes=b"",
        )
        with patch(
            "housemaker.main.convert_to_export_scene_model",
            return_value=base_model,
        ) as convert:
            scene = _build_surface_ao_pre_atlas_scene(
                snapshot, lambda: False
            )

        self.assertIs(scene.model, base_model)
        rebuilt = convert.call_args.kwargs["surface_materials"][
            assignment.surface_ids[0]
        ]
        self.assertIsInstance(rebuilt, SurfaceMaterialSourceSpec)
        self.assertEqual(rebuilt.tiling_mode, SURFACE_TILING_MODE_WHOLE_REPEATS)
        self.assertEqual(rebuilt.tiling_seed, 73)
        self.assertEqual(rebuilt.texture_repeat_size_m, 1.5)

    def test_surface_tiling_revision_updates_atlas_in_place_and_undoes(
        self,
    ) -> None:
        surface_asset_directory = self.settings.path.parent / "surface_textures"
        assignment = _wall_texture_assignment_with_pbr_variants(
            surface_asset_directory,
            assignment_id="tiling-wall",
        )
        surface_workspace = self.workspace.surface_texture_generation
        surface_workspace.set_data(
            SurfaceTextureData(assignments=[assignment])
        )
        object_record = _generated_object_record_with_variants(
            self.settings.path.parent / "generated",
            object_id="tiling-neighbour",
            object_name="Tiling neighbour",
            resolutions=(512,),
            selected_resolution=512,
        )
        self.workspace.generation.set_data(
            GenerationData(generated_objects=[object_record])
        )
        self.workspace._atlas_generation_signature = None
        self.workspace._sync_atlas_object_texture_sources(
            automatically_assign_scene_textures=False
        )

        source_id = build_atlas_wall_texture_source_id(
            assignment.assignment_id
        )
        atlas_workspace = self.workspace.texture_atlas_workspace
        surface_source = atlas_workspace._sources_by_object_id[source_id]
        object_source = atlas_workspace._sources_by_object_id[
            object_record.object_id
        ]
        atlas_data = TextureAtlasData()
        atlas = atlas_data.create_atlas(
            "Tiling transaction",
            2048,
            atlas_id="tiling-transaction",
        )
        atlas_data.assign_object(
            atlas.atlas_id,
            surface_source.object_id,
            surface_source.texture_path,
            surface_source.texture_resolution,
        )
        atlas_data.assign_object(
            atlas.atlas_id,
            object_source.object_id,
            object_source.texture_path,
            object_source.texture_resolution,
        )
        atlas_workspace.set_data(atlas_data)
        self.assertEqual(atlas_workspace.materialize_missing_atlases(), 1)

        original_atlas = atlas_workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert original_atlas is not None
        original_surface_placement = original_atlas.placement_for_object(source_id)
        original_object_placement = original_atlas.placement_for_object(
            object_record.object_id
        )
        assert original_surface_placement is not None
        assert original_object_placement is not None
        original_surface_files = {
            raw_path: (surface_asset_directory / raw_path).read_bytes()
            for variant in assignment.texture_variants
            for raw_path in variant.map_asset_paths.values()
        }
        object_texture_path = object_source.physical_texture_path
        original_object_texture = object_texture_path.read_bytes()
        atlas_map_paths = {
            map_type: (
                self.settings.path.parent
                / "texture_atlases"
                / build_texture_atlas_map_image_relative_path(
                    atlas.atlas_id,
                    map_type,
                )
            )
            for map_type in (
                ATLAS_MAP_BASE_COLOR,
                PBR_MAP_NORMAL,
                PBR_MAP_ROUGHNESS,
                PBR_MAP_METALLIC,
            )
        }
        original_atlas_payloads = {
            map_type: path.read_bytes()
            for map_type, path in atlas_map_paths.items()
        }
        repaired_colors = {
            ATLAS_MAP_BASE_COLOR: (210, 35, 65, 255),
            PBR_MAP_NORMAL: (90, 175, 240, 255),
            PBR_MAP_ROUGHNESS: (145, 145, 145, 255),
            PBR_MAP_METALLIC: (25, 25, 25, 255),
        }
        candidate = _prepared_surface_tiling_repair(
            surface_workspace,
            assignment,
            map_colors=repaired_colors,
        )

        self.workspace._commit_surface_texture_tiling_repair(
            source_id,
            candidate,
        )

        repaired_assignment = surface_workspace.get_assignment(
            assignment.assignment_id
        )
        assert repaired_assignment is not None
        self.assertEqual(
            replace(
                repaired_assignment,
                asset_path=assignment.asset_path,
                texture_variants=assignment.texture_variants,
                tiling_mode=assignment.tiling_mode,
                tiling_seed=assignment.tiling_seed,
            ),
            assignment,
        )
        self.assertEqual(
            repaired_assignment.tiling_mode,
            SURFACE_TILING_MODE_EDGE_VARIANTS,
        )
        self.assertEqual(
            tuple(
                (variant.resolution, tuple(variant.map_asset_paths))
                for variant in repaired_assignment.texture_variants
            ),
            tuple(
                (variant.resolution, tuple(variant.map_asset_paths))
                for variant in assignment.texture_variants
            ),
        )
        self.assertNotEqual(
            repaired_assignment.asset_path,
            assignment.asset_path,
            atlas_workspace.status_label.text(),
        )
        for raw_path, payload in original_surface_files.items():
            self.assertEqual(
                (surface_asset_directory / raw_path).read_bytes(),
                payload,
            )
        for variant in repaired_assignment.texture_variants:
            for map_type, raw_path in variant.map_asset_paths.items():
                self.assertNotIn(raw_path, original_surface_files)
                with Image.open(surface_asset_directory / raw_path) as image:
                    self.assertEqual(
                        image.convert("RGBA").getpixel((0, 0)),
                        repaired_colors[map_type],
                    )

        repaired_atlas = atlas_workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert repaired_atlas is not None
        repaired_surface_placement = repaired_atlas.placement_for_object(source_id)
        repaired_object_placement = repaired_atlas.placement_for_object(
            object_record.object_id
        )
        assert repaired_surface_placement is not None
        assert repaired_object_placement is not None
        self.assertEqual(
            (
                repaired_surface_placement.x,
                repaired_surface_placement.y,
                repaired_surface_placement.size,
                repaired_surface_placement.packing_mode,
                repaired_surface_placement.slot_half,
                repaired_surface_placement.slot_quadrant,
            ),
            (
                original_surface_placement.x,
                original_surface_placement.y,
                original_surface_placement.size,
                original_surface_placement.packing_mode,
                original_surface_placement.slot_half,
                original_surface_placement.slot_quadrant,
            ),
        )
        self.assertNotEqual(
            repaired_surface_placement.texture_path,
            original_surface_placement.texture_path,
        )
        self.assertEqual(
            repaired_object_placement.to_dict(),
            original_object_placement.to_dict(),
        )
        self.assertEqual(object_texture_path.read_bytes(), original_object_texture)
        for map_type, path in atlas_map_paths.items():
            self.assertNotEqual(
                path.read_bytes(),
                original_atlas_payloads[map_type],
            )
            with Image.open(path) as atlas_image:
                self.assertEqual(
                    atlas_image.convert("RGBA").getpixel(
                        (
                            repaired_surface_placement.x + 256,
                            repaired_surface_placement.y + 256,
                        )
                    ),
                    repaired_colors[map_type],
                )
        self.assertEqual(len(self.workspace._canvas_undo_stack), 1)
        repaired_paths = tuple(
            surface_asset_directory / raw_path
            for variant in repaired_assignment.texture_variants
            for raw_path in variant.map_asset_paths.values()
        )

        self.workspace._handle_canvas_undo_requested()

        self.assertEqual(
            surface_workspace.get_assignment(assignment.assignment_id),
            assignment,
        )
        restored_atlas = atlas_workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert restored_atlas is not None
        restored_surface_placement = restored_atlas.placement_for_object(source_id)
        restored_object_placement = restored_atlas.placement_for_object(
            object_record.object_id
        )
        assert restored_surface_placement is not None
        assert restored_object_placement is not None
        self.assertEqual(
            restored_surface_placement.to_dict(),
            original_surface_placement.to_dict(),
        )
        self.assertEqual(
            restored_object_placement.to_dict(),
            original_object_placement.to_dict(),
        )
        for map_type, path in atlas_map_paths.items():
            self.assertEqual(
                path.read_bytes(),
                original_atlas_payloads[map_type],
            )
        self.assertTrue(all(not path.exists() for path in repaired_paths))
        self.assertEqual(object_texture_path.read_bytes(), original_object_texture)
        self.assertEqual(self.workspace._canvas_undo_stack, [])

    def test_surface_tiling_atlas_failure_rolls_back_derived_revision(
        self,
    ) -> None:
        surface_asset_directory = self.settings.path.parent / "surface_textures"
        assignment = _wall_texture_assignment(
            surface_asset_directory,
            assignment_id="tiling-rollback",
        )
        surface_workspace = self.workspace.surface_texture_generation
        surface_workspace.set_data(
            SurfaceTextureData(assignments=[assignment])
        )
        self.workspace._atlas_generation_signature = None
        self.workspace._sync_atlas_object_texture_sources(
            automatically_assign_scene_textures=False
        )
        source_id = build_atlas_wall_texture_source_id(
            assignment.assignment_id
        )
        atlas_workspace = self.workspace.texture_atlas_workspace
        source = atlas_workspace._sources_by_object_id[source_id]
        atlas_data = TextureAtlasData()
        atlas = atlas_data.create_atlas(
            "Tiling rollback",
            2048,
            atlas_id="tiling-rollback",
        )
        atlas_data.assign_object(
            atlas.atlas_id,
            source.object_id,
            source.texture_path,
            source.texture_resolution,
        )
        atlas_workspace.set_data(atlas_data)
        self.assertEqual(atlas_workspace.materialize_missing_atlases(), 1)
        before_data = atlas_workspace.get_data().to_dict()
        before_files = {
            path.name: path.read_bytes()
            for path in surface_asset_directory.iterdir()
            if path.is_file()
        }
        atlas_path = (
            self.settings.path.parent
            / "texture_atlases"
            / f"{atlas.atlas_id}.png"
        )
        before_atlas_payload = atlas_path.read_bytes()
        candidate = _prepared_surface_tiling_repair(
            surface_workspace,
            assignment,
            map_colors={ATLAS_MAP_BASE_COLOR: (25, 170, 210, 255)},
        )

        with patch.object(
            atlas_workspace,
            "_materialize_atlas",
            side_effect=ValueError("forced Atlas failure"),
        ):
            self.workspace._commit_surface_texture_tiling_repair(
                source_id,
                candidate,
            )

        self.assertEqual(
            surface_workspace.get_assignment(assignment.assignment_id),
            assignment,
        )
        self.assertEqual(atlas_workspace.get_data().to_dict(), before_data)
        self.assertEqual(atlas_path.read_bytes(), before_atlas_payload)
        self.assertEqual(
            {
                path.name: path.read_bytes()
                for path in surface_asset_directory.iterdir()
                if path.is_file()
            },
            before_files,
        )
        self.assertEqual(self.workspace._canvas_undo_stack, [])
        self.assertIn(
            "original texture and Atlas were kept",
            atlas_workspace.status_label.text(),
        )

    def test_direct_object_change_refreshes_pbr_path_and_revision_changes(
        self,
    ) -> None:
        asset_directory = self.settings.path.parent / "generated"
        asset_directory.mkdir(parents=True, exist_ok=True)
        glb_name = "pbr-refresh.glb"
        base_name = "pbr-refresh.png"
        first_normal_name = "pbr-refresh-normal-a.png"
        second_normal_name = "pbr-refresh-normal-b.png"
        (asset_directory / glb_name).write_bytes(b"available glb")
        Image.new("RGBA", (512, 512), (40, 60, 80, 255)).save(
            asset_directory / base_name
        )
        Image.new("RGBA", (512, 512), (110, 120, 230, 255)).save(
            asset_directory / first_normal_name
        )
        variant_metadata = {
            "glb_asset_path": glb_name,
            "texture_asset_path": base_name,
            "map_texture_asset_paths": {
                "base_color": base_name,
                "normal": first_normal_name,
            },
        }
        record = GeneratedObjectRecord(
            object_id="pbr-refresh",
            frame_index=0,
            object_name="PBR refresh",
            pipeline={
                "texture_variants": {"512": variant_metadata},
                "selected_texture_resolution": 512,
            },
            provider_task_id="pbr-refresh-task",
            asset_path=glb_name,
        )
        self.workspace.generation._data.generated_objects = [record]
        self.workspace._atlas_generation_signature = None
        self.workspace._atlas_source_content_paths = None
        self.workspace._atlas_source_content_revisions = None
        self.workspace._sync_atlas_object_texture_sources()

        atlas_workspace = self.workspace.texture_atlas_workspace
        source = atlas_workspace._sources_by_object_id[record.object_id]
        atlas_data = TextureAtlasData()
        atlas = atlas_data.create_atlas(
            "PBR refresh",
            2048,
            atlas_id="pbr-refresh-atlas",
        )
        atlas_data.assign_object(
            atlas.atlas_id,
            source.object_id,
            source.texture_path,
            source.texture_resolution,
            source.packing_mode,
        )
        atlas_workspace.set_data(atlas_data)
        self.assertEqual(atlas_workspace.materialize_missing_atlases(), 1)
        normal_atlas_path = (
            self.settings.path.parent
            / "texture_atlases"
            / build_texture_atlas_map_image_relative_path(
                atlas.atlas_id,
                PBR_MAP_NORMAL,
            )
        )
        with Image.open(normal_atlas_path) as normal_atlas:
            self.assertEqual(
                normal_atlas.convert("RGBA").getpixel((256, 256)),
                (110, 120, 230, 255),
            )

        Image.new("RGBA", (512, 512), (70, 150, 210, 255)).save(
            asset_directory / second_normal_name
        )
        variant_metadata["map_texture_asset_paths"][
            "normal"
        ] = second_normal_name
        self.workspace._handle_generated_object_changed_for_atlases(
            record,
            object(),
        )

        with Image.open(normal_atlas_path) as normal_atlas:
            self.assertEqual(
                normal_atlas.convert("RGBA").getpixel((256, 256)),
                (70, 150, 210, 255),
            )

        second_normal_path = asset_directory / second_normal_name
        Image.new("RGBA", (512, 512), (25, 190, 140, 255)).save(
            second_normal_path
        )
        current_stat = second_normal_path.stat()
        os.utime(
            second_normal_path,
            ns=(
                current_stat.st_atime_ns,
                current_stat.st_mtime_ns + 1_000_000,
            ),
        )
        self.workspace._handle_generated_object_changed_for_atlases(
            record,
            object(),
        )

        with Image.open(normal_atlas_path) as normal_atlas:
            self.assertEqual(
                normal_atlas.convert("RGBA").getpixel((256, 256)),
                (25, 190, 140, 255),
            )

    def test_fixed_wall_replacement_keeps_its_pinned_atlas_resolution(
        self,
    ) -> None:
        surface_asset_directory = self.settings.path.parent / "surface_textures"
        assignment = _wall_texture_assignment(
            surface_asset_directory,
            assignment_id="cross-bucket-wall",
            size=(400, 300),
            color=(170, 70, 25, 255),
        )
        self.workspace.surface_texture_generation.set_data(
            SurfaceTextureData(assignments=[assignment])
        )
        self.workspace._atlas_generation_signature = None
        self.workspace._sync_atlas_object_texture_sources()
        source_id = build_atlas_wall_texture_source_id(
            assignment.assignment_id
        )
        source = self.workspace.texture_atlas_workspace \
            ._sources_by_object_id[source_id]
        self.assertEqual(source.texture_resolution, 512)

        atlas_data = TextureAtlasData()
        atlas = atlas_data.create_atlas(
            "Pinned fixed wall",
            2048,
            atlas_id="pinned-fixed-wall",
        )
        atlas_data.assign_object(
            atlas.atlas_id,
            source.object_id,
            source.texture_path,
            source.texture_resolution,
            source.packing_mode,
        )
        atlas_workspace = self.workspace.texture_atlas_workspace
        atlas_workspace.set_data(atlas_data)
        self.assertEqual(atlas_workspace.materialize_missing_atlases(), 1)
        atlas_path = (
            self.settings.path.parent
            / "texture_atlases"
            / f"{atlas.atlas_id}.png"
        )

        texture_path = surface_asset_directory / assignment.asset_path
        Image.new("RGBA", (900, 700), (20, 90, 160, 255)).save(texture_path)
        current_stat = texture_path.stat()
        os.utime(
            texture_path,
            ns=(
                current_stat.st_atime_ns,
                current_stat.st_mtime_ns + 1_000_000,
            ),
        )
        self.workspace._sync_atlas_object_texture_sources()

        updated = atlas_workspace.get_data().atlas_by_id(atlas.atlas_id)
        assert updated is not None
        placement = updated.placement_for_object(source_id)
        assert placement is not None
        self.assertEqual(placement.texture_resolution, 512)
        with Image.open(atlas_path) as atlas_image:
            self.assertEqual(
                atlas_image.convert("RGBA").getpixel((256, 256)),
                (20, 90, 160, 255),
            )

    def test_generated_png_reappearance_rematerializes_same_atlas_slot(
        self,
    ) -> None:
        asset_directory = self.settings.path.parent / "generated"
        asset_directory.mkdir(parents=True, exist_ok=True)
        glb_name = "reappearing-object.glb"
        png_name = "reappearing-object.png"
        (asset_directory / glb_name).write_bytes(b"available glb")
        texture_path = asset_directory / png_name
        Image.new("RGBA", (512, 512), (170, 70, 25, 255)).save(texture_path)
        record = GeneratedObjectRecord(
            object_id="reappearing-object",
            frame_index=0,
            object_name="Reappearing object",
            pipeline={
                "texture_variants": {
                    "512": {
                        "glb_asset_path": glb_name,
                        "texture_asset_path": png_name,
                    },
                },
                "selected_texture_resolution": 512,
            },
            provider_task_id="reappearing-object-task",
            asset_path=glb_name,
        )
        self.workspace.generation._data.generated_objects = [record]
        self.workspace._atlas_generation_signature = None
        self.workspace._atlas_source_content_paths = None
        self.workspace._atlas_source_content_revisions = None
        self.workspace._sync_atlas_object_texture_sources()
        atlas_workspace = self.workspace.texture_atlas_workspace
        source = atlas_workspace._sources_by_object_id[record.object_id]
        atlas_data = TextureAtlasData()
        atlas = atlas_data.create_atlas(
            "Reappearing generated texture",
            2048,
            atlas_id="reappearing-generated",
        )
        atlas_data.assign_object(
            atlas.atlas_id,
            source.object_id,
            source.texture_path,
            source.texture_resolution,
            source.packing_mode,
        )
        atlas_workspace.set_data(atlas_data)
        self.assertEqual(atlas_workspace.materialize_missing_atlases(), 1)
        atlas_path = (
            self.settings.path.parent
            / "texture_atlases"
            / f"{atlas.atlas_id}.png"
        )

        texture_path.unlink()
        self.workspace._sync_atlas_object_texture_sources()
        self.assertIsNotNone(self.workspace._atlas_generation_signature)
        Image.new("RGBA", (512, 512), (20, 90, 160, 255)).save(texture_path)
        self.workspace._sync_atlas_object_texture_sources()

        with Image.open(atlas_path) as atlas_image:
            self.assertEqual(
                atlas_image.convert("RGBA").getpixel((256, 256)),
                (20, 90, 160, 255),
            )

    def test_missing_wall_does_not_block_other_atlas_sources(self) -> None:
        asset_directory = self.settings.path.parent / "surface_textures"
        valid_assignment = _wall_texture_assignment(
            asset_directory,
            assignment_id="available-wall",
        )
        missing_assignment = _wall_texture_assignment(
            asset_directory,
            assignment_id="missing-wall",
        )
        (asset_directory / missing_assignment.asset_path).unlink()
        self.workspace.surface_texture_generation.set_data(
            SurfaceTextureData(
                assignments=[valid_assignment, missing_assignment]
            )
        )
        self.workspace._atlas_generation_signature = None

        self.workspace._sync_atlas_object_texture_sources()

        sources = self.workspace.texture_atlas_workspace._sources_by_object_id
        self.assertIn(
            build_atlas_wall_texture_source_id(valid_assignment.assignment_id),
            sources,
        )
        self.assertNotIn(
            build_atlas_wall_texture_source_id(missing_assignment.assignment_id),
            sources,
        )
        self.assertIsNotNone(self.workspace._atlas_generation_signature)

    def test_missing_used_surface_texture_still_blocks_export(self) -> None:
        wall_surface_id = _add_square_room_to_level(
            self.workspace.current_level
        )
        self.workspace.surface_texture_generation.set_levels(
            self.workspace.levels
        )
        assignment = _wall_texture_assignment(
            self.settings.path.parent / "surface_textures",
            assignment_id="missing-used-wall",
            surface_ids=(wall_surface_id,),
        )
        (
            self.settings.path.parent
            / "surface_textures"
            / assignment.asset_path
        ).unlink()
        surface_data = SurfaceTextureData(assignments=[assignment])

        self.workspace.surface_texture_generation.set_data(surface_data)
        self.workspace.surface_texture_generation.data_changed.emit(
            surface_data
        )

        source_id = build_atlas_wall_texture_source_id(
            assignment.assignment_id
        )
        self.assertEqual(
            self.workspace.texture_atlas_workspace
            .get_unpacked_scene_texture_source_ids(),
            (source_id,),
        )
        with patch("housemaker.main.QMessageBox.warning") as warning:
            self.assertTrue(
                self.workspace._show_unpacked_scene_texture_export_error()
            )
        warning.assert_called_once()
        self.assertIn("Wall texture", warning.call_args.args[2])

    def test_failed_atlas_source_build_retries_same_revision(self) -> None:
        variant = SimpleNamespace(
            object_id="retry-source",
            object_name="Retry source",
            resolution=512,
            texture_asset_relative_path="retry-source.png",
            texture_asset_path=Path(self._temporary_directory.name)
            / "retry-source.png",
        )
        dependency_signature = (
            (512, "retry-source.glb", ("glb",), "retry-source.png", ("png",)),
        )
        replacement_source = SimpleNamespace(object_id="retry-source")
        self.workspace._atlas_generation_signature = None

        with (
            patch.object(
                self.workspace.generation,
                "get_generated_object_ids",
                return_value=(variant.object_id,),
            ),
            patch.object(
                self.workspace.generation,
                "get_active_texture_variant",
                return_value=variant,
            ),
            patch.object(
                self.workspace.generation,
                "get_texture_variant_dependency_signature",
                return_value=dependency_signature,
            ),
            patch.object(
                self.workspace,
                "_build_atlas_object_texture_source",
                side_effect=(None, replacement_source),
            ) as build_source,
            patch.object(
                self.workspace.texture_atlas_workspace,
                "set_object_texture_sources",
            ) as set_sources,
            patch.object(
                self.workspace.texture_atlas_workspace,
                "refresh_texture_source_content",
                return_value=True,
            ),
        ):
            self.workspace._sync_atlas_object_texture_sources()
            self.workspace._sync_atlas_object_texture_sources()

        self.assertEqual(build_source.call_count, 2)
        self.assertEqual(set_sources.call_count, 2)
        self.assertEqual(set_sources.call_args_list[0].args[0], [])
        self.assertEqual(
            set_sources.call_args_list[1].args[0],
            [replacement_source],
        )
        self.assertIsNotNone(self.workspace._atlas_generation_signature)

    def test_failed_atlas_source_does_not_block_a_healthy_source(self) -> None:
        variants = {
            object_id: SimpleNamespace(
                object_id=object_id,
                object_name=object_id,
                resolution=512,
                texture_asset_relative_path=f"{object_id}.png",
                texture_asset_path=(
                    Path(self._temporary_directory.name) / f"{object_id}.png"
                ),
            )
            for object_id in ("retry-source", "healthy-source")
        }
        sources = {
            object_id: SimpleNamespace(object_id=object_id)
            for object_id in variants
        }
        build_attempts = {object_id: 0 for object_id in variants}

        def build_source(variant, _symmetry=None):
            object_id = variant.object_id
            build_attempts[object_id] += 1
            if object_id == "retry-source" and build_attempts[object_id] == 1:
                return None
            return sources[object_id]

        def dependency_signature(object_id: str):
            return (
                (
                    512,
                    f"{object_id}.glb",
                    ("glb",),
                    f"{object_id}.png",
                    ("png",),
                ),
            )

        self.workspace._atlas_generation_signature = None
        with (
            patch.object(
                self.workspace.generation,
                "get_generated_object_ids",
                return_value=tuple(variants),
            ),
            patch.object(
                self.workspace.generation,
                "get_active_texture_variant",
                side_effect=lambda object_id: variants[object_id],
            ),
            patch.object(
                self.workspace.generation,
                "get_texture_variant_dependency_signature",
                side_effect=dependency_signature,
            ),
            patch.object(
                self.workspace,
                "_build_atlas_object_texture_source",
                side_effect=build_source,
            ),
            patch.object(
                self.workspace.texture_atlas_workspace,
                "set_object_texture_sources",
            ) as set_sources,
            patch.object(
                self.workspace.texture_atlas_workspace,
                "refresh_texture_source_content",
                return_value=True,
            ),
        ):
            self.workspace._sync_atlas_object_texture_sources()
            self.workspace._sync_atlas_object_texture_sources()

        self.assertEqual(
            [source.object_id for source in set_sources.call_args_list[0].args[0]],
            ["healthy-source"],
        )
        self.assertEqual(
            [source.object_id for source in set_sources.call_args_list[1].args[0]],
            ["retry-source", "healthy-source"],
        )
        self.assertEqual(build_attempts["retry-source"], 2)
        self.assertEqual(build_attempts["healthy-source"], 2)
        self.assertIsNotNone(self.workspace._atlas_generation_signature)

    def test_generated_object_wins_reserved_wall_source_id_collision(
        self,
    ) -> None:
        assignment = _wall_texture_assignment(
            self.settings.path.parent / "surface_textures",
            surface_ids=(),
        )
        collision_id = build_atlas_wall_texture_source_id(
            assignment.assignment_id
        )
        self.workspace.surface_texture_generation.set_data(
            SurfaceTextureData(assignments=[assignment])
        )
        record = SimpleNamespace(object_id=collision_id)
        variant = SimpleNamespace(
            object_id=collision_id,
            object_name="Reserved name object",
            resolution=512,
            texture_asset_relative_path="object.png",
        )
        object_source = object()
        self.workspace._atlas_generation_signature = None

        with (
            patch.object(
                self.workspace.generation,
                "get_generated_object_ids",
                return_value=(record.object_id,),
            ),
            patch.object(
                self.workspace.generation,
                "get_placeable_object_names_by_id",
                return_value={collision_id: "Reserved name object"},
            ),
            patch.object(
                self.workspace.generation,
                "get_active_texture_variant",
                return_value=variant,
            ),
            patch.object(
                self.workspace,
                "_build_atlas_object_texture_source",
                return_value=object_source,
            ),
            patch.object(
                self.workspace,
                "_build_atlas_wall_texture_source",
            ) as wall_source_builder,
            patch.object(
                self.workspace.texture_atlas_workspace,
                "set_object_texture_sources",
            ) as set_sources,
            patch.object(
                self.workspace.texture_atlas_workspace,
                "remove_scene_texture_from_atlases",
            ) as remove_surface_source,
        ):
            self.workspace._sync_atlas_object_texture_sources()
            surface_source_ids = (
                self.workspace._build_atlas_surface_source_ids()
            )
            collision_is_surface = (
                self.workspace._is_atlas_surface_texture_source_id(
                    collision_id
                )
            )

        wall_source_builder.assert_not_called()
        self.assertEqual(set_sources.call_args.args[0], [object_source])
        self.assertEqual(
            set_sources.call_args.kwargs["surface_texture_entries"],
            [],
        )
        self.assertFalse(collision_is_surface)
        remove_surface_source.assert_not_called()
        self.assertEqual(surface_source_ids, {})
        self.assertIn(
            "generated object uses the same reserved Atlas ID",
            self.workspace.texture_atlas_workspace.status_label.text(),
        )

    def test_surface_cleanup_never_removes_colliding_generated_object(
        self,
    ) -> None:
        assignment = _wall_texture_assignment(
            self.settings.path.parent / "surface_textures",
            assignment_id="colliding-cleanup",
            surface_ids=("level:2/room:1/wall:1:2",),
        )
        collision_id = build_atlas_wall_texture_source_id(
            assignment.assignment_id
        )
        surface_workspace = self.workspace.surface_texture_generation
        surface_workspace.set_data(
            SurfaceTextureData(assignments=[assignment])
        )

        with patch.object(
            self.workspace.generation,
            "get_placeable_object_names_by_id",
            return_value={collision_id: "Colliding object"},
        ):
            self.assertEqual(
                self.workspace._atlas_surface_sources_displaced_by(
                    "replacement",
                    assignment.surface_ids,
                ),
                (),
            )
            surface_workspace.set_data(SurfaceTextureData())
            with (
                patch.object(
                    self.workspace.texture_atlas_workspace,
                    "remove_deleted_wall_texture_assignments",
                ) as remove_assignments,
                patch.object(
                    self.workspace,
                    "_sync_atlas_object_texture_sources",
                ),
            ):
                remove_handler = (
                    self.workspace
                    ._handle_surface_texture_assignments_removed_for_atlases
                )
                remove_handler((assignment.assignment_id,))

        remove_assignments.assert_not_called()

    def test_pinned_atlas_resolution_uses_png_only_exact_resolver(self) -> None:
        record = SimpleNamespace(object_id="chair")
        active = SimpleNamespace(
            object_id="chair",
            object_name="Chair",
            resolution=1024,
            texture_asset_relative_path="chair-1024.png",
        )
        pinned_png = SimpleNamespace(
            object_id="chair",
            object_name="Chair",
            resolution=2048,
            texture_asset_relative_path="chair-2048.png",
        )
        self.workspace._atlas_generation_signature = None

        with patch.object(
            self.workspace.generation,
            "get_generated_object_ids",
            return_value=(record.object_id,),
        ), patch.object(
            self.workspace.generation,
            "get_active_texture_variant",
            return_value=active,
        ), patch.object(
            self.workspace.generation,
            "get_texture_image_variant",
            return_value=pinned_png,
        ) as image_resolver, patch.object(
            self.workspace.generation,
            "get_texture_variant",
            return_value=None,
        ) as complete_variant_resolver, patch.object(
            self.workspace,
            "_build_atlas_object_texture_source",
            side_effect=lambda variant, _symmetry=None: variant,
        ), patch.object(
            self.workspace.texture_atlas_workspace,
            "set_object_texture_sources",
        ) as set_sources:
            self.workspace._sync_atlas_object_texture_sources()
            image_resolver.reset_mock()
            resolver = set_sources.call_args.kwargs["variant_resolver"]
            selectability_resolver = set_sources.call_args.kwargs[
                "selectability_resolver"
            ]
            resolved_pinned = resolver("chair", 2048)
            is_selectable = selectability_resolver("chair", 2048)

        self.assertIs(resolved_pinned, pinned_png)
        self.assertFalse(is_selectable)
        image_resolver.assert_called_once_with("chair", 2048)
        complete_variant_resolver.assert_called_once_with("chair", 2048)

    def test_unrelated_object_variant_stays_selectable_during_generation(
        self,
    ) -> None:
        active = SimpleNamespace(
            object_id="chair",
            object_name="Chair",
            resolution=1024,
            texture_asset_relative_path="chair-1024.png",
        )
        glb_path = Path(self._temporary_directory.name) / "chair-2048.glb"
        glb_path.write_bytes(b"test glb")
        complete_variant = SimpleNamespace(glb_asset_path=glb_path)
        self.workspace._atlas_generation_signature = None

        with (
            patch.object(
                self.workspace.generation,
                "get_generated_object_ids",
                return_value=("chair",),
            ),
            patch.object(
                self.workspace.generation,
                "get_active_texture_variant",
                return_value=active,
            ),
            patch.object(
                self.workspace.generation,
                "has_active_object_job",
                side_effect=lambda object_id: object_id == "busy-table",
            ) as has_active_object_job,
            patch.object(
                self.workspace.generation,
                "get_texture_variant",
                return_value=complete_variant,
            ) as complete_variant_resolver,
            patch.object(
                self.workspace,
                "_build_atlas_object_texture_source",
                side_effect=lambda variant, _symmetry=None: variant,
            ),
            patch(
                "housemaker.main.import_generated_glb",
                return_value=_generated_box_model(),
            ),
            patch.object(
                self.workspace.texture_atlas_workspace,
                "set_object_texture_sources",
            ) as set_sources,
        ):
            self.workspace._sync_atlas_object_texture_sources()
            selectability_resolver = set_sources.call_args.kwargs[
                "selectability_resolver"
            ]

            self.assertTrue(selectability_resolver("chair", 2048))
            self.assertFalse(
                selectability_resolver("busy-table", 2048)
            )

        self.assertEqual(
            has_active_object_job.call_args_list,
            [call("chair"), call("busy-table")],
        )
        complete_variant_resolver.assert_called_once_with("chair", 2048)

    def test_explicit_generated_object_deletion_is_routed_to_atlas(self) -> None:
        self.workspace.atlas_object_preview_viewer.set_model(
            _generated_box_model()
        )
        self.workspace._atlas_preview_variant_key = (
            "chair",
            1024,
            "chair.glb",
            1,
            1,
        )
        with patch.object(
            self.workspace.texture_atlas_workspace,
            "remove_deleted_object",
            return_value=1,
        ) as remove_deleted_object:
            self.workspace.generation.generated_object_deleted.emit("chair")
            _qt_application.processEvents()

        remove_deleted_object.assert_called_once_with("chair")
        self.assertIsNone(self.workspace.atlas_object_preview_viewer.model)
        self.assertIsNone(self.workspace._atlas_preview_variant_key)

    def test_atlas_permanent_object_delete_can_be_cancelled_then_confirmed(
        self,
    ) -> None:
        generation = self.workspace.generation
        record = _generated_object_record_with_variants(
            generation._asset_directory,
            object_id="delete-chair",
            object_name="Delete chair",
            resolutions=(512,),
            selected_resolution=512,
        )
        generation_data = GenerationData(generated_objects=[record])
        generation.set_data(generation_data)
        generation.data_changed.emit(generation_data)
        atlas_workspace = self.workspace.texture_atlas_workspace
        self.workspace.workspace_tabs.setCurrentWidget(atlas_workspace)
        _qt_application.processEvents()
        atlas_workspace.object_list.object_clicked.emit(
            record.object_id,
            Qt.MouseButton.LeftButton,
        )
        _qt_application.processEvents()

        self.assertTrue(atlas_workspace.delete_object_button.isEnabled())
        with patch(
            "housemaker.main.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Cancel,
        ) as question:
            atlas_workspace.delete_object_button.click()

        question.assert_called_once()
        self.assertIs(
            question.call_args.args[0],
            atlas_workspace,
        )
        self.assertEqual(generation.get_generated_object_ids(), (record.object_id,))
        self.assertTrue(atlas_workspace.delete_object_button.isEnabled())

        with patch(
            "housemaker.main.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Yes,
        ) as question:
            atlas_workspace.delete_object_button.click()
            _qt_application.processEvents()

        question.assert_called_once()
        self.assertEqual(generation.get_generated_object_ids(), ())
        self.assertFalse(atlas_workspace.delete_object_button.isEnabled())
        self.assertEqual(
            [
                atlas_workspace.object_list.item(row).data(
                    Qt.ItemDataRole.UserRole
                )
                for row in range(atlas_workspace.object_list.count())
            ],
            [],
        )
        self.assertIn("Deleted generated object", atlas_workspace.status_label.text())

    def test_changed_object_is_routed_to_atlas_path_refresh_once(self) -> None:
        changed_record = SimpleNamespace(object_id="chair")

        with patch.object(
            self.workspace.texture_atlas_workspace,
            "refresh_regenerated_object_texture",
            return_value=1,
        ) as refresh_regenerated:
            self.workspace.generation.generated_object_changed.emit(
                changed_record,
                _generated_box_model(),
            )
            self.workspace.generation.texture_regeneration_completed.emit(
                changed_record,
                _generated_box_model(),
            )
            _qt_application.processEvents()

        refresh_regenerated.assert_called_once_with("chair")

    def test_generated_objects_and_textures_start_atlas_list_highlights(self) -> None:
        generated_record = SimpleNamespace(object_id="chair")

        with patch.object(
            self.workspace.texture_atlas_workspace,
            "mark_sources_new",
        ) as mark_sources_new:
            self.workspace.generation.generation_completed.emit(
                generated_record,
                _generated_box_model(),
            )
            self.workspace.generation.texture_regeneration_completed.emit(
                generated_record,
                _generated_box_model(),
            )
            _qt_application.processEvents()

        self.assertEqual(
            mark_sources_new.call_args_list,
            [call(("chair",)), call(("chair",))],
        )

    def test_generated_surface_texture_starts_atlas_list_highlight(self) -> None:
        assignment = SimpleNamespace(assignment_id="brick-wall")

        with patch.object(
            self.workspace.texture_atlas_workspace,
            "mark_sources_new",
        ) as mark_sources_new:
            self.workspace.surface_texture_generation.generation_completed.emit(
                assignment
            )
            _qt_application.processEvents()

        mark_sources_new.assert_called_once_with(
            (build_atlas_wall_texture_source_id("brick-wall"),)
        )

    def test_object_generation_exposes_no_texture_inpaint_signal(self) -> None:
        self.assertFalse(
            hasattr(self.workspace.generation, "texture_inpaint_completed")
        )
        self.assertFalse(
            hasattr(self.workspace.generation, "face_purge_completed")
        )

    def test_invalid_object_change_does_not_refresh_atlas(self) -> None:
        with patch.object(
            self.workspace.texture_atlas_workspace,
            "refresh_regenerated_object_texture",
        ) as refresh_regenerated:
            self.workspace.generation.generated_object_changed.emit(
                object(),
                _generated_box_model(),
            )
            _qt_application.processEvents()

        refresh_regenerated.assert_not_called()

    def test_preview_error_does_not_hide_a_capacity_block_message(self) -> None:
        status = self.workspace.texture_atlas_workspace.status_label
        status.setText(
            "Texture size change blocked by atlas capacity. Keeping 1024 x 1024."
        )

        with patch.object(
            self.workspace.generation,
            "get_texture_variant",
            return_value=None,
        ):
            self.workspace._handle_atlas_object_preview_requested(
                "chair",
                1024,
            )

        self.assertIn("blocked by atlas capacity", status.text())
        self.assertIn("3D texture variant is missing", status.text())

    def test_atlas_resolution_change_selects_the_global_object_variant(
        self,
    ) -> None:
        with patch.object(
            self.workspace.generation,
            "select_object_texture_resolution",
            return_value=True,
        ) as select_resolution:
            self.workspace.texture_atlas_workspace \
                .object_texture_resolution_changed.emit("chair", 2048)
            _qt_application.processEvents()

        select_resolution.assert_called_once_with("chair", 2048)

    def test_corrupt_target_glb_blocks_atlas_and_global_resolution_change(
        self,
    ) -> None:
        asset_root = Path(self._temporary_directory.name)
        active_png = asset_root / "chair-512.png"
        target_png = asset_root / "chair-1024.png"
        corrupt_glb = asset_root / "chair-1024.glb"
        _write_texture_png(active_png, 512, (20, 40, 60, 255))
        _write_texture_png(target_png, 1024, (60, 40, 20, 255))
        corrupt_glb.write_bytes(b"not a GLB")
        active_variant = SimpleNamespace(
            object_id="chair",
            object_name="Chair",
            resolution=512,
            texture_asset_relative_path=active_png.name,
            texture_asset_path=active_png,
        )
        target_image_variant = SimpleNamespace(
            object_id="chair",
            object_name="Chair",
            resolution=1024,
            texture_asset_relative_path=target_png.name,
            texture_asset_path=target_png,
        )
        target_complete_variant = SimpleNamespace(
            glb_asset_path=corrupt_glb,
        )
        atlas_data = TextureAtlasData()
        atlas = atlas_data.create_atlas(
            "Corrupt target test",
            2048,
            atlas_id="atlas-a",
        )
        atlas_data.assign_object(
            atlas.atlas_id,
            "chair",
            active_png.name,
            512,
        )
        self.workspace.texture_atlas_workspace.set_data(atlas_data)
        self.workspace._atlas_generation_signature = None

        with (
            patch.object(
                self.workspace.generation,
                "get_generated_object_ids",
                return_value=("chair",),
            ),
            patch.object(
                self.workspace.generation,
                "get_active_texture_variant",
                return_value=active_variant,
            ),
            patch.object(
                self.workspace.generation,
                "get_texture_image_variant",
                return_value=target_image_variant,
            ),
            patch.object(
                self.workspace.generation,
                "get_texture_variant",
                return_value=target_complete_variant,
            ),
            patch.object(
                self.workspace.generation,
                "select_object_texture_resolution",
                return_value=True,
            ) as select_resolution,
        ):
            self.workspace._sync_atlas_object_texture_sources()
            changed = self.workspace.texture_atlas_workspace \
                ._cycle_object_texture_resolution("chair", 1)

        self.assertFalse(changed)
        retained_atlas = self.workspace.texture_atlas_workspace.selected_atlas
        self.assertIsNotNone(retained_atlas)
        assert retained_atlas is not None
        retained = retained_atlas.placement_for_object("chair")
        self.assertIsNotNone(retained)
        assert retained is not None
        self.assertEqual(retained.texture_resolution, 512)
        self.assertEqual(retained.texture_path, active_png.name)
        select_resolution.assert_not_called()
        self.assertIn(
            "3D texture variant",
            self.workspace.texture_atlas_workspace.status_label.text(),
        )


# ### Test helpers ###
def _generated_box_model() -> GeneratedModel:
    mesh = trimesh.creation.box()
    return GeneratedModel(
        mesh=mesh,
        scene=trimesh.Scene(mesh),
        glb_bytes=b"",
    )


def _generated_textured_surface_model(surface_id: str) -> GeneratedModel:
    mesh = trimesh.Trimesh(
        vertices=np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=float,
        ),
        faces=np.asarray(((0, 1, 2),), dtype=np.int64),
        process=False,
        metadata={"housemaker_surface_id": surface_id},
    )
    mesh.visual = TextureVisuals(
        uv=np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        material=PBRMaterial(
            name=f"Surface {surface_id}",
            baseColorTexture=Image.new("RGBA", (4, 4), (80, 90, 100, 255)),
        ),
    )
    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name="surface")
    return GeneratedModel(mesh=mesh, scene=scene, glb_bytes=b"")


def _write_texture_png(
    path: Path,
    resolution: int,
    color: tuple[int, int, int, int],
) -> None:
    Image.new("RGBA", (resolution, resolution), color).save(path, format="PNG")


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
