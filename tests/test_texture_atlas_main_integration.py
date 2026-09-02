# ### Environment setup ###
from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
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
from PySide6.QtWidgets import QApplication
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.glb import GeneratedModel
from housemaker.generation_state import (
    GeneratedObjectPlacement,
    GeneratedObjectRecord,
    GenerationData,
)
from housemaker.main import BlueprintWorkspace
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
from housemaker.surface_texture_state import (
    SURFACE_TYPE_FLOOR,
    SURFACE_TYPE_WALL,
    SURFACE_TEXTURE_RESOLUTIONS,
    SurfaceTextureAssignment,
    SurfaceTextureData,
    SurfaceTextureVariant,
)
from housemaker.surface_geometry import build_fixed_surfaces
from housemaker.texture_atlas_state import (
    ATLAS_PACKING_MODE_SYMMETRIC_HALF,
    ATLAS_PACKING_MODE_SYMMETRIC_PAIR,
    ATLAS_PACKING_MODE_SYMMETRIC_QUARTER,
    ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR,
    TextureAtlasData,
)
from housemaker.texture_atlas_workspace import (
    OBJECT_SCENE_REQUIRED_UNPACKED_COLOR,
    OBJECT_SCENE_REQUIRED_UNPACKED_ROLE,
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
    level.floor_contour_vertex_ids = boundary_ids
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

    def test_atlas_is_after_canvas_and_keeps_local_preview_layout_free(self) -> None:
        names = [
            self.workspace.workspace_tabs.tabText(index)
            for index in range(self.workspace.workspace_tabs.count())
        ]

        self.assertEqual(names[:2], ["Canvas", "Atlas"])
        self.workspace.workspace_tabs.setCurrentWidget(
            self.workspace.texture_atlas_workspace
        )
        _qt_application.processEvents()
        self.assertFalse(self.workspace.side_panel.isVisible())
        self.assertIs(
            self.workspace._active_workspace_3d_viewer(),
            self.workspace.atlas_object_preview_viewer,
        )
        self.assertEqual(
            self.workspace.atlas_object_preview_viewer
            .get_ambient_light_intensity(),
            1.0,
        )
        self.assertTrue(self.workspace.atlas_object_preview_viewer.isHidden())
        self.assertEqual(
            self.workspace.texture_atlas_workspace.layout().indexOf(
                self.workspace.atlas_object_preview_viewer
            ),
            -1,
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

    def test_generated_wall_texture_can_be_selected_and_added_to_atlas(
        self,
    ) -> None:
        assignment = _wall_texture_assignment(
            self.settings.path.parent / "surface_textures"
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
            self.workspace.texture_atlas_workspace.object_list.item(index)
            for index in range(
                self.workspace.texture_atlas_workspace.object_list.count()
            )
        ]
        source_item = next(
            item
            for item in source_items
            if item.data(Qt.ItemDataRole.UserRole) == source_id
        )
        self.assertIn("Wall texture", source_item.text())
        self.workspace.texture_atlas_workspace.object_list.setCurrentItem(
            source_item
        )
        self.workspace.texture_atlas_workspace.assign_object_button.click()

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

    def test_required_texture_stays_unpacked_and_red_when_atlas_is_full(
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
        self.assertTrue(
            source_item.data(OBJECT_SCENE_REQUIRED_UNPACKED_ROLE)
        )
        self.assertEqual(
            source_item.foreground().color(),
            OBJECT_SCENE_REQUIRED_UNPACKED_COLOR,
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
            self.workspace.texture_atlas_workspace.object_list.item(index)
            for index in range(
                self.workspace.texture_atlas_workspace.object_list.count()
            )
        ]
        source_item = next(
            item
            for item in source_items
            if item.data(Qt.ItemDataRole.UserRole) == source_id
        )

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
            self.workspace.texture_atlas_workspace.object_list.item(index)
            for index in range(
                self.workspace.texture_atlas_workspace.object_list.count()
            )
            if self.workspace.texture_atlas_workspace.object_list.item(
                index
            ).data(Qt.ItemDataRole.UserRole)
            == source_id
        )
        self.workspace.texture_atlas_workspace.object_list.setCurrentItem(
            source_item
        )
        self.workspace.texture_atlas_workspace.assign_object_button.click()

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
        self.assertEqual(atlas_materials, ["HouseMaker Atlas architecture"])

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

    def test_wall_wheel_resize_selects_surface_resolution_globally(
        self,
    ) -> None:
        assignment = _wall_texture_assignment_with_variants(
            self.settings.path.parent / "surface_textures"
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
            atlas_workspace.object_list.item(index)
            for index in range(atlas_workspace.object_list.count())
            if atlas_workspace.object_list.item(index).data(
                Qt.ItemDataRole.UserRole
            )
            == source_id
        )
        wheel = _wheel_event(
            QPointF(atlas_workspace.object_list.visualItemRect(source_item).center()),
            120,
        )

        atlas_workspace.object_list.wheelEvent(wheel)
        _qt_application.processEvents()

        self.assertTrue(wheel.isAccepted())
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
        assignment = _wall_texture_assignment_with_variants(
            self.settings.path.parent / "surface_textures"
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
            changed = surface_workspace._request_global_texture_resolution_change(
                assignment.assignment_id,
                1024,
            )
        _qt_application.processEvents()

        self.assertTrue(changed)
        self.assertEqual(select_resolution.call_count, 1)
        self.assertEqual(resolution_signals, [])
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

        changed = surface_workspace._request_global_texture_resolution_change(
            assignment.assignment_id,
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
        item = self.workspace.texture_atlas_workspace.object_list.item(0)
        self.assertIn("1 surface", item.text())

    def test_project_load_materializes_persisted_wall_texture_placement(
        self,
    ) -> None:
        assignment = _wall_texture_assignment(
            self.settings.path.parent / "surface_textures"
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
            levels=create_default_levels(),
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
        generation_data = SimpleNamespace(generated_objects=[record])
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
            self.settings.path.parent / "surface_textures"
        )
        collision_id = build_atlas_wall_texture_source_id(
            assignment.assignment_id
        )
        self.workspace.surface_texture_generation.set_data(
            SurfaceTextureData(assignments=[assignment])
        )
        record = SimpleNamespace(object_id=collision_id)
        generation_data = SimpleNamespace(generated_objects=[record])
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
        ):
            self.workspace._sync_atlas_object_texture_sources()
            surface_source_ids = (
                self.workspace._build_atlas_surface_source_ids()
            )

        wall_source_builder.assert_not_called()
        self.assertEqual(set_sources.call_args.args[0], [object_source])
        self.assertEqual(surface_source_ids, {})
        self.assertIn(
            "generated object uses the same reserved Atlas ID",
            self.workspace.texture_atlas_workspace.status_label.text(),
        )

    def test_pinned_atlas_resolution_uses_png_only_exact_resolver(self) -> None:
        record = SimpleNamespace(object_id="chair")
        generation_data = SimpleNamespace(generated_objects=[record])
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
        generation_data = SimpleNamespace(
            generated_objects=[SimpleNamespace(object_id="chair")]
        )
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
