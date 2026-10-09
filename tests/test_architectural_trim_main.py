# ### Environment setup ###
from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.architectural_trim import (
    TRIM_HANDLE_HEIGHT,
    ArchitecturalTrimPlacementRequest,
)
from housemaker.generation_state import MESHY_GENERATION_PROVIDER
from housemaker.generation_workspace import OBJECT_OPERATION_GENERATE_TEXTURE
from housemaker.glb import (
    GeneratedModel,
    build_placed_generated_model_transform,
    convert_to_glb,
    import_generated_glb,
)
from housemaker.main import BlueprintWorkspace
from housemaker.meshy_generation import MeshyGenerationResult
from housemaker.models import (
    TRIM_KIND_CORNICE,
    TRIM_KIND_SKIRTING_BOARD,
    LevelData,
    RoomData,
    VertexData,
)
from housemaker.pbr_maps import ATLAS_MAP_BASE_COLOR, PBR_MAP_NORMAL
from housemaker.surface_geometry import build_fixed_surfaces
from housemaker.texture_atlas_state import TextureAtlasData
from housemaker.viewer import ArchitecturalTrimDimensionEdit

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_square_level() -> LevelData:
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
    return LevelData(
        index=2,
        name="Ground",
        vertex_data=vertex_data,
        rooms=[
            RoomData(
                name="Room",
                vertex_ids=boundary_ids,
                center_vertex_id=center.id,
                color_rgb=(140, 180, 220),
            )
        ],
    )


def _wall_surface_id(start_id: int, end_id: int) -> str:
    return f"level:2/room:5/wall:{start_id}:{end_id}"


def _build_authored_uv_glb(model: GeneratedModel) -> bytes:
    """Attach a distinctive authored UV layout to every fixture primitive."""

    scene = import_generated_glb(model.glb_bytes).scene
    for geometry_index, geometry in enumerate(scene.geometry.values(), start=1):
        vertex_count = len(geometry.vertices)
        uv = np.column_stack(
            (
                np.linspace(0.07, 0.83, vertex_count),
                np.mod(
                    np.arange(vertex_count, dtype=float) * 0.37,
                    0.8,
                )
                + 0.1,
            )
        )
        geometry.visual = TextureVisuals(
            uv=uv,
            material=PBRMaterial(
                name=f"authored_cornice_{geometry_index}",
                baseColorTexture=Image.new(
                    "RGBA",
                    (2, 2),
                    (80, 120, 170, 255),
                ),
            ),
        )
    exported = scene.export(file_type="glb")
    if not isinstance(exported, bytes):
        raise TypeError("The authored-UV cornice fixture could not be serialized.")
    return exported


# ### Main integration tests ###
class ArchitecturalTrimMainTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(
                Path(self._temporary_directory.name) / "settings.json"
            )
        )
        self.level = _build_square_level()
        self.workspace.levels = [self.level]
        self.workspace.current_level_index = 0
        self.workspace.surface_texture_generation.set_levels([self.level])
        self.workspace._reset_viewer_doorway_snapshots()
        self.workspace._set_canvas_viewer_targets(build_fixed_surfaces([self.level]))

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self._temporary_directory.cleanup()

    def _place_skirting(self, trim_id: str, wall_surface_id: str) -> None:
        self.workspace._handle_architectural_trim_placement_requested(
            ArchitecturalTrimPlacementRequest(
                trim_id=trim_id,
                kind=TRIM_KIND_SKIRTING_BOARD,
                wall_surface_ids=(wall_surface_id,),
            )
        )

    def _place_cornice(self, trim_id: str, wall_surface_id: str) -> None:
        self.workspace._handle_architectural_trim_placement_requested(
            ArchitecturalTrimPlacementRequest(
                trim_id=trim_id,
                kind=TRIM_KIND_CORNICE,
                wall_surface_ids=(wall_surface_id,),
            )
        )

    def _seed_stale_atlas_object(self) -> str:
        mesh = trimesh.creation.box(extents=(0.4, 0.5, 0.6))
        glb_bytes = bytes(trimesh.Scene(mesh).export(file_type="glb"))
        model = import_generated_glb(glb_bytes)
        self.workspace.generation._handle_generation_succeeded(
            MeshyGenerationResult(
                "stale-atlas-object-task",
                glb_bytes,
                "Atlas object",
            ),
            model,
        )
        return self.workspace.generation.get_data().generated_objects[-1].object_id

    def _install_trim_texture_family(
        self,
        object_id: str,
        *,
        color: tuple[int, int, int, int],
        task_id: str,
    ):
        """Persist one exact object-texture family on a procedural trim."""

        generation = self.workspace.generation
        records = generation._data.generated_objects
        record_index = next(
            index
            for index, record in enumerate(records)
            if record.object_id == object_id
        )
        record = records[record_index]
        model = generation.get_generated_object_model(object_id)
        self.assertIsNotNone(model)
        assert model is not None
        authored_uv_glb = _build_authored_uv_glb(model)
        authored_uv_model = import_generated_glb(authored_uv_glb)

        variants: dict[str, dict[str, object]] = {}
        for resolution in (512, 1024, 2048):
            stem = f"{object_id}-{task_id}-{resolution}"
            glb_name = f"{stem}.glb"
            base_color_name = f"{stem}.png"
            normal_name = f"{stem}.normal.png"
            (generation._asset_directory / glb_name).write_bytes(authored_uv_glb)
            Image.new("RGBA", (resolution, resolution), color).save(
                generation._asset_directory / base_color_name,
                format="PNG",
            )
            Image.new("RGBA", (resolution, resolution), (128, 128, 255, 255)).save(
                generation._asset_directory / normal_name,
                format="PNG",
            )
            variants[str(resolution)] = {
                "glb_asset_path": glb_name,
                "texture_asset_path": base_color_name,
                "map_texture_asset_paths": {
                    ATLAS_MAP_BASE_COLOR: base_color_name,
                    PBR_MAP_NORMAL: normal_name,
                },
            }

        updated_pipeline = dict(record.pipeline)
        updated_pipeline["texture_variants"] = variants
        updated_pipeline["selected_texture_resolution"] = 512
        updated_record = replace(
            record,
            pipeline=updated_pipeline,
            provider=MESHY_GENERATION_PROVIDER,
            provider_task_id=task_id,
            asset_path=str(variants["512"]["glb_asset_path"]),
        )
        records[record_index] = updated_record
        return updated_record, authored_uv_model

    def test_placement_selects_semantic_parts_and_undo_removes_component(
        self,
    ) -> None:
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self._place_skirting("1" * 32, _wall_surface_id(1, 2))

        self.assertEqual(len(self.level.architectural_trims), 1)
        selected_ids = self.workspace._desired_canvas_architectural_trim_part_ids
        self.assertEqual(len(selected_ids), 1)
        self.assertIn("/part:front:wall", selected_ids[0])
        self.assertEqual(
            self.workspace.viewer.get_selected_architectural_trim_part_ids(),
            selected_ids,
        )
        self.assertEqual(
            self.workspace._atlas_surface_assignment_target_ids,
            selected_ids,
        )

        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self.workspace._handle_canvas_undo_requested()

        self.assertEqual(self.level.architectural_trims, [])

    def test_selecting_cornice_replaces_stale_atlas_generation_target(
        self,
    ) -> None:
        trim_id = "c" * 32
        stale_object_id = self._seed_stale_atlas_object()
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self._place_cornice(trim_id, _wall_surface_id(1, 2))
        semantic_id = next(
            part.semantic_id
            for part in self.workspace._canvas_architectural_trim_parts_by_id.values()
            if part.trim_id == trim_id and part.part_kind == "front"
        )

        self.assertTrue(
            self.workspace.generation.select_generated_object(stale_object_id)
        )
        self.assertEqual(
            self.workspace.generation._selected_object_id,
            stale_object_id,
        )

        self.workspace._handle_architectural_trim_part_selection_changed(
            (semantic_id,)
        )

        target = self.workspace.generation._architectural_trim_editing_target
        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(target.level_index, self.level.index)
        self.assertEqual(target.trim_id, trim_id)
        self.assertEqual(target.trim_kind, TRIM_KIND_CORNICE)
        self.assertNotEqual(target.object_id, stale_object_id)
        self.assertEqual(len(target.fit_dimensions), 3)
        self.assertTrue(all(value > 0.0 for value in target.fit_dimensions))
        self.assertEqual(
            self.workspace.generation._selected_object_id,
            target.object_id,
        )
        trim_model = self.workspace.generation.get_generated_object_model(
            target.object_id
        )
        self.assertIsNotNone(trim_model)
        self.assertIs(self.workspace.generation.result_view.model, trim_model)

    def test_selected_cornice_uses_object_retexture_only(self) -> None:
        settings = replace(
            self.workspace.generation.get_runtime_settings(),
            meshy_api_key="meshy-test-key",
        )
        self.workspace.generation.set_runtime_settings(settings)
        self.workspace.surface_texture_generation.set_runtime_settings(settings)
        reference = np.full((12, 16, 4), 255, dtype=np.uint8)
        reference[:, :, :3] = (25, 90, 180)
        self.workspace.generation.set_temporary_object_reference(reference)
        self.workspace.surface_texture_generation.set_temporary_reference(
            reference
        )

        trim_id = "f" * 32
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self._place_cornice(trim_id, _wall_surface_id(1, 2))
        semantic_id = next(
            part.semantic_id
            for part in self.workspace._canvas_architectural_trim_parts_by_id.values()
            if part.trim_id == trim_id and part.part_kind == "front"
        )

        target = self.workspace.generation._architectural_trim_editing_target
        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(
            self.workspace.generation._selected_object_id,
            target.object_id,
        )
        self.assertEqual(
            self.workspace.surface_texture_generation.get_selected_surface_ids(),
            (semantic_id,),
        )
        self.assertFalse(self.workspace.generation.generate_button.isEnabled())
        self.assertFalse(
            self.workspace.generation.generate_geometry_button.isEnabled()
        )
        self.assertTrue(
            self.workspace.generation.generate_texture_button.isEnabled()
        )
        self.assertIs(
            self.workspace.merged_generation_workspace
            .generate_texture_stack.currentWidget(),
            self.workspace.generation.generate_texture_button,
        )

    def test_selected_skirting_uses_surface_texture_pipeline(self) -> None:
        trim_id = "e" * 32
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self._place_skirting(trim_id, _wall_surface_id(1, 2))

        self.assertFalse(self.workspace.generation.generate_button.isEnabled())
        self.assertFalse(
            self.workspace.generation.generate_geometry_button.isEnabled()
        )
        self.assertIs(
            self.workspace.merged_generation_workspace
            .generate_texture_stack.currentWidget(),
            self.workspace.surface_texture_generation.generate_button,
        )

    def test_generated_cornice_surface_texture_is_listed_and_assignable(
        self,
    ) -> None:
        first_trim_id = "a" * 32
        second_trim_id = "b" * 32
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self._place_cornice(first_trim_id, _wall_surface_id(1, 2))
            first_target = self.workspace.generation._architectural_trim_editing_target
            self.assertIsNotNone(first_target)
            assert first_target is not None
            source_object_id = first_target.object_id
            self._place_cornice(second_trim_id, _wall_surface_id(2, 3))
            second_target = self.workspace.generation._architectural_trim_editing_target
            self.assertIsNotNone(second_target)
            assert second_target is not None
            target_object_id = second_target.object_id
        front_ids = {
            part.trim_id: part.semantic_id
            for part in self.workspace._canvas_architectural_trim_parts_by_id.values()
            if part.part_kind == "front"
        }
        source_surface_id = front_ids[first_trim_id]
        target_surface_id = front_ids[second_trim_id]
        target_trim_surface_ids_by_kind = {
            part.part_kind: part.semantic_id
            for part in self.workspace._canvas_architectural_trim_parts_by_id.values()
            if part.trim_id == second_trim_id
        }
        self.assertEqual(
            set(target_trim_surface_ids_by_kind),
            {"bottom", "front", "sides", "top"},
        )
        target_trim_surface_ids = set(target_trim_surface_ids_by_kind.values())
        atlas_data = TextureAtlasData()
        atlas_data.create_atlas("Cornices", 2048, atlas_id="cornices")
        self.workspace.texture_atlas_workspace.set_data(atlas_data)

        first_color = (170, 80, 30, 255)
        generated_record, generated_model = self._install_trim_texture_family(
            source_object_id,
            color=first_color,
            task_id="cornice-texture-1",
        )
        self.workspace.generation.texture_regeneration_completed.emit(
            generated_record,
            generated_model,
        )
        self.workspace.generation.generated_object_changed.emit(
            generated_record,
            generated_model,
        )
        _qt_application.processEvents()

        linked_assignments = tuple(
            assignment
            for assignment in (
                self.workspace.surface_texture_generation.get_assignments()
            )
            if assignment.source_object_id == source_object_id
        )
        self.assertEqual(len(linked_assignments), 1)
        assignment = linked_assignments[0]
        self.assertIn(source_surface_id, assignment.surface_ids)
        source_trim_surface_ids = set(assignment.surface_ids)
        self.assertEqual(
            tuple(variant.resolution for variant in assignment.texture_variants),
            (512, 1024, 2048),
        )
        surface_asset_directory = (
            self.workspace.surface_texture_generation._asset_directory
        )
        copied_paths = {
            relative_path
            for variant in assignment.texture_variants
            for relative_path in variant.map_asset_paths.values()
        }
        self.assertEqual(len(copied_paths), 6)
        self.assertTrue(
            all((surface_asset_directory / path).is_file() for path in copied_paths)
        )
        self.assertTrue(
            all(
                not (self.workspace.generation._asset_directory / path).exists()
                for path in copied_paths
            )
        )

        self.assertIn(
            source_object_id,
            self.workspace.texture_atlas_workspace
            ._surface_texture_entries_by_id,
        )
        listed_surface_ids = tuple(
            self.workspace.texture_atlas_workspace.surface_list.item(row).data(
                Qt.ItemDataRole.UserRole
            )
            for row in range(
                self.workspace.texture_atlas_workspace.surface_list.count()
            )
        )
        listed_object_ids = tuple(
            self.workspace.texture_atlas_workspace.object_list.item(row).data(
                Qt.ItemDataRole.UserRole
            )
            for row in range(
                self.workspace.texture_atlas_workspace.object_list.count()
            )
        )
        self.assertIn(source_object_id, listed_surface_ids)
        self.assertNotIn(source_object_id, listed_object_ids)

        source_model = self.workspace.generation.get_generated_object_model(
            source_object_id
        )
        self.assertIsNotNone(source_model)
        assert source_model is not None
        single_trim_models, single_trim_keys = (
            self.workspace._build_active_architectural_trim_component_models(
                include_excluded_levels=True,
            )
        )
        self.assertEqual(
            single_trim_keys,
            ((2, first_trim_id), (2, second_trim_id)),
        )
        self.assertEqual(len(single_trim_models), 2)
        source_corner_placement = next(
            placement
            for placement in single_trim_models
            if placement.object_id == source_object_id
        )
        companion_corner_placement = next(
            placement
            for placement in single_trim_models
            if placement.object_id == target_object_id
        )
        self.assertEqual(
            source_corner_placement.atlas_source_object_id,
            source_object_id,
        )
        self.assertEqual(
            companion_corner_placement.atlas_source_object_id,
            target_object_id,
        )
        self.assertNotIn(target_object_id, listed_surface_ids)
        self.assertNotIn(target_object_id, listed_object_ids)

        world_corner_meshes: list[trimesh.Trimesh] = []
        for placement in single_trim_models:
            world_mesh = placement.model.mesh.copy()
            world_mesh.apply_transform(
                build_placed_generated_model_transform(placement)
            )
            world_corner_meshes.append(world_mesh)
        vertex_blocks: list[np.ndarray] = []
        face_blocks: list[np.ndarray] = []
        vertex_offset = 0
        for world_mesh in world_corner_meshes:
            vertex_blocks.append(np.asarray(world_mesh.vertices, dtype=float))
            face_blocks.append(
                np.asarray(world_mesh.faces, dtype=np.int64) + vertex_offset
            )
            vertex_offset += len(world_mesh.vertices)
        joined_corner = trimesh.Trimesh(
            vertices=np.vstack(vertex_blocks),
            faces=np.vstack(face_blocks),
            process=True,
        )
        self.assertTrue(joined_corner.is_watertight)
        self.assertTrue(joined_corner.is_winding_consistent)

        atlas_workspace = self.workspace.texture_atlas_workspace
        if not atlas_workspace.is_source_assigned_to_any_atlas(source_object_id):
            self.assertTrue(
                atlas_workspace.assign_source_to_selected_atlas(source_object_id)
            )
        atlas = atlas_workspace.get_data().atlas_by_id("cornices")
        self.assertIsNotNone(atlas)
        assert atlas is not None
        self.assertIsNotNone(atlas.placement_for_object(source_object_id))
        first_atlas_path = atlas_workspace._resolve_owned_atlas_path(
            atlas.image_path
        )
        self.assertIsNotNone(first_atlas_path)
        assert first_atlas_path is not None
        first_atlas_png = first_atlas_path.read_bytes()

        self.workspace._handle_architectural_trim_part_selection_changed(
            (target_surface_id,)
        )
        self.assertEqual(
            self.workspace._atlas_surface_assignment_target_ids,
            (target_surface_id,),
        )

        self.workspace._handle_atlas_surface_assign_requested(source_object_id)

        applied = self.workspace.surface_texture_generation.get_assignment(
            assignment.assignment_id
        )
        self.assertIsNotNone(applied)
        assert applied is not None
        self.assertEqual(
            set(applied.surface_ids),
            source_trim_surface_ids | target_trim_surface_ids,
        )
        self.assertEqual(
            self.workspace.surface_texture_generation.get_surface_material_sources(),
            {},
        )

        with patch("housemaker.main.convert_to_glb", wraps=convert_to_glb) as convert:
            export_scene = self.workspace._build_pre_atlas_export_scene()

        self.assertEqual(convert.call_args.kwargs["surface_materials"], {})
        target_placement = next(
            placement
            for placement in export_scene.placed_models
            if placement.object_id == target_object_id
        )
        source_placement = next(
            placement
            for placement in export_scene.placed_models
            if placement.object_id == source_object_id
        )
        self.assertEqual(target_placement.source_object_id, target_object_id)
        self.assertEqual(
            target_placement.atlas_source_object_id,
            source_object_id,
        )
        self.assertEqual(
            {
                export_scene.surface_source_ids[surface_id]
                for surface_id in target_trim_surface_ids
            },
            {source_object_id},
        )
        source_geometries = tuple(
            geometry
            for _name, geometry in sorted(
                source_placement.model.scene.geometry.items(),
                key=lambda item: str(item[0]),
            )
        )
        target_geometries = tuple(
            geometry
            for _name, geometry in sorted(
                target_placement.model.scene.geometry.items(),
                key=lambda item: str(item[0]),
            )
        )
        persisted_geometries = tuple(
            geometry
            for _name, geometry in sorted(
                source_model.scene.geometry.items(),
                key=lambda item: str(item[0]),
            )
        )
        self.assertEqual(len(target_geometries), len(source_geometries))
        self.assertEqual(len(persisted_geometries), len(source_geometries))
        for persisted_geometry, source_geometry, target_geometry in zip(
            persisted_geometries,
            source_geometries,
            target_geometries,
            strict=True,
        ):
            self.assertIsInstance(persisted_geometry.visual, TextureVisuals)
            self.assertIsInstance(source_geometry.visual, TextureVisuals)
            self.assertIsInstance(target_geometry.visual, TextureVisuals)
            persisted_uv_rows = {
                tuple(float(value) for value in row)
                for row in np.round(
                    np.asarray(persisted_geometry.visual.uv, dtype=float),
                    decimals=6,
                )
            }
            source_uv = np.asarray(source_geometry.visual.uv, dtype=float)
            target_uv = np.asarray(target_geometry.visual.uv, dtype=float)
            self.assertTrue(
                {
                    tuple(float(value) for value in row)
                    for row in np.round(source_uv, decimals=6)
                }.issubset(persisted_uv_rows)
            )
            self.assertTrue(
                {
                    tuple(float(value) for value in row)
                    for row in np.round(target_uv, decimals=6)
                }.issubset(persisted_uv_rows)
            )
            np.testing.assert_array_equal(
                np.asarray(source_geometry.visual.material.baseColorTexture),
                np.asarray(target_geometry.visual.material.baseColorTexture),
            )

        self.workspace.surface_texture_generation.rename_assignment(
            assignment.assignment_id,
            "Custom cornice finish",
        )
        previous_paths = {
            relative_path
            for variant in applied.texture_variants
            for relative_path in variant.map_asset_paths.values()
        }
        second_color = (25, 140, 210, 255)
        regenerated_record, regenerated_model = self._install_trim_texture_family(
            source_object_id,
            color=second_color,
            task_id="cornice-texture-2",
        )
        self.workspace.generation.texture_regeneration_completed.emit(
            regenerated_record,
            regenerated_model,
        )
        self.workspace.generation.generated_object_changed.emit(
            regenerated_record,
            regenerated_model,
        )
        _qt_application.processEvents()

        current_assignments = tuple(
            assignment
            for assignment in (
                self.workspace.surface_texture_generation.get_assignments()
            )
            if assignment.source_object_id == source_object_id
        )
        self.assertEqual(len(current_assignments), 1)
        regenerated_assignment = current_assignments[0]
        self.assertEqual(regenerated_assignment.assignment_id, assignment.assignment_id)
        self.assertEqual(regenerated_assignment.display_name, "Custom cornice finish")
        self.assertEqual(
            set(regenerated_assignment.surface_ids),
            source_trim_surface_ids | target_trim_surface_ids,
        )
        regenerated_paths = {
            relative_path
            for variant in regenerated_assignment.texture_variants
            for relative_path in variant.map_asset_paths.values()
        }
        self.assertTrue(previous_paths.isdisjoint(regenerated_paths))
        self.assertTrue(
            all(
                not (surface_asset_directory / path).exists()
                for path in previous_paths
            )
        )
        self.assertTrue(
            all((surface_asset_directory / path).is_file() for path in regenerated_paths)
        )
        refreshed_source = atlas_workspace._sources_by_object_id[source_object_id]
        self.assertEqual(
            tuple(int(value) for value in refreshed_source.load_texture_rgba()[0, 0]),
            second_color,
        )
        refreshed_atlas = atlas_workspace.get_data().atlas_by_id("cornices")
        self.assertIsNotNone(refreshed_atlas)
        assert refreshed_atlas is not None
        refreshed_atlas_path = atlas_workspace._resolve_owned_atlas_path(
            refreshed_atlas.image_path
        )
        self.assertIsNotNone(refreshed_atlas_path)
        assert refreshed_atlas_path is not None
        self.assertNotEqual(refreshed_atlas_path.read_bytes(), first_atlas_png)

        self.workspace._handle_atlas_object_texture_resolution_changed(
            source_object_id,
            1024,
        )

        resized_assignment = (
            self.workspace.surface_texture_generation.get_assignment(
                assignment.assignment_id
            )
        )
        resized_object_variant = self.workspace.generation.get_active_texture_variant(
            source_object_id
        )
        self.assertIsNotNone(resized_assignment)
        self.assertIsNotNone(resized_object_variant)
        assert resized_assignment is not None and resized_object_variant is not None
        self.assertEqual(resized_assignment.selected_texture_resolution, 1024)
        self.assertEqual(resized_object_variant.resolution, 1024)

    def test_generated_cornice_texture_immediately_targets_selected_components(
        self,
    ) -> None:
        first_trim_id = "1" * 32
        second_trim_id = "2" * 32
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self._place_cornice(first_trim_id, _wall_surface_id(1, 2))
            self._place_cornice(second_trim_id, _wall_surface_id(2, 3))

        first_ids = tuple(
            part.semantic_id
            for part in self.workspace._canvas_architectural_trim_parts_by_id.values()
            if part.trim_id == first_trim_id
        )
        second_ids = tuple(
            part.semantic_id
            for part in self.workspace._canvas_architectural_trim_parts_by_id.values()
            if part.trim_id == second_trim_id
        )
        expected_ids = {*first_ids, *second_ids}
        self.assertEqual(len(expected_ids), 8)
        self.workspace._handle_architectural_trim_part_selection_changed(
            (*first_ids, *second_ids)
        )
        active_target = self.workspace.generation._architectural_trim_editing_target
        self.assertIsNotNone(active_target)
        assert active_target is not None
        self.assertEqual(active_target.trim_id, second_trim_id)

        operation_id = "multi-cornice-texture-operation"
        self.workspace.generation.operation_started.emit(
            operation_id,
            OBJECT_OPERATION_GENERATE_TEXTURE,
            active_target.object_id,
        )
        self.assertEqual(
            set(
                self.workspace
                ._architectural_trim_texture_targets_by_operation_id[
                    operation_id
                ][1]
            ),
            expected_ids,
        )

        generated_record, generated_model = self._install_trim_texture_family(
            active_target.object_id,
            color=(145, 90, 35, 255),
            task_id="multi-cornice-immediate-texture",
        )
        self.workspace.generation.texture_regeneration_completed.emit(
            generated_record,
            generated_model,
        )

        assignment = next(
            assignment
            for assignment in self.workspace.surface_texture_generation.get_assignments()
            if assignment.source_object_id == active_target.object_id
        )
        self.assertEqual(set(assignment.surface_ids), expected_ids)

        # Match the production completion order; its second publication must
        # retain the immutable generation-start target snapshot.
        self.workspace.generation.generated_object_changed.emit(
            generated_record,
            generated_model,
        )
        _qt_application.processEvents()
        republished_assignment = self.workspace.surface_texture_generation.get_assignment(
            assignment.assignment_id
        )
        self.assertIsNotNone(republished_assignment)
        assert republished_assignment is not None
        self.assertEqual(set(republished_assignment.surface_ids), expected_ids)

        self.workspace.generation.operation_finished.emit(operation_id)
        self.assertNotIn(
            operation_id,
            self.workspace._architectural_trim_texture_targets_by_operation_id,
        )

    def test_finished_cornice_job_retries_deferred_atlas_assignment(self) -> None:
        trim_id = "f" * 32
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self._place_cornice(trim_id, _wall_surface_id(1, 2))
        target = self.workspace.generation._architectural_trim_editing_target
        self.assertIsNotNone(target)
        assert target is not None
        source_object_id = target.object_id

        atlas_data = TextureAtlasData()
        atlas_data.create_atlas("Cornices", 2048, atlas_id="cornices")
        self.workspace.texture_atlas_workspace.set_data(atlas_data)
        generated_record, generated_model = self._install_trim_texture_family(
            source_object_id,
            color=(185, 95, 35, 255),
            task_id="active-cornice-texture-job",
        )

        with patch.object(
            self.workspace.generation,
            "has_active_object_job",
            return_value=True,
        ):
            self.workspace.generation.generated_object_changed.emit(
                generated_record,
                generated_model,
            )
            _qt_application.processEvents()

        atlas_workspace = self.workspace.texture_atlas_workspace
        pending_atlas = atlas_workspace.get_data().atlas_by_id("cornices")
        self.assertIsNotNone(pending_atlas)
        assert pending_atlas is not None
        self.assertIsNone(pending_atlas.placement_for_object(source_object_id))
        self.assertIn(
            source_object_id,
            atlas_workspace._surface_texture_entries_by_id,
        )

        with patch.object(
            self.workspace.generation,
            "has_active_object_job",
            return_value=False,
        ):
            self.workspace.generation.operation_finished.emit(
                "active-cornice-texture-job"
            )
            _qt_application.processEvents()

        packed_atlas = atlas_workspace.get_data().atlas_by_id("cornices")
        self.assertIsNotNone(packed_atlas)
        assert packed_atlas is not None
        self.assertIsNotNone(packed_atlas.placement_for_object(source_object_id))
        atlas_path = atlas_workspace._resolve_owned_atlas_path(packed_atlas.image_path)
        self.assertIsNotNone(atlas_path)
        assert atlas_path is not None
        self.assertTrue(atlas_path.is_file())
        self.assertGreater(atlas_path.stat().st_size, 0)

    def test_generated_cornice_multi_object_selection_assigns_every_component(
        self,
    ) -> None:
        source_trim_id = "6" * 32
        first_target_trim_id = "7" * 32
        second_target_trim_id = "8" * 32
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self._place_cornice(source_trim_id, _wall_surface_id(1, 2))
            source_target = self.workspace.generation._architectural_trim_editing_target
            self.assertIsNotNone(source_target)
            assert source_target is not None
            source_object_id = source_target.object_id

            self._place_cornice(first_target_trim_id, _wall_surface_id(2, 3))
            first_target = self.workspace.generation._architectural_trim_editing_target
            self.assertIsNotNone(first_target)
            assert first_target is not None

            self._place_cornice(second_target_trim_id, _wall_surface_id(3, 4))
            second_target = self.workspace.generation._architectural_trim_editing_target
            self.assertIsNotNone(second_target)
            assert second_target is not None

        atlas_data = TextureAtlasData()
        atlas_data.create_atlas("Cornices", 2048, atlas_id="cornices")
        self.workspace.texture_atlas_workspace.set_data(atlas_data)

        generated_record, generated_model = self._install_trim_texture_family(
            source_object_id,
            color=(165, 95, 45, 255),
            task_id="cornice-multi-source",
        )
        self.workspace.generation.texture_regeneration_completed.emit(
            generated_record,
            generated_model,
        )
        self.workspace.generation.generated_object_changed.emit(
            generated_record,
            generated_model,
        )
        _qt_application.processEvents()

        # Give both targets generated assets so their Canvas representations
        # use the placed-object selection path that originally lost all but
        # the active cornice.
        self._install_trim_texture_family(
            first_target.object_id,
            color=(65, 115, 175, 255),
            task_id="cornice-multi-target-one",
        )
        self._install_trim_texture_family(
            second_target.object_id,
            color=(75, 125, 185, 255),
            task_id="cornice-multi-target-two",
        )

        preview_model = self.workspace._build_viewer_preview_model(None)
        self.assertIsNotNone(preview_model)
        assert preview_model is not None
        self.workspace._sync_viewer_architectural_trim_placed_object_bindings(
            preview_model
        )

        expected_target_ids = {
            part.semantic_id
            for part in self.workspace._canvas_architectural_trim_parts_by_id.values()
            if part.trim_id in {first_target_trim_id, second_target_trim_id}
        }
        self.assertEqual(len(expected_target_ids), 8)
        bound_target_ids = {
            semantic_id
            for object_id in (first_target.object_id, second_target.object_id)
            for semantic_id in (
                self.workspace.viewer._architectural_trim_placed_object_part_ids[
                    object_id
                ]
            )
        }
        self.assertEqual(bound_target_ids, expected_target_ids)

        self.workspace._handle_canvas_placed_object_selection_set_changed(
            (first_target.object_id, second_target.object_id)
        )
        # A generic active-object signal can follow the selection-set signal.
        # It must not collapse the already translated semantic union.
        self.workspace._handle_canvas_placed_object_selection_changed(
            second_target.object_id
        )

        self.assertEqual(
            set(self.workspace._desired_canvas_architectural_trim_part_ids),
            expected_target_ids,
        )
        self.assertEqual(
            set(self.workspace._atlas_surface_assignment_target_ids),
            expected_target_ids,
        )

        assignment = next(
            assignment
            for assignment in self.workspace.surface_texture_generation.get_assignments()
            if assignment.source_object_id == source_object_id
        )
        source_ids = set(assignment.surface_ids)
        atlas_workspace = self.workspace.texture_atlas_workspace
        if not atlas_workspace.is_source_assigned_to_any_atlas(source_object_id):
            self.assertTrue(
                atlas_workspace.assign_source_to_selected_atlas(source_object_id)
            )

        self.workspace._handle_atlas_surface_assign_requested(source_object_id)

        applied = self.workspace.surface_texture_generation.get_assignment(
            assignment.assignment_id
        )
        self.assertIsNotNone(applied)
        assert applied is not None
        self.assertEqual(
            set(applied.surface_ids),
            source_ids | expected_target_ids,
        )

    def test_generated_cornice_texture_rejects_non_cornice_target(self) -> None:
        trim_id = "9" * 32
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self._place_cornice(trim_id, _wall_surface_id(1, 2))
        target = self.workspace.generation._architectural_trim_editing_target
        self.assertIsNotNone(target)
        assert target is not None
        source_object_id = target.object_id
        generated_record, generated_model = self._install_trim_texture_family(
            source_object_id,
            color=(100, 70, 40, 255),
            task_id="cornice-non-cornice-target",
        )
        self.workspace.generation.texture_regeneration_completed.emit(
            generated_record,
            generated_model,
        )
        self.workspace.generation.generated_object_changed.emit(
            generated_record,
            generated_model,
        )
        _qt_application.processEvents()
        assignment = next(
            assignment
            for assignment in self.workspace.surface_texture_generation.get_assignments()
            if assignment.source_object_id == source_object_id
        )
        original_surface_ids = assignment.surface_ids
        self.workspace._atlas_surface_assignment_target_ids = (
            _wall_surface_id(2, 3),
        )

        with patch.object(
            self.workspace.surface_texture_generation,
            "apply_assignment_texture",
            wraps=(
                self.workspace.surface_texture_generation.apply_assignment_texture
            ),
        ) as apply_assignment:
            self.workspace._handle_atlas_surface_assign_requested(source_object_id)

        apply_assignment.assert_not_called()
        rejected = self.workspace.surface_texture_generation.get_assignment(
            assignment.assignment_id
        )
        self.assertIsNotNone(rejected)
        assert rejected is not None
        self.assertEqual(rejected.surface_ids, original_surface_ids)
        self.assertEqual(
            self.workspace.texture_atlas_workspace.status_label.text(),
            "Generated cornice textures can only be assigned to complete "
            "cornice components.",
        )

    def test_reselecting_cornice_reuses_its_generation_component(self) -> None:
        trim_id = "d" * 32
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self._place_cornice(trim_id, _wall_surface_id(1, 2))
        semantic_id = next(
            part.semantic_id
            for part in self.workspace._canvas_architectural_trim_parts_by_id.values()
            if part.trim_id == trim_id and part.part_kind == "front"
        )

        self.workspace._handle_architectural_trim_part_selection_changed(
            (semantic_id,)
        )
        first_target = self.workspace.generation._architectural_trim_editing_target
        self.assertIsNotNone(first_target)
        first_record_count = len(
            self.workspace.generation.get_data().generated_objects
        )

        self.workspace._handle_architectural_trim_part_selection_changed(())
        self.workspace._handle_architectural_trim_part_selection_changed(
            (semantic_id,)
        )

        second_target = self.workspace.generation._architectural_trim_editing_target
        self.assertIsNotNone(second_target)
        assert first_target is not None and second_target is not None
        self.assertEqual(second_target.object_id, first_target.object_id)
        self.assertEqual(
            len(self.workspace.generation.get_data().generated_objects),
            first_record_count,
        )

    def test_generated_cornice_replaces_procedural_preview_and_export(self) -> None:
        trim_id = "e" * 32
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self._place_cornice(trim_id, _wall_surface_id(1, 2))
        target = self.workspace.generation._architectural_trim_editing_target
        self.assertIsNotNone(target)
        assert target is not None

        record_index = next(
            index
            for index, record in enumerate(
                self.workspace.generation._data.generated_objects
            )
            if record.object_id == target.object_id
        )
        seed_record = self.workspace.generation._data.generated_objects[
            record_index
        ]
        self.workspace.generation._data.generated_objects[record_index] = replace(
            seed_record,
            provider=MESHY_GENERATION_PROVIDER,
            provider_task_id="generated-cornice-task",
        )

        preview = self.workspace._build_viewer_preview_model(None)
        self.assertIsNotNone(preview)
        assert preview is not None
        self.assertEqual(
            [
                placed.object_id
                for placed in preview.preview_placed_objects
                if placed.object_id == target.object_id
            ],
            [target.object_id],
        )
        self.assertFalse(
            any(
                part.trim_id == trim_id
                for part in preview.preview_architectural_trim_parts
            )
        )

        export_scene = self.workspace._build_pre_atlas_export_scene()
        self.assertEqual(
            [
                placed.object_id
                for placed in export_scene.placed_models
                if placed.object_id == target.object_id
            ],
            [target.object_id],
        )

    def test_joined_run_dimension_edit_updates_both_members_and_deletes_both(
        self,
    ) -> None:
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self._place_skirting("2" * 32, _wall_surface_id(1, 2))
            self._place_skirting("3" * 32, _wall_surface_id(2, 3))

        edit = ArchitecturalTrimDimensionEdit(
            trim_id="2" * 32,
            handle_kind=TRIM_HANDLE_HEIGHT,
            value_meters=0.18,
        )
        with patch.object(
            self.workspace,
            "_schedule_viewer_preview_refresh",
        ) as schedule_refresh:
            self.workspace._handle_architectural_trim_edit_started(edit)
            self.workspace._handle_architectural_trim_edit_preview_changed(edit)
            schedule_refresh.assert_not_called()
            self.assertIsNotNone(self.workspace._active_architectural_trim_undo_state)
            self.workspace._handle_architectural_trim_edit_finished(edit, True)
            schedule_refresh.assert_not_called()
            self.assertTrue(
                self.workspace._architectural_trim_mesh_update_timer.isActive()
            )
            self.workspace._commit_pending_architectural_trim_mesh_update()
            schedule_refresh.assert_called_once_with(preserve_camera=True)

        self.assertEqual(
            [trim.height_meters for trim in self.level.architectural_trims],
            [0.18, 0.18],
        )
        active_target = self.workspace.generation._architectural_trim_editing_target
        self.assertIsNotNone(active_target)
        assert active_target is not None
        self.assertAlmostEqual(active_target.fit_dimensions[2], 0.18)
        self.assertEqual(
            {
                part.run_id
                for part in self.workspace._canvas_architectural_trim_parts_by_id.values()
            },
            {"2" * 32},
        )

        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self.workspace._handle_architectural_trim_deletion_requested(
                ("2" * 32, "3" * 32)
            )

        self.assertEqual(self.level.architectural_trims, [])

    def test_dimension_edit_release_keeps_preview_until_delayed_commit(
        self,
    ) -> None:
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self._place_skirting("5" * 32, _wall_surface_id(1, 2))
            self._place_skirting("6" * 32, _wall_surface_id(2, 3))
        self.workspace._canvas_undo_stack.clear()
        edit = ArchitecturalTrimDimensionEdit(
            trim_id="5" * 32,
            handle_kind=TRIM_HANDLE_HEIGHT,
            value_meters=0.18,
        )

        self.workspace._handle_architectural_trim_edit_started(edit)
        self.workspace._handle_architectural_trim_edit_preview_changed(edit)
        self.assertFalse(
            self.workspace._architectural_trim_mesh_update_timer.isActive()
        )
        preview_parts = tuple(
            part
            for part in self.workspace._canvas_architectural_trim_parts_by_id.values()
            if part.run_id == "5" * 32
        )
        self.workspace.viewer.set_architectural_trim_edit_preview_parts(preview_parts)

        with (
            patch.object(
                self.workspace,
                "_schedule_viewer_preview_refresh",
            ) as schedule_refresh,
            patch.object(
                self.workspace,
                "_reconcile_surface_assignments_with_scene",
                return_value=False,
            ) as reconcile_assignments,
            patch.object(
                self.workspace,
                "_record_canvas_undo_state",
            ) as record_undo,
        ):
            self.workspace._handle_architectural_trim_edit_finished(edit, True)

            self.assertTrue(
                self.workspace._architectural_trim_mesh_update_timer.isActive()
            )
            self.assertIsNotNone(self.workspace._pending_architectural_trim_undo_state)
            self.assertTrue(
                self.workspace.viewer._architectural_trim_edit_preview_parts
            )
            schedule_refresh.assert_not_called()
            reconcile_assignments.assert_not_called()
            record_undo.assert_not_called()

            self.workspace._commit_pending_architectural_trim_mesh_update()

        self.assertFalse(
            self.workspace._architectural_trim_mesh_update_timer.isActive()
        )
        self.assertIsNone(self.workspace._pending_architectural_trim_undo_state)
        self.assertFalse(self.workspace.viewer._architectural_trim_edit_preview_parts)
        reconcile_assignments.assert_called_once()
        record_undo.assert_called_once()
        schedule_refresh.assert_called_once_with(preserve_camera=True)
        self.assertEqual(
            [trim.height_meters for trim in self.level.architectural_trims],
            [0.18, 0.18],
        )

    def test_dimension_drag_rebuilds_only_the_affected_run_preview(self) -> None:
        """Dragging must not rebuild or reconcile the complete viewer scene."""

        first_trim_id = "a" * 32
        second_trim_id = "b" * 32
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self._place_skirting(first_trim_id, _wall_surface_id(1, 2))
            self._place_skirting(second_trim_id, _wall_surface_id(3, 4))

        first_run_id = self.workspace._get_architectural_trim_run_id(first_trim_id)
        second_run_id = self.workspace._get_architectural_trim_run_id(second_trim_id)
        self.assertIsNotNone(first_run_id)
        self.assertIsNotNone(second_run_id)
        self.assertNotEqual(first_run_id, second_run_id)
        original_parts = dict(self.workspace._canvas_architectural_trim_parts_by_id)
        original_first_bounds = {
            part.semantic_id: tuple(float(value) for value in part.mesh.bounds.flat)
            for part in original_parts.values()
            if part.run_id == first_run_id
        }
        original_second_bounds = {
            part.semantic_id: tuple(float(value) for value in part.mesh.bounds.flat)
            for part in original_parts.values()
            if part.run_id == second_run_id
        }
        self.workspace.viewer.set_architectural_trim_edit_preview_parts(
            tuple(
                part for part in original_parts.values() if part.run_id == first_run_id
            )
        )
        edit = ArchitecturalTrimDimensionEdit(
            trim_id=first_trim_id,
            handle_kind=TRIM_HANDLE_HEIGHT,
            value_meters=0.18,
        )

        with (
            patch.object(
                self.workspace,
                "_refresh_canvas_architectural_trim_targets",
            ) as refresh_all_targets,
            patch.object(
                self.workspace,
                "_reconcile_surface_assignments_with_scene",
            ) as reconcile_assignments,
            patch.object(
                self.workspace,
                "_schedule_viewer_preview_refresh",
            ) as schedule_scene_refresh,
            patch.object(
                self.workspace.viewer,
                "set_architectural_trim_parts",
            ) as replace_all_viewer_parts,
            patch.object(
                self.workspace.viewer,
                "set_architectural_trim_edit_targets",
            ) as replace_all_edit_targets,
            patch(
                "housemaker.main.build_base_fixed_surfaces",
            ) as rebuild_all_base_surfaces,
        ):
            self.workspace._handle_architectural_trim_edit_started(edit)
            self.workspace._handle_architectural_trim_edit_preview_changed(edit)

        refresh_all_targets.assert_not_called()
        reconcile_assignments.assert_not_called()
        schedule_scene_refresh.assert_not_called()
        replace_all_viewer_parts.assert_not_called()
        replace_all_edit_targets.assert_not_called()
        rebuild_all_base_surfaces.assert_not_called()
        self.assertEqual(
            self.workspace._canvas_architectural_trim_parts_by_id,
            original_parts,
        )
        preview_parts = self.workspace.viewer._architectural_trim_edit_preview_parts
        self.assertTrue(preview_parts)
        self.assertEqual({part.run_id for part in preview_parts}, {first_run_id})
        preview_bounds = {
            part.semantic_id: tuple(float(value) for value in part.mesh.bounds.flat)
            for part in preview_parts
        }
        self.assertEqual(preview_bounds.keys(), original_first_bounds.keys())
        self.assertTrue(
            any(
                preview_bounds[semantic_id] != original_first_bounds[semantic_id]
                for semantic_id in preview_bounds
            )
        )
        self.assertEqual(
            {
                part.semantic_id: tuple(float(value) for value in part.mesh.bounds.flat)
                for part in self.workspace._canvas_architectural_trim_parts_by_id.values()
                if part.run_id == second_run_id
            },
            original_second_bounds,
        )

    def test_repeated_dimension_drag_preserves_first_delayed_baseline(
        self,
    ) -> None:
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self._place_skirting("7" * 32, _wall_surface_id(1, 2))
        original_trim = self.level.architectural_trims[0]
        first_edit = ArchitecturalTrimDimensionEdit(
            trim_id="7" * 32,
            handle_kind=TRIM_HANDLE_HEIGHT,
            value_meters=0.18,
        )
        second_edit = ArchitecturalTrimDimensionEdit(
            trim_id="7" * 32,
            handle_kind=TRIM_HANDLE_HEIGHT,
            value_meters=0.24,
        )

        self.workspace._handle_architectural_trim_edit_started(first_edit)
        self.workspace._handle_architectural_trim_edit_preview_changed(first_edit)
        self.workspace._handle_architectural_trim_edit_finished(first_edit, True)
        original_baseline = self.workspace._pending_architectural_trim_undo_state
        self.assertIsNotNone(original_baseline)
        self.assertTrue(self.workspace._architectural_trim_mesh_update_timer.isActive())

        self.workspace._handle_architectural_trim_edit_started(second_edit)

        self.assertFalse(
            self.workspace._architectural_trim_mesh_update_timer.isActive()
        )
        self.assertIs(
            self.workspace._pending_architectural_trim_undo_state,
            original_baseline,
        )
        self.workspace._handle_architectural_trim_edit_preview_changed(second_edit)
        self.workspace._handle_architectural_trim_edit_finished(second_edit, True)
        self.assertTrue(self.workspace._architectural_trim_mesh_update_timer.isActive())
        self.assertIs(
            self.workspace._pending_architectural_trim_undo_state,
            original_baseline,
        )
        baseline_trims = dict(original_baseline.architectural_trims_by_level)[
            self.level.index
        ]
        self.assertEqual(baseline_trims, (original_trim,))
        self.assertAlmostEqual(
            self.level.architectural_trims[0].height_meters,
            0.24,
        )

    def test_shared_mesh_delay_restarts_pending_trim_timer(self) -> None:
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self._place_skirting("8" * 32, _wall_surface_id(1, 2))
        edit = ArchitecturalTrimDimensionEdit(
            trim_id="8" * 32,
            handle_kind=TRIM_HANDLE_HEIGHT,
            value_meters=0.18,
        )
        self.workspace._handle_architectural_trim_edit_started(edit)
        self.workspace._handle_architectural_trim_edit_preview_changed(edit)
        self.workspace._handle_architectural_trim_edit_finished(edit, True)
        self.assertTrue(self.workspace._architectural_trim_mesh_update_timer.isActive())

        self.workspace._set_mesh_edit_update_delay_seconds(2.4)

        timer = self.workspace._architectural_trim_mesh_update_timer
        self.assertEqual(timer.interval(), 2400)
        self.assertTrue(timer.isActive())

    def test_ctrl_z_restores_pending_dimension_edit_before_commit(self) -> None:
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self._place_skirting("9" * 32, _wall_surface_id(1, 2))
        original_trim = self.level.architectural_trims[0]
        self.workspace._canvas_undo_stack.clear()
        edit = ArchitecturalTrimDimensionEdit(
            trim_id="9" * 32,
            handle_kind=TRIM_HANDLE_HEIGHT,
            value_meters=0.18,
        )
        self.workspace._handle_architectural_trim_edit_started(edit)
        self.workspace._handle_architectural_trim_edit_preview_changed(edit)
        self.workspace._handle_architectural_trim_edit_finished(edit, True)

        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self.workspace._handle_canvas_undo_requested()

        self.assertEqual(self.level.architectural_trims, [original_trim])
        self.assertFalse(
            self.workspace._architectural_trim_mesh_update_timer.isActive()
        )
        self.assertIsNone(self.workspace._pending_architectural_trim_undo_state)
        self.assertFalse(self.workspace.viewer._architectural_trim_edit_preview_parts)
        self.assertEqual(self.workspace._canvas_undo_stack, [])

    def test_deleting_a_host_wall_removes_its_trim_and_undo_restores_it(
        self,
    ) -> None:
        self.workspace._sync_canvas_to_current_level()
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self._place_skirting("4" * 32, _wall_surface_id(1, 2))
            self.workspace.canvas.set_selected_vertex_ids((1,))
            self.workspace.canvas._delete_selected_vertices()

        self.assertEqual(self.level.architectural_trims, [])

        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self.workspace._handle_canvas_undo_requested()

        self.assertEqual(
            tuple(trim.trim_id for trim in self.level.architectural_trims),
            ("4" * 32,),
        )


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
