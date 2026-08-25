# ### Environment setup ###
from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import trimesh
from PIL import Image
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.glb import GeneratedModel
from housemaker.main import BlueprintWorkspace
from housemaker.models import GROUND_LEVEL_INDEX, create_default_levels
from housemaker.project_io import ProjectData
from housemaker.surface_texture_state import (
    SURFACE_TYPE_WALL,
    SURFACE_TEXTURE_RESOLUTIONS,
    SurfaceTextureAssignment,
    SurfaceTextureData,
    SurfaceTextureVariant,
)
from housemaker.texture_atlas_state import TextureAtlasData
from housemaker.texture_atlas_workspace import (
    build_atlas_wall_texture_source_id,
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
    size: tuple[int, int] = (12, 8),
    color: tuple[int, int, int, int] = (170, 70, 25, 255),
) -> SurfaceTextureAssignment:
    directory = Path(asset_directory)
    directory.mkdir(parents=True, exist_ok=True)
    asset_path = f"{assignment_id}.png"
    Image.new("RGBA", size, color).save(directory / asset_path)
    return SurfaceTextureAssignment(
        assignment_id=assignment_id,
        surface_type=SURFACE_TYPE_WALL,
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
        self.assertFalse(
            self.workspace.texture_atlas_workspace
            .request_selected_object_preview()
        )
        self.assertIsNone(self.workspace.atlas_object_preview_viewer.model)
        self.assertNotIn(
            "3D texture variant is missing",
            self.workspace.texture_atlas_workspace.status_label.text(),
        )

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
                "get_data",
                return_value=generation_data,
            ),
            patch.object(
                self.workspace.generation,
                "get_active_texture_variant",
                return_value=variant,
            ),
            patch.object(
                self.workspace,
                "_build_atlas_object_texture_source",
                return_value=None,
            ) as source_builder,
        ):
            self.workspace._handle_generation_data_changed_for_atlases(
                object()
            )
            self.workspace._handle_generation_data_changed_for_atlases(
                object()
            )

        source_builder.assert_called_once_with(variant)

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
                "get_data",
                return_value=generation_data,
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

        wall_source_builder.assert_not_called()
        self.assertEqual(set_sources.call_args.args[0], [object_source])
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
            "get_data",
            return_value=generation_data,
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
            side_effect=lambda variant: variant,
        ), patch.object(
            self.workspace.texture_atlas_workspace,
            "set_object_texture_sources",
        ) as set_sources:
            self.workspace._sync_atlas_object_texture_sources()
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
            self.workspace.generation.face_purge_completed.emit(
                changed_record,
                _generated_box_model(),
            )
            _qt_application.processEvents()

        refresh_regenerated.assert_called_once_with("chair")

    def test_object_generation_exposes_no_texture_inpaint_signal(self) -> None:
        self.assertFalse(
            hasattr(self.workspace.generation, "texture_inpaint_completed")
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
                "get_data",
                return_value=generation_data,
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


def _write_texture_png(
    path: Path,
    resolution: int,
    color: tuple[int, int, int, int],
) -> None:
    Image.new("RGBA", (resolution, resolution), color).save(path, format="PNG")


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
