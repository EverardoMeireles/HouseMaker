# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import trimesh
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.glb import GeneratedModel
from housemaker.main import BlueprintWorkspace
from housemaker.settings_widget import (
    GENERATION_DISPLAY_SCREEN_SETTING_KEY,
    SCENE_3D_DISPLAY_SCREEN_SETTING_KEY,
)
from housemaker.surface_texture_state import (
    SURFACE_TYPE_WALL,
    SurfaceTextureAssignment,
    SurfaceTextureData,
)
from housemaker.texture_atlas_state import TextureAtlasData
from housemaker.texture_atlas_workspace import (
    build_atlas_wall_texture_source_id,
)


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Main-workspace external viewer integration tests ###
class ExternalViewerMainIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(
                Path(self._temporary_directory.name) / "settings.json"
            )
        )
        self.workspace.resize(1400, 850)
        self.workspace.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self._temporary_directory.cleanup()

    def test_scene_display_detaches_top_level_workspace_and_restores_it(
        self,
    ) -> None:
        screen = _primary_screen()
        screen_id = "screen:external-test"

        with patch(
            "housemaker.main.resolve_fullscreen_3d_viewer_screen",
            return_value=screen,
        ) as resolve_mock:
            self.workspace._apply_scene_3d_display_screen(screen_id)

        resolve_mock.assert_called_once_with(screen_id)
        host = self.workspace._external_scene_3d_host
        scene_workspace = self.workspace.scene_3d_workspace
        self.assertTrue(host.is_active)
        self.assertIs(host.viewer, scene_workspace)
        self.assertIs(host.screen, screen)
        self.assertIs(
            scene_workspace.parentWidget(),
            host.window,
        )
        self.assertIs(
            self.workspace.viewer.parentWidget(),
            scene_workspace,
        )
        self.assertTrue(host.window.isMaximized())
        self.assertEqual(self.workspace.workspace_tabs.indexOf(scene_workspace), -1)
        self.assertEqual(self.workspace.scene_3d_workspace_tab_index, -1)

        self.workspace._apply_scene_3d_display_screen(None)

        self.assertFalse(host.is_active)
        self.assertIs(
            self.workspace.workspace_tabs.widget(
                self.workspace.scene_3d_workspace_tab_index
            ),
            scene_workspace,
        )
        self.assertEqual(
            self.workspace.scene_3d_workspace_tab_index,
            self.workspace.workspace_tabs.indexOf(
                self.workspace.canvas_viewer_workspace
            )
            + 1,
        )

    def test_scene_model_is_visible_after_external_handoff(
        self,
    ) -> None:
        model = _generated_box_model()
        self.workspace.viewer.set_model(model)
        self.workspace._remember_current_canvas_preview_model(
            model,
            validated_dependency_signature=(
                self.workspace._build_viewer_preview_dependency_signature()
            ),
        )

        self._detach_scene("screen:external-test")
        _qt_application.processEvents()

        viewer = self.workspace.viewer
        host = self.workspace._external_scene_3d_host
        self.assertIs(viewer.model, model)
        self.assertIsNotNone(viewer.mesh_item)
        self.assertIn(viewer.mesh_item, viewer.view.items)
        self.assertTrue(host.window.isVisible())
        self.assertTrue(self.workspace.scene_3d_workspace.isVisible())
        self.assertTrue(viewer.isVisible())
        self.assertTrue(viewer.view.isVisible())
        self.assertGreater(viewer.view.width(), 0)
        self.assertGreater(viewer.view.height(), 0)

    def test_scene_and_generation_displays_remain_hosted_independently(
        self,
    ) -> None:
        scene_screen_id = "screen:external-scene-test"
        generation_screen_id = "screen:external-generation-test"
        scene_combo = self.workspace.settings_widget.scene_3d_display_screen_combo
        generation_combo = (
            self.workspace.settings_widget.generation_display_screen_combo
        )
        scene_combo.addItem("External scene display", scene_screen_id)
        generation_combo.addItem(
            "External Generation display",
            generation_screen_id,
        )

        with patch(
            "housemaker.main.resolve_fullscreen_3d_viewer_screen",
            return_value=_primary_screen(),
        ):
            scene_combo.setCurrentIndex(scene_combo.findData(scene_screen_id))
            generation_combo.setCurrentIndex(
                generation_combo.findData(generation_screen_id)
            )
            _qt_application.processEvents()

            self._assert_externally_hosted_workspace_is(
                self.workspace._external_scene_3d_host,
                self.workspace.scene_3d_workspace,
            )
            self._assert_externally_hosted_workspace_is(
                self.workspace._external_generation_host,
                self.workspace.merged_generation_workspace,
            )
            self.assertEqual(
                self.workspace.workspace_tabs.indexOf(
                    self.workspace.scene_3d_workspace
                ),
                -1,
            )
            self.assertEqual(
                self.workspace.workspace_tabs.indexOf(
                    self.workspace.merged_generation_workspace
                ),
                -1,
            )
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.texture_atlas_workspace
            )
            self.assertTrue(self.workspace._external_scene_3d_host.is_active)
            self.assertTrue(self.workspace._external_generation_host.is_active)
            self.assertIs(
                self.workspace.atlas_object_preview_viewer.parentWidget(),
                self.workspace.texture_atlas_workspace.object_preview_container,
            )
            self.assertFalse(
                self.workspace.atlas_object_preview_viewer.isHidden()
            )

            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.settings_widget
            )
            self.assertTrue(self.workspace._external_scene_3d_host.is_active)
            self.assertTrue(self.workspace._external_generation_host.is_active)

            scene_combo.setCurrentIndex(0)
            _qt_application.processEvents()

        self.assertFalse(self.workspace._external_scene_3d_host.is_active)
        self.assertTrue(self.workspace._external_generation_host.is_active)
        self.assertGreaterEqual(
            self.workspace.workspace_tabs.indexOf(
                self.workspace.scene_3d_workspace
            ),
            0,
        )
        self.assertEqual(
            self.workspace.workspace_tabs.indexOf(
                self.workspace.merged_generation_workspace
            ),
            -1,
        )

    def test_atlas_click_loads_exact_variant_in_embedded_viewer(self) -> None:
        asset_path = Path(self._temporary_directory.name) / "chair-2048.glb"
        asset_path.write_bytes(b"test glb")
        variant = SimpleNamespace(
            object_id="chair",
            resolution=2048,
            glb_asset_path=asset_path,
        )
        model = _generated_box_model()

        with (
            patch.object(
                self.workspace.generation,
                "get_texture_variant",
                return_value=variant,
            ) as variant_resolver,
            patch(
                "housemaker.main.import_generated_glb",
                return_value=model,
            ) as importer,
        ):
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.texture_atlas_workspace
            )
            self.workspace.texture_atlas_workspace.object_preview_requested.emit(
                "chair",
                2048,
            )
            _qt_application.processEvents()

        variant_resolver.assert_called_once_with("chair", 2048)
        importer.assert_called_once_with(b"test glb")
        self.assertIs(
            self.workspace.atlas_object_preview_viewer.parentWidget(),
            self.workspace.texture_atlas_workspace.object_preview_container,
        )
        self.assertEqual(
            self.workspace.atlas_object_preview_viewer
            .get_ambient_light_intensity(),
            1.0,
        )
        self.assertIs(self.workspace.atlas_object_preview_viewer.model, model)
        self.assertTrue(self.workspace.atlas_object_preview_viewer.isVisible())

    def test_atlas_surface_texture_uses_plane_in_embedded_viewer(self) -> None:
        assignment, source_id, _atlas_id = self._seed_packed_wall_texture()
        self.workspace.workspace_tabs.setCurrentWidget(
            self.workspace.texture_atlas_workspace
        )
        self.assertTrue(
            self.workspace.texture_atlas_workspace
            .request_selected_object_preview()
        )
        _qt_application.processEvents()

        viewer = self.workspace.atlas_object_preview_viewer
        self.assertIs(
            viewer.parentWidget(),
            self.workspace.texture_atlas_workspace.object_preview_container,
        )
        self.assertTrue(viewer.isVisible())
        self.assertIsNotNone(viewer.model)
        assert viewer.model is not None
        self.assertEqual(len(viewer.model.mesh.faces), 2)
        bounds = np.asarray(viewer.model.mesh.bounds, dtype=float)
        np.testing.assert_allclose(bounds[:, 0], (-1.0, 1.0))
        np.testing.assert_allclose(bounds[:, 1], (0.0, 0.0))
        np.testing.assert_allclose(bounds[:, 2], (0.0, 2.0))
        texture = viewer.model.mesh.visual.material.baseColorTexture
        texture_rgba = np.asarray(texture.convert("RGBA"), dtype=np.uint8)
        self.assertEqual(texture_rgba.shape, (512, 512, 4))
        np.testing.assert_array_equal(
            texture_rgba[256, 256],
            np.asarray((140, 75, 30, 255), dtype=np.uint8),
        )
        self.assertEqual(
            self.workspace._atlas_preview_variant_key[0],
            source_id,
        )
        self.assertEqual(assignment.assignment_id, "detached-wall")

    def test_cached_surface_plane_preview_skips_state_clone_and_png_decode(
        self,
    ) -> None:
        _assignment, _source_id, _atlas_id = self._seed_packed_wall_texture()
        atlas_workspace = self.workspace.texture_atlas_workspace
        self.assertTrue(atlas_workspace.request_selected_object_preview())
        cached_model = self.workspace.atlas_object_preview_viewer.model
        self.assertIsNotNone(cached_model)

        with (
            patch.object(
                self.workspace.surface_texture_generation,
                "get_data",
                side_effect=AssertionError("Surface data must not be cloned"),
            ),
            patch.object(
                self.workspace,
                "_build_atlas_wall_texture_source",
                wraps=self.workspace._build_atlas_wall_texture_source,
            ) as build_source,
            patch(
                "housemaker.main.build_texture_preview_plane_model"
            ) as build_plane,
        ):
            self.assertTrue(atlas_workspace.request_selected_object_preview())

        build_source.assert_not_called()
        build_plane.assert_not_called()
        self.assertIs(
            self.workspace.atlas_object_preview_viewer.model,
            cached_model,
        )

    def test_delete_key_in_embedded_atlas_viewer_unassigns_texture(self) -> None:
        assignment, source_id, atlas_id = self._seed_packed_wall_texture()
        atlas_workspace = self.workspace.texture_atlas_workspace
        self.workspace.workspace_tabs.setCurrentWidget(atlas_workspace)
        changes: list[TextureAtlasData] = []
        atlas_workspace.data_changed.connect(changes.append)

        viewer = self.workspace.atlas_object_preview_viewer
        viewer.view.setFocus()
        QTest.keyClick(viewer.view, Qt.Key.Key_Delete)
        _qt_application.processEvents()

        updated = atlas_workspace.get_data().atlas_by_id(atlas_id)
        assert updated is not None
        self.assertIsNone(updated.placement_for_object(source_id))
        self.assertEqual(len(changes), 1)
        source_path = (
            self.workspace.surface_texture_generation.get_assignment_asset_path(
                assignment.assignment_id
            )
        )
        self.assertIsNotNone(source_path)
        assert source_path is not None
        self.assertTrue(source_path.is_file())

    def test_atlas_display_detaches_complete_workspace_and_restores_tab(
        self,
    ) -> None:
        screen = _primary_screen()
        screen_id = "screen:external-atlas-test"
        combo = self.workspace.settings_widget.atlas_display_screen_combo
        combo.addItem("External Atlas display", screen_id)

        with patch(
            "housemaker.main.resolve_fullscreen_3d_viewer_screen",
            side_effect=lambda requested_id: (
                screen if requested_id == screen_id else None
            ),
        ):
            combo.setCurrentIndex(combo.findData(screen_id))
            _qt_application.processEvents()

            self.assertTrue(self.workspace._external_atlas_host.is_active)
            self.assertIs(
                self.workspace._external_atlas_host.viewer,
                self.workspace.texture_atlas_workspace,
            )
            self.assertIs(
                self.workspace.texture_atlas_workspace.parentWidget(),
                self.workspace._external_atlas_host.window,
            )
            self.assertTrue(
                self.workspace._external_atlas_host.window.isMaximized()
            )
            self.assertEqual(
                self.workspace.workspace_tabs.indexOf(
                    self.workspace.texture_atlas_workspace
                ),
                -1,
            )
            self.assertEqual(self.workspace.atlas_workspace_tab_index, -1)
            self.assertNotIn(
                "Atlas",
                [
                    self.workspace.workspace_tabs.tabText(index)
                    for index in range(self.workspace.workspace_tabs.count())
                ],
            )

            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.merged_generation_workspace
            )
            self.assertTrue(self.workspace._external_atlas_host.is_active)

            combo.setCurrentIndex(0)
            _qt_application.processEvents()

        self.assertFalse(self.workspace._external_atlas_host.is_active)
        restored_atlas_index = self.workspace.workspace_tabs.indexOf(
            self.workspace.texture_atlas_workspace
        )
        self.assertEqual(
            restored_atlas_index,
            self.workspace.workspace_tabs.indexOf(
                self.workspace.scene_3d_workspace
            )
            + 1,
        )
        self.assertEqual(
            self.workspace.atlas_workspace_tab_index,
            restored_atlas_index,
        )
        self.assertIs(
            self.workspace.workspace_tabs.widget(
                self.workspace.atlas_workspace_tab_index
            ),
            self.workspace.texture_atlas_workspace,
        )
        self.assertIs(
            self.workspace.workspace_tabs.currentWidget(),
            self.workspace.merged_generation_workspace,
        )
        self.assertTrue(self.workspace.merged_generation_workspace.isVisible())
        self.assertFalse(
            self.workspace.texture_atlas_workspace.isVisible()
        )

    def test_closing_detached_atlas_restores_tab_without_stealing_focus(
        self,
    ) -> None:
        screen = _primary_screen()
        screen_id = "screen:external-atlas-close-test"
        combo = self.workspace.settings_widget.atlas_display_screen_combo
        combo.addItem("External Atlas display", screen_id)

        with patch(
            "housemaker.main.resolve_fullscreen_3d_viewer_screen",
            side_effect=lambda requested_id: (
                screen if requested_id == screen_id else None
            ),
        ):
            combo.setCurrentIndex(combo.findData(screen_id))
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.merged_generation_workspace
            )
            self.workspace._external_atlas_host.window.close()
            _qt_application.processEvents()

        self.assertFalse(self.workspace._external_atlas_host.is_active)
        self.assertIsNone(combo.currentData())
        self.assertIs(
            self.workspace.workspace_tabs.currentWidget(),
            self.workspace.merged_generation_workspace,
        )
        self.assertEqual(
            self.workspace.workspace_tabs.indexOf(
                self.workspace.texture_atlas_workspace
            ),
            self.workspace.workspace_tabs.indexOf(
                self.workspace.scene_3d_workspace
            )
            + 1,
        )

    def test_missing_atlas_variant_clears_stale_detached_preview(self) -> None:
        viewer = self.workspace.atlas_object_preview_viewer
        viewer.set_model(_generated_box_model())
        self.workspace._atlas_preview_variant_key = (
            "old-object",
            1024,
            "old.glb",
            1,
            1,
        )

        with patch.object(
            self.workspace.generation,
            "get_texture_variant",
            return_value=None,
        ):
            self.workspace.texture_atlas_workspace.object_preview_requested.emit(
                "missing-object",
                2048,
            )
            _qt_application.processEvents()

        self.assertIsNone(viewer.model)
        self.assertIsNone(self.workspace._atlas_preview_variant_key)
        self.assertIn(
            "exact 3D texture variant is missing",
            self.workspace.texture_atlas_workspace.status_label.text(),
        )

    def test_repeated_missing_atlas_preview_clears_the_scene_once(self) -> None:
        viewer = self.workspace.atlas_object_preview_viewer
        viewer.set_model(_generated_box_model())
        self.workspace._atlas_preview_variant_key = ("old-object", 1024)

        with (
            patch.object(
                self.workspace.generation,
                "get_texture_variant",
                return_value=None,
            ),
            patch.object(
                viewer,
                "clear_model",
                wraps=viewer.clear_model,
            ) as clear_model,
        ):
            for _request_index in range(2):
                self.workspace.texture_atlas_workspace \
                    .object_preview_requested.emit("missing-object", 2048)
                _qt_application.processEvents()

        clear_model.assert_called_once_with()

    def test_generation_display_detaches_complete_workspace_and_restores_tab(
        self,
    ) -> None:
        screen_id = "screen:external-generation-test"
        combo = self.workspace.settings_widget.generation_display_screen_combo
        combo.addItem("External Generation display", screen_id)
        merged = self.workspace.merged_generation_workspace
        generation = self.workspace.generation
        panel = generation.object_3d_panel
        model = _generated_box_model()
        panel.viewer.set_model(model)
        generation._sync_model_statistics(model)

        with (
            patch(
                "housemaker.main.resolve_fullscreen_3d_viewer_screen",
                return_value=_primary_screen(),
            ),
            patch.object(generation, "refresh_file_backed_previews"),
        ):
            combo.setCurrentIndex(combo.findData(screen_id))
            _qt_application.processEvents()

        host = self.workspace._external_generation_host
        external_window = host.window
        self._assert_externally_hosted_workspace_is(host, merged)
        self.assertTrue(external_window.isMaximized())
        self.assertEqual(self.workspace.workspace_tabs.indexOf(merged), -1)
        self.assertEqual(self.workspace.generation_workspace_tab_index, -1)
        self.assertIn("12 triangles", panel.statistics_label.text())

        expected_external_widgets = (
            panel.viewer,
            generation.symmetric_division_checkbox,
            generation.delete_selected_faces_button,
            generation.convert_faces_to_glass_button,
            panel.projection_camera_controls,
            panel.statistics_label,
        )
        for widget in expected_external_widgets:
            with self.subTest(widget=widget.objectName() or type(widget).__name__):
                self.assertTrue(merged.isAncestorOf(widget))
                self.assertTrue(widget.isVisibleTo(external_window))
                self.assertGreater(widget.width(), 0)
                self.assertGreater(widget.height(), 0)

        for relocated_control in (
            generation.delete_selected_faces_button,
            generation.convert_faces_to_glass_button,
            panel.projection_camera_controls,
            panel.statistics_label,
        ):
            with self.subTest(
                relocated_control=(
                    relocated_control.objectName()
                    or type(relocated_control).__name__
                )
            ):
                self.assertTrue(
                    merged.object_controls.isAncestorOf(relocated_control)
                )
        self.assertFalse(hasattr(panel, "delete_object_button"))

        self.workspace._apply_generation_display_screen(None)
        _qt_application.processEvents()

        self.assertFalse(host.is_active)
        self.assertEqual(
            self.workspace.workspace_tabs.indexOf(merged),
            self.workspace.workspace_tabs.indexOf(
                self.workspace.texture_atlas_workspace
            )
            + 1,
        )
        self.assertTrue(merged.isAncestorOf(panel))

    def test_object_wireframe_syncs_through_detached_generation_workspace(
        self,
    ) -> None:
        screen_id = "screen:external-object-wireframe-test"
        combo = self.workspace.settings_widget.generation_display_screen_combo
        combo.addItem("External object wireframe display", screen_id)
        merged = self.workspace.merged_generation_workspace
        generation = self.workspace.generation
        generation.wireframe_checkbox.setChecked(True)

        with (
            patch(
                "housemaker.main.resolve_fullscreen_3d_viewer_screen",
                return_value=_primary_screen(),
            ),
            patch.object(generation, "refresh_file_backed_previews"),
        ):
            combo.setCurrentIndex(combo.findData(screen_id))
            _qt_application.processEvents()

        self._assert_externally_hosted_workspace_is(
            self.workspace._external_generation_host,
            merged,
        )
        self.assertTrue(merged.isAncestorOf(generation.result_view))
        self.assertTrue(generation.result_view.get_wireframe_enabled())

        generation.wireframe_checkbox.setChecked(False)

        self.assertFalse(generation.result_view.get_wireframe_enabled())
        generation.wireframe_checkbox.setChecked(True)
        self.workspace._apply_generation_display_screen(None)
        _qt_application.processEvents()

        self.assertTrue(generation.result_view.get_wireframe_enabled())
        self.assertGreaterEqual(
            self.workspace.workspace_tabs.indexOf(merged),
            0,
        )

    def test_external_window_close_resets_only_its_workspace_display(
        self,
    ) -> None:
        scene_screen_id = "screen:external-scene-test"
        generation_screen_id = "screen:external-generation-test"
        scene_combo = self.workspace.settings_widget.scene_3d_display_screen_combo
        generation_combo = (
            self.workspace.settings_widget.generation_display_screen_combo
        )
        scene_combo.addItem("External scene display", scene_screen_id)
        generation_combo.addItem(
            "External Generation display",
            generation_screen_id,
        )
        with patch(
            "housemaker.main.resolve_fullscreen_3d_viewer_screen",
            return_value=_primary_screen(),
        ):
            scene_combo.setCurrentIndex(scene_combo.findData(scene_screen_id))
            generation_combo.setCurrentIndex(
                generation_combo.findData(generation_screen_id)
            )
            _qt_application.processEvents()
            self.workspace._external_scene_3d_host.window.close()
            _qt_application.processEvents()

        self.assertEqual(scene_combo.currentIndex(), 0)
        self.assertIsNone(scene_combo.currentData())
        self.assertIsNone(
            self.workspace._application_settings.get(
                SCENE_3D_DISPLAY_SCREEN_SETTING_KEY
            )
        )
        self.assertFalse(self.workspace._external_scene_3d_host.is_active)
        self.assertTrue(self.workspace._external_generation_host.is_active)
        self.assertEqual(
            generation_combo.currentData(),
            generation_screen_id,
        )
        self.assertEqual(
            self.workspace._application_settings.get(
                GENERATION_DISPLAY_SCREEN_SETTING_KEY
            ),
            generation_screen_id,
        )
        self.assertGreaterEqual(
            self.workspace.workspace_tabs.indexOf(
                self.workspace.scene_3d_workspace
            ),
            0,
        )
        self.assertEqual(
            self.workspace.workspace_tabs.indexOf(
                self.workspace.merged_generation_workspace
            ),
            -1,
        )

    def test_scene_preview_is_refreshed_while_scene_is_external(self) -> None:
        screen_id = "screen:external-test"
        combo = self.workspace.settings_widget.scene_3d_display_screen_combo
        combo.addItem("External test display", screen_id)

        with patch(
            "housemaker.main.resolve_fullscreen_3d_viewer_screen",
            return_value=_primary_screen(),
        ):
            combo.setCurrentIndex(combo.findData(screen_id))
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.settings_widget
            )
            _qt_application.processEvents()

            with patch.object(
                self.workspace,
                "_refresh_viewer_preview",
            ) as refresh_mock:
                self.workspace._schedule_viewer_preview_refresh(
                    preserve_camera=False
                )
                _qt_application.processEvents()

        refresh_mock.assert_called_once_with(preserve_camera=False)

    def test_detached_workspaces_restore_in_canonical_order(self) -> None:
        screen = _primary_screen()
        with patch(
            "housemaker.main.resolve_fullscreen_3d_viewer_screen",
            return_value=screen,
        ):
            self.workspace._apply_scene_3d_display_screen("screen:scene")
            self.workspace._apply_generation_display_screen(
                "screen:generation"
            )
            self.workspace._apply_atlas_display_screen("screen:atlas")

        self.assertEqual(self._top_level_tab_labels(), ["Canvas", "Settings"])

        self.workspace._apply_generation_display_screen(None)
        self.assertEqual(
            self._top_level_tab_labels(),
            ["Canvas", "Generation", "Settings"],
        )
        self.workspace._apply_atlas_display_screen(None)
        self.assertEqual(
            self._top_level_tab_labels(),
            ["Canvas", "Atlas", "Generation", "Settings"],
        )
        self.workspace._apply_scene_3d_display_screen(None)
        self.assertEqual(
            self._top_level_tab_labels(),
            ["Canvas", "3D scene", "Atlas", "Generation", "Settings"],
        )
        self.assertEqual(self.workspace.canvas_workspace_tab_index, 0)
        self.assertEqual(self.workspace.scene_3d_workspace_tab_index, 1)
        self.assertEqual(self.workspace.atlas_workspace_tab_index, 2)
        self.assertEqual(self.workspace.generation_workspace_tab_index, 3)
        self.assertEqual(self.workspace.settings_workspace_tab_index, 4)

    # ### Test helpers ###
    def _seed_packed_wall_texture(
        self,
    ) -> tuple[SurfaceTextureAssignment, str, str]:
        assignment = _wall_texture_assignment(
            Path(self._temporary_directory.name) / "surface_textures"
        )
        surface_data = SurfaceTextureData(assignments=[assignment])
        self.workspace.surface_texture_generation.set_data(surface_data)
        self.workspace.surface_texture_generation.data_changed.emit(surface_data)

        atlas_data = TextureAtlasData()
        atlas = atlas_data.create_atlas(
            "Detached surfaces",
            2048,
            atlas_id="detached-surfaces",
        )
        atlas_workspace = self.workspace.texture_atlas_workspace
        atlas_workspace.set_data(atlas_data)
        source_id = build_atlas_wall_texture_source_id(
            assignment.assignment_id
        )
        self.assertTrue(atlas_workspace._select_object_row(source_id))
        self.assertTrue(
            atlas_workspace.assign_source_to_selected_atlas(source_id)
        )
        return assignment, source_id, atlas.atlas_id

    def _detach_scene(self, screen_id: str) -> None:
        with patch(
            "housemaker.main.resolve_fullscreen_3d_viewer_screen",
            return_value=_primary_screen(),
        ):
            self.workspace._apply_scene_3d_display_screen(screen_id)

        self.assertTrue(self.workspace._external_scene_3d_host.is_active)

    def _assert_externally_hosted_workspace_is(self, host, workspace) -> None:
        self.assertTrue(host.is_active)
        self.assertIs(host.viewer, workspace)
        self.assertIs(
            workspace.parentWidget(),
            host.window,
        )

    def _top_level_tab_labels(self) -> list[str]:
        return [
            self.workspace.workspace_tabs.tabText(index)
            for index in range(self.workspace.workspace_tabs.count())
        ]


# ### Test helpers ###
def _wall_texture_assignment(
    asset_directory: str | Path,
) -> SurfaceTextureAssignment:
    directory = Path(asset_directory)
    directory.mkdir(parents=True, exist_ok=True)
    asset_path = "detached-wall.png"
    Image.new("RGBA", (12, 8), (140, 75, 30, 255)).save(
        directory / asset_path,
        format="PNG",
    )
    return SurfaceTextureAssignment(
        assignment_id="detached-wall",
        surface_type=SURFACE_TYPE_WALL,
        surface_ids=("level:2/wall:1:2",),
        provider="test",
        asset_path=asset_path,
        texture_width=12,
        texture_height=8,
    )


def _generated_box_model() -> GeneratedModel:
    mesh = trimesh.creation.box()
    return GeneratedModel(
        mesh=mesh,
        scene=trimesh.Scene(mesh),
        glb_bytes=b"",
    )


def _primary_screen():
    screen = QGuiApplication.primaryScreen()
    if screen is None:
        raise RuntimeError("A QScreen is required for external viewer tests.")
    return screen


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
