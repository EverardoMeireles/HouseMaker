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
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.glb import GeneratedModel
from housemaker.main import BlueprintWorkspace
from housemaker.settings_widget import FULLSCREEN_3D_VIEWER_SCREEN_SETTING_KEY
from housemaker.texture_atlas_view import TextureAtlasEntry
from housemaker.unused_face_removal import ALL_CAMERA_IDS


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Test constants ###
_EXPECTED_UNUSED_FACE_CAMERA_LABELS = {
    "pos_x": "+X",
    "neg_x": "-X",
    "pos_y": "+Y",
    "neg_y": "-Y",
    "top": "Top",
    "bottom": "Bottom",
}


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

    def test_resolved_display_detaches_viewer_and_none_restores_it(self) -> None:
        screen = _primary_screen()
        screen_id = "screen:external-test"

        with patch(
            "housemaker.main.resolve_fullscreen_3d_viewer_screen",
            return_value=screen,
        ) as resolve_mock:
            self.workspace._apply_fullscreen_3d_viewer_screen(screen_id)

        resolve_mock.assert_called_once_with(screen_id)
        self.assertTrue(self.workspace._external_viewer_host.is_active)
        self.assertIs(self.workspace._external_viewer_host.viewer, self.workspace.viewer)
        self.assertIs(self.workspace._external_viewer_host.screen, screen)
        self.assertIs(
            self.workspace.viewer.parentWidget(),
            self.workspace._external_viewer_host.window,
        )
        self.assertIsNot(
            self._canvas_3d_tab_widget(),
            self.workspace.viewer,
        )

        self.workspace._apply_fullscreen_3d_viewer_screen(None)

        self.assertFalse(self.workspace._external_viewer_host.is_active)
        self.assertIs(
            self._canvas_3d_tab_widget(),
            self.workspace.viewer,
        )

    def test_canvas_model_is_visible_after_external_fullscreen_handoff(
        self,
    ) -> None:
        """A hidden Canvas 3D subtab must become visible on its display."""

        self.assertEqual(
            self.workspace.canvas_viewer_tabs.currentIndex(),
            self.workspace.canvas_2d_view_tab_index,
        )
        model = _generated_box_model()
        self.workspace.viewer.set_model(model)

        self._detach_viewer("screen:external-test")
        _qt_application.processEvents()

        viewer = self.workspace.viewer
        self.assertIs(viewer.model, model)
        self.assertIsNotNone(viewer.mesh_item)
        self.assertIn(viewer.mesh_item, viewer.view.items)
        self.assertTrue(self.workspace._external_viewer_host.window.isVisible())
        self.assertTrue(viewer.isVisible())
        self.assertTrue(viewer.view.isVisible())
        self.assertGreater(viewer.view.width(), 0)
        self.assertGreater(viewer.view.height(), 0)

    def test_external_display_routes_to_the_active_tab_3d_view(self) -> None:
        screen_id = "screen:external-test"
        combo = self.workspace.settings_widget.fullscreen_3d_viewer_screen_combo
        combo.addItem("External test display", screen_id)

        with patch(
            "housemaker.main.resolve_fullscreen_3d_viewer_screen",
            return_value=_primary_screen(),
        ):
            combo.setCurrentIndex(combo.findData(screen_id))
            self._assert_externally_hosted_viewer_is(self.workspace.viewer)
            self.assertTrue(self.workspace.canvas_viewer_tabs.tabBar().isHidden())
            self.assertFalse(
                self.workspace.canvas_viewer_tabs.isTabEnabled(
                    self.workspace.canvas_3d_view_tab_index
                )
            )

            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.texture_atlas_workspace
            )
            self._assert_externally_hosted_viewer_is(
                self.workspace.atlas_object_preview_viewer
            )
            self.assertFalse(
                self.workspace.atlas_object_preview_viewer.isHidden()
            )

            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.surface_texture_generation
            )
            self._assert_externally_hosted_viewer_is(
                self.workspace.surface_texture_generation.surface_view
            )
            self.assertIs(
                self.workspace.surface_texture_generation.right_view_stack.currentWidget(),
                self.workspace.surface_texture_generation.texture_view_page,
            )
            self.assertFalse(
                self.workspace.canvas_viewer_tabs.tabBar().isHidden()
            )
            self.assertTrue(
                self.workspace.canvas_viewer_tabs.isTabEnabled(
                    self.workspace.canvas_3d_view_tab_index
                )
            )

            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.generation
            )
            self._assert_externally_hosted_viewer_is(
                self.workspace.generation.object_3d_panel
            )
            self.assertIs(
                self.workspace.generation.right_view_stack.currentWidget(),
                self.workspace.generation.texture_view_page,
            )
            self.assertIs(
                self.workspace.generation.generated_objects_list.parentWidget(),
                self.workspace.generation.object_3d_panel.details_panel,
            )

            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.settings_widget
            )
            self.assertFalse(self.workspace._external_viewer_host.is_active)
            self.assertIs(
                self.workspace.surface_texture_generation.right_view_stack.currentWidget(),
                self.workspace.surface_texture_generation.surface_3d_page,
            )
            self.assertIs(
                self.workspace.generation.right_view_stack.currentWidget(),
                self.workspace.generation.object_3d_page,
            )

            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.canvas_viewer_workspace
            )
            self._assert_externally_hosted_viewer_is(self.workspace.viewer)

    def test_atlas_click_loads_exact_variant_in_detached_viewer(self) -> None:
        screen_id = "screen:external-atlas-preview-test"
        combo = self.workspace.settings_widget.fullscreen_3d_viewer_screen_combo
        combo.addItem("External Atlas preview display", screen_id)
        asset_path = Path(self._temporary_directory.name) / "chair-2048.glb"
        asset_path.write_bytes(b"test glb")
        variant = SimpleNamespace(
            object_id="chair",
            resolution=2048,
            glb_asset_path=asset_path,
        )
        model = _generated_box_model()

        with (
            patch(
                "housemaker.main.resolve_fullscreen_3d_viewer_screen",
                return_value=_primary_screen(),
            ),
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
            combo.setCurrentIndex(combo.findData(screen_id))
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
        self._assert_externally_hosted_viewer_is(
            self.workspace.atlas_object_preview_viewer
        )
        self.assertEqual(
            self.workspace.atlas_object_preview_viewer
            .get_ambient_light_intensity(),
            1.0,
        )
        self.assertIs(self.workspace.atlas_object_preview_viewer.model, model)
        self.assertTrue(self.workspace.atlas_object_preview_viewer.isVisible())

    def test_returning_to_atlas_refreshes_the_selected_external_preview(
        self,
    ) -> None:
        screen_id = "screen:external-atlas-refresh-test"
        combo = self.workspace.settings_widget.fullscreen_3d_viewer_screen_combo
        combo.addItem("External Atlas refresh display", screen_id)

        with (
            patch(
                "housemaker.main.resolve_fullscreen_3d_viewer_screen",
                return_value=_primary_screen(),
            ),
            patch.object(
                self.workspace.texture_atlas_workspace,
                "request_selected_object_preview",
            ) as request_preview,
        ):
            combo.setCurrentIndex(combo.findData(screen_id))
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.texture_atlas_workspace
            )
            request_preview.assert_called_once_with()

            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.generation
            )
            request_preview.reset_mock()
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.texture_atlas_workspace
            )

        request_preview.assert_called_once_with()
        self._assert_externally_hosted_viewer_is(
            self.workspace.atlas_object_preview_viewer
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

    def test_complete_object_panel_is_visible_beside_external_viewer(
        self,
    ) -> None:
        screen_id = "screen:external-object-panel-test"
        combo = self.workspace.settings_widget.fullscreen_3d_viewer_screen_combo
        combo.addItem("External object-panel display", screen_id)
        generation = self.workspace.generation
        panel = generation.object_3d_panel
        model = _generated_box_model()
        panel.viewer.set_model(model)
        generation._sync_model_statistics(model)
        generation.generated_objects_list.addItem("Generated test object")

        with patch(
            "housemaker.main.resolve_fullscreen_3d_viewer_screen",
            return_value=_primary_screen(),
        ):
            combo.setCurrentIndex(combo.findData(screen_id))
            self.workspace.workspace_tabs.setCurrentWidget(generation)
            _qt_application.processEvents()

        external_window = self.workspace._external_viewer_host.window
        self._assert_externally_hosted_viewer_is(panel)
        self.assertTrue(panel.is_external_presentation_active)
        self.assertIs(
            generation.right_view_stack.currentWidget(),
            generation.texture_view_page,
        )
        self.assertGreaterEqual(
            panel.details_panel.geometry().left(),
            panel.viewer.geometry().right(),
        )
        self.assertIn("12 triangles", panel.statistics_label.text())
        self.assertEqual(panel.object_list.count(), 1)

        expected_external_widgets = (
            panel.viewer,
            panel.details_panel,
            panel.unused_face_camera_controls,
            panel.object_list,
            panel.delete_object_button,
            panel.statistics_label,
        )
        for widget in expected_external_widgets:
            with self.subTest(widget=widget.objectName() or type(widget).__name__):
                self.assertTrue(panel.isAncestorOf(widget))
                self.assertTrue(widget.isVisibleTo(external_window))
                self.assertGreater(widget.width(), 0)
                self.assertGreater(widget.height(), 0)

        self.workspace._apply_fullscreen_3d_viewer_screen(None)
        _qt_application.processEvents()

        self.assertFalse(panel.is_external_presentation_active)
        self.assertIs(
            generation.right_view_stack.currentWidget(),
            generation.object_3d_page,
        )

    def test_object_wireframe_syncs_external_3d_and_local_uv_preview(
        self,
    ) -> None:
        screen_id = "screen:external-object-uv-test"
        combo = self.workspace.settings_widget.fullscreen_3d_viewer_screen_combo
        combo.addItem("External object UV display", screen_id)
        generation = self.workspace.generation
        uv_triangles = (
            ((0.1, 0.1), (0.9, 0.1), (0.9, 0.9)),
            ((0.1, 0.1), (0.9, 0.9), (0.1, 0.9)),
        )
        generation.texture_view.set_atlases(
            (
                TextureAtlasEntry(
                    "object:resolution:1024",
                    "1024 x 1024",
                    np.full((64, 64, 4), (30, 50, 70, 255), dtype=np.uint8),
                    owner_id="object",
                ),
            )
        )
        generation.texture_view.set_uv_overlay_triangles(uv_triangles)
        generation.wireframe_checkbox.setChecked(True)

        with patch(
            "housemaker.main.resolve_fullscreen_3d_viewer_screen",
            return_value=_primary_screen(),
        ):
            combo.setCurrentIndex(combo.findData(screen_id))
            self.workspace.workspace_tabs.setCurrentWidget(generation)
            _qt_application.processEvents()

        self._assert_externally_hosted_viewer_is(generation.object_3d_panel)
        self.assertIs(
            generation.right_view_stack.currentWidget(),
            generation.texture_view_page,
        )
        self.assertTrue(generation.texture_view.isVisibleTo(self.workspace))
        self.assertTrue(generation.result_view.get_wireframe_enabled())
        self.assertTrue(generation.texture_view.uv_overlay_enabled)
        self.assertEqual(
            generation.texture_view.uv_overlay_triangles,
            uv_triangles,
        )
        self.assertFalse(generation.texture_view.preview_label.pixmap().isNull())

        generation.wireframe_checkbox.setChecked(False)

        self.assertFalse(generation.result_view.get_wireframe_enabled())
        self.assertFalse(generation.texture_view.uv_overlay_enabled)
        generation.wireframe_checkbox.setChecked(True)
        self.workspace._apply_fullscreen_3d_viewer_screen(None)
        _qt_application.processEvents()

        self.assertTrue(generation.result_view.get_wireframe_enabled())
        self.assertTrue(generation.texture_view.uv_overlay_enabled)
        self.assertIs(
            generation.right_view_stack.currentWidget(),
            generation.object_3d_page,
        )
        self.assertFalse(
            self.workspace.surface_texture_generation.texture_view.uv_overlay_enabled
        )

    def test_external_window_close_resets_display_dropdown_and_restores_viewer(
        self,
    ) -> None:
        screen_id = "screen:external-test"
        combo = self.workspace.settings_widget.fullscreen_3d_viewer_screen_combo
        combo.addItem("External test display", screen_id)
        combo.setCurrentIndex(combo.findData(screen_id))
        self._detach_viewer(screen_id)

        self.workspace._external_viewer_host.window.close()
        _qt_application.processEvents()

        self.assertEqual(combo.currentIndex(), 0)
        self.assertIsNone(combo.currentData())
        self.assertIsNone(
            self.workspace._application_settings.get(
                FULLSCREEN_3D_VIEWER_SCREEN_SETTING_KEY
            )
        )
        self.assertFalse(self.workspace._external_viewer_host.is_active)
        self.assertIs(
            self._canvas_3d_tab_widget(),
            self.workspace.viewer,
        )

    def test_unused_face_camera_controls_follow_object_panel_to_external_window(
        self,
    ) -> None:
        screen_id = "screen:external-test"
        combo = self.workspace.settings_widget.fullscreen_3d_viewer_screen_combo
        combo.addItem("External test display", screen_id)
        self.workspace.settings_widget.unused_face_removal_checkbox.setChecked(
            True
        )

        with patch(
            "housemaker.main.resolve_fullscreen_3d_viewer_screen",
            return_value=_primary_screen(),
        ):
            combo.setCurrentIndex(combo.findData(screen_id))
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.generation
            )
            _qt_application.processEvents()

        panel = self.workspace.generation.object_3d_panel
        controls = panel.unused_face_camera_controls
        checkboxes = panel.unused_face_camera_checkboxes
        panel.viewer.set_model(_generated_box_model())
        _qt_application.processEvents()
        self._assert_externally_hosted_viewer_is(panel)
        self.assertIs(controls.parentWidget(), panel.details_panel)
        self.assertTrue(controls.isEnabled())
        self.assertTrue(controls.isVisibleTo(panel))
        self.assertEqual(tuple(checkboxes), ALL_CAMERA_IDS)
        self.assertTrue(
            all(checkbox.isChecked() for checkbox in checkboxes.values())
        )
        self.assertTrue(
            panel.viewer.get_unused_face_camera_indicators_visible()
        )
        self.assertEqual(
            tuple(panel.viewer.unused_face_camera_indicator_items),
            ALL_CAMERA_IDS,
        )
        labels = panel.viewer.unused_face_camera_indicator_labels
        self.assertEqual(
            {
                camera_id: label.text
                for camera_id, label in labels.items()
            },
            _EXPECTED_UNUSED_FACE_CAMERA_LABELS,
        )
        self.assertTrue(
            all(
                item.visible() and item in panel.viewer.view.items
                for camera_items in (
                    panel.viewer.unused_face_camera_indicator_items.values()
                )
                for item in camera_items
            )
        )
        self.assertTrue(
            all(
                label.visible() and label in panel.viewer.view.items
                for label in labels.values()
            )
        )

        checkboxes["bottom"].setChecked(False)
        _qt_application.processEvents()

        self.assertEqual(
            panel.get_enabled_postprocess_camera_ids(),
            tuple(
                camera_id
                for camera_id in ALL_CAMERA_IDS
                if camera_id != "bottom"
            ),
        )
        self.assertFalse(
            any(
                item.visible()
                for item in (
                    panel.viewer.unused_face_camera_indicator_items["bottom"]
                )
            )
        )
        self.assertFalse(labels["bottom"].visible())
        self.assertTrue(
            all(
                item.visible()
                for camera_id, camera_items in (
                    panel.viewer.unused_face_camera_indicator_items.items()
                )
                if camera_id != "bottom"
                for item in camera_items
            )
        )
        self.assertTrue(
            all(
                label.visible()
                for camera_id, label in labels.items()
                if camera_id != "bottom"
            )
        )
        self.assertIs(
            self.workspace.generation.right_view_stack.currentWidget(),
            self.workspace.generation.texture_view_page,
        )

    def test_canvas_preview_is_not_refreshed_from_another_3d_tab(self) -> None:
        screen_id = "screen:external-test"
        combo = self.workspace.settings_widget.fullscreen_3d_viewer_screen_combo
        combo.addItem("External test display", screen_id)

        with patch(
            "housemaker.main.resolve_fullscreen_3d_viewer_screen",
            return_value=_primary_screen(),
        ):
            combo.setCurrentIndex(combo.findData(screen_id))
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.generation
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

        refresh_mock.assert_not_called()

    # ### Test helpers ###
    def _detach_viewer(self, screen_id: str) -> None:
        with patch(
            "housemaker.main.resolve_fullscreen_3d_viewer_screen",
            return_value=_primary_screen(),
        ):
            self.workspace._apply_fullscreen_3d_viewer_screen(screen_id)

        self.assertTrue(self.workspace._external_viewer_host.is_active)

    def _assert_externally_hosted_viewer_is(self, viewer) -> None:
        self.assertTrue(self.workspace._external_viewer_host.is_active)
        self.assertIs(self.workspace._external_viewer_host.viewer, viewer)
        self.assertIs(
            viewer.parentWidget(),
            self.workspace._external_viewer_host.window,
        )

    def _canvas_3d_tab_widget(self):
        return self.workspace.canvas_viewer_tabs.widget(
            self.workspace.canvas_3d_view_tab_index
        )


# ### Test helpers ###
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
