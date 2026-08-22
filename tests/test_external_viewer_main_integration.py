# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import trimesh
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.glb import GeneratedModel
from housemaker.main import BlueprintWorkspace
from housemaker.settings_widget import FULLSCREEN_3D_VIEWER_SCREEN_SETTING_KEY


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
                self.workspace.generation.object_3d_panel,
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
