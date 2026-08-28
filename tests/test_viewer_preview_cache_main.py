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

import trimesh
from PIL import Image
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.generation_state import GeneratedObjectPlacement
from housemaker.glb import GeneratedModel
from housemaker.main import BlueprintWorkspace


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _preview_model(name: str) -> GeneratedModel:
    mesh = trimesh.creation.box(extents=(1.0, 1.5, 2.0))
    return GeneratedModel(
        mesh=mesh,
        scene=trimesh.Scene({name: mesh.copy()}),
        glb_bytes=b"",
    )


# ### Preview-cache integration tests ###
class ViewerPreviewCacheMainTests(unittest.TestCase):
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

    def test_canvas_2d_activation_defers_a_dirty_3d_build(self) -> None:
        expected_model = _preview_model("canvas-deferred")
        self.workspace.workspace_tabs.setCurrentWidget(
            self.workspace.settings_widget
        )

        with patch.object(
            self.workspace,
            "_build_viewer_preview_model",
            return_value=expected_model,
        ) as build_preview:
            self.workspace._schedule_viewer_preview_refresh(
                preserve_camera=False
            )
            _qt_application.processEvents()
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.canvas_viewer_workspace
            )
            _qt_application.processEvents()

            build_preview.assert_not_called()

            self.workspace.canvas_viewer_tabs.setCurrentIndex(
                self.workspace.canvas_3d_view_tab_index
            )
            _qt_application.processEvents()

        build_preview.assert_called_once_with(None)
        self.assertIs(self.workspace.viewer.model, expected_model)

    def test_canvas_3d_activation_checks_the_blueprint_once(self) -> None:
        self.workspace.workspace_tabs.setCurrentWidget(
            self.workspace.settings_widget
        )
        self.workspace.canvas_viewer_tabs.setCurrentIndex(
            self.workspace.canvas_3d_view_tab_index
        )

        with patch.object(
            self.workspace.canvas,
            "refresh_blueprint_image_if_stale",
            wraps=self.workspace.canvas.refresh_blueprint_image_if_stale,
        ) as refresh_blueprint:
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.canvas_viewer_workspace
            )
            _qt_application.processEvents()

        refresh_blueprint.assert_called_once_with()

    def test_returning_to_canvas_reloads_same_path_blueprint_replacement(
        self,
    ) -> None:
        image_path = Path(self._temporary_directory.name) / "blueprint.png"
        Image.new("RGB", (32, 24), (20, 40, 60)).save(image_path)
        self.workspace._set_current_level_image(str(image_path))
        self.workspace.canvas_viewer_tabs.setCurrentIndex(
            self.workspace.canvas_3d_view_tab_index
        )
        _qt_application.processEvents()
        self.workspace.workspace_tabs.setCurrentWidget(
            self.workspace.settings_widget
        )
        previous_stat = image_path.stat()
        Image.new("RGB", (48, 20), (180, 40, 20)).save(image_path)
        os.utime(
            image_path,
            ns=(
                previous_stat.st_atime_ns,
                previous_stat.st_mtime_ns + 1_000_000,
            ),
        )

        refreshed_model = _preview_model("resized-blueprint")
        with patch.object(
            self.workspace,
            "_build_viewer_preview_model",
            return_value=refreshed_model,
        ) as build_preview:
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.canvas_viewer_workspace
            )
            _qt_application.processEvents()
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.settings_widget
            )
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.canvas_viewer_workspace
            )
            _qt_application.processEvents()

        self.assertEqual(
            self.workspace.canvas.get_image_size_pixels(),
            (48.0, 20.0),
        )
        self.assertEqual(
            self.workspace.current_level.image_size_pixels,
            (48.0, 20.0),
        )
        build_preview.assert_called_once_with(None)
        self.assertIs(self.workspace.viewer.model, refreshed_model)

    def test_opening_canvas_3d_refreshes_a_changed_blueprint(self) -> None:
        image_path = Path(self._temporary_directory.name) / "canvas-3d.png"
        Image.new("RGB", (32, 24), (20, 40, 60)).save(image_path)
        self.workspace._set_current_level_image(str(image_path))
        previous_stat = image_path.stat()
        Image.new("RGB", (52, 28), (180, 40, 20)).save(image_path)
        os.utime(
            image_path,
            ns=(
                previous_stat.st_atime_ns,
                previous_stat.st_mtime_ns + 1_000_000,
            ),
        )

        refreshed_model = _preview_model("canvas-subtab-blueprint")
        with patch.object(
            self.workspace,
            "_build_viewer_preview_model",
            return_value=refreshed_model,
        ) as build_preview:
            self.workspace.canvas_viewer_tabs.setCurrentIndex(
                self.workspace.canvas_3d_view_tab_index
            )
            _qt_application.processEvents()

        self.assertEqual(
            self.workspace.current_level.image_size_pixels,
            (52.0, 28.0),
        )
        build_preview.assert_called_once_with(None)
        self.assertIs(self.workspace.viewer.model, refreshed_model)

    def test_opening_surface_refreshes_a_changed_blueprint(self) -> None:
        image_path = Path(self._temporary_directory.name) / "surface.png"
        Image.new("RGB", (32, 24), (20, 40, 60)).save(image_path)
        self.workspace._set_current_level_image(str(image_path))
        previous_stat = image_path.stat()
        Image.new("RGB", (60, 36), (180, 40, 20)).save(image_path)
        os.utime(
            image_path,
            ns=(
                previous_stat.st_atime_ns,
                previous_stat.st_mtime_ns + 1_000_000,
            ),
        )

        refreshed_model = _preview_model("surface-blueprint")
        with patch.object(
            self.workspace,
            "_build_viewer_preview_model",
            return_value=refreshed_model,
        ) as build_preview:
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.surface_texture_generation
            )
            _qt_application.processEvents()

        self.assertEqual(
            self.workspace.current_level.image_size_pixels,
            (60.0, 36.0),
        )
        build_preview.assert_called_once_with(None)
        self.assertIs(
            self.workspace.surface_texture_generation.surface_view
            .get_scene_model(),
            refreshed_model,
        )

    def test_surface_refreshes_a_changed_non_current_blueprint(self) -> None:
        level = next(
            item
            for position, item in enumerate(self.workspace.levels)
            if position != self.workspace.current_level_index
        )
        image_path = Path(self._temporary_directory.name) / "other-level.png"
        Image.new("RGB", (32, 24), (20, 40, 60)).save(image_path)
        level.image_path = str(image_path)
        level.image_size_pixels = (32.0, 24.0)
        self.workspace._refresh_blueprint_file_dependencies(
            include_exported_levels=True
        )
        previous_stat = image_path.stat()
        Image.new("RGB", (72, 44), (180, 40, 20)).save(image_path)
        os.utime(
            image_path,
            ns=(
                previous_stat.st_atime_ns,
                previous_stat.st_mtime_ns + 1_000_000,
            ),
        )

        refreshed_model = _preview_model("other-level-blueprint")
        with patch.object(
            self.workspace,
            "_build_viewer_preview_model",
            return_value=refreshed_model,
        ) as build_preview:
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.surface_texture_generation
            )
            _qt_application.processEvents()

        self.assertEqual(level.image_size_pixels, (72.0, 44.0))
        build_preview.assert_called_once_with(None)

    def test_failed_current_decode_retries_after_switching_levels(self) -> None:
        level = self.workspace.current_level
        image_path = Path(self._temporary_directory.name) / "retry-level.png"
        Image.new("RGB", (32, 24), (20, 40, 60)).save(image_path)
        self.workspace._set_current_level_image(str(image_path))
        self.workspace._refresh_blueprint_file_dependencies(
            include_exported_levels=False
        )
        validated_revision = self.workspace._level_blueprint_image_revisions[
            level.index
        ]
        previous_stat = image_path.stat()
        Image.new("RGB", (68, 40), (180, 40, 20)).save(image_path)
        os.utime(
            image_path,
            ns=(
                previous_stat.st_atime_ns,
                previous_stat.st_mtime_ns + 1_000_000,
            ),
        )

        with patch(
            "housemaker.blueprint_canvas._load_qimage_from_path",
            side_effect=ValueError("temporary decode failure"),
        ):
            self.workspace._refresh_blueprint_file_dependencies(
                include_exported_levels=False
            )

        self.assertEqual(
            self.workspace._level_blueprint_image_revisions[level.index],
            validated_revision,
        )
        self.workspace.current_level_index = next(
            position
            for position in range(len(self.workspace.levels))
            if position != self.workspace.current_level_index
        )
        self.workspace._sync_canvas_to_current_level()

        refreshed_model = _preview_model("retried-other-level")
        with patch.object(
            self.workspace,
            "_build_viewer_preview_model",
            return_value=refreshed_model,
        ) as build_preview:
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.surface_texture_generation
            )
            _qt_application.processEvents()

        self.assertEqual(level.image_size_pixels, (68.0, 40.0))
        build_preview.assert_called_once_with(None)

    def test_missing_blueprint_keeps_last_geometry_dimensions(self) -> None:
        image_path = Path(self._temporary_directory.name) / "missing-blueprint.png"
        Image.new("RGB", (32, 24), (20, 40, 60)).save(image_path)
        self.workspace._set_current_level_image(str(image_path))
        with patch.object(
            self.workspace,
            "_build_viewer_preview_model",
            return_value=_preview_model("before-missing-blueprint"),
        ):
            self.workspace.canvas_viewer_tabs.setCurrentIndex(
                self.workspace.canvas_3d_view_tab_index
            )
            _qt_application.processEvents()
        self.workspace.workspace_tabs.setCurrentWidget(
            self.workspace.settings_widget
        )
        image_path.unlink()

        with patch.object(
            self.workspace,
            "_build_viewer_preview_model",
            return_value=_preview_model("unexpected-missing-rebuild"),
        ) as build_preview:
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.canvas_viewer_workspace
            )
            _qt_application.processEvents()
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.settings_widget
            )
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.canvas_viewer_workspace
            )
            _qt_application.processEvents()

        self.assertIsNone(self.workspace.canvas.blueprint_image)
        self.assertIsNone(self.workspace.canvas.get_image_size_pixels())
        self.assertEqual(
            self.workspace.current_level.image_size_pixels,
            (32.0, 24.0),
        )
        build_preview.assert_not_called()

    def test_reopening_current_tabs_reuses_both_gl_scenes(self) -> None:
        expected_model = _preview_model("shared-current")
        with (
            patch.object(
                self.workspace,
                "_build_viewer_preview_model",
                return_value=expected_model,
            ) as build_preview,
            patch.object(
                self.workspace.viewer,
                "set_model",
                wraps=self.workspace.viewer.set_model,
            ) as set_canvas_model,
            patch.object(
                self.workspace.surface_texture_generation,
                "set_preview_context",
                wraps=(
                    self.workspace.surface_texture_generation.set_preview_context
                ),
            ) as set_surface_model,
            patch.object(
                self.workspace.surface_texture_generation.surface_view,
                "set_levels",
                wraps=(
                    self.workspace.surface_texture_generation.surface_view
                    .set_levels
                ),
            ) as set_surface_levels,
        ):
            self.workspace.canvas_viewer_tabs.setCurrentIndex(
                self.workspace.canvas_3d_view_tab_index
            )
            _qt_application.processEvents()
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.surface_texture_generation
            )
            _qt_application.processEvents()
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.texture_atlas_workspace
            )
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.canvas_viewer_workspace
            )
            _qt_application.processEvents()
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.texture_atlas_workspace
            )
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.surface_texture_generation
            )
            _qt_application.processEvents()

        build_preview.assert_called_once_with(None)
        set_canvas_model.assert_called_once()
        set_surface_model.assert_called_once_with(
            self.workspace.levels,
            self.workspace.initial_first_person_camera,
            expected_model,
        )
        set_surface_levels.assert_not_called()

    def test_reopening_every_workspace_skips_unchanged_heavy_work(self) -> None:
        expected_model = _preview_model("all-workspaces-current")
        with patch.object(
            self.workspace,
            "_build_viewer_preview_model",
            return_value=expected_model,
        ):
            self.workspace.canvas_viewer_tabs.setCurrentIndex(
                self.workspace.canvas_3d_view_tab_index
            )
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.surface_texture_generation
            )
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.settings_widget
            )
            _qt_application.processEvents()

        with (
            patch.object(
                self.workspace.settings_widget,
                "get_settings",
                side_effect=AssertionError("unexpected full settings reload"),
            ),
            patch.object(
                self.workspace,
                "_build_viewer_preview_model",
            ) as build_preview,
            patch.object(
                self.workspace.viewer,
                "set_model",
            ) as set_canvas_model,
            patch.object(
                self.workspace.surface_texture_generation,
                "set_preview_context",
            ) as set_surface_context,
            patch.object(
                self.workspace.texture_atlas_workspace,
                "_refresh_all",
            ) as refresh_atlas,
            patch.object(
                self.workspace.generation,
                "_display_generated_object",
            ) as display_generated_object,
            patch.object(
                self.workspace.generation.object_3d_panel,
                "set_external_presentation_active",
                wraps=(
                    self.workspace.generation.object_3d_panel
                    .set_external_presentation_active
                ),
            ) as arrange_object_panel,
        ):
            workspaces = (
                self.workspace.canvas_viewer_workspace,
                self.workspace.texture_atlas_workspace,
                self.workspace.surface_texture_generation,
                self.workspace.generation,
                self.workspace.settings_widget,
            )
            for _pass_index in range(2):
                for workspace in workspaces:
                    self.workspace.workspace_tabs.setCurrentWidget(workspace)
            _qt_application.processEvents()

        build_preview.assert_not_called()
        set_canvas_model.assert_not_called()
        set_surface_context.assert_not_called()
        refresh_atlas.assert_not_called()
        display_generated_object.assert_not_called()
        arrange_object_panel.assert_not_called()

    def test_inactive_change_builds_once_and_updates_only_stale_views(self) -> None:
        first_model = _preview_model("first")
        second_model = _preview_model("second")
        with (
            patch.object(
                self.workspace,
                "_build_viewer_preview_model",
                side_effect=(first_model, second_model),
            ) as build_preview,
            patch.object(
                self.workspace.viewer,
                "set_model",
                wraps=self.workspace.viewer.set_model,
            ) as set_canvas_model,
            patch.object(
                self.workspace.surface_texture_generation,
                "set_preview_context",
                wraps=(
                    self.workspace.surface_texture_generation.set_preview_context
                ),
            ) as set_surface_model,
        ):
            self.workspace.canvas_viewer_tabs.setCurrentIndex(
                self.workspace.canvas_3d_view_tab_index
            )
            _qt_application.processEvents()
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.surface_texture_generation
            )
            _qt_application.processEvents()
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.settings_widget
            )

            self.workspace._schedule_viewer_preview_refresh()
            _qt_application.processEvents()
            self.assertEqual(build_preview.call_count, 1)

            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.surface_texture_generation
            )
            _qt_application.processEvents()
            self.assertEqual(build_preview.call_count, 2)
            self.assertEqual(set_canvas_model.call_count, 1)
            self.assertEqual(set_surface_model.call_count, 2)
            self.assertIs(
                self.workspace.surface_texture_generation.surface_view
                .get_scene_model(),
                second_model,
            )

            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.canvas_viewer_workspace
            )
            _qt_application.processEvents()

        self.assertEqual(build_preview.call_count, 2)
        self.assertEqual(set_canvas_model.call_count, 2)
        self.assertIs(self.workspace.viewer.model, second_model)

    def test_failed_preview_build_stays_stale_and_retries_on_reopen(self) -> None:
        expected_model = _preview_model("recovered")
        self.workspace.workspace_tabs.setCurrentWidget(
            self.workspace.settings_widget
        )
        self.workspace._schedule_viewer_preview_refresh()

        with patch.object(
            self.workspace,
            "_build_viewer_preview_model",
            side_effect=(None, expected_model),
        ) as build_preview:
            self.workspace.canvas_viewer_tabs.setCurrentIndex(
                self.workspace.canvas_3d_view_tab_index
            )
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.canvas_viewer_workspace
            )
            _qt_application.processEvents()

            self.assertNotEqual(
                self.workspace._canvas_viewer_preview_revision,
                self.workspace._viewer_preview_revision,
            )
            self.assertNotEqual(
                self.workspace._viewer_preview_model_revision,
                self.workspace._viewer_preview_revision,
            )

            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.settings_widget
            )
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.canvas_viewer_workspace
            )
            _qt_application.processEvents()

        self.assertEqual(build_preview.call_count, 2)
        self.assertEqual(
            self.workspace._canvas_viewer_preview_revision,
            self.workspace._viewer_preview_revision,
        )
        self.assertIs(self.workspace.viewer.model, expected_model)

    def test_window_apply_does_not_bless_changed_dependencies(self) -> None:
        generated_model = _preview_model("window-race")
        validated_signature = (("before",),)
        revision_before = self.workspace._viewer_preview_revision

        with (
            patch.object(
                self.workspace,
                "_build_viewer_preview_dependency_signature",
                return_value=(("after",),),
            ),
            patch.object(self.workspace.viewer, "set_wall_targets"),
            patch.object(self.workspace.viewer, "set_model"),
            patch.object(
                self.workspace,
                "_queue_viewer_preview_refresh",
            ) as queue_refresh,
        ):
            was_remembered = self.workspace._apply_canvas_window_preview(
                generated_model,
                dependency_signature=validated_signature,
            )

        self.assertFalse(was_remembered)
        self.assertEqual(
            self.workspace._viewer_preview_revision,
            revision_before + 1,
        )
        self.assertNotEqual(
            self.workspace._canvas_viewer_preview_revision,
            self.workspace._viewer_preview_revision,
        )
        queue_refresh.assert_called_once_with()

    def test_export_does_not_publish_a_cross_revision_model(self) -> None:
        generated_model = _preview_model("export-race")
        export_path = Path(self._temporary_directory.name) / "unstable.glb"

        with (
            patch(
                "housemaker.main.QFileDialog.getSaveFileName",
                return_value=(str(export_path), "GLB Files (*.glb)"),
            ),
            patch.object(
                self.workspace,
                "_build_generated_model",
                return_value=generated_model,
            ) as build_model,
            patch.object(
                self.workspace,
                "_build_viewer_preview_dependency_signature",
                side_effect=(
                    (("before",),),
                    (("after",),),
                ),
            ),
            patch("housemaker.main.export_glb_file") as export_model,
            patch("housemaker.main.QMessageBox.warning") as show_warning,
        ):
            self.workspace._handle_glb_export_clicked()

        build_model.assert_called_once_with("Export failed")
        export_model.assert_not_called()
        show_warning.assert_called_once()
        self.assertFalse(export_path.exists())

    def test_reopening_detects_out_of_band_preview_asset_changes(self) -> None:
        first_model = _preview_model("dependency-before")
        second_model = _preview_model("dependency-after")
        dependency_signature = ("before",)

        def get_dependency_signature() -> tuple[str, ...]:
            return dependency_signature

        with (
            patch.object(
                self.workspace,
                "_build_viewer_preview_dependency_signature",
                side_effect=get_dependency_signature,
            ),
            patch.object(
                self.workspace,
                "_build_viewer_preview_model",
                side_effect=(first_model, second_model),
            ) as build_preview,
        ):
            self.workspace.canvas_viewer_tabs.setCurrentIndex(
                self.workspace.canvas_3d_view_tab_index
            )
            _qt_application.processEvents()
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.settings_widget
            )

            dependency_signature = ("after",)
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.canvas_viewer_workspace
            )
            _qt_application.processEvents()
            self.assertIs(self.workspace.viewer.model, second_model)

            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.surface_texture_generation
            )
            _qt_application.processEvents()

        self.assertEqual(build_preview.call_count, 2)
        self.assertIs(
            self.workspace.surface_texture_generation.surface_view
            .get_scene_model(),
            second_model,
        )

    def test_gizmo_commit_defers_only_the_surface_rebuild(self) -> None:
        first_model = _preview_model("before-gizmo")
        second_model = _preview_model("after-gizmo")
        with patch.object(
            self.workspace,
            "_build_viewer_preview_model",
            return_value=first_model,
        ):
            self.workspace.canvas_viewer_tabs.setCurrentIndex(
                self.workspace.canvas_3d_view_tab_index
            )
            _qt_application.processEvents()
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.surface_texture_generation
            )
            _qt_application.processEvents()
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.canvas_viewer_workspace
            )
            _qt_application.processEvents()

        level = self.workspace.current_level
        existing_placement = GeneratedObjectPlacement(
            level.index,
            10.0,
            20.0,
        )
        previous_revision = self.workspace._viewer_preview_revision
        with (
            patch.object(
                self.workspace.generation,
                "get_generated_object_placement",
                return_value=existing_placement,
            ),
            patch.object(
                self.workspace.generation,
                "update_generated_object_placement",
                return_value=True,
            ),
            patch(
                "housemaker.main.level_world_to_image_xy",
                return_value=(30.0, 40.0),
            ),
            patch.object(
                self.workspace,
                "_build_viewer_preview_model",
                return_value=second_model,
            ) as build_preview,
            patch.object(self.workspace.viewer, "set_model") as set_canvas_model,
        ):
            self.workspace._handle_placed_object_transform_changed(
                "chair",
                (1.0, 2.0, 3.0),
                (0.0, 0.0, 45.0),
            )
            _qt_application.processEvents()

            self.assertEqual(
                self.workspace._viewer_preview_revision,
                previous_revision + 1,
            )
            self.assertEqual(
                self.workspace._canvas_viewer_preview_revision,
                self.workspace._viewer_preview_revision,
            )
            self.assertNotEqual(
                self.workspace._surface_viewer_preview_revision,
                self.workspace._viewer_preview_revision,
            )
            build_preview.assert_not_called()
            set_canvas_model.assert_not_called()

            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.surface_texture_generation
            )
            _qt_application.processEvents()

        build_preview.assert_called_once_with(None)
        set_canvas_model.assert_not_called()
        self.assertIs(
            self.workspace.surface_texture_generation.surface_view
            .get_scene_model(),
            second_model,
        )

    def test_canvas_reopen_detects_asset_change_after_gizmo_fast_path(
        self,
    ) -> None:
        first_model = _preview_model("before-gizmo-asset-change")
        second_model = _preview_model("after-gizmo-asset-change")
        dependency_signature = ("asset-before",)

        def get_dependency_signature() -> tuple[str, ...]:
            return dependency_signature

        with (
            patch.object(
                self.workspace,
                "_build_viewer_preview_dependency_signature",
                side_effect=get_dependency_signature,
            ),
            patch.object(
                self.workspace,
                "_build_viewer_preview_model",
                side_effect=(first_model, second_model),
            ) as build_preview,
        ):
            self.workspace.canvas_viewer_tabs.setCurrentIndex(
                self.workspace.canvas_3d_view_tab_index
            )
            _qt_application.processEvents()

            level = self.workspace.current_level
            placement = GeneratedObjectPlacement(level.index, 10.0, 20.0)
            with (
                patch.object(
                    self.workspace.generation,
                    "get_generated_object_placement",
                    return_value=placement,
                ),
                patch.object(
                    self.workspace.generation,
                    "update_generated_object_placement",
                    return_value=True,
                ),
                patch(
                    "housemaker.main.level_world_to_image_xy",
                    return_value=(30.0, 40.0),
                ),
            ):
                self.workspace._handle_placed_object_transform_changed(
                    "chair",
                    (1.0, 2.0, 3.0),
                    (0.0, 0.0, 45.0),
                )

            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.settings_widget
            )
            dependency_signature = ("asset-after",)
            self.workspace.workspace_tabs.setCurrentWidget(
                self.workspace.canvas_viewer_workspace
            )
            _qt_application.processEvents()

        self.assertEqual(build_preview.call_count, 2)
        self.assertIs(self.workspace.viewer.model, second_model)

    def test_asset_change_before_gizmo_disables_the_fast_path(self) -> None:
        first_model = _preview_model("before-pre-gizmo-change")
        second_model = _preview_model("after-pre-gizmo-change")
        dependency_signature = ("asset-before",)

        def get_dependency_signature() -> tuple[str, ...]:
            return dependency_signature

        with (
            patch.object(
                self.workspace,
                "_build_viewer_preview_dependency_signature",
                side_effect=get_dependency_signature,
            ),
            patch.object(
                self.workspace,
                "_build_viewer_preview_model",
                side_effect=(first_model, second_model),
            ) as build_preview,
        ):
            self.workspace.canvas_viewer_tabs.setCurrentIndex(
                self.workspace.canvas_3d_view_tab_index
            )
            _qt_application.processEvents()
            dependency_signature = ("asset-after",)

            level = self.workspace.current_level
            placement = GeneratedObjectPlacement(level.index, 10.0, 20.0)
            with (
                patch.object(
                    self.workspace.generation,
                    "get_generated_object_placement",
                    return_value=placement,
                ),
                patch.object(
                    self.workspace.generation,
                    "update_generated_object_placement",
                    return_value=True,
                ),
                patch(
                    "housemaker.main.level_world_to_image_xy",
                    return_value=(30.0, 40.0),
                ),
            ):
                self.workspace._handle_placed_object_transform_changed(
                    "chair",
                    (1.0, 2.0, 3.0),
                    (0.0, 0.0, 45.0),
                )
            _qt_application.processEvents()

        self.assertEqual(build_preview.call_count, 2)
        self.assertIs(self.workspace.viewer.model, second_model)

    def test_gizmo_fast_path_accepts_only_the_target_placement_change(
        self,
    ) -> None:
        level = self.workspace.current_level
        existing_placement = GeneratedObjectPlacement(
            level.index,
            10.0,
            20.0,
        )
        dependency_before = (
            (("room",),),
            (("chair", existing_placement, ("asset",), 512, None),),
            (("surface",),),
        )
        dependency_after: tuple[object, ...] | None = None

        def update_placement(
            _object_id: str,
            placement: GeneratedObjectPlacement,
            *,
            emit_change_signals: bool,
        ) -> bool:
            nonlocal dependency_after
            self.assertFalse(emit_change_signals)
            dependency_after = (
                dependency_before[0],
                (("chair", placement, ("asset",), 512, None),),
                dependency_before[2],
            )
            return True

        def dependency_signature() -> tuple[object, ...]:
            return (
                dependency_before
                if dependency_after is None
                else dependency_after
            )

        revision_before = self.workspace._viewer_preview_revision
        self.workspace._canvas_viewer_preview_revision = revision_before
        self.workspace._viewer_preview_dependency_signature = dependency_before
        self.workspace._viewer_preview_dependency_signature_revision = (
            revision_before
        )
        with (
            patch.object(
                self.workspace.generation,
                "get_generated_object_placement",
                return_value=existing_placement,
            ),
            patch.object(
                self.workspace.generation,
                "update_generated_object_placement",
                side_effect=update_placement,
            ),
            patch(
                "housemaker.main.level_world_to_image_xy",
                return_value=(30.0, 40.0),
            ),
            patch.object(
                self.workspace,
                "_build_viewer_preview_dependency_signature",
                side_effect=dependency_signature,
            ),
            patch.object(
                self.workspace,
                "_queue_viewer_preview_refresh",
            ) as queue_refresh,
        ):
            self.workspace._handle_placed_object_transform_changed(
                "chair",
                (1.0, 2.0, 3.0),
                (0.0, 0.0, 45.0),
            )

        self.assertIsNotNone(dependency_after)
        self.assertEqual(
            self.workspace._viewer_preview_revision,
            revision_before + 1,
        )
        self.assertEqual(
            self.workspace._canvas_viewer_preview_revision,
            self.workspace._viewer_preview_revision,
        )
        self.assertEqual(
            self.workspace._viewer_preview_dependency_signature,
            dependency_after,
        )
        queue_refresh.assert_not_called()

    def test_unrelated_change_during_gizmo_queues_a_rebuild(self) -> None:
        level = self.workspace.current_level
        existing_placement = GeneratedObjectPlacement(
            level.index,
            10.0,
            20.0,
        )
        dependency_before = (
            (("room-before",),),
            (("chair", existing_placement, ("asset",), 512, None),),
            (("surface",),),
        )
        dependency_after: tuple[object, ...] | None = None

        def update_placement(
            _object_id: str,
            placement: GeneratedObjectPlacement,
            *,
            emit_change_signals: bool,
        ) -> bool:
            nonlocal dependency_after
            self.assertFalse(emit_change_signals)
            dependency_after = (
                (("room-after",),),
                (("chair", placement, ("asset",), 512, None),),
                dependency_before[2],
            )
            return True

        def dependency_signature() -> tuple[object, ...]:
            return (
                dependency_before
                if dependency_after is None
                else dependency_after
            )

        revision_before = self.workspace._viewer_preview_revision
        self.workspace._canvas_viewer_preview_revision = revision_before
        self.workspace._viewer_preview_dependency_signature = dependency_before
        self.workspace._viewer_preview_dependency_signature_revision = (
            revision_before
        )
        with (
            patch.object(
                self.workspace.generation,
                "get_generated_object_placement",
                return_value=existing_placement,
            ),
            patch.object(
                self.workspace.generation,
                "update_generated_object_placement",
                side_effect=update_placement,
            ),
            patch(
                "housemaker.main.level_world_to_image_xy",
                return_value=(30.0, 40.0),
            ),
            patch.object(
                self.workspace,
                "_build_viewer_preview_dependency_signature",
                side_effect=dependency_signature,
            ),
            patch.object(
                self.workspace,
                "_queue_viewer_preview_refresh",
            ) as queue_refresh,
        ):
            self.workspace._handle_placed_object_transform_changed(
                "chair",
                (1.0, 2.0, 3.0),
                (0.0, 0.0, 45.0),
            )

        self.assertEqual(
            self.workspace._viewer_preview_revision,
            revision_before + 1,
        )
        self.assertNotEqual(
            self.workspace._canvas_viewer_preview_revision,
            self.workspace._viewer_preview_revision,
        )
        self.assertEqual(
            self.workspace._viewer_preview_dependency_signature,
            dependency_before,
        )
        queue_refresh.assert_called_once_with()

    def test_known_missing_placed_asset_is_skipped_until_it_reappears(self) -> None:
        record = SimpleNamespace(
            object_id="missing-chair",
            object_name="Missing chair",
            placement=GeneratedObjectPlacement(
                self.workspace.current_level.index,
                20.0,
                30.0,
            ),
        )

        with (
            patch.object(
                self.workspace.generation,
                "get_data",
                return_value=SimpleNamespace(generated_objects=[record]),
            ),
            patch.object(
                self.workspace.generation,
                "get_generated_object_model",
                return_value=None,
            ),
            patch.object(
                self.workspace.generation,
                "is_generated_object_asset_available",
                return_value=False,
            ),
        ):
            placed_models = self.workspace._build_placed_generated_models()

        self.assertEqual(placed_models, ())

    def test_transient_placed_model_failure_is_not_committed_as_success(
        self,
    ) -> None:
        record = SimpleNamespace(
            object_id="retry-chair",
            object_name="Retry chair",
            placement=GeneratedObjectPlacement(
                self.workspace.current_level.index,
                20.0,
                30.0,
            ),
        )
        recovered_model = _preview_model("retry-chair")

        with (
            patch.object(
                self.workspace.generation,
                "get_data",
                return_value=SimpleNamespace(generated_objects=[record]),
            ),
            patch.object(
                self.workspace.generation,
                "get_generated_object_model",
                side_effect=(None, recovered_model),
            ) as load_model,
            patch.object(
                self.workspace.generation,
                "is_generated_object_asset_available",
                return_value=True,
            ),
            patch.object(
                self.workspace.generation,
                "resolve_symmetric_division_for_record",
                return_value=None,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "temporarily unavailable"):
                self.workspace._build_placed_generated_models()
            recovered_models = self.workspace._build_placed_generated_models()

        self.assertEqual(len(recovered_models), 1)
        self.assertIs(recovered_models[0].model, recovered_model)
        self.assertEqual(load_model.call_count, 2)


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
