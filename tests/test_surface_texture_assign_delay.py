"""Actual Assign buttons retain scene materials until the mesh-edit delay."""

# ### Imports and Qt fixtures ###
from __future__ import annotations

import time
from unittest.mock import patch

from PySide6.QtCore import Qt

from housemaker.texture_atlas_workspace import build_atlas_wall_texture_source_id
from tests import test_surface_texture_library_main_integration as library_fixtures
from tests.test_texture_atlas_main_integration import _qt_application


# ### Real Main assignment routes ###
class TestSurfaceTextureAssignDelay:
    def setup_method(self):
        library_fixtures.SurfaceTextureLibraryMainIntegrationTests.setUp(self)

    def teardown_method(self):
        library_fixtures.SurfaceTextureLibraryMainIntegrationTests.tearDown(self)

    _install_room = (
        library_fixtures.SurfaceTextureLibraryMainIntegrationTests._install_room
    )
    _assignment = library_fixtures.SurfaceTextureLibraryMainIntegrationTests._assignment
    _load_assignments = (
        library_fixtures.SurfaceTextureLibraryMainIntegrationTests._load_assignments
    )

    def _prepare_assign(self, *, delay_seconds=60):
        workspace = self.workspace
        target = self._install_room()["wall"]
        workspace._set_mesh_edit_update_delay_seconds(delay_seconds)
        assignment = self._assignment("delayed-material")
        self._load_assignments(assignment)
        workspace.workspace_tabs.setCurrentWidget(workspace.scene_3d_workspace)
        workspace._refresh_viewer_preview(preserve_camera=True)
        _qt_application.processEvents()
        original_model = workspace.viewer.model
        assert original_model is not None
        assert workspace.surface_texture_generation.surface_view is None
        assert workspace.viewer.select_canvas_surface_target(target)
        atlas = workspace.texture_atlas_workspace
        source_id = build_atlas_wall_texture_source_id(assignment.assignment_id)
        atlas.surface_list.object_clicked.emit(source_id, Qt.MouseButton.LeftButton)
        assert atlas.place_assign_button.text() == "Assign"
        assert atlas.place_assign_button.isEnabled()
        return target, atlas, original_model

    def test_assign_button_defers_the_visible_scene_across_event_processing(self):
        workspace = self.workspace
        target, atlas, original_model = self._prepare_assign()

        with patch.object(
            workspace.viewer, "set_model", wraps=workspace.viewer.set_model
        ) as set_model:
            atlas.place_assign_button.click()
            _qt_application.processEvents()
            _qt_application.processEvents()

            assert workspace._pending_surface_texture_ids == {target}
            assert workspace._surface_texture_mesh_update_timer.isActive()
            assert workspace.viewer.model is original_model
            assert target in workspace.viewer._pending_surface_texture_preview_items
            set_model.assert_not_called()

    def test_slow_atlas_refresh_does_not_consume_the_visible_preview_quiet_period(self):
        workspace = self.workspace
        target, atlas, original_model = self._prepare_assign(delay_seconds=0.25)
        original_refresh = atlas._refresh_all

        def slow_post_commit_refresh():
            # Real large Atlas/PBR writes and thumbnail refreshes can take
            # longer than the configured delay without returning to Qt.
            original_refresh()
            if workspace._is_assigning_surface_texture_from_atlas:
                time.sleep(0.6)

        with (
            patch.object(atlas, "_refresh_all", side_effect=slow_post_commit_refresh),
            patch.object(
                workspace.viewer, "set_model", wraps=workspace.viewer.set_model
            ) as set_model,
        ):
            atlas.place_assign_button.click()
            _qt_application.processEvents()

            assert workspace._pending_surface_texture_ids == {target}
            assert workspace._surface_texture_mesh_update_timer.isActive()
            assert workspace.viewer.model is original_model
            assert target in workspace.viewer._pending_surface_texture_preview_items
            set_model.assert_not_called()

    def test_reentrant_atlas_events_cannot_start_delay_before_the_transaction_finishes(
        self,
    ):
        workspace = self.workspace
        target, atlas, original_model = self._prepare_assign(delay_seconds=0.25)
        original_refresh = atlas._refresh_all
        observed_pending = []
        workspace.surface_texture_generation.data_changed.connect(
            lambda _data: observed_pending.append(
                set(workspace._pending_surface_texture_ids)
            )
        )

        def reentrant_post_commit_refresh():
            original_refresh()
            _qt_application.processEvents()
            assert workspace._pending_surface_texture_ids == {target}
            assert not workspace._surface_texture_mesh_update_timer.isActive()
            time.sleep(0.6)

        with (
            patch.object(
                atlas, "_refresh_all", side_effect=reentrant_post_commit_refresh
            ),
            patch.object(
                workspace.viewer, "set_model", wraps=workspace.viewer.set_model
            ) as set_model,
        ):
            atlas.place_assign_button.click()
            _qt_application.processEvents()

            assert observed_pending and all(
                target in pending for pending in observed_pending
            )
            assert workspace._surface_texture_mesh_update_timer.isActive()
            assert workspace._pending_surface_texture_ids == {target}
            assert workspace.viewer.model is original_model
            assert target in workspace.viewer._pending_surface_texture_preview_items
            set_model.assert_not_called()
