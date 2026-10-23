"""Selection must not rebuild geometry or refresh a hidden Atlas preview."""

# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ### Imports ###
import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.main import BlueprintWorkspace
from housemaker.models import LevelData, VertexData
from housemaker.surface_geometry import build_fixed_surfaces

# ### Fixtures ###
_application = QApplication.instance() or QApplication([])
_application.setQuitOnLastWindowClosed(False)


@pytest.fixture
def workspace(tmp_path):
    widget = BlueprintWorkspace(
        application_settings=ApplicationSettingsStore(tmp_path / "settings.json"),
    )
    graph = VertexData()
    for point in ((0.0, 0.0), (200.0, 0.0), (200.0, 150.0), (0.0, 150.0)):
        graph.add_vertex(*point)
    for edge in ((1, 2), (2, 3), (3, 4), (4, 1)):
        graph.add_edge(*edge)
    widget.levels = [LevelData(index=2, name="Ground", vertex_data=graph)]
    widget.current_level_index = 0
    widget._reset_viewer_doorway_snapshots()
    widget._sync_canvas_to_current_level()
    widget._set_canvas_viewer_targets(tuple(build_fixed_surfaces(widget.levels)))
    widget._sync_viewer_scene_levels()
    yield widget
    widget.shutdown()
    widget.close()
    _application.processEvents()


# ### Selection-only integration tests ###
def test_changed_wall_selection_never_notifies_content_or_rebuilds_geometry(workspace):
    walls = tuple(
        surface
        for surface in workspace._canvas_surface_targets_by_id.values()
        if surface.wall_key is not None
    )
    revision = workspace._viewer_preview_revision
    content_updates = []
    workspace.surface_texture_generation.data_changed.connect(content_updates.append)
    with (
        patch(
            "housemaker.main.build_fixed_surfaces",
            side_effect=AssertionError("selection rebuilt geometry"),
        ) as build,
        patch.object(
            workspace,
            "_sync_atlas_object_texture_sources",
            side_effect=AssertionError("selection scanned content"),
        ) as sync,
        patch.object(
            workspace.texture_atlas_workspace, "request_selected_object_preview"
        ) as preview,
    ):
        for wall in (*walls, *reversed(walls)):
            workspace.viewer.select_canvas_surface_target(wall.surface_id)
            assert workspace.viewer.get_selected_canvas_surface_ids() == (
                wall.surface_id,
            )
            assert workspace.surface_texture_generation.get_selected_surface_ids() == (
                wall.surface_id,
            )
            assert (
                workspace.merged_generation_workspace.generate_texture_stack.currentWidget()
                is workspace.surface_texture_generation.generate_button
            )
        workspace.viewer.select_canvas_surface_target(None)

    assert not content_updates
    build.assert_not_called()
    sync.assert_not_called()
    preview.assert_not_called()
    assert workspace._viewer_preview_revision == revision
    assert not workspace.surface_texture_generation.get_selected_surface_ids()
    assert (
        workspace.merged_generation_workspace.generate_texture_stack.currentWidget()
        is workspace.generation.generate_texture_button
    )


@pytest.mark.parametrize("atlas_active", (False, True))
def test_scene_selection_only_requests_preview_for_visible_atlas(
    workspace, atlas_active
):
    with (
        patch.object(
            workspace, "_atlas_workspace_is_active", return_value=atlas_active
        ),
        patch.object(
            workspace.texture_atlas_workspace, "select_source_ids", return_value=()
        ) as select,
    ):
        workspace._sync_atlas_texture_selection_from_canvas_scene()
    select.assert_called_once_with(
        (), active_source_id=None, request_preview=atlas_active
    )


def test_actual_atlas_tab_entry_flushes_hidden_selection_without_source_rescan(
    workspace,
):
    atlas = workspace.texture_atlas_workspace
    with (
        patch.object(workspace, "_atlas_sources_need_tab_sync", return_value=False),
        patch.object(workspace, "_sync_atlas_object_texture_sources") as sync,
        patch.object(workspace, "_schedule_atlas_draw_call_estimate"),
        patch.object(workspace, "_ensure_viewer_preview_current"),
        patch.object(atlas, "request_selected_object_preview") as preview,
    ):
        workspace._sync_atlas_texture_selection_from_canvas_scene()
        preview.assert_not_called()
        workspace.workspace_tabs.setCurrentWidget(atlas)
        preview.assert_called_once()
        sync.assert_not_called()

        workspace.workspace_tabs.setCurrentWidget(workspace.canvas_viewer_workspace)
        workspace._sync_atlas_texture_selection_from_canvas_scene()
        assert preview.call_count == 1
        workspace.workspace_tabs.setCurrentWidget(atlas)
        assert preview.call_count == 2
        sync.assert_not_called()


# ### Cached geometry, live assignment and invalidation tests ###
def test_requirements_reuse_geometry_but_keep_material_bindings_live(workspace):
    wall_id = "level:2/wall:1:2"
    mapping = {wall_id: "surface-a"}
    with (
        patch.object(
            workspace,
            "_build_atlas_surface_source_ids",
            side_effect=lambda: dict(mapping),
        ),
        patch(
            "housemaker.main.build_fixed_surfaces", wraps=build_fixed_surfaces
        ) as build,
    ):
        assert workspace._build_required_scene_atlas_source_ids() == ("surface-a",)
        assert workspace._build_required_scene_atlas_source_ids() == ("surface-a",)
        assert build.call_count == 1
        mapping[wall_id] = "surface-b"
        assert workspace._build_required_scene_atlas_source_ids() == ("surface-b",)
        assert build.call_count == 1

        workspace.current_level.height_meters += 0.2
        assert workspace._build_required_scene_atlas_source_ids() == ("surface-b",)
        assert build.call_count == 2

        workspace.current_level.include_in_export = False
        assert workspace._build_required_scene_atlas_source_ids() == ()
        assert build.call_count == 3


def test_pending_live_topology_and_undo_or_load_invalidate_without_viewer_revision(
    workspace,
):
    wall_id = "level:2/wall:1:2"
    saved_levels = copy.deepcopy(workspace.levels)
    revision = workspace._viewer_preview_revision
    with (
        patch.object(
            workspace,
            "_build_atlas_surface_source_ids",
            return_value={wall_id: "surface-a"},
        ),
        patch(
            "housemaker.main.build_fixed_surfaces", wraps=build_fixed_surfaces
        ) as build,
    ):
        assert workspace._build_required_scene_atlas_source_ids() == ("surface-a",)
        workspace.current_level.vertex_data.remove_edge(1, 2)
        assert workspace._viewer_preview_revision == revision
        assert workspace._build_required_scene_atlas_source_ids() == ()
        assert build.call_count == 2
        workspace.levels = copy.deepcopy(saved_levels)
        assert workspace._build_required_scene_atlas_source_ids() == ("surface-a",)
        assert build.call_count == 3


def test_instance_requirement_changes_do_not_rebuild_fixed_surface_geometry(workspace):
    instances = []
    workspace._atlas_available_source_ids = {"chair-source"}
    with (
        patch.object(
            workspace.generation,
            "get_generated_object_instances",
            side_effect=lambda: tuple(instances),
        ),
        patch(
            "housemaker.main.build_fixed_surfaces", wraps=build_fixed_surfaces
        ) as build,
    ):
        assert workspace._build_required_scene_atlas_source_ids() == ()
        instance = SimpleNamespace(
            source_object_id="chair-source",
            placement=SimpleNamespace(level_index=2),
        )
        instances.append(instance)
        assert workspace._build_required_scene_atlas_source_ids() == ("chair-source",)
        instance.placement.level_index = 99
        assert workspace._build_required_scene_atlas_source_ids() == ()
        assert build.call_count == 1


def test_generated_trim_replacement_keys_are_part_of_geometry_cache(workspace):
    with patch(
        "housemaker.main.build_fixed_surfaces", wraps=build_fixed_surfaces
    ) as build:
        workspace._get_atlas_exported_surface_ids(())
        workspace._get_atlas_exported_surface_ids(())
        assert build.call_count == 1
        workspace._get_atlas_exported_surface_ids(((2, "generated-cornice"),))
        assert build.call_count == 2


def test_failed_geometry_rebuild_does_not_cache_partial_or_stale_ids(workspace):
    original_ids = workspace._get_atlas_exported_surface_ids(())
    signature = workspace._atlas_surface_geometry_signature
    workspace.current_level.height_meters += 0.5
    with (
        patch(
            "housemaker.main.build_fixed_surfaces",
            side_effect=ValueError("invalid pending geometry"),
        ),
        pytest.raises(ValueError, match="invalid pending geometry"),
    ):
        workspace._get_atlas_exported_surface_ids(())
    assert workspace._atlas_surface_geometry_signature == signature
    assert workspace._atlas_exported_surface_ids == original_ids
    with patch(
        "housemaker.main.build_fixed_surfaces", wraps=build_fixed_surfaces
    ) as build:
        workspace._get_atlas_exported_surface_ids(())
    build.assert_called_once()


def test_no_trim_requirements_do_not_build_trim_host_geometry(workspace):
    with patch.object(
        workspace,
        "_get_base_fixed_surface_snapshot",
        side_effect=AssertionError("no trims need hosts"),
    ):
        assert workspace._build_active_architectural_trim_source_requirements({2}) == (
            frozenset(),
            (),
        )
