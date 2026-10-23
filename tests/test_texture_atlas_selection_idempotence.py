# ### Environment ###
from __future__ import annotations

import os
from contextlib import ExitStack
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ### Imports ###
import pytest
from PySide6.QtWidgets import QApplication

from housemaker.texture_atlas_state import TextureAtlasData
from housemaker.texture_atlas_workspace import (
    AtlasSurfaceTextureEntry,
    TextureAtlasWorkspace,
    build_atlas_wall_texture_source_id,
)
from tests import test_texture_atlas_workspace as atlas_fixtures

# ### Fixtures ###
_application = QApplication.instance() or QApplication([])


@pytest.fixture
def workspace(tmp_path):
    widget = TextureAtlasWorkspace(asset_directory=tmp_path)
    yield widget
    widget.close()
    widget.deleteLater()
    _application.processEvents()


def _watch_selection_work(workspace, stack):
    """Count synchronous work without touching a GPU or loading a model."""

    targets = (
        (workspace, "_sync_controls"),
        (workspace, "request_selected_object_preview"),
        (workspace.object_list, "setCurrentRow"),
        (workspace.object_list, "clearSelection"),
        (workspace.object_list, "scrollToItem"),
        (workspace.surface_list, "setCurrentRow"),
        (workspace.surface_list, "clearSelection"),
        (workspace.surface_list, "scrollToItem"),
        *(
            (preview, "set_selected_object_id")
            for preview in workspace.map_previews.values()
        ),
    )
    return tuple(
        stack.enter_context(patch.object(target, name, wraps=getattr(target, name)))
        for target, name in targets
    )


# ### Scene requirements ###
def test_equal_normalized_scene_requirements_skip_list_rebuild(workspace):
    workspace.set_scene_texture_source_ids(("chair", "wall"))
    changed = Mock()
    workspace.data_changed.connect(changed)

    with patch.object(workspace, "_refresh_object_list") as refresh:
        for _ in range(100):
            workspace.set_scene_texture_source_ids(
                (" chair ", "wall", "", "chair", " wall ")
            )

    refresh.assert_not_called()
    changed.assert_not_called()
    assert workspace._scene_texture_source_ids == ("chair", "wall")


def test_scene_requirement_order_and_membership_changes_still_refresh(workspace):
    workspace.set_scene_texture_source_ids(("chair", "wall"))

    with patch.object(workspace, "_refresh_object_list") as refresh:
        workspace.set_scene_texture_source_ids(("wall", "chair"))
        workspace.set_scene_texture_source_ids(("wall",))

    assert refresh.call_count == 2
    assert workspace._scene_texture_source_ids == ("wall",)


# ### Silent selection idempotence ###
def test_repeated_selection_performs_no_list_control_or_preview_work(
    workspace, tmp_path
):
    sources = tuple(
        atlas_fixtures._source(source_id, directory=tmp_path)
        for source_id in ("chair", "table")
    )
    workspace.set_object_texture_sources(sources)
    workspace.mark_sources_new(("chair", "table"))
    assert workspace.select_source_ids(
        ("table", "chair"), active_source_id="table"
    ) == ("chair", "table")
    selection = workspace._last_published_source_selection
    changed = Mock()
    workspace.data_changed.connect(changed)

    with ExitStack() as stack:
        counters = _watch_selection_work(workspace, stack)
        for _ in range(100):
            assert workspace.select_source_ids(
                (" table ", "chair", "chair", "missing"),
                active_source_id=" table ",
            ) == ("chair", "table")

    for counter in counters:
        counter.assert_not_called()
    changed.assert_not_called()
    assert workspace._last_published_source_selection == selection
    assert workspace._new_source_attention_ids == {"chair", "table"}


def test_hidden_selection_updates_ui_without_requesting_or_clearing_preview(
    workspace, tmp_path
):
    source = atlas_fixtures._wall_source("plaster", directory=tmp_path)
    workspace.set_object_texture_sources((source,))
    preview_requests = Mock()
    preview_clears = Mock()
    workspace.object_preview_requested.connect(preview_requests)
    workspace.object_preview_clear_requested.connect(preview_clears)

    with patch.object(workspace, "_sync_controls", wraps=workspace._sync_controls) as sync:
        selected = workspace.select_source_ids(
            (source.object_id,), request_preview=False
        )

    assert selected == (source.object_id,)
    assert workspace.selected_surface_texture_id == source.object_id
    assert workspace.place_assign_button.text() == "Assign"
    sync.assert_called_once()
    assert all(
        preview._selected_object_id == source.object_id
        for preview in workspace.map_previews.values()
    )
    preview_requests.assert_not_called()
    preview_clears.assert_not_called()

    workspace.select_source_ids((), request_preview=False)
    assert workspace.selected_surface_texture_ids == ()
    preview_clears.assert_not_called()


def test_deferred_preview_catches_up_once_without_reselecting_rows(workspace, tmp_path):
    source = atlas_fixtures._source("chair", directory=tmp_path)
    workspace.set_object_texture_sources((source,))
    workspace.select_source_ids(("chair",), request_preview=False)
    requests = Mock()
    workspace.object_preview_requested.connect(requests)

    with (
        patch.object(workspace, "_sync_controls") as sync,
        patch.object(workspace.object_list, "setCurrentRow") as set_row,
        patch.object(workspace.object_list, "clearSelection") as clear,
    ):
        workspace.select_source_ids(("chair",), request_preview=True)
        workspace.select_source_ids(("chair",), request_preview=True)

    requests.assert_called_once_with("chair", 512)
    sync.assert_not_called()
    set_row.assert_not_called()
    clear.assert_not_called()


def test_direct_preview_request_satisfies_deferred_selection(workspace, tmp_path):
    source = atlas_fixtures._source("chair", directory=tmp_path)
    workspace.set_object_texture_sources((source,))
    workspace.select_source_ids(("chair",), request_preview=False)
    requests = Mock()
    workspace.object_preview_requested.connect(requests)

    workspace.request_selected_object_preview()
    workspace.select_source_ids(("chair",), request_preview=True)

    requests.assert_called_once_with("chair", 512)


def test_direct_refresh_after_content_update_does_not_repeat_preview(workspace, tmp_path):
    source = atlas_fixtures._source("chair", directory=tmp_path)
    workspace.set_object_texture_sources((source,))
    workspace.select_source_ids(("chair",))
    workspace.set_object_texture_sources((source,))
    requests = Mock()
    workspace.object_preview_requested.connect(requests)

    workspace.request_selected_object_preview()
    workspace.select_source_ids(("chair",), request_preview=True)
    workspace.select_source_ids(("chair",), request_preview=True)

    requests.assert_called_once_with("chair", 512)


def test_deferred_empty_selection_clears_preview_once(workspace, tmp_path):
    source = atlas_fixtures._source("chair", directory=tmp_path)
    workspace.set_object_texture_sources((source,))
    workspace.select_source_ids(("chair",))
    clears = Mock()
    workspace.object_preview_clear_requested.connect(clears)

    workspace.select_source_ids((), request_preview=False)
    clears.assert_not_called()
    workspace.select_source_ids((), request_preview=True)
    workspace.select_source_ids((), request_preview=True)

    clears.assert_called_once()


@pytest.mark.parametrize("refresh_kind", ("sources", "project"))
def test_content_refresh_invalidates_deferred_preview_marker(
    workspace, tmp_path, refresh_kind
):
    source = atlas_fixtures._source("chair", directory=tmp_path)
    workspace.set_object_texture_sources((source,))
    workspace.select_source_ids(("chair",))
    if refresh_kind == "sources":
        workspace.set_object_texture_sources((source,))
    else:
        workspace.set_data(TextureAtlasData())
    workspace.select_source_ids(("chair",), request_preview=False)
    requests = Mock()
    workspace.object_preview_requested.connect(requests)

    workspace.select_source_ids(("chair",), request_preview=True)
    workspace.select_source_ids(("chair",), request_preview=True)

    requests.assert_called_once_with("chair", 512)


def test_missing_source_becoming_available_does_not_skip_selection(
    workspace, tmp_path
):
    assert workspace.select_source_ids(("future",)) == ()
    source = atlas_fixtures._source("future", directory=tmp_path)
    workspace.set_object_texture_sources((source,))

    with patch.object(workspace, "request_selected_object_preview") as request:
        assert workspace.select_source_ids(("future",)) == ("future",)

    request.assert_called_once()


def test_missing_surface_pixels_becoming_available_refreshes_same_row(
    workspace, tmp_path
):
    source_id = build_atlas_wall_texture_source_id("plaster")
    entry = AtlasSurfaceTextureEntry(source_id, "Plaster", 1)
    workspace.set_object_texture_sources((), surface_texture_entries=(entry,))
    assert workspace.select_source_ids((source_id,)) == (source_id,)
    source = atlas_fixtures._wall_source("plaster", directory=tmp_path)
    workspace.set_object_texture_sources((source,), surface_texture_entries=(entry,))

    with patch.object(workspace, "_sync_controls", wraps=workspace._sync_controls) as sync:
        assert workspace.select_source_ids((source_id,)) == (source_id,)

    sync.assert_called_once()
    assert workspace.place_assign_button.isEnabled()


def test_source_revision_with_same_id_refreshes_preview_once(workspace, tmp_path):
    initial = atlas_fixtures._source("chair", directory=tmp_path)
    workspace.set_object_texture_sources((initial,))
    workspace.select_source_ids(("chair",))
    replacement = atlas_fixtures._source("chair", directory=tmp_path, resolution=1024)
    workspace.set_object_texture_sources((replacement,))
    requests = Mock()
    workspace.object_preview_requested.connect(requests)

    workspace.select_source_ids(("chair",))
    workspace.select_source_ids(("chair",))

    requests.assert_called_once_with("chair", 1024)


def test_project_data_refresh_invalidates_same_source_selection(workspace, tmp_path):
    source = atlas_fixtures._source("chair", directory=tmp_path)
    workspace.set_object_texture_sources((source,))
    workspace.select_source_ids(("chair",))
    workspace.set_data(TextureAtlasData())

    with patch.object(workspace, "request_selected_object_preview") as request:
        workspace.select_source_ids(("chair",))

    request.assert_called_once()


def test_unchanged_request_repairs_modified_selection_and_preview(workspace, tmp_path):
    sources = tuple(
        atlas_fixtures._source(source_id, directory=tmp_path)
        for source_id in ("chair", "table")
    )
    workspace.set_object_texture_sources(sources)
    workspace.select_source_ids(("chair", "table"), active_source_id="table")
    workspace.object_list.item(0).setSelected(False)

    workspace.select_source_ids(("chair", "table"), active_source_id="table")
    assert workspace.selected_object_texture_ids == ("chair", "table")
    next(iter(workspace.map_previews.values())).set_selected_object_id("chair")

    with patch.object(workspace, "_sync_controls", wraps=workspace._sync_controls) as sync:
        workspace.select_source_ids(
            ("chair", "table"), active_source_id="table", request_preview=False
        )

    sync.assert_called_once()
    assert all(
        preview._selected_object_id == "table"
        for preview in workspace.map_previews.values()
    )


def test_repeated_unknown_selection_clears_only_once(workspace, tmp_path):
    source = atlas_fixtures._source("chair", directory=tmp_path)
    workspace.set_object_texture_sources((source,))

    with patch.object(
        workspace,
        "request_selected_object_preview",
        wraps=workspace.request_selected_object_preview,
    ) as request:
        workspace.select_source_ids(("missing",))
        workspace.select_source_ids(("other-missing",))

    request.assert_called_once()
    assert workspace.selected_object_id is None
