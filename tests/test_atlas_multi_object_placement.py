"""Atlas Place publishes the whole available object selection as one batch."""

# ### Environment and imports ###
from __future__ import annotations

import os
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.texture_atlas_state import ATLAS_TYPE_OBJECT, TextureAtlasData
from housemaker.texture_atlas_workspace import TextureAtlasWorkspace
from tests.test_texture_atlas_workspace import _source, _wall_source

# ### Workspace and pointer helpers ###
_application = QApplication.instance() or QApplication([])
_application.setQuitOnLastWindowClosed(False)


@pytest.fixture
def workspace(tmp_path):
    widget = TextureAtlasWorkspace(asset_directory=tmp_path)
    widget.resize(1100, 720)
    widget.show()
    _application.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    _application.processEvents()


def _click_row(workspace, row, modifiers=Qt.KeyboardModifier.NoModifier):
    source_list = workspace.object_list
    QTest.mouseClick(
        source_list.viewport(),
        Qt.MouseButton.LeftButton,
        modifiers,
        source_list.visualItemRect(source_list.item(row)).center(),
    )


# ### Group publication retains real Ctrl/Shift selection and scene feedback ###
@pytest.mark.parametrize(
    ("modifier", "expected"),
    (
        (Qt.KeyboardModifier.ControlModifier, ("chair", "lamp")),
        (Qt.KeyboardModifier.ShiftModifier, ("chair", "table", "lamp")),
    ),
)
def test_place_publishes_selected_rows_in_order_after_refresh_and_feedback(
    workspace, tmp_path, modifier, expected
):
    sources = tuple(
        _source(key, directory=tmp_path) for key in ("chair", "table", "lamp")
    )
    workspace.set_object_texture_sources(sources)
    groups, singles = Mock(), Mock()
    workspace.objects_place_requested.connect(groups)
    workspace.object_place_requested.connect(singles)
    # Main mirrors group selections into the scene and echoes them back. Such
    # synchronous context synchronization must not collapse the selected rows.
    workspace.object_textures_selected.connect(
        lambda ids: workspace.select_source_ids(
            ids,
            active_source_id=workspace.selected_object_texture_id,
            request_preview=False,
        )
    )
    _click_row(workspace, 0)
    _click_row(workspace, 2, modifier)
    assert workspace.selected_object_texture_ids == expected

    workspace.set_object_texture_sources(sources)
    assert workspace.selected_object_texture_ids == expected
    assert workspace.place_assign_button.isEnabled()
    assert "grid" in workspace.place_assign_button.toolTip()
    workspace.place_assign_button.click()

    groups.assert_called_once_with(expected)
    singles.assert_not_called()
    assert workspace.selected_object_texture_ids == expected


def test_geometry_only_objects_are_included_but_missing_packed_rows_are_not(
    workspace, tmp_path
):
    data = TextureAtlasData()
    atlas = data.create_atlas("Missing assets", 2048, atlas_type=ATLAS_TYPE_OBJECT)
    data.assign_object(atlas.atlas_id, "missing", "missing.png", 512)
    workspace.set_data(data)
    workspace.set_object_texture_sources(
        (_source("chair", directory=tmp_path),),
        placeable_objects={"geometry-only": "Geometry only"},
    )
    workspace.select_source_ids(
        ("missing", "geometry-only", "chair"),
        active_source_id="missing",
        request_preview=False,
    )
    groups, singles = Mock(), Mock()
    workspace.objects_place_requested.connect(groups)
    workspace.object_place_requested.connect(singles)

    assert workspace.place_assign_button.isEnabled()
    workspace.place_assign_button.click()

    groups.assert_called_once_with(("chair", "geometry-only"))
    singles.assert_not_called()
    workspace.select_source_ids(("missing",), request_preview=False)
    assert not workspace.place_assign_button.isEnabled()


@pytest.mark.parametrize("geometry_only", (False, True))
def test_one_available_object_keeps_the_original_single_place_signal(
    workspace, tmp_path, geometry_only
):
    workspace.set_object_texture_sources(
        () if geometry_only else (_source("chair", directory=tmp_path),),
        placeable_objects={"chair": "Chair"} if geometry_only else None,
    )
    groups, singles = Mock(), Mock()
    workspace.objects_place_requested.connect(groups)
    workspace.object_place_requested.connect(singles)

    assert workspace.place_assign_button.isEnabled()
    workspace.place_assign_button.click()

    singles.assert_called_once_with("chair")
    groups.assert_not_called()


def test_surface_texture_selection_still_assigns_and_never_places_a_group(
    workspace, tmp_path
):
    surface = _wall_source("plaster", directory=tmp_path)
    workspace.set_object_texture_sources(
        (_source("chair", directory=tmp_path), surface)
    )
    groups, singles, assigns = Mock(), Mock(), Mock()
    workspace.objects_place_requested.connect(groups)
    workspace.object_place_requested.connect(singles)
    workspace.surface_assign_requested.connect(assigns)
    workspace.surface_list.setCurrentRow(0)

    assert workspace.place_assign_button.text() == "Assign"
    workspace.place_assign_button.click()

    assigns.assert_called_once_with(surface.object_id)
    groups.assert_not_called()
    singles.assert_not_called()
