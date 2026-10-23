"""Room sections decorate real Atlas rows without changing their identity."""

# ### Environment and imports ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLineEdit, QStyleOptionViewItem

from housemaker.texture_atlas_state import (
    ATLAS_TYPE_OBJECT,
    ATLAS_TYPE_SURFACE,
    TextureAtlasData,
)
from housemaker.texture_atlas_workspace import (
    ATLAS_ID_ROLE,
    ATLAS_SECTION_LABEL_ROLE,
    TextureAtlasWorkspace,
)

# ### Workspace and deliberately interleaved group fixtures ###
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


def _data():
    data = TextureAtlasData()
    for identifier, atlas_type, room_id, room_name in (
        ("shared-a", ATLAS_TYPE_OBJECT, None, None),
        ("room-b", ATLAS_TYPE_OBJECT, "room:2:b", "L2 — Room 2"),
        ("surface", ATLAS_TYPE_SURFACE, None, None),
        ("room-a-first", ATLAS_TYPE_OBJECT, "room:2:a", "L2 — Room 1"),
        ("room-a-second", ATLAS_TYPE_OBJECT, "room:2:a", "L2 — Room 1"),
        ("shared-b", ATLAS_TYPE_OBJECT, None, None),
    ):
        atlas = data.create_atlas(
            identifier, 2048, atlas_id=identifier, atlas_type=atlas_type
        )
        atlas.room_id, atlas.room_name = room_id, room_name
    data.selected_atlas_id = "room-a-second"
    return data


def _ids(workspace):
    return [
        workspace.atlas_list.item(row).data(ATLAS_ID_ROLE)
        for row in range(workspace.atlas_list.count())
    ]


# ### Grouping keeps every row selectable and renameable ###
def test_sections_are_contiguous_real_rows_and_preserve_selection(workspace):
    data = _data()
    workspace.set_data(data)
    assert workspace.atlas_list.count() == len(data.atlases)
    assert _ids(workspace) == [
        "surface",
        "room-a-first",
        "room-a-second",
        "room-b",
        "shared-a",
        "shared-b",
    ]
    assert [
        workspace.atlas_list.item(row).data(ATLAS_SECTION_LABEL_ROLE)
        for row in range(workspace.atlas_list.count())
    ] == [
        "Surface atlases",
        "L2 — Room 1",
        None,
        "L2 — Room 2",
        "Shared / unassigned object atlases",
        None,
    ]
    assert workspace.selected_atlas.atlas_id == "room-a-second"
    assert workspace.atlas_list.currentRow() == 2
    workspace.atlas_list.setCurrentRow(3)
    assert workspace.selected_atlas.atlas_id == "room-b"
    workspace._refresh_atlas_list("room-b")
    assert workspace.atlas_list.currentRow() == 3


def test_heading_height_is_only_added_to_first_group_row_and_paints(workspace):
    workspace.set_data(_data())
    delegate = workspace.atlas_list.itemDelegate()
    option = QStyleOptionViewItem()
    option.fontMetrics = workspace.atlas_list.fontMetrics()
    headed = workspace.atlas_list.model().index(1, 0)
    unheaded = workspace.atlas_list.model().index(2, 0)
    assert (
        delegate.sizeHint(option, headed).height()
        > delegate.sizeHint(option, unheaded).height()
    )
    rendered = QImage(workspace.size(), QImage.Format.Format_ARGB32)
    rendered.fill(Qt.GlobalColor.transparent)
    workspace.render(rendered)
    assert not rendered.isNull()


def test_inline_rename_changes_only_atlas_name_not_group_identity(workspace):
    workspace.set_data(_data())
    source_list = workspace.atlas_list
    source_list.setCurrentRow(1)
    source_list.setFocus()
    QTest.keyClick(source_list, Qt.Key.Key_F2)
    editor = source_list.findChild(QLineEdit)
    assert editor is not None
    editor.setText("Renamed furniture")
    QTest.keyClick(editor, Qt.Key.Key_Return)
    _application.processEvents()
    assert workspace.selected_atlas.atlas_id == "room-a-first"
    assert workspace.selected_atlas.name == "[OBJECT] Renamed furniture"
    assert source_list.currentItem().data(ATLAS_SECTION_LABEL_ROLE) == "L2 — Room 1"
    assert _ids(workspace) == [
        "surface",
        "room-a-first",
        "room-a-second",
        "room-b",
        "shared-a",
        "shared-b",
    ]
