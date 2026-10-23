"""Canvas controls and the complete room/material export stay in sync."""

# ### Environment and imports ###
from __future__ import annotations

import copy
import hashlib
import json
import os
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import trimesh
from PIL import Image
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.glb import GeneratedModel
from housemaker.global_scale import GlobalScale
from housemaker.main import BlueprintWorkspace
from housemaker.material_pack_export import extract_material_packs
from housemaker.room_export_metadata import build_room_asset_manifest
from housemaker.room_zones import build_room_zones
from housemaker.scene_export import _parse_glb, build_runtime_scene_manifest
from housemaker.texture_atlas_state import TextureAtlasData, TextureAtlasRecord
from tests.test_pvs_state import _bake
from tests.test_room_zones import _level

# ### Isolated UI fixture ###
_application = QApplication.instance() or QApplication([])
_application.setQuitOnLastWindowClosed(False)


@pytest.fixture
def workspace(tmp_path):
    widget = BlueprintWorkspace(
        application_settings=ApplicationSettingsStore(tmp_path / "settings.json")
    )
    try:
        yield widget
    finally:
        widget.shutdown()
        widget.close()
        _application.processEvents()


def _textured_model():
    mesh = trimesh.creation.box()
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2)),
        material=trimesh.visual.material.PBRMaterial(
            name="Full-resolution wood",
            baseColorTexture=Image.new("RGBA", (768, 512), (120, 80, 40, 255)),
            normalTexture=Image.new("RGB", (768, 512), (128, 128, 255)),
        ),
    )
    scene = trimesh.Scene(mesh)
    return GeneratedModel(
        mesh=mesh, scene=scene, glb_bytes=scene.export(file_type="glb")
    )


# ### Non-editable room preview ###
def test_show_rooms_is_below_angle_highlights_and_never_edits_geometry(workspace):
    level = _level(divided=True)
    workspace.levels = [level]
    workspace.current_level_index = 0
    workspace._sync_canvas_to_current_level()
    original = copy.deepcopy(level)
    parent_layout = workspace.show_rooms_checkbox.parentWidget().layout()
    assert parent_layout.indexOf(workspace.show_rooms_checkbox) == (
        parent_layout.indexOf(workspace.highlight_non_right_angles_checkbox) + 1
    )
    workspace.show_rooms_checkbox.setChecked(True)
    assert len(workspace.canvas._room_overlays) == 2
    assert workspace.canvas._room_overlays_visible
    assert workspace.levels[0] == original
    workspace.show_rooms_checkbox.setChecked(False)
    assert not workspace.canvas._room_overlays_visible


def test_room_overlay_cache_rebuilds_only_after_structure_changes(workspace):
    workspace.levels = [_level()]
    workspace.current_level_index = 0
    first = workspace._get_room_layout()
    assert workspace._get_room_layout() is first
    workspace.current_level.height_meters += 0.1
    assert workspace._get_room_layout() is not first


def test_publishing_room_atlases_preserves_library_records_and_duplicate_names(
    workspace,
):
    library = TextureAtlasRecord("library", "Room A", 512, atlas_type="object")
    old = TextureAtlasRecord("old", "Original", 512, atlas_type="object")
    workspace.texture_atlas_workspace.set_data(
        TextureAtlasData([library, old], "library")
    )
    planned = TextureAtlasRecord(
        "new", "Room A", 512, atlas_type="object", room_id="room-a", room_name="Room A"
    )
    workspace._last_room_export_plan = SimpleNamespace(
        atlases=(SimpleNamespace(atlas=planned),), warnings=()
    )
    workspace._last_room_export_input_atlas_ids = ("old",)
    previous_visibility = _bake()
    workspace._pvs_visibility = previous_visibility
    workspace._publish_room_export_atlases()
    assert workspace._pvs_visibility == previous_visibility
    first = workspace.texture_atlas_workspace.get_data()
    assert first.atlas_by_id("library") == library
    assert first.atlas_by_id("old") is None
    assert first.atlas_by_id("new").room_id == "room-a"
    assert first.atlas_by_id("new").name != library.name
    assert first.selected_atlas_id == "library"
    workspace._publish_room_export_atlases()
    assert workspace.texture_atlas_workspace.get_data() == first


# ### Runtime metadata is independent of loading policy ###
def test_room_asset_bounds_include_global_scale_once():
    layout = build_room_zones((_level(divided=True),))
    model = _textured_model()
    pack_export = extract_material_packs(model)
    scale = GlobalScale(2, 3, 4, linked=False)
    metadata = build_room_asset_manifest(layout, pack_export, global_scale=scale)
    assert len(metadata["rooms"]) == 2
    assert metadata["portals"][0]["kind"] == "doorway"
    room = layout.rooms[0]
    entry = next(r for r in metadata["rooms"] if r["id"] == room.room_id)
    x0, y0, x1, y1 = room.polygon.bounds
    assert entry["bounds"]["min"] == pytest.approx(
        [x0 * 2, room.floor_z_meters * 4, -y1 * 3]
    )
    assert entry["bounds"]["max"] == pytest.approx(
        [x1 * 2, room.top_z_meters * 4, -y0 * 3]
    )
    manifest = build_runtime_scene_manifest(
        glb_name="house.glb",
        glb_bytes=model.glb_bytes,
        source_placements={},
        instance_placements=(),
        global_scale=scale,
        room_assets=metadata,
    )
    assert manifest["roomAssets"] == metadata
    assert manifest["roomAssets"] is not metadata


# ### Fixed export destination and external full material cubes ###
def test_export_writes_model_json_pvs_and_cube_materials_without_save_dialog(
    workspace, tmp_path
):
    workspace.levels = [_level(divided=True)]
    workspace.current_level_index = 0
    workspace._sync_canvas_to_current_level()
    model = _textured_model()
    unrelated = tmp_path / "public" / "materials" / "user-file.txt"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("keep me", encoding="utf-8")
    with (
        patch.object(workspace, "_sync_atlas_object_texture_sources"),
        patch.object(
            workspace, "_show_unpacked_scene_texture_export_error", return_value=False
        ),
        patch.object(
            workspace, "_show_mixed_scene_atlas_export_error", return_value=False
        ),
        patch.object(
            workspace, "_build_model_with_stable_dependencies", return_value=(model, ())
        ),
        patch.object(workspace, "_build_export_placed_models", return_value=((), ())),
        patch.object(workspace, "_ensure_viewer_preview_current"),
        patch("housemaker.main.QFileDialog.getSaveFileName") as save_dialog,
        patch("housemaker.main.QMessageBox.information"),
        patch("housemaker.main.QMessageBox.critical") as failure,
    ):
        workspace._handle_glb_export_clicked()
    failure.assert_not_called()
    save_dialog.assert_not_called()
    public = tmp_path / "public"
    glb = public / "models" / "housemaker_export.glb"
    assert glb.is_file()
    assert glb.with_suffix(".pvs.mjs").is_file()
    data = json.loads(glb.with_suffix(".json").read_text(encoding="utf-8"))
    assert data["asset"]["sha256"] == hashlib.sha256(glb.read_bytes()).hexdigest()
    assert len(data["roomAssets"]["rooms"]) == 2
    assert len(data["roomAssets"]["materials"]) == 1
    assert unrelated.read_text(encoding="utf-8") == "keep me"
    document, binary = _parse_glb(glb.read_bytes(), "Invalid export")
    for image in document.get("images", []):
        view = document["bufferViews"][image["bufferView"]]
        offset = view.get("byteOffset", 0)
        with Image.open(
            BytesIO(binary[offset : offset + view["byteLength"]])
        ) as preview:
            assert max(preview.size) <= 256
    for binding in data["roomAssets"]["materials"]:
        path = glb.parent / binding["file"]
        assert (
            path.is_file() and path.parent.resolve() == (public / "materials").resolve()
        )
        cube, _ = _parse_glb(path.read_bytes(), "Invalid material cube")
        assert len(cube["meshes"]) == len(cube["materials"]) == 1
        assert hashlib.sha256(path.read_bytes()).hexdigest() == binding["sha256"]
