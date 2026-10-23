"""Pending surface materials use independent, geometry-only face overlays."""

# ### Environment setup ###
from __future__ import annotations

import os
from dataclasses import replace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import numpy as np
import pytest
import trimesh
from OpenGL import GL
from PIL import Image
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication

from housemaker.architectural_trim import (
    ArchitecturalTrimPart,
    build_architectural_trim_surface_id,
)
from housemaker.glb import (
    GeneratedModel,
    PreviewStairPart,
    PreviewTexturedSurface,
    PreviewTexturedWall,
)
from housemaker.surface_geometry import FixedSurface
from housemaker.viewer import (
    PENDING_SURFACE_TEXTURE_PREVIEW_COLOR,
    TEXTURE_PREVIEW_OFFSET_METERS,
    GlbViewerWidget,
)

# ### Fixtures ###
_application = QApplication.instance() or QApplication([])
_application.setQuitOnLastWindowClosed(False)


def _surface(surface_id="level:2/wall:1:2", *, level_index=2, **options):
    mesh = trimesh.Trimesh(
        vertices=((0, 0, 0), (2, 0, 0), (2, 0, 2), (0, 0, 2)),
        faces=((0, 1, 2), (0, 2, 3)),
        process=False,
    )
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.asarray(((0, 0), (1, 0), (1, 1), (0, 1))),
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.new("RGBA", (2, 2), (20, 40, 200, 255)),
        ),
    )
    return FixedSurface(
        surface_id=surface_id,
        surface_type="wall",
        level_index=level_index,
        room_index=None,
        mesh=mesh,
        area_square_meters=4.0,
        wall_key="1:2",
        wall_start_world=(0.0, 0.0, 0.0),
        wall_end_world=(2.0, 0.0, 0.0),
        wall_height_meters=2.0,
        **options,
    )


def _model(surface):
    mesh = surface.mesh.copy()
    return GeneratedModel(mesh=mesh, scene=trimesh.Scene(mesh), glb_bytes=b"")


def _vertices(item):
    return item.opts["meshdata"].vertexes()


@pytest.fixture
def viewer():
    widget = GlbViewerWidget(window_editing_enabled=True)
    yield widget
    widget.close()
    widget.deleteLater()
    _application.processEvents()


# ### Overlay ownership and independence ###
def test_pending_faces_are_transparent_yellow_geometry_without_cloning_textures(viewer):
    surface = _surface()
    original_vertices = surface.mesh.vertices.copy()
    original_faces = surface.mesh.faces.copy()
    original_image = surface.mesh.visual.material.baseColorTexture
    viewer.set_wall_targets((surface,))

    with patch.object(surface.mesh, "copy", side_effect=AssertionError("mesh clone")):
        assert viewer.set_pending_surface_texture_preview_ids((surface.surface_id,))

    item = viewer._pending_surface_texture_preview_items[surface.surface_id]
    assert viewer.model is None
    assert item in viewer.view.items
    assert item.opts["drawFaces"] is True
    assert item.opts["drawEdges"] is False
    assert item.opts["color"] == PENDING_SURFACE_TEXTURE_PREVIEW_COLOR
    assert 0.0 < item.opts["color"][3] < 1.0
    np.testing.assert_array_equal(_vertices(item), original_vertices)
    np.testing.assert_array_equal(item.opts["meshdata"].faces(), original_faces)
    assert not np.shares_memory(_vertices(item), surface.mesh.vertices)
    assert surface.mesh.visual.material.baseColorTexture is original_image
    np.testing.assert_array_equal(surface.mesh.vertices, original_vertices)
    np.testing.assert_array_equal(surface.mesh.faces, original_faces)
    options = item._GLGraphicsItem__glOpts
    assert options[GL.GL_DEPTH_TEST] is True
    assert options[GL.GL_BLEND] is True
    assert options[GL.GL_CULL_FACE] is False
    assert options[GL.GL_POLYGON_OFFSET_FILL] is True
    assert options["glDepthMask"] == (False,)


def test_pending_preview_does_not_change_selection_highlights_or_model(viewer):
    surface = _surface()
    model = _model(surface)
    viewer.set_model(model)
    viewer.set_wall_targets((surface,))
    viewer.select_wall_target(surface.surface_id)
    viewer.set_highlighted_canvas_surface_ids((surface.surface_id,))
    selected = viewer.get_selected_canvas_surface_ids()
    highlight_items = tuple(viewer._atlas_surface_highlight_items)
    selection_changes = QSignalSpy(viewer.canvas_surface_selection_changed)
    scene_geometry = dict(model.scene.geometry)

    with patch.object(viewer, "_populate_scene", side_effect=AssertionError("rebuild")):
        viewer.set_pending_surface_texture_preview_ids((surface.surface_id,))
        viewer.clear_pending_surface_texture_previews()

    assert viewer.get_selected_canvas_surface_ids() == selected
    assert viewer.get_highlighted_canvas_surface_ids() == (surface.surface_id,)
    assert tuple(viewer._atlas_surface_highlight_items) == highlight_items
    assert selection_changes.count() == 0
    assert viewer.model is model
    assert dict(model.scene.geometry) == scene_geometry


def test_multiple_ids_are_idempotent_case_exact_and_unknown_targets_render_nothing(viewer):
    wall = _surface()
    reveal_id = "level:2/window:Window%2FCase/reveal:header"
    reveal = _surface(reveal_id)
    viewer.set_wall_targets((wall, reveal))

    assert viewer.set_pending_surface_texture_preview_ids(
        (wall.surface_id, reveal_id, wall.surface_id, "missing", "")
    )
    assert viewer.get_pending_surface_texture_preview_ids() == (
        wall.surface_id, reveal_id, "missing",
    )
    assert set(viewer._pending_surface_texture_preview_items) == {wall.surface_id, reveal_id}
    original_items = dict(viewer._pending_surface_texture_preview_items)
    assert not viewer.set_pending_surface_texture_preview_ids(
        (wall.surface_id, reveal_id, "missing")
    )
    assert viewer._pending_surface_texture_preview_items == original_items
    assert viewer.clear_pending_surface_texture_previews()
    assert not viewer.clear_pending_surface_texture_previews()
    assert not viewer._pending_surface_texture_preview_items
    assert viewer.get_pending_surface_texture_preview_ids() == ()


def test_source_and_alias_ids_cover_each_matching_fragment_once(viewer):
    surface = _surface(
        "level:2/wall:fragment",
        source_surface_id="level:2/wall:source",
        surface_alias_ids=("level:2/wall:old",),
    )
    viewer.set_wall_targets((surface,))
    viewer.set_pending_surface_texture_preview_ids(
        (surface.surface_id, surface.source_surface_id, "level:2/wall:old")
    )
    assert tuple(viewer._pending_surface_texture_preview_items) == (surface.surface_id,)


@pytest.mark.parametrize("paint_error", (False, True))
def test_overlay_restores_depth_state_even_when_drawing_fails(viewer, paint_error):
    surface = _surface()
    viewer.set_wall_targets((surface,))
    viewer.set_pending_surface_texture_preview_ids((surface.surface_id,))
    item = viewer._pending_surface_texture_preview_items[surface.surface_id]
    with (
        patch("housemaker.viewer.gl.GLMeshItem.paint",
              side_effect=RuntimeError("draw failed") if paint_error else None),
        patch("housemaker.viewer.GL.glDepthFunc") as depth_function,
        patch("housemaker.viewer.GL.glDisable") as disable,
        patch("housemaker.viewer.GL.glDepthMask") as depth_mask,
    ):
        if paint_error:
            with pytest.raises(RuntimeError, match="draw failed"):
                item.paint()
        else:
            item.paint()
    assert [call.args for call in depth_function.call_args_list] == [
        (GL.GL_LEQUAL,), (GL.GL_LESS,),
    ]
    disable.assert_called_once_with(GL.GL_POLYGON_OFFSET_FILL)
    depth_mask.assert_called_once_with(True)


# ### Rebuild, deletion and visibility lifecycle ###
def test_geometry_snapshots_refresh_and_survive_model_replacement_until_explicit_clear(viewer):
    surface = _surface()
    viewer.set_wall_targets((surface,))
    viewer.set_pending_surface_texture_preview_ids((surface.surface_id,))
    original_item = viewer._pending_surface_texture_preview_items[surface.surface_id]
    original_vertices = _vertices(original_item).copy()
    surface.mesh.vertices[:, 0] += 3.0
    np.testing.assert_array_equal(_vertices(original_item), original_vertices)

    viewer.set_wall_targets((surface,))
    refreshed = viewer._pending_surface_texture_preview_items[surface.surface_id]
    assert refreshed is not original_item
    assert original_item not in viewer.view.items
    np.testing.assert_allclose(_vertices(refreshed), surface.mesh.vertices)
    viewer.set_model(_model(surface), preserve_camera=True)
    assert viewer.get_pending_surface_texture_preview_ids() == (surface.surface_id,)
    assert surface.surface_id in viewer._pending_surface_texture_preview_items

    viewer.set_wall_targets(())
    assert not viewer._pending_surface_texture_preview_items
    assert viewer.get_pending_surface_texture_preview_ids() == (surface.surface_id,)
    viewer.clear_pending_surface_texture_previews()
    assert viewer.get_pending_surface_texture_preview_ids() == ()


def test_visibility_filters_hide_and_restore_pending_faces_without_losing_ids(viewer):
    surface = _surface()
    viewer.set_wall_targets((surface,))
    viewer.set_canvas_scene_levels(((2, "Ground"),))
    viewer.set_pending_surface_texture_preview_ids((surface.surface_id,))
    assert surface.surface_id in viewer._pending_surface_texture_preview_items
    viewer.set_visible_canvas_level_indices(())
    assert not viewer._pending_surface_texture_preview_items
    viewer.set_visible_canvas_level_indices((2,))
    assert surface.surface_id in viewer._pending_surface_texture_preview_items
    viewer.set_canvas_surface_focus_type("floor")
    assert not viewer._pending_surface_texture_preview_items
    viewer.set_canvas_surface_focus_type(None)
    assert surface.surface_id in viewer._pending_surface_texture_preview_items


@pytest.mark.parametrize("boundary", ("clear_model", "close"))
def test_clear_or_close_removes_pending_state_and_items(viewer, boundary):
    surface = _surface()
    viewer.set_wall_targets((surface,))
    viewer.set_pending_surface_texture_preview_ids((surface.surface_id,))
    item = viewer._pending_surface_texture_preview_items[surface.surface_id]
    getattr(viewer, boundary)()
    assert viewer.get_pending_surface_texture_preview_ids() == ()
    assert not viewer._pending_surface_texture_preview_items
    assert item not in viewer.view.items


# ### Stair and architectural trim semantic surfaces ###
def test_stair_and_trim_faces_use_the_same_pending_api_and_refresh_with_catalogs(viewer):
    mesh = _surface().mesh
    stair = PreviewStairPart(
        stair_id="stairs-a", stair_index=0,
        semantic_id="stair:stairs-a/part:treads:floor",
        part_kind="treads", surface_type="floor", mesh=mesh, level_indices=(2,),
    )
    trim_id = "a" * 32
    trim = ArchitecturalTrimPart(
        trim_id=trim_id, run_id="trim-run",
        semantic_id=build_architectural_trim_surface_id(trim_id, "front"),
        part_kind="front", surface_type="wall", mesh=mesh, level_index=2,
    )
    viewer.set_pending_surface_texture_preview_ids((stair.semantic_id, trim.semantic_id))
    viewer.set_canvas_stair_part_targets((stair,))
    viewer.set_architectural_trim_parts((trim,))
    assert set(viewer._pending_surface_texture_preview_items) == {
        stair.semantic_id, trim.semantic_id,
    }
    shifted = mesh.copy()
    shifted.apply_translation((0.0, 2.0, 0.0))
    viewer.set_architectural_trim_parts((replace(trim, mesh=shifted),))
    np.testing.assert_allclose(
        _vertices(viewer._pending_surface_texture_preview_items[trim.semantic_id]),
        shifted.vertices,
    )
    viewer.set_canvas_stair_part_targets(())
    viewer.set_architectural_trim_parts(())
    assert not viewer._pending_surface_texture_preview_items


# ### Legacy wall texture planes cannot conceal the pending tint ###
def test_yellow_preview_follows_only_the_owning_legacy_wall_texture_plane(viewer):
    surface = replace(_surface(), room_index=0)
    model = _model(surface)
    model.preview_textured_walls = [PreviewTexturedWall(
        level_index=surface.level_index, room_index=surface.room_index,
        wall_key=surface.wall_key, start_point=(0, 0, 0), end_point=(2, 0, 0),
        height_meters=2.0, texture_rgba=np.full((2, 2, 4), 255, dtype=np.uint8),
    )]
    viewer.set_model(model)
    viewer.set_wall_targets((surface,))
    viewer.set_pending_surface_texture_preview_ids((surface.surface_id,))
    item = viewer._pending_surface_texture_preview_items[surface.surface_id]
    np.testing.assert_allclose(
        _vertices(item), surface.mesh.vertices + (0, -TEXTURE_PREVIEW_OFFSET_METERS, 0),
    )
    # This remains a depth-tested tint, not an overlay through unrelated walls.
    assert item._GLGraphicsItem__glOpts[GL.GL_DEPTH_TEST] is True
    assert item._GLGraphicsItem__glOpts["glDepthMask"] == (False,)

    viewer.set_textures_enabled(False)
    np.testing.assert_allclose(
        _vertices(viewer._pending_surface_texture_preview_items[surface.surface_id]),
        surface.mesh.vertices,
    )
    viewer.set_textures_enabled(True)
    viewer.set_wireframe_only(True)
    np.testing.assert_allclose(
        _vertices(viewer._pending_surface_texture_preview_items[surface.surface_id]),
        surface.mesh.vertices,
    )
    viewer.set_wireframe_only(False)
    assert not np.allclose(
        _vertices(viewer._pending_surface_texture_preview_items[surface.surface_id]),
        surface.mesh.vertices,
    )
    np.testing.assert_allclose(surface.mesh.vertices[:, 1], 0)


def test_generated_surface_texture_does_not_receive_the_legacy_wall_offset(viewer):
    surface = replace(_surface(), room_index=0)
    model = _model(surface)
    model.preview_textured_walls = [PreviewTexturedWall(
        level_index=surface.level_index, room_index=surface.room_index,
        wall_key=surface.wall_key, start_point=(0, 0, 0), end_point=(2, 0, 0),
        height_meters=2.0, texture_rgba=np.full((2, 2, 4), 255, dtype=np.uint8),
    )]
    model.preview_textured_surfaces = [PreviewTexturedSurface(
        surface_id=surface.surface_id, surface_type="wall", mesh=surface.mesh,
        level_index=surface.level_index, room_index=surface.room_index,
        wall_key=surface.wall_key,
    )]
    viewer.set_model(model)
    viewer.set_wall_targets((surface,))
    viewer.set_pending_surface_texture_preview_ids((surface.surface_id,))
    np.testing.assert_allclose(
        _vertices(viewer._pending_surface_texture_preview_items[surface.surface_id]),
        surface.mesh.vertices,
    )
