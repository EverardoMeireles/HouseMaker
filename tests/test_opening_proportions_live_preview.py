"""Opening outline edits render temporary geometry, not committed mesh changes."""

# ### Environment setup ###
from __future__ import annotations

import os
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ### Imports ###
import numpy as np
import pyqtgraph.opengl as gl
import pytest
import trimesh
from PySide6.QtWidgets import QApplication

from housemaker.canvas_openings import (
    CanvasOpeningBounds,
    CanvasOpeningReference,
    CanvasOpeningTarget,
)
from housemaker.glb import GeneratedModel
from housemaker.viewer import (
    CANVAS_OPENING_OVERLAY_DEPTH_VALUE,
    GlbViewerWidget,
    _build_opening_proportions_preview_geometry,
)

# ### Test fixtures ###
_application = QApplication.instance() or QApplication([])
_application.setQuitOnLastWindowClosed(False)


def _target(kind="window", *, arch_amount=None):
    return CanvasOpeningTarget(
        reference=CanvasOpeningReference(kind, 0, 0, "opening-a"),
        wall_surface_id="level:0/wall:1:2",
        plane_start_world=(0.0, 0.0, 0.0),
        wall_tangent_world=(1.0, 0.0, 0.0),
        wall_normal_world=(0.0, -1.0, 0.0),
        wall_width_meters=4.0,
        wall_height_meters=3.0,
        minimum_width_meters=0.2,
        minimum_height_meters=0.3,
        bounds=CanvasOpeningBounds(0.25, 0.75, 0.2, 0.8),
        arch_amount=arch_amount,
    )


@pytest.fixture
def viewer():
    widget = GlbViewerWidget(window_editing_enabled=True)
    mesh = trimesh.creation.box(extents=(4.0, 0.2, 3.0))
    mesh.apply_translation((2.0, 0.0, 1.5))
    widget.set_model(
        GeneratedModel(mesh=mesh, scene=trimesh.Scene(mesh), glb_bytes=b"")
    )
    widget.set_canvas_opening_targets((_target(),))
    yield widget
    widget.exit_first_person_mode()
    widget.close()
    widget.deleteLater()
    _application.processEvents()


def _items(widget):
    return (
        widget._opening_proportions_preview_outline_item,
        widget._opening_proportions_preview_profile_item,
    )


# ### Immediate visible profile and retained updates ###
@pytest.mark.parametrize(
    "kind,arch", [("window", None), ("doorway", None), ("doorway", 0.6)]
)
def test_first_preview_call_renders_filled_opening_sized_geometry(viewer, kind, arch):
    target = _target(kind, arch_amount=arch)
    viewer.set_opening_proportions_preview(target)
    outline, profile = _items(viewer)
    assert isinstance(outline, gl.GLLinePlotItem)
    assert isinstance(profile, gl.GLMeshItem)
    assert outline in viewer.view.items and profile in viewer.view.items
    assert outline.depthValue() >= CANVAS_OPENING_OVERLAY_DEPTH_VALUE
    assert profile.depthValue() >= CANVAS_OPENING_OVERLAY_DEPTH_VALUE
    np.testing.assert_allclose(np.min(outline.pos, axis=0), (1.0, 0.0, 0.6), atol=1e-6)
    np.testing.assert_allclose(np.max(outline.pos, axis=0), (3.0, 0.0, 2.4), atol=1e-6)
    assert 0 < profile.opts["color"][3] < 1
    assert len(profile.opts["meshdata"].faces()) > 0


def test_subsequent_height_changes_update_existing_items_without_accumulating_geometry(
    viewer,
):
    target = _target()
    viewer.set_opening_proportions_preview(target)
    initial_items = _items(viewer)
    count = len(viewer.view.items)
    changed = replace(target, bounds=replace(target.bounds, top_ratio=0.95))
    viewer.set_opening_proportions_preview(changed)
    assert _items(viewer) == initial_items
    assert len(viewer.view.items) == count
    assert float(np.max(initial_items[0].pos[:, 2])) == pytest.approx(2.85)
    assert viewer._opening_proportions_preview_target == changed


def test_doorway_apex_changes_rebuild_the_curved_silhouette_not_just_a_bounding_rectangle(
    viewer,
):
    target = _target("doorway", arch_amount=0.0)
    viewer.set_opening_proportions_preview(target)
    outline, profile = _items(viewer)
    old_positions = outline.pos.copy()
    viewer.set_opening_proportions_preview(replace(target, arch_amount=0.8))
    assert _items(viewer) == (outline, profile)
    assert len(outline.pos) > len(old_positions)
    assert len(np.unique(np.round(outline.pos[:, 2], decimals=5))) > 10
    assert float(np.max(outline.pos[:, 2])) == pytest.approx(2.4)
    assert float(np.min(outline.pos[:, 2])) == pytest.approx(0.6)


def test_back_plane_renders_both_profiles_and_connecting_depth_edges(viewer):
    target = replace(_target(), opposite_wall_plane_world=(0.0, 0.25, 0.0))
    viewer.set_opening_proportions_preview(target)
    outline, profile = _items(viewer)
    np.testing.assert_allclose(np.unique(outline.pos[:, 1]), (0.0, 0.25))
    segments = np.asarray(outline.pos).reshape(-1, 2, 3)
    assert len(segments) == 12
    depth_segments = segments[np.abs(segments[:, 1, 1] - segments[:, 0, 1]) > 0.1]
    assert len(depth_segments) == 4
    np.testing.assert_allclose(
        depth_segments[:, 0, [0, 2]], depth_segments[:, 1, [0, 2]]
    )
    assert len(profile.opts["meshdata"].faces()) == 8


# ### No committed state, selection or camera mutation ###
def test_preview_never_mutates_model_targets_selection_or_fps_camera(viewer):
    before_targets = viewer._canvas_opening_targets.copy()
    before_vertices = viewer.model.mesh.vertices.copy()
    before_faces = viewer.model.mesh.faces.copy()
    selection_changes = []
    edits = []
    viewer.canvas_opening_selection_changed.connect(selection_changes.append)
    viewer.canvas_opening_edit_preview_changed.connect(edits.append)
    viewer.enter_first_person_mode()
    pose = viewer.get_first_person_camera_pose()
    selected = viewer.get_selected_canvas_opening_reference()
    target = replace(_target(), bounds=CanvasOpeningBounds(0.25, 0.75, 0.1, 0.95))
    viewer.set_opening_proportions_preview(target)
    assert viewer._canvas_opening_targets == before_targets
    assert viewer.get_selected_canvas_opening_reference() == selected
    assert viewer.get_first_person_camera_pose() == pose
    assert viewer.is_first_person_pointer_captured
    assert selection_changes == [] and edits == []
    np.testing.assert_array_equal(viewer.model.mesh.vertices, before_vertices)
    np.testing.assert_array_equal(viewer.model.mesh.faces, before_faces)


def test_invalid_preview_type_does_not_replace_the_last_good_geometry(viewer):
    target = _target()
    viewer.set_opening_proportions_preview(target)
    items = _items(viewer)
    with pytest.raises(TypeError):
        viewer.set_opening_proportions_preview(object())
    assert _items(viewer) == items
    assert viewer._opening_proportions_preview_target == target


# ### Preview cleanup ###
def test_clear_is_idempotent_and_removes_only_temporary_geometry(viewer):
    original_items = tuple(viewer.view.items)
    viewer.set_opening_proportions_preview(_target())
    preview_items = _items(viewer)
    viewer.clear_opening_proportions_preview()
    viewer.clear_opening_proportions_preview()
    assert _items(viewer) == (None, None)
    assert viewer._opening_proportions_preview_target is None
    assert all(item not in viewer.view.items for item in preview_items)
    assert all(item in viewer.view.items for item in original_items)


def test_opening_target_reset_discards_the_obsolete_preview(viewer):
    viewer.set_opening_proportions_preview(_target())
    preview_items = _items(viewer)
    viewer.set_canvas_opening_targets(())
    assert _items(viewer) == (None, None)
    assert viewer._opening_proportions_preview_target is None
    assert all(item not in viewer.view.items for item in preview_items)


def test_scene_clear_discards_preview_state_and_items(viewer):
    viewer.set_opening_proportions_preview(_target())
    viewer.clear_model()
    assert _items(viewer) == (None, None)
    assert viewer._opening_proportions_preview_target is None


# ### Deterministic filled profile geometry ###
@pytest.mark.parametrize("arch", [None, 0.0, 0.3, 1.0])
def test_preview_fill_has_no_degenerate_triangles_and_matches_its_polygon_area(arch):
    outline, vertices, faces = _build_opening_proportions_preview_geometry(
        _target("doorway", arch_amount=arch)
    )
    triangles = vertices[faces]
    areas = (
        np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            ),
            axis=1,
        )
        / 2
    )
    assert np.all(areas > 1e-10)
    polygon = outline.reshape(-1, 2, 3)[:, 0, :][:, [0, 2]]
    polygon_area = (
        abs(
            np.sum(
                polygon[:, 0] * np.roll(polygon[:, 1], -1)
                - polygon[:, 1] * np.roll(polygon[:, 0], -1)
            )
        )
        / 2
    )
    assert float(np.sum(areas)) == pytest.approx(polygon_area)
