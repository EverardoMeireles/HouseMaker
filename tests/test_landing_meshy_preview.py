"""Material-preserving Meshy landing previews in canonical and scene frames."""

# ### Environment ###
from __future__ import annotations

import os
from dataclasses import replace
from io import BytesIO
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ### Imports ###
import numpy as np
import pytest
import trimesh
from PIL import Image
from PySide6.QtWidgets import QApplication

from housemaker.generation_workspace import GenerationWorkspace
from housemaker.glb import (
    GLTF_Y_UP_TO_Z_UP_TRANSFORM,
    Z_UP_TO_GLTF_Y_UP_TRANSFORM,
    convert_to_glb,
)
from housemaker.landing_geometry import (
    LANDING_IMAGE_TO_LOCAL_Z_UP_TRANSFORM,
    build_landing_meshes,
    load_landing_local_meshes,
    prepare_generated_landing,
)
from housemaker.landing_workspace import LandingWorkspaceMixin
from housemaker.models import LandingData, LevelData
from housemaker.viewer import GlbViewerWidget

# ### Fixtures ###
_application = QApplication.instance() or QApplication([])
_application.setQuitOnLastWindowClosed(False)


@pytest.fixture
def generated_landing(tmp_path):
    level = LevelData(index=2, name="Ground", scale=1.7)
    level.offset_x_meters, level.offset_y_meters = 3.0, -2.0
    level.image_size_pixels = (500, 500)
    landing = LandingData(
        100,
        150,
        elevation_meters=1.2,
        rotation_degrees=37,
        scale_x=1.4,
        scale_y=0.7,
        scale_z=2,
    )
    scene = trimesh.Scene()
    for index in range(2):
        mesh = trimesh.creation.box((1, 1.5, 0.3))
        mesh.apply_translation((index - 0.5, 0, 0.15))
        mesh.visual = trimesh.visual.texture.TextureVisuals(
            uv=np.asarray(mesh.vertices[:, :2]) % 1,
            material=trimesh.visual.material.PBRMaterial(
                name=f"part_{index}",
                baseColorTexture=Image.new("RGB", (4, 4), (100 + index * 60, 40, 20)),
                normalTexture=Image.new("RGB", (4, 4), (128, 128, 255)),
                roughnessFactor=0.6,
            ),
        )
        mesh.apply_transform(Z_UP_TO_GLTF_Y_UP_TRANSFORM)
        scene.add_geometry(mesh, geom_name=f"part_{index}", node_name=f"part_{index}")
    source = tmp_path / "meshy.glb"
    source.write_bytes(scene.export(file_type="glb"))
    landing = prepare_generated_landing(landing, source)
    level.landings[:] = [landing]
    return level, landing


def _parts_by_material(parts):
    return {part.visual.material.name: part for part in parts}


# ### Canonical generation previews and texture roundtrips ###
def test_generation_preview_removes_world_transforms_without_losing_materials(
    generated_landing,
):
    level, landing = generated_landing
    owner = SimpleNamespace(levels=[level])
    model = LandingWorkspaceMixin._landing_generation_preview_model(
        owner, level, landing
    )
    originals = _parts_by_material(load_landing_local_meshes(landing))
    for original in originals.values():
        original.apply_transform(LANDING_IMAGE_TO_LOCAL_Z_UP_TRANSFORM)
    assert len(model.scene.geometry) == 2
    for part in model.scene.geometry.values():
        canonical = part.copy()
        canonical.apply_transform(GLTF_Y_UP_TO_Z_UP_TRANSFORM)
        expected = originals[canonical.visual.material.name]
        assert canonical.vertices == pytest.approx(expected.vertices)
        assert canonical.visual.uv == pytest.approx(expected.visual.uv)
        assert canonical.visual.material.normalTexture is not None
        assert canonical.visual.material.roughnessFactor == pytest.approx(0.6)
    expected_vertices = np.vstack([part.vertices for part in originals.values()])
    assert model.mesh.bounds == pytest.approx(
        np.asarray(
            (np.min(expected_vertices, axis=0), np.max(expected_vertices, axis=0))
        )
    )
    assert landing.rotation_degrees == 37
    assert landing.elevation_meters == 1.2


def test_generation_preview_glb_is_meshy_y_up_and_roundtrips_for_texture(
    generated_landing, tmp_path
):
    level, landing = generated_landing
    model = LandingWorkspaceMixin._landing_generation_preview_model(
        SimpleNamespace(levels=[level]), level, landing
    )
    serialized = trimesh.load_scene(
        BytesIO(model.glb_bytes), file_type="glb", process=False
    )
    originals = _parts_by_material(load_landing_local_meshes(landing))
    for original in originals.values():
        original.apply_transform(LANDING_IMAGE_TO_LOCAL_Z_UP_TRANSFORM)
    for part in serialized.geometry.values():
        part.apply_transform(GLTF_Y_UP_TO_Z_UP_TRANSFORM)
        assert part.vertices == pytest.approx(
            originals[part.visual.material.name].vertices
        )
    source = tmp_path / "texture-only.glb"
    source.write_bytes(model.glb_bytes)
    textured = prepare_generated_landing(landing, source, texture_only=True)
    assert textured.footprint_meters == landing.footprint_meters
    assert textured.walk_height_meters == landing.walk_height_meters
    assert textured.thickness_meters == landing.thickness_meters
    assert (textured.scale_x, textured.scale_y, textured.scale_z) == (1.4, 0.7, 2)


def test_generation_tab_asset_refresh_keeps_active_landing_preview(
    generated_landing, tmp_path
):
    level, landing = generated_landing
    model = LandingWorkspaceMixin._landing_generation_preview_model(
        SimpleNamespace(levels=[level]), level, landing
    )
    workspace = GenerationWorkspace(asset_directory=tmp_path / "generation")
    try:
        assert workspace.set_landing_editing_target(landing, model)
        workspace.refresh_file_backed_previews()
        assert workspace.result_view.model is model
        assert workspace.get_landing_editing_target() == landing
        assert workspace.result_view.model_material_items
    finally:
        workspace.shutdown()
        workspace.close()
        workspace.deleteLater()
        _application.processEvents()


# ### Dedicated viewer proxy geometry and materials ###
def test_viewer_landing_proxy_retains_all_textured_parts_and_correct_world_geometry(
    generated_landing,
):
    level, landing = generated_landing
    viewer = GlbViewerWidget(window_editing_enabled=True)
    try:
        viewer.set_model(convert_to_glb([level]))
        viewer.set_landing_context([level])
        group = viewer._placed_object_render_groups[f"landing:{landing.landing_id}"]
        assert len(group.retained_parts) == 2
        assert all(part.textured_item is not None for part in group.retained_parts)
        expected = _parts_by_material(build_landing_meshes(level, landing, [level]))
        for part in group.preview.meshes:
            world = part.copy()
            world.apply_transform(group.current_transform)
            assert world.vertices == pytest.approx(
                expected[world.visual.material.name].vertices
            )
            assert world.visual.material.normalTexture is not None
    finally:
        viewer.close()
        viewer.deleteLater()
        _application.processEvents()


def test_viewer_live_landing_transform_preserves_materials_and_immutable_source(
    generated_landing,
):
    level, landing = generated_landing
    viewer = GlbViewerWidget(window_editing_enabled=True)
    try:
        viewer.set_model(convert_to_glb([level]))
        viewer.set_landing_context([level])
        original = _parts_by_material(load_landing_local_meshes(landing))
        moved = replace(landing, center_x=240, rotation_degrees=82, scale_y=1.3)
        level.landings[0] = moved
        viewer.set_landing_context([level], selected_id=moved.landing_id)
        group = viewer._placed_object_render_groups[f"landing:{landing.landing_id}"]
        expected = _parts_by_material(build_landing_meshes(level, moved, [level]))
        for part in group.preview.meshes:
            world = part.copy()
            world.apply_transform(group.current_transform)
            assert world.vertices == pytest.approx(
                expected[world.visual.material.name].vertices
            )
            assert world.visual.kind == "texture"
        for canonical in load_landing_local_meshes(landing):
            assert canonical.vertices == pytest.approx(
                original[canonical.visual.material.name].vertices
            )
    finally:
        viewer.close()
        viewer.deleteLater()
        _application.processEvents()
