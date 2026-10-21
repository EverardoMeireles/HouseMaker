"""Persistent polygon landings, export and dynamically attached stair flights."""

# ### Imports ###
from __future__ import annotations

import json
import math
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image
from shapely.geometry import Point, Polygon

from housemaker.floor_geometry import build_level_floor_footprint
from housemaker.glb import (
    _resolve_stair_route_sections,
    _split_landing_stair_flights,
    build_stair_meshes,
    convert_to_glb,
)
from housemaker.landing_geometry import (
    attach_stair_point_to_landing,
    build_landing_mesh,
    build_landing_meshes,
    create_landing,
    find_landing_snap,
    landing_attachment_for_image_point,
    landing_top_world_z,
    landing_world_footprint,
    landing_world_transform,
    load_landing_local_meshes,
    prepare_generated_landing,
    resolve_landing_attachment,
    resolve_stair_section_image_points,
)
from housemaker.level_coordinates import (
    build_level_base_z_lookup,
    level_image_to_world_xy,
)
from housemaker.models import (
    LandingAttachment,
    LandingData,
    LevelData,
    StairData,
    StairSectionData,
)
from housemaker.project_io import load_project, save_project


# ### Fixture helpers ###
def _scene():
    ground, upper = LevelData(2, "Ground", height_meters=3.0), LevelData(3, "Story 1")
    landing = LandingData(100.0, 100.0, elevation_meters=1.2, thickness_meters=0.2)
    ground.landings.append(landing)
    return [ground, upper], ground, upper, landing


def _stair_with_landing():
    levels, ground, upper, landing = _scene()
    sections = (
        StairSectionData(2, 65.0, 80.0, 65.0, 120.0),
        StairSectionData(2, 135.0, 80.0, 135.0, 120.0),
    )
    stair = StairData(
        start_level_index=2,
        end_level_index=3,
        start_a_x=0.0,
        start_a_y=80.0,
        start_b_x=0.0,
        start_b_y=120.0,
        end_a_x=200.0,
        end_a_y=80.0,
        end_b_x=200.0,
        end_b_y=120.0,
        intermediate_sections=sections,
        stringer_placement="none",
        tread_thickness_meters=0.06,
        tread_overhang_meters=0.0,
    )
    for index in (1, 2):
        for name in ("a", "b"):
            stair = attach_stair_point_to_landing(stair, index, name, ground, landing)
    return levels, ground, upper, landing, stair


# ### Models and project persistence ###
def test_default_landing_is_a_real_square_slab_resting_on_floor():
    level = LevelData(2, "Ground")
    landing = create_landing(level, 100.0, 200.0)
    assert level.landings == []
    assert Polygon(landing.footprint_meters).area == pytest.approx(2.25)
    assert landing.elevation_meters == 0.0
    assert landing.thickness_meters == 0.15
    assert LandingData.from_dict(landing.to_dict()) == landing
    assert create_landing(level, 100, 200).landing_id != landing.landing_id


@pytest.mark.parametrize(
    "changes",
    [
        {"thickness_meters": 0},
        {"scale_x": -1},
        {"center_x": float("nan")},
        {"footprint_meters": ((0, 0), (1, 1), (0, 1), (1, 0))},
        {"footprint_meters": ((0, 0), (1, 0), (2, 0))},
        {"landing_id": "not-a-uuid"},
        {"generated_model_path": ""},
        {"walk_height_meters": 0},
    ],
)
def test_invalid_landing_models_are_rejected(changes):
    with pytest.raises(ValueError):
        LandingData(**{"center_x": 0, "center_y": 0, **changes})


def test_landing_and_all_stair_attachment_fields_survive_project_roundtrip(tmp_path):
    levels, ground, _, landing, stair = _stair_with_landing()
    stair = attach_stair_point_to_landing(stair, 0, "a", ground, landing, (65, 80))
    path = tmp_path / "project.json"
    save_project(path, 0, levels, stairs=[stair])
    restored = load_project(path)
    assert next(level for level in restored.levels if level.index == 2).landings == [
        landing
    ]
    assert restored.stairs == [stair]
    assert restored.stairs[0].intermediate_sections[0].a_landing is not None
    payload = json.loads(path.read_text())
    for level in payload["levels"]:
        level.pop("landings", None)
    # Ordinary old projects without the new collection remain valid.
    payload["stairs"] = []
    path.write_text(json.dumps(payload))
    assert all(not level.landings for level in load_project(path).levels)


def test_attachment_and_section_json_roundtrip():
    _, ground, _, landing = _scene()
    attachment = landing_attachment_for_image_point(ground, landing, (65, 80))
    assert LandingAttachment.from_dict(attachment.to_dict()) == attachment
    section = StairSectionData(2, 65, 80, 65, 120, a_landing=attachment)
    assert StairSectionData.from_dict(section.to_dict()) == section


# ### Actual polygon geometry and transforms ###
def test_concave_polygon_is_extruded_without_filling_its_missing_corner():
    levels, ground, _, landing = _scene()
    landing = replace(
        landing, footprint_meters=((0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2))
    )
    mesh = build_landing_mesh(ground, landing, levels)
    assert mesh.is_watertight
    assert mesh.is_volume
    assert mesh.volume == pytest.approx(3.0 * 0.2)
    assert Polygon(landing_world_footprint(ground, landing)).area == pytest.approx(3.0)
    assert mesh.metadata["landing_id"] == landing.landing_id


def test_world_transform_matches_actual_polygon_after_all_transform_edits():
    levels, ground, _, landing = _scene()
    ground.scale = 1.5
    ground.offset_x_meters, ground.offset_y_meters = 2.0, -3.0
    landing = replace(
        landing,
        center_x=130,
        center_y=70,
        rotation_degrees=37,
        scale_x=1.8,
        scale_y=0.7,
        scale_z=2.0,
    )
    matrix = landing_world_transform(ground, landing, levels)
    local = np.asarray([(x, y, 0.0, 1.0) for x, y in landing.footprint_meters])
    transformed = local @ matrix.T
    assert transformed[:, :2] == pytest.approx(
        np.asarray(landing_world_footprint(ground, landing))
    )
    mesh = build_landing_mesh(ground, landing, levels)
    expected_top = (
        build_level_base_z_lookup(levels)[ground.index] + landing.elevation_meters + 0.4
    )
    assert mesh.bounds[1, 2] == pytest.approx(expected_top)
    assert landing_top_world_z(ground, landing, levels) == pytest.approx(expected_top)


def test_attachment_tracks_translation_rotation_and_individual_scale():
    levels, ground, _, landing = _scene()
    attachment = landing_attachment_for_image_point(ground, landing, (65, 80))
    ground.landings[0] = moved = replace(
        landing, center_x=150, center_y=200, rotation_degrees=90, scale_x=2, scale_y=0.5
    )
    point, z = resolve_landing_attachment(ground, attachment, levels)
    assert point == pytest.approx((160, 130))
    assert z == pytest.approx(landing_top_world_z(ground, moved, levels))


def test_regenerated_shape_projects_attachment_to_actual_profile_without_rewriting_anchor():
    levels, ground, _, landing = _scene()
    attachment = landing_attachment_for_image_point(ground, landing, (65, 80))
    ground.landings[0] = replace(
        landing, footprint_meters=((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5))
    )
    assert resolve_landing_attachment(ground, attachment, levels)[0] == pytest.approx(
        (75, 80)
    )
    assert attachment.local_x_meters == pytest.approx(-0.7)
    ground.landings[0] = landing
    assert resolve_landing_attachment(ground, attachment, levels)[0] == pytest.approx(
        (65, 80)
    )


def test_attachment_elevation_tracks_floor_top_bottom_height_and_slab_thickness():
    levels, ground, upper, landing = _scene()
    attachment = landing_attachment_for_image_point(ground, landing, (65, 80))
    original = resolve_landing_attachment(ground, attachment, levels)[1]
    ground.floor_thickness_meters += 0.1
    assert resolve_landing_attachment(ground, attachment, levels)[1] == pytest.approx(
        original + 0.1
    )
    ground.landings[0] = replace(
        landing, elevation_meters=1.6, thickness_meters=0.4, scale_z=2.0
    )
    assert resolve_landing_attachment(ground, attachment, levels)[1] == pytest.approx(
        original + 1.1
    )
    upper_landing = replace(landing, center_x=200)
    upper.landings.append(upper_landing)
    upper_attachment = landing_attachment_for_image_point(
        upper, upper_landing, (165, 80)
    )
    before = resolve_landing_attachment(upper, upper_attachment, levels)[1]
    ground.height_meters += 0.5
    assert resolve_landing_attachment(upper, upper_attachment, levels)[
        1
    ] == pytest.approx(before + 0.5)


def test_snap_uses_polygon_boundary_not_the_bounding_box():
    _, ground, _, landing = _scene()
    ground.landings[0] = replace(
        landing, footprint_meters=((0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2))
    )
    assert find_landing_snap(ground, (199, 199), 2.0) is None
    snapped, attachment = find_landing_snap(ground, (151, 175), 2.0)
    assert snapped == pytest.approx((150, 175))
    assert attachment.landing_id == landing.landing_id


# ### Piecewise landing flights and exports ###
def test_intermediate_platform_creates_two_flights_and_no_steps_across_platform():
    levels, ground, _, landing, stair = _stair_with_landing()
    flights = _split_landing_stair_flights(
        stair, {level.index: level for level in levels}
    )
    assert len(flights) == 2
    assert flights[0].end_a_landing.landing_id == landing.landing_id
    assert flights[1].start_a_landing.landing_id == landing.landing_id
    meshes = build_stair_meshes(levels, [stair])
    assert meshes
    vertices = np.vstack([named.mesh.vertices for named in meshes])
    assert not np.any((vertices[:, 0] > 1.4) & (vertices[:, 0] < 2.6))
    assert np.any(
        np.isclose(vertices[:, 2], landing_top_world_z(ground, landing, levels))
    )


def test_same_level_stair_is_valid_when_landing_gives_distinct_elevation():
    levels, ground, _, landing, original = _stair_with_landing()
    endpoint = original.intermediate_sections[0]
    stair = replace(
        original,
        end_level_index=ground.index,
        end_a_x=endpoint.a_x,
        end_a_y=endpoint.a_y,
        end_b_x=endpoint.b_x,
        end_b_y=endpoint.b_y,
        end_a_landing=endpoint.a_landing,
        end_b_landing=endpoint.b_landing,
        intermediate_sections=(),
    )
    meshes = build_stair_meshes(levels, [stair])
    assert max(named.mesh.bounds[1, 2] for named in meshes) == pytest.approx(
        landing_top_world_z(ground, landing, levels)
    )


def test_landing_binding_updates_resolved_stair_route_without_rewriting_fallbacks():
    levels, ground, _, landing, stair = _stair_with_landing()
    fallback = stair.intermediate_sections[0].points
    ground.landings[0] = replace(landing, center_x=110)
    points = resolve_stair_section_image_points(ground, stair.intermediate_sections[0])
    assert points[0] == pytest.approx((75, 80))
    assert stair.intermediate_sections[0].points == fallback
    route = _resolve_stair_route_sections(
        stair, {level.index: level for level in levels}
    )
    assert any(np.isclose(section[0][0], 1.5) for section in route)


def test_landing_perimeter_participates_in_floor_coherency():
    _, ground, _, landing = _scene()
    footprint = build_level_floor_footprint(
        ground,
        None,
        lambda point, _: np.asarray(level_image_to_world_xy(ground, *point)),
    )
    assert footprint is not None
    assert footprint.area == pytest.approx(
        Polygon(landing_world_footprint(ground, landing)).area
    )


def test_landing_mesh_exports_even_without_surface_material_assignments():
    levels, _, _, landing = _scene()
    model = convert_to_glb(levels, export_untextured_surfaces=False)
    assert f"landing:{landing.landing_id}" in model.scene.geometry
    loaded = trimesh.load(BytesIO(model.glb_bytes), file_type="glb", force="scene")
    assert f"landing:{landing.landing_id}" in loaded.geometry
    assert loaded.geometry[f"landing:{landing.landing_id}"].volume > 0.0


# ### Meshy source normalization, materials and persistence ###
def _meshy_landing_asset(tmp_path, *, railing=False):
    """A non-square L platform in Meshy's Y-up frame, with independent PBR parts."""

    scene = trimesh.Scene()
    parts = [((3.0, 1.0, 0.4), (1.5, 0.5, 0.2)), ((1.0, 1.0, 0.4), (0.5, 1.5, 0.2))]
    if railing:
        parts.append(((0.1, 0.1, 1.6), (0.5, 0.5, 1.2)))
    for index, (extents, center) in enumerate(parts):
        mesh = trimesh.creation.box(extents)
        mesh.apply_translation(np.asarray(center) + (4.0, 6.0, 9.0))
        mesh.visual = trimesh.visual.texture.TextureVisuals(
            uv=np.asarray(mesh.vertices)[:, :2] % 1,
            material=trimesh.visual.material.PBRMaterial(
                name=f"landing_material_{index}",
                baseColorTexture=Image.new("RGB", (4, 4), (100 + index * 30, 50, 20)),
                normalTexture=Image.new("RGB", (4, 4), (128, 128, 255)),
                metallicFactor=0.2 + index * 0.1,
                roughnessFactor=0.7,
            ),
        )
        scene.add_geometry(mesh, geom_name=f"part_{index}", node_name=f"part_{index}")
    scene.apply_transform(
        trimesh.transformations.rotation_matrix(-math.pi / 2, (1, 0, 0))
    )
    path = tmp_path / "meshy-source.glb"
    path.write_bytes(scene.export(file_type="glb"))
    return path


def test_meshy_landing_keeps_actual_non_square_shape_and_authored_transforms(tmp_path):
    levels, ground, _, original = _scene()
    original = replace(
        original, rotation_degrees=31, scale_x=2, scale_y=0.8, scale_z=1.5
    )
    landing = prepare_generated_landing(original, _meshy_landing_asset(tmp_path))
    assert landing.generated_model_path != str(tmp_path / "meshy-source.glb")
    assert landing.landing_id == original.landing_id
    assert (
        landing.center_x,
        landing.center_y,
        landing.rotation_degrees,
        landing.scale_x,
        landing.scale_y,
        landing.scale_z,
    ) == (
        original.center_x,
        original.center_y,
        original.rotation_degrees,
        original.scale_x,
        original.scale_y,
        original.scale_z,
    )
    assert np.ptp(landing.footprint_meters, axis=0) == pytest.approx((1.5, 1.0))
    assert Polygon(landing.footprint_meters).area == pytest.approx(1.0)
    assert landing.thickness_meters == pytest.approx(0.2)
    local = load_landing_local_meshes(landing)
    assert len(local) == 2
    bounds = np.asarray(
        (
            np.min([m.bounds[0] for m in local], axis=0),
            np.max([m.bounds[1] for m in local], axis=0),
        )
    )
    assert bounds == pytest.approx(np.asarray(((-0.75, -0.5, 0), (0.75, 0.5, 0.2))))
    assert len(build_landing_meshes(ground, landing, levels)) == 2


def test_asymmetric_meshy_landing_does_not_mirror_its_shape_in_placed_world(tmp_path):
    _, ground, _, original = _scene()
    landing = prepare_generated_landing(original, _meshy_landing_asset(tmp_path))
    footprint = Polygon(landing_world_footprint(ground, landing))
    world_x, world_y = level_image_to_world_xy(
        ground, landing.center_x, landing.center_y
    )
    # Source L occupies its full lower bar, but its upper-right corner is empty.
    # Image-coordinate Y inversion must not mirror this physical mesh anatomy.
    assert footprint.covers(Point(world_x + 0.5, world_y - 0.25))
    assert not footprint.covers(Point(world_x + 0.5, world_y + 0.25))


def test_generated_landing_walk_height_uses_platform_not_railing(tmp_path):
    levels, ground, _, original = _scene()
    landing = prepare_generated_landing(
        original, _meshy_landing_asset(tmp_path, railing=True)
    )
    ground.landings[0] = landing
    assert landing.thickness_meters == pytest.approx(1.0)
    assert landing.walk_height_meters == pytest.approx(0.2)
    expected = (
        build_level_base_z_lookup(levels)[ground.index]
        + original.elevation_meters
        + 0.2
    )
    assert landing_top_world_z(ground, landing, levels) == pytest.approx(expected)
    point = np.asarray(landing.footprint_meters[0]) / ground.meters_per_pixel + (
        landing.center_x,
        landing.center_y,
    )
    attachment = landing_attachment_for_image_point(ground, landing, tuple(point))
    assert resolve_landing_attachment(ground, attachment, levels)[1] == pytest.approx(
        expected
    )


def test_generated_landing_pbr_parts_and_source_survive_save_load_and_export(tmp_path):
    levels, ground, _, original = _scene()
    landing = prepare_generated_landing(original, _meshy_landing_asset(tmp_path))
    ground.landings[0] = landing
    path = tmp_path / "project.json"
    save_project(path, 0, levels)
    restored = load_project(path)
    restored_landing = next(
        level for level in restored.levels if level.index == ground.index
    ).landings[0]
    assert restored_landing == landing
    model = convert_to_glb(restored.levels, export_untextured_surfaces=False)
    scene = trimesh.load_scene(BytesIO(model.glb_bytes), file_type="glb", process=False)
    parts = [
        mesh
        for name, mesh in scene.geometry.items()
        if name.startswith(f"landing:{landing.landing_id}")
    ]
    assert len(parts) == 2
    assert sum(len(mesh.faces) for mesh in parts) == 24
    assert {mesh.visual.material.name for mesh in parts} == {
        "landing_material_0",
        "landing_material_1",
    }
    for mesh in parts:
        assert mesh.visual.kind == "texture"
        assert len(mesh.visual.uv) == len(mesh.vertices)
        assert mesh.visual.material.baseColorTexture is not None
        assert mesh.visual.material.normalTexture is not None
        assert mesh.visual.material.roughnessFactor == pytest.approx(0.7)


def test_generated_landing_source_cache_returns_independent_meshes(tmp_path):
    landing = prepare_generated_landing(
        LandingData(0, 0), _meshy_landing_asset(tmp_path)
    )
    first = load_landing_local_meshes(landing)
    first[0].apply_translation((100, 100, 100))
    second = load_landing_local_meshes(landing)
    assert second[0].bounds[0, 0] < 1


def test_relative_generated_landing_path_is_resolved_against_project_directory(
    tmp_path,
):
    levels, ground, _, original = _scene()
    landing = prepare_generated_landing(original, _meshy_landing_asset(tmp_path))
    ground.landings[0] = landing
    path = tmp_path / "project.json"
    save_project(path, 0, levels)
    payload = json.loads(path.read_text())
    next(item for item in payload["levels"] if item["index"] == ground.index)[
        "landings"
    ][0]["generated_model_path"] = Path(landing.generated_model_path).name
    path.write_text(json.dumps(payload))
    restored = load_project(path)
    restored_landing = next(
        level for level in restored.levels if level.index == ground.index
    ).landings[0]
    assert restored_landing.generated_model_path == landing.generated_model_path


def test_missing_generated_landing_mesh_does_not_silently_revert_to_square(tmp_path):
    levels, ground, _, original = _scene()
    landing = replace(original, generated_model_path=str(tmp_path / "missing.glb"))
    with pytest.raises(ValueError, match="unavailable"):
        build_landing_mesh(ground, landing, levels)


def test_texture_regeneration_preserves_existing_landing_profile_and_scale(tmp_path):
    source = _meshy_landing_asset(tmp_path)
    landing = prepare_generated_landing(LandingData(0, 0), source)
    textured = prepare_generated_landing(landing, source, texture_only=True)
    assert textured.generated_model_path != landing.generated_model_path
    assert textured.footprint_meters == landing.footprint_meters
    assert textured.thickness_meters == landing.thickness_meters
    assert textured.walk_height_meters == landing.walk_height_meters


def test_texture_regeneration_rejects_replacement_geometry(tmp_path):
    source = _meshy_landing_asset(tmp_path)
    landing = prepare_generated_landing(LandingData(0, 0), source)
    replacement = trimesh.creation.box((2, 1, 1))
    other = tmp_path / "different.glb"
    other.write_bytes(replacement.export(file_type="glb"))
    with pytest.raises(ValueError, match="changed its geometry"):
        prepare_generated_landing(landing, other, texture_only=True)
