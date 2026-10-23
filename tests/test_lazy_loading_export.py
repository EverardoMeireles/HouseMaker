"""Streaming policies keep fallbacks, physical spawn coordinates and stable bindings."""

# ### Imports ###
from __future__ import annotations

import json
import math

import pytest

from housemaker.global_scale import GlobalScale
from housemaker.lazy_loading_export import (
    build_material_streaming_policy,
    resolve_camera_bootstrap,
    select_bootstrap_material_indices,
)
from housemaker.lazy_loading_state import CameraStartPoint, RuntimeLoadingData
from housemaker.level_coordinates import (
    build_level_base_z_lookup,
    level_image_to_world_xy,
)
from housemaker.material_pack_export import build_progressive_material_export
from housemaker.models import create_default_levels
from housemaker.project_io import ProjectData, load_project, save_project
from housemaker.room_export_metadata import build_room_asset_manifest
from housemaker.room_zones import build_room_zones
from housemaker.scene_export import build_runtime_scene_manifest
from tests.test_room_runtime_bundle import _fixture
from tests.test_room_zones import _level


# ### Persistence and strict authoring values ###
def test_runtime_loading_roundtrip_and_old_project_defaults(tmp_path):
    levels = create_default_levels()
    settings = RuntimeLoadingData(CameraStartPoint(2, 40, 80, 0.2, 35, -10), True)
    path = save_project(tmp_path / "project.json", 2, levels, runtime_loading=settings)
    assert load_project(path).runtime_loading == settings
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("runtime_loading")
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_project(path).runtime_loading == RuntimeLoadingData()
    assert RuntimeLoadingData.from_dict(settings.to_dict()) == settings


@pytest.mark.parametrize(
    "payload", ({"use_ktx2_basis_compression": "true"}, {"start_point": []})
)
def test_invalid_saved_choices_are_not_silently_enabled(payload):
    with pytest.raises((TypeError, ValueError)):
        RuntimeLoadingData.from_dict(payload)


@pytest.mark.parametrize("value", (math.nan, math.inf, True, "2"))
def test_camera_start_rejects_nonfinite_or_nonnumeric_coordinates(value):
    with pytest.raises((TypeError, ValueError)):
        CameraStartPoint(2, value, 0)


def test_project_rejects_unvalidated_runtime_settings():
    with pytest.raises(TypeError, match="runtime loading"):
        ProjectData(None, 2, create_default_levels(), runtime_loading={})


# ### Camera start remains plan-attached and eye height stays physical ###
def test_authored_start_applies_global_scale_once_and_does_not_scale_eye_height():
    level = _level(divided=True)
    level.scale = 1.3
    level.offset_x_meters = 2
    level.offset_y_meters = -3
    layout = build_room_zones((level,))
    start = CameraStartPoint(2, 200, 300, 0.1, 90, -30)
    scale = GlobalScale(1.8, 1.9, 2.2, linked=False)
    camera = resolve_camera_bootstrap(
        (level,), layout, RuntimeLoadingData(start), scale, 1.7
    )
    x, y = level_image_to_world_xy(level, 200, 300)
    z = build_level_base_z_lookup((level,))[2] + 0.1
    assert camera.floor_position == pytest.approx((x * 1.8, z * 2.2, -y * 1.9))
    assert camera.position == pytest.approx((x * 1.8, z * 2.2 + 1.7, -y * 1.9))
    assert camera.direction == pytest.approx((0, -0.5, -math.sqrt(3) / 2))
    assert camera.source == "authored" and len(camera.room_ids) == 1


def test_automatic_start_is_inside_a_confirmed_room_and_handles_empty_scenes():
    level = _level(divided=True)
    layout = build_room_zones((level,))
    camera = resolve_camera_bootstrap(
        (level,), layout, RuntimeLoadingData(), GlobalScale(), 1.7
    )
    assert camera.source == "automatic" and len(camera.room_ids) == 1
    assert camera.position[1] - camera.floor_position[1] == pytest.approx(1.7)
    empty = resolve_camera_bootstrap(
        (), build_room_zones(()), RuntimeLoadingData(), GlobalScale(), 1.7
    )
    assert empty.position == (0, 1.7, 0) and empty.room_ids == ()


def test_start_on_excluded_level_blocks_export_instead_of_teleporting_camera():
    level = _level()
    level.include_in_export = False
    with pytest.raises(ValueError, match="excluded level"):
        resolve_camera_bootstrap(
            (level,),
            build_room_zones((level,)),
            RuntimeLoadingData(CameraStartPoint(2, 20, 20)),
            GlobalScale(),
            1.7,
        )


# ### Real room Atlas bindings and scheduling semantics ###
def test_policy_contains_variants_shared_rules_tutorial_and_failure_protection(
    tmp_path,
):
    layout, _composed, plan, model, _sources, ordinary, source, instance = _fixture(
        tmp_path
    )
    level = _level(divided=True)
    settings = RuntimeLoadingData(CameraStartPoint(2, 200, 300))
    scale = GlobalScale()
    camera = resolve_camera_bootstrap((level,), layout, settings, scale, 1.7)
    indices = select_bootstrap_material_indices(
        model, camera, plan, (ordinary, source, instance), scale=scale
    )
    assert indices
    exported = build_progressive_material_export(
        model, bootstrap_material_indices=indices
    )
    metadata = build_room_asset_manifest(
        layout, exported, plan.atlases, plan.source_room_ids
    )
    policy = build_material_streaming_policy(
        metadata, exported, camera, bootstrap_indices=indices, settings=settings
    )
    assert policy["format"] == "housemaker-material-streaming"
    assert policy["bootstrap"]["minimumTutorialSeconds"] == 0
    assert (
        policy["tutorial"]["skippable"]
        and not policy["tutorial"]["waitForFullMaterials"]
    )
    assert policy["sharedMaterials"]["inferFullPriorityFromRoomDependencies"] is False
    assert policy["failurePolicy"]["neverHideGeometryForMissingMaterial"]
    assert policy["failurePolicy"]["onFullFailure"] == "retain-preview"
    assert policy["binding"]["addMaterialCubeToScene"] is False
    assert policy["cache"]["keepEmbeddedFallbacks"] is True
    assert policy["scheduler"]["swapWhen"] == "shader-ready"
    assert policy["priorityOrder"][1]["scope"] == "visible-in-current-room"
    assert policy["priorityOrder"][2]["scope"] == "visible-outside-current-room"
    assert metadata["previewMaxSize"] == 1024 and metadata["fallbackMaxSize"] == 128
    for material in metadata["materials"]:
        assert set(material["variants"]) == {"embedded", "preview", "full"}
        assert material["variants"]["preview"]["maxResolution"] <= 1024
        assert material["variants"]["full"]["materialIndex"] == 0
        assert material["variants"]["full"]["byteSize"] > 0
        assert (
            material["variants"]["preview"]["file"]
            != material["variants"]["full"]["file"]
        )
        assert material["variants"]["full"]["file"] == material["file"]
        assert material["binding"]["materialExtrasId"] == material["materialId"]
    assert all(
        material["shared"]
        for material in metadata["materials"]
        if material["atlasType"] == "surface"
    )
    json.dumps(policy, allow_nan=False)

    manifest = build_runtime_scene_manifest(
        glb_name="scene.glb",
        glb_bytes=exported.stripped_model.glb_bytes,
        source_placements={},
        instance_placements=(),
        global_scale=GlobalScale(2, 3, 4),
        room_assets=metadata,
        material_streaming=policy,
    )
    assert manifest["firstPersonCamera"]["position"] == list(camera.position)
    assert manifest["firstPersonCamera"]["eyeHeightMeters"] == 1.7
    assert manifest["materialStreaming"] == policy


def test_unknown_unatlased_material_does_not_eagerly_load_in_every_room(tmp_path):
    layout, _, plan, model, _, *_ = _fixture(tmp_path)
    level = _level(divided=True)
    camera = resolve_camera_bootstrap(
        (level,), layout, RuntimeLoadingData(), GlobalScale(), 1.7
    )
    export = build_progressive_material_export(model)
    metadata = build_room_asset_manifest(
        layout, export, plan.atlases, plan.source_room_ids
    )
    metadata["materials"][0].pop("atlasId", None)
    metadata["materials"][0]["roomIds"] = [room.room_id for room in layout.rooms]
    build_material_streaming_policy(metadata, export, camera)
    assert metadata["materials"][0]["shared"]
