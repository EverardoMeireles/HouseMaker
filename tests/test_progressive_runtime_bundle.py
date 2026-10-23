"""Room atlases, startup pose, PVS and instance materials stream together."""

# ### Imports ###
from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import replace
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
from PIL import Image

from housemaker.atlas_export import apply_texture_atlases_to_export
from housemaker.glb import (
    GeneratedModel,
    PlacedGeneratedModel,
    build_placed_generated_model_gltf_transform,
    compose_generated_model_instance_sources,
    compose_placed_generated_models,
    convert_to_glb,
    export_glb_file,
)
from housemaker.global_scale import GlobalScale
from housemaker.ktx2_compression import KTX2_IDENTIFIER, find_ktx2_encoder
from housemaker.lazy_loading_export import resolve_camera_bootstrap
from housemaker.lazy_loading_state import CameraStartPoint, RuntimeLoadingData
from housemaker.level_coordinates import level_world_to_image_xy
from housemaker.pvs_runtime import write_pvs_runtime_module
from housemaker.pvs_state import PvsBakeData, PvsCell
from housemaker.pvs_workspace import prepare_pvs_geometry, pvs_world_transform
from housemaker.room_atlas_export import prepare_room_export_atlases
from housemaker.room_export_workspace import RoomExportWorkspaceMixin
from housemaker.room_zones import build_room_zones
from housemaker.scene_export import _parse_glb, write_runtime_scene_manifest
from housemaker.surface_geometry import build_fixed_surfaces
from housemaker.surface_materials import SurfaceMaterialSourceSpec
from housemaker.texture_atlas_state import ATLAS_TYPE_SURFACE
from tests.test_material_pack_export import _bytes
from tests.test_room_atlas_export import _atlas, _placement
from tests.test_room_runtime_bundle import _descendants
from tests.test_room_zones import _level


# ### Real room packing with a distant room-owned object ###
def _streaming_fixture(tmp_path):
    level = _level(divided=True)
    layout = build_room_zones((level,))
    first = layout.rooms[0].polygon.centroid
    destination = layout.rooms[1].polygon.centroid
    mesh = trimesh.creation.box((0.6, 0.5, 1.0))
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.asarray(mesh.vertices)[:, :2] + 0.5
    )
    scene = trimesh.Scene(mesh)
    source_model = GeneratedModel(
        mesh=mesh, scene=scene, glb_bytes=scene.export(file_type="glb")
    )
    floor_z = layout.rooms[0].floor_z_meters
    ordinary = PlacedGeneratedModel(
        "table",
        source_model,
        (first.x - 0.8, first.y, floor_z),
        rotation_degrees=(0, 0, 15),
        export_node_name="Object_1",
        export_source_node_name="Object_1",
    )
    source = replace(
        ordinary,
        object_id="chair-source",
        source_object_id="chair-source",
        atlas_source_object_id="chair-source",
        object_name="chair-source",
        world_position=(first.x + 0.8, first.y, floor_z),
        rotation_degrees=(0, 0, 35),
        export_node_name="Object_2",
        export_source_node_name="Object_2",
    )
    instance = replace(
        source,
        object_id="chair-copy",
        source_object_id="chair-source",
        world_position=(destination.x, destination.y, floor_z),
        rotation_degrees=(0, 0, 80),
        scale=1.2,
        axis_scales=(1.1, 0.9, 1.0),
        export_node_name="Object_3",
    )
    distant = replace(
        ordinary,
        object_id="distant-cupboard",
        source_object_id="distant-cupboard",
        object_name="distant-cupboard",
        world_position=(
            destination.x,
            destination.y - 0.8,
            layout.rooms[1].floor_z_meters,
        ),
        export_node_name="Object_4",
        export_source_node_name="Object_4",
        atlas_source_object_id="distant-cupboard",
    )
    atlases = (
        _atlas(
            tmp_path,
            (_placement("table", size=1024),),
            identifier="table",
            resolution=1024,
        ),
        _atlas(
            tmp_path, (_placement("chair-source"),), identifier="chair", resolution=512
        ),
        _atlas(
            tmp_path,
            (_placement("distant-cupboard", size=1024),),
            identifier="cupboard",
            resolution=1024,
        ),
        _atlas(
            tmp_path,
            (_placement("plaster", size=2048),),
            identifier="shared-surface",
            resolution=2048,
            atlas_type=ATLAS_TYPE_SURFACE,
        ),
    )
    # Distinct pixels avoid trimesh sharing one identical image between the
    # near and distant materials, which legitimately raises both to 1024px.
    Image.new("RGBA", (1024, 1024), (150, 90, 60, 255)).save(
        atlases[2].map_paths["base_color"]
    )
    placements = (ordinary, source, instance, distant)
    materials = {
        surface.surface_id: SurfaceMaterialSourceSpec(atlases[-1].map_paths)
        for surface in build_fixed_surfaces((level,))
    }
    surface_sources = dict.fromkeys(materials, "plaster")
    base = convert_to_glb(
        (level,), surface_materials=materials, export_untextured_surfaces=True
    )
    composed = compose_placed_generated_models(base, (ordinary, distant))
    composed = compose_generated_model_instance_sources(composed, (source,))
    plan = prepare_room_export_atlases(
        atlases,
        layout,
        placements,
        output_directory=tmp_path / "room-atlases",
        placement_level_indices=dict.fromkeys(
            (item.object_id for item in placements), 2
        ),
    )
    model = apply_texture_atlases_to_export(
        composed,
        plan.atlases,
        surface_source_ids=surface_sources,
        room_layout=layout,
    )
    return layout, plan, model, ordinary, source, instance, distant


def _image_dimensions(payload):
    if payload.startswith(KTX2_IDENTIFIER):
        return struct.unpack_from("<II", payload, 20)
    with Image.open(BytesIO(payload)) as image:
        return image.size


def _material_base_color_dimensions(document, binary, material_index):
    material = document["materials"][material_index]
    texture_index = material["pbrMetallicRoughness"]["baseColorTexture"]["index"]
    texture = document["textures"][texture_index]
    source = texture.get(
        "source",
        texture.get("extensions", {}).get("KHR_texture_basisu", {}).get("source"),
    )
    image = document["images"][source]
    return _image_dimensions(_bytes(document, binary, image["bufferView"]))


# ### Complete export without constructing the heavyweight Qt workspace ###
@pytest.mark.parametrize("use_ktx2", (False, True))
def test_progressive_bundle_preserves_camera_geometry_instances_and_pvs(
    tmp_path, use_ktx2
):
    if use_ktx2 and find_ktx2_encoder() is None:
        pytest.skip("The optional local KTX Software encoder is unavailable.")
    layout, plan, model, ordinary, source, instance, distant = _streaming_fixture(
        tmp_path
    )
    level = _level(divided=True)
    scale = GlobalScale(3.0, 2.0, 1.2, linked=False)
    eye_height = 1.83
    center = layout.rooms[0].polygon.centroid
    image_xy = level_world_to_image_xy(level, center.x, center.y)
    point = CameraStartPoint(
        2, *image_xy, height_offset_meters=0.05, yaw_degrees=33, pitch_degrees=-5
    )
    settings = RuntimeLoadingData(point, use_ktx2)
    sources = (ordinary, source, distant)
    partitioned, clusters, _triangles, signature = prepare_pvs_geometry(
        model,
        sources,
        (instance,),
        scale,
        eye_height,
    )
    model = replace(model, glb_bytes=partitioned)
    original_document, original_binary = _parse_glb(
        model.glb_bytes, "Invalid original GLB"
    )
    visibility = PvsBakeData(
        signature,
        clusters,
        (
            PvsCell(
                "all-rooms",
                (-100, -20, -100),
                (100, 30, 100),
                tuple(cluster.cluster_id for cluster in clusters),
            ),
        ),
        eye_height_meters=eye_height,
    )

    class ExportHarness(RoomExportWorkspaceMixin):
        def _get_room_layout(self):
            return layout

    workspace = ExportHarness()
    workspace.levels = (level,)
    workspace.global_scale = scale
    workspace.runtime_loading = settings
    workspace._generation_settings = SimpleNamespace(
        first_person_eye_height_meters=eye_height
    )
    workspace._last_room_export_layout = layout
    workspace._last_room_export_plan = plan
    workspace._last_room_export_placements = sources + (instance,)
    material_export, room_assets = workspace._prepare_export_material_packs(model)
    policy = workspace._last_material_streaming_policy
    assert len(material_export.packs) == len(material_export.preview_packs) == 4
    assert set(plan.source_room_ids[source.object_id]) == {
        room.room_id for room in layout.rooms
    }
    assert len(plan.source_room_ids[distant.object_id]) == 1

    model_directory, material_directory = tmp_path / "models", tmp_path / "materials"
    model_directory.mkdir()
    material_directory.mkdir()
    workspace._write_export_material_packs(material_export, material_directory)
    path = export_glb_file(
        material_export.stripped_model, model_directory / "house.glb"
    )
    arguments = {
        "source_placements": {item.object_id: item for item in sources},
        "instance_placements": (instance,),
        "global_scale": scale,
        "first_person_eye_height_meters": eye_height,
        "visibility": visibility,
        "room_assets": room_assets,
        "material_streaming": policy,
    }
    manifest_path = write_runtime_scene_manifest(path, **arguments)
    helper_path = write_pvs_runtime_module(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    exported_bytes = path.read_bytes()
    document, binary = _parse_glb(exported_bytes, "Invalid progressive GLB")
    assert helper_path.parent == model_directory and helper_path.is_file()
    assert manifest["asset"]["sha256"] == hashlib.sha256(exported_bytes).hexdigest()
    assert manifest["visibility"]["glbSha256"] == manifest["asset"]["sha256"]

    # Physical camera height is scaled floor elevation plus an unscaled human
    # eye height. Do not multiply the operator's 1.83m eye height by Global Z.
    camera = manifest["materialStreaming"]["initialCamera"]
    expected_floor = (
        center.x * scale.x,
        (layout.rooms[0].floor_z_meters + 0.05) * scale.z,
        -center.y * scale.y,
    )
    assert camera["floorPosition"] == pytest.approx(expected_floor)
    assert camera["position"] == pytest.approx(
        (expected_floor[0], expected_floor[1] + eye_height, expected_floor[2])
    )
    assert camera["source"] == "authored" and camera["globalScaleAlreadyApplied"]
    assert manifest["firstPersonCamera"]["position"] == camera["position"]
    assert camera["roomIds"] == [layout.rooms[0].room_id]
    resolved = resolve_camera_bootstrap((level,), layout, settings, scale, eye_height)
    assert list(resolved.direction) == pytest.approx(camera["lookDirection"])
    assert (
        policy["tutorial"]["skippable"]
        and not policy["tutorial"]["waitForFullMaterials"]
    )
    assert policy["sharedMaterials"]["inferFullPriorityFromRoomDependencies"] is False

    # Atlas UVs/accessors and source-local geometry stay byte exact across all
    # tiers. Only material/image dependencies and root runtime scaling differ.
    assert document["meshes"] == original_document["meshes"]
    assert document["accessors"] == original_document["accessors"]
    for accessor in original_document["accessors"]:
        index = accessor["bufferView"]
        assert _bytes(document, binary, index) == _bytes(
            original_document, original_binary, index
        )
    _payload, after_clusters, _occluders, after_signature = prepare_pvs_geometry(
        material_export.stripped_model,
        sources,
        (instance,),
        scale,
        eye_height,
    )
    assert after_signature == signature and after_clusters == clusters

    atlas_for_source = {
        placement.object_id: item.atlas
        for item in plan.atlases
        for placement in item.atlas.placements
    }
    materials_by_atlas = {
        entry["atlasId"]: entry for entry in manifest["roomAssets"]["materials"]
    }
    distant_material = materials_by_atlas[atlas_for_source[distant.object_id].atlas_id]
    table_material = materials_by_atlas[atlas_for_source[ordinary.object_id].atlas_id]
    assert not distant_material["bootstrap"] and table_material["bootstrap"]
    assert _material_base_color_dimensions(
        document, binary, distant_material["materialIndex"]
    ) == (128, 128)
    assert _material_base_color_dimensions(
        document, binary, table_material["materialIndex"]
    ) == (1024, 1024)
    surface_material = next(
        entry
        for entry in manifest["roomAssets"]["materials"]
        if entry["atlasType"] == "surface"
    )
    assert surface_material["bootstrap"] and surface_material["shared"]
    assert _material_base_color_dimensions(
        document, binary, surface_material["materialIndex"]
    ) == (1024, 1024)

    for material in manifest["roomAssets"]["materials"]:
        material_index = material["materialIndex"]
        assert (
            document["materials"][material_index]["extras"]["housemaker_material_id"]
            == material["materialId"]
        )
        for tier in ("preview", "full"):
            variant = material["variants"][tier]
            pack_path = model_directory / variant["file"]
            assert pack_path.parent.resolve() == material_directory.resolve()
            pack_bytes = pack_path.read_bytes()
            assert variant["byteSize"] == len(pack_bytes)
            assert variant["sha256"] == hashlib.sha256(pack_bytes).hexdigest()
            cube, cube_binary = _parse_glb(pack_bytes, "Invalid streamed cube")
            assert cube["asset"]["extras"]["materialId"] == material["materialId"]
            assert (
                len(cube["meshes"]) == len(cube["materials"]) == len(cube["nodes"]) == 1
            )
            primitive = cube["meshes"][0]["primitives"][0]
            assert cube["accessors"][primitive["indices"]]["count"] == 36
            sizes = [
                _image_dimensions(_bytes(cube, cube_binary, image["bufferView"]))
                for image in cube["images"]
            ]
            assert max(max(size) for size in sizes) == variant["maxResolution"]
            assert ("normalTexture" in cube["materials"][0]) == (tier == "full")
            assert (
                "KHR_texture_basisu" in cube.get("extensionsRequired", ())
            ) == use_ktx2
            if tier == "preview":
                assert max(max(size) for size in sizes) <= 1024
            if tier == "full" and material["atlasType"] == "surface":
                assert max(max(size) for size in sizes) == 2048
    assert len(list(material_directory.glob("*.glb"))) == 8

    # Shared instance prototypes and their copies use the same material ID and
    # both variants. Neither global scaling nor streaming changes placements.
    group = manifest["instanceGroups"][0]
    source_node = next(
        index
        for index, node in enumerate(document["nodes"])
        if node.get("name") == group["sourceNodeName"]
    )
    prototype_material_indices = {
        primitive["material"]
        for index in _descendants(document, source_node)
        if "mesh" in document["nodes"][index]
        for primitive in document["meshes"][document["nodes"][index]["mesh"]][
            "primitives"
        ]
    }
    shared_material = materials_by_atlas[atlas_for_source[source.object_id].atlas_id]
    assert prototype_material_indices == {shared_material["materialIndex"]}
    assert set(shared_material["variants"]) == {"embedded", "preview", "full"}
    instances = {item["id"]: item for item in group["instances"]}
    for placement in (source, instance):
        matrix = (
            np.asarray(instances[placement.object_id]["worldMatrix"]).reshape(4, 4).T
        )
        expected = pvs_world_transform(
            scale
        ) @ build_placed_generated_model_gltf_transform(placement)
        np.testing.assert_allclose(matrix, expected)
    assert "detached-instance-prototypes" in policy["binding"]["applyTo"]

    # Rewriting only the runtime manifest is byte-idempotent, including hashes.
    manifest_bytes = manifest_path.read_bytes()
    write_runtime_scene_manifest(path, **arguments)
    assert path.read_bytes() == exported_bytes
    assert manifest_path.read_bytes() == manifest_bytes
