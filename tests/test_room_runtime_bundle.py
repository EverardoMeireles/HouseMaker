"""A real room-repacked house keeps geometry, instances and full material packs."""

# ### Imports ###
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from io import BytesIO

import numpy as np
import pytest
import trimesh
from PIL import Image
from trimesh.visual.texture import TextureVisuals

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
from housemaker.material_pack_export import extract_material_packs
from housemaker.pvs_state import PvsBakeData, PvsCell
from housemaker.pvs_workspace import prepare_pvs_geometry, pvs_world_transform
from housemaker.room_atlas_export import prepare_room_export_atlases
from housemaker.room_export_metadata import build_room_asset_manifest
from housemaker.room_zones import build_room_zones
from housemaker.scene_export import _parse_glb, write_runtime_scene_manifest
from housemaker.surface_geometry import build_fixed_surfaces
from housemaker.surface_materials import SurfaceMaterialSourceSpec
from housemaker.texture_atlas_state import ATLAS_TYPE_SURFACE
from tests.test_material_pack_export import _bytes
from tests.test_room_atlas_export import _atlas, _placement
from tests.test_room_zones import _level


# ### Real geometry, split rooms, and genuine Atlas raster fixtures ###
def _fixture(tmp_path):
    level = _level(divided=True)
    layout = build_room_zones((level,))
    centers = [room.polygon.centroid for room in layout.rooms]
    mesh = trimesh.creation.box((0.6, 0.5, 1.0))
    mesh.visual = TextureVisuals(uv=np.asarray(mesh.vertices)[:, :2] + 0.5)
    scene = trimesh.Scene(mesh)
    model = GeneratedModel(mesh=mesh, scene=scene, glb_bytes=scene.export(file_type="glb"))
    ordinary = PlacedGeneratedModel(
        "table", model, (centers[0].x - 0.8, centers[0].y, layout.rooms[0].floor_z_meters),
        rotation_degrees=(0, 0, 15), export_node_name="Object_1",
        export_source_node_name="Object_1",
    )
    source = PlacedGeneratedModel(
        "chair-source", model, (centers[0].x + 0.8, centers[0].y, layout.rooms[0].floor_z_meters),
        rotation_degrees=(0, 0, 35), scale=0.9, export_node_name="Object_2",
        export_source_node_name="Object_2",
    )
    instance = PlacedGeneratedModel(
        "chair-copy", model, (centers[1].x, centers[1].y, layout.rooms[1].floor_z_meters),
        rotation_degrees=(0, 0, 80), scale=1.2, axis_scales=(1.1, 0.9, 1.0),
        source_object_id=source.object_id, export_node_name="Object_3",
        export_source_node_name="Object_2",
    )
    objects = _atlas(tmp_path, (_placement("table"), _placement("chair-source", x=512)), resolution=1024)
    surfaces = _atlas(tmp_path, (_placement("plaster"),), identifier="shared-surface",
                      resolution=512, atlas_type=ATLAS_TYPE_SURFACE)
    materials = {
        surface.surface_id: SurfaceMaterialSourceSpec(surfaces.map_paths)
        for surface in build_fixed_surfaces((level,))
    }
    base = convert_to_glb((level,), surface_materials=materials, export_untextured_surfaces=True)
    composed = compose_placed_generated_models(base, (ordinary,))
    composed = compose_generated_model_instance_sources(composed, (source,))
    surface_sources = dict.fromkeys(materials, "plaster")
    placements = (ordinary, source, instance)
    plan = prepare_room_export_atlases(
        (objects, surfaces), layout, placements, output_directory=tmp_path / "room-atlases",
        placement_level_indices=dict.fromkeys((item.object_id for item in placements), level.index),
    )
    exported = apply_texture_atlases_to_export(
        composed, plan.atlases, surface_source_ids=surface_sources, room_layout=layout,
    )
    return layout, composed, plan, exported, surface_sources, ordinary, source, instance


def _image_payloads(document, binary):
    return {_bytes(document, binary, image["bufferView"]) for image in document.get("images", ())}


def _descendants(document, index):
    result = {index}
    for child in document["nodes"][index].get("children", ()):
        result.update(_descendants(document, child))
    return result


# ### Complete material/GLB/runtime contract, before and after PVS partitioning ###
@pytest.mark.parametrize("with_pvs", (False, True))
def test_room_runtime_bundle_preserves_maps_identity_transforms_and_final_hash(tmp_path, with_pvs):
    layout, composed, plan, model, surface_sources, ordinary, source, instance = _fixture(tmp_path)
    scale = GlobalScale(1.8, 1.9, 1.2, linked=False)
    original_plan = [item.atlas.to_dict() for item in plan.atlases]
    assert len(plan.atlases) == 3  # one room-owned object, one shared object, one Surface Atlas
    assert len(plan.source_room_ids[source.object_id]) == 2
    assert len(plan.source_room_ids[ordinary.object_id]) == 1
    visibility = None
    if with_pvs:
        payload, clusters, _triangles, signature = prepare_pvs_geometry(
            model, (ordinary, source), (instance,), scale, 1.7,
        )
        model = replace(model, glb_bytes=payload)
        visibility = PvsBakeData(signature, clusters, (
            PvsCell("all-rooms", (-20, -20, -20), (20, 20, 20),
                    tuple(cluster.cluster_id for cluster in clusters)),
        ))
    original_document, original_binary = _parse_glb(model.glb_bytes, "Invalid house")
    materials = extract_material_packs(model)
    assert len(materials.packs) == 3
    room_assets = build_room_asset_manifest(
        layout, materials, plan.atlases, plan.source_room_ids, global_scale=scale,
    )
    model_directory, material_directory = tmp_path / "models", tmp_path / "materials"
    model_directory.mkdir()
    material_directory.mkdir()
    for pack in materials.packs:
        (material_directory / pack.file_name).write_bytes(pack.glb_bytes)
    path = export_glb_file(materials.stripped_model, model_directory / "house.glb")
    kwargs = {
        "source_placements": {ordinary.object_id: ordinary, source.object_id: source},
        "instance_placements": (instance,), "global_scale": scale,
        "visibility": visibility, "room_assets": room_assets,
    }
    manifest_path = write_runtime_scene_manifest(path, **kwargs)
    final_bytes = path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    final_document, final_binary = _parse_glb(final_bytes, "Invalid exported house")
    manifest = json.loads(manifest_bytes)

    assert manifest["asset"]["sha256"] == hashlib.sha256(final_bytes).hexdigest()
    if with_pvs:
        assert manifest["visibility"]["glbSha256"] == manifest["asset"]["sha256"]
    assert len(manifest["roomAssets"]["rooms"]) == 2
    assert len(manifest["roomAssets"]["materials"]) == 3
    assert set(final_document["meshes"][0]) >= {"primitives"}
    assert len(final_document["meshes"]) == len(original_document["meshes"])
    assert final_document["meshes"] == original_document["meshes"]
    assert final_document["accessors"] == original_document["accessors"]

    # Prototype library stays detached; numbered ordinary furniture never merges
    # with wall geometry, and source-local data is not globally scaled twice.
    nodes = final_document["nodes"]
    numbered_root = next(index for index, node in enumerate(nodes) if node.get("name") == "Object_1")
    source_root = next(index for index, node in enumerate(nodes) if node.get("name") == "Object_2")
    rendered = set().union(*(
        _descendants(final_document, index)
        for index in final_document["scenes"][final_document.get("scene", 0)]["nodes"]
    ))
    assert numbered_root in rendered and source_root not in rendered
    object_descendants = _descendants(final_document, numbered_root)
    assert all(not nodes[index].get("extras", {}).get("housemaker_room_ids")
               for index in object_descendants)
    ordinary_faces = sum(
        final_document["accessors"][primitive["indices"]]["count"] // 3
        for index in object_descendants if "mesh" in nodes[index]
        for primitive in final_document["meshes"][nodes[index]["mesh"]]["primitives"]
    )
    assert ordinary_faces == 12
    groups = manifest["instanceGroups"]
    assert len(groups) == 1 and groups[0]["sourceNodeName"] == "Object_2"
    instances = {item["id"]: item for item in groups[0]["instances"]}
    for placement in (source, instance):
        actual = np.asarray(instances[placement.object_id]["worldMatrix"]).reshape(4, 4).T
        expected = pvs_world_transform(scale) @ build_placed_generated_model_gltf_transform(placement)
        np.testing.assert_allclose(actual, expected)
        assert np.linalg.norm(actual[:3, 3]) > 1.0

    # Full bytes belong only to the cubes. Every material binding resolves from
    # models/ into materials/, matches its final material index, and owns one cube.
    full_image_payloads = _image_payloads(original_document, original_binary)
    packed_images = set()
    bindings = manifest["roomAssets"]["materials"]
    for binding, pack in zip(bindings, materials.packs, strict=True):
        pack_path = model_directory / binding["file"]
        assert pack_path.resolve() == (material_directory / pack.file_name).resolve()
        assert binding["sha256"] == hashlib.sha256(pack_path.read_bytes()).hexdigest()
        index = binding["materialIndex"]
        assert final_document["materials"][index]["extras"]["housemaker_material_id"] == binding["materialId"]
        pack_document, pack_binary = _parse_glb(pack_path.read_bytes(), "Invalid cube")
        assert len(pack_document["meshes"]) == len(pack_document["nodes"]) == 1
        primitive = pack_document["meshes"][0]["primitives"][0]
        assert pack_document["accessors"][primitive["indices"]]["count"] == 36
        packed_images.update(_image_payloads(pack_document, pack_binary))
        assert binding["roomIds"]
    assert packed_images == full_image_payloads
    for payload in _image_payloads(final_document, final_binary):
        with Image.open(BytesIO(payload)) as image:
            assert max(image.size) <= 256
    assert all(payload not in final_bytes for payload in full_image_payloads)
    shared_surfaces = next(binding for binding in manifest["roomAssets"]["atlases"]
                           if binding["type"] == "surface")
    assert shared_surfaces["shared"]
    assert set(shared_surfaces["roomIds"]) == {room.room_id for room in layout.rooms}
    shared_surface_material = next(binding for binding in bindings
                                   if binding.get("atlasId") == shared_surfaces["atlasId"])
    assert set(shared_surface_material["roomIds"]) == {room.room_id for room in layout.rooms}

    # Rewriting the runtime data is byte-idempotent even with the library scene,
    # physical-scale wrapper and PVS already present. Repacking adopts stable IDs.
    write_runtime_scene_manifest(path, **kwargs)
    assert path.read_bytes() == final_bytes
    assert manifest_path.read_bytes() == manifest_bytes
    second_plan = prepare_room_export_atlases(
        plan.atlases, layout, (ordinary, source, instance),
        output_directory=tmp_path / "room-atlases",
        placement_level_indices=dict.fromkeys((ordinary.object_id, source.object_id, instance.object_id), 2),
    )
    assert [item.atlas.to_dict() for item in second_plan.atlases] == original_plan
    repeated = apply_texture_atlases_to_export(
        composed, second_plan.atlases, surface_source_ids=surface_sources, room_layout=layout,
    )
    repeated_materials = extract_material_packs(repeated)
    assert [(pack.material_id, pack.file_name, pack.glb_bytes) for pack in repeated_materials.packs] == [
        (pack.material_id, pack.file_name, pack.glb_bytes) for pack in materials.packs
    ]
