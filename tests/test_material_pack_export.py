# ### Imports ###
from __future__ import annotations

import copy
from io import BytesIO

import numpy as np
import pytest
import trimesh
from PIL import Image

from housemaker.glb import GeneratedModel
from housemaker.material_pack_export import extract_material_packs
from housemaker.scene_export import _parse_glb, _serialize_glb


# ### Authoritative GLB fixtures ###
def _view(document, binary, payload):
    binary.extend(b"\0" * (-len(binary) % 4))
    document["bufferViews"].append({
        "buffer": 0, "byteOffset": len(binary), "byteLength": len(payload),
    })
    binary.extend(payload)
    return len(document["bufferViews"]) - 1


@pytest.fixture(scope="module")
def model():
    document = {
        "asset": {"version": "2.0", "extras": {"preserved": "authoring"}},
        "scene": 0, "scenes": [{"nodes": [0, 1]}],
        "nodes": [
            {"name": "Object_1", "mesh": 0, "translation": [3, 4, 5],
             "extras": {"housemaker_atlas_id": "atlas-a", "halfMesh": True,
                        "housemaker_instance_source_id": "asset-a"}},
            {"name": "level_floor", "mesh": 1,
             "extras": {"housemaker_atlas_id": "atlas-b", "room_id": 7}},
        ],
        "meshes": [
            {"name": "geometry-a", "primitives": [{"material": 0, "indices": 3,
                "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2, "TEXCOORD_1": 2}}]},
            {"name": "geometry-b", "primitives": [{"material": 1, "indices": 3,
                "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2, "TEXCOORD_1": 2}}]},
        ],
        "accessors": [], "bufferViews": [], "images": [],
        "textures": [{"source": index, "sampler": 0} for index in range(4)],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 10497}],
        "buffers": [{"byteLength": 0}],
        "extensionsUsed": ["KHR_texture_transform", "KHR_materials_clearcoat", "unused_geometry_extension"],
        "extensionsRequired": ["KHR_texture_transform", "unused_geometry_extension"],
    }
    binary = bytearray()
    arrays = (
        (np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), "<f4"), 5126, "VEC3"),
        (np.asarray(((0, 0, 1),) * 3, "<f4"), 5126, "VEC3"),
        (np.asarray(((0, 0), (1, 0), (0, 1)), "<f4"), 5126, "VEC2"),
        (np.asarray((0, 1, 2), "<u2"), 5123, "SCALAR"),
    )
    for array, component, kind in arrays:
        index = _view(document, binary, array.tobytes())
        document["accessors"].append({"bufferView": index, "componentType": component,
                                      "count": len(array), "type": kind})
    document["accessors"][0].update(min=[0, 0, 0], max=[1, 1, 0])
    random = np.random.default_rng(42)
    for index in range(4):
        image = Image.fromarray(random.integers(0, 256, (512, 768, 4), dtype=np.uint8))
        output = BytesIO()
        image.save(output, format="PNG")
        document["images"].append({"name": f"original-map-{index}", "mimeType": "image/png",
                                    "bufferView": _view(document, binary, output.getvalue())})
    material = {
        "name": "[OBJECT] Furniture",
        "pbrMetallicRoughness": {"baseColorTexture": {"index": 0},
                                 "metallicRoughnessTexture": {"index": 2},
                                 "roughnessFactor": 0.8, "metallicFactor": 1.0},
        "normalTexture": {"index": 1, "scale": 0.37},
        "occlusionTexture": {"index": 2, "texCoord": 1, "strength": 0.63},
        "alphaMode": "MASK", "alphaCutoff": 0.3, "doubleSided": True,
        "extensions": {"KHR_materials_clearcoat": {"clearcoatFactor": 0.4,
            "clearcoatTexture": {"index": 0, "extensions": {"KHR_texture_transform": {
                "offset": [0.2, 0.3], "scale": [0.8, 0.9], "rotation": 0.1, "texCoord": 1,
            }}}}},
    }
    other = copy.deepcopy(material)
    other["name"] = "[SURFACE] Walls"
    other["pbrMetallicRoughness"]["baseColorTexture"]["index"] = 3
    document["materials"] = [material, other]
    binary.extend(b"unused trailing bytes are not copied")
    document["buffers"][0]["byteLength"] = len(binary)
    return GeneratedModel(mesh=trimesh.Trimesh(process=False), scene=trimesh.Scene(),
                          glb_bytes=_serialize_glb(document, bytes(binary)))


def _parse(payload):
    return _parse_glb(payload, "Invalid test GLB")


def _bytes(document, binary, index):
    view = document["bufferViews"][index]
    return binary[view.get("byteOffset", 0):view.get("byteOffset", 0) + view["byteLength"]]


# ### Full-fidelity material cubes ###
def test_packs_contain_one_cube_and_original_full_image_payloads(model):
    original, original_binary = _parse(model.glb_bytes)
    result = extract_material_packs(model)
    assert len(result.packs) == 2
    for pack in result.packs:
        document, binary = _parse(pack.glb_bytes)
        assert len(document["nodes"]) == len(document["meshes"]) == len(document["materials"]) == 1
        primitive = document["meshes"][0]["primitives"][0]
        assert document["accessors"][primitive["indices"]]["count"] == 36
        position = document["accessors"][primitive["attributes"]["POSITION"]]
        assert position["count"] == 24
        assert position["min"] == [-0.5, -0.5, -0.5]
        assert position["max"] == [0.5, 0.5, 0.5]
        assert {"TEXCOORD_0", "TEXCOORD_1"}.issubset(primitive["attributes"])
        for image in document["images"]:
            source = next(entry for entry in original["images"] if entry["name"] == image["name"])
            assert _bytes(document, binary, image["bufferView"]) == _bytes(
                original, original_binary, source["bufferView"]
            )
        assert "unused_geometry_extension" not in document.get("extensionsUsed", [])
        assert document["extensionsRequired"] == ["KHR_texture_transform"]


def test_material_parameters_orm_ao_uv1_and_extensions_are_preserved(model):
    pack = extract_material_packs(model).packs[0]
    document, _binary = _parse(pack.glb_bytes)
    material = document["materials"][0]
    assert material["name"] == "[OBJECT] Furniture"
    assert material["normalTexture"] == {"index": 1, "scale": 0.37}
    assert material["occlusionTexture"] == {"index": 2, "texCoord": 1, "strength": 0.63}
    assert material["pbrMetallicRoughness"]["metallicRoughnessTexture"]["index"] == 2
    assert material["alphaMode"] == "MASK" and material["alphaCutoff"] == 0.3
    assert material["doubleSided"] is True
    assert material["extensions"]["KHR_materials_clearcoat"]["clearcoatFactor"] == 0.4
    assert document["samplers"] == [{"magFilter": 9729, "minFilter": 9987,
                                     "wrapS": 33071, "wrapT": 10497}]


# ### Preview-only main textures without geometry mutation ###
def test_main_bin_is_compacted_and_keeps_bounded_embedded_previews(model):
    result = extract_material_packs(model)
    document, binary = _parse(result.stripped_model.glb_bytes)
    original, original_binary = _parse(model.glb_bytes)
    assert len(result.stripped_model.glb_bytes) < len(model.glb_bytes) / 4
    assert b"unused trailing bytes" not in binary
    for image in document["images"]:
        with Image.open(BytesIO(_bytes(document, binary, image["bufferView"]))) as decoded:
            assert max(decoded.size) <= 256
    for image in original["images"]:
        assert _bytes(original, original_binary, image["bufferView"]) not in binary


def test_uv_normals_geometry_indices_transforms_ids_and_extras_remain_exact(model):
    original, original_binary = _parse(model.glb_bytes)
    result = extract_material_packs(model)
    document, binary = _parse(result.stripped_model.glb_bytes)
    for key in ("nodes", "scenes", "scene", "meshes", "accessors", "asset"):
        assert document[key] == original[key]
    for accessor in original["accessors"]:
        index = accessor["bufferView"]
        assert _bytes(document, binary, index) == _bytes(original, original_binary, index)
    assert result.stripped_model.scene is model.scene
    assert result.stripped_model.mesh is model.mesh
    assert model.glb_bytes == _serialize_glb(original, original_binary)


def test_material_ids_and_filenames_are_unique_stable_and_atlas_bound(model):
    first = extract_material_packs(model)
    second = extract_material_packs(model)
    assert first.material_bindings() == second.material_bindings()
    assert len({pack.material_id for pack in first.packs}) == 2
    assert {pack.atlas_id for pack in first.packs} == {"atlas-a", "atlas-b"}
    for pack in first.packs:
        assert "/" not in pack.file_name and "\\" not in pack.file_name
        assert pack.file_name.endswith(".glb")
        assert first.material_bindings()[pack.material_index]["file"] == pack.file_name


def test_no_texture_house_has_no_material_packs(model):
    document, binary = _parse(model.glb_bytes)
    document["materials"] = [{"name": "glass", "alphaMode": "BLEND"}]
    document["nodes"] = [{"name": "plain", "mesh": 0}]
    document["meshes"] = [document["meshes"][0]]
    plain = GeneratedModel(mesh=model.mesh, scene=model.scene,
                           glb_bytes=_serialize_glb(document, binary))
    result = extract_material_packs(plain)
    assert result.stripped_model is plain
    assert result.packs == ()


@pytest.mark.parametrize("size", (0, 257, -1, True, 10.5))
def test_invalid_preview_dimensions_are_rejected(model, size):
    with pytest.raises((TypeError, ValueError)):
        extract_material_packs(model, preview_size=size)


def test_missing_texture_dependencies_fail_before_any_result(model):
    document, binary = _parse(model.glb_bytes)
    document["materials"][0]["normalTexture"]["index"] = 100
    broken = GeneratedModel(mesh=model.mesh, scene=model.scene,
                            glb_bytes=_serialize_glb(document, binary))
    with pytest.raises(ValueError, match="unavailable texture"):
        extract_material_packs(broken)


def test_codec_full_image_and_extension_remain_only_in_the_material_pack(model):
    document, binary = _parse(model.glb_bytes)
    # A codec source can coexist with a regular fallback. Its full dependency
    # remains in the pack; all bounded main-file images use ordinary PNG refs.
    document["textures"][0]["extensions"] = {"EXT_texture_webp": {"source": 0}}
    document["extensionsUsed"].append("EXT_texture_webp")
    document["extensionsRequired"].append("EXT_texture_webp")
    codec_model = GeneratedModel(mesh=model.mesh, scene=model.scene,
                                glb_bytes=_serialize_glb(document, binary))
    result = extract_material_packs(codec_model)
    preview_document, _binary = _parse(result.stripped_model.glb_bytes)
    assert "EXT_texture_webp" not in preview_document["extensionsRequired"]
    assert "extensions" not in preview_document["textures"][0]
    pack_document, _binary = _parse(result.packs[0].glb_bytes)
    assert "EXT_texture_webp" in pack_document["extensionsRequired"]
    assert pack_document["textures"][0]["extensions"]["EXT_texture_webp"] == {"source": 0}


def test_real_atlas_export_keeps_full_pbr_and_stable_atlas_identity(tmp_path):
    from housemaker.atlas_export import (
        MaterializedTextureAtlas,
        apply_texture_atlases_to_export,
    )
    from housemaker.pbr_maps import (
        ATLAS_MAP_BASE_COLOR,
        PBR_MAP_METALLIC,
        PBR_MAP_NORMAL,
        PBR_MAP_ROUGHNESS,
    )
    from housemaker.texture_atlas_state import (
        ATLAS_TYPE_OBJECT,
        TextureAtlasPlacement,
        TextureAtlasRecord,
    )
    from tests.test_atlas_export import _textured_triangle, _write_atlas_maps

    mesh = _textured_triangle(name="Original", metadata={"housemaker_object_id": "source"})
    scene = trimesh.Scene(mesh)
    original = GeneratedModel(mesh=mesh, scene=scene, glb_bytes=b"")
    atlas = TextureAtlasRecord(
        atlas_id="actual-atlas", name="[OBJECT] Actual", resolution=2048,
        atlas_type=ATLAS_TYPE_OBJECT,
        placements=[TextureAtlasPlacement("source", "source.png", 512, 0, 0, 512)],
    )
    materialized = MaterializedTextureAtlas(
        atlas, _write_atlas_maps(tmp_path, 2048),
        active_map_types=frozenset((ATLAS_MAP_BASE_COLOR, PBR_MAP_NORMAL,
                                   PBR_MAP_ROUGHNESS, PBR_MAP_METALLIC)),
    )
    exported = apply_texture_atlases_to_export(original, (materialized,))
    result = extract_material_packs(exported)
    assert len(result.packs) == 1
    assert result.packs[0].atlas_id == "actual-atlas"
    pack_document, pack_binary = _parse(result.packs[0].glb_bytes)
    material = pack_document["materials"][0]
    assert "normalTexture" in material
    assert material["occlusionTexture"]["index"] == material["pbrMetallicRoughness"]["metallicRoughnessTexture"]["index"]
    image = pack_document["images"][0]
    with Image.open(BytesIO(_bytes(pack_document, pack_binary, image["bufferView"]))) as decoded:
        assert decoded.size == (2048, 2048)
