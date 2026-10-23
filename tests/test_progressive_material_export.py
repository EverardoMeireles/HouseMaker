# ### Imports ###
from __future__ import annotations

import copy
from dataclasses import replace
from io import BytesIO

import pytest
from PIL import Image

from housemaker.material_pack_export import (
    build_progressive_material_export,
    extract_material_packs,
)
from housemaker.scene_export import _serialize_glb
from tests.test_material_pack_export import _bytes, _parse, _view

# ### Shared test fixtures ###
pytest_plugins = ("tests.test_material_pack_export",)


# ### Representative full-resolution PBR fixture ###
@pytest.fixture(scope="module")
def large_model(model):
    document, old_binary = _parse(model.glb_bytes)
    for material in document["materials"]:
        material["alphaMode"] = "OPAQUE"
    old_document = copy.deepcopy(document)
    document["bufferViews"] = []
    binary = bytearray()
    for index in range(4):
        _view(document, binary, _bytes(old_document, old_binary, index))
    for index, image in enumerate(document["images"]):
        output = BytesIO()
        Image.new("RGBA", (2048, 1024), (40 + index * 30, 70, 90, 255)).save(
            output, format="PNG",
        )
        image["bufferView"] = _view(document, binary, output.getvalue())
    document["buffers"][0]["byteLength"] = len(binary)
    return replace(model, glb_bytes=_serialize_glb(document, bytes(binary)))


def _image_sizes(payload):
    document, binary = _parse(payload)
    result = []
    for image in document.get("images", []):
        with Image.open(BytesIO(_bytes(document, binary, image["bufferView"]))) as source:
            result.append(source.size)
    return result


# ### Full, prefetch and bootstrap tiers ###
def test_progressive_export_keeps_full_pbr_and_same_source_pixels(large_model):
    original, original_binary = _parse(large_model.glb_bytes)
    result = build_progressive_material_export(large_model)
    assert len(result.packs) == len(result.preview_packs) == 2
    assert result.all_packs == result.preview_packs + result.packs
    for pack in result.packs:
        document, binary = _parse(pack.glb_bytes)
        material = document["materials"][0]
        assert "normalTexture" in material
        assert "occlusionTexture" in material
        assert "metallicRoughnessTexture" in material["pbrMetallicRoughness"]
        assert pack.tier == "full" and pack.resolution == 2048
        assert {"normal", "metallic_roughness", "occlusion"}.issubset(pack.map_types)
        for image in document["images"]:
            source = next(entry for entry in original["images"] if entry["name"] == image["name"])
            assert _bytes(document, binary, image["bufferView"]) == _bytes(
                original, original_binary, source["bufferView"],
            )


def test_prefetch_materials_are_lit_1024_base_color_cubes(large_model):
    result = build_progressive_material_export(large_model)
    for pack in result.preview_packs:
        document, _binary = _parse(pack.glb_bytes)
        assert len(document["meshes"]) == len(document["materials"]) == 1
        material = document["materials"][0]
        assert "normalTexture" not in material and "occlusionTexture" not in material
        assert "metallicRoughnessTexture" not in material["pbrMetallicRoughness"]
        assert material["pbrMetallicRoughness"]["roughnessFactor"] == pytest.approx(0.8 * 70 / 255)
        assert material["pbrMetallicRoughness"]["metallicFactor"] == pytest.approx(90 / 255)
        assert material["extensions"]["KHR_materials_clearcoat"] == {"clearcoatFactor": 0.4}
        assert "KHR_materials_unlit" not in material.get("extensions", {})
        assert _image_sizes(pack.glb_bytes) == [(1024, 512)]
        assert pack.tier == "preview" and pack.resolution == 1024
        assert pack.map_types == ("base_color",)
        assert pack.file_name.endswith(".preview.glb")
        assert pack.to_dict()["byteSize"] == len(pack.glb_bytes)
        assert len(pack.to_dict()["sha256"]) == 64


def test_main_has_1024_bootstrap_and_128_elsewhere_without_uv_changes(large_model):
    first = build_progressive_material_export(large_model)
    bootstrap_id = first.packs[0].material_id
    result = build_progressive_material_export(
        large_model, bootstrap_material_ids=(bootstrap_id,),
    )
    document, binary = _parse(result.stripped_model.glb_bytes)
    original, original_binary = _parse(large_model.glb_bytes)
    assert sorted(_image_sizes(result.stripped_model.glb_bytes)) == [(128, 64), (1024, 512)]
    for key in ("nodes", "meshes", "accessors", "scenes", "scene", "asset"):
        assert document[key] == original[key]
    for accessor in document["accessors"]:
        index = accessor["bufferView"]
        assert _bytes(document, binary, index) == _bytes(original, original_binary, index)
    assert all(image["mimeType"] == "image/jpeg" for image in document["images"])
    assert len(document["textures"]) == 2
    assert result.stripped_model.scene is large_model.scene
    assert result.stripped_model.mesh is large_model.mesh
    assert [pack.file_name for pack in result.packs] == [pack.file_name for pack in first.packs]
    assert [pack.material_id for pack in result.packs] == [pack.material_id for pack in first.packs]
    assert large_model.glb_bytes == _serialize_glb(original, original_binary)


def test_no_bootstrap_keeps_every_material_visible_and_no_orphan_full_maps(large_model):
    result = build_progressive_material_export(large_model)
    document, binary = _parse(result.stripped_model.glb_bytes)
    assert _image_sizes(result.stripped_model.glb_bytes) == [(128, 64), (128, 64)]
    original, original_binary = _parse(large_model.glb_bytes)
    for image in original["images"]:
        assert _bytes(original, original_binary, image["bufferView"]) not in binary
    for pack in result.packs:
        material = document["materials"][pack.material_index]
        assert material["extras"]["housemaker_material_id"] == pack.material_id
        assert "baseColorTexture" in material["pbrMetallicRoughness"]


def test_bootstrap_indices_allow_one_pass_material_identity_assignment(large_model):
    result = build_progressive_material_export(
        large_model, bootstrap_material_indices=(1,),
    )
    assert _image_sizes(result.stripped_model.glb_bytes) == [(128, 64), (1024, 512)]


# ### Transparency, glass and optional compression ###
def test_alpha_emissive_and_transmission_remain_in_previews(model):
    document, binary = _parse(model.glb_bytes)
    document["materials"][0].update({
        "emissiveTexture": {"index": 3}, "emissiveFactor": [0.1, 0.2, 0.3],
    })
    document["materials"][0]["extensions"].update({
        "KHR_materials_transmission": {
            "transmissionFactor": 0.75, "transmissionTexture": {"index": 0},
        },
        "KHR_materials_volume": {
            "thicknessFactor": 0.2, "thicknessTexture": {"index": 3},
            "attenuationColor": [0.9, 0.8, 0.7],
        },
    })
    source = replace(model, glb_bytes=_serialize_glb(document, binary))
    result = build_progressive_material_export(source)
    for payload in (result.preview_packs[0].glb_bytes, result.stripped_model.glb_bytes):
        preview, _preview_binary = _parse(payload)
        material = preview["materials"][0]
        assert material["alphaMode"] == "MASK" and material["alphaCutoff"] == 0.3
        assert material["doubleSided"] is True
        assert material["emissiveFactor"] == [0.1, 0.2, 0.3]
        assert "emissiveTexture" in material
        assert material["extensions"]["KHR_materials_transmission"]["transmissionFactor"] == 0.75
        assert "transmissionTexture" in material["extensions"]["KHR_materials_transmission"]
        assert "thicknessTexture" in material["extensions"]["KHR_materials_volume"]
        assert all(image["mimeType"] == "image/png" for image in preview["images"])


def test_small_images_are_not_upscaled(model):
    result = build_progressive_material_export(model)
    assert all(pack.resolution == 768 for pack in result.preview_packs)


def test_removed_black_metallic_map_does_not_make_preview_metal(large_model):
    document, binary = _parse(large_model.glb_bytes)
    output = BytesIO()
    Image.new("RGBA", (2048, 1024), (255, 204, 0, 255)).save(output, format="PNG")
    output_binary = bytearray(binary)
    document["images"][2]["bufferView"] = _view(document, output_binary, output.getvalue())
    document["buffers"][0]["byteLength"] = len(output_binary)
    # Rewrite the abandoned old image view so it does not inflate this fixture.
    old_view = document["bufferViews"][6]
    old_view["byteLength"] = 4
    source = replace(large_model, glb_bytes=_serialize_glb(document, bytes(output_binary)))
    result = build_progressive_material_export(source)
    for payload in (result.stripped_model.glb_bytes, result.preview_packs[0].glb_bytes):
        preview, _preview_binary = _parse(payload)
        pbr = preview["materials"][0]["pbrMetallicRoughness"]
        assert pbr["metallicFactor"] == 0.0
        assert pbr["roughnessFactor"] == pytest.approx(0.64)
    full, _full_binary = _parse(result.packs[0].glb_bytes)
    assert full["materials"][0]["pbrMetallicRoughness"]["metallicFactor"] == 1.0


def test_preview_orm_average_ignores_transparent_atlas_space(large_model):
    document, binary = _parse(large_model.glb_bytes)
    document["materials"][0]["alphaMode"] = "MASK"
    output_binary = bytearray(binary)
    for index in (0, 2):
        image = Image.new("RGBA", (8, 8), (255, 0, 255, 0))
        image.paste((100, 100, 100, 255) if index == 0 else (255, 153, 0, 255), (0, 0, 4, 8))
        output = BytesIO()
        image.save(output, format="PNG")
        document["images"][index]["bufferView"] = _view(document, output_binary, output.getvalue())
    document["buffers"][0]["byteLength"] = len(output_binary)
    source = replace(large_model, glb_bytes=_serialize_glb(document, bytes(output_binary)))
    result = build_progressive_material_export(source)
    preview, _preview_binary = _parse(result.preview_packs[0].glb_bytes)
    pbr = preview["materials"][0]["pbrMetallicRoughness"]
    assert pbr["metallicFactor"] == 0.0
    assert pbr["roughnessFactor"] == pytest.approx(0.48)


@pytest.mark.parametrize("alpha_mode, mime_type", (("OPAQUE", "image/jpeg"), ("BLEND", "image/png"), ("MASK", "image/png")))
def test_only_opaque_base_color_can_drop_unused_alpha_padding(model, alpha_mode, mime_type):
    document, binary = _parse(model.glb_bytes)
    for material in document["materials"]:
        material["alphaMode"] = alpha_mode
    source = replace(model, glb_bytes=_serialize_glb(document, binary))
    result = build_progressive_material_export(source)
    for payload in (result.stripped_model.glb_bytes, *(pack.glb_bytes for pack in result.preview_packs)):
        preview, _preview_binary = _parse(payload)
        assert all(image["mimeType"] == mime_type for image in preview["images"])


def test_shared_image_keeps_png_when_any_material_uses_transparency(model):
    document, binary = _parse(model.glb_bytes)
    document["materials"][0]["alphaMode"] = "OPAQUE"
    document["materials"][1]["alphaMode"] = "BLEND"
    document["materials"][1]["pbrMetallicRoughness"]["baseColorTexture"]["index"] = 0
    source = replace(model, glb_bytes=_serialize_glb(document, binary))
    result = build_progressive_material_export(source)
    preview, _preview_binary = _parse(result.stripped_model.glb_bytes)
    assert len(preview["images"]) == 1
    assert preview["images"][0]["mimeType"] == "image/png"


def test_opaque_image_used_as_transmission_or_emission_mask_keeps_png(model):
    document, binary = _parse(model.glb_bytes)
    for material in document["materials"]:
        material["alphaMode"] = "OPAQUE"
        material["emissiveTexture"] = {"index": 0}
    source = replace(model, glb_bytes=_serialize_glb(document, binary))
    result = build_progressive_material_export(source)
    for pack in result.preview_packs:
        preview, _preview_binary = _parse(pack.glb_bytes)
        source_texture = preview["textures"][preview["materials"][0]["emissiveTexture"]["index"]]
        assert preview["images"][source_texture["source"]]["mimeType"] == "image/png"


def test_ktx2_applies_only_to_external_cubes(large_model, monkeypatch):
    import housemaker.ktx2_compression as compression

    calls = []

    def compress(payload, **kwargs):
        calls.append((payload, kwargs))
        return payload

    monkeypatch.setattr(compression, "compress_material_glb", compress)
    result = build_progressive_material_export(
        large_model, use_ktx2_basis_compression=True,
    )
    assert len(calls) == 4
    assert [kwargs for _payload, kwargs in calls] == [{}, {}, {"preview": True}, {"preview": True}]
    assert all(pack.compression == "ktx2-basis" for pack in result.all_packs)
    document, _binary = _parse(result.stripped_model.glb_bytes)
    assert "KHR_texture_basisu" not in document.get("extensionsRequired", [])


def test_legacy_extraction_keeps_all_256px_pbr_previews(large_model):
    result = extract_material_packs(large_model)
    document, _binary = _parse(result.stripped_model.glb_bytes)
    assert all(max(size) == 256 for size in _image_sizes(result.stripped_model.glb_bytes))
    assert "normalTexture" in document["materials"][0]
    assert result.preview_packs == ()
    assert result.all_packs == result.packs
