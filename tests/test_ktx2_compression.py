# ### Imports ###
from __future__ import annotations

import copy
import hashlib
import struct
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from housemaker import ktx2_compression as module
from housemaker.scene_export import _parse_glb, _serialize_glb


# ### Small GLB and image fixtures ###
def _png(size=(16, 16)):
    output = BytesIO()
    Image.new("RGBA", size, (40, 100, 170, 255)).save(output, format="PNG")
    return output.getvalue()


def _append_view(document, binary, payload):
    binary.extend(b"\0" * (-len(binary) % 4))
    document["bufferViews"].append(
        {"buffer": 0, "byteOffset": len(binary), "byteLength": len(payload)}
    )
    binary.extend(payload)
    return len(document["bufferViews"]) - 1


@pytest.fixture
def material_glb():
    document = {
        "asset": {"version": "2.0", "extras": {"materialId": "material_test"}},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {
                "mesh": 0,
                "name": "Object_12",
                "translation": [2, 3, 4],
                "extras": {"halfMesh": True, "housemaker_room_ids": ["room:1:a"]},
            }
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "TEXCOORD_0": 1, "TEXCOORD_1": 1},
                        "material": 0,
                    }
                ]
            }
        ],
        "materials": [
            {
                "name": "material",
                "extras": {"housemaker_material_id": "material_test"},
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicRoughnessTexture": {"index": 2},
                },
                "normalTexture": {"index": 1, "scale": 0.4},
                "occlusionTexture": {"index": 2, "texCoord": 1, "strength": 0.7},
                "alphaMode": "MASK",
                "alphaCutoff": 0.25,
                "doubleSided": True,
            }
        ],
        "textures": [
            {"source": 0, "sampler": 0},
            {"source": 1, "sampler": 0},
            {"source": 1, "sampler": 0},
        ],
        "samplers": [{"wrapS": 10497, "wrapT": 33071, "minFilter": 9987}],
        "images": [],
        "bufferViews": [],
        "buffers": [{"byteLength": 0}],
        "accessors": [],
    }
    binary = bytearray()
    for index in range(2):
        document["images"].append(
            {
                "name": f"map-{index}",
                "mimeType": "image/png",
                "bufferView": _append_view(document, binary, _png()),
            }
        )
    for raw, kind in ((bytes(range(36)), "VEC3"), (bytes(range(24)), "VEC2")):
        view = _append_view(document, binary, raw)
        document["accessors"].append(
            {"bufferView": view, "componentType": 5126, "count": 3, "type": kind}
        )
    document["buffers"][0]["byteLength"] = len(binary)
    return _serialize_glb(document, bytes(binary))


def _parse(payload):
    return _parse_glb(payload, "Invalid test GLB")


def _view(document, binary, index):
    view = document["bufferViews"][index]
    return binary[
        view.get("byteOffset", 0) : view.get("byteOffset", 0) + view["byteLength"]
    ]


def _fake_encoder(image, *, color_space):
    return (
        module.KTX2_IDENTIFIER + color_space.encode() + hashlib.sha256(image).digest()
    )


# ### Exact geometry identity, material channels, and buffer compaction ###
def test_material_compression_preserves_geometry_uv1_identity_and_parameters(
    material_glb,
):
    original, original_binary = _parse(material_glb)
    result, binary = _parse(
        module.compress_material_glb(material_glb, encoder=_fake_encoder)
    )
    for field in ("asset", "nodes", "meshes", "materials", "samplers"):
        assert result[field] == original[field]
    for accessor, before in zip(result["accessors"], original["accessors"]):
        assert _view(result, binary, accessor["bufferView"]) == _view(
            original, original_binary, before["bufferView"]
        )
        assert {
            key: value for key, value in accessor.items() if key != "bufferView"
        } == {key: value for key, value in before.items() if key != "bufferView"}
    assert len(result["images"]) == 2  # normal and ORM share one linear image.
    assert len(result["bufferViews"]) == 4  # old full PNG views are actually removed.
    assert all(image["mimeType"] == "image/ktx2" for image in result["images"])
    assert all("source" not in texture for texture in result["textures"])
    assert module.KTX2_EXTENSION in result["extensionsRequired"]
    assert module.KTX2_EXTENSION in result["extensionsUsed"]
    assert b"IHDR" not in binary


def test_color_and_linear_uses_of_one_texture_get_independent_transfer_functions(
    material_glb,
):
    document, binary = _parse(material_glb)
    document["materials"][0]["normalTexture"]["index"] = 0
    calls = []

    def encode(image, *, color_space):
        calls.append(color_space)
        return _fake_encoder(image, color_space=color_space)

    result, _binary = _parse(
        module.compress_material_glb(_serialize_glb(document, binary), encoder=encode)
    )
    material = result["materials"][0]
    assert (
        material["normalTexture"]["index"]
        != material["pbrMetallicRoughness"]["baseColorTexture"]["index"]
    )
    assert set(calls) == {"srgb", "linear"}
    assert len(result["textures"]) == 4


def test_existing_ktx2_images_are_preserved_without_reencoding(material_glb):
    compressed = module.compress_material_glb(material_glb, encoder=_fake_encoder)

    def fail(*args, **kwargs):
        pytest.fail("Already compressed KTX2 was encoded again")

    assert module.compress_material_glb(compressed, encoder=fail) == compressed


def test_webp_codec_and_its_full_fallback_payload_are_removed(material_glb):
    document, binary = _parse(material_glb)
    document["textures"][0]["extensions"] = {"EXT_texture_webp": {"source": 1}}
    document["extensionsRequired"] = ["EXT_texture_webp", "KHR_materials_ior"]
    result, _binary = _parse(
        module.compress_material_glb(
            _serialize_glb(document, binary), encoder=_fake_encoder
        )
    )
    assert "EXT_texture_webp" not in result["extensionsRequired"]
    assert "KHR_materials_ior" in result["extensionsRequired"]
    assert all(
        "EXT_texture_webp" not in texture["extensions"]
        for texture in result["textures"]
    )


def test_shared_image_geometry_view_is_not_overwritten(material_glb):
    document, binary = _parse(material_glb)
    document["accessors"][0]["bufferView"] = 0
    original = _view(document, binary, 0)
    result, output = _parse(
        module.compress_material_glb(
            _serialize_glb(document, binary), encoder=_fake_encoder
        )
    )
    assert _view(result, output, result["accessors"][0]["bufferView"]) == original
    assert _view(result, output, result["images"][0]["bufferView"]).startswith(
        module.KTX2_IDENTIFIER
    )


# ### Actionable setup and encoder failures ###
def test_missing_encoder_never_silently_exports_uncompressed(monkeypatch):
    monkeypatch.setattr(module, "find_ktx2_encoder", lambda: None)
    with pytest.raises(
        module.Ktx2CompressionError, match="Uncheck 'Use KTX2/Basis compression'"
    ):
        module.require_ktx2_encoder()


def test_explicit_encoder_override_is_authoritative(monkeypatch, tmp_path):
    monkeypatch.setenv("HOUSEMAKER_TOKTX", str(tmp_path / "missing.exe"))
    assert module.find_ktx2_encoder() is None
    encoder = tmp_path / "toktx.exe"
    encoder.touch()
    assert module.require_ktx2_encoder(encoder) == encoder.resolve()


def test_nonzero_encoder_exit_is_reported_and_shell_overrides_are_removed(monkeypatch):
    monkeypatch.setenv("TOKTX_OPTIONS", "--normal_mode --lower_left_maps_to_s0t0")
    captured = []

    def run(command, **kwargs):
        captured.append(kwargs)
        return SimpleNamespace(returncode=2, stdout="", stderr="Cannot encode image")

    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises(module.Ktx2CompressionError, match="Cannot encode image"):
        module._run_encoder_command(["toktx"])
    assert "TOKTX_OPTIONS" not in captured[0]["env"]
    assert captured[0]["capture_output"] is True


def test_invalid_image_codec_output_is_not_published(material_glb):
    with pytest.raises(module.Ktx2CompressionError, match="invalid KTX2 bytes"):
        module.compress_material_glb(
            material_glb, encoder=lambda *args, **kwargs: b"not ktx2"
        )


def test_external_images_and_meshopt_buffers_are_rejected(material_glb):
    original, binary = _parse(material_glb)
    document = copy.deepcopy(original)
    document["images"][0] = {"uri": "external.png"}
    with pytest.raises(module.Ktx2CompressionError, match="embedded material images"):
        module.compress_material_glb(
            _serialize_glb(document, binary), encoder=_fake_encoder
        )
    document = copy.deepcopy(original)
    document["bufferViews"][0]["extensions"] = {"EXT_meshopt_compression": {}}
    with pytest.raises(module.Ktx2CompressionError, match="meshopt"):
        module.compress_material_glb(
            _serialize_glb(document, binary), encoder=_fake_encoder
        )


# ### Optional real official-encoder smoke and cache verification ###
@pytest.mark.parametrize(
    "color_space,primaries,transfer,preview,supercompression",
    [
        ("srgb", 1, 2, False, 2),
        ("linear", 0, 1, False, 2),
        ("srgb", 1, 2, True, 1),
        ("linear", 0, 1, True, 2),
    ],
)
def test_real_encoder_produces_conformant_mips_and_reuses_cache(
    tmp_path, monkeypatch, color_space, primaries, transfer, preview, supercompression
):
    if module.find_ktx2_encoder() is None:
        pytest.skip("Optional official KTX Software tool is not installed")
    result = module.encode_ktx2_image(
        _png(),
        color_space=color_space,
        cache_directory=tmp_path,
        preview=preview,
    )
    assert result.startswith(module.KTX2_IDENTIFIER)
    width, height = struct.unpack_from("<II", result, 20)
    assert (width, height) == (16, 16)
    assert struct.unpack_from("<I", result, 40)[0] == 5  # complete 16 -> 1 mip pyramid.
    assert struct.unpack_from("<I", result, 44)[0] == supercompression
    dfd_offset = struct.unpack_from("<I", result, 48)[0]
    assert result[dfd_offset + 13 : dfd_offset + 15] == bytes((primaries, transfer))
    validator = module.require_ktx2_encoder().with_name(
        "ktx.exe" if module.os.name == "nt" else "ktx"
    )
    if validator.is_file():
        image_path = next(tmp_path.glob("*.ktx2"))
        module._run_encoder_command(
            [str(validator), "validate", "--gltf-basisu", str(image_path)]
        )
        # Exercise the desktop GPU transcode path and the portable RGBA path.
        module._run_encoder_command(
            [
                str(validator),
                "extract",
                "--transcode",
                "bc7",
                "--raw",
                str(image_path),
                str(tmp_path / "level0.bc7"),
            ]
        )
        assert (tmp_path / "level0.bc7").stat().st_size == 16 * 16
        module._run_encoder_command(
            [
                str(validator),
                "extract",
                "--transcode",
                "rgba8",
                str(image_path),
                str(tmp_path / "level0.png"),
            ]
        )
        with Image.open(tmp_path / "level0.png") as image:
            assert image.size == (16, 16)
            pixel = image.convert("RGBA").getpixel((8, 8))
            assert all(abs(a - b) <= 8 for a, b in zip(pixel, (40, 100, 170, 255)))

    def fail(*args, **kwargs):
        pytest.fail("Cached image should not launch the encoder")

    monkeypatch.setattr(module, "_run_encoder_command", fail)
    assert (
        module.encode_ktx2_image(
            _png(),
            color_space=color_space,
            cache_directory=tmp_path,
            preview=preview,
        )
        == result
    )


def test_non_multiple_of_four_image_resamples_without_uv_shifting_padding(
    tmp_path, monkeypatch
):
    encoder = tmp_path / "toktx.exe"
    encoder.touch()
    captured = []

    def run(command, **kwargs):
        with Image.open(command[-1]) as image:
            captured.append(
                (image.size, image.getpixel((image.width - 1, image.height - 1)))
            )
        Path(command[-2]).write_bytes(module.KTX2_IDENTIFIER + b"mock")

    monkeypatch.setattr(module, "_run_encoder_command", run)
    module.encode_ktx2_image(
        _png((7, 9)), color_space="srgb", tool_path=encoder, cache_directory=tmp_path
    )
    assert captured == [((8, 12), (40, 100, 170, 255))]
