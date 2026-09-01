# ### Imports ###
from __future__ import annotations

import json
import struct

import numpy as np
import trimesh
from trimesh.visual.texture import TextureVisuals

from housemaker.glass_material import (
    HOUSEMAKER_GLASS_MATERIAL_NAME,
    build_housemaker_glass_material,
    get_housemaker_glass_runtime_key,
)


# ### Fixture helpers ###
def _build_glass_triangle(x_offset: float, *, double_sided: bool):
    mesh = trimesh.Trimesh(
        vertices=np.asarray(
            (
                (x_offset, 0.0, 0.0),
                (x_offset + 1.0, 0.0, 0.0),
                (x_offset, 1.0, 0.0),
            ),
            dtype=float,
        ),
        faces=np.asarray(((0, 1, 2),), dtype=np.int64),
        process=False,
    )
    mesh.visual = TextureVisuals(
        uv=np.zeros((3, 2), dtype=float),
        material=build_housemaker_glass_material(double_sided),
    )
    return mesh


def _read_glb_json(payload: bytes) -> dict[str, object]:
    json_length, json_type = struct.unpack_from("<II", payload, 12)
    assert json_type == 0x4E4F534A
    return json.loads(
        payload[20 : 20 + json_length].decode("utf-8").rstrip("\0 ")
    )


# ### Tests ###
def test_prefab_glass_is_untextured_and_has_stable_runtime_keys() -> None:
    single_sided = build_housemaker_glass_material(False)
    double_sided = build_housemaker_glass_material(True)

    assert single_sided.name == HOUSEMAKER_GLASS_MATERIAL_NAME
    assert single_sided.baseColorTexture is None
    assert single_sided.alphaMode == "BLEND"
    assert single_sided.doubleSided is False
    assert double_sided.doubleSided is True
    assert get_housemaker_glass_runtime_key(single_sided) == (
        "housemaker_glass_v1:single_sided"
    )
    assert get_housemaker_glass_runtime_key(double_sided) == (
        "housemaker_glass_v1:double_sided"
    )


def test_same_side_panels_share_one_exported_gltf_material() -> None:
    scene = trimesh.Scene()
    scene.add_geometry(
        _build_glass_triangle(0.0, double_sided=True),
        geom_name="glass_a",
    )
    scene.add_geometry(
        _build_glass_triangle(2.0, double_sided=True),
        geom_name="glass_b",
    )
    scene.add_geometry(
        _build_glass_triangle(4.0, double_sided=False),
        geom_name="glass_single_sided",
    )

    document = _read_glb_json(scene.export(file_type="glb"))
    materials = document["materials"]
    primitive_materials = [
        primitive["material"]
        for mesh in document["meshes"]
        for primitive in mesh["primitives"]
    ]

    assert len(materials) == 2
    assert primitive_materials[0] == primitive_materials[1]
    assert primitive_materials[2] != primitive_materials[0]
    assert "images" not in document
    assert "textures" not in document
