# ### Imports ###
from __future__ import annotations

import json
import math
import os
import struct
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.glb import (
    PACKED_ORM_AO_UV_ATTRIBUTE,
    PackedOrmMaterialSpec,
    _inject_packed_orm_occlusion_textures,
    _serialize_scene_glb_with_half_mesh_extras,
)

# ### Constants ###
MATERIAL_MARKER = "__housemaker_packed_orm__:fixture"


# ### Fixture helpers ###
def _build_packed_orm_scene(*, include_ao_uv: bool) -> trimesh.Scene:
    material = PBRMaterial(
        name=MATERIAL_MARKER,
        baseColorTexture=Image.new("RGBA", (2, 2), (80, 90, 100, 255)),
        metallicRoughnessTexture=Image.new(
            "RGBA",
            (2, 2),
            (255, 170, 20, 255),
        ),
    )
    mesh = trimesh.Trimesh(
        vertices=np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=float,
        ),
        faces=np.asarray(((0, 1, 2),), dtype=np.int64),
        visual=TextureVisuals(
            uv=np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
            material=material,
        ),
        process=False,
    )
    if include_ao_uv:
        mesh.vertex_attributes[PACKED_ORM_AO_UV_ATTRIBUTE] = np.asarray(
            ((0.2, 0.8), (0.8, 0.8), (0.2, 0.2)),
            dtype=np.float32,
        )
    return trimesh.Scene(mesh)


def _load_glb_json(payload: bytes) -> dict[str, object]:
    json_length, json_type = struct.unpack_from("<II", payload, 12)
    if json_type != 0x4E4F534A:
        raise AssertionError("The first GLB chunk is not JSON.")
    return json.loads(payload[20 : 20 + json_length].decode("utf-8"))


def _minimal_packed_orm_document(
    *,
    ao_count: int = 3,
    include_custom_uv: bool = True,
    include_standard_uv1: bool = False,
) -> dict[str, object]:
    attributes: dict[str, int] = {"POSITION": 0}
    if include_custom_uv:
        attributes[PACKED_ORM_AO_UV_ATTRIBUTE] = 1
    if include_standard_uv1:
        attributes["TEXCOORD_1"] = 1
    return {
        "materials": [
            {
                "name": MATERIAL_MARKER,
                "pbrMetallicRoughness": {
                    "metallicRoughnessTexture": {"index": 0},
                },
            }
        ],
        "textures": [{}],
        "accessors": [{"count": 3}, {"count": ao_count}],
        "meshes": [
            {
                "primitives": [
                    {
                        "material": 0,
                        "attributes": attributes,
                    }
                ]
            }
        ],
    }


# ### Packed ORM material tests ###
class PackedOrmMaterialSpecTests(unittest.TestCase):
    def test_spec_validates_texture_coordinate_and_strength(self) -> None:
        for invalid_tex_coord in (True, -1, 2, 1.5):
            with (
                self.subTest(ao_tex_coord=invalid_tex_coord),
                self.assertRaisesRegex(ValueError, "coordinate"),
            ):
                PackedOrmMaterialSpec(
                    "Atlas",
                    ao_tex_coord=invalid_tex_coord,  # type: ignore[arg-type]
                )
        for invalid_strength in (True, -0.01, 1.01, math.nan):
            with (
                self.subTest(ao_strength=invalid_strength),
                self.assertRaisesRegex(ValueError, "strength"),
            ):
                PackedOrmMaterialSpec(
                    "Atlas",
                    ao_strength=invalid_strength,
                )


# ### GLB serialization tests ###
class PackedOrmGlbSerializationTests(unittest.TestCase):
    def test_uv1_spec_exports_shared_orm_texture_and_promotes_attribute(
        self,
    ) -> None:
        payload = _serialize_scene_glb_with_half_mesh_extras(
            _build_packed_orm_scene(include_ao_uv=True),
            failure_message="serialization failed",
            packed_orm_material_names={
                MATERIAL_MARKER: PackedOrmMaterialSpec(
                    final_name="Surface Atlas",
                    ao_tex_coord=1,
                    ao_strength=0.35,
                )
            },
        )

        document = _load_glb_json(payload)
        material = document["materials"][0]
        pbr = material["pbrMetallicRoughness"]
        metallic_roughness = pbr["metallicRoughnessTexture"]
        occlusion = material["occlusionTexture"]
        self.assertEqual(material["name"], "Surface Atlas")
        self.assertEqual(occlusion["index"], metallic_roughness["index"])
        self.assertNotIn("texCoord", metallic_roughness)
        self.assertEqual(occlusion["texCoord"], 1)
        self.assertEqual(occlusion["strength"], 0.35)

        attributes = document["meshes"][0]["primitives"][0]["attributes"]
        self.assertIn("TEXCOORD_0", attributes)
        self.assertIn("TEXCOORD_1", attributes)
        self.assertNotIn(PACKED_ORM_AO_UV_ATTRIBUTE, attributes)
        accessors = document["accessors"]
        self.assertEqual(
            accessors[attributes["POSITION"]]["count"],
            accessors[attributes["TEXCOORD_1"]]["count"],
        )

    def test_legacy_name_mapping_keeps_uv0_object_ao_behavior(self) -> None:
        payload = _serialize_scene_glb_with_half_mesh_extras(
            _build_packed_orm_scene(include_ao_uv=False),
            failure_message="serialization failed",
            packed_orm_material_names={MATERIAL_MARKER: "Object Atlas"},
        )

        document = _load_glb_json(payload)
        material = document["materials"][0]
        metallic_roughness = material["pbrMetallicRoughness"][
            "metallicRoughnessTexture"
        ]
        occlusion = material["occlusionTexture"]
        self.assertEqual(material["name"], "Object Atlas")
        self.assertEqual(occlusion["index"], metallic_roughness["index"])
        self.assertNotIn("texCoord", occlusion)
        self.assertEqual(occlusion["strength"], 1.0)

    def test_uv1_promotion_rejects_attribute_collision(self) -> None:
        document = _minimal_packed_orm_document(include_standard_uv1=True)

        with self.assertRaisesRegex(ValueError, "conflicting AO UV"):
            _inject_packed_orm_occlusion_textures(
                document,
                {
                    MATERIAL_MARKER: PackedOrmMaterialSpec(
                        "Surface Atlas",
                        ao_tex_coord=1,
                    )
                },
            )

    def test_uv1_promotion_rejects_accessor_count_mismatch(self) -> None:
        document = _minimal_packed_orm_document(ao_count=2)

        with self.assertRaisesRegex(ValueError, "count does not match"):
            _inject_packed_orm_occlusion_textures(
                document,
                {
                    MATERIAL_MARKER: PackedOrmMaterialSpec(
                        "Surface Atlas",
                        ao_tex_coord=1,
                    )
                },
            )

    def test_uv1_material_requires_an_ao_uv_attribute(self) -> None:
        document = _minimal_packed_orm_document(include_custom_uv=False)

        with self.assertRaisesRegex(ValueError, "no AO UV attribute"):
            _inject_packed_orm_occlusion_textures(
                document,
                {
                    MATERIAL_MARKER: PackedOrmMaterialSpec(
                        "Surface Atlas",
                        ao_tex_coord=1,
                    )
                },
            )


if __name__ == "__main__":
    unittest.main()
