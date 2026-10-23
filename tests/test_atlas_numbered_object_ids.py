"""Atlas remapping keeps stable object roots without duplicating materials."""

# ### Imports ###
from __future__ import annotations

import json
import struct
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.atlas_export import (
    MaterializedTextureAtlas,
    apply_texture_atlases_to_export,
    estimate_texture_atlas_draw_calls,
)
from housemaker.glass_material import build_housemaker_glass_material
from housemaker.glb import (
    GeneratedModel,
    PlacedGeneratedModel,
    compose_generated_model_instance_sources,
    compose_placed_generated_models,
)
from housemaker.pbr_maps import (
    ATLAS_MAP_BASE_COLOR,
    PBR_MAP_METALLIC,
    PBR_MAP_NORMAL,
    PBR_MAP_ROUGHNESS,
)
from housemaker.pvs_geometry import (
    collect_pvs_occluder_triangles,
    partition_glb_for_pvs,
)
from housemaker.texture_atlas_state import (
    ATLAS_TYPE_OBJECT,
    TextureAtlasPlacement,
    TextureAtlasRecord,
)


# ### Small scene and Atlas fixtures ###
def _triangle(*, glass=False):
    mesh = trimesh.Trimesh(
        vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        faces=((0, 1, 2),),
        process=False,
    )
    mesh.visual = TextureVisuals(
        uv=np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        material=(
            build_housemaker_glass_material(False)
            if glass
            else PBRMaterial(name="Original", doubleSided=False)
        ),
    )
    return mesh


def _source_model(*, glass=False):
    mesh = _triangle()
    scene = trimesh.Scene()
    scene.graph.update(
        frame_to="furniture-group",
        matrix=trimesh.transformations.translation_matrix((0.1, 0.2, 0.3)),
    )
    scene.add_geometry(mesh, node_name="wood", parent_node_name="furniture-group")
    if glass:
        scene.add_geometry(
            _triangle(glass=True), node_name="glass",
            parent_node_name="furniture-group",
        )
    return GeneratedModel(mesh=mesh, scene=scene, glb_bytes=b"")


def _empty_model():
    return GeneratedModel(
        mesh=trimesh.Trimesh(process=False), scene=trimesh.Scene(), glb_bytes=b"",
    )


def _placement(identifier, number, *, glass=False, half=False):
    return PlacedGeneratedModel(
        object_id=identifier,
        object_name="My cupboard",
        model=_source_model(glass=glass),
        world_position=(float(number) * 2.0, 3.0, 0.5),
        rotation_degrees=(0.0, 0.0, 20.0 * number),
        axis_scales=(1.5, 0.8, 1.2),
        export_node_name=f"Object_{number}",
        export_source_node_name=f"Object_{number}",
        symmetric_preview_orientation="vertical" if half else None,
        symmetric_preview_plane_coordinate=0.0 if half else None,
    )


@pytest.fixture
def materialized_atlas(tmp_path):
    colors = {
        ATLAS_MAP_BASE_COLOR: (120, 130, 140, 255),
        PBR_MAP_NORMAL: (128, 128, 255, 255),
        PBR_MAP_ROUGHNESS: (170, 170, 170, 255),
        PBR_MAP_METALLIC: (20, 20, 20, 255),
    }
    paths = {}
    for map_type, color in colors.items():
        path = tmp_path / f"{map_type}.png"
        Image.new("RGBA", (2048, 2048), color).save(path)
        paths[map_type] = path
    atlas = TextureAtlasRecord(
        atlas_id="objects", name="Furniture", atlas_type=ATLAS_TYPE_OBJECT,
        resolution=2048,
        placements=[
            TextureAtlasPlacement(
                object_id=identifier, texture_path=f"{identifier}.png",
                texture_resolution=512, x=index * 512, y=0, size=512,
            )
            for index, identifier in enumerate(("uuid-a", "uuid-b"))
        ],
    )
    return MaterializedTextureAtlas(atlas, paths)


def _document(payload):
    json_size, chunk_type = struct.unpack_from("<II", payload, 12)
    assert chunk_type == 0x4E4F534A
    return json.loads(payload[20 : 20 + json_size])


def _export(model, atlas):
    with patch(
        "housemaker.atlas_export.bake_placed_object_ambient_occlusion",
        return_value={},
    ):
        return apply_texture_atlases_to_export(model, (atlas,))


def _world_geometry(scene, node_name):
    transform, geometry_name = scene.graph.get(node_name)
    mesh = scene.geometry[geometry_name].copy()
    mesh.apply_transform(transform)
    return mesh


# ### Numbered ordinary-object export tests ###
def test_same_named_objects_keep_distinct_roots_uuid_extras_and_one_material(
    materialized_atlas,
):
    model = compose_placed_generated_models(
        _empty_model(), (_placement("uuid-a", 1), _placement("uuid-b", 2)),
    )
    exported = _export(model, materialized_atlas)
    document = _document(exported.glb_bytes)
    nodes = {node["name"]: node for node in document["nodes"]}
    assert len(document["materials"]) == 1
    material = document["materials"][0]
    assert material["name"] == "[OBJECT] Furniture"
    assert "normalTexture" in material
    assert "metallicRoughnessTexture" in material["pbrMetallicRoughness"]
    for number, identifier in ((1, "uuid-a"), (2, "uuid-b")):
        root = nodes[f"Object_{number}"]
        assert root["extras"] == {
            "housemaker_object_id": identifier,
            "housemaker_object_name": "My cupboard",
            "housemaker_export_object_id": f"Object {number}",
            "housemaker_atlas_type": ATLAS_TYPE_OBJECT,
            "housemaker_atlas_id": "objects",
        }
        assert len(root["children"]) == 1
        child = document["nodes"][root["children"][0]]
        primitive = document["meshes"][child["mesh"]]["primitives"][0]
        assert primitive["material"] == 0
        assert {"POSITION", "NORMAL", "TEXCOORD_0"} <= primitive["attributes"].keys()
        assert "housemaker_instance_source_id" not in root["extras"]
        np.testing.assert_allclose(
            exported.scene.graph.get(f"Object_{number}")[0],
            model.scene.graph.get(f"Object_{number}")[0],
        )
    actual_materials = [
        mesh.visual.material for mesh in exported.scene.geometry.values()
    ]
    assert actual_materials[0] is actual_materials[1]
    estimate = estimate_texture_atlas_draw_calls(model, (materialized_atlas.atlas,))
    assert estimate.exported_count == 2
    assert estimate.atlas_batch_count == 0


def test_remapping_preserves_world_geometry_normals_and_object_ao(materialized_atlas):
    model = compose_placed_generated_models(
        _empty_model(), (_placement("uuid-a", 1), _placement("uuid-b", 2)),
    )
    with patch(
        "housemaker.atlas_export.bake_placed_object_ambient_occlusion",
        return_value={},
    ) as bake:
        exported = apply_texture_atlases_to_export(model, (materialized_atlas,))
    assert len(bake.call_args.args[0]["objects"]) == 2
    assert len(bake.call_args.args[2]) == 2
    for number, identifier in ((1, "uuid-a"), (2, "uuid-b")):
        incoming_node = next(
            name for name in model.scene.graph.nodes_geometry
            if model.scene.geometry[model.scene.graph.get(name)[1]].metadata[
                "housemaker_object_id"
            ] == identifier
        )
        expected = _world_geometry(model.scene, incoming_node)
        actual = _world_geometry(exported.scene, f"Object_{number}_part_1")
        np.testing.assert_allclose(actual.vertices, expected.vertices)
        np.testing.assert_allclose(actual.vertex_normals, expected.vertex_normals)
        np.testing.assert_allclose(actual.face_normals, expected.face_normals)
        uv = np.asarray(actual.visual.uv)
        assert np.isfinite(uv).all()
        assert uv[:, 0].min() >= (number - 1) * 0.25
        assert uv[:, 0].max() <= number * 0.25
        assert uv[:, 1].min() >= 0.75
        assert uv[:, 1].max() <= 1.0
        material = actual.visual.material
        assert material.normalTexture.getpixel((0, 0)) == (128, 128, 255, 255)
        assert material.metallicRoughnessTexture.getpixel((0, 0)) == (
            255, 170, 20, 255,
        )


def test_unbound_and_glass_parts_still_belong_to_numbered_roots(materialized_atlas):
    atlas = replace(
        materialized_atlas.atlas,
        placements=materialized_atlas.atlas.placements[:1],
    )
    materialized_atlas = replace(materialized_atlas, atlas=atlas)
    model = compose_placed_generated_models(
        _empty_model(), (_placement("uuid-a", 1, glass=True), _placement("uuid-b", 2)),
    )
    exported = _export(model, materialized_atlas)
    parents = exported.scene.graph.transforms.parents
    assert sorted(parents[name] for name in exported.scene.graph.nodes_geometry) == [
        "Object_1", "Object_1", "Object_2",
    ]


def test_shared_texture_source_does_not_replace_per_object_uuid(materialized_atlas):
    model = compose_placed_generated_models(
        _empty_model(),
        (
            _placement("uuid-a", 1),
            replace(_placement("uuid-b", 2), atlas_source_object_id="uuid-a"),
        ),
    )
    exported = _export(model, materialized_atlas)
    nodes = {node["name"]: node for node in _document(exported.glb_bytes)["nodes"]}
    assert nodes["Object_1"]["extras"]["housemaker_object_id"] == "uuid-a"
    assert nodes["Object_2"]["extras"]["housemaker_object_id"] == "uuid-b"
    for name in ("Object_1_part_1", "Object_2_part_1"):
        uv = np.asarray(_world_geometry(exported.scene, name).visual.uv)
        assert 0.0 <= uv[:, 0].min() <= uv[:, 0].max() <= 0.25


def test_legacy_unnumbered_objects_keep_single_batch(materialized_atlas):
    scene = trimesh.Scene()
    for identifier in ("uuid-a", "uuid-b"):
        mesh = _triangle()
        mesh.metadata["housemaker_object_id"] = identifier
        scene.add_geometry(mesh, node_name=identifier)
    exported = _export(
        GeneratedModel(mesh=_triangle(), scene=scene, glb_bytes=b""),
        materialized_atlas,
    )
    assert tuple(exported.scene.graph.nodes_geometry) == ("housemaker_atlas_objects",)


# ### Half meshes, shared prototypes, and PVS compatibility ###
def test_numbered_half_keeps_exact_name_and_mirror_metadata(materialized_atlas):
    model = compose_placed_generated_models(
        _empty_model(), (_placement("uuid-a", 1, half=True),),
    )
    exported = _export(model, materialized_atlas)
    nodes = _document(exported.glb_bytes)["nodes"]
    half_node = next(node for node in nodes if node.get("name") == "Object_1")
    assert half_node["extras"]["housemaker_export_object_id"] == "Object 1"
    assert half_node["extras"]["housemaker_object_id"] == "uuid-a"
    assert "halfMesh" in half_node["extras"]
    assert not any(node.get("name", "").startswith("[HALF]") for node in nodes)


@pytest.mark.parametrize("half", (False, True))
def test_numbered_instance_prototype_keeps_source_identity(materialized_atlas, half):
    placement = _placement("uuid-a", 1, half=half)
    model = compose_generated_model_instance_sources(_empty_model(), (placement,))
    exported = _export(model, materialized_atlas)
    nodes = _document(exported.glb_bytes)["nodes"]
    root = next(node for node in nodes if node.get("name") == "Object_1")
    assert root["extras"]["housemaker_export_object_id"] == "Object 1"
    assert root["extras"]["housemaker_object_name"] == "My cupboard"
    assert root["extras"]["housemaker_object_id"] == "uuid-a"
    assert root["extras"]["housemaker_instance_source_id"] == "uuid-a"
    assert root["extras"]["housemaker_instance_source_name"] == "Object_1"
    assert all(
        "halfMesh" in nodes[index]["extras"] for index in root["children"]
    ) is half


def test_pvs_keeps_numbered_objects_as_individual_complete_subtrees(materialized_atlas):
    model = compose_placed_generated_models(
        _empty_model(), (_placement("uuid-a", 1), _placement("uuid-b", 2)),
    )
    exported = _export(model, materialized_atlas)
    payload, clusters = partition_glb_for_pvs(exported.glb_bytes)
    assert collect_pvs_occluder_triangles(payload).shape == (0, 3, 3)
    nodes = {node["name"]: node for node in _document(payload)["nodes"]}
    assert {"Object_1", "Object_2"} <= nodes.keys()
    for name in ("Object_1", "Object_2"):
        matches = [cluster for cluster in clusters if name in cluster.node_names]
        assert len(matches) == 1
        assert f"{name}_part_1" in matches[0].node_names
