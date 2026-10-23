# ### Imports ###
from __future__ import annotations

import json
import struct
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
from housemaker.glb import (
    GeneratedModel,
    PlacedGeneratedModel,
    compose_generated_model_instance_sources,
)
from housemaker.pbr_maps import ATLAS_MAP_TYPES
from housemaker.texture_atlas_state import (
    ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR,
    ATLAS_SLOT_HALF_LEFT,
    ATLAS_TYPE_OBJECT,
    ATLAS_TYPE_SURFACE,
    TextureAtlasPlacement,
    TextureAtlasRecord,
)


# ### Fixture helpers ###
def _triangle(metadata, *, half=False):
    mesh = trimesh.Trimesh(
        vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        faces=((0, 1, 2),),
        metadata=metadata,
        process=False,
    )
    mesh.visual = TextureVisuals(
        uv=np.asarray(((0.0, 0.0), (0.5 if half else 1.0, 0.0), (0.0, 1.0))),
        material=PBRMaterial(name="Original"),
    )
    return mesh


def _placement(source_id, x, *, half=False):
    return TextureAtlasPlacement(
        object_id=source_id,
        texture_path=f"{source_id}.png",
        texture_resolution=512,
        x=x,
        y=0,
        size=512,
        **(
            {
                "packing_mode": ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR,
                "slot_half": ATLAS_SLOT_HALF_LEFT,
            }
            if half
            else {}
        ),
    )


def _materialized_atlas(tmp_path, atlas):
    paths = {}
    for map_type in ATLAS_MAP_TYPES:
        path = tmp_path / f"{atlas.atlas_id}.{map_type}.png"
        Image.new(
            "RGBA", (atlas.resolution, atlas.resolution), (120, 130, 140, 255),
        ).save(path)
        paths[map_type] = path
    return MaterializedTextureAtlas(atlas, paths)


def _glb_document(payload):
    json_length, chunk_type = struct.unpack_from("<II", payload, 12)
    assert chunk_type == 0x4E4F534A
    return json.loads(payload[20 : 20 + json_length].decode("utf-8"))


# ### Typed Atlas export regression tests ###
def test_typed_atlases_keep_objects_and_walls_separate_without_changing_mirroring(tmp_path):
    scene = trimesh.Scene()
    scene.add_geometry(_triangle({"housemaker_object_id": "chair"}), node_name="chair")
    scene.add_geometry(_triangle({"housemaker_object_id": "table"}), node_name="table")
    scene.add_geometry(_triangle({"housemaker_surface_id": "wall-a"}), node_name="wall-a")
    scene.add_geometry(_triangle({"housemaker_surface_id": "wall-b"}), node_name="wall-b")
    half_mesh = {
        "mirrorPlane": {"point": [0.0, 0.0, 0.0], "normal": [1.0, 0.0, 0.0]},
        "uvMode": "reuse",
    }
    scene.graph.update(
        frame_from=scene.graph.base_frame,
        frame_to="half-cabinet-root",
        metadata={"halfMesh": half_mesh},
    )
    scene.add_geometry(
        _triangle({"housemaker_object_id": "cabinet"}, half=True),
        node_name="half-cabinet",
        parent_node_name="half-cabinet-root",
    )
    model = GeneratedModel(
        mesh=trimesh.util.concatenate(tuple(scene.geometry.values())),
        scene=scene,
        glb_bytes=b"",
    )
    objects = TextureAtlasRecord(
        atlas_id="objects",
        name="Furniture",
        resolution=2048,
        atlas_type=ATLAS_TYPE_OBJECT,
        placements=[
            _placement("chair", 0),
            _placement("table", 512),
            _placement("cabinet", 1024, half=True),
        ],
    )
    surfaces = TextureAtlasRecord(
        atlas_id="surfaces",
        name="Walls",
        resolution=2048,
        atlas_type=ATLAS_TYPE_SURFACE,
        placements=[_placement("plaster", 0)],
    )
    surface_sources = {"wall-a": "plaster", "wall-b": "plaster"}
    estimate = estimate_texture_atlas_draw_calls(
        model, (objects, surfaces), surface_source_ids=surface_sources,
    )
    with patch(
        "housemaker.atlas_export.bake_placed_object_ambient_occlusion", return_value={},
    ):
        exported = apply_texture_atlases_to_export(
            model,
            tuple(_materialized_atlas(tmp_path, atlas) for atlas in (objects, surfaces)),
            surface_source_ids=surface_sources,
        )

    document = _glb_document(exported.glb_bytes)
    nodes = {node["name"]: node for node in document["nodes"] if "mesh" in node}
    assert len(nodes) == estimate.exported_count == 3
    assert estimate.atlas_batch_count == 2
    assert estimate.half_primitive_count == 1
    assert estimate.mirrored_runtime_count == 4
    assert {material["name"] for material in document["materials"]} == {
        "[OBJECT] Furniture",
        "[SURFACE] Walls",
    }
    for name, atlas_type in (
        ("housemaker_atlas_objects", ATLAS_TYPE_OBJECT),
        ("housemaker_atlas_surfaces", ATLAS_TYPE_SURFACE),
        ("[HALF] half-cabinet-root", ATLAS_TYPE_OBJECT),
    ):
        node = nodes[name]
        mesh = document["meshes"][node["mesh"]]
        assert node["extras"]["housemaker_atlas_type"] == atlas_type
        assert mesh["extras"]["housemaker_atlas_type"] == atlas_type
        assert len(mesh["primitives"]) == 1
    half_node = nodes["[HALF] half-cabinet-root"]
    half_geometry = document["meshes"][half_node["mesh"]]
    assert half_node["extras"]["halfMesh"] == half_mesh
    assert half_geometry["extras"]["halfMesh"] == half_mesh
    assert all(
        "halfMesh" not in node.get("extras", {})
        for name, node in nodes.items()
        if name != "[HALF] half-cabinet-root"
    )


@pytest.mark.parametrize("half", (False, True))
def test_typed_instance_source_preserves_root_child_and_mirror_metadata(tmp_path, half):
    source_mesh = _triangle({}, half=half)
    source_model = GeneratedModel(
        mesh=source_mesh,
        scene=trimesh.Scene(source_mesh),
        glb_bytes=b"",
    )
    source = PlacedGeneratedModel(
        object_id="cabinet-copy",
        source_object_id="cabinet",
        object_name="cabinet_prototype",
        model=source_model,
        world_position=(2.0, 3.0, 0.0),
        symmetric_preview_orientation="vertical" if half else None,
        symmetric_preview_plane_coordinate=0.0 if half else None,
    )
    base = GeneratedModel(
        mesh=trimesh.Trimesh(process=False),
        scene=trimesh.Scene(),
        glb_bytes=b"",
    )
    model = compose_generated_model_instance_sources(base, (source,))
    atlas = TextureAtlasRecord(
        atlas_id="objects",
        name="Furniture",
        resolution=2048,
        atlas_type=ATLAS_TYPE_OBJECT,
        placements=[_placement("cabinet", 0, half=half)],
    )
    exported = apply_texture_atlases_to_export(
        model, (_materialized_atlas(tmp_path, atlas),),
    )

    document = _glb_document(exported.glb_bytes)
    root = next(
        node for node in document["nodes"]
        if node.get("name") == "cabinet_prototype"
    )
    assert root["extras"]["housemaker_instance_source_id"] == "cabinet"
    assert root["extras"]["housemaker_atlas_type"] == ATLAS_TYPE_OBJECT
    assert len(root["children"]) == 1
    child = document["nodes"][root["children"][0]]
    mesh = document["meshes"][child["mesh"]]
    assert child["extras"]["housemaker_atlas_type"] == ATLAS_TYPE_OBJECT
    assert mesh["extras"]["housemaker_atlas_type"] == ATLAS_TYPE_OBJECT
    assert ("halfMesh" in child.get("extras", {})) is half
    assert ("halfMesh" in mesh.get("extras", {})) is half
    assert {material["name"] for material in document["materials"]} == {
        "[OBJECT] Furniture",
    }
