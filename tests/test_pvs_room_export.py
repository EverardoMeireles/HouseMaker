"""Room batching and Atlas preparation are identical for PVS and real export."""

# ### Imports ###
from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import trimesh
from trimesh.visual.texture import TextureVisuals

from housemaker.atlas_export import apply_texture_atlases_to_export
from housemaker.glb import (
    GeneratedModel,
    PlacedGeneratedModel,
    compose_generated_model_instance_sources,
    compose_placed_generated_models,
    convert_to_glb,
)
from housemaker.global_scale import GlobalScale
from housemaker.pvs_geometry import (
    collect_pvs_occluder_triangles,
    partition_glb_for_pvs,
)
from housemaker.pvs_workspace import (
    PvsSceneSnapshot,
    PvsWorkspaceMixin,
    _build_snapshot_model,
    _export_placement_groups,
    prepare_pvs_geometry,
)
from housemaker.room_atlas_export import prepare_room_export_atlases
from housemaker.room_zones import build_room_zones
from tests.test_pvs_geometry import _decode, _encode, _fixture
from tests.test_room_atlas_export import _atlas, _placement
from tests.test_room_zones import _level


# ### Shared room/placement snapshots and independent export construction ###
def _snapshot(tmp_path, *, with_atlas):
    level = _level(divided=True)
    layout = build_room_zones((level,))
    centers = [room.polygon.centroid for room in layout.rooms]
    mesh = trimesh.creation.box((0.5, 0.5, 1.0))
    mesh.visual = TextureVisuals(uv=np.asarray(mesh.vertices)[:, :2] + 0.5)
    scene = trimesh.Scene(mesh)
    model = GeneratedModel(mesh=mesh, scene=scene, glb_bytes=scene.export(file_type="glb"))
    source = PlacedGeneratedModel(
        "chair-source", model, (centers[0].x, centers[0].y, layout.rooms[0].floor_z_meters),
    )
    instance = PlacedGeneratedModel(
        "chair-copy", model, (centers[1].x, centers[1].y, layout.rooms[1].floor_z_meters),
        source_object_id=source.object_id,
    )
    atlases = (
        (_atlas(tmp_path, (_placement(source.object_id),), resolution=512),)
        if with_atlas else ()
    )
    return PvsSceneSnapshot(
        (level,), (), (), (source,), (instance,), atlases, (),
        GlobalScale(1.8, 1.9, 1.2, linked=False), 1.7, "a" * 64,
        output_directory=tmp_path / "bake",
        placement_level_indices=((source.object_id, level.index), (instance.object_id, level.index)),
    )


def _normal_export(snapshot, directory):
    base = convert_to_glb(
        snapshot.levels, stairs=snapshot.stairs,
        surface_materials=dict(snapshot.surface_materials), export_untextured_surfaces=True,
        make_floors_double_sided=snapshot.make_floors_double_sided,
        make_ceilings_double_sided=snapshot.make_ceilings_double_sided,
    )
    placed, sources = _export_placement_groups(snapshot.authored_models, snapshot.instance_models)
    model = compose_placed_generated_models(base, placed) if placed else base
    if sources:
        model = compose_generated_model_instance_sources(model, sources)
    layout = build_room_zones(snapshot.levels, snapshot.stairs)
    plan = prepare_room_export_atlases(
        snapshot.materialized_atlases, layout,
        (*snapshot.authored_models, *snapshot.instance_models),
        output_directory=directory, placement_level_indices=dict(snapshot.placement_level_indices),
    )
    return apply_texture_atlases_to_export(
        model, plan.atlases, surface_source_ids=dict(snapshot.surface_source_ids), room_layout=layout,
    ), plan


def _visibility_dependencies(model, snapshot):
    return prepare_pvs_geometry(
        model, snapshot.authored_models, snapshot.instance_models,
        snapshot.global_scale, snapshot.eye_height_meters,
    )


# ### Snapshot and export geometry remain interchangeable ###
@pytest.mark.parametrize("with_atlas", (False, True))
def test_snapshot_matches_room_export_clusters_and_signature(tmp_path, with_atlas):
    snapshot = _snapshot(tmp_path, with_atlas=with_atlas)
    original_atlases = tuple(item.atlas.to_dict() for item in snapshot.materialized_atlases)

    _base, baked_model = _build_snapshot_model(snapshot)
    exported_model, plan = _normal_export(snapshot, tmp_path / "export")
    bake_bytes, bake_clusters, bake_occluders, bake_signature = _visibility_dependencies(baked_model, snapshot)
    export_bytes, export_clusters, export_occluders, export_signature = _visibility_dependencies(exported_model, snapshot)

    assert bake_clusters == export_clusters
    assert bake_signature == export_signature
    np.testing.assert_array_equal(bake_occluders, export_occluders)
    assert tuple(item.atlas.to_dict() for item in snapshot.materialized_atlases) == original_atlases
    # Shared prototypes are classified from the real placements in BOTH rooms,
    # rather than their canonical library origin or a representative alone.
    assert len(plan.source_room_ids["chair-source"]) == 2
    if with_atlas:
        assert plan.atlases[0].atlas.room_id is None
        assert plan.atlases[0].atlas.room_name == "Shared objects"

    document, _binary, _chunks = _decode(bake_bytes)
    room_nodes = [node for node in document["nodes"] if node.get("extras", {}).get("housemaker_room_ids")]
    assert room_nodes
    assert all("mesh" in node for node in room_nodes)
    assert all(not node.get("children") for node in room_nodes)
    room_names = {node["name"] for node in room_nodes}
    assert all(cluster.cell is None for cluster in partition_glb_for_pvs(baked_model.glb_bytes)[1]
               if room_names.intersection(cluster.node_names))
    assert _decode(export_bytes)[0]["nodes"] == document["nodes"]


def test_legacy_snapshot_without_asset_directory_keeps_only_embedded_output(tmp_path):
    snapshot = replace(_snapshot(tmp_path, with_atlas=True), output_directory=None)
    _base, model = _build_snapshot_model(snapshot)

    document, _binary, _chunks = _decode(model.glb_bytes)
    assert document.get("images")
    assert all("bufferView" in image and "uri" not in image for image in document["images"])
    assert _visibility_dependencies(model, snapshot)[1]


def test_capture_owns_level_map_and_asset_directory_without_publishing(tmp_path):
    level = _level()
    generation = SimpleNamespace(
        get_generated_object_records=lambda: (
            SimpleNamespace(object_id="source", placement=SimpleNamespace(level_index=2)),
            SimpleNamespace(object_id="unplaced", placement=None),
        ),
        get_generated_object_instances=lambda: (
            SimpleNamespace(instance_id="copy", placement=SimpleNamespace(level_index=3)),
        ),
    )
    atlas_workspace = SimpleNamespace(
        _asset_directory=tmp_path, prepare_export_atlases=Mock(return_value=()), set_data=Mock(),
    )
    workspace = SimpleNamespace(
        levels=[level], stairs=[], generation=generation, texture_atlas_workspace=atlas_workspace,
        surface_texture_generation=SimpleNamespace(get_surface_material_sources=dict),
        global_scale=GlobalScale(),
        _generation_settings=SimpleNamespace(
            first_person_eye_height_meters=1.7, make_floors_double_sided=False,
            make_ceilings_double_sided=False,
        ),
        _build_active_architectural_trim_component_models=lambda: ((), ()),
        _build_export_placed_models=lambda: ((), ()),
        _build_atlas_surface_source_ids=dict,
        _build_pvs_authoring_signature=lambda: "a" * 64,
    )

    snapshot = PvsWorkspaceMixin._capture_pvs_snapshot(workspace)
    _build_snapshot_model(snapshot)

    assert snapshot.output_directory == tmp_path
    assert snapshot.placement_level_indices == (("copy", 3), ("source", 2))
    assert snapshot.levels[0] is not level
    atlas_workspace.set_data.assert_not_called()


# ### Room batch ownership preserves source nodes and draw call reductions ###
@pytest.mark.parametrize("tag_on_mesh", (False, True))
@pytest.mark.parametrize("room_ids", (("room-a",), ("room-a", "room-b")))
def test_room_surface_batches_are_whole_byte_exact_clusters(tag_on_mesh, room_ids):
    document, binary = _fixture()
    metadata = document["meshes"][0]["extras"] if tag_on_mesh else document["nodes"][0]["extras"]
    metadata["housemaker_room_ids"] = list(room_ids)
    source = _encode(document, binary)
    original = copy.deepcopy(document)

    output, clusters = partition_glb_for_pvs(source)

    assert output == source
    assert document == original
    assert len(clusters) == 1
    assert clusters[0].cell is None
    assert clusters[0].node_names == ("atlas",)
    assert clusters[0].minimum == (0.0, 0.0, 0.0)
    assert clusters[0].maximum == (9.0, 1.0, 0.0)
    assert partition_glb_for_pvs(output) == (output, clusters)
    assert len(collect_pvs_occluder_triangles(output)) == 2


def test_room_metadata_does_not_turn_object_atlases_into_whole_house_clusters():
    document, binary = _fixture()
    document["nodes"][0]["extras"].update({
        "housemaker_atlas_type": "object", "housemaker_room_ids": ["room-a"],
    })

    _output, clusters = partition_glb_for_pvs(_encode(document, binary))

    assert len(clusters) == 2
    assert all(cluster.cell is not None for cluster in clusters)
    assert not len(collect_pvs_occluder_triangles(_encode(document, binary)))


def test_room_subtree_owns_transform_and_mesh_names_and_fails_open_for_ambiguous_names():
    document, binary = _fixture()
    document["nodes"] = [
        {"name": "room-group", "children": [1], "extras": {"housemaker_room_id": "room-a"}},
        {"name": "room-mesh", "mesh": 0},
    ]
    source = _encode(document, binary)
    output, clusters = partition_glb_for_pvs(source)
    assert output == source
    assert len(clusters) == 1
    assert clusters[0].node_names == ("room-group", "room-mesh")

    document["nodes"][1]["name"] = "room-group"
    source = _encode(document, binary)
    assert partition_glb_for_pvs(source) == (source, ())
