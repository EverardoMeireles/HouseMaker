"""Exact export geometry and final physical PVS data agree end to end."""

# ### Imports ###
from __future__ import annotations

import json
import struct
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from housemaker.glb import (
    INSTANCE_SOURCE_ID_METADATA_KEY,
    GeneratedModel,
    PlacedGeneratedModel,
)
from housemaker.global_scale import GlobalScale
from housemaker.pvs_bake import PvsAabb, build_occluder_patches, occluder_hides_aabb
from housemaker.pvs_state import PvsBakeData
from housemaker.pvs_workspace import (
    PvsSceneSnapshot,
    _build_snapshot_model,
    _instance_clusters,
    bake_scene_visibility,
    prepare_pvs_geometry,
    pvs_fingerprint,
)
from housemaker.surface_geometry import build_fixed_surfaces
from tests.test_pvs_geometry import _encode, _fixture
from tests.test_surface_material_integration import _build_one_room_level


# ### Scene fixtures ###
def _snapshot(scale=None):
    level, _room = _build_one_room_level()
    return PvsSceneSnapshot(
        (level,), (), (), (), (), (), (), scale or GlobalScale(), 1.7, "a" * 64,
    )


def _glb_document(payload):
    length, kind = struct.unpack_from("<I4s", payload, 12)
    assert kind == b"JSON"
    return json.loads(payload[20:20 + length])


# ### Full bake/export consistency ###
def test_tiny_house_bakes_without_explicit_camera_cells():
    snapshot = _snapshot()
    progress = []
    data = bake_scene_visibility(snapshot, progress=progress.append)
    assert isinstance(data, PvsBakeData)
    assert data.cells
    assert data.clusters
    assert data.authoring_signature == snapshot.authoring_signature
    assert data.eye_height_meters == 1.7
    assert data.coordinate_system == "gltf-y-up"
    assert progress[-1].endswith("(99%)")
    assert PvsBakeData.from_dict(data.to_dict()) == data

    _base, model = _build_snapshot_model(snapshot)
    payload, clusters, occluders, signature = prepare_pvs_geometry(
        model, (), (), snapshot.global_scale, snapshot.eye_height_meters,
    )
    assert clusters == data.clusters
    assert signature == data.source_signature
    assert len(occluders)
    nodes = {node.get("name") for node in _glb_document(payload)["nodes"]}
    assert all(set(cluster.node_names) <= nodes for cluster in data.clusters)


def test_cells_use_physical_eye_height_and_final_y_up_coordinates():
    scale = GlobalScale(1.8, 1.9, 2.4, linked=False)
    snapshot = _snapshot(scale)
    data = bake_scene_visibility(snapshot)
    floor = next(surface for surface in build_fixed_surfaces(snapshot.levels)
                 if surface.surface_type == "floor")
    physical_eye_y = float(floor.mesh.vertices[:, 2].max()) * scale.z + 1.7
    assert any(cell.minimum[1] <= physical_eye_y <= cell.maximum[1]
               for cell in data.cells)
    assert all(np.isfinite((*cell.minimum, *cell.maximum)).all() for cell in data.cells)


def test_exact_geometry_and_eye_height_changes_invalidate_bake_signature():
    snapshot = _snapshot()
    _base, model = _build_snapshot_model(snapshot)
    signature = prepare_pvs_geometry(model, (), (), snapshot.global_scale, 1.7)[-1]
    assert prepare_pvs_geometry(model, (), (), snapshot.global_scale, 1.8)[-1] != signature
    assert prepare_pvs_geometry(model, (), (), GlobalScale(2, 2, 2), 1.7)[-1] != signature
    graph = snapshot.levels[0].vertex_data
    first = graph.vertices[0]
    graph.move_vertex(first.id, first.x + 10, first.y)
    _base, edited_model = _build_snapshot_model(snapshot)
    assert prepare_pvs_geometry(edited_model, (), (), snapshot.global_scale, 1.7)[-1] != signature


def test_snapshot_bake_can_be_cancelled_before_any_result_is_published():
    from housemaker.pvs_bake import PvsBakeCancelled

    with pytest.raises(PvsBakeCancelled):
        bake_scene_visibility(_snapshot(), is_cancelled=lambda: True)


def test_stable_authoring_fingerprint_is_independent_of_set_iteration_order():
    assert pvs_fingerprint({"a": {1, 2, 3}, "b": frozenset(("x", "y"))}) == pvs_fingerprint(
        {"b": frozenset(("y", "x")), "a": {3, 1, 2}},
    )


def test_instance_only_membership_is_kept_separate_from_source_nodes():
    assert _instance_clusters((), (), GlobalScale()) == ()


def test_scene_without_included_geometry_fails_instead_of_installing_empty_pvs():
    snapshot = _snapshot()
    snapshot.levels[0].include_in_export = False
    with pytest.raises(ValueError):
        bake_scene_visibility(replace(snapshot, authoring_signature="b" * 64))


# ### Runtime instance membership and conservative exclusions ###
def _instance_model():
    mesh = trimesh.creation.box(extents=(2, 4, 6))
    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name="chair-mesh", geom_name="chair-geometry")
    return GeneratedModel(mesh=mesh, scene=scene, glb_bytes=scene.export(file_type="glb"))


def _instance_pair(model=None):
    source_model = _instance_model() if model is None else model
    source = PlacedGeneratedModel(
        "chair-source", source_model, (1, 2, 0.5), object_name="Source chair"
    )
    instance = PlacedGeneratedModel(
        "chair-copy", source_model, (10, -2, 1), source_object_id=source.object_id,
        object_name="Source chair", rotation_degrees=(0, 0, 90), scale=0.5,
        axis_scales=(2, 0.5, 1),
    )
    return source, instance


def _edit_instance_glb(model, edit):
    payload = model.glb_bytes
    length = struct.unpack_from("<I", payload, 12)[0]
    document = _glb_document(payload)
    edit(document)
    encoded = json.dumps(document, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    remaining_chunks = payload[20 + length:]
    body = struct.pack("<I4s", len(encoded), b"JSON") + encoded + remaining_chunks
    return replace(model, glb_bytes=struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)


def test_source_and_runtime_copy_have_independent_physical_instance_bounds():
    source, instance = _instance_pair()
    scale = GlobalScale(2, 3, 4, linked=False)

    clusters = _instance_clusters((source,), (instance,), scale)

    assert len(clusters) == 2
    by_instance = {cluster.instance_ids[0]: cluster for cluster in clusters}
    assert set(by_instance) == {source.object_id, instance.object_id}
    assert len({cluster.cluster_id for cluster in clusters}) == 2
    assert all(cluster.node_names == () for cluster in clusters)
    # Source bottom-center pivot, object transforms, Z-up -> glTF Y-up, then
    # physical X/Y/Z multipliers. The rotated copy is NOT its prototype bound.
    np.testing.assert_allclose(by_instance[source.object_id].minimum, (0, 2, -12))
    np.testing.assert_allclose(by_instance[source.object_id].maximum, (4, 26, 0))
    np.testing.assert_allclose(by_instance[instance.object_id].minimum, (19, 4, 3))
    np.testing.assert_allclose(by_instance[instance.object_id].maximum, (21, 16, 9))


def test_full_scene_bake_keeps_instance_membership_out_of_shared_prototype_nodes():
    source, instance = _instance_pair()
    snapshot = replace(_snapshot(), authored_models=(source,), instance_models=(instance,))

    data = bake_scene_visibility(snapshot)
    instance_clusters = [cluster for cluster in data.clusters if cluster.instance_ids]
    assert {cluster.instance_ids for cluster in instance_clusters} == {
        (source.object_id,), (instance.object_id,),
    }
    assert all(not cluster.node_names for cluster in instance_clusters)

    _base, model = _build_snapshot_model(snapshot)
    payload, clusters, _occluders, signature = prepare_pvs_geometry(
        model, (source,), (instance,), snapshot.global_scale, 1.7,
    )
    assert clusters == data.clusters
    assert signature == data.source_signature
    document = _glb_document(payload)
    pending = [index for index, node in enumerate(document["nodes"])
               if node.get("extras", {}).get(INSTANCE_SOURCE_ID_METADATA_KEY) == source.object_id]
    prototype_nodes = set()
    while pending:
        index = pending.pop()
        node = document["nodes"][index]
        prototype_nodes.add(node.get("name"))
        pending.extend(node.get("children", ()))
    assert prototype_nodes
    assert all(not prototype_nodes.intersection(cluster.node_names) for cluster in clusters)


def test_symmetric_source_and_copies_are_unmapped_and_therefore_always_visible():
    source, instance = _instance_pair()
    source = replace(source, symmetric_preview_orientation="vertical",
                     symmetric_preview_plane_coordinate=0)
    instance = replace(instance, symmetric_preview_orientation="vertical",
                       symmetric_preview_plane_coordinate=0)

    assert _instance_clusters((source,), (instance,), GlobalScale()) == ()


@pytest.mark.parametrize("flag", (
    "animation", "skin", "dynamic_node", "half_node", "door_mesh", "morph_target",
))
def test_sources_advertising_animation_or_runtime_reconstruction_are_not_culled(flag):
    def mark(document):
        mesh_node = next(node for node in document["nodes"] if "mesh" in node)
        if flag == "animation":
            document["animations"] = [{"channels": [{"target": {"node": 0, "path": "translation"}}]}]
        elif flag == "skin":
            document["skins"] = [{"joints": [0]}]
        elif flag == "dynamic_node":
            mesh_node["extras"] = {"housemaker_dynamic": True}
        elif flag == "half_node":
            mesh_node["extras"] = {"halfMesh": {"keptSide": "left"}}
        elif flag == "door_mesh":
            document["meshes"][0]["extras"] = {"housemakerDoorBody": {"id": "door"}}
        elif flag == "morph_target":
            primitive = document["meshes"][0]["primitives"][0]
            primitive["targets"] = [{"POSITION": primitive["attributes"]["POSITION"]}]

    source, instance = _instance_pair(_edit_instance_glb(_instance_model(), mark))

    assert _instance_clusters((source,), (instance,), GlobalScale()) == ()


@pytest.mark.parametrize("prefix", ("door-placement:", "window-placement:"))
def test_opening_placements_and_their_identified_copies_fail_open(prefix):
    source, instance = _instance_pair()
    source = replace(source, object_id=prefix + "source", source_object_id=prefix + "source")
    instance = replace(instance, object_id=prefix + "copy", source_object_id=source.object_id)

    assert _instance_clusters((source,), (instance,), GlobalScale()) == ()


@pytest.mark.parametrize("prefix", ("door-placement:", "window-placement:"))
def test_opening_source_identity_also_excludes_an_ordinary_named_runtime_copy(prefix):
    source, instance = _instance_pair()
    source = replace(source, object_id=prefix + "source", source_object_id=prefix + "source")
    instance = replace(instance, source_object_id=source.object_id)
    assert not instance.object_id.startswith(prefix)

    assert _instance_clusters((source,), (instance,), GlobalScale()) == ()


@pytest.mark.parametrize("record_kind", ("node", "mesh"))
def test_gpu_instancing_extension_cannot_use_one_prototype_bound_for_all_instances(record_kind):
    def mark(document):
        primitive = document["meshes"][0]["primitives"][0]
        record = (next(node for node in document["nodes"] if "mesh" in node)
                  if record_kind == "node" else document["meshes"][0])
        record["extensions"] = {"EXT_mesh_gpu_instancing": {
            "attributes": {"TRANSLATION": primitive["attributes"]["POSITION"]},
        }}
        document["extensionsUsed"] = ["EXT_mesh_gpu_instancing"]

    source, instance = _instance_pair(_edit_instance_glb(_instance_model(), mark))

    assert _instance_clusters((source,), (instance,), GlobalScale()) == ()


# ### Rendered face winding remains part of the exact bake contract ###
def test_single_sided_backfaces_do_not_hide_geometry_and_flip_invalidates_signature():
    points = np.asarray(((0, -5, -5), (0, 5, -5), (0, 5, 5),
                         (0, -5, -5), (0, 5, 5), (0, -5, 5)), dtype=float)

    def prepare(vertices, double_sided=False):
        document, binary = _fixture(vertices=vertices)
        document["materials"][0]["doubleSided"] = double_sided
        return prepare_pvs_geometry(
            SimpleNamespace(glb_bytes=_encode(document, binary)), (), (), GlobalScale(), 1.7,
        )

    _payload, _clusters, triangles, signature = prepare(points)
    patches = build_occluder_patches((triangles,), one_sided=True)
    front = PvsAabb((1, -0.1, -0.1), (1.2, 0.1, 0.1))
    back = PvsAabb((-1.2, -0.1, -0.1), (-1, 0.1, 0.1))
    assert any(occluder_hides_aabb(front, back, patch) for patch in patches)
    assert not any(occluder_hides_aabb(back, front, patch) for patch in patches)

    flipped = points.reshape(-1, 3, 3)[:, ::-1].reshape(-1, 3)
    assert prepare(flipped)[-1] != signature
    cyclic = points.reshape(-1, 3, 3)[:, (1, 2, 0)].reshape(-1, 3)
    assert prepare(cyclic)[-1] == signature
    double_triangles = prepare(points, double_sided=True)[2]
    double_patches = build_occluder_patches((double_triangles,), one_sided=True)
    assert any(occluder_hides_aabb(front, back, patch) for patch in double_patches)
    assert any(occluder_hides_aabb(back, front, patch) for patch in double_patches)
