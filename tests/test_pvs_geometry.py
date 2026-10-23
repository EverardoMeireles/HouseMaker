"""PVS clusters preserve original GLB streams, shading and reconstruction IDs."""

# ### Imports ###
from __future__ import annotations

import copy
import json
import struct

import numpy as np
import pytest

from housemaker.pvs_geometry import (
    PVS_CLUSTER_EXTRAS_KEY,
    collect_pvs_occluder_triangles,
    partition_glb_for_pvs,
)
from tests.test_pvs_runtime import _run_javascript

# ### Minimal image-free GLB fixtures ###
_JSON = 0x4E4F534A
_BIN = 0x004E4942
_POINTS = np.asarray((
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
    (8.0, 0.0, 0.0), (9.0, 0.0, 0.0), (8.0, 1.0, 0.0),
), dtype="<f4")


def _encode(document, binary, extra_chunks=()):
    raw = json.dumps(document, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 4)
    binary += b"\0" * (-len(binary) % 4)
    chunks = ((_JSON, raw), (_BIN, binary), *extra_chunks)
    body = b"".join(struct.pack("<II", len(value), kind) + value for kind, value in chunks)
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


def _decode(payload):
    offset = 12
    chunks = []
    while offset < len(payload):
        length, kind = struct.unpack_from("<II", payload, offset)
        offset += 8
        chunks.append((kind, payload[offset:offset + length]))
        offset += length
    return json.loads(chunks[0][1]), next(value for kind, value in chunks if kind == _BIN), chunks


def _fixture(*, vertices=None, indexed=True):
    points = _POINTS if vertices is None else np.asarray(vertices, dtype="<f4")
    binary = bytearray()
    views, accessors = [], []

    def append(values, kind, width):
        values = np.ascontiguousarray(values)
        binary.extend(b"\0" * (-len(binary) % 4))
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": values.nbytes})
        binary.extend(values.tobytes())
        accessors.append({"bufferView": len(views)-1, "componentType": kind,
                          "count": len(values), "type": {1: "SCALAR", 2: "VEC2", 3: "VEC3"}[width]})
        return len(accessors)-1

    attributes = {
        "POSITION": append(points, 5126, 3),
        "NORMAL": append(np.tile(np.asarray((0, 0, 1), "<f4"), (len(points), 1)), 5126, 3),
        "TEXCOORD_0": append(np.arange(len(points)*2, dtype="<f4").reshape(-1, 2) / 17, 5126, 2),
        "TEXCOORD_1": append(np.arange(len(points)*2, dtype="<f4").reshape(-1, 2) / 37, 5126, 2),
        "_CUSTOM": append(np.arange(len(points), dtype="<u2"), 5123, 1),
    }
    primitive = {"attributes": attributes, "material": 0,
                 "extras": {"opaque-token": [1, 2, 3]},
                 "extensions": {"KHR_materials_variants": {"mappings": []}}}
    if indexed:
        primitive["indices"] = append(np.arange(len(points), dtype="<u2"), 5123, 1)
    document = {
        "asset": {"version": "2.0", "extras": {"preserve": True}},
        "scene": 0, "scenes": [{"nodes": [0], "extras": ["keep-scene"]}],
        "nodes": [{"name": "atlas", "mesh": 0, "extras": {"housemaker_atlas_type": "surface", "nested": {"keep": "me"}}}],
        "meshes": [{"name": "original-mesh", "primitives": [primitive], "extras": {"keep": "mesh-extra"}}],
        "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(binary)}],
        "materials": [{"name": "Atlas material", "occlusionTexture": {"index": 0, "texCoord": 1, "strength": 0.7},
                       "pbrMetallicRoughness": {"metallicRoughnessTexture": {"index": 0}, "baseColorTexture": {"index": 1}},
                       "normalTexture": {"index": 2}, "extras": {"keep": "material"}}],
        "textures": [{"source": 0, "sampler": 0}, {"source": 1}, {"source": 2}],
        "images": [{"uri": "ao.png"}, {"uri": "color.png"}, {"uri": "normal.png"}],
        "samplers": [{"wrapS": 10497, "wrapT": 10497}],
        "extensionsUsed": ["KHR_materials_variants"], "extras": ["top-level-extras"],
    }
    return document, bytes(binary)


def _indices(document, binary, primitive):
    accessor = document["accessors"][primitive["indices"]]
    view = document["bufferViews"][accessor["bufferView"]]
    dtype = {5121: "u1", 5123: "<u2", 5125: "<u4"}[accessor["componentType"]]
    return np.frombuffer(binary, dtype=dtype, count=accessor["count"],
                         offset=view.get("byteOffset", 0) + accessor.get("byteOffset", 0))


# ### Static clusters and untouched shading streams ###
def test_partition_appends_only_indices_and_nodes_without_reserializing_materials_or_vertices():
    document, binary = _fixture()
    original = copy.deepcopy(document)
    payload = _encode(document, binary, ((0x12345678, b"keep"),))

    output, clusters = partition_glb_for_pvs(payload)
    result, result_binary, chunks = _decode(output)

    assert document == original
    assert len(clusters) == 2
    assert result["nodes"][0]["name"] == "atlas"
    assert "mesh" not in result["nodes"][0]
    assert result["nodes"][0]["extras"] == original["nodes"][0]["extras"]
    assert result["meshes"][0] == original["meshes"][0]
    assert result["accessors"][:len(original["accessors"])] == original["accessors"]
    assert result["bufferViews"][:len(original["bufferViews"])] == original["bufferViews"]
    assert result_binary.startswith(_decode(payload)[1])
    for key in ("materials", "textures", "images", "samplers", "extensionsUsed", "asset", "scenes", "extras"):
        assert result[key] == original[key]
    assert chunks[-1] == (0x12345678, b"keep")
    selected_faces = []
    for cluster in clusters:
        assert cluster.node_names == (cluster.cluster_id,)
        assert cluster.cluster_id.startswith("pvs_cluster_")
        node = next(node for node in result["nodes"] if node.get("name") == cluster.node_names[0])
        assert node["extras"][PVS_CLUSTER_EXTRAS_KEY]["id"] == cluster.cluster_id
        mesh = result["meshes"][node["mesh"]]
        assert mesh["extras"] == original["meshes"][0]["extras"]
        primitive = mesh["primitives"][0]
        source = original["meshes"][0]["primitives"][0]
        for key in ("attributes", "material", "extensions", "extras"):
            assert primitive[key] == source[key]
        selected_faces.extend(_indices(result, result_binary, primitive).reshape(-1, 3).tolist())
    assert sorted(selected_faces) == [[0, 1, 2], [3, 4, 5]]


def test_physical_scaling_controls_grid_and_bounds_without_baking_transform_into_glb():
    document, binary = _fixture()
    document["nodes"][0]["translation"] = [1, 2, 3]
    document["nodes"][0]["scale"] = [2, 3, 4]
    world = np.diag((1.5, 2.0, 0.5, 1.0))

    output, clusters = partition_glb_for_pvs(_encode(document, binary), world_transform=world)
    result, result_binary, _ = _decode(output)

    assert result["nodes"][0]["translation"] == [1, 2, 3]
    assert result["nodes"][0]["scale"] == [2, 3, 4]
    assert result_binary.startswith(binary)
    points = (_POINTS * (2, 3, 4) + (1, 2, 3)) * (1.5, 2.0, 0.5)
    np.testing.assert_allclose(np.min([cluster.minimum for cluster in clusters], axis=0), points.min(axis=0))
    np.testing.assert_allclose(np.max([cluster.maximum for cluster in clusters], axis=0), points.max(axis=0))
    assert {cluster.cell[0] for cluster in clusters} == {0, 6}


def test_large_triangle_retains_full_bounds_in_one_centroid_bucket_without_clipping():
    document, binary = _fixture(vertices=((-10, 0, -10), (15, 0, -10), (0, 0, 15)))

    output, clusters = partition_glb_for_pvs(_encode(document, binary))

    assert len(clusters) == 1
    assert clusters[0].minimum == (-10.0, 0.0, -10.0)
    assert clusters[0].maximum == (15.0, 0.0, 15.0)
    assert collect_pvs_occluder_triangles(output).shape == (1, 3, 3)


@pytest.mark.parametrize("indexed", (False, True))
def test_indexed_and_unindexed_faces_remain_exact_and_partitioning_is_idempotent(indexed):
    document, binary = _fixture(indexed=indexed)
    output, clusters = partition_glb_for_pvs(_encode(document, binary))

    second, second_clusters = partition_glb_for_pvs(output)

    assert second == output
    assert second_clusters == clusters
    np.testing.assert_array_equal(collect_pvs_occluder_triangles(output), _POINTS.reshape(-1, 3, 3))
    with pytest.raises(ValueError, match="original GLB"):
        partition_glb_for_pvs(output, cell_size=2)


def test_multiple_primitives_keep_independent_materials_in_each_bucket():
    document, binary = _fixture()
    primitive = copy.deepcopy(document["meshes"][0]["primitives"][0])
    primitive["material"] = 1
    document["meshes"][0]["primitives"].append(primitive)
    document["materials"].append({"name": "Other material"})

    output, clusters = partition_glb_for_pvs(_encode(document, binary))
    result, _, _ = _decode(output)

    assert len(clusters) == 2
    for node in result["nodes"][1:]:
        assert [primitive["material"] for primitive in result["meshes"][node["mesh"]]["primitives"]] == [0, 1]


def test_duplicate_static_node_names_still_produce_unique_stable_cluster_names():
    document, binary = _fixture()
    document["nodes"].append(copy.deepcopy(document["nodes"][0]))
    document["scenes"][0]["nodes"].append(1)

    output, clusters = partition_glb_for_pvs(_encode(document, binary))
    repeated, repeated_clusters = partition_glb_for_pvs(_encode(document, binary))

    assert output == repeated
    assert clusters == repeated_clusters
    assert len({cluster.cluster_id for cluster in clusters}) == 4


def test_parent_groups_children_and_column_major_matrix_are_preserved():
    document, binary = _fixture()
    document["nodes"].insert(0, {"name": "parent", "translation": [0, 3, 5], "children": [1]})
    matrix = np.eye(4)
    matrix[:3, 3] = (2, 0, 1)
    document["nodes"][1]["matrix"] = matrix.reshape(-1, order="F").tolist()

    output, clusters = partition_glb_for_pvs(_encode(document, binary))
    result, _, _ = _decode(output)

    assert result["nodes"][0] == document["nodes"][0]
    assert result["nodes"][1]["matrix"] == document["nodes"][1]["matrix"]
    np.testing.assert_allclose(np.min([cluster.minimum for cluster in clusters], axis=0), (2, 3, 6))


# ### Whole-object bounds and runtime reconstruction preservation ###
def test_full_furniture_subtree_keeps_exact_graph_and_one_whole_object_cluster():
    document, binary = _fixture()
    document["nodes"][0]["extras"] = {"housemaker_object_id": "cupboard"}
    document["nodes"].insert(0, {"name": "Furniture root", "children": [1, 2], "translation": [3, 0, 0],
                                  "extras": {"housemaker_object_id": "cupboard"}})
    document["nodes"].append({"name": "Second part", "mesh": 0, "translation": [10, 0, 0]})
    payload = _encode(document, binary)

    output, clusters = partition_glb_for_pvs(payload)

    assert output == payload
    assert len(clusters) == 1
    assert clusters[0].node_names == ("Furniture root", "atlas", "Second part")
    assert clusters[0].source_node == "Furniture root"
    assert clusters[0].cell is None
    assert clusters[0].minimum == (3.0, 0.0, 0.0)
    assert clusters[0].maximum == (22.0, 1.0, 0.0)
    assert collect_pvs_occluder_triangles(payload).shape == (0, 3, 3)


def test_nested_furniture_cluster_owns_all_transform_and_mesh_nodes_for_runtime_culling(tmp_path):
    document, binary = _fixture()
    document["nodes"].insert(0, {
        "name": "Furniture root", "children": [2],
        "extras": {"housemaker_object_id": "cupboard"},
    })
    document["nodes"].append({"name": "Shelf group", "children": [1]})
    payload = _encode(document, binary)

    output, clusters = partition_glb_for_pvs(payload)

    assert output == payload
    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster.node_names == ("Furniture root", "Shelf group", "atlas")
    descriptor = {
        "id": cluster.cluster_id, "nodeNames": list(cluster.node_names),
        "instanceIds": [], "min": list(cluster.minimum), "max": list(cluster.maximum),
    }
    _run_javascript(tmp_path, r'''
      const leaf = node("atlas");
      const shelf = node("Shelf group", { isMesh: false, children: [leaf] });
      const furniture = node("Furniture root", { isMesh: false, children: [shelf] });
      const visibility = manifest();
      visibility.clusters = [''' + json.dumps(descriptor) + r'''];
      visibility.cells = [{ id: "empty", min: [-2,-2,-2], max: [2,2,2], visibleClusterIds: [] }];
      const pvs = createPvsController({ scene: scene(furniture), visibility });
      assert.equal(pvs.valid, true);
      pvs.update({x:0,y:0,z:0});
      assert.equal(furniture.visible, false);
      assert.equal(shelf.visible, false);
      assert.equal(leaf.visible, false);
      pvs.update({x:100,y:0,z:0});
      assert.equal(furniture.visible, true);
      assert.equal(shelf.visible, true);
      assert.equal(leaf.visible, true);
    ''')


@pytest.mark.parametrize("descendant_name", (None, "", "Furniture root"))
def test_ambiguous_furniture_descendant_names_leave_whole_object_visible(descendant_name):
    document, binary = _fixture()
    document["nodes"].insert(0, {
        "name": "Furniture root", "children": [1, 2],
        "extras": {"housemaker_object_id": "cupboard"},
    })
    document["nodes"].append({"children": [], **({"name": descendant_name} if descendant_name is not None else {})})
    payload = _encode(document, binary)

    assert partition_glb_for_pvs(payload) == (payload, ())


def test_object_atlas_batch_can_be_partitioned_but_never_becomes_an_occluder():
    document, binary = _fixture()
    document["nodes"][0]["extras"] = {"housemaker_atlas_type": "object"}
    payload = _encode(document, binary)
    output, clusters = partition_glb_for_pvs(payload)

    assert output != payload
    assert len(clusters) == 2
    assert _decode(output)[0]["nodes"][0]["name"] == "atlas"
    assert not len(collect_pvs_occluder_triangles(payload))
    assert not len(collect_pvs_occluder_triangles(output))


@pytest.mark.parametrize("extras", (
    {"halfMesh": {"mirrorPlane": {"normal": [1, 0, 0]}}},
    {"housemaker_door_body": {"bodyObjectId": "door"}},
    {"housemakerDoorComponent": {"componentObjectId": "knob"}},
    {"housemaker_instance_source_id": "source", "housemaker_instance_source_name": "Source original name"},
    {"housemaker_dynamic": True},
))
def test_half_door_and_instance_prototype_nodes_are_untouched_and_always_visible(extras):
    document, binary = _fixture()
    document["nodes"][0]["extras"] = extras
    payload = _encode(document, binary)

    output, clusters = partition_glb_for_pvs(payload)

    assert output == payload and clusters == ()
    assert not len(collect_pvs_occluder_triangles(payload))


def test_dynamic_child_excludes_its_entire_object_root_from_culling():
    document, binary = _fixture()
    document["nodes"].insert(0, {"name": "Door root", "children": [1], "extras": {"housemaker_object_id": "door-root"}})
    document["nodes"][1]["extras"] = {"housemakerDoorBody": {"bodyObjectId": "door-body"}}
    payload = _encode(document, binary)

    output, clusters = partition_glb_for_pvs(payload)

    assert output == payload and clusters == ()


@pytest.mark.parametrize("node_flag", ({"skin": 0}, {"weights": [0.5]}, {"extensions": {"EXT_mesh_gpu_instancing": {}}}))
def test_animated_skinned_and_gpu_instance_nodes_are_never_split(node_flag):
    document, binary = _fixture()
    document["nodes"][0].update(node_flag)
    payload = _encode(document, binary)
    assert partition_glb_for_pvs(payload) == (payload, ())
    assert not len(collect_pvs_occluder_triangles(payload))


def test_animation_target_and_its_descendants_are_always_visible():
    document, binary = _fixture()
    document["nodes"].insert(0, {"name": "Animated parent", "children": [1]})
    document["animations"] = [{"channels": [{"target": {"node": 0, "path": "translation"}}]}]
    payload = _encode(document, binary)
    assert partition_glb_for_pvs(payload) == (payload, ())


# ### Opaque occluders and unsupported streams fail open ###
@pytest.mark.parametrize("double_sided", (None, False, 0, 1, "true"))
def test_single_sided_occluders_preserve_original_triangle_winding(double_sided):
    document, binary = _fixture()
    if double_sided is not None:
        document["materials"][0]["doubleSided"] = double_sided

    triangles = collect_pvs_occluder_triangles(_encode(document, binary))

    originals = _POINTS.reshape(-1, 3, 3)
    np.testing.assert_array_equal(triangles, originals)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    np.testing.assert_array_equal(normals, np.tile((0, 0, 1), (2, 1)))


def test_explicitly_double_sided_opaque_material_adds_reverse_winding_copies():
    document, binary = _fixture()
    document["materials"][0]["doubleSided"] = True

    triangles = collect_pvs_occluder_triangles(_encode(document, binary))

    originals = _POINTS.reshape(-1, 3, 3)
    assert triangles.shape == (4, 3, 3)
    np.testing.assert_array_equal(triangles[:2], originals)
    np.testing.assert_array_equal(triangles[2:], originals[:, (0, 2, 1)])
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    np.testing.assert_array_equal(normals[:2], -normals[2:])


@pytest.mark.parametrize("mirror_source", ("node", "parent", "world"))
def test_mirrored_transforms_preserve_the_rendered_front_side_not_reflected_winding(mirror_source):
    document, binary = _fixture()
    world = np.eye(4)
    if mirror_source == "node":
        document["nodes"][0]["scale"] = [-1, 1, 1]
    elif mirror_source == "parent":
        document["nodes"].insert(0, {"name": "Mirrored parent", "scale": [-1, 1, 1], "children": [1]})
    else:
        world[0, 0] = -1
    original_payload = _encode(document, binary)

    triangles = collect_pvs_occluder_triangles(original_payload, world_transform=world)

    reflected = (_POINTS * (-1, 1, 1)).reshape(-1, 3, 3)
    np.testing.assert_array_equal(triangles, reflected[:, (0, 2, 1)])
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    np.testing.assert_array_equal(normals, np.tile((0, 0, 1), (2, 1)))
    # Bounds and serialized geometry remain based on the original transforms.
    output, clusters = partition_glb_for_pvs(original_payload, world_transform=world)
    np.testing.assert_allclose(np.min([cluster.minimum for cluster in clusters], axis=0), reflected.reshape(-1, 3).min(axis=0))
    after_partition = collect_pvs_occluder_triangles(output, world_transform=world)
    assert sorted(triangle.reshape(-1).tolist() for triangle in after_partition) == sorted(
        triangle.reshape(-1).tolist() for triangle in triangles
    )


def test_two_reflections_cancel_without_an_extra_front_face_reversal():
    document, binary = _fixture()
    document["nodes"][0]["scale"] = [-1, 1, 1]
    world = np.diag((-1, 1, 1, 1))

    triangles = collect_pvs_occluder_triangles(_encode(document, binary), world_transform=world)

    np.testing.assert_array_equal(triangles, _POINTS.reshape(-1, 3, 3))


def test_singular_transform_stays_visible_but_cannot_become_a_one_sided_occluder():
    document, binary = _fixture()
    document["nodes"][0]["scale"] = [1, 1, 0]
    payload = _encode(document, binary)

    assert collect_pvs_occluder_triangles(payload).shape == (0, 3, 3)
    assert len(partition_glb_for_pvs(payload)[1]) == 2


@pytest.mark.parametrize("exclusion", ("glass", "door", "object", "prototype", "sparse"))
def test_double_sided_flag_never_overrides_occluder_exclusions(exclusion):
    document, binary = _fixture()
    document["materials"][0]["doubleSided"] = True
    if exclusion == "glass":
        document["materials"][0]["alphaMode"] = "BLEND"
    elif exclusion == "door":
        document["nodes"][0]["extras"] = {"housemakerDoorBody": {"bodyObjectId": "door"}}
    elif exclusion == "object":
        document["nodes"][0]["extras"] = {"housemaker_atlas_type": "object"}
    elif exclusion == "prototype":
        document["nodes"][0]["extras"] = {"housemaker_instance_source_id": "source"}
    else:
        document["accessors"][0]["sparse"] = {"count": 1}

    assert collect_pvs_occluder_triangles(_encode(document, binary)).shape == (0, 3, 3)


@pytest.mark.parametrize("material", (
    {"alphaMode": "BLEND"}, {"alphaMode": "MASK"}, {"name": "HouseMaker Glass"},
    {"extensions": {"KHR_materials_transmission": {"transmissionFactor": 0.5}}},
))
def test_transparent_materials_do_not_occlude_but_keep_their_original_drawable(material):
    document, binary = _fixture()
    document["materials"][0].update(material)
    payload = _encode(document, binary)

    output, clusters = partition_glb_for_pvs(payload)

    assert len(clusters) == 2
    assert not len(collect_pvs_occluder_triangles(output))
    assert _decode(output)[0]["materials"] == document["materials"]


@pytest.mark.parametrize("unsupported", ("sparse-position", "sparse-normal", "compressed", "quantized", "non-triangles", "meshopt"))
def test_unsupported_meshes_remain_byte_exact_and_always_visible(unsupported):
    document, binary = _fixture()
    primitive = document["meshes"][0]["primitives"][0]
    position = document["accessors"][primitive["attributes"]["POSITION"]]
    if unsupported.startswith("sparse"):
        index = primitive["attributes"]["NORMAL" if unsupported == "sparse-normal" else "POSITION"]
        document["accessors"][index]["sparse"] = {"count": 1}
    elif unsupported == "compressed":
        primitive["extensions"]["KHR_draco_mesh_compression"] = {"bufferView": 0}
    elif unsupported == "quantized":
        position["componentType"] = 5123
    elif unsupported == "non-triangles":
        primitive["mode"] = 5
    else:
        document["bufferViews"][position["bufferView"]]["extensions"] = {"EXT_meshopt_compression": {}}
    payload = _encode(document, binary)

    assert partition_glb_for_pvs(payload) == (payload, ())
    assert not len(collect_pvs_occluder_triangles(payload))


def test_interleaved_position_and_index_accessor_offsets_are_respected():
    document, binary = _fixture()
    primitive = document["meshes"][0]["primitives"][0]
    position = document["accessors"][primitive["attributes"]["POSITION"]]
    view = document["bufferViews"][position["bufferView"]]
    interleaved = np.column_stack((np.full(len(_POINTS), -99, dtype="<f4"), _POINTS)).astype("<f4").tobytes()
    offset = len(binary)
    binary += interleaved
    view.update(byteOffset=offset, byteLength=len(interleaved), byteStride=16)
    position["byteOffset"] = 4
    document["buffers"][0]["byteLength"] = len(binary)

    output, clusters = partition_glb_for_pvs(_encode(document, binary))

    assert len(clusters) == 2
    np.testing.assert_array_equal(collect_pvs_occluder_triangles(output), _POINTS.reshape(-1, 3, 3))


def test_rotated_nodes_and_index_offsets_have_matching_partition_and_occluder_bounds():
    document, binary = _fixture()
    document["nodes"][0]["rotation"] = [0, 0, 2**-0.5, 2**-0.5]
    document["nodes"][0]["translation"] = [3, 2, 1]
    primitive = document["meshes"][0]["primitives"][0]
    accessor = document["accessors"][primitive["indices"]]
    view = document["bufferViews"][accessor["bufferView"]]
    view["byteOffset"] = len(binary)
    values = np.concatenate((np.asarray((99, 99), dtype="<u2"), np.arange(6, dtype="<u2"))).tobytes()
    view["byteLength"] = len(values)
    accessor["byteOffset"] = 4
    binary += values
    document["buffers"][0]["byteLength"] = len(binary)

    output, clusters = partition_glb_for_pvs(_encode(document, binary))
    triangles = collect_pvs_occluder_triangles(output)

    expected = np.column_stack((-_POINTS[:, 1], _POINTS[:, 0], _POINTS[:, 2])) + (3, 2, 1)
    np.testing.assert_allclose(triangles.reshape(-1, 3), expected, atol=1e-12)
    np.testing.assert_allclose(np.min([cluster.minimum for cluster in clusters], axis=0), expected.min(axis=0))
    np.testing.assert_allclose(np.max([cluster.maximum for cluster in clusters], axis=0), expected.max(axis=0))


def test_inactive_library_scene_is_not_partitioned_or_collected_as_occluding_geometry():
    document, binary = _fixture()
    document["nodes"].append({"name": "Library only", "mesh": 0, "translation": [100, 0, 0]})
    document["scenes"].append({"name": "Library", "nodes": [1]})
    payload = _encode(document, binary)

    output, clusters = partition_glb_for_pvs(payload)
    result, _, _ = _decode(output)

    assert len(clusters) == 2
    assert result["nodes"][1] == document["nodes"][1]
    assert result["scenes"] == document["scenes"]
    assert collect_pvs_occluder_triangles(output).shape == (2, 3, 3)


def test_unnamed_furniture_root_stays_visible_instead_of_receiving_a_nonexistent_runtime_name():
    document, binary = _fixture()
    document["nodes"][0].pop("name")
    document["nodes"][0]["extras"] = {"housemaker_object_id": "unnamed-furniture"}
    payload = _encode(document, binary)

    assert partition_glb_for_pvs(payload) == (payload, ())


# ### Input validation ###
@pytest.mark.parametrize("size", (True, 0, -1, float("nan"), float("inf"), "4", 10**1000))
def test_invalid_cell_sizes_are_rejected(size):
    document, binary = _fixture()
    with pytest.raises((TypeError, ValueError)):
        partition_glb_for_pvs(_encode(document, binary), cell_size=size)


def test_invalid_glb_and_non_affine_world_transform_are_rejected():
    with pytest.raises(ValueError, match="GLB"):
        partition_glb_for_pvs(b"not-glb")
    document, binary = _fixture()
    transform = np.eye(4)
    transform[3, 0] = 1
    with pytest.raises(ValueError, match="affine"):
        partition_glb_for_pvs(_encode(document, binary), world_transform=transform)
