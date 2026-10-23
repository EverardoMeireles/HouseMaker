"""Numbered nodes remove custom-name ambiguity without replacing stable UUIDs."""

# ### Imports ###
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from housemaker.glb import (
    OBJECT_EXPORT_ID_METADATA_KEY,
    OBJECT_NAME_METADATA_KEY,
    PlacedGeneratedModel,
    compose_generated_model_instance_sources,
    compose_placed_generated_models,
    export_glb_file,
)
from housemaker.scene_export import (
    build_runtime_scene_manifest,
    write_runtime_scene_manifest,
)
from tests.test_scene_export import (
    _empty_model,
    _read_glb_document,
    _source_model_with_node_name,
)


# ### Numbered sources and copies with deliberately identical display names ###
def _placements():
    model = _source_model_with_node_name("identical-imported-mesh")
    source_a = PlacedGeneratedModel(
        "uuid-source-a",
        model,
        (1, 2, 0),
        object_name="Cupboard",
        export_node_name="Object_1",
        export_source_node_name="Object_1",
    )
    source_b = replace(
        source_a,
        object_id="uuid-source-b",
        source_object_id="uuid-source-b",
        export_node_name="Object_2",
        export_source_node_name="Object_2",
    )
    copy_a = replace(
        source_a,
        object_id="uuid-copy-a",
        world_position=(4, 5, 0),
        export_node_name="Object_3",
    )
    copy_b = replace(
        source_b,
        object_id="uuid-copy-b",
        world_position=(7, 8, 0),
        export_node_name="Object_4",
    )
    return source_a, source_b, copy_a, copy_b


def test_duplicate_custom_names_export_as_distinct_safe_prototypes_and_logical_ids(
    tmp_path,
):
    source_a, source_b, copy_a, copy_b = _placements()
    model = compose_generated_model_instance_sources(
        _empty_model(), (source_a, source_b)
    )
    path = export_glb_file(model, tmp_path / "house.glb")
    manifest_path = write_runtime_scene_manifest(
        path,
        source_placements={source_a.object_id: source_a, source_b.object_id: source_b},
        instance_placements=(copy_a, copy_b),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    groups = manifest["instanceGroups"]
    assert {group["sourceNodeName"] for group in groups} == {"Object_1", "Object_2"}
    by_id = {entry["id"]: entry for group in groups for entry in group["instances"]}
    assert set(by_id) == {
        source_a.object_id,
        source_b.object_id,
        copy_a.object_id,
        copy_b.object_id,
    }
    assert by_id[copy_a.object_id]["objectId"] == "Object 3"
    assert by_id[copy_a.object_id]["sourceObjectId"] == "Object 1"
    assert by_id[copy_b.object_id]["objectId"] == "Object 4"
    assert by_id[copy_b.object_id]["sourceObjectId"] == "Object 2"
    assert all(entry["name"] == "Cupboard" for entry in by_id.values())
    document = _read_glb_document(path.read_bytes())
    for label, uuid in (
        ("Object_1", source_a.object_id),
        ("Object_2", source_b.object_id),
    ):
        root = next(node for node in document["nodes"] if node.get("name") == label)
        assert root["extras"]["housemaker_instance_source_id"] == uuid
        assert root["extras"][OBJECT_NAME_METADATA_KEY] == "Cupboard"
        assert root["extras"][OBJECT_EXPORT_ID_METADATA_KEY] == label.replace("_", " ")


def test_unplaced_source_carrier_uses_source_identifier_not_its_own_instance_number(
    tmp_path,
):
    _source_a, _source_b, copy_a, _copy_b = _placements()
    model = compose_generated_model_instance_sources(_empty_model(), (copy_a,))
    path = export_glb_file(model, tmp_path / "house.glb")
    manifest_path = write_runtime_scene_manifest(
        path, source_placements={}, instance_placements=(copy_a,)
    )
    group = json.loads(manifest_path.read_text(encoding="utf-8"))["instanceGroups"][0]
    assert group["sourceNodeName"] == "Object_1"
    assert group["instances"][0]["id"] == copy_a.object_id
    assert group["instances"][0]["objectId"] == "Object 3"
    assert group["instances"][0]["sourceObjectId"] == "Object 1"
    nodes = _read_glb_document(path.read_bytes())["nodes"]
    root = next(node for node in nodes if node.get("name") == "Object_1")
    assert root["extras"]["housemaker_instance_source_id"] == copy_a.source_object_id


def test_ordinary_object_roots_use_numbered_ids_but_keep_uuid_and_custom_display_metadata():
    a, b, _copy_a, _copy_b = _placements()
    model = compose_placed_generated_models(_empty_model(), (a, b))
    nodes = _read_glb_document(model.glb_bytes)["nodes"]
    roots = {
        node["name"]: node
        for node in nodes
        if node.get("name") in {"Object_1", "Object_2"}
    }
    assert set(roots) == {"Object_1", "Object_2"}
    assert roots["Object_1"]["extras"]["housemaker_object_id"] == a.object_id
    assert roots["Object_2"]["extras"]["housemaker_object_id"] == b.object_id
    assert all(
        root["extras"][OBJECT_NAME_METADATA_KEY] == "Cupboard"
        for root in roots.values()
    )
    assert a.object_name == b.object_name == "Cupboard"


def test_duplicate_numbered_prototypes_are_rejected_instead_of_binding_the_wrong_asset():
    a, b, copy_a, copy_b = _placements()
    b = replace(b, export_node_name="Object_1", export_source_node_name="Object_1")
    copy_b = replace(copy_b, export_source_node_name="Object_1")
    with pytest.raises(ValueError, match="unique"):
        compose_generated_model_instance_sources(_empty_model(), (a, b))
    with pytest.raises(ValueError, match="unique"):
        build_runtime_scene_manifest(
            glb_name="house.glb",
            glb_bytes=b"glb",
            source_placements={a.object_id: a, b.object_id: b},
            instance_placements=(copy_a, copy_b),
        )


# ### Numbered half-mesh reconstruction stays independent of display names ###
@pytest.mark.parametrize("as_instance_source", (False, True))
def test_numbered_half_mesh_keeps_safe_name_uuid_and_reconstruction_metadata(
    tmp_path, as_instance_source
):
    source, _other, copy, _other_copy = _placements()
    source = replace(
        source,
        symmetric_preview_orientation="vertical",
        symmetric_preview_plane_coordinate=0.25,
    )
    copy = replace(
        copy,
        symmetric_preview_orientation="vertical",
        symmetric_preview_plane_coordinate=0.25,
    )
    compose = (
        compose_generated_model_instance_sources
        if as_instance_source
        else compose_placed_generated_models
    )
    model = compose(_empty_model(), (source,))
    path = export_glb_file(model, tmp_path / "half.glb")
    if as_instance_source:
        manifest_path = write_runtime_scene_manifest(
            path,
            source_placements={source.object_id: source},
            instance_placements=(copy,),
        )
        group = json.loads(manifest_path.read_text(encoding="utf-8"))["instanceGroups"][
            0
        ]
        assert group["sourceNodeName"] == "Object_1"
        assert group["halfMesh"] is True
        assert {item["objectId"] for item in group["instances"]} == {
            "Object 1",
            "Object 3",
        }
    document = _read_glb_document(path.read_bytes())
    root = next(node for node in document["nodes"] if node.get("name") == "Object_1")
    assert root["extras"]["housemaker_object_id"] == source.object_id
    assert root["extras"][OBJECT_EXPORT_ID_METADATA_KEY] == "Object 1"
    assert root["extras"][OBJECT_NAME_METADATA_KEY] == "Cupboard"
    half_nodes = [
        node for node in document["nodes"] if "halfMesh" in node.get("extras", {})
    ]
    assert half_nodes
    assert all("mesh" in node for node in half_nodes)
    legacy = compose(
        _empty_model(),
        (replace(source, export_node_name=None, export_source_node_name=None),),
    )
    legacy_half_extras = [
        node["extras"]["halfMesh"]
        for node in _read_glb_document(legacy.glb_bytes)["nodes"]
        if "halfMesh" in node.get("extras", {})
    ]
    assert [node["extras"]["halfMesh"] for node in half_nodes] == legacy_half_extras


@pytest.mark.parametrize("name", ("Object 1", "Object_0", "Object_01", "Cupboard", ""))
def test_unsafe_or_non_numbered_explicit_node_overrides_are_rejected(name):
    a, *_rest = _placements()
    with pytest.raises(ValueError):
        replace(a, export_node_name=name)
