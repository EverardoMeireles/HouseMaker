"""Numbered export identifiers persist independently from object UUID ownership."""

# ### Imports ###
from __future__ import annotations

import json

import pytest

from housemaker.models import GROUND_LEVEL_INDEX, create_default_levels
from housemaker.object_identifiers import (
    allocate_object_export_ids,
    normalize_object_export_ids,
    object_export_label_from_node_name,
    object_export_node_name,
)
from housemaker.project_io import ProjectData, load_project, save_project


# ### Stable labels and Three.js-safe node identifiers ###
def test_allocation_keeps_existing_deleted_entries_and_never_reuses_numbers():
    initial = {"deleted-uuid": "Object 1", "existing-uuid": "Object 4"}
    result = allocate_object_export_ids(
        initial, ("new-uuid", "existing-uuid", "new-copy")
    )
    assert result == {**initial, "new-uuid": "Object 5", "new-copy": "Object 6"}
    assert initial == {"deleted-uuid": "Object 1", "existing-uuid": "Object 4"}
    assert allocate_object_export_ids(result, reversed(tuple(result))) == result


def test_new_labels_follow_input_order_and_repeated_source_ids_do_not_allocate_twice():
    assert allocate_object_export_ids(
        None, ("source", "copy", "source", "unplaced")
    ) == {
        "source": "Object 1",
        "copy": "Object 2",
        "unplaced": "Object 3",
    }


def test_node_names_avoid_gltfloader_whitespace_sanitization():
    assert object_export_node_name("Object 12") == "Object_12"
    assert object_export_label_from_node_name("Object_12") == "Object 12"
    assert normalize_object_export_ids(None) == {}


@pytest.mark.parametrize(
    "invalid",
    (
        [],
        "Object 1",
        {"a": "Object 0"},
        {"a": "Object 01"},
        {"a": "Object_1"},
        {"a": "object 1"},
        {"a": "Object -1"},
        {"a": "Object 1", "b": "Object 1"},
        {" a ": "Object 1", "a": "Object 2"},
        {"": "Object 1"},
        {True: "Object 1"},
    ),
)
def test_malformed_or_ambiguous_identity_maps_are_rejected(invalid):
    with pytest.raises((TypeError, ValueError)):
        normalize_object_export_ids(invalid)


def test_failed_allocation_never_changes_the_previous_mapping():
    initial = {"a": "Object 1"}
    with pytest.raises(ValueError):
        allocate_object_export_ids(initial, ("new", ""))
    assert initial == {"a": "Object 1"}


# ### Project persistence, legacy loading and transactional validation ###
def test_project_restores_the_same_numbered_labels_without_dropping_deleted_ids(
    tmp_path,
):
    labels = {
        "object-uuid": "Object 1",
        "instance-uuid": "Object 2",
        "deleted": "Object 3",
    }
    path = save_project(
        tmp_path / "house.json",
        GROUND_LEVEL_INDEX,
        create_default_levels(),
        object_export_ids=labels,
    )
    loaded = load_project(path)
    assert loaded.object_export_ids == labels
    assert (
        allocate_object_export_ids(loaded.object_export_ids, ("later",))["later"]
        == "Object 4"
    )
    owned = ProjectData(
        None, GROUND_LEVEL_INDEX, create_default_levels(), object_export_ids=labels
    )
    labels.clear()
    assert len(owned.object_export_ids) == 3


@pytest.mark.parametrize("invalid", (None, {"bad": "duplicate name"}))
def test_legacy_or_invalid_optional_labels_do_not_damage_the_editable_project(
    tmp_path, invalid
):
    path = save_project(
        tmp_path / "house.json", GROUND_LEVEL_INDEX, create_default_levels()
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if invalid is None:
        payload.pop("object_export_ids")
    else:
        payload["object_export_ids"] = invalid
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_project(path).object_export_ids == {}


def test_invalid_labels_cannot_overwrite_a_valid_project(tmp_path):
    path = save_project(
        tmp_path / "house.json", GROUND_LEVEL_INDEX, create_default_levels()
    )
    original = path.read_bytes()
    with pytest.raises(ValueError):
        save_project(
            path,
            GROUND_LEVEL_INDEX,
            create_default_levels(),
            object_export_ids={"a": "Object 1", "b": "Object 1"},
        )
    assert path.read_bytes() == original
