"""Export numbering is stable without replacing UUIDs or custom object names."""

# ### Environment and imports ###
from __future__ import annotations

import os
from dataclasses import replace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import trimesh
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.generation_state import (
    GeneratedObjectInstance,
    GeneratedObjectPlacement,
    GeneratedObjectRecord,
)
from housemaker.glb import GeneratedModel, PlacedGeneratedModel
from housemaker.main import BlueprintWorkspace
from housemaker.models import GROUND_LEVEL_INDEX, create_default_levels
from housemaker.project_io import load_project, save_project

# ### Lightweight Main and catalog fixtures ###
_application = QApplication.instance() or QApplication([])
_application.setQuitOnLastWindowClosed(False)


@pytest.fixture
def workspace(tmp_path):
    widget = BlueprintWorkspace(
        application_settings=ApplicationSettingsStore(tmp_path / "settings.json"),
    )
    try:
        yield widget
    finally:
        widget.shutdown()
        widget.close()
        widget.deleteLater()
        _application.processEvents()


def _placement(level_index=GROUND_LEVEL_INDEX):
    return GeneratedObjectPlacement(level_index, 10.0, 20.0)


def _record(identifier, *, name="Cupboard", placed=True, level_index=GROUND_LEVEL_INDEX):
    return GeneratedObjectRecord(
        object_id=identifier, frame_index=0, object_name=name, pipeline={},
        provider_task_id=f"task-{identifier}", asset_path=f"unused-{identifier}.glb",
        placement=_placement(level_index) if placed else None,
    )


def _instance(identifier, source, *, level_index=GROUND_LEVEL_INDEX):
    return GeneratedObjectInstance(identifier, source, _placement(level_index))


def _model():
    mesh = trimesh.creation.box()
    return GeneratedModel(mesh=mesh, scene=trimesh.Scene(mesh), glb_bytes=b"")


# ### Stable monotonic allocation across the whole authoring catalog ###
def test_empty_project_has_no_numbered_object_ids(workspace):
    assert workspace.object_export_ids == {}
    assert workspace._ensure_numbered_object_ids() == {}


def test_unplaced_excluded_and_instance_records_all_reserve_numbers(workspace):
    excluded_level = workspace.levels[-1]
    excluded_level.include_in_export = False
    records = (
        _record("source-a"),
        _record("source-b", level_index=excluded_level.index),
        _record("unplaced-source", placed=False),
    )
    instances = (_instance("copy-a", "source-a", level_index=excluded_level.index),)
    with (
        patch.object(workspace.generation, "get_generated_object_records", return_value=records),
        patch.object(workspace.generation, "get_generated_object_instances", return_value=instances),
    ):
        assert workspace._ensure_numbered_object_ids() == {
            "source-a": "Object 1", "source-b": "Object 2",
            "unplaced-source": "Object 3", "copy-a": "Object 4",
        }


def test_deleted_numbers_are_retained_and_never_recycled(workspace):
    with (
        patch.object(workspace.generation, "get_generated_object_records") as records,
        patch.object(workspace.generation, "get_generated_object_instances", return_value=()),
    ):
        records.return_value = (_record("a"), _record("b"), _record("c"))
        workspace._ensure_numbered_object_ids()
        records.return_value = (_record("a"), _record("d"))
        labels = workspace._ensure_numbered_object_ids()
        assert labels["b"] == "Object 2"
        assert labels["c"] == "Object 3"
        assert labels["d"] == "Object 4"
        records.return_value = (_record("e"),)
        labels = workspace._ensure_numbered_object_ids()
        assert labels["d"] == "Object 4"
        assert labels["e"] == "Object 5"


def test_rename_and_reordering_never_reassign_numbered_identities(workspace):
    first, second = _record("a"), _record("b")
    with (
        patch.object(workspace.generation, "get_generated_object_records",
                     return_value=(first, second)) as records,
        patch.object(workspace.generation, "get_generated_object_instances", return_value=()),
    ):
        initial = dict(workspace._ensure_numbered_object_ids())
        records.return_value = (replace(second, object_name="New name"),
                                replace(first, object_name="New name"))
        assert workspace._ensure_numbered_object_ids() == initial


def test_derived_opening_object_ids_can_reserve_stable_extra_numbers(workspace):
    labels = workspace._ensure_numbered_object_ids(("door-placement:body", "window-placement:handle"))
    assert labels == {
        "door-placement:body": "Object 1", "window-placement:handle": "Object 2",
    }
    assert workspace._ensure_numbered_object_ids(("window-placement:handle",)) == labels


# ### Duplicate names and copy/source bindings use loader-safe node names ###
def test_two_same_named_sources_and_copies_keep_distinct_export_nodes(workspace):
    records = (_record("source-a"), _record("source-b"))
    instances = (_instance("copy-a1", "source-a"), _instance("copy-a2", "source-a"),
                 _instance("copy-b1", "source-b"))
    model = _model()
    raw = tuple(PlacedGeneratedModel(
        identifier, model, (float(index), 0.0, 0.0), object_name="Cupboard",
        source_object_id=source,
    ) for index, (identifier, source) in enumerate((
        ("source-a", "source-a"), ("source-b", "source-b"),
        ("copy-a1", "source-a"), ("copy-a2", "source-a"), ("copy-b1", "source-b"),
    )))
    with (
        patch.object(workspace.generation, "get_generated_object_records", return_value=records),
        patch.object(workspace.generation, "get_generated_object_instances", return_value=instances),
        patch.object(workspace, "_build_opening_target_lookup_for_levels", return_value={}),
        patch.object(workspace, "_build_placed_generated_models", return_value=raw),
    ):
        sources, copies = workspace._build_export_placed_models()
    assert len(sources) == 2
    assert len(copies) == 3
    all_placements = (*sources, *copies)
    assert [placement.export_node_name for placement in all_placements] == [
        "Object_1", "Object_2", "Object_3", "Object_4", "Object_5",
    ]
    assert [placement.export_source_node_name for placement in all_placements] == [
        "Object_1", "Object_2", "Object_1", "Object_1", "Object_2",
    ]
    for decorated, original in zip(all_placements, raw, strict=True):
        assert decorated.object_id == original.object_id
        assert decorated.source_object_id == original.source_object_id
        assert decorated.object_name == "Cupboard"
        assert decorated.model is original.model
        assert decorated.world_position == original.world_position
        assert original.export_node_name is None
        assert " " not in decorated.export_node_name


def test_runtime_only_copy_also_reserves_its_absent_source_identity(workspace):
    model = _model()
    raw = (PlacedGeneratedModel("copy", model, (0, 0, 0), object_name="Same name",
                                source_object_id="not-currently-placed-source"),)
    with (
        patch.object(workspace.generation, "get_generated_object_records", return_value=()),
        patch.object(workspace.generation, "get_generated_object_instances", return_value=()),
        patch.object(workspace, "_build_opening_target_lookup_for_levels", return_value={}),
        patch.object(workspace, "_build_placed_generated_models", return_value=raw),
    ):
        sources, copies = workspace._build_export_placed_models()
    assert sources == ()
    assert len(copies) == 1
    assert copies[0].export_node_name != copies[0].export_source_node_name
    assert workspace.object_export_ids["not-currently-placed-source"] != workspace.object_export_ids["copy"]


# ### Main save/apply and real project load preserve the high-water mark ###
def test_save_reserves_unplaced_records_and_passes_the_mapping_to_project_io(workspace, tmp_path):
    with (
        patch.object(workspace.generation, "get_generated_object_records",
                     return_value=(_record("unplaced", placed=False),)),
        patch.object(workspace.generation, "get_generated_object_instances", return_value=()),
        patch("housemaker.main.QFileDialog.getSaveFileName",
              return_value=(str(tmp_path / "project.json"), "JSON Files (*.json)")),
        patch("housemaker.main.save_project") as save,
        patch("housemaker.main.QMessageBox.information"),
    ):
        workspace._handle_save_clicked()
    assert save.call_count == 1
    assert save.call_args.kwargs["object_export_ids"] == {"unplaced": "Object 1"}


def test_apply_copies_persisted_mapping_and_allocates_after_deleted_highest_id(workspace):
    saved = {"old-source": "Object 3", "deleted-object": "Object 17"}
    workspace._apply_project_state(
        levels=create_default_levels(), current_level_index=GROUND_LEVEL_INDEX,
        object_export_ids=saved,
    )
    assert workspace.object_export_ids == saved
    assert workspace.object_export_ids is not saved
    labels = workspace._ensure_numbered_object_ids(("next-source",))
    assert labels["next-source"] == "Object 18"
    assert saved == {"old-source": "Object 3", "deleted-object": "Object 17"}


def test_loading_real_project_restores_numbering_without_reusing_deleted_labels(workspace, tmp_path):
    path = tmp_path / "numbered-project.json"
    labels = {"old-source": "Object 4", "deleted-copy": "Object 12"}
    save_project(path, GROUND_LEVEL_INDEX, create_default_levels(), object_export_ids=labels)
    assert load_project(path).object_export_ids == labels
    with (
        patch("housemaker.main.QFileDialog.getOpenFileName",
              return_value=(str(path), "JSON Files (*.json)")),
        patch("housemaker.main.QMessageBox.critical") as error,
    ):
        workspace._handle_load_clicked()
    error.assert_not_called()
    assert workspace.object_export_ids == labels
    assert workspace._ensure_numbered_object_ids(("new-copy",))["new-copy"] == "Object 13"
