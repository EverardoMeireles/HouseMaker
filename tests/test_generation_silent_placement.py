# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication

from housemaker.generation_state import (
    GeneratedObjectPlacement,
    GeneratedObjectRecord,
    GenerationData,
)
from housemaker.generation_workspace import GenerationWorkspace

# ### Fixtures ###
_application = QApplication.instance() or QApplication([])
_application.setQuitOnLastWindowClosed(False)


@pytest.fixture
def workspace(tmp_path):
    widget = GenerationWorkspace(asset_directory=tmp_path)
    widget._data = GenerationData(
        generated_objects=[
            GeneratedObjectRecord(
                object_id="source-a",
                frame_index=0,
                object_name="Source A",
                pipeline={},
                provider="meshy",
                provider_task_id="task-a",
                asset_path="source-a.glb",
            )
        ]
    )
    yield widget
    widget._object_job_runtimes.clear()
    widget.shutdown()
    widget.close()
    _application.processEvents()


def _placement_signals(workspace):
    return tuple(
        QSignalSpy(signal)
        for signal in (
            workspace.data_changed,
            workspace.generated_object_placement_changed,
            workspace.generated_object_instances_changed,
            workspace.placeable_objects_changed,
        )
    )


def _request(workspace):
    assert workspace.request_generated_object_placement("source-a")
    return workspace._existing_object_placement_request.request_id


# ### Source placement notification tests ###
def test_silent_source_placement_is_saved_immediately_without_refresh(workspace):
    request_id = _request(workspace)
    signals = _placement_signals(workspace)
    placement = GeneratedObjectPlacement(2, 10.0, 20.0)
    with patch.object(workspace, "_sync_controls") as controls, patch.object(
        workspace, "_display_generated_object", side_effect=AssertionError("display")
    ):
        assert workspace.set_active_object_placement(
            request_id, placement, emit_change_signals=False
        )
    assert workspace.get_generated_object_placement("source-a") == placement
    assert workspace.get_data().generated_objects[0].placement == placement
    assert workspace._existing_object_placement_request is None
    assert [spy.count() for spy in signals] == [0, 0, 0, 0]
    controls.assert_not_called()


def test_default_source_placement_preserves_existing_notifications(workspace):
    request_id = _request(workspace)
    signals = _placement_signals(workspace)
    with patch.object(workspace, "_sync_controls") as controls:
        assert workspace.set_active_object_placement(
            request_id, GeneratedObjectPlacement(2, 10.0, 20.0)
        )
    assert [spy.count() for spy in signals] == [1, 1, 0, 0]
    controls.assert_called_once_with()


def test_silent_replacement_keeps_height_rotation_and_scale(workspace):
    previous = GeneratedObjectPlacement(
        2, 1.0, 2.0, height_offset_meters=0.7,
        rotation_degrees=(10.0, 20.0, 30.0), scale=1.3,
        axis_scales=(1.0, 1.2, 1.4),
    )
    workspace._data.generated_objects[0] = replace(
        workspace._data.generated_objects[0], placement=previous
    )
    request_id = _request(workspace)
    assert workspace.set_active_object_placement(
        request_id, GeneratedObjectPlacement(2, 11.0, 22.0),
        emit_change_signals=False,
    )
    assert workspace.get_generated_object_placement("source-a") == replace(
        previous, image_x=11.0, image_y=22.0
    )


def test_silent_stale_request_cannot_change_or_close_current_request(workspace):
    request_id = _request(workspace)
    assert not workspace.set_active_object_placement(
        "stale-request", GeneratedObjectPlacement(2, 10.0, 20.0),
        emit_change_signals=False,
    )
    assert workspace._existing_object_placement_request.request_id == request_id
    assert workspace.get_generated_object_placement("source-a") is None


def test_silent_running_job_keeps_pending_placement_without_catalog_publish(workspace):
    operation = SimpleNamespace(pending_placement=None)
    workspace._object_job_runtimes["operation-a"] = SimpleNamespace(operation=operation)
    signals = _placement_signals(workspace)
    placement = GeneratedObjectPlacement(2, 10.0, 20.0)
    with patch.object(workspace, "_can_place_active_operation", return_value=True):
        assert workspace.set_active_object_placement(
            "operation-a", placement, emit_change_signals=False
        )
    assert operation.pending_placement == placement
    assert [spy.count() for spy in signals] == [0, 0, 0, 0]


def test_silent_source_placement_can_publish_current_value_later(workspace):
    request_id = _request(workspace)
    assert workspace.set_active_object_placement(
        request_id, GeneratedObjectPlacement(2, 10.0, 20.0),
        emit_change_signals=False,
    )
    latest = GeneratedObjectPlacement(2, 30.0, 40.0)
    assert workspace.update_generated_object_placement(
        "source-a", latest, emit_change_signals=False
    )
    signals = _placement_signals(workspace)
    workspace.publish_placeable_object_placement_changes(("source-a",))
    assert [spy.count() for spy in signals] == [1, 1, 0, 1]
    assert signals[1].at(0)[0].placement == latest


def test_deferred_placement_publication_syncs_controls_without_loading_preview(workspace):
    request_id = _request(workspace)
    with patch.object(workspace, "_sync_controls") as controls, patch.object(
        workspace, "_display_generated_object", side_effect=AssertionError("display")
    ):
        assert workspace.set_active_object_placement(
            request_id, GeneratedObjectPlacement(2, 10.0, 20.0),
            emit_change_signals=False,
        )
        controls.assert_not_called()

        workspace.publish_placeable_object_placement_changes(("source-a",))

        controls.assert_called_once_with()
    assert workspace._existing_object_placement_request is None


# ### Instance placement notification tests ###
def test_silent_instance_is_saved_and_linked_without_notifying(workspace):
    signals = _placement_signals(workspace)
    placement = GeneratedObjectPlacement(2, 10.0, 20.0)
    instance = workspace.add_generated_object_instance(
        "source-a", placement, emit_change_signals=False
    )
    assert instance is not None
    assert instance.source_object_id == "source-a"
    assert instance.placement == placement
    assert workspace.get_generated_object_placement("source-a") is None
    assert workspace.get_data().object_instances == [instance]
    assert [spy.count() for spy in signals] == [0, 0, 0, 0]
    workspace.publish_generated_object_instance_changes()
    assert [spy.count() for spy in signals] == [1, 0, 1, 0]


def test_default_instance_placement_preserves_notifications(workspace):
    signals = _placement_signals(workspace)
    assert workspace.add_generated_object_instance(
        "source-a", GeneratedObjectPlacement(2, 10.0, 20.0)
    ) is not None
    assert [spy.count() for spy in signals] == [1, 0, 1, 0]


def test_silent_instance_rejects_missing_source_without_changing_data(workspace):
    signals = _placement_signals(workspace)
    assert workspace.add_generated_object_instance(
        "missing-source", GeneratedObjectPlacement(2, 10.0, 20.0),
        emit_change_signals=False,
    ) is None
    assert workspace.get_generated_object_instances() == ()
    assert [spy.count() for spy in signals] == [0, 0, 0, 0]
