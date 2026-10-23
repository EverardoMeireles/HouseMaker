"""Bake records own their inputs and reject unsafe runtime visibility tables."""

# ### Imports ###
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from housemaker.pvs_state import PvsBakeData, PvsCell, PvsCluster


# ### Fixtures ###
def _bake(**overrides) -> PvsBakeData:
    values = {
        "source_signature": "a" * 64,
        "clusters": (PvsCluster("wall", ("wall-node",), (0, 0, 0), (6, 3, 0)),),
        "cells": (PvsCell("camera", (0, 1, 0), (2, 2, 2), ("wall",)),),
    }
    values.update(overrides)
    return PvsBakeData(**values)


# ### Immutable state and serialization ###
def test_bake_round_trip_uses_final_physical_y_up_bounds_and_safe_fallback():
    bake = _bake()
    payload = bake.to_dict()

    assert PvsBakeData.from_dict(payload) == bake
    assert payload["coordinateSystem"] == "gltf-y-up"
    assert payload["worldBounds"] == {"min": [0, 0, 0], "max": [6, 3, 2]}
    assert payload["fallback"] == payload["unlistedNodes"] == "frustum-only"
    assert payload["clusters"][0]["nodeNames"] == ["wall-node"]
    assert payload["cells"][0]["visibleClusterIds"] == ["wall"]
    assert bake.matches_source("a" * 64, eye_height_meters=1.70)
    assert not bake.matches_source("b" * 64)
    assert not bake.matches_source("a" * 64, eye_height_meters=1.8)


def test_mutable_inputs_are_copied_and_records_are_frozen():
    names, low, high, visible = ["node"], [0, 0, 0], [2, 3, 4], ["cluster"]
    cluster = PvsCluster("cluster", names, low, high)
    cell = PvsCell("cell", low, high, visible)
    clusters, cells = [cluster], [cell]
    bake = PvsBakeData("a" * 64, clusters, cells)
    names[0], low[0], high[0], visible[0] = "changed", -50, 100, "other"
    clusters.clear()
    cells.clear()

    assert bake.clusters == (cluster,)
    assert bake.cells == (cell,)
    assert cluster.node_names == ("node",)
    assert cluster.minimum == (0.0, 0.0, 0.0)
    assert cell.maximum == (2.0, 3.0, 4.0)
    assert cell.visible_cluster_ids == ("cluster",)
    with pytest.raises(FrozenInstanceError):
        bake.source_signature = "changed"


def test_instance_only_clusters_round_trip_without_claiming_the_prototype_node():
    cluster = PvsCluster("chair", (), (1, 0, 1), (2, 2, 2), ("chair-instance",))
    bake = _bake(
        clusters=(cluster,),
        cells=(PvsCell("camera", (0, 1, 0), (1, 2, 1), ("chair",)),),
    )

    assert PvsBakeData.from_dict(bake.to_dict()) == bake
    assert bake.to_dict()["clusters"][0]["instanceIds"] == ["chair-instance"]


def test_runtime_metadata_is_bound_to_the_specific_glb_bytes():
    payload = _bake().to_runtime_dict(glb_sha256="AB" * 32)
    assert payload["glbSha256"] == "ab" * 32
    assert payload["sourceSignature"] == "a" * 64


def test_optional_authoring_signature_round_trips_and_hashes_normalize():
    bake = _bake(source_signature="A" * 64, authoring_signature="B" * 64)
    assert bake.source_signature == "a" * 64
    assert bake.authoring_signature == "b" * 64
    assert bake.to_dict()["authoringSignature"] == "b" * 64
    assert PvsBakeData.from_dict(bake.to_dict()) == bake
    assert "authoringSignature" not in _bake().to_dict()


def test_invalid_optional_authoring_signature_is_rejected():
    with pytest.raises(ValueError, match="authoring signature"):
        _bake(authoring_signature="invalid")


# ### Rejected unsafe inputs ###
@pytest.mark.parametrize("value", ("", "bad\nname", None, 1, True))
def test_invalid_identifiers_are_rejected(value):
    with pytest.raises((TypeError, ValueError)):
        _bake(source_signature=value)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), True, "1"))
def test_nonfinite_or_non_numeric_bounds_are_rejected(value):
    with pytest.raises((TypeError, ValueError)):
        PvsCluster("id", ("node",), (value, 0, 0), (1, 1, 1))


def test_cluster_may_be_planar_but_camera_cells_require_positive_volume():
    assert _bake().clusters[0].minimum[2] == _bake().clusters[0].maximum[2]
    with pytest.raises(ValueError, match="positive volume"):
        PvsCell("cell", (0, 0, 0), (1, 1, 0), ("wall",))
    with pytest.raises(ValueError, match="minimum"):
        PvsCell("cell", (1, 0, 0), (0, 1, 1), ("wall",))


def test_empty_cluster_requires_a_node_or_instance_and_ids_are_unique():
    with pytest.raises(ValueError, match="node or instance"):
        PvsCluster("cluster", (), (0, 0, 0), (1, 1, 1))
    with pytest.raises(ValueError, match="duplicates"):
        PvsCell("cell", (0, 0, 0), (1, 1, 1), ("wall", "wall"))
    with pytest.raises(ValueError, match="unique"):
        _bake(cells=(_bake().cells[0], _bake().cells[0]))


def test_visibility_cannot_reference_unknown_clusters():
    with pytest.raises(ValueError, match="unknown"):
        _bake(cells=(PvsCell("cell", (0, 0, 0), (1, 1, 1), ("missing",)),))


@pytest.mark.parametrize("instance_only", (False, True))
def test_each_render_node_or_instance_belongs_to_only_one_cluster(instance_only):
    a = PvsCluster(
        "a",
        () if instance_only else ("node",),
        (0, 0, 0),
        (1, 1, 1),
        ("instance",) if instance_only else (),
    )
    b = replace(a, cluster_id="b")
    with pytest.raises(ValueError, match="multiple PVS clusters"):
        _bake(clusters=(a, b))


@pytest.mark.parametrize(
    "field,value",
    (
        ("version", 2),
        ("algorithmVersion", 2),
        ("coordinateSystem", "z-up"),
        ("eyeHeightMeters", 0),
        ("clusters", []),
    ),
)
def test_invalid_or_future_serialized_state_is_rejected(field, value):
    payload = _bake().to_dict()
    payload[field] = value
    with pytest.raises((TypeError, ValueError)):
        PvsBakeData.from_dict(payload)


@pytest.mark.parametrize("value", (None, "abc", "x" * 64, True))
def test_runtime_hash_must_be_a_sha256(value):
    with pytest.raises(ValueError, match="SHA256"):
        _bake().to_runtime_dict(glb_sha256=value)
