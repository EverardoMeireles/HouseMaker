"""The runtime companion binds baked visibility to physical export coordinates."""

# ### Imports ###
from __future__ import annotations

import hashlib
import json

import pytest

from housemaker.glb import export_glb_file
from housemaker.global_scale import GlobalScale
from housemaker.scene_export import (
    build_runtime_scene_manifest,
    write_runtime_scene_manifest,
)
from tests.test_pvs_state import _bake
from tests.test_scene_export import _source_model


# ### Manifest construction ###
def _manifest(**kwargs):
    return build_runtime_scene_manifest(
        glb_name="house.glb",
        glb_bytes=b"glb",
        source_placements={},
        instance_placements=(),
        **kwargs,
    )


def test_no_bake_keeps_the_existing_runtime_manifest_shape():
    assert "visibility" not in _manifest()
    assert "visibility" not in _manifest(visibility=None)


def test_runtime_bake_preserves_all_membership_and_uses_the_final_asset_hash():
    payload = _manifest(visibility=_bake())
    visibility = payload["visibility"]
    assert visibility == _bake().to_runtime_dict(glb_sha256=payload["asset"]["sha256"])
    assert visibility["glbSha256"] == hashlib.sha256(b"glb").hexdigest()


def test_final_physical_visibility_bounds_are_not_scaled_a_second_time():
    payload = _manifest(
        visibility=_bake(), global_scale=GlobalScale(2, 3, 4, linked=False)
    )
    assert payload["visibility"]["worldBounds"] == _bake().to_dict()["worldBounds"]
    assert payload["visibility"]["cells"] == _bake().to_dict()["cells"]
    assert payload["visibility"]["clusters"] == _bake().to_dict()["clusters"]


@pytest.mark.parametrize("invalid", ({}, False, "bad"))
def test_unvalidated_visibility_is_rejected(invalid):
    with pytest.raises(TypeError, match="visibility"):
        _manifest(visibility=invalid)


def test_a_changed_eye_height_requires_a_new_bake():
    with pytest.raises(ValueError, match="bake PVS again"):
        _manifest(visibility=_bake(), first_person_eye_height_meters=1.8)


# ### File writing ###
def test_writer_binds_the_actual_scaled_glb_and_builder_uses_the_same_schema(tmp_path):
    path = export_glb_file(_source_model(), tmp_path / "house.glb")
    scale = GlobalScale(2, 3, 4, linked=False)
    manifest_path = write_runtime_scene_manifest(
        path,
        source_placements={},
        instance_placements=(),
        visibility=_bake(),
        global_scale=scale,
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = build_runtime_scene_manifest(
        glb_name=path.name,
        glb_bytes=path.read_bytes(),
        source_placements={},
        instance_placements=(),
        visibility=_bake(),
        global_scale=scale,
    )

    assert payload == expected
    assert (
        payload["visibility"]["glbSha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert payload["visibility"]["worldBounds"] == _bake().to_dict()["worldBounds"]


@pytest.mark.parametrize(
    "options",
    (
        {"visibility": {}},
        {"visibility": _bake(), "first_person_eye_height_meters": 1.8},
    ),
)
def test_invalid_bake_does_not_change_existing_glb_or_json(tmp_path, options):
    path = export_glb_file(_source_model(), tmp_path / "house.glb")
    manifest_path = write_runtime_scene_manifest(
        path, source_placements={}, instance_placements=()
    )
    original_glb, original_json = path.read_bytes(), manifest_path.read_bytes()

    with pytest.raises((TypeError, ValueError)):
        write_runtime_scene_manifest(
            path,
            source_placements={},
            instance_placements=(),
            global_scale=GlobalScale(2, 3, 4, linked=False),
            **options,
        )
    assert path.read_bytes() == original_glb
    assert manifest_path.read_bytes() == original_json
