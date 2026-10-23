"""R3F exports include a validated, physical first-person camera eye height."""

# ### Imports ###
from __future__ import annotations

import json

import pytest

from housemaker.first_person_navigation import DEFAULT_FIRST_PERSON_EYE_HEIGHT_METERS
from housemaker.glb import export_glb_file
from housemaker.global_scale import GlobalScale
from housemaker.scene_export import (
    RUNTIME_SCALED_SCENE_VERSION,
    RUNTIME_SCENE_VERSION,
    build_runtime_scene_manifest,
    write_runtime_scene_manifest,
)
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


def test_default_navigation_metadata_is_additive_and_uses_existing_schema_conventions():
    manifest = _manifest()

    assert manifest["firstPersonCamera"] == {
        "eyeHeightMeters": DEFAULT_FIRST_PERSON_EYE_HEIGHT_METERS,
    }
    assert manifest["firstPersonCamera"]["eyeHeightMeters"] == 1.70
    assert manifest["version"] == RUNTIME_SCENE_VERSION == 9
    assert manifest["coordinateSystem"] == "gltf-y-up"
    assert manifest["matrixLayout"] == "column-major"
    assert manifest["asset"]["glb"] == "house.glb"
    assert manifest["instanceGroups"] == []


@pytest.mark.parametrize(
    "value, expected", ((1.63, 1.63), (1.678, 1.68), (2, 2.0), (0.1, 0.1), (10.0, 10.0))
)
def test_custom_eye_height_uses_the_shared_navigation_normalizer(value, expected):
    manifest = _manifest(first_person_eye_height_meters=value)

    assert manifest["firstPersonCamera"] == {"eyeHeightMeters": expected}
    assert isinstance(manifest["firstPersonCamera"]["eyeHeightMeters"], float)


@pytest.mark.parametrize("value", (True, False, None, "1.7", [], {}))
def test_non_numeric_eye_height_is_rejected(value):
    with pytest.raises(TypeError, match="eye height"):
        _manifest(first_person_eye_height_meters=value)


@pytest.mark.parametrize(
    "value", (float("nan"), float("inf"), float("-inf"), 0, -1, 0.09, 10.01)
)
def test_nonfinite_or_out_of_range_eye_height_is_rejected(value):
    with pytest.raises(ValueError, match="eye height"):
        _manifest(first_person_eye_height_meters=value)


def test_global_house_scale_does_not_scale_the_physical_camera_height():
    manifest = _manifest(
        first_person_eye_height_meters=1.63,
        global_scale=GlobalScale(2.0, 3.0, 4.0, linked=False),
    )

    assert manifest["version"] == RUNTIME_SCALED_SCENE_VERSION == 10
    assert manifest["globalScale"]["appliedToRuntimeData"] is True
    assert manifest["firstPersonCamera"] == {"eyeHeightMeters": 1.63}


# ### Companion JSON writing and rejected-input safety ###
@pytest.mark.parametrize("eye_height", (None, 1.63))
def test_writer_and_builder_emit_the_same_camera_schema(tmp_path, eye_height):
    path = export_glb_file(_source_model(), tmp_path / "house.glb")
    original_glb = path.read_bytes()
    options = (
        {} if eye_height is None else {"first_person_eye_height_meters": eye_height}
    )

    manifest_path = write_runtime_scene_manifest(
        path,
        source_placements={},
        instance_placements=(),
        **options,
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = build_runtime_scene_manifest(
        glb_name=path.name,
        glb_bytes=path.read_bytes(),
        source_placements={},
        instance_placements=(),
        **options,
    )
    assert payload == expected
    assert payload["firstPersonCamera"] == {"eyeHeightMeters": eye_height or 1.70}
    assert path.read_bytes() == original_glb


@pytest.mark.parametrize("eye_height", (True, float("nan"), 0.0))
def test_invalid_writer_eye_height_leaves_existing_export_files_unchanged(
    tmp_path, eye_height
):
    path = export_glb_file(_source_model(), tmp_path / "house.glb")
    manifest_path = write_runtime_scene_manifest(
        path,
        source_placements={},
        instance_placements=(),
    )
    original_glb = path.read_bytes()
    original_manifest = manifest_path.read_bytes()

    with pytest.raises((TypeError, ValueError), match="eye height"):
        write_runtime_scene_manifest(
            path,
            source_placements={},
            instance_placements=(),
            first_person_eye_height_meters=eye_height,
            global_scale=GlobalScale(2.0, 3.0, 4.0, linked=False),
        )

    assert path.read_bytes() == original_glb
    assert manifest_path.read_bytes() == original_manifest
