"""Runtime orbit metadata retains initial light data and physical directions."""

# ### Imports ###
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from housemaker.directional_light_state import SunOrbitData, create_default_sun_light
from housemaker.glb import export_glb_file
from housemaker.global_scale import GlobalScale
from housemaker.scene_export import (
    build_runtime_scene_manifest,
    write_runtime_scene_manifest,
)
from tests.test_scene_export import _source_model


# ### Runtime metadata ###
def _manifest(**kwargs):
    return build_runtime_scene_manifest(
        glb_name="house.glb",
        glb_bytes=b"glb",
        source_placements={},
        instance_placements=(),
        **kwargs,
    )


def test_default_orbit_is_enabled_at_half_degree_and_empty_scene_has_no_target_light():
    assert _manifest()["sunlightOrbit"] == {
        "enabled": True,
        "speedDegreesPerSecond": 0.5,
        "axis": [0, 1, 0],
        "lightId": None,
        "initialDirection": None,
    }


def test_runtime_orbit_targets_only_the_first_light_and_preserves_authored_vectors():
    sun = create_default_sun_light()
    second = replace(sun, light_id="fill-light", name="Fill", target=(-1, 0, 0))
    manifest = _manifest(
        directional_lights=(sun, second), sun_orbit=SunOrbitData(True, 12)
    )
    orbit = manifest["sunlightOrbit"]
    assert orbit["enabled"] is True
    assert orbit["speedDegreesPerSecond"] == 12
    assert orbit["lightId"] == sun.light_id
    np.testing.assert_allclose(
        orbit["initialDirection"], np.array((1, -1, 0)) / np.sqrt(2)
    )
    assert manifest["directionalLights"][0]["position"] == [0, 3, 0]
    assert manifest["directionalLights"][0]["target"] == [1, 2, 0]
    assert sun.target == (1, 0, 2)


def test_initial_direction_uses_final_glTF_y_up_nonuniform_physical_scale():
    manifest = _manifest(
        directional_lights=(create_default_sun_light(),),
        sun_orbit=SunOrbitData(True),
        global_scale=GlobalScale(2, 3, 4),
    )
    np.testing.assert_allclose(
        manifest["sunlightOrbit"]["initialDirection"],
        np.array((2, -4, 0)) / np.sqrt(20),
    )
    assert manifest["sunlightOrbit"]["axis"] == [0, 1, 0]
    assert manifest["sunlightOrbit"]["speedDegreesPerSecond"] == 0.5


def test_explicit_disabled_orbit_and_custom_speed_are_not_replaced_by_new_defaults():
    orbit = _manifest(
        directional_lights=(create_default_sun_light(),),
        sun_orbit=SunOrbitData(False, 7.25),
    )["sunlightOrbit"]
    assert orbit["enabled"] is False
    assert orbit["speedDegreesPerSecond"] == 7.25


def test_distant_finite_light_points_still_produce_a_finite_unit_direction():
    sun = replace(
        create_default_sun_light(), position=(1e308, 0, 0), target=(-1e308, 0, 0)
    )
    assert _manifest(directional_lights=(sun,))["sunlightOrbit"][
        "initialDirection"
    ] == [-1, 0, 0]


# ### Writer consistency and rejected inputs ###
def test_writer_and_builder_share_metadata_without_mutating_the_light(tmp_path):
    sun, orbit = create_default_sun_light(), SunOrbitData(True, 7.5)
    path = export_glb_file(_source_model(), tmp_path / "house.glb")
    manifest_path = write_runtime_scene_manifest(
        path,
        source_placements={},
        instance_placements=(),
        directional_lights=(sun,),
        sun_orbit=orbit,
    )
    expected = build_runtime_scene_manifest(
        glb_name=path.name,
        glb_bytes=path.read_bytes(),
        source_placements={},
        instance_placements=(),
        directional_lights=(sun,),
        sun_orbit=orbit,
    )
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == expected
    assert sun == create_default_sun_light()


def test_invalid_orbit_config_cannot_replace_existing_export_files(tmp_path):
    path = export_glb_file(_source_model(), tmp_path / "house.glb")
    manifest_path = write_runtime_scene_manifest(
        path, source_placements={}, instance_placements=()
    )
    original_glb, original_json = path.read_bytes(), manifest_path.read_bytes()
    with pytest.raises(TypeError, match="sun orbit"):
        write_runtime_scene_manifest(
            path, source_placements={}, instance_placements=(), sun_orbit={}
        )
    assert path.read_bytes() == original_glb
    assert manifest_path.read_bytes() == original_json
