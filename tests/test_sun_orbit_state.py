"""Sun orbit is optional authored configuration, never accumulated light edits."""

# ### Imports ###
from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from housemaker.directional_light_state import (
    SunOrbitData,
    create_default_sun_light,
    rotate_sun_light_about_z,
)
from housemaker.models import GROUND_LEVEL_INDEX, create_default_levels
from housemaker.project_io import ProjectData, load_project, save_project


# ### Immutable settings and angled default ###
def test_default_sun_is_global_angled_downward_and_uses_compatible_identity():
    sun = create_default_sun_light()
    assert sun.light_id == "directional-light-1"
    assert sun.name == "Sunlight"
    assert sun.position == (0, 0, 3)
    assert sun.target == (1, 0, 2)
    assert sun.level_index is None
    assert sun.target[2] < sun.position[2]
    assert rotate_sun_light_about_z(sun, 90).target != sun.target


def test_default_direction_is_minus_45_degrees_about_authoring_y_and_can_be_positioned():
    sun = create_default_sun_light(position=(10, -5, 12))
    assert sun.position == (10, -5, 12)
    assert sun.target == (11, -5, 11)
    offset = np.subtract(sun.target, sun.position)
    expected = np.array((math.sin(math.pi / 4), 0, -math.cos(math.pi / 4)))
    np.testing.assert_allclose(offset / np.linalg.norm(offset), expected)


@pytest.mark.parametrize("position", (None, (0, 0), (0, 0, float("nan"))))
def test_default_sun_rejects_invalid_position(position):
    with pytest.raises(ValueError):
        create_default_sun_light(position=position)


def test_settings_default_on_at_half_degree_per_second_validate_and_round_trip():
    assert SunOrbitData() == SunOrbitData(True, 0.5)
    assert SunOrbitData.from_dict(None) == SunOrbitData()
    assert SunOrbitData.from_dict({}) == SunOrbitData()
    assert SunOrbitData.from_dict({"enabled": False}) == SunOrbitData(False, 0.5)
    assert SunOrbitData.from_dict({"speed_degrees_per_second": 2}) == SunOrbitData(
        True, 2
    )
    config = SunOrbitData(True, 12.5)
    assert SunOrbitData.from_dict(config.to_dict()) == config
    assert config.to_runtime_dict() == {"enabled": True, "speedDegreesPerSecond": 12.5}
    with pytest.raises(FrozenInstanceError):
        config.enabled = False


@pytest.mark.parametrize("value", (1, 0, "true", None, []))
def test_enabled_must_be_a_real_boolean(value):
    with pytest.raises(TypeError):
        SunOrbitData(value)


@pytest.mark.parametrize("speed", (True, None, "5", [], {}))
def test_speed_rejects_non_numeric_values(speed):
    with pytest.raises(TypeError):
        SunOrbitData(speed_degrees_per_second=speed)


@pytest.mark.parametrize("speed", (-0.1, 360.1, float("nan"), float("inf")))
def test_speed_rejects_nonfinite_or_unsupported_values(speed):
    with pytest.raises(ValueError):
        SunOrbitData(speed_degrees_per_second=speed)


@pytest.mark.parametrize("speed", (0, 0.25, 5, 360))
def test_zero_and_finite_supported_speeds_are_preserved(speed):
    assert SunOrbitData(True, speed).speed_degrees_per_second == speed


# ### Rotation preserves authored state and appearance ###
@pytest.mark.parametrize("degrees", (0, 45, 90, -90, 360, 720, 1e100))
def test_rotation_changes_only_direction_around_z_without_accumulating(degrees):
    sun = replace(
        create_default_sun_light(), color="#abc123", intensity=2.5, cast_shadow=True
    )
    original_target = sun.target
    moved = rotate_sun_light_about_z(sun, degrees)
    assert sun.target == original_target
    assert moved.position == sun.position
    assert moved.target[2] - moved.position[2] == sun.target[2] - sun.position[2]
    assert math.dist(moved.position, moved.target) == pytest.approx(
        math.dist(sun.position, sun.target)
    )
    assert moved.color == sun.color
    assert moved.intensity == sun.intensity
    assert moved.cast_shadow == sun.cast_shadow
    assert moved.light_id == sun.light_id


def test_positive_quarter_rotation_and_inverse_return_to_initial_direction():
    sun = create_default_sun_light()
    rotated = rotate_sun_light_about_z(sun, 90)
    np.testing.assert_allclose(rotated.target, (0, 1, 2), atol=1e-12)
    np.testing.assert_allclose(
        rotate_sun_light_about_z(rotated, -90).target, sun.target
    )


@pytest.mark.parametrize("degrees", (True, "90", None, float("nan"), float("inf")))
def test_rotation_rejects_invalid_angles(degrees):
    with pytest.raises((TypeError, ValueError)):
        rotate_sun_light_about_z(create_default_sun_light(), degrees)


# ### Project persistence and legacy safety ###
@pytest.mark.parametrize("enabled", (False, True))
def test_project_round_trip_preserves_orbit_and_initial_vectors(tmp_path, enabled):
    light = replace(create_default_sun_light(), position=(4, -2, 15), target=(-1, 3, 1))
    orbit = SunOrbitData(enabled, 15.5)
    path = save_project(
        tmp_path / "house.json",
        GROUND_LEVEL_INDEX,
        create_default_levels(),
        directional_lights=(light,),
        sun_orbit=orbit,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["sun_orbit"] == orbit.to_dict()
    loaded = load_project(path)
    assert loaded.sun_orbit == orbit
    assert loaded.directional_lights == (light,)


@pytest.mark.parametrize("missing", (True, False))
def test_legacy_project_uses_new_orbit_defaults_without_inserting_a_light(
    tmp_path, missing
):
    path = save_project(
        tmp_path / "house.json", GROUND_LEVEL_INDEX, create_default_levels()
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if missing:
        payload.pop("sun_orbit")
    else:
        payload["sun_orbit"] = None
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_project(path)
    assert loaded.sun_orbit == SunOrbitData()
    assert loaded.directional_lights == ()


def test_invalid_optional_saved_orbit_disables_animation_instead_of_losing_the_project(
    tmp_path,
):
    path = save_project(
        tmp_path / "house.json", GROUND_LEVEL_INDEX, create_default_levels()
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["sun_orbit"] = {"enabled": True, "speed_degrees_per_second": -20}
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_project(path).sun_orbit == SunOrbitData(enabled=False)


def test_unvalidated_settings_cannot_overwrite_a_project(tmp_path):
    path = save_project(
        tmp_path / "house.json", GROUND_LEVEL_INDEX, create_default_levels()
    )
    original = path.read_bytes()
    with pytest.raises(TypeError, match="sun orbit"):
        save_project(path, GROUND_LEVEL_INDEX, create_default_levels(), sun_orbit={})
    with pytest.raises(TypeError, match="sun orbit"):
        ProjectData(None, GROUND_LEVEL_INDEX, create_default_levels(), sun_orbit={})
    assert path.read_bytes() == original
