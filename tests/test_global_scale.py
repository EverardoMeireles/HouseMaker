# ### Imports ###
from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from housemaker.global_scale import GlobalScale
from housemaker.models import GROUND_LEVEL_INDEX, create_default_levels
from housemaker.project_io import load_project, save_project


# ### Linked and independent scale tests ###
def test_linked_axis_edits_preserve_existing_proportions() -> None:
    scale = GlobalScale(x=2.0, y=3.0, z=0.5)

    enlarged = scale.with_axis("y", 6.0)

    assert enlarged.factors == (4.0, 6.0, 1.0)
    assert enlarged.with_axis("x", 2.0) == scale
    assert scale.factors == (2.0, 3.0, 0.5)


def test_independent_edits_only_change_the_requested_axis() -> None:
    scale = GlobalScale(linked=False)

    changed = scale.with_axis("z", 0.75)

    assert changed.factors == (1.0, 1.0, 0.75)
    assert changed.with_linked(True).factors == changed.factors
    assert changed.with_linked(True).with_axis("x", 2.0).factors == (2.0, 2.0, 1.5)


def test_linked_edit_rejects_overflow_without_changing_any_axis() -> None:
    scale = GlobalScale(x=2.0, y=90.0, z=1.0)

    with pytest.raises(ValueError, match="Global Y multiplier"):
        scale.with_axis("x", 4.0)

    assert scale.factors == (2.0, 90.0, 1.0)
    with pytest.raises(FrozenInstanceError):
        scale.x = 4.0


@pytest.mark.parametrize(
    "value", [0.0, -1.0, 0.009, 101.0, float("inf"), float("nan"), None, True]
)
def test_invalid_factors_are_rejected(value: object) -> None:
    with pytest.raises((ValueError, TypeError), match="Global X multiplier"):
        GlobalScale(x=value)


def test_scene_coordinates_round_trip_without_scaling_source_data() -> None:
    scale = GlobalScale(x=1.25, y=2.0, z=0.5, linked=False)
    authored = [4.0, -7.0, 2.0]

    displayed = scale.apply_point(authored)

    assert displayed == (5.0, -14.0, 1.0)
    assert scale.inverse_point(displayed) == pytest.approx(authored)
    assert authored == [4.0, -7.0, 2.0]
    for row in range(4):
        for column in range(4):
            product = sum(
                scale.matrix[row][index] * scale.inverse_matrix[index][column]
                for index in range(4)
            )
            assert product == pytest.approx(float(row == column))


# ### Project persistence tests ###
def test_project_round_trip_retains_factors_and_original_level_measurements(
    tmp_path,
) -> None:
    levels = create_default_levels()
    level = next(level for level in levels if level.index == GROUND_LEVEL_INDEX)
    level.height_meters = 2.45
    level.floor_thickness_meters = 0.25
    level.canvas_level_scale = 1.25
    vertex = level.vertex_data.add_vertex(120.0, 80.0)
    scale = GlobalScale(x=1.5, y=2.0, z=0.75, linked=False)
    path = tmp_path / "scaled-project.json"

    save_project(path, GROUND_LEVEL_INDEX, levels, global_scale=scale)
    restored = load_project(path)
    restored_level = next(
        level for level in restored.levels if level.index == GROUND_LEVEL_INDEX
    )

    assert restored.global_scale == scale
    assert restored_level.height_meters == level.height_meters
    assert restored_level.floor_thickness_meters == level.floor_thickness_meters
    assert restored_level.canvas_level_scale == level.canvas_level_scale
    assert restored_level.vertex_data.vertices[0] == vertex
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["global_scale"] == scale.to_dict()


def test_legacy_project_without_global_scale_loads_at_identity(tmp_path) -> None:
    path = tmp_path / "legacy-project.json"
    save_project(path, GROUND_LEVEL_INDEX, create_default_levels())
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["global_scale"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_project(path).global_scale == GlobalScale()


@pytest.mark.parametrize(
    "payload",
    [[], {"z": -2.0}, {"x": "invalid"}, {"y": float("nan")}, {"linked": "false"}],
)
def test_project_rejects_corrupt_global_scale_instead_of_silently_resizing(
    tmp_path, payload
) -> None:
    path = tmp_path / "invalid-project.json"
    save_project(path, GROUND_LEVEL_INDEX, create_default_levels())
    project_payload = json.loads(path.read_text(encoding="utf-8"))
    project_payload["global_scale"] = payload
    path.write_text(json.dumps(project_payload), encoding="utf-8")

    with pytest.raises(
        (ValueError, TypeError), match="Global multiplier|Global [XYZ] multiplier"
    ):
        load_project(path)
