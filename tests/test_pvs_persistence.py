"""Optional project visibility never prevents restoring the editable project."""

# ### Imports ###
from __future__ import annotations

import json

import pytest

from housemaker.models import GROUND_LEVEL_INDEX, create_default_levels
from housemaker.project_io import ProjectData, load_project, save_project
from tests.test_pvs_state import _bake


# ### Valid and legacy projects ###
def test_bake_round_trips_alongside_existing_project_fields(tmp_path):
    levels = create_default_levels()
    levels[GROUND_LEVEL_INDEX].name = "Preserved level"
    path = save_project(
        tmp_path / "project.json", GROUND_LEVEL_INDEX, levels, visibility=_bake()
    )

    loaded = load_project(path)

    assert loaded.visibility == _bake()
    assert loaded.visibility_warning is None
    assert loaded.current_level_index == GROUND_LEVEL_INDEX
    assert loaded.levels[GROUND_LEVEL_INDEX].name == "Preserved level"
    assert (
        json.loads(path.read_text(encoding="utf-8"))["visibility"] == _bake().to_dict()
    )


@pytest.mark.parametrize("missing", (True, False))
def test_old_project_missing_or_null_bake_remains_supported(tmp_path, missing):
    path = save_project(
        tmp_path / "project.json", GROUND_LEVEL_INDEX, create_default_levels()
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if missing:
        payload.pop("visibility")
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_project(path)
    assert loaded.visibility is None
    assert loaded.visibility_warning is None


# ### Fail-safe malformed bake handling ###
@pytest.mark.parametrize("invalid", ({}, [], "invalid", {"version": 999}))
def test_invalid_optional_bake_is_disabled_without_discarding_the_project(
    tmp_path, invalid
):
    path = save_project(
        tmp_path / "project.json", GROUND_LEVEL_INDEX, create_default_levels()
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["visibility"] = invalid
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_project(path)
    assert loaded.visibility is None
    assert "disabled" in loaded.visibility_warning
    assert len(loaded.levels) == len(create_default_levels())


def test_invalid_visibility_cannot_overwrite_a_valid_project(tmp_path):
    path = save_project(
        tmp_path / "project.json",
        GROUND_LEVEL_INDEX,
        create_default_levels(),
        visibility=_bake(),
    )
    original = path.read_bytes()

    with pytest.raises(TypeError, match="visibility"):
        save_project(path, GROUND_LEVEL_INDEX, create_default_levels(), visibility={})
    with pytest.raises(TypeError, match="visibility"):
        ProjectData(None, GROUND_LEVEL_INDEX, create_default_levels(), visibility={})
    assert path.read_bytes() == original
