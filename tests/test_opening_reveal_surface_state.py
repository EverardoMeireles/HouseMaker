"""Independent opening-reveal material identity, selection and persistence."""

# ### Imports ###
import json
from urllib.parse import quote

import pytest

from housemaker.surface_texture_state import (
    SURFACE_TYPE_CEILING,
    SURFACE_TYPE_FLOOR,
    SURFACE_TYPE_WALL,
    SurfaceTextureAssignment,
    SurfaceTextureData,
    get_surface_type_for_id,
    normalize_surface_id,
    surface_id_accepts_texture_type,
)


# ### Semantic identity and material compatibility ###
def _surface_id(kind="doorway", opening_id="opening-1", part="left"):
    return f"level:2/opening:{kind}:{quote(opening_id, safe='-._~')}/reveal:{part}"


@pytest.mark.parametrize("kind", ("doorway", "window"))
@pytest.mark.parametrize("part", ("left", "right", "header", "sill"))
def test_reveal_parts_are_independent_wall_material_targets(kind, part):
    surface_id = _surface_id(kind, part=part)
    assert normalize_surface_id(surface_id) == surface_id
    assert get_surface_type_for_id(surface_id) == SURFACE_TYPE_WALL
    assert surface_id_accepts_texture_type(surface_id, SURFACE_TYPE_WALL)
    assert not surface_id_accepts_texture_type(surface_id, SURFACE_TYPE_FLOOR)
    assert not surface_id_accepts_texture_type(surface_id, SURFACE_TYPE_CEILING)
    selection = SurfaceTextureData(selected_surface_ids=(surface_id,))
    assert selection.selected_surface_type == SURFACE_TYPE_WALL


@pytest.mark.parametrize(
    "opening_id",
    ("window-1", "doorway-1", "a" * 32, "Window A/left:é", "name%with~punctuation"),
)
def test_reveal_identity_accepts_encoded_existing_opening_identifiers(opening_id):
    surface_id = _surface_id(opening_id=opening_id)
    assert normalize_surface_id(surface_id) == surface_id
    assert get_surface_type_for_id(surface_id) == SURFACE_TYPE_WALL


@pytest.mark.parametrize(
    "surface_id",
    (
        "level:2/opening:door:one/reveal:left",
        "level:-1/opening:window:one/reveal:left",
        "level:2/opening:window:/reveal:left",
        "level:2/opening:window:one/reveal:outside",
        "level:2/opening:window:one/reveal:left:ceiling",
        "level:2/opening:window:window one/reveal:left",
        "level:2/opening:window:one/other/reveal:left",
        "level:2/opening:window:one%ZZ/reveal:left",
        "level:2/opening:window:one%/reveal:left",
        "level:2/opening:window:one%00/reveal:left",
        "level:2/opening:window:%FF/reveal:left",
        "level:2/opening:window:%20one/reveal:left",
        _surface_id(opening_id="x" * 129),
    ),
)
def test_reveal_identity_rejects_malformed_or_invalid_opening_ids(surface_id):
    with pytest.raises(ValueError):
        normalize_surface_id(surface_id)


# ### Texture assignment round-trips without merging the owning wall ###
def test_reveal_assignment_and_selection_roundtrip_without_owner_wall_or_other_parts():
    left = _surface_id(part="left")
    right = _surface_id(part="right")
    header = _surface_id(part="header")
    owner_wall = "level:2/room:0/wall:1:2"
    assignment = SurfaceTextureAssignment(
        assignment_id="reveal-plaster",
        surface_type=SURFACE_TYPE_WALL,
        surface_ids=(left, right),
        provider="openai",
        asset_path="surface_textures/reveal-plaster.png",
        combined_area_m2=0.8,
        texture_width=1024,
        texture_height=1024,
    )
    state = SurfaceTextureData(
        selected_surface_ids=(left, right),
        assignments=[assignment],
    )
    restored = SurfaceTextureData.from_dict(json.loads(json.dumps(state.to_dict())))
    assert restored.selected_surface_ids == (left, right)
    assert restored.selected_surface_type == SURFACE_TYPE_WALL
    assert restored.assignments == [assignment]
    assert restored.assignments_for_surface(left) == [assignment]
    assert restored.assignments_for_surface(right) == [assignment]
    assert restored.assignments_for_surface(header) == []
    assert restored.assignments_for_surface(owner_wall) == []
    restored.set_selection(SURFACE_TYPE_WALL, (header,))
    assert restored.selected_surface_ids == (header,)
    assert restored.assignments == [assignment]
