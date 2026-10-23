# ### Imports ###
from __future__ import annotations

import json

import pytest

from housemaker.texture_atlas_state import (
    ATLAS_NAME_PREFIX_BY_TYPE,
    ATLAS_PACKING_MODE_SYMMETRIC_HALF,
    ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR,
    ATLAS_STATE_SCHEMA_VERSION,
    ATLAS_TYPE_OBJECT,
    ATLAS_TYPE_SURFACE,
    TextureAtlasData,
    TextureAtlasRecord,
    normalize_atlas_name,
    normalize_atlas_type,
)


# ### Type and name normalization ###
@pytest.mark.parametrize(
    ("value", "expected"),
    ((None, None), ("object", "object"), (" OBJECT ", "object"), (" Surface ", "surface")),
)
def test_atlas_type_normalization(value, expected):
    assert normalize_atlas_type(value) == expected


@pytest.mark.parametrize("value", ("", "half", "objects", "floor", False, 7, [], {}))
def test_invalid_atlas_type_is_rejected(value):
    with pytest.raises(ValueError, match="atlas type"):
        normalize_atlas_type(value)


@pytest.mark.parametrize(
    ("name", "atlas_type", "expected"),
    (
        (" Furniture ", ATLAS_TYPE_OBJECT, "[OBJECT] Furniture"),
        ("[object] Furniture", ATLAS_TYPE_OBJECT, "[OBJECT] Furniture"),
        ("[SURFACE] Furniture", ATLAS_TYPE_OBJECT, "[OBJECT] Furniture"),
        ("[OBJECT] [surface] Walls", ATLAS_TYPE_SURFACE, "[SURFACE] Walls"),
        ("[HALF] Custom suffix", ATLAS_TYPE_OBJECT, "[OBJECT] Custom suffix"),
        ("[OBJECT] [HALF] Custom suffix", ATLAS_TYPE_OBJECT, "[OBJECT] Custom suffix"),
        ("[Walls] Custom suffix", ATLAS_TYPE_SURFACE, "[SURFACE] [Walls] Custom suffix"),
        ("[HALF] Legacy", None, "[HALF] Legacy"),
        ("[OBJECT] Legacy", None, "[OBJECT] Legacy"),
    ),
)
def test_explicit_type_normalizes_only_leading_reserved_name_tags(name, atlas_type, expected):
    assert normalize_atlas_name(name, atlas_type) == expected


@pytest.mark.parametrize("name", ("", "   ", "[OBJECT]", "[HALF] [SURFACE]"))
def test_typed_atlas_still_needs_a_nonempty_user_name(name):
    with pytest.raises(ValueError, match="name"):
        normalize_atlas_name(name, ATLAS_TYPE_OBJECT)


# ### Record and collection persistence ###
def test_explicit_type_is_appended_without_breaking_positional_record_arguments():
    legacy = TextureAtlasRecord("legacy", "[HALF] Custom", 2048, [], None, None, None, 0.5)
    typed = TextureAtlasRecord(
        "typed", "Custom", 2048, [], None, None, None, 0.5, ATLAS_TYPE_SURFACE
    )

    assert legacy.atlas_type is None
    assert legacy.name == "[HALF] Custom"
    assert typed.atlas_type == ATLAS_TYPE_SURFACE
    assert typed.name == "[SURFACE] Custom"
    assert typed.surface_ao_intensity == 0.5


def test_typed_creation_clone_and_json_round_trip_preserve_metadata():
    data = TextureAtlasData()
    objects = data.create_atlas("Furniture", 2048, atlas_id="objects", atlas_type=" OBJECT ")
    surfaces = data.create_atlas("Plaster", 4096, atlas_id="surfaces", atlas_type=ATLAS_TYPE_SURFACE)
    data.assign_object(objects.atlas_id, "chair", "textures/chair.png", 512)
    data.assign_object(surfaces.atlas_id, "wall-texture", "textures/wall.png", 1024)
    objects.image_path = "atlases/objects.png"
    surfaces.image_path = "atlases/surfaces.png"
    surfaces.set_surface_ambient_occlusion("atlases/surfaces-ao.png", "geometry-v1")
    data.select_atlas(surfaces.atlas_id)

    for restored in (data.clone(), TextureAtlasData.from_dict(json.loads(json.dumps(data.to_dict())))):
        assert restored == data
        assert restored.atlases[0] is not objects
        assert restored.atlases[0].placements is not objects.placements
        assert restored.selected_atlas_id == surfaces.atlas_id
        assert [atlas.atlas_type for atlas in restored.atlases] == ["object", "surface"]
        assert [atlas.name for atlas in restored.atlases] == ["[OBJECT] Furniture", "[SURFACE] Plaster"]
    assert ATLAS_NAME_PREFIX_BY_TYPE == {"object": "[OBJECT]", "surface": "[SURFACE]"}
    assert data.to_dict()["schema_version"] == ATLAS_STATE_SCHEMA_VERSION == 8


@pytest.mark.parametrize("schema", range(1, 9))
@pytest.mark.parametrize("name", ("Custom legacy", "[HALF] My furniture", "[OBJECT] Unclassified"))
def test_legacy_names_are_not_guessed_or_rewritten_before_catalog_migration(schema, name):
    payload = {
        "schema_version": schema,
        "selected_atlas_id": "legacy",
        "atlases": [{"atlas_id": "legacy", "name": name, "resolution": 2048, "placements": []}],
    }

    restored = TextureAtlasData.from_dict(payload)

    assert restored.atlases[0].name == name
    assert restored.atlases[0].atlas_type is None
    assert restored.clone() == restored


@pytest.mark.parametrize("invalid_type", ("unknown", "", True, {"type": "object"}))
def test_invalid_persisted_type_is_rejected(invalid_type):
    payload = TextureAtlasRecord("atlas", "Legacy", 2048).to_dict()
    payload["atlas_type"] = invalid_type

    with pytest.raises(ValueError, match="atlas type"):
        TextureAtlasRecord.from_dict(payload)


# ### Renaming and legacy classification ###
def test_renaming_cannot_change_a_typed_atlas_category_or_lose_baked_assets():
    data = TextureAtlasData()
    atlas = data.create_atlas("Original", 2048, atlas_id="atlas", atlas_type=ATLAS_TYPE_OBJECT)
    atlas.image_path = "atlases/atlas.png"
    atlas.set_surface_ambient_occlusion("atlases/atlas-ao.png", "geometry-v1")

    assert data.rename_atlas(atlas.atlas_id, "[SURFACE] Renamed") is atlas

    assert atlas.atlas_type == ATLAS_TYPE_OBJECT
    assert atlas.name == "[OBJECT] Renamed"
    assert atlas.image_path == "atlases/atlas.png"
    assert atlas.surface_ao_image_path == "atlases/atlas-ao.png"
    assert atlas.surface_ao_geometry_signature == "geometry-v1"


def test_duplicate_names_are_checked_after_type_prefix_normalization():
    data = TextureAtlasData()
    objects = data.create_atlas("Furniture", 2048, atlas_id="objects", atlas_type=ATLAS_TYPE_OBJECT)
    surfaces = data.create_atlas("Furniture", 2048, atlas_id="surfaces", atlas_type=ATLAS_TYPE_SURFACE)
    other = data.create_atlas("Other", 2048, atlas_id="other", atlas_type=ATLAS_TYPE_OBJECT)

    with pytest.raises(ValueError, match="already exists"):
        data.create_atlas("[SURFACE] Furniture", 2048, atlas_type=ATLAS_TYPE_OBJECT)
    with pytest.raises(ValueError, match="already exists"):
        data.rename_atlas(other.atlas_id, "[SURFACE] Furniture")
    assert other.name == "[OBJECT] Other"
    assert objects.name != surfaces.name
    assert len(data.atlases) == 3


@pytest.mark.parametrize(
    "packing_mode", (ATLAS_PACKING_MODE_SYMMETRIC_HALF, ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR)
)
def test_classifying_legacy_half_atlas_does_not_change_real_half_placements(packing_mode):
    data = TextureAtlasData()
    atlas = data.create_atlas("[HALF] My furniture", 2048, atlas_id="legacy")
    for source_id in ("left", "right"):
        data.assign_object(atlas.atlas_id, source_id, f"textures/{source_id}.png", 512, packing_mode)
    placements = tuple(atlas.placements)
    atlas.image_path = "atlases/legacy.png"
    atlas.set_surface_ambient_occlusion("atlases/legacy-ao.png", "geometry-v1")

    assert data.set_atlas_type(atlas.atlas_id, ATLAS_TYPE_OBJECT) is atlas

    assert atlas.name == "[OBJECT] My furniture"
    assert atlas.atlas_type == ATLAS_TYPE_OBJECT
    assert tuple(atlas.placements) == placements
    assert atlas.placements[0].x == atlas.placements[1].x
    assert atlas.placements[0].y == atlas.placements[1].y
    assert atlas.placements[0].slot_half != atlas.placements[1].slot_half
    assert atlas.image_path == "atlases/legacy.png"
    assert atlas.surface_ao_image_path == "atlases/legacy-ao.png"
    assert data.clone() == data


def test_failed_legacy_classification_is_atomic():
    data = TextureAtlasData()
    legacy = data.create_atlas("Furniture", 2048, atlas_id="legacy")
    data.create_atlas("Furniture", 2048, atlas_type=ATLAS_TYPE_OBJECT)

    with pytest.raises(ValueError, match="already exists"):
        data.set_atlas_type(legacy.atlas_id, ATLAS_TYPE_OBJECT)

    assert legacy.name == "Furniture"
    assert legacy.atlas_type is None


def test_state_layer_does_not_guess_source_category_from_an_id():
    data = TextureAtlasData()
    atlas = data.create_atlas("Furniture", 2048, atlas_type=ATLAS_TYPE_OBJECT)

    # The source catalog, not an ID prefix or packing shape, owns routing.
    data.assign_object(atlas.atlas_id, "surface-wall-texture:legacy", "textures/wall.png", 512)

    assert atlas.atlas_type == ATLAS_TYPE_OBJECT
    assert len(atlas.placements) == 1
