# ### Environment ###
from __future__ import annotations

import os
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ### Imports ###
import numpy as np
import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication

from housemaker.texture_atlas_state import (
    ATLAS_DOUBLE_SIZED_PACKING_MODES,
    ATLAS_PACKING_MODE_FULL,
    ATLAS_PACKING_MODE_SYMMETRIC_HALF,
    ATLAS_PACKING_MODE_SYMMETRIC_PAIR,
    ATLAS_PACKING_MODE_SYMMETRIC_QUARTER,
    ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR,
    ATLAS_SLOT_HALF_LEFT,
    ATLAS_TYPE_OBJECT,
    ATLAS_TYPE_SURFACE,
    TextureAtlasData,
    atlas_placement_pixel_regions,
    normalize_atlas_partial_slots,
    write_texture_atlas_png,
)
from housemaker.texture_atlas_workspace import (
    AtlasObjectTextureSource,
    AtlasSurfaceTextureEntry,
    TextureAtlasWorkspace,
    build_atlas_wall_texture_source_id,
)

# ### Fixtures and raster helpers ###
_application = QApplication.instance() or QApplication([])


@pytest.fixture
def workspace(tmp_path):
    widget = TextureAtlasWorkspace(asset_directory=tmp_path)
    try:
        yield widget
    finally:
        widget.close()
        widget.deleteLater()
        _application.processEvents()


def _source(
    tmp_path, index, *, surface, packing_mode=ATLAS_PACKING_MODE_FULL, resolution=512
):
    source_id = (
        build_atlas_wall_texture_source_id(f"surface-{index}")
        if surface else f"object-{index}"
    )
    color = (
        (220, 30, 40, 255), (35, 80, 220, 255),
        (40, 190, 70, 255), (230, 150, 40, 255),
    )[index]
    physical_size = resolution * (2 if packing_mode in ATLAS_DOUBLE_SIZED_PACKING_MODES else 1)
    path = tmp_path / f"source-{index}-{resolution}.png"
    Image.new("RGBA", (physical_size, physical_size), color).save(path)
    preview = np.full((8, 8, 4), color, dtype=np.uint8)
    mirrored = packing_mode not in {ATLAS_PACKING_MODE_FULL, ATLAS_PACKING_MODE_SYMMETRIC_HALF}
    return AtlasObjectTextureSource(
        object_id=source_id,
        object_name=f"Source {index}",
        texture_path=path.name,
        texture_resolution=resolution,
        physical_texture_path=path,
        preview_rgba=preview,
        packing_mode=packing_mode,
        symmetric_preview_orientation="vertical" if mirrored else None,
        symmetric_preview_plane_coordinate=0.0 if mirrored else None,
    ), np.asarray(color, dtype=np.uint8)


def _mixed_legacy_data(tmp_path, packing_mode, *, resolution=512):
    count = 4 if packing_mode == ATLAS_PACKING_MODE_SYMMETRIC_QUARTER else 2
    source_colors = tuple(
        _source(
            tmp_path, index, surface=bool(index % 2),
            packing_mode=packing_mode, resolution=resolution,
        )
        for index in range(count)
    )
    sources = tuple(source for source, _color in source_colors)
    colors = {source.object_id: color for source, color in source_colors}
    data = TextureAtlasData()
    atlas = data.create_atlas("[HALF] Custom furniture", 2048, atlas_id="legacy")
    for source in sources:
        data.assign_object(
            atlas.atlas_id, source.object_id, source.texture_path,
            source.texture_resolution, source.packing_mode,
        )
    by_id = {source.object_id: source for source in sources}
    old_path = tmp_path / "original-atlas.png"
    write_texture_atlas_png(
        atlas, old_path,
        source_loader=lambda placement: by_id[placement.object_id].load_texture_rgba()[:, :, (2, 1, 0, 3)],
        project_relative_image_path=old_path.name,
    )
    Image.new("RGBA", (2048, 2048), (128, 128, 128, 255)).save(tmp_path / "original-ao.png")
    atlas.set_surface_ambient_occlusion("original-ao.png", "original-geometry")
    return data, sources, colors


def _pixel_snapshot(tmp_path):
    return {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*.png")}


def _assert_placement_color(tmp_path, atlas, placement, color):
    assert atlas.image_path is not None
    with Image.open(tmp_path / atlas.image_path) as image:
        rgba = np.asarray(image.convert("RGBA"))
    _outer, inner = atlas_placement_pixel_regions(placement)
    left, top, right, bottom = inner
    np.testing.assert_array_equal(rgba[(top + bottom) // 2, (left + right) // 2], color)


# ### Safe legacy partitioning ###
@pytest.mark.parametrize(
    "packing_mode",
    (
        ATLAS_PACKING_MODE_SYMMETRIC_HALF,
        ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR,
        ATLAS_PACKING_MODE_SYMMETRIC_PAIR,
        ATLAS_PACKING_MODE_SYMMETRIC_QUARTER,
    ),
)
@pytest.mark.parametrize("resolution", (512, 1024))
def test_cross_category_shared_slots_split_without_pixel_or_resolution_loss(
    workspace, tmp_path, packing_mode, resolution
):
    data, sources, colors = _mixed_legacy_data(tmp_path, packing_mode, resolution=resolution)
    previous = {
        placement.object_id: placement for placement in data.atlases[0].placements
    }
    original_png = (tmp_path / "original-atlas.png").read_bytes()
    original_ao = (tmp_path / "original-ao.png").read_bytes()
    workspace.set_data(data)
    # The canvas/scene source catalog is authoritative; loading alone cannot
    # infer source categories from a legacy name or its half packing shape.
    assert workspace.get_data() == data

    workspace.set_object_texture_sources(sources)

    migrated = workspace.get_data()
    assert len(migrated.atlases) == 2
    assert {atlas.atlas_type for atlas in migrated.atlases} == {"object", "surface"}
    assert migrated.selected_atlas_id == "legacy"
    assert migrated.atlas_by_id("legacy").atlas_type == ATLAS_TYPE_SURFACE
    actual_ids = []
    for atlas in migrated.atlases:
        assert "[HALF]" not in atlas.name
        assert atlas.resolution == 2048
        assert atlas.name.endswith("Custom furniture")
        assert atlas.surface_ao_image_path is None
        assert atlas.surface_ao_geometry_signature is None
        for placement in atlas.placements:
            actual_ids.append(placement.object_id)
            old = previous[placement.object_id]
            assert (placement.x, placement.y, placement.size) == (old.x, old.y, old.size)
            assert placement.texture_resolution == old.texture_resolution
            assert placement.packing_mode == old.packing_mode
            assert placement.texture_path == old.texture_path
            assert workspace._source_atlas_type(placement.object_id) == atlas.atlas_type
            _assert_placement_color(tmp_path, atlas, placement, colors[placement.object_id])
    assert sorted(actual_ids) == sorted(previous)
    assert TextureAtlasData.from_dict(migrated.to_dict()) == migrated
    assert (tmp_path / "original-atlas.png").read_bytes() == original_png
    assert (tmp_path / "original-ao.png").read_bytes() == original_ao
    before_repeat = _pixel_snapshot(tmp_path)
    assert workspace.normalize_legacy_atlas_types() == 0
    assert _pixel_snapshot(tmp_path) == before_repeat


@pytest.mark.parametrize("missing", ("catalog", "file"))
def test_missing_exact_source_keeps_mixed_metadata_and_all_pngs_untouched(
    workspace, tmp_path, missing
):
    data, sources, _colors = _mixed_legacy_data(tmp_path, ATLAS_PACKING_MODE_SYMMETRIC_HALF)
    workspace.set_data(data)
    if missing == "file":
        sources[-1].physical_texture_path.unlink()
        available = sources
    else:
        available = sources[:-1]
    before = _pixel_snapshot(tmp_path)
    missing_entry = AtlasSurfaceTextureEntry(sources[-1].object_id, "Unavailable surface", 1)

    workspace.set_object_texture_sources(available, surface_texture_entries=(missing_entry,))

    assert workspace.get_data() == data
    assert _pixel_snapshot(tmp_path) == before
    assert workspace._atlas_type_migration_warning
    assert "kept unchanged" in workspace.status_label.text()


def test_failed_second_split_image_rolls_back_first_split_and_metadata(workspace, tmp_path):
    data, sources, _colors = _mixed_legacy_data(tmp_path, ATLAS_PACKING_MODE_SYMMETRIC_HALF)
    workspace.set_data(data)
    before = _pixel_snapshot(tmp_path)
    materialize = workspace._materialize_atlas
    attempted = []

    def fail_second(atlas, **kwargs):
        attempted.append(atlas.atlas_id)
        if len(attempted) == 2:
            raise ValueError("Synthetic second-atlas raster failure")
        materialize(atlas, **kwargs)

    with patch.object(workspace, "_materialize_atlas", side_effect=fail_second):
        workspace.set_object_texture_sources(sources)

    assert len(attempted) == 2
    assert workspace.get_data() == data
    assert _pixel_snapshot(tmp_path) == before


def test_migration_resolves_original_texture_variant_not_the_active_resolution(
    workspace, tmp_path
):
    data, originals, colors = _mixed_legacy_data(tmp_path, ATLAS_PACKING_MODE_SYMMETRIC_HALF)
    workspace.set_data(data)
    active = tuple(
        _source(
            tmp_path, index, surface=bool(index % 2),
            packing_mode=ATLAS_PACKING_MODE_SYMMETRIC_HALF, resolution=1024,
        )[0]
        for index in range(2)
    )
    by_id = {source.object_id: source for source in originals}
    resolver = Mock(side_effect=lambda object_id, resolution: by_id.get(object_id) if resolution == 512 else None)

    workspace.set_object_texture_sources(active, variant_resolver=resolver)

    assert resolver.call_count >= 2
    result = workspace.get_data()
    assert len(result.atlases) == 2
    for atlas in result.atlases:
        for placement in atlas.placements:
            assert placement.texture_resolution == 512
            assert placement.texture_path == by_id[placement.object_id].texture_path
            _assert_placement_color(tmp_path, atlas, placement, colors[placement.object_id])


def test_migration_type_change_does_not_collide_with_an_intermediate_old_prefix(
    workspace, tmp_path
):
    source, _ = _source(tmp_path, 1, surface=True)
    data = TextureAtlasData()
    original = data.create_atlas("Furniture", 2048, atlas_id="wrong", atlas_type=ATLAS_TYPE_OBJECT)
    data.assign_object(original.atlas_id, source.object_id, source.texture_path, 512)
    data.create_atlas("Furniture", 2048, atlas_type=ATLAS_TYPE_SURFACE)
    data.create_atlas("Furniture 2", 2048, atlas_type=ATLAS_TYPE_OBJECT)
    workspace.set_data(data)

    workspace.set_object_texture_sources((source,))

    migrated = workspace.get_data().atlas_by_id(original.atlas_id)
    assert migrated.atlas_type == ATLAS_TYPE_SURFACE
    assert migrated.name == "[SURFACE] Furniture 2"
    assert migrated.placements == original.placements
    assert workspace.get_data().clone() == workspace.get_data()


@pytest.mark.parametrize("name", ("[HALF]", "[OBJECT]", "[SURFACE]"))
def test_valid_legacy_bare_prefix_names_have_a_safe_typed_fallback(workspace, name):
    data = TextureAtlasData()
    data.create_atlas(name, 2048, atlas_id="legacy")
    workspace.set_data(data)

    workspace.set_object_texture_sources(())

    migrated = workspace.get_data().atlas_by_id("legacy")
    assert migrated.atlas_type == ATLAS_TYPE_OBJECT
    assert migrated.name == "[OBJECT] Atlas"


# ### Partial-slot normalization metadata ###
def test_partial_slot_compaction_keeps_geometry_and_invalidates_only_stale_bakes(tmp_path):
    data, sources, _colors = _mixed_legacy_data(tmp_path, ATLAS_PACKING_MODE_SYMMETRIC_HALF)
    atlas = data.atlases[0]
    right = atlas.placements[-1]
    atlas.placements = [right]
    atlas.set_surface_ambient_occlusion("ao.png", "geometry-signature")

    normalize_atlas_partial_slots(atlas)

    compacted = atlas.placements[0]
    assert compacted.slot_half == ATLAS_SLOT_HALF_LEFT
    assert (compacted.x, compacted.y, compacted.size) == (right.x, right.y, right.size)
    assert compacted.packing_mode == right.packing_mode
    assert compacted.texture_path == right.texture_path
    assert compacted.texture_resolution == right.texture_resolution
    assert atlas.image_path is None
    assert atlas.surface_ao_image_path is None
    assert atlas.surface_ao_geometry_signature is None
    assert all(source.physical_texture_path.is_file() for source in sources)
    atlas.image_path = "rebuilt.png"
    atlas.set_surface_ambient_occlusion("rebuilt-ao.png", "new-geometry")
    normalize_atlas_partial_slots(atlas)
    assert atlas.image_path == "rebuilt.png"
    assert atlas.surface_ao_image_path == "rebuilt-ao.png"
    assert TextureAtlasData.from_dict(data.to_dict()) == data


# ### Typed manual guards and automatic destinations ###
def test_manual_wrong_typed_assignment_redirects_without_mixing(workspace, tmp_path):
    object_source, _ = _source(tmp_path, 0, surface=False)
    surface_source, _ = _source(tmp_path, 1, surface=True)
    workspace.set_object_texture_sources((object_source, surface_source))
    data = TextureAtlasData()
    data.create_atlas("Furniture", 2048, atlas_id="objects", atlas_type=ATLAS_TYPE_OBJECT)
    workspace.set_data(data)
    before = workspace.get_data()
    before_pngs = _pixel_snapshot(tmp_path)
    commit = Mock(return_value=True)

    with pytest.raises(ValueError, match="separate Atlases"):
        workspace._build_selected_source_assignment_candidate(
            surface_source, source_ids_to_remove=()
        )

    assert workspace.get_data() == before
    assert _pixel_snapshot(tmp_path) == before_pngs
    commit.assert_not_called()
    assert workspace.assign_source_to_selected_atlas(surface_source.object_id, commit)
    assert workspace.get_data().atlas_by_id("objects") == before.atlas_by_id("objects")
    assert workspace.selected_atlas.atlas_type == ATLAS_TYPE_SURFACE
    assert workspace.selected_atlas.placement_for_object(surface_source.object_id)
    commit.assert_called_once_with()


def test_surface_assignment_finds_typed_destination_instead_of_mixing(workspace, tmp_path):
    object_source, _ = _source(tmp_path, 0, surface=False)
    surface_source, color = _source(tmp_path, 1, surface=True)
    workspace.set_object_texture_sources((object_source, surface_source))
    data = TextureAtlasData()
    data.create_atlas("Furniture", 2048, atlas_id="objects", atlas_type=ATLAS_TYPE_OBJECT)
    workspace.set_data(data)
    commit = Mock(return_value=True)

    assert workspace.ensure_source_in_typed_atlas(surface_source.object_id, commit)

    commit.assert_called_once_with()
    result = workspace.get_data()
    assert result.atlas_by_id("objects").placements == []
    assert workspace.selected_atlas.atlas_type == ATLAS_TYPE_SURFACE
    assert workspace.selected_atlas.name.startswith("[SURFACE]")
    placement = workspace.selected_atlas.placement_for_object(surface_source.object_id)
    assert placement is not None
    _assert_placement_color(tmp_path, workspace.selected_atlas, placement, color)
