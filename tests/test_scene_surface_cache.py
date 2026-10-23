# ### Imports ###
from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from housemaker.models import (
    ArchitecturalTrimData,
    DoorwayData,
    EditableSurfaceMeshData,
    EditableSurfaceVertexData,
    LandingData,
    LevelData,
    OpenSpaceData,
    RoomData,
    SlopedCeilingData,
    StairData,
    WallTextureData,
    WindowData,
    create_default_levels,
)
from housemaker.project_io import load_project, save_project
from housemaker.scene_surface_cache import build_scene_surface_geometry_signature


# ### Fixture helpers ###
def _level() -> LevelData:
    level = LevelData(index=2, name="Ground")
    for point in ((0, 0), (100, 0), (100, 100), (0, 100), (50, 50)):
        level.vertex_data.add_vertex(*point)
    for first, second in ((1, 2), (2, 3), (3, 4), (4, 1)):
        level.vertex_data.add_edge(first, second)
    level.rooms = [RoomData("Room", (1, 2, 3, 4), 5, (120, 130, 140))]
    level.doorways = [
        DoorwayData(50, 0, 0.8, 2.0, doorway_id="doorway-a")
    ]
    level.windows = [
        WindowData("window-a", "level:2/room:5/wall:1:2", 0.1, 0.4, 0.3, 0.7)
    ]
    return level


def _signature(level: LevelData) -> tuple[object, ...]:
    return build_scene_surface_geometry_signature((level,), ())


# ### Immutable snapshot tests ###
def test_signature_owns_mutable_doorway_room_and_graph_values() -> None:
    level = _level()
    saved = copy.deepcopy(level)
    initial = _signature(level)
    assert hash(initial) == hash(_signature(saved))
    graph_revision = level.vertex_data.revision

    level.doorways[0].height_meters = 2.4
    level.rooms[0].height_meters = 3.5
    level.rooms[0].wall_subdivisions["1:2"] = 3
    level.vertex_data.vertices[0] = replace(level.vertex_data.vertices[0], x=1.0)
    assert level.vertex_data.revision == graph_revision
    assert _signature(level) != initial
    assert _signature(saved) == initial

    # Undo/reload may replace all model objects without changing their geometry.
    level = copy.deepcopy(saved)
    assert _signature(level) == initial


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("index", 3),
        ("include_in_export", False),
        ("height_meters", 3.4),
        ("floor_thickness_meters", 0.4),
        ("scale", 1.3),
        ("offset_x_meters", 0.4),
        ("offset_y_meters", 0.4),
        ("canvas_level_scale", 1.3),
        ("canvas_offset_x_pixels", 20.0),
        ("canvas_offset_y_pixels", 20.0),
        ("image_size_pixels", (1200, 900)),
        ("plan_meters_per_pixel", 0.03),
    ),
)
def test_pending_level_transform_or_metric_edit_changes_signature(name, value) -> None:
    level = _level()
    before = _signature(level)
    setattr(level, name, value)
    assert _signature(level) != before


@pytest.mark.parametrize(
    "edit",
    (
        lambda level: level.vertex_data.remove_edge(1, 2),
        lambda level: setattr(level.doorways[0], "center_x", 55.0),
        lambda level: setattr(level.doorways[0], "corner_vertex_ids", (1, 2, 3, 4)),
        lambda level: setattr(level.doorways[0], "arch_amount", 0.5),
        lambda level: level.windows.__setitem__(0, replace(level.windows[0], top_ratio=0.8)),
        lambda level: level.open_spaces.append(OpenSpaceData("hole", 10, 10, 30, 30)),
        lambda level: level.flipped_surface_ids.add("level:2/room:5/wall:1:2"),
        lambda level: level.sloped_ceilings.append(SlopedCeilingData("1" * 32, "level:2/room:5/wall:1:2")),
        lambda level: level.architectural_trims.append(ArchitecturalTrimData("2" * 32, "cornice", ("level:2/room:5/wall:1:2",))),
        lambda level: level.landings.append(LandingData(50, 50, landing_id="3" * 32)),
        lambda level: level.editable_surfaces.append(EditableSurfaceMeshData(
            "level:2/room:5/wall:1:2", "wall_ratio", 1,
            (EditableSurfaceVertexData("4" * 32, 0, 0, 0),), (),
        )),
        lambda level: level.rooms[0].wall_subdivision_source_ranges.__setitem__("1:2", ((0.0, 0.5),)),
    ),
)
def test_geometry_collections_and_opening_edits_invalidate_without_viewer_revision(edit) -> None:
    level = _level()
    before = _signature(level)
    edit(level)
    assert _signature(level) != before


def test_excluded_levels_stairs_and_generated_trim_replacements_are_dependencies() -> None:
    level = _level()
    excluded = LevelData(index=3, name="Upper", include_in_export=False)
    stairs = [StairData(2, 20, 20, 3, 80, 80, stair_id="5" * 32)]
    before = build_scene_surface_geometry_signature((level, excluded), stairs)
    excluded.height_meters += 0.5
    assert build_scene_surface_geometry_signature((level, excluded), stairs) != before
    before = build_scene_surface_geometry_signature((level, excluded), stairs)
    stairs[0] = replace(stairs[0], target_rise_meters=0.2)
    assert build_scene_surface_geometry_signature((level, excluded), stairs) != before
    assert build_scene_surface_geometry_signature(
        (level, excluded), stairs, generated_trim_keys=((2, "2" * 32),)
    ) != build_scene_surface_geometry_signature((level, excluded), stairs)
    keys = ((2, "a"), (3, "b"))
    assert build_scene_surface_geometry_signature((level,), (), generated_trim_keys=keys) == (
        build_scene_surface_geometry_signature((level,), (), generated_trim_keys=keys[::-1])
    )


# ### Non-geometry exclusions and reload tests ###
def test_texture_files_names_and_ocr_details_do_not_invalidate_geometry() -> None:
    level = _level()
    initial = _signature(level)
    level.name = "Renamed"
    level.rooms[0].name = "Different name"
    level.rooms[0].color_rgb = (10, 20, 30)
    level.image_path = "missing-image.png"
    level.original_image_path = "original-image.png"
    level.rooms[0].wall_textures["1:2"] = WallTextureData("missing-texture.png", 0, 0, 10, 10)
    level.plan_metric_calibration = SimpleNamespace(
        target_meters_per_pixel=level.meters_per_pixel,
        measurements=object(),
        diagnostics=object(),
    )
    assert _signature(level) == initial
    level.plan_metric_calibration.target_meters_per_pixel *= 1.1
    assert _signature(level) != initial


def test_save_reload_with_equal_geometry_retains_signature(tmp_path) -> None:
    levels = create_default_levels()
    levels[2] = _level()
    before = build_scene_surface_geometry_signature(levels, ())
    path = tmp_path / "project.json"
    save_project(path, current_level_index=2, levels=levels)
    loaded = load_project(path)
    assert build_scene_surface_geometry_signature(loaded.levels, loaded.stairs) == before
