# ### Imports ###
from __future__ import annotations

from dataclasses import replace
from io import BytesIO

import numpy as np
import trimesh
from PIL import Image

from housemaker.glb import convert_to_glb
from housemaker.models import (
    DoorwayData,
    LevelData,
    SlopedCeilingData,
    VertexData,
    WindowData,
)
from housemaker.sloped_ceiling import (
    add_sloped_ceiling_window,
    apply_sloped_ceiling_geometry,
    build_sloped_ceiling_window_hosts,
    build_sloped_ceiling_window_placement,
)
from housemaker.surface_geometry import (
    build_base_fixed_surfaces,
    build_fixed_surface_scene_snapshot,
    build_fixed_surfaces,
    build_opening_reveal_surface_id,
)


# ### Fixture helpers ###
def _parallel_wall_level(*, doorway: bool = False, arched: bool = False) -> LevelData:
    vertices = VertexData()
    for point in ((0.0, 0.0), (200.0, 0.0), (0.0, 20.0), (200.0, 20.0)):
        vertices.add_vertex(*point)
    vertices.add_edge(1, 2)
    vertices.add_edge(3, 4)
    level = LevelData(index=0, name="Ground", vertex_data=vertices)
    if doorway:
        level.doorways = [
            DoorwayData(
                doorway_id="door-1",
                center_x=100.0,
                center_y=10.0,
                width_meters=1.2,
                height_meters=2.0,
                depth_meters=0.4,
                rotation_degrees=90.0,
                shape="arch" if arched else "rectangular",
            )
        ]
    else:
        level.windows = [
            WindowData(
                window_id="window-1",
                wall_surface_id="level:0/wall:1:2",
                start_ratio=0.2,
                end_ratio=0.8,
                bottom_ratio=0.2,
                top_ratio=0.8,
            )
        ]
    return level


def _reveals(level: LevelData):
    return {
        surface.opening_part: surface
        for surface in build_fixed_surfaces((level,))
        if surface.opening_id is not None
    }


def _png() -> bytes:
    output = BytesIO()
    Image.new("RGBA", (8, 8), (120, 150, 180, 255)).save(output, format="PNG")
    return output.getvalue()


def _assert_valid_uvs(surface) -> None:
    _assert_valid_mesh_uvs(surface.mesh)


def _assert_valid_mesh_uvs(mesh: trimesh.Trimesh) -> None:
    uv = np.asarray(mesh.visual.uv)
    assert uv.shape == (len(mesh.vertices), 2)
    assert np.all(np.isfinite(uv))
    triangles = uv[np.asarray(mesh.faces)]
    first = triangles[:, 1] - triangles[:, 0]
    second = triangles[:, 2] - triangles[:, 0]
    areas = np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    assert np.all(areas > 1e-10)


# ### Independent reveal geometry tests ###
def test_windows_have_four_independent_reveals_without_wall_or_edit_identity() -> None:
    level = _parallel_wall_level()
    reveals = _reveals(level)
    assert set(reveals) == {"left", "right", "header", "sill"}
    for part, surface in reveals.items():
        assert surface.surface_id == f"level:0/opening:window:window-1/reveal:{part}"
        assert surface.surface_type == "wall"
        assert surface.opening_owner_surface_id == "level:0/wall:1:2"
        assert surface.source_surface_id is None
        assert surface.wall_key is None
        assert surface.wall_start_world is None
        assert surface.wall_end_world is None
        assert surface.surface_alias_ids == ()
        assert len(surface.mesh.faces) == 2
        _assert_valid_uvs(surface)
    for surface in build_fixed_surfaces((level,)):
        if surface.wall_key is not None:
            assert np.all(np.abs(surface.mesh.face_normals[:, 1]) > 0.999)


def test_doorways_have_three_reveals_and_a_sill_only_when_raised() -> None:
    level = _parallel_wall_level(doorway=True)
    assert set(_reveals(level)) == {"left", "right", "header"}
    level.doorways[0] = replace(level.doorways[0], bottom_height_meters=0.3)
    reveals = _reveals(level)
    assert set(reveals) == {"left", "right", "header", "sill"}
    assert np.all(reveals["header"].mesh.face_normals[:, 2] < -0.999)
    assert np.all(reveals["sill"].mesh.face_normals[:, 2] > 0.999)


def test_arch_header_remains_one_texturable_surface_with_valid_uvs() -> None:
    reveals = _reveals(_parallel_wall_level(doorway=True, arched=True))
    assert set(reveals) == {"left", "right", "header"}
    assert len(reveals["header"].mesh.faces) > 4
    for surface in reveals.values():
        _assert_valid_uvs(surface)


def test_reveal_ids_survive_opening_size_and_level_transform_changes() -> None:
    level = _parallel_wall_level()
    initial = {surface.surface_id for surface in _reveals(level).values()}
    level.windows[0] = replace(level.windows[0], start_ratio=0.25, top_ratio=0.75)
    level.height_meters = 3.4
    assert {surface.surface_id for surface in _reveals(level).values()} == initial
    snapshot = build_fixed_surface_scene_snapshot((level,))
    assert initial.issubset(surface.surface_id for surface in snapshot.fixed_surfaces)


def test_opening_ids_are_encoded_without_collisions() -> None:
    assert build_opening_reveal_surface_id(0, "window", "A/B : é", "sill") == (
        "level:0/opening:window:A%2FB%20%3A%20%C3%A9/reveal:sill"
    )
    assert build_opening_reveal_surface_id(0, "window", "a/b", "left") != (
        build_opening_reveal_surface_id(0, "window", "a%2Fb", "left")
    )


# ### Preview and export integration tests ###
def test_texturing_one_reveal_does_not_texture_walls_or_other_reveals() -> None:
    level = _parallel_wall_level()
    reveals = _reveals(level)
    selected = reveals["left"]
    model = convert_to_glb((level,), surface_materials={selected.surface_id: _png()})
    assert [surface.surface_id for surface in model.preview_textured_surfaces] == [selected.surface_id]
    np.testing.assert_allclose(model.preview_textured_surfaces[0].mesh.area, selected.mesh.area)
    assert not any(name.endswith(("_doorway_reveals", "_window_reveals")) for name in model.scene.geometry)
    assert len(model.glb_bytes) > 0
    expected_area = sum(surface.mesh.area for surface in build_fixed_surfaces((level,)))
    assert np.isclose(model.mesh.area, expected_area)


def test_untextured_export_contains_each_physical_reveal_face_once() -> None:
    level = _parallel_wall_level()
    surfaces = build_fixed_surfaces((level,))
    model = convert_to_glb((level,))
    for surface in _reveals(level).values():
        named = [
            mesh for mesh in model.scene.geometry.values()
            if mesh.metadata.get("housemaker_surface_id") == surface.surface_id
        ]
        assert len(named) == 1
        assert len(named[0].faces) == 2
    assert np.isclose(model.mesh.area, sum(surface.mesh.area for surface in surfaces))


def test_textured_arch_header_keeps_nondegenerate_uvs_in_preview_and_glb() -> None:
    level = _parallel_wall_level(doorway=True, arched=True)
    header = _reveals(level)["header"]
    model = convert_to_glb((level,), surface_materials={header.surface_id: _png()})
    assert len(model.preview_textured_surfaces) == 1
    textured = model.preview_textured_surfaces[0]
    _assert_valid_uvs(textured)
    assert textured.mesh.area == header.mesh.area
    exported = trimesh.load(BytesIO(model.glb_bytes), file_type="glb", force="scene")
    assigned = [
        mesh for mesh in exported.geometry.values()
        if getattr(getattr(mesh.visual, "material", None), "name", "")
        == f"Surface {header.surface_id}"
    ]
    assert len(assigned) == 1
    _assert_valid_mesh_uvs(assigned[0])


def test_textured_slope_window_header_and_sill_export_valid_uvs_without_duplicates() -> None:
    level = LevelData(index=0, name="Ground")
    for point in ((0, 0), (200, 0), (200, 200), (0, 200)):
        level.vertex_data.add_vertex(*point)
    for first, second in ((1, 2), (2, 3), (3, 4), (4, 1)):
        level.vertex_data.add_edge(first, second)
    flat = build_base_fixed_surfaces((level,), apply_sloped_ceilings=False)
    level.sloped_ceilings = [
        SlopedCeilingData(
            slope_id="0" * 32,
            wall_surface_id="level:0/wall:1:2",
            depth_meters=1.2,
            wall_drop_meters=0.8,
            start_ratio=0.1,
            end_ratio=0.9,
        )
    ]
    (host,) = build_sloped_ceiling_window_hosts((level,), flat)
    placement = build_sloped_ceiling_window_placement(
        host, host.local_to_world(0.25, 0.3), host.local_to_world(0.75, 0.7)
    )
    add_sloped_ceiling_window((level,), placement, window_id="inclined-window")
    surfaces = build_fixed_surfaces((level,))
    reveals = {surface.opening_part: surface for surface in surfaces if surface.opening_id}
    assert set(reveals) == {"left", "right", "header", "sill"}
    for surface in reveals.values():
        _assert_valid_uvs(surface)
    rebuilt = apply_sloped_ceiling_geometry((level,), surfaces)
    assert len([surface for surface in rebuilt if surface.opening_id]) == 4
    selected = {reveals[part].surface_id for part in ("header", "sill")}
    model = convert_to_glb((level,), surface_materials={surface_id: _png() for surface_id in selected})
    assert {surface.surface_id for surface in model.preview_textured_surfaces} == selected
    for surface in model.preview_textured_surfaces:
        _assert_valid_uvs(surface)
    exported = trimesh.load(BytesIO(model.glb_bytes), file_type="glb", force="scene")
    assigned = [
        mesh for mesh in exported.geometry.values()
        if getattr(getattr(mesh.visual, "material", None), "name", "")
        in {f"Surface {surface_id}" for surface_id in selected}
    ]
    assert len(assigned) == 2
    for mesh in assigned:
        assert len(mesh.faces) == 2
        _assert_valid_mesh_uvs(mesh)
