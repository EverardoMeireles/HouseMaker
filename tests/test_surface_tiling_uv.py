# ### Imports ###
from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
import trimesh
from PIL import Image

from housemaker.pbr_maps import ATLAS_MAP_BASE_COLOR, PBR_MAP_NORMAL
from housemaker.surface_materials import (
    SurfaceMaterialSourceSpec,
    build_world_planar_textured_mesh,
    resolve_surface_material,
)
from housemaker.surface_tiling_uv import (
    SURFACE_TILING_MODE_EDGE_VARIANTS,
    SURFACE_TILING_MODE_WHOLE_REPEATS,
    quarter_turns_for_tile,
)


# ### Fixtures ###
def _png(width: int, height: int, rgba: tuple[int, int, int, int]) -> bytes:
    image = Image.new("RGBA", (width, height), rgba)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _floor_mesh(lower: float, upper: float) -> trimesh.Trimesh:
    vertices = np.asarray(
        ((lower, lower, 0), (upper, lower, 0), (upper, upper, 0), (lower, upper, 0)),
        dtype=float,
    )
    return trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        metadata={"surface-test": "retained"},
        process=False,
    )


def _expected_rotated_uv(local: np.ndarray, turns: int) -> np.ndarray:
    u, v = local[:, 0], local[:, 1]
    if turns == 0:
        return local
    if turns == 1:
        return np.column_stack((v, 1 - u))
    if turns == 2:
        return 1 - local
    return np.column_stack((1 - v, u))


# ### Source metadata tests ###
def test_source_spec_is_immutable_and_preserves_pbr_maps() -> None:
    normal_png = _png(8, 8, (36, 144, 228, 255))
    sources = {
        ATLAS_MAP_BASE_COLOR: _png(8, 8, (80, 120, 190, 255)),
        PBR_MAP_NORMAL: normal_png,
    }
    spec = SurfaceMaterialSourceSpec(
        map_sources=sources,
        tiling_mode=SURFACE_TILING_MODE_WHOLE_REPEATS,
        tiling_seed=123,
    )
    sources.clear()
    material = resolve_surface_material(spec)
    assert material.tiling_mode == SURFACE_TILING_MODE_WHOLE_REPEATS
    assert material.tiling_seed == 123
    assert material.normal_texture_rgba is not None
    np.testing.assert_array_equal(
        material.normal_texture_rgba[0, 0], (36, 144, 228, 255)
    )
    with pytest.raises(TypeError):
        spec.map_sources[PBR_MAP_NORMAL] = normal_png  # type: ignore[index]


# ### Repeat geometry tests ###
def test_repeat_size_overrides_world_uv_scale_without_resizing_texture() -> None:
    texture_png = _png(8, 8, (90, 120, 150, 255))
    default_material = resolve_surface_material(texture_png)
    repeated_material = resolve_surface_material(
        SurfaceMaterialSourceSpec(
            map_sources={ATLAS_MAP_BASE_COLOR: texture_png},
            texture_repeat_size_m=0.5,
        )
    )
    mesh = _floor_mesh(0, 1)
    default_mesh = build_world_planar_textured_mesh(
        mesh, "floor", default_material, texture_world_size_meters=2
    )
    repeated_mesh = build_world_planar_textured_mesh(
        mesh, "floor", repeated_material, texture_world_size_meters=2
    )

    np.testing.assert_allclose(repeated_mesh.vertices, default_mesh.vertices)
    np.testing.assert_allclose(
        np.asarray(repeated_mesh.visual.uv),
        np.asarray(default_mesh.visual.uv) * 4,
    )
    assert repeated_material.png_bytes == default_material.png_bytes == texture_png
    assert repeated_material.texture_rgba.shape == default_material.texture_rgba.shape


def test_repeat_size_reclips_edge_variant_tiles_at_new_world_spacing() -> None:
    texture_png = _png(8, 8, (90, 120, 150, 255))
    def material(repeat_size: float):
        return resolve_surface_material(
            SurfaceMaterialSourceSpec(
                map_sources={ATLAS_MAP_BASE_COLOR: texture_png},
                tiling_mode=SURFACE_TILING_MODE_EDGE_VARIANTS,
                texture_repeat_size_m=repeat_size,
            )
        )

    mesh = _floor_mesh(0, 2)
    broad = build_world_planar_textured_mesh(
        mesh, "floor", material(2), texture_world_size_meters=2
    )
    dense = build_world_planar_textured_mesh(
        mesh, "floor", material(0.5), texture_world_size_meters=2
    )

    assert len(dense.faces) > len(broad.faces)
    assert np.isclose(dense.area, mesh.area)
    assert np.isclose(broad.area, mesh.area)
    assert np.all(np.asarray(dense.visual.uv) >= -1e-8)
    assert np.all(np.asarray(dense.visual.uv) <= 1 + 1e-8)


def test_whole_repeats_rotate_each_clipped_tile_without_changing_geometry() -> None:
    material = resolve_surface_material(
        SurfaceMaterialSourceSpec(
            map_sources={ATLAS_MAP_BASE_COLOR: _png(8, 8, (90, 120, 150, 255))},
            tiling_mode=SURFACE_TILING_MODE_WHOLE_REPEATS,
            tiling_seed=41,
        )
    )
    result = build_world_planar_textured_mesh(
        _floor_mesh(-1, 1), "floor", material, texture_world_size_meters=1
    )
    assert result.metadata["surface-test"] == "retained"
    assert len(result.faces) > 2
    assert np.isclose(result.area, 4.0)
    np.testing.assert_allclose(
        result.vertex_normals,
        np.repeat(np.asarray(((0, 0, 1),), dtype=float), len(result.vertices), axis=0),
    )
    uv = np.asarray(result.visual.uv)
    for face in result.faces:
        positions = result.vertices[face, :2]
        tile = np.floor(np.mean(positions, axis=0)).astype(int)
        local = positions - tile
        turns = quarter_turns_for_tile(int(tile[0]), int(tile[1]), 41)
        np.testing.assert_allclose(
            uv[face], _expected_rotated_uv(local, turns), atol=1e-8
        )


def test_edge_variants_choose_quadrants_for_positive_and_negative_tiles() -> None:
    material = resolve_surface_material(
        SurfaceMaterialSourceSpec(
            map_sources={ATLAS_MAP_BASE_COLOR: _png(8, 8, (90, 120, 150, 255))},
            tiling_mode=SURFACE_TILING_MODE_EDGE_VARIANTS,
            tiling_seed=41,
        )
    )
    result = build_world_planar_textured_mesh(
        _floor_mesh(-1, 1), "floor", material, texture_world_size_meters=1
    )
    uv = np.asarray(result.visual.uv)
    observed: set[tuple[int, int]] = set()
    for face in result.faces:
        positions = result.vertices[face, :2]
        tile = np.floor(np.mean(positions, axis=0)).astype(int)
        tile_u, tile_v = int(tile[0]), int(tile[1])
        observed.add((tile_u, tile_v))
        expected = (positions - tile + (tile_u % 2, tile_v % 2)) * 0.5
        np.testing.assert_allclose(uv[face], expected, atol=1e-8)
    assert observed == {(-1, -1), (0, -1), (-1, 0), (0, 0)}


def test_rectangular_texture_uses_only_half_turns_and_keeps_normal_pixels() -> None:
    normal_color = (36, 144, 228, 255)
    material = resolve_surface_material(
        SurfaceMaterialSourceSpec(
            map_sources={
                ATLAS_MAP_BASE_COLOR: _png(8, 4, (90, 120, 150, 255)),
                PBR_MAP_NORMAL: _png(8, 4, normal_color),
            },
            tiling_mode=SURFACE_TILING_MODE_WHOLE_REPEATS,
            tiling_seed=7,
        )
    )
    result = build_world_planar_textured_mesh(
        _floor_mesh(-1, 1), "floor", material, texture_world_size_meters=1
    )
    uv = np.asarray(result.visual.uv)
    for face in result.faces:
        positions = result.vertices[face, :2]
        tile = np.floor(np.mean(positions, axis=0)).astype(int)
        turns = quarter_turns_for_tile(
            int(tile[0]), int(tile[1]), 7, allow_odd_turns=False
        )
        assert turns in (0, 2)
        np.testing.assert_allclose(
            uv[face], _expected_rotated_uv(positions - tile, turns), atol=1e-8
        )
    assert result.visual.material.normalTexture is not None
    np.testing.assert_array_equal(
        np.asarray(result.visual.material.normalTexture)[0, 0], normal_color
    )


def test_quarter_turns_for_signed_tiles_is_stable_and_validates_seed() -> None:
    assert quarter_turns_for_tile(-3, 2, 123) == quarter_turns_for_tile(-3, 2, 123)
    assert quarter_turns_for_tile(-3, 2, 123, allow_odd_turns=False) in (0, 2)
    with pytest.raises(ValueError):
        quarter_turns_for_tile(0, 0, -1)
