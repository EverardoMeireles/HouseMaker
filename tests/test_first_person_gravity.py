"""Pure support-height queries need neither ray-tree nor physics dependencies."""

# ### Imports ###
from __future__ import annotations

import math

import numpy as np
import pytest
import trimesh

from housemaker.first_person_gravity import FirstPersonGroundIndex


# ### Geometry fixtures ###
def _square(x=0.0, y=0.0, z=0.0, *, width=1.0):
    a, b = (x, y, z), (x + width, y, z)
    c, d = (x + width, y + width, z), (x, y + width, z)
    return np.asarray(((a, b, c), (a, c, d)), dtype=float)


def _grid(size=100):
    return np.concatenate(
        tuple(_square(float(x), float(y)) for x in range(size) for y in range(size))
    )


# ### Triangle geometry and height caps ###
@pytest.mark.parametrize("reverse", (False, True))
def test_floor_winding_does_not_change_support(reverse):
    triangles = _square(z=2.0)
    if reverse:
        triangles = triangles[:, ::-1]
    index = FirstPersonGroundIndex(triangles)
    assert index.ground_height(0.3, 0.7, 3.0) == pytest.approx(2.0)
    assert index.ground_height(0.3, 0.7, 1.0) is None


@pytest.mark.parametrize("point", ((0.0, 0.0), (1.0, 1.0), (0.5, 0.5), (1.0, 0.4)))
def test_shared_edges_and_vertices_include_support(point):
    index = FirstPersonGroundIndex(_square(z=1.0))
    assert index.ground_height(*point, 1.0, 1.0) == pytest.approx(1.0)


def test_highest_support_is_below_foot_cap_and_above_optional_minimum():
    index = FirstPersonGroundIndex(
        np.concatenate(tuple(_square(z=z) for z in (-12.0, -6.0, 0.0, 3.0, 6.0)))
    )
    assert index.ground_height(0.5, 0.5, 6.4) == 6.0
    assert index.ground_height(0.5, 0.5, 5.0) == 3.0
    assert index.ground_height(0.5, 0.5, 0.4) == 0.0
    assert index.ground_height(0.5, 0.5, -6.1) == -12.0
    assert index.ground_height(0.5, 0.5, 2.0, 1.0) is None
    assert index.ground_height(0.5, 0.5, -20.0) is None
    assert index.ground_height(0.5, 0.5, 1.0, 2.0) is None


def test_stair_treads_and_missing_floor_cells_have_distinct_support():
    stairs = np.concatenate(tuple(_square(float(step), z=step * 0.2) for step in range(5)))
    index = FirstPersonGroundIndex(stairs, cell_size=0.5)
    assert index.ground_height(0.5, 0.5, 0.3) == 0.0
    assert index.ground_height(1.5, 0.5, 0.3) == pytest.approx(0.2)
    assert index.ground_height(2.5, 0.5, 0.3) is None
    assert index.ground_height(2.5, 0.5, 0.5) == pytest.approx(0.4)
    ring = np.concatenate(tuple(
        _square(float(x), float(y))
        for x in range(-1, 2) for y in range(-1, 2) if (x, y) != (0, 0)
    ))
    ring_index = FirstPersonGroundIndex(ring)
    assert ring_index.ground_height(0.5, 0.5, 1.0) is None
    assert ring_index.ground_height(-0.5, 0.5, 1.0) == 0.0


def test_sloped_support_is_interpolated_at_the_query_position():
    slope = np.asarray((((0, 0, 0), (2, 0, 2), (0, 2, 0)),), dtype=float)
    index = FirstPersonGroundIndex(slope)
    assert index.ground_height(0.5, 0.5, 1.0) == pytest.approx(0.5)
    assert index.ground_height(1.5, 0.25, 2.0) == pytest.approx(1.5)
    assert index.ground_height(1.5, 1.5, 5.0) is None


def test_vertical_degenerate_and_nonfinite_faces_do_not_supply_support():
    unusable = np.asarray((
        ((0, 0, 0), (1, 0, 0), (0, 0, 3)),
        ((0, 0, 0), (1, 1, 1), (2, 2, 2)),
        ((0, 0, 0), (0, 0, 0), (0, 0, 0)),
        ((0, 0, math.nan), (1, 0, 0), (0, 1, 0)),
        ((0, 0, 0), (math.inf, 0, 0), (0, 1, 0)),
    ), dtype=float)
    index = FirstPersonGroundIndex(np.concatenate((unusable, _square(z=-1.0))))
    assert index.triangle_count == 2
    assert index.ground_height(0.25, 0.25, 10.0) == -1.0


# ### Spatial indexing, large triangles and ownership ###
def test_large_floor_does_not_expand_into_millions_of_grid_buckets():
    large = np.asarray((((0, 0, -1), (1e6, 0, -1), (0, 1e6, -1)),), dtype=float)
    index = FirstPersonGroundIndex(large, max_cells_per_triangle=16)
    assert index.triangle_count == index.large_triangle_count == 1
    assert index.cell_count == 0
    assert index.ground_height(100.0, 100.0, 0.0) == -1.0
    assert index.ground_height(999_999.0, 999_999.0, 0.0) is None
    assert index.ground_height(-1.0, 100.0, 0.0) is None


def test_local_grid_and_large_triangle_candidates_are_combined():
    index = FirstPersonGroundIndex(
        np.concatenate((_square(z=-2.0, width=1e6), _square(z=1.0))),
        max_cells_per_triangle=16,
    )
    assert index.large_triangle_count == 2
    assert index.ground_height(0.25, 0.25, 2.0) == 1.0
    assert index.ground_height(0.25, 0.25, 0.0) == -2.0
    assert index.ground_height(100.0, 100.0, 0.0) == -2.0


def test_dense_mesh_queries_consider_only_small_local_buckets():
    index = FirstPersonGroundIndex(_grid())
    assert index.triangle_count == 20_000
    assert index.large_triangle_count == 0
    assert index.cell_count == 10_201
    assert max(len(bucket) for bucket in index._cells.values()) <= 8
    rng = np.random.default_rng(51)
    for x, y in rng.uniform(0.01, 99.99, size=(250, 2)):
        assert index.ground_height(x, y, 0.3) == 0.0


def test_input_arrays_and_meshes_are_owned_immutable_snapshots():
    triangles = _square(z=2.0)
    index = FirstPersonGroundIndex(triangles)
    triangles[:, :, 2] = 20.0
    assert index.ground_height(0.5, 0.5, math.inf) == 2.0
    assert not index._origins.flags.writeable
    box = trimesh.creation.box()
    mesh_index = FirstPersonGroundIndex(box)
    assert mesh_index.triangle_count == 4
    assert mesh_index.ground_height(0.0, 0.0, 1.0) == pytest.approx(0.5)
    assert mesh_index.ground_height(0.0, 0.0, 0.0) == pytest.approx(-0.5)


# ### Input validation and empty geometry ###
@pytest.mark.parametrize("triangles", ([], np.empty((0, 3, 3))))
def test_empty_geometry_has_no_support(triangles):
    index = FirstPersonGroundIndex(triangles)
    assert index.triangle_count == index.cell_count == index.large_triangle_count == 0
    assert index.ground_height(0.0, 0.0, 1.0) is None


@pytest.mark.parametrize("cell_size", (0.0, -1.0, math.nan, math.inf))
def test_invalid_cell_sizes_are_rejected(cell_size):
    with pytest.raises(ValueError, match="cell size"):
        FirstPersonGroundIndex([], cell_size=cell_size)


@pytest.mark.parametrize("max_cells", (0, -1, True, 1.5))
def test_invalid_bucket_limits_are_rejected(max_cells):
    with pytest.raises(ValueError, match="positive integer"):
        FirstPersonGroundIndex([], max_cells_per_triangle=max_cells)


def test_malformed_triangles_and_nonfinite_queries_are_rejected():
    with pytest.raises(ValueError, match="triangles shaped"):
        FirstPersonGroundIndex(np.zeros((2, 3)))
    index = FirstPersonGroundIndex(_square())
    with pytest.raises(ValueError, match="X and Y"):
        index.ground_height(math.inf, 0.0, 1.0)
    with pytest.raises(ValueError, match="height limits"):
        index.ground_height(0.0, 0.0, math.nan)
