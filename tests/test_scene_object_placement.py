"""Pure footprint-aware placement stays compact, ordered and non-overlapping."""

# ### Imports ###
from __future__ import annotations

import math

import numpy as np
import pytest

from housemaker.scene_object_placement import build_object_placement_grid_offsets


# ### Bounds helpers ###
def _translated(bounds, offset):
    return (
        bounds[0] + offset[0], bounds[1] + offset[1],
        bounds[2] + offset[0], bounds[3] + offset[1],
    )


def _center(bounds):
    return (bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0


def _assert_separated(bounds, *, gap=0.25):
    for index, first in enumerate(bounds):
        for second in bounds[index + 1 :]:
            separations = (
                first[0] - second[2], second[0] - first[2],
                first[1] - second[3], second[1] - first[3],
            )
            assert max(separations) >= gap - 1e-10


# ### Grid placement and stable input order ###
def test_single_centered_footprint_stays_at_anchor():
    assert build_object_placement_grid_offsets(((-1.0, -2.0, 1.0, 2.0),)) == (
        (0.0, 0.0, 0.0),
    )


def test_single_offset_footprint_is_centered_not_its_original_pivot():
    assert build_object_placement_grid_offsets(((2.0, -3.0, 6.0, 1.0),)) == (
        (-4.0, 1.0, 0.0),
    )


def test_two_different_widths_touch_their_cell_edges_with_exact_gap():
    bounds = ((-2.0, 0.0, 2.0, 2.0), (0.0, 0.0, 2.0, 1.0))
    offsets = build_object_placement_grid_offsets(bounds)
    np.testing.assert_allclose(
        offsets, ((-1.125, -1.0, 0.0), (1.125, -0.5, 0.0)),
    )
    moved = tuple(_translated(bound, offset) for bound, offset in zip(bounds, offsets))
    assert moved[1][0] - moved[0][2] == pytest.approx(0.25)
    assert moved[0][0] == pytest.approx(-moved[1][2])
    assert _center(moved[0])[1] == _center(moved[1])[1] == 0.0


def test_row_major_grid_uses_maximum_width_per_column_and_depth_per_row():
    bounds = (
        (-1.0, -0.5, 1.0, 0.5),
        (-0.5, -1.5, 0.5, 1.5),
        (-2.5, -1.0, 2.5, 1.0),
        (-2.0, -0.25, 2.0, 0.25),
        (-1.5, -1.0, 1.5, 1.0),
    )
    offsets = build_object_placement_grid_offsets(bounds, gap_meters=0.5)
    # Five objects use three columns (widths 4/3/5) and two rows (depths 3/2).
    expected = ((-4.5, -1.25), (-0.5, -1.25), (4.0, -1.25), (-4.5, 1.75), (-0.5, 1.75))
    np.testing.assert_allclose(np.asarray(offsets)[:, :2], expected)
    _assert_separated(
        tuple(_translated(bound, offset) for bound, offset in zip(bounds, offsets)),
        gap=0.5,
    )
    assert offsets == build_object_placement_grid_offsets(bounds, gap_meters=0.5)


def test_input_collection_and_bounds_are_not_modified():
    bounds = np.asarray(((0.0, 0.0, 1.0, 2.0), (3.0, -2.0, 6.0, 2.0)))
    original = bounds.copy()
    assert isinstance(build_object_placement_grid_offsets(bounds), tuple)
    np.testing.assert_array_equal(bounds, original)


@pytest.mark.parametrize("count", (3, 4, 5, 8, 9, 10, 30, 100))
def test_variable_footprints_never_overlap_and_return_one_offset_per_object(count):
    random = np.random.default_rng(count)
    lower = random.uniform(-5.0, 5.0, (count, 2))
    upper = lower + random.uniform(0.01, 5.0, (count, 2))
    bounds = tuple((*minimum, *maximum) for minimum, maximum in zip(lower, upper))
    offsets = build_object_placement_grid_offsets(bounds)
    assert len(offsets) == count
    assert all(offset[2] == 0.0 for offset in offsets)
    _assert_separated(tuple(_translated(bound, offset) for bound, offset in zip(bounds, offsets)))


def test_already_rotated_mirrored_and_nonuniformly_scaled_physical_bounds_fit_grid():
    # Reconstruct a full footprint from a half model before applying its scale
    # and rotation. The helper must honor these physical extents, not unit tiles.
    half = np.asarray(((0.0, -0.5), (2.0, -0.5), (2.0, 0.5), (0.0, 0.5)))
    mirrored = half * (-1.0, 1.0)
    full = np.vstack((half, mirrored)) * (2.0, 3.0)
    angle = math.radians(35.0)
    rotation = np.asarray(((math.cos(angle), -math.sin(angle)),
                           (math.sin(angle), math.cos(angle))))
    transformed = full @ rotation.T + (0.7, -0.3)
    transformed_bounds = (*transformed.min(axis=0), *transformed.max(axis=0))
    bounds = (transformed_bounds, (-1.0, -1.0, 1.0, 1.0))
    offsets = build_object_placement_grid_offsets(bounds)
    _assert_separated(tuple(_translated(bound, offset) for bound, offset in zip(bounds, offsets)))
    assert offsets[0][1] == pytest.approx(0.3)


# ### Numeric validation and degenerate footprints ###
def test_zero_extents_remain_valid_and_get_gap_separation():
    offsets = build_object_placement_grid_offsets(((2.0, 3.0, 2.0, 3.0),) * 2)
    np.testing.assert_allclose(offsets, ((-2.125, -3.0, 0.0), (-1.875, -3.0, 0.0)))


@pytest.mark.parametrize("bounds", ((), "invalid", None, ((0.0, 0.0, 1.0),),
                                     ((1.0, 0.0, 0.0, 1.0),),
                                     ((0.0, 2.0, 1.0, 1.0),),
                                     ((False, 0.0, 1.0, 1.0),),
                                     ((0.0, 0.0, math.inf, 1.0),),
                                     ((0.0, math.nan, 1.0, 1.0),),
                                     ((-1e308, 0.0, 1e308, 1.0),)))
def test_invalid_footprints_fail_clearly(bounds):
    with pytest.raises((TypeError, ValueError)):
        build_object_placement_grid_offsets(bounds)


@pytest.mark.parametrize("gap", (0.0, -0.1, math.inf, math.nan, True, None, "bad"))
def test_gap_requires_a_positive_finite_number(gap):
    with pytest.raises((TypeError, ValueError)):
        build_object_placement_grid_offsets(((0.0, 0.0, 1.0, 1.0),), gap_meters=gap)


def test_finite_inputs_that_overflow_the_grid_fail_instead_of_returning_infinity():
    with pytest.raises(ValueError, match="grid extents"):
        build_object_placement_grid_offsets(((0.0, 0.0, 1e308, 1.0),) * 2)


def test_finite_grid_with_overflowing_pivot_offset_is_rejected():
    bounds = ((1.6e308, 0.0, 1.7e308, 1.0), (0.0, 0.0, 1.5e308, 1.0))
    with pytest.raises(ValueError, match="offsets"):
        build_object_placement_grid_offsets(bounds)
