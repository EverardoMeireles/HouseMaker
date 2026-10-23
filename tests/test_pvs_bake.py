"""Volume proofs stay conservative across portals, floors and unknown cameras."""

# ### Imports ###
from __future__ import annotations

import math

import numpy as np
import pytest
import shapely
from shapely.geometry import Polygon

from housemaker.pvs_bake import (
    PvsAabb,
    PvsBakeCancelled,
    PvsCameraCell,
    PvsOccluderPatch,
    PvsRenderCluster,
    _occluder_hides_bounds_batch,
    bake_pvs,
    build_camera_cells,
    build_occluder_patches,
    occluder_hides_aabb,
)


# ### Geometry fixtures ###
def _wall(y0=-5.0, y1=5.0, z0=-2.0, z1=5.0, *, x=0.0):
    return np.asarray((
        ((x, y0, z0), (x, y1, z0), (x, y1, z1)),
        ((x, y0, z0), (x, y1, z1), (x, y0, z1)),
    ), dtype=float)


def _floor(x0=-2.0, x1=2.0, y0=-2.0, y1=2.0, *, z=0.0):
    return np.asarray((
        ((x0, y0, z), (x1, y0, z), (x1, y1, z)),
        ((x0, y0, z), (x1, y1, z), (x0, y1, z)),
    ), dtype=float)


def _wall_patch():
    return PvsOccluderPatch(((0, -5, -2), (0, 5, -2), (0, 5, 5), (0, -5, 5)))


def _camera():
    return PvsAabb((-3, -0.2, 1.2), (-2, 0.2, 2.0))


def _target():
    return PvsAabb((2, -0.2, 0.5), (3, 0.2, 2.5))


# ### Convex full-volume occlusion ###
def test_solid_wall_hides_entire_target_from_entire_camera_cell():
    assert occluder_hides_aabb(_camera(), _target(), _wall_patch())
    assert occluder_hides_aabb(_target(), _camera(), _wall_patch())
    assert occluder_hides_aabb(_camera(), _target(), PvsOccluderPatch(
        tuple(reversed(_wall_patch().vertices))
    ))


@pytest.mark.parametrize("camera", (
    PvsAabb((-1, -0.2, 1), (0, 0.2, 2)),
    PvsAabb((-1, -0.2, 1), (1, 0.2, 2)),
    PvsAabb((1, -0.2, 1), (2, 0.2, 2)),
))
def test_touching_crossing_or_same_side_boxes_never_hide(camera):
    assert not occluder_hides_aabb(camera, _target(), _wall_patch())


def test_center_occlusion_is_not_enough_when_cell_corner_can_see_around_wall():
    patch = PvsOccluderPatch(((0, -0.3, 0), (0, 0.3, 0), (0, 0.3, 3), (0, -0.3, 3)))
    camera = PvsAabb((-3, -2, 1), (-2, 2, 2))
    assert not occluder_hides_aabb(camera, _target(), patch)


def test_target_extending_past_wall_and_exact_patch_edge_remain_visible():
    target = PvsAabb((2, -20, 1), (3, 20, 2))
    assert not occluder_hides_aabb(_camera(), target, _wall_patch())
    patch = PvsOccluderPatch(((0, -1, 0), (0, 1, 0), (0, 1, 3), (0, -1, 3)))
    camera = PvsAabb((-2, 1, 1), (-1, 1, 2))
    target = PvsAabb((1, 1, 1), (2, 1, 2))
    assert not occluder_hides_aabb(camera, target, patch)


def test_horizontal_slab_hides_lower_level_but_not_box_crossing_slab():
    patch = PvsOccluderPatch(((-5, -5, 3), (5, -5, 3), (5, 5, 3), (-5, 5, 3)))
    camera = PvsAabb((-1, -1, 4), (1, 1, 5))
    assert occluder_hides_aabb(camera, PvsAabb((-1, -1, 0), (1, 1, 2)), patch)
    assert not occluder_hides_aabb(camera, PvsAabb((-1, -1, 0), (1, 1, 3.1)), patch)


def test_oblique_convex_patch_supports_non_axis_aligned_walls():
    transform = np.asarray(((0.8, -0.6, 0), (0.6, 0.8, 0), (0, 0, 1)))
    vertices = np.asarray(_wall_patch().vertices) @ transform.T
    patch = PvsOccluderPatch(tuple(map(tuple, vertices)))
    assert occluder_hides_aabb(
        PvsAabb((-3, -3, 1), (-2, -2, 2)),
        PvsAabb((2, 2, 1), (3, 3, 2)), patch,
    )


def test_duplicate_adjacent_polygon_points_do_not_produce_zero_length_edges():
    patch = PvsOccluderPatch(((0, -5, -2), (0, 5, -2), (0, 5, -2),
                             (0, 5, 5), (0, -5, 5), (0, -5, -2)))
    assert len(patch.vertices) == 4
    assert occluder_hides_aabb(_camera(), _target(), patch)


# ### Coplanar grouping and hole-preserving convex decomposition ###
def test_two_wall_triangles_become_one_convex_occluder_even_reversed():
    wall = _wall()
    wall[1] = wall[1, ::-1]
    patches = build_occluder_patches((wall,))
    assert len(patches) == 1
    assert occluder_hides_aabb(_camera(), _target(), patches[0])


def test_nearly_coplanar_sheets_are_not_flattened_to_seal_a_real_crack():
    patches = build_occluder_patches((np.concatenate((
        _wall(-5, 0, x=1e-10), _wall(0, 5, x=-1e-10),
    )),))
    assert len(patches) == 2
    assert not any(occluder_hides_aabb(_camera(), _target(), patch) for patch in patches)


# ### Rendered front-side visibility and bounded vectorized proofs ###
def test_one_sided_wall_blocks_only_when_its_rendered_front_faces_the_camera():
    patches = build_occluder_patches((_wall(),), one_sided=True)
    assert len(patches) == 1
    assert patches[0].front_side == 1
    assert not occluder_hides_aabb(_camera(), _target(), patches[0])
    assert occluder_hides_aabb(_target(), _camera(), patches[0])
    result = bake_pvs((PvsCameraCell("back", _camera()),),
                      (PvsRenderCluster("target", _target()),), patches)
    assert result.hidden_pair_count == 0


def test_double_sided_material_winding_duplication_blocks_from_both_sides():
    triangles = np.concatenate((_wall(), _wall()[:, ::-1]))
    patches = build_occluder_patches((triangles,), one_sided=True)
    assert len(patches) == 2
    assert {patch.front_side for patch in patches} == {-1, 1}
    assert any(occluder_hides_aabb(_camera(), _target(), patch) for patch in patches)
    assert any(occluder_hides_aabb(_target(), _camera(), patch) for patch in patches)


def test_opposite_facing_halves_are_never_unioned_into_a_solid_front_face():
    triangles = np.concatenate((_wall(-5, 0), _wall(0, 5)[:, ::-1]))
    patches = build_occluder_patches((triangles,), one_sided=True)
    assert len(patches) == 2
    assert not any(occluder_hides_aabb(_camera(), _target(), patch) for patch in patches)
    assert not any(occluder_hides_aabb(_target(), _camera(), patch) for patch in patches)


@pytest.mark.parametrize("side", (True, 2, -2, 1.0, "front"))
def test_invalid_front_side_values_are_rejected(side):
    with pytest.raises(ValueError, match="front side"):
        PvsOccluderPatch(_wall_patch().vertices, front_side=side)


@pytest.mark.parametrize("angle", (30, 45, 60, 135))
def test_reconstructed_oblique_patch_keeps_its_original_world_front(angle):
    radians = math.radians(angle)
    rotation = np.asarray(((math.cos(radians), 0, math.sin(radians)),
                           (0, 1, 0), (-math.sin(radians), 0, math.cos(radians))))
    triangles = _wall() @ rotation.T
    patches = build_occluder_patches((triangles,), one_sided=True)
    center = rotation @ np.asarray((0, 2, 0))
    normal = rotation @ np.asarray((1, 0, 0))
    front = PvsAabb(tuple(center + normal * 3 - 0.01), tuple(center + normal * 3 + 0.01))
    back = PvsAabb(tuple(center - normal * 3 - 0.01), tuple(center - normal * 3 + 0.01))
    assert patches
    assert any(occluder_hides_aabb(front, back, patch) for patch in patches)
    assert not any(occluder_hides_aabb(back, front, patch) for patch in patches)


@pytest.mark.parametrize("side", (0, 1, -1))
def test_batched_proof_matches_independent_scalar_corner_intersections(side):
    random = np.random.default_rng(721)
    patches = (
        PvsOccluderPatch(_wall_patch().vertices, front_side=side),
        PvsOccluderPatch(((0, -5, -2), (0, 5, -2), (0, 0, 5)), front_side=side),
    )
    centers = random.uniform((-4, -8, -3), (4, 8, 6), size=(150, 3))
    extents = random.uniform(0.01, 0.6, size=(150, 3))
    targets = tuple(PvsAabb(tuple(center - extent), tuple(center + extent))
                    for center, extent in zip(centers, extents, strict=True))
    for camera in (_camera(), _target(), PvsAabb((-1, -1, 0), (1, 1, 3))):
        for patch in patches:
            expected = [occluder_hides_aabb(camera, target, patch) for target in targets]
            actual = _occluder_hides_bounds_batch(
                camera, np.asarray([target.corners for target in targets]), patch,
            )
            np.testing.assert_array_equal(actual, expected)


def test_bake_batches_many_clusters_without_changing_whole_volume_results():
    targets = tuple(PvsRenderCluster(f"target-{index}", PvsAabb(
        (2, index / 30.0 - 5, 1), (3, index / 30.0 - 4.9, 2),
    )) for index in range(400))
    patches = (_wall_patch(),)
    result = bake_pvs((PvsCameraCell("camera", _camera()),), targets, patches)
    expected = tuple(cluster.cluster_id for cluster in targets
                     if not occluder_hides_aabb(_camera(), cluster.bounds, patches[0]))
    assert result.visible_cluster_ids_by_cell == (expected,)


@pytest.mark.parametrize("gap_width", (2.0, 0.0001))
def test_window_hole_is_not_filled_even_when_extremely_small(gap_width):
    half = gap_width / 2.0
    triangles = np.concatenate((
        _wall(-5, -half), _wall(half, 5),
        _wall(-half, half, -2, 1), _wall(-half, half, 2, 5),
    ))
    patches = build_occluder_patches((triangles,))
    camera = PvsAabb((-3, -0.01, 1.45), (-2, 0.01, 1.55))
    target = PvsAabb((2, -0.01, 1.45), (3, 0.01, 1.55))
    assert patches
    assert not any(occluder_hides_aabb(camera, target, patch) for patch in patches)
    for patch in patches:
        polygon = Polygon([(point[1], point[2]) for point in patch.vertices])
        assert polygon.intersection(shapely.box(-half, 1, half, 2)).area < 1e-12


def test_door_portal_does_not_block_but_neighboring_solid_wall_does():
    triangles = np.concatenate((_wall(-5, -0.5), _wall(0.5, 5), _wall(-0.5, 0.5, 2.2, 5)))
    patches = build_occluder_patches((triangles,))
    assert not any(occluder_hides_aabb(_camera(), _target(), patch) for patch in patches)
    side_camera = PvsAabb((-3, 2, 1), (-2, 2.2, 2))
    side_target = PvsAabb((2, 2, 1), (3, 2.2, 2))
    assert any(occluder_hides_aabb(side_camera, side_target, patch) for patch in patches)


def test_floor_opening_is_preserved_by_patch_union():
    triangles = np.concatenate((
        _floor(-5, -1, -5, 5, z=3), _floor(1, 5, -5, 5, z=3),
        _floor(-1, 1, -5, -1, z=3), _floor(-1, 1, 1, 5, z=3),
    ))
    patches = build_occluder_patches((triangles,))
    assert not any(occluder_hides_aabb(
        PvsAabb((-0.1, -0.1, 4), (0.1, 0.1, 5)),
        PvsAabb((-0.1, -0.1, 1), (0.1, 0.1, 2)), patch,
    ) for patch in patches)


def test_missing_glass_or_movable_occluders_cannot_hide_anything():
    result = bake_pvs((PvsCameraCell("camera", _camera()),),
                      (PvsRenderCluster("object", _target()),), ())
    assert result.visible_cluster_ids_by_cell == (("object",),)
    assert result.hidden_pair_count == 0


# ### Camera-cell coverage and multiple levels ###
def test_floor_tiles_use_physical_eye_height_and_keep_negative_coordinates():
    cells = build_camera_cells(_floor(), cell_size=2, eye_height=1.8, eye_margin=0.2)
    assert len(cells) == 4
    assert cells[0].bounds.minimum == (-2.0, -2.0, 1.6)
    assert cells[0].bounds.maximum == (0.0, 0.0, 2.0)
    assert any(cell.bounds.contains((-0.5, -0.5, 1.8)) for cell in cells)


def test_stacked_floors_get_separate_camera_bands_and_stairs_span_tread_heights():
    triangles = np.concatenate((_floor(), _floor(z=3),
                                _floor(x0=3, x1=3.5, y0=0, y1=1, z=0.2),
                                _floor(x0=3.5, x1=4, y0=0, y1=1, z=0.4)))
    cells = build_camera_cells(triangles, cell_size=2)
    assert any(cell.bounds.contains((0.5, 0.5, 1.7)) for cell in cells)
    assert any(cell.bounds.contains((0.5, 0.5, 4.7)) for cell in cells)
    assert not any(cell.bounds.contains((0.5, 0.5, 3.1)) for cell in cells)
    assert any(cell.bounds.contains((3.25, 0.5, 2.0)) for cell in cells)


def test_sloped_walkable_triangle_covers_the_full_clipped_height_range():
    triangles = np.asarray((((0, 0, 0), (2, 0, 2), (0, 2, 0)),), dtype=float)
    cells = build_camera_cells(triangles, cell_size=1, eye_margin=0.1)
    cell = next(cell for cell in cells if cell.bounds.minimum[:2] == (1.0, 0.0))
    assert cell.bounds.minimum[2] == pytest.approx(2.6)
    assert cell.bounds.maximum[2] == pytest.approx(3.8)


def test_empty_or_vertical_walkable_mesh_has_no_coverage():
    assert build_camera_cells([]) == ()
    assert build_camera_cells(_wall()) == ()


def test_excess_camera_cells_are_explicit_error_not_silent_truncation():
    with pytest.raises(ValueError, match="maximum cell count"):
        build_camera_cells(_floor(-1000, 1000, -1000, 1000), max_cells=10)
    with pytest.raises(ValueError, match="maximum cell count"):
        build_camera_cells(np.concatenate((_floor(), _floor(z=3))), max_cells=4)


# ### Bake result, cancellation and fail-open lookup ###
def test_result_culls_only_proved_clusters_and_fails_open_outside_cells():
    clusters = (PvsRenderCluster("behind", _target()),
                PvsRenderCluster("same-side", _camera()))
    result = bake_pvs((PvsCameraCell("camera", _camera()),), clusters, (_wall_patch(),))
    assert result.visible_cluster_ids_by_cell == (("same-side",),)
    assert result.hidden_pair_count == 1
    assert result.visible_cluster_ids_at((-2.5, 0, 1.6)) == ("same-side",)
    assert result.visible_cluster_ids_at((10, 10, 1.6)) == ("behind", "same-side")
    assert bake_pvs((), clusters, ()).visible_cluster_ids_at((0, 0, 0)) == (
        "behind", "same-side"
    )


def test_touching_and_overlapping_cells_union_their_visibility():
    first = PvsCameraCell("first", PvsAabb((-3, -1, 1), (-2, 1, 2)))
    second = PvsCameraCell("second", PvsAabb((-2, -1, 1), (1, 1, 2)))
    result = bake_pvs((first, second), (PvsRenderCluster("target", _target()),), (_wall_patch(),))
    assert result.visible_cluster_ids_by_cell == ((), ("target",))
    assert result.visible_cluster_ids_at((-2, 0, 1.5)) == ("target",)


def test_progress_reports_completed_pairs_and_cancel_never_returns_partial_result():
    progress = []
    result = bake_pvs((PvsCameraCell("camera", _camera()),),
                      (PvsRenderCluster("target", _target()),), (_wall_patch(),),
                      progress=lambda done, total: progress.append((done, total)))
    assert result.hidden_pair_count == 1
    assert progress == [(0, 1), (1, 1)]
    for operation in (
        lambda: bake_pvs((), (), (), is_cancelled=lambda: True),
        lambda: build_camera_cells(_floor(), is_cancelled=lambda: True),
        lambda: build_occluder_patches((_wall(),), is_cancelled=lambda: True),
    ):
        with pytest.raises(PvsBakeCancelled):
            operation()


def test_cancellation_after_processing_started_never_yields_partial_visibility():
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        return calls > 2

    with pytest.raises(PvsBakeCancelled):
        bake_pvs((PvsCameraCell("camera", _camera()),),
                 (PvsRenderCluster("target", _target()),),
                 (_wall_patch(), _wall_patch(), _wall_patch()),
                 is_cancelled=cancelled)


@pytest.mark.parametrize("kwargs", (
    {"cell_size": 0}, {"cell_size": -1}, {"cell_size": math.inf},
    {"eye_height": 0}, {"eye_margin": -1}, {"max_cells": 0},
))
def test_invalid_camera_grid_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        build_camera_cells(_floor(), **kwargs)


# ### Malformed input and bounded occluder work ###
@pytest.mark.parametrize("minimum,maximum", (
    ((0, 0, 0), (-1, 1, 1)), ((0, math.nan, 0), (1, 1, 1)),
    ((0, 0, 0), (1, 1, math.inf)), ((0, 0), (1, 1, 1)),
))
def test_invalid_or_inverted_scaled_bounds_are_rejected(minimum, maximum):
    with pytest.raises(ValueError):
        PvsAabb(minimum, maximum)


@pytest.mark.parametrize("vertices", (
    ((0, 0, 0), (0, 0, 0), (0, 0, 0)),
    ((0, 0, 0), (1, 0, 0), (1, 1, 1), (0, 1, 0)),
    ((0, 0, 0), (2, 0, 0), (1, 0.2, 0), (2, 1, 0), (0, 1, 0)),
))
def test_degenerate_nonplanar_or_concave_occluder_is_rejected(vertices):
    with pytest.raises(ValueError):
        PvsOccluderPatch(vertices)


def test_nonfinite_geometry_is_rejected_and_degenerate_faces_are_ignored():
    with pytest.raises(ValueError, match="finite"):
        build_occluder_patches((np.full((1, 3, 3), math.nan),))
    assert build_occluder_patches((np.zeros((2, 3, 3)),)) == ()
    assert len(build_occluder_patches((_wall(), _wall(x=5)), max_planes=1)) == 1
    assert len(build_occluder_patches((_wall(), _wall(x=5)), max_patches=1)) == 1


def test_duplicate_cluster_or_cell_ids_are_rejected():
    cluster = PvsRenderCluster("same", _target())
    cell = PvsCameraCell("same", _camera())
    with pytest.raises(ValueError, match="unique"):
        bake_pvs((cell,), (cluster, cluster), ())
    with pytest.raises(ValueError, match="unique"):
        bake_pvs((cell, cell), (cluster,), ())
