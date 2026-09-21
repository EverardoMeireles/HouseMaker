# ### Imports ###
from __future__ import annotations

import unittest

import numpy as np

from housemaker.pbr_maps import (
    ATLAS_MAP_BASE_COLOR,
    PBR_MAP_METALLIC,
    PBR_MAP_NORMAL,
    PBR_MAP_ROUGHNESS,
)
from housemaker.texture_tiling import (
    TilingRepairOperation,
    TilingRepairPlan,
    apply_tiling_repair_plan,
    build_tiling_repair_plan,
    create_edge_compatible_variants,
    repair_texture_tiling,
    texture_tiling_seam_score,
)


# ### Fixture helpers ###
def _seamed_color_texture(height: int = 80, width: int = 112) -> np.ndarray:
    y_coordinates, x_coordinates = np.mgrid[:height, :width]
    x_fraction = x_coordinates.astype(np.float32) / max(width - 1, 1)
    y_fraction = y_coordinates.astype(np.float32) / max(height - 1, 1)
    material = np.empty((height, width, 4), dtype=np.uint8)
    material[:, :, 0] = np.clip(
        np.rint(25.0 + x_fraction * 205.0),
        0,
        255,
    ).astype(np.uint8)
    material[:, :, 1] = np.clip(
        np.rint(30.0 + y_fraction * 190.0),
        0,
        255,
    ).astype(np.uint8)
    material[:, :, 2] = np.clip(
        np.rint(
            100.0
            + 35.0 * np.sin(x_coordinates * np.pi / 9.0)
            + 20.0 * np.cos(y_coordinates * np.pi / 7.0)
        ),
        0,
        255,
    ).astype(np.uint8)
    material[:, :, 3] = 255
    return material


def _detailed_normal_texture(height: int, width: int) -> np.ndarray:
    y_coordinates, x_coordinates = np.mgrid[:height, :width]
    normal_x = 0.48 * np.sin(x_coordinates * 2.0 * np.pi / max(width, 1))
    normal_y = 0.38 * np.cos(y_coordinates * 2.0 * np.pi / max(height, 1))
    normal_z = np.sqrt(np.maximum(1.0 - normal_x**2 - normal_y**2, 0.0))
    vectors = np.stack((normal_x, normal_y, normal_z), axis=2)
    normal = np.empty((height, width, 4), dtype=np.uint8)
    normal[:, :, :3] = np.clip(
        np.rint((vectors * 0.5 + 0.5) * 255.0),
        0,
        255,
    ).astype(np.uint8)
    normal[:, :, 3] = 255
    return normal


def _seamed_normal_texture(height: int, width: int) -> np.ndarray:
    y_coordinates, x_coordinates = np.mgrid[:height, :width]
    normal_x = -0.55 + 1.1 * x_coordinates / max(width - 1, 1)
    normal_y = -0.35 + 0.7 * y_coordinates / max(height - 1, 1)
    normal_z = np.sqrt(np.maximum(1.0 - normal_x**2 - normal_y**2, 0.0))
    vectors = np.stack((normal_x, normal_y, normal_z), axis=2)
    normal = np.empty((height, width, 4), dtype=np.uint8)
    normal[:, :, :3] = np.clip(
        np.rint((vectors * 0.5 + 0.5) * 255.0),
        0,
        255,
    ).astype(np.uint8)
    normal[:, :, 3] = 255
    return normal


def _seamed_roughness_texture(height: int, width: int) -> np.ndarray:
    _y_coordinates, x_coordinates = np.mgrid[:height, :width]
    values = np.clip(
        np.rint(20.0 + x_coordinates * 210.0 / max(width - 1, 1)),
        0,
        255,
    ).astype(np.uint8)
    roughness = np.empty((height, width, 4), dtype=np.uint8)
    roughness[:, :, :3] = values[:, :, None]
    roughness[:, :, 3] = 255
    return roughness


def _decoded_normal_lengths(normal: np.ndarray) -> np.ndarray:
    vectors = normal[:, :, :3].astype(np.float32) / 127.5 - 1.0
    return np.linalg.norm(vectors, axis=2)


def _wrapped_seam_gradient_jump(
    pixels: np.ndarray, *, axis: int, split: int
) -> float:
    """Measure derivative discontinuity, including a sheet's wrapped edge."""

    positions = np.asarray([split - 2, split - 1, split, split + 1])
    strip = np.take(pixels.astype(np.float32), positions, axis=axis, mode="wrap")
    slopes = np.diff(strip, axis=axis)
    before = np.take(slopes, 0, axis=axis)
    across = np.take(slopes, 1, axis=axis)
    after = np.take(slopes, 2, axis=axis)
    return float(np.mean(np.abs(across - before) + np.abs(after - across)))


# ### Repair behavior tests ###
class TextureTilingRepairTests(unittest.TestCase):
    def test_prominent_horizontal_and_vertical_seams_are_reduced(self) -> None:
        source = _seamed_color_texture()
        original = source.copy()

        result = repair_texture_tiling({ATLAS_MAP_BASE_COLOR: source})

        repaired = result.maps[ATLAS_MAP_BASE_COLOR]
        np.testing.assert_array_equal(source, original)
        self.assertEqual(repaired.shape, source.shape)
        self.assertEqual(repaired.dtype, np.uint8)
        self.assertLess(result.seam_score_after, result.seam_score_before * 0.08)
        self.assertEqual(
            result.seam_score_before,
            texture_tiling_seam_score(source),
        )
        self.assertTrue(result.plan.operations)
        self.assertTrue(
            all(
                operation.quarter_turns in {1, 2, 3}
                for operation in result.plan.operations
            )
        )
        np.testing.assert_array_equal(repaired[:, 0], repaired[:, -1])
        np.testing.assert_array_equal(repaired[0], repaired[-1])

    def test_repair_is_seeded_and_deterministic(
        self,
    ) -> None:
        source = _seamed_color_texture(height=72, width=104)

        first = repair_texture_tiling({ATLAS_MAP_BASE_COLOR: source}, seed=17)
        second = repair_texture_tiling({ATLAS_MAP_BASE_COLOR: source}, seed=17)
        different = repair_texture_tiling({ATLAS_MAP_BASE_COLOR: source}, seed=18)

        np.testing.assert_array_equal(
            first.maps[ATLAS_MAP_BASE_COLOR],
            second.maps[ATLAS_MAP_BASE_COLOR],
        )
        self.assertEqual(first.plan, second.plan)
        self.assertNotEqual(first.plan.operations, different.plan.operations)
        self.assertFalse(
            np.array_equal(
                first.maps[ATLAS_MAP_BASE_COLOR],
                different.maps[ATLAS_MAP_BASE_COLOR],
            )
        )
        self.assertEqual(first.plan.offset_y, 0)
        self.assertEqual(first.plan.offset_x, 0)

    def test_already_seamless_texture_is_not_made_worse(self) -> None:
        source = np.full((48, 64, 4), (75, 105, 135, 255), dtype=np.uint8)

        result = repair_texture_tiling({ATLAS_MAP_BASE_COLOR: source})

        np.testing.assert_array_equal(
            result.maps[ATLAS_MAP_BASE_COLOR],
            source,
        )
        self.assertEqual(result.plan.offset_y, 0)
        self.assertEqual(result.plan.offset_x, 0)
        self.assertEqual(result.plan.operations, ())
        self.assertEqual(result.seam_score_after, result.seam_score_before)

    def test_periodic_source_gets_randomized_rotations(self) -> None:
        height, width = 96, 144
        y_coordinates, x_coordinates = np.mgrid[:height, :width]
        values = np.clip(
            np.rint(
                128.0
                + 55.0 * np.sin(2.0 * np.pi * x_coordinates / 12.0)
                + 38.0 * np.cos(2.0 * np.pi * y_coordinates / 16.0)
            ),
            0,
            255,
        ).astype(np.uint8)
        source = np.empty((height, width, 4), dtype=np.uint8)
        source[:, :, :3] = values[:, :, None]
        source[:, :, 3] = 255

        result = repair_texture_tiling({ATLAS_MAP_BASE_COLOR: source})

        self.assertTrue(result.plan.operations)
        self.assertTrue(
            any(
                operation.quarter_turns in {1, 3}
                for operation in result.plan.operations
            )
        )
        self.assertFalse(np.array_equal(result.maps[ATLAS_MAP_BASE_COLOR], source))
        self.assertLessEqual(result.seam_score_after, result.seam_score_before)

    def test_small_texture_gets_a_turn_even_when_seed_draws_zero(self) -> None:
        source = _seamed_color_texture(height=16, width=16)

        plan = build_tiling_repair_plan(source, seed=11)

        self.assertEqual(len(plan.operations), 1)
        self.assertIn(plan.operations[0].quarter_turns, {1, 2, 3})

    def test_one_plan_is_reused_for_every_aligned_map(self) -> None:
        base_color = _seamed_color_texture(height=64, width=96)
        roughness = np.repeat(base_color[:, :, :1], 4, axis=2)
        roughness[:, :, 3] = 255
        maps = {
            ATLAS_MAP_BASE_COLOR: base_color,
            PBR_MAP_ROUGHNESS: roughness,
        }

        result = repair_texture_tiling(maps)

        for map_type, source in maps.items():
            expected = apply_tiling_repair_plan(
                source,
                map_type,
                result.plan,
            )
            np.testing.assert_array_equal(result.maps[map_type], expected)
            self.assertEqual(result.maps[map_type].shape, source.shape)
            self.assertEqual(result.maps[map_type].dtype, np.uint8)

    def test_quarter_turn_rotates_tangent_normal_xy_without_resampling(self) -> None:
        normal_x, normal_y = 0.6, 0.2
        normal_z = np.sqrt(1.0 - normal_x**2 - normal_y**2)
        encoded = np.clip(
            np.rint((np.asarray((normal_x, normal_y, normal_z)) * 0.5 + 0.5) * 255.0),
            0,
            255,
        ).astype(np.uint8)
        source = np.empty((48, 48, 4), dtype=np.uint8)
        source[:, :, :3] = encoded
        source[:, :, 3] = 255
        plan = TilingRepairPlan(
            height=48,
            width=48,
            offset_y=0,
            offset_x=0,
            estimated_period_y=None,
            estimated_period_x=None,
            operations=(TilingRepairOperation(16, 16, 16, 1, 2),),
        )

        repaired = apply_tiling_repair_plan(source, PBR_MAP_NORMAL, plan)

        expected = np.clip(
            np.rint((np.asarray((-normal_y, normal_x, normal_z)) * 0.5 + 0.5) * 255.0),
            0,
            255,
        ).astype(np.uint8)
        np.testing.assert_allclose(repaired[24, 24, :3], expected, atol=1)
        np.testing.assert_array_equal(repaired[0, 0], source[0, 0])
        self.assertLess(
            float(np.max(np.abs(_decoded_normal_lengths(repaired) - 1.0))),
            0.012,
        )

    def test_seamed_normal_drives_repair_when_base_color_is_solid(self) -> None:
        height, width = 72, 104
        base_color = np.full(
            (height, width, 4),
            (90, 110, 130, 255),
            dtype=np.uint8,
        )
        normal = _seamed_normal_texture(height, width)
        normal_score_before = texture_tiling_seam_score(
            normal,
            map_type=PBR_MAP_NORMAL,
        )

        result = repair_texture_tiling(
            {
                ATLAS_MAP_BASE_COLOR: base_color,
                PBR_MAP_NORMAL: normal,
            }
        )

        repaired_normal = result.maps[PBR_MAP_NORMAL]
        self.assertNotEqual(result.plan.operations, ())
        np.testing.assert_array_equal(
            result.maps[ATLAS_MAP_BASE_COLOR],
            base_color,
        )
        self.assertLess(
            texture_tiling_seam_score(
                repaired_normal,
                map_type=PBR_MAP_NORMAL,
            ),
            normal_score_before * 0.05,
        )
        self.assertLess(result.seam_score_after, result.seam_score_before)
        self.assertLess(
            float(np.max(np.abs(_decoded_normal_lengths(repaired_normal) - 1.0))),
            0.012,
        )

    def test_seamed_roughness_drives_repair_when_base_color_is_solid(
        self,
    ) -> None:
        height, width = 68, 100
        base_color = np.full(
            (height, width, 4),
            (80, 100, 120, 255),
            dtype=np.uint8,
        )
        roughness = _seamed_roughness_texture(height, width)
        roughness_score_before = texture_tiling_seam_score(
            roughness,
            map_type=PBR_MAP_ROUGHNESS,
        )

        result = repair_texture_tiling(
            {
                ATLAS_MAP_BASE_COLOR: base_color,
                PBR_MAP_ROUGHNESS: roughness,
            }
        )

        self.assertNotEqual(result.plan.operations, ())
        np.testing.assert_array_equal(
            result.maps[ATLAS_MAP_BASE_COLOR],
            base_color,
        )
        self.assertLess(
            texture_tiling_seam_score(
                result.maps[PBR_MAP_ROUGHNESS],
                map_type=PBR_MAP_ROUGHNESS,
            ),
            roughness_score_before * 0.05,
        )
        self.assertLess(result.seam_score_after, result.seam_score_before)

    def test_normal_vectors_are_renormalized_after_soft_blending(self) -> None:
        base_color = _seamed_color_texture(height=68, width=92)
        normal = _detailed_normal_texture(68, 92)

        result = repair_texture_tiling(
            {
                ATLAS_MAP_BASE_COLOR: base_color,
                PBR_MAP_NORMAL: normal,
            }
        )

        repaired_normal = result.maps[PBR_MAP_NORMAL]
        lengths = _decoded_normal_lengths(repaired_normal)
        self.assertLess(float(np.max(np.abs(lengths - 1.0))), 0.012)
        self.assertTrue(np.all(repaired_normal[:, :, 3] == 255))
        self.assertEqual(repaired_normal.shape, normal.shape)

    def test_metallic_rotation_does_not_create_intermediate_material_values(
        self,
    ) -> None:
        height, width = 76, 100
        base_color = _seamed_color_texture(height, width)
        y_coordinates, x_coordinates = np.mgrid[:height, :width]
        metallic_value = (((x_coordinates // 9) + (y_coordinates // 7)) % 2) * 255
        metallic = np.empty((height, width, 4), dtype=np.uint8)
        metallic[:, :, :3] = metallic_value[:, :, None]
        metallic[:, :, 3] = 255

        repaired = repair_texture_tiling(
            {
                ATLAS_MAP_BASE_COLOR: base_color,
                PBR_MAP_METALLIC: metallic,
            }
        ).maps[PBR_MAP_METALLIC]

        self.assertEqual(set(np.unique(repaired[:, :, :3])), {0, 255})
        self.assertTrue(np.all(repaired[:, :, 3] == 255))

    def test_cancellation_stops_between_aligned_map_applications(self) -> None:
        base_color = _seamed_color_texture(height=64, width=96)
        roughness = np.repeat(base_color[:, :, :1], 4, axis=2)
        roughness[:, :, 3] = 255
        original_base_color = base_color.copy()
        original_roughness = roughness.copy()
        cancellation_checks = 0

        def cancellation_check() -> bool:
            nonlocal cancellation_checks
            cancellation_checks += 1
            return cancellation_checks >= 8

        with self.assertRaisesRegex(InterruptedError, "cancelled"):
            repair_texture_tiling(
                {
                    ATLAS_MAP_BASE_COLOR: base_color,
                    PBR_MAP_ROUGHNESS: roughness,
                },
                cancellation_check=cancellation_check,
            )

        self.assertGreaterEqual(cancellation_checks, 8)
        np.testing.assert_array_equal(base_color, original_base_color)
        np.testing.assert_array_equal(roughness, original_roughness)


# ### Edge-compatible variant tests ###
class EdgeCompatibleVariantTests(unittest.TestCase):
    def test_full_tile_rotations_remove_the_common_unrotated_border(self) -> None:
        size = 64
        source = np.full((size, size, 4), 96, dtype=np.uint8)
        source[:, :, 3] = 255
        source[3, 11, :3] = 255
        result = create_edge_compatible_variants(
            {ATLAS_MAP_BASE_COLOR: source}, seed=11
        )

        self.assertEqual(set(result.quarter_turns), {0, 1, 2, 3})
        for index, turns in enumerate(result.quarter_turns):
            row, column = divmod(index, 2)
            actual = result.maps[ATLAS_MAP_BASE_COLOR][
                row * size : (row + 1) * size,
                column * size : (column + 1) * size,
            ]
            expected = np.rot90(source, turns)
            spot_row, spot_column = np.argwhere(expected[:, :, 0] == 255)[0]
            self.assertGreaterEqual(int(actual[spot_row, spot_column, 0]), 200)
            if turns:
                self.assertLess(int(actual[3, 11, 0]), 160)

    def test_full_tile_joins_reduce_native_and_mip_seams(self) -> None:
        size = 128
        color = _seamed_color_texture(size, size)
        roughness = _seamed_roughness_texture(size, size)
        result = create_edge_compatible_variants(
            {ATLAS_MAP_BASE_COLOR: color, PBR_MAP_ROUGHNESS: roughness},
            seed=23,
        )

        for map_type, source in (
            (ATLAS_MAP_BASE_COLOR, color),
            (PBR_MAP_ROUGHNESS, roughness),
        ):
            sheet = result.maps[map_type].astype(np.float32)
            raw = np.empty_like(sheet)
            for index, turns in enumerate(result.quarter_turns):
                row, column = divmod(index, 2)
                raw[
                    row * size : (row + 1) * size,
                    column * size : (column + 1) * size,
                ] = np.rot90(source, turns)
            for factor in (1, 2, 4, 8):
                mip_size = size * 2 // factor
                sampled = sheet.reshape(
                    mip_size, factor, mip_size, factor, source.shape[2]
                ).mean(axis=(1, 3))
                raw_sampled = raw.reshape(
                    mip_size, factor, mip_size, factor, source.shape[2]
                ).mean(axis=(1, 3))
                half = size // factor
                for axis, split in ((1, half), (1, 0), (0, half), (0, 0)):
                    before = _wrapped_seam_gradient_jump(
                        raw_sampled, axis=axis, split=split
                    )
                    if before > 1.0:
                        self.assertLess(
                            _wrapped_seam_gradient_jump(
                                sampled, axis=axis, split=split
                            ),
                            before * 0.8,
                            f"{map_type}, mip {factor}, axis {axis}, seam {split}",
                        )

    def test_noisy_edges_do_not_become_mip_scale_stripes(self) -> None:
        size = 512
        axis = np.linspace(0.0, 1.0, size, dtype=np.float32)
        grain = np.random.default_rng(41).integers(
            -10, 11, (size, size), dtype=np.int16
        )
        gray = np.clip(
            np.rint(118 + 32 * axis[None, :] + 14 * axis[:, None] + grain),
            0,
            255,
        ).astype(np.uint8)
        color = np.repeat(gray[:, :, None], 3, axis=2)
        sheet = create_edge_compatible_variants(
            {ATLAS_MAP_BASE_COLOR: color}, seed=41
        ).maps[ATLAS_MAP_BASE_COLOR]

        for factor in (1, 4, 8, 16):
            mip_size = size * 2 // factor
            pixels = sheet.reshape(
                mip_size, factor, mip_size, factor, 3
            ).mean(axis=(1, 3))
            half = mip_size // 2
            for axis_index, split in (
                (1, 0), (1, half), (0, 0), (0, half)
            ):
                seam_step = np.mean(
                    np.abs(
                        np.take(pixels, split, axis=axis_index, mode="wrap")
                        - np.take(
                            pixels, split - 1, axis=axis_index, mode="wrap"
                        )
                    )
                )
                interior_step = np.mean(
                    np.abs(
                        np.take(
                            pixels, half // 2, axis=axis_index, mode="wrap"
                        )
                        - np.take(
                            pixels, half // 2 - 1,
                            axis=axis_index, mode="wrap"
                        )
                    )
                )
                self.assertLessEqual(
                    seam_step,
                    interior_step * 1.5 + 1.0,
                    f"mip {factor}, axis {axis_index}, seam {split}",
                )

    def test_non_square_sources_use_full_half_turns_without_resampling(self) -> None:
        source = _seamed_color_texture(64, 96)
        result = create_edge_compatible_variants(
            {ATLAS_MAP_BASE_COLOR: source}, seed=9
        )
        self.assertEqual(set(result.quarter_turns), {0, 2})
        self.assertEqual(result.maps[ATLAS_MAP_BASE_COLOR].shape, (128, 192, 4))

    def test_rotated_interior_normals_follow_each_quarter_turn(self) -> None:
        height = width = 72
        color = _seamed_color_texture(height, width)
        normal_x, normal_y = 0.6, 0.2
        normal_z = np.sqrt(1.0 - normal_x**2 - normal_y**2)
        encoded = np.clip(
            np.rint((np.asarray((normal_x, normal_y, normal_z)) * 0.5 + 0.5) * 255.0),
            0,
            255,
        ).astype(np.uint8)
        normal = np.empty((height, width, 4), dtype=np.uint8)
        normal[:, :, :3] = encoded
        normal[:, :, 3] = 255

        result = create_edge_compatible_variants(
            {ATLAS_MAP_BASE_COLOR: color, PBR_MAP_NORMAL: normal}, seed=7
        )

        directions = (
            (normal_x, normal_y, normal_z),
            (-normal_y, normal_x, normal_z),
            (-normal_x, -normal_y, normal_z),
            (normal_y, -normal_x, normal_z),
        )
        for index, turns in enumerate(result.quarter_turns):
            row, column = divmod(index, 2)
            center = result.maps[PBR_MAP_NORMAL][
                row * height + height // 2, column * width + width // 2, :3
            ]
            expected = np.clip(
                np.rint((np.asarray(directions[turns]) * 0.5 + 0.5) * 255.0),
                0,
                255,
            ).astype(np.uint8)
            np.testing.assert_allclose(center, expected, atol=1)

    def test_seed_and_shared_transform_are_deterministic_for_aligned_maps(self) -> None:
        color = _seamed_color_texture(64, 64)
        roughness = _seamed_roughness_texture(64, 64)
        maps = {
            ATLAS_MAP_BASE_COLOR: color,
            PBR_MAP_ROUGHNESS: roughness,
            "height": roughness.copy(),
        }

        first = create_edge_compatible_variants(maps, seed=17)
        again = create_edge_compatible_variants(maps, seed=17)
        different = create_edge_compatible_variants(maps, seed=19)

        self.assertEqual(first.quarter_turns, again.quarter_turns)
        self.assertNotEqual(first.quarter_turns, different.quarter_turns)
        for map_type in maps:
            np.testing.assert_array_equal(first.maps[map_type], again.maps[map_type])
        np.testing.assert_array_equal(
            first.maps[PBR_MAP_ROUGHNESS], first.maps["height"]
        )
        self.assertFalse(
            np.array_equal(
                first.maps[ATLAS_MAP_BASE_COLOR],
                different.maps[ATLAS_MAP_BASE_COLOR],
            )
        )


# ### Boundary and validation tests ###
class TextureTilingBoundaryTests(unittest.TestCase):
    def test_tiny_and_non_square_images_are_safe(self) -> None:
        tiny = np.asarray([[[12, 34, 56, 78]]], dtype=np.uint8)
        narrow = _seamed_color_texture(height=7, width=29)[:, :, :3]

        tiny_result = repair_texture_tiling({ATLAS_MAP_BASE_COLOR: tiny})
        narrow_result = repair_texture_tiling({ATLAS_MAP_BASE_COLOR: narrow})

        np.testing.assert_array_equal(
            tiny_result.maps[ATLAS_MAP_BASE_COLOR],
            tiny,
        )
        self.assertEqual(tiny_result.plan.operations, ())
        self.assertEqual(
            narrow_result.maps[ATLAS_MAP_BASE_COLOR].shape,
            (7, 29, 3),
        )

    def test_single_channel_map_can_use_an_explicit_reference(self) -> None:
        values = np.arange(18 * 31, dtype=np.uint16).reshape((18, 31, 1))
        source = np.asarray(values % 256, dtype=np.uint8)

        result = repair_texture_tiling(
            {PBR_MAP_ROUGHNESS: source},
            reference_map_type=PBR_MAP_ROUGHNESS,
        )

        self.assertEqual(result.maps[PBR_MAP_ROUGHNESS].shape, source.shape)
        self.assertEqual(result.maps[PBR_MAP_ROUGHNESS].dtype, np.uint8)

    def test_invalid_map_sets_fail_before_mutating_sources(self) -> None:
        base_color = _seamed_color_texture(height=32, width=48)
        original = base_color.copy()
        mismatched_normal = _detailed_normal_texture(31, 48)

        with self.assertRaisesRegex(ValueError, "identical dimensions"):
            repair_texture_tiling(
                {
                    ATLAS_MAP_BASE_COLOR: base_color,
                    PBR_MAP_NORMAL: mismatched_normal,
                }
            )
        with self.assertRaisesRegex(ValueError, "uint8"):
            repair_texture_tiling({ATLAS_MAP_BASE_COLOR: base_color.astype(np.float32)})
        with self.assertRaisesRegex(ValueError, "reference map"):
            repair_texture_tiling({PBR_MAP_ROUGHNESS: base_color})

        np.testing.assert_array_equal(base_color, original)

    def test_seed_must_be_non_negative_integer(self) -> None:
        source = _seamed_color_texture(height=32, width=32)

        with self.assertRaisesRegex(ValueError, "seed"):
            build_tiling_repair_plan(source, seed=-1)
        with self.assertRaisesRegex(ValueError, "seed"):
            repair_texture_tiling({ATLAS_MAP_BASE_COLOR: source}, seed=1.5)


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
