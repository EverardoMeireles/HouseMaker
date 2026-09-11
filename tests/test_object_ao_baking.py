# ### Imports ###
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import trimesh
from trimesh.visual.texture import TextureVisuals

from housemaker import object_ao_baking
from housemaker.object_ao_baking import (
    ObjectAmbientOcclusionTarget,
    bake_placed_object_ambient_occlusion,
)


# ### Fixture helpers ###
def _horizontal_quad(
    *,
    minimum: float,
    maximum: float,
    height: float,
    with_uv: bool,
) -> trimesh.Trimesh:
    mesh = trimesh.Trimesh(
        vertices=np.asarray(
            (
                (minimum, minimum, height),
                (maximum, minimum, height),
                (maximum, maximum, height),
                (minimum, maximum, height),
            ),
            dtype=float,
        ),
        faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        process=False,
    )
    if with_uv:
        mesh.visual = TextureVisuals(
            uv=np.asarray(
                ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
                dtype=float,
            )
        )
    return mesh


def _horizontal_triangle(*, with_uv: bool, size: float = 1.0) -> trimesh.Trimesh:
    mesh = trimesh.Trimesh(
        vertices=np.asarray(
            ((0.0, 0.0, 0.0), (size, 0.0, 0.0), (0.0, size, 0.0)),
            dtype=float,
        ),
        faces=np.asarray(((0, 1, 2),), dtype=np.int64),
        process=False,
    )
    if with_uv:
        mesh.visual = TextureVisuals(
            uv=np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=float)
        )
    return mesh


def _small_parallel_blocker(
    *,
    center_x: float,
    center_y: float,
    half_size: float,
    height: float,
) -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=np.asarray(
            (
                (center_x - half_size, center_y - half_size, height),
                (center_x + half_size, center_y - half_size, height),
                (center_x + half_size, center_y + half_size, height),
                (center_x - half_size, center_y + half_size, height),
            ),
            dtype=float,
        ),
        faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        process=False,
    )


def _uv_mapped_box() -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    mesh.unmerge_vertices()
    uv = np.zeros((len(mesh.vertices), 2), dtype=float)
    for face_index, face in enumerate(np.asarray(mesh.faces, dtype=np.int64)):
        column = face_index % 4
        row = face_index // 4
        lower = np.asarray(((column + 0.1) / 4.0, (row + 0.1) / 3.0))
        upper = np.asarray(((column + 0.9) / 4.0, (row + 0.9) / 3.0))
        uv[face] = np.asarray(
            (
                lower,
                (upper[0], lower[1]),
                (lower[0], upper[1]),
            )
        )
    mesh.visual = TextureVisuals(uv=uv)
    return mesh


# ### Baking tests ###
class ObjectAmbientOcclusionBakingTests(unittest.TestCase):
    def test_isolated_target_stays_white(self) -> None:
        target = _horizontal_quad(
            minimum=0.0,
            maximum=1.0,
            height=0.0,
            with_uv=True,
        )

        result = bake_placed_object_ambient_occlusion(
            {"objects": (target,)},
            {"objects": 32},
            (target,),
        )

        np.testing.assert_array_equal(
            result["objects"],
            np.full((32, 32), 255, dtype=np.uint8),
        )

    def test_isolated_hard_edge_mesh_ignores_cached_smooth_normals(self) -> None:
        target = _uv_mapped_box()
        first = bake_placed_object_ambient_occlusion(
            {"objects": (target,)},
            {"objects": 64},
            (target,),
        )["objects"]
        target._cache.cache["vertex_normals"] = np.tile(
            np.asarray((0.0, 0.0, -1.0)),
            (len(target.vertices), 1),
        )
        second = bake_placed_object_ambient_occlusion(
            {"objects": (target,)},
            {"objects": 64},
            (target,),
        )["objects"]

        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(
            first,
            np.full((64, 64), 255, dtype=np.uint8),
        )

    def test_nearby_parallel_occluder_darkens_target_deterministically(
        self,
    ) -> None:
        target = _horizontal_quad(
            minimum=0.0,
            maximum=1.0,
            height=0.0,
            with_uv=True,
        )
        occluder = _horizontal_quad(
            minimum=-10.0,
            maximum=10.0,
            height=0.1,
            with_uv=False,
        )

        first = bake_placed_object_ambient_occlusion(
            {"objects": (target,)},
            {"objects": 32},
            (target, occluder),
        )["objects"]
        second = bake_placed_object_ambient_occlusion(
            {"objects": (target,)},
            {"objects": 32},
            (target, occluder),
        )["objects"]

        np.testing.assert_array_equal(first, second)
        self.assertLess(int(np.max(first)), 32)

    def test_missing_ray_backend_becomes_a_clear_export_error(self) -> None:
        target = _horizontal_triangle(with_uv=True)
        with (
            patch.object(
                object_ao_baking,
                "_build_ray_intersector",
                side_effect=RuntimeError("Embree is unavailable."),
            ),
            self.assertRaisesRegex(ValueError, "Embree is unavailable"),
        ):
            bake_placed_object_ambient_occlusion(
                {"objects": (target,)},
                {"objects": 32},
                (target,),
            )

    def test_adaptive_samples_find_small_blocker_on_large_uv_face(self) -> None:
        resolution = 32
        target = _horizontal_triangle(with_uv=True, size=10.0)
        blocker = _small_parallel_blocker(
            center_x=2.0,
            center_y=2.0,
            half_size=0.1,
            height=0.02,
        )

        adaptive = bake_placed_object_ambient_occlusion(
            {"objects": (target,)},
            {"objects": resolution},
            (target, blocker),
        )["objects"]
        with patch.object(
            object_ao_baking,
            "OBJECT_AO_MAX_EXTRA_ADAPTIVE_SAMPLES_PER_TARGET",
            0,
        ):
            centroid_only = bake_placed_object_ambient_occlusion(
                {"objects": (target,)},
                {"objects": resolution},
                (target, blocker),
            )["objects"]

        sample_x = round(0.2 * resolution - 0.5)
        sample_y = round(0.8 * resolution - 0.5)
        sample_region = np.s_[sample_y - 2 : sample_y + 3, sample_x - 2 : sample_x + 3]
        self.assertLess(
            int(np.min(adaptive[sample_region])),
            int(np.min(centroid_only[sample_region])),
        )
        self.assertLess(int(np.min(adaptive[sample_region])), 240)

    def test_mirrored_receiver_is_averaged_into_reused_uvs(self) -> None:
        authored = _horizontal_quad(
            minimum=-2.0,
            maximum=-1.0,
            height=0.0,
            with_uv=True,
        )
        blocker = _small_parallel_blocker(
            center_x=1.5,
            center_y=-1.5,
            half_size=1.4,
            height=0.1,
        )
        target = ObjectAmbientOcclusionTarget(
            mesh=authored,
            atlas_pixel_bounds=(0, 0, 32, 32),
            mirror_plane_point=(0.0, 0.0, 0.0),
            mirror_plane_normal=(1.0, 0.0, 0.0),
        )

        without_mirror = bake_placed_object_ambient_occlusion(
            {"objects": (authored,)},
            {"objects": 32},
            (authored, blocker),
        )["objects"]
        with_mirror = bake_placed_object_ambient_occlusion(
            {"objects": (target,)},
            {"objects": 32},
            (authored, blocker),
        )["objects"]

        self.assertTrue(np.all(without_mirror == 255))
        self.assertLess(int(np.max(with_mirror)), 160)
        self.assertGreater(int(np.min(with_mirror)), 96)

    def test_placed_instances_sharing_uvs_average_their_scene_ao(self) -> None:
        first = _horizontal_triangle(with_uv=True)
        second = first.copy()
        second.apply_translation((10.0, 0.0, 0.0))
        targets = (
            ObjectAmbientOcclusionTarget(first, (0, 0, 32, 32)),
            ObjectAmbientOcclusionTarget(second, (0, 0, 32, 32)),
        )

        def fake_sample(
            positions: np.ndarray,
            *_arguments: object,
        ) -> np.ndarray:
            value = 0.0 if float(np.mean(positions[:, 0])) < 5.0 else 1.0
            return np.full(len(positions), value, dtype=float)

        with patch.object(
            object_ao_baking,
            "_sample_ambient_occlusion",
            side_effect=fake_sample,
        ):
            result = bake_placed_object_ambient_occlusion(
                {"objects": targets},
                {"objects": 32},
                (first, second),
            )["objects"]

        covered_values = result[result < 255]
        self.assertTrue(len(covered_values))
        np.testing.assert_array_equal(
            covered_values,
            np.full_like(covered_values, 128),
        )

    def test_coverage_padding_stays_inside_declared_target_bounds(self) -> None:
        resolution = 16
        target_mesh = trimesh.Trimesh(
            vertices=np.asarray(
                ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                dtype=float,
            ),
            faces=np.asarray(((0, 1, 2),), dtype=np.int64),
            process=False,
        )
        target_mesh.visual = TextureVisuals(
            uv=np.asarray(
                (
                    ((6.0 + 0.5) / resolution, 1.0 - (6.0 + 0.5) / resolution),
                    ((9.0 + 0.5) / resolution, 1.0 - (6.0 + 0.5) / resolution),
                    ((6.0 + 0.5) / resolution, 1.0 - (9.0 + 0.5) / resolution),
                ),
                dtype=float,
            )
        )
        target = ObjectAmbientOcclusionTarget(
            mesh=target_mesh,
            atlas_pixel_bounds=(4, 4, 12, 12),
        )

        with patch.object(
            object_ao_baking,
            "_sample_ambient_occlusion",
            side_effect=lambda positions, *_arguments: np.zeros(
                len(positions),
                dtype=float,
            ),
        ):
            result = bake_placed_object_ambient_occlusion(
                {"objects": (target,)},
                {"objects": resolution},
                (target_mesh,),
            )["objects"]

        self.assertEqual(int(result[6, 4]), 0)
        self.assertEqual(int(result[6, 3]), 255)


if __name__ == "__main__":
    unittest.main()
