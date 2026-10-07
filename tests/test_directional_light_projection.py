# ### Imports ###
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import trimesh

import housemaker.directional_light_projection as projection
from housemaker.directional_light_projection import (
    DIRECTIONAL_LIGHT_PROJECTION_DEPTH_BIAS_METERS,
    build_directional_light_projection_basis,
    build_directional_light_projection_grid,
)


# ### Fixture helpers ###
def _horizontal_plane(*, z_value: float = 0.0) -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=np.asarray(
            (
                (-1.0, -1.0, z_value),
                (1.0, -1.0, z_value),
                (1.0, 1.0, z_value),
                (-1.0, 1.0, z_value),
            ),
            dtype=float,
        ),
        faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        process=False,
    )


def _stacked_horizontal_planes() -> trimesh.Trimesh:
    lower = _horizontal_plane(z_value=0.0)
    upper = _horizontal_plane(z_value=1.0)
    return trimesh.util.concatenate((lower, upper))


# ### Projection-grid tests ###
class DirectionalLightProjectionTests(unittest.TestCase):
    def test_basis_is_orthonormal_for_vertical_and_oblique_directions(self) -> None:
        for direction in ((0.0, 0.0, -1.0), (2.0, -3.0, 4.0)):
            right, up, forward = build_directional_light_projection_basis(
                direction
            )

            np.testing.assert_allclose(np.linalg.norm(right), 1.0)
            np.testing.assert_allclose(np.linalg.norm(up), 1.0)
            np.testing.assert_allclose(np.linalg.norm(forward), 1.0)
            np.testing.assert_allclose(np.dot(right, up), 0.0, atol=1e-12)
            np.testing.assert_allclose(np.dot(right, forward), 0.0, atol=1e-12)
            np.testing.assert_allclose(np.dot(up, forward), 0.0, atol=1e-12)
            np.testing.assert_allclose(
                forward,
                np.asarray(direction, dtype=float) / np.linalg.norm(direction),
            )

    def test_fallback_builds_complete_depth_biased_grid_on_flat_plane(self) -> None:
        with patch.object(
            projection,
            "_cast_first_hits_with_embree",
            return_value=None,
        ):
            positions = build_directional_light_projection_grid(
                _horizontal_plane(),
                (0.0, 0.0, -2.0),
            )

        self.assertEqual(positions.shape, (288, 3))
        self.assertEqual(positions.dtype, np.float32)
        np.testing.assert_allclose(
            positions[:, 2],
            DIRECTIONAL_LIGHT_PROJECTION_DEPTH_BIAS_METERS,
            atol=1e-7,
        )
        self.assertGreaterEqual(float(np.min(positions[:, 0])), -1.0)
        self.assertLessEqual(float(np.max(positions[:, 0])), 1.0)
        self.assertGreaterEqual(float(np.min(positions[:, 1])), -1.0)
        self.assertLessEqual(float(np.max(positions[:, 1])), 1.0)

    def test_first_hit_grid_uses_nearest_occluding_plane(self) -> None:
        with patch.object(
            projection,
            "_cast_first_hits_with_embree",
            return_value=None,
        ):
            positions = build_directional_light_projection_grid(
                _stacked_horizontal_planes(),
                (0.0, 0.0, -1.0),
            )

        self.assertEqual(positions.shape, (288, 3))
        np.testing.assert_allclose(
            positions[:, 2],
            1.0 + DIRECTIONAL_LIGHT_PROJECTION_DEPTH_BIAS_METERS,
            atol=1e-7,
        )

    def test_embree_result_is_preferred_when_available(self) -> None:
        def build_hits(
            _mesh: trimesh.Trimesh,
            origins: np.ndarray,
            directions: np.ndarray,
        ) -> np.ndarray:
            return origins + directions * 2.0

        with (
            patch.object(
                projection,
                "_cast_first_hits_with_embree",
                side_effect=build_hits,
            ) as embree_cast,
            patch.object(
                projection,
                "_cast_first_hits_moller_trumbore",
            ) as fallback_cast,
        ):
            positions = build_directional_light_projection_grid(
                _horizontal_plane(),
                (0.0, 0.0, -1.0),
            )

        embree_cast.assert_called_once()
        fallback_cast.assert_not_called()
        self.assertEqual(positions.shape, (288, 3))

    def test_fallback_output_is_deterministic(self) -> None:
        mesh = _horizontal_plane()
        with patch.object(
            projection,
            "_cast_first_hits_with_embree",
            return_value=None,
        ):
            first = build_directional_light_projection_grid(
                mesh,
                (0.3, 0.2, -1.0),
            )
            second = build_directional_light_projection_grid(
                mesh,
                (0.3, 0.2, -1.0),
            )

        np.testing.assert_array_equal(first, second)

    def test_missing_and_discontinuous_neighbors_are_not_connected(self) -> None:
        hits = np.full((9, 9, 3), np.nan, dtype=float)
        hits[0, 0] = (0.0, 0.0, 0.0)
        hits[0, 1] = (0.1, 0.0, 1.0)
        hits[0, 2] = (0.2, 0.0, 1.0)

        positions = projection._build_hit_line_segments(
            hits,
            np.asarray((0.0, 0.0, 1.0), dtype=float),
            horizontal_step=0.1,
            vertical_step=0.1,
        )

        np.testing.assert_allclose(
            positions,
            np.asarray(((0.1, 0.0, 1.0), (0.2, 0.0, 1.0))),
        )

    def test_invalid_mesh_direction_and_depth_bias_are_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "trimesh"):
            build_directional_light_projection_grid(
                object(),  # type: ignore[arg-type]
                (0.0, 0.0, -1.0),
            )
        with self.assertRaisesRegex(ValueError, "vertices"):
            build_directional_light_projection_grid(
                trimesh.Trimesh(process=False),
                (0.0, 0.0, -1.0),
            )
        with self.assertRaisesRegex(ValueError, "cannot be zero"):
            build_directional_light_projection_grid(
                _horizontal_plane(),
                (0.0, 0.0, 0.0),
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            build_directional_light_projection_grid(
                _horizontal_plane(),
                (0.0, 0.0, -1.0),
                depth_bias_meters=-0.1,
            )


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
