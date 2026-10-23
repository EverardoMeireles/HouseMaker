# ### Imports ###
from __future__ import annotations

from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import trimesh

import housemaker.directional_light_projection as projection


# ### Fixture helpers ###
def _plane(z=0.0):
    return trimesh.Trimesh(
        vertices=((-1, -1, z), (1, -1, z), (1, 1, z), (-1, 1, z)),
        faces=((0, 1, 2), (0, 2, 3)), process=False,
    )


# ### Independent first-hit ray guides ###
def test_parallel_segments_start_upstream_and_stop_at_first_geometry():
    mesh = trimesh.util.concatenate((_plane(), _plane(1.0)))
    with patch.object(projection, "_cast_first_hits_with_embree", return_value=None):
        result = projection.build_directional_light_projection_rays(mesh, (0, 0, -1))
    segments = result.reshape((-1, 2, 3))
    assert len(segments) == 25
    assert result.dtype == np.float32
    assert np.all(segments[:, 0, 2] > 1.5)
    np.testing.assert_allclose(segments[:, 1, 2], 1.003, atol=1e-6)
    np.testing.assert_allclose(segments[:, 0, :2], segments[:, 1, :2])
    # No horizontal segments connect neighboring hit points into a grid.
    assert np.all(np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1) > 0.5)


def test_oblique_ray_segments_retain_exact_parallel_direction():
    direction = np.asarray((0.3, 0.2, -1.0))
    direction /= np.linalg.norm(direction)
    with patch.object(projection, "_cast_first_hits_with_embree", return_value=None):
        result = projection.build_directional_light_projection_rays(_plane(), direction)
    segments = result.reshape((-1, 2, 3))
    deltas = segments[:, 1] - segments[:, 0]
    deltas /= np.linalg.norm(deltas, axis=1)[:, None]
    np.testing.assert_allclose(deltas, np.broadcast_to(direction, deltas.shape), atol=1e-6)


def test_missing_hits_are_omitted_instead_of_drawing_through_empty_space():
    with patch.object(projection, "_cast_first_hits_with_embree",
                      return_value=np.full((81, 3), np.nan)):
        result = projection.build_directional_light_projection_rays(_plane(), (0, 0, -1))
    assert result.shape == (0, 3)


def test_embree_acceleration_is_reused_while_only_sun_direction_changes():
    intersector = MagicMock()
    intersector.intersects_location.return_value = (
        np.empty((0, 3)), np.empty(0, dtype=int), np.empty(0, dtype=int),
    )
    fake = ModuleType("trimesh.ray.ray_pyembree")
    fake.RayMeshIntersector = MagicMock(return_value=intersector)
    mesh = _plane()
    with patch.dict("sys.modules", {"trimesh.ray.ray_pyembree": fake}):
        projection.build_directional_light_projection_rays(mesh, (0, 0, -1))
        projection.build_directional_light_projection_rays(mesh, (0.2, 0, -1))
    fake.RayMeshIntersector.assert_called_once_with(mesh)
    assert intersector.intersects_location.call_count == 2
