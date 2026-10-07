# ### Imports ###
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import trimesh

# ### Constants ###
DIRECTIONAL_LIGHT_PROJECTION_GRID_SIZE = 9
DIRECTIONAL_LIGHT_PROJECTION_DEPTH_BIAS_METERS = 0.003
DIRECTIONAL_LIGHT_PROJECTION_UPSTREAM_BIAS_RATIO = 1e-5
DIRECTIONAL_LIGHT_PROJECTION_MINIMUM_UPSTREAM_BIAS_METERS = 1e-5
DIRECTIONAL_LIGHT_PROJECTION_MAXIMUM_NEIGHBOR_STRETCH = 3.0
_GEOMETRY_EPSILON = 1e-10


# ### Public projection helpers ###
def build_directional_light_projection_grid(
    collision_mesh: trimesh.Trimesh,
    direction: Sequence[float] | np.ndarray,
    *,
    depth_bias_meters: float = DIRECTIONAL_LIGHT_PROJECTION_DEPTH_BIAS_METERS,
) -> np.ndarray:
    """Return depth-biased line segments showing first directional-light hits.

    A fixed 9-by-9 lattice is placed immediately upstream of the collision
    mesh. Its parallel rays model a directional light rather than a point or
    spot light. Adjacent samples are joined only when both rays hit nearby
    geometry, keeping lines from spanning empty space or depth discontinuities.
    """

    vertices, faces = _validate_collision_mesh(collision_mesh)
    normalized_direction = _normalize_direction(direction)
    normalized_depth_bias = _normalize_depth_bias(depth_bias_meters)
    ray_origins, horizontal_step, vertical_step = _build_ray_lattice(
        vertices,
        normalized_direction,
    )
    ray_directions = np.broadcast_to(
        normalized_direction,
        ray_origins.shape,
    ).copy()
    hits = _cast_first_hits_with_embree(
        collision_mesh,
        ray_origins,
        ray_directions,
    )
    if hits is None:
        hits = _cast_first_hits_moller_trumbore(
            vertices,
            faces,
            ray_origins,
            normalized_direction,
        )
    biased_hits = hits.copy()
    hit_mask = np.all(np.isfinite(biased_hits), axis=1)
    biased_hits[hit_mask] -= normalized_direction * normalized_depth_bias
    return _build_hit_line_segments(
        biased_hits.reshape(
            (
                DIRECTIONAL_LIGHT_PROJECTION_GRID_SIZE,
                DIRECTIONAL_LIGHT_PROJECTION_GRID_SIZE,
                3,
            )
        ),
        normalized_direction,
        horizontal_step=horizontal_step,
        vertical_step=vertical_step,
    )


def build_directional_light_projection_basis(
    direction: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return stable right, up, and forward unit vectors for one direction."""

    forward = _normalize_direction(direction)
    reference = np.asarray((0.0, 0.0, 1.0), dtype=float)
    if abs(float(np.dot(forward, reference))) > 0.92:
        reference = np.asarray((0.0, 1.0, 0.0), dtype=float)
    right = np.cross(forward, reference)
    right /= float(np.linalg.norm(right))
    up = np.cross(forward, right)
    up /= float(np.linalg.norm(up))
    return (
        np.ascontiguousarray(right),
        np.ascontiguousarray(up),
        np.ascontiguousarray(forward),
    )


# ### Input validation helpers ###
def _validate_collision_mesh(
    collision_mesh: object,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(collision_mesh, trimesh.Trimesh):
        raise TypeError("Directional-light projection requires a trimesh mesh.")
    vertices = np.asarray(collision_mesh.vertices, dtype=float)
    faces = np.asarray(collision_mesh.faces, dtype=np.int64)
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or not len(vertices)
        or not np.all(np.isfinite(vertices))
    ):
        raise ValueError(
            "Directional-light projection mesh vertices must be finite XYZ values."
        )
    if faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces):
        raise ValueError(
            "Directional-light projection requires at least one triangular face."
        )
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("Directional-light projection mesh faces are out of range.")
    return np.ascontiguousarray(vertices), np.ascontiguousarray(faces)


def _normalize_direction(
    direction: Sequence[float] | np.ndarray,
) -> np.ndarray:
    if isinstance(direction, (str, bytes, bytearray)):
        raise TypeError("Directional-light projection direction must be a 3D vector.")
    try:
        normalized = np.asarray(direction, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(
            "Directional-light projection direction must be a 3D vector."
        ) from error
    if normalized.shape != (3,) or not np.all(np.isfinite(normalized)):
        raise ValueError(
            "Directional-light projection direction must contain three finite values."
        )
    length = float(np.linalg.norm(normalized))
    if length <= _GEOMETRY_EPSILON:
        raise ValueError("Directional-light projection direction cannot be zero.")
    return np.ascontiguousarray(normalized / length)


def _normalize_depth_bias(depth_bias_meters: object) -> float:
    if isinstance(depth_bias_meters, bool):
        raise TypeError("Directional-light projection depth bias must be numeric.")
    try:
        normalized = float(depth_bias_meters)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(
            "Directional-light projection depth bias must be numeric."
        ) from error
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(
            "Directional-light projection depth bias must be finite and non-negative."
        )
    return normalized


# ### Ray-lattice helpers ###
def _build_ray_lattice(
    vertices: np.ndarray,
    direction: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    right, up, forward = build_directional_light_projection_basis(direction)
    minimum = np.min(vertices, axis=0)
    maximum = np.max(vertices, axis=0)
    corners = np.asarray(
        tuple(
            (x_value, y_value, z_value)
            for x_value in (minimum[0], maximum[0])
            for y_value in (minimum[1], maximum[1])
            for z_value in (minimum[2], maximum[2])
        ),
        dtype=float,
    )
    right_coordinates = corners @ right
    up_coordinates = corners @ up
    forward_coordinates = corners @ forward
    scene_diagonal = float(np.linalg.norm(maximum - minimum))
    upstream_bias = max(
        scene_diagonal * DIRECTIONAL_LIGHT_PROJECTION_UPSTREAM_BIAS_RATIO,
        DIRECTIONAL_LIGHT_PROJECTION_MINIMUM_UPSTREAM_BIAS_METERS,
    )
    upstream_coordinate = float(np.min(forward_coordinates) - upstream_bias)
    right_samples = np.linspace(
        float(np.min(right_coordinates)),
        float(np.max(right_coordinates)),
        DIRECTIONAL_LIGHT_PROJECTION_GRID_SIZE,
    )
    up_samples = np.linspace(
        float(np.min(up_coordinates)),
        float(np.max(up_coordinates)),
        DIRECTIONAL_LIGHT_PROJECTION_GRID_SIZE,
    )
    origins = np.asarray(
        tuple(
            right * right_coordinate
            + up * up_coordinate
            + forward * upstream_coordinate
            for up_coordinate in up_samples
            for right_coordinate in right_samples
        ),
        dtype=float,
    )
    return (
        np.ascontiguousarray(origins),
        _sample_step(right_samples),
        _sample_step(up_samples),
    )


def _sample_step(samples: np.ndarray) -> float:
    if len(samples) < 2:
        return 0.0
    return abs(float(samples[1] - samples[0]))


# ### Ray-casting helpers ###
def _cast_first_hits_with_embree(
    collision_mesh: trimesh.Trimesh,
    ray_origins: np.ndarray,
    ray_directions: np.ndarray,
) -> np.ndarray | None:
    """Return first hits through Embree, or ``None`` when it is unavailable."""

    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector

        intersector = RayMeshIntersector(collision_mesh)
        locations, ray_indices, _triangle_indices = intersector.intersects_location(
            ray_origins,
            ray_directions,
            multiple_hits=False,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        return None
    hits = np.full(ray_origins.shape, np.nan, dtype=float)
    locations = np.asarray(locations, dtype=float)
    ray_indices = np.asarray(ray_indices, dtype=np.int64)
    if (
        locations.ndim != 2
        or locations.shape[1:] != (3,)
        or ray_indices.ndim != 1
        or len(locations) != len(ray_indices)
    ):
        return None
    for location, ray_index in zip(locations, ray_indices, strict=True):
        if (
            ray_index < 0
            or ray_index >= len(ray_origins)
            or not np.all(np.isfinite(location))
        ):
            continue
        distance = float(np.dot(location - ray_origins[ray_index], ray_directions[ray_index]))
        if distance < -_GEOMETRY_EPSILON:
            continue
        previous = hits[ray_index]
        if not np.all(np.isfinite(previous)):
            hits[ray_index] = location
            continue
        previous_distance = float(
            np.dot(
                previous - ray_origins[ray_index],
                ray_directions[ray_index],
            )
        )
        if distance < previous_distance:
            hits[ray_index] = location
    return np.ascontiguousarray(hits)


def _cast_first_hits_moller_trumbore(
    vertices: np.ndarray,
    faces: np.ndarray,
    ray_origins: np.ndarray,
    ray_direction: np.ndarray,
) -> np.ndarray:
    """Cast deterministic first-hit rays without optional acceleration."""

    triangles = vertices[faces]
    first_edges = triangles[:, 1] - triangles[:, 0]
    second_edges = triangles[:, 2] - triangles[:, 0]
    repeated_direction = np.broadcast_to(ray_direction, second_edges.shape)
    cross_direction = np.cross(repeated_direction, second_edges)
    determinants = np.einsum("ij,ij->i", first_edges, cross_direction)
    determinant_mask = np.abs(determinants) > _GEOMETRY_EPSILON
    inverse_determinants = np.zeros_like(determinants)
    inverse_determinants[determinant_mask] = 1.0 / determinants[determinant_mask]
    hits = np.full(ray_origins.shape, np.nan, dtype=float)
    for ray_index, ray_origin in enumerate(ray_origins):
        origin_offsets = ray_origin[np.newaxis, :] - triangles[:, 0]
        first_coordinates = (
            np.einsum("ij,ij->i", origin_offsets, cross_direction)
            * inverse_determinants
        )
        offset_crosses = np.cross(origin_offsets, first_edges)
        second_coordinates = (
            np.einsum("j,ij->i", ray_direction, offset_crosses)
            * inverse_determinants
        )
        distances = (
            np.einsum("ij,ij->i", second_edges, offset_crosses)
            * inverse_determinants
        )
        usable = determinant_mask.copy()
        usable &= first_coordinates >= -_GEOMETRY_EPSILON
        usable &= second_coordinates >= -_GEOMETRY_EPSILON
        usable &= (
            first_coordinates + second_coordinates <= 1.0 + _GEOMETRY_EPSILON
        )
        usable &= distances >= -_GEOMETRY_EPSILON
        hit_indices = np.flatnonzero(usable)
        if not len(hit_indices):
            continue
        hit_index = int(hit_indices[np.argmin(distances[hit_indices])])
        hits[ray_index] = ray_origin + ray_direction * max(
            float(distances[hit_index]),
            0.0,
        )
    return np.ascontiguousarray(hits)


# ### Grid-segment helpers ###
def _build_hit_line_segments(
    hits: np.ndarray,
    direction: np.ndarray,
    *,
    horizontal_step: float,
    vertical_step: float,
) -> np.ndarray:
    positions: list[np.ndarray] = []
    grid_size = DIRECTIONAL_LIGHT_PROJECTION_GRID_SIZE
    for row_index in range(grid_size):
        for column_index in range(grid_size - 1):
            _append_continuous_segment(
                positions,
                hits[row_index, column_index],
                hits[row_index, column_index + 1],
                direction,
                horizontal_step,
            )
    for row_index in range(grid_size - 1):
        for column_index in range(grid_size):
            _append_continuous_segment(
                positions,
                hits[row_index, column_index],
                hits[row_index + 1, column_index],
                direction,
                vertical_step,
            )
    if not positions:
        return np.empty((0, 3), dtype=np.float32)
    return np.ascontiguousarray(positions, dtype=np.float32)


def _append_continuous_segment(
    positions: list[np.ndarray],
    first: np.ndarray,
    second: np.ndarray,
    direction: np.ndarray,
    lattice_step: float,
) -> None:
    if (
        lattice_step <= _GEOMETRY_EPSILON
        or not np.all(np.isfinite(first))
        or not np.all(np.isfinite(second))
    ):
        return
    delta = second - first
    depth_change = abs(float(np.dot(delta, direction)))
    if (
        depth_change
        > lattice_step * DIRECTIONAL_LIGHT_PROJECTION_MAXIMUM_NEIGHBOR_STRETCH
    ):
        return
    positions.extend((first, second))
