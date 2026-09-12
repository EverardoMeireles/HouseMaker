# ### Imports ###
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import trimesh

# ### Constants ###
OBJECT_AO_RAY_COUNT = 32
OBJECT_AO_MAX_DISTANCE_METERS = 0.5
OBJECT_AO_MINIMUM_RAY_BIAS_METERS = 1e-5
OBJECT_AO_MAXIMUM_RAY_BIAS_METERS = 5e-4
OBJECT_AO_RAY_BIAS_SCENE_SCALE = 1e-5
OBJECT_AO_SAMPLE_BATCH_SIZE = 2_048
OBJECT_AO_RASTER_ROW_BATCH_SIZE = 64
OBJECT_AO_ADAPTIVE_SAMPLE_SPACING_PIXELS = 96.0
OBJECT_AO_ADAPTIVE_SAMPLE_SPACING_METERS = OBJECT_AO_MAX_DISTANCE_METERS * 0.25
OBJECT_AO_MAX_ADAPTIVE_SUBDIVISIONS = 96
OBJECT_AO_MAX_EXTRA_ADAPTIVE_SAMPLES_PER_TARGET = 32_768
OBJECT_AO_COVERAGE_PADDING_PIXELS = 2
OBJECT_AO_GEOMETRY_BATCH_SIZE = 65_536
OBJECT_AO_CANCELLATION_POLL_INTERVAL = 64
GEOMETRY_EPSILON = 1e-12
RASTER_EPSILON = 1e-8


# ### Public exceptions ###
class ObjectAmbientOcclusionBakeCancelled(RuntimeError):
    """Raised when ambient-occlusion work is cancelled cooperatively."""


# ### Public data models ###
@dataclass(frozen=True)
class ObjectAmbientOcclusionTarget:
    """One world-space receiver and its half-open top-origin Atlas bounds.

    A mirror plane describes a runtime mirror which reuses the authored
    receiver's UVs. Both receiver evaluations are averaged into those UVs.
    """

    mesh: trimesh.Trimesh
    atlas_pixel_bounds: tuple[int, int, int, int]
    mirror_plane_point: tuple[float, float, float] | None = None
    mirror_plane_normal: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mesh, trimesh.Trimesh):
            raise TypeError("An object AO target requires a triangle mesh.")
        try:
            raw_bounds = tuple(self.atlas_pixel_bounds)
        except TypeError as error:
            raise TypeError(
                "Object AO target bounds must contain four integers."
            ) from error
        if len(raw_bounds) != 4 or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in raw_bounds
        ):
            raise TypeError("Object AO target bounds must contain four integers.")
        bounds = tuple(int(value) for value in raw_bounds)
        left, top, right, bottom = bounds
        if left < 0 or top < 0 or right <= left or bottom <= top:
            raise ValueError("Object AO target bounds must have positive area.")

        point = _normalize_optional_vector(self.mirror_plane_point)
        normal = _normalize_optional_vector(self.mirror_plane_normal)
        if (point is None) != (normal is None):
            raise ValueError(
                "Object AO mirror plane point and normal must be supplied together."
            )
        if normal is not None:
            length = float(np.linalg.norm(normal))
            if length <= GEOMETRY_EPSILON:
                raise ValueError("An object AO mirror plane normal cannot be zero.")
            normal = tuple(float(value) for value in np.asarray(normal) / length)

        object.__setattr__(self, "atlas_pixel_bounds", bounds)
        object.__setattr__(self, "mirror_plane_point", point)
        object.__setattr__(self, "mirror_plane_normal", normal)


@dataclass(frozen=True)
class _FaceSamplingPatch:
    """Texture-space triangulation and world-space samples for one face."""

    pixel_coordinates: np.ndarray
    world_positions: np.ndarray
    raster_triangles: np.ndarray
    normal: np.ndarray


class _RayIntersector(Protocol):
    """Narrow interface supplied by the lazily loaded Embree backend."""

    def intersects_location(
        self,
        ray_origins: np.ndarray,
        ray_directions: np.ndarray,
        *,
        multiple_hits: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


# ### Public helpers ###
def bake_placed_object_ambient_occlusion(
    targets_by_atlas: Mapping[
        str,
        Sequence[ObjectAmbientOcclusionTarget | trimesh.Trimesh],
    ],
    atlas_resolutions: Mapping[str, int],
    occluder_meshes: Sequence[trimesh.Trimesh],
    *,
    cancellation_check: Callable[[], bool] | None = None,
) -> dict[str, np.ndarray]:
    """Bake deterministic object AO into top-origin grayscale Atlas planes.

    Large UV faces receive a bounded texture-space sampling grid so localized
    blockers cannot be hidden by vertex-only interpolation. Legacy bare mesh
    targets remain accepted and use the full Atlas as their padding boundary.
    Placed instances sharing one texture region receive their average scene AO,
    since a shared Atlas cannot store a different bake for each instance.
    """

    _check_cancelled(cancellation_check)
    normalized_targets = _normalize_targets(
        targets_by_atlas,
        atlas_resolutions,
    )
    if not normalized_targets:
        return {}

    occluder = _build_occluder_mesh(
        occluder_meshes,
        cancellation_check=cancellation_check,
    )
    if occluder is None:
        return {
            atlas_id: np.full((resolution, resolution), 255, dtype=np.uint8)
            for atlas_id, (resolution, _targets) in normalized_targets.items()
        }

    try:
        _check_cancelled(cancellation_check)
        intersector = _build_ray_intersector(occluder)
        _check_cancelled(cancellation_check)
    except ObjectAmbientOcclusionBakeCancelled:
        raise
    except RuntimeError as error:
        raise ValueError(str(error)) from error
    ray_bias = _build_scale_aware_ray_bias(occluder)
    results: dict[str, np.ndarray] = {}
    try:
        for atlas_id, (resolution, targets) in normalized_targets.items():
            _check_cancelled(cancellation_check)
            ambient_occlusion = np.full(
                (resolution, resolution),
                255,
                dtype=np.uint8,
            )
            targets_by_bounds: dict[
                tuple[int, int, int, int],
                list[ObjectAmbientOcclusionTarget],
            ] = {}
            for target in targets:
                _check_cancelled(cancellation_check)
                targets_by_bounds.setdefault(
                    target.atlas_pixel_bounds,
                    [],
                ).append(target)
            for shared_targets in targets_by_bounds.values():
                if cancellation_check is None:
                    _bake_shared_target_region(
                        tuple(shared_targets),
                        ambient_occlusion,
                        intersector,
                        ray_bias,
                    )
                else:
                    _bake_shared_target_region(
                        tuple(shared_targets),
                        ambient_occlusion,
                        intersector,
                        ray_bias,
                        cancellation_check=cancellation_check,
                    )
            results[atlas_id] = ambient_occlusion
    except ObjectAmbientOcclusionBakeCancelled:
        raise
    except (MemoryError, RuntimeError, ValueError) as error:
        raise ValueError(
            "Placed-object ambient occlusion could not be baked."
        ) from error
    _check_cancelled(cancellation_check)
    return results


# ### Input normalization helpers ###
def _normalize_targets(
    targets_by_atlas: Mapping[
        str,
        Sequence[ObjectAmbientOcclusionTarget | trimesh.Trimesh],
    ],
    atlas_resolutions: Mapping[str, int],
) -> dict[str, tuple[int, tuple[ObjectAmbientOcclusionTarget, ...]]]:
    if not isinstance(targets_by_atlas, Mapping):
        raise TypeError("Object AO targets must be grouped by texture Atlas.")
    if not isinstance(atlas_resolutions, Mapping):
        raise TypeError("Object AO Atlas resolutions must be mapped by ID.")

    normalized: dict[
        str,
        tuple[int, tuple[ObjectAmbientOcclusionTarget, ...]],
    ] = {}
    for raw_atlas_id, raw_targets in targets_by_atlas.items():
        atlas_id = str(raw_atlas_id)
        if not atlas_id:
            raise ValueError("Object AO Atlas IDs cannot be empty.")
        if isinstance(raw_targets, (str, bytes, bytearray)) or not isinstance(
            raw_targets,
            Sequence,
        ):
            raise TypeError("Object AO targets must be mesh sequences.")
        raw_target_items = tuple(raw_targets)
        if not raw_target_items:
            continue
        raw_resolution = atlas_resolutions.get(atlas_id)
        if isinstance(raw_resolution, bool):
            raise TypeError("Object AO Atlas resolutions must be integers.")
        try:
            resolution = int(raw_resolution)
        except (TypeError, ValueError, OverflowError) as error:
            raise TypeError("Object AO Atlas resolutions must be integers.") from error
        if resolution <= 0 or raw_resolution != resolution:
            raise ValueError("Object AO Atlas resolutions must be positive.")
        targets: list[ObjectAmbientOcclusionTarget] = []
        for raw_target in raw_target_items:
            if isinstance(raw_target, ObjectAmbientOcclusionTarget):
                target = raw_target
            elif isinstance(raw_target, trimesh.Trimesh):
                target = ObjectAmbientOcclusionTarget(
                    mesh=raw_target,
                    atlas_pixel_bounds=(0, 0, resolution, resolution),
                )
            else:
                raise TypeError("Object AO targets must be meshes or target records.")
            _validate_target(target, resolution)
            targets.append(target)
        normalized[atlas_id] = (resolution, tuple(targets))
    return normalized


def _validate_target(
    target: ObjectAmbientOcclusionTarget,
    resolution: int,
) -> None:
    mesh = target.mesh
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    uv = _get_valid_uv(mesh)
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or faces.ndim != 2
        or faces.shape[1:] != (3,)
        or not len(vertices)
        or not len(faces)
        or not np.all(np.isfinite(vertices))
        or uv is None
    ):
        raise ValueError(
            "Object AO targets require finite triangle meshes with Atlas UVs."
        )
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("Object AO target faces reference missing vertices.")
    if np.any(uv < -RASTER_EPSILON) or np.any(uv > 1.0 + RASTER_EPSILON):
        raise ValueError("Object AO target UVs leave the texture Atlas.")

    left, top, right, bottom = target.atlas_pixel_bounds
    if right > resolution or bottom > resolution:
        raise ValueError("Object AO target bounds leave the texture Atlas.")
    pixel_coordinates = _uv_to_top_origin_pixels(uv, resolution)
    lower = np.asarray((left - 0.5, top - 0.5), dtype=float)
    upper = np.asarray((right - 0.5, bottom - 0.5), dtype=float)
    if np.any(pixel_coordinates < lower - RASTER_EPSILON) or np.any(
        pixel_coordinates > upper + RASTER_EPSILON
    ):
        raise ValueError("Object AO target UVs leave their declared Atlas bounds.")


def _normalize_optional_vector(
    raw_value: Sequence[float] | np.ndarray | None,
) -> tuple[float, float, float] | None:
    if raw_value is None:
        return None
    try:
        value = np.asarray(raw_value, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(
            "Object AO mirror plane vectors must have three values."
        ) from error
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("Object AO mirror plane vectors must be finite 3D values.")
    return tuple(float(component) for component in value)


# ### Occluder helpers ###
def _build_ray_intersector(occluder: trimesh.Trimesh) -> _RayIntersector:
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "Placed-object AO export requires a working Embree ray tracer. "
            "Reinstall the embreex dependency and retry."
        ) from error
    try:
        return RayMeshIntersector(occluder)
    except (OSError, RuntimeError, ValueError) as error:
        raise RuntimeError(
            "Placed-object AO export could not initialize the Embree ray tracer."
        ) from error


def _build_occluder_mesh(
    meshes: Sequence[trimesh.Trimesh],
    *,
    cancellation_check: Callable[[], bool] | None = None,
) -> trimesh.Trimesh | None:
    if isinstance(meshes, (str, bytes, bytearray)) or not isinstance(
        meshes,
        Sequence,
    ):
        raise TypeError("Object AO occluders must be a mesh sequence.")

    _check_cancelled(cancellation_check)
    triangles: list[np.ndarray] = []
    for mesh in meshes:
        _check_cancelled(cancellation_check)
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError("Object AO occluders must be triangle meshes.")
        vertices = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if (
            vertices.ndim != 2
            or vertices.shape[1:] != (3,)
            or faces.ndim != 2
            or faces.shape[1:] != (3,)
            or not len(vertices)
            or not len(faces)
            or not np.all(np.isfinite(vertices))
            or np.any(faces < 0)
            or np.any(faces >= len(vertices))
        ):
            continue
        for face_start in range(
            0,
            len(faces),
            OBJECT_AO_GEOMETRY_BATCH_SIZE,
        ):
            _check_cancelled(cancellation_check)
            face_end = min(
                face_start + OBJECT_AO_GEOMETRY_BATCH_SIZE,
                len(faces),
            )
            mesh_triangles = vertices[faces[face_start:face_end]]
            cross_products = np.cross(
                mesh_triangles[:, 1] - mesh_triangles[:, 0],
                mesh_triangles[:, 2] - mesh_triangles[:, 0],
            )
            usable = np.all(np.isfinite(mesh_triangles), axis=(1, 2)) & (
                np.linalg.norm(cross_products, axis=1) > GEOMETRY_EPSILON
            )
            if np.any(usable):
                triangles.append(mesh_triangles[usable])
    if not triangles:
        _check_cancelled(cancellation_check)
        return None

    _check_cancelled(cancellation_check)
    triangle_array = np.ascontiguousarray(np.concatenate(triangles), dtype=float)
    vertices = triangle_array.reshape((-1, 3))
    faces = np.arange(len(vertices), dtype=np.int64).reshape((-1, 3))
    result = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    )
    _check_cancelled(cancellation_check)
    return result


def _build_scale_aware_ray_bias(occluder: trimesh.Trimesh) -> float:
    bounds = np.asarray(occluder.bounds, dtype=float)
    diagonal = float(np.linalg.norm(bounds[1] - bounds[0]))
    if not math.isfinite(diagonal):
        diagonal = 0.0
    return float(
        np.clip(
            diagonal * OBJECT_AO_RAY_BIAS_SCENE_SCALE,
            OBJECT_AO_MINIMUM_RAY_BIAS_METERS,
            OBJECT_AO_MAXIMUM_RAY_BIAS_METERS,
        )
    )


# ### AO sampling helpers ###
def _bake_target_mesh(
    target: ObjectAmbientOcclusionTarget,
    intersector: _RayIntersector,
    ray_bias: float,
    resolution: int,
    *,
    cancellation_check: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    _check_cancelled(cancellation_check)
    mesh = target.mesh
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    uv = _get_valid_uv(mesh)
    assert uv is not None

    patches = _build_face_sampling_patches(
        vertices,
        faces,
        uv,
        resolution,
        target.atlas_pixel_bounds,
        cancellation_check=cancellation_check,
    )
    if not patches:
        return None
    sample_positions = np.concatenate(
        tuple(patch.world_positions for patch in patches),
        axis=0,
    )
    sample_normals = np.concatenate(
        tuple(
            np.broadcast_to(patch.normal, patch.world_positions.shape)
            for patch in patches
        ),
        axis=0,
    )
    sample_values = _call_sample_ambient_occlusion(
        sample_positions,
        sample_normals,
        intersector,
        ray_bias,
        cancellation_check,
    )
    if target.mirror_plane_point is not None:
        mirrored_positions, mirrored_normals = _reflect_receiver_samples(
            sample_positions,
            sample_normals,
            target,
        )
        mirrored_values = _call_sample_ambient_occlusion(
            mirrored_positions,
            mirrored_normals,
            intersector,
            ray_bias,
            cancellation_check,
        )
        sample_values = (sample_values + mirrored_values) * 0.5

    left, top, right, bottom = target.atlas_pixel_bounds
    target_output = np.full((bottom - top, right - left), 255, dtype=np.uint8)
    coverage = np.zeros(target_output.shape, dtype=bool)
    sample_offset = 0
    for patch in patches:
        sample_end = sample_offset + len(patch.world_positions)
        _rasterize_sampling_patch(
            target_output,
            coverage,
            patch,
            sample_values[sample_offset:sample_end],
            cancellation_check=cancellation_check,
        )
        sample_offset = sample_end
    _pad_ambient_occlusion_coverage(
        target_output,
        coverage,
        OBJECT_AO_COVERAGE_PADDING_PIXELS,
        cancellation_check=cancellation_check,
    )
    _check_cancelled(cancellation_check)
    return target_output, coverage


def _bake_shared_target_region(
    targets: Sequence[ObjectAmbientOcclusionTarget],
    output: np.ndarray,
    intersector: _RayIntersector,
    ray_bias: float,
    *,
    cancellation_check: Callable[[], bool] | None = None,
) -> None:
    """Average scene AO for instances which reuse one Atlas region."""

    if not targets:
        return
    left, top, right, bottom = targets[0].atlas_pixel_bounds
    shape = (bottom - top, right - left)
    value_sums = np.zeros(shape, dtype=np.uint32)
    sample_counts = np.zeros(shape, dtype=np.uint32)
    for target in targets:
        _check_cancelled(cancellation_check)
        if cancellation_check is None:
            baked = _bake_target_mesh(
                target,
                intersector,
                ray_bias,
                int(output.shape[0]),
            )
        else:
            baked = _bake_target_mesh(
                target,
                intersector,
                ray_bias,
                int(output.shape[0]),
                cancellation_check=cancellation_check,
            )
        if baked is None:
            continue
        target_values, coverage = baked
        value_sums[coverage] += target_values[coverage]
        sample_counts[coverage] += 1
    covered = sample_counts > 0
    if not np.any(covered):
        return
    averaged = (value_sums[covered] + sample_counts[covered] // 2) // sample_counts[
        covered
    ]
    output_region = output[top:bottom, left:right]
    output_region[covered] = np.asarray(averaged, dtype=np.uint8)
    _check_cancelled(cancellation_check)


def _build_face_sampling_patches(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    resolution: int,
    atlas_pixel_bounds: tuple[int, int, int, int],
    *,
    cancellation_check: Callable[[], bool] | None = None,
) -> tuple[_FaceSamplingPatch, ...]:
    _check_cancelled(cancellation_check)
    left, top, _right, _bottom = atlas_pixel_bounds
    pixel_coordinates = _uv_to_top_origin_pixels(uv, resolution)
    pixel_coordinates -= np.asarray((left, top), dtype=float)
    pixel_triangles = pixel_coordinates[faces]
    world_triangles = vertices[faces]
    subdivisions = _allocate_adaptive_face_subdivisions(
        pixel_triangles,
        world_triangles,
        cancellation_check=cancellation_check,
    )
    patches: list[_FaceSamplingPatch] = []
    for face_index, face in enumerate(faces):
        if face_index % OBJECT_AO_CANCELLATION_POLL_INTERVAL == 0:
            _check_cancelled(cancellation_check)
        world_triangle = world_triangles[face_index]
        cross_product = np.cross(
            world_triangle[1] - world_triangle[0],
            world_triangle[2] - world_triangle[0],
        )
        length = float(np.linalg.norm(cross_product))
        if not math.isfinite(length) or length <= GEOMETRY_EPSILON:
            continue
        normal = np.ascontiguousarray(cross_product / length)
        subdivision = int(subdivisions[face_index])
        if subdivision >= 3:
            patch = _build_subdivided_face_patch(
                world_triangle,
                pixel_triangles[face_index],
                normal,
                subdivision,
                cancellation_check=cancellation_check,
            )
        else:
            patch = _build_centroid_face_patch(
                world_triangle,
                pixel_triangles[face_index],
                normal,
            )
        patches.append(patch)
    _check_cancelled(cancellation_check)
    return tuple(patches)


def _allocate_adaptive_face_subdivisions(
    pixel_triangles: np.ndarray,
    world_triangles: np.ndarray,
    *,
    cancellation_check: Callable[[], bool] | None = None,
) -> np.ndarray:
    _check_cancelled(cancellation_check)
    pixel_edge_lengths = np.stack(
        (
            np.linalg.norm(pixel_triangles[:, 1] - pixel_triangles[:, 0], axis=1),
            np.linalg.norm(pixel_triangles[:, 2] - pixel_triangles[:, 1], axis=1),
            np.linalg.norm(pixel_triangles[:, 0] - pixel_triangles[:, 2], axis=1),
        ),
        axis=1,
    )
    world_edge_lengths = np.stack(
        (
            np.linalg.norm(world_triangles[:, 1] - world_triangles[:, 0], axis=1),
            np.linalg.norm(world_triangles[:, 2] - world_triangles[:, 1], axis=1),
            np.linalg.norm(world_triangles[:, 0] - world_triangles[:, 2], axis=1),
        ),
        axis=1,
    )
    maximum_pixel_edges = np.max(pixel_edge_lengths, axis=1)
    maximum_world_edges = np.max(world_edge_lengths, axis=1)
    texture_subdivisions = np.ceil(
        maximum_pixel_edges / OBJECT_AO_ADAPTIVE_SAMPLE_SPACING_PIXELS
    )
    world_subdivisions = np.ceil(
        maximum_world_edges / OBJECT_AO_ADAPTIVE_SAMPLE_SPACING_METERS
    )
    desired = np.asarray(
        np.maximum(texture_subdivisions, world_subdivisions),
        dtype=np.int64,
    )

    doubled_areas = np.abs(
        (pixel_triangles[:, 1, 0] - pixel_triangles[:, 0, 0])
        * (pixel_triangles[:, 2, 1] - pixel_triangles[:, 0, 1])
        - (pixel_triangles[:, 1, 1] - pixel_triangles[:, 0, 1])
        * (pixel_triangles[:, 2, 0] - pixel_triangles[:, 0, 0])
    )
    needs_adaptive_sampling = (
        (desired > 1) & (maximum_pixel_edges >= 1.0) & (doubled_areas > RASTER_EPSILON)
    )
    desired[needs_adaptive_sampling] = 3 * np.ceil(
        desired[needs_adaptive_sampling] / 3.0
    ).astype(np.int64)
    representable_subdivisions = 3 * np.ceil(maximum_pixel_edges / 3.0).astype(np.int64)
    desired = np.minimum(desired, representable_subdivisions)
    desired = np.clip(desired, 0, OBJECT_AO_MAX_ADAPTIVE_SUBDIVISIONS)
    desired[~needs_adaptive_sampling] = 0

    candidate_indices = np.flatnonzero(needs_adaptive_sampling)
    candidate_indices = np.asarray(
        sorted(
            candidate_indices.tolist(),
            key=lambda index: (
                -float(doubled_areas[index]),
                -float(maximum_pixel_edges[index]),
                -float(maximum_world_edges[index]),
                int(index),
            ),
        ),
        dtype=np.int64,
    )

    _check_cancelled(cancellation_check)
    allocated = np.zeros(len(pixel_triangles), dtype=np.int64)
    remaining_samples = OBJECT_AO_MAX_EXTRA_ADAPTIVE_SAMPLES_PER_TARGET
    while remaining_samples >= 6:
        allocation_changed = False
        for candidate_index, face_index in enumerate(candidate_indices):
            if candidate_index % OBJECT_AO_CANCELLATION_POLL_INTERVAL == 0:
                _check_cancelled(cancellation_check)
            current_subdivision = int(allocated[face_index])
            next_subdivision = (
                3 if current_subdivision == 0 else (current_subdivision + 3)
            )
            if next_subdivision > int(desired[face_index]):
                continue
            current_sample_count = 4
            if current_subdivision:
                current_sample_count = (
                    (current_subdivision + 1) * (current_subdivision + 2)
                ) // 2
            next_sample_count = ((next_subdivision + 1) * (next_subdivision + 2)) // 2
            incremental_sample_count = next_sample_count - current_sample_count
            if incremental_sample_count > remaining_samples:
                continue
            allocated[face_index] = next_subdivision
            remaining_samples -= incremental_sample_count
            allocation_changed = True
        if not allocation_changed:
            break
    _check_cancelled(cancellation_check)
    return allocated


def _build_centroid_face_patch(
    world_triangle: np.ndarray,
    pixel_triangle: np.ndarray,
    normal: np.ndarray,
) -> _FaceSamplingPatch:
    world_positions = np.vstack((world_triangle, np.mean(world_triangle, axis=0)))
    pixel_coordinates = np.vstack((pixel_triangle, np.mean(pixel_triangle, axis=0)))
    return _FaceSamplingPatch(
        pixel_coordinates=np.ascontiguousarray(pixel_coordinates),
        world_positions=np.ascontiguousarray(world_positions),
        raster_triangles=np.asarray(
            ((0, 1, 3), (1, 2, 3), (2, 0, 3)),
            dtype=np.int64,
        ),
        normal=normal,
    )


def _build_subdivided_face_patch(
    world_triangle: np.ndarray,
    pixel_triangle: np.ndarray,
    normal: np.ndarray,
    subdivision: int,
    *,
    cancellation_check: Callable[[], bool] | None = None,
) -> _FaceSamplingPatch:
    sample_indices: dict[tuple[int, int], int] = {}
    barycentric_coordinates: list[tuple[float, float, float]] = []
    denominator = float(subdivision)
    for first_index in range(subdivision + 1):
        _check_cancelled(cancellation_check)
        for second_index in range(subdivision - first_index + 1):
            sample_indices[(first_index, second_index)] = len(barycentric_coordinates)
            first_weight = first_index / denominator
            second_weight = second_index / denominator
            barycentric_coordinates.append(
                (1.0 - first_weight - second_weight, first_weight, second_weight)
            )
    barycentric = np.asarray(barycentric_coordinates, dtype=float)

    raster_triangles: list[tuple[int, int, int]] = []
    for first_index in range(subdivision):
        _check_cancelled(cancellation_check)
        for second_index in range(subdivision - first_index):
            first = sample_indices[(first_index, second_index)]
            second = sample_indices[(first_index + 1, second_index)]
            third = sample_indices[(first_index, second_index + 1)]
            raster_triangles.append((first, second, third))
            if first_index + second_index < subdivision - 1:
                fourth = sample_indices[(first_index + 1, second_index + 1)]
                raster_triangles.append((second, fourth, third))

    result = _FaceSamplingPatch(
        pixel_coordinates=np.ascontiguousarray(barycentric @ pixel_triangle),
        world_positions=np.ascontiguousarray(barycentric @ world_triangle),
        raster_triangles=np.asarray(raster_triangles, dtype=np.int64),
        normal=normal,
    )
    _check_cancelled(cancellation_check)
    return result


def _reflect_receiver_samples(
    positions: np.ndarray,
    normals: np.ndarray,
    target: ObjectAmbientOcclusionTarget,
) -> tuple[np.ndarray, np.ndarray]:
    point = np.asarray(target.mirror_plane_point, dtype=float)
    plane_normal = np.asarray(target.mirror_plane_normal, dtype=float)
    position_distances = (positions - point) @ plane_normal
    normal_components = normals @ plane_normal
    mirrored_positions = positions - (
        2.0 * position_distances[:, np.newaxis] * plane_normal
    )
    mirrored_normals = normals - (2.0 * normal_components[:, np.newaxis] * plane_normal)
    return (
        np.ascontiguousarray(mirrored_positions),
        np.ascontiguousarray(mirrored_normals),
    )


def _sample_ambient_occlusion(
    positions: np.ndarray,
    normals: np.ndarray,
    intersector: _RayIntersector,
    ray_bias: float,
    *,
    cancellation_check: Callable[[], bool] | None = None,
) -> np.ndarray:
    _check_cancelled(cancellation_check)
    local_directions = _build_cosine_weighted_hemisphere_directions(OBJECT_AO_RAY_COUNT)
    ambient_occlusion = np.ones(len(positions), dtype=float)
    for start in range(0, len(positions), OBJECT_AO_SAMPLE_BATCH_SIZE):
        _check_cancelled(cancellation_check)
        end = min(start + OBJECT_AO_SAMPLE_BATCH_SIZE, len(positions))
        batch_positions = positions[start:end]
        batch_normals = normals[start:end]
        directions = _orient_hemisphere_directions(
            batch_normals,
            local_directions,
        )
        origins = np.broadcast_to(
            (batch_positions + batch_normals * ray_bias)[:, np.newaxis, :],
            directions.shape,
        ).reshape((-1, 3))
        flat_directions = directions.reshape((-1, 3))
        locations, ray_indices, _triangle_indices = intersector.intersects_location(
            origins,
            flat_directions,
            multiple_hits=False,
        )
        occluded = np.zeros(len(origins), dtype=bool)
        if len(ray_indices):
            distances = np.linalg.norm(
                np.asarray(locations, dtype=float) - origins[ray_indices],
                axis=1,
            )
            within_distance = (
                np.isfinite(distances)
                & (distances > GEOMETRY_EPSILON)
                & (distances <= OBJECT_AO_MAX_DISTANCE_METERS)
            )
            occluded[np.asarray(ray_indices, dtype=np.int64)[within_distance]] = True
        ambient_occlusion[start:end] = 1.0 - np.mean(
            occluded.reshape((-1, OBJECT_AO_RAY_COUNT)),
            axis=1,
        )
    _check_cancelled(cancellation_check)
    return np.ascontiguousarray(np.clip(ambient_occlusion, 0.0, 1.0))


def _call_sample_ambient_occlusion(
    positions: np.ndarray,
    normals: np.ndarray,
    intersector: _RayIntersector,
    ray_bias: float,
    cancellation_check: Callable[[], bool] | None,
) -> np.ndarray:
    """Preserve the legacy internal call shape when cancellation is unused."""

    if cancellation_check is None:
        return _sample_ambient_occlusion(
            positions,
            normals,
            intersector,
            ray_bias,
        )
    return _sample_ambient_occlusion(
        positions,
        normals,
        intersector,
        ray_bias,
        cancellation_check=cancellation_check,
    )


def _build_cosine_weighted_hemisphere_directions(count: int) -> np.ndarray:
    indices = np.arange(count, dtype=float)
    radius_squared = (indices + 0.5) / float(count)
    angle_turns = np.mod(indices * ((math.sqrt(5.0) - 1.0) / 2.0), 1.0)
    angles = angle_turns * (2.0 * math.pi)
    radius = np.sqrt(radius_squared)
    return np.column_stack(
        (
            radius * np.cos(angles),
            radius * np.sin(angles),
            np.sqrt(1.0 - radius_squared),
        )
    )


def _orient_hemisphere_directions(
    normals: np.ndarray,
    local_directions: np.ndarray,
) -> np.ndarray:
    references = np.tile(np.asarray((0.0, 0.0, 1.0)), (len(normals), 1))
    near_vertical = np.abs(normals[:, 2]) > 0.9
    references[near_vertical] = np.asarray((0.0, 1.0, 0.0))
    tangents = _normalize_rows(np.cross(references, normals))
    bitangents = _normalize_rows(np.cross(normals, tangents))
    directions = (
        tangents[:, np.newaxis, :] * local_directions[np.newaxis, :, 0:1]
        + bitangents[:, np.newaxis, :] * local_directions[np.newaxis, :, 1:2]
        + normals[:, np.newaxis, :] * local_directions[np.newaxis, :, 2:3]
    )
    normalized = _normalize_rows(directions.reshape((-1, 3)))
    return np.ascontiguousarray(
        normalized.reshape((len(normals), len(local_directions), 3))
    )


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    normalized = np.asarray(values, dtype=float).copy()
    lengths = np.linalg.norm(normalized, axis=1)
    usable = np.isfinite(lengths) & (lengths > GEOMETRY_EPSILON)
    normalized[usable] /= lengths[usable, np.newaxis]
    normalized[~usable] = np.asarray((0.0, 0.0, 1.0))
    return normalized


# ### UV rasterization helpers ###
def _uv_to_top_origin_pixels(
    uv: np.ndarray,
    resolution: int,
) -> np.ndarray:
    return np.ascontiguousarray(
        np.column_stack(
            (
                uv[:, 0] * resolution - 0.5,
                (1.0 - uv[:, 1]) * resolution - 0.5,
            )
        )
    )


def _rasterize_sampling_patch(
    output: np.ndarray,
    coverage: np.ndarray,
    patch: _FaceSamplingPatch,
    sample_values: np.ndarray,
    *,
    cancellation_check: Callable[[], bool] | None = None,
) -> None:
    for triangle_index, triangle in enumerate(patch.raster_triangles):
        if triangle_index % OBJECT_AO_CANCELLATION_POLL_INTERVAL == 0:
            _check_cancelled(cancellation_check)
        _rasterize_scalar_triangle(
            output,
            coverage,
            patch.pixel_coordinates[triangle],
            sample_values[triangle],
            cancellation_check=cancellation_check,
        )
    _check_cancelled(cancellation_check)


def _rasterize_scalar_triangle(
    output: np.ndarray,
    coverage: np.ndarray,
    coordinates: np.ndarray,
    values: np.ndarray,
    *,
    cancellation_check: Callable[[], bool] | None = None,
) -> None:
    _check_cancelled(cancellation_check)
    height, width = output.shape
    minimum_x = max(0, math.floor(float(np.min(coordinates[:, 0]))))
    maximum_x = min(
        width - 1,
        math.ceil(float(np.max(coordinates[:, 0]))),
    )
    minimum_y = max(0, math.floor(float(np.min(coordinates[:, 1]))))
    maximum_y = min(
        height - 1,
        math.ceil(float(np.max(coordinates[:, 1]))),
    )
    if maximum_x < minimum_x or maximum_y < minimum_y:
        return

    x0, y0 = coordinates[0]
    x1, y1 = coordinates[1]
    x2, y2 = coordinates[2]
    denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if not math.isfinite(float(denominator)) or abs(denominator) <= (GEOMETRY_EPSILON):
        return

    columns = np.arange(minimum_x, maximum_x + 1, dtype=float)[np.newaxis, :]
    for row_start in range(
        minimum_y,
        maximum_y + 1,
        OBJECT_AO_RASTER_ROW_BATCH_SIZE,
    ):
        _check_cancelled(cancellation_check)
        row_end = min(
            row_start + OBJECT_AO_RASTER_ROW_BATCH_SIZE,
            maximum_y + 1,
        )
        rows = np.arange(row_start, row_end, dtype=float)[:, np.newaxis]
        weight_0 = ((y1 - y2) * (columns - x2) + (x2 - x1) * (rows - y2)) / denominator
        weight_1 = ((y2 - y0) * (columns - x2) + (x0 - x2) * (rows - y2)) / denominator
        weight_2 = 1.0 - weight_0 - weight_1
        inside = (
            (weight_0 >= -RASTER_EPSILON)
            & (weight_1 >= -RASTER_EPSILON)
            & (weight_2 >= -RASTER_EPSILON)
        )
        if not np.any(inside):
            continue
        interpolated = (
            weight_0 * values[0] + weight_1 * values[1] + weight_2 * values[2]
        )
        encoded = np.asarray(
            np.rint(np.clip(interpolated, 0.0, 1.0) * 255.0),
            dtype=np.uint8,
        )
        region = output[row_start:row_end, minimum_x : maximum_x + 1]
        region[inside] = np.minimum(region[inside], encoded[inside])
        coverage_region = coverage[
            row_start:row_end,
            minimum_x : maximum_x + 1,
        ]
        coverage_region[inside] = True
    _check_cancelled(cancellation_check)


def _pad_ambient_occlusion_coverage(
    values: np.ndarray,
    coverage: np.ndarray,
    padding_pixels: int,
    *,
    cancellation_check: Callable[[], bool] | None = None,
) -> None:
    _check_cancelled(cancellation_check)
    if padding_pixels <= 0 or not np.any(coverage):
        return
    height, width = coverage.shape
    neighbor_offsets = tuple(
        (row_offset, column_offset)
        for row_offset in (-1, 0, 1)
        for column_offset in (-1, 0, 1)
        if row_offset or column_offset
    )
    for _step in range(padding_pixels):
        _check_cancelled(cancellation_check)
        neighbor_coverage = np.zeros_like(coverage)
        neighbor_values = np.full_like(values, 255)
        for row_offset, column_offset in neighbor_offsets:
            source_rows = slice(
                max(0, -row_offset),
                min(height, height - row_offset),
            )
            destination_rows = slice(
                max(0, row_offset),
                min(height, height + row_offset),
            )
            source_columns = slice(
                max(0, -column_offset),
                min(width, width - column_offset),
            )
            destination_columns = slice(
                max(0, column_offset),
                min(width, width + column_offset),
            )
            source_mask = coverage[source_rows, source_columns]
            destination_mask = neighbor_coverage[
                destination_rows,
                destination_columns,
            ]
            destination_values = neighbor_values[
                destination_rows,
                destination_columns,
            ]
            source_values = values[source_rows, source_columns]
            destination_values[source_mask] = np.minimum(
                destination_values[source_mask],
                source_values[source_mask],
            )
            destination_mask |= source_mask
        added = ~coverage & neighbor_coverage
        if not np.any(added):
            break
        values[added] = neighbor_values[added]
        coverage[added] = True
    _check_cancelled(cancellation_check)


# ### Cancellation helpers ###
def _check_cancelled(
    cancellation_check: Callable[[], bool] | None,
) -> None:
    if cancellation_check is not None and bool(cancellation_check()):
        raise ObjectAmbientOcclusionBakeCancelled(
            "Ambient-occlusion baking was cancelled."
        )


# ### Mesh attribute helpers ###
def _get_valid_uv(mesh: trimesh.Trimesh) -> np.ndarray | None:
    raw_uv = getattr(getattr(mesh, "visual", None), "uv", None)
    if raw_uv is None:
        return None
    uv = np.asarray(raw_uv, dtype=float)
    if uv.shape != (len(mesh.vertices), 2) or not np.all(np.isfinite(uv)):
        return None
    return np.ascontiguousarray(uv)


__all__ = [
    "OBJECT_AO_MAX_DISTANCE_METERS",
    "OBJECT_AO_RAY_COUNT",
    "ObjectAmbientOcclusionBakeCancelled",
    "ObjectAmbientOcclusionTarget",
    "bake_placed_object_ambient_occlusion",
]
