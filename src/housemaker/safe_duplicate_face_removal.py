# ### Imports ###
from __future__ import annotations

import hashlib
import itertools
import math
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image

from housemaker.glb import GLTF_Y_UP_TO_Z_UP_TRANSFORM
from housemaker.object_face_edit import (
    _MeshInstance,
    _collect_scene_geometry,
    _export_instances,
    _filter_instances,
    _load_glb_scene,
    _validate_exported_face_count,
)


# ### Constants ###
DEFAULT_RELATIVE_VERTEX_TOLERANCE = 1e-7
_MINIMUM_VERTEX_TOLERANCE = np.finfo(np.float64).eps * 64.0
_MINIMUM_NORMAL_LENGTH = np.finfo(np.float64).eps * 64.0
_MINIMUM_SAME_FACING_COSINE = 1.0 - 1e-7
_UV_EQUIVALENCE_TOLERANCE = 1e-7
_NORMAL_EQUIVALENCE_TOLERANCE = 1e-6
_CANCELLATION_CHECK_INTERVAL = 256
_CORNER_PERMUTATIONS = tuple(itertools.permutations(range(3)))
_NEIGHBOR_OFFSETS = tuple(itertools.product((-1, 0, 1), repeat=3))


# ### Public data models ###
@dataclass(frozen=True)
class SafeDuplicateFaceRemovalResult:
    """A conservative duplicate-only topology cleanup result."""

    glb_bytes: bytes
    original_face_count: int
    retained_face_count: int
    removed_face_count: int
    duplicate_group_count: int
    vertex_tolerance: float

    @property
    def changed(self) -> bool:
        """Return whether at least one duplicate face was removed."""

        return self.removed_face_count > 0


# ### Public exceptions ###
class SafeDuplicateFaceRemovalCancelled(RuntimeError):
    """Raised when a duplicate-removal generation step is cancelled."""


# ### Internal data models ###
@dataclass(frozen=True)
class _FaceRecord:
    face_index: int
    triangle: np.ndarray
    centroid: np.ndarray
    unit_normal: np.ndarray | None
    visual_identity: _FaceVisualIdentity


@dataclass(frozen=True)
class _FaceVisualIdentity:
    material_key: Hashable
    face_color: tuple[int, ...] | None
    uv_triangle: np.ndarray | None
    vertex_color_triangle: np.ndarray | None
    vertex_normal_triangle: np.ndarray | None

    @property
    def bucket_key(self) -> Hashable:
        """Return cheap visual attributes that require exact equivalence."""

        return self.material_key, self.face_color


# ### Public API ###
def remove_safe_duplicate_faces_from_glb(
    glb_bytes: bytes,
    *,
    relative_vertex_tolerance: float = DEFAULT_RELATIVE_VERTEX_TOLERANCE,
    cancel_requested: Callable[[], bool] | None = None,
) -> SafeDuplicateFaceRemovalResult:
    """Remove same-facing duplicate triangles without touching nearby layers.

    Triangle corners are compared in world space independently of their cyclic
    order. Only faces with equivalent material identities are candidates. The
    first face in the canonical object-face order is retained deterministically.
    """

    relative_tolerance = _normalize_relative_tolerance(
        relative_vertex_tolerance
    )
    _raise_if_cancelled(cancel_requested)
    source_payload = bytes(glb_bytes)
    scene = _load_glb_scene(source_payload)
    _raise_if_cancelled(cancel_requested)
    instances, geometry = _collect_scene_geometry(scene)
    original_face_count = geometry.face_count
    if original_face_count == 0:
        raise ValueError("The object GLB contains no triangle faces.")

    vertex_tolerance = _resolve_vertex_tolerance(
        geometry.vertices,
        relative_tolerance,
    )
    visual_identities = _collect_face_visual_identities(
        instances,
        cancel_requested=cancel_requested,
    )
    duplicate_indices, duplicate_group_count = _find_duplicate_face_indices(
        geometry.vertices,
        geometry.faces,
        visual_identities,
        vertex_tolerance=vertex_tolerance,
        cancel_requested=cancel_requested,
    )
    removed_face_count = len(duplicate_indices)
    retained_face_count = original_face_count - removed_face_count
    if not duplicate_indices:
        return SafeDuplicateFaceRemovalResult(
            glb_bytes=source_payload,
            original_face_count=original_face_count,
            retained_face_count=original_face_count,
            removed_face_count=0,
            duplicate_group_count=0,
            vertex_tolerance=vertex_tolerance,
        )

    keep_faces = np.ones(original_face_count, dtype=bool)
    keep_faces[np.asarray(sorted(duplicate_indices), dtype=np.int64)] = False
    _raise_if_cancelled(cancel_requested)
    filtered_instances = _filter_instances(instances, keep_faces)
    _raise_if_cancelled(cancel_requested)
    edited_glb = _export_instances(
        filtered_instances,
        scene_metadata=scene.metadata,
    )
    _validate_exported_face_count(edited_glb, retained_face_count)
    return SafeDuplicateFaceRemovalResult(
        glb_bytes=edited_glb,
        original_face_count=original_face_count,
        retained_face_count=retained_face_count,
        removed_face_count=removed_face_count,
        duplicate_group_count=duplicate_group_count,
        vertex_tolerance=vertex_tolerance,
    )


# ### Option validation ###
def _normalize_relative_tolerance(value: float) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        int | float | np.integer | np.floating,
    ):
        raise TypeError("Duplicate-face tolerance must be a finite number.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(
            "Duplicate-face tolerance must be finite and greater than zero."
        )
    return normalized


def _raise_if_cancelled(
    cancel_requested: Callable[[], bool] | None,
) -> None:
    if cancel_requested is not None and cancel_requested():
        raise SafeDuplicateFaceRemovalCancelled(
            "Safe duplicate-face removal was cancelled."
        )


def _resolve_vertex_tolerance(
    vertices: np.ndarray,
    relative_tolerance: float,
) -> float:
    minimum = np.min(vertices, axis=0)
    maximum = np.max(vertices, axis=0)
    diagonal = float(np.linalg.norm(maximum - minimum))
    if not math.isfinite(diagonal):
        raise ValueError("The object GLB contains invalid vertex coordinates.")
    return max(diagonal * relative_tolerance, _MINIMUM_VERTEX_TOLERANCE)


# ### Duplicate detection ###
def _find_duplicate_face_indices(
    vertices: np.ndarray,
    faces: np.ndarray,
    visual_identities: Sequence[_FaceVisualIdentity],
    *,
    vertex_tolerance: float,
    cancel_requested: Callable[[], bool] | None,
) -> tuple[frozenset[int], int]:
    if len(visual_identities) != len(faces):
        raise ValueError("Duplicate-face material indexing is inconsistent.")

    triangles = np.asarray(vertices[faces], dtype=float)
    retained_buckets: dict[
        tuple[Hashable, tuple[int, int, int]],
        list[_FaceRecord],
    ] = {}
    duplicate_indices: set[int] = set()
    duplicate_group_roots: set[int] = set()

    for face_index, triangle in enumerate(triangles):
        if face_index % _CANCELLATION_CHECK_INTERVAL == 0:
            _raise_if_cancelled(cancel_requested)
        record = _build_face_record(
            face_index,
            triangle,
            visual_identities[face_index],
        )
        cell = _centroid_cell(record.centroid, vertex_tolerance)
        duplicate_root: int | None = None
        for offset in _NEIGHBOR_OFFSETS:
            neighboring_cell = tuple(
                cell[axis] + offset[axis] for axis in range(3)
            )
            candidates = retained_buckets.get(
                (record.visual_identity.bucket_key, neighboring_cell),
                (),
            )
            for candidate in candidates:
                if _faces_are_safe_duplicates(
                    candidate,
                    record,
                    vertex_tolerance=vertex_tolerance,
                ):
                    duplicate_root = candidate.face_index
                    break
            if duplicate_root is not None:
                break

        if duplicate_root is not None:
            duplicate_indices.add(record.face_index)
            duplicate_group_roots.add(duplicate_root)
            continue
        retained_buckets.setdefault(
            (record.visual_identity.bucket_key, cell),
            [],
        ).append(record)

    return frozenset(duplicate_indices), len(duplicate_group_roots)


def _build_face_record(
    face_index: int,
    triangle: np.ndarray,
    visual_identity: _FaceVisualIdentity,
) -> _FaceRecord:
    edge_a = triangle[1] - triangle[0]
    edge_b = triangle[2] - triangle[0]
    normal = np.cross(edge_a, edge_b)
    normal_length = float(np.linalg.norm(normal))
    unit_normal = (
        normal / normal_length
        if normal_length > _MINIMUM_NORMAL_LENGTH
        else None
    )
    return _FaceRecord(
        face_index=face_index,
        triangle=np.ascontiguousarray(triangle, dtype=float),
        centroid=np.mean(triangle, axis=0),
        unit_normal=unit_normal,
        visual_identity=visual_identity,
    )


def _centroid_cell(
    centroid: np.ndarray,
    vertex_tolerance: float,
) -> tuple[int, int, int]:
    return tuple(
        math.floor(float(coordinate) / vertex_tolerance)
        for coordinate in centroid
    )


def _faces_are_safe_duplicates(
    retained: _FaceRecord,
    candidate: _FaceRecord,
    *,
    vertex_tolerance: float,
) -> bool:
    if retained.unit_normal is None or candidate.unit_normal is None:
        return False
    normal_alignment = float(
        np.dot(retained.unit_normal, candidate.unit_normal)
    )
    if normal_alignment < _MINIMUM_SAME_FACING_COSINE:
        return False

    for permutation in _CORNER_PERMUTATIONS:
        reordered = candidate.triangle[np.asarray(permutation, dtype=np.int64)]
        corner_distances = np.linalg.norm(
            retained.triangle - reordered,
            axis=1,
        )
        if (
            np.all(corner_distances <= vertex_tolerance)
            and _visual_attributes_match(
                retained.visual_identity,
                candidate.visual_identity,
                candidate_permutation=permutation,
            )
        ):
            return True
    return False


def _visual_attributes_match(
    retained: _FaceVisualIdentity,
    candidate: _FaceVisualIdentity,
    *,
    candidate_permutation: tuple[int, int, int],
) -> bool:
    if retained.bucket_key != candidate.bucket_key:
        return False
    if not _optional_corner_values_match(
        retained.uv_triangle,
        candidate.uv_triangle,
        candidate_permutation=candidate_permutation,
        absolute_tolerance=_UV_EQUIVALENCE_TOLERANCE,
    ):
        return False
    if not _optional_corner_values_match(
        retained.vertex_color_triangle,
        candidate.vertex_color_triangle,
        candidate_permutation=candidate_permutation,
        absolute_tolerance=0.0,
    ):
        return False
    if (
        retained.vertex_normal_triangle is None
        or candidate.vertex_normal_triangle is None
    ):
        return False
    return _optional_corner_values_match(
        retained.vertex_normal_triangle,
        candidate.vertex_normal_triangle,
        candidate_permutation=candidate_permutation,
        absolute_tolerance=_NORMAL_EQUIVALENCE_TOLERANCE,
    )


def _optional_corner_values_match(
    retained: np.ndarray | None,
    candidate: np.ndarray | None,
    *,
    candidate_permutation: tuple[int, int, int],
    absolute_tolerance: float,
) -> bool:
    if retained is None or candidate is None:
        return retained is None and candidate is None
    reordered_candidate = candidate[
        np.asarray(candidate_permutation, dtype=np.int64)
    ]
    return bool(
        np.allclose(
            retained,
            reordered_candidate,
            rtol=0.0,
            atol=absolute_tolerance,
        )
    )


# ### Visual identity ###
def _collect_face_visual_identities(
    instances: Sequence[_MeshInstance],
    *,
    cancel_requested: Callable[[], bool] | None,
) -> tuple[_FaceVisualIdentity, ...]:
    identities: list[_FaceVisualIdentity] = []
    fingerprint_cache: dict[int, Hashable] = {}
    for instance in instances:
        _raise_if_cancelled(cancel_requested)
        mesh = instance.mesh
        material = getattr(mesh.visual, "material", None)
        material_leaves = _material_leaves(material)
        face_material_indices = _face_material_indices(
            mesh,
            material_count=len(material_leaves),
        )
        face_colors = _face_colors(mesh)
        uv_triangles = _uv_triangles(mesh)
        vertex_color_triangles = _vertex_color_triangles(mesh)
        vertex_normal_triangles = _world_vertex_normal_triangles(instance)
        for local_face_index in range(instance.face_count):
            material_index = int(face_material_indices[local_face_index])
            leaf = (
                material_leaves[material_index]
                if material_leaves
                else None
            )
            material_key = _material_fingerprint(leaf, fingerprint_cache)
            color_key = (
                tuple(int(value) for value in face_colors[local_face_index])
                if face_colors is not None
                else None
            )
            identities.append(
                _FaceVisualIdentity(
                    material_key=material_key,
                    face_color=color_key,
                    uv_triangle=(
                        uv_triangles[local_face_index]
                        if uv_triangles is not None
                        else None
                    ),
                    vertex_color_triangle=(
                        vertex_color_triangles[local_face_index]
                        if vertex_color_triangles is not None
                        else None
                    ),
                    vertex_normal_triangle=(
                        vertex_normal_triangles[local_face_index]
                        if vertex_normal_triangles is not None
                        else None
                    ),
                )
            )
    return tuple(identities)


def _material_leaves(material: object) -> tuple[object, ...]:
    if material is None:
        return ()
    nested = getattr(material, "materials", None)
    if isinstance(nested, list | tuple):
        return tuple(nested)
    return (material,)


def _face_material_indices(
    mesh: object,
    *,
    material_count: int,
) -> np.ndarray:
    face_count = len(mesh.faces)
    raw_indices = getattr(mesh.visual, "face_materials", None)
    if raw_indices is None:
        return np.zeros(face_count, dtype=np.int64)
    indices = np.asarray(raw_indices, dtype=np.int64)
    if indices.shape != (face_count,):
        raise ValueError("The object GLB contains invalid face material indices.")
    if material_count and (
        np.any(indices < 0) or np.any(indices >= material_count)
    ):
        raise ValueError("The object GLB contains invalid face material indices.")
    if not material_count and np.any(indices != 0):
        raise ValueError("The object GLB contains invalid face material indices.")
    return indices


def _face_colors(mesh: object) -> np.ndarray | None:
    if getattr(mesh.visual, "kind", None) not in {"face", "vertex"}:
        return None
    try:
        colors = np.asarray(mesh.visual.face_colors, dtype=np.uint8)
    except (AttributeError, TypeError, ValueError):
        return None
    if colors.shape != (len(mesh.faces), 4):
        return None
    return colors


def _uv_triangles(mesh: object) -> np.ndarray | None:
    try:
        uvs = np.asarray(getattr(mesh.visual, "uv", None), dtype=float)
    except (TypeError, ValueError):
        return None
    if uvs.shape != (len(mesh.vertices), 2) or not np.all(np.isfinite(uvs)):
        return None
    return np.ascontiguousarray(uvs[np.asarray(mesh.faces, dtype=np.int64)])


def _vertex_color_triangles(mesh: object) -> np.ndarray | None:
    if getattr(mesh.visual, "kind", None) not in {"face", "vertex"}:
        return None
    try:
        colors = np.asarray(mesh.visual.vertex_colors, dtype=np.uint8)
    except (AttributeError, TypeError, ValueError):
        return None
    if colors.shape != (len(mesh.vertices), 4):
        return None
    return np.ascontiguousarray(
        colors[np.asarray(mesh.faces, dtype=np.int64)]
    )


def _world_vertex_normal_triangles(
    instance: _MeshInstance,
) -> np.ndarray | None:
    """Transform authored normals into the same world basis as face corners."""

    local_normals = np.asarray(instance.vertex_normals, dtype=float)
    mesh = instance.mesh
    if (
        local_normals.shape != (len(mesh.vertices), 3)
        or not np.all(np.isfinite(local_normals))
    ):
        return None

    combined_transform = (
        np.asarray(GLTF_Y_UP_TO_Z_UP_TRANSFORM, dtype=float)
        @ np.asarray(instance.transform, dtype=float)
    )
    linear_transform = combined_transform[:3, :3]
    try:
        normal_matrix = np.linalg.inv(linear_transform).T
    except np.linalg.LinAlgError:
        return None
    world_normals = local_normals @ normal_matrix.T
    lengths = np.linalg.norm(world_normals, axis=1)
    if (
        not np.all(np.isfinite(world_normals))
        or np.any(lengths <= _MINIMUM_NORMAL_LENGTH)
    ):
        return None
    world_normals /= lengths[:, np.newaxis]
    return np.ascontiguousarray(
        world_normals[np.asarray(mesh.faces, dtype=np.int64)]
    )


def _material_fingerprint(
    material: object,
    cache: dict[int, Hashable],
) -> Hashable:
    if material is None:
        return ("no-material",)
    cache_key = id(material)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    fingerprint = (
        type(material).__module__,
        type(material).__qualname__,
        _freeze_fingerprint_value(getattr(material, "__dict__", {})),
    )
    cache[cache_key] = fingerprint
    return fingerprint


def _freeze_fingerprint_value(value: object) -> Hashable:
    if value is None or isinstance(value, str | bytes | int | float | bool):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return (
            "ndarray",
            contiguous.dtype.str,
            contiguous.shape,
            hashlib.sha256(contiguous.tobytes()).digest(),
        )
    if isinstance(value, Image.Image):
        pixels = value.convert("RGBA")
        return (
            "image",
            pixels.size,
            hashlib.sha256(pixels.tobytes()).digest(),
        )
    if isinstance(value, dict):
        return tuple(
            sorted(
                (
                    str(key),
                    _freeze_fingerprint_value(nested_value),
                )
                for key, nested_value in value.items()
            )
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_fingerprint_value(item) for item in value)
    return (type(value).__module__, type(value).__qualname__, repr(value))
