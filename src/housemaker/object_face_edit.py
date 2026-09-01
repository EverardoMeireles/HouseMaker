# ### Imports ###
from __future__ import annotations

import copy
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from io import BytesIO

import numpy as np
import trimesh

from housemaker.glass_material import is_housemaker_glass_material
from housemaker.glb import GLTF_Y_UP_TO_Z_UP_TRANSFORM


# ### Constants ###
_NORMAL_EPSILON = 1e-12


# ### Public data models ###
@dataclass(frozen=True)
class ObjectFaceGeometry:
    """Flattened geometry and UVs sharing stable global triangle indices."""

    vertices: np.ndarray
    faces: np.ndarray
    uv_triangles: np.ndarray
    uv_face_indices: np.ndarray

    @property
    def face_count(self) -> int:
        return int(len(self.faces))


@dataclass(frozen=True)
class ObjectFaceDeletionResult:
    """One topology-only face edit retaining existing UVs and materials."""

    glb_bytes: bytes
    original_face_count: int
    retained_face_count: int
    deleted_face_count: int
    preserved_textured_uvs: bool


# ### Internal data models ###
@dataclass
class _MeshInstance:
    node_name: str
    geometry_name: str
    mesh: trimesh.Trimesh
    transform: np.ndarray
    first_face_index: int
    vertex_normals: np.ndarray

    @property
    def face_count(self) -> int:
        return int(len(self.mesh.faces))


# ### Public API ###
def load_object_face_geometry(glb_bytes: bytes) -> ObjectFaceGeometry:
    """Load the exact deterministic face index space used by deletion."""

    scene = _load_glb_scene(glb_bytes)
    return load_object_face_geometry_from_scene(scene)


def load_object_face_geometry_from_scene(
    scene: trimesh.Scene,
) -> ObjectFaceGeometry:
    """Index an already imported scene without decoding its GLB again."""

    if not isinstance(scene, trimesh.Scene):
        raise TypeError("Object face geometry requires a trimesh scene.")
    _instances, geometry = _collect_scene_geometry(
        scene,
        collect_instances=False,
    )
    if geometry.face_count == 0:
        raise ValueError("The object GLB contains no triangle faces.")
    return geometry


def delete_object_faces_preserving_uvs(
    glb_bytes: bytes,
    selected_face_indices: Iterable[int],
) -> ObjectFaceDeletionResult:
    """Delete selected faces without changing retained UVs or textures."""

    result, _geometry = _delete_object_faces_preserving_uvs_with_geometry(
        glb_bytes,
        selected_face_indices,
        validate_export=True,
    )
    return result


def _delete_object_faces_preserving_uvs_with_geometry(
    glb_bytes: bytes,
    selected_face_indices: Iterable[int],
    *,
    validate_export: bool,
) -> tuple[ObjectFaceDeletionResult, ObjectFaceGeometry]:
    """Filter one GLB while returning its already parsed source geometry."""

    scene = _load_glb_scene(glb_bytes)
    instances, geometry = _collect_scene_geometry(scene)
    original_face_count = geometry.face_count
    if original_face_count == 0:
        raise ValueError("The object GLB contains no triangle faces.")
    selected = _normalize_selected_face_indices(
        selected_face_indices,
        face_count=original_face_count,
    )
    if len(selected) == original_face_count:
        raise ValueError("Face deletion cannot remove every object face.")

    keep_faces = np.ones(original_face_count, dtype=bool)
    keep_faces[np.fromiter(selected, dtype=np.int64)] = False
    filtered_instances = _filter_instances(instances, keep_faces)
    retained_face_count = sum(
        instance.face_count for instance in filtered_instances
    )
    edited_glb = _export_instances(
        filtered_instances,
        scene_metadata=scene.metadata,
    )
    if validate_export:
        _validate_exported_face_count(edited_glb, retained_face_count)
    return (
        ObjectFaceDeletionResult(
            glb_bytes=edited_glb,
            original_face_count=original_face_count,
            retained_face_count=retained_face_count,
            deleted_face_count=original_face_count - retained_face_count,
            preserved_textured_uvs=(
                _instances_have_textured_uvs(filtered_instances)
            ),
        ),
        geometry,
    )


# ### Scene loading and indexing ###
def _load_glb_scene(glb_bytes: bytes) -> trimesh.Scene:
    payload = bytes(glb_bytes)
    if not payload:
        raise ValueError("The object GLB is empty.")
    try:
        loaded = trimesh.load(
            BytesIO(payload),
            file_type="glb",
            force="scene",
            process=False,
        )
    except Exception as error:
        raise ValueError("The object GLB could not be loaded.") from error
    if isinstance(loaded, trimesh.Trimesh):
        return trimesh.Scene(loaded)
    if isinstance(loaded, trimesh.Scene):
        return loaded
    raise ValueError("The object GLB contains no mesh scene.")


def _collect_scene_geometry(
    scene: trimesh.Scene,
    *,
    collect_instances: bool = True,
) -> tuple[list[_MeshInstance], ObjectFaceGeometry]:
    instances: list[_MeshInstance] = []
    world_vertices: list[np.ndarray] = []
    global_faces: list[np.ndarray] = []
    uv_triangles: list[np.ndarray] = []
    uv_face_indices: list[np.ndarray] = []
    vertex_offset = 0
    face_offset = 0
    for node_name in sorted(scene.graph.nodes_geometry, key=str):
        transform, geometry_name = scene.graph.get(node_name)
        geometry = scene.geometry.get(geometry_name)
        if not isinstance(geometry, trimesh.Trimesh):
            continue
        if len(geometry.vertices) == 0 or len(geometry.faces) == 0:
            continue
        local_vertices = np.asarray(geometry.vertices, dtype=float)
        local_faces = np.asarray(geometry.faces, dtype=np.int64)
        node_transform = np.asarray(transform, dtype=float)
        _validate_mesh_arrays(local_vertices, local_faces, node_transform)
        transformed_vertices = trimesh.transform_points(
            local_vertices,
            node_transform,
        )
        world_vertices.append(
            trimesh.transform_points(
                transformed_vertices,
                GLTF_Y_UP_TO_Z_UP_TRANSFORM,
            )
        )
        global_faces.append(local_faces + vertex_offset)
        local_uv_triangles, local_uv_face_indices = (
            _collect_instance_uv_triangles(
                geometry,
                local_faces,
                first_face_index=face_offset,
            )
        )
        if len(local_uv_triangles):
            uv_triangles.append(local_uv_triangles)
            uv_face_indices.append(local_uv_face_indices)
        if collect_instances:
            instances.append(
                _MeshInstance(
                    node_name=str(node_name),
                    geometry_name=str(geometry_name),
                    mesh=geometry.copy(),
                    transform=node_transform.copy(),
                    first_face_index=face_offset,
                    vertex_normals=_get_vertex_normals(geometry),
                )
            )
        vertex_offset += len(local_vertices)
        face_offset += len(local_faces)
    if not world_vertices:
        return (
            instances,
            ObjectFaceGeometry(
                vertices=np.empty((0, 3), dtype=float),
                faces=np.empty((0, 3), dtype=np.int64),
                uv_triangles=np.empty((0, 3, 2), dtype=float),
                uv_face_indices=np.empty((0,), dtype=np.int64),
            ),
        )
    return (
        instances,
        ObjectFaceGeometry(
            vertices=np.ascontiguousarray(np.vstack(world_vertices), dtype=float),
            faces=np.ascontiguousarray(np.vstack(global_faces), dtype=np.int64),
            uv_triangles=(
                np.ascontiguousarray(np.vstack(uv_triangles), dtype=float)
                if uv_triangles
                else np.empty((0, 3, 2), dtype=float)
            ),
            uv_face_indices=(
                np.ascontiguousarray(
                    np.concatenate(uv_face_indices),
                    dtype=np.int64,
                )
                if uv_face_indices
                else np.empty((0,), dtype=np.int64)
            ),
        ),
    )


def _collect_instance_uv_triangles(
    geometry: trimesh.Trimesh,
    faces: np.ndarray,
    *,
    first_face_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite UV triangles with their canonical global face IDs."""

    try:
        uvs = np.asarray(getattr(geometry.visual, "uv", None), dtype=float)
    except (TypeError, ValueError):
        return (
            np.empty((0, 3, 2), dtype=float),
            np.empty((0,), dtype=np.int64),
        )
    if uvs.shape != (len(geometry.vertices), 2):
        return (
            np.empty((0, 3, 2), dtype=float),
            np.empty((0,), dtype=np.int64),
        )
    triangles = np.asarray(uvs[faces], dtype=float)
    valid_faces = np.all(np.isfinite(triangles), axis=(1, 2))
    local_indices = np.flatnonzero(valid_faces)
    return (
        np.ascontiguousarray(triangles[valid_faces], dtype=float),
        np.ascontiguousarray(
            local_indices + int(first_face_index),
            dtype=np.int64,
        ),
    )


def _validate_mesh_arrays(
    vertices: np.ndarray,
    faces: np.ndarray,
    transform: np.ndarray,
) -> None:
    if vertices.ndim != 2 or vertices.shape[1:] != (3,):
        raise ValueError("The object GLB contains invalid vertex coordinates.")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("The object GLB contains invalid vertex coordinates.")
    if faces.ndim != 2 or faces.shape[1:] != (3,):
        raise ValueError("The object GLB contains invalid triangle faces.")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("The object GLB contains invalid triangle indices.")
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("The object GLB contains an invalid node transform.")


# ### Selection validation ###
def _normalize_selected_face_indices(
    selected_face_indices: Iterable[int],
    *,
    face_count: int,
) -> frozenset[int]:
    if isinstance(selected_face_indices, str | bytes | bytearray):
        raise TypeError("Selected face indices must be an iterable of integers.")
    try:
        requested = tuple(selected_face_indices)
    except TypeError as error:
        raise TypeError(
            "Selected face indices must be an iterable of integers."
        ) from error
    if not requested:
        raise ValueError("Select at least one object face to delete.")
    normalized: set[int] = set()
    for face_index in requested:
        if isinstance(face_index, bool) or not isinstance(
            face_index,
            int | np.integer,
        ):
            raise TypeError("Selected face indices must be integers.")
        normalized_index = int(face_index)
        if not 0 <= normalized_index < face_count:
            raise ValueError(
                f"Selected face index {normalized_index} is outside the object."
            )
        normalized.add(normalized_index)
    return frozenset(normalized)


# ### Face filtering ###
def _filter_instances(
    instances: Sequence[_MeshInstance],
    keep_faces: np.ndarray,
) -> list[_MeshInstance]:
    filtered: list[_MeshInstance] = []
    for instance in instances:
        local_keep = np.asarray(
            keep_faces[
                instance.first_face_index : instance.first_face_index
                + instance.face_count
            ],
            dtype=bool,
        ).copy()
        if not np.any(local_keep):
            continue

        mesh = instance.mesh.copy()
        _filter_face_materials(mesh, local_keep)
        mesh.update_faces(local_keep)
        retained_vertex_indices = np.unique(
            np.asarray(mesh.faces, dtype=np.int64).reshape(-1)
        )
        mesh.remove_unreferenced_vertices()
        if len(mesh.vertices) != len(retained_vertex_indices):
            raise ValueError("Face deletion produced invalid retained vertices.")
        mesh.vertex_normals = instance.vertex_normals[
            retained_vertex_indices
        ].copy()
        filtered.append(
            _MeshInstance(
                node_name=instance.node_name,
                geometry_name=instance.geometry_name,
                mesh=mesh,
                transform=instance.transform.copy(),
                first_face_index=0,
                vertex_normals=np.asarray(mesh.vertex_normals, dtype=float).copy(),
            )
        )
    if not filtered:
        raise ValueError("Face deletion produced an empty object scene.")
    return filtered


def _filter_face_materials(mesh: trimesh.Trimesh, keep_faces: np.ndarray) -> None:
    """Subset TextureVisuals face-material ownership before face filtering."""

    raw_face_materials = getattr(mesh.visual, "face_materials", None)
    if raw_face_materials is None:
        return
    face_materials = np.asarray(raw_face_materials, dtype=np.int64)
    if face_materials.shape != (len(mesh.faces),):
        raise ValueError("The object GLB contains invalid face material indices.")
    mesh.visual.face_materials = face_materials[keep_faces].copy()


def _get_vertex_normals(mesh: trimesh.Trimesh) -> np.ndarray:
    """Preserve authored normals, or calculate a finite fallback."""

    try:
        source_normals = np.asarray(mesh.vertex_normals, dtype=float)
    except (AttributeError, TypeError, ValueError):
        source_normals = np.empty((0, 3), dtype=float)
    if (
        source_normals.shape == (len(mesh.vertices), 3)
        and np.all(np.isfinite(source_normals))
        and np.all(np.linalg.norm(source_normals, axis=1) > _NORMAL_EPSILON)
    ):
        return source_normals.copy()

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    face_vectors = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    normals = np.zeros_like(vertices, dtype=float)
    for corner_index in range(3):
        np.add.at(normals, faces[:, corner_index], face_vectors)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > _NORMAL_EPSILON
    normals[valid] /= lengths[valid, np.newaxis]
    normals[~valid] = (0.0, 0.0, 1.0)
    return normals


# ### Texture and UV inspection ###
def _instances_have_textured_uvs(instances: Sequence[_MeshInstance]) -> bool:
    """Return whether every retained mesh supports original-UV retexture."""

    if not instances:
        return False
    for instance in instances:
        mesh = instance.mesh
        uvs = np.asarray(getattr(mesh.visual, "uv", None), dtype=float)
        if (
            uvs.shape != (len(mesh.vertices), 2)
            or not np.all(np.isfinite(uvs))
            or not _material_supports_preserved_uvs(
                getattr(mesh.visual, "material", None)
            )
        ):
            return False
    return True


def _material_supports_preserved_uvs(material: object) -> bool:
    """Accept either an atlas texture or the untextured glass prefab."""

    if material is None:
        return False
    if is_housemaker_glass_material(material):
        return True
    nested_materials = getattr(material, "materials", None)
    if isinstance(nested_materials, list | tuple):
        return bool(nested_materials) and all(
            _material_supports_preserved_uvs(nested)
            for nested in nested_materials
        )
    return any(
        getattr(material, attribute_name, None) is not None
        for attribute_name in ("baseColorTexture", "image")
    )


# ### Scene export ###
def _export_instances(
    instances: Sequence[_MeshInstance],
    *,
    scene_metadata: dict[str, object],
) -> bytes:
    scene = trimesh.Scene(metadata=copy.deepcopy(scene_metadata))
    used_geometry_names: set[str] = set()
    used_node_names: set[str] = set()
    for instance_index, instance in enumerate(instances):
        geometry_name = _make_unique_name(
            instance.geometry_name,
            used_geometry_names,
            fallback=f"geometry_{instance_index}",
        )
        node_name = _make_unique_name(
            instance.node_name,
            used_node_names,
            fallback=f"node_{instance_index}",
        )
        scene.add_geometry(
            instance.mesh,
            geom_name=geometry_name,
            node_name=node_name,
            transform=instance.transform,
        )
    if not scene.geometry:
        raise ValueError("Face deletion produced an empty object scene.")
    try:
        return bytes(scene.export(file_type="glb"))
    except Exception as error:
        raise ValueError("The face-edited object GLB could not be exported.") from error


def _make_unique_name(
    requested_name: str,
    used_names: set[str],
    *,
    fallback: str,
) -> str:
    base_name = requested_name.strip() or fallback
    candidate = base_name
    suffix = 2
    while candidate in used_names:
        candidate = f"{base_name}_{suffix}"
        suffix += 1
    used_names.add(candidate)
    return candidate


# ### Export validation ###
def _validate_exported_face_count(
    glb_bytes: bytes,
    expected_face_count: int,
) -> None:
    scene = _load_glb_scene(glb_bytes)
    actual_face_count = sum(
        len(geometry.faces)
        for geometry in scene.geometry.values()
        if isinstance(geometry, trimesh.Trimesh)
    )
    if actual_face_count != expected_face_count:
        raise ValueError("Face editing changed an unexpected number of faces.")
