# ### Imports ###
from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO

import numpy as np
import shapely
import trimesh
from shapely import Polygon

from housemaker.glass_material import is_housemaker_glass_material
from housemaker.glb import (
    GLTF_Y_UP_TO_Z_UP_TRANSFORM,
    _serialize_scene_glb_with_half_mesh_extras,
)

# ### Constants ###
_NORMAL_EPSILON = 1e-12
_FACE_EDIT_ABSOLUTE_EPSILON = 1e-9
_FACE_EDIT_RELATIVE_EPSILON = 1e-7


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
        return len(self.faces)


@dataclass(frozen=True)
class ObjectFaceDeletionResult:
    """One topology-only face edit retaining existing UVs and materials."""

    glb_bytes: bytes
    original_face_count: int
    retained_face_count: int
    deleted_face_count: int
    preserved_textured_uvs: bool


@dataclass(frozen=True)
class ObjectFaceAdditionResult:
    """One added polygon represented by triangles in an existing primitive."""

    glb_bytes: bytes
    original_face_count: int
    result_face_count: int
    added_triangle_count: int
    added_face_indices: tuple[int, ...]
    selected_vertex_indices: tuple[int, ...]
    resolved_vertex_indices: tuple[int, ...]
    preserved_textured_uvs: bool


# ### Internal data models ###
@dataclass
class _MeshInstance:
    node_name: str
    geometry_name: str
    mesh: trimesh.Trimesh
    transform: np.ndarray
    first_vertex_index: int
    first_face_index: int
    vertex_normals: np.ndarray
    node_metadata: dict[str, object]

    @property
    def face_count(self) -> int:
        return len(self.mesh.faces)


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


def add_object_face_preserving_uvs(
    glb_bytes: bytes,
    selected_vertex_indices: Iterable[int],
) -> ObjectFaceAdditionResult:
    """Add one selected polygon without changing existing vertices or UVs."""

    result, _geometry = _add_object_face_preserving_uvs_with_geometry(
        glb_bytes,
        selected_vertex_indices,
        validate_export=True,
    )
    return result


def _add_object_face_preserving_uvs_with_geometry(
    glb_bytes: bytes,
    selected_vertex_indices: Iterable[int],
    *,
    validate_export: bool,
) -> tuple[ObjectFaceAdditionResult, ObjectFaceGeometry]:
    """Add one polygon while returning its parsed canonical source geometry."""

    scene = _load_glb_scene(glb_bytes)
    instances, geometry = _collect_scene_geometry(scene)
    if geometry.face_count == 0:
        raise ValueError("The object GLB contains no triangle faces.")
    selected = _normalize_selected_vertex_indices(
        selected_vertex_indices,
        vertex_count=len(geometry.vertices),
    )
    (
        target,
        local_vertex_indices,
        resolved_vertex_indices,
    ) = _resolve_selected_mesh_instance(
        instances,
        geometry.vertices,
        selected,
    )
    edited_target, added_local_faces = _add_polygon_to_instance(
        target,
        local_vertex_indices,
    )
    edited_instances = [
        edited_target if instance is target else instance for instance in instances
    ]
    added_face_indices = tuple(
        range(
            target.first_face_index + target.face_count,
            target.first_face_index + target.face_count + len(added_local_faces),
        )
    )
    result_face_count = geometry.face_count + len(added_local_faces)
    edited_glb = _export_instances(
        edited_instances,
        scene_metadata=scene.metadata,
    )
    if validate_export:
        _validate_exported_face_count(edited_glb, result_face_count)
    return (
        ObjectFaceAdditionResult(
            glb_bytes=edited_glb,
            original_face_count=geometry.face_count,
            result_face_count=result_face_count,
            added_triangle_count=len(added_local_faces),
            added_face_indices=added_face_indices,
            selected_vertex_indices=selected,
            resolved_vertex_indices=resolved_vertex_indices,
            preserved_textured_uvs=(_instances_have_textured_uvs(edited_instances)),
        ),
        geometry,
    )


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
    retained_face_count = sum(instance.face_count for instance in filtered_instances)
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
            preserved_textured_uvs=(_instances_have_textured_uvs(filtered_instances)),
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
        local_uv_triangles, local_uv_face_indices = _collect_instance_uv_triangles(
            geometry,
            local_faces,
            first_face_index=face_offset,
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
                    first_vertex_index=vertex_offset,
                    first_face_index=face_offset,
                    vertex_normals=_get_vertex_normals(geometry),
                    node_metadata=_get_scene_node_metadata(scene, node_name),
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


def _normalize_selected_vertex_indices(
    selected_vertex_indices: Iterable[int],
    *,
    vertex_count: int,
) -> tuple[int, ...]:
    """Validate canonical vertex IDs while retaining their selection order."""

    if isinstance(selected_vertex_indices, str | bytes | bytearray):
        raise TypeError("Selected vertex indices must be an iterable of integers.")
    try:
        requested = tuple(selected_vertex_indices)
    except TypeError as error:
        raise TypeError(
            "Selected vertex indices must be an iterable of integers."
        ) from error
    if len(requested) < 3:
        raise ValueError("Select at least three object vertices to add a face.")
    normalized: list[int] = []
    for vertex_index in requested:
        if isinstance(vertex_index, bool) or not isinstance(
            vertex_index,
            int | np.integer,
        ):
            raise TypeError("Selected vertex indices must be integers.")
        normalized_index = int(vertex_index)
        if not 0 <= normalized_index < vertex_count:
            raise ValueError(
                f"Selected vertex index {normalized_index} is outside the object."
            )
        normalized.append(normalized_index)
    if len(set(normalized)) != len(normalized):
        raise ValueError("Face creation requires unique selected vertices.")
    return tuple(normalized)


def _resolve_selected_mesh_instance(
    instances: Sequence[_MeshInstance],
    world_vertices: np.ndarray,
    selected_vertex_indices: Sequence[int],
) -> tuple[_MeshInstance, tuple[int, ...], tuple[int, ...]]:
    """Resolve visible positions to deterministic IDs in one mesh primitive."""

    normalized_world_vertices = np.asarray(world_vertices, dtype=float)
    selected_points = normalized_world_vertices[list(selected_vertex_indices)]
    tolerance = _get_face_edit_tolerance(selected_points)
    groups, group_points = _cluster_selected_vertex_positions(
        selected_vertex_indices,
        normalized_world_vertices,
        tolerance=tolerance,
    )
    if len(groups) < 3:
        raise ValueError(
            "Select at least three geometrically distinct visible vertices; "
            "coincident UV-seam vertices count as one vertex."
        )

    resolutions: list[tuple[_MeshInstance, tuple[int, ...], tuple[int, ...]]] = []
    selected_set = set(selected_vertex_indices)
    for instance in instances:
        first_vertex = instance.first_vertex_index
        vertex_count = len(instance.mesh.vertices)
        instance_world_vertices = normalized_world_vertices[
            first_vertex : first_vertex + vertex_count
        ]
        candidate_groups: list[tuple[int, ...]] = []
        for point in group_points:
            distances = np.linalg.norm(instance_world_vertices - point, axis=1)
            candidates = tuple(
                int(index) for index in np.flatnonzero(distances <= tolerance)
            )
            if not candidates:
                break
            candidate_groups.append(candidates)
        if len(candidate_groups) != len(groups):
            continue

        ordered_groups = _order_geometric_vertex_groups(
            group_points,
            candidate_groups,
            np.asarray(instance.mesh.faces, dtype=np.int64),
            tolerance=tolerance,
        )
        ordered_candidate_groups = tuple(
            candidate_groups[group_index] for group_index in ordered_groups
        )
        explicitly_selected_local = {
            vertex_index - first_vertex
            for vertex_index in selected_set
            if first_vertex <= vertex_index < first_vertex + vertex_count
        }
        explicit_group_count = sum(
            bool(set(candidates).intersection(explicitly_selected_local))
            for candidates in candidate_groups
        )
        if explicit_group_count != len(groups):
            continue
        local_indices = _choose_geometric_vertex_representatives(
            ordered_candidate_groups,
            np.asarray(instance.mesh.faces, dtype=np.int64),
            explicitly_selected_local,
        )
        resolved_global = tuple(
            first_vertex + vertex_index for vertex_index in local_indices
        )
        resolutions.append((instance, local_indices, resolved_global))

    if not resolutions:
        raise ValueError(
            "The selected visible vertices do not all belong to one mesh "
            "primitive. Select vertices from one connected object part."
        )
    if len(resolutions) > 1:
        raise ValueError(
            "The selected visible vertices ambiguously match multiple mesh "
            "primitives. Select vertices from only one object part."
        )
    target, local_indices, resolved_global = resolutions[0]
    return target, local_indices, resolved_global


def _cluster_selected_vertex_positions(
    selected_vertex_indices: Sequence[int],
    world_vertices: np.ndarray,
    *,
    tolerance: float,
) -> tuple[tuple[tuple[int, ...], ...], np.ndarray]:
    """Collapse canonical UV-seam IDs into logical visible positions."""

    groups: list[list[int]] = []
    representative_points: list[np.ndarray] = []
    for vertex_index in selected_vertex_indices:
        point = np.asarray(world_vertices[int(vertex_index)], dtype=float)
        matching_group = next(
            (
                group_index
                for group_index, representative in enumerate(representative_points)
                if float(np.linalg.norm(point - representative)) <= tolerance
            ),
            None,
        )
        if matching_group is None:
            groups.append([int(vertex_index)])
            representative_points.append(point.copy())
        else:
            groups[matching_group].append(int(vertex_index))
    return (
        tuple(tuple(group) for group in groups),
        np.ascontiguousarray(representative_points, dtype=float),
    )


def _order_geometric_vertex_groups(
    group_points: np.ndarray,
    candidate_groups: Sequence[Sequence[int]],
    source_faces: np.ndarray,
    *,
    tolerance: float,
) -> tuple[int, ...]:
    """Order logical vertices from source edges, then their planar perimeter."""

    group_by_vertex = {
        int(vertex_index): group_index
        for group_index, candidates in enumerate(candidate_groups)
        for vertex_index in candidates
    }
    adjacency = {group_index: set() for group_index in range(len(group_points))}
    for face in source_faces:
        for start, end in (
            (int(face[0]), int(face[1])),
            (int(face[1]), int(face[2])),
            (int(face[2]), int(face[0])),
        ):
            start_group = group_by_vertex.get(start)
            end_group = group_by_vertex.get(end)
            if (
                start_group is not None
                and end_group is not None
                and start_group != end_group
            ):
                adjacency[start_group].add(end_group)
                adjacency[end_group].add(start_group)
    edge_order = _traverse_simple_group_perimeter(adjacency, group_points)
    if edge_order is not None:
        return edge_order

    plane_normal = _validate_selected_point_plane(
        group_points,
        tolerance=tolerance,
    )
    dominant_axis = int(np.argmax(np.abs(plane_normal)))
    if plane_normal[dominant_axis] < 0.0:
        plane_normal *= -1.0
    reference_axis = np.zeros(3, dtype=float)
    reference_axis[int(np.argmin(np.abs(plane_normal)))] = 1.0
    basis_u = np.cross(reference_axis, plane_normal)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(plane_normal, basis_u)
    centered = group_points - np.mean(group_points, axis=0)
    coordinates = np.column_stack((centered @ basis_u, centered @ basis_v))
    angles = np.arctan2(coordinates[:, 1], coordinates[:, 0])
    distances = np.linalg.norm(coordinates, axis=1)
    order = np.lexsort(
        (
            np.arange(len(group_points), dtype=np.int64),
            distances,
            angles,
        )
    )
    return tuple(int(index) for index in order)


def _traverse_simple_group_perimeter(
    adjacency: Mapping[int, set[int]],
    group_points: np.ndarray,
) -> tuple[int, ...] | None:
    """Traverse one simple group-edge chain or cycle deterministically."""

    degrees = {group: len(neighbors) for group, neighbors in adjacency.items()}
    endpoints = [group for group, degree in degrees.items() if degree == 1]
    is_chain = (
        len(endpoints) == 2
        and all(degree in {1, 2} for degree in degrees.values())
        and sum(degrees.values()) == 2 * (len(adjacency) - 1)
    )
    is_cycle = (
        not endpoints
        and all(degree == 2 for degree in degrees.values())
        and sum(degrees.values()) == 2 * len(adjacency)
    )
    if not is_chain and not is_cycle:
        return None

    point_keys = {
        group: tuple(float(value) for value in np.round(group_points[group], 12))
        for group in adjacency
    }
    start = min(endpoints if is_chain else adjacency, key=point_keys.__getitem__)
    candidates: list[tuple[int, ...]] = []
    for first_neighbor in sorted(adjacency[start], key=point_keys.__getitem__):
        path = [start, first_neighbor]
        previous = start
        current = first_neighbor
        while len(path) < len(adjacency):
            available = adjacency[current] - {previous} - set(path)
            if len(available) != 1:
                break
            following = next(iter(available))
            previous, current = current, following
            path.append(current)
        if len(path) != len(adjacency):
            continue
        if is_cycle and start not in adjacency[path[-1]]:
            continue
        candidates.append(tuple(path))
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda path: tuple(point_keys[group] for group in path),
    )


def _choose_geometric_vertex_representatives(
    candidate_groups: Sequence[Sequence[int]],
    source_faces: np.ndarray,
    explicitly_selected: set[int],
) -> tuple[int, ...]:
    """Choose seam representatives that retain the most boundary edges."""

    source_edges = _collect_undirected_face_edges(source_faces)
    chosen: list[int] = []
    group_count = len(candidate_groups)
    for group_index, candidates in enumerate(candidate_groups):
        previous_candidates = candidate_groups[(group_index - 1) % group_count]
        following_candidates = candidate_groups[(group_index + 1) % group_count]

        def candidate_score(
            vertex_index: int,
            previous_candidates: Sequence[int] = previous_candidates,
            following_candidates: Sequence[int] = following_candidates,
        ) -> tuple[int, int, int, int]:
            previous_connections = sum(
                tuple(sorted((int(vertex_index), int(other)))) in source_edges
                for other in previous_candidates
            )
            following_connections = sum(
                tuple(sorted((int(vertex_index), int(other)))) in source_edges
                for other in following_candidates
            )
            return (
                int(previous_connections > 0) + int(following_connections > 0),
                previous_connections + following_connections,
                int(vertex_index in explicitly_selected),
                -int(vertex_index),
            )

        chosen.append(max((int(index) for index in candidates), key=candidate_score))
    return tuple(chosen)


def _collect_undirected_face_edges(
    faces: np.ndarray,
) -> set[tuple[int, int]]:
    return {
        tuple(sorted((int(start), int(end))))
        for face in faces
        for start, end in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        )
    }


# ### Scene metadata ###
def _get_scene_node_metadata(
    scene: trimesh.Scene,
    node_name: object,
) -> dict[str, object]:
    """Copy metadata attached to the graph edge that owns one mesh node."""

    parent_name = scene.graph.transforms.parents.get(node_name)
    if parent_name is None:
        return {}
    edge_data = scene.graph.transforms.edge_data.get(
        (parent_name, node_name),
        {},
    )
    raw_metadata = edge_data.get("metadata")
    if not isinstance(raw_metadata, Mapping):
        return {}
    return copy.deepcopy(dict(raw_metadata))


# ### Face addition ###
def _add_polygon_to_instance(
    instance: _MeshInstance,
    local_vertex_indices: Sequence[int],
) -> tuple[_MeshInstance, np.ndarray]:
    """Validate, triangulate, and append one polygon to a mesh instance."""

    source_mesh = instance.mesh
    vertices = np.asarray(source_mesh.vertices, dtype=float)
    source_faces = np.asarray(source_mesh.faces, dtype=np.int64)
    selected_indices = tuple(int(index) for index in local_vertex_indices)
    selected_points = np.asarray(vertices[list(selected_indices)], dtype=float)
    tolerance = _get_face_edit_tolerance(selected_points)
    ordered_indices = _order_selected_polygon_vertices(
        selected_indices,
        selected_points,
        source_faces,
        tolerance=tolerance,
    )
    ordered_points = np.asarray(vertices[list(ordered_indices)], dtype=float)
    normal, coordinates, polygon = _prepare_selected_polygon(
        ordered_points,
        tolerance=tolerance,
    )
    _reject_existing_coplanar_overlap(
        polygon,
        vertices,
        source_faces,
        plane_origin=ordered_points[0],
        plane_normal=normal,
        selected_vertex_indices=ordered_indices,
        tolerance=tolerance,
    )
    _validate_polygon_boundary_manifold(source_faces, ordered_indices)
    oriented_indices, oriented_points, oriented_coordinates, normal = (
        _orient_polygon_from_adjacent_faces(
            ordered_indices,
            ordered_points,
            coordinates,
            normal,
            source_faces,
        )
    )
    added_faces = _triangulate_selected_polygon(
        oriented_indices,
        oriented_points,
        oriented_coordinates,
        normal,
        tolerance=tolerance,
    )
    _validate_manifold_edge_counts(source_faces, added_faces)

    mesh = source_mesh.copy()
    original_normals = instance.vertex_normals.copy()
    original_face_materials = _get_face_materials(mesh)
    material_index = _infer_added_face_material(
        source_faces,
        oriented_indices,
        original_face_materials,
    )
    mesh.faces = np.ascontiguousarray(
        np.vstack((source_faces, added_faces)),
        dtype=np.int64,
    )
    if original_face_materials is not None:
        mesh.visual.face_materials = np.concatenate(
            (
                original_face_materials,
                np.full(len(added_faces), material_index, dtype=np.int64),
            )
        )
    mesh.vertex_normals = original_normals
    return (
        _MeshInstance(
            node_name=instance.node_name,
            geometry_name=instance.geometry_name,
            mesh=mesh,
            transform=instance.transform.copy(),
            first_vertex_index=instance.first_vertex_index,
            first_face_index=instance.first_face_index,
            vertex_normals=original_normals,
            node_metadata=copy.deepcopy(instance.node_metadata),
        ),
        added_faces,
    )


def _get_face_edit_tolerance(points: np.ndarray) -> float:
    span = float(np.linalg.norm(np.ptp(points, axis=0)))
    return max(
        _FACE_EDIT_ABSOLUTE_EPSILON,
        span * _FACE_EDIT_RELATIVE_EPSILON,
    )


def _order_selected_polygon_vertices(
    selected_indices: tuple[int, ...],
    selected_points: np.ndarray,
    source_faces: np.ndarray,
    *,
    tolerance: float,
) -> tuple[int, ...]:
    """Resolve unordered selection IDs into one deterministic perimeter."""

    plane_normal = _validate_selected_point_plane(
        selected_points,
        tolerance=tolerance,
    )
    edge_order = _order_vertices_from_existing_edges(
        selected_indices,
        source_faces,
    )
    if edge_order is not None:
        return edge_order

    dominant_axis = int(np.argmax(np.abs(plane_normal)))
    plane_normal = plane_normal.copy()
    if plane_normal[dominant_axis] < 0.0:
        plane_normal *= -1.0
    reference_axis = np.zeros(3, dtype=float)
    reference_axis[int(np.argmin(np.abs(plane_normal)))] = 1.0
    basis_u = np.cross(reference_axis, plane_normal)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(plane_normal, basis_u)
    centered = selected_points - np.mean(selected_points, axis=0)
    coordinates = np.column_stack((centered @ basis_u, centered @ basis_v))
    angles = np.arctan2(coordinates[:, 1], coordinates[:, 0])
    distances = np.linalg.norm(coordinates, axis=1)
    order = np.lexsort(
        (
            np.asarray(selected_indices, dtype=np.int64),
            distances,
            angles,
        )
    )
    return tuple(selected_indices[int(position)] for position in order)


def _validate_selected_point_plane(
    points: np.ndarray,
    *,
    tolerance: float,
) -> np.ndarray:
    """Validate unordered points and return their stable best-fit normal."""

    pairwise_distances = np.linalg.norm(
        points[:, np.newaxis, :] - points[np.newaxis, :, :],
        axis=2,
    )
    pairwise_distances += np.eye(len(points)) * (tolerance * 2.0)
    if np.any(pairwise_distances <= tolerance):
        raise ValueError("Face creation vertices must have distinct positions.")

    centered = points - np.mean(points, axis=0)
    try:
        _u, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError as error:
        raise ValueError("The selected face plane could not be resolved.") from error
    if len(singular_values) < 2 or singular_values[1] <= tolerance:
        raise ValueError("Face creation vertices cannot be collinear.")
    plane_normal = np.asarray(vh[-1], dtype=float)
    plane_distances = np.abs(centered @ plane_normal)
    if np.any(plane_distances > tolerance):
        raise ValueError("Face creation vertices must be coplanar.")
    return plane_normal / np.linalg.norm(plane_normal)


def _order_vertices_from_existing_edges(
    selected_indices: tuple[int, ...],
    source_faces: np.ndarray,
) -> tuple[int, ...] | None:
    """Traverse a selected source-edge cycle or chain when one exists."""

    selected = set(selected_indices)
    adjacency = {index: set() for index in selected_indices}
    for face in source_faces:
        for start, end in (
            (int(face[0]), int(face[1])),
            (int(face[1]), int(face[2])),
            (int(face[2]), int(face[0])),
        ):
            if start in selected and end in selected:
                adjacency[start].add(end)
                adjacency[end].add(start)
    degrees = {index: len(neighbors) for index, neighbors in adjacency.items()}
    endpoints = sorted(index for index, degree in degrees.items() if degree == 1)
    is_chain = (
        len(endpoints) == 2
        and all(degree in {1, 2} for degree in degrees.values())
        and sum(degrees.values()) == 2 * (len(selected_indices) - 1)
    )
    is_cycle = (
        not endpoints
        and all(degree == 2 for degree in degrees.values())
        and sum(degrees.values()) == 2 * len(selected_indices)
    )
    if not is_chain and not is_cycle:
        return None

    start = endpoints[0] if is_chain else min(selected_indices)
    first_neighbors = sorted(adjacency[start])
    candidates: list[tuple[int, ...]] = []
    for first_neighbor in first_neighbors:
        path = [start, first_neighbor]
        previous = start
        current = first_neighbor
        while len(path) < len(selected_indices):
            available = sorted(adjacency[current] - {previous} - set(path))
            if len(available) != 1:
                break
            previous, current = current, available[0]
            path.append(current)
        if len(path) != len(selected_indices):
            continue
        if is_cycle and start not in adjacency[path[-1]]:
            continue
        candidates.append(tuple(path))
    return min(candidates) if candidates else None


def _prepare_selected_polygon(
    points: np.ndarray,
    *,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, Polygon]:
    """Return a stable plane and valid two-dimensional polygon."""

    plane_normal = _validate_selected_point_plane(points, tolerance=tolerance)

    newell_normal = np.sum(
        np.cross(points, np.roll(points, -1, axis=0)),
        axis=0,
    )
    newell_length = float(np.linalg.norm(newell_normal))
    if newell_length <= tolerance * tolerance:
        raise ValueError("Face creation vertices form a degenerate polygon.")
    if float(np.dot(plane_normal, newell_normal)) < 0.0:
        plane_normal *= -1.0
    plane_normal /= np.linalg.norm(plane_normal)

    basis_u = points[1] - points[0]
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(plane_normal, basis_u)
    basis_v /= np.linalg.norm(basis_v)
    coordinates = np.column_stack(
        (
            (points - points[0]) @ basis_u,
            (points - points[0]) @ basis_v,
        )
    )
    polygon = Polygon(coordinates)
    if (
        not polygon.is_valid
        or not polygon.exterior.is_simple
        or polygon.area <= tolerance * tolerance
    ):
        raise ValueError(
            "Face creation vertices must form a simple non-self-intersecting polygon."
        )
    return plane_normal, coordinates, polygon


def _reject_existing_coplanar_overlap(
    candidate: Polygon,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    selected_vertex_indices: Sequence[int],
    tolerance: float,
) -> None:
    """Reject duplicate faces and any positive-area coplanar overlap."""

    selected_points = vertices[list(selected_vertex_indices)]
    basis_u = selected_points[1] - selected_points[0]
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(plane_normal, basis_u)
    basis_v /= np.linalg.norm(basis_v)
    area_tolerance = max(tolerance * tolerance, candidate.area * 1e-10)
    for face in faces:
        triangle = vertices[face]
        if np.any(np.abs((triangle - plane_origin) @ plane_normal) > tolerance):
            continue
        triangle_coordinates = np.column_stack(
            (
                (triangle - plane_origin) @ basis_u,
                (triangle - plane_origin) @ basis_v,
            )
        )
        triangle_polygon = Polygon(triangle_coordinates)
        if not triangle_polygon.is_valid or triangle_polygon.area <= area_tolerance:
            continue
        if candidate.intersection(triangle_polygon).area > area_tolerance:
            raise ValueError("The selected polygon overlaps an existing object face.")


def _orient_polygon_from_adjacent_faces(
    indices: tuple[int, ...],
    points: np.ndarray,
    coordinates: np.ndarray,
    normal: np.ndarray,
    faces: np.ndarray,
) -> tuple[tuple[int, ...], np.ndarray, np.ndarray, np.ndarray]:
    """Make new boundary edges oppose the winding of neighboring faces."""

    flip_requirements: set[bool] = set()
    for edge_start, edge_end in zip(indices, indices[1:] + indices[:1]):
        for face in faces:
            for corner_index in range(3):
                current = int(face[corner_index])
                following = int(face[(corner_index + 1) % 3])
                if current == edge_start and following == edge_end:
                    flip_requirements.add(True)
                elif current == edge_end and following == edge_start:
                    flip_requirements.add(False)
    if len(flip_requirements) > 1:
        raise ValueError(
            "Adjacent object faces have inconsistent boundary orientation."
        )
    if flip_requirements == {True}:
        return (
            tuple(reversed(indices)),
            points[::-1].copy(),
            coordinates[::-1].copy(),
            -normal,
        )
    return indices, points, coordinates, normal


def _triangulate_selected_polygon(
    indices: Sequence[int],
    points: np.ndarray,
    coordinates: np.ndarray,
    normal: np.ndarray,
    *,
    tolerance: float,
) -> np.ndarray:
    """Triangulate a simple polygon using only its selected vertices."""

    polygon = Polygon(coordinates)
    triangulation = shapely.constrained_delaunay_triangles(polygon)
    triangle_parts = tuple(shapely.get_parts(triangulation))
    if not triangle_parts:
        raise ValueError("The selected object face could not be triangulated.")
    covered = shapely.union_all(triangle_parts)
    area_tolerance = max(tolerance * tolerance, polygon.area * 1e-9)
    if (
        covered.symmetric_difference(polygon).area > area_tolerance
        or abs(sum(part.area for part in triangle_parts) - polygon.area)
        > area_tolerance
    ):
        raise ValueError("The selected object face could not be triangulated safely.")

    faces: list[tuple[int, int, int]] = []
    coordinate_tolerance = max(tolerance, np.sqrt(area_tolerance) * 1e-3)
    for triangle in triangle_parts:
        triangle_coordinates = np.asarray(triangle.exterior.coords, dtype=float)[:3]
        local_positions: list[int] = []
        for coordinate in triangle_coordinates:
            distances = np.linalg.norm(coordinates - coordinate, axis=1)
            position = int(np.argmin(distances))
            if float(distances[position]) > coordinate_tolerance:
                raise ValueError("Face triangulation introduced a non-selected vertex.")
            local_positions.append(position)
        if len(set(local_positions)) != 3:
            raise ValueError("Face triangulation produced a degenerate triangle.")
        face = [int(indices[position]) for position in local_positions]
        triangle_points = points[local_positions]
        triangle_normal = np.cross(
            triangle_points[1] - triangle_points[0],
            triangle_points[2] - triangle_points[0],
        )
        if float(np.dot(triangle_normal, normal)) < 0.0:
            face[1], face[2] = face[2], face[1]
        faces.append(tuple(face))
    faces.sort(key=lambda face: tuple(sorted(face)))
    return np.ascontiguousarray(faces, dtype=np.int64)


def _validate_manifold_edge_counts(
    source_faces: np.ndarray,
    added_faces: np.ndarray,
) -> None:
    """Ensure no source or newly appended edge gains over two incident faces."""

    source_edge_counts: dict[tuple[int, int], int] = {}
    for face in source_faces:
        for start, end in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            edge = tuple(sorted((int(start), int(end))))
            source_edge_counts[edge] = source_edge_counts.get(edge, 0) + 1
    added_edge_counts: dict[tuple[int, int], int] = {}
    for face in added_faces:
        for start, end in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            edge = tuple(sorted((int(start), int(end))))
            added_edge_counts[edge] = added_edge_counts.get(edge, 0) + 1
    if any(
        source_edge_counts.get(edge, 0) + added_count > 2
        for edge, added_count in added_edge_counts.items()
    ):
        raise ValueError("Face creation would produce a non-manifold object edge.")


def _validate_polygon_boundary_manifold(
    source_faces: np.ndarray,
    polygon_indices: Sequence[int],
) -> None:
    """Reject a polygon boundary already owned by two source triangles."""

    source_edge_counts: dict[tuple[int, int], int] = {}
    for face in source_faces:
        for start, end in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            edge = tuple(sorted((int(start), int(end))))
            source_edge_counts[edge] = source_edge_counts.get(edge, 0) + 1
    for start, end in zip(
        polygon_indices,
        tuple(polygon_indices[1:]) + tuple(polygon_indices[:1]),
    ):
        edge = tuple(sorted((int(start), int(end))))
        if source_edge_counts.get(edge, 0) >= 2:
            raise ValueError("Face creation would produce a non-manifold object edge.")


def _get_face_materials(mesh: trimesh.Trimesh) -> np.ndarray | None:
    raw_face_materials = getattr(mesh.visual, "face_materials", None)
    if raw_face_materials is None:
        return None
    face_materials = np.asarray(raw_face_materials, dtype=np.int64)
    if face_materials.shape != (len(mesh.faces),):
        raise ValueError("The object GLB contains invalid face material indices.")
    return face_materials.copy()


def _infer_added_face_material(
    source_faces: np.ndarray,
    polygon_indices: Sequence[int],
    face_materials: np.ndarray | None,
) -> int:
    """Use the deterministic majority material of adjacent boundary faces."""

    if face_materials is None or not len(face_materials):
        return 0
    adjacent_face_indices: set[int] = set()
    for start, end in zip(
        polygon_indices,
        tuple(polygon_indices[1:]) + tuple(polygon_indices[:1]),
    ):
        matches = np.flatnonzero(
            np.logical_and(
                np.any(source_faces == int(start), axis=1),
                np.any(source_faces == int(end), axis=1),
            )
        )
        adjacent_face_indices.update(int(index) for index in matches)
    if not adjacent_face_indices:
        selected = {int(index) for index in polygon_indices}
        adjacent_face_indices.update(
            face_index
            for face_index, face in enumerate(source_faces)
            if selected.intersection(int(index) for index in face)
        )
    if not adjacent_face_indices:
        return 0
    materials, counts = np.unique(
        face_materials[sorted(adjacent_face_indices)],
        return_counts=True,
    )
    maximum_count = int(np.max(counts))
    return int(np.min(materials[counts == maximum_count]))


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
        mesh.vertex_normals = instance.vertex_normals[retained_vertex_indices].copy()
        filtered.append(
            _MeshInstance(
                node_name=instance.node_name,
                geometry_name=instance.geometry_name,
                mesh=mesh,
                transform=instance.transform.copy(),
                first_vertex_index=0,
                first_face_index=0,
                vertex_normals=np.asarray(mesh.vertex_normals, dtype=float).copy(),
                node_metadata=copy.deepcopy(instance.node_metadata),
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
            _material_supports_preserved_uvs(nested) for nested in nested_materials
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
            metadata=copy.deepcopy(instance.node_metadata),
        )
    if not scene.geometry:
        raise ValueError("Face editing produced an empty object scene.")
    try:
        return _serialize_scene_glb_with_half_mesh_extras(
            scene,
            failure_message="The face-edited object GLB could not be exported.",
        )
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
