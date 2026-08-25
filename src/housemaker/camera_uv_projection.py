# ### Imports ###
from __future__ import annotations

import copy
import heapq
import math
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from io import BytesIO
from itertools import product

import numpy as np
import trimesh
from shapely import STRtree
from shapely.geometry import Polygon
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.glb import (
    GLTF_Y_UP_TO_Z_UP_TRANSFORM,
    GeneratedModel,
    import_generated_glb,
)
from housemaker.unused_face_removal import (
    ALL_CAMERA_IDS,
    UnusedFaceRemovalCancelled,
    capture_visible_face_indices,
    get_fixed_camera_view,
)


# ### Projection constants ###
CAMERA_UV_TEXTURE_SIZES = (512, 1024, 2048)
DEFAULT_CAMERA_UV_BASE_TEXTURE_SIZE = 512
DEFAULT_CAMERA_UV_CAPTURE_SIZE = 512
DEFAULT_CAMERA_UV_PADDING_PIXELS = 2
DEFAULT_CAMERA_UV_MAX_FACE_COUNT = 200_000
DEFAULT_CAMERA_UV_PROGRESS_INTERVAL_FACES = 128
MINIMUM_PROJECTED_EXTENT = 1e-12
PACKING_SEARCH_PASSES = 24
INTERSECTION_AREA_EPSILON_RATIO = 1e-12
TOPOLOGY_WELD_EDGE_RATIO = 1e-7
TOPOLOGY_WELD_ULP_MULTIPLIER = 8.0
MINIMUM_TOPOLOGY_WELD_DISTANCE = 1e-12
TOPOLOGY_NORMAL_CONNECTION_COSINE = math.cos(math.radians(30.0))
TOPOLOGY_NORMAL_WEIGHT_FLOOR = 0.02
TOPOLOGY_NORMAL_WEIGHT_EXPONENT = 4
OUTPUT_NORMAL_WELD_TOLERANCE = float(np.finfo(np.float32).eps * 8.0)
OUTPUT_UV_WELD_TOLERANCE = 1e-9
FALLBACK_CHART_CAMERA_ID = "fallback"
FALLBACK_REGION_SIDE_PIXELS = 128
FALLBACK_CELL_MARGIN_RATIO = 0.1
MINIMUM_ACCEPTED_PROJECTED_QUALITY = 0.6
FALLBACK_REASON_INVISIBLE = "invisible"
FALLBACK_REASON_QUALITY = "quality"
FALLBACK_REASON_CONFLICT = "conflict"
GLOBAL_CAMERA_ASSIGNMENT_SEEDS = (0, 2, 15, 44)
EXACT_CAMERA_ASSIGNMENT_COMPONENT_LIMIT = 20
EXACT_CAMERA_ASSIGNMENT_MAX_SEARCH_NODES = 50_000
CAMERA_ASSIGNMENT_SMOOTHING_PASSES = 12
CAMERA_ASSIGNMENT_REGION_SMOOTHING_PASSES = 6


# ### Callback types ###
CancelCallback = Callable[[], bool]
ProgressCallback = Callable[["CameraUvProjectionProgress"], None]


# ### Public data models ###
@dataclass(frozen=True)
class CameraUvProjectionOptions:
    """Limits and pixel-grid rules for one six-view UV projection."""

    base_texture_size: int = DEFAULT_CAMERA_UV_BASE_TEXTURE_SIZE
    capture_image_size: int = DEFAULT_CAMERA_UV_CAPTURE_SIZE
    padding_pixels: int = DEFAULT_CAMERA_UV_PADDING_PIXELS
    max_face_count: int = DEFAULT_CAMERA_UV_MAX_FACE_COUNT
    progress_interval_faces: int = DEFAULT_CAMERA_UV_PROGRESS_INTERVAL_FACES

    def __post_init__(self) -> None:
        values = {
            "base texture size": self.base_texture_size,
            "capture size": self.capture_image_size,
            "padding": self.padding_pixels,
            "maximum face count": self.max_face_count,
            "progress interval": self.progress_interval_faces,
        }
        for label, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"Camera UV {label} must be an integer.")
        if self.base_texture_size != DEFAULT_CAMERA_UV_BASE_TEXTURE_SIZE:
            raise ValueError("Camera UV projection uses a fixed 512-pixel grid.")
        if not 32 <= self.capture_image_size <= 1024:
            raise ValueError(
                "Camera UV capture size must be between 32 and 1024 pixels."
            )
        if not 1 <= self.padding_pixels <= 32:
            raise ValueError("Camera UV padding must be between 1 and 32 pixels.")
        if not 1 <= self.max_face_count <= DEFAULT_CAMERA_UV_MAX_FACE_COUNT:
            raise ValueError(
                "Camera UV maximum face count must be between 1 and "
                f"{DEFAULT_CAMERA_UV_MAX_FACE_COUNT}."
            )
        if self.progress_interval_faces < 1:
            raise ValueError("Camera UV progress interval must be positive.")


@dataclass(frozen=True)
class CameraUvProjectionProgress:
    """One background-safe status update from camera UV processing."""

    stage: str
    completed_face_count: int
    total_face_count: int
    camera_id: str | None = None


@dataclass(frozen=True)
class CameraUvChartRegion:
    """One packed chart rectangle expressed on the 512-pixel grid."""

    chart_index: int
    camera_id: str
    is_leftover: bool
    x: int
    y: int
    width: int
    height: int
    face_count: int


@dataclass(frozen=True)
class CameraUvProjectionResult:
    """UV-authored model plus auditable camera and packing metadata."""

    model: GeneratedModel
    camera_face_counts: dict[str, int]
    leftover_face_count: int
    invisible_face_count: int
    quality_fallback_face_count: int
    conflict_fallback_face_count: int
    original_face_count: int
    original_vertex_count: int
    output_vertex_count: int
    compatible_texture_sizes: tuple[int, ...]
    chart_regions: tuple[CameraUvChartRegion, ...]

    @property
    def glb_bytes(self) -> bytes:
        return self.model.glb_bytes

    @property
    def projected_face_count(self) -> int:
        return sum(self.camera_face_counts.values())


class CameraUvProjectionCancelled(RuntimeError):
    """Raised when the caller cancels camera UV processing."""


# ### Internal data models ###
@dataclass(frozen=True)
class _MeshInstance:
    node_name: str
    geometry_name: str
    mesh: trimesh.Trimesh
    transform: np.ndarray
    world_vertices: np.ndarray
    vertex_normals: np.ndarray
    first_face_index: int

    @property
    def face_count(self) -> int:
        return len(self.mesh.faces)


@dataclass(frozen=True)
class _ProjectedFace:
    global_face_index: int
    instance_index: int
    local_face_index: int
    camera_id: str
    coordinates: np.ndarray
    is_leftover: bool
    fallback_reason: str | None = None
    force_separate_chart: bool = False


@dataclass(frozen=True)
class _UvChart:
    chart_index: int
    camera_id: str
    is_leftover: bool
    faces: tuple[_ProjectedFace, ...]
    minimum: np.ndarray
    maximum: np.ndarray

    @property
    def width(self) -> float:
        return max(float(self.maximum[0] - self.minimum[0]), MINIMUM_PROJECTED_EXTENT)

    @property
    def height(self) -> float:
        return max(float(self.maximum[1] - self.minimum[1]), MINIMUM_PROJECTED_EXTENT)


@dataclass(frozen=True)
class _PixelRectangle:
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def top(self) -> int:
        return self.y + self.height


@dataclass(frozen=True)
class _PackedLayout:
    scale: float
    placements: dict[int, _PixelRectangle]


@dataclass(frozen=True)
class _FaceNeighbor:
    face_index: int
    seam_weight: float


@dataclass(frozen=True)
class _MeshTopology:
    """Instance-scoped coordinate welding and shared-edge face graph."""

    face_neighbors: tuple[tuple[_FaceNeighbor, ...], ...]
    welded_vertex_ids_by_instance: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class _CameraProjectionFrame:
    """Full-model orthographic bounds shared by one canonical camera."""

    minimum: np.ndarray
    maximum: np.ndarray


# ### Public projection API ###
def project_uvs_from_camera_views(
    model: GeneratedModel,
    *,
    options: CameraUvProjectionOptions | None = None,
    cancel_requested: CancelCallback | None = None,
    progress_callback: ProgressCallback | None = None,
) -> CameraUvProjectionResult:
    """Project one generated model through the six fixed camera views."""

    if not isinstance(model, GeneratedModel):
        raise TypeError("Camera UV projection requires a GeneratedModel.")
    return project_uvs_from_camera_views_from_glb(
        model.glb_bytes,
        options=options,
        cancel_requested=cancel_requested,
        progress_callback=progress_callback,
    )


def project_uvs_from_camera_views_from_glb(
    glb_bytes: bytes,
    *,
    options: CameraUvProjectionOptions | None = None,
    cancel_requested: CancelCallback | None = None,
    progress_callback: ProgressCallback | None = None,
) -> CameraUvProjectionResult:
    """Return a shape-preserving GLB with compact six-view UV charts.

    Topology smoothing keeps neighboring faces on non-grazing camera views.
    Each camera retains one conflict-free layer using full-model orthographic
    bounds. Other layers and uncaptured faces remain inside the bottom-left
    128-pixel fallback square.
    """

    normalized_options = options or CameraUvProjectionOptions()
    payload = bytes(glb_bytes)
    if not payload:
        raise ValueError("The generated GLB is empty.")
    _raise_if_cancelled(cancel_requested)
    scene = _load_glb_scene(payload)
    instances, world_vertices, global_faces = _collect_scene_instances(scene)
    total_face_count = len(global_faces)
    if total_face_count == 0:
        raise ValueError("The generated GLB contains no triangle faces.")
    if total_face_count > normalized_options.max_face_count:
        raise ValueError(
            f"The generated GLB has {total_face_count} faces; the camera UV "
            f"limit is {normalized_options.max_face_count}."
        )
    topology = _build_mesh_topology(
        instances,
        world_vertices,
        global_faces,
        cancel_requested,
        normalized_options.progress_interval_faces,
    )
    camera_projection_frames = _build_camera_projection_frames(world_vertices)

    visible_faces_by_camera = _capture_all_camera_faces(
        world_vertices,
        global_faces,
        normalized_options,
        cancel_requested,
        progress_callback,
    )
    projected_faces, _camera_face_counts, _leftover_face_count = (
        _assign_projected_faces(
            instances,
            world_vertices,
            global_faces,
            visible_faces_by_camera,
            topology,
            cancel_requested,
            normalized_options.progress_interval_faces,
            progress_callback,
        )
    )
    (
        charts,
        invisible_face_count,
        quality_fallback_face_count,
        conflict_fallback_face_count,
    ) = _build_conflict_free_charts(
        projected_faces,
        camera_projection_frames,
        cancel_requested,
        normalized_options.progress_interval_faces,
    )
    projected_faces = sorted(
        (face for chart in charts for face in chart.faces),
        key=lambda face: face.global_face_index,
    )
    camera_face_counts = {
        camera_id: sum(
            len(chart.faces)
            for chart in charts
            if not chart.is_leftover and chart.camera_id == camera_id
        )
        for camera_id in ALL_CAMERA_IDS
    }
    leftover_face_count = sum(
        len(chart.faces) for chart in charts if chart.is_leftover
    )
    if leftover_face_count != (
        invisible_face_count
        + quality_fallback_face_count
        + conflict_fallback_face_count
    ):
        raise RuntimeError("Camera UV fallback accounting is inconsistent.")
    _report_progress(
        progress_callback,
        "packing",
        0,
        total_face_count,
    )
    packed_layout = _pack_charts(
        charts,
        normalized_options,
        cancel_requested,
    )
    _raise_if_cancelled(cancel_requested)
    output_scene, output_vertex_count = _build_output_scene(
        instances,
        projected_faces,
        charts,
        packed_layout,
        normalized_options,
        topology,
        cancel_requested,
    )
    _report_progress(
        progress_callback,
        "exporting",
        total_face_count,
        total_face_count,
    )
    _raise_if_cancelled(cancel_requested)
    output_glb = _export_scene(output_scene)
    _raise_if_cancelled(cancel_requested)
    output_model = import_generated_glb(output_glb)
    _raise_if_cancelled(cancel_requested)
    _validate_exported_uvs(output_glb)
    _raise_if_cancelled(cancel_requested)
    chart_regions = tuple(
        CameraUvChartRegion(
            chart_index=chart.chart_index,
            camera_id=chart.camera_id,
            is_leftover=chart.is_leftover,
            x=packed_layout.placements[chart.chart_index].x,
            y=packed_layout.placements[chart.chart_index].y,
            width=packed_layout.placements[chart.chart_index].width,
            height=packed_layout.placements[chart.chart_index].height,
            face_count=len(chart.faces),
        )
        for chart in charts
    )
    _raise_if_cancelled(cancel_requested)
    _report_progress(
        progress_callback,
        "complete",
        total_face_count,
        total_face_count,
    )
    return CameraUvProjectionResult(
        model=output_model,
        camera_face_counts=camera_face_counts,
        leftover_face_count=leftover_face_count,
        invisible_face_count=invisible_face_count,
        quality_fallback_face_count=quality_fallback_face_count,
        conflict_fallback_face_count=conflict_fallback_face_count,
        original_face_count=total_face_count,
        original_vertex_count=sum(
            len(instance.mesh.vertices) for instance in instances
        ),
        output_vertex_count=output_vertex_count,
        compatible_texture_sizes=CAMERA_UV_TEXTURE_SIZES,
        chart_regions=chart_regions,
    )


# ### Scene collection helpers ###
def _load_glb_scene(payload: bytes) -> trimesh.Scene:
    try:
        loaded = trimesh.load(
            BytesIO(payload),
            file_type="glb",
            force="scene",
            process=False,
        )
    except Exception as error:
        raise ValueError("The generated GLB could not be loaded.") from error
    if isinstance(loaded, trimesh.Trimesh):
        return trimesh.Scene(loaded)
    if isinstance(loaded, trimesh.Scene):
        return loaded
    raise ValueError("The generated GLB contains no mesh scene.")


def _collect_scene_instances(
    scene: trimesh.Scene,
) -> tuple[list[_MeshInstance], np.ndarray, np.ndarray]:
    instances: list[_MeshInstance] = []
    world_vertices_parts: list[np.ndarray] = []
    face_parts: list[np.ndarray] = []
    vertex_offset = 0
    face_offset = 0
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph.get(node_name)
        geometry = scene.geometry.get(geometry_name)
        if not isinstance(geometry, trimesh.Trimesh):
            continue
        if len(geometry.vertices) == 0 or len(geometry.faces) == 0:
            continue
        local_vertices = np.asarray(geometry.vertices, dtype=float)
        local_faces = np.asarray(geometry.faces, dtype=np.int64)
        node_transform = np.asarray(transform, dtype=float)
        if node_transform.shape != (4, 4) or not np.all(np.isfinite(node_transform)):
            raise ValueError("The generated GLB contains an invalid node transform.")
        if local_faces.ndim != 2 or local_faces.shape[1:] != (3,):
            raise ValueError("Camera UV projection requires triangle faces.")
        if not np.all(np.isfinite(local_vertices)):
            raise ValueError("The generated GLB contains invalid vertices.")
        gltf_world_vertices = trimesh.transform_points(
            local_vertices,
            node_transform,
        )
        z_up_world_vertices = trimesh.transform_points(
            gltf_world_vertices,
            GLTF_Y_UP_TO_Z_UP_TRANSFORM,
        )
        instances.append(
            _MeshInstance(
                node_name=str(node_name),
                geometry_name=str(geometry_name),
                mesh=geometry.copy(),
                transform=node_transform.copy(),
                world_vertices=z_up_world_vertices,
                vertex_normals=_get_source_vertex_normals(geometry),
                first_face_index=face_offset,
            )
        )
        world_vertices_parts.append(z_up_world_vertices)
        face_parts.append(local_faces + vertex_offset)
        vertex_offset += len(local_vertices)
        face_offset += len(local_faces)
    if not world_vertices_parts:
        return (
            instances,
            np.empty((0, 3), dtype=float),
            np.empty((0, 3), dtype=np.int64),
        )
    return instances, np.vstack(world_vertices_parts), np.vstack(face_parts)


# ### Coordinate-welded topology helpers ###
def _build_mesh_topology(
    instances: Sequence[_MeshInstance],
    world_vertices: np.ndarray,
    global_faces: np.ndarray,
    cancel_requested: CancelCallback | None,
    progress_interval_faces: int,
) -> _MeshTopology:
    face_count = len(global_faces)
    face_neighbors: list[dict[int, float]] = [
        {} for _face_index in range(face_count)
    ]
    triangles = world_vertices[global_faces]
    weighted_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    normal_lengths = np.linalg.norm(weighted_normals, axis=1)
    face_normals = np.zeros_like(weighted_normals)
    valid_normals = normal_lengths > np.finfo(float).eps
    face_normals[valid_normals] = (
        weighted_normals[valid_normals]
        / normal_lengths[valid_normals, np.newaxis]
    )
    face_perimeters = np.sum(
        np.linalg.norm(
            triangles - np.roll(triangles, -1, axis=1),
            axis=2,
        ),
        axis=1,
    )

    welded_vertex_ids_by_instance: list[np.ndarray] = []
    for instance in instances:
        _raise_if_cancelled(cancel_requested)
        welded_vertex_ids = _weld_instance_vertex_coordinates(
            instance.world_vertices,
            np.asarray(instance.mesh.faces, dtype=np.int64),
            cancel_requested,
            progress_interval_faces,
        )
        welded_vertex_ids_by_instance.append(welded_vertex_ids)
        face_material_ids = _get_face_material_ids(instance.mesh)
        edge_owners: dict[
            tuple[int, int],
            list[tuple[int, float, int]],
        ] = defaultdict(list)
        local_faces = np.asarray(instance.mesh.faces, dtype=np.int64)
        for local_face_index, face in enumerate(local_faces):
            if local_face_index % progress_interval_faces == 0:
                _raise_if_cancelled(cancel_requested)
            global_face_index = instance.first_face_index + local_face_index
            for corner_index in range(3):
                first_vertex_index = int(face[corner_index])
                second_vertex_index = int(face[(corner_index + 1) % 3])
                first_welded_id = int(welded_vertex_ids[first_vertex_index])
                second_welded_id = int(welded_vertex_ids[second_vertex_index])
                if first_welded_id == second_welded_id:
                    continue
                edge_key = (
                    min(first_welded_id, second_welded_id),
                    max(first_welded_id, second_welded_id),
                )
                edge_length = float(
                    np.linalg.norm(
                        instance.world_vertices[first_vertex_index]
                        - instance.world_vertices[second_vertex_index]
                    )
                )
                edge_owners[edge_key].append(
                    (
                        global_face_index,
                        edge_length,
                        int(face_material_ids[local_face_index]),
                    )
                )

        checked_owner_pair_count = 0
        for owners in edge_owners.values():
            for first_owner_index, first_owner in enumerate(owners):
                first_face_index, first_edge_length, first_material_id = (
                    first_owner
                )
                for second_owner in owners[first_owner_index + 1 :]:
                    if checked_owner_pair_count % progress_interval_faces == 0:
                        _raise_if_cancelled(cancel_requested)
                    checked_owner_pair_count += 1
                    second_face_index, second_edge_length, second_material_id = (
                        second_owner
                    )
                    if (
                        first_face_index == second_face_index
                        or first_material_id != second_material_id
                    ):
                        continue
                    normal_alignment = min(
                        abs(
                        float(
                            np.dot(
                                face_normals[first_face_index],
                                face_normals[second_face_index],
                            )
                        ),
                        ),
                        1.0,
                    )
                    if normal_alignment < TOPOLOGY_NORMAL_CONNECTION_COSINE:
                        continue
                    normal_weight = TOPOLOGY_NORMAL_WEIGHT_FLOOR + (
                        1.0 - TOPOLOGY_NORMAL_WEIGHT_FLOOR
                    ) * normal_alignment**TOPOLOGY_NORMAL_WEIGHT_EXPONENT
                    shared_edge_length = min(
                        first_edge_length,
                        second_edge_length,
                    )
                    perimeter_scale = max(
                        math.sqrt(
                            face_perimeters[first_face_index]
                            * face_perimeters[second_face_index]
                        ),
                        np.finfo(float).eps,
                    )
                    seam_weight = (
                        shared_edge_length / perimeter_scale * normal_weight
                    )
                    if seam_weight <= 0.0:
                        continue
                    _store_face_neighbor_weight(
                        face_neighbors,
                        first_face_index,
                        second_face_index,
                        seam_weight,
                    )

    return _MeshTopology(
        face_neighbors=tuple(
            tuple(
                _FaceNeighbor(face_index=neighbor_index, seam_weight=weight)
                for neighbor_index, weight in sorted(neighbors.items())
            )
            for neighbors in face_neighbors
        ),
        welded_vertex_ids_by_instance=tuple(welded_vertex_ids_by_instance),
    )


def _weld_instance_vertex_coordinates(
    vertices: np.ndarray,
    faces: np.ndarray,
    cancel_requested: CancelCallback | None,
    progress_interval_faces: int,
) -> np.ndarray:
    normalized_vertices = np.asarray(vertices, dtype=float)
    if len(normalized_vertices) == 0:
        return np.empty(0, dtype=np.int64)
    edge_lengths = np.linalg.norm(
        normalized_vertices[faces].reshape(-1, 3, 3)[:, (1, 2, 0)]
        - normalized_vertices[faces].reshape(-1, 3, 3)[:, (0, 1, 2)],
        axis=2,
    ).reshape(-1)
    positive_edge_lengths = edge_lengths[
        edge_lengths > np.finfo(float).eps
    ]
    representative_edge_length = (
        float(np.median(positive_edge_lengths))
        if len(positive_edge_lengths)
        else 1.0
    )
    maximum_coordinate = max(
        float(np.max(np.abs(normalized_vertices))),
        1.0,
    )
    weld_distance = max(
        representative_edge_length * TOPOLOGY_WELD_EDGE_RATIO,
        abs(float(np.spacing(maximum_coordinate)))
        * TOPOLOGY_WELD_ULP_MULTIPLIER,
        MINIMUM_TOPOLOGY_WELD_DISTANCE,
    )
    origin = np.min(normalized_vertices, axis=0)
    cell_offsets = tuple(product((-1, 0, 1), repeat=3))
    exact_lookup: dict[tuple[float, float, float], int] = {}
    cell_entries: dict[
        tuple[int, int, int],
        list[tuple[np.ndarray, int]],
    ] = defaultdict(list)
    welded_ids = np.empty(len(normalized_vertices), dtype=np.int64)
    next_welded_id = 0
    for vertex_index, vertex in enumerate(normalized_vertices):
        if vertex_index % progress_interval_faces == 0:
            _raise_if_cancelled(cancel_requested)
        exact_key = tuple(float(value) for value in vertex)
        exact_welded_id = exact_lookup.get(exact_key)
        if exact_welded_id is not None:
            welded_ids[vertex_index] = exact_welded_id
            continue
        raw_cell = np.floor((vertex - origin) / weld_distance).astype(np.int64)
        cell = tuple(int(value) for value in raw_cell)
        matched_welded_id: int | None = None
        for offset in cell_offsets:
            neighbor_cell = (
                cell[0] + offset[0],
                cell[1] + offset[1],
                cell[2] + offset[2],
            )
            for existing_vertex, welded_id in cell_entries.get(
                neighbor_cell,
                (),
            ):
                if float(np.linalg.norm(vertex - existing_vertex)) <= weld_distance:
                    matched_welded_id = welded_id
                    break
            if matched_welded_id is not None:
                break
        if matched_welded_id is None:
            matched_welded_id = next_welded_id
            next_welded_id += 1
            cell_entries[cell].append((vertex.copy(), matched_welded_id))
        exact_lookup[exact_key] = matched_welded_id
        welded_ids[vertex_index] = matched_welded_id
    return welded_ids


def _get_face_material_ids(mesh: trimesh.Trimesh) -> np.ndarray:
    face_materials = getattr(mesh.visual, "face_materials", None)
    if face_materials is None:
        return np.zeros(len(mesh.faces), dtype=np.int64)
    normalized = np.asarray(face_materials, dtype=np.int64)
    if normalized.shape != (len(mesh.faces),):
        return np.zeros(len(mesh.faces), dtype=np.int64)
    return normalized.copy()


def _store_face_neighbor_weight(
    face_neighbors: list[dict[int, float]],
    first_face_index: int,
    second_face_index: int,
    seam_weight: float,
) -> None:
    first_neighbors = face_neighbors[first_face_index]
    second_neighbors = face_neighbors[second_face_index]
    first_neighbors[second_face_index] = max(
        first_neighbors.get(second_face_index, 0.0),
        seam_weight,
    )
    second_neighbors[first_face_index] = max(
        second_neighbors.get(first_face_index, 0.0),
        seam_weight,
    )


# ### Visibility and projection helpers ###
def _capture_all_camera_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    options: CameraUvProjectionOptions,
    cancel_requested: CancelCallback | None,
    progress_callback: ProgressCallback | None,
) -> dict[str, frozenset[int]]:
    visible_faces: dict[str, frozenset[int]] = {}
    for camera_id in ALL_CAMERA_IDS:
        _raise_if_cancelled(cancel_requested)
        _report_progress(
            progress_callback,
            "capturing",
            0,
            len(faces),
            camera_id,
        )
        try:
            visible_faces[camera_id] = capture_visible_face_indices(
                vertices,
                faces,
                camera_id,
                image_size=options.capture_image_size,
                cancel_requested=cancel_requested,
                progress_interval_faces=options.progress_interval_faces,
            )
        except UnusedFaceRemovalCancelled as error:
            raise CameraUvProjectionCancelled(
                "Camera UV projection was cancelled."
            ) from error
        _report_progress(
            progress_callback,
            "capturing",
            len(faces),
            len(faces),
            camera_id,
        )
    return visible_faces


def _build_camera_projection_frames(
    world_vertices: np.ndarray,
) -> dict[str, _CameraProjectionFrame]:
    frames: dict[str, _CameraProjectionFrame] = {}
    for camera_id in ALL_CAMERA_IDS:
        camera_view = get_fixed_camera_view(camera_id)
        coordinates = _project_face(world_vertices, camera_view)
        minimum = np.min(coordinates, axis=0)
        maximum = np.max(coordinates, axis=0)
        collapsed_axes = maximum - minimum <= MINIMUM_PROJECTED_EXTENT
        if np.any(collapsed_axes):
            center = (minimum + maximum) / 2.0
            minimum = minimum.copy()
            maximum = maximum.copy()
            minimum[collapsed_axes] = (
                center[collapsed_axes] - MINIMUM_PROJECTED_EXTENT / 2.0
            )
            maximum[collapsed_axes] = (
                center[collapsed_axes] + MINIMUM_PROJECTED_EXTENT / 2.0
            )
        frames[camera_id] = _CameraProjectionFrame(
            minimum=minimum,
            maximum=maximum,
        )
    return frames


# ### Global camera-assignment helpers ###
def _build_camera_projection_conflicts(
    triangles: np.ndarray,
    camera_views: Sequence[object],
    camera_candidates: np.ndarray,
    cancel_requested: CancelCallback | None,
    progress_interval_faces: int,
) -> tuple[dict[int, set[int]], ...]:
    """Return positive-area projection conflicts for every candidate camera."""

    conflicts_by_camera: list[dict[int, set[int]]] = []
    checked_pair_count = 0
    for camera_index, camera_view in enumerate(camera_views):
        _raise_if_cancelled(cancel_requested)
        candidate_indices = np.flatnonzero(camera_candidates[:, camera_index])
        horizontal_axis = np.asarray(camera_view.horizontal_axis, dtype=float)
        vertical_axis = np.asarray(camera_view.vertical_axis, dtype=float)
        projected_triangles = np.stack(
            (
                triangles @ horizontal_axis,
                triangles @ vertical_axis,
            ),
            axis=2,
        )
        candidate_coordinates = projected_triangles[candidate_indices]
        if len(candidate_coordinates):
            coordinate_origin = (
                np.min(candidate_coordinates, axis=(0, 1))
                + np.max(candidate_coordinates, axis=(0, 1))
            ) / 2.0
        else:
            coordinate_origin = np.zeros(2, dtype=float)
        polygons: list[Polygon] = []
        for local_index, coordinates in enumerate(candidate_coordinates):
            if local_index % progress_interval_faces == 0:
                _raise_if_cancelled(cancel_requested)
            polygons.append(Polygon(coordinates - coordinate_origin))
        camera_conflicts = {
            int(face_index): set() for face_index in candidate_indices
        }
        if polygons:
            tree = STRtree(polygons)
            for local_index, polygon in enumerate(polygons):
                for raw_other_local_index in tree.query(polygon):
                    if checked_pair_count % progress_interval_faces == 0:
                        _raise_if_cancelled(cancel_requested)
                    checked_pair_count += 1
                    other_local_index = int(raw_other_local_index)
                    if other_local_index <= local_index:
                        continue
                    other_polygon = polygons[other_local_index]
                    if not _polygons_have_positive_overlap(
                        polygon,
                        other_polygon,
                    ):
                        continue
                    face_index = int(candidate_indices[local_index])
                    other_face_index = int(
                        candidate_indices[other_local_index]
                    )
                    camera_conflicts[face_index].add(other_face_index)
                    camera_conflicts[other_face_index].add(face_index)
        conflicts_by_camera.append(camera_conflicts)
    return tuple(conflicts_by_camera)


def _select_global_conflict_free_camera_assignments(
    camera_candidates: np.ndarray,
    projected_quality: np.ndarray,
    conflicts_by_camera: Sequence[dict[int, set[int]]],
    topology: _MeshTopology,
    cancel_requested: CancelCallback | None,
    progress_interval_faces: int,
) -> np.ndarray:
    """Choose one high-coverage conflict-free face set across all cameras."""

    best_assignment: np.ndarray | None = None
    best_face_count = -1
    best_quality = -1.0
    best_topology_edge_count = -1
    best_topology_continuity = -1.0
    best_used_camera_count = len(ALL_CAMERA_IDS) + 1
    for seed in GLOBAL_CAMERA_ASSIGNMENT_SEEDS:
        _raise_if_cancelled(cancel_requested)
        assignment = _run_global_camera_assignment(
            camera_candidates,
            projected_quality,
            conflicts_by_camera,
            topology,
            seed,
            cancel_requested,
            progress_interval_faces,
        )
        assigned_faces = assignment >= 0
        assigned_face_count = int(np.count_nonzero(assigned_faces))
        assigned_quality = float(
            np.sum(
                projected_quality[
                    assigned_faces,
                    assignment[assigned_faces],
                ]
            )
        )
        topology_edge_count, topology_continuity = (
            _camera_assignment_topology_score(
                assignment,
                topology,
                cancel_requested,
                progress_interval_faces,
            )
        )
        used_camera_count = len(set(int(value) for value in assignment[assigned_faces]))
        if (
            assigned_face_count > best_face_count
            or (
                assigned_face_count == best_face_count
                and (
                    topology_edge_count > best_topology_edge_count
                    or (
                        topology_edge_count == best_topology_edge_count
                        and (
                            topology_continuity
                            > best_topology_continuity + 1e-12
                            or (
                                abs(
                                    topology_continuity
                                    - best_topology_continuity
                                )
                                <= 1e-12
                                and (
                                    assigned_quality > best_quality + 1e-12
                                    or (
                                        abs(assigned_quality - best_quality)
                                        <= 1e-12
                                        and used_camera_count
                                        < best_used_camera_count
                                    )
                                )
                            )
                        )
                    )
                )
            )
        ):
            best_assignment = assignment
            best_face_count = assigned_face_count
            best_quality = assigned_quality
            best_topology_edge_count = topology_edge_count
            best_topology_continuity = topology_continuity
            best_used_camera_count = used_camera_count
    assert best_assignment is not None
    return best_assignment


def _camera_assignment_topology_score(
    assignment: np.ndarray,
    topology: _MeshTopology,
    cancel_requested: CancelCallback | None,
    progress_interval_faces: int,
) -> tuple[int, float]:
    matching_edge_count = 0
    continuity = 0.0
    for face_index, neighbors in enumerate(topology.face_neighbors):
        if face_index % progress_interval_faces == 0:
            _raise_if_cancelled(cancel_requested)
        camera_index = int(assignment[face_index])
        if camera_index < 0:
            continue
        for neighbor in neighbors:
            if (
                neighbor.face_index > face_index
                and int(assignment[neighbor.face_index]) == camera_index
            ):
                matching_edge_count += 1
                continuity += neighbor.seam_weight
    return matching_edge_count, continuity


def _run_global_camera_assignment(
    camera_candidates: np.ndarray,
    projected_quality: np.ndarray,
    conflicts_by_camera: Sequence[dict[int, set[int]]],
    topology: _MeshTopology,
    seed: int,
    cancel_requested: CancelCallback | None,
    progress_interval_faces: int,
) -> np.ndarray:
    face_count, camera_count = camera_candidates.shape
    generator = random.Random(seed)
    assigned = np.full(face_count, -1, dtype=np.int8)
    processed = ~np.any(camera_candidates, axis=1)
    blocker_counts = np.zeros((face_count, camera_count), dtype=np.int32)
    random_face_ties = np.asarray(
        [generator.random() for _face_index in range(face_count)],
        dtype=float,
    )
    versions = np.zeros(face_count, dtype=np.int32)
    pending: list[tuple[float, ...]] = []

    def feasible_cameras(face_index: int) -> np.ndarray:
        return np.flatnonzero(
            camera_candidates[face_index]
            & (blocker_counts[face_index] == 0)
        )

    def push(face_index: int) -> None:
        if processed[face_index]:
            return
        feasible = feasible_cameras(face_index)
        pressure = sum(
            len(conflicts_by_camera[int(camera_index)].get(face_index, ()))
            for camera_index in feasible
        )
        assigned_topology_weight = sum(
            neighbor.seam_weight
            for neighbor in topology.face_neighbors[face_index]
            if int(assigned[neighbor.face_index]) >= 0
        )
        versions[face_index] += 1
        heapq.heappush(
            pending,
            (
                len(feasible),
                pressure,
                -assigned_topology_weight,
                random_face_ties[face_index],
                face_index,
                int(versions[face_index]),
            ),
        )

    for face_index in range(face_count):
        if face_index % progress_interval_faces == 0:
            _raise_if_cancelled(cancel_requested)
        push(face_index)

    checked_face_count = 0
    while pending:
        if checked_face_count % progress_interval_faces == 0:
            _raise_if_cancelled(cancel_requested)
        checked_face_count += 1
        (
            _feasible_count,
            _pressure,
            _assigned_topology_weight,
            _random_tie,
            face_index,
            version,
        ) = heapq.heappop(pending)
        if processed[face_index] or version != versions[face_index]:
            continue
        feasible = feasible_cameras(face_index)
        if not len(feasible):
            processed[face_index] = True
            continue
        choices: list[tuple[tuple[float, ...], int]] = []
        for raw_camera_index in feasible:
            camera_index = int(raw_camera_index)
            critical_neighbor_count = 0
            fractional_damage = 0.0
            live_conflict_count = 0
            for neighbor_index in conflicts_by_camera[camera_index].get(
                face_index,
                (),
            ):
                if processed[neighbor_index]:
                    continue
                neighbor_feasible_count = len(
                    feasible_cameras(neighbor_index)
                )
                if neighbor_feasible_count <= 0:
                    continue
                live_conflict_count += 1
                fractional_damage += 1.0 / neighbor_feasible_count
                if neighbor_feasible_count == 1:
                    critical_neighbor_count += 1
            topology_continuity = sum(
                neighbor.seam_weight
                for neighbor in topology.face_neighbors[face_index]
                if int(assigned[neighbor.face_index]) == camera_index
            )
            choices.append(
                (
                    (
                        critical_neighbor_count,
                        fractional_damage,
                        live_conflict_count,
                        -topology_continuity,
                        -float(projected_quality[face_index, camera_index]),
                        camera_index,
                        generator.random(),
                    ),
                    camera_index,
                )
            )
        camera_index = min(choices)[1]
        assigned[face_index] = camera_index
        processed[face_index] = True
        for neighbor_index in conflicts_by_camera[camera_index].get(
            face_index,
            (),
        ):
            if processed[neighbor_index]:
                continue
            blocker_counts[neighbor_index, camera_index] += 1
            push(neighbor_index)
        for neighbor in topology.face_neighbors[face_index]:
            push(neighbor.face_index)

    _augment_camera_assignments(
        assigned,
        camera_candidates,
        projected_quality,
        conflicts_by_camera,
        generator,
        cancel_requested,
        progress_interval_faces,
    )
    _solve_small_camera_assignment_components_exactly(
        assigned,
        camera_candidates,
        projected_quality,
        conflicts_by_camera,
        cancel_requested,
        progress_interval_faces,
    )
    _smooth_conflict_free_camera_assignments(
        assigned,
        camera_candidates,
        projected_quality,
        conflicts_by_camera,
        topology,
        cancel_requested,
        progress_interval_faces,
    )
    return assigned


def _augment_camera_assignments(
    assigned: np.ndarray,
    camera_candidates: np.ndarray,
    projected_quality: np.ndarray,
    conflicts_by_camera: Sequence[dict[int, set[int]]],
    generator: random.Random,
    cancel_requested: CancelCallback | None,
    progress_interval_faces: int,
) -> None:
    changed = True
    while changed:
        _raise_if_cancelled(cancel_requested)
        changed = False
        rejected_faces = [
            int(face_index) for face_index in np.flatnonzero(assigned < 0)
            if np.any(camera_candidates[face_index])
        ]
        generator.shuffle(rejected_faces)
        for rejected_index, face_index in enumerate(rejected_faces):
            if rejected_index % progress_interval_faces == 0:
                _raise_if_cancelled(cancel_requested)
            choices: list[
                tuple[tuple[float, ...], int, int | None, int | None]
            ] = []
            for raw_camera_index in np.flatnonzero(
                camera_candidates[face_index]
            ):
                camera_index = int(raw_camera_index)
                blockers = [
                    neighbor_index
                    for neighbor_index in conflicts_by_camera[
                        camera_index
                    ].get(face_index, ())
                    if int(assigned[neighbor_index]) == camera_index
                ]
                if not blockers:
                    choices.append(
                        (
                            (
                                0,
                                -float(
                                    projected_quality[face_index, camera_index]
                                ),
                            ),
                            camera_index,
                            None,
                            None,
                        )
                    )
                    continue
                if len(blockers) != 1:
                    continue
                blocker = blockers[0]
                for raw_alternative in np.flatnonzero(
                    camera_candidates[blocker]
                ):
                    alternative = int(raw_alternative)
                    if alternative == camera_index:
                        continue
                    if any(
                        int(assigned[neighbor_index]) == alternative
                        for neighbor_index in conflicts_by_camera[
                            alternative
                        ].get(blocker, ())
                    ):
                        continue
                    choices.append(
                        (
                            (
                                1,
                                -float(
                                    projected_quality[face_index, camera_index]
                                ),
                            ),
                            camera_index,
                            blocker,
                            alternative,
                        )
                    )
            if not choices:
                continue
            _score, camera_index, blocker, alternative = min(choices)
            if blocker is not None and alternative is not None:
                assigned[blocker] = alternative
            assigned[face_index] = camera_index
            changed = True


def _solve_small_camera_assignment_components_exactly(
    assigned: np.ndarray,
    camera_candidates: np.ndarray,
    projected_quality: np.ndarray,
    conflicts_by_camera: Sequence[dict[int, set[int]]],
    cancel_requested: CancelCallback | None,
    progress_interval_faces: int,
) -> None:
    """Improve small conflict components with exact branch-and-bound."""

    components = _candidate_conflict_components(
        camera_candidates,
        conflicts_by_camera,
        cancel_requested,
        progress_interval_faces,
    )
    for component_index, component in enumerate(components):
        if component_index % progress_interval_faces == 0:
            _raise_if_cancelled(cancel_requested)
        if (
            len(component) > EXACT_CAMERA_ASSIGNMENT_COMPONENT_LIMIT
            or np.all(assigned[np.asarray(component, dtype=np.int64)] >= 0)
        ):
            continue
        optimized = _solve_camera_assignment_component_exactly(
            component,
            assigned,
            camera_candidates,
            projected_quality,
            conflicts_by_camera,
            cancel_requested,
            progress_interval_faces,
        )
        component_indices = np.asarray(component, dtype=np.int64)
        if np.count_nonzero(optimized >= 0) > np.count_nonzero(
            assigned[component_indices] >= 0
        ):
            assigned[component_indices] = optimized


def _candidate_conflict_components(
    camera_candidates: np.ndarray,
    conflicts_by_camera: Sequence[dict[int, set[int]]],
    cancel_requested: CancelCallback | None,
    progress_interval_faces: int,
) -> list[tuple[int, ...]]:
    eligible_face_indices = [
        int(face_index)
        for face_index in np.flatnonzero(np.any(camera_candidates, axis=1))
    ]
    unvisited = set(eligible_face_indices)
    components: list[tuple[int, ...]] = []
    visited_face_count = 0
    for first_face_index in eligible_face_indices:
        if first_face_index not in unvisited:
            continue
        _raise_if_cancelled(cancel_requested)
        unvisited.remove(first_face_index)
        pending = [first_face_index]
        component: list[int] = []
        while pending:
            if visited_face_count % progress_interval_faces == 0:
                _raise_if_cancelled(cancel_requested)
            visited_face_count += 1
            face_index = pending.pop()
            component.append(face_index)
            for raw_camera_index in np.flatnonzero(
                camera_candidates[face_index]
            ):
                camera_index = int(raw_camera_index)
                for neighbor_index in conflicts_by_camera[
                    camera_index
                ].get(face_index, ()):
                    if neighbor_index not in unvisited:
                        continue
                    unvisited.remove(neighbor_index)
                    pending.append(neighbor_index)
        components.append(tuple(sorted(component)))
    return components


def _solve_camera_assignment_component_exactly(
    component: Sequence[int],
    assigned: np.ndarray,
    camera_candidates: np.ndarray,
    projected_quality: np.ndarray,
    conflicts_by_camera: Sequence[dict[int, set[int]]],
    cancel_requested: CancelCallback | None,
    progress_interval_faces: int,
) -> np.ndarray:
    """Run bounded exact maximum-clique search for one small component."""

    component_size = len(component)
    component_indices = np.asarray(component, dtype=np.int64)
    incumbent = np.asarray(assigned[component_indices], dtype=np.int8).copy()
    best_assignment = incumbent.copy()
    best_assigned_count = int(np.count_nonzero(incumbent >= 0))
    if best_assigned_count == component_size:
        return best_assignment

    options = sorted(
        (
            (local_face_index, int(camera_index))
            for local_face_index, global_face_index in enumerate(component)
            for camera_index in np.flatnonzero(
                camera_candidates[global_face_index]
            )
        ),
        key=lambda option: (
            int(incumbent[option[0]]) == option[1],
            float(
                projected_quality[component[option[0]], option[1]]
            ),
            -option[1],
            -option[0],
        ),
    )
    option_count = len(options)
    compatibility_masks = [0 for _option_index in range(option_count)]
    for first_option_index, (first_face, first_camera) in enumerate(options):
        if first_option_index % progress_interval_faces == 0:
            _raise_if_cancelled(cancel_requested)
        first_global_face = component[first_face]
        for second_option_index in range(first_option_index + 1, option_count):
            second_face, second_camera = options[second_option_index]
            if first_face == second_face:
                continue
            if (
                first_camera == second_camera
                and component[second_face]
                in conflicts_by_camera[first_camera].get(
                    first_global_face,
                    (),
                )
            ):
                continue
            compatibility_masks[first_option_index] |= 1 << second_option_index
            compatibility_masks[second_option_index] |= 1 << first_option_index

    best_clique = [
        option_index
        for option_index, (face_index, camera_index) in enumerate(options)
        if int(incumbent[face_index]) == camera_index
    ]
    current_clique: list[int] = []
    checked_node_count = 0
    search_budget_exhausted = False

    def color_order(candidate_mask: int) -> tuple[list[int], list[int]]:
        order: list[int] = []
        color_bounds: list[int] = []
        remaining_mask = candidate_mask
        color_index = 0
        while remaining_mask:
            color_index += 1
            available_mask = remaining_mask
            while available_mask:
                option_bit = available_mask & -available_mask
                option_index = option_bit.bit_length() - 1
                order.append(option_index)
                color_bounds.append(color_index)
                remaining_mask ^= option_bit
                available_mask ^= option_bit
                available_mask &= ~compatibility_masks[option_index]
        return order, color_bounds

    def expand(candidate_mask: int) -> None:
        nonlocal best_clique
        nonlocal checked_node_count
        nonlocal search_budget_exhausted

        if (
            len(best_clique) == component_size
            or search_budget_exhausted
        ):
            return
        if checked_node_count % progress_interval_faces == 0:
            _raise_if_cancelled(cancel_requested)
        if checked_node_count >= EXACT_CAMERA_ASSIGNMENT_MAX_SEARCH_NODES:
            search_budget_exhausted = True
            return
        checked_node_count += 1
        order, color_bounds = color_order(candidate_mask)
        for order_index in range(len(order) - 1, -1, -1):
            if len(current_clique) + color_bounds[order_index] <= len(
                best_clique
            ):
                return
            option_index = order[order_index]
            option_bit = 1 << option_index
            if not candidate_mask & option_bit:
                continue
            current_clique.append(option_index)
            compatible_candidates = (
                candidate_mask & compatibility_masks[option_index]
            )
            if compatible_candidates:
                expand(compatible_candidates)
            elif len(current_clique) > len(best_clique):
                best_clique = current_clique.copy()
            current_clique.pop()
            candidate_mask ^= option_bit
            if (
                len(best_clique) == component_size
                or search_budget_exhausted
            ):
                return

    expand((1 << option_count) - 1)
    best_assignment.fill(-1)
    for option_index in best_clique:
        face_index, camera_index = options[option_index]
        best_assignment[face_index] = camera_index
    return best_assignment


def _smooth_conflict_free_camera_assignments(
    assigned: np.ndarray,
    camera_candidates: np.ndarray,
    projected_quality: np.ndarray,
    conflicts_by_camera: Sequence[dict[int, set[int]]],
    topology: _MeshTopology,
    cancel_requested: CancelCallback | None,
    progress_interval_faces: int,
) -> None:
    """Reduce topology seams without dropping any accepted face."""

    _smooth_individual_camera_assignments(
        assigned,
        camera_candidates,
        projected_quality,
        conflicts_by_camera,
        topology,
        cancel_requested,
        progress_interval_faces,
    )
    _smooth_camera_assignment_regions(
        assigned,
        camera_candidates,
        projected_quality,
        conflicts_by_camera,
        topology,
        cancel_requested,
        progress_interval_faces,
    )


def _smooth_individual_camera_assignments(
    assigned: np.ndarray,
    camera_candidates: np.ndarray,
    projected_quality: np.ndarray,
    conflicts_by_camera: Sequence[dict[int, set[int]]],
    topology: _MeshTopology,
    cancel_requested: CancelCallback | None,
    progress_interval_faces: int,
) -> None:
    face_count = len(assigned)
    for pass_index in range(CAMERA_ASSIGNMENT_SMOOTHING_PASSES):
        _raise_if_cancelled(cancel_requested)
        changed = False
        face_indices: Iterable[int]
        if pass_index % 2:
            face_indices = range(face_count - 1, -1, -1)
        else:
            face_indices = range(face_count)
        for checked_face_count, face_index in enumerate(face_indices):
            if checked_face_count % progress_interval_faces == 0:
                _raise_if_cancelled(cancel_requested)
            current_camera_index = int(assigned[face_index])
            if current_camera_index < 0:
                continue
            best_camera_index = current_camera_index
            best_score = _single_face_assignment_score(
                face_index,
                current_camera_index,
                assigned,
                projected_quality,
                topology,
            )
            for raw_camera_index in np.flatnonzero(
                camera_candidates[face_index]
            ):
                camera_index = int(raw_camera_index)
                if camera_index == current_camera_index:
                    continue
                if any(
                    int(assigned[neighbor_index]) == camera_index
                    for neighbor_index in conflicts_by_camera[
                        camera_index
                    ].get(face_index, ())
                ):
                    continue
                score = _single_face_assignment_score(
                    face_index,
                    camera_index,
                    assigned,
                    projected_quality,
                    topology,
                )
                if score > best_score:
                    best_camera_index = camera_index
                    best_score = score
            if best_camera_index != current_camera_index:
                assigned[face_index] = best_camera_index
                changed = True
        if not changed:
            break


def _single_face_assignment_score(
    face_index: int,
    camera_index: int,
    assigned: np.ndarray,
    projected_quality: np.ndarray,
    topology: _MeshTopology,
) -> tuple[int, float, float, int]:
    matching_neighbors = [
        neighbor
        for neighbor in topology.face_neighbors[face_index]
        if int(assigned[neighbor.face_index]) == camera_index
    ]
    return (
        len(matching_neighbors),
        sum(neighbor.seam_weight for neighbor in matching_neighbors),
        float(projected_quality[face_index, camera_index]),
        -camera_index,
    )


def _smooth_camera_assignment_regions(
    assigned: np.ndarray,
    camera_candidates: np.ndarray,
    projected_quality: np.ndarray,
    conflicts_by_camera: Sequence[dict[int, set[int]]],
    topology: _MeshTopology,
    cancel_requested: CancelCallback | None,
    progress_interval_faces: int,
) -> None:
    for _pass_index in range(CAMERA_ASSIGNMENT_REGION_SMOOTHING_PASSES):
        _raise_if_cancelled(cancel_requested)
        regions = _camera_assignment_regions(
            assigned,
            topology,
            cancel_requested,
            progress_interval_faces,
        )
        changed = False
        for region_index, region in enumerate(
            sorted(regions, key=lambda value: (len(value), value[0]))
        ):
            if region_index % progress_interval_faces == 0:
                _raise_if_cancelled(cancel_requested)
            current_camera_index = int(assigned[region[0]])
            common_candidates = set(
                int(camera_index)
                for camera_index in np.flatnonzero(
                    camera_candidates[region[0]]
                )
            )
            for face_index in region[1:]:
                common_candidates.intersection_update(
                    int(camera_index)
                    for camera_index in np.flatnonzero(
                        camera_candidates[face_index]
                    )
                )
            common_candidates.discard(current_camera_index)
            if not common_candidates:
                continue
            region_set = set(region)
            best_camera_index = current_camera_index
            best_score = (0, 0.0, 0.0, 0)
            for camera_index in sorted(common_candidates):
                if not _region_camera_assignment_is_conflict_free(
                    region,
                    region_set,
                    camera_index,
                    assigned,
                    conflicts_by_camera,
                ):
                    continue
                score = _region_assignment_delta_score(
                    region,
                    region_set,
                    current_camera_index,
                    camera_index,
                    assigned,
                    projected_quality,
                    topology,
                )
                if score > best_score:
                    best_camera_index = camera_index
                    best_score = score
            if best_camera_index == current_camera_index:
                continue
            assigned[np.asarray(region, dtype=np.int64)] = best_camera_index
            changed = True
        if not changed:
            break


def _camera_assignment_regions(
    assigned: np.ndarray,
    topology: _MeshTopology,
    cancel_requested: CancelCallback | None,
    progress_interval_faces: int,
) -> list[tuple[int, ...]]:
    assigned_face_indices = [
        int(index) for index in np.flatnonzero(assigned >= 0)
    ]
    unvisited = set(assigned_face_indices)
    regions: list[tuple[int, ...]] = []
    checked_face_count = 0
    for first_face_index in assigned_face_indices:
        if first_face_index not in unvisited:
            continue
        _raise_if_cancelled(cancel_requested)
        unvisited.remove(first_face_index)
        camera_index = int(assigned[first_face_index])
        pending = [first_face_index]
        region: list[int] = []
        while pending:
            if checked_face_count % progress_interval_faces == 0:
                _raise_if_cancelled(cancel_requested)
            checked_face_count += 1
            face_index = pending.pop()
            region.append(face_index)
            for neighbor in topology.face_neighbors[face_index]:
                neighbor_index = neighbor.face_index
                if (
                    neighbor_index in unvisited
                    and int(assigned[neighbor_index]) == camera_index
                ):
                    unvisited.remove(neighbor_index)
                    pending.append(neighbor_index)
        regions.append(tuple(sorted(region)))
    return regions


def _region_camera_assignment_is_conflict_free(
    region: Sequence[int],
    region_set: set[int],
    camera_index: int,
    assigned: np.ndarray,
    conflicts_by_camera: Sequence[dict[int, set[int]]],
) -> bool:
    for face_index in region:
        for neighbor_index in conflicts_by_camera[camera_index].get(
            face_index,
            (),
        ):
            if neighbor_index in region_set:
                return False
            if int(assigned[neighbor_index]) == camera_index:
                return False
    return True


def _region_assignment_delta_score(
    region: Sequence[int],
    region_set: set[int],
    current_camera_index: int,
    target_camera_index: int,
    assigned: np.ndarray,
    projected_quality: np.ndarray,
    topology: _MeshTopology,
) -> tuple[int, float, float, int]:
    matching_edge_delta = 0
    continuity_delta = 0.0
    for face_index in region:
        for neighbor in topology.face_neighbors[face_index]:
            if neighbor.face_index in region_set:
                continue
            neighbor_camera_index = int(assigned[neighbor.face_index])
            if neighbor_camera_index == target_camera_index:
                matching_edge_delta += 1
                continuity_delta += neighbor.seam_weight
            elif neighbor_camera_index == current_camera_index:
                matching_edge_delta -= 1
                continuity_delta -= neighbor.seam_weight
    quality_delta = sum(
        float(
            projected_quality[face_index, target_camera_index]
            - projected_quality[face_index, current_camera_index]
        )
        for face_index in region
    )
    canonical_delta = (
        current_camera_index - target_camera_index
    ) * len(region)
    return (
        matching_edge_delta,
        continuity_delta,
        quality_delta,
        canonical_delta,
    )


# ### Projected-face construction helpers ###
def _assign_projected_faces(
    instances: Sequence[_MeshInstance],
    world_vertices: np.ndarray,
    global_faces: np.ndarray,
    visible_faces_by_camera: dict[str, frozenset[int]],
    topology: _MeshTopology,
    cancel_requested: CancelCallback | None,
    progress_interval_faces: int,
    progress_callback: ProgressCallback | None,
) -> tuple[list[_ProjectedFace], dict[str, int], int]:
    camera_views = tuple(
        get_fixed_camera_view(camera_id) for camera_id in ALL_CAMERA_IDS
    )
    instance_by_face: list[tuple[int, int]] = []
    for instance_index, instance in enumerate(instances):
        instance_by_face.extend(
            (instance_index, local_face_index)
            for local_face_index in range(instance.face_count)
        )
    triangles = world_vertices[global_faces]
    weighted_face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    projected_areas = np.column_stack(
        tuple(
            np.abs(
                weighted_face_normals
                @ np.asarray(camera_view.depth_axis, dtype=float)
            )
            / 2.0
            for camera_view in camera_views
        )
    )
    maximum_projected_area = np.max(projected_areas, axis=1)
    projected_quality = projected_areas / np.maximum(
        maximum_projected_area[:, np.newaxis],
        np.finfo(float).tiny,
    )
    depth_visible_candidates = np.zeros(
        (len(global_faces), len(ALL_CAMERA_IDS)),
        dtype=bool,
    )
    for camera_index, camera_id in enumerate(ALL_CAMERA_IDS):
        visible_face_indices = np.fromiter(
            visible_faces_by_camera[camera_id],
            dtype=np.int64,
        )
        depth_visible_candidates[visible_face_indices, camera_index] = (
            projected_areas[visible_face_indices, camera_index]
            > MINIMUM_PROJECTED_EXTENT
        )
    camera_candidates = depth_visible_candidates & (
        projected_quality >= MINIMUM_ACCEPTED_PROJECTED_QUALITY
    )
    invisible_faces = ~np.any(depth_visible_candidates, axis=1)
    quality_fallback_faces = (
        ~invisible_faces & ~np.any(camera_candidates, axis=1)
    )
    conflicts_by_camera = _build_camera_projection_conflicts(
        triangles,
        camera_views,
        camera_candidates,
        cancel_requested,
        progress_interval_faces,
    )
    accepted_camera_labels = _select_global_conflict_free_camera_assignments(
        camera_candidates,
        projected_quality,
        conflicts_by_camera,
        topology,
        cancel_requested,
        progress_interval_faces,
    )
    conflict_fallback_faces = (
        np.any(camera_candidates, axis=1) & (accepted_camera_labels < 0)
    )
    leftover_faces = (
        invisible_faces | quality_fallback_faces | conflict_fallback_faces
    )
    camera_labels = np.argmax(projected_areas, axis=1).astype(np.int8)
    camera_labels[accepted_camera_labels >= 0] = accepted_camera_labels[
        accepted_camera_labels >= 0
    ]
    for face_index in np.flatnonzero(conflict_fallback_faces):
        candidates = np.flatnonzero(camera_candidates[face_index])
        camera_labels[face_index] = max(
            candidates,
            key=lambda camera_index: (
                float(projected_quality[face_index, camera_index]),
                -len(
                    conflicts_by_camera[int(camera_index)].get(
                        int(face_index),
                        (),
                    )
                ),
                -int(camera_index),
            ),
        )

    _report_progress(
        progress_callback,
        "assigning",
        0,
        len(global_faces),
    )
    projected_faces: list[_ProjectedFace] = []
    for global_face_index, face in enumerate(global_faces):
        if global_face_index % progress_interval_faces == 0:
            _raise_if_cancelled(cancel_requested)
        points = world_vertices[face]
        camera_index = int(camera_labels[global_face_index])
        camera_id = ALL_CAMERA_IDS[camera_index]
        coordinates = _project_face(points, camera_views[camera_index])
        force_separate = (
            projected_areas[global_face_index, camera_index]
            <= MINIMUM_PROJECTED_EXTENT
        )
        if force_separate:
            coordinates = _fallback_degenerate_triangle(world_vertices)
        instance_index, local_face_index = instance_by_face[global_face_index]
        projected_faces.append(
            _ProjectedFace(
                global_face_index=global_face_index,
                instance_index=instance_index,
                local_face_index=local_face_index,
                camera_id=camera_id,
                coordinates=np.asarray(coordinates, dtype=float),
                is_leftover=bool(leftover_faces[global_face_index]),
                fallback_reason=(
                    FALLBACK_REASON_INVISIBLE
                    if invisible_faces[global_face_index]
                    else (
                        FALLBACK_REASON_QUALITY
                        if quality_fallback_faces[global_face_index]
                        else (
                            FALLBACK_REASON_CONFLICT
                            if conflict_fallback_faces[global_face_index]
                            else None
                        )
                    )
                ),
                force_separate_chart=force_separate,
            )
        )
    camera_face_counts = {
        camera_id: int(
            np.count_nonzero(accepted_camera_labels == camera_index)
        )
        for camera_index, camera_id in enumerate(ALL_CAMERA_IDS)
    }
    leftover_count = int(np.count_nonzero(leftover_faces))
    _report_progress(
        progress_callback,
        "assigning",
        len(global_faces),
        len(global_faces),
    )
    return projected_faces, camera_face_counts, leftover_count


def _project_face(points: np.ndarray, camera_view: object) -> np.ndarray:
    horizontal_axis = np.asarray(camera_view.horizontal_axis, dtype=float)
    vertical_axis = np.asarray(camera_view.vertical_axis, dtype=float)
    return np.column_stack(
        (points @ horizontal_axis, points @ vertical_axis)
    )


def _fallback_degenerate_triangle(vertices: np.ndarray) -> np.ndarray:
    extent = max(float(np.linalg.norm(np.ptp(vertices, axis=0))), 1.0)
    side = extent * 1e-6
    return np.asarray(((0.0, 0.0), (side, 0.0), (0.0, side)), dtype=float)


# ### Conflict-free chart helpers ###
def _build_conflict_free_charts(
    projected_faces: Sequence[_ProjectedFace],
    camera_projection_frames: dict[str, _CameraProjectionFrame],
    cancel_requested: CancelCallback | None,
    progress_interval_faces: int,
) -> tuple[list[_UvChart], int, int, int]:
    visible_faces_by_camera: dict[str, list[_ProjectedFace]] = defaultdict(list)
    conflict_faces_by_camera: dict[str, list[_ProjectedFace]] = defaultdict(list)
    synthetic_fallback_faces: list[_ProjectedFace] = []
    invisible_face_count = 0
    quality_fallback_face_count = 0
    conflict_fallback_face_count = 0
    for projected_face in projected_faces:
        if projected_face.is_leftover or projected_face.force_separate_chart:
            if projected_face.fallback_reason == FALLBACK_REASON_CONFLICT:
                conflict_faces_by_camera[projected_face.camera_id].append(
                    projected_face
                )
                conflict_fallback_face_count += 1
            elif projected_face.fallback_reason == FALLBACK_REASON_QUALITY:
                synthetic_fallback_faces.append(projected_face)
                quality_fallback_face_count += 1
            else:
                synthetic_fallback_faces.append(projected_face)
                invisible_face_count += 1
        else:
            visible_faces_by_camera[projected_face.camera_id].append(projected_face)

    primary_layers: list[tuple[_ProjectedFace, ...]] = []
    conflict_fallback_layers: list[tuple[_ProjectedFace, ...]] = []
    for camera_index, camera_id in enumerate(ALL_CAMERA_IDS):
        if camera_index % progress_interval_faces == 0:
            _raise_if_cancelled(cancel_requested)
        primary_camera_faces = visible_faces_by_camera.get(camera_id, ())
        primary_camera_layers = _color_non_overlapping_layers(
            primary_camera_faces,
            cancel_requested,
            progress_interval_faces,
        )
        if len(primary_camera_layers) > 1:
            raise RuntimeError(
                f"Global camera UV assignment left overlapping {camera_id} "
                "primary faces."
            )
        primary_layers.extend(primary_camera_layers)
        conflict_fallback_layers.extend(
            _color_non_overlapping_layers(
                conflict_faces_by_camera.get(camera_id, ()),
                cancel_requested,
                progress_interval_faces,
            )
        )

    charts: list[_UvChart] = []
    if synthetic_fallback_faces:
        laid_out_invisible_faces = _layout_invisible_faces(
            synthetic_fallback_faces,
            cancel_requested,
            progress_interval_faces,
        )
        charts.append(_make_chart(len(charts), laid_out_invisible_faces))
    for fallback_layer in conflict_fallback_layers:
        charts.append(
            _make_chart(
                len(charts),
                fallback_layer,
                camera_projection_frames[fallback_layer[0].camera_id],
            )
        )
    for primary_layer in primary_layers:
        charts.append(
            _make_chart(
                len(charts),
                primary_layer,
                camera_projection_frames[primary_layer[0].camera_id],
            )
        )
    if not charts:
        raise ValueError("Camera UV projection could not create any charts.")
    return (
        charts,
        invisible_face_count,
        quality_fallback_face_count,
        conflict_fallback_face_count,
    )


def _layout_invisible_faces(
    faces: Sequence[_ProjectedFace],
    cancel_requested: CancelCallback | None,
    progress_interval_faces: int,
) -> tuple[_ProjectedFace, ...]:
    """Place uncaptured triangles in one deterministic non-overlapping grid."""

    ordered_faces = sorted(faces, key=lambda face: face.global_face_index)
    column_count = max(1, int(math.ceil(math.sqrt(len(ordered_faces)))))
    laid_out_faces: list[_ProjectedFace] = []
    for fallback_index, face in enumerate(ordered_faces):
        if fallback_index % progress_interval_faces == 0:
            _raise_if_cancelled(cancel_requested)
        coordinates = np.asarray(face.coordinates, dtype=float)
        minimum = np.min(coordinates, axis=0)
        maximum = np.max(coordinates, axis=0)
        extent = np.maximum(maximum - minimum, MINIMUM_PROJECTED_EXTENT)
        usable_side = 1.0 - 2.0 * FALLBACK_CELL_MARGIN_RATIO
        uniform_scale = usable_side / max(float(extent[0]), float(extent[1]))
        normalized = (coordinates - minimum) * uniform_scale
        normalized_extent = np.max(normalized, axis=0)
        cell_offset = np.asarray(
            (
                fallback_index % column_count,
                fallback_index // column_count,
            ),
            dtype=float,
        )
        centered_offset = (
            cell_offset
            + FALLBACK_CELL_MARGIN_RATIO
            + (usable_side - normalized_extent) / 2.0
        )
        laid_out_faces.append(
            replace(
                face,
                camera_id=FALLBACK_CHART_CAMERA_ID,
                coordinates=normalized + centered_offset,
                is_leftover=True,
                force_separate_chart=False,
            )
        )
    return tuple(laid_out_faces)


def _color_non_overlapping_layers(
    faces: Sequence[_ProjectedFace],
    cancel_requested: CancelCallback | None,
    progress_interval_faces: int,
) -> list[tuple[_ProjectedFace, ...]]:
    if not faces:
        return []
    all_coordinates = np.vstack([face.coordinates for face in faces])
    coordinate_origin = (
        np.min(all_coordinates, axis=0) + np.max(all_coordinates, axis=0)
    ) / 2.0
    polygons = [
        Polygon(face.coordinates - coordinate_origin) for face in faces
    ]
    tree = STRtree(polygons)
    conflicts: list[set[int]] = [set() for _face in faces]
    candidate_count = 0
    for face_index, polygon in enumerate(polygons):
        for raw_other_index in tree.query(polygon):
            candidate_count += 1
            if candidate_count % progress_interval_faces == 0:
                _raise_if_cancelled(cancel_requested)
            other_index = int(raw_other_index)
            if other_index <= face_index:
                continue
            other_polygon = polygons[other_index]
            if not _polygons_have_positive_overlap(polygon, other_polygon):
                continue
            conflicts[face_index].add(other_index)
            conflicts[other_index].add(face_index)

    colors: dict[int, int] = {}
    order = sorted(
        range(len(faces)),
        key=lambda index: (
            -len(conflicts[index]),
            faces[index].global_face_index,
        ),
    )
    for face_index in order:
        unavailable = {
            colors[neighbor]
            for neighbor in conflicts[face_index]
            if neighbor in colors
        }
        color = 0
        while color in unavailable:
            color += 1
        colors[face_index] = color
    layers: dict[int, list[_ProjectedFace]] = defaultdict(list)
    for face_index, face in enumerate(faces):
        layers[colors[face_index]].append(face)
    return [
        tuple(sorted(layer, key=lambda face: face.global_face_index))
        for _color, layer in sorted(layers.items())
    ]


def _polygons_have_positive_overlap(
    first: Polygon,
    second: Polygon,
) -> bool:
    intersection_area = float(first.intersection(second).area)
    reference_area = max(
        min(float(first.area), float(second.area)),
        np.finfo(float).tiny,
    )
    area_epsilon = max(
        reference_area * INTERSECTION_AREA_EPSILON_RATIO,
        np.finfo(float).eps
        * max(float(first.area), float(second.area))
        * 64.0,
        np.finfo(float).tiny,
    )
    return intersection_area > area_epsilon


def _make_chart(
    chart_index: int,
    faces: Iterable[_ProjectedFace],
    projection_frame: _CameraProjectionFrame | None = None,
) -> _UvChart:
    normalized_faces = tuple(faces)
    coordinates = np.vstack([face.coordinates for face in normalized_faces])
    first = normalized_faces[0]
    return _UvChart(
        chart_index=chart_index,
        camera_id=first.camera_id,
        is_leftover=first.is_leftover,
        faces=normalized_faces,
        minimum=(
            np.min(coordinates, axis=0)
            if projection_frame is None
            else projection_frame.minimum.copy()
        ),
        maximum=(
            np.max(coordinates, axis=0)
            if projection_frame is None
            else projection_frame.maximum.copy()
        ),
    )


# ### Rectangle packing helpers ###
def _pack_charts(
    charts: Sequence[_UvChart],
    options: CameraUvProjectionOptions,
    cancel_requested: CancelCallback | None,
) -> _PackedLayout:
    base_size = options.base_texture_size
    fallback_charts = [chart for chart in charts if chart.is_leftover]
    primary_charts = [chart for chart in charts if not chart.is_leftover]
    placements: dict[int, _PixelRectangle] = {}
    fitted_scales: list[float] = []

    if fallback_charts:
        fallback_region = _PixelRectangle(
            0,
            0,
            FALLBACK_REGION_SIDE_PIXELS,
            FALLBACK_REGION_SIDE_PIXELS,
        )
        fallback_layout = _pack_chart_group(
            fallback_charts,
            (fallback_region,),
            options,
            cancel_requested,
        )
        placements.update(fallback_layout.placements)
        fitted_scales.append(fallback_layout.scale)
        primary_free_rectangles = (
            _PixelRectangle(
                FALLBACK_REGION_SIDE_PIXELS,
                0,
                base_size - FALLBACK_REGION_SIDE_PIXELS,
                base_size,
            ),
            _PixelRectangle(
                0,
                FALLBACK_REGION_SIDE_PIXELS,
                FALLBACK_REGION_SIDE_PIXELS,
                base_size - FALLBACK_REGION_SIDE_PIXELS,
            ),
        )
    else:
        primary_free_rectangles = (_PixelRectangle(0, 0, base_size, base_size),)

    if primary_charts:
        primary_layout = _pack_chart_group(
            primary_charts,
            primary_free_rectangles,
            options,
            cancel_requested,
        )
        placements.update(primary_layout.placements)
        fitted_scales.append(primary_layout.scale)
    return _PackedLayout(
        scale=min(fitted_scales, default=1.0),
        placements=placements,
    )


def _pack_chart_group(
    charts: Sequence[_UvChart],
    free_rectangles: Sequence[_PixelRectangle],
    options: CameraUvProjectionOptions,
    cancel_requested: CancelCallback | None,
) -> _PackedLayout:
    best = _try_pack_chart_group(
        charts,
        free_rectangles,
        options,
        0.0,
        cancel_requested,
    )
    if best is None:
        raise ValueError(
            "The camera-projected UV charts cannot fit on the 512 texture "
            "grid with the required gutters."
        )
    maximum_width = max(chart.width for chart in charts)
    maximum_height = max(chart.height for chart in charts)
    total_chart_area = sum(chart.width * chart.height for chart in charts)
    total_free_area = sum(
        rectangle.width * rectangle.height for rectangle in free_rectangles
    )
    maximum_free_width = max(rectangle.width for rectangle in free_rectangles)
    maximum_free_height = max(rectangle.height for rectangle in free_rectangles)
    high = min(
        maximum_free_width / maximum_width,
        maximum_free_height / maximum_height,
        math.sqrt(total_free_area / max(total_chart_area, 1e-24)),
    ) * 2.0
    while _try_pack_chart_group(
        charts,
        free_rectangles,
        options,
        high,
        cancel_requested,
    ) is not None:
        high *= 2.0
        if high > 1e15:
            break
    low = 0.0
    for _pass_index in range(PACKING_SEARCH_PASSES):
        _raise_if_cancelled(cancel_requested)
        midpoint = (low + high) / 2.0
        candidate = _try_pack_chart_group(
            charts,
            free_rectangles,
            options,
            midpoint,
            cancel_requested,
        )
        if candidate is None:
            high = midpoint
        else:
            low = midpoint
            best = candidate
    if best.scale <= 0.0:
        raise ValueError("Camera UV projection could not allocate usable chart area.")
    return best


def _try_pack_chart_group(
    charts: Sequence[_UvChart],
    free_rectangles: Sequence[_PixelRectangle],
    options: CameraUvProjectionOptions,
    scale: float,
    cancel_requested: CancelCallback | None,
) -> _PackedLayout | None:
    sizes = {
        chart.chart_index: _chart_outer_size(chart, scale, options.padding_pixels)
        for chart in charts
    }
    placements = _maxrects_pack(
        [chart.chart_index for chart in charts],
        sizes,
        free_rectangles,
        cancel_requested,
        options.progress_interval_faces,
    )
    if placements is None:
        return None
    return _PackedLayout(scale=scale, placements=placements)


def _chart_outer_size(
    chart: _UvChart,
    scale: float,
    padding: int,
) -> tuple[int, int]:
    content_width = max(1, int(math.ceil(chart.width * scale)))
    content_height = max(1, int(math.ceil(chart.height * scale)))
    return content_width + 2 * padding, content_height + 2 * padding


def _maxrects_pack(
    chart_ids: Sequence[int],
    sizes: dict[int, tuple[int, int]],
    initial_free_rectangles: Sequence[_PixelRectangle],
    cancel_requested: CancelCallback | None,
    progress_interval_faces: int,
) -> dict[int, _PixelRectangle] | None:
    free_rectangles = list(initial_free_rectangles)
    placements: dict[int, _PixelRectangle] = {}
    ordered_ids = sorted(
        chart_ids,
        key=lambda chart_id: (
            -(sizes[chart_id][0] * sizes[chart_id][1]),
            -max(sizes[chart_id]),
            chart_id,
        ),
    )
    for item_index, chart_id in enumerate(ordered_ids):
        if item_index % progress_interval_faces == 0:
            _raise_if_cancelled(cancel_requested)
        width, height = sizes[chart_id]
        candidates: list[tuple[tuple[int, ...], _PixelRectangle]] = []
        for free_rectangle in free_rectangles:
            if width > free_rectangle.width or height > free_rectangle.height:
                continue
            horizontal_remainder = free_rectangle.width - width
            vertical_remainder = free_rectangle.height - height
            placement = _PixelRectangle(
                free_rectangle.x,
                free_rectangle.y,
                width,
                height,
            )
            score = (
                min(horizontal_remainder, vertical_remainder),
                max(horizontal_remainder, vertical_remainder),
                placement.y,
                placement.x,
            )
            candidates.append((score, placement))
        if not candidates:
            return None
        _score, placement = min(candidates, key=lambda candidate: candidate[0])
        placements[chart_id] = placement
        free_rectangles = _split_free_rectangles(free_rectangles, placement)
    return placements


def _split_free_rectangles(
    free_rectangles: Sequence[_PixelRectangle],
    used: _PixelRectangle,
) -> list[_PixelRectangle]:
    split_rectangles: list[_PixelRectangle] = []
    for free in free_rectangles:
        if not _rectangles_overlap(free, used):
            split_rectangles.append(free)
            continue
        if used.x > free.x:
            split_rectangles.append(
                _PixelRectangle(free.x, free.y, used.x - free.x, free.height)
            )
        if used.right < free.right:
            split_rectangles.append(
                _PixelRectangle(
                    used.right,
                    free.y,
                    free.right - used.right,
                    free.height,
                )
            )
        if used.y > free.y:
            split_rectangles.append(
                _PixelRectangle(free.x, free.y, free.width, used.y - free.y)
            )
        if used.top < free.top:
            split_rectangles.append(
                _PixelRectangle(
                    free.x,
                    used.top,
                    free.width,
                    free.top - used.top,
                )
            )
    positive = [
        rectangle
        for rectangle in split_rectangles
        if rectangle.width > 0 and rectangle.height > 0
    ]
    return [
        rectangle
        for index, rectangle in enumerate(positive)
        if not any(
            index != other_index
            and _rectangle_contains(other, rectangle)
            for other_index, other in enumerate(positive)
        )
    ]


def _rectangles_overlap(first: _PixelRectangle, second: _PixelRectangle) -> bool:
    return not (
        first.right <= second.x
        or second.right <= first.x
        or first.top <= second.y
        or second.top <= first.y
    )


def _rectangle_contains(outer: _PixelRectangle, inner: _PixelRectangle) -> bool:
    return (
        outer.x <= inner.x
        and outer.y <= inner.y
        and outer.right >= inner.right
        and outer.top >= inner.top
    )


# ### Mesh rebuilding helpers ###
def _build_output_scene(
    instances: Sequence[_MeshInstance],
    projected_faces: Sequence[_ProjectedFace],
    charts: Sequence[_UvChart],
    packed_layout: _PackedLayout,
    options: CameraUvProjectionOptions,
    topology: _MeshTopology,
    cancel_requested: CancelCallback | None,
) -> tuple[trimesh.Scene, int]:
    chart_by_face = {
        face.global_face_index: chart
        for chart in charts
        for face in chart.faces
    }
    projected_by_face = {
        face.global_face_index: face
        for face in projected_faces
    }
    output_scene = trimesh.Scene()
    used_geometry_names: set[str] = set()
    used_node_names: set[str] = set()
    output_vertex_count = 0
    for instance_index, instance in enumerate(instances):
        _raise_if_cancelled(cancel_requested)
        rebuilt = _rebuild_instance_mesh(
            instance,
            projected_by_face,
            chart_by_face,
            packed_layout,
            options,
            topology.welded_vertex_ids_by_instance[instance_index],
            cancel_requested,
        )
        output_vertex_count += len(rebuilt.vertices)
        geometry_name = _make_unique_name(
            instance.geometry_name,
            used_geometry_names,
            f"camera_uv_geometry_{instance_index}",
        )
        node_name = _make_unique_name(
            instance.node_name,
            used_node_names,
            f"camera_uv_node_{instance_index}",
        )
        output_scene.add_geometry(
            rebuilt,
            geom_name=geometry_name,
            node_name=node_name,
            transform=instance.transform,
        )
    if not output_scene.geometry:
        raise ValueError("Camera UV projection produced an empty scene.")
    return output_scene, output_vertex_count


def _rebuild_instance_mesh(
    instance: _MeshInstance,
    projected_by_face: dict[int, _ProjectedFace],
    chart_by_face: dict[int, _UvChart],
    packed_layout: _PackedLayout,
    options: CameraUvProjectionOptions,
    welded_vertex_ids: np.ndarray,
    cancel_requested: CancelCallback | None,
) -> trimesh.Trimesh:
    source_mesh = instance.mesh
    source_vertices = np.asarray(source_mesh.vertices, dtype=float)
    source_faces = np.asarray(source_mesh.faces, dtype=np.int64)
    source_normals = instance.vertex_normals
    new_vertices: list[np.ndarray] = []
    new_normals: list[np.ndarray] = []
    new_uvs: list[np.ndarray] = []
    new_faces: list[tuple[int, int, int]] = []
    face_material_ids = _get_face_material_ids(source_mesh)
    compatible_vertices: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for local_face_index, source_face in enumerate(source_faces):
        if local_face_index % options.progress_interval_faces == 0:
            _raise_if_cancelled(cancel_requested)
        global_face_index = instance.first_face_index + local_face_index
        projected_face = projected_by_face[global_face_index]
        chart = chart_by_face[global_face_index]
        placement = packed_layout.placements[chart.chart_index]
        material_id = int(face_material_ids[local_face_index])
        face_indices: list[int] = []
        for corner_index, source_vertex_index in enumerate(source_face):
            source_vertex_index = int(source_vertex_index)
            uv = _map_chart_coordinate_to_uv(
                projected_face.coordinates[corner_index],
                chart,
                placement,
                options,
            )
            seam_key = (
                int(welded_vertex_ids[source_vertex_index]),
                chart.chart_index,
                material_id,
            )
            new_vertex_index = _find_compatible_output_vertex(
                compatible_vertices.get(seam_key, ()),
                source_vertices[source_vertex_index],
                source_normals[source_vertex_index],
                uv,
                new_vertices,
                new_normals,
                new_uvs,
            )
            if new_vertex_index is None:
                new_vertex_index = len(new_vertices)
                compatible_vertices[seam_key].append(new_vertex_index)
                new_vertices.append(source_vertices[source_vertex_index].copy())
                new_normals.append(source_normals[source_vertex_index].copy())
                new_uvs.append(np.asarray(uv, dtype=float).copy())
            face_indices.append(new_vertex_index)
        new_faces.append(tuple(face_indices))
    source_visual = source_mesh.visual
    material = (
        copy.deepcopy(source_visual.material)
        if isinstance(source_visual, TextureVisuals)
        else PBRMaterial(
            name="Camera projected UV material",
            baseColorFactor=(255, 255, 255, 255),
            metallicFactor=0.0,
            roughnessFactor=1.0,
        )
    )
    face_materials = getattr(source_visual, "face_materials", None)
    visual = TextureVisuals(
        uv=np.asarray(new_uvs, dtype=float),
        material=material,
        face_materials=(
            None
            if face_materials is None
            else np.asarray(face_materials, dtype=np.int64).copy()
        ),
    )
    rebuilt = trimesh.Trimesh(
        vertices=np.asarray(new_vertices, dtype=float),
        faces=np.asarray(new_faces, dtype=np.int64),
        vertex_normals=np.asarray(new_normals, dtype=float),
        visual=visual,
        process=False,
        metadata=copy.deepcopy(source_mesh.metadata),
    )
    rebuilt.units = source_mesh.units
    return rebuilt


def _find_compatible_output_vertex(
    candidate_indices: Iterable[int],
    vertex: np.ndarray,
    normal: np.ndarray,
    uv: np.ndarray,
    output_vertices: Sequence[np.ndarray],
    output_normals: Sequence[np.ndarray],
    output_uvs: Sequence[np.ndarray],
) -> int | None:
    """Find a shape-, shading-, and UV-identical output vertex."""

    for candidate_index in candidate_indices:
        if not np.array_equal(output_vertices[candidate_index], vertex):
            continue
        if not np.allclose(
            output_normals[candidate_index],
            normal,
            rtol=0.0,
            atol=OUTPUT_NORMAL_WELD_TOLERANCE,
        ):
            continue
        if not np.allclose(
            output_uvs[candidate_index],
            uv,
            rtol=0.0,
            atol=OUTPUT_UV_WELD_TOLERANCE,
        ):
            continue
        return candidate_index
    return None


def _get_source_vertex_normals(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return stable normals without requiring trimesh's optional SciPy path."""

    mesh_cache = getattr(mesh, "_cache", None)
    cached_normals = getattr(mesh_cache, "cache", {}).get("vertex_normals")
    if cached_normals is not None:
        normalized_cached = np.asarray(cached_normals, dtype=float)
        cached_lengths = np.linalg.norm(normalized_cached, axis=1)
        if (
            normalized_cached.shape == (len(mesh.vertices), 3)
            and np.all(np.isfinite(normalized_cached))
            and np.all(cached_lengths > np.finfo(float).eps)
        ):
            return normalized_cached.copy()
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    weighted_face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    vertex_normals = np.zeros((len(vertices), 3), dtype=float)
    for corner_index in range(3):
        np.add.at(
            vertex_normals,
            faces[:, corner_index],
            weighted_face_normals,
        )
    lengths = np.linalg.norm(vertex_normals, axis=1)
    nonzero = lengths > np.finfo(float).eps
    vertex_normals[nonzero] /= lengths[nonzero, np.newaxis]
    vertex_normals[~nonzero] = (0.0, 0.0, 1.0)
    return vertex_normals


def _map_chart_coordinate_to_uv(
    coordinate: np.ndarray,
    chart: _UvChart,
    placement: _PixelRectangle,
    options: CameraUvProjectionOptions,
) -> np.ndarray:
    padding = options.padding_pixels
    content_width = placement.width - 2 * padding
    content_height = placement.height - 2 * padding
    uniform_scale = min(
        content_width / chart.width,
        content_height / chart.height,
    )
    used_width = chart.width * uniform_scale
    used_height = chart.height * uniform_scale
    origin_x = placement.x + padding + (content_width - used_width) / 2.0
    origin_y = placement.y + padding + (content_height - used_height) / 2.0
    pixel_x = (
        origin_x
        + (float(coordinate[0]) - chart.minimum[0]) * uniform_scale
    )
    pixel_y = (
        origin_y
        + (float(coordinate[1]) - chart.minimum[1]) * uniform_scale
    )
    return np.asarray(
        (
            pixel_x / options.base_texture_size,
            pixel_y / options.base_texture_size,
        ),
        dtype=float,
    )


def _make_unique_name(
    requested_name: str,
    used_names: set[str],
    fallback: str,
) -> str:
    base_name = str(requested_name).strip() or fallback
    candidate = base_name
    suffix = 2
    while candidate in used_names:
        candidate = f"{base_name}_{suffix}"
        suffix += 1
    used_names.add(candidate)
    return candidate


# ### Export validation helpers ###
def _export_scene(scene: trimesh.Scene) -> bytes:
    try:
        return bytes(scene.export(file_type="glb"))
    except Exception as error:
        raise ValueError("The camera UV GLB could not be exported.") from error


def _validate_exported_uvs(payload: bytes) -> None:
    scene = _load_glb_scene(payload)
    found_mesh = False
    for geometry in scene.geometry.values():
        if not isinstance(geometry, trimesh.Trimesh):
            continue
        found_mesh = True
        uv = getattr(geometry.visual, "uv", None)
        if uv is None:
            raise ValueError("The exported camera UV GLB lost its UV coordinates.")
        normalized_uv = np.asarray(uv, dtype=float)
        if normalized_uv.shape != (len(geometry.vertices), 2):
            raise ValueError("The exported camera UV coordinates are incomplete.")
        if not np.all(np.isfinite(normalized_uv)):
            raise ValueError("The exported camera UV coordinates are invalid.")
        if np.any(normalized_uv < -1e-9) or np.any(normalized_uv > 1.0 + 1e-9):
            raise ValueError("The exported camera UV coordinates leave the map.")
    if not found_mesh:
        raise ValueError("The exported camera UV GLB contains no mesh.")


# ### Cancellation and progress helpers ###
def _raise_if_cancelled(cancel_requested: CancelCallback | None) -> None:
    if cancel_requested is not None and cancel_requested():
        raise CameraUvProjectionCancelled("Camera UV projection was cancelled.")


def _report_progress(
    progress_callback: ProgressCallback | None,
    stage: str,
    completed_face_count: int,
    total_face_count: int,
    camera_id: str | None = None,
) -> None:
    if progress_callback is None:
        return
    progress_callback(
        CameraUvProjectionProgress(
            stage=stage,
            completed_face_count=int(completed_face_count),
            total_face_count=int(total_face_count),
            camera_id=camera_id,
        )
    )
