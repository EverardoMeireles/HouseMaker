# ### Imports ###
from __future__ import annotations

import math
import operator
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from io import BytesIO

import cv2
import numpy as np
import trimesh

from housemaker.glb import (
    GLTF_Y_UP_TO_Z_UP_TRANSFORM,
    GeneratedModel,
    import_generated_glb,
)


# ### Camera constants ###
CAMERA_ID_POS_X = "pos_x"
CAMERA_ID_NEG_X = "neg_x"
CAMERA_ID_POS_Y = "pos_y"
CAMERA_ID_NEG_Y = "neg_y"
CAMERA_ID_TOP = "top"
CAMERA_ID_BOTTOM = "bottom"

CAMERA_OPTIONS: tuple[tuple[str, str], ...] = (
    (CAMERA_ID_POS_X, "+X"),
    (CAMERA_ID_NEG_X, "-X"),
    (CAMERA_ID_POS_Y, "+Y"),
    (CAMERA_ID_NEG_Y, "-Y"),
    (CAMERA_ID_TOP, "Top (+Z)"),
    (CAMERA_ID_BOTTOM, "Bottom (-Z)"),
)
ALL_CAMERA_IDS = tuple(camera_id for camera_id, _label in CAMERA_OPTIONS)


# ### Processing constants ###
DEFAULT_CAPTURE_IMAGE_SIZE = 256
MIN_CAPTURE_IMAGE_SIZE = 32
MAX_CAPTURE_IMAGE_SIZE = 1024
DEFAULT_MAX_FACE_COUNT = 200_000
MAX_SUPPORTED_FACE_COUNT = (1 << 24) - 1
DEFAULT_PROGRESS_INTERVAL_FACES = 128
DEFAULT_MINIMUM_VISIBLE_FRACTION = 0.05
DEFAULT_MINIMUM_PROJECTED_SAMPLES = 4
CAMERA_FRAME_MARGIN_RATIO = 0.05
MINIMUM_CAMERA_EXTENT = 1e-9
DEPTH_EPSILON_RATIO = 1e-9
MINIMUM_DEPTH_EPSILON = 1e-12
RASTER_AREA_EPSILON = 1e-18
BARYCENTRIC_EPSILON = 1e-9
STACK_NORMAL_ALIGNMENT_COSINE = 0.995
STACK_MINIMUM_OCCLUDED_FRACTION = 0.95
STACK_MAXIMUM_DEPTH_GAP_RATIO = 0.005


# ### Callback types ###
CancelCallback = Callable[[], bool]
ProgressCallback = Callable[["UnusedFaceRemovalProgress"], None]


# ### Public data models ###
@dataclass(frozen=True)
class UnusedFaceRemovalOptions:
    """Bounds and selected views for one removal operation."""

    enabled_camera_ids: tuple[str, ...] = ALL_CAMERA_IDS
    image_size: int = DEFAULT_CAPTURE_IMAGE_SIZE
    max_face_count: int = DEFAULT_MAX_FACE_COUNT
    progress_interval_faces: int = DEFAULT_PROGRESS_INTERVAL_FACES
    minimum_visible_fraction: float = DEFAULT_MINIMUM_VISIBLE_FRACTION
    minimum_projected_samples: int = DEFAULT_MINIMUM_PROJECTED_SAMPLES

    def __post_init__(self) -> None:
        normalized_camera_ids = _normalize_camera_ids(self.enabled_camera_ids)
        object.__setattr__(self, "enabled_camera_ids", normalized_camera_ids)
        minimum_visible_fraction = _normalize_minimum_visible_fraction(
            self.minimum_visible_fraction
        )
        minimum_projected_samples = _normalize_minimum_projected_samples(
            self.minimum_projected_samples
        )
        object.__setattr__(
            self,
            "minimum_visible_fraction",
            minimum_visible_fraction,
        )
        object.__setattr__(
            self,
            "minimum_projected_samples",
            minimum_projected_samples,
        )
        _validate_processing_bounds(
            image_size=self.image_size,
            max_face_count=self.max_face_count,
            progress_interval_faces=self.progress_interval_faces,
        )


@dataclass(frozen=True)
class UnusedFaceRemovalProgress:
    """One throttled progress notification suitable for a worker signal."""

    stage: str
    completed_face_count: int
    total_face_count: int
    camera_id: str | None = None


@dataclass(frozen=True)
class UnusedFaceRemovalResult:
    """Post-processed model and auditable face counts."""

    model: GeneratedModel
    enabled_camera_ids: tuple[str, ...]
    original_face_count: int
    retained_face_count: int
    removed_face_count: int
    protected_face_count: int
    visibility_removed_face_count: int = 0
    stacked_face_removed_count: int = 0

    @property
    def glb_bytes(self) -> bytes:
        return self.model.glb_bytes


class UnusedFaceRemovalCancelled(RuntimeError):
    """Raised when a caller cancels a removal operation."""


# ### Public camera data models ###
@dataclass(frozen=True)
class FixedCameraView:
    """Canonical orthographic axes shared by model post-processing steps."""

    camera_id: str
    depth_axis: tuple[float, float, float]
    horizontal_axis: tuple[float, float, float]
    vertical_axis: tuple[float, float, float]


# ### Internal data models ###
@dataclass(frozen=True)
class _CameraDefinition:
    camera_id: str
    depth_axis: tuple[float, float, float]
    horizontal_axis: tuple[float, float, float]
    vertical_axis: tuple[float, float, float]


@dataclass
class _MeshInstance:
    node_name: str
    geometry_name: str
    mesh: trimesh.Trimesh
    transform: np.ndarray
    first_face_index: int

    @property
    def face_count(self) -> int:
        return len(self.mesh.faces)


@dataclass(frozen=True)
class _CameraCapture:
    camera_id: str
    projected_sample_counts: np.ndarray
    visible_sample_counts: np.ndarray
    top_depth: np.ndarray
    top_faces: np.ndarray


# ### Camera definitions ###
_CAMERA_DEFINITIONS = {
    CAMERA_ID_POS_X: _CameraDefinition(
        camera_id=CAMERA_ID_POS_X,
        depth_axis=(1.0, 0.0, 0.0),
        horizontal_axis=(0.0, 1.0, 0.0),
        vertical_axis=(0.0, 0.0, 1.0),
    ),
    CAMERA_ID_NEG_X: _CameraDefinition(
        camera_id=CAMERA_ID_NEG_X,
        depth_axis=(-1.0, 0.0, 0.0),
        horizontal_axis=(0.0, -1.0, 0.0),
        vertical_axis=(0.0, 0.0, 1.0),
    ),
    CAMERA_ID_POS_Y: _CameraDefinition(
        camera_id=CAMERA_ID_POS_Y,
        depth_axis=(0.0, 1.0, 0.0),
        horizontal_axis=(-1.0, 0.0, 0.0),
        vertical_axis=(0.0, 0.0, 1.0),
    ),
    CAMERA_ID_NEG_Y: _CameraDefinition(
        camera_id=CAMERA_ID_NEG_Y,
        depth_axis=(0.0, -1.0, 0.0),
        horizontal_axis=(1.0, 0.0, 0.0),
        vertical_axis=(0.0, 0.0, 1.0),
    ),
    CAMERA_ID_TOP: _CameraDefinition(
        camera_id=CAMERA_ID_TOP,
        depth_axis=(0.0, 0.0, 1.0),
        horizontal_axis=(1.0, 0.0, 0.0),
        vertical_axis=(0.0, 1.0, 0.0),
    ),
    CAMERA_ID_BOTTOM: _CameraDefinition(
        camera_id=CAMERA_ID_BOTTOM,
        depth_axis=(0.0, 0.0, -1.0),
        horizontal_axis=(1.0, 0.0, 0.0),
        vertical_axis=(0.0, -1.0, 0.0),
    ),
}


# ### Public camera helpers ###
def get_fixed_camera_view(camera_id: str) -> FixedCameraView:
    """Return a copy of one canonical unused-face camera definition."""

    normalized_id = str(camera_id)
    definition = _CAMERA_DEFINITIONS.get(normalized_id)
    if definition is None:
        raise ValueError(f"Unknown unused-face camera ID: {normalized_id!r}.")
    return FixedCameraView(
        camera_id=definition.camera_id,
        depth_axis=definition.depth_axis,
        horizontal_axis=definition.horizontal_axis,
        vertical_axis=definition.vertical_axis,
    )


def capture_visible_face_indices(
    vertices: np.ndarray,
    faces: np.ndarray,
    camera_id: str,
    *,
    image_size: int = DEFAULT_CAPTURE_IMAGE_SIZE,
    cancel_requested: CancelCallback | None = None,
    progress_interval_faces: int = DEFAULT_PROGRESS_INTERVAL_FACES,
) -> frozenset[int]:
    """Depth-rasterize one fixed view and return faces owning any top pixel."""

    normalized_vertices = np.asarray(vertices, dtype=float)
    normalized_faces = np.asarray(faces, dtype=np.int64)
    if normalized_vertices.ndim != 2 or normalized_vertices.shape[1:] != (3,):
        raise ValueError("Camera capture vertices must have shape (n, 3).")
    if normalized_faces.ndim != 2 or normalized_faces.shape[1:] != (3,):
        raise ValueError("Camera capture faces must have shape (n, 3).")
    if len(normalized_vertices) == 0 or len(normalized_faces) == 0:
        return frozenset()
    if not np.all(np.isfinite(normalized_vertices)):
        raise ValueError("Camera capture vertices must contain finite coordinates.")
    if (
        np.any(normalized_faces < 0)
        or np.any(normalized_faces >= len(normalized_vertices))
    ):
        raise ValueError("Camera capture faces contain an invalid vertex index.")
    normalized_image_size = int(image_size)
    if not MIN_CAPTURE_IMAGE_SIZE <= normalized_image_size <= MAX_CAPTURE_IMAGE_SIZE:
        raise ValueError(
            "Camera capture size must be between "
            f"{MIN_CAPTURE_IMAGE_SIZE} and {MAX_CAPTURE_IMAGE_SIZE} pixels."
        )
    normalized_progress_interval = int(progress_interval_faces)
    if normalized_progress_interval < 1:
        raise ValueError("Progress interval must contain at least one face.")
    definition = _CAMERA_DEFINITIONS.get(str(camera_id))
    if definition is None:
        raise ValueError(f"Unknown unused-face camera ID: {str(camera_id)!r}.")
    _raise_if_cancelled(cancel_requested)
    capture = _capture_camera(
        vertices=normalized_vertices,
        faces=normalized_faces,
        definition=definition,
        image_size=normalized_image_size,
        cancel_requested=cancel_requested,
        progress_interval_faces=normalized_progress_interval,
    )
    return frozenset(
        int(face_index)
        for face_index in np.flatnonzero(capture.visible_sample_counts > 0)
    )


# ### Public processing API ###
def remove_unused_faces(
    model: GeneratedModel,
    *,
    options: UnusedFaceRemovalOptions | None = None,
    cancel_requested: CancelCallback | None = None,
    progress_callback: ProgressCallback | None = None,
) -> UnusedFaceRemovalResult:
    """Remove faces invisible to every selected orthographic camera."""

    if not isinstance(model, GeneratedModel):
        raise TypeError("Unused-face removal requires a GeneratedModel.")
    return remove_unused_faces_from_glb(
        model.glb_bytes,
        options=options,
        cancel_requested=cancel_requested,
        progress_callback=progress_callback,
    )


def remove_unused_faces_from_glb(
    glb_bytes: bytes,
    *,
    options: UnusedFaceRemovalOptions | None = None,
    cancel_requested: CancelCallback | None = None,
    progress_callback: ProgressCallback | None = None,
) -> UnusedFaceRemovalResult:
    """Return a self-contained GLB after conservative six-view face removal.

    Each capture measures exact projected pixel-center coverage and top-depth
    ownership rather than material color. A face must have meaningful coverage
    and meet the configured visible fraction in at least one selected view. A
    separate, conservative pass removes same-facing, near-coplanar wafer layers
    that are almost completely covered by a parallel face.
    """

    normalized_options = options or UnusedFaceRemovalOptions()
    _raise_if_cancelled(cancel_requested)
    payload = bytes(glb_bytes)
    if not payload:
        raise ValueError("The generated GLB is empty.")
    scene = _load_glb_scene(payload)
    instances, vertices, faces = _collect_scene_geometry(scene)
    total_face_count = len(faces)
    if total_face_count == 0:
        raise ValueError("The generated GLB contains no triangle faces.")
    if total_face_count > normalized_options.max_face_count:
        raise ValueError(
            f"The generated GLB has {total_face_count} faces; the configured "
            f"unused-face limit is {normalized_options.max_face_count}."
        )

    _report_progress(
        progress_callback,
        stage="capturing",
        completed_face_count=0,
        total_face_count=total_face_count,
    )
    captures: list[_CameraCapture] = []
    for camera_id in normalized_options.enabled_camera_ids:
        _raise_if_cancelled(cancel_requested)
        captures.append(
            _capture_camera(
                vertices=vertices,
                faces=faces,
                definition=_CAMERA_DEFINITIONS[camera_id],
                image_size=normalized_options.image_size,
                cancel_requested=cancel_requested,
                progress_interval_faces=(
                    normalized_options.progress_interval_faces
                ),
            )
        )
        _report_progress(
            progress_callback,
            stage="capturing",
            completed_face_count=total_face_count,
            total_face_count=total_face_count,
            camera_id=camera_id,
        )

    _report_progress(
        progress_callback,
        stage="checking",
        completed_face_count=0,
        total_face_count=total_face_count,
    )
    visibility_keep_faces = _build_visibility_keep_mask(
        captures,
        minimum_visible_fraction=(
            normalized_options.minimum_visible_fraction
        ),
        minimum_projected_samples=(
            normalized_options.minimum_projected_samples
        ),
    )
    stacked_faces = _find_stacked_faces(
        vertices=vertices,
        faces=faces,
        captures=captures,
        image_size=normalized_options.image_size,
        minimum_projected_samples=(
            normalized_options.minimum_projected_samples
        ),
        cancel_requested=cancel_requested,
        progress_interval_faces=(
            normalized_options.progress_interval_faces
        ),
    )
    keep_faces = visibility_keep_faces & ~stacked_faces
    _report_progress(
        progress_callback,
        stage="checking",
        completed_face_count=total_face_count,
        total_face_count=total_face_count,
    )
    retained_face_count = int(np.count_nonzero(keep_faces))
    visibility_removed_face_count = int(
        np.count_nonzero(~visibility_keep_faces)
    )
    stacked_face_removed_count = int(
        np.count_nonzero(visibility_keep_faces & stacked_faces)
    )
    if retained_face_count == 0:
        raise ValueError(
            "Unused-face removal would remove every face; increase the "
            "capture resolution, lower the minimum visible percentage, or "
            "select additional cameras."
        )
    _raise_if_cancelled(cancel_requested)
    _report_progress(
        progress_callback,
        stage="exporting",
        completed_face_count=total_face_count,
        total_face_count=total_face_count,
    )

    if retained_face_count == total_face_count:
        processed_model = import_generated_glb(payload)
    else:
        processed_glb = _export_filtered_scene(instances, keep_faces)
        processed_model = import_generated_glb(processed_glb)
    _report_progress(
        progress_callback,
        stage="complete",
        completed_face_count=total_face_count,
        total_face_count=total_face_count,
    )
    return UnusedFaceRemovalResult(
        model=processed_model,
        enabled_camera_ids=normalized_options.enabled_camera_ids,
        original_face_count=total_face_count,
        retained_face_count=retained_face_count,
        removed_face_count=total_face_count - retained_face_count,
        protected_face_count=retained_face_count,
        visibility_removed_face_count=visibility_removed_face_count,
        stacked_face_removed_count=stacked_face_removed_count,
    )


# ### Scene loading helpers ###
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


def _collect_scene_geometry(
    scene: trimesh.Scene,
) -> tuple[list[_MeshInstance], np.ndarray, np.ndarray]:
    instances: list[_MeshInstance] = []
    world_vertices: list[np.ndarray] = []
    global_faces: list[np.ndarray] = []
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
        node_transform = np.asarray(transform, dtype=float)
        if node_transform.shape != (4, 4) or not np.all(np.isfinite(node_transform)):
            raise ValueError("The generated GLB contains an invalid node transform.")
        if not np.all(np.isfinite(local_vertices)):
            raise ValueError("The generated GLB contains invalid vertex coordinates.")
        transformed_vertices = trimesh.transform_points(
            local_vertices,
            node_transform,
        )
        z_up_vertices = trimesh.transform_points(
            transformed_vertices,
            GLTF_Y_UP_TO_Z_UP_TRANSFORM,
        )
        geometry_faces = np.asarray(geometry.faces, dtype=np.int64)
        world_vertices.append(z_up_vertices)
        global_faces.append(geometry_faces + vertex_offset)
        instances.append(
            _MeshInstance(
                node_name=str(node_name),
                geometry_name=str(geometry_name),
                mesh=geometry.copy(),
                transform=node_transform.copy(),
                first_face_index=face_offset,
            )
        )
        vertex_offset += len(local_vertices)
        face_offset += len(geometry_faces)
    if not world_vertices or not global_faces:
        return (
            instances,
            np.empty((0, 3), dtype=float),
            np.empty((0, 3), dtype=np.int64),
        )
    return instances, np.vstack(world_vertices), np.vstack(global_faces)


# ### Software capture helpers ###
def _capture_camera(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    definition: _CameraDefinition,
    image_size: int,
    cancel_requested: CancelCallback | None,
    progress_interval_faces: int,
) -> _CameraCapture:
    screen_vertices, vertex_depths = _project_vertices(
        vertices,
        definition,
        image_size,
    )
    top_depth = np.full((image_size, image_size), -np.inf, dtype=np.float64)
    top_faces = np.full((image_size, image_size), -1, dtype=np.int32)
    projected_sample_counts = np.zeros(len(faces), dtype=np.int64)
    visible_sample_counts = np.zeros(len(faces), dtype=np.int64)
    bounds_diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    depth_epsilon = max(
        bounds_diagonal * DEPTH_EPSILON_RATIO,
        MINIMUM_DEPTH_EPSILON,
    )
    for face_index, face in enumerate(faces):
        if face_index % progress_interval_faces == 0:
            _raise_if_cancelled(cancel_requested)
        screen_points = screen_vertices[face]
        rows, columns, sample_depths = _rasterize_face_samples(
            screen_points=screen_points,
            depths=vertex_depths[face],
            image_shape=top_faces.shape,
        )
        projected_sample_counts[face_index] = len(rows)
        if not len(rows):
            continue
        current_depths = top_depth[rows, columns]
        becomes_top = sample_depths > current_depths + depth_epsilon
        if not np.any(becomes_top):
            continue
        selected_rows = rows[becomes_top]
        selected_columns = columns[becomes_top]
        previous_owners = top_faces[selected_rows, selected_columns]
        replaced_owners, replaced_counts = np.unique(
            previous_owners[previous_owners >= 0],
            return_counts=True,
        )
        if len(replaced_owners):
            visible_sample_counts[replaced_owners] -= replaced_counts
        visible_sample_counts[face_index] += len(selected_rows)
        top_depth[selected_rows, selected_columns] = sample_depths[becomes_top]
        top_faces[selected_rows, selected_columns] = face_index
    return _CameraCapture(
        camera_id=definition.camera_id,
        projected_sample_counts=projected_sample_counts,
        visible_sample_counts=visible_sample_counts,
        top_depth=top_depth,
        top_faces=top_faces,
    )


def _project_vertices(
    vertices: np.ndarray,
    definition: _CameraDefinition,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    horizontal_axis = np.asarray(definition.horizontal_axis, dtype=float)
    vertical_axis = np.asarray(definition.vertical_axis, dtype=float)
    depth_axis = np.asarray(definition.depth_axis, dtype=float)
    horizontal = vertices @ horizontal_axis
    vertical = vertices @ vertical_axis
    depths = vertices @ depth_axis
    horizontal_center = (float(np.min(horizontal)) + float(np.max(horizontal))) / 2.0
    vertical_center = (float(np.min(vertical)) + float(np.max(vertical))) / 2.0
    horizontal_extent = float(np.ptp(horizontal)) / 2.0
    vertical_extent = float(np.ptp(vertical)) / 2.0
    extent = max(horizontal_extent, vertical_extent, MINIMUM_CAMERA_EXTENT)
    extent *= 1.0 + CAMERA_FRAME_MARGIN_RATIO
    scale = (image_size - 1.0) / (2.0 * extent)
    screen_vertices = np.column_stack(
        (
            (horizontal - horizontal_center) * scale + (image_size - 1.0) / 2.0,
            (vertical_center - vertical) * scale + (image_size - 1.0) / 2.0,
        )
    )
    return screen_vertices, depths


# ### Exact raster helpers ###
def _rasterize_face_samples(
    *,
    screen_points: np.ndarray,
    depths: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return only pixel centers inside the original projected triangle."""

    image_height, image_width = image_shape
    minimum_x = max(0, int(np.floor(np.min(screen_points[:, 0]))))
    maximum_x = min(image_width - 1, int(np.ceil(np.max(screen_points[:, 0]))))
    minimum_y = max(0, int(np.floor(np.min(screen_points[:, 1]))))
    maximum_y = min(image_height - 1, int(np.ceil(np.max(screen_points[:, 1]))))
    if maximum_x < minimum_x or maximum_y < minimum_y:
        return _empty_face_samples()

    denominator = _cross_2d(
        screen_points[1] - screen_points[0],
        screen_points[2] - screen_points[0],
    )
    if abs(denominator) <= RASTER_AREA_EPSILON:
        return _empty_face_samples()

    mask = np.zeros(
        (maximum_y - minimum_y + 1, maximum_x - minimum_x + 1),
        dtype=np.uint8,
    )
    local_polygon = np.rint(
        screen_points - np.array([minimum_x, minimum_y], dtype=float)
    ).astype(np.int32)
    cv2.fillConvexPoly(mask, local_polygon, 255, lineType=cv2.LINE_8)
    mask = cv2.dilate(mask, None, iterations=1)
    local_rows, local_columns = np.nonzero(mask)
    if not len(local_columns):
        return _empty_face_samples()
    rows = local_rows.astype(np.int64) + minimum_y
    columns = local_columns.astype(np.int64) + minimum_x
    sample_points = np.column_stack(
        (columns.astype(float) + 0.5, rows.astype(float) + 0.5)
    )
    barycentric = _triangle_barycentric_weights(sample_points, screen_points)
    inside = np.all(barycentric >= -BARYCENTRIC_EPSILON, axis=1)
    inside &= np.all(barycentric <= 1.0 + BARYCENTRIC_EPSILON, axis=1)
    if not np.any(inside):
        return _empty_face_samples()
    return (
        rows[inside],
        columns[inside],
        barycentric[inside] @ depths,
    )


def _empty_face_samples() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=float),
    )


def _triangle_barycentric_weights(
    points: np.ndarray,
    triangle: np.ndarray,
) -> np.ndarray:
    first = triangle[0]
    second = triangle[1]
    third = triangle[2]
    denominator = _cross_2d(second - first, third - first)
    first_weights = (
        (second[0] - points[:, 0]) * (third[1] - points[:, 1])
        - (second[1] - points[:, 1]) * (third[0] - points[:, 0])
    ) / denominator
    second_weights = (
        (third[0] - points[:, 0]) * (first[1] - points[:, 1])
        - (third[1] - points[:, 1]) * (first[0] - points[:, 0])
    ) / denominator
    third_weights = 1.0 - first_weights - second_weights
    return np.column_stack((first_weights, second_weights, third_weights))


def _cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


# ### Visibility classification helpers ###
def _build_visibility_keep_mask(
    captures: Sequence[_CameraCapture],
    *,
    minimum_visible_fraction: float,
    minimum_projected_samples: int,
) -> np.ndarray:
    """Keep measured visible faces and protect genuine subpixel projections."""

    if not captures:
        return np.empty(0, dtype=bool)
    face_count = len(captures[0].projected_sample_counts)
    has_meaningful_projection = np.zeros(face_count, dtype=bool)
    has_visible_sample = np.zeros(face_count, dtype=bool)
    qualifies_as_visible = np.zeros(face_count, dtype=bool)
    for capture in captures:
        projected = capture.projected_sample_counts
        visible = capture.visible_sample_counts
        meaningful = projected >= minimum_projected_samples
        has_meaningful_projection |= meaningful
        has_visible_sample |= visible > 0
        qualifies_as_visible |= (
            meaningful
            & (visible > 0)
            & (
                visible.astype(float)
                >= projected.astype(float) * minimum_visible_fraction
            )
        )
    unmeasurable_subpixel = ~has_meaningful_projection & has_visible_sample
    return qualifies_as_visible | unmeasurable_subpixel


# ### Stacked-layer helpers ###
def _find_stacked_faces(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    captures: Sequence[_CameraCapture],
    image_size: int,
    minimum_projected_samples: int,
    cancel_requested: CancelCallback | None,
    progress_interval_faces: int,
) -> np.ndarray:
    """Find same-facing wafer layers hidden by a nearby parallel surface."""

    face_count = len(faces)
    stacked_faces = np.zeros(face_count, dtype=bool)
    if face_count == 0 or not captures:
        return stacked_faces
    face_normals, normal_is_reliable = _build_face_unit_normals(
        vertices,
        faces,
    )
    camera_directions = np.asarray(
        [
            _CAMERA_DEFINITIONS[camera_id].depth_axis
            for camera_id in ALL_CAMERA_IDS
        ],
        dtype=float,
    )
    camera_alignment = face_normals @ camera_directions.T
    dominant_camera_indices = np.argmax(camera_alignment, axis=1)
    dominant_camera_indices[~normal_is_reliable] = -1
    capture_by_id = {capture.camera_id: capture for capture in captures}
    bounds_diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    depth_epsilon = max(
        bounds_diagonal * DEPTH_EPSILON_RATIO,
        MINIMUM_DEPTH_EPSILON,
    )
    maximum_depth_gap = max(
        bounds_diagonal * STACK_MAXIMUM_DEPTH_GAP_RATIO,
        depth_epsilon,
    )
    checked_face_count = 0
    for camera_index, camera_id in enumerate(ALL_CAMERA_IDS):
        capture = capture_by_id.get(camera_id)
        if capture is None:
            continue
        definition = _CAMERA_DEFINITIONS[camera_id]
        screen_vertices, vertex_depths = _project_vertices(
            vertices,
            definition,
            image_size,
        )
        candidate_indices = np.flatnonzero(
            dominant_camera_indices == camera_index
        )
        for face_index in candidate_indices:
            if checked_face_count % progress_interval_faces == 0:
                _raise_if_cancelled(cancel_requested)
            checked_face_count += 1
            face = faces[face_index]
            rows, columns, sample_depths = _rasterize_face_samples(
                screen_points=screen_vertices[face],
                depths=vertex_depths[face],
                image_shape=capture.top_faces.shape,
            )
            if len(rows) < minimum_projected_samples:
                continue
            top_face_indices = capture.top_faces[rows, columns]
            occluded = (
                (top_face_indices >= 0)
                & (top_face_indices != face_index)
            )
            if not np.any(occluded):
                continue
            occluded_positions = np.flatnonzero(occluded)
            occluder_indices = top_face_indices[occluded_positions]
            normal_alignment = (
                face_normals[occluder_indices] @ face_normals[face_index]
            )
            depth_gaps = (
                capture.top_depth[
                    rows[occluded_positions],
                    columns[occluded_positions],
                ]
                - sample_depths[occluded_positions]
            )
            qualifying_occlusion = (
                normal_alignment >= STACK_NORMAL_ALIGNMENT_COSINE
            ) & (depth_gaps >= -depth_epsilon) & (
                depth_gaps <= maximum_depth_gap
            )
            if (
                np.count_nonzero(qualifying_occlusion) / len(rows)
                >= STACK_MINIMUM_OCCLUDED_FRACTION
            ):
                stacked_faces[face_index] = True
    return stacked_faces


def _build_face_unit_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    triangles = vertices[faces]
    weighted_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    magnitudes = np.linalg.norm(weighted_normals, axis=1)
    reliable = magnitudes > np.finfo(float).eps
    unit_normals = np.zeros_like(weighted_normals, dtype=float)
    unit_normals[reliable] = (
        weighted_normals[reliable] / magnitudes[reliable, np.newaxis]
    )
    return unit_normals, reliable


# ### Export helpers ###
def _export_filtered_scene(
    instances: Sequence[_MeshInstance],
    keep_faces: np.ndarray,
) -> bytes:
    output_scene = trimesh.Scene()
    used_geometry_names: set[str] = set()
    used_node_names: set[str] = set()
    for instance_index, instance in enumerate(instances):
        local_keep = keep_faces[
            instance.first_face_index : instance.first_face_index
            + instance.face_count
        ]
        if not np.any(local_keep):
            continue
        filtered_mesh = instance.mesh.copy()
        filtered_mesh.update_faces(local_keep)
        filtered_mesh.remove_unreferenced_vertices()
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
        output_scene.add_geometry(
            filtered_mesh,
            geom_name=geometry_name,
            node_name=node_name,
            transform=instance.transform,
        )
    if not output_scene.geometry:
        raise ValueError("Unused-face removal produced an empty scene.")
    try:
        return bytes(output_scene.export(file_type="glb"))
    except Exception as error:
        raise ValueError("The post-processed GLB could not be exported.") from error


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


# ### Validation and callback helpers ###
def _normalize_camera_ids(camera_ids: Iterable[str]) -> tuple[str, ...]:
    requested_camera_ids = tuple(str(camera_id) for camera_id in camera_ids)
    unknown_camera_ids = set(requested_camera_ids).difference(ALL_CAMERA_IDS)
    if unknown_camera_ids:
        unknown_labels = ", ".join(sorted(unknown_camera_ids))
        raise ValueError(f"Unknown unused-face camera IDs: {unknown_labels}.")
    normalized_camera_ids = tuple(
        camera_id for camera_id in ALL_CAMERA_IDS if camera_id in requested_camera_ids
    )
    if not normalized_camera_ids:
        raise ValueError("Select at least one unused-face camera.")
    return normalized_camera_ids


def _normalize_minimum_visible_fraction(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            "Unused-face minimum visible fraction must be a number."
        )
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(
            "Unused-face minimum visible fraction must be between 0 and 1."
        )
    return normalized


def _normalize_minimum_projected_samples(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(
            "Unused-face minimum projected samples must be a positive integer."
        )
    try:
        normalized = operator.index(value)
    except TypeError as error:
        raise ValueError(
            "Unused-face minimum projected samples must be a positive integer."
        ) from error
    if normalized < 1:
        raise ValueError(
            "Unused-face minimum projected samples must be a positive integer."
        )
    return int(normalized)


def _validate_processing_bounds(
    *,
    image_size: int,
    max_face_count: int,
    progress_interval_faces: int,
) -> None:
    if not MIN_CAPTURE_IMAGE_SIZE <= int(image_size) <= MAX_CAPTURE_IMAGE_SIZE:
        raise ValueError(
            "Unused-face capture size must be between "
            f"{MIN_CAPTURE_IMAGE_SIZE} and {MAX_CAPTURE_IMAGE_SIZE} pixels."
        )
    if not 1 <= int(max_face_count) <= MAX_SUPPORTED_FACE_COUNT:
        raise ValueError(
            "Unused-face maximum face count must be between 1 and "
            f"{MAX_SUPPORTED_FACE_COUNT}."
        )
    if int(progress_interval_faces) < 1:
        raise ValueError("Progress interval must contain at least one face.")


def _raise_if_cancelled(cancel_requested: CancelCallback | None) -> None:
    if cancel_requested is not None and cancel_requested():
        raise UnusedFaceRemovalCancelled("Unused-face removal was cancelled.")


def _report_progress(
    progress_callback: ProgressCallback | None,
    *,
    stage: str,
    completed_face_count: int,
    total_face_count: int,
    camera_id: str | None = None,
) -> None:
    if progress_callback is None:
        return
    progress_callback(
        UnusedFaceRemovalProgress(
            stage=stage,
            completed_face_count=int(completed_face_count),
            total_face_count=int(total_face_count),
            camera_id=camera_id,
        )
    )
