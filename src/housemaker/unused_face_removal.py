# ### Imports ###
from __future__ import annotations

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
MAX_FACE_ID_COLOR_COUNT = (1 << 24) - 1
DEFAULT_PROGRESS_INTERVAL_FACES = 128
CAMERA_FRAME_MARGIN_RATIO = 0.05
MINIMUM_CAMERA_EXTENT = 1e-9
DEPTH_EPSILON_RATIO = 1e-9
MINIMUM_DEPTH_EPSILON = 1e-12


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

    def __post_init__(self) -> None:
        normalized_camera_ids = _normalize_camera_ids(self.enabled_camera_ids)
        object.__setattr__(self, "enabled_camera_ids", normalized_camera_ids)
        _validate_processing_bounds(
            image_size=self.image_size,
            max_face_count=self.max_face_count,
            progress_interval_faces=self.progress_interval_faces,
        )


@dataclass(frozen=True)
class UncheckedCameraFacePurgeOptions:
    """Bounds for removing faces exposed to unchecked fixed cameras."""

    image_size: int = DEFAULT_CAPTURE_IMAGE_SIZE
    max_face_count: int = DEFAULT_MAX_FACE_COUNT
    progress_interval_faces: int = DEFAULT_PROGRESS_INTERVAL_FACES

    def __post_init__(self) -> None:
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

    @property
    def glb_bytes(self) -> bytes:
        return self.model.glb_bytes


@dataclass(frozen=True)
class UncheckedCameraFacePurgeResult:
    """Model produced by deleting the union visible from unchecked cameras."""

    model: GeneratedModel
    unchecked_camera_ids: tuple[str, ...]
    original_face_count: int
    retained_face_count: int
    removed_face_count: int

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
class _FaceChangeSample:
    before_bgr: np.ndarray
    after_bgr: np.ndarray


@dataclass(frozen=True)
class _CameraCapture:
    camera_id: str
    face_samples: dict[int, _FaceChangeSample]


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
    return frozenset(capture.face_samples)


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

    Each capture stores a face-ID image rather than a beauty render. This is
    deliberately conservative: deleting any raster-visible triangle changes
    at least one OpenCV pixel even when two materials happen to share a color.
    Faces that do not own a pixel in any checked view are removed. Because a
    visible face is always retained, removing hidden faces cannot expose
    another candidate during the same operation.
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

    keep_faces = np.zeros(total_face_count, dtype=bool)
    _report_progress(
        progress_callback,
        stage="checking",
        completed_face_count=0,
        total_face_count=total_face_count,
    )
    for face_index in range(total_face_count):
        if face_index % normalized_options.progress_interval_faces == 0:
            _raise_if_cancelled(cancel_requested)
            _report_progress(
                progress_callback,
                stage="checking",
                completed_face_count=face_index,
                total_face_count=total_face_count,
            )
        keep_faces[face_index] = any(
            _sample_changes_frame(capture.face_samples.get(face_index))
            for capture in captures
        )

    retained_face_count = int(np.count_nonzero(keep_faces))
    if retained_face_count == 0:
        raise ValueError(
            "Unused-face removal would remove every face; increase the "
            "capture resolution or select additional cameras."
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
    )


def purge_faces_visible_from_unchecked_cameras_from_glb(
    glb_bytes: bytes,
    *,
    unchecked_camera_ids: Iterable[str],
    options: UncheckedCameraFacePurgeOptions | None = None,
    cancel_requested: CancelCallback | None = None,
    progress_callback: ProgressCallback | None = None,
) -> UncheckedCameraFacePurgeResult:
    """Delete every face owning a depth pixel in any unchecked camera view.

    Visibility comes from the same six fixed orthographic depth captures used
    by unused-face removal and camera UV projection. An unchecked face is
    removed once even when it is visible from several unchecked cameras.
    Passing no unchecked cameras is an explicit no-op that preserves the
    original GLB bytes.
    """

    normalized_options = options or UncheckedCameraFacePurgeOptions()
    normalized_camera_ids = _normalize_optional_camera_ids(
        unchecked_camera_ids
    )
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
            f"camera-face purge limit is {normalized_options.max_face_count}."
        )

    if not normalized_camera_ids:
        _report_progress(
            progress_callback,
            stage="complete",
            completed_face_count=total_face_count,
            total_face_count=total_face_count,
        )
        return UncheckedCameraFacePurgeResult(
            model=import_generated_glb(payload),
            unchecked_camera_ids=(),
            original_face_count=total_face_count,
            retained_face_count=total_face_count,
            removed_face_count=0,
        )

    _report_progress(
        progress_callback,
        stage="capturing",
        completed_face_count=0,
        total_face_count=total_face_count,
    )
    captures: list[_CameraCapture] = []
    for camera_id in normalized_camera_ids:
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

    remove_faces = np.zeros(total_face_count, dtype=bool)
    _report_progress(
        progress_callback,
        stage="checking",
        completed_face_count=0,
        total_face_count=total_face_count,
    )
    for face_index in range(total_face_count):
        if face_index % normalized_options.progress_interval_faces == 0:
            _raise_if_cancelled(cancel_requested)
            _report_progress(
                progress_callback,
                stage="checking",
                completed_face_count=face_index,
                total_face_count=total_face_count,
            )
        remove_faces[face_index] = any(
            _sample_changes_frame(capture.face_samples.get(face_index))
            for capture in captures
        )

    removed_face_count = int(np.count_nonzero(remove_faces))
    retained_face_count = total_face_count - removed_face_count
    if retained_face_count == 0:
        raise ValueError(
            "Camera-face purge would remove every face; check at least one "
            "fixed camera."
        )
    _raise_if_cancelled(cancel_requested)
    _report_progress(
        progress_callback,
        stage="exporting",
        completed_face_count=total_face_count,
        total_face_count=total_face_count,
    )

    if removed_face_count == 0:
        processed_model = import_generated_glb(payload)
    else:
        processed_glb = _export_filtered_scene(instances, ~remove_faces)
        processed_model = import_generated_glb(processed_glb)
    _report_progress(
        progress_callback,
        stage="complete",
        completed_face_count=total_face_count,
        total_face_count=total_face_count,
    )
    return UncheckedCameraFacePurgeResult(
        model=processed_model,
        unchecked_camera_ids=normalized_camera_ids,
        original_face_count=total_face_count,
        retained_face_count=retained_face_count,
        removed_face_count=removed_face_count,
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
    second_depth = np.full((image_size, image_size), -np.inf, dtype=np.float64)
    top_faces = np.full((image_size, image_size), -1, dtype=np.int32)
    second_faces = np.full((image_size, image_size), -1, dtype=np.int32)
    bounds_diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    depth_epsilon = max(
        bounds_diagonal * DEPTH_EPSILON_RATIO,
        MINIMUM_DEPTH_EPSILON,
    )
    for face_index, face in enumerate(faces):
        if face_index % progress_interval_faces == 0:
            _raise_if_cancelled(cancel_requested)
        _rasterize_face_depth_layers(
            face_index=face_index,
            screen_points=screen_vertices[face],
            depths=vertex_depths[face],
            top_depth=top_depth,
            second_depth=second_depth,
            top_faces=top_faces,
            second_faces=second_faces,
            depth_epsilon=depth_epsilon,
        )
    return _CameraCapture(
        camera_id=definition.camera_id,
        face_samples=_build_face_change_samples(top_faces, second_faces),
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


def _rasterize_face_depth_layers(
    *,
    face_index: int,
    screen_points: np.ndarray,
    depths: np.ndarray,
    top_depth: np.ndarray,
    second_depth: np.ndarray,
    top_faces: np.ndarray,
    second_faces: np.ndarray,
    depth_epsilon: float,
) -> None:
    image_height, image_width = top_faces.shape
    minimum_x = max(0, int(np.floor(np.min(screen_points[:, 0]))))
    maximum_x = min(image_width - 1, int(np.ceil(np.max(screen_points[:, 0]))))
    minimum_y = max(0, int(np.floor(np.min(screen_points[:, 1]))))
    maximum_y = min(image_height - 1, int(np.ceil(np.max(screen_points[:, 1]))))
    if maximum_x < minimum_x or maximum_y < minimum_y:
        return

    first, second, third = screen_points
    denominator = (
        (second[1] - third[1]) * (first[0] - third[0])
        + (third[0] - second[0]) * (first[1] - third[1])
    )
    if abs(float(denominator)) <= np.finfo(float).eps:
        return

    mask = np.zeros(
        (maximum_y - minimum_y + 1, maximum_x - minimum_x + 1),
        dtype=np.uint8,
    )
    local_polygon = np.rint(
        screen_points - np.array([minimum_x, minimum_y], dtype=float)
    ).astype(np.int32)
    cv2.fillConvexPoly(mask, local_polygon, 255, lineType=cv2.LINE_8)
    local_y, local_x = np.nonzero(mask)
    if len(local_x) == 0:
        return
    pixel_x = local_x.astype(float) + minimum_x
    pixel_y = local_y.astype(float) + minimum_y
    first_weight = (
        (second[1] - third[1]) * (pixel_x - third[0])
        + (third[0] - second[0]) * (pixel_y - third[1])
    ) / denominator
    second_weight = (
        (third[1] - first[1]) * (pixel_x - third[0])
        + (first[0] - third[0]) * (pixel_y - third[1])
    ) / denominator
    third_weight = 1.0 - first_weight - second_weight
    interpolated_depth = (
        first_weight * depths[0]
        + second_weight * depths[1]
        + third_weight * depths[2]
    )
    pixel_y_indices = local_y + minimum_y
    pixel_x_indices = local_x + minimum_x
    current_top_depth = top_depth[pixel_y_indices, pixel_x_indices]
    current_second_depth = second_depth[pixel_y_indices, pixel_x_indices]

    becomes_top = interpolated_depth > current_top_depth + depth_epsilon
    if np.any(becomes_top):
        top_y = pixel_y_indices[becomes_top]
        top_x = pixel_x_indices[becomes_top]
        second_depth[top_y, top_x] = current_top_depth[becomes_top]
        second_faces[top_y, top_x] = top_faces[top_y, top_x]
        top_depth[top_y, top_x] = interpolated_depth[becomes_top]
        top_faces[top_y, top_x] = face_index

    remains_below_top = ~becomes_top
    becomes_second = remains_below_top & (
        interpolated_depth > current_second_depth + depth_epsilon
    )
    if np.any(becomes_second):
        second_y = pixel_y_indices[becomes_second]
        second_x = pixel_x_indices[becomes_second]
        second_depth[second_y, second_x] = interpolated_depth[becomes_second]
        second_faces[second_y, second_x] = face_index


def _build_face_change_samples(
    top_faces: np.ndarray,
    second_faces: np.ndarray,
) -> dict[int, _FaceChangeSample]:
    flattened_top = top_faces.reshape(-1)
    visible_pixels = np.flatnonzero(flattened_top >= 0)
    if len(visible_pixels) == 0:
        return {}
    visible_face_ids, first_occurrences = np.unique(
        flattened_top[visible_pixels],
        return_index=True,
    )
    sample_pixels = visible_pixels[first_occurrences]
    underlying_face_ids = second_faces.reshape(-1)[sample_pixels]
    return {
        int(face_id): _FaceChangeSample(
            before_bgr=_encode_face_id_bgr(int(face_id)),
            after_bgr=_encode_face_id_bgr(int(underlying_face_id)),
        )
        for face_id, underlying_face_id in zip(
            visible_face_ids,
            underlying_face_ids,
        )
    }


def _encode_face_id_bgr(face_id: int) -> np.ndarray:
    encoded_id = int(face_id) + 1
    if encoded_id < 0 or encoded_id > MAX_FACE_ID_COLOR_COUNT:
        raise ValueError("Face ID cannot be encoded in a capture frame.")
    return np.asarray(
        [
            encoded_id & 0xFF,
            (encoded_id >> 8) & 0xFF,
            (encoded_id >> 16) & 0xFF,
        ],
        dtype=np.uint8,
    ).reshape(1, 1, 3)


def _sample_changes_frame(sample: _FaceChangeSample | None) -> bool:
    if sample is None:
        return False
    pixel_difference = cv2.absdiff(sample.before_bgr, sample.after_bgr)
    return bool(cv2.countNonZero(pixel_difference.reshape(-1, 1)))


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
    normalized_camera_ids = _normalize_optional_camera_ids(camera_ids)
    if not normalized_camera_ids:
        raise ValueError("Select at least one unused-face camera.")
    return normalized_camera_ids


def _normalize_optional_camera_ids(
    camera_ids: Iterable[str],
) -> tuple[str, ...]:
    requested_camera_ids = tuple(str(camera_id) for camera_id in camera_ids)
    unknown_camera_ids = set(requested_camera_ids).difference(ALL_CAMERA_IDS)
    if unknown_camera_ids:
        unknown_labels = ", ".join(sorted(unknown_camera_ids))
        raise ValueError(f"Unknown unused-face camera IDs: {unknown_labels}.")
    return tuple(
        camera_id for camera_id in ALL_CAMERA_IDS if camera_id in requested_camera_ids
    )


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
    if not 1 <= int(max_face_count) <= MAX_FACE_ID_COLOR_COUNT:
        raise ValueError(
            "Unused-face maximum face count must be between 1 and "
            f"{MAX_FACE_ID_COLOR_COUNT}."
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
