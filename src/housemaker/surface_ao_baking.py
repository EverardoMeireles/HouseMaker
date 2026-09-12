# ### Imports ###
from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import trimesh
import xatlas
from trimesh.visual.texture import TextureVisuals

from housemaker import object_ao_baking
from housemaker.object_ao_baking import ObjectAmbientOcclusionTarget

# ### Public constants ###
SURFACE_AO_BAKE_VERSION = 3
SURFACE_AO_MINIMUM_PADDING_PIXELS = 2
SURFACE_AO_DEFAULT_PADDING_PIXELS = 4


# ### Configuration constants ###
_MINIMUM_TEXTURE_RESOLUTION = 16
_MAXIMUM_RECEIVER_FACE_COUNT = 2_000_000
_MAXIMUM_PACK_ATTEMPTS_PER_TARGET = 3
_PACKING_STRATEGIES = (
    ("fixed", False, False),
    ("rotate_charts", True, False),
    ("align_charts", False, True),
    ("rotate_and_align_charts", True, True),
)
_GEOMETRY_EPSILON = 1e-12
_UV_EPSILON = 1e-7
_XATLAS_MINIMUM_FACE_ALTITUDE_PIXELS = 0.25
_XATLAS_MINIMUM_NORMALIZED_FACE_ALTITUDE = 1e-5
_XATLAS_PROXY_FACE_EDGE_PIXELS = 2.0
_XATLAS_MINIMUM_PROXY_FACE_EDGE = 0.002
_SIGNATURE_CHUNK_BYTE_COUNT = 1_048_576
_SIGNATURE_PREFIX = b"housemaker-surface-ambient-occlusion\0"


# ### Public exceptions ###
class SurfaceAmbientOcclusionBakeCancelled(RuntimeError):
    """Raised when a surface AO layout or bake is cancelled safely."""


# ### Public input models ###
@dataclass(frozen=True)
class SurfaceAmbientOcclusionReceiver:
    """One ordered world-space AO receiver with its existing UV0 mesh."""

    key: str
    mesh: trimesh.Trimesh
    mirror_plane_point: tuple[float, float, float] | None = None
    mirror_plane_normal: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        key = str(self.key).strip()
        if not key:
            raise ValueError("A surface AO receiver key cannot be empty.")
        if not isinstance(self.mesh, trimesh.Trimesh):
            raise TypeError("A surface AO receiver requires a triangle mesh.")
        point = _normalize_optional_vector(self.mirror_plane_point)
        normal = _normalize_optional_vector(self.mirror_plane_normal)
        if (point is None) != (normal is None):
            raise ValueError(
                "A surface AO mirror point and normal must be supplied together."
            )
        if normal is not None:
            length = float(np.linalg.norm(normal))
            if length <= _GEOMETRY_EPSILON:
                raise ValueError("A surface AO mirror-plane normal cannot be zero.")
            normal = tuple(float(value) for value in np.asarray(normal) / length)
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "mirror_plane_point", point)
        object.__setattr__(self, "mirror_plane_normal", normal)


# ### Public layout models ###
@dataclass(frozen=True)
class SurfaceAmbientOcclusionReceiverLayout:
    """One receiver expanded at UV1 seams without changing face topology.

    ``vertex_mapping`` maps each layout vertex back to the corresponding
    vertex in the receiver's original mesh. A caller can therefore preserve
    POSITION, NORMAL, UV0, and any other per-vertex attributes by indexing
    each original attribute with this array. ``uv1`` uses trimesh's V-up
    texture-coordinate convention; a raw custom glTF attribute must flip V
    before it is renamed to ``TEXCOORD_1``.
    """

    key: str
    source_world_vertices: np.ndarray
    vertex_mapping: np.ndarray
    faces: np.ndarray
    uv1: np.ndarray
    mirror_plane_point: tuple[float, float, float] | None = None
    mirror_plane_normal: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        key = str(self.key).strip()
        if not key:
            raise ValueError("A surface AO receiver layout key cannot be empty.")
        source_vertices = _immutable_array(
            self.source_world_vertices,
            dtype=np.float64,
        )
        vertex_mapping = _immutable_array(
            self.vertex_mapping,
            dtype=np.int64,
        )
        faces = _immutable_array(self.faces, dtype=np.int64)
        uv1 = _immutable_array(self.uv1, dtype=np.float64)
        if source_vertices.ndim != 2 or source_vertices.shape[1:] != (3,):
            raise ValueError("Surface AO source vertices must be 3D values.")
        if not len(source_vertices) or not np.all(np.isfinite(source_vertices)):
            raise ValueError("Surface AO source vertices must be finite and nonempty.")
        if vertex_mapping.ndim != 1 or not len(vertex_mapping):
            raise ValueError("A surface AO vertex mapping cannot be empty.")
        if np.any(vertex_mapping < 0) or np.any(vertex_mapping >= len(source_vertices)):
            raise ValueError("A surface AO vertex mapping references a missing vertex.")
        if faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces):
            raise ValueError("Surface AO layout faces must be triangles.")
        if np.any(faces < 0) or np.any(faces >= len(vertex_mapping)):
            raise ValueError("Surface AO layout faces reference a missing vertex.")
        if uv1.shape != (len(vertex_mapping), 2) or not np.all(np.isfinite(uv1)):
            raise ValueError("Surface AO UV1 coordinates are invalid.")
        if np.any(uv1 < -_UV_EPSILON) or np.any(uv1 > 1.0 + _UV_EPSILON):
            raise ValueError("Surface AO UV1 coordinates leave the texture atlas.")
        point = _normalize_optional_vector(self.mirror_plane_point)
        normal = _normalize_optional_vector(self.mirror_plane_normal)
        if (point is None) != (normal is None):
            raise ValueError(
                "A surface AO mirror point and normal must be supplied together."
            )
        if normal is not None:
            length = float(np.linalg.norm(normal))
            if length <= _GEOMETRY_EPSILON:
                raise ValueError("A surface AO mirror-plane normal cannot be zero.")
            normal = tuple(float(value) for value in np.asarray(normal) / length)

        clipped_uv1 = np.ascontiguousarray(np.clip(uv1, 0.0, 1.0))
        clipped_uv1.setflags(write=False)
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "source_world_vertices", source_vertices)
        object.__setattr__(self, "vertex_mapping", vertex_mapping)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "uv1", clipped_uv1)
        object.__setattr__(self, "mirror_plane_point", point)
        object.__setattr__(self, "mirror_plane_normal", normal)

    @property
    def world_vertices(self) -> np.ndarray:
        """Return expanded world vertices corresponding to ``faces`` and UV1."""

        expanded = np.ascontiguousarray(self.source_world_vertices[self.vertex_mapping])
        expanded.setflags(write=False)
        return expanded

    def map_vertex_attribute(self, values: np.ndarray) -> np.ndarray:
        """Expand one source per-vertex attribute through the UV1 seam map."""

        source = np.asarray(values)
        if source.ndim < 1 or len(source) != len(self.source_world_vertices):
            raise ValueError(
                "A mapped surface AO attribute must match the source vertex count."
            )
        mapped = np.ascontiguousarray(source[self.vertex_mapping])
        mapped.setflags(write=False)
        return mapped

    @property
    def gltf_uv1(self) -> np.ndarray:
        """Return UV1 in glTF convention for a raw ``TEXCOORD_1`` accessor."""

        coordinates = self.uv1.copy()
        coordinates[:, 1] = 1.0 - coordinates[:, 1]
        coordinates = np.ascontiguousarray(coordinates, dtype=np.float32)
        coordinates.setflags(write=False)
        return coordinates


@dataclass(frozen=True)
class SurfaceAmbientOcclusionUvLayout:
    """One globally packed, deterministic UV1 layout for an Atlas."""

    resolution: int
    requested_padding_pixels: int
    effective_padding_pixels: float
    atlas_width: int
    atlas_height: int
    packing_strategy: str
    receivers: tuple[SurfaceAmbientOcclusionReceiverLayout, ...]
    layout_signature: str

    def __post_init__(self) -> None:
        _validate_resolution(self.resolution)
        _validate_padding(self.requested_padding_pixels)
        if (
            not math.isfinite(float(self.effective_padding_pixels))
            or float(self.effective_padding_pixels)
            < float(self.requested_padding_pixels) - _UV_EPSILON
        ):
            raise ValueError("A surface AO layout has insufficient UV padding.")
        if int(self.atlas_width) <= 0 or int(self.atlas_height) <= 0:
            raise ValueError("A surface AO layout has invalid Atlas dimensions.")
        receivers = tuple(self.receivers)
        if not receivers or not all(
            isinstance(item, SurfaceAmbientOcclusionReceiverLayout)
            for item in receivers
        ):
            raise ValueError("A surface AO layout requires receiver layouts.")
        keys = tuple(item.key for item in receivers)
        if len(keys) != len(set(keys)):
            raise ValueError("Surface AO receiver layout keys must be unique.")
        signature = str(self.layout_signature).strip().lower()
        if len(signature) != 64 or any(
            character not in "0123456789abcdef" for character in signature
        ):
            raise ValueError("A surface AO layout signature must be SHA-256.")
        object.__setattr__(self, "resolution", int(self.resolution))
        object.__setattr__(
            self,
            "requested_padding_pixels",
            int(self.requested_padding_pixels),
        )
        object.__setattr__(
            self,
            "effective_padding_pixels",
            float(self.effective_padding_pixels),
        )
        object.__setattr__(self, "atlas_width", int(self.atlas_width))
        object.__setattr__(self, "atlas_height", int(self.atlas_height))
        object.__setattr__(self, "packing_strategy", str(self.packing_strategy))
        object.__setattr__(self, "receivers", receivers)
        object.__setattr__(self, "layout_signature", signature)


@dataclass(frozen=True)
class SurfaceAmbientOcclusionBakeResult:
    """A UV1 layout plus its full-resolution grayscale AO plane."""

    layout: SurfaceAmbientOcclusionUvLayout
    ambient_occlusion: np.ndarray
    geometry_signature: str

    def __post_init__(self) -> None:
        if not isinstance(self.layout, SurfaceAmbientOcclusionUvLayout):
            raise TypeError("A surface AO bake requires a UV1 layout.")
        ambient_occlusion = _immutable_array(
            self.ambient_occlusion,
            dtype=np.uint8,
        )
        expected_shape = (self.layout.resolution, self.layout.resolution)
        if ambient_occlusion.shape != expected_shape:
            raise ValueError("A surface AO plane must match its Atlas resolution.")
        signature = str(self.geometry_signature).strip().lower()
        if len(signature) != 64 or any(
            character not in "0123456789abcdef" for character in signature
        ):
            raise ValueError("A surface AO geometry signature must be SHA-256.")
        object.__setattr__(self, "ambient_occlusion", ambient_occlusion)
        object.__setattr__(self, "geometry_signature", signature)


# ### Internal packing models ###
@dataclass(frozen=True)
class _PackedReceiver:
    vertex_mapping: np.ndarray
    faces: np.ndarray
    uv1: np.ndarray


@dataclass(frozen=True)
class _PackingMesh:
    """Geometry-only adjacency proxy submitted to xatlas.

    Scan-projected objects intentionally carry separate UV0 vertices at every
    triangle corner. AO UV1 does not need those UV0 seams, so submitting the
    source indices directly would make xatlas treat every triangle as an
    independent chart. ``source_faces`` retains the authored vertex indices
    while ``faces`` joins only unambiguous, consistently wound geometric
    neighbours for parameterization.
    """

    vertices: np.ndarray
    faces: np.ndarray
    source_faces: np.ndarray


@dataclass(frozen=True)
class _PackingCandidate:
    receivers: tuple[_PackedReceiver, ...]
    internal_padding_pixels: int
    effective_padding_pixels: float
    atlas_width: int
    atlas_height: int
    packing_strategy: str
    uv_triangle_area: float
    strategy_index: int


class _PackingCandidateError(RuntimeError):
    """One xatlas packing policy failed without invalidating other policies."""


# ### Public layout helpers ###
def build_surface_ao_uv_layout(
    receivers: Sequence[SurfaceAmbientOcclusionReceiver],
    resolution: int,
    *,
    padding_pixels: int = SURFACE_AO_DEFAULT_PADDING_PIXELS,
    cancellation_check: Callable[[], bool] | None = None,
) -> SurfaceAmbientOcclusionUvLayout:
    """Globally unwrap and pack ordered world-space receivers into UV1.

    All receivers are submitted to one xatlas instance for every candidate, so
    charts from different meshes cannot unknowingly occupy the same texture
    region. The existing UV0 data is validated but never changed.
    """

    normalized_receivers = _normalize_receivers(receivers)
    _validate_resolution(resolution)
    _validate_padding(padding_pixels)
    _check_cancelled(cancellation_check)
    candidate = _build_best_packing_candidate(
        normalized_receivers,
        int(resolution),
        int(padding_pixels),
        cancellation_check,
    )
    guaranteed_padding_pixels = min(
        int(padding_pixels),
        math.floor(candidate.effective_padding_pixels + _UV_EPSILON),
    )
    receiver_layouts = tuple(
        SurfaceAmbientOcclusionReceiverLayout(
            key=receiver.key,
            source_world_vertices=np.asarray(receiver.mesh.vertices, dtype=float),
            vertex_mapping=packed.vertex_mapping,
            faces=packed.faces,
            uv1=packed.uv1,
            mirror_plane_point=receiver.mirror_plane_point,
            mirror_plane_normal=receiver.mirror_plane_normal,
        )
        for receiver, packed in zip(
            normalized_receivers,
            candidate.receivers,
            strict=True,
        )
    )
    layout_signature = _build_layout_signature(
        receiver_layouts,
        resolution=int(resolution),
        requested_padding_pixels=guaranteed_padding_pixels,
        candidate=candidate,
        cancellation_check=cancellation_check,
    )
    return SurfaceAmbientOcclusionUvLayout(
        resolution=int(resolution),
        requested_padding_pixels=guaranteed_padding_pixels,
        effective_padding_pixels=candidate.effective_padding_pixels,
        atlas_width=candidate.atlas_width,
        atlas_height=candidate.atlas_height,
        packing_strategy=candidate.packing_strategy,
        receivers=receiver_layouts,
        layout_signature=layout_signature,
    )


def build_surface_ao_geometry_signature(
    layout: SurfaceAmbientOcclusionUvLayout,
    occluder_meshes: Sequence[trimesh.Trimesh],
    *,
    cancellation_check: Callable[[], bool] | None = None,
) -> str:
    """Hash the exact UV1 layout and usable opaque occluder triangle soup."""

    if not isinstance(layout, SurfaceAmbientOcclusionUvLayout):
        raise TypeError("A surface AO geometry signature requires a UV1 layout.")
    _check_cancelled(cancellation_check)
    try:
        occluder = object_ao_baking._build_occluder_mesh(
            occluder_meshes,
            cancellation_check=cancellation_check,
        )
    except object_ao_baking.ObjectAmbientOcclusionBakeCancelled as error:
        raise SurfaceAmbientOcclusionBakeCancelled(str(error)) from error
    return _build_geometry_signature_from_occluder(
        layout,
        occluder,
        cancellation_check=cancellation_check,
    )


# ### Public baking helpers ###
def bake_surface_ambient_occlusion(
    layout: SurfaceAmbientOcclusionUvLayout,
    occluder_meshes: Sequence[trimesh.Trimesh],
    *,
    cancellation_check: Callable[[], bool] | None = None,
) -> SurfaceAmbientOcclusionBakeResult:
    """Bake independent UV1 receivers into one grayscale AO texture plane.

    Tight receiver images are composited into their Atlas slices with a minimum
    operation. Correct xatlas output makes their unpadded triangles disjoint;
    minimum composition also gives deterministic behavior where dilation
    gutters meet.
    """

    if not isinstance(layout, SurfaceAmbientOcclusionUvLayout):
        raise TypeError("A surface AO bake requires a UV1 layout.")
    _check_cancelled(cancellation_check)
    try:
        occluder = object_ao_baking._build_occluder_mesh(
            occluder_meshes,
            cancellation_check=cancellation_check,
        )
    except object_ao_baking.ObjectAmbientOcclusionBakeCancelled as error:
        raise SurfaceAmbientOcclusionBakeCancelled(str(error)) from error
    geometry_signature = _build_geometry_signature_from_occluder(
        layout,
        occluder,
        cancellation_check=cancellation_check,
    )
    ambient_occlusion = np.full(
        (layout.resolution, layout.resolution),
        255,
        dtype=np.uint8,
    )
    if occluder is None:
        return SurfaceAmbientOcclusionBakeResult(
            layout=layout,
            ambient_occlusion=ambient_occlusion,
            geometry_signature=geometry_signature,
        )

    try:
        intersector = object_ao_baking._build_ray_intersector(occluder)
    except RuntimeError as error:
        raise ValueError(str(error)) from error
    _check_cancelled(cancellation_check)
    ray_bias = object_ao_baking._build_scale_aware_ray_bias(occluder)
    try:
        for receiver in layout.receivers:
            _check_cancelled(cancellation_check)
            target = _build_bake_target(receiver, layout.resolution)
            if cancellation_check is None:
                baked = object_ao_baking._bake_target_mesh(
                    target,
                    intersector,
                    ray_bias,
                    layout.resolution,
                )
            else:
                baked = object_ao_baking._bake_target_mesh(
                    target,
                    intersector,
                    ray_bias,
                    layout.resolution,
                    cancellation_check=cancellation_check,
                )
            if baked is None:
                continue
            receiver_values, receiver_coverage = baked
            left, top, right, bottom = target.atlas_pixel_bounds
            receiver_region = ambient_occlusion[top:bottom, left:right]
            receiver_region[receiver_coverage] = np.minimum(
                receiver_region[receiver_coverage],
                receiver_values[receiver_coverage],
            )
    except SurfaceAmbientOcclusionBakeCancelled:
        raise
    except object_ao_baking.ObjectAmbientOcclusionBakeCancelled as error:
        raise SurfaceAmbientOcclusionBakeCancelled(str(error)) from error
    except (MemoryError, RuntimeError, ValueError) as error:
        raise ValueError("Surface ambient occlusion could not be baked.") from error
    _check_cancelled(cancellation_check)
    return SurfaceAmbientOcclusionBakeResult(
        layout=layout,
        ambient_occlusion=ambient_occlusion,
        geometry_signature=geometry_signature,
    )


def build_and_bake_surface_ambient_occlusion(
    receivers: Sequence[SurfaceAmbientOcclusionReceiver],
    resolution: int,
    occluder_meshes: Sequence[trimesh.Trimesh],
    *,
    padding_pixels: int = SURFACE_AO_DEFAULT_PADDING_PIXELS,
    cancellation_check: Callable[[], bool] | None = None,
) -> SurfaceAmbientOcclusionBakeResult:
    """Build UV1 and bake AO in one call for callers without a layout cache."""

    layout = build_surface_ao_uv_layout(
        receivers,
        resolution,
        padding_pixels=padding_pixels,
        cancellation_check=cancellation_check,
    )
    return bake_surface_ambient_occlusion(
        layout,
        occluder_meshes,
        cancellation_check=cancellation_check,
    )


# ### Input validation helpers ###
def _normalize_receivers(
    receivers: Sequence[SurfaceAmbientOcclusionReceiver],
) -> tuple[SurfaceAmbientOcclusionReceiver, ...]:
    if isinstance(receivers, (str, bytes, bytearray)) or not isinstance(
        receivers,
        Sequence,
    ):
        raise TypeError("Surface AO receivers must be an ordered sequence.")
    normalized = tuple(receivers)
    if not normalized:
        raise ValueError("Surface AO requires at least one receiver.")
    if not all(
        isinstance(receiver, SurfaceAmbientOcclusionReceiver) for receiver in normalized
    ):
        raise TypeError("Surface AO receivers are invalid.")
    keys = tuple(receiver.key for receiver in normalized)
    if len(keys) != len(set(keys)):
        raise ValueError("Surface AO receiver keys must be unique.")

    face_count = 0
    for receiver in normalized:
        _validate_receiver_mesh(receiver.mesh)
        face_count += len(receiver.mesh.faces)
        if face_count > _MAXIMUM_RECEIVER_FACE_COUNT:
            raise ValueError(
                "Surface AO contains too many receiver faces to unwrap safely."
            )
    return normalized


def _validate_receiver_mesh(mesh: trimesh.Trimesh) -> None:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    uv0 = _get_valid_uv0(mesh)
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or not len(vertices)
        or not np.all(np.isfinite(vertices))
    ):
        raise ValueError("Surface AO receivers require finite 3D vertices.")
    if faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces):
        raise ValueError("Surface AO receivers require triangle faces.")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("A surface AO receiver face references a missing vertex.")
    triangles = vertices[faces]
    doubled_areas = np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    scale = max(float(np.ptp(vertices, axis=0).max()), _GEOMETRY_EPSILON)
    if np.any(doubled_areas <= _GEOMETRY_EPSILON * scale * scale):
        raise ValueError("Surface AO receivers contain degenerate faces.")
    if uv0 is None:
        raise ValueError("Surface AO receivers require finite existing UV0 data.")


def _validate_resolution(resolution: object) -> None:
    if isinstance(resolution, bool) or not isinstance(resolution, (int, np.integer)):
        raise TypeError("The surface AO texture resolution must be an integer.")
    if int(resolution) < _MINIMUM_TEXTURE_RESOLUTION:
        raise ValueError(
            f"The surface AO texture resolution must be at least "
            f"{_MINIMUM_TEXTURE_RESOLUTION} pixels."
        )


def _validate_padding(padding_pixels: object) -> None:
    if isinstance(padding_pixels, bool) or not isinstance(
        padding_pixels,
        (int, np.integer),
    ):
        raise TypeError("Surface AO UV padding must be an integer.")
    if int(padding_pixels) < SURFACE_AO_MINIMUM_PADDING_PIXELS:
        raise ValueError(
            f"Surface AO UV padding must be at least "
            f"{SURFACE_AO_MINIMUM_PADDING_PIXELS} pixels."
        )


def _get_valid_uv0(mesh: trimesh.Trimesh) -> np.ndarray | None:
    raw_uv = getattr(getattr(mesh, "visual", None), "uv", None)
    if raw_uv is None:
        return None
    uv = np.asarray(raw_uv, dtype=float)
    if uv.shape != (len(mesh.vertices), 2) or not np.all(np.isfinite(uv)):
        return None
    return np.ascontiguousarray(uv)


def _normalize_optional_vector(
    raw_value: Sequence[float] | np.ndarray | None,
) -> tuple[float, float, float] | None:
    if raw_value is None:
        return None
    try:
        value = np.asarray(raw_value, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError("Surface AO vectors must contain three values.") from error
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("Surface AO vectors must be finite 3D values.")
    return tuple(float(component) for component in value)


# ### Xatlas packing helpers ###
def _build_best_packing_candidate(
    receivers: tuple[SurfaceAmbientOcclusionReceiver, ...],
    resolution: int,
    requested_padding_pixels: int,
    cancellation_check: Callable[[], bool] | None,
) -> _PackingCandidate:
    packing_meshes = tuple(
        _build_geometric_adjacency_proxy(
            receiver.mesh,
            cancellation_check=cancellation_check,
        )
        for receiver in receivers
    )
    xatlas_meshes = _build_normalized_xatlas_meshes(packing_meshes, resolution)
    failure_reasons: set[str] = set()
    candidate = _search_xatlas_packing_meshes(
        receivers,
        xatlas_meshes,
        resolution=resolution,
        requested_padding_pixels=requested_padding_pixels,
        strategy_prefix="",
        failure_reasons=failure_reasons,
        cancellation_check=cancellation_check,
    )
    if candidate is not None:
        return candidate

    detached_xatlas_meshes = _build_detached_xatlas_meshes(
        xatlas_meshes,
        cancellation_check=cancellation_check,
    )
    candidate = _search_xatlas_packing_meshes(
        receivers,
        detached_xatlas_meshes,
        resolution=resolution,
        requested_padding_pixels=requested_padding_pixels,
        strategy_prefix="detached_faces+",
        failure_reasons=failure_reasons,
        cancellation_check=cancellation_check,
    )
    if candidate is not None:
        return candidate

    details = ""
    if failure_reasons:
        details = " " + " ".join(sorted(failure_reasons))
    raise ValueError("Xatlas could not fit the surface AO UV1 layout." + details)


def _search_xatlas_packing_meshes(
    receivers: tuple[SurfaceAmbientOcclusionReceiver, ...],
    xatlas_meshes: tuple[_PackingMesh, ...],
    *,
    resolution: int,
    requested_padding_pixels: int,
    strategy_prefix: str,
    failure_reasons: set[str],
    cancellation_check: Callable[[], bool] | None,
) -> _PackingCandidate | None:
    """Search bounded packing policies for one prepared topology variant."""

    padding_targets = tuple(
        dict.fromkeys(
            (
                requested_padding_pixels,
                SURFACE_AO_MINIMUM_PADDING_PIXELS,
            )
        )
    )
    best_safe: _PackingCandidate | None = None
    for target_padding in padding_targets:
        internal_padding = target_padding
        previous_effective_padding: float | None = None
        for _attempt in range(_MAXIMUM_PACK_ATTEMPTS_PER_TARGET):
            _check_cancelled(cancellation_check)
            candidate_count = 0
            best_effective_padding = 0.0
            best_accepted: _PackingCandidate | None = None
            for strategy_index, (
                base_strategy_name,
                rotate_charts,
                align_charts,
            ) in enumerate(_PACKING_STRATEGIES):
                strategy_name = f"{strategy_prefix}{base_strategy_name}"
                try:
                    candidate = _build_packing_candidate(
                        receivers,
                        xatlas_meshes=xatlas_meshes,
                        resolution=resolution,
                        internal_padding_pixels=internal_padding,
                        strategy_index=strategy_index,
                        strategy_name=strategy_name,
                        rotate_charts=rotate_charts,
                        align_charts=align_charts,
                        cancellation_check=cancellation_check,
                    )
                except _PackingCandidateError as error:
                    failure_reasons.add(str(error))
                    continue
                candidate_count += 1
                best_effective_padding = max(
                    best_effective_padding,
                    candidate.effective_padding_pixels,
                )
                if (
                    candidate.effective_padding_pixels + _UV_EPSILON
                    >= target_padding
                    and (
                        best_accepted is None
                        or _packing_candidate_score(candidate)
                        > _packing_candidate_score(best_accepted)
                    )
                ):
                    best_accepted = candidate
                if (
                    candidate.effective_padding_pixels + _UV_EPSILON
                    >= SURFACE_AO_MINIMUM_PADDING_PIXELS
                    and (
                        best_safe is None
                        or _safe_packing_candidate_score(candidate)
                        > _safe_packing_candidate_score(best_safe)
                    )
                ):
                    best_safe = candidate
                del candidate
            if candidate_count == 0:
                break
            if best_accepted is not None:
                if target_padding == SURFACE_AO_MINIMUM_PADDING_PIXELS:
                    assert best_safe is not None
                    return best_safe
                return best_accepted
            if (
                previous_effective_padding is not None
                and best_effective_padding
                <= previous_effective_padding + _UV_EPSILON
            ):
                break
            previous_effective_padding = best_effective_padding
            next_internal_padding = _next_internal_padding(
                internal_padding,
                target_padding,
                best_effective_padding,
                resolution,
            )
            if next_internal_padding <= internal_padding:
                break
            internal_padding = next_internal_padding
        if best_safe is not None:
            return best_safe
    return None


def _build_packing_candidate(
    receivers: tuple[SurfaceAmbientOcclusionReceiver, ...],
    *,
    xatlas_meshes: tuple[_PackingMesh, ...] | None = None,
    resolution: int,
    internal_padding_pixels: int,
    strategy_index: int,
    strategy_name: str,
    rotate_charts: bool,
    align_charts: bool,
    cancellation_check: Callable[[], bool] | None,
) -> _PackingCandidate:
    _check_cancelled(cancellation_check)
    if xatlas_meshes is None:
        packing_meshes = tuple(
            _build_geometric_adjacency_proxy(
                receiver.mesh,
                cancellation_check=cancellation_check,
            )
            for receiver in receivers
        )
        xatlas_meshes = _build_normalized_xatlas_meshes(
            packing_meshes,
            resolution,
        )
    if len(xatlas_meshes) != len(receivers):
        raise _PackingCandidateError("A surface AO packing proxy is missing.")
    atlas = xatlas.Atlas()
    for xatlas_mesh in xatlas_meshes:
        _check_cancelled(cancellation_check)
        atlas.add_mesh(
            np.ascontiguousarray(xatlas_mesh.vertices, dtype=np.float32),
            np.ascontiguousarray(xatlas_mesh.faces, dtype=np.uint32),
        )

    chart_options = xatlas.ChartOptions()
    chart_options.fix_winding = False
    chart_options.use_input_mesh_uvs = False
    pack_options = xatlas.PackOptions()
    pack_options.resolution = resolution
    pack_options.padding = internal_padding_pixels
    pack_options.bilinear = True
    pack_options.blockAlign = False
    pack_options.bruteForce = False
    pack_options.create_image = False
    pack_options.rotate_charts = rotate_charts
    pack_options.rotate_charts_to_axis = align_charts
    _check_cancelled(cancellation_check)
    try:
        atlas.generate(chart_options, pack_options, False)
    except RuntimeError as error:
        raise _PackingCandidateError(
            f"Xatlas surface AO policy {strategy_name!r} failed."
        ) from error
    _check_cancelled(cancellation_check)
    if int(atlas.atlas_count) != 1:
        raise _PackingCandidateError(
            f"Xatlas surface AO policy {strategy_name!r} required multiple atlases."
        )

    atlas_width = int(atlas.width)
    atlas_height = int(atlas.height)
    if atlas_width <= 0 or atlas_height <= 0:
        raise _PackingCandidateError("Xatlas returned empty surface AO dimensions.")
    packed_receivers: list[_PackedReceiver] = []
    uv_triangle_area = 0.0
    for receiver_index, (receiver, xatlas_mesh) in enumerate(
        zip(receivers, xatlas_meshes, strict=True)
    ):
        _check_cancelled(cancellation_check)
        proxy_vertex_mapping, proxy_faces, proxy_uv1 = atlas[receiver_index]
        proxy_vertex_mapping = np.ascontiguousarray(
            proxy_vertex_mapping,
            dtype=np.int64,
        )
        proxy_faces = np.ascontiguousarray(proxy_faces, dtype=np.int64)
        proxy_uv1 = np.ascontiguousarray(proxy_uv1, dtype=np.float64)
        source_faces = np.asarray(receiver.mesh.faces, dtype=np.int64)
        if proxy_faces.shape != source_faces.shape:
            raise _PackingCandidateError(
                "Xatlas changed a surface AO receiver's face count."
            )
        if (
            proxy_vertex_mapping.ndim != 1
            or proxy_uv1.shape != (len(proxy_vertex_mapping), 2)
            or np.any(proxy_faces < 0)
            or np.any(proxy_faces >= len(proxy_vertex_mapping))
            or np.any(proxy_vertex_mapping < 0)
            or np.any(proxy_vertex_mapping >= len(xatlas_mesh.vertices))
            or not np.all(np.isfinite(proxy_uv1))
        ):
            raise _PackingCandidateError(
                "Xatlas returned an invalid surface AO receiver layout."
            )
        if not np.array_equal(
            proxy_vertex_mapping[proxy_faces],
            xatlas_mesh.faces,
        ):
            raise _PackingCandidateError(
                "Xatlas changed a surface AO receiver's face topology."
            )
        if np.any(proxy_uv1 < -_UV_EPSILON) or np.any(
            proxy_uv1 > 1.0 + _UV_EPSILON
        ):
            raise _PackingCandidateError(
                "Xatlas placed surface AO UVs outside the texture atlas."
            )
        vertex_mapping, faces, uv1 = _restore_source_vertex_mapping(
            xatlas_mesh,
            proxy_faces,
            np.ascontiguousarray(np.clip(proxy_uv1, 0.0, 1.0)),
        )
        if not np.array_equal(vertex_mapping[faces], source_faces):
            raise _PackingCandidateError(
                "Surface AO could not restore the authored face topology."
            )
        triangles = uv1[faces]
        vectors_a = triangles[:, 1] - triangles[:, 0]
        vectors_b = triangles[:, 2] - triangles[:, 0]
        areas = (
            np.abs(
                vectors_a[:, 0] * vectors_b[:, 1] - vectors_a[:, 1] * vectors_b[:, 0]
            )
            * 0.5
        )
        if np.any(areas <= _GEOMETRY_EPSILON):
            raise _PackingCandidateError(
                "Xatlas returned a degenerate surface AO UV chart."
            )
        uv_triangle_area += float(np.sum(areas))
        packed_receivers.append(
            _PackedReceiver(
                vertex_mapping=vertex_mapping,
                faces=faces,
                uv1=uv1,
            )
        )

    effective_padding = min(
        float(internal_padding_pixels) * float(resolution) / float(atlas_width),
        float(internal_padding_pixels) * float(resolution) / float(atlas_height),
    )
    return _PackingCandidate(
        receivers=tuple(packed_receivers),
        internal_padding_pixels=internal_padding_pixels,
        effective_padding_pixels=effective_padding,
        atlas_width=atlas_width,
        atlas_height=atlas_height,
        packing_strategy=strategy_name,
        uv_triangle_area=uv_triangle_area,
        strategy_index=strategy_index,
    )


def _build_geometric_adjacency_proxy(
    mesh: trimesh.Trimesh,
    *,
    cancellation_check: Callable[[], bool] | None,
) -> _PackingMesh:
    """Reconnect exact duplicate corners only across safe manifold edges.

    UV projection may duplicate every corner even when the geometry remains a
    connected surface. Two face edges are joined only when they are the sole
    uses of the same exact geometric edge and have opposite winding. This
    does not newly weld coincident duplicate sheets, non-manifold edges, or
    nearby wafer layers together.
    """

    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    source_faces = np.ascontiguousarray(mesh.faces, dtype=np.int64)
    _check_cancelled(cancellation_check)
    _unique_positions, position_groups = np.unique(
        vertices,
        axis=0,
        return_inverse=True,
    )
    position_groups = np.ascontiguousarray(position_groups, dtype=np.int64)
    parents = np.arange(len(vertices), dtype=np.int64)

    def find(vertex_index: int) -> int:
        root = int(vertex_index)
        while int(parents[root]) != root:
            root = int(parents[root])
        while int(parents[vertex_index]) != vertex_index:
            next_index = int(parents[vertex_index])
            parents[vertex_index] = root
            vertex_index = next_index
        return root

    def union(first_index: int, second_index: int) -> None:
        first_root = find(first_index)
        second_root = find(second_index)
        if first_root == second_root:
            return
        if first_root < second_root:
            parents[second_root] = first_root
        else:
            parents[first_root] = second_root

    edge_uses: dict[
        tuple[int, int],
        list[tuple[int, int, int, int, int]],
    ] = {}
    for face_index, face in enumerate(source_faces):
        if face_index % 65_536 == 0:
            _check_cancelled(cancellation_check)
        for first_corner, second_corner in ((0, 1), (1, 2), (2, 0)):
            first_vertex = int(face[first_corner])
            second_vertex = int(face[second_corner])
            first_group = int(position_groups[first_vertex])
            second_group = int(position_groups[second_vertex])
            edge_key = (
                (first_group, second_group)
                if first_group < second_group
                else (second_group, first_group)
            )
            uses = edge_uses.setdefault(edge_key, [])
            if len(uses) <= 2:
                uses.append(
                    (
                        face_index,
                        first_vertex,
                        second_vertex,
                        first_group,
                        second_group,
                    )
                )

    for edge_index, uses in enumerate(edge_uses.values()):
        if edge_index % 65_536 == 0:
            _check_cancelled(cancellation_check)
        if len(uses) != 2:
            continue
        first_use, second_use = uses
        first_face_groups = np.sort(position_groups[source_faces[first_use[0]]])
        second_face_groups = np.sort(position_groups[source_faces[second_use[0]]])
        if (
            first_use[0] == second_use[0]
            or np.array_equal(first_face_groups, second_face_groups)
            or first_use[3] != second_use[4]
            or first_use[4] != second_use[3]
        ):
            continue
        union(first_use[1], second_use[2])
        union(first_use[2], second_use[1])

    proxy_index_by_root: dict[int, int] = {}
    proxy_vertices: list[np.ndarray] = []
    source_to_proxy = np.empty(len(vertices), dtype=np.int64)
    for source_index in range(len(vertices)):
        root = find(source_index)
        proxy_index = proxy_index_by_root.get(root)
        if proxy_index is None:
            proxy_index = len(proxy_vertices)
            proxy_index_by_root[root] = proxy_index
            proxy_vertices.append(vertices[source_index])
        source_to_proxy[source_index] = proxy_index
    proxy_faces = np.ascontiguousarray(source_to_proxy[source_faces], dtype=np.int64)
    _check_cancelled(cancellation_check)
    return _PackingMesh(
        vertices=np.ascontiguousarray(proxy_vertices, dtype=np.float64),
        faces=proxy_faces,
        source_faces=source_faces,
    )


def _build_normalized_xatlas_meshes(
    packing_meshes: tuple[_PackingMesh, ...],
    resolution: int,
) -> tuple[_PackingMesh, ...]:
    """Normalize coordinates and isolate faces too thin for xatlas safely."""

    extents = tuple(
        float(np.ptp(packing_mesh.vertices, axis=0).max())
        for packing_mesh in packing_meshes
    )
    common_scale = max((*extents, _GEOMETRY_EPSILON))
    normalized: list[_PackingMesh] = []
    for packing_mesh in packing_meshes:
        vertices = np.asarray(packing_mesh.vertices, dtype=np.float64)
        minimum = np.min(vertices, axis=0)
        maximum = np.max(vertices, axis=0)
        center = minimum + (maximum - minimum) * 0.5
        normalized_vertices = np.ascontiguousarray(
            (vertices - center) / common_scale,
            dtype=np.float32,
        )
        normalized_faces = np.asarray(packing_mesh.faces, dtype=np.int64).copy()
        triangles = np.asarray(
            normalized_vertices[normalized_faces],
            dtype=np.float64,
        )
        edge_lengths = np.stack(
            (
                np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
                np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
                np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
            ),
            axis=1,
        )
        longest_edges = np.max(edge_lengths, axis=1)
        doubled_areas = np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            ),
            axis=1,
        )
        altitudes = np.divide(
            doubled_areas,
            longest_edges,
            out=np.zeros_like(doubled_areas),
            where=longest_edges > _GEOMETRY_EPSILON,
        )
        minimum_altitude = max(
            _XATLAS_MINIMUM_FACE_ALTITUDE_PIXELS / float(resolution),
            _XATLAS_MINIMUM_NORMALIZED_FACE_ALTITUDE,
        )
        proxy_edge_length = _minimum_proxy_face_edge_length(resolution)
        minimum_doubled_area = (
            math.sqrt(3.0) * 0.5 * proxy_edge_length * proxy_edge_length
        )
        unstable_faces = np.flatnonzero(
            ~np.isfinite(altitudes)
            | (altitudes < minimum_altitude)
            | (doubled_areas < minimum_doubled_area)
        )
        if len(unstable_faces):
            proxy_vertices = _build_minimum_area_proxy_triangles(
                len(unstable_faces),
                resolution,
            )
            first_proxy_vertex = len(normalized_vertices)
            normalized_vertices = np.ascontiguousarray(
                np.vstack((normalized_vertices, proxy_vertices.reshape((-1, 3)))),
                dtype=np.float32,
            )
            normalized_faces[unstable_faces] = (
                first_proxy_vertex
                + np.arange(len(unstable_faces) * 3, dtype=np.int64).reshape((-1, 3))
            )
        normalized.append(
            _PackingMesh(
                vertices=normalized_vertices,
                faces=np.ascontiguousarray(normalized_faces, dtype=np.int64),
                source_faces=packing_mesh.source_faces,
            )
        )
    return tuple(normalized)


def _build_detached_xatlas_meshes(
    xatlas_meshes: tuple[_PackingMesh, ...],
    *,
    cancellation_check: Callable[[], bool] | None,
) -> tuple[_PackingMesh, ...]:
    """Detach every stable triangle as a last-resort topology repair."""

    detached_meshes: list[_PackingMesh] = []
    for xatlas_mesh in xatlas_meshes:
        _check_cancelled(cancellation_check)
        triangles = np.asarray(
            xatlas_mesh.vertices[xatlas_mesh.faces],
            dtype=np.float32,
        )
        vertices = np.ascontiguousarray(triangles.reshape((-1, 3)))
        faces = np.arange(len(vertices), dtype=np.int64).reshape((-1, 3))
        detached_meshes.append(
            _PackingMesh(
                vertices=vertices,
                faces=np.ascontiguousarray(faces),
                source_faces=xatlas_mesh.source_faces,
            )
        )
    _check_cancelled(cancellation_check)
    return tuple(detached_meshes)


def _build_minimum_area_proxy_triangles(
    face_count: int,
    resolution: int,
) -> np.ndarray:
    """Create isolated stable charts for subpixel geometric sliver faces."""

    edge_length = _minimum_proxy_face_edge_length(resolution)
    column_count = max(1, math.ceil(math.sqrt(face_count)))
    face_indices = np.arange(face_count, dtype=np.int64)
    bases = np.zeros((face_count, 3), dtype=np.float64)
    bases[:, 0] = 2.0 + (face_indices % column_count) * edge_length * 2.0
    bases[:, 1] = (face_indices // column_count) * edge_length * 2.0
    bases[:, 2] = 2.0
    template = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (edge_length, 0.0, 0.0),
            (edge_length * 0.5, edge_length * math.sqrt(3.0) * 0.5, 0.0),
        ),
        dtype=np.float64,
    )
    return np.ascontiguousarray(
        bases[:, np.newaxis, :] + template[np.newaxis, :, :],
        dtype=np.float32,
    )


def _minimum_proxy_face_edge_length(resolution: int) -> float:
    """Return a chart edge large enough for stable xatlas parameterization."""

    return max(
        _XATLAS_PROXY_FACE_EDGE_PIXELS / float(resolution),
        _XATLAS_MINIMUM_PROXY_FACE_EDGE,
    )


def _restore_source_vertex_mapping(
    packing_mesh: _PackingMesh,
    proxy_faces: np.ndarray,
    proxy_uv1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Restore source UV0 seam vertices without breaking the new UV1 charts."""

    atlas_corner_indices = np.asarray(proxy_faces, dtype=np.int64).reshape(-1)
    source_corner_indices = np.asarray(
        packing_mesh.source_faces,
        dtype=np.int64,
    ).reshape(-1)
    vertex_pairs = np.column_stack((atlas_corner_indices, source_corner_indices))
    unique_pairs, restored_corner_indices = np.unique(
        vertex_pairs,
        axis=0,
        return_inverse=True,
    )
    restored_vertex_mapping = np.ascontiguousarray(
        unique_pairs[:, 1],
        dtype=np.int64,
    )
    restored_faces = np.ascontiguousarray(
        restored_corner_indices.reshape(packing_mesh.source_faces.shape),
        dtype=np.int64,
    )
    restored_uv1 = np.ascontiguousarray(
        proxy_uv1[unique_pairs[:, 0]],
        dtype=np.float64,
    )
    return restored_vertex_mapping, restored_faces, restored_uv1


def _packing_candidate_score(
    candidate: _PackingCandidate,
) -> tuple[float, float, int, int, int, int]:
    return (
        candidate.uv_triangle_area,
        candidate.effective_padding_pixels,
        -(candidate.atlas_width * candidate.atlas_height),
        -max(candidate.atlas_width, candidate.atlas_height),
        -candidate.internal_padding_pixels,
        -candidate.strategy_index,
    )


def _safe_packing_candidate_score(
    candidate: _PackingCandidate,
) -> tuple[float, float, int, int, int, int]:
    """Prefer the widest safe gutter before UV occupancy for a fallback."""

    return (
        candidate.effective_padding_pixels,
        candidate.uv_triangle_area,
        -(candidate.atlas_width * candidate.atlas_height),
        -max(candidate.atlas_width, candidate.atlas_height),
        -candidate.internal_padding_pixels,
        -candidate.strategy_index,
    )


def _next_internal_padding(
    current_padding: int,
    target_padding: int,
    effective_padding: float,
    resolution: int,
) -> int:
    """Increase xatlas padding gradually and keep every retry bounded."""

    proportional_padding = math.ceil(
        float(current_padding)
        * float(target_padding)
        / max(effective_padding, np.finfo(np.float64).eps)
    )
    increment_limit = current_padding + max(1, current_padding // 2)
    maximum_padding = max(
        target_padding,
        min(resolution // 4, target_padding * 8),
    )
    return min(
        max(current_padding + 1, proportional_padding),
        increment_limit,
        maximum_padding,
    )


# ### AO target helpers ###
def _build_bake_target(
    receiver: SurfaceAmbientOcclusionReceiverLayout,
    resolution: int,
) -> ObjectAmbientOcclusionTarget:
    mesh = trimesh.Trimesh(
        vertices=np.asarray(receiver.world_vertices, dtype=float),
        faces=np.asarray(receiver.faces, dtype=np.int64),
        visual=TextureVisuals(
            uv=np.asarray(receiver.uv1, dtype=float),
            material=None,
        ),
        process=False,
    )
    mesh.metadata["housemaker_surface_ao_receiver_key"] = receiver.key
    return ObjectAmbientOcclusionTarget(
        mesh=mesh,
        atlas_pixel_bounds=_receiver_pixel_bounds(receiver, resolution),
        mirror_plane_point=receiver.mirror_plane_point,
        mirror_plane_normal=receiver.mirror_plane_normal,
    )


def _receiver_pixel_bounds(
    receiver: SurfaceAmbientOcclusionReceiverLayout,
    resolution: int,
) -> tuple[int, int, int, int]:
    """Return the smallest padded Atlas rectangle containing one receiver.

    Object AO rasterization works in pixel-center coordinates, where the Atlas
    edges are ``-0.5`` and ``resolution - 0.5``. Keeping that convention here
    avoids allocating and dilating a full-resolution image for every globally
    packed receiver while retaining the exact edge pixels and coverage gutter.
    """

    pixel_coordinates = object_ao_baking._uv_to_top_origin_pixels(
        np.asarray(receiver.uv1, dtype=float),
        resolution,
    )
    minimum = np.min(pixel_coordinates, axis=0)
    maximum = np.max(pixel_coordinates, axis=0)
    padding = int(object_ao_baking.OBJECT_AO_COVERAGE_PADDING_PIXELS)
    left = max(0, math.floor(float(minimum[0]) + 0.5) - padding)
    top = max(0, math.floor(float(minimum[1]) + 0.5) - padding)
    right = min(resolution, math.ceil(float(maximum[0]) + 0.5) + padding)
    bottom = min(resolution, math.ceil(float(maximum[1]) + 0.5) + padding)
    if right <= left or bottom <= top:
        raise ValueError("A surface AO receiver has empty pixel bounds.")
    return left, top, right, bottom


# ### Signature helpers ###
def _build_layout_signature(
    receiver_layouts: tuple[SurfaceAmbientOcclusionReceiverLayout, ...],
    *,
    resolution: int,
    requested_padding_pixels: int,
    candidate: _PackingCandidate,
    cancellation_check: Callable[[], bool] | None,
) -> str:
    _check_cancelled(cancellation_check)
    digest = hashlib.sha256()
    digest.update(_SIGNATURE_PREFIX)
    _update_signature_integer(digest, SURFACE_AO_BAKE_VERSION)
    _update_signature_text(digest, str(getattr(xatlas, "__version__", "unknown")))
    _update_signature_integer(digest, resolution)
    _update_signature_integer(digest, requested_padding_pixels)
    _update_signature_integer(digest, candidate.internal_padding_pixels)
    _update_signature_integer(digest, candidate.atlas_width)
    _update_signature_integer(digest, candidate.atlas_height)
    _update_signature_text(digest, candidate.packing_strategy)
    _update_signature_integer(digest, len(receiver_layouts))
    for layout in receiver_layouts:
        _check_cancelled(cancellation_check)
        _update_signature_text(digest, layout.key)
        _update_signature_array(
            digest,
            layout.source_world_vertices,
            "<f8",
            cancellation_check=cancellation_check,
        )
        _update_signature_array(
            digest,
            layout.vertex_mapping,
            "<i8",
            cancellation_check=cancellation_check,
        )
        _update_signature_array(
            digest,
            layout.faces,
            "<i8",
            cancellation_check=cancellation_check,
        )
        _update_signature_array(
            digest,
            layout.uv1,
            "<f8",
            cancellation_check=cancellation_check,
        )
        _update_signature_optional_vector(
            digest,
            layout.mirror_plane_point,
            cancellation_check=cancellation_check,
        )
        _update_signature_optional_vector(
            digest,
            layout.mirror_plane_normal,
            cancellation_check=cancellation_check,
        )
    _check_cancelled(cancellation_check)
    return digest.hexdigest()


def _build_geometry_signature_from_occluder(
    layout: SurfaceAmbientOcclusionUvLayout,
    occluder: trimesh.Trimesh | None,
    *,
    cancellation_check: Callable[[], bool] | None = None,
) -> str:
    _check_cancelled(cancellation_check)
    digest = hashlib.sha256()
    digest.update(_SIGNATURE_PREFIX)
    digest.update(b"geometry\0")
    _update_signature_text(digest, layout.layout_signature)
    _update_signature_integer(digest, object_ao_baking.OBJECT_AO_RAY_COUNT)
    _update_signature_float(
        digest,
        object_ao_baking.OBJECT_AO_MAX_DISTANCE_METERS,
        cancellation_check=cancellation_check,
    )
    _update_signature_integer(
        digest,
        object_ao_baking.OBJECT_AO_COVERAGE_PADDING_PIXELS,
    )
    if occluder is None:
        digest.update(b"no-occluders\0")
    else:
        _update_signature_array(
            digest,
            occluder.vertices,
            "<f8",
            cancellation_check=cancellation_check,
        )
        _update_signature_array(
            digest,
            occluder.faces,
            "<i8",
            cancellation_check=cancellation_check,
        )
        _update_signature_float(
            digest,
            object_ao_baking._build_scale_aware_ray_bias(occluder),
            cancellation_check=cancellation_check,
        )
    _check_cancelled(cancellation_check)
    return digest.hexdigest()


def _update_signature_text(digest: object, value: str) -> None:
    encoded = str(value).encode("utf-8")
    _update_signature_integer(digest, len(encoded))
    digest.update(encoded)


def _update_signature_integer(digest: object, value: int) -> None:
    digest.update(int(value).to_bytes(8, "little", signed=True))


def _update_signature_float(
    digest: object,
    value: float,
    *,
    cancellation_check: Callable[[], bool] | None = None,
) -> None:
    _update_signature_array(
        digest,
        np.asarray((float(value),)),
        "<f8",
        cancellation_check=cancellation_check,
    )


def _update_signature_optional_vector(
    digest: object,
    value: tuple[float, float, float] | None,
    *,
    cancellation_check: Callable[[], bool] | None = None,
) -> None:
    if value is None:
        digest.update(b"none\0")
        return
    digest.update(b"vector\0")
    _update_signature_array(
        digest,
        np.asarray(value),
        "<f8",
        cancellation_check=cancellation_check,
    )


def _update_signature_array(
    digest: object,
    values: np.ndarray,
    dtype: str,
    *,
    cancellation_check: Callable[[], bool] | None = None,
) -> None:
    _check_cancelled(cancellation_check)
    array = np.ascontiguousarray(values, dtype=np.dtype(dtype))
    _update_signature_integer(digest, array.ndim)
    for dimension in array.shape:
        _update_signature_integer(digest, int(dimension))
    byte_view = memoryview(array).cast("B")
    for start in range(0, len(byte_view), _SIGNATURE_CHUNK_BYTE_COUNT):
        _check_cancelled(cancellation_check)
        digest.update(byte_view[start : start + _SIGNATURE_CHUNK_BYTE_COUNT])
    _check_cancelled(cancellation_check)


# ### General helpers ###
def _check_cancelled(cancellation_check: Callable[[], bool] | None) -> None:
    if cancellation_check is not None and bool(cancellation_check()):
        raise SurfaceAmbientOcclusionBakeCancelled(
            "Surface ambient-occlusion baking was cancelled."
        )


def _immutable_array(values: np.ndarray, *, dtype: np.dtype | type) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


# ### Public exports ###
__all__ = [
    "SURFACE_AO_BAKE_VERSION",
    "SURFACE_AO_DEFAULT_PADDING_PIXELS",
    "SURFACE_AO_MINIMUM_PADDING_PIXELS",
    "SurfaceAmbientOcclusionBakeCancelled",
    "SurfaceAmbientOcclusionBakeResult",
    "SurfaceAmbientOcclusionReceiver",
    "SurfaceAmbientOcclusionReceiverLayout",
    "SurfaceAmbientOcclusionUvLayout",
    "bake_surface_ambient_occlusion",
    "build_and_bake_surface_ambient_occlusion",
    "build_surface_ao_geometry_signature",
    "build_surface_ao_uv_layout",
]
