# ### Imports ###
from __future__ import annotations

import copy
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from io import BytesIO
from typing import Literal

import numpy as np
import trimesh
import xatlas
from trimesh.visual.color import ColorVisuals
from trimesh.visual.material import MultiMaterial, PBRMaterial, SimpleMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.object_symmetry import SYMMETRIC_LEFT_PACKED_UV_SAFETY_INSET


# ### Public constants ###
VISIBILITY_UV_UNWRAP_VERSION = 4
UV_TARGET_DOMAIN_FULL = "full"
UV_TARGET_DOMAIN_LEFT_HALF = "left_half"


# ### Configuration constants ###
DEFAULT_TEXTURE_RESOLUTION = 2048
DEFAULT_EXTERIOR_UV_SHARE = 0.95
DEFAULT_GUTTER_PIXELS = 8
DEFAULT_RAYCAST_RESOLUTION = 96
DEFAULT_MAX_FACE_COUNT = 15_000
_UV_TARGET_DOMAINS = frozenset(
    {UV_TARGET_DOMAIN_FULL, UV_TARGET_DOMAIN_LEFT_HALF}
)
_LEFT_HALF_MINIMUM_XATLAS_GUTTER_PIXELS = 17
_REFERENCE_GEOMETRY_RELATIVE_TOLERANCE = 1e-7
_XATLAS_PACKING_STRATEGIES = (
    ("fixed", False, False),
    ("rotate_charts", True, False),
    ("align_charts", False, True),
    ("rotate_and_align_charts", True, True),
)
_RELAXED_CHART_MAX_COST = 4.0
_RELAXED_CHART_STRAIGHTNESS_WEIGHT = 0.0
_RELAXED_CHART_OCCUPANCY_THRESHOLD = 0.70
_RELAXED_CHART_MAX_FACE_RATIO = 0.25
_TRUE_ASPECT_PACKING_STRATEGY_SUFFIX = "true_aspect_maxrects"
_TRUE_ASPECT_DIRECT_OCCUPANCY_THRESHOLD = 0.72
_TRUE_ASPECT_MINIMUM_CHART_COUNT = 24
_TRUE_ASPECT_MAXIMUM_CHART_COUNT = 256
_TRUE_ASPECT_PACKING_ORDER_POLICIES = (
    "area",
    "long_side",
    "perimeter",
)
_TRUE_ASPECT_BINARY_SEARCH_STEPS = 12
_TRUE_ASPECT_MAXIMUM_FREE_RECTANGLES = 512
_CAMERA_DIRECTIONS = (
    (-1.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, 0.0, 1.0),
    (-1.0, -1.0, -1.0),
    (-1.0, -1.0, 1.0),
    (-1.0, 1.0, -1.0),
    (-1.0, 1.0, 1.0),
    (1.0, -1.0, -1.0),
    (1.0, -1.0, 1.0),
    (1.0, 1.0, -1.0),
    (1.0, 1.0, 1.0),
)
_AREA_EPSILON = 1e-14
_NORMAL_EPSILON = 1e-12
_DEPTH_EPSILON = 1e-10
_BARYCENTRIC_EPSILON = 1e-9
_CANCELLATION_CHECK_INTERVAL = 128
_MAX_GUTTER_PACK_ATTEMPTS = 4


# ### Public data models ###
VisibilityUvTargetDomain = Literal["full", "left_half"]


@dataclass(frozen=True)
class VisibilityUvUnwrapStats:
    """Auditable measurements produced by visibility-optimized unwrapping."""

    face_count: int
    instance_face_count: int
    chart_count: int
    exterior_face_count: int
    hidden_face_count: int
    camera_count: int
    ray_sample_count: int
    texture_resolution: int
    gutter_pixels: int
    effective_gutter_pixels: float
    atlas_width: int
    atlas_height: int
    atlas_utilization: float
    requested_exterior_uv_share: float
    achieved_exterior_uv_share: float
    uv_triangle_occupancy: float
    exterior_face_indices: tuple[int, ...]
    visibility_hits: tuple[int, ...]
    target_domain: str = UV_TARGET_DOMAIN_FULL
    effective_horizontal_gutter_pixels: float | None = None
    effective_vertical_gutter_pixels: float | None = None
    packing_strategy: str = "rotate_and_align_charts"

    def __post_init__(self) -> None:
        if self.effective_horizontal_gutter_pixels is None:
            object.__setattr__(
                self,
                "effective_horizontal_gutter_pixels",
                float(self.effective_gutter_pixels),
            )
        if self.effective_vertical_gutter_pixels is None:
            object.__setattr__(
                self,
                "effective_vertical_gutter_pixels",
                float(self.effective_gutter_pixels),
            )


@dataclass(frozen=True)
class VisibilityUvUnwrapResult:
    """An untextured GLB with new UVs and the measurements behind them."""

    glb_bytes: bytes
    stats: VisibilityUvUnwrapStats


class VisibilityUvUnwrapCancelled(RuntimeError):
    """Raised when a caller cancels visibility-optimized unwrapping."""


# ### Internal exceptions ###
class _AtlasCandidatePackingError(RuntimeError):
    """One policy failed without invalidating the remaining policies."""


class _AtlasPolicyPackingError(RuntimeError):
    """One complete xatlas charting policy could not produce an atlas."""


class _TrueAspectPackingError(RuntimeError):
    """The optional true-aspect half-atlas candidate could not be packed."""


# ### Internal data models ###
@dataclass(frozen=True)
class _GeometryData:
    name: object
    mesh: trimesh.Trimesh
    world_vertices: np.ndarray
    exterior_faces: np.ndarray
    visibility_hits: np.ndarray


@dataclass(frozen=True)
class _SceneData:
    geometries: tuple[_GeometryData, ...]
    instance_triangles: np.ndarray
    instance_face_references: tuple[tuple[object, int], ...]
    face_count: int

    @property
    def instance_face_count(self) -> int:
        return int(len(self.instance_triangles))


@dataclass(frozen=True)
class _AtlasSubmission:
    submission_id: int
    geometry_name: object
    source_face_indices: np.ndarray
    source_vertices: np.ndarray
    source_faces: np.ndarray
    exterior: bool


@dataclass(frozen=True)
class _AtlasOutput:
    submission: _AtlasSubmission
    vertex_mapping: np.ndarray
    faces: np.ndarray
    uvs: np.ndarray


@dataclass(frozen=True)
class _GeneratedAtlas:
    atlas: xatlas.Atlas
    outputs: tuple[_AtlasOutput, ...]
    effective_gutter_pixels: float
    packing_gutter_pixels: int
    packing_strategy: str


@dataclass(frozen=True)
class _AtlasPackCandidate:
    generated: _GeneratedAtlas
    triangle_occupancy: float
    utilization: float
    strategy_index: int


@dataclass(frozen=True)
class _TargetDomainLayout:
    outputs: tuple[_AtlasOutput, ...]
    chart_count: int
    layout_width: int
    layout_height: int
    utilization: float
    horizontal_gutter_pixels: float
    vertical_gutter_pixels: float
    packing_strategy: str


@dataclass(frozen=True)
class _ConnectedUvChart:
    chart_index: int
    output_index: int
    vertex_indices: np.ndarray
    minimum_pixels: np.ndarray
    extent_pixels: np.ndarray


@dataclass(frozen=True)
class _PackingRectangle:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class _ChartPlacement:
    rectangle: _PackingRectangle
    rotated: bool


@dataclass(frozen=True)
class _TrueAspectPackResult:
    scale: float
    placements: tuple[_ChartPlacement, ...]
    order_policy: str


# ### Public API ###
def unwrap_object_uvs_by_visibility(
    glb_bytes: bytes,
    *,
    texture_resolution: int = DEFAULT_TEXTURE_RESOLUTION,
    exterior_uv_share: float = DEFAULT_EXTERIOR_UV_SHARE,
    gutter_pixels: int = DEFAULT_GUTTER_PIXELS,
    raycast_resolution: int = DEFAULT_RAYCAST_RESOLUTION,
    max_face_count: int = DEFAULT_MAX_FACE_COUNT,
    cancellation_check: Callable[[], bool] | None = None,
    target_domain: VisibilityUvTargetDomain = UV_TARGET_DOMAIN_FULL,
    visibility_reference_glb: bytes | None = None,
) -> VisibilityUvUnwrapResult:
    """Create packed UVs weighted toward faces visible from outside.

    A deterministic orthographic ray grid classifies exterior faces. Xatlas
    then unwraps and packs exterior and hidden geometry together, with their
    input scales chosen to reserve approximately 95% and 5% of UV triangle
    area respectively. Xatlas may use a non-square working atlas; its UVs are
    normalized before the later square texture is generated. ``left_half``
    maps UVs into U 0..0.5 for symmetric pair atlases. Fragmented, sparse
    layouts are conditionally repacked into that true-aspect region with one
    uniform chart scale. An optional visibility reference may contain matching
    target geometry plus occluders; only hits belonging to strictly matching
    target geometry are transferred. This must run before an object receives
    textures.
    """

    _validate_options(
        texture_resolution=texture_resolution,
        exterior_uv_share=exterior_uv_share,
        gutter_pixels=gutter_pixels,
        raycast_resolution=raycast_resolution,
        max_face_count=max_face_count,
        target_domain=target_domain,
    )
    _check_cancelled(cancellation_check)
    scene = _load_scene(glb_bytes)
    scene_data = _collect_scene_data(scene, max_face_count=max_face_count)
    _validate_untextured_materials(scene_data.geometries)
    _check_cancelled(cancellation_check)

    visibility_scene_data = scene_data
    if visibility_reference_glb is not None:
        reference_scene = _load_visibility_reference(visibility_reference_glb)
        visibility_scene_data = _collect_scene_data(
            reference_scene,
            max_face_count=max_face_count * 2,
        )
        _validate_visibility_reference(
            scene,
            scene_data,
            reference_scene,
            visibility_scene_data,
        )
        _check_cancelled(cancellation_check)

    instance_hits, ray_sample_count = _measure_exterior_visibility(
        visibility_scene_data.instance_triangles,
        raycast_resolution=raycast_resolution,
        cancellation_check=cancellation_check,
    )
    measured_geometries = _assign_geometry_visibility(
        visibility_scene_data,
        instance_hits,
    )
    geometries = (
        measured_geometries
        if visibility_reference_glb is None
        else _transfer_reference_visibility(
            scene_data.geometries,
            measured_geometries,
        )
    )
    submissions = _build_atlas_submissions(
        geometries,
        exterior_uv_share=exterior_uv_share,
    )
    packing_gutter_pixels = _resolve_packing_gutter_pixels(
        gutter_pixels,
        target_domain=target_domain,
    )
    generated_candidates = _generate_atlas_candidates(
        submissions,
        texture_resolution=texture_resolution,
        gutter_pixels=packing_gutter_pixels,
        cancellation_check=cancellation_check,
    )
    target_layout = _build_best_target_domain_layout(
        generated_candidates,
        texture_resolution=texture_resolution,
        gutter_pixels=gutter_pixels,
        target_domain=target_domain,
        cancellation_check=cancellation_check,
    )
    target_outputs = target_layout.outputs
    output_scene = _apply_atlas_uvs(scene, geometries, target_outputs)
    output_glb = _export_scene(output_scene)
    _validate_output(
        output_glb,
        scene_data.face_count,
        target_domain=target_domain,
    )

    achieved_share, triangle_occupancy = _measure_uv_area(
        target_outputs,
        target_domain=target_domain,
    )
    ordered_hits = np.concatenate(
        [geometry.visibility_hits for geometry in geometries]
    )
    exterior_indices = tuple(
        int(index) for index in np.flatnonzero(ordered_hits > 0)
    )
    return VisibilityUvUnwrapResult(
        glb_bytes=output_glb,
        stats=VisibilityUvUnwrapStats(
            face_count=scene_data.face_count,
            instance_face_count=len(scene_data.instance_triangles),
            chart_count=target_layout.chart_count,
            exterior_face_count=len(exterior_indices),
            hidden_face_count=scene_data.face_count - len(exterior_indices),
            camera_count=len(_CAMERA_DIRECTIONS),
            ray_sample_count=ray_sample_count,
            texture_resolution=texture_resolution,
            gutter_pixels=gutter_pixels,
            effective_gutter_pixels=min(
                target_layout.horizontal_gutter_pixels,
                target_layout.vertical_gutter_pixels,
            ),
            atlas_width=target_layout.layout_width,
            atlas_height=target_layout.layout_height,
            atlas_utilization=target_layout.utilization,
            requested_exterior_uv_share=exterior_uv_share,
            achieved_exterior_uv_share=achieved_share,
            uv_triangle_occupancy=triangle_occupancy,
            exterior_face_indices=exterior_indices,
            visibility_hits=tuple(int(value) for value in ordered_hits),
            target_domain=target_domain,
            effective_horizontal_gutter_pixels=(
                target_layout.horizontal_gutter_pixels
            ),
            effective_vertical_gutter_pixels=(
                target_layout.vertical_gutter_pixels
            ),
            packing_strategy=target_layout.packing_strategy,
        ),
    )


# ### Input validation ###
def _validate_options(
    *,
    texture_resolution: int,
    exterior_uv_share: float,
    gutter_pixels: int,
    raycast_resolution: int,
    max_face_count: int,
    target_domain: object,
) -> None:
    integer_options = {
        "texture resolution": texture_resolution,
        "gutter": gutter_pixels,
        "raycast resolution": raycast_resolution,
        "maximum face count": max_face_count,
    }
    for label, value in integer_options.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"The {label} must be an integer.")
    if texture_resolution < 64:
        raise ValueError("The texture resolution must be at least 64 pixels.")
    minimum_gutter = max(1, math.ceil(texture_resolution / 512.0))
    if gutter_pixels < minimum_gutter:
        raise ValueError(
            "The UV gutter is too small for the selected texture resolution."
        )
    if 2 * gutter_pixels + 1 >= texture_resolution:
        raise ValueError("The UV gutter leaves no room for UV charts.")
    if raycast_resolution < 16:
        raise ValueError("The raycast resolution must be at least 16.")
    if max_face_count < 1:
        raise ValueError("The maximum face count must be positive.")
    if not isinstance(target_domain, str):
        raise TypeError("The UV target domain must be a string.")
    if target_domain not in _UV_TARGET_DOMAINS:
        raise ValueError("The UV target domain must be 'full' or 'left_half'.")
    if isinstance(exterior_uv_share, bool) or not isinstance(
        exterior_uv_share,
        int | float,
    ):
        raise TypeError("The exterior UV share must be a number.")
    if not math.isfinite(float(exterior_uv_share)) or not (
        0.5 <= float(exterior_uv_share) < 1.0
    ):
        raise ValueError("The exterior UV share must be at least 0.5 and below 1.")


def _load_scene(glb_bytes: bytes) -> trimesh.Scene:
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


def _load_visibility_reference(glb_bytes: bytes) -> trimesh.Scene:
    if not isinstance(glb_bytes, bytes | bytearray | memoryview):
        raise TypeError("The visibility reference GLB must be bytes.")
    payload = bytes(glb_bytes)
    if not payload:
        raise ValueError("The visibility reference GLB is empty.")
    try:
        return _load_scene(payload)
    except ValueError as error:
        raise ValueError(
            "The visibility reference GLB could not be loaded."
        ) from error


# ### Visibility reference validation ###
def _validate_visibility_reference(
    target_scene: trimesh.Scene,
    target_data: _SceneData,
    reference_scene: trimesh.Scene,
    reference_data: _SceneData,
) -> None:
    """Require an exact authoritative subset inside a bounded reference."""

    if reference_data.instance_face_count > target_data.instance_face_count * 2:
        raise ValueError(
            "The visibility reference may contain at most twice the target's "
            "instanced face count."
        )
    reference_by_name = {
        geometry.name: geometry for geometry in reference_data.geometries
    }
    target_transforms = _collect_attached_transforms(target_scene)
    reference_transforms = _collect_attached_transforms(reference_scene)
    for target in target_data.geometries:
        reference = reference_by_name.get(target.name)
        if reference is None:
            raise ValueError(
                "The visibility reference is missing an authoritative target "
                f"geometry: {target.name}."
            )
        _validate_matching_reference_geometry(target, reference)
        if not _transform_sets_match(
            target_transforms.get(target.name, ()),
            reference_transforms.get(target.name, ()),
        ):
            raise ValueError(
                "The visibility reference changed the authoritative target's "
                f"node transforms: {target.name}."
            )


def _validate_matching_reference_geometry(
    target: _GeometryData,
    reference: _GeometryData,
) -> None:
    target_faces = np.asarray(target.mesh.faces, dtype=np.int64)
    reference_faces = np.asarray(reference.mesh.faces, dtype=np.int64)
    if not np.array_equal(target_faces, reference_faces):
        raise ValueError(
            "The visibility reference changed the authoritative target's "
            f"face topology: {target.name}."
        )

    target_vertices = np.asarray(target.mesh.vertices, dtype=np.float64)
    reference_vertices = np.asarray(reference.mesh.vertices, dtype=np.float64)
    scale = max(
        float(np.max(np.abs(target_vertices), initial=0.0)),
        float(np.max(np.ptp(target_vertices, axis=0), initial=0.0)),
        1.0,
    )
    if target_vertices.shape != reference_vertices.shape or not np.allclose(
        target_vertices,
        reference_vertices,
        rtol=0.0,
        atol=_REFERENCE_GEOMETRY_RELATIVE_TOLERANCE * scale,
    ):
        raise ValueError(
            "The visibility reference changed the authoritative target's "
            f"local vertices: {target.name}."
        )


def _collect_attached_transforms(
    scene: trimesh.Scene,
) -> dict[object, tuple[np.ndarray, ...]]:
    transforms: dict[object, list[np.ndarray]] = {}
    for node_name in sorted(scene.graph.nodes_geometry, key=str):
        transform, geometry_name = scene.graph.get(node_name)
        transforms.setdefault(geometry_name, []).append(
            np.asarray(transform, dtype=np.float64).copy()
        )
    return {
        name: tuple(values)
        for name, values in transforms.items()
    }


def _transform_sets_match(
    target: Sequence[np.ndarray],
    reference: Sequence[np.ndarray],
) -> bool:
    if len(target) != len(reference):
        return False
    unmatched = list(reference)
    for target_transform in target:
        match_index = next(
            (
                index
                for index, reference_transform in enumerate(unmatched)
                if np.allclose(
                    target_transform,
                    reference_transform,
                    rtol=0.0,
                    atol=_REFERENCE_GEOMETRY_RELATIVE_TOLERANCE,
                )
            ),
            None,
        )
        if match_index is None:
            return False
        unmatched.pop(match_index)
    return True


def _transfer_reference_visibility(
    targets: Sequence[_GeometryData],
    measured_reference: Sequence[_GeometryData],
) -> tuple[_GeometryData, ...]:
    measured_by_name = {
        geometry.name: geometry for geometry in measured_reference
    }
    return tuple(
        _GeometryData(
            name=target.name,
            mesh=target.mesh,
            world_vertices=target.world_vertices,
            exterior_faces=measured_by_name[target.name].exterior_faces.copy(),
            visibility_hits=measured_by_name[target.name].visibility_hits.copy(),
        )
        for target in targets
    )


def _collect_scene_data(
    scene: trimesh.Scene,
    *,
    max_face_count: int,
) -> _SceneData:
    first_world_vertices: dict[object, np.ndarray] = {}
    instance_triangles: list[np.ndarray] = []
    instance_references: list[tuple[object, int]] = []
    attached_names: set[object] = set()
    for node_name in sorted(scene.graph.nodes_geometry, key=str):
        transform, geometry_name = scene.graph.get(node_name)
        mesh = scene.geometry.get(geometry_name)
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            continue
        _validate_mesh(mesh)
        world_vertices = trimesh.transform_points(
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(transform, dtype=np.float64),
        )
        first_world_vertices.setdefault(geometry_name, world_vertices)
        triangles = world_vertices[np.asarray(mesh.faces, dtype=np.int64)]
        instance_triangles.append(triangles)
        instance_references.extend(
            (geometry_name, face_index) for face_index in range(len(mesh.faces))
        )
        attached_names.add(geometry_name)

    geometry_names = sorted(attached_names, key=str)
    face_count = sum(len(scene.geometry[name].faces) for name in geometry_names)
    instance_face_count = sum(len(triangles) for triangles in instance_triangles)
    if face_count == 0 or instance_face_count == 0:
        raise ValueError("The object GLB contains no triangle faces.")
    if instance_face_count > max_face_count:
        raise ValueError(
            f"Visibility UV unwrapping supports at most {max_face_count:,} "
            "instanced faces."
        )

    geometries: list[_GeometryData] = []
    for geometry_name in geometry_names:
        mesh = scene.geometry[geometry_name]
        mesh_face_count = len(mesh.faces)
        geometries.append(
            _GeometryData(
                name=geometry_name,
                mesh=mesh,
                world_vertices=first_world_vertices[geometry_name],
                exterior_faces=np.zeros(mesh_face_count, dtype=bool),
                visibility_hits=np.zeros(mesh_face_count, dtype=np.int64),
            )
        )
    return _SceneData(
        geometries=tuple(geometries),
        instance_triangles=np.ascontiguousarray(
            np.concatenate(instance_triangles, axis=0),
            dtype=np.float64,
        ),
        instance_face_references=tuple(instance_references),
        face_count=face_count,
    )


def _validate_mesh(mesh: trimesh.Trimesh) -> None:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,):
        raise ValueError("The object GLB contains invalid vertex coordinates.")
    if faces.ndim != 2 or faces.shape[1:] != (3,):
        raise ValueError("Visibility UV unwrapping requires triangle faces.")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("The object GLB contains invalid vertex coordinates.")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("The object GLB contains invalid triangle indices.")
    triangles = vertices[faces]
    doubled_areas = np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    scale = max(float(np.ptp(vertices, axis=0).max()), 1e-12)
    if np.any(doubled_areas <= _AREA_EPSILON * scale * scale):
        raise ValueError("Visibility UV unwrapping requires nondegenerate faces.")


# ### Material validation ###
def _validate_untextured_materials(
    geometries: Sequence[_GeometryData],
) -> None:
    for geometry in geometries:
        material = getattr(geometry.mesh.visual, "material", None)
        if _material_contains_image(material):
            raise ValueError(
                "Visibility UV unwrapping must run before object texturing."
            )


def _material_contains_image(material: object) -> bool:
    if material is None:
        return False
    if isinstance(material, MultiMaterial):
        return any(_material_contains_image(item) for item in material.materials)
    if isinstance(material, SimpleMaterial):
        return material.image is not None
    texture_names = (
        "baseColorTexture",
        "emissiveTexture",
        "metallicRoughnessTexture",
        "normalTexture",
        "occlusionTexture",
    )
    return any(getattr(material, name, None) is not None for name in texture_names)


# ### Exterior raycast ###
def _measure_exterior_visibility(
    triangles: np.ndarray,
    *,
    raycast_resolution: int,
    cancellation_check: Callable[[], bool] | None,
) -> tuple[np.ndarray, int]:
    hits = np.zeros(len(triangles), dtype=np.int64)
    ray_sample_count = 0
    for raw_direction in _CAMERA_DIRECTIONS:
        _check_cancelled(cancellation_check)
        direction = _normalize(np.asarray(raw_direction, dtype=np.float64))
        basis_u, basis_v = _camera_basis(direction)
        projected = np.stack(
            (
                triangles @ basis_u,
                triangles @ basis_v,
                triangles @ direction,
            ),
            axis=2,
        )
        owner = _rasterize_nearest_faces(
            projected,
            resolution=raycast_resolution,
            cancellation_check=cancellation_check,
        )
        visible = owner[owner >= 0]
        if len(visible):
            hits[np.unique(visible)] += 1
        ray_sample_count += int(np.count_nonzero(owner >= 0))
    return hits, ray_sample_count


def _camera_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axes = np.eye(3, dtype=np.float64)
    helper = axes[int(np.argmin(np.abs(axes @ direction)))]
    basis_u = _normalize(np.cross(helper, direction))
    basis_v = _normalize(np.cross(direction, basis_u))
    return basis_u, basis_v


def _rasterize_nearest_faces(
    projected_triangles: np.ndarray,
    *,
    resolution: int,
    cancellation_check: Callable[[], bool] | None,
) -> np.ndarray:
    flat_xy = projected_triangles[:, :, :2].reshape((-1, 2))
    minimum = np.min(flat_xy, axis=0)
    maximum = np.max(flat_xy, axis=0)
    longest_extent = float(np.max(maximum - minimum))
    if longest_extent <= 0.0:
        raise ValueError("The object has no measurable projected extent.")
    padding = longest_extent / float(resolution)
    minimum -= padding
    maximum += padding
    extent = np.maximum(maximum - minimum, np.finfo(np.float64).eps)

    pixel_triangles = projected_triangles.copy()
    pixel_triangles[:, :, 0] = (
        (projected_triangles[:, :, 0] - minimum[0]) / extent[0] * resolution
    )
    pixel_triangles[:, :, 1] = (
        (projected_triangles[:, :, 1] - minimum[1]) / extent[1] * resolution
    )
    depths = np.full((resolution, resolution), -np.inf, dtype=np.float64)
    owners = np.full((resolution, resolution), -1, dtype=np.int64)
    for face_index, triangle in enumerate(pixel_triangles):
        if face_index % _CANCELLATION_CHECK_INTERVAL == 0:
            _check_cancelled(cancellation_check)
        _rasterize_triangle_depth(
            triangle,
            face_index=face_index,
            depths=depths,
            owners=owners,
        )
    return owners


def _rasterize_triangle_depth(
    triangle: np.ndarray,
    *,
    face_index: int,
    depths: np.ndarray,
    owners: np.ndarray,
) -> None:
    resolution = depths.shape[0]
    xy = triangle[:, :2]
    minimum = np.floor(np.min(xy, axis=0) - 0.5).astype(int)
    maximum = np.ceil(np.max(xy, axis=0) - 0.5).astype(int)
    minimum = np.clip(minimum, 0, resolution - 1)
    maximum = np.clip(maximum, 0, resolution - 1)
    if np.any(maximum < minimum):
        return

    columns = np.arange(minimum[0], maximum[0] + 1, dtype=np.float64) + 0.5
    rows = np.arange(minimum[1], maximum[1] + 1, dtype=np.float64) + 0.5
    grid_x, grid_y = np.meshgrid(columns, rows)
    sample_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    barycentric = _triangle_barycentric_coordinates(sample_points, xy)
    inside = np.all(barycentric >= -_BARYCENTRIC_EPSILON, axis=1)
    if not np.any(inside):
        center = np.mean(xy, axis=0)
        column = int(np.clip(math.floor(center[0]), 0, resolution - 1))
        row = int(np.clip(math.floor(center[1]), 0, resolution - 1))
        _update_depth_sample(
            row,
            column,
            float(np.mean(triangle[:, 2])),
            face_index,
            depths,
            owners,
        )
        return

    selected = barycentric[inside]
    selected_depths = selected @ triangle[:, 2]
    selected_rows = grid_y.ravel()[inside].astype(np.int64)
    selected_columns = grid_x.ravel()[inside].astype(np.int64)
    previous_depths = depths[selected_rows, selected_columns]
    previous_owners = owners[selected_rows, selected_columns]
    replace = (selected_depths > previous_depths + _DEPTH_EPSILON) | (
        (np.abs(selected_depths - previous_depths) <= _DEPTH_EPSILON)
        & ((previous_owners < 0) | (face_index < previous_owners))
    )
    rows_to_write = selected_rows[replace]
    columns_to_write = selected_columns[replace]
    depths[rows_to_write, columns_to_write] = selected_depths[replace]
    owners[rows_to_write, columns_to_write] = face_index


def _triangle_barycentric_coordinates(
    points: np.ndarray,
    triangle_xy: np.ndarray,
) -> np.ndarray:
    first, second, third = triangle_xy
    denominator = (
        (second[1] - third[1]) * (first[0] - third[0])
        + (third[0] - second[0]) * (first[1] - third[1])
    )
    if abs(float(denominator)) <= np.finfo(np.float64).eps:
        return np.full((len(points), 3), -1.0, dtype=np.float64)
    first_weight = (
        (second[1] - third[1]) * (points[:, 0] - third[0])
        + (third[0] - second[0]) * (points[:, 1] - third[1])
    ) / denominator
    second_weight = (
        (third[1] - first[1]) * (points[:, 0] - third[0])
        + (first[0] - third[0]) * (points[:, 1] - third[1])
    ) / denominator
    return np.column_stack(
        (first_weight, second_weight, 1.0 - first_weight - second_weight)
    )


def _update_depth_sample(
    row: int,
    column: int,
    depth: float,
    face_index: int,
    depths: np.ndarray,
    owners: np.ndarray,
) -> None:
    previous_depth = depths[row, column]
    previous_owner = owners[row, column]
    if depth > previous_depth + _DEPTH_EPSILON or (
        abs(depth - previous_depth) <= _DEPTH_EPSILON
        and (previous_owner < 0 or face_index < previous_owner)
    ):
        depths[row, column] = depth
        owners[row, column] = face_index


def _assign_geometry_visibility(
    scene_data: _SceneData,
    instance_hits: np.ndarray,
) -> tuple[_GeometryData, ...]:
    hit_by_geometry = {
        geometry.name: np.zeros(len(geometry.mesh.faces), dtype=np.int64)
        for geometry in scene_data.geometries
    }
    for instance_index, (geometry_name, face_index) in enumerate(
        scene_data.instance_face_references
    ):
        hit_by_geometry[geometry_name][face_index] += instance_hits[instance_index]
    return tuple(
        _GeometryData(
            name=geometry.name,
            mesh=geometry.mesh,
            world_vertices=geometry.world_vertices,
            exterior_faces=hit_by_geometry[geometry.name] > 0,
            visibility_hits=hit_by_geometry[geometry.name],
        )
        for geometry in scene_data.geometries
    )


# ### Xatlas input preparation ###
def _build_atlas_submissions(
    geometries: Sequence[_GeometryData],
    *,
    exterior_uv_share: float,
) -> tuple[_AtlasSubmission, ...]:
    surface_area_by_visibility = {False: 0.0, True: 0.0}
    for geometry in geometries:
        faces = np.asarray(geometry.mesh.faces, dtype=np.int64)
        triangles = geometry.world_vertices[faces]
        areas = _triangle_areas_3d(triangles)
        for exterior in (False, True):
            surface_area_by_visibility[exterior] += float(
                np.sum(areas[geometry.exterior_faces == exterior])
            )
    density_factors = _visibility_density_factors(
        surface_area_by_visibility,
        exterior_uv_share=exterior_uv_share,
    )

    submissions: list[_AtlasSubmission] = []
    for geometry in geometries:
        source_faces = np.asarray(geometry.mesh.faces, dtype=np.uint32)
        center = np.mean(geometry.world_vertices, axis=0)
        for exterior in (True, False):
            face_indices = np.flatnonzero(
                geometry.exterior_faces == exterior
            ).astype(np.int64)
            if not len(face_indices):
                continue
            scaled_vertices = (
                (geometry.world_vertices - center) * density_factors[exterior]
            )
            submissions.append(
                _AtlasSubmission(
                    submission_id=len(submissions),
                    geometry_name=geometry.name,
                    source_face_indices=face_indices,
                    source_vertices=np.ascontiguousarray(
                        scaled_vertices,
                        dtype=np.float32,
                    ),
                    source_faces=np.ascontiguousarray(
                        source_faces[face_indices],
                        dtype=np.uint32,
                    ),
                    exterior=exterior,
                )
            )
    if not submissions:
        raise ValueError("The object contains no faces to unwrap.")
    return tuple(submissions)


def _visibility_density_factors(
    surface_area_by_visibility: dict[bool, float],
    *,
    exterior_uv_share: float,
) -> dict[bool, float]:
    exterior_area = surface_area_by_visibility[True]
    hidden_area = surface_area_by_visibility[False]
    if exterior_area <= 0.0:
        return {False: math.sqrt(1.0 / hidden_area), True: 1.0}
    if hidden_area <= 0.0:
        return {False: 1.0, True: math.sqrt(1.0 / exterior_area)}
    return {
        True: math.sqrt(exterior_uv_share / exterior_area),
        False: math.sqrt((1.0 - exterior_uv_share) / hidden_area),
    }


# ### Xatlas unwrap and packing ###
def _resolve_packing_gutter_pixels(
    requested_gutter_pixels: int,
    *,
    target_domain: VisibilityUvTargetDomain,
) -> int:
    if target_domain == UV_TARGET_DOMAIN_LEFT_HALF:
        safe_scale = 1.0 - 4.0 * SYMMETRIC_LEFT_PACKED_UV_SAFETY_INSET
        compensated_gutter = math.ceil(
            requested_gutter_pixels * 2.0 / safe_scale
        )
        return max(
            _LEFT_HALF_MINIMUM_XATLAS_GUTTER_PIXELS,
            compensated_gutter,
        )
    return requested_gutter_pixels


def _generate_atlas(
    submissions: Sequence[_AtlasSubmission],
    *,
    texture_resolution: int,
    gutter_pixels: int,
    cancellation_check: Callable[[], bool] | None,
) -> _GeneratedAtlas:
    candidates = _generate_atlas_candidates(
        submissions,
        texture_resolution=texture_resolution,
        gutter_pixels=gutter_pixels,
        cancellation_check=cancellation_check,
    )
    return max(
        candidates,
        key=lambda generated: _measure_uv_area(generated.outputs)[1],
    )


def _generate_atlas_candidates(
    submissions: Sequence[_AtlasSubmission],
    *,
    texture_resolution: int,
    gutter_pixels: int,
    cancellation_check: Callable[[], bool] | None,
) -> tuple[_GeneratedAtlas, ...]:
    """Return baseline and useful alternate chartings for final comparison."""

    try:
        baseline = _generate_atlas_for_chart_policy(
            submissions,
            texture_resolution=texture_resolution,
            gutter_pixels=gutter_pixels,
            chart_max_cost=None,
            chart_straightness_weight=None,
            strategy_suffix=None,
            cancellation_check=cancellation_check,
        )
    except _AtlasPolicyPackingError as error:
        raise ValueError(str(error)) from error

    _exterior_share, baseline_occupancy = _measure_uv_area(
        baseline.outputs
    )
    face_count = sum(len(submission.source_faces) for submission in submissions)
    chart_face_ratio = float(baseline.atlas.chart_count) / max(face_count, 1)
    if not (
        baseline_occupancy < _RELAXED_CHART_OCCUPANCY_THRESHOLD
        and chart_face_ratio < _RELAXED_CHART_MAX_FACE_RATIO
    ):
        return (baseline,)

    try:
        relaxed = _generate_atlas_for_chart_policy(
            submissions,
            texture_resolution=texture_resolution,
            gutter_pixels=gutter_pixels,
            chart_max_cost=_RELAXED_CHART_MAX_COST,
            chart_straightness_weight=(
                _RELAXED_CHART_STRAIGHTNESS_WEIGHT
            ),
            strategy_suffix="relaxed_charts",
            cancellation_check=cancellation_check,
        )
    except _AtlasPolicyPackingError:
        return (baseline,)
    return baseline, relaxed


def _generate_atlas_for_chart_policy(
    submissions: Sequence[_AtlasSubmission],
    *,
    texture_resolution: int,
    gutter_pixels: int,
    chart_max_cost: float | None,
    chart_straightness_weight: float | None,
    strategy_suffix: str | None,
    cancellation_check: Callable[[], bool] | None,
) -> _GeneratedAtlas:
    padding_pixels = gutter_pixels
    best_accepted: _AtlasPackCandidate | None = None
    for _attempt in range(_MAX_GUTTER_PACK_ATTEMPTS):
        _check_cancelled(cancellation_check)
        candidates: list[_AtlasPackCandidate] = []
        for strategy_index, (
            strategy_name,
            rotate_charts,
            rotate_charts_to_axis,
        ) in enumerate(_XATLAS_PACKING_STRATEGIES):
            try:
                candidate = _generate_atlas_candidate(
                    submissions,
                    texture_resolution=texture_resolution,
                    padding_pixels=padding_pixels,
                    strategy_index=strategy_index,
                    strategy_name=strategy_name,
                    rotate_charts=rotate_charts,
                    rotate_charts_to_axis=rotate_charts_to_axis,
                    chart_max_cost=chart_max_cost,
                    chart_straightness_weight=chart_straightness_weight,
                    strategy_suffix=strategy_suffix,
                    cancellation_check=cancellation_check,
                )
            except _AtlasCandidatePackingError:
                continue
            candidates.append(candidate)
        if not candidates:
            raise _AtlasPolicyPackingError(
                "Xatlas could not fit the object into one UV atlas."
            )
        accepted_candidates = tuple(
            candidate
            for candidate in candidates
            if (
                candidate.generated.effective_gutter_pixels + 1e-6
                >= gutter_pixels
            )
        )
        if accepted_candidates:
            attempt_best = max(
                accepted_candidates,
                key=_atlas_candidate_score,
            )
            if (
                best_accepted is None
                or _atlas_candidate_score(attempt_best)
                > _atlas_candidate_score(best_accepted)
            ):
                best_accepted = attempt_best
        if len(accepted_candidates) == len(candidates):
            assert best_accepted is not None
            return best_accepted.generated

        best_effective_gutter = max(
            candidate.generated.effective_gutter_pixels
            for candidate in candidates
        )
        padding_pixels = max(
            padding_pixels + 1,
            math.ceil(
                float(padding_pixels)
                * float(gutter_pixels)
                / max(best_effective_gutter, np.finfo(np.float64).eps)
            ),
        )
    if best_accepted is not None:
        return best_accepted.generated
    raise _AtlasPolicyPackingError(
        "Xatlas could not provide the required UV chart gutter."
    )


def _generate_atlas_candidate(
    submissions: Sequence[_AtlasSubmission],
    *,
    texture_resolution: int,
    padding_pixels: int,
    strategy_index: int,
    strategy_name: str,
    rotate_charts: bool,
    rotate_charts_to_axis: bool,
    chart_max_cost: float | None,
    chart_straightness_weight: float | None,
    strategy_suffix: str | None,
    cancellation_check: Callable[[], bool] | None,
) -> _AtlasPackCandidate:
    _check_cancelled(cancellation_check)
    atlas = xatlas.Atlas()
    for submission in submissions:
        atlas.add_mesh(
            submission.source_vertices,
            submission.source_faces,
        )
    chart_options = xatlas.ChartOptions()
    chart_options.fix_winding = False
    chart_options.use_input_mesh_uvs = False
    if chart_max_cost is not None:
        chart_options.max_cost = chart_max_cost
    if chart_straightness_weight is not None:
        chart_options.straightness_weight = chart_straightness_weight
    pack_options = xatlas.PackOptions()
    pack_options.resolution = texture_resolution
    pack_options.padding = padding_pixels
    pack_options.bilinear = True
    pack_options.blockAlign = False
    pack_options.bruteForce = False
    pack_options.create_image = False
    pack_options.rotate_charts = rotate_charts
    pack_options.rotate_charts_to_axis = rotate_charts_to_axis
    try:
        atlas.generate(chart_options, pack_options, False)
    except RuntimeError as error:
        raise _AtlasCandidatePackingError(
            f"Xatlas packing policy '{strategy_name}' failed."
        ) from error
    _check_cancelled(cancellation_check)
    if int(atlas.atlas_count) != 1:
        raise _AtlasCandidatePackingError(
            f"Xatlas packing policy '{strategy_name}' required multiple "
            "atlases."
        )

    longest_dimension = max(int(atlas.width), int(atlas.height), 1)
    effective_gutter = (
        float(padding_pixels)
        * float(texture_resolution)
        / float(longest_dimension)
    )
    outputs = _collect_atlas_outputs(atlas, submissions)
    _exterior_share, triangle_occupancy = _measure_uv_area(outputs)
    reported_strategy = (
        strategy_name
        if strategy_suffix is None
        else f"{strategy_name}+{strategy_suffix}"
    )
    generated = _GeneratedAtlas(
        atlas=atlas,
        outputs=outputs,
        effective_gutter_pixels=effective_gutter,
        packing_gutter_pixels=padding_pixels,
        packing_strategy=reported_strategy,
    )
    return _AtlasPackCandidate(
        generated=generated,
        triangle_occupancy=triangle_occupancy,
        utilization=float(atlas.get_utilization(0)),
        strategy_index=strategy_index,
    )


def _atlas_candidate_score(
    candidate: _AtlasPackCandidate,
) -> tuple[float, float, int, int, int, int]:
    atlas = candidate.generated.atlas
    width = int(atlas.width)
    height = int(atlas.height)
    return (
        candidate.triangle_occupancy,
        candidate.utilization,
        -(width * height),
        -max(width, height),
        -width,
        -candidate.strategy_index,
    )


def _collect_atlas_outputs(
    atlas: xatlas.Atlas,
    submissions: Sequence[_AtlasSubmission],
) -> tuple[_AtlasOutput, ...]:
    outputs: list[_AtlasOutput] = []
    for submission in submissions:
        vertex_mapping, faces, uvs = atlas[submission.submission_id]
        vertex_mapping = np.asarray(vertex_mapping, dtype=np.int64)
        faces = np.asarray(faces, dtype=np.int64)
        uvs = np.asarray(uvs, dtype=np.float64)
        if faces.shape != submission.source_faces.shape:
            raise RuntimeError("Xatlas changed the object's face count.")
        if not np.array_equal(
            vertex_mapping[faces],
            np.asarray(submission.source_faces, dtype=np.int64),
        ):
            raise RuntimeError("Xatlas changed the object's face topology.")
        if uvs.shape != (len(vertex_mapping), 2) or not np.all(np.isfinite(uvs)):
            raise RuntimeError("Xatlas produced invalid UV coordinates.")
        outputs.append(
            _AtlasOutput(
                submission=submission,
                vertex_mapping=vertex_mapping,
                faces=faces,
                uvs=uvs,
            )
        )
    return tuple(outputs)


def _map_outputs_to_target_domain(
    outputs: Sequence[_AtlasOutput],
    *,
    target_domain: VisibilityUvTargetDomain,
) -> tuple[_AtlasOutput, ...]:
    if target_domain == UV_TARGET_DOMAIN_FULL:
        return tuple(outputs)
    mapped: list[_AtlasOutput] = []
    for output in outputs:
        uvs = output.uvs.copy()
        uvs[:, 0] *= 0.5
        mapped.append(
            _AtlasOutput(
                submission=output.submission,
                vertex_mapping=output.vertex_mapping,
                faces=output.faces,
                uvs=uvs,
            )
        )
    return tuple(mapped)


def _build_target_domain_layout(
    generated: _GeneratedAtlas,
    *,
    texture_resolution: int,
    gutter_pixels: int,
    target_domain: VisibilityUvTargetDomain,
    cancellation_check: Callable[[], bool] | None,
) -> _TargetDomainLayout:
    direct_outputs = _map_outputs_to_target_domain(
        generated.outputs,
        target_domain=target_domain,
    )
    _exterior_share, direct_occupancy = _measure_uv_area(
        direct_outputs,
        target_domain=target_domain,
    )
    horizontal_gutter, vertical_gutter = _measure_effective_gutters(
        generated,
        texture_resolution=texture_resolution,
        target_domain=target_domain,
    )
    direct = _TargetDomainLayout(
        outputs=direct_outputs,
        chart_count=int(generated.atlas.chart_count),
        layout_width=int(generated.atlas.width),
        layout_height=int(generated.atlas.height),
        utilization=direct_occupancy,
        horizontal_gutter_pixels=horizontal_gutter,
        vertical_gutter_pixels=vertical_gutter,
        packing_strategy=generated.packing_strategy,
    )
    if target_domain != UV_TARGET_DOMAIN_LEFT_HALF:
        return direct
    direct = _ensure_left_half_boundary_inset(direct)

    chart_count = int(generated.atlas.chart_count)
    if not (
        direct.utilization < _TRUE_ASPECT_DIRECT_OCCUPANCY_THRESHOLD
        and _TRUE_ASPECT_MINIMUM_CHART_COUNT
        <= chart_count
        <= _TRUE_ASPECT_MAXIMUM_CHART_COUNT
    ):
        return direct

    try:
        candidate = _build_true_aspect_layout(
            generated,
            texture_resolution=texture_resolution,
            gutter_pixels=gutter_pixels,
            cancellation_check=cancellation_check,
        )
    except _TrueAspectPackingError:
        return direct
    candidate = _ensure_left_half_boundary_inset(candidate)
    _exterior_share, candidate_occupancy = _measure_uv_area(
        candidate.outputs,
        target_domain=target_domain,
    )
    if candidate_occupancy > direct.utilization + _AREA_EPSILON:
        return candidate
    return direct


def _build_best_target_domain_layout(
    generated_candidates: Sequence[_GeneratedAtlas],
    *,
    texture_resolution: int,
    gutter_pixels: int,
    target_domain: VisibilityUvTargetDomain,
    cancellation_check: Callable[[], bool] | None,
) -> _TargetDomainLayout:
    """Compare complete target layouts, including conditional repacking."""

    if not generated_candidates:
        raise ValueError("UV generation returned no atlas candidates.")
    layouts: list[_TargetDomainLayout] = []
    for generated in generated_candidates:
        _check_cancelled(cancellation_check)
        layouts.append(
            _build_target_domain_layout(
                generated,
                texture_resolution=texture_resolution,
                gutter_pixels=gutter_pixels,
                target_domain=target_domain,
                cancellation_check=cancellation_check,
            )
        )
    return max(
        enumerate(layouts),
        key=lambda indexed: (
            indexed[1].utilization,
            -indexed[0],
        ),
    )[1]


def _ensure_left_half_boundary_inset(
    layout: _TargetDomainLayout,
) -> _TargetDomainLayout:
    """Reserve the filtering border required by symmetric pair variants."""

    tolerance = 1e-12
    inset = SYMMETRIC_LEFT_PACKED_UV_SAFETY_INSET
    minimum = np.asarray((inset, inset), dtype=np.float64)
    maximum = np.asarray((0.5 - inset, 1.0 - inset), dtype=np.float64)
    all_uvs = np.concatenate(
        [np.asarray(output.uvs, dtype=np.float64) for output in layout.outputs],
        axis=0,
    )
    if np.all(all_uvs >= minimum - tolerance) and np.all(
        all_uvs <= maximum + tolerance
    ):
        return layout

    uniform_scale = 1.0 - 4.0 * inset
    vertical_offset = (1.0 - uniform_scale) * 0.5
    transformed_outputs: list[_AtlasOutput] = []
    for output in layout.outputs:
        transformed_uvs = np.asarray(output.uvs, dtype=np.float64).copy()
        transformed_uvs *= uniform_scale
        transformed_uvs[:, 0] += inset
        transformed_uvs[:, 1] += vertical_offset
        transformed_outputs.append(
            _AtlasOutput(
                submission=output.submission,
                vertex_mapping=output.vertex_mapping,
                faces=output.faces,
                uvs=transformed_uvs,
            )
        )
    outputs = tuple(transformed_outputs)
    _exterior_share, utilization = _measure_uv_area(
        outputs,
        target_domain=UV_TARGET_DOMAIN_LEFT_HALF,
    )
    return _TargetDomainLayout(
        outputs=outputs,
        chart_count=layout.chart_count,
        layout_width=layout.layout_width,
        layout_height=layout.layout_height,
        utilization=utilization,
        horizontal_gutter_pixels=(
            layout.horizontal_gutter_pixels * uniform_scale
        ),
        vertical_gutter_pixels=(
            layout.vertical_gutter_pixels * uniform_scale
        ),
        packing_strategy=f"{layout.packing_strategy}+boundary_inset",
    )


def _measure_effective_gutters(
    generated: _GeneratedAtlas,
    *,
    texture_resolution: int,
    target_domain: VisibilityUvTargetDomain,
) -> tuple[float, float]:
    atlas_width = max(int(generated.atlas.width), 1)
    atlas_height = max(int(generated.atlas.height), 1)
    horizontal_span = (
        0.5 if target_domain == UV_TARGET_DOMAIN_LEFT_HALF else 1.0
    )
    horizontal = (
        float(generated.packing_gutter_pixels)
        * float(texture_resolution)
        * horizontal_span
        / float(atlas_width)
    )
    vertical = (
        float(generated.packing_gutter_pixels)
        * float(texture_resolution)
        / float(atlas_height)
    )
    return horizontal, vertical


# ### True-aspect half-atlas packing ###
def _build_true_aspect_layout(
    generated: _GeneratedAtlas,
    *,
    texture_resolution: int,
    gutter_pixels: int,
    cancellation_check: Callable[[], bool] | None,
) -> _TargetDomainLayout:
    _check_cancelled(cancellation_check)
    charts = _extract_connected_uv_charts(
        generated,
        cancellation_check=cancellation_check,
    )
    if len(charts) != int(generated.atlas.chart_count):
        raise _TrueAspectPackingError(
            "Connected UV charts did not match the xatlas chart count."
        )
    if not (
        _TRUE_ASPECT_MINIMUM_CHART_COUNT
        <= len(charts)
        <= _TRUE_ASPECT_MAXIMUM_CHART_COUNT
    ):
        raise _TrueAspectPackingError(
            "The UV chart count is outside the bounded packing range."
        )

    target_width = float(texture_resolution) * 0.5
    target_height = float(texture_resolution)
    packed = _pack_connected_uv_charts(
        charts,
        target_width=target_width,
        target_height=target_height,
        gutter_pixels=float(gutter_pixels),
        cancellation_check=cancellation_check,
    )
    outputs = _apply_true_aspect_chart_placements(
        generated,
        charts,
        packed,
        texture_resolution=texture_resolution,
        cancellation_check=cancellation_check,
    )
    _validate_true_aspect_chart_placements(
        charts,
        packed,
        target_width=target_width,
        target_height=target_height,
        gutter_pixels=float(gutter_pixels),
    )
    _exterior_share, triangle_utilization = _measure_uv_area(
        outputs,
        target_domain=UV_TARGET_DOMAIN_LEFT_HALF,
    )
    return _TargetDomainLayout(
        outputs=outputs,
        chart_count=len(charts),
        layout_width=int(round(target_width)),
        layout_height=int(round(target_height)),
        utilization=triangle_utilization,
        horizontal_gutter_pixels=float(gutter_pixels),
        vertical_gutter_pixels=float(gutter_pixels),
        packing_strategy=(
            f"{generated.packing_strategy}+"
            f"{_TRUE_ASPECT_PACKING_STRATEGY_SUFFIX}:"
            f"{packed.order_policy}"
        ),
    )


def _extract_connected_uv_charts(
    generated: _GeneratedAtlas,
    *,
    cancellation_check: Callable[[], bool] | None,
) -> tuple[_ConnectedUvChart, ...]:
    atlas_width = float(generated.atlas.width)
    atlas_height = float(generated.atlas.height)
    if atlas_width <= 0.0 or atlas_height <= 0.0:
        raise _TrueAspectPackingError("Xatlas returned invalid layout dimensions.")

    charts: list[_ConnectedUvChart] = []
    for output_index, output in enumerate(generated.outputs):
        faces = np.asarray(output.faces, dtype=np.int64)
        if len(faces) == 0:
            continue
        components = _connected_face_components(
            faces,
            cancellation_check=cancellation_check,
        )
        pixel_uvs = np.asarray(output.uvs, dtype=np.float64) * (
            atlas_width,
            atlas_height,
        )
        for face_indices in components:
            vertex_indices = np.unique(faces[face_indices].reshape(-1))
            points = pixel_uvs[vertex_indices]
            minimum = np.min(points, axis=0)
            extent = np.max(points, axis=0) - minimum
            if (
                not np.all(np.isfinite(extent))
                or np.any(extent <= np.finfo(np.float64).eps)
            ):
                raise _TrueAspectPackingError(
                    "A connected UV chart has no measurable bounds."
                )
            charts.append(
                _ConnectedUvChart(
                    chart_index=len(charts),
                    output_index=output_index,
                    vertex_indices=vertex_indices,
                    minimum_pixels=minimum,
                    extent_pixels=extent,
                )
            )
    if not charts:
        raise _TrueAspectPackingError("Xatlas returned no connected UV charts.")
    return tuple(charts)


def _connected_face_components(
    faces: np.ndarray,
    *,
    cancellation_check: Callable[[], bool] | None,
) -> tuple[np.ndarray, ...]:
    parents = np.arange(len(faces), dtype=np.int64)

    def find(face_index: int) -> int:
        while int(parents[face_index]) != face_index:
            parents[face_index] = parents[int(parents[face_index])]
            face_index = int(parents[face_index])
        return face_index

    first_face_by_vertex: dict[int, int] = {}
    for face_index, face in enumerate(faces):
        if face_index % _CANCELLATION_CHECK_INTERVAL == 0:
            _check_cancelled(cancellation_check)
        for raw_vertex_index in face:
            vertex_index = int(raw_vertex_index)
            previous_face = first_face_by_vertex.get(vertex_index)
            if previous_face is None:
                first_face_by_vertex[vertex_index] = face_index
                continue
            first_root = find(face_index)
            second_root = find(previous_face)
            if first_root != second_root:
                parents[second_root] = first_root

    grouped_faces: dict[int, list[int]] = {}
    for face_index in range(len(faces)):
        grouped_faces.setdefault(find(face_index), []).append(face_index)
    ordered_groups = sorted(grouped_faces.values(), key=lambda group: group[0])
    return tuple(np.asarray(group, dtype=np.int64) for group in ordered_groups)


def _pack_connected_uv_charts(
    charts: Sequence[_ConnectedUvChart],
    *,
    target_width: float,
    target_height: float,
    gutter_pixels: float,
    cancellation_check: Callable[[], bool] | None,
) -> _TrueAspectPackResult:
    if target_width <= 2.0 * gutter_pixels or target_height <= 2.0 * gutter_pixels:
        raise _TrueAspectPackingError(
            "The requested gutter leaves no room in the half atlas."
        )
    bounding_area = sum(
        float(chart.extent_pixels[0] * chart.extent_pixels[1])
        for chart in charts
    )
    if not math.isfinite(bounding_area) or bounding_area <= _AREA_EPSILON:
        raise _TrueAspectPackingError("The UV charts have no measurable area.")
    scale_upper_bound = math.sqrt(
        target_width * target_height / bounding_area
    )

    best: _TrueAspectPackResult | None = None
    for order_policy in _TRUE_ASPECT_PACKING_ORDER_POLICIES:
        _check_cancelled(cancellation_check)
        lower = 0.0
        upper = scale_upper_bound
        policy_placements: tuple[_ChartPlacement, ...] | None = None
        for _step in range(_TRUE_ASPECT_BINARY_SEARCH_STEPS):
            _check_cancelled(cancellation_check)
            scale = (lower + upper) * 0.5
            placements = _try_pack_charts_at_scale(
                charts,
                scale=scale,
                target_width=target_width,
                target_height=target_height,
                gutter_pixels=gutter_pixels,
                order_policy=order_policy,
                cancellation_check=cancellation_check,
            )
            if placements is None:
                upper = scale
            else:
                lower = scale
                policy_placements = placements
        if policy_placements is None:
            continue
        candidate = _TrueAspectPackResult(
            scale=lower,
            placements=policy_placements,
            order_policy=order_policy,
        )
        if best is None or candidate.scale > best.scale + _AREA_EPSILON:
            best = candidate
    if best is None or best.scale <= np.finfo(np.float64).eps:
        raise _TrueAspectPackingError(
            "The connected UV charts did not fit in the half atlas."
        )
    return best


def _try_pack_charts_at_scale(
    charts: Sequence[_ConnectedUvChart],
    *,
    scale: float,
    target_width: float,
    target_height: float,
    gutter_pixels: float,
    order_policy: str,
    cancellation_check: Callable[[], bool] | None,
) -> tuple[_ChartPlacement, ...] | None:
    ordered_charts = sorted(
        charts,
        key=lambda chart: _chart_order_key(chart, order_policy=order_policy),
    )
    free_rectangles = [
        _PackingRectangle(
            x=gutter_pixels,
            y=gutter_pixels,
            width=target_width - gutter_pixels,
            height=target_height - gutter_pixels,
        )
    ]
    placements: list[_ChartPlacement | None] = [None] * len(charts)
    for chart_offset, chart in enumerate(ordered_charts):
        if chart_offset % 16 == 0:
            _check_cancelled(cancellation_check)
        chart_width = float(chart.extent_pixels[0]) * scale + gutter_pixels
        chart_height = float(chart.extent_pixels[1]) * scale + gutter_pixels
        selected = _select_maxrects_placement(
            free_rectangles,
            chart_width=chart_width,
            chart_height=chart_height,
        )
        if selected is None:
            return None
        rectangle, rotated = selected
        placements[chart.chart_index] = _ChartPlacement(
            rectangle=rectangle,
            rotated=rotated,
        )
        free_rectangles = _split_maxrects_free_space(
            free_rectangles,
            rectangle,
            cancellation_check=cancellation_check,
        )
        if len(free_rectangles) > _TRUE_ASPECT_MAXIMUM_FREE_RECTANGLES:
            raise _TrueAspectPackingError(
                "The bounded MaxRects search produced too many free regions."
            )
    if any(placement is None for placement in placements):
        raise _TrueAspectPackingError("A UV chart was not assigned a placement.")
    return tuple(
        placement
        for placement in placements
        if placement is not None
    )


def _chart_order_key(
    chart: _ConnectedUvChart,
    *,
    order_policy: str,
) -> tuple[float, float, float, int]:
    width = float(chart.extent_pixels[0])
    height = float(chart.extent_pixels[1])
    long_side = max(width, height)
    short_side = min(width, height)
    area = width * height
    if order_policy == "area":
        return (-area, -long_side, -short_side, chart.chart_index)
    if order_policy == "long_side":
        return (-long_side, -area, -short_side, chart.chart_index)
    if order_policy == "perimeter":
        return (-(width + height), -area, -long_side, chart.chart_index)
    raise _TrueAspectPackingError("The MaxRects ordering policy is invalid.")


def _select_maxrects_placement(
    free_rectangles: Sequence[_PackingRectangle],
    *,
    chart_width: float,
    chart_height: float,
) -> tuple[_PackingRectangle, bool] | None:
    best: tuple[tuple[float, ...], _PackingRectangle, bool] | None = None
    for free_index, free in enumerate(free_rectangles):
        orientations = (
            (False, chart_width, chart_height),
            (True, chart_height, chart_width),
        )
        for rotated, width, height in orientations:
            if width > free.width + 1e-9 or height > free.height + 1e-9:
                continue
            remaining_width = free.width - width
            remaining_height = free.height - height
            score = (
                min(remaining_width, remaining_height),
                max(remaining_width, remaining_height),
                free.y,
                free.x,
                float(free_index),
                float(rotated),
            )
            rectangle = _PackingRectangle(
                x=free.x,
                y=free.y,
                width=width,
                height=height,
            )
            if best is None or score < best[0]:
                best = (score, rectangle, rotated)
    if best is None:
        return None
    return best[1], best[2]


def _split_maxrects_free_space(
    free_rectangles: Sequence[_PackingRectangle],
    used: _PackingRectangle,
    *,
    cancellation_check: Callable[[], bool] | None,
) -> list[_PackingRectangle]:
    split_rectangles: list[_PackingRectangle] = []
    for free in free_rectangles:
        if not _rectangles_overlap(free, used):
            split_rectangles.append(free)
            continue
        free_right = free.x + free.width
        free_top = free.y + free.height
        used_right = used.x + used.width
        used_top = used.y + used.height
        if used.x > free.x + 1e-9:
            split_rectangles.append(
                _PackingRectangle(free.x, free.y, used.x - free.x, free.height)
            )
        if used_right < free_right - 1e-9:
            split_rectangles.append(
                _PackingRectangle(
                    used_right,
                    free.y,
                    free_right - used_right,
                    free.height,
                )
            )
        if used.y > free.y + 1e-9:
            split_rectangles.append(
                _PackingRectangle(free.x, free.y, free.width, used.y - free.y)
            )
        if used_top < free_top - 1e-9:
            split_rectangles.append(
                _PackingRectangle(
                    free.x,
                    used_top,
                    free.width,
                    free_top - used_top,
                )
            )

    pruned: list[_PackingRectangle] = []
    for rectangle_index, rectangle in enumerate(split_rectangles):
        if rectangle_index % 64 == 0:
            _check_cancelled(cancellation_check)
        if rectangle.width <= 1e-9 or rectangle.height <= 1e-9:
            continue
        if any(
            _rectangle_contains(existing, rectangle)
            for existing in pruned
        ):
            continue
        pruned = [
            existing
            for existing in pruned
            if not _rectangle_contains(rectangle, existing)
        ]
        pruned.append(rectangle)
    return pruned


def _rectangles_overlap(
    first: _PackingRectangle,
    second: _PackingRectangle,
) -> bool:
    return (
        first.x < second.x + second.width - 1e-9
        and first.x + first.width > second.x + 1e-9
        and first.y < second.y + second.height - 1e-9
        and first.y + first.height > second.y + 1e-9
    )


def _rectangle_contains(
    outer: _PackingRectangle,
    inner: _PackingRectangle,
) -> bool:
    return (
        inner.x >= outer.x - 1e-9
        and inner.y >= outer.y - 1e-9
        and inner.x + inner.width <= outer.x + outer.width + 1e-9
        and inner.y + inner.height <= outer.y + outer.height + 1e-9
    )


def _apply_true_aspect_chart_placements(
    generated: _GeneratedAtlas,
    charts: Sequence[_ConnectedUvChart],
    packed: _TrueAspectPackResult,
    *,
    texture_resolution: int,
    cancellation_check: Callable[[], bool] | None,
) -> tuple[_AtlasOutput, ...]:
    atlas_size = np.asarray(
        (float(generated.atlas.width), float(generated.atlas.height)),
        dtype=np.float64,
    )
    transformed_uvs = [output.uvs.copy() for output in generated.outputs]
    assigned_vertices = [
        np.zeros(len(output.uvs), dtype=bool) for output in generated.outputs
    ]
    for chart_offset, chart in enumerate(charts):
        if chart_offset % 16 == 0:
            _check_cancelled(cancellation_check)
        placement = packed.placements[chart.chart_index]
        source_uvs = generated.outputs[chart.output_index].uvs[
            chart.vertex_indices
        ]
        local_pixels = source_uvs * atlas_size - chart.minimum_pixels
        if placement.rotated:
            target_pixels = np.column_stack(
                (
                    local_pixels[:, 1],
                    chart.extent_pixels[0] - local_pixels[:, 0],
                )
            )
        else:
            target_pixels = local_pixels
        target_pixels *= packed.scale
        target_pixels += (
            placement.rectangle.x,
            placement.rectangle.y,
        )
        transformed_uvs[chart.output_index][chart.vertex_indices] = (
            target_pixels / float(texture_resolution)
        )
        assigned_vertices[chart.output_index][chart.vertex_indices] = True

    if any(not np.all(assigned) for assigned in assigned_vertices):
        raise _TrueAspectPackingError(
            "A generated UV vertex was not assigned to a connected chart."
        )
    return tuple(
        _AtlasOutput(
            submission=output.submission,
            vertex_mapping=output.vertex_mapping,
            faces=output.faces,
            uvs=transformed_uvs[output_index],
        )
        for output_index, output in enumerate(generated.outputs)
    )


def _validate_true_aspect_chart_placements(
    charts: Sequence[_ConnectedUvChart],
    packed: _TrueAspectPackResult,
    *,
    target_width: float,
    target_height: float,
    gutter_pixels: float,
) -> None:
    for chart, placement in zip(
        charts,
        packed.placements,
        strict=True,
    ):
        rectangle = placement.rectangle
        content_width = float(
            chart.extent_pixels[1 if placement.rotated else 0]
        ) * packed.scale
        content_height = float(
            chart.extent_pixels[0 if placement.rotated else 1]
        ) * packed.scale
        if not np.allclose(
            (rectangle.width, rectangle.height),
            (content_width + gutter_pixels, content_height + gutter_pixels),
            rtol=0.0,
            atol=1e-6,
        ):
            raise _TrueAspectPackingError(
                "A packed UV chart has inconsistent bounds."
            )
        if (
            rectangle.x < gutter_pixels - 1e-6
            or rectangle.y < gutter_pixels - 1e-6
            or rectangle.x + content_width > target_width - gutter_pixels + 1e-6
            or rectangle.y + content_height > target_height - gutter_pixels + 1e-6
        ):
            raise _TrueAspectPackingError(
                "A packed UV chart escaped the half-atlas bounds."
            )
    rectangles = [placement.rectangle for placement in packed.placements]
    for first_index, first in enumerate(rectangles):
        for second in rectangles[first_index + 1 :]:
            if _rectangles_overlap(first, second):
                raise _TrueAspectPackingError(
                    "Packed UV chart bounds overlap."
                )


# ### UV application and export ###
def _apply_atlas_uvs(
    source_scene: trimesh.Scene,
    geometries: Sequence[_GeometryData],
    atlas_outputs: Sequence[_AtlasOutput],
) -> trimesh.Scene:
    output_scene = source_scene.copy()
    outputs_by_geometry: dict[object, list[_AtlasOutput]] = {
        geometry.name: [] for geometry in geometries
    }
    for output in atlas_outputs:
        outputs_by_geometry[output.submission.geometry_name].append(output)
    for geometry in geometries:
        output_scene.geometry[geometry.name] = _rebuild_geometry_with_uvs(
            geometry.mesh,
            outputs_by_geometry[geometry.name],
        )
    return output_scene


def _rebuild_geometry_with_uvs(
    source: trimesh.Trimesh,
    atlas_outputs: Sequence[_AtlasOutput],
) -> trimesh.Trimesh:
    output_by_face: dict[int, tuple[_AtlasOutput, int]] = {}
    for output in atlas_outputs:
        for local_face_index, source_face_index in enumerate(
            output.submission.source_face_indices
        ):
            output_by_face[int(source_face_index)] = (output, local_face_index)
    if len(output_by_face) != len(source.faces):
        raise RuntimeError("Xatlas did not return UVs for every object face.")

    source_vertices = np.asarray(source.vertices, dtype=np.float64)
    source_normals = _get_vertex_normals(source)
    source_vertex_colors, source_face_colors = _get_authored_colors(source)
    output_vertices: list[np.ndarray] = []
    output_normals: list[np.ndarray] = []
    output_uvs: list[np.ndarray] = []
    output_colors: list[np.ndarray] = []
    output_faces: list[tuple[int, int, int]] = []
    rebuilt_vertex_indices: dict[tuple[int, ...], int] = {}
    for source_face_index in range(len(source.faces)):
        atlas_output, local_face_index = output_by_face[source_face_index]
        atlas_face = atlas_output.faces[local_face_index]
        rebuilt_face: list[int] = []
        for atlas_vertex_index in atlas_face:
            atlas_vertex = int(atlas_vertex_index)
            key = (
                atlas_output.submission.submission_id,
                atlas_vertex,
                source_face_index if source_face_colors is not None else -1,
            )
            rebuilt_index = rebuilt_vertex_indices.get(key)
            if rebuilt_index is None:
                source_vertex = int(atlas_output.vertex_mapping[atlas_vertex])
                rebuilt_index = len(output_vertices)
                rebuilt_vertex_indices[key] = rebuilt_index
                output_vertices.append(source_vertices[source_vertex].copy())
                output_normals.append(source_normals[source_vertex].copy())
                output_uvs.append(atlas_output.uvs[atlas_vertex].copy())
                if source_face_colors is not None:
                    output_colors.append(
                        source_face_colors[source_face_index].copy()
                    )
                elif source_vertex_colors is not None:
                    output_colors.append(
                        source_vertex_colors[source_vertex].copy()
                    )
            rebuilt_face.append(rebuilt_index)
        output_faces.append(tuple(rebuilt_face))

    face_materials = getattr(source.visual, "face_materials", None)
    material = getattr(source.visual, "material", None)
    if material is None:
        material = PBRMaterial()
    visual = TextureVisuals(
        uv=np.asarray(output_uvs, dtype=np.float64),
        material=copy.deepcopy(material),
        face_materials=(
            np.asarray(face_materials, dtype=np.int64).copy()
            if face_materials is not None
            else None
        ),
    )
    if output_colors:
        visual.vertex_attributes["color"] = np.asarray(output_colors).copy()
    return trimesh.Trimesh(
        vertices=np.asarray(output_vertices, dtype=np.float64),
        faces=np.asarray(output_faces, dtype=np.int64),
        vertex_normals=np.asarray(output_normals, dtype=np.float64),
        visual=visual,
        metadata=copy.deepcopy(source.metadata),
        process=False,
        validate=False,
    )


def _get_authored_colors(
    mesh: trimesh.Trimesh,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return authored vertex or face colors without synthesizing either."""

    visual = mesh.visual
    vertex_attributes = getattr(visual, "vertex_attributes", None)
    if vertex_attributes is not None and "color" in vertex_attributes:
        colors = np.asarray(vertex_attributes["color"])
        if len(colors) == len(mesh.vertices):
            return colors.copy(), None
    if not isinstance(visual, ColorVisuals):
        return None, None
    if visual.kind == "vertex":
        colors = np.asarray(visual.vertex_colors)
        if len(colors) == len(mesh.vertices):
            return colors.copy(), None
    if visual.kind == "face":
        colors = np.asarray(visual.face_colors)
        if len(colors) == len(mesh.faces):
            return None, colors.copy()
    return None, None


def _measure_uv_area(
    atlas_outputs: Sequence[_AtlasOutput],
    *,
    target_domain: VisibilityUvTargetDomain = UV_TARGET_DOMAIN_FULL,
) -> tuple[float, float]:
    area_by_visibility = {False: 0.0, True: 0.0}
    for output in atlas_outputs:
        triangles = output.uvs[output.faces]
        vectors_a = triangles[:, 1] - triangles[:, 0]
        vectors_b = triangles[:, 2] - triangles[:, 0]
        areas = np.abs(
            vectors_a[:, 0] * vectors_b[:, 1]
            - vectors_a[:, 1] * vectors_b[:, 0]
        ) / 2.0
        area_by_visibility[output.submission.exterior] += float(np.sum(areas))
    total_area = area_by_visibility[False] + area_by_visibility[True]
    exterior_share = (
        area_by_visibility[True] / total_area if total_area > 0.0 else 0.0
    )
    target_area = (
        0.5 if target_domain == UV_TARGET_DOMAIN_LEFT_HALF else 1.0
    )
    return float(exterior_share), float(total_area / target_area)


def _export_scene(scene: trimesh.Scene) -> bytes:
    try:
        payload = scene.export(file_type="glb")
    except Exception as error:
        raise ValueError(
            "The visibility-unwrapped GLB could not be exported."
        ) from error
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("The visibility-unwrapped GLB export was empty.")
    return payload


def _validate_output(
    glb_bytes: bytes,
    expected_face_count: int,
    *,
    target_domain: VisibilityUvTargetDomain = UV_TARGET_DOMAIN_FULL,
) -> None:
    scene = _load_scene(glb_bytes)
    face_count = sum(
        len(mesh.faces)
        for mesh in scene.geometry.values()
        if isinstance(mesh, trimesh.Trimesh)
    )
    if face_count != expected_face_count:
        raise RuntimeError("Visibility UV unwrapping changed the object's face count.")
    for mesh in scene.geometry.values():
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            continue
        uv = getattr(mesh.visual, "uv", None)
        if uv is None or np.asarray(uv).shape != (len(mesh.vertices), 2):
            raise RuntimeError("Visibility UV unwrapping produced invalid UVs.")
        uv_array = np.asarray(uv, dtype=np.float64)
        if not np.all(np.isfinite(uv_array)) or np.any(uv_array < -1e-7) or np.any(
            uv_array > 1.0 + 1e-7
        ):
            raise RuntimeError("Visibility UV unwrapping produced invalid UVs.")
        if (
            target_domain == UV_TARGET_DOMAIN_LEFT_HALF
            and (
                np.any(
                    uv_array[:, 0]
                    < SYMMETRIC_LEFT_PACKED_UV_SAFETY_INSET - 1e-7
                )
                or np.any(
                    uv_array[:, 0]
                    > 0.5
                    - SYMMETRIC_LEFT_PACKED_UV_SAFETY_INSET
                    + 1e-7
                )
                or np.any(
                    uv_array[:, 1]
                    < SYMMETRIC_LEFT_PACKED_UV_SAFETY_INSET - 1e-7
                )
                or np.any(
                    uv_array[:, 1]
                    > 1.0
                    - SYMMETRIC_LEFT_PACKED_UV_SAFETY_INSET
                    + 1e-7
                )
            )
        ):
            raise RuntimeError(
                "Visibility UV unwrapping did not preserve the symmetric "
                "texture boundary inset."
            )


# ### Numeric and cancellation helpers ###
def _get_vertex_normals(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return cached authored normals or a dependency-free smooth fallback."""

    cached_normals = getattr(mesh, "_cache", None)
    cached_values = getattr(cached_normals, "cache", {}).get("vertex_normals")
    if cached_values is not None:
        normals = np.asarray(cached_values, dtype=np.float64)
        if (
            normals.shape == (len(mesh.vertices), 3)
            and np.all(np.isfinite(normals))
            and np.all(np.linalg.norm(normals, axis=1) > _NORMAL_EPSILON)
        ):
            return normals.copy()

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    face_vectors = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    normals = np.zeros_like(vertices)
    for corner_index in range(3):
        np.add.at(normals, faces[:, corner_index], face_vectors)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > _NORMAL_EPSILON
    normals[valid] /= lengths[valid, np.newaxis]
    normals[~valid] = (0.0, 0.0, 1.0)
    return normals


def _triangle_areas_3d(triangles: np.ndarray) -> np.ndarray:
    return np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    ) / 2.0


def _normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if not math.isfinite(length) or length <= np.finfo(np.float64).eps:
        raise ValueError("The object contains an invalid surface direction.")
    return vector / length


def _check_cancelled(cancellation_check: Callable[[], bool] | None) -> None:
    if cancellation_check is not None and cancellation_check():
        raise VisibilityUvUnwrapCancelled(
            "Visibility UV unwrapping was cancelled."
        )
