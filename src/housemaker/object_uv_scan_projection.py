# ### Imports ###
from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import trimesh
from trimesh.visual.texture import TextureVisuals

from housemaker.glb import GLTF_Y_UP_TO_Z_UP_TRANSFORM
from housemaker.object_texture_variants import (
    _collect_material_textures,
    _load_glb_scene,
    _replace_material_textures,
    _validate_shared_texture,
)
from housemaker.scan_projection_layout import (
    SCAN_PROJECTION_LAYOUT_ALIGNMENT_PIXELS,
    SCAN_PROJECTION_LAYOUT_METADATA_KEY,
    SCAN_PROJECTION_UV_EDGE_INSET_TEXELS,
    build_scan_projection_layout_metadata,
)
from housemaker.unused_face_removal import (
    ALL_CAMERA_IDS,
    FixedCameraView,
    get_fixed_camera_view,
)


# ### Public constants ###
SCAN_PROJECTION_VERSION = 4
SCAN_PROJECTION_TARGET_FULL = "full"
SCAN_PROJECTION_TARGET_LEFT_HALF = "left_half"
SCAN_PROJECTION_TARGET_TOP_LEFT_QUARTER = "top_left_quarter"
DEFAULT_SCAN_PROJECTION_TEXTURE_RESOLUTION = 2048
DEFAULT_PROJECTION_CAMERA_PERCENTAGES = (17, 17, 17, 17, 16, 16)
SCAN_PROJECTION_ISLAND_PADDING_PIXELS = 0
SCAN_PROJECTION_MINIMUM_VISIBLE_FRACTION = 0.5
SCAN_PROJECTION_VISIBILITY_IMAGE_SIZE = 512
LEFT_HALF_OUTER_SAFETY_INSET_PIXELS = 4
TOP_LEFT_QUARTER_OUTER_SAFETY_INSET_PIXELS = 8


# ### Processing constants ###
_TARGET_DOMAINS = frozenset(
    {
        SCAN_PROJECTION_TARGET_FULL,
        SCAN_PROJECTION_TARGET_LEFT_HALF,
        SCAN_PROJECTION_TARGET_TOP_LEFT_QUARTER,
    }
)
_MINIMUM_TEXTURE_RESOLUTION = 32
_MAXIMUM_FACE_COUNT = 200_000
_AREA_EPSILON = 1e-18
_BARYCENTRIC_EPSILON = 1e-9
_MINIMUM_DESTINATION_CELL_EXTENT_PIXELS = 4
_DESTINATION_PIXELS_PER_GROUP = (
    _MINIMUM_DESTINATION_CELL_EXTENT_PIXELS**2
)
_MINIMUM_EXPORTED_UV_TOLERANCE = 1e-9
_CONTINUOUS_EDGE_POSITION_TOLERANCE = 1e-7
_CONTINUOUS_EDGE_UV_TOLERANCE = 1e-7
_FALLBACK_CAMERA_INDEX = len(ALL_CAMERA_IDS)
_CAMERA_AREA_TIE_RELATIVE_TOLERANCE = 1e-6
_MINIMUM_VISIBILITY_SAMPLES_FOR_CAMERA_ALLOCATION = 4
_VISIBILITY_CAMERA_FRAME_MARGIN_RATIO = 0.05
_VISIBILITY_MINIMUM_CAMERA_EXTENT = 1e-9
_VISIBILITY_DEPTH_EPSILON_RATIO = 1e-9
_VISIBILITY_MINIMUM_DEPTH_EPSILON = 1e-12
_OPAQUE_BLACK = np.asarray((0, 0, 0, 255), dtype=np.uint8)
_AUXILIARY_TEXTURE_ATTRIBUTES = (
    "normalTexture",
    "occlusionTexture",
    "emissiveTexture",
    "metallicRoughnessTexture",
    "specularGlossinessTexture",
)


# ### Callback types ###
CancellationCheck = Callable[[], bool]


# ### Public data models ###
@dataclass(frozen=True)
class ScanProjectionStats:
    """Auditable measurements for one weighted six-view scan projection."""

    version: int
    camera_percentages: tuple[int, ...]
    view_face_counts: tuple[int, ...]
    view_pixel_counts: tuple[int, ...]
    face_count: int
    output_face_count: int
    source_vertex_count: int
    output_vertex_count: int
    texture_resolution: int
    target_domain: str
    target_width: int
    target_height: int
    island_padding_pixels: int
    outer_safety_inset_pixels: int
    usable_pixel_count: int
    covered_pixel_count: int
    triangle_occupancy: float
    fallback_face_count: int = 0
    fallback_pixel_count: int = 0

    @property
    def utilization(self) -> float:
        """Return the covered share of the usable target region."""

        return self.triangle_occupancy

    def to_pipeline_dict(self) -> dict[str, object]:
        """Return stable JSON-safe provenance for a generated object."""

        return {
            "version": self.version,
            "camera_percentages": {
                camera_id: percentage
                for camera_id, percentage in zip(
                    ALL_CAMERA_IDS,
                    self.camera_percentages,
                    strict=True,
                )
            },
            "view_face_counts": {
                camera_id: count
                for camera_id, count in zip(
                    ALL_CAMERA_IDS,
                    self.view_face_counts,
                    strict=True,
                )
            },
            "view_pixel_counts": {
                camera_id: count
                for camera_id, count in zip(
                    ALL_CAMERA_IDS,
                    self.view_pixel_counts,
                    strict=True,
                )
            },
            "face_count": self.face_count,
            "output_face_count": self.output_face_count,
            "source_vertex_count": self.source_vertex_count,
            "output_vertex_count": self.output_vertex_count,
            "texture_resolution": self.texture_resolution,
            "target_domain": self.target_domain,
            "target_width": self.target_width,
            "target_height": self.target_height,
            "island_padding_pixels": self.island_padding_pixels,
            "outer_safety_inset_pixels": self.outer_safety_inset_pixels,
            "usable_pixel_count": self.usable_pixel_count,
            "covered_pixel_count": self.covered_pixel_count,
            "utilization": self.utilization,
            "triangle_occupancy": self.triangle_occupancy,
            "fallback_face_count": self.fallback_face_count,
            "fallback_pixel_count": self.fallback_pixel_count,
        }


@dataclass(frozen=True)
class ScanProjectionResult:
    """A texture-preserving GLB rebuilt onto a dense scan atlas."""

    glb_bytes: bytes
    stats: ScanProjectionStats

    def __post_init__(self) -> None:
        if not isinstance(self.glb_bytes, bytes) or not self.glb_bytes:
            raise ValueError("The scan-projected GLB is empty.")
        if not isinstance(self.stats, ScanProjectionStats):
            raise TypeError("Scan projection statistics are invalid.")


class ScanProjectionCancelled(RuntimeError):
    """Raised when a caller cancels local scan projection."""


# ### Internal data models ###
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
    def bottom(self) -> int:
        return self.y + self.height


@dataclass(frozen=True)
class _ProjectedFace:
    geometry_name: object
    local_face_index: int
    camera_index: int
    source_positions: np.ndarray
    source_normals: np.ndarray
    source_uvs: np.ndarray
    face_material_index: int | None
    projected_area: float
    visible_fraction: float = 1.0

    @property
    def stable_key(self) -> tuple[str, int]:
        return str(self.geometry_name), self.local_face_index


@dataclass(frozen=True)
class _FaceGroup:
    faces: tuple[_ProjectedFace, ...]
    weight: float


@dataclass(frozen=True)
class _FacePlacement:
    face: _ProjectedFace
    fragment_index: int
    source_positions: np.ndarray
    source_normals: np.ndarray
    source_uvs: np.ndarray
    destination_points: np.ndarray


@dataclass(frozen=True)
class _SceneGeometry:
    geometry_name: object
    mesh: trimesh.Trimesh
    world_vertices: np.ndarray


# ### Public validation API ###
def normalize_projection_camera_percentages(
    values: Sequence[int] | Mapping[str, int],
) -> tuple[int, ...]:
    """Return six canonical integer percentages whose sum is exactly 100."""

    if isinstance(values, Mapping):
        unknown_ids = set(values) - set(ALL_CAMERA_IDS)
        missing_ids = set(ALL_CAMERA_IDS) - set(values)
        if unknown_ids or missing_ids:
            raise ValueError(
                "Projection camera percentages must contain each canonical "
                "camera exactly once."
            )
        raw_values = tuple(values[camera_id] for camera_id in ALL_CAMERA_IDS)
    elif isinstance(values, str | bytes | bytearray):
        raise ValueError("Projection camera percentages must be a sequence.")
    else:
        try:
            raw_values = tuple(values)
        except TypeError as error:
            raise ValueError(
                "Projection camera percentages must be a sequence."
            ) from error
    if len(raw_values) != len(ALL_CAMERA_IDS):
        raise ValueError("Projection camera percentages require six values.")
    normalized: list[int] = []
    for value in raw_values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                "Projection camera percentages must be integers."
            )
        if not 0 <= value <= 100:
            raise ValueError(
                "Projection camera percentages must be between 0 and 100."
            )
        normalized.append(int(value))
    if sum(normalized) != 100:
        raise ValueError(
            "Projection camera percentages must add up to exactly 100%."
        )
    return tuple(normalized)


# ### Public projection API ###
def scan_project_textured_glb(
    glb_bytes: bytes,
    camera_percentages: Sequence[int] | Mapping[str, int] = (
        DEFAULT_PROJECTION_CAMERA_PERCENTAGES
    ),
    *,
    target_domain: str = SCAN_PROJECTION_TARGET_FULL,
    texture_resolution: int = DEFAULT_SCAN_PROJECTION_TEXTURE_RESOLUTION,
    cancellation_check: CancellationCheck | None = None,
) -> ScanProjectionResult:
    """Rebuild UVs and continuously bake source texels into a dense atlas.

    A face qualifies for a canonical view only when at least half of its
    projected samples lie at that view's top depth. Among qualifying views,
    the most front-facing projection wins independently of triangle winding.
    Each view normally receives a horizontal atlas band matching its
    requested integer percentage. Faces without a qualifying view occupy
    only a minimum-sized fallback band. At exact capacity, all bands may share
    scan rows instead.
    Every covered pixel pulls one bilinear sample from the original texture;
    no source pixel is ever splatted into the destination.
    """

    payload = bytes(glb_bytes)
    if not payload:
        raise ValueError("Scan projection requires a non-empty textured GLB.")
    percentages = normalize_projection_camera_percentages(camera_percentages)
    normalized_resolution = _validate_projection_options(
        target_domain=target_domain,
        texture_resolution=texture_resolution,
    )
    _raise_if_cancelled(cancellation_check)
    scene = _load_glb_scene(payload)
    _reject_auxiliary_material_textures(scene)
    source_textures = _collect_material_textures(scene)
    if not source_textures:
        raise ValueError("Scan projection requires an embedded base-color atlas.")
    source_rgba = _validate_shared_texture(source_textures).copy()
    if source_rgba.shape != (
        normalized_resolution,
        normalized_resolution,
        4,
    ):
        raise ValueError(
            "The source base-color atlas must match the requested square "
            "texture resolution."
        )
    geometries = _collect_scene_geometries(scene)
    projected_faces = _assign_faces_to_cameras(
        geometries,
        percentages,
        cancellation_check,
    )
    if not projected_faces:
        raise ValueError("Scan projection requires at least one triangle face.")
    if len(projected_faces) > _MAXIMUM_FACE_COUNT:
        raise ValueError(
            f"Scan projection supports at most {_MAXIMUM_FACE_COUNT} faces."
        )
    target = _build_target_rectangle(normalized_resolution, target_domain)
    placements = _build_face_placements(
        projected_faces,
        percentages,
        target,
    )
    baked_rgba, owner_camera = _bake_scanlines(
        placements,
        source_rgba,
        normalized_resolution,
        cancellation_check,
    )
    output_vertex_count = _apply_face_placements(
        scene,
        geometries,
        placements,
        normalized_resolution,
    )
    _replace_material_textures(scene, [baked_rgba] * len(source_textures))
    _raise_if_cancelled(cancellation_check)
    output_glb = _export_scene(scene)
    _validate_exported_scene(
        output_glb,
        expected_face_count=len(placements),
        target=target,
        texture_resolution=normalized_resolution,
    )

    usable_owner_camera = owner_camera[
        target.y : target.bottom,
        target.x : target.right,
    ]
    view_face_counts = tuple(
        sum(face.camera_index == camera_index for face in projected_faces)
        for camera_index in range(len(ALL_CAMERA_IDS))
    )
    view_pixel_counts = tuple(
        int(np.count_nonzero(usable_owner_camera == camera_index))
        for camera_index in range(len(ALL_CAMERA_IDS))
    )
    fallback_face_count = sum(
        face.camera_index == _FALLBACK_CAMERA_INDEX
        for face in projected_faces
    )
    fallback_pixel_count = int(
        np.count_nonzero(usable_owner_camera == _FALLBACK_CAMERA_INDEX)
    )
    covered_pixel_count = int(np.count_nonzero(usable_owner_camera >= 0))
    usable_pixel_count = target.width * target.height
    source_vertex_count = sum(
        len(geometry.mesh.vertices) for geometry in geometries
    )
    return ScanProjectionResult(
        glb_bytes=output_glb,
        stats=ScanProjectionStats(
            version=SCAN_PROJECTION_VERSION,
            camera_percentages=percentages,
            view_face_counts=view_face_counts,
            view_pixel_counts=view_pixel_counts,
            face_count=len(projected_faces),
            output_face_count=len(placements),
            source_vertex_count=source_vertex_count,
            output_vertex_count=output_vertex_count,
            texture_resolution=normalized_resolution,
            target_domain=target_domain,
            target_width=target.width,
            target_height=target.height,
            island_padding_pixels=SCAN_PROJECTION_ISLAND_PADDING_PIXELS,
            outer_safety_inset_pixels=(
                _target_outer_safety_inset_pixels(target_domain)
            ),
            usable_pixel_count=usable_pixel_count,
            covered_pixel_count=covered_pixel_count,
            triangle_occupancy=(
                covered_pixel_count / usable_pixel_count
                if usable_pixel_count
                else 0.0
            ),
            fallback_face_count=fallback_face_count,
            fallback_pixel_count=fallback_pixel_count,
        ),
    )


# ### Option validation helpers ###
def _validate_projection_options(
    *,
    target_domain: object,
    texture_resolution: object,
) -> int:
    if target_domain not in _TARGET_DOMAINS:
        raise ValueError("Unknown scan-projection target domain.")
    if isinstance(texture_resolution, bool) or not isinstance(
        texture_resolution,
        int,
    ):
        raise ValueError("Scan-projection texture resolution must be an integer.")
    if texture_resolution < _MINIMUM_TEXTURE_RESOLUTION:
        raise ValueError(
            "Scan-projection texture resolution must be at least "
            f"{_MINIMUM_TEXTURE_RESOLUTION}."
        )
    if texture_resolution % SCAN_PROJECTION_LAYOUT_ALIGNMENT_PIXELS:
        raise ValueError(
            "Scan-projection texture resolution must be even and divisible "
            f"by {SCAN_PROJECTION_LAYOUT_ALIGNMENT_PIXELS}."
        )
    if (
        target_domain != SCAN_PROJECTION_TARGET_FULL
        and texture_resolution
        % (2 * SCAN_PROJECTION_LAYOUT_ALIGNMENT_PIXELS)
    ):
        raise ValueError(
            "Compact scan-projection texture resolution must be divisible "
            f"by {2 * SCAN_PROJECTION_LAYOUT_ALIGNMENT_PIXELS}."
        )
    if (
        target_domain == SCAN_PROJECTION_TARGET_LEFT_HALF
        and texture_resolution // 2
        <= 2 * LEFT_HALF_OUTER_SAFETY_INSET_PIXELS
    ):
        raise ValueError("The left-half safety inset leaves no texture space.")
    if (
        target_domain == SCAN_PROJECTION_TARGET_TOP_LEFT_QUARTER
        and texture_resolution // 2
        <= 2 * TOP_LEFT_QUARTER_OUTER_SAFETY_INSET_PIXELS
    ):
        raise ValueError(
            "The top-left-quarter safety inset leaves no texture space."
        )
    return texture_resolution


# ### Material validation helpers ###
def _reject_auxiliary_material_textures(scene: trimesh.Scene) -> None:
    found_textured_material = False
    for geometry in scene.geometry.values():
        if not isinstance(geometry, trimesh.Trimesh) or not len(geometry.faces):
            continue
        material = getattr(getattr(geometry, "visual", None), "material", None)
        if material is None:
            raise ValueError("Every scan-projected mesh must be textured.")
        leaf_materials = _iter_material_leaves(material)
        if not leaf_materials:
            raise ValueError("Every scan-projected mesh must be textured.")
        for leaf in leaf_materials:
            if any(
                getattr(leaf, attribute_name, None) is not None
                for attribute_name in _AUXILIARY_TEXTURE_ATTRIBUTES
            ):
                raise ValueError(
                    "Scan projection does not support auxiliary material "
                    "textures because they cannot safely retain the new UVs."
                )
            if not any(
                getattr(leaf, attribute_name, None) is not None
                for attribute_name in ("baseColorTexture", "image")
            ):
                raise ValueError(
                    "Every scan-projected material needs a base-color texture."
                )
            found_textured_material = True
    if not found_textured_material:
        raise ValueError("Scan projection requires textured triangle meshes.")


def _iter_material_leaves(material: object) -> tuple[object, ...]:
    nested = getattr(material, "materials", None)
    if isinstance(nested, list | tuple):
        return tuple(
            leaf
            for nested_material in nested
            for leaf in _iter_material_leaves(nested_material)
        )
    return (material,)


# ### Scene collection helpers ###
def _collect_scene_geometries(scene: trimesh.Scene) -> list[_SceneGeometry]:
    geometries: list[_SceneGeometry] = []
    referenced_names: set[object] = set()
    for node_name in sorted(scene.graph.nodes_geometry, key=str):
        transform, geometry_name = scene.graph.get(node_name)
        if geometry_name in referenced_names:
            raise ValueError(
                "Scan projection does not support geometry shared by multiple "
                "scene nodes."
            )
        geometry = scene.geometry.get(geometry_name)
        if not isinstance(geometry, trimesh.Trimesh):
            continue
        referenced_names.add(geometry_name)
        vertices = np.asarray(geometry.vertices, dtype=float)
        faces = np.asarray(geometry.faces, dtype=np.int64)
        if not len(vertices) or not len(faces):
            continue
        if faces.ndim != 2 or faces.shape[1:] != (3,):
            raise ValueError("Scan projection requires triangle faces.")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("Scan projection found non-finite vertices.")
        if np.any(faces < 0) or np.any(faces >= len(vertices)):
            raise ValueError("Scan projection found invalid face indices.")
        node_transform = np.asarray(transform, dtype=float)
        if node_transform.shape != (4, 4) or not np.all(
            np.isfinite(node_transform)
        ):
            raise ValueError("Scan projection found an invalid node transform.")
        uvs = np.asarray(getattr(geometry.visual, "uv", None), dtype=float)
        if uvs.shape != (len(vertices), 2) or not np.all(np.isfinite(uvs)):
            raise ValueError("Every scan-projected mesh needs finite UVs.")
        gltf_world_vertices = trimesh.transform_points(vertices, node_transform)
        world_vertices = trimesh.transform_points(
            gltf_world_vertices,
            GLTF_Y_UP_TO_Z_UP_TRANSFORM,
        )
        geometries.append(
            _SceneGeometry(
                geometry_name=geometry_name,
                mesh=geometry,
                world_vertices=np.asarray(world_vertices, dtype=float),
            )
        )
    if not geometries:
        raise ValueError("Scan projection found no triangle mesh geometry.")
    return geometries


# ### Camera assignment helpers ###
def _assign_faces_to_cameras(
    geometries: Sequence[_SceneGeometry],
    percentages: tuple[int, ...],
    cancellation_check: CancellationCheck | None,
) -> list[_ProjectedFace]:
    """Assign sufficiently visible faces without trusting triangle winding."""

    camera_views = tuple(
        get_fixed_camera_view(camera_id) for camera_id in ALL_CAMERA_IDS
    )
    camera_directions = np.asarray(
        [camera_view.depth_axis for camera_view in camera_views],
        dtype=float,
    )
    enabled_indices = np.flatnonzero(np.asarray(percentages, dtype=int) > 0)
    world_triangles = np.concatenate(
        tuple(
            geometry.world_vertices[
                np.asarray(geometry.mesh.faces, dtype=np.int64)
            ]
            for geometry in geometries
        ),
        axis=0,
    )
    weighted_normals = np.cross(
        world_triangles[:, 1] - world_triangles[:, 0],
        world_triangles[:, 2] - world_triangles[:, 0],
    )
    projected_areas = (
        np.abs(weighted_normals @ camera_directions.T) * 0.5
    )
    visibility_fractions = _measure_face_visibility_fractions(
        world_triangles,
        camera_views,
        enabled_indices,
        cancellation_check,
    )
    camera_assignments = tuple(
        _select_face_camera(
            projected_areas[face_index],
            visibility_fractions[face_index],
            percentages,
            enabled_indices,
        )
        for face_index in range(len(world_triangles))
    )

    projected_faces: list[_ProjectedFace] = []
    global_face_offset = 0
    for geometry in geometries:
        faces = np.asarray(geometry.mesh.faces, dtype=np.int64)
        local_vertices = np.asarray(geometry.mesh.vertices, dtype=float)
        local_normals = _get_source_vertex_normals(geometry.mesh)
        source_uvs = np.asarray(geometry.mesh.visual.uv, dtype=float)
        raw_face_materials = getattr(
            geometry.mesh.visual,
            "face_materials",
            None,
        )
        face_materials = (
            None
            if raw_face_materials is None
            else np.asarray(raw_face_materials, dtype=np.int64)
        )
        if face_materials is not None and face_materials.shape != (len(faces),):
            raise ValueError("Scan projection found invalid face materials.")
        for face_index, face in enumerate(faces):
            if face_index % 128 == 0:
                _raise_if_cancelled(cancellation_check)
            global_face_index = global_face_offset + face_index
            camera_index = camera_assignments[global_face_index]
            best_projected_area = float(
                np.max(projected_areas[global_face_index, enabled_indices])
            )
            selected_projected_area = (
                best_projected_area
                if camera_index == _FALLBACK_CAMERA_INDEX
                else float(projected_areas[global_face_index, camera_index])
            )
            visible_fraction = (
                float(
                    np.max(
                        visibility_fractions[
                            global_face_index,
                            enabled_indices,
                        ]
                    )
                )
                if camera_index == _FALLBACK_CAMERA_INDEX
                else float(
                    visibility_fractions[global_face_index, camera_index]
                )
            )
            projected_area = max(
                selected_projected_area,
                _AREA_EPSILON,
            )
            projected_faces.append(
                _ProjectedFace(
                    geometry_name=geometry.geometry_name,
                    local_face_index=face_index,
                    camera_index=camera_index,
                    source_positions=local_vertices[face].copy(),
                    source_normals=local_normals[face].copy(),
                    source_uvs=source_uvs[face].copy(),
                    face_material_index=(
                        None
                        if face_materials is None
                        else int(face_materials[face_index])
                    ),
                    projected_area=projected_area,
                    visible_fraction=visible_fraction,
                )
            )
        global_face_offset += len(faces)
    return projected_faces


def _measure_face_visibility_fractions(
    world_triangles: np.ndarray,
    camera_views: Sequence[FixedCameraView],
    enabled_indices: np.ndarray,
    cancellation_check: CancellationCheck | None,
) -> np.ndarray:
    """Return top-depth samples divided by all samples for every face."""

    fractions = np.zeros(
        (len(world_triangles), len(camera_views)),
        dtype=float,
    )
    bounds_diagonal = float(
        np.linalg.norm(
            np.ptp(world_triangles.reshape((-1, 3)), axis=0)
        )
    )
    depth_epsilon = max(
        bounds_diagonal * _VISIBILITY_DEPTH_EPSILON_RATIO,
        _VISIBILITY_MINIMUM_DEPTH_EPSILON,
    )
    for camera_index in enabled_indices:
        _raise_if_cancelled(cancellation_check)
        screen_triangles, triangle_depths = (
            _project_visibility_camera_triangles(
                world_triangles,
                camera_views[int(camera_index)],
            )
        )
        top_depth = np.full(
            (
                SCAN_PROJECTION_VISIBILITY_IMAGE_SIZE,
                SCAN_PROJECTION_VISIBILITY_IMAGE_SIZE,
            ),
            -np.inf,
            dtype=float,
        )
        for face_index, (screen_points, depths) in enumerate(
            zip(screen_triangles, triangle_depths, strict=True)
        ):
            if face_index % 128 == 0:
                _raise_if_cancelled(cancellation_check)
            rows, columns, sample_depths = _rasterize_visibility_face_samples(
                screen_points,
                depths,
            )
            if not len(rows):
                continue
            current_depths = top_depth[rows, columns]
            nearer = sample_depths > current_depths + depth_epsilon
            if not np.any(nearer):
                continue
            top_depth[rows[nearer], columns[nearer]] = sample_depths[nearer]

        projected_sample_counts = np.zeros(
            len(world_triangles),
            dtype=np.int64,
        )
        visible_sample_counts = np.zeros(
            len(world_triangles),
            dtype=np.int64,
        )
        for face_index, (screen_points, depths) in enumerate(
            zip(screen_triangles, triangle_depths, strict=True)
        ):
            if face_index % 128 == 0:
                _raise_if_cancelled(cancellation_check)
            rows, columns, sample_depths = _rasterize_visibility_face_samples(
                screen_points,
                depths,
            )
            projected_sample_counts[face_index] = len(rows)
            if not len(rows):
                continue
            visible_sample_counts[face_index] = int(
                np.count_nonzero(
                    sample_depths
                    >= top_depth[rows, columns] - depth_epsilon
                )
            )
        # Fewer samples cannot distinguish a genuinely exposed tiny face from
        # one accidental raster hit through a sub-pixel gap. Such faces remain
        # in the explicitly minimal fallback pool instead of consuming a
        # camera's user-requested percentage.
        valid = (
            projected_sample_counts
            >= _MINIMUM_VISIBILITY_SAMPLES_FOR_CAMERA_ALLOCATION
        )
        fractions[valid, int(camera_index)] = (
            visible_sample_counts[valid] / projected_sample_counts[valid]
        )
    return fractions


def _project_visibility_camera_triangles(
    world_triangles: np.ndarray,
    camera_view: FixedCameraView,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world triangles into one shared square orthographic capture."""

    vertices = world_triangles.reshape((-1, 3))
    horizontal_axis = np.asarray(camera_view.horizontal_axis, dtype=float)
    vertical_axis = np.asarray(camera_view.vertical_axis, dtype=float)
    depth_axis = np.asarray(camera_view.depth_axis, dtype=float)
    horizontal = vertices @ horizontal_axis
    vertical = vertices @ vertical_axis
    depths = vertices @ depth_axis
    horizontal_center = (
        float(np.min(horizontal)) + float(np.max(horizontal))
    ) * 0.5
    vertical_center = (
        float(np.min(vertical)) + float(np.max(vertical))
    ) * 0.5
    extent = max(
        float(np.ptp(horizontal)) * 0.5,
        float(np.ptp(vertical)) * 0.5,
        _VISIBILITY_MINIMUM_CAMERA_EXTENT,
    )
    extent *= 1.0 + _VISIBILITY_CAMERA_FRAME_MARGIN_RATIO
    image_span = SCAN_PROJECTION_VISIBILITY_IMAGE_SIZE - 1.0
    scale = image_span / (2.0 * extent)
    screen_vertices = np.column_stack(
        (
            (horizontal - horizontal_center) * scale + image_span * 0.5,
            (vertical_center - vertical) * scale + image_span * 0.5,
        )
    )
    return (
        screen_vertices.reshape((-1, 3, 2)),
        depths.reshape((-1, 3)),
    )


def _rasterize_visibility_face_samples(
    screen_points: np.ndarray,
    depths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic covered pixels and interpolated face depths."""

    image_size = SCAN_PROJECTION_VISIBILITY_IMAGE_SIZE
    minimum_x = max(0, int(math.floor(float(np.min(screen_points[:, 0])))))
    maximum_x = min(
        image_size - 1,
        int(math.ceil(float(np.max(screen_points[:, 0])))),
    )
    minimum_y = max(0, int(math.floor(float(np.min(screen_points[:, 1])))))
    maximum_y = min(
        image_size - 1,
        int(math.ceil(float(np.max(screen_points[:, 1])))),
    )
    denominator = _cross_2d(
        screen_points[1] - screen_points[0],
        screen_points[2] - screen_points[0],
    )
    if (
        maximum_x < minimum_x
        or maximum_y < minimum_y
        or abs(denominator) <= _AREA_EPSILON
    ):
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=float),
        )

    mask = np.zeros(
        (maximum_y - minimum_y + 1, maximum_x - minimum_x + 1),
        dtype=np.uint8,
    )
    local_polygon = np.rint(
        screen_points - np.asarray((minimum_x, minimum_y), dtype=float)
    ).astype(np.int32)
    cv2.fillConvexPoly(mask, local_polygon, 255, lineType=cv2.LINE_8)
    # Integer-rounded OpenCV edges are only a fast candidate filter. Expand
    # them by one pixel so the exact barycentric test cannot lose a valid
    # pixel center near a sub-pixel triangle boundary.
    mask = cv2.dilate(mask, None, iterations=1)
    local_rows, local_columns = np.nonzero(mask)
    rows = local_rows.astype(np.int64) + minimum_y
    columns = local_columns.astype(np.int64) + minimum_x
    sample_points = np.column_stack(
        (columns.astype(float) + 0.5, rows.astype(float) + 0.5)
    )
    barycentric = _triangle_barycentric_weights(
        sample_points,
        screen_points,
    )
    inside = np.all(barycentric >= -_BARYCENTRIC_EPSILON, axis=1)
    if np.any(inside):
        rows = rows[inside]
        columns = columns[inside]
        barycentric = barycentric[inside]
    else:
        # A synthetic centroid pixel could both make a sub-pixel face appear
        # 100% visible and incorrectly occlude a neighboring real sample.
        # With no covered pixel center, the conservative fallback pool is the
        # only allocation whose visibility can be supported by this capture.
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=float),
        )
    return rows, columns, barycentric @ depths


def _select_face_camera(
    projected_areas: np.ndarray,
    visibility_fractions: np.ndarray,
    percentages: tuple[int, ...],
    enabled_indices: np.ndarray,
) -> int:
    """Choose the highest-quality qualifying view with stable tie breaks."""

    qualified = enabled_indices[
        visibility_fractions[enabled_indices]
        >= SCAN_PROJECTION_MINIMUM_VISIBLE_FRACTION
    ]
    qualified = qualified[projected_areas[qualified] > _AREA_EPSILON]
    if not len(qualified):
        return _FALLBACK_CAMERA_INDEX
    maximum_area = float(np.max(projected_areas[qualified]))
    near_maximum = qualified[
        projected_areas[qualified]
        >= maximum_area * (1.0 - _CAMERA_AREA_TIE_RELATIVE_TOLERANCE)
    ]
    return min(
        (int(camera_index) for camera_index in near_maximum),
        key=lambda camera_index: (-percentages[camera_index], camera_index),
    )


# ### Target layout helpers ###
def _target_outer_safety_inset_pixels(target_domain: str) -> int:
    if target_domain == SCAN_PROJECTION_TARGET_LEFT_HALF:
        return LEFT_HALF_OUTER_SAFETY_INSET_PIXELS
    if target_domain == SCAN_PROJECTION_TARGET_TOP_LEFT_QUARTER:
        return TOP_LEFT_QUARTER_OUTER_SAFETY_INSET_PIXELS
    return 0


def _build_target_rectangle(
    texture_resolution: int,
    target_domain: str,
) -> _PixelRectangle:
    if target_domain == SCAN_PROJECTION_TARGET_FULL:
        return _PixelRectangle(0, 0, texture_resolution, texture_resolution)
    if target_domain == SCAN_PROJECTION_TARGET_TOP_LEFT_QUARTER:
        inset = TOP_LEFT_QUARTER_OUTER_SAFETY_INSET_PIXELS
        return _PixelRectangle(
            inset,
            inset,
            texture_resolution // 2 - 2 * inset,
            texture_resolution // 2 - 2 * inset,
        )
    inset = LEFT_HALF_OUTER_SAFETY_INSET_PIXELS
    return _PixelRectangle(
        inset,
        inset,
        texture_resolution // 2 - 2 * inset,
        texture_resolution - 2 * inset,
    )


def _validate_aligned_rectangle(
    rectangle: _PixelRectangle,
    description: str,
) -> None:
    """Require boundaries that survive exact 4-to-1 texture reduction."""

    alignment = SCAN_PROJECTION_LAYOUT_ALIGNMENT_PIXELS
    values = (
        rectangle.x,
        rectangle.y,
        rectangle.width,
        rectangle.height,
    )
    if any(value % alignment for value in values):
        raise ValueError(
            f"The {description} must align to {alignment}-pixel boundaries."
        )


def _build_face_placements(
    faces: Sequence[_ProjectedFace],
    percentages: tuple[int, ...],
    target: _PixelRectangle,
) -> list[_FacePlacement]:
    """Allocate capacity-safe camera regions and build triangle placements."""

    _validate_aligned_rectangle(target, "scan target")

    camera_faces = tuple(
        tuple(
            sorted(
                (
                    face
                    for face in faces
                    if face.camera_index == camera_index
                ),
                key=lambda face: (-face.projected_area, face.stable_key),
            )
        )
        for camera_index in range(_FALLBACK_CAMERA_INDEX + 1)
    )
    camera_groups = tuple(
        tuple(_group_camera_faces(view_faces))
        for view_faces in camera_faces
    )
    group_counts = tuple(len(groups) for groups in camera_groups)
    total_group_count = sum(group_counts)
    required_pixel_count = total_group_count * _DESTINATION_PIXELS_PER_GROUP
    target_pixel_count = target.width * target.height
    if required_pixel_count > target_pixel_count:
        raise ValueError(
            "The scan atlas cannot represent all face groups: "
            f"{total_group_count} groups require at least "
            f"{required_pixel_count} pixels, but the target contains "
            f"{target_pixel_count}. Reduce the model's face count before "
            "texture generation."
        )
    qualified_group_count = sum(group_counts[:_FALLBACK_CAMERA_INDEX])
    effective_percentages = tuple(
        percentage if group_counts[camera_index] else 0
        for camera_index, percentage in enumerate(percentages)
    ) + (
        int(bool(group_counts[_FALLBACK_CAMERA_INDEX]) and not qualified_group_count),
    )
    alignment = SCAN_PROJECTION_LAYOUT_ALIGNMENT_PIXELS
    horizontal_cell_capacity = (
        target.width // _MINIMUM_DESTINATION_CELL_EXTENT_PIXELS
    )
    minimum_band_height_units = tuple(
        (
            math.ceil(group_count / horizontal_cell_capacity)
            * (_MINIMUM_DESTINATION_CELL_EXTENT_PIXELS // alignment)
        )
        if group_count
        else 0
        for group_count in group_counts
    )
    target_height_units = target.height // alignment
    if sum(minimum_band_height_units) <= target_height_units:
        height_units = _allocate_bounded_integer_shares(
            target_height_units,
            effective_percentages,
            minimum_band_height_units,
        )
        heights = tuple(height * alignment for height in height_units)
    else:
        return _build_shared_row_face_placements(
            faces,
            camera_groups,
            effective_percentages,
            group_counts,
            target,
        )
    placements: list[_FacePlacement] = []
    band_y = target.y
    for camera_index, band_height in enumerate(heights):
        if band_height <= 0:
            continue
        band = _PixelRectangle(target.x, band_y, target.width, band_height)
        band_y += band_height
        if not camera_faces[camera_index]:
            continue
        groups = camera_groups[camera_index]
        rectangles = _partition_group_rectangles(band, groups)
        for group, rectangle in zip(groups, rectangles, strict=True):
            _append_group_placements(placements, group, rectangle)
    return _validate_and_sort_face_placements(placements, faces)


def _build_shared_row_face_placements(
    faces: Sequence[_ProjectedFace],
    camera_groups: Sequence[Sequence[_FaceGroup]],
    effective_percentages: tuple[int, ...],
    group_counts: tuple[int, ...],
    target: _PixelRectangle,
) -> list[_FacePlacement]:
    """Pack camera groups together when separate full-width bands cannot fit."""

    pixel_budgets = _allocate_bounded_integer_shares(
        target.width * target.height,
        effective_percentages,
        tuple(
            group_count * _DESTINATION_PIXELS_PER_GROUP
            for group_count in group_counts
        ),
    )
    weighted_groups: list[_FaceGroup] = []
    for camera_index, groups in enumerate(camera_groups):
        if not groups:
            continue
        total_projected_weight = sum(group.weight for group in groups)
        camera_budget = pixel_budgets[camera_index]
        for group in groups:
            weighted_groups.append(
                _FaceGroup(
                    faces=group.faces,
                    weight=(
                        camera_budget * group.weight / total_projected_weight
                    ),
                )
            )
    rectangles = _partition_group_rectangles(target, weighted_groups)
    placements: list[_FacePlacement] = []
    for group, rectangle in zip(
        weighted_groups,
        rectangles,
        strict=True,
    ):
        _append_group_placements(placements, group, rectangle)
    return _validate_and_sort_face_placements(placements, faces)


def _append_group_placements(
    placements: list[_FacePlacement],
    group: _FaceGroup,
    rectangle: _PixelRectangle,
) -> None:
    if len(group.faces) == 1:
        source_fragments = _split_singleton_face(group.faces[0])
    else:
        source_fragments = _orient_continuous_face_pair(
            group.faces[0],
            group.faces[1],
        )
        if source_fragments is None:
            raise RuntimeError(
                "A scan cell paired faces whose texture edge is discontinuous."
            )
    destinations = _triangulate_group_rectangle(
        rectangle,
        len(source_fragments),
    )
    for fragment_index, (source_fragment, destination) in enumerate(
        zip(source_fragments, destinations, strict=True)
    ):
        face, positions, normals, source_uvs = source_fragment
        placements.append(
            _FacePlacement(
                face=face,
                fragment_index=fragment_index,
                source_positions=positions,
                source_normals=normals,
                source_uvs=source_uvs,
                destination_points=destination,
            )
        )


def _validate_and_sort_face_placements(
    placements: Sequence[_FacePlacement],
    faces: Sequence[_ProjectedFace],
) -> list[_FacePlacement]:
    represented_faces = {
        (placement.face.geometry_name, placement.face.local_face_index)
        for placement in placements
    }
    if len(represented_faces) != len(faces):
        raise ValueError(
            "A positive camera percentage is required for every assigned face."
        )
    return sorted(
        placements,
        key=lambda placement: (
            placement.face.stable_key,
            placement.fragment_index,
        ),
    )


def _allocate_bounded_integer_shares(
    total: int,
    weights: Sequence[int | float],
    minimums: Sequence[int],
) -> tuple[int, ...]:
    """Apportion an integer total by weight while honoring lower bounds."""

    if len(weights) != len(minimums) or not weights:
        raise ValueError("Bounded scan shares require matching non-empty inputs.")
    normalized_weights = tuple(float(weight) for weight in weights)
    normalized_minimums = tuple(int(minimum) for minimum in minimums)
    if any(
        not math.isfinite(weight) or weight < 0.0
        for weight in normalized_weights
    ):
        raise ValueError("Bounded scan-share weights must be finite and nonnegative.")
    if any(minimum < 0 for minimum in normalized_minimums):
        raise ValueError("Bounded scan-share minimums must be nonnegative.")
    minimum_total = sum(normalized_minimums)
    if minimum_total > total:
        raise ValueError(
            f"The scan layout needs at least {minimum_total} allocation units, "
            f"but only {total} are available."
        )
    if sum(normalized_weights) <= 0.0:
        raise ValueError("At least one bounded scan-share weight must be positive.")

    exact = [0.0] * len(normalized_weights)
    active = set(range(len(normalized_weights)))
    remaining_total = total
    while active:
        active_weight = sum(
            normalized_weights[index] for index in sorted(active)
        )
        if active_weight <= 0.0:
            for index in sorted(active):
                exact[index] = float(normalized_minimums[index])
            unassigned = remaining_total - sum(
                normalized_minimums[index] for index in active
            )
            for offset in range(unassigned):
                index = sorted(active)[offset % len(active)]
                exact[index] += 1.0
            break
        constrained = tuple(
            index
            for index in sorted(active)
            if (
                remaining_total
                * normalized_weights[index]
                / active_weight
                < normalized_minimums[index]
            )
        )
        if not constrained:
            for index in sorted(active):
                exact[index] = (
                    remaining_total
                    * normalized_weights[index]
                    / active_weight
                )
            break
        for index in constrained:
            exact[index] = float(normalized_minimums[index])
            remaining_total -= normalized_minimums[index]
            active.remove(index)

    shares = [int(math.floor(value)) for value in exact]
    remainder = total - sum(shares)
    order = sorted(
        range(len(shares)),
        key=lambda index: (-(exact[index] - shares[index]), index),
    )
    for index in order[:remainder]:
        shares[index] += 1
    if any(
        share < minimum
        for share, minimum in zip(shares, normalized_minimums, strict=True)
    ):
        raise RuntimeError("Bounded scan-share allocation violated a lower bound.")
    return tuple(shares)


def _split_singleton_face(
    face: _ProjectedFace,
) -> tuple[
    tuple[_ProjectedFace, np.ndarray, np.ndarray, np.ndarray],
    tuple[_ProjectedFace, np.ndarray, np.ndarray, np.ndarray],
]:
    """Split one source triangle into two texture-equivalent surface pieces."""

    midpoint_position = (face.source_positions[1] + face.source_positions[2]) * 0.5
    midpoint_normal = _normalize_interpolated_normal(
        (face.source_normals[1] + face.source_normals[2]) * 0.5
    )
    midpoint_uv = (face.source_uvs[1] + face.source_uvs[2]) * 0.5
    first_positions = np.asarray(
        (face.source_positions[0], face.source_positions[1], midpoint_position),
        dtype=float,
    )
    second_positions = np.asarray(
        (face.source_positions[0], midpoint_position, face.source_positions[2]),
        dtype=float,
    )
    first_normals = np.asarray(
        (face.source_normals[0], face.source_normals[1], midpoint_normal),
        dtype=float,
    )
    second_normals = np.asarray(
        (face.source_normals[0], midpoint_normal, face.source_normals[2]),
        dtype=float,
    )
    first_uvs = np.asarray(
        (face.source_uvs[0], face.source_uvs[1], midpoint_uv),
        dtype=float,
    )
    second_uvs = np.asarray(
        (face.source_uvs[0], midpoint_uv, face.source_uvs[2]),
        dtype=float,
    )
    return (
        (face, first_positions, first_normals, first_uvs),
        (face, second_positions, second_normals, second_uvs),
    )


def _orient_continuous_face_pair(
    first: _ProjectedFace,
    second: _ProjectedFace,
) -> tuple[
    tuple[_ProjectedFace, np.ndarray, np.ndarray, np.ndarray],
    tuple[_ProjectedFace, np.ndarray, np.ndarray, np.ndarray],
] | None:
    """Orient adjacent textured faces along one continuous cell diagonal."""

    if first.geometry_name != second.geometry_name:
        return None
    cyclic_orders = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    for first_order in cyclic_orders:
        first_positions = first.source_positions[list(first_order)]
        first_uvs = first.source_uvs[list(first_order)]
        for second_order in cyclic_orders:
            second_positions = second.source_positions[list(second_order)]
            second_uvs = second.source_uvs[list(second_order)]
            if not _scan_edges_match(
                first_positions[[0, 2]],
                first_uvs[[0, 2]],
                second_positions[[0, 1]],
                second_uvs[[0, 1]],
            ):
                continue
            return (
                (
                    first,
                    first_positions,
                    first.source_normals[list(first_order)],
                    first_uvs,
                ),
                (
                    second,
                    second_positions,
                    second.source_normals[list(second_order)],
                    second_uvs,
                ),
            )
    return None


def _scan_edges_match(
    first_positions: np.ndarray,
    first_uvs: np.ndarray,
    second_positions: np.ndarray,
    second_uvs: np.ndarray,
) -> bool:
    """Return whether two equally directed edges share geometry and texels."""

    return bool(
        np.allclose(
            first_positions,
            second_positions,
            rtol=0.0,
            atol=_CONTINUOUS_EDGE_POSITION_TOLERANCE,
        )
        and np.allclose(
            first_uvs,
            second_uvs,
            rtol=0.0,
            atol=_CONTINUOUS_EDGE_UV_TOLERANCE,
        )
    )


def _normalize_interpolated_normal(normal: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(normal))
    if length <= _AREA_EPSILON:
        return np.asarray((0.0, 0.0, 1.0), dtype=float)
    return np.asarray(normal, dtype=float) / length


def _group_camera_faces(faces: Sequence[_ProjectedFace]) -> list[_FaceGroup]:
    """Give source faces independent, texture-safe projected-area rectangles.

    A rectangle diagonal always divides its area equally, so pairing unequal
    source faces cannot preserve their requested texel ratio. Equal faces are
    paired only when their shared geometric edge also has continuous source
    UVs. Every other triangle owns a rectangle and is split into two
    texture-preserving fragments. Consequently no unrelated textures ever
    meet along the diagonal inside a scan cell.
    """

    groups: list[_FaceGroup] = []
    face_index = 0
    while face_index < len(faces):
        first = faces[face_index]
        if face_index + 1 < len(faces):
            second = faces[face_index + 1]
            if math.isclose(
                first.projected_area,
                second.projected_area,
                rel_tol=_CAMERA_AREA_TIE_RELATIVE_TOLERANCE,
                abs_tol=_AREA_EPSILON,
            ) and _orient_continuous_face_pair(first, second) is not None:
                groups.append(
                    _FaceGroup(
                        faces=(first, second),
                        weight=first.projected_area + second.projected_area,
                    )
                )
                face_index += 2
                continue
        groups.append(
            _FaceGroup(faces=(first,), weight=first.projected_area)
        )
        face_index += 1
    return groups


def _partition_group_rectangles(
    rectangle: _PixelRectangle,
    groups: Sequence[_FaceGroup],
) -> list[_PixelRectangle]:
    """Tile groups into reduction-aligned cells with safe edge extents."""

    if not groups:
        return []
    _validate_aligned_rectangle(rectangle, "scan region")
    group_count = len(groups)
    minimum_extent = _MINIMUM_DESTINATION_CELL_EXTENT_PIXELS
    groups_per_row = rectangle.width // minimum_extent
    available_row_count = rectangle.height // minimum_extent
    group_capacity = groups_per_row * available_row_count
    if group_count > group_capacity:
        required_pixel_count = group_count * _DESTINATION_PIXELS_PER_GROUP
        raise ValueError(
            "The scan atlas cannot represent all face groups: "
            f"{group_count} groups require at least {required_pixel_count} "
            "reduction-safe pixels, but this region can hold only "
            f"{group_capacity} groups. Reduce the model's face count before "
            "texture generation."
        )

    minimum_row_count = math.ceil(group_count / groups_per_row)
    balanced_row_count = int(
        round(
            math.sqrt(
                group_count * rectangle.height / rectangle.width
            )
        )
    )
    row_count = min(
        available_row_count,
        group_count,
        max(minimum_row_count, balanced_row_count, 1),
    )
    indexed_rows = tuple(
        tuple(
            (group_index, groups[group_index])
            for group_index in range(row_index, group_count, row_count)
        )
        for row_index in range(row_count)
    )
    if any(len(row) > groups_per_row for row in indexed_rows):
        raise RuntimeError("The scan-row layout exceeded its proven capacity.")

    row_weights = tuple(
        sum(group.weight for _group_index, group in row)
        for row in indexed_rows
    )
    alignment = SCAN_PROJECTION_LAYOUT_ALIGNMENT_PIXELS
    minimum_extent_units = minimum_extent // alignment
    row_height_units = _allocate_bounded_integer_shares(
        rectangle.height // alignment,
        row_weights,
        (minimum_extent_units,) * row_count,
    )
    row_heights = tuple(height * alignment for height in row_height_units)
    output: list[_PixelRectangle | None] = [None] * group_count
    row_y = rectangle.y
    for row, row_height in zip(indexed_rows, row_heights, strict=True):
        column_width_units = _allocate_bounded_integer_shares(
            rectangle.width // alignment,
            tuple(group.weight for _group_index, group in row),
            (minimum_extent_units,) * len(row),
        )
        column_widths = tuple(
            width * alignment for width in column_width_units
        )
        column_x = rectangle.x
        for (group_index, _group), column_width in zip(
            row,
            column_widths,
            strict=True,
        ):
            output[group_index] = _PixelRectangle(
                column_x,
                row_y,
                column_width,
                row_height,
            )
            column_x += column_width
        row_y += row_height
    if any(item is None for item in output):
        raise RuntimeError("The scan-row layout lost a face group.")
    return [item for item in output if item is not None]


def _triangulate_group_rectangle(
    rectangle: _PixelRectangle,
    face_count: int,
) -> tuple[np.ndarray, ...]:
    if face_count != 2:
        raise RuntimeError("Every scan cell must contain two face fragments.")
    inset = SCAN_PROJECTION_UV_EDGE_INSET_TEXELS
    x0 = float(rectangle.x) + inset
    y0 = float(rectangle.y) + inset
    x1 = float(rectangle.right) - inset
    y1 = float(rectangle.bottom) - inset
    top_left = np.asarray((x0, y0), dtype=float)
    top_right = np.asarray((x1, y0), dtype=float)
    bottom_right = np.asarray((x1, y1), dtype=float)
    bottom_left = np.asarray((x0, y1), dtype=float)
    return (
        np.asarray((top_left, top_right, bottom_right)),
        np.asarray((top_left, bottom_right, bottom_left)),
    )


# ### Destination scanline baking ###
def _bake_scanlines(
    placements: Sequence[_FacePlacement],
    source_rgba: np.ndarray,
    texture_resolution: int,
    cancellation_check: CancellationCheck | None,
) -> tuple[np.ndarray, np.ndarray]:
    destination = np.empty(
        (texture_resolution, texture_resolution, 4),
        dtype=np.uint8,
    )
    destination[:] = _OPAQUE_BLACK
    owner_camera = np.full(
        (texture_resolution, texture_resolution),
        -1,
        dtype=np.int8,
    )
    for placement_index, placement in enumerate(placements):
        if placement_index % 64 == 0:
            _raise_if_cancelled(cancellation_check)
        _bake_face_scanlines(
            destination,
            owner_camera,
            placement,
            source_rgba,
            cancellation_check,
        )
    return destination, owner_camera


def _bake_face_scanlines(
    destination: np.ndarray,
    owner_camera: np.ndarray,
    placement: _FacePlacement,
    source_rgba: np.ndarray,
    cancellation_check: CancellationCheck | None,
) -> None:
    points = np.asarray(placement.destination_points, dtype=float)
    minimum_x = max(0, int(math.floor(float(np.min(points[:, 0])))))
    maximum_x = min(
        destination.shape[1],
        int(math.ceil(float(np.max(points[:, 0])))),
    )
    minimum_y = max(0, int(math.floor(float(np.min(points[:, 1])))))
    maximum_y = min(
        destination.shape[0],
        int(math.ceil(float(np.max(points[:, 1])))),
    )
    denominator = _cross_2d(points[1] - points[0], points[2] - points[0])
    if abs(denominator) <= _AREA_EPSILON:
        return
    for row in range(minimum_y, maximum_y):
        if row % 64 == 0:
            _raise_if_cancelled(cancellation_check)
        columns = np.arange(minimum_x, maximum_x, dtype=float)
        sample_points = np.column_stack(
            (
                columns + 0.5,
                np.full(len(columns), row + 0.5, dtype=float),
            )
        )
        barycentric = _triangle_barycentric_weights(sample_points, points)
        inside = np.all(barycentric >= -_BARYCENTRIC_EPSILON, axis=1)
        integer_columns = columns.astype(np.int64)
        inside &= owner_camera[row, integer_columns] < 0
        if not np.any(inside):
            continue
        selected_columns = integer_columns[inside]
        selected_weights = barycentric[inside]
        source_uvs = selected_weights @ placement.source_uvs
        destination[row, selected_columns] = _sample_repeat_bilinear_rgba(
            source_rgba,
            source_uvs,
        )
        owner_camera[row, selected_columns] = placement.face.camera_index


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


def _sample_repeat_bilinear_rgba(
    source_rgba: np.ndarray,
    uvs: np.ndarray,
) -> np.ndarray:
    height, width = source_rgba.shape[:2]
    u = np.mod(uvs[:, 0], 1.0)
    v = np.mod(uvs[:, 1], 1.0)
    pixel_x = u * width - 0.5
    pixel_y = (1.0 - v) * height - 0.5
    x0_unwrapped = np.floor(pixel_x).astype(np.int64)
    y0_unwrapped = np.floor(pixel_y).astype(np.int64)
    x1_unwrapped = x0_unwrapped + 1
    y1_unwrapped = y0_unwrapped + 1
    fraction_x = (pixel_x - x0_unwrapped)[:, np.newaxis]
    fraction_y = (pixel_y - y0_unwrapped)[:, np.newaxis]
    x0 = np.mod(x0_unwrapped, width)
    x1 = np.mod(x1_unwrapped, width)
    y0 = np.mod(y0_unwrapped, height)
    y1 = np.mod(y1_unwrapped, height)
    top = (
        source_rgba[y0, x0].astype(float) * (1.0 - fraction_x)
        + source_rgba[y0, x1].astype(float) * fraction_x
    )
    bottom = (
        source_rgba[y1, x0].astype(float) * (1.0 - fraction_x)
        + source_rgba[y1, x1].astype(float) * fraction_x
    )
    samples = top * (1.0 - fraction_y) + bottom * fraction_y
    return np.asarray(np.clip(np.rint(samples), 0, 255), dtype=np.uint8)


# ### Mesh rebuilding helpers ###
def _apply_face_placements(
    scene: trimesh.Scene,
    geometries: Sequence[_SceneGeometry],
    placements: Sequence[_FacePlacement],
    texture_resolution: int,
) -> int:
    placements_by_geometry: dict[object, list[_FacePlacement]] = {}
    for placement in placements:
        placements_by_geometry.setdefault(
            placement.face.geometry_name,
            [],
        ).append(placement)
    output_vertex_count = 0
    for geometry in geometries:
        mesh = geometry.mesh
        geometry_placements = sorted(
            placements_by_geometry[geometry.geometry_name],
            key=lambda placement: (
                placement.face.local_face_index,
                placement.fragment_index,
            ),
        )
        output_vertices: list[np.ndarray] = []
        output_normals: list[np.ndarray] = []
        output_uvs: list[np.ndarray] = []
        output_faces: list[tuple[int, int, int]] = []
        output_face_materials: list[int] = []
        for placement in geometry_placements:
            next_indices: list[int] = []
            for corner_index in range(3):
                next_index = len(output_vertices)
                next_indices.append(next_index)
                output_vertices.append(
                    placement.source_positions[corner_index].copy()
                )
                output_normals.append(
                    placement.source_normals[corner_index].copy()
                )
                point = placement.destination_points[corner_index]
                output_uvs.append(
                    np.asarray(
                        (
                            point[0] / texture_resolution,
                            1.0 - point[1] / texture_resolution,
                        ),
                        dtype=float,
                    )
                )
            output_faces.append(tuple(next_indices))
            if placement.face.face_material_index is not None:
                output_face_materials.append(
                    placement.face.face_material_index
                )
        source_face_materials = getattr(mesh.visual, "face_materials", None)
        if source_face_materials is not None and len(output_face_materials) != len(
            output_faces
        ):
            raise ValueError("Scan projection lost a face material index.")
        rebuilt = trimesh.Trimesh(
            vertices=np.asarray(output_vertices, dtype=float),
            faces=np.asarray(output_faces, dtype=np.int64),
            vertex_normals=np.asarray(output_normals, dtype=float),
            process=False,
            metadata=copy.deepcopy(mesh.metadata),
        )
        rebuilt.visual = TextureVisuals(
            uv=np.asarray(output_uvs, dtype=float),
            material=copy.deepcopy(mesh.visual.material),
            face_materials=(
                None
                if source_face_materials is None
                else np.asarray(output_face_materials, dtype=np.int64)
            ),
        )
        rebuilt.metadata[SCAN_PROJECTION_LAYOUT_METADATA_KEY] = (
            build_scan_projection_layout_metadata(
                canonical_texture_resolution=texture_resolution,
                version=SCAN_PROJECTION_VERSION,
            )
        )
        rebuilt.units = mesh.units
        scene.geometry[geometry.geometry_name] = rebuilt
        output_vertex_count += len(output_vertices)
    return output_vertex_count


def _get_source_vertex_normals(mesh: trimesh.Trimesh) -> np.ndarray:
    mesh_cache = getattr(mesh, "_cache", None)
    cached_normals = getattr(mesh_cache, "cache", {}).get("vertex_normals")
    if cached_normals is not None:
        raw_normals = np.asarray(cached_normals, dtype=float)
        if (
            raw_normals.shape == (len(mesh.vertices), 3)
            and np.all(np.isfinite(raw_normals))
        ):
            return raw_normals.copy()
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    weighted_face_normals = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    normals = np.zeros((len(vertices), 3), dtype=float)
    for corner_index in range(3):
        np.add.at(normals, faces[:, corner_index], weighted_face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > _AREA_EPSILON
    normals[valid] /= lengths[valid, np.newaxis]
    normals[~valid] = (0.0, 0.0, 1.0)
    return normals


# ### Export validation helpers ###
def _export_scene(scene: trimesh.Scene) -> bytes:
    try:
        payload = bytes(scene.export(file_type="glb"))
    except Exception as error:
        raise ValueError("The scan-projected GLB could not be exported.") from error
    if not payload:
        raise ValueError("The scan-projected GLB export was empty.")
    return payload


def _validate_exported_scene(
    payload: bytes,
    *,
    expected_face_count: int,
    target: _PixelRectangle,
    texture_resolution: int,
) -> None:
    scene = _load_glb_scene(payload)
    face_count = 0
    minimum_u = target.x / texture_resolution
    maximum_u = target.right / texture_resolution
    minimum_v = 1.0 - target.bottom / texture_resolution
    maximum_v = 1.0 - target.y / texture_resolution
    # GLB accessors serialize UVs as float32, so exact pixel-edge fractions
    # can move by one representable step during an export/import round trip.
    uv_tolerance = max(
        _MINIMUM_EXPORTED_UV_TOLERANCE,
        float(np.finfo(np.float32).eps)
        * max(
            1.0,
            abs(minimum_u),
            abs(maximum_u),
            abs(minimum_v),
            abs(maximum_v),
        ),
    )
    for geometry in scene.geometry.values():
        if not isinstance(geometry, trimesh.Trimesh):
            continue
        face_count += len(geometry.faces)
        uvs = np.asarray(getattr(geometry.visual, "uv", None), dtype=float)
        if uvs.shape != (len(geometry.vertices), 2):
            raise ValueError("The scan-projected GLB lost its UV coordinates.")
        if not np.all(np.isfinite(uvs)):
            raise ValueError("The scan-projected GLB contains invalid UVs.")
        if (
            np.any(uvs[:, 0] < minimum_u - uv_tolerance)
            or np.any(uvs[:, 0] > maximum_u + uv_tolerance)
            or np.any(uvs[:, 1] < minimum_v - uv_tolerance)
            or np.any(uvs[:, 1] > maximum_v + uv_tolerance)
        ):
            raise ValueError("The scan-projected UVs leave their target region.")
    if face_count != expected_face_count:
        raise ValueError("The scan-projected GLB changed the face count.")


# ### Cancellation helpers ###
def _raise_if_cancelled(
    cancellation_check: CancellationCheck | None,
) -> None:
    if cancellation_check is not None and cancellation_check():
        raise ScanProjectionCancelled("Scan projection was cancelled.")
