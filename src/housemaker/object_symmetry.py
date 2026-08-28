# ### Imports ###
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import trimesh
from trimesh.visual.color import ColorVisuals
from trimesh.visual.texture import TextureVisuals

from housemaker.glb import (
    GLTF_Y_UP_TO_Z_UP_TRANSFORM,
    Z_UP_TO_GLTF_Y_UP_TRANSFORM,
)
from housemaker.object_texture_variants import (
    TEXTURE_RESOLUTION_512,
    TEXTURE_RESOLUTION_1024,
    TEXTURE_RESOLUTION_2048,
    TEXTURE_RESOLUTIONS,
    ObjectTextureVariants,
    _collect_material_textures,
    _encode_rgba_png,
    _load_glb_scene,
    _replace_material_textures,
    _resize_rgba,
    _validate_shared_2048_texture,
)


# ### Constants ###
SYMMETRIC_DIVISION_METADATA_VERSION = 1
SYMMETRIC_QUARTER_METADATA_VERSION = 2
LEGACY_SYMMETRIC_PAIR_METADATA_VERSION = 3
AUTOMATIC_SYMMETRIC_DIVISION_METADATA_VERSION = 4
SYMMETRIC_DIVISION_ORIENTATION_VERTICAL = "vertical"
SYMMETRIC_DIVISION_ORIENTATION_HORIZONTAL = "horizontal"
SYMMETRIC_DIVISION_SIDE_LEFT = "left"
SYMMETRIC_DIVISION_SIDE_RIGHT = "right"
SYMMETRIC_DIVISION_SIDE_BOTTOM = "bottom"
SYMMETRIC_DIVISION_SIDE_TOP = "top"
SYMMETRIC_DIVISION_ORIENTATIONS = frozenset(
    {
        SYMMETRIC_DIVISION_ORIENTATION_VERTICAL,
        SYMMETRIC_DIVISION_ORIENTATION_HORIZONTAL,
    }
)
SYMMETRIC_DIVISION_SIDES_BY_ORIENTATION = {
    SYMMETRIC_DIVISION_ORIENTATION_VERTICAL: frozenset(
        {SYMMETRIC_DIVISION_SIDE_LEFT, SYMMETRIC_DIVISION_SIDE_RIGHT}
    ),
    SYMMETRIC_DIVISION_ORIENTATION_HORIZONTAL: frozenset(
        {SYMMETRIC_DIVISION_SIDE_BOTTOM, SYMMETRIC_DIVISION_SIDE_TOP}
    ),
}
SYMMETRIC_DIVISION_SIDE_ORDER_BY_ORIENTATION = {
    SYMMETRIC_DIVISION_ORIENTATION_VERTICAL: (
        SYMMETRIC_DIVISION_SIDE_LEFT,
        SYMMETRIC_DIVISION_SIDE_RIGHT,
    ),
    SYMMETRIC_DIVISION_ORIENTATION_HORIZONTAL: (
        SYMMETRIC_DIVISION_SIDE_BOTTOM,
        SYMMETRIC_DIVISION_SIDE_TOP,
    ),
}
SYMMETRIC_TEXTURE_PACKING_MODE_TOP_LEFT_QUARTER = "symmetric_quarter"
SYMMETRIC_TEXTURE_PACKING_MODE_PAIR = "symmetric_pair"
SYMMETRIC_TEXTURE_CONTENT_QUADRANT_TOP_LEFT = "top_left"
SYMMETRIC_TEXTURE_CONTENT_HALF_LEFT = "left"
SYMMETRIC_SELECTION_MODE_FEWEST_TRIANGLES_RANDOM_TIE = (
    "fewest_triangles_random_tie"
)
SYMMETRIC_QUARTER_CONTENT_RESOLUTIONS = (
    TEXTURE_RESOLUTION_512,
    TEXTURE_RESOLUTION_1024,
)
SYMMETRIC_QUARTER_ATLAS_RESOLUTION_BY_CONTENT_RESOLUTION = {
    TEXTURE_RESOLUTION_512: TEXTURE_RESOLUTION_1024,
    TEXTURE_RESOLUTION_1024: TEXTURE_RESOLUTION_2048,
}
SYMMETRIC_PAIR_CONTENT_RESOLUTIONS = (
    TEXTURE_RESOLUTION_512,
    TEXTURE_RESOLUTION_1024,
)
SYMMETRIC_PAIR_ATLAS_RESOLUTION_BY_CONTENT_RESOLUTION = {
    TEXTURE_RESOLUTION_512: TEXTURE_RESOLUTION_1024,
    TEXTURE_RESOLUTION_1024: TEXTURE_RESOLUTION_2048,
}
SYMMETRIC_SQUARE_PAIR_CONTENT_RESOLUTIONS = (
    TEXTURE_RESOLUTION_512,
    TEXTURE_RESOLUTION_1024,
)
SYMMETRIC_SQUARE_PAIR_ATLAS_RESOLUTION_BY_CONTENT_RESOLUTION = {
    TEXTURE_RESOLUTION_512: TEXTURE_RESOLUTION_512,
    TEXTURE_RESOLUTION_1024: TEXTURE_RESOLUTION_1024,
}
_CLIP_RELATIVE_EPSILON = 1e-9
_NORMAL_EPSILON = 1e-12
_UV_RASTER_FIXED_POINT_BITS = 8
_UV_PACK_LATTICE_PIXELS = 4
_PRESERVED_UV_SAFETY_PIXELS = _UV_PACK_LATTICE_PIXELS
_UV_GUTTER_PIXELS = 16
_SCALED_UV_GUTTER_PIXELS = 8
_QUARTER_CONTENT_GUTTER_PIXELS_1024 = 8
_MIN_GLOBAL_UV_SCALE = 1.0 / TEXTURE_RESOLUTION_2048
_UNIFORM_SCALE_SEARCH_PASSES = 12
_OPAQUE_BLACK = np.asarray((0, 0, 0, 255), dtype=np.uint8)
_MAX_UV_ISLAND_COUNT = 16_384
_MAX_UV_MASK_PIXELS = 64 * 1024 * 1024
_MAX_UV_OVERLAP_COMPARISONS = 2_000_000
_MAX_UV_OVERLAP_TEST_PIXELS = 128 * 1024 * 1024
_MAX_UNIFORM_SCALE_MAXRECTS_GROUPS = 128
_MAX_REPEAT_TILE_SPAN = 256
_MAX_REPEAT_TILE_MAGNITUDE = 1_000_000
_MAX_REPEAT_TILES_PER_FACE = 4_096
_MAX_REPEAT_OUTPUT_FACES = 262_144
_MAX_REPEAT_CLIP_WORK = 16_384
_SOURCE_NEIGHBORHOOD_SAMPLE_CHUNK = 262_144
_REPEAT_UV_TOLERANCE = 1e-7
_REPEAT_SEAM_TOLERANCE = 1e-6
_REPEAT_CLIP_EPSILON = 1e-12
_REPAIRED_POINT_CHART_EXTENT = 2.0
_REPAIRED_LINE_CHART_HEIGHT = 2.0
_REPAIRED_CHART_MARGIN = 1.0


# ### Public data models ###
SymmetricDivisionOrientation = Literal["vertical", "horizontal"]
SymmetricDivisionKeptSide = Literal["left", "right", "bottom", "top"]


@dataclass(frozen=True)
class SymmetricQuarterTextureVariants:
    """Selectable quarter-content GLBs keyed by logical content resolution."""

    glb_by_resolution: dict[int, bytes]
    texture_png_by_resolution: dict[int, bytes]
    preview_rgba_by_resolution: dict[int, np.ndarray]

    def __post_init__(self) -> None:
        required = set(SYMMETRIC_QUARTER_CONTENT_RESOLUTIONS)
        if set(self.glb_by_resolution) != required:
            raise ValueError(
                "Quarter texture variants require 512 and 1024 GLBs."
            )
        if set(self.texture_png_by_resolution) != required:
            raise ValueError(
                "Quarter texture variants require 512 and 1024 PNGs."
            )
        if set(self.preview_rgba_by_resolution) != required:
            raise ValueError(
                "Quarter texture variants require 512 and 1024 previews."
            )
        for content_resolution, preview in self.preview_rgba_by_resolution.items():
            atlas_resolution = self.atlas_resolution(content_resolution)
            if np.asarray(preview).shape != (
                atlas_resolution,
                atlas_resolution,
                4,
            ):
                raise ValueError(
                    "A quarter texture preview has the wrong physical atlas "
                    "size."
                )

    @property
    def selectable_resolutions(self) -> tuple[int, int]:
        """Return logical texture-content resolutions in ascending order."""

        return SYMMETRIC_QUARTER_CONTENT_RESOLUTIONS

    @staticmethod
    def atlas_resolution(content_resolution: int) -> int:
        """Return the physical square-atlas size for one logical variant."""

        try:
            return SYMMETRIC_QUARTER_ATLAS_RESOLUTION_BY_CONTENT_RESOLUTION[
                content_resolution
            ]
        except KeyError as error:
            raise ValueError(
                "Unknown symmetric quarter-content resolution."
            ) from error


@dataclass(frozen=True)
class SymmetricPairTextureVariants:
    """Legacy selectable left-half GLBs stored in double-sized atlases."""

    glb_by_resolution: dict[int, bytes]
    texture_png_by_resolution: dict[int, bytes]
    preview_rgba_by_resolution: dict[int, np.ndarray]

    def __post_init__(self) -> None:
        required = set(SYMMETRIC_PAIR_CONTENT_RESOLUTIONS)
        if set(self.glb_by_resolution) != required:
            raise ValueError("Pair texture variants require 512 and 1024 GLBs.")
        if set(self.texture_png_by_resolution) != required:
            raise ValueError("Pair texture variants require 512 and 1024 PNGs.")
        if set(self.preview_rgba_by_resolution) != required:
            raise ValueError(
                "Pair texture variants require 512 and 1024 previews."
            )
        for content_resolution, preview in self.preview_rgba_by_resolution.items():
            atlas_resolution = self.atlas_resolution(content_resolution)
            if np.asarray(preview).shape != (
                atlas_resolution,
                atlas_resolution,
                4,
            ):
                raise ValueError(
                    "A pair texture preview has the wrong physical atlas size."
                )

    @property
    def selectable_resolutions(self) -> tuple[int, int]:
        """Return logical texture resolutions in ascending order."""

        return SYMMETRIC_PAIR_CONTENT_RESOLUTIONS

    @staticmethod
    def atlas_resolution(content_resolution: int) -> int:
        """Return the legacy double-sized physical square-atlas size."""

        try:
            return SYMMETRIC_PAIR_ATLAS_RESOLUTION_BY_CONTENT_RESOLUTION[
                content_resolution
            ]
        except KeyError as error:
            raise ValueError("Unknown symmetric pair resolution.") from error


@dataclass(frozen=True)
class SymmetricSquarePairTextureVariants:
    """Selectable left-half GLBs stored at their selected square size."""

    glb_by_resolution: dict[int, bytes]
    texture_png_by_resolution: dict[int, bytes]
    preview_rgba_by_resolution: dict[int, np.ndarray]

    def __post_init__(self) -> None:
        required = set(SYMMETRIC_SQUARE_PAIR_CONTENT_RESOLUTIONS)
        if set(self.glb_by_resolution) != required:
            raise ValueError(
                "Square-pair texture variants require 512 and 1024 GLBs."
            )
        if set(self.texture_png_by_resolution) != required:
            raise ValueError(
                "Square-pair texture variants require 512 and 1024 PNGs."
            )
        if set(self.preview_rgba_by_resolution) != required:
            raise ValueError(
                "Square-pair texture variants require 512 and 1024 previews."
            )
        for resolution, preview in self.preview_rgba_by_resolution.items():
            atlas_resolution = self.atlas_resolution(resolution)
            if np.asarray(preview).shape != (
                atlas_resolution,
                atlas_resolution,
                4,
            ):
                raise ValueError(
                    "A square-pair texture preview has the wrong physical "
                    "atlas size."
                )

    @property
    def selectable_resolutions(self) -> tuple[int, int]:
        """Return selectable physical texture sizes in ascending order."""

        return SYMMETRIC_SQUARE_PAIR_CONTENT_RESOLUTIONS

    @staticmethod
    def atlas_resolution(resolution: int) -> int:
        """Return the physical square-atlas size for one variant."""

        try:
            return (
                SYMMETRIC_SQUARE_PAIR_ATLAS_RESOLUTION_BY_CONTENT_RESOLUTION[
                    resolution
                ]
            )
        except KeyError as error:
            raise ValueError(
                "Unknown symmetric square-pair resolution."
            ) from error


@dataclass(frozen=True)
class SymmetricDivisionMetadata:
    """Serializable provenance needed to mirror the retained half locally."""

    orientation: SymmetricDivisionOrientation
    kept_side: SymmetricDivisionKeptSide
    plane_coordinate: float
    version: int = SYMMETRIC_DIVISION_METADATA_VERSION
    packing_mode: str | None = None
    texture_content_quadrant: str | None = None
    texture_content_half: str | None = None
    selection_mode: str | None = None
    triangle_count_by_side: tuple[tuple[str, int], ...] = ()
    tie_broken_randomly: bool = False

    def __post_init__(self) -> None:
        _validate_orientation_and_side(self.orientation, self.kept_side)
        supported_versions = {
            SYMMETRIC_DIVISION_METADATA_VERSION,
            SYMMETRIC_QUARTER_METADATA_VERSION,
            LEGACY_SYMMETRIC_PAIR_METADATA_VERSION,
            AUTOMATIC_SYMMETRIC_DIVISION_METADATA_VERSION,
        }
        if isinstance(self.version, bool) or self.version not in supported_versions:
            raise ValueError("Unsupported symmetric-division metadata version.")
        if not math.isfinite(self.plane_coordinate):
            raise ValueError("The symmetric-division plane must be finite.")
        if self.version == SYMMETRIC_DIVISION_METADATA_VERSION:
            self._validate_legacy_fields()
        else:
            self._validate_automatic_fields()

    def _validate_legacy_fields(self) -> None:
        if (
            self.packing_mode is not None
            or self.texture_content_quadrant is not None
            or self.texture_content_half is not None
            or self.selection_mode is not None
            or self.triangle_count_by_side
            or self.tie_broken_randomly
        ):
            raise ValueError(
                "Legacy symmetric-division metadata cannot contain automatic "
                "packing fields."
            )

    def _validate_automatic_fields(self) -> None:
        if self.version == SYMMETRIC_QUARTER_METADATA_VERSION:
            if (
                self.packing_mode
                != SYMMETRIC_TEXTURE_PACKING_MODE_TOP_LEFT_QUARTER
                or self.texture_content_quadrant
                != SYMMETRIC_TEXTURE_CONTENT_QUADRANT_TOP_LEFT
                or self.texture_content_half is not None
            ):
                raise ValueError(
                    "Version 2 symmetry must use top-left quarter packing."
                )
        elif (
            self.packing_mode != SYMMETRIC_TEXTURE_PACKING_MODE_PAIR
            or self.texture_content_half
            != SYMMETRIC_TEXTURE_CONTENT_HALF_LEFT
            or self.texture_content_quadrant is not None
        ):
            raise ValueError(
                "Pair symmetry must use left-half texture packing."
            )
        if (
            self.selection_mode
            != SYMMETRIC_SELECTION_MODE_FEWEST_TRIANGLES_RANDOM_TIE
        ):
            raise ValueError("Unknown automatic symmetry selection mode.")
        expected_sides = SYMMETRIC_DIVISION_SIDE_ORDER_BY_ORIENTATION[
            self.orientation
        ]
        actual_sides = tuple(
            side for side, _count in self.triangle_count_by_side
        )
        if actual_sides != expected_sides:
            raise ValueError(
                "Automatic symmetry triangle counts do not match the "
                "orientation."
            )
        counts = tuple(count for _side, count in self.triangle_count_by_side)
        if any(type(count) is not int or count < 0 for count in counts):
            raise ValueError("Automatic symmetry triangle counts are invalid.")
        selected_count = dict(self.triangle_count_by_side)[self.kept_side]
        if selected_count != min(counts):
            raise ValueError("Automatic symmetry did not keep the smaller half.")
        if self.tie_broken_randomly != (counts[0] == counts[1]):
            raise ValueError("Automatic symmetry tie provenance is inconsistent.")

    def to_pipeline_dict(self) -> dict[str, object]:
        """Return the stable JSON-safe pipeline representation."""

        pipeline = {
            "version": self.version,
            "orientation": self.orientation,
            "kept_side": self.kept_side,
            "plane_coordinate": self.plane_coordinate,
        }
        if self.version in {
            SYMMETRIC_QUARTER_METADATA_VERSION,
            LEGACY_SYMMETRIC_PAIR_METADATA_VERSION,
            AUTOMATIC_SYMMETRIC_DIVISION_METADATA_VERSION,
        }:
            pipeline["packing_mode"] = self.packing_mode
            if self.version == SYMMETRIC_QUARTER_METADATA_VERSION:
                pipeline["texture_content_quadrant"] = (
                    self.texture_content_quadrant
                )
            else:
                pipeline["texture_content_half"] = self.texture_content_half
            pipeline.update(
                {
                    "selection_mode": self.selection_mode,
                    "triangle_count_by_side": dict(
                        self.triangle_count_by_side
                    ),
                    "tie_broken_randomly": self.tie_broken_randomly,
                }
            )
        return pipeline


@dataclass(frozen=True)
class SymmetricDivisionResult:
    """Half-only GLB variants plus preview reconstruction provenance."""

    variants: SymmetricSquarePairTextureVariants
    orientation: SymmetricDivisionOrientation
    kept_side: SymmetricDivisionKeptSide
    plane_coordinate: float
    metadata: SymmetricDivisionMetadata

    def __post_init__(self) -> None:
        if not isinstance(
            self.variants,
            SymmetricSquarePairTextureVariants,
        ):
            raise TypeError(
                "Automatic symmetric division requires square-pair variants."
            )
        _validate_orientation_and_side(self.orientation, self.kept_side)
        if not math.isfinite(self.plane_coordinate):
            raise ValueError("The symmetric-division plane must be finite.")
        if (
            self.metadata.orientation != self.orientation
            or self.metadata.kept_side != self.kept_side
            or self.metadata.plane_coordinate != self.plane_coordinate
        ):
            raise ValueError("Symmetric-division metadata does not match its result.")
        if (
            self.metadata.version
            != AUTOMATIC_SYMMETRIC_DIVISION_METADATA_VERSION
        ):
            raise ValueError(
                "Symmetric variants do not match their metadata packing mode."
            )


# ### Internal data models ###
@dataclass(frozen=True)
class _MeshInstance:
    node_name: str
    geometry_name: str
    mesh: trimesh.Trimesh
    transform: np.ndarray
    world_vertices: np.ndarray


@dataclass(frozen=True)
class _ClipVertex:
    local_position: np.ndarray
    world_position: np.ndarray
    normal: np.ndarray
    uv: np.ndarray | None
    vertex_color: np.ndarray | None
    distance: float
    source_indices: tuple[int, ...]


@dataclass(frozen=True)
class _RepeatClipVertex:
    """One source or interpolated corner while splitting a repeated UV tile."""

    local_position: np.ndarray
    normal: np.ndarray
    uv: np.ndarray
    vertex_color: np.ndarray | None
    source_indices: tuple[int, ...]


@dataclass(frozen=True)
class _PixelRectangle:
    """Integer texture-space rectangle with an exclusive right and bottom."""

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
class _RepeatSeamEdges:
    """Record which exact source-atlas seams a UV chart contains."""

    u_zero: bool
    u_one: bool
    v_zero: bool
    v_one: bool


@dataclass(frozen=True)
class _UvIsland:
    """One independently movable UV chart in a rebuilt scene mesh."""

    island_id: int
    geometry_name: object
    face_indices: np.ndarray
    vertex_indices: np.ndarray
    source_bounds: _PixelRectangle
    source_mask: np.ndarray
    exact_repeat_seams: _RepeatSeamEdges
    source_pixel_coordinates: np.ndarray | None = None
    source_rgba: np.ndarray | None = None


@dataclass(frozen=True)
class _UvPackGroup:
    """UV islands which overlap and therefore require one shared transform."""

    group_id: int
    island_ids: tuple[int, ...]
    source_bounds: _PixelRectangle
    source_mask: np.ndarray
    exact_repeat_seams: _RepeatSeamEdges
    source_rgba: np.ndarray | None = None


@dataclass(frozen=True)
class _UvPackPlacement:
    """Canonical-pixel transform shared by one UV pack group."""

    group_id: int
    destination: _PixelRectangle
    rotated_clockwise: bool
    gutter_pixels: int
    scale: float


# ### Public symmetric transforms ###
def build_symmetric_retexture_proxy_glb(
    retained_glb: bytes,
    orientation: SymmetricDivisionOrientation,
    plane_coordinate: float,
) -> bytes:
    """Mirror a retained half solely for Meshy's full-object context.

    The authoritative half keeps its left-atlas UVs. Its temporary mirror uses
    the corresponding right-atlas coordinates, so the full reference image
    and the submitted geometry describe the same complete object. This proxy
    is never persisted or exported as the user's model.
    """

    _validate_orientation(orientation)
    normalized_plane = float(plane_coordinate)
    if not math.isfinite(normalized_plane):
        raise ValueError("The symmetric Retexture plane must be finite.")
    payload = bytes(retained_glb)
    if not payload:
        raise ValueError("The symmetric Retexture source GLB is empty.")

    source_scene = _load_glb_scene(payload)
    _reject_auxiliary_material_textures(source_scene)
    source_textures = _collect_material_textures(source_scene)
    if not source_textures:
        raise ValueError(
            "The symmetric Retexture source has no base-color atlas."
        )
    source_texture = _validate_shared_square_pair_texture(
        source_textures
    ).copy()
    _validate_scene_uvs_in_left_half(source_scene)
    instances = _collect_mesh_instances(source_scene)

    proxy_scene = copy.deepcopy(source_scene)
    occupied_geometry_names = {str(name) for name in proxy_scene.geometry}
    occupied_node_names = {str(name) for name in proxy_scene.graph.nodes}
    axis = _get_z_up_axis(orientation)
    reflection = np.eye(4, dtype=float)
    reflection[axis, axis] = -1.0
    reflection[axis, 3] = 2.0 * normalized_plane
    for instance in instances:
        mirrored_mesh = copy.deepcopy(instance.mesh)
        mirrored_uvs: np.ndarray | None = None
        if _material_textures(mirrored_mesh):
            mirrored_uvs = _get_vertex_uvs(mirrored_mesh)
            if mirrored_uvs is None:
                raise ValueError(
                    "A textured symmetric Retexture mesh is missing UV "
                    "coordinates."
                )
            mirrored_uvs[:, 0] += 0.5
            if np.any(
                mirrored_uvs[:, 0] > 1.0 + _REPEAT_UV_TOLERANCE
            ):
                raise ValueError(
                    "The symmetric Retexture mirror UVs exceed the atlas."
                )
        mirror_transform = (
            Z_UP_TO_GLTF_Y_UP_TRANSFORM
            @ reflection
            @ GLTF_Y_UP_TO_Z_UP_TRANSFORM
            @ instance.transform
        )
        mirrored_mesh.apply_transform(mirror_transform)
        if mirrored_uvs is not None:
            mirrored_mesh.visual.uv = np.clip(mirrored_uvs, 0.0, 1.0)
        geometry_name = _reserve_unique_proxy_name(
            f"{instance.geometry_name}-symmetric-mirror",
            occupied_geometry_names,
        )
        node_name = _reserve_unique_proxy_name(
            f"{instance.node_name}-symmetric-mirror",
            occupied_node_names,
        )
        proxy_scene.add_geometry(
            mirrored_mesh,
            geom_name=geometry_name,
            node_name=node_name,
        )

    half_width = source_texture.shape[1] // 2
    proxy_texture = source_texture.copy()
    proxy_texture[:, half_width:] = source_texture[:, :half_width]
    proxy_textures = _collect_material_textures(proxy_scene)
    _replace_material_textures(
        proxy_scene,
        [proxy_texture] * len(proxy_textures),
    )
    try:
        return bytes(proxy_scene.export(file_type="glb"))
    except Exception as error:
        raise ValueError(
            "The full symmetric Retexture proxy could not be exported."
        ) from error


def build_automatic_symmetric_object_variants(
    canonical_2048_glb: bytes,
    orientation: SymmetricDivisionOrientation,
    *,
    rng: random.Random | None = None,
) -> SymmetricDivisionResult:
    """Clip both midpoint halves and pair-pack the lower-triangle side.

    Exact triangle-count ties use ``rng.choice``. When no source is injected,
    ``random.SystemRandom`` provides a nondeterministic tie choice.
    """

    _validate_orientation(orientation)
    if rng is not None and not callable(getattr(rng, "choice", None)):
        raise TypeError("The automatic symmetry RNG must provide choice().")
    payload = bytes(canonical_2048_glb)
    if not payload:
        raise ValueError("The canonical 2048 GLB is empty.")

    scene = _load_glb_scene(payload)
    _reject_auxiliary_material_textures(scene)
    source_textures = _collect_material_textures(scene)
    if not source_textures:
        raise ValueError(
            "Automatic symmetry requires one embedded base-color texture atlas."
        )
    canonical_texture = _validate_shared_2048_texture(source_textures).copy()
    instances = _collect_mesh_instances(scene)
    axis = _get_z_up_axis(orientation)
    plane_coordinate = _get_world_midpoint(instances, axis)
    ordered_sides = SYMMETRIC_DIVISION_SIDE_ORDER_BY_ORIENTATION[orientation]
    clipped_scene_by_side = {
        side: _build_clipped_scene(
            instances,
            axis=axis,
            plane_coordinate=plane_coordinate,
            kept_side=side,
            metadata=scene.metadata,
        )
        for side in ordered_sides
    }
    triangle_count_by_side = tuple(
        (side, _count_scene_triangles(clipped_scene_by_side[side]))
        for side in ordered_sides
    )
    minimum_count = min(count for _side, count in triangle_count_by_side)
    minimum_sides = tuple(
        side
        for side, count in triangle_count_by_side
        if count == minimum_count
    )
    tie_broken_randomly = len(minimum_sides) > 1
    if tie_broken_randomly:
        random_source = rng if rng is not None else random.SystemRandom()
        kept_side = random_source.choice(minimum_sides)
    else:
        kept_side = minimum_sides[0]
    clipped_scene = clipped_scene_by_side[kept_side]
    _normalize_scene_repeat_uvs(clipped_scene)
    output_textures = _collect_material_textures(clipped_scene)
    if not output_textures:
        raise ValueError(
            "Automatic symmetry removed every textured mesh from the kept half."
        )
    _validate_shared_2048_texture(output_textures)
    packed_texture = _repack_retained_texture(
        clipped_scene,
        canonical_texture,
    )
    variants = _build_square_pair_texture_variants(
        clipped_scene,
        packed_texture,
        len(output_textures),
    )
    metadata = SymmetricDivisionMetadata(
        orientation=orientation,
        kept_side=kept_side,
        plane_coordinate=plane_coordinate,
        version=AUTOMATIC_SYMMETRIC_DIVISION_METADATA_VERSION,
        packing_mode=SYMMETRIC_TEXTURE_PACKING_MODE_PAIR,
        texture_content_half=SYMMETRIC_TEXTURE_CONTENT_HALF_LEFT,
        selection_mode=(
            SYMMETRIC_SELECTION_MODE_FEWEST_TRIANGLES_RANDOM_TIE
        ),
        triangle_count_by_side=triangle_count_by_side,
        tie_broken_randomly=tie_broken_randomly,
    )
    return SymmetricDivisionResult(
        variants=variants,
        orientation=orientation,
        kept_side=kept_side,
        plane_coordinate=plane_coordinate,
        metadata=metadata,
    )


def build_symmetric_pair_texture_variants(
    canonical_2048_glb: bytes,
    *,
    uvs_already_left_packed: bool = False,
) -> SymmetricPairTextureVariants:
    """Pair-pack provider output without clipping half geometry again."""

    if not isinstance(uvs_already_left_packed, bool):
        raise TypeError("The existing left-packed UV mode must be boolean.")
    payload = bytes(canonical_2048_glb)
    if not payload:
        raise ValueError("The retextured canonical 2048 GLB is empty.")
    scene = _load_glb_scene(payload)
    _reject_auxiliary_material_textures(scene)
    source_textures = _collect_material_textures(scene)
    if not source_textures:
        raise ValueError(
            "Symmetric pair packing requires one embedded base-color atlas."
        )
    canonical_texture = _validate_shared_2048_texture(source_textures).copy()
    if uvs_already_left_packed and _scene_uvs_are_left_packed(scene):
        packed_texture = _mask_existing_left_texture(
            scene,
            canonical_texture,
        )
    else:
        _normalize_scene_repeat_uvs(scene)
        packed_texture = _repack_retained_texture(scene, canonical_texture)
    return _build_pair_texture_variants(
        scene,
        packed_texture,
        len(source_textures),
    )


def build_symmetric_square_pair_texture_variants(
    source_glb: bytes,
    *,
    uvs_already_left_packed: bool = False,
) -> SymmetricSquarePairTextureVariants:
    """Build current square pair variants from a 2048 or packed 1024 GLB.

    Each selected resolution is also its physical texture size. Retained UVs
    and pixels occupy the left half, leaving room for one same-size partner.
    Fresh provider output must be 2048. A 1024 operation result is accepted
    only when its UVs are already left-packed, so it is never upscaled.
    """

    if not isinstance(uvs_already_left_packed, bool):
        raise TypeError("The existing left-packed UV mode must be boolean.")
    payload = bytes(source_glb)
    if not payload:
        raise ValueError("The symmetric square-pair source GLB is empty.")
    scene = _load_glb_scene(payload)
    _reject_auxiliary_material_textures(scene)
    source_textures = _collect_material_textures(scene)
    if not source_textures:
        raise ValueError(
            "Symmetric square-pair packing requires one embedded base-color "
            "atlas."
        )
    source_texture = _validate_shared_square_pair_texture(source_textures).copy()
    source_resolution = int(source_texture.shape[0])
    left_packed = (
        uvs_already_left_packed and _scene_uvs_are_left_packed(scene)
    )
    if left_packed and source_resolution == TEXTURE_RESOLUTION_2048:
        packed_texture = _mask_existing_left_texture(scene, source_texture)
    elif left_packed:
        packed_texture = _mask_existing_square_pair_texture(
            scene,
            source_texture,
        )
    elif source_resolution == TEXTURE_RESOLUTION_2048:
        _normalize_scene_repeat_uvs(scene)
        packed_texture = _repack_retained_texture(scene, source_texture)
    else:
        raise ValueError(
            "A 1024 square-pair source requires verified left-packed UVs."
        )
    return _build_square_pair_texture_variants(
        scene,
        packed_texture,
        len(source_textures),
    )


def build_symmetric_quarter_texture_variants(
    canonical_2048_glb: bytes,
    *,
    uvs_already_top_left_quarter: bool = False,
) -> SymmetricQuarterTextureVariants:
    """Quarter-pack provider output without clipping half geometry again."""

    if not isinstance(uvs_already_top_left_quarter, bool):
        raise TypeError("The existing top-left-quarter UV mode must be boolean.")
    payload = bytes(canonical_2048_glb)
    if not payload:
        raise ValueError("The retextured canonical 2048 GLB is empty.")
    scene = _load_glb_scene(payload)
    _reject_auxiliary_material_textures(scene)
    source_textures = _collect_material_textures(scene)
    if not source_textures:
        raise ValueError(
            "Symmetric quarter packing requires one embedded base-color atlas."
        )
    canonical_texture = _validate_shared_2048_texture(source_textures).copy()
    if uvs_already_top_left_quarter:
        _validate_scene_uvs_in_top_left_quarter(scene)
    else:
        _normalize_scene_repeat_uvs(scene)
        _map_scene_uvs_to_top_left_quarter(scene)
    return _build_quarter_texture_variants(
        scene,
        canonical_texture,
        len(source_textures),
        source_is_already_quarter=uvs_already_top_left_quarter,
    )

def build_symmetric_half_texture_variants(
    canonical_2048_glb: bytes,
    *,
    uvs_already_left_packed: bool = False,
) -> ObjectTextureVariants:
    """Repack a retextured half object without clipping geometry again.

    Fresh provider UV islands first retain original texel density; when they
    cannot fit, one maximum global uniform scale is applied to all islands and
    pixels. Verified left-packed UVs stay fixed while unused pixels are
    cleared; legacy or provider-modified UVs are repacked automatically.
    """

    if not isinstance(uvs_already_left_packed, bool):
        raise TypeError("The existing symmetric UV mode must be a boolean.")
    payload = bytes(canonical_2048_glb)
    if not payload:
        raise ValueError("The retextured canonical 2048 GLB is empty.")
    scene = _load_glb_scene(payload)
    _reject_auxiliary_material_textures(scene)
    source_textures = _collect_material_textures(scene)
    if not source_textures:
        raise ValueError(
            "Symmetric texture packing requires one embedded base-color atlas."
        )
    canonical_texture = _validate_shared_2048_texture(source_textures).copy()
    if uvs_already_left_packed and _scene_uvs_are_left_packed(scene):
        packed_texture = _mask_existing_left_texture(scene, canonical_texture)
    else:
        _normalize_scene_repeat_uvs(scene)
        packed_texture = _repack_retained_texture(scene, canonical_texture)
    return _build_half_texture_variants(
        scene,
        packed_texture,
        len(source_textures),
    )


# ### Scene collection helpers ###
def _reserve_unique_proxy_name(
    preferred_name: str,
    occupied_names: set[str],
) -> str:
    candidate = str(preferred_name)
    suffix = 2
    while candidate in occupied_names:
        candidate = f"{preferred_name}-{suffix}"
        suffix += 1
    occupied_names.add(candidate)
    return candidate


def _collect_mesh_instances(scene: trimesh.Scene) -> list[_MeshInstance]:
    node_names = sorted(scene.graph.nodes_geometry, key=str)
    referenced_geometry_names: set[str] = set()
    instances: list[_MeshInstance] = []
    for raw_node_name in node_names:
        transform, raw_geometry_name = scene.graph.get(raw_node_name)
        geometry_name = str(raw_geometry_name)
        if geometry_name in referenced_geometry_names:
            raise ValueError(
                "Symmetric division does not support one geometry shared by "
                "multiple scene nodes."
            )
        geometry = scene.geometry.get(raw_geometry_name)
        if not isinstance(geometry, trimesh.Trimesh):
            continue
        if not len(geometry.vertices) or not len(geometry.faces):
            continue
        referenced_geometry_names.add(geometry_name)
        node_transform = _validate_node_transform(transform)
        local_vertices = _validate_mesh_geometry(geometry)
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
                node_name=str(raw_node_name),
                geometry_name=geometry_name,
                mesh=geometry.copy(),
                transform=node_transform,
                world_vertices=np.asarray(z_up_world_vertices, dtype=float),
            )
        )
    if not instances:
        raise ValueError("The canonical 2048 GLB contains no triangle mesh.")
    orphaned = {
        str(name)
        for name in scene.geometry
        if str(name) not in referenced_geometry_names
    }
    if orphaned:
        raise ValueError(
            "Symmetric division does not support unreferenced scene geometry."
        )
    return instances


def _count_scene_triangles(scene: trimesh.Scene) -> int:
    return sum(
        len(geometry.faces)
        for geometry in scene.geometry.values()
        if isinstance(geometry, trimesh.Trimesh)
    )


def _validate_node_transform(raw_transform: object) -> np.ndarray:
    transform = np.asarray(raw_transform, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("The canonical 2048 GLB has an invalid node transform.")
    if abs(float(np.linalg.det(transform[:3, :3]))) <= _NORMAL_EPSILON:
        raise ValueError(
            "Symmetric division requires invertible mesh node transforms."
        )
    return transform.copy()


def _validate_mesh_geometry(mesh: trimesh.Trimesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,):
        raise ValueError("Symmetric division requires three-dimensional vertices.")
    if faces.ndim != 2 or faces.shape[1:] != (3,):
        raise ValueError("Symmetric division requires triangle faces.")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("The canonical 2048 GLB has invalid vertex coordinates.")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("The canonical 2048 GLB has invalid triangle indices.")
    return vertices


def _get_world_midpoint(instances: list[_MeshInstance], axis: int) -> float:
    coordinates = np.concatenate(
        [instance.world_vertices[:, axis] for instance in instances]
    )
    if not len(coordinates) or not np.all(np.isfinite(coordinates)):
        raise ValueError("The canonical 2048 GLB has invalid world bounds.")
    coordinate_minimum = float(np.min(coordinates))
    coordinate_maximum = float(np.max(coordinates))
    if coordinate_maximum - coordinate_minimum <= _NORMAL_EPSILON:
        raise ValueError(
            "The selected symmetric-division axis has no measurable extent."
        )
    return (coordinate_minimum + coordinate_maximum) * 0.5


# ### Geometry clipping helpers ###
def _build_clipped_scene(
    instances: list[_MeshInstance],
    *,
    axis: int,
    plane_coordinate: float,
    kept_side: str,
    metadata: dict[str, object] | None,
) -> trimesh.Scene:
    output_scene = trimesh.Scene()
    if isinstance(metadata, dict):
        output_scene.metadata.update(copy.deepcopy(metadata))
    direction = _get_kept_direction(kept_side)
    world_extent = max(
        float(np.ptp(instance.world_vertices[:, axis]))
        for instance in instances
    )
    epsilon = max(1.0, world_extent) * _CLIP_RELATIVE_EPSILON
    for instance in instances:
        clipped_mesh = _clip_instance_mesh(
            instance,
            axis=axis,
            plane_coordinate=plane_coordinate,
            direction=direction,
            epsilon=epsilon,
        )
        if clipped_mesh is None:
            continue
        output_scene.add_geometry(
            clipped_mesh,
            geom_name=instance.geometry_name,
            node_name=instance.node_name,
            transform=instance.transform,
        )
    if not output_scene.geometry:
        raise ValueError("Symmetric division produced an empty retained half.")
    return output_scene


def _clip_instance_mesh(
    instance: _MeshInstance,
    *,
    axis: int,
    plane_coordinate: float,
    direction: float,
    epsilon: float,
) -> trimesh.Trimesh | None:
    source_mesh = instance.mesh
    source_vertices = np.asarray(source_mesh.vertices, dtype=float)
    source_faces = np.asarray(source_mesh.faces, dtype=np.int64)
    source_normals = _get_vertex_normals(source_mesh)
    source_uvs = _get_vertex_uvs(source_mesh, allow_repeat=True)
    source_vertex_colors = _get_vertex_colors(source_mesh)
    source_face_colors = _get_face_colors(source_mesh)
    source_face_materials = _get_face_materials(source_mesh)

    output_vertices: list[np.ndarray] = []
    output_normals: list[np.ndarray] = []
    output_uvs: list[np.ndarray] = []
    output_vertex_colors: list[np.ndarray] = []
    output_faces: list[tuple[int, int, int]] = []
    output_face_materials: list[int] = []
    output_face_colors: list[np.ndarray] = []
    output_index_by_source: dict[tuple[int, ...], int] = {}

    for face_index, face in enumerate(source_faces):
        polygon = [
            _build_clip_vertex(
                vertex_index=int(vertex_index),
                instance=instance,
                source_vertices=source_vertices,
                source_normals=source_normals,
                source_uvs=source_uvs,
                source_vertex_colors=source_vertex_colors,
                axis=axis,
                plane_coordinate=plane_coordinate,
                direction=direction,
            )
            for vertex_index in face
        ]
        polygon = _clip_polygon_to_half_space(polygon, epsilon)
        if len(polygon) < 3:
            continue
        for corner_index in range(1, len(polygon) - 1):
            triangle = (polygon[0], polygon[corner_index], polygon[corner_index + 1])
            if _is_degenerate_triangle(triangle, epsilon):
                continue
            triangle_indices: list[int] = []
            for vertex in triangle:
                output_index = output_index_by_source.get(
                    vertex.source_indices
                )
                if output_index is None:
                    output_index = len(output_vertices)
                    output_index_by_source[vertex.source_indices] = (
                        output_index
                    )
                    output_vertices.append(vertex.local_position)
                    output_normals.append(vertex.normal)
                    if vertex.uv is not None:
                        output_uvs.append(vertex.uv)
                    if vertex.vertex_color is not None:
                        output_vertex_colors.append(vertex.vertex_color)
                triangle_indices.append(output_index)
            output_faces.append(tuple(triangle_indices))
            if source_face_materials is not None:
                output_face_materials.append(
                    int(source_face_materials[face_index])
                )
            if source_face_colors is not None:
                output_face_colors.append(source_face_colors[face_index])

    if not output_faces:
        return None
    return _build_output_mesh(
        source_mesh,
        vertices=np.asarray(output_vertices, dtype=float),
        faces=np.asarray(output_faces, dtype=np.int64),
        normals=np.asarray(output_normals, dtype=float),
        uvs=(np.asarray(output_uvs, dtype=float) if source_uvs is not None else None),
        vertex_colors=(
            np.asarray(output_vertex_colors)
            if source_vertex_colors is not None
            else None
        ),
        face_colors=(
            np.asarray(output_face_colors)
            if source_face_colors is not None
            else None
        ),
        face_materials=(
            np.asarray(output_face_materials, dtype=np.int64)
            if source_face_materials is not None
            else None
        ),
    )


def _build_clip_vertex(
    *,
    vertex_index: int,
    instance: _MeshInstance,
    source_vertices: np.ndarray,
    source_normals: np.ndarray,
    source_uvs: np.ndarray | None,
    source_vertex_colors: np.ndarray | None,
    axis: int,
    plane_coordinate: float,
    direction: float,
) -> _ClipVertex:
    world_position = instance.world_vertices[vertex_index]
    return _ClipVertex(
        local_position=source_vertices[vertex_index].copy(),
        world_position=world_position.copy(),
        normal=source_normals[vertex_index].copy(),
        uv=(
            source_uvs[vertex_index].copy()
            if source_uvs is not None
            else None
        ),
        vertex_color=(
            source_vertex_colors[vertex_index].copy()
            if source_vertex_colors is not None
            else None
        ),
        distance=(
            float(world_position[axis]) - plane_coordinate
        )
        * direction,
        source_indices=(vertex_index,),
    )


def _clip_polygon_to_half_space(
    polygon: list[_ClipVertex],
    epsilon: float,
) -> list[_ClipVertex]:
    output: list[_ClipVertex] = []
    previous = polygon[-1]
    previous_inside = previous.distance >= -epsilon
    for current in polygon:
        current_inside = current.distance >= -epsilon
        if current_inside != previous_inside:
            output.append(_interpolate_clip_edge(previous, current))
        if current_inside:
            output.append(current)
        previous = current
        previous_inside = current_inside
    return _deduplicate_polygon(output, epsilon)


def _interpolate_clip_edge(
    first: _ClipVertex,
    second: _ClipVertex,
) -> _ClipVertex:
    denominator = first.distance - second.distance
    if abs(denominator) <= _NORMAL_EPSILON:
        fraction = 0.5
    else:
        fraction = float(np.clip(first.distance / denominator, 0.0, 1.0))
    if fraction <= _NORMAL_EPSILON:
        return first
    if fraction >= 1.0 - _NORMAL_EPSILON:
        return second
    normal = _normalize_vector(
        _interpolate(first.normal, second.normal, fraction),
        fallback=first.normal,
    )
    return _ClipVertex(
        local_position=_interpolate(
            first.local_position,
            second.local_position,
            fraction,
        ),
        world_position=_interpolate(
            first.world_position,
            second.world_position,
            fraction,
        ),
        normal=normal,
        uv=_interpolate_optional(first.uv, second.uv, fraction),
        vertex_color=_interpolate_optional(
            first.vertex_color,
            second.vertex_color,
            fraction,
        ),
        distance=0.0,
        source_indices=tuple(
            sorted(set(first.source_indices + second.source_indices))
        ),
    )


def _deduplicate_polygon(
    polygon: list[_ClipVertex],
    epsilon: float,
) -> list[_ClipVertex]:
    deduplicated: list[_ClipVertex] = []
    for vertex in polygon:
        if deduplicated and np.linalg.norm(
            vertex.world_position - deduplicated[-1].world_position
        ) <= epsilon:
            continue
        deduplicated.append(vertex)
    if len(deduplicated) > 1 and np.linalg.norm(
        deduplicated[0].world_position - deduplicated[-1].world_position
    ) <= epsilon:
        deduplicated.pop()
    return deduplicated


def _is_degenerate_triangle(
    triangle: tuple[_ClipVertex, _ClipVertex, _ClipVertex],
    epsilon: float,
) -> bool:
    first, second, third = (
        vertex.world_position for vertex in triangle
    )
    doubled_area = np.linalg.norm(np.cross(second - first, third - first))
    return bool(doubled_area <= epsilon * epsilon)


# ### Output mesh helpers ###
def _build_output_mesh(
    source_mesh: trimesh.Trimesh,
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    uvs: np.ndarray | None,
    vertex_colors: np.ndarray | None,
    face_colors: np.ndarray | None,
    face_materials: np.ndarray | None,
) -> trimesh.Trimesh:
    output = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=normals,
        process=False,
        metadata=copy.deepcopy(source_mesh.metadata),
    )
    if uvs is not None:
        output.visual = TextureVisuals(
            uv=np.asarray(uvs, dtype=float).copy(),
            material=copy.deepcopy(source_mesh.visual.material),
            face_materials=face_materials,
        )
    elif vertex_colors is not None or face_colors is not None:
        output.visual = ColorVisuals(
            mesh=output,
            vertex_colors=vertex_colors,
            face_colors=face_colors,
        )
    return output


def _get_vertex_normals(mesh: trimesh.Trimesh) -> np.ndarray:
    cached_normals = mesh._cache.cache.get("vertex_normals")
    if cached_normals is None:
        normals = _calculate_vertex_normals(mesh)
    else:
        normals = np.asarray(cached_normals, dtype=float)
    if normals.shape != (len(mesh.vertices), 3) or not np.all(np.isfinite(normals)):
        raise ValueError("The canonical 2048 GLB has invalid vertex normals.")
    return normals


def _calculate_vertex_normals(mesh: trimesh.Trimesh) -> np.ndarray:
    """Build area-weighted normals without Trimesh's optional SciPy path."""

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


def _get_vertex_uvs(
    mesh: trimesh.Trimesh,
    *,
    allow_repeat: bool = False,
) -> np.ndarray | None:
    raw_uvs = getattr(mesh.visual, "uv", None)
    material_textures = _material_textures(mesh)
    if raw_uvs is None:
        if material_textures:
            raise ValueError("A textured mesh is missing UV coordinates.")
        return None
    uvs = np.asarray(raw_uvs, dtype=float)
    if uvs.shape != (len(mesh.vertices), 2) or not np.all(np.isfinite(uvs)):
        raise ValueError("A textured mesh has invalid UV coordinates.")
    if allow_repeat:
        return uvs.copy()
    if (
        np.any(uvs < -_REPEAT_UV_TOLERANCE)
        or np.any(uvs > 1.0 + _REPEAT_UV_TOLERANCE)
    ):
        raise ValueError("Symmetric division requires UVs inside the atlas.")
    return np.clip(uvs, 0.0, 1.0)


def _material_textures(mesh: trimesh.Trimesh) -> list[np.ndarray]:
    scene = trimesh.Scene()
    scene.add_geometry(mesh.copy())
    return _collect_material_textures(scene)


def _get_vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray | None:
    if not isinstance(mesh.visual, ColorVisuals):
        return None
    if getattr(mesh.visual, "kind", None) != "vertex":
        return None
    colors = np.asarray(mesh.visual.vertex_colors)
    if colors.shape[0] != len(mesh.vertices):
        return None
    return colors


def _get_face_colors(mesh: trimesh.Trimesh) -> np.ndarray | None:
    if not isinstance(mesh.visual, ColorVisuals):
        return None
    if getattr(mesh.visual, "kind", None) != "face":
        return None
    colors = np.asarray(mesh.visual.face_colors)
    if colors.shape[0] != len(mesh.faces):
        return None
    return colors


def _get_face_materials(mesh: trimesh.Trimesh) -> np.ndarray | None:
    raw_face_materials = getattr(mesh.visual, "face_materials", None)
    if raw_face_materials is None:
        return None
    face_materials = np.asarray(raw_face_materials, dtype=np.int64)
    if face_materials.shape != (len(mesh.faces),):
        raise ValueError("A textured mesh has invalid face material indices.")
    return face_materials


# ### Material compatibility helpers ###
def _reject_auxiliary_material_textures(scene: trimesh.Scene) -> None:
    """Reject maps which cannot share a base-color-only atlas transform."""

    auxiliary_names = (
        "normalTexture",
        "emissiveTexture",
        "occlusionTexture",
        "metallicRoughnessTexture",
    )
    for geometry in scene.geometry.values():
        material = getattr(getattr(geometry, "visual", None), "material", None)
        for nested_material in _iter_material_tree(material):
            if any(
                getattr(nested_material, name, None) is not None
                for name in auxiliary_names
            ):
                raise ValueError(
                    "Symmetric UV packing does not support auxiliary material "
                    "textures. Remove normal, emissive, occlusion and "
                    "metallic-roughness maps before dividing the object."
                )


def _iter_material_tree(material: object) -> list[object]:
    if material is None:
        return []
    nested = getattr(material, "materials", None)
    if isinstance(nested, list | tuple):
        flattened: list[object] = []
        for child in nested:
            flattened.extend(_iter_material_tree(child))
        return flattened
    return [material]


# ### Repeated UV normalization ###
def _normalize_scene_repeat_uvs(scene: trimesh.Scene) -> None:
    """Split repeated UV triangles into unit tiles before atlas packing.

    glTF samplers use REPEAT by default. Subtracting a single integer tile is
    exact for a triangle contained by one tile; triangles crossing a tile seam
    must first be clipped so interpolation never crosses the wrapped jump.
    """

    output_face_count = 0
    clipping_work = 0
    for geometry_name, geometry in sorted(
        list(scene.geometry.items()),
        key=lambda item: str(item[0]),
    ):
        if not isinstance(geometry, trimesh.Trimesh):
            continue
        if not _material_textures(geometry):
            continue
        uvs = _get_vertex_uvs(geometry, allow_repeat=True)
        if uvs is None:
            raise ValueError("A textured mesh is missing UV coordinates.")
        if _uvs_fit_unit_atlas(uvs):
            continue
        remaining_faces = _MAX_REPEAT_OUTPUT_FACES - output_face_count
        remaining_work = _MAX_REPEAT_CLIP_WORK - clipping_work
        normalized, geometry_work = _normalize_mesh_repeat_uvs(
            geometry,
            uvs,
            maximum_output_faces=remaining_faces,
            maximum_clipping_work=remaining_work,
        )
        scene.geometry[geometry_name] = normalized
        output_face_count += len(normalized.faces)
        clipping_work += geometry_work


def _uvs_fit_unit_atlas(uvs: np.ndarray) -> bool:
    return bool(
        np.all(uvs >= -_REPEAT_UV_TOLERANCE)
        and np.all(uvs <= 1.0 + _REPEAT_UV_TOLERANCE)
    )


def _normalize_mesh_repeat_uvs(
    mesh: trimesh.Trimesh,
    source_uvs: np.ndarray,
    *,
    maximum_output_faces: int,
    maximum_clipping_work: int,
) -> tuple[trimesh.Trimesh, int]:
    """Wrap one textured mesh while retaining repeat-correct interpolation."""

    working_uvs = _snap_repeat_uv_seams(source_uvs)
    if maximum_output_faces <= 0:
        raise ValueError(
            "Repeated UV normalization would create too many retained faces."
        )
    source_vertices = np.asarray(mesh.vertices, dtype=float)
    source_faces = np.asarray(mesh.faces, dtype=np.int64)
    source_normals = _get_vertex_normals(mesh)
    source_vertex_colors = _get_vertex_colors(mesh)
    source_face_colors = _get_face_colors(mesh)
    source_face_materials = _get_face_materials(mesh)

    output_vertices: list[np.ndarray] = []
    output_normals: list[np.ndarray] = []
    output_uvs: list[np.ndarray] = []
    output_vertex_colors: list[np.ndarray] = []
    output_faces: list[tuple[int, int, int]] = []
    output_face_colors: list[np.ndarray] = []
    output_face_materials: list[int] = []
    output_index_by_key: dict[tuple[object, ...], int] = {}
    clipping_work = 0

    for face_index, face in enumerate(source_faces):
        polygon = [
            _RepeatClipVertex(
                local_position=source_vertices[int(vertex_index)].copy(),
                normal=source_normals[int(vertex_index)].copy(),
                uv=working_uvs[int(vertex_index)].copy(),
                vertex_color=(
                    source_vertex_colors[int(vertex_index)].copy()
                    if source_vertex_colors is not None
                    else None
                ),
                source_indices=(int(vertex_index),),
            )
            for vertex_index in face
        ]
        u_tiles = _repeat_tile_indices(working_uvs[face, 0])
        v_tiles = _repeat_tile_indices(working_uvs[face, 1])
        if (
            len(u_tiles) > _MAX_REPEAT_TILE_SPAN
            or len(v_tiles) > _MAX_REPEAT_TILE_SPAN
        ):
            raise ValueError(
                "Repeated UV normalization exceeds the bounded tile "
                "processing limit."
            )
        face_tile_count = len(u_tiles) * len(v_tiles)
        if face_tile_count > _MAX_REPEAT_TILES_PER_FACE:
            raise ValueError(
                "Repeated UV normalization exceeds the bounded tile "
                "processing limit."
            )
        if face_tile_count > 1:
            clipping_work += face_tile_count
        if clipping_work > maximum_clipping_work:
            raise ValueError(
                "Repeated UV normalization exceeds the bounded tile "
                "processing limit."
            )

        for tile_u in u_tiles:
            for tile_v in v_tiles:
                tile_polygon = polygon
                if face_tile_count != 1:
                    tile_polygon = _clip_repeat_polygon_to_tile(
                        polygon,
                        tile_u,
                        tile_v,
                    )
                if len(tile_polygon) < 3:
                    continue
                for corner_index in range(1, len(tile_polygon) - 1):
                    triangle = (
                        tile_polygon[0],
                        tile_polygon[corner_index],
                        tile_polygon[corner_index + 1],
                    )
                    if _is_repeat_triangle_degenerate(triangle):
                        continue
                    triangle_indices = tuple(
                        _append_repeat_output_vertex(
                            vertex,
                            tile_u=tile_u,
                            tile_v=tile_v,
                            output_vertices=output_vertices,
                            output_normals=output_normals,
                            output_uvs=output_uvs,
                            output_vertex_colors=output_vertex_colors,
                            output_index_by_key=output_index_by_key,
                        )
                        for vertex in triangle
                    )
                    if len(set(triangle_indices)) < 3:
                        continue
                    output_faces.append(triangle_indices)
                    if len(output_faces) > maximum_output_faces:
                        raise ValueError(
                            "Repeated UV normalization would create too many "
                            "retained faces."
                        )
                    if source_face_materials is not None:
                        output_face_materials.append(
                            int(source_face_materials[face_index])
                        )
                    if source_face_colors is not None:
                        output_face_colors.append(
                            source_face_colors[face_index].copy()
                        )

    if not output_faces:
        raise ValueError(
            "Repeated UV normalization removed every textured triangle."
        )
    return (
        _build_output_mesh(
            mesh,
            vertices=np.asarray(output_vertices, dtype=float),
            faces=np.asarray(output_faces, dtype=np.int64),
            normals=np.asarray(output_normals, dtype=float),
            uvs=np.asarray(output_uvs, dtype=float),
            vertex_colors=(
                np.asarray(output_vertex_colors)
                if source_vertex_colors is not None
                else None
            ),
            face_colors=(
                np.asarray(output_face_colors)
                if source_face_colors is not None
                else None
            ),
            face_materials=(
                np.asarray(output_face_materials, dtype=np.int64)
                if source_face_materials is not None
                else None
            ),
        ),
        clipping_work,
    )


def _repeat_tile_indices(coordinates: np.ndarray) -> range:
    """Return positive-area tile owners; an exact upper seam stays below."""

    minimum = float(np.min(coordinates))
    maximum = float(np.max(coordinates))
    if max(abs(minimum), abs(maximum)) > _MAX_REPEAT_TILE_MAGNITUDE:
        raise ValueError(
            "Repeated UV normalization exceeds the bounded tile processing "
            "limit."
        )
    first = math.floor(minimum)
    last = math.ceil(maximum) - 1
    if last < first:
        last = first
    return range(first, last + 1)


def _snap_repeat_uv_seams(uvs: np.ndarray) -> np.ndarray:
    """Remove subpixel export noise around every integer repeat seam."""

    values = np.asarray(uvs, dtype=float).copy()
    nearest_integers = np.rint(values)
    near_seam = np.abs(values - nearest_integers) <= _REPEAT_SEAM_TOLERANCE
    values[near_seam] = nearest_integers[near_seam]
    return values


def _clip_repeat_polygon_to_tile(
    polygon: list[_RepeatClipVertex],
    tile_u: int,
    tile_v: int,
) -> list[_RepeatClipVertex]:
    clipped = polygon
    for axis, boundary, keep_greater in (
        (0, float(tile_u), True),
        (0, float(tile_u + 1), False),
        (1, float(tile_v), True),
        (1, float(tile_v + 1), False),
    ):
        clipped = _clip_repeat_polygon_to_boundary(
            clipped,
            axis=axis,
            boundary=boundary,
            keep_greater=keep_greater,
        )
        if len(clipped) < 3:
            return []
    return clipped


def _clip_repeat_polygon_to_boundary(
    polygon: list[_RepeatClipVertex],
    *,
    axis: int,
    boundary: float,
    keep_greater: bool,
) -> list[_RepeatClipVertex]:
    if not polygon:
        return []
    output: list[_RepeatClipVertex] = []
    previous = polygon[-1]
    previous_inside = _repeat_boundary_contains(
        float(previous.uv[axis]),
        boundary,
        keep_greater,
    )
    for current in polygon:
        current_inside = _repeat_boundary_contains(
            float(current.uv[axis]),
            boundary,
            keep_greater,
        )
        if current_inside != previous_inside:
            output.append(
                _interpolate_repeat_boundary(
                    previous,
                    current,
                    axis,
                    boundary,
                )
            )
        if current_inside:
            output.append(current)
        previous = current
        previous_inside = current_inside
    return _deduplicate_repeat_polygon(output)


def _repeat_boundary_contains(
    coordinate: float,
    boundary: float,
    keep_greater: bool,
) -> bool:
    return coordinate >= boundary if keep_greater else coordinate <= boundary


def _interpolate_repeat_boundary(
    first: _RepeatClipVertex,
    second: _RepeatClipVertex,
    axis: int,
    boundary: float,
) -> _RepeatClipVertex:
    denominator = float(second.uv[axis] - first.uv[axis])
    if abs(denominator) <= _NORMAL_EPSILON:
        fraction = 0.5
    else:
        fraction = float(
            np.clip((boundary - first.uv[axis]) / denominator, 0.0, 1.0)
        )
    if fraction <= _REPEAT_CLIP_EPSILON:
        return first
    if fraction >= 1.0 - _REPEAT_CLIP_EPSILON:
        return second
    uv = _interpolate(first.uv, second.uv, fraction)
    uv[axis] = boundary
    return _RepeatClipVertex(
        local_position=_interpolate(
            first.local_position,
            second.local_position,
            fraction,
        ),
        normal=_interpolate(first.normal, second.normal, fraction),
        uv=uv,
        vertex_color=_interpolate_optional(
            first.vertex_color,
            second.vertex_color,
            fraction,
        ),
        source_indices=tuple(
            sorted(set(first.source_indices + second.source_indices))
        ),
    )


def _deduplicate_repeat_polygon(
    polygon: list[_RepeatClipVertex],
) -> list[_RepeatClipVertex]:
    deduplicated: list[_RepeatClipVertex] = []
    for vertex in polygon:
        if deduplicated and _repeat_vertices_coincide(
            vertex,
            deduplicated[-1],
        ):
            continue
        deduplicated.append(vertex)
    if (
        len(deduplicated) > 1
        and _repeat_vertices_coincide(deduplicated[0], deduplicated[-1])
    ):
        deduplicated.pop()
    return deduplicated


def _repeat_vertices_coincide(
    first: _RepeatClipVertex,
    second: _RepeatClipVertex,
) -> bool:
    return bool(
        np.allclose(
            first.uv,
            second.uv,
            rtol=0.0,
            atol=_REPEAT_CLIP_EPSILON,
        )
        and np.allclose(
            first.local_position,
            second.local_position,
            rtol=_REPEAT_CLIP_EPSILON,
            atol=_REPEAT_CLIP_EPSILON,
        )
    )


def _is_repeat_triangle_degenerate(
    triangle: tuple[_RepeatClipVertex, _RepeatClipVertex, _RepeatClipVertex],
) -> bool:
    first, second, third = (
        vertex.local_position for vertex in triangle
    )
    doubled_area = float(
        np.linalg.norm(np.cross(second - first, third - first))
    )
    return doubled_area == 0.0


def _append_repeat_output_vertex(
    vertex: _RepeatClipVertex,
    *,
    tile_u: int,
    tile_v: int,
    output_vertices: list[np.ndarray],
    output_normals: list[np.ndarray],
    output_uvs: list[np.ndarray],
    output_vertex_colors: list[np.ndarray],
    output_index_by_key: dict[tuple[object, ...], int],
) -> int:
    key = (
        tile_u,
        tile_v,
        vertex.source_indices,
        round(float(vertex.uv[0]), 12),
        round(float(vertex.uv[1]), 12),
        round(float(vertex.local_position[0]), 12),
        round(float(vertex.local_position[1]), 12),
        round(float(vertex.local_position[2]), 12),
    )
    output_index = output_index_by_key.get(key)
    if output_index is not None:
        return output_index
    wrapped_uv = vertex.uv - np.asarray((tile_u, tile_v), dtype=float)
    wrapped_uv = np.clip(wrapped_uv, 0.0, 1.0)
    output_index = len(output_vertices)
    output_index_by_key[key] = output_index
    output_vertices.append(vertex.local_position.copy())
    output_normals.append(
        _normalize_vector(
            vertex.normal,
            fallback=np.asarray((0.0, 0.0, 1.0), dtype=float),
        )
    )
    output_uvs.append(wrapped_uv)
    if vertex.vertex_color is not None:
        output_vertex_colors.append(vertex.vertex_color.copy())
    return output_index


# ### UV island preparation ###
def _prepare_uv_islands(
    scene: trimesh.Scene,
    source_rgba: np.ndarray,
) -> list[_UvIsland]:
    """Split movable charts at UV seams and rasterize retained face coverage."""

    islands: list[_UvIsland] = []
    allocated_mask_pixels = 0
    deleted_geometry_names: list[object] = []
    for geometry_name, geometry in sorted(
        list(scene.geometry.items()),
        key=lambda item: str(item[0]),
    ):
        if not isinstance(geometry, trimesh.Trimesh):
            continue
        if not _material_textures(geometry):
            continue
        geometry = _drop_zero_area_faces(geometry)
        if geometry is None:
            deleted_geometry_names.append(geometry_name)
            continue
        uvs = _get_vertex_uvs(geometry)
        if uvs is None:
            raise ValueError("A textured mesh is missing UV coordinates.")
        faces = np.asarray(geometry.faces, dtype=np.int64)
        repair_kind_by_face = {
            face_index: repair_kind
            for face_index, face in enumerate(faces)
            if (
                repair_kind := _classify_collapsed_uv_face(uvs[face])
            ) is not None
        }
        regular_face_indices = np.asarray(
            [
                face_index
                for face_index in range(len(faces))
                if face_index not in repair_kind_by_face
            ],
            dtype=np.int64,
        )
        components: list[np.ndarray] = []
        if len(regular_face_indices):
            components.extend(
                regular_face_indices[component]
                for component in _collect_face_edge_components(
                    faces[regular_face_indices]
                )
            )
        components.extend(
            np.asarray((face_index,), dtype=np.int64)
            for face_index in repair_kind_by_face
        )
        components.sort(key=lambda component: int(component[0]))
        if len(islands) + len(components) > _MAX_UV_ISLAND_COUNT:
            raise ValueError(
                "The retained model has too many independent UV islands for "
                "bounded symmetric packing."
            )
        rebuilt, component_vertices = _split_mesh_vertices_by_components(
            geometry,
            components,
        )
        scene.geometry[geometry_name] = rebuilt
        rebuilt_uvs = _get_vertex_uvs(rebuilt)
        if rebuilt_uvs is None:
            raise ValueError("A textured mesh is missing UV coordinates.")
        rebuilt_faces = np.asarray(rebuilt.faces, dtype=np.int64)
        for face_indices, vertex_indices in zip(
            components,
            component_vertices,
            strict=True,
        ):
            repair_kind = repair_kind_by_face.get(int(face_indices[0]))
            if repair_kind is None:
                estimated_pixels = _estimate_uv_mask_pixels(
                    rebuilt_uvs,
                    rebuilt_faces[face_indices],
                )
                bounds, mask = _rasterize_uv_triangles(
                    rebuilt_uvs,
                    rebuilt_faces[face_indices],
                )
                bounds, mask = _align_uv_mask_to_pack_lattice(bounds, mask)
                estimated_pixels = max(estimated_pixels, int(mask.size))
                source_pixel_coordinates = None
                repaired_rgba = None
            else:
                repaired_vertices = rebuilt_faces[int(face_indices[0])]
                (
                    bounds,
                    mask,
                    repaired_rgba,
                    source_pixel_coordinates,
                ) = _build_repaired_uv_face_scratch(
                    rebuilt_uvs[repaired_vertices],
                    source_rgba,
                    repair_kind,
                )
                vertex_indices = np.asarray(
                    repaired_vertices,
                    dtype=np.int64,
                )
                estimated_pixels = int(mask.size)
            if (
                allocated_mask_pixels + estimated_pixels
                > _MAX_UV_MASK_PIXELS
            ):
                raise ValueError(
                    "The retained UV island masks exceed the bounded packing "
                    "memory limit."
                )
            allocated_mask_pixels += int(mask.size)
            islands.append(
                _UvIsland(
                    island_id=len(islands),
                    geometry_name=geometry_name,
                    face_indices=face_indices,
                    vertex_indices=vertex_indices,
                    source_bounds=bounds,
                    source_mask=mask,
                    exact_repeat_seams=_get_exact_repeat_seam_edges(
                        rebuilt_uvs[vertex_indices]
                    ),
                    source_pixel_coordinates=source_pixel_coordinates,
                    source_rgba=repaired_rgba,
                )
            )
    for geometry_name in deleted_geometry_names:
        scene.delete_geometry(geometry_name)
    if not islands:
        raise ValueError("The retextured GLB contains no textured mesh UVs.")
    return islands


def _get_exact_repeat_seam_edges(uvs: np.ndarray) -> _RepeatSeamEdges:
    """Identify exact GL_REPEAT seams without classifying nearby UVs."""

    values = np.asarray(uvs, dtype=float)
    return _RepeatSeamEdges(
        u_zero=bool(
            np.any(
                np.isclose(
                    values[:, 0],
                    0.0,
                    rtol=0.0,
                    atol=_REPEAT_SEAM_TOLERANCE,
                )
            )
        ),
        u_one=bool(
            np.any(
                np.isclose(
                    values[:, 0],
                    1.0,
                    rtol=0.0,
                    atol=_REPEAT_SEAM_TOLERANCE,
                )
            )
        ),
        v_zero=bool(
            np.any(
                np.isclose(
                    values[:, 1],
                    0.0,
                    rtol=0.0,
                    atol=_REPEAT_SEAM_TOLERANCE,
                )
            )
        ),
        v_one=bool(
            np.any(
                np.isclose(
                    values[:, 1],
                    1.0,
                    rtol=0.0,
                    atol=_REPEAT_SEAM_TOLERANCE,
                )
            )
        ),
    )


def _drop_zero_area_faces(
    mesh: trimesh.Trimesh,
) -> trimesh.Trimesh | None:
    """Remove only triangles whose 3D area is numerically zero."""

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    first_edges = triangles[:, 1] - triangles[:, 0]
    second_edges = triangles[:, 2] - triangles[:, 0]
    doubled_areas = np.linalg.norm(
        np.cross(first_edges, second_edges),
        axis=1,
    )
    retained = doubled_areas > 0.0
    if np.all(retained):
        return mesh
    if not np.any(retained):
        return None
    normals = _get_vertex_normals(mesh)
    uvs = _get_vertex_uvs(mesh)
    face_materials = _get_face_materials(mesh)
    return _build_output_mesh(
        mesh,
        vertices=vertices.copy(),
        faces=faces[retained].copy(),
        normals=normals.copy(),
        uvs=uvs.copy() if uvs is not None else None,
        vertex_colors=None,
        face_colors=None,
        face_materials=(
            face_materials[retained].copy()
            if face_materials is not None
            else None
        ),
    )


def _classify_collapsed_uv_face(
    triangle_uvs: np.ndarray,
) -> Literal["point", "line"] | None:
    """Distinguish rank-deficient UVs from valid subpixel triangles."""

    coordinates = np.asarray(triangle_uvs, dtype=np.float64)
    first = coordinates[1] - coordinates[0]
    second = coordinates[2] - coordinates[0]
    third = coordinates[2] - coordinates[1]
    maximum_edge_squared = max(
        float(np.dot(first, first)),
        float(np.dot(second, second)),
        float(np.dot(third, third)),
    )
    if maximum_edge_squared == 0.0:
        return "point"
    doubled_area = abs(float(first[0] * second[1] - first[1] * second[0]))
    if doubled_area == 0.0:
        return "line"
    return None


def _build_repaired_uv_face_scratch(
    original_uvs: np.ndarray,
    source_rgba: np.ndarray,
    repair_kind: Literal["point", "line"],
) -> tuple[_PixelRectangle, np.ndarray, np.ndarray, np.ndarray]:
    """Bake one collapsed face into a private nonzero scratch chart."""

    source_pixels = _uvs_to_source_pixels(original_uvs)
    if repair_kind == "point":
        margin = _REPAIRED_CHART_MARGIN
        extent = _REPAIRED_POINT_CHART_EXTENT
        chart_points = np.asarray(
            (
                (margin, margin),
                (margin, margin + extent),
                (margin + extent, margin),
            ),
            dtype=np.float64,
        )
    else:
        chart_points = _build_line_repair_chart_points(source_pixels)
    width = _ceil_to_pack_lattice(
        max(1, int(math.floor(float(np.max(chart_points[:, 0])))) + 2)
    )
    height = _ceil_to_pack_lattice(
        max(1, int(math.floor(float(np.max(chart_points[:, 1])))) + 2)
    )
    mask = _rasterize_pixel_triangle(chart_points, width, height)
    if not np.any(mask):
        raise RuntimeError("A repaired UV chart could not be rasterized.")
    scratch = np.empty((height, width, 4), dtype=np.uint8)
    scratch[:] = _OPAQUE_BLACK
    rows, columns = np.nonzero(mask)
    if repair_kind == "point":
        representative_uv = np.mean(original_uvs, axis=0)
        color = _sample_repeat_bilinear_rgba(source_rgba, representative_uv)
        scratch[rows, columns] = color
    else:
        weights = _triangle_barycentric_weights(
            np.column_stack((columns, rows)),
            chart_points,
        )
        sampled_uvs = weights @ np.asarray(original_uvs, dtype=np.float64)
        scratch[rows, columns] = _sample_repeat_bilinear_rgba(
            source_rgba,
            sampled_uvs,
        )
    return (
        _PixelRectangle(0, 0, width, height),
        mask,
        scratch,
        chart_points,
    )


def _uvs_to_source_pixels(uvs: np.ndarray) -> np.ndarray:
    resolution = float(TEXTURE_RESOLUTION_2048)
    values = np.asarray(uvs, dtype=np.float64)
    return np.column_stack(
        (
            values[:, 0] * resolution - 0.5,
            (1.0 - values[:, 1]) * resolution - 0.5,
        )
    )


def _build_line_repair_chart_points(source_pixels: np.ndarray) -> np.ndarray:
    farthest_pair = (0, 1)
    farthest_squared = -1.0
    for first in range(3):
        for second in range(first + 1, 3):
            difference = source_pixels[second] - source_pixels[first]
            squared = float(np.dot(difference, difference))
            if squared > farthest_squared:
                farthest_squared = squared
                farthest_pair = (first, second)
    if farthest_squared <= 0.0:
        raise RuntimeError("A line UV repair requires a measurable span.")
    first, second = farthest_pair
    direction = source_pixels[second] - source_pixels[first]
    direction /= math.sqrt(farthest_squared)
    scalars = (source_pixels - source_pixels[first]) @ direction
    scalars -= float(np.min(scalars))
    margin = _REPAIRED_CHART_MARGIN
    chart_points = np.column_stack(
        (
            margin + scalars,
            np.full(3, margin, dtype=np.float64),
        )
    )
    offset_index = max(
        range(3),
        key=lambda index: (
            abs(
                float(
                    scalars[(index + 1) % 3]
                    - scalars[(index + 2) % 3]
                )
            ),
            -index,
        ),
    )
    chart_points[offset_index, 1] += _REPAIRED_LINE_CHART_HEIGHT
    doubled_area = float(
        np.linalg.det(
            np.vstack(
                (
                    chart_points[1] - chart_points[0],
                    chart_points[2] - chart_points[0],
                )
            )
        )
    )
    if doubled_area > 0.0:
        chart_points[:, 1] = (
            2.0 * margin
            + _REPAIRED_LINE_CHART_HEIGHT
            - chart_points[:, 1]
        )
    return chart_points


def _rasterize_pixel_triangle(
    triangle: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    fixed_scale = 1 << _UV_RASTER_FIXED_POINT_BITS
    fixed_points = np.rint(
        np.asarray(triangle, dtype=np.float64) * fixed_scale
    ).astype(np.int32)
    cv2.fillConvexPoly(
        mask,
        fixed_points,
        255,
        lineType=cv2.LINE_8,
        shift=_UV_RASTER_FIXED_POINT_BITS,
    )
    return np.asarray(mask != 0, dtype=bool)


def _triangle_barycentric_weights(
    points: np.ndarray,
    triangle: np.ndarray,
) -> np.ndarray:
    first, second, third = np.asarray(triangle, dtype=np.float64)
    denominator = (
        (second[1] - third[1]) * (first[0] - third[0])
        + (third[0] - second[0]) * (first[1] - third[1])
    )
    if abs(float(denominator)) <= _NORMAL_EPSILON:
        raise RuntimeError("A repaired UV chart is degenerate.")
    values = np.asarray(points, dtype=np.float64)
    first_weights = (
        (second[1] - third[1]) * (values[:, 0] - third[0])
        + (third[0] - second[0]) * (values[:, 1] - third[1])
    ) / denominator
    second_weights = (
        (third[1] - first[1]) * (values[:, 0] - third[0])
        + (first[0] - third[0]) * (values[:, 1] - third[1])
    ) / denominator
    weights = np.column_stack(
        (first_weights, second_weights, 1.0 - first_weights - second_weights)
    )
    weights = np.maximum(weights, 0.0)
    totals = np.sum(weights, axis=1)
    return weights / totals[:, np.newaxis]


def _sample_repeat_bilinear_rgba(
    source_rgba: np.ndarray,
    uvs: np.ndarray,
) -> np.ndarray:
    """Sample straight RGBA with the glTF default REPEAT wrapping mode."""

    sampled = _sample_repeat_bilinear_values(source_rgba, uvs)
    return np.asarray(
        np.clip(np.rint(sampled), 0.0, 255.0),
        dtype=np.uint8,
    )


def _sample_repeat_bilinear_premultiplied_rgba(
    source_rgba: np.ndarray,
    uvs: np.ndarray,
) -> np.ndarray:
    """Sample repeated RGBA without exposing color hidden by zero alpha."""

    values = np.asarray(uvs, dtype=np.float64)
    single = values.ndim == 1
    values = np.atleast_2d(values)
    height, width = source_rgba.shape[:2]
    wrapped = values - np.floor(values)
    columns = wrapped[:, 0] * width - 0.5
    rows = (1.0 - wrapped[:, 1]) * height - 0.5
    column0_raw = np.floor(columns).astype(np.int64)
    row0_raw = np.floor(rows).astype(np.int64)
    column_fraction = columns - column0_raw
    row_fraction = rows - row0_raw
    column0 = np.mod(column0_raw, width)
    column1 = np.mod(column0_raw + 1, width)
    row0 = np.mod(row0_raw, height)
    row1 = np.mod(row0_raw + 1, height)

    neighbors = tuple(
        _premultiply_rgba_samples(source_rgba[sample_rows, sample_columns])
        for sample_rows, sample_columns in (
            (row0, column0),
            (row0, column1),
            (row1, column0),
            (row1, column1),
        )
    )
    top_left, top_right, bottom_left, bottom_right = neighbors
    top = (
        top_left * (1.0 - column_fraction[:, np.newaxis])
        + top_right * column_fraction[:, np.newaxis]
    )
    bottom = (
        bottom_left * (1.0 - column_fraction[:, np.newaxis])
        + bottom_right * column_fraction[:, np.newaxis]
    )
    sampled = (
        top * (1.0 - row_fraction[:, np.newaxis])
        + bottom * row_fraction[:, np.newaxis]
    )
    visible = sampled[:, 3] > 1e-8
    result = np.zeros(sampled.shape, dtype=np.float64)
    result[:, 3] = sampled[:, 3]
    result[visible, :3] = (
        sampled[visible, :3]
        * 255.0
        / sampled[visible, 3, np.newaxis]
    )
    encoded = np.asarray(
        np.clip(np.rint(result), 0.0, 255.0),
        dtype=np.uint8,
    )
    return encoded[0] if single else encoded


def _premultiply_rgba_samples(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64).copy()
    values[:, :3] *= values[:, 3, np.newaxis] / 255.0
    return values


def _sample_repeat_bilinear_values(
    source_values: np.ndarray,
    uvs: np.ndarray,
) -> np.ndarray:
    values = np.asarray(uvs, dtype=np.float64)
    single = values.ndim == 1
    values = np.atleast_2d(values)
    height, width = source_values.shape[:2]
    wrapped = values - np.floor(values)
    columns = wrapped[:, 0] * width - 0.5
    rows = (1.0 - wrapped[:, 1]) * height - 0.5
    column0_raw = np.floor(columns).astype(np.int64)
    row0_raw = np.floor(rows).astype(np.int64)
    column_fraction = columns - column0_raw
    row_fraction = rows - row0_raw
    column0 = np.mod(column0_raw, width)
    column1 = np.mod(column0_raw + 1, width)
    row0 = np.mod(row0_raw, height)
    row1 = np.mod(row0_raw + 1, height)
    top_left = np.asarray(source_values[row0, column0], dtype=np.float64)
    top_right = np.asarray(source_values[row0, column1], dtype=np.float64)
    bottom_left = np.asarray(source_values[row1, column0], dtype=np.float64)
    bottom_right = np.asarray(source_values[row1, column1], dtype=np.float64)
    top = (
        top_left * (1.0 - column_fraction[:, np.newaxis])
        + top_right * column_fraction[:, np.newaxis]
    )
    bottom = (
        bottom_left * (1.0 - column_fraction[:, np.newaxis])
        + bottom_right * column_fraction[:, np.newaxis]
    )
    sampled = (
        top * (1.0 - row_fraction[:, np.newaxis])
        + bottom * row_fraction[:, np.newaxis]
    )
    return sampled[0] if single else sampled


def _estimate_uv_mask_pixels(uvs: np.ndarray, faces: np.ndarray) -> int:
    resolution = TEXTURE_RESOLUTION_2048
    triangle_uvs = np.asarray(uvs[faces], dtype=float)
    pixel_x = triangle_uvs[:, :, 0] * resolution - 0.5
    pixel_y = (1.0 - triangle_uvs[:, :, 1]) * resolution - 0.5
    minimum_x = max(0, int(math.floor(float(np.min(pixel_x)))))
    minimum_y = max(0, int(math.floor(float(np.min(pixel_y)))))
    maximum_x = min(
        resolution,
        int(math.floor(float(np.max(pixel_x)))) + 1,
    )
    maximum_y = min(
        resolution,
        int(math.floor(float(np.max(pixel_y)))) + 1,
    )
    return max(1, maximum_x - minimum_x) * max(1, maximum_y - minimum_y)


def _collect_face_edge_components(faces: np.ndarray) -> list[np.ndarray]:
    face_count = len(faces)
    parents = list(range(face_count))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[max(first_root, second_root)] = min(first_root, second_root)

    owner_by_edge: dict[tuple[int, int], int] = {}
    for face_index, face in enumerate(faces):
        for first, second in (
            (int(face[0]), int(face[1])),
            (int(face[1]), int(face[2])),
            (int(face[2]), int(face[0])),
        ):
            edge = (min(first, second), max(first, second))
            prior = owner_by_edge.setdefault(edge, face_index)
            union(face_index, prior)
    members_by_root: dict[int, list[int]] = {}
    for face_index in range(face_count):
        members_by_root.setdefault(find(face_index), []).append(face_index)
    return [
        np.asarray(indices, dtype=np.int64)
        for _root, indices in sorted(
            members_by_root.items(),
            key=lambda item: item[1][0],
        )
    ]


def _split_mesh_vertices_by_components(
    mesh: trimesh.Trimesh,
    components: list[np.ndarray],
) -> tuple[trimesh.Trimesh, list[np.ndarray]]:
    """Give each movable island independent vertex and UV storage."""

    source_vertices = np.asarray(mesh.vertices, dtype=float)
    source_faces = np.asarray(mesh.faces, dtype=np.int64)
    source_normals = _get_vertex_normals(mesh)
    source_uvs = _get_vertex_uvs(mesh)
    if source_uvs is None:
        raise ValueError("A textured mesh is missing UV coordinates.")
    rebuilt_faces = np.empty_like(source_faces)
    rebuilt_vertices: list[np.ndarray] = []
    rebuilt_normals: list[np.ndarray] = []
    rebuilt_uvs: list[np.ndarray] = []
    component_vertices: list[np.ndarray] = []
    for face_indices in components:
        new_by_old: dict[int, int] = {}
        for face_index in face_indices:
            for corner_index, old_index_value in enumerate(
                source_faces[int(face_index)]
            ):
                old_index = int(old_index_value)
                new_index = new_by_old.get(old_index)
                if new_index is None:
                    new_index = len(rebuilt_vertices)
                    new_by_old[old_index] = new_index
                    rebuilt_vertices.append(source_vertices[old_index].copy())
                    rebuilt_normals.append(source_normals[old_index].copy())
                    rebuilt_uvs.append(source_uvs[old_index].copy())
                rebuilt_faces[int(face_index), corner_index] = new_index
        component_vertices.append(
            np.asarray(sorted(new_by_old.values()), dtype=np.int64)
        )
    face_materials = _get_face_materials(mesh)
    rebuilt = trimesh.Trimesh(
        vertices=np.asarray(rebuilt_vertices, dtype=float),
        faces=rebuilt_faces,
        vertex_normals=np.asarray(rebuilt_normals, dtype=float),
        process=False,
        metadata=copy.deepcopy(mesh.metadata),
    )
    rebuilt.visual = TextureVisuals(
        uv=np.asarray(rebuilt_uvs, dtype=float),
        material=copy.deepcopy(mesh.visual.material),
        face_materials=(
            face_materials.copy() if face_materials is not None else None
        ),
    )
    return rebuilt, component_vertices


def _rasterize_uv_triangles(
    uvs: np.ndarray,
    faces: np.ndarray,
) -> tuple[_PixelRectangle, np.ndarray]:
    resolution = TEXTURE_RESOLUTION_2048
    triangle_uvs = np.asarray(uvs[faces], dtype=float)
    pixel_points = np.empty_like(triangle_uvs)
    pixel_points[:, :, 0] = triangle_uvs[:, :, 0] * resolution - 0.5
    pixel_points[:, :, 1] = (
        (1.0 - triangle_uvs[:, :, 1]) * resolution - 0.5
    )
    minimum = np.floor(np.min(pixel_points, axis=(0, 1))).astype(int)
    maximum = np.floor(np.max(pixel_points, axis=(0, 1))).astype(int) + 1
    x0 = int(np.clip(minimum[0], 0, resolution - 1))
    y0 = int(np.clip(minimum[1], 0, resolution - 1))
    x1 = int(np.clip(maximum[0], x0 + 1, resolution))
    y1 = int(np.clip(maximum[1], y0 + 1, resolution))
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    fixed_scale = 1 << _UV_RASTER_FIXED_POINT_BITS
    origin = np.asarray((x0, y0), dtype=float)
    for triangle in pixel_points:
        fixed_points = np.rint(
            (triangle - origin) * fixed_scale
        ).astype(np.int32)
        cv2.fillConvexPoly(
            mask,
            fixed_points,
            255,
            lineType=cv2.LINE_8,
            shift=_UV_RASTER_FIXED_POINT_BITS,
        )
        first_edge = triangle[1] - triangle[0]
        second_edge = triangle[2] - triangle[0]
        third_edge = triangle[2] - triangle[1]
        longest_edge = max(
            float(np.linalg.norm(first_edge)),
            float(np.linalg.norm(second_edge)),
            float(np.linalg.norm(third_edge)),
        )
        doubled_area = abs(
            float(
                first_edge[0] * second_edge[1]
                - first_edge[1] * second_edge[0]
            )
        )
        clipped_triangle = np.clip(
            triangle,
            0.0,
            float(resolution - 1),
        )
        representatives = np.vstack(
            (
                clipped_triangle,
                np.clip(
                    np.mean(triangle, axis=0),
                    0.0,
                    float(resolution - 1),
                ),
            )
        )
        representative_pixels = np.rint(representatives).astype(np.int64)
        misses_vertex_coverage = any(
            mask[
                int(np.clip(row - y0, 0, mask.shape[0] - 1)),
                int(np.clip(column - x0, 0, mask.shape[1] - 1)),
            ]
            == 0
            for column, row in representative_pixels[:3]
        )
        if (
            longest_edge == 0.0
            or doubled_area < longest_edge
            or misses_vertex_coverage
        ):
            clipped_fixed_points = np.rint(
                (
                    clipped_triangle - origin
                )
                * fixed_scale
            ).astype(np.int32)
            cv2.polylines(
                mask,
                [clipped_fixed_points],
                isClosed=True,
                color=255,
                thickness=1,
                lineType=cv2.LINE_8,
                shift=_UV_RASTER_FIXED_POINT_BITS,
            )
            for column, row in representative_pixels:
                mask[
                    int(np.clip(row - y0, 0, mask.shape[0] - 1)),
                    int(np.clip(column - x0, 0, mask.shape[1] - 1)),
                ] = 255
    rows, _columns = np.nonzero(mask)
    if not len(rows):
        representatives = np.concatenate(
            (pixel_points.reshape((-1, 2)), np.mean(pixel_points, axis=1)),
            axis=0,
        )
        representative_columns = np.clip(
            np.rint(representatives[:, 0]).astype(np.int64),
            x0,
            x1 - 1,
        )
        representative_rows = np.clip(
            np.rint(representatives[:, 1]).astype(np.int64),
            y0,
            y1 - 1,
        )
        for row, column in zip(
            representative_rows,
            representative_columns,
            strict=True,
        ):
            mask[int(row) - y0, int(column) - x0] = 255
    # Keep the complete vertex-bounded rectangle even where a subpixel taper
    # does not cover a texel center. UV placement transforms every retained
    # vertex, while the mask only decides which source pixels are copied. If
    # bounds are cropped to nonzero mask pixels, a long thin triangle's apex
    # can transform outside the destination atlas after rotation or scaling.
    bounds = _PixelRectangle(
        x=x0,
        y=y0,
        width=x1 - x0,
        height=y1 - y0,
    )
    return bounds, np.asarray(mask != 0, dtype=bool)


def _align_uv_mask_to_pack_lattice(
    bounds: _PixelRectangle,
    mask: np.ndarray,
) -> tuple[_PixelRectangle, np.ndarray]:
    """Pad a regular chart so rigid moves preserve every lower mip phase."""

    lattice = _UV_PACK_LATTICE_PIXELS
    x0 = (bounds.x // lattice) * lattice
    y0 = (bounds.y // lattice) * lattice
    x1 = min(
        TEXTURE_RESOLUTION_2048,
        _ceil_to_pack_lattice(bounds.right),
    )
    y1 = min(
        TEXTURE_RESOLUTION_2048,
        _ceil_to_pack_lattice(bounds.bottom),
    )
    aligned = _PixelRectangle(x0, y0, x1 - x0, y1 - y0)
    padded = np.zeros((aligned.height, aligned.width), dtype=bool)
    offset_x = bounds.x - aligned.x
    offset_y = bounds.y - aligned.y
    padded[
        offset_y : offset_y + bounds.height,
        offset_x : offset_x + bounds.width,
    ] = mask
    return aligned, padded


def _ceil_to_pack_lattice(value: int) -> int:
    lattice = _UV_PACK_LATTICE_PIXELS
    return ((int(value) + lattice - 1) // lattice) * lattice


# ### Overlapping UV consolidation ###
def _coalesce_overlapping_islands(
    islands: list[_UvIsland],
) -> list[_UvPackGroup]:
    parents = list(range(len(islands)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[max(first_root, second_root)] = min(first_root, second_root)

    ordered = sorted(
        islands,
        key=lambda island: (
            island.source_bounds.x,
            island.source_bounds.y,
            island.island_id,
        ),
    )
    active: list[_UvIsland] = []
    comparison_count = 0
    compared_pixel_count = 0
    for island in ordered:
        active = [
            candidate
            for candidate in active
            if candidate.source_bounds.right > island.source_bounds.x
        ]
        if island.source_rgba is not None:
            continue
        for candidate in active:
            comparison_count += 1
            compared_pixel_count += _rectangle_intersection_area(
                candidate.source_bounds,
                island.source_bounds,
            )
            if comparison_count > _MAX_UV_OVERLAP_COMPARISONS:
                raise ValueError(
                    "The retained UV overlap graph exceeds the bounded "
                    "symmetric packing complexity limit."
                )
            if compared_pixel_count > _MAX_UV_OVERLAP_TEST_PIXELS:
                raise ValueError(
                    "The retained UV overlap masks exceed the bounded "
                    "symmetric packing work limit."
                )
            if _island_masks_overlap(candidate, island):
                union(candidate.island_id, island.island_id)
        active.append(island)
    ids_by_root: dict[int, list[int]] = {}
    for island in islands:
        ids_by_root.setdefault(find(island.island_id), []).append(
            island.island_id
        )
    island_by_id = {island.island_id: island for island in islands}
    groups: list[_UvPackGroup] = []
    combined_mask_pixels = 0
    for island_ids in sorted(ids_by_root.values(), key=lambda values: values[0]):
        members = [island_by_id[island_id] for island_id in island_ids]
        repaired_members = [
            member for member in members if member.source_rgba is not None
        ]
        if repaired_members and len(members) != 1:
            raise RuntimeError("A repaired UV chart was incorrectly coalesced.")
        x0 = min(member.source_bounds.x for member in members)
        y0 = min(member.source_bounds.y for member in members)
        x1 = max(member.source_bounds.right for member in members)
        y1 = max(member.source_bounds.bottom for member in members)
        combined_mask_pixels += (x1 - x0) * (y1 - y0)
        if combined_mask_pixels > _MAX_UV_MASK_PIXELS:
            raise ValueError(
                "The consolidated UV masks exceed the bounded packing memory "
                "limit."
            )
        combined = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        for member in members:
            bounds = member.source_bounds
            local_y = bounds.y - y0
            local_x = bounds.x - x0
            combined[
                local_y : local_y + bounds.height,
                local_x : local_x + bounds.width,
            ] |= member.source_mask
        groups.append(
            _UvPackGroup(
                group_id=len(groups),
                island_ids=tuple(sorted(island_ids)),
                source_bounds=_PixelRectangle(x0, y0, x1 - x0, y1 - y0),
                source_mask=combined,
                exact_repeat_seams=_RepeatSeamEdges(
                    u_zero=any(
                        member.exact_repeat_seams.u_zero
                        for member in members
                    ),
                    u_one=any(
                        member.exact_repeat_seams.u_one
                        for member in members
                    ),
                    v_zero=any(
                        member.exact_repeat_seams.v_zero
                        for member in members
                    ),
                    v_one=any(
                        member.exact_repeat_seams.v_one
                        for member in members
                    ),
                ),
                source_rgba=(
                    repaired_members[0].source_rgba
                    if repaired_members
                    else None
                ),
            )
        )
    return groups


def _island_masks_overlap(first: _UvIsland, second: _UvIsland) -> bool:
    first_bounds = first.source_bounds
    second_bounds = second.source_bounds
    if not _rectangles_overlap(first_bounds, second_bounds):
        return False
    x0 = max(first_bounds.x, second_bounds.x)
    y0 = max(first_bounds.y, second_bounds.y)
    x1 = min(first_bounds.right, second_bounds.right)
    y1 = min(first_bounds.bottom, second_bounds.bottom)
    first_region = first.source_mask[
        y0 - first_bounds.y : y1 - first_bounds.y,
        x0 - first_bounds.x : x1 - first_bounds.x,
    ]
    second_region = second.source_mask[
        y0 - second_bounds.y : y1 - second_bounds.y,
        x0 - second_bounds.x : x1 - second_bounds.x,
    ]
    return bool(np.any(first_region & second_region))


# ### UV rectangle packing ###
def _repack_retained_texture(
    scene: trimesh.Scene,
    source_rgba: np.ndarray,
) -> np.ndarray:
    _validate_canonical_texture(source_rgba)
    islands = _prepare_uv_islands(scene, source_rgba)
    groups = _coalesce_overlapping_islands(islands)
    placements = _pack_uv_groups(groups)
    packed = _opaque_black_texture()
    group_by_id = {group.group_id: group for group in groups}
    for placement in placements:
        group = group_by_id[placement.group_id]
        patch = _build_retained_texture_patch(
            source_rgba,
            group,
            placement,
        )
        if placement.rotated_clockwise:
            patch = np.rot90(patch, k=-1)
        destination = placement.destination
        if patch.shape[:2] != (destination.height, destination.width):
            raise RuntimeError("The UV packer produced an invalid texture patch.")
        packed[
            destination.y : destination.bottom,
            destination.x : destination.right,
        ] = patch
    _apply_uv_pack_placements(scene, islands, groups, placements)
    _validate_scene_uvs_in_left_half(scene)
    return np.ascontiguousarray(packed)


def _group_touches_repeat_boundary(group: _UvPackGroup) -> bool:
    bounds = group.source_bounds
    return bool(
        group.source_rgba is None
        and (
            bounds.x == 0
            or bounds.y == 0
            or bounds.right == TEXTURE_RESOLUTION_2048
            or bounds.bottom == TEXTURE_RESOLUTION_2048
        )
    )


def _pack_uv_groups(groups: list[_UvPackGroup]) -> list[_UvPackPlacement]:
    half_width = TEXTURE_RESOLUTION_2048 // 2
    full_height = TEXTURE_RESOLUTION_2048
    for gutter_pixels in (16, 8, 4):
        rigid = _try_pack_uv_groups_at_scale(
            groups,
            scale=1.0,
            gutter_pixels=gutter_pixels,
            target_width=half_width,
            target_height=full_height,
        )
        if rigid is not None:
            return rigid
    return _find_maximum_uniform_scale_pack(
        groups,
        target_width=half_width,
        target_height=full_height,
    )


def _find_maximum_uniform_scale_pack(
    groups: list[_UvPackGroup],
    *,
    target_width: int,
    target_height: int,
) -> list[_UvPackPlacement]:
    best = _try_pack_uv_groups_at_scale(
        groups,
        scale=_MIN_GLOBAL_UV_SCALE,
        gutter_pixels=_SCALED_UV_GUTTER_PIXELS,
        target_width=target_width,
        target_height=target_height,
    )
    if best is None:
        raise RuntimeError("The minimum uniform UV grid unexpectedly failed.")
    low = _MIN_GLOBAL_UV_SCALE
    high = 1.0
    for _pass_index in range(_UNIFORM_SCALE_SEARCH_PASSES):
        midpoint = (low + high) * 0.5
        candidate = _try_pack_uv_groups_at_scale(
            groups,
            scale=midpoint,
            gutter_pixels=_SCALED_UV_GUTTER_PIXELS,
            target_width=target_width,
            target_height=target_height,
        )
        if candidate is None:
            high = midpoint
        else:
            low = midpoint
            best = candidate
    return best


def _try_pack_uv_groups_at_scale(
    groups: list[_UvPackGroup],
    *,
    scale: float,
    gutter_pixels: int,
    target_width: int,
    target_height: int,
) -> list[_UvPackPlacement] | None:
    gutter_pixels_by_group = {
        group.group_id: max(
            gutter_pixels,
            _SCALED_UV_GUTTER_PIXELS,
        )
        if _group_touches_repeat_boundary(group)
        else gutter_pixels
        for group in groups
    }
    sizes = {
        group.group_id: _packed_group_outer_size(
            group,
            scale,
            gutter_pixels_by_group[group.group_id],
        )
        for group in groups
    }
    if sum(width * height for width, height in sizes.values()) > (
        target_width * target_height
    ):
        return None
    if len(groups) > _MAX_UNIFORM_SCALE_MAXRECTS_GROUPS:
        return _try_shelf_pack_uv_groups(
            groups,
            sizes,
            target_width,
            target_height,
            gutter_pixels_by_group,
            scale,
        )
    orderings = (
        lambda group: (
            -(sizes[group.group_id][0] * sizes[group.group_id][1]),
            -max(sizes[group.group_id]),
            group.group_id,
        ),
        lambda group: (-max(sizes[group.group_id]), group.group_id),
        lambda group: (-sizes[group.group_id][1], group.group_id),
        lambda group: (-sizes[group.group_id][0], group.group_id),
    )
    for ordering in orderings:
        packed = _try_pack_uv_groups(
            sorted(groups, key=ordering),
            sizes,
            target_width,
            target_height,
            gutter_pixels_by_group,
            scale,
        )
        if packed is not None:
            return packed
    return None


def _try_shelf_pack_uv_groups(
    groups: list[_UvPackGroup],
    sizes: dict[int, tuple[int, int]],
    target_width: int,
    target_height: int,
    gutter_pixels_by_group: dict[int, int],
    scale: float,
) -> list[_UvPackPlacement] | None:
    """Bound packing work for very high chart counts with ordered shelves."""

    oriented: list[tuple[_UvPackGroup, int, int, bool]] = []
    for group in groups:
        width, height = sizes[group.group_id]
        candidates = [
            (width, height, False),
            (height, width, True),
        ]
        fitting = [
            candidate
            for candidate in candidates
            if candidate[0] <= target_width
            and candidate[1] <= target_height
        ]
        if not fitting:
            return None
        chosen_width, chosen_height, rotated = min(
            fitting,
            key=lambda candidate: (
                candidate[1],
                -candidate[0],
                int(candidate[2]),
            ),
        )
        oriented.append(
            (group, chosen_width, chosen_height, rotated)
        )
    oriented.sort(
        key=lambda item: (-item[2], -item[1], item[0].group_id)
    )
    placements: list[_UvPackPlacement] = []
    cursor_x = 0
    cursor_y = 0
    shelf_height = 0
    for group, width, height, rotated in oriented:
        if cursor_x + width > target_width:
            cursor_y += shelf_height
            cursor_x = 0
            shelf_height = 0
        if cursor_y + height > target_height:
            return None
        destination = _PixelRectangle(cursor_x, cursor_y, width, height)
        placements.append(
            _UvPackPlacement(
                group_id=group.group_id,
                destination=destination,
                rotated_clockwise=rotated,
                gutter_pixels=gutter_pixels_by_group[group.group_id],
                scale=scale,
            )
        )
        cursor_x += width
        shelf_height = max(shelf_height, height)
    result = sorted(placements, key=lambda placement: placement.group_id)
    _validate_pack_lattice(result)
    return result


def _scaled_group_content_size(
    group: _UvPackGroup,
    scale: float,
) -> tuple[int, int]:
    return _scaled_group_content_size_from_bounds(group.source_bounds, scale)


def _packed_group_outer_size(
    group: _UvPackGroup,
    scale: float,
    gutter_pixels: int,
) -> tuple[int, int]:
    return _packed_outer_size_from_bounds(
        group.source_bounds,
        scale,
        gutter_pixels,
    )


def _packed_outer_size_from_bounds(
    bounds: _PixelRectangle,
    scale: float,
    gutter_pixels: int,
) -> tuple[int, int]:
    content_width, content_height = _scaled_group_content_size_from_bounds(
        bounds,
        scale,
    )
    return (
        _ceil_to_pack_lattice(content_width + 2 * gutter_pixels),
        _ceil_to_pack_lattice(content_height + 2 * gutter_pixels),
    )


def _scaled_group_content_size_from_bounds(
    bounds: _PixelRectangle,
    scale: float,
) -> tuple[int, int]:
    if not math.isfinite(scale) or scale <= 0.0 or scale > 1.0:
        raise ValueError("The uniform UV packing scale must be in (0, 1].")
    return (
        max(1, int(math.ceil(bounds.width * scale - 1e-9))),
        max(1, int(math.ceil(bounds.height * scale - 1e-9))),
    )


def _try_pack_uv_groups(
    ordered_groups: list[_UvPackGroup],
    sizes: dict[int, tuple[int, int]],
    target_width: int,
    target_height: int,
    gutter_pixels_by_group: dict[int, int],
    scale: float,
) -> list[_UvPackPlacement] | None:
    free_rectangles = [_PixelRectangle(0, 0, target_width, target_height)]
    placements: list[_UvPackPlacement] = []
    for group in ordered_groups:
        source_width, source_height = sizes[group.group_id]
        candidates: list[
            tuple[tuple[int, ...], _PixelRectangle, bool]
        ] = []
        orientations = ((source_width, source_height, False),)
        if source_width != source_height:
            orientations += ((source_height, source_width, True),)
        for width, height, rotated in orientations:
            for free in free_rectangles:
                if width > free.width or height > free.height:
                    continue
                horizontal_remainder = free.width - width
                vertical_remainder = free.height - height
                destination = _PixelRectangle(
                    free.x,
                    free.y,
                    width,
                    height,
                )
                score = (
                    min(horizontal_remainder, vertical_remainder),
                    max(horizontal_remainder, vertical_remainder),
                    free.width * free.height - width * height,
                    destination.y,
                    destination.x,
                    int(rotated),
                )
                candidates.append((score, destination, rotated))
        if not candidates:
            return None
        _score, destination, rotated = min(
            candidates,
            key=lambda candidate: candidate[0],
        )
        placements.append(
            _UvPackPlacement(
                group_id=group.group_id,
                destination=destination,
                rotated_clockwise=rotated,
                gutter_pixels=gutter_pixels_by_group[group.group_id],
                scale=scale,
            )
        )
        free_rectangles = _split_free_rectangles(
            free_rectangles,
            destination,
        )
    result = sorted(placements, key=lambda placement: placement.group_id)
    _validate_pack_lattice(result)
    return result


def _validate_pack_lattice(placements: list[_UvPackPlacement]) -> None:
    lattice = _UV_PACK_LATTICE_PIXELS
    for placement in placements:
        destination = placement.destination
        if any(
            value % lattice
            for value in (
                destination.x,
                destination.y,
                destination.width,
                destination.height,
            )
        ):
            raise RuntimeError("A UV placement escaped the lower-mip lattice.")


def _split_free_rectangles(
    free_rectangles: list[_PixelRectangle],
    used: _PixelRectangle,
) -> list[_PixelRectangle]:
    split: list[_PixelRectangle] = []
    for free in free_rectangles:
        if not _rectangles_overlap(free, used):
            split.append(free)
            continue
        if used.x > free.x:
            split.append(
                _PixelRectangle(free.x, free.y, used.x - free.x, free.height)
            )
        if used.right < free.right:
            split.append(
                _PixelRectangle(
                    used.right,
                    free.y,
                    free.right - used.right,
                    free.height,
                )
            )
        if used.y > free.y:
            split.append(
                _PixelRectangle(free.x, free.y, free.width, used.y - free.y)
            )
        if used.bottom < free.bottom:
            split.append(
                _PixelRectangle(
                    free.x,
                    used.bottom,
                    free.width,
                    free.bottom - used.bottom,
                )
            )
    positive = [
        rectangle
        for rectangle in split
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


def _rectangles_overlap(
    first: _PixelRectangle,
    second: _PixelRectangle,
) -> bool:
    return not (
        first.right <= second.x
        or second.right <= first.x
        or first.bottom <= second.y
        or second.bottom <= first.y
    )


def _rectangle_contains(
    outer: _PixelRectangle,
    inner: _PixelRectangle,
) -> bool:
    return (
        outer.x <= inner.x
        and outer.y <= inner.y
        and outer.right >= inner.right
        and outer.bottom >= inner.bottom
    )


def _rectangle_intersection_area(
    first: _PixelRectangle,
    second: _PixelRectangle,
) -> int:
    width = max(0, min(first.right, second.right) - max(first.x, second.x))
    height = max(
        0,
        min(first.bottom, second.bottom) - max(first.y, second.y),
    )
    return width * height


# ### Retained texture reconstruction ###
def _build_retained_texture_patch(
    source_rgba: np.ndarray,
    group: _UvPackGroup,
    placement: _UvPackPlacement,
) -> np.ndarray:
    if placement.scale == 1.0:
        return _build_rigid_retained_texture_patch(
            source_rgba,
            group,
            placement.gutter_pixels,
        )
    return _build_uniformly_scaled_texture_patch(
        source_rgba,
        group,
        placement,
    )


def _build_rigid_retained_texture_patch(
    source_rgba: np.ndarray,
    group: _UvPackGroup,
    gutter: int,
) -> np.ndarray:
    bounds = group.source_bounds
    outer_width, outer_height = _packed_group_outer_size(
        group,
        1.0,
        gutter,
    )
    outer_core = np.zeros(
        (outer_height, outer_width),
        dtype=bool,
    )
    outer_core[
        gutter : gutter + bounds.height,
        gutter : gutter + bounds.width,
    ] = group.source_mask
    source_region = _get_group_source_region(source_rgba, group)
    patch = np.empty((*outer_core.shape, 4), dtype=np.uint8)
    patch[:] = _OPAQUE_BLACK
    core_region = patch[
        gutter : gutter + bounds.height,
        gutter : gutter + bounds.width,
    ]
    core_region[group.source_mask] = source_region[group.source_mask]
    patch = _extend_texture_edge_colors(patch, outer_core, gutter)
    if group.source_rgba is None and gutter:
        _restore_source_texture_neighborhood(
            patch,
            _build_texture_gutter_mask(outer_core, gutter),
            bounds,
            source_rgba,
            1.0,
            gutter,
        )
        if _group_touches_repeat_boundary(group):
            _restore_scaled_repeat_boundary_samples(
                patch,
                outer_core,
                bounds,
                source_rgba,
                1.0,
                gutter,
                gutter,
                group.exact_repeat_seams,
            )
    return patch


def _restore_repeat_atlas_boundary_gutters(
    patch: np.ndarray,
    source_bounds: _PixelRectangle,
    source_rgba: np.ndarray,
    gutter: int,
    eligible_gutter: np.ndarray,
) -> None:
    """Restore glTF REPEAT samples only beyond a global atlas boundary."""

    resolution = TEXTURE_RESOLUTION_2048
    source_columns = (
        source_bounds.x + np.arange(patch.shape[1], dtype=np.int64) - gutter
    )
    source_rows = (
        source_bounds.y + np.arange(patch.shape[0], dtype=np.int64) - gutter
    )
    outside_columns = (source_columns < 0) | (source_columns >= resolution)
    outside_rows = (source_rows < 0) | (source_rows >= resolution)
    if not np.any(outside_columns) and not np.any(outside_rows):
        return
    wrapped_columns = np.mod(source_columns, resolution)
    wrapped_rows = np.mod(source_rows, resolution)
    outside = (
        outside_rows[:, np.newaxis] | outside_columns[np.newaxis, :]
    ) & eligible_gutter
    repeated = source_rgba[np.ix_(wrapped_rows, wrapped_columns)]
    patch[outside] = repeated[outside]


def _build_uniformly_scaled_texture_patch(
    source_rgba: np.ndarray,
    group: _UvPackGroup,
    placement: _UvPackPlacement,
) -> np.ndarray:
    bounds = group.source_bounds
    source_region = _get_group_source_region(source_rgba, group)
    filled_source = np.empty((*group.source_mask.shape, 4), dtype=np.uint8)
    filled_source[:] = _OPAQUE_BLACK
    filled_source[group.source_mask] = source_region[group.source_mask]
    _extend_texture_edge_colors(
        filled_source,
        group.source_mask,
        bounds.width + bounds.height,
    )
    uses_repeat_boundary = _group_touches_repeat_boundary(group)
    sampling_halo = 1
    sampling_source = np.pad(
        filled_source,
        (
            (sampling_halo, sampling_halo),
            (sampling_halo, sampling_halo),
            (0, 0),
        ),
        mode="edge",
    )
    if uses_repeat_boundary:
        _restore_repeat_atlas_boundary_gutters(
            sampling_source,
            bounds,
            source_rgba,
            sampling_halo,
            np.ones(sampling_source.shape[:2], dtype=bool),
        )
    content_width, content_height = _scaled_group_content_size(
        group,
        placement.scale,
    )
    center_offset = (placement.scale - 1.0) * 0.5
    color_affine = np.asarray(
        (
            (
                placement.scale,
                0.0,
                center_offset - placement.scale * sampling_halo,
            ),
            (
                0.0,
                placement.scale,
                center_offset - placement.scale * sampling_halo,
            ),
        ),
        dtype=np.float64,
    )
    coverage_affine = np.asarray(
        (
            (placement.scale, 0.0, center_offset),
            (0.0, placement.scale, center_offset),
        ),
        dtype=np.float64,
    )
    source_alpha = (
        np.asarray(sampling_source[:, :, 3], dtype=np.float32) / 255.0
    )
    premultiplied = np.empty(sampling_source.shape, dtype=np.float32)
    premultiplied[:, :, :3] = (
        np.asarray(sampling_source[:, :, :3], dtype=np.float32)
        * source_alpha[:, :, np.newaxis]
    )
    premultiplied[:, :, 3] = source_alpha
    scaled_premultiplied = cv2.warpAffine(
        premultiplied,
        color_affine,
        (content_width, content_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0.0, 0.0, 0.0, 0.0),
    )
    scaled_colors = np.zeros(
        (content_height, content_width, 4),
        dtype=np.uint8,
    )
    scaled_alpha = np.asarray(scaled_premultiplied[:, :, 3], dtype=np.float32)
    visible = scaled_alpha > 1e-8
    scaled_colors[:, :, 3] = np.asarray(
        np.clip(np.rint(scaled_alpha * 255.0), 0.0, 255.0),
        dtype=np.uint8,
    )
    scaled_colors[visible, :3] = np.asarray(
        np.clip(
            np.rint(
                scaled_premultiplied[visible, :3]
                / scaled_alpha[visible, np.newaxis]
            ),
            0.0,
            255.0,
        ),
        dtype=np.uint8,
    )
    scaled_coverage = cv2.warpAffine(
        np.asarray(group.source_mask, dtype=np.uint8) * 255,
        coverage_affine,
        (content_width, content_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    scaled_mask = scaled_coverage != 0
    if not np.any(scaled_mask):
        source_rows, source_columns = np.nonzero(group.source_mask)
        source_row = float(np.mean(source_rows))
        source_column = float(np.mean(source_columns))
        target_row = int(
            np.clip(
                round(placement.scale * (source_row + 0.5) - 0.5),
                0,
                content_height - 1,
            )
        )
        target_column = int(
            np.clip(
                round(placement.scale * (source_column + 0.5) - 0.5),
                0,
                content_width - 1,
            )
        )
        scaled_mask[target_row, target_column] = True
        representative_row = int(round(source_row))
        representative_column = int(round(source_column))
        scaled_colors[target_row, target_column] = filled_source[
            representative_row,
            representative_column,
        ]
    gutter = placement.gutter_pixels
    outer_width, outer_height = _packed_group_outer_size(
        group,
        placement.scale,
        gutter,
    )
    outer_mask = np.zeros(
        (outer_height, outer_width),
        dtype=bool,
    )
    outer_mask[
        gutter : gutter + content_height,
        gutter : gutter + content_width,
    ] = scaled_mask
    patch = np.empty((*outer_mask.shape, 4), dtype=np.uint8)
    patch[:] = _OPAQUE_BLACK
    content = patch[
        gutter : gutter + content_height,
        gutter : gutter + content_width,
    ]
    content[scaled_mask] = scaled_colors[scaled_mask]
    gutter_radius = gutter + _UV_PACK_LATTICE_PIXELS - 1
    patch = _extend_texture_edge_colors(
        patch,
        outer_mask,
        gutter_radius,
    )
    if group.source_rgba is None:
        source_neighborhood = outer_mask | _build_texture_gutter_mask(
            outer_mask,
            gutter_radius,
        )
        _restore_source_texture_neighborhood(
            patch,
            source_neighborhood,
            bounds,
            source_rgba,
            placement.scale,
            gutter,
        )
    if uses_repeat_boundary:
        _restore_scaled_repeat_boundary_samples(
            patch,
            outer_mask,
            bounds,
            source_rgba,
            placement.scale,
            gutter,
            gutter_radius,
            group.exact_repeat_seams,
        )
    return patch


def _restore_source_texture_neighborhood(
    patch: np.ndarray,
    eligible_gutter: np.ndarray,
    source_bounds: _PixelRectangle,
    source_rgba: np.ndarray,
    scale: float,
    gutter: int,
) -> None:
    """Copy the bounded source neighborhood required by texture filtering."""

    rows, columns = np.nonzero(eligible_gutter)
    if not len(rows):
        return
    resolution = float(TEXTURE_RESOLUTION_2048)
    center_offset = (scale - 1.0) * 0.5
    for start in range(0, len(rows), _SOURCE_NEIGHBORHOOD_SAMPLE_CHUNK):
        stop = min(start + _SOURCE_NEIGHBORHOOD_SAMPLE_CHUNK, len(rows))
        chunk_rows = rows[start:stop]
        chunk_columns = columns[start:stop]
        source_pixel_columns = (
            float(source_bounds.x)
            + (
                chunk_columns.astype(np.float64)
                - float(gutter)
                - center_offset
            )
            / scale
        )
        source_pixel_rows = (
            float(source_bounds.y)
            + (
                chunk_rows.astype(np.float64)
                - float(gutter)
                - center_offset
            )
            / scale
        )
        source_uvs = np.column_stack(
            (
                (source_pixel_columns + 0.5) / resolution,
                1.0 - (source_pixel_rows + 0.5) / resolution,
            )
        )
        patch[chunk_rows, chunk_columns] = (
            _sample_repeat_bilinear_premultiplied_rgba(
                source_rgba,
                source_uvs,
            )
        )


def _restore_scaled_repeat_boundary_samples(
    patch: np.ndarray,
    core_mask: np.ndarray,
    source_bounds: _PixelRectangle,
    source_rgba: np.ndarray,
    scale: float,
    gutter: int,
    gutter_radius: int,
    exact_repeat_seams: _RepeatSeamEdges,
) -> None:
    """Rebuild the repeat neighborhood consumed by lower-resolution filters."""

    resolution = TEXTURE_RESOLUTION_2048
    center_offset = (scale - 1.0) * 0.5
    eligible = core_mask | _build_texture_gutter_mask(
        core_mask,
        gutter_radius,
    )
    source_pixel_rows = (
        float(source_bounds.y)
        + (
            np.arange(patch.shape[0], dtype=np.float64)
            - float(gutter)
            - center_offset
        )
        / scale
    )
    source_pixel_columns = (
        float(source_bounds.x)
        + (
            np.arange(patch.shape[1], dtype=np.float64)
            - float(gutter)
            - center_offset
        )
        / scale
    )
    repeat_columns = np.zeros(patch.shape[1], dtype=bool)
    repeat_rows = np.zeros(patch.shape[0], dtype=bool)
    source_radius = float(gutter_radius) / scale + 1.0
    if source_bounds.x == 0 and exact_repeat_seams.u_zero:
        repeat_columns |= source_pixel_columns < source_radius
    if source_bounds.right == resolution and exact_repeat_seams.u_one:
        repeat_columns |= source_pixel_columns > (
            float(resolution - 1) - source_radius
        )
    if source_bounds.y == 0 and exact_repeat_seams.v_one:
        repeat_rows |= source_pixel_rows < source_radius
    if source_bounds.bottom == resolution and exact_repeat_seams.v_zero:
        repeat_rows |= source_pixel_rows > (
            float(resolution - 1) - source_radius
        )
    repeat_neighborhood = eligible & (
        repeat_rows[:, np.newaxis] | repeat_columns[np.newaxis, :]
    )
    rows, columns = np.nonzero(repeat_neighborhood)
    if not len(rows):
        return
    source_uvs = np.column_stack(
        (
            (source_pixel_columns[columns] + 0.5) / float(resolution),
            1.0
            - (source_pixel_rows[rows] + 0.5) / float(resolution),
        )
    )
    patch[rows, columns] = _sample_repeat_bilinear_premultiplied_rgba(
        source_rgba,
        source_uvs,
    )
    sampled_v = 1.0 - (source_pixel_rows + 0.5) / float(resolution)
    if source_bounds.x == 0:
        _write_scaled_repeat_column_samples(
            patch,
            eligible,
            gutter - 0.5,
            np.column_stack((np.zeros_like(sampled_v), sampled_v)),
            source_rgba,
        )
    if source_bounds.right == resolution:
        _write_scaled_repeat_column_samples(
            patch,
            eligible,
            gutter + scale * source_bounds.width - 0.5,
            np.column_stack((np.ones_like(sampled_v), sampled_v)),
            source_rgba,
        )
    sampled_u = (source_pixel_columns + 0.5) / float(resolution)
    if source_bounds.y == 0:
        _write_scaled_repeat_row_samples(
            patch,
            eligible,
            gutter - 0.5,
            np.column_stack((sampled_u, np.ones_like(sampled_u))),
            source_rgba,
        )
    if source_bounds.bottom == resolution:
        _write_scaled_repeat_row_samples(
            patch,
            eligible,
            gutter + scale * source_bounds.height - 0.5,
            np.column_stack((sampled_u, np.zeros_like(sampled_u))),
            source_rgba,
        )


def _write_scaled_repeat_column_samples(
    patch: np.ndarray,
    eligible: np.ndarray,
    seam_column: float,
    source_uvs: np.ndarray,
    source_rgba: np.ndarray,
) -> None:
    samples = _sample_repeat_bilinear_premultiplied_rgba(
        source_rgba,
        source_uvs,
    )
    for column in {math.floor(seam_column), math.floor(seam_column) + 1}:
        if column < 0 or column >= patch.shape[1]:
            continue
        rows = eligible[:, column]
        patch[rows, column] = samples[rows]


def _write_scaled_repeat_row_samples(
    patch: np.ndarray,
    eligible: np.ndarray,
    seam_row: float,
    source_uvs: np.ndarray,
    source_rgba: np.ndarray,
) -> None:
    samples = _sample_repeat_bilinear_premultiplied_rgba(
        source_rgba,
        source_uvs,
    )
    for row in {math.floor(seam_row), math.floor(seam_row) + 1}:
        if row < 0 or row >= patch.shape[0]:
            continue
        columns = eligible[row]
        patch[row, columns] = samples[columns]


def _get_group_source_region(
    source_rgba: np.ndarray,
    group: _UvPackGroup,
) -> np.ndarray:
    if group.source_rgba is not None:
        expected_shape = (
            group.source_bounds.height,
            group.source_bounds.width,
            4,
        )
        if group.source_rgba.shape != expected_shape:
            raise RuntimeError("A repaired UV chart has invalid source pixels.")
        return group.source_rgba
    bounds = group.source_bounds
    return source_rgba[
        bounds.y : bounds.bottom,
        bounds.x : bounds.right,
    ]


def _extend_texture_edge_colors(
    texture: np.ndarray,
    core_mask: np.ndarray,
    radius: int,
) -> np.ndarray:
    gutter_mask = _build_texture_gutter_mask(core_mask, radius)
    if not np.any(gutter_mask):
        return texture
    inverse = np.asarray(~core_mask, dtype=np.uint8)
    distances, labels = cv2.distanceTransformWithLabels(
        inverse,
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    core_positions = np.argwhere(core_mask)
    nearest_labels = labels[gutter_mask] - 1
    if np.any(nearest_labels < 0) or np.any(nearest_labels >= len(core_positions)):
        raise RuntimeError("The UV gutter could not find a retained texel.")
    nearest_positions = core_positions[nearest_labels]
    texture[gutter_mask] = texture[
        nearest_positions[:, 0],
        nearest_positions[:, 1],
    ]
    return texture


def _build_texture_gutter_mask(
    core_mask: np.ndarray,
    radius: int,
) -> np.ndarray:
    if radius <= 0:
        return np.zeros_like(core_mask, dtype=bool)
    inverse = np.asarray(~core_mask, dtype=np.uint8)
    distances, _labels = cv2.distanceTransformWithLabels(
        inverse,
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    return (~core_mask) & (distances <= float(radius) + 1e-6)


def _mask_existing_left_texture(
    scene: trimesh.Scene,
    source_rgba: np.ndarray,
) -> np.ndarray:
    _validate_canonical_texture(source_rgba)
    _validate_scene_uvs_in_left_half(scene)
    resolution = TEXTURE_RESOLUTION_2048
    half_width = resolution // 2
    core_mask = np.zeros((resolution, half_width), dtype=bool)
    found_faces = False
    for geometry in scene.geometry.values():
        if not isinstance(geometry, trimesh.Trimesh):
            continue
        if not _material_textures(geometry):
            continue
        uvs = _get_vertex_uvs(geometry)
        if uvs is None:
            raise ValueError("A textured mesh is missing UV coordinates.")
        faces = np.asarray(geometry.faces, dtype=np.int64)
        if not len(faces):
            continue
        found_faces = True
        bounds, source_mask = _rasterize_uv_triangles(uvs, faces)
        if bounds.x >= half_width:
            continue
        copied_width = min(bounds.right, half_width) - bounds.x
        core_mask[
            bounds.y : bounds.bottom,
            bounds.x : bounds.x + copied_width,
        ] |= source_mask[:, :copied_width]
    if not found_faces or not np.any(core_mask):
        raise ValueError("The retained GLB has no textured UV triangle area.")
    packed = _opaque_black_texture()
    left_texture = packed[:, :half_width]
    source_left = source_rgba[:, :half_width]
    left_texture[core_mask] = source_left[core_mask]
    _extend_texture_edge_colors(left_texture, core_mask, _UV_GUTTER_PIXELS)
    return np.ascontiguousarray(packed)


def _mask_existing_square_pair_texture(
    scene: trimesh.Scene,
    source_rgba: np.ndarray,
) -> np.ndarray:
    """Preserve a physical 1024 pair source without an upscale round trip."""

    expected_shape = (
        TEXTURE_RESOLUTION_1024,
        TEXTURE_RESOLUTION_1024,
        4,
    )
    if source_rgba.shape != expected_shape:
        raise ValueError(
            "The preserved square-pair texture must be 1024 x 1024 RGBA."
        )
    _validate_scene_uvs_in_left_half(scene)
    packed = np.asarray(source_rgba, dtype=np.uint8).copy()
    packed[:, TEXTURE_RESOLUTION_1024 // 2 :] = _OPAQUE_BLACK
    return np.ascontiguousarray(packed)


def _opaque_black_texture() -> np.ndarray:
    packed = np.empty(
        (
            TEXTURE_RESOLUTION_2048,
            TEXTURE_RESOLUTION_2048,
            4,
        ),
        dtype=np.uint8,
    )
    packed[:] = _OPAQUE_BLACK
    return packed


def _validate_canonical_texture(source_rgba: np.ndarray) -> None:
    expected_shape = (
        TEXTURE_RESOLUTION_2048,
        TEXTURE_RESOLUTION_2048,
        4,
    )
    if source_rgba.shape != expected_shape:
        raise ValueError("The canonical texture must be 2048 x 2048 RGBA.")


def _validate_shared_square_pair_texture(
    source_textures: list[np.ndarray],
) -> np.ndarray:
    """Accept one shared 2048 provider or packed 1024 operation texture."""

    source_texture = np.asarray(source_textures[0])
    if any(
        not np.array_equal(source_texture, np.asarray(texture))
        for texture in source_textures[1:]
    ):
        raise ValueError(
            "The generated object contains more than one distinct base-color "
            "texture atlas."
        )
    valid_shapes = {
        (TEXTURE_RESOLUTION_1024, TEXTURE_RESOLUTION_1024, 4),
        (TEXTURE_RESOLUTION_2048, TEXTURE_RESOLUTION_2048, 4),
    }
    if source_texture.shape not in valid_shapes:
        raise ValueError(
            "Square-pair packing requires a 2048 x 2048 provider texture or "
            "a preserved 1024 x 1024 pair texture."
        )
    return source_texture


# ### UV transform application ###
def _apply_uv_pack_placements(
    scene: trimesh.Scene,
    islands: list[_UvIsland],
    groups: list[_UvPackGroup],
    placements: list[_UvPackPlacement],
) -> None:
    island_by_id = {island.island_id: island for island in islands}
    group_by_id = {group.group_id: group for group in groups}
    uvs_by_geometry: dict[object, np.ndarray] = {}
    for geometry_name, geometry in scene.geometry.items():
        if isinstance(geometry, trimesh.Trimesh) and _material_textures(geometry):
            uvs = _get_vertex_uvs(geometry)
            if uvs is None:
                raise ValueError("A textured mesh is missing UV coordinates.")
            uvs_by_geometry[geometry_name] = uvs.copy()
    for placement in placements:
        group = group_by_id[placement.group_id]
        for island_id in group.island_ids:
            island = island_by_id[island_id]
            geometry_uvs = uvs_by_geometry[island.geometry_name]
            indices = island.vertex_indices
            source_uvs = geometry_uvs[indices]
            if island.source_pixel_coordinates is not None:
                resolution = float(TEXTURE_RESOLUTION_2048)
                pixels = island.source_pixel_coordinates
                source_uvs = np.column_stack(
                    (
                        (pixels[:, 0] + 0.5) / resolution,
                        1.0 - (pixels[:, 1] + 0.5) / resolution,
                    )
                )
            geometry_uvs[indices] = _transform_uvs_for_placement(
                source_uvs,
                group.source_bounds,
                placement,
            )
    for geometry_name, uvs in uvs_by_geometry.items():
        geometry = scene.geometry[geometry_name]
        geometry.visual.uv = uvs


def _transform_uvs_for_placement(
    source_uvs: np.ndarray,
    source_bounds: _PixelRectangle,
    placement: _UvPackPlacement,
) -> np.ndarray:
    resolution = float(TEXTURE_RESOLUTION_2048)
    source_pixel_x = source_uvs[:, 0] * resolution - 0.5
    source_pixel_y = (1.0 - source_uvs[:, 1]) * resolution - 0.5
    gutter = float(placement.gutter_pixels)
    scale = float(placement.scale)
    destination = placement.destination
    if scale == 1.0:
        scaled_local_x = source_pixel_x - float(source_bounds.x)
        scaled_local_y = source_pixel_y - float(source_bounds.y)
    else:
        scaled_local_x = scale * (
            source_pixel_x - float(source_bounds.x) + 0.5
        ) - 0.5
        scaled_local_y = scale * (
            source_pixel_y - float(source_bounds.y) + 0.5
        ) - 0.5
    if placement.rotated_clockwise:
        _outer_width, outer_height = _packed_outer_size_from_bounds(
            source_bounds,
            scale,
            placement.gutter_pixels,
        )
        destination_pixel_x = (
            float(destination.x)
            + float(outer_height)
            - 1.0
            - (gutter + scaled_local_y)
        )
        destination_pixel_y = (
            float(destination.y)
            + gutter
            + scaled_local_x
        )
    else:
        destination_pixel_x = (
            float(destination.x)
            + gutter
            + scaled_local_x
        )
        destination_pixel_y = (
            float(destination.y)
            + gutter
            + scaled_local_y
        )
    return np.column_stack(
        (
            (destination_pixel_x + 0.5) / resolution,
            1.0 - (destination_pixel_y + 0.5) / resolution,
        )
    )


def _map_scene_uvs_to_top_left_quarter(scene: trimesh.Scene) -> None:
    gutter = float(_QUARTER_CONTENT_GUTTER_PIXELS_1024)
    atlas_resolution = float(TEXTURE_RESOLUTION_2048)
    inner_resolution = float(
        TEXTURE_RESOLUTION_1024
        - 2 * _QUARTER_CONTENT_GUTTER_PIXELS_1024
    )
    found_textured = False
    for geometry in scene.geometry.values():
        if not isinstance(geometry, trimesh.Trimesh):
            continue
        if not _material_textures(geometry):
            continue
        found_textured = True
        uvs = _get_vertex_uvs(geometry)
        if uvs is None:
            raise ValueError("A textured mesh is missing UV coordinates.")
        geometry.visual.uv = np.column_stack(
            (
                (gutter + inner_resolution * uvs[:, 0])
                / atlas_resolution,
                1.0
                - (
                    gutter
                    + inner_resolution * (1.0 - uvs[:, 1])
                )
                / atlas_resolution,
            )
        )
    if not found_textured:
        raise ValueError("The retained GLB contains no textured mesh UVs.")


def _validate_scene_uvs_in_top_left_quarter(scene: trimesh.Scene) -> None:
    found_textured = False
    tolerance = _REPEAT_UV_TOLERANCE
    inset = (
        float(_QUARTER_CONTENT_GUTTER_PIXELS_1024)
        / float(TEXTURE_RESOLUTION_2048)
    )
    for geometry in scene.geometry.values():
        if not isinstance(geometry, trimesh.Trimesh):
            continue
        if not _material_textures(geometry):
            continue
        found_textured = True
        uvs = _get_vertex_uvs(geometry, allow_repeat=True)
        if uvs is None:
            raise ValueError("A textured mesh is missing UV coordinates.")
        if (
            np.any(uvs[:, 0] < inset - tolerance)
            or np.any(uvs[:, 0] > 0.5 - inset + tolerance)
            or np.any(uvs[:, 1] < 0.5 + inset - tolerance)
            or np.any(uvs[:, 1] > 1.0 - inset + tolerance)
        ):
            raise ValueError(
                "Preserved symmetric UVs must fit inside the top-left "
                "texture quadrant."
            )
    if not found_textured:
        raise ValueError("The retextured GLB contains no textured mesh UVs.")


def _validate_scene_uvs_in_left_half(scene: trimesh.Scene) -> None:
    found_textured = False
    tolerance = 1e-7
    for geometry in scene.geometry.values():
        if not isinstance(geometry, trimesh.Trimesh):
            continue
        if not _material_textures(geometry):
            continue
        found_textured = True
        uvs = _get_vertex_uvs(geometry)
        if uvs is None:
            raise ValueError("A textured mesh is missing UV coordinates.")
        if (
            np.any(uvs[:, 0] < -tolerance)
            or np.any(uvs[:, 0] > 0.5 + tolerance)
            or np.any(uvs[:, 1] < -tolerance)
            or np.any(uvs[:, 1] > 1.0 + tolerance)
        ):
            raise ValueError(
                "Preserved symmetric UV triangles must fit inside the left "
                "half of the texture."
            )
    if not found_textured:
        raise ValueError("The retextured GLB contains no textured mesh UVs.")


def _scene_uvs_are_left_packed(scene: trimesh.Scene) -> bool:
    """Return whether compatibility mode can safely preserve textured UVs."""

    found_textured = False
    tolerance = _REPEAT_UV_TOLERANCE
    safety_inset = (
        float(_PRESERVED_UV_SAFETY_PIXELS)
        / float(TEXTURE_RESOLUTION_2048)
    )
    minimum_u = safety_inset
    maximum_u = 0.5 - safety_inset
    minimum_v = safety_inset
    maximum_v = 1.0 - safety_inset
    for geometry in scene.geometry.values():
        if not isinstance(geometry, trimesh.Trimesh):
            continue
        if not _material_textures(geometry):
            continue
        found_textured = True
        uvs = _get_vertex_uvs(geometry, allow_repeat=True)
        if uvs is None:
            raise ValueError("A textured mesh is missing UV coordinates.")
        if (
            np.any(uvs[:, 0] < minimum_u - tolerance)
            or np.any(uvs[:, 0] > maximum_u + tolerance)
            or np.any(uvs[:, 1] < minimum_v - tolerance)
            or np.any(uvs[:, 1] > maximum_v + tolerance)
        ):
            return False
    if not found_textured:
        raise ValueError("The retextured GLB contains no textured mesh UVs.")
    return True


# ### Texture variant export ###
def _build_square_pair_texture_variants(
    scene: trimesh.Scene,
    packed_source_texture: np.ndarray,
    material_texture_count: int,
) -> SymmetricSquarePairTextureVariants:
    """Export left-half content in physical 512 and 1024 square textures."""

    source_resolution = int(packed_source_texture.shape[0])
    if packed_source_texture.shape not in {
        (TEXTURE_RESOLUTION_1024, TEXTURE_RESOLUTION_1024, 4),
        (TEXTURE_RESOLUTION_2048, TEXTURE_RESOLUTION_2048, 4),
    }:
        raise ValueError(
            "A square-pair source texture must be 1024 or 2048 square RGBA."
        )
    texture_by_resolution = {
        resolution: (
            packed_source_texture.copy()
            if resolution == source_resolution
            else _resize_rgba(
                packed_source_texture,
                resolution,
                cv2.INTER_AREA,
            )
        )
        for resolution in SYMMETRIC_SQUARE_PAIR_CONTENT_RESOLUTIONS
    }
    glb_by_resolution: dict[int, bytes] = {}
    for resolution in SYMMETRIC_SQUARE_PAIR_CONTENT_RESOLUTIONS:
        variant_scene = _clone_scene_with_vertex_normals(scene)
        _replace_material_textures(
            variant_scene,
            [texture_by_resolution[resolution]] * material_texture_count,
        )
        try:
            glb_by_resolution[resolution] = bytes(
                variant_scene.export(file_type="glb")
            )
        except Exception as error:
            raise ValueError(
                "The automatic-symmetry square-pair GLB could not be "
                "exported."
            ) from error
    return SymmetricSquarePairTextureVariants(
        glb_by_resolution=glb_by_resolution,
        texture_png_by_resolution={
            resolution: _encode_rgba_png(texture)
            for resolution, texture in texture_by_resolution.items()
        },
        preview_rgba_by_resolution={
            resolution: texture.copy()
            for resolution, texture in texture_by_resolution.items()
        },
    )


def _build_pair_texture_variants(
    scene: trimesh.Scene,
    packed_texture_2048: np.ndarray,
    material_texture_count: int,
) -> SymmetricPairTextureVariants:
    """Export full-height left-half content at logical 512 and 1024."""

    _validate_canonical_texture(packed_texture_2048)
    texture_by_resolution = {
        TEXTURE_RESOLUTION_512: _resize_rgba(
            packed_texture_2048,
            TEXTURE_RESOLUTION_1024,
            cv2.INTER_AREA,
        ),
        TEXTURE_RESOLUTION_1024: packed_texture_2048.copy(),
    }
    glb_by_resolution: dict[int, bytes] = {}
    for content_resolution in SYMMETRIC_PAIR_CONTENT_RESOLUTIONS:
        variant_scene = _clone_scene_with_vertex_normals(scene)
        _replace_material_textures(
            variant_scene,
            [texture_by_resolution[content_resolution]]
            * material_texture_count,
        )
        try:
            glb_by_resolution[content_resolution] = bytes(
                variant_scene.export(file_type="glb")
            )
        except Exception as error:
            raise ValueError(
                "The automatic-symmetry pair GLB could not be exported."
            ) from error
    return SymmetricPairTextureVariants(
        glb_by_resolution=glb_by_resolution,
        texture_png_by_resolution={
            resolution: _encode_rgba_png(texture)
            for resolution, texture in texture_by_resolution.items()
        },
        preview_rgba_by_resolution={
            resolution: texture.copy()
            for resolution, texture in texture_by_resolution.items()
        },
    )


def _build_quarter_texture_variants(
    scene: trimesh.Scene,
    source_texture_2048: np.ndarray,
    material_texture_count: int,
    *,
    source_is_already_quarter: bool,
) -> SymmetricQuarterTextureVariants:
    """Export logical 512/1024 content in double-sized square atlases."""

    _validate_canonical_texture(source_texture_2048)
    if source_is_already_quarter:
        content_1024 = source_texture_2048[
            :TEXTURE_RESOLUTION_1024,
            :TEXTURE_RESOLUTION_1024,
        ].copy()
    else:
        inner_resolution = (
            TEXTURE_RESOLUTION_1024
            - 2 * _QUARTER_CONTENT_GUTTER_PIXELS_1024
        )
        inner_content = _resize_rgba(
            source_texture_2048,
            inner_resolution,
            cv2.INTER_AREA,
        )
        gutter = _QUARTER_CONTENT_GUTTER_PIXELS_1024
        content_1024 = np.pad(
            inner_content,
            ((gutter, gutter), (gutter, gutter), (0, 0)),
            mode="wrap",
        )
    content_512 = _resize_rgba(
        content_1024,
        TEXTURE_RESOLUTION_512,
        cv2.INTER_AREA,
    )
    texture_by_resolution = {
        TEXTURE_RESOLUTION_512: _place_content_in_top_left_quarter(
            content_512,
            TEXTURE_RESOLUTION_1024,
        ),
        TEXTURE_RESOLUTION_1024: _place_content_in_top_left_quarter(
            content_1024,
            TEXTURE_RESOLUTION_2048,
        ),
    }
    glb_by_resolution: dict[int, bytes] = {}
    for content_resolution in SYMMETRIC_QUARTER_CONTENT_RESOLUTIONS:
        variant_scene = _clone_scene_with_vertex_normals(scene)
        _replace_material_textures(
            variant_scene,
            [texture_by_resolution[content_resolution]]
            * material_texture_count,
        )
        try:
            glb_by_resolution[content_resolution] = bytes(
                variant_scene.export(file_type="glb")
            )
        except Exception as error:
            raise ValueError(
                "The automatic-symmetry GLB could not be exported."
            ) from error
    return SymmetricQuarterTextureVariants(
        glb_by_resolution=glb_by_resolution,
        texture_png_by_resolution={
            resolution: _encode_rgba_png(texture)
            for resolution, texture in texture_by_resolution.items()
        },
        preview_rgba_by_resolution={
            resolution: texture.copy()
            for resolution, texture in texture_by_resolution.items()
        },
    )


def _place_content_in_top_left_quarter(
    content_rgba: np.ndarray,
    atlas_resolution: int,
) -> np.ndarray:
    content_resolution = atlas_resolution // 2
    if np.asarray(content_rgba).shape != (
        content_resolution,
        content_resolution,
        4,
    ):
        raise ValueError("Quarter texture content has the wrong dimensions.")
    atlas = np.empty(
        (atlas_resolution, atlas_resolution, 4),
        dtype=np.uint8,
    )
    atlas[:] = _OPAQUE_BLACK
    atlas[:content_resolution, :content_resolution] = content_rgba
    return atlas


def _build_half_texture_variants(
    scene: trimesh.Scene,
    packed_texture_2048: np.ndarray,
    material_texture_count: int,
) -> ObjectTextureVariants:
    """Export direct texture reductions while retaining authored normals."""

    texture_by_resolution = {
        TEXTURE_RESOLUTION_512: _resize_rgba(
            packed_texture_2048,
            TEXTURE_RESOLUTION_512,
            cv2.INTER_AREA,
        ),
        TEXTURE_RESOLUTION_1024: _resize_rgba(
            packed_texture_2048,
            TEXTURE_RESOLUTION_1024,
            cv2.INTER_AREA,
        ),
        TEXTURE_RESOLUTION_2048: packed_texture_2048.copy(),
    }
    glb_by_resolution: dict[int, bytes] = {}
    for resolution in TEXTURE_RESOLUTIONS:
        variant_scene = _clone_scene_with_vertex_normals(scene)
        _replace_material_textures(
            variant_scene,
            [texture_by_resolution[resolution]] * material_texture_count,
        )
        try:
            glb_by_resolution[resolution] = bytes(
                variant_scene.export(file_type="glb")
            )
        except Exception as error:
            raise ValueError(
                "The symmetric-division GLB could not be exported."
            ) from error
    return ObjectTextureVariants(
        glb_by_resolution=glb_by_resolution,
        texture_png_by_resolution={
            resolution: _encode_rgba_png(texture)
            for resolution, texture in texture_by_resolution.items()
        },
        preview_rgba_by_resolution={
            resolution: texture.copy()
            for resolution, texture in texture_by_resolution.items()
        },
    )


def _clone_scene_with_vertex_normals(scene: trimesh.Scene) -> trimesh.Scene:
    """Restore normal caches that Trimesh intentionally drops on deepcopy."""

    normal_by_geometry: dict[str, np.ndarray] = {}
    for geometry_name, geometry in scene.geometry.items():
        cached = geometry._cache.cache.get("vertex_normals")
        if cached is not None:
            normal_by_geometry[str(geometry_name)] = np.asarray(
                cached,
                dtype=float,
            ).copy()
    cloned = copy.deepcopy(scene)
    for geometry_name, normals in normal_by_geometry.items():
        geometry = cloned.geometry.get(geometry_name)
        if isinstance(geometry, trimesh.Trimesh):
            geometry.vertex_normals = normals.copy()
    return cloned


# ### Validation and math helpers ###
def _validate_orientation_and_side(
    orientation: str,
    kept_side: str,
) -> None:
    _validate_orientation(orientation)
    if kept_side not in SYMMETRIC_DIVISION_SIDES_BY_ORIENTATION[orientation]:
        raise ValueError("The kept side does not match the orientation.")


def _validate_orientation(orientation: str) -> None:
    if orientation not in SYMMETRIC_DIVISION_ORIENTATIONS:
        raise ValueError("Unknown symmetric-division orientation.")


def _get_z_up_axis(orientation: str) -> int:
    if orientation == SYMMETRIC_DIVISION_ORIENTATION_VERTICAL:
        return 0
    return 2


def _get_kept_direction(kept_side: str) -> float:
    if kept_side in {
        SYMMETRIC_DIVISION_SIDE_RIGHT,
        SYMMETRIC_DIVISION_SIDE_TOP,
    }:
        return 1.0
    return -1.0


def _interpolate(
    first: np.ndarray,
    second: np.ndarray,
    fraction: float,
) -> np.ndarray:
    return np.asarray(first + (second - first) * fraction)


def _interpolate_optional(
    first: np.ndarray | None,
    second: np.ndarray | None,
    fraction: float,
) -> np.ndarray | None:
    if first is None and second is None:
        return None
    if first is None or second is None:
        raise ValueError("Mesh vertex attributes are inconsistent.")
    return _interpolate(first, second, fraction)


def _normalize_vector(vector: np.ndarray, *, fallback: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= _NORMAL_EPSILON:
        fallback_length = float(np.linalg.norm(fallback))
        if fallback_length <= _NORMAL_EPSILON:
            return np.array((0.0, 0.0, 1.0), dtype=float)
        return np.asarray(fallback, dtype=float) / fallback_length
    return np.asarray(vector, dtype=float) / length
