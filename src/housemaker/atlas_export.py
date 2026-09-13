# ### Imports ###
from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import MultiMaterial, PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.glass_material import (
    get_housemaker_glass_runtime_key,
    is_housemaker_glass_material,
)
from housemaker.glb import (
    GLB_CHUNK_HEADER_BYTE_COUNT,
    GLB_HEADER_BYTE_COUNT,
    GLB_JSON_CHUNK_TYPE,
    GLB_MAGIC,
    GLB_VERSION,
    GLTF_Y_UP_TO_Z_UP_TRANSFORM,
    HALF_MESH_EXTRAS_KEY,
    HALF_NODE_NAME_PREFIX,
    PACKED_ORM_AO_UV_ATTRIBUTE,
    GeneratedModel,
    PackedOrmMaterialSpec,
    PreviewSymmetricObject,
    _serialize_scene_glb_with_half_mesh_extras,
)
from housemaker.object_ao_baking import (
    ObjectAmbientOcclusionTarget,
    bake_placed_object_ambient_occlusion,
)
from housemaker.pbr_maps import (
    ATLAS_MAP_BASE_COLOR,
    ATLAS_MAP_TYPES,
    PBR_MAP_METALLIC,
    PBR_MAP_NORMAL,
    PBR_MAP_ROUGHNESS,
)
from housemaker.surface_ao_baking import (
    SurfaceAmbientOcclusionBakeCancelled,
    SurfaceAmbientOcclusionReceiver,
    SurfaceAmbientOcclusionReceiverLayout,
    SurfaceAmbientOcclusionUvLayout,
    bake_surface_ambient_occlusion,
    build_surface_ao_geometry_signature,
    build_surface_ao_uv_layout,
)
from housemaker.texture_atlas_state import (
    ATLAS_HALF_SLOT_PACKING_MODES,
    ATLAS_PACKING_MODE_FULL,
    ATLAS_PACKING_MODE_SYMMETRIC_QUARTER,
    TextureAtlasPlacement,
    TextureAtlasRecord,
    atlas_placement_pixel_regions,
)

# ### Constants ###
SURFACE_ID_METADATA_KEY = "housemaker_surface_id"
OBJECT_ID_METADATA_KEY = "housemaker_object_id"
ATLAS_ID_METADATA_KEY = "housemaker_atlas_id"
UV_TOLERANCE = 1e-6
GEOMETRY_EPSILON = 1e-12
MAX_TILED_SURFACE_TRIANGLES = 2_000_000
PACKED_ORM_MATERIAL_MARKER_PREFIX = "__housemaker_packed_orm__:"
AO_ONLY_MATERIAL_MARKER_PREFIX = "__housemaker_ao_only__:"
ATLAS_SAMPLER_MATERIAL_MARKER_PREFIX = "__housemaker_atlas_sampler__:"


# ### Public data models ###
@dataclass(frozen=True)
class MaterializedTextureAtlas:
    """One immutable Atlas and the genuine maps enabled for GLB export.

    Materialization still supplies neutral fallback PNGs for every map so the
    Atlas editor can switch views consistently. ``active_map_types`` keeps
    those fallbacks out of the exported material unless a source owns the map.
    The ``surface_ao_*`` names are retained project-format compatibility fields;
    their cached AO plane can cover both objects and architectural surfaces.
    """

    atlas: TextureAtlasRecord
    map_paths: Mapping[str, Path]
    active_map_types: frozenset[str] = frozenset(ATLAS_MAP_TYPES)
    surface_ao_image_path: Path | None = None
    surface_ao_geometry_signature: str | None = None
    surface_ao_intensity: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.atlas, TextureAtlasRecord):
            raise TypeError("A materialized texture Atlas requires Atlas data.")
        normalized_paths = {
            str(map_type): Path(path) for map_type, path in self.map_paths.items()
        }
        if set(normalized_paths) != set(ATLAS_MAP_TYPES):
            raise ValueError("A materialized texture Atlas requires every map.")
        missing_paths = [
            path for path in normalized_paths.values() if not path.is_file()
        ]
        if missing_paths:
            raise ValueError("A materialized texture Atlas map is missing.")
        if isinstance(
            self.active_map_types,
            (str, bytes, bytearray),
        ):
            raise TypeError(
                "Active texture Atlas maps must be provided as a collection."
            )
        try:
            active_map_types = frozenset(
                str(map_type).strip().lower() for map_type in self.active_map_types
            )
        except TypeError as error:
            raise TypeError(
                "Active texture Atlas maps must be provided as a collection."
            ) from error
        unknown_map_types = active_map_types - set(ATLAS_MAP_TYPES)
        if unknown_map_types:
            raise ValueError(
                "A materialized texture Atlas has unknown active maps: "
                + ", ".join(sorted(unknown_map_types))
            )
        if ATLAS_MAP_BASE_COLOR not in active_map_types:
            raise ValueError("A materialized texture Atlas requires active base color.")
        surface_ao_image_path = (
            None
            if self.surface_ao_image_path is None
            else Path(self.surface_ao_image_path)
        )
        has_surface_ao_image = surface_ao_image_path is not None
        has_surface_ao_signature = self.surface_ao_geometry_signature is not None
        if has_surface_ao_image != has_surface_ao_signature:
            raise ValueError(
                "A materialized surface AO image and signature must be set together."
            )
        if surface_ao_image_path is not None and not surface_ao_image_path.is_file():
            raise ValueError("A materialized surface AO image is missing.")
        surface_ao_geometry_signature = (
            None
            if self.surface_ao_geometry_signature is None
            else str(self.surface_ao_geometry_signature).strip().lower()
        )
        if has_surface_ao_signature and not surface_ao_geometry_signature:
            raise ValueError("A materialized surface AO signature is empty.")
        if isinstance(self.surface_ao_intensity, bool):
            raise ValueError("AO intensity must be between 0 and 1.")
        try:
            surface_ao_intensity = float(self.surface_ao_intensity)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("AO intensity must be between 0 and 1.") from error
        if (
            not math.isfinite(surface_ao_intensity)
            or not 0.0 <= surface_ao_intensity <= 1.0
        ):
            raise ValueError("AO intensity must be between 0 and 1.")
        object.__setattr__(self, "atlas", copy.deepcopy(self.atlas))
        object.__setattr__(
            self,
            "map_paths",
            MappingProxyType(normalized_paths),
        )
        object.__setattr__(self, "active_map_types", active_map_types)
        object.__setattr__(self, "surface_ao_image_path", surface_ao_image_path)
        object.__setattr__(
            self,
            "surface_ao_geometry_signature",
            surface_ao_geometry_signature,
        )
        object.__setattr__(self, "surface_ao_intensity", surface_ao_intensity)


@dataclass(frozen=True)
class AtlasDrawCallEstimate:
    """One-pass primitive counts produced by the Atlas export layout."""

    atlas_batch_count: int = 0
    half_primitive_count: int = 0
    glass_primitive_count: int = 0
    passthrough_primitive_count: int = 0

    def __post_init__(self) -> None:
        values = (
            self.atlas_batch_count,
            self.half_primitive_count,
            self.glass_primitive_count,
            self.passthrough_primitive_count,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            raise ValueError("Draw-call estimate counts must be non-negative integers.")

    @property
    def exported_count(self) -> int:
        """Return the authored GLB primitive count."""

        return (
            self.atlas_batch_count
            + self.half_primitive_count
            + self.glass_primitive_count
            + self.passthrough_primitive_count
        )

    @property
    def mirrored_runtime_count(self) -> int:
        """Include one expected R3F mirror for every authored half primitive."""

        return self.exported_count + self.half_primitive_count


@dataclass(frozen=True)
class SurfaceAmbientOcclusionAtlasContext:
    """Lightweight Atlas binding data used by background AO preparation.

    Unlike :class:`MaterializedTextureAtlas`, this snapshot does not require or
    rebuild any Atlas PNG. Atlas AO needs placement precedence, resolution, and
    active-map metadata. Cached AO metadata is optional and lets the 3D preview
    reconstruct UV1 without materializing the UV0 Atlas family. The class and
    field names remain unchanged for project and integration compatibility.
    """

    atlas: TextureAtlasRecord
    active_map_types: frozenset[str]
    surface_ao_intensity: float = 1.0
    surface_ao_image_path: Path | None = None
    surface_ao_geometry_signature: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.atlas, TextureAtlasRecord):
            raise TypeError("A surface AO Atlas context requires Atlas data.")
        if isinstance(self.active_map_types, (str, bytes, bytearray)):
            raise TypeError("Surface AO active maps must be provided as a collection.")
        try:
            active_map_types = frozenset(
                str(map_type).strip().lower() for map_type in self.active_map_types
            )
        except TypeError as error:
            raise TypeError(
                "Surface AO active maps must be provided as a collection."
            ) from error
        unknown_map_types = active_map_types - set(ATLAS_MAP_TYPES)
        if unknown_map_types:
            raise ValueError(
                "A surface AO Atlas context has unknown active maps: "
                + ", ".join(sorted(unknown_map_types))
            )
        if ATLAS_MAP_BASE_COLOR not in active_map_types:
            raise ValueError("A surface AO Atlas context requires base color.")
        if isinstance(self.surface_ao_intensity, bool):
            raise ValueError("AO intensity must be between 0 and 1.")
        try:
            intensity = float(self.surface_ao_intensity)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("AO intensity must be between 0 and 1.") from error
        if not math.isfinite(intensity) or not 0.0 <= intensity <= 1.0:
            raise ValueError("AO intensity must be between 0 and 1.")
        surface_ao_image_path = (
            None
            if self.surface_ao_image_path is None
            else Path(self.surface_ao_image_path)
        )
        has_surface_ao_image = surface_ao_image_path is not None
        has_surface_ao_signature = self.surface_ao_geometry_signature is not None
        if has_surface_ao_image != has_surface_ao_signature:
            raise ValueError(
                "A surface AO preview image and signature must be set together."
            )
        if surface_ao_image_path is not None and not surface_ao_image_path.is_file():
            raise ValueError("A surface AO preview image is missing.")
        surface_ao_geometry_signature = (
            None
            if self.surface_ao_geometry_signature is None
            else str(self.surface_ao_geometry_signature).strip().lower()
        )
        if has_surface_ao_signature and not surface_ao_geometry_signature:
            raise ValueError("A surface AO preview signature is empty.")
        object.__setattr__(self, "atlas", copy.deepcopy(self.atlas))
        object.__setattr__(self, "active_map_types", active_map_types)
        object.__setattr__(self, "surface_ao_intensity", intensity)
        object.__setattr__(self, "surface_ao_image_path", surface_ao_image_path)
        object.__setattr__(
            self,
            "surface_ao_geometry_signature",
            surface_ao_geometry_signature,
        )


@dataclass(frozen=True)
class SurfaceAmbientOcclusionBakeResult:
    """One raw Atlas AO plane and the scene signature which produced it."""

    atlas_id: str
    ambient_occlusion: np.ndarray
    geometry_signature: str
    preview_model: GeneratedModel | None = None

    def __post_init__(self) -> None:
        atlas_id = str(self.atlas_id).strip()
        geometry_signature = str(self.geometry_signature).strip()
        pixels = np.asarray(self.ambient_occlusion)
        if not atlas_id:
            raise ValueError("A surface AO bake requires an Atlas ID.")
        if not geometry_signature:
            raise ValueError("A surface AO bake requires a geometry signature.")
        if pixels.ndim != 2 or pixels.dtype != np.uint8 or not pixels.size:
            raise ValueError("A surface AO bake requires grayscale uint8 pixels.")
        if self.preview_model is not None and not isinstance(
            self.preview_model,
            GeneratedModel,
        ):
            raise TypeError("A surface AO preview must be a GeneratedModel.")
        object.__setattr__(self, "atlas_id", atlas_id)
        object.__setattr__(
            self,
            "ambient_occlusion",
            np.ascontiguousarray(pixels.copy()),
        )
        object.__setattr__(self, "geometry_signature", geometry_signature)


@dataclass(frozen=True)
class _HalfModelContext:
    """One half-model marker retained through Atlas rebuilding."""

    source_marker_name: object
    export_mesh_name: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class _HalfModelPart:
    """One authored material fragment carrying a half-model marker."""

    fragment: trimesh.Trimesh
    world_transform: np.ndarray
    metadata: dict[str, object]
    atlas_id: str | None = None


@dataclass(frozen=True)
class _PendingAtlasPart:
    """One Atlas-bound fragment before UV0 tiling and destination remapping."""

    receiver_key: str
    fragment: trimesh.Trimesh
    world_transform: np.ndarray
    source_id: str
    is_surface: bool
    placement: TextureAtlasPlacement
    half_context: _HalfModelContext | None
    node_metadata: dict[str, object]
    mirror_plane: (
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
        ]
        | None
    ) = None


# ### Public export helpers ###
def estimate_texture_atlas_draw_calls(
    model: GeneratedModel,
    atlases: Sequence[TextureAtlasRecord],
    *,
    surface_source_ids: Mapping[str, str] | None = None,
) -> AtlasDrawCallEstimate:
    """Estimate the exporter output without remapping UVs or loading images.

    Ordinary opaque geometry is flattened to one primitive per used Atlas.
    Half meshes, glass, and non-Atlas passthrough groups remain independent,
    matching :func:`apply_texture_atlases_to_export` routing.
    """

    if not isinstance(model, GeneratedModel):
        raise TypeError("Draw-call estimation requires a GeneratedModel.")
    if not isinstance(model.scene, trimesh.Scene):
        raise TypeError("Draw-call estimation requires a triangle-mesh scene.")
    if isinstance(atlases, (str, bytes, bytearray)):
        raise TypeError("Draw-call estimation requires texture Atlas records.")
    normalized_atlases = tuple(atlases)
    if not all(isinstance(atlas, TextureAtlasRecord) for atlas in normalized_atlases):
        raise TypeError("Draw-call estimation requires texture Atlas records.")
    normalized_surface_sources = {
        str(surface_id): str(source_id)
        for surface_id, source_id in (surface_source_ids or {}).items()
    }
    atlas_id_by_source_id: dict[str, str] = {}
    for atlas in normalized_atlases:
        for placement in atlas.placements:
            atlas_id_by_source_id.setdefault(placement.object_id, atlas.atlas_id)

    _half_contexts, half_context_by_node = _collect_half_model_contexts(model.scene)
    used_atlas_ids: set[str] = set()
    half_primitive_count = 0
    glass_primitive_count = 0
    passthrough_primitive_count = 0
    for node_name in sorted(model.scene.graph.nodes_geometry, key=str):
        _transform, geometry_name = model.scene.graph.get(node_name)
        geometry = model.scene.geometry.get(geometry_name)
        if not isinstance(geometry, trimesh.Trimesh) or not len(geometry.faces):
            continue
        source_id, _is_surface = _resolve_geometry_source_id(
            geometry,
            normalized_surface_sources,
        )
        atlas_id = None if source_id is None else atlas_id_by_source_id.get(source_id)
        is_half = node_name in half_context_by_node
        for _face_indices, material in _iter_face_material_groups(geometry):
            if is_half:
                half_primitive_count += 1
            elif is_housemaker_glass_material(material):
                glass_primitive_count += 1
            elif atlas_id is not None:
                used_atlas_ids.add(atlas_id)
            else:
                passthrough_primitive_count += 1

    return AtlasDrawCallEstimate(
        atlas_batch_count=len(used_atlas_ids),
        half_primitive_count=half_primitive_count,
        glass_primitive_count=glass_primitive_count,
        passthrough_primitive_count=passthrough_primitive_count,
    )


def apply_texture_atlases_to_export(
    model: GeneratedModel,
    atlases: Sequence[MaterializedTextureAtlas],
    *,
    surface_source_ids: Mapping[str, str] | None = None,
) -> GeneratedModel:
    """Apply all texture Atlases and any valid cached Atlas-AO bakes."""

    exported, _surface_ao_bakes = _apply_texture_atlases_to_export(
        model,
        atlases,
        surface_source_ids=surface_source_ids,
        surface_ao_bake_atlas_ids=frozenset(),
        surface_ao_preview_atlas_ids=frozenset(),
    )
    return exported


def bake_surface_ambient_occlusion_for_atlas(
    model: GeneratedModel,
    atlas: MaterializedTextureAtlas | SurfaceAmbientOcclusionAtlasContext,
    *,
    atlas_context: Sequence[
        MaterializedTextureAtlas | SurfaceAmbientOcclusionAtlasContext
    ]
    | None = None,
    surface_source_ids: Mapping[str, str] | None = None,
    cancellation_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> SurfaceAmbientOcclusionBakeResult:
    """Bake one Atlas AO plane with the full export binding precedence.

    ``atlas_context`` must use the same ordering as the eventual export. This
    matters when a source is present in more than one Atlas because export
    deliberately binds that source to the first Atlas. Cached AO on context
    Atlases is ignored while producing the requested independent bake.
    """

    if not isinstance(
        atlas,
        (MaterializedTextureAtlas, SurfaceAmbientOcclusionAtlasContext),
    ):
        raise TypeError("Surface AO baking requires an Atlas context.")
    atlas_id = atlas.atlas.atlas_id
    bake_context = _build_surface_ao_bake_context(
        atlas,
        atlas_context,
    )
    _exported, bakes = _apply_texture_atlases_to_export(
        model,
        bake_context,
        surface_source_ids=surface_source_ids,
        surface_ao_bake_atlas_ids=frozenset((atlas_id,)),
        surface_ao_preview_atlas_ids=frozenset(),
        cancellation_check=cancellation_check,
        progress_callback=progress_callback,
        serialize_output=False,
    )
    bake = bakes.get(atlas_id)
    if bake is None:
        raise ValueError(
            "The selected Atlas contains no geometry used by the exported scene."
        )
    return bake


def prepare_surface_ambient_occlusion_preview_for_atlas(
    model: GeneratedModel,
    atlas: MaterializedTextureAtlas | SurfaceAmbientOcclusionAtlasContext,
    *,
    atlas_context: Sequence[
        MaterializedTextureAtlas | SurfaceAmbientOcclusionAtlasContext
    ]
    | None = None,
    surface_source_ids: Mapping[str, str] | None = None,
    cancellation_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> GeneratedModel:
    """Prepare an exact AO-only model from one cached Atlas bake.

    The helper rebuilds and validates the deterministic UV1 layout but does
    not ray trace or materialize any base-color/PBR Atlas images. Callers
    should run it outside the GUI thread and cache the returned preview for
    the lifetime of the matching scene and AO geometry signature.
    """

    if not isinstance(
        atlas,
        (MaterializedTextureAtlas, SurfaceAmbientOcclusionAtlasContext),
    ):
        raise TypeError("Surface AO preview preparation requires an Atlas context.")
    if (
        atlas.surface_ao_image_path is None
        or atlas.surface_ao_geometry_signature is None
    ):
        raise ValueError("The selected Atlas has no ambient-occlusion bake.")
    atlas_id = atlas.atlas.atlas_id
    preview_context = _build_surface_ao_preview_context(
        atlas,
        atlas_context,
    )
    _unchanged, results = _apply_texture_atlases_to_export(
        model,
        preview_context,
        surface_source_ids=surface_source_ids,
        surface_ao_bake_atlas_ids=frozenset(),
        surface_ao_preview_atlas_ids=frozenset((atlas_id,)),
        cancellation_check=cancellation_check,
        progress_callback=progress_callback,
        serialize_output=False,
    )
    result = results.get(atlas_id)
    if result is None or result.preview_model is None:
        raise ValueError(
            "The selected Atlas contains no ambient occlusion to preview."
        )
    return result.preview_model


def _build_surface_ao_bake_context(
    target: MaterializedTextureAtlas | SurfaceAmbientOcclusionAtlasContext,
    atlas_context: Sequence[
        MaterializedTextureAtlas | SurfaceAmbientOcclusionAtlasContext
    ]
    | None,
) -> tuple[SurfaceAmbientOcclusionAtlasContext, ...]:
    """Return ordered, AO-free Atlas inputs with the target snapshot inserted."""

    context = (target,) if atlas_context is None else tuple(atlas_context)
    if not context:
        raise ValueError("Surface AO baking requires a non-empty Atlas context.")
    if not all(
        isinstance(
            item,
            (MaterializedTextureAtlas, SurfaceAmbientOcclusionAtlasContext),
        )
        for item in context
    ):
        raise TypeError("Surface AO baking received an invalid Atlas context.")
    context_ids = tuple(item.atlas.atlas_id for item in context)
    if len(context_ids) != len(set(context_ids)):
        raise ValueError("Surface AO baking requires unique Atlas IDs.")
    target_id = target.atlas.atlas_id
    if target_id not in context_ids:
        raise ValueError("The surface AO target is missing from the Atlas context.")

    result: list[SurfaceAmbientOcclusionAtlasContext] = []
    for context_item in context:
        item = target if context_item.atlas.atlas_id == target_id else context_item
        result.append(
            SurfaceAmbientOcclusionAtlasContext(
                atlas=item.atlas,
                active_map_types=item.active_map_types,
                surface_ao_intensity=item.surface_ao_intensity,
            )
        )
    return tuple(result)


def _build_surface_ao_preview_context(
    target: MaterializedTextureAtlas | SurfaceAmbientOcclusionAtlasContext,
    atlas_context: Sequence[
        MaterializedTextureAtlas | SurfaceAmbientOcclusionAtlasContext
    ]
    | None,
) -> tuple[SurfaceAmbientOcclusionAtlasContext, ...]:
    """Return ordered inputs retaining cached AO only on the preview target."""

    context = _build_surface_ao_bake_context(target, atlas_context)
    target_id = target.atlas.atlas_id
    return tuple(
        SurfaceAmbientOcclusionAtlasContext(
            atlas=item.atlas,
            active_map_types=item.active_map_types,
            surface_ao_intensity=item.surface_ao_intensity,
            surface_ao_image_path=(
                target.surface_ao_image_path
                if item.atlas.atlas_id == target_id
                else None
            ),
            surface_ao_geometry_signature=(
                target.surface_ao_geometry_signature
                if item.atlas.atlas_id == target_id
                else None
            ),
        )
        for item in context
    )


def _apply_texture_atlases_to_export(
    model: GeneratedModel,
    atlases: Sequence[MaterializedTextureAtlas | SurfaceAmbientOcclusionAtlasContext],
    *,
    surface_source_ids: Mapping[str, str] | None,
    surface_ao_bake_atlas_ids: frozenset[str],
    surface_ao_preview_atlas_ids: frozenset[str],
    cancellation_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[str], None] | None = None,
    serialize_output: bool = True,
) -> tuple[GeneratedModel, dict[str, SurfaceAmbientOcclusionBakeResult]]:
    """Remap Atlas-bound UVs and batch opaque geometry per shared material.

    The interactive preview fields remain unchanged. Ordinary opaque geometry
    is flattened and batched, while marked symmetric half-models remain
    separate mesh nodes with their authored geometry only.
    """

    if not isinstance(model, GeneratedModel):
        raise TypeError("Texture Atlas export requires a GeneratedModel.")
    normalized_atlases = tuple(atlases)
    valid_atlas_types = (
        (MaterializedTextureAtlas, SurfaceAmbientOcclusionAtlasContext)
        if not serialize_output
        else (MaterializedTextureAtlas,)
    )
    if not all(isinstance(atlas, valid_atlas_types) for atlas in normalized_atlases):
        raise TypeError("Texture Atlas export inputs are invalid.")
    if not normalized_atlases:
        return model, {}
    if not isinstance(model.scene, trimesh.Scene):
        raise TypeError("Texture Atlas export requires a triangle-mesh scene.")

    bindings = _build_source_bindings(normalized_atlases)
    normalized_surface_sources = {
        str(surface_id): str(source_id)
        for surface_id, source_id in (surface_source_ids or {}).items()
    }
    output_scene = trimesh.Scene()
    half_contexts, half_context_by_node = _collect_half_model_contexts(model.scene)
    half_parts_by_marker: dict[object, list[_HalfModelPart]] = {
        marker_name: [] for marker_name in half_contexts
    }
    atlas_parts: dict[str, list[trimesh.Trimesh]] = {
        item.atlas.atlas_id: [] for item in normalized_atlases
    }
    atlas_source_ids: dict[str, set[str]] = {
        item.atlas.atlas_id: set() for item in normalized_atlases
    }
    materialized_by_id = {item.atlas.atlas_id: item for item in normalized_atlases}
    atlas_materials: dict[str, PBRMaterial] = {}
    shared_glass_materials: dict[str, object] = {}
    object_ao_targets: dict[str, list[ObjectAmbientOcclusionTarget]] = {
        item.atlas.atlas_id: [] for item in normalized_atlases
    }
    pending_atlas_parts: dict[str, list[_PendingAtlasPart]] = {
        item.atlas.atlas_id: [] for item in normalized_atlases
    }
    opaque_occluders: list[trimesh.Trimesh] = []
    object_ao_source_ids = {
        placement.object_id
        for item in normalized_atlases
        for placement in item.atlas.placements
    }
    has_object_ao_receiver = _scene_contains_object_ao_receiver(
        model.scene,
        object_ao_source_ids,
        normalized_surface_sources,
    )
    materialized_atlas_ids = frozenset(materialized_by_id)
    unknown_request_ids = (
        surface_ao_bake_atlas_ids | surface_ao_preview_atlas_ids
    ) - materialized_atlas_ids
    if unknown_request_ids:
        raise ValueError("A surface AO request references an unknown texture Atlas.")
    cached_surface_ao_atlas_ids = frozenset(
        atlas_id
        for atlas_id, item in materialized_by_id.items()
        if item.surface_ao_image_path is not None
    )
    surface_ao_atlas_ids = (
        cached_surface_ao_atlas_ids
        | surface_ao_bake_atlas_ids
        | surface_ao_preview_atlas_ids
    )
    missing_preview_bakes = surface_ao_preview_atlas_ids - cached_surface_ao_atlas_ids
    if missing_preview_bakes:
        raise ValueError("A surface AO preview requires an existing cached bake.")
    has_any_ao_receiver = bool(surface_ao_atlas_ids) or has_object_ao_receiver
    occupied_names: set[str] = {str(output_scene.graph.base_frame)}
    passthrough_index = 0

    for node_name in sorted(model.scene.graph.nodes_geometry, key=str):
        transform, geometry_name = model.scene.graph.get(node_name)
        geometry = model.scene.geometry.get(geometry_name)
        if not isinstance(geometry, trimesh.Trimesh) or not len(geometry.faces):
            continue
        world_transform = _normalize_transform(transform)
        source_id, is_surface = _resolve_geometry_source_id(
            geometry,
            normalized_surface_sources,
        )
        binding = bindings.get(source_id) if source_id is not None else None
        half_context = half_context_by_node.get(node_name)
        node_metadata = _get_scene_node_metadata(model.scene, node_name)

        for material_group_index, (face_indices, material) in enumerate(
            _iter_face_material_groups(geometry)
        ):
            fragment = _build_face_fragment(geometry, face_indices, material)
            is_glass = is_housemaker_glass_material(material)
            mirror_plane = None
            if has_any_ao_receiver and not is_glass:
                world_occluder = _build_world_occluder(
                    fragment,
                    world_transform,
                )
                opaque_occluders.append(world_occluder)
                if half_context is not None:
                    mirror_plane = _resolve_half_mirror_plane(half_context.metadata)
                    mirrored_occluder = _build_mirrored_half_occluder(
                        world_occluder,
                        mirror_plane,
                    )
                    opaque_occluders.append(mirrored_occluder)
            if binding is not None and not is_glass:
                atlas_item, placement = binding
                atlas_id = atlas_item.atlas.atlas_id
                assert source_id is not None
                pending_atlas_parts[atlas_id].append(
                    _PendingAtlasPart(
                        receiver_key=(
                            f"{atlas_id}:{node_name!s}:{material_group_index}:"
                            f"{len(pending_atlas_parts[atlas_id])}"
                        ),
                        fragment=fragment,
                        world_transform=world_transform,
                        source_id=source_id,
                        is_surface=is_surface,
                        placement=placement,
                        half_context=half_context,
                        node_metadata=node_metadata,
                        mirror_plane=mirror_plane,
                    )
                )
                continue

            if is_glass:
                _apply_shared_glass_material(
                    fragment,
                    material,
                    shared_glass_materials,
                )
            if half_context is not None:
                half_parts_by_marker[half_context.source_marker_name].append(
                    _HalfModelPart(
                        fragment=fragment,
                        world_transform=world_transform,
                        metadata=node_metadata,
                    )
                )
                continue

            fragment.apply_transform(world_transform)
            passthrough_index += 1
            name = _reserve_name(
                f"{node_name}_part_{passthrough_index}",
                occupied_names,
            )
            output_scene.add_geometry(
                fragment,
                geom_name=name,
                node_name=name,
            )

    active_surface_ao_atlas_ids = frozenset(
        atlas_id
        for atlas_id in surface_ao_atlas_ids
        if pending_atlas_parts[atlas_id]
    )
    missing_requested_surface_ids = (
        surface_ao_bake_atlas_ids | surface_ao_preview_atlas_ids
    ) - active_surface_ao_atlas_ids
    if missing_requested_surface_ids:
        raise ValueError(
            "Ambient occlusion requires at least one placed object or assigned "
            "surface in the selected Atlas."
        )
    surface_ao_atlas_ids = active_surface_ao_atlas_ids
    (
        surface_ao_fragments,
        surface_ambient_occlusion_by_atlas,
        surface_ao_bakes,
    ) = _prepare_surface_ambient_occlusion_atlases(
        pending_atlas_parts,
        materialized_by_id,
        surface_ao_atlas_ids,
        surface_ao_bake_atlas_ids,
        surface_ao_preview_atlas_ids,
        tuple(opaque_occluders),
        cancellation_check=cancellation_check,
        progress_callback=progress_callback,
        expand_fragments=serialize_output,
    )
    if not serialize_output:
        return model, surface_ao_bakes

    for atlas_id, pending_parts in pending_atlas_parts.items():
        for pending in pending_parts:
            fragment = surface_ao_fragments.get(
                pending.receiver_key,
                pending.fragment,
            )
            try:
                remapped = _remap_fragment_to_atlas(
                    fragment,
                    pending.placement,
                    materialized_by_id[atlas_id].atlas.resolution,
                    repeat_source_uvs=pending.is_surface,
                )
            except ValueError as error:
                raise ValueError(
                    f"Atlas source {pending.source_id!r} cannot share its "
                    f"material: {error}"
                ) from error
            _route_bound_atlas_fragment(
                pending,
                remapped,
                atlas_id=atlas_id,
                uses_surface_ao=atlas_id in surface_ao_atlas_ids,
                half_parts_by_marker=half_parts_by_marker,
                atlas_parts=atlas_parts,
                atlas_source_ids=atlas_source_ids,
                object_ao_targets=object_ao_targets,
            )

    active_object_ao_targets = {
        atlas_id: tuple(targets)
        for atlas_id, targets in object_ao_targets.items()
        if targets
    }
    ambient_occlusion_by_atlas = (
        bake_placed_object_ambient_occlusion(
            active_object_ao_targets,
            {
                atlas_id: materialized_by_id[atlas_id].atlas.resolution
                for atlas_id in active_object_ao_targets
            },
            tuple(opaque_occluders),
        )
        if active_object_ao_targets
        else {}
    )
    ambient_occlusion_by_atlas.update(surface_ambient_occlusion_by_atlas)

    half_model_count = 0
    for marker_name, context in half_contexts.items():
        parts = half_parts_by_marker[marker_name]
        if not parts:
            continue
        _append_half_model_meshes_to_scene(
            output_scene,
            context,
            parts,
            occupied_names,
            materialized_by_id,
            atlas_materials,
            ambient_occlusion_by_atlas,
        )
        half_model_count += 1

    batched_count = 0
    for atlas_id, parts in atlas_parts.items():
        if not parts:
            continue
        atlas_item = materialized_by_id[atlas_id]
        material = _get_atlas_material(
            atlas_item,
            atlas_materials,
            ambient_occlusion_by_atlas,
        )
        combined = _combine_textured_parts(parts, material)
        combined.metadata[ATLAS_ID_METADATA_KEY] = atlas_id
        combined.metadata["housemaker_atlas_source_ids"] = sorted(
            atlas_source_ids[atlas_id]
        )
        name = _reserve_name(
            f"housemaker_atlas_{atlas_id}",
            occupied_names,
        )
        output_scene.add_geometry(
            combined,
            geom_name=name,
            node_name=name,
        )
        batched_count += 1

    if batched_count == 0 and half_model_count == 0:
        return model, surface_ao_bakes
    packed_orm_material_names: dict[str, str | PackedOrmMaterialSpec] = {}
    atlas_texture_material_names: dict[str, str] = {}
    ao_only_material_names: dict[str, str] = {}
    renamed_materials: list[tuple[PBRMaterial, str]] = []
    ambient_occlusion_materials: list[tuple[PBRMaterial, bool]] = []
    occupied_material_names = _collect_scene_material_names(output_scene)
    for atlas_id in sorted(atlas_materials):
        material = atlas_materials[atlas_id]
        final_name = materialized_by_id[atlas_id].atlas.name
        has_ambient_occlusion = atlas_id in ambient_occlusion_by_atlas
        marker_name = _reserve_name(
            (
                f"{PACKED_ORM_MATERIAL_MARKER_PREFIX}{atlas_id}"
                if has_ambient_occlusion
                else f"{ATLAS_SAMPLER_MATERIAL_MARKER_PREFIX}{atlas_id}"
            ),
            occupied_material_names,
        )
        material.name = marker_name
        atlas_item = materialized_by_id[atlas_id]
        has_metallic_roughness = bool(
            atlas_item.active_map_types & {PBR_MAP_ROUGHNESS, PBR_MAP_METALLIC}
        )
        renamed_materials.append((material, final_name))
        serialized_name = final_name
        if has_ambient_occlusion and not has_metallic_roughness:
            serialized_name = _reserve_name(
                f"{AO_ONLY_MATERIAL_MARKER_PREFIX}{atlas_id}",
                occupied_material_names,
            )
            ao_only_material_names[serialized_name] = final_name
        atlas_texture_material_names[marker_name] = serialized_name
        if has_ambient_occlusion:
            uses_surface_ao = atlas_id in surface_ao_atlas_ids
            packed_orm_material_names[marker_name] = PackedOrmMaterialSpec(
                final_name=serialized_name,
                ao_tex_coord=1 if uses_surface_ao else 0,
                ao_strength=(
                    atlas_item.surface_ao_intensity if uses_surface_ao else 1.0
                ),
            )
            ambient_occlusion_materials.append((material, has_metallic_roughness))
    try:
        exported = _serialize_scene_glb_with_half_mesh_extras(
            output_scene,
            failure_message="The texture Atlas scene could not be exported.",
            packed_orm_material_names=packed_orm_material_names,
            atlas_texture_material_names=atlas_texture_material_names,
        )
        exported = _finalize_ao_only_glb_materials(
            exported,
            ao_only_material_names,
        )
    finally:
        for material, final_name in renamed_materials:
            material.name = final_name
    for material, has_metallic_roughness in ambient_occlusion_materials:
        material.occlusionTexture = material.metallicRoughnessTexture
        if not has_metallic_roughness:
            material.metallicRoughnessTexture = None
    return (
        replace(
            model,
            scene=output_scene,
            glb_bytes=exported,
        ),
        surface_ao_bakes,
    )


# ### Half-model hierarchy helpers ###
def _collect_half_model_contexts(
    scene: trimesh.Scene,
) -> tuple[
    dict[object, _HalfModelContext],
    dict[object, _HalfModelContext],
]:
    """Resolve marked roots and their descendant geometry nodes."""

    contexts: dict[object, _HalfModelContext] = {}
    for node_name in scene.graph.nodes:
        metadata = _get_scene_node_metadata(scene, node_name)
        if not isinstance(metadata.get(HALF_MESH_EXTRAS_KEY), Mapping):
            continue
        source_name = str(node_name)
        export_name = (
            source_name
            if source_name.startswith(HALF_NODE_NAME_PREFIX)
            else f"{HALF_NODE_NAME_PREFIX}{source_name}"
        )
        contexts[node_name] = _HalfModelContext(
            source_marker_name=node_name,
            export_mesh_name=export_name,
            metadata=metadata,
        )

    context_by_geometry_node: dict[object, _HalfModelContext] = {}
    parents = scene.graph.transforms.parents
    base_frame = scene.graph.base_frame
    for node_name in scene.graph.nodes_geometry:
        current_name = node_name
        visited: set[object] = set()
        while current_name != base_frame and current_name not in visited:
            visited.add(current_name)
            context = contexts.get(current_name)
            if context is not None:
                context_by_geometry_node[node_name] = context
                break
            parent_name = parents.get(current_name)
            if parent_name is None:
                break
            current_name = parent_name
    return contexts, context_by_geometry_node


def _get_scene_node_metadata(
    scene: trimesh.Scene,
    node_name: object,
) -> dict[str, object]:
    """Copy metadata from the incoming edge that becomes glTF node extras."""

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


def _resolve_half_mirror_plane(
    node_metadata: Mapping[str, object],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Read one validated world-space mirror plane for AO processing."""

    raw_half_mesh = node_metadata.get(HALF_MESH_EXTRAS_KEY)
    if not isinstance(raw_half_mesh, Mapping):
        raise TypeError("A half-model AO mirror plane is missing.")
    raw_plane = raw_half_mesh.get("mirrorPlane")
    if not isinstance(raw_plane, Mapping):
        raise TypeError("A half-model AO mirror plane is invalid.")
    point = np.asarray(raw_plane.get("point"), dtype=float)
    normal = np.asarray(raw_plane.get("normal"), dtype=float)
    if (
        point.shape != (3,)
        or normal.shape != (3,)
        or not np.all(np.isfinite(point))
        or not np.all(np.isfinite(normal))
    ):
        raise ValueError("A half-model AO mirror plane is invalid.")
    normal_length = float(np.linalg.norm(normal))
    if normal_length <= GEOMETRY_EPSILON:
        raise ValueError("A half-model AO mirror plane is invalid.")
    normal /= normal_length
    return (
        tuple(float(value) for value in point),
        tuple(float(value) for value in normal),
    )


def _build_mirrored_half_occluder(
    world_fragment: trimesh.Trimesh,
    mirror_plane: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
    ],
) -> trimesh.Trimesh:
    """Build the runtime mirror as ray-only geometry for symmetric AO."""

    point = np.asarray(mirror_plane[0], dtype=float)
    normal = np.asarray(mirror_plane[1], dtype=float)
    vertices = np.asarray(world_fragment.vertices, dtype=float)
    signed_distances = (vertices - point) @ normal
    mirrored_vertices = vertices - 2.0 * signed_distances[:, np.newaxis] * normal
    mirrored_faces = np.asarray(world_fragment.faces, dtype=np.int64)[:, (0, 2, 1)]
    return trimesh.Trimesh(
        vertices=np.ascontiguousarray(mirrored_vertices),
        faces=np.ascontiguousarray(mirrored_faces),
        process=False,
    )


def _build_world_occluder(
    fragment: trimesh.Trimesh,
    world_transform: np.ndarray,
) -> trimesh.Trimesh:
    """Copy only geometry needed by Embree, without texture payloads."""

    occluder = trimesh.Trimesh(
        vertices=np.asarray(fragment.vertices, dtype=float).copy(),
        faces=np.asarray(fragment.faces, dtype=np.int64).copy(),
        process=False,
    )
    occluder.apply_transform(world_transform)
    return occluder


def _append_half_model_meshes_to_scene(
    output_scene: trimesh.Scene,
    context: _HalfModelContext,
    parts: Sequence[_HalfModelPart],
    occupied_names: set[str],
    materialized_by_id: Mapping[str, MaterializedTextureAtlas],
    material_cache: dict[str, PBRMaterial],
    ambient_occlusion_by_atlas: Mapping[str, np.ndarray],
) -> None:
    """Emit marked authored meshes directly, without an empty parent node."""

    for part_index, part in enumerate(parts, start=1):
        if part.atlas_id is not None:
            atlas_item = materialized_by_id[part.atlas_id]
            material = _get_atlas_material(
                atlas_item,
                material_cache,
                ambient_occlusion_by_atlas,
            )
            part.fragment.visual = TextureVisuals(
                uv=_optional_valid_uv(part.fragment),
                material=material,
            )
        child_metadata = copy.deepcopy(part.metadata)
        child_metadata.pop(HALF_MESH_EXTRAS_KEY, None)
        child_metadata.update(copy.deepcopy(context.metadata))
        preferred_name = (
            context.export_mesh_name
            if part_index == 1
            else f"{context.export_mesh_name}_{part_index}"
        )
        mesh_name = _reserve_name(
            preferred_name,
            occupied_names,
        )
        output_scene.add_geometry(
            part.fragment,
            geom_name=mesh_name,
            node_name=mesh_name,
            transform=_normalize_transform(part.world_transform),
            metadata=child_metadata or None,
        )


# ### Binding helpers ###
def _build_source_bindings(
    atlases: Sequence[MaterializedTextureAtlas | SurfaceAmbientOcclusionAtlasContext],
) -> dict[
    str,
    tuple[
        MaterializedTextureAtlas | SurfaceAmbientOcclusionAtlasContext,
        TextureAtlasPlacement,
    ],
]:
    """Choose the first Atlas deterministically when a source is duplicated."""

    bindings: dict[
        str,
        tuple[
            MaterializedTextureAtlas | SurfaceAmbientOcclusionAtlasContext,
            TextureAtlasPlacement,
        ],
    ] = {}
    for atlas_item in atlases:
        for placement in atlas_item.atlas.placements:
            bindings.setdefault(
                placement.object_id,
                (atlas_item, placement),
            )
    return bindings


def _resolve_geometry_source_id(
    geometry: trimesh.Trimesh,
    surface_source_ids: Mapping[str, str],
) -> tuple[str | None, bool]:
    metadata = getattr(geometry, "metadata", {})
    if not isinstance(metadata, Mapping):
        return None, False
    surface_id = metadata.get(SURFACE_ID_METADATA_KEY)
    if surface_id is not None:
        source_id = surface_source_ids.get(str(surface_id))
        if source_id is not None:
            return source_id, True
    object_id = metadata.get(OBJECT_ID_METADATA_KEY)
    if object_id is None:
        return None, False
    return str(object_id), False


def _scene_contains_object_ao_receiver(
    scene: trimesh.Scene,
    eligible_source_ids: set[str],
    surface_source_ids: Mapping[str, str],
) -> bool:
    """Return whether the exported scene can produce an object AO target."""

    if not eligible_source_ids:
        return False
    for geometry in scene.geometry.values():
        if not isinstance(geometry, trimesh.Trimesh) or not len(geometry.faces):
            continue
        source_id, is_surface = _resolve_geometry_source_id(
            geometry,
            surface_source_ids,
        )
        if not is_surface and source_id in eligible_source_ids:
            return True
    return False


# ### Atlas ambient-occlusion preparation ###
def _prepare_surface_ambient_occlusion_atlases(
    pending_parts_by_atlas: Mapping[str, Sequence[_PendingAtlasPart]],
    materialized_by_id: Mapping[
        str,
        MaterializedTextureAtlas | SurfaceAmbientOcclusionAtlasContext,
    ],
    surface_ao_atlas_ids: frozenset[str],
    bake_atlas_ids: frozenset[str],
    preview_atlas_ids: frozenset[str],
    opaque_occluders: Sequence[trimesh.Trimesh],
    *,
    cancellation_check: Callable[[], bool] | None,
    progress_callback: Callable[[str], None] | None,
    expand_fragments: bool,
) -> tuple[
    dict[str, trimesh.Trimesh],
    dict[str, np.ndarray],
    dict[str, SurfaceAmbientOcclusionBakeResult],
]:
    """Build UV1 for active Atlases and either bake or validate their AO."""

    expanded_fragments: dict[str, trimesh.Trimesh] = {}
    ambient_occlusion_by_atlas: dict[str, np.ndarray] = {}
    bake_results: dict[str, SurfaceAmbientOcclusionBakeResult] = {}
    for atlas_id in sorted(surface_ao_atlas_ids):
        pending_parts = tuple(pending_parts_by_atlas.get(atlas_id, ()))
        if not pending_parts:
            raise ValueError(
                "Ambient occlusion requires scene geometry in Atlas "
                f"{materialized_by_id[atlas_id].atlas.name!r}."
            )
        _report_surface_ao_progress(
            progress_callback,
            f"Unwrapping {materialized_by_id[atlas_id].atlas.name} AO (15%)",
        )
        receivers = tuple(_build_surface_ao_receiver(part) for part in pending_parts)
        layout = build_surface_ao_uv_layout(
            receivers,
            materialized_by_id[atlas_id].atlas.resolution,
            cancellation_check=cancellation_check,
        )
        atlas_item = materialized_by_id[atlas_id]
        if atlas_id in bake_atlas_ids:
            _report_surface_ao_progress(
                progress_callback,
                f"Ray tracing {atlas_item.atlas.name} AO (35%)",
            )
            baked = bake_surface_ambient_occlusion(
                layout,
                opaque_occluders,
                cancellation_check=cancellation_check,
            )
            ambient_occlusion = np.asarray(
                baked.ambient_occlusion,
                dtype=np.uint8,
            ).copy()
            signature = baked.geometry_signature
        else:
            signature = build_surface_ao_geometry_signature(
                layout,
                opaque_occluders,
                cancellation_check=cancellation_check,
            )
            if signature != atlas_item.surface_ao_geometry_signature:
                raise ValueError(
                    f"The ambient-occlusion bake for Atlas "
                    f"{atlas_item.atlas.name!r} is out of date. Bake ambient "
                    "occlusion again."
                )
            assert atlas_item.surface_ao_image_path is not None
            _raise_if_surface_ao_cancelled(cancellation_check)
            ambient_occlusion = _load_surface_ambient_occlusion(
                atlas_item.surface_ao_image_path,
                atlas_item.atlas.resolution,
            )
        ambient_occlusion_by_atlas[atlas_id] = ambient_occlusion
        if atlas_id in bake_atlas_ids or atlas_id in preview_atlas_ids:
            preview_model = _build_surface_ao_preview_model(
                layout,
                ambient_occlusion,
                cancellation_check=cancellation_check,
            )
            bake_results[atlas_id] = SurfaceAmbientOcclusionBakeResult(
                atlas_id=atlas_id,
                ambient_occlusion=ambient_occlusion,
                geometry_signature=signature,
                preview_model=preview_model,
            )
        if expand_fragments:
            layout_by_key = {receiver.key: receiver for receiver in layout.receivers}
            for part in pending_parts:
                receiver_layout = layout_by_key.get(part.receiver_key)
                if receiver_layout is None:
                    raise ValueError("A surface AO receiver layout is missing.")
                expanded_fragments[part.receiver_key] = _expand_fragment_for_surface_ao(
                    part.fragment,
                    receiver_layout,
                )
        _report_surface_ao_progress(
            progress_callback,
            f"Finalizing {atlas_item.atlas.name} AO (94%)",
        )
    return expanded_fragments, ambient_occlusion_by_atlas, bake_results


def _build_surface_ao_receiver(
    pending: _PendingAtlasPart,
) -> SurfaceAmbientOcclusionReceiver:
    """Build one world-space receiver while retaining its existing UV0."""

    world_mesh = pending.fragment.copy()
    world_mesh.apply_transform(pending.world_transform)
    mirror_point = None
    mirror_normal = None
    if pending.mirror_plane is not None:
        mirror_point, mirror_normal = pending.mirror_plane
    return SurfaceAmbientOcclusionReceiver(
        key=pending.receiver_key,
        mesh=world_mesh,
        mirror_plane_point=mirror_point,
        mirror_plane_normal=mirror_normal,
    )


def _expand_fragment_for_surface_ao(
    fragment: trimesh.Trimesh,
    receiver_layout: SurfaceAmbientOcclusionReceiverLayout,
) -> trimesh.Trimesh:
    """Apply xatlas seam duplication to local geometry and preserve UV0."""

    vertex_mapping = np.asarray(receiver_layout.vertex_mapping, dtype=np.int64)
    faces = np.asarray(receiver_layout.faces, dtype=np.int64)
    source_uv = _valid_uv(fragment)
    if source_uv is None:
        raise ValueError("A surface AO receiver has no valid source UVs.")
    expanded = trimesh.Trimesh(
        vertices=np.asarray(fragment.vertices, dtype=float)[vertex_mapping],
        faces=faces,
        vertex_normals=np.asarray(fragment.vertex_normals, dtype=float)[vertex_mapping],
        visual=TextureVisuals(
            uv=source_uv[vertex_mapping],
            material=None,
        ),
        metadata=copy.deepcopy(dict(getattr(fragment, "metadata", {}) or {})),
        process=False,
    )
    expanded.vertex_attributes[PACKED_ORM_AO_UV_ATTRIBUTE] = np.asarray(
        receiver_layout.gltf_uv1,
        dtype=np.float32,
    ).copy()
    return expanded


def _build_surface_ao_preview_model(
    layout: SurfaceAmbientOcclusionUvLayout,
    ambient_occlusion: np.ndarray,
    *,
    cancellation_check: Callable[[], bool] | None,
) -> GeneratedModel:
    """Build a viewer-only Z-up model using exact baked UV1 coordinates."""

    _raise_if_surface_ao_cancelled(cancellation_check)
    pixels = _normalize_ambient_occlusion(
        ambient_occlusion,
        (layout.resolution, layout.resolution),
    )
    material = PBRMaterial(
        name="Ambient occlusion preview",
        baseColorFactor=[255, 255, 255, 255],
        baseColorTexture=Image.fromarray(pixels.copy()),
        metallicFactor=0.0,
        roughnessFactor=1.0,
        doubleSided=True,
    )
    retained_meshes: list[trimesh.Trimesh] = []
    symmetric_previews: list[PreviewSymmetricObject] = []
    for receiver in layout.receivers:
        _raise_if_surface_ao_cancelled(cancellation_check)
        retained = _build_surface_ao_preview_receiver_mesh(
            receiver,
            material,
        )
        retained_meshes.append(retained)
        if receiver.mirror_plane_point is None:
            continue
        mirrored = _build_surface_ao_mirrored_preview_receiver_mesh(
            receiver,
            material,
        )
        symmetric_previews.append(
            PreviewSymmetricObject(
                object_id=f"surface-ao:{receiver.key}",
                meshes=(retained,),
                # Explicit mirrored meshes support arbitrary world-space
                # planes; these compatibility values are not used to reflect.
                orientation="vertical",
                plane_coordinate=0.0,
                mirrored_meshes=(mirrored,),
            )
        )
    if not retained_meshes:
        raise ValueError("Surface AO preview contains no receiver geometry.")
    combined = _combine_textured_parts(retained_meshes, material)
    scene = trimesh.Scene()
    scene.add_geometry(
        combined,
        geom_name="surface_ambient_occlusion_preview",
        node_name="surface_ambient_occlusion_preview",
    )
    _raise_if_surface_ao_cancelled(cancellation_check)
    return GeneratedModel(
        mesh=combined,
        scene=scene,
        glb_bytes=b"",
        preview_symmetric_objects=symmetric_previews,
    )


def _build_surface_ao_preview_receiver_mesh(
    receiver: SurfaceAmbientOcclusionReceiverLayout,
    material: PBRMaterial,
) -> trimesh.Trimesh:
    """Create one authored AO receiver in the viewer's Z-up coordinates."""

    mesh = trimesh.Trimesh(
        vertices=np.asarray(receiver.world_vertices, dtype=float),
        faces=np.asarray(receiver.faces, dtype=np.int64),
        visual=TextureVisuals(
            uv=np.asarray(receiver.uv1, dtype=float),
            material=material,
        ),
        process=False,
    )
    mesh.apply_transform(GLTF_Y_UP_TO_Z_UP_TRANSFORM)
    return mesh


def _build_surface_ao_mirrored_preview_receiver_mesh(
    receiver: SurfaceAmbientOcclusionReceiverLayout,
    material: PBRMaterial,
) -> trimesh.Trimesh:
    """Reflect one half receiver while reusing its exact AO UV1 values."""

    if receiver.mirror_plane_point is None or receiver.mirror_plane_normal is None:
        raise ValueError("A mirrored surface AO preview requires a mirror plane.")
    point = np.asarray(receiver.mirror_plane_point, dtype=float)
    normal = np.asarray(receiver.mirror_plane_normal, dtype=float)
    vertices = np.asarray(receiver.world_vertices, dtype=float)
    distances = (vertices - point) @ normal
    mirrored_vertices = vertices - 2.0 * distances[:, np.newaxis] * normal
    mirrored = trimesh.Trimesh(
        vertices=np.ascontiguousarray(mirrored_vertices),
        faces=np.asarray(receiver.faces, dtype=np.int64)[:, (0, 2, 1)],
        visual=TextureVisuals(
            uv=np.asarray(receiver.uv1, dtype=float),
            material=material,
        ),
        process=False,
    )
    mirrored.apply_transform(GLTF_Y_UP_TO_Z_UP_TRANSFORM)
    return mirrored


def _load_surface_ambient_occlusion(path: Path, resolution: int) -> np.ndarray:
    """Load a cached grayscale AO plane without changing its source file."""

    try:
        with Image.open(path) as image:
            image.load()
            pixels = np.asarray(image.convert("L"), dtype=np.uint8).copy()
    except (OSError, SyntaxError) as error:
        raise ValueError(f"Unable to read surface AO bake: {path}") from error
    if pixels.shape != (resolution, resolution):
        raise ValueError(
            "The surface AO bake dimensions do not match its texture Atlas."
        )
    return np.ascontiguousarray(pixels)


def _raise_if_surface_ao_cancelled(
    cancellation_check: Callable[[], bool] | None,
) -> None:
    """Stop expensive preview preparation at safe Python boundaries."""

    if cancellation_check is not None and cancellation_check():
        raise SurfaceAmbientOcclusionBakeCancelled(
            "Ambient-occlusion preparation was cancelled."
        )


def _report_surface_ao_progress(
    callback: Callable[[str], None] | None,
    stage: str,
) -> None:
    if callback is not None:
        callback(str(stage))


# ### Atlas-bound fragment routing ###
def _route_bound_atlas_fragment(
    pending: _PendingAtlasPart,
    remapped: trimesh.Trimesh,
    *,
    atlas_id: str,
    uses_surface_ao: bool,
    half_parts_by_marker: dict[object, list[_HalfModelPart]],
    atlas_parts: dict[str, list[trimesh.Trimesh]],
    atlas_source_ids: dict[str, set[str]],
    object_ao_targets: dict[str, list[ObjectAmbientOcclusionTarget]],
) -> None:
    """Route one remapped fragment and collect its object-AO receiver."""

    half_context = pending.half_context
    if half_context is not None:
        world_fragment = remapped.copy()
        world_fragment.apply_transform(pending.world_transform)
        if not uses_surface_ao and not pending.is_surface:
            if pending.mirror_plane is None:
                raise ValueError("A half-model AO mirror plane is missing.")
            mirror_point, mirror_normal = pending.mirror_plane
            object_ao_targets[atlas_id].append(
                ObjectAmbientOcclusionTarget(
                    mesh=world_fragment,
                    atlas_pixel_bounds=_placement_pixel_bounds(pending.placement),
                    mirror_plane_point=mirror_point,
                    mirror_plane_normal=mirror_normal,
                )
            )
        half_parts_by_marker[half_context.source_marker_name].append(
            _HalfModelPart(
                fragment=remapped,
                world_transform=pending.world_transform,
                metadata=pending.node_metadata,
                atlas_id=atlas_id,
            )
        )
        return

    remapped.apply_transform(pending.world_transform)
    if not uses_surface_ao and not pending.is_surface:
        object_ao_targets[atlas_id].append(
            ObjectAmbientOcclusionTarget(
                mesh=remapped,
                atlas_pixel_bounds=_placement_pixel_bounds(pending.placement),
            )
        )
    atlas_parts[atlas_id].append(remapped)
    atlas_source_ids[atlas_id].add(pending.source_id)


# ### Material helpers ###
def _get_atlas_material(
    atlas_item: MaterializedTextureAtlas,
    material_cache: dict[str, PBRMaterial],
    ambient_occlusion_by_atlas: Mapping[str, np.ndarray],
) -> PBRMaterial:
    """Build one shared material instance per exported Atlas."""

    atlas_id = atlas_item.atlas.atlas_id
    material = material_cache.get(atlas_id)
    if material is None:
        material = _build_atlas_material(
            atlas_item,
            ambient_occlusion=ambient_occlusion_by_atlas.get(atlas_id),
        )
        material_cache[atlas_id] = material
    return material


def _apply_shared_glass_material(
    fragment: trimesh.Trimesh,
    source_material: object,
    material_cache: dict[str, object],
) -> None:
    """Reuse the prefab glass instance without changing fragment geometry."""

    runtime_key = get_housemaker_glass_runtime_key(source_material)
    if runtime_key is None:
        return
    material = material_cache.setdefault(
        runtime_key,
        copy.deepcopy(source_material),
    )
    fragment.visual = TextureVisuals(
        uv=_optional_valid_uv(fragment),
        material=material,
    )


def _collect_scene_material_names(scene: trimesh.Scene) -> set[str]:
    """Collect user-visible names before reserving transient GLB markers."""

    result: set[str] = set()
    for geometry in scene.geometry.values():
        material = getattr(getattr(geometry, "visual", None), "material", None)
        materials = (
            tuple(material.materials)
            if isinstance(material, MultiMaterial)
            else (material,)
        )
        for leaf in materials:
            name = getattr(leaf, "name", None)
            if name is not None:
                result.add(str(name))
    return result


def _build_atlas_material(
    atlas_item: MaterializedTextureAtlas,
    *,
    ambient_occlusion: np.ndarray | None = None,
) -> PBRMaterial:
    active_map_types = atlas_item.active_map_types
    maps = {
        map_type: _load_rgba(atlas_item.map_paths[map_type])
        for map_type in ATLAS_MAP_TYPES
        if map_type in active_map_types
    }
    base = maps[ATLAS_MAP_BASE_COLOR]
    expected_shape = base.shape
    if any(texture.shape != expected_shape for texture in maps.values()):
        raise ValueError("Texture Atlas PBR maps must have identical dimensions.")
    expected_size = (atlas_item.atlas.resolution, atlas_item.atlas.resolution)
    if base.shape[:2] != expected_size:
        raise ValueError(
            "Texture Atlas map dimensions do not match the Atlas resolution."
        )

    normal_texture = (
        None
        if PBR_MAP_NORMAL not in active_map_types
        else Image.fromarray(maps[PBR_MAP_NORMAL], mode="RGBA")
    )
    has_roughness = PBR_MAP_ROUGHNESS in active_map_types
    has_metallic = PBR_MAP_METALLIC in active_map_types
    normalized_ambient_occlusion = _normalize_ambient_occlusion(
        ambient_occlusion,
        expected_size,
    )
    metallic_roughness_texture = None
    if has_roughness or has_metallic:
        metallic_roughness = np.empty_like(base)
        metallic_roughness[:, :, 0] = (
            255
            if normalized_ambient_occlusion is None
            else normalized_ambient_occlusion
        )
        metallic_roughness[:, :, 1] = (
            maps[PBR_MAP_ROUGHNESS][:, :, 0] if has_roughness else 255
        )
        metallic_roughness[:, :, 2] = (
            maps[PBR_MAP_METALLIC][:, :, 0] if has_metallic else 0
        )
        metallic_roughness[:, :, 3] = 255
        metallic_roughness_texture = Image.fromarray(
            metallic_roughness,
            mode="RGBA",
        )
    elif normalized_ambient_occlusion is not None:
        metallic_roughness_texture = Image.fromarray(
            normalized_ambient_occlusion,
            mode="L",
        )
    return PBRMaterial(
        name=atlas_item.atlas.name,
        baseColorFactor=[255, 255, 255, 255],
        baseColorTexture=Image.fromarray(base, mode="RGBA"),
        normalTexture=normal_texture,
        metallicRoughnessTexture=metallic_roughness_texture,
        metallicFactor=1.0 if has_metallic else 0.0,
        roughnessFactor=1.0,
        doubleSided=True,
    )


# ### AO-only GLB serialization helpers ###
def _finalize_ao_only_glb_materials(
    payload: bytes,
    material_names: Mapping[str, str],
) -> bytes:
    """Remove the temporary MR binding from AO-only glTF materials."""

    if not material_names:
        return payload
    try:
        if (
            len(payload) < GLB_HEADER_BYTE_COUNT + GLB_CHUNK_HEADER_BYTE_COUNT
            or payload[:4] != GLB_MAGIC
            or int.from_bytes(payload[4:8], "little") != GLB_VERSION
            or int.from_bytes(payload[8:12], "little") != len(payload)
        ):
            raise ValueError("The GLB header is invalid.")
        json_byte_count = int.from_bytes(payload[12:16], "little")
        if payload[16:20] != GLB_JSON_CHUNK_TYPE:
            raise ValueError("The first GLB chunk is not JSON.")
        json_end = GLB_HEADER_BYTE_COUNT + GLB_CHUNK_HEADER_BYTE_COUNT + json_byte_count
        if json_byte_count <= 0 or json_end > len(payload):
            raise ValueError("The GLB JSON chunk is invalid.")
        raw_document = payload[20:json_end].rstrip(b" \t\r\n\0")
        document = json.loads(raw_document.decode("utf-8"))
        if not isinstance(document, dict):
            raise ValueError("The GLB JSON root is invalid.")
        _remove_ao_only_metallic_roughness_bindings(
            document,
            material_names,
        )
        encoded_document = json.dumps(
            document,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, UnicodeError, ValueError) as error:
        raise ValueError("The texture Atlas scene could not be exported.") from error

    encoded_document += b" " * (-len(encoded_document) % 4)
    trailing_chunks = payload[json_end:]
    total_byte_count = (
        GLB_HEADER_BYTE_COUNT
        + GLB_CHUNK_HEADER_BYTE_COUNT
        + len(encoded_document)
        + len(trailing_chunks)
    )
    return b"".join(
        (
            GLB_MAGIC,
            GLB_VERSION.to_bytes(4, "little"),
            total_byte_count.to_bytes(4, "little"),
            len(encoded_document).to_bytes(4, "little"),
            GLB_JSON_CHUNK_TYPE,
            encoded_document,
            trailing_chunks,
        )
    )


def _remove_ao_only_metallic_roughness_bindings(
    document: dict[str, object],
    material_names: Mapping[str, str],
) -> None:
    """Keep AO-only images bound exclusively through ``occlusionTexture``."""

    materials = document.get("materials")
    textures = document.get("textures")
    if not isinstance(materials, list) or not isinstance(textures, list):
        raise ValueError("The AO-only GLB material data is missing.")
    found_names: set[str] = set()
    for material in materials:
        if not isinstance(material, dict):
            continue
        marker_name = material.get("name")
        if not isinstance(marker_name, str) or marker_name not in material_names:
            continue
        if marker_name in found_names:
            raise ValueError("AO-only material markers must be unique.")
        occlusion_texture = material.get("occlusionTexture")
        texture_index = (
            occlusion_texture.get("index")
            if isinstance(occlusion_texture, dict)
            else None
        )
        if (
            isinstance(texture_index, bool)
            or not isinstance(texture_index, int)
            or texture_index < 0
            or texture_index >= len(textures)
        ):
            raise ValueError("An AO-only material has no occlusion texture.")
        pbr = material.get("pbrMetallicRoughness")
        if not isinstance(pbr, dict):
            raise ValueError("An AO-only material has invalid PBR data.")
        pbr.pop("metallicRoughnessTexture", None)
        material["name"] = material_names[marker_name]
        found_names.add(marker_name)
    if found_names != set(material_names):
        raise ValueError("An AO-only material was not exported.")


def _normalize_ambient_occlusion(
    ambient_occlusion: np.ndarray | None,
    expected_size: tuple[int, int],
) -> np.ndarray | None:
    if ambient_occlusion is None:
        return None
    values = np.asarray(ambient_occlusion)
    if values.shape != expected_size or values.dtype != np.uint8:
        raise ValueError(
            "Ambient-occlusion dimensions must match the texture Atlas resolution."
        )
    return np.ascontiguousarray(values)


def _load_rgba(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            image.load()
            rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    except (OSError, SyntaxError) as error:
        raise ValueError(f"Unable to read texture Atlas map: {path}") from error
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("Texture Atlas maps must contain RGBA pixels.")
    return np.ascontiguousarray(rgba)


# ### Geometry splitting helpers ###
def _iter_face_material_groups(
    mesh: trimesh.Trimesh,
) -> tuple[tuple[np.ndarray, object | None], ...]:
    material = getattr(getattr(mesh, "visual", None), "material", None)
    if not isinstance(material, MultiMaterial):
        return ((np.arange(len(mesh.faces), dtype=np.int64), material),)

    materials = tuple(material.materials)
    if not materials:
        return ((np.arange(len(mesh.faces), dtype=np.int64), None),)
    raw_face_materials = getattr(mesh.visual, "face_materials", None)
    if raw_face_materials is None:
        return ((np.arange(len(mesh.faces), dtype=np.int64), materials[0]),)
    face_materials = np.asarray(raw_face_materials, dtype=np.int64)
    if face_materials.shape != (len(mesh.faces),):
        raise ValueError("A scene mesh has invalid face-material indices.")
    if np.any(face_materials < 0) or np.any(face_materials >= len(materials)):
        raise ValueError("A scene mesh references a missing material.")
    return tuple(
        (np.flatnonzero(face_materials == index), leaf)
        for index, leaf in enumerate(materials)
        if np.any(face_materials == index)
    )


def _build_face_fragment(
    mesh: trimesh.Trimesh,
    face_indices: np.ndarray,
    material: object | None,
) -> trimesh.Trimesh:
    selected_faces = np.asarray(mesh.faces, dtype=np.int64)[face_indices]
    referenced_vertices, inverse = np.unique(selected_faces, return_inverse=True)
    vertices = np.asarray(mesh.vertices, dtype=float)[referenced_vertices]
    faces = inverse.reshape((-1, 3))
    vertex_normals = np.asarray(mesh.vertex_normals, dtype=float)[referenced_vertices]
    uv = _valid_uv(mesh)
    visual: object | None = None
    if material is not None or uv is not None:
        visual = TextureVisuals(
            uv=None if uv is None else uv[referenced_vertices],
            material=copy.deepcopy(material),
        )
    kwargs: dict[str, object] = {}
    if visual is None:
        vertex_colors = np.asarray(
            getattr(mesh.visual, "vertex_colors", ()),
            dtype=np.uint8,
        )
        if vertex_colors.shape == (len(mesh.vertices), 4):
            kwargs["vertex_colors"] = vertex_colors[referenced_vertices]
    return trimesh.Trimesh(
        vertices=np.ascontiguousarray(vertices),
        faces=np.ascontiguousarray(faces),
        vertex_normals=np.ascontiguousarray(vertex_normals),
        visual=visual,
        metadata=copy.deepcopy(dict(getattr(mesh, "metadata", {}) or {})),
        process=False,
        **kwargs,
    )


def _valid_uv(mesh: trimesh.Trimesh) -> np.ndarray | None:
    raw_uv = getattr(getattr(mesh, "visual", None), "uv", None)
    if raw_uv is None:
        return None
    uv = np.asarray(raw_uv, dtype=float)
    if uv.shape != (len(mesh.vertices), 2) or not np.all(np.isfinite(uv)):
        return None
    return np.ascontiguousarray(uv)


def _optional_valid_uv(mesh: trimesh.Trimesh) -> np.ndarray | None:
    uv = _valid_uv(mesh)
    return None if uv is None else uv.copy()


def _valid_surface_ao_uv(mesh: trimesh.Trimesh) -> np.ndarray | None:
    """Return glTF-oriented UV1 stored in trimesh's custom attribute map."""

    raw_uv = mesh.vertex_attributes.get(PACKED_ORM_AO_UV_ATTRIBUTE)
    if raw_uv is None:
        return None
    uv = np.asarray(raw_uv, dtype=float)
    if (
        uv.shape != (len(mesh.vertices), 2)
        or not np.all(np.isfinite(uv))
        or np.any(uv < -UV_TOLERANCE)
        or np.any(uv > 1.0 + UV_TOLERANCE)
    ):
        raise ValueError("Surface AO UVs are invalid.")
    return np.ascontiguousarray(np.clip(uv, 0.0, 1.0))


# ### UV remapping helpers ###
def _remap_fragment_to_atlas(
    fragment: trimesh.Trimesh,
    placement: TextureAtlasPlacement,
    atlas_resolution: int,
    *,
    repeat_source_uvs: bool,
) -> trimesh.Trimesh:
    uv = _valid_uv(fragment)
    if uv is None:
        raise ValueError("Atlas-bound geometry requires valid UV coordinates.")
    if repeat_source_uvs:
        fragment = _split_repeating_uv_triangles(fragment)
        uv = _valid_uv(fragment)
        assert uv is not None
    else:
        _validate_source_uv_domain(uv, placement)
    mapped_uv = _map_uv_to_placement(
        uv,
        placement,
        atlas_resolution,
        repeat_source_uvs=repeat_source_uvs,
    )
    fragment.visual = TextureVisuals(uv=mapped_uv, material=None)
    return fragment


def _validate_source_uv_domain(
    uv: np.ndarray,
    placement: TextureAtlasPlacement,
) -> None:
    minimum = np.min(uv, axis=0)
    maximum = np.max(uv, axis=0)
    if placement.packing_mode == ATLAS_PACKING_MODE_FULL:
        lower = np.asarray((0.0, 0.0))
        upper = np.asarray((1.0, 1.0))
    elif placement.packing_mode == ATLAS_PACKING_MODE_SYMMETRIC_QUARTER:
        lower = np.asarray((0.0, 0.5))
        upper = np.asarray((0.5, 1.0))
    else:
        lower = np.asarray((0.0, 0.0))
        upper = np.asarray((0.5, 1.0))
    if np.any(minimum < lower - UV_TOLERANCE) or np.any(maximum > upper + UV_TOLERANCE):
        raise ValueError(
            "Atlas-bound object UVs extend outside their packed texture region."
        )


def _map_uv_to_placement(
    uv: np.ndarray,
    placement: TextureAtlasPlacement,
    atlas_resolution: int,
    *,
    repeat_source_uvs: bool = False,
) -> np.ndarray:
    """Map authored UVs into one placement's guarded inner rectangle.

    Repeating surfaces land on the inner pixel boundaries so their opposite
    wrapped guards meet continuously. Non-repeating objects remain inset to
    texel centers so their dilated edge colors behave like ordinary clamping.
    """

    (
        source_lower,
        source_upper,
        content_x,
        content_y,
        content_width,
        content_height,
    ) = _placement_content_region(placement)
    source_span = source_upper - source_lower
    normalized_u = (uv[:, 0] - source_lower[0]) / source_span[0]
    normalized_y_from_top = (source_upper[1] - uv[:, 1]) / source_span[1]
    if (
        np.any(normalized_u < -UV_TOLERANCE)
        or np.any(normalized_u > 1.0 + UV_TOLERANCE)
        or np.any(normalized_y_from_top < -UV_TOLERANCE)
        or np.any(normalized_y_from_top > 1.0 + UV_TOLERANCE)
    ):
        raise ValueError("Source UVs leave their packed texture region.")

    normalized_u = np.clip(normalized_u, 0.0, 1.0)
    normalized_y_from_top = np.clip(normalized_y_from_top, 0.0, 1.0)
    resolution = float(atlas_resolution)
    mapped = np.empty_like(uv, dtype=float)
    if repeat_source_uvs:
        mapped[:, 0] = (content_x + normalized_u * content_width) / resolution
        mapped[:, 1] = (
            1.0
            - (content_y + normalized_y_from_top * content_height) / resolution
        )
    else:
        mapped[:, 0] = (
            content_x + 0.5 + normalized_u * max(content_width - 1.0, 0.0)
        ) / resolution
        mapped[:, 1] = (
            1.0
            - (
                content_y
                + 0.5
                + normalized_y_from_top * max(content_height - 1.0, 0.0)
            )
            / resolution
        )
    if np.any(mapped < -UV_TOLERANCE) or np.any(mapped > 1.0 + UV_TOLERANCE):
        raise ValueError("Remapped texture Atlas UVs leave the Atlas bounds.")
    return np.ascontiguousarray(np.clip(mapped, 0.0, 1.0))


def _placement_content_region(
    placement: TextureAtlasPlacement,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """Return source UV bounds and guarded destination content bounds."""

    source_lower = np.asarray((0.0, 0.0), dtype=float)
    source_upper = np.asarray((1.0, 1.0), dtype=float)
    if placement.packing_mode in ATLAS_HALF_SLOT_PACKING_MODES:
        source_upper[0] = 0.5
    elif placement.packing_mode == ATLAS_PACKING_MODE_SYMMETRIC_QUARTER:
        source_upper[0] = 0.5
        source_lower[1] = 0.5
    _outer_bounds, inner_bounds = atlas_placement_pixel_regions(placement)
    content_x, content_y, content_right, content_bottom = (
        float(value) for value in inner_bounds
    )
    return (
        source_lower,
        source_upper,
        content_x,
        content_y,
        content_right - content_x,
        content_bottom - content_y,
    )


def _placement_pixel_bounds(
    placement: TextureAtlasPlacement,
) -> tuple[int, int, int, int]:
    """Return the exclusive Atlas-pixel bounds owned by one placement."""

    outer_bounds, _inner_bounds = atlas_placement_pixel_regions(placement)
    return outer_bounds


# ### Repeating-surface helpers ###
def _split_repeating_uv_triangles(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    uv = _valid_uv(mesh)
    if uv is None:
        raise ValueError("Repeating Atlas surfaces require valid UV coordinates.")
    vertices = np.asarray(mesh.vertices, dtype=float)
    normals = np.asarray(mesh.vertex_normals, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    surface_ao_uv = _valid_surface_ao_uv(mesh)
    output_vertices: list[np.ndarray] = []
    output_normals: list[np.ndarray] = []
    output_uvs: list[np.ndarray] = []
    output_surface_ao_uvs: list[np.ndarray] = []

    for face in faces:
        face_uv = uv[face]
        u_tiles = _covered_tile_indices(face_uv[:, 0])
        v_tiles = _covered_tile_indices(face_uv[:, 1])
        maximum_new_triangles = len(u_tiles) * len(v_tiles) * 4
        if len(output_vertices) // 3 + maximum_new_triangles > (
            MAX_TILED_SURFACE_TRIANGLES
        ):
            raise ValueError("A repeating Atlas surface produces too many tiles.")
        polygon_columns = (vertices[face], normals[face], face_uv)
        if surface_ao_uv is not None:
            polygon_columns = (*polygon_columns, surface_ao_uv[face])
        polygon = np.column_stack(polygon_columns)
        for tile_u in u_tiles:
            for tile_v in v_tiles:
                clipped = _clip_polygon_to_uv_tile(polygon, tile_u, tile_v)
                if len(clipped) < 3:
                    continue
                for index in range(1, len(clipped) - 1):
                    triangle = np.asarray(
                        (clipped[0], clipped[index], clipped[index + 1]),
                        dtype=float,
                    )
                    edge_a = triangle[1, :3] - triangle[0, :3]
                    edge_b = triangle[2, :3] - triangle[0, :3]
                    if np.linalg.norm(np.cross(edge_a, edge_b)) <= GEOMETRY_EPSILON:
                        continue
                    if len(output_vertices) // 3 >= MAX_TILED_SURFACE_TRIANGLES:
                        raise ValueError(
                            "A repeating Atlas surface produces too many tiles."
                        )
                    triangle_uv = triangle[:, 6:8] - np.asarray(
                        (float(tile_u), float(tile_v))
                    )
                    output_vertices.extend(triangle[:, :3])
                    output_normals.extend(_normalize_rows(triangle[:, 3:6]))
                    output_uvs.extend(np.clip(triangle_uv, 0.0, 1.0))
                    if surface_ao_uv is not None:
                        output_surface_ao_uvs.extend(
                            np.clip(triangle[:, 8:10], 0.0, 1.0)
                        )

    if not output_vertices:
        raise ValueError("A repeating Atlas surface contains no usable triangles.")
    output_faces = np.arange(len(output_vertices), dtype=np.int64).reshape((-1, 3))
    result = trimesh.Trimesh(
        vertices=np.asarray(output_vertices, dtype=float),
        faces=output_faces,
        vertex_normals=np.asarray(output_normals, dtype=float),
        visual=TextureVisuals(
            uv=np.asarray(output_uvs, dtype=float),
            material=None,
        ),
        metadata=copy.deepcopy(dict(getattr(mesh, "metadata", {}) or {})),
        process=False,
    )
    if surface_ao_uv is not None:
        result.vertex_attributes[PACKED_ORM_AO_UV_ATTRIBUTE] = np.asarray(
            output_surface_ao_uvs, dtype=np.float32
        )
    return result


def _covered_tile_indices(coordinates: np.ndarray) -> range:
    minimum = float(np.min(coordinates))
    maximum = float(np.max(coordinates))
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise ValueError("Repeating Atlas UVs must be finite.")
    first = math.floor(minimum + UV_TOLERANCE)
    last = math.floor(maximum - UV_TOLERANCE)
    if last < first:
        last = first
    return range(first, last + 1)


def _clip_polygon_to_uv_tile(
    polygon: np.ndarray,
    tile_u: int,
    tile_v: int,
) -> np.ndarray:
    clipped = np.asarray(polygon, dtype=float)
    for axis, boundary, keep_greater in (
        (6, float(tile_u), True),
        (6, float(tile_u + 1), False),
        (7, float(tile_v), True),
        (7, float(tile_v + 1), False),
    ):
        clipped = _clip_polygon_half_space(
            clipped,
            axis=axis,
            boundary=boundary,
            keep_greater=keep_greater,
        )
        if len(clipped) < 3:
            break
    return clipped


def _clip_polygon_half_space(
    polygon: np.ndarray,
    *,
    axis: int,
    boundary: float,
    keep_greater: bool,
) -> np.ndarray:
    if not len(polygon):
        return polygon

    def is_inside(point: np.ndarray) -> bool:
        if keep_greater:
            return bool(point[axis] >= boundary - UV_TOLERANCE)
        return bool(point[axis] <= boundary + UV_TOLERANCE)

    output: list[np.ndarray] = []
    previous = polygon[-1]
    previous_inside = is_inside(previous)
    for current in polygon:
        current_inside = is_inside(current)
        if current_inside != previous_inside:
            delta = current[axis] - previous[axis]
            if abs(float(delta)) > GEOMETRY_EPSILON:
                fraction = (boundary - previous[axis]) / delta
                output.append(previous + (current - previous) * fraction)
        if current_inside:
            output.append(current)
        previous = current
        previous_inside = current_inside
    if not output:
        return np.empty((0, polygon.shape[1]), dtype=float)
    return _remove_adjacent_duplicate_rows(np.asarray(output, dtype=float))


def _remove_adjacent_duplicate_rows(values: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return values
    retained = [values[0]]
    for value in values[1:]:
        if not np.allclose(value, retained[-1], rtol=0.0, atol=GEOMETRY_EPSILON):
            retained.append(value)
    if len(retained) > 1 and np.allclose(
        retained[0],
        retained[-1],
        rtol=0.0,
        atol=GEOMETRY_EPSILON,
    ):
        retained.pop()
    return np.asarray(retained, dtype=float)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(values, axis=1)
    normalized = np.asarray(values, dtype=float).copy()
    usable = lengths > GEOMETRY_EPSILON
    normalized[usable] /= lengths[usable, np.newaxis]
    normalized[~usable] = np.asarray((0.0, 0.0, 1.0))
    return normalized


# ### Scene assembly helpers ###
def _combine_textured_parts(
    parts: Sequence[trimesh.Trimesh],
    material: PBRMaterial,
) -> trimesh.Trimesh:
    vertices: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    uvs: list[np.ndarray] = []
    surface_ao_uvs: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    has_surface_ao_uv: bool | None = None
    vertex_offset = 0
    for part in parts:
        uv = _valid_uv(part)
        if uv is None:
            raise ValueError("An Atlas batch contains geometry without UVs.")
        surface_ao_uv = _valid_surface_ao_uv(part)
        part_has_surface_ao_uv = surface_ao_uv is not None
        if has_surface_ao_uv is None:
            has_surface_ao_uv = part_has_surface_ao_uv
        elif has_surface_ao_uv != part_has_surface_ao_uv:
            raise ValueError("An Atlas batch has incomplete surface AO UVs.")
        part_vertices = np.asarray(part.vertices, dtype=float)
        part_faces = np.asarray(part.faces, dtype=np.int64)
        vertices.append(part_vertices)
        normals.append(np.asarray(part.vertex_normals, dtype=float))
        uvs.append(uv)
        if surface_ao_uv is not None:
            surface_ao_uvs.append(surface_ao_uv)
        faces.append(part_faces + vertex_offset)
        vertex_offset += len(part_vertices)
    result = trimesh.Trimesh(
        vertices=np.vstack(vertices),
        faces=np.vstack(faces),
        vertex_normals=np.vstack(normals),
        visual=TextureVisuals(
            uv=np.vstack(uvs),
            material=material,
        ),
        process=False,
    )
    if has_surface_ao_uv:
        result.vertex_attributes[PACKED_ORM_AO_UV_ATTRIBUTE] = np.vstack(
            surface_ao_uvs
        ).astype(np.float32, copy=False)
    return result


def _normalize_transform(raw_transform: object) -> np.ndarray:
    transform = np.asarray(raw_transform, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("A scene node has an invalid transform.")
    return transform.copy()


def _reserve_name(preferred_name: object, occupied: set[str]) -> str:
    stem = str(preferred_name).strip() or "geometry"
    candidate = stem
    suffix = 2
    while candidate in occupied:
        candidate = f"{stem}_{suffix}"
        suffix += 1
    occupied.add(candidate)
    return candidate
