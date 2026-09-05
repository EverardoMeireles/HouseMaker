# ### Imports ###
from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
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
    HALF_MESH_EXTRAS_KEY,
    HALF_NODE_NAME_PREFIX,
    GeneratedModel,
    _serialize_scene_glb_with_half_mesh_extras,
)
from housemaker.pbr_maps import (
    ATLAS_MAP_BASE_COLOR,
    ATLAS_MAP_TYPES,
    PBR_MAP_METALLIC,
    PBR_MAP_NORMAL,
    PBR_MAP_ROUGHNESS,
)
from housemaker.texture_atlas_state import (
    ATLAS_HALF_SLOT_PACKING_MODES,
    ATLAS_PACKING_MODE_FULL,
    ATLAS_PACKING_MODE_SYMMETRIC_QUARTER,
    ATLAS_SLOT_HALF_RIGHT,
    ATLAS_SLOT_QUADRANT_BOTTOM_LEFT,
    ATLAS_SLOT_QUADRANT_BOTTOM_RIGHT,
    ATLAS_SLOT_QUADRANT_TOP_RIGHT,
    TextureAtlasPlacement,
    TextureAtlasRecord,
)


# ### Constants ###
SURFACE_ID_METADATA_KEY = "housemaker_surface_id"
OBJECT_ID_METADATA_KEY = "housemaker_object_id"
ATLAS_ID_METADATA_KEY = "housemaker_atlas_id"
UV_TOLERANCE = 1e-6
GEOMETRY_EPSILON = 1e-12
MAX_TILED_SURFACE_TRIANGLES = 2_000_000


# ### Public data models ###
@dataclass(frozen=True)
class MaterializedTextureAtlas:
    """One immutable Atlas and the genuine maps enabled for GLB export.

    Materialization still supplies neutral fallback PNGs for every map so the
    Atlas editor can switch views consistently. ``active_map_types`` keeps
    those fallbacks out of the exported material unless a source owns the map.
    """

    atlas: TextureAtlasRecord
    map_paths: Mapping[str, Path]
    active_map_types: frozenset[str] = frozenset(ATLAS_MAP_TYPES)

    def __post_init__(self) -> None:
        if not isinstance(self.atlas, TextureAtlasRecord):
            raise TypeError("A materialized texture Atlas requires Atlas data.")
        normalized_paths = {
            str(map_type): Path(path)
            for map_type, path in self.map_paths.items()
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
                str(map_type).strip().lower()
                for map_type in self.active_map_types
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
            raise ValueError(
                "A materialized texture Atlas requires active base color."
            )
        object.__setattr__(self, "atlas", copy.deepcopy(self.atlas))
        object.__setattr__(
            self,
            "map_paths",
            MappingProxyType(normalized_paths),
        )
        object.__setattr__(self, "active_map_types", active_map_types)


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


# ### Public export helpers ###
def apply_texture_atlases_to_export(
    model: GeneratedModel,
    atlases: Sequence[MaterializedTextureAtlas],
    *,
    surface_source_ids: Mapping[str, str] | None = None,
) -> GeneratedModel:
    """Remap Atlas-bound UVs and batch opaque geometry per shared material.

    The interactive preview fields remain unchanged. Ordinary opaque geometry
    is flattened and batched, while marked symmetric half-models remain
    separate mesh nodes with their authored geometry only.
    """

    if not isinstance(model, GeneratedModel):
        raise TypeError("Texture Atlas export requires a GeneratedModel.")
    normalized_atlases = tuple(atlases)
    if not all(
        isinstance(atlas, MaterializedTextureAtlas)
        for atlas in normalized_atlases
    ):
        raise TypeError("Texture Atlas export inputs are invalid.")
    if not normalized_atlases:
        return model
    if not isinstance(model.scene, trimesh.Scene):
        raise TypeError("Texture Atlas export requires a triangle-mesh scene.")

    bindings = _build_source_bindings(normalized_atlases)
    normalized_surface_sources = {
        str(surface_id): str(source_id)
        for surface_id, source_id in (surface_source_ids or {}).items()
    }
    output_scene = trimesh.Scene()
    half_contexts, half_context_by_node = _collect_half_model_contexts(
        model.scene
    )
    half_parts_by_marker: dict[object, list[_HalfModelPart]] = {
        marker_name: [] for marker_name in half_contexts
    }
    atlas_parts: dict[str, list[trimesh.Trimesh]] = {
        item.atlas.atlas_id: [] for item in normalized_atlases
    }
    atlas_source_ids: dict[str, set[str]] = {
        item.atlas.atlas_id: set() for item in normalized_atlases
    }
    materialized_by_id = {
        item.atlas.atlas_id: item for item in normalized_atlases
    }
    atlas_materials: dict[str, PBRMaterial] = {}
    shared_glass_materials: dict[str, object] = {}
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

        for face_indices, material in _iter_face_material_groups(geometry):
            fragment = _build_face_fragment(geometry, face_indices, material)
            if (
                binding is not None
                and not is_housemaker_glass_material(material)
            ):
                atlas_item, placement = binding
                try:
                    remapped = _remap_fragment_to_atlas(
                        fragment,
                        placement,
                        atlas_item.atlas.resolution,
                        repeat_source_uvs=is_surface,
                    )
                except ValueError as error:
                    raise ValueError(
                        f"Atlas source {source_id!r} cannot share its material: "
                        f"{error}"
                    ) from error
                if half_context is not None:
                    atlas_material = _get_atlas_material(
                        atlas_item,
                        atlas_materials,
                    )
                    remapped.visual = TextureVisuals(
                        uv=_optional_valid_uv(remapped),
                        material=atlas_material,
                    )
                    half_parts_by_marker[
                        half_context.source_marker_name
                    ].append(
                        _HalfModelPart(
                            fragment=remapped,
                            world_transform=world_transform,
                            metadata=node_metadata,
                        )
                    )
                    continue
                remapped.apply_transform(world_transform)
                atlas_parts[atlas_item.atlas.atlas_id].append(remapped)
                atlas_source_ids[atlas_item.atlas.atlas_id].add(source_id)
                continue

            if is_housemaker_glass_material(material):
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
        )
        half_model_count += 1

    batched_count = 0
    for atlas_id, parts in atlas_parts.items():
        if not parts:
            continue
        atlas_item = materialized_by_id[atlas_id]
        material = _get_atlas_material(atlas_item, atlas_materials)
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
        return model
    exported = _serialize_scene_glb_with_half_mesh_extras(
        output_scene,
        failure_message="The texture Atlas scene could not be exported.",
    )
    return replace(
        model,
        scene=output_scene,
        glb_bytes=exported,
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


def _append_half_model_meshes_to_scene(
    output_scene: trimesh.Scene,
    context: _HalfModelContext,
    parts: Sequence[_HalfModelPart],
    occupied_names: set[str],
) -> None:
    """Emit marked authored meshes directly, without an empty parent node."""

    for part_index, part in enumerate(parts, start=1):
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
    atlases: Sequence[MaterializedTextureAtlas],
) -> dict[
    str,
    tuple[MaterializedTextureAtlas, TextureAtlasPlacement],
]:
    """Choose the first Atlas deterministically when a source is duplicated."""

    bindings: dict[
        str,
        tuple[MaterializedTextureAtlas, TextureAtlasPlacement],
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


# ### Material helpers ###
def _get_atlas_material(
    atlas_item: MaterializedTextureAtlas,
    material_cache: dict[str, PBRMaterial],
) -> PBRMaterial:
    """Build one shared material instance per exported Atlas."""

    atlas_id = atlas_item.atlas.atlas_id
    material = material_cache.get(atlas_id)
    if material is None:
        material = _build_atlas_material(atlas_item)
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


def _build_atlas_material(atlas_item: MaterializedTextureAtlas) -> PBRMaterial:
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
    metallic_roughness_texture = None
    if has_roughness or has_metallic:
        metallic_roughness = np.empty_like(base)
        metallic_roughness[:, :, 0] = 255
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
    if np.any(minimum < lower - UV_TOLERANCE) or np.any(
        maximum > upper + UV_TOLERANCE
    ):
        raise ValueError(
            "Atlas-bound object UVs extend outside their packed texture region."
        )


def _map_uv_to_placement(
    uv: np.ndarray,
    placement: TextureAtlasPlacement,
    atlas_resolution: int,
) -> np.ndarray:
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
    normalized_y_from_top = (
        source_upper[1] - uv[:, 1]
    ) / source_span[1]
    if np.any(normalized_u < -UV_TOLERANCE) or np.any(
        normalized_u > 1.0 + UV_TOLERANCE
    ) or np.any(normalized_y_from_top < -UV_TOLERANCE) or np.any(
        normalized_y_from_top > 1.0 + UV_TOLERANCE
    ):
        raise ValueError("Source UVs leave their packed texture region.")

    normalized_u = np.clip(normalized_u, 0.0, 1.0)
    normalized_y_from_top = np.clip(normalized_y_from_top, 0.0, 1.0)
    resolution = float(atlas_resolution)
    mapped = np.empty_like(uv, dtype=float)
    mapped[:, 0] = (
        content_x + 0.5 + normalized_u * max(content_width - 1.0, 0.0)
    ) / resolution
    mapped[:, 1] = 1.0 - (
        content_y
        + 0.5
        + normalized_y_from_top * max(content_height - 1.0, 0.0)
    ) / resolution
    if np.any(mapped < -UV_TOLERANCE) or np.any(mapped > 1.0 + UV_TOLERANCE):
        raise ValueError("Remapped texture Atlas UVs leave the Atlas bounds.")
    return np.ascontiguousarray(np.clip(mapped, 0.0, 1.0))


def _placement_content_region(
    placement: TextureAtlasPlacement,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """Return source UV bounds and destination pixel bounds for one slot."""

    source_lower = np.asarray((0.0, 0.0), dtype=float)
    source_upper = np.asarray((1.0, 1.0), dtype=float)
    content_x = float(placement.x)
    content_y = float(placement.y)
    content_width = float(placement.size)
    content_height = float(placement.size)
    if placement.packing_mode in ATLAS_HALF_SLOT_PACKING_MODES:
        source_upper[0] = 0.5
        content_width /= 2.0
        if placement.slot_half == ATLAS_SLOT_HALF_RIGHT:
            content_x += content_width
    elif placement.packing_mode == ATLAS_PACKING_MODE_SYMMETRIC_QUARTER:
        source_upper[0] = 0.5
        source_lower[1] = 0.5
        content_width = float(placement.texture_resolution)
        content_height = float(placement.texture_resolution)
        if placement.slot_quadrant in {
            ATLAS_SLOT_QUADRANT_TOP_RIGHT,
            ATLAS_SLOT_QUADRANT_BOTTOM_RIGHT,
        }:
            content_x += content_width
        if placement.slot_quadrant in {
            ATLAS_SLOT_QUADRANT_BOTTOM_LEFT,
            ATLAS_SLOT_QUADRANT_BOTTOM_RIGHT,
        }:
            content_y += content_height
    return (
        source_lower,
        source_upper,
        content_x,
        content_y,
        content_width,
        content_height,
    )


# ### Repeating-surface helpers ###
def _split_repeating_uv_triangles(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    uv = _valid_uv(mesh)
    if uv is None:
        raise ValueError("Repeating Atlas surfaces require valid UV coordinates.")
    vertices = np.asarray(mesh.vertices, dtype=float)
    normals = np.asarray(mesh.vertex_normals, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    output_vertices: list[np.ndarray] = []
    output_normals: list[np.ndarray] = []
    output_uvs: list[np.ndarray] = []

    for face in faces:
        face_uv = uv[face]
        u_tiles = _covered_tile_indices(face_uv[:, 0])
        v_tiles = _covered_tile_indices(face_uv[:, 1])
        maximum_new_triangles = len(u_tiles) * len(v_tiles) * 4
        if len(output_vertices) // 3 + maximum_new_triangles > (
            MAX_TILED_SURFACE_TRIANGLES
        ):
            raise ValueError("A repeating Atlas surface produces too many tiles.")
        polygon = np.column_stack((vertices[face], normals[face], face_uv))
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

    if not output_vertices:
        raise ValueError("A repeating Atlas surface contains no usable triangles.")
    output_faces = np.arange(len(output_vertices), dtype=np.int64).reshape((-1, 3))
    return trimesh.Trimesh(
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
    faces: list[np.ndarray] = []
    vertex_offset = 0
    for part in parts:
        uv = _valid_uv(part)
        if uv is None:
            raise ValueError("An Atlas batch contains geometry without UVs.")
        part_vertices = np.asarray(part.vertices, dtype=float)
        part_faces = np.asarray(part.faces, dtype=np.int64)
        vertices.append(part_vertices)
        normals.append(np.asarray(part.vertex_normals, dtype=float))
        uvs.append(uv)
        faces.append(part_faces + vertex_offset)
        vertex_offset += len(part_vertices)
    return trimesh.Trimesh(
        vertices=np.vstack(vertices),
        faces=np.vstack(faces),
        vertex_normals=np.vstack(normals),
        visual=TextureVisuals(
            uv=np.vstack(uvs),
            material=material,
        ),
        process=False,
    )


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
