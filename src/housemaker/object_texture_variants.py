# ### Imports ###
from __future__ import annotations

import copy
import hashlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from io import BytesIO

import cv2
import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.glass_material import (
    is_housemaker_glass_material,
)
from housemaker.pbr_maps import (
    ATLAS_MAP_BASE_COLOR,
    ATLAS_MAP_LABELS as _ATLAS_MAP_LABELS,
    ATLAS_MAP_TYPES,
    PBR_MAP_METALLIC,
    PBR_MAP_NORMAL,
    PBR_MAP_ROUGHNESS,
    PBR_MAP_TYPES as _PBR_MAP_TYPES,
)
from housemaker.scan_projection_layout import remap_scan_projection_scene_uvs


# ### Shared-map compatibility exports ###
ATLAS_MAP_LABELS = _ATLAS_MAP_LABELS
PBR_MAP_TYPES = _PBR_MAP_TYPES


# ### Constants ###
TEXTURE_RESOLUTION_512 = 512
TEXTURE_RESOLUTION_1024 = 1024
TEXTURE_RESOLUTION_2048 = 2048
TEXTURE_RESOLUTIONS = (
    TEXTURE_RESOLUTION_512,
    TEXTURE_RESOLUTION_1024,
    TEXTURE_RESOLUTION_2048,
)
DEFAULT_TEXTURE_RESOLUTION = TEXTURE_RESOLUTION_1024
MATERIAL_TEXTURE_BASE_COLOR = ATLAS_MAP_BASE_COLOR
MATERIAL_TEXTURE_NORMAL = PBR_MAP_NORMAL
MATERIAL_TEXTURE_METALLIC_ROUGHNESS = "metallic_roughness"
MATERIAL_TEXTURE_TYPES = (
    MATERIAL_TEXTURE_BASE_COLOR,
    MATERIAL_TEXTURE_NORMAL,
    MATERIAL_TEXTURE_METALLIC_ROUGHNESS,
)
_MATERIAL_ATTRIBUTE_BY_TEXTURE_TYPE = {
    MATERIAL_TEXTURE_BASE_COLOR: "baseColorTexture",
    MATERIAL_TEXTURE_NORMAL: "normalTexture",
    MATERIAL_TEXTURE_METALLIC_ROUGHNESS: "metallicRoughnessTexture",
}
_OPAQUE_BLACK = np.asarray((0, 0, 0, 255), dtype=np.uint8)
_NEUTRAL_NORMAL = np.asarray((128, 128, 255, 255), dtype=np.uint8)
_NEUTRAL_METALLIC_ROUGHNESS = np.asarray((0, 255, 0, 255), dtype=np.uint8)


# ### Data models ###
@dataclass(frozen=True)
class ObjectTextureVariants:
    """Three exact-resolution GLBs sharing one geometry and material layout.

    Tagged scan-projection atlases use resolution-specific half-texel UV edge
    insets so linear filtering cannot cross into an unrelated packed cell.
    """

    glb_by_resolution: dict[int, bytes]
    texture_png_by_resolution: dict[int, bytes]
    preview_rgba_by_resolution: dict[int, np.ndarray]
    map_png_by_resolution: dict[str, dict[int, bytes]] | None = None
    map_preview_rgba_by_resolution: (
        dict[str, dict[int, np.ndarray]] | None
    ) = None

    def __post_init__(self) -> None:
        if set(self.glb_by_resolution) != set(TEXTURE_RESOLUTIONS):
            raise ValueError("Object texture variants require 512, 1024 and 2048 GLBs.")
        if set(self.preview_rgba_by_resolution) != set(TEXTURE_RESOLUTIONS):
            raise ValueError(
                "Object texture variants require 512, 1024 and 2048 previews."
            )
        if set(self.texture_png_by_resolution) != set(TEXTURE_RESOLUTIONS):
            raise ValueError(
                "Object texture variants require 512, 1024 and 2048 PNGs."
            )
        map_pngs = dict(self.map_png_by_resolution or {})
        map_previews = dict(self.map_preview_rgba_by_resolution or {})
        map_pngs.setdefault(
            ATLAS_MAP_BASE_COLOR,
            dict(self.texture_png_by_resolution),
        )
        map_previews.setdefault(
            ATLAS_MAP_BASE_COLOR,
            dict(self.preview_rgba_by_resolution),
        )
        if set(map_pngs) != set(map_previews):
            raise ValueError("Texture map PNGs and previews must match.")
        if any(map_type not in ATLAS_MAP_TYPES for map_type in map_pngs):
            raise ValueError("Object texture variants contain an unknown map.")
        for map_type in map_pngs:
            if set(map_pngs[map_type]) != set(TEXTURE_RESOLUTIONS):
                raise ValueError(
                    f"The {map_type} map requires every texture resolution."
                )
            if set(map_previews[map_type]) != set(TEXTURE_RESOLUTIONS):
                raise ValueError(
                    f"The {map_type} preview requires every resolution."
                )
        object.__setattr__(self, "map_png_by_resolution", map_pngs)
        object.__setattr__(
            self,
            "map_preview_rgba_by_resolution",
            map_previews,
        )

    @property
    def available_map_types(self) -> tuple[str, ...]:
        """Return Atlas-visible maps in canonical UI order."""

        assert self.map_png_by_resolution is not None
        return tuple(
            map_type
            for map_type in ATLAS_MAP_TYPES
            if map_type in self.map_png_by_resolution
        )


# ### Public variant generation ###
def build_object_texture_variants(
    glb_bytes: bytes,
) -> ObjectTextureVariants | None:
    """Normalize Meshy textures and export selectable 512/1024/2048 GLBs.

    Meshy's 2K atlas is the canonical 2048 variant. The 1024 and 512 variants
    are each reduced directly from that canonical image with area resampling.
    ``None`` means that the GLB contains no supported embedded base-color
    texture. Geometry, scene graph transforms and materials are retained.
    Ordinary UVs remain unchanged; tagged scan-projection UV bounds receive
    the half-texel inset required by each output resolution.
    """

    payload = bytes(glb_bytes)
    if not payload:
        raise ValueError("The generated GLB is empty.")
    scene = _load_glb_scene(payload)
    source_maps = _collect_material_texture_maps(scene)
    source_textures = source_maps.get(MATERIAL_TEXTURE_BASE_COLOR, [])
    if not source_textures:
        return None
    texture_maps_2048 = _validate_shared_2048_texture_maps(source_maps)
    return _build_variants_from_scene(
        scene,
        texture_maps_2048,
    )


def build_object_texture_variants_from_texture(
    glb_bytes: bytes,
    texture_png: bytes,
) -> ObjectTextureVariants:
    """Replace one shared 2048 atlas and build resolution-safe variants.

    Geometry stays fixed. Ordinary UVs remain unchanged, while tagged scan
    UV bounds receive the exact half-texel inset for each output resolution.
    """

    payload = bytes(glb_bytes)
    if not payload:
        raise ValueError("The generated GLB is empty.")
    scene = _load_glb_scene(payload)
    source_maps = _collect_material_texture_maps(scene)
    source_textures = source_maps.get(MATERIAL_TEXTURE_BASE_COLOR, [])
    if not source_textures:
        raise ValueError("The generated GLB has no embedded base-color texture.")
    _validate_shared_2048_texture(source_textures)
    replacement_texture = _decode_png_rgba(texture_png)
    expected_shape = (TEXTURE_RESOLUTION_2048, TEXTURE_RESOLUTION_2048)
    if replacement_texture.shape[:2] != expected_shape:
        raise ValueError("The replacement texture must be 2048 x 2048.")
    canonical_maps = _validate_shared_2048_texture_maps(source_maps)
    canonical_maps[MATERIAL_TEXTURE_BASE_COLOR] = replacement_texture
    return _build_variants_from_scene(scene, canonical_maps)


def replace_object_base_color_texture_from_glb(
    model_glb: bytes,
    texture_source_glb: bytes,
) -> bytes:
    """Apply provider texture maps without accepting replacement geometry.

    Texture-only operations keep the submitted model's scene, geometry and
    UVs authoritative. The shared 2048 base-color and supported PBR atlases
    are copied from the provider result. An untextured UV model receives a
    new material; a textured model keeps its existing material structure.
    """

    model_payload = bytes(model_glb)
    texture_payload = bytes(texture_source_glb)
    if not model_payload:
        raise ValueError("The preserved object GLB is empty.")
    if not texture_payload:
        raise ValueError("The generated texture GLB is empty.")

    model_scene = _load_glb_scene(model_payload)
    texture_scene = _load_glb_scene(texture_payload)
    generated_maps = _collect_material_texture_maps(texture_scene)
    generated_textures = generated_maps.get(MATERIAL_TEXTURE_BASE_COLOR, [])
    if not generated_textures:
        raise ValueError(
            "Meshy returned no embedded base-color texture."
        )
    generated_texture_maps = _validate_shared_2048_texture_maps(
        generated_maps
    )
    _attach_texture_maps_to_uv_meshes(model_scene, generated_texture_maps)
    try:
        return bytes(model_scene.export(file_type="glb"))
    except Exception as error:
        raise ValueError(
            "The generated texture could not be applied to the preserved "
            "object geometry."
        ) from error

def _build_variants_from_scene(
    scene: trimesh.Scene,
    texture_maps_2048: Mapping[str, np.ndarray],
) -> ObjectTextureVariants:
    """Build exact-resolution images and matching resolution-safe GLBs."""

    normalized_maps = {
        map_type: _normalize_rgba_array(texture).copy()
        for map_type, texture in texture_maps_2048.items()
    }
    if MATERIAL_TEXTURE_BASE_COLOR not in normalized_maps:
        raise ValueError("Object texture variants require a base-color map.")
    maps_by_resolution = {
        TEXTURE_RESOLUTION_2048: {
            map_type: texture.copy()
            for map_type, texture in normalized_maps.items()
        },
        TEXTURE_RESOLUTION_1024: {
            map_type: _resize_rgba(
                texture,
                TEXTURE_RESOLUTION_1024,
                cv2.INTER_AREA,
            )
            for map_type, texture in normalized_maps.items()
        },
        TEXTURE_RESOLUTION_512: {
            map_type: _resize_rgba(
                texture,
                TEXTURE_RESOLUTION_512,
                cv2.INTER_AREA,
            )
            for map_type, texture in normalized_maps.items()
        },
    }

    glb_by_resolution: dict[int, bytes] = {}
    for resolution in TEXTURE_RESOLUTIONS:
        variant_scene = copy.deepcopy(scene)
        remap_scan_projection_scene_uvs(variant_scene, resolution)
        _replace_material_texture_maps_with_shared(
            variant_scene,
            maps_by_resolution[resolution],
        )
        glb_by_resolution[resolution] = bytes(
            variant_scene.export(file_type="glb")
        )
    atlas_maps_by_resolution = {
        resolution: _split_atlas_texture_maps(texture_maps)
        for resolution, texture_maps in maps_by_resolution.items()
    }
    base_by_resolution = {
        resolution: atlas_maps[ATLAS_MAP_BASE_COLOR]
        for resolution, atlas_maps in atlas_maps_by_resolution.items()
    }
    map_types = tuple(
        map_type
        for map_type in ATLAS_MAP_TYPES
        if any(
            map_type in atlas_maps
            for atlas_maps in atlas_maps_by_resolution.values()
        )
    )
    return ObjectTextureVariants(
        glb_by_resolution=glb_by_resolution,
        texture_png_by_resolution={
            resolution: _encode_rgba_png(texture)
            for resolution, texture in base_by_resolution.items()
        },
        preview_rgba_by_resolution={
            resolution: texture.copy()
            for resolution, texture in base_by_resolution.items()
        },
        map_png_by_resolution={
            map_type: {
                resolution: _encode_rgba_png(
                    atlas_maps_by_resolution[resolution][map_type]
                )
                for resolution in TEXTURE_RESOLUTIONS
            }
            for map_type in map_types
        },
        map_preview_rgba_by_resolution={
            map_type: {
                resolution: atlas_maps_by_resolution[resolution][
                    map_type
                ].copy()
                for resolution in TEXTURE_RESOLUTIONS
            }
            for map_type in map_types
        },
    )


def _split_atlas_texture_maps(
    material_maps: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Expose glTF's packed material maps as separate Atlas images."""

    base_color = _normalize_rgba_array(
        material_maps[MATERIAL_TEXTURE_BASE_COLOR]
    )
    output = {ATLAS_MAP_BASE_COLOR: base_color.copy()}
    normal = material_maps.get(MATERIAL_TEXTURE_NORMAL)
    if normal is not None:
        output[PBR_MAP_NORMAL] = _normalize_rgba_array(normal).copy()
    metallic_roughness = material_maps.get(
        MATERIAL_TEXTURE_METALLIC_ROUGHNESS
    )
    if metallic_roughness is not None:
        packed = _normalize_rgba_array(metallic_roughness)
        roughness = packed[:, :, 1]
        metallic = packed[:, :, 2]
        output[PBR_MAP_ROUGHNESS] = _grayscale_rgba(roughness)
        output[PBR_MAP_METALLIC] = _grayscale_rgba(metallic)
    return output


def _grayscale_rgba(channel: np.ndarray) -> np.ndarray:
    normalized = np.asarray(channel, dtype=np.uint8)
    return np.ascontiguousarray(
        np.dstack(
            (
                normalized,
                normalized,
                normalized,
                np.full(normalized.shape, 255, dtype=np.uint8),
            )
        )
    )


def _validate_shared_2048_texture(
    source_textures: list[np.ndarray],
) -> np.ndarray:
    source_texture = _validate_shared_texture(source_textures)
    expected_source_shape = (
        TEXTURE_RESOLUTION_2048,
        TEXTURE_RESOLUTION_2048,
    )
    if source_texture.shape[:2] != expected_source_shape:
        raise ValueError(
            "Meshy returned a base-color texture that is not 2048 x 2048. "
            "HouseMaker will not upscale or stretch it."
        )
    return source_texture


def _validate_shared_texture(
    source_textures: list[np.ndarray],
) -> np.ndarray:
    if not source_textures:
        raise ValueError("The generated GLB has no base-color texture atlas.")
    if len({_texture_digest(texture) for texture in source_textures}) > 1:
        raise ValueError(
            "The generated object contains more than one distinct base-color "
            "texture atlas. Resolution variants currently require one shared "
            "atlas so UV-to-texture ownership remains unambiguous."
        )
    return source_textures[0]


# ### GLB material helpers ###
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


def _collect_material_textures(scene: trimesh.Scene) -> list[np.ndarray]:
    """Compatibility accessor for shared base-color material textures."""

    return _collect_material_texture_maps(scene).get(
        MATERIAL_TEXTURE_BASE_COLOR,
        [],
    )


def _collect_material_texture_maps(
    scene: trimesh.Scene,
) -> dict[str, list[np.ndarray]]:
    """Collect every supported embedded map in deterministic leaf order."""

    texture_maps: dict[str, list[np.ndarray]] = {
        map_type: [] for map_type in MATERIAL_TEXTURE_TYPES
    }
    textured_leaf_count = 0
    leaves_with_map: dict[str, int] = {
        map_type: 0 for map_type in MATERIAL_TEXTURE_TYPES
    }
    for _geometry_name, geometry in sorted(
        scene.geometry.items(),
        key=lambda item: str(item[0]),
    ):
        material = getattr(getattr(geometry, "visual", None), "material", None)
        for leaf in _iter_material_leaves(material):
            if is_housemaker_glass_material(leaf):
                continue
            base_texture = _get_material_texture(
                leaf,
                MATERIAL_TEXTURE_BASE_COLOR,
            )
            if base_texture is None:
                continue
            textured_leaf_count += 1
            for map_type in MATERIAL_TEXTURE_TYPES:
                texture = _get_material_texture(leaf, map_type)
                if texture is None:
                    continue
                texture_maps[map_type].append(_decode_texture_rgba(texture))
                leaves_with_map[map_type] += 1
    for map_type in MATERIAL_TEXTURE_TYPES:
        if leaves_with_map[map_type] not in {0, textured_leaf_count}:
            raise ValueError(
                f"The generated object has an inconsistent {map_type} map "
                "layout across its textured materials."
            )
    return {
        map_type: textures
        for map_type, textures in texture_maps.items()
        if textures
    }


def _iter_material_leaves(material: object) -> tuple[object, ...]:
    if material is None:
        return ()
    nested_materials = getattr(material, "materials", None)
    if isinstance(nested_materials, list | tuple):
        return tuple(
            leaf
            for nested_material in nested_materials
            for leaf in _iter_material_leaves(nested_material)
        )
    return (material,)


def prepare_uv_rewrite_material_textures(
    scene: trimesh.Scene,
    *,
    operation_name: str,
) -> None:
    """Remove unsafe optional maps before HouseMaker rewrites shared UVs.

    glTF commonly aliases occlusion to the red channel of the packed
    metallic-roughness image. That alias remains valid because HouseMaker
    rebakes the whole packed image. Independent occlusion, emissive, and
    legacy specular-glossiness images are not requested output maps, so they
    must not retain UVs that no longer address their pixels.
    """

    normalized_operation_name = str(operation_name).strip()
    if not normalized_operation_name:
        raise ValueError("A UV texture operation name is required.")
    for geometry in scene.geometry.values():
        material = getattr(getattr(geometry, "visual", None), "material", None)
        for leaf in _iter_material_leaves(material):
            if is_housemaker_glass_material(leaf):
                continue
            _prepare_uv_rewrite_material_leaf(leaf)


def _prepare_uv_rewrite_material_leaf(material: object) -> None:
    """Preserve a packed AO alias and clear UV-dependent optional maps."""

    metallic_roughness = getattr(material, "metallicRoughnessTexture", None)
    occlusion = getattr(material, "occlusionTexture", None)
    packed_occlusion = (
        metallic_roughness is not None
        and occlusion is not None
        and _material_textures_share_pixels(occlusion, metallic_roughness)
    )
    if hasattr(material, "occlusionTexture"):
        setattr(
            material,
            "occlusionTexture",
            metallic_roughness if packed_occlusion else None,
        )

    if getattr(material, "emissiveTexture", None) is not None:
        if hasattr(material, "emissiveTexture"):
            setattr(material, "emissiveTexture", None)
        if hasattr(material, "emissiveFactor"):
            setattr(material, "emissiveFactor", np.zeros(3, dtype=float))

    if hasattr(material, "specularGlossinessTexture"):
        setattr(material, "specularGlossinessTexture", None)


def _material_textures_share_pixels(first: object, second: object) -> bool:
    """Return whether two material slots address one equivalent image."""

    if first is second:
        return True
    try:
        first_rgba = _decode_texture_rgba(first)
        second_rgba = _decode_texture_rgba(second)
    except Exception:
        return False
    return first_rgba.shape == second_rgba.shape and np.array_equal(
        first_rgba,
        second_rgba,
    )


def _get_material_texture(
    material: object,
    map_type: str,
) -> object | None:
    if is_housemaker_glass_material(material):
        return None
    if map_type == MATERIAL_TEXTURE_BASE_COLOR:
        for attribute_name in ("baseColorTexture", "image"):
            texture = getattr(material, attribute_name, None)
            if texture is not None:
                return texture
        return None
    attribute_name = _MATERIAL_ATTRIBUTE_BY_TEXTURE_TYPE.get(map_type)
    if attribute_name is None:
        raise ValueError("Unknown material texture map type.")
    return getattr(material, attribute_name, None)


def _replace_material_textures(
    scene: trimesh.Scene,
    textures: list[np.ndarray],
) -> None:
    texture_iter = iter(textures)
    for _geometry_name, geometry in sorted(
        scene.geometry.items(),
        key=lambda item: str(item[0]),
    ):
        material = getattr(getattr(geometry, "visual", None), "material", None)
        _replace_material_texture_tree(material, texture_iter)
    try:
        next(texture_iter)
    except StopIteration:
        return
    raise ValueError("The GLB material texture layout changed during export.")


def _replace_material_texture_maps_with_shared(
    scene: trimesh.Scene,
    shared_maps: Mapping[str, np.ndarray],
) -> None:
    """Attach one shared UV atlas per map to every textured material leaf."""

    normalized_maps = {
        map_type: _normalize_rgba_array(texture)
        for map_type, texture in shared_maps.items()
    }
    if MATERIAL_TEXTURE_BASE_COLOR not in normalized_maps:
        raise ValueError("A shared material map set needs base color.")
    attached_count = 0
    for geometry in scene.geometry.values():
        if not isinstance(geometry, trimesh.Trimesh) or not len(geometry.faces):
            continue
        material = getattr(getattr(geometry, "visual", None), "material", None)
        leaves = _iter_material_leaves(material)
        if not leaves:
            continue
        for leaf in leaves:
            if is_housemaker_glass_material(leaf):
                continue
            if _get_material_texture(leaf, MATERIAL_TEXTURE_BASE_COLOR) is None:
                continue
            _attach_texture_maps_to_material_leaf(leaf, normalized_maps)
            attached_count += 1
    if attached_count == 0:
        raise ValueError("The GLB has no textured material leaves.")


def _validate_shared_2048_texture_maps(
    source_maps: Mapping[str, list[np.ndarray]],
) -> dict[str, np.ndarray]:
    """Validate one shared square atlas for each supported material map."""

    validated = _validate_shared_texture_maps(source_maps)
    base = validated[MATERIAL_TEXTURE_BASE_COLOR]
    if base.shape[:2] != (
        TEXTURE_RESOLUTION_2048,
        TEXTURE_RESOLUTION_2048,
    ):
        raise ValueError(
            "Meshy returned a base-color texture that is not 2048 x 2048. "
            "HouseMaker will not upscale or stretch it."
        )
    return validated


def _validate_shared_texture_maps(
    source_maps: Mapping[str, list[np.ndarray]],
) -> dict[str, np.ndarray]:
    """Validate one shared same-sized atlas for every supported map."""

    base_textures = source_maps.get(MATERIAL_TEXTURE_BASE_COLOR, [])
    base = _validate_shared_texture(base_textures).copy()
    validated = {MATERIAL_TEXTURE_BASE_COLOR: base}
    for map_type in (
        MATERIAL_TEXTURE_NORMAL,
        MATERIAL_TEXTURE_METALLIC_ROUGHNESS,
    ):
        textures = source_maps.get(map_type, [])
        if not textures:
            continue
        texture = _validate_shared_texture(textures).copy()
        if texture.shape != base.shape:
            raise ValueError(
                f"The {map_type} map must match the base-color atlas."
            )
        validated[map_type] = texture
    return validated


def _replace_material_texture_tree(
    material: object,
    textures: Iterator[np.ndarray],
) -> None:
    if material is None:
        return
    if is_housemaker_glass_material(material):
        return
    nested_materials = getattr(material, "materials", None)
    if isinstance(nested_materials, list | tuple):
        for nested_material in nested_materials:
            _replace_material_texture_tree(nested_material, textures)
        return
    for attribute_name in ("baseColorTexture", "image"):
        if getattr(material, attribute_name, None) is None:
            continue
        try:
            rgba = next(textures)
        except StopIteration as error:
            raise ValueError("Not enough replacement material textures.") from error
        setattr(material, attribute_name, Image.fromarray(rgba, mode="RGBA"))
        return


def _attach_texture_to_uv_meshes(
    scene: trimesh.Scene,
    texture_rgba: np.ndarray,
) -> None:
    """Attach one provider atlas to an untextured authoritative UV scene."""

    _attach_texture_maps_to_uv_meshes(
        scene,
        {MATERIAL_TEXTURE_BASE_COLOR: texture_rgba},
    )


def _attach_texture_maps_to_uv_meshes(
    scene: trimesh.Scene,
    texture_maps: Mapping[str, np.ndarray],
) -> None:
    """Attach provider maps to the authoritative UV scene and materials."""

    attached_count = 0
    for geometry in scene.geometry.values():
        if not isinstance(geometry, trimesh.Trimesh) or len(geometry.faces) == 0:
            continue
        uv = np.asarray(getattr(geometry.visual, "uv", None), dtype=float)
        if (
            uv.shape != (len(geometry.vertices), 2)
            or not np.all(np.isfinite(uv))
        ):
            raise ValueError(
                "The preserved object GLB has no valid UV coordinates for "
                "the generated texture."
            )
        raw_material = getattr(geometry.visual, "material", None)
        material = (
            PBRMaterial()
            if raw_material is None
            else copy.deepcopy(raw_material)
        )
        attachable_leaf_count = sum(
            not is_housemaker_glass_material(leaf)
            for leaf in _iter_material_leaves(material)
        )
        if attachable_leaf_count == 0:
            continue
        _attach_texture_maps_to_material_tree(material, texture_maps)
        raw_face_materials = getattr(
            geometry.visual,
            "face_materials",
            None,
        )
        face_materials = (
            None
            if raw_face_materials is None
            else np.asarray(raw_face_materials, dtype=np.int64).copy()
        )
        if face_materials is not None and face_materials.shape != (
            len(geometry.faces),
        ):
            raise ValueError(
                "The preserved object GLB has invalid face material indices."
            )
        geometry.visual = TextureVisuals(
            uv=uv.copy(),
            material=material,
            face_materials=face_materials,
        )
        attached_count += attachable_leaf_count
    if attached_count == 0:
        raise ValueError(
            "The preserved object GLB has no textured triangle meshes."
        )


def _attach_texture_to_material_tree(
    material: object,
    texture_rgba: np.ndarray,
) -> None:
    """Set one shared atlas on every leaf while retaining material factors."""

    if is_housemaker_glass_material(material):
        return
    nested_materials = getattr(material, "materials", None)
    if isinstance(nested_materials, list | tuple):
        if not nested_materials:
            raise ValueError("The preserved object has an empty material set.")
        for nested_material in nested_materials:
            _attach_texture_to_material_tree(nested_material, texture_rgba)
        return
    texture_image = Image.fromarray(texture_rgba.copy(), mode="RGBA")
    if hasattr(material, "baseColorTexture"):
        setattr(material, "baseColorTexture", texture_image)
        return
    if hasattr(material, "image"):
        setattr(material, "image", texture_image)
        return
    raise ValueError(
        "The preserved object uses an unsupported material type."
    )


def _attach_texture_maps_to_material_tree(
    material: object,
    texture_maps: Mapping[str, np.ndarray],
) -> None:
    """Set every supported map on each leaf while retaining material factors."""

    if is_housemaker_glass_material(material):
        return
    nested_materials = getattr(material, "materials", None)
    if isinstance(nested_materials, list | tuple):
        if not nested_materials:
            raise ValueError("The preserved object has an empty material set.")
        for nested_material in nested_materials:
            _attach_texture_maps_to_material_tree(nested_material, texture_maps)
        return
    _attach_texture_maps_to_material_leaf(material, texture_maps)


def _attach_texture_maps_to_material_leaf(
    material: object,
    texture_maps: Mapping[str, np.ndarray],
) -> None:
    if is_housemaker_glass_material(material):
        return
    packed_occlusion = (
        getattr(material, "occlusionTexture", None) is not None
        and getattr(material, "metallicRoughnessTexture", None) is not None
        and _material_textures_share_pixels(
            getattr(material, "occlusionTexture"),
            getattr(material, "metallicRoughnessTexture"),
        )
    )
    base_color = texture_maps.get(MATERIAL_TEXTURE_BASE_COLOR)
    if base_color is None:
        raise ValueError("Provider material maps require base color.")
    _attach_texture_to_material_tree(material, base_color)
    for map_type in (
        MATERIAL_TEXTURE_NORMAL,
        MATERIAL_TEXTURE_METALLIC_ROUGHNESS,
    ):
        attribute_name = _MATERIAL_ATTRIBUTE_BY_TEXTURE_TYPE[map_type]
        texture = texture_maps.get(map_type)
        if hasattr(material, attribute_name):
            setattr(
                material,
                attribute_name,
                None
                if texture is None
                else Image.fromarray(
                    _normalize_rgba_array(texture).copy(),
                    mode="RGBA",
                ),
            )
    if packed_occlusion and hasattr(material, "occlusionTexture"):
        setattr(
            material,
            "occlusionTexture",
            getattr(material, "metallicRoughnessTexture", None),
        )
    if MATERIAL_TEXTURE_METALLIC_ROUGHNESS in texture_maps:
        if hasattr(material, "metallicFactor"):
            setattr(material, "metallicFactor", 1.0)
        if hasattr(material, "roughnessFactor"):
            setattr(material, "roughnessFactor", 1.0)


# ### Image helpers ###
def _decode_texture_rgba(texture: object) -> np.ndarray:
    if hasattr(texture, "convert"):
        rgba = np.asarray(texture.convert("RGBA"), dtype=np.uint8)
    elif isinstance(texture, bytes | bytearray | memoryview):
        with Image.open(BytesIO(bytes(texture))) as image:
            rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    else:
        raw = np.asarray(texture)
        if raw.ndim == 1:
            with Image.open(BytesIO(raw.tobytes())) as image:
                rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        else:
            rgba = raw
    return _normalize_rgba_array(rgba)


def _decode_png_rgba(payload: bytes) -> np.ndarray:
    normalized = bytes(payload)
    if not normalized.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("The replacement texture must be a PNG image.")
    try:
        with Image.open(BytesIO(normalized)) as image:
            if str(image.format or "").upper() != "PNG":
                raise ValueError("The replacement texture must be a PNG image.")
            if int(getattr(image, "n_frames", 1)) != 1:
                raise ValueError("The replacement texture must be a static PNG.")
            if image.size != (
                TEXTURE_RESOLUTION_2048,
                TEXTURE_RESOLUTION_2048,
            ):
                raise ValueError("The replacement texture must be 2048 x 2048.")
            image.load()
            rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    except ValueError:
        raise
    except (OSError, SyntaxError) as error:
        raise ValueError("The replacement texture is not a valid PNG image.") from error
    return _normalize_rgba_array(rgba)


def _normalize_rgba_array(source: np.ndarray) -> np.ndarray:
    rgba = np.asarray(source)
    if rgba.dtype != np.uint8:
        if np.issubdtype(rgba.dtype, np.floating):
            rgba = np.nan_to_num(rgba, nan=0.0)
            if rgba.size and float(np.max(rgba)) <= 1.0:
                rgba = rgba * 255.0
        rgba = np.asarray(np.clip(rgba, 0, 255), dtype=np.uint8)
    if rgba.ndim == 2:
        rgba = np.repeat(rgba[:, :, np.newaxis], 3, axis=2)
    if rgba.ndim != 3 or rgba.shape[2] not in {3, 4}:
        raise ValueError("A material texture must be a grayscale, RGB or RGBA image.")
    if rgba.shape[2] == 3:
        rgba = np.dstack(
            (rgba, np.full(rgba.shape[:2], 255, dtype=np.uint8))
        )
    if rgba.shape[0] <= 0 or rgba.shape[1] <= 0:
        raise ValueError("A material texture cannot be empty.")
    return np.ascontiguousarray(rgba)


def _resize_rgba(
    source_rgba: np.ndarray,
    resolution: int,
    interpolation: int,
) -> np.ndarray:
    return np.ascontiguousarray(
        cv2.resize(
            _normalize_rgba_array(source_rgba),
            (int(resolution), int(resolution)),
            interpolation=interpolation,
        )
    )


def _encode_rgba_png(source_rgba: np.ndarray) -> bytes:
    bgra = cv2.cvtColor(_normalize_rgba_array(source_rgba), cv2.COLOR_RGBA2BGRA)
    did_encode, encoded = cv2.imencode(".png", bgra)
    if not did_encode:
        raise ValueError("A generated texture variant could not be encoded.")
    return bytes(encoded)


def _texture_digest(source_rgba: np.ndarray) -> str:
    rgba = _normalize_rgba_array(source_rgba)
    digest = hashlib.sha256()
    digest.update(np.asarray(rgba.shape, dtype=np.int64).tobytes())
    digest.update(rgba.tobytes())
    return digest.hexdigest()
