# ### Imports ###
from __future__ import annotations

import copy
import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from io import BytesIO

import cv2
import numpy as np
import trimesh
from PIL import Image


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


# ### Data models ###
@dataclass(frozen=True)
class ObjectTextureVariants:
    """Three GLBs that differ only in embedded base-color texture size."""

    glb_by_resolution: dict[int, bytes]
    texture_png_by_resolution: dict[int, bytes]
    preview_rgba_by_resolution: dict[int, np.ndarray]

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


# ### Public variant generation ###
def build_object_texture_variants(
    glb_bytes: bytes,
) -> ObjectTextureVariants | None:
    """Normalize Meshy textures and export selectable 512/1024/2048 GLBs.

    Meshy's 2K atlas is the canonical 2048 variant. The 1024 and 512 variants
    are each reduced directly from that canonical image with area resampling.
    ``None`` means that the GLB contains no supported embedded base-color
    texture. Geometry, scene graph transforms, materials and UV coordinates
    are retained; only the material images are replaced.
    """

    payload = bytes(glb_bytes)
    if not payload:
        raise ValueError("The generated GLB is empty.")
    scene = _load_glb_scene(payload)
    source_textures = _collect_material_textures(scene)
    if not source_textures:
        return None
    texture_2048 = _validate_shared_2048_texture(source_textures).copy()
    return _build_variants_from_scene(
        scene,
        texture_2048,
        len(source_textures),
    )


def build_object_texture_variants_from_texture(
    glb_bytes: bytes,
    texture_png: bytes,
) -> ObjectTextureVariants:
    """Replace one shared 2048 atlas without changing geometry or UVs."""

    payload = bytes(glb_bytes)
    if not payload:
        raise ValueError("The generated GLB is empty.")
    scene = _load_glb_scene(payload)
    source_textures = _collect_material_textures(scene)
    if not source_textures:
        raise ValueError("The generated GLB has no embedded base-color texture.")
    _validate_shared_2048_texture(source_textures)
    replacement_texture = _decode_png_rgba(texture_png)
    expected_shape = (TEXTURE_RESOLUTION_2048, TEXTURE_RESOLUTION_2048)
    if replacement_texture.shape[:2] != expected_shape:
        raise ValueError("The replacement texture must be 2048 x 2048.")
    return _build_variants_from_scene(
        scene,
        replacement_texture,
        len(source_textures),
    )


def _build_variants_from_scene(
    scene: trimesh.Scene,
    texture_2048: np.ndarray,
    material_texture_count: int,
) -> ObjectTextureVariants:
    """Build every exact-resolution artifact from one owned 2048 atlas."""

    texture_1024 = _resize_rgba(
        texture_2048,
        TEXTURE_RESOLUTION_1024,
        cv2.INTER_AREA,
    )
    texture_512 = _resize_rgba(
        texture_2048,
        TEXTURE_RESOLUTION_512,
        cv2.INTER_AREA,
    )
    textures_by_resolution = {
        TEXTURE_RESOLUTION_512: [texture_512] * material_texture_count,
        TEXTURE_RESOLUTION_1024: [texture_1024] * material_texture_count,
        TEXTURE_RESOLUTION_2048: [texture_2048] * material_texture_count,
    }

    glb_by_resolution: dict[int, bytes] = {}
    for resolution in TEXTURE_RESOLUTIONS:
        variant_scene = copy.deepcopy(scene)
        _replace_material_textures(
            variant_scene,
            textures_by_resolution[resolution],
        )
        glb_by_resolution[resolution] = bytes(
            variant_scene.export(file_type="glb")
        )
    return ObjectTextureVariants(
        glb_by_resolution=glb_by_resolution,
        texture_png_by_resolution={
            resolution: _encode_rgba_png(textures[0])
            for resolution, textures in textures_by_resolution.items()
        },
        preview_rgba_by_resolution={
            resolution: textures[0].copy()
            for resolution, textures in textures_by_resolution.items()
        },
    )


def _validate_shared_2048_texture(
    source_textures: list[np.ndarray],
) -> np.ndarray:
    if len({_texture_digest(texture) for texture in source_textures}) > 1:
        raise ValueError(
            "The generated object contains more than one distinct base-color "
            "texture atlas. Resolution variants currently require one shared "
            "atlas so UV-to-texture ownership remains unambiguous."
        )
    source_texture = source_textures[0]
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
    textures: list[np.ndarray] = []
    for _geometry_name, geometry in sorted(
        scene.geometry.items(),
        key=lambda item: str(item[0]),
    ):
        material = getattr(getattr(geometry, "visual", None), "material", None)
        for texture in _iter_material_textures(material):
            textures.append(_decode_texture_rgba(texture))
    return textures


def _iter_material_textures(material: object) -> list[object]:
    if material is None:
        return []
    nested_materials = getattr(material, "materials", None)
    if isinstance(nested_materials, list | tuple):
        textures: list[object] = []
        for nested_material in nested_materials:
            textures.extend(_iter_material_textures(nested_material))
        return textures
    for attribute_name in ("baseColorTexture", "image"):
        texture = getattr(material, attribute_name, None)
        if texture is not None:
            return [texture]
    return []


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


def _replace_material_texture_tree(
    material: object,
    textures: Iterator[np.ndarray],
) -> None:
    if material is None:
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
