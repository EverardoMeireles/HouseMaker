# ### Imports ###
from __future__ import annotations

import base64
import copy
import hashlib
import re
from collections import Counter
from dataclasses import dataclass, replace
from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError

from housemaker.glb import GeneratedModel
from housemaker.scene_export import _parse_glb, _serialize_glb

# ### Constants ###
DEFAULT_MATERIAL_PREVIEW_SIZE = 256
PROGRESSIVE_MATERIAL_PREVIEW_SIZE = 1024
PROGRESSIVE_MATERIAL_FALLBACK_SIZE = 128
MATERIAL_ID_EXTRAS_KEY = "housemaker_material_id"
ATLAS_ID_EXTRAS_KEY = "housemaker_atlas_id"
_MATERIAL_PACK_FORMAT = "housemaker-material-pack"
_PREVIEW_TEXTURE_KEYS = frozenset((
    "baseColorTexture", "diffuseTexture", "emissiveTexture",
    "transmissionTexture", "thicknessTexture", "diffuseTransmissionTexture",
    "diffuseTransmissionColorTexture",
))
_MAP_NAMES = {
    "baseColorTexture": "base_color", "diffuseTexture": "base_color",
    "normalTexture": "normal", "metallicRoughnessTexture": "metallic_roughness",
    "occlusionTexture": "occlusion", "emissiveTexture": "emissive",
    "transmissionTexture": "transmission", "thicknessTexture": "thickness",
}


# ### Export results ###
@dataclass(frozen=True, slots=True)
class MaterialPack:
    """One cube carrying one authoritative material and its embedded maps."""

    material_id: str
    material_name: str
    file_name: str
    glb_bytes: bytes
    material_index: int
    atlas_id: str | None = None
    tier: str = "full"
    resolution: int | None = None
    map_types: tuple[str, ...] = ()
    compression: str = "original"

    @property
    def byte_size(self) -> int:
        return len(self.glb_bytes)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.glb_bytes).hexdigest()

    def to_dict(self) -> dict[str, object]:
        result = {
            "materialId": self.material_id,
            "materialName": self.material_name,
            "materialIndex": self.material_index,
            "file": self.file_name,
            "tier": self.tier,
            "byteSize": self.byte_size,
            "sha256": self.sha256,
            "mapTypes": list(self.map_types),
            "compression": self.compression,
        }
        if self.resolution is not None:
            result["resolution"] = self.resolution
        if self.atlas_id is not None:
            result["atlasId"] = self.atlas_id
        return result


@dataclass(frozen=True, slots=True)
class MaterialPackExport:
    """Geometry-identical preview payload plus separately loadable materials.

    Only ``stripped_model.glb_bytes`` is rewritten. The retained authoring
    scene remains untouched for exact geometry and half-mesh metadata queries.
    """

    stripped_model: GeneratedModel
    packs: tuple[MaterialPack, ...]
    preview_packs: tuple[MaterialPack, ...] = ()

    @property
    def all_packs(self) -> tuple[MaterialPack, ...]:
        return self.preview_packs + self.packs

    def material_bindings(self) -> list[dict[str, object]]:
        return [pack.to_dict() for pack in self.packs]


# ### Public extraction API ###
def extract_material_packs(
    model: GeneratedModel,
    *,
    preview_size: int = DEFAULT_MATERIAL_PREVIEW_SIZE,
) -> MaterialPackExport:
    """Separate full embedded materials without reserializing house geometry.

    The main GLB keeps bounded PNG previews, original material indices, UVs,
    normals, transforms, node names and extras. Each full material pack is a
    single unit cube, not a copy of any house geometry. No files are written.
    """

    if not isinstance(model, GeneratedModel):
        raise TypeError("Material extraction requires a GeneratedModel.")
    if model.preview_only:
        raise ValueError("Material extraction requires the authoritative GLB.")
    if isinstance(preview_size, bool) or not isinstance(preview_size, int):
        raise TypeError("Material preview size must be an integer.")
    if not 1 <= preview_size <= DEFAULT_MATERIAL_PREVIEW_SIZE:
        raise ValueError("Material preview size must be between 1 and 256 pixels.")
    document, binary = _parse_glb(model.glb_bytes, "The material export GLB is invalid.")
    _validate_embedded_buffer(document, binary)
    packs = _extract_full_material_packs(document, binary)
    if not packs:
        return MaterialPackExport(model, ())
    preview_document = copy.deepcopy(document)
    _stamp_material_ids(preview_document, packs)
    preview_binary = _replace_images_with_previews(
        preview_document, binary, preview_size
    )
    payload = _serialize_glb(preview_document, preview_binary)
    return MaterialPackExport(replace(model, glb_bytes=payload), packs)


def build_progressive_material_export(
    model: GeneratedModel,
    *,
    bootstrap_material_ids=(),
    bootstrap_material_indices=(),
    use_ktx2_basis_compression: bool = False,
) -> MaterialPackExport:
    """Keep small lit fallbacks and emit same-UV preview/full material cubes.

    Every material is visible immediately. Starting-area materials retain at
    most 1024px base-color previews in the house; other materials retain 128px
    fallbacks. Separate 1024px preview packs support early room prefetching.
    Normal/ORM/AO maps remain exclusively in full packs. Glass/transmission,
    alpha and emissive appearance are preserved; removed ORM maps become
    representative scalar lighting parameters rather than changing shading.
    """

    if not isinstance(model, GeneratedModel):
        raise TypeError("Material extraction requires a GeneratedModel.")
    if model.preview_only:
        raise ValueError("Material extraction requires the authoritative GLB.")
    if isinstance(bootstrap_material_ids, str):
        raise TypeError("Bootstrap material IDs must be a sequence, not one string.")
    if not isinstance(use_ktx2_basis_compression, bool):
        raise TypeError("KTX2/Basis compression must be enabled with a boolean.")
    bootstrap_ids = frozenset(bootstrap_material_ids)
    bootstrap_indices = frozenset(bootstrap_material_indices)
    if any(isinstance(index, bool) or not isinstance(index, int) or index < 0
           for index in bootstrap_indices):
        raise ValueError("Bootstrap material indices must be non-negative integers.")
    document, binary = _parse_glb(model.glb_bytes, "The material export GLB is invalid.")
    _validate_embedded_buffer(document, binary)
    packs = _extract_full_material_packs(document, binary)
    if not packs:
        return MaterialPackExport(model, ())
    preview_document = copy.deepcopy(document)
    _stamp_material_ids(preview_document, packs)
    for material in preview_document.get("materials", []):
        _set_preview_pbr_constants(material, document, binary)
        _remove_preview_pbr_maps(material)
    preview_packs = []
    for pack in packs:
        payload = _build_material_cube(
            preview_document, binary, pack.material_index,
            pack.material_id, pack.material_name,
        )
        cube_document, cube_binary = _parse_glb(payload, "The material cube is invalid.")
        cube_binary = _replace_images_with_previews(
            cube_document, cube_binary, PROGRESSIVE_MATERIAL_PREVIEW_SIZE,
            lossy_color=True,
        )
        payload = _serialize_glb(cube_document, cube_binary)
        preview_packs.append(replace(
            pack,
            file_name=pack.file_name[:-4] + ".preview.glb",
            glb_bytes=payload,
            tier="preview",
            resolution=_maximum_image_dimension(cube_document, cube_binary),
            map_types=_material_map_types(cube_document["materials"][0]),
            compression="jpeg-png",
        ))
    discarded_views = _prune_unused_textures(preview_document)
    image_sizes = {}
    for pack in packs:
        size = (
            PROGRESSIVE_MATERIAL_PREVIEW_SIZE
            if pack.material_id in bootstrap_ids or pack.material_index in bootstrap_indices
            else PROGRESSIVE_MATERIAL_FALLBACK_SIZE
        )
        material = preview_document["materials"][pack.material_index]
        for texture_index in _texture_indices(material):
            for image_index in _texture_image_indices(preview_document["textures"][texture_index]):
                image_sizes[image_index] = max(size, image_sizes.get(image_index, 0))
    preview_binary = _replace_images_with_previews(
        preview_document, binary, PROGRESSIVE_MATERIAL_FALLBACK_SIZE,
        image_sizes=image_sizes, discarded_image_views=discarded_views,
        lossy_color=True,
    )
    if use_ktx2_basis_compression:
        from housemaker.ktx2_compression import compress_material_glb

        packs = tuple(replace(
            pack, glb_bytes=compress_material_glb(pack.glb_bytes),
            compression="ktx2-basis",
        ) for pack in packs)
        preview_packs = [replace(
            pack, glb_bytes=compress_material_glb(pack.glb_bytes, preview=True),
            compression="ktx2-basis",
        ) for pack in preview_packs]
    payload = _serialize_glb(preview_document, preview_binary)
    return MaterialPackExport(
        replace(model, glb_bytes=payload), packs, tuple(preview_packs),
    )


# ### Shared material identities and full-fidelity extraction ###
def _extract_full_material_packs(document, binary):
    atlas_ids = _material_atlas_ids(document)
    used = _used_material_indices(document)
    materials = document.get("materials", [])
    names = Counter(str(material.get("name", "")) for material in materials)
    packs = []
    for index in sorted(used):
        if not 0 <= index < len(materials):
            raise ValueError("A house primitive references an unavailable material.")
        material = materials[index]
        if not _texture_indices(material) and index not in atlas_ids:
            continue
        name = str(material.get("name") or f"Material {index + 1}")
        atlas_id = atlas_ids.get(index)
        identity = atlas_id or name
        if atlas_id is None and names[str(material.get("name", ""))] > 1:
            identity += f":{index}"
        existing = material.get("extras", {})
        material_id = (
            existing.get(MATERIAL_ID_EXTRAS_KEY)
            if isinstance(existing, dict) else None
        )
        if not isinstance(material_id, str) or not material_id.strip():
            material_id = "material_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        material_id = material_id.strip()
        file_name = _safe_file_stem(name) + "__" + _safe_file_stem(material_id) + ".glb"
        try:
            pack_bytes = _build_material_cube(document, binary, index, material_id, name)
        except (IndexError, KeyError, TypeError) as error:
            raise ValueError("A material texture dependency is invalid.") from error
        pack_document, pack_binary = _parse_glb(pack_bytes, "The material cube is invalid.")
        packs.append(MaterialPack(
            material_id, name, file_name, pack_bytes, index, atlas_id,
            resolution=_maximum_image_dimension(pack_document, pack_binary),
            map_types=_material_map_types(material),
        ))
    if len({pack.material_id for pack in packs}) != len(packs):
        raise ValueError("Material IDs must uniquely identify exported materials.")
    if len({pack.file_name.casefold() for pack in packs}) != len(packs):
        raise ValueError("Material pack filenames must be unique.")
    return tuple(packs)


def _stamp_material_ids(document, packs):
    for pack in packs:
        material = document["materials"][pack.material_index]
        material.setdefault("name", pack.material_name)
        extras = material.setdefault("extras", {})
        if isinstance(extras, dict):
            extras[MATERIAL_ID_EXTRAS_KEY] = pack.material_id


# ### Progressive map selection and dependency pruning ###
def _set_preview_pbr_constants(material, document, binary):
    """Approximate removed ORM maps instead of turning dielectrics metallic.

    glTF commonly uses metallicFactor=1 with a black metallic map. Keeping
    that scalar after removing the map would incorrectly make everything
    metal. Average linear ORM channels, respecting the base-color alpha when
    both texture coordinates agree. Authoritative full materials are untouched.
    """

    pbr = material.get("pbrMetallicRoughness", {})
    texture_info = pbr.get("metallicRoughnessTexture")
    if not isinstance(texture_info, dict):
        return
    orm = _small_texture_image(document, binary, texture_info)
    values = np.asarray(orm.convert("RGB"), dtype=np.float64) / 255.0
    weights = np.ones(values.shape[:2], dtype=np.float64)
    base = pbr.get("baseColorTexture")
    alpha_mode = material.get("alphaMode", "OPAQUE")
    if isinstance(base, dict) and alpha_mode != "OPAQUE" and _same_texture_coordinates(base, texture_info):
        alpha = _small_texture_image(document, binary, base).getchannel("A")
        alpha = alpha.resize(orm.size, Image.Resampling.BOX)
        weights = np.asarray(alpha, dtype=np.float64) / 255.0
        factors = pbr.get("baseColorFactor", (1.0, 1.0, 1.0, 1.0))
        weights *= float(factors[3])
        if alpha_mode == "MASK":
            weights = (weights >= float(material.get("alphaCutoff", 0.5))).astype(float)
    total = float(weights.sum())
    if total > 0:
        roughness = float((values[:, :, 1] * weights).sum() / total)
        metallic = float((values[:, :, 2] * weights).sum() / total)
    else:
        roughness, metallic = 1.0, 0.0
    pbr["roughnessFactor"] = float(pbr.get("roughnessFactor", 1.0)) * roughness
    pbr["metallicFactor"] = float(pbr.get("metallicFactor", 1.0)) * metallic


def _small_texture_image(document, binary, texture_info):
    texture = document["textures"][texture_info["index"]]
    source_index = texture.get("source")
    if not isinstance(source_index, int):
        source_index = min(_texture_image_indices(texture))
    payload = _image_bytes(document, binary, document["images"][source_index])
    with Image.open(BytesIO(payload)) as source:
        result = source.convert("RGBA")
        result.thumbnail((128, 128), Image.Resampling.BOX)
        return result


def _same_texture_coordinates(left, right):
    return (
        left.get("texCoord", 0) == right.get("texCoord", 0)
        and left.get("extensions", {}).get("KHR_texture_transform", {})
        == right.get("extensions", {}).get("KHR_texture_transform", {})
    )


def _remove_preview_pbr_maps(value):
    if isinstance(value, dict):
        for key in tuple(value):
            if key.endswith("Texture") and key not in _PREVIEW_TEXTURE_KEYS:
                value.pop(key)
            else:
                _remove_preview_pbr_maps(value[key])
    elif isinstance(value, list):
        for child in value:
            _remove_preview_pbr_maps(child)


def _material_map_types(value):
    result = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key.endswith("Texture") and isinstance(child, dict):
                result.add(_MAP_NAMES.get(
                    key, re.sub(r"(?<!^)(?=[A-Z])", "_", key[:-7]).lower(),
                ))
            result.update(_material_map_types(child))
    elif isinstance(value, list):
        for child in value:
            result.update(_material_map_types(child))
    return tuple(sorted(result))


def _texture_image_indices(value):
    result = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "source" and isinstance(child, int):
                result.add(child)
            else:
                result.update(_texture_image_indices(child))
    elif isinstance(value, list):
        for child in value:
            result.update(_texture_image_indices(child))
    return result


def _prune_unused_textures(document):
    """Remove full-PBR dependencies without changing geometry view indices."""

    old_textures = document.get("textures", [])
    used_textures = sorted(_texture_indices(document.get("materials", [])))
    texture_map = {old: new for new, old in enumerate(used_textures)}

    def remap_material(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key.endswith("Texture") and isinstance(child, dict) and "index" in child:
                    child["index"] = texture_map[child["index"]]
                remap_material(child)
        elif isinstance(value, list):
            for child in value:
                remap_material(child)

    remap_material(document.get("materials", []))
    textures = [old_textures[index] for index in used_textures]
    old_images = document.get("images", [])
    used_images = sorted(_texture_image_indices(textures))
    image_map = {old: new for new, old in enumerate(used_images)}

    def remap_images(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "source" and isinstance(child, int):
                    value[key] = image_map[child]
                else:
                    remap_images(child)
        elif isinstance(value, list):
            for child in value:
                remap_images(child)

    remap_images(textures)
    images = [old_images[index] for index in used_images]
    removed_views = {
        image["bufferView"] for image in old_images if "bufferView" in image
    } - {
        image["bufferView"] for image in images if "bufferView" in image
    }
    document["textures"] = textures
    document["images"] = images
    return removed_views


def _maximum_image_dimension(document, binary):
    largest = 0
    for image in document.get("images", []):
        payload = _image_bytes(document, binary, image)
        try:
            with Image.open(BytesIO(payload)) as source:
                largest = max(largest, *source.size)
        except (OSError, UnidentifiedImageError) as error:
            raise ValueError("A material image cannot be decoded for its dimensions.") from error
    return largest or None


# ### Original GLB data and material identity ###
def _validate_embedded_buffer(document, binary):
    buffers = document.get("buffers", [])
    if len(buffers) != 1 or buffers[0].get("uri") is not None:
        raise ValueError("Material export requires one embedded GLB buffer.")
    if buffers[0].get("byteLength", 0) > len(binary):
        raise ValueError("The embedded GLB buffer is truncated.")
    for view in document.get("bufferViews", []):
        if view.get("buffer", 0) != 0:
            raise ValueError("Material export cannot read external buffers.")
        if "EXT_meshopt_compression" in view.get("extensions", {}):
            raise ValueError("Compressed meshopt buffers cannot be compacted safely.")


def _used_material_indices(document):
    return {
        primitive["material"]
        for mesh in document.get("meshes", [])
        for primitive in mesh.get("primitives", [])
        if isinstance(primitive.get("material"), int)
    }


def _material_atlas_ids(document):
    result = {}
    meshes = document.get("meshes", [])
    for node in document.get("nodes", []):
        index = node.get("mesh")
        if not isinstance(index, int) or not 0 <= index < len(meshes):
            continue
        mesh = meshes[index]
        extras = node.get("extras", {})
        mesh_extras = mesh.get("extras", {})
        atlas_id = (
            extras.get(ATLAS_ID_EXTRAS_KEY) if isinstance(extras, dict) else None
        ) or (
            mesh_extras.get(ATLAS_ID_EXTRAS_KEY)
            if isinstance(mesh_extras, dict) else None
        )
        if not isinstance(atlas_id, str) or not atlas_id:
            continue
        for primitive in mesh.get("primitives", []):
            material = primitive.get("material")
            if isinstance(material, int):
                previous = result.setdefault(material, atlas_id)
                if previous != atlas_id:
                    raise ValueError("One material cannot belong to different Atlas IDs.")
    return result


def _safe_file_stem(name):
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")[:80] or "material"


def _texture_indices(value):
    result = set()
    if isinstance(value, dict):
        for key, entry in value.items():
            if key.endswith("Texture") and isinstance(entry, dict):
                index = entry.get("index")
                if isinstance(index, int):
                    result.add(index)
            result.update(_texture_indices(entry))
    elif isinstance(value, list):
        for entry in value:
            result.update(_texture_indices(entry))
    return result


def _buffer_view_bytes(document, binary, index):
    views = document.get("bufferViews", [])
    if not isinstance(index, int) or not 0 <= index < len(views):
        raise ValueError("A material image references an unavailable buffer view.")
    view = views[index]
    start = view.get("byteOffset", 0)
    length = view.get("byteLength", 0)
    if not isinstance(start, int) or not isinstance(length, int) or start < 0 or length < 0:
        raise ValueError("A GLB buffer view has invalid bounds.")
    if start + length > len(binary):
        raise ValueError("A GLB buffer view extends beyond its embedded buffer.")
    return binary[start:start + length]


def _image_bytes(document, binary, image):
    if "bufferView" in image:
        return _buffer_view_bytes(document, binary, image["bufferView"])
    uri = image.get("uri", "")
    if isinstance(uri, str) and uri.startswith("data:") and ";base64," in uri:
        try:
            return base64.b64decode(uri.split(",", 1)[1], validate=True)
        except ValueError as error:
            raise ValueError("An embedded material image is invalid.") from error
    raise ValueError("Material export requires embedded images, not external image paths.")


# ### Main-file bounded previews and binary compaction ###
def _jpeg_compatible_image_indices(document):
    """Only opaque base color may discard alpha, including unused Atlas padding."""

    compatible, requires_alpha = set(), set()

    def collect(value, opaque_material):
        if isinstance(value, dict):
            for key, child in value.items():
                if key.endswith("Texture") and isinstance(child, dict) and isinstance(child.get("index"), int):
                    texture = document["textures"][child["index"]]
                    indices = _texture_image_indices(texture)
                    if opaque_material and key in {"baseColorTexture", "diffuseTexture"}:
                        compatible.update(indices)
                    else:
                        requires_alpha.update(indices)
                collect(child, opaque_material)
        elif isinstance(value, list):
            for child in value:
                collect(child, opaque_material)

    for material in document.get("materials", []):
        collect(material, material.get("alphaMode", "OPAQUE") == "OPAQUE")
    return compatible - requires_alpha


def _replace_images_with_previews(
    document, binary, size, *, image_sizes=None, discarded_image_views=(),
    lossy_color=False,
):
    image_views = {}
    original_images = document.get("images", [])
    jpeg_images = _jpeg_compatible_image_indices(document) if lossy_color else set()
    for image_index, image in enumerate(original_images):
        payload = _image_bytes(document, binary, image)
        try:
            with Image.open(BytesIO(payload)) as source:
                jpeg_compatible = image_index in jpeg_images
                preview = source.convert("RGB" if jpeg_compatible else "RGBA")
                image_size = (image_sizes or {}).get(image_index, size)
                preview.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
                output = BytesIO()
                if jpeg_compatible:
                    preview.save(
                        output, format="JPEG", quality=65 if image_size > 128 else 50,
                        optimize=True, subsampling=2,
                    )
                    mime_type = "image/jpeg"
                else:
                    preview.save(output, format="PNG", optimize=lossy_color)
                    mime_type = "image/png"
        except (OSError, UnidentifiedImageError) as error:
            raise ValueError("A material image cannot be decoded for its lightweight preview.") from error
        if "bufferView" not in image:
            image["bufferView"] = len(document.setdefault("bufferViews", []))
            document["bufferViews"].append({"buffer": 0, "byteLength": 0})
        image_views[image["bufferView"]] = output.getvalue()
        image.pop("uri", None)
        image["mimeType"] = mime_type
    # Previews use ordinary PNG/JPEG refs, even if the original uses WebP/KTX.
    # Keep the authoritative codec only in the separate full material pack.
    preview_codecs = {"EXT_texture_webp", "KHR_texture_basisu"}
    for texture in document.get("textures", []):
        extensions = texture.get("extensions", {})
        for codec in preview_codecs.intersection(extensions):
            payload = extensions.pop(codec)
            if "source" not in texture:
                texture["source"] = payload["source"]
        if not extensions:
            texture.pop("extensions", None)
    remaining_extensions = _extension_names(document)
    for field in ("extensionsUsed", "extensionsRequired"):
        if field in document:
            document[field] = [name for name in document[field]
                               if name not in preview_codecs or name in remaining_extensions]
    # Never overwrite a buffer view shared with vertex/accessor data.
    without_images = {key: value for key, value in document.items() if key != "images"}
    referenced = set()

    def collect(value):
        if isinstance(value, dict):
            if isinstance(value.get("bufferView"), int):
                referenced.add(value["bufferView"])
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(without_images)
    if referenced.intersection(set(image_views) | set(discarded_image_views)):
        raise ValueError("Image and geometry buffer views must be independent.")
    result = bytearray()
    for index, view in enumerate(document.get("bufferViews", [])):
        payload = image_views.get(index)
        if index in discarded_image_views:
            # Leave an inert, valid view to preserve accessor/view indices.
            # The old full-resolution image bytes must not remain in the BIN.
            payload = b"\0" * 4
        if payload is None:
            payload = _buffer_view_bytes(document, binary, index)
        result.extend(b"\0" * (-len(result) % 4))
        view["byteOffset"] = len(result)
        view["byteLength"] = len(payload)
        result.extend(payload)
    document["buffers"][0]["byteLength"] = len(result)
    return bytes(result)


# ### Material-cube geometry and exact texture dependencies ###
def _build_material_cube(document, binary, material_index, material_id, name):
    material = copy.deepcopy(document["materials"][material_index])
    pack = {
        "asset": {"version": "2.0", "generator": "HouseMaker",
                  "extras": {"format": _MATERIAL_PACK_FORMAT, "materialId": material_id}},
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"name": name, "mesh": 0}],
        "meshes": [{"name": name, "primitives": []}],
        "materials": [material], "textures": [], "images": [], "samplers": [],
        "accessors": [], "bufferViews": [], "buffers": [{"byteLength": 0}],
    }
    output = bytearray()
    positions, normals, uv, indices = _cube_arrays()
    position = _append_accessor(pack, output, positions, 5126, "VEC3", bounds=True)
    normal = _append_accessor(pack, output, normals, 5126, "VEC3")
    texcoord = _append_accessor(pack, output, uv, 5126, "VEC2")
    triangles = _append_accessor(pack, output, indices, 5123, "SCALAR")
    attributes = {"POSITION": position, "NORMAL": normal,
                  "TEXCOORD_0": texcoord, "TEXCOORD_1": texcoord}
    pack["meshes"][0]["primitives"] = [{
        "attributes": attributes, "indices": triangles, "material": 0, "mode": 4,
    }]
    textures, images, samplers = {}, {}, {}

    def copy_image(index):
        if not isinstance(index, int) or not 0 <= index < len(document.get("images", [])):
            raise ValueError("A texture references an unavailable embedded image.")
        if index not in images:
            original = document.get("images", [])[index]
            image = copy.deepcopy(original)
            image.pop("uri", None)
            image["bufferView"] = _append_view(pack, output, _image_bytes(document, binary, original))
            if "mimeType" not in image:
                uri = original.get("uri", "")
                image["mimeType"] = uri[5:].split(";", 1)[0] if uri.startswith("data:") else "image/png"
            images[index] = len(pack["images"])
            pack["images"].append(image)
        return images[index]

    def remap_texture_sources(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "source" and isinstance(child, int):
                    value[key] = copy_image(child)
                elif key == "sampler" and isinstance(child, int):
                    if not 0 <= child < len(document.get("samplers", [])):
                        raise ValueError("A texture references an unavailable sampler.")
                    if child not in samplers:
                        samplers[child] = len(pack["samplers"])
                        pack["samplers"].append(copy.deepcopy(document["samplers"][child]))
                    value[key] = samplers[child]
                else:
                    remap_texture_sources(child)
        elif isinstance(value, list):
            for child in value:
                remap_texture_sources(child)

    for index in sorted(_texture_indices(material)):
        if not 0 <= index < len(document.get("textures", [])):
            raise ValueError("A material references an unavailable texture.")
        texture = copy.deepcopy(document.get("textures", [])[index])
        remap_texture_sources(texture)
        textures[index] = len(pack["textures"])
        pack["textures"].append(texture)

    def remap_material(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key.endswith("Texture") and isinstance(child, dict) and "index" in child:
                    child["index"] = textures[child["index"]]
                if key == "texCoord" and isinstance(child, int) and child >= 0:
                    attributes[f"TEXCOORD_{child}"] = texcoord
                remap_material(child)
        elif isinstance(value, list):
            for child in value:
                remap_material(child)

    remap_material(material)
    extensions = _extension_names(pack)
    if extensions:
        pack["extensionsUsed"] = sorted(extensions)
    required = set(document.get("extensionsRequired", ())).intersection(extensions)
    if required:
        pack["extensionsRequired"] = sorted(required)
    pack["buffers"][0]["byteLength"] = len(output)
    return _serialize_glb(pack, bytes(output))


def _extension_names(value):
    result = set()
    if isinstance(value, dict):
        extensions = value.get("extensions", {})
        if isinstance(extensions, dict):
            result.update(extensions)
        for child in value.values():
            result.update(_extension_names(child))
    elif isinstance(value, list):
        for child in value:
            result.update(_extension_names(child))
    return result


def _append_view(document, output, payload):
    output.extend(b"\0" * (-len(output) % 4))
    index = len(document["bufferViews"])
    document["bufferViews"].append({"buffer": 0, "byteOffset": len(output), "byteLength": len(payload)})
    output.extend(payload)
    return index


def _append_accessor(document, output, values, component_type, kind, *, bounds=False):
    view = _append_view(document, output, values.tobytes())
    accessor = {"bufferView": view, "componentType": component_type,
                "count": len(values), "type": kind}
    if bounds:
        accessor["min"] = values.min(axis=0).tolist()
        accessor["max"] = values.max(axis=0).tolist()
    document["accessors"].append(accessor)
    return len(document["accessors"]) - 1


def _cube_arrays():
    # Four vertices per face keep normals sharp and each map visible on all sides.
    faces = (
        (((1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)), (1, 0, 0)),
        (((-1, 1, -1), (-1, -1, -1), (-1, -1, 1), (-1, 1, 1)), (-1, 0, 0)),
        (((1, 1, -1), (-1, 1, -1), (-1, 1, 1), (1, 1, 1)), (0, 1, 0)),
        (((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)), (0, -1, 0)),
        (((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)), (0, 0, 1)),
        (((-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, -1)), (0, 0, -1)),
    )
    positions = np.asarray([point for points, _normal in faces for point in points], dtype="<f4") * 0.5
    normals = np.asarray([normal for _points, normal in faces for _point in range(4)], dtype="<f4")
    uv = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)) * 6, dtype="<f4")
    indices = np.asarray([face * 4 + value for face in range(6) for value in (0, 1, 2, 0, 2, 3)], dtype="<u2")
    return positions, normals, uv, indices
