# ### Imports ###
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.surface_geometry import (
    SURFACE_TYPE_WALL,
    SURFACE_TYPES,
)


# ### Constants ###
DEFAULT_SURFACE_TEXTURE_WORLD_SIZE_METERS = 2.0
SURFACE_MATERIAL_EPSILON = 1e-8
MAX_SURFACE_TEXTURE_BYTES = 64 * 1024 * 1024
MAX_SURFACE_TEXTURE_DIMENSION_PIXELS = 16_384


# ### Material models ###
@dataclass(frozen=True)
class ResolvedSurfaceMaterial:
    """Validated PNG data shared by GLB export and OpenGL preview."""

    png_bytes: bytes
    texture_rgba: np.ndarray

    def __post_init__(self) -> None:
        if not self.png_bytes:
            raise ValueError("A surface material PNG cannot be empty.")
        rgba = np.asarray(self.texture_rgba, dtype=np.uint8)
        if rgba.ndim != 3 or rgba.shape[2] != 4:
            raise ValueError("A surface material must contain RGBA pixels.")
        object.__setattr__(self, "texture_rgba", np.ascontiguousarray(rgba))


# ### Public material helpers ###
def resolve_surface_materials(
    surface_materials: Mapping[str, bytes | bytearray | memoryview | str | Path],
) -> dict[str, ResolvedSurfaceMaterial]:
    """Resolve a stable-surface material map without retaining open files."""

    if not isinstance(surface_materials, Mapping):
        raise TypeError("Surface materials must be provided as a mapping.")
    resolved: dict[str, ResolvedSurfaceMaterial] = {}
    for raw_surface_id, source in surface_materials.items():
        surface_id = str(raw_surface_id).strip()
        if not surface_id:
            raise ValueError("A surface material requires a non-empty surface ID.")
        resolved[surface_id] = resolve_surface_material(source)
    return resolved


def build_assignment_surface_material_source_map(
    assignments: Iterable[object],
    asset_directory: str | Path,
) -> dict[str, Path]:
    """Resolve persisted assignment assets with last-valid-assignment wins.

    Missing files and paths escaping ``asset_directory`` are ignored. The
    helper accepts dataclass-like assignment objects or mappings so the GLB
    layer does not depend on the persistence model.
    """

    root = Path(asset_directory).expanduser().resolve()
    sources: dict[str, Path] = {}
    for assignment in assignments:
        raw_asset_path = _get_assignment_member(assignment, "asset_path")
        raw_surface_ids = _get_assignment_member(assignment, "surface_ids")
        if not isinstance(raw_asset_path, (str, Path)):
            continue
        if isinstance(raw_surface_ids, (str, bytes)):
            continue
        try:
            surface_ids = tuple(raw_surface_ids)  # type: ignore[arg-type]
        except TypeError:
            continue
        candidate = (root / raw_asset_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        if not candidate.is_file() or candidate.suffix.lower() != ".png":
            continue
        for raw_surface_id in surface_ids:
            surface_id = str(raw_surface_id).strip()
            if surface_id:
                sources[surface_id] = candidate
    return sources


def resolve_surface_material(
    source: bytes | bytearray | memoryview | str | Path,
) -> ResolvedSurfaceMaterial:
    """Load one PNG byte string or path and normalize it to owned RGBA data."""

    if isinstance(source, (str, Path)):
        try:
            raw_bytes = Path(source).expanduser().read_bytes()
        except OSError as error:
            raise ValueError(f"Unable to read surface texture: {source}") from error
    elif isinstance(source, (bytes, bytearray, memoryview)):
        raw_bytes = bytes(source)
    else:
        raise TypeError("Surface material values must be PNG bytes or paths.")
    if not raw_bytes:
        raise ValueError("A surface texture PNG cannot be empty.")
    if len(raw_bytes) > MAX_SURFACE_TEXTURE_BYTES:
        raise ValueError("A surface texture PNG is too large.")

    try:
        with Image.open(BytesIO(raw_bytes)) as loaded_image:
            if loaded_image.format != "PNG":
                raise ValueError("Surface material images must use PNG format.")
            width, height = loaded_image.size
            if (
                width <= 0
                or height <= 0
                or width > MAX_SURFACE_TEXTURE_DIMENSION_PIXELS
                or height > MAX_SURFACE_TEXTURE_DIMENSION_PIXELS
            ):
                raise ValueError("Surface texture dimensions are outside the limit.")
            texture_rgba = np.asarray(
                loaded_image.convert("RGBA"),
                dtype=np.uint8,
            ).copy()
    except (OSError, SyntaxError) as error:
        raise ValueError("Surface material data is not a valid PNG image.") from error
    return ResolvedSurfaceMaterial(
        png_bytes=raw_bytes,
        texture_rgba=texture_rgba,
    )


def build_world_planar_textured_mesh(
    mesh: trimesh.Trimesh,
    surface_type: str,
    material: ResolvedSurfaceMaterial,
    *,
    texture_world_size_meters: float = DEFAULT_SURFACE_TEXTURE_WORLD_SIZE_METERS,
    material_name: str | None = None,
    double_sided: bool = False,
) -> trimesh.Trimesh:
    """Expand faces and attach stable world-scale planar UV coordinates."""

    if surface_type not in SURFACE_TYPES:
        raise ValueError(f"Unknown fixed surface type: {surface_type!r}.")
    tile_size = normalize_texture_world_size(texture_world_size_meters)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=float)
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or vertices.ndim != 2
        or vertices.shape[1] != 3
        or faces.size == 0
    ):
        raise ValueError("Surface material meshes must contain triangles.")
    face_vertices = vertices[faces]
    face_normals = np.asarray(mesh.face_normals, dtype=float)
    if face_normals.shape != faces.shape:
        raise ValueError("Surface material mesh normals are invalid.")
    uv_coordinates = build_world_planar_face_uvs(
        face_vertices,
        face_normals,
        surface_type,
        texture_world_size_meters=tile_size,
    )
    expanded_vertices = face_vertices.reshape(-1, 3)
    expanded_faces = np.arange(len(expanded_vertices), dtype=np.int64).reshape(-1, 3)
    texture_image = Image.fromarray(material.texture_rgba, mode="RGBA")
    return trimesh.Trimesh(
        vertices=np.ascontiguousarray(expanded_vertices),
        faces=np.ascontiguousarray(expanded_faces),
        visual=TextureVisuals(
            uv=np.ascontiguousarray(uv_coordinates.reshape(-1, 2)),
            material=PBRMaterial(
                name=material_name,
                baseColorFactor=[255, 255, 255, 255],
                baseColorTexture=texture_image,
                metallicFactor=0.0,
                roughnessFactor=0.72,
                doubleSided=bool(double_sided),
            ),
        ),
        process=False,
    )


def build_world_planar_face_uvs(
    face_vertices: np.ndarray,
    face_normals: np.ndarray,
    surface_type: str,
    *,
    texture_world_size_meters: float = DEFAULT_SURFACE_TEXTURE_WORLD_SIZE_METERS,
) -> np.ndarray:
    """Return per-face UVs so hard seams never share incompatible coordinates."""

    tile_size = normalize_texture_world_size(texture_world_size_meters)
    vertices = np.asarray(face_vertices, dtype=float)
    normals = np.asarray(face_normals, dtype=float)
    if (
        vertices.ndim != 3
        or vertices.shape[1:] != (3, 3)
        or normals.shape != (len(vertices), 3)
    ):
        raise ValueError("Planar UV inputs must contain triangle vertices and normals.")
    uv_coordinates = np.empty((len(vertices), 3, 2), dtype=float)
    for face_index, triangle_vertices in enumerate(vertices):
        normal = normals[face_index]
        if surface_type == SURFACE_TYPE_WALL and abs(normal[2]) < 0.7:
            tangent = np.array((-normal[1], normal[0], 0.0), dtype=float)
            tangent_length = float(np.linalg.norm(tangent))
            if tangent_length <= SURFACE_MATERIAL_EPSILON:
                tangent = np.array((1.0, 0.0, 0.0), dtype=float)
            else:
                tangent /= tangent_length
            uv_coordinates[face_index, :, 0] = (
                triangle_vertices @ tangent
            ) / tile_size
            uv_coordinates[face_index, :, 1] = (
                triangle_vertices[:, 2] / tile_size
            )
        else:
            uv_coordinates[face_index, :, 0] = (
                triangle_vertices[:, 0] / tile_size
            )
            uv_coordinates[face_index, :, 1] = (
                triangle_vertices[:, 1] / tile_size
            )
    return np.ascontiguousarray(uv_coordinates)


def normalize_texture_world_size(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("Surface texture world size must be a number.")
    try:
        size = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Surface texture world size must be a number.") from error
    if not math.isfinite(size) or size <= 0.0:
        raise ValueError("Surface texture world size must be finite and positive.")
    return size


# ### Assignment helpers ###
def _get_assignment_member(assignment: object, name: str) -> object:
    if isinstance(assignment, Mapping):
        return assignment.get(name)
    return getattr(assignment, name, None)
