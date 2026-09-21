"""Deterministic per-repeat UV choices for architectural surfaces."""

# ### Imports ###
from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping

import numpy as np
import trimesh
from trimesh.visual.texture import TextureVisuals

# ### Constants ###
SURFACE_TILING_MODE_WHOLE_REPEATS = "whole_repeats"
SURFACE_TILING_MODE_EDGE_VARIANTS = "edge_variants"
SURFACE_TILING_MODES = frozenset(
    (SURFACE_TILING_MODE_WHOLE_REPEATS, SURFACE_TILING_MODE_EDGE_VARIANTS)
)


# ### Public validation ###
def normalize_surface_tiling_mode(value: object) -> str | None:
    """Return a supported optional tiling mode."""

    if value is None or value == "none":
        return None
    if not isinstance(value, str) or value not in SURFACE_TILING_MODES:
        raise ValueError("Unknown surface tiling mode.")
    return value


def normalize_surface_tiling_seed(value: object) -> int:
    """Keep repeat choices stable across process runs and exports."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("A surface tiling seed must be a non-negative integer.")
    return value


def quarter_turns_for_tile(
    tile_u: int,
    tile_v: int,
    seed: int,
    *,
    allow_odd_turns: bool = True,
) -> int:
    """Choose a reproducible rotation for one signed source-UV tile."""

    normalized_seed = normalize_surface_tiling_seed(seed)
    if isinstance(tile_u, bool) or not isinstance(tile_u, int):
        raise TypeError("A surface tile U index must be an integer.")
    if isinstance(tile_v, bool) or not isinstance(tile_v, int):
        raise TypeError("A surface tile V index must be an integer.")
    payload = f"{normalized_seed}:{tile_u}:{tile_v}".encode("ascii")
    turns = hashlib.blake2b(payload, digest_size=1).digest()[0] & 3
    return turns if allow_odd_turns else 2 * (turns % 2)


# ### UV transformation ###
def transform_repeating_surface_uv_mesh(
    mesh: trimesh.Trimesh,
    *,
    mode: str,
    seed: int,
    allow_quarter_turns: bool = True,
) -> trimesh.Trimesh:
    """Clip source-UV tiles and remap UV0 without altering physical geometry.

    Whole-repeat rotations use the same image for each tile. The tangent frame
    follows the changed UV gradients, so normal-map pixels must not be rotated.
    Edge variants select one quadrant of a 2 x 2 image using tile parity.
    """

    normalized_mode = normalize_surface_tiling_mode(mode)
    if normalized_mode is None:
        raise ValueError("A surface tiling transformation needs a tiling mode.")
    normalized_seed = normalize_surface_tiling_seed(seed)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("Surface tiling requires a triangle mesh.")
    source_uv = np.asarray(getattr(mesh.visual, "uv", None), dtype=float)
    vertices = np.asarray(mesh.vertices, dtype=float)
    normals = np.asarray(mesh.vertex_normals, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or source_uv.shape != (len(vertices), 2)
        or normals.shape != vertices.shape
        or faces.ndim != 2
        or faces.shape[1] != 3
        or not np.all(np.isfinite(source_uv))
    ):
        raise ValueError("Surface tiling requires finite per-vertex UVs and normals.")

    # Atlas export already owns the precise tile clipper and triangle limit.
    # Import lazily because that module imports GLB conversion utilities.
    from housemaker.atlas_export import (
        GEOMETRY_EPSILON,
        MAX_TILED_SURFACE_TRIANGLES,
        _clip_polygon_to_uv_tile,
        _covered_tile_indices,
        _normalize_rows,
    )

    output_vertices: list[np.ndarray] = []
    output_normals: list[np.ndarray] = []
    output_uv: list[np.ndarray] = []
    attribute_arrays = _vertex_attribute_arrays(mesh, len(vertices))
    output_attributes: dict[str, list[np.ndarray]] = {
        name: [] for name in attribute_arrays
    }
    attribute_widths = {
        name: value.shape[1] for name, value in attribute_arrays.items()
    }

    for face in faces:
        face_uv = source_uv[face]
        u_tiles = _covered_tile_indices(face_uv[:, 0])
        v_tiles = _covered_tile_indices(face_uv[:, 1])
        maximum_new_triangles = len(u_tiles) * len(v_tiles) * 4
        if (
            len(output_vertices) // 3 + maximum_new_triangles
            > MAX_TILED_SURFACE_TRIANGLES
        ):
            raise ValueError("A repeating surface produces too many tiles.")
        polygon_columns = [vertices[face], normals[face], face_uv]
        polygon_columns.extend(value[face] for value in attribute_arrays.values())
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
                    if np.linalg.norm(
                        np.cross(
                            triangle[1, :3] - triangle[0, :3],
                            triangle[2, :3] - triangle[0, :3],
                        )
                    ) <= GEOMETRY_EPSILON:
                        continue
                    if len(output_vertices) // 3 >= MAX_TILED_SURFACE_TRIANGLES:
                        raise ValueError("A repeating surface produces too many tiles.")
                    local_uv = np.clip(
                        triangle[:, 6:8] - (float(tile_u), float(tile_v)),
                        0.0,
                        1.0,
                    )
                    mapped_uv = _map_tile_uv(
                        local_uv,
                        mode=normalized_mode,
                        seed=normalized_seed,
                        tile_u=tile_u,
                        tile_v=tile_v,
                        allow_quarter_turns=allow_quarter_turns,
                    )
                    output_vertices.extend(triangle[:, :3])
                    output_normals.extend(_normalize_rows(triangle[:, 3:6]))
                    output_uv.extend(mapped_uv)
                    offset = 8
                    for name, width in attribute_widths.items():
                        output_attributes[name].extend(
                            triangle[:, offset : offset + width]
                        )
                        offset += width

    if not output_vertices:
        raise ValueError("A repeating surface contains no usable triangles.")
    result = trimesh.Trimesh(
        vertices=np.asarray(output_vertices, dtype=float),
        faces=np.arange(len(output_vertices), dtype=np.int64).reshape((-1, 3)),
        vertex_normals=np.asarray(output_normals, dtype=float),
        visual=TextureVisuals(uv=np.asarray(output_uv, dtype=float), material=None),
        metadata=copy.deepcopy(dict(getattr(mesh, "metadata", {}) or {})),
        process=False,
    )
    for name, values in output_attributes.items():
        result.vertex_attributes[name] = np.asarray(values, dtype=np.float32)
    return result


# ### Internal helpers ###
def _vertex_attribute_arrays(
    mesh: trimesh.Trimesh, vertex_count: int
) -> Mapping[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, raw in mesh.vertex_attributes.items():
        value = np.asarray(raw, dtype=float)
        if value.ndim == 1:
            value = value[:, np.newaxis]
        if (
            value.ndim != 2
            or len(value) != vertex_count
            or not np.all(np.isfinite(value))
        ):
            raise ValueError(
                "Surface tiling cannot interpolate an invalid vertex attribute."
            )
        arrays[str(name)] = value
    return arrays


def _map_tile_uv(
    local_uv: np.ndarray,
    *,
    mode: str,
    seed: int,
    tile_u: int,
    tile_v: int,
    allow_quarter_turns: bool,
) -> np.ndarray:
    if mode == SURFACE_TILING_MODE_EDGE_VARIANTS:
        quadrant = np.asarray((tile_u % 2, tile_v % 2), dtype=float)
        return (local_uv + quadrant) * 0.5
    turns = quarter_turns_for_tile(
        tile_u, tile_v, seed, allow_odd_turns=allow_quarter_turns
    )
    u, v = local_uv[:, 0], local_uv[:, 1]
    if turns == 0:
        return local_uv
    if turns == 1:
        return np.column_stack((v, 1.0 - u))
    if turns == 2:
        return 1.0 - local_uv
    return np.column_stack((1.0 - v, u))
