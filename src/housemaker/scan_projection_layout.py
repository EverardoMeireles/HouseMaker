# ### Imports ###
from __future__ import annotations

import numpy as np


# ### Public constants ###
SCAN_PROJECTION_LAYOUT_ALIGNMENT_PIXELS = 4
SCAN_PROJECTION_LAYOUT_METADATA_KEY = "housemaker_scan_projection_layout"
SCAN_PROJECTION_UV_EDGE_INSET_TEXELS = 0.5


# ### Metadata helpers ###
def build_scan_projection_layout_metadata(
    *,
    canonical_texture_resolution: int,
    version: int,
) -> dict[str, object]:
    """Build JSON-safe layout data used by exact-resolution GLB variants."""

    return {
        "version": int(version),
        "canonical_texture_resolution": int(canonical_texture_resolution),
        "uv_texture_resolution": int(canonical_texture_resolution),
    }


def clear_scan_projection_layout_metadata(scene: object) -> None:
    """Remove scan-only UV provenance after a different UV transform."""

    geometries = getattr(scene, "geometry", None)
    if not isinstance(geometries, dict):
        return
    for geometry in geometries.values():
        metadata = getattr(geometry, "metadata", None)
        if isinstance(metadata, dict):
            metadata.pop(SCAN_PROJECTION_LAYOUT_METADATA_KEY, None)


# ### Scene remapping API ###
def remap_scan_projection_scene_uvs(
    scene: object,
    target_texture_resolution: int,
) -> bool:
    """Align tagged scan-island edges to texel centers at one resolution."""

    geometries = getattr(scene, "geometry", None)
    if not isinstance(geometries, dict):
        return False
    remapped = False
    for geometry in geometries.values():
        metadata = getattr(geometry, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        raw_layout = metadata.get(SCAN_PROJECTION_LAYOUT_METADATA_KEY)
        if raw_layout is None:
            continue
        _remap_scan_projection_mesh_uvs(
            geometry,
            raw_layout,
            target_texture_resolution,
        )
        remapped = True
    return remapped


# ### Mesh remapping helpers ###
def _remap_scan_projection_mesh_uvs(
    mesh: object,
    raw_layout: object,
    target_texture_resolution: int,
) -> None:
    if not isinstance(raw_layout, dict):
        raise ValueError("The scan-projection layout metadata is invalid.")
    canonical_resolution = _positive_integer(
        raw_layout.get("canonical_texture_resolution"),
        "canonical texture resolution",
    )
    target_resolution = _positive_integer(
        target_texture_resolution,
        "target texture resolution",
    )
    if canonical_resolution % target_resolution:
        raise ValueError(
            "A scan-projection texture variant must divide its canonical "
            "resolution exactly."
        )
    reduction = canonical_resolution // target_resolution
    if reduction > SCAN_PROJECTION_LAYOUT_ALIGNMENT_PIXELS:
        raise ValueError(
            "A scan-projection texture variant exceeds the recorded layout "
            "alignment."
        )
    current_uv_resolution = _positive_integer(
        raw_layout.get("uv_texture_resolution", canonical_resolution),
        "current UV texture resolution",
    )
    if canonical_resolution % current_uv_resolution:
        raise ValueError(
            "The recorded scan-projection UV resolution does not divide its "
            "canonical resolution."
        )
    current_reduction = canonical_resolution // current_uv_resolution

    vertices = np.asarray(getattr(mesh, "vertices", None), dtype=float)
    faces = np.asarray(getattr(mesh, "faces", None), dtype=np.int64)
    visual = getattr(mesh, "visual", None)
    source_uvs = np.asarray(getattr(visual, "uv", None), dtype=float)
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or faces.ndim != 2
        or faces.shape[1:] != (3,)
        or source_uvs.shape != (len(vertices), 2)
        or not np.all(np.isfinite(source_uvs))
    ):
        raise ValueError("The tagged scan-projection mesh has invalid UVs.")
    if current_uv_resolution == target_resolution:
        current_inset = (
            current_reduction * SCAN_PROJECTION_UV_EDGE_INSET_TEXELS
        )
        for face in faces:
            source_points = np.column_stack(
                (
                    source_uvs[face, 0] * canonical_resolution,
                    (1.0 - source_uvs[face, 1]) * canonical_resolution,
                )
            )
            _reconstruct_face_cell_rectangle(
                source_points,
                inset_pixels=current_inset,
                canonical_resolution=canonical_resolution,
                required_alignment=current_reduction,
            )
        next_layout = dict(raw_layout)
        next_layout["uv_texture_resolution"] = target_resolution
        mesh.metadata[SCAN_PROJECTION_LAYOUT_METADATA_KEY] = next_layout
        return

    output_uvs = source_uvs.copy()
    assigned_uvs: dict[int, np.ndarray] = {}
    current_inset = current_reduction * SCAN_PROJECTION_UV_EDGE_INSET_TEXELS
    for face in faces:
        source_points = np.column_stack(
            (
                source_uvs[face, 0] * canonical_resolution,
                (1.0 - source_uvs[face, 1]) * canonical_resolution,
            )
        )
        rectangle = _reconstruct_face_cell_rectangle(
            source_points,
            inset_pixels=current_inset,
            canonical_resolution=canonical_resolution,
            required_alignment=max(reduction, current_reduction),
        )
        x, y, width, height = rectangle
        current_width = width - 2.0 * current_inset
        current_height = height - 2.0 * current_inset
        if current_width <= 0.0 or current_height <= 0.0:
            raise ValueError(
                "A collapsed scan-projection UV cell cannot be expanded to "
                "a larger texture resolution."
            )
        normalized_x = (source_points[:, 0] - (x + current_inset)) / current_width
        normalized_y = (source_points[:, 1] - (y + current_inset)) / current_height
        target_inset = reduction * SCAN_PROJECTION_UV_EDGE_INSET_TEXELS
        target_points = np.column_stack(
            (
                x + target_inset + normalized_x * (width - 2.0 * target_inset),
                y + target_inset + normalized_y * (height - 2.0 * target_inset),
            )
        )
        next_uvs = np.column_stack(
            (
                target_points[:, 0] / canonical_resolution,
                1.0 - target_points[:, 1] / canonical_resolution,
            )
        )
        for vertex_index, next_uv in zip(face, next_uvs, strict=True):
            normalized_index = int(vertex_index)
            existing = assigned_uvs.get(normalized_index)
            if existing is not None and not np.allclose(
                existing,
                next_uv,
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError(
                    "A tagged scan-projection vertex belongs to conflicting "
                    "atlas cells."
                )
            assigned_uvs[normalized_index] = next_uv
            output_uvs[normalized_index] = next_uv
    visual.uv = output_uvs
    next_layout = dict(raw_layout)
    next_layout["uv_texture_resolution"] = target_resolution
    mesh.metadata[SCAN_PROJECTION_LAYOUT_METADATA_KEY] = next_layout


def _reconstruct_face_cell_rectangle(
    source_points: np.ndarray,
    *,
    inset_pixels: float,
    canonical_resolution: int,
    required_alignment: int,
) -> tuple[int, int, int, int]:
    minimums = np.min(source_points, axis=0) - inset_pixels
    maximums = np.max(source_points, axis=0) + inset_pixels
    x, y = (int(round(float(value))) for value in minimums)
    right, bottom = (int(round(float(value))) for value in maximums)
    rectangle = (x, y, right - x, bottom - y)
    if (
        min(rectangle) < 0
        or right > canonical_resolution
        or bottom > canonical_resolution
        or rectangle[2] < required_alignment
        or rectangle[3] < required_alignment
        or any(value % required_alignment for value in rectangle)
        or not np.allclose(minimums, (x, y), rtol=0.0, atol=1e-3)
        or not np.allclose(maximums, (right, bottom), rtol=0.0, atol=1e-3)
    ):
        raise ValueError(
            "A tagged scan-projection face no longer matches its aligned "
            "atlas cell."
        )
    return rectangle


# ### Scalar validation helpers ###
def _positive_integer(value: object, description: str) -> int:
    normalized = _nonnegative_integer(value, description)
    if normalized <= 0:
        raise ValueError(f"The {description} must be positive.")
    return normalized


def _nonnegative_integer(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"The {description} must be an integer.")
    if value < 0:
        raise ValueError(f"The {description} cannot be negative.")
    return int(value)
