"""Authored landing meshes, starter slabs and stable stair attachments."""

# ### Imports ###
from __future__ import annotations

import math
import os
import tempfile
import uuid
from collections.abc import Sequence
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import shapely
import trimesh
from shapely.geometry import Point, Polygon

from .level_coordinates import build_level_base_z_lookup, level_image_to_world_xy
from .models import (
    LandingAttachment,
    LandingData,
    LevelData,
    StairData,
    StairSectionData,
)

# ### Constants ###
_EPSILON = 1e-8
# Footprint/attachment XY coordinates follow the plan image (Y down), while
# provider and generation-preview geometry use physical local Z-up (Y up).
LANDING_IMAGE_TO_LOCAL_Z_UP_TRANSFORM = np.diag((1.0, -1.0, 1.0, 1.0))


# ### Landing construction and coordinate transforms ###
def create_landing(level: LevelData, x: float, y: float) -> LandingData:
    """Create a default slab without changing the level or its undo history."""

    if not isinstance(level, LevelData):
        raise TypeError("A landing requires an owning level.")
    return LandingData(center_x=x, center_y=y)


def landing_image_footprint(
    level: LevelData, landing: LandingData
) -> tuple[tuple[float, float], ...]:
    """Return the actual transformed polygon, not its enclosing rectangle."""

    return tuple(
        _local_to_image(level, landing, *point) for point in landing.footprint_meters
    )


def landing_world_footprint(
    level: LevelData, landing: LandingData
) -> tuple[tuple[float, float], ...]:
    return tuple(
        level_image_to_world_xy(level, *point)
        for point in landing_image_footprint(level, landing)
    )


def landing_world_transform(
    level: LevelData, landing: LandingData, levels: Sequence[LevelData]
) -> np.ndarray:
    """Map original local XY and slab Z (0..thickness) to world coordinates."""

    origin = np.asarray(
        level_image_to_world_xy(level, landing.center_x, landing.center_y)
    )
    matrix = np.eye(4, dtype=float)
    for axis, local in enumerate(((1.0, 0.0), (0.0, 1.0))):
        xy = level_image_to_world_xy(level, *_local_to_image(level, landing, *local))
        matrix[:2, axis] = np.asarray(xy) - origin
    matrix[2, 2] = landing.scale_z
    matrix[:2, 3] = origin
    matrix[2, 3] = _floor_top_z(level, levels) + landing.elevation_meters
    return matrix


def landing_top_world_z(
    level: LevelData, landing: LandingData, levels: Sequence[LevelData]
) -> float:
    return (
        _floor_top_z(level, levels)
        + landing.elevation_meters
        + (landing.walk_height_meters or landing.thickness_meters) * landing.scale_z
    )


def build_landing_mesh(
    level: LevelData, landing: LandingData, levels: Sequence[LevelData]
) -> trimesh.Trimesh:
    """Return combined geometry; use ``build_landing_meshes`` to retain PBR parts."""

    return trimesh.util.concatenate(build_landing_meshes(level, landing, levels))


def build_landing_meshes(
    level: LevelData, landing: LandingData, levels: Sequence[LevelData]
) -> tuple[trimesh.Trimesh, ...]:
    """Transform authored meshes without replacing their materials or topology."""

    if landing.generated_model_path is not None:
        meshes = load_landing_local_meshes(landing)
    else:
        meshes = (_build_landing_slab(landing),)
    transform = landing_world_transform(level, landing, levels)
    for mesh in meshes:
        mesh.apply_transform(transform)
        mesh.metadata.update(
            {
                "landing_id": landing.landing_id,
                "name": f"landing:{landing.landing_id}",
                "level_index": level.index,
            }
        )
    return meshes


def _build_landing_slab(landing: LandingData) -> trimesh.Trimesh:
    """Create the starter slab, before any Meshy geometry has been generated."""

    polygon = Polygon(landing.footprint_meters)
    vertices, faces, indices = [], [], {}
    for triangle in shapely.get_parts(shapely.constrained_delaunay_triangles(polygon)):
        coordinates = tuple(triangle.exterior.coords)[:-1]
        face = []
        for point in coordinates:
            if point not in indices:
                indices[point] = len(vertices)
                vertices.append(point)
            face.append(indices[point])
        faces.append(face)
    return trimesh.creation.extrude_triangulation(
        np.asarray(vertices), np.asarray(faces), landing.thickness_meters
    )


# ### Generated landing assets and physical profiles ###
def prepare_generated_landing(
    landing: LandingData,
    model_path: str | Path,
    *,
    output_path: str | Path | None = None,
    texture_only: bool = False,
) -> LandingData:
    """Fit a Meshy Y-up asset uniformly, preserving shape, details and PBR maps.

    The starter footprint supplies the otherwise unknown physical scale, not a
    square shape constraint. Stored GLBs use landing/image XY and Z-up, with
    their bottom centered at zero; authored transforms stay on ``LandingData``.
    """

    source = Path(model_path).resolve()
    meshes = _load_asset_meshes(source)
    y_up_to_z_up = trimesh.transformations.rotation_matrix(math.pi / 2, (1, 0, 0))
    for mesh in meshes:
        mesh.apply_transform(y_up_to_z_up)
        mesh.apply_transform(LANDING_IMAGE_TO_LOCAL_Z_UP_TRANSFORM)
    bounds = np.asarray(
        (
            np.min([mesh.bounds[0] for mesh in meshes], axis=0),
            np.max([mesh.bounds[1] for mesh in meshes], axis=0),
        )
    )
    extents = bounds[1] - bounds[0]
    if min(extents) <= _EPSILON:
        raise ValueError("The generated landing has no measurable 3D geometry.")
    old_points = np.asarray(landing.footprint_meters)
    target_size = float(np.max(np.ptp(old_points, axis=0)))
    original_bounds = None
    if texture_only and landing.generated_model_path is not None:
        original_meshes = load_landing_local_meshes(landing)
        original_bounds = np.asarray(
            (
                np.min([mesh.bounds[0] for mesh in original_meshes], axis=0),
                np.max([mesh.bounds[1] for mesh in original_meshes], axis=0),
            )
        )
        target_size = float(np.max(np.ptp(original_bounds[:, :2], axis=0)))
    factor = target_size / float(np.max(extents[:2]))
    origin = np.asarray((*np.mean(bounds[:, :2], axis=0), bounds[0, 2]))
    normalization = np.eye(4)
    normalization[:3, :3] *= factor
    normalization[:3, 3] = -origin * factor
    for mesh in meshes:
        mesh.apply_transform(normalization)
    footprint, walk_height = _generated_landing_profile(meshes)
    thickness = float(extents[2] * factor)
    if texture_only:
        tolerance = max(target_size * 0.005, 1e-5)
        if (
            Polygon(footprint).hausdorff_distance(Polygon(landing.footprint_meters))
            > tolerance
            or abs(thickness - landing.thickness_meters) > tolerance
            or abs(
                walk_height - (landing.walk_height_meters or landing.thickness_meters)
            )
            > tolerance
            or (
                original_bounds is not None
                and not np.allclose(
                    np.ptp(original_bounds, axis=0),
                    extents * factor,
                    atol=tolerance,
                    rtol=0,
                )
            )
        ):
            raise ValueError(
                "The textured landing changed its geometry. Generate geometry to replace its shape."
            )
        footprint = landing.footprint_meters
        walk_height = landing.walk_height_meters
        thickness = landing.thickness_meters
    destination = (
        Path(output_path).resolve()
        if output_path is not None
        else source.with_name(f"{source.stem}.landing-{uuid.uuid4().hex}.glb")
    )
    scene = trimesh.Scene()
    scene.metadata["housemaker_landing_local_z_up"] = True
    scene.metadata["housemaker_landing_image_xy_basis"] = True
    for index, mesh in enumerate(meshes):
        scene.add_geometry(
            mesh, node_name=f"landing_part_{index}", geom_name=f"landing_part_{index}"
        )
    _write_landing_asset(destination, scene.export(file_type="glb"))
    return replace(
        landing,
        generated_model_path=str(destination),
        footprint_meters=footprint,
        thickness_meters=thickness,
        walk_height_meters=walk_height,
    )


def load_landing_local_meshes(landing: LandingData) -> tuple[trimesh.Trimesh, ...]:
    """Return independent canonical meshes; repeated live transforms use a cache."""

    if landing.generated_model_path is None:
        return (_build_landing_slab(landing),)
    path = Path(landing.generated_model_path).resolve()
    try:
        stat = path.stat()
    except OSError as error:
        raise ValueError(
            f"The generated landing mesh is unavailable: {path}"
        ) from error
    return tuple(
        mesh.copy()
        for mesh in _load_landing_asset_cached(
            str(path), stat.st_mtime_ns, stat.st_size
        )
    )


@lru_cache(maxsize=32)
def _load_landing_asset_cached(path: str, _modified_ns: int, _size: int):
    return _load_asset_meshes(Path(path))


def _load_asset_meshes(path: Path) -> tuple[trimesh.Trimesh, ...]:
    try:
        scene = trimesh.load_scene(path, process=False)
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(
            f"The generated landing mesh could not be loaded: {path}"
        ) from error
    meshes = []
    for node in sorted(scene.graph.nodes_geometry, key=str):
        transform, geometry_name = scene.graph[node]
        geometry = scene.geometry[geometry_name]
        if not isinstance(geometry, trimesh.Trimesh) or not len(geometry.faces):
            continue
        mesh = geometry.copy()
        mesh.apply_transform(transform)
        if not np.isfinite(mesh.vertices).all():
            raise ValueError(
                "The generated landing contains invalid vertex coordinates."
            )
        meshes.append(mesh)
    if not meshes:
        raise ValueError(
            "The generated landing mesh does not contain any triangle faces."
        )
    return tuple(meshes)


def _generated_landing_profile(meshes):
    triangles = np.concatenate([mesh.triangles for mesh in meshes])
    edges_a = triangles[:, 1] - triangles[:, 0]
    edges_b = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(edges_a, edges_b)
    projected_area = np.abs(normals[:, 2]) / 2
    useful = projected_area > _EPSILON**2
    if not useful.any():
        raise ValueError("The generated landing has no measurable top-down footprint.")
    projected = shapely.polygons(triangles[useful, :, :2])
    outline = shapely.union_all(projected, grid_size=1e-7)
    polygons = [
        part for part in shapely.get_parts(outline) if isinstance(part, Polygon)
    ]
    if not polygons:
        raise ValueError("The generated landing footprint could not be reconstructed.")
    polygon = max(polygons, key=lambda part: part.area)
    tolerance = 1e-6
    while len(polygon.exterior.coords) > 513:
        polygon = polygon.simplify(tolerance, preserve_topology=True)
        tolerance *= 2
    footprint = tuple((float(x), float(y)) for x, y in polygon.exterior.coords[:-1])
    height = float(np.max(triangles[:, :, 2]))
    lengths = np.linalg.norm(normals, axis=1)
    horizontal = normals[:, 2] > lengths * 0.9
    if not horizontal.any():
        return footprint, height
    centers = np.mean(triangles[horizontal, :, 2], axis=1)
    weights = projected_area[horizontal]
    buckets = np.rint(centers / max(height * 0.005, 0.001)).astype(np.int64)
    ids, inverse = np.unique(buckets, return_inverse=True)
    areas = np.bincount(inverse, weights=weights, minlength=len(ids))
    chosen = inverse == int(np.argmax(areas))
    walk_height = float(np.average(centers[chosen], weights=weights[chosen]))
    return footprint, walk_height if walk_height > _EPSILON else height


def _write_landing_asset(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".tmp", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(data)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


# ### Stable attachment coordinates and snapping ###
def landing_attachment_for_image_point(
    level: LevelData, landing: LandingData, point: tuple[float, float]
) -> LandingAttachment:
    """Record a point in pre-scale local meters so all transforms carry it."""

    local = _image_to_local(level, landing, *point)
    if not Polygon(landing.footprint_meters).buffer(_EPSILON).covers(Point(local)):
        raise ValueError("A stair attachment must lie on the landing footprint.")
    return LandingAttachment(landing.landing_id, *local)


def resolve_landing_attachment(
    level: LevelData, attachment: LandingAttachment, levels: Sequence[LevelData]
) -> tuple[tuple[float, float], float]:
    landing = next(
        (item for item in level.landings if item.landing_id == attachment.landing_id),
        None,
    )
    if landing is None:
        raise ValueError("The stair's attached landing no longer exists.")
    return (
        _attachment_image_point(level, landing, attachment),
        landing_top_world_z(level, landing, levels),
    )


def find_landing_snap(
    level: LevelData, point: tuple[float, float], tolerance_pixels: float
) -> tuple[tuple[float, float], LandingAttachment] | None:
    """Snap to the closest real polygon edge/corner within image-space reach."""

    if not math.isfinite(tolerance_pixels) or tolerance_pixels < 0.0:
        return None
    pointer = Point(point)
    candidates = []
    for landing in level.landings:
        boundary = Polygon(landing_image_footprint(level, landing)).boundary
        nearest = boundary.interpolate(boundary.project(pointer))
        distance = nearest.distance(pointer)
        if distance <= tolerance_pixels:
            xy = float(nearest.x), float(nearest.y)
            candidates.append((distance, landing.landing_id, xy, landing))
    if not candidates:
        return None
    _, _, xy, landing = min(candidates, key=lambda item: item[:2])
    return xy, landing_attachment_for_image_point(level, landing, xy)


def attach_stair_point_to_landing(
    stair: StairData,
    section_index: int,
    point_name: str,
    level: LevelData,
    landing: LandingData,
    point: tuple[float, float] | None = None,
) -> StairData:
    """Return one stair with a bound A/B point; do not change any collection."""

    if point_name not in {"a", "b"} or not 0 <= section_index < len(stair.sections):
        raise ValueError("A stair attachment requires an existing A/B section point.")
    section = stair.sections[section_index]
    if section.level_index != level.index:
        raise ValueError("A stair point must attach to a landing on its own level.")
    point = section.points[0 if point_name == "a" else 1] if point is None else point
    attachment = landing_attachment_for_image_point(level, landing, point)
    if section_index in (0, len(stair.sections) - 1):
        prefix = "start" if section_index == 0 else "end"
        return replace(
            stair,
            **{
                f"{prefix}_{point_name}_x": point[0],
                f"{prefix}_{point_name}_y": point[1],
                f"{prefix}_{point_name}_vertex_id": None,
                f"{prefix}_{point_name}_landing": attachment,
            },
        )
    sections = list(stair.intermediate_sections)
    sections[section_index - 1] = replace(
        section,
        **{
            f"{point_name}_x": point[0],
            f"{point_name}_y": point[1],
            f"{point_name}_vertex_id": None,
            f"{point_name}_landing": attachment,
        },
    )
    return replace(stair, intermediate_sections=tuple(sections))


def resolve_stair_section_image_points(
    level: LevelData, section: StairSectionData
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Resolve landing bindings before wall bindings, retaining legacy fallbacks."""

    points = []
    for name in ("a", "b"):
        attachment = getattr(section, f"{name}_landing")
        landing = next(
            (
                item
                for item in level.landings
                if attachment is not None and item.landing_id == attachment.landing_id
            ),
            None,
        )
        vertex = level.vertex_data.get_vertex(getattr(section, f"{name}_vertex_id"))
        if landing is not None:
            point = _attachment_image_point(level, landing, attachment)
        elif vertex is not None:
            point = vertex.x, vertex.y
        else:
            point = getattr(section, f"{name}_x"), getattr(section, f"{name}_y")
        points.append(point)
    return tuple(points)


def stair_section_landing_z(
    level: LevelData, section: StairSectionData, levels: Sequence[LevelData]
) -> float | None:
    """Use a bound landing's top as the complete cross-section's elevation."""

    heights = [
        resolve_landing_attachment(level, attachment, levels)[1]
        for attachment in (section.a_landing, section.b_landing)
        if attachment is not None
    ]
    if not heights:
        return None
    if max(heights) - min(heights) > _EPSILON:
        raise ValueError(
            "A stair cross-section cannot span landings of different heights."
        )
    return heights[0]


# ### Local coordinate helpers ###
def _attachment_image_point(level, landing, attachment):
    local = Point(attachment.local_x_meters, attachment.local_y_meters)
    polygon = Polygon(landing.footprint_meters)
    if not polygon.buffer(_EPSILON).covers(local):
        # Shape regeneration may remove the former boundary. Keep the stable
        # saved anchor, but project its evaluated position to the new profile;
        # undoing the shape edit restores the original attachment exactly.
        boundary = polygon.boundary
        local = boundary.interpolate(boundary.project(local))
    return _local_to_image(level, landing, local.x, local.y)


def _local_to_image(level, landing, x, y):
    angle = math.radians(landing.rotation_degrees)
    x, y = x * landing.scale_x, y * landing.scale_y
    return (
        landing.center_x
        + (x * math.cos(angle) - y * math.sin(angle)) / level.meters_per_pixel,
        landing.center_y
        + (x * math.sin(angle) + y * math.cos(angle)) / level.meters_per_pixel,
    )


def _image_to_local(level, landing, x, y):
    x, y = (
        (x - landing.center_x) * level.meters_per_pixel,
        (y - landing.center_y) * level.meters_per_pixel,
    )
    angle = math.radians(landing.rotation_degrees)
    return (
        (x * math.cos(angle) + y * math.sin(angle)) / landing.scale_x,
        (-x * math.sin(angle) + y * math.cos(angle)) / landing.scale_y,
    )


def _floor_top_z(level, levels):
    heights = build_level_base_z_lookup(levels)
    if level.index not in heights:
        raise ValueError("The landing's level is missing from the scene.")
    return heights[level.index]
