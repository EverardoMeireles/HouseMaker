# ### Imports ###
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from housemaker.models import (
    DEFAULT_LEVEL_HEIGHT_METERS,
    GROUND_LEVEL_INDEX,
    Edge,
    LevelData,
    Vertex,
    VertexData,
)

# ### Constants ###
DEFAULT_WALL_HEIGHT_METERS = DEFAULT_LEVEL_HEIGHT_METERS
DEFAULT_WALL_THICKNESS_METERS = 0.18
PIXEL_TO_METER = 0.02

# ### Data models ###
@dataclass
class GeneratedModel:
    mesh: trimesh.Trimesh
    scene: trimesh.Scene
    glb_bytes: bytes


@dataclass
class NamedMesh:
    name: str
    mesh: trimesh.Trimesh


# ### Public helpers ###
def convert_to_glb(
    level_source: VertexData | Sequence[LevelData],
    wall_height_meters: float = DEFAULT_WALL_HEIGHT_METERS,
) -> GeneratedModel:
    if isinstance(level_source, VertexData):
        wall_meshes = _build_level_meshes(
            vertex_data=level_source,
            wall_height_meters=wall_height_meters,
            base_z_meters=0.0,
        )
        named_meshes = _build_named_meshes_for_single_level(wall_meshes)
    else:
        named_meshes = _build_multi_level_meshes(level_source)
        wall_meshes = [named_mesh.mesh for named_mesh in named_meshes]

    if not wall_meshes:
        raise ValueError("The current blueprint data does not contain usable edges.")

    combined_mesh = trimesh.util.concatenate(wall_meshes)
    scene = trimesh.Scene()
    for named_mesh in named_meshes:
        scene.add_geometry(
            named_mesh.mesh,
            geom_name=named_mesh.name,
            node_name=named_mesh.name,
        )
    glb_bytes = scene.export(file_type="glb")
    return GeneratedModel(mesh=combined_mesh, scene=scene, glb_bytes=glb_bytes)


def export_glb_file(model: GeneratedModel, path: str | Path) -> Path:
    export_path = Path(path)
    export_path.write_bytes(model.glb_bytes)
    return export_path


# ### Internal helpers ###
def _build_named_meshes_for_single_level(
    wall_meshes: list[trimesh.Trimesh],
) -> list[NamedMesh]:
    if not wall_meshes:
        return []

    return [
        NamedMesh(
            name="level_2_ground",
            mesh=trimesh.util.concatenate(wall_meshes),
        )
    ]


def _build_multi_level_meshes(levels: Sequence[LevelData]) -> list[NamedMesh]:
    if not levels:
        raise ValueError("No levels are available for GLB conversion.")

    sorted_levels = sorted(levels, key=lambda level: level.index)
    level_lookup = {level.index: level for level in sorted_levels}
    named_meshes: list[NamedMesh] = []

    for level in sorted_levels:
        if level.height_meters <= 0.0:
            raise ValueError(f"Level {level.index} height must be greater than zero.")

        wall_meshes = _build_level_meshes(
            vertex_data=level.vertex_data,
            wall_height_meters=level.height_meters,
            base_z_meters=_get_level_base_z(level_lookup, level.index),
        )
        if not wall_meshes:
            continue

        named_meshes.append(
            NamedMesh(
                name=_get_level_object_name(level),
                mesh=trimesh.util.concatenate(wall_meshes),
            )
        )

    return named_meshes


def _build_level_meshes(
    vertex_data: VertexData,
    wall_height_meters: float,
    base_z_meters: float,
) -> list[trimesh.Trimesh]:
    if wall_height_meters <= 0.0:
        raise ValueError("Height level must be greater than zero.")

    vertex_lookup = {vertex.id: vertex for vertex in vertex_data.vertices}
    return [
        wall_mesh
        for edge in vertex_data.edges
        if (
            wall_mesh := _build_wall_mesh(
                edge=edge,
                vertex_lookup=vertex_lookup,
                wall_height_meters=wall_height_meters,
                base_z_meters=base_z_meters,
            )
        )
        is not None
    ]


def _get_level_base_z(
    level_lookup: dict[int, LevelData],
    level_index: int,
) -> float:
    if level_index >= GROUND_LEVEL_INDEX:
        return sum(
            level_lookup[index].height_meters
            for index in range(GROUND_LEVEL_INDEX, level_index)
            if index in level_lookup
        )

    return -sum(
        level_lookup[index].height_meters
        for index in range(level_index, GROUND_LEVEL_INDEX)
        if index in level_lookup
    )


def _get_level_object_name(level: LevelData) -> str:
    return level.display_name.lower().replace(" ", "_")


def _build_wall_mesh(
    edge: Edge,
    vertex_lookup: dict[int, Vertex],
    wall_height_meters: float,
    base_z_meters: float,
) -> trimesh.Trimesh | None:
    start_vertex = vertex_lookup.get(edge.start_vertex_id)
    end_vertex = vertex_lookup.get(edge.end_vertex_id)
    if start_vertex is None or end_vertex is None:
        return None

    start_xy = np.array(
        [start_vertex.x * PIXEL_TO_METER, -start_vertex.y * PIXEL_TO_METER],
        dtype=float,
    )
    end_xy = np.array(
        [end_vertex.x * PIXEL_TO_METER, -end_vertex.y * PIXEL_TO_METER],
        dtype=float,
    )

    direction = end_xy - start_xy
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return None

    wall_mesh = trimesh.creation.box(
        extents=(length, DEFAULT_WALL_THICKNESS_METERS, wall_height_meters)
    )
    angle_radians = float(np.arctan2(direction[1], direction[0]))
    transform = trimesh.transformations.rotation_matrix(angle_radians, [0.0, 0.0, 1.0])
    midpoint = (start_xy + end_xy) / 2.0
    transform[:3, 3] = [
        float(midpoint[0]),
        float(midpoint[1]),
        base_z_meters + wall_height_meters / 2.0,
    ]

    wall_mesh.apply_transform(transform)
    return wall_mesh
