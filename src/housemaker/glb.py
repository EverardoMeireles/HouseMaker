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
PIXEL_TO_METER = 0.02
Z_UP_TO_GLTF_Y_UP_TRANSFORM = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)

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
    blueprint_size_pixels: tuple[float, float] | None = None,
) -> GeneratedModel:
    if isinstance(level_source, VertexData):
        wall_meshes = _build_level_meshes(
            vertex_data=level_source,
            wall_height_meters=wall_height_meters,
            base_z_meters=0.0,
            blueprint_size_pixels=blueprint_size_pixels,
        )
        named_meshes = _build_named_meshes_for_single_level(wall_meshes)
    else:
        named_meshes = _build_multi_level_meshes(
            level_source,
            blueprint_size_pixels=blueprint_size_pixels,
        )
        wall_meshes = [named_mesh.mesh for named_mesh in named_meshes]

    if not wall_meshes:
        raise ValueError("The current blueprint data does not contain usable edges.")

    combined_mesh = trimesh.util.concatenate(wall_meshes)
    scene = _build_export_scene(named_meshes)
    glb_bytes = scene.export(file_type="glb")
    return GeneratedModel(mesh=combined_mesh, scene=scene, glb_bytes=glb_bytes)


def export_glb_file(model: GeneratedModel, path: str | Path) -> Path:
    export_path = Path(path)
    export_path.write_bytes(model.glb_bytes)
    return export_path


# ### Internal helpers ###
def _build_export_scene(named_meshes: list[NamedMesh]) -> trimesh.Scene:
    scene = trimesh.Scene()
    for named_mesh in named_meshes:
        scene.add_geometry(
            _to_gltf_y_up_mesh(named_mesh.mesh),
            geom_name=named_mesh.name,
            node_name=named_mesh.name,
        )
    return scene


def _to_gltf_y_up_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    export_mesh = mesh.copy()
    export_mesh.apply_transform(Z_UP_TO_GLTF_Y_UP_TRANSFORM)
    return export_mesh


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


def _build_multi_level_meshes(
    levels: Sequence[LevelData],
    blueprint_size_pixels: tuple[float, float] | None,
) -> list[NamedMesh]:
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
            blueprint_size_pixels=level.image_size_pixels or blueprint_size_pixels,
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
    blueprint_size_pixels: tuple[float, float] | None,
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
                blueprint_size_pixels=blueprint_size_pixels,
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
    blueprint_size_pixels: tuple[float, float] | None,
) -> trimesh.Trimesh | None:
    start_vertex = vertex_lookup.get(edge.start_vertex_id)
    end_vertex = vertex_lookup.get(edge.end_vertex_id)
    if start_vertex is None or end_vertex is None:
        return None

    start_xy = _vertex_to_world_xy(start_vertex, blueprint_size_pixels)
    end_xy = _vertex_to_world_xy(end_vertex, blueprint_size_pixels)

    direction = end_xy - start_xy
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return None

    wall_mesh = trimesh.Trimesh(
        vertices=[
            [0.0, 0.0, -wall_height_meters / 2.0],
            [length, 0.0, -wall_height_meters / 2.0],
            [length, 0.0, wall_height_meters / 2.0],
            [0.0, 0.0, wall_height_meters / 2.0],
        ],
        faces=[
            [0, 1, 2],
            [0, 2, 3],
            [2, 1, 0],
            [3, 2, 0],
        ],
        process=False,
    )
    angle_radians = float(np.arctan2(direction[1], direction[0]))
    transform = trimesh.transformations.rotation_matrix(angle_radians, [0.0, 0.0, 1.0])
    transform[:3, 3] = [
        float(start_xy[0]),
        float(start_xy[1]),
        base_z_meters + wall_height_meters / 2.0,
    ]

    wall_mesh.apply_transform(transform)
    return wall_mesh


def _vertex_to_world_xy(
    vertex: Vertex,
    blueprint_size_pixels: tuple[float, float] | None,
) -> np.ndarray:
    if blueprint_size_pixels is None:
        centered_x = vertex.x
        centered_y = vertex.y
    else:
        blueprint_width, blueprint_height = blueprint_size_pixels
        centered_x = vertex.x - blueprint_width / 2.0
        centered_y = vertex.y - blueprint_height / 2.0

    return np.array(
        [
            centered_x * PIXEL_TO_METER,
            -centered_y * PIXEL_TO_METER,
        ],
        dtype=float,
    )
