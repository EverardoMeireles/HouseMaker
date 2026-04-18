# ### Imports ###
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from housemaker.models import Edge, Vertex, VertexData

# ### Constants ###
DEFAULT_WALL_HEIGHT_METERS = 3.0
DEFAULT_WALL_THICKNESS_METERS = 0.18
PIXEL_TO_METER = 0.02

# ### Data models ###
@dataclass
class GeneratedModel:
    mesh: trimesh.Trimesh
    scene: trimesh.Scene
    glb_bytes: bytes


# ### Public helpers ###
def convert_to_glb(
    vertex_data: VertexData,
    wall_height_meters: float = DEFAULT_WALL_HEIGHT_METERS,
) -> GeneratedModel:
    if not vertex_data.edges:
        raise ValueError("Create at least one edge before converting to GLB.")
    if wall_height_meters <= 0.0:
        raise ValueError("Height level must be greater than zero.")

    vertex_lookup = {vertex.id: vertex for vertex in vertex_data.vertices}
    wall_meshes = [
        wall_mesh
        for edge in vertex_data.edges
        if (
            wall_mesh := _build_wall_mesh(
                edge,
                vertex_lookup,
                wall_height_meters=wall_height_meters,
            )
        )
        is not None
    ]

    if not wall_meshes:
        raise ValueError("The current blueprint data does not contain usable edges.")

    combined_mesh = trimesh.util.concatenate(wall_meshes)
    scene = trimesh.Scene()
    scene.add_geometry(combined_mesh, geom_name="walls")
    glb_bytes = scene.export(file_type="glb")
    return GeneratedModel(mesh=combined_mesh, scene=scene, glb_bytes=glb_bytes)


def export_glb_file(model: GeneratedModel, path: str | Path) -> Path:
    export_path = Path(path)
    export_path.write_bytes(model.glb_bytes)
    return export_path


# ### Internal helpers ###
def _build_wall_mesh(
    edge: Edge,
    vertex_lookup: dict[int, Vertex],
    wall_height_meters: float,
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
        wall_height_meters / 2.0,
    ]

    wall_mesh.apply_transform(transform)
    return wall_mesh
