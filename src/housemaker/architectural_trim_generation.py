# ### Imports ###
from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import trimesh

from housemaker.architectural_trim import (
    TRIM_HANDLE_DEPTH,
    build_architectural_trim_edit_targets,
    build_architectural_trim_geometry,
)
from housemaker.glb import (
    Z_UP_TO_GLTF_Y_UP_TRANSFORM,
    GeneratedModel,
    PlacedGeneratedModel,
)
from housemaker.models import ArchitecturalTrimData, LevelData

# ### Constants ###
TRIM_COMPONENT_GEOMETRY_EPSILON = 1e-8
TRIM_COMPONENT_NODE_PREFIX = "architectural_trim_component"


# ### Component frame model ###
@dataclass(frozen=True)
class ArchitecturalTrimComponentFrame:
    """Canonical local-to-world fit for one generated trim component."""

    level_index: int
    trim_id: str
    world_bottom_center: tuple[float, float, float]
    yaw_degrees: float
    target_extents: tuple[float, float, float]


# ### Public component builders ###
def build_architectural_trim_component_seed_model(
    level: LevelData,
    trim: ArchitecturalTrimData,
    wall_surfaces: Sequence[object],
) -> tuple[GeneratedModel, ArchitecturalTrimComponentFrame]:
    """Build the selected procedural trim in its stable local generation frame."""

    world_mesh, frame_basis = _build_trim_world_mesh_and_basis(
        level,
        trim,
        wall_surfaces,
    )
    local_mesh, frame = _move_trim_mesh_into_local_frame(
        world_mesh,
        frame_basis,
        level_index=level.index,
        trim_id=trim.trim_id,
    )
    return _build_generated_model(local_mesh, trim.trim_id), frame


def build_architectural_trim_component_frame(
    level: LevelData,
    trim: ArchitecturalTrimData,
    wall_surfaces: Sequence[object],
) -> ArchitecturalTrimComponentFrame:
    """Resolve the current wall-bound frame without retaining procedural meshes."""

    world_mesh, frame_basis = _build_trim_world_mesh_and_basis(
        level,
        trim,
        wall_surfaces,
    )
    _local_mesh, frame = _move_trim_mesh_into_local_frame(
        world_mesh,
        frame_basis,
        level_index=level.index,
        trim_id=trim.trim_id,
    )
    return frame


def build_architectural_trim_component_placement(
    model: GeneratedModel,
    frame: ArchitecturalTrimComponentFrame,
    *,
    object_id: str,
    object_name: str,
) -> PlacedGeneratedModel:
    """Fit one generated model to its authored trim span in the scene."""

    if not isinstance(model, GeneratedModel):
        raise TypeError("Architectural trim components require a generated model.")
    bounds = np.asarray(model.mesh.bounds, dtype=float)
    if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
        raise ValueError("The generated architectural trim has invalid bounds.")
    source_extents = bounds[1] - bounds[0]
    if np.any(source_extents <= TRIM_COMPONENT_GEOMETRY_EPSILON):
        raise ValueError(
            "The generated architectural trim has no measurable size on every axis."
        )
    target_extents = np.asarray(frame.target_extents, dtype=float)
    axis_scales = target_extents / source_extents
    if not np.all(np.isfinite(axis_scales)) or np.any(axis_scales <= 0.0):
        raise ValueError("The architectural trim fit produced invalid scales.")
    return PlacedGeneratedModel(
        object_id=str(object_id).strip(),
        source_object_id=str(object_id).strip(),
        object_name=str(object_name).strip(),
        model=model,
        world_position=frame.world_bottom_center,
        rotation_degrees=(0.0, 0.0, frame.yaw_degrees),
        axis_scales=tuple(float(value) for value in axis_scales),
    )


def remove_generated_architectural_trims_from_levels(
    levels: Sequence[LevelData],
    generated_trim_keys: Sequence[tuple[int, str]],
) -> list[LevelData]:
    """Copy levels while omitting procedural trims replaced by generated models."""

    generated_keys = {
        (int(level_index), str(trim_id).strip().lower())
        for level_index, trim_id in generated_trim_keys
    }
    effective_levels: list[LevelData] = []
    for level in levels:
        copied_level = copy.copy(level)
        copied_level.architectural_trims = [
            trim
            for trim in level.architectural_trims
            if (level.index, trim.trim_id) not in generated_keys
        ]
        effective_levels.append(copied_level)
    return effective_levels


# ### Geometry-frame helpers ###
def _build_trim_world_mesh_and_basis(
    level: LevelData,
    trim: ArchitecturalTrimData,
    wall_surfaces: Sequence[object],
) -> tuple[trimesh.Trimesh, np.ndarray]:
    if not isinstance(level, LevelData) or not isinstance(
        trim,
        ArchitecturalTrimData,
    ):
        raise TypeError("Architectural trim generation requires level and trim data.")
    preview_level = copy.copy(level)
    preview_level.architectural_trims = [trim]
    geometry = build_architectural_trim_geometry(
        (preview_level,),
        wall_surfaces,
    )
    owned_parts = tuple(
        part for part in geometry.parts if part.trim_id == trim.trim_id
    )
    if not owned_parts:
        raise ValueError("The selected architectural trim has no usable geometry.")
    world_mesh = trimesh.util.concatenate(
        tuple(part.mesh.copy() for part in owned_parts)
    )
    if world_mesh.is_empty:
        raise ValueError("The selected architectural trim has no usable geometry.")

    edit_target = next(
        (
            target
            for target in build_architectural_trim_edit_targets(
                (preview_level,),
                wall_surfaces,
            )
            if target.trim_id == trim.trim_id
        ),
        None,
    )
    depth_handle = (
        None
        if edit_target is None
        else next(
            (
                handle
                for handle in edit_target.handles
                if handle.handle_kind == TRIM_HANDLE_DEPTH
            ),
            None,
        )
    )
    if depth_handle is None:
        raise ValueError("The selected architectural trim has no stable wall frame.")
    depth_axis = np.asarray(depth_handle.axis_world, dtype=float)
    depth_axis[2] = 0.0
    depth_length = float(np.linalg.norm(depth_axis))
    if depth_length <= TRIM_COMPONENT_GEOMETRY_EPSILON:
        raise ValueError("The selected architectural trim has no horizontal depth.")
    depth_axis /= depth_length
    up_axis = np.asarray((0.0, 0.0, 1.0), dtype=float)
    width_axis = np.cross(depth_axis, up_axis)
    width_axis /= np.linalg.norm(width_axis)
    frame_basis = np.column_stack((width_axis, depth_axis, up_axis))
    if not np.allclose(
        frame_basis.T @ frame_basis,
        np.eye(3, dtype=float),
        atol=1e-7,
    ):
        raise ValueError("The selected architectural trim frame is invalid.")
    return world_mesh, frame_basis


def _move_trim_mesh_into_local_frame(
    world_mesh: trimesh.Trimesh,
    frame_basis: np.ndarray,
    *,
    level_index: int,
    trim_id: str,
) -> tuple[trimesh.Trimesh, ArchitecturalTrimComponentFrame]:
    world_vertices = np.asarray(world_mesh.vertices, dtype=float)
    local_vertices = world_vertices @ frame_basis
    local_minimum = np.min(local_vertices, axis=0)
    local_maximum = np.max(local_vertices, axis=0)
    target_extents = local_maximum - local_minimum
    if np.any(target_extents <= TRIM_COMPONENT_GEOMETRY_EPSILON):
        raise ValueError(
            "The selected architectural trim has no measurable size on every axis."
        )
    local_bottom_center = np.asarray(
        (
            (local_minimum[0] + local_maximum[0]) * 0.5,
            (local_minimum[1] + local_maximum[1]) * 0.5,
            local_minimum[2],
        ),
        dtype=float,
    )
    world_bottom_center = frame_basis @ local_bottom_center
    local_mesh = world_mesh.copy()
    local_mesh.vertices = local_vertices - local_bottom_center
    width_axis = frame_basis[:, 0]
    frame = ArchitecturalTrimComponentFrame(
        level_index=int(level_index),
        trim_id=str(trim_id).strip().lower(),
        world_bottom_center=tuple(float(value) for value in world_bottom_center),
        yaw_degrees=math.degrees(math.atan2(width_axis[1], width_axis[0])),
        target_extents=tuple(float(value) for value in target_extents),
    )
    return local_mesh, frame


# ### Generated-model helpers ###
def _build_generated_model(
    local_mesh: trimesh.Trimesh,
    trim_id: str,
) -> GeneratedModel:
    node_name = f"{TRIM_COMPONENT_NODE_PREFIX}_{str(trim_id).strip().lower()}"
    preview_mesh = copy.deepcopy(local_mesh)
    export_mesh = copy.deepcopy(local_mesh)
    export_mesh.apply_transform(Z_UP_TO_GLTF_Y_UP_TRANSFORM)
    scene = trimesh.Scene()
    scene.add_geometry(
        export_mesh,
        geom_name=node_name,
        node_name=node_name,
    )
    exported = scene.export(file_type="glb")
    if not isinstance(exported, bytes) or not exported:
        raise ValueError("The architectural trim preview could not be serialized.")
    return GeneratedModel(
        mesh=preview_mesh,
        scene=scene,
        glb_bytes=exported,
    )
