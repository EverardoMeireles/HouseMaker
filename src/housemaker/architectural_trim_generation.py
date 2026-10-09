# ### Imports ###
from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import trimesh

from housemaker.architectural_trim import (
    TRIM_HANDLE_DEPTH,
    ArchitecturalTrimCorniceMiterDescriptor,
    build_architectural_trim_edit_targets,
    build_architectural_trim_geometry,
)
from housemaker.glb import (
    GLTF_Y_UP_TO_Z_UP_TRANSFORM,
    Z_UP_TO_GLTF_Y_UP_TRANSFORM,
    GeneratedModel,
    PlacedGeneratedModel,
    import_generated_glb,
)
from housemaker.models import ArchitecturalTrimData, LevelData

# ### Constants ###
TRIM_COMPONENT_GEOMETRY_EPSILON = 1e-8
TRIM_COMPONENT_FIT_RELATIVE_TOLERANCE = 1e-5
TRIM_COMPONENT_FIT_ABSOLUTE_TOLERANCE = 1e-7
TRIM_COMPONENT_NODE_PREFIX = "architectural_trim_component"
TRIM_REPEAT_LAYOUT_VERSION = 1
TRIM_REPEAT_MAX_COUNT = 256
TRIM_MITER_GEOMETRY_TOLERANCE_FACTOR = 1e-7


# ### Data models ###
@dataclass(frozen=True)
class ArchitecturalTrimComponentFrame:
    """Canonical local-to-world fit for one generated trim component."""

    level_index: int
    trim_id: str
    world_bottom_center: tuple[float, float, float]
    yaw_degrees: float
    target_extents: tuple[float, float, float]


@dataclass(frozen=True)
class ArchitecturalTrimRepeatLayout:
    """Validated dimensions required to texture and repeat one trim module."""

    target_extents: tuple[float, float, float]
    module_extents: tuple[float, float, float]
    repeat_count: int

    def __post_init__(self) -> None:
        target_extents = _normalize_trim_component_dimensions(
            self.target_extents
        )
        module_extents = _normalize_trim_component_dimensions(
            self.module_extents
        )
        if isinstance(self.repeat_count, (bool, np.bool_)) or not isinstance(
            self.repeat_count,
            (int, np.integer),
        ):
            raise TypeError("Architectural trim repeat count must be an integer.")
        repeat_count = int(self.repeat_count)
        if not 1 <= repeat_count <= TRIM_REPEAT_MAX_COUNT:
            raise ValueError(
                "Architectural trim repeat count is outside the supported range."
            )
        expected_target = module_extents.copy()
        expected_target[0] *= repeat_count
        if not np.allclose(
            target_extents,
            expected_target,
            rtol=TRIM_COMPONENT_FIT_RELATIVE_TOLERANCE,
            atol=TRIM_COMPONENT_FIT_ABSOLUTE_TOLERANCE,
        ):
            raise ValueError(
                "Architectural trim modules must exactly fill the target span."
            )
        object.__setattr__(
            self,
            "target_extents",
            tuple(float(value) for value in target_extents),
        )
        object.__setattr__(
            self,
            "module_extents",
            tuple(float(value) for value in module_extents),
        )
        object.__setattr__(self, "repeat_count", repeat_count)

    def to_pipeline_dict(self) -> dict[str, object]:
        """Serialize stable repeat metadata for persisted object records."""

        return {
            "version": TRIM_REPEAT_LAYOUT_VERSION,
            "targetExtents": list(self.target_extents),
            "moduleExtents": list(self.module_extents),
            "repeatCount": self.repeat_count,
        }

    @classmethod
    def from_pipeline_dict(
        cls,
        payload: Mapping[str, object],
    ) -> ArchitecturalTrimRepeatLayout:
        """Restore and validate repeat metadata from an object pipeline."""

        if not isinstance(payload, Mapping):
            raise TypeError("Architectural trim repeat metadata must be a mapping.")
        if payload.get("version") != TRIM_REPEAT_LAYOUT_VERSION:
            raise ValueError("Architectural trim repeat metadata is unsupported.")
        try:
            target_extents = payload["targetExtents"]
            module_extents = payload["moduleExtents"]
            repeat_count = payload["repeatCount"]
        except KeyError as error:
            raise ValueError(
                "Architectural trim repeat metadata is incomplete."
            ) from error
        return cls(
            target_extents=target_extents,  # type: ignore[arg-type]
            module_extents=module_extents,  # type: ignore[arg-type]
            repeat_count=repeat_count,  # type: ignore[arg-type]
        )


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


def fit_architectural_trim_glb_to_dimensions(
    glb_bytes: bytes,
    dimensions: Sequence[float],
) -> bytes:
    """Fit a trim GLB to exact Z-up dimensions before texture generation.

    The fitted geometry is bottom-center anchored at the local origin. Scaling
    the geometry before retexturing lets generated pixels follow the final
    proportions instead of stretching the finished texture during placement.
    """

    target_extents = _normalize_trim_component_dimensions(dimensions)
    source_model = import_generated_glb(glb_bytes)
    source_bounds = _get_valid_trim_component_bounds(source_model.mesh)
    if _trim_component_bounds_match_target(
        source_bounds,
        target_extents,
    ):
        return glb_bytes
    source_extents = source_bounds[1] - source_bounds[0]
    source_bottom_center = _bounds_bottom_center(source_bounds)

    z_up_transform = np.eye(4, dtype=float)
    axis_scales = target_extents / source_extents
    z_up_transform[:3, :3] = np.diag(axis_scales)
    z_up_transform[:3, 3] = -(axis_scales * source_bottom_center)
    gltf_transform = (
        Z_UP_TO_GLTF_Y_UP_TRANSFORM
        @ z_up_transform
        @ GLTF_Y_UP_TO_Z_UP_TRANSFORM
    )

    fitted_scene = _build_baked_trim_component_scene(
        source_model.scene,
        gltf_transform,
    )
    exported = fitted_scene.export(file_type="glb")
    if not isinstance(exported, bytes) or not exported:
        raise ValueError("The fitted architectural trim could not be serialized.")

    fitted_model = import_generated_glb(exported)
    _validate_fitted_trim_component_bounds(
        fitted_model.mesh,
        target_extents,
    )
    return exported


def prepare_architectural_trim_repeat_module(
    glb_bytes: bytes,
    target_dimensions: Sequence[float],
    *,
    maximum_module_aspect_ratio: float | None = None,
) -> tuple[bytes, ArchitecturalTrimRepeatLayout]:
    """Fit one short trim module before texturing and describe its repetition.

    The target depth and height are exact. The preferred module length applies
    the geometric mean of those cross-section scales to the source length,
    minimizing combined proportional distortion against both fitted axes. The
    nearest repeat count then minimizes the remaining longitudinal adjustment.
    """

    target_extents = _normalize_trim_component_dimensions(target_dimensions)
    source_model = import_generated_glb(glb_bytes)
    source_bounds = _get_valid_trim_component_bounds(source_model.mesh)
    source_extents = source_bounds[1] - source_bounds[0]
    cross_section_scales = target_extents[1:] / source_extents[1:]
    preferred_longitudinal_scale = math.sqrt(
        float(cross_section_scales[0] * cross_section_scales[1])
    )
    preferred_module_length = (
        float(source_extents[0]) * preferred_longitudinal_scale
    )
    raw_repeat_count = math.floor(
        float(target_extents[0]) / preferred_module_length + 0.5
    )
    minimum_repeat_count = 1
    if maximum_module_aspect_ratio is not None:
        if isinstance(maximum_module_aspect_ratio, (bool, np.bool_)):
            raise ValueError(
                "Architectural trim module aspect ratios must be positive."
            )
        try:
            normalized_aspect_ratio = float(maximum_module_aspect_ratio)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "Architectural trim module aspect ratios must be positive."
            ) from error
        if not math.isfinite(normalized_aspect_ratio) or normalized_aspect_ratio <= 0.0:
            raise ValueError(
                "Architectural trim module aspect ratios must be positive."
            )
        maximum_module_length = (
            max(float(target_extents[1]), float(target_extents[2]))
            * normalized_aspect_ratio
        )
        minimum_repeat_count = math.ceil(
            float(target_extents[0]) / maximum_module_length
        )
    repeat_count = min(
        max(raw_repeat_count, minimum_repeat_count, 1),
        TRIM_REPEAT_MAX_COUNT,
    )
    module_extents = target_extents.copy()
    module_extents[0] = target_extents[0] / repeat_count
    layout = ArchitecturalTrimRepeatLayout(
        target_extents=tuple(float(value) for value in target_extents),
        module_extents=tuple(float(value) for value in module_extents),
        repeat_count=repeat_count,
    )
    module_glb = fit_architectural_trim_glb_to_dimensions(
        glb_bytes,
        layout.module_extents,
    )
    return module_glb, layout


def repeat_architectural_trim_module_glb(
    module_glb_bytes: bytes,
    layout: ArchitecturalTrimRepeatLayout,
) -> bytes:
    """Repeat a textured module over the exact X span without extra draw calls.

    Copies belonging to each source primitive remain in one merged primitive,
    retaining that primitive's material and repeating its UV coordinates.
    """

    if not isinstance(layout, ArchitecturalTrimRepeatLayout):
        raise TypeError("Architectural trim repetition requires a repeat layout.")
    normalized_module_glb = fit_architectural_trim_glb_to_dimensions(
        module_glb_bytes,
        layout.module_extents,
    )
    module_model = import_generated_glb(normalized_module_glb)
    _validate_fitted_trim_component_bounds(
        module_model.mesh,
        np.asarray(layout.module_extents, dtype=float),
    )
    baked_module_scene = _build_baked_trim_component_scene(
        module_model.scene,
        np.eye(4, dtype=float),
    )
    repeated_scene = _build_repeated_trim_component_scene(
        baked_module_scene,
        layout,
    )
    exported = repeated_scene.export(file_type="glb")
    if not isinstance(exported, bytes) or not exported:
        raise ValueError("The repeated architectural trim could not be serialized.")
    repeated_model = import_generated_glb(exported)
    _validate_fitted_trim_component_bounds(
        repeated_model.mesh,
        np.asarray(layout.target_extents, dtype=float),
    )
    return exported


def retarget_architectural_trim_repeat_module_glb(
    module_glb_bytes: bytes,
    source_layout: ArchitecturalTrimRepeatLayout,
    target_dimensions: Sequence[float],
) -> tuple[bytes, ArchitecturalTrimRepeatLayout]:
    """Fit and repeat one textured module for a different trim span.

    The nearest supported repeat count keeps each target module as close as
    possible to the source module's authored length. Fitting changes only the
    vertex positions, so the persisted UV topology and motif layout remain
    shared by every retargeted cornice.
    """

    if not isinstance(source_layout, ArchitecturalTrimRepeatLayout):
        raise TypeError(
            "Architectural trim retargeting requires a source repeat layout."
        )
    target_extents = _normalize_trim_component_dimensions(target_dimensions)
    source_module_length = float(source_layout.module_extents[0])
    raw_repeat_count = math.floor(
        float(target_extents[0]) / source_module_length + 0.5
    )
    repeat_count = min(max(raw_repeat_count, 1), TRIM_REPEAT_MAX_COUNT)
    target_module_extents = target_extents.copy()
    target_module_extents[0] = target_extents[0] / repeat_count
    target_layout = ArchitecturalTrimRepeatLayout(
        target_extents=tuple(float(value) for value in target_extents),
        module_extents=tuple(float(value) for value in target_module_extents),
        repeat_count=repeat_count,
    )
    fitted_module_glb = fit_architectural_trim_glb_to_dimensions(
        module_glb_bytes,
        target_layout.module_extents,
    )
    return (
        repeat_architectural_trim_module_glb(
            fitted_module_glb,
            target_layout,
        ),
        target_layout,
    )


def build_architectural_trim_component_placement(
    model: GeneratedModel,
    frame: ArchitecturalTrimComponentFrame,
    *,
    object_id: str,
    object_name: str,
    atlas_source_object_id: str | None = None,
    cornice_miter: ArchitecturalTrimCorniceMiterDescriptor | None = None,
) -> PlacedGeneratedModel:
    """Fit one generated model to its authored trim span in the scene."""

    if not isinstance(model, GeneratedModel):
        raise TypeError("Architectural trim components require a generated model.")
    if cornice_miter is not None:
        return _build_mitered_architectural_trim_component_placement(
            model,
            frame,
            cornice_miter,
            object_id=object_id,
            object_name=object_name,
            atlas_source_object_id=atlas_source_object_id,
        )
    bounds = _get_valid_trim_component_bounds(model.mesh)
    source_extents = bounds[1] - bounds[0]
    target_extents = _normalize_trim_component_dimensions(frame.target_extents)
    if np.allclose(
        source_extents,
        target_extents,
        rtol=TRIM_COMPONENT_FIT_RELATIVE_TOLERANCE,
        atol=TRIM_COMPONENT_FIT_ABSOLUTE_TOLERANCE,
    ):
        axis_scales = np.ones(3, dtype=float)
    else:
        axis_scales = target_extents / source_extents
    if not np.all(np.isfinite(axis_scales)) or np.any(axis_scales <= 0.0):
        raise ValueError("The architectural trim fit produced invalid scales.")
    return PlacedGeneratedModel(
        object_id=str(object_id).strip(),
        source_object_id=str(object_id).strip(),
        atlas_source_object_id=atlas_source_object_id,
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


# ### Corner-miter helpers ###
def _build_mitered_architectural_trim_component_placement(
    model: GeneratedModel,
    frame: ArchitecturalTrimComponentFrame,
    miter: ArchitecturalTrimCorniceMiterDescriptor,
    *,
    object_id: str,
    object_name: str,
    atlas_source_object_id: str | None,
) -> PlacedGeneratedModel:
    """Miter authored cornice positions without changing their UV layout."""

    if not isinstance(miter, ArchitecturalTrimCorniceMiterDescriptor):
        raise TypeError("Architectural trim miters require a cornice descriptor.")
    if (
        miter.level_index != frame.level_index
        or miter.trim_id != str(frame.trim_id).strip().lower()
    ):
        raise ValueError("The cornice miter does not match its placement frame.")
    if not (miter.minimum_x.is_joined or miter.maximum_x.is_joined):
        return build_architectural_trim_component_placement(
            model,
            frame,
            object_id=object_id,
            object_name=object_name,
            atlas_source_object_id=atlas_source_object_id,
        )

    target_extents = _normalize_trim_component_dimensions(frame.target_extents)
    source_bounds = _get_valid_trim_component_bounds(model.mesh)
    normalized_model = model
    if not _trim_component_bounds_match_target(source_bounds, target_extents):
        normalized_model = import_generated_glb(
            fit_architectural_trim_glb_to_dimensions(
                model.glb_bytes,
                target_extents,
            )
        )
    original_bottom_center = _bounds_bottom_center(
        _get_valid_trim_component_bounds(normalized_model.mesh)
    )
    mitered_model = _miter_architectural_trim_component_model(
        normalized_model,
        miter,
    )
    mitered_bottom_center = _bounds_bottom_center(
        _get_valid_trim_component_bounds(mitered_model.mesh)
    )

    yaw_radians = math.radians(frame.yaw_degrees)
    cosine = math.cos(yaw_radians)
    sine = math.sin(yaw_radians)
    local_pivot_shift = mitered_bottom_center - original_bottom_center
    world_pivot_shift = np.asarray(
        (
            cosine * local_pivot_shift[0] - sine * local_pivot_shift[1],
            sine * local_pivot_shift[0] + cosine * local_pivot_shift[1],
            local_pivot_shift[2],
        ),
        dtype=float,
    )
    world_position = np.asarray(frame.world_bottom_center, dtype=float) + (
        world_pivot_shift
    )
    return PlacedGeneratedModel(
        object_id=str(object_id).strip(),
        source_object_id=str(object_id).strip(),
        atlas_source_object_id=atlas_source_object_id,
        object_name=str(object_name).strip(),
        model=mitered_model,
        world_position=tuple(float(value) for value in world_position),
        rotation_degrees=(0.0, 0.0, frame.yaw_degrees),
        axis_scales=(1.0, 1.0, 1.0),
    )


def _miter_architectural_trim_component_model(
    model: GeneratedModel,
    miter: ArchitecturalTrimCorniceMiterDescriptor,
) -> GeneratedModel:
    """Shear joined ends to their procedural bisectors and remove seam caps."""

    source_scene = model.scene
    if not isinstance(source_scene, trimesh.Scene):
        raise TypeError("Architectural trim miters require a mesh scene.")
    local_primitives: list[tuple[str, trimesh.Trimesh]] = []
    occupied_names: set[str] = set()
    for node_index, source_node_name in enumerate(
        sorted(source_scene.graph.nodes_geometry, key=str),
        start=1,
    ):
        source_transform, source_geometry_name = source_scene.graph.get(
            source_node_name
        )
        source_geometry = source_scene.geometry.get(source_geometry_name)
        if not isinstance(source_geometry, trimesh.Trimesh):
            raise TypeError("The architectural trim contains invalid geometry.")
        node_transform = np.asarray(source_transform, dtype=float)
        if node_transform.shape != (4, 4) or not np.all(np.isfinite(node_transform)):
            raise ValueError("The architectural trim contains an invalid node.")
        local_mesh = copy.deepcopy(source_geometry)
        local_mesh.apply_transform(
            GLTF_Y_UP_TO_Z_UP_TRANSFORM @ node_transform
        )
        base_name = str(source_node_name).strip() or f"trim_part_{node_index}"
        node_name = base_name
        suffix = 2
        while node_name in occupied_names:
            node_name = f"{base_name}_{suffix}"
            suffix += 1
        occupied_names.add(node_name)
        local_primitives.append((node_name, local_mesh))
    if not local_primitives:
        raise ValueError("The architectural trim contains no usable geometry.")

    global_bounds = np.asarray(
        (
            np.min([mesh.bounds[0] for _name, mesh in local_primitives], axis=0),
            np.max([mesh.bounds[1] for _name, mesh in local_primitives], axis=0),
        ),
        dtype=float,
    )
    extents = global_bounds[1] - global_bounds[0]
    if not np.all(np.isfinite(global_bounds)) or np.any(
        extents <= TRIM_COMPONENT_GEOMETRY_EPSILON
    ):
        raise ValueError("The architectural trim has invalid miter bounds.")
    minimum_x = float(global_bounds[0, 0])
    maximum_x = float(global_bounds[1, 0])
    minimum_y = float(global_bounds[0, 1])
    transition_length = min(
        float(extents[0]) * 0.5,
        max(float(extents[1]), float(extents[2])),
    )
    transition_length = max(
        transition_length,
        TRIM_COMPONENT_GEOMETRY_EPSILON,
    )
    boundary_tolerance = max(float(np.max(extents)), 1.0) * (
        TRIM_MITER_GEOMETRY_TOLERANCE_FACTOR
    )

    output_scene = trimesh.Scene()
    for node_name, source_mesh in local_primitives:
        mesh = copy.deepcopy(source_mesh)
        vertices = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        retained_faces = np.ones(len(faces), dtype=bool)
        if miter.minimum_x.is_joined:
            retained_faces &= ~np.all(
                np.isclose(
                    vertices[faces, 0],
                    minimum_x,
                    rtol=0.0,
                    atol=boundary_tolerance,
                ),
                axis=1,
            )
        if miter.maximum_x.is_joined:
            retained_faces &= ~np.all(
                np.isclose(
                    vertices[faces, 0],
                    maximum_x,
                    rtol=0.0,
                    atol=boundary_tolerance,
                ),
                axis=1,
            )
        if not np.any(retained_faces):
            continue
        if not np.all(retained_faces):
            mesh.update_faces(retained_faces)
            mesh.remove_unreferenced_vertices()

        warped_vertices = np.asarray(mesh.vertices, dtype=float).copy()
        original_x = warped_vertices[:, 0].copy()
        profile_offsets = np.clip(
            warped_vertices[:, 1] - minimum_y,
            0.0,
            float(extents[1]),
        )
        minimum_weights = np.clip(
            1.0 - ((original_x - minimum_x) / transition_length),
            0.0,
            1.0,
        )
        maximum_weights = np.clip(
            1.0 - ((maximum_x - original_x) / transition_length),
            0.0,
            1.0,
        )
        warped_vertices[:, 0] += profile_offsets * (
            minimum_weights * miter.minimum_x.local_x_shift_per_depth
            + maximum_weights * miter.maximum_x.local_x_shift_per_depth
        )
        mesh.vertices = warped_vertices
        mesh.apply_transform(Z_UP_TO_GLTF_Y_UP_TRANSFORM)
        output_scene.add_geometry(
            mesh,
            geom_name=node_name,
            node_name=node_name,
        )
    if not output_scene.geometry:
        raise ValueError("The mitered architectural trim contains no usable geometry.")
    exported = output_scene.export(file_type="glb")
    if not isinstance(exported, bytes) or not exported:
        raise ValueError("The mitered architectural trim could not be serialized.")
    preview_mesh = output_scene.to_geometry()
    if not isinstance(preview_mesh, trimesh.Trimesh) or preview_mesh.is_empty:
        raise ValueError("The mitered architectural trim has no preview geometry.")
    preview_mesh = preview_mesh.copy()
    preview_mesh.apply_transform(GLTF_Y_UP_TO_Z_UP_TRANSFORM)
    return GeneratedModel(
        mesh=preview_mesh,
        scene=output_scene,
        glb_bytes=exported,
    )


# ### Geometry-frame helpers ###
def _build_baked_trim_component_scene(
    source_scene: trimesh.Scene,
    gltf_transform: np.ndarray,
) -> trimesh.Scene:
    """Bake every source-node transform into final fitted vertex positions."""

    if not isinstance(source_scene, trimesh.Scene):
        raise TypeError("Architectural trim components require a mesh scene.")
    fit_transform = np.asarray(gltf_transform, dtype=float)
    if fit_transform.shape != (4, 4) or not np.all(np.isfinite(fit_transform)):
        raise ValueError("The architectural trim fit transform is invalid.")

    fitted_scene = trimesh.Scene()
    occupied_names: set[str] = set()
    for node_index, source_node_name in enumerate(
        sorted(source_scene.graph.nodes_geometry, key=str),
        start=1,
    ):
        source_transform, source_geometry_name = source_scene.graph.get(
            source_node_name
        )
        node_transform = np.asarray(source_transform, dtype=float)
        if node_transform.shape != (4, 4) or not np.all(np.isfinite(node_transform)):
            raise ValueError("The architectural trim contains an invalid node.")
        source_geometry = source_scene.geometry.get(source_geometry_name)
        if not isinstance(source_geometry, trimesh.Trimesh):
            raise TypeError("The architectural trim contains invalid geometry.")

        fitted_geometry = copy.deepcopy(source_geometry)
        fitted_geometry.apply_transform(fit_transform @ node_transform)
        base_name = str(source_node_name).strip() or f"trim_part_{node_index}"
        node_name = base_name
        suffix = 2
        while node_name in occupied_names:
            node_name = f"{base_name}_{suffix}"
            suffix += 1
        occupied_names.add(node_name)
        fitted_scene.add_geometry(
            fitted_geometry,
            geom_name=node_name,
            node_name=node_name,
        )
    if not fitted_scene.geometry:
        raise ValueError("The architectural trim contains no usable geometry.")
    return fitted_scene


def _build_repeated_trim_component_scene(
    module_scene: trimesh.Scene,
    layout: ArchitecturalTrimRepeatLayout,
) -> trimesh.Scene:
    """Merge all copies of each module primitive into one scene primitive."""

    repeated_scene = trimesh.Scene()
    module_length = layout.module_extents[0]
    target_minimum_x = -layout.target_extents[0] * 0.5
    first_center_x = target_minimum_x + module_length * 0.5
    occupied_names: set[str] = set()
    for node_index, source_node_name in enumerate(
        sorted(module_scene.graph.nodes_geometry, key=str),
        start=1,
    ):
        source_transform, source_geometry_name = module_scene.graph.get(
            source_node_name
        )
        node_transform = np.asarray(source_transform, dtype=float)
        if node_transform.shape != (4, 4) or not np.all(np.isfinite(node_transform)):
            raise ValueError("The architectural trim contains an invalid node.")
        source_geometry = module_scene.geometry.get(source_geometry_name)
        if not isinstance(source_geometry, trimesh.Trimesh):
            raise TypeError("The architectural trim contains invalid geometry.")

        source_primitive = copy.deepcopy(source_geometry)
        source_primitive.apply_transform(node_transform)
        repeated_copies: list[trimesh.Trimesh] = []
        for repeat_index in range(layout.repeat_count):
            repeated_copy = copy.deepcopy(source_primitive)
            repeated_copy.apply_translation(
                (
                    first_center_x + repeat_index * module_length,
                    0.0,
                    0.0,
                )
            )
            repeated_copies.append(repeated_copy)
        repeated_geometry = trimesh.util.concatenate(repeated_copies)
        base_name = str(source_node_name).strip() or f"trim_part_{node_index}"
        node_name = base_name
        suffix = 2
        while node_name in occupied_names:
            node_name = f"{base_name}_{suffix}"
            suffix += 1
        occupied_names.add(node_name)
        repeated_scene.add_geometry(
            repeated_geometry,
            geom_name=node_name,
            node_name=node_name,
        )
    if not repeated_scene.geometry:
        raise ValueError("The architectural trim contains no usable geometry.")
    return repeated_scene


def _normalize_trim_component_dimensions(
    dimensions: Sequence[float],
) -> np.ndarray:
    if isinstance(dimensions, (str, bytes)):
        raise TypeError(
            "Architectural trim dimensions must contain three positive numbers."
        )
    try:
        raw_dimensions = tuple(dimensions)
    except TypeError as error:
        raise ValueError(
            "Architectural trim dimensions must contain three positive numbers."
        ) from error
    if len(raw_dimensions) != 3 or any(
        isinstance(value, (bool, np.bool_)) for value in raw_dimensions
    ):
        raise ValueError(
            "Architectural trim dimensions must contain three positive numbers."
        )
    try:
        normalized = np.asarray(raw_dimensions, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "Architectural trim dimensions must contain three positive numbers."
        ) from error
    if not np.all(np.isfinite(normalized)) or np.any(
        normalized <= TRIM_COMPONENT_GEOMETRY_EPSILON
    ):
        raise ValueError(
            "Architectural trim dimensions must contain three positive numbers."
        )
    return normalized


def _get_valid_trim_component_bounds(mesh: trimesh.Trimesh) -> np.ndarray:
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("Architectural trim components require a triangle mesh.")
    bounds = np.asarray(mesh.bounds, dtype=float)
    if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
        raise ValueError("The generated architectural trim has invalid bounds.")
    if np.any(bounds[1] - bounds[0] <= TRIM_COMPONENT_GEOMETRY_EPSILON):
        raise ValueError(
            "The generated architectural trim has no measurable size on every axis."
        )
    return bounds


def _bounds_bottom_center(bounds: np.ndarray) -> np.ndarray:
    return np.asarray(
        (
            (bounds[0, 0] + bounds[1, 0]) * 0.5,
            (bounds[0, 1] + bounds[1, 1]) * 0.5,
            bounds[0, 2],
        ),
        dtype=float,
    )


def _validate_fitted_trim_component_bounds(
    mesh: trimesh.Trimesh,
    target_extents: np.ndarray,
) -> None:
    fitted_bounds = _get_valid_trim_component_bounds(mesh)
    if not _trim_component_bounds_match_target(
        fitted_bounds,
        target_extents,
    ):
        raise ValueError("The architectural trim GLB could not be fitted reliably.")


def _trim_component_bounds_match_target(
    fitted_bounds: np.ndarray,
    target_extents: np.ndarray,
) -> bool:
    """Return whether bounds already use the canonical trim module frame."""

    expected_bounds = np.asarray(
        (
            (-target_extents[0] * 0.5, -target_extents[1] * 0.5, 0.0),
            (target_extents[0] * 0.5, target_extents[1] * 0.5, target_extents[2]),
        ),
        dtype=float,
    )
    return bool(
        np.allclose(
            fitted_bounds,
            expected_bounds,
            rtol=TRIM_COMPONENT_FIT_RELATIVE_TOLERANCE,
            atol=TRIM_COMPONENT_FIT_ABSOLUTE_TOLERANCE,
        )
    )


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
