# ### Imports ###
from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import trimesh
import xatlas
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.door_state import (
    DOOR_LEAF_LEFT,
    DOOR_SIDE_DUPLICATION_FRONT,
    DOOR_SLOT_BACK_BODY,
    DOOR_SLOT_BACK_KNOB,
    DOOR_SLOT_BACK_LEFT_BODY,
    DOOR_SLOT_BACK_LEFT_KNOB,
    DOOR_SLOT_BODY,
    DOOR_SLOT_KNOB,
    DOOR_SLOT_LEFT_BODY,
    DOOR_SLOT_LEFT_KNOB,
    DoorDefinition,
    DoorSlotData,
    WindowDefinition,
    door_slot_display_name,
    door_slot_leaf,
    door_slot_side,
    is_door_body_slot,
    is_door_knob_slot,
)
from housemaker.doorway_geometry import (
    build_doorway_cross_section_outline,
    build_double_door_leaf_source_outline,
)
from housemaker.glb import (
    DOOR_BODY_MARKER_METADATA_KEY,
    DOOR_COMPONENT_MARKER_METADATA_KEY,
    GLTF_Y_UP_TO_Z_UP_TRANSFORM,
    SYMMETRIC_PREVIEW_AXIS_BY_ORIENTATION,
    Z_UP_TO_GLTF_Y_UP_TRANSFORM,
    GeneratedModel,
    PlacedGeneratedModel,
    PreviewSymmetricObject,
    compose_placed_generated_models,
    compose_placed_generated_models_preview,
)
from housemaker.opening_glass import apply_opening_glass_areas

# ### Constants ###
DOOR_BODY_COLOR = (164, 109, 68, 255)
DOOR_BODY_GEOMETRY_NAME = "door_body"
DOOR_GEOMETRY_EPSILON = 1e-9
DOOR_UV_ATLAS_RESOLUTION = 1024
DOOR_UV_PADDING_PIXELS = 2
DOOR_BODY_SOURCE_HALF_LEFT = "left"
DOOR_BODY_SOURCE_HALF_RIGHT = "right"
DOOR_BODY_HANDLE_SLOT_IDS = {
    DOOR_SLOT_BODY: DOOR_SLOT_KNOB,
    DOOR_SLOT_LEFT_BODY: DOOR_SLOT_LEFT_KNOB,
    DOOR_SLOT_BACK_BODY: DOOR_SLOT_BACK_KNOB,
    DOOR_SLOT_BACK_LEFT_BODY: DOOR_SLOT_BACK_LEFT_KNOB,
}


# ### Door processing models ###
@dataclass(frozen=True, slots=True)
class DoorBodyMirrorConfiguration:
    """Viewer/export provenance for one processed authored door body."""

    symmetric_orientation: str | None = None
    symmetric_plane_coordinate: float | None = None
    side_duplication_plane_coordinate: float | None = None

    def __post_init__(self) -> None:
        has_symmetric_orientation = self.symmetric_orientation is not None
        has_symmetric_plane = self.symmetric_plane_coordinate is not None
        if has_symmetric_orientation != has_symmetric_plane:
            raise ValueError(
                "Door body symmetry requires both an orientation and plane."
            )
        if has_symmetric_orientation:
            orientation = str(self.symmetric_orientation).strip().lower()
            if orientation not in SYMMETRIC_PREVIEW_AXIS_BY_ORIENTATION:
                raise ValueError("The door body symmetry orientation is invalid.")
            plane_coordinate = float(self.symmetric_plane_coordinate)
            if not math.isfinite(plane_coordinate):
                raise ValueError("The door body symmetry plane must be finite.")
            object.__setattr__(self, "symmetric_orientation", orientation)
            object.__setattr__(
                self,
                "symmetric_plane_coordinate",
                plane_coordinate,
            )
        if self.side_duplication_plane_coordinate is not None:
            side_plane = float(self.side_duplication_plane_coordinate)
            if not math.isfinite(side_plane):
                raise ValueError("The door side-duplication plane must be finite.")
            object.__setattr__(
                self,
                "side_duplication_plane_coordinate",
                side_plane,
            )


# ### Public body construction ###
def build_door_body_model(
    width_meters: float,
    height_meters: float,
    thickness_meters: float,
    *,
    shape: str,
    arch_amount: float,
    geometry_name: str = DOOR_BODY_GEOMETRY_NAME,
) -> GeneratedModel:
    """Build a watertight door body with the exact requested outer profile.

    The canonical model is Z-up, centered across X and Y, and rests on Z=0.
    Its glTF scene is stored Y-up to match every other ``GeneratedModel``.
    """

    outline = build_doorway_cross_section_outline(
        width_meters,
        height_meters,
        shape,
        arch_amount,
    )
    return _build_door_body_model_from_profile(
        outline,
        thickness_meters,
        geometry_name,
    )


def build_default_door_slot_models(
    door: DoorDefinition,
) -> dict[str, GeneratedModel]:
    """Build the doorway-fitting body supplied for each configured side."""

    _require_door_definition(door)
    return {
        slot.slot_id: _build_door_body_model_from_profile(
            build_door_body_source_outline(door, slot),
            door.thickness_meters,
            (
                DOOR_BODY_GEOMETRY_NAME
                if slot.slot_id == DOOR_SLOT_BODY
                else f"{DOOR_BODY_GEOMETRY_NAME}_{slot.slot_id}"
            ),
        )
        for slot in door.slots
        if is_door_body_slot(slot.slot_id)
    }


def build_door_body_source_outline(
    door: DoorDefinition,
    slot: DoorSlotData,
) -> tuple[tuple[float, float], ...]:
    """Resolve the profile authored into one door body's source model."""

    _require_door_definition(door)
    if not isinstance(slot, DoorSlotData) or not is_door_body_slot(slot.slot_id):
        raise TypeError("Door body profiles require a body slot.")
    if not door.make_double:
        return build_doorway_cross_section_outline(
            door.width_meters,
            door.height_meters,
            door.shape,
            door.arch_amount,
        )

    outline = build_double_door_leaf_source_outline(
        door.width_meters,
        door.height_meters,
        door.shape,
        door.arch_amount,
    )
    if door_body_slot_source_half(slot) == DOOR_BODY_SOURCE_HALF_LEFT:
        return outline
    return _mirror_closed_profile_horizontally(outline)


def door_body_slot_source_half(slot: DoorSlotData) -> str:
    """Return the global profile half expected before slot-level mirroring."""

    if not isinstance(slot, DoorSlotData) or not is_door_body_slot(slot.slot_id):
        raise TypeError("Door leaf source orientation requires a body slot.")
    is_left_leaf = door_slot_leaf(slot.slot_id) == DOOR_LEAF_LEFT
    uses_left_source = is_left_leaf != slot.mirrored_horizontally
    return (
        DOOR_BODY_SOURCE_HALF_LEFT
        if uses_left_source
        else DOOR_BODY_SOURCE_HALF_RIGHT
    )


def _build_door_body_model_from_profile(
    profile_outline: Sequence[tuple[float, float]],
    thickness_meters: float,
    geometry_name: str,
) -> GeneratedModel:
    """Extrude one already-resolved canonical door body profile."""

    thickness = _normalize_positive_measurement(
        thickness_meters,
        "Door thickness",
    )
    mesh = _extrude_profile(profile_outline, thickness)
    mesh = _unwrap_mesh(mesh, DOOR_BODY_COLOR, geometry_name)
    return _build_generated_model(mesh, geometry_name)


def build_default_window_slot_models(
    window: WindowDefinition,
) -> dict[str, GeneratedModel]:
    """Build a rectangular fallback for every active window body slot."""

    _require_window_definition(window)
    return {
        slot.slot_id: build_door_body_model(
            window.width_meters,
            window.height_meters,
            window.thickness_meters,
            shape="rectangular",
            arch_amount=0.0,
            geometry_name=(
                "window_body"
                if slot.slot_id == DOOR_SLOT_BODY
                else f"window_body_{slot.slot_id}"
            ),
        )
        for slot in window.slots
        if is_door_body_slot(slot.slot_id)
    }


# ### Body profile helpers ###
def _mirror_closed_profile_horizontally(
    outline: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    """Reflect a canonical closed X/Z profile and retain its point order."""

    unique_points = [(-float(x), float(z)) for x, z in outline[:-1]]
    reversed_points = list(reversed(unique_points))
    start_index = min(
        range(len(reversed_points)),
        key=lambda point_index: (
            reversed_points[point_index][1],
            reversed_points[point_index][0],
        ),
    )
    ordered = reversed_points[start_index:] + reversed_points[:start_index]
    ordered.append(ordered[0])
    return tuple(ordered)


# ### Door assembly ###
def assemble_door_model(
    door: DoorDefinition,
    slot_models: Mapping[str, GeneratedModel] | None = None,
    *,
    include_unjoined: bool = False,
    body_mirror: DoorBodyMirrorConfiguration | None = None,
    body_mirrors: Mapping[str, DoorBodyMirrorConfiguration] | None = None,
) -> GeneratedModel:
    """Join a door's confirmed slots while retaining each slot material.

    ``slot_models`` may be keyed by slot ID or by ``source_object_id``. This
    lets callers replace a procedural component with its textured generated
    model without rewriting the persistent door definition.
    """

    _require_door_definition(door)
    if not isinstance(include_unjoined, bool):
        raise TypeError("Door unjoined-slot preview state must be boolean.")
    if body_mirror is not None and not isinstance(
        body_mirror,
        DoorBodyMirrorConfiguration,
    ):
        raise TypeError("Door body mirror configuration is invalid.")
    normalized_body_mirrors = dict(body_mirrors or {})
    if body_mirror is not None:
        normalized_body_mirrors.setdefault(DOOR_SLOT_BODY, body_mirror)
    if any(
        not is_door_body_slot(slot_id)
        or not isinstance(configuration, DoorBodyMirrorConfiguration)
        for slot_id, configuration in normalized_body_mirrors.items()
    ):
        raise TypeError("Door body mirror configurations are invalid.")
    models = (
        build_default_door_slot_models(door)
        if slot_models is None
        else dict(slot_models)
    )
    placements: list[PlacedGeneratedModel] = []
    effective_body_mirrors: dict[str, DoorBodyMirrorConfiguration] = {}
    for slot in door.slots:
        if (
            not is_door_body_slot(slot.slot_id)
            and not include_unjoined
            and not slot.joined
        ):
            continue
        component_model = _resolve_slot_model(models, slot)
        if component_model is None:
            if is_door_body_slot(slot.slot_id):
                raise ValueError("The door body has no generated source model.")
            continue
        slot_side = door_slot_side(slot.slot_id)
        depth_mirrored = slot_side != DOOR_SIDE_DUPLICATION_FRONT
        if depth_mirrored:
            component_model = mirror_door_model_depthwise(component_model)
        slot_body_mirror = _transform_body_mirror_configuration(
            normalized_body_mirrors.get(slot.slot_id),
            horizontal_mirror=slot.mirrored_horizontally,
            depth_mirror=depth_mirrored,
        )
        if slot_body_mirror is not None:
            effective_body_mirrors[slot.slot_id] = slot_body_mirror
        if is_door_body_slot(slot.slot_id):
            if slot.mirrored_horizontally:
                component_model = mirror_door_model_horizontally(
                    component_model
                )
            omitted_glass_depth_cap = None
            if door.make_double_sided:
                omitted_glass_depth_cap = (
                    "maximum"
                    if door_slot_side(slot.slot_id)
                    == DOOR_SIDE_DUPLICATION_FRONT
                    else "minimum"
                )
            component_model = _with_opening_glass_areas(
                component_model,
                tuple(
                    area
                    for area in door.glass_areas
                    if _glass_area_applies_to_body_slot(
                        door,
                        area.body_slot_id,
                        slot.slot_id,
                    )
                ),
                width_meters=door.width_meters,
                height_meters=door.height_meters,
                omitted_glass_depth_cap=omitted_glass_depth_cap,
            )
            # Glass cutting serializes authoritative source models.  Apply it
            # before the placement-dependent body marker exists, then tag the
            # opaque body and its glass panes as one reconstruction group.
            component_model = _with_door_body_marker(
                component_model,
                slot.source_object_id,
            )
        elif is_door_knob_slot(slot.slot_id):
            if slot.mirrored_horizontally:
                component_model = mirror_door_model_horizontally(
                    component_model
                )
            component_model = _with_door_component_marker(
                component_model,
                slot.source_object_id,
                DOOR_SLOT_KNOB,
            )
        component_display_name = door_slot_display_name(
            slot.slot_id,
            make_double=door.make_double,
        )
        world_position = slot.position_meters
        axis_scales = slot.axis_scales
        if is_door_body_slot(slot.slot_id):
            reconstructed_extents = _body_fitting_extents(
                component_model,
                slot_body_mirror,
                target_extents=np.asarray(
                    (
                        door.width_meters,
                        door.thickness_meters,
                        door.height_meters,
                    ),
                    dtype=float,
                ),
            )
            axis_scales = (
                axis_scales[0]
                * door.width_meters
                / reconstructed_extents[0],
                axis_scales[1]
                * door.thickness_meters
                / reconstructed_extents[1],
                axis_scales[2]
                * door.height_meters
                / reconstructed_extents[2],
            )
            world_position = _reconstructed_body_world_position(
                component_model,
                slot_body_mirror,
                target_position=slot.position_meters,
                rotation_degrees=slot.rotation_degrees,
                axis_scales=axis_scales,
                target_extents=(
                    door.width_meters,
                    door.thickness_meters,
                    door.height_meters,
                ),
            )
            if door.make_double_sided:
                body_depth = _mesh_axis_extent(component_model.mesh, 1)
                side_depth = (
                    door.thickness_meters * slot.axis_scales[1] * 0.5
                )
                side_sign = (
                    -1.0
                    if door_slot_side(slot.slot_id)
                    == DOOR_SIDE_DUPLICATION_FRONT
                    else 1.0
                )
                world_position = (
                    world_position[0],
                    world_position[1]
                    + side_sign * side_depth * 0.5,
                    world_position[2],
                )
                axis_scales = (
                    axis_scales[0],
                    side_depth / body_depth,
                    axis_scales[2],
                )
        elif is_door_knob_slot(slot.slot_id) and door.make_double_sided:
            side_sign = (
                -1.0
                if slot_side == DOOR_SIDE_DUPLICATION_FRONT
                else 1.0
            )
            world_position = (
                world_position[0],
                world_position[1]
                + side_sign * door.thickness_meters * 0.5,
                world_position[2],
            )
        placements.append(
            PlacedGeneratedModel(
                object_id=slot.source_object_id,
                source_object_id=slot.source_object_id,
                object_name=f"{door.name} {component_display_name}",
                model=component_model,
                world_position=world_position,
                symmetric_preview_orientation=(
                    None
                    if slot_body_mirror is None
                    else slot_body_mirror.symmetric_orientation
                ),
                symmetric_preview_plane_coordinate=(
                    None
                    if slot_body_mirror is None
                    else slot_body_mirror.symmetric_plane_coordinate
                ),
                rotation_degrees=slot.rotation_degrees,
                axis_scales=axis_scales,
            )
        )
    # Every slot must pass through the placed-model preview splitter so the
    # body's embedded texture is not flattened into an untextured mesh.
    empty_base = GeneratedModel(
        mesh=trimesh.Trimesh(process=False),
        scene=trimesh.Scene(),
        glb_bytes=b"",
    )
    compose_models = (
        compose_placed_generated_models_preview
        if any(placement.model.preview_only for placement in placements)
        else compose_placed_generated_models
    )
    assembled = compose_models(empty_base, placements)
    assembled.preview_symmetric_objects = [
        replace(preview, fade_enabled=False)
        for preview in assembled.preview_symmetric_objects
    ]
    for body_slot_id, body_mirror_configuration in effective_body_mirrors.items():
        if body_mirror_configuration.side_duplication_plane_coordinate is None:
            continue
        _append_side_duplication_previews(
            assembled,
            door.get_slot(body_slot_id),
            door.get_slot(DOOR_BODY_HANDLE_SLOT_IDS[body_slot_id]),
            body_mirror_configuration,
        )
    return assembled


def _glass_area_applies_to_body_slot(
    door: DoorDefinition,
    owner_slot_id: str,
    candidate_slot_id: str,
) -> bool:
    """Match one opening to its leaf on each authored door side."""

    if owner_slot_id == candidate_slot_id:
        return True
    if not door.make_double_sided:
        return False
    return door_slot_leaf(owner_slot_id) == door_slot_leaf(candidate_slot_id)


def assemble_window_model(
    window: WindowDefinition,
    slot_models: Mapping[str, GeneratedModel] | None = None,
    *,
    include_unjoined: bool = False,
    body_mirror: DoorBodyMirrorConfiguration | None = None,
    body_mirrors: Mapping[str, DoorBodyMirrorConfiguration] | None = None,
) -> GeneratedModel:
    """Assemble a reusable window through the proven door component path."""

    _require_window_definition(window)
    surrogate = DoorDefinition(
        door_id=window.window_id,
        name=window.name,
        source_doorway_id=window.source_window_frame_id,
        width_meters=window.width_meters,
        height_meters=window.height_meters,
        thickness_meters=window.thickness_meters,
        make_double=window.make_double,
        make_double_sided=window.make_double_sided,
        slots=window.slots,
        glass_areas=window.glass_areas,
    )
    models = (
        build_default_window_slot_models(window)
        if slot_models is None
        else slot_models
    )
    return assemble_door_model(
        surrogate,
        models,
        include_unjoined=include_unjoined,
        body_mirror=body_mirror,
        body_mirrors=body_mirrors,
    )


def _with_opening_glass_areas(
    body_model: GeneratedModel,
    areas: Sequence[object],
    *,
    width_meters: float,
    height_meters: float,
    omitted_glass_depth_cap: str | None = None,
) -> GeneratedModel:
    """Rebuild one body with all persistent through-cuts and glass cuboids."""

    if not areas:
        return body_model
    half_width = _normalize_positive_measurement(
        width_meters,
        "Opening width",
    ) * 0.5
    height = _normalize_positive_measurement(
        height_meters,
        "Opening height",
    )
    return apply_opening_glass_areas(
        body_model,
        tuple(areas),
        profile_bounds_meters=(-half_width, half_width, 0.0, height),
        omitted_glass_depth_cap=omitted_glass_depth_cap,
    ).model


def _with_door_body_marker(
    model: GeneratedModel,
    body_object_id: str,
) -> GeneratedModel:
    """Tag only body mesh nodes so runtime reconstruction excludes hardware."""

    tagged = copy.deepcopy(model)
    marker = {"bodyObjectId": str(body_object_id).strip()}
    for geometry in tagged.scene.geometry.values():
        if not isinstance(geometry, trimesh.Trimesh):
            continue
        geometry.metadata = copy.deepcopy(
            dict(getattr(geometry, "metadata", {}) or {})
        )
        geometry.metadata[DOOR_BODY_MARKER_METADATA_KEY] = copy.deepcopy(marker)
    for node_name in tagged.scene.graph.nodes_geometry:
        parent_name = tagged.scene.graph.transforms.parents.get(node_name)
        if parent_name is None:
            continue
        edge_data = tagged.scene.graph.transforms.edge_data.get(
            (parent_name, node_name)
        )
        if not isinstance(edge_data, dict):
            continue
        metadata = copy.deepcopy(dict(edge_data.get("metadata") or {}))
        metadata[DOOR_BODY_MARKER_METADATA_KEY] = copy.deepcopy(marker)
        edge_data["metadata"] = metadata
    return tagged


def _with_door_component_marker(
    model: GeneratedModel,
    component_object_id: str,
    component_kind: str,
) -> GeneratedModel:
    """Tag one authored component so runtime reconstruction can locate it."""

    tagged = copy.deepcopy(model)
    marker = {
        "componentObjectId": str(component_object_id).strip(),
        "kind": str(component_kind).strip(),
    }
    for geometry in tagged.scene.geometry.values():
        if not isinstance(geometry, trimesh.Trimesh):
            continue
        geometry.metadata = copy.deepcopy(
            dict(getattr(geometry, "metadata", {}) or {})
        )
        geometry.metadata[DOOR_COMPONENT_MARKER_METADATA_KEY] = copy.deepcopy(
            marker
        )
    for node_name in tagged.scene.graph.nodes_geometry:
        parent_name = tagged.scene.graph.transforms.parents.get(node_name)
        if parent_name is None:
            continue
        edge_data = tagged.scene.graph.transforms.edge_data.get(
            (parent_name, node_name)
        )
        if not isinstance(edge_data, dict):
            continue
        metadata = copy.deepcopy(dict(edge_data.get("metadata") or {}))
        metadata[DOOR_COMPONENT_MARKER_METADATA_KEY] = copy.deepcopy(marker)
        edge_data["metadata"] = metadata
    return tagged


def _append_side_duplication_previews(
    assembled: GeneratedModel,
    body_slot: DoorSlotData | None,
    knob_slot: DoorSlotData | None,
    body_mirror: DoorBodyMirrorConfiguration,
) -> None:
    """Show body reconstruction mirrors and duplicate its knob by depth."""

    if body_slot is None:
        raise ValueError("A side-duplicated door requires a body slot.")
    preview = next(
        (
            item
            for item in assembled.preview_placed_objects
            if item.object_id == body_slot.source_object_id
        ),
        None,
    )
    if preview is None:
        raise ValueError("The assembled door has no body preview meshes.")
    world_meshes = _transform_preview_meshes(
        preview.meshes,
        preview.placement_transform,
    )
    local_side_plane = body_mirror.side_duplication_plane_coordinate
    assert local_side_plane is not None
    side_plane = _transform_axis_plane_coordinate(
        preview.placement_transform,
        axis=1,
        plane_coordinate=local_side_plane,
    )
    side_mirrors = _reflect_preview_meshes(world_meshes, 1, side_plane)
    assembled.preview_symmetric_objects.append(
        PreviewSymmetricObject(
            object_id=f"{body_slot.source_object_id}:side",
            meshes=world_meshes,
            orientation="depth",
            plane_coordinate=side_plane,
            mirrored_meshes=side_mirrors,
            fade_enabled=False,
        )
    )
    if body_mirror.symmetric_orientation is not None:
        symmetric_axis = SYMMETRIC_PREVIEW_AXIS_BY_ORIENTATION[
            body_mirror.symmetric_orientation
        ]
        local_symmetric_plane = body_mirror.symmetric_plane_coordinate
        assert local_symmetric_plane is not None
        symmetric_plane = _transform_axis_plane_coordinate(
            preview.placement_transform,
            axis=symmetric_axis,
            plane_coordinate=local_symmetric_plane,
        )
        symmetric_mirrors = _reflect_preview_meshes(
            world_meshes,
            symmetric_axis,
            symmetric_plane,
        )
        combined_mirrors = _reflect_preview_meshes(
            symmetric_mirrors,
            1,
            side_plane,
        )
        assembled.preview_symmetric_objects.append(
            PreviewSymmetricObject(
                object_id=f"{body_slot.source_object_id}:symmetric-side",
                meshes=symmetric_mirrors,
                orientation="depth",
                plane_coordinate=side_plane,
                mirrored_meshes=combined_mirrors,
                fade_enabled=False,
            )
        )

    if knob_slot is None:
        return
    knob_preview = next(
        (
            item
            for item in assembled.preview_placed_objects
            if item.object_id == knob_slot.source_object_id
        ),
        None,
    )
    if knob_preview is None:
        return
    knob_meshes = _transform_preview_meshes(
        knob_preview.meshes,
        knob_preview.placement_transform,
    )
    assembled.preview_symmetric_objects.append(
        PreviewSymmetricObject(
            object_id=f"{knob_slot.source_object_id}:side",
            meshes=knob_meshes,
            orientation="depth",
            plane_coordinate=side_plane,
            mirrored_meshes=_reflect_preview_meshes(
                knob_meshes,
                1,
                side_plane,
            ),
            fade_enabled=False,
        )
    )


def _transform_preview_meshes(
    meshes: Sequence[trimesh.Trimesh],
    transform: np.ndarray,
) -> tuple[trimesh.Trimesh, ...]:
    transformed: list[trimesh.Trimesh] = []
    for mesh in meshes:
        copied = copy.deepcopy(mesh)
        copied.apply_transform(np.asarray(transform, dtype=float))
        transformed.append(copied)
    return tuple(transformed)


def _transform_axis_plane_coordinate(
    transform: np.ndarray,
    *,
    axis: int,
    plane_coordinate: float,
) -> float:
    """Move one local axis-aligned mirror plane into assembled coordinates."""

    point = np.zeros(4, dtype=float)
    point[axis] = float(plane_coordinate)
    point[3] = 1.0
    transformed = np.asarray(transform, dtype=float) @ point
    return float(transformed[axis])


def _reflect_preview_meshes(
    meshes: Sequence[trimesh.Trimesh],
    axis: int,
    plane_coordinate: float,
) -> tuple[trimesh.Trimesh, ...]:
    reflection = np.eye(4, dtype=float)
    reflection[axis, axis] = -1.0
    reflection[axis, 3] = 2.0 * float(plane_coordinate)
    reflected: list[trimesh.Trimesh] = []
    for mesh in meshes:
        copied = copy.deepcopy(mesh)
        copied.apply_transform(reflection)
        reflected.append(copied)
    return tuple(reflected)


def _mesh_axis_extent(mesh: trimesh.Trimesh, axis: int) -> float:
    """Return one finite positive local mesh extent."""

    minimum, maximum = np.asarray(mesh.bounds, dtype=float)
    extent = float(maximum[axis] - minimum[axis])
    if not math.isfinite(extent) or extent <= DOOR_GEOMETRY_EPSILON:
        raise ValueError("A door body has no measurable depth.")
    return extent


def _body_fitting_extents(
    model: GeneratedModel,
    mirror: DoorBodyMirrorConfiguration | None,
    *,
    target_extents: np.ndarray,
) -> np.ndarray:
    """Return body extents used to fit one slot to its canonical profile.

    Generated body assets from older projects are not guaranteed to have the
    canonical door dimensions.  Slot offsets are expressed in that canonical
    coordinate system, so fitting by these reconstructed extents keeps paired
    leaves flush without changing their meshes, UVs, or materials.
    """

    bounds = _body_fitting_bounds(
        model,
        mirror,
        target_extents=target_extents,
    )
    extents = bounds[1] - bounds[0]
    if np.any(~np.isfinite(extents)) or np.any(
        extents <= DOOR_GEOMETRY_EPSILON
    ):
        raise ValueError("A door body has no measurable size on every axis.")
    return extents


def _body_fitting_bounds(
    model: GeneratedModel,
    mirror: DoorBodyMirrorConfiguration | None,
    *,
    target_extents: Sequence[float],
) -> np.ndarray:
    """Return the effective full-body bounds used by fitting and anchoring."""

    authored_bounds = np.asarray(model.mesh.bounds, dtype=float)
    if authored_bounds.shape != (2, 3) or not np.all(
        np.isfinite(authored_bounds)
    ):
        raise ValueError("A door body has invalid bounds.")
    normalized_targets = np.asarray(target_extents, dtype=float)
    if normalized_targets.shape != (3,) or np.any(
        ~np.isfinite(normalized_targets)
    ) or np.any(normalized_targets <= DOOR_GEOMETRY_EPSILON):
        raise ValueError("Door body target dimensions are invalid.")
    reconstructed_bounds = _reconstructed_body_bounds(model, mirror)
    authored_extents = authored_bounds[1] - authored_bounds[0]
    use_authored_bounds = np.isclose(
        authored_extents,
        normalized_targets,
        rtol=1e-5,
        atol=1e-7,
    )
    effective_bounds = reconstructed_bounds.copy()
    effective_bounds[:, use_authored_bounds] = authored_bounds[
        :, use_authored_bounds
    ]
    return effective_bounds


def _reconstructed_body_bounds(
    model: GeneratedModel,
    mirror: DoorBodyMirrorConfiguration | None,
) -> np.ndarray:
    """Return authored bounds expanded by every runtime body reflection."""

    bounds = np.asarray(model.mesh.bounds, dtype=float)
    if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
        raise ValueError("A door body has invalid bounds.")
    minimum = bounds[0].copy()
    maximum = bounds[1].copy()
    if mirror is None:
        return np.stack((minimum, maximum))

    mirror_planes: list[tuple[int, float]] = []
    if mirror.symmetric_orientation is not None:
        assert mirror.symmetric_plane_coordinate is not None
        mirror_planes.append(
            (
                SYMMETRIC_PREVIEW_AXIS_BY_ORIENTATION[
                    mirror.symmetric_orientation
                ],
                mirror.symmetric_plane_coordinate,
            )
        )
    if mirror.side_duplication_plane_coordinate is not None:
        mirror_planes.append((1, mirror.side_duplication_plane_coordinate))
    for axis, plane_coordinate in mirror_planes:
        reflected_minimum = 2.0 * plane_coordinate - maximum[axis]
        reflected_maximum = 2.0 * plane_coordinate - minimum[axis]
        minimum[axis] = min(minimum[axis], reflected_minimum)
        maximum[axis] = max(maximum[axis], reflected_maximum)
    return np.stack((minimum, maximum))


def _reconstructed_body_world_position(
    model: GeneratedModel,
    mirror: DoorBodyMirrorConfiguration | None,
    *,
    target_position: Sequence[float],
    rotation_degrees: Sequence[float],
    axis_scales: Sequence[float],
    target_extents: Sequence[float],
) -> tuple[float, float, float]:
    """Anchor a reconstructed body's bottom center on its canonical slot.

    Placed generated models normally pivot around the retained mesh's bottom
    center.  A symmetric body contains only one authored half, so that pivot
    is not the center of the body reconstructed by the viewer.  Compensating
    the placement here keeps paired generated leaves flush at the centerline.
    """

    authored_bounds = np.asarray(model.mesh.bounds, dtype=float)
    reconstructed_bounds = _body_fitting_bounds(
        model,
        mirror,
        target_extents=target_extents,
    )
    authored_anchor = _bounds_bottom_center(authored_bounds)
    reconstructed_anchor = _bounds_bottom_center(reconstructed_bounds)
    local_offset = np.asarray(axis_scales, dtype=float) * (
        authored_anchor - reconstructed_anchor
    )
    rotation = trimesh.transformations.euler_matrix(
        *np.radians(np.asarray(rotation_degrees, dtype=float)),
        axes="sxyz",
    )[:3, :3]
    world_position = np.asarray(target_position, dtype=float) + rotation @ local_offset
    return tuple(float(value) for value in world_position)


def _bounds_bottom_center(bounds: np.ndarray) -> np.ndarray:
    """Return the Z-up bottom-center anchor of finite XYZ bounds."""

    normalized = np.asarray(bounds, dtype=float)
    if normalized.shape != (2, 3) or not np.all(np.isfinite(normalized)):
        raise ValueError("Door body bounds are invalid.")
    return np.asarray(
        (
            (normalized[0, 0] + normalized[1, 0]) * 0.5,
            (normalized[0, 1] + normalized[1, 1]) * 0.5,
            normalized[0, 2],
        ),
        dtype=float,
    )


def _transform_body_mirror_configuration(
    mirror: DoorBodyMirrorConfiguration | None,
    *,
    horizontal_mirror: bool,
    depth_mirror: bool,
) -> DoorBodyMirrorConfiguration | None:
    """Follow slot-level reflections with the body's local mirror planes."""

    if mirror is None:
        return None
    symmetric_plane = mirror.symmetric_plane_coordinate
    if mirror.symmetric_orientation is not None:
        symmetric_axis = SYMMETRIC_PREVIEW_AXIS_BY_ORIENTATION[
            mirror.symmetric_orientation
        ]
        if (symmetric_axis == 0 and horizontal_mirror) or (
            symmetric_axis == 1 and depth_mirror
        ):
            assert symmetric_plane is not None
            symmetric_plane = -symmetric_plane
    side_plane = mirror.side_duplication_plane_coordinate
    if side_plane is not None and depth_mirror:
        side_plane = -side_plane
    return DoorBodyMirrorConfiguration(
        symmetric_orientation=mirror.symmetric_orientation,
        symmetric_plane_coordinate=symmetric_plane,
        side_duplication_plane_coordinate=side_plane,
    )


def mirror_door_model_horizontally(model: GeneratedModel) -> GeneratedModel:
    """Return a material-preserving copy reflected across local X=0."""

    return _mirror_door_model_across_axis(model, 0)


def mirror_door_model_depthwise(model: GeneratedModel) -> GeneratedModel:
    """Return a material-preserving copy reflected across local Y=0."""

    return _mirror_door_model_across_axis(model, 1)


def _mirror_door_model_across_axis(
    model: GeneratedModel,
    axis: int,
) -> GeneratedModel:
    """Reflect one door component while preserving every preview payload."""

    if not isinstance(model, GeneratedModel):
        raise TypeError("Only GeneratedModel doors can be mirrored.")
    if axis not in {0, 1}:
        raise ValueError("Door models can only be mirrored across X or Y.")
    reflection = np.eye(4, dtype=float)
    reflection[axis, axis] = -1.0

    preview_mesh = copy.deepcopy(model.mesh)
    _reflect_mesh_axis(preview_mesh, axis)
    scene = copy.deepcopy(model.scene)
    scene.apply_transform(
        Z_UP_TO_GLTF_Y_UP_TRANSFORM
        @ reflection
        @ GLTF_Y_UP_TO_Z_UP_TRANSFORM
    )
    glb_bytes = scene.export(file_type="glb") if scene.geometry else b""

    preview_untextured_mesh = None
    if model.preview_untextured_mesh is not None:
        preview_untextured_mesh = copy.deepcopy(model.preview_untextured_mesh)
        _reflect_mesh_axis(preview_untextured_mesh, axis)
    preview_base_mesh = None
    if model.preview_base_mesh is not None:
        preview_base_mesh = copy.deepcopy(model.preview_base_mesh)
        _reflect_mesh_axis(preview_base_mesh, axis)
    return GeneratedModel(
        mesh=preview_mesh,
        scene=scene,
        glb_bytes=bytes(glb_bytes),
        preview_textured_walls=list(model.preview_textured_walls),
        preview_textured_surfaces=list(model.preview_textured_surfaces),
        preview_untextured_mesh=preview_untextured_mesh,
        object_texture_variants=model.object_texture_variants,
        preview_symmetric_objects=_reflect_symmetric_previews(
            model.preview_symmetric_objects,
            axis,
        ),
        preview_placed_objects=list(model.preview_placed_objects),
        preview_stair_parts=list(model.preview_stair_parts),
        preview_architectural_trim_parts=list(
            model.preview_architectural_trim_parts
        ),
        preview_base_mesh=preview_base_mesh,
    )


def _reflect_symmetric_previews(
    previews: Sequence[PreviewSymmetricObject],
    axis: int,
) -> list[PreviewSymmetricObject]:
    """Keep embedded reconstruction previews aligned with a mirrored door."""

    reflected_previews: list[PreviewSymmetricObject] = []
    for preview in previews:
        retained_meshes = tuple(
            _reflected_mesh(mesh, axis) for mesh in preview.meshes
        )
        mirrored_meshes = tuple(
            _reflected_mesh(mesh, axis) for mesh in preview.mirrored_meshes
        )
        plane_coordinate = preview.plane_coordinate
        if SYMMETRIC_PREVIEW_AXIS_BY_ORIENTATION[preview.orientation] == axis:
            plane_coordinate *= -1.0
        reflected_previews.append(
            PreviewSymmetricObject(
                object_id=preview.object_id,
                meshes=retained_meshes,
                orientation=preview.orientation,
                plane_coordinate=plane_coordinate,
                mirrored_meshes=mirrored_meshes,
                fade_enabled=preview.fade_enabled,
            )
        )
    return reflected_previews


def _reflected_mesh(
    mesh: trimesh.Trimesh,
    axis: int,
) -> trimesh.Trimesh:
    reflected = copy.deepcopy(mesh)
    _reflect_mesh_axis(reflected, axis)
    return reflected


# ### Profile extrusion helpers ###
def _extrude_profile(
    closed_outline: Sequence[tuple[float, float]],
    thickness_meters: float,
) -> trimesh.Trimesh:
    points = np.asarray(closed_outline, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1:] != (2,)
        or len(points) < 4
        or not np.all(np.isfinite(points))
    ):
        raise ValueError("Door profiles must contain a valid closed polygon.")
    if np.linalg.norm(points[0] - points[-1]) > DOOR_GEOMETRY_EPSILON:
        raise ValueError("Door profiles must be closed.")
    outline = points[:-1]
    if len(outline) < 3:
        raise ValueError("Door profiles require at least three unique points.")
    center = np.mean(outline, axis=0)
    half_thickness = thickness_meters / 2.0
    vertex_count = len(outline)
    vertices = np.empty((vertex_count * 2 + 2, 3), dtype=float)
    vertices[:vertex_count, 0] = outline[:, 0]
    vertices[:vertex_count, 1] = -half_thickness
    vertices[:vertex_count, 2] = outline[:, 1]
    vertices[vertex_count : vertex_count * 2, 0] = outline[:, 0]
    vertices[vertex_count : vertex_count * 2, 1] = half_thickness
    vertices[vertex_count : vertex_count * 2, 2] = outline[:, 1]
    front_center_index = vertex_count * 2
    back_center_index = front_center_index + 1
    vertices[front_center_index] = (center[0], -half_thickness, center[1])
    vertices[back_center_index] = (center[0], half_thickness, center[1])

    faces: list[tuple[int, int, int]] = []
    for point_index in range(vertex_count):
        next_index = (point_index + 1) % vertex_count
        front_current = point_index
        front_next = next_index
        back_current = vertex_count + point_index
        back_next = vertex_count + next_index
        faces.extend(
            (
                (front_center_index, front_current, front_next),
                (back_center_index, back_next, back_current),
                (front_current, back_current, back_next),
                (front_current, back_next, front_next),
            )
        )
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    mesh.remove_unreferenced_vertices()
    return mesh


# ### Generated-model helpers ###
def _build_generated_model(
    mesh: trimesh.Trimesh,
    geometry_name: str,
) -> GeneratedModel:
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError("Door components require a non-empty triangle mesh.")
    normalized_name = str(geometry_name).strip()
    if not normalized_name:
        raise ValueError("Door component geometry names cannot be empty.")
    preview_mesh = copy.deepcopy(mesh)
    export_mesh = copy.deepcopy(mesh)
    export_mesh.apply_transform(Z_UP_TO_GLTF_Y_UP_TRANSFORM)
    scene = trimesh.Scene()
    scene.add_geometry(
        export_mesh,
        geom_name=normalized_name,
        node_name=normalized_name,
    )
    return GeneratedModel(
        mesh=preview_mesh,
        scene=scene,
        glb_bytes=bytes(scene.export(file_type="glb")),
    )


def _resolve_slot_model(
    models: Mapping[str, GeneratedModel],
    slot: DoorSlotData,
) -> GeneratedModel | None:
    model = models.get(slot.slot_id)
    if model is None:
        model = models.get(slot.source_object_id)
    if model is None:
        return None
    if not isinstance(model, GeneratedModel):
        raise TypeError(
            f"Door slot {slot.display_name!r} source must be a GeneratedModel."
        )
    return model


def _unwrap_mesh(
    mesh: trimesh.Trimesh,
    rgba: tuple[int, int, int, int],
    material_name: str,
) -> trimesh.Trimesh:
    """Create a complete, non-overlapping UV atlas for one door component."""

    source_vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    source_faces = np.ascontiguousarray(mesh.faces, dtype=np.uint32)
    atlas = xatlas.Atlas()
    atlas.add_mesh(source_vertices, source_faces)
    chart_options = xatlas.ChartOptions()
    chart_options.fix_winding = False
    chart_options.use_input_mesh_uvs = False
    pack_options = xatlas.PackOptions()
    pack_options.resolution = DOOR_UV_ATLAS_RESOLUTION
    pack_options.padding = DOOR_UV_PADDING_PIXELS
    pack_options.bilinear = True
    pack_options.blockAlign = False
    pack_options.bruteForce = False
    pack_options.create_image = False
    pack_options.rotate_charts = True
    pack_options.rotate_charts_to_axis = True
    try:
        atlas.generate(chart_options, pack_options, False)
    except RuntimeError as error:
        raise ValueError("Door component UV unwrapping failed.") from error
    if int(atlas.atlas_count) != 1:
        raise ValueError("Door component UVs require exactly one texture atlas.")

    vertex_mapping, output_faces, output_uv = atlas[0]
    vertex_mapping = np.ascontiguousarray(vertex_mapping, dtype=np.int64)
    output_faces = np.ascontiguousarray(output_faces, dtype=np.int64)
    output_uv = np.ascontiguousarray(output_uv, dtype=np.float64)
    if (
        output_faces.shape != source_faces.shape
        or vertex_mapping.ndim != 1
        or output_uv.shape != (len(vertex_mapping), 2)
        or np.any(output_faces < 0)
        or np.any(output_faces >= len(vertex_mapping))
        or np.any(vertex_mapping < 0)
        or np.any(vertex_mapping >= len(source_vertices))
        or not np.all(np.isfinite(output_uv))
        or np.any(output_uv < -DOOR_GEOMETRY_EPSILON)
        or np.any(output_uv > 1.0 + DOOR_GEOMETRY_EPSILON)
        or not np.array_equal(vertex_mapping[output_faces], source_faces)
    ):
        raise ValueError("Door component UV unwrapping returned invalid data.")

    material = PBRMaterial(
        name=str(material_name),
        baseColorFactor=list(rgba),
        metallicFactor=0.0,
        roughnessFactor=0.7,
    )
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64)[vertex_mapping],
        faces=output_faces,
        visual=TextureVisuals(
            uv=output_uv,
            material=material,
        ),
        metadata=copy.deepcopy(mesh.metadata),
        process=False,
        validate=False,
    )


def _reflect_mesh_axis(mesh: trimesh.Trimesh, axis: int) -> None:
    vertices = np.asarray(mesh.vertices, dtype=float).copy()
    vertices[:, axis] *= -1.0
    mesh.vertices = vertices
    mesh.faces = np.ascontiguousarray(
        np.asarray(mesh.faces, dtype=np.int64)[:, (0, 2, 1)]
    )


# ### Validation helpers ###
def _require_door_definition(door: object) -> DoorDefinition:
    if not isinstance(door, DoorDefinition):
        raise TypeError("Door geometry requires a DoorDefinition.")
    return door


def _require_window_definition(window: object) -> WindowDefinition:
    if not isinstance(window, WindowDefinition):
        raise TypeError("Window geometry requires a WindowDefinition.")
    return window


def _normalize_positive_measurement(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a number.")
    try:
        measurement = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{label} must be a number.") from error
    if not math.isfinite(measurement) or measurement <= 0.0:
        raise ValueError(f"{label} must be finite and greater than zero.")
    return measurement
