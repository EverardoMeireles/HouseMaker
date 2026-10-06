# ### Imports ###
from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import trimesh
import xatlas
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.door_state import (
    DOOR_SLOT_BODY,
    DOOR_SLOT_HINGES,
    DOOR_SLOT_KNOB,
    DoorDefinition,
    DoorSlotData,
)
from housemaker.doorway_geometry import build_doorway_cross_section_outline
from housemaker.glb import (
    DOOR_BODY_MARKER_METADATA_KEY,
    SYMMETRIC_PREVIEW_AXIS_BY_ORIENTATION,
    Z_UP_TO_GLTF_Y_UP_TRANSFORM,
    GeneratedModel,
    PlacedGeneratedModel,
    PreviewSymmetricObject,
    compose_placed_generated_models,
)

# ### Constants ###
DOOR_BODY_COLOR = (164, 109, 68, 255)
DOOR_HARDWARE_COLOR = (132, 138, 145, 255)
DOOR_BODY_GEOMETRY_NAME = "door_body"
DOOR_HINGES_GEOMETRY_NAME = "door_hinges"
DOOR_KNOB_GEOMETRY_NAME = "door_knob"
DOOR_HINGE_CYLINDER_SECTIONS = 20
DOOR_KNOB_SPHERE_SUBDIVISIONS = 2
DOOR_GEOMETRY_EPSILON = 1e-9
DOOR_UV_ATLAS_RESOLUTION = 1024
DOOR_UV_PADDING_PIXELS = 2


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

    thickness = _normalize_positive_measurement(
        thickness_meters,
        "Door thickness",
    )
    outline = build_doorway_cross_section_outline(
        width_meters,
        height_meters,
        shape,
        arch_amount,
    )
    mesh = _extrude_profile(outline, thickness)
    mesh = _unwrap_mesh(mesh, DOOR_BODY_COLOR, geometry_name)
    return _build_generated_model(mesh, geometry_name)


# ### Public hardware construction ###
def build_door_hinges_model(
    door_height_meters: float,
    *,
    radius_meters: float = 0.012,
    geometry_name: str = DOOR_HINGES_GEOMETRY_NAME,
) -> GeneratedModel:
    """Build two simple hinge barrels in one movable component slot."""

    door_height = _normalize_positive_measurement(
        door_height_meters,
        "Door height",
    )
    radius = _normalize_positive_measurement(radius_meters, "Hinge radius")
    barrel_height = min(max(door_height * 0.06, 0.06), 0.14)
    barrel_separation = min(
        max(door_height * 0.45, barrel_height * 1.5),
        max(door_height - barrel_height, barrel_height * 1.5),
    )
    barrels: list[trimesh.Trimesh] = []
    for base_z in (0.0, barrel_separation):
        barrel = trimesh.creation.cylinder(
            radius=radius,
            height=barrel_height,
            sections=DOOR_HINGE_CYLINDER_SECTIONS,
        )
        barrel.apply_translation((0.0, 0.0, base_z + barrel_height / 2.0))
        barrels.append(barrel)
    mesh = _concatenate_meshes(barrels)
    mesh = _unwrap_mesh(mesh, DOOR_HARDWARE_COLOR, geometry_name)
    return _build_generated_model(mesh, geometry_name)


def build_door_knob_model(
    door_thickness_meters: float,
    *,
    radius_meters: float = 0.035,
    projection_meters: float = 0.035,
    geometry_name: str = DOOR_KNOB_GEOMETRY_NAME,
) -> GeneratedModel:
    """Build paired knobs and their spindle as one movable component slot."""

    thickness = _normalize_positive_measurement(
        door_thickness_meters,
        "Door thickness",
    )
    radius = _normalize_positive_measurement(radius_meters, "Door-knob radius")
    projection = _normalize_positive_measurement(
        projection_meters,
        "Door-knob projection",
    )
    center_z = radius
    center_y = thickness / 2.0 + projection
    front = trimesh.creation.icosphere(
        subdivisions=DOOR_KNOB_SPHERE_SUBDIVISIONS,
        radius=radius,
    )
    front.apply_translation((0.0, -center_y, center_z))
    back = trimesh.creation.icosphere(
        subdivisions=DOOR_KNOB_SPHERE_SUBDIVISIONS,
        radius=radius,
    )
    back.apply_translation((0.0, center_y, center_z))

    spindle = trimesh.creation.cylinder(
        radius=max(radius * 0.22, 0.003),
        height=center_y * 2.0,
        sections=DOOR_HINGE_CYLINDER_SECTIONS,
    )
    spindle.apply_transform(
        trimesh.transformations.rotation_matrix(math.pi / 2.0, (1.0, 0.0, 0.0))
    )
    spindle.apply_translation((0.0, 0.0, center_z))
    mesh = _concatenate_meshes((front, back, spindle))
    mesh = _unwrap_mesh(mesh, DOOR_HARDWARE_COLOR, geometry_name)
    return _build_generated_model(mesh, geometry_name)


def build_default_door_slot_models(
    door: DoorDefinition,
) -> dict[str, GeneratedModel]:
    """Build all canonical procedural source models for a new door."""

    _require_door_definition(door)
    return {
        DOOR_SLOT_BODY: build_door_body_model(
            door.width_meters,
            door.height_meters,
            door.thickness_meters,
            shape=door.shape,
            arch_amount=door.arch_amount,
        ),
        DOOR_SLOT_HINGES: build_door_hinges_model(door.height_meters),
        DOOR_SLOT_KNOB: build_door_knob_model(door.thickness_meters),
    }


# ### Door assembly ###
def assemble_door_model(
    door: DoorDefinition,
    slot_models: Mapping[str, GeneratedModel] | None = None,
    *,
    include_unjoined: bool = False,
    body_mirror: DoorBodyMirrorConfiguration | None = None,
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
    models = (
        build_default_door_slot_models(door)
        if slot_models is None
        else dict(slot_models)
    )
    placements: list[PlacedGeneratedModel] = []
    for slot in door.slots:
        if (
            slot.slot_id != DOOR_SLOT_BODY
            and not include_unjoined
            and not slot.joined
        ):
            continue
        component_model = _resolve_slot_model(models, slot)
        if slot.slot_id == DOOR_SLOT_BODY:
            component_model = _with_door_body_marker(
                component_model,
                slot.source_object_id,
            )
        world_position = slot.position_meters
        if slot.slot_id == DOOR_SLOT_BODY:
            bottom_center = _mesh_bottom_center(component_model.mesh)
            world_position = tuple(
                float(slot.position_meters[index] + bottom_center[index])
                for index in range(3)
            )
        placements.append(
            PlacedGeneratedModel(
                object_id=slot.source_object_id,
                source_object_id=slot.source_object_id,
                object_name=f"{door.name} {slot.display_name}",
                model=component_model,
                world_position=world_position,
                symmetric_preview_orientation=(
                    None
                    if slot.slot_id != DOOR_SLOT_BODY or body_mirror is None
                    else body_mirror.symmetric_orientation
                ),
                symmetric_preview_plane_coordinate=(
                    None
                    if slot.slot_id != DOOR_SLOT_BODY or body_mirror is None
                    else body_mirror.symmetric_plane_coordinate
                ),
                rotation_degrees=slot.rotation_degrees,
            )
        )
    # Every slot must pass through the placed-model preview splitter so the
    # body's embedded texture is not flattened into an untextured mesh.
    empty_base = GeneratedModel(
        mesh=trimesh.Trimesh(process=False),
        scene=trimesh.Scene(),
        glb_bytes=b"",
    )
    assembled = compose_placed_generated_models(empty_base, placements)
    if (
        body_mirror is not None
        and body_mirror.side_duplication_plane_coordinate is not None
    ):
        _append_side_duplication_previews(
            assembled,
            door.get_slot(DOOR_SLOT_BODY),
            body_mirror,
        )
    return assembled


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


def _append_side_duplication_previews(
    assembled: GeneratedModel,
    body_slot: DoorSlotData | None,
    body_mirror: DoorBodyMirrorConfiguration,
) -> None:
    """Show Y and optional X+Y mirrors while keeping hardware single."""

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
    side_plane = body_mirror.side_duplication_plane_coordinate
    assert side_plane is not None
    side_mirrors = _reflect_preview_meshes(world_meshes, 1, side_plane)
    assembled.preview_symmetric_objects.append(
        PreviewSymmetricObject(
            object_id=f"{body_slot.source_object_id}:side",
            meshes=world_meshes,
            orientation="depth",
            plane_coordinate=side_plane,
            mirrored_meshes=side_mirrors,
        )
    )
    if body_mirror.symmetric_orientation is None:
        return
    symmetric_axis = SYMMETRIC_PREVIEW_AXIS_BY_ORIENTATION[
        body_mirror.symmetric_orientation
    ]
    symmetric_plane = body_mirror.symmetric_plane_coordinate
    assert symmetric_plane is not None
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


def _mesh_bottom_center(mesh: trimesh.Trimesh) -> np.ndarray:
    minimum, maximum = np.asarray(mesh.bounds, dtype=float)
    return np.asarray(
        (
            (minimum[0] + maximum[0]) * 0.5,
            (minimum[1] + maximum[1]) * 0.5,
            minimum[2],
        ),
        dtype=float,
    )


def mirror_door_model_horizontally(model: GeneratedModel) -> GeneratedModel:
    """Return a material-preserving copy reflected across local X=0."""

    if not isinstance(model, GeneratedModel):
        raise TypeError("Only GeneratedModel doors can be mirrored.")
    reflection = np.eye(4, dtype=float)
    reflection[0, 0] = -1.0

    preview_mesh = copy.deepcopy(model.mesh)
    _reflect_mesh_x(preview_mesh)
    scene = copy.deepcopy(model.scene)
    scene.apply_transform(reflection)
    glb_bytes = scene.export(file_type="glb") if scene.geometry else b""

    preview_untextured_mesh = None
    if model.preview_untextured_mesh is not None:
        preview_untextured_mesh = copy.deepcopy(model.preview_untextured_mesh)
        _reflect_mesh_x(preview_untextured_mesh)
    preview_base_mesh = None
    if model.preview_base_mesh is not None:
        preview_base_mesh = copy.deepcopy(model.preview_base_mesh)
        _reflect_mesh_x(preview_base_mesh)
    return GeneratedModel(
        mesh=preview_mesh,
        scene=scene,
        glb_bytes=bytes(glb_bytes),
        preview_textured_walls=list(model.preview_textured_walls),
        preview_textured_surfaces=list(model.preview_textured_surfaces),
        preview_untextured_mesh=preview_untextured_mesh,
        object_texture_variants=model.object_texture_variants,
        preview_symmetric_objects=_reflect_symmetric_previews_x(
            model.preview_symmetric_objects
        ),
        preview_placed_objects=list(model.preview_placed_objects),
        preview_stair_parts=list(model.preview_stair_parts),
        preview_architectural_trim_parts=list(
            model.preview_architectural_trim_parts
        ),
        preview_base_mesh=preview_base_mesh,
    )


def _reflect_symmetric_previews_x(
    previews: Sequence[PreviewSymmetricObject],
) -> list[PreviewSymmetricObject]:
    """Keep embedded body-only mirrors aligned with a mirrored door."""

    reflected_previews: list[PreviewSymmetricObject] = []
    for preview in previews:
        retained_meshes = tuple(
            _reflected_mesh_x(mesh) for mesh in preview.meshes
        )
        mirrored_meshes = tuple(
            _reflected_mesh_x(mesh) for mesh in preview.mirrored_meshes
        )
        plane_coordinate = preview.plane_coordinate
        if SYMMETRIC_PREVIEW_AXIS_BY_ORIENTATION[preview.orientation] == 0:
            plane_coordinate *= -1.0
        reflected_previews.append(
            PreviewSymmetricObject(
                object_id=preview.object_id,
                meshes=retained_meshes,
                orientation=preview.orientation,
                plane_coordinate=plane_coordinate,
                mirrored_meshes=mirrored_meshes,
            )
        )
    return reflected_previews


def _reflected_mesh_x(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    reflected = copy.deepcopy(mesh)
    _reflect_mesh_x(reflected)
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
) -> GeneratedModel:
    model = models.get(slot.slot_id)
    if model is None:
        model = models.get(slot.source_object_id)
    if model is None:
        raise ValueError(
            f"Door slot {slot.display_name!r} has no generated source model."
        )
    if not isinstance(model, GeneratedModel):
        raise TypeError(
            f"Door slot {slot.display_name!r} source must be a GeneratedModel."
        )
    return model


def _concatenate_meshes(meshes: Sequence[trimesh.Trimesh]) -> trimesh.Trimesh:
    normalized = tuple(mesh for mesh in meshes if not mesh.is_empty)
    if not normalized:
        raise ValueError("Door component geometry cannot be empty.")
    combined = trimesh.util.concatenate(normalized)
    if not isinstance(combined, trimesh.Trimesh):
        raise TypeError("Door component geometry could not be combined.")
    return combined


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


def _reflect_mesh_x(mesh: trimesh.Trimesh) -> None:
    vertices = np.asarray(mesh.vertices, dtype=float).copy()
    vertices[:, 0] *= -1.0
    mesh.vertices = vertices
    mesh.faces = np.ascontiguousarray(
        np.asarray(mesh.faces, dtype=np.int64)[:, (0, 2, 1)]
    )


# ### Validation helpers ###
def _require_door_definition(door: object) -> DoorDefinition:
    if not isinstance(door, DoorDefinition):
        raise TypeError("Door geometry requires a DoorDefinition.")
    return door


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
