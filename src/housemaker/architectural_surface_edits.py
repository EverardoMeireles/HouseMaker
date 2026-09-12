# ### Imports ###
from __future__ import annotations

import math
import re
import uuid
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import trimesh
from shapely import LineString, Point, Polygon, constrained_delaunay_triangles
from shapely.errors import ShapelyError
from shapely.geometry.polygon import orient
from shapely.ops import split as split_geometry
from shapely.ops import unary_union

from housemaker.level_coordinates import (
    level_image_to_world_xy,
    level_world_to_image_xy,
)
from housemaker.models import (
    EDITABLE_SURFACE_FRAME_LEVEL_IMAGE,
    EDITABLE_SURFACE_FRAME_WALL_RATIO,
    EditableSurfaceEdgeData,
    EditableSurfaceFaceData,
    EditableSurfaceMeshData,
    EditableSurfaceVertexData,
    LevelData,
)
from housemaker.surface_geometry import (
    SURFACE_TYPE_CEILING,
    SURFACE_TYPE_FLOOR,
    SURFACE_TYPE_WALL,
    SURFACE_TYPES,
    FixedSurface,
    build_base_fixed_surfaces,
)

# ### Constants ###
EDITABLE_SURFACE_ID_PATTERN = re.compile(
    r"^level:(?P<level_index>0|[1-9]\d*)/"
    r"edit-face:(?P<face_id>[0-9a-f]{32}):"
    r"(?P<surface_type>wall|floor|ceiling)$"
)
WALL_KEY_PATTERN = re.compile(r"^(?P<first>[1-9]\d*):(?P<second>[1-9]\d*)$")
SURFACE_EDIT_EPSILON_METERS = 1e-7
SURFACE_EDIT_PLANAR_TOLERANCE_METERS = 1e-5
SURFACE_EDIT_NORMAL_DOT_TOLERANCE = 1.0 - 1e-6
# Candidate rays use architectural 45-degree increments. Nearby vertices only
# influence a point within 2 m, and every snap type uses the same 1 cm capture
# tolerance.
SURFACE_VERTEX_SNAP_ANGLE_DEGREES = 45.0
SURFACE_VERTEX_SNAP_MAX_DISPLACEMENT_METERS = 0.01
SURFACE_VERTEX_SNAP_NEARBY_RADIUS_METERS = 2.0
SURFACE_SIDE_HORIZONTAL_NORMAL_THRESHOLD = math.sqrt(0.5)
LOCAL_COORDINATE_KEY_DECIMALS = 10
SURFACE_VERTEX_SNAP_KIND_SURFACE = "surface"
SURFACE_VERTEX_SNAP_KIND_ANGLE = "angle"
SURFACE_VERTEX_SNAP_KIND_EDGE = "edge"
SURFACE_VERTEX_SNAP_KIND_VERTEX = "vertex"


# ### Public request and result models ###
@dataclass(frozen=True)
class SurfaceVertexInsertionRequest:
    surface_id: str
    world_point: tuple[float, float, float]
    active_vertex_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "surface_id",
            _normalize_surface_id(self.surface_id),
        )
        object.__setattr__(
            self,
            "world_point",
            _normalize_world_point(self.world_point),
        )
        if self.active_vertex_id is not None:
            object.__setattr__(
                self,
                "active_vertex_id",
                _normalize_uuid(self.active_vertex_id, "active vertex ID"),
            )


@dataclass(frozen=True)
class SurfaceFaceExtrusionRequest:
    surface_ids: tuple[str, ...]
    delta_meters: float

    def __post_init__(self) -> None:
        try:
            surface_ids = tuple(
                dict.fromkeys(
                    _normalize_surface_id(value) for value in self.surface_ids
                )
            )
        except TypeError as error:
            raise ValueError(
                "Surface extrusion IDs must contain a sequence."
            ) from error
        if not surface_ids:
            raise ValueError("Select at least one surface face to extrude.")
        delta = _normalize_finite_number(self.delta_meters, "extrusion distance")
        if abs(delta) <= SURFACE_EDIT_EPSILON_METERS:
            raise ValueError("Surface extrusion distance must not be zero.")
        object.__setattr__(self, "surface_ids", surface_ids)
        object.__setattr__(self, "delta_meters", delta)


@dataclass(frozen=True)
class SurfaceTopologyEditResult:
    """Texture-target replacements plus the faces selected after one edit."""

    replacements: Mapping[str, tuple[str, ...]]
    selected_surface_ids: tuple[str, ...]
    active_vertex_id: str | None = None
    created_edge_vertex_ids: tuple[str, str] | None = None
    created_surface_ids: tuple[str, ...] = ()
    requires_mesh_refresh: bool = False
    state_changed: bool = False

    def __post_init__(self) -> None:
        normalized_replacements = {
            _normalize_surface_id(source_id): tuple(
                dict.fromkeys(
                    _normalize_surface_id(target_id) for target_id in target_ids
                )
            )
            for source_id, target_ids in self.replacements.items()
        }
        selected_ids = tuple(
            dict.fromkeys(
                _normalize_surface_id(surface_id)
                for surface_id in self.selected_surface_ids
            )
        )
        active_vertex_id = (
            None
            if self.active_vertex_id is None
            else _normalize_uuid(self.active_vertex_id, "active vertex ID")
        )
        created_edge_vertex_ids = self.created_edge_vertex_ids
        if created_edge_vertex_ids is not None:
            if len(created_edge_vertex_ids) != 2:
                raise ValueError("A created surface edge requires two vertex IDs.")
            created_edge_vertex_ids = tuple(
                _normalize_uuid(vertex_id, "created edge vertex ID")
                for vertex_id in created_edge_vertex_ids
            )
            if created_edge_vertex_ids[0] == created_edge_vertex_ids[1]:
                raise ValueError("A created surface edge requires two vertices.")
        created_surface_ids = tuple(
            dict.fromkeys(
                _normalize_surface_id(surface_id)
                for surface_id in self.created_surface_ids
            )
        )
        if not isinstance(self.requires_mesh_refresh, bool):
            raise TypeError("Surface mesh refresh state must be a boolean.")
        if not isinstance(self.state_changed, bool):
            raise TypeError("Surface topology state change must be a boolean.")
        object.__setattr__(self, "replacements", normalized_replacements)
        object.__setattr__(self, "selected_surface_ids", selected_ids)
        object.__setattr__(self, "active_vertex_id", active_vertex_id)
        object.__setattr__(
            self,
            "created_edge_vertex_ids",
            created_edge_vertex_ids,
        )
        object.__setattr__(self, "created_surface_ids", created_surface_ids)


@dataclass(frozen=True)
class SurfaceVertexPreview:
    """One exact hover/click resolution for the continuous vertex tool."""

    surface_id: str
    source_surface_id: str
    world_point: tuple[float, float, float]
    snapped_vertex_id: str | None = None
    snapped_edge_vertex_ids: tuple[str, str] | None = None
    snap_kind: str = "surface"


@dataclass(frozen=True)
class SurfaceDrawingVertexTarget:
    """One snap vertex plus its optional manual-marker ownership."""

    vertex_id: str
    source_surface_id: str
    world_point: tuple[float, float, float]
    show_marker: bool = True
    direct_face_surface_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SurfaceDrawingEdgeTarget:
    """One unfinished authored edge exposed to the viewer overlay."""

    source_surface_id: str
    vertex_ids: tuple[str, str]
    world_points: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
    ]


@dataclass(frozen=True)
class SurfaceDrawingOverlay:
    """Snap vertices and unfinished topology rendered outside surface meshes."""

    vertices: tuple[SurfaceDrawingVertexTarget, ...]
    edges: tuple[SurfaceDrawingEdgeTarget, ...]


# ### Internal models ###
@dataclass(frozen=True)
class _SurfaceFrame:
    kind: str
    normal_sign: int
    origin_world: np.ndarray
    u_vector_world: np.ndarray
    v_vector_world: np.ndarray
    normal_world: np.ndarray


@dataclass(frozen=True)
class _ReversibleExtrusionLayer:
    """The immediately preceding cap-to-base extrusion side ring."""

    cap_to_base_vertex_ids: tuple[tuple[str, str], ...]
    side_face_ids: tuple[str, ...]
    depth_meters: float
    side_vertex_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class _InsertionContext:
    level: LevelData
    source_surface: FixedSurface
    editable_mesh: EditableSurfaceMeshData
    target_face: EditableSurfaceFaceData
    frame: _SurfaceFrame
    is_first_edit: bool

    @property
    def target_face_surface_id(self) -> str:
        if not self.editable_mesh.replaces_source_surface:
            return self.source_surface.surface_id
        return build_editable_surface_id(
            self.level.index,
            self.target_face.face_id,
            self.target_face.surface_type,
        )


# ### Stable surface identity helpers ###
def build_editable_surface_id(
    level_index: int,
    face_id: str,
    surface_type: str,
) -> str:
    """Build the stable semantic ID exposed by one authored polygon face."""

    normalized_level_index = _normalize_level_index(level_index)
    normalized_face_id = _normalize_uuid(face_id, "face ID")
    normalized_type = _normalize_surface_type(surface_type)
    return (
        f"level:{normalized_level_index}/edit-face:"
        f"{normalized_face_id}:{normalized_type}"
    )


def parse_editable_surface_id(surface_id: object) -> tuple[int, str, str] | None:
    """Parse an authored face ID without accepting partial or legacy IDs."""

    match = EDITABLE_SURFACE_ID_PATTERN.fullmatch(str(surface_id).strip().lower())
    if match is None:
        return None
    return (
        int(match.group("level_index")),
        match.group("face_id"),
        match.group("surface_type"),
    )


# ### Public topology operations ###
def snap_surface_vertex_world_point(
    surfaces: Sequence[FixedSurface],
    surface_id: str,
    world_point: Sequence[float],
) -> tuple[float, float, float]:
    """Apply insertion projection and 45-degree snapping to current surfaces."""

    surface_sequence = tuple(surfaces)
    if not all(isinstance(surface, FixedSurface) for surface in surface_sequence):
        raise TypeError("Surface vertex snapping requires FixedSurface values.")
    normalized_id = _normalize_surface_id(surface_id)
    target = next(
        (
            surface
            for surface in surface_sequence
            if surface.surface_id == normalized_id
        ),
        None,
    )
    if target is None:
        raise ValueError("The selected surface no longer exists.")
    requested = np.asarray(_normalize_world_point(world_point), dtype=float)
    target_points, normal = _get_surface_insertion_polygon(target, requested)
    frame = _fallback_face_frame(target_points, normal)
    root_id = target.source_surface_id or target.surface_id
    nearby_points = np.asarray(
        [
            vertex
            for surface in surface_sequence
            if (surface.source_surface_id or surface.surface_id) == root_id
            for vertex in np.asarray(surface.mesh.vertices, dtype=float)
        ],
        dtype=float,
    )
    resolved = _snap_point_in_polygon(
        requested,
        target_points,
        nearby_points,
        frame,
    )
    return tuple(float(value) for value in resolved)


def resolve_surface_vertex_insertion_point(
    levels: Sequence[LevelData],
    surface_id: str,
    world_point: Sequence[float],
    active_vertex_id: str | None = None,
) -> tuple[float, float, float]:
    """Resolve the exact raw-or-snapped point used by vertex placement."""

    return resolve_surface_vertex_preview(
        levels,
        surface_id=surface_id,
        world_point=world_point,
        active_vertex_id=active_vertex_id,
    ).world_point


def resolve_surface_vertex_preview(
    levels: Sequence[LevelData],
    surface_id: str,
    world_point: Sequence[float],
    active_vertex_id: str | None = None,
) -> SurfaceVertexPreview:
    """Resolve hover feedback with stable vertex and edge snap identities."""

    request = SurfaceVertexInsertionRequest(
        surface_id=surface_id,
        world_point=tuple(world_point),
        active_vertex_id=active_vertex_id,
    )
    level_sequence = _normalize_levels(levels)
    context = _build_insertion_context(
        level_sequence,
        request.surface_id,
        np.asarray(request.world_point, dtype=float),
    )
    return _resolve_context_vertex_preview(
        context,
        np.asarray(request.world_point, dtype=float),
        request.active_vertex_id,
    )


def build_surface_drawing_overlay(
    levels: Sequence[LevelData],
) -> SurfaceDrawingOverlay:
    """Build snap targets, manual markers, and unfinished chain edges."""

    level_sequence = _normalize_levels(levels)
    base_by_id = {
        surface.surface_id: surface
        for surface in build_base_fixed_surfaces(level_sequence)
    }
    vertex_targets: list[SurfaceDrawingVertexTarget] = []
    edge_targets: list[SurfaceDrawingEdgeTarget] = []
    for level in sorted(level_sequence, key=lambda candidate: candidate.index):
        for editable_mesh in sorted(
            level.editable_surfaces,
            key=lambda candidate: candidate.source_surface_id,
        ):
            if not editable_mesh.faces:
                continue
            source_surface = base_by_id.get(editable_mesh.source_surface_id)
            if source_surface is None:
                continue
            frame = _build_surface_frame(
                level,
                source_surface,
                editable_mesh.frame_kind,
                editable_mesh.frame_normal_sign,
            )
            world_vertices = _build_world_vertex_lookup(
                level,
                source_surface,
                editable_mesh,
                frame,
            )
            visible_vertex_ids = _get_overlay_vertex_ids(editable_mesh)
            direct_surface_ids_by_vertex: dict[str, list[str]] = defaultdict(list)
            for face in editable_mesh.faces:
                if not face.is_directly_drawn:
                    continue
                direct_surface_id = build_editable_surface_id(
                    level.index,
                    face.face_id,
                    face.surface_type,
                )
                for vertex_id in face.vertex_ids:
                    direct_surface_ids_by_vertex[vertex_id].append(
                        direct_surface_id
                    )
            open_chain_edges = _get_open_chain_edges(editable_mesh.edges)
            vertex_targets.extend(
                SurfaceDrawingVertexTarget(
                    vertex_id=vertex.vertex_id,
                    source_surface_id=editable_mesh.source_surface_id,
                    world_point=tuple(
                        float(component)
                        for component in world_vertices[vertex.vertex_id]
                    ),
                    show_marker=vertex.is_user_placed,
                    direct_face_surface_ids=tuple(
                        direct_surface_ids_by_vertex.get(vertex.vertex_id, ())
                    ),
                )
                for vertex in editable_mesh.vertices
                if vertex.vertex_id in visible_vertex_ids
            )
            edge_targets.extend(
                SurfaceDrawingEdgeTarget(
                    source_surface_id=editable_mesh.source_surface_id,
                    vertex_ids=(edge.start_vertex_id, edge.end_vertex_id),
                    world_points=tuple(
                        tuple(
                            float(component)
                            for component in world_vertices[vertex_id]
                        )
                        for vertex_id in (
                            edge.start_vertex_id,
                            edge.end_vertex_id,
                        )
                    ),
                )
                for edge in open_chain_edges
            )
    return SurfaceDrawingOverlay(
        vertices=tuple(vertex_targets),
        edges=tuple(edge_targets),
    )


def place_surface_vertex(
    levels: Sequence[LevelData],
    surface_id: str,
    world_point: Sequence[float],
    active_vertex_id: str | None = None,
) -> SurfaceTopologyEditResult:
    """Select or place a vertex and extend the persistent authored edge chain."""

    request = SurfaceVertexInsertionRequest(
        surface_id=surface_id,
        world_point=tuple(world_point),
        active_vertex_id=active_vertex_id,
    )
    level_sequence = _normalize_levels(levels)
    context = _build_insertion_context(
        level_sequence,
        request.surface_id,
        np.asarray(request.world_point, dtype=float),
    )
    preview = _resolve_context_vertex_preview(
        context,
        np.asarray(request.world_point, dtype=float),
        request.active_vertex_id,
    )
    next_mesh, placed_vertex_id, vertex_was_added = _place_preview_vertex(
        context,
        preview,
    )
    state_changed = context.is_first_edit or vertex_was_added
    created_edge: tuple[str, str] | None = None
    created_surface_ids: tuple[str, ...] = ()
    replacements: dict[str, tuple[str, ...]] = {}
    selected_surface_ids = (request.surface_id,)
    mesh_was_subdivided = False

    if request.active_vertex_id is not None:
        _validate_active_vertex(
            context,
            next_mesh,
            request.active_vertex_id,
            placed_vertex_id,
        )
        if request.active_vertex_id != placed_vertex_id:
            world_vertices = _build_world_vertex_lookup(
                context.level,
                context.source_surface,
                next_mesh,
                context.frame,
            )
            candidate_edge_key = frozenset(
                (request.active_vertex_id, placed_vertex_id)
            )
            edge_already_exists = any(
                frozenset((edge.start_vertex_id, edge.end_vertex_id))
                == candidate_edge_key
                for edge in next_mesh.edges
            )
            if not edge_already_exists:
                _validate_new_authored_edge(
                    context,
                    next_mesh,
                    request.active_vertex_id,
                    placed_vertex_id,
                    world_vertices,
                )
                existing_path = _find_authored_edge_path(
                    next_mesh.edges,
                    request.active_vertex_id,
                    placed_vertex_id,
                )
                created_edge = (request.active_vertex_id, placed_vertex_id)
                state_changed = True
                next_mesh = replace(
                    next_mesh,
                    edges=(
                        *next_mesh.edges,
                        EditableSurfaceEdgeData(
                            start_vertex_id=created_edge[0],
                            end_vertex_id=created_edge[1],
                        ),
                    ),
                )
                if existing_path is not None:
                    (
                        next_mesh,
                        replacements,
                        created_surface_ids,
                    ) = _subdivide_faces_with_loop(
                        context,
                        next_mesh,
                        existing_path,
                    )
                    selected_surface_ids = created_surface_ids
                    mesh_was_subdivided = True
                else:
                    open_paths = _find_open_chain_paths_containing_edge(
                        next_mesh.edges,
                        created_edge,
                    )
                    for open_path in open_paths:
                        open_partition = _subdivide_faces_with_open_path(
                            context,
                            next_mesh,
                            open_path,
                        )
                        if open_partition is not None:
                            (
                                next_mesh,
                                replacements,
                                selected_surface_ids,
                            ) = open_partition
                            mesh_was_subdivided = True
                            break

    _validate_editable_mesh_geometry(
        context.level,
        context.source_surface,
        next_mesh,
    )
    _replace_level_editable_mesh(context.level, next_mesh)
    return SurfaceTopologyEditResult(
        replacements=replacements,
        selected_surface_ids=selected_surface_ids,
        active_vertex_id=placed_vertex_id,
        created_edge_vertex_ids=created_edge,
        created_surface_ids=created_surface_ids,
        requires_mesh_refresh=mesh_was_subdivided,
        state_changed=state_changed,
    )


def delete_directly_drawn_surface_faces(
    levels: Sequence[LevelData],
    surface_ids: Sequence[str],
) -> SurfaceTopologyEditResult:
    """Delete selected faces authored by the Add vertices tool.

    Automatically generated remainder and extrusion-side faces are deliberately
    retained. Boundary edges owned only by deleted faces are removed so their
    closed loops cannot block the same region from being drawn again.
    """

    level_sequence = _normalize_levels(levels)
    try:
        normalized_ids = tuple(
            dict.fromkeys(_normalize_surface_id(value) for value in surface_ids)
        )
    except TypeError as error:
        raise ValueError("Surface deletion IDs must contain a sequence.") from error
    if not normalized_ids:
        raise ValueError(
            "Select at least one face created with Add vertices to delete."
        )

    levels_by_index = {level.index: level for level in level_sequence}
    selected_by_mesh: dict[
        tuple[int, str],
        set[str],
    ] = defaultdict(set)
    mesh_by_key: dict[tuple[int, str], EditableSurfaceMeshData] = {}
    source_by_key: dict[tuple[int, str], FixedSurface] = {}
    base_surfaces_by_id = {
        surface.surface_id: surface
        for surface in build_base_fixed_surfaces(level_sequence)
    }

    for surface_id in normalized_ids:
        parsed = parse_editable_surface_id(surface_id)
        if parsed is None:
            raise ValueError("Only faces created with Add vertices can be deleted.")
        level_index, face_id, surface_type = parsed
        level = levels_by_index.get(level_index)
        if level is None:
            raise ValueError("The selected editable surface level no longer exists.")
        editable_mesh, face = _find_editable_face(level, face_id)
        if face.surface_type != surface_type or not face.is_directly_drawn:
            raise ValueError("Only faces created with Add vertices can be deleted.")
        source_surface = base_surfaces_by_id.get(editable_mesh.source_surface_id)
        if source_surface is None:
            raise ValueError("The edited source surface no longer exists.")
        key = (level.index, editable_mesh.source_surface_id)
        selected_by_mesh[key].add(face.face_id)
        mesh_by_key[key] = editable_mesh
        source_by_key[key] = source_surface

    replacements: dict[str, tuple[str, ...]] = {
        surface_id: () for surface_id in normalized_ids
    }
    pending_meshes: list[tuple[LevelData, EditableSurfaceMeshData]] = []
    for key, selected_face_ids in selected_by_mesh.items():
        level = levels_by_index[key[0]]
        editable_mesh = mesh_by_key[key]
        remaining_faces = tuple(
            face
            for face in editable_mesh.faces
            if face.face_id not in selected_face_ids
        )
        deleted_boundary_keys = {
            frozenset(edge)
            for face in editable_mesh.faces
            if face.face_id in selected_face_ids
            for edge in _iter_face_edges(face)
        }
        retained_direct_boundary_keys = {
            frozenset(edge)
            for face in remaining_faces
            if face.is_directly_drawn
            for edge in _iter_face_edges(face)
        }
        retained_edges = (
            tuple(
                edge
                for edge in editable_mesh.edges
                if (
                    frozenset((edge.start_vertex_id, edge.end_vertex_id))
                    not in deleted_boundary_keys
                    or frozenset((edge.start_vertex_id, edge.end_vertex_id))
                    in retained_direct_boundary_keys
                )
            )
            if remaining_faces
            else ()
        )
        retained_vertex_ids = {
            vertex_id
            for face in remaining_faces
            for vertex_id in face.vertex_ids
        }
        retained_vertex_ids.update(
            vertex_id
            for edge in retained_edges
            for vertex_id in (edge.start_vertex_id, edge.end_vertex_id)
        )
        next_mesh = replace(
            editable_mesh,
            vertices=tuple(
                vertex
                for vertex in editable_mesh.vertices
                if (
                    not remaining_faces
                    or vertex.vertex_id in retained_vertex_ids
                )
            ),
            faces=remaining_faces,
            edges=retained_edges,
            replaces_source_surface=True,
        )
        _validate_editable_mesh_geometry(
            level,
            source_by_key[key],
            next_mesh,
        )
        pending_meshes.append((level, next_mesh))

    for level, editable_mesh in pending_meshes:
        _replace_level_editable_mesh(level, editable_mesh)
    return SurfaceTopologyEditResult(
        replacements=replacements,
        selected_surface_ids=(),
        requires_mesh_refresh=True,
        state_changed=True,
    )


def insert_surface_vertex(
    levels: Sequence[LevelData],
    surface_id: str,
    world_point: Sequence[float],
) -> SurfaceTopologyEditResult:
    """Legacy one-shot insertion retained for non-interactive callers."""

    request = SurfaceVertexInsertionRequest(
        surface_id=surface_id,
        world_point=tuple(world_point),
    )
    level_sequence = _normalize_levels(levels)
    current_surfaces = apply_editable_surfaces(
        level_sequence,
        build_base_fixed_surfaces(level_sequence),
    )
    insertion_world = np.asarray(
        snap_surface_vertex_world_point(
            current_surfaces,
            request.surface_id,
            request.world_point,
        ),
        dtype=float,
    )
    context = _build_insertion_context(
        level_sequence,
        request.surface_id,
        insertion_world,
        draft_mode=False,
    )
    world_vertices = _build_world_vertex_lookup(
        context.level,
        context.source_surface,
        context.editable_mesh,
        context.frame,
    )
    _validate_resolved_insertion_point(
        insertion_world,
        context.target_face,
        world_vertices,
        context.frame,
    )
    insertion_local = _world_to_local(
        context.level,
        context.frame,
        insertion_world,
    )
    inserted_vertex = EditableSurfaceVertexData(
        vertex_id=_new_uuid(),
        u=insertion_local[0],
        v=insertion_local[1],
        normal_offset_meters=insertion_local[2],
        is_user_placed=True,
    )
    child_faces = tuple(
        EditableSurfaceFaceData(
            face_id=_new_uuid(),
            vertex_ids=(
                context.target_face.vertex_ids[edge_index],
                context.target_face.vertex_ids[
                    (edge_index + 1) % len(context.target_face.vertex_ids)
                ],
                inserted_vertex.vertex_id,
            ),
            surface_type=context.target_face.surface_type,
        )
        for edge_index in range(len(context.target_face.vertex_ids))
    )
    next_faces = tuple(
        child
        for face in context.editable_mesh.faces
        for child in (
            child_faces
            if face.face_id == context.target_face.face_id
            else (face,)
        )
    )
    next_mesh = replace(
        context.editable_mesh,
        vertices=(*context.editable_mesh.vertices, inserted_vertex),
        faces=next_faces,
        replaces_source_surface=True,
    )
    _validate_editable_mesh_geometry(
        context.level,
        context.source_surface,
        next_mesh,
    )
    _replace_level_editable_mesh(context.level, next_mesh)
    child_surface_ids = tuple(
        build_editable_surface_id(
            context.level.index,
            child.face_id,
            child.surface_type,
        )
        for child in child_faces
    )
    replacements = (
        {
            context.source_surface.surface_id: tuple(
                build_editable_surface_id(
                    context.level.index,
                    face.face_id,
                    face.surface_type,
                )
                for face in next_mesh.faces
            )
        }
        if context.is_first_edit
        else {request.surface_id: child_surface_ids}
    )
    return SurfaceTopologyEditResult(
        replacements=replacements,
        selected_surface_ids=child_surface_ids,
        active_vertex_id=inserted_vertex.vertex_id,
        requires_mesh_refresh=True,
        state_changed=True,
    )


def extrude_surface_faces(
    levels: Sequence[LevelData],
    surface_ids: Sequence[str],
    delta_meters: float,
) -> SurfaceTopologyEditResult:
    """Extrude a face group while retaining at most one generated side ring."""

    request = SurfaceFaceExtrusionRequest(
        surface_ids=tuple(surface_ids),
        delta_meters=delta_meters,
    )
    level_sequence = _normalize_levels(levels)
    context = _find_extrusion_context(level_sequence, request.surface_ids)
    level, source_surface, editable_mesh, selected_faces, frame = context
    next_mesh, replacements = _apply_extrusion_to_editable_mesh(
        level,
        source_surface,
        editable_mesh,
        tuple(face.face_id for face in selected_faces),
        frame,
        request.delta_meters,
    )
    _validate_editable_mesh_geometry(level, source_surface, next_mesh)
    _replace_level_editable_mesh(level, next_mesh)
    retained_ids = tuple(
        build_editable_surface_id(level.index, face.face_id, face.surface_type)
        for face in selected_faces
    )
    return SurfaceTopologyEditResult(
        replacements=replacements,
        selected_surface_ids=retained_ids,
        requires_mesh_refresh=True,
        state_changed=True,
    )


def apply_editable_surfaces(
    levels: Sequence[LevelData],
    base_surfaces: Sequence[FixedSurface],
) -> list[FixedSurface]:
    """Replace edited roots with their current stable polygon surfaces."""

    level_sequence = _normalize_levels(levels)
    base_sequence = tuple(base_surfaces)
    if not all(isinstance(surface, FixedSurface) for surface in base_sequence):
        raise TypeError("Editable surface application requires FixedSurface values.")
    editable_by_source: dict[str, tuple[LevelData, EditableSurfaceMeshData]] = {}
    for level in level_sequence:
        for editable_mesh in level.editable_surfaces:
            if editable_mesh.source_surface_id in editable_by_source:
                raise ValueError(
                    "Editable source surfaces must be unique across the project."
                )
            editable_by_source[editable_mesh.source_surface_id] = (
                level,
                editable_mesh,
            )

    surfaces: list[FixedSurface] = []
    for source_surface in base_sequence:
        editable_entry = editable_by_source.get(source_surface.surface_id)
        if editable_entry is None:
            surfaces.append(source_surface)
            continue
        level, editable_mesh = editable_entry
        if level.index != source_surface.level_index:
            raise ValueError("An editable surface belongs to the wrong level.")
        if not editable_mesh.replaces_source_surface:
            surfaces.append(source_surface)
            continue
        frame = _build_surface_frame(
            level,
            source_surface,
            editable_mesh.frame_kind,
            editable_mesh.frame_normal_sign,
        )
        world_vertices = _build_world_vertex_lookup(
            level,
            source_surface,
            editable_mesh,
            frame,
        )
        for face in editable_mesh.faces:
            mesh = _build_face_mesh(face, world_vertices)
            area = float(mesh.area)
            if not math.isfinite(area) or area <= SURFACE_EDIT_EPSILON_METERS:
                raise ValueError("An editable surface face has no measurable area.")
            surfaces.append(
                FixedSurface(
                    surface_id=build_editable_surface_id(
                        level.index,
                        face.face_id,
                        face.surface_type,
                    ),
                    surface_type=face.surface_type,
                    level_index=source_surface.level_index,
                    room_index=source_surface.room_index,
                    mesh=mesh,
                    area_square_meters=area,
                    source_surface_id=source_surface.surface_id,
                    is_directly_drawn=face.is_directly_drawn,
                )
            )
    return surfaces


# ### Insertion preparation ###
def _build_insertion_context(
    levels: tuple[LevelData, ...],
    surface_id: str,
    world_point: np.ndarray,
    *,
    draft_mode: bool = True,
) -> _InsertionContext:
    base_surfaces = build_base_fixed_surfaces(levels)
    base_by_id = {surface.surface_id: surface for surface in base_surfaces}
    parsed_id = parse_editable_surface_id(surface_id)
    if parsed_id is None:
        source_surface = base_by_id.get(surface_id)
        if source_surface is None:
            raise ValueError("The selected source surface no longer exists.")
        level = _find_level(levels, source_surface.level_index)
        existing_mesh = next(
            (
                mesh
                for mesh in level.editable_surfaces
                if mesh.source_surface_id == source_surface.surface_id
            ),
            None,
        )
        if existing_mesh is not None and (
            existing_mesh.replaces_source_surface or not draft_mode
        ):
            raise ValueError("Select one of the source surface's editable faces.")
        editable_mesh = existing_mesh or _snapshot_source_surface(
            level,
            source_surface,
            replaces_source_surface=not draft_mode,
        )
        frame = _build_surface_frame(
            level,
            source_surface,
            editable_mesh.frame_kind,
            editable_mesh.frame_normal_sign,
        )
        target_face = _find_face_containing_or_touching_world_point(
            editable_mesh.faces,
            _build_world_vertex_lookup(
                level,
                source_surface,
                editable_mesh,
                frame,
            ),
            world_point,
        )
        return _InsertionContext(
            level=level,
            source_surface=source_surface,
            editable_mesh=editable_mesh,
            target_face=target_face,
            frame=frame,
            is_first_edit=existing_mesh is None,
        )

    level_index, face_id, surface_type = parsed_id
    level = _find_level(levels, level_index)
    editable_mesh, target_face = _find_editable_face(level, face_id)
    if target_face.surface_type != surface_type:
        raise ValueError("The selected editable surface ID has the wrong type.")
    source_surface = base_by_id.get(editable_mesh.source_surface_id)
    if source_surface is None:
        raise ValueError("The editable source surface no longer exists.")
    frame = _build_surface_frame(
        level,
        source_surface,
        editable_mesh.frame_kind,
        editable_mesh.frame_normal_sign,
    )
    return _InsertionContext(
        level=level,
        source_surface=source_surface,
        editable_mesh=editable_mesh,
        target_face=target_face,
        frame=frame,
        is_first_edit=False,
    )


def _snapshot_source_surface(
    level: LevelData,
    source_surface: FixedSurface,
    *,
    replaces_source_surface: bool = True,
) -> EditableSurfaceMeshData:
    frame_kind = (
        EDITABLE_SURFACE_FRAME_WALL_RATIO
        if source_surface.surface_type == SURFACE_TYPE_WALL
        else EDITABLE_SURFACE_FRAME_LEVEL_IMAGE
    )
    normal_sign = _get_source_frame_normal_sign(level, source_surface, frame_kind)
    frame = _build_surface_frame(level, source_surface, frame_kind, normal_sign)
    mesh_vertices = np.asarray(source_surface.mesh.vertices, dtype=float)
    mesh_faces = np.asarray(source_surface.mesh.faces, dtype=np.int64)
    source_face_indices: list[tuple[int, int, int]] = []
    for raw_face in mesh_faces:
        indices = tuple(int(index) for index in raw_face)
        if (
            len(indices) != 3
            or len(set(indices)) != 3
            or any(index < 0 or index >= len(mesh_vertices) for index in indices)
        ):
            continue
        triangle = mesh_vertices[np.asarray(indices, dtype=np.int64)]
        triangle_area = float(
            np.linalg.norm(
                np.cross(
                    triangle[1] - triangle[0],
                    triangle[2] - triangle[0],
                )
            )
            * 0.5
        )
        if (
            not math.isfinite(triangle_area)
            or triangle_area <= SURFACE_EDIT_EPSILON_METERS
        ):
            continue
        source_face_indices.append(indices)
    if not source_face_indices:
        raise ValueError("The selected surface has no measurable faces.")

    used_source_vertex_indices = sorted(
        {index for face in source_face_indices for index in face}
    )
    vertices: list[EditableSurfaceVertexData] = []
    vertex_id_by_key: dict[tuple[float, float, float], str] = {}
    source_vertex_ids: dict[int, str] = {}
    for source_index in used_source_vertex_indices:
        world_point = mesh_vertices[source_index]
        local = _world_to_local(level, frame, world_point)
        key = tuple(
            round(float(value), LOCAL_COORDINATE_KEY_DECIMALS) for value in local
        )
        vertex_id = vertex_id_by_key.get(key)
        if vertex_id is None:
            vertex_id = _new_uuid()
            vertex_id_by_key[key] = vertex_id
            vertices.append(
                EditableSurfaceVertexData(
                    vertex_id=vertex_id,
                    u=local[0],
                    v=local[1],
                    normal_offset_meters=local[2],
                )
            )
        source_vertex_ids[source_index] = vertex_id
    faces = tuple(
        EditableSurfaceFaceData(
            face_id=_new_uuid(),
            vertex_ids=tuple(source_vertex_ids[int(index)] for index in face),
            surface_type=source_surface.surface_type,
        )
        for face in source_face_indices
        if len({source_vertex_ids[index] for index in face}) == 3
    )
    return EditableSurfaceMeshData(
        source_surface_id=source_surface.surface_id,
        frame_kind=frame_kind,
        frame_normal_sign=normal_sign,
        vertices=tuple(vertices),
        faces=faces,
        replaces_source_surface=replaces_source_surface,
    )


# ### Continuous drawing helpers ###
def _resolve_context_vertex_preview(
    context: _InsertionContext,
    requested_world: np.ndarray,
    active_vertex_id: str | None,
) -> SurfaceVertexPreview:
    world_vertices = _build_world_vertex_lookup(
        context.level,
        context.source_surface,
        context.editable_mesh,
        context.frame,
    )
    target_points = np.asarray(
        [world_vertices[vertex_id] for vertex_id in context.target_face.vertex_ids],
        dtype=float,
    )
    normal = _get_polygon_normal(target_points)
    projected = requested_world - normal * float(
        np.dot(requested_world - target_points[0], normal)
    )
    basis_u, basis_v = _build_face_plane_basis(context.frame, normal)
    origin = target_points[0]
    face_entries = _build_coplanar_face_polygons(
        context.editable_mesh.faces,
        world_vertices,
        origin,
        normal,
        basis_u,
        basis_v,
    )
    region = unary_union([entry[1] for entry in face_entries])
    projected_2d = _project_point_to_basis(projected, origin, basis_u, basis_v)
    requested_point = Point(float(projected_2d[0]), float(projected_2d[1]))
    if not region.buffer(SURFACE_EDIT_EPSILON_METERS).covers(requested_point):
        raise ValueError("A surface vertex must stay on the selected surface.")

    available_vertex_ids = _get_overlay_vertex_ids(context.editable_mesh)
    coplanar_vertex_ids = tuple(
        vertex_id
        for vertex_id, vertex_world in world_vertices.items()
        if vertex_id in available_vertex_ids
        if abs(float(np.dot(vertex_world - origin, normal)))
        <= SURFACE_EDIT_PLANAR_TOLERANCE_METERS
    )
    direct_vertex_candidates = sorted(
        (
            round(float(np.linalg.norm(world_vertices[vertex_id] - projected)), 12),
            vertex_id,
        )
        for vertex_id in coplanar_vertex_ids
        if float(np.linalg.norm(world_vertices[vertex_id] - projected))
        <= SURFACE_VERTEX_SNAP_MAX_DISPLACEMENT_METERS
    )
    if direct_vertex_candidates:
        _distance, vertex_id = direct_vertex_candidates[0]
        return SurfaceVertexPreview(
            surface_id=context.target_face_surface_id,
            source_surface_id=context.source_surface.surface_id,
            world_point=tuple(
                float(component) for component in world_vertices[vertex_id]
            ),
            snapped_vertex_id=vertex_id,
            snap_kind=SURFACE_VERTEX_SNAP_KIND_VERTEX,
        )

    face_edge_counts: dict[tuple[str, str], int] = defaultdict(int)
    for face, _polygon in face_entries:
        for first, second in _iter_face_edges(face):
            face_edge_counts[tuple(sorted((first, second)))] += 1
    edge_keys = {
        edge_key
        for edge_key, owner_count in face_edge_counts.items()
        if context.editable_mesh.replaces_source_surface or owner_count == 1
    }
    authored_edge_keys = {
        tuple(sorted((edge.start_vertex_id, edge.end_vertex_id)))
        for edge in context.editable_mesh.edges
        if edge.start_vertex_id in coplanar_vertex_ids
        and edge.end_vertex_id in coplanar_vertex_ids
    }
    edge_keys.update(authored_edge_keys)
    edge_candidates: list[
        tuple[float, int, float, tuple[str, str], np.ndarray]
    ] = []
    for edge_key in sorted(edge_keys):
        first_2d = _project_point_to_basis(
            world_vertices[edge_key[0]],
            origin,
            basis_u,
            basis_v,
        )
        second_2d = _project_point_to_basis(
            world_vertices[edge_key[1]],
            origin,
            basis_u,
            basis_v,
        )
        line = LineString((first_2d, second_2d))
        if line.length <= SURFACE_EDIT_EPSILON_METERS:
            continue
        snapped_2d = np.asarray(
            line.interpolate(line.project(requested_point)).coords[0],
            dtype=float,
        )
        displacement = float(np.linalg.norm(snapped_2d - projected_2d))
        if displacement > SURFACE_VERTEX_SNAP_MAX_DISPLACEMENT_METERS:
            continue
        snapped_world = origin + basis_u * snapped_2d[0] + basis_v * snapped_2d[1]
        edge_candidates.append((
            displacement,
            0 if edge_key in authored_edge_keys else 1,
            float(line.length),
            edge_key,
            snapped_world,
        ))
    if edge_candidates:
        (
            _distance,
            _authored_priority,
            _edge_length,
            edge_key,
            snapped_world,
        ) = min(
            edge_candidates,
            key=lambda candidate: (
                round(candidate[0], 12),
                candidate[1],
                round(candidate[2], 12),
                candidate[3],
            ),
        )
        return SurfaceVertexPreview(
            surface_id=context.target_face_surface_id,
            source_surface_id=context.source_surface.surface_id,
            world_point=tuple(float(component) for component in snapped_world),
            snapped_edge_vertex_ids=edge_key,
            snap_kind=SURFACE_VERTEX_SNAP_KIND_EDGE,
        )

    reference_ids = list(coplanar_vertex_ids)
    if active_vertex_id in reference_ids:
        reference_ids.remove(active_vertex_id)
        reference_ids.insert(0, active_vertex_id)
    angle_candidates: list[tuple[tuple[object, ...], np.ndarray]] = []
    for reference_index, vertex_id in enumerate(reference_ids):
        vertex_world = world_vertices[vertex_id]
        if (
            float(np.linalg.norm(vertex_world - projected))
            > SURFACE_VERTEX_SNAP_NEARBY_RADIUS_METERS
        ):
            continue
        vertex_2d = _project_point_to_basis(
            vertex_world,
            origin,
            basis_u,
            basis_v,
        )
        raw_delta = projected_2d - vertex_2d
        for angle_index in range(8):
            angle = math.radians(angle_index * SURFACE_VERTEX_SNAP_ANGLE_DEGREES)
            ray = np.asarray((math.cos(angle), math.sin(angle)), dtype=float)
            ray_distance = float(np.dot(raw_delta, ray))
            candidate_2d = vertex_2d + ray * ray_distance
            displacement = float(np.linalg.norm(candidate_2d - projected_2d))
            if displacement > SURFACE_VERTEX_SNAP_MAX_DISPLACEMENT_METERS:
                continue
            candidate_point = Point(
                float(candidate_2d[0]),
                float(candidate_2d[1]),
            )
            if not region.buffer(SURFACE_EDIT_EPSILON_METERS).covers(candidate_point):
                continue
            candidate_world = (
                origin + basis_u * candidate_2d[0] + basis_v * candidate_2d[1]
            )
            angle_candidates.append(
                (
                    (
                        0 if vertex_id == active_vertex_id else 1,
                        round(displacement, 12),
                        reference_index,
                        angle_index,
                        vertex_id,
                    ),
                    candidate_world,
                )
            )
    resolved_world = projected
    snap_kind = SURFACE_VERTEX_SNAP_KIND_SURFACE
    if angle_candidates:
        _key, resolved_world = min(
            angle_candidates,
            key=lambda candidate: candidate[0],
        )
        snap_kind = SURFACE_VERTEX_SNAP_KIND_ANGLE
    return SurfaceVertexPreview(
        surface_id=context.target_face_surface_id,
        source_surface_id=context.source_surface.surface_id,
        world_point=tuple(float(component) for component in resolved_world),
        snap_kind=snap_kind,
    )


def _place_preview_vertex(
    context: _InsertionContext,
    preview: SurfaceVertexPreview,
) -> tuple[EditableSurfaceMeshData, str, bool]:
    if preview.snapped_vertex_id is not None:
        return context.editable_mesh, preview.snapped_vertex_id, False

    insertion_local = _world_to_local(
        context.level,
        context.frame,
        preview.world_point,
    )
    inserted_vertex = EditableSurfaceVertexData(
        vertex_id=_new_uuid(),
        u=insertion_local[0],
        v=insertion_local[1],
        normal_offset_meters=insertion_local[2],
        is_user_placed=True,
    )
    next_edges: list[EditableSurfaceEdgeData] = []
    split_edge_key = (
        None
        if preview.snapped_edge_vertex_ids is None
        else frozenset(preview.snapped_edge_vertex_ids)
    )
    for edge in context.editable_mesh.edges:
        if frozenset((edge.start_vertex_id, edge.end_vertex_id)) != split_edge_key:
            next_edges.append(edge)
            continue
        next_edges.extend(
            (
                EditableSurfaceEdgeData(
                    start_vertex_id=edge.start_vertex_id,
                    end_vertex_id=inserted_vertex.vertex_id,
                ),
                EditableSurfaceEdgeData(
                    start_vertex_id=inserted_vertex.vertex_id,
                    end_vertex_id=edge.end_vertex_id,
                ),
            )
        )
    return (
        replace(
            context.editable_mesh,
            vertices=(*context.editable_mesh.vertices, inserted_vertex),
            edges=tuple(next_edges),
        ),
        inserted_vertex.vertex_id,
        True,
    )


def _validate_active_vertex(
    context: _InsertionContext,
    editable_mesh: EditableSurfaceMeshData,
    active_vertex_id: str,
    placed_vertex_id: str,
) -> None:
    known_vertex_ids = {vertex.vertex_id for vertex in editable_mesh.vertices}
    if active_vertex_id not in known_vertex_ids:
        raise ValueError("The active surface vertex no longer exists.")
    if placed_vertex_id not in known_vertex_ids:
        raise ValueError("The placed surface vertex no longer exists.")
    world_vertices = _build_world_vertex_lookup(
        context.level,
        context.source_surface,
        editable_mesh,
        context.frame,
    )
    target_points = np.asarray(
        [world_vertices[vertex_id] for vertex_id in context.target_face.vertex_ids],
        dtype=float,
    )
    normal = _get_polygon_normal(target_points)
    if abs(
        float(
            np.dot(
                world_vertices[active_vertex_id] - target_points[0],
                normal,
            )
        )
    ) > SURFACE_EDIT_PLANAR_TOLERANCE_METERS:
        raise ValueError("Connected surface vertices must be coplanar.")


def _validate_new_authored_edge(
    context: _InsertionContext,
    editable_mesh: EditableSurfaceMeshData,
    start_vertex_id: str,
    end_vertex_id: str,
    world_vertices: Mapping[str, np.ndarray],
) -> None:
    new_key = frozenset((start_vertex_id, end_vertex_id))
    if any(
        frozenset((edge.start_vertex_id, edge.end_vertex_id)) == new_key
        for edge in editable_mesh.edges
    ):
        raise ValueError("That surface edge already exists.")

    target_points = np.asarray(
        [world_vertices[vertex_id] for vertex_id in context.target_face.vertex_ids],
        dtype=float,
    )
    normal = _get_polygon_normal(target_points)
    basis_u, basis_v = _build_face_plane_basis(context.frame, normal)
    origin = target_points[0]
    face_entries = _build_coplanar_face_polygons(
        editable_mesh.faces,
        world_vertices,
        origin,
        normal,
        basis_u,
        basis_v,
    )
    region = unary_union([entry[1] for entry in face_entries])
    start_2d = _project_point_to_basis(
        world_vertices[start_vertex_id],
        origin,
        basis_u,
        basis_v,
    )
    end_2d = _project_point_to_basis(
        world_vertices[end_vertex_id],
        origin,
        basis_u,
        basis_v,
    )
    new_line = LineString((start_2d, end_2d))
    if new_line.length <= SURFACE_EDIT_EPSILON_METERS:
        raise ValueError("A surface edge requires two distinct points.")
    if not region.buffer(SURFACE_EDIT_EPSILON_METERS).covers(new_line):
        raise ValueError("A surface edge cannot leave the selected surface.")

    for edge in editable_mesh.edges:
        edge_start = _project_point_to_basis(
            world_vertices[edge.start_vertex_id],
            origin,
            basis_u,
            basis_v,
        )
        edge_end = _project_point_to_basis(
            world_vertices[edge.end_vertex_id],
            origin,
            basis_u,
            basis_v,
        )
        existing_line = LineString((edge_start, edge_end))
        intersection = new_line.intersection(existing_line)
        if intersection.is_empty:
            continue
        shared_ids = new_key.intersection(
            (edge.start_vertex_id, edge.end_vertex_id)
        )
        if (
            intersection.geom_type == "Point"
            and len(shared_ids) == 1
            and np.linalg.norm(
                np.asarray(intersection.coords[0], dtype=float)
                - _project_point_to_basis(
                    world_vertices[next(iter(shared_ids))],
                    origin,
                    basis_u,
                    basis_v,
                )
            )
            <= SURFACE_EDIT_EPSILON_METERS
        ):
            continue
        raise ValueError("Surface edges cannot cross or overlap.")


def _find_authored_edge_path(
    edges: Sequence[EditableSurfaceEdgeData],
    start_vertex_id: str,
    end_vertex_id: str,
) -> tuple[str, ...] | None:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        adjacency[edge.start_vertex_id].add(edge.end_vertex_id)
        adjacency[edge.end_vertex_id].add(edge.start_vertex_id)
    queue = deque(((start_vertex_id, (start_vertex_id,)),))
    visited = {start_vertex_id}
    while queue:
        vertex_id, path = queue.popleft()
        for neighbor_id in sorted(adjacency.get(vertex_id, ())):
            if neighbor_id == end_vertex_id:
                return (*path, neighbor_id)
            if neighbor_id in visited:
                continue
            visited.add(neighbor_id)
            queue.append((neighbor_id, (*path, neighbor_id)))
    return None


# ### Open-path subdivision ###
def _find_open_chain_paths_containing_edge(
    edges: Sequence[EditableSurfaceEdgeData],
    required_edge: tuple[str, str],
) -> tuple[tuple[str, ...], ...]:
    """Return shortest bridge paths whose route contains the newly drawn edge."""

    open_edges = _get_open_chain_edges(edges)
    required_key = frozenset(required_edge)
    if not any(
        frozenset((edge.start_vertex_id, edge.end_vertex_id)) == required_key
        for edge in open_edges
    ):
        return ()

    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in open_edges:
        adjacency[edge.start_vertex_id].add(edge.end_vertex_id)
        adjacency[edge.end_vertex_id].add(edge.start_vertex_id)

    component_vertex_ids: set[str] = set()
    pending = [required_edge[0]]
    while pending:
        vertex_id = pending.pop()
        if vertex_id in component_vertex_ids:
            continue
        component_vertex_ids.add(vertex_id)
        pending.extend(adjacency.get(vertex_id, ()))
    endpoint_ids = sorted(
        vertex_id
        for vertex_id in component_vertex_ids
        if len(adjacency[vertex_id]) == 1
    )
    candidate_paths: list[tuple[str, ...]] = []
    for first_index, first_id in enumerate(endpoint_ids):
        for second_id in endpoint_ids[first_index + 1 :]:
            path = _find_authored_edge_path(
                open_edges,
                first_id,
                second_id,
            )
            if path is None or not any(
                frozenset((path[index], path[index + 1])) == required_key
                for index in range(len(path) - 1)
            ):
                continue
            candidate_paths.append(path)
    return tuple(
        sorted(
            candidate_paths,
            key=lambda path: (len(path), path),
        )
    )


def _subdivide_faces_with_open_path(
    context: _InsertionContext,
    editable_mesh: EditableSurfaceMeshData,
    path_vertex_ids: tuple[str, ...],
) -> tuple[
    EditableSurfaceMeshData,
    dict[str, tuple[str, ...]],
    tuple[str, ...],
] | None:
    """Commit an authored open path only when it partitions visible topology."""

    if len(path_vertex_ids) < 2 or len(set(path_vertex_ids)) != len(
        path_vertex_ids
    ):
        return None
    world_vertices = _build_world_vertex_lookup(
        context.level,
        context.source_surface,
        editable_mesh,
        context.frame,
    )
    target_world = np.asarray(
        [world_vertices[vertex_id] for vertex_id in context.target_face.vertex_ids],
        dtype=float,
    )
    normal = _get_polygon_normal(target_world)
    basis_u, basis_v = _build_face_plane_basis(context.frame, normal)
    origin = target_world[0]
    path_coordinates = tuple(
        _project_point_to_basis(
            world_vertices[vertex_id],
            origin,
            basis_u,
            basis_v,
        )
        for vertex_id in path_vertex_ids
    )
    path_line = LineString(path_coordinates)
    if (
        not path_line.is_valid
        or not path_line.is_simple
        or path_line.length <= SURFACE_EDIT_EPSILON_METERS
    ):
        return None

    face_entries = _build_coplanar_face_polygons(
        editable_mesh.faces,
        world_vertices,
        origin,
        normal,
        basis_u,
        basis_v,
    )
    vertices = list(editable_mesh.vertices)
    vertex_id_by_key = {
        _local_coordinate_key((
            vertex.u,
            vertex.v,
            vertex.normal_offset_meters,
        )): vertex.vertex_id
        for vertex in vertices
    }

    def get_vertex_id(coordinate: Sequence[float]) -> str:
        world = origin + basis_u * coordinate[0] + basis_v * coordinate[1]
        local = _world_to_local(context.level, context.frame, world)
        key = _local_coordinate_key(local)
        existing_id = vertex_id_by_key.get(key)
        if existing_id is not None:
            return existing_id
        vertex = EditableSurfaceVertexData(
            vertex_id=_new_uuid(),
            u=local[0],
            v=local[1],
            normal_offset_meters=local[2],
        )
        vertices.append(vertex)
        vertex_id_by_key[key] = vertex.vertex_id
        world_vertices[vertex.vertex_id] = world
        return vertex.vertex_id

    def build_faces(
        template: EditableSurfaceFaceData,
        polygons: Sequence[Polygon],
    ) -> tuple[EditableSurfaceFaceData, ...]:
        faces: list[EditableSurfaceFaceData] = []
        for polygon in polygons:
            ring = tuple(orient(polygon, sign=1.0).exterior.coords)[:-1]
            vertex_ids = _compact_polygon_vertex_ids(
                tuple(get_vertex_id(coordinate) for coordinate in ring)
            )
            if len(set(vertex_ids)) < 3:
                continue
            faces.append(
                EditableSurfaceFaceData(
                    face_id=_new_uuid(),
                    vertex_ids=vertex_ids,
                    surface_type=template.surface_type,
                )
            )
        return tuple(faces)

    if not editable_mesh.replaces_source_surface:
        source_region = unary_union(
            [polygon for _face, polygon in face_entries]
        )
        split_polygons = _split_surface_region(source_region, path_line)
        if split_polygons is None:
            return None
        next_faces = build_faces(context.target_face, split_polygons)
        if len(next_faces) < 2:
            return None
        next_mesh = replace(
            editable_mesh,
            vertices=tuple(vertices),
            faces=next_faces,
            edges=_node_open_path_edges_at_face_vertices(
                editable_mesh.edges,
                path_vertex_ids,
                next_faces,
                world_vertices,
            ),
            replaces_source_surface=True,
        )
        selected_surface_ids = tuple(
            build_editable_surface_id(
                context.level.index,
                face.face_id,
                face.surface_type,
            )
            for face in next_faces
        )
        return (
            next_mesh,
            {context.source_surface.surface_id: selected_surface_ids},
            selected_surface_ids,
        )

    generated_faces: dict[str, tuple[EditableSurfaceFaceData, ...]] = {}
    for face, face_polygon in face_entries:
        split_polygons = _split_surface_region(face_polygon, path_line)
        if split_polygons is None:
            continue
        child_faces = build_faces(face, split_polygons)
        if len(child_faces) < 2:
            continue
        generated_faces[face.face_id] = child_faces
    if not generated_faces:
        return None

    next_faces = tuple(
        child
        for face in editable_mesh.faces
        for child in generated_faces.get(face.face_id, (face,))
    )
    next_mesh = replace(
        editable_mesh,
        vertices=tuple(vertices),
        faces=next_faces,
        edges=_node_open_path_edges_at_face_vertices(
            editable_mesh.edges,
            path_vertex_ids,
            next_faces,
            world_vertices,
        ),
        replaces_source_surface=True,
    )
    replacements = {
        build_editable_surface_id(
            context.level.index,
            original_face.face_id,
            original_face.surface_type,
        ): tuple(
            build_editable_surface_id(
                context.level.index,
                child.face_id,
                child.surface_type,
            )
            for child in generated_faces[original_face.face_id]
        )
        for original_face, _polygon in face_entries
        if original_face.face_id in generated_faces
    }
    selected_surface_ids = tuple(
        surface_id
        for child_ids in replacements.values()
        for surface_id in child_ids
    )
    return next_mesh, replacements, selected_surface_ids


def _split_surface_region(
    geometry: object,
    path_line: LineString,
) -> tuple[Polygon, ...] | None:
    """Return stable hole-free pieces only when a line truly partitions a region."""

    original_polygons = _iter_surface_polygons(geometry)
    if not original_polygons:
        return None
    partitioned: list[Polygon] = []
    did_split = False
    for polygon in original_polygons:
        try:
            split_result = split_geometry(polygon, path_line)
        except ShapelyError as error:
            raise ValueError(
                "The open surface cut could not be resolved safely."
            ) from error
        raw_pieces = tuple(
            candidate
            for candidate in _iter_surface_polygons(split_result)
            if candidate.area > SURFACE_EDIT_EPSILON_METERS**2
        )
        if len(raw_pieces) > 1:
            did_split = True
        partitioned.extend(raw_pieces or (polygon,))
    if not did_split:
        return None

    original_area = sum(polygon.area for polygon in original_polygons)
    partitioned_area = sum(polygon.area for polygon in partitioned)
    area_tolerance = max(
        SURFACE_EDIT_PLANAR_TOLERANCE_METERS**2,
        original_area * 1e-9,
    )
    if abs(partitioned_area - original_area) > area_tolerance:
        raise ValueError("The open surface cut did not preserve the surface area.")

    hole_free: list[Polygon] = []
    for polygon in partitioned:
        if polygon.interiors:
            hole_free.extend(_decompose_surface_geometry(polygon))
        else:
            hole_free.append(orient(polygon, sign=1.0))
    return tuple(
        sorted(
            hole_free,
            key=lambda polygon: (
                round(polygon.centroid.x, 12),
                round(polygon.centroid.y, 12),
                round(polygon.area, 12),
            ),
        )
    )


def _node_open_path_edges_at_face_vertices(
    edges: Sequence[EditableSurfaceEdgeData],
    path_vertex_ids: Sequence[str],
    faces: Sequence[EditableSurfaceFaceData],
    world_vertices: Mapping[str, np.ndarray],
) -> tuple[EditableSurfaceEdgeData, ...]:
    """Node the committed path wherever rebuilt face topology added a vertex."""

    path_edge_keys = {
        frozenset((start_vertex_id, end_vertex_id))
        for start_vertex_id, end_vertex_id in zip(
            path_vertex_ids,
            path_vertex_ids[1:],
        )
    }
    face_vertex_ids = {
        vertex_id
        for face in faces
        for vertex_id in face.vertex_ids
    }
    next_edges: list[EditableSurfaceEdgeData] = []
    for edge in edges:
        edge_key = frozenset((edge.start_vertex_id, edge.end_vertex_id))
        if edge_key not in path_edge_keys:
            next_edges.append(edge)
            continue
        start = np.asarray(world_vertices[edge.start_vertex_id], dtype=float)
        end = np.asarray(world_vertices[edge.end_vertex_id], dtype=float)
        delta = end - start
        squared_length = float(np.dot(delta, delta))
        if squared_length <= SURFACE_EDIT_EPSILON_METERS**2:
            raise ValueError("An authored surface edge has no measurable length.")
        length = math.sqrt(squared_length)
        endpoint_tolerance = SURFACE_EDIT_EPSILON_METERS / length
        interior_vertices: list[tuple[float, str]] = []
        for vertex_id in face_vertex_ids:
            if vertex_id in {edge.start_vertex_id, edge.end_vertex_id}:
                continue
            point = np.asarray(world_vertices[vertex_id], dtype=float)
            parameter = float(np.dot(point - start, delta) / squared_length)
            if not endpoint_tolerance < parameter < 1.0 - endpoint_tolerance:
                continue
            projected = start + delta * parameter
            if float(np.linalg.norm(point - projected)) > (
                SURFACE_EDIT_PLANAR_TOLERANCE_METERS
            ):
                continue
            interior_vertices.append((parameter, vertex_id))

        ordered_vertex_ids = (
            edge.start_vertex_id,
            *(
                vertex_id
                for _parameter, vertex_id in sorted(
                    interior_vertices,
                    key=lambda item: (round(item[0], 12), item[1]),
                )
            ),
            edge.end_vertex_id,
        )
        next_edges.extend(
            EditableSurfaceEdgeData(
                start_vertex_id=start_vertex_id,
                end_vertex_id=end_vertex_id,
            )
            for start_vertex_id, end_vertex_id in zip(
                ordered_vertex_ids,
                ordered_vertex_ids[1:],
            )
        )
    return tuple(next_edges)


# ### Closed-loop subdivision ###
def _subdivide_faces_with_loop(
    context: _InsertionContext,
    editable_mesh: EditableSurfaceMeshData,
    loop_vertex_ids: tuple[str, ...],
) -> tuple[
    EditableSurfaceMeshData,
    dict[str, tuple[str, ...]],
    tuple[str, ...],
]:
    if len(loop_vertex_ids) < 3 or len(set(loop_vertex_ids)) != len(
        loop_vertex_ids
    ):
        raise ValueError("A surface face requires a simple loop of 3 or more vertices.")
    world_vertices = _build_world_vertex_lookup(
        context.level,
        context.source_surface,
        editable_mesh,
        context.frame,
    )
    loop_world = np.asarray(
        [world_vertices[vertex_id] for vertex_id in loop_vertex_ids],
        dtype=float,
    )
    loop_normal = _get_polygon_normal(loop_world)
    target_world = np.asarray(
        [world_vertices[vertex_id] for vertex_id in context.target_face.vertex_ids],
        dtype=float,
    )
    normal = _get_polygon_normal(target_world)
    if (
        abs(float(np.dot(loop_normal, normal)))
        < SURFACE_EDIT_NORMAL_DOT_TOLERANCE
        or np.max(np.abs((loop_world - loop_world[0]) @ normal))
        > SURFACE_EDIT_PLANAR_TOLERANCE_METERS
    ):
        raise ValueError("Surface face vertices must be coplanar.")
    basis_u, basis_v = _build_face_plane_basis(context.frame, normal)
    origin = loop_world[0]
    loop_coordinates = np.asarray(
        [
            _project_point_to_basis(point, origin, basis_u, basis_v)
            for point in loop_world
        ],
        dtype=float,
    )
    loop_polygon = orient(Polygon(loop_coordinates), sign=1.0)
    if (
        not loop_polygon.is_valid
        or loop_polygon.area <= SURFACE_EDIT_EPSILON_METERS**2
    ):
        raise ValueError("Surface edges must form one non-self-intersecting face.")

    face_entries = _build_coplanar_face_polygons(
        editable_mesh.faces,
        world_vertices,
        origin,
        normal,
        basis_u,
        basis_v,
    )
    region = unary_union([entry[1] for entry in face_entries])
    if not region.buffer(SURFACE_EDIT_EPSILON_METERS).covers(loop_polygon):
        raise ValueError("A new surface face must stay inside one surface region.")
    affected_entries = [
        entry
        for entry in face_entries
        if entry[1].intersection(loop_polygon).area
        > SURFACE_EDIT_EPSILON_METERS**2
    ]
    if not affected_entries:
        raise ValueError("The new surface face does not cover measurable area.")
    owner_face, _owner_polygon = max(
        affected_entries,
        key=lambda entry: (
            round(entry[1].intersection(loop_polygon).area, 12),
            entry[0].face_id,
        ),
    )

    vertices = list(editable_mesh.vertices)
    vertex_id_by_key = {
        _local_coordinate_key((
            vertex.u,
            vertex.v,
            vertex.normal_offset_meters,
        )): vertex.vertex_id
        for vertex in vertices
    }

    def get_vertex_id(coordinate: Sequence[float]) -> str:
        world = origin + basis_u * coordinate[0] + basis_v * coordinate[1]
        local = _world_to_local(context.level, context.frame, world)
        key = _local_coordinate_key(local)
        existing_id = vertex_id_by_key.get(key)
        if existing_id is not None:
            return existing_id
        vertex = EditableSurfaceVertexData(
            vertex_id=_new_uuid(),
            u=local[0],
            v=local[1],
            normal_offset_meters=local[2],
        )
        vertices.append(vertex)
        vertex_id_by_key[key] = vertex.vertex_id
        return vertex.vertex_id

    affected_ids = {face.face_id for face, _polygon in affected_entries}
    generated_faces: dict[str, list[EditableSurfaceFaceData]] = defaultdict(list)
    for face, face_polygon in affected_entries:
        remainder = face_polygon.difference(loop_polygon)
        for polygon in _decompose_surface_geometry(remainder):
            ring = tuple(polygon.exterior.coords)[:-1]
            vertex_ids = _compact_polygon_vertex_ids(
                tuple(get_vertex_id(coordinate) for coordinate in ring)
            )
            if len(set(vertex_ids)) < 3:
                continue
            generated_faces[face.face_id].append(
                EditableSurfaceFaceData(
                    face_id=_new_uuid(),
                    vertex_ids=vertex_ids,
                    surface_type=face.surface_type,
                )
            )

    oriented_loop_ids = loop_vertex_ids
    if not Polygon(loop_coordinates).exterior.is_ccw:
        oriented_loop_ids = tuple(reversed(oriented_loop_ids))
    enclosed_face = EditableSurfaceFaceData(
        face_id=_new_uuid(),
        vertex_ids=oriented_loop_ids,
        surface_type=owner_face.surface_type,
        is_directly_drawn=True,
    )
    generated_faces[owner_face.face_id].append(enclosed_face)

    next_faces: list[EditableSurfaceFaceData] = []
    for face in editable_mesh.faces:
        if face.face_id not in affected_ids:
            next_faces.append(face)
            continue
        next_faces.extend(generated_faces[face.face_id])
    next_mesh = replace(
        editable_mesh,
        vertices=tuple(vertices),
        faces=tuple(next_faces),
        replaces_source_surface=True,
    )
    created_surface_id = build_editable_surface_id(
        context.level.index,
        enclosed_face.face_id,
        enclosed_face.surface_type,
    )
    if not editable_mesh.replaces_source_surface:
        replacements = {
            context.source_surface.surface_id: tuple(
                build_editable_surface_id(
                    context.level.index,
                    face.face_id,
                    face.surface_type,
                )
                for face in next_mesh.faces
            )
        }
    else:
        replacements = {
            build_editable_surface_id(
                context.level.index,
                original_face.face_id,
                original_face.surface_type,
            ): tuple(
                build_editable_surface_id(
                    context.level.index,
                    child.face_id,
                    child.surface_type,
                )
                for child in generated_faces[original_face.face_id]
            )
            for original_face, _polygon in affected_entries
        }
    return next_mesh, replacements, (created_surface_id,)


def _build_coplanar_face_polygons(
    faces: Sequence[EditableSurfaceFaceData],
    world_vertices: Mapping[str, np.ndarray],
    origin: np.ndarray,
    normal: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
) -> tuple[tuple[EditableSurfaceFaceData, Polygon], ...]:
    entries: list[tuple[EditableSurfaceFaceData, Polygon]] = []
    for face in faces:
        points = np.asarray(
            [world_vertices[vertex_id] for vertex_id in face.vertex_ids],
            dtype=float,
        )
        face_normal = _get_polygon_normal(points)
        if abs(float(np.dot(face_normal, normal))) < (
            SURFACE_EDIT_NORMAL_DOT_TOLERANCE
        ):
            continue
        if np.max(np.abs((points - origin) @ normal)) > (
            SURFACE_EDIT_PLANAR_TOLERANCE_METERS
        ):
            continue
        polygon = orient(Polygon(
            [
                _project_point_to_basis(point, origin, basis_u, basis_v)
                for point in points
            ]
        ), sign=1.0)
        if not polygon.is_valid or polygon.area <= SURFACE_EDIT_EPSILON_METERS**2:
            raise ValueError("An editable surface face has invalid planar topology.")
        entries.append((face, polygon))
    if not entries:
        raise ValueError("No coplanar surface region is available for drawing.")
    return tuple(entries)


def _decompose_surface_geometry(geometry: object) -> tuple[Polygon, ...]:
    polygons = [
        polygon
        for polygon in _iter_surface_polygons(geometry)
        if polygon.area > SURFACE_EDIT_EPSILON_METERS**2
    ]
    pieces: list[Polygon] = []
    for polygon in polygons:
        triangles = [
            orient(candidate, sign=1.0)
            for candidate in _iter_surface_polygons(
                constrained_delaunay_triangles(polygon)
            )
            if candidate.area > SURFACE_EDIT_EPSILON_METERS**2
        ]
        covered = unary_union(triangles) if triangles else Polygon()
        if covered.symmetric_difference(polygon).area > (
            SURFACE_EDIT_PLANAR_TOLERANCE_METERS**2
        ):
            raise ValueError("The surface remainder could not be triangulated safely.")
        pieces.extend(triangles)
    return tuple(
        sorted(
            pieces,
            key=lambda polygon: (
                round(polygon.centroid.x, 12),
                round(polygon.centroid.y, 12),
                round(polygon.area, 12),
            ),
        )
    )


def _compact_polygon_vertex_ids(vertex_ids: Sequence[str]) -> tuple[str, ...]:
    compacted: list[str] = []
    for vertex_id in vertex_ids:
        if compacted and compacted[-1] == vertex_id:
            continue
        compacted.append(vertex_id)
    if len(compacted) > 1 and compacted[0] == compacted[-1]:
        compacted.pop()
    if len(compacted) != len(set(compacted)):
        raise ValueError("A generated surface remainder is not a simple polygon.")
    return tuple(compacted)


def _iter_surface_polygons(geometry: object) -> tuple[Polygon, ...]:
    if isinstance(geometry, Polygon):
        return () if geometry.is_empty else (geometry,)
    geometries = getattr(geometry, "geoms", ())
    return tuple(
        polygon
        for item in geometries
        for polygon in _iter_surface_polygons(item)
    )


def _local_coordinate_key(
    coordinate: Sequence[float],
) -> tuple[float, float, float]:
    return tuple(
        round(float(value), LOCAL_COORDINATE_KEY_DECIMALS)
        for value in coordinate
    )


def _get_overlay_vertex_ids(
    editable_mesh: EditableSurfaceMeshData,
) -> set[str]:
    if editable_mesh.replaces_source_surface:
        return {vertex.vertex_id for vertex in editable_mesh.vertices}
    face_vertex_ids = {
        vertex_id for face in editable_mesh.faces for vertex_id in face.vertex_ids
    }
    face_edge_counts: dict[frozenset[str], int] = defaultdict(int)
    for face in editable_mesh.faces:
        for first, second in _iter_face_edges(face):
            face_edge_counts[frozenset((first, second))] += 1
    boundary_ids = {
        vertex_id
        for edge_key, owner_count in face_edge_counts.items()
        if owner_count == 1
        for vertex_id in edge_key
    }
    authored_ids = {
        vertex_id
        for edge in editable_mesh.edges
        for vertex_id in (edge.start_vertex_id, edge.end_vertex_id)
    }
    unattached_ids = {
        vertex.vertex_id
        for vertex in editable_mesh.vertices
        if vertex.vertex_id not in face_vertex_ids
    }
    return boundary_ids | authored_ids | unattached_ids


def _get_open_chain_edges(
    edges: Sequence[EditableSurfaceEdgeData],
) -> tuple[EditableSurfaceEdgeData, ...]:
    """Return authored graph bridges, which disappear when a loop closes."""

    edge_sequence = tuple(edges)
    adjacency: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for edge_index, edge in enumerate(edge_sequence):
        adjacency[edge.start_vertex_id].append(
            (edge.end_vertex_id, edge_index)
        )
        adjacency[edge.end_vertex_id].append(
            (edge.start_vertex_id, edge_index)
        )
    for entries in adjacency.values():
        entries.sort()

    discovery: dict[str, int] = {}
    low_link: dict[str, int] = {}
    parent_vertex: dict[str, str] = {}
    parent_edge: dict[str, int] = {}
    bridge_indices: set[int] = set()
    next_discovery_index = 0
    for root_vertex_id in sorted(adjacency):
        if root_vertex_id in discovery:
            continue
        discovery[root_vertex_id] = next_discovery_index
        low_link[root_vertex_id] = next_discovery_index
        next_discovery_index += 1
        stack: list[tuple[str, int]] = [(root_vertex_id, 0)]
        while stack:
            vertex_id, neighbor_index = stack[-1]
            neighbors = adjacency[vertex_id]
            if neighbor_index >= len(neighbors):
                stack.pop()
                parent_id = parent_vertex.get(vertex_id)
                if parent_id is None:
                    continue
                low_link[parent_id] = min(
                    low_link[parent_id],
                    low_link[vertex_id],
                )
                if low_link[vertex_id] > discovery[parent_id]:
                    bridge_indices.add(parent_edge[vertex_id])
                continue

            neighbor_id, edge_index = neighbors[neighbor_index]
            stack[-1] = (vertex_id, neighbor_index + 1)
            if edge_index == parent_edge.get(vertex_id):
                continue
            if neighbor_id not in discovery:
                parent_vertex[neighbor_id] = vertex_id
                parent_edge[neighbor_id] = edge_index
                discovery[neighbor_id] = next_discovery_index
                low_link[neighbor_id] = next_discovery_index
                next_discovery_index += 1
                stack.append((neighbor_id, 0))
                continue
            low_link[vertex_id] = min(
                low_link[vertex_id],
                discovery[neighbor_id],
            )

    return tuple(
        edge
        for edge_index, edge in enumerate(edge_sequence)
        if edge_index in bridge_indices
    )


# ### Extrusion preparation ###
def _find_extrusion_context(
    levels: tuple[LevelData, ...],
    surface_ids: tuple[str, ...],
) -> tuple[
    LevelData,
    FixedSurface,
    EditableSurfaceMeshData,
    tuple[EditableSurfaceFaceData, ...],
    _SurfaceFrame,
]:
    parsed_ids = [parse_editable_surface_id(surface_id) for surface_id in surface_ids]
    if any(parsed is None for parsed in parsed_ids):
        raise ValueError("Only authored surface faces can be extruded.")
    normalized_parsed = [parsed for parsed in parsed_ids if parsed is not None]
    level_indices = {parsed[0] for parsed in normalized_parsed}
    if len(level_indices) != 1:
        raise ValueError("Extruded surface faces must belong to one level.")
    level = _find_level(levels, next(iter(level_indices)))
    face_ids = tuple(parsed[1] for parsed in normalized_parsed)
    matching_meshes = [
        mesh
        for mesh in level.editable_surfaces
        if any(face.face_id in face_ids for face in mesh.faces)
    ]
    unique_meshes = {mesh.source_surface_id: mesh for mesh in matching_meshes}
    if len(unique_meshes) != 1:
        raise ValueError("Extruded faces must belong to one editable surface.")
    editable_mesh = next(iter(unique_meshes.values()))
    face_by_id = {face.face_id: face for face in editable_mesh.faces}
    if any(face_id not in face_by_id for face_id in face_ids):
        raise ValueError("One selected editable face no longer exists.")
    selected_faces = tuple(face_by_id[face_id] for face_id in face_ids)
    for parsed, face in zip(normalized_parsed, selected_faces, strict=True):
        if parsed[2] != face.surface_type:
            raise ValueError("A selected editable surface ID has the wrong type.")
    source_surface = next(
        (
            surface
            for surface in build_base_fixed_surfaces(levels)
            if surface.surface_id == editable_mesh.source_surface_id
        ),
        None,
    )
    if source_surface is None:
        raise ValueError("The editable source surface no longer exists.")
    frame = _build_surface_frame(
        level,
        source_surface,
        editable_mesh.frame_kind,
        editable_mesh.frame_normal_sign,
    )
    return level, source_surface, editable_mesh, selected_faces, frame


def _validate_extrusion_selection(
    selected_faces: tuple[EditableSurfaceFaceData, ...],
    world_vertices: Mapping[str, np.ndarray],
) -> np.ndarray:
    normals: list[np.ndarray] = []
    reference_point: np.ndarray | None = None
    for face in selected_faces:
        points = np.asarray(
            [world_vertices[vertex_id] for vertex_id in face.vertex_ids],
            dtype=float,
        )
        normal = _get_polygon_normal(points)
        if reference_point is None:
            reference_point = points[0]
        if np.max(np.abs((points - points[0]) @ normal)) > (
            SURFACE_EDIT_PLANAR_TOLERANCE_METERS
        ):
            raise ValueError("Editable surface faces must be planar to extrude.")
        normals.append(normal)
    assert reference_point is not None
    reference_normal = normals[0]
    if any(
        float(np.dot(reference_normal, normal)) < SURFACE_EDIT_NORMAL_DOT_TOLERANCE
        for normal in normals[1:]
    ):
        raise ValueError("Extruded surface faces must have matching normals.")
    for face in selected_faces[1:]:
        first_vertex = world_vertices[face.vertex_ids[0]]
        if abs(float(np.dot(first_vertex - reference_point, reference_normal))) > (
            SURFACE_EDIT_PLANAR_TOLERANCE_METERS
        ):
            raise ValueError("Extruded surface faces must be coplanar.")
    _validate_faces_are_connected(selected_faces)
    averaged = np.sum(np.asarray(normals, dtype=float), axis=0)
    length = float(np.linalg.norm(averaged))
    if length <= SURFACE_EDIT_EPSILON_METERS:
        raise ValueError("Extruded surface normals cancel each other out.")
    return averaged / length


def _validate_faces_are_connected(
    selected_faces: tuple[EditableSurfaceFaceData, ...],
) -> None:
    if len(selected_faces) <= 1:
        return
    edge_owners: dict[frozenset[str], list[str]] = {}
    for face in selected_faces:
        for first, second in _iter_face_edges(face):
            edge_owners.setdefault(frozenset((first, second)), []).append(face.face_id)
    adjacency = {face.face_id: set() for face in selected_faces}
    for owners in edge_owners.values():
        for first in owners:
            adjacency[first].update(owner for owner in owners if owner != first)
    visited: set[str] = set()
    queue = deque((selected_faces[0].face_id,))
    while queue:
        face_id = queue.popleft()
        if face_id in visited:
            continue
        visited.add(face_id)
        queue.extend(sorted(adjacency[face_id] - visited))
    if len(visited) != len(selected_faces):
        raise ValueError("Extruded surface faces must form one connected region.")


def _build_selection_boundary_edges(
    selected_faces: tuple[EditableSurfaceFaceData, ...],
) -> tuple[tuple[str, str], ...]:
    edge_counts: dict[frozenset[str], int] = {}
    directed_edges: dict[frozenset[str], tuple[str, str]] = {}
    for face in selected_faces:
        for first, second in _iter_face_edges(face):
            key = frozenset((first, second))
            edge_counts[key] = edge_counts.get(key, 0) + 1
            directed_edges.setdefault(key, (first, second))
    if any(count > 2 for count in edge_counts.values()):
        raise ValueError("The selected surface boundary is non-manifold.")
    boundary = [directed_edges[key] for key, count in edge_counts.items() if count == 1]
    if not boundary:
        raise ValueError("The selected surface region has no boundary.")
    return tuple(sorted(boundary, key=lambda edge: tuple(sorted(edge))))


# ### Reversible extrusion helpers ###
def _apply_extrusion_to_editable_mesh(
    level: LevelData,
    source_surface: FixedSurface,
    editable_mesh: EditableSurfaceMeshData,
    selected_face_ids: tuple[str, ...],
    frame: _SurfaceFrame,
    delta_meters: float,
) -> tuple[EditableSurfaceMeshData, dict[str, tuple[str, ...]]]:
    """Apply one delta, peeling fully reversed layers before continuing."""

    selected_face_id_set = set(selected_face_ids)
    selected_faces = tuple(
        face
        for face in editable_mesh.faces
        if face.face_id in selected_face_id_set
    )
    if len(selected_faces) != len(selected_face_id_set):
        raise ValueError("One selected editable face no longer exists.")
    world_vertices = _build_world_vertex_lookup(
        level,
        source_surface,
        editable_mesh,
        frame,
    )
    extrusion_normal = _validate_extrusion_selection(
        selected_faces,
        world_vertices,
    )
    boundary_edges = _build_selection_boundary_edges(selected_faces)
    reversible_layer = _find_reversible_extrusion_layer(
        editable_mesh,
        selected_faces,
        boundary_edges,
        world_vertices,
        extrusion_normal,
    )
    if reversible_layer is None:
        return (
            _build_new_extrusion_mesh(
                level,
                editable_mesh,
                selected_faces,
                boundary_edges,
                frame,
                world_vertices,
                extrusion_normal,
                delta_meters,
            ),
            {},
        )
    return _update_existing_extrusion_layer(
        level,
        source_surface,
        editable_mesh,
        selected_faces,
        selected_face_ids,
        frame,
        world_vertices,
        extrusion_normal,
        delta_meters,
        reversible_layer,
    )


def _find_reversible_extrusion_layer(
    editable_mesh: EditableSurfaceMeshData,
    selected_faces: tuple[EditableSurfaceFaceData, ...],
    boundary_edges: tuple[tuple[str, str], ...],
    world_vertices: Mapping[str, np.ndarray],
    extrusion_normal: np.ndarray,
) -> _ReversibleExtrusionLayer | None:
    """Recognize the selected cap's immediately preceding extrusion ring."""

    selected_face_ids = {face.face_id for face in selected_faces}
    selected_vertex_ids = {
        vertex_id for face in selected_faces for vertex_id in face.vertex_ids
    }
    face_owners_by_edge: dict[
        frozenset[str],
        list[EditableSurfaceFaceData],
    ] = defaultdict(list)
    for face in editable_mesh.faces:
        for first_vertex_id, second_vertex_id in _iter_face_edges(face):
            face_owners_by_edge[
                frozenset((first_vertex_id, second_vertex_id))
            ].append(face)

    cap_to_base: dict[str, str] = {}
    side_faces_by_id: dict[str, EditableSurfaceFaceData] = {}
    swept_edges_by_component: dict[
        frozenset[str],
        list[tuple[str, str, str, str]],
    ] = defaultdict(list)
    for first_vertex_id, second_vertex_id in boundary_edges:
        owners = face_owners_by_edge.get(
            frozenset((first_vertex_id, second_vertex_id)),
            (),
        )
        selected_owners = [
            face for face in owners if face.face_id in selected_face_ids
        ]
        side_candidates = [
            face for face in owners if face.face_id not in selected_face_ids
        ]
        if len(selected_owners) != 1 or len(side_candidates) != 1:
            return None
        side_component = _collect_coplanar_extrusion_side_faces(
            side_candidates[0],
            editable_mesh.faces,
            face_owners_by_edge,
            selected_face_ids,
            world_vertices,
            extrusion_normal,
        )
        if not side_component:
            return None
        base_edge = (
            _match_extrusion_side_base_edge(
                side_candidates[0],
                first_vertex_id,
                second_vertex_id,
                selected_vertex_ids,
            )
            if len(side_component) == 1
            else None
        )
        if base_edge is None:
            base_edge = _match_subdivided_extrusion_side_base_edge(
                side_component,
                first_vertex_id,
                second_vertex_id,
                selected_vertex_ids,
                world_vertices,
                extrusion_normal,
            )
        if base_edge is None:
            return None
        for cap_vertex_id, base_vertex_id in zip(
            (first_vertex_id, second_vertex_id),
            base_edge,
            strict=True,
        ):
            previous_base_id = cap_to_base.setdefault(
                cap_vertex_id,
                base_vertex_id,
            )
            if previous_base_id != base_vertex_id:
                return None
        component_key = frozenset(face.face_id for face in side_component)
        swept_edges_by_component[component_key].append(
            (
                first_vertex_id,
                second_vertex_id,
                base_edge[0],
                base_edge[1],
            )
        )
        side_faces_by_id.update(
            (face.face_id, face) for face in side_component
        )

    base_vertex_ids = set(cap_to_base.values())
    if len(base_vertex_ids) != len(cap_to_base):
        return None
    offsets = tuple(
        world_vertices[cap_vertex_id] - world_vertices[base_vertex_id]
        for cap_vertex_id, base_vertex_id in cap_to_base.items()
    )
    if not offsets:
        return None
    reference_offset = offsets[0]
    if any(
        float(np.linalg.norm(offset - reference_offset))
        > SURFACE_EDIT_PLANAR_TOLERANCE_METERS
        for offset in offsets[1:]
    ):
        return None
    depth_meters = float(np.dot(reference_offset, extrusion_normal))
    if abs(depth_meters) <= SURFACE_EDIT_EPSILON_METERS:
        return None
    if (
        float(
            np.linalg.norm(
                reference_offset - extrusion_normal * depth_meters
            )
        )
        > SURFACE_EDIT_PLANAR_TOLERANCE_METERS
    ):
        return None

    if not _map_internal_extrusion_vertices(
        editable_mesh,
        selected_vertex_ids,
        world_vertices,
        reference_offset,
        cap_to_base,
    ):
        return None
    side_vertex_ids: set[str] = set()
    for component_ids, swept_edges in swept_edges_by_component.items():
        component_faces = tuple(
            face
            for face in editable_mesh.faces
            if face.face_id in component_ids
        )
        component_vertex_ids = _resolve_extrusion_side_component_vertex_ids(
            component_faces,
            tuple(swept_edges),
            world_vertices,
        )
        if component_vertex_ids is None:
            return None
        side_vertex_ids.update(component_vertex_ids)
    side_faces = tuple(
        face
        for face in editable_mesh.faces
        if face.face_id in side_faces_by_id
    )
    base_plane_origin = world_vertices[next(iter(cap_to_base.values()))]
    allowed_face_ids = selected_face_ids | {
        face.face_id for face in side_faces
    }
    if any(
        face.face_id not in allowed_face_ids
        and any(
            vertex_id in selected_vertex_ids
            or (
                vertex_id in side_vertex_ids
                and abs(
                    float(
                        np.dot(
                            world_vertices[vertex_id] - base_plane_origin,
                            extrusion_normal,
                        )
                    )
                )
                > SURFACE_EDIT_PLANAR_TOLERANCE_METERS
            )
            for vertex_id in face.vertex_ids
        )
        for face in editable_mesh.faces
    ):
        return None
    return _ReversibleExtrusionLayer(
        cap_to_base_vertex_ids=tuple(cap_to_base.items()),
        side_face_ids=tuple(face.face_id for face in side_faces),
        depth_meters=depth_meters,
        side_vertex_ids=tuple(
            vertex.vertex_id
            for vertex in editable_mesh.vertices
            if vertex.vertex_id in side_vertex_ids
        ),
    )


def _collect_coplanar_extrusion_side_faces(
    seed_face: EditableSurfaceFaceData,
    faces: Sequence[EditableSurfaceFaceData],
    face_owners_by_edge: Mapping[
        frozenset[str],
        Sequence[EditableSurfaceFaceData],
    ],
    selected_face_ids: set[str],
    world_vertices: Mapping[str, np.ndarray],
    extrusion_normal: np.ndarray,
) -> tuple[EditableSurfaceFaceData, ...]:
    """Collect one connected side patch, including later face subdivisions."""

    seed_points = np.asarray(
        [world_vertices[vertex_id] for vertex_id in seed_face.vertex_ids],
        dtype=float,
    )
    seed_normal = _get_polygon_normal(seed_points)
    if abs(float(np.dot(seed_normal, extrusion_normal))) > (
        1.0 - SURFACE_EDIT_NORMAL_DOT_TOLERANCE
    ):
        return ()
    seed_origin = seed_points[0]
    face_by_id = {face.face_id: face for face in faces}
    collected_ids: set[str] = set()
    queue = deque((seed_face.face_id,))
    while queue:
        face_id = queue.popleft()
        if face_id in collected_ids or face_id in selected_face_ids:
            continue
        face = face_by_id.get(face_id)
        if face is None:
            continue
        points = np.asarray(
            [world_vertices[vertex_id] for vertex_id in face.vertex_ids],
            dtype=float,
        )
        face_normal = _get_polygon_normal(points)
        if abs(float(np.dot(face_normal, seed_normal))) < (
            SURFACE_EDIT_NORMAL_DOT_TOLERANCE
        ):
            continue
        if np.max(np.abs((points - seed_origin) @ seed_normal)) > (
            SURFACE_EDIT_PLANAR_TOLERANCE_METERS
        ):
            continue
        collected_ids.add(face_id)
        for first_vertex_id, second_vertex_id in _iter_face_edges(face):
            queue.extend(
                owner.face_id
                for owner in face_owners_by_edge.get(
                    frozenset((first_vertex_id, second_vertex_id)),
                    (),
                )
                if owner.face_id not in collected_ids
            )
    return tuple(face for face in faces if face.face_id in collected_ids)


def _match_subdivided_extrusion_side_base_edge(
    side_faces: Sequence[EditableSurfaceFaceData],
    cap_first_vertex_id: str,
    cap_second_vertex_id: str,
    selected_vertex_ids: set[str],
    world_vertices: Mapping[str, np.ndarray],
    extrusion_normal: np.ndarray,
) -> tuple[str, str] | None:
    """Recover a swept side's base edge after its quad was subdivided."""

    component_vertex_ids = {
        vertex_id for face in side_faces for vertex_id in face.vertex_ids
    }

    def find_base_candidates(cap_vertex_id: str) -> list[tuple[float, str]]:
        cap_point = world_vertices[cap_vertex_id]
        candidates: list[tuple[float, str]] = []
        for vertex_id in component_vertex_ids - selected_vertex_ids:
            offset = cap_point - world_vertices[vertex_id]
            depth = float(np.dot(offset, extrusion_normal))
            transverse = offset - extrusion_normal * depth
            if (
                abs(depth) <= SURFACE_EDIT_EPSILON_METERS
                or float(np.linalg.norm(transverse))
                > SURFACE_EDIT_PLANAR_TOLERANCE_METERS
            ):
                continue
            candidates.append((depth, vertex_id))
        return sorted(
            candidates,
            key=lambda candidate: (
                -round(abs(candidate[0]), 12),
                candidate[1],
            ),
        )

    first_candidates = find_base_candidates(cap_first_vertex_id)
    second_candidates = find_base_candidates(cap_second_vertex_id)
    matching_pairs = [
        (first_depth, first_id, second_id)
        for first_depth, first_id in first_candidates
        for second_depth, second_id in second_candidates
        if first_id != second_id
        and abs(first_depth - second_depth)
        <= SURFACE_EDIT_PLANAR_TOLERANCE_METERS
    ]
    if not matching_pairs:
        return None
    _depth, first_id, second_id = min(
        matching_pairs,
        key=lambda candidate: (
            -round(abs(candidate[0]), 12),
            candidate[1],
            candidate[2],
        ),
    )
    return first_id, second_id


def _resolve_extrusion_side_component_vertex_ids(
    side_faces: Sequence[EditableSurfaceFaceData],
    swept_edges: Sequence[tuple[str, str, str, str]],
    world_vertices: Mapping[str, np.ndarray],
) -> set[str] | None:
    """Validate one swept side patch and collect its face and draft vertices."""

    if not side_faces or not swept_edges:
        return None
    first_face = side_faces[0]
    first_points = np.asarray(
        [world_vertices[vertex_id] for vertex_id in first_face.vertex_ids],
        dtype=float,
    )
    normal = _get_polygon_normal(first_points)
    origin = first_points[0]
    frame = _fallback_face_frame(first_points, normal)
    basis_u, basis_v = _build_face_plane_basis(frame, normal)
    try:
        face_entries = _build_coplanar_face_polygons(
            side_faces,
            world_vertices,
            origin,
            normal,
            basis_u,
            basis_v,
        )
    except ValueError:
        return None
    side_polygons = [polygon for _face, polygon in face_entries]
    swept_polygons: list[Polygon] = []
    for cap_first_id, cap_second_id, base_first_id, base_second_id in swept_edges:
        points = (
            world_vertices[cap_first_id],
            world_vertices[cap_second_id],
            world_vertices[base_second_id],
            world_vertices[base_first_id],
        )
        polygon = Polygon(
            [
                _project_point_to_basis(point, origin, basis_u, basis_v)
                for point in points
            ]
        )
        if not polygon.is_valid or (
            polygon.area <= SURFACE_EDIT_EPSILON_METERS**2
        ):
            return None
        swept_polygons.append(polygon)
    side_union = unary_union(side_polygons)
    swept_union = unary_union(swept_polygons)
    tolerance_area = SURFACE_EDIT_PLANAR_TOLERANCE_METERS**2
    if (
        side_union.symmetric_difference(swept_union).area > tolerance_area
        or abs(sum(polygon.area for polygon in side_polygons) - side_union.area)
        > tolerance_area
    ):
        return None
    padded_region = swept_union.buffer(SURFACE_EDIT_PLANAR_TOLERANCE_METERS)
    return {
        vertex_id
        for vertex_id, world_point in world_vertices.items()
        if abs(float(np.dot(world_point - origin, normal)))
        <= SURFACE_EDIT_PLANAR_TOLERANCE_METERS
        and padded_region.covers(
            Point(
                _project_point_to_basis(
                    world_point,
                    origin,
                    basis_u,
                    basis_v,
                )
            )
        )
    }


def _match_extrusion_side_base_edge(
    side_face: EditableSurfaceFaceData,
    cap_first_vertex_id: str,
    cap_second_vertex_id: str,
    selected_vertex_ids: set[str],
) -> tuple[str, str] | None:
    """Match one generated quad's opposite edge to a directed cap edge."""

    side_vertex_ids = side_face.vertex_ids
    if len(side_vertex_ids) != 4 or (
        set(side_vertex_ids) & selected_vertex_ids
        != {cap_first_vertex_id, cap_second_vertex_id}
    ):
        return None
    for index, vertex_id in enumerate(side_vertex_ids):
        if vertex_id != cap_second_vertex_id:
            continue
        if side_vertex_ids[(index + 1) % 4] != cap_first_vertex_id:
            continue
        base_first_vertex_id = side_vertex_ids[(index + 2) % 4]
        base_second_vertex_id = side_vertex_ids[(index + 3) % 4]
        if base_first_vertex_id == base_second_vertex_id:
            return None
        return base_first_vertex_id, base_second_vertex_id
    return None


def _map_internal_extrusion_vertices(
    editable_mesh: EditableSurfaceMeshData,
    selected_vertex_ids: set[str],
    world_vertices: Mapping[str, np.ndarray],
    extrusion_offset: np.ndarray,
    cap_to_base: dict[str, str],
) -> bool:
    """Resolve non-boundary cap duplicates by their translated base points."""

    vertex_order = {
        vertex.vertex_id: index
        for index, vertex in enumerate(editable_mesh.vertices)
    }
    used_base_vertex_ids = set(cap_to_base.values())
    for cap_vertex_id in sorted(
        selected_vertex_ids - set(cap_to_base),
        key=vertex_order.__getitem__,
    ):
        target_world = world_vertices[cap_vertex_id] - extrusion_offset
        candidates = [
            candidate.vertex_id
            for candidate in editable_mesh.vertices
            if candidate.vertex_id not in selected_vertex_ids
            and candidate.vertex_id not in used_base_vertex_ids
            and vertex_order[candidate.vertex_id] < vertex_order[cap_vertex_id]
            and float(
                np.linalg.norm(
                    world_vertices[candidate.vertex_id] - target_world
                )
            )
            <= SURFACE_EDIT_PLANAR_TOLERANCE_METERS
        ]
        if not candidates:
            return False
        base_vertex_id = max(candidates, key=vertex_order.__getitem__)
        cap_to_base[cap_vertex_id] = base_vertex_id
        used_base_vertex_ids.add(base_vertex_id)
    return True


def _build_new_extrusion_mesh(
    level: LevelData,
    editable_mesh: EditableSurfaceMeshData,
    selected_faces: tuple[EditableSurfaceFaceData, ...],
    boundary_edges: tuple[tuple[str, str], ...],
    frame: _SurfaceFrame,
    world_vertices: Mapping[str, np.ndarray],
    extrusion_normal: np.ndarray,
    delta_meters: float,
) -> EditableSurfaceMeshData:
    """Create one new cap layer and its boundary side ring."""

    selected_vertex_ids = {
        vertex_id for face in selected_faces for vertex_id in face.vertex_ids
    }
    selected_face_ids = {face.face_id for face in selected_faces}
    retained_marker_vertex_ids = {
        vertex_id
        for face in editable_mesh.faces
        if face.face_id not in selected_face_ids and face.is_directly_drawn
        for vertex_id in face.vertex_ids
    }
    duplicated_vertex_ids: dict[str, str] = {}
    added_vertices: list[EditableSurfaceVertexData] = []
    offset_world = extrusion_normal * delta_meters
    for vertex in editable_mesh.vertices:
        if vertex.vertex_id not in selected_vertex_ids:
            continue
        duplicate_id = _new_uuid()
        duplicated_vertex_ids[vertex.vertex_id] = duplicate_id
        shifted_world = world_vertices[vertex.vertex_id] + offset_world
        shifted_local = _world_to_local(level, frame, shifted_world)
        added_vertices.append(
            EditableSurfaceVertexData(
                vertex_id=duplicate_id,
                u=shifted_local[0],
                v=shifted_local[1],
                normal_offset_meters=shifted_local[2],
                is_user_placed=vertex.is_user_placed,
            )
        )

    next_faces = [
        (
            replace(
                face,
                vertex_ids=tuple(
                    duplicated_vertex_ids[vertex_id]
                    for vertex_id in face.vertex_ids
                ),
            )
            if face.face_id in selected_face_ids
            else face
        )
        for face in editable_mesh.faces
    ]
    for first_vertex_id, second_vertex_id in boundary_edges:
        side_vertex_ids = (
            first_vertex_id,
            second_vertex_id,
            duplicated_vertex_ids[second_vertex_id],
            duplicated_vertex_ids[first_vertex_id],
        )
        side_world = np.asarray(
            (
                world_vertices[first_vertex_id],
                world_vertices[second_vertex_id],
                world_vertices[second_vertex_id] + offset_world,
                world_vertices[first_vertex_id] + offset_world,
            ),
            dtype=float,
        )
        next_faces.append(
            EditableSurfaceFaceData(
                face_id=_new_uuid(),
                vertex_ids=side_vertex_ids,
                surface_type=_classify_surface_type(
                    _get_polygon_normal(side_world)
                ),
            )
        )
    return replace(
        editable_mesh,
        vertices=(
            *(
                replace(vertex, is_user_placed=False)
                if vertex.vertex_id in selected_vertex_ids
                and vertex.vertex_id not in retained_marker_vertex_ids
                and vertex.is_user_placed
                else vertex
                for vertex in editable_mesh.vertices
            ),
            *added_vertices,
        ),
        faces=tuple(next_faces),
        edges=tuple(
            EditableSurfaceEdgeData(
                start_vertex_id=duplicated_vertex_ids.get(
                    edge.start_vertex_id,
                    edge.start_vertex_id,
                ),
                end_vertex_id=duplicated_vertex_ids.get(
                    edge.end_vertex_id,
                    edge.end_vertex_id,
                ),
            )
            for edge in editable_mesh.edges
        ),
        replaces_source_surface=True,
    )


def _update_existing_extrusion_layer(
    level: LevelData,
    source_surface: FixedSurface,
    editable_mesh: EditableSurfaceMeshData,
    selected_faces: tuple[EditableSurfaceFaceData, ...],
    selected_face_ids: tuple[str, ...],
    frame: _SurfaceFrame,
    world_vertices: Mapping[str, np.ndarray],
    extrusion_normal: np.ndarray,
    delta_meters: float,
    layer: _ReversibleExtrusionLayer,
) -> tuple[EditableSurfaceMeshData, dict[str, tuple[str, ...]]]:
    """Resize, collapse, or cross the existing single side ring."""

    remaining_depth = layer.depth_meters + delta_meters
    if abs(remaining_depth) <= SURFACE_EDIT_PLANAR_TOLERANCE_METERS:
        return _collapse_extrusion_layer(
            level,
            editable_mesh,
            selected_faces,
            layer,
        )
    if layer.depth_meters * remaining_depth > 0.0:
        return _resize_extrusion_layer(
            level,
            editable_mesh,
            selected_faces,
            frame,
            world_vertices,
            extrusion_normal,
            delta_meters,
            layer,
        )
    collapsed_mesh, removed_replacements = _collapse_extrusion_layer(
        level,
        editable_mesh,
        selected_faces,
        layer,
    )
    crossed_mesh, crossed_replacements = _apply_extrusion_to_editable_mesh(
        level,
        source_surface,
        collapsed_mesh,
        selected_face_ids,
        frame,
        remaining_depth,
    )
    return crossed_mesh, {
        **removed_replacements,
        **crossed_replacements,
    }


def _collapse_extrusion_layer(
    level: LevelData,
    editable_mesh: EditableSurfaceMeshData,
    selected_faces: tuple[EditableSurfaceFaceData, ...],
    layer: _ReversibleExtrusionLayer,
) -> tuple[EditableSurfaceMeshData, dict[str, tuple[str, ...]]]:
    """Weld the cap to its base and retire the now-degenerate side faces."""

    cap_to_base = dict(layer.cap_to_base_vertex_ids)
    selected_face_ids = {face.face_id for face in selected_faces}
    side_face_ids = set(layer.side_face_ids)
    removed_side_vertex_ids = set(layer.side_vertex_ids) | {
        vertex_id
        for face in editable_mesh.faces
        if face.face_id in side_face_ids
        for vertex_id in face.vertex_ids
    }
    removed_side_ids = {
        build_editable_surface_id(level.index, face.face_id, face.surface_type)
        for face in editable_mesh.faces
        if face.face_id in side_face_ids
    }
    next_faces = tuple(
        replace(
            face,
            vertex_ids=tuple(
                cap_to_base.get(vertex_id, vertex_id)
                for vertex_id in face.vertex_ids
            ),
        )
        if face.face_id in selected_face_ids
        else face
        for face in editable_mesh.faces
        if face.face_id not in side_face_ids
    )
    remapped_edges = _remap_editable_surface_edges(
        editable_mesh.edges,
        cap_to_base,
    )
    retained_face_vertex_ids = {
        vertex_id for face in next_faces for vertex_id in face.vertex_ids
    }
    cap_vertices_by_id = {
        vertex.vertex_id: vertex
        for vertex in editable_mesh.vertices
        if vertex.vertex_id in cap_to_base
    }
    restored_marker_vertex_ids = {
        base_vertex_id
        for cap_vertex_id, base_vertex_id in cap_to_base.items()
        if cap_vertices_by_id[cap_vertex_id].is_user_placed
    }
    retired_vertex_ids = (
        removed_side_vertex_ids | set(cap_to_base)
    ) - retained_face_vertex_ids
    next_edges = tuple(
        edge
        for edge in remapped_edges
        if edge.start_vertex_id not in retired_vertex_ids
        and edge.end_vertex_id not in retired_vertex_ids
    )
    next_mesh = replace(
        editable_mesh,
        vertices=tuple(
            (
                replace(vertex, is_user_placed=True)
                if vertex.vertex_id in restored_marker_vertex_ids
                and not vertex.is_user_placed
                else vertex
            )
            for vertex in editable_mesh.vertices
            if vertex.vertex_id not in retired_vertex_ids
        ),
        faces=next_faces,
        edges=next_edges,
    )
    replacements = {
        surface_id: () for surface_id in sorted(removed_side_ids)
    }
    return next_mesh, replacements


def _resize_extrusion_layer(
    level: LevelData,
    editable_mesh: EditableSurfaceMeshData,
    selected_faces: tuple[EditableSurfaceFaceData, ...],
    frame: _SurfaceFrame,
    world_vertices: Mapping[str, np.ndarray],
    extrusion_normal: np.ndarray,
    delta_meters: float,
    layer: _ReversibleExtrusionLayer,
) -> tuple[EditableSurfaceMeshData, dict[str, tuple[str, ...]]]:
    """Scale the existing side patch while retaining one extrusion ring."""

    selected_vertex_ids = {
        vertex_id for face in selected_faces for vertex_id in face.vertex_ids
    }
    cap_to_base = dict(layer.cap_to_base_vertex_ids)
    base_vertex_ids = set(cap_to_base.values())
    side_face_ids = set(layer.side_face_ids)
    side_vertex_ids = set(layer.side_vertex_ids) | {
        vertex_id
        for face in editable_mesh.faces
        if face.face_id in side_face_ids
        for vertex_id in face.vertex_ids
    }
    base_origin = world_vertices[layer.cap_to_base_vertex_ids[0][1]]
    next_depth = layer.depth_meters + delta_meters

    def move_vertex(vertex: EditableSurfaceVertexData) -> EditableSurfaceVertexData:
        vertex_id = vertex.vertex_id
        if vertex_id in selected_vertex_ids:
            offset = extrusion_normal * delta_meters
        elif vertex_id in side_vertex_ids and vertex_id not in base_vertex_ids:
            current_depth = float(
                np.dot(
                    world_vertices[vertex_id] - base_origin,
                    extrusion_normal,
                )
            )
            scaled_depth = current_depth * next_depth / layer.depth_meters
            offset = extrusion_normal * (scaled_depth - current_depth)
        else:
            return vertex
        return _move_editable_surface_vertex(
            level,
            frame,
            vertex,
            world_vertices[vertex_id] + offset,
        )

    next_vertices = tuple(
        move_vertex(vertex)
        for vertex in editable_mesh.vertices
    )
    return replace(editable_mesh, vertices=next_vertices), {}


def _move_editable_surface_vertex(
    level: LevelData,
    frame: _SurfaceFrame,
    vertex: EditableSurfaceVertexData,
    world_point: np.ndarray,
) -> EditableSurfaceVertexData:
    """Retain a vertex ID while replacing its local-frame coordinates."""

    local_point = _world_to_local(level, frame, world_point)
    return replace(
        vertex,
        u=local_point[0],
        v=local_point[1],
        normal_offset_meters=local_point[2],
    )


def _remap_editable_surface_edges(
    edges: tuple[EditableSurfaceEdgeData, ...],
    vertex_replacements: Mapping[str, str],
) -> tuple[EditableSurfaceEdgeData, ...]:
    """Remap authored edges while preserving order and uniqueness."""

    next_edges: list[EditableSurfaceEdgeData] = []
    seen_edge_keys: set[frozenset[str]] = set()
    for edge in edges:
        start_vertex_id = vertex_replacements.get(
            edge.start_vertex_id,
            edge.start_vertex_id,
        )
        end_vertex_id = vertex_replacements.get(
            edge.end_vertex_id,
            edge.end_vertex_id,
        )
        if start_vertex_id == end_vertex_id:
            continue
        edge_key = frozenset((start_vertex_id, end_vertex_id))
        if edge_key in seen_edge_keys:
            continue
        seen_edge_keys.add(edge_key)
        next_edges.append(
            EditableSurfaceEdgeData(
                start_vertex_id=start_vertex_id,
                end_vertex_id=end_vertex_id,
            )
        )
    return tuple(next_edges)


# ### Vertex snapping ###
def _get_surface_insertion_polygon(
    surface: FixedSurface,
    world_point: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(surface.mesh.vertices, dtype=float)
    faces = np.asarray(surface.mesh.faces, dtype=np.int64)
    if surface.source_surface_id is not None:
        normal = _get_polygon_normal(vertices)
        frame = _fallback_face_frame(vertices, normal)
        if _point_is_strictly_inside_polygon(
            world_point,
            vertices,
            normal,
            frame,
        ):
            return vertices, normal
    candidates: list[tuple[float, int, np.ndarray, np.ndarray]] = []
    for face_index, face in enumerate(faces):
        triangle = vertices[face]
        normal = _get_polygon_normal(triangle)
        plane_distance = abs(float(np.dot(world_point - triangle[0], normal)))
        if plane_distance > SURFACE_EDIT_PLANAR_TOLERANCE_METERS:
            continue
        frame = _fallback_face_frame(triangle, normal)
        if _point_is_strictly_inside_polygon(
            world_point,
            triangle,
            normal,
            frame,
        ):
            candidates.append((plane_distance, face_index, triangle, normal))
    if not candidates:
        raise ValueError("A new surface vertex must be strictly inside one face.")
    _distance, _index, points, normal = min(
        candidates,
        key=lambda candidate: candidate[:2],
    )
    return points, normal


def _validate_resolved_insertion_point(
    world_point: Sequence[float],
    target_face: EditableSurfaceFaceData,
    world_vertices: Mapping[str, np.ndarray],
    frame: _SurfaceFrame,
) -> None:
    face_points = np.asarray(
        [world_vertices[vertex_id] for vertex_id in target_face.vertex_ids],
        dtype=float,
    )
    normal = _get_polygon_normal(face_points)
    if not _point_is_strictly_inside_polygon(
        np.asarray(world_point, dtype=float),
        face_points,
        normal,
        frame,
    ):
        raise ValueError("A new surface vertex must be strictly inside one face.")


def _snap_point_in_polygon(
    world_point: Sequence[float],
    face_points: np.ndarray,
    nearby_world_points: np.ndarray,
    frame: _SurfaceFrame,
) -> np.ndarray:
    normal = _get_polygon_normal(face_points)
    requested = np.asarray(world_point, dtype=float)
    projected = requested - normal * float(np.dot(requested - face_points[0], normal))
    if not _point_is_strictly_inside_polygon(
        projected,
        face_points,
        normal,
        frame,
    ):
        raise ValueError("A new surface vertex must be strictly inside one face.")

    basis_u, basis_v = _build_face_plane_basis(frame, normal)
    polygon_origin = face_points[0]
    raw_2d = _project_point_to_basis(projected, polygon_origin, basis_u, basis_v)
    polygon_2d = np.asarray(
        [
            _project_point_to_basis(point, polygon_origin, basis_u, basis_v)
            for point in face_points
        ],
        dtype=float,
    )
    polygon = Polygon(polygon_2d)
    candidates: list[tuple[tuple[object, ...], np.ndarray]] = []
    coplanar_vertices = sorted(
        (
            tuple(round(float(component), 12) for component in point),
            point,
        )
        for point in nearby_world_points
        if abs(float(np.dot(point - face_points[0], normal)))
        <= SURFACE_EDIT_PLANAR_TOLERANCE_METERS
        and float(np.linalg.norm(point - projected))
        <= SURFACE_VERTEX_SNAP_NEARBY_RADIUS_METERS
    )
    for vertex_key, vertex_world in coplanar_vertices:
        vertex_2d = _project_point_to_basis(
            vertex_world,
            polygon_origin,
            basis_u,
            basis_v,
        )
        raw_delta = raw_2d - vertex_2d
        for angle_index in range(8):
            angle = math.radians(angle_index * SURFACE_VERTEX_SNAP_ANGLE_DEGREES)
            ray = np.asarray((math.cos(angle), math.sin(angle)), dtype=float)
            ray_distance = float(np.dot(raw_delta, ray))
            candidate_2d = vertex_2d + ray * ray_distance
            displacement = float(np.linalg.norm(candidate_2d - raw_2d))
            if displacement > SURFACE_VERTEX_SNAP_MAX_DISPLACEMENT_METERS:
                continue
            if (
                float(np.linalg.norm(candidate_2d - vertex_2d))
                > SURFACE_VERTEX_SNAP_NEARBY_RADIUS_METERS
            ):
                continue
            candidate_point = Point(float(candidate_2d[0]), float(candidate_2d[1]))
            if not polygon.contains(candidate_point):
                continue
            if (
                polygon.boundary.distance(candidate_point)
                <= SURFACE_EDIT_EPSILON_METERS
            ):
                continue
            if any(
                np.linalg.norm(candidate_2d - existing) <= SURFACE_EDIT_EPSILON_METERS
                for existing in polygon_2d
            ):
                continue
            candidate_world = (
                polygon_origin + basis_u * candidate_2d[0] + basis_v * candidate_2d[1]
            )
            candidates.append(
                (
                    (
                        round(displacement, 12),
                        round(abs(ray_distance), 12),
                        vertex_key,
                        angle_index,
                    ),
                    candidate_world,
                )
            )
    if not candidates:
        return projected
    return min(candidates, key=lambda candidate: candidate[0])[1]


# ### Surface frame conversion ###
def _get_source_frame_normal_sign(
    level: LevelData,
    source_surface: FixedSurface,
    frame_kind: str,
) -> int:
    provisional = _build_surface_frame(level, source_surface, frame_kind, 1)
    normals = np.asarray(source_surface.mesh.face_normals, dtype=float)
    areas = np.asarray(source_surface.mesh.area_faces, dtype=float)
    weighted = np.sum(normals * areas[:, np.newaxis], axis=0)
    if float(np.linalg.norm(weighted)) <= SURFACE_EDIT_EPSILON_METERS:
        weighted = normals[0]
    return 1 if float(np.dot(weighted, provisional.normal_world)) >= 0.0 else -1


def _build_surface_frame(
    level: LevelData,
    source_surface: FixedSurface,
    frame_kind: str,
    normal_sign: int,
) -> _SurfaceFrame:
    if frame_kind == EDITABLE_SURFACE_FRAME_WALL_RATIO:
        if (
            source_surface.wall_key is None
            or source_surface.wall_height_meters is None
            or source_surface.wall_start_world is None
        ):
            raise ValueError("An editable wall has no stable source frame.")
        match = WALL_KEY_PATTERN.fullmatch(source_surface.wall_key)
        if match is None:
            raise ValueError("An editable wall has an invalid source key.")
        first_vertex = level.vertex_data.get_vertex(int(match.group("first")))
        second_vertex = level.vertex_data.get_vertex(int(match.group("second")))
        if first_vertex is None or second_vertex is None:
            raise ValueError("An editable wall source vertex no longer exists.")
        first_xy = level_image_to_world_xy(level, first_vertex.x, first_vertex.y)
        second_xy = level_image_to_world_xy(level, second_vertex.x, second_vertex.y)
        base_z = float(source_surface.wall_start_world[2])
        origin = np.asarray((*first_xy, base_z), dtype=float)
        u_vector = np.asarray(
            (second_xy[0] - first_xy[0], second_xy[1] - first_xy[1], 0.0),
            dtype=float,
        )
        height = float(source_surface.wall_height_meters)
        if float(np.linalg.norm(u_vector)) <= SURFACE_EDIT_EPSILON_METERS or (
            not math.isfinite(height) or height <= SURFACE_EDIT_EPSILON_METERS
        ):
            raise ValueError("An editable wall source frame is degenerate.")
        v_vector = np.asarray((0.0, 0.0, height), dtype=float)
        raw_normal = np.cross(u_vector, v_vector)
        raw_normal /= np.linalg.norm(raw_normal)
        return _SurfaceFrame(
            kind=frame_kind,
            normal_sign=normal_sign,
            origin_world=origin,
            u_vector_world=u_vector,
            v_vector_world=v_vector,
            normal_world=raw_normal * normal_sign,
        )

    if frame_kind != EDITABLE_SURFACE_FRAME_LEVEL_IMAGE:
        raise ValueError(f"Unknown editable surface frame: {frame_kind!r}.")
    vertices = np.asarray(source_surface.mesh.vertices, dtype=float)
    if vertices.shape[0] == 0:
        raise ValueError("An editable horizontal source has no vertices.")
    base_z = float(np.median(vertices[:, 2]))
    origin_xy = level_image_to_world_xy(level, 0.0, 0.0)
    u_xy = level_image_to_world_xy(level, 1.0, 0.0)
    v_xy = level_image_to_world_xy(level, 0.0, 1.0)
    return _SurfaceFrame(
        kind=frame_kind,
        normal_sign=normal_sign,
        origin_world=np.asarray((*origin_xy, base_z), dtype=float),
        u_vector_world=np.asarray(
            (u_xy[0] - origin_xy[0], u_xy[1] - origin_xy[1], 0.0),
            dtype=float,
        ),
        v_vector_world=np.asarray(
            (v_xy[0] - origin_xy[0], v_xy[1] - origin_xy[1], 0.0),
            dtype=float,
        ),
        normal_world=np.asarray((0.0, 0.0, float(normal_sign)), dtype=float),
    )


def _world_to_local(
    level: LevelData,
    frame: _SurfaceFrame,
    world_point: Sequence[float],
) -> tuple[float, float, float]:
    point = np.asarray(world_point, dtype=float)
    if frame.kind == EDITABLE_SURFACE_FRAME_WALL_RATIO:
        delta = point - frame.origin_world
        u_length_squared = float(np.dot(frame.u_vector_world, frame.u_vector_world))
        v_length_squared = float(np.dot(frame.v_vector_world, frame.v_vector_world))
        return (
            float(np.dot(delta, frame.u_vector_world) / u_length_squared),
            float(np.dot(delta, frame.v_vector_world) / v_length_squared),
            float(np.dot(delta, frame.normal_world)),
        )
    image_x, image_y = level_world_to_image_xy(level, point[0], point[1])
    return (
        float(image_x),
        float(image_y),
        float((point[2] - frame.origin_world[2]) * frame.normal_sign),
    )


def _local_to_world(
    level: LevelData,
    frame: _SurfaceFrame,
    vertex: EditableSurfaceVertexData,
) -> np.ndarray:
    if frame.kind == EDITABLE_SURFACE_FRAME_WALL_RATIO:
        return (
            frame.origin_world
            + frame.u_vector_world * vertex.u
            + frame.v_vector_world * vertex.v
            + frame.normal_world * vertex.normal_offset_meters
        )
    world_x, world_y = level_image_to_world_xy(level, vertex.u, vertex.v)
    return np.asarray(
        (
            world_x,
            world_y,
            frame.origin_world[2] + frame.normal_sign * vertex.normal_offset_meters,
        ),
        dtype=float,
    )


# ### Geometry construction ###
def _build_world_vertex_lookup(
    level: LevelData,
    source_surface: FixedSurface,
    editable_mesh: EditableSurfaceMeshData,
    frame: _SurfaceFrame,
) -> dict[str, np.ndarray]:
    del source_surface
    return {
        vertex.vertex_id: _local_to_world(level, frame, vertex)
        for vertex in editable_mesh.vertices
    }


def _build_face_mesh(
    face: EditableSurfaceFaceData,
    world_vertices: Mapping[str, np.ndarray],
) -> trimesh.Trimesh:
    vertices = np.asarray(
        [world_vertices[vertex_id] for vertex_id in face.vertex_ids],
        dtype=float,
    )
    normal = _get_polygon_normal(vertices)
    frame = _fallback_face_frame(vertices, normal)
    basis_u, basis_v = _build_face_plane_basis(frame, normal)
    origin = vertices[0]
    coordinates = np.asarray(
        [
            _project_point_to_basis(vertex, origin, basis_u, basis_v)
            for vertex in vertices
        ],
        dtype=float,
    )
    polygon = Polygon(coordinates)
    if not polygon.is_valid or polygon.area <= SURFACE_EDIT_EPSILON_METERS**2:
        raise ValueError("An editable surface face has invalid polygon topology.")
    index_by_key = {
        tuple(round(float(value), 10) for value in coordinate): index
        for index, coordinate in enumerate(coordinates)
    }
    face_indices: list[tuple[int, int, int]] = []
    for triangle in _iter_surface_polygons(
        constrained_delaunay_triangles(polygon)
    ):
        triangle_indices = tuple(
            index_by_key[tuple(round(float(value), 10) for value in coordinate)]
            for coordinate in tuple(triangle.exterior.coords)[:-1]
        )
        triangle_world = vertices[np.asarray(triangle_indices, dtype=np.int64)]
        triangle_normal = np.cross(
            triangle_world[1] - triangle_world[0],
            triangle_world[2] - triangle_world[0],
        )
        if float(np.dot(triangle_normal, normal)) < 0.0:
            triangle_indices = (
                triangle_indices[0],
                triangle_indices[2],
                triangle_indices[1],
            )
        face_indices.append(triangle_indices)
    if not face_indices:
        raise ValueError("An editable surface face could not be triangulated.")
    faces = np.asarray(face_indices, dtype=np.int64)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _get_polygon_normal(points: np.ndarray) -> np.ndarray:
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 3:
        raise ValueError("Editable faces require at least three XYZ vertices.")
    origin = points[0]
    normal = np.zeros(3, dtype=float)
    for index in range(1, len(points) - 1):
        normal += np.cross(points[index] - origin, points[index + 1] - origin)
    length = float(np.linalg.norm(normal))
    if not math.isfinite(length) or length <= SURFACE_EDIT_EPSILON_METERS:
        raise ValueError("An editable surface face is degenerate.")
    normal /= length
    if np.max(np.abs((points - origin) @ normal)) > (
        SURFACE_EDIT_PLANAR_TOLERANCE_METERS
    ):
        raise ValueError("An editable surface face is not planar.")
    return normal


def _find_face_containing_world_point(
    faces: Sequence[EditableSurfaceFaceData],
    world_vertices: Mapping[str, np.ndarray],
    world_point: np.ndarray,
) -> EditableSurfaceFaceData:
    candidates: list[tuple[float, str, EditableSurfaceFaceData]] = []
    for face in faces:
        points = np.asarray(
            [world_vertices[vertex_id] for vertex_id in face.vertex_ids],
            dtype=float,
        )
        normal = _get_polygon_normal(points)
        plane_distance = abs(float(np.dot(world_point - points[0], normal)))
        if plane_distance > SURFACE_EDIT_PLANAR_TOLERANCE_METERS:
            continue
        frame = _fallback_face_frame(points, normal)
        if not _point_is_strictly_inside_polygon(world_point, points, normal, frame):
            continue
        candidates.append((plane_distance, face.face_id, face))
    if not candidates:
        raise ValueError("A new surface vertex must be strictly inside one face.")
    return min(candidates, key=lambda candidate: candidate[:2])[2]


def _find_face_containing_or_touching_world_point(
    faces: Sequence[EditableSurfaceFaceData],
    world_vertices: Mapping[str, np.ndarray],
    world_point: np.ndarray,
) -> EditableSurfaceFaceData:
    candidates: list[tuple[float, str, EditableSurfaceFaceData]] = []
    for face in faces:
        points = np.asarray(
            [world_vertices[vertex_id] for vertex_id in face.vertex_ids],
            dtype=float,
        )
        normal = _get_polygon_normal(points)
        plane_distance = abs(float(np.dot(world_point - points[0], normal)))
        if plane_distance > SURFACE_EDIT_PLANAR_TOLERANCE_METERS:
            continue
        frame = _fallback_face_frame(points, normal)
        if not _point_is_inside_or_on_polygon(
            world_point,
            points,
            normal,
            frame,
        ):
            continue
        candidates.append((plane_distance, face.face_id, face))
    if not candidates:
        raise ValueError("A surface vertex must stay on the selected surface.")
    return min(candidates, key=lambda candidate: candidate[:2])[2]


def _point_is_strictly_inside_polygon(
    point: np.ndarray,
    face_points: np.ndarray,
    normal: np.ndarray,
    frame: _SurfaceFrame,
) -> bool:
    projected = point - normal * float(np.dot(point - face_points[0], normal))
    basis_u, basis_v = _build_face_plane_basis(frame, normal)
    origin = face_points[0]
    polygon = Polygon(
        [
            _project_point_to_basis(vertex, origin, basis_u, basis_v)
            for vertex in face_points
        ]
    )
    projected_2d = _project_point_to_basis(projected, origin, basis_u, basis_v)
    candidate = Point(float(projected_2d[0]), float(projected_2d[1]))
    return bool(
        polygon.is_valid
        and polygon.area > SURFACE_EDIT_EPSILON_METERS**2
        and polygon.contains(candidate)
        and polygon.boundary.distance(candidate) > SURFACE_EDIT_EPSILON_METERS
    )


def _point_is_inside_or_on_polygon(
    point: np.ndarray,
    face_points: np.ndarray,
    normal: np.ndarray,
    frame: _SurfaceFrame,
) -> bool:
    projected = point - normal * float(np.dot(point - face_points[0], normal))
    basis_u, basis_v = _build_face_plane_basis(frame, normal)
    origin = face_points[0]
    polygon = Polygon(
        [
            _project_point_to_basis(vertex, origin, basis_u, basis_v)
            for vertex in face_points
        ]
    )
    projected_2d = _project_point_to_basis(projected, origin, basis_u, basis_v)
    candidate = Point(float(projected_2d[0]), float(projected_2d[1]))
    return bool(
        polygon.is_valid
        and polygon.area > SURFACE_EDIT_EPSILON_METERS**2
        and polygon.buffer(SURFACE_EDIT_EPSILON_METERS).covers(candidate)
    )


def _build_face_plane_basis(
    frame: _SurfaceFrame,
    normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    for candidate in (frame.u_vector_world, frame.v_vector_world, np.eye(3)):
        candidate_vectors = candidate if candidate.ndim == 2 else (candidate,)
        for candidate_vector in candidate_vectors:
            projected = candidate_vector - normal * float(
                np.dot(candidate_vector, normal)
            )
            length = float(np.linalg.norm(projected))
            if length <= SURFACE_EDIT_EPSILON_METERS:
                continue
            basis_u = projected / length
            basis_v = np.cross(normal, basis_u)
            basis_v /= np.linalg.norm(basis_v)
            return basis_u, basis_v
    raise ValueError("An editable surface face has no local plane basis.")


def _fallback_face_frame(points: np.ndarray, normal: np.ndarray) -> _SurfaceFrame:
    first_edge = points[1] - points[0]
    return _SurfaceFrame(
        kind=EDITABLE_SURFACE_FRAME_LEVEL_IMAGE,
        normal_sign=1,
        origin_world=points[0],
        u_vector_world=first_edge,
        v_vector_world=np.cross(normal, first_edge),
        normal_world=normal,
    )


def _project_point_to_basis(
    point: np.ndarray,
    origin: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
) -> np.ndarray:
    delta = point - origin
    return np.asarray(
        (float(np.dot(delta, basis_u)), float(np.dot(delta, basis_v))),
        dtype=float,
    )


def _classify_surface_type(normal: np.ndarray) -> str:
    if abs(float(normal[2])) < SURFACE_SIDE_HORIZONTAL_NORMAL_THRESHOLD:
        return SURFACE_TYPE_WALL
    return SURFACE_TYPE_FLOOR if normal[2] > 0.0 else SURFACE_TYPE_CEILING


def _validate_editable_mesh_geometry(
    level: LevelData,
    source_surface: FixedSurface,
    editable_mesh: EditableSurfaceMeshData,
) -> None:
    frame = _build_surface_frame(
        level,
        source_surface,
        editable_mesh.frame_kind,
        editable_mesh.frame_normal_sign,
    )
    world_vertices = _build_world_vertex_lookup(
        level,
        source_surface,
        editable_mesh,
        frame,
    )
    for face in editable_mesh.faces:
        mesh = _build_face_mesh(face, world_vertices)
        if not math.isfinite(float(mesh.area)) or (
            float(mesh.area) <= SURFACE_EDIT_EPSILON_METERS
        ):
            raise ValueError("An editable surface face has no measurable area.")


# ### State mutation helpers ###
def _replace_level_editable_mesh(
    level: LevelData,
    editable_mesh: EditableSurfaceMeshData,
) -> None:
    next_meshes = [
        candidate
        for candidate in level.editable_surfaces
        if candidate.source_surface_id != editable_mesh.source_surface_id
    ]
    next_meshes.append(editable_mesh)
    level.editable_surfaces = sorted(
        next_meshes,
        key=lambda candidate: candidate.source_surface_id,
    )


def _find_editable_face(
    level: LevelData,
    face_id: str,
) -> tuple[EditableSurfaceMeshData, EditableSurfaceFaceData]:
    matches = [
        (mesh, face)
        for mesh in level.editable_surfaces
        for face in mesh.faces
        if face.face_id == face_id
    ]
    if len(matches) != 1:
        raise ValueError("The selected editable surface face no longer exists.")
    return matches[0]


# ### Collection and validation helpers ###
def _iter_face_edges(
    face: EditableSurfaceFaceData,
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (face.vertex_ids[index], face.vertex_ids[(index + 1) % len(face.vertex_ids)])
        for index in range(len(face.vertex_ids))
    )


def _normalize_levels(levels: Sequence[LevelData]) -> tuple[LevelData, ...]:
    try:
        normalized = tuple(levels)
    except TypeError as error:
        raise TypeError("Surface topology edits require a level sequence.") from error
    if not all(isinstance(level, LevelData) for level in normalized):
        raise TypeError("Surface topology edit levels must contain LevelData values.")
    if len({level.index for level in normalized}) != len(normalized):
        raise ValueError("Surface topology edit level indices must be unique.")
    return normalized


def _find_level(levels: Sequence[LevelData], level_index: int) -> LevelData:
    matching = [level for level in levels if level.index == level_index]
    if len(matching) != 1:
        raise ValueError("The editable surface level no longer exists.")
    return matching[0]


def _normalize_surface_id(value: object) -> str:
    normalized = str(value).strip().lower()
    if not normalized or len(normalized) > 512:
        raise ValueError("Surface ID is empty or too long.")
    return normalized


def _normalize_world_point(value: object) -> tuple[float, float, float]:
    try:
        point = np.asarray(tuple(value), dtype=float)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("A surface vertex requires finite XYZ coordinates.") from error
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("A surface vertex requires finite XYZ coordinates.")
    return tuple(float(component) for component in point)


def _normalize_level_index(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError("Editable surface level index must be a non-negative integer.")
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "Editable surface level index must be a non-negative integer."
        ) from error
    if normalized < 0 or normalized != value:
        raise ValueError("Editable surface level index must be a non-negative integer.")
    return normalized


def _normalize_uuid(value: object, field_name: str) -> str:
    normalized = str(value).strip().lower()
    if re.fullmatch(r"[0-9a-f]{32}", normalized) is None:
        raise ValueError(f"Editable surface {field_name} must be a UUID.")
    return normalized


def _normalize_surface_type(value: object) -> str:
    normalized = str(value).strip().lower()
    if normalized not in SURFACE_TYPES:
        raise ValueError(f"Unknown editable surface type: {value!r}.")
    return normalized


def _normalize_finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"Surface {field_name} must be a number.")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Surface {field_name} must be a number.") from error
    if not math.isfinite(normalized):
        raise ValueError(f"Surface {field_name} must be finite.")
    return normalized


def _new_uuid() -> str:
    return uuid.uuid4().hex
