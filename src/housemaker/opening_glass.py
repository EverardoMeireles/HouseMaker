# ### Imports ###
from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import shapely
import trimesh
from shapely import Polygon
from shapely.errors import ShapelyError
from trimesh.visual.color import ColorVisuals
from trimesh.visual.material import MultiMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.glass_material import (
    build_housemaker_glass_material,
    is_housemaker_glass_material,
)
from housemaker.glb import (
    GLTF_Y_UP_TO_Z_UP_TRANSFORM,
    HALF_MESH_EXTRAS_KEY,
    Z_UP_TO_GLTF_Y_UP_TRANSFORM,
    GeneratedModel,
    _serialize_scene_glb_with_half_mesh_extras,
)

# ### Public constants ###
OPENING_GLASS_METADATA_KEY = "housemaker_opening_glass_area"
OPENING_GLASS_EXTRAS_KEY = "housemakerGlassArea"
OPENING_GLASS_NODE_PREFIX = "[GLASS]"
MIN_NORMALIZED_GLASS_SPAN = 1e-6


# ### Internal constants ###
_GLB_HEADER_BYTE_COUNT = 12
_GLB_CHUNK_HEADER_BYTE_COUNT = 8
_GLB_JSON_CHUNK_TYPE = b"JSON"
_ABSOLUTE_GEOMETRY_EPSILON = 1e-10
_RELATIVE_GEOMETRY_EPSILON = 1e-9
_SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")
_MINIMUM_GLASS_DEPTH_CAP = "minimum"
_MAXIMUM_GLASS_DEPTH_CAP = "maximum"
_GLASS_DEPTH_CAP_NAMES = frozenset(
    (_MINIMUM_GLASS_DEPTH_CAP, _MAXIMUM_GLASS_DEPTH_CAP)
)


# ### Public data models ###
@dataclass(frozen=True, slots=True)
class NormalizedGlassArea:
    """One stable glass rectangle in normalized body-local X/Z space."""

    area_id: str
    minimum_x_ratio: float
    maximum_x_ratio: float
    minimum_z_ratio: float
    maximum_z_ratio: float
    name: str | None = None

    def __post_init__(self) -> None:
        area_id = str(self.area_id).strip()
        if not area_id:
            raise ValueError("Glass area IDs cannot be empty.")
        name = None if self.name is None else str(self.name).strip()
        if self.name is not None and not name:
            raise ValueError("Glass area names cannot be empty.")
        minimum_x, maximum_x = _normalize_ratio_bounds(
            self.minimum_x_ratio,
            self.maximum_x_ratio,
            "Glass area X",
        )
        minimum_z, maximum_z = _normalize_ratio_bounds(
            self.minimum_z_ratio,
            self.maximum_z_ratio,
            "Glass area Z",
        )
        object.__setattr__(self, "area_id", area_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "minimum_x_ratio", minimum_x)
        object.__setattr__(self, "maximum_x_ratio", maximum_x)
        object.__setattr__(self, "minimum_z_ratio", minimum_z)
        object.__setattr__(self, "maximum_z_ratio", maximum_z)

    @property
    def normalized_bounds(self) -> tuple[float, float, float, float]:
        """Return ``min X, max X, min Z, max Z`` normalized bounds."""

        return (
            self.minimum_x_ratio,
            self.maximum_x_ratio,
            self.minimum_z_ratio,
            self.maximum_z_ratio,
        )


@dataclass(frozen=True, slots=True)
class OpeningGlassPrimitive:
    """Stable assembly information for one generated glass cuboid."""

    area: NormalizedGlassArea
    node_name: str
    geometry_name: str
    bounds_meters: tuple[float, float, float, float]
    depth_bounds_meters: tuple[float, float]

    @property
    def center_meters(self) -> tuple[float, float, float]:
        """Return the cuboid center in body-local, Z-up coordinates."""

        minimum_x, maximum_x, minimum_z, maximum_z = self.bounds_meters
        minimum_y, maximum_y = self.depth_bounds_meters
        return (
            (minimum_x + maximum_x) * 0.5,
            (minimum_y + maximum_y) * 0.5,
            (minimum_z + maximum_z) * 0.5,
        )

    @property
    def extents_meters(self) -> tuple[float, float, float]:
        """Return the cuboid X/Y/Z extents in meters."""

        minimum_x, maximum_x, minimum_z, maximum_z = self.bounds_meters
        minimum_y, maximum_y = self.depth_bounds_meters
        return (
            maximum_x - minimum_x,
            maximum_y - minimum_y,
            maximum_z - minimum_z,
        )


@dataclass(frozen=True, slots=True)
class OpeningGlassResult:
    """A clipped body model and its separately addressable glass primitives."""

    model: GeneratedModel
    primitives: tuple[OpeningGlassPrimitive, ...]
    original_body_face_count: int
    result_body_face_count: int
    clipped_source_face_count: int
    glass_face_count: int
    preserved_textured_uvs: bool


# ### Internal data models ###
@dataclass(slots=True)
class _ClipVertex:
    """One interpolatable source vertex used by the exact polygon clipper."""

    local_position: np.ndarray
    body_position: np.ndarray
    normal: np.ndarray
    uv: np.ndarray | None
    vertex_color: np.ndarray | None


@dataclass(frozen=True, slots=True)
class _MetricGlassArea:
    """One validated normalized area resolved to body-local metric bounds."""

    source: NormalizedGlassArea
    minimum_x: float
    maximum_x: float
    minimum_z: float
    maximum_z: float

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (
            self.minimum_x,
            self.maximum_x,
            self.minimum_z,
            self.maximum_z,
        )


@dataclass(frozen=True, slots=True)
class _ClippedMeshResult:
    """One rebuilt mesh and exact source-face clipping statistics."""

    mesh: trimesh.Trimesh
    clipped_source_face_indices: frozenset[int]
    touched_area_indices: frozenset[int]


# ### Public API ###
def coerce_normalized_glass_area(value: object) -> NormalizedGlassArea:
    """Normalize a neutral mapping or ``GlassAreaData``-shaped object.

    The geometry module intentionally does not import door/window state.  It
    accepts either :class:`NormalizedGlassArea`, mappings, or objects exposing
    the persistent ``GlassAreaData`` field names.
    """

    if isinstance(value, NormalizedGlassArea):
        return value
    if isinstance(value, Mapping):
        read = value.get
    else:
        read = lambda key, default=None: getattr(value, key, default)
    area_id = read("area_id")
    if area_id is None:
        area_id = read("glass_area_id")
    if area_id is None:
        raise TypeError("Glass areas require an area_id or glass_area_id.")
    return NormalizedGlassArea(
        area_id=str(area_id),
        name=read("name"),
        minimum_x_ratio=read("minimum_x_ratio"),
        maximum_x_ratio=read("maximum_x_ratio"),
        minimum_z_ratio=read("minimum_z_ratio"),
        maximum_z_ratio=read("maximum_z_ratio"),
    )


def apply_opening_glass_areas(
    body_model: GeneratedModel,
    areas: Sequence[object],
    *,
    profile_bounds_meters: Sequence[float] | None = None,
    double_sided_glass: bool = False,
    omitted_glass_depth_cap: str | None = None,
) -> OpeningGlassResult:
    """Subtract X/Z rectangles through a body and fill them with glass.

    ``profile_bounds_meters`` is ``(min_x, max_x, min_z, max_z)`` in the
    model's Z-up coordinate system.  When omitted, the non-glass scene bounds
    are used.  Every cut spans the complete body-local Y depth.  The input is
    never mutated, and validation is atomic: no result is returned unless all
    requested areas intersect body geometry and remain inside its projected
    silhouette. ``omitted_glass_depth_cap`` is reserved for assemblies where
    two pane halves meet: it removes only the named, hidden local-Y cap while
    retaining the complete through-depth cut.

    This function must receive the unmodified source body.  Passing one of its
    own prior results is rejected because removed triangles cannot be restored
    when a persistent glass rectangle is later edited or deleted.
    """

    if not isinstance(body_model, GeneratedModel):
        raise TypeError("Opening glass requires a GeneratedModel body.")
    if not isinstance(double_sided_glass, bool):
        raise TypeError("Glass sidedness must be a boolean.")
    omitted_depth_cap = _normalize_omitted_glass_depth_cap(
        omitted_glass_depth_cap
    )
    normalized_areas = _normalize_glass_areas(areas)
    source_instances = _collect_source_instances(body_model.scene)
    inherited_half_mesh = _resolve_inherited_half_mesh_metadata(
        source_instances
    )
    if any(
        OPENING_GLASS_METADATA_KEY in instance[4]
        for instance in source_instances
    ):
        raise ValueError(
            "Opening glass must be rebuilt from the unmodified body model."
        )

    body_bounds = _resolve_body_bounds(source_instances)
    profile_bounds = _normalize_profile_bounds(
        profile_bounds_meters,
        body_bounds=body_bounds,
    )
    metric_areas = _resolve_metric_areas(normalized_areas, profile_bounds)
    tolerance = _geometry_tolerance(body_bounds)

    output_scene = trimesh.Scene(metadata=copy.deepcopy(body_model.scene.metadata))
    used_geometry_names: set[str] = set()
    used_node_names: set[str] = set()
    touched_area_indices: set[int] = set()
    clipped_source_faces = 0
    original_body_faces = 0
    result_body_faces = 0
    textured_sources = 0
    textured_results = 0

    for instance_index, (
        requested_node_name,
        requested_geometry_name,
        source_mesh,
        node_transform,
        node_metadata,
    ) in enumerate(source_instances):
        original_body_faces += len(source_mesh.faces)
        source_has_uv = _mesh_has_textured_uvs(source_mesh)
        textured_sources += int(source_has_uv)
        body_from_local = (
            GLTF_Y_UP_TO_Z_UP_TRANSFORM @ node_transform
        )
        clipped = _clip_mesh_by_rectangles(
            source_mesh,
            body_from_local=body_from_local,
            areas=metric_areas,
            tolerance=tolerance,
        )
        clipped_source_faces += len(clipped.clipped_source_face_indices)
        touched_area_indices.update(clipped.touched_area_indices)
        result_body_faces += len(clipped.mesh.faces)
        textured_results += int(_mesh_has_textured_uvs(clipped.mesh))
        geometry_name = _make_unique_name(
            requested_geometry_name,
            used_geometry_names,
            fallback=f"body_geometry_{instance_index}",
        )
        node_name = _make_unique_name(
            requested_node_name,
            used_node_names,
            fallback=f"body_node_{instance_index}",
        )
        output_scene.add_geometry(
            clipped.mesh,
            geom_name=geometry_name,
            node_name=node_name,
            transform=node_transform,
            metadata=copy.deepcopy(node_metadata),
        )

    untouched_area_indices = set(range(len(metric_areas))) - touched_area_indices
    if untouched_area_indices:
        untouched_names = ", ".join(
            metric_areas[index].source.name
            or metric_areas[index].source.area_id
            for index in sorted(untouched_area_indices)
        )
        raise ValueError(
            "Glass areas must intersect the body geometry. "
            f"No body faces were cut for: {untouched_names}."
        )
    _validate_areas_inside_projected_body(
        source_instances,
        metric_areas,
        tolerance=tolerance,
    )

    primitives: list[OpeningGlassPrimitive] = []
    extras_by_node_name: dict[str, dict[str, object]] = {}
    minimum_y, maximum_y = body_bounds[2], body_bounds[3]
    for area_index, metric_area in enumerate(metric_areas):
        node_name = _make_unique_name(
            _glass_node_name(metric_area.source),
            used_node_names,
            fallback=f"{OPENING_GLASS_NODE_PREFIX} area_{area_index + 1}",
        )
        geometry_name = _make_unique_name(
            f"opening_glass_{_safe_name(metric_area.source.area_id)}",
            used_geometry_names,
            fallback=f"opening_glass_{area_index + 1}",
        )
        metadata_payload = _build_glass_metadata_payload(metric_area)
        glass_mesh = _build_glass_cuboid(
            metric_area,
            minimum_y=minimum_y,
            maximum_y=maximum_y,
            double_sided=double_sided_glass,
            omitted_depth_cap=omitted_depth_cap,
            metadata_payload=metadata_payload,
        )
        node_metadata: dict[str, object] = {
            OPENING_GLASS_METADATA_KEY: metadata_payload,
        }
        if inherited_half_mesh is not None:
            inherited_marker = copy.deepcopy(inherited_half_mesh)
            glass_mesh.metadata[HALF_MESH_EXTRAS_KEY] = inherited_marker
            node_metadata[HALF_MESH_EXTRAS_KEY] = copy.deepcopy(
                inherited_marker
            )
        output_scene.add_geometry(
            glass_mesh,
            geom_name=geometry_name,
            node_name=node_name,
            metadata=node_metadata,
        )
        extras_by_node_name[node_name] = metadata_payload
        primitives.append(
            OpeningGlassPrimitive(
                area=metric_area.source,
                node_name=node_name,
                geometry_name=geometry_name,
                bounds_meters=metric_area.bounds,
                depth_bounds_meters=(minimum_y, maximum_y),
            )
        )

    output_model = _build_output_model(
        output_scene,
        extras_by_node_name=extras_by_node_name,
        source_model=body_model,
    )
    return OpeningGlassResult(
        model=output_model,
        primitives=tuple(primitives),
        original_body_face_count=original_body_faces,
        result_body_face_count=result_body_faces,
        clipped_source_face_count=clipped_source_faces,
        glass_face_count=sum(
            len(output_scene.geometry[primitive.geometry_name].faces)
            for primitive in primitives
        ),
        preserved_textured_uvs=(
            textured_sources == 0 or textured_sources == textured_results
        ),
    )


# ### Area validation ###
def _normalize_glass_areas(
    raw_areas: Sequence[object],
) -> tuple[NormalizedGlassArea, ...]:
    if isinstance(raw_areas, (str, bytes, bytearray)) or not isinstance(
        raw_areas,
        Sequence,
    ):
        raise TypeError("Opening glass areas must be a sequence.")
    areas = tuple(coerce_normalized_glass_area(area) for area in raw_areas)
    area_ids = tuple(area.area_id for area in areas)
    if len(area_ids) != len(set(area_ids)):
        raise ValueError("Opening glass area IDs must be unique.")
    for first_index, first in enumerate(areas):
        for second in areas[first_index + 1 :]:
            if _normalized_areas_overlap(first, second):
                raise ValueError(
                    "Opening glass rectangles cannot overlap: "
                    f"{first.area_id!r} and {second.area_id!r}."
                )
    return areas


def _normalize_omitted_glass_depth_cap(raw_cap: object) -> str | None:
    if raw_cap is None:
        return None
    if not isinstance(raw_cap, str):
        raise TypeError("The omitted glass depth cap must be a string or None.")
    cap = raw_cap.strip().lower()
    if cap not in _GLASS_DEPTH_CAP_NAMES:
        raise ValueError(
            "The omitted glass depth cap must be 'minimum' or 'maximum'."
        )
    return cap


def _normalize_ratio_bounds(
    raw_minimum: object,
    raw_maximum: object,
    label: str,
) -> tuple[float, float]:
    minimum = _normalize_finite_number(raw_minimum, f"{label} minimum")
    maximum = _normalize_finite_number(raw_maximum, f"{label} maximum")
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(f"{label} ratios must remain between zero and one.")
    if maximum - minimum < MIN_NORMALIZED_GLASS_SPAN:
        raise ValueError(f"{label} span is too small.")
    return minimum, maximum


def _normalized_areas_overlap(
    first: NormalizedGlassArea,
    second: NormalizedGlassArea,
) -> bool:
    overlap_x = min(first.maximum_x_ratio, second.maximum_x_ratio) - max(
        first.minimum_x_ratio,
        second.minimum_x_ratio,
    )
    overlap_z = min(first.maximum_z_ratio, second.maximum_z_ratio) - max(
        first.minimum_z_ratio,
        second.minimum_z_ratio,
    )
    return overlap_x > MIN_NORMALIZED_GLASS_SPAN and (
        overlap_z > MIN_NORMALIZED_GLASS_SPAN
    )


def _normalize_profile_bounds(
    raw_bounds: Sequence[float] | None,
    *,
    body_bounds: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float]:
    if raw_bounds is None:
        return body_bounds[0], body_bounds[1], body_bounds[4], body_bounds[5]
    if isinstance(raw_bounds, (str, bytes, bytearray)) or not isinstance(
        raw_bounds,
        Sequence,
    ):
        raise TypeError("Glass profile bounds must contain four numbers.")
    if len(raw_bounds) != 4:
        raise ValueError("Glass profile bounds must contain four numbers.")
    minimum_x, maximum_x, minimum_z, maximum_z = (
        _normalize_finite_number(value, "Glass profile bound")
        for value in raw_bounds
    )
    if maximum_x <= minimum_x or maximum_z <= minimum_z:
        raise ValueError("Glass profile bounds must have positive X and Z spans.")
    return minimum_x, maximum_x, minimum_z, maximum_z


def _resolve_metric_areas(
    areas: Sequence[NormalizedGlassArea],
    profile_bounds: tuple[float, float, float, float],
) -> tuple[_MetricGlassArea, ...]:
    minimum_x, maximum_x, minimum_z, maximum_z = profile_bounds
    width = maximum_x - minimum_x
    height = maximum_z - minimum_z
    return tuple(
        _MetricGlassArea(
            source=area,
            minimum_x=minimum_x + area.minimum_x_ratio * width,
            maximum_x=minimum_x + area.maximum_x_ratio * width,
            minimum_z=minimum_z + area.minimum_z_ratio * height,
            maximum_z=minimum_z + area.maximum_z_ratio * height,
        )
        for area in areas
    )


def _validate_areas_inside_projected_body(
    instances: Sequence[
        tuple[
            str,
            str,
            trimesh.Trimesh,
            np.ndarray,
            dict[str, object],
        ]
    ],
    areas: Sequence[_MetricGlassArea],
    *,
    tolerance: float,
) -> None:
    """Require each rectangular pane to fit the body's X/Z silhouette."""

    if not areas:
        return
    triangles: list[Polygon] = []
    for _node_name, _geometry_name, mesh, transform, _metadata in instances:
        body_vertices = trimesh.transform_points(
            np.asarray(mesh.vertices, dtype=float),
            GLTF_Y_UP_TO_Z_UP_TRANSFORM @ transform,
        )
        for face in np.asarray(mesh.faces, dtype=np.int64):
            triangle = Polygon(body_vertices[face][:, (0, 2)])
            if triangle.area > tolerance * tolerance:
                triangles.append(triangle)
    if not triangles:
        raise ValueError("The opening body has no measurable X/Z silhouette.")
    try:
        silhouette = shapely.union_all(triangles)
        safe_silhouette = shapely.buffer(silhouette, tolerance)
        outside_areas = [
            area
            for area in areas
            if not shapely.covers(
                safe_silhouette,
                shapely.box(
                    area.minimum_x,
                    area.minimum_z,
                    area.maximum_x,
                    area.maximum_z,
                ),
            )
        ]
    except ShapelyError as error:
        raise ValueError(
            "The opening body's projected silhouette could not be validated."
        ) from error
    if not outside_areas:
        return
    outside_names = ", ".join(
        area.source.name or area.source.area_id for area in outside_areas
    )
    raise ValueError(
        "Rectangular glass areas must remain inside the body silhouette. "
        f"Outside the silhouette: {outside_names}."
    )


# ### Scene collection ###
def _collect_source_instances(
    scene: trimesh.Scene,
) -> list[
    tuple[
        str,
        str,
        trimesh.Trimesh,
        np.ndarray,
        dict[str, object],
    ]
]:
    if not isinstance(scene, trimesh.Scene):
        raise TypeError("Opening glass requires a trimesh scene.")
    instances: list[
        tuple[
            str,
            str,
            trimesh.Trimesh,
            np.ndarray,
            dict[str, object],
        ]
    ] = []
    for node_name in sorted(scene.graph.nodes_geometry, key=str):
        node_transform, geometry_name = scene.graph.get(node_name)
        geometry = scene.geometry.get(geometry_name)
        if not isinstance(geometry, trimesh.Trimesh) or geometry.is_empty:
            continue
        transform = np.asarray(node_transform, dtype=float)
        vertices = np.asarray(geometry.vertices, dtype=float)
        faces = np.asarray(geometry.faces, dtype=np.int64)
        if (
            transform.shape != (4, 4)
            or not np.all(np.isfinite(transform))
            or vertices.ndim != 2
            or vertices.shape[1:] != (3,)
            or not np.all(np.isfinite(vertices))
            or faces.ndim != 2
            or faces.shape[1:] != (3,)
            or np.any(faces < 0)
            or np.any(faces >= len(vertices))
        ):
            raise ValueError("The opening body contains invalid mesh geometry.")
        if _mesh_is_prefab_glass(geometry):
            raise ValueError(
                "Opening glass must be rebuilt from the unmodified body "
                "model, without prefab glass geometry."
            )
        instances.append(
            (
                str(node_name),
                str(geometry_name),
                geometry,
                transform.copy(),
                _get_scene_node_metadata(scene, node_name),
            )
        )
    if not instances:
        raise ValueError("The opening body contains no triangle geometry.")
    return instances


def _resolve_body_bounds(
    instances: Sequence[
        tuple[
            str,
            str,
            trimesh.Trimesh,
            np.ndarray,
            dict[str, object],
        ]
    ],
) -> tuple[float, float, float, float, float, float]:
    body_points: list[np.ndarray] = []
    for _node_name, _geometry_name, mesh, transform, _metadata in instances:
        body_points.append(
            trimesh.transform_points(
                np.asarray(mesh.vertices, dtype=float),
                GLTF_Y_UP_TO_Z_UP_TRANSFORM @ transform,
            )
        )
    points = np.vstack(body_points)
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    if np.any(maximum - minimum <= _ABSOLUTE_GEOMETRY_EPSILON):
        raise ValueError(
            "The opening body must have positive width, depth, and height."
        )
    return (
        float(minimum[0]),
        float(maximum[0]),
        float(minimum[1]),
        float(maximum[1]),
        float(minimum[2]),
        float(maximum[2]),
    )


def _resolve_inherited_half_mesh_metadata(
    instances: Sequence[
        tuple[
            str,
            str,
            trimesh.Trimesh,
            np.ndarray,
            dict[str, object],
        ]
    ],
) -> dict[str, object] | None:
    """Copy one consistent half-mesh reconstruction marker to every pane."""

    markers = [
        metadata[HALF_MESH_EXTRAS_KEY]
        for _node, _geometry, _mesh, _transform, metadata in instances
        if HALF_MESH_EXTRAS_KEY in metadata
    ]
    if not markers:
        return None
    if any(not isinstance(marker, Mapping) for marker in markers):
        raise ValueError("The opening body has invalid half-mesh metadata.")
    try:
        encoded_markers = {
            json.dumps(marker, sort_keys=True, separators=(",", ":"))
            for marker in markers
        }
    except (TypeError, ValueError) as error:
        raise ValueError(
            "The opening body has invalid half-mesh metadata."
        ) from error
    if len(encoded_markers) != 1:
        raise ValueError(
            "Opening glass requires one consistent half-mesh mirror plane."
        )
    return copy.deepcopy(dict(markers[0]))


def _get_scene_node_metadata(
    scene: trimesh.Scene,
    node_name: object,
) -> dict[str, object]:
    parent_name = scene.graph.transforms.parents.get(node_name)
    if parent_name is None:
        return {}
    edge_data = scene.graph.transforms.edge_data.get(
        (parent_name, node_name),
        {},
    )
    raw_metadata = edge_data.get("metadata")
    if not isinstance(raw_metadata, Mapping):
        return {}
    return copy.deepcopy(dict(raw_metadata))


# ### Exact rectangle subtraction ###
def _clip_mesh_by_rectangles(
    source_mesh: trimesh.Trimesh,
    *,
    body_from_local: np.ndarray,
    areas: Sequence[_MetricGlassArea],
    tolerance: float,
) -> _ClippedMeshResult:
    if not areas:
        return _ClippedMeshResult(
            mesh=copy.deepcopy(source_mesh),
            clipped_source_face_indices=frozenset(),
            touched_area_indices=frozenset(),
        )
    vertices = np.asarray(source_mesh.vertices, dtype=float)
    faces = np.asarray(source_mesh.faces, dtype=np.int64)
    body_vertices = trimesh.transform_points(vertices, body_from_local)
    normals = _source_vertex_normals(source_mesh)
    source_uv = _source_uv(source_mesh)
    source_vertex_colors = _source_vertex_colors(source_mesh)
    source_face_colors = _source_face_colors(source_mesh)
    source_face_materials = _source_face_materials(source_mesh)

    output_vertices: list[np.ndarray] = []
    output_normals: list[np.ndarray] = []
    output_uv: list[np.ndarray] = []
    output_vertex_colors: list[np.ndarray] = []
    output_faces: list[tuple[int, int, int]] = []
    output_face_colors: list[np.ndarray] = []
    output_face_materials: list[int] = []
    clipped_source_faces: set[int] = set()
    touched_area_indices: set[int] = set()

    for source_face_index, face in enumerate(faces):
        fragments: list[list[_ClipVertex]] = [
            [
                _ClipVertex(
                    local_position=vertices[int(vertex_index)].copy(),
                    body_position=body_vertices[int(vertex_index)].copy(),
                    normal=normals[int(vertex_index)].copy(),
                    uv=(
                        None
                        if source_uv is None
                        else source_uv[int(vertex_index)].copy()
                    ),
                    vertex_color=(
                        None
                        if source_vertex_colors is None
                        else source_vertex_colors[int(vertex_index)].copy()
                    ),
                )
                for vertex_index in face
            ]
        ]
        for area_index, area in enumerate(areas):
            next_fragments: list[list[_ClipVertex]] = []
            area_touched = False
            for fragment in fragments:
                retained, changed = _subtract_rectangle_from_polygon(
                    fragment,
                    area,
                    tolerance=tolerance,
                )
                next_fragments.extend(retained)
                area_touched = area_touched or changed
            if area_touched:
                touched_area_indices.add(area_index)
                clipped_source_faces.add(source_face_index)
            fragments = next_fragments
            if not fragments:
                break

        for fragment in fragments:
            for triangle in _triangulate_polygon(
                fragment,
                tolerance=tolerance,
            ):
                first_vertex = len(output_vertices)
                for vertex in triangle:
                    output_vertices.append(vertex.local_position.copy())
                    output_normals.append(vertex.normal.copy())
                    if source_uv is not None and vertex.uv is not None:
                        output_uv.append(vertex.uv.copy())
                    if (
                        source_vertex_colors is not None
                        and vertex.vertex_color is not None
                    ):
                        output_vertex_colors.append(vertex.vertex_color.copy())
                output_faces.append(
                    (first_vertex, first_vertex + 1, first_vertex + 2)
                )
                if source_face_colors is not None:
                    output_face_colors.append(
                        source_face_colors[source_face_index].copy()
                    )
                if source_face_materials is not None:
                    output_face_materials.append(
                        int(source_face_materials[source_face_index])
                    )

    if not output_faces:
        raise ValueError("Opening glass would remove the complete body mesh.")
    rebuilt = trimesh.Trimesh(
        vertices=np.asarray(output_vertices, dtype=float),
        faces=np.asarray(output_faces, dtype=np.int64),
        metadata=copy.deepcopy(source_mesh.metadata),
        process=False,
        validate=False,
    )
    rebuilt.vertex_normals = _normalize_vectors(
        np.asarray(output_normals, dtype=float)
    )
    _apply_rebuilt_visual(
        rebuilt,
        source_mesh=source_mesh,
        output_uv=(
            None
            if source_uv is None
            else np.asarray(output_uv, dtype=float)
        ),
        output_vertex_colors=(
            None
            if source_vertex_colors is None
            else np.asarray(output_vertex_colors)
        ),
        output_face_colors=(
            None
            if source_face_colors is None
            else np.asarray(output_face_colors)
        ),
        output_face_materials=(
            None
            if source_face_materials is None
            else np.asarray(output_face_materials, dtype=np.int64)
        ),
    )
    return _ClippedMeshResult(
        mesh=rebuilt,
        clipped_source_face_indices=frozenset(clipped_source_faces),
        touched_area_indices=frozenset(touched_area_indices),
    )


def _subtract_rectangle_from_polygon(
    polygon: Sequence[_ClipVertex],
    area: _MetricGlassArea,
    *,
    tolerance: float,
) -> tuple[list[list[_ClipVertex]], bool]:
    """Return a convex polygon minus one closed, through-depth rectangle."""

    pending = list(polygon)
    outside_fragments: list[list[_ClipVertex]] = []
    planes = (
        (0, area.minimum_x, 1.0),
        (0, area.maximum_x, -1.0),
        (2, area.minimum_z, 1.0),
        (2, area.maximum_z, -1.0),
    )
    for axis, coordinate, sign in planes:
        inside, outside = _split_polygon_by_plane(
            pending,
            axis=axis,
            coordinate=coordinate,
            inside_sign=sign,
            tolerance=tolerance,
        )
        if _polygon_has_area(outside, tolerance=tolerance):
            outside_fragments.append(outside)
        pending = inside
        if not _polygon_has_area(pending, tolerance=tolerance):
            return [list(polygon)], False
    return outside_fragments, True


def _split_polygon_by_plane(
    polygon: Sequence[_ClipVertex],
    *,
    axis: int,
    coordinate: float,
    inside_sign: float,
    tolerance: float,
) -> tuple[list[_ClipVertex], list[_ClipVertex]]:
    if len(polygon) < 3:
        return [], []
    inside: list[_ClipVertex] = []
    outside: list[_ClipVertex] = []
    previous = polygon[-1]
    previous_distance = inside_sign * (
        float(previous.body_position[axis]) - coordinate
    )
    previous_inside = previous_distance >= -tolerance
    for current in polygon:
        current_distance = inside_sign * (
            float(current.body_position[axis]) - coordinate
        )
        current_inside = current_distance >= -tolerance
        if current_inside != previous_inside:
            denominator = previous_distance - current_distance
            fraction = (
                0.5
                if abs(denominator) <= _ABSOLUTE_GEOMETRY_EPSILON
                else previous_distance / denominator
            )
            intersection = _interpolate_clip_vertex(
                previous,
                current,
                min(max(float(fraction), 0.0), 1.0),
            )
            inside.append(intersection)
            outside.append(copy.deepcopy(intersection))
        if current_inside:
            inside.append(copy.deepcopy(current))
        else:
            outside.append(copy.deepcopy(current))
        previous = current
        previous_distance = current_distance
        previous_inside = current_inside
    return (
        _remove_duplicate_polygon_vertices(inside, tolerance=tolerance),
        _remove_duplicate_polygon_vertices(outside, tolerance=tolerance),
    )


def _interpolate_clip_vertex(
    start: _ClipVertex,
    end: _ClipVertex,
    fraction: float,
) -> _ClipVertex:
    inverse = 1.0 - fraction
    normal = start.normal * inverse + end.normal * fraction
    normal_length = float(np.linalg.norm(normal))
    if normal_length > _ABSOLUTE_GEOMETRY_EPSILON:
        normal = normal / normal_length
    return _ClipVertex(
        local_position=start.local_position * inverse + end.local_position * fraction,
        body_position=start.body_position * inverse + end.body_position * fraction,
        normal=normal,
        uv=(
            None
            if start.uv is None or end.uv is None
            else start.uv * inverse + end.uv * fraction
        ),
        vertex_color=(
            None
            if start.vertex_color is None or end.vertex_color is None
            else start.vertex_color * inverse + end.vertex_color * fraction
        ),
    )


def _remove_duplicate_polygon_vertices(
    polygon: Sequence[_ClipVertex],
    *,
    tolerance: float,
) -> list[_ClipVertex]:
    retained: list[_ClipVertex] = []
    for vertex in polygon:
        if retained and np.linalg.norm(
            vertex.body_position - retained[-1].body_position
        ) <= tolerance:
            continue
        retained.append(vertex)
    if len(retained) > 1 and np.linalg.norm(
        retained[0].body_position - retained[-1].body_position
    ) <= tolerance:
        retained.pop()
    return retained


def _triangulate_polygon(
    polygon: Sequence[_ClipVertex],
    *,
    tolerance: float,
) -> list[tuple[_ClipVertex, _ClipVertex, _ClipVertex]]:
    if len(polygon) < 3:
        return []
    triangles: list[tuple[_ClipVertex, _ClipVertex, _ClipVertex]] = []
    anchor = polygon[0]
    for index in range(1, len(polygon) - 1):
        triangle = (anchor, polygon[index], polygon[index + 1])
        points = np.asarray(
            [vertex.body_position for vertex in triangle],
            dtype=float,
        )
        doubled_area = np.linalg.norm(
            np.cross(points[1] - points[0], points[2] - points[0])
        )
        if doubled_area > tolerance * tolerance:
            triangles.append(triangle)
    return triangles


def _polygon_has_area(
    polygon: Sequence[_ClipVertex],
    *,
    tolerance: float,
) -> bool:
    return bool(_triangulate_polygon(polygon, tolerance=tolerance))


# ### Visual preservation ###
def _source_uv(mesh: trimesh.Trimesh) -> np.ndarray | None:
    if not isinstance(mesh.visual, TextureVisuals):
        return None
    raw_uv = getattr(mesh.visual, "uv", None)
    if raw_uv is None:
        return None
    uv = np.asarray(raw_uv, dtype=float)
    if uv.shape != (len(mesh.vertices), 2) or not np.all(np.isfinite(uv)):
        raise ValueError("The opening body contains invalid UV coordinates.")
    return uv.copy()


def _source_face_materials(mesh: trimesh.Trimesh) -> np.ndarray | None:
    if not isinstance(mesh.visual, TextureVisuals):
        return None
    raw_face_materials = getattr(mesh.visual, "face_materials", None)
    if raw_face_materials is None:
        return None
    face_materials = np.asarray(raw_face_materials, dtype=np.int64)
    if face_materials.shape != (len(mesh.faces),) or np.any(face_materials < 0):
        raise ValueError("The opening body contains invalid material indices.")
    return face_materials.copy()


def _source_vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray | None:
    if not isinstance(mesh.visual, ColorVisuals) or mesh.visual.kind != "vertex":
        return None
    colors = np.asarray(mesh.visual.vertex_colors)
    if colors.shape[0] != len(mesh.vertices):
        raise ValueError("The opening body contains invalid vertex colors.")
    return colors.copy()


def _source_face_colors(mesh: trimesh.Trimesh) -> np.ndarray | None:
    if not isinstance(mesh.visual, ColorVisuals) or mesh.visual.kind != "face":
        return None
    colors = np.asarray(mesh.visual.face_colors)
    if colors.shape[0] != len(mesh.faces):
        raise ValueError("The opening body contains invalid face colors.")
    return colors.copy()


def _source_vertex_normals(mesh: trimesh.Trimesh) -> np.ndarray:
    try:
        normals = np.asarray(mesh.vertex_normals, dtype=float)
    except (AttributeError, TypeError, ValueError):
        normals = np.empty((0, 3), dtype=float)
    if normals.shape != (len(mesh.vertices), 3) or not np.all(np.isfinite(normals)):
        raise ValueError("The opening body contains invalid vertex normals.")
    return _normalize_vectors(normals)


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    result = np.asarray(vectors, dtype=float).copy()
    lengths = np.linalg.norm(result, axis=1)
    nonzero = lengths > _ABSOLUTE_GEOMETRY_EPSILON
    result[nonzero] /= lengths[nonzero, None]
    return result


def _apply_rebuilt_visual(
    rebuilt: trimesh.Trimesh,
    *,
    source_mesh: trimesh.Trimesh,
    output_uv: np.ndarray | None,
    output_vertex_colors: np.ndarray | None,
    output_face_colors: np.ndarray | None,
    output_face_materials: np.ndarray | None,
) -> None:
    source_visual = source_mesh.visual
    if isinstance(source_visual, TextureVisuals):
        rebuilt.visual = TextureVisuals(
            uv=output_uv,
            material=copy.deepcopy(source_visual.material),
            face_materials=output_face_materials,
        )
        return
    if output_vertex_colors is not None:
        rebuilt.visual = ColorVisuals(
            mesh=rebuilt,
            vertex_colors=np.clip(
                np.rint(output_vertex_colors),
                0,
                255,
            ).astype(np.uint8),
        )
        return
    if output_face_colors is not None:
        rebuilt.visual = ColorVisuals(
            mesh=rebuilt,
            face_colors=output_face_colors,
        )


def _mesh_has_textured_uvs(mesh: trimesh.Trimesh) -> bool:
    return isinstance(mesh.visual, TextureVisuals) and (
        _source_uv(mesh) is not None
    )


def _mesh_is_prefab_glass(mesh: trimesh.Trimesh) -> bool:
    if not isinstance(mesh.visual, TextureVisuals):
        return False
    material = mesh.visual.material
    if isinstance(material, MultiMaterial):
        materials = tuple(material.materials)
        return bool(materials) and all(
            is_housemaker_glass_material(item) for item in materials
        )
    return is_housemaker_glass_material(material)


# ### Glass primitive construction ###
def _build_glass_cuboid(
    area: _MetricGlassArea,
    *,
    minimum_y: float,
    maximum_y: float,
    double_sided: bool,
    omitted_depth_cap: str | None,
    metadata_payload: Mapping[str, object],
) -> trimesh.Trimesh:
    extents = (
        area.maximum_x - area.minimum_x,
        maximum_y - minimum_y,
        area.maximum_z - area.minimum_z,
    )
    center = (
        (area.minimum_x + area.maximum_x) * 0.5,
        (minimum_y + maximum_y) * 0.5,
        (area.minimum_z + area.maximum_z) * 0.5,
    )
    glass_mesh = trimesh.creation.box(
        extents=extents,
        transform=trimesh.transformations.translation_matrix(center),
    )
    if omitted_depth_cap is not None:
        _remove_glass_depth_cap(
            glass_mesh,
            cap_name=omitted_depth_cap,
            minimum_y=minimum_y,
            maximum_y=maximum_y,
        )
    glass_mesh.metadata[OPENING_GLASS_METADATA_KEY] = copy.deepcopy(
        dict(metadata_payload)
    )
    glass_mesh.visual = TextureVisuals(
        uv=np.zeros((len(glass_mesh.vertices), 2), dtype=float),
        material=build_housemaker_glass_material(double_sided),
    )
    glass_mesh.apply_transform(Z_UP_TO_GLTF_Y_UP_TRANSFORM)
    return glass_mesh


def _remove_glass_depth_cap(
    glass_mesh: trimesh.Trimesh,
    *,
    cap_name: str,
    minimum_y: float,
    maximum_y: float,
) -> None:
    """Remove one hidden pane cap where two half-depth panes meet."""

    coordinate = (
        minimum_y
        if cap_name == _MINIMUM_GLASS_DEPTH_CAP
        else maximum_y
    )
    vertices = np.asarray(glass_mesh.vertices, dtype=float)
    faces = np.asarray(glass_mesh.faces, dtype=np.int64)
    face_vertices = vertices[faces]
    tolerance = max(
        _ABSOLUTE_GEOMETRY_EPSILON,
        abs(maximum_y - minimum_y) * _RELATIVE_GEOMETRY_EPSILON,
    )
    cap_faces = np.all(
        np.isclose(
            face_vertices[:, :, 1],
            coordinate,
            atol=tolerance,
            rtol=0.0,
        ),
        axis=1,
    )
    if int(np.count_nonzero(cap_faces)) != 2:
        raise ValueError("The hidden glass depth cap could not be identified.")
    glass_mesh.update_faces(~cap_faces)
    glass_mesh.remove_unreferenced_vertices()


def _build_glass_metadata_payload(
    area: _MetricGlassArea,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "areaId": area.source.area_id,
        "normalizedBounds": {
            "minimumX": area.source.minimum_x_ratio,
            "maximumX": area.source.maximum_x_ratio,
            "minimumZ": area.source.minimum_z_ratio,
            "maximumZ": area.source.maximum_z_ratio,
        },
        "uvMode": "independent",
    }
    if area.source.name is not None:
        payload["name"] = area.source.name
    return payload


# ### Output model construction ###
def _build_output_model(
    scene: trimesh.Scene,
    *,
    extras_by_node_name: Mapping[str, Mapping[str, object]],
    source_model: GeneratedModel,
) -> GeneratedModel:
    if not scene.geometry:
        raise ValueError("Opening glass produced an empty object scene.")
    try:
        if source_model.preview_only:
            # Preview scenes intentionally retain the authoritative source GLB
            # while using bounded texture proxies.  Their edited scene is
            # render-only and must not be serialized as an export payload.
            glb_bytes = source_model.glb_bytes
        else:
            glb_bytes = _serialize_scene_glb_with_half_mesh_extras(
                scene,
                failure_message="The opening-glass object could not be exported.",
            )
            glb_bytes = _inject_opening_glass_extras(
                glb_bytes,
                extras_by_node_name,
            )
        preview_mesh = scene.to_geometry()
    except Exception as error:
        raise ValueError(
            "The opening-glass object could not be exported."
        ) from error
    if not isinstance(preview_mesh, trimesh.Trimesh) or preview_mesh.is_empty:
        raise ValueError("Opening glass produced no preview geometry.")
    preview_mesh = copy.deepcopy(preview_mesh)
    preview_mesh.apply_transform(GLTF_Y_UP_TO_Z_UP_TRANSFORM)
    return GeneratedModel(
        mesh=preview_mesh,
        scene=scene,
        glb_bytes=glb_bytes,
        preview_only=source_model.preview_only,
    )


def _inject_opening_glass_extras(
    payload: bytes,
    extras_by_node_name: Mapping[str, Mapping[str, object]],
) -> bytes:
    if not extras_by_node_name:
        return payload
    try:
        if (
            len(payload) < _GLB_HEADER_BYTE_COUNT + _GLB_CHUNK_HEADER_BYTE_COUNT
            or payload[:4] != b"glTF"
            or int.from_bytes(payload[8:12], "little") != len(payload)
            or payload[16:20] != _GLB_JSON_CHUNK_TYPE
        ):
            raise ValueError("The GLB header is invalid.")
        json_byte_count = int.from_bytes(payload[12:16], "little")
        json_end = (
            _GLB_HEADER_BYTE_COUNT
            + _GLB_CHUNK_HEADER_BYTE_COUNT
            + json_byte_count
        )
        if json_byte_count <= 0 or json_end > len(payload):
            raise ValueError("The GLB JSON chunk is invalid.")
        document = json.loads(
            payload[20:json_end].rstrip(b" \t\r\n\0").decode("utf-8")
        )
        nodes = document.get("nodes", ())
        if not isinstance(nodes, list):
            raise TypeError("The GLB node table is invalid.")
        matched: set[str] = set()
        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_name = str(node.get("name", ""))
            metadata = extras_by_node_name.get(node_name)
            if metadata is None:
                continue
            extras = node.setdefault("extras", {})
            if not isinstance(extras, dict):
                extras = {}
                node["extras"] = extras
            extras[OPENING_GLASS_EXTRAS_KEY] = copy.deepcopy(dict(metadata))
            matched.add(node_name)
        missing = set(extras_by_node_name) - matched
        if missing:
            raise ValueError("The exported glass nodes could not be identified.")
        encoded_document = json.dumps(
            document,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (AttributeError, TypeError, UnicodeError, ValueError) as error:
        raise ValueError(
            "The opening-glass metadata could not be serialized."
        ) from error
    encoded_document += b" " * (-len(encoded_document) % 4)
    trailing_chunks = payload[json_end:]
    total_length = (
        _GLB_HEADER_BYTE_COUNT
        + _GLB_CHUNK_HEADER_BYTE_COUNT
        + len(encoded_document)
        + len(trailing_chunks)
    )
    return b"".join(
        (
            b"glTF",
            payload[4:8],
            total_length.to_bytes(4, "little"),
            len(encoded_document).to_bytes(4, "little"),
            _GLB_JSON_CHUNK_TYPE,
            encoded_document,
            trailing_chunks,
        )
    )


# ### Naming helpers ###
def _glass_node_name(area: NormalizedGlassArea) -> str:
    label = area.name or area.area_id
    return f"{OPENING_GLASS_NODE_PREFIX} {_safe_name(label)}"


def _safe_name(value: object) -> str:
    normalized = _SAFE_NAME_PATTERN.sub("_", str(value).strip()).strip("_.")
    return normalized or "area"


def _make_unique_name(
    requested_name: str,
    used_names: set[str],
    *,
    fallback: str,
) -> str:
    base_name = str(requested_name).strip() or fallback
    candidate = base_name
    suffix = 2
    while candidate in used_names:
        candidate = f"{base_name}_{suffix}"
        suffix += 1
    used_names.add(candidate)
    return candidate


# ### Numeric helpers ###
def _normalize_finite_number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a number.")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{label} must be a number.") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be finite.")
    return normalized


def _geometry_tolerance(
    bounds: tuple[float, float, float, float, float, float],
) -> float:
    span = max(
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4],
        1.0,
    )
    return max(_ABSOLUTE_GEOMETRY_EPSILON, span * _RELATIVE_GEOMETRY_EPSILON)
