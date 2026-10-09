# ### Imports ###
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import trimesh
import xatlas
from trimesh.visual.texture import TextureVisuals

from housemaker.glb import import_generated_glb
from housemaker.object_texture_variants import prepare_uv_rewrite_material_textures
from housemaker.scan_projection_layout import clear_scan_projection_layout_metadata

# ### Constants ###
CORNICE_UV_MAX_STRIP_ASPECT = 4.0
CORNICE_UV_MAX_SEGMENT_COUNT = 64
CORNICE_UV_GUTTER_PIXELS = 8
CORNICE_UV_MIN_TEXTURE_RESOLUTION = 64
CORNICE_UV_MAX_TEXTURE_RESOLUTION = 16_384
_GEOMETRY_EPSILON = 1e-10
_UV_EPSILON = 1e-8
_PACKING_BINARY_SEARCH_STEPS = 28


# ### Data models ###
@dataclass(frozen=True)
class _SourcePrimitive:
    """One scene primitive with its node transform baked into its vertices."""

    node_name: str
    mesh: trimesh.Trimesh


@dataclass(frozen=True)
class _UvSubmission:
    """One longitudinal piece submitted to xatlas as an independent mesh."""

    submission_index: int
    segment_index: int
    node_name: str
    mesh: trimesh.Trimesh


@dataclass(frozen=True)
class _UvOutput:
    """Validated xatlas output for one longitudinal mesh piece."""

    submission: _UvSubmission
    vertex_mapping: np.ndarray
    faces: np.ndarray
    pixel_uvs: np.ndarray


@dataclass(frozen=True)
class _UvChart:
    """One cardinally oriented chart awaiting horizontal-row packing."""

    chart_index: int
    output_index: int
    vertex_indices: np.ndarray
    local_uvs: np.ndarray
    width: float
    height: float


@dataclass(frozen=True)
class _ChartPlacement:
    """Top-left pixel position for one packed chart."""

    x: float
    y: float


class _SegmentMeshBuilder:
    """Weld clipped triangles while retaining their source material indices."""

    def __init__(
        self,
        *,
        weld_tolerance: float,
        area_epsilon: float,
    ) -> None:
        self._weld_tolerance = max(float(weld_tolerance), _GEOMETRY_EPSILON)
        self._area_epsilon = max(float(area_epsilon), np.finfo(np.float64).eps)
        self._vertices: list[np.ndarray] = []
        self._faces: list[tuple[int, int, int]] = []
        self._face_materials: list[int] = []
        self._vertex_by_key: dict[tuple[int, int, int], int] = {}

    def add_triangle(
        self,
        triangle: np.ndarray,
        *,
        face_material: int,
    ) -> None:
        points = np.asarray(triangle, dtype=np.float64)
        doubled_area = float(
            np.linalg.norm(
                np.cross(points[1] - points[0], points[2] - points[0])
            )
        )
        if not math.isfinite(doubled_area) or doubled_area <= self._area_epsilon:
            return
        indices = tuple(self._add_vertex(point) for point in points)
        if len(set(indices)) != 3:
            return
        self._faces.append(indices)
        self._face_materials.append(int(face_material))

    def build(self, source_mesh: trimesh.Trimesh) -> trimesh.Trimesh | None:
        if not self._faces:
            return None
        material = copy.deepcopy(getattr(source_mesh.visual, "material", None))
        face_materials = np.asarray(self._face_materials, dtype=np.int64)
        return trimesh.Trimesh(
            vertices=np.asarray(self._vertices, dtype=np.float64),
            faces=np.asarray(self._faces, dtype=np.int64),
            visual=TextureVisuals(
                material=material,
                face_materials=face_materials,
            ),
            metadata=copy.deepcopy(source_mesh.metadata),
            process=False,
            validate=False,
        )

    def _add_vertex(self, point: np.ndarray) -> int:
        key = tuple(
            int(round(float(value) / self._weld_tolerance))
            for value in point
        )
        existing = self._vertex_by_key.get(key)
        if existing is not None:
            return existing
        vertex_index = len(self._vertices)
        self._vertices.append(np.asarray(point, dtype=np.float64).copy())
        self._vertex_by_key[key] = vertex_index
        return vertex_index


# ### Public API ###
def build_cornice_segmented_uv_glb(
    glb_bytes: bytes,
    *,
    texture_resolution: int = 2048,
) -> bytes:
    """Build a true-aspect, horizontal-strip UV atlas for a cornice GLB.

    Long X-axis geometry is divided before unwrapping so a full cornice cannot
    consume the atlas as one extremely thin chart. Xatlas parameterizes all
    pieces together, after which every chart is packed with one shared scale.
    Consequently no chart is independently stretched to fill its row.
    """

    resolution = _normalize_texture_resolution(texture_resolution)
    source_model = import_generated_glb(glb_bytes)
    clear_scan_projection_layout_metadata(source_model.scene)
    prepare_uv_rewrite_material_textures(
        source_model.scene,
        operation_name="Cornice segmented UV layout",
    )
    source_primitives = _collect_source_primitives(source_model.scene)
    global_bounds = _primitive_bounds(source_primitives)
    segment_count = _choose_segment_count(global_bounds, resolution=resolution)
    submissions = _split_primitives_into_segments(
        source_primitives,
        global_bounds=global_bounds,
        segment_count=segment_count,
    )
    outputs = _unwrap_submissions(submissions, resolution=resolution)
    charts = _collect_oriented_charts(outputs)
    final_uvs = _pack_charts_into_horizontal_rows(
        outputs,
        charts,
        texture_resolution=resolution,
    )
    output_scene = _build_output_scene(outputs, final_uvs)
    exported = output_scene.export(file_type="glb")
    if not isinstance(exported, bytes) or not exported:
        raise ValueError("The cornice UV layout could not be serialized.")
    _validate_uv_glb(
        exported,
        expected_bounds=np.asarray(source_model.mesh.bounds, dtype=np.float64),
    )
    return exported


# ### Input helpers ###
def _normalize_texture_resolution(texture_resolution: object) -> int:
    if isinstance(texture_resolution, (bool, np.bool_)) or not isinstance(
        texture_resolution,
        (int, np.integer),
    ):
        raise TypeError("Cornice texture resolution must be an integer.")
    resolution = int(texture_resolution)
    if not CORNICE_UV_MIN_TEXTURE_RESOLUTION <= resolution <= (
        CORNICE_UV_MAX_TEXTURE_RESOLUTION
    ):
        raise ValueError(
            "Cornice texture resolution must be between "
            f"{CORNICE_UV_MIN_TEXTURE_RESOLUTION} and "
            f"{CORNICE_UV_MAX_TEXTURE_RESOLUTION}."
        )
    return resolution


def _collect_source_primitives(scene: trimesh.Scene) -> tuple[_SourcePrimitive, ...]:
    primitives: list[_SourcePrimitive] = []
    for node_name in sorted(scene.graph.nodes_geometry, key=str):
        transform, geometry_name = scene.graph.get(node_name)
        source_mesh = scene.geometry.get(geometry_name)
        if not isinstance(source_mesh, trimesh.Trimesh) or not len(source_mesh.faces):
            continue
        mesh = copy.deepcopy(source_mesh)
        node_transform = np.asarray(transform, dtype=np.float64)
        if node_transform.shape != (4, 4) or not np.all(np.isfinite(node_transform)):
            raise ValueError("The cornice GLB contains an invalid node transform.")
        mesh.apply_transform(node_transform)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if (
            vertices.ndim != 2
            or vertices.shape[1:] != (3,)
            or faces.ndim != 2
            or faces.shape[1:] != (3,)
            or not np.all(np.isfinite(vertices))
            or np.any(faces < 0)
            or np.any(faces >= len(vertices))
        ):
            raise ValueError("The cornice GLB contains invalid triangle geometry.")
        primitives.append(
            _SourcePrimitive(
                node_name=str(node_name),
                mesh=mesh,
            )
        )
    if not primitives:
        raise ValueError("The cornice GLB contains no triangle geometry.")
    return tuple(primitives)


def _primitive_bounds(primitives: tuple[_SourcePrimitive, ...]) -> np.ndarray:
    bounds = np.asarray(
        (
            np.min([primitive.mesh.bounds[0] for primitive in primitives], axis=0),
            np.max([primitive.mesh.bounds[1] for primitive in primitives], axis=0),
        ),
        dtype=np.float64,
    )
    extents = bounds[1] - bounds[0]
    if (
        bounds.shape != (2, 3)
        or not np.all(np.isfinite(bounds))
        or np.any(extents <= _GEOMETRY_EPSILON)
    ):
        raise ValueError("The cornice GLB must have measurable size on every axis.")
    return bounds


# ### Geometry segmentation ###
def _choose_segment_count(bounds: np.ndarray, *, resolution: int) -> int:
    extents = np.asarray(bounds[1] - bounds[0], dtype=np.float64)
    cross_section_extent = min(float(extents[1]), float(extents[2]))
    requested_count = max(
        1,
        int(
            math.ceil(
                float(extents[0])
                / (CORNICE_UV_MAX_STRIP_ASPECT * cross_section_extent)
            )
        ),
    )
    resolution_limit = max(1, resolution // 32)
    return min(
        requested_count,
        CORNICE_UV_MAX_SEGMENT_COUNT,
        resolution_limit,
    )


def _split_primitives_into_segments(
    primitives: tuple[_SourcePrimitive, ...],
    *,
    global_bounds: np.ndarray,
    segment_count: int,
) -> tuple[_UvSubmission, ...]:
    minimum_x = float(global_bounds[0, 0])
    maximum_x = float(global_bounds[1, 0])
    segment_width = (maximum_x - minimum_x) / float(segment_count)
    scene_scale = max(float(np.ptp(global_bounds, axis=0).max()), 1.0)
    weld_tolerance = scene_scale * 1e-9
    area_epsilon = weld_tolerance * weld_tolerance
    submissions: list[_UvSubmission] = []

    for primitive in primitives:
        builders = tuple(
            _SegmentMeshBuilder(
                weld_tolerance=weld_tolerance,
                area_epsilon=area_epsilon,
            )
            for _index in range(segment_count)
        )
        vertices = np.asarray(primitive.mesh.vertices, dtype=np.float64)
        faces = np.asarray(primitive.mesh.faces, dtype=np.int64)
        raw_face_materials = getattr(primitive.mesh.visual, "face_materials", None)
        face_materials = (
            np.zeros(len(faces), dtype=np.int64)
            if raw_face_materials is None
            else np.asarray(raw_face_materials, dtype=np.int64)
        )
        if face_materials.shape != (len(faces),):
            face_materials = np.zeros(len(faces), dtype=np.int64)

        for face_index, face in enumerate(faces):
            triangle = vertices[face]
            for segment_index in _triangle_segment_indices(
                triangle,
                minimum_x=minimum_x,
                segment_width=segment_width,
                segment_count=segment_count,
            ):
                lower_x = minimum_x + segment_index * segment_width
                upper_x = lower_x + segment_width
                clipped = _clip_polygon_to_x_interval(
                    triangle,
                    lower_x=lower_x,
                    upper_x=upper_x,
                )
                for clipped_triangle in _triangulate_polygon(clipped):
                    builders[segment_index].add_triangle(
                        clipped_triangle,
                        face_material=int(face_materials[face_index]),
                    )

        for segment_index, builder in enumerate(builders):
            segment_mesh = builder.build(primitive.mesh)
            if segment_mesh is None:
                continue
            submissions.append(
                _UvSubmission(
                    submission_index=len(submissions),
                    segment_index=segment_index,
                    node_name=primitive.node_name,
                    mesh=segment_mesh,
                )
            )

    if not submissions:
        raise ValueError("The cornice GLB contains no nondegenerate triangle faces.")
    return tuple(submissions)


def _triangle_segment_indices(
    triangle: np.ndarray,
    *,
    minimum_x: float,
    segment_width: float,
    segment_count: int,
) -> range:
    face_minimum = float(np.min(triangle[:, 0]))
    face_maximum = float(np.max(triangle[:, 0]))
    if face_maximum - face_minimum <= _GEOMETRY_EPSILON:
        segment_index = int(math.floor((face_minimum - minimum_x) / segment_width))
        segment_index = min(max(segment_index, 0), segment_count - 1)
        return range(segment_index, segment_index + 1)
    inset = min(
        (face_maximum - face_minimum) * 0.25,
        segment_width * 1e-9,
    )
    first_index = int(
        math.floor((face_minimum + inset - minimum_x) / segment_width)
    )
    last_index = int(
        math.floor((face_maximum - inset - minimum_x) / segment_width)
    )
    first_index = min(max(first_index, 0), segment_count - 1)
    last_index = min(max(last_index, first_index), segment_count - 1)
    return range(first_index, last_index + 1)


def _clip_polygon_to_x_interval(
    triangle: np.ndarray,
    *,
    lower_x: float,
    upper_x: float,
) -> np.ndarray:
    polygon = [np.asarray(point, dtype=np.float64) for point in triangle]
    polygon = _clip_polygon_to_x_plane(
        polygon,
        boundary=lower_x,
        keep_greater=True,
    )
    polygon = _clip_polygon_to_x_plane(
        polygon,
        boundary=upper_x,
        keep_greater=False,
    )
    if len(polygon) < 3:
        return np.empty((0, 3), dtype=np.float64)
    cleaned: list[np.ndarray] = []
    for point in polygon:
        if cleaned and np.allclose(
            point,
            cleaned[-1],
            rtol=0.0,
            atol=_GEOMETRY_EPSILON,
        ):
            continue
        cleaned.append(point)
    if len(cleaned) > 1 and np.allclose(
        cleaned[0],
        cleaned[-1],
        rtol=0.0,
        atol=_GEOMETRY_EPSILON,
    ):
        cleaned.pop()
    return np.asarray(cleaned, dtype=np.float64)


def _clip_polygon_to_x_plane(
    polygon: list[np.ndarray],
    *,
    boundary: float,
    keep_greater: bool,
) -> list[np.ndarray]:
    if not polygon:
        return []

    def inside(point: np.ndarray) -> bool:
        if keep_greater:
            return bool(point[0] >= boundary - _GEOMETRY_EPSILON)
        return bool(point[0] <= boundary + _GEOMETRY_EPSILON)

    output: list[np.ndarray] = []
    previous = polygon[-1]
    previous_inside = inside(previous)
    for current in polygon:
        current_inside = inside(current)
        if current_inside != previous_inside:
            delta = float(current[0] - previous[0])
            if abs(delta) > _GEOMETRY_EPSILON:
                interpolation = (boundary - float(previous[0])) / delta
                intersection = previous + interpolation * (current - previous)
                intersection = np.asarray(intersection, dtype=np.float64)
                intersection[0] = boundary
                output.append(intersection)
        if current_inside:
            output.append(current)
        previous = current
        previous_inside = current_inside
    return output


def _triangulate_polygon(polygon: np.ndarray) -> tuple[np.ndarray, ...]:
    if len(polygon) < 3:
        return ()
    return tuple(
        np.asarray((polygon[0], polygon[index], polygon[index + 1]), dtype=np.float64)
        for index in range(1, len(polygon) - 1)
    )


# ### Xatlas unwrapping ###
def _unwrap_submissions(
    submissions: tuple[_UvSubmission, ...],
    *,
    resolution: int,
) -> tuple[_UvOutput, ...]:
    atlas = xatlas.Atlas()
    normalization_origin, normalization_scale = _xatlas_normalization(submissions)
    for submission in submissions:
        normalized_vertices = (
            np.asarray(submission.mesh.vertices, dtype=np.float64)
            - normalization_origin
        ) * normalization_scale
        atlas.add_mesh(
            np.ascontiguousarray(normalized_vertices, dtype=np.float32),
            np.ascontiguousarray(submission.mesh.faces, dtype=np.uint32),
        )
    chart_options = xatlas.ChartOptions()
    chart_options.fix_winding = False
    chart_options.use_input_mesh_uvs = False
    pack_options = xatlas.PackOptions()
    pack_options.resolution = resolution
    pack_options.padding = 0
    pack_options.bilinear = False
    pack_options.blockAlign = False
    pack_options.bruteForce = False
    pack_options.create_image = False
    pack_options.rotate_charts = False
    pack_options.rotate_charts_to_axis = False
    try:
        atlas.generate(chart_options, pack_options, False)
    except RuntimeError as error:
        raise ValueError("The cornice UV charts could not be unwrapped.") from error
    if int(atlas.atlas_count) != 1:
        raise ValueError("The cornice UV charts require more than one texture atlas.")

    atlas_width = int(atlas.width)
    atlas_height = int(atlas.height)
    if atlas_width <= 0 or atlas_height <= 0:
        raise ValueError("Xatlas returned invalid cornice UV dimensions.")
    raw_outputs: list[_UvOutput] = []
    for submission in submissions:
        vertex_mapping, faces, uvs = atlas[submission.submission_index]
        vertex_mapping = np.asarray(vertex_mapping, dtype=np.int64)
        faces = np.asarray(faces, dtype=np.int64)
        uvs = np.asarray(uvs, dtype=np.float64)
        source_faces = np.asarray(submission.mesh.faces, dtype=np.int64)
        if (
            faces.shape != source_faces.shape
            or vertex_mapping.ndim != 1
            or uvs.shape != (len(vertex_mapping), 2)
            or np.any(faces < 0)
            or np.any(faces >= len(vertex_mapping))
            or np.any(vertex_mapping < 0)
            or np.any(vertex_mapping >= len(submission.mesh.vertices))
            or not np.array_equal(vertex_mapping[faces], source_faces)
        ):
            raise ValueError("Xatlas returned invalid cornice UV topology.")
        raw_outputs.append(
            _UvOutput(
                submission=submission,
                vertex_mapping=vertex_mapping,
                faces=faces,
                pixel_uvs=uvs * np.asarray((atlas_width, atlas_height), dtype=float),
            )
        )
    return _repair_collapsed_uv_faces(tuple(raw_outputs), resolution=resolution)


def _xatlas_normalization(
    submissions: tuple[_UvSubmission, ...],
) -> tuple[np.ndarray, float]:
    """Return one shared transform that keeps float32 xatlas inputs stable."""

    minimum = np.min(
        [np.asarray(item.mesh.bounds[0], dtype=np.float64) for item in submissions],
        axis=0,
    )
    maximum = np.max(
        [np.asarray(item.mesh.bounds[1], dtype=np.float64) for item in submissions],
        axis=0,
    )
    largest_extent = float(np.max(maximum - minimum))
    if not math.isfinite(largest_extent) or largest_extent <= _GEOMETRY_EPSILON:
        raise ValueError("The cornice UV source has no measurable extent.")
    return (minimum + maximum) * 0.5, 1.0 / largest_extent


# ### Collapsed-chart recovery ###
def _repair_collapsed_uv_faces(
    outputs: tuple[_UvOutput, ...],
    *,
    resolution: int,
) -> tuple[_UvOutput, ...]:
    """Restore valid thin faces that xatlas leaves at a zero-area UV."""

    texel_density = _estimate_xatlas_texel_density(outputs, resolution=resolution)
    repaired_outputs: list[_UvOutput] = []
    for output in outputs:
        vertex_mapping = np.asarray(output.vertex_mapping, dtype=np.int64).copy()
        faces = np.asarray(output.faces, dtype=np.int64).copy()
        pixel_uvs = np.asarray(output.pixel_uvs, dtype=np.float64).copy()
        source_vertices = np.asarray(output.submission.mesh.vertices, dtype=np.float64)
        for face_index, face in enumerate(faces):
            triangle_uvs = pixel_uvs[face]
            if _triangle_has_measurable_2d_area(triangle_uvs):
                continue
            source_indices = vertex_mapping[face]
            triangle = source_vertices[source_indices]
            local_uvs = _flatten_triangle_isometrically(
                triangle,
                texel_density=texel_density,
            )
            first_new_vertex = len(vertex_mapping)
            vertex_mapping = np.concatenate((vertex_mapping, source_indices))
            pixel_uvs = np.vstack((pixel_uvs, local_uvs))
            faces[face_index] = np.arange(
                first_new_vertex,
                first_new_vertex + 3,
                dtype=np.int64,
            )
        used_vertices = np.unique(faces.reshape(-1))
        compact_indices = np.full(len(vertex_mapping), -1, dtype=np.int64)
        compact_indices[used_vertices] = np.arange(
            len(used_vertices),
            dtype=np.int64,
        )
        faces = compact_indices[faces]
        vertex_mapping = vertex_mapping[used_vertices]
        pixel_uvs = pixel_uvs[used_vertices]
        repaired_outputs.append(
            _UvOutput(
                submission=output.submission,
                vertex_mapping=vertex_mapping,
                faces=faces,
                pixel_uvs=pixel_uvs,
            )
        )
    return tuple(repaired_outputs)


def _estimate_xatlas_texel_density(
    outputs: tuple[_UvOutput, ...],
    *,
    resolution: int,
) -> float:
    """Estimate xatlas's common pixels-per-world-unit scale from valid faces."""

    density_samples: list[float] = []
    for output in outputs:
        source_vertices = np.asarray(output.submission.mesh.vertices, dtype=np.float64)
        mapped_vertices = source_vertices[output.vertex_mapping]
        for face in output.faces:
            triangle = mapped_vertices[face]
            triangle_uvs = output.pixel_uvs[face]
            world_area = _triangle_doubled_area_3d(triangle)
            uv_area = _triangle_doubled_area_2d(triangle_uvs)
            if (
                math.isfinite(world_area)
                and math.isfinite(uv_area)
                and world_area > np.finfo(np.float64).eps
                and uv_area > np.finfo(np.float64).eps
            ):
                density_samples.append(math.sqrt(uv_area / world_area))
    if density_samples:
        density = float(np.median(np.asarray(density_samples, dtype=np.float64)))
        if math.isfinite(density) and density > 0.0:
            return density

    world_extent = max(
        (
            float(np.max(np.ptp(output.submission.mesh.vertices, axis=0)))
            for output in outputs
        ),
        default=0.0,
    )
    if not math.isfinite(world_extent) or world_extent <= _GEOMETRY_EPSILON:
        raise ValueError("The cornice UV source has no measurable extent.")
    return float(resolution) / world_extent


def _flatten_triangle_isometrically(
    triangle: np.ndarray,
    *,
    texel_density: float,
) -> np.ndarray:
    """Flatten one 3D triangle without changing its edge-length proportions."""

    points = np.asarray(triangle, dtype=np.float64)
    edge_candidates = (
        (0, 1, 2),
        (1, 2, 0),
        (2, 0, 1),
    )
    start_index, end_index, remaining_index = max(
        edge_candidates,
        key=lambda indices: float(
            np.linalg.norm(points[indices[1]] - points[indices[0]])
        ),
    )
    baseline = points[end_index] - points[start_index]
    baseline_length = float(np.linalg.norm(baseline))
    doubled_area = _triangle_doubled_area_3d(points)
    if (
        not math.isfinite(baseline_length)
        or baseline_length <= _GEOMETRY_EPSILON
        or not math.isfinite(doubled_area)
        or doubled_area <= np.finfo(np.float64).eps
    ):
        raise ValueError("A cornice face has no measurable geometric area.")
    baseline_direction = baseline / baseline_length
    remaining_edge = points[remaining_index] - points[start_index]
    projected_x = float(np.dot(remaining_edge, baseline_direction))
    projected_y = doubled_area / baseline_length
    flattened = np.zeros((3, 2), dtype=np.float64)
    flattened[end_index, 0] = baseline_length
    flattened[remaining_index] = (projected_x, projected_y)
    flattened -= np.min(flattened, axis=0)
    flattened *= float(texel_density)
    if not _triangle_has_measurable_2d_area(flattened):
        raise ValueError("A cornice face could not be assigned measurable UVs.")
    return flattened


def _triangle_doubled_area_3d(triangle: np.ndarray) -> float:
    points = np.asarray(triangle, dtype=np.float64)
    return float(
        np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0]))
    )


def _triangle_doubled_area_2d(triangle: np.ndarray) -> float:
    points = np.asarray(triangle, dtype=np.float64)
    first_edge = points[1] - points[0]
    second_edge = points[2] - points[0]
    return abs(
        float(
            first_edge[0] * second_edge[1]
            - first_edge[1] * second_edge[0]
        )
    )


def _triangle_has_measurable_2d_area(triangle: np.ndarray) -> bool:
    points = np.asarray(triangle, dtype=np.float64)
    if points.shape != (3, 2) or not np.all(np.isfinite(points)):
        return False
    doubled_area = _triangle_doubled_area_2d(points)
    coordinate_scale = max(1.0, float(np.max(np.abs(points))))
    tolerance = np.spacing(coordinate_scale) * coordinate_scale * 8.0
    return math.isfinite(doubled_area) and doubled_area > tolerance


# ### Chart extraction ###
def _collect_oriented_charts(
    outputs: tuple[_UvOutput, ...],
) -> tuple[_UvChart, ...]:
    charts: list[_UvChart] = []
    for output_index, output in enumerate(outputs):
        for face_indices in _connected_face_components(output.faces):
            vertex_indices = np.unique(output.faces[face_indices].reshape(-1))
            points = np.asarray(output.pixel_uvs[vertex_indices], dtype=np.float64)
            points -= np.min(points, axis=0)
            extents = np.ptp(points, axis=0)
            extent_scale = max(1.0, float(np.max(np.abs(points))))
            extent_tolerance = np.spacing(extent_scale) * 8.0
            if (
                np.any(extents <= extent_tolerance)
                or not np.all(np.isfinite(extents))
            ):
                raise ValueError("A cornice UV chart has no measurable area.")
            if extents[1] > extents[0]:
                points = np.column_stack((points[:, 1], extents[0] - points[:, 0]))
                points -= np.min(points, axis=0)
                extents = np.ptp(points, axis=0)
            charts.append(
                _UvChart(
                    chart_index=len(charts),
                    output_index=output_index,
                    vertex_indices=vertex_indices,
                    local_uvs=points,
                    width=float(extents[0]),
                    height=float(extents[1]),
                )
            )
    if not charts:
        raise ValueError("Xatlas returned no cornice UV charts.")
    return tuple(charts)


def _connected_face_components(faces: np.ndarray) -> tuple[np.ndarray, ...]:
    parents = np.arange(len(faces), dtype=np.int64)

    def find(face_index: int) -> int:
        while int(parents[face_index]) != face_index:
            parents[face_index] = parents[int(parents[face_index])]
            face_index = int(parents[face_index])
        return face_index

    first_face_by_vertex: dict[int, int] = {}
    for face_index, face in enumerate(faces):
        for raw_vertex_index in face:
            vertex_index = int(raw_vertex_index)
            previous_face = first_face_by_vertex.get(vertex_index)
            if previous_face is None:
                first_face_by_vertex[vertex_index] = face_index
                continue
            first_root = find(face_index)
            second_root = find(previous_face)
            if first_root != second_root:
                parents[second_root] = first_root

    groups: dict[int, list[int]] = {}
    for face_index in range(len(faces)):
        groups.setdefault(find(face_index), []).append(face_index)
    return tuple(
        np.asarray(group, dtype=np.int64)
        for group in sorted(groups.values(), key=lambda values: values[0])
    )


# ### Horizontal-row packing ###
def _pack_charts_into_horizontal_rows(
    outputs: tuple[_UvOutput, ...],
    charts: tuple[_UvChart, ...],
    *,
    texture_resolution: int,
) -> tuple[np.ndarray, ...]:
    requested_gutter = min(
        CORNICE_UV_GUTTER_PIXELS,
        max(1, texture_resolution // 32),
    )
    packed: tuple[float, tuple[_ChartPlacement, ...]] | None = None
    for gutter in _gutter_candidates(requested_gutter):
        packed = _find_largest_row_packing(
            charts,
            texture_resolution=texture_resolution,
            gutter_pixels=float(gutter),
        )
        if packed is not None:
            break
    if packed is None:
        raise ValueError("The cornice UV charts could not fit into one texture.")
    scale, placements = packed

    final_uvs = [
        np.full((len(output.vertex_mapping), 2), np.nan, dtype=np.float64)
        for output in outputs
    ]
    for chart, placement in zip(charts, placements, strict=True):
        packed_points = chart.local_uvs * scale + np.asarray(
            (placement.x, placement.y),
            dtype=np.float64,
        )
        final_uvs[chart.output_index][chart.vertex_indices] = (
            packed_points / float(texture_resolution)
        )
    for output_uvs in final_uvs:
        if (
            not np.all(np.isfinite(output_uvs))
            or np.any(output_uvs < -_UV_EPSILON)
            or np.any(output_uvs > 1.0 + _UV_EPSILON)
        ):
            raise ValueError("The packed cornice UVs leave the texture atlas.")
    clipped_uvs = tuple(np.clip(uvs, 0.0, 1.0) for uvs in final_uvs)
    return _stabilize_uv_faces_for_float32(outputs, clipped_uvs)


def _stabilize_uv_faces_for_float32(
    outputs: tuple[_UvOutput, ...],
    final_uvs: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, ...]:
    """Keep subpixel fallback triangles measurable after GLB float32 storage."""

    stabilized_outputs: list[np.ndarray] = []
    for output, output_uvs in zip(outputs, final_uvs, strict=True):
        stabilized = np.asarray(output_uvs, dtype=np.float32).copy()
        for face in output.faces:
            if _triangle_has_measurable_2d_area(
                np.asarray(stabilized[face], dtype=np.float64)
            ):
                continue
            best_candidate: tuple[float, int, int, np.float32] | None = None
            for vertex_index in (int(value) for value in face):
                for axis in range(2):
                    current = stabilized[vertex_index, axis]
                    for target in (np.float32(0.0), np.float32(1.0)):
                        candidate_value = np.nextafter(current, target)
                        if candidate_value == current:
                            continue
                        candidate_triangle = stabilized[face].copy()
                        local_index = int(np.flatnonzero(face == vertex_index)[0])
                        candidate_triangle[local_index, axis] = candidate_value
                        candidate_area = _triangle_doubled_area_2d(
                            np.asarray(candidate_triangle, dtype=np.float64)
                        )
                        candidate = (
                            candidate_area,
                            vertex_index,
                            axis,
                            candidate_value,
                        )
                        if best_candidate is None or candidate[0] > best_candidate[0]:
                            best_candidate = candidate
            if best_candidate is None or best_candidate[0] <= 0.0:
                raise ValueError("A cornice UV face cannot survive GLB precision.")
            _area, vertex_index, axis, candidate_value = best_candidate
            stabilized[vertex_index, axis] = candidate_value
        stabilized_outputs.append(np.asarray(stabilized, dtype=np.float64))
    return tuple(stabilized_outputs)


def _gutter_candidates(requested_gutter: int) -> tuple[int, ...]:
    candidates = [requested_gutter]
    while candidates[-1] > 1:
        next_value = max(1, candidates[-1] // 2)
        if next_value == candidates[-1]:
            break
        candidates.append(next_value)
    return tuple(candidates)


def _find_largest_row_packing(
    charts: tuple[_UvChart, ...],
    *,
    texture_resolution: int,
    gutter_pixels: float,
) -> tuple[float, tuple[_ChartPlacement, ...]] | None:
    content_size = float(texture_resolution) - 2.0 * gutter_pixels
    if content_size <= 0.0:
        return None
    total_area = sum(chart.width * chart.height for chart in charts)
    maximum_width = max(chart.width for chart in charts)
    maximum_height = max(chart.height for chart in charts)
    upper = min(
        content_size / maximum_width,
        content_size / maximum_height,
        math.sqrt((content_size * content_size) / total_area),
    )
    lower = 0.0
    best: tuple[_ChartPlacement, ...] | None = None
    for _step in range(_PACKING_BINARY_SEARCH_STEPS):
        scale = (lower + upper) * 0.5
        placements = _try_pack_horizontal_rows(
            charts,
            scale=scale,
            texture_resolution=float(texture_resolution),
            gutter_pixels=gutter_pixels,
        )
        if placements is None:
            upper = scale
        else:
            lower = scale
            best = placements
    if best is None or lower <= np.finfo(np.float64).eps:
        return None
    return lower, best


def _try_pack_horizontal_rows(
    charts: tuple[_UvChart, ...],
    *,
    scale: float,
    texture_resolution: float,
    gutter_pixels: float,
) -> tuple[_ChartPlacement, ...] | None:
    ordered_indices = sorted(
        range(len(charts)),
        key=lambda index: (
            -charts[index].height,
            -charts[index].width,
            charts[index].output_index,
            charts[index].chart_index,
        ),
    )
    placements: list[_ChartPlacement | None] = [None] * len(charts)
    cursor_x = gutter_pixels
    cursor_y = gutter_pixels
    row_height = 0.0
    content_limit = texture_resolution - gutter_pixels

    for chart_index in ordered_indices:
        chart = charts[chart_index]
        width = chart.width * scale
        height = chart.height * scale
        if width > content_limit - gutter_pixels + _UV_EPSILON:
            return None
        if cursor_x > gutter_pixels and cursor_x + width > content_limit + _UV_EPSILON:
            cursor_x = gutter_pixels
            cursor_y += row_height + gutter_pixels
            row_height = 0.0
        if cursor_y + height > content_limit + _UV_EPSILON:
            return None
        placements[chart_index] = _ChartPlacement(x=cursor_x, y=cursor_y)
        cursor_x += width + gutter_pixels
        row_height = max(row_height, height)

    if any(placement is None for placement in placements):
        return None
    return tuple(placement for placement in placements if placement is not None)


# ### Output helpers ###
def _build_output_scene(
    outputs: tuple[_UvOutput, ...],
    final_uvs: tuple[np.ndarray, ...],
) -> trimesh.Scene:
    scene = trimesh.Scene()
    occupied_names: set[str] = set()
    for output, uvs in zip(outputs, final_uvs, strict=True):
        source_mesh = output.submission.mesh
        material = copy.deepcopy(getattr(source_mesh.visual, "material", None))
        raw_face_materials = getattr(source_mesh.visual, "face_materials", None)
        face_materials = (
            None
            if raw_face_materials is None
            else np.asarray(raw_face_materials, dtype=np.int64).copy()
        )
        mesh = trimesh.Trimesh(
            vertices=np.asarray(source_mesh.vertices, dtype=np.float64)[
                output.vertex_mapping
            ],
            faces=output.faces,
            visual=TextureVisuals(
                uv=uvs,
                material=material,
                face_materials=face_materials,
            ),
            metadata=copy.deepcopy(source_mesh.metadata),
            process=False,
            validate=False,
        )
        base_name = (
            f"{output.submission.node_name}_uv_strip_"
            f"{output.submission.segment_index + 1:02d}"
        )
        node_name = base_name
        suffix = 2
        while node_name in occupied_names:
            node_name = f"{base_name}_{suffix}"
            suffix += 1
        occupied_names.add(node_name)
        scene.add_geometry(mesh, geom_name=node_name, node_name=node_name)
    if not scene.geometry:
        raise ValueError("The cornice UV layout contains no output geometry.")
    return scene


def _validate_uv_glb(exported: bytes, *, expected_bounds: np.ndarray) -> None:
    model = import_generated_glb(exported)
    actual_bounds = np.asarray(model.mesh.bounds, dtype=np.float64)
    scale = max(float(np.ptp(expected_bounds, axis=0).max()), 1.0)
    if not np.allclose(
        actual_bounds,
        expected_bounds,
        rtol=0.0,
        atol=scale * 1e-6,
    ):
        raise ValueError("The cornice UV layout changed the model bounds.")
    for geometry in model.scene.geometry.values():
        if not isinstance(geometry, trimesh.Trimesh) or not len(geometry.faces):
            continue
        uvs = np.asarray(getattr(geometry.visual, "uv", None), dtype=np.float64)
        if (
            uvs.shape != (len(geometry.vertices), 2)
            or not np.all(np.isfinite(uvs))
            or np.any(uvs < -_UV_EPSILON)
            or np.any(uvs > 1.0 + _UV_EPSILON)
        ):
            raise ValueError("The serialized cornice GLB contains invalid UVs.")
