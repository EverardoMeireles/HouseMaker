"""Conservative, volume-based visibility baking in physical Z-up coordinates.

Only an opaque convex patch which blocks *every* segment between a camera
cell and a render-cluster bounding box can remove that cluster. Missing
occluders, ambiguous geometry and viewpoints outside the baked cells stay
visible. Callers must exclude glass and anything movable from occluder input.
"""

# ### Imports ###
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import shapely
from shapely.geometry import Polygon, box

# ### Constants and callback types ###
PVS_GEOMETRY_EPSILON = 1e-8
DEFAULT_PVS_CELL_SIZE = 2.0
DEFAULT_PVS_EYE_HEIGHT = 1.7
DEFAULT_PVS_EYE_MARGIN = 0.35
DEFAULT_PVS_MAX_CELLS = 4096
DEFAULT_PVS_MAX_PLANES = 4096
DEFAULT_PVS_MAX_PATCHES = 8192
PVS_MAX_CONVEX_MERGE_TESTS = 2048
PVS_PROOF_BATCH_SIZE = 128
PVS_MAX_PROOF_BATCH_VALUES = 1_000_000
PVS_CELL_LIMIT_ERROR = (
    "PVS camera coverage exceeds the maximum cell count; use a larger cell size."
)
ProgressCallback = Callable[[int, int], None]
CancellationCallback = Callable[[], bool]


# ### Immutable bake inputs and results ###
class PvsBakeCancelled(RuntimeError):
    """The caller cancelled a bake; no partial PVS should be installed."""


@dataclass(frozen=True)
class PvsAabb:
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]
    _corners: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        minimum = _point(self.minimum, "AABB minimum")
        maximum = _point(self.maximum, "AABB maximum")
        if any(low > high for low, high in zip(minimum, maximum, strict=True)):
            raise ValueError("PVS bounds must have minimum <= maximum on every axis.")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        corners = np.asarray(tuple(product(*zip(minimum, maximum, strict=True))))
        corners.setflags(write=False)
        object.__setattr__(self, "_corners", corners)

    @property
    def corners(self) -> np.ndarray:
        return self._corners

    def contains(self, position: Sequence[float]) -> bool:
        point = _point(position, "Camera position")
        return all(low <= value <= high for low, value, high in zip(
            self.minimum, point, self.maximum, strict=True
        ))


@dataclass(frozen=True)
class PvsCameraCell:
    cell_id: str
    bounds: PvsAabb

    def __post_init__(self) -> None:
        _validate_named_bounds(self.cell_id, self.bounds, "Camera cell")


@dataclass(frozen=True)
class PvsRenderCluster:
    cluster_id: str
    bounds: PvsAabb

    def __post_init__(self) -> None:
        _validate_named_bounds(self.cluster_id, self.bounds, "Render cluster")


@dataclass(frozen=True)
class PvsOccluderPatch:
    """One strictly convex or convex planar, opaque polygon, either winding."""

    vertices: tuple[tuple[float, float, float], ...]
    front_side: int = 0
    _origin: np.ndarray = field(init=False, repr=False, compare=False)
    _normal: np.ndarray = field(init=False, repr=False, compare=False)
    _u: np.ndarray = field(init=False, repr=False, compare=False)
    _v: np.ndarray = field(init=False, repr=False, compare=False)
    _polygon_xy: np.ndarray = field(init=False, repr=False, compare=False)
    _edge_normals: np.ndarray = field(init=False, repr=False, compare=False)
    _edge_offsets: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.front_side, bool)
            or not isinstance(self.front_side, (int, np.integer))
            or self.front_side not in (-1, 0, 1)
        ):
            raise ValueError("PVS front side must be -1, 0 (both sides), or 1.")
        supplied = tuple(_point(vertex, "Occluder vertex") for vertex in self.vertices)
        vertices = tuple(vertex for index, vertex in enumerate(supplied)
                         if index == 0 or vertex != supplied[index - 1])
        if len(vertices) > 3 and vertices[0] == vertices[-1]:
            vertices = vertices[:-1]
        if len(vertices) < 3:
            raise ValueError("PVS occluders need at least three vertices.")
        points = np.asarray(vertices, dtype=float)
        frame = _plane_frame(points)
        if frame is None:
            raise ValueError("PVS occluders must be nondegenerate planar polygons.")
        origin, normal, u, v = frame
        projected = np.column_stack(((points - origin) @ u, (points - origin) @ v))
        polygon = Polygon(projected)
        if (
            not polygon.is_valid or polygon.area <= PVS_GEOMETRY_EPSILON**2
            or not polygon.equals(polygon.convex_hull)
        ):
            raise ValueError("PVS occluder polygons must be valid and convex.")
        # The signed-edge test below expects counterclockwise projected points.
        signed_area = np.sum(
            projected[:, 0] * np.roll(projected[:, 1], -1)
            - projected[:, 1] * np.roll(projected[:, 0], -1)
        )
        if signed_area < 0.0:
            projected = projected[::-1].copy()
        edges = np.roll(projected, -1, axis=0) - projected
        edge_normals = np.column_stack((-edges[:, 1], edges[:, 0]))
        edge_normals /= np.linalg.norm(edges, axis=1)[:, None]
        edge_offsets = np.sum(edge_normals * projected, axis=1)
        object.__setattr__(self, "vertices", vertices)
        for name, value in (
            ("_origin", origin), ("_normal", normal), ("_u", u),
            ("_v", v), ("_polygon_xy", projected),
            ("_edge_normals", edge_normals), ("_edge_offsets", edge_offsets),
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class PvsBakeResult:
    cells: tuple[PvsCameraCell, ...]
    clusters: tuple[PvsRenderCluster, ...]
    visible_cluster_ids_by_cell: tuple[tuple[str, ...], ...]
    occluder_count: int

    @property
    def hidden_pair_count(self) -> int:
        return len(self.cells) * len(self.clusters) - sum(
            len(visible) for visible in self.visible_cluster_ids_by_cell
        )

    def visible_cluster_ids_at(self, position: Sequence[float]) -> tuple[str, ...]:
        """Union overlapping cells; outside coverage always returns all clusters."""

        point = _point(position, "Camera position")
        matched = False
        visible: set[str] = set()
        for cell, cell_visible in zip(
            self.cells, self.visible_cluster_ids_by_cell, strict=True
        ):
            if cell.bounds.contains(point):
                matched = True
                visible.update(cell_visible)
        return tuple(
            cluster.cluster_id for cluster in self.clusters
            if not matched or cluster.cluster_id in visible
        )


# ### Input validation and planar frames ###
def _point(value: Sequence[float], name: str) -> tuple[float, float, float]:
    try:
        values = tuple(float(component) for component in value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} requires three finite coordinates.") from error
    if len(values) != 3 or not all(math.isfinite(component) for component in values):
        raise ValueError(f"{name} requires three finite coordinates.")
    return values


def _positive(value: float, name: str, *, allow_zero: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (not allow_zero and result == 0.0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}.")
    return result


def _limit(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _validate_named_bounds(identifier: str, bounds: PvsAabb, name: str) -> None:
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError(f"{name} requires a nonempty identifier.")
    if not isinstance(bounds, PvsAabb):
        raise TypeError(f"{name} bounds must be PvsAabb.")


def _triangles(mesh_or_triangles: object) -> np.ndarray:
    values = np.asarray(getattr(mesh_or_triangles, "triangles", mesh_or_triangles), dtype=float)
    if not values.size:
        return np.empty((0, 3, 3), dtype=float)
    if values.ndim != 3 or values.shape[1:] != (3, 3):
        raise ValueError("PVS geometry must contain triangles shaped (N, 3, 3).")
    if not np.all(np.isfinite(values)):
        raise ValueError("PVS geometry must contain only finite coordinates.")
    return values


def _plane_frame(points: np.ndarray) -> tuple[np.ndarray, ...] | None:
    origin = points[0].copy()
    normal = None
    for index in range(1, len(points) - 1):
        candidate = np.cross(points[index] - origin, points[index + 1] - origin)
        length = np.linalg.norm(candidate)
        if length > PVS_GEOMETRY_EPSILON**2:
            normal = candidate / length
            break
    if normal is None:
        return None
    if normal[np.argmax(np.abs(normal))] < 0.0:
        normal = -normal
    if np.max(np.abs((points - origin) @ normal)) > PVS_GEOMETRY_EPSILON:
        return None
    axis = np.eye(3)[np.argmin(np.abs(normal))]
    u = np.cross(normal, axis)
    u /= np.linalg.norm(u)
    return origin, normal, u, np.cross(normal, u)


def _check_cancelled(is_cancelled: CancellationCallback | None) -> None:
    if is_cancelled is not None and is_cancelled():
        raise PvsBakeCancelled("Visibility bake cancelled.")


# ### Whole-volume occlusion proof ###
def occluder_hides_aabb(
    camera_bounds: PvsAabb,
    target_bounds: PvsAabb,
    occluder: PvsOccluderPatch,
) -> bool:
    """Prove that every camera-to-target segment crosses an opaque patch.

    For boxes on strictly opposite sides of a plane, the plane section of
    their convex hull is the convex hull of the 8x8 corner-segment crossings.
    If all those crossings lie strictly inside a convex opaque patch, so do
    *all* camera-to-target crossings. Touching/overlapping cases fail open.
    """

    cameras, targets = camera_bounds.corners, target_bounds.corners
    camera_distances = (cameras - occluder._origin) @ occluder._normal
    target_distances = (targets - occluder._origin) @ occluder._normal
    epsilon = PVS_GEOMETRY_EPSILON
    if (
        occluder.front_side == 1 and np.min(camera_distances) <= epsilon * 2
        or occluder.front_side == -1 and np.max(camera_distances) >= -epsilon * 2
    ):
        return False
    opposite = (
        np.min(camera_distances) > epsilon * 2 and np.max(target_distances) < -epsilon * 2
        or np.max(camera_distances) < -epsilon * 2 and np.min(target_distances) > epsilon * 2
    )
    if not opposite:
        return False
    distances = camera_distances[:, None] - target_distances[None, :]
    ratios = camera_distances[:, None] / distances
    directions = targets[None, :, :] - cameras[:, None, :]
    crossings = cameras[:, None, :] + ratios[:, :, None] * directions
    crossings = crossings.reshape((-1, 3)) - occluder._origin
    projected = np.column_stack((crossings @ occluder._u, crossings @ occluder._v))
    polygon = occluder._polygon_xy
    edges = np.roll(polygon, -1, axis=0) - polygon
    relative = projected[:, None, :] - polygon[None, :, :]
    signed_distances = (
        edges[None, :, 0] * relative[:, :, 1]
        - edges[None, :, 1] * relative[:, :, 0]
    ) / np.linalg.norm(edges, axis=1)[None, :]
    # Very grazing rays amplify even tiny plane roundoff. Erode the patch by
    # the complete projected ray displacement across a +/- epsilon plane band,
    # rather than accepting a fixed tiny edge margin at arbitrarily bad angles.
    projected_directions = np.stack(
        (directions @ occluder._u, directions @ occluder._v), axis=-1,
    )
    margins = epsilon * (1.0 + np.linalg.norm(
        projected_directions, axis=-1
    ) / np.abs(distances))
    return bool(np.all(signed_distances > margins.reshape((-1, 1))))


def _occluder_hides_bounds_batch(
    camera_bounds: PvsAabb,
    target_corners: np.ndarray,
    occluder: PvsOccluderPatch,
) -> np.ndarray:
    """Evaluate the same whole-volume proof for a bounded batch of clusters.

    Edge values at a segment's plane intersection are interpolated rationally.
    Multiplying by the positive denominator removes division, allowing all
    target boxes and their 64 segments to be evaluated in one NumPy batch.
    An additional roundoff margin only rejects occlusion; it cannot add it.
    """

    hidden = np.zeros(len(target_corners), dtype=bool)
    if not len(target_corners):
        return hidden
    cameras = camera_bounds.corners - occluder._origin
    targets = target_corners - occluder._origin
    camera_distances = cameras @ occluder._normal
    target_distances = targets @ occluder._normal
    epsilon = PVS_GEOMETRY_EPSILON
    camera_positive = np.min(camera_distances) > epsilon * 2
    camera_negative = np.max(camera_distances) < -epsilon * 2
    if occluder.front_side == 1 and not camera_positive:
        return hidden
    if occluder.front_side == -1 and not camera_negative:
        return hidden
    valid = (
        camera_positive & (np.max(target_distances, axis=1) < -epsilon * 2)
        | camera_negative & (np.min(target_distances, axis=1) > epsilon * 2)
    )
    if not np.any(valid):
        return hidden
    targets, target_distances = targets[valid], target_distances[valid]
    camera_xy = np.column_stack((cameras @ occluder._u, cameras @ occluder._v))
    target_xy = np.stack((targets @ occluder._u, targets @ occluder._v), axis=-1)
    camera_edges = camera_xy @ occluder._edge_normals.T - occluder._edge_offsets
    target_edges = target_xy @ occluder._edge_normals.T - occluder._edge_offsets
    denominators = camera_distances[None, :, None] - target_distances[:, None, :]
    first = camera_distances[None, :, None, None] * target_edges[:, None, :, :]
    second = target_distances[:, None, :, None] * camera_edges[None, :, None, :]
    numerators = (first - second) * np.sign(denominators)[:, :, :, None]
    tangential_lengths = np.linalg.norm(
        target_xy[:, None, :, :] - camera_xy[None, :, None, :], axis=-1,
    )
    margins = epsilon * (np.abs(denominators) + tangential_lengths)
    roundoff = np.finfo(float).eps * 16 * (np.abs(first) + np.abs(second))
    hidden[valid] = np.all(
        numerators > margins[:, :, :, None] + roundoff, axis=(1, 2, 3),
    )
    return hidden


# ### Opaque structural triangle patch construction ###
def build_occluder_patches(
    opaque_triangle_batches: Iterable[object],
    *,
    max_planes: int = DEFAULT_PVS_MAX_PLANES,
    max_patches: int = DEFAULT_PVS_MAX_PATCHES,
    one_sided: bool = False,
    progress: ProgressCallback | None = None,
    is_cancelled: CancellationCallback | None = None,
) -> tuple[PvsOccluderPatch, ...]:
    """Union coplanar triangles without filling door, window or stair holes.

    Batches are explicitly trusted opaque structural geometry, not an entire
    scene containing glass, generated objects or openable doors. Bounds on
    work discard excess occluders only, which reduces culling, never visibility.
    ``one_sided`` retains source triangle winding. Supply reversed triangles
    too only when their actual exported material is rendered double-sided.
    """

    _limit(max_planes, "Maximum PVS planes")
    _limit(max_patches, "Maximum PVS patches")
    if not isinstance(one_sided, bool):
        raise TypeError("PVS one-sided occlusion must be boolean.")
    batches = tuple(_triangles(batch) for batch in opaque_triangle_batches)
    total = sum(len(batch) for batch in batches)
    # Quantization only finds candidates: every triangle must independently
    # pass the actual group's plane test before it can join the union.
    groups: dict[
        tuple[int, ...], list[tuple[tuple[np.ndarray, ...], list[Polygon], int]]
    ] = {}
    completed = 0
    plane_count = 0
    if progress is not None:
        progress(0, total)
    for batch in batches:
        for triangle in batch:
            _check_cancelled(is_cancelled)
            completed += 1
            frame = _plane_frame(triangle)
            if frame is not None:
                origin, normal, u, v = frame
                offset = float(origin @ normal)
                winding = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
                front_side = (1 if float(winding @ normal) > 0.0 else -1) if one_sided else 0
                key = (*tuple(round(float(value) / PVS_GEOMETRY_EPSILON)
                              for value in (*normal, offset)), front_side)
                candidates = groups.get(key)
                if candidates is None:
                    if plane_count >= max_planes:
                        continue
                    candidates = []
                    groups[key] = candidates
                # A near-parallel patch is not the same opaque plane. Joining
                # sheets at slightly different depths could seal a real crack.
                # Refuse approximate membership; a smaller patch is safe.
                group = next((candidate for candidate in candidates if np.all(
                    (triangle - candidate[0][0]) @ candidate[0][1] == 0.0
                )), None)
                if group is None and plane_count < max_planes:
                    group = (frame, [], front_side)
                    candidates.append(group)
                    plane_count += 1
                if group is not None:
                    group_origin, _normal, group_u, group_v = group[0]
                    projected = np.column_stack((
                        (triangle - group_origin) @ group_u,
                        (triangle - group_origin) @ group_v,
                    ))
                    polygon = Polygon(projected)
                    if polygon.is_valid and polygon.area > PVS_GEOMETRY_EPSILON**2:
                        group[1].append(polygon)
            if progress is not None and (completed % 128 == 0 or completed == total):
                progress(completed, total)
    if progress is not None and completed == total:
        progress(total, total)
    result: list[PvsOccluderPatch] = []
    for candidates in groups.values():
        for frame, polygons, front_side in candidates:
            _check_cancelled(is_cancelled)
            if not polygons:
                continue
            try:
                union = shapely.union_all(polygons)
                convex_parts = _convex_parts(union, is_cancelled)
            except shapely.errors.GEOSException:
                # A failed planar union is never evidence that something is hidden.
                continue
            origin, _normal, u, v = frame
            for polygon in convex_parts:
                if len(result) >= max_patches:
                    return tuple(result)
                coordinates = np.asarray(polygon.exterior.coords[:-1])
                vertices = origin + coordinates[:, :1] * u + coordinates[:, 1:] * v
                try:
                    patch_frame = _plane_frame(vertices)
                    if patch_frame is None:
                        continue
                    # Canonical normals can change sign at a tie after planar
                    # reconstruction. Preserve the original front in world space.
                    relative = 1 if float(patch_frame[1] @ _normal) >= 0.0 else -1
                    result.append(PvsOccluderPatch(
                        tuple(map(tuple, vertices)), front_side=front_side * relative,
                    ))
                except ValueError:
                    continue
    _check_cancelled(is_cancelled)
    return tuple(result)


def _convex_parts(
    geometry: object, is_cancelled: CancellationCallback | None,
) -> list[Polygon]:
    parts: list[Polygon] = []
    for polygon in shapely.get_parts(geometry):
        _check_cancelled(is_cancelled)
        if not isinstance(polygon, Polygon) or polygon.area <= PVS_GEOMETRY_EPSILON**2:
            continue
        if not polygon.interiors and polygon.equals(polygon.convex_hull):
            parts.append(polygon)
            continue
        triangulation = shapely.constrained_delaunay_triangles(polygon)
        parts.extend(triangle for triangle in shapely.get_parts(triangulation)
                     if isinstance(triangle, Polygon) and polygon.covers(triangle))
    # Joining adjacent triangles into convex patches improves useful culling.
    # Convex-hull replacement is allowed ONLY when geometrically equal: never
    # bridge an opening, notch or hole just because its area is small.
    tests = 0
    index = 0
    while index < len(parts) and tests < PVS_MAX_CONVEX_MERGE_TESTS:
        _check_cancelled(is_cancelled)
        other_index = index + 1
        while other_index < len(parts) and tests < PVS_MAX_CONVEX_MERGE_TESTS:
            other = parts[other_index]
            tests += 1
            if parts[index].intersects(other):
                joined = parts[index].union(other)
                if (
                    isinstance(joined, Polygon) and not joined.interiors
                    and joined.equals(joined.convex_hull)
                ):
                    parts[index] = joined
                    parts.pop(other_index)
                    other_index = index + 1
                    continue
            other_index += 1
        index += 1
    return parts


# ### Volumetric camera-cell construction ###
def build_camera_cells(
    walkable_triangles: object,
    *,
    cell_size: float = DEFAULT_PVS_CELL_SIZE,
    eye_height: float = DEFAULT_PVS_EYE_HEIGHT,
    eye_margin: float = DEFAULT_PVS_EYE_MARGIN,
    max_cells: int = DEFAULT_PVS_MAX_CELLS,
    progress: ProgressCallback | None = None,
    is_cancelled: CancellationCallback | None = None,
) -> tuple[PvsCameraCell, ...]:
    """Cover floor/stair triangles with XY tiles and bounded eye-height bands.

    Tiles may extend beyond actual walkable polygons, deliberately making
    culling more conservative. Separate vertical bands preserve multiple
    levels; cameras outside all bands fail open at query time. Exceeding the
    explicit cell limit aborts instead of silently omitting coverage.
    """

    size = _positive(cell_size, "PVS cell size")
    height = _positive(eye_height, "PVS eye height")
    margin = _positive(eye_margin, "PVS eye margin", allow_zero=True)
    _limit(max_cells, "Maximum PVS camera cells")
    triangles = _triangles(walkable_triangles)
    bands: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
    if progress is not None:
        progress(0, len(triangles))
    for triangle_index, triangle in enumerate(triangles):
        _check_cancelled(is_cancelled)
        xy = triangle[:, :2]
        matrix = np.column_stack((xy[1] - xy[0], xy[2] - xy[0]))
        determinant = float(np.linalg.det(matrix))
        if abs(determinant) <= PVS_GEOMETRY_EPSILON**2:
            continue
        polygon = Polygon(xy)
        limits = tuple(math.floor(float(value) / size)
                       for value in (*np.min(xy, axis=0), *np.max(xy, axis=0)))
        left, bottom, right, top = limits
        # One vast triangle must not allocate millions of grid cells first.
        if (int(right) - int(left) + 1) * (int(top) - int(bottom) + 1) > max_cells * 4:
            raise ValueError(PVS_CELL_LIMIT_ERROR)
        inverse = np.linalg.inv(matrix)
        for cell_x in range(int(left), int(right) + 1):
            for cell_y in range(int(bottom), int(top) + 1):
                _check_cancelled(is_cancelled)
                tile = box(cell_x * size, cell_y * size,
                           (cell_x + 1) * size, (cell_y + 1) * size)
                clipped = polygon.intersection(tile)
                if clipped.is_empty or clipped.area <= PVS_GEOMETRY_EPSILON**2:
                    continue
                coordinates = shapely.get_coordinates(clipped)
                weights = (coordinates - xy[0]) @ inverse.T
                z_values = triangle[0, 2] + weights @ (
                    triangle[1:, 2] - triangle[0, 2]
                )
                key = (cell_x, cell_y)
                bands[key].append((float(np.min(z_values)) + height - margin,
                                   float(np.max(z_values)) + height + margin))
                if len(bands) > max_cells:
                    raise ValueError(PVS_CELL_LIMIT_ERROR)
        if progress is not None:
            progress(triangle_index + 1, len(triangles))
    result = []
    for (cell_x, cell_y), intervals in sorted(bands.items()):
        merged: list[list[float]] = []
        for low, high in sorted(intervals):
            if merged and low <= merged[-1][1] + PVS_GEOMETRY_EPSILON:
                merged[-1][1] = max(merged[-1][1], high)
            else:
                merged.append([low, high])
        for band_index, (low, high) in enumerate(merged):
            if len(result) >= max_cells:
                raise ValueError(PVS_CELL_LIMIT_ERROR)
            result.append(PvsCameraCell(
                f"cell:{cell_x}:{cell_y}:{band_index}",
                PvsAabb((cell_x * size, cell_y * size, low),
                        ((cell_x + 1) * size, (cell_y + 1) * size, high)),
            ))
    _check_cancelled(is_cancelled)
    if progress is not None:
        progress(len(triangles), len(triangles))
    return tuple(result)


# ### Background-friendly PVS bake ###
def bake_pvs(
    cells: Sequence[PvsCameraCell],
    clusters: Sequence[PvsRenderCluster],
    occluders: Sequence[PvsOccluderPatch],
    *,
    progress: ProgressCallback | None = None,
    is_cancelled: CancellationCallback | None = None,
) -> PvsBakeResult:
    """Keep a cluster unless one patch proves complete cell-volume occlusion."""

    cells, clusters, occluders = tuple(cells), tuple(clusters), tuple(occluders)
    for values, expected, attribute in (
        (cells, PvsCameraCell, "cell_id"),
        (clusters, PvsRenderCluster, "cluster_id"),
    ):
        if any(not isinstance(value, expected) for value in values):
            raise TypeError(f"PVS inputs must contain {expected.__name__} values.")
        ids = tuple(getattr(value, attribute) for value in values)
        if len(set(ids)) != len(ids):
            raise ValueError("PVS camera-cell and cluster identifiers must be unique.")
    if any(not isinstance(patch, PvsOccluderPatch) for patch in occluders):
        raise TypeError("PVS occluders must contain PvsOccluderPatch values.")
    total = len(cells) * len(clusters)
    visibility = []
    if progress is not None:
        progress(0, total)
    minimums = np.asarray([cluster.bounds.minimum for cluster in clusters]).reshape((-1, 3))
    maximums = np.asarray([cluster.bounds.maximum for cluster in clusters]).reshape((-1, 3))
    centers = minimums * 0.5 + maximums * 0.5
    extents = maximums * 0.5 - minimums * 0.5
    corners = np.asarray([cluster.bounds.corners for cluster in clusters]).reshape((-1, 8, 3))
    for cell_index, cell in enumerate(cells):
        visible = np.ones(len(clusters), dtype=bool)
        # Reject same-side pairs in vectorized batches before the 64-crossing
        # proof. This is a broad phase only: it never removes a visible cluster.
        for patch in occluders:
            _check_cancelled(is_cancelled)
            if not np.any(visible):
                break
            camera_distances = (cell.bounds.corners - patch._origin) @ patch._normal
            minimum = float(np.min(camera_distances))
            maximum = float(np.max(camera_distances))
            if (
                patch.front_side == 1 and minimum <= PVS_GEOMETRY_EPSILON * 2
                or patch.front_side == -1 and maximum >= -PVS_GEOMETRY_EPSILON * 2
            ):
                continue
            distances = (centers - patch._origin) @ patch._normal
            radii = extents @ np.abs(patch._normal)
            if minimum > PVS_GEOMETRY_EPSILON * 2:
                candidates = visible & (distances + radii < -PVS_GEOMETRY_EPSILON * 2)
            elif maximum < -PVS_GEOMETRY_EPSILON * 2:
                candidates = visible & (distances - radii > PVS_GEOMETRY_EPSILON * 2)
            else:
                continue
            indices = np.flatnonzero(candidates)
            batch_size = max(1, min(
                PVS_PROOF_BATCH_SIZE,
                PVS_MAX_PROOF_BATCH_VALUES // (64 * len(patch._edge_normals)),
            ))
            for start in range(0, len(indices), batch_size):
                _check_cancelled(is_cancelled)
                batch = indices[start:start + batch_size]
                hidden = _occluder_hides_bounds_batch(cell.bounds, corners[batch], patch)
                visible[batch[hidden]] = False
        visibility.append(tuple(cluster.cluster_id for cluster, is_visible in zip(
            clusters, visible, strict=True
        ) if is_visible))
        if progress is not None:
            progress((cell_index + 1) * len(clusters), total)
    _check_cancelled(is_cancelled)
    return PvsBakeResult(cells, clusters, tuple(visibility), len(occluders))
