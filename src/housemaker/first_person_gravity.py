"""Cached vertical ground queries in canonical Z-up scene coordinates."""

# ### Imports ###
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

# ### Constants ###
DEFAULT_GROUND_CELL_SIZE = 1.0
DEFAULT_MAX_CELLS_PER_TRIANGLE = 64
GROUND_BARYCENTRIC_EPSILON = 1e-9
GROUND_HEIGHT_EPSILON = 1e-9
GROUND_PROJECTED_AREA_EPSILON = 1e-12


# ### Ground support index ###
class FirstPersonGroundIndex:
    """Find support heights without per-frame full-mesh ray casts.

    Input is a trimesh-like object exposing ``triangles``, or an array shaped
    ``(triangle_count, 3, 3)``. All coordinates and height limits are canonical
    Z-up values. Geometry is copied once; changing its owner requires a new
    index. Winding is irrelevant, so flipped floors still provide support.
    """

    def __init__(
        self,
        mesh_or_triangles: object,
        *,
        cell_size: float = DEFAULT_GROUND_CELL_SIZE,
        max_cells_per_triangle: int = DEFAULT_MAX_CELLS_PER_TRIANGLE,
    ) -> None:
        self.cell_size = float(cell_size)
        if not math.isfinite(self.cell_size) or self.cell_size <= 0.0:
            raise ValueError("Ground-index cell size must be finite and positive.")
        if (
            isinstance(max_cells_per_triangle, bool)
            or not isinstance(max_cells_per_triangle, int)
            or max_cells_per_triangle < 1
        ):
            raise ValueError("The maximum cells per triangle must be a positive integer.")

        triangles = np.asarray(
            getattr(mesh_or_triangles, "triangles", mesh_or_triangles),
            dtype=np.float64,
        )
        if triangles.size == 0:
            triangles = np.empty((0, 3, 3), dtype=np.float64)
        if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
            raise ValueError("Ground geometry must contain triangles shaped (N, 3, 3).")
        triangles = triangles[np.isfinite(triangles).all(axis=(1, 2))].copy()
        edge_a = triangles[:, 1] - triangles[:, 0]
        edge_b = triangles[:, 2] - triangles[:, 0]
        determinant = edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]
        projected_scale = np.maximum(
            np.linalg.norm(edge_a[:, :2], axis=1)
            * np.linalg.norm(edge_b[:, :2], axis=1),
            np.finfo(np.float64).tiny,
        )
        usable = np.abs(determinant) > GROUND_PROJECTED_AREA_EPSILON * projected_scale
        triangles, edge_a, edge_b = triangles[usable], edge_a[usable], edge_b[usable]
        self._origins = triangles[:, 0].copy()
        self._edge_a = edge_a
        self._edge_b = edge_b
        self._inverse_determinants = 1.0 / determinant[usable]
        self._minimum_xy = triangles[:, :, :2].min(axis=1)
        self._maximum_xy = triangles[:, :, :2].max(axis=1)

        cells: dict[tuple[int, int], list[int]] = defaultdict(list)
        large_triangles: list[int] = []
        for triangle_index, (minimum, maximum) in enumerate(
            zip(self._minimum_xy, self._maximum_xy, strict=True)
        ):
            left, bottom = self._cell_coordinates(*minimum)
            right, top = self._cell_coordinates(*maximum)
            if (right - left + 1) * (top - bottom + 1) > max_cells_per_triangle:
                large_triangles.append(triangle_index)
                continue
            for cell_x in range(left, right + 1):
                for cell_y in range(bottom, top + 1):
                    cells[(cell_x, cell_y)].append(triangle_index)
        self._cells = {
            cell: np.asarray(indices, dtype=np.intp)
            for cell, indices in cells.items()
        }
        self._large_triangles = np.asarray(large_triangles, dtype=np.intp)
        for values in (
            self._origins, self._edge_a, self._edge_b,
            self._inverse_determinants, self._minimum_xy, self._maximum_xy,
            self._large_triangles, *self._cells.values(),
        ):
            values.setflags(write=False)

    # ### Index statistics ###
    @property
    def triangle_count(self) -> int:
        """Number of finite triangles with nonzero XY support area."""

        return len(self._origins)

    @property
    def cell_count(self) -> int:
        return len(self._cells)

    @property
    def large_triangle_count(self) -> int:
        """Triangles kept outside the grid to bound index memory usage."""

        return len(self._large_triangles)

    # ### Vertical support queries ###
    def ground_height(
        self,
        x: float,
        y: float,
        max_height: float,
        min_height: float = -math.inf,
    ) -> float | None:
        """Return the highest supporting triangle within inclusive height limits.

        Use the current foot height plus the allowed step-up as ``max_height``
        to exclude ceilings and upper floors. An omitted lower limit permits
        finding support several levels below the camera.
        """

        x, y = float(x), float(y)
        maximum, minimum = float(max_height), float(min_height)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Ground query X and Y must be finite.")
        if math.isnan(maximum) or math.isnan(minimum):
            raise ValueError("Ground query height limits cannot be NaN.")
        if minimum > maximum or not self.triangle_count:
            return None
        local_indices = self._cells.get(self._cell_coordinates(x, y))
        if local_indices is None or not len(local_indices):
            indices = self._large_triangles
        elif self.large_triangle_count:
            indices = np.concatenate((local_indices, self._large_triangles))
        else:
            indices = local_indices
        if not len(indices):
            return None

        within_bounds = (
            (x >= self._minimum_xy[indices, 0])
            & (x <= self._maximum_xy[indices, 0])
            & (y >= self._minimum_xy[indices, 1])
            & (y <= self._maximum_xy[indices, 1])
        )
        indices = indices[within_bounds]
        if not len(indices):
            return None
        offset_x = x - self._origins[indices, 0]
        offset_y = y - self._origins[indices, 1]
        edge_a, edge_b = self._edge_a[indices], self._edge_b[indices]
        inverse = self._inverse_determinants[indices]
        u = (offset_x * edge_b[:, 1] - offset_y * edge_b[:, 0]) * inverse
        v = (edge_a[:, 0] * offset_y - edge_a[:, 1] * offset_x) * inverse
        heights = self._origins[indices, 2] + u * edge_a[:, 2] + v * edge_b[:, 2]
        supported = (
            (u >= -GROUND_BARYCENTRIC_EPSILON)
            & (v >= -GROUND_BARYCENTRIC_EPSILON)
            & (u + v <= 1.0 + GROUND_BARYCENTRIC_EPSILON)
            & (heights <= maximum + GROUND_HEIGHT_EPSILON)
            & (heights >= minimum - GROUND_HEIGHT_EPSILON)
        )
        return float(np.max(heights[supported])) if np.any(supported) else None

    def _cell_coordinates(self, x: float, y: float) -> tuple[int, int]:
        return math.floor(x / self.cell_size), math.floor(y / self.cell_size)
