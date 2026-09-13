# ### Imports ###
from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np
import trimesh

from housemaker.glb import remove_covered_surface_faces


# ### Test fixtures ###
@dataclass(frozen=True)
class _SurfaceStub:
    mesh: trimesh.Trimesh
    surface_type: str = "floor"
    room_index: int | None = 0


def _build_square_with_distant_triangle() -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=np.asarray(
            (
                (0.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                (2.0, 2.0, 0.0),
                (0.0, 2.0, 0.0),
                (3.0, 0.0, 0.0),
                (4.0, 0.0, 0.0),
                (3.0, 1.0, 0.0),
            ),
            dtype=float,
        ),
        faces=np.asarray(
            ((0, 1, 2), (0, 2, 3), (4, 5, 6)),
            dtype=np.int64,
        ),
        process=False,
    )


def _build_surface(
    vertices: tuple[tuple[float, float, float], ...],
    faces: tuple[tuple[int, int, int], ...],
) -> _SurfaceStub:
    return _SurfaceStub(
        mesh=trimesh.Trimesh(
            vertices=np.asarray(vertices, dtype=float),
            faces=np.asarray(faces, dtype=np.int64),
            process=False,
        )
    )


def _square_surface_with_opposite_diagonal() -> _SurfaceStub:
    return _build_surface(
        (
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 2.0, 0.0),
            (0.0, 2.0, 0.0),
        ),
        ((0, 1, 3), (1, 2, 3)),
    )


# ### Tests ###
class RemoveCoveredSurfaceFacesTests(unittest.TestCase):
    def test_removes_an_exact_oriented_triangle(self) -> None:
        source = _build_square_with_distant_triangle()
        surface = _build_surface(
            (
                (0.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                (2.0, 2.0, 0.0),
            ),
            ((0, 1, 2),),
        )

        filtered = remove_covered_surface_faces(source, (surface,))

        self.assertEqual(len(filtered.faces), 2)
        self.assertFalse(
            any(
                np.allclose(triangle, surface.mesh.triangles[0])
                for triangle in filtered.triangles
            )
        )

    def test_removes_coplanar_coverage_with_different_triangulation(self) -> None:
        source = _build_square_with_distant_triangle()

        filtered = remove_covered_surface_faces(
            source,
            (_square_surface_with_opposite_diagonal(),),
        )

        self.assertEqual(len(filtered.faces), 1)
        np.testing.assert_allclose(
            filtered.triangles[0],
            source.triangles[2],
        )

    def test_empty_surface_collection_returns_an_independent_copy(self) -> None:
        source = _build_square_with_distant_triangle()
        original_vertices = source.vertices.copy()
        original_faces = source.faces.copy()

        filtered = remove_covered_surface_faces(source, ())
        filtered.vertices[0, 0] = 100.0

        self.assertIsNot(filtered, source)
        np.testing.assert_array_equal(source.vertices, original_vertices)
        np.testing.assert_array_equal(source.faces, original_faces)

    def test_full_coverage_returns_an_empty_mesh_without_mutating_source(
        self,
    ) -> None:
        source = _build_square_with_distant_triangle()
        source = trimesh.Trimesh(
            vertices=source.vertices[:4],
            faces=source.faces[:2],
            process=False,
        )
        original_vertices = source.vertices.copy()
        original_faces = source.faces.copy()

        filtered = remove_covered_surface_faces(
            source,
            (_square_surface_with_opposite_diagonal(),),
        )

        self.assertEqual(len(filtered.faces), 0)
        self.assertEqual(len(filtered.vertices), 0)
        np.testing.assert_array_equal(source.vertices, original_vertices)
        np.testing.assert_array_equal(source.faces, original_faces)


if __name__ == "__main__":
    unittest.main()
