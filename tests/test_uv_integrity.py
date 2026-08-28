# ### Imports ###
from __future__ import annotations

import unittest

import numpy as np
import trimesh
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.uv_integrity import (
    UvIntegrityError,
    build_uv_fingerprint,
)


# ### Integrity tests ###
class UvIntegrityTests(unittest.TestCase):
    def test_redundant_vertex_welding_preserves_face_uvs(self) -> None:
        submitted = build_uv_fingerprint(
            _square_glb(redundant_vertices=True)
        )
        returned = build_uv_fingerprint(
            _square_glb(redundant_vertices=False)
        )

        self.assertEqual(submitted, returned)
        self.assertEqual(submitted.face_count, 2)

    def test_reordered_primitives_and_reindexed_vertices_match(self) -> None:
        submitted = build_uv_fingerprint(
            _two_primitive_glb(("first", "second"))
        )
        returned = build_uv_fingerprint(
            _two_primitive_glb(
                ("second", "first"),
                reindex_vertices=True,
            )
        )

        self.assertEqual(submitted, returned)
        self.assertEqual(submitted.face_count, 2)

    def test_one_mutated_face_uv_changes_the_fingerprint(self) -> None:
        submitted = build_uv_fingerprint(
            _two_primitive_glb(("first", "second"))
        )
        returned = build_uv_fingerprint(
            _two_primitive_glb(
                ("second", "first"),
                reindex_vertices=True,
                mutate_second_uv=True,
            )
        )

        self.assertNotEqual(submitted.sha256, returned.sha256)
        self.assertEqual(submitted.face_count, returned.face_count)

    def test_empty_payload_is_rejected(self) -> None:
        with self.assertRaisesRegex(UvIntegrityError, "empty GLB"):
            build_uv_fingerprint(b"")


# ### GLB fixtures ###
def _square_glb(*, redundant_vertices: bool) -> bytes:
    shared_vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        dtype=float,
    )
    shared_uv = np.asarray(
        (
            (0.1, 0.1),
            (0.9, 0.1),
            (0.9, 0.9),
            (0.1, 0.9),
        ),
        dtype=float,
    )
    if redundant_vertices:
        vertices = shared_vertices[[0, 1, 2, 0, 2, 3]]
        uv = shared_uv[[0, 1, 2, 0, 2, 3]]
        faces = np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64)
    else:
        vertices = shared_vertices
        uv = shared_uv
        faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual = TextureVisuals(uv=uv, material=PBRMaterial())
    return bytes(trimesh.Scene(mesh).export(file_type="glb"))


def _two_primitive_glb(
    primitive_order: tuple[str, str],
    *,
    reindex_vertices: bool = False,
    mutate_second_uv: bool = False,
) -> bytes:
    scene = trimesh.Scene()
    for primitive_name in primitive_order:
        mesh = _primitive_mesh(
            primitive_name,
            reindex_vertices=reindex_vertices,
            mutate_uv=(mutate_second_uv and primitive_name == "second"),
        )
        scene.add_geometry(
            mesh,
            geom_name=f"returned-{primitive_name}-{len(scene.geometry)}",
            node_name=f"node-{primitive_name}-{len(scene.geometry)}",
        )
    return bytes(scene.export(file_type="glb"))


def _primitive_mesh(
    primitive_name: str,
    *,
    reindex_vertices: bool,
    mutate_uv: bool,
) -> trimesh.Trimesh:
    x_offset = 0.0 if primitive_name == "first" else 2.0
    uv_offset = 0.0 if primitive_name == "first" else 0.5
    vertices = np.asarray(
        (
            (x_offset, 0.0, 0.0),
            (x_offset + 1.0, 0.0, 0.0),
            (x_offset, 1.0, 0.0),
        ),
        dtype=float,
    )
    uv = np.asarray(
        (
            (uv_offset + 0.05, 0.05),
            (uv_offset + 0.4, 0.05),
            (uv_offset + 0.05, 0.4),
        ),
        dtype=float,
    )
    if reindex_vertices:
        vertices = vertices[[2, 0, 1]]
        uv = uv[[2, 0, 1]]
    if mutate_uv:
        uv[0] = (0.75, 0.75)
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(((0, 1, 2),), dtype=np.int64),
        process=False,
    )
    mesh.visual = TextureVisuals(uv=uv, material=PBRMaterial())
    return mesh


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
