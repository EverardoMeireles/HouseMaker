# ### Imports ###
from __future__ import annotations

import json
import unittest
from io import BytesIO
from itertools import combinations

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.glb import (
    GLTF_Y_UP_TO_Z_UP_TRANSFORM,
    _serialize_scene_glb_with_half_mesh_extras,
)
from housemaker.object_face_edit import (
    add_object_face_preserving_uvs,
    load_object_face_geometry,
)


# ### Fixture helpers ###
def _pattern_texture() -> Image.Image:
    rows, columns = np.indices((7, 9))
    pixels = np.empty((7, 9, 4), dtype=np.uint8)
    pixels[..., 0] = (columns * 31 + rows * 7) % 256
    pixels[..., 1] = (columns * 13 + rows * 43) % 256
    pixels[..., 2] = (columns * 19 + rows * 17) % 256
    pixels[..., 3] = 255
    return Image.fromarray(pixels, mode="RGBA")


def _apply_test_material(mesh: trimesh.Trimesh) -> None:
    vertices = np.asarray(mesh.vertices, dtype=float)
    minimum = np.min(vertices[:, :2], axis=0)
    span = np.maximum(np.ptp(vertices[:, :2], axis=0), 1.0)
    mesh.visual = TextureVisuals(
        uv=(vertices[:, :2] - minimum) / span,
        material=PBRMaterial(
            name="preserved-material",
            baseColorTexture=_pattern_texture(),
            metallicFactor=0.25,
            roughnessFactor=0.75,
        ),
    )


def _open_box_glb(
    *,
    half_mesh: bool = False,
    duplicate_bottom_face: bool = False,
) -> tuple[bytes, tuple[int, ...]]:
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    top_z = float(np.max(vertices[:, 2]))
    top_face_mask = np.all(np.isclose(vertices[faces, 2], top_z), axis=1)
    mesh.update_faces(~top_face_mask)
    if duplicate_bottom_face:
        retained_faces = np.asarray(mesh.faces, dtype=np.int64)
        bottom_face_index = int(
            np.flatnonzero(
                np.all(
                    np.isclose(vertices[retained_faces, 2], -top_z),
                    axis=1,
                )
            )[0]
        )
        mesh.faces = np.vstack((retained_faces, retained_faces[bottom_face_index]))
    _apply_test_material(mesh)
    transform = trimesh.transformations.concatenate_matrices(
        trimesh.transformations.translation_matrix((2.0, -1.0, 3.0)),
        trimesh.transformations.rotation_matrix(0.25, (0.0, 0.0, 1.0)),
    )
    node_name = "[HALF] editable-box" if half_mesh else "editable-box"
    node_metadata = (
        {
            "halfMesh": {
                "mirrorPlane": {
                    "point": [0.0, 0.0, 0.0],
                    "normal": [1.0, 0.0, 0.0],
                },
                "uvMode": "reuse",
            }
        }
        if half_mesh
        else {"selection_role": "editable"}
    )
    scene = trimesh.Scene(metadata={"fixture": "open-box"})
    scene.add_geometry(
        mesh,
        geom_name="editable-geometry",
        node_name=node_name,
        transform=transform,
        metadata=node_metadata,
    )
    source = (
        _serialize_scene_glb_with_half_mesh_extras(
            scene,
            failure_message="Fixture export failed.",
        )
        if half_mesh
        else bytes(scene.export(file_type="glb"))
    )
    top_points = np.asarray(
        (
            (-0.5, -0.5, top_z),
            (0.5, -0.5, top_z),
            (0.5, 0.5, top_z),
            (-0.5, 0.5, top_z),
        ),
        dtype=float,
    )
    expected_world = trimesh.transform_points(top_points, transform)
    expected_world = trimesh.transform_points(
        expected_world,
        GLTF_Y_UP_TO_Z_UP_TRANSFORM,
    )
    return source, _match_canonical_vertices(source, expected_world)


def _open_concave_prism_glb() -> tuple[bytes, tuple[int, ...]]:
    outline = np.asarray(
        ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (1.0, 1.0), (0.0, 2.0)),
        dtype=float,
    )
    bottom = np.column_stack((outline, np.full(len(outline), -0.5)))
    top = np.column_stack((outline, np.full(len(outline), 0.5)))
    vertices = np.vstack((bottom, top))
    faces: list[tuple[int, int, int]] = []
    for index in range(len(outline)):
        following = (index + 1) % len(outline)
        faces.extend(
            (
                (index, following, following + len(outline)),
                (index, following + len(outline), index + len(outline)),
            )
        )
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    _apply_test_material(mesh)
    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name="concave-prism")
    source = bytes(scene.export(file_type="glb"))
    expected_world = trimesh.transform_points(
        top,
        GLTF_Y_UP_TO_Z_UP_TRANSFORM,
    )
    return source, _match_canonical_vertices(source, expected_world)


def _face_expanded_open_box_glb() -> tuple[bytes, tuple[int, ...]]:
    """Return an open box whose every triangle owns separate UV vertices."""

    source_mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    source_vertices = np.asarray(source_mesh.vertices, dtype=float)
    source_faces = np.asarray(source_mesh.faces, dtype=np.int64)
    top_z = float(np.max(source_vertices[:, 2]))
    top_face_mask = np.all(
        np.isclose(source_vertices[source_faces, 2], top_z),
        axis=1,
    )
    retained_faces = source_faces[~top_face_mask]
    expanded_vertices = source_vertices[retained_faces].reshape((-1, 3))
    expanded_faces = np.arange(
        len(expanded_vertices),
        dtype=np.int64,
    ).reshape((-1, 3))
    expanded_mesh = trimesh.Trimesh(
        vertices=expanded_vertices,
        faces=expanded_faces,
        process=False,
    )
    expanded_mesh.visual = TextureVisuals(
        uv=np.tile(
            np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=float),
            (len(expanded_faces), 1),
        ),
        material=PBRMaterial(
            name="face-expanded-material",
            baseColorTexture=_pattern_texture(),
        ),
    )
    source = bytes(trimesh.Scene(expanded_mesh).export(file_type="glb"))
    geometry = load_object_face_geometry(source)
    top_points = trimesh.transform_points(
        np.asarray(
            (
                (-0.5, -0.5, top_z),
                (0.5, -0.5, top_z),
                (0.5, 0.5, top_z),
                (-0.5, 0.5, top_z),
            ),
            dtype=float,
        ),
        GLTF_Y_UP_TO_Z_UP_TRANSFORM,
    )
    selected = tuple(
        int(index)
        for point in top_points
        for index in np.flatnonzero(
            np.all(np.isclose(geometry.vertices, point, atol=1e-7), axis=1)
        )
    )
    if len(selected) <= len(top_points):
        raise AssertionError("The fixture contains no expanded UV-seam vertices.")
    return source, selected


def _match_canonical_vertices(
    source: bytes,
    expected_points: np.ndarray,
) -> tuple[int, ...]:
    geometry = load_object_face_geometry(source)
    indices: list[int] = []
    for point in expected_points:
        matches = np.flatnonzero(
            np.all(np.isclose(geometry.vertices, point, atol=1e-7), axis=1)
        )
        if len(matches) != 1:
            raise AssertionError("Fixture point did not map to one canonical vertex.")
        indices.append(int(matches[0]))
    return tuple(indices)


def _load_scene(payload: bytes) -> trimesh.Scene:
    loaded = trimesh.load(
        BytesIO(payload),
        file_type="glb",
        force="scene",
        process=False,
    )
    assert isinstance(loaded, trimesh.Scene)
    return loaded


def _only_mesh(scene: trimesh.Scene) -> trimesh.Trimesh:
    meshes = [
        geometry
        for geometry in scene.geometry.values()
        if isinstance(geometry, trimesh.Trimesh)
    ]
    assert len(meshes) == 1
    return meshes[0]


def _texture_pixels(mesh: trimesh.Trimesh) -> np.ndarray:
    return np.asarray(
        mesh.visual.material.baseColorTexture.convert("RGBA"),
        dtype=np.uint8,
    )


def _gltf_document(payload: bytes) -> dict[str, object]:
    json_byte_count = int.from_bytes(payload[12:16], "little")
    return json.loads(payload[20 : 20 + json_byte_count].decode("utf-8"))


# ### Successful face-addition tests ###
class ObjectFaceAdditionTests(unittest.TestCase):
    def test_quad_hole_is_filled_without_changing_existing_asset_data(self) -> None:
        source, selected = _open_box_glb()
        source_scene = _load_scene(source)
        source_mesh = _only_mesh(source_scene)
        source_geometry = load_object_face_geometry(source)

        result = add_object_face_preserving_uvs(source, selected)

        output_scene = _load_scene(result.glb_bytes)
        output_mesh = _only_mesh(output_scene)
        output_geometry = load_object_face_geometry(result.glb_bytes)
        self.assertEqual(result.original_face_count, 10)
        self.assertEqual(result.result_face_count, 12)
        self.assertEqual(result.added_triangle_count, 2)
        self.assertEqual(result.added_face_indices, (10, 11))
        self.assertEqual(result.selected_vertex_indices, selected)
        self.assertTrue(result.preserved_textured_uvs)
        self.assertEqual(output_geometry.face_count, 12)
        np.testing.assert_allclose(
            output_geometry.vertices[output_geometry.faces[:10]],
            source_geometry.vertices[source_geometry.faces],
            rtol=0.0,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            output_geometry.uv_triangles[:10],
            source_geometry.uv_triangles,
            rtol=0.0,
            atol=1e-7,
        )
        np.testing.assert_array_equal(
            _texture_pixels(output_mesh),
            _texture_pixels(source_mesh),
        )
        np.testing.assert_allclose(
            np.asarray(output_mesh.vertex_normals, dtype=float)[
                np.asarray(output_mesh.faces, dtype=np.int64)[:10]
            ],
            np.asarray(source_mesh.vertex_normals, dtype=float)[
                np.asarray(source_mesh.faces, dtype=np.int64)
            ],
            rtol=0.0,
            atol=1e-7,
        )
        self.assertEqual(
            output_mesh.visual.material.name,
            source_mesh.visual.material.name,
        )
        self.assertEqual(output_scene.metadata, source_scene.metadata)
        source_transform, _source_geometry_name = source_scene.graph.get("editable-box")
        output_transform, _output_geometry_name = output_scene.graph.get("editable-box")
        np.testing.assert_array_equal(output_transform, source_transform)
        parent_name = output_scene.graph.transforms.parents["editable-box"]
        self.assertEqual(
            output_scene.graph.transforms.edge_data[(parent_name, "editable-box")][
                "metadata"
            ],
            {"selection_role": "editable"},
        )

    def test_concave_ngon_uses_all_selected_vertices_and_is_deterministic(
        self,
    ) -> None:
        source, selected = _open_concave_prism_glb()

        first = add_object_face_preserving_uvs(source, selected)
        second = add_object_face_preserving_uvs(source, selected)

        first_geometry = load_object_face_geometry(first.glb_bytes)
        second_geometry = load_object_face_geometry(second.glb_bytes)
        self.assertEqual(first.added_triangle_count, len(selected) - 2)
        self.assertEqual(first.added_face_indices, (10, 11, 12))
        np.testing.assert_array_equal(first_geometry.faces, second_geometry.faces)
        added_vertices = {
            int(index)
            for index in first_geometry.faces[list(first.added_face_indices)].ravel()
        }
        self.assertEqual(added_vertices, set(selected))

    def test_scrambled_quad_selection_is_normalized_to_its_edge_cycle(self) -> None:
        source, selected = _open_box_glb()
        scrambled = (selected[0], selected[2], selected[1], selected[3])

        result = add_object_face_preserving_uvs(source, scrambled)

        self.assertEqual(result.selected_vertex_indices, scrambled)
        self.assertEqual(result.added_triangle_count, 2)
        self.assertEqual(result.result_face_count, 12)

    def test_half_mesh_extras_survive_face_addition(self) -> None:
        source, selected = _open_box_glb(half_mesh=True)

        result = add_object_face_preserving_uvs(source, selected)

        document = _gltf_document(result.glb_bytes)
        half_node = next(
            node
            for node in document["nodes"]
            if node.get("name") == "[HALF] editable-box"
        )
        self.assertEqual(
            half_node["extras"]["halfMesh"],
            {
                "mirrorPlane": {
                    "point": [0.0, 0.0, 0.0],
                    "normal": [1.0, 0.0, 0.0],
                },
                "uvMode": "reuse",
            },
        )

    def test_unrelated_preexisting_nonmanifold_edges_do_not_block_fill(
        self,
    ) -> None:
        source, selected = _open_box_glb(duplicate_bottom_face=True)

        result = add_object_face_preserving_uvs(source, selected)

        self.assertEqual(result.added_triangle_count, 2)
        self.assertEqual(result.original_face_count, 11)
        self.assertEqual(result.result_face_count, 13)

    def test_face_expanded_uv_seam_ids_collapse_to_visible_vertices(self) -> None:
        source, selected = _face_expanded_open_box_glb()
        source_scene = _load_scene(source)
        source_mesh = _only_mesh(source_scene)
        source_geometry = load_object_face_geometry(source)

        result = add_object_face_preserving_uvs(source, reversed(selected))

        output_scene = _load_scene(result.glb_bytes)
        output_mesh = _only_mesh(output_scene)
        output_geometry = load_object_face_geometry(result.glb_bytes)
        self.assertEqual(result.selected_vertex_indices, tuple(reversed(selected)))
        self.assertEqual(len(result.resolved_vertex_indices), 4)
        self.assertEqual(
            len(
                {
                    tuple(np.round(source_geometry.vertices[index], 7))
                    for index in result.resolved_vertex_indices
                }
            ),
            4,
        )
        self.assertEqual(result.original_face_count, 10)
        self.assertEqual(result.added_triangle_count, 2)
        self.assertEqual(result.result_face_count, 12)
        np.testing.assert_allclose(
            output_geometry.uv_triangles[:10],
            source_geometry.uv_triangles,
            rtol=0.0,
            atol=1e-7,
        )
        np.testing.assert_array_equal(
            _texture_pixels(output_mesh),
            _texture_pixels(source_mesh),
        )


# ### Face-addition validation tests ###
class ObjectFaceAdditionValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source, self.selected = _open_box_glb()
        self.geometry = load_object_face_geometry(self.source)

    def test_selection_must_contain_three_unique_valid_integer_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least three"):
            add_object_face_preserving_uvs(self.source, self.selected[:2])
        with self.assertRaisesRegex(ValueError, "unique"):
            add_object_face_preserving_uvs(
                self.source,
                (self.selected[0], self.selected[1], self.selected[0]),
            )
        with self.assertRaisesRegex(ValueError, "outside the object"):
            add_object_face_preserving_uvs(self.source, (0, 1, 999))
        with self.assertRaisesRegex(TypeError, "must be integers"):
            add_object_face_preserving_uvs(self.source, (0, 1, 2.5))

    def test_vertices_from_two_mesh_primitives_are_rejected(self) -> None:
        scene = trimesh.Scene()
        scene.add_geometry(
            trimesh.creation.icosphere(subdivisions=1),
            node_name="a-node",
        )
        scene.add_geometry(
            trimesh.creation.icosphere(subdivisions=1),
            node_name="b-node",
            transform=trimesh.transformations.translation_matrix((5.0, 0.0, 0.0)),
        )
        source = bytes(scene.export(file_type="glb"))
        geometry = load_object_face_geometry(source)
        first_face = geometry.faces[0]
        second_node_first_vertex = len(scene.geometry["geometry_0"].vertices)

        with self.assertRaisesRegex(ValueError, "one mesh primitive"):
            add_object_face_preserving_uvs(
                source,
                (
                    int(first_face[0]),
                    int(first_face[1]),
                    second_node_first_vertex,
                ),
            )

    def test_overlapping_primitives_are_not_resolved_silently(self) -> None:
        vertices = np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=float,
        )
        faces = np.asarray(((0, 1, 2),), dtype=np.int64)
        scene = trimesh.Scene()
        scene.add_geometry(
            trimesh.Trimesh(vertices=vertices, faces=faces, process=False),
            geom_name="first-geometry",
            node_name="first-node",
        )
        scene.add_geometry(
            trimesh.Trimesh(vertices=vertices, faces=faces, process=False),
            geom_name="second-geometry",
            node_name="second-node",
        )
        source = bytes(scene.export(file_type="glb"))
        geometry = load_object_face_geometry(source)

        with self.assertRaisesRegex(ValueError, "ambiguously match multiple"):
            add_object_face_preserving_uvs(
                source,
                tuple(range(len(geometry.vertices))),
            )

    def test_nonplanar_and_existing_faces_are_rejected(
        self,
    ) -> None:
        nonplanar: tuple[int, ...] | None = None
        for candidate in combinations(range(len(self.geometry.vertices)), 4):
            points = self.geometry.vertices[list(candidate)]
            determinant = np.linalg.det(
                np.column_stack(
                    (
                        points[1] - points[0],
                        points[2] - points[0],
                        points[3] - points[0],
                    )
                )
            )
            if abs(float(determinant)) > 1e-5:
                nonplanar = tuple(int(index) for index in candidate)
                break
        assert nonplanar is not None
        with self.assertRaisesRegex(ValueError, "coplanar"):
            add_object_face_preserving_uvs(self.source, nonplanar)

        existing_face = tuple(int(index) for index in self.geometry.faces[0])
        with self.assertRaisesRegex(ValueError, "overlaps an existing"):
            add_object_face_preserving_uvs(self.source, existing_face)

    def test_too_few_geometric_positions_and_collinear_vertices_are_rejected(
        self,
    ) -> None:
        vertices = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (1.0, 0.0, 0.0),
            ),
            dtype=float,
        )
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=np.asarray(((0, 1, 3), (1, 2, 3), (0, 3, 4)), dtype=np.int64),
            process=False,
        )
        source = bytes(trimesh.Scene(mesh).export(file_type="glb"))
        geometry = load_object_face_geometry(source)
        points = geometry.vertices
        collinear = tuple(
            int(np.flatnonzero(np.all(np.isclose(points, point), axis=1))[0])
            for point in (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
            )
        )
        with self.assertRaisesRegex(ValueError, "collinear"):
            add_object_face_preserving_uvs(source, collinear)

        duplicate_position_ids = tuple(
            int(index)
            for index in np.flatnonzero(
                np.all(np.isclose(points, (1.0, 0.0, 0.0)), axis=1)
            )
        )
        if len(duplicate_position_ids) >= 2:
            with self.assertRaisesRegex(
                ValueError,
                "three geometrically distinct visible vertices",
            ):
                add_object_face_preserving_uvs(
                    source,
                    (
                        duplicate_position_ids[0],
                        duplicate_position_ids[1],
                        collinear[0],
                    ),
                )

    def test_adding_a_third_face_to_an_edge_is_rejected(self) -> None:
        vertices = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, -1.0, 1.0),
            ),
            dtype=float,
        )
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=np.asarray(((0, 1, 2), (1, 0, 3), (0, 3, 4)), dtype=np.int64),
            process=False,
        )
        source = bytes(trimesh.Scene(mesh).export(file_type="glb"))
        selected = _match_canonical_vertices(
            source,
            trimesh.transform_points(
                vertices[[0, 1, 4]],
                GLTF_Y_UP_TO_Z_UP_TRANSFORM,
            ),
        )

        with self.assertRaisesRegex(ValueError, "non-manifold"):
            add_object_face_preserving_uvs(source, selected)


# ### Test runner ###
if __name__ == "__main__":
    unittest.main()
