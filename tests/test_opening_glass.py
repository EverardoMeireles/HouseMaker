# ### Imports ###
from __future__ import annotations

import copy
import json
import unittest
from dataclasses import dataclass

import numpy as np
import trimesh
from trimesh.visual.material import MultiMaterial, PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.glass_material import is_housemaker_glass_material
from housemaker.glb import (
    GLTF_Y_UP_TO_Z_UP_TRANSFORM,
    Z_UP_TO_GLTF_Y_UP_TRANSFORM,
    GeneratedModel,
)
from housemaker.opening_glass import (
    OPENING_GLASS_EXTRAS_KEY,
    OPENING_GLASS_METADATA_KEY,
    NormalizedGlassArea,
    apply_opening_glass_areas,
    coerce_normalized_glass_area,
)


# ### Test data models ###
@dataclass(frozen=True)
class _PersistentAreaFixture:
    glass_area_id: str
    source_object_id: str
    name: str
    minimum_x_ratio: float
    maximum_x_ratio: float
    minimum_z_ratio: float
    maximum_z_ratio: float


# ### Fixture helpers ###
def _build_textured_body_model(
    *,
    node_name: str = "body",
    node_transform: np.ndarray | None = None,
    node_metadata: dict[str, object] | None = None,
) -> GeneratedModel:
    source = trimesh.creation.box(extents=(2.0, 0.2, 3.0))
    source.apply_translation((0.0, 0.0, 1.5))
    source_vertices = np.asarray(source.vertices, dtype=float)
    source_faces = np.asarray(source.faces, dtype=np.int64)
    expanded_vertices = source_vertices[source_faces].reshape((-1, 3))
    expanded_faces = np.arange(
        len(expanded_vertices),
        dtype=np.int64,
    ).reshape((-1, 3))
    body = trimesh.Trimesh(
        vertices=expanded_vertices,
        faces=expanded_faces,
        process=False,
    )
    uv = np.column_stack(
        (
            (expanded_vertices[:, 0] + 1.0) * 0.5,
            expanded_vertices[:, 2] / 3.0,
        )
    )
    face_materials = np.arange(len(expanded_faces), dtype=np.int64) % 2
    body.visual = TextureVisuals(
        uv=uv,
        material=MultiMaterial(
            (
                PBRMaterial(
                    name="body-a",
                    metallicFactor=0.2,
                    roughnessFactor=0.7,
                ),
                PBRMaterial(
                    name="body-b",
                    metallicFactor=0.1,
                    roughnessFactor=0.8,
                ),
            )
        ),
        face_materials=face_materials,
    )
    export_body = copy.deepcopy(body)
    export_body.apply_transform(Z_UP_TO_GLTF_Y_UP_TRANSFORM)
    transform = (
        np.eye(4, dtype=float)
        if node_transform is None
        else np.asarray(node_transform, dtype=float)
    )
    scene = trimesh.Scene(metadata={"fixture": "opening-glass"})
    scene.add_geometry(
        export_body,
        geom_name="body-geometry",
        node_name=node_name,
        transform=transform,
        metadata=copy.deepcopy(node_metadata or {"role": "body"}),
    )
    preview = scene.to_geometry()
    preview.apply_transform(GLTF_Y_UP_TO_Z_UP_TRANSFORM)
    return GeneratedModel(
        mesh=preview,
        scene=scene,
        glb_bytes=bytes(scene.export(file_type="glb")),
    )


def _decode_glb_document(payload: bytes) -> dict[str, object]:
    json_byte_count = int.from_bytes(payload[12:16], "little")
    json_end = 20 + json_byte_count
    return json.loads(
        payload[20:json_end].rstrip(b" \t\r\n\0").decode("utf-8")
    )


def _body_geometry(result_model: GeneratedModel) -> trimesh.Trimesh:
    return result_model.scene.geometry["body-geometry"]


def _body_vertices_z_up(result_model: GeneratedModel) -> np.ndarray:
    body = _body_geometry(result_model)
    node_transform, _geometry_name = result_model.scene.graph.get("body")
    return trimesh.transform_points(
        np.asarray(body.vertices, dtype=float),
        GLTF_Y_UP_TO_Z_UP_TRANSFORM @ np.asarray(node_transform, dtype=float),
    )


# ### Area validation tests ###
class OpeningGlassAreaValidationTests(unittest.TestCase):
    def test_coerces_persistent_door_or_window_area_shape(self) -> None:
        persistent = _PersistentAreaFixture(
            glass_area_id="area-1",
            source_object_id="source-1",
            name="Glass area 1",
            minimum_x_ratio=0.1,
            maximum_x_ratio=0.4,
            minimum_z_ratio=0.2,
            maximum_z_ratio=0.8,
        )

        normalized = coerce_normalized_glass_area(persistent)

        self.assertEqual(normalized.area_id, "area-1")
        self.assertEqual(normalized.name, "Glass area 1")
        self.assertEqual(normalized.normalized_bounds, (0.1, 0.4, 0.2, 0.8))

    def test_rejects_invalid_or_overlapping_normalized_areas(self) -> None:
        with self.assertRaisesRegex(ValueError, "between zero and one"):
            NormalizedGlassArea("bad", -0.1, 0.5, 0.2, 0.8)
        model = _build_textured_body_model()
        first = NormalizedGlassArea("first", 0.1, 0.6, 0.1, 0.6)
        second = NormalizedGlassArea("second", 0.4, 0.9, 0.4, 0.9)

        with self.assertRaisesRegex(ValueError, "cannot overlap"):
            apply_opening_glass_areas(model, (first, second))

    def test_rejects_nonintersecting_area_atomically(self) -> None:
        model = _build_textured_body_model()
        source_vertices = np.asarray(model.mesh.vertices, dtype=float).copy()

        with self.assertRaisesRegex(ValueError, "No body faces were cut"):
            apply_opening_glass_areas(
                model,
                (NormalizedGlassArea("outside", 0.2, 0.8, 0.2, 0.8),),
                profile_bounds_meters=(10.0, 12.0, 10.0, 13.0),
            )

        np.testing.assert_allclose(model.mesh.vertices, source_vertices)
        self.assertEqual(tuple(model.scene.geometry), ("body-geometry",))


# ### Geometry and visual preservation tests ###
class OpeningGlassGeometryTests(unittest.TestCase):
    def test_exact_cut_preserves_uvs_materials_and_adds_prefab_glass(self) -> None:
        model = _build_textured_body_model()
        source_body = model.scene.geometry["body-geometry"]
        source_vertices = np.asarray(source_body.vertices, dtype=float).copy()
        source_uv = np.asarray(source_body.visual.uv, dtype=float).copy()
        source_face_materials = np.asarray(
            source_body.visual.face_materials,
            dtype=np.int64,
        ).copy()
        area = NormalizedGlassArea(
            "center-pane",
            0.25,
            0.75,
            0.25,
            0.75,
            name="Glass area 1",
        )

        result = apply_opening_glass_areas(
            model,
            (area,),
            profile_bounds_meters=(-1.0, 1.0, 0.0, 3.0),
        )

        self.assertEqual(result.original_body_face_count, 12)
        self.assertEqual(result.clipped_source_face_count, 4)
        self.assertEqual(result.glass_face_count, 12)
        self.assertTrue(result.preserved_textured_uvs)
        self.assertEqual(len(result.primitives), 1)
        primitive = result.primitives[0]
        self.assertEqual(primitive.bounds_meters, (-0.5, 0.5, 0.75, 2.25))
        self.assertEqual(primitive.depth_bounds_meters, (-0.1, 0.1))

        rebuilt_body = _body_geometry(result.model)
        self.assertIsInstance(rebuilt_body.visual, TextureVisuals)
        self.assertIsInstance(rebuilt_body.visual.material, MultiMaterial)
        self.assertEqual(
            tuple(material.name for material in rebuilt_body.visual.material.materials),
            ("body-a", "body-b"),
        )
        rebuilt_materials = np.asarray(
            rebuilt_body.visual.face_materials,
            dtype=np.int64,
        )
        self.assertTrue(set(rebuilt_materials).issubset({0, 1}))

        body_vertices = _body_vertices_z_up(result.model)
        body_faces = np.asarray(rebuilt_body.faces, dtype=np.int64)
        centroids = np.mean(body_vertices[body_faces], axis=1)
        centroid_inside = (
            (centroids[:, 0] > -0.5 + 1e-8)
            & (centroids[:, 0] < 0.5 - 1e-8)
            & (centroids[:, 2] > 0.75 + 1e-8)
            & (centroids[:, 2] < 2.25 - 1e-8)
        )
        self.assertFalse(np.any(centroid_inside))

        glass_mesh = result.model.scene.geometry[primitive.geometry_name]
        self.assertTrue(is_housemaker_glass_material(glass_mesh.visual.material))
        self.assertIn(OPENING_GLASS_METADATA_KEY, glass_mesh.metadata)
        glass_preview = glass_mesh.copy()
        glass_preview.apply_transform(GLTF_Y_UP_TO_Z_UP_TRANSFORM)
        np.testing.assert_allclose(
            glass_preview.bounds,
            np.asarray(((-0.5, -0.1, 0.75), (0.5, 0.1, 2.25))),
            atol=1e-8,
        )

        # A face wholly outside the aperture is copied with exact UVs and its
        # original face-material owner, even though cut faces are subdivided.
        unaffected_source_index = next(
            index
            for index, face in enumerate(np.asarray(source_body.faces))
            if np.all(np.isclose(source_vertices[face, 0], -1.0))
        )
        expected_positions = source_vertices[source_body.faces[unaffected_source_index]]
        expected_uv = source_uv[source_body.faces[unaffected_source_index]]
        matching_face_index = _find_triangle_by_positions(
            rebuilt_body,
            expected_positions,
        )
        actual_face = rebuilt_body.faces[matching_face_index]
        _assert_vertex_values_by_position(
            np.asarray(rebuilt_body.vertices)[actual_face],
            np.asarray(rebuilt_body.visual.uv)[actual_face],
            expected_positions,
            expected_uv,
        )
        self.assertEqual(
            int(rebuilt_materials[matching_face_index]),
            int(source_face_materials[unaffected_source_index]),
        )

        # The original model remains byte-for-byte/topology unchanged.
        np.testing.assert_allclose(source_body.vertices, source_vertices)
        np.testing.assert_allclose(source_body.visual.uv, source_uv)
        np.testing.assert_array_equal(
            source_body.visual.face_materials,
            source_face_materials,
        )

    def test_multiple_glass_areas_are_independent_and_serialized(self) -> None:
        model = _build_textured_body_model(
            node_name="[HALF] body",
            node_metadata={
                "halfMesh": {
                    "mirrorPlane": {
                        "point": [0.0, 0.0, 0.0],
                        "normal": [1.0, 0.0, 0.0],
                    },
                    "uvMode": "reuse",
                }
            },
        )
        areas = (
            NormalizedGlassArea("left", 0.1, 0.35, 0.2, 0.8),
            NormalizedGlassArea("right", 0.65, 0.9, 0.2, 0.8),
        )

        result = apply_opening_glass_areas(model, areas)

        self.assertEqual(len(result.primitives), 2)
        self.assertEqual(len({item.node_name for item in result.primitives}), 2)
        self.assertTrue(
            all(
                is_housemaker_glass_material(
                    result.model.scene.geometry[item.geometry_name].visual.material
                )
                for item in result.primitives
            )
        )
        document = _decode_glb_document(result.model.glb_bytes)
        nodes = {
            node.get("name"): node
            for node in document["nodes"]
            if isinstance(node, dict)
        }
        self.assertIn("halfMesh", nodes["[HALF] body"]["extras"])
        for primitive in result.primitives:
            extras = nodes[primitive.node_name]["extras"]
            self.assertEqual(
                extras["halfMesh"],
                nodes["[HALF] body"]["extras"]["halfMesh"],
            )
            self.assertEqual(
                extras[OPENING_GLASS_EXTRAS_KEY]["areaId"],
                primitive.area.area_id,
            )

    def test_node_transform_is_respected_in_body_local_cut_coordinates(self) -> None:
        z_up_translation = trimesh.transformations.translation_matrix(
            (4.0, 0.0, 2.0)
        )
        gltf_transform = (
            Z_UP_TO_GLTF_Y_UP_TRANSFORM
            @ z_up_translation
            @ GLTF_Y_UP_TO_Z_UP_TRANSFORM
        )
        model = _build_textured_body_model(node_transform=gltf_transform)

        result = apply_opening_glass_areas(
            model,
            (NormalizedGlassArea("translated", 0.25, 0.75, 0.25, 0.75),),
            profile_bounds_meters=(3.0, 5.0, 2.0, 5.0),
        )

        np.testing.assert_allclose(
            result.primitives[0].center_meters,
            (4.0, 0.0, 3.5),
            atol=1e-8,
        )
        np.testing.assert_allclose(
            result.model.scene.graph.get("body")[0],
            gltf_transform,
            atol=1e-8,
        )

    def test_rejects_reapplying_to_already_modified_model(self) -> None:
        model = _build_textured_body_model()
        first = apply_opening_glass_areas(
            model,
            (NormalizedGlassArea("first", 0.2, 0.8, 0.2, 0.8),),
        )

        with self.assertRaisesRegex(ValueError, "unmodified body"):
            apply_opening_glass_areas(
                first.model,
                (NormalizedGlassArea("second", 0.1, 0.3, 0.1, 0.3),),
            )


# ### Assertion helpers ###
def _find_triangle_by_positions(
    mesh: trimesh.Trimesh,
    expected_positions: np.ndarray,
) -> int:
    vertices = np.asarray(mesh.vertices, dtype=float)
    for face_index, face in enumerate(np.asarray(mesh.faces, dtype=np.int64)):
        actual = vertices[face]
        if all(
            np.any(np.all(np.isclose(actual, point, atol=1e-8), axis=1))
            for point in expected_positions
        ):
            return face_index
    raise AssertionError("The unaffected source triangle was not retained.")


def _assert_vertex_values_by_position(
    actual_positions: np.ndarray,
    actual_values: np.ndarray,
    expected_positions: np.ndarray,
    expected_values: np.ndarray,
) -> None:
    for expected_position, expected_value in zip(
        expected_positions,
        expected_values,
    ):
        matches = np.flatnonzero(
            np.all(
                np.isclose(actual_positions, expected_position, atol=1e-8),
                axis=1,
            )
        )
        if len(matches) != 1:
            raise AssertionError("A retained vertex could not be matched uniquely.")
        np.testing.assert_allclose(
            actual_values[int(matches[0])],
            expected_value,
            atol=1e-8,
        )


if __name__ == "__main__":
    unittest.main()
