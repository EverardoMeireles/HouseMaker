# ### Imports ###
from __future__ import annotations

import copy
import math
import unittest
from io import BytesIO

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.glb import (
    GLTF_Y_UP_TO_Z_UP_TRANSFORM,
    Z_UP_TO_GLTF_Y_UP_TRANSFORM,
    GeneratedModel,
    PlacedGeneratedModel,
    PreviewPlacedObject,
    PreviewSymmetricObject,
    PreviewTexturedSurface,
    PreviewTexturedWall,
    compose_placed_generated_models,
    import_generated_glb,
)


# ### Fixture helpers ###
def _to_gltf_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    converted = copy.deepcopy(mesh)
    converted.apply_transform(Z_UP_TO_GLTF_Y_UP_TRANSFORM)
    return converted


def _to_gltf_transform(z_up_transform: np.ndarray) -> np.ndarray:
    return (
        Z_UP_TO_GLTF_Y_UP_TRANSFORM
        @ np.asarray(z_up_transform, dtype=float)
        @ GLTF_Y_UP_TO_Z_UP_TRANSFORM
    )


def _translation(x: float, y: float, z: float) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = (x, y, z)
    return transform


def _single_mesh_model(
    mesh: trimesh.Trimesh,
    *,
    geometry_name: str = "mesh",
    node_name: str = "node",
    node_transform: np.ndarray | None = None,
) -> GeneratedModel:
    scene = trimesh.Scene()
    scene.add_geometry(
        _to_gltf_mesh(mesh),
        geom_name=geometry_name,
        node_name=node_name,
        transform=_to_gltf_transform(
            np.eye(4, dtype=float)
            if node_transform is None
            else node_transform
        ),
    )
    return import_generated_glb(bytes(scene.export(file_type="glb")))


def _base_box_model(
    *,
    geometry_name: str = "base_geometry",
    node_name: str = "base_node",
) -> GeneratedModel:
    mesh = trimesh.creation.box(extents=(2.0, 2.0, 1.0))
    mesh.apply_translation((0.0, 0.0, 0.5))
    return _single_mesh_model(
        mesh,
        geometry_name=geometry_name,
        node_name=node_name,
    )


def _textured_quad(color: tuple[int, int, int, int]) -> trimesh.Trimesh:
    vertices = np.array(
        [
            (-0.5, -0.5, 0.0),
            (0.5, -0.5, 0.0),
            (0.5, 0.5, 0.0),
            (-0.5, 0.5, 0.0),
        ],
        dtype=float,
    )
    faces = np.array(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    texture = Image.fromarray(
        np.full((4, 4, 4), color, dtype=np.uint8),
        mode="RGBA",
    )
    visual = TextureVisuals(
        uv=np.array(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=float),
        material=PBRMaterial(
            name="fixture_texture",
            baseColorTexture=texture,
        ),
    )
    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        visual=visual,
        process=False,
    )


def _mixed_texture_model() -> GeneratedModel:
    textured_mesh = _textured_quad((220, 20, 30, 255))
    untextured_mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    untextured_mesh.apply_translation((2.0, 0.0, 0.5))
    scene = trimesh.Scene()
    scene.add_geometry(
        _to_gltf_mesh(textured_mesh),
        geom_name="shared_name",
        node_name="textured_node",
    )
    scene.add_geometry(
        _to_gltf_mesh(untextured_mesh),
        geom_name="plain_geometry",
        node_name="plain_node",
    )
    return import_generated_glb(bytes(scene.export(file_type="glb")))


def _hierarchical_model() -> GeneratedModel:
    mesh = _to_gltf_mesh(trimesh.creation.box(extents=(1.0, 1.0, 1.0)))
    scene = trimesh.Scene()
    scene.geometry["shared_name"] = mesh
    scene.graph.update(
        frame_to="parent",
        frame_from=scene.graph.base_frame,
        matrix=_to_gltf_transform(_translation(2.0, 0.0, 0.5)),
        metadata={"kind": "parent"},
    )
    scene.graph.update(
        frame_to="first_node",
        frame_from="parent",
        matrix=_to_gltf_transform(_translation(0.0, 3.0, 0.0)),
        geometry="shared_name",
        metadata={"slot": 1},
    )
    scene.graph.update(
        frame_to="second_node",
        frame_from="parent",
        matrix=_to_gltf_transform(_translation(0.0, -3.0, 0.0)),
        geometry="shared_name",
        metadata={"slot": 2},
    )
    return import_generated_glb(bytes(scene.export(file_type="glb")))


def _node_world_mesh(
    scene: trimesh.Scene,
    node_name: object,
) -> trimesh.Trimesh:
    transform, geometry_name = scene.graph.get(node_name)
    mesh = copy.deepcopy(scene.geometry[geometry_name])
    mesh.apply_transform(
        GLTF_Y_UP_TO_Z_UP_TRANSFORM
        @ np.asarray(transform, dtype=float)
    )
    return mesh


def _placed_world_mesh(scene: trimesh.Scene) -> trimesh.Trimesh:
    placed_nodes = [
        node_name
        for node_name in scene.graph.nodes_geometry
        if str(node_name).startswith("placed_")
    ]
    return trimesh.util.concatenate(
        [_node_world_mesh(scene, node_name) for node_name in placed_nodes]
    )


def _texture_average(mesh: trimesh.Trimesh) -> np.ndarray:
    material = getattr(getattr(mesh, "visual", None), "material", None)
    texture = getattr(material, "baseColorTexture", None)
    if texture is None:
        texture = getattr(material, "image", None)
    texture_rgba = np.asarray(texture.convert("RGBA"), dtype=np.uint8)
    return np.mean(texture_rgba, axis=(0, 1))


# ### Composition tests ###
class PlacedGeneratedModelCompositionTests(unittest.TestCase):
    def test_rotation_keeps_the_source_bottom_center_on_the_world_anchor(
        self,
    ) -> None:
        base_model = _base_box_model()
        object_mesh = trimesh.creation.box(extents=(2.0, 4.0, 2.0))
        object_mesh.apply_translation((4.0, -2.0, 1.0))
        object_model = _single_mesh_model(object_mesh)

        composed = compose_placed_generated_models(
            base_model,
            [
                PlacedGeneratedModel(
                    object_id="rotated-chair",
                    model=object_model,
                    world_position=(10.0, 20.0, 30.0),
                    rotation_degrees=(0.0, 0.0, 90.0),
                )
            ],
        )

        placed_mesh = _placed_world_mesh(composed.scene)
        np.testing.assert_allclose(
            placed_mesh.bounds,
            ((8.0, 19.0, 30.0), (12.0, 21.0, 32.0)),
            atol=1e-7,
        )
        np.testing.assert_allclose(
            (
                placed_mesh.bounding_box.centroid[0],
                placed_mesh.bounding_box.centroid[1],
                placed_mesh.bounds[0, 2],
            ),
            (10.0, 20.0, 30.0),
            atol=1e-7,
        )
        exported = import_generated_glb(composed.glb_bytes)
        np.testing.assert_allclose(
            exported.mesh.bounds,
            composed.mesh.bounds,
            atol=1e-7,
        )

    def test_symmetric_placement_adds_only_a_world_space_preview_mirror(
        self,
    ) -> None:
        base_model = _base_box_model()
        retained_mesh = trimesh.creation.box(extents=(1.0, 2.0, 2.0))
        retained_mesh.apply_translation((-0.5, 0.0, 1.0))
        retained_model = _single_mesh_model(retained_mesh)
        retained_faces = len(retained_model.mesh.faces)
        source_payload = bytes(retained_model.glb_bytes)

        composed = compose_placed_generated_models(
            base_model,
            [
                PlacedGeneratedModel(
                    object_id="half-chair",
                    model=retained_model,
                    world_position=(10.0, 20.0, 30.0),
                    symmetric_preview_orientation="VERTICAL",
                    symmetric_preview_plane_coordinate=0.0,
                )
            ],
        )

        self.assertEqual(len(composed.preview_symmetric_objects), 1)
        preview = composed.preview_symmetric_objects[0]
        self.assertIsInstance(preview, PreviewSymmetricObject)
        self.assertEqual(preview.object_id, "half-chair")
        self.assertEqual(preview.orientation, "vertical")
        self.assertAlmostEqual(preview.plane_coordinate, 10.5)
        self.assertEqual(len(preview.meshes), 1)
        np.testing.assert_allclose(
            preview.meshes[0].bounds,
            ((9.5, 19.0, 30.0), (10.5, 21.0, 32.0)),
            atol=1e-7,
        )

        exported = import_generated_glb(composed.glb_bytes)
        self.assertEqual(
            len(exported.mesh.faces),
            len(base_model.mesh.faces) + retained_faces,
        )
        self.assertEqual(
            len(composed.mesh.faces),
            len(base_model.mesh.faces) + retained_faces,
        )
        self.assertEqual(retained_model.glb_bytes, source_payload)

    def test_rotated_symmetric_preview_keeps_local_mesh_and_full_transform(
        self,
    ) -> None:
        base_model = _base_box_model()
        retained_mesh = trimesh.creation.box(extents=(1.0, 2.0, 2.0))
        retained_mesh.apply_translation((-0.5, 0.0, 1.0))
        retained_model = _single_mesh_model(retained_mesh)
        retained_face_count = len(retained_model.mesh.faces)

        composed = compose_placed_generated_models(
            base_model,
            [
                PlacedGeneratedModel(
                    object_id="rotated-half-chair",
                    model=retained_model,
                    world_position=(10.0, 20.0, 30.0),
                    rotation_degrees=(0.0, 0.0, 90.0),
                    symmetric_preview_orientation="vertical",
                    symmetric_preview_plane_coordinate=0.0,
                )
            ],
        )

        self.assertEqual(len(composed.preview_placed_objects), 1)
        preview = composed.preview_placed_objects[0]
        self.assertIsInstance(preview, PreviewPlacedObject)
        self.assertEqual(preview.object_id, "rotated-half-chair")
        self.assertEqual(preview.world_position, (10.0, 20.0, 30.0))
        self.assertEqual(preview.rotation_degrees, (0.0, 0.0, 90.0))
        self.assertEqual(preview.symmetric_preview_orientation, "vertical")
        self.assertEqual(preview.symmetric_preview_plane_coordinate, 0.0)
        self.assertEqual(len(preview.meshes), 1)
        np.testing.assert_allclose(
            preview.meshes[0].bounds,
            retained_model.mesh.bounds,
            atol=1e-7,
        )

        retained_world = preview.meshes[0].copy()
        retained_world.apply_transform(preview.placement_transform)
        mirrored_world = preview.meshes[0].copy()
        mirrored_world.vertices[:, 0] = -mirrored_world.vertices[:, 0]
        mirrored_world.apply_transform(preview.placement_transform)
        np.testing.assert_allclose(
            retained_world.bounds,
            ((9.0, 19.5, 30.0), (11.0, 20.5, 32.0)),
            atol=1e-7,
        )
        np.testing.assert_allclose(
            mirrored_world.bounds,
            ((9.0, 20.5, 30.0), (11.0, 21.5, 32.0)),
            atol=1e-7,
        )

        exported = import_generated_glb(composed.glb_bytes)
        self.assertEqual(
            len(exported.mesh.faces),
            len(base_model.mesh.faces) + retained_face_count,
        )

    def test_bottom_center_moves_to_world_target_and_exports_exact_bounds(
        self,
    ) -> None:
        base_model = _base_box_model()
        object_mesh = trimesh.creation.box(extents=(2.0, 4.0, 6.0))
        object_mesh.apply_translation((4.0, -2.0, 3.0))
        object_model = _single_mesh_model(object_mesh)
        original_payload = bytes(object_model.glb_bytes)
        original_vertices = np.asarray(object_model.mesh.vertices).copy()
        original_edges = copy.deepcopy(object_model.scene.graph.to_edgelist())

        composed = compose_placed_generated_models(
            base_model,
            [
                PlacedGeneratedModel(
                    object_id="chair",
                    model=object_model,
                    world_position=(10.0, 20.0, 30.0),
                )
            ],
        )

        placed_mesh = _placed_world_mesh(composed.scene)
        np.testing.assert_allclose(
            placed_mesh.bounds,
            ((9.0, 18.0, 30.0), (11.0, 22.0, 36.0)),
            atol=1e-7,
        )
        np.testing.assert_allclose(
            composed.mesh.bounds,
            ((-1.0, -1.0, 0.0), (11.0, 22.0, 36.0)),
            atol=1e-7,
        )
        exported = import_generated_glb(composed.glb_bytes)
        np.testing.assert_allclose(
            exported.mesh.bounds,
            composed.mesh.bounds,
            atol=1e-7,
        )
        self.assertEqual(
            len(composed.mesh.faces),
            len(base_model.mesh.faces) + len(object_model.mesh.faces),
        )
        self.assertEqual(object_model.glb_bytes, original_payload)
        np.testing.assert_array_equal(object_model.mesh.vertices, original_vertices)
        self.assertEqual(object_model.scene.graph.to_edgelist(), original_edges)
        self.assertEqual(len(object_model.scene.geometry), 1)

    def test_hierarchy_instancing_and_duplicate_names_remain_unique(
        self,
    ) -> None:
        base_model = _base_box_model(
            geometry_name="placed_1_chair_geometry_shared_name",
            node_name="placed_1_chair_root",
        )
        object_model = _hierarchical_model()

        composed = compose_placed_generated_models(
            base_model,
            (
                PlacedGeneratedModel("chair", object_model, (0.0, 0.0, 0.0)),
                PlacedGeneratedModel("chair", object_model, (20.0, 0.0, 0.0)),
            ),
        )

        geometry_names = list(composed.scene.geometry)
        node_names = list(composed.scene.graph.nodes)
        self.assertEqual(len(geometry_names), len(set(geometry_names)))
        self.assertEqual(len(node_names), len(set(node_names)))
        self.assertEqual(len(geometry_names), 3)
        self.assertIn("placed_1_chair_geometry_shared_name", geometry_names)
        self.assertTrue(
            any(name.endswith("_2") for name in geometry_names)
        )
        self.assertEqual(len(composed.scene.graph.nodes_geometry), 5)

        first_nodes = [
            node
            for node in composed.scene.graph.nodes_geometry
            if str(node).startswith("placed_1_")
            and str(node) != "placed_1_chair_root"
        ]
        second_nodes = [
            node
            for node in composed.scene.graph.nodes_geometry
            if str(node).startswith("placed_2_")
        ]
        self.assertEqual(len(first_nodes), 2)
        self.assertEqual(len(second_nodes), 2)
        first_centers = sorted(
            (_node_world_mesh(composed.scene, node).centroid for node in first_nodes),
            key=lambda center: float(center[1]),
        )
        second_centers = sorted(
            (_node_world_mesh(composed.scene, node).centroid for node in second_nodes),
            key=lambda center: float(center[1]),
        )
        self.assertAlmostEqual(first_centers[1][1] - first_centers[0][1], 6.0)
        np.testing.assert_allclose(
            np.asarray(second_centers) - np.asarray(first_centers),
            np.array(((20.0, 0.0, 0.0), (20.0, 0.0, 0.0))),
            atol=1e-7,
        )

        loaded_scene = trimesh.load(
            BytesIO(composed.glb_bytes),
            file_type="glb",
            force="scene",
            process=False,
        )
        self.assertIsInstance(loaded_scene, trimesh.Scene)
        assert isinstance(loaded_scene, trimesh.Scene)
        self.assertEqual(len(loaded_scene.geometry), 3)
        self.assertEqual(len(loaded_scene.graph.nodes_geometry), 5)

    def test_textures_and_viewer_preview_parts_are_preserved(self) -> None:
        base_untextured = trimesh.creation.box(extents=(2.0, 2.0, 1.0))
        base_untextured.apply_translation((0.0, 0.0, 0.5))
        base_textured = _textured_quad((10, 120, 230, 255))
        base_textured.apply_translation((0.0, 0.0, 1.0))
        base_scene = trimesh.Scene()
        base_scene.add_geometry(
            _to_gltf_mesh(base_untextured),
            geom_name="house",
            node_name="house",
        )
        base_scene.add_geometry(
            _to_gltf_mesh(base_textured),
            geom_name="house_texture",
            node_name="house_texture",
        )
        base_model = import_generated_glb(
            bytes(base_scene.export(file_type="glb"))
        )
        base_surface = PreviewTexturedSurface(
            surface_id="base:surface",
            surface_type="wall",
            mesh=base_textured,
        )
        base_wall = PreviewTexturedWall(
            level_index=0,
            room_index=0,
            wall_key="1:2",
            start_point=(0.0, 0.0, 0.0),
            end_point=(1.0, 0.0, 0.0),
            height_meters=1.0,
            texture_rgba=np.zeros((2, 2, 4), dtype=np.uint8),
        )
        base_model.preview_textured_surfaces = [base_surface]
        base_model.preview_textured_walls = [base_wall]
        base_model.preview_untextured_mesh = base_untextured
        object_model = _mixed_texture_model()
        source_payload = bytes(object_model.glb_bytes)

        composed = compose_placed_generated_models(
            base_model,
            [PlacedGeneratedModel("lamp", object_model, (5.0, 6.0, 7.0))],
        )

        self.assertEqual(object_model.glb_bytes, source_payload)
        self.assertEqual(composed.preview_textured_walls, [base_wall])
        self.assertEqual(len(composed.preview_textured_surfaces), 2)
        self.assertIs(composed.preview_textured_surfaces[0], base_surface)
        placed_surface = composed.preview_textured_surfaces[1]
        self.assertEqual(placed_surface.surface_type, "generated_object")
        self.assertEqual(placed_surface.mesh.visual.kind, "texture")
        np.testing.assert_allclose(
            _texture_average(placed_surface.mesh),
            (220.0, 20.0, 30.0, 255.0),
            atol=1.0,
        )
        self.assertIsNotNone(composed.preview_untextured_mesh)
        assert composed.preview_untextured_mesh is not None
        self.assertEqual(
            len(composed.preview_untextured_mesh.faces),
            len(base_untextured.faces) + 12,
        )

        loaded_scene = trimesh.load(
            BytesIO(composed.glb_bytes),
            file_type="glb",
            force="scene",
            process=False,
        )
        assert isinstance(loaded_scene, trimesh.Scene)
        textured_meshes = [
            mesh
            for mesh in loaded_scene.geometry.values()
            if getattr(mesh.visual, "kind", None) == "texture"
        ]
        self.assertEqual(len(textured_meshes), 2)
        texture_averages = sorted(
            (_texture_average(mesh) for mesh in textured_meshes),
            key=lambda average: float(average[0]),
        )
        np.testing.assert_allclose(
            texture_averages[0],
            (10.0, 120.0, 230.0, 255.0),
            atol=1.0,
        )
        np.testing.assert_allclose(
            texture_averages[1],
            (220.0, 20.0, 30.0, 255.0),
            atol=1.0,
        )

    def test_zero_placements_and_invalid_inputs_are_deterministic(self) -> None:
        base_model = _base_box_model()
        object_model = _single_mesh_model(trimesh.creation.icosphere())

        self.assertIs(
            compose_placed_generated_models(base_model, ()),
            base_model,
        )
        with self.assertRaises(TypeError):
            compose_placed_generated_models("invalid", ())  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            compose_placed_generated_models(base_model, None)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            compose_placed_generated_models(
                base_model,
                [object_model],  # type: ignore[list-item]
            )
        with self.assertRaises(TypeError):
            PlacedGeneratedModel(1, object_model, (0, 0, 0))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            PlacedGeneratedModel(" ", object_model, (0, 0, 0))
        with self.assertRaises(TypeError):
            PlacedGeneratedModel("id", object(), (0, 0, 0))  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            PlacedGeneratedModel("id", object_model, "0,0,0")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            PlacedGeneratedModel("id", object_model, (0, 0))  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            PlacedGeneratedModel("id", object_model, (True, 0, 0))
        with self.assertRaises(TypeError):
            PlacedGeneratedModel("id", object_model, ("x", 0, 0))  # type: ignore[arg-type]
        for invalid_coordinate in (math.nan, math.inf, -math.inf):
            with self.subTest(invalid_coordinate=invalid_coordinate):
                with self.assertRaises(ValueError):
                    PlacedGeneratedModel(
                        "id",
                        object_model,
                        (invalid_coordinate, 0.0, 0.0),
                    )

        for invalid_rotation in (
            "0,0,0",
            (0.0, 0.0),
            (0.0, True, 0.0),
            (0.0, "1.0", 0.0),
            (math.nan, 0.0, 0.0),
            (0.0, math.inf, 0.0),
        ):
            with (
                self.subTest(invalid_rotation=invalid_rotation),
                self.assertRaises((TypeError, ValueError)),
            ):
                PlacedGeneratedModel(
                    "id",
                    object_model,
                    (0.0, 0.0, 0.0),
                    rotation_degrees=invalid_rotation,
                )

        with self.assertRaises(ValueError):
            PlacedGeneratedModel(
                "id",
                object_model,
                (0, 0, 0),
                symmetric_preview_orientation="vertical",
            )
        with self.assertRaises(ValueError):
            PlacedGeneratedModel(
                "id",
                object_model,
                (0, 0, 0),
                symmetric_preview_plane_coordinate=0.0,
            )
        with self.assertRaises(ValueError):
            PlacedGeneratedModel(
                "id",
                object_model,
                (0, 0, 0),
                symmetric_preview_orientation="diagonal",
                symmetric_preview_plane_coordinate=0.0,
            )
        with self.assertRaises(ValueError):
            PlacedGeneratedModel(
                "id",
                object_model,
                (0, 0, 0),
                symmetric_preview_orientation="horizontal",
                symmetric_preview_plane_coordinate=math.nan,
            )

        empty_model = GeneratedModel(
            mesh=trimesh.Trimesh(process=False),
            scene=trimesh.Scene(),
            glb_bytes=b"empty",
        )
        with self.assertRaises(ValueError):
            compose_placed_generated_models(
                base_model,
                [PlacedGeneratedModel("empty", empty_model, (0, 0, 0))],
            )


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
