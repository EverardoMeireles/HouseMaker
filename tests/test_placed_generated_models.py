# ### Imports ###
from __future__ import annotations

import copy
import json
import math
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import trimesh
from PIL import Image
from trimesh.exchange.gltf import load_glb
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
    _serialize_scene_glb_with_half_mesh_extras,
    compose_placed_generated_models,
    export_glb_file,
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


def _load_glb_json(payload: bytes) -> dict[str, object]:
    """Read the JSON chunk without depending on scene-import metadata policy."""

    if payload[:4] != b"glTF" or len(payload) < 20:
        raise ValueError("The fixture GLB is invalid.")
    offset = 12
    while offset + 8 <= len(payload):
        chunk_length = int.from_bytes(payload[offset : offset + 4], "little")
        chunk_type = payload[offset + 4 : offset + 8]
        offset += 8
        chunk = payload[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type == b"JSON":
            document = json.loads(chunk.rstrip(b" \x00").decode("utf-8"))
            if not isinstance(document, dict):
                raise ValueError("The fixture GLB JSON root is invalid.")
            return document
    raise ValueError("The fixture GLB has no JSON chunk.")


def _build_json_only_glb(document: dict[str, object]) -> bytes:
    """Serialize a minimal JSON-only GLB for final-byte rewrite tests."""

    encoded = json.dumps(document, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 4)
    total_byte_count = 12 + 8 + len(encoded)
    return b"".join(
        (
            b"glTF",
            (2).to_bytes(4, "little"),
            total_byte_count.to_bytes(4, "little"),
            len(encoded).to_bytes(4, "little"),
            b"JSON",
            encoded,
        )
    )


def _find_exported_node(
    payload: bytes,
    node_name: str,
) -> dict[str, object]:
    document = _load_glb_json(payload)
    nodes = document.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("The fixture GLB has no nodes.")
    matches = [
        node
        for node in nodes
        if isinstance(node, dict) and node.get("name") == node_name
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one exported node named {node_name!r}.")
    return matches[0]


def _assert_direct_half_mesh_nodes(
    payload: bytes,
    expected_half_mesh_by_name: dict[str, dict[str, object]],
) -> None:
    """Require every half marker to live directly on one root mesh node."""

    document = _load_glb_json(payload)
    nodes = document.get("nodes")
    meshes = document.get("meshes")
    scenes = document.get("scenes")
    if (
        not isinstance(nodes, list)
        or not isinstance(meshes, list)
        or not isinstance(scenes, list)
    ):
        raise AssertionError("The exported GLB has no node hierarchy.")
    half_node_indices = {
        index
        for index, node in enumerate(nodes)
        if isinstance(node, dict)
        and str(node.get("name", "")).startswith("[HALF] ")
    }
    actual_names = {
        str(nodes[index].get("name")) for index in half_node_indices
    }
    if actual_names != set(expected_half_mesh_by_name):
        raise AssertionError("The exported half-mesh node names are invalid.")
    metadata_node_indices = {
        index
        for index, node in enumerate(nodes)
        if isinstance(node, dict)
        and isinstance(node.get("extras"), dict)
        and "halfMesh" in node["extras"]
    }
    if metadata_node_indices != half_node_indices:
        raise AssertionError("Half-mesh metadata depends on a wrapper node.")

    scene_index = int(document.get("scene", 0))
    active_scene = scenes[scene_index]
    if not isinstance(active_scene, dict):
        raise AssertionError("The exported GLB scene is invalid.")
    root_node_indices = set(active_scene.get("nodes", ()))
    if not half_node_indices.issubset(root_node_indices):
        raise AssertionError("A half mesh depends on a separate parent node.")

    for index in half_node_indices:
        node = nodes[index]
        assert isinstance(node, dict)
        name = str(node["name"])
        if "mesh" not in node or "children" in node:
            raise AssertionError("A half marker is not on a leaf mesh node.")
        expected_half_mesh = expected_half_mesh_by_name[name]
        extras = node.get("extras")
        if (
            not isinstance(extras, dict)
            or extras.get("halfMesh") != expected_half_mesh
        ):
            raise AssertionError("A half mesh has invalid mirror instructions.")
        mesh_index = node["mesh"]
        if (
            isinstance(mesh_index, bool)
            or not isinstance(mesh_index, int)
            or mesh_index < 0
            or mesh_index >= len(meshes)
            or not isinstance(meshes[mesh_index], dict)
        ):
            raise AssertionError("A half mesh has no glTF mesh definition.")
        mesh_extras = meshes[mesh_index].get("extras")
        if (
            not isinstance(mesh_extras, dict)
            or mesh_extras.get("halfMesh") != expected_half_mesh
        ):
            raise AssertionError(
                "A renderable half mesh has no mirror instructions."
            )
        if set(expected_half_mesh) != {"mirrorPlane", "uvMode"}:
            raise AssertionError("The expected half-mesh schema is invalid.")
        if expected_half_mesh["uvMode"] != "reuse":
            raise AssertionError("The mirrored half must reuse authored UVs.")
        mirror_plane = expected_half_mesh["mirrorPlane"]
        if not isinstance(mirror_plane, dict) or set(mirror_plane) != {
            "point",
            "normal",
        }:
            raise AssertionError("The half-mesh mirror plane is invalid.")


def _metadata_preserving_trimesh_round_trip(payload: bytes) -> bytes:
    """Exercise Trimesh's glTF parser while retaining parsed graph extras."""

    loaded = load_glb(BytesIO(payload))
    scene = trimesh.Scene(base_frame=str(loaded["base_frame"]))
    for geometry_name, mesh_kwargs in loaded["geometry"].items():
        scene.geometry[geometry_name] = trimesh.Trimesh(**mesh_kwargs)
    for raw_edge in loaded["graph"]:
        scene.graph.update(**dict(raw_edge))
    return bytes(scene.export(file_type="glb"))


# ### Half-mesh serialization tests ###
class HalfMeshGlbSerializationTests(unittest.TestCase):
    def _build_half_mesh_scene(
        self,
    ) -> tuple[trimesh.Scene, str, dict[str, object]]:
        node_name = "[HALF] serializer_fixture"
        half_mesh = {
            "mirrorPlane": {
                "point": [1.0, 2.0, 3.0],
                "normal": [1.0, 0.0, 0.0],
            },
            "uvMode": "reuse",
        }
        scene = trimesh.Scene()
        scene.add_geometry(
            trimesh.creation.box(),
            node_name=node_name,
            metadata={"halfMesh": half_mesh},
        )
        return scene, node_name, half_mesh

    def test_serializer_injects_extras_when_gltf_tree_omits_them(
        self,
    ) -> None:
        scene, node_name, half_mesh = self._build_half_mesh_scene()
        source_tree = {
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"name": node_name, "mesh": 0}],
            "meshes": [{"primitives": []}],
        }

        def export_without_implicit_extras(
            *,
            file_type: str,
        ) -> bytes:
            self.assertEqual(file_type, "glb")
            return _build_json_only_glb(source_tree)

        with patch.object(
            scene,
            "export",
            side_effect=export_without_implicit_extras,
        ):
            payload = _serialize_scene_glb_with_half_mesh_extras(
                scene,
                failure_message="serialization failed",
            )

        exported_tree = _load_glb_json(payload)
        self.assertEqual(
            exported_tree["nodes"][0]["extras"],
            {"halfMesh": half_mesh},
        )
        self.assertEqual(
            exported_tree["meshes"][0]["extras"],
            {"halfMesh": half_mesh},
        )

    def test_serializer_rejects_missing_or_non_mesh_half_target(self) -> None:
        invalid_nodes = (
            ({"name": "ordinary", "mesh": 0}, "omitted half-mesh"),
            ({"name": "[HALF] serializer_fixture"}, "mesh node"),
        )
        for raw_node, expected_message in invalid_nodes:
            with self.subTest(raw_node=raw_node):
                scene, _node_name, _half_mesh = self._build_half_mesh_scene()

                def export_invalid_tree(
                    *,
                    file_type: str,
                ) -> bytes:
                    self.assertEqual(file_type, "glb")
                    return _build_json_only_glb(
                        {
                            "asset": {"version": "2.0"},
                            "nodes": [dict(raw_node)],
                            "meshes": [{"primitives": []}],
                        }
                    )

                with patch.object(
                    scene,
                    "export",
                    side_effect=export_invalid_tree,
                ), self.assertRaisesRegex(ValueError, expected_message):
                    _serialize_scene_glb_with_half_mesh_extras(
                        scene,
                        failure_message="serialization failed",
                    )

    def test_final_file_writer_repairs_stripped_half_mesh_extras(self) -> None:
        scene, node_name, half_mesh = self._build_half_mesh_scene()
        stripped_payload = _build_json_only_glb(
            {
                "asset": {"version": "2.0"},
                "scene": 0,
                "scenes": [{"nodes": [0]}],
                "nodes": [{"name": node_name, "mesh": 0}],
                "meshes": [{"primitives": []}],
            }
        )
        model = GeneratedModel(
            mesh=trimesh.creation.box(),
            scene=scene,
            glb_bytes=stripped_payload,
        )

        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "final.glb"
            export_glb_file(model, output_path)
            document = _load_glb_json(output_path.read_bytes())

        self.assertEqual(
            document["nodes"][0]["extras"]["halfMesh"],
            half_mesh,
        )
        self.assertEqual(
            document["meshes"][0]["extras"]["halfMesh"],
            half_mesh,
        )


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

    def test_multiple_half_models_export_as_distinct_named_authored_objects(
        self,
    ) -> None:
        base_model = _base_box_model()
        retained_mesh = trimesh.creation.box(extents=(1.0, 2.0, 2.0))
        retained_mesh.apply_translation((-0.5, 0.0, 1.0))
        retained_model = _single_mesh_model(retained_mesh)
        retained_face_count = len(retained_model.mesh.faces)

        composed = compose_placed_generated_models(
            base_model,
            (
                PlacedGeneratedModel(
                    object_id="half-window",
                    object_name="  window_frame_01  ",
                    model=retained_model,
                    world_position=(10.0, 20.0, 30.0),
                    rotation_degrees=(0.0, 0.0, 90.0),
                    symmetric_preview_orientation="vertical",
                    symmetric_preview_plane_coordinate=0.0,
                ),
                PlacedGeneratedModel(
                    object_id="half-arch",
                    object_name="arch_frame_01",
                    model=retained_model,
                    world_position=(40.0, 50.0, 60.0),
                    symmetric_preview_orientation="horizontal",
                    symmetric_preview_plane_coordinate=1.0,
                ),
            ),
        )

        document = _load_glb_json(composed.glb_bytes)
        nodes = document.get("nodes")
        self.assertIsInstance(nodes, list)
        assert isinstance(nodes, list)
        half_node_names = {
            str(node.get("name"))
            for node in nodes
            if isinstance(node, dict)
            and str(node.get("name", "")).startswith("[HALF] ")
        }
        self.assertEqual(
            half_node_names,
            {"[HALF] window_frame_01", "[HALF] arch_frame_01"},
        )
        half_mesh_nodes = [
            node
            for node in nodes
            if isinstance(node, dict)
            and isinstance(node.get("extras"), dict)
            and "halfMesh" in node["extras"]
        ]
        self.assertEqual(len(half_mesh_nodes), 2)
        self.assertTrue(all("mesh" in node for node in half_mesh_nodes))
        self.assertTrue(
            all(
                str(node.get("name", "")).startswith("[HALF] ")
                for node in half_mesh_nodes
            )
        )

        vertical_node = _find_exported_node(
            composed.glb_bytes,
            "[HALF] window_frame_01",
        )
        self.assertIn("mesh", vertical_node)
        vertical_half = vertical_node["extras"]["halfMesh"]
        self.assertEqual(set(vertical_half), {"mirrorPlane", "uvMode"})
        self.assertEqual(vertical_half["uvMode"], "reuse")
        self.assertEqual(
            set(vertical_half["mirrorPlane"]),
            {"point", "normal"},
        )
        np.testing.assert_allclose(
            vertical_half["mirrorPlane"]["point"],
            (10.0, 30.0, -20.5),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            vertical_half["mirrorPlane"]["normal"],
            (0.0, 0.0, -1.0),
            atol=1e-12,
        )
        horizontal_node = _find_exported_node(
            composed.glb_bytes,
            "[HALF] arch_frame_01",
        )
        self.assertIn("mesh", horizontal_node)
        horizontal_half = horizontal_node["extras"]["halfMesh"]
        self.assertEqual(set(horizontal_half), {"mirrorPlane", "uvMode"})
        self.assertEqual(horizontal_half["uvMode"], "reuse")
        np.testing.assert_allclose(
            horizontal_half["mirrorPlane"]["point"],
            (40.5, 61.0, -50.0),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            horizontal_half["mirrorPlane"]["normal"],
            (0.0, 1.0, 0.0),
            atol=1e-12,
        )
        _assert_direct_half_mesh_nodes(
            composed.glb_bytes,
            {
                "[HALF] window_frame_01": {
                    "mirrorPlane": {
                        "point": [10.0, 30.0, -20.5],
                        "normal": [0.0, 0.0, -1.0],
                    },
                    "uvMode": "reuse",
                },
                "[HALF] arch_frame_01": {
                    "mirrorPlane": {
                        "point": [40.5, 61.0, -50.0],
                        "normal": [0.0, 1.0, 0.0],
                    },
                    "uvMode": "reuse",
                },
            },
        )

        exported = import_generated_glb(composed.glb_bytes)
        self.assertEqual(
            len(exported.mesh.faces),
            len(base_model.mesh.faces) + retained_face_count * 2,
        )

    def test_half_name_falls_back_to_id_and_non_half_export_stays_normal(
        self,
    ) -> None:
        base_model = _base_box_model()
        object_model = _single_mesh_model(trimesh.creation.icosphere())

        composed = compose_placed_generated_models(
            base_model,
            (
                PlacedGeneratedModel(
                    object_id="fallback_half",
                    model=object_model,
                    world_position=(2.0, 3.0, 4.0),
                    symmetric_preview_orientation="vertical",
                    symmetric_preview_plane_coordinate=0.0,
                ),
                PlacedGeneratedModel(
                    object_id="ordinary-object",
                    object_name="ordinary_display_name",
                    model=object_model,
                    world_position=(5.0, 6.0, 7.0),
                ),
            ),
        )

        half_node = _find_exported_node(
            composed.glb_bytes,
            "[HALF] fallback_half",
        )
        self.assertIn("mesh", half_node)
        self.assertIn("halfMesh", half_node["extras"])
        document = _load_glb_json(composed.glb_bytes)
        nodes = document.get("nodes")
        self.assertIsInstance(nodes, list)
        assert isinstance(nodes, list)
        ordinary_nodes = [
            node
            for node in nodes
            if isinstance(node, dict)
            and isinstance(node.get("extras"), dict)
            and node["extras"].get("housemaker_object_id")
            == "ordinary-object"
        ]
        self.assertEqual(len(ordinary_nodes), 1)
        ordinary_node = ordinary_nodes[0]
        self.assertFalse(str(ordinary_node.get("name", "")).startswith("[HALF]"))
        self.assertNotIn("halfMesh", ordinary_node["extras"])

    def test_half_extras_survive_gltf_parser_and_reexport(self) -> None:
        base_model = _base_box_model()
        retained_mesh = trimesh.creation.box(extents=(1.0, 2.0, 2.0))
        retained_mesh.apply_translation((-0.5, 0.0, 1.0))
        retained_model = _single_mesh_model(retained_mesh)
        composed = compose_placed_generated_models(
            base_model,
            (
                PlacedGeneratedModel(
                    object_id="round-trip-half",
                    object_name="round_trip_half",
                    model=retained_model,
                    world_position=(1.0, 2.0, 3.0),
                    symmetric_preview_orientation="vertical",
                    symmetric_preview_plane_coordinate=0.0,
                ),
            ),
        )
        expected_half_mesh = {
            "mirrorPlane": {
                "point": [1.5, 3.0, -2.0],
                "normal": [1.0, 0.0, 0.0],
            },
            "uvMode": "reuse",
        }
        _assert_direct_half_mesh_nodes(
            composed.glb_bytes,
            {"[HALF] round_trip_half": expected_half_mesh},
        )

        loaded = load_glb(BytesIO(composed.glb_bytes))
        half_edges = [
            dict(edge)
            for edge in loaded["graph"]
            if edge.get("frame_to") == "[HALF] round_trip_half"
        ]
        self.assertEqual(len(half_edges), 1)
        self.assertIn("geometry", half_edges[0])
        self.assertEqual(
            half_edges[0]["metadata"]["halfMesh"],
            expected_half_mesh,
        )

        round_trip_payload = _metadata_preserving_trimesh_round_trip(
            composed.glb_bytes
        )
        round_trip_node = _find_exported_node(
            round_trip_payload,
            "[HALF] round_trip_half",
        )
        self.assertIn("mesh", round_trip_node)
        self.assertEqual(
            round_trip_node["extras"]["halfMesh"],
            half_edges[0]["metadata"]["halfMesh"],
        )
        _assert_direct_half_mesh_nodes(
            round_trip_payload,
            {"[HALF] round_trip_half": expected_half_mesh},
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
