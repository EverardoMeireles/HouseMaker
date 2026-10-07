# ### Imports ###
from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from housemaker.door_geometry import (
    DoorBodyMirrorConfiguration,
    assemble_door_model,
    build_default_door_slot_models,
)
from housemaker.door_state import (
    DOOR_SLOT_KNOB,
    create_door_definition_for_doorway,
)
from housemaker.glb import (
    DOOR_COMPONENT_MARKER_METADATA_KEY,
    GeneratedModel,
    PlacedGeneratedModel,
    build_gltf_mirror_plane_from_z_up_transform,
    build_placed_generated_model_gltf_mirror_plane,
    build_placed_generated_model_gltf_transform,
    build_placed_generated_model_transform,
    compose_generated_model_instance_sources,
    compose_placed_generated_models,
    export_glb_file,
)
from housemaker.models import DoorwayData
from housemaker.scene_export import (
    INSTANCE_SOURCE_SCENE_NAME,
    RUNTIME_SCENE_FORMAT,
    RUNTIME_SCENE_VERSION,
    DoorBodyReconstruction,
    build_runtime_scene_manifest,
    write_runtime_scene_manifest,
)


# ### Fixture helpers ###
def _source_model() -> GeneratedModel:
    mesh = trimesh.creation.box(extents=(2.0, 1.0, 1.0))
    scene = trimesh.Scene(mesh)
    payload = scene.export(file_type="glb")
    assert isinstance(payload, bytes)
    return GeneratedModel(mesh=mesh, scene=scene, glb_bytes=payload)


def _source_model_with_node_name(node_name: str) -> GeneratedModel:
    mesh = trimesh.creation.box(extents=(2.0, 1.0, 1.0))
    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name=node_name, node_name=node_name)
    payload = scene.export(file_type="glb")
    assert isinstance(payload, bytes)
    return GeneratedModel(mesh=mesh, scene=scene, glb_bytes=payload)


def _empty_model() -> GeneratedModel:
    return GeneratedModel(
        mesh=trimesh.Trimesh(process=False),
        scene=trimesh.Scene(),
        glb_bytes=b"",
    )


def _model_without_scene_geometry() -> GeneratedModel:
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    return GeneratedModel(
        mesh=mesh,
        scene=trimesh.Scene(),
        glb_bytes=b"",
    )


def _matrix_from_column_major(values: object) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape((4, 4)).T


def _read_glb_document(payload: bytes) -> dict[str, object]:
    json_byte_count = struct.unpack_from("<I", payload, 12)[0]
    return json.loads(payload[20 : 20 + json_byte_count].rstrip(b" \0"))


def _doorway() -> DoorwayData:
    return DoorwayData(
        doorway_id="doorway-a",
        center_x=0.0,
        center_y=0.0,
        width_meters=1.0,
        height_meters=2.2,
        depth_meters=0.2,
    )


# ### GLB mirror-plane tests ###
class GltfMirrorPlaneTests(unittest.TestCase):
    def test_arbitrary_z_up_transform_maps_point_and_normal_to_gltf_world(
        self,
    ) -> None:
        translation = np.eye(4, dtype=float)
        translation[:3, 3] = (4.0, -3.0, 2.0)
        rotation = trimesh.transformations.rotation_matrix(
            np.pi / 2.0,
            (0.0, 0.0, 1.0),
        )
        scale = np.diag((2.0, 3.0, 4.0, 1.0))

        mirror_plane = build_gltf_mirror_plane_from_z_up_transform(
            translation @ rotation @ scale,
            axis=0,
            plane_coordinate=0.5,
        )

        np.testing.assert_allclose(
            mirror_plane["point"],
            (4.0, 2.0, 2.0),
            atol=1e-9,
        )
        np.testing.assert_allclose(
            mirror_plane["normal"],
            (0.0, 0.0, -1.0),
            atol=1e-9,
        )

    def test_placed_model_wrapper_uses_the_general_transform_helper(self) -> None:
        placement = PlacedGeneratedModel(
            object_id="nested-door-body",
            model=_source_model(),
            world_position=(3.0, -2.0, 1.5),
            rotation_degrees=(10.0, 20.0, 30.0),
            scale=1.25,
            axis_scales=(0.5, 2.0, 1.5),
        )
        transform = build_placed_generated_model_transform(placement)

        self.assertEqual(
            build_placed_generated_model_gltf_mirror_plane(
                placement,
                axis=1,
                plane_coordinate=0.2,
            ),
            build_gltf_mirror_plane_from_z_up_transform(
                transform,
                axis=1,
                plane_coordinate=0.2,
            ),
        )

    def test_transform_helper_rejects_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "axis 0, 1, or 2"):
            build_gltf_mirror_plane_from_z_up_transform(
                np.eye(4, dtype=float),
                axis=3,
                plane_coordinate=0.0,
            )
        with self.assertRaisesRegex(ValueError, "coordinates must be finite"):
            build_gltf_mirror_plane_from_z_up_transform(
                np.eye(4, dtype=float),
                axis=0,
                plane_coordinate=float("nan"),
            )
        with self.assertRaisesRegex(ValueError, "finite 4 by 4 matrix"):
            build_gltf_mirror_plane_from_z_up_transform(
                np.eye(3, dtype=float),
                axis=0,
                plane_coordinate=0.0,
            )


# ### Runtime manifest tests ###
class RuntimeSceneManifestTests(unittest.TestCase):
    def test_manifest_treats_the_source_placement_as_an_instance(self) -> None:
        source_model = _source_model()
        source = PlacedGeneratedModel(
            object_id="chair",
            model=source_model,
            world_position=(1.0, 2.0, 0.5),
            rotation_degrees=(0.0, 0.0, 15.0),
        )
        instance = PlacedGeneratedModel(
            object_id="instance-chair-1",
            source_object_id="chair",
            object_name="chair",
            model=source_model,
            world_position=(4.0, -2.0, 1.0),
            rotation_degrees=(0.0, 0.0, 60.0),
            scale=1.25,
        )
        payload = build_runtime_scene_manifest(
            glb_name="house.glb",
            glb_bytes=b"glb",
            source_placements={"chair": source},
            instance_placements=(instance,),
        )

        self.assertEqual(payload["format"], RUNTIME_SCENE_FORMAT)
        self.assertEqual(payload["version"], RUNTIME_SCENE_VERSION)
        self.assertEqual(payload["version"], 8)
        self.assertEqual(payload["coordinateSystem"], "gltf-y-up")
        self.assertEqual(payload["matrixLayout"], "column-major")
        self.assertEqual(payload["asset"]["glb"], "house.glb")
        self.assertEqual(
            payload["asset"]["sha256"],
            hashlib.sha256(b"glb").hexdigest(),
        )
        group = payload["instanceGroups"][0]
        self.assertEqual(
            set(group),
            {"sourceNodeName", "instances"},
        )
        self.assertEqual(group["sourceNodeName"], "chair")
        self.assertEqual(payload["doorBodyReconstructions"], [])
        self.assertEqual(
            [serialized["id"] for serialized in group["instances"]],
            ["chair", "instance-chair-1"],
        )
        self.assertTrue(
            all(
                set(serialized) == {"id", "worldMatrix"}
                for serialized in group["instances"]
            )
        )
        np.testing.assert_allclose(
            _matrix_from_column_major(group["instances"][0]["worldMatrix"]),
            build_placed_generated_model_gltf_transform(source),
            atol=1e-9,
        )
        np.testing.assert_allclose(
            _matrix_from_column_major(group["instances"][1]["worldMatrix"]),
            build_placed_generated_model_gltf_transform(instance),
            atol=1e-9,
        )

    def test_manifest_uses_only_real_instances_for_an_unplaced_source(
        self,
    ) -> None:
        source_model = _source_model()
        instance_a = PlacedGeneratedModel(
            object_id="chair-copy-a",
            source_object_id="chair",
            object_name="chair",
            model=source_model,
            world_position=(4.0, -2.0, 1.0),
            rotation_degrees=(0.0, 0.0, 60.0),
        )
        instance_b = PlacedGeneratedModel(
            object_id="chair-copy-b",
            source_object_id="chair",
            object_name="chair",
            model=source_model,
            world_position=(-3.0, 5.0, 0.25),
            scale=1.5,
        )
        payload = build_runtime_scene_manifest(
            glb_name="house.glb",
            glb_bytes=b"glb",
            source_placements={},
            instance_placements=(instance_b, instance_a),
        )

        group = payload["instanceGroups"][0]
        self.assertEqual(
            [entry["id"] for entry in group["instances"]],
            ["chair-copy-a", "chair-copy-b"],
        )
        self.assertNotIn(
            "chair",
            [entry["id"] for entry in group["instances"]],
        )
        self.assertEqual(group["sourceNodeName"], "chair")
        for placement, serialized in zip(
            (instance_a, instance_b),
            group["instances"],
            strict=True,
        ):
            np.testing.assert_allclose(
                _matrix_from_column_major(serialized["worldMatrix"]),
                build_placed_generated_model_gltf_transform(placement),
                atol=1e-9,
            )

    def test_manifest_marks_a_placed_half_mesh_instance_group(self) -> None:
        source_model = _source_model()
        source = PlacedGeneratedModel(
            object_id="half-chair",
            object_name="half-chair",
            model=source_model,
            world_position=(1.0, 2.0, 0.5),
            symmetric_preview_orientation="vertical",
            symmetric_preview_plane_coordinate=0.25,
        )
        instance = PlacedGeneratedModel(
            object_id="half-chair-copy",
            source_object_id="half-chair",
            object_name="half-chair",
            model=source_model,
            world_position=(4.0, -2.0, 1.0),
            symmetric_preview_orientation="vertical",
            symmetric_preview_plane_coordinate=0.25,
        )

        payload = build_runtime_scene_manifest(
            glb_name="house.glb",
            glb_bytes=b"glb",
            source_placements={"half-chair": source},
            instance_placements=(instance,),
        )

        group = payload["instanceGroups"][0]
        self.assertEqual(
            set(group),
            {"sourceNodeName", "halfMesh", "instances"},
        )
        self.assertIs(group["halfMesh"], True)
        self.assertEqual(group["sourceNodeName"], "half-chair")

    def test_manifest_marks_an_unplaced_half_mesh_instance_group(self) -> None:
        source_model = _source_model()
        instance = PlacedGeneratedModel(
            object_id="half-table-copy",
            source_object_id="half-table",
            object_name="half-table",
            model=source_model,
            world_position=(4.0, -2.0, 1.0),
            symmetric_preview_orientation="vertical",
            symmetric_preview_plane_coordinate=0.0,
        )

        payload = build_runtime_scene_manifest(
            glb_name="house.glb",
            glb_bytes=b"glb",
            source_placements={},
            instance_placements=(instance,),
        )

        group = payload["instanceGroups"][0]
        self.assertIs(group["halfMesh"], True)
        self.assertEqual(group["sourceNodeName"], "half-table")

    def test_manifest_exports_door_side_duplication_after_half_mesh(self) -> None:
        reconstruction = DoorBodyReconstruction(
            placement_object_id="door-placement:placement-a",
            body_object_id="door:door-a:slot:body",
            kept_side="front",
            mirror_point=(2.0, 1.0, -3.0),
            mirror_normal=(0.0, 0.0, -1.0),
            mirrored_component_object_ids=(
                "door:door-a:slot:door_knob",
            ),
        )

        payload = build_runtime_scene_manifest(
            glb_name="house.glb",
            glb_bytes=b"glb",
            source_placements={},
            instance_placements=(),
            door_body_reconstructions=(reconstruction,),
        )

        self.assertEqual(
            payload["doorBodyReconstructions"],
            [
                {
                    "placementObjectId": "door-placement:placement-a",
                    "bodyObjectId": "door:door-a:slot:body",
                    "sideDuplication": {
                        "keptSide": "front",
                        "mirrorPlane": {
                            "point": [2.0, 1.0, -3.0],
                            "normal": [0.0, 0.0, -1.0],
                        },
                        "uvMode": "reuse",
                        "applyAfter": "halfMesh",
                        "mirroredComponentObjectIds": [
                            "door:door-a:slot:door_knob"
                        ],
                    },
                }
            ],
        )

    def test_manifest_omits_empty_mirrored_component_ids(self) -> None:
        reconstruction = DoorBodyReconstruction(
            placement_object_id="door-placement:placement-a",
            body_object_id="door:door-a:slot:body",
            kept_side="front",
            mirror_point=(0.0, 0.0, 0.0),
            mirror_normal=(0.0, 0.0, 1.0),
        )

        side_duplication = reconstruction.to_runtime_dict()["sideDuplication"]

        self.assertNotIn("mirroredComponentObjectIds", side_duplication)

    def test_export_preserves_nested_door_component_locator(self) -> None:
        component_model = _source_model_with_node_name("door_knob")
        component_node = next(iter(component_model.scene.graph.nodes_geometry))
        component_parent = component_model.scene.graph.transforms.parents[
            component_node
        ]
        component_edge = component_model.scene.graph.transforms.edge_data[
            (component_parent, component_node)
        ]
        component_edge["metadata"] = {
            DOOR_COMPONENT_MARKER_METADATA_KEY: {
                "componentObjectId": "door:door-a:slot:door_knob",
                "kind": "door_knob",
            }
        }
        placed = PlacedGeneratedModel(
            object_id="door-placement:placement-a",
            object_name="Placed door",
            model=component_model,
            world_position=(2.0, 3.0, 0.0),
        )
        complete_scene = compose_placed_generated_models(
            _empty_model(),
            (placed,),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            glb_path = export_glb_file(
                complete_scene,
                Path(temporary_directory) / "scene.glb",
            )
            document = _read_glb_document(glb_path.read_bytes())

        marked_nodes = [
            node
            for node in document["nodes"]
            if isinstance(node, dict)
            and "housemakerDoorComponent" in node.get("extras", {})
        ]
        self.assertEqual(len(marked_nodes), 1)
        self.assertEqual(
            marked_nodes[0]["extras"]["housemakerDoorComponent"],
            {
                "placementObjectId": "door-placement:placement-a",
                "componentObjectId": "door:door-a:slot:door_knob",
                "kind": "door_knob",
            },
        )

    def test_export_preserves_door_body_and_component_locators(self) -> None:
        door = create_door_definition_for_doorway(
            _doorway(),
            door_id="door-a",
            name="New door 1",
        )
        knob = door.get_slot(DOOR_SLOT_KNOB)
        assert knob is not None
        door = door.replace_slot(
            knob.with_transform(
                position_meters=(0.4, -0.04, 1.0),
                joined=True,
            )
        )
        body = door.get_slot("body")
        assert body is not None
        slot_models = build_default_door_slot_models(door)
        slot_models[knob.source_object_id] = _source_model_with_node_name(
            "door_knob"
        )
        assembled = assemble_door_model(
            door,
            slot_models,
            body_mirror=DoorBodyMirrorConfiguration(
                symmetric_orientation="vertical",
                symmetric_plane_coordinate=0.0,
                side_duplication_plane_coordinate=0.0,
            ),
        )
        placed = PlacedGeneratedModel(
            object_id="door-placement:placement-a",
            object_name="Placed door",
            model=assembled,
            world_position=(2.0, 3.0, 0.0),
        )
        complete_scene = compose_placed_generated_models(
            _empty_model(),
            (placed,),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            glb_path = export_glb_file(
                complete_scene,
                Path(temporary_directory) / "scene.glb",
            )
            document = _read_glb_document(glb_path.read_bytes())

        mesh_nodes = [
            node
            for node in document["nodes"]
            if isinstance(node, dict) and "mesh" in node
        ]
        marked_nodes = [
            node
            for node in mesh_nodes
            if "housemakerDoorBody" in node.get("extras", {})
        ]
        self.assertEqual(len(marked_nodes), 1)
        self.assertEqual(
            marked_nodes[0]["extras"]["housemakerDoorBody"],
            {
                "placementObjectId": "door-placement:placement-a",
                "bodyObjectId": body.source_object_id,
            },
        )
        self.assertIn("halfMesh", marked_nodes[0]["extras"])
        component_nodes = [
            node
            for node in mesh_nodes
            if "housemakerDoorComponent" in node.get("extras", {})
        ]
        self.assertEqual(len(component_nodes), 1)
        self.assertEqual(
            component_nodes[0]["extras"]["housemakerDoorComponent"],
            {
                "placementObjectId": "door-placement:placement-a",
                "componentObjectId": knob.source_object_id,
                "kind": DOOR_SLOT_KNOB,
            },
        )
        self.assertTrue(
            all(
                "housemakerDoorBody" not in node.get("extras", {})
                for node in mesh_nodes
                if "door_knob" in str(node.get("name", ""))
            )
        )

    def test_manifest_rejects_mixed_half_mesh_instance_metadata(self) -> None:
        source_model = _source_model()
        source = PlacedGeneratedModel(
            object_id="half-chair",
            object_name="half-chair",
            model=source_model,
            world_position=(1.0, 2.0, 0.5),
            symmetric_preview_orientation="vertical",
            symmetric_preview_plane_coordinate=0.0,
        )
        mismatched_instance = PlacedGeneratedModel(
            object_id="half-chair-copy",
            source_object_id="half-chair",
            object_name="half-chair",
            model=source_model,
            world_position=(4.0, -2.0, 1.0),
        )

        with self.assertRaisesRegex(ValueError, "half-mesh mirror plane"):
            build_runtime_scene_manifest(
                glb_name="house.glb",
                glb_bytes=b"glb",
                source_placements={"half-chair": source},
                instance_placements=(mismatched_instance,),
            )

    def test_manifest_orders_groups_and_instances_deterministically(self) -> None:
        source_model = _source_model()
        chair = PlacedGeneratedModel(
            object_id="chair",
            model=source_model,
            world_position=(0.0, 0.0, 0.0),
        )
        table = PlacedGeneratedModel(
            object_id="table",
            model=source_model,
            world_position=(0.0, 0.0, 0.0),
        )
        table_instance = PlacedGeneratedModel(
            object_id="table-copy-z",
            source_object_id="table",
            object_name="table",
            model=source_model,
            world_position=(0.0, 0.0, 0.0),
        )
        chair_instance_z = PlacedGeneratedModel(
            object_id="chair-copy-z",
            source_object_id="chair",
            object_name="chair",
            model=source_model,
            world_position=(0.0, 0.0, 0.0),
        )
        chair_instance_a = PlacedGeneratedModel(
            object_id="chair-copy-a",
            source_object_id="chair",
            object_name="chair",
            model=source_model,
            world_position=(0.0, 0.0, 0.0),
        )
        payload = build_runtime_scene_manifest(
            glb_name="house.glb",
            glb_bytes=b"glb",
            source_placements={"table": table, "chair": chair},
            instance_placements=(
                table_instance,
                chair_instance_z,
                chair_instance_a,
            ),
        )

        groups = payload["instanceGroups"]
        self.assertEqual(
            [group["sourceNodeName"] for group in groups],
            ["chair", "table"],
        )
        self.assertEqual(
            [entry["id"] for entry in groups[0]["instances"]],
            ["chair", "chair-copy-a", "chair-copy-z"],
        )
        self.assertEqual(
            [entry["id"] for entry in groups[1]["instances"]],
            ["table", "table-copy-z"],
        )

    def test_manifest_allows_a_source_model_without_geometry_nodes(self) -> None:
        instance = PlacedGeneratedModel(
            object_id="empty-copy",
            source_object_id="empty-source",
            object_name="empty-source",
            model=_model_without_scene_geometry(),
            world_position=(1.0, 2.0, 3.0),
        )

        payload = build_runtime_scene_manifest(
            glb_name="house.glb",
            glb_bytes=b"glb",
            source_placements={},
            instance_placements=(instance,),
        )

        self.assertEqual(
            payload["instanceGroups"][0]["sourceNodeName"],
            "empty-source",
        )

    def test_writer_moves_distinct_named_sources_outside_default_scene(
        self,
    ) -> None:
        source_model_a = _source_model_with_node_name("0")
        source_model_b = _source_model_with_node_name("0")
        chair_instance = PlacedGeneratedModel(
            object_id="chair-copy",
            source_object_id="chair-source",
            object_name="chair_prototype",
            model=source_model_a,
            world_position=(1.0, 0.0, 0.0),
        )
        table_instance = PlacedGeneratedModel(
            object_id="table-copy",
            source_object_id="table-source",
            object_name="table_prototype",
            model=source_model_b,
            world_position=(2.0, 0.0, 0.0),
        )
        integrated = compose_generated_model_instance_sources(
            _source_model(),
            (chair_instance, table_instance),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            glb_path = export_glb_file(
                integrated,
                Path(temporary_directory) / "scene.glb",
            )
            manifest_path = write_runtime_scene_manifest(
                glb_path,
                source_placements={},
                instance_placements=(table_instance, chair_instance),
            )
            final_glb = glb_path.read_bytes()
            document = _read_glb_document(final_glb)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        nodes = document["nodes"]
        named_node_indices = {
            node.get("name"): index
            for index, node in enumerate(nodes)
            if isinstance(node, dict)
        }
        default_scene = document["scenes"][document["scene"]]
        default_root_indices = set(default_scene.get("nodes", []))
        source_scene = next(
            scene
            for scene in document["scenes"]
            if scene.get("name") == INSTANCE_SOURCE_SCENE_NAME
        )
        source_root_indices = set(source_scene["nodes"])
        for root_name in ("chair_prototype", "table_prototype"):
            root_index = named_node_indices[root_name]
            self.assertNotIn(root_index, default_root_indices)
            self.assertIn(root_index, source_root_indices)
            self.assertTrue(nodes[root_index]["children"])
        self.assertEqual(
            [
                group["sourceNodeName"]
                for group in manifest["instanceGroups"]
            ],
            ["chair_prototype", "table_prototype"],
        )
        self.assertEqual(manifest["version"], 8)
        self.assertEqual(
            manifest["asset"]["sha256"],
            hashlib.sha256(final_glb).hexdigest(),
        )

    def test_writer_rejects_a_source_missing_from_the_prepared_glb(self) -> None:
        source_model = _source_model_with_node_name("0")
        instance = PlacedGeneratedModel(
            object_id="chair-copy",
            source_object_id="chair-source",
            object_name="chair_prototype",
            model=source_model,
            world_position=(1.0, 0.0, 0.0),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            glb_path = export_glb_file(
                _source_model(),
                Path(temporary_directory) / "scene.glb",
            )
            with self.assertRaisesRegex(ValueError, "missing the prepared"):
                write_runtime_scene_manifest(
                    glb_path,
                    source_placements={},
                    instance_placements=(instance,),
                )

    def test_writer_detaches_an_integrated_source_without_copying_its_mesh(
        self,
    ) -> None:
        source_model = _source_model_with_node_name("0")
        source = PlacedGeneratedModel(
            object_id="chair-source",
            source_object_id="chair-source",
            object_name="chair_prototype",
            model=source_model,
            world_position=(0.0, 0.0, 0.0),
        )
        instance = PlacedGeneratedModel(
            object_id="chair-copy",
            source_object_id="chair-source",
            object_name="chair_prototype",
            model=source_model,
            world_position=(2.0, 0.0, 0.0),
        )
        integrated = compose_generated_model_instance_sources(
            _source_model(),
            (source,),
        )
        before = _read_glb_document(integrated.glb_bytes)

        with tempfile.TemporaryDirectory() as temporary_directory:
            glb_path = export_glb_file(
                integrated,
                Path(temporary_directory) / "scene.glb",
            )
            manifest_path = write_runtime_scene_manifest(
                glb_path,
                source_placements={source.object_id: source},
                instance_placements=(instance,),
            )
            after = _read_glb_document(glb_path.read_bytes())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(len(after["meshes"]), len(before["meshes"]))
        named_node_indices = {
            node.get("name"): index
            for index, node in enumerate(after["nodes"])
            if isinstance(node, dict)
        }
        source_root_index = named_node_indices["chair_prototype"]
        default_scene = after["scenes"][after["scene"]]
        self.assertNotIn(source_root_index, default_scene.get("nodes", []))
        source_scene = next(
            scene
            for scene in after["scenes"]
            if scene.get("name") == INSTANCE_SOURCE_SCENE_NAME
        )
        self.assertIn(source_root_index, source_scene["nodes"])
        self.assertEqual(
            manifest["instanceGroups"][0]["sourceNodeName"],
            "chair_prototype",
        )

    def test_writer_marks_half_group_and_preserves_source_mirror_extras(
        self,
    ) -> None:
        source_model = _source_model_with_node_name("0")
        source = PlacedGeneratedModel(
            object_id="half-chair-source",
            object_name="half-chair",
            model=source_model,
            world_position=(0.0, 0.0, 0.0),
            symmetric_preview_orientation="vertical",
            symmetric_preview_plane_coordinate=0.25,
        )
        instance = PlacedGeneratedModel(
            object_id="half-chair-copy",
            source_object_id="half-chair-source",
            object_name="half-chair",
            model=source_model,
            world_position=(2.0, 0.0, 0.0),
            symmetric_preview_orientation="vertical",
            symmetric_preview_plane_coordinate=0.25,
        )
        integrated = compose_generated_model_instance_sources(
            _source_model(),
            (source,),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            glb_path = export_glb_file(
                integrated,
                Path(temporary_directory) / "scene.glb",
            )
            manifest_path = write_runtime_scene_manifest(
                glb_path,
                source_placements={source.object_id: source},
                instance_placements=(instance,),
            )
            document = _read_glb_document(glb_path.read_bytes())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        group = manifest["instanceGroups"][0]
        self.assertIs(group["halfMesh"], True)
        self.assertEqual(group["sourceNodeName"], "half-chair")

        nodes = document["nodes"]
        source_root_index = next(
            index
            for index, node in enumerate(nodes)
            if node.get("name") == "half-chair"
        )
        pending = list(nodes[source_root_index].get("children", []))
        descendant_half_meshes: list[object] = []
        while pending:
            node = nodes[pending.pop()]
            pending.extend(node.get("children", []))
            extras = node.get("extras", {})
            if "halfMesh" in extras:
                descendant_half_meshes.append(extras["halfMesh"])

        self.assertTrue(descendant_half_meshes)
        self.assertTrue(
            all(
                half_mesh
                == {
                    "mirrorPlane": {
                        "point": [0.25, 0.0, 0.0],
                        "normal": [1.0, 0.0, 0.0],
                    },
                    "uvMode": "reuse",
                }
                for half_mesh in descendant_half_meshes
            )
        )

    def test_manifest_rejects_an_instance_id_used_by_another_source(self) -> None:
        source_model = _source_model()
        chair = PlacedGeneratedModel(
            object_id="chair",
            model=source_model,
            world_position=(0.0, 0.0, 0.0),
        )
        table = PlacedGeneratedModel(
            object_id="table",
            model=source_model,
            world_position=(0.0, 0.0, 0.0),
        )
        conflicting_instance = PlacedGeneratedModel(
            object_id="table",
            source_object_id="chair",
            model=source_model,
            world_position=(0.0, 0.0, 0.0),
        )

        with self.assertRaisesRegex(ValueError, "globally unique"):
            build_runtime_scene_manifest(
                glb_name="house.glb",
                glb_bytes=b"glb",
                source_placements={"chair": chair, "table": table},
                instance_placements=(conflicting_instance,),
            )

    def test_writer_uses_the_exact_exported_glb_hash(self) -> None:
        source_model = _source_model()
        source = PlacedGeneratedModel(
            object_id="table",
            model=source_model,
            world_position=(0.0, 0.0, 0.0),
        )
        exported_model = compose_placed_generated_models(
            _empty_model(),
            (source,),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            glb_path = export_glb_file(
                exported_model,
                Path(temporary_directory) / "scene.glb",
            )
            manifest_path = write_runtime_scene_manifest(
                glb_path,
                source_placements={},
                instance_placements=(),
            )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(manifest_path, glb_path.with_suffix(".json"))
            self.assertEqual(payload["instanceGroups"], [])
            self.assertEqual(
                payload["asset"]["sha256"],
                hashlib.sha256(glb_path.read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
