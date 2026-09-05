# ### Imports ###
from __future__ import annotations

import json
import struct
import tempfile
import unittest
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.atlas_export import (
    MaterializedTextureAtlas,
    _map_uv_to_placement,
    apply_texture_atlases_to_export,
)
from housemaker.glass_material import (
    HOUSEMAKER_GLASS_MATERIAL_NAME,
    build_housemaker_glass_material,
)
from housemaker.glb import GeneratedModel, convert_to_glb
from housemaker.models import LevelData, RoomData, VertexData
from housemaker.pbr_maps import (
    ATLAS_MAP_BASE_COLOR,
    PBR_MAP_METALLIC,
    PBR_MAP_NORMAL,
    PBR_MAP_ROUGHNESS,
)
from housemaker.texture_atlas_state import (
    ATLAS_PACKING_MODE_FULL,
    ATLAS_PACKING_MODE_SYMMETRIC_QUARTER,
    ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR,
    ATLAS_SLOT_HALF_LEFT,
    ATLAS_SLOT_HALF_RIGHT,
    ATLAS_SLOT_QUADRANT_BOTTOM_RIGHT,
    TextureAtlasPlacement,
    TextureAtlasRecord,
)
from housemaker.surface_geometry import build_fixed_surfaces


# ### Fixture helpers ###
@dataclass(frozen=True)
class _HalfAtlasSpec:
    root_name: str
    source_id: str
    placement: TextureAtlasPlacement
    root_transform: np.ndarray
    child_transform: np.ndarray
    extras: dict[str, object]


def _textured_triangle(
    *,
    name: str,
    metadata: dict[str, str],
    uv: np.ndarray | None = None,
    glass: bool = False,
    factor_only: bool = False,
    x_offset: float = 0.0,
) -> trimesh.Trimesh:
    mesh = trimesh.Trimesh(
        vertices=np.asarray(
            (
                (x_offset, 0.0, 0.0),
                (x_offset + 1.0, 0.0, 0.0),
                (x_offset, 1.0, 0.0),
            ),
            dtype=float,
        ),
        faces=np.asarray(((0, 1, 2),), dtype=np.int64),
        process=False,
        metadata=metadata,
    )
    if glass:
        material = build_housemaker_glass_material(False)
    elif factor_only:
        material = PBRMaterial(
            name=name,
            baseColorFactor=[90, 100, 110, 255],
        )
    else:
        material = PBRMaterial(
            name=name,
            baseColorTexture=Image.new("RGBA", (2, 2), (80, 90, 100, 255)),
        )
    mesh.visual = TextureVisuals(
        uv=(
            np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
            if uv is None
            else np.asarray(uv, dtype=float)
        ),
        material=material,
    )
    return mesh


def _write_atlas_maps(directory: Path, resolution: int) -> dict[str, Path]:
    colors = {
        ATLAS_MAP_BASE_COLOR: (80, 120, 160, 255),
        PBR_MAP_NORMAL: (128, 128, 255, 255),
        PBR_MAP_ROUGHNESS: (170, 170, 170, 255),
        PBR_MAP_METALLIC: (20, 20, 20, 255),
    }
    paths: dict[str, Path] = {}
    for map_type, color in colors.items():
        path = directory / f"atlas.{map_type}.png"
        Image.new("RGBA", (resolution, resolution), color).save(path)
        paths[map_type] = path
    return paths


def _export_selective_atlas_document(
    directory: Path,
    active_map_types: frozenset[str],
) -> tuple[dict[str, object], dict[str, object]]:
    """Export one Atlas primitive and return its GLB document and material."""

    resolution = 2048
    source_id = "selective-object"
    placement = TextureAtlasPlacement(
        object_id=source_id,
        texture_path="selective.png",
        texture_resolution=512,
        x=0,
        y=0,
        size=512,
    )
    atlas = TextureAtlasRecord(
        atlas_id="selective",
        name="Selective",
        resolution=resolution,
        placements=[placement],
    )
    mesh = _textured_triangle(
        name="Source material",
        metadata={"housemaker_object_id": source_id},
    )
    scene = trimesh.Scene(mesh)
    model = GeneratedModel(mesh=mesh, scene=scene, glb_bytes=b"")
    result = apply_texture_atlases_to_export(
        model,
        (
            MaterializedTextureAtlas(
                atlas,
                _write_atlas_maps(directory, resolution),
                active_map_types=active_map_types,
            ),
        ),
    )
    document = _read_glb_json(result.glb_bytes)
    material = next(
        item
        for item in document["materials"]
        if item["name"] == atlas.name
    )
    return document, material


def _read_glb_json(payload: bytes) -> dict[str, object]:
    json_length, json_type = struct.unpack_from("<II", payload, 12)
    if json_type != 0x4E4F534A:
        raise AssertionError("The first GLB chunk is not JSON.")
    return json.loads(payload[20 : 20 + json_length].decode("utf-8"))


def _assert_direct_half_mesh_nodes(
    document: dict[str, object],
    expected_half_mesh_by_name: dict[str, dict[str, object]],
) -> None:
    """Require every half marker to live directly on one root mesh node."""

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


def _architectural_surface_model() -> tuple[GeneratedModel, tuple[str, ...]]:
    vertex_data = VertexData()
    boundary_ids = tuple(
        vertex_data.add_vertex(*point).id
        for point in (
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (0.0, 100.0),
        )
    )
    for start_id, end_id in zip(
        boundary_ids,
        (*boundary_ids[1:], boundary_ids[0]),
        strict=True,
    ):
        vertex_data.add_edge(start_id, end_id)
    center = vertex_data.add_vertex(50.0, 50.0)
    room = RoomData(
        name="Atlas room",
        vertex_ids=boundary_ids,
        center_vertex_id=center.id,
        color_rgb=(120, 140, 160),
    )
    level = LevelData(
        index=2,
        name="Ground",
        vertex_data=vertex_data,
        rooms=[room],
        floor_contour_vertex_ids=boundary_ids,
    )
    selected_ids = tuple(
        next(
            surface.surface_id
            for surface in build_fixed_surfaces([level])
            if surface.surface_type == surface_type
        )
        for surface_type in ("wall", "floor", "ceiling")
    )
    texture_buffer = BytesIO()
    Image.new("RGBA", (8, 8), (70, 100, 130, 255)).save(
        texture_buffer,
        format="PNG",
    )
    texture_buffer.seek(0)
    texture_png = texture_buffer.read()
    return (
        convert_to_glb(
            [level],
            surface_materials={
                surface_id: texture_png for surface_id in selected_ids
            },
        ),
        selected_ids,
    )


# ### Export tests ###
class TextureAtlasExportTests(unittest.TestCase):
    def test_base_only_atlas_omits_all_pbr_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            document, material = _export_selective_atlas_document(
                Path(temporary_directory),
                frozenset({ATLAS_MAP_BASE_COLOR}),
            )

        pbr = material["pbrMetallicRoughness"]
        self.assertIn("baseColorTexture", pbr)
        self.assertNotIn("normalTexture", material)
        self.assertNotIn("metallicRoughnessTexture", pbr)
        self.assertEqual(pbr.get("metallicFactor", 1.0), 0.0)
        self.assertEqual(pbr.get("roughnessFactor", 1.0), 1.0)
        self.assertEqual(len(document["images"]), 1)

    def test_normal_only_atlas_exports_no_metallic_roughness_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            document, material = _export_selective_atlas_document(
                Path(temporary_directory),
                frozenset({ATLAS_MAP_BASE_COLOR, PBR_MAP_NORMAL}),
            )

        pbr = material["pbrMetallicRoughness"]
        self.assertIn("normalTexture", material)
        self.assertNotIn("metallicRoughnessTexture", pbr)
        self.assertEqual(pbr.get("metallicFactor", 1.0), 0.0)
        self.assertEqual(pbr.get("roughnessFactor", 1.0), 1.0)
        self.assertEqual(len(document["images"]), 2)

    def test_partial_metallic_roughness_atlases_export_one_packed_image(
        self,
    ) -> None:
        cases = (
            (frozenset({PBR_MAP_ROUGHNESS}), 0.0),
            (frozenset({PBR_MAP_METALLIC}), 1.0),
            (frozenset({PBR_MAP_ROUGHNESS, PBR_MAP_METALLIC}), 1.0),
        )
        for active_pbr_maps, expected_metallic_factor in cases:
            with self.subTest(active_pbr_maps=active_pbr_maps):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    document, material = _export_selective_atlas_document(
                        Path(temporary_directory),
                        frozenset(
                            {ATLAS_MAP_BASE_COLOR, *active_pbr_maps}
                        ),
                    )

                pbr = material["pbrMetallicRoughness"]
                self.assertNotIn("normalTexture", material)
                self.assertIn("metallicRoughnessTexture", pbr)
                self.assertEqual(
                    pbr.get("metallicFactor", 1.0),
                    expected_metallic_factor,
                )
                self.assertEqual(pbr.get("roughnessFactor", 1.0), 1.0)
                self.assertEqual(len(document["images"]), 2)

    def test_materialized_atlas_requires_active_base_color(self) -> None:
        atlas = TextureAtlasRecord(
            atlas_id="validation",
            name="Validation",
            resolution=2048,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            map_paths = _write_atlas_maps(
                Path(temporary_directory),
                atlas.resolution,
            )
            with self.assertRaisesRegex(ValueError, "active base color"):
                MaterializedTextureAtlas(
                    atlas,
                    map_paths,
                    active_map_types=frozenset({PBR_MAP_NORMAL}),
                )

    def test_generated_objects_use_atlas_record_name_and_one_material(
        self,
    ) -> None:
        resolution = 2048
        placements = [
            TextureAtlasPlacement(
                object_id=object_id,
                texture_path=f"{object_id}.png",
                texture_resolution=512,
                x=index * 512,
                y=0,
                size=512,
            )
            for index, object_id in enumerate(("object-a", "object-b"))
        ]
        atlas = TextureAtlasRecord(
            atlas_id="generated-assets-id",
            name="[HALF] Generated assets",
            resolution=resolution,
            placements=placements,
        )
        meshes = tuple(
            _textured_triangle(
                name=f"Source {index}",
                metadata={"housemaker_object_id": placement.object_id},
                x_offset=float(index * 2),
            )
            for index, placement in enumerate(placements)
        )
        scene = trimesh.Scene()
        for index, mesh in enumerate(meshes):
            scene.add_geometry(mesh, node_name=f"object-{index}")
        model = GeneratedModel(
            mesh=trimesh.util.concatenate(meshes),
            scene=scene,
            glb_bytes=b"",
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            result = apply_texture_atlases_to_export(
                model,
                (
                    MaterializedTextureAtlas(
                        atlas,
                        _write_atlas_maps(
                            Path(temporary_directory),
                            resolution,
                        ),
                        active_map_types=frozenset(
                            {ATLAS_MAP_BASE_COLOR}
                        ),
                    ),
                ),
            )

        self.assertEqual(len(result.scene.geometry), 1)
        exported_geometry = next(iter(result.scene.geometry.values()))
        self.assertEqual(exported_geometry.visual.material.name, atlas.name)
        document = _read_glb_json(result.glb_bytes)
        self.assertEqual(
            [material["name"] for material in document["materials"]],
            [atlas.name],
        )
        self.assertEqual(
            sum(
                "material" in primitive
                for mesh in document["meshes"]
                for primitive in mesh["primitives"]
            ),
            1,
        )

    def test_wall_floor_and_ceiling_share_one_atlas_primitive(self) -> None:
        model, surface_ids = _architectural_surface_model()
        source_id = "surface-texture:room"
        placement = TextureAtlasPlacement(
            object_id=source_id,
            texture_path="room.png",
            texture_resolution=512,
            x=0,
            y=0,
            size=512,
        )
        atlas = TextureAtlasRecord(
            atlas_id="architecture",
            name="Architecture",
            resolution=2048,
            placements=[placement],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            map_paths = _write_atlas_maps(
                Path(temporary_directory),
                2048,
            )
            result = apply_texture_atlases_to_export(
                model,
                (MaterializedTextureAtlas(atlas, map_paths),),
                surface_source_ids={
                    surface_id: source_id for surface_id in surface_ids
                },
            )

        atlas_meshes = [
            geometry
            for geometry in result.scene.geometry.values()
            if getattr(
                getattr(geometry.visual, "material", None),
                "name",
                None,
            )
            == atlas.name
        ]
        self.assertEqual(len(atlas_meshes), 1)
        self.assertEqual(
            atlas_meshes[0].metadata["housemaker_atlas_source_ids"],
            [source_id],
        )
        document = _read_glb_json(result.glb_bytes)
        atlas_material_index = next(
            index
            for index, material in enumerate(document["materials"])
            if material["name"] == atlas.name
        )
        self.assertEqual(
            sum(
                primitive.get("material") == atlas_material_index
                for mesh in document["meshes"]
                for primitive in mesh["primitives"]
            ),
            1,
        )

    def test_opaque_objects_and_surfaces_share_one_batched_atlas_material(
        self,
    ) -> None:
        resolution = 2048
        placements = [
            TextureAtlasPlacement(
                object_id="object-a",
                texture_path="a.png",
                texture_resolution=512,
                x=0,
                y=0,
                size=512,
            ),
            TextureAtlasPlacement(
                object_id="object-b",
                texture_path="b.png",
                texture_resolution=512,
                x=512,
                y=0,
                size=512,
            ),
            TextureAtlasPlacement(
                object_id="surface-texture:stone",
                texture_path="surface.png",
                texture_resolution=512,
                x=0,
                y=512,
                size=512,
            ),
        ]
        atlas = TextureAtlasRecord(
            atlas_id="shared",
            name="Shared",
            resolution=resolution,
            placements=placements,
        )
        scene = trimesh.Scene()
        meshes = (
            _textured_triangle(
                name="Object A",
                metadata={"housemaker_object_id": "object-a"},
                factor_only=True,
            ),
            _textured_triangle(
                name="Object B",
                metadata={"housemaker_object_id": "object-b"},
                x_offset=2.0,
            ),
            _textured_triangle(
                name="Surface",
                metadata={"housemaker_surface_id": "surface-1"},
                uv=np.asarray(((0.0, 0.0), (2.0, 0.0), (0.0, 1.0))),
                x_offset=4.0,
            ),
            _textured_triangle(
                name="Second surface",
                metadata={"housemaker_surface_id": "surface-2"},
                x_offset=5.0,
            ),
            _textured_triangle(
                name="Glass A",
                metadata={"housemaker_object_id": "object-a"},
                glass=True,
                x_offset=6.0,
            ),
            _textured_triangle(
                name="Glass B",
                metadata={"housemaker_object_id": "object-b"},
                glass=True,
                x_offset=8.0,
            ),
        )
        for index, mesh in enumerate(meshes):
            scene.add_geometry(mesh, node_name=f"node-{index}")
        model = GeneratedModel(
            mesh=trimesh.util.concatenate(meshes),
            scene=scene,
            glb_bytes=b"",
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            map_paths = _write_atlas_maps(
                Path(temporary_directory),
                resolution,
            )
            result = apply_texture_atlases_to_export(
                model,
                (MaterializedTextureAtlas(atlas, map_paths),),
                surface_source_ids={
                    "surface-1": "surface-texture:stone",
                    "surface-2": "surface-texture:stone",
                },
            )

        materials = [
            geometry.visual.material
            for geometry in result.scene.geometry.values()
            if isinstance(geometry, trimesh.Trimesh)
        ]
        atlas_materials = [
            material
            for material in materials
            if getattr(material, "name", None) == atlas.name
        ]
        glass_materials = [
            material
            for material in materials
            if getattr(material, "name", None) == HOUSEMAKER_GLASS_MATERIAL_NAME
        ]
        self.assertEqual(len(atlas_materials), 1)
        self.assertEqual(len(glass_materials), 2)
        self.assertIs(glass_materials[0], glass_materials[1])
        atlas_geometry = next(
            geometry
            for geometry in result.scene.geometry.values()
            if geometry.visual.material in atlas_materials
        )
        self.assertGreater(len(atlas_geometry.faces), 3)
        atlas_uv = np.asarray(atlas_geometry.visual.uv, dtype=float)
        self.assertTrue(np.all(atlas_uv >= 0.0))
        self.assertTrue(np.all(atlas_uv <= 1.0))

        document = _read_glb_json(result.glb_bytes)
        material_names = {
            material["name"] for material in document.get("materials", [])
        }
        self.assertEqual(len(document["materials"]), 2)
        self.assertEqual(
            material_names,
            {
                atlas.name,
                HOUSEMAKER_GLASS_MATERIAL_NAME,
            },
        )
        atlas_material_index = next(
            index
            for index, material in enumerate(document["materials"])
            if material["name"] == atlas.name
        )
        atlas_material = document["materials"][atlas_material_index]
        self.assertIn("normalTexture", atlas_material)
        self.assertIn(
            "baseColorTexture",
            atlas_material["pbrMetallicRoughness"],
        )
        self.assertIn(
            "metallicRoughnessTexture",
            atlas_material["pbrMetallicRoughness"],
        )
        self.assertEqual(len(document["images"]), 3)
        atlas_primitive_count = sum(
            primitive.get("material") == atlas_material_index
            for mesh in document["meshes"]
            for primitive in mesh["primitives"]
        )
        self.assertEqual(atlas_primitive_count, 1)

    def test_half_model_mesh_nodes_remain_separate_through_atlas_export(
        self,
    ) -> None:
        resolution = 2048
        half_uv = np.asarray(
            ((0.0, 1.0), (0.5, 0.0), (0.0, 0.0)),
            dtype=float,
        )
        half_specs = (
            _HalfAtlasSpec(
                root_name="[HALF] window_frame_01",
                source_id="half-window",
                placement=TextureAtlasPlacement(
                    object_id="half-window",
                    texture_path="half-window.png",
                    texture_resolution=512,
                    x=0,
                    y=0,
                    size=512,
                    packing_mode=ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR,
                    slot_half=ATLAS_SLOT_HALF_LEFT,
                ),
                root_transform=trimesh.transformations.translation_matrix(
                    (3.0, 4.0, 5.0)
                ),
                child_transform=trimesh.transformations.rotation_matrix(
                    np.pi / 2.0,
                    (0.0, 0.0, 1.0),
                ),
                extras={
                    "halfMesh": {
                        "mirrorPlane": {
                            "point": [3.5, 4.0, 5.0],
                            "normal": [0.0, 1.0, 0.0],
                        },
                        "uvMode": "reuse",
                    }
                },
            ),
            _HalfAtlasSpec(
                root_name="[HALF] cabinet_01",
                source_id="half-cabinet",
                placement=TextureAtlasPlacement(
                    object_id="half-cabinet",
                    texture_path="half-cabinet.png",
                    texture_resolution=512,
                    x=0,
                    y=0,
                    size=512,
                    packing_mode=ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR,
                    slot_half=ATLAS_SLOT_HALF_RIGHT,
                ),
                root_transform=trimesh.transformations.translation_matrix(
                    (-2.0, 1.0, 7.0)
                ),
                child_transform=trimesh.transformations.translation_matrix(
                    (0.0, 2.0, 0.0)
                ),
                extras={
                    "halfMesh": {
                        "mirrorPlane": {
                            "point": [-2.0, 1.0, 7.0],
                            "normal": [1.0, 0.0, 0.0],
                        },
                        "uvMode": "reuse",
                    }
                },
            ),
        )
        ordinary_placements = (
            TextureAtlasPlacement(
                object_id="ordinary-a",
                texture_path="ordinary-a.png",
                texture_resolution=512,
                x=512,
                y=0,
                size=512,
            ),
            TextureAtlasPlacement(
                object_id="ordinary-b",
                texture_path="ordinary-b.png",
                texture_resolution=512,
                x=1024,
                y=0,
                size=512,
            ),
        )
        atlas = TextureAtlasRecord(
            atlas_id="half-models",
            name="Half models",
            resolution=resolution,
            placements=[
                *(spec.placement for spec in half_specs),
                *ordinary_placements,
            ],
        )

        scene = trimesh.Scene()
        expected_world_vertices: dict[str, np.ndarray] = {}
        for index, spec in enumerate(half_specs, start=1):
            mesh = _textured_triangle(
                name=spec.root_name,
                metadata={"housemaker_object_id": spec.source_id},
                uv=half_uv,
            )
            scene.graph.update(
                frame_from=scene.graph.base_frame,
                frame_to=spec.root_name,
                matrix=spec.root_transform,
                metadata=spec.extras,
            )
            scene.add_geometry(
                mesh,
                geom_name=f"half-geometry-{index}",
                node_name=f"half-mesh-{index}",
                parent_node_name=spec.root_name,
                transform=spec.child_transform,
            )
            expected_world_vertices[spec.root_name] = trimesh.transform_points(
                np.asarray(mesh.vertices, dtype=float),
                spec.root_transform @ spec.child_transform,
            )

        ordinary_meshes = (
            _textured_triangle(
                name="Ordinary A",
                metadata={"housemaker_object_id": "ordinary-a"},
                x_offset=2.0,
            ),
            _textured_triangle(
                name="Ordinary B",
                metadata={"housemaker_object_id": "ordinary-b"},
                x_offset=4.0,
            ),
        )
        for index, mesh in enumerate(ordinary_meshes, start=1):
            scene.add_geometry(
                mesh,
                geom_name=f"ordinary-geometry-{index}",
                node_name=f"ordinary-node-{index}",
            )
        model = GeneratedModel(
            mesh=trimesh.util.concatenate(
                [
                    *(
                        scene.geometry[f"half-geometry-{index}"]
                        for index in range(1, 3)
                    ),
                    *ordinary_meshes,
                ]
            ),
            scene=scene,
            glb_bytes=b"",
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            result = apply_texture_atlases_to_export(
                model,
                (
                    MaterializedTextureAtlas(
                        atlas,
                        _write_atlas_maps(
                            Path(temporary_directory),
                            resolution,
                        ),
                    ),
                ),
            )

        geometry_nodes = {str(node) for node in result.scene.graph.nodes_geometry}
        half_geometry_nodes = {spec.root_name for spec in half_specs}
        self.assertTrue(half_geometry_nodes.issubset(geometry_nodes))
        self.assertEqual(
            {
                str(node)
                for node in result.scene.graph.nodes
                if str(node).startswith("[HALF] ")
            },
            half_geometry_nodes,
        )
        for spec in half_specs:
            parent_name = result.scene.graph.transforms.parents.get(
                spec.root_name
            )
            self.assertEqual(parent_name, result.scene.graph.base_frame)
            edge_data = result.scene.graph.transforms.edge_data[
                (parent_name, spec.root_name)
            ]
            self.assertEqual(
                edge_data.get("metadata"),
                spec.extras,
            )
            expected_transform = spec.root_transform @ spec.child_transform
            np.testing.assert_allclose(
                result.scene.graph.get(spec.root_name)[0],
                expected_transform,
                atol=1e-8,
            )
            node_transform, geometry_name = result.scene.graph.get(
                spec.root_name
            )
            geometry = result.scene.geometry[geometry_name]
            self.assertEqual(len(geometry.faces), 1)
            self.assertEqual(
                geometry.visual.material.name,
                atlas.name,
            )
            actual_world_vertices = trimesh.transform_points(
                np.asarray(geometry.vertices, dtype=float),
                node_transform,
            )
            np.testing.assert_allclose(
                actual_world_vertices,
                expected_world_vertices[spec.root_name],
                atol=1e-8,
            )
            np.testing.assert_allclose(
                np.asarray(geometry.visual.uv, dtype=float),
                _map_uv_to_placement(
                    half_uv,
                    spec.placement,
                    resolution,
                ),
                atol=1e-8,
            )

        ordinary_nodes = geometry_nodes - half_geometry_nodes
        self.assertEqual(len(ordinary_nodes), 1)
        ordinary_transform, ordinary_geometry_name = result.scene.graph.get(
            next(iter(ordinary_nodes))
        )
        np.testing.assert_allclose(ordinary_transform, np.eye(4), atol=1e-8)
        self.assertEqual(
            len(result.scene.geometry[ordinary_geometry_name].faces),
            sum(len(mesh.faces) for mesh in ordinary_meshes),
        )
        self.assertEqual(
            sum(
                len(result.scene.geometry[result.scene.graph.get(node)[1]].faces)
                for node in half_geometry_nodes
            ),
            2,
        )

        document = _read_glb_json(result.glb_bytes)
        _assert_direct_half_mesh_nodes(
            document,
            {
                spec.root_name: spec.extras["halfMesh"]
                for spec in half_specs
            },
        )
        nodes_by_name = {node["name"]: node for node in document["nodes"]}
        for spec in half_specs:
            self.assertEqual(
                nodes_by_name[spec.root_name].get("extras"),
                spec.extras,
            )
            self.assertIn("mesh", nodes_by_name[spec.root_name])
        node_extras = {
            node["name"]: node["extras"]
            for node in document["nodes"]
            if "extras" in node
        }
        self.assertEqual(
            node_extras,
            {spec.root_name: spec.extras for spec in half_specs},
        )
        atlas_material_index = next(
            index
            for index, material in enumerate(document["materials"])
            if material["name"] == atlas.name
        )
        self.assertEqual(
            sum(
                primitive.get("material") == atlas_material_index
                for mesh in document["meshes"]
                for primitive in mesh["primitives"]
            ),
            3,
        )

        loaded_scene = trimesh.load(
            BytesIO(result.glb_bytes),
            file_type="glb",
            force="scene",
            process=False,
        )
        self.assertIsInstance(loaded_scene, trimesh.Scene)
        assert isinstance(loaded_scene, trimesh.Scene)
        loaded_geometry_nodes = {
            str(node) for node in loaded_scene.graph.nodes_geometry
        }
        self.assertTrue(half_geometry_nodes.issubset(loaded_geometry_nodes))
        loaded_metadata = {
            str(child): attributes.get("metadata")
            for _parent, child, attributes in loaded_scene.graph.to_edgelist()
            if str(child).startswith("[HALF] ")
        }
        self.assertEqual(
            loaded_metadata,
            {spec.root_name: spec.extras for spec in half_specs},
        )
        round_trip_payload = bytes(loaded_scene.export(file_type="glb"))
        _assert_direct_half_mesh_nodes(
            _read_glb_json(round_trip_payload),
            {
                spec.root_name: spec.extras["halfMesh"]
                for spec in half_specs
            },
        )

    def test_packed_uv_transforms_match_atlas_pixel_regions(self) -> None:
        full = TextureAtlasPlacement(
            object_id="full",
            texture_path="full.png",
            texture_resolution=512,
            x=512,
            y=1024,
            size=512,
            packing_mode=ATLAS_PACKING_MODE_FULL,
        )
        half = TextureAtlasPlacement(
            object_id="half",
            texture_path="half.png",
            texture_resolution=512,
            x=0,
            y=0,
            size=512,
            packing_mode=ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR,
            slot_half=ATLAS_SLOT_HALF_RIGHT,
        )
        quarter = TextureAtlasPlacement(
            object_id="quarter",
            texture_path="quarter.png",
            texture_resolution=512,
            x=1024,
            y=1024,
            size=1024,
            packing_mode=ATLAS_PACKING_MODE_SYMMETRIC_QUARTER,
            slot_quadrant=ATLAS_SLOT_QUADRANT_BOTTOM_RIGHT,
        )

        np.testing.assert_allclose(
            _map_uv_to_placement(
                np.asarray(((0.0, 1.0), (1.0, 0.0))),
                full,
                2048,
            ),
            np.asarray(
                (
                    (512.5 / 2048.0, 1.0 - 1024.5 / 2048.0),
                    (1023.5 / 2048.0, 1.0 - 1535.5 / 2048.0),
                )
            ),
        )
        np.testing.assert_allclose(
            _map_uv_to_placement(
                np.asarray(((0.0, 1.0), (0.5, 0.0))),
                half,
                2048,
            ),
            np.asarray(
                (
                    (256.5 / 2048.0, 1.0 - 0.5 / 2048.0),
                    (511.5 / 2048.0, 1.0 - 511.5 / 2048.0),
                )
            ),
        )
        np.testing.assert_allclose(
            _map_uv_to_placement(
                np.asarray(((0.0, 1.0), (0.5, 0.5))),
                quarter,
                2048,
            ),
            np.asarray(
                (
                    (1536.5 / 2048.0, 1.0 - 1536.5 / 2048.0),
                    (2047.5 / 2048.0, 1.0 - 2047.5 / 2048.0),
                )
            ),
        )

    def test_nested_object_transform_is_baked_into_shared_atlas_batch(
        self,
    ) -> None:
        resolution = 512
        placement = TextureAtlasPlacement(
            object_id="transformed-object",
            texture_path="object.png",
            texture_resolution=resolution,
            x=0,
            y=0,
            size=resolution,
        )
        atlas = TextureAtlasRecord(
            atlas_id="transformed",
            name="Transformed",
            resolution=2048,
            placements=[placement],
        )
        mesh = _textured_triangle(
            name="Transformed object",
            metadata={
                "housemaker_surface_id": "unrelated-surface",
                "housemaker_object_id": "transformed-object",
            },
        )
        parent_transform = trimesh.transformations.translation_matrix(
            (4.0, 5.0, 6.0)
        )
        local_transform = trimesh.transformations.rotation_matrix(
            np.pi / 2.0,
            (0.0, 0.0, 1.0),
        )
        scene = trimesh.Scene()
        scene.graph.update(
            frame_from=scene.graph.base_frame,
            frame_to="object-parent",
            matrix=parent_transform,
        )
        scene.add_geometry(
            mesh,
            node_name="object-leaf",
            parent_node_name="object-parent",
            transform=local_transform,
        )
        model = GeneratedModel(mesh=mesh, scene=scene, glb_bytes=b"")

        with tempfile.TemporaryDirectory() as temporary_directory:
            map_paths = _write_atlas_maps(
                Path(temporary_directory),
                2048,
            )
            result = apply_texture_atlases_to_export(
                model,
                (MaterializedTextureAtlas(atlas, map_paths),),
            )

        atlas_mesh = next(iter(result.scene.geometry.values()))
        expected_vertices = trimesh.transform_points(
            np.asarray(mesh.vertices, dtype=float),
            parent_transform @ local_transform,
        )
        np.testing.assert_allclose(
            np.asarray(atlas_mesh.vertices, dtype=float),
            expected_vertices,
        )


if __name__ == "__main__":
    unittest.main()
