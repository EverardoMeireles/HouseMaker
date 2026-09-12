# ### Imports ###
from __future__ import annotations

import json
import struct
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.atlas_export import (
    MaterializedTextureAtlas,
    SurfaceAmbientOcclusionAtlasContext,
    _split_repeating_uv_triangles,
    apply_texture_atlases_to_export,
    bake_surface_ambient_occlusion_for_atlas,
    prepare_surface_ambient_occlusion_preview_for_atlas,
)
from housemaker.glb import PACKED_ORM_AO_UV_ATTRIBUTE, GeneratedModel
from housemaker.pbr_maps import (
    ATLAS_MAP_BASE_COLOR,
    ATLAS_MAP_TYPES,
    PBR_MAP_METALLIC,
    PBR_MAP_NORMAL,
    PBR_MAP_ROUGHNESS,
)
from housemaker.surface_ao_baking import (
    SurfaceAmbientOcclusionBakeResult as CoreSurfaceAoBakeResult,
)
from housemaker.surface_ao_baking import (
    build_surface_ao_geometry_signature,
)
from housemaker.texture_atlas_state import (
    ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR,
    ATLAS_SLOT_HALF_LEFT,
    TextureAtlasPlacement,
    TextureAtlasRecord,
)


# ### Fixture helpers ###
def _textured_triangle(metadata: dict[str, str], x_offset: float) -> trimesh.Trimesh:
    vertices = np.asarray(
        (
            (x_offset, 0.0, 0.0),
            (x_offset + 1.0, 0.0, 0.0),
            (x_offset, 1.0, 0.0),
        ),
        dtype=float,
    )
    return trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(((0, 1, 2),), dtype=np.int64),
        vertex_normals=np.asarray(((0.0, 0.0, 1.0),) * 3),
        visual=TextureVisuals(
            uv=np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
            material=PBRMaterial(
                baseColorTexture=Image.new("RGBA", (2, 2), "white")
            ),
        ),
        metadata=metadata,
        process=False,
    )


def _scene_model() -> GeneratedModel:
    surface = _textured_triangle(
        {"housemaker_surface_id": "wall-1"},
        0.0,
    )
    placed_object = _textured_triangle(
        {"housemaker_object_id": "object-1"},
        2.0,
    )
    scene = trimesh.Scene()
    scene.add_geometry(surface, node_name="surface")
    scene.add_geometry(placed_object, node_name="object")
    return GeneratedModel(
        mesh=trimesh.util.concatenate((surface, placed_object)),
        scene=scene,
        glb_bytes=b"",
    )


def _tiled_surface_and_half_model(
) -> tuple[GeneratedModel, str, dict[str, object]]:
    surface = trimesh.Trimesh(
        vertices=np.asarray(
            ((-2.0, 0.0, 0.0), (0.5, 0.0, 0.0), (-2.0, 2.5, 0.0)),
            dtype=float,
        ),
        faces=np.asarray(((0, 1, 2),), dtype=np.int64),
        vertex_normals=np.asarray(((0.0, 0.0, 1.0),) * 3),
        visual=TextureVisuals(
            uv=np.asarray(
                ((-0.25, -0.25), (2.25, -0.25), (-0.25, 2.25)),
                dtype=float,
            ),
            material=PBRMaterial(
                baseColorTexture=Image.new("RGBA", (2, 2), "white")
            ),
        ),
        metadata={"housemaker_surface_id": "tiled-wall"},
        process=False,
    )
    half = _textured_triangle(
        {"housemaker_object_id": "authored-half"},
        1.0,
    )
    half.visual = TextureVisuals(
        uv=np.asarray(((0.0, 1.0), (0.5, 0.0), (0.0, 0.0))),
        material=half.visual.material,
    )
    half_node_name = "[HALF] authored_panel"
    half_mesh = {
        "mirrorPlane": {
            "point": [0.0, 0.0, 0.0],
            "normal": [1.0, 0.0, 0.0],
        },
        "uvMode": "reuse",
    }
    scene = trimesh.Scene()
    scene.add_geometry(
        surface,
        geom_name="tiled-surface-geometry",
        node_name="tiled-surface-node",
    )
    scene.graph.update(
        frame_from=scene.graph.base_frame,
        frame_to=half_node_name,
        matrix=np.eye(4),
        metadata={"halfMesh": half_mesh},
    )
    scene.add_geometry(
        half,
        geom_name="authored-half-geometry",
        node_name="authored-half-child",
        parent_node_name=half_node_name,
    )
    return (
        GeneratedModel(
            mesh=trimesh.util.concatenate((surface, half)),
            scene=scene,
            glb_bytes=b"",
        ),
        half_node_name,
        half_mesh,
    )


def _atlas() -> TextureAtlasRecord:
    return TextureAtlasRecord(
        atlas_id="surface-ao",
        name="Surface AO",
        resolution=2048,
        placements=[
            TextureAtlasPlacement(
                object_id="surface-texture",
                texture_path="surface.png",
                texture_resolution=512,
                x=0,
                y=0,
                size=512,
            ),
            TextureAtlasPlacement(
                object_id="object-1",
                texture_path="object.png",
                texture_resolution=512,
                x=512,
                y=0,
                size=512,
            ),
        ],
    )


def _write_maps(directory: Path) -> dict[str, Path]:
    colors = {
        ATLAS_MAP_BASE_COLOR: (80, 100, 120, 255),
        PBR_MAP_NORMAL: (128, 128, 255, 255),
        PBR_MAP_ROUGHNESS: (170, 170, 170, 255),
        PBR_MAP_METALLIC: (20, 20, 20, 255),
    }
    paths: dict[str, Path] = {}
    for map_type in ATLAS_MAP_TYPES:
        path = directory / f"{map_type}.png"
        Image.new("RGBA", (2048, 2048), colors[map_type]).save(path)
        paths[map_type] = path
    return paths


def _read_glb_json(payload: bytes) -> dict[str, object]:
    json_length, json_type = struct.unpack_from("<II", payload, 12)
    if json_type != 0x4E4F534A:
        raise AssertionError("The first GLB chunk is not JSON.")
    return json.loads(payload[20 : 20 + json_length].decode("utf-8"))


# ### Export integration tests ###
class SurfaceAmbientOcclusionAtlasExportTests(unittest.TestCase):
    def test_object_only_atlas_can_bake_and_reuse_cached_uv1_ao(self) -> None:
        placed_object = _textured_triangle(
            {"housemaker_object_id": "object-only"},
            0.0,
        )
        model = GeneratedModel(
            mesh=placed_object,
            scene=trimesh.Scene(placed_object),
            glb_bytes=b"",
        )
        atlas = TextureAtlasRecord(
            atlas_id="object-only-atlas",
            name="Object-only Atlas",
            resolution=2048,
            placements=[
                TextureAtlasPlacement(
                    object_id="object-only",
                    texture_path="object.png",
                    texture_resolution=512,
                    x=0,
                    y=0,
                    size=512,
                )
            ],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            maps = _write_maps(directory)
            source = MaterializedTextureAtlas(
                atlas,
                maps,
                active_map_types=frozenset(
                    (ATLAS_MAP_BASE_COLOR, PBR_MAP_ROUGHNESS)
                ),
            )
            receiver_keys: list[str] = []

            def fake_bake(layout, occluders, **_kwargs):
                receiver_keys.extend(receiver.key for receiver in layout.receivers)
                return CoreSurfaceAoBakeResult(
                    layout=layout,
                    ambient_occlusion=np.full(
                        (layout.resolution, layout.resolution),
                        181,
                        dtype=np.uint8,
                    ),
                    geometry_signature=build_surface_ao_geometry_signature(
                        layout,
                        occluders,
                    ),
                )

            with patch(
                "housemaker.atlas_export.bake_surface_ambient_occlusion",
                side_effect=fake_bake,
            ):
                bake = bake_surface_ambient_occlusion_for_atlas(model, source)

            self.assertEqual(len(receiver_keys), 1)
            self.assertIn("object-only", receiver_keys[0])
            ao_path = directory / "object-only-ao.png"
            Image.fromarray(bake.ambient_occlusion, mode="L").save(ao_path)
            cached_source = MaterializedTextureAtlas(
                atlas,
                maps,
                active_map_types=source.active_map_types,
                surface_ao_image_path=ao_path,
                surface_ao_geometry_signature=bake.geometry_signature,
            )
            preview = prepare_surface_ambient_occlusion_preview_for_atlas(
                model,
                cached_source,
            )
            with patch(
                "housemaker.atlas_export.bake_placed_object_ambient_occlusion",
                side_effect=AssertionError("cached AO used the fallback object baker"),
            ):
                result = apply_texture_atlases_to_export(
                    model,
                    (cached_source,),
                )

        atlas_mesh = next(iter(result.scene.geometry.values()))
        self.assertGreater(len(preview.scene.geometry), 0)
        self.assertIn(PACKED_ORM_AO_UV_ATTRIBUTE, atlas_mesh.vertex_attributes)
        self.assertIsNotNone(atlas_mesh.visual.material.occlusionTexture)

    def test_bake_uses_export_order_for_duplicate_source_ownership(self) -> None:
        duplicated_surface = _textured_triangle(
            {"housemaker_surface_id": "wall-shared"},
            0.0,
        )
        target_surface = _textured_triangle(
            {"housemaker_surface_id": "wall-target"},
            2.0,
        )
        scene = trimesh.Scene()
        scene.add_geometry(duplicated_surface, node_name="shared-surface")
        scene.add_geometry(target_surface, node_name="target-surface")
        model = GeneratedModel(
            mesh=trimesh.util.concatenate(
                (duplicated_surface, target_surface)
            ),
            scene=scene,
            glb_bytes=b"",
        )
        first_atlas = TextureAtlasRecord(
            atlas_id="first-atlas",
            name="First atlas",
            resolution=2048,
            placements=[
                TextureAtlasPlacement(
                    object_id="shared-texture",
                    texture_path="shared.png",
                    texture_resolution=512,
                    x=0,
                    y=0,
                    size=512,
                )
            ],
        )
        target_atlas = TextureAtlasRecord(
            atlas_id="target-atlas",
            name="Target atlas",
            resolution=2048,
            placements=[
                TextureAtlasPlacement(
                    object_id="shared-texture",
                    texture_path="shared.png",
                    texture_resolution=512,
                    x=0,
                    y=0,
                    size=512,
                ),
                TextureAtlasPlacement(
                    object_id="target-texture",
                    texture_path="target.png",
                    texture_resolution=512,
                    x=512,
                    y=0,
                    size=512,
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            maps = _write_maps(directory)
            active_maps = frozenset(
                (ATLAS_MAP_BASE_COLOR, PBR_MAP_ROUGHNESS)
            )
            first = MaterializedTextureAtlas(
                first_atlas,
                maps,
                active_map_types=active_maps,
            )
            target = MaterializedTextureAtlas(
                target_atlas,
                maps,
                active_map_types=active_maps,
            )
            baked_receiver_keys: list[str] = []

            def fake_bake(layout, occluders, **_kwargs):
                baked_receiver_keys.extend(
                    receiver.key for receiver in layout.receivers
                )
                return CoreSurfaceAoBakeResult(
                    layout=layout,
                    ambient_occlusion=np.full(
                        (layout.resolution, layout.resolution),
                        211,
                        dtype=np.uint8,
                    ),
                    geometry_signature=build_surface_ao_geometry_signature(
                        layout,
                        occluders,
                    ),
                )

            with patch(
                "housemaker.atlas_export.bake_surface_ambient_occlusion",
                side_effect=fake_bake,
            ):
                bake = bake_surface_ambient_occlusion_for_atlas(
                    model,
                    target,
                    atlas_context=(first, target),
                    surface_source_ids={
                        "wall-shared": "shared-texture",
                        "wall-target": "target-texture",
                    },
                )

            self.assertEqual(len(baked_receiver_keys), 1)
            self.assertIn("target-surface", baked_receiver_keys[0])
            self.assertNotIn("shared-surface", baked_receiver_keys[0])

            ao_path = directory / "target-surface-ao.png"
            Image.fromarray(bake.ambient_occlusion, mode="L").save(ao_path)
            cached_target = MaterializedTextureAtlas(
                target_atlas,
                maps,
                active_map_types=active_maps,
                surface_ao_image_path=ao_path,
                surface_ao_geometry_signature=bake.geometry_signature,
            )
            result = apply_texture_atlases_to_export(
                model,
                (first, cached_target),
                surface_source_ids={
                    "wall-shared": "shared-texture",
                    "wall-target": "target-texture",
                },
            )

        atlas_sources = {
            geometry.metadata.get("housemaker_atlas_id"): set(
                geometry.metadata.get("housemaker_atlas_source_ids", ())
            )
            for geometry in result.scene.geometry.values()
            if geometry.metadata.get("housemaker_atlas_id") is not None
        }
        self.assertEqual(atlas_sources["first-atlas"], {"shared-texture"})
        self.assertEqual(atlas_sources["target-atlas"], {"target-texture"})

    def test_repeating_surface_split_interpolates_secondary_ao_uvs(self) -> None:
        mesh = trimesh.Trimesh(
            vertices=np.asarray(
                ((0.0, 0.0, 0.0), (2.5, 0.0, 0.0), (0.0, 2.5, 0.0))
            ),
            faces=np.asarray(((0, 1, 2),), dtype=np.int64),
            vertex_normals=np.asarray(((0.0, 0.0, 1.0),) * 3),
            visual=TextureVisuals(
                uv=np.asarray(((-0.25, -0.25), (2.25, -0.25), (-0.25, 2.25)))
            ),
            process=False,
        )
        mesh.vertex_attributes[PACKED_ORM_AO_UV_ATTRIBUTE] = np.asarray(
            ((0.2, 0.3), (0.45, 0.3), (0.2, 0.55)),
            dtype=np.float32,
        )

        split = _split_repeating_uv_triangles(mesh)

        self.assertGreater(len(split.faces), 1)
        uv1 = split.vertex_attributes[PACKED_ORM_AO_UV_ATTRIBUTE]
        vertices = np.asarray(split.vertices)
        np.testing.assert_allclose(uv1[:, 0], 0.2 + vertices[:, 0] * 0.1)
        np.testing.assert_allclose(uv1[:, 1], 0.3 + vertices[:, 1] * 0.1)
        self.assertTrue(np.all(np.asarray(split.visual.uv) >= 0.0))
        self.assertTrue(np.all(np.asarray(split.visual.uv) <= 1.0))

    def test_bake_exports_uv1_and_packs_ao_into_shared_orm(self) -> None:
        model = _scene_model()
        atlas = _atlas()
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            maps = _write_maps(directory)
            source = MaterializedTextureAtlas(
                atlas,
                maps,
                active_map_types=frozenset(
                    (ATLAS_MAP_BASE_COLOR, PBR_MAP_ROUGHNESS, PBR_MAP_METALLIC)
                ),
            )
            receiver_counts: list[int] = []

            def fake_bake(layout, occluders, **_kwargs):
                receiver_counts.append(len(layout.receivers))
                return CoreSurfaceAoBakeResult(
                    layout=layout,
                    ambient_occlusion=np.full((2048, 2048), 73, dtype=np.uint8),
                    geometry_signature=build_surface_ao_geometry_signature(
                        layout,
                        occluders,
                    ),
                )

            with patch(
                "housemaker.atlas_export.bake_surface_ambient_occlusion",
                side_effect=fake_bake,
            ):
                bake = bake_surface_ambient_occlusion_for_atlas(
                    model,
                    source,
                    surface_source_ids={"wall-1": "surface-texture"},
                )
            self.assertEqual(receiver_counts, [2])
            self.assertIsNotNone(bake.preview_model)
            assert bake.preview_model is not None
            preview_pixels = np.asarray(
                bake.preview_model.mesh.visual.material.baseColorTexture.convert(
                    "L"
                )
            )
            self.assertTrue(np.all(preview_pixels == 73))
            preview_uv = np.asarray(bake.preview_model.mesh.visual.uv)
            self.assertEqual(
                preview_uv.shape,
                (len(bake.preview_model.mesh.vertices), 2),
            )
            self.assertTrue(np.all(preview_uv >= 0.0))
            self.assertTrue(np.all(preview_uv <= 1.0))
            ao_path = directory / "surface-ao.png"
            Image.fromarray(bake.ambient_occlusion, mode="L").save(ao_path)
            cached_source = MaterializedTextureAtlas(
                atlas,
                maps,
                active_map_types=source.active_map_types,
                surface_ao_image_path=ao_path,
                surface_ao_geometry_signature=bake.geometry_signature,
                surface_ao_intensity=0.35,
            )
            cached_preview = prepare_surface_ambient_occlusion_preview_for_atlas(
                model,
                SurfaceAmbientOcclusionAtlasContext(
                    atlas=atlas,
                    active_map_types=source.active_map_types,
                    surface_ao_image_path=ao_path,
                    surface_ao_geometry_signature=bake.geometry_signature,
                ),
                surface_source_ids={"wall-1": "surface-texture"},
            )
            cached_preview_pixels = np.asarray(
                cached_preview.mesh.visual.material.baseColorTexture.convert("L")
            )
            self.assertTrue(np.all(cached_preview_pixels == 73))
            result = apply_texture_atlases_to_export(
                model,
                (cached_source,),
                surface_source_ids={"wall-1": "surface-texture"},
            )

        atlas_mesh = next(iter(result.scene.geometry.values()))
        self.assertIn(PACKED_ORM_AO_UV_ATTRIBUTE, atlas_mesh.vertex_attributes)
        orm = np.asarray(
            atlas_mesh.visual.material.metallicRoughnessTexture.convert("RGBA")
        )
        self.assertTrue(np.all(orm[:, :, 0] == 73))
        self.assertTrue(np.all(orm[:, :, 1] == 170))
        self.assertTrue(np.all(orm[:, :, 2] == 20))

        document = _read_glb_json(result.glb_bytes)
        material = next(
            item for item in document["materials"] if item["name"] == atlas.name
        )
        metallic_roughness = material["pbrMetallicRoughness"][
            "metallicRoughnessTexture"
        ]
        self.assertEqual(
            material["occlusionTexture"]["index"],
            metallic_roughness["index"],
        )
        self.assertNotIn("texCoord", metallic_roughness)
        self.assertEqual(material["occlusionTexture"]["texCoord"], 1)
        self.assertEqual(material["occlusionTexture"]["strength"], 0.35)
        atlas_primitive = next(
            primitive
            for mesh in document["meshes"]
            for primitive in mesh["primitives"]
            if primitive.get("material") == document["materials"].index(material)
        )
        self.assertIn("TEXCOORD_0", atlas_primitive["attributes"])
        self.assertIn("TEXCOORD_1", atlas_primitive["attributes"])
        self.assertNotIn(
            PACKED_ORM_AO_UV_ATTRIBUTE,
            atlas_primitive["attributes"],
        )

    def test_base_only_surface_atlas_exports_ao_only_with_uv1(self) -> None:
        model = _scene_model()
        atlas = _atlas()
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            maps = _write_maps(directory)
            source = MaterializedTextureAtlas(
                atlas,
                maps,
                active_map_types=frozenset((ATLAS_MAP_BASE_COLOR,)),
            )

            def fake_bake(layout, occluders, **_kwargs):
                return CoreSurfaceAoBakeResult(
                    layout=layout,
                    ambient_occlusion=np.full((2048, 2048), 73, dtype=np.uint8),
                    geometry_signature=build_surface_ao_geometry_signature(
                        layout,
                        occluders,
                    ),
                )

            with patch(
                "housemaker.atlas_export.bake_surface_ambient_occlusion",
                side_effect=fake_bake,
            ):
                bake = bake_surface_ambient_occlusion_for_atlas(
                    model,
                    source,
                    surface_source_ids={"wall-1": "surface-texture"},
                )
            ao_path = directory / "base-only-surface-ao.png"
            Image.fromarray(bake.ambient_occlusion, mode="L").save(ao_path)
            cached_source = MaterializedTextureAtlas(
                atlas,
                maps,
                active_map_types=source.active_map_types,
                surface_ao_image_path=ao_path,
                surface_ao_geometry_signature=bake.geometry_signature,
                surface_ao_intensity=0.4,
            )
            result = apply_texture_atlases_to_export(
                model,
                (cached_source,),
                surface_source_ids={"wall-1": "surface-texture"},
            )

        atlas_mesh = next(iter(result.scene.geometry.values()))
        exported_material = atlas_mesh.visual.material
        self.assertIsNone(exported_material.metallicRoughnessTexture)
        self.assertIsNotNone(exported_material.occlusionTexture)
        self.assertTrue(
            np.all(
                np.asarray(exported_material.occlusionTexture.convert("L"))
                == 73
            )
        )

        document = _read_glb_json(result.glb_bytes)
        material = next(
            item for item in document["materials"] if item["name"] == atlas.name
        )
        pbr = material["pbrMetallicRoughness"]
        self.assertNotIn("metallicRoughnessTexture", pbr)
        self.assertEqual(material["occlusionTexture"]["texCoord"], 1)
        self.assertEqual(material["occlusionTexture"]["strength"], 0.4)
        self.assertEqual(len(document["images"]), 2)
        material_index = document["materials"].index(material)
        atlas_primitives = [
            primitive
            for mesh in document["meshes"]
            for primitive in mesh["primitives"]
            if primitive.get("material") == material_index
        ]
        self.assertEqual(len(atlas_primitives), 1)
        self.assertIn("TEXCOORD_0", atlas_primitives[0]["attributes"])
        self.assertIn("TEXCOORD_1", atlas_primitives[0]["attributes"])
        self.assertNotIn(
            PACKED_ORM_AO_UV_ATTRIBUTE,
            atlas_primitives[0]["attributes"],
        )
        reloaded_scene = trimesh.load(
            BytesIO(result.glb_bytes),
            file_type="glb",
            force="scene",
        )
        reloaded_material = next(
            geometry.visual.material
            for geometry in reloaded_scene.geometry.values()
            if getattr(geometry.visual.material, "name", None) == atlas.name
        )
        self.assertIsNone(reloaded_material.metallicRoughnessTexture)
        self.assertIsNotNone(reloaded_material.occlusionTexture)

    def test_mixed_tiled_surface_and_half_mesh_export_uv1_on_every_part(
        self,
    ) -> None:
        model, half_node_name, half_mesh = _tiled_surface_and_half_model()
        atlas = TextureAtlasRecord(
            atlas_id="mixed-surface-ao",
            name="Mixed surface AO",
            resolution=2048,
            placements=[
                TextureAtlasPlacement(
                    object_id="surface-texture",
                    texture_path="surface.png",
                    texture_resolution=512,
                    x=0,
                    y=0,
                    size=512,
                ),
                TextureAtlasPlacement(
                    object_id="authored-half",
                    texture_path="half.png",
                    texture_resolution=512,
                    x=512,
                    y=0,
                    size=512,
                    packing_mode=ATLAS_PACKING_MODE_SYMMETRIC_SQUARE_PAIR,
                    slot_half=ATLAS_SLOT_HALF_LEFT,
                ),
            ],
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            maps = _write_maps(directory)
            materialized = MaterializedTextureAtlas(
                atlas,
                maps,
                active_map_types=frozenset(
                    (ATLAS_MAP_BASE_COLOR, PBR_MAP_ROUGHNESS, PBR_MAP_METALLIC)
                ),
            )
            ray_calls: list[object] = []

            def fake_ray_values(target, *_args, **_kwargs):
                ray_calls.append(target)
                left, top, right, bottom = target.atlas_pixel_bounds
                values = np.full(
                    (bottom - top, right - left),
                    255,
                    dtype=np.uint8,
                )
                coverage = np.zeros(values.shape, dtype=bool)
                values[0, 0] = 80
                coverage[0, 0] = True
                return values, coverage

            with (
                patch(
                    "housemaker.surface_ao_baking.object_ao_baking."
                    "_build_ray_intersector",
                    return_value=object(),
                ),
                patch(
                    "housemaker.surface_ao_baking.object_ao_baking."
                    "_bake_target_mesh",
                    side_effect=fake_ray_values,
                ),
            ):
                bake = bake_surface_ambient_occlusion_for_atlas(
                    model,
                    materialized,
                    surface_source_ids={"tiled-wall": "surface-texture"},
                )
            self.assertIsNotNone(bake.preview_model)
            assert bake.preview_model is not None
            self.assertEqual(len(bake.preview_model.preview_symmetric_objects), 1)
            half_preview = bake.preview_model.preview_symmetric_objects[0]
            self.assertEqual(len(half_preview.meshes), 1)
            self.assertEqual(len(half_preview.mirrored_meshes), 1)
            retained_x = np.asarray(half_preview.meshes[0].vertices)[:, 0]
            mirrored_x = np.asarray(half_preview.mirrored_meshes[0].vertices)[:, 0]
            np.testing.assert_allclose(np.sort(mirrored_x), np.sort(-retained_x))
            np.testing.assert_allclose(
                half_preview.meshes[0].visual.uv,
                half_preview.mirrored_meshes[0].visual.uv,
            )
            ao_path = directory / "mixed-surface-ao.png"
            Image.fromarray(bake.ambient_occlusion, mode="L").save(ao_path)
            result = apply_texture_atlases_to_export(
                model,
                (
                    MaterializedTextureAtlas(
                        atlas,
                        maps,
                        active_map_types=materialized.active_map_types,
                        surface_ao_image_path=ao_path,
                        surface_ao_geometry_signature=bake.geometry_signature,
                    ),
                ),
                surface_source_ids={"tiled-wall": "surface-texture"},
            )

        self.assertEqual(len(ray_calls), 2)
        half_scene_nodes = {
            str(node)
            for node in result.scene.graph.nodes_geometry
            if str(node).startswith("[HALF] ")
        }
        self.assertEqual(half_scene_nodes, {half_node_name})
        half_transform, half_geometry_name = result.scene.graph.get(
            half_node_name
        )
        half_geometry = result.scene.geometry[half_geometry_name]
        self.assertEqual(len(half_geometry.faces), 1)
        authored_world_vertices = trimesh.transform_points(
            np.asarray(half_geometry.vertices),
            half_transform,
        )
        self.assertTrue(np.all(authored_world_vertices[:, 0] >= 1.0))

        document = _read_glb_json(result.glb_bytes)
        material_index = next(
            index
            for index, material in enumerate(document["materials"])
            if material["name"] == atlas.name
        )
        atlas_primitives = [
            primitive
            for mesh in document["meshes"]
            for primitive in mesh["primitives"]
            if primitive.get("material") == material_index
        ]
        self.assertEqual(len(atlas_primitives), 2)
        for primitive in atlas_primitives:
            with self.subTest(attributes=primitive["attributes"]):
                self.assertIn("TEXCOORD_0", primitive["attributes"])
                self.assertIn("TEXCOORD_1", primitive["attributes"])
                self.assertNotIn(
                    PACKED_ORM_AO_UV_ATTRIBUTE,
                    primitive["attributes"],
                )
        self.assertNotIn(PACKED_ORM_AO_UV_ATTRIBUTE, json.dumps(document))

        half_nodes = [
            node
            for node in document["nodes"]
            if node.get("name") == half_node_name
        ]
        self.assertEqual(len(half_nodes), 1)
        half_node = half_nodes[0]
        self.assertEqual(half_node.get("extras"), {"halfMesh": half_mesh})
        self.assertNotIn("children", half_node)
        half_gltf_mesh = document["meshes"][half_node["mesh"]]
        self.assertEqual(
            half_gltf_mesh.get("extras", {}).get("halfMesh"),
            half_mesh,
        )
        half_primitives = half_gltf_mesh["primitives"]
        self.assertEqual(len(half_primitives), 1)
        half_index_accessor = document["accessors"][
            half_primitives[0]["indices"]
        ]
        self.assertEqual(half_index_accessor["count"], 3)

        index_counts = sorted(
            document["accessors"][primitive["indices"]]["count"]
            for primitive in atlas_primitives
        )
        self.assertEqual(index_counts[0], 3)
        self.assertGreater(index_counts[1], 3)

    def test_export_rejects_a_surface_ao_bake_after_scene_changes(self) -> None:
        model = _scene_model()
        atlas = _atlas()
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            maps = _write_maps(directory)
            source = MaterializedTextureAtlas(
                atlas,
                maps,
                active_map_types=frozenset(
                    (ATLAS_MAP_BASE_COLOR, PBR_MAP_ROUGHNESS)
                ),
            )

            def fake_bake(layout, occluders, **_kwargs):
                return CoreSurfaceAoBakeResult(
                    layout=layout,
                    ambient_occlusion=np.full((2048, 2048), 255, dtype=np.uint8),
                    geometry_signature=build_surface_ao_geometry_signature(
                        layout,
                        occluders,
                    ),
                )

            with patch(
                "housemaker.atlas_export.bake_surface_ambient_occlusion",
                side_effect=fake_bake,
            ):
                bake = bake_surface_ambient_occlusion_for_atlas(
                    model,
                    source,
                    surface_source_ids={"wall-1": "surface-texture"},
                )
            ao_path = directory / "surface-ao.png"
            Image.fromarray(bake.ambient_occlusion, mode="L").save(ao_path)
            cached = MaterializedTextureAtlas(
                atlas,
                maps,
                active_map_types=source.active_map_types,
                surface_ao_image_path=ao_path,
                surface_ao_geometry_signature=bake.geometry_signature,
            )
            model.scene.graph.update(
                frame_from=model.scene.graph.base_frame,
                frame_to="object",
                matrix=trimesh.transformations.translation_matrix((0.5, 0.0, 0.0)),
            )

            with self.assertRaisesRegex(ValueError, "out of date"):
                apply_texture_atlases_to_export(
                    model,
                    (cached,),
                    surface_source_ids={"wall-1": "surface-texture"},
                )


if __name__ == "__main__":
    unittest.main()
