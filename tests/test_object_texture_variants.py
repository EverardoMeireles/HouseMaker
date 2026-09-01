# ### Imports ###
from __future__ import annotations

import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import trimesh
from PIL import Image
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.generation_workspace import GenerationWorkspace
from housemaker.glb import import_generated_glb
from housemaker.meshy_generation import MeshyGenerationResult
import housemaker.object_texture_variants as object_texture_variants
from housemaker.object_texture_variants import (
    ATLAS_MAP_BASE_COLOR,
    PBR_MAP_METALLIC,
    PBR_MAP_NORMAL,
    PBR_MAP_ROUGHNESS,
    TEXTURE_RESOLUTIONS,
    build_object_texture_variants,
    build_object_texture_variants_from_texture,
    replace_object_base_color_texture_from_glb,
)


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])


# ### Fixture helpers ###
def _textured_glb(
    *,
    distinct_second_texture: bool = False,
    texture_size: tuple[int, int] = (2048, 2048),
    texture_color: tuple[int, int, int, int] = (24, 80, 160, 96),
) -> bytes:
    first = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    first.visual = TextureVisuals(
        uv=np.linspace(0.0, 1.0, len(first.vertices) * 2).reshape((-1, 2)),
        material=PBRMaterial(
            baseColorTexture=Image.new(
                "RGBA",
                texture_size,
                texture_color,
            )
        ),
    )
    second = trimesh.creation.icosphere(subdivisions=1, radius=0.25)
    second.visual = TextureVisuals(
        uv=np.zeros((len(second.vertices), 2), dtype=float),
        material=PBRMaterial(
            baseColorTexture=Image.new(
                "RGBA",
                texture_size,
                (
                    (210, 40, 12, 224)
                    if distinct_second_texture
                    else texture_color
                ),
            )
        ),
    )
    scene = trimesh.Scene()
    scene.add_geometry(first, geom_name="box", node_name="box-node")
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = (4.0, 5.0, 6.0)
    scene.add_geometry(
        second,
        geom_name="sphere",
        node_name="sphere-node",
        transform=transform,
    )
    return bytes(scene.export(file_type="glb"))


def _nonuniform_2048_rgba_glb() -> tuple[bytes, np.ndarray]:
    """Build a compact atlas whose direct and chained area resizes differ."""

    averaging_block = np.asarray(
        (
            (5, 7, 5, 5),
            (7, 5, 0, 6),
            (7, 4, 6, 0),
            (1, 1, 6, 6),
        ),
        dtype=np.uint8,
    )
    channel = np.tile(averaging_block, (512, 512))
    source_rgba = np.empty((2048, 2048, 4), dtype=np.uint8)
    source_rgba[:, :, 0] = channel
    source_rgba[:, :, 1] = 255 - channel
    source_rgba[:, :, 2] = channel * 31
    source_rgba[:, :, 3] = channel * 37

    mesh = trimesh.creation.box()
    mesh.visual = TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2), dtype=float),
        material=PBRMaterial(baseColorTexture=Image.fromarray(source_rgba)),
    )
    return bytes(trimesh.Scene(mesh).export(file_type="glb")), source_rgba


def _untextured_uv_glb() -> bytes:
    """Build geometry-only GLB content with authoritative UV coordinates."""

    mesh = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    mesh.visual = TextureVisuals(
        uv=np.linspace(0.05, 0.95, len(mesh.vertices) * 2).reshape((-1, 2))
    )
    transform = trimesh.transformations.translation_matrix((3.0, 4.0, 5.0))
    scene = trimesh.Scene()
    scene.add_geometry(
        mesh,
        geom_name="uv-box",
        node_name="uv-box-node",
        transform=transform,
    )
    return bytes(scene.export(file_type="glb"))


def _texture_images(glb_bytes: bytes) -> list[Image.Image]:
    scene = trimesh.load(
        BytesIO(glb_bytes),
        file_type="glb",
        force="scene",
        process=False,
    )
    return [
        geometry.visual.material.baseColorTexture
        for _name, geometry in sorted(scene.geometry.items())
    ]


def _rewrite_uv_layout(
    glb_bytes: bytes,
    *,
    scale: float = 0.5,
    offset: float = 0.25,
) -> bytes:
    scene = trimesh.load(
        BytesIO(glb_bytes),
        file_type="glb",
        force="scene",
        process=False,
    )
    for geometry in scene.geometry.values():
        uv = getattr(geometry.visual, "uv", None)
        if uv is None:
            continue
        geometry.visual.uv = (
            np.asarray(uv, dtype=float) * float(scale) + float(offset)
        )
    return bytes(scene.export(file_type="glb"))


# ### Variant algorithm tests ###
class ObjectTextureVariantAlgorithmTests(unittest.TestCase):
    def test_builds_and_splits_pbr_maps_at_every_resolution(self) -> None:
        scene = trimesh.load(
            BytesIO(_textured_glb()),
            file_type="glb",
            force="scene",
            process=False,
        )
        for geometry in scene.geometry.values():
            geometry.visual.material.normalTexture = Image.new(
                "RGBA",
                (2048, 2048),
                (96, 144, 240, 255),
            )
            geometry.visual.material.metallicRoughnessTexture = Image.new(
                "RGBA",
                (2048, 2048),
                (0, 72, 180, 255),
            )

        variants = build_object_texture_variants(
            bytes(scene.export(file_type="glb"))
        )

        self.assertIsNotNone(variants)
        assert variants is not None
        self.assertEqual(
            variants.available_map_types,
            (
                ATLAS_MAP_BASE_COLOR,
                PBR_MAP_NORMAL,
                PBR_MAP_ROUGHNESS,
                PBR_MAP_METALLIC,
            ),
        )
        assert variants.map_png_by_resolution is not None
        expected_pixels = {
            PBR_MAP_NORMAL: (96, 144, 240, 255),
            PBR_MAP_ROUGHNESS: (72, 72, 72, 255),
            PBR_MAP_METALLIC: (180, 180, 180, 255),
        }
        for resolution in TEXTURE_RESOLUTIONS:
            for map_type, expected_pixel in expected_pixels.items():
                with self.subTest(
                    resolution=resolution,
                    map_type=map_type,
                ):
                    with Image.open(
                        BytesIO(
                            variants.map_png_by_resolution[map_type][
                                resolution
                            ]
                        )
                    ) as image:
                        self.assertEqual(image.size, (resolution, resolution))
                        self.assertEqual(image.getpixel((0, 0)), expected_pixel)

    def test_native_2048_rgba_source_is_preserved_and_directly_downsized(
        self,
    ) -> None:
        source_glb, authored_source_rgba = _nonuniform_2048_rgba_glb()
        provider_source_rgba = np.asarray(
            _texture_images(source_glb)[0].convert("RGBA"),
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(provider_source_rgba, authored_source_rgba)

        variants = build_object_texture_variants(source_glb)

        self.assertIsNotNone(variants)
        assert variants is not None
        expected_1024 = cv2.resize(
            provider_source_rgba,
            (1024, 1024),
            interpolation=cv2.INTER_AREA,
        )
        expected_512 = cv2.resize(
            provider_source_rgba,
            (512, 512),
            interpolation=cv2.INTER_AREA,
        )
        chained_512 = cv2.resize(
            expected_1024,
            (512, 512),
            interpolation=cv2.INTER_AREA,
        )

        np.testing.assert_array_equal(
            variants.preview_rgba_by_resolution[2048],
            provider_source_rgba,
        )
        np.testing.assert_array_equal(
            variants.preview_rgba_by_resolution[1024],
            expected_1024,
        )
        np.testing.assert_array_equal(
            variants.preview_rgba_by_resolution[512],
            expected_512,
        )
        self.assertGreater(np.count_nonzero(expected_512 != chained_512), 0)

        png_2048 = np.asarray(
            Image.open(
                BytesIO(variants.texture_png_by_resolution[2048])
            ).convert("RGBA"),
            dtype=np.uint8,
        )
        embedded_2048 = np.asarray(
            _texture_images(variants.glb_by_resolution[2048])[0].convert("RGBA"),
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(png_2048, provider_source_rgba)
        np.testing.assert_array_equal(embedded_2048, provider_source_rgba)

    def test_builds_three_sizes_and_preserves_scene_geometry_uvs_and_transforms(
        self,
    ) -> None:
        source_glb = _textured_glb()
        source_scene = trimesh.load(
            BytesIO(source_glb),
            file_type="glb",
            force="scene",
            process=False,
        )

        with patch.object(
            object_texture_variants,
            "_resize_rgba",
            wraps=object_texture_variants._resize_rgba,
        ) as resize_rgba:
            variants = build_object_texture_variants(source_glb)

        self.assertIsNotNone(variants)
        assert variants is not None
        self.assertEqual(set(variants.glb_by_resolution), set(TEXTURE_RESOLUTIONS))
        self.assertEqual(
            [
                (call.args[0].shape, call.args[1], call.args[2])
                for call in resize_rgba.call_args_list
            ],
            [
                ((2048, 2048, 4), 1024, cv2.INTER_AREA),
                ((2048, 2048, 4), 512, cv2.INTER_AREA),
            ],
        )
        np.testing.assert_array_equal(
            variants.preview_rgba_by_resolution[2048],
            np.asarray(
                source_scene.geometry["box"].visual.material.baseColorTexture,
                dtype=np.uint8,
            ),
        )
        for resolution in TEXTURE_RESOLUTIONS:
            textures = _texture_images(variants.glb_by_resolution[resolution])
            self.assertEqual(
                [texture.size for texture in textures],
                [(resolution, resolution), (resolution, resolution)],
            )
            self.assertEqual(textures[0].getpixel((0, 0)), (24, 80, 160, 96))
            self.assertEqual(textures[1].getpixel((0, 0)), (24, 80, 160, 96))
            variant_scene = trimesh.load(
                BytesIO(variants.glb_by_resolution[resolution]),
                file_type="glb",
                force="scene",
                process=False,
            )
            self.assertEqual(set(variant_scene.geometry), set(source_scene.geometry))
            for geometry_name in source_scene.geometry:
                np.testing.assert_allclose(
                    variant_scene.geometry[geometry_name].vertices,
                    source_scene.geometry[geometry_name].vertices,
                )
                np.testing.assert_allclose(
                    variant_scene.geometry[geometry_name].visual.uv,
                    source_scene.geometry[geometry_name].visual.uv,
                )
            np.testing.assert_allclose(
                variant_scene.graph.get("sphere-node")[0],
                source_scene.graph.get("sphere-node")[0],
            )

    def test_rejects_multiple_distinct_base_color_atlases(self) -> None:
        with self.assertRaisesRegex(ValueError, "more than one distinct"):
            build_object_texture_variants(
                _textured_glb(distinct_second_texture=True)
            )

    def test_rejects_a_non_2048_meshy_texture_instead_of_upscaling_it(self) -> None:
        with self.assertRaisesRegex(ValueError, "not 2048 x 2048"):
            build_object_texture_variants(
                _textured_glb(texture_size=(1024, 1024))
            )

    def test_replacement_atlas_preserves_geometry_uvs_and_scene_transforms(
        self,
    ) -> None:
        source_glb = _textured_glb()
        replacement = np.full(
            (2048, 2048, 4),
            (190, 70, 35, 231),
            dtype=np.uint8,
        )
        replacement[0, 0] = (1, 2, 3, 4)
        replacement_output = BytesIO()
        Image.fromarray(replacement, mode="RGBA").save(
            replacement_output,
            format="PNG",
        )
        source_scene = trimesh.load(
            BytesIO(source_glb),
            file_type="glb",
            force="scene",
            process=False,
        )

        variants = build_object_texture_variants_from_texture(
            source_glb,
            replacement_output.getvalue(),
        )

        np.testing.assert_array_equal(
            variants.preview_rgba_by_resolution[2048],
            replacement,
        )
        for resolution in TEXTURE_RESOLUTIONS:
            variant_scene = trimesh.load(
                BytesIO(variants.glb_by_resolution[resolution]),
                file_type="glb",
                force="scene",
                process=False,
            )
            self.assertEqual(set(variant_scene.geometry), set(source_scene.geometry))
            for geometry_name in source_scene.geometry:
                np.testing.assert_allclose(
                    variant_scene.geometry[geometry_name].vertices,
                    source_scene.geometry[geometry_name].vertices,
                )
                np.testing.assert_allclose(
                    variant_scene.geometry[geometry_name].visual.uv,
                    source_scene.geometry[geometry_name].visual.uv,
                )
            np.testing.assert_allclose(
                variant_scene.graph.get("sphere-node")[0],
                source_scene.graph.get("sphere-node")[0],
            )
        embedded = np.asarray(
            _texture_images(variants.glb_by_resolution[2048])[0].convert("RGBA"),
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(embedded, replacement)

    def test_replacement_atlas_requires_a_static_2048_png(self) -> None:
        source_glb = _textured_glb()
        output = BytesIO()
        Image.new("RGBA", (1024, 1024), (1, 2, 3, 255)).save(
            output,
            format="PNG",
        )

        with self.assertRaisesRegex(ValueError, "2048 x 2048"):
            build_object_texture_variants_from_texture(
                source_glb,
                output.getvalue(),
            )
        with self.assertRaisesRegex(ValueError, "PNG"):
            build_object_texture_variants_from_texture(
                source_glb,
                b"not a png",
            )

    def test_provider_texture_replaces_only_the_authoritative_model_atlas(
        self,
    ) -> None:
        model_glb = _textured_glb(texture_size=(1024, 1024))
        model_scene = trimesh.load(
            BytesIO(model_glb),
            file_type="glb",
            force="scene",
            process=False,
        )
        texture_source_glb, generated_texture = _nonuniform_2048_rgba_glb()
        texture_source_glb = _rewrite_uv_layout(
            texture_source_glb,
            scale=0.2,
            offset=0.6,
        )

        replaced_glb = replace_object_base_color_texture_from_glb(
            model_glb,
            texture_source_glb,
        )

        replaced_scene = trimesh.load(
            BytesIO(replaced_glb),
            file_type="glb",
            force="scene",
            process=False,
        )
        self.assertEqual(set(replaced_scene.geometry), set(model_scene.geometry))
        for geometry_name in model_scene.geometry:
            np.testing.assert_allclose(
                replaced_scene.geometry[geometry_name].vertices,
                model_scene.geometry[geometry_name].vertices,
            )
            np.testing.assert_allclose(
                replaced_scene.geometry[geometry_name].faces,
                model_scene.geometry[geometry_name].faces,
            )
            np.testing.assert_allclose(
                replaced_scene.geometry[geometry_name].visual.uv,
                model_scene.geometry[geometry_name].visual.uv,
            )
        np.testing.assert_allclose(
            replaced_scene.graph.get("sphere-node")[0],
            model_scene.graph.get("sphere-node")[0],
        )
        for texture in _texture_images(replaced_glb):
            self.assertEqual(texture.size, (2048, 2048))
            np.testing.assert_array_equal(
                np.asarray(texture.convert("RGBA"), dtype=np.uint8),
                generated_texture,
            )

    def test_provider_texture_attaches_to_untextured_authoritative_uvs(
        self,
    ) -> None:
        model_glb = _untextured_uv_glb()
        model_scene = trimesh.load(
            BytesIO(model_glb),
            file_type="glb",
            force="scene",
            process=False,
        )
        texture_source_glb, generated_texture = _nonuniform_2048_rgba_glb()

        replaced_glb = replace_object_base_color_texture_from_glb(
            model_glb,
            texture_source_glb,
        )

        replaced_scene = trimesh.load(
            BytesIO(replaced_glb),
            file_type="glb",
            force="scene",
            process=False,
        )
        np.testing.assert_allclose(
            replaced_scene.geometry["uv-box"].vertices,
            model_scene.geometry["uv-box"].vertices,
        )
        np.testing.assert_array_equal(
            replaced_scene.geometry["uv-box"].faces,
            model_scene.geometry["uv-box"].faces,
        )
        np.testing.assert_allclose(
            replaced_scene.geometry["uv-box"].visual.uv,
            model_scene.geometry["uv-box"].visual.uv,
        )
        np.testing.assert_allclose(
            replaced_scene.graph.get("uv-box-node")[0],
            model_scene.graph.get("uv-box-node")[0],
        )
        embedded_texture = _texture_images(replaced_glb)[0]
        self.assertEqual(embedded_texture.size, (2048, 2048))
        np.testing.assert_array_equal(
            np.asarray(embedded_texture.convert("RGBA"), dtype=np.uint8),
            generated_texture,
        )

    def test_provider_pbr_maps_preserve_the_local_glass_alpha_mask(self) -> None:
        model_scene = trimesh.load(
            BytesIO(
                _textured_glb(
                    texture_size=(1024, 1024),
                    texture_color=(12, 34, 56, 48),
                )
            ),
            file_type="glb",
            force="scene",
            process=False,
        )
        for geometry in model_scene.geometry.values():
            material = geometry.visual.material
            material.name = (
                object_texture_variants.HOUSEMAKER_GLASS_MATERIAL_NAME
            )
            material.alphaMode = "BLEND"

        provider_scene = trimesh.load(
            BytesIO(_textured_glb(texture_color=(91, 72, 53, 255))),
            file_type="glb",
            force="scene",
            process=False,
        )
        for geometry in provider_scene.geometry.values():
            material = geometry.visual.material
            material.normalTexture = Image.new(
                "RGBA",
                (2048, 2048),
                (110, 140, 240, 255),
            )
            material.metallicRoughnessTexture = Image.new(
                "RGBA",
                (2048, 2048),
                (0, 9, 0, 255),
            )

        replaced_glb = replace_object_base_color_texture_from_glb(
            bytes(model_scene.export(file_type="glb")),
            bytes(provider_scene.export(file_type="glb")),
        )

        replaced_scene = trimesh.load(
            BytesIO(replaced_glb),
            file_type="glb",
            force="scene",
            process=False,
        )
        for geometry in replaced_scene.geometry.values():
            material = geometry.visual.material
            self.assertEqual(
                material.name,
                object_texture_variants.HOUSEMAKER_GLASS_MATERIAL_NAME,
            )
            self.assertEqual(material.alphaMode, "BLEND")
            base_color = np.asarray(
                material.baseColorTexture.convert("RGBA"),
                dtype=np.uint8,
            )
            np.testing.assert_array_equal(
                base_color[0, 0],
                np.asarray((91, 72, 53, 48), dtype=np.uint8),
            )
            np.testing.assert_array_equal(
                np.asarray(
                    material.normalTexture.convert("RGBA"),
                    dtype=np.uint8,
                )[0, 0],
                np.asarray((110, 140, 240, 255), dtype=np.uint8),
            )
            np.testing.assert_array_equal(
                np.asarray(
                    material.metallicRoughnessTexture.convert("RGBA"),
                    dtype=np.uint8,
                )[0, 0],
                np.asarray((0, 9, 0, 255), dtype=np.uint8),
            )

    def test_provider_texture_requires_one_shared_2048_atlas(self) -> None:
        model_glb = _textured_glb(texture_size=(1024, 1024))

        with self.assertRaisesRegex(ValueError, "not 2048 x 2048"):
            replace_object_base_color_texture_from_glb(
                model_glb,
                _textured_glb(texture_size=(1024, 1024)),
            )
        with self.assertRaisesRegex(ValueError, "more than one distinct"):
            replace_object_base_color_texture_from_glb(
                model_glb,
                _textured_glb(distinct_second_texture=True),
            )


# ### Workspace persistence tests ###
class ObjectTextureVariantWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.asset_directory = Path(self.temporary_directory.name) / "generated"
        self.workspace = GenerationWorkspace(asset_directory=self.asset_directory)

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        self.temporary_directory.cleanup()

    def _generated_model_with_variants(self):
        source_glb = _textured_glb()
        model = import_generated_glb(source_glb)
        model.object_texture_variants = build_object_texture_variants(source_glb)
        return model

    def test_selection_round_trips_and_exact_variant_resolver_is_stable(self) -> None:
        model = self._generated_model_with_variants()
        result = MeshyGenerationResult("task", model.glb_bytes, "Table")
        self.workspace._handle_generation_succeeded(result, model)
        record = self.workspace.get_data().generated_objects[0]

        self.assertEqual(len(self.workspace.texture_view.entries), 3)
        self.assertIsNone(
            self.workspace._generated_model_cache[
                record.object_id
            ].object_texture_variants
        )
        self.assertEqual(record.pipeline["selected_texture_resolution"], 1024)
        exact_512 = self.workspace.get_texture_variant(record.object_id, 512)
        self.assertIsNotNone(exact_512)
        assert exact_512 is not None
        self.assertEqual(exact_512.resolution, 512)
        self.assertTrue(exact_512.texture_asset_path.is_file())

        self.workspace.texture_view.select_atlas(
            f"{record.object_id}:resolution:2048"
        )
        selected_data = self.workspace.get_data()
        selected = selected_data.generated_objects[0]
        self.assertEqual(selected.pipeline["selected_texture_resolution"], 2048)
        self.assertTrue(str(selected.asset_path).endswith("texture-2048.glb"))
        active = self.workspace.get_active_texture_variant(record.object_id)
        self.assertIsNotNone(active)
        assert active is not None
        self.assertEqual(active.resolution, 2048)
        self.assertEqual(exact_512.resolution, 512)

        self.workspace.set_data(selected_data)
        self.assertEqual(
            self.workspace.texture_view.selected_atlas_id,
            f"{record.object_id}:resolution:2048",
        )
        self.assertEqual(
            self.workspace.get_active_texture_variant(record.object_id).resolution,
            2048,
        )

        all_paths = []
        for variant in selected.pipeline["texture_variants"].values():
            all_paths.extend(
                self.asset_directory / variant[path_key]
                for path_key in ("glb_asset_path", "texture_asset_path")
            )
            all_paths.extend(
                self.asset_directory / raw_path
                for raw_path in variant.get(
                    "map_texture_asset_paths",
                    {},
                ).values()
            )
        all_paths = list(dict.fromkeys(all_paths))
        self.assertTrue(all(path.is_file() for path in all_paths))
        exact_512.glb_asset_path.unlink()
        self.assertIsNone(
            self.workspace.get_texture_variant(record.object_id, 512)
        )
        png_only_512 = self.workspace.get_texture_image_variant(
            record.object_id,
            512,
        )
        self.assertIsNotNone(png_only_512)
        self.assertEqual(png_only_512.resolution, 512)
        self.assertTrue(png_only_512.texture_asset_path.is_file())
        self.workspace.set_data(selected_data)
        self.assertEqual(len(self.workspace.texture_view.entries), 2)
        self.assertTrue(self.workspace.delete_generated_object(record.object_id))
        self.assertTrue(all(not path.exists() for path in all_paths))

    def test_wireframe_uv_overlay_tracks_resolution_in_external_mode(self) -> None:
        model = self._generated_model_with_variants()
        result = MeshyGenerationResult("task", model.glb_bytes, "Table")
        self.workspace._handle_generation_succeeded(result, model)
        record = self.workspace.get_data().generated_objects[0]
        initial_triangles = self.workspace.texture_view.uv_overlay_triangles
        variant_2048 = self.workspace.get_texture_variant(
            record.object_id,
            2048,
        )
        self.assertIsNotNone(variant_2048)
        assert variant_2048 is not None
        modified_glb = _rewrite_uv_layout(
            variant_2048.glb_asset_path.read_bytes()
        )
        variant_2048.glb_asset_path.write_bytes(modified_glb)

        self.workspace.wireframe_checkbox.setChecked(True)
        self.workspace.set_external_3d_viewer_active(True)
        self.workspace.texture_view.select_atlas(
            f"{record.object_id}:resolution:2048"
        )
        _qt_application.processEvents()

        selected_triangles = self.workspace.texture_view.uv_overlay_triangles
        self.assertTrue(initial_triangles)
        self.assertTrue(selected_triangles)
        self.assertNotEqual(selected_triangles, initial_triangles)
        self.assertTrue(self.workspace.texture_view.uv_overlay_enabled)
        self.assertTrue(self.workspace.result_view.get_wireframe_enabled())
        self.assertIs(
            self.workspace.right_view_stack.currentWidget(),
            self.workspace.texture_view_page,
        )
        self.assertEqual(
            self.workspace.get_data().generated_objects[0].pipeline[
                "selected_texture_resolution"
            ],
            2048,
        )
        self.assertEqual(self.workspace.result_view.model.glb_bytes, modified_glb)

        self.workspace.set_external_3d_viewer_active(False)

        self.assertTrue(self.workspace.texture_view.uv_overlay_enabled)
        self.assertEqual(
            self.workspace.texture_view.uv_overlay_triangles,
            selected_triangles,
        )
        self.assertIs(
            self.workspace.right_view_stack.currentWidget(),
            self.workspace.object_3d_page,
        )

    def test_wireframe_uv_overlay_follows_generated_object_selection(self) -> None:
        first_model = self._generated_model_with_variants()
        self.workspace._handle_generation_succeeded(
            MeshyGenerationResult("first-task", first_model.glb_bytes, "First"),
            first_model,
        )
        first_triangles = self.workspace.texture_view.uv_overlay_triangles

        second_glb = _rewrite_uv_layout(_textured_glb(), scale=0.6, offset=0.15)
        second_model = import_generated_glb(second_glb)
        second_model.object_texture_variants = build_object_texture_variants(second_glb)
        self.workspace.wireframe_checkbox.setChecked(True)
        self.workspace._handle_generation_succeeded(
            MeshyGenerationResult("second-task", second_glb, "Second"),
            second_model,
        )
        second_triangles = self.workspace.texture_view.uv_overlay_triangles

        self.assertNotEqual(second_triangles, first_triangles)
        self.assertTrue(self.workspace.texture_view.uv_overlay_enabled)
        self.workspace.generated_objects_list.setCurrentRow(0)
        _qt_application.processEvents()

        self.assertEqual(
            self.workspace.texture_view.uv_overlay_triangles,
            first_triangles,
        )
        self.assertTrue(self.workspace.texture_view.uv_overlay_enabled)

    def test_variant_persistence_rolls_back_after_an_injected_write_failure(
        self,
    ) -> None:
        model = self._generated_model_with_variants()
        result = MeshyGenerationResult("task", model.glb_bytes, "Table")
        original_persist = self.workspace._persist_meshy_named_asset
        call_count = 0

        def fail_on_fourth_write(file_name: str, payload: bytes) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 4:
                raise OSError("injected write failure")
            return original_persist(file_name, payload)

        with patch.object(
            self.workspace,
            "_persist_meshy_named_asset",
            side_effect=fail_on_fourth_write,
        ), patch("housemaker.generation_workspace.QMessageBox.warning"):
            self.workspace._handle_generation_succeeded(result, model)

        self.assertEqual(self.workspace.get_data().generated_objects, [])
        self.assertEqual(
            (
                list(self.asset_directory.glob("*"))
                if self.asset_directory.exists()
                else []
            ),
            [],
        )

    def test_missing_active_variant_selects_first_complete_resolution(self) -> None:
        model = self._generated_model_with_variants()
        result = MeshyGenerationResult("task", model.glb_bytes, "Table")
        self.workspace._handle_generation_succeeded(result, model)
        object_id = self.workspace.get_data().generated_objects[0].object_id
        self.workspace.texture_view.select_atlas(
            f"{object_id}:resolution:2048"
        )
        saved_data = self.workspace.get_data()
        active_2048 = self.workspace.get_active_texture_variant(object_id)
        self.assertEqual(active_2048.resolution, 2048)
        active_2048.glb_asset_path.unlink()

        self.workspace.set_data(saved_data)

        repaired = self.workspace.get_data().generated_objects[0]
        self.assertEqual(repaired.pipeline["selected_texture_resolution"], 1024)
        self.assertTrue(str(repaired.asset_path).endswith("texture-1024.glb"))
        self.assertEqual(
            self.workspace.texture_view.selected_atlas_id,
            f"{object_id}:resolution:1024",
        )
        self.assertIsNotNone(self.workspace.result_view.model)

    def test_runtime_repair_emits_one_synchronized_data_change(self) -> None:
        model = self._generated_model_with_variants()
        result = MeshyGenerationResult("task", model.glb_bytes, "Table")
        self.workspace._handle_generation_succeeded(result, model)
        object_id = self.workspace.get_data().generated_objects[0].object_id
        self.workspace.texture_view.select_atlas(
            f"{object_id}:resolution:2048"
        )
        active = self.workspace.get_active_texture_variant(object_id)
        active.glb_asset_path.unlink()
        changed_spy = QSignalSpy(self.workspace.data_changed)

        selected_record = self.workspace.get_data().generated_objects[0]
        self.workspace._display_generated_object(selected_record)

        self.assertEqual(changed_spy.count(), 1)
        emitted_data = changed_spy.at(0)[0]
        repaired = emitted_data.generated_objects[0]
        self.assertEqual(repaired.pipeline["selected_texture_resolution"], 1024)
        self.assertTrue(str(repaired.asset_path).endswith("texture-1024.glb"))

    def test_variant_persistence_rolls_back_after_reimport_failure(self) -> None:
        model = self._generated_model_with_variants()
        result = MeshyGenerationResult("task", model.glb_bytes, "Table")

        with patch(
            "housemaker.generation_workspace.import_generated_glb",
            side_effect=ValueError("injected reimport failure"),
        ), patch("housemaker.generation_workspace.QMessageBox.warning"):
            self.workspace._handle_generation_succeeded(result, model)

        self.assertEqual(self.workspace.get_data().generated_objects, [])
        self.assertEqual(
            (
                list(self.asset_directory.glob("*"))
                if self.asset_directory.exists()
                else []
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
