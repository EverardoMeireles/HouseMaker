# ### Environment setup ###
from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, call, patch

import numpy as np
import pyqtgraph.opengl as gl
import trimesh
from OpenGL import GL

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PIL import Image
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QFocusEvent, QKeyEvent, QOpenGLContext, QVector3D
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication, QPushButton
from trimesh.visual.material import MultiMaterial, PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.camera_indicators import INDICATOR_SELECTED_COLOR
from housemaker.camera_models import CameraPose
from housemaker.first_person_navigation import (
    DEFAULT_FIRST_PERSON_NAVIGATION_MODE,
    FIRST_PERSON_NAVIGATION_MODE_GRAVITY,
    FIRST_PERSON_NAVIGATION_MODE_NOCLIP,
)
from housemaker.glass_material import build_housemaker_glass_material
from housemaker.glb import (
    GeneratedModel,
    PreviewPlacedObject,
    PreviewTexturedSurface,
    PreviewTexturedWall,
    import_generated_glb,
)
from housemaker.object_texture_variants import (
    PBR_MAP_METALLIC,
    PBR_MAP_NORMAL,
    PBR_MAP_ROUGHNESS,
)
from housemaker.texture_color_balance import (
    ColorBalanceAdjustment,
    TextureColorBalanceSettings,
)
from housemaker.unused_face_removal import ALL_CAMERA_IDS
from housemaker.viewer import (
    _SHARED_GL_TEXTURE_CACHE,
    INITIAL_TEXTURE_PROXY_DIMENSION,
    MAX_TEXTURE_PREVIEW_DIMENSION,
    NAVIGATION_MODE_FIRST_PERSON,
    NAVIGATION_MODE_ORBIT,
    GlbViewerWidget,
    SelectableGLViewWidget,
    TexturedMeshItem,
    TexturePreviewBundle,
    _build_texture_mesh_data,
    _decode_texture_rgba,
    _draw_texture_ranges,
    _limit_texture_preview_size,
    _project_vertices_to_view,
    _set_float_uniform,
    _TextureDrawRange,
    build_initial_texture_proxy_bundle,
)


# ### Test doubles ###
class _FakeOpenGlContext:
    """Weak-referenceable identity used to exercise context-local caches."""


class _EqualFakeOpenGlContext:
    """Distinct context identity that deliberately compares equal."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _EqualFakeOpenGlContext)

    def __hash__(self) -> int:
        return 1


class FakeMouseMoveEvent:
    def __init__(
        self,
        *,
        position: QPointF,
        buttons: Qt.MouseButton,
        modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
    ) -> None:
        self._position = position
        self._buttons = buttons
        self._modifiers = modifiers
        self.was_accepted = False

    def position(self) -> QPointF:
        return self._position

    def buttons(self) -> Qt.MouseButton:
        return self._buttons

    def modifiers(self) -> Qt.KeyboardModifier:
        return self._modifiers

    def accept(self) -> None:
        self.was_accepted = True


class FakeWheelEvent:
    def __init__(self, delta: int) -> None:
        self._delta = int(delta)
        self.was_accepted = False

    def angleDelta(self) -> QPoint:
        return QPoint(0, self._delta)

    def accept(self) -> None:
        self.was_accepted = True


class FakeMousePressEvent:
    def __init__(
        self,
        *,
        button: Qt.MouseButton,
        position: QPointF = QPointF(),
    ) -> None:
        self._button = button
        self._position = position
        self.was_accepted = False

    def button(self) -> Qt.MouseButton:
        return self._button

    def position(self) -> QPointF:
        return self._position

    def accept(self) -> None:
        self.was_accepted = True


# ### Fixture helpers ###
def _build_generated_model(*, textured: bool = False) -> GeneratedModel:
    if not textured:
        mesh = trimesh.creation.box()
        return GeneratedModel(
            mesh=mesh,
            scene=trimesh.Scene(mesh),
            glb_bytes=b"",
        )

    mesh = trimesh.Trimesh(
        vertices=np.array(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            ),
            dtype=float,
        ),
        faces=np.array(((0, 1, 2),), dtype=int),
        process=False,
    )
    texture = Image.fromarray(
        np.full((2, 2, 4), (220, 90, 40, 255), dtype=np.uint8),
        mode="RGBA",
    )
    mesh.visual = TextureVisuals(
        uv=np.array(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=float),
        material=PBRMaterial(baseColorTexture=texture),
    )
    return GeneratedModel(
        mesh=mesh,
        scene=trimesh.Scene(mesh),
        glb_bytes=b"",
    )


def _build_mixed_prefab_glass_model(
    *,
    double_sided: bool,
) -> GeneratedModel:
    """Build an imported object with separate opaque and prefab primitives."""

    scene = trimesh.Scene()
    opaque_model = _build_generated_model(textured=True)
    scene.add_geometry(opaque_model.mesh, geom_name="opaque")
    glass_mesh = trimesh.Trimesh(
        vertices=np.asarray(
            ((2.0, 0.0, 0.0), (3.0, 0.0, 0.0), (2.0, 1.0, 0.0)),
            dtype=float,
        ),
        faces=np.asarray(((0, 1, 2),), dtype=np.int64),
        process=False,
    )
    glass_mesh.visual = TextureVisuals(
        uv=np.zeros((3, 2), dtype=float),
        material=build_housemaker_glass_material(double_sided),
    )
    scene.add_geometry(glass_mesh, geom_name="glass")
    return import_generated_glb(scene.export(file_type="glb"))


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Tests ###
class GlbViewerRenderingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.widgets: list[GlbViewerWidget] = []

    def tearDown(self) -> None:
        for widget in self.widgets:
            widget.close()
            widget.deleteLater()
        _qt_application.processEvents()

    def _build_viewer(self, **options: bool) -> GlbViewerWidget:
        viewer = GlbViewerWidget(**options)
        self.widgets.append(viewer)
        return viewer

    def test_textures_can_be_hidden_without_hiding_geometry(self) -> None:
        viewer = self._build_viewer()
        viewer.set_model(_build_generated_model(textured=True))

        self.assertIsNotNone(viewer.textured_mesh_item)
        self.assertIsNotNone(viewer.mesh_item)
        assert viewer.textured_mesh_item is not None
        assert viewer.mesh_item is not None
        self.assertTrue(viewer.get_textures_enabled())
        self.assertTrue(viewer.textured_mesh_item.visible())
        self.assertFalse(viewer.mesh_item.opts["drawFaces"])

        viewer.set_textures_enabled(False)

        self.assertFalse(viewer.get_textures_enabled())
        self.assertFalse(viewer.textured_mesh_item.visible())
        self.assertTrue(viewer.mesh_item.opts["drawFaces"])

    def test_preview_textures_are_capped_without_mutating_source_pixels(self) -> None:
        source = np.zeros((8, MAX_TEXTURE_PREVIEW_DIMENSION * 2, 4), dtype=np.uint8)

        preview = _limit_texture_preview_size(source)

        self.assertEqual(preview.shape[1], MAX_TEXTURE_PREVIEW_DIMENSION)
        self.assertEqual(source.shape[1], MAX_TEXTURE_PREVIEW_DIMENSION * 2)

    def test_initial_texture_proxy_keeps_every_map_aligned_at_256px(self) -> None:
        base_color = np.full((512, 1024, 4), 40, dtype=np.uint8)
        normal = np.full((512, 1024, 4), 128, dtype=np.uint8)

        proxy = build_initial_texture_proxy_bundle(
            base_color,
            normal_rgba=normal,
        )

        self.assertEqual(
            proxy.base_color_rgba.shape,
            (INITIAL_TEXTURE_PROXY_DIMENSION // 2, INITIAL_TEXTURE_PROXY_DIMENSION, 4),
        )
        self.assertIsNotNone(proxy.normal_rgba)
        assert proxy.normal_rgba is not None
        self.assertEqual(proxy.normal_rgba.shape, proxy.base_color_rgba.shape)
        self.assertIsNone(proxy.roughness_rgba)
        self.assertFalse(proxy.base_color_rgba.flags.writeable)

    def test_progressive_texture_bundle_requires_aligned_pbr_maps(self) -> None:
        with self.assertRaisesRegex(ValueError, "must match"):
            TexturePreviewBundle(
                base_color_rgba=np.zeros((4, 4, 4), dtype=np.uint8),
                normal_rgba=np.zeros((2, 4, 4), dtype=np.uint8),
            )

    def test_identical_images_share_one_decoded_preview_array(self) -> None:
        pixels = np.full((3, 4, 4), (20, 80, 140, 255), dtype=np.uint8)
        first_source = Image.fromarray(pixels, mode="RGBA")
        second_source = Image.fromarray(pixels.copy(), mode="RGBA")

        first = _decode_texture_rgba(first_source)
        second = _decode_texture_rgba(second_source)

        self.assertIsNotNone(first)
        self.assertIs(first, second)
        second_source.putpixel((0, 0), (240, 10, 30, 255))
        changed = _decode_texture_rgba(second_source)
        self.assertIsNotNone(changed)
        self.assertIsNot(changed, first)
        assert changed is not None
        self.assertEqual(tuple(changed[0, 0]), (240, 10, 30, 255))

    def test_shader_uniform_locations_are_reused_by_one_render_item(self) -> None:
        location_cache: dict[str, int] = {}
        with (
            patch.object(GL, "glGetUniformLocation", return_value=17) as lookup,
            patch.object(GL, "glUniform1f") as upload,
        ):
            _set_float_uniform(31, "u_opacity", 0.25, location_cache)
            _set_float_uniform(31, "u_opacity", 0.75, location_cache)

        lookup.assert_called_once_with(31, "u_opacity")
        self.assertEqual(upload.call_count, 2)

    def test_desktop_animation_requires_an_active_application(self) -> None:
        viewer = self._build_viewer()
        with (
            patch.object(viewer, "isVisible", return_value=True),
            patch.object(QApplication, "platformName", return_value="windows"),
            patch.object(
                QApplication,
                "applicationState",
                return_value=Qt.ApplicationState.ApplicationInactive,
            ),
        ):
            self.assertFalse(viewer._continuous_preview_animation_allowed())

        with (
            patch.object(viewer, "isVisible", return_value=True),
            patch.object(QApplication, "platformName", return_value="offscreen"),
            patch.object(
                QApplication,
                "applicationState",
                return_value=Qt.ApplicationState.ApplicationInactive,
            ),
            patch.object(QApplication, "activeWindow", return_value=None),
        ):
            self.assertTrue(viewer._continuous_preview_animation_allowed())

    def test_texture_edit_mask_is_previewed_without_mutating_base_texture(
        self,
    ) -> None:
        viewer = self._build_viewer()
        viewer.set_model(_build_generated_model(textured=True))
        assert viewer.textured_mesh_item is not None
        textured_item = viewer.textured_mesh_item
        base_texture = textured_item._texture_rgba.copy()
        edit_mask = np.zeros((2, 2), dtype=np.uint8)
        edit_mask[0, 0] = 255

        viewer.set_texture_edit_mask(edit_mask)

        self.assertTrue(textured_item._edit_mask_enabled)
        np.testing.assert_array_equal(textured_item._edit_mask, edit_mask)
        np.testing.assert_array_equal(textured_item._texture_rgba, base_texture)

        viewer.set_texture_edit_mask(None)

        self.assertFalse(textured_item._edit_mask_enabled)
        np.testing.assert_array_equal(textured_item._texture_rgba, base_texture)

    def test_nested_material_pbr_maps_are_extracted_by_gltf_channel(self) -> None:
        model = _build_generated_model(textured=True)
        normal_pixels = np.full(
            (2, 2, 4),
            (80, 120, 240, 255),
            dtype=np.uint8,
        )
        packed_pixels = np.full(
            (2, 2, 4),
            (11, 73, 191, 255),
            dtype=np.uint8,
        )
        nested_material = MultiMaterial(
            materials=[
                PBRMaterial(
                    baseColorTexture=model.mesh.visual.material.baseColorTexture,
                    normalTexture=Image.fromarray(normal_pixels, mode="RGBA"),
                    metallicRoughnessTexture=Image.fromarray(
                        packed_pixels,
                        mode="RGBA",
                    ),
                )
            ]
        )
        model.mesh.visual = TextureVisuals(
            uv=np.asarray(model.mesh.visual.uv, dtype=float),
            material=nested_material,
            face_materials=np.zeros(len(model.mesh.faces), dtype=np.int64),
        )

        texture_data = _build_texture_mesh_data(model.mesh)

        self.assertIsNotNone(texture_data)
        assert texture_data is not None
        np.testing.assert_array_equal(
            texture_data.normal_texture_rgba,
            normal_pixels,
        )
        assert texture_data.roughness_texture_rgba is not None
        assert texture_data.metallic_texture_rgba is not None
        np.testing.assert_array_equal(
            texture_data.roughness_texture_rgba[:, :, 0],
            np.full((2, 2), 73, dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            texture_data.metallic_texture_rgba[:, :, 0],
            np.full((2, 2), 191, dtype=np.uint8),
        )

    def test_missing_pbr_maps_use_neutral_preview_textures(self) -> None:
        viewer = self._build_viewer()
        viewer.set_model(_build_generated_model(textured=True))

        assert viewer.textured_mesh_item is not None
        textured_item = viewer.textured_mesh_item
        self.assertFalse(textured_item._normal_texture_available)
        self.assertFalse(textured_item._roughness_texture_available)
        self.assertFalse(textured_item._metallic_texture_available)
        self.assertIsNone(textured_item._tangents)
        self.assertIsNone(textured_item._bitangents)
        np.testing.assert_array_equal(
            textured_item._normal_texture_rgba[0, 0],
            np.asarray((128, 128, 255, 255), dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            textured_item._roughness_texture_rgba[0, 0],
            np.asarray((255, 255, 255, 255), dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            textured_item._metallic_texture_rgba[0, 0],
            np.asarray((0, 0, 0, 255), dtype=np.uint8),
        )

    def test_prefab_glass_preview_is_atlas_independent_and_respects_sides(
        self,
    ) -> None:
        for double_sided in (False, True):
            with self.subTest(double_sided=double_sided):
                model = _build_mixed_prefab_glass_model(
                    double_sided=double_sided
                )
                viewer = self._build_viewer()

                viewer.set_model(model)

                self.assertEqual(len(viewer.model_material_items), 2)
                glass_items = [
                    item
                    for item in viewer.model_material_items
                    if item._is_prefab_glass
                ]
                self.assertEqual(len(glass_items), 1)
                glass_item = glass_items[0]
                self.assertEqual(glass_item._double_sided, double_sided)
                np.testing.assert_array_equal(
                    glass_item._texture_rgba,
                    np.asarray([[[205, 232, 242, 48]]], dtype=np.uint8),
                )
                self.assertEqual(glass_item._texture_rgba.shape, (1, 1, 4))
                np.testing.assert_array_equal(
                    glass_item._roughness_texture_rgba[0, 0],
                    np.asarray((10, 10, 10, 255), dtype=np.uint8),
                )
                np.testing.assert_array_equal(
                    glass_item._metallic_texture_rgba[0, 0],
                    np.asarray((255, 255, 255, 255), dtype=np.uint8),
                )

                viewer.set_textures_enabled(False)

                self.assertFalse(glass_item.visible())
                assert viewer.mesh_item is not None
                self.assertTrue(viewer.mesh_item.opts["drawFaces"])

    def test_prefab_glass_stays_separate_in_symmetric_preview(self) -> None:
        viewer = self._build_viewer()
        viewer.set_model(
            _build_mixed_prefab_glass_model(double_sided=False)
        )

        viewer.set_symmetric_division_preview("vertical", 0.0)

        groups = viewer._explicit_symmetric_preview_groups
        self.assertEqual(len(groups), 2)
        glass_items = [
            group.textured_item
            for group in groups
            if group.textured_item is not None
            and group.textured_item._is_prefab_glass
        ]
        self.assertEqual(len(glass_items), 1)
        self.assertFalse(glass_items[0]._double_sided)

    def test_color_balance_preview_reaches_future_model_and_mirror_items(
        self,
    ) -> None:
        viewer = self._build_viewer()
        settings = TextureColorBalanceSettings(
            shadows=ColorBalanceAdjustment(cyan_red=-20),
            midtones=ColorBalanceAdjustment(
                magenta_green=35,
                yellow_blue=-15,
            ),
            highlights=ColorBalanceAdjustment(cyan_red=10),
            preserve_luminosity=False,
        )

        viewer.set_texture_color_balance_preview(settings)
        viewer.set_model(
            _build_mixed_prefab_glass_model(double_sided=False)
        )
        viewer.set_symmetric_division_preview("vertical", 0.0)

        textured_items = viewer._iter_textured_mesh_items()
        self.assertGreaterEqual(len(textured_items), 4)
        opaque_items = [
            item for item in textured_items if not item._is_prefab_glass
        ]
        glass_items = [
            item for item in textured_items if item._is_prefab_glass
        ]
        self.assertTrue(opaque_items)
        self.assertTrue(glass_items)
        self.assertTrue(
            all(
                item._color_balance_preview_settings == settings
                for item in opaque_items
            )
        )
        self.assertTrue(
            all(
                item._color_balance_preview_settings.is_neutral
                for item in glass_items
            )
        )

        viewer.set_texture_color_balance_preview(None)

        self.assertTrue(
            all(
                item._color_balance_preview_settings.is_neutral
                for item in opaque_items
            )
        )

    def test_color_balance_preview_does_not_mutate_texture_or_pbr_pixels(
        self,
    ) -> None:
        model = _build_generated_model(textured=True)
        normal_pixels = np.full(
            (2, 2, 4),
            (128, 128, 255, 255),
            dtype=np.uint8,
        )
        model.mesh.visual.material.normalTexture = Image.fromarray(
            normal_pixels,
            mode="RGBA",
        )
        viewer = self._build_viewer()
        viewer.set_model(model)
        assert viewer.textured_mesh_item is not None
        textured_item = viewer.textured_mesh_item
        original_base = textured_item._texture_rgba.copy()
        original_normal = textured_item._normal_texture_rgba.copy()

        viewer.set_texture_color_balance_preview(
            TextureColorBalanceSettings(
                midtones=ColorBalanceAdjustment(
                    cyan_red=80,
                    magenta_green=-40,
                    yellow_blue=25,
                )
            )
        )

        np.testing.assert_array_equal(textured_item._texture_rgba, original_base)
        np.testing.assert_array_equal(
            textured_item._normal_texture_rgba,
            original_normal,
        )

    def test_selected_placed_object_color_balance_preview_is_scoped_and_cleared(
        self,
    ) -> None:
        model = _build_generated_model(textured=True)
        model.preview_base_mesh = model.mesh
        model.preview_textured_surfaces = [
            PreviewTexturedSurface(
                surface_id="floor-one",
                surface_type="floor",
                mesh=model.mesh.copy(),
            )
        ]
        model.preview_placed_objects = [
            PreviewPlacedObject(
                object_id="selected-object",
                meshes=(model.mesh.copy(),),
                placement_transform=np.eye(4, dtype=float),
                world_position=(0.0, 0.0, 0.0),
                rotation_degrees=(0.0, 0.0, 0.0),
                symmetric_preview_orientation="vertical",
                symmetric_preview_plane_coordinate=0.5,
            ),
            PreviewPlacedObject(
                object_id="other-object",
                meshes=(model.mesh.copy(),),
                placement_transform=np.eye(4, dtype=float),
                world_position=(2.0, 0.0, 0.0),
                rotation_degrees=(0.0, 0.0, 0.0),
            ),
        ]
        viewer = self._build_viewer(window_editing_enabled=True)
        viewer.set_model(model)
        self.assertTrue(
            viewer.set_selected_placed_object_ids(("selected-object",))
        )
        settings = TextureColorBalanceSettings(
            shadows=ColorBalanceAdjustment(cyan_red=-35),
            highlights=ColorBalanceAdjustment(yellow_blue=45),
        )

        self.assertTrue(
            viewer.set_placed_object_texture_color_balance_preview(
                "selected-object",
                settings,
            )
        )

        selected_group = viewer._placed_object_render_groups["selected-object"]
        selected_items = [
            part.textured_item for part in selected_group.retained_parts
        ] + [
            group.textured_item for group in selected_group.symmetric_groups
        ]
        self.assertTrue(selected_group.symmetric_groups)
        self.assertTrue(
            all(
                item is not None
                and item._color_balance_preview_settings == settings
                for item in selected_items
            )
        )
        other_items = viewer._iter_placed_object_textured_mesh_items(
            "other-object"
        )
        self.assertTrue(other_items)
        self.assertTrue(
            all(
                item._color_balance_preview_settings.is_neutral
                for item in other_items
            )
        )
        self.assertTrue(viewer.textured_surface_items)
        self.assertTrue(
            all(
                item._color_balance_preview_settings.is_neutral
                for item in viewer.textured_surface_items
            )
        )

        self.assertTrue(viewer.set_selected_placed_object_ids(()))

        self.assertTrue(
            all(
                item is not None
                and item._color_balance_preview_settings.is_neutral
                for item in selected_items
            )
        )
        self.assertIsNone(
            viewer._placed_object_color_balance_preview_object_id
        )

    def test_missing_placed_object_drops_color_balance_preview_across_rebuilds(
        self,
    ) -> None:
        placed_model = _build_generated_model(textured=True)
        placed_model.preview_base_mesh = placed_model.mesh
        placed_model.preview_placed_objects = [
            PreviewPlacedObject(
                object_id="temporary-object",
                meshes=(placed_model.mesh.copy(),),
                placement_transform=np.eye(4, dtype=float),
                world_position=(0.0, 0.0, 0.0),
                rotation_degrees=(0.0, 0.0, 0.0),
            )
        ]
        empty_model = _build_generated_model(textured=True)
        empty_model.preview_base_mesh = empty_model.mesh
        viewer = self._build_viewer(window_editing_enabled=True)
        viewer.set_model(placed_model)
        self.assertTrue(
            viewer.set_selected_placed_object_ids(("temporary-object",))
        )
        settings = TextureColorBalanceSettings(
            midtones=ColorBalanceAdjustment(cyan_red=55),
        )
        self.assertTrue(
            viewer.set_placed_object_texture_color_balance_preview(
                "temporary-object",
                settings,
            )
        )

        viewer.set_model(empty_model)

        self.assertEqual(viewer.get_selected_placed_object_ids(), ())
        self.assertIsNone(
            viewer._placed_object_color_balance_preview_object_id
        )
        self.assertTrue(
            viewer._placed_object_color_balance_preview_settings.is_neutral
        )

        viewer.set_model(placed_model)
        self.assertTrue(
            viewer.set_selected_placed_object_ids(("temporary-object",))
        )

        reintroduced_items = viewer._iter_placed_object_textured_mesh_items(
            "temporary-object"
        )
        self.assertTrue(reintroduced_items)
        self.assertTrue(
            all(
                item._color_balance_preview_settings.is_neutral
                for item in reintroduced_items
            )
        )

    def test_pbr_map_state_updates_all_textured_preview_kinds(self) -> None:
        model = _build_generated_model(textured=True)
        normal_pixels = np.full(
            (2, 2, 4),
            (128, 128, 255, 255),
            dtype=np.uint8,
        )
        packed_pixels = np.full(
            (2, 2, 4),
            (0, 180, 40, 255),
            dtype=np.uint8,
        )
        model.mesh.visual.material.normalTexture = Image.fromarray(
            normal_pixels,
            mode="RGBA",
        )
        model.mesh.visual.material.metallicRoughnessTexture = Image.fromarray(
            packed_pixels,
            mode="RGBA",
        )
        surface_mesh = model.mesh.copy()
        placed_mesh = model.mesh.copy()
        model.preview_base_mesh = model.mesh
        model.preview_textured_surfaces = [
            PreviewTexturedSurface(
                surface_id="floor-one",
                surface_type="floor",
                mesh=surface_mesh,
            )
        ]
        model.preview_placed_objects = [
            PreviewPlacedObject(
                object_id="placed-one",
                meshes=(placed_mesh,),
                placement_transform=np.eye(4, dtype=float),
                world_position=(0.0, 0.0, 0.0),
                rotation_degrees=(0.0, 0.0, 0.0),
                symmetric_preview_orientation="vertical",
                symmetric_preview_plane_coordinate=0.5,
            )
        ]
        viewer = self._build_viewer(window_editing_enabled=True)
        viewer.set_pbr_maps_enabled((PBR_MAP_NORMAL,))

        viewer.set_model(model)

        initial_items = viewer._iter_textured_mesh_items()
        self.assertGreaterEqual(len(initial_items), 4)
        for item in initial_items:
            self.assertTrue(np.all(np.isfinite(item._tangents)))
            self.assertTrue(np.all(np.isfinite(item._bitangents)))
            self.assertEqual(
                item.get_pbr_maps_enabled(),
                {
                    PBR_MAP_NORMAL: True,
                    PBR_MAP_ROUGHNESS: False,
                    PBR_MAP_METALLIC: False,
                },
            )

        expected = {
            PBR_MAP_NORMAL: False,
            PBR_MAP_ROUGHNESS: True,
            PBR_MAP_METALLIC: True,
        }
        viewer.set_pbr_maps_enabled(expected)

        self.assertEqual(viewer.get_pbr_maps_enabled(), expected)
        for item in viewer._iter_textured_mesh_items():
            self.assertEqual(item.get_pbr_maps_enabled(), expected)

        viewer.set_textures_enabled(False)

        for item in viewer._iter_textured_mesh_items():
            self.assertFalse(item.visible())

    def test_orbiting_point_light_follows_any_enabled_pbr_map(self) -> None:
        model = _build_generated_model(textured=True)
        viewer = self._build_viewer(orbiting_point_light_with_pbr=True)
        viewer.set_model(model)

        self.assertFalse(viewer.is_orbiting_point_light_enabled())
        self.assertIsNone(viewer.get_orbiting_point_light_position())

        viewer.set_pbr_maps_enabled((PBR_MAP_NORMAL,))

        first_position = viewer.get_orbiting_point_light_position()
        self.assertTrue(viewer.is_orbiting_point_light_enabled())
        self.assertIsNotNone(first_position)
        self.assertTrue(np.all(np.isfinite(first_position)))
        for item in viewer._iter_textured_mesh_items():
            self.assertEqual(item.get_point_light_world_position(), first_position)
        marker = viewer._orbiting_point_light_marker_item
        self.assertIsInstance(marker, gl.GLMeshItem)
        assert marker is not None
        self.assertIn(marker, viewer.view.items)
        self.assertEqual(marker.opts["color"], (1.0, 1.0, 1.0, 1.0))
        self.assertIsNone(marker.opts["shader"])
        marker_center = marker.transform().map(QVector3D(0.0, 0.0, 0.0))
        np.testing.assert_allclose(
            (marker_center.x(), marker_center.y(), marker_center.z()),
            first_position,
        )

        viewer.set_pbr_maps_enabled((PBR_MAP_NORMAL, PBR_MAP_METALLIC))
        viewer.set_pbr_maps_enabled((PBR_MAP_METALLIC,))

        self.assertTrue(viewer.is_orbiting_point_light_enabled())
        self.assertIs(viewer.model, model)

        viewer.show()
        _qt_application.processEvents()
        viewer._orbiting_point_light_timer.stop()
        position_before_tick = viewer.get_orbiting_point_light_position()
        with (
            patch.object(TexturedMeshItem, "update") as item_update,
            patch.object(viewer.view, "update") as view_update,
        ):
            viewer._advance_orbiting_point_light()

        item_update.assert_not_called()
        view_update.assert_called_once_with()
        position_after_tick = viewer.get_orbiting_point_light_position()

        self.assertIsNotNone(position_before_tick)
        self.assertIsNotNone(position_after_tick)
        self.assertNotEqual(position_after_tick, position_before_tick)
        self.assertIs(viewer._orbiting_point_light_marker_item, marker)
        marker_center = marker.transform().map(QVector3D(0.0, 0.0, 0.0))
        np.testing.assert_allclose(
            (marker_center.x(), marker_center.y(), marker_center.z()),
            position_after_tick,
        )
        orbit_geometry = viewer._get_orbiting_point_light_geometry()
        self.assertIsNotNone(orbit_geometry)
        assert orbit_geometry is not None
        center, radius, _height = orbit_geometry
        for position in (position_before_tick, position_after_tick):
            assert position is not None
            self.assertAlmostEqual(
                float(np.linalg.norm(np.asarray(position[:2]) - center[:2])),
                radius,
            )

        viewer.set_pbr_maps_enabled(())

        self.assertFalse(viewer.is_orbiting_point_light_enabled())
        self.assertIsNone(viewer.get_orbiting_point_light_position())
        self.assertFalse(viewer._orbiting_point_light_timer.isActive())
        self.assertIsNone(viewer._orbiting_point_light_marker_item)
        self.assertNotIn(marker, viewer.view.items)
        for item in viewer._iter_textured_mesh_items():
            self.assertIsNone(item.get_point_light_world_position())

    def test_orbiting_point_light_marker_is_rebuilt_with_the_model(self) -> None:
        viewer = self._build_viewer(orbiting_point_light_with_pbr=True)
        viewer.set_pbr_maps_enabled((PBR_MAP_NORMAL,))
        self.assertIsNone(viewer._orbiting_point_light_marker_item)

        viewer.set_model(_build_generated_model(textured=True))
        first_marker = viewer._orbiting_point_light_marker_item
        self.assertIsNotNone(first_marker)
        assert first_marker is not None

        replacement_model = _build_generated_model(textured=True)
        replacement_model.mesh.apply_scale(2.0)
        viewer.set_model(replacement_model)

        replacement_marker = viewer._orbiting_point_light_marker_item
        self.assertIsNotNone(replacement_marker)
        self.assertIsNot(replacement_marker, first_marker)
        self.assertNotIn(first_marker, viewer.view.items)
        self.assertEqual(
            sum(
                item is replacement_marker
                for item in viewer.view.items
            ),
            1,
        )

        viewer.clear_model()

        self.assertIsNone(viewer._orbiting_point_light_marker_item)
        self.assertNotIn(replacement_marker, viewer.view.items)

    def test_pbr_point_light_is_opt_in_per_viewer(self) -> None:
        viewer = self._build_viewer()
        viewer.set_model(_build_generated_model(textured=True))

        viewer.set_pbr_maps_enabled((PBR_MAP_ROUGHNESS,))

        self.assertFalse(viewer.is_orbiting_point_light_enabled())
        self.assertIsNone(viewer.get_orbiting_point_light_position())

    def test_image_wall_items_are_not_treated_as_textured_mesh_items(self) -> None:
        viewer = self._build_viewer()
        wall_item = gl.GLImageItem(
            np.zeros((2, 2, 4), dtype=np.uint8)
        )
        viewer.textured_wall_items = [wall_item]

        viewer.set_pbr_maps_enabled((PBR_MAP_ROUGHNESS,))

        self.assertNotIn(wall_item, viewer._iter_textured_mesh_items())

    def test_disabled_pbr_maps_defer_tangents_and_gl_uploads(self) -> None:
        model = _build_generated_model(textured=True)
        normal_pixels = np.full(
            (2, 2, 4),
            (128, 128, 255, 255),
            dtype=np.uint8,
        )
        packed_pixels = np.full(
            (2, 2, 4),
            (0, 160, 70, 255),
            dtype=np.uint8,
        )
        model.mesh.visual.material.normalTexture = Image.fromarray(
            normal_pixels,
            mode="RGBA",
        )
        model.mesh.visual.material.metallicRoughnessTexture = Image.fromarray(
            packed_pixels,
            mode="RGBA",
        )
        texture_data = _build_texture_mesh_data(model.mesh)
        assert texture_data is not None

        item = TexturedMeshItem(texture_data, 1.0)

        self.assertIsNone(item._tangents)
        self.assertIsNone(item._bitangents)
        self.assertIsNone(item._normal_texture_id)
        self.assertIsNone(item._roughness_texture_id)
        self.assertIsNone(item._metallic_texture_id)

        item.set_pbr_maps_enabled((PBR_MAP_ROUGHNESS,))
        self.assertIsNone(item._tangents)
        item.set_pbr_maps_enabled((PBR_MAP_NORMAL, PBR_MAP_ROUGHNESS))
        self.assertIsNotNone(item._tangents)
        self.assertIsNotNone(item._bitangents)
        self.assertIsNone(item._normal_texture_id)
        self.assertIsNone(item._roughness_texture_id)

    def test_texture_bundle_swap_retains_geometry_and_shader_resources(self) -> None:
        texture_data = _build_texture_mesh_data(
            _build_generated_model(textured=True).mesh
        )
        assert texture_data is not None
        item = TexturedMeshItem(
            texture_data,
            1.0,
            pbr_maps_enabled=(PBR_MAP_NORMAL, PBR_MAP_ROUGHNESS),
        )
        context = _FakeOpenGlContext()
        item._gl_context = context
        item._resources_uploaded = True
        item._position_buffer = 11
        item._normal_buffer = 12
        item._texture_coordinate_buffer = 13
        item._tangent_buffer = 14
        item._bitangent_buffer = 15
        item._shader_program = 31
        item._texture_id = 101
        item._normal_texture_id = 102
        item._roughness_texture_id = 103
        replacement = TexturePreviewBundle(
            base_color_rgba=np.full((4, 4, 4), (20, 40, 60, 255), dtype=np.uint8),
            normal_rgba=np.full((4, 4, 4), (128, 128, 255, 255), dtype=np.uint8),
            roughness_rgba=np.full((4, 4, 4), (180, 180, 180, 255), dtype=np.uint8),
        )

        self.assertTrue(item.set_texture_bundle(replacement))
        with (
            patch.object(QOpenGLContext, "currentContext", return_value=context),
            patch.object(GL, "glDeleteTextures") as delete_textures,
            patch.object(
                item,
                "_acquire_texture",
                side_effect=(201, 202, 203),
            ) as acquire_texture,
        ):
            item._ensure_gl_resources()

        self.assertEqual(item._position_buffer, 11)
        self.assertEqual(item._normal_buffer, 12)
        self.assertEqual(item._texture_coordinate_buffer, 13)
        self.assertEqual(item._shader_program, 31)
        self.assertEqual(item._texture_id, 201)
        self.assertEqual(item._normal_texture_id, 202)
        self.assertEqual(item._roughness_texture_id, 203)
        self.assertFalse(item._texture_bundle_dirty)
        self.assertEqual(
            tuple(delete_textures.call_args.args[1]),
            (101, 102, 103),
        )
        self.assertEqual(
            [call.args[0] for call in acquire_texture.call_args_list],
            ["base_color", PBR_MAP_NORMAL, PBR_MAP_ROUGHNESS],
        )

    def test_progressive_texture_update_rejects_stale_worker_results(self) -> None:
        viewer = self._build_viewer()
        viewer.set_model(_build_generated_model(textured=True))
        assert viewer.textured_mesh_item is not None
        original_mesh_item = viewer.mesh_item
        original_vertices = viewer.textured_mesh_item._vertices
        first = viewer.begin_progressive_texture_update(
            "atlas:main",
            include_primary_model=True,
        )
        latest = viewer.begin_progressive_texture_update(
            "atlas:main",
            include_primary_model=True,
        )
        bundle = TexturePreviewBundle(
            base_color_rgba=np.full((8, 8, 4), (15, 25, 35, 255), dtype=np.uint8)
        )

        self.assertEqual(viewer.apply_progressive_texture_update(first, bundle), 0)
        self.assertEqual(viewer.apply_progressive_texture_update(latest, bundle), 1)
        self.assertIs(viewer.mesh_item, original_mesh_item)
        self.assertIs(viewer.textured_mesh_item._vertices, original_vertices)
        np.testing.assert_array_equal(
            viewer.textured_mesh_item._texture_rgba,
            bundle.base_color_rgba,
        )

        obsolete_scene = viewer.begin_progressive_texture_update(
            "atlas:secondary",
            include_primary_model=True,
        )
        viewer.set_model(_build_generated_model(textured=True), preserve_camera=True)
        self.assertEqual(
            viewer.apply_progressive_texture_update(obsolete_scene, bundle),
            0,
        )

    def test_compatible_textured_surfaces_share_one_retained_item(self) -> None:
        model = _build_generated_model(textured=True)
        model.preview_base_mesh = model.mesh
        first_mesh = model.mesh.copy()
        second_mesh = model.mesh.copy()
        second_mesh.apply_translation((2.0, 0.0, 0.0))
        model.preview_textured_surfaces = [
            PreviewTexturedSurface(
                surface_id="floor-a",
                surface_type="floor",
                mesh=first_mesh,
                level_index=0,
                room_index=0,
            ),
            PreviewTexturedSurface(
                surface_id="floor-b",
                surface_type="floor",
                mesh=second_mesh,
                level_index=0,
                room_index=0,
            ),
        ]
        viewer = self._build_viewer()

        viewer.set_model(model)

        self.assertEqual(len(viewer.textured_surface_items), 1)
        item = viewer.textured_surface_items[0]
        self.assertEqual(
            viewer._get_textured_surface_item_surface_ids(item),
            frozenset(("floor-a", "floor-b")),
        )
        self.assertEqual(len(item._vertices), 6)
        self.assertEqual(len(item._draw_ranges), 1)
        self.assertEqual(item._draw_ranges[0].vertex_count, 6)

    def test_ineligible_duplicate_surface_ids_remain_separate(self) -> None:
        model = _build_generated_model(textured=True)
        model.preview_base_mesh = model.mesh
        first_mesh = model.mesh.copy()
        second_mesh = model.mesh.copy()
        second_mesh.apply_translation((2.0, 0.0, 0.0))
        for mesh in (first_mesh, second_mesh):
            mesh.visual.material.alphaMode = "BLEND"
        model.preview_textured_surfaces = [
            PreviewTexturedSurface(
                surface_id="two-sided-glass",
                surface_type="wall",
                mesh=first_mesh,
                level_index=0,
            ),
            PreviewTexturedSurface(
                surface_id="two-sided-glass",
                surface_type="wall",
                mesh=second_mesh,
                level_index=0,
            ),
        ]
        viewer = self._build_viewer()

        viewer.set_model(model)

        self.assertEqual(len(viewer.textured_surface_items), 2)

    def test_surface_batch_culls_each_room_draw_range_independently(self) -> None:
        model = _build_generated_model(textured=True)
        model.preview_base_mesh = model.mesh
        visible_mesh = model.mesh.copy()
        offscreen_mesh = model.mesh.copy()
        offscreen_mesh.apply_translation((10.0, 0.0, 0.0))
        model.preview_textured_surfaces = [
            PreviewTexturedSurface(
                surface_id="visible-room",
                surface_type="floor",
                mesh=visible_mesh,
                level_index=0,
                room_index=0,
            ),
            PreviewTexturedSurface(
                surface_id="offscreen-room",
                surface_type="floor",
                mesh=offscreen_mesh,
                level_index=0,
                room_index=1,
            ),
        ]
        viewer = self._build_viewer()
        viewer.set_model(model)
        self.assertEqual(len(viewer.textured_surface_items), 1)
        item = viewer.textured_surface_items[0]
        self.assertEqual(len(item._draw_ranges), 2)
        identity_matrix = Mock()
        identity_matrix.data.return_value = np.eye(4, dtype=np.float32).reshape(
            -1,
            order="F",
        )

        with patch.object(item, "mvpMatrix", return_value=identity_matrix):
            visible_ranges = item._get_visible_draw_ranges()

        self.assertEqual(len(visible_ranges), 1)
        self.assertEqual(visible_ranges[0].first_vertex, 0)
        self.assertEqual(visible_ranges[0].vertex_count, 3)
        item._shader_program = 37
        with (
            patch("housemaker.viewer._set_float_uniform"),
            patch.object(GL, "glDrawArrays") as draw_arrays,
        ):
            item._draw_bound_triangles(visible_ranges)
        draw_arrays.assert_called_once_with(GL.GL_TRIANGLES, 0, 3)

    def test_compatible_placed_primitives_share_one_retained_render(self) -> None:
        model = _build_generated_model(textured=True)
        model.preview_base_mesh = model.mesh
        first_mesh = model.mesh.copy()
        second_mesh = model.mesh.copy()
        second_mesh.apply_translation((2.0, 0.0, 0.0))
        model.preview_placed_objects = [
            PreviewPlacedObject(
                object_id="batched-object",
                meshes=(first_mesh, second_mesh),
                placement_transform=np.eye(4, dtype=float),
                world_position=(0.0, 0.0, 0.0),
                rotation_degrees=(0.0, 0.0, 0.0),
            )
        ]
        viewer = self._build_viewer(window_editing_enabled=True)

        viewer.set_model(model)

        group = viewer._placed_object_render_groups["batched-object"]
        self.assertEqual(len(group.retained_parts), 1)
        retained_part = group.retained_parts[0]
        self.assertIsNotNone(retained_part.textured_item)
        assert retained_part.textured_item is not None
        self.assertEqual(len(retained_part.textured_item._vertices), 6)
        mesh_data = retained_part.mesh_item.opts["meshdata"]
        self.assertEqual(mesh_data.vertexes().shape, (6, 3))
        self.assertEqual(mesh_data.faces().shape, (2, 3))

    def test_progressive_surface_update_splits_only_target_from_batch(self) -> None:
        model = _build_generated_model(textured=True)
        model.preview_base_mesh = model.mesh
        first_mesh = model.mesh.copy()
        second_mesh = model.mesh.copy()
        second_mesh.apply_translation((2.0, 0.0, 0.0))
        model.preview_textured_surfaces = [
            PreviewTexturedSurface(
                surface_id="target-surface",
                surface_type="floor",
                mesh=first_mesh,
                level_index=0,
                room_index=0,
            ),
            PreviewTexturedSurface(
                surface_id="untouched-surface",
                surface_type="floor",
                mesh=second_mesh,
                level_index=0,
                room_index=0,
            ),
        ]
        viewer = self._build_viewer()
        viewer.set_model(model)
        self.assertEqual(len(viewer.textured_surface_items), 1)
        original_item = viewer.textured_surface_items[0]
        original_pixels = original_item._texture_rgba.copy()
        no_op_token = viewer.begin_progressive_texture_update(
            "no-op-surface-upgrade",
            surface_ids=("target-surface",),
        )

        no_op_count = viewer.apply_progressive_texture_update(
            no_op_token,
            TexturePreviewBundle(base_color_rgba=original_pixels),
        )

        self.assertEqual(no_op_count, 0)
        self.assertEqual(viewer.textured_surface_items, [original_item])
        color_balance_settings = TextureColorBalanceSettings(
            midtones=ColorBalanceAdjustment(cyan_red=35)
        )
        viewer.set_texture_color_balance_preview(color_balance_settings)
        replacement_pixels = np.full(
            original_pixels.shape,
            (15, 45, 215, 255),
            dtype=np.uint8,
        )
        token = viewer.begin_progressive_texture_update(
            "targeted-surface-upgrade",
            surface_ids=("target-surface",),
        )

        changed_count = viewer.apply_progressive_texture_update(
            token,
            TexturePreviewBundle(base_color_rgba=replacement_pixels),
        )

        self.assertEqual(changed_count, 1)
        self.assertEqual(len(viewer.textured_surface_items), 2)
        items_by_surface_id = {
            next(iter(viewer._get_textured_surface_item_surface_ids(item))): item
            for item in viewer.textured_surface_items
        }
        self.assertEqual(
            set(items_by_surface_id),
            {"target-surface", "untouched-surface"},
        )
        np.testing.assert_array_equal(
            items_by_surface_id["target-surface"]._texture_rgba,
            replacement_pixels,
        )
        np.testing.assert_array_equal(
            items_by_surface_id["untouched-surface"]._texture_rgba,
            original_pixels,
        )
        self.assertTrue(
            all(
                item._color_balance_preview_settings == color_balance_settings
                for item in viewer.textured_surface_items
            )
        )

    def test_textured_paint_restores_known_cull_state_without_readback(
        self,
    ) -> None:
        texture_data = _build_texture_mesh_data(
            _build_generated_model(textured=True).mesh
        )
        assert texture_data is not None
        item = TexturedMeshItem(texture_data, 1.0, double_sided=False)

        with (
            patch.object(
                item,
                "_get_visible_draw_ranges",
                return_value=item._draw_ranges,
            ),
            patch.object(item, "setupGLState"),
            patch.object(item, "_ensure_gl_resources"),
            patch.object(GL, "glIsEnabled") as read_cull_state,
            patch.object(GL, "glGetIntegerv") as read_integer_state,
            patch.object(GL, "glEnable") as enable,
            patch.object(GL, "glCullFace"),
            patch.object(GL, "glDisable") as disable,
        ):
            item.paint()

        read_cull_state.assert_not_called()
        read_integer_state.assert_not_called()
        enable.assert_called_once_with(GL.GL_CULL_FACE)
        disable.assert_called_once_with(GL.GL_CULL_FACE)

    def test_texture_draw_ranges_fall_back_on_opengl_es(self) -> None:
        ranges = (
            _TextureDrawRange(0, 3, np.zeros((8, 3), dtype=np.float32)),
            _TextureDrawRange(6, 3, np.zeros((8, 3), dtype=np.float32)),
        )
        context = Mock()
        context.isOpenGLES.return_value = True

        with (
            patch.object(QOpenGLContext, "currentContext", return_value=context),
            patch.object(GL, "glMultiDrawArrays") as multi_draw,
            patch.object(GL, "glDrawArrays") as draw_arrays,
        ):
            _draw_texture_ranges(ranges)

        multi_draw.assert_not_called()
        self.assertEqual(
            draw_arrays.call_args_list,
            [
                call(GL.GL_TRIANGLES, 0, 3),
                call(GL.GL_TRIANGLES, 6, 3),
            ],
        )

    def test_viewport_establishes_depth_function_baseline_each_frame(self) -> None:
        viewer = self._build_viewer()

        with (
            patch.object(GL, "glDepthFunc") as depth_function,
            patch.object(gl.GLViewWidget, "paintGL") as base_paint,
        ):
            viewer.view.paintGL()

        depth_function.assert_called_once_with(GL.GL_LESS)
        base_paint.assert_called_once_with()

    def test_textured_item_releases_every_owned_gl_resource(self) -> None:
        model = _build_generated_model(textured=True)
        texture_data = _build_texture_mesh_data(model.mesh)
        assert texture_data is not None
        item = TexturedMeshItem(texture_data, 1.0)
        item._position_buffer = 11
        item._normal_buffer = 12
        item._tangent_buffer = 13
        item._bitangent_buffer = 14
        item._texture_coordinate_buffer = 15
        item._texture_id = 21
        item._edit_mask_texture_id = 22
        item._normal_texture_id = 23
        item._roughness_texture_id = 24
        item._metallic_texture_id = 25
        item._shader_program = 31
        item._resources_uploaded = True

        with (
            patch.object(GL, "glDeleteBuffers") as delete_buffers,
            patch.object(GL, "glDeleteTextures") as delete_textures,
            patch.object(GL, "glDeleteProgram") as delete_program,
        ):
            released = item._release_gl_resources_in_current_context()

        self.assertTrue(released)
        self.assertEqual(delete_buffers.call_args.args[0], 5)
        self.assertEqual(
            tuple(delete_buffers.call_args.args[1]),
            (11, 12, 13, 14, 15),
        )
        self.assertEqual(delete_textures.call_args.args[0], 5)
        self.assertEqual(
            tuple(delete_textures.call_args.args[1]),
            (21, 22, 23, 24, 25),
        )
        delete_program.assert_called_once_with(31)
        self.assertFalse(item._has_gl_resource_handles())
        self.assertFalse(item._resources_uploaded)

    def test_identical_items_share_context_shader_until_last_release(self) -> None:
        texture_data = _build_texture_mesh_data(
            _build_generated_model(textured=True).mesh
        )
        assert texture_data is not None
        first = TexturedMeshItem(texture_data, 1.0)
        faded = TexturedMeshItem(
            texture_data,
            1.0,
            opacity=0.5,
            texture_repeat=True,
            pbr_maps_enabled=(PBR_MAP_NORMAL,),
        )
        context = _FakeOpenGlContext()

        with (
            patch.object(QOpenGLContext, "currentContext", return_value=context),
            patch(
                "housemaker.viewer.opengl_shaders.compileShader",
                side_effect=(7, 8),
            ) as compile_shader,
            patch(
                "housemaker.viewer.opengl_shaders.compileProgram",
                return_value=31,
            ) as compile_program,
            patch("housemaker.viewer._upload_array_buffer", return_value=11),
            patch("housemaker.viewer._upload_mask_texture", return_value=201),
            patch("housemaker.viewer._upload_texture", return_value=101),
            patch.object(_SHARED_GL_TEXTURE_CACHE, "acquire", return_value=None),
        ):
            first._ensure_gl_resources()
            faded._ensure_gl_resources()

        self.assertEqual(first._shader_program, 31)
        self.assertEqual(faded._shader_program, 31)
        self.assertEqual(compile_shader.call_count, 2)
        compile_program.assert_called_once_with(7, 8)
        self.assertIsNot(first._uniform_locations, faded._uniform_locations)
        self.assertIsNot(first._attribute_locations, faded._attribute_locations)
        first._uniform_locations["u_opacity"] = 17
        first._attribute_locations["a_position"] = 4
        self.assertNotIn("u_opacity", faded._uniform_locations)
        self.assertNotIn("a_position", faded._attribute_locations)
        self.assertFalse(first.get_pbr_maps_enabled()[PBR_MAP_NORMAL])
        self.assertTrue(faded.get_pbr_maps_enabled()[PBR_MAP_NORMAL])

        with (
            patch.object(QOpenGLContext, "currentContext", return_value=context),
            patch.object(GL, "glDeleteBuffers"),
            patch.object(GL, "glDeleteTextures"),
            patch.object(GL, "glDeleteProgram") as delete_program,
        ):
            first._release_gl_resources_in_current_context()
            delete_program.assert_not_called()
            faded._release_gl_resources_in_current_context()

        delete_program.assert_called_once_with(31)

    def test_shader_sharing_uses_exact_context_identity(self) -> None:
        texture_data = _build_texture_mesh_data(
            _build_generated_model(textured=True).mesh
        )
        assert texture_data is not None
        first = TexturedMeshItem(texture_data, 1.0)
        second = TexturedMeshItem(texture_data, 1.0)
        first_context = _EqualFakeOpenGlContext()
        second_context = _EqualFakeOpenGlContext()

        with (
            patch(
                "housemaker.viewer.opengl_shaders.compileShader",
                side_effect=(7, 8, 9, 10),
            ) as compile_shader,
            patch(
                "housemaker.viewer.opengl_shaders.compileProgram",
                side_effect=(31, 32),
            ) as compile_program,
            patch("housemaker.viewer._upload_array_buffer", return_value=11),
            patch("housemaker.viewer._upload_mask_texture", return_value=201),
            patch("housemaker.viewer._upload_texture", return_value=101),
            patch.object(_SHARED_GL_TEXTURE_CACHE, "acquire", return_value=None),
        ):
            with patch.object(
                QOpenGLContext,
                "currentContext",
                return_value=first_context,
            ):
                first._ensure_gl_resources()
            with patch.object(
                QOpenGLContext,
                "currentContext",
                return_value=second_context,
            ):
                second._ensure_gl_resources()

        self.assertEqual(first._shader_program, 31)
        self.assertEqual(second._shader_program, 32)
        self.assertEqual(compile_shader.call_count, 4)
        self.assertEqual(compile_program.call_count, 2)

        with (
            patch.object(GL, "glDeleteBuffers"),
            patch.object(GL, "glDeleteTextures"),
            patch.object(GL, "glDeleteProgram") as delete_program,
        ):
            with patch.object(
                QOpenGLContext,
                "currentContext",
                return_value=first_context,
            ):
                first._release_gl_resources_in_current_context()
            with patch.object(
                QOpenGLContext,
                "currentContext",
                return_value=second_context,
            ):
                second._release_gl_resources_in_current_context()

        self.assertEqual(
            delete_program.call_args_list,
            [call(31), call(32)],
        )

    def test_context_replacement_discards_old_shader_lease(self) -> None:
        texture_data = _build_texture_mesh_data(
            _build_generated_model(textured=True).mesh
        )
        assert texture_data is not None
        item = TexturedMeshItem(texture_data, 1.0)
        first_context = _FakeOpenGlContext()
        replacement_context = _FakeOpenGlContext()

        with (
            patch(
                "housemaker.viewer.opengl_shaders.compileShader",
                side_effect=(7, 8, 9, 10),
            ),
            patch(
                "housemaker.viewer.opengl_shaders.compileProgram",
                side_effect=(31, 32),
            ) as compile_program,
            patch("housemaker.viewer._upload_array_buffer", return_value=11),
            patch("housemaker.viewer._upload_mask_texture", return_value=201),
            patch("housemaker.viewer._upload_texture", return_value=101),
            patch.object(_SHARED_GL_TEXTURE_CACHE, "acquire", return_value=None),
            patch.object(GL, "glDeleteProgram") as delete_program,
        ):
            with patch.object(
                QOpenGLContext,
                "currentContext",
                return_value=first_context,
            ):
                item._ensure_gl_resources()
            item._uniform_locations["u_opacity"] = 17
            item._attribute_locations["a_position"] = 4
            with patch.object(
                QOpenGLContext,
                "currentContext",
                return_value=replacement_context,
            ):
                item._ensure_gl_resources()

            self.assertEqual(item._shader_program, 32)
            self.assertEqual(compile_program.call_count, 2)
            self.assertEqual(item._uniform_locations, {})
            self.assertEqual(item._attribute_locations, {})
            delete_program.assert_not_called()

            with (
                patch.object(GL, "glDeleteBuffers"),
                patch.object(GL, "glDeleteTextures"),
                patch.object(
                    QOpenGLContext,
                    "currentContext",
                    return_value=replacement_context,
                ),
            ):
                item._release_gl_resources_in_current_context()

        delete_program.assert_called_once_with(32)

    def test_identical_items_share_context_texture_until_last_release(self) -> None:
        model = _build_generated_model(textured=True)
        mixed_pixels = np.full((2, 2, 4), (90, 120, 180, 255), dtype=np.uint8)
        mixed_pixels[0, 0, 3] = 64
        model.mesh.visual.material.baseColorTexture = Image.fromarray(
            mixed_pixels,
            mode="RGBA",
        )
        texture_data = _build_texture_mesh_data(model.mesh)
        assert texture_data is not None
        first = TexturedMeshItem(texture_data, 1.0)
        faded = TexturedMeshItem(texture_data, 1.0, opacity=0.5)
        context = _FakeOpenGlContext()

        with (
            patch.object(QOpenGLContext, "currentContext", return_value=context),
            patch("housemaker.viewer.opengl_shaders.compileShader", return_value=7),
            patch("housemaker.viewer.opengl_shaders.compileProgram", return_value=31),
            patch("housemaker.viewer._upload_array_buffer", return_value=11),
            patch(
                "housemaker.viewer._upload_mask_texture",
                side_effect=(201, 202),
            ),
            patch("housemaker.viewer._upload_texture", return_value=101) as upload,
            patch("housemaker.viewer.hashlib.sha256") as hash_pixels,
        ):
            first._ensure_gl_resources()
            faded._ensure_gl_resources()

        self.assertEqual(first._texture_id, 101)
        self.assertEqual(faded._texture_id, 101)
        upload.assert_called_once()
        hash_pixels.assert_not_called()
        self.assertFalse(texture_data.uses_texture_alpha)
        self.assertEqual(first._transparency_mode, "opaque")
        self.assertEqual(faded._transparency_mode, "translucent")

        with (
            patch.object(QOpenGLContext, "currentContext", return_value=context),
            patch.object(GL, "glDeleteBuffers"),
            patch.object(GL, "glDeleteProgram"),
            patch.object(GL, "glDeleteTextures") as delete_textures,
        ):
            first._release_gl_resources_in_current_context()
            first_deleted = tuple(delete_textures.call_args.args[1])
            faded._release_gl_resources_in_current_context()
            second_deleted = tuple(delete_textures.call_args.args[1])

        self.assertEqual(first_deleted, (201,))
        self.assertEqual(second_deleted, (101, 202))

    def test_pbr_toggle_acquires_existing_shared_map_independently(self) -> None:
        model = _build_generated_model(textured=True)
        normal_pixels = np.full(
            (2, 2, 4),
            (128, 150, 245, 255),
            dtype=np.uint8,
        )
        model.mesh.visual.material.normalTexture = Image.fromarray(
            normal_pixels,
            mode="RGBA",
        )
        texture_data = _build_texture_mesh_data(model.mesh)
        assert texture_data is not None
        enabled = TexturedMeshItem(
            texture_data,
            1.0,
            pbr_maps_enabled=(PBR_MAP_NORMAL,),
        )
        disabled = TexturedMeshItem(texture_data, 1.0)
        context = _FakeOpenGlContext()

        with (
            patch.object(QOpenGLContext, "currentContext", return_value=context),
            patch("housemaker.viewer.opengl_shaders.compileShader", return_value=7),
            patch("housemaker.viewer.opengl_shaders.compileProgram", return_value=31),
            patch("housemaker.viewer._upload_array_buffer", return_value=11),
            patch("housemaker.viewer._upload_mask_texture", return_value=201),
            patch(
                "housemaker.viewer._upload_texture",
                side_effect=(101, 102),
            ) as upload,
        ):
            enabled._ensure_gl_resources()
            disabled._ensure_gl_resources()
            self.assertIsNone(disabled._normal_texture_id)
            disabled.set_pbr_maps_enabled((PBR_MAP_NORMAL,))
            disabled._ensure_gl_resources()

        self.assertEqual(enabled._normal_texture_id, 102)
        self.assertEqual(disabled._normal_texture_id, 102)
        self.assertEqual(upload.call_count, 2)
        self.assertTrue(enabled.get_pbr_maps_enabled()[PBR_MAP_NORMAL])
        self.assertTrue(disabled.get_pbr_maps_enabled()[PBR_MAP_NORMAL])
        enabled.set_pbr_maps_enabled(())
        self.assertFalse(enabled.get_pbr_maps_enabled()[PBR_MAP_NORMAL])
        self.assertTrue(disabled.get_pbr_maps_enabled()[PBR_MAP_NORMAL])
        enabled._reset_gl_resource_handles()
        disabled._reset_gl_resource_handles()

    def test_texture_sharing_never_crosses_contexts_or_repeat_modes(self) -> None:
        texture_data = _build_texture_mesh_data(
            _build_generated_model(textured=True).mesh
        )
        assert texture_data is not None
        first = TexturedMeshItem(texture_data, 1.0)
        other_context = TexturedMeshItem(texture_data, 1.0)
        repeating = TexturedMeshItem(texture_data, 1.0, texture_repeat=True)
        first_context = _FakeOpenGlContext()
        second_context = _FakeOpenGlContext()

        with (
            patch("housemaker.viewer.opengl_shaders.compileShader", return_value=7),
            patch("housemaker.viewer.opengl_shaders.compileProgram", return_value=31),
            patch("housemaker.viewer._upload_array_buffer", return_value=11),
            patch("housemaker.viewer._upload_mask_texture", return_value=201),
            patch(
                "housemaker.viewer._upload_texture",
                side_effect=(101, 102, 103),
            ) as upload,
        ):
            with patch.object(
                QOpenGLContext,
                "currentContext",
                return_value=first_context,
            ):
                first._ensure_gl_resources()
                repeating._ensure_gl_resources()
            with patch.object(
                QOpenGLContext,
                "currentContext",
                return_value=second_context,
            ):
                other_context._ensure_gl_resources()

        self.assertEqual(first._texture_id, 101)
        self.assertEqual(repeating._texture_id, 102)
        self.assertEqual(other_context._texture_id, 103)
        self.assertEqual(upload.call_count, 3)
        first._reset_gl_resource_handles()
        repeating._reset_gl_resource_handles()
        other_context._reset_gl_resource_handles()

    def test_model_rebuild_releases_textured_resources_before_clear(self) -> None:
        viewer = self._build_viewer()
        viewer.set_model(_build_generated_model(textured=True))

        with patch.object(
            viewer,
            "_release_textured_mesh_gl_resources",
        ) as release_resources:
            viewer.set_model(_build_generated_model(textured=True))

        release_resources.assert_called_once_with()

    def test_mixed_alpha_uses_scoped_opaque_and_translucent_passes(self) -> None:
        model = _build_generated_model(textured=True)
        mixed_texture = np.full((2, 2, 4), 255, dtype=np.uint8)
        mixed_texture[0, 0, 3] = 48
        model.mesh.visual.material.baseColorTexture = Image.fromarray(
            mixed_texture,
            mode="RGBA",
        )
        model.mesh.visual.material.alphaMode = "BLEND"
        texture_data = _build_texture_mesh_data(model.mesh)
        assert texture_data is not None
        item = TexturedMeshItem(texture_data, 1.0)
        item._shader_program = 37

        options = item._GLGraphicsItem__glOpts
        self.assertFalse(options[GL.GL_BLEND])
        self.assertEqual(item._transparency_mode, "mixed")
        with (
            patch("housemaker.viewer._set_float_uniform") as set_uniform,
            patch.object(GL, "glDrawArrays") as draw_arrays,
            patch.object(GL, "glDepthMask") as depth_mask,
            patch.object(GL, "glBlendFuncSeparate") as blend_function,
            patch.object(GL, "glEnable") as enable,
            patch.object(GL, "glDisable") as disable,
        ):
            item._draw_bound_triangles()

        self.assertEqual(draw_arrays.call_count, 2)
        self.assertEqual(
            [call.args[2] for call in set_uniform.call_args_list],
            [1.0, 2.0],
        )
        self.assertEqual(
            [call.args[0] for call in depth_mask.call_args_list],
            [GL.GL_TRUE, GL.GL_FALSE, GL.GL_TRUE],
        )
        blend_function.assert_called_once_with(
            GL.GL_SRC_ALPHA,
            GL.GL_ONE_MINUS_SRC_ALPHA,
            GL.GL_ONE,
            GL.GL_ONE_MINUS_SRC_ALPHA,
        )
        enable.assert_called_once_with(GL.GL_BLEND)
        self.assertEqual(
            disable.call_args_list,
            [call(GL.GL_BLEND), call(GL.GL_BLEND)],
        )

    def test_transparent_atlas_padding_is_single_pass_for_opaque_material(
        self,
    ) -> None:
        model = _build_generated_model(textured=True)
        atlas_texture = np.full((2, 2, 4), 255, dtype=np.uint8)
        atlas_texture[0, 0, 3] = 0
        model.mesh.visual.material.baseColorTexture = Image.fromarray(
            atlas_texture,
            mode="RGBA",
        )
        model.mesh.visual.material.alphaMode = "OPAQUE"
        texture_data = _build_texture_mesh_data(model.mesh)
        assert texture_data is not None
        item = TexturedMeshItem(texture_data, 1.0)
        item._shader_program = 37

        with (
            patch("housemaker.viewer._set_float_uniform") as set_uniform,
            patch.object(GL, "glDrawArrays") as draw_arrays,
            patch.object(GL, "glDepthMask") as depth_mask,
            patch.object(GL, "glEnable") as enable,
        ):
            item._draw_bound_triangles()

        self.assertFalse(texture_data.uses_texture_alpha)
        self.assertEqual(item._transparency_mode, "opaque")
        draw_arrays.assert_called_once_with(
            GL.GL_TRIANGLES,
            0,
            len(item._vertices),
        )
        self.assertEqual(set_uniform.call_args.args[2], 0.0)
        depth_mask.assert_not_called()
        enable.assert_not_called()

    def test_blend_and_mask_materials_use_their_gltf_rendering_modes(self) -> None:
        expected = {
            "BLEND": ("mixed", 2),
            "MASK": ("opaque", 1),
        }
        for alpha_mode, (transparency_mode, draw_count) in expected.items():
            with self.subTest(alpha_mode=alpha_mode):
                model = _build_generated_model(textured=True)
                atlas_texture = np.full((2, 2, 4), 255, dtype=np.uint8)
                atlas_texture[0, 0, 3] = 0
                model.mesh.visual.material.baseColorTexture = Image.fromarray(
                    atlas_texture,
                    mode="RGBA",
                )
                model.mesh.visual.material.alphaMode = alpha_mode
                texture_data = _build_texture_mesh_data(model.mesh)
                assert texture_data is not None
                item = TexturedMeshItem(texture_data, 1.0)
                item._shader_program = 37

                with (
                    patch("housemaker.viewer._set_float_uniform"),
                    patch.object(GL, "glDrawArrays") as draw_arrays,
                    patch.object(GL, "glDepthMask"),
                    patch.object(GL, "glBlendFuncSeparate"),
                    patch.object(GL, "glEnable"),
                    patch.object(GL, "glDisable"),
                ):
                    item._draw_bound_triangles()

                self.assertTrue(texture_data.uses_texture_alpha)
                self.assertEqual(item._transparency_mode, transparency_mode)
                self.assertEqual(draw_arrays.call_count, draw_count)

    def test_material_alpha_cutoff_and_factor_are_preserved(self) -> None:
        model = _build_generated_model(textured=True)
        model.mesh.visual.material.alphaMode = "MASK"
        model.mesh.visual.material.alphaCutoff = 0.73
        model.mesh.visual.material.baseColorFactor = (1.0, 1.0, 1.0, 0.6)

        texture_data = _build_texture_mesh_data(model.mesh)

        assert texture_data is not None
        self.assertEqual(texture_data.alpha_mode, "MASK")
        self.assertAlmostEqual(texture_data.alpha_cutoff, 0.73)
        self.assertAlmostEqual(texture_data.base_color_alpha, 0.6)
        item = TexturedMeshItem(texture_data, 1.0)
        self.assertEqual(item._transparency_mode, "opaque")

    def test_blend_factor_alpha_uses_one_translucent_pass(self) -> None:
        model = _build_generated_model(textured=True)
        model.mesh.visual.material.alphaMode = "BLEND"
        model.mesh.visual.material.baseColorFactor = (1.0, 1.0, 1.0, 0.5)

        texture_data = _build_texture_mesh_data(model.mesh)

        assert texture_data is not None
        item = TexturedMeshItem(texture_data, 1.0)
        self.assertEqual(item._transparency_mode, "translucent")
        self.assertAlmostEqual(item._base_color_alpha, 0.5, places=2)

    def test_textured_wall_uses_translucent_gl_options_for_alpha_mask(
        self,
    ) -> None:
        model = _build_generated_model()
        opaque_texture = np.full((2, 2, 4), 255, dtype=np.uint8)
        masked_texture = opaque_texture.copy()
        masked_texture[0, 0, 3] = 0
        model.preview_textured_walls = [
            PreviewTexturedWall(
                level_index=2,
                room_index=0,
                wall_key="opaque",
                start_point=(0.0, 0.0, 0.0),
                end_point=(1.0, 0.0, 0.0),
                height_meters=2.0,
                texture_rgba=opaque_texture,
            ),
            PreviewTexturedWall(
                level_index=2,
                room_index=0,
                wall_key="masked",
                start_point=(0.0, 1.0, 0.0),
                end_point=(1.0, 1.0, 0.0),
                height_meters=2.0,
                texture_rgba=masked_texture,
            ),
        ]
        viewer = self._build_viewer()

        viewer.set_model(model)

        self.assertEqual(len(viewer.textured_wall_items), 2)
        opaque_options = (
            viewer.textured_wall_items[0]._GLGraphicsItem__glOpts
        )
        masked_options = (
            viewer.textured_wall_items[1]._GLGraphicsItem__glOpts
        )
        self.assertFalse(opaque_options[GL.GL_BLEND])
        self.assertTrue(masked_options[GL.GL_BLEND])

    def test_wireframe_only_forces_edges_and_hides_all_surfaces(self) -> None:
        viewer = self._build_viewer()
        viewer.set_model(_build_generated_model(textured=True))

        self.assertIsNotNone(viewer.textured_mesh_item)
        self.assertIsNotNone(viewer.mesh_item)
        assert viewer.textured_mesh_item is not None
        assert viewer.mesh_item is not None
        viewer.set_wireframe_enabled(False)
        viewer.set_wireframe_only(True)

        self.assertTrue(viewer.get_wireframe_only())
        self.assertFalse(viewer.textured_mesh_item.visible())
        self.assertFalse(viewer.mesh_item.opts["drawFaces"])
        self.assertTrue(viewer.mesh_item.opts["drawEdges"])

        viewer.set_wireframe_only(False)

        self.assertFalse(viewer.get_wireframe_only())
        self.assertTrue(viewer.textured_mesh_item.visible())
        self.assertFalse(viewer.mesh_item.opts["drawFaces"])
        self.assertFalse(viewer.mesh_item.opts["drawEdges"])

    def test_wireframe_can_be_disabled_for_a_non_textured_model(self) -> None:
        viewer = self._build_viewer(wireframe_enabled=False)
        viewer.set_model(_build_generated_model())

        self.assertIsNotNone(viewer.mesh_item)
        assert viewer.mesh_item is not None
        self.assertTrue(viewer.mesh_item.opts["drawFaces"])
        self.assertFalse(viewer.get_wireframe_enabled())
        self.assertTrue(viewer.mesh_item.opts["drawFaces"])
        self.assertFalse(viewer.mesh_item.opts["drawEdges"])

        viewer.set_wireframe_enabled(True)

        self.assertTrue(viewer.get_wireframe_enabled())
        self.assertTrue(viewer.mesh_item.opts["drawEdges"])

    def test_hidden_wireframe_is_skipped_by_view_traversal_until_needed(
        self,
    ) -> None:
        viewer = self._build_viewer(wireframe_enabled=False)
        viewer.set_model(_build_generated_model(textured=True))
        assert viewer.mesh_item is not None

        self.assertFalse(viewer.mesh_item.opts["drawFaces"])
        self.assertFalse(viewer.mesh_item.opts["drawEdges"])
        self.assertFalse(viewer.mesh_item.visible())

        viewer.set_wireframe_enabled(True)

        self.assertTrue(viewer.mesh_item.visible())
        viewer.set_wireframe_enabled(False)
        self.assertFalse(viewer.mesh_item.visible())

        viewer.set_textures_enabled(False)

        self.assertTrue(viewer.mesh_item.opts["drawFaces"])
        self.assertTrue(viewer.mesh_item.visible())

    def test_reference_grid_stays_below_loaded_model_geometry(self) -> None:
        viewer = self._build_viewer(wireframe_enabled=False)
        model = _build_generated_model(textured=True)
        model.mesh.apply_translation((0.0, 0.0, 4.0))

        viewer.set_model(model)

        self.assertIsNotNone(viewer.grid_item)
        assert viewer.grid_item is not None
        grid_height = float(viewer.grid_item.transform().matrix()[2, 3])
        model_minimum_z = float(np.min(model.mesh.vertices[:, 2]))
        self.assertLess(grid_height, model_minimum_z)
        self.assertAlmostEqual(grid_height, model_minimum_z - 0.01)

    def test_hidden_initial_wireframe_prepares_edges_for_later_toggle(
        self,
    ) -> None:
        viewer = self._build_viewer(wireframe_enabled=False)
        viewer.set_model(_build_generated_model(textured=True))

        assert viewer.mesh_item is not None
        self.assertFalse(viewer.mesh_item.opts["drawEdges"])

        viewer.mesh_item.parseMeshData()

        self.assertFalse(viewer.mesh_item.opts["drawEdges"])
        self.assertIsNotNone(viewer.mesh_item.edges)
        self.assertIsNotNone(viewer.mesh_item.edgeVerts)

        viewer.set_wireframe_enabled(True)

        self.assertTrue(viewer.mesh_item.opts["drawEdges"])
        self.assertGreater(viewer.mesh_item.edges.size, 0)

    def test_wireframe_edges_accept_coplanar_depth_then_restore_gl_state(
        self,
    ) -> None:
        viewer = self._build_viewer(wireframe_enabled=False)
        viewer.set_model(_build_generated_model(textured=True))
        viewer.set_wireframe_enabled(True)

        assert viewer.mesh_item is not None
        with (
            patch.object(
                GL,
                "glGetIntegerv",
                return_value=GL.GL_LESS,
            ) as read_depth_function,
            patch.object(GL, "glDepthFunc") as depth_function,
            patch.object(gl.GLMeshItem, "paint") as base_paint,
        ):
            viewer.mesh_item.paint()

        base_paint.assert_called_once_with()
        read_depth_function.assert_not_called()
        gl_options = viewer.mesh_item._GLGraphicsItem__glOpts
        self.assertFalse(gl_options[GL.GL_CULL_FACE])
        self.assertEqual(
            [call.args[0] for call in depth_function.call_args_list],
            [GL.GL_LEQUAL, GL.GL_LESS],
        )

    def test_doorway_outline_updates_in_place_and_survives_model_refresh(
        self,
    ) -> None:
        viewer = self._build_viewer()
        viewer.set_model(_build_generated_model())
        viewer.view.opts["center"] = QVector3D(1.0, 2.0, 3.0)
        viewer.view.setCameraPosition(
            distance=9.0,
            elevation=17.0,
            azimuth=31.0,
        )
        camera_center = viewer.view.opts["center"]
        camera_before = (
            (camera_center.x(), camera_center.y(), camera_center.z()),
            float(viewer.view.opts["distance"]),
            float(viewer.view.opts["elevation"]),
            float(viewer.view.opts["azimuth"]),
        )
        positions = np.arange(72, dtype=float).reshape(24, 3) / 10.0

        viewer.set_doorway_preview_outline(positions)

        first_item = viewer._doorway_preview_outline_item
        self.assertIsNotNone(first_item)
        assert first_item is not None
        self.assertIn(first_item, viewer.view.items)
        np.testing.assert_allclose(first_item.pos, positions)
        positions[:] = -1.0
        self.assertFalse(
            np.all(viewer._doorway_preview_outline_positions == -1.0)
        )

        updated_positions = np.arange(90, dtype=float).reshape(30, 3) / 5.0
        viewer.set_doorway_preview_outline(updated_positions)

        self.assertIs(viewer._doorway_preview_outline_item, first_item)
        np.testing.assert_allclose(first_item.pos, updated_positions)
        camera_center = viewer.view.opts["center"]
        self.assertEqual(
            (
                (camera_center.x(), camera_center.y(), camera_center.z()),
                float(viewer.view.opts["distance"]),
                float(viewer.view.opts["elevation"]),
                float(viewer.view.opts["azimuth"]),
            ),
            camera_before,
        )

        viewer.set_model(_build_generated_model(), preserve_camera=True)

        rebuilt_item = viewer._doorway_preview_outline_item
        self.assertIsNotNone(rebuilt_item)
        self.assertIsNot(rebuilt_item, first_item)
        self.assertIn(rebuilt_item, viewer.view.items)
        np.testing.assert_allclose(rebuilt_item.pos, updated_positions)
        camera_center = viewer.view.opts["center"]
        self.assertEqual(
            (
                (camera_center.x(), camera_center.y(), camera_center.z()),
                float(viewer.view.opts["distance"]),
                float(viewer.view.opts["elevation"]),
                float(viewer.view.opts["azimuth"]),
            ),
            camera_before,
        )

        viewer.set_doorway_preview_outline(None)

        self.assertIsNone(viewer._doorway_preview_outline_positions)
        self.assertIsNone(viewer._doorway_preview_outline_item)
        self.assertNotIn(rebuilt_item, viewer.view.items)

    def test_doorway_outline_rejects_wrong_shape_and_non_finite_positions(
        self,
    ) -> None:
        viewer = self._build_viewer()

        with self.assertRaisesRegex(ValueError, "shaped"):
            viewer.set_doorway_preview_outline(np.zeros((2, 2)))
        with self.assertRaisesRegex(ValueError, "at least two"):
            viewer.set_doorway_preview_outline(np.zeros((1, 3)))
        with self.assertRaisesRegex(ValueError, "even number"):
            viewer.set_doorway_preview_outline(np.zeros((23, 3)))

        positions = np.zeros((24, 3))
        positions[5, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            viewer.set_doorway_preview_outline(positions)

        viewer.set_doorway_preview_outline(np.zeros((2, 3)))
        self.assertEqual(
            viewer._doorway_preview_outline_positions.shape,
            (2, 3),
        )

    def test_viewer_forwards_first_person_navigation_apis(self) -> None:
        viewer = self._build_viewer()
        pose = CameraPose(x=1.0, y=2.0, z=1.7, yaw_degrees=45.0)

        self.assertTrue(viewer.first_person_crosshair_label.isHidden())
        self.assertEqual(
            viewer.get_first_person_movement_mode(),
            DEFAULT_FIRST_PERSON_NAVIGATION_MODE,
        )

        viewer.set_first_person_camera_pose(pose)
        viewer.set_first_person_movement_mode(
            FIRST_PERSON_NAVIGATION_MODE_NOCLIP
        )
        viewer.set_navigation_mode(NAVIGATION_MODE_FIRST_PERSON)

        self.assertEqual(viewer.get_navigation_mode(), NAVIGATION_MODE_FIRST_PERSON)
        self.assertEqual(viewer.get_first_person_camera_pose(), pose)
        self.assertEqual(
            viewer.get_first_person_movement_mode(),
            FIRST_PERSON_NAVIGATION_MODE_NOCLIP,
        )
        self.assertTrue(viewer.is_first_person_pointer_captured)
        self.assertFalse(viewer.first_person_crosshair_label.isHidden())
        viewer.release_first_person_pointer_capture()
        self.assertFalse(viewer.is_first_person_pointer_captured)
        self.assertEqual(viewer.get_navigation_mode(), NAVIGATION_MODE_FIRST_PERSON)
        self.assertEqual(viewer.toggle_navigation_mode(), NAVIGATION_MODE_ORBIT)
        self.assertEqual(viewer.get_navigation_mode(), NAVIGATION_MODE_ORBIT)
        self.assertTrue(viewer.first_person_crosshair_label.isHidden())

    def test_first_person_frame_overlay_can_be_set_and_cleared(self) -> None:
        viewer = self._build_viewer()
        frame_bgr = np.zeros((3, 5, 3), dtype=np.uint8)
        frame_bgr[:, :] = (17, 83, 211)
        viewer.enter_first_person_mode()

        viewer.set_first_person_frame_overlay(frame_bgr)

        self.assertTrue(viewer.is_first_person_frame_overlay_visible)

        viewer.clear_first_person_frame_overlay()

        self.assertFalse(viewer.is_first_person_frame_overlay_visible)

    def test_leaving_first_person_clears_the_frame_overlay(self) -> None:
        viewer = self._build_viewer()
        viewer.enter_first_person_mode()
        viewer.set_first_person_frame_overlay(
            np.full((3, 5, 3), 127, dtype=np.uint8)
        )
        self.assertTrue(viewer.is_first_person_frame_overlay_visible)

        viewer.exit_first_person_mode()

        self.assertFalse(viewer.is_first_person_frame_overlay_visible)

    def test_first_person_pose_survives_a_model_scene_refresh(self) -> None:
        viewer = self._build_viewer()
        pose = CameraPose(x=1.0, y=2.0, z=1.7, yaw_degrees=45.0)
        viewer.set_first_person_camera_pose(pose)
        viewer.enter_first_person_mode()

        viewer.set_model(_build_generated_model())

        self.assertEqual(viewer.get_navigation_mode(), NAVIGATION_MODE_FIRST_PERSON)
        self.assertEqual(viewer.get_first_person_camera_pose(), pose)
        self.assertAlmostEqual(float(viewer.view.opts["distance"]), 1.0)

    def test_preserved_model_refresh_retains_default_first_person_pose(
        self,
    ) -> None:
        viewer = self._build_viewer()
        viewer.set_model(_build_generated_model())
        pose = CameraPose(x=4.0, y=-3.0, z=2.1, yaw_degrees=67.0)
        viewer.view.set_default_first_person_camera_pose(pose)
        viewer.enter_first_person_mode()

        viewer.set_model(
            _build_generated_model(),
            preserve_camera=True,
        )

        self.assertEqual(
            viewer.get_navigation_mode(),
            NAVIGATION_MODE_FIRST_PERSON,
        )
        self.assertEqual(viewer.get_first_person_camera_pose(), pose)
        self.assertFalse(viewer.view.has_custom_first_person_camera_pose)

    def test_projection_camera_indicators_are_opt_in(self) -> None:
        viewer = self._build_viewer()

        viewer.set_model(_build_generated_model())

        self.assertFalse(viewer.get_projection_camera_indicators_visible())
        self.assertEqual(viewer.projection_camera_indicator_items, {})
        self.assertFalse(
            any(
                isinstance(item, (gl.GLLinePlotItem, gl.GLTextItem))
                for item in viewer.view.items
            )
        )

    def test_projection_camera_indicators_follow_model_lifecycle(self) -> None:
        viewer = self._build_viewer()

        viewer.set_projection_camera_indicators_visible(True)

        self.assertTrue(viewer.get_projection_camera_indicators_visible())
        self.assertEqual(viewer.projection_camera_indicator_items, {})

        viewer.set_model(_build_generated_model())

        first_indicators = viewer.projection_camera_indicator_items
        self.assertEqual(tuple(first_indicators), ALL_CAMERA_IDS)
        first_items = tuple(
            item
            for camera_items in first_indicators.values()
            for item in camera_items
        )
        self.assertTrue(
            all(item.visible() and item in viewer.view.items for item in first_items)
        )

        viewer.clear_model()

        self.assertTrue(viewer.get_projection_camera_indicators_visible())
        self.assertEqual(viewer.projection_camera_indicator_items, {})
        self.assertTrue(all(item not in viewer.view.items for item in first_items))

        viewer.set_model(_build_generated_model())

        rebuilt_indicators = viewer.projection_camera_indicator_items
        self.assertEqual(tuple(rebuilt_indicators), ALL_CAMERA_IDS)
        self.assertTrue(
            all(
                item not in first_items
                for camera_items in rebuilt_indicators.values()
                for item in camera_items
            )
        )

        viewer.set_projection_camera_indicators_visible(False)

        self.assertFalse(viewer.get_projection_camera_indicators_visible())
        self.assertEqual(viewer.projection_camera_indicator_items, {})
        self.assertTrue(
            all(
                item not in viewer.view.items
                for camera_items in rebuilt_indicators.values()
                for item in camera_items
            )
        )

    def test_projection_camera_bounds_include_symmetric_mirror(self) -> None:
        viewer = self._build_viewer()
        model = _build_generated_model()
        model.mesh.apply_translation((0.5, 0.0, 0.0))
        viewer.set_projection_camera_indicators_visible(True)
        viewer.set_model(model)

        np.testing.assert_allclose(
            viewer._get_projection_camera_indicator_bounds(),
            ((0.0, -0.5, -0.5), (1.0, 0.5, 0.5)),
        )
        original_items = tuple(
            item
            for camera_items in (
                viewer.projection_camera_indicator_items.values()
            )
            for item in camera_items
        )

        viewer.set_symmetric_division_preview("vertical", 0.0)

        np.testing.assert_allclose(
            viewer._get_projection_camera_indicator_bounds(),
            ((-1.0, -0.5, -0.5), (1.0, 0.5, 0.5)),
        )
        self.assertTrue(
            all(item not in viewer.view.items for item in original_items)
        )
        self.assertEqual(
            tuple(viewer.projection_camera_indicator_items),
            ALL_CAMERA_IDS,
        )

    def test_projection_camera_picker_selects_screen_geometry_without_items_at(
        self,
    ) -> None:
        viewer = self._build_viewer(face_editing_enabled=True)
        viewer.resize(640, 480)
        viewer.set_projection_camera_indicators_visible(True)
        viewer.set_model(_build_generated_model())
        viewer.show()
        _qt_application.processEvents()
        viewport = (0, 0, viewer.view.width(), viewer.view.height())
        view_projection = (
            viewer.view.projectionMatrix(viewport, viewport)
            * viewer.view.viewMatrix()
        )
        target_id = ALL_CAMERA_IDS[0]
        geometry = viewer.projection_camera_indicator_geometries[target_id]
        projected = _project_vertices_to_view(
            geometry.selection_line_positions,
            view_projection,
            viewer.view.width(),
            viewer.view.height(),
        )
        usable = projected[
            np.all(np.isfinite(projected), axis=1)
            & (projected[:, 3] > 0.0)
            & (projected[:, 2] >= -1.0)
            & (projected[:, 2] <= 1.0)
        ]
        self.assertGreater(len(usable), 0)
        position = QPointF(float(usable[0, 0]), float(usable[0, 1]))
        selected = QSignalSpy(viewer.projection_camera_selection_changed)

        with patch.object(
            viewer.view,
            "_get_clicked_items",
            side_effect=AssertionError("Camera picking must never use itemsAt()."),
        ):
            QTest.mouseClick(
                viewer.view,
                Qt.MouseButton.LeftButton,
                pos=position.toPoint(),
            )

        self.assertEqual(viewer.get_selected_projection_camera_id(), target_id)
        self.assertEqual(selected.count(), 1)
        selected_line = viewer.projection_camera_indicator_items[target_id][0]
        self.assertEqual(tuple(selected_line.color), INDICATOR_SELECTED_COLOR)

    def test_selected_camera_wheel_emits_ticks_without_zooming(self) -> None:
        viewer = self._build_viewer()
        viewer.set_projection_camera_indicators_visible(True)
        viewer.set_model(_build_generated_model())
        viewer.set_selected_projection_camera_id(ALL_CAMERA_IDS[1])
        requested = QSignalSpy(
            viewer.projection_camera_percentage_step_requested
        )
        original_distance = float(viewer.view.opts["distance"])

        viewer.view.wheelEvent(FakeWheelEvent(120))
        viewer.view.wheelEvent(FakeWheelEvent(-240))

        self.assertEqual(requested.count(), 2)
        self.assertEqual(requested.at(0), [ALL_CAMERA_IDS[1], 1])
        self.assertEqual(requested.at(1), [ALL_CAMERA_IDS[1], -2])
        self.assertEqual(float(viewer.view.opts["distance"]), original_distance)

        viewer.set_selected_projection_camera_id(None)
        viewer.view.wheelEvent(FakeWheelEvent(120))

        self.assertLess(float(viewer.view.opts["distance"]), original_distance)

    def test_partial_wheel_tick_does_not_leak_between_selected_cameras(self) -> None:
        viewer = self._build_viewer()
        viewer.set_projection_camera_indicators_visible(True)
        viewer.set_model(_build_generated_model())
        viewer.set_selected_projection_camera_id(ALL_CAMERA_IDS[0])
        requested = QSignalSpy(
            viewer.projection_camera_percentage_step_requested
        )

        viewer.view.wheelEvent(FakeWheelEvent(60))
        viewer.set_selected_projection_camera_id(ALL_CAMERA_IDS[1])
        viewer.view.wheelEvent(FakeWheelEvent(60))

        self.assertEqual(requested.count(), 0)

        viewer.view.wheelEvent(FakeWheelEvent(60))

        self.assertEqual(requested.count(), 1)
        self.assertEqual(requested.at(0), [ALL_CAMERA_IDS[1], 1])

    def test_projection_camera_percentage_api_updates_bars_and_labels(self) -> None:
        viewer = self._build_viewer()
        viewer.set_projection_camera_indicators_visible(True)
        viewer.set_model(_build_generated_model())
        percentages = (40, 20, 10, 10, 10, 10)

        viewer.set_projection_camera_percentages(percentages)

        self.assertEqual(viewer.get_projection_camera_percentages(), percentages)
        for index, camera_id in enumerate(ALL_CAMERA_IDS):
            label_item = viewer.projection_camera_indicator_items[camera_id][3]
            self.assertTrue(label_item.text.endswith(f"{percentages[index]}%"))

        with self.assertRaisesRegex(ValueError, "between 1 and 95"):
            viewer.set_projection_camera_percentages((0, 20, 20, 20, 20, 20))
        with self.assertRaisesRegex(ValueError, "more than 100"):
            viewer.set_projection_camera_percentages((20, 20, 20, 20, 20, 1))


class BlenderNavigationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.view = SelectableGLViewWidget()

    def tearDown(self) -> None:
        self.view.close()
        self.view.deleteLater()
        _qt_application.processEvents()

    def test_middle_drag_orbits_and_shift_middle_drag_pans(self) -> None:
        orbit = Mock()
        pan = Mock()
        self.view.orbit = orbit
        self.view.pan = pan
        self.view.mousePos = QPointF(10.0, 20.0)

        orbit_event = FakeMouseMoveEvent(
            position=QPointF(16.0, 25.0),
            buttons=Qt.MouseButton.MiddleButton,
        )
        self.view.mouseMoveEvent(orbit_event)

        orbit.assert_called_once_with(-6.0, 5.0)
        self.assertTrue(orbit_event.was_accepted)
        self.assertFalse(pan.called)

        self.view.mousePos = QPointF(16.0, 25.0)
        pan_event = FakeMouseMoveEvent(
            position=QPointF(19.0, 21.0),
            buttons=Qt.MouseButton.MiddleButton,
            modifiers=Qt.KeyboardModifier.ShiftModifier,
        )
        self.view.mouseMoveEvent(pan_event)

        pan.assert_called_once_with(3.0, -4.0, 0.0, relative="view")
        self.assertTrue(pan_event.was_accepted)

    def test_left_drag_does_not_orbit_and_wheel_zooms(self) -> None:
        orbit = Mock()
        pan = Mock()
        self.view.orbit = orbit
        self.view.pan = pan
        self.view.mousePos = QPointF(1.0, 1.0)

        left_drag_event = FakeMouseMoveEvent(
            position=QPointF(8.0, 5.0),
            buttons=Qt.MouseButton.LeftButton,
        )
        self.view.mouseMoveEvent(left_drag_event)

        self.assertTrue(left_drag_event.was_accepted)
        self.assertFalse(orbit.called)
        self.assertFalse(pan.called)

        original_distance = float(self.view.opts["distance"])
        wheel_event = FakeWheelEvent(120)
        self.view.wheelEvent(wheel_event)

        self.assertTrue(wheel_event.was_accepted)
        self.assertLess(float(self.view.opts["distance"]), original_distance)


class FirstPersonNavigationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.view = SelectableGLViewWidget()
        self.view.resize(320, 180)

    def tearDown(self) -> None:
        self.view.exit_first_person_mode()
        self.view.close()
        self.view.deleteLater()
        _qt_application.processEvents()

    def test_toggle_switches_between_orbit_and_first_person(self) -> None:
        self.assertEqual(self.view.get_navigation_mode(), NAVIGATION_MODE_ORBIT)

        self.assertEqual(
            self.view.toggle_navigation_mode(),
            NAVIGATION_MODE_FIRST_PERSON,
        )
        self.assertTrue(self.view.is_first_person_active)
        self.assertEqual(
            self.view.toggle_navigation_mode(),
            NAVIGATION_MODE_ORBIT,
        )
        self.assertFalse(self.view.is_first_person_active)

    def test_gravity_zqsd_uses_yaw_and_preserves_camera_z(self) -> None:
        self.view.set_first_person_camera_pose(
            CameraPose(
                x=1.0,
                y=2.0,
                z=1.7,
                pitch_degrees=60.0,
            )
        )
        self.view.enter_first_person_mode()
        self.view.keyPressEvent(
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_Z,
                Qt.KeyboardModifier.NoModifier,
            )
        )

        self.view.step_first_person_movement(0.4)
        moved_pose = self.view.get_first_person_camera_pose()

        self.assertAlmostEqual(moved_pose.x, 2.0)
        self.assertAlmostEqual(moved_pose.y, 2.0)
        self.assertAlmostEqual(moved_pose.z, 1.7)

    def test_noclip_forward_movement_follows_camera_pitch(self) -> None:
        self.view.set_first_person_movement_mode(
            FIRST_PERSON_NAVIGATION_MODE_NOCLIP
        )
        self.view.set_first_person_camera_pose(
            CameraPose(
                x=1.0,
                y=2.0,
                z=1.7,
                pitch_degrees=30.0,
            )
        )
        self.view.enter_first_person_mode()
        self.view.keyPressEvent(
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_Z,
                Qt.KeyboardModifier.NoModifier,
            )
        )

        self.view.step_first_person_movement(0.4)
        moved_pose = self.view.get_first_person_camera_pose()

        self.assertAlmostEqual(moved_pose.x, 1.0 + np.sqrt(3.0) / 2.0)
        self.assertAlmostEqual(moved_pose.y, 2.0)
        self.assertAlmostEqual(moved_pose.z, 2.2)

    def test_noclip_strafe_remains_horizontal_at_nonzero_pitch(self) -> None:
        self.view.set_first_person_movement_mode(
            FIRST_PERSON_NAVIGATION_MODE_NOCLIP
        )
        self.view.set_first_person_camera_pose(
            CameraPose(
                x=1.0,
                y=2.0,
                z=1.7,
                pitch_degrees=75.0,
            )
        )
        self.view.enter_first_person_mode()
        self.view.keyPressEvent(
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_D,
                Qt.KeyboardModifier.NoModifier,
            )
        )

        self.view.step_first_person_movement(0.4)
        moved_pose = self.view.get_first_person_camera_pose()

        self.assertAlmostEqual(moved_pose.x, 1.0)
        self.assertAlmostEqual(moved_pose.y, 1.0)
        self.assertAlmostEqual(moved_pose.z, 1.7)

    def test_noclip_has_no_automatic_fall_or_floor_clamp(self) -> None:
        self.view.set_first_person_movement_mode(
            FIRST_PERSON_NAVIGATION_MODE_NOCLIP
        )
        self.view.set_first_person_camera_pose(CameraPose(z=0.2))
        self.view.enter_first_person_mode()

        self.view.step_first_person_movement(1.0)
        self.assertAlmostEqual(
            self.view.get_first_person_camera_pose().z,
            0.2,
        )

        self.view.keyPressEvent(
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_R,
                Qt.KeyboardModifier.NoModifier,
            )
        )
        self.view.step_first_person_movement(0.4)

        self.assertAlmostEqual(
            self.view.get_first_person_camera_pose().z,
            -0.8,
        )

    def test_first_person_movement_mode_rejects_unknown_values(self) -> None:
        self.assertEqual(
            self.view.get_first_person_movement_mode(),
            FIRST_PERSON_NAVIGATION_MODE_GRAVITY,
        )

        with self.assertRaisesRegex(ValueError, "expected one of"):
            self.view.set_first_person_movement_mode("fly")

    def test_q_moves_left_and_d_moves_right_on_french_layout(self) -> None:
        self.view.enter_first_person_mode()
        cases = (
            (0.0, Qt.Key.Key_Q, (0.0, 1.0)),
            (0.0, Qt.Key.Key_D, (0.0, -1.0)),
            (90.0, Qt.Key.Key_Q, (-1.0, 0.0)),
            (90.0, Qt.Key.Key_D, (1.0, 0.0)),
        )
        for yaw_degrees, key, expected_xy in cases:
            with self.subTest(yaw_degrees=yaw_degrees, key=key):
                self.view.set_first_person_camera_pose(
                    CameraPose(z=1.7, yaw_degrees=yaw_degrees)
                )
                self.view.keyPressEvent(
                    QKeyEvent(
                        QEvent.Type.KeyPress,
                        key,
                        Qt.KeyboardModifier.NoModifier,
                    )
                )
                self.view.step_first_person_movement(0.4)
                self.view.keyReleaseEvent(
                    QKeyEvent(
                        QEvent.Type.KeyRelease,
                        key,
                        Qt.KeyboardModifier.NoModifier,
                    )
                )

                moved_pose = self.view.get_first_person_camera_pose()
                self.assertAlmostEqual(moved_pose.x, expected_xy[0])
                self.assertAlmostEqual(moved_pose.y, expected_xy[1])
                self.assertAlmostEqual(moved_pose.z, 1.7)

    def test_gravity_mode_r_moves_down_and_f_moves_up(self) -> None:
        self.view.enter_first_person_mode()
        cases = (
            (Qt.Key.Key_R, 0.7),
            (Qt.Key.Key_F, 2.7),
        )
        for key, expected_z in cases:
            with self.subTest(key=key):
                self.view.set_first_person_camera_pose(
                    CameraPose(x=1.0, y=2.0, z=1.7)
                )
                self.view.keyPressEvent(
                    QKeyEvent(
                        QEvent.Type.KeyPress,
                        key,
                        Qt.KeyboardModifier.NoModifier,
                    )
                )
                self.view.step_first_person_movement(0.4)
                self.view.keyReleaseEvent(
                    QKeyEvent(
                        QEvent.Type.KeyRelease,
                        key,
                        Qt.KeyboardModifier.NoModifier,
                    )
                )

                moved_pose = self.view.get_first_person_camera_pose()
                self.assertAlmostEqual(moved_pose.x, 1.0)
                self.assertAlmostEqual(moved_pose.y, 2.0)
                self.assertAlmostEqual(moved_pose.z, expected_z)

    def test_repeated_vertical_key_press_keeps_moving_until_release(self) -> None:
        self.view.set_first_person_camera_pose(CameraPose(z=1.7))
        self.view.enter_first_person_mode()
        press_event = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_F,
            Qt.KeyboardModifier.NoModifier,
        )

        self.view.keyPressEvent(press_event)
        self.view.keyPressEvent(press_event)
        self.view.step_first_person_movement(0.2)
        self.view.step_first_person_movement(0.2)

        self.assertAlmostEqual(
            self.view.get_first_person_camera_pose().z,
            2.7,
        )
        self.view.keyReleaseEvent(
            QKeyEvent(
                QEvent.Type.KeyRelease,
                Qt.Key.Key_F,
                Qt.KeyboardModifier.NoModifier,
            )
        )
        self.view.step_first_person_movement(0.4)
        self.assertAlmostEqual(
            self.view.get_first_person_camera_pose().z,
            2.7,
        )

    def test_right_click_releases_pointer_but_preserves_first_person_view(
        self,
    ) -> None:
        self.view.set_first_person_camera_pose(CameraPose(z=1.7))
        self.view.enter_first_person_mode()
        orbit = Mock()
        self.view.orbit = orbit
        center = QPointF(self.view.rect().center())
        move_event = FakeMouseMoveEvent(
            position=center + QPointF(10.0, 5.0),
            buttons=Qt.MouseButton.NoButton,
        )

        self.view.mouseMoveEvent(move_event)
        pose = self.view.get_first_person_camera_pose()

        self.assertTrue(move_event.was_accepted)
        self.assertFalse(orbit.called)
        self.assertAlmostEqual(pose.yaw_degrees, -1.6)
        self.assertAlmostEqual(pose.pitch_degrees, -0.8)

        first_person_distance = float(self.view.opts["distance"])
        release_event = FakeMousePressEvent(button=Qt.MouseButton.RightButton)
        self.view.mousePressEvent(release_event)

        self.assertTrue(release_event.was_accepted)
        self.assertEqual(
            self.view.get_navigation_mode(),
            NAVIGATION_MODE_FIRST_PERSON,
        )
        self.assertFalse(self.view.is_first_person_pointer_captured)
        self.assertAlmostEqual(
            float(self.view.opts["distance"]),
            first_person_distance,
        )

    def test_canvas_ctrl_temporarily_frees_and_recaptures_pointer(self) -> None:
        panel_button = QPushButton(self.view)
        self.view.set_first_person_ctrl_interaction_enabled(True)
        self.view.enter_first_person_mode()

        QTest.keyPress(self.view, Qt.Key.Key_Control)

        self.assertTrue(self.view.is_first_person_ctrl_interaction_active)
        self.assertFalse(self.view.is_first_person_pointer_captured)

        QTest.keyRelease(panel_button, Qt.Key.Key_Control)

        self.assertFalse(self.view.is_first_person_ctrl_interaction_active)
        self.assertTrue(self.view.is_first_person_pointer_captured)

    def test_canvas_ctrl_filter_tracks_orbit_child_events_without_consuming(
        self,
    ) -> None:
        panel_button = QPushButton(self.view)
        self.view.set_first_person_ctrl_interaction_enabled(True)
        changed = QSignalSpy(self.view.control_modifier_changed)
        press = QKeyEvent(
            QKeyEvent.Type.KeyPress,
            Qt.Key.Key_Control,
            Qt.KeyboardModifier.ControlModifier,
        )
        press.ignore()

        self.assertFalse(self.view.eventFilter(panel_button, press))

        self.assertTrue(self.view.is_control_modifier_pressed)
        self.assertFalse(self.view.is_first_person_ctrl_interaction_active)
        self.assertFalse(press.isAccepted())
        self.assertEqual(changed.count(), 1)
        self.assertEqual(changed.at(0), [True])

        release = QKeyEvent(
            QKeyEvent.Type.KeyRelease,
            Qt.Key.Key_Control,
            Qt.KeyboardModifier.NoModifier,
        )
        release.ignore()

        self.assertFalse(self.view.eventFilter(panel_button, release))

        self.assertFalse(self.view.is_control_modifier_pressed)
        self.assertFalse(release.isAccepted())
        self.assertEqual(changed.count(), 2)
        self.assertEqual(changed.at(1), [False])

    def test_canvas_ctrl_filter_consumes_first_person_child_events(self) -> None:
        panel_button = QPushButton(self.view)
        self.view.set_first_person_ctrl_interaction_enabled(True)
        self.view.enter_first_person_mode()
        press = QKeyEvent(
            QKeyEvent.Type.KeyPress,
            Qt.Key.Key_Control,
            Qt.KeyboardModifier.ControlModifier,
        )
        press.ignore()

        self.assertTrue(self.view.eventFilter(panel_button, press))

        self.assertTrue(press.isAccepted())
        self.assertTrue(self.view.is_control_modifier_pressed)
        self.assertTrue(self.view.is_first_person_ctrl_interaction_active)
        self.assertFalse(self.view.is_first_person_pointer_captured)

        release = QKeyEvent(
            QKeyEvent.Type.KeyRelease,
            Qt.Key.Key_Control,
            Qt.KeyboardModifier.NoModifier,
        )
        release.ignore()

        self.assertTrue(self.view.eventFilter(panel_button, release))

        self.assertTrue(release.isAccepted())
        self.assertFalse(self.view.is_control_modifier_pressed)
        self.assertFalse(self.view.is_first_person_ctrl_interaction_active)
        self.assertTrue(self.view.is_first_person_pointer_captured)

    def test_canvas_right_click_persistently_frees_the_pointer(self) -> None:
        panel_button = QPushButton(self.view)
        self.view.set_first_person_ctrl_interaction_enabled(True)
        self.view.enter_first_person_mode()

        right_click = FakeMousePressEvent(button=Qt.MouseButton.RightButton)
        self.view.mousePressEvent(right_click)

        self.assertTrue(right_click.was_accepted)
        self.assertEqual(
            self.view.get_navigation_mode(),
            NAVIGATION_MODE_FIRST_PERSON,
        )
        self.assertFalse(self.view.is_first_person_pointer_captured)

        QTest.keyPress(self.view, Qt.Key.Key_Control)
        QTest.keyRelease(panel_button, Qt.Key.Key_Control)
        self.view.eventFilter(
            _qt_application,
            QEvent(QEvent.Type.ApplicationDeactivate),
        )
        self.view.eventFilter(
            _qt_application,
            QEvent(QEvent.Type.ApplicationActivate),
        )
        self.view.set_rectangle_drawing_enabled(True)
        self.view.set_rectangle_drawing_enabled(False)

        self.assertFalse(self.view.is_first_person_pointer_captured)

        self.view.exit_first_person_mode()
        self.view.enter_first_person_mode()

        self.assertTrue(self.view.is_first_person_pointer_captured)

    def test_canvas_left_click_recaptures_after_right_click_release(self) -> None:
        self.view.set_first_person_ctrl_interaction_enabled(True)
        self.view.enter_first_person_mode()
        self.view.mousePressEvent(
            FakeMousePressEvent(button=Qt.MouseButton.RightButton)
        )

        self.assertFalse(self.view.is_first_person_pointer_captured)

        left_click = FakeMousePressEvent(button=Qt.MouseButton.LeftButton)
        self.view.mousePressEvent(left_click)

        self.assertTrue(left_click.was_accepted)
        self.assertTrue(self.view.is_first_person_pointer_captured)
        self.assertEqual(
            self.view.get_navigation_mode(),
            NAVIGATION_MODE_FIRST_PERSON,
        )

    def test_canvas_ctrl_click_does_not_recapture_a_right_click_release(
        self,
    ) -> None:
        self.view.set_first_person_ctrl_interaction_enabled(True)
        self.view.enter_first_person_mode()
        self.view.mousePressEvent(
            FakeMousePressEvent(button=Qt.MouseButton.RightButton)
        )
        QTest.keyPress(self.view, Qt.Key.Key_Control)

        left_click = FakeMousePressEvent(button=Qt.MouseButton.LeftButton)
        self.view.mousePressEvent(left_click)

        self.assertTrue(left_click.was_accepted)
        self.assertFalse(self.view.is_first_person_pointer_captured)

        QTest.keyRelease(self.view, Qt.Key.Key_Control)
        self.assertFalse(self.view.is_first_person_pointer_captured)

    def test_canvas_window_tool_keeps_cursor_until_tool_finishes(self) -> None:
        panel_button = QPushButton(self.view)
        self.view.set_first_person_ctrl_interaction_enabled(True)
        self.view.enter_first_person_mode()
        QTest.keyPress(self.view, Qt.Key.Key_Control)

        self.view.set_rectangle_drawing_enabled(True)
        QTest.keyRelease(panel_button, Qt.Key.Key_Control)

        self.assertTrue(self.view.is_rectangle_drawing_enabled)
        self.assertFalse(self.view.is_first_person_ctrl_interaction_active)
        self.assertFalse(self.view.is_first_person_pointer_captured)

        self.view.set_rectangle_drawing_enabled(False)

        self.assertTrue(self.view.is_first_person_pointer_captured)

    def test_canvas_gizmo_drag_delays_recapture_until_drag_finishes(self) -> None:
        panel_button = QPushButton(self.view)
        self.view.set_first_person_ctrl_interaction_enabled(True)
        self.view.enter_first_person_mode()
        QTest.keyPress(self.view, Qt.Key.Key_Control)

        self.view.reserve_primary_pointer_drag()
        QTest.keyRelease(panel_button, Qt.Key.Key_Control)

        self.assertFalse(self.view.is_first_person_pointer_captured)

        self.view.release_primary_pointer_drag()

        self.assertTrue(self.view.is_first_person_pointer_captured)

    def test_canvas_ctrl_release_does_not_recapture_while_app_is_inactive(
        self,
    ) -> None:
        panel_button = QPushButton(self.view)
        self.view.set_first_person_ctrl_interaction_enabled(True)
        self.view.enter_first_person_mode()
        QTest.keyPress(self.view, Qt.Key.Key_Control)

        self.view.eventFilter(
            _qt_application,
            QEvent(QEvent.Type.ApplicationDeactivate),
        )
        QTest.keyRelease(panel_button, Qt.Key.Key_Control)

        self.assertFalse(self.view.is_first_person_pointer_captured)

        self.view.eventFilter(
            _qt_application,
            QEvent(QEvent.Type.ApplicationActivate),
        )
        self.assertTrue(self.view.is_first_person_pointer_captured)
        QTest.keyPress(self.view, Qt.Key.Key_Control)
        QTest.keyRelease(panel_button, Qt.Key.Key_Control)

        self.assertTrue(self.view.is_first_person_pointer_captured)

    def test_canvas_first_person_recaptures_after_becoming_visible(self) -> None:
        self.view.set_first_person_ctrl_interaction_enabled(True)
        self.view.show()
        self.view.enter_first_person_mode()

        self.view.hide()
        self.assertFalse(self.view.is_first_person_pointer_captured)

        self.view.show()

        self.assertTrue(self.view.is_first_person_pointer_captured)

    def test_canvas_first_person_selection_uses_ctrl_and_cpu_click(self) -> None:
        self.view.set_item_click_selection_enabled(False)
        self.view.set_viewport_click_selection_enabled(True)
        self.view.set_first_person_ctrl_interaction_enabled(True)
        self.view.enter_first_person_mode()
        clicked_positions: list[QPointF] = []
        self.view.viewport_clicked.connect(clicked_positions.append)
        self.view._get_clicked_items = Mock(
            side_effect=AssertionError("Canvas must not use GL item picking.")
        )
        position = QPointF(42.0, 24.0)

        QTest.keyPress(self.view, Qt.Key.Key_Control)
        self.view.mousePressEvent(
            FakeMousePressEvent(
                button=Qt.MouseButton.LeftButton,
                position=position,
            )
        )
        self.view.mouseReleaseEvent(
            FakeMousePressEvent(
                button=Qt.MouseButton.LeftButton,
                position=position,
            )
        )

        self.assertEqual(clicked_positions, [position])
        self.assertFalse(self.view._get_clicked_items.called)

        QTest.keyRelease(self.view, Qt.Key.Key_Control)
        self.assertTrue(self.view.is_first_person_pointer_captured)

    def test_released_first_person_pointer_can_select_viewport_content(
        self,
    ) -> None:
        self.view.enter_first_person_mode()
        self.view.release_first_person_pointer_capture()
        clicked_positions: list[QPointF] = []
        clicked_items: list[object] = []
        self.view.viewport_clicked.connect(clicked_positions.append)
        self.view.items_clicked.connect(clicked_items.append)
        self.view._get_clicked_items = Mock(return_value=["wall"])
        position = QPointF(42.0, 24.0)

        self.view.mousePressEvent(
            FakeMousePressEvent(
                button=Qt.MouseButton.LeftButton,
                position=position,
            )
        )
        self.view.mouseReleaseEvent(
            FakeMousePressEvent(
                button=Qt.MouseButton.LeftButton,
                position=position,
            )
        )

        self.assertEqual(clicked_positions, [position])
        self.assertEqual(clicked_items, [["wall"]])
        self.assertEqual(
            self.view.get_navigation_mode(),
            NAVIGATION_MODE_FIRST_PERSON,
        )

    def test_focus_loss_releases_capture_without_leaving_first_person(
        self,
    ) -> None:
        self.view.enter_first_person_mode()
        self.view.keyPressEvent(
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_Z,
                Qt.KeyboardModifier.NoModifier,
            )
        )

        self.view.focusOutEvent(QFocusEvent(QEvent.Type.FocusOut))

        self.assertEqual(
            self.view.get_navigation_mode(),
            NAVIGATION_MODE_FIRST_PERSON,
        )
        self.assertFalse(self.view.is_first_person_pointer_captured)
        self.assertEqual(self.view._pressed_movement_keys, set())

    def test_rectangle_tool_preserves_first_person_camera_mode(self) -> None:
        self.view.enter_first_person_mode()

        self.view.set_rectangle_drawing_enabled(True)

        self.assertTrue(self.view.is_rectangle_drawing_enabled)
        self.assertFalse(self.view.is_first_person_pointer_captured)
        self.assertEqual(
            self.view.get_navigation_mode(),
            NAVIGATION_MODE_FIRST_PERSON,
        )
