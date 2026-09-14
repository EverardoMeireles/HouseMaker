# ### Environment setup ###
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import numpy as np
import trimesh
from PIL import Image
from PySide6.QtWidgets import QApplication, QPushButton

from housemaker.glb import (
    GeneratedModel,
    PreviewPlacedObject,
    PreviewTexturedSurface,
    PreviewTexturedWall,
    convert_to_preview_model,
)
from housemaker.models import LevelData, VertexData
from housemaker.surface_geometry import (
    SURFACE_TYPE_CEILING,
    SURFACE_TYPE_FLOOR,
    SURFACE_TYPE_WALL,
    FixedSurface,
    build_fixed_surfaces,
)
from housemaker.viewer import GlbViewerWidget

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_surface(
    surface_id: str,
    surface_type: str,
    vertices: tuple[tuple[float, float, float], ...],
    *,
    level_index: int = 0,
) -> FixedSurface:
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        process=False,
    )
    wall_kwargs = (
        {
            "wall_key": "1:2",
            "wall_start_world": vertices[0],
            "wall_end_world": vertices[1],
            "wall_height_meters": 3.0,
        }
        if surface_type == SURFACE_TYPE_WALL
        else {}
    )
    return FixedSurface(
        surface_id=surface_id,
        surface_type=surface_type,
        level_index=level_index,
        room_index=None,
        mesh=mesh,
        area_square_meters=16.0 if surface_type != SURFACE_TYPE_WALL else 12.0,
        **wall_kwargs,
    )


def _build_surfaces() -> tuple[FixedSurface, ...]:
    return (
        _build_surface(
            "level:0/wall:1:2",
            SURFACE_TYPE_WALL,
            (
                (0.0, 0.0, 0.0),
                (4.0, 0.0, 0.0),
                (4.0, 0.0, 3.0),
                (0.0, 0.0, 3.0),
            ),
        ),
        _build_surface(
            "level:0/floor",
            SURFACE_TYPE_FLOOR,
            (
                (0.0, 0.0, 0.0),
                (4.0, 0.0, 0.0),
                (4.0, 4.0, 0.0),
                (0.0, 4.0, 0.0),
            ),
        ),
        _build_surface(
            "level:0/ceiling",
            SURFACE_TYPE_CEILING,
            (
                (0.0, 0.0, 3.0),
                (4.0, 0.0, 3.0),
                (4.0, 4.0, 3.0),
                (0.0, 4.0, 3.0),
            ),
        ),
    )


def _build_square_level(index: int, name: str) -> LevelData:
    vertex_data = VertexData()
    vertices = tuple(
        vertex_data.add_vertex(x, y)
        for x, y in (
            (0.0, 0.0),
            (200.0, 0.0),
            (200.0, 200.0),
            (0.0, 200.0),
        )
    )
    for start, end in zip(vertices, (*vertices[1:], vertices[0]), strict=True):
        vertex_data.add_edge(start.id, end.id)
    return LevelData(
        index=index,
        name=name,
        vertex_data=vertex_data,
        image_size_pixels=(200.0, 200.0),
    )


def _build_model(
    surfaces: tuple[FixedSurface, ...],
    *,
    object_id: str,
) -> GeneratedModel:
    base_mesh = trimesh.util.concatenate(
        tuple(surface.mesh.copy() for surface in surfaces)
    )
    local_object_mesh = trimesh.creation.box(extents=(0.5, 0.5, 0.5))
    placement_transform = np.eye(4, dtype=float)
    placement_transform[:3, 3] = (2.0, 2.0, 0.25)
    placed_object = PreviewPlacedObject(
        object_id=object_id,
        meshes=(local_object_mesh,),
        placement_transform=placement_transform,
        world_position=(2.0, 2.0, 0.25),
        rotation_degrees=(0.0, 0.0, 0.0),
    )
    return GeneratedModel(
        mesh=base_mesh,
        scene=trimesh.Scene(base_mesh),
        glb_bytes=b"",
        preview_base_mesh=base_mesh,
        preview_placed_objects=[placed_object],
    )


def _build_textured_surface(surface: FixedSurface) -> PreviewTexturedSurface:
    mesh = surface.mesh.copy()
    mesh.visual = trimesh.visual.texture.TextureVisuals(
        uv=np.asarray(
            ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
            dtype=float,
        ),
        image=Image.new("RGBA", (2, 2), (180, 140, 100, 255)),
    )
    return PreviewTexturedSurface(
        surface_id=surface.surface_id,
        surface_type=surface.surface_type,
        mesh=mesh,
        level_index=surface.level_index,
    )


# ### Canvas scene visibility tests ###
class CanvasSceneVisibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.viewer = GlbViewerWidget(window_editing_enabled=True)
        self.surfaces = _build_surfaces()
        self.viewer.set_model(_build_model(self.surfaces, object_id="chair"))
        self.viewer.set_wall_targets(self.surfaces)

    def tearDown(self) -> None:
        self.viewer.close()
        self.viewer.deleteLater()
        _qt_application.processEvents()

    def test_side_panel_controls_surface_focus_and_ceiling_visibility(self) -> None:
        floor_id = "level:0/floor"
        ceiling_id = "level:0/ceiling"
        wall_id = "level:0/wall:1:2"

        self.assertIsNotNone(self.viewer.highlight_floor_button)
        self.assertIsNotNone(self.viewer.hide_ceiling_button)
        floor_button = self.viewer.highlight_floor_button
        hide_ceiling_button = self.viewer.hide_ceiling_button
        assert floor_button is not None
        assert hide_ceiling_button is not None
        self.assertTrue(floor_button.isCheckable())
        self.assertTrue(hide_ceiling_button.isCheckable())
        self.assertIsNone(
            self.viewer.findChild(
                QPushButton,
                "canvas-highlight-ceiling-button",
            )
        )
        self.assertIsNone(self.viewer.get_canvas_surface_focus_type())
        self.assertFalse(self.viewer.get_canvas_ceiling_hidden())
        self.assertEqual(
            set(self.viewer.get_visible_canvas_surface_ids()),
            {wall_id, floor_id, ceiling_id},
        )
        self.assertEqual(
            self.viewer.get_visible_placed_object_ids(),
            ("chair",),
        )

        floor_button.click()

        self.assertEqual(
            self.viewer.get_canvas_surface_focus_type(),
            SURFACE_TYPE_FLOOR,
        )
        self.assertTrue(floor_button.isChecked())
        self.assertEqual(
            self.viewer.get_visible_canvas_surface_ids(),
            (floor_id,),
        )
        self.assertEqual(self.viewer.get_visible_placed_object_ids(), ())

        floor_button.click()

        self.assertIsNone(self.viewer.get_canvas_surface_focus_type())
        self.assertFalse(floor_button.isChecked())
        self.assertEqual(
            set(self.viewer.get_visible_canvas_surface_ids()),
            {wall_id, floor_id, ceiling_id},
        )
        self.assertEqual(self.viewer.get_visible_placed_object_ids(), ("chair",))

        hide_ceiling_button.click()

        self.assertTrue(self.viewer.get_canvas_ceiling_hidden())
        self.assertTrue(hide_ceiling_button.isChecked())
        self.assertEqual(
            set(self.viewer.get_visible_canvas_surface_ids()),
            {wall_id, floor_id},
        )
        self.assertEqual(
            self.viewer.get_visible_placed_object_ids(),
            ("chair",),
        )

    def test_visibility_state_survives_preserved_camera_model_rebuild(self) -> None:
        floor_id = "level:0/floor"
        assert self.viewer.highlight_floor_button is not None
        assert self.viewer.hide_ceiling_button is not None
        self.viewer.hide_ceiling_button.click()
        self.viewer.highlight_floor_button.click()

        self.viewer.set_model(
            _build_model(self.surfaces, object_id="replacement-chair"),
            preserve_camera=True,
        )

        self.assertEqual(
            self.viewer.get_canvas_surface_focus_type(),
            SURFACE_TYPE_FLOOR,
        )
        self.assertTrue(self.viewer.get_canvas_ceiling_hidden())
        self.assertTrue(self.viewer.highlight_floor_button.isChecked())
        self.assertTrue(self.viewer.hide_ceiling_button.isChecked())
        self.assertEqual(
            self.viewer.get_visible_canvas_surface_ids(),
            (floor_id,),
        )
        self.assertEqual(self.viewer.get_visible_placed_object_ids(), ())

        self.viewer.highlight_floor_button.click()

        self.assertEqual(
            set(self.viewer.get_visible_canvas_surface_ids()),
            {"level:0/wall:1:2", floor_id},
        )
        self.assertEqual(
            self.viewer.get_visible_placed_object_ids(),
            ("replacement-chair",),
        )

    def test_focus_is_latent_while_a_model_has_no_semantic_targets(self) -> None:
        assert self.viewer.highlight_floor_button is not None
        self.viewer.set_canvas_surface_focus_type(SURFACE_TYPE_FLOOR)
        ao_mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
        ao_model = GeneratedModel(
            mesh=ao_mesh,
            scene=trimesh.Scene(ao_mesh.copy()),
            glb_bytes=b"",
        )

        self.viewer.set_model(ao_model, preserve_camera=True)
        self.viewer.set_wall_targets(())

        displayed_ao_mesh = self.viewer._get_display_mesh()
        self.assertIs(displayed_ao_mesh, ao_mesh)
        self.assertEqual(
            self.viewer.get_canvas_surface_focus_type(),
            SURFACE_TYPE_FLOOR,
        )
        self.assertTrue(self.viewer.highlight_floor_button.isChecked())

        regular_model = _build_model(
            self.surfaces,
            object_id="replacement-chair",
        )
        self.viewer.set_model(regular_model, preserve_camera=True)
        self.assertIs(
            self.viewer._get_display_mesh(),
            regular_model.preview_base_mesh,
        )

        self.viewer.set_wall_targets(self.surfaces)

        focused_mesh = self.viewer._get_display_mesh()
        self.assertIsNotNone(focused_mesh)
        assert focused_mesh is not None
        self.assertEqual(len(focused_mesh.faces), 2)
        np.testing.assert_allclose(
            focused_mesh.bounds,
            self.surfaces[1].mesh.bounds,
        )

    def test_programmatic_selection_cannot_restore_hidden_surfaces(self) -> None:
        wall_id = "level:0/wall:1:2"
        floor_id = "level:0/floor"
        ceiling_id = "level:0/ceiling"
        self.viewer.set_canvas_surface_focus_type(SURFACE_TYPE_FLOOR)

        self.assertFalse(self.viewer.select_wall_target(wall_id))
        self.assertFalse(
            self.viewer.select_canvas_surface_target(ceiling_id)
        )
        self.viewer.set_selected_canvas_surface_ids(
            (wall_id, ceiling_id, floor_id)
        )

        self.assertEqual(
            self.viewer.get_selected_canvas_surface_ids(),
            (floor_id,),
        )
        self.assertIsNone(self.viewer.get_selected_wall_surface_id())
        assert self.viewer.add_window_button is not None
        self.assertFalse(self.viewer.add_window_button.isEnabled())

        self.viewer.set_canvas_surface_focus_type(None)
        self.viewer.set_canvas_ceiling_hidden(True)
        self.viewer.set_selected_canvas_surface_ids((ceiling_id,))

        self.assertEqual(self.viewer.get_selected_canvas_surface_ids(), ())

    def test_hidden_ceiling_is_not_in_a_pending_level_transform_outline(
        self,
    ) -> None:
        wall = self.surfaces[0]
        floor = self.surfaces[1]
        distant_ceiling = _build_surface(
            "level:0/ceiling",
            SURFACE_TYPE_CEILING,
            (
                (10.0, 10.0, 3.0),
                (14.0, 10.0, 3.0),
                (14.0, 14.0, 3.0),
                (10.0, 14.0, 3.0),
            ),
        )
        surfaces = (wall, floor, distant_ceiling)
        self.viewer.set_wall_targets(surfaces)
        self.viewer.set_model(
            _build_model(surfaces, object_id="replacement-chair"),
            preserve_camera=True,
        )
        self.viewer.set_canvas_ceiling_hidden(True)

        self.viewer.set_level_transform_preview(0, np.eye(4, dtype=float))

        positions = self.viewer._level_transform_preview_source_positions
        self.assertIsNotNone(positions)
        assert positions is not None
        self.assertLessEqual(float(np.max(positions[:, 0])), 4.0)
        self.assertLessEqual(float(np.max(positions[:, 1])), 4.0)

    def test_visibility_change_rebuilds_an_active_level_transform_outline(
        self,
    ) -> None:
        wall = self.surfaces[0]
        floor = self.surfaces[1]
        distant_ceiling = _build_surface(
            "level:0/ceiling",
            SURFACE_TYPE_CEILING,
            (
                (10.0, 10.0, 3.0),
                (14.0, 10.0, 3.0),
                (14.0, 14.0, 3.0),
                (10.0, 14.0, 3.0),
            ),
        )
        surfaces = (wall, floor, distant_ceiling)
        self.viewer.set_wall_targets(surfaces)
        self.viewer.set_model(
            _build_model(surfaces, object_id="replacement-chair"),
            preserve_camera=True,
        )
        self.viewer.set_level_transform_preview(0, np.eye(4, dtype=float))

        initial_positions = self.viewer._level_transform_preview_source_positions
        self.assertIsNotNone(initial_positions)
        assert initial_positions is not None
        self.assertGreater(float(np.max(initial_positions[:, 0])), 4.0)

        self.viewer.set_canvas_ceiling_hidden(True)

        hidden_positions = self.viewer._level_transform_preview_source_positions
        self.assertIsNotNone(hidden_positions)
        assert hidden_positions is not None
        self.assertLessEqual(float(np.max(hidden_positions[:, 0])), 4.0)
        self.assertLessEqual(float(np.max(hidden_positions[:, 1])), 4.0)

    def test_level_list_hides_complete_levels_and_their_placed_objects(
        self,
    ) -> None:
        upper_surfaces = (
            _build_surface(
                "level:1/wall:1:2",
                SURFACE_TYPE_WALL,
                (
                    (0.0, 0.0, 4.0),
                    (4.0, 0.0, 4.0),
                    (4.0, 0.0, 7.0),
                    (0.0, 0.0, 7.0),
                ),
                level_index=1,
            ),
            _build_surface(
                "level:1/floor",
                SURFACE_TYPE_FLOOR,
                (
                    (0.0, 0.0, 4.0),
                    (4.0, 0.0, 4.0),
                    (4.0, 4.0, 4.0),
                    (0.0, 4.0, 4.0),
                ),
                level_index=1,
            ),
            _build_surface(
                "level:1/ceiling",
                SURFACE_TYPE_CEILING,
                (
                    (0.0, 0.0, 7.0),
                    (4.0, 0.0, 7.0),
                    (4.0, 4.0, 7.0),
                    (0.0, 4.0, 7.0),
                ),
                level_index=1,
            ),
        )
        surfaces = (*self.surfaces, *upper_surfaces)
        self.viewer.set_model(
            _build_model(surfaces, object_id="chair"),
            preserve_camera=True,
        )
        self.viewer.set_wall_targets(surfaces)
        self.viewer.set_canvas_scene_levels(
            ((1, "L3 Upper"), (0, "L2 Ground")),
            placed_object_levels={"chair": 0},
        )

        level_list = self.viewer.canvas_level_visibility_list
        self.assertIsNotNone(level_list)
        assert level_list is not None
        self.assertEqual(level_list.count(), 2)
        self.assertEqual(
            tuple(level_list.item(index).text() for index in range(2)),
            ("L3 Upper", "L2 Ground"),
        )
        self.assertEqual(
            self.viewer.get_visible_canvas_level_indices(),
            (1, 0),
        )

        self.viewer.set_selected_canvas_surface_ids(("level:0/floor",))
        self.viewer.set_selected_placed_object_ids(("chair",))
        level_list.item(1).setSelected(False)
        _qt_application.processEvents()

        self.assertEqual(self.viewer.get_visible_canvas_level_indices(), (1,))
        self.assertEqual(
            set(self.viewer.get_visible_canvas_surface_ids()),
            {surface.surface_id for surface in upper_surfaces},
        )
        self.assertEqual(self.viewer.get_visible_placed_object_ids(), ())
        self.assertEqual(self.viewer.get_selected_canvas_surface_ids(), ())
        self.assertEqual(self.viewer.get_selected_placed_object_ids(), ())
        self.assertNotIn("chair", self.viewer._placed_object_render_groups)
        display_mesh = self.viewer._get_display_mesh()
        self.assertIsNotNone(display_mesh)
        assert display_mesh is not None
        self.assertEqual(len(display_mesh.faces), 6)
        self.assertGreaterEqual(float(display_mesh.bounds[0, 2]), 4.0)

    def test_level_visibility_is_preserved_and_can_be_reset(self) -> None:
        self.viewer.set_canvas_scene_levels(
            ((1, "Upper"), (0, "Ground")),
            placed_object_levels={"chair": 0},
        )
        self.viewer.set_visible_canvas_level_indices((1,))

        self.viewer.set_canvas_scene_levels(
            ((2, "Roof"), (1, "Renamed upper"), (0, "Ground")),
            placed_object_levels={"chair": 0},
        )

        self.assertEqual(
            self.viewer.get_visible_canvas_level_indices(),
            (2, 1),
        )

        self.viewer.set_canvas_scene_levels(
            ((2, "Roof"), (1, "Renamed upper"), (0, "Ground")),
            placed_object_levels={"chair": 0},
            reset_visibility=True,
        )

        self.assertEqual(
            self.viewer.get_visible_canvas_level_indices(),
            (2, 1, 0),
        )

    def test_level_list_filters_complete_generated_level_primitives(self) -> None:
        levels = (
            _build_square_level(2, "Ground"),
            _build_square_level(3, "Story 1"),
        )
        model = convert_to_preview_model(levels)
        surfaces = tuple(build_fixed_surfaces(levels))
        self.viewer.set_canvas_scene_levels(
            tuple(
                (level.index, level.display_name)
                for level in reversed(levels)
            )
        )
        self.viewer.set_model(model, preserve_camera=True)
        self.viewer.set_wall_targets(surfaces)

        self.viewer.set_visible_canvas_level_indices((3,))

        display_mesh = self.viewer._get_display_mesh()
        self.assertIsNotNone(display_mesh)
        assert display_mesh is not None
        self.assertGreaterEqual(float(display_mesh.bounds[0, 2]), 3.3)
        self.assertTrue(
            all(
                surface_id.startswith("level:3/")
                for surface_id in self.viewer.get_visible_canvas_surface_ids()
            )
        )

    def test_level_list_filters_surface_and_legacy_wall_textures(self) -> None:
        lower_floor = self.surfaces[1]
        upper_floor = _build_surface(
            "level:1/floor",
            SURFACE_TYPE_FLOOR,
            (
                (0.0, 0.0, 4.0),
                (4.0, 0.0, 4.0),
                (4.0, 4.0, 4.0),
                (0.0, 4.0, 4.0),
            ),
            level_index=1,
        )
        surfaces = (*self.surfaces, upper_floor)
        model = _build_model(surfaces, object_id="chair")
        model.preview_textured_surfaces = [
            _build_textured_surface(lower_floor),
            _build_textured_surface(upper_floor),
        ]
        wall_texture = np.full((2, 2, 4), 255, dtype=np.uint8)
        model.preview_textured_walls = [
            PreviewTexturedWall(
                level_index=0,
                room_index=0,
                wall_key="lower",
                start_point=(0.0, 0.0, 0.0),
                end_point=(4.0, 0.0, 0.0),
                height_meters=3.0,
                texture_rgba=wall_texture,
            ),
            PreviewTexturedWall(
                level_index=1,
                room_index=0,
                wall_key="upper",
                start_point=(0.0, 0.0, 4.0),
                end_point=(4.0, 0.0, 4.0),
                height_meters=3.0,
                texture_rgba=wall_texture,
            ),
        ]
        self.viewer.set_wall_targets(surfaces)
        self.viewer.set_canvas_scene_levels(
            ((1, "Upper"), (0, "Ground")),
            placed_object_levels={"chair": 0},
        )
        self.viewer.set_model(model, preserve_camera=True)

        self.assertEqual(len(self.viewer.textured_surface_items), 2)
        self.assertEqual(len(self.viewer.textured_wall_items), 2)

        self.viewer.set_visible_canvas_level_indices((1,))

        self.assertEqual(len(self.viewer.textured_surface_items), 1)
        self.assertEqual(len(self.viewer.textured_wall_items), 1)

    def test_ignore_top_down_ceiling_tracks_navigation_mode(self) -> None:
        ceiling_id = "level:0/ceiling"
        self.viewer.set_selected_canvas_surface_ids((ceiling_id,))

        self.assertTrue(self.viewer.set_ignore_top_down_ceiling(True))

        self.assertTrue(self.viewer.get_ignore_top_down_ceiling())
        self.assertNotIn(
            ceiling_id,
            self.viewer.get_visible_canvas_surface_ids(),
        )
        self.assertEqual(self.viewer.get_selected_canvas_surface_ids(), ())

        self.viewer.enter_first_person_mode()

        self.assertIn(
            ceiling_id,
            self.viewer.get_visible_canvas_surface_ids(),
        )

        self.viewer.exit_first_person_mode()

        self.assertNotIn(
            ceiling_id,
            self.viewer.get_visible_canvas_surface_ids(),
        )

    def test_ceiling_focus_is_no_longer_available(self) -> None:
        with self.assertRaisesRegex(ValueError, "floor or None"):
            self.viewer.set_canvas_surface_focus_type(SURFACE_TYPE_CEILING)


if __name__ == "__main__":
    unittest.main()
