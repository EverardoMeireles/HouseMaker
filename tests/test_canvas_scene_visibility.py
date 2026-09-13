# ### Environment setup ###
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import numpy as np
import trimesh
from PySide6.QtWidgets import QApplication

from housemaker.glb import GeneratedModel, PreviewPlacedObject
from housemaker.surface_geometry import (
    SURFACE_TYPE_CEILING,
    SURFACE_TYPE_FLOOR,
    SURFACE_TYPE_WALL,
    FixedSurface,
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
        level_index=0,
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
        self.assertIsNotNone(self.viewer.highlight_ceiling_button)
        self.assertIsNotNone(self.viewer.hide_ceiling_button)
        floor_button = self.viewer.highlight_floor_button
        ceiling_button = self.viewer.highlight_ceiling_button
        hide_ceiling_button = self.viewer.hide_ceiling_button
        assert floor_button is not None
        assert ceiling_button is not None
        assert hide_ceiling_button is not None
        self.assertTrue(floor_button.isCheckable())
        self.assertTrue(ceiling_button.isCheckable())
        self.assertTrue(hide_ceiling_button.isCheckable())
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
        self.assertFalse(ceiling_button.isChecked())
        self.assertEqual(
            self.viewer.get_visible_canvas_surface_ids(),
            (floor_id,),
        )
        self.assertEqual(self.viewer.get_visible_placed_object_ids(), ())

        ceiling_button.click()

        self.assertEqual(
            self.viewer.get_canvas_surface_focus_type(),
            SURFACE_TYPE_CEILING,
        )
        self.assertFalse(floor_button.isChecked())
        self.assertTrue(ceiling_button.isChecked())
        self.assertEqual(
            self.viewer.get_visible_canvas_surface_ids(),
            (ceiling_id,),
        )
        self.assertEqual(self.viewer.get_visible_placed_object_ids(), ())

        hide_ceiling_button.click()

        self.assertTrue(self.viewer.get_canvas_ceiling_hidden())
        self.assertTrue(hide_ceiling_button.isChecked())
        self.assertEqual(
            self.viewer.get_visible_canvas_surface_ids(),
            (ceiling_id,),
            "Ceiling focus must visually override the independent hide state.",
        )

        ceiling_button.click()

        self.assertIsNone(self.viewer.get_canvas_surface_focus_type())
        self.assertFalse(ceiling_button.isChecked())
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


if __name__ == "__main__":
    unittest.main()
