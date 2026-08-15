# ### Environment setup ###
from __future__ import annotations

import os
import math
import unittest

import numpy as np
import trimesh

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from housemaker.camera_models import CameraPose
from housemaker.generation_state import MASK_MODE_PAINT, MaskPoint, MaskStroke
from housemaker.surface_geometry import (
    SURFACE_TYPE_CEILING,
    SURFACE_TYPE_FLOOR,
    SURFACE_TYPE_WALL,
    FixedSurface,
)
from housemaker.surface_texture_viewer import (
    SELECTED_SURFACE_EDGE_COLOR,
    SELECTED_SURFACE_OUTLINE_RADIUS_METERS,
    SurfaceTextureViewer,
    _build_camera_ray,
    _build_surface_texture_mesh_data,
    rasterize_texture_mask_strokes,
)


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_quad_surface(
    surface_id: str,
    surface_type: str,
    *,
    x_offset: float = 0.0,
) -> FixedSurface:
    if surface_type == SURFACE_TYPE_WALL:
        vertices = np.asarray(
            (
                (x_offset, 0.0, 0.0),
                (x_offset + 2.0, 0.0, 0.0),
                (x_offset + 2.0, 0.0, 3.0),
                (x_offset, 0.0, 3.0),
            ),
            dtype=float,
        )
        area = 6.0
    else:
        z = 0.0 if surface_type == SURFACE_TYPE_FLOOR else 3.0
        vertices = np.asarray(
            (
                (x_offset, 0.0, z),
                (x_offset + 2.0, 0.0, z),
                (x_offset + 2.0, 2.0, z),
                (x_offset, 2.0, z),
            ),
            dtype=float,
        )
        area = 4.0
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        process=False,
    )
    return FixedSurface(
        surface_id=surface_id,
        surface_type=surface_type,
        level_index=2,
        room_index=0,
        wall_key=surface_id if surface_type == SURFACE_TYPE_WALL else None,
        mesh=mesh,
        area_square_meters=area,
    )


# ### Tests ###
class SurfaceTextureViewerSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.viewer = SurfaceTextureViewer()
        self.wall_one = _build_quad_surface("wall-one", SURFACE_TYPE_WALL)
        self.wall_two = _build_quad_surface(
            "wall-two",
            SURFACE_TYPE_WALL,
            x_offset=3.0,
        )
        self.floor = _build_quad_surface("floor", SURFACE_TYPE_FLOOR)
        self.viewer.set_surfaces((self.wall_one, self.wall_two, self.floor))

    def tearDown(self) -> None:
        self.viewer.close()
        self.viewer.deleteLater()
        _qt_application.processEvents()

    def test_regular_click_replaces_and_shift_toggles_same_type(self) -> None:
        emitted: list[tuple[str, ...]] = []
        self.viewer.selection_changed.connect(emitted.append)

        self.assertTrue(self.viewer.select_surface("wall-one"))
        self.assertTrue(
            self.viewer.select_surface("wall-two", shift_pressed=True)
        )

        self.assertEqual(
            self.viewer.get_selected_surface_ids(),
            ("wall-one", "wall-two"),
        )
        self.assertEqual(
            self.viewer.get_selected_surface_type(),
            SURFACE_TYPE_WALL,
        )
        self.assertAlmostEqual(self.viewer.get_combined_selected_area(), 12.0)
        self.assertTrue(
            self.viewer.select_surface("wall-one", shift_pressed=True)
        )
        self.assertEqual(self.viewer.get_selected_surface_ids(), ("wall-two",))
        self.assertEqual(emitted[-1], ("wall-two",))

    def test_shift_click_of_another_surface_type_is_ignored(self) -> None:
        self.viewer.select_surface("wall-one")

        did_change = self.viewer.select_surface("floor", shift_pressed=True)

        self.assertFalse(did_change)
        self.assertEqual(self.viewer.get_selected_surface_ids(), ("wall-one",))

    def test_regular_click_of_another_type_replaces_selection(self) -> None:
        self.viewer.set_selected_surface_ids(("wall-one", "wall-two"))

        self.viewer.select_surface("floor")

        self.assertEqual(self.viewer.get_selected_surface_ids(), ("floor",))
        self.assertEqual(
            self.viewer.get_selected_surface_type(),
            SURFACE_TYPE_FLOOR,
        )

    def test_mixed_programmatic_selection_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "only one surface type"):
            self.viewer.set_selected_surface_ids(("wall-one", "floor"))

    def test_unknown_legacy_room_index_id_is_skipped_instead_of_retargeted(
        self,
    ) -> None:
        stable_surface = _build_quad_surface(
            "level:2/room:5/floor",
            SURFACE_TYPE_FLOOR,
        )
        self.viewer.set_surfaces((stable_surface,))

        self.viewer.set_selected_surface_ids(("level:2/room:0/floor",))
        self.viewer.set_surface_texture(
            ("level:2/room:0/floor",),
            np.full((2, 2, 4), 255, dtype=np.uint8),
        )

        self.assertEqual(self.viewer.get_selected_surface_ids(), ())
        self.assertEqual(self.viewer._surface_textures, {})

    def test_cpu_ray_pick_returns_nearest_surface_and_drives_selection(self) -> None:
        near_wall = _build_quad_surface("near-wall", SURFACE_TYPE_WALL)
        far_wall = _build_quad_surface("far-wall", SURFACE_TYPE_WALL)
        far_wall.mesh.apply_translation((0.0, 2.0, 0.0))
        self.viewer.set_surfaces((near_wall, far_wall))
        self.viewer.set_camera_pose(
            CameraPose(x=1.0, y=-2.0, z=1.5, yaw_degrees=90.0)
        )

        surface_id = self.viewer.pick_surface_from_ray(
            (1.0, -2.0, 1.5),
            (0.0, 1.0, 0.0),
        )
        self.viewer._handle_surface_pick_requested(
            QPointF(self.viewer.view.width() / 2.0, self.viewer.view.height() / 2.0),
            Qt.KeyboardModifier.NoModifier,
        )

        self.assertEqual(surface_id, "near-wall")
        self.assertEqual(
            self.viewer.get_selected_surface_ids(),
            ("near-wall",),
        )

    def test_selected_surface_uses_visible_solid_orange_outline(self) -> None:
        first_items = self.viewer._render_items_by_surface_id["wall-one"]
        second_items = self.viewer._render_items_by_surface_id["wall-two"]
        self.assertIsNotNone(first_items.outline_item)
        self.assertIsNotNone(second_items.outline_item)
        assert first_items.outline_item is not None
        assert second_items.outline_item is not None
        self.assertFalse(first_items.outline_item.visible())

        self.viewer.select_surface("wall-one")

        self.assertTrue(first_items.outline_item.visible())
        self.assertFalse(second_items.outline_item.visible())
        self.assertFalse(first_items.face_item.opts["drawEdges"])
        self.assertEqual(
            first_items.outline_item.opts["color"],
            SELECTED_SURFACE_EDGE_COLOR,
        )
        outline_vertices = np.asarray(
            first_items.outline_item.opts["meshdata"].vertexes(),
            dtype=float,
        )
        self.assertGreaterEqual(
            float(np.ptp(outline_vertices[:, 1])),
            SELECTED_SURFACE_OUTLINE_RADIUS_METERS * 1.8,
        )

        self.viewer.select_surface("wall-two")

        self.assertFalse(first_items.outline_item.visible())
        self.assertTrue(second_items.outline_item.visible())


class SurfaceTextureViewerCameraTests(unittest.TestCase):
    def setUp(self) -> None:
        self.viewer = SurfaceTextureViewer()

    def tearDown(self) -> None:
        self.viewer.exit_first_person_mode()
        self.viewer.close()
        self.viewer.deleteLater()
        _qt_application.processEvents()

    def test_zqsd_movement_uses_camera_yaw_without_gravity(self) -> None:
        self.viewer.set_camera_pose(CameraPose(x=1.0, y=2.0, z=1.7))
        view = self.viewer.view
        view.enter_first_person_mode()
        view.keyPressEvent(
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_Z,
                Qt.KeyboardModifier.NoModifier,
            )
        )

        view.step_movement(0.4)
        moved_pose = view.get_camera_pose()

        self.assertAlmostEqual(moved_pose.x, 2.0)
        self.assertAlmostEqual(moved_pose.y, 2.0)
        self.assertAlmostEqual(moved_pose.z, 1.7)
        view.keyReleaseEvent(
            QKeyEvent(
                QEvent.Type.KeyRelease,
                Qt.Key.Key_Z,
                Qt.KeyboardModifier.NoModifier,
            )
        )

    def test_q_moves_camera_left_and_d_moves_camera_right(self) -> None:
        view = self.viewer.view
        view.enter_first_person_mode()
        cases = (
            (0.0, Qt.Key.Key_Q, (0.0, 1.0)),
            (0.0, Qt.Key.Key_D, (0.0, -1.0)),
            (90.0, Qt.Key.Key_Q, (-1.0, 0.0)),
            (90.0, Qt.Key.Key_D, (1.0, 0.0)),
        )
        for yaw_degrees, key, expected_xy in cases:
            with self.subTest(yaw_degrees=yaw_degrees, key=key):
                self.viewer.set_camera_pose(
                    CameraPose(z=1.7, yaw_degrees=yaw_degrees)
                )
                view.keyPressEvent(
                    QKeyEvent(
                        QEvent.Type.KeyPress,
                        key,
                        Qt.KeyboardModifier.NoModifier,
                    )
                )
                view.step_movement(0.4)
                view.keyReleaseEvent(
                    QKeyEvent(
                        QEvent.Type.KeyRelease,
                        key,
                        Qt.KeyboardModifier.NoModifier,
                    )
                )
                moved_pose = view.get_camera_pose()
                self.assertAlmostEqual(moved_pose.x, expected_xy[0])
                self.assertAlmostEqual(moved_pose.y, expected_xy[1])
                self.assertAlmostEqual(moved_pose.z, 1.7)

    def test_camera_ray_uses_horizontal_fov_on_non_square_viewport(self) -> None:
        _origin, right_edge_ray = _build_camera_ray(
            CameraPose(fov_degrees=70.0),
            800.0,
            200.0,
            800,
            400,
        )
        _origin, top_edge_ray = _build_camera_ray(
            CameraPose(fov_degrees=70.0),
            400.0,
            0.0,
            800,
            400,
        )

        self.assertAlmostEqual(
            math.degrees(math.atan2(right_edge_ray[1], right_edge_ray[0])),
            35.0,
        )
        self.assertAlmostEqual(
            math.degrees(math.atan2(top_edge_ray[2], top_edge_ray[0])),
            math.degrees(math.atan(math.tan(math.radians(35.0)) * 0.5)),
        )

    def test_first_person_mode_remains_active_until_explicit_exit(self) -> None:
        self.viewer.enter_first_person_mode()
        self.assertTrue(self.viewer.view.is_first_person_active)

        self.viewer.exit_first_person_mode()

        self.assertFalse(self.viewer.view.is_first_person_active)


class SurfaceTextureMappingTests(unittest.TestCase):
    def test_planar_uvs_use_world_scale_instead_of_each_surface_bounds(self) -> None:
        first = _build_quad_surface("wall-one", SURFACE_TYPE_WALL)
        second = _build_quad_surface(
            "wall-two",
            SURFACE_TYPE_WALL,
            x_offset=2.0,
        )
        texture = np.full((4, 4, 4), 255, dtype=np.uint8)

        first_data = _build_surface_texture_mesh_data(first, texture, 2.0)
        second_data = _build_surface_texture_mesh_data(second, texture, 2.0)

        self.assertAlmostEqual(
            float(first_data.texture_coordinates[:, 0].min()),
            0.0,
        )
        self.assertAlmostEqual(
            float(first_data.texture_coordinates[:, 0].max()),
            1.0,
        )
        self.assertAlmostEqual(
            float(second_data.texture_coordinates[:, 0].min()),
            1.0,
        )
        self.assertAlmostEqual(
            float(second_data.texture_coordinates[:, 0].max()),
            2.0,
        )

    def test_texture_can_be_applied_to_many_selected_surfaces(self) -> None:
        viewer = SurfaceTextureViewer()
        try:
            surfaces = (
                _build_quad_surface("wall-one", SURFACE_TYPE_WALL),
                _build_quad_surface(
                    "wall-two",
                    SURFACE_TYPE_WALL,
                    x_offset=3.0,
                ),
            )
            viewer.set_surfaces(surfaces)

            viewer.set_surface_texture(
                ("wall-one", "wall-two"),
                np.full((8, 8, 4), (80, 120, 160, 255), dtype=np.uint8),
            )

            self.assertIsNotNone(
                viewer._render_items_by_surface_id["wall-one"].texture_item
            )
            self.assertIsNotNone(
                viewer._render_items_by_surface_id["wall-two"].texture_item
            )
        finally:
            viewer.close()
            viewer.deleteLater()
            _qt_application.processEvents()

    def test_repeating_texture_mask_wraps_edges_without_center_artifact(self) -> None:
        seam_stroke = MaskStroke(
            mode=MASK_MODE_PAINT,
            radius_normalized=0.04,
            points=(
                MaskPoint(x=0.98, y=0.5),
                MaskPoint(x=0.02, y=0.5),
            ),
        )

        mask = rasterize_texture_mask_strokes((100, 100), [seam_stroke])

        self.assertGreater(int(mask[:, :5].max()), 0)
        self.assertGreater(int(mask[:, -5:].max()), 0)
        self.assertEqual(int(mask[50, 50]), 0)

    def test_repeating_texture_mask_wraps_corner_brush(self) -> None:
        corner_stroke = MaskStroke(
            mode=MASK_MODE_PAINT,
            radius_normalized=0.08,
            points=(MaskPoint(x=0.0, y=0.0),),
        )

        mask = rasterize_texture_mask_strokes((100, 100), [corner_stroke])

        for row, column in ((0, 0), (0, 99), (99, 0), (99, 99)):
            self.assertEqual(int(mask[row, column]), 255)

    def test_partial_edit_data_targets_only_masked_selected_surfaces(self) -> None:
        viewer = SurfaceTextureViewer()
        try:
            surfaces = (
                _build_quad_surface("wall-one", SURFACE_TYPE_WALL),
                _build_quad_surface(
                    "wall-two",
                    SURFACE_TYPE_WALL,
                    x_offset=3.0,
                ),
            )
            viewer.set_surfaces(surfaces)
            texture = np.full((32, 32, 4), (20, 40, 80, 255), dtype=np.uint8)
            viewer.set_surface_texture(("wall-one", "wall-two"), texture)
            viewer.set_selected_surface_ids(("wall-one", "wall-two"))
            viewer.add_texture_mask_stroke(
                "wall-one",
                MaskStroke(
                    mode=MASK_MODE_PAINT,
                    radius_normalized=0.1,
                    points=(MaskPoint(x=0.5, y=0.5),),
                ),
            )

            result = viewer.get_selected_texture_edit_data()

            self.assertEqual(viewer.get_masked_selected_surface_ids(), ("wall-one",))
            self.assertIsNotNone(result)
            assert result is not None
            base_rgba, editable_mask = result
            np.testing.assert_array_equal(base_rgba, texture)
            self.assertGreater(int(editable_mask.max()), 0)
        finally:
            viewer.close()
            viewer.deleteLater()
            _qt_application.processEvents()


if __name__ == "__main__":
    unittest.main()
